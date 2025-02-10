# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import OrderedDict

import paddle
from paddle.distributed.auto_parallel.process_mesh import ProcessMesh
from paddle.distributed.auto_parallel.static.dist_attribute import (
    OperatorDistAttr,
    TensorDistAttr,
)
from paddle.distributed.auto_parallel.static.operators.common import (
    is_data_parallel_reduce_op,
    is_data_parallel_scale_op,
)
from paddle.distributed.auto_parallel.static.utils import (
    find_higher_order_backward_op,
    get_var_numel,
    insert_dependencies_for_vars,
    is_forward_op,
    is_loss_grad_op,
    is_optimize_op,
    ring_id_to_process_group,
)
from paddle.distributed.fleet.meta_optimizers.common import OP_ROLE_KEY, OpRole
from paddle.static import default_main_program
from paddle.utils import unique_name

from .pass_base import PassBase, PassType, register_pass

# add new optimizers supporting rescale_grad here
__rescale_grad_supported_opts__ = [
    'lars_momentum',
    'sparse_momentum',
    'dgc_momentum',
    'momentum',
    'merge_momentum',
]

# a heuristic number
__max_stream_num_allow__ = 16


@register_pass("auto_parallel_data_parallel_optimization")
class DataParallelOptimizationPass(PassBase):
    """
    Apply Optimizations that specialized for data parallelism in Auto Parallel.
    1. prune grad scaling
    2. overlap comm and calc
    3. fuse allreduce
    """

    def __init__(self):
        super().__init__()
        # NOTE not use dependence on loss and param_grads
        self.set_attr("dist_context", None)
        self.set_attr("global_rank", -1)
        self.set_attr("use_sharding", False)
        # {grad1: group1, grad2: group1, grad3: group2}
        # record the order for fuse grad data memory
        self._grad_name_to_group_map = OrderedDict()
        # {group1:[grad1, grad2] , group2:[grad3]}
        self._group_to_grad_name_map = OrderedDict()
        self._support_rescale_grad = False

    def _check_self(self):
        if self.get_attr("dist_context") is None:
            return False
        if (not isinstance(self.get_attr("global_rank"), int)) or self.get_attr(
            "global_rank"
        ) < 0:
            return False

        return True

    def _check_conflict(self, other_pass):
        return True

    def _type(self):
        return PassType.COMM_OPT

    def _apply_single_impl(self, main_program, startup_program, context):
        self.dist_context = self.get_attr("dist_context")
        self.global_rank = int(self.get_attr("global_rank"))
        self.use_sharding = self.get_attr("use_sharding")
        self.coalesce_prefix = 'coalesce_grad'
        self.gradient_sync_stream = "gradient_sync_stream"

        with paddle.static.program_guard(main_program, startup_program):
            self._analyze_program()

            # TODO refactor here to first fuse then overlap
            if self.is_data_parallel_applied():
                self._prune_grad_scaling()
                self._calc_comm_overlap()
                grad_group = self._fuse_allreduce()
                self._add_dependencies(grad_group)
                self.summary(grad_group)

    def _prune_grad_scaling(self):
        if not self._could_be_prune():
            return

        if self._all_dp_groups_same_degree():
            self._scale_backward_initial_grad()
        else:
            self._update_opt_rescale_grad()

        self._remove_grad_scaling()

    def _calc_comm_overlap(self):
        if not self._could_be_overlap():
            return
        self._comms_overlap_calc()
        self._calc_wait_comms()

    def _fuse_allreduce(self):
        if not self._could_be_fuse():
            return []

        grad_group = self._group_grads()
        self._update_program(grad_group)

        return grad_group

    def _analyze_program(self):
        """
        build two maps
        {param_grad_name: data_parallel_group}
        {pdata_parallel_group: aram_grad_name}
        """

        block = default_main_program().global_block()
        ops = block.ops
        scaled_grads = []

        for op in ops:
            if is_data_parallel_reduce_op(op):
                grad_name = op.output_arg_names[0]
                if grad_name in self._grad_name_to_group_map:
                    continue
                assert op.has_attr(
                    "ring_id"
                ), f"Unexpected: comm op [{op}] has NOT ring id."
                group = ring_id_to_process_group(op.attr("ring_id"))

                assert (
                    group is not None
                ), f"Unexpected: data parallel group of [{grad_name}] from op [{op}] is None"

                self._grad_name_to_group_map[grad_name] = group

                if group not in self._group_to_grad_name_map:
                    self._group_to_grad_name_map[group] = [grad_name]
                else:
                    self._group_to_grad_name_map[group].append(grad_name)

            elif is_data_parallel_scale_op(op):
                grad_name = op.output_arg_names[0]
                scaled_grads.append(grad_name)

            # TODO support multiple optimizers in on network in future.
            # here we assume that the optimizer is unique in network.
            elif (
                is_optimize_op(op)
                and op.type in __rescale_grad_supported_opts__
            ):
                self._support_rescale_grad = True

        not_synchronized_grads = []
        for grad_name in scaled_grads:
            if grad_name not in self._grad_name_to_group_map:
                not_synchronized_grads.append(grad_name)
        assert (
            len(not_synchronized_grads) == 0
        ), f"Unexpected: gradients [{not_synchronized_grads}] is scaled BUT NOT synchronized."

    def is_data_parallel_applied(self):
        return len(self._group_to_grad_name_map) > 0

    def _could_be_prune(self):
        return self.dist_context.gradient_scale and (
            self._support_rescale_grad or self._all_dp_groups_same_degree()
        )

    def _all_dp_groups_same_degree(self):
        return (
            len(
                {
                    len(group.ranks)
                    for group in self._group_to_grad_name_map.keys()
                }
            )
            == 1
        )

    def _scale_backward_initial_grad(self):
        block = default_main_program().global_block()
        dp_degree = len(next(iter(self._group_to_grad_name_map.keys())).ranks)

        for idx, op in reversed(list(enumerate(block.ops))):
            if is_loss_grad_op(op):
                assert op.type == 'fill_constant', (
                    "loss_grad_op must be fill_constant op, "
                    f"but this op is {op.type}"
                )
                assert op.has_attr('value')
                loss_scale = float(op.attr('value'))
                loss_scale = loss_scale / dp_degree
                op._set_attr('value', loss_scale)
                break

    def _remove_grad_scaling(self):
        block = default_main_program().global_block()

        for op_idx, op in reversed(list(enumerate(block.ops))):
            if is_data_parallel_scale_op(op):
                block._remove_op(op_idx, False)

        block._sync_with_cpp()

    def _update_opt_rescale_grad(self):
        block = default_main_program().global_block()
        scaled_grads = set()

        for idx, op in reversed(list(enumerate(block.ops))):
            if (
                is_optimize_op(op)
                and op.type in __rescale_grad_supported_opts__
            ):
                assert op.has_attr(
                    'rescale_grad'
                ), f"Unexpected: op [{op}] is supported to have [rescale_grad] attribute."
                assert (
                    len(op.input("Grad")) == 1
                ), f"Unexpected: op [{op}] is supported to have only one input grad var."

                grad_name = op.input("Grad")[0]
                dp_degree = len(
                    list(self._grad_name_to_group_map[grad_name].ranks)
                )
                scaled_grads.add(grad_name)

                rescale_grad = float(op.attr('rescale_grad')) / dp_degree
                op._set_attr('rescale_grad', rescale_grad)

        assert scaled_grads == set(
            self._grad_name_to_group_map.keys()
        ), f"Unexpected: gradients [{set(self._grad_name_to_group_map.keys()) - scaled_grads}] are unscaled."

    def _could_be_overlap(self):
        # NOTE current different nccl comm will use different cuda stream
        # so if there too many dp group there will be too many stream need to be
        # created and sync.
        # revise here when framework support custom stream in static graph mode.
        num_dp_comm_stream = len(set(self._group_to_grad_name_map.keys()))
        if num_dp_comm_stream > __max_stream_num_allow__:
            return False
        if self.use_sharding:
            return False
        return True

    def _comms_overlap_calc(self):
        # TODO support InterpreterCore executor for overlap.
        # InterpreterCore has a different logic for overlapping
        # which is different from use_calc_stream
        block = default_main_program().global_block()

        # comm wait calc to finish
        for idx, op in reversed(list(enumerate(block.ops))):
            if is_data_parallel_reduce_op(op):
                assert op.has_attr('ring_id')

                op._set_attr('use_calc_stream', False)
                ring_id = op.attr("ring_id")
                block._insert_op_without_sync(
                    idx,
                    type='c_wait_compute',
                    inputs={'X': []},
                    outputs={'Out': []},
                    attrs={'op_role': OpRole.Backward, 'ring_id': ring_id},
                )

        block._sync_with_cpp()

    def _calc_wait_comms(self):
        return

        block = default_main_program().global_block()

        # NOTE the naive overlap implement in static hybrid parallel only sync comm stream
        # at the end of Backward phase, based on a strong constraint that
        # all communicating gradient would NOT be used after communication in Backward phase.
        # BUT this constraint will fail for scenario like Weight-Sharing and Higher-Order Differentiation,
        # where gradient will be involved in other calculation between data-parallel allreduce kernel submitted
        # into comm streams and the synchronization of comm stream at the end of Backward phase.
        # synchronization of  comm stream should add according to the usage of communicating gradients
        # to support Overlapping for Weight-Sharing and Higher-Order Differentiation.

        ring_id_to_un_sync_grad_map = {}
        op_idx_to_sync_ring_id_map = {}
        for group in self._group_to_grad_name_map.keys():
            ring_id_to_un_sync_grad_map[group.id] = []

        # analyze the where need to sync
        for i, op in enumerate(block.ops):
            if is_data_parallel_reduce_op(op):
                ring_id = op.attr("ring_id")
                grad_name = op.output_arg_names[0]
                ring_id_to_un_sync_grad_map[ring_id].append(grad_name)
            elif is_data_parallel_scale_op(op):
                continue
            # other ops that might use communicating grad
            else:
                for input_var_name in op.input_arg_names:
                    for (
                        ring_id,
                        unsync_grad_names,
                    ) in ring_id_to_un_sync_grad_map.items():
                        if input_var_name in unsync_grad_names:
                            # need to sync before op_i
                            if i in op_idx_to_sync_ring_id_map:
                                op_idx_to_sync_ring_id_map[i].append(ring_id)
                            else:
                                op_idx_to_sync_ring_id_map[i] = [ring_id]
                            # all grads in this comm stream are synced
                            ring_id_to_un_sync_grad_map[ring_id] = []

        # insert synchronization
        indices = list(op_idx_to_sync_ring_id_map.keys())
        # TODO the synchronization could be optimized
        # we should record the event of a gradient is communicating and
        # only wait for that event to be completed.
        # BUT paddle static currently not support op api for event record only, so
        # here we try to wait for all kernel in that comm stream to be finish which is not that optimized.
        for i in sorted(indices, reverse=True):
            for ring_id in op_idx_to_sync_ring_id_map[i]:
                block._insert_op_without_sync(
                    i,
                    type='c_wait_comm',
                    inputs={'X': []},
                    outputs={'Out': []},
                    attrs={'op_role': OpRole.Backward, 'ring_id': ring_id},
                )
        block._sync_with_cpp()

    def _could_be_fuse(self):
        # TODO  support gradient fuse higher order gradient.
        # should analyse the dependencies of gradient in backward.
        if find_higher_order_backward_op(default_main_program()):
            return False
        if self.use_sharding:
            return False
        return True

    def _group_grads(self):
        """
        conditions for gradients to be grouped:
        1. group size < max_fuse_numel
        2. same dp group
        3. same dtype
        4. dependency: grad would NOT be used by other ops within group segment

        gradients inside same group would be fuse into one coalesce tensor
        """

        block = default_main_program().global_block()
        ops = block.ops

        # group individual grad vars
        # TODO consider fuse gradient for sharding reduce
        # TODO let user to set fuse_grad_size
        # emb = 50000 * h, ffn = 8 * h * h, mha = 4 * h * h
        h = 2048
        ffn_numel = 2 * (4 * h) * h
        mha_numel = 3 * h * h + h * h
        max_fuse_numel = ffn_numel + mha_numel
        grad_groups = []
        cur_group = GradientsGroup(ops, max_fuse_numel)
        grouped_grad_names = set()

        def collect_group(cur_group, grad_var, ring_id, i):
            if len(cur_group.gradients) == 0:
                cur_group = None
            else:
                cur_group.finalize()
                grad_groups.append(cur_group)

            new_group = GradientsGroup(ops, max_fuse_numel)
            if grad_var:
                new_group.add(grad_var, ring_id, i)
                grouped_grad_names.add(grad_var.name)
            return new_group

        def op_depend_on_group(op, group):
            vars_ = set(op.input_arg_names + op.output_arg_names)
            grad_names = {grad.name for grad in group.gradients}
            return len(vars_.intersection(grad_names)) > 0

        for i, op in enumerate(ops):
            if is_data_parallel_reduce_op(op):
                ring_id = op.attr("ring_id")
                grad_name = op.output_arg_names[0]
                grad_var = block.var(grad_name)
                grad_numel = get_var_numel(grad_var)

                if cur_group.acceptable(grad_var, ring_id):
                    assert grad_name not in grouped_grad_names
                    grouped_grad_names.add(grad_name)
                    cur_group.add(grad_var, ring_id, i)
                else:
                    cur_group = collect_group(cur_group, grad_var, ring_id, i)
            else:
                if op_depend_on_group(op, cur_group):
                    cur_group = collect_group(cur_group, None, None, None)

        # collect last group
        collect_group(cur_group, None, None, None)

        return grad_groups

    def _update_program(self, grad_groups):
        block = default_main_program().global_block()

        remove_op_types = [
            'scale',
            'c_allreduce_avg',
            'c_allreduce_sum',
            'c_wait_compute',
        ]

        for i, group in enumerate(grad_groups[::-1]):
            # skip unfused big tensor
            if len(group.gradients) <= 1:
                group.coalesce_var = group.gradients[0]
                continue

            ref_process_mesh = set()
            concated_shapes = []
            concated_ranks = []
            for grad_ in group.gradients:
                grad_dist_attr = (
                    self.dist_context.get_tensor_dist_attr_for_program(grad_)
                )
                ref_process_mesh.update(
                    set(grad_dist_attr.process_mesh.process_ids)
                )

                shape = grad_.shape
                concated_shapes.extend(shape)
                concated_ranks.append(len(shape))

            # create coalesce tensor
            group.coalesce_var = block.create_var(
                name=unique_name.generate(self.coalesce_prefix + f'_{i}'),
                dtype=group.dtype,
                persistable=False,
                stop_gradient=True,
            )

            tensor_dist_attr = TensorDistAttr()
            tensor_dist_attr.process_mesh = ProcessMesh(list(ref_process_mesh))
            tensor_dist_attr.dims_mapping = []
            self.dist_context.set_tensor_dist_attr_for_program(
                group.coalesce_var, tensor_dist_attr
            )

            # update allreduce & scale op
            if group.scale_op_idx != -1:
                scale_op = block.ops[group.scale_op_idx]
                assert (
                    scale_op.type == 'scale'
                ), f"should found scale op but found {scale_op}"
                scale_op._rename_input(
                    scale_op.input_arg_names[0], group.coalesce_var.name
                )
                scale_op._rename_output(
                    scale_op.output_arg_names[0], group.coalesce_var.name
                )

            allreduce_op = block.ops[group.allreduce_op_idx]
            assert allreduce_op.type in [
                'all_reduce',
            ], f"should found all_reduce op but found {allreduce_op}"
            allreduce_op_dist_attr = (
                self.dist_context.get_op_dist_attr_for_program(allreduce_op)
            )
            old_in_name = allreduce_op.input_arg_names[0]
            new_in_name = group.coalesce_var.name
            allreduce_op._rename_input(old_in_name, new_in_name)
            input_dist_attr = allreduce_op_dist_attr.get_input_dist_attr(
                old_in_name
            )
            allreduce_op_dist_attr.set_input_dist_attr(
                new_in_name, input_dist_attr
            )

            old_out_name = allreduce_op.output_arg_names[0]
            new_out_name = group.coalesce_var.name
            allreduce_op._rename_output(old_out_name, new_out_name)
            out_dist_attr = allreduce_op_dist_attr.get_output_dist_attr(
                old_out_name
            )
            allreduce_op_dist_attr.set_output_dist_attr(
                new_out_name, out_dist_attr
            )

            # remove un-used op
            remove_op_indices = (
                group.remove_wait_op_indices
                + group.remove_allreduce_op_indices
                + group.remove_scale_op_indices
            )
            for idx in sorted(remove_op_indices, reverse=True):
                assert (
                    block.ops[idx].type in remove_op_types
                ), f"Unexpected: try to remove op {block.ops[idx]}"
                block._remove_op(idx, False)

            # insert coalesce op
            grad_names = [grad.name for grad in group.gradients]
            coalesce_op = block._insert_op_without_sync(
                group.coalesce_op_idx,
                type="coalesce_tensor",
                inputs={"Input": grad_names},
                outputs={
                    "Output": grad_names,
                    "FusedOutput": group.coalesce_var,
                },
                attrs={
                    "copy_data": False,
                    "use_align": True,
                    "dtype": group.dtype,
                    "concated_shapes": concated_shapes,
                    "concated_ranks": concated_ranks,
                    OP_ROLE_KEY: OpRole.Backward,
                },
            )

            op_dist_attr = OperatorDistAttr()
            op_dist_attr.impl_idx = 0
            op_dist_attr.impl_type = "default"
            op_dist_attr.process_mesh = ProcessMesh(list(ref_process_mesh))
            for in_name in coalesce_op.input_arg_names:
                in_var = block.var(in_name)
                in_var_dist_attr = (
                    self.dist_context.get_tensor_dist_attr_for_program(in_var)
                )
                op_dist_attr.set_input_dims_mapping(
                    in_name, in_var_dist_attr.dims_mapping
                )
            for out_name in coalesce_op.output_arg_names:
                out_var = block.var(out_name)
                out_var_dist_attr = (
                    self.dist_context.get_tensor_dist_attr_for_program(out_var)
                )
                op_dist_attr.set_output_dims_mapping(
                    out_name, out_var_dist_attr.dims_mapping
                )

            self.dist_context.set_op_dist_attr_for_program(
                coalesce_op, op_dist_attr
            )

        block._sync_with_cpp()

    def _add_dependencies(self, grad_groups):
        # NOTE Currently, auto_parallel need to adopt for two executors: Sequential executor (old exe) and Graph based
        # multiple stream executor(standalone exe). This function just for standalone exe. Refactor here
        # in future when only one executor stay.

        if len(grad_groups) == 0:
            return
        block = default_main_program().global_block()

        # Build maps
        coalesce_to_vars_map = {}
        for group in grad_groups:
            coalesce_to_vars_map[group.coalesce_var.name] = group

        # analyze dependencies
        dep_map = {}
        for idx, op in reversed(list(enumerate(block.ops))):
            if is_forward_op(op):
                break
            if is_optimize_op(op):
                continue

            if is_data_parallel_reduce_op(op):
                coalesce_var_name = op.output_arg_names[0]
                if self.coalesce_prefix in coalesce_var_name:
                    group = coalesce_to_vars_map[coalesce_var_name]
                    dep_map[idx] = [
                        (
                            idx,
                            group.gradients[-1],
                            group.coalesce_var,
                            op.attr(OP_ROLE_KEY),
                        )
                    ]
                    dep_map[idx].append(
                        (
                            idx + 1,
                            group.coalesce_var,
                            group.gradients,
                            op.attr(OP_ROLE_KEY),
                        )
                    )

        # insert dependency op
        indice = sorted(dep_map.keys(), reverse=True)
        for i in indice:
            for idx, prior_vars, post_vars, op_role in dep_map[i][::-1]:
                depend_op = insert_dependencies_for_vars(
                    block,
                    idx,
                    prior_vars,
                    post_vars,
                    self.dist_context,
                    op_role,
                    is_recompute=False,
                    sync=False,
                    op_namescope="data_parallel_overlap_dep",
                )
                depend_op.dist_attr.execution_stream = self.gradient_sync_stream
        block._sync_with_cpp()

        # remove naive synchronization & assign allreduce stream
        def remove_cond(op):
            if op.type != "c_wait_compute":
                return False
            if len(op.input_arg_names) != 0:
                return False
            if len(op.output_arg_names) != 0:
                return False
            return True

        for idx, op in reversed(list(enumerate(block.ops))):
            if is_data_parallel_reduce_op(op):
                op._set_attr('use_calc_stream', True)
                op.dist_attr.execution_stream = self.gradient_sync_stream

            if remove_cond(op):
                block._remove_op(idx, sync=False)

        block._sync_with_cpp()

    def summary(self, grad_groups=[]):
        # TODO: add logger module
        import logging

        self._logger = logging.getLogger()
        self._logger.propagate = False
        if not self._logger.handlers:
            self._logger.setLevel(logging.INFO)
            log_handler = logging.StreamHandler()
            log_format = logging.Formatter(
                '[%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s'
            )
            log_handler.setFormatter(log_format)
            self._logger.addHandler(log_handler)

        if len(grad_groups) > 0:
            self._logger.info("Data Parallel Optimization: ")
            self._logger.info(
                f" {len(self._grad_name_to_group_map.keys())} Allreduce ops are fused into {len(grad_groups)} coalesce allreduce ops."
            )
            self._logger.debug("gradient fusing group are following: ")
            fused_grads = set()
            for i, group in enumerate(grad_groups):
                self._logger.debug(
                    f"coalesce gradient [{i}] is composed by: {[grad.name for grad in group.gradients]}"
                )
                fused_grads.update([grad.name for grad in group.gradients])
            individual_grads = set(self._grad_name_to_group_map.keys()) - set(
                fused_grads
            )
            self._logger.debug(
                f"the following [{len(individual_grads)}] gradients are not fused: "
            )
            self._logger.debug(f"individual gradient {individual_grads}")


class GradientsGroup:
    def __init__(self, ops, max_group_size):
        self.max_group_size = max_group_size
        self.ops = ops

        self.gradients = []
        self.numel = 0
        self.dtype = None
        self.ring_id = None
        self.coalesce_var = None
        self.coalesce_op_idx = -1
        self.allreduce_op_idx = -1
        self.scale_op_idx = -1
        self.remove_wait_op_indices = []
        self.remove_allreduce_op_indices = []
        self.remove_scale_op_indices = []

    def acceptable(self, grad_var, ring_id):
        if len(self.gradients) == 0:
            return True
        if ring_id != self.ring_id:
            return False
        if get_var_numel(grad_var) + self.numel > self.max_group_size:
            return False
        if grad_var.dtype != self.dtype:
            return False

        return True

    def add(self, grad_var, ring_id, i):
        self.gradients.append(grad_var)
        self.ring_id = ring_id
        self.dtype = grad_var.dtype
        self.numel += get_var_numel(grad_var)

        # remove auxiliary ops in non-fuse dp allreduce
        self.remove_allreduce_op_indices.append(i)

        # NOTE this pass rely on the original synchronization add in previous passes
        # (same stream or calc_wait_comm & comm_wait_calc)
        # to guarantee the correctness of comm_calc execution order.
        # so the calc_wait_comm should be keep.
        grad_op_idx = i - 1
        if i > 0 and self.ops[i - 1].type == 'c_wait_compute':
            self.remove_wait_op_indices.append(i - 1)
            grad_op_idx -= 1
        if i + 1 < len(self.ops) and is_data_parallel_scale_op(self.ops[i - 1]):
            self.remove_scale_op_indices.append(i + 1)

        if len(self.gradients) == 1:
            # TODO Remove this is a temporary hack for Tensor Parallel. the logic
            # for find grad_op should be more general.
            if self.ops[grad_op_idx].type == "c_allreduce_sum":
                grad_op_idx -= 1

            grad_op = self.ops[grad_op_idx]
            assert (
                grad_var.name in grad_op.output_arg_names
            ), f"grad [{grad_var.name}] should be output of {grad_op}"
            self.coalesce_op_idx = grad_op_idx

    def finalize(self):
        self.allreduce_op_idx = self.remove_allreduce_op_indices.pop()
        if len(self.remove_wait_op_indices) > 1:
            self.remove_wait_op_indices.pop()
        if len(self.remove_scale_op_indices) > 1:
            self.scale_op_idx = self.remove_scale_op_indices.pop()
