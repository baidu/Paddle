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

from functools import reduce

import numpy as np

import paddle
import paddle.distributed as dist
from paddle.distributed.fleet.meta_optimizers.common import OP_ROLE_KEY, OpRole

from ..auto_parallel.process_mesh import ProcessMesh
from ..auto_parallel.static.dist_attribute import (
    OperatorDistAttr,
    TensorDistAttr,
)
from ..auto_parallel.static.operators.common import (
    SyncMode,
    is_data_parallel_reduce_op,
)
from ..auto_parallel.static.process_group import (
    get_all_process_groups,
    get_world_process_group,
)
from ..auto_parallel.static.reshard import Resharder
from ..auto_parallel.static.utils import (
    _get_comm_group,
    insert_dependencies_for_vars,
    is_gradient_clip_op,
    is_optimize_op,
    is_reshard_op,
)
from .auto_parallel_sharding import ShardingPass
from .pass_base import PassBase, register_pass


def _get_params_grads(block):
    params_grads = []
    for op in reversed(block.ops):
        if not is_optimize_op(op):
            break
        if "Param" in op.input_names and "Grad" in op.input_names:
            param_name = op.input("Param")[0]
            grad_name = op.input("Grad")[0]
            param = block.var(param_name)
            grad = block.var(grad_name)
            params_grads.append((param, grad))
    return params_grads


def _get_dpmp_topology(origin_topology, sharding_group):
    """
    Get dpmp topology from origin_topology

    Example:
        the parallel strategy: dp4-mp2-sharding2
        the complete process_mesh:
            topology: [4, 2]
            processes: [0, 1, 2, 3, 4, 5, 6, 7]
        the dpmp topology: [2, 2]
        the sharding axis: 1
    """
    sharding_axis = 1
    dp_sharding_topology = [
        origin_topology[0] // sharding_group.nranks,
        sharding_group.nranks,
    ]
    if dp_sharding_topology[0] == 1:
        sharding_axis = 0
        dp_sharding_topology = dp_sharding_topology[1:]

    product_dp_sharding = reduce(lambda x, y: x * y, dp_sharding_topology, 1)
    product_topology = reduce(lambda x, y: x * y, origin_topology, 1)

    if product_topology == product_dp_sharding:
        dpmp_topology = dp_sharding_topology
    else:
        assert product_topology % product_dp_sharding == 0
        mp_degree = product_topology // product_dp_sharding
        dpmp_topology = [*dp_sharding_topology, mp_degree]

    return dpmp_topology, sharding_axis


def _get_dpmp_process_mesh(rank_id, topology, processes, sharding_group):
    """
    Get dpmp process_mesh from the complete process_mesh which apply sharding.

    Example:
        the parallel strategy: dp4-mp2-sharding2
        the complete process_mesh:
            topology: [4, 2]
            processes: [0, 1, 2, 3, 4, 5, 6, 7]
        the dpmp process_mesh is:
            1) topology: [2, 2], processes: [0, 1, 4, 5]
            2) topology: [2, 2], processes: [2, 3, 6, 7]
    """
    if sharding_group is None:
        return topology, processes

    # get dpmp_topology
    dpmp_topology, sharding_axis = _get_dpmp_topology(topology, sharding_group)

    # get all sharding_groups of ranks
    sharding_groups = []
    for rank in processes:
        group = _get_comm_group(processes, dpmp_topology, sharding_axis, rank)
        if group not in sharding_groups:
            sharding_groups.append(group)

    # get dpmp_processes
    sharding_groups = np.array(sharding_groups)
    dpmp_processes_in_sharding = None
    for i in range(sharding_groups.shape[-1]):
        if rank_id in sharding_groups[:, i]:
            dpmp_processes_in_sharding = sharding_groups[:, i]

    assert dpmp_processes_in_sharding is not None
    return dpmp_topology, list(dpmp_processes_in_sharding)


def _is_about_global_norm(
    rank_id, tensor_shape, topology, processes, dims_mapping, sharding_group
):
    # get current process_mesh where the parameter exist.
    dpmp_topology, dpmp_processes = _get_dpmp_process_mesh(
        rank_id, topology, processes, sharding_group
    )

    complete_shape = Resharder.compute_complete_shape(
        tensor_shape, dpmp_topology, dims_mapping
    )

    complete_partitions = []
    complete_param_ranks = []
    for process in dpmp_processes:
        partition_index = Resharder.compute_partition_index(
            process, complete_shape, dims_mapping, dpmp_topology, dpmp_processes
        )
        if partition_index not in complete_partitions:
            complete_partitions.append(partition_index)
            complete_param_ranks.append(process)

    return rank_id in complete_param_ranks


class ClipHelper:
    def __init__(
        self, params_grads, rank_id, block, dist_context, pass_context
    ):
        params, _ = zip(*params_grads)
        self.params = list(params)
        self.params_name = [p.name for p in self.params]
        self.rank_id = rank_id
        self.block = block
        self.dist_context = dist_context
        self.pass_context = pass_context
        self.sharding_group = None
        self.world_ranks = get_world_process_group().ranks
        if hasattr(dist_context, '_sharding_group'):
            self.sharding_group = dist_context._sharding_group

        self.world_nranks = len(self.world_ranks)
        self.pure_data_parallel = self._is_pure_data_parallel()
        self.rank_to_params = self._partition_parameters(params)

    def is_calculate_norm(self, name):
        """
        whether the param_name@GRAD participate in the calculation of global_norm
        """
        if not self.is_local_param(name):
            return False

        param = self.params[self.params_name.index(name)]
        if not self.pure_data_parallel:
            dist_attr = self._get_dist_attr(name)
            topology = dist_attr.process_mesh.shape
            processes = dist_attr.process_mesh.process_ids
            dims_mapping = dist_attr.dims_mapping
            return _is_about_global_norm(
                self.rank_id,
                param.shape,
                topology,
                processes,
                dims_mapping,
                self.sharding_group,
            )
        else:
            return param.name in self.rank_to_params[self.rank_id]

    def is_local_param(self, name):
        """
        whether the param_name is updated with opt in cur_rank
        """
        if name not in self.params_name:
            return False
        return True

    def _get_dist_attr(self, name):
        var = self.block.vars[name]
        return self.dist_context.get_tensor_dist_attr_for_program(var)

    def is_local_var_with_dist_attr(self, name):
        """
        whether the var_name is belong to cur_rank
        """
        dist_attr = self._get_dist_attr(name)
        assert dist_attr is not None
        return self.rank_id in dist_attr.process_mesh.process_ids

    def _init_dist_attr(self, op):
        op_dist_attr = OperatorDistAttr()
        op_dist_attr.process_mesh = ProcessMesh(self.world_ranks)
        for in_name in op.input_arg_names:
            in_var = self.block.vars[in_name]
            in_dist_attr = TensorDistAttr()
            in_dist_attr.process_mesh = ProcessMesh(self.world_ranks)
            in_dist_attr.dims_mapping = [-1 for i in in_var.shape]
            self.dist_context.set_tensor_dist_attr_for_program(
                in_var, in_dist_attr
            )
            op_dist_attr.set_input_dist_attr(in_name, in_dist_attr)
        for out_name in op.output_arg_names:
            out_var = self.block.vars[out_name]
            out_dist_attr = TensorDistAttr()
            out_dist_attr.process_mesh = ProcessMesh(self.world_ranks)
            out_dist_attr.dims_mapping = [-1 for i in out_var.shape]
            self.dist_context.set_tensor_dist_attr_for_program(
                out_var, out_dist_attr
            )
            op_dist_attr.set_output_dist_attr(out_name, out_dist_attr)
        self.dist_context.set_op_dist_attr_for_program(op, op_dist_attr)

    def _is_pure_data_parallel(self):
        for applied_pass in self.pass_context.passes:
            if isinstance(applied_pass, ShardingPass):
                return False

        groups = get_all_process_groups()
        for g in groups:
            if g.nranks != self.world_nranks:
                return False

        for op in self.block.ops:
            if (
                (
                    op.type == "reduce"
                    and op.desc.attr("reduce_type") == dist.ReduceOp.SUM
                )
                or (
                    op.type == "all_reduce"
                    and op.desc.attr("reduce_type") == dist.ReduceOp.SUM
                )
                and not is_data_parallel_reduce_op(op)
            ):
                return False
            if op.type in ["send_v2", "recv_v2"]:
                return False

        return True

    def _partition_parameters(self, params):
        """
        build rank_id_to_params by the param's numel
        to guarantee params in every rank of dp_group as even as possible.
        """
        mapping = {}
        if not self.pure_data_parallel:
            for rank_ in range(self.world_nranks):
                mapping[rank_] = [p.name for p in params]
        else:
            for rank_ in range(self.world_nranks):
                mapping[rank_] = []
            sizes = [0] * self.world_nranks
            for param in params:
                rank = sizes.index(min(sizes))
                mapping[rank].append(param.name)
                numel = reduce(lambda x, y: x * y, param.shape, 1)
                assert (
                    numel > 0
                ), f"param [{param.name}] should larger than 0, but it is [{numel}]"
                sizes[rank] += numel
        return mapping


@register_pass("auto_parallel_grad_clip")
class ClipGradByGlobalNormPass(PassBase):
    """
    1. Remove norm-compute op and grad-scale op when the grad is not in current rank
       or is independent of the calculation of norm.
    2. Each rank computes its own norm value, then gets global_norm by allreduce_sum only once.
    """

    def __init__(self):
        super().__init__()
        self.set_attr("rank_id", None)
        self.set_attr("dist_context", None)
        self.set_attr("params_grads", None)

    def _check_self(self):
        if self.get_attr("dist_context") is None:
            return False
        dist_context = self.get_attr("dist_context")
        if dist_context._serial_optimizer._grad_clip is None:
            return False
        if self.get_attr("params_grads") is None:
            return False
        return True

    def _check_conflict(self, other_pass):
        return True

    def _apply_single_impl(self, main_program, startup_program, context):
        dist_context = self.get_attr("dist_context", None)
        rank_id = self.get_attr("rank_id", None)
        block = main_program.global_block()
        dist_params_grads = self.get_attr("params_grads", None)
        # dist_params_grads = _get_params_grads(block)

        self.clip_helper = ClipHelper(
            dist_params_grads, rank_id, block, dist_context, context
        )
        self._remove_no_need_ops_vars(block)

    def _remove_no_need_ops_vars(self, block):
        removed_op_out_type = [
            'squared_l2_norm',
            'square',
            'reduce_sum',
        ]

        removed_op_idx = set()
        removed_tmp_var = set()
        for idx, op in enumerate(block.ops):
            if not is_gradient_clip_op(op):
                continue

            if op.type == 'clip_by_norm':
                # remove 'clip_by_norm' op if the param is not updated with opt in current rank
                input_name = op.input("X")[0]
                if input_name.find("@GRAD") != -1:
                    param_name = input_name[: input_name.find("@GRAD")]
                    is_local = self.clip_helper.is_local_param(param_name)
                    if not is_local:
                        removed_op_idx.add(idx)
                        removed_tmp_var.update(set(op.output_arg_names))

            elif op.type in removed_op_out_type:
                input_name = op.input("X")[0]
                if input_name.find("@GRAD") != -1:
                    # remove 'squared_l2_norm' and 'square' ops,
                    # if the param@GRAD in cur_rank does not participate in the calculation of global_norm
                    param_name = input_name[: input_name.find("@GRAD")]
                    is_local = self.clip_helper.is_local_param(param_name)
                    is_calculate = self.clip_helper.is_calculate_norm(
                        param_name
                    )
                    if not is_local or not is_calculate:
                        removed_op_idx.add(idx)
                        removed_tmp_var.update(set(op.output_arg_names))
                else:
                    # 'reduce_sum' must be behind 'square'
                    if idx - 1 in removed_op_idx:
                        removed_op_idx.add(idx)
                        removed_tmp_var.update(set(op.output_arg_names))

            elif op.type == 'elementwise_mul':
                # 'elementwise_mul' scale the param@GRAD with global_norm
                # remove 'elementwise_mul' op if the param is not updated with opt in current rank
                input_name = op.input("X")[0]
                if input_name.find("@GRAD") != -1:
                    param_name = input_name[: input_name.find("@GRAD")]
                    is_local = self.clip_helper.is_local_param(param_name)
                    if not is_local:
                        removed_op_idx.add(idx)
                        if block.ops[idx - 1].type == 'cast':
                            removed_op_idx.add(idx - 1)
                            removed_tmp_var.update(
                                set(block.ops[idx - 1].output_arg_names)
                            )

            elif op.type == 'sum':
                # 'sum' op is used to calculate global_norm, and need to filter inputs which is not in cur_rank
                reserved_vars = []
                for input_name in op.input_arg_names:
                    if (
                        input_name not in removed_tmp_var
                        and self.clip_helper.is_local_var_with_dist_attr(
                            input_name
                        )
                    ):
                        reserved_vars.append(input_name)
                if not reserved_vars:
                    removed_op_idx.add(idx)
                    removed_tmp_var.update(set(op.output_arg_names))
                    if block.ops[idx + 1].type == 'cast':
                        removed_op_idx.add(idx + 1)
                        removed_tmp_var.update(
                            set(block.ops[idx + 1].output_arg_names)
                        )
                else:
                    op.desc.set_input("X", reserved_vars)

            elif op.type == 'stack':
                # 'stack' op is also used to calculate global_norm ('stack' + 'reduce_sum'), and need to filter inputs which is not in cur_rank
                reserved_vars = []
                for input_name in op.input_arg_names:
                    if (
                        input_name not in removed_tmp_var
                        and self.clip_helper.is_local_var_with_dist_attr(
                            input_name
                        )
                    ):
                        reserved_vars.append(input_name)
                if not reserved_vars:
                    removed_op_idx.add(idx)
                    removed_tmp_var.update(set(op.output_arg_names))
                    if block.ops[idx + 1].type == 'reduce_sum':
                        removed_op_idx.add(idx + 1)
                        removed_tmp_var.update(
                            set(block.ops[idx + 1].output_arg_names)
                        )
                    if block.ops[idx + 2].type == 'cast':
                        removed_op_idx.add(idx + 2)
                        removed_tmp_var.update(
                            set(block.ops[idx + 2].output_arg_names)
                        )
                else:
                    op.desc.set_input("X", reserved_vars)

        for idx, op in reversed(list(enumerate(block.ops))):
            if not (is_optimize_op(op) or is_reshard_op(op)):
                break
            if not is_gradient_clip_op(op):
                continue
            if idx in removed_op_idx:
                block._remove_op(idx, sync=False)

        for idx, op in reversed(list(enumerate(block.ops))):
            if not (is_optimize_op(op) or is_reshard_op(op)):
                break
            if not is_gradient_clip_op(op):
                continue
            if op.type == 'sqrt':
                input_name = op.input("X")[0]
                input_var = block.vars[input_name]
                insert_leaf_fill_constant_node = False
                if paddle.distributed.get_world_size() > 1:
                    offset = 0
                    if input_name in removed_tmp_var:
                        removed_tmp_var.remove(input_name)
                        fill_constant_op = block._insert_op(
                            idx,
                            type='fill_constant',
                            inputs={},
                            outputs={'Out': [input_var]},
                            attrs={
                                'shape': [],
                                'dtype': input_var.dtype,
                                'value': 0,
                                'force_cpu': False,
                                OP_ROLE_KEY: OpRole.Optimize,
                            },
                        )
                        fill_constant_op._set_attr(
                            'op_namescope', "/gradient_clip_pass"
                        )
                        offset += 1
                        self.clip_helper._init_dist_attr(fill_constant_op)
                        insert_leaf_fill_constant_node = True

                    allreduce_op = block._insert_op(
                        idx + offset,
                        type='c_allreduce_sum',
                        inputs={'X': [input_var]},
                        outputs={'Out': [input_var]},
                        attrs={
                            'ring_id': 0,
                            'use_calc_stream': True,
                            OP_ROLE_KEY: OpRole.Optimize,
                        },
                    )
                    # TODO better regular the usage of op namescope
                    allreduce_op._set_attr(
                        'op_namescope', '/' + SyncMode.GlobalNormSync
                    )
                    self.clip_helper._init_dist_attr(allreduce_op)

                    if insert_leaf_fill_constant_node:
                        # NOTE add naive deps for global norm sync in graph exe
                        j = idx - 1
                        prior_op = None
                        while j > 0:
                            op_type = block.ops[j].type
                            if op_type in [
                                'update_loss_scaling',
                                'check_finite_and_unscale',
                            ] or op_type.endswith("_grad"):
                                prior_op = block.ops[j]
                                break
                            j -= 1
                        assert (
                            prior_op is not None
                        ), "Unexpected: ClipByGlobalNorm could not find priory depend op"
                        prior_var = block.vars[prior_op.output_arg_names[0]]
                        assert (
                            prior_var is not None
                        ), "Unexpected: ClipByGlobalNorm could not find priory depend var"
                        insert_dependencies_for_vars(
                            block,
                            idx,
                            prior_var,
                            input_var,
                            self.clip_helper.dist_context,
                            OpRole.Optimize,
                            process_mesh=[
                                -1
                            ],  # hack to avoid initialize the dist attr for coalesce var
                            is_recompute=False,
                            sync=False,
                            op_namescope="grad_clip_fill_constant_dep",
                        )

        for varname in removed_tmp_var:
            block._remove_var(varname, sync=False)

        block._sync_with_cpp()
