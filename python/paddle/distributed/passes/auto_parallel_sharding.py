# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import logging
from functools import reduce

import paddle
import paddle.distributed as dist
from paddle.distributed.auto_parallel.static.operators.common import (
    ParallelMode,
    is_data_parallel_reduce_op,
    is_parameter_related,
)
from paddle.distributed.auto_parallel.static.process_group import (
    new_process_group,
)
from paddle.distributed.auto_parallel.static.utils import (
    _get_comm_group,
    get_logger,
    get_var_numel,
    insert_dependencies_for_vars,
    is_backward_op,
    is_dep_skip_op,
    is_forward_op,
    is_optimize_op,
    naive_set_dist_op_attr_for_program_by_mesh,
    naive_set_dist_op_attr_for_program_by_mesh_and_mapping,
    set_var_dist_attr,
)
from paddle.distributed.fleet.meta_optimizers.sharding.utils import get_var_size
from paddle.framework import core
from paddle.static import default_main_program, default_startup_program
from paddle.utils import unique_name

from .auto_parallel_master_grad import _is_master_grad_cast_op
from .pass_base import PassBase, register_pass
from .pass_utils import AutoParallelStreamType

OpRole = core.op_proto_and_checker_maker.OpRole
OP_ROLE_KEY = core.op_proto_and_checker_maker.kOpRoleAttrName()
_skip_ops = [
    'create_py_reader',
    'create_double_buffer_reader',
    'read',
    'slice',
    'split',
    'assign',
    "send_v2",
]
# update here to support new optimizers
_supported_optimizer_type = [
    "adam",
    "adamax",
    "adamw",
    "decayed_adagrad",
    "momentum",
    "dgc_momentum",
    "lars_momentum",
    "merged_momentum",
    "lamb",
    "sgd",
]

_logger = get_logger(logging.INFO)

__amp_target_dtype__ = core.VarDesc.VarType.FP16
__amp_target_dtype_name__ = "float16"


def _is_reshard_op(op):
    return op.desc.has_attr(
        "op_namescope"
    ) and "/auto_parallel/reshard" in op.desc.attr('op_namescope')


# NOTE we add the "auto_parallel" prefix to the pass in order to
# indicate that this pass should obey some constrains by auto_parallel
# for example all ops and vars should has dist attr before and after pass
# should use dist op instead of custom comm op
@register_pass("auto_parallel_sharding")
class ShardingPass(PassBase):
    def __init__(self):
        super().__init__()
        self.set_attr("dist_context", None)
        self.set_attr("stage", None)
        self.set_attr("sharding_degree", None)  # for parallelizer
        self.set_attr("degree", None)  # for parallelizer_v2
        self.set_attr("enable_overlap", None)
        self.set_attr("param_comm_stream_num", None)
        self.set_attr("grad_comm_stream_num", None)
        self.set_attr("param_bucket_size_numel", None)
        self.set_attr("grad_bucket_size_numel", None)
        self.set_attr("partition_algor", None)
        self.set_attr("enable_hierarchical_comm", None)
        self.set_attr("params_grads", [])
        self.set_attr("global_rank", -1)
        self.set_attr("amp_dtype", "float16")
        self.set_attr("gradient_sync_after_accumulate", False)
        self.dp_groups = set()
        self.sharding_infos = []
        self.varname_to_sharding_info = {}
        self.sharding_hybrid_dp = False
        self.outer_dp_group = None
        self.shared_params_grads = []

    def _check_self(self):
        if self.get_attr("dist_context") is None:
            return False

        if self.get_attr("stage") not in [1, 2, 3]:
            return False
        if self.get_attr("sharding_degree") is not None:
            if (
                not isinstance(self.get_attr("sharding_degree"), int)
            ) or self.get_attr("sharding_degree") <= 1:
                return False
        elif self.get_attr("degree") is not None:
            if (not isinstance(self.get_attr("degree"), int)) or self.get_attr(
                "degree"
            ) <= 1:
                return False
        else:
            return False
        if len(self.get_attr("params_grads")) <= 0:
            return False
        if (not isinstance(self.get_attr("global_rank"), int)) or self.get_attr(
            "global_rank"
        ) < 0:
            return False
        if self.get_attr("enable_overlap") is None:
            return False
        if self.get_attr("param_comm_stream_num") is None:
            return False
        if self.get_attr("grad_comm_stream_num") is None:
            return False
        if self.get_attr("param_bucket_size_numel") is None:
            return False
        if self.get_attr("grad_bucket_size_numel") is None:
            return False
        if self.get_attr("partition_algor") is None:
            return False
        if self.get_attr("enable_hierarchical_comm") is None:
            return False
        return True

    def _check_conflict(self, other_pass):
        return True

    def _apply_single_impl(self, main_program, startup_program, context):
        self._dist_context = self.get_attr("dist_context")
        self.sharding_world_size = int(
            self.get_attr("sharding_degree") or self.get_attr("degree")
        )
        self.stage = int(self.get_attr("stage"))
        self.global_rank = int(self.get_attr("global_rank"))
        self.enable_overlap = self.get_attr("enable_overlap")
        self.param_comm_stream_num = int(self.get_attr("param_comm_stream_num"))
        self.grad_comm_stream_num = int(self.get_attr("grad_comm_stream_num"))
        self.enable_hierarchical_comm = self.get_attr(
            "enable_hierarchical_comm"
        )
        if self.param_comm_stream_num > 1 or self.grad_comm_stream_num > 1:
            assert (
                self.enable_overlap
            ), "multiple comm stream need enable_overlap to be True"
        self.param_bucket_size_numel = int(
            self.get_attr("param_bucket_size_numel")
        )
        self.grad_bucket_size_numel = int(
            self.get_attr("grad_bucket_size_numel")
        )
        self.partition_algor = self.get_attr("partition_algor")

        params_grads = self.get_attr("params_grads")
        main_block, startup_block = (
            main_program.global_block(),
            startup_program.global_block(),
        )

        self.amp_dtype = self.get_attr("amp_dtype")
        if self.amp_dtype == "bfloat16":
            __amp_target_dtype__ = core.VarDesc.VarType.BF16
            __amp_target_dtype_name__ = "bfloat16"

        # NOTE Multi / Sub-Block Support
        # we assume that only parameter are present and partitioned in main_block,
        # there is NO new param in sub_block, and all params in sub_block follows the same
        # partition as main_block. the above constraint fulfill the 3 most common use-cases in Paddle sub_block:
        # 1. subblock for lr scheduler
        # 2. sub-block uses the same or partial network of main-block, e.g. GPT3 generation model
        # 3. sub-block used for double backward

        self._build_sharding_groups(main_block, params_grads)
        for block in main_program.blocks:
            self._shard_optimizer(block, startup_block)
            self._shard_gradient_synchronization(block)
            self._shard_parameter(block, startup_block)

        context.set_attr("params_grads", self.shared_params_grads)
        self._optimization_pass(main_program, startup_program)

    def _build_sharding_groups(self, main_block, params_grads):
        self._collective_data_parallel_groups(main_block)
        self._build_sharding_infos(main_block, params_grads)

    def _collective_data_parallel_groups(self, main_block):
        for op in main_block.ops:
            if not is_forward_op(op) or op.type in _skip_ops:
                continue
            # NOTE: there aren't dist_attr in the ops which reshard insert,
            # and should be skip in sharding.
            if _is_reshard_op(op):
                continue
            group = _inference_data_parallel_group_for_operator(
                self.global_rank, op, self._dist_context, 0
            )
            if group is not None:
                self.dp_groups.add(group)

        # TODO(JZ-LIANG) allow more than one dp groups in network, support more general distribution
        # generated by auto search
        if len(self.dp_groups) != 1:
            raise NotImplementedError(
                f"So far Only and Exactly one data parallel group in network are supported, but got [{len(self.dp_groups)}] different data parallel groups"
            )

    def _build_sharding_infos(self, main_block, params_grads):
        # order params
        params_grads = re_order_program(
            main_block, params_grads, self._dist_context
        )

        # partition
        for dp_group in self.dp_groups:
            assert (
                dp_group.nranks >= self.sharding_world_size
            ), f"sharding world size [{self.sharding_world_size}] should not larger than dp world size [{dp_group.nranks}]"
            assert (
                dp_group.nranks % self.sharding_world_size == 0
            ), f"sharding world size [{self.sharding_world_size}] should be divisible by dp world size [{dp_group.nranks}]"
            assert (
                self.global_rank in dp_group.ranks
            ), f"current ranks [{self.global_rank}] does NOT belong to the data parallel group [{dp_group.ranks}]"
            assert (
                len(params_grads) >= self.sharding_world_size
            ), f"number of parameters [{len(params_grads)}] is not enough to be shard among [{self.sharding_world_size}] ranks"

            # sharding hybrid data parallel: partial sharding param within
            if dp_group.nranks > self.sharding_world_size:
                self.sharding_hybrid_dp = True
                assert self.param_comm_stream_num < 2
                assert self.grad_comm_stream_num < 2
                assert (
                    len(self.dp_groups) == 1
                ), "hybrid sharding and data parallelism are supported only when there is exactly one data parallel group in the network"
                outer_dp_group, sharding_group = _get_dp_and_sharding_groups(
                    dp_group.ranks, self.sharding_world_size, self.global_rank
                )
                sharding_group = new_process_group(sharding_group)
                self.outer_dp_group = new_process_group(outer_dp_group)
            else:
                sharding_group = dp_group

            self._dist_context._sharding_group = sharding_group
            # TODO(JZ-LIANG) when support multiple dp groups in future, should group param and bind them to corresponding dp group
            sharding_info = ShardingInfo(
                sharding_group,
                self.global_rank,
                params_grads,
                self.partition_algor,
            )
            self.sharding_infos.append(sharding_info)
            for param in sharding_info.params:
                self.varname_to_sharding_info[param.name] = sharding_info

    def _shard_optimizer(self, main_block, startup_block):
        """
        sharding all optimizer related ops and vars, include:
        gradient clip ops & vars
        weight decay ops & vars
        optimizer ops and states
        """
        self._shard_amp_related_op_and_vars(main_block)
        self._shard_weight_decay(main_block)
        # self._shard_gradient_clip(main_block)
        self._shard_optimizer_ops_and_states(main_block, startup_block)
        self._insert_optimizer_broadcasts(main_block, startup_block)

    def _shard_amp_related_op_and_vars(self, main_block):
        if self.stage < 1:
            return

        for idx, op in reversed(list(enumerate(main_block.ops))):
            # shard amp related param_grad cast
            if _is_param_grad_fp32_cast_op(main_block, op) and self.stage > 1:
                output_name = op.output_arg_names[0]
                param_name = output_name[: output_name.find("@")]
                if not self._is_parameter_in_local_shard(param_name):
                    main_block._remove_op(idx, sync=False)
                    main_block._remove_var(output_name, sync=False)

            # shard check nan inf
            elif op.type in ["check_finite_and_unscale", "update_loss_scaling"]:
                reversed_x = []
                for input_name in op.desc.input('X'):
                    param_name = input_name[: input_name.find("@")]

                    if self._is_parameter_in_local_shard(param_name):
                        reversed_x.append(input_name)

                # NOTE: When `reversed_x` is [], check_finite_and_unscale will be replaced by `fill_constant` op.
                # The output of check_finite_and_unscale is be set False
                if reversed_x:
                    op.desc.set_input('X', reversed_x)
                    op.desc.set_output('Out', reversed_x)
                else:
                    if op.type == "check_finite_and_unscale":
                        op_role = op.attr('op_role')
                        out_name = op.output_arg_names[0]
                        out_var = main_block.vars[out_name]
                        main_block._remove_op(idx, sync=False)
                        main_block._insert_op_without_sync(
                            idx,
                            type="fill_constant",
                            outputs={"Out": out_var},
                            attrs={
                                "shape": out_var.shape,
                                "dtype": out_var.dtype,
                                "value": 0,
                                OP_ROLE_KEY: op_role,
                            },
                        )
                        dist_attr = (
                            self._dist_context.get_tensor_dist_attr_for_program(
                                out_var
                            )
                        )

                        naive_set_dist_op_attr_for_program_by_mesh_and_mapping(
                            main_block.ops[idx],
                            dist_attr.process_mesh,
                            dist_attr.dims_mapping,
                            self._dist_context,
                            chunk_id=dist_attr.chunk_id,
                        )

                    else:
                        main_block._remove_op(idx, sync=False)

        main_block._sync_with_cpp()

    def _shard_gradient_clip(self, main_block):
        if self.stage < 2:
            return

        # TODO (JZ-LIANG) support calculate global norm with tensor parallelism
        removed_op_type = ['elementwise_mul', 'squared_l2_norm', 'clip_by_norm']
        removed_op_idx = set()
        removed_tmp_var = set()

        for idx, op in list(enumerate(main_block.ops)):
            if not _is_gradient_clip_op(op):
                continue

            if op.type in removed_op_type:
                input_name = op.input("X")[0]
                param_name = input_name[: input_name.find("@GRAD")]
                if not self._is_parameter_in_local_shard(param_name):
                    removed_op_idx.add(idx)
                    if op.type in ['squared_l2_norm', 'clip_by_norm']:
                        for output_name in op.output_arg_names:
                            removed_tmp_var.add(output_name)

        for idx, op in reversed(list(enumerate(main_block.ops))):
            if not _is_gradient_clip_op(op):
                continue
            if idx in removed_op_idx:
                main_block._remove_op(idx, sync=False)

        for varname in removed_tmp_var:
            main_block._remove_var(varname, sync=False)

        for idx, op in list(enumerate(main_block.ops)):
            if not _is_gradient_clip_op(op):
                continue
            if op.type == 'sum':
                reserved_vars = []
                for input_name in op.input_arg_names:
                    if input_name not in removed_tmp_var:
                        reserved_vars.append(input_name)
                op.desc.set_input("X", reserved_vars)

                sum_op_output = op.output_arg_names[0]
                for i, sharding_info in enumerate(self.sharding_infos):
                    new_op = main_block._insert_op(
                        idx + i + 1,
                        type='c_allreduce_sum',
                        inputs={'X': [sum_op_output]},
                        outputs={'Out': [sum_op_output]},
                        attrs={
                            'ring_id': sharding_info.group.id,
                            'op_namescope': "/gradient_clip_model_parallelism",
                            'use_calc_stream': True,
                            OP_ROLE_KEY: OpRole.Optimize,
                        },
                    )
                    dist_attr = (
                        self._dist_context.get_tensor_dist_attr_for_program(
                            main_block.var(sum_op_output)
                        )
                    )
                    # assert dist_attr is not None
                    # naive_set_dist_op_attr_for_program_by_mesh_and_mapping(
                    #     new_op, dist_attr.process_mesh, dist_attr.dims_mapping,
                    #     self._dist_context)
                break

        main_block._sync_with_cpp()

    def _shard_weight_decay(self, main_block):
        if self.stage < 2:
            return

        for idx, op in reversed(list(enumerate(main_block.ops))):
            if not _is_weight_decay_op(op):
                continue
            else:
                raise NotImplementedError(
                    "weight decay is NOT supported by now"
                )
        main_block._sync_with_cpp()

    def _shard_optimizer_ops_and_states(self, main_block, startup_block):
        should_removed_optimizer_states = []
        for idx, op in reversed(list(enumerate(main_block.ops))):
            if not is_optimize_op(op):
                break

            if op.type in _supported_optimizer_type:
                assert "Param" in op.input_names
                assert len(op.input("Param")) == 1
                param_name = op.input("Param")[0]
                if not self._is_parameter_in_local_shard(param_name):
                    should_removed_optimizer_states.extend(
                        [
                            varname
                            for varname in op.output_arg_names
                            if varname != param_name
                        ]
                    )
                    main_block._remove_op(idx, sync=False)
                else:
                    self.shared_params_grads.append(
                        self._get_param_grad(param_name)
                    )

        for idx, op in reversed(list(enumerate(startup_block.ops))):
            if (
                len(op.output_arg_names) == 1
                and op.output_arg_names[0] in should_removed_optimizer_states
            ):
                startup_block._remove_op(idx, sync=False)

        for varname in should_removed_optimizer_states:
            if main_block.has_var(varname):
                main_block._remove_var(varname, sync=False)
            if startup_block.has_var(varname):
                startup_block._remove_var(varname, sync=False)

        main_block._sync_with_cpp()
        startup_block._sync_with_cpp()

    def _insert_optimizer_broadcasts(self, main_block, startup_block):
        if self.stage > 2 or self.param_bucket_size_numel > 1:
            return

        for sharding_info in self.sharding_infos:
            for param in sharding_info.params:
                assert main_block.has_var(param.name)
                assert startup_block.has_var(param.name)

                new_op = main_block.append_op(
                    type='broadcast',
                    inputs={'x': param},
                    outputs={'out': param},
                    attrs={
                        'ring_id': sharding_info.group.id,
                        'root': sharding_info.get_var_rank(param.name),
                        OP_ROLE_KEY: OpRole.Optimize,
                    },
                )
                new_op._set_attr(
                    'op_namescope', '/' + ParallelMode.DataParallel
                )
                param_dist_attr = (
                    self._dist_context.get_tensor_dist_attr_for_program(param)
                )
                assert param_dist_attr is not None
                naive_set_dist_op_attr_for_program_by_mesh_and_mapping(
                    new_op,
                    param_dist_attr.process_mesh,
                    param_dist_attr.dims_mapping,
                    self._dist_context,
                    chunk_id=param_dist_attr.chunk_id,
                )
        main_block._sync_with_cpp()

    def _is_parameter_in_local_shard(self, param_name):
        assert param_name in self.varname_to_sharding_info
        sharding_info = self.varname_to_sharding_info[param_name]
        return sharding_info.is_in_local_shard(param_name)

    def _get_param_grad(self, param_name):
        assert param_name in self.varname_to_sharding_info
        sharding_info = self.varname_to_sharding_info[param_name]
        p_g = sharding_info.get_param_grad(param_name)
        assert p_g is not None
        return p_g

    def _shard_gradient_synchronization(self, main_block):
        if self.stage < 2:
            return

        dp_ring_ids = [group.id for group in self.dp_groups]
        for idx, op in reversed(list(enumerate(main_block.ops))):
            if _is_param_grad_allreduce_op(op, main_block):
                if op.type == "c_allreduce_sum" or (
                    op.type == "reduce"
                    and op.attr("reduce_type") == dist.ReduceOp.SUM
                ):
                    reduce_op_type = "reduce"
                    reduce_type = dist.ReduceOp.SUM
                else:
                    reduce_op_type = "reduce"
                    reduce_type = dist.ReduceOp.AVG
                input_name = op.input_arg_names[0]
                base_name = _get_base_name_from_grad_name(input_name)
                sharding_info = self.varname_to_sharding_info[base_name]
                reduce_op = _insert_reduce_op(
                    main_block,
                    reduce_op_type,
                    idx,
                    input_name,
                    sharding_info.group.id,
                    sharding_info.get_var_rank(base_name),
                    self._dist_context,
                    reduce_type,
                )
                if (
                    not self.sharding_hybrid_dp
                    or not sharding_info.is_in_local_shard(base_name)
                ):
                    main_block._remove_op(idx + 1, sync=False)
                else:
                    op._set_attr("ring_id", self.outer_dp_group.id)
                    op._set_attr(
                        'op_namescope', '/' + ParallelMode.DataParallel
                    )

            # NOTE:
            # var@GRAD = sum(var@GRAD@RENAME@0, var@GRAD@RENAME@1)
            # If the var is not in local rank and it is output of many ops, or the var is renamed in another words,
            # the sum op should be removed.
            if _is_param_grad_sum_op(op, main_block):
                out_name = op.output_arg_names[0]
                base_name = _get_base_name_from_grad_name(out_name)
                sharding_info = self.varname_to_sharding_info[base_name]
                if not sharding_info.is_in_local_shard(base_name):
                    main_block._remove_op(idx, sync=False)

        main_block._sync_with_cpp()

    def _shard_parameter(self, main_block, startup_block):
        if self.stage < 3:
            return

        dp_ring_ids = [group.id for group in self.dp_groups]
        for sharding_info in self.sharding_infos:
            (
                need_broadcast_vars,
                param_usage,
            ) = sharding_info.get_broadcast_vars_and_param_usage(main_block)
            not_used_param_name = []
            for param_name in param_usage:
                if (
                    param_usage[param_name] == 0
                    and sharding_info.get_var_rank(param_name)
                    != sharding_info.local_rank
                ):
                    not_used_param_name.append(param_name)

            for idx, op in reversed(list(enumerate(main_block.ops))):
                if is_optimize_op(op):
                    continue

                for input_name in op.input_arg_names:
                    # NOTE hack for embedding op when AMP 02-3
                    # paddle amp force embedding (lookup table) to be run on fp32
                    if _is_param_fp16_cast_op(
                        main_block, op, sharding_info.param_names
                    ):
                        # NOTE:
                        # param.cast_fp16 = cast(param)
                        # When param is not in current rank, the cast op need to be removed.
                        if not self._is_parameter_in_local_shard(input_name):
                            not_used_param_name.append(input_name)
                        continue
                    if input_name not in need_broadcast_vars:
                        continue
                    root_rank = sharding_info.get_var_rank(input_name)
                    if root_rank == sharding_info.local_rank:
                        broadcast_varname = input_name
                    else:
                        broadcast_varname = unique_name.generate(
                            input_name + "@BroadCast"
                        )
                        input_var = main_block.var(input_name)
                        new_var = main_block.create_var(
                            name=broadcast_varname,
                            shape=input_var.shape,
                            dtype=input_var.dtype,
                            persistable=False,
                        )
                        ref_dist_attr = (
                            self._dist_context.get_tensor_dist_attr_for_program(
                                input_var
                            )
                        )
                        set_var_dist_attr(
                            self._dist_context,
                            new_var,
                            ref_dist_attr.dims_mapping,
                            ref_dist_attr.process_mesh,
                            chunk_id=ref_dist_attr.chunk_id,
                        )
                        op_dist_attr = (
                            self._dist_context.get_op_dist_attr_for_program(op)
                        )
                        input_dist_attr = op_dist_attr.get_input_dist_attr(
                            input_name
                        )
                        op._rename_input(input_name, broadcast_varname)
                        op_dist_attr.set_input_dist_attr(
                            broadcast_varname, input_dist_attr
                        )

                    _insert_init_and_broadcast_op(
                        main_block,
                        idx,
                        broadcast_varname,
                        sharding_info.local_rank,
                        root_rank,
                        sharding_info.group.id,
                        op.attr('op_role'),
                        self._dist_context,
                    )

            for idx, op in reversed(list(enumerate(main_block.ops))):
                if op.type != "cast":
                    continue
                input_name = op.input_arg_names[0]
                output_name = op.output_arg_names[0]
                if input_name in not_used_param_name:
                    main_block._remove_op(idx, sync=False)
                    main_block._remove_var(output_name, sync=False)

            for idx, op in reversed(list(enumerate(startup_block.ops))):
                assert len(op.output_arg_names) == 1
                output_name = op.output_arg_names[0]

                if op.type == "broadcast":
                    if op.attr("ring_id") in dp_ring_ids:
                        if (
                            self.outer_dp_group
                            and sharding_info.get_var_rank(output_name)
                            == sharding_info.local_rank
                        ):
                            op._set_attr("ring_id", self.outer_dp_group.id)
                        else:
                            startup_block._remove_op(idx, sync=False)
                    else:  # We should remove the `broadcast` between `TensorParallel` mesh dim.
                        if (
                            sharding_info.get_var_rank(output_name)
                            != sharding_info.local_rank
                        ):
                            startup_block._remove_op(idx, sync=False)
                    continue

                if (
                    op.type != "broadcast"
                    and output_name in param_usage
                    and sharding_info.get_var_rank(output_name)
                    != sharding_info.local_rank
                ):
                    startup_block._remove_op(idx, sync=False)

            for param_name in param_usage:
                if (
                    sharding_info.get_var_rank(param_name)
                    != sharding_info.local_rank
                ):
                    main_block._remove_var(param_name, sync=False)
                    startup_block._remove_var(param_name, sync=False)

        main_block._sync_with_cpp()
        startup_block._sync_with_cpp()

    def _optimization_pass(self, main_program, startup_program):
        if self.stage <= 1:
            return

        self.grad_coalesce_prefix = 'sharding_coalesce_grad_'
        self.param_coalesce_prefix = 'sharding_coalesce_param_'
        # NOTE PR#49275 for detail
        self.comm_op_scheduling_priority = -1

        # TODO support multiple sub_blocks
        assert (
            len(self.sharding_infos) == 1
        ), f"gradient synchronization optimization only support one sharding group right now, but got [{len(self.sharding_infos)}]."
        sharding_info = self.sharding_infos[0]

        with paddle.static.program_guard(main_program, startup_program):
            self._gradient_sync_optimization(sharding_info)
            # TODO independent the logic of fuse and overlap
            # support overlap when no fuse
            if self.param_bucket_size_numel > 1:
                if self.stage == 2:
                    self._fuse_overlap_parameter_comm_stage_two(sharding_info)
                elif self.stage == 3:
                    self._fuse_overlap_parameter_comm_stage_three(sharding_info)

    def _gradient_sync_optimization(self, sharding_info):
        if self.grad_bucket_size_numel <= 1 and (not self.enable_overlap):
            return

        main_block = default_main_program().global_block()
        startup_block = default_startup_program().global_block()
        coalesce_to_group_map, grad_name_to_group_map = self._group_grads(
            main_block,
            sharding_info,
        )
        self._overlap_grad_comm(
            main_block,
            sharding_info,
            coalesce_to_group_map,
            grad_name_to_group_map,
        )

    def _fuse_overlap_parameter_comm_stage_two(self, sharding_info):
        main_block = default_main_program().global_block()
        startup_block = default_startup_program().global_block()

        group_to_param_map, param_to_group_map = group_param(
            sharding_info, self.param_bucket_size_numel
        )
        _logger.info("Sharding Stage2 Optimization:")
        _logger.info(
            f"Param Bucket size is [{self.param_bucket_size_numel}], [{len(param_to_group_map.keys())}] Parameters are fused into [{len(group_to_param_map.keys())}] Buckets"
        )
        broadcast_var_to_group_map = {}

        if self.enable_overlap:
            # if the communication is cross node, comm will be slow and calc will therefore
            # wait for comm. enable multi-comm-stream
            # TODO revise me in future
            # 1. manager the comm and corresponding stream
            # 2. allow more than two streams and open to be config
            self.param_comm_group_stream_pairs = []
            ranks = sharding_info.group.ranks
            for i in range(self.param_comm_stream_num):
                if i == 0:
                    group = sharding_info.group
                else:
                    group = new_process_group(ranks, force_new_group=True)

                self.param_comm_group_stream_pairs.append(
                    {
                        "comm_group": group,
                        "comm_stream": AutoParallelStreamType.SHARDING_STREAM.value,
                    }
                )
            _logger.info(
                f"Parameter Communication would use [{self.param_comm_stream_num}] streams."
            )
            self.op_to_stream_idx = {}

        for i, param_group in enumerate(group_to_param_map.keys()):
            assert len(param_group) >= 1
            if len(param_group) > 1:
                coalesce_var_name = unique_name.generate(
                    self.param_coalesce_prefix + str(i)
                )
                startup_block.create_var(
                    name=coalesce_var_name,
                    dtype=param_group.dtype,
                    persistable=True,
                    stop_gradient=True,
                )
                param_group.coalesce_var = main_block.create_var(
                    name=coalesce_var_name,
                    dtype=param_group.dtype,
                    persistable=True,
                    stop_gradient=True,
                )
                startup_block.append_op(
                    type="coalesce_tensor",
                    inputs={"Input": param_group.vars},
                    outputs={
                        "Output": param_group.vars,
                        "FusedOutput": param_group.coalesce_var,
                    },
                    attrs={
                        "copy_data": True,
                        "use_align": True,
                        "dtype": param_group.dtype,
                        OP_ROLE_KEY: OpRole.Forward,
                    },
                )
            else:
                param_group.coalesce_var = param_group.vars[0]
            _logger.info(
                f"Bucket[{i}] size [{sum([get_var_size(p) for p in param_group.vars])}]MB."
            )
            _logger.debug(
                f"Bucket[{i}] parameters: {[p.name for p in param_group.vars]}."
            )

            broadcast_var_to_group_map[param_group.coalesce_var.name] = (
                param_group
            )

            # TODO revise me to manager stream and comm
            comm_stream_idx = i % self.param_comm_stream_num
            comm_group = self.param_comm_group_stream_pairs[comm_stream_idx][
                'comm_group'
            ]
            comm_stream = self.param_comm_group_stream_pairs[comm_stream_idx][
                'comm_stream'
            ]
            new_op = main_block.append_op(
                type='broadcast',
                inputs={'x': param_group.coalesce_var},
                outputs={'out': param_group.coalesce_var},
                attrs={
                    'ring_id': comm_group.id,
                    'root': param_group.rank,
                    OP_ROLE_KEY: OpRole.Optimize,
                },
            )
            self.op_to_stream_idx[new_op] = comm_stream_idx
            new_op._set_attr('op_namescope', '/' + ParallelMode.DataParallel)
            if self.enable_overlap:
                new_op.dist_attr.execution_stream = comm_stream
                new_op.dist_attr.scheduling_priority = (
                    self.comm_op_scheduling_priority
                )

            # NOTE the current dist context lack the presentation for bucket tensor which
            # composes many tensor with different dims_mapping. we DO NOT assign dist attr
            # for it currently.

        # add dependencies:
        # 1. all broadcast depend on its pre collective
        # 2. coalesce broadcast add nop to resolute data flow dependencies
        dep_map = {}
        for i, op in enumerate(main_block.ops):
            if is_sharding_param_broadcast_op(op):
                broadcast_varname = op.output("Out")[0]
                broadcast_var = main_block.vars[broadcast_varname]
                param_group = broadcast_var_to_group_map[broadcast_varname]
                comm_stream = None
                if self.enable_overlap:
                    comm_stream = op.dist_attr.execution_stream

                # FIXME remove me when upgrade to multi-comm version
                if len(dep_map.keys()) < self.param_comm_stream_num:
                    op = _get_broadcast_first_depend_op(main_block)
                    prior_var = main_block.vars[op.output("ParamOut")[0]]
                else:
                    pre_op = main_block.ops[i - self.param_comm_stream_num]
                    assert is_sharding_param_broadcast_op(
                        pre_op
                    ), "Unexpected: sharding broadcast pre op should be broadcast."
                    prior_var = main_block.vars[pre_op.output("Out")[0]]
                # broadcast order dependencies
                dep_map[i] = [(i, [prior_var], [broadcast_var], comm_stream)]

                if len(param_group.vars) > 1:
                    # in shard coalesce depend to optimizer
                    if param_group.is_in_local_shard:
                        last_grad = param_group.vars[-1]
                        dep_map[i].append(
                            (i, [last_grad], [broadcast_var], comm_stream)
                        )
                    # coalesce resolution post deps
                    dep_map[i].append(
                        (i + 1, [broadcast_var], param_group.vars, comm_stream)
                    )

        # insert deps
        indice = sorted(dep_map.keys(), reverse=True)
        for i in indice:
            for idx, prior_vars, post_vars, comm_stream in dep_map[i][::-1]:
                depend_op = insert_dependencies_for_vars(
                    main_block,
                    idx,
                    prior_vars,
                    post_vars,
                    self._dist_context,
                    OpRole.Optimize,
                    process_mesh=[
                        -1
                    ],  # hack to avoid initialize the dist attr for coalesce var
                    is_recompute=False,
                    sync=False,
                    op_namescope="sharding_stage2_broadcast_dep",
                )
                if self.enable_overlap and depend_op is not None:
                    depend_op.dist_attr.execution_stream = comm_stream
                    depend_op.dist_attr.scheduling_priority = (
                        self.comm_op_scheduling_priority
                    )

        main_block._sync_with_cpp()

    def _fuse_overlap_parameter_comm_stage_three(self, sharding_info):
        pass

    def _group_grads(
        self,
        block,
        sharding_info,
    ):
        """
        conditions for gradients to be grouped:
            1. group size < grad_bucket_size_numel
            2. same dp group (TODO)
            3. same src rank
            4. same dtype
            5. dependency: grad would NOT be used by other ops within group segment

        main logic:
            1. record coalesce group
            2. record all dp allreduce/reduce op idx

            3. insert coalesce op
            4. insert coalesce dependency (avoid allocate memory too early)
            5. modify and remove allreduce/reduce op
            6. ensure sharding-dp hybrid parallel logic

        gradients inside same group would be fuse into one coalesce tensor
        """
        ops = block.ops
        if self.grad_bucket_size_numel < 1:
            # numel for transformer layer
            # h = 4096 + 1
            # ffn_numel = 2 * (4 * h) * h
            # mha_numel = 3 * h * h + h * h
            # max_fuse_numel = ffn_numel + mha_numel
            self.grad_bucket_size_numel = 1

        first_backward_op = None
        for op in ops:
            if is_backward_op(op):
                first_backward_op = op
                break
        # not backward op, sharding for inference
        if first_backward_op is None:
            return
        first_backward_varname = first_backward_op.output_arg_names[0]

        cur_group = VarGroup(self.grad_bucket_size_numel)
        grad_groups = []
        grouped_grad_names = set()

        def op_depend_on_group(op, group):
            vars_ = set(op.input_arg_names + op.output_arg_names)
            var_names = {var.name for var in group.vars}
            return len(vars_.intersection(var_names)) > 0

        # analyze groups
        i = 0
        while i < len(ops):
            op = ops[i]
            if is_data_parallel_reduce_op(op):
                is_reduce = op.type == "reduce" and op.attr("reduce_type") in [
                    dist.ReduceOp.AVG,
                    dist.ReduceOp.SUM,
                ]
                assert (
                    is_reduce
                ), "Sharding should reduce grad first and than allreduce if Hybrid Sharding with Data-Parallel"

                grad_name = op.output_arg_names[0]
                param_name = _get_base_name_from_grad_name(grad_name)
                rank = sharding_info.get_var_rank(param_name)
                grad_var = block.var(grad_name)

                if cur_group.acceptable(grad_var, rank):
                    assert grad_name not in grouped_grad_names
                    cur_group.collect(grad_var, rank)
                else:
                    grad_groups.append(cur_group)
                    cur_group = VarGroup(self.grad_bucket_size_numel)
                    cur_group.collect(grad_var, rank)

                if len(cur_group.vars) == 1:
                    cur_group.coalesce_op_idx = i - 1
                    # NOTE coalesce dependency: control when allocate memory for gradients
                    # too early would increase the peak memory requirement, too later would hurt the performance
                    j = 2
                    while is_dep_skip_op(ops[i - j]):
                        j += 1
                    dep_op = ops[i - j]
                    dep_varname = dep_op.output_arg_names[0]
                    cur_group.coalesce_dep_varname = dep_varname

                grouped_grad_names.add(grad_name)
                cur_group.reduce_op_indices.append(i)

                if self.sharding_hybrid_dp and sharding_info.is_in_local_shard(
                    param_name
                ):
                    cur_group.is_in_local_shard = True
                    assert ops[i + 1].type in [
                        "c_allreduce_avg",
                        "c_allreduce_sum",
                    ], "Sharding should reduce grad first and than allreduce if Hybrid Sharding with Data-Parallel"
                    assert (
                        ops[i + 1].output_arg_names[0] == grad_name
                    ), "Hybrid Sharding with Data-Parallel should sync same gradient var"
                    cur_group.allreduce_op_indices.append(i + 1)
                    i += 1
            elif op_depend_on_group(op, cur_group):
                grad_groups.append(cur_group)
                cur_group = VarGroup(self.grad_bucket_size_numel)

            i += 1
        # some grad not in this rank may not be used after dp reduced
        if len(cur_group.vars) >= 1:
            grad_groups.append(cur_group)

        _logger.info("Sharding Gradient Communication Optimization:")
        _logger.info(
            f"Gradient Bucket size is [{self.grad_bucket_size_numel}], [{len(grouped_grad_names)}] Gradients are fused into [{len(grad_groups)}] Buckets."
        )

        # create coalesce tensor and record op idx
        grad_name_to_group_map = {}
        coalesce_to_group_map = {}
        modify_reduce_op_map = {}
        coalesce_op_map = {}
        remove_reduce_op_indices = []

        for i, group in enumerate(grad_groups):
            if len(group.vars) > 1:
                group.coalesce_var = block.create_var(
                    name=unique_name.generate(
                        self.grad_coalesce_prefix + str(i)
                    ),
                    dtype=group.dtype,
                    persistable=False,
                    stop_gradient=True,
                )
                ref_dist_attr = (
                    self._dist_context.get_tensor_dist_attr_for_program(
                        group.vars[0]
                    )
                )
                set_var_dist_attr(
                    self._dist_context,
                    group.coalesce_var,
                    ref_dist_attr.dims_mapping,
                    ref_dist_attr.process_mesh,
                    chunk_id=ref_dist_attr.chunk_id,
                )
                coalesce_op_map[group.coalesce_op_idx] = group
                last_reduce_op_idx = group.reduce_op_indices.pop()
                modify_reduce_op_map[last_reduce_op_idx] = group
                remove_reduce_op_indices.extend(group.reduce_op_indices)
                if group.is_in_local_shard:
                    last_allreduce_op_idx = group.allreduce_op_indices.pop()
                    modify_reduce_op_map[last_allreduce_op_idx] = group
                    remove_reduce_op_indices.extend(group.allreduce_op_indices)
            else:
                group.coalesce_var = group.vars[0]
            for grad in group.vars:
                grad_name_to_group_map[grad.name] = group
            coalesce_to_group_map[group.coalesce_var.name] = group

        coalesce_op_set = set(coalesce_op_map.keys())
        modify_op_set = set(modify_reduce_op_map.keys())
        remove_op_set = set(remove_reduce_op_indices)
        conflict = coalesce_op_set.intersection(modify_op_set)

        assert len(conflict) == 0
        conflict = coalesce_op_set.intersection(remove_op_set)
        assert len(conflict) == 0
        conflict = modify_op_set.intersection(remove_op_set)
        assert len(conflict) == 0

        # update block
        for idx, op in reversed(list(enumerate(block.ops))):
            if idx in modify_reduce_op_map:
                group = modify_reduce_op_map[idx]
                grad_name = op.output_arg_names[0]
                assert (
                    grad_name == group.vars[-1].name
                ), f"Unexpected: it is supposed to sync [{group.vars[-1].name}] but got [{grad_name}]"
                op._rename_input(grad_name, group.coalesce_var.name)
                op._rename_output(grad_name, group.coalesce_var.name)

            if idx in remove_reduce_op_indices:
                block._remove_op(idx, sync=False)

            if idx in coalesce_op_map:
                group = coalesce_op_map[idx]
                first_grad_name = group.vars[0].name
                assert (
                    first_grad_name in op.output_arg_names
                ), f"Unexpected: op is supposed to generate grad [{first_grad_name}] but got [{op}]"
                grad_names = [grad.name for grad in group.vars]

                concated_shapes = []
                concated_ranks = []
                for grad_ in group.vars:
                    shape = grad_.shape
                    concated_shapes.extend(shape)
                    concated_ranks.append(len(shape))

                coalesce_op = block._insert_op_without_sync(
                    idx,
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

                ref_dist_attr = (
                    self._dist_context.get_tensor_dist_attr_for_program(
                        group.coalesce_var
                    )
                )
                naive_set_dist_op_attr_for_program_by_mesh_and_mapping(
                    coalesce_op,
                    ref_dist_attr.process_mesh,
                    ref_dist_attr.dims_mapping,
                    self._dist_context,
                    chunk_id=ref_dist_attr.chunk_id,
                )

                depend_op = insert_dependencies_for_vars(
                    block,
                    idx,
                    block.var(group.coalesce_dep_varname),
                    group.coalesce_var,
                    self._dist_context,
                    OpRole.Backward,
                    process_mesh=[
                        -1
                    ],  # hack to avoid initialize the dist attr for coalesce var
                    is_recompute=False,
                    sync=False,
                    op_namescope="sharding_grad_coalesce_dep",
                )
        block._sync_with_cpp()

        return coalesce_to_group_map, grad_name_to_group_map

    def _overlap_grad_comm(
        self,
        block,
        sharding_info,
        coalesce_to_group_map,
        grad_name_to_group_map,
    ):
        """
        overlap gradient communication with backward & optimizer computation.

        1. assign gradient communications to grad comm stream
        2. for coalesce gradient communication:
            2.1 insert before communication dependencies
            2.2 insert after communication dependencies only when need
        3. there is not need to add explicit dependencies for non-coalesce gradient communication

        P.S. this overlap pass is ONLY adapted for standalone executor (graph based) and stream award allocator.
        """

        if not self.enable_overlap:
            return

        self.grad_comm_group_stream_pairs = []
        ranks = sharding_info.group.ranks
        # NOTE since the gradient synchronization has calculation, there would be computation
        # competition between backward calculation. therefore should limit the number of stream used.
        for i in range(self.grad_comm_stream_num):
            if i == 0:
                group = sharding_info.group
            else:
                group = new_process_group(ranks, force_new_group=True)
            # NOTE here stream is just a presentation with different name,
            # it is up to executor to create the exact streams given the name.
            stream = f"sharding_grad_comm_stream{i}"
            self.grad_comm_group_stream_pairs.append(
                {
                    "comm_group": group,
                    "comm_stream": stream,
                }
            )

        ops = block.ops
        # analyze dependencies
        dep_map = {}
        reduce_op_count = 0
        grad_comm_op_to_stream_idx = {}
        for idx, op in enumerate(ops):
            if is_data_parallel_reduce_op(op):
                if op.type in ["c_allreduce_avg", "c_allreduce_sum"]:
                    continue
                stream_idx = reduce_op_count % self.grad_comm_stream_num
                grad_comm_op_to_stream_idx[op] = stream_idx
                comm_group = self.grad_comm_group_stream_pairs[stream_idx][
                    "comm_group"
                ]
                comm_stream = self.grad_comm_group_stream_pairs[stream_idx][
                    "comm_stream"
                ]

                reduce_varname = op.output("Out")[0]
                grad_group = coalesce_to_group_map[reduce_varname]
                assert grad_group.coalesce_var.name == reduce_varname

                # coalesce deps
                if len(grad_group.vars) > 1:
                    # NOTE should prior vars to be all grads ?
                    # when the grad_ops' order is random
                    # prior dep
                    dep_map[idx] = [
                        (
                            idx,
                            grad_group.vars[-1],
                            grad_group.coalesce_var,
                            comm_stream,
                            "sharding_grad_comm_dep",
                            op.dist_attr,
                        )
                    ]
                    # post dep
                    post_idx = idx + 1
                    if self.sharding_hybrid_dp and grad_group.is_in_local_shard:
                        post_idx += 1
                    dep_map[idx].append(
                        (
                            post_idx,
                            grad_group.coalesce_var,
                            grad_group.vars,
                            comm_stream,
                            "sharding_grad_comm_dep",
                            op.dist_attr,
                        )
                    )

                # assign stream
                op.dist_attr.execution_stream = comm_stream
                op.dist_attr.scheduling_priority = (
                    self.comm_op_scheduling_priority
                )
                op._set_attr("ring_id", comm_group.id)
                if self.sharding_hybrid_dp and grad_group.is_in_local_shard:
                    next_op = ops[idx + 1]
                    assert next_op.type in [
                        "c_allreduce_avg",
                        "c_allreduce_sum",
                    ]
                    assert next_op.output("Out")[0] == reduce_varname
                    # FIXME hybrid sharding-dp support multi comm & stream in feature
                    # next_op._set_attr("ring_id", comm_group.id)
                    next_op.dist_attr.execution_stream = comm_stream
                    next_op.dist_attr.scheduling_priority = (
                        self.comm_op_scheduling_priority
                    )
                    idx += 1

                # NOTE(Ruibiao): Why add dependency here?
                # It is hack to delay GC for coalesce_var, which significantly reduce memory usage.
                # With the pattern of reduce_sum + scale, the coalesce_var is used by the reduce_sum
                # op on the comm-stream, and then released by the scale op on the comp-stream. Since
                # the generated and released op are both in comp-stream, the allocation of the
                # coalesce_var can be fast-GC and reused by subsequent comp-op. However in reduce_avg
                # parent, the coalesce_var is released on the reduce_avg op in comm-stream,
                # triggering a cross-stream GC. In such case, an event is recorded on the underlying
                # allocation, and the memory is unable to reused by other comp-ops, resulting in an
                # increase in memory usage. For more details, see the code of StreamSafeCUDAAllocator.
                # This issue should be fixed using CUDAMallocAsyncAllocator in the future.
                if (
                    op.type == "reduce"
                    and op.attr("reduce_type") == dist.ReduceOp.AVG
                    and not grad_group.is_in_local_shard
                    and not self.get_attr("gradient_sync_after_accumulate")
                ):
                    if idx not in dep_map:
                        dep_map[idx] = []
                    dep_map[idx].append(
                        (
                            idx + 1,
                            grad_group.coalesce_var,
                            grad_group.coalesce_var,
                            None,
                            "sharding_reduce_avg_dep",
                            op.dist_attr,
                        )
                    )

                reduce_op_count += 1

            idx += 1

        # insert deps
        indice = sorted(dep_map.keys(), reverse=True)
        for i in indice:
            for (
                idx,
                prior_vars,
                post_vars,
                comm_stream,
                op_namescope,
                dist_attr,
            ) in dep_map[i][::-1]:
                skip_insert_when_sequential_run = (
                    False if op_namescope == "sharding_reduce_avg_dep" else True
                )

                depend_op = insert_dependencies_for_vars(
                    block,
                    idx,
                    prior_vars,
                    post_vars,
                    self._dist_context,
                    OpRole.Backward,
                    process_mesh=[
                        -1
                    ],  # hack to avoid initialize the dist attr for coalesce var
                    is_recompute=False,
                    sync=False,
                    op_namescope=op_namescope,
                    skip_insert_when_sequential_run=skip_insert_when_sequential_run,
                )

                if depend_op is not None:
                    naive_set_dist_op_attr_for_program_by_mesh(
                        depend_op,
                        process_mesh=dist_attr.process_mesh,
                        ctx=self._dist_context,
                        chunk_id=dist_attr.chunk_id,
                    )
                    if comm_stream is not None:
                        depend_op.dist_attr.execution_stream = comm_stream
                        depend_op.dist_attr.scheduling_priority = (
                            self.comm_op_scheduling_priority
                        )

        # hierarchical grad comm
        if self.enable_hierarchical_comm:
            # NOTE so far we only support Isomorphic cluster with 8 ranks per node
            # TODO unify here create communicators
            # create communicators
            nranks_per_node = 8
            assert self.sharding_world_size % nranks_per_node == 0
            global_group = sharding_info.group
            global_ranks = global_group.ranks
            relative_idx_in_node = self.global_rank % nranks_per_node
            node_idx = self.global_rank // nranks_per_node
            inter_node_ranks = [
                rank
                for rank in global_ranks
                if rank % nranks_per_node == relative_idx_in_node
            ]
            _logger.info(
                "Sharding Gradient Hierarchical Communication Optimization."
            )
            _logger.info(f"current global rank idx: {self.global_rank}.")
            _logger.info(f"local inter node ranks idx: {inter_node_ranks}.")
            assert (
                len(inter_node_ranks)
                == self.sharding_world_size // nranks_per_node
            )
            intra_node_ranks = [
                rank
                for rank in global_ranks
                if rank // nranks_per_node == node_idx
            ]
            assert len(intra_node_ranks) == nranks_per_node
            _logger.info(f"local intra node ranks idx: {intra_node_ranks}.")
            inter_node_groups = []
            intra_node_groups = []
            for _ in range(self.grad_comm_stream_num):
                # TODO re-use one origin communicator
                inter_node_groups.append(
                    new_process_group(inter_node_ranks, force_new_group=True)
                )
                intra_node_groups.append(
                    new_process_group(intra_node_ranks, force_new_group=True)
                )

            # update program
            for idx, op in reversed(list(enumerate(block.ops))):
                if is_data_parallel_reduce_op(op):
                    assert (
                        op.type == "reduce"
                        and op.attr("reduce_type") == dist.ReduceOp.SUM
                    )
                    grad_comm_stream_idx = grad_comm_op_to_stream_idx[op]
                    inter_node_group = inter_node_groups[grad_comm_stream_idx]
                    intra_node_group = intra_node_groups[grad_comm_stream_idx]

                    reduce_varname = op.output("Out")[0]
                    if self.enable_overlap:
                        comm_stream = op.dist_attr.execution_stream
                    dst_rank = int(op.attr("root_id"))

                    in_peer = False
                    if dst_rank % nranks_per_node == relative_idx_in_node:
                        in_peer = True
                    intra_node_dst = dst_rank % nranks_per_node

                    op._set_attr('ring_id', intra_node_group.id)
                    op._set_attr('root_id', intra_node_dst)

                    if in_peer:
                        inter_node_dst = dst_rank // nranks_per_node
                        new_op = block._insert_op_without_sync(
                            idx + 1,
                            type='reduce',
                            inputs={"x": reduce_varname},
                            outputs={
                                "out": reduce_varname,
                            },
                            attrs={
                                'ring_id': inter_node_group.id,
                                'root_id': inter_node_dst,
                                'reduce_type': dist.ReduceOp.SUM,
                                OP_ROLE_KEY: OpRole.Backward,
                            },
                        )
                        new_op._set_attr(
                            'op_namescope', '/' + ParallelMode.DataParallel
                        )

                        if self.enable_overlap:
                            new_op.dist_attr.execution_stream = comm_stream
                            new_op.dist_attr.scheduling_priority = (
                                self.comm_op_scheduling_priority
                            )

        block._sync_with_cpp()


def _get_broadcast_first_depend_op(block):
    for op in block.ops:
        if op.type in _supported_optimizer_type:
            return op

    raise Exception("Could not find optimizer op.")


def _insert_init_and_broadcast_op(
    block,
    insert_idx,
    varname,
    local_rank,
    root_rank,
    ring_id,
    op_role,
    dist_context,
):
    """
    empty op for initialization
    """
    broadcast_var = block.var(varname)
    broadcast_var_dist_attr = dist_context.get_tensor_dist_attr_for_program(
        broadcast_var
    )

    new_op = block._insert_op_without_sync(
        insert_idx,
        type='broadcast',
        inputs={'x': varname},
        outputs={'out': varname},
        attrs={
            'ring_id': ring_id,
            'root': root_rank,
            OP_ROLE_KEY: op_role,
        },
    )
    new_op._set_attr('op_namescope', '/' + ParallelMode.DataParallel)
    naive_set_dist_op_attr_for_program_by_mesh_and_mapping(
        new_op,
        broadcast_var_dist_attr.process_mesh,
        broadcast_var_dist_attr.dims_mapping,
        dist_context,
        chunk_id=broadcast_var_dist_attr.chunk_id,
    )
    if local_rank != root_rank:
        new_op = block._insert_op_without_sync(
            insert_idx,
            type="empty",
            outputs={"Out": broadcast_var.name},
            attrs={
                "shape": broadcast_var.shape,
                "dtype": broadcast_var.dtype,
                OP_ROLE_KEY: op_role,
            },
        )
        naive_set_dist_op_attr_for_program_by_mesh_and_mapping(
            new_op,
            broadcast_var_dist_attr.process_mesh,
            broadcast_var_dist_attr.dims_mapping,
            dist_context,
            chunk_id=broadcast_var_dist_attr.chunk_id,
        )


def _insert_reduce_op(
    block,
    op_type,
    insert_idx,
    reduce_var,
    ring_id,
    root_id,
    dist_context,
    reduce_type,
    op_role=OpRole.Backward,
):
    assert (
        root_id >= 0
    ), f"root id should be a positive int, but now root id is {root_id}"
    new_op = block._insert_op_without_sync(
        insert_idx,
        type=op_type,
        inputs={'x': [reduce_var]},
        outputs={'out': [reduce_var]},
        attrs={
            'ring_id': ring_id,
            'root_id': root_id,
            'reduce_type': reduce_type,
            OP_ROLE_KEY: op_role,
        },
    )

    dist_attr = dist_context.get_tensor_dist_attr_for_program(
        block.var(reduce_var)
    )
    naive_set_dist_op_attr_for_program_by_mesh_and_mapping(
        new_op,
        dist_attr.process_mesh,
        dist_attr.dims_mapping,
        dist_context,
        chunk_id=dist_attr.chunk_id,
    )
    new_op._set_attr('op_namescope', '/' + ParallelMode.DataParallel)
    return new_op


def _get_dp_and_sharding_groups(origin_group, sharding_group_size, rank):
    dp_axis = 0
    sharding_axis = 1
    shape = [len(origin_group) // sharding_group_size, sharding_group_size]

    dp_group = _get_comm_group(origin_group, shape, dp_axis, rank)
    sharding_group = _get_comm_group(origin_group, shape, sharding_axis, rank)

    return dp_group, sharding_group


def _is_gradient_clip_op(op):
    return op.desc.has_attr("op_namescope") and op.desc.attr(
        "op_namescope"
    ).startswith("/gradient_clip")


def _is_weight_decay_op(op):
    return op.desc.has_attr("op_namescope") and op.desc.attr(
        "op_namescope"
    ).startswith("/regularization")


def _is_param_grad_fp32_cast_op(block, op):
    if not is_backward_op(op):
        return False
    if not _is_desired_cast_op(
        block, op, __amp_target_dtype__, core.VarDesc.VarType.FP32
    ):
        return False
    if _is_master_grad_cast_op(block, op):
        return False
    output_name = op.output_arg_names[0]
    base_name = output_name[: output_name.find("@")]
    if not block.has_var(base_name):
        return False
    return block.var(base_name).is_parameter


def _is_param_fp16_cast_op(block, op, params):
    if is_optimize_op(op):
        return False
    if not _is_desired_cast_op(block, op):
        return False
    input_name = op.input_arg_names[0]
    if input_name not in params:
        return False
    return True


def _is_desired_cast_op(
    block,
    op,
    src_var_type=core.VarDesc.VarType.FP32,
    dst_var_type=__amp_target_dtype__,
):
    if op.type != "cast":
        return False
    assert len(op.input_arg_names) == 1
    assert len(op.output_arg_names) == 1
    input_var = block.var(op.input_arg_names[0])
    output_var = block.var(op.output_arg_names[0])

    if input_var.dtype != src_var_type or output_var.dtype != dst_var_type:
        return False

    return True


def _get_base_name_from_grad_name(grad_name):
    base_name = None
    if ".cast_fp16@GRAD" in grad_name:
        base_name = grad_name[: grad_name.find(".cast_fp16@GRAD")]
    elif ".cast_bf16@GRAD" in grad_name:
        base_name = grad_name[: grad_name.find(".cast_bf16@GRAD")]
    elif "@GRAD" in grad_name:
        base_name = grad_name[: grad_name.find("@GRAD")]
    return base_name


def _is_param_grad_allreduce_op(op, block):
    if not is_data_parallel_reduce_op(op):
        return False

    output_name = op.output_arg_names[0]
    base_name = _get_base_name_from_grad_name(output_name)

    if not block.has_var(base_name):
        return False

    return block.var(base_name).is_parameter


def _is_param_grad_sum_op(op, block):
    if not is_backward_op(op):
        return False
    if op.type != "sum":
        return False

    output_name = op.output_arg_names[0]
    base_name = _get_base_name_from_grad_name(output_name)

    if not block.has_var(base_name):
        return False

    return block.var(base_name).is_parameter


def is_sharding_param_broadcast_op(op):
    return (
        op.type == "broadcast"
        and op.desc.has_attr("op_namescope")
        and ParallelMode.DataParallel in op.desc.attr("op_namescope")
    )


def _inference_data_parallel_group_for_operator(
    rank_id, op, dist_context, dp_axis=None
):
    dp_group = None
    for input_name in op.input_arg_names:
        # TODO(zhaoyingli): maintain a dict in dist_context to record all variables which are renamed,
        # to solve the param@RESHARD cannot be identified.
        if not is_parameter_related(input_name, op.block, dist_context):
            dist_attr = dist_context.get_op_dist_attr_for_program(op)
            process_mesh = dist_attr.process_mesh
            input_dim_mapping = dist_attr.get_input_dims_mapping(input_name)
            mesh_shape = process_mesh.shape
            # NOTE(zhaoyingli): OD-tensor's dims_mapping is empty list.
            if len(input_dim_mapping) == 0:
                continue
            # TODO(JZ-LIANG) replace with specific batch size dimension
            batch_size_axis = input_dim_mapping[0]
            if batch_size_axis > -1 and mesh_shape[batch_size_axis] > 1:
                if dp_axis is None or batch_size_axis == dp_axis:
                    group_ranks = _get_comm_group(
                        process_mesh.process_ids,
                        process_mesh.shape,
                        batch_size_axis,
                        rank_id,
                    )
                    dp_group = new_process_group(group_ranks)
                    break

    return dp_group


def partition_by_use_order(params, group_size):
    """
    shard the continuous param into same rank and divide the forward&backward computation into segment,
    which will favor the fuse pass in later.

    we assume that the params is already sorted by utilization order.
    """
    mapping = {}
    total_param_mem = 0.0
    param2mem = []
    for param in params:
        mem = get_var_size(param)
        total_param_mem += mem
        param2mem.append((param, mem))
    mapping = {x: [] for x in range(group_size)}
    cur_rank = 0
    mem_accu = 0.0
    for param, mem in param2mem:
        if mem_accu > total_param_mem * 1.0 * (cur_rank + 1) / group_size:
            cur_rank += 1
        mapping[cur_rank].append(param)
        mem_accu += mem

    return mapping


def partition_by_greedy_even(params, group_size):
    """
    use greedy algorithm to partition parameter as even as possible.
    """
    mapping = {}
    for rank_ in range(group_size):
        mapping[rank_] = []
    sizes = [0] * group_size
    for param in params:
        rank = sizes.index(min(sizes))
        mapping[rank].append(param)
        numel = reduce(lambda x, y: x * y, param.shape, 1)
        assert (
            numel > 0
        ), f"param [{param.name}] should larger than 0, but it is [{numel}]"
        sizes[rank] += numel

    return mapping


def partition_parameters(params, group_size, algor="greedy_even"):
    if algor == "greedy_even":
        rank_to_params = partition_by_greedy_even(params, group_size)
    else:
        rank_to_params = partition_by_use_order(params, group_size)

    _logger.info("Sharding Parameter Partition:")
    for k, v in rank_to_params.items():
        _logger.info(
            f"Rank:{k}, Parameter Size:{sum([get_var_size(var) for var in v])} MB."
        )
        _logger.info(f"Params in this rank: {[var.name for var in v]}.")

    return rank_to_params


def re_order_program(block, param_grads, dist_context):
    # record order
    pname_to_pg_pairs = {}
    for p, g in param_grads:
        pname_to_pg_pairs[p.name] = (p, g)

    use_order = []
    for op in block.ops:
        for input_name in op.input_arg_names:
            if (input_name in pname_to_pg_pairs) and (
                input_name not in use_order
            ):
                use_order.append(input_name)
        if len(use_order) == len(pname_to_pg_pairs):
            break

    # reorder optimizer
    last_op = block.ops[-1]
    pname_to_op = {}
    num_ops = len(block.ops)
    remove_op_indices = []
    # TODO support case when optimizer is not the last op
    if is_optimize_op(last_op) and last_op.type in _supported_optimizer_type:
        # record optimizer
        for idx, op in reversed(list(enumerate(block.ops))):
            if op.type in _supported_optimizer_type:
                assert len(op.input("Param")) == 1
                pname_to_op[op.input("Param")[0]] = op
                remove_op_indices.append(idx)
        assert len(use_order) == len(pname_to_op)

        # append new opts
        for pname in use_order:
            new_op = block.append_op(type='nop')
            new_op.desc.copy_from(pname_to_op[pname].desc)
            dist_context.set_op_dist_attr_for_program(
                new_op,
                dist_context.get_op_dist_attr_for_program(pname_to_op[pname]),
            )

        # remove old opts
        for idx in remove_op_indices:
            block._remove_op(idx, sync=False)

        block._sync_with_cpp()
        assert len(block.ops) == num_ops

    # TODO reorder gradient clip order
    _logger.info(f"Sharding the Order of param being used: {use_order}.")
    return [pname_to_pg_pairs[p] for p in use_order]


def group_param(sharding_info, fuse_size):
    """
    param are group by:
    rank id
    fuse_size
    dtype
    """
    group_to_param_map = {}
    param_to_group_map = {}
    bucket = []
    cur_group = VarGroup(fuse_size)
    for param in sharding_info.params:
        rank = sharding_info.get_var_rank(param.name)

        if cur_group.acceptable(param, rank):
            cur_group.collect(param, rank)
        else:
            cur_group = VarGroup(fuse_size)
            cur_group.collect(param, rank)

        cur_group.is_in_local_shard = sharding_info.is_in_local_shard(
            param.name
        )

        if cur_group in group_to_param_map:
            group_to_param_map[cur_group].append(param.name)
        else:
            group_to_param_map[cur_group] = [param.name]

        param_to_group_map[param.name] = cur_group

    return group_to_param_map, param_to_group_map


class ShardingInfo:
    def __init__(self, group, rank, params_grads, partition_algor):
        self.group = group
        self.params_grads = {p.name: (p, g) for p, g in params_grads}
        assert len(self.params_grads) == len(
            set(self.params_grads)
        ), "found duplicated param in params_grads"

        self.params = [p for p, _ in params_grads]
        self.param_names = [p.name for p in self.params]
        self.group_size = group.nranks
        self.global_rank = rank
        self.local_rank = group.ranks.index(self.global_rank)
        self.partition_algor = partition_algor
        # rank in below mapping are local rank in this sharding group
        self.rank_to_params = partition_parameters(
            self.params, self.group_size, self.partition_algor
        )
        # include fp32 and fp16 param
        self.param_to_rank = {}
        self._map_param_to_rank()

    def _map_param_to_rank(self):
        """
        mapping parameters to the rank which holds it.
        """
        for rank, params in self.rank_to_params.items():
            for param in params:
                self.param_to_rank[param.name] = rank

    def get_var_rank(self, varname):
        if varname in self.param_to_rank:
            return self.param_to_rank[varname]
        return -1

    # determine fp32 and fp16 (cast) param
    def is_in_local_shard(self, param_name):
        return self.get_var_rank(param_name) == self.local_rank

    # NOTE the follow logic is designed for supporting AMP O1 when
    # the param would be cast to fp16 before used for calculation.
    # and sharding should only broadcast the casted fp16 param
    # instead of the origin fp32 version param.
    def get_broadcast_vars_and_param_usage(self, block):
        broadcast_vars = set()
        fp16_params = set()
        fp16_to_fp32 = {}

        param_usage = {x: 0 for x in self.param_names}
        for op in block.ops:
            if is_optimize_op(op):
                continue
            for input_name in op.input_arg_names:
                if input_name in self.param_names:
                    param_usage[input_name] += 1

        for op in block.ops:
            if not _is_param_fp16_cast_op(block, op, self.param_names):
                continue
            input_name = op.input_arg_names[0]
            output_name = op.output_arg_names[0]
            broadcast_vars.add(output_name)
            fp16_params.add(output_name)
            fp16_to_fp32[output_name] = input_name
            param_usage[input_name] -= 1
            self.param_to_rank[output_name] = self.param_to_rank[input_name]

        for param, usage in param_usage.items():
            if usage > 0:
                broadcast_vars.add(param)
        return broadcast_vars, param_usage

    def get_param_grad(self, param_name):
        if not self.is_in_local_shard(param_name):
            raise ValueError(f"param[{param_name}] not in current rank.")
        if param_name not in self.params_grads:
            raise ValueError(f'param[{param_name}] not in params_grads')
        return self.params_grads.get(param_name, None)


class VarGroup:
    def __init__(self, max_size):
        self.max_size = max_size
        self.dtype = None
        self.rank = -1
        self.numel = 0
        self.vars = []
        self.coalesce_var = None
        self.coalesce_dep_varname = None
        self.coalesce_op_idx = None
        self.reduce_op_indices = []
        self.allreduce_op_indices = []
        self.is_in_local_shard = False

    def acceptable(self, param, rank):
        if self.numel == 0:
            return True
        else:
            if param.dtype != self.dtype:
                return False
            if rank != self.rank:
                return False
            if self.numel + get_var_numel(param) > self.max_size:
                return False
            return True

    def collect(self, param, rank):
        self.dtype = param.dtype
        self.rank = rank
        self.numel += get_var_numel(param)
        self.vars.append(param)

    def __len__(self):
        return len(self.vars)
