# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from __future__ import annotations

import copy
import logging
from collections import OrderedDict
from typing import TYPE_CHECKING

import paddle
from paddle.distributed.auto_parallel.static.utils import (
    is_backward_op,
    is_gradient_clip_op,
    is_optimize_op,
    naive_set_dist_op_attr_for_program_by_mesh_and_mapping,
    set_var_dist_attr,
)
from paddle.distributed.fleet.meta_optimizers.common import (
    OP_ROLE_KEY,
    OpRole,
)
from paddle.framework import core
from paddle.static import program_guard

from ..utils.log_utils import get_logger
from .pass_base import PassBase, register_pass

if TYPE_CHECKING:
    from paddle.base import Variable

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

logger = get_logger(logging.INFO, "MasterGradPass")


def _is_master_grad_cast_op(block, op):
    op_name = op.type
    if op_name != "cast":
        return False
    input_names = op.input_arg_names
    output_names = op.output_arg_names

    assert len(input_names) == 1
    assert len(output_names) == 1

    input_var_name = input_names[0]

    return (
        "@master_grad_fp16" in input_var_name
        or "@master_grad_bf16" in input_var_name
    )


def get_output_in_varlist(op, var_names) -> list[str]:
    grad_names = []
    for output_name in op.output_arg_names:
        if output_name in var_names:
            grad_names.append(output_name)
    return grad_names


@register_pass("auto_parallel_master_grad_pass")
class MasterGradPass(PassBase):
    """
    Use the high precision gradient to replace the low precision gradient in optimizer to avoid inf/nan values of low precision.
    The high precision gradient 'master grad' will be used by communication operator, `update_loss_scaling`, `GradClip` and `optimizer`.
    """

    def __init__(self):
        super().__init__()

    def _check_self(self):
        if self.get_attr("dist_context") is None:
            return False
        dist_context = self.get_attr("dist_context")
        if dist_context._serial_optimizer._grad_clip is None:
            logger.warning(
                "grad_clip not found. now only support master grad with grad clip."
            )
            return False
        return True

    def _check_conflict(self, other_pass):
        return True

    def _apply_single_impl(self, main_program, startup_program, context):
        self._completer = self.get_attr("completer")
        dist_context = self.get_attr("dist_context")
        params_grads = self.get_attr("params_grads")
        logger.debug(f"Origin main_program: {main_program}")
        self._add_master_grad(main_program, params_grads, dist_context)
        self._regenerate_optimizer(
            main_program, startup_program, params_grads, dist_context
        )
        logger.debug(f"After main program: {main_program}")

    def _add_cast_op(self, cur_block, grad_names: list[str], dist_context):
        grad_first_ids = OrderedDict()
        for idx, op in enumerate(cur_block.ops):
            if is_optimize_op(op):
                break
            elif is_backward_op(op):
                var_names = get_output_in_varlist(op, grad_names)
                for var_name in var_names:
                    if var_name not in grad_first_ids:
                        grad_first_ids[var_name] = idx
                    # Communication operators such as 'allreduce_sum' use input var as output.
                    else:
                        pass

        # insert cast op
        for grad_name, idx in reversed(grad_first_ids.items()):
            grad_var = cur_block.var(grad_name)
            if (
                grad_var.dtype == paddle.float16
                or grad_var.dtype == paddle.bfloat16
            ):
                is_fp16 = grad_var.dtype == paddle.float16
                producer_op = cur_block.ops[idx]
                producer_op_dist_attr = (
                    dist_context.get_op_dist_attr_for_program(producer_op)
                )
                assert (
                    producer_op_dist_attr is not None
                ), f"The op: '{producer_op}' should be distributed"
                ref_output_dist_attr = (
                    producer_op_dist_attr.get_output_dist_attr(grad_name)
                )
                assert (
                    ref_output_dist_attr is not None
                ), f"The output: '{grad_name}' should be distributed"
                ref_mesh = ref_output_dist_attr.process_mesh
                ref_dims_mapping = ref_output_dist_attr.dims_mapping
                ref_chunk_id = producer_op_dist_attr.chunk_id
                grad_half_precision_name = (
                    grad_name + '@master_grad_fp16'
                    if is_fp16
                    else grad_name + '@master_grad_bf16'
                )
                grad_half_precision = cur_block.create_var(
                    name=grad_half_precision_name,
                    dtype=grad_var.dtype,
                    shape=grad_var.shape,
                    persistable=False,
                    stop_gradient=False,
                )
                set_var_dist_attr(
                    dist_context,
                    grad_half_precision,
                    ref_dims_mapping,
                    ref_mesh,
                    chunk_id=ref_chunk_id,
                )

                producer_op_dist_attr = (
                    dist_context.get_op_dist_attr_for_program(producer_op)
                )
                origin_out_dims_mapping = (
                    producer_op_dist_attr.get_output_dims_mapping(grad_name)
                )
                producer_op._rename_output(grad_name, grad_half_precision.name)
                producer_op_dist_attr.set_output_dims_mapping(
                    grad_half_precision.name, origin_out_dims_mapping
                )
                grad_var.desc.set_dtype(core.VarDesc.VarType.FP32)

                cast_op = cur_block._insert_op_without_sync(
                    idx + 1,
                    type="cast",
                    inputs={"X": grad_half_precision},
                    outputs={"Out": grad_var},
                    attrs={
                        "in_dtype": grad_half_precision.dtype,
                        "out_dtype": grad_var.dtype,
                    },
                )
                cast_op._set_attr(OP_ROLE_KEY, OpRole.Backward)
                naive_set_dist_op_attr_for_program_by_mesh_and_mapping(
                    cast_op,
                    ref_mesh,
                    ref_dims_mapping,
                    dist_context,
                    chunk_id=ref_chunk_id,
                )
        cur_block._sync_with_cpp()

    def _regenerate_optimizer(
        self,
        main_program,
        startup_program,
        params_grads: list[tuple[Variable, Variable]],
        dist_context,
    ):
        grad_names = [g.name for _, g in params_grads]
        # 1. delete the origin optimizer op
        # 1.1 delete the var and op associated with the optimizer op in main_program
        main_ops = main_program.global_block().ops
        main_ops_len = len(main_ops)
        first_optimize_idx = main_ops_len
        for idx, op in enumerate(main_ops):
            # We don't delete the operators for check_nan_inf
            if is_optimize_op(op) and first_optimize_idx >= main_ops_len:
                first_optimize_idx = idx
            if is_optimize_op(op) and is_gradient_clip_op(op):
                first_optimize_idx = idx
                break
        assert (
            first_optimize_idx < main_ops_len
        ), "The first optimizer op is not found!"
        deleted_temp_var_names = []
        deleted_persist_var_names = []
        reserved_var_names = []
        for idx in range(main_ops_len - 1, first_optimize_idx - 1, -1):
            op = main_ops[idx]
            inout_arg_names = op.input_arg_names + op.output_arg_names
            if op.type in _supported_optimizer_type:
                param_names = op.input("Param")
                skip_update_names = op.input("SkipUpdate")
                for reserved_name in param_names + skip_update_names:
                    if reserved_name not in reserved_var_names:
                        reserved_var_names.append(reserved_name)
            for input_name in inout_arg_names:
                if input_name in grad_names:
                    continue
                var = main_program.global_block().var(input_name)
                if (
                    var.persistable
                    and input_name not in deleted_persist_var_names
                ):
                    deleted_persist_var_names.append(input_name)
                elif (
                    not var.persistable
                    and input_name not in deleted_temp_var_names
                ):
                    deleted_temp_var_names.append(input_name)
            main_program.global_block()._remove_op(idx)

        for var_name in deleted_temp_var_names + deleted_persist_var_names:
            if var_name not in reserved_var_names:
                main_program.global_block()._remove_var(var_name)
        main_program.global_block()._sync_with_cpp()

        # 1.2 delete the var and op in startup_program
        for reserved_name in reserved_var_names:
            if reserved_name in deleted_persist_var_names:
                deleted_persist_var_names.remove(reserved_name)
        startup_global_block = startup_program.global_block()
        for var_name in deleted_persist_var_names:
            if startup_global_block.has_var(var_name):
                startup_global_block._remove_var(var_name)
        for idx, op in reversed(list(enumerate(startup_global_block.ops))):
            inout_arg_names = op.input_arg_names + op.output_arg_names
            for var_name in inout_arg_names:
                if var_name in deleted_persist_var_names:
                    startup_program.global_block()._remove_op(idx)
                    break

        # 2. re-generate new optimizer op
        serial_optimizer = copy.deepcopy(dist_context._serial_optimizer)
        serial_optimizer._learning_rate = (
            dist_context._serial_optimizer._learning_rate
        )
        serial_optimizer._sorted = False
        with program_guard(main_program, startup_program):
            with main_program.switch_name_generator_guard("opt_"):
                _ = serial_optimizer.apply_gradients(params_grads)
        self._completer.complete_update_annotation(main_program)

    def _add_master_grad(self, main_program, params_grads, dist_context):
        grad_names = [g.name for _, g in params_grads]
        for sub_block in main_program.blocks:
            self._add_cast_op(sub_block, grad_names, dist_context)
