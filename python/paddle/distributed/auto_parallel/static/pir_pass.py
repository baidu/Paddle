# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import collections
import logging
import re
from dataclasses import dataclass

import paddle
import paddle.distributed as dist
from paddle import pir
from paddle.autograd.backward_utils import ValueDict
from paddle.base.framework import EagerParamBase, pir_op_role_guard
from paddle.base.log_helper import get_logger
from paddle.distributed.fleet.meta_optimizers.common import OpRole
from paddle.distributed.passes.pass_base import PassContext, new_pass
from paddle.distributed.passes.pass_utils import infer_chunk_id

from .mix_to_dist_pass import dist_skip_op_list
from .process_group import get_process_group
from .reshard_funcs.base_reshard_func import (
    choose_reshard_func,
    copy_dist_attr_with_new_member,
    copy_op_attr_with_new_member,
    copy_process_mesh_with_new_member,
)
from .reshard_funcs.reshard_func_register import register_reshard_funcs
from .utils import (
    _complete_op_dist_attr,
    fuse_param_func,
    get_pp_stage_by_pp_degree,
    get_pp_stage_by_process_mesh,
    get_sub_process_mesh_by_program,
    partition_skip_op_list,
    update_pylayer_output,
)

_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s'
)

register_reshard_funcs()

amp_ops = ["pd_op.check_finite_and_unscale_", "pd_op.update_loss_scaling_"]


def reshard_single_value(program, op, operand, attr):
    prev_var = operand.source()
    if prev_var.is_dist() and prev_var.dist_attr() != attr:
        operand_attr = attr.as_tensor_dist_attr()
        paddle.pir.set_insertion_point(op)
        with pir_op_role_guard(op.op_role):
            # fold reshard
            if prev_var.get_defining_op().name() == 'dist_op.reshard':
                prev_reshard = prev_var.get_defining_op()
                prev_reshard_input = prev_reshard.operand_source(0)
                prev_reshard_result = prev_reshard.result(0)
                # skil global to sub mesh reshard
                if (
                    prev_reshard_input.dist_attr().process_mesh.ndim
                    == prev_reshard_result.dist_attr().process_mesh.ndim
                ):
                    if prev_reshard_input.dist_attr() == operand_attr:
                        return prev_reshard_input
                    reshard_var = paddle._C_ops.reshard_v2(
                        prev_reshard_input, operand_attr
                    )
                    return reshard_var
            # insert reshard
            reshard_var = paddle._C_ops.reshard_v2(prev_var, operand_attr)
            return reshard_var
    return prev_var


def reshard_combine_value(program, op, operand, attr):
    prev_var = operand.source()

    assert (
        prev_var.get_defining_op().name() == 'builtin.combine'
    ), f"TensorList must be defined by builtin.combine op, but is {prev_var.get_defining_op().name()}."

    combine_op = prev_var.get_defining_op()
    array_attr = attr.as_array_attr()

    assert len(combine_op.operands()) == len(
        array_attr
    ), "The number of combine op operands and the number of dist array_attr are not equal in op"

    reshard_vars = []
    for inner_operand, inner_attr in zip(combine_op.operands(), array_attr):
        reshard_vars.append(
            reshard_single_value(program, op, inner_operand, inner_attr)
        )

    paddle.pir.set_insertion_point(op)
    with pir_op_role_guard(op.op_role):
        combine_value = paddle._C_ops.builtin_combine(reshard_vars)
    return combine_value


def apply_partition_pass(program, block=None):
    if block is None:
        block = program.global_block()
    for op in block.ops:
        for sub_block in op.blocks():
            apply_partition_pass(program, block=sub_block)

        if op.dist_attr is None:
            continue
        if op.name() in partition_skip_op_list:
            continue

        assert len(op.operands()) == len(
            op.dist_attr.operands()
        ), f"The number of operands and the number of op_dist_attr's operands are not equal in op: {op}"
        assert len(op.results()) == len(
            op.dist_attr.results()
        ), f"The number of results and the number of op_dist_attr's results are not equal in op: {op}"

        # deal with inplace value
        for out_idx, in_idx in paddle.core.pir.get_op_inplace_info(op).items():
            ref_op_role = op.op_role

            operand = op.operand(in_idx)
            operand_attr = op.dist_attr.operand(in_idx)
            prev_var = operand.source()
            if not prev_var.is_dist() or operand_attr == prev_var.dist_attr():
                continue

            assert (
                not prev_var.is_combine()
            ), f"The current partition pass not support inplace value of {op} is tensor list."

            operand_attr = operand_attr.as_tensor_dist_attr()

            # reshard input
            paddle.pir.set_insertion_point(op)
            with pir_op_role_guard(ref_op_role):
                reshard_var = paddle._C_ops.reshard_v2(prev_var, operand_attr)
                operand.set_source(reshard_var)

            result = op.result(out_idx)
            result_attr = op.dist_attr.result(out_idx).as_tensor_dist_attr()
            assert (
                operand_attr == result_attr
            ), f"For inplace value, The operend dist attr should be equal to result dist attr , please check your infer_spmd func of {op}"

            # reshard output
            paddle.pir.set_insertion_point_after(op)
            old_dist_attr = result.dist_attr()
            result.update_dist_attr(result_attr)

            with pir_op_role_guard(ref_op_role):
                prev_op = prev_var.get_defining_op()

                # reshard output to assign out input
                reshard_var_1 = paddle._C_ops.reshard_v2(
                    result, prev_var.dist_attr()
                )
                assign_out = paddle._C_ops.assign_out_(reshard_var_1, prev_var)
                assign_out.get_defining_op().dist_attr = (
                    copy_op_attr_with_new_member(
                        assign_out.get_defining_op().dist_attr,
                        new_chunk_id=op.dist_attr.chunk_id,
                    )
                )

            if old_dist_attr == result.dist_attr():
                continue

            reshard_var_2 = reshard_var_1
            if old_dist_attr != reshard_var_1.dist_attr():
                with pir_op_role_guard(ref_op_role):
                    reshard_var_2 = paddle._C_ops.reshard_v2(
                        result, old_dist_attr
                    )

            result.replace_all_uses_with(reshard_var_1)
            reshard_var_1.get_defining_op().operand(0).set_source(result)
            reshard_var_2.get_defining_op().operand(0).set_source(result)

        for operand, attr in zip(op.operands(), op.dist_attr.operands()):
            if not attr:
                continue
            prev_var = operand.source()
            if prev_var.is_combine():
                operand.set_source(
                    reshard_combine_value(program, op, operand, attr)
                )
            else:
                operand.set_source(
                    reshard_single_value(program, op, operand, attr)
                )
            prev_op = prev_var.get_defining_op()
            if prev_op and prev_op.num_results() == 1 and prev_var.use_empty():
                prev_op.erase()

        for var, attr in zip(op.results(), op.dist_attr.results()):
            if (
                attr
                and var.initialized()
                and var.is_dist()
                and var.dist_attr() != attr
            ):
                paddle.pir.set_insertion_point_after(op)
                old_dist_attr = var.dist_attr()
                var.update_dist_attr(attr.as_tensor_dist_attr())

                # insert reshard
                with pir_op_role_guard(op.op_role):
                    reshard_var = paddle._C_ops.reshard_v2(var, old_dist_attr)
                    var.replace_all_uses_with(reshard_var)
                    reshard_var.get_defining_op().operand(0).set_source(var)
                    var.get_defining_op().set_bool_attr(
                        "replace_all_uses_with_reshard_var", True
                    )


class ReshardPasses:

    @staticmethod
    def decompose_reshard_pass(dist_program):
        # split composed reshard op into atomic reshard ops, which would increase the opportunity of reshard Re-Use in following fold_reshard_pass.
        del_ops = []
        for op in dist_program.global_block().ops:
            if op.name() != 'dist_op.reshard':
                continue
            input = op.operand_source(0)
            result = op.result(0)

            # split the reshard compose p2p and collective into one p2p reshard and one collective reshard.
            # avoid global to sub mesh case
            if (
                input.dist_attr().process_mesh
                != result.dist_attr().process_mesh
            ) and input.dist_attr().process_mesh.ndim == result.dist_attr().process_mesh.ndim:
                if (
                    input.dist_attr().placements
                    != result.dist_attr().placements
                ):
                    ref_op_role = op.op_role
                    with pir_op_role_guard(ref_op_role):
                        intermediate_dist_attr = copy_dist_attr_with_new_member(
                            input.dist_attr(),
                            new_process_mesh=result.dist_attr().process_mesh,
                        )
                        intermediate_dist_type = (
                            paddle.base.libpaddle.pir.cvt_to_dist_type(
                                input.type(), intermediate_dist_attr
                            )
                        )
                        paddle.pir.set_insertion_point(op)
                        intermediate_var = paddle._C_ops.reshard_v2(
                            input, intermediate_dist_attr
                        )
                        new_reshard_result = paddle._C_ops.reshard_v2(
                            intermediate_var, result.dist_attr()
                        )
                        result.replace_all_uses_with(new_reshard_result)
                        del_ops.append(op)

        for op in del_ops:
            _logger.info(f"[Reshard Pass] atomic composed reshard op: {op!s}")
            op.erase()

    @staticmethod
    def fold_reshard_pass(dist_program):
        del_ops = []
        value_dict = ValueDict()
        for op in dist_program.global_block().ops:
            if op.name() != 'dist_op.reshard':
                continue
            input = op.operand_source(0)
            result = op.result(0)
            if input.type() == result.type():
                result.replace_all_uses_with(input)
                del_ops.append(op)
                continue
            if input not in value_dict:
                value_dict[input] = [(result.type(), result)]
                continue
            no_find = True
            for type, val in value_dict[input]:
                if type == result.type():
                    result.replace_all_uses_with(val)
                    del_ops.append(op)
                    no_find = False
                    break
            if no_find:
                value_dict[input].append((result.type(), result))
        for op in del_ops:
            op.erase()

    @staticmethod
    def reshard_op_pass(dist_program, global_params_grads=None, block=None):
        if block is None:
            block = dist_program.global_block()
        for op in block.ops:
            for sub_block in op.blocks():
                ReshardPasses.reshard_op_pass(dist_program, block=sub_block)
            if op.name() == 'dist_op.reshard':
                var = op.operand_source(0)

                op_dist_attr = op.dist_attr
                src_dist_attr = op_dist_attr.operand(0).as_tensor_dist_attr()
                dst_dist_attr = op_dist_attr.result(0).as_tensor_dist_attr()

                assert (
                    not var.initialized() or var.dist_attr() == src_dist_attr
                ), f"The dist_attr of reshard op's input and operand should be equal, but got {var.dist_attr()} and {src_dist_attr}"

                if src_dist_attr == dst_dist_attr:
                    op.result(0).replace_all_uses_with(var)
                    if global_params_grads is not None:
                        for idx, (p, g) in enumerate(global_params_grads):
                            if g is not None and g.is_same(op.result(0)):
                                global_params_grads[idx] = (p, var)
                    op.erase()
                    continue

                reshard_func = choose_reshard_func(src_dist_attr, dst_dist_attr)
                assert (
                    reshard_func is not None
                ), f'There is no reshard function that matches src_dist_attr: {src_dist_attr} and dst_dist_attr: {dst_dist_attr}, {var.get_defining_op()}'

                paddle.pir.set_insertion_point(op)
                ref_op_role = op.op_role

                with pir_op_role_guard(ref_op_role):
                    out_value = reshard_func.reshard(
                        src_dist_attr,
                        dst_dist_attr,
                        op.operand_source(0),
                        op.result(0).type(),
                    )

                if out_value is not None:
                    op.result(0).replace_all_uses_with(out_value)

                if op.result(0).use_empty():
                    if global_params_grads is not None:
                        for idx, (p, g) in enumerate(global_params_grads):
                            if g is not None and g.is_same(op.result(0)):
                                global_params_grads[idx] = (
                                    (p, out_value)
                                    if out_value is not None
                                    else (p, var)
                                )
                    op.erase()

    @staticmethod
    def apply_reshard_pass(dist_program, global_params_grads=None):
        ReshardPasses.decompose_reshard_pass(dist_program)
        ReshardPasses.fold_reshard_pass(dist_program)
        ReshardPasses.reshard_op_pass(dist_program, global_params_grads)


# Replace the specific MoE-related dist op with the
# executable op in the dense program. In expert parallelism
# of the MoE model, the process mesh of each expert is
# different. Two specific apis are used to transform the
# input tensor's global process mesh to the experts' local
# process meshes, which will add two dist ops in the program.
# The following two functions are used to replace the two
# dist ops with the executable share_data_ ops.
def replace_moe_sub_mesh_tensors(op):
    cur_rank = paddle.distributed.get_rank()
    in_value = op.operand_source(0)
    out_value = None
    out_idx = -1
    for idx, val in enumerate(op.results()):
        val_mesh = val.dist_attr().process_mesh
        if cur_rank in val_mesh.process_ids:
            assert (
                out_value is None
            ), f'{op} has more than one results on rank {cur_rank}'
            out_value = val
            out_idx = idx

    paddle.pir.set_insertion_point(op)
    local_value = paddle._C_ops.share_data_(in_value)
    local_value_type = paddle.base.libpaddle.pir.cvt_to_dist_type(
        out_value.type(), out_value.dist_attr()
    )
    local_value.set_type(local_value_type)
    out_value.replace_all_uses_with(local_value)

    op_dist_attr = op.dist_attr
    share_data_op = local_value.get_defining_op()
    share_data_op.dist_attr = (
        paddle.base.libpaddle.pir.create_op_dist_attribute(
            op_dist_attr.process_mesh,
            [op_dist_attr.operand(0).as_tensor_dist_attr()],
            [op_dist_attr.result(out_idx).as_tensor_dist_attr()],
        )
    )
    for val in op.results():
        if not val.use_empty():
            update_pylayer_output(val)
    assert all(val.use_empty() for val in op.results())
    op.erase()


def remove_sub_block_unused_inputs(op):
    inputs_size = op.operand_source.num_operands()
    inputs = [op.operand_source(i) for i in range(inputs_size)]
    # remove unused inputs


class RemovePasses:

    @staticmethod
    def remove_other_rank_op_pass(dist_program):
        # pruning op and value not belong to cur rank
        def prune_op(block):
            cur_rank = paddle.distributed.get_rank()
            reverse_block_ops = block.ops[::-1]
            skip_idx = 0
            for idx, op in enumerate(reverse_block_ops):
                if idx < skip_idx:
                    continue
                skip_idx += 1

                if op.name() == "dist_op.moe_sub_mesh_tensors":
                    replace_moe_sub_mesh_tensors(op)
                    continue
                elif op.name() == "dist_op.moe_global_mesh_tensor":
                    replace_moe_global_mesh_tensor(op)
                    continue
                elif op.name() == "cf.tuple_push":
                    stack_create_op = op.operand_source(0).get_defining_op()
                    if stack_create_op.result(2).use_empty():
                        op.erase()
                    continue
                elif op.name() == "cf.yield":
                    continue
                elif op.name() == "pd_op.pylayer":
                    # if the pylayer op is not on the current rank, we should delete it
                    is_cur_rank = False
                    for pylayer_block in list(op.blocks())[::-1]:
                        for sub_block_op in pylayer_block.ops:
                            if (
                                sub_block_op.dist_attr
                                and cur_rank
                                in sub_block_op.dist_attr.process_mesh.process_ids
                            ):
                                is_cur_rank = True
                                break
                    if not is_cur_rank:
                        op.erase()
                        continue
                    for pylayer_block in list(op.blocks())[::-1]:
                        prune_op(pylayer_block)
                    # update pylayer op's inputs
                    op.as_pylayer_op().update_input()
                    continue
                elif op.name() == "dist_op.dtensor_from_local":
                    dtensor_to_local_idx = idx
                    while (
                        reverse_block_ops[dtensor_to_local_idx].name()
                        != "dist_op.dtensor_to_local"
                    ):
                        dtensor_to_local_idx += 1
                    if (
                        op.dist_attr
                        and cur_rank
                        not in op.dist_attr.process_mesh.process_ids
                    ):
                        for i in range(idx, dtensor_to_local_idx + 1):
                            reverse_block_ops[i].erase()
                    skip_idx = dtensor_to_local_idx + 1
                    continue
                elif op.name() in partition_skip_op_list:
                    can_delete = True
                    for val in op.results():
                        if not val.use_empty():
                            can_delete = False
                    if can_delete:
                        op.erase()
                    continue

                if (
                    op.dist_attr
                    and cur_rank not in op.dist_attr.process_mesh.process_ids
                ):
                    op.erase()
                elif op.name() == "dist_op.reshard":
                    assert op.result(
                        0
                    ).use_empty(), f'There should not have useful dist.reshard op in remove_other_rank_op_pass. but find : {op}'
                    op.erase()

        prune_op(dist_program.global_block())

        # merge pd.data ops for
        lr_ops = []
        lr_parameters = []
        for op in dist_program.global_block().ops[::-1]:
            if (
                op.name() == 'pd_op.data'
                and "learning_rate" in op.attrs()["name"]
            ):
                lr_ops.append(op)
            if (
                op.name() == 'builtin.parameter'
                and "learning_rate" in op.attrs()["parameter_name"]
            ):
                lr_parameters.append(op)

        if len(lr_ops) > 1:
            lr_value = lr_ops[0].result(0)
            for op in lr_ops[1:]:
                lr = op.result(0)
                lr.replace_all_uses_with(lr_value)
                op.erase()

        if len(lr_parameters) > 1:
            lr_value = lr_parameters[0].result(0)
            for op in lr_parameters[1:]:
                lr = op.result(0)
                lr.replace_all_uses_with(lr_value)
                op.erase()
        for keyword, argument in dist_program.global_block().kwargs().items():
            if argument.use_empty():
                dist_program.global_block().erase_kwarg(keyword)

    @staticmethod
    def remove_no_need_in_startup(startup_program, main_program):
        # 1. vars used in main_program
        main_program_var_names = []
        for key in main_program.global_block().kwargs():
            main_program_var_names.append(key)
        for op in main_program.global_block().ops:
            for var in op.operands_source():
                if var.has_name:
                    main_program_var_names.append(var.name)
            for var in op.results():
                if var.has_name:
                    main_program_var_names.append(var.name)
        # 2. remove var op not used in main_program
        for op in startup_program.global_block().ops:
            for var in op.operands_source():
                if var.has_name and var.name not in main_program_var_names:
                    op.erase()

        # 3. dead code elimination
        pm = pir.PassManager()
        pm.add_pass('dead_code_elimination_pass', {})
        pm.run(startup_program)
        for op in startup_program.global_block().ops:
            if op.name() == "pd_op.coalesce_tensor_":
                if op.result(0).use_empty() and op.result(1).use_empty():
                    op.erase()
        pm.run(startup_program)

    @staticmethod
    def remove_other_rank_input_output_pass(dist_program):
        '''
        Pruning value not belong to cur rank especially used for check_finite_and_unscale
        and update_loss_scaling op in amp.

        For example, w0 on mesh0, w1 on mesh1, before pass, the ops is:
            [w0_g, w1_g], is_finite = check_finite_and_scale([w0_g, w1_g], loss_scaling)
        after pass, on mesh0, the op is:
            [w0_g], is_finite = check_finite_and_scale([w0_g], loss_scaling)

        Note that here we do not set the op_dist_attr, since it is not used afterwards.
        '''
        cur_rank = paddle.distributed.get_rank()
        for op in dist_program.global_block().ops[::-1]:
            if op.name() not in amp_ops:
                continue
            new_vars = []
            combine_op = op.operand_source(0).get_defining_op()
            for inner_operand in (
                op.operand_source(0).get_defining_op().operands()
            ):
                if (
                    cur_rank
                    in inner_operand.source()
                    .dist_attr()
                    .process_mesh.process_ids
                ):
                    new_vars.append(inner_operand.source())
                    continue
            result = op.operand_source(0).get_defining_op().result(0)
            paddle.pir.set_insertion_point_after(combine_op)
            res = paddle._C_ops.builtin_combine(new_vars)
            res.get_defining_op().op_role = op.op_role
            result.replace_all_uses_with(res)
            combine_op.erase()
            # since it is inplace op, set type of output as the same as input
            op.result(0).set_type(res.type())

    @staticmethod
    def remove_other_rank_params_grads_pass(dist_program, dist_params_grads):
        cur_rank_param = []
        cur_rank = paddle.distributed.get_rank()

        for op in dist_program.global_block().ops:
            if op.name() == 'builtin.parameter':
                if cur_rank in op.dist_attr.process_mesh.process_ids:
                    cur_rank_param.append(op.attrs()['parameter_name'])

        need_remove_idx = []
        for idx, (param, grad) in enumerate(dist_params_grads):
            if grad is None:
                continue
            if param.name not in cur_rank_param:
                need_remove_idx.append(idx)

        for idx in need_remove_idx[::-1]:
            dist_params_grads.pop(idx)

    @staticmethod
    def apply_all(
        dist_main_program, dist_startup_program, dist_params_grads=[]
    ):
        RemovePasses.remove_other_rank_input_output_pass(dist_main_program)
        RemovePasses.remove_other_rank_params_grads_pass(
            dist_main_program, dist_params_grads
        )
        _complete_op_dist_attr(dist_main_program)
        RemovePasses.remove_other_rank_op_pass(dist_main_program)
        RemovePasses.remove_no_need_in_startup(
            dist_startup_program, dist_main_program
        )


def replace_moe_global_mesh_tensor(op):
    cur_rank = paddle.distributed.get_rank()
    out_value = op.result(0)
    in_value = None
    in_idx = -1
    for idx, val in enumerate(op.operands_source()):
        val_mesh = val.dist_attr().process_mesh
        if cur_rank not in val_mesh.process_ids:
            continue
        assert (
            in_value is None
        ), f'{op} has more than one inputs on rank {cur_rank}'
        in_value = val
        in_idx = idx

    paddle.pir.set_insertion_point(op)
    local_value = paddle._C_ops.share_data_(in_value)
    # local_value = paddle.assign(in_value)
    local_value_type = paddle.base.libpaddle.pir.cvt_to_dist_type(
        out_value.type(), out_value.dist_attr()
    )
    local_value.set_type(local_value_type)
    out_value.replace_all_uses_with(local_value)

    op_dist_attr = op.dist_attr
    share_data_op = local_value.get_defining_op()
    share_data_op.dist_attr = (
        paddle.base.libpaddle.pir.create_op_dist_attribute(
            op_dist_attr.process_mesh,
            [op_dist_attr.operand(in_idx).as_tensor_dist_attr()],
            [op_dist_attr.result(0).as_tensor_dist_attr()],
        )
    )

    assert all(val.use_empty() for val in op.results())
    op.erase()


# Note: this is the pass in the dense program
comm_ops = [
    "pd_op.all_gather",
    "pd_op.reduce_scatter",
]


def remove_unuseful_comm_op_pass(program):
    for op in program.global_block().ops:
        if op.name() in comm_ops or (
            op.name() == "pd_op.all_reduce"
            and op.int_attr("reduce_type")
            in [dist.ReduceOp.SUM, dist.ReduceOp.MAX]
        ):
            ring_id = op.int_attr("ring_id")
            process_group = get_process_group(ring_id)
            if process_group.nranks == 1:
                op.result(0).replace_all_uses_with(op.operand_source(0))
                op.erase()
        if op.name() == "pd_op.share_data_":
            if op.operand_source(0).has_one_use():
                op.result(0).replace_all_uses_with(op.operand_source(0))
                op.erase()
        if (
            op.name() == "pd_op.cast"
            and op.result(0).dtype == op.operand_source(0).dtype
        ):
            op.result(0).replace_all_uses_with(op.operand_source(0))
            op.erase()


# In sequence_parallel, we need to transpose hidden_states
# from [bs, seq, hidden] to [seq, bs, hidden] to perform
# split and allgather at dim 0.
# The transpose may lead to about 3% performance
# in llama-70B model (tp4pp8).
# We found that, when bs=1, which is the common case in llm
# training, the transpose is equal to reshape.
# So, this pass is to haddle the specific case.
def eliminate_transpose_by_reshape(program):
    for op in program.global_block().ops:
        if (
            op.name() == 'pd_op.transpose'
            or op.name() == 'pd_op.transpose_grad'
        ):
            var = op.operand(0).source()
            rank = len(var.shape)
            perm = op.attrs()['perm']
            perm = [p + rank if p < 0 else p for p in perm]
            # only support transpose dim 0 and dim 1
            expected_perm = [1, 0] + [i + 2 for i in range(rank - 2)]
            if perm == expected_perm and (
                var.shape[0] == 1 or var.shape[1] == 1
            ):
                paddle.pir.set_insertion_point(op)
                transpose_var = op.result(0)
                reshape_var = paddle._C_ops.reshape(var, transpose_var.shape)
                reshape_var.get_defining_op().op_role = op.op_role
                transpose_var.replace_all_uses_with(reshape_var)
                op.erase()
    return program


def complete_op_role(main_program, op_role_scope: list):
    assert (
        len(op_role_scope) == 3 and len(op_role_scope[0]) == 2
    ), "op_role_scope should has the shape[3, 2]"
    forward_op_start = op_role_scope[0][0]
    forward_op_end = op_role_scope[0][1]

    backward_op_start = op_role_scope[1][0]
    backward_op_end = op_role_scope[1][1]

    opt_op_start = op_role_scope[2][0]
    opt_op_end = op_role_scope[2][1]

    global_op_idx = 0
    for blk in main_program.blocks:
        for op in blk.ops:
            if (
                global_op_idx >= forward_op_start
                and global_op_idx < forward_op_end
            ):
                op.op_role = 0
            elif (
                global_op_idx >= backward_op_start
                and global_op_idx < backward_op_end
            ):
                op.op_role = 1
            elif global_op_idx >= opt_op_start and global_op_idx < opt_op_end:
                op.op_role = 2
            else:
                pass
            global_op_idx += 1


def pipeline_pass(dense_main_program, dense_startup_program, pipeline_strategy):
    """
    Pipeline schedule pass for auto parallel. Enables the pipeline parallel scheduling
    strategies like FThenB, 1F1B, VPP, etc.
    """
    import os

    pass_name = pipeline_strategy.schedule_mode
    assert pass_name in [
        "FThenB",
        "1F1B",
        "VPP",
    ], f"pipeline scheduler only support FThenB, 1F1B and VPP now, but receive {pass_name}"

    pass_attr = {}
    pass_attr["num_micro_batches"] = pipeline_strategy.accumulate_steps
    pass_attr["pp_degree"] = pipeline_strategy.pp_degree
    pass_attr["pp_stage"] = get_pp_stage_by_pp_degree(
        pipeline_strategy.pp_degree
    )
    pass_attr["vpp_degree"] = pipeline_strategy.vpp_degree
    pass_attr["split_backward"] = pipeline_strategy.split_backward

    if pass_name == "1F1B":
        # TODO(Ruibiao): Move FLAGS_1f1b_backward_forward_overlap and
        # FLAGS_mp_async_allreduce_in_backward to auto parallel Strategy
        # after these two optimizations are available.
        pass_attr["enable_backward_forward_overlap"] = int(
            os.environ.get("FLAGS_1f1b_backward_forward_overlap", 0)
        )

    pipeline_pass = new_pass("pipeline_scheduler_" + pass_name, pass_attr)
    pass_context = PassContext()
    pipeline_pass.apply(
        dense_main_program,
        dense_startup_program,
        pass_context,
    )
    plan = pass_context.get_attr("plan")
    return plan


def _extract_seg_method(op, seg_method):
    regex = re.compile(seg_method, re.IGNORECASE)
    struct_name = (
        op.attrs()["struct_name"] if op.has_attr("struct_name") else "/"
    )
    m = regex.search(struct_name)
    if not m:
        return None
    return struct_name[m.start(0) :].split("/")[0]


def _get_seg_struct_names(ops, seg_method):
    fwd_start_op_index = 0
    for i, op in enumerate(ops):
        if _extract_seg_method(op, seg_method):
            fwd_start_op_index = i
            break

    total_op_num = len(ops)
    fwd_end_op_index = total_op_num - 1
    for i in reversed(range(total_op_num)):
        if _extract_seg_method(ops[i], seg_method):
            fwd_end_op_index = i
            break

    struct_names = collections.OrderedDict()
    seg_op_mesh = collections.OrderedDict()

    for i in range(fwd_start_op_index, fwd_end_op_index + 1):
        if ops[i].dist_attr is None:
            continue

        struct_name = _extract_seg_method(ops[i], seg_method)
        if struct_name:
            struct_names[struct_name] = 1
            if struct_name in seg_op_mesh:
                assert (
                    seg_op_mesh[struct_name] == ops[i].dist_attr.process_mesh
                ), "The segment's ops should have same process_mesh."

            seg_op_mesh[struct_name] = ops[i].dist_attr.process_mesh
        else:
            if ops[i].name() != "dist_op.reshard":
                raise ValueError(
                    f"The op {ops[i].name()} without seg_method in its struct_name should only be reshard"
                )

    return list(struct_names.keys())


def _analyze_use_custom_mesh(ops, seg_method, pp_degree):
    non_use_custom_mesh = True
    seg_pp_stages = [-1]

    for op in ops:
        if _extract_seg_method(op, seg_method) and op.dist_attr:
            op_mesh = op.dist_attr.process_mesh
            pp_stage = get_pp_stage_by_process_mesh(op_mesh, pp_degree)
            if pp_stage is None:
                continue

            if seg_pp_stages[-1] > pp_stage:
                non_use_custom_mesh = False
                break
            seg_pp_stages.append(pp_stage)

    if not non_use_custom_mesh:
        _logger.info("Cannot Use Auto VPP")
    else:
        _logger.info("Using Auto VPP")

    return non_use_custom_mesh


def _set_process_mesh_and_chunk_id(
    op,
    chunk_process_mesh,
    chunk_id,
    set_input_mesh=False,
    set_output_mesh=False,
):
    def set_var_origin_op_process_mesh(var_origin_op):
        var_origin_op_input_attr = var_origin_op.dist_attr.operands()
        var_origin_op_output_attr = var_origin_op.dist_attr.results()
        var_origin_op_output_attr[0] = var_origin_op_output_attr[
            0
        ].as_tensor_dist_attr()
        var_origin_op_output_attr[0] = (
            paddle.base.libpaddle.pir.create_tensor_dist_attribute(
                chunk_process_mesh,
                var_origin_op_output_attr[0].dims_mapping,
                var_origin_op_output_attr[0].partial_status,
            )
        )

        var_origin_op.dist_attr = (
            paddle.base.libpaddle.pir.create_op_dist_attribute(
                chunk_process_mesh,
                var_origin_op_input_attr,
                var_origin_op_output_attr,
                0,
            )
        )

    def get_var_process_mesh(var):
        var_process_mesh = None
        var_dist_attr = var.dist_attr()

        def get_attr_mesh(var_dist_attr):
            if var_dist_attr:
                if var_dist_attr.as_array_attr():
                    var_array_attr = var_dist_attr.as_array_attr()
                    return var_array_attr[0].as_tensor_dist_attr().process_mesh
                else:
                    return var_dist_attr.process_mesh

        if var_dist_attr:
            var_process_mesh = get_attr_mesh(var_dist_attr)
        elif var.is_combine():
            # NOTE(zhangwl): op var may is vec_type , need get var dist_attr one by one
            var_list = var.type().as_vec_type()
            var_list = var_list.as_list() if var_list is not None else var_list
            var_attr_list = []
            for combine_var in var_list:
                var_dist_attr = combine_var.as_dist_type().dist_attr()
                var_process_mesh = get_attr_mesh(var_dist_attr)
                if var_process_mesh is not None:
                    return var_process_mesh

    def get_var_attr_with_process_mesh(
        var_dist_attr, var_origin_op, process_mesh
    ):
        # Note(luchang): the var generated by builtin.combine will have mutilple dist_attr
        if var_dist_attr and var_dist_attr.as_array_attr():
            var_array_attr = var_dist_attr.as_array_attr()
            for i in range(len(var_array_attr)):
                var_dist_attr = var_array_attr[i].as_tensor_dist_attr()
                if op_mesh is not None:
                    if var_dist_attr.process_mesh == op_mesh:
                        var_array_attr[i] = copy_dist_attr_with_new_member(
                            var_dist_attr, new_process_mesh=process_mesh
                        )
                else:
                    var_array_attr[i] = copy_dist_attr_with_new_member(
                        var_dist_attr, new_process_mesh=process_mesh
                    )
            return var_array_attr
        elif var_dist_attr:
            if op_mesh is not None:
                if var_dist_attr.process_mesh == op_mesh:
                    if var_origin_op.name() in [
                        "pd_op.data",
                        "builtin.parameter",
                    ]:
                        set_var_origin_op_process_mesh(var_origin_op)
                    var_attr = copy_dist_attr_with_new_member(
                        var_dist_attr, new_process_mesh=process_mesh
                    )
                    return var_attr
            else:
                var_attr = copy_dist_attr_with_new_member(
                    var_dist_attr, new_process_mesh=process_mesh
                )
                return var_attr
        return var_dist_attr

    def set_var_process_mesh(var, process_mesh):
        var_dist_attr = var.dist_attr()
        var_origin_op = var.get_defining_op()
        if var_dist_attr:
            var_attr = get_var_attr_with_process_mesh(
                var_dist_attr, var_origin_op, process_mesh
            )
            if var_attr is not None:
                var.update_dist_attr(var_attr)
        elif var.is_combine():
            # NOTE(zhangwl): op var may is vec_type , need set var dist_attr one by one
            var_list = var.type().as_vec_type()
            var_list = var_list.as_list() if var_list is not None else var_list
            var_attr_list = []
            for combine_var in var_list:
                var_dist_attr = combine_var.as_dist_type().dist_attr()
                var_attr_list.append(
                    get_var_attr_with_process_mesh(
                        var_dist_attr, var_origin_op, process_mesh
                    )
                )
            var_array_attr = (
                paddle.base.libpaddle.pir.create_array_dist_attribute(
                    var_attr_list
                )
            )
            var.update_dist_attr(var_array_attr)

    def set_attrs_process_mesh(attrs, process_mesh):
        for idx, attr in enumerate(attrs):
            if attr.as_array_attr():
                array_attr = attr.as_array_attr()
                new_array_attr = []
                for i in range(len(array_attr)):
                    tensor_attr = array_attr[i].as_tensor_dist_attr()
                    new_array_attr.append(tensor_attr)
                    if tensor_attr and tensor_attr.process_mesh == op_mesh:
                        new_array_attr[i] = copy_dist_attr_with_new_member(
                            tensor_attr, new_process_mesh=process_mesh
                        )
                attrs[idx] = (
                    paddle.base.libpaddle.pir.create_array_dist_attribute(
                        new_array_attr
                    )
                )
            else:
                tensor_attr = attr.as_tensor_dist_attr()
                if tensor_attr and tensor_attr.process_mesh == op_mesh:
                    attrs[idx] = copy_dist_attr_with_new_member(
                        tensor_attr, new_process_mesh=process_mesh
                    )

    def set_process_mesh(vars, attrs, process_mesh):
        if vars is not None:
            for var in vars:
                set_var_process_mesh(var, process_mesh)
        if attrs is not None:
            set_attrs_process_mesh(attrs, process_mesh)

    op_input_vars = op.operands_source()
    op_output_vars = op.results()

    # NOTE(zhangwl):dist_skip_op do not have op_mesh
    op_mesh = None
    if op.name() in dist_skip_op_list:
        input_var_process_mesh = None
        # NOTE(zhangwl):dist_skip_op output_process_mesh must equal to input_process_mesh
        for var in op_input_vars:
            input_var_process_mesh = get_var_process_mesh(var)
            if input_var_process_mesh is not None:
                break
        if input_var_process_mesh is not None:
            set_process_mesh(op_output_vars, None, input_var_process_mesh)
        return

    op_dist_attr = op.dist_attr
    op_mesh = op_dist_attr.process_mesh
    op_input_attrs = op_dist_attr.operands()
    op_output_attrs = op_dist_attr.results()

    # if op in seq_chunk , vpp need set var and op chunk_process_mesh and chunk_id
    if set_input_mesh:
        set_process_mesh(op_input_vars, op_input_attrs, chunk_process_mesh)
    if set_output_mesh:
        set_process_mesh(op_output_vars, op_output_attrs, chunk_process_mesh)
    if set_input_mesh or set_output_mesh:
        op_mesh = chunk_process_mesh

    op.dist_attr = paddle.base.libpaddle.pir.create_op_dist_attribute(
        op_mesh,
        op_input_attrs,
        op_output_attrs,
        chunk_id,
    )


def complete_chunk_id(dist_program, startup_program, pipeline_strategy):
    if not pipeline_strategy.enable:
        return

    sub_process_meshes = get_sub_process_mesh_by_program(dist_program)
    pp_degree = pipeline_strategy.pp_degree
    vpp_degree = pipeline_strategy.vpp_degree
    seg_method = pipeline_strategy.vpp_seg_method
    schedule_mode = pipeline_strategy.schedule_mode
    num_chunks = pp_degree * vpp_degree

    if pp_degree < 2 and vpp_degree > 1:
        raise ValueError("VPP schedule mode only can be set in pipeline mode.")
    if vpp_degree > 1 and (not seg_method or schedule_mode != "VPP"):
        raise ValueError(
            "Please set right schedule_mode and vpp_seg_method for VPP."
        )
    if vpp_degree < 2:
        return

    ReshardPasses.fold_reshard_pass(dist_program)
    seg_struct_names = _get_seg_struct_names(
        dist_program.global_block().ops, seg_method
    )
    ops = dist_program.global_block().ops

    # Step2: analysis whether the pp_stage is non-decreasing among segments
    # 1. if non_use_custom_mesh is True, the ops' process_mesh will be changed by vpp strategy
    # 2. if non_use_custom_mesh is False, the ops's process_mesh will not be changed.
    non_use_custom_mesh = _analyze_use_custom_mesh(ops, seg_method, pp_degree)

    # Step3: Get op index boundary, pp_stage, chunk_id, struct_names of each segment
    seg_pp_stages = [i % pp_degree for i in range(num_chunks)]
    seg_chunk_ids = [i // pp_degree for i in range(num_chunks)]
    seg_parts = [0]
    last_struct_name = None
    # stage_ids[i] represents the stage number assigned to the i-th layer.
    stage_ids = []

    for idx, op in enumerate(ops):
        if len(seg_parts) == len(seg_struct_names):
            break
        struct_name = _extract_seg_method(op, seg_method)
        if op.dist_attr is not None and last_struct_name != struct_name:
            pp_stage = get_pp_stage_by_process_mesh(
                op.dist_attr.process_mesh, pp_degree
            )
            if pp_stage is not None:
                stage_ids.append(pp_stage)
            last_struct_name = struct_name
        if struct_name == seg_struct_names[len(seg_parts)]:
            seg_parts.append(idx)
    seg_parts.append(len(ops))

    pp_stage_layer_nums = [0] * pp_degree
    for i in stage_ids:
        pp_stage_layer_nums[i] = pp_stage_layer_nums[i] + 1
    assert all(
        value >= vpp_degree for value in pp_stage_layer_nums
    ), "The number of layers on each pp_stage must not be less than the vpp_degree in the pp_stage to ensure that each chunk contains at least one layer."

    seg_layer_num = [0] * num_chunks
    for pp_stage in range(
        0, pp_degree
    ):  # Each pp_stage is assigned a number of layers based on user intent.
        pp_stage_layer_num = pp_stage_layer_nums[pp_stage]
        for i in range(0, pp_stage_layer_num):
            # The pp_stage uses a Round robin scheduling algorithm to allocate layers one by one.
            virtual_chunk_id = i % vpp_degree
            real_chunk_id = (virtual_chunk_id) * pp_degree + pp_stage
            seg_layer_num[real_chunk_id] = seg_layer_num[real_chunk_id] + 1

    # Step4: Set the process_mesh of each op
    seg_id = 0
    reshard_ops = []
    previous_seg_parts_end_idx = 0
    for seg_id in range(num_chunks):
        start_idx = seg_parts[previous_seg_parts_end_idx]
        end_idx = seg_parts[previous_seg_parts_end_idx + seg_layer_num[seg_id]]
        pp_stage = seg_pp_stages[seg_id]
        chunk_id = seg_chunk_ids[seg_id]
        struct_name = ",".join(
            seg_struct_names[
                previous_seg_parts_end_idx : previous_seg_parts_end_idx
                + seg_layer_num[seg_id]
            ]
        )
        previous_seg_parts_end_idx = (
            previous_seg_parts_end_idx + seg_layer_num[seg_id]
        )
        process_mesh = sub_process_meshes[pp_stage]

        _logger.info(
            f"stage=[{pp_stage}], chunk_id=[{chunk_id}], layer_name=[{struct_name}]"
        )
        _logger.info(
            f"start op: [{ops[start_idx].name()}], end op: [{ops[end_idx - 1].name()}]"
        )

        skip_idx = 0
        for idx in range(start_idx, end_idx):
            if idx < skip_idx:
                continue

            is_seg_op = _extract_seg_method(ops[idx], seg_method) is not None
            set_mesh = non_use_custom_mesh & is_seg_op

            if ops[idx].name() == "dist_op.reshard":
                reshard_ops.append(ops[idx])
                continue
            elif ops[idx].name() == "dist_op.dtensor_to_local":
                _set_process_mesh_and_chunk_id(
                    ops[idx],
                    process_mesh,
                    chunk_id,
                    set_input_mesh=set_mesh,
                )
                dtensor_from_local_idx = idx + 1
                while (
                    ops[dtensor_from_local_idx].name()
                    != "dist_op.dtensor_from_local"
                ):
                    dtensor_from_local_idx += 1
                for local_op_idx in range(idx + 1, dtensor_from_local_idx):
                    ops[local_op_idx].set_int_attr("chunk_id", chunk_id)

                _set_process_mesh_and_chunk_id(
                    ops[dtensor_from_local_idx],
                    process_mesh,
                    chunk_id,
                    set_output_mesh=set_mesh,
                )
                skip_idx = dtensor_from_local_idx + 1
                continue

            for sub_block in ops[idx].blocks():
                # TODO(luchang): support condition block
                pass

            _set_process_mesh_and_chunk_id(
                ops[idx],
                process_mesh,
                chunk_id,
                set_input_mesh=set_mesh,
                set_output_mesh=set_mesh,
            )
            skip_idx = idx + 1

    # Step5: set right process_mesh for reshard op
    for op in reshard_ops:
        var = op.operand_source(0)

        op_dist_attr = op.dist_attr
        src_dist_attr = op_dist_attr.operand(0).as_tensor_dist_attr()
        dst_dist_attr = op_dist_attr.result(0).as_tensor_dist_attr()

        if src_dist_attr == dst_dist_attr:
            op.result(0).replace_all_uses_with(var)
            op.erase()
            continue

        reshard_func = choose_reshard_func(src_dist_attr, dst_dist_attr)
        reshard_func_name = reshard_func.__class__.__name__

        if reshard_func_name == "NdMeshReshardFunction":
            new_process_mesh = var.dist_attr().process_mesh
            new_src_dist_attr = copy_dist_attr_with_new_member(
                src_dist_attr, new_process_mesh=new_process_mesh
            )
            new_dst_dist_attr = copy_dist_attr_with_new_member(
                dst_dist_attr, new_process_mesh=new_process_mesh
            )
            op.dist_attr = copy_op_attr_with_new_member(
                op_dist_attr,
                new_operands=[new_src_dist_attr],
                new_results=[new_dst_dist_attr],
                new_process_mesh=new_process_mesh,
            )
        elif reshard_func_name == "GlobaleToSubMeshFunction":
            result_var = op.result(0)
            new_process_mesh = result_var.dist_attr().process_mesh
            new_dst_dist_attr = copy_dist_attr_with_new_member(
                dst_dist_attr, new_process_mesh=new_process_mesh
            )
            op.dist_attr = copy_op_attr_with_new_member(
                op_dist_attr, new_results=[new_dst_dist_attr]
            )
        elif reshard_func_name == "NdMeshReshardFunctionCrossMesh":
            result_var = op.result(0)
            src_process_mesh = var.dist_attr().process_mesh
            dst_process_mesh = result_var.dist_attr().process_mesh
            new_src_dist_attr = copy_dist_attr_with_new_member(
                src_dist_attr, new_process_mesh=src_process_mesh
            )
            new_dst_dist_attr = copy_dist_attr_with_new_member(
                dst_dist_attr, new_process_mesh=dst_process_mesh
            )
            new_process_ids = (
                src_process_mesh.process_ids + dst_process_mesh.process_ids
            )
            new_process_mesh = copy_process_mesh_with_new_member(
                op.dist_attr.process_mesh,
                new_process_ids=new_process_ids,
            )

            op.dist_attr = copy_op_attr_with_new_member(
                op_dist_attr,
                new_operands=[new_src_dist_attr],
                new_results=[new_dst_dist_attr],
                new_process_mesh=new_process_mesh,
            )
        elif reshard_func_name == "SameStatusReshardFunction":
            op.result(0).replace_all_uses_with(var)
            op.erase()
        else:
            raise ValueError(
                f"Unsupported reshard function: {reshard_func_name}, reshard op's dist_attr: {op.dist_attr}"
            )

    # Step6: add reshard op between pipeline chunks
    apply_partition_pass(dist_program)

    for op in startup_program.global_block().ops:
        if op.name() == "builtin.set_parameter":
            param_name = op.str_attr("parameter_name")
            startup_param = op.operand_source(0)
            param = dist_program.get_parameter_value_by_name(param_name)
            if param.dist_attr():
                startup_param.update_dist_attr(param.dist_attr())

    ReshardPasses.fold_reshard_pass(dist_program)


def check_chunk_id(dist_program):
    all_ops = dist_program.global_block().ops
    for idx, op in enumerate(all_ops):
        if op.op_role in [int(OpRole.Forward), int(OpRole.Backward)]:
            if op.name() in dist_skip_op_list:
                continue

            if op.has_attr("chunk_id"):
                # op between dtensor_from_local and dtensor_to_local will
                # be assigned a chunk_id attribute.
                continue
            elif op.name() in [
                "dist_op.dtensor_from_local",
                "dist_op.dtensor_to_local",
            ]:
                # dtensor_from_local and dtensor_to_local ops will be removed after
                # converting the program to a dense program.
                continue
            elif op.dist_attr.chunk_id == -1:
                if op.name() in ["pd_op.data", "builtin.parameter"]:
                    op.dist_attr = copy_op_attr_with_new_member(
                        op.dist_attr, new_chunk_id=0
                    )
                elif op.name() in ["pd_op.full", "pd_op.full_int_array"]:
                    all_used_ops = op.result(0).all_used_ops()
                    for used_op in all_used_ops:
                        if used_op.dist_attr.chunk_id != -1:
                            op.dist_attr = copy_op_attr_with_new_member(
                                op.dist_attr,
                                new_chunk_id=used_op.dist_attr.chunk_id,
                            )
                            break

                else:
                    op_chunk_id = infer_chunk_id(idx, all_ops)
                    op.dist_attr = copy_op_attr_with_new_member(
                        op.dist_attr, new_chunk_id=op_chunk_id
                    )

            if op.dist_attr.chunk_id == -1:
                raise ValueError(
                    f"The chunk_id of op[{op.name()}] is not set. Please check the chunk_id setting."
                )


def check_order(op_list, order):
    pointer = 0
    for item in order:
        if item == "pd_op.add":
            while (
                pointer < len(op_list)
                and op_list[pointer].name() == "pd_op.add"
            ):
                pointer += 1
        else:
            if pointer >= len(op_list) or op_list[pointer].name() != item:
                return False
            pointer += 1
    return True


def is_ffn_pattern(op_list):
    if len(op_list) != 3 and len(op_list) != 5:
        return False
    order = [
        "pd_op.matmul",
        "pd_op.add",
        "pd_op.matmul",
        "pd_op.add",
        "pd_op.swiglu",
    ]
    return check_order(op_list, order)


def is_qkv_pattern(op_list):
    if len(op_list) != 9 and len(op_list) != 12:
        return False
    order = [
        "pd_op.matmul",
        "pd_op.add",
        "pd_op.full_int_array",
        "pd_op.reshape",
        "pd_op.matmul",
        "pd_op.add",
        "pd_op.full_int_array",
        "pd_op.reshape",
        "pd_op.matmul",
        "pd_op.add",
        "pd_op.full_int_array",
        "pd_op.reshape",
    ]
    return check_order(op_list, order)


def get_param_op(program, param_name):
    all_ops = program.global_block().ops
    for i in range(len(all_ops)):
        if (
            all_ops[i].name() == "builtin.set_parameter"
            and all_ops[i].str_attr("parameter_name") == param_name
        ):
            return [all_ops[i], all_ops[i].operand_source(0).get_defining_op()]


@dataclass
class ParamMeta:
    name: str = None
    local_shape: list = None
    local_num_head: int = None
    local_head_dims: int = None


def fuse_attention_ffn_qkv_pass(
    startup_program, main_program, concrete_program, mode="all"
):
    # 0. Prepare the data structure
    pir_param_names = []
    dy_param_names = []
    for i in range(len(concrete_program.parameters[1])):
        dy_param_names.append(concrete_program.parameters[0][i].name)
        pir_param_names.append(concrete_program.parameters[1][i].name)
    fused_pattern_map = {"ffn": [], "qkv": []}
    fusion_map = {"ffn": [], "qkv": []}

    # 1. Traverse main_program, extract all ffn and qkv patterns.
    all_ops = main_program.global_block().ops
    for i in range(len(all_ops)):
        # check ffn pattern
        if mode == "all" or mode == "ffn":
            pat = all_ops[i : i + 3] if i + 3 <= len(all_ops) else all_ops[i:]
            if is_ffn_pattern(pat):
                fused_pattern_map['ffn'].append(pat)
                i = i + 3
                continue
            else:
                pat = (
                    all_ops[i : i + 5] if i + 5 <= len(all_ops) else all_ops[i:]
                )
                if is_ffn_pattern(pat):
                    fused_pattern_map['ffn'].append(pat)
                    i = i + 5
                    continue
        # check qkv pattern
        if mode == "all" or mode == "qkv":
            pat = all_ops[i : i + 9] if i + 9 <= len(all_ops) else all_ops[i:]
            if is_qkv_pattern(pat):
                fused_pattern_map['qkv'].append(pat)
                i = i + 9
                continue
            else:
                pat = (
                    all_ops[i : i + 12]
                    if i + 12 <= len(all_ops)
                    else all_ops[i:]
                )
                if is_qkv_pattern(pat):
                    fused_pattern_map['qkv'].append(pat)
                    i = i + 12
                    continue

    name2pir_param_map = {}

    # 2. Replace all ffn and qkv patterns with fusion patterns, and record the weights after replacement.
    for pat in fused_pattern_map['ffn']:
        if len(pat) == 5:
            mm_gate = pat[0]
            add_gate = pat[1]
            mm_up = pat[2]
            add_up = pat[3]
        else:
            mm_gate = pat[0]
            add_gate = None
            mm_up = pat[1]
            add_up = None

        fusion_w_name = f"fused_{mm_gate.operand_source(1).name}_{mm_up.operand_source(1).name}"
        fusion_map["ffn"].append(
            {
                fusion_w_name: [
                    ParamMeta(mm_gate.operand_source(1).name, None, None, None),
                    ParamMeta(mm_up.operand_source(1).name, None, None, None),
                ]
            }
        )

        fusion_w_dtype = mm_gate.operand_source(1).dtype
        fusion_w_shape = mm_gate.operand_source(1).shape
        fusion_w_shape[-1] += mm_up.operand_source(1).shape[-1]
        fusion_w_process_mesh = mm_gate.operand_source(1).process_mesh
        # Insert fusion parameter
        with paddle.static.program_guard(main_program, startup_program):
            fused_w = paddle.pir.core.create_parameter(
                dtype=fusion_w_dtype,
                shape=fusion_w_shape,
                name=fusion_w_name,
                process_mesh=fusion_w_process_mesh,
                placements=[
                    paddle.distributed.Replicate(),
                    paddle.distributed.Shard(1),
                ],
                initializer=paddle.nn.initializer.Constant(value=0),
            )
            name2pir_param_map[fusion_w_name] = fused_w
        if add_gate is not None and add_up is not None:
            fusion_bias_name = f"fused_{add_gate.operand_source(1).name}_{add_up.operand_source(1).name}"
            fusion_map["ffn"].append(
                {
                    fusion_bias_name: [
                        ParamMeta(
                            add_gate.operand_source(1).name, None, None, None
                        ),
                        ParamMeta(
                            add_up.operand_source(1).name, None, None, None
                        ),
                    ]
                }
            )

            fusion_bias_dtype = add_gate.operand_source(1).dtype
            fusion_bias_shape = add_gate.operand_source(1).shape
            fusion_bias_shape[-1] += add_up.operand_source(1).shape[-1]
            fusion_bias_process_mesh = add_gate.operand_source(1).process_mesh
            # Insert fusion parameter
            with paddle.static.program_guard(main_program, startup_program):
                fused_bias = paddle.pir.core.create_parameter(
                    dtype=fusion_bias_dtype,
                    shape=fusion_bias_shape,
                    name=fusion_bias_name,
                    process_mesh=fusion_bias_process_mesh,
                    placements=[
                        paddle.distributed.Replicate(),
                        paddle.distributed.Shard(0),
                    ],
                    initializer=paddle.nn.initializer.Constant(value=0),
                )
                name2pir_param_map[fusion_bias_name] = fused_bias

        # Insert dst pattern
        paddle.pir.set_insertion_point_after(pat[-1])
        fused_o = paddle.matmul(
            mm_gate.operand_source(0),
            fused_w,
            transpose_x=False,
            transpose_y=False,
        )
        fused_o.get_defining_op().copy_attrs_from(mm_gate)
        if add_gate is not None and add_up is not None:
            fused_o = paddle.add(fused_o, fused_bias)
            fused_o.get_defining_op().copy_attrs_from(add_gate)
        out = paddle.incubate.nn.functional.swiglu(fused_o)
        out.get_defining_op().copy_attrs_from(pat[-1])
        pat[-1].result(0).replace_all_uses_with(out)

    for pat in fused_pattern_map['qkv']:
        if len(pat) == 12:
            mm_q = pat[0]
            add_q = pat[1]
            reshape_q = pat[3]
            mm_k = pat[4]
            add_k = pat[5]
            reshape_k = pat[7]
            mm_v = pat[8]
            add_v = pat[9]
            reshape_v = pat[11]
        else:
            mm_q = pat[0]
            add_q = None
            reshape_q = pat[2]
            mm_k = pat[3]
            add_k = None
            reshape_k = pat[5]
            mm_v = pat[6]
            add_v = None
            reshape_v = pat[8]

        head_dim = [
            reshape_q.result(0).shape[-1],
            reshape_k.result(0).shape[-1],
            reshape_v.result(0).shape[-1],
        ]
        fusion_w_name = f"fused_{mm_q.operand_source(1).name}_{mm_k.operand_source(1).name}_{mm_v.operand_source(1).name}"
        fusion_map["qkv"].append(
            {
                fusion_w_name: [
                    ParamMeta(
                        mm_q.operand_source(1).name,
                        None,
                        None,
                        reshape_q.result(0).shape[-1],
                    ),
                    ParamMeta(
                        mm_k.operand_source(1).name,
                        None,
                        None,
                        reshape_k.result(0).shape[-1],
                    ),
                    ParamMeta(
                        mm_v.operand_source(1).name,
                        None,
                        None,
                        reshape_v.result(0).shape[-1],
                    ),
                ]
            }
        )
        fusion_w_dtype = mm_q.operand_source(1).dtype
        fusion_w_shape = mm_q.operand_source(1).shape
        fusion_w_shape[-1] += (
            mm_k.operand_source(1).shape[-1] + mm_v.operand_source(1).shape[-1]
        )
        fusion_w_process_mesh = mm_q.operand_source(1).process_mesh
        # insert fusion parameter
        with paddle.static.program_guard(main_program, startup_program):
            fused_w = paddle.pir.core.create_parameter(
                dtype=fusion_w_dtype,
                shape=fusion_w_shape,
                name=fusion_w_name,
                process_mesh=fusion_w_process_mesh,
                placements=[
                    paddle.distributed.Replicate(),
                    paddle.distributed.Shard(1),
                ],
                initializer=paddle.nn.initializer.Constant(value=0),
            )
            name2pir_param_map[fusion_w_name] = fused_w
        if add_q is not None and add_k is not None and add_v is not None:
            fusion_bias_name = f"fused_{add_q.operand_source(1).name}_{add_k.operand_source(1).name}_{add_v.operand_source(1).name}"
            fusion_map["qkv"].append(
                {
                    fusion_bias_name: [
                        ParamMeta(
                            add_q.operand_source(1).name,
                            None,
                            None,
                            reshape_q.result(0).shape[-1],
                        ),
                        ParamMeta(
                            add_k.operand_source(1).name,
                            None,
                            None,
                            reshape_k.result(0).shape[-1],
                        ),
                        ParamMeta(
                            add_v.operand_source(1).name,
                            None,
                            None,
                            reshape_v.result(0).shape[-1],
                        ),
                    ]
                }
            )
            fusion_bias_dtype = add_q.operand_source(1).dtype
            fusion_bias_shape = add_q.operand_source(1).shape
            fusion_bias_shape[-1] += (
                add_k.operand_source(1).shape[-1]
                + add_v.operand_source(1).shape[-1]
            )
            fusion_bias_process_mesh = add_q.operand_source(1).process_mesh
            # insert fusion parameter
            with paddle.static.program_guard(main_program, startup_program):
                fused_bias = paddle.pir.core.create_parameter(
                    dtype=fusion_bias_dtype,
                    shape=fusion_bias_shape,
                    name=fusion_bias_name,
                    process_mesh=fusion_bias_process_mesh,
                    placements=[
                        paddle.distributed.Replicate(),
                        paddle.distributed.Shard(0),
                    ],
                    initializer=paddle.nn.initializer.Constant(value=0),
                )
                name2pir_param_map[fusion_bias_name] = fused_bias
        # insert dst pattern
        paddle.pir.set_insertion_point_after(pat[-1])
        fused_o = paddle.matmul(
            mm_q.operand_source(0),
            fused_w,
            transpose_x=False,
            transpose_y=False,
        )
        fused_o.get_defining_op().copy_attrs_from(mm_q)
        if add_q is not None and add_k is not None and add_v is not None:
            fused_o = paddle.add(fused_o, fused_bias)
            fused_o.get_defining_op().copy_attrs_from(add_q)
        out = paddle.reshape(
            fused_o,
            shape=[
                0,
                0,
                reshape_k.result(0).shape[-2],
                int(
                    (
                        reshape_q.result(0).shape[-2]
                        / reshape_k.result(0).shape[-2]
                        + 2
                    )
                    * reshape_q.result(0).shape[-1]
                ),
            ],
        )
        out.get_defining_op().copy_attrs_from(reshape_q)
        reshape_op = out.get_defining_op()
        if reshape_op.has_attr("struct_name"):
            full_int_array_op = reshape_op.operand_source(1).get_defining_op()
            full_int_array_op.set_str_attr(
                "struct_name", reshape_op.attrs()["struct_name"]
            )
        out_q, out_k, out_v = paddle.split(
            out,
            num_or_sections=[
                int(
                    (
                        reshape_q.result(0).shape[-2]
                        / reshape_k.result(0).shape[-2]
                    )
                    * reshape_q.result(0).shape[-1]
                ),
                reshape_k.result(0).shape[-1],
                reshape_v.result(0).shape[-1],
            ],
            axis=-1,
        )
        if reshape_op.has_attr("struct_name"):
            builtin_split_op = out_q.get_defining_op()
            split_op = builtin_split_op.operand_source(0).get_defining_op()
            builtin_split_op.set_str_attr(
                "struct_name", reshape_op.attrs()["struct_name"]
            )
            split_op.set_str_attr(
                "struct_name", reshape_op.attrs()["struct_name"]
            )
            full_int_array_op = split_op.operand_source(1).get_defining_op()
            full_int_array_op.set_str_attr(
                "struct_name", reshape_op.attrs()["struct_name"]
            )
            full_op = split_op.operand_source(2).get_defining_op()
            full_op.set_str_attr(
                "struct_name", reshape_op.attrs()["struct_name"]
            )
        if reshape_q.result(0).shape[-2] != reshape_k.result(0).shape[-2]:
            out_q = paddle.reshape(
                out_q,
                shape=[
                    0,
                    0,
                    reshape_q.result(0).shape[-2],
                    reshape_q.result(0).shape[-1],
                ],
            )
            if builtin_split_op.has_attr("struct_name"):
                reshape_op = out_q.get_defining_op()
                reshape_op.set_str_attr(
                    "struct_name", builtin_split_op.attrs()["struct_name"]
                )
                full_int_array_op = reshape_op.operand_source(
                    1
                ).get_defining_op()
                full_int_array_op.set_str_attr(
                    "struct_name", builtin_split_op.attrs()["struct_name"]
                )

        reshape_q.result(0).replace_all_uses_with(out_q)
        reshape_k.result(0).replace_all_uses_with(out_k)
        reshape_v.result(0).replace_all_uses_with(out_v)

    # 3. Delete src pattern from origin program.
    del_ops = []
    for pat in fused_pattern_map['ffn']:
        for op in reversed(pat):
            del_ops.append(op)
            if op.name() == "pd_op.matmul" or op.name() == "pd_op.add":
                del_ops.append(op.operand_source(1).get_defining_op())
                del_ops.extend(
                    get_param_op(startup_program, op.operand_source(1).name)
                )
    for pat in fused_pattern_map['qkv']:
        for op in reversed(pat):
            del_ops.append(op)
            if op.name() == "pd_op.matmul" or op.name() == "pd_op.add":
                del_ops.append(op.operand_source(1).get_defining_op())
                del_ops.extend(
                    get_param_op(startup_program, op.operand_source(1).name)
                )
    for op in del_ops:
        op.erase()

    # 4. Initialize fused parameters and delete original parameters.
    concated_dy_param_index = []
    # for key, pat_list in fused_name_map.items():
    for key, pat_list in fusion_map.items():
        for pat in pat_list:
            for pir_param, dy_param_list in pat.items():
                # Retrieve the params of ffn and qkv patterns from concrete_program for fusion.
                concated_dy_param_list = []
                for dy_param in dy_param_list:
                    param_index = dy_param_names.index(dy_param.name)
                    concated_dy_param_list.append(
                        concrete_program.parameters[0][param_index]
                    )
                    dy_param.local_shape = (
                        concrete_program.parameters[0][param_index]
                        ._local_value()
                        .shape
                    )
                    if dy_param.local_head_dims is not None:
                        dy_param.local_num_head = (
                            dy_param.local_shape[-1] // dy_param.local_head_dims
                        )
                    concated_dy_param_index.append(param_index)

                dy_param_init = True
                for p in concated_dy_param_list:
                    if not p._local_value()._is_initialized():
                        dy_param_init = False
                        break

                # Fuse params and init pir program fusion params.
                with paddle.base.dygraph.guard():

                    dyparam_dtype = concated_dy_param_list[0].dtype
                    for param in concated_dy_param_list:
                        assert (
                            dyparam_dtype == param.dtype
                        ), "The dtypes of dy parameters to be fused are not the same."

                    dtensor = paddle.zeros(
                        shape=name2pir_param_map[pir_param].shape,
                        dtype=dyparam_dtype,
                    )
                    fused_dy_param = EagerParamBase.from_tensor(dtensor)
                    fused_dy_param = dist.shard_tensor(
                        fused_dy_param,
                        concated_dy_param_list[0].process_mesh,
                        concated_dy_param_list[0].placements,
                    )
                    fused_dy_param.name = pir_param

                    if dy_param_init:
                        if len(dy_param_list) == 3:
                            is_qkv = True
                            num_heads = dy_param_list[0].local_num_head
                            num_key_value_heads = dy_param_list[
                                1
                            ].local_num_head
                        else:
                            is_qkv = False
                            num_heads = None
                            num_key_value_heads = None
                        concated_param = fuse_param_func(
                            [
                                obj._local_value()
                                for obj in concated_dy_param_list
                            ],
                            is_qkv=is_qkv,
                            num_heads=num_heads,
                            num_key_value_heads=num_key_value_heads,
                        )
                        paddle.assign(
                            concated_param, fused_dy_param._local_value()
                        )
                        concated_param._clear()

                    # Pop and release original params from concrete_program
                    for param in concated_dy_param_list:
                        param.get_tensor()._clear()

                concrete_program.parameters[0].append(fused_dy_param)
                concrete_program.parameters[1].append(
                    name2pir_param_map[pir_param]
                )

    concated_dy_param_index.sort(reverse=True)
    for index in concated_dy_param_index:
        concrete_program.parameters[0].pop(index)
        concrete_program.parameters[1].pop(index)

    return fusion_map
