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

import logging
import warnings
from typing import TYPE_CHECKING

import paddle.pir
from paddle.autograd.backward_utils import (
    State,
    ValueDict,
    ValueSet,
    _as_list,
    all_input_stop_gradient_true,
    all_output_grad_none,
    all_stop_gradient_true,
    argument_to_value,
    check_type,
    dynamic_shape_prim_vjp_guard,
    get_grad_semantic_info,
    get_real_op_inputs,
    get_real_op_outputs,
    get_split_op,
    inverse_sort_op,
    is_builtin_op,
    is_control_flow,
    is_inplace_net,
    op_has_vjp,
    parent_total_ops,
    remove_op,
    remove_useless_full_like_ops,
    return_map_value,
    return_map_value_list,
    some_in_set,
    update_if_output_stopgradient,
    update_no_grad_set_by_stopgradient,
    update_while_output_stopgradient,
    warning_once,
    while_prune_check,
)
from paddle.base.libpaddle.pir import (
    build_pipe_for_block,
    get_used_external_value,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from paddle.pir import Value

"""
    grad: for template test, will combine in paddle.grad .
    calc_gradient: for internal use, optest, parallel etc .
    calc_gradient_helper: for dygraph to static .
"""
__all__ = ['grad', 'calc_gradient', 'calc_gradient_helper']


def append_full_like(float_value, copy_value, value, state, backward_ops):
    with paddle.amp.auto_cast(enable=False):
        if paddle.pir.is_fake_value(value):
            state.value_to_valuegrad[value] = [[paddle.pir.fake_value()]]
            return
        if copy_value.is_dense_tensor_array_type():
            value_grad = paddle._C_ops.create_array_like(
                copy_value,
                float_value,
            )
            full_like_op = value_grad.get_defining_op()
            backward_ops_ = [full_like_op]
        elif copy_value.is_combine():
            raise ValueError(
                "This kind of scene, where VectorType grad be fulled with zeros should not occur."
            )
            # values = paddle._C_ops.builtin_split(copy_value)
            # value_grad = []
            # backward_ops_ = []
            # backward_ops_.append(values[0].get_defining_op())
            # for v in values:
            #     grad = paddle.full_like(
            #         v,
            #         float_value,
            #         dtype=v.dtype,
            #     )
            #     value_grad.append(grad)
            #     full_like_op = grad.get_defining_op()
            #     full_op = full_like_op.operand_source(1).get_defining_op()
            #     backward_ops_.append(full_like_op)
            #     backward_ops_.append(full_op)
        else:
            value_grad = paddle.full_like(
                copy_value,
                float_value,
                dtype=copy_value.dtype,
            )
            full_like_op = value_grad.get_defining_op()
            full_op = full_like_op.operand_source(1).get_defining_op()
            backward_ops_ = [full_like_op, full_op]
        update_bwdop_structure(
            backward_ops,
            state.op_to_opgrad[value.get_defining_op()],
            backward_ops_,
        )
        if copy_value.is_combine():
            state.value_to_valuegrad[value] = [value_grad]
        else:
            state.value_to_valuegrad[value] = [[value_grad]]
        return value_grad


def append_add_n(
    op, value, state, backward_ops, bwd_value_to_block_argument_map
):
    _MAX_ADD_NUM_ = paddle.framework._global_flags()[
        'FLAGS_max_inplace_grad_add'
    ]
    with paddle.amp.auto_cast(enable=False):
        # value is input of more than one fwd_op,
        # so more than one bwd_op create input_grad,
        # need add sum op to accumulate gradient
        add_n_list = []
        for item in state.value_to_valuegrad[value]:
            if item[0] is not None:
                add_n_list.append(
                    return_map_value(item[0], bwd_value_to_block_argument_map)
                )

        from paddle.amp.auto_cast import amp_global_state

        cast_op = []
        if (
            amp_global_state().use_master_grad
            and amp_global_state().amp_dtype in ['float16', 'bfloat6']
            and op.name() == 'builtin.parameter'
            and value.dtype in [paddle.float16, paddle.bfloat16]
        ):
            add_n_list = [paddle.cast(v, 'float32') for v in add_n_list]
            cast_op = [v.get_defining_op() for v in add_n_list]

        if len(add_n_list) == 0:
            for tmp in state.value_to_valuegrad[value]:
                state.value_to_sumvaluegrad[value].append(tmp)
            state.value_to_valuegrad[value] = []
        elif len(add_n_list) <= _MAX_ADD_NUM_:
            grad_value = add_n_list[0]
            index = 1
            grad_op_list = []
            while index < len(add_n_list):
                grad_value = paddle._C_ops.add_(grad_value, add_n_list[index])
                grad_op_list.append(grad_value.get_defining_op())
                index += 1
            update_bwdop_structure(
                backward_ops, state.op_to_opgrad[op], cast_op + grad_op_list
            )
            for tmp in state.value_to_valuegrad[value]:
                state.value_to_sumvaluegrad[value].append(tmp)
            state.value_to_valuegrad[value] = [[grad_value]]
        elif len(add_n_list) == 1:
            update_bwdop_structure(
                backward_ops,
                state.op_to_opgrad[op],
                cast_op,
            )
            for tmp in state.value_to_valuegrad[value]:
                state.value_to_sumvaluegrad[value].append(tmp)
            state.value_to_valuegrad[value] = [add_n_list]
        else:
            if value.is_dense_tensor_array_type():
                add_n_value = paddle._C_ops.add_n_array(add_n_list)
            else:
                add_n_value = paddle.add_n(add_n_list)

            add_n_op = add_n_value.get_defining_op()
            combine_op = add_n_op.operand_source(0).get_defining_op()
            update_bwdop_structure(
                backward_ops,
                state.op_to_opgrad[op],
                [*cast_op, combine_op, add_n_op],
            )

            for tmp in state.value_to_valuegrad[value]:
                state.value_to_sumvaluegrad[value].append(tmp)
            state.value_to_valuegrad[value] = [[add_n_value]]


def update_bwdop_structure(backward_ops, op_to_opgrad_list, grad_op_list):
    for grad_op in grad_op_list:
        backward_ops.append(grad_op)
        op_to_opgrad_list.append(grad_op)


def prepare_grad_outputs(grad_outputs, outputs, state):
    """
    if grad_outputs is none, add fill_1 op to create grad_outputs,
    else check whether outputs shape and dtype is same to grad_outputs, otherwise raise error.

    if only part of op's outputs in outputs, add fill_0 op to create other grad_outputs.
    eg: split.

    update value_to_valuegrad and op_to_opgrad.

    return complete_outputs, backward_ops.

    """
    if not grad_outputs:
        grad_outputs = [None] * len(outputs)

    if len(grad_outputs) != len(outputs):
        raise ValueError(
            "grad_outputs should have the same length of as outputs."
        )
    backward_ops = []
    for i, grad in enumerate(grad_outputs):
        output = outputs[i]
        # fwd : op1 -> op2 -> op3 -> output
        # bwd : op1G <- op2G <- op3G <- outputG <- full_likeop/feedop
        if grad is None:
            grad_value = append_full_like(
                1.0, output, output, state, backward_ops
            )
            grad_outputs[i] = grad_value
        else:
            if output.shape != grad.shape:
                raise ValueError(
                    f"The shape of grad_output[{i}] {grad.shape} should be the same as the shape of output[{i}] {output.shape}"
                )
            if output.dtype != grad.dtype:
                warnings.warn(
                    f"The dtype of grad_output[{i}] {grad.dtype} is not same as the dtype of output[{i}] {output.dtype}"
                )
            feedop = grad.get_defining_op()
            update_bwdop_structure(
                backward_ops,
                state.op_to_opgrad[output.get_defining_op()],
                [feedop],
            )
            state.value_to_valuegrad[output] = [[grad]]

    # add input for bwd first op
    complete_outputs = outputs

    visited_output = ValueSet()
    for output in outputs:
        if output in visited_output:
            continue
        for opresult in output.get_defining_op().results():
            if opresult.is_combine():
                continue
            if opresult in state.value_to_valuegrad:
                visited_output.add(opresult)
                continue
            else:
                if paddle.pir.is_fake_value(opresult):
                    state.value_to_valuegrad[opresult] = [
                        [paddle.pir.fake_value()]
                    ]
                else:
                    grad_value = append_full_like(
                        0.0, opresult, opresult, state, backward_ops
                    )
                    visited_output.add(opresult)

                    complete_outputs.append(opresult)

    return grad_outputs, complete_outputs, backward_ops


def prune_ops(total_ops, inputs_set, outputs_set, no_grad_set):
    '''
    prune ops which do not in the path from inputs_set to outputs_set,
    prune ops which do not in the path from outputs_set to inputs_set,

    pruned op in total_ops is uneffective_ops, else is effective_ops

    '''
    intersection_op_flags = [True] * len(total_ops)
    union_op_flags = [False] * len(total_ops)

    # from input to output
    if inputs_set:
        for i, op in enumerate(total_ops):
            if some_in_set(get_real_op_outputs(op), inputs_set):
                union_op_flags[i] = True
                continue

            if some_in_set(get_real_op_inputs(op), inputs_set):
                union_op_flags[i] = True
                for value in get_real_op_outputs(op):
                    if value not in no_grad_set:
                        inputs_set.add(value)
            else:
                intersection_op_flags[i] = False

    # from output to input
    for i, op in reversed(list(enumerate(total_ops))):
        if some_in_set(get_real_op_outputs(op), outputs_set):
            union_op_flags[i] = True
            for operand in get_real_op_inputs(op):
                if operand not in no_grad_set:
                    outputs_set.add(operand)
        else:
            union_op_flags[i] = False
            intersection_op_flags[i] = False

    # some inputs in no_grad_set but its next op is effective,
    # add their defining op here.
    total_ops_list = list(total_ops)
    for i, op in enumerate(total_ops_list):
        if union_op_flags[i] is False:
            for result in op.results():
                if result.has_one_use():
                    next_op = result.first_use().owner()
                    if (
                        next_op in total_ops
                        and union_op_flags[total_ops_list.index(next_op)]
                        is True
                    ):
                        union_op_flags[i] = True
                else:
                    continue

    effective_ops = [
        total_ops[i] for i in range(len(total_ops)) if intersection_op_flags[i]
    ]
    uneffective_ops = [
        total_ops[i] for i in range(len(total_ops)) if not union_op_flags[i]
    ]

    return effective_ops, uneffective_ops


def append_backward_ops(
    base_op,
    base_inputs,
    base_input_grads,
    fwd_block,
    bwd_block,
    effective_forward_ops,
    no_grad_set,
    backward_ops,
    state,
    bwd_value_to_block_argument_map=ValueDict(),
    control_flow_value_to_copyvalue_map=ValueDict(),
):
    '''
    add grad_op in order of topological inverse sort
        eg:
        from :op1 -> v1 -> op2 -> v2 -> op3 -> v3
        to: og1_g <- v1_g <- op2_g <- v2_g <- op3_g <- v3_g

    if op has grad_op, prepare its grad_op's inputs by value_to_valuegrad,
        eg:
        value_to_valuegrad[v3] = [[v3_g]];
        v2_g = call_vjp(op3, [[v2]], [[v3]],[[v3_g]], [[v2_stopgradient]])


    special pattern:
        v11 -> combine_op -> v1 -> op -> v3
        v12 ->
                             v2 ->
        value_to_valuegrad[v3] = [[v3_g]]

        v1 is inside python api, we don't describe it in backward process(state)
        so v1_grad is inside vjp, we don't describe it in backward process(state)
        [[v11_g, v12_g], v2_g] = call_vjp(op, [[v11, v12]], [[v3]],[[v3_g]], [[v11_stopgradient, v12_stopgradient], v2_stop_gradient])


        op_vjp is:
        v11_g <- split_op <- v1_g <- op_g <- v3_g
        v12_g <-
                             v2_g <-

        value_to_valuegrad[v11] = [[v11_g]]
        value_to_valuegrad[v12] = [[v12_g]]
        value_to_valuegrad[v2] = [[v2_g]]

    if op don't has grad_op:
        if it don't has input and it's output has more than
        one output_grad, add sumop for grad aggregation.
        (eg: full op and parameter op etc.)

        else continue to next op.
    '''

    def make_output_with_output_grad(op):
        zero_flag = [False] * op.num_results()
        outputs = []
        output_grads = []
        if op.name() == "pd_op.array_write_":
            output_list = [op.operand_source(0)]
        elif op.name() == "pd_op.assign_out_":
            output_list = [op.operand_source(1)]
        else:
            output_list = op.results()

        for i, value in enumerate(output_list):
            new_value = [
                return_map_value(value, control_flow_value_to_copyvalue_map)
            ]

            value = return_map_value(
                value, state.inside_value_to_outside_value_map
            )

            if (
                value in state.value_to_valuegrad
                and len(state.value_to_valuegrad[value]) > 1
            ):
                append_add_n(
                    op,
                    value,
                    state,
                    backward_ops,
                    bwd_value_to_block_argument_map,
                )

            if (
                value not in state.value_to_valuegrad
                or state.value_to_valuegrad[value] == []
            ):
                if not value.use_empty() and get_split_op(value) is not None:
                    # pattern case:
                    # this fwd_op's output is vectorType, it will split to
                    # Type by builtin_split op, so need get from split op's outputs.
                    (
                        split_zero_flag,
                        split_outputs,
                        split_output_grad,
                    ) = make_output_with_output_grad(get_split_op(value))
                    zero_flag[i] = all(split_zero_flag)
                    grad_values = [value[0] for value in split_output_grad]
                    state.value_to_valuegrad[value] = [grad_values]
                    new_value = [info[0] for info in split_outputs]
                else:
                    # first case:
                    # this fwd_op's output didn't used by other fwd_op,
                    # so no output_grad created.

                    # second case:
                    # last bwd_op return None because input in no_grad_set,
                    # but this bwd_op need a input.

                    append_full_like(
                        0.0, new_value[0], value, state, backward_ops
                    )
                    zero_flag[i] = True

            outputs.append(new_value)
            grad_value = state.value_to_valuegrad[value][0]
            if grad_value[0] is None:
                zero_flag[i] = True
            output_grads.append(
                return_map_value_list(
                    grad_value, bwd_value_to_block_argument_map
                )
            )

        if op.name() == "pd_op.array_read":
            value = op.operand_source(0)
            value = return_map_value(
                value, state.inside_value_to_outside_value_map
            )

            if value in state.value_to_valuegrad:
                if len(state.value_to_valuegrad[value]) > 1:
                    append_add_n(
                        op,
                        value,
                        state,
                        backward_ops,
                        bwd_value_to_block_argument_map,
                    )

            if (
                value not in state.value_to_valuegrad
                or state.value_to_valuegrad[value] == []
            ):
                append_full_like(
                    0.0,
                    return_map_value(
                        value, control_flow_value_to_copyvalue_map
                    ),
                    value,
                    state,
                    backward_ops,
                )

            grad_value = state.value_to_valuegrad[value][0]
            output_grads.append(
                return_map_value_list(
                    grad_value, bwd_value_to_block_argument_map
                )
            )

        return zero_flag, outputs, output_grads

    def make_input_with_input_stopgradient(op):
        inputs = []
        input_grad_stopgradients = []
        for input, grad_semantic in zip(
            get_real_op_inputs(op), get_grad_semantic_info(op)
        ):
            if not grad_semantic:
                if (
                    op.name() not in ["cf.tuple_push", "pd_op.if"]
                    and input.get_defining_op() is not None
                    and input.get_defining_op().name() == "builtin.combine"
                ):
                    tmp_input = []
                    for tmp in input.get_defining_op().operands_source():
                        tmp_input.append(
                            return_map_value(
                                tmp, control_flow_value_to_copyvalue_map
                            )
                        )

                    inputs.append(tmp_input)
                else:
                    tmp_input = [
                        return_map_value(
                            input, control_flow_value_to_copyvalue_map
                        )
                    ]
                    inputs.append(tmp_input)
                continue

            if (
                op.name() != "cf.tuple_push"
                and input.get_defining_op() is not None
                and input.get_defining_op().name() == "builtin.combine"
            ):
                (
                    combine_inputs,
                    combine_stop_gradient,
                ) = make_input_with_input_stopgradient(input.get_defining_op())
                inputs.append([info[0] for info in combine_inputs])
                input_grad_stopgradients.append(
                    [info[0] for info in combine_stop_gradient]
                )
            else:
                tmp_input = [
                    return_map_value(input, control_flow_value_to_copyvalue_map)
                ]
                inputs.append(tmp_input)

                if input in no_grad_set or input.stop_gradient is True:
                    input_grad_stopgradients.append([True])
                else:
                    input_grad_stopgradients.append([False])

        return inputs, input_grad_stopgradients

    def update_input_grad_map(op, input_grads, all_inputs):
        i = 0
        for input, grad_semantic in zip(all_inputs, get_grad_semantic_info(op)):
            if not grad_semantic:
                continue

            if (
                op.name() != "cf.tuple_push"
                and input.get_defining_op() is not None
                and input.get_defining_op().name() == "builtin.combine"
            ):
                update_input_grad_map(
                    input.get_defining_op(),
                    input_grads[i],
                    input.get_defining_op().operands_source(),
                )
            else:
                input_grad = input_grads[i]
                if isinstance(input_grad, list):
                    state.value_to_valuegrad[input].append(input_grad)
                else:
                    state.value_to_valuegrad[input].append([input_grad])
            i += 1

    def append_yield(
        block,
        base_op,
        base_grad_op,
        base_inputs,
        base_inputs_grad,
    ):
        (
            fwd_block_argument_to_value_map,
            fwd_value_to_block_argument_map,
        ) = argument_to_value(base_op)

        with block:
            inputs_grad = []
            if base_op.name() == "pd_op.while":
                new_cond = paddle.base.libpaddle.pir.cf_has_elements(base_op)
                inputs_grad.append(new_cond)
                # while use block_arg to create grad_op
                for idx in range(len(base_inputs[: base_op.num_operands()])):
                    operands = base_inputs[idx]
                    operands = return_map_value(
                        operands, fwd_value_to_block_argument_map
                    )
                    base_inputs[idx] = operands

            for value, value_grad in zip(base_inputs, base_inputs_grad):
                if value_grad is None:
                    continue

                value = return_map_value(
                    value, state.inside_value_to_outside_value_map
                )

                if value in state.value_to_valuegrad:
                    if len(state.value_to_valuegrad[value]) > 1:
                        append_add_n(
                            base_op,
                            value,
                            state,
                            backward_ops,
                            bwd_value_to_block_argument_map,
                        )
                else:
                    new_value = return_map_value(
                        value, control_flow_value_to_copyvalue_map
                    )
                    append_full_like(0.0, new_value, value, state, backward_ops)

                input_grad = return_map_value(
                    state.value_to_valuegrad[value][0][0],
                    bwd_value_to_block_argument_map,
                )

                inputs_grad.append(input_grad)

            paddle.base.libpaddle.pir.cf_yield(inputs_grad)

    # there are four patterns:
    # [builtin.combine , op1] (op1's one input is vectorType, outputs are not vectorType)
    # [op2 , builtin.split] (op2's inputs are not vectorType, one output is vectorType)
    # [builtin.combine , op3 , builtin.split] (op3's one input and one output are vectorType)
    # [op4] (op4's inputs and outputs are not vectorType)

    if (
        len(effective_forward_ops) > 1
        and effective_forward_ops[-1].name() == "cf.yield"
    ):
        yield_op = effective_forward_ops[-1]
        if base_op.name() == "pd_op.while":
            # while op yield [cond, loop_vars],
            # but outputs only has loop_vars.
            inside_outputs = yield_op.operands_source()[1:]
        else:
            inside_outputs = yield_op.operands_source()

        for outside_output, inside_output in zip(
            base_op.results(), inside_outputs
        ):
            state.inside_value_to_outside_value_map[inside_output] = (
                outside_output
            )
        forward_ops = effective_forward_ops[:-1]
    else:
        forward_ops = effective_forward_ops

    if is_inplace_net(forward_ops):
        inverse_effective_forward_ops = reversed(forward_ops)
    else:
        inverse_effective_forward_ops = inverse_sort_op(forward_ops)

    clear_effective_forward_ops = []
    for op in inverse_effective_forward_ops:
        if op.name() != "builtin.combine" and op.name() != "builtin.split":
            clear_effective_forward_ops.append(op)
    with bwd_block:
        while_tuple_ops = []
        for op in clear_effective_forward_ops:
            if op_has_vjp(op):
                # prepare output_grad
                zero_flag, outputs, output_grads = make_output_with_output_grad(
                    op
                )

                # prepare input_grad stop_gradient info.
                (
                    inputs,
                    input_grad_stopgradients,
                ) = make_input_with_input_stopgradient(op)

                if op.name() == "cf.tuple_push":
                    stackop = op.operand_source(0).get_defining_op()
                    if stackop.result(2).use_empty():
                        with dynamic_shape_prim_vjp_guard(op, inputs):
                            copy_out = paddle.framework.core.call_vjp(
                                op,
                                inputs,
                                outputs,
                                output_grads,
                                input_grad_stopgradients,
                            )

                        pop_op = bwd_block.ops[-1]
                        while_tuple_ops.append(pop_op)
                        while_tuple_ops.append(op)
                        while_tuple_ops.append(stackop)
                        bwd_ops = [pop_op]
                        for output, copy_output in zip(
                            inputs[1:], copy_out[1:]
                        ):
                            control_flow_value_to_copyvalue_map[output[0]] = (
                                copy_output[0]
                            )
                    else:
                        bwd_ops = [stackop.result(2).first_use().owner()]
                else:
                    # all(zero_flag) support this op has no contribution for grad
                    # should be delete (prune sub_graph)
                    if (
                        len(output_grads) == 0
                        or all(zero_flag)
                        or all_output_grad_none(output_grads)
                    ) and op.name() not in [
                        "pd_op.while",
                        "pd_op.if",
                        "pd_op.increment_",
                    ]:
                        continue

                    if all_input_stop_gradient_true(
                        input_grad_stopgradients
                    ) and op.name() not in [
                        "pd_op.array_read",
                        "pd_op.array_write_",
                        "pd_op.increment_",
                    ]:
                        continue
                    if op.name() == "pd_op.if":
                        origin_inputs = get_real_op_inputs(op)
                        for sub_block in op.blocks():
                            build_pipe_for_block(sub_block)
                        with dynamic_shape_prim_vjp_guard(op, inputs):
                            input_grads = paddle.framework.core.call_vjp(
                                op,
                                inputs,
                                outputs,
                                output_grads,
                                input_grad_stopgradients,
                            )
                        grad_op = bwd_block.ops[-1]
                        bwd_ops = [grad_op]

                        inputs_used_by_other_op = []
                        for sub_fwd_block, sub_bwd_block in zip(
                            op.blocks(), grad_op.blocks()
                        ):
                            sub_state = state.copy(sub_fwd_block)
                            for input_ in origin_inputs:
                                if input_ in state.value_to_valuegrad:
                                    origin_grad = state.value_to_valuegrad[
                                        input_
                                    ].copy()
                                    inputs_used_by_other_op.append(
                                        (input_, origin_grad)
                                    )

                            sub_backward_ops = []
                            sub_control_flow_value_to_copyvalue_map = (
                                control_flow_value_to_copyvalue_map.copy()
                            )
                            append_backward_ops(
                                op,
                                [input[0] for input in inputs[1:]],
                                [input_grad[0] for input_grad in input_grads],
                                sub_fwd_block,
                                sub_bwd_block,
                                sub_fwd_block.ops,
                                no_grad_set,
                                sub_backward_ops,
                                sub_state,
                                control_flow_value_to_copyvalue_map=sub_control_flow_value_to_copyvalue_map,
                            )
                            for input_tuple in inputs_used_by_other_op:
                                state.value_to_valuegrad[input_tuple[0]] = (
                                    input_tuple[1]
                                )

                        update_if_output_stopgradient(
                            grad_op,
                            grad_op.as_if_op().true_block().ops[-1],
                            grad_op.as_if_op().false_block().ops[-1],
                        )
                        for input_tuple in inputs_used_by_other_op:
                            state.value_to_valuegrad[input_tuple[0]] = []
                        # update input_grad map
                        update_input_grad_map(op, input_grads, origin_inputs)
                    elif op.name() == "pd_op.while":
                        origin_inputs = get_real_op_inputs(op)
                        # prepare while[cond, loop_vars, other_input] other_input's grad
                        while_block = op.as_while_op().body()
                        sub_state = state.copy(while_block)
                        for i, input in enumerate(
                            get_used_external_value(while_block)
                        ):
                            if input in sub_state.value_to_valuegrad:
                                if len(sub_state.value_to_valuegrad[input]) > 1:
                                    append_add_n(
                                        op,
                                        input,
                                        state,
                                        backward_ops,
                                        bwd_value_to_block_argument_map,
                                    )

                            if (
                                input not in sub_state.value_to_valuegrad
                                or sub_state.value_to_valuegrad[input] == []
                            ):
                                append_full_like(
                                    0.0, input, input, sub_state, backward_ops
                                )

                            grad_value = sub_state.value_to_valuegrad[input][0]
                            for tmp in state.value_to_valuegrad[input]:
                                state.value_to_sumvaluegrad[input].append(tmp)
                            state.value_to_valuegrad[input] = []
                            output_grads.append(
                                return_map_value_list(
                                    grad_value,
                                    bwd_value_to_block_argument_map,
                                )
                            )
                        build_pipe_for_block(while_block)
                        with dynamic_shape_prim_vjp_guard(op, inputs):
                            input_grads = paddle.framework.core.call_vjp(
                                op,
                                inputs,
                                outputs,
                                output_grads,
                                input_grad_stopgradients,
                            )
                        grad_op = bwd_block.ops[-1]
                        bwd_ops = [grad_op]

                        # update grad_op structure
                        (
                            _,
                            sub_bwd_value_to_block_argument_map,
                        ) = argument_to_value(grad_op)
                        sub_bwd_value_to_block_argument_map.update(
                            bwd_value_to_block_argument_map
                        )
                        sub_control_flow_value_to_copyvalue_map = (
                            control_flow_value_to_copyvalue_map.copy()
                        )

                        while_grad_block = grad_op.as_while_op().body()
                        sub_backward_ops = []
                        append_backward_ops(
                            op,
                            [input[0] for input in inputs],
                            [input_grad[0] for input_grad in input_grads],
                            while_block,
                            while_grad_block,
                            while_block.ops,
                            no_grad_set,
                            sub_backward_ops,
                            sub_state,
                            sub_bwd_value_to_block_argument_map,
                            sub_control_flow_value_to_copyvalue_map,
                        )

                        update_while_output_stopgradient(
                            grad_op, while_grad_block.ops[-1]
                        )
                        # update input_grad map
                        update_input_grad_map(op, input_grads, origin_inputs)
                    elif op.name() == "pd_op.pylayer":
                        # create grad_op
                        before_ops_num = len(bwd_block.ops)

                        with dynamic_shape_prim_vjp_guard(op, inputs):
                            input_grads = paddle.framework.core.call_vjp(
                                op,
                                inputs,
                                outputs,
                                output_grads,
                                input_grad_stopgradients,
                            )
                        after_ops_num = len(bwd_block.ops)

                        # update grad_op structure
                        bwd_ops = bwd_block.ops[before_ops_num:after_ops_num]

                        # update input_grad map
                        update_input_grad_map(
                            op, input_grads, get_real_op_inputs(op)
                        )
                    else:
                        # create grad_op

                        before_ops_num = len(bwd_block.ops)
                        with dynamic_shape_prim_vjp_guard(op, inputs):
                            input_grads = paddle.framework.core.call_vjp(
                                op,
                                inputs,
                                outputs,
                                output_grads,
                                input_grad_stopgradients,
                            )
                        after_ops_num = len(bwd_block.ops)

                        # update grad_op structure
                        bwd_ops = bwd_block.ops[before_ops_num:after_ops_num]

                        # update input_grad map
                        update_input_grad_map(
                            op, input_grads, op.operands_source()
                        )

                update_bwdop_structure(
                    backward_ops, state.op_to_opgrad[op], bwd_ops
                )

            else:
                if (
                    op.num_operands() == 0
                    and op.num_results() != 0
                    or op.name() == "pd_op.full_like"
                ):
                    for value in op.results():
                        if len(state.value_to_valuegrad[value]) > 1:
                            append_add_n(
                                op,
                                value,
                                state,
                                backward_ops,
                                bwd_value_to_block_argument_map,
                            )
                        else:
                            state.op_to_opgrad[op] = []
                else:
                    all_results_stop_gradient = True
                    for value in op.results():
                        if not value.stop_gradient:
                            all_results_stop_gradient = False
                    if (
                        not is_builtin_op(op)
                        and not paddle.core.is_forward_only(op)
                        and not all_results_stop_gradient
                    ):
                        raise ValueError(
                            f"op '{op.name()}' has no grad op, consider enable prim to decompose it."
                        )
                    state.op_to_opgrad[op] = []

        if fwd_block != bwd_block:
            if while_prune_check(while_tuple_ops):
                remove_op(bwd_block, while_tuple_ops[0], state)
                while_tuple_ops[1].get_parent_block().remove_op(
                    while_tuple_ops[1]
                )
                while_tuple_ops[2].get_parent_block().remove_op(
                    while_tuple_ops[2]
                )

            append_yield(
                bwd_block,
                base_op,
                bwd_block.parent_op,
                base_inputs,
                base_input_grads,
            )


def prepare_backward_prune_set(inputs, outputs):
    outputs_fwd_set = ValueSet()
    for input_ in inputs:
        if not input_.use_empty():
            for used_op in input_.all_used_ops():
                for item in get_real_op_inputs(used_op):
                    outputs_fwd_set.add(item)
        else:
            warning_once("input provided by inputs has no use")

    inputs_fwd_set = ValueSet()
    for output in outputs:
        inputs_fwd_set.add(output)

    return outputs_fwd_set, inputs_fwd_set


def create_backward_prune_set(
    outputs_fwd_set, inputs_fwd_set, no_grad_set, state
):
    outputs_set = ValueSet()
    for item in outputs_fwd_set:
        if state.value_to_valuegrad[item] != []:
            outputs_set.add(state.value_to_valuegrad[item][0][0])

    inputs_set = ValueSet()
    for item in inputs_fwd_set:
        if state.value_to_valuegrad[item] != []:
            inputs_set.add(state.value_to_valuegrad[item][0][0])

    inputs_set_tmp = ValueSet()
    for out_grad in inputs_set:
        if not out_grad.use_empty():
            for item in get_real_op_inputs(out_grad.first_use().owner()):
                inputs_set_tmp.add(item)
    inputs_set.update(inputs_set_tmp)

    no_gradvar_set = ValueSet()  # grad_value of value in no_grad_set
    for key in state.value_to_valuegrad:
        if key in no_grad_set and state.value_to_valuegrad[key] != []:
            no_gradvar_set.add(state.value_to_valuegrad[key][0][0])
    for key in state.value_to_sumvaluegrad:
        if key in no_grad_set:
            for item in state.value_to_sumvaluegrad[key][0]:
                no_gradvar_set.add(item)

    return outputs_set, inputs_set, no_gradvar_set


def _complete_grad_op_chunk_id(block, state):
    dist_skip_op_list = ["builtin.split", "builtin.combine"]

    def infer_dist_skip_op_chunk_id(op):
        if op.name() == "builtin.split":
            op_chunk_id = (
                op.operand_source(0).get_defining_op().dist_attr.chunk_id
            )
        elif op.name() == "builtin.combine":
            op_chunk_id = op.result(0).get_defining_op().dist_attr.chunk_id
        else:
            # TODO(luchang): need to support more ops such as pd_op.pylayer and so on
            op_chunk_id = -1
        return op_chunk_id

    # TODO(Ruibiao): Reorganize these unclear codes about chunk_id
    def get_op_chunk_id(op):
        if op.has_attr("chunk_id"):
            op_chunk_id = op.chunk_id
        elif op.dist_attr is None:
            op_chunk_id = -1
            if op.name() in dist_skip_op_list:
                op_chunk_id = infer_dist_skip_op_chunk_id(op)
        else:
            op_chunk_id = op.dist_attr.chunk_id
            if op_chunk_id == -1 and op.name() == "dist_op.reshard":
                prev_op = op.operand_source(0).get_defining_op()
                if prev_op.name() in dist_skip_op_list:
                    op_chunk_id = infer_dist_skip_op_chunk_id(prev_op)
                else:
                    op_chunk_id = prev_op.dist_attr.chunk_id
        return op_chunk_id

    is_dist_program = False
    for op in block.ops:
        if op.dist_attr is not None:
            is_dist_program = True
            break

    if not is_dist_program:
        return

    for op in reversed(block.ops):
        if op not in state.op_to_opgrad:
            continue

        fwd_op_chunk_id = get_op_chunk_id(op)

        for bwd_op in state.op_to_opgrad[op]:
            if op.has_attr("chunk_id"):
                bwd_op.set_int_attr("chunk_id", op.chunk_id)
                continue
            elif bwd_op.dist_attr is None:
                continue

            if bwd_op.name() in ["pd_op.add_", "pd_op.add_n_"]:
                bwd_op_chunk_id = -1
                for operand_idx in range(bwd_op.num_operands()):
                    prev_op_chunk_id = get_op_chunk_id(
                        bwd_op.operand_source(operand_idx).get_defining_op()
                    )
                    if bwd_op_chunk_id == -1:
                        bwd_op_chunk_id = prev_op_chunk_id
                    else:
                        assert bwd_op_chunk_id == prev_op_chunk_id, (
                            f"Inconsistent prev_op chunk id with {bwd_op_chunk_id} != {prev_op_chunk_id}\n"
                            "{bwd_op.operand_source(operand_idx-1).get_defining_op()}\n"
                            "{bwd_op.operand_source(operand_idx).get_defining_op()}"
                        )
            else:
                bwd_op_chunk_id = fwd_op_chunk_id

            bwd_op.dist_attr = (
                paddle.base.libpaddle.pir.create_op_dist_attribute(
                    bwd_op.dist_attr.process_mesh,
                    bwd_op.dist_attr.operands(),
                    bwd_op.dist_attr.results(),
                    bwd_op_chunk_id,
                )
            )


def calc_gradient_helper(
    outputs: Value | Sequence[Value],
    inputs: Value | Sequence[Value],
    grad_outputs: Value | Sequence[Value | None] | None = None,
    no_grad_set: set[Value] | None = None,
) -> ValueDict:
    block = paddle.base.libpaddle.pir.get_current_insertion_point().block()
    state = State(block)
    if all_stop_gradient_true(block):
        logging.warning(
            "all op in block stop_gradient is True, no grad will be calculate"
        )
        return state.value_to_valuegrad

    total_ops = parent_total_ops(block)

    # update no_grad_set if some value stop_gradient=True
    update_no_grad_set_by_stopgradient(block, no_grad_set)
    with block:
        (
            complete_grad_outputs,
            complete_outputs,
            backward_ops,
        ) = prepare_grad_outputs(grad_outputs, outputs, state)

    inputs_set = ValueSet(inputs)
    stop_gradient_false_outputs = []
    for output in complete_outputs:
        if output not in no_grad_set:
            stop_gradient_false_outputs.append(output)
    outputs_set = ValueSet(stop_gradient_false_outputs)

    if is_inplace_net(total_ops):
        effective_forward_ops = total_ops
    else:
        effective_forward_ops, _ = prune_ops(
            total_ops, inputs_set, outputs_set, no_grad_set
        )

    outputs_fwd_set, inputs_fwd_set = prepare_backward_prune_set(
        inputs, complete_outputs
    )

    append_backward_ops(
        None,
        None,
        None,
        block,
        block,
        effective_forward_ops,
        no_grad_set,
        backward_ops,
        state,
        ValueDict(),
    )

    # now value_to_valuegrad should be value <-> value (add sum op for the same values's grad value)
    outputs_set, inputs_set, no_gradvar_set = create_backward_prune_set(
        outputs_fwd_set, inputs_fwd_set, no_grad_set, state
    )

    # set chunk id for grad ops
    _complete_grad_op_chunk_id(block, state)

    remove_ops = []
    if not is_inplace_net(backward_ops) and inputs:
        _, remove_ops = prune_ops(
            backward_ops, inputs_set, outputs_set, no_gradvar_set
        )
    state.turn_map()
    remove_ops = set(remove_ops)
    for op in inverse_sort_op(list(backward_ops)):
        if op.name() == 'pd_op.full_like':
            if op.result(0).use_empty():
                remove_ops.add(op)
                remove_ops.add(op.operand_source(1).get_defining_op())
        elif is_control_flow(op):
            for sub_block in op.blocks():
                remove_useless_full_like_ops(sub_block, sub_block.ops, state)

    for bwd_op in inverse_sort_op(remove_ops):
        if bwd_op.result(0) in ValueSet(complete_grad_outputs):
            continue
        if bwd_op.result(0).use_empty():
            remove_op(block, bwd_op, state)
    state.turn_map()

    input_grad_map = state.value_to_valuegrad

    return input_grad_map


def calc_gradient(
    outputs: Value | Sequence[Value],
    inputs: Value | Sequence[Value],
    grad_outputs: Value | Sequence[Value | None] | None = None,
    no_grad_set: set[Value] | None = None,
) -> list[Value | None]:
    """
    calculate gradient of input

    Args:
        outputs (Value|list(Value)|tuple(Value)): the output Value or
            Value list/tuple of the graph to compute gradients.
        inputs (Value|list(Value)|tuple(Value)): the input Value or
            Value list/tuple of the graph to compute gradients. The returned
            values of this API are the gradients of `inputs` .
        grad_outputs (Value|list(Value|None)|tuple(Value|None), optional):
            initial gradient values of `outputs` . If `grad_outputs` is None,
            the initial gradient values of `outputs` would be Values filled with 1;
            if `grad_outputs` is not None, it must have the same length as `outputs` ,
            and in this case, the initial gradient value of the i-th `outputs` would
            be: (1) a Value filled with 1 when the i-th element of `grad_outputs`
            is None; (2) the i-th element of `grad_outputs` when the i-th element of
            `grad_outputs` is a Value. Default None.
        no_grad_set (set(Value), optional):
            the Values whose gradients are not needed to compute. Default None.

    Return:
        list[Value]:A list of gradients for inputs
        If an input does not affect targets, the corresponding gradient Tensor
        will be None
        TODO if allow_unused=False raise TypeError() if input_grad has None
    """
    # record input value and its gradient (Value to Value)
    input_to_inputgrad_map = calc_gradient_helper(
        outputs,
        inputs,
        grad_outputs=grad_outputs,
        no_grad_set=ValueSet(no_grad_set),
    )

    inputgrad = []
    for input in inputs:
        inputgrad.append(
            input_to_inputgrad_map[input][0][0]
            if input_to_inputgrad_map[input] != []
            else None
        )
    return inputgrad


def grad(
    outputs: Value | Sequence[Value],
    inputs: Value | Sequence[Value],
    grad_outputs: Value | Sequence[Value | None] | None = None,
    retain_graph: bool | None = None,
    create_graph: bool | None = False,
    only_inputs: bool | None = True,
    allow_unused: bool | None = False,
    no_grad_vars: Value | Sequence[Value] | set[Value] | None = None,
) -> list[Value | None]:
    '''
    .. note::
        **This API is ONLY available in imperative mode.**

    This API computes the sum of gradients of `outputs` with respect to each `inputs` .

    Parameters:
        outputs (Value|list(Value)|tuple(Value)): the output Value or
            Value list/tuple of the graph to compute gradients.
        inputs (Value|list(Value)|tuple(Value)): the input Value or
            Value list/tuple of the graph to compute gradients. The returned
            values of this API are the gradients of `inputs` .
        grad_outputs (Value|list(Value|None)|tuple(Value|None), optional):
            initial gradient values of `outputs` . If `grad_outputs` is None,
            the initial gradient values of `outputs` would be Values filled with 1;
            if `grad_outputs` is not None, it must have the same length as `outputs` ,
            and in this case, the initial gradient value of the i-th `outputs` would
            be: (1) a Value filled with 1 when the i-th element of `grad_outputs`
            is None; (2) the i-th element of `grad_outputs` when the i-th element of
            `grad_outputs` is a Value. Default None.
        retain_graph (bool, optional): whether to retain the forward graph which
            is used to calculate the gradient. When it is True, the graph would
            be retained, in which way users can calculate backward twice for the
            same graph. When it is False, the graph would be freed. Default None,
            which means it is equal to `create_graph` .
        create_graph (bool, optional): whether to create the gradient graphs of
            the computing process. When it is True, higher order derivatives are
            supported to compute; when it is False, the gradient graphs of the
            computing process would be discarded. Default False.
        only_inputs (bool, optional): whether to only compute the gradients of
            `inputs` . If it is False, the gradients of all remaining leaf
            Values in the graph would be also computed and accumulated.
            If it is True, only the gradients of `inputs` would be computed.
            Default True. only_inputs=False is under development, and it is
            not supported yet.
        allow_unused (bool, optional): whether to raise error or return None if some
            Values of `inputs` are unreachable in the graph. If some Values of
            `inputs` are unreachable in the graph (i.e., their gradients are None),
            error would be raised if allow_unused=False, or None would be returned as
            their gradients if allow_unused=True. Default False.
        no_grad_vars (Value|list(Value)|tuple(Value)|set(Value), optional):
            the Values whose gradients are not needed to compute. Default None.

    Returns:
        list: a list of Values, whose length is the same as the Value number
        inside `inputs`, and the i-th returned Value is the sum of gradients of
        `outputs` with respect to the i-th `inputs`.
    '''
    check_type(
        outputs,
        'outputs',
        (paddle.pir.Value, list, tuple),
        'paddle.autograd.ir_backward.grad',
    )
    check_type(
        inputs,
        'inputs',
        (paddle.pir.Value, list, tuple),
        'paddle.autograd.ir_backward.grad',
    )
    check_type(
        grad_outputs,
        'grad_outputs',
        (paddle.pir.Value, list, tuple, type(None)),
        'paddle.autograd.ir_backward.grad',
    )

    check_type(
        no_grad_vars,
        'no_grad_vars',
        (
            paddle.pir.Value,
            list,
            tuple,
            set,
            ValueSet,
            type(None),
        ),
        'paddle.autograd.ir_backward.grad',
    )
    outputs = _as_list(outputs)
    inputs = _as_list(inputs)
    grad_outputs = _as_list(grad_outputs)
    if no_grad_vars is None:
        no_grad_set = ValueSet()
    else:
        no_grad_set = ValueSet(no_grad_vars)

    input_grad = calc_gradient(outputs, inputs, grad_outputs, no_grad_set)

    return input_grad


def append_backward(loss, parameter_list=None, no_grad_set=None):
    '''
    Parameters:
        loss (Value): The loss Tensor of the network
        parameter_list (Value|list(Value)|tuple(Value)):  List/Tuple of Parameters
            that need to be updated by optimizers.
            If it is None, all parameters
            will be updated.
            Default: None.
        no_grad_vars (Value|list(Value)|tuple(Value)|set(Value), optional):
            the Values whose gradients are not needed to compute. Default None.

    Returns:
        list of tuple (Value): Pairs of parameter and its corresponding gradients.
        The key is the parameter and the value is gradient Tensor.
    '''

    check_type(
        loss,
        'loss',
        paddle.pir.Value,
        'paddle.autograd.ir_backward.append_backward',
    )

    if parameter_list is not None:
        check_type(
            parameter_list,
            'parameter_list',
            (list, tuple, set),
            'paddle.autograd.ir_backward.append_backward',
        )
        for i, param in enumerate(parameter_list):
            check_type(
                param,
                f'parameter_list[{i}]',
                paddle.pir.Value,
                'base.backward.append_backward',
            )

    else:
        ops = loss.get_defining_op().get_parent_block().ops
        parameter_list = []
        for op in ops:
            if not op.has_attr("persistable"):
                continue
            persist_value = [
                result for result in op.results() if result.persistable
            ]
            parameter_list.extend(persist_value)

    if no_grad_set is None:
        no_grad_set_ = ValueSet()
    else:
        no_grad_set_ = ValueSet(no_grad_set)

    input_to_inputgrad_map = calc_gradient_helper(
        _as_list(loss),
        [],
        grad_outputs=[],
        no_grad_set=ValueSet(no_grad_set_),
    )

    input_inputs_grad = []
    for input in parameter_list:
        input_inputs_grad.append(
            (
                input,
                (
                    input_to_inputgrad_map[input][0][0]
                    if input_to_inputgrad_map[input] != []
                    else None
                ),
            )
        )

    return input_inputs_grad
