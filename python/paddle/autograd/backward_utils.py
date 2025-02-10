# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,tes
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import collections
import logging
import warnings
from collections.abc import Sequence
from functools import lru_cache
from typing import Any

from paddle import pir
from paddle.base import core
from paddle.base.libpaddle.pir import (
    get_used_external_value,
)
from paddle.base.wrapped_decorator import signature_safe_contextmanager

# TODO(CZ): to be removed when we support dynamic shape by default.
ALLOW_DYNAMIC_SHAPE_VJP_OPS = [
    "pd_op.abs",
    "pd_op.add",
    "pd_op.amax",
    "pd_op.amin",
    "pd_op.angle",
    "pd_op.argsort",
    "pd_op.assign",
    "pd_op.batch_norm_",
    "pd_op.cast",
    "pd_op.ceil",
    "pd_op.concat",
    "pd_op.cos",
    "pd_op.cumprod",
    "pd_op.cumsum",
    "pd_op.divide",
    "pd_op.dot",
    "pd_op.dropout",
    "pd_op.elementwise_pow",
    "pd_op.erf",
    "pd_op.exp",
    "pd_op.expand",
    "pd_op.floor",
    "pd_op.fmax",
    "pd_op.fmin",
    "pd_op.gather",
    "pd_op.gather_nd",
    "pd_op.gelu",
    "pd_op.group_norm",
    "pd_op.hardsigmoid",
    "pd_op.hardswish",
    "pd_op.kron",
    "pd_op.kthvalue",
    "pd_op.layer_norm",
    "pd_op.leaky_relu",
    "pd_op.log",
    "pd_op.logcumsumexp",
    "pd_op.logsumexp",
    "pd_op.matmul",
    "pd_op.max",
    "pd_op.maximum",
    "pd_op.mean",
    "pd_op.minimum",
    "pd_op.multiply",
    "pd_op.pad",
    "pd_op.pow",
    "pd_op.prod",
    "pd_op.reduce_as",
    "pd_op.relu",
    "pd_op.relu6",
    "pd_op.reshape",
    "pd_op.roll",
    "pd_op.rsqrt",
    "pd_op.scale",
    "pd_op.scatter",
    "pd_op.scatter_nd_add",
    "pd_op.sigmoid",
    "pd_op.silu",
    "pd_op.sin",
    "pd_op.softmax",
    "pd_op.softsign",
    "pd_op.split",
    "pd_op.sqrt",
    "pd_op.square",
    "pd_op.squeeze",
    "pd_op.stack",
    "pd_op.subtract",
    "pd_op.sum",
    "pd_op.swiglu",
    "pd_op.swish",
    "pd_op.take_along_axis",
    "pd_op.tanh",
    "pd_op.tile",
    "pd_op.topk",
    "pd_op.transpose",
    "pd_op.trunc",
    "pd_op.unsqueeze",
    "pd_op.where",
    "pd_op.p_norm",
]


class ValueWrapper:
    def __init__(self, value) -> None:
        if isinstance(value, ValueWrapper):
            assert isinstance(value._value, (type(None), pir.Value))
        else:
            if not isinstance(value, (type(None), pir.Value)):
                raise TypeError(
                    "Value Wrapper is only support None and pir.Value"
                )
        self._value = value._value if isinstance(value, ValueWrapper) else value

    def __hash__(self) -> int:
        if isinstance(self._value, pir.Value):
            return self._value.hash()
        else:
            return hash(self._value)

    def __eq__(self, other) -> bool:
        if not isinstance(other, ValueWrapper):
            warnings.warn(
                f'In ValueWrapper.__eq__ expected type of `other` is ValueWrapper but received {other.__class__}.'
            )
            return False

        if self._value is None or other._value is None:
            return self._value is None and other._value is None
        return self._value.is_same(other._value)


class ValueDict:
    def __init__(
        self,
        iter=None,
        *,
        default_factory=None,
    ):
        self._items: dict[ValueWrapper] = {}
        self._default_factory = default_factory
        if iter is not None:
            for key, val in iter.items():
                self[key] = val

    def copy(self):
        ret = ValueDict()
        ret._items = self._items.copy()
        ret._default_factory = self._default_factory
        return ret

    def update(self, other_dict):
        for key, val in other_dict.items():
            self[key] = val

    def keys(self):
        for key in self._items.keys():
            yield key._value

    def values(self):
        return self._items.values()

    def items(self):
        for key, val in self._items.items():
            yield key._value, val

    def get(self, key, default=None):
        if not self.__contains__(key):
            return default
        return self._items[ValueWrapper(key)]

    def pop(self, key):
        if not self.__contains__(key):
            raise KeyError(f'{key} is not in ValueDict')
        return self._items.pop(ValueWrapper(key))

    def setdefault(self, key, default=None):
        if not self.__contains__(key):
            self[key] = default
        return self[key]

    def __setitem__(self, key, val: Any):
        self._items[ValueWrapper(key)] = val

    def __getitem__(self, key):
        if not self.__contains__(key):
            if self._default_factory is not None:
                self[key] = self._default_factory()
            else:
                raise KeyError(f'{key} is not in ValueDict')
        return self._items[ValueWrapper(key)]

    def __bool__(self):
        return bool(self._items)

    def __len__(self):
        return len(self._items)

    def __iter__(self):
        return self.keys()

    def __contains__(self, key):
        return ValueWrapper(key) in self._items

    def __repr__(self) -> str:
        items_str = ", ".join(f"{key}: {val}" for key, val in self.items())
        return f'ValueDict({items_str})'


class ValueSet:
    def __init__(
        self, iter: Sequence[ValueWrapper] | set[ValueWrapper] | None = None
    ):
        self._set: set[ValueWrapper] = set()
        if iter is not None:
            for val in iter:
                self.add(val)

    def copy(self):
        ret = ValueSet()
        ret._set = self._set.copy()
        return ret

    def add(self, val):
        if not self.__contains__(val):
            self._set.add(ValueWrapper(val))

    def update(self, other: set):
        for val in other:
            self.add(val)

    def pop(self):
        return self._set.pop()._value

    def remove(self, val):
        self._set.remove(ValueWrapper(val))

    def discard(self, val):
        self._set.discard(ValueWrapper(val))

    def __and__(self, other: ValueSet):
        return ValueSet(self._set & other._set)

    def __sub__(self, other: ValueSet):
        return ValueSet(self._set - other._set)

    def __or__(self, other: ValueSet):
        return ValueSet(self._set | other._set)

    def __bool__(self):
        return bool(self._set)

    def __len__(self):
        return len(self._set)

    def __iter__(self):
        for val in self._set:
            yield val._value

    def __contains__(self, val):
        return ValueWrapper(val) in self._set

    def __repr__(self) -> str:
        items_str = ", ".join(repr(item) for item in self)
        return f'ValueSet({items_str})'


class State:
    """
    record relationship of forward op/value and backward op/value
    one state must be binding with a block, if block has parent block,
    state will include parent block info.

    """

    def __init__(self, block):
        self.block = block
        # value -> list(list(value))
        self.value_to_valuegrad = ValueDict(default_factory=list)
        self.value_to_sumvaluegrad = ValueDict(default_factory=list)
        # operation -> list(operation)
        self.op_to_opgrad = collections.defaultdict(list)

        # value -> list(value)
        self.valuegrad_to_value = ValueDict(default_factory=list)
        self.sumvaluegrad_to_value = ValueDict(default_factory=list)
        # operation -> list(operation)
        self.opgrad_to_op = collections.defaultdict(list)
        # only for controlflow
        # inside_value is sub block value, which will yield to parent block,
        # parent block value is outside_value
        self.inside_value_to_outside_value_map = ValueDict()

    def turn_map(self) -> None:
        self.valuegrad_to_value = ValueDict(default_factory=list)
        self.sumvaluegrad_to_value = ValueDict(default_factory=list)
        self.opgrad_to_op = collections.defaultdict(list)

        for k, v in self.value_to_valuegrad.items():
            if v != []:
                for value in v[0]:
                    self.valuegrad_to_value[value] = [k]
        for k, v in self.value_to_sumvaluegrad.items():
            if v != []:
                for value in v[0]:
                    self.sumvaluegrad_to_value[value] = [k]
        for k, v in self.op_to_opgrad.items():
            if v != []:
                self.opgrad_to_op[v[0]] = [k]

    def copy(self, new_block):
        state = State(new_block)
        state.value_to_valuegrad = self.value_to_valuegrad.copy()
        state.value_to_sumvaluegrad = self.value_to_sumvaluegrad.copy()

        # operation -> list(operation)
        state.op_to_opgrad = self.op_to_opgrad.copy()

        # value -> list(value)
        state.valuegrad_to_value = self.valuegrad_to_value.copy()
        state.sumvaluegrad_to_value = self.sumvaluegrad_to_value.copy()
        # operation -> list(operation)
        state.opgrad_to_op = self.opgrad_to_op.copy()

        # only for controlflow
        state.inside_value_to_outside_value_map = (
            self.inside_value_to_outside_value_map.copy()
        )

        return state


def _check_vjp_dynamic_shape(op, inputs):
    for items in inputs:
        for item in items:
            if (
                item.is_dense_tensor_type()
                and item.initialized()
                and -1 in item.shape
            ):
                return True


# Prim currently does not support dynamic shape, when dynamic shape exits in shape of op inputs, prim will be skipped its vjp op.
@signature_safe_contextmanager
def dynamic_shape_prim_vjp_guard(op, inputs):
    origin_prim = core._is_bwd_prim_enabled()
    if op.name() == "cf.tuple_push":
        skip_prim = True
    else:
        skip_prim = (
            origin_prim
            and core._enable_prim_skip_dynamic_shape()
            and _check_vjp_dynamic_shape(op, inputs)
            and op.name() not in ALLOW_DYNAMIC_SHAPE_VJP_OPS
        )

    try:
        if origin_prim and skip_prim:
            core._set_prim_backward_enabled(False)
        yield
    finally:
        if origin_prim:
            core._set_prim_backward_enabled(True)


def check_type(input, input_name, expected_type, op_name, extra_message=''):
    if not isinstance(input, expected_type):
        raise TypeError(
            f"The type of '{input_name}' in {op_name} must be {expected_type}, but received {type(input)}. {extra_message}"
        )


def _as_list(x):
    if x is None:
        return []
    return list(x) if isinstance(x, Sequence) else [x]


def some_in_set(value_list, value_set):
    return any(v in value_set for v in value_list)


def is_control_flow(op):
    return op.name() == "pd_op.if" or op.name() == "pd_op.while"


def is_builtin_op(op):
    dialect_name, opname = op.name().split(".")
    return dialect_name == "builtin"


def update_no_grad_set_by_stopgradient(block, no_grad_set):
    for op in block.ops:
        if is_control_flow(op):
            for sub_block in op.blocks():
                update_no_grad_set_by_stopgradient(sub_block, no_grad_set)
        for value in op.results():
            if value.stop_gradient and value not in no_grad_set:
                no_grad_set.add(value)


def get_real_op_inputs(op):
    if op.name() == "pd_op.if":
        return get_used_external_value(op)
    elif op.name() == "pd_op.while":
        return op.operands_source() + get_used_external_value(
            op.as_while_op().body()
        )
    elif op.name() == "pd_op.pylayer":
        return get_used_external_value(op)
    else:
        return op.operands_source()


def get_real_op_outputs(op):
    outputs = op.results()
    if op.name() == "pd_op.array_write_":
        for x in op.operands():
            outputs.append(x.source())
    if op.name() == "pd_op.while":
        for internal_op in op.as_while_op().body().ops:
            if internal_op.name() == "pd_op.array_write_":
                for x in internal_op.operands():
                    outputs.append(x.source())
    return outputs


def inverse_sort_op(old_ops):
    '''
    if topo graph is op1 -> op2 -> op3
    return [op3, op2, op1]

    '''

    # init pending_count[op] which describes number of
    # pending edges for its grad_op

    pending_count = collections.defaultdict(int)
    ops = []
    [ops.append(x) for x in old_ops if x not in ops]
    ops_set = set(ops)
    sorted_list = []
    for op in ops:
        for x in get_real_op_inputs(op):
            if not pir.is_fake_value(x) and x.get_defining_op() in ops_set:
                pending_count[x.get_defining_op()] += 1

    queue = collections.deque()

    for op in ops:
        if pending_count[op] == 0:
            queue.append(op)

    while queue:
        op = queue.popleft()
        sorted_list.append(op)
        for x in get_real_op_inputs(op):
            x_op = x.get_defining_op()
            pending_count[x_op] -= 1
            if pending_count[x_op] == 0:
                queue.append(x_op)

    if len(sorted_list) != len(ops):
        raise ValueError(
            "inverse_sort_op wrong, sorted_list size is not equal to origin_list size"
        )
    change_list = []
    # true  %0 = op1, 1% = increment(0%), 3% = op2(0%), tuple_push(%0, 1%, 3%),
    # no one use 1% so increment be the first op, actually op2 use 1% ,
    # sorted_list = [increment, op2, op1] should be [op2, increment, op1],
    # tuple_push(0%) must be forward last op, backward first op, so skip it.
    for op in reversed(sorted_list):
        if op.name() == 'pd_op.increment_':
            idx_1 = sorted_list.index(op)
            idx_2 = sorted_list.index(op)

            for op_in in reversed(sorted_list[: sorted_list.index(op)]):
                if (
                    some_in_set(
                        op.operands_source(),
                        ValueSet(get_real_op_inputs(op_in)),
                    )
                    and op_in.name() != "cf.tuple_push"
                ):
                    idx_2 = sorted_list.index(op_in)
            if idx_1 != idx_2:
                change_list.append((idx_1, idx_2))
    for idx_1, idx_2 in change_list:
        sorted_list[idx_1], sorted_list[idx_2] = (
            sorted_list[idx_2],
            sorted_list[idx_1],
        )

    return sorted_list


def is_inplace_net(op_list):
    '''
    when program has inplace op , it's difficult to find the actual pending_count.
    '''
    for op in op_list:
        if op.name() in ["pd_op.array_write_", "pd_op.assign_out_"]:
            return True
        if is_control_flow(op):
            for block in op.blocks():
                if is_inplace_net(block.ops):
                    return True

    return False


def remove_op(block, op, state):
    '''
    remove op from block
    '''
    if state.opgrad_to_op[op] != []:
        fwd_op = state.opgrad_to_op[op][0]
        state.op_to_opgrad[fwd_op].remove(op)

    for valuegrad in op.results():
        if state.valuegrad_to_value[valuegrad] != []:
            value = state.valuegrad_to_value[valuegrad][0]
            state.value_to_valuegrad[value] = []

            if value in state.sumvaluegrad_to_value:
                raise ValueError(
                    f'input_grad in [%s] is value which need to sum {op.name()}'
                )
    # NOTE(SigureMo): Ensure access to the op's results before removing it.
    # Otherwise, the op will be deconstructed and access the num_results
    # will be undefined behavior, it always cause hanging on the macOS.
    block.remove_op(op)


def while_prune_check(while_tuple_ops):
    if len(while_tuple_ops) != 0:
        for opresult in while_tuple_ops[0].results():
            if not opresult.use_empty():
                return False
        return True
    return False


def remove_useless_full_like_ops(block, ops, state):
    '''
    remove ops which are not in use recursively,

    '''
    remove_ops = []
    inverse_ops = inverse_sort_op(list(ops))
    # from output to input
    for op in inverse_ops:
        if op.name() == "pd_op.full_like":
            if op.result(0).use_empty():
                full_op = op.operand_source(1).get_defining_op()
                remove_ops.append(op)
                remove_ops.append(full_op)
        elif is_control_flow(op):
            for sub_block in op.blocks():
                remove_useless_full_like_ops(sub_block, sub_block.ops, state)

    for op in remove_ops:
        remove_op(block, op, state)


def all_stop_gradient_true(block):
    for op in block.ops:
        for value in op.results():
            if value.stop_gradient is False:
                return False
    return True


def all_input_stop_gradient_true(list_of_list):
    for list_ in list_of_list:
        for stop_gradient in list_:
            if stop_gradient is False:
                return False
    return True


def all_output_grad_none(list_of_list):
    for list_ in list_of_list:
        for value in list_:
            if value is not None:
                return False
    return True


def op_has_vjp(op):
    # NOTE(MarioLulab): In PIR mode, even though the `PyLayer` op does
    # not have a vjp interface, we still need to generate the backward
    # block based on its registered backward function. To achieve this,
    # we add more handling logic for `PyLayer` Op in the `call_vjp` function
    return core.has_vjp(op) or op.name() == "pd_op.pylayer"


def parent_total_ops(block):
    '''
    when block is sub_block, forward op should include its parent block ops
    (sub block nest should Add on demand to avoid block copy)
    '''
    total_ops = []
    if block.parent_block is not None:
        if block.parent_block.parent_block:
            total_ops += block.parent_block.parent_block.ops
        total_ops += block.parent_block.ops
    total_ops += block.ops

    return total_ops


# only for control_flow to find corresponding value or value_list
def return_map_value(value, map):
    output = value
    while output in map:
        output = map[output]
    return output


def return_map_value_list(value, map):
    output = []
    for i in range(len(value)):
        if value[i] in map:
            output.append(return_map_value(value[i], map))
        else:
            output.append(value[i])
    return output


def argument_to_value(while_op):
    '''
    return while op's relationship of (block_argument to input value) and (input value to block_argument).
    '''
    if while_op.name() != "pd_op.while":
        return ValueDict(), ValueDict()

    assert len(while_op.as_while_op().block_arguments()) + 1 == len(
        while_op.operands_source()
    ), "while op's block_arguments size + 1 should same to while op's operands_source size"
    arg_to_value_map = ValueDict()
    value_to_arg_map = ValueDict()
    for arg, value in zip(
        while_op.as_while_op().block_arguments(),
        while_op.operands_source()[1:],
    ):
        arg_to_value_map[arg] = value
        value_to_arg_map[value] = arg
    return arg_to_value_map, value_to_arg_map


def get_grad_semantic_info(op):
    '''
    return whether op's inputs has grad, usually handled from yaml.
    some op has uncertain inputs need special handling.
    '''
    if op.name() in [
        "builtin.combine",
        "pd_op.if",
        "pd_op.while",
        "pd_op.pylayer",
        "cf.tuple_push",
        "dist_op.moe_global_mesh_tensor",
        "dist_op.moe_sub_mesh_tensors",
        "dist_op.dist_reshape",
    ]:
        grad_semantic_info = [True for _ in range(len(get_real_op_inputs(op)))]
        if op.name() == "pd_op.if":
            grad_semantic_info[0] = False
    else:
        grad_semantic_info = op.get_input_grad_semantics()
    return grad_semantic_info


def get_split_op(value):
    for op in value.all_used_ops():
        if op.name() == "builtin.split":
            return op
    return None


@lru_cache
def warning_once(message: str):
    logging.warning(message)


def update_if_output_stopgradient(if_op, true_yield_op, false_yield_op):
    """
    Update if_op's stop_gradient based on true_yield_op and false_yield_op.

    Args:
    true_yield_op: true block of if_op's last op.
    false_yield_op: false block of if_op's last op.
    if_op: update it's op_results()'s stop_gradient.
    """
    if (
        true_yield_op.name() != 'cf.yield'
        or false_yield_op.name() != 'cf.yield'
    ):
        raise ValueError("param is not yield op")

    # Check if operands_source sizes match
    if len(true_yield_op.operands_source()) != len(
        false_yield_op.operands_source()
    ):
        raise ValueError("Mismatched yield operands_source sizes")

    # Check if op_results size matches operands_source
    if len(if_op.results()) != len(true_yield_op.operands_source()):
        raise ValueError(
            "Mismatched if op_results size with yield operands_source"
        )

    # Update if_op's stop_gradient
    for i in range(len(true_yield_op.operands_source())):
        stop_grad1 = true_yield_op.operand_source(i).stop_gradient
        stop_grad2 = false_yield_op.operand_source(i).stop_gradient

        # Set to False if either stop_gradient is False
        if not stop_grad1 or not stop_grad2:
            if_op.result(i).stop_gradient = False


def update_while_output_stopgradient(while_op, yield_op):
    """
    Update while_op's stop_gradient based on yield_op.

    Args:
    yield_op: The yield operation associated with the while loop.
    while_op: The while operation whose op_results()'s stop_gradient needs to be updated.
    """
    # Check if yield_op is indeed a yield operation
    if yield_op.name() != 'cf.yield':
        raise ValueError("yield_op is not a yield operation")

    # Check if operands_source size of yield_op matches op_results size of while_op
    if len(while_op.results()) + 1 != len(yield_op.operands_source()):
        raise ValueError(
            f"Mismatched while op_results  size %d with yield operands_source %d. {len(while_op.results()) + 1, len(yield_op.operands_source())}"
        )

    # Update while_op's stop_gradient
    for i in range(1, len(yield_op.operands_source())):
        stop_grad = yield_op.operand_source(i).stop_gradient

        # Set to False if stop_gradient is False
        if not stop_grad:
            while_op.result(i - 1).stop_gradient = False
