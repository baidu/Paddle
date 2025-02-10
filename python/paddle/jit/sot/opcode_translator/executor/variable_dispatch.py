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

import inspect
import math
import operator
from functools import partial, reduce
from typing import TYPE_CHECKING

import paddle

from ...utils import BreakGraphError, FallbackError
from ...utils.magic_methods import (
    BINARY_OPS,
    UNARY_OPS,
    magic_method_builtin_dispatch,
)
from ...utils.paddle_api_config import get_tensor_methods
from .dispatch_functions import (
    operator_in,
    operator_is_none,
    operator_is_not_none,
    operator_not_in,
    raise_break_graph_fn,
    tensor_numel,
)
from .dispatcher import Dispatcher, optional
from .tracker import ConstTracker, DanglingTracker, DummyTracker
from .variables import (
    BuiltinVariable,
    ConstantVariable,
    ContainerVariable,
    DictVariable,
    EnumerateVariable,
    ListVariable,
    MapVariable,
    NumpyVariable,
    RangeVariable,
    SliceVariable,
    SymbolicVariable,
    TupleVariable,
    VariableBase,
    VariableFactory,
    ZipVariable,
)

if TYPE_CHECKING:
    from .variables import DataVariable, TensorVariable


def add_guard(var: VariableBase):
    var.graph.add_global_guarded_variable(var)
    return var


def raise_err_handle(error):
    def inner(*args, **kwargs):
        raise error

    return inner


# slice
Dispatcher.register(
    slice,
    ("VariableBase",),
    lambda stop: SliceVariable(
        slice(stop),
        graph=stop.graph,
        tracker=DummyTracker([stop]),
    ),
)

Dispatcher.register(
    slice,
    ("VariableBase", "VariableBase"),
    lambda start, stop: SliceVariable(
        slice(start, stop),
        graph=stop.graph,
        tracker=DummyTracker([start, stop]),
    ),
)

Dispatcher.register(
    slice,
    ("VariableBase", "VariableBase", "VariableBase"),
    lambda start, stop, step: SliceVariable(
        slice(start, stop, step),
        graph=stop.graph,
        tracker=DummyTracker([start, stop, step]),
    ),
)


# iter
Dispatcher.register(
    iter,
    ("VariableBase",),
    lambda variable: variable.get_iter(),
)


# in
Dispatcher.register(
    operator_in,
    ("VariableBase", "IterVariable"),
    raise_err_handle(BreakGraphError("Codes like: `variable in iterator`.")),
)

Dispatcher.register(
    operator_in,
    ("TensorVariable", "VariableBase"),
    lambda left, right: ConstantVariable(
        left.id
        in [
            x.id
            for x in right.get_py_value(allow_tensor=True)
            if hasattr(x, "id")
        ],
        left.graph,
        tracker=DummyTracker([left, right]),
    ),
)

Dispatcher.register(
    operator_in,
    ("VariableBase", "VariableBase"),
    lambda left, right: ConstantVariable(
        left.get_py_value(allow_tensor=True)
        in right.get_py_value(allow_tensor=True),
        left.graph,
        tracker=DummyTracker([left, right]),
    ),
)

Dispatcher.register(
    operator_not_in,
    ("VariableBase", "IterVariable"),
    raise_err_handle(
        BreakGraphError("Codes like: `variable not in iterator`.")
    ),
)

Dispatcher.register(
    operator_not_in,
    ("TensorVariable", "VariableBase"),
    lambda left, right: ConstantVariable(
        left.id
        not in [
            x.id
            for x in right.get_py_value(allow_tensor=True)
            if hasattr(x, "id")
        ],
        left.graph,
        tracker=DummyTracker([left, right]),
    ),
)

Dispatcher.register(
    operator_not_in,
    ("VariableBase", "VariableBase"),
    lambda left, right: ConstantVariable(
        left.get_py_value(allow_tensor=True)
        not in right.get_py_value(allow_tensor=True),
        left.graph,
        tracker=DummyTracker([left, right]),
    ),
)


# type
Dispatcher.register(
    type,
    ("ConstantVariable | SymbolicVariable",),
    lambda var: VariableFactory.from_value(
        var.get_py_type(), graph=var.graph, tracker=DummyTracker([var])
    ),
)

# dict
Dispatcher.register(
    dict,
    (),
    lambda: DictVariable(
        {},
        graph=Dispatcher.graph,
        tracker=DummyTracker([]),
    ),
)

Dispatcher.register(
    dict,
    ("DictVariable",),
    lambda var: var.copy(),
)


@Dispatcher.register_decorator(dict)
def dispatch_dict_kwargs(**kwargs: VariableBase):
    res_dict = {}
    graph = Dispatcher.graph
    for key, value in kwargs.items():
        res_dict[key] = value
    return DictVariable(res_dict, graph, DummyTracker(list(kwargs.values())))


@Dispatcher.register_decorator(dict)
def dispatch_dict(var: ListVariable | TupleVariable):
    res_dict = {}
    length_var = BuiltinVariable(len, var.graph, DanglingTracker())(var)
    getitem = BuiltinVariable(operator.getitem, var.graph, DanglingTracker())
    for index in range(length_var.get_py_value()):
        index_value = getitem(var, index)
        # check
        assert isinstance(index_value, (ListVariable, TupleVariable))
        assert len(index_value) == 2
        # recombination
        key = getitem(index_value, 0)
        value = getitem(index_value, 1)
        value.graph.add_global_guarded_variable(key)
        res_dict.update({key.get_py_value(): value})
    return DictVariable(res_dict, var.graph, DummyTracker([var]))


@Dispatcher.register_decorator(dict.fromkeys)
def dispatch_dict_fromkeys(
    seq: ListVariable | TupleVariable,
    default: VariableBase = None,  # type: ignore
):
    if default is None:
        default = ConstantVariable.wrap_literal(None, seq.graph)
    res_dict = {}
    getitem = BuiltinVariable(operator.getitem, seq.graph, DanglingTracker())
    for index in range(len(seq)):
        index_value = getitem(seq, index)
        seq.graph.add_global_guarded_variable(index_value)
        res_dict.update({index_value.get_py_value(): default})
    return DictVariable(res_dict, seq.graph, DummyTracker([seq]))


Dispatcher.register(
    dict.get,
    ("DictVariable", "ConstantVariable", optional("VariableBase")),
    lambda var, key, default=None: var.get(key.get_py_value(), default),
)
Dispatcher.register(
    dict.keys,
    ("DictVariable",),
    lambda var: var.keys(),
)

Dispatcher.register(
    operator.not_,
    ("VariableBase",),
    lambda x: ConstantVariable(
        not x.get_py_value(allow_tensor=False), x.graph, DummyTracker([x])
    ),
)

Dispatcher.register(
    dict.values,
    ("DictVariable",),
    lambda var: var.values(),
)
Dispatcher.register(
    dict.items,
    ("DictVariable",),
    lambda var: var.items(),
)
Dispatcher.register(
    dict.setdefault,
    ("DictVariable", "ConstantVariable", optional("VariableBase")),
    lambda var, key, default=None: var.setdefault(key.get_py_value(), default),
)
Dispatcher.register(
    dict.update,
    ("DictVariable", "DictVariable"),
    lambda var, other: var.update(other),
)
Dispatcher.register(
    dict.copy,
    ("DictVariable",),
    lambda var: var.copy(),
)
Dispatcher.register(
    dict.clear,
    ("DictVariable",),
    lambda var: var.clear(),
)
Dispatcher.register(
    dict.pop,
    ("DictVariable", "ConstantVariable"),
    lambda var, key: var.pop(key.get_py_value()),
)
Dispatcher.register(
    dict.pop,
    ("DictVariable", "ConstantVariable", "VariableBase"),
    lambda var, key, default: var.pop(key.get_py_value(), default),
)
Dispatcher.register(
    dict.popitem,
    ("DictVariable",),
    lambda var: var.popitem(),
)

# tuple
Dispatcher.register(
    tuple,
    ("ContainerVariable",),
    lambda var: TupleVariable(
        tuple(var.get_wrapped_items()),
        graph=var.graph,
        tracker=DummyTracker([var]),
    ),
)
Dispatcher.register(
    tuple,
    ("SequenceIterVariable",),
    lambda var: TupleVariable(
        tuple(var.to_list()),
        graph=var.graph,
        tracker=DummyTracker([var]),
    ),
)
Dispatcher.register(
    tuple.count,
    ("TupleVariable", "VariableBase"),
    lambda var, value: var.count(value),
)
Dispatcher.register(
    tuple.index,
    ("TupleVariable", "VariableBase"),
    lambda var, value: var.index(value),
)
Dispatcher.register(
    operator.add,
    ("TupleVariable", "TupleVariable"),
    lambda var, other: var.concat(other),
)
Dispatcher.register(
    operator.iadd,
    ("TupleVariable", "TupleVariable"),
    lambda var, other: var.concat(other),
)


@Dispatcher.register_decorator(operator.eq)
def dispatch_tuple_eq(lhs: TupleVariable, rhs: TupleVariable):
    if len(lhs) != len(rhs):
        return ConstantVariable(False, lhs.graph, DummyTracker([lhs, rhs]))
    size = len(lhs)

    return ConstantVariable(
        all(
            Dispatcher.call(operator.eq, lhs[i], rhs[i]).get_py_value()
            for i in range(size)
        ),
        lhs.graph,
        DummyTracker([lhs, rhs]),
    )


@Dispatcher.register_decorator(operator.ne)
def dispatch_tuple_ne(lhs: TupleVariable, rhs: TupleVariable):
    return Dispatcher.call(operator.eq, lhs, rhs).bool_not()


# list
Dispatcher.register(
    list,
    (),
    lambda: ListVariable(
        [],
        graph=Dispatcher.graph,
        tracker=DummyTracker([]),
    ),
)

Dispatcher.register(
    list,
    ("ContainerVariable",),
    lambda var: ListVariable(
        list(var.get_wrapped_items()),
        graph=var.graph,
        tracker=DummyTracker([var]),
    ),
)

Dispatcher.register(
    list,
    ("IterVariable",),
    lambda var: ListVariable(
        var.to_list(),
        graph=var.graph,
        tracker=DummyTracker([var]),
    ),
)
Dispatcher.register(
    list.extend,
    (
        "ListVariable",
        "ListVariable | TupleVariable | DictVariable | RangeVariable",
    ),
    lambda var, other: var.extend(other),
)
Dispatcher.register(
    list.append,
    ("ListVariable", "VariableBase"),
    lambda var, other: var.append(other),
)
Dispatcher.register(
    list.insert,
    ("ListVariable", "ConstantVariable", "VariableBase"),
    lambda var, index, obj: var.insert(index.get_py_value(), obj),
)
Dispatcher.register(
    list.remove,
    ("ListVariable", "VariableBase"),
    lambda var, other: var.remove(other),
)
Dispatcher.register(
    list.pop,
    ("ListVariable", optional("ConstantVariable")),
    lambda var, index=None: var.pop(index),
)
Dispatcher.register(
    list.clear,
    ("ListVariable",),
    lambda var: var.clear(),
)
Dispatcher.register(
    list.sort,
    ("ListVariable",),
    lambda var: var.sort(),
)
Dispatcher.register(
    list.reverse,
    ("ListVariable",),
    lambda var: var.reverse(),
)
Dispatcher.register(
    list.copy,
    ("ListVariable",),
    lambda var: var.copy(),
)
Dispatcher.register(
    list.count,
    ("ListVariable", "VariableBase"),
    lambda var, obj: var.count(obj),
)
Dispatcher.register(
    list.index,
    ("ListVariable", "VariableBase"),
    lambda var, obj: var.index(obj),
)
Dispatcher.register(
    operator.add,
    ("ListVariable", "ListVariable"),
    lambda var, other: var.concat(other),
)
Dispatcher.register(
    operator.iadd,
    ("ListVariable", "ListVariable"),
    lambda var, other: var.inplace_concat(other),
)
Dispatcher.register(
    operator.mul,
    ("ListVariable | TupleVariable", "ConstantVariable"),
    lambda var, other: var.repeat(other),
)


@Dispatcher.register_decorator(operator.eq)
def dispatch_list_eq(lhs: ListVariable, rhs: ListVariable):
    if len(lhs) != len(rhs):
        return ConstantVariable(False, lhs.graph, DummyTracker([lhs, rhs]))
    size = len(lhs)

    return ConstantVariable(
        all(
            Dispatcher.call(operator.eq, lhs[i], rhs[i]).get_py_value()
            for i in range(size)
        ),
        lhs.graph,
        DummyTracker([lhs, rhs]),
    )


@Dispatcher.register_decorator(operator.ne)
def dispatch_list_ne(lhs: ListVariable, rhs: ListVariable):
    return Dispatcher.call(operator.eq, lhs, rhs).bool_not()


BUILTIN_EQ_DISPATCH_TYPES = [
    "ListVariable",
    "TupleVariable",
    "DictVariable",
    "ConstantVariable",
]

for i in range(len(BUILTIN_EQ_DISPATCH_TYPES)):
    current_type = BUILTIN_EQ_DISPATCH_TYPES[i]
    other_types = (
        BUILTIN_EQ_DISPATCH_TYPES[:i] + BUILTIN_EQ_DISPATCH_TYPES[i + 1 :]
    )
    Dispatcher.register(
        operator.eq,
        (current_type, " | ".join(other_types)),
        lambda var, other: ConstantVariable(
            False, var.graph, DummyTracker([var, other])
        ),
    )

    Dispatcher.register(
        operator.ne,
        (current_type, " | ".join(other_types)),
        lambda var, other: ConstantVariable(
            True, var.graph, DummyTracker([var, other])
        ),
    )


# getattr
Dispatcher.register(
    getattr,
    ("VariableBase", "ConstantVariable", optional("VariableBase")),
    lambda var, name, default=None: var.getattr(
        add_guard(name).get_py_value(), default
    ),
)

# hasattr
Dispatcher.register(
    hasattr,
    ("VariableBase", "ConstantVariable"),
    lambda var, name: var.hasattr(add_guard(name).get_py_value()),
)

Dispatcher.register(
    delattr,
    ("VariableBase", "VariableBase"),
    lambda var, name: var.delattr(add_guard(name).get_py_value()),
)

Dispatcher.register(
    setattr,
    ("VariableBase", "VariableBase", "VariableBase"),
    lambda var, name, value: var.setattr(add_guard(name).get_py_value(), value),
)

# len
Dispatcher.register(
    len,
    ("ContainerVariable | ContainerLayerVariable",),
    lambda var: var.len(),
)

# range
# stop
Dispatcher.register(
    range,
    ("ConstantVariable | TensorVariable",),
    lambda stop: RangeVariable(
        ConstantVariable.wrap_literal(0, stop.graph),
        stop,
        ConstantVariable.wrap_literal(1, stop.graph),
        graph=stop.graph,
        tracker=DummyTracker([stop]),
    ),
)

# start, stop
Dispatcher.register(
    range,
    ("ConstantVariable | TensorVariable", "ConstantVariable | TensorVariable"),
    lambda start, stop: RangeVariable(
        start,
        stop,
        ConstantVariable.wrap_literal(1, stop.graph),
        graph=stop.graph,
        tracker=DummyTracker([start, stop]),
    ),
)
# start, stop, step
Dispatcher.register(
    range,
    (
        "ConstantVariable | TensorVariable",
        "ConstantVariable | TensorVariable",
        "ConstantVariable | TensorVariable",
    ),
    lambda start, stop, step: RangeVariable(
        start,
        stop,
        step,
        graph=stop.graph,
        tracker=DummyTracker([start, stop, step]),
    ),
)
# TODO(zmh): Modify
# enumerate
Dispatcher.register(
    enumerate,
    ("VariableBase",),
    lambda var: EnumerateVariable.from_iterator(
        var, graph=var.graph, tracker=DummyTracker([var])
    ),
)


# zip
@Dispatcher.register_decorator(zip)
def create_zip(*var: VariableBase):
    return ZipVariable.from_iterator(
        var, graph=Dispatcher.graph, tracker=DummyTracker(list(var))
    )


# map
Dispatcher.register(
    map,
    (
        "CallableVariable",
        "VariableBase",
    ),
    lambda fn, var: MapVariable.from_iterator(
        fn, var, graph=var.graph, tracker=DummyTracker([var])
    ),
)


# reversed
@Dispatcher.register_decorator(reversed)
def dispatch_reversed(var: ContainerVariable):
    from .tracker import DanglingTracker
    from .variables import BuiltinVariable, SequenceIterVariable

    length_var = BuiltinVariable(len, var.graph, DanglingTracker())(var)
    assert isinstance(length_var, ConstantVariable)
    getitem = BuiltinVariable(operator.getitem, var.graph, DanglingTracker())
    out = reversed([getitem(var, i) for i in range(length_var.get_py_value())])
    out_var = ListVariable(
        list(out), graph=var.graph, tracker=DummyTracker([var])
    )
    return SequenceIterVariable(
        out_var,
        graph=var.graph,
        tracker=DummyTracker([var]),
    )


# isinstance
Dispatcher.register(
    isinstance,
    ("TensorVariable", "VariableBase"),
    lambda left, right: ConstantVariable(
        isinstance(
            paddle.to_tensor(0),
            right.get_py_value(allow_tensor=True),
        ),
        left.graph,
        DummyTracker([left, right]),
    ),
)

Dispatcher.register(
    isinstance,
    ("VariableBase", "VariableBase"),
    lambda left, right: ConstantVariable(
        isinstance(
            left.get_py_value(allow_tensor=True),
            right.get_py_value(allow_tensor=True),
        ),
        left.graph,
        DummyTracker([left, right]),
    ),
)

# bool
Dispatcher.register(
    bool,
    ("ContainerVariable",),
    lambda var: var.bool(),
)

Dispatcher.register(
    operator.truth,
    ("ConstantVariable",),
    lambda var: Dispatcher.call(bool, var),
)

# str
Dispatcher.register(
    str,
    ("ConstantVariable",),
    lambda var: var.str(),
)


@Dispatcher.register_decorator(str.format)
def str_format(var: ConstantVariable, *args: ConstantVariable):
    return var.format(*args)


Dispatcher.register(
    str.lower,
    ("ConstantVariable",),
    lambda var: var.lower(),
)


@Dispatcher.register_decorator(str.startswith)
def str_startswith(
    var: ConstantVariable,
    substr: ConstantVariable,
    beg: ConstantVariable = None,  # type: ignore
    end: ConstantVariable = None,  # type: ignore
):
    value = var.get_py_value()
    if end is None:
        end = ConstantVariable(len(value), var.graph, DanglingTracker())
    if beg is None:
        beg = ConstantVariable(0, var.graph, DanglingTracker())

    res = value.startswith(
        substr.get_py_value(), beg.get_py_value(), end.get_py_value()
    )
    return ConstantVariable(
        res, var.graph, DummyTracker([var, substr, beg, end])
    )


@Dispatcher.register_decorator(str.endswith)
def str_endswith(
    var: ConstantVariable,
    substr: ConstantVariable,
    beg: ConstantVariable = None,  # type: ignore
    end: ConstantVariable = None,  # type: ignore
):
    value = var.get_py_value()
    if end is None:
        end = ConstantVariable(len(value), var.graph, DanglingTracker())
    if beg is None:
        beg = ConstantVariable(0, var.graph, DanglingTracker())

    res = value.endswith(
        substr.get_py_value(), beg.get_py_value(), end.get_py_value()
    )
    return ConstantVariable(
        res, var.graph, DummyTracker([var, substr, beg, end])
    )


# getitem
# TODO: Should pass its Variable into the getitem and perform operations such as getting value in the getitem. like this:https://github.com/PaddlePaddle/PaddleSOT/pull/198#discussion_r1241110949
Dispatcher.register(
    operator.getitem,
    (
        "TensorVariable",
        "Any",
    ),
    lambda var, key: var.getitem(
        VariableFactory.from_value(
            key, graph=var.graph, tracker=ConstTracker(key)
        )
    ),
)

Dispatcher.register(
    operator.getitem,
    (
        "VariableBase",
        "int | str",
    ),
    lambda var, key: var.getitem(
        VariableFactory.from_value(
            key, graph=var.graph, tracker=ConstTracker(key)
        )
    ),
)

Dispatcher.register(
    operator.getitem,
    (
        "VariableBase",
        "ConstantVariable | SliceVariable",
    ),
    lambda var, key: var.getitem(key),
)

# setitem
Dispatcher.register(
    operator.setitem,
    (
        "TensorVariable",
        "Any",
        "VariableBase",
    ),
    lambda var, key, value: var.setitem(
        VariableFactory.from_value(
            key, graph=var.graph, tracker=ConstTracker(key)
        ),
        value,
    ),
)

Dispatcher.register(
    operator.setitem,
    (
        "VariableBase",
        "int | str | ConstantVariable | TensorVariable | ContainerVariable",
        "VariableBase",
    ),
    lambda var, key, value: var.setitem(add_guard(key).get_py_value(), value),
)

# delitem
Dispatcher.register(
    operator.delitem,
    (
        "VariableBase",
        "int | str | TensorVariable",
    ),
    lambda var, key: var.delitem(key),
)
Dispatcher.register(
    operator.delitem,
    (
        "VariableBase",
        "ConstantVariable",
    ),
    lambda var, key: var.delitem(add_guard(key).get_py_value()),
)


# TensorVariable
Dispatcher.register(
    paddle.is_tensor,
    ("TensorVariable",),
    lambda var: var.is_tensor(),
)
Dispatcher.register(
    paddle.is_complex,
    ("TensorVariable",),
    lambda var: var.is_complex(),
)
Dispatcher.register(
    paddle.is_integer,
    ("TensorVariable",),
    lambda var: var.is_integer(),
)
Dispatcher.register(
    paddle.is_floating_point,
    ("TensorVariable",),
    lambda var: var.is_floating_point(),
)
Dispatcher.register(
    paddle.rank,
    ("TensorVariable",),
    lambda var: var.ndim,
)

Dispatcher.register(
    operator.is_,
    ("TensorVariable", "TensorVariable"),
    lambda var, other: ConstantVariable(
        var.get_symbol() == other.get_symbol(),
        var.graph,
        tracker=DummyTracker([var, other]),
    ),
)

Dispatcher.register(
    operator.is_,
    ("TensorVariable", "VariableBase"),
    lambda var, other: ConstantVariable(
        False,
        var.graph,
        tracker=DummyTracker([var, other]),
    ),
)

Dispatcher.register(
    operator.is_,
    ("VariableBase", "TensorVariable"),
    lambda var, other: ConstantVariable(
        False,
        var.graph,
        tracker=DummyTracker([var, other]),
    ),
)

# VariableBase
Dispatcher.register(
    operator.is_,
    ("VariableBase", "VariableBase"),
    lambda var, other: ConstantVariable(
        var.get_py_value() is other.get_py_value(),
        var.graph,
        tracker=DummyTracker([var, other]),
    ),
)


@Dispatcher.register_decorator(operator.is_not)
def is_not_func(var: VariableBase, other: VariableBase):
    handler = Dispatcher.dispatch(operator.is_, var, other)
    if handler is None:
        raise FallbackError(
            f"Not found implementation operator.is for {var} and {other}."
        )
    return handler(var, other).bool_not()


# is None
Dispatcher.register(
    operator_is_none,
    ("ConstantVariable",),
    lambda var: BuiltinVariable(operator.is_, var.graph, DanglingTracker())(
        var, ConstantVariable.wrap_literal(None, var.graph)
    ),
)

# is not None
Dispatcher.register(
    operator_is_not_none,
    ("ConstantVariable",),
    lambda var: BuiltinVariable(operator.is_not, var.graph, DanglingTracker())(
        var, ConstantVariable.wrap_literal(None, var.graph)
    ),
)

# is None
Dispatcher.register(
    operator_is_none,
    ("VariableBase",),
    lambda var: ConstantVariable(False, var.graph, DummyTracker([var])),
)

# is not None
Dispatcher.register(
    operator_is_not_none,
    ("VariableBase",),
    lambda var: ConstantVariable(True, var.graph, DummyTracker([var])),
)


# NOTE(SigureMo): Don't directly capture free var inside for-loop, use partial instead.
# ```python
# lambdas = []
# for i in range(10):
#     lambdas.append(lambda: i)
# for fn in lambdas:
#     print(fn()) # result is 9, 9, 9, 9, 9, 9, 9, 9, 9, 9
# ```
# Rewrite by partial:
# ```python
# lambdas = []
# for i in range(10):
#     lambdas.append(partial(lambda i: i, i))
# for fn in lambdas:
#     print(fn()) # result is 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
# ```

# Constant
for unary_fn in UNARY_OPS:
    for magic_method in magic_method_builtin_dispatch(unary_fn):
        Dispatcher.register(
            unary_fn,
            ("ConstantVariable",),
            partial(
                lambda fn, var: VariableFactory.from_value(
                    fn(var.get_py_value()),
                    var.graph,
                    tracker=DummyTracker([var]),
                ),
                unary_fn,
            ),
        )
for binary_fn in BINARY_OPS:
    for magic_method in magic_method_builtin_dispatch(binary_fn):
        Dispatcher.register(
            binary_fn,
            ("ConstantVariable", "ConstantVariable"),
            partial(
                lambda fn, var, other: VariableFactory.from_value(
                    fn(var.get_py_value(), other.get_py_value()),
                    var.graph,
                    tracker=DummyTracker([var, other]),
                ),
                binary_fn,
            ),
        )
# Tensor and Symbolic
fallback_tensor_unary_method = {
    int,
    bool,
    float,
    operator.truth,
}

Dispatcher.register(tensor_numel, ("TensorVariable",), lambda x: x.numel())

for unary_fn in UNARY_OPS:
    if unary_fn in fallback_tensor_unary_method:
        Dispatcher.register(
            unary_fn,
            ("TensorVariable",),
            raise_break_graph_fn,
        )
        continue

    if unary_fn is len:
        Dispatcher.register(
            unary_fn,
            ("TensorVariable",),
            lambda x: x.len(),
        )
        continue

    for magic_method in magic_method_builtin_dispatch(unary_fn):
        Dispatcher.register(
            unary_fn,
            ("TensorVariable",),
            partial(
                lambda magic_name, var: var.graph.call_tensor_method(
                    magic_name, var
                ),
                magic_method.name,
            ),
        )
        Dispatcher.register(
            unary_fn,
            ("SymbolicVariable",),
            partial(
                lambda magic_name, var: var.graph.call_symbolic_method(
                    magic_name, var
                ),
                magic_method.name,
            ),
        )
for binary_fn in BINARY_OPS:
    for magic_method in magic_method_builtin_dispatch(binary_fn):
        # skip all inplace magic method name, we will dispatch it to non-inplace
        # magic methods
        if magic_method.is_inplace:
            continue

        if not magic_method.is_reverse:
            Dispatcher.register(
                binary_fn,
                (
                    "TensorVariable",
                    "TensorVariable | SymbolicVariable | ConstantVariable | NumpyVariable",
                ),
                partial(
                    lambda magic_name, var, other: var.graph.call_tensor_method(
                        magic_name, var, other
                    ),
                    magic_method.name,
                ),
            )
        else:
            # skip __mod__ for str and TensorVariable
            if magic_method.name == "__rmod__":

                @Dispatcher.register_decorator(operator.mod)
                def tensor_mod_dispatcher(
                    var: ConstantVariable | SymbolicVariable,
                    other: TensorVariable,
                ):
                    if var.get_py_type() is str:
                        raise BreakGraphError(
                            "(ConstantVariable % TensorVariable) raise a callback. "
                        )
                    raise FallbackError("Tensor doesn't support __rmod__")

            else:
                Dispatcher.register(
                    binary_fn,
                    (
                        "SymbolicVariable | ConstantVariable | NumpyVariable",
                        "TensorVariable",
                    ),
                    partial(
                        lambda reverse_magic_name, var, other: other.graph.call_tensor_method(
                            reverse_magic_name, other, var
                        ),
                        magic_method.name,
                    ),
                )

for binary_fn in BINARY_OPS:
    for magic_method in magic_method_builtin_dispatch(binary_fn):
        if magic_method.name not in get_tensor_methods():
            continue
        # skip all inplace magic method name, we will dispatch it to non-inplace
        # magic methods
        if magic_method.is_inplace:
            continue

        if not magic_method.is_reverse:
            Dispatcher.register(
                binary_fn,
                (
                    "SymbolicVariable",
                    "ConstantVariable | SymbolicVariable",
                ),
                partial(
                    lambda magic_name, var, other: var.graph.call_symbolic_method(
                        magic_name, var, other
                    ),
                    magic_method.name,
                ),
            )
        else:
            Dispatcher.register(
                binary_fn,
                ("ConstantVariable", "SymbolicVariable"),
                partial(
                    lambda reverse_magic_name, var, other: var.graph.call_symbolic_method(
                        reverse_magic_name, other, var
                    ),
                    magic_method.name,
                ),
            )

# Register dispatch for NumpyVariable: fallback !
for unary_fn in UNARY_OPS:
    if unary_fn in [bool]:
        continue
    for magic_method in magic_method_builtin_dispatch(unary_fn):

        @Dispatcher.register_decorator(unary_fn)
        def numpy_unary_dispatcher(var: NumpyVariable):
            raise FallbackError('Numpy operator need fallback to dygraph')


Dispatcher.register(
    operator.eq,
    ("NumpyVariable", "ConstantVariable | NumpyVariable"),
    lambda left, right: constant_numpy_equal(right, left),
)


for binary_fn in BINARY_OPS:
    for magic_method in magic_method_builtin_dispatch(binary_fn):

        @Dispatcher.register_decorator(binary_fn)
        def numpy_binary_dispatcher(var: NumpyVariable, other: NumpyVariable):
            raise FallbackError('Numpy operator need fallback to dygraph')


# Register dispatch for DataVariable: directly call and return a wrapped variable.
def data_variable_binary_dispatcher(var, other, operator):
    return VariableFactory.from_value(
        operator(var.get_py_value(), other.get_py_value()),
        var.graph,
        DummyTracker([var, other]),
    )


for binary_fn in BINARY_OPS:
    for magic_method in magic_method_builtin_dispatch(binary_fn):
        Dispatcher.register(
            binary_fn,
            ("DataVariable", "Any"),
            partial(data_variable_binary_dispatcher, operator=binary_fn),
        )
        Dispatcher.register(
            binary_fn,
            ("Any", "DataVariable"),
            partial(data_variable_binary_dispatcher, operator=binary_fn),
        )

for unary_fn in UNARY_OPS:
    for magic_method in magic_method_builtin_dispatch(unary_fn):

        def data_variable_unary_dispatcher(var: DataVariable, fn):
            return VariableFactory.from_value(
                fn(var.get_py_value()),
                var.graph,
                DummyTracker([var]),
            )

        Dispatcher.register(
            unary_fn,
            ("DataVariable",),
            partial(data_variable_unary_dispatcher, fn=unary_fn),
        )


Dispatcher.register(
    math.ceil,
    ("ConstantVariable",),
    lambda var: ConstantVariable(
        math.ceil(var.get_py_value()),
        var.graph,
        tracker=DummyTracker([var]),
    ),
)

Dispatcher.register(
    math.floor,
    ("ConstantVariable",),
    lambda var: ConstantVariable(
        math.floor(var.get_py_value()),
        var.graph,
        tracker=DummyTracker([var]),
    ),
)

Dispatcher.register(
    ord,
    ("ConstantVariable",),
    lambda var: var.ord(),
)

Dispatcher.register(
    chr,
    ("ConstantVariable",),
    lambda var: var.chr(),
)


# pow
# base ** exp % mod
@Dispatcher.register_decorator(pow)
def dispatch_pow(
    base: VariableBase,
    exp: VariableBase,
    mod: VariableBase = None,  # type: ignore
):
    graph = base.graph
    result = BuiltinVariable(operator.pow, graph, DanglingTracker())(base, exp)
    if exp is not None:
        result = BuiltinVariable(operator.mod, graph, DanglingTracker())(
            result, mod
        )
    return result


Dispatcher.register(
    math.pow,
    ("ConstantVariable", "ConstantVariable"),
    lambda var1, var2: ConstantVariable(
        math.pow(var1.get_py_value(), var2.get_py_value()),
        var1.graph,
        tracker=DummyTracker([var1, var2]),
    ),
)


@Dispatcher.register_decorator(sum)
def dispatch_sum(
    var: ContainerVariable | TensorVariable,
    start: VariableBase = None,  # type: ignore
):
    if start is None:
        start = ConstantVariable.wrap_literal(0, var.graph)
    elements = [
        var.getitem(ConstantVariable.wrap_literal(i, var.graph))
        for i in range(len(var))
    ]
    result = reduce(
        BuiltinVariable(operator.add, var.graph, DanglingTracker()),
        elements,
        start,
    )
    return result


Dispatcher.register(
    max,
    ("ListVariable",),
    lambda var: var.max(),
)

Dispatcher.register(
    min,
    ("ListVariable",),
    lambda var: var.min(),
)


@Dispatcher.register_decorator(max)
def dispatch_max_star_args(*args: VariableBase):
    if not args:
        raise TypeError("max expected at least 1 arguments, got 0")
    res = args[0]
    graph = res.graph
    for arg in args:
        gt = BuiltinVariable(operator.gt, graph, DanglingTracker())(arg, res)
        if gt.get_py_value() is True:
            res = arg
    return res


@Dispatcher.register_decorator(min)
def dispatch_min_star_args(*args: VariableBase):
    if not args:
        raise TypeError("min expected at least 1 arguments, got 0")
    res = args[0]
    graph = res.graph
    for arg in args:
        lt = BuiltinVariable(operator.lt, graph, DanglingTracker())(arg, res)
        if lt.get_py_value() is True:
            res = arg
    return res


# math functions, e.g. math.log, math.sqrt, math.sin, etc.
def get_math_unary_functions():
    unary_fns = []
    for name, fn in inspect.getmembers(math, inspect.isbuiltin):
        try:
            signature = inspect.signature(fn)
        except ValueError:
            continue
        if len(signature.parameters.keys()) != 1:
            continue
        param = next(iter(signature.parameters.values()))
        if param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.POSITIONAL_ONLY,
        ):
            unary_fns.append(fn)
    return unary_fns


for fn in get_math_unary_functions():
    Dispatcher.register(
        fn,
        ("ConstantVariable",),
        partial(
            lambda fn, var: ConstantVariable(
                fn(var.get_py_value()),
                var.graph,
                tracker=DummyTracker([var]),
            ),
            fn,
        ),
    )
Dispatcher.register(
    math.log,
    ("ConstantVariable",),
    lambda var: ConstantVariable(
        math.log(var.get_py_value()),
        var.graph,
        tracker=DummyTracker([var]),
    ),
)


def constant_numpy_equal(left, right):
    numpy_ans = left.get_py_value() == right.get_py_value()
    return NumpyVariable(
        numpy_ans,
        left.graph,
        tracker=DummyTracker([left, right]),
    )


Dispatcher.register(
    operator.eq,
    ("ConstantVariable", "NumpyVariable"),
    lambda left, right: constant_numpy_equal(left, right),
)

Dispatcher.register(
    bool,
    ("NumpyVariable",),
    lambda x: ConstantVariable(
        bool(x.get_py_value()),
        x.graph,
        tracker=DummyTracker([x]),
    ),
)
