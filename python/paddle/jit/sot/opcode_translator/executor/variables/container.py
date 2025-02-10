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

import operator
from collections import OrderedDict
from functools import reduce
from typing import TYPE_CHECKING, Any

import paddle
from paddle.jit.sot.opcode_translator.executor.variables.base import (
    VariableBase,
)

from ....utils import ConstTypes
from ....utils.exceptions import FallbackError, InnerError
from ..dispatcher import Dispatcher
from ..guard import (
    FasterStringifiedExpression,
    StringifiedExpression,
    check_guard,
)
from ..mutable_data import MutableDictLikeData, MutableListLikeData
from ..tracker import (
    ConstTracker,
    DanglingTracker,
    DummyTracker,
    GetAttrTracker,
    GetItemTracker,
    GetIterTracker,
    Tracker,
)
from .base import VariableFactory
from .basic import ConstantVariable
from .callable import BuiltinVariable, UserDefinedFunctionVariable

if TYPE_CHECKING:
    from ..function_graph import FunctionGraph
    from ..pycode_generator import PyCodeGen


class ContainerVariable(VariableBase):
    """
    ContainerVariable is a wrapper for container types, such as range, list, tuple, dict.
    """

    @property
    def init_value(self):
        return self.value

    def get_wrapped_items(self):
        raise FallbackError(
            "ContainerVariable.get_wrapped_items do not implement"
        )

    def __len__(self) -> int:
        raise FallbackError('ContainerVariable.__len__ do not implement')

    def len(self):
        return ConstantVariable(len(self), self.graph, DummyTracker([self]))

    def __bool__(self) -> bool:
        return len(self) > 0

    def bool(self):
        return ConstantVariable(bool(self), self.graph, DummyTracker([self]))

    def flatten_inner_vars(self) -> list[VariableBase]:
        items = []
        for item in self.get_wrapped_items():
            items.extend(item.flatten_inner_vars())
        return items

    @check_guard
    def make_stringified_guard(self) -> list[StringifiedExpression]:
        frame_value_tracer = self.tracker.trace_value_from_frame()

        if self.get_py_type() is dict:
            type_guard = FasterStringifiedExpression(
                f"isinstance({{}}, {self.get_py_type().__name__})",
                paddle.framework.core.InstanceCheckGuard(self.get_py_type()),
                [frame_value_tracer],
                frame_value_tracer.free_vars,
            )
        else:
            type_guard = FasterStringifiedExpression(
                f"id(type({{}})) == {id(self.get_py_type())}",
                paddle.framework.core.TypeMatchGuard(self.get_py_type()),
                [frame_value_tracer],
                frame_value_tracer.free_vars,
            )
        len_guard = FasterStringifiedExpression(
            f"len({{}}) == {len(self.init_value)}",
            paddle.framework.core.LengthMatchGuard(len(self.init_value)),
            [frame_value_tracer],
            frame_value_tracer.free_vars,
        )
        if isinstance(self, (ListVariable, TupleVariable)):
            guard_variables = self.proxy.reproduce(0)

        elif isinstance(self, DictVariable):
            guard_variables = filter(
                lambda var: not isinstance(var, MutableDictLikeData.Empty),
                self.proxy.reproduce(0).values(),
            )
        else:
            raise InnerError(f"Unsupported container type: {type(self)}")
        return reduce(
            operator.add,
            [[type_guard, len_guard]]
            + [
                item.make_stringified_guard()
                for item in guard_variables
                if item.tracker.need_guard()
            ],
        )


class ListVariable(ContainerVariable):
    """
    ListVariable is a wrapper for list and contains common APIs for list methods

    Args:
        val_list(List[VariableBase]): the list to wrap
        graph(FunctionGraph): The FunctionGraph object that this variable is associated with.
        tracker(Tracker): The Tracker object that tracks the information of this variable.
    """

    def __init__(
        self,
        val_list: list[Any],
        graph: FunctionGraph,
        tracker: Tracker,
    ):
        super().__init__(graph, tracker)

        # everything in stack is VariableBase, so just accept the input list is ok
        self.proxy = self.graph.side_effects.get_proxy(
            MutableListLikeData, val_list, self.proxy_getter
        )
        self.value = val_list

    def proxy_getter(self, proxy: MutableListLikeData, key: Any):
        if key < 0 or key >= len(proxy.original_data):
            return MutableListLikeData.Empty()
        return VariableFactory.from_value(
            proxy.original_data[key],
            self.graph,
            tracker=GetItemTracker(self, key, changed=proxy.has_changed),
        )

    def get_py_value(self, allow_tensor=False):
        items = self.proxy.get_all()
        return [item.get_py_value(allow_tensor) for item in items]

    def get_py_type(self):
        return list

    def _reconstruct(self, codegen: PyCodeGen):
        size = len(self)
        for idx in range(size):
            Dispatcher.call(operator.getitem, self, idx).reconstruct(codegen)
        codegen.gen_build_list(size)

    def get_wrapped_items(self):
        size = len(self)
        return [
            Dispatcher.call(operator.getitem, self, idx) for idx in range(size)
        ]

    def get_iter(self):
        from .iter import SequenceIterVariable

        return SequenceIterVariable(self, self.graph, GetIterTracker(self))

    @property
    def main_info(self) -> dict[str, Any]:
        return {
            "len": len(self),
        }

    def __len__(self):
        return self.proxy.length

    def getitem(self, key):
        self.graph.add_global_guarded_variable(key)
        key = key.get_py_value()
        if isinstance(key, int):
            res = self.proxy.get(key)
            if self.proxy.is_empty(res):
                raise InnerError(f"List {self} out of range (index={key})")
            return res
        elif isinstance(key, slice):
            items = self.proxy.get_all()
            return VariableFactory.from_value(
                items[key],
                self.graph,
                tracker=GetItemTracker(
                    self, key, changed=self.proxy.has_changed
                ),
            )
        else:
            raise InnerError(
                f"Unsupported key type {key.__class__.__name__} for ListVariable"
            )

    def setitem(self, key, value):
        if not isinstance(value, VariableBase):
            raise InnerError(
                f"[{self.__class__.__name__}]: received {value} to set value."
            )
        if isinstance(key, int):
            self.proxy.set(key, value)
        elif isinstance(key, slice) and isinstance(
            value, (ListVariable, TupleVariable)
        ):
            start, end, step = key.indices(self.proxy.length)
            indices = list(range(start, end, step))
            if step == 1:
                # replace a continuous range
                for i, idx in enumerate(indices):
                    self.proxy.delete(idx - i)
                for i, item in enumerate(value.get_wrapped_items()):
                    self.proxy.insert(start + i, item)
            else:
                # replace some elements
                if len(indices) != len(value):
                    raise InnerError(
                        f"Attempt to replace {len(indices)} items with {len(value)}"
                    )
                for i, idx in enumerate(indices):
                    self.proxy.set(idx, value[i])
        else:
            raise InnerError(
                f"Unsupported key type {key.__class__.__name__} and value type {value.__class__.__name__} for ListVariable"
            )

        self.graph.side_effects.record_proxy_variable(self)
        return ConstantVariable.wrap_literal(None, self.graph)

    def __delitem__(self, key):
        return self.delitem(key)

    def delitem(self, key):
        if isinstance(key, VariableBase):
            raise InnerError(
                f"[{self.__class__.__name__}]: received {key} as key to delete."
            )
        self.proxy.delete(key)
        self.graph.side_effects.record_proxy_variable(self)
        return ConstantVariable.wrap_literal(None, self.graph)

    def insert(self, index: int, value: VariableBase):
        self.proxy.insert(index, value)
        self.graph.side_effects.record_proxy_variable(self)
        return ConstantVariable.wrap_literal(None, self.graph)

    def append(self, value: VariableBase):
        self.insert(self.proxy.length, value)
        self.graph.side_effects.record_proxy_variable(self)
        return ConstantVariable.wrap_literal(None, self.graph)

    def extend(self, data):
        for item in data.get_iter().to_list():
            self.append(item)
        self.graph.side_effects.record_proxy_variable(self)
        return ConstantVariable.wrap_literal(None, self.graph)

    def concat(self, list_):
        assert isinstance(list_, ListVariable)
        return ListVariable(
            self.proxy.get_all() + list_.proxy.get_all(),
            self.graph,
            DummyTracker([self, list_]),
        )

    def inplace_concat(self, list_):
        assert isinstance(list_, ListVariable)
        self.extend(list_)
        return self

    def repeat(self, length):
        assert isinstance(length, ConstantVariable)
        return ListVariable(
            self.proxy.get_all() * length.value,
            self.graph,
            DummyTracker([self, length]),
        )

    def pop(self, index: ConstantVariable | None = None):
        if index is None:
            index = ConstantVariable.wrap_literal(-1, self.graph)
        res = self.proxy.get(index.get_py_value())
        self.proxy.delete(index.get_py_value())
        self.graph.side_effects.record_proxy_variable(self)
        return res

    def copy(self):
        return ListVariable(
            self.proxy.get_all(),
            self.graph,
            DummyTracker([self]),
        )

    def clear(self):
        for idx in range(self.proxy.length):
            self.delitem(0)
        self.graph.side_effects.record_proxy_variable(self)
        return ConstantVariable.wrap_literal(None, self.graph)

    def remove(self, value):
        for idx in range(self.proxy.length):
            if self[idx].get_py_value(allow_tensor=True) == value.get_py_value(
                allow_tensor=True
            ):
                self.delitem(idx)
                break
        else:
            raise InnerError(f"List {self} does not contain {value}")
        self.graph.side_effects.record_proxy_variable(self)
        return ConstantVariable.wrap_literal(None, self.graph)

    def sort(self, key=None, reverse=None):
        if (
            key is None
            or isinstance(key, ConstantVariable)
            and key.get_py_value() is None
        ):
            key = UserDefinedFunctionVariable(
                lambda x: x, self.graph, DanglingTracker()
            )
            assert key is not None
        if reverse is None:
            reverse = ConstantVariable.wrap_literal(False, self.graph)

        permutation = list(range(self.proxy.length))
        permutation.sort(
            key=lambda x: key.get_py_value()(
                Dispatcher.call(operator.getitem, self, x).value
            ),
            reverse=reverse.get_py_value(),
        )
        self.proxy.permutate(permutation)
        self.graph.side_effects.record_proxy_variable(self)
        return ConstantVariable.wrap_literal(None, self.graph)

    def reverse(self):
        permutation = list(range(self.proxy.length))
        permutation.reverse()
        self.proxy.permutate(permutation)
        self.graph.side_effects.record_proxy_variable(self)
        return ConstantVariable.wrap_literal(None, self.graph)

    def count(self, value: VariableBase):
        count: int = 0
        getitem = BuiltinVariable(
            operator.getitem, self.graph, DanglingTracker()
        )
        for index in range(len(self)):
            index_value = getitem(self, index)
            if index_value.id == value.id:
                count += 1
                continue
            eq = BuiltinVariable(operator.eq, self.graph, DanglingTracker())(
                index_value, value
            )
            eq_bool = BuiltinVariable(bool, self.graph, DanglingTracker())(eq)
            assert isinstance(
                eq_bool, ConstantVariable
            ), "bool should return ConstantVariable"
            if eq.get_py_value() is True:
                count += 1
                continue

        return ConstantVariable(count, self.graph, DummyTracker([self, value]))

    def index(self, value: VariableBase):
        res = 0
        getitem = BuiltinVariable(
            operator.getitem, self.graph, DanglingTracker()
        )
        for index in range(len(self)):
            index_value = getitem(self, index)
            if index_value.id == value.id:
                return ConstantVariable(
                    res, self.graph, DummyTracker([self, value])
                )
            eq = BuiltinVariable(operator.eq, self.graph, DanglingTracker())(
                index_value, value
            )
            eq_bool = BuiltinVariable(bool, self.graph, DanglingTracker())(eq)
            assert isinstance(
                eq_bool, ConstantVariable
            ), "bool should return ConstantVariable"
            if eq.get_py_value() is True:
                return ConstantVariable(
                    res, self.graph, DummyTracker([self, value])
                )
            res += 1

        return ConstantVariable(-1, self.graph, DummyTracker([self, value]))

    def max(self):
        if len(self) == 0:
            raise ValueError("max() arg is an empty sequence")
        res = self[0]
        getitem = BuiltinVariable(
            operator.getitem, self.graph, DanglingTracker()
        )
        for index in range(len(self)):
            index_value = getitem(self, index)
            gt = BuiltinVariable(operator.gt, self.graph, DanglingTracker())(
                index_value, res
            )
            if gt.get_py_value() is True:
                res = index_value
        return res

    def min(self):
        if len(self) == 0:
            raise ValueError("min() arg is an empty sequence")
        res = self[0]
        getitem = BuiltinVariable(
            operator.getitem, self.graph, DanglingTracker()
        )
        for index in range(len(self)):
            index_value = getitem(self, index)
            lt = BuiltinVariable(operator.lt, self.graph, DanglingTracker())(
                index_value, res
            )
            if lt.get_py_value() is True:
                res = index_value
        return res

    def getattr(self, name: str, default=None):
        from .callable import BuiltinVariable

        if default is not None:
            raise FallbackError(
                "default argument for getattr is not implemented"
            )

        method_name_to_builtin_fn = {
            "insert": list.insert,
            "append": list.append,
            "extend": list.extend,
            "pop": list.pop,
            "copy": list.copy,
            "clear": list.clear,
            "remove": list.remove,
            "sort": list.sort,
            "reverse": list.reverse,
            "count": list.count,
            "index": list.index,
        }

        if name in method_name_to_builtin_fn:
            builtin_fn = method_name_to_builtin_fn[name]
            return BuiltinVariable(
                builtin_fn, self.graph, DanglingTracker()
            ).bind(self, name)
        else:
            raise FallbackError(f"attribute {name} for list is not implemented")

    @VariableFactory.register_from_value()
    def from_value(value: Any, graph: FunctionGraph, tracker: Tracker):
        # Note(SigureMo): Why not use isinstance?
        # Because user may define a class that inherit from list.
        # We should convert it to ObjectVariable instead of ListVariable.
        if type(value) is list:
            return ListVariable(value, graph=graph, tracker=tracker)
        return None


class TupleVariable(ContainerVariable):
    """
    TupleVariable is a wrapper for tuple and contains common APIs for tuple methods.

    Args:
        val_tuple(tuple[VariableBase, ...]): the tuple to wrap
        graph(FunctionGraph): The FunctionGraph object that this variable is associated with.
        tracker(Tracker): The Tracker object that tracks the information of this variable.
    """

    def __init__(
        self,
        val_tuple: tuple[VariableBase, ...],
        graph: FunctionGraph,
        tracker: Tracker,
    ):
        super().__init__(graph, tracker)

        self.proxy = self.graph.side_effects.get_proxy(
            MutableListLikeData, list(val_tuple), self.proxy_getter
        )
        self.value = val_tuple

    def getattr(self, name: str, default=None):
        from .callable import BuiltinVariable

        if default is not None:
            raise FallbackError(
                "default argument for getattr is not implemented"
            )

        method_name_to_builtin_fn = {
            "count": tuple.count,
            "index": tuple.index,
        }
        if name in method_name_to_builtin_fn:
            builtin_fn = method_name_to_builtin_fn[name]
            return BuiltinVariable(
                builtin_fn, self.graph, DanglingTracker()
            ).bind(self, name)
        else:
            raise FallbackError(
                f"attribute {name} for tuple is not implemented"
            )

    def proxy_getter(self, proxy: MutableListLikeData, key: Any):
        if key < 0 or key >= len(proxy.original_data):
            return MutableListLikeData.Empty()
        return VariableFactory.from_value(
            proxy.original_data[key],
            self.graph,
            tracker=GetItemTracker(self, key, changed=False),
        )

    def get_py_value(self, allow_tensor=False):
        return tuple(
            self[idx].get_py_value(allow_tensor) for idx in range(len(self))
        )

    def get_py_type(self):
        return tuple

    def _reconstruct(self, codegen: PyCodeGen):
        size = len(self)
        for idx in range(size):
            Dispatcher.call(operator.getitem, self, idx).reconstruct(codegen)
        codegen.gen_build_tuple(size)

    def get_wrapped_items(self):
        size = len(self)
        return tuple(
            Dispatcher.call(operator.getitem, self, idx) for idx in range(size)
        )

    def get_iter(self):
        from .iter import SequenceIterVariable

        return SequenceIterVariable(self, self.graph, GetIterTracker(self))

    @property
    def main_info(self) -> dict[str, Any]:
        return {
            "len": len(self),
        }

    def __len__(self):
        return self.proxy.length

    def getitem(self, key):
        self.graph.add_global_guarded_variable(key)
        key = key.get_py_value()
        if isinstance(key, int):
            res = self.proxy.get(key)
            if self.proxy.is_empty(res):
                raise InnerError(f"List {self} out of range (index={key})")
            return res
        elif isinstance(key, slice):
            return TupleVariable(
                tuple(self.proxy.get_all())[key],
                self.graph,
                tracker=GetItemTracker(self, key, changed=False),
            )
        else:
            raise InnerError(
                f"Unsupported key type {key.__class__.__name__} for TupleVariable"
            )

    def setitem(self, key, value):
        raise InnerError(
            f"[{self.__class__.__name__}]: setitem is not allowed."
        )

    def __delitem__(self, key):
        return self.delitem(key)

    def delitem(self, key):
        raise InnerError(
            f"[{self.__class__.__name__}]: delitem is not allowed."
        )

    def concat(self, tuple_):
        assert isinstance(tuple_, TupleVariable)
        new_tuple_variable = TupleVariable(
            tuple(self.proxy.get_all() + tuple_.proxy.get_all()),
            self.graph,
            DummyTracker([self, tuple_]),
        )
        return new_tuple_variable

    def repeat(self, length):
        assert isinstance(length, ConstantVariable)
        new_tuple_variable = TupleVariable(
            tuple(self.proxy.get_all()) * length.value,
            self.graph,
            DummyTracker([self, length]),
        )
        return new_tuple_variable

    def count(self, value: VariableBase):
        count: int = 0
        getitem = BuiltinVariable(
            operator.getitem, self.graph, DanglingTracker()
        )
        for index in range(len(self)):
            index_value = getitem(self, index)
            if index_value.id == value.id:
                count += 1
                continue
            eq = BuiltinVariable(operator.eq, self.graph, DanglingTracker())(
                index_value, value
            )
            eq_bool = BuiltinVariable(bool, self.graph, DanglingTracker())(eq)
            assert isinstance(
                eq_bool, ConstantVariable
            ), "bool should return ConstantVariable"
            if eq.get_py_value() is True:
                count += 1
                continue

        return ConstantVariable(count, self.graph, DummyTracker([self, value]))

    def index(self, value: VariableBase):
        res = 0
        getitem = BuiltinVariable(
            operator.getitem, self.graph, DanglingTracker()
        )
        for index in range(len(self)):
            index_value = getitem(self, index)
            if index_value.id == value.id:
                return ConstantVariable(
                    res, self.graph, DummyTracker([self, value])
                )
            eq = BuiltinVariable(operator.eq, self.graph, DanglingTracker())(
                index_value, value
            )
            eq_bool = BuiltinVariable(bool, self.graph, DanglingTracker())(eq)
            assert isinstance(
                eq_bool, ConstantVariable
            ), "bool should return ConstantVariable"
            if eq.get_py_value() is True:
                return ConstantVariable(
                    res, self.graph, DummyTracker([self, value])
                )
            res += 1

        return ConstantVariable(-1, self.graph, DummyTracker([self, value]))

    @VariableFactory.register_from_value()
    def from_value(value: Any, graph: FunctionGraph, tracker: Tracker):
        if type(value) is tuple:
            return TupleVariable(value, graph, tracker)
        return None


class RangeVariable(ContainerVariable):
    """
    RangeVariable is a wrapper for range.

    Args:
        val_range(range): the range to wrap
        graph(FunctionGraph): The FunctionGraph object that this variable is associated with.
        tracker(Tracker): The Tracker object that tracks the information of this variable.
    """

    def __init__(
        self,
        start: VariableBase,
        stop: VariableBase,
        step: VariableBase,
        graph: FunctionGraph,
        tracker: Tracker,
    ):
        super().__init__(graph, tracker)
        self.start = start
        self.stop = stop
        self.step = step

    def get_py_type(self):
        return range

    def get_py_value(self, allow_tensor=False):
        return range(
            self.start.get_py_value(),
            self.stop.get_py_value(),
            self.step.get_py_value(),
        )

    def getitem(self, key):
        self.graph.add_global_guarded_variable(key)
        key = key.get_py_value()
        retval = self.get_py_value()[key]
        return ConstantVariable(retval, self.graph, GetItemTracker(self, key))

    def get_items(self):
        return [self.start, self.stop, self.step]

    def get_wrapped_items(self):
        size = len(self)
        return [self[idx] for idx in range(size)]

    def get_iter(self):
        from .iter import SequenceIterVariable

        return SequenceIterVariable(self, self.graph, GetIterTracker(self))

    def __len__(self):
        return len(self.get_py_value())

    def _reconstruct(self, codegen: PyCodeGen):
        codegen.gen_load_global("range", push_null=True)
        self.start.reconstruct(codegen)
        self.stop.reconstruct(codegen)
        self.step.reconstruct(codegen)
        codegen.gen_call_function(3)

    @VariableFactory.register_from_value()
    def from_value(value: Any, graph: FunctionGraph, tracker: Tracker):
        if type(value) is range:
            range_variable = RangeVariable(None, None, None, graph, tracker)
            start = VariableFactory.from_value(
                value.start, graph, GetAttrTracker(range_variable, "start")
            )
            stop = VariableFactory.from_value(
                value.stop, graph, GetAttrTracker(range_variable, "stop")
            )
            step = VariableFactory.from_value(
                value.step, graph, GetAttrTracker(range_variable, "step")
            )
            range_variable.__init__(start, stop, step, graph, tracker)
            return range_variable
        return None

    @check_guard
    def make_stringified_guard(self) -> list[StringifiedExpression]:
        frame_value_tracer = self.tracker.trace_value_from_frame()
        return [
            StringifiedExpression(
                "isinstance({0}, range)",
                [frame_value_tracer],
                frame_value_tracer.free_vars,
            ),
            *self.start.make_stringified_guard(),
            *self.stop.make_stringified_guard(),
            *self.step.make_stringified_guard(),
        ]

    @property
    def main_info(self) -> dict[str, Any]:
        return {
            "start": self.start,
            "stop": self.stop,
            "step": self.step,
        }


class DictVariable(ContainerVariable):
    """
    DictVariable is a wrapper for dict and contains common APIs for dict methods

    Args:
        val_dict(dict[object, VariableBase]): the dict to wrap
        graph(FunctionGraph): The FunctionGraph object that this variable is associated with.
        tracker(Tracker): The Tracker object that tracks the information of this variable.
    """

    def __init__(
        self,
        val_dict: dict[object, VariableBase],
        graph: FunctionGraph,
        tracker: Tracker,
    ):
        super().__init__(graph, tracker)

        self.proxy = self.graph.side_effects.get_proxy(
            MutableDictLikeData, val_dict, self.proxy_getter
        )
        self.value = val_dict

    def proxy_getter(self, proxy: MutableDictLikeData, key: Any):
        if key not in proxy.original_data:
            return MutableDictLikeData.Empty()
        return VariableFactory.from_value(
            proxy.original_data[key],
            self.graph,
            tracker=GetItemTracker(self, key, changed=proxy.has_changed),
        )

    def get_py_value(self, allow_tensor=False):
        return {
            key: value.get_py_value(allow_tensor)
            for key, value in self.proxy.get_all().items()
        }

    def get_py_type(self):
        return dict

    def _reconstruct(self, codegen: PyCodeGen):
        from .basic import ConstantVariable

        size = len(self)
        for key in self.proxy.get_all().keys():
            if not isinstance(key, ConstTypes):
                raise InnerError(
                    f"[{self.__class__.__name__}]: received {key} as key."
                )
            key_var = ConstantVariable.wrap_literal(key, self.graph)
            value_var = self[key]
            key_var.reconstruct(codegen)
            value_var.reconstruct(codegen)
        codegen.gen_build_map(size)

    def flatten_inner_vars(self):
        items = self.get_wrapped_items()
        return [
            inner_var
            for key in items.keys()
            for key_var in [
                VariableFactory.from_value(
                    key, self.graph, tracker=ConstTracker(key)
                )
            ]
            for inner_var in (
                key_var.flatten_inner_vars()
                + self[key_var].flatten_inner_vars()
            )
        ]

    def get_wrapped_items(self):
        items = {}
        for key in self.proxy.get_all().keys():
            if not isinstance(key, ConstTypes):
                raise InnerError(
                    f"[{self.__class__.__name__}]: received {key} as key."
                )
            items[key] = self[key]
        return items

    def get_iter(self):
        return self.keys()

    @property
    def main_info(self) -> dict[str, Any]:
        return {
            "len": len(self),
        }

    def __len__(self):
        return len(self.proxy.get_all())

    def get(self, key, default=None):
        if isinstance(key, VariableBase):
            raise InnerError(
                f"[{self.__class__.__name__}]: received {key} to get value."
            )

        if default is None:
            return Dispatcher.call(operator.getitem, self, key)

        if isinstance(self.proxy.get(key), MutableDictLikeData.Empty):
            assert isinstance(default, VariableBase)
            return default

        return Dispatcher.call(operator.getitem, self, key)

    def getitem(self, key):
        self.graph.add_global_guarded_variable(key)
        key = key.get_py_value()
        return self.proxy.get(key)

    def setitem(self, key, value):
        if isinstance(key, VariableBase):
            raise InnerError(
                f"[{self.__class__.__name__}]: received {key} as key."
            )

        if not isinstance(value, VariableBase):
            raise InnerError(
                f"[{self.__class__.__name__}]: received {value} to set value."
            )

        self.proxy.set(key, value)
        self.graph.side_effects.record_proxy_variable(self)

        return ConstantVariable.wrap_literal(None, self.graph)

    def clear(self):
        # TODO: Replace with self.proxy.clear()
        for key in self.value:
            self.delitem(key)
        self.graph.side_effects.record_proxy_variable(self)
        return ConstantVariable.wrap_literal(None, self.graph)

    def __delitem__(self, key):
        return self.delitem(key)

    def delitem(self, key):
        if isinstance(key, VariableBase):
            raise InnerError(
                f"[{self.__class__.__name__}]: received {key} as key to delete."
            )
        self.proxy.delete(key)
        self.graph.side_effects.record_proxy_variable(self)
        return ConstantVariable.wrap_literal(None, self.graph)

    def keys(self):
        from .iter import SequenceIterVariable

        raw_list = [
            ConstantVariable(x, self.graph, ConstTracker(x))
            for x in self.proxy.get_all().keys()
        ]
        key_list = ListVariable(raw_list, self.graph, DummyTracker(raw_list))
        assert key_list is not None
        return SequenceIterVariable(
            key_list, self.graph, DummyTracker([key_list])
        )

    def values(self):
        from .iter import SequenceIterVariable

        raw_list = list(self.get_wrapped_items().values())
        value_list = ListVariable(raw_list, self.graph, DummyTracker([self]))
        assert value_list is not None
        return SequenceIterVariable(
            value_list, self.graph, DummyTracker([value_list])
        )

    def items(self):
        from .iter import SequenceIterVariable

        keys = [
            ConstantVariable(x, self.graph, ConstTracker(x))
            for x in self.proxy.get_all().keys()
        ]
        values = list(self.get_wrapped_items().values())
        raw_list = list(zip(keys, values))
        item_list = ListVariable(raw_list, self.graph, DummyTracker([self]))
        assert item_list is not None
        return SequenceIterVariable(
            item_list, self.graph, DummyTracker([item_list])
        )

    def update(self, data: DictVariable):
        for key, value in data.proxy.get_all().items():
            self.setitem(key, value)
        return ConstantVariable.wrap_literal(None, self.graph)

    def copy(self):
        new_dict_variable = DictVariable(
            self.get_wrapped_items(), self.graph, DummyTracker([self])
        )
        return new_dict_variable

    def setdefault(self, key, default=None):
        if isinstance(self.proxy.get(key), MutableDictLikeData.Empty):
            if default is None:
                self.setitem(
                    key, ConstantVariable.wrap_literal(default, self.graph)
                )
            else:
                self.setitem(key, default)

        return Dispatcher.call(operator.getitem, self, key)

    def pop(self, key, default=None):
        if isinstance(self.proxy.get(key), MutableDictLikeData.Empty):
            assert isinstance(default, VariableBase)
            return default

        # default is not None, or key is in dict
        temp_value = Dispatcher.call(operator.getitem, self, key)
        self.delitem(key)
        return temp_value

    def popitem(self):
        key = list(self.proxy.get_all().keys())[-1]
        value = Dispatcher.call(operator.getitem, self, key)
        # TODO: key, value should be VariableBase but key maybe a int
        # assert isinstance(key, VariableBase), key
        # assert isinstance(value, VariableBase), value
        new_tuple_variable = TupleVariable(
            (key, value), self.graph, DummyTracker([self])
        )
        self.delitem(key)
        return new_tuple_variable

    def getattr(self, name: str, default=None):
        from .callable import BuiltinVariable

        if default is not None:
            raise FallbackError(
                "default argument for getattr is not implemented"
            )

        method_name_to_builtin_fn = {
            "keys": dict.keys,
            "values": dict.values,
            "items": dict.items,
            "update": dict.update,
            "setdefault": dict.setdefault,
            "get": dict.get,
            "copy": dict.copy,
            "clear": dict.clear,
            "pop": dict.pop,
            "popitem": dict.popitem,
        }

        if name in method_name_to_builtin_fn:
            builtin_fn = method_name_to_builtin_fn[name]
            return BuiltinVariable(
                builtin_fn, self.graph, DanglingTracker()
            ).bind(self, name)
        else:
            raise FallbackError(f"attribute {name} for dict is not implemented")

    @VariableFactory.register_from_value()
    def from_value(value: Any, graph: FunctionGraph, tracker: Tracker):
        if type(value) in (dict, OrderedDict):
            return DictVariable(value, graph=graph, tracker=tracker)
