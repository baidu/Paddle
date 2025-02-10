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

from typing import TYPE_CHECKING, Any

from paddle.jit.sot.opcode_translator.executor.variables.base import (
    VariableBase,
)

from ....utils import BreakGraphError, FallbackError
from ..tracker import ConstTracker, DummyTracker
from .base import VariableFactory
from .basic import ConstantVariable
from .container import ContainerVariable, TupleVariable

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..function_graph import FunctionGraph
    from ..pycode_generator import PyCodeGen
    from ..tracker import Tracker


class IterVariable(VariableBase):
    """
    This Variable (include subclasses) should be generated only when simulate GET_ITER opcode
    """

    def __init__(
        self, obj: VariableBase, graph: FunctionGraph, tracker: Tracker
    ):
        super().__init__(graph, tracker)
        self.hold = obj

    def make_stringified_guard(self):
        return self.hold.make_stringified_guard()

    def next(self):
        raise NotImplementedError(f"Can not simulate `next` for {type(self)}")

    def get_iter(self):
        return self

    def flatten_inner_vars(self) -> list[VariableBase]:
        holds = self.hold
        if not isinstance(holds, list):
            holds = [holds]
        return [
            inner_var
            for hold in holds
            for inner_var in hold.flatten_inner_vars()
        ]


class SequenceIterVariable(IterVariable):
    """
    The basic SequenceIterVariable wraps iterators which can be simulated by call getitem
    Currently includes: List | Tuple | Dict (keys) | Range | Tensor | nn.LayerList

    these interfaces is needed:
    - next
    - to_list
    - has_side_effect
    - _reconstruct
    """

    mutable_attrs = ["idx"]

    def __init__(self, obj, graph: FunctionGraph, tracker: Tracker):
        super().__init__(obj, graph, tracker)
        self.idx = 0
        self.graph.side_effects.record_mutable_variable(self)

    def next(self):
        # TODO: self.hold should have a __len__ method
        if self.idx < len(self.hold):
            val = self.hold[self.idx]
            self.idx += 1
            return val
        else:
            raise StopIteration

    def to_list(self) -> list:
        if self.has_side_effect():
            raise FallbackError("Can not convert an used iterator into list")
        self.idx = len(self.hold)
        retval = []
        for i in range(len(self.hold)):
            retval.append(self.hold[i])
        return retval

    def has_side_effect(self) -> bool:
        return self.idx != 0

    def _reconstruct(self, codegen: PyCodeGen):
        if self.has_side_effect():
            super()._reconstruct(codegen)
        else:
            self.hold.reconstruct(codegen)
            codegen.gen_get_iter()

    @property
    def main_info(self) -> dict[str, Any]:
        return {
            "idx": self.idx,
        }


class EnumerateVariable(SequenceIterVariable):
    """
    EnumerateVariable holds a SequenceIterVariable and return additional index
    """

    def __init__(self, val_iterator, graph, tracker):
        super().__init__(val_iterator, graph, tracker)

    def next(self):
        val = self.hold.next()
        idx_var = ConstantVariable(self.idx, self.graph, ConstTracker(self.idx))
        self.idx += 1
        return TupleVariable(
            (idx_var, val), self.graph, DummyTracker([idx_var, val])
        )

    def to_list(self):
        values = self.hold.to_list()
        idx = [
            ConstantVariable(i, self.graph, ConstTracker(i))
            for i in range(len(values))
        ]
        return list(zip(idx, values))

    def has_side_effect(self) -> bool:
        return self.hold.has_side_effect()

    def _reconstruct(self, codegen: PyCodeGen):
        if self.has_side_effect():
            super()._reconstruct(codegen)
        else:
            codegen.gen_load_global("enumerate", push_null=True)
            self.hold.reconstruct(codegen)
            codegen.gen_call_function(1)

    @staticmethod
    def from_iterator(value, graph: FunctionGraph | None, tracker: Tracker):
        iter_variable = value.get_iter()
        if isinstance(iter_variable, SequenceIterVariable):
            return EnumerateVariable(iter_variable, graph, tracker)
        else:
            return UserDefinedIterVariable(value, graph, tracker)


class ZipVariable(SequenceIterVariable):
    """
    ZipVariable holds a list of SequenceIterVariable
    """

    def __init__(self, iters, graph, tracker):
        super().__init__(iters, graph, tracker)

    def next(self):
        # can not use <listcomp> here, because it will raise a RuntimeError("StopIteration")
        # but we want a StopIteration Exception
        values = []
        for iter_var in self.hold:
            next_var = iter_var.next()
            values.append(next_var)

        return VariableFactory.from_value(
            tuple(values), self.graph, DummyTracker(values)
        )

    def to_list(self):
        lists = [iter_vars.to_list() for iter_vars in self.hold]
        min_len = min(len(l) for l in lists)
        result = []
        for i in range(min_len):
            result.append(tuple(l[i] for l in lists))
        return result

    def has_side_effect(self) -> bool:
        return any(iter_var.has_side_effect() for iter_var in self.hold)

    def _reconstruct(self, codegen: PyCodeGen):
        if self.has_side_effect():
            super()._reconstruct(codegen)
        else:
            codegen.gen_load_global("zip", push_null=True)
            for iter_var in self.hold:
                iter_var.reconstruct(codegen)
            codegen.gen_call_function(len(self.hold))

    @staticmethod
    def from_iterator(
        value: Sequence[VariableBase],
        graph: FunctionGraph | None,
        tracker: Tracker,
    ):
        assert isinstance(value, (list, tuple))
        zip_targets = []

        for variable in value:
            iter_variable = variable.get_iter()
            if not isinstance(iter_variable, SequenceIterVariable):
                return UserDefinedIterVariable(value, graph, tracker)
            zip_targets.append(iter_variable)

        return ZipVariable(zip_targets, graph, tracker)


class MapVariable(SequenceIterVariable):
    """
    MapVariable holds a SequenceIterVariable and return a Iterable Variable after map function
    """

    def __init__(self, func, val_iterator, graph, tracker):
        super().__init__(val_iterator, graph, tracker)
        self.func = func

    def next(self):
        return self.func(self.hold.next())

    def to_list(self) -> list:
        retval = []
        while True:
            try:
                retval.append(self.func(self.hold.next()))
            except StopIteration:
                break
        return retval

    def has_side_effect(self) -> bool:
        return self.hold.has_side_effect()

    def _reconstruct(self, codegen: PyCodeGen):
        if self.has_side_effect():
            super()._reconstruct(codegen)
        else:
            codegen.gen_load_global("map", push_null=True)
            self.func.reconstruct(codegen)
            self.hold.reconstruct(codegen)
            codegen.gen_call_function(2)

    @staticmethod
    def from_iterator(
        func, value, graph: FunctionGraph | None, tracker: Tracker
    ):
        iter_variable = (
            value.get_iter() if isinstance(value, ContainerVariable) else value
        )

        if isinstance(iter_variable, IterVariable):
            return MapVariable(func, iter_variable, graph, tracker)
        else:
            return UserDefinedIterVariable(value, graph, tracker)


# what UserDefinedIterVariable holds doesn't matter, because use user defined iterator will trigger break graph
class UserDefinedIterVariable(IterVariable):
    def __init__(self, obj, graph, tracker):
        super().__init__(obj, graph, tracker)

    def next(self):
        raise BreakGraphError("Break graph when using user defined iterator")
