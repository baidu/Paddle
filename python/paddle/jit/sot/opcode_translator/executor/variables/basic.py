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
import sys
import types
from functools import cached_property, reduce
from typing import TYPE_CHECKING, Any

import numpy as np

import paddle
from paddle.framework import core

from ....infer_meta import (
    MetaInfo,
    SymbolicBool,
    SymbolicFloat,
    SymbolicInt,
    SymbolicValue,
)
from ....symbolic.statement_ir import Symbol
from ....utils import (
    ENV_SOT_ALLOW_DYNAMIC_SHAPE,
    BreakGraphError,
    ConstTypes,
    FallbackError,
    NameGenerator,
    get_tensor_methods,
    log,
    printable,
)
from ....utils.envs import ENV_SOT_BREAK_GRAPH_ON_GET_SYMBOLIC_VALUE
from ....utils.exceptions import HasNoAttributeError, InnerError
from ..dispatch_functions import tensor_numel
from ..guard import (
    FasterStringifiedExpression,
    StringifiedExpression,
    check_guard,
    object_equal_stringified_guard,
    stringify_pyobject,
    union_free_vars,
)
from ..mutable_data import MutableDictLikeData
from ..tracker import (
    ConstTracker,
    DanglingTracker,
    DummyTracker,
    GetAttrTracker,
    GetItemTracker,
    GetIterTracker,
    GlobalTracker,
    SymbolicOperationTracker,
    Tracker,
)
from .base import VariableBase, VariableFactory

if TYPE_CHECKING:
    from ..function_graph import FunctionGraph
    from ..pycode_generator import PyCodeGen
    from .callable import FunctionVariable


FP_DTYPE_ABBRS = {
    core.DataType.BFLOAT16: "bfloat16",
    core.DataType.FLOAT64: "float64",
    core.DataType.FLOAT32: "float32",
    core.DataType.FLOAT16: "float16",
    core.DataType.FLOAT8_E4M3FN: "float8_e4m3fn",
    core.DataType.FLOAT8_E5M2: "float8_e5m2",
}

CP_DTYPE_ABBRS = {
    core.DataType.COMPLEX64: "complex64",
    core.DataType.COMPLEX128: "complex128",
}

INT_DTYPE_ABBRS = {
    core.DataType.INT8: "int8",
    core.DataType.INT16: "int16",
    core.DataType.INT32: "int32",
    core.DataType.INT64: "int64",
    core.DataType.UINT8: "uint8",
}

DTYPE_ABBRS = {
    **FP_DTYPE_ABBRS,
    **CP_DTYPE_ABBRS,
    **INT_DTYPE_ABBRS,
    core.DataType.BOOL: "bool",
}

STATIC_DIM_FREQ_THRESHOLD = 5


def method_to_reverse_method(method_name: str) -> str | None:
    if not method_name.startswith("__") or not method_name.endswith("__"):
        return None
    name = method_name[2:-2]
    return f"__r{name}__"


class ConstantVariable(VariableBase):
    """
    ConstantVariable is a subclass of VariableBase used to wrap a Variable of the const type.

    Args:
        value(Any): The value to be wrapped.
        tracker(Tracker): The Tracker object that tracks the information of this variable.
    """

    def __init__(
        self,
        value: Any,
        graph: FunctionGraph,
        tracker: Tracker,
    ):
        super().__init__(graph, tracker)
        self.value = value

    def get_py_value(self, allow_tensor=False):
        return self.value

    @property
    def debug_name(self) -> str:
        return f"{self.value}"

    @debug_name.setter
    def debug_name(self, name):
        pass

    def _reconstruct(self, codegen: PyCodeGen):
        codegen.gen_load_const(self.value)

    @property
    def main_info(self) -> dict[str, Any]:
        return {"value": self.value}

    def __bool__(self) -> bool:
        return bool(self.value)

    def bool(self):
        return ConstantVariable(bool(self), self.graph, DummyTracker([self]))

    def bool_not(self):
        assert isinstance(
            self.get_py_value(), bool
        ), "Bool_not can only be applied to a bool variable."
        return ConstantVariable(
            not bool(self.get_py_value()), self.graph, DummyTracker([self])
        )

    def getitem(self, key):
        track_vars: list[VariableBase] = [self]
        if self.get_py_type() is not str:
            raise InnerError(
                f"getitem can only be applied to a str variable, but got {self.get_py_type()}"
            )
        if isinstance(key, VariableBase):
            track_vars.append(key)
            key = key.get_py_value()
        retval = self.value[key]
        return ConstantVariable(retval, self.graph, DummyTracker(track_vars))

    def str(self):
        return ConstantVariable(
            str(self.value), self.graph, DummyTracker([self])
        )

    def format(self, *args):
        return ConstantVariable(
            str(self.value).format(*[str(a.value) for a in args]),
            self.graph,
            DummyTracker([self, *args]),
        )

    def lower(self):
        return ConstantVariable(
            str(self.value).lower(),
            self.graph,
            DummyTracker([self]),
        )

    def ord(self):
        return ConstantVariable(
            ord(self.value),
            self.graph,
            DummyTracker([self]),
        )

    def chr(self):
        return ConstantVariable(
            chr(self.value),
            self.graph,
            DummyTracker([self]),
        )

    @VariableFactory.register_from_value()
    def from_value(value: Any, graph: FunctionGraph, tracker: Tracker):
        if type(value) in ConstTypes:
            return ConstantVariable(value, graph, tracker)
        return None

    @staticmethod
    def wrap_literal(value: Any, graph: FunctionGraph) -> ConstantVariable:
        """
        Wrap a literal value in a ConstantVariable.

        Args:
            value(Any): The literal value to be wrapped.

        Returns:
            ConstantVariable: A new ConstantVariable object that wraps the given value.
        """
        if isinstance(value, ConstantVariable):
            return value
        assert isinstance(
            value, ConstTypes
        ), f"value: {value},type: {type(value)}"
        return ConstantVariable(value, graph, ConstTracker(value))


class PrintStmtVariable(VariableBase):
    def __init__(self, value: Any, graph: FunctionGraph):
        # TODO: graph should be not None
        super().__init__(None, DanglingTracker())
        self.args, self.kwargs = value
        self.graph = graph

    def _reconstruct(self, codegen: PyCodeGen):
        # do we need ? may be too strict.
        for var in self.args:
            self.graph.add_global_guarded_variable(var)
        for var in self.kwargs.values():
            self.graph.add_global_guarded_variable(var)
        # currently don't consider kwargs
        codegen.gen_load_global("print", push_null=True)
        for var in self.args:
            var.reconstruct(codegen)
        codegen.gen_call_function(len(self.args))
        codegen.gen_pop_top()

    def flatten_inner_vars(self):
        return self.args.flatten_inner_vars()


IMPLEMENTED_TENSOR_PROPERTIES = set()


def tensor_property(func):
    IMPLEMENTED_TENSOR_PROPERTIES.add(func.__name__)
    return property(func)


class DataVariable(VariableBase):
    """
    A value only object.
    If it's all magic method don't change the function_graph state, [tensor op, guard, side_effect]
    we will call it a ValueObjectVariable, we directly call python operator on it.
    """

    def __init__(
        self,
        value: Any,
        graph: FunctionGraph,
        tracker: Tracker,
    ):
        super().__init__(graph, tracker)
        self.value = value

    def get_py_value(self, allow_tensor=False):
        return self.value


class TensorDtypeVariable(DataVariable):
    def __init__(self, value, graph, tracker):
        super().__init__(value, graph, tracker)

    @check_guard
    def make_stringified_guard(self) -> list[StringifiedExpression]:
        if isinstance(self.tracker, GetAttrTracker) and isinstance(
            self.tracker.obj, TensorVariable
        ):
            tensor_value_tracer = (
                self.tracker.obj.tracker.trace_value_from_frame()
            )
            dtype_str, dtype_free_vars = stringify_pyobject(self.value)
            # TODO(cleanup-legacy-ir): Remove this branch after we remove legacy IR
            if not paddle.framework.use_pir_api():
                return [
                    StringifiedExpression(
                        f"MetaInfo.from_tensor({{}}).dtype == {dtype_str}",
                        [tensor_value_tracer],
                        union_free_vars(
                            {"MetaInfo": MetaInfo},
                            tensor_value_tracer.free_vars,
                            dtype_free_vars,
                        ),
                    )
                ]
            return [
                FasterStringifiedExpression(
                    f"{{}}.dtype == {dtype_str}",
                    paddle.framework.core.DtypeMatchGuard(self.value),
                    [tensor_value_tracer],
                    union_free_vars(
                        tensor_value_tracer.free_vars,
                        dtype_free_vars,
                    ),
                )
            ]

        else:
            return object_equal_stringified_guard(self)

    def get_py_value(self, allow_tensor=False):
        return super().get_py_value(allow_tensor)

    def get_py_type(self):
        return super().get_py_type()

    def _reconstruct(self, codegen: PyCodeGen):
        # dtype of paddle.Tensor is hashable, we can just load it as const var
        codegen.gen_load_const(self.value)

    @property
    def main_info(self) -> dict[str, Any]:
        return {
            "dtype": self.value,
        }

    @VariableFactory.register_from_value()
    def from_value(value: Any, graph: FunctionGraph, tracker: Tracker):
        if isinstance(
            value, (paddle.core.VarDesc.VarType, paddle.core.DataType)
        ):
            return TensorDtypeVariable(value, graph, tracker)


class TensorVariable(VariableBase):
    """
    TensorVariable is a subclass of VariableBase used to wrap a Variable of the tensor type.

    Args:
        tensor (paddle.Tensor | MetaInfo): The tensor to be wrapped.
        graph (FunctionGraph): The FunctionGraph object that this variable is associated with.
        tracker (Tracker): The Tracker object that tracks the information of this variable.
    """

    var_name_generator = NameGenerator("var_")
    mutable_attrs = ["meta"]

    def __init__(
        self,
        meta: MetaInfo,
        graph: FunctionGraph,
        tracker: Tracker,
    ):
        super().__init__(graph, tracker)
        self.value = None
        self.meta = meta
        dynamic_axes: list[int] = []
        if ENV_SOT_ALLOW_DYNAMIC_SHAPE.get() and self.tracker.is_traceable():
            dynamic_axes = self.analyse_dynamic_axes(tracker)
        self.meta = self.meta.with_dynamic_axes(dynamic_axes)
        self.origin_meta = self.meta
        self.var_name = TensorVariable.var_name_generator.next()
        self.graph.side_effects.record_mutable_variable(self)

    def analyse_dynamic_axes(self, tracker: Tracker):
        from ..executor_cache import OpcodeExecutorCache

        shape_dims = (
            self.shape.proxy.get_all()
        )  # Trigger convert all shape dims to Variable
        dynamic_axes = [
            i
            for i, dim in enumerate(shape_dims)
            if isinstance(dim, SymbolicVariable)
        ]
        if dynamic_axes:
            tracker_expr = tracker.trace_value_from_frame().inlined_expr
            symbolic_inputs = OpcodeExecutorCache().get_symbolic_inputs(
                self.graph.pycode_gen._origin_code
            )
            log(
                1,
                f"Start analyse dynamic axes for {tracker.trace_value_from_frame().inlined_expr} in {self.graph.pycode_gen._origin_code}\n",
            )
            for key, symbolic_input in symbolic_inputs.items():
                if key.startswith(tracker_expr):
                    log(1, f"  {key}: {symbolic_input}\n")
            log(
                1,
                f"  -> Tensor {tracker_expr} with dynamic axes {dynamic_axes}\n",
            )
        return dynamic_axes

    def __len__(self):
        if isinstance(self.meta.shape[0], SymbolicInt):
            raise BreakGraphError(
                "length of tensor variable with first dimension is dynamic shape causes graph break."
            )
        return self.meta.shape[0]

    def get_py_value(self, allow_tensor=False):
        if allow_tensor:

            class SotTensor:
                def __init__(self, id_):
                    self.id = id_

                def __eq__(self, var):
                    if not hasattr(var, "id"):
                        return False
                    else:
                        return self.id == var.id

                def __hash__(self):
                    return hash(self.id)

            return SotTensor(self.id)

        raise BreakGraphError(
            "Called TensorVariable.get_py_value. Should not use Tensor's value in simulating."
        )

    def get_py_type(self):
        return paddle.Tensor

    def get_symbol(self) -> Symbol:
        return Symbol(self.var_name)

    @property
    def out_var_name(self):
        return f"{self.graph.OUT_VAR_PREFIX}{self.var_name}"

    def _reconstruct(self, codegen: PyCodeGen):
        codegen.gen_load_fast(self.out_var_name)

    @check_guard
    def make_stringified_guard(self) -> list[StringifiedExpression]:
        frame_value_tracer = self.tracker.trace_value_from_frame()

        # TODO(cleanup-legacy-ir): Remove this branch after we remove legacy IR
        if not paddle.framework.use_pir_api():
            if ENV_SOT_ALLOW_DYNAMIC_SHAPE.get():
                str_left_expr = f"MetaInfo.from_tensor({{}}, dynamic_axes={self.meta.dynamic_axes}).guard_str()"
            else:
                str_left_expr = "MetaInfo.from_tensor({}).guard_str()"
            return [
                StringifiedExpression(
                    f"{str_left_expr} == '{self.origin_meta.guard_str()}'",
                    [frame_value_tracer],
                    union_free_vars(
                        {"MetaInfo": MetaInfo},
                        frame_value_tracer.free_vars,
                    ),
                )
            ]

        # A quick check path for PIR, we don't need dtype conversion for AMP in PIR
        meta = self.origin_meta
        dtype_str, dtype_free_vars = stringify_pyobject(meta.dtype)
        return [
            # Check rank
            StringifiedExpression(
                f"len({{}}.shape) == {len(meta.shape)}",
                [frame_value_tracer],
                union_free_vars(frame_value_tracer.free_vars),
            ),
            # Check each dim except dynamic dim
            *[
                StringifiedExpression(
                    f"{{}}.shape[{i}] == {meta.shape[i]}",
                    [frame_value_tracer],
                    union_free_vars(frame_value_tracer.free_vars),
                )
                for i in range(len(meta.shape))
                if not isinstance(meta.shape[i], SymbolicInt)
            ],
            # Check dtype
            StringifiedExpression(
                f"{{}}.dtype == {dtype_str}",
                [frame_value_tracer],
                union_free_vars(frame_value_tracer.free_vars, dtype_free_vars),
            ),
            # Check stop_gradient
            StringifiedExpression(
                f"{{}}.stop_gradient == {meta.stop_gradient!r}",
                [frame_value_tracer],
                union_free_vars(frame_value_tracer.free_vars),
            ),
        ]

    def get_iter(self):
        from .iter import SequenceIterVariable

        return SequenceIterVariable(self, self.graph, GetIterTracker(self))

    @property
    def main_info(self) -> dict[str, Any]:
        dtype = self.meta.dtype
        if isinstance(dtype, paddle.core.VarDesc.VarType):
            dtype = paddle.pir.core.vartype_to_datatype[dtype]
        return {
            "shape": self.meta.shape,
            "dtype": DTYPE_ABBRS[dtype],
            "stop_gradient": self.meta.stop_gradient,
            "var_name": self.var_name,
            "dist_info": self.meta.dist_info,
        }

    def getitem(self, key):
        return self.graph.call_tensor_method("__getitem__", self, key)

    def setitem(self, key, value):
        new_tensor = self.graph.call_paddle_api(
            paddle.static.setitem,
            self,
            key,
            value,
        )

        self.meta = new_tensor.meta
        self.graph.add_inplace_tensors(self)

    @tensor_property
    def T(self):
        """
        Return a new TensorVariable object that wraps the result of calling the transpose method on the wrapped value of this TensorVariable.
        """
        from .container import ListVariable

        perm = list(range(len(self.meta.shape) - 1, -1, -1))
        perm_var = ListVariable(perm, self.graph, tracker=ConstTracker(perm))
        out = self.graph.call_paddle_api(paddle.transpose, self, perm_var)
        return out

    @tensor_property
    def mT(self):
        """
        Return a new TensorVariable object that wraps the result of calling the mT method on the wrapped value of this TensorVariable.
        """
        from .container import ListVariable

        if len(self.meta.shape) < 2:
            raise ValueError(
                f"Variable.ndim({self.ndim}) is required to be greater than or equal to 2."
            )

        perm = list(range(len(self.meta.shape)))
        perm[-1], perm[-2] = perm[-2], perm[-1]
        perm_var = ListVariable(perm, self.graph, tracker=DummyTracker([self]))
        out = self.graph.call_paddle_api(paddle.transpose, self, perm_var)
        return out

    @tensor_property
    def ndim(self):
        """
        Return a ConstantVariable object that represents the number of dimensions of the wrapped value of this TensorVariable.
        """
        return ConstantVariable(
            len(self.meta.shape), self.graph, DummyTracker([self])
        )

    @tensor_property
    def size(self):
        """
        Return a ConstantVariable object that represents the total number of elements in the wrapped value of this TensorVariable.
        """
        # TODO: maybe break graph.
        if self.meta.is_dynamic_shape():
            raise BreakGraphError(
                f"Getting size for a dynamic shape tensor causes graph break. shape = {self.meta.shape}"
            )
        elements = reduce(operator.mul, self.meta.shape, 1)
        return ConstantVariable(elements, self.graph, DummyTracker([self]))

    @tensor_property
    def shape(self):
        if (
            not ENV_SOT_ALLOW_DYNAMIC_SHAPE.get()
            and self.meta.is_dynamic_shape()
        ):
            raise BreakGraphError(
                f"Getting shape for a dynamic shape tensor causes graph break. shape = {self.meta.shape}"
            )
        from .container import ListVariable

        tracker = GetAttrTracker(self, "shape")
        return ListVariable(self.meta.shape, self.graph, tracker=tracker)

    def numel(self):
        return self.size

    def len(self):
        if len(self.meta.shape) == 0:
            raise InnerError("len() of a 0-D tensor is wrong")
        first_dim = self.meta.shape[0]
        if isinstance(first_dim, SymbolicInt):
            raise BreakGraphError(
                "Getting len() for a dynamic shape tensor causes graph break."
            )

        return ConstantVariable(first_dim, self.graph, DummyTracker([self]))

    def is_tensor(self):
        return ConstantVariable(True, self.graph, DummyTracker([self]))

    def is_complex(self):
        dtype = self.meta.dtype
        if isinstance(dtype, paddle.core.VarDesc.VarType):
            dtype = paddle.pir.core.vartype_to_datatype[dtype]
        is_cp_dtype = dtype in CP_DTYPE_ABBRS
        return ConstantVariable(is_cp_dtype, self.graph, DummyTracker([self]))

    def is_integer(self):
        dtype = self.meta.dtype
        if isinstance(dtype, paddle.core.VarDesc.VarType):
            dtype = paddle.pir.core.vartype_to_datatype[dtype]
        is_int_dtype = dtype in INT_DTYPE_ABBRS
        return ConstantVariable(is_int_dtype, self.graph, DummyTracker([self]))

    def is_floating_point(self):
        dtype = self.meta.dtype
        if isinstance(dtype, paddle.core.VarDesc.VarType):
            dtype = paddle.pir.core.vartype_to_datatype[dtype]
        is_fp_dtype = dtype in FP_DTYPE_ABBRS
        return ConstantVariable(is_fp_dtype, self.graph, DummyTracker([self]))

    def getattr(self, name: str, default=None):
        if default is not None:
            raise FallbackError(
                "default argument for getattr is not implemented"
            )
        method_name_to_builtin_fn = {
            "dim": paddle.rank,
            "numel": tensor_numel,
            "ndimension": paddle.rank,
            "is_tensor": paddle.is_tensor,
            "is_complex": paddle.is_complex,
            "is_integer": paddle.is_integer,
            "is_floating_point": paddle.is_floating_point,
        }
        if name in ["name", "place", "type"] and self.meta.is_inner_var():
            raise BreakGraphError(
                f"{self.meta.name} is a middle tensor. get {name} property."
            )
        if name in [
            "dtype",
            "type",
            "name",
            "persistable",
            "stop_gradient",
            "place",
        ]:
            return VariableFactory.from_value(
                getattr(self.meta, name),
                self.graph,
                tracker=GetAttrTracker(self, name),
            )
        elif name in IMPLEMENTED_TENSOR_PROPERTIES:
            return getattr(self, name)
        elif name in method_name_to_builtin_fn:
            # TODO: backward, gradient
            from .callable import BuiltinVariable

            builtin_fn = method_name_to_builtin_fn[name]

            return BuiltinVariable(
                builtin_fn, self.graph, DanglingTracker()
            ).bind(self, name)
        elif name in get_tensor_methods():
            from .callable import TensorFunctionVariable

            fn_var = TensorFunctionVariable(
                name, graph=self.graph, tracker=DanglingTracker()
            )
            return fn_var.bind(self, name)
        else:
            raise HasNoAttributeError(f"Unknown Tensor attribute: {name}")

    def setattr(self, key, val):
        # support tensor variable store attr, like:
        # t.stop_gradient = True
        self.graph.call_tensor_method(
            "__setattr__",
            self,
            VariableFactory().from_value(key, self.graph, ConstTracker(key)),
            val,
        )

    def delattr(self, key):
        raise BreakGraphError("Don't support TensorVariable delattr")

    @VariableFactory.register_from_value()
    def from_value(value: Any, graph: FunctionGraph, tracker: Tracker):
        if isinstance(value, (paddle.Tensor, MetaInfo)):
            value = (
                MetaInfo.from_tensor(value)
                if isinstance(value, paddle.Tensor)
                else value
            )
            return TensorVariable(value, graph, tracker)
        return None


def get_symbolic_from_meta(meta: MetaInfo) -> SymbolicValue:
    if meta.dtype in [paddle.bool]:
        value = SymbolicBool()
    elif meta.dtype in [
        paddle.int8,
        paddle.uint8,
        paddle.int16,
        paddle.int32,
        paddle.int64,
    ]:
        value = SymbolicInt()
    elif meta.dtype in [
        paddle.bfloat16,
        paddle.float16,
        paddle.float32,
        paddle.float64,
        paddle.float8_e4m3fn,
        paddle.float8_e5m2,
    ]:
        value = SymbolicFloat()
    else:
        raise InnerError(f"Unsupported dtype {meta.dtype} for SymbolicVariable")
    return value


class SymbolicVariable(VariableBase):
    """
    SymbolicVariable is a subclass of VariableBase used to wrap a symbolic value.

    Args:
        value_or_meta (int | SymbolicInt | MetaInfo): The symbolic value  to be wrapped or metadata.
        graph (FunctionGraph): The FunctionGraph object that this variable is associated with.
        tracker (Tracker): The Tracker object that tracks the information of this variable.
    """

    var_name_generator = NameGenerator("symint_")
    value: int | SymbolicValue
    mutable_attrs = ["need_guard_value"]

    def __init__(
        self,
        value_or_meta: int | SymbolicInt | MetaInfo,
        graph: FunctionGraph,
        tracker: Tracker,
    ):
        super().__init__(graph, tracker)
        self.var_name = self.var_name_generator.next()
        if isinstance(value_or_meta, MetaInfo):
            assert len(value_or_meta.shape) == 0
            self.value = get_symbolic_from_meta(value_or_meta)
            self.meta = value_or_meta
        else:
            self.value = value_or_meta
            self.meta = MetaInfo(
                [], paddle.int64, True, self.var_name, False, None, None
            )
        self.need_guard_value = False
        self.graph.side_effects.record_mutable_variable(self)

    def to_constant(self):
        from ..executor_cache import (
            OpcodeExecutorCache,
        )

        symbolic_inputs = OpcodeExecutorCache().get_symbolic_inputs(
            self.graph.pycode_gen._origin_code
        )

        disabled_vars = set()

        def disable_symbolic(var: VariableBase):
            if var in disabled_vars:
                return

            disabled_vars.add(var)
            if var.tracker.is_traceable():
                tracker_expr = var.tracker.trace_value_from_frame().inlined_expr
                symbolic_inputs[tracker_expr] = None
                return
            for input_var in var.tracker.inputs:
                disable_symbolic(input_var)

        disable_symbolic(self)
        self.graph.need_cache = False
        log(3, f"Fallback {self} to ConstantVariable")
        return ConstantVariable(
            self.get_py_value(), self.graph, DummyTracker([self])
        )

    def get_py_value(self, allow_tensor: bool = False) -> bool | int | float:
        if ENV_SOT_BREAK_GRAPH_ON_GET_SYMBOLIC_VALUE.get():
            raise BreakGraphError("get_py_value from SymbolicVariable")
        self.need_guard_value = True
        log(
            3,
            f"get_py_value from SymbolicVariable {self} caused value need guard",
        )
        if isinstance(self.value, SymbolicValue):
            assert isinstance(
                self.tracker, SymbolicOperationTracker
            ), f"self.value is None, but tracker is not SymbolicOperationTracker. tracker: {self.tracker}"
            inputs = self.tracker.inputs
            assert len(inputs) >= 1
            input_values = [x.get_py_value() for x in inputs]
            value = getattr(input_values[0], self.tracker.method_name)(
                *input_values[1:]
            )
            # TODO(SigureMo): A Temporary solution for the case that the method is not implemented.
            # e.g. In user code, we have `1 * 0.1`, the lhs is a SymbolicVariable, and the rhs is a float.
            # We trace the method `__mul__` from the lhs, but actually, python use `float.__rmul__`,
            # `int.__mul__(float)` is not implemented. So we get NotImplemented here.
            # We need to find a better way to handle this case.
            if isinstance(value, type(NotImplemented)):
                reversed_method = method_to_reverse_method(
                    self.tracker.method_name
                )
                if reversed_method is None:
                    raise InnerError(
                        f"Unsupported method {self.tracker.method_name} for SymbolicVariable"
                    )
                value = getattr(input_values[1], reversed_method)(
                    input_values[0], *input_values[2:]
                )
            self.value = value
            assert isinstance(
                self.value, (bool, int, float)
            ), f"SymbolicVariable.get_py_value() should return bool, int or float, but got {type(self.value)}"
        return self.value

    def get_py_type(self):
        if isinstance(self.value, int):
            return int
        return self.value.get_static_type()

    def get_symbol(self) -> Symbol:
        return Symbol(self.var_name)

    def __bool__(self) -> bool:
        return bool(self.get_py_value())

    def bool(self):
        return ConstantVariable(bool(self), self.graph, DummyTracker([self]))

    def __int__(self) -> int:
        return int(self.get_py_value())

    def int(self):
        return ConstantVariable(int(self), self.graph, DummyTracker([self]))

    def __float__(self) -> float:
        return float(self.get_py_value())

    def float(self):
        return ConstantVariable(float(self), self.graph, DummyTracker([self]))

    def __complex__(self) -> complex:
        return complex(self.get_py_value())

    def complex(self):
        return ConstantVariable(complex(self), self.graph, DummyTracker([self]))

    @property
    def out_var_name(self):
        return f"{self.graph.OUT_VAR_PREFIX}{self.var_name}"

    def _reconstruct(self, codegen: PyCodeGen):
        codegen.gen_load_fast(self.out_var_name)
        codegen.gen_load_method("item")
        codegen.gen_call_method(0)  # TODO

    @check_guard
    def make_stringified_guard(self) -> list[StringifiedExpression]:
        assert ENV_SOT_ALLOW_DYNAMIC_SHAPE.get()
        from ..executor_cache import OpcodeExecutorCache

        frame_value_tracer = self.tracker.trace_value_from_frame()
        symbolic_inputs = OpcodeExecutorCache().get_symbolic_inputs(
            self.graph.pycode_gen._origin_code
        )

        assert frame_value_tracer.inlined_expr in symbolic_inputs

        if self.need_guard_value:
            return super().make_stringified_guard()
        return [
            FasterStringifiedExpression(
                f"id(type({{}})) == {id(self.get_py_type())}",
                paddle.core.TypeMatchGuard(self.get_py_type()),
                [frame_value_tracer],
                union_free_vars(frame_value_tracer.free_vars),
            ),
        ]

    @staticmethod
    def should_create_symbolic_variable(
        value: Any,
        tracker: Tracker,
        symbolic_inputs: dict[str, dict[int, int] | None],
    ):
        tracker_expr = tracker.trace_value_from_frame().inlined_expr
        symbolic_inputs.setdefault(tracker_expr, {})
        if tracker_expr in symbolic_inputs:
            symbolic_input = symbolic_inputs[tracker_expr]
            if symbolic_input is None:
                return False
            symbolic_input.setdefault(value, 0)
            symbolic_input[value] += 1
            if symbolic_input[value] >= STATIC_DIM_FREQ_THRESHOLD:
                return False
            if len(symbolic_input.keys()) > 1:
                return True
            return False
        return False

    @staticmethod
    def find_tensor_shape_source(dim_tracker: Tracker):
        from .container import ListVariable

        if not isinstance(dim_tracker, GetItemTracker):
            return None
        if not isinstance(dim_tracker.container, ListVariable):
            return None
        if not isinstance(dim_tracker.container.tracker, GetAttrTracker):
            return None
        if dim_tracker.container.tracker.attr != "shape":
            return None
        if not isinstance(dim_tracker.container.tracker.obj, TensorVariable):
            return None
        tensor_var = dim_tracker.container.tracker.obj
        shape_idx = dim_tracker.key
        return tensor_var, shape_idx

    @VariableFactory.register_from_value(successor="ConstantVariable")
    def from_value(value: Any, graph: FunctionGraph, tracker: Tracker):
        if not ENV_SOT_ALLOW_DYNAMIC_SHAPE.get():
            return None
        if isinstance(value, SymbolicInt):
            tensor_shape_source_result = (
                SymbolicVariable.find_tensor_shape_source(tracker)
            )
            assert tensor_shape_source_result is not None
            tensor_var, shape_idx = tensor_shape_source_result
            tensor_call_shape_var = graph.call_paddle_api(
                paddle.shape, tensor_var
            )
            return graph.call_symbolic_method(
                "__getitem__",
                tensor_call_shape_var,
                ConstantVariable.wrap_literal(shape_idx, graph),
            )
        if type(value) is not int:
            return None
        if not tracker.is_traceable():
            return None

        from ..executor_cache import OpcodeExecutorCache

        symbolic_inputs = OpcodeExecutorCache().get_symbolic_inputs(
            graph.pycode_gen._origin_code
        )

        if SymbolicVariable.should_create_symbolic_variable(
            value, tracker, symbolic_inputs
        ):
            return SymbolicVariable(value, graph, tracker)
        return None


class ParameterVariable(TensorVariable):
    def __init__(
        self,
        meta: MetaInfo,
        graph: FunctionGraph,
        tracker: Tracker,
    ):
        super().__init__(meta, graph, tracker)

    @VariableFactory.register_from_value(successor="TensorVariable")
    def from_value(value: Any, graph: FunctionGraph, tracker: Tracker):
        if isinstance(value, (paddle.base.framework.EagerParamBase)):
            value = MetaInfo.from_tensor(value)
            return ParameterVariable(value, graph, tracker)
        return None


class ObjectVariable(VariableBase):
    """
    ObjectVariable is a subclass of VariableBase used to wrap a Variable of the object type.

    Args:
        obj(Any): The object to be wrapped.
        graph(FunctionGraph): The FunctionGraph object that this variable is associated with.
        tracker(Tracker): The Tracker object that tracks the information of this variable.
    """

    make_stringified_guard = object_equal_stringified_guard

    def __init__(self, obj, graph, tracker):
        super().__init__(graph, tracker)
        self.value = obj

    @property
    def main_info(self) -> dict[str, Any]:
        # NOTE(SigureMo): There are some objects that cannot be printed, such as
        # uninitialized dataclass, we should fallback to the class name.
        if printable(self.value):
            return {"value": self.value}
        else:
            return {"value": f"instance {self.value.__class__.__name__}"}

    def get_py_value(self, allow_tensor=False) -> Any:
        return self.value


class SliceVariable(VariableBase):
    """
    SliceVariable is a subclass of VariableBase used to wrap a Variable of the slice type.

    Args:
        slice_(slice): The slice to be wrapped.
        graph(FunctionGraph): The FunctionGraph object that this variable is associated with.
        tracker(Tracker): The Tracker object that tracks the information of this variable.
    """

    def __init__(self, slice_: slice, graph, tracker):
        super().__init__(graph, tracker)
        self.value = slice_

    @property
    def debug_name(self) -> str:
        return ":".join(
            [
                str(self.value.start) if self.value.start is not None else "",
                str(self.value.stop) if self.value.stop is not None else "",
                str(self.value.step) if self.value.step is not None else "",
            ]
        )

    @debug_name.setter
    def debug_name(self, name):
        pass

    @cached_property
    def attr_proxy(self):
        return self.graph.side_effects.get_proxy(
            MutableDictLikeData, self.value, self.attr_proxy_getter
        )

    @property
    def main_info(self) -> dict[str, Any]:
        return {"value": self.value}

    def get_py_value(self, allow_tensor=False):
        return slice(
            self.getattr("start").get_py_value(),
            self.getattr("stop").get_py_value(),
            self.getattr("step").get_py_value(),
        )

    @check_guard
    def make_stringified_guard(self) -> list[StringifiedExpression]:
        frame_value_tracer = self.tracker.trace_value_from_frame()
        result = [
            FasterStringifiedExpression(
                "id(type({{}})) == id(slice)",
                paddle.framework.core.TypeMatchGuard(slice),
                [frame_value_tracer],
                frame_value_tracer.free_vars,
            ),
            *self.getattr("start").make_stringified_guard(),
            *self.getattr("stop").make_stringified_guard(),
            *self.getattr("step").make_stringified_guard(),
        ]
        return result

    def _reconstruct(self, codegen: PyCodeGen):
        if all(
            isinstance(x, ConstantVariable)
            for x in [
                self.getattr("start"),
                self.getattr("stop"),
                self.getattr("step"),
            ]
        ):
            self.graph.add_global_guarded_variable(self)
            self.getattr("start").reconstruct(codegen)
            self.getattr("stop").reconstruct(codegen)
            self.getattr("step").reconstruct(codegen)
            codegen.gen_build_slice(3)
        else:
            super()._reconstruct(codegen)

    def setattr(self, key, val):
        raise BreakGraphError("Don't support SliceVariable setattr")

    def delattr(self, key):
        raise BreakGraphError("Don't support SliceVariable delattr")

    @VariableFactory.register_from_value()
    def from_value(value: Any, graph: FunctionGraph, tracker: Tracker):
        if isinstance(value, slice):
            return SliceVariable(value, graph, tracker)
        return None


class ModuleVariable(VariableBase):
    """
    ModuleVariable is a subclass of VariableBase used to wrap a Variable of the module type.

    Args:
        func: The module to be wrapped.
        graph: The FunctionGraph object that this variable is associated with.
        tracker: The Tracker object that tracks the information of this variable.
    """

    def __init__(self, func, graph, tracker):
        super().__init__(graph, tracker)
        self.value = func

    def get_py_value(self, allow_tensor=False):
        return self.value

    @property
    def main_info(self) -> dict[str, Any]:
        return {"value": self.value}

    @VariableFactory.register_from_value()
    def from_value(value: Any, graph: FunctionGraph, tracker: Tracker):
        if isinstance(value, types.ModuleType):
            return ModuleVariable(value, graph, tracker)
        return None

    # Happened in a inline import statement.
    make_stringified_guard = object_equal_stringified_guard


class DygraphTracerVariable(VariableBase):
    # TODO(SigureMo): Remove this trick after we add CompareTracker
    def __init__(self, value, graph, tracker):
        super().__init__(graph, tracker)
        self.value = value

    def get_py_value(self, allow_tensor=False):
        return self.value

    @check_guard
    def make_stringified_guard(self) -> list[StringifiedExpression]:
        return []

    @property
    def main_info(self) -> dict[str, Any]:
        return {
            "is_none": self.value is None,
        }

    @VariableFactory.register_from_value()
    def from_value(value: Any, graph: FunctionGraph, tracker: Tracker):
        if isinstance(value, paddle.base.dygraph.tracer.Tracer):
            return DygraphTracerVariable(value, graph, tracker)
        return None


class NumpyVariable(VariableBase):
    """
    NumpyVariable is a subclass of VariableBase used to wrap a Variable of the numpy type.

    Args:
        value: The numpy value to be wrapped.
        graph: The FunctionGraph object that this variable is associated with.
        tracker: The Tracker object that tracks the information of this variable.
    """

    def __init__(self, value, graph, tracker):
        super().__init__(graph, tracker)
        self.value = value

    @property
    def main_info(self) -> dict[str, Any]:
        return {"value": self.value}

    def get_py_value(self, allow_tensor=False) -> Any:
        return self.value

    @check_guard
    def make_stringified_guard(self) -> list[StringifiedExpression]:
        if isinstance(self.get_py_value(), np.number):
            frame_value_tracer = self.tracker.trace_value_from_frame()

            def format_dtype(dtype: np.dtype):
                return f"np.{dtype}"

            def format_number(number: np.number):
                return f"{format_dtype(number.dtype)}({number.item()})"

            return [
                StringifiedExpression(
                    f"{{}} == {format_number(self.get_py_value())}",
                    [frame_value_tracer],
                    union_free_vars(frame_value_tracer.free_vars, {"np": np}),
                ),
                StringifiedExpression(
                    f"{{}}.dtype == {format_dtype(self.get_py_value().dtype)}",
                    [frame_value_tracer],
                    union_free_vars(frame_value_tracer.free_vars, {"np": np}),
                ),
            ]
        else:
            return object_equal_stringified_guard(self)

    @VariableFactory.register_from_value()
    def from_value(value: Any, graph: FunctionGraph, tracker: Tracker):
        if isinstance(value, (np.ndarray, np.number)):
            return NumpyVariable(value, graph, tracker)
        return None


class NullVariable(VariableBase):
    """
    NullVariable is a subclass of VariableBase used to represent a placeholder variable that has no value or reference associated with it.
    """

    def __init__(self):
        # TODO: graph should be not None
        super().__init__(None, DanglingTracker())

    def __call__(self, *args, **kwargs):
        func = args[0]
        assert callable(func)
        return func(*args[1:], **kwargs)

    def reconstruct(self, codegen: PyCodeGen):
        if sys.version_info >= (3, 13):
            codegen.gen_push_null()
            return
        codegen.gen_load_null_variable()


class CellVariable(VariableBase):
    def __init__(self, value=None):
        # TODO: graph should be not None
        super().__init__(
            None, DanglingTracker()
        )  # should reconstruct cell variable
        assert isinstance(value, (VariableBase, type(None)))
        self.set_value(value)

    def reconstruct(
        self,
        codegen: PyCodeGen,
        *,
        use_tracker: bool = True,
        add_to_global_guarded_vars: bool = True,
    ):
        raise FallbackError("Break graph in closure is not support.")

    def cell_content(self):
        return self.value

    def set_value(self, value):
        self.value = value

    def empty(self):
        return self.value is None


class GlobalVariable(VariableBase):
    def __init__(
        self,
        val_dict,
        graph: FunctionGraph,
        tracker: Tracker,
    ):
        super().__init__(graph, tracker)
        self.proxy = self.graph.side_effects.get_proxy(
            MutableDictLikeData, val_dict, self.proxy_getter
        )

    def proxy_getter(self, proxy: MutableDictLikeData, key: Any):
        if key not in proxy.original_data:
            return MutableDictLikeData.Empty()
        return VariableFactory.from_value(
            proxy.original_data[key],
            self.graph,
            tracker=GlobalTracker(key),
        )

    def get_value(self):
        return dict(self.proxy.get_all().items())

    def keys(self):
        return self.proxy.get_all().keys()

    def get(self, key):
        if isinstance(key, VariableBase):
            raise InnerError(
                f"[{self.__class__.__name__}]: received {key} to get value."
            )
        return self.proxy.get(key)

    def set(self, key, value):
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

    def delete(self, key):
        self.proxy.delete(key)
        self.graph.side_effects.record_proxy_variable(self)


class FunctionGlobalVariable(GlobalVariable):
    def __init__(
        self,
        fn: FunctionVariable,
        val_dict: dict[str, Any],
        graph: FunctionGraph,
        tracker: Tracker,
    ):
        super().__init__(val_dict, graph, tracker)
        self.fn = fn

    def proxy_getter(self, proxy: MutableDictLikeData, key: Any):
        from ..opcode_inline_executor import FunctionGlobalTracker

        if key not in proxy.original_data:
            return MutableDictLikeData.Empty()
        return VariableFactory.from_value(
            proxy.original_data[key],
            self.graph,
            tracker=FunctionGlobalTracker(self.fn, key),
        )
