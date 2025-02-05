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

import contextlib
import inspect
import re
import sys
from typing import TYPE_CHECKING

import paddle

from ...profiler import event_register
from ...utils import BreakGraphError, log
from ..instruction_utils import Instruction
from .guard import StringifiedExpression, union_free_vars
from .opcode_executor import OpcodeExecutorBase, Stop
from .tracker import ConstTracker, DanglingTracker, DummyTracker, Tracker
from .variables import (
    CellVariable,
    FunctionGlobalVariable,
    IterVariable,
    SequenceIterVariable,
    VariableBase,
)

if TYPE_CHECKING:
    from .pycode_generator import PyCodeGen
    from .variables import FunctionVariable


class FunctionGlobalTracker(Tracker):
    """
    A tracker class that represents a function global variable.

    Args:
        fn: FunctionVariable object.
        name: The name of the global variable.

    """

    def __init__(self, fn: FunctionVariable, name: str):
        super().__init__([fn])
        self.fn = fn
        self.name = name

    def gen_instructions(self, codegen: PyCodeGen):
        """
        Generate bytecode instructions in order to put the variables at the top of the stack.

        Args:
            codegen: The PyCodeGen object used to generate bytecode.

        """
        self.fn.tracker.gen_instructions(codegen)
        codegen.gen_load_attr("__globals__")
        codegen.gen_load_const(self.name)
        codegen.gen_subscribe()

    def guard_tree_expr_node(self) -> paddle.framework.core.ExprNode:
        fn_tracer = self.fn.tracker.guard_tree_expr_node()
        return paddle.framework.core.ItemExprNode(
            paddle.framework.core.AttributeExprNode(
                fn_tracer,
                "__globals__",
            ),
            paddle.framework.core.ConstantExprNode(self.name),
        )

    def trace_value_from_frame(self) -> StringifiedExpression:
        """
        Trace the value of the function global variable from the frame.

        Returns:
            StringifiedExpression: The traced value of the function global variable.

        """
        fn_tracer = self.fn.tracker.trace_value_from_frame()
        return StringifiedExpression(
            f"{{}}.__globals__['{self.name}']",
            [fn_tracer],
            union_free_vars(fn_tracer.free_vars),
        )

    def __repr__(self) -> str:
        return f"FunctionGlobalTracker(fn={self.fn}, name={self.name})"


class FunctionClosureTracker(Tracker):
    """
    A tracker class that represents a function closure variable.

    Args:
        fn: The FunctionVariable object.
        idx: The index of the closure variable.

    """

    def __init__(self, fn: FunctionVariable, idx: int):
        super().__init__([fn])
        self.fn = fn
        self.idx = idx

    def gen_instructions(self, codegen: PyCodeGen):
        """
        Generate bytecode instructions to trace the value of the function closure variable.

        Args:
            codegen: The PyCodeGen object used to generate bytecode.

        """
        self.fn.tracker.gen_instructions(codegen)
        codegen.gen_load_attr("__closure__")
        codegen.gen_load_const(self.idx)
        codegen.gen_subscribe()
        codegen.gen_load_attr("cell_contents")

    def guard_tree_expr_node(self) -> paddle.framework.core.ExprNode:
        # TODO(zrr1999): implement FunctionClosureExprNode
        raise NotImplementedError("FunctionClosureExprNode is not implemented")

    def trace_value_from_frame(self):
        """
        Trace the value of the function closure variable from the frame.

        Returns:
            The traced value of the function closure variable.

        """
        fn_tracer = self.fn.tracker.trace_value_from_frame()
        return StringifiedExpression(
            f"{{}}.__closure__[{self.idx}].cell_contents",
            [fn_tracer],
            union_free_vars(fn_tracer.free_vars),
        )

    def __repr__(self) -> str:
        return f"FunctionClosureTracker(fn={self.fn}, idx={self.idx})"


@contextlib.contextmanager
def signature_clear_guard(fn, name):
    if not hasattr(fn, name):
        yield
    else:
        saved_attr = getattr(fn, name)
        delattr(fn, name)
        yield
        setattr(fn, name, saved_attr)


class OpcodeInlineExecutor(OpcodeExecutorBase):
    """
    A class that represents an executor for inlined opcode operations.

    Args:
        fn_variable: The function variable.

    """

    def __init__(
        self,
        fn_variable: FunctionVariable,
        *args,
        **kwargs,
    ):
        self._fn_var = fn_variable
        self.return_value: VariableBase | None = None
        self._fn_value = fn_variable.value
        self._code_var = fn_variable.get_code()
        super().__init__(self._code_var.value, fn_variable.graph)
        self._name = "Inline"
        self._prepare_locals(*args, **kwargs)
        self._prepare_closure()

    def _handle_comps(self):
        is_comp = any(
            x in self._fn_value.__name__
            for x in ['<listcomp>', '<dictcomp>', '<genexpr>']
        )
        if not is_comp:
            return
        pattern = r'implicit\d+'
        for name in list(self._locals.keys()):
            if re.match(pattern, name):
                self._locals[name.replace('implicit', '.')] = self._locals[name]

    def _prepare_locals(self, *args, **kwargs):
        """
        Prepare local variables for execution by adding them to the locals dictionary.

        """
        from .variables import VariableBase, VariableFactory

        # temparay clear the fn.__signature__ to avoid signature check error
        with signature_clear_guard(
            self._fn_value, "__signature__"
        ), signature_clear_guard(self._fn_value, "__wrapped__"):
            sig = inspect.signature(self._fn_value)
            bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        for name, value in bound_args.arguments.items():
            assert name in sig.parameters
            # Convert varargs and kwargs to Variable
            if sig.parameters[name].kind == inspect.Parameter.VAR_POSITIONAL:
                tracker = DummyTracker(value)
            elif sig.parameters[name].kind == inspect.Parameter.VAR_KEYWORD:
                tracker = DummyTracker(list(value.values()))
            # Convert default args to Variable
            elif not isinstance(value, VariableBase):
                tracker = ConstTracker(value)
            else:
                tracker = value.tracker
            value = VariableFactory.from_value(value, self._graph, tracker)
            self._locals[name] = value

        self._handle_comps()

        log(
            5, f"[INLINE CALL] {self._code.co_name} with locals: ", self._locals
        )

    def _prepare_closure(self):
        """
        Prepare closure variables for execution by adding them to the closure list.

        """
        from .variables import VariableFactory

        closure = self._fn_var.get_py_value().__closure__
        for name in self._code.co_cellvars + self._code.co_freevars:
            # create a cell for each variable.
            self._cells[name] = CellVariable()  # put in cells.
            if name in self._locals:
                self._cells[name].set_value(self._locals[name])

        if closure is None:
            return
        assert len(closure) == len(self._code.co_freevars)
        for idx, (name, cell) in enumerate(
            zip(self._code.co_freevars, closure)
        ):
            value = cell.cell_contents
            value = VariableFactory.from_value(
                value, self._graph, FunctionClosureTracker(self._fn_var, idx)
            )
            # wrapped by a CellVariable
            if not isinstance(value, CellVariable):
                value = CellVariable(value)
            self._cells[name] = value

    @event_register("OpcodeInlineExecutor: _prepare_virtual_env", event_level=2)
    def _prepare_virtual_env(self):
        """
        Prepare the virtual environment for execution by adding variables from globals, builtins, and constants.

        """
        from .variables import VariableFactory

        self._globals = FunctionGlobalVariable(
            self._fn_var,
            self._fn_value.__globals__,
            self._graph,
            DanglingTracker(),
        )

        self._builtins = self._graph._builtins

        # prepare consts
        for value in self._code.co_consts:
            self._co_consts.append(
                VariableFactory.from_value(
                    value, self._graph, ConstTracker(value)
                )
            )

    def inline_call(self) -> VariableBase:
        """
        Execute the inline call of the function.
        """
        self._graph.add_global_guarded_variable(self._code_var)
        self.run()
        assert self.return_value is not None
        return self.return_value

    def RETURN_VALUE(self, instr: Instruction):
        assert (
            len(self.stack) == 1
        ), f"Stack must have one element, but get {len(self.stack)} elements."
        self.return_value = self.stack.pop()
        return Stop(state="Return")

    def RETURN_CONST(self, instr: Instruction):
        self.return_value = self._co_consts[instr.arg]
        return Stop(state="Return")

    def _break_graph_when_if(self, result, instr: Instruction):
        """
        Helper method to raise a BreakGraphError when breaking the graph in a jump operation.

        Args:
            result: The result of the operation.
            instr (Instruction): The jump instruction.
        """
        raise BreakGraphError(
            "OpcodeInlineExecutor want break graph when simulate `if`."
        )

    def FOR_ITER(self, instr: Instruction):
        iterator = self.stack.top
        assert isinstance(iterator, IterVariable)

        self._graph.add_global_guarded_variable(iterator)

        # simply get next
        if isinstance(
            iterator,
            SequenceIterVariable,
        ):
            try:
                self.stack.push(iterator.next())
            except StopIteration:
                self.stack.pop()
                assert isinstance(instr.jump_to, Instruction)
                self._lasti = self.indexof(instr.jump_to)
                if sys.version_info >= (3, 12):
                    assert self._instructions[self._lasti].opname == "END_FOR"
                    skip_n_instrs = 2 if sys.version_info >= (3, 13) else 1
                    self._lasti += skip_n_instrs

        else:
            self._graph.remove_global_guarded_variable(iterator)
            raise BreakGraphError(
                f"Found {iterator.__class__.__name__} as iterator."
            )
