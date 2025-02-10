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

# This file is specifically used to handle the problem
# of generating a Graph from a linear function call.

from __future__ import annotations

import builtins
import inspect
from collections import namedtuple
from copy import deepcopy
from functools import cached_property, reduce
from typing import Any, Callable, Tuple, Union

from typing_extensions import TypeAlias, TypeGuard

import paddle
from paddle.jit.utils import OrderedSet
from paddle.utils import flatten, map_structure

from .....utils.layers_utils import NotSupportedTensorArgumentError
from ...infer_meta import (
    InferMetaCache,
    LayerInferMetaCache,
    MetaInfo,
    ast_infer_meta,
)
from ...profiler import EventGuard, event_register
from ...symbolic.statement_ir import Reference, StatementIR, Symbol
from ...symbolic.symbolic_context import SymbolicTraceContext
from ...utils import (
    ENV_SOT_ALLOW_DYNAMIC_SHAPE,
    NameGenerator,
    SotUndefinedVar,
    inner_error_default_handler,
    is_inplace_api,
    is_paddle_api,
    log,
    log_do,
    map_if,
    switch_symbol_registry,
)
from ...utils.exceptions import BreakGraphError, SotExtraInfo
from ..instruction_utils import get_instructions
from .guard import Guard, StringifiedExpression, make_guard
from .mutable_data import MutationDel, MutationNew, MutationSet
from .pycode_generator import PyCodeGen
from .side_effects import (
    DictSideEffectRestorer,
    GlobalDelSideEffectRestorer,
    GlobalSetSideEffectRestorer,
    ListSideEffectRestorer,
    ObjDelSideEffectRestorer,
    ObjSetSideEffectRestorer,
    SideEffectRestorer,
    SideEffects,
)
from .tracker import BuiltinTracker, DummyTracker, SymbolicOperationTracker
from .variables import (
    DictVariable,
    GlobalVariable,
    ListVariable,
    NullVariable,
    PaddleLayerVariable,
    ParameterVariable,
    SymbolicVariable,
    TensorVariable,
    VariableBase,
    VariableFactory,
    find_traceable_vars,
    map_variables,
)

CompileGraphResult: TypeAlias = Tuple[
    Callable[..., Any],
    Tuple[
        StatementIR,
        OrderedSet[Union[TensorVariable, SymbolicVariable]],
        OrderedSet[Union[TensorVariable, SymbolicVariable]],
    ],
]


def convert_to_meta(inputs: Any):
    """
    Convert the input variables to meta if it is TensorVariable.
    """

    def func(x):
        if isinstance(x, (TensorVariable, SymbolicVariable)):
            return x.meta
        if isinstance(x, VariableBase):
            return x.get_py_value()
        return x

    return map_variables(func, inputs)


def convert_to_symbol(inputs: Any):
    """
    Convert the input variables to symbol if it can be symbolic.
    """

    def func(x):
        if isinstance(x, (TensorVariable, SymbolicVariable)):
            return x.get_symbol()
        if isinstance(x, VariableBase):
            return x.get_py_value()
        return x

    return map_variables(func, inputs)


def convert_to_py_value(inputs):
    def func(x):
        if isinstance(x, VariableBase):
            return x.get_py_value()
        return x

    return map_variables(func, inputs)


def record_symbols(SIR, *args, **kwargs):
    symbol_meta_map = {}
    params = set()
    non_params = set()

    def fn(value):
        if isinstance(value, (TensorVariable, SymbolicVariable)):
            symbol_meta_map[value.get_symbol()] = value.meta
            if isinstance(value, ParameterVariable):
                params.add(value.get_symbol())
            else:
                non_params.add(value.get_symbol())
        return value

    map_variables(fn, [args, kwargs])  # type: ignore
    SIR.set_symbol_meta_map(symbol_meta_map)
    SIR.set_parameter_info(params, non_params)


def get_params_and_non_param_symbol(*args, **kwargs):
    params = set()
    non_params = set()

    for value in flatten([args, kwargs]):
        if isinstance(value, ParameterVariable):
            params.add(value.get_symbol())
        elif isinstance(value, TensorVariable):
            non_params.add(value.get_symbol())

    return params, non_params


def replace_symbolic_var_with_constant_var(inputs):
    def func(x):
        if isinstance(x, SymbolicVariable):
            return x.to_constant()
        return x

    return map_variables(func, inputs, restore_variable=True)


class VariableLoader:
    def __init__(self, store_var_info, pycode_gen):
        self._store_var_info = store_var_info
        self._pycode_gen: PyCodeGen = pycode_gen

    def load(self, var):
        if var is SotUndefinedVar():
            self._pycode_gen.gen_load_const(SotUndefinedVar())
        elif isinstance(var, NullVariable):
            var.reconstruct(self._pycode_gen)
        else:
            # NOTE: One variable may have multiple names, we can
            # use any name to load it.
            self._pycode_gen.gen_load(self._store_var_info[var.id][0])


class FunctionGraph:
    """
    A Graph representation corresponding to each FunctionFrame
    The input binding diagram containing the current call represents three parts of output settings,
    This Graph can be compiled as a f_locals dependency function which produce the same outputs.
    """

    OUT_VAR_PREFIX = "___SIR_out_"
    Memo = namedtuple(
        "function_graph_memo",
        [
            'inner_out',
            'input_variables',
            "stmt_ir",
            "global_guards",
            "side_effects_state",
            "print_variables",
            "inplace_tensors",
            "need_cache",
        ],
    )

    def __init__(self, frame, **kwargs):
        self.sir_ctx = SymbolicTraceContext()
        self.inner_out = set()
        self.input_variables = []  # Store variables required within a function
        self.pycode_gen = PyCodeGen(frame, disable_eval_frame=True)
        self.side_effects = SideEffects()
        self.need_cache = True
        self._global_guarded_variables: OrderedSet[VariableBase] = OrderedSet()
        self._print_variables = []
        self._inplace_tensors = OrderedSet()
        self._kwargs = kwargs

    @cached_property
    def _builtins(self):
        builtins_ = {}
        # prepare builtins
        for name, value in builtins.__dict__.items():
            builtins_[name] = VariableFactory.from_value(
                value, self, BuiltinTracker(name), debug_name=name
            )
        return builtins_

    def add_print_variables(self, variable):
        """
        Used to support psdb_print
        """
        self._print_variables.append(variable)

    def add_inplace_tensors(self, variable):
        """
        Used to support psdb_print
        """
        self._inplace_tensors.add(variable)

    def need_add_input(self, var):
        """
        Determine if it is the input of graph.

        Args:
            var: The input variable.

        """
        if var.id in self.inner_out:
            return False
        for v in self.input_variables:
            if v.id == var.id:
                return False
        return True

    def save_memo(self) -> FunctionGraph.Memo:
        """
        Save the state of the current FunctionGraph, for future state recovery, it is used for state recovery during inline call error reporting

        NOTE:
            Why don't use __deepcopy__, because memo is not a deepcopy, i.e inner_out is only a shallow copy, SIR is a deepcopy.
        """
        saved_stmt_ir = deepcopy(self.sir_ctx.TOS)
        return FunctionGraph.Memo(
            inner_out=set(self.inner_out),
            input_variables=list(self.input_variables),
            stmt_ir=saved_stmt_ir,
            global_guards=OrderedSet(self._global_guarded_variables),
            side_effects_state=self.side_effects.get_state(),
            print_variables=list(self._print_variables),
            inplace_tensors=OrderedSet(self._inplace_tensors),
            need_cache=self.need_cache,
        )

    def restore_memo(self, memo: FunctionGraph.Memo):
        """
        Restore the state of graph to memo.

        Args:
            memo: Previously recorded memo

        """
        self.inner_out = memo.inner_out
        self.input_variables = memo.input_variables
        self.sir_ctx.replace_TOS(memo.stmt_ir)
        self._global_guarded_variables = memo.global_guards
        self.side_effects.restore_state(memo.side_effects_state)
        self._print_variables = memo.print_variables
        self._inplace_tensors = memo.inplace_tensors
        self.need_cache = memo.need_cache

    def collect_input_variables(self, inputs: list[VariableBase]):
        """
        Variables required within the method

        Args:
            inputs: Required VariableBase
        """

        def collect(inp):
            if isinstance(inp, VariableBase) and self.need_add_input(inp):
                self.input_variables.append(inp)

        map_variables(
            collect,
            inputs,
        )

    @property
    @event_register("guard_fn")
    def guard_fn(self) -> Guard:
        with switch_symbol_registry():
            guards: list[StringifiedExpression] = []
            with EventGuard("guard_fn: find vars and make stringified guard"):
                for variable in find_traceable_vars(
                    self.input_variables + list(self._global_guarded_variables)
                ):
                    guards.extend(variable.make_stringified_guard())

            guards = OrderedSet(guards)  # type: ignore

            for guard in guards:
                assert isinstance(
                    guard, StringifiedExpression
                ), "guard must be StringifiedExpression."

            return make_guard(guards)

    def _restore_origin_opcode(self, stack_vars, store_var_info, instr_idx):
        origin_instrs = get_instructions(self.pycode_gen._origin_code)
        is_precall = origin_instrs[instr_idx].opname == "PRECALL"
        current_idx = instr_idx
        # skip CALL if current instr is PRECALL
        next_idx = instr_idx + 1 + int(is_precall)

        restore_instrs = origin_instrs[:current_idx]
        restore_instr_names = [
            instr.opname for instr in restore_instrs[:current_idx]
        ]
        # NOTE(SigureMo): Trailing KW_NAMES is no need to restore in Python 3.11+
        if restore_instr_names[-1:] == ["KW_NAMES"]:
            restore_instrs = restore_instrs[:-1]
            restore_instr_names = restore_instr_names[:-1]

        self.pycode_gen.extend_instrs(restore_instrs)
        nop = self.pycode_gen.add_instr("NOP")

        for instr in origin_instrs:
            if instr.jump_to == origin_instrs[current_idx]:
                instr.jump_to = nop

        self.pycode_gen.hooks.append(
            lambda: self.pycode_gen.extend_instrs(
                iter(origin_instrs[next_idx:])
            )
        )

        self.pycode_gen.gen_enable_eval_frame()

        name_gen = NameGenerator("___graph_fn_saved_orig_")

        # here is not update changed values, it just give names to stack vars
        # and want keep same interface as _build_compile_fn_with_name_store
        for var in stack_vars[::-1]:
            if not store_var_info.get(var.id, []):
                name = name_gen.next()
                store_var_info.setdefault(var.id, [])
                store_var_info[var.id].append(name)
                self.pycode_gen.gen_store_fast(name)
            else:
                all_names = store_var_info[var.id]
                for _ in range(len(all_names) - 1):
                    self.pycode_gen.gen_dup_top()
                for name in all_names:
                    self.pycode_gen.gen_store(
                        name, self.pycode_gen._origin_code
                    )

        return VariableLoader(store_var_info, self.pycode_gen)

    def _build_compile_fn_with_name_store(
        self,
        compile_graph_result: CompileGraphResult,
        to_store_vars,
        store_var_info,
    ):
        # var_id -> local_name mapping
        to_store_vars = list(
            filter(lambda x: not isinstance(x, NullVariable), to_store_vars)
        )
        self.compile_function(compile_graph_result, to_store_vars)
        name_gen = NameGenerator("___graph_fn_saved_")

        for var in to_store_vars[::-1]:
            if not store_var_info.get(var.id, []):
                name = name_gen.next()
                store_var_info.setdefault(var.id, [])
                store_var_info[var.id].append(name)
                self.pycode_gen.gen_store_fast(name)
            else:
                all_names = store_var_info[var.id]
                for _ in range(len(all_names) - 1):
                    self.pycode_gen.gen_dup_top()
                for name in all_names:
                    self.pycode_gen.gen_store(
                        name, self.pycode_gen._origin_code
                    )

        return VariableLoader(store_var_info, self.pycode_gen)

    def compile_graph(self, *ret_vars: VariableBase) -> CompileGraphResult:
        ret_items = [
            ret_item
            for ret_var in ret_vars
            for ret_item in ret_var.flatten_inner_vars()
        ]

        symbolic_outputs = self._find_tensor_outputs(ret_items)
        statement_ir = self.sir_ctx.return_TOS(
            [Symbol(tensor_var.var_name) for tensor_var in symbolic_outputs]
        )
        if not statement_ir.statements:
            return self.sir_ctx.compile_do_nothing(), (
                statement_ir,
                OrderedSet(),
                OrderedSet(),
            )
        input_names = statement_ir.inputs
        symbolic_inputs = self._find_tensor_inputs(input_names)
        compiled_fn = self.sir_ctx.compile_fn(
            statement_ir.name,
            tuple(var.meta.to_input_spec() for var in symbolic_inputs),
            **self._kwargs,
        )
        return compiled_fn, (statement_ir, symbolic_inputs, symbolic_outputs)

    @event_register("compile_function", event_level=2)
    def compile_function(
        self,
        compile_graph_result: CompileGraphResult,
        ret_vars: list[VariableBase],
    ):
        """
        Generate bytecode based on the information collected by the simulation execution.

        This consists of the following steps:
        - Compile the FunctionGraph into a dy2st StaticFunction and load it in the generated bytecode
        - Load the group network input
        - Calling the generated dy2st StaticFunction
        - Restore the side effects
        - Restore the output
        - Return the top of the stack
        """
        from ..breakpoint import BreakpointManager

        BreakpointManager().on_event("compile_function")
        graph_fn, (statement_ir, symbolic_inputs, symbolic_outputs) = (
            compile_graph_result
        )
        compiled_fn_name = f"___graph_fn_{statement_ir.name}"
        # prepare function and inputs
        self.pycode_gen.gen_load_object(graph_fn, compiled_fn_name)
        self.gen_load_inputs(symbolic_inputs)
        # Pack all args into a tuple, because we don't support *args now.
        self.pycode_gen.gen_build_tuple(count=len(symbolic_inputs))
        # call the graph_fn
        self.pycode_gen.gen_call_function(argc=1)

        # Store outputs to f_locals
        self.pycode_gen.gen_unpack_sequence(count=len(symbolic_outputs))
        for tensor_var in symbolic_outputs:
            self.pycode_gen.gen_store_fast(tensor_var.out_var_name)
        # restore the outputs.
        for ret_var in ret_vars:
            ret_var.reconstruct(self.pycode_gen)

        # deal side effect
        self.restore_inplace_tensor(self._inplace_tensors)
        self.restore_print_stmts(self._print_variables)
        self.restore_side_effects(self.side_effects.proxy_variables)
        self.pycode_gen.gen_enable_eval_frame()

    def call_paddle_api(
        self,
        func: Callable[..., Any],
        *args: VariableBase,
        **kwargs: VariableBase,
    ):
        """
        Record Paddle Networking API to SIR

        Args:
            func: paddle api
        """
        assert is_paddle_api(func)
        # not fallback api, start symbolic trace.
        # TODO(xiokgun): may have python builtin object inside metas.
        # TODO(xiokgun): 4 kinds of python arguments. support it !!
        log(3, f"call paddle.api : {func.__name__}", "\n")

        def message_handler(*args, **kwargs):
            return f"Call paddle_api error: {func.__name__}, may be not a operator api?"

        return inner_error_default_handler(self.symbolic_call, message_handler)(
            InferMetaCache(),
            self.sir_ctx.call_API,
            func,
            False,
            *args,
            **kwargs,
        )

    def call_tensor_method(
        self, method_name: str, *args: VariableBase, **kwargs
    ):
        """
        call tensor method, start symbolic trace.

        Args:
            method_name: tensor method name
        """

        def message_handler(*args, **kwargs):
            return f"Call tensor_method error: Tensor.{method_name}, may be not a valid operator api?"

        return inner_error_default_handler(self.symbolic_call, message_handler)(
            InferMetaCache(),
            self.sir_ctx.call_METHOD,
            method_name,
            False,
            *args,
            **kwargs,
        )

    def call_symbolic_method(
        self, method_name: str, *args: VariableBase, **kwargs
    ):
        """
        call symbolic method, start symbolic trace.

        Args:
            method_name: symbolic method name
        """

        def message_handler(*args, **kwargs):
            return f"Call symbolic_method error: Symbolic.{method_name}, may be not a valid operator api?"

        return inner_error_default_handler(self.symbolic_call, message_handler)(
            InferMetaCache(),
            self.sir_ctx.call_METHOD,
            method_name,
            True,
            *args,
            **kwargs,
        )

    def call_layer(
        self,
        layer: PaddleLayerVariable,
        weak_ref: bool,
        *args: VariableBase,
        **kwargs: VariableBase,
    ):
        """
        call paddle layer, start symbolic trace.

        Args:
            layer: paddle layer
        """

        def infer_meta_fn(layer, *metas, **kwmetas):
            metas = LayerInferMetaCache()(layer.value, *metas, **kwmetas)
            return metas

        def compute_fn(layer, inputs, outputs, stacks):
            self.sir_ctx.call_LAYER(
                Reference(layer.value, weak_ref),
                inputs=inputs,
                outputs=outputs,
                stacks=stacks,
            )

        def message_handler(*args, **kwargs):
            return f"Call paddle layer error: {layer}, may be not a valid paddle layer?"

        return inner_error_default_handler(self.symbolic_call, message_handler)(
            infer_meta_fn, compute_fn, layer, False, *args, **kwargs
        )

    def call_ast(
        self,
        static_function: tuple,
        *args: VariableBase,
        **kwargs: VariableBase,
    ):
        """
        call paddle layer, start symbolic trace.

        Args:
            layer: paddle layer
        """

        def compute_fn(static_function, inputs, outputs, stacks):
            self.sir_ctx.call_AST(
                static_function,
                inputs=inputs,
                outputs=outputs,
                stacks=stacks,
            )

        def message_handler(*args, **kwargs):
            return "Call ast failed"

        try:
            return inner_error_default_handler(
                self.symbolic_call, message_handler
            )(
                ast_infer_meta,
                compute_fn,
                static_function,
                False,
                *args,
                **kwargs,
            )
        except Exception as e:
            log(3, f"[call AST] {e}")
            return None

    def symbolic_call(
        self, infer_meta_fn, compute_fn, func, is_symbolic_var, *args, **kwargs
    ):
        """
        Using infer_meta_fn and compute_fn convert func to symbolic function.

        Args:
            infer_meta_fn: function for infer meta, (func, metas, kwmetas) -> output_metas
            compute_fn   : function for add stmt to sir, (func, input_symbols, outputs_symbols, stacks) -> None
            func         : the logical function which will be represent as a stmt
        """

        def try_infer_meta_fn(args, kwargs) -> Any:
            try:
                metas = convert_to_meta(args)
                kwmetas = convert_to_meta(kwargs)
                return args, kwargs, infer_meta_fn(func, *metas, **kwmetas)
            except (NotSupportedTensorArgumentError, TypeError) as e:
                bound_arguments = inspect.signature(func).bind(*args, **kwargs)
                bound_arguments.apply_defaults()
                if (
                    isinstance(e, NotSupportedTensorArgumentError)
                    and e.name in bound_arguments.arguments
                ):
                    original_var = bound_arguments.arguments[e.name]
                    flatten_vars = original_var.flatten_inner_vars()
                    if not any(
                        isinstance(arg, SymbolicVariable)
                        for arg in flatten_vars
                    ):
                        # TODO(zrr1999): maybe we can continue to fallback to all args are constant.
                        raise BreakGraphError(
                            f"InferMeta encount {type(e)}, but all args are not symbolic."
                        )

                    args, kwargs = map_if(
                        (args, kwargs),
                        pred=lambda x: x is original_var,
                        true_fn=lambda x: replace_symbolic_var_with_constant_var(
                            x
                        ),
                        false_fn=lambda x: x,
                    )
                else:
                    flatten_vars = reduce(
                        lambda x, y: (
                            x + y.flatten_inner_vars()
                            if isinstance(y, VariableBase)
                            else x
                        ),
                        bound_arguments.arguments.values(),
                        [],
                    )

                    if not any(
                        isinstance(arg, SymbolicVariable)
                        for arg in flatten_vars
                    ):
                        raise BreakGraphError(
                            f"InferMeta encount {type(e)}, but all args are not symbolic."
                        )

                    args, kwargs = map_structure(
                        replace_symbolic_var_with_constant_var, (args, kwargs)
                    )

                metas = convert_to_meta(args)
                kwmetas = convert_to_meta(kwargs)
                return args, kwargs, infer_meta_fn(func, *metas, **kwmetas)

            except Exception as e:
                if SotExtraInfo.from_exception(e).need_breakgraph:
                    raise BreakGraphError(
                        f"API {func} encountered a need break graph error {e}"
                    )
                raise e

        if ENV_SOT_ALLOW_DYNAMIC_SHAPE.get():
            args, kwargs, out_metas = try_infer_meta_fn(args, kwargs)
        else:
            metas = convert_to_meta(args)
            kwmetas = convert_to_meta(kwargs)
            out_metas = infer_meta_fn(func, *metas, **kwmetas)

        self.collect_input_variables(list(args))
        self.collect_input_variables(list(kwargs.values()))

        inputs_symbols = (
            convert_to_symbol(args),
            convert_to_symbol(kwargs),
        )

        record_symbols(self.sir_ctx.TOS, *args, **kwargs)

        log(3, f"         inputs : {inputs_symbols}", "\n")

        if is_symbolic_var:
            var_cls = SymbolicVariable
            tracker = SymbolicOperationTracker(
                list(args) + list(kwargs.values()), func
            )
        else:
            var_cls = TensorVariable
            tracker = DummyTracker(list(args) + list(kwargs.values()))
        outputs = map_if(
            out_metas,
            pred=lambda x: isinstance(x, MetaInfo),
            true_fn=lambda x: var_cls(
                x,
                self,
                tracker=tracker,
            ),
            false_fn=lambda x: x,
        )
        stmt_stacks = []
        log_do(
            3,
            lambda: stmt_stacks.extend(
                FunctionGraph.get_opcode_executor_stack()
            ),
        )

        if is_inplace_api(func):
            # if we want to use a non-inplace api (static api) to replace an inplace behavior (in simulation)
            # just set it back in SIR, and return outputs to replace tensor meta (it might changes?)
            # in this case, the output will not exactly be used
            compute_fn(
                func,
                inputs_symbols,
                convert_to_symbol(args[0]),
                stmt_stacks,
            )
        else:
            compute_fn(
                func,
                inputs_symbols,
                convert_to_symbol(outputs),
                stmt_stacks,
            )  # symbolic only contain symbols.
            self._put_inner(outputs)
        if is_symbolic_var:
            # compute_fn should be call_method
            tracker = SymbolicOperationTracker(
                list(args) + list(kwargs.values()), func
            )
        else:
            tracker = DummyTracker(list(args) + list(kwargs.values()))

        return VariableFactory.from_value(outputs, self, tracker)

    @staticmethod
    def get_opcode_executor_stack():
        # NOTE: only for debug.
        # dependent on OpcodeExecutor.
        from .opcode_executor import OpcodeExecutorBase

        if len(OpcodeExecutorBase.call_stack) == 0:
            # In test case, we can meet this scenario.
            return []
        current_executor = OpcodeExecutorBase.call_stack[-1]
        current_line = current_executor._current_line
        filename = current_executor._code.co_filename
        source_lines, start_line = inspect.getsourcelines(
            current_executor._code
        )
        # TODO(SigureMo): In 3.11, lineno maybe changed after multiple breakgraph,
        # We need to find a way to fix this.
        line_idx = max(min(current_line - start_line, len(source_lines) - 1), 0)
        code_line = source_lines[line_idx]
        stack = []
        stack.append(
            f'  File "{filename}", line {current_line}, in {current_executor._code.co_name}'
        )
        stack.append(f'    {code_line}')
        return stack

    def _put_inner(self, vars: VariableBase):
        """
        put inner variable to inner_out
        """
        map_if(
            vars,
            pred=lambda x: isinstance(x, VariableBase),
            true_fn=lambda x: self.inner_out.add(x.id),
            false_fn=lambda x: None,
        )

    def add_global_guarded_variable(self, variable: VariableBase):
        """
        Add variable to global guarded variable
        """
        self._global_guarded_variables.add(variable)

    def remove_global_guarded_variable(self, variable: VariableBase):
        """
        Remove variable to global guarded variable
        """
        if variable in self._global_guarded_variables:
            self._global_guarded_variables.remove(variable)

    def _find_tensor_inputs(
        self, input_names: list[str]
    ) -> OrderedSet[TensorVariable | SymbolicVariable]:
        inputs: OrderedSet[TensorVariable | SymbolicVariable] = OrderedSet()
        for name in input_names:
            found = False
            for variable in self.input_variables:
                if (
                    isinstance(variable, (TensorVariable, SymbolicVariable))
                    and variable.get_symbol().name == name
                ):
                    inputs.add(variable)
                    found = True
                    break
            assert found, f"can't find input {name} in SIR."
        assert len(inputs) == len(input_names), "Number of inputs not match."
        return inputs

    def gen_load_inputs(
        self, inputs: OrderedSet[TensorVariable | SymbolicVariable]
    ):
        for input_var in inputs:
            # For SymbolicVariable, we use paddle.full([], value, "int64")
            # to convert it to a Tensor
            if isinstance(input_var, SymbolicVariable):
                self.pycode_gen.gen_load_object(
                    paddle.full,
                    "___paddle_full",
                )
                self.pycode_gen.gen_build_list(0)
            input_var.tracker.gen_instructions(self.pycode_gen)
            if isinstance(input_var, SymbolicVariable):
                self.pycode_gen.gen_load_const("int64")
                self.pycode_gen.gen_call_function(3)

    def _find_tensor_outputs(
        self, outputs: list[VariableBase]
    ) -> OrderedSet[TensorVariable | SymbolicVariable]:
        """
        Return all TensorVariable. find TensorVariables participating in networking from the output Variables

        Args:
            outputs: output variables
        """

        def is_graph_output(
            var,
        ) -> TypeGuard[TensorVariable | SymbolicVariable]:
            return isinstance(
                var.tracker, (DummyTracker, SymbolicOperationTracker)
            ) and isinstance(var, (TensorVariable, SymbolicVariable))

        def collect_related_dummy_tensor(var):
            if isinstance(
                var.tracker, (DummyTracker, SymbolicOperationTracker)
            ):
                if is_graph_output(var):
                    return [var]
                else:
                    retval = []
                    for inp in var.tracker.inputs:
                        retval.extend(collect_related_dummy_tensor(inp))
                    return retval
            return []

        output_tensors: OrderedSet[TensorVariable | SymbolicVariable] = (
            OrderedSet()
        )
        # Find Tensor Variables from outputs.
        for output in outputs:
            if isinstance(
                output.tracker, (DummyTracker, SymbolicOperationTracker)
            ):
                if is_graph_output(output):
                    output_tensors.add(output)
                else:
                    for inp in output.tracker.inputs:
                        for _var in collect_related_dummy_tensor(inp):
                            output_tensors.add(_var)
                    # Guard output that can not be traced.
                    self.add_global_guarded_variable(output)
        # Find Tensor Variables from side effects Variables.
        for side_effect_var in self.side_effects.proxy_variables:
            if isinstance(side_effect_var, (ListVariable, DictVariable)):
                for var in side_effect_var.flatten_inner_vars():
                    if (
                        is_graph_output(var)
                        and side_effect_var.tracker.is_traceable()
                    ):
                        output_tensors.add(var)
            else:
                if isinstance(side_effect_var, GlobalVariable):
                    proxy_records = side_effect_var.proxy.records
                elif side_effect_var.tracker.is_traceable():
                    # for attr side effect
                    proxy_records = side_effect_var.attr_proxy.records
                else:
                    continue
                for record in proxy_records:
                    if isinstance(record, (MutationSet, MutationNew)):
                        for var in record.value.flatten_inner_vars():
                            if is_graph_output(var):
                                output_tensors.add(var)
        # Find Tensor in print_stmts
        for print_stmt in self._print_variables:
            for var in print_stmt.flatten_inner_vars():
                if is_graph_output(var):
                    output_tensors.add(var)

        # add inplace tensors into output tensors.
        for inplace_tensor in self._inplace_tensors:
            output_tensors.add(inplace_tensor)

        return output_tensors

    def restore_print_stmts(self, variables: list[VariableBase]):
        for var in variables:
            var.reconstruct(
                self.pycode_gen,
                use_tracker=False,
                add_to_global_guarded_vars=False,
            )

    def restore_inplace_tensor(self, variables: OrderedSet[VariableBase]):
        for var in variables:
            if not var.tracker.is_traceable():
                continue
            var.reconstruct(
                self.pycode_gen,
                use_tracker=True,
                add_to_global_guarded_vars=False,
            )
            self.pycode_gen.gen_load_method(
                "_inplace_assign"
            )  # NOTE: paddle related logic.
            var.reconstruct(
                self.pycode_gen,
                use_tracker=False,
                add_to_global_guarded_vars=True,
            )
            self.pycode_gen.gen_call_method(1)
            self.pycode_gen.gen_pop_top()

    def restore_side_effects(self, variables: list[VariableBase]):
        """
        Generate side effect recovery code for variables with side effects

        Args:
            variables: Variables that may have side effects.
        """
        restorers: list[SideEffectRestorer] = []

        for var in variables:
            # skip inner variables
            if not var.tracker.is_traceable() and not isinstance(
                var, GlobalVariable
            ):
                continue
            if isinstance(var, DictVariable):
                restorers.append(DictSideEffectRestorer(var))
            elif isinstance(var, ListVariable):
                restorers.append(ListSideEffectRestorer(var))
            else:
                if isinstance(var, GlobalVariable):
                    for record in var.proxy.records[::-1]:
                        if isinstance(record, (MutationSet, MutationNew)):
                            restorers.append(
                                GlobalSetSideEffectRestorer(
                                    record.key,
                                    record.value,
                                )
                            )
                        elif isinstance(record, MutationDel):
                            restorers.append(
                                GlobalDelSideEffectRestorer(record.key)
                            )
                else:
                    for record in var.attr_proxy.records[::-1]:
                        if isinstance(record, (MutationSet, MutationNew)):
                            restorers.append(
                                ObjSetSideEffectRestorer(
                                    var,
                                    record.key,
                                    record.value,
                                )
                            )
                        elif isinstance(record, MutationDel):
                            restorers.append(
                                ObjDelSideEffectRestorer(
                                    var,
                                    record.key,
                                )
                            )

        for restorer in restorers:
            restorer.pre_gen(self.pycode_gen)
        for restorer in restorers[::-1]:
            restorer.post_gen(self.pycode_gen)
