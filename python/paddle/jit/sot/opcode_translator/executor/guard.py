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

import types
import weakref
from functools import cached_property
from typing import TYPE_CHECKING, Any, Callable, TypeVar

import paddle

from ...profiler import EventGuard
from ...utils import (
    ENV_SOT_ENABLE_FASTER_GUARD,
    current_symbol_registry,
    log,
    log_do,
)

Guard = Callable[[types.FrameType], bool]

if TYPE_CHECKING:
    from .variables import VariableBase

    GuardBase = paddle.framework.core.GuardBase
    CheckGuardInputT = TypeVar("CheckGuardInputT", bound=VariableBase)

# NOTE(SigureMo): [How to write Stringified Guard?]
# 1. we should capture free variables manually, the string cannot capture free
#    variables automatically.
# 2. Be aware that the comparison logic before and after stringify may be different.
# 3. we should compute as much as possible at "compile time" and encode the
#    computation in the Guard string, rather than passing it to runtime to minimize
#    runtime overhead.


class StringifiedExpression:
    """
    Used to store string based expressions for generating Guard.
    """

    def __init__(
        self,
        expr_template: str,
        sub_exprs: list[StringifiedExpression],
        free_vars: dict[str, Any],
    ):
        self.expr_template = expr_template
        expr = self.expr_template.format(
            *[sub_expr.symbol for sub_expr in sub_exprs]
        )
        self.registered_expr = expr
        self.symbol = current_symbol_registry().request_symbol(expr)
        self.sub_exprs = sub_exprs
        self.free_vars = free_vars

    @cached_property
    def inlined_expr(self):
        return self.expr_template.format(
            *[sub_expr.inlined_expr for sub_expr in self.sub_exprs]
        )

    def gen_expr(self):
        def gen_expr_fn():
            return self.expr_template.format(
                *[sub_expr.gen_expr() for sub_expr in self.sub_exprs]
            )

        return current_symbol_registry().gen_expr(
            self.registered_expr, gen_expr_fn
        )

    def __hash__(self):
        if self.free_vars:
            return hash((self.inlined_expr, id(self)))
        else:
            return hash(self.inlined_expr)


class FasterStringifiedExpression(StringifiedExpression):
    def __init__(
        self,
        expr_template: str,
        faster_guard: GuardBase,
        sub_exprs: list[StringifiedExpression],
        free_vars: dict[str, Any],
    ):
        self.faster_guard = faster_guard
        if ENV_SOT_ENABLE_FASTER_GUARD.get():
            original_expr_template = expr_template
            guard_cls_name = faster_guard.__class__.__name__
            guard_name = f"{guard_cls_name}_{id(faster_guard)}"
            expr_template = (
                guard_name + "(" + ", ".join(["{}"] * len(sub_exprs)) + ")"
            )
            free_vars = union_free_vars(
                free_vars, {guard_name: faster_guard.check}
            )
            log(
                3,
                f"[FasterGuard]: transform {original_expr_template} to {expr_template}\n",
            )

        super().__init__(expr_template, sub_exprs, free_vars)


def union_free_vars(*free_vars: dict[str, Any]):
    return {k: v for d in free_vars for k, v in d.items()}


def make_guard(stringified_guards: list[StringifiedExpression]) -> Guard:
    """
    Make a guard from a list of StringifiedExpression.

    For more design ideas, refer to the `Stringified guard <https://github.com/PaddlePaddle/PaddleSOT/blob/develop/docs/design/stringify-guard.md>`_ for details.

    Args:
        stringified_guards: a list of StringifiedExpression.
    """
    with EventGuard("make_guard"):
        num_guards = len(stringified_guards)
        if not num_guards:
            guard = lambda frame: True
            guard.expr = "lambda frame: True"
            guard.original_guard = guard
            return guard

        free_vars = union_free_vars(
            *(expr.free_vars for expr in stringified_guards)
        )
        inlined_guard_expr = "lambda frame: " + " and ".join(
            [expr.inlined_expr for expr in stringified_guards]
        )
        guard_expr: str = "lambda frame: " + " and ".join(
            [expr.gen_expr() for expr in stringified_guards]
        )

        guard = eval(guard_expr, free_vars)

        log(3, f"[Guard]: {inlined_guard_expr}\n")
        guard.inlined_expr = inlined_guard_expr
        guard.expr = guard_expr

        assert callable(guard), "guard must be callable."

        return guard


def support_weak_ref(obj):
    if isinstance(obj, types.FunctionType):
        return True
    return False


def check_guard(
    fn: Callable[[CheckGuardInputT], list[StringifiedExpression]],
) -> Callable[[CheckGuardInputT], list[StringifiedExpression]]:
    def wrapper(self: CheckGuardInputT) -> list[StringifiedExpression]:
        assert (
            self.tracker.is_traceable()
        ), "Cannot make guard from a non-tracable guard variable."

        def guard_log():
            frame_value_tracer = self.tracker.trace_value_from_frame()
            print(
                f"[Guard]: guard_fn for {self}, tracker={self.tracker.__class__.__name__}, value={frame_value_tracer.registered_expr}"
            )

        log_do(4, guard_log)
        return fn(self)

    return wrapper


@check_guard
def object_equal_stringified_guard(self) -> list[StringifiedExpression]:
    frame_value_tracer = self.tracker.trace_value_from_frame()

    obj_free_var_name = f"__{self.id}"
    weak_ref_obj = self.get_py_value()
    if support_weak_ref(weak_ref_obj):
        weak_ref_obj = weakref.ref(self.get_py_value())
        return [
            StringifiedExpression(
                f"{obj_free_var_name}() is not None and {{}} == {obj_free_var_name}()",
                [frame_value_tracer],
                union_free_vars(
                    frame_value_tracer.free_vars,
                    {obj_free_var_name: weak_ref_obj},
                ),
            )
        ]
    return [
        FasterStringifiedExpression(
            f"{{}} == {obj_free_var_name}",
            paddle.framework.core.StringCompareGuard(obj_free_var_name),
            [frame_value_tracer],
            union_free_vars(
                frame_value_tracer.free_vars,
                {obj_free_var_name: self.get_py_value()},
            ),
        )
    ]


def stringify_pyobject(obj: object) -> tuple[str, dict[str, Any]]:
    if isinstance(obj, paddle.core.VarDesc.VarType):
        return f"paddle.core.VarDesc.VarType({obj.value})", {"paddle": paddle}
    elif isinstance(obj, paddle.core.DataType):
        return f"paddle.core.DataType({obj.value})", {"paddle": paddle}
    # For builtin values
    return f"{obj!r}", {}
