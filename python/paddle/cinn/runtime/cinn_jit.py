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

import ast
import functools
import inspect
import textwrap
from typing import Callable, Generic, TypeVar, cast

from .utils import inspect_function_scope

T = TypeVar('T')


class CinnLowerLevelIrJit(Generic[T]):
    def __init__(self, fn):
        self.fn = fn
        # function prototype
        signature = inspect.signature(fn)
        self.arg_names = [v.name for v in signature.parameters.values()]

        self.src = textwrap.dedent(inspect.getsource(fn))
        self.src = self.src[self.src.find("def") :]
        self.scope = inspect_function_scope(fn)

        # docs of wrapped function
        self.__doc__ = fn.__doc__
        self.__name__ = fn.__name__
        self.__globals__ = fn.__globals__
        self.__module__ = fn.__module__

        # Encapsulates the compile and run processes
        self.run = self._make_launcher()

    def _make_launcher(self):
        # Gets information about runtime input parameters
        jit_input_args = ', '.join(arg_name for arg_name in self.arg_names)
        lazy_compile = f"""
import paddle.cinn as cinn
def {self.fn.__name__}({jit_input_args}, target=cinn.common.DefaultHostTarget()):
    from paddle.cinn.compiler import compile
    jit_inputs = {', '.join([f'{arg}' for arg in self.arg_names])}
    jit_inputs_signature = {{ i: self._convert_arg_type(arg) \
                             for i, arg in enumerate(jit_inputs)}}
    module = compile(self, jit_inputs_signature=jit_inputs_signature, arg_names={
                     self.arg_names}, target=target)
    module({jit_input_args})

    return module
        """
        scope = {
            "self": self,
        }
        exec(lazy_compile, scope)
        return scope[self.fn.__name__]

    def convert_to_llir(self):
        from paddle.cinn.compiler import compile

        return compile(self, just_convert=True)

    def parse(self):
        tree = ast.parse(self.src)
        assert isinstance(tree, ast.Module)
        return tree

    def __getitem__(self, target):
        return cast(
            T, functools.partial(cast(Callable, self.run), target=target)
        )

    def _convert_arg_type(self, arg):
        # arg is a Tensor
        if hasattr(arg, "dtype"):
            return arg
        # arg is a Var
        else:
            if isinstance(arg, int):
                if -(2**21) <= arg and arg <= 2**31 - 1:
                    return "i32"
                elif 2**63 <= arg and arg <= 2**64 - 1:
                    return "u64"
                else:
                    return "i64"
            elif isinstance(arg, float):
                return "fp32"
            else:
                raise TypeError(f'Unsupported type {type(arg)} for {arg}')

    def __str__(self):
        return str(self.convert_to_llir())


def to_cinn_llir(fn: T | None = None) -> CinnLowerLevelIrJit[T]:
    def decorator(fn: T) -> CinnLowerLevelIrJit[T]:
        return CinnLowerLevelIrJit(fn)

    if fn is not None:
        return decorator(fn)
    else:
        return decorator
