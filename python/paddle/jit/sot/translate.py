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

from typing import Callable, TypeVar

from typing_extensions import ParamSpec

import paddle

from .opcode_translator import eval_frame_callback
from .profiler import SotStepProfilerGuard
from .utils import (
    GraphLogger,
    InfoCollector,
    StepInfoManager,
    StepState,
    log_do,
)

P = ParamSpec("P")
R = TypeVar("R")


def symbolic_translate(fn: Callable[P, R], **kwargs) -> Callable[P, R]:
    """
    This function is the entry point of PaddleSOT. It sets eval_frame_callback before input
    function to achieve Opcode-level translation. The translation process depends on the
    simulation execution, in which information will be collected, especially the network
    code. After the simulation execution is completed, the network code will be compiled
    into a static graph Program to improve performance.

    Args:
        fn: The input function.

    Returns:
        Callable, The wrapped function.

    Examples:
        >>> # doctest: +SKIP("Could not get source code of function foo."")
        >>> import paddle
        >>> import numpy as np
        >>> from sot.translate import symbolic_translate
        >>> def foo(cond: paddle.Tensor, x: paddle.Tensor):
        ...     x += 1
        ...     if cond:
        ...         x += 1
        ...     else:
        ...         x -= 1
        ...     return x
        >>> symbolic_translate_foo = symbolic_translate(foo)
        >>> # For the true branch, the output is 2.
        >>> cond = paddle.to_tensor(True)
        >>> x = paddle.to_tensor(0)
        >>> dygraph_out = foo(cond, x)
        >>> symbolic_translate_out = symbolic_translate_foo(cond, x)
        >>> dygraph_out
        Tensor(shape=[], dtype=int64, place=Place(cpu), stop_gradient=True,
        2)
        >>> symbolic_translate_out
        Tensor(shape=[], dtype=int64, place=Place(cpu), stop_gradient=True,
        2)
        >>> np.testing.assert_allclose(
        ...     dygraph_out.numpy(), symbolic_translate_out.numpy()
        ... )
        >>> # For the false branch, the output is 0.
        >>> cond = paddle.to_tensor(False)
        >>> dygraph_out = foo(cond, x)
        >>> symbolic_translate_out = symbolic_translate_foo(cond, x)
        >>> dygraph_out
        Tensor(shape=[], dtype=int64, place=Place(cpu), stop_gradient=True,
        0)
        >>> symbolic_translate_out
        Tensor(shape=[], dtype=int64, place=Place(cpu), stop_gradient=True,
        0)
        >>> np.testing.assert_allclose(
        ...     dygraph_out.numpy(), symbolic_translate_out.numpy()
        ... )

    """

    kwargs.setdefault('training', True)

    def callback(frame):
        return eval_frame_callback(frame, **kwargs)

    def impl_sot(*args: P.args, **kwargs: P.kwargs) -> R:
        assert hasattr(
            fn, "__code__"
        ), "Target function doesn't have code for simulating."
        StepInfoManager().sot_step()
        GraphLogger().clear()
        InfoCollector().clear_step_info()
        paddle.framework.core.set_eval_frame(callback)
        try:
            outs = fn(*args, **kwargs)
        except Exception as e:
            raise e
        finally:
            paddle.framework.core.set_eval_frame(None)

        log_do(1, lambda: GraphLogger().print_info())
        InfoCollector().print_step_report()
        return outs

    def impl_dynamic(*args: P.args, **kwargs: P.kwargs) -> R:
        outs = fn(*args, **kwargs)
        return outs

    def impl(*args: P.args, **kwargs: P.kwargs) -> R:
        with StepInfoManager().step_guard(fn.__code__), SotStepProfilerGuard():
            state = StepInfoManager().current_state

            if state == StepState.RUN_SOT:
                return impl_sot(*args, **kwargs)
            elif state == StepState.RUN_DYN:
                return impl_dynamic(*args, **kwargs)
            elif state == StepState.COLLECT_INFO:
                return StepInfoManager().collect_info(
                    impl_dynamic, impl_sot, *args, **kwargs
                )
            else:
                raise RuntimeError("Unknown state.")

    return impl
