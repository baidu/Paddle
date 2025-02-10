# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import sys
import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    TypeVar,
    overload,
)

import decorator
from typing_extensions import ParamSpec

import paddle
from paddle.base import core, framework
from paddle.base.framework import global_var
from paddle.base.multiprocess_utils import CleanupFuncRegistrar

from ..framework import _get_paddle_place
from ..wrapped_decorator import signature_safe_contextmanager, wrap_decorator
from .tracer import Tracer

if TYPE_CHECKING:
    from collections import OrderedDict
    from collections.abc import Generator, Sequence
    from contextlib import AbstractContextManager
    from types import TracebackType

    from typing_extensions import Self

    from paddle import Tensor
    from paddle._typing import PlaceLike

__all__ = []

_InputT = ParamSpec("_InputT")
_RetT = TypeVar("_RetT")

NON_PERSISTABLE_VAR_NAME_SUFFIX = "__non_persistable"


def in_to_static_mode() -> bool:
    """
    Return a bool value that indicates whether running code under `@to_static`

    """
    return global_var._in_to_static_mode_


def in_sot_simulation_mode() -> bool:
    """
    Return a bool value that indicates whether running code under SOT simulation context.

    """
    return global_var._in_sot_simulation_mode_


# TODO(Aurelius84): Need to remove this alias after clean usage in PaddleX
in_declarative_mode = in_to_static_mode


def to_static_unsupported_argument_warning(
    func_name, input_names, inputs, support_values
):
    """
    Warning if inputs do not elementwisely equals to support_values.
    It's a utility function for dy2static when dygraph interface have
    more inputs than static interface such as paddle.grad.

    """
    for name, inp, sup in zip(input_names, inputs, support_values):
        if inp != sup:
            warnings.warn(
                f"{func_name} has unsupported parameter in jit: "
                + f"{name}, jit will discard it"
            )


def _switch_to_static_graph_(
    func: Callable[_InputT, _RetT]
) -> Callable[_InputT, _RetT]:
    def __impl__(*args: _InputT.args, **kwargs: _InputT.kwargs) -> _RetT:
        with framework._dygraph_guard(None):
            return func(*args, **kwargs)

    return __impl__


switch_to_static_graph = wrap_decorator(_switch_to_static_graph_)


@signature_safe_contextmanager
def to_static_mode_guard(
    is_to_static: bool = True,
) -> Generator[None, None, None]:
    global global_var
    original_val = global_var._in_to_static_mode_
    global_var._in_to_static_mode_ = is_to_static
    try:
        yield
    finally:
        global_var._in_to_static_mode_ = original_val


@signature_safe_contextmanager
def sot_simulation_mode_guard(
    is_sot_simulation: bool = True,
) -> Generator[None, None, None]:
    global global_var
    original_val = global_var._in_sot_simulation_mode_
    global_var._in_sot_simulation_mode_ = is_sot_simulation
    try:
        yield
    finally:
        global_var._in_sot_simulation_mode_ = original_val


@signature_safe_contextmanager
def param_guard(
    parameters: OrderedDict[str, Tensor]
) -> Generator[None, None, None]:
    # Note: parameters is a reference of self._parameters or self._buffers
    if in_to_static_mode() and not paddle.in_dynamic_mode() and parameters:
        try:
            origin_parameters = parameters.copy()
            for name, var_base in parameters.items():
                if isinstance(var_base, list):
                    new_var = [_convert_into_variable(var) for var in var_base]
                else:
                    new_var = _convert_into_variable(var_base)
                parameters[name] = new_var
            yield
        finally:
            parameters.update(origin_parameters)
    else:
        yield


def _convert_into_variable(tensor):
    """
    Convert Tensor into Variable.
    """
    if paddle.framework.use_pir_api():
        return paddle.pir.core._convert_into_value(tensor)
    if isinstance(tensor, paddle.Tensor):
        # Check whether has been created before.
        new_var = tensor.block._find_var_recursive(tensor.name)
        if new_var is not None:
            assert isinstance(new_var, framework.Variable)
        # Convert EagerParamBase into Parameter with same attributes in dy2stat.
        elif isinstance(tensor, framework.EagerParamBase):
            new_var = tensor._to_static_var(to_parameter=True)
        else:
            # Note(Aurelius84): Convert Tensor in self._buffers into Variable with
            # same attributes and set persistable=True to allow saving this var.
            # Because users can create a Tensor in `__init__`  like a
            # `mask` Tensor or `hidden_0` in RNN layers, which is equivalent to a Parameter
            # and necessary for inferring. It will be pruned if it's not necessary for inferring.

            # But if its shape is empty while created from `create_variable()`, we consider this buffer
            # non-persistable. See case of `dropout_state` in lstm api.
            is_persistable = True
            # NOTE(SigureMo): Why do not use `tensor.name.endswith(NON_PERSISTABLE_VAR_NAME_SUFFIX)`?
            # Because the tensor maybe copied, the name of the tensor will be appended with a new suffix.
            # Such as `lstm_0.dropout_state__non_persistable_deepcopy_204`
            if NON_PERSISTABLE_VAR_NAME_SUFFIX in tensor.name:
                is_persistable = False

            new_var = tensor._to_static_var(
                to_parameter=False, persistable=is_persistable
            )
        # add param into parameter recorder to collect all the params used in this program.
        if new_var.persistable is True:
            from paddle.jit.dy2static.program_translator import (
                ProgramTranslator,
            )

            ProgramTranslator.get_instance()._params_recorder.add(
                tensor.block.program, tensor
            )
        return new_var
    else:
        return tensor


def enabled() -> bool:
    """
    This function checks whether the program runs in dynamic graph mode or not.
    You can enable dynamic graph mode with :ref:`api_paddle_disable_static` api,
    or disable dynamic graph mode with :ref:`api_paddle_enable_static` .

    **Note**:
        ``base.dygraph.enabled`` is the alias of ``base.in_dygraph_mode``, and
        ``base.in_dygraph_mode`` is recommended to use for now.

    Returns:
        bool: Whether the program is running in dynamic graph mode.

    Examples:
        .. code-block:: python

            >>> import paddle.base as base

            >>> base.enable_dygraph()  # Now we are in dygragh mode
            >>> print(base.dygraph.enabled())
            True
            >>> base.disable_dygraph()
            >>> print(base.dygraph.enabled())
            False
    """
    # TODO(jiabin): Make this check as in_dygraph_mode when we support default eager mode.
    return framework.in_dygraph_mode()


def enable_dygraph(place: PlaceLike | None = None) -> None:
    """

    .. note::
        Dynamic graph mode is turn ON by default since paddle 2.0.0

    This API turn OFF static graph mode. You can turn ON static graph mode by `enable_static <./disable_dygraph_en.html>`_ .

    Parameters:
        place(paddle.CPUPlace|paddle.CUDAPlace|str, optional): Place to run dynamic graph. Default: None. Which means that the running place will be
            determined according to the way of paddle compilation. If ``place`` is string, It can be ``cpu``, and ``gpu:x``, where ``x`` is the
            index of the GPUs.

    return:
        None

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> print(paddle.in_dynamic_mode())
            True

            >>> paddle.enable_static()
            >>> print(paddle.in_dynamic_mode())
            False

            >>> paddle.disable_static()
            >>> print(paddle.in_dynamic_mode())
            True

    """
    global global_var
    if global_var._functional_dygraph_context_manager is None:
        global_var._functional_dygraph_context_manager = guard(
            place=_get_paddle_place(place)
        )
        global_var._functional_dygraph_context_manager.__enter__()

        # call disable_dygraph when Python exit
        CleanupFuncRegistrar.register(disable_dygraph)


def disable_dygraph() -> None:
    """

    .. note::
        Dynamic graph mode is turn ON by default since paddle 2.0.0

    This API turn ON static graph mode. You can turn ON static graph mode by `disable_static <./enable_dygraph_en.html>`_ .

    return:
        None

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> print(paddle.in_dynamic_mode())
            True

            >>> paddle.enable_static()
            >>> print(paddle.in_dynamic_mode())
            False

            >>> paddle.disable_static()
            >>> print(paddle.in_dynamic_mode())
            True

    """
    global global_var
    if global_var._functional_dygraph_context_manager is not None:
        global_var._functional_dygraph_context_manager.__exit__(*sys.exc_info())
        global_var._functional_dygraph_context_manager = None


@signature_safe_contextmanager
def _switch_tracer_mode_guard_(
    is_train: bool = True,
) -> Generator[None, None, None]:
    has_grad = core._has_grad()
    core._set_has_grad(is_train)
    try:
        yield
    finally:
        core._set_has_grad(has_grad)


@overload
def no_grad(func: None = ...) -> AbstractContextManager: ...


@overload
def no_grad(func: Callable[_InputT, _RetT]) -> Callable[_InputT, _RetT]: ...


def no_grad(func=None):
    """
    :api_attr: imperative

    Create a context which disables dygraph gradient calculation.
    In this mode, the result of every computation will have `stop_gradient=True`.

    Also functions as a decorator. (Make sure to instantiate without parenthesis.)

    Examples:

        .. code-block:: python

            >>> import numpy as np
            >>> import paddle.base as base

            >>> # use as generator

            >>> data = np.array([[2, 3], [4, 5]]).astype('float32')
            >>> with base.dygraph.guard():
            ...     l0 = paddle.nn.Linear(2, 2)  # l0.weight.gradient() is None
            ...     l1 = paddle.nn.Linear(2, 2)
            ...     with base.dygraph.no_grad():
            ...         # l1.weight.stop_gradient is False
            ...         tmp = l1.weight * 2  # tmp.stop_gradient is True
            ...     x = base.dygraph.to_variable(data)
            ...     y = l0(x) + tmp
            ...     o = l1(y)
            ...     o.backward()
            ...     print(tmp.gradient() is None)
            ...     print(l0.weight.gradient() is None)
            True
            False

            >>> @base.dygraph.no_grad
            >>> def test_layer():
            ...     with base.dygraph.guard():
            ...         inp = np.ones([3, 1024], dtype='float32')
            ...         t = base.dygraph.base.to_variable(inp)
            ...         linear1 = paddle.nn.Linear(1024, 4, bias_attr=False)
            ...         linear2 = paddle.nn.Linear(4, 4)
            ...         ret = linear1(t)
            ...         dy_ret = linear2(ret)
            ...
            >>> test_layer()

    """
    if func is None:
        return _switch_tracer_mode_guard_(is_train=False)
    else:

        @decorator.decorator
        def __impl__(
            func: Callable[_InputT, _RetT],
            *args: _InputT.args,
            **kwargs: _InputT.kwargs,
        ) -> _RetT:
            with _switch_tracer_mode_guard_(is_train=False):
                return func(*args, **kwargs)

        return __impl__(func)


class _DecoratorContextManager:
    """Allow a context manager to be used as a decorator"""

    def __call__(
        self, func: Callable[_InputT, _RetT]
    ) -> Callable[_InputT, _RetT]:
        @decorator.decorator
        def _decorate_function(func, *args, **kwargs):
            with self:
                return func(*args, **kwargs)

        @decorator.decorator
        def _decorate_generator(func, *args, **kwargs):
            gen = func(*args, **kwargs)
            with self:
                yield from gen

        if inspect.isgeneratorfunction(func):
            return _decorate_generator(func)
        else:
            return _decorate_function(func)

    def __enter__(self) -> Any:
        raise NotImplementedError

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        raise NotImplementedError

    def clone(self) -> Self:
        # override this method if your children class takes __init__ parameters
        return self.__class__()


def is_grad_enabled() -> bool:
    """
    Returns whether current dygraph gradient calculation mode is enabled.

    Returns:
        bool: True if current dygraph gradient calculation mode is enabled, otherwise false.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> # Dygraph gradient calculation mode is enabled by default.
            >>> paddle.is_grad_enabled()
            True

            >>> with paddle.set_grad_enabled(False):
            ...     paddle.is_grad_enabled()
            False

            >>> paddle.enable_static()
            >>> paddle.is_grad_enabled()
            False
    """
    return core._has_grad()


def _set_grad_enabled(mode: bool) -> None:
    core._set_has_grad(mode)


class set_grad_enabled(_DecoratorContextManager):
    """
    Create a context which enables or disables dygraph gradient calculation.

    Args:
        mode(bool): whether to enable (`True`), or disable (`False`) grad.

    Returns:
        None.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> x = paddle.to_tensor([1.], stop_gradient=False)
            >>> is_train = False
            >>> with paddle.set_grad_enabled(is_train):
            ...     y = x * 2
            >>> print(y.stop_gradient)
            True

            >>> paddle.set_grad_enabled(True)
            >>> y = x * 2
            >>> print(y.stop_gradient)
            False

            >>> paddle.set_grad_enabled(False)
            >>> y = x * 2
            >>> print(y.stop_gradient)
            True
    """

    def __init__(self, mode) -> None:
        self.prev = is_grad_enabled()
        _set_grad_enabled(mode)
        self.mode = mode

    def __enter__(self) -> None: ...

    def __exit__(self, *args: object) -> None:
        _set_grad_enabled(self.prev)

    def clone(self) -> Self:
        return self.__class__(self.mode)


class no_grad_(_DecoratorContextManager):
    """
    :api_attr: imperative

    Create a context which disables dygraph gradient calculation.
    In this mode, the result of every computation will have `stop_gradient` set
    to `True`.

    Also functions as a decorator. (Make sure to use an instance.)

    Examples:

        .. code-block:: python

            >>> import numpy as np
            >>> import paddle

            >>> # use as generator

            >>> data = np.array([[2, 3], [4, 5]]).astype('float32')
            >>> l0 = paddle.nn.Linear(2, 2)  # l0.weight.gradient() is None
            >>> l1 = paddle.nn.Linear(2, 2)
            >>> with paddle.no_grad():
            ...     # l1.weight.stop_gradient is False
            ...     tmp = l1.weight * 2  # tmp.stop_gradient is True
            >>> x = paddle.to_tensor(data)
            >>> y = l0(x) + tmp
            >>> o = l1(y)
            >>> o.backward()
            >>> print(tmp.gradient() is None)
            True
            >>> print(l0.weight.gradient() is None)
            False

            >>> # use as decorator

            >>> @paddle.no_grad()
            >>> def test_layer():
            ...     inp = np.ones([3, 1024], dtype='float32')
            ...     t = paddle.to_tensor(inp)
            ...     linear1 = paddle.nn.Linear(1024, 4, bias_attr=False)
            ...     linear2 = paddle.nn.Linear(4, 4)
            ...     ret = linear1(t)
            ...     dy_ret = linear2(ret)
            ...
            >>> test_layer()
    """

    def __enter__(self) -> None:
        self.prev = is_grad_enabled()
        _set_grad_enabled(False)

    def __exit__(self, *args: object) -> None:
        _set_grad_enabled(self.prev)


class enable_grad(_DecoratorContextManager):
    """
    :api_attr: imperative

    Create a context which enable dygraph gradient calculation,
    if it has been disabled by `no_grad` or `set_grad_enabled`.

    In this mode, the result of every computation will have `stop_gradient` set
    to `False`.

    Also functions as a decorator. (Make sure to use an instance.)

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> # use as generator

            >>> x = paddle.to_tensor([1.], stop_gradient=False)
            >>> with paddle.no_grad():
            ...     with paddle.enable_grad():
            ...         y = x * 2
            >>> assert(y.stop_gradient == False)
            >>> y.backward()
            >>> assert(x.grad is not None)

            >>> # use as decorator

            >>> @paddle.enable_grad()
            >>> def double(x):
            ...     return x * 2
            ...
            >>> with paddle.no_grad():
            ...     z = double(x)
            ...
            >>> assert(z.stop_gradient == False)
    """

    def __enter__(self) -> None:
        self.prev = is_grad_enabled()
        _set_grad_enabled(True)

    def __exit__(self, *args: object) -> None:
        _set_grad_enabled(self.prev)


@signature_safe_contextmanager
def guard(place: PlaceLike | None = None) -> Generator[None, None, None]:
    """
    :api_attr: imperative

    This context will create a dygraph context for dygraph to run, using python ``with`` statement.

    Parameters:
        place(base.CPUPlace| base.CUDAPlace|str, optional): Place to execute dygraph.
            If None, the running place will be determined according to the way of paddle compilation.
            If ``place`` is string, It can be ``cpu``, ``gpu:x`` and ``xpu:x``, where ``x`` is the
            index of the GPUs or XPUs. Default: None

    return:
        None

    Examples:

        .. code-block:: python

            >>> import numpy as np
            >>> import paddle.base as base

            >>> with base.dygraph.guard():
            ...     inp = np.ones([3, 1024], dtype='float32')
            ...     t = base.dygraph.base.to_variable(inp)
            ...     linear1 = paddle.nn.Linear(1024, 4, bias_attr=False)
            ...     linear2 = paddle.nn.Linear(4, 4)
            ...     ret = linear1(t)
            ...     dy_ret = linear2(ret)
            ...
    """
    train = framework.Program()
    startup = framework.Program()
    tracer = Tracer()

    if place is not None:
        expected_place = _get_paddle_place(place)
    else:
        expected_place = framework._current_expected_place_()

    with framework.program_guard(train, startup):
        with framework.unique_name.guard():
            with framework._dygraph_guard(tracer):
                with framework._dygraph_place_guard(expected_place):
                    yield


@framework.non_static_only
def grad(
    outputs: Tensor | Sequence[Tensor],
    inputs: Tensor | Sequence[Tensor],
    grad_outputs: Tensor | Sequence[Tensor | None] | None = None,
    retain_graph: bool | None = None,
    create_graph: bool = False,
    only_inputs: bool = True,
    allow_unused: bool = False,
    no_grad_vars: Tensor | Sequence[Tensor] | set[Tensor] | None = None,
) -> list[Tensor]:
    '''
    .. note::
        **This API is ONLY available in imperative mode.**

    This API computes the sum of gradients of `outputs` with respect to each `inputs` .

    Parameters:
        outputs (Tensor|list[Tensor]|tuple[Tensor]): the output Tensor or
            Tensor list/tuple of the graph to compute gradients.
        inputs (Tensor|list[Tensor]|tuple[Tensor]): the input Tensor or
            Tensor list/tuple of the graph to compute gradients. The returned
            values of this API are the gradients of `inputs` .
        grad_outputs (Tensor|list[Tensor|None]|tuple[Tensor|None], optional):
            initial gradient values of `outputs` . If `grad_outputs` is None,
            the initial gradient values of `outputs` would be Tensors filled with 1;
            if `grad_outputs` is not None, it must have the same length as `outputs` ,
            and in this case, the initial gradient value of the i-th `outputs` would
            be: (1) a Tensor filled with 1 when the i-th element of `grad_outputs`
            is None; (2) the i-th element of `grad_outputs` when the i-th element of
            `grad_outputs` is a Tensor. Default None.
        retain_graph (bool|None, optional): whether to retain the forward graph which
            is used to calculate the gradient. When it is True, the graph would
            be retained, in which way users can calculate backward twice for the
            same graph. When it is False, the graph would be freed. Default None,
            which means it is equal to `create_graph` .
        create_graph (bool, optional): whether to create the gradient graphs of
            the computing process. When it is True, higher order derivatives are
            supported to compute; when it is False, the gradient graphs of the
            computing process would be discarded. Default False.
        only_inputs (bool, optional): whether to only compute the gradients of
            `inputs` . If it is False, the gradients of all remaining leaf
            Tensors in the graph would be also computed and accumulated.
            If it is True, only the gradients of `inputs` would be computed.
            Default True. only_inputs=False is under development, and it is
            not supported yet.
        allow_unused (bool, optional): whether to raise error or return None if some
            Tensors of `inputs` are unreachable in the graph. If some Tensors of
            `inputs` are unreachable in the graph (i.e., their gradients are None),
            error would be raised if allow_unused=False, or None would be returned as
            their gradients if allow_unused=True. Default False.
        no_grad_vars (Tensor|list[Tensor]|tuple[Tensor]|set[Tensor], optional):
            the Tensors whose gradients are not needed to compute. Default None.

    Returns:
        list: a list of Tensors, whose length is the same as the Tensor number
        inside `inputs`, and the i-th returned Tensor is the sum of gradients of
        `outputs` with respect to the i-th `inputs`.

    Examples:
        .. code-block:: python
            :name: code-example-1

            >>> import paddle

            >>> def test_dygraph_grad(create_graph):
            ...     x = paddle.ones(shape=[1], dtype='float32')
            ...     x.stop_gradient = False
            ...     y = x * x
            ...
            ...     # Since y = x * x, dx = 2 * x
            ...     dx = paddle.grad(
            ...         outputs=[y],
            ...         inputs=[x],
            ...         create_graph=create_graph,
            ...         retain_graph=True
            ...     )[0]
            ...
            ...     z = y + dx
            ...
            ...     # If create_graph = False, the gradient of dx
            ...     # would not be backpropagated. Therefore,
            ...     # z = x * x + dx, and x.gradient() = 2 * x = 2.0
            ...
            ...     # If create_graph = True, the gradient of dx
            ...     # would be backpropagated. Therefore,
            ...     # z = x * x + dx = x * x + 2 * x, and
            ...     # x.gradient() = 2 * x + 2 = 4.0
            ...
            ...     z.backward()
            ...     return x.gradient()
            ...
            >>> print(test_dygraph_grad(create_graph=False))
            [2.]
            >>> print(test_dygraph_grad(create_graph=True))
            [4.]

        .. code-block:: python
            :name: code-example-2

            >>> import paddle

            >>> def test_dygraph_grad(grad_outputs=None):
            ...     x = paddle.to_tensor(2.0)
            ...     x.stop_gradient = False
            ...
            ...     y1 = x * x
            ...     y2 = x * 3
            ...
            ...     # If grad_outputs=None, dy1 = [1], dy2 = [1].
            ...     # If grad_outputs=[g1, g2], then:
            ...     #    - dy1 = [1] if g1 is None else g1
            ...     #    - dy2 = [1] if g2 is None else g2
            ...
            ...     # Since y1 = x * x, dx = 2 * x * dy1.
            ...     # Since y2 = x * 3, dx = 3 * dy2.
            ...     # Therefore, the final result would be:
            ...     # dx = 2 * x * dy1 + 3 * dy2 = 4 * dy1 + 3 * dy2.
            ...
            ...     dx = paddle.grad(
            ...         outputs=[y1, y2],
            ...         inputs=[x],
            ...         grad_outputs=grad_outputs)[0]
            ...
            ...     return dx.numpy()
            ...
            >>> grad_value = paddle.to_tensor(4.0)
            >>> # dy1 = [1], dy2 = [1]
            >>> print(test_dygraph_grad(None))
            7.

            >>> # dy1 = [1], dy2 = [4]
            >>> print(test_dygraph_grad([None, grad_value]))
            16.

            >>> # dy1 = [4], dy2 = [1]
            >>> print(test_dygraph_grad([grad_value, None]))
            19.

            >>> # dy1 = [3], dy2 = [4]
            >>> grad_y1 = paddle.to_tensor(3.0)
            >>> print(test_dygraph_grad([grad_y1, grad_value]))
            24.
    '''
    if in_to_static_mode():
        # In dy2static context, we call static interface `gradients`
        # to calculate grads.
        from paddle.static import gradients

        to_static_unsupported_argument_warning(
            "paddle.grad",
            ["retain_graph", "create_grad", "only_inputs", "allow_unused"],
            [retain_graph, create_graph, only_inputs, allow_unused],
            [None, False, True, False],
        )
        return gradients(outputs, inputs, grad_outputs, no_grad_vars)

    def check_in_out(in_out_list, name):
        assert in_out_list is not None, f"{name} should not be None"

        if isinstance(in_out_list, (list, tuple)):
            assert len(in_out_list) > 0, f"{name} cannot be empty"
            for each_var in in_out_list:
                assert isinstance(
                    each_var, core.eager.Tensor
                ), f"Elements of {name} must be Tensor"
            return in_out_list
        else:
            assert isinstance(
                in_out_list, core.eager.Tensor
            ), f"{name} must be Tensor or list of Tensor"
            return [in_out_list]

    outputs = check_in_out(outputs, 'outputs')
    inputs = check_in_out(inputs, 'inputs')

    if grad_outputs is not None:
        if not isinstance(grad_outputs, (list, tuple)):
            grad_outputs = [grad_outputs]

        for each_var in grad_outputs:
            if each_var is not None:
                assert isinstance(
                    each_var, core.eager.Tensor
                ), "grad_outputs must be None, a Variable or a list containing None or Variables"
    else:
        grad_outputs = []

    if len(grad_outputs) > 0:
        assert len(grad_outputs) == len(
            outputs
        ), "The length of grad_outputs must be equal to outputs"

    if no_grad_vars is None:
        no_grad_vars = []
    elif isinstance(no_grad_vars, core.eager.Tensor):
        no_grad_vars = [no_grad_vars]
    elif isinstance(no_grad_vars, (list, tuple, set)):
        no_grad_vars = list(no_grad_vars)
        for var in no_grad_vars:
            assert isinstance(
                var, core.eager.Tensor
            ), "no_grad_vars can only contains Tensor"
    else:
        raise AssertionError(
            "no_grad_vars must be None, Tensor or list/tuple/set of Tensors"
        )

    assert isinstance(create_graph, bool), "create_graph must be True or False"

    if retain_graph is None:
        retain_graph = create_graph

    assert isinstance(
        retain_graph, bool
    ), "retain_graph must be None, True or False"

    assert isinstance(allow_unused, bool), "allow_unused must be True or False"

    assert isinstance(only_inputs, bool), "only_inputs must be True or False"
    assert only_inputs, "only_inputs=False is not supported yet"

    return core.eager.run_partial_grad(
        outputs,
        inputs,
        grad_outputs,
        retain_graph,
        create_graph,
        only_inputs,
        allow_unused,
        no_grad_vars,
    )
