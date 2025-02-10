# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
# Copyright (c) 2021 NVIDIA Corporation. All rights reserved.
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
import os
import pickle
import sys
import threading
import types
import warnings
from collections import OrderedDict
from collections.abc import Sequence
from contextlib import contextmanager
from types import ModuleType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Protocol,
    TypedDict,
    TypeVar,
    overload,
)

from typing_extensions import (
    Literal,
    NotRequired,
    ParamSpec,
    TypeAlias,
    Unpack,
)

import paddle
from paddle.base import core, dygraph
from paddle.base.compiler import (
    BuildStrategy,
)
from paddle.base.dygraph.base import (
    switch_to_static_graph,
)
from paddle.base.executor import Executor, scope_guard
from paddle.base.framework import (
    EagerParamBase,
    Parameter,
    Variable,
    _current_expected_place,
    dygraph_only,
)
from paddle.base.wrapped_decorator import wrap_decorator
from paddle.framework import use_pir_api
from paddle.nn import Layer
from paddle.static.io import save_inference_model
from paddle.utils.environments import (
    BooleanEnvironmentVariable,
    EnvironmentVariableGuard,
)

from .dy2static import logging_utils
from .dy2static.convert_call_func import ConversionOptions, add_ignore_module
from .dy2static.program_translator import (
    ASTStaticFunction,
    ProgramTranslator,
    StaticFunction,
    SymbolicStaticFunction,
    unwrap_decorators,
)
from .pir_translated_layer import PIR_INFER_MODEL_SUFFIX, PirTranslatedLayer
from .translated_layer import (
    INFER_MODEL_SUFFIX,
    INFER_PARAMS_INFO_SUFFIX,
    INFER_PARAMS_SUFFIX,
    INFER_PROPERTY_SUFFIX,
    TranslatedLayer,
)

if TYPE_CHECKING:
    from paddle import Tensor
    from paddle._typing import NestedStructure
    from paddle.static import InputSpec

    class _SaveOptions(TypedDict):
        output_spec: NotRequired[Sequence[Tensor | int]]
        with_hook: NotRequired[bool]
        combine_params: NotRequired[bool]
        clip_extra: NotRequired[bool]
        skip_forward: NotRequired[bool]
        input_names_after_prune: NotRequired[list[str]]
        skip_prune_program: NotRequired[bool]
        separate_parameters: NotRequired[bool]

    class _LoadOptions(TypedDict):
        model_filename: NotRequired[str]
        params_filename: NotRequired[str]


ENV_ENABLE_SOT = BooleanEnvironmentVariable("ENABLE_FALL_BACK", True)


_LayerT = TypeVar("_LayerT", bound=Layer)
_RetT = TypeVar("_RetT")
_InputT = ParamSpec("_InputT")
Backends: TypeAlias = Literal["CINN"]


@contextmanager
def sot_mode_guard(value: bool):
    with EnvironmentVariableGuard(ENV_ENABLE_SOT, value):
        yield


def copy_decorator_attrs(original_func, decorated_obj):
    """
    Copies some necessary attributes from original function into decorated function.

    Args:
        original_func(callable): the original decorated function.
        decorated_obj(StaticFunction): the target decorated StaticFunction object.
    """
    decorator_name = "to_static"

    decorated_obj.__name__ = original_func.__name__
    decorated_obj._decorator_name = decorator_name
    if not inspect.ismethod(original_func):
        decorated_obj.__wrapped__ = original_func
    decorated_obj.__doc__ = original_func.__doc__
    if hasattr(original_func, "__module__"):
        decorated_obj.__module__ = original_func.__module__

    return decorated_obj


def ignore_module(modules: list[ModuleType]) -> None:
    """
    Adds modules that ignore transcription.
    Builtin modules that have been ignored are collections, pdb, copy, inspect, re, numpy, logging, six

    Args:
        modules (list[ModuleType]): Ignored modules that you want to add

    Examples:
        .. code-block:: python

            >>> import scipy
            >>> import astor

            >>> import paddle
            >>> from paddle.jit import ignore_module
            >>> modules = [
            ...     scipy,
            ...     astor,
            ... ]
            >>> ignore_module(modules)

    """
    add_ignore_module(modules)


def _check_and_set_backend(backend, build_strategy):
    if backend not in ['CINN', None]:
        raise ValueError(
            f"The backend of to_static should be 'CINN' or None, but received {backend}."
        )
    if backend == 'CINN':
        build_strategy.build_cinn_pass = True


class _ToStaticOptions(TypedDict):
    property: NotRequired[bool]
    full_graph: NotRequired[bool]


class _ToStaticDecorator(Protocol):
    @overload
    def __call__(self, function: _LayerT) -> _LayerT: ...

    @overload
    def __call__(
        self, function: Callable[_InputT, _RetT]
    ) -> StaticFunction[_InputT, _RetT]: ...


@overload
def to_static(
    function: _LayerT,
    input_spec: NestedStructure[InputSpec] | None = ...,
    build_strategy: BuildStrategy | None = ...,
    backend: Backends | None = ...,
    **kwargs: Unpack[_ToStaticOptions],
) -> _LayerT: ...


@overload
def to_static(
    function: Callable[_InputT, _RetT],
    input_spec: NestedStructure[InputSpec] | None = ...,
    build_strategy: BuildStrategy | None = ...,
    backend: Backends | None = ...,
    **kwargs: Unpack[_ToStaticOptions],
) -> StaticFunction[_InputT, _RetT]: ...


@overload
def to_static(
    function: None = ...,
    input_spec: NestedStructure[InputSpec] | None = ...,
    build_strategy: BuildStrategy | None = ...,
    backend: Backends | None = ...,
    **kwargs: Unpack[_ToStaticOptions],
) -> _ToStaticDecorator: ...


def to_static(
    function=None,
    input_spec=None,
    build_strategy=None,
    backend=None,
    **kwargs,
):
    """
    Converts dynamic graph APIs into static graph function APIs. Decorator
    @to_static handles the Program and Executor of static graph mode and returns
    the result as dynamic graph Tensor(s). Users could use the returned dynamic
    graph Tensor(s) to do dynamic graph training, inference, or other operations.
    If the decorated function calls other dynamic graph function, the called one
    will be converted into static graph function as well.

    Args:
        function (callable): Callable dynamic graph function. If it used as a
            decorator, the decorated function will be parsed as this parameter.
        input_spec (list[InputSpec]|tuple[InputSpec]): list/tuple of InputSpec to
            specific the shape/dtype/name information of each input Tensor.
        build_strategy (BuildStrategy|None): This argument is used to compile the
            converted program with the specified options, such as operators' fusion
            in the computational graph and memory optimization during the execution
            of the computational graph. For more information about build_strategy,
            please refer to :code:`paddle.static.BuildStrategy`. The default is None.
        backend(str, Optional): Specifies compilation backend, which can be `CINN` or
            None. When backend is `CINN`, CINN compiler will be used to speed up
            training and inference.
        kwargs: Support keys including `property`, set `property` to True if the function
            is python property.

    Returns:
        Tensor(s): containing the numerical result.

    Examples:
        .. code-block:: python

            >>> # doctest: +SKIP('`paddle.jit.to_static` can not run in xdoctest')
            >>> import paddle
            >>> from paddle.jit import to_static

            >>> @to_static
            >>> def func(x):
            ...     if paddle.mean(x) < 0:
            ...         x_v = x - 1
            ...     else:
            ...         x_v = x + 1
            ...     return x_v
            ...
            >>> x = paddle.ones([1, 2], dtype='float32')
            >>> x_v = func(x)
            >>> print(x_v)
            Tensor(shape=[1, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[2., 2.]])

    """
    property = kwargs.get("property", False)
    full_graph = kwargs.get("full_graph", None)

    def decorated(python_func):
        """
        Decorates a python function into a ASTStaticFunction object.
        """

        nonlocal full_graph
        if full_graph is None:
            flag = ENV_ENABLE_SOT.get()
            full_graph = not flag

        if sys.version_info >= (3, 14) and not full_graph:
            warnings.warn(
                "full_graph=False is not supported in Python 3.14+. Set full_graph=True automatically"
            )
            full_graph = True

        StaticClass = {
            False: SymbolicStaticFunction,
            True: ASTStaticFunction,
        }[full_graph]

        # Step 1. unwrap the function if it is already decorated.
        _, python_func = unwrap_decorators(python_func)

        # Step 2. copy some attributes from original python function.
        static_layer = copy_decorator_attrs(
            original_func=python_func,
            decorated_obj=StaticClass(
                function=python_func,
                input_spec=input_spec,
                build_strategy=build_strategy,
                property=property,
                backend=backend,
            ),
        )

        return static_layer

    build_strategy = build_strategy or BuildStrategy()
    if not isinstance(build_strategy, BuildStrategy):
        raise TypeError(
            f"Required type(build_strategy) shall be `paddle.static.BuildStrategy`, but received {type(build_strategy).__name__}"
        )
    _check_and_set_backend(backend, build_strategy)

    # for usage: `to_static(foo, ...)`
    if function is not None:
        if isinstance(function, Layer):
            if isinstance(function.forward, StaticFunction):
                class_name = function.__class__.__name__
                logging_utils.warn(
                    f"`{class_name}.forward` has already been decorated somewhere. It will be redecorated to replace previous one."
                )
            function._original_funcs["forward"] = function.forward
            function.forward = decorated(function.forward)
            return function
        else:
            return decorated(function)

    # for usage: `@to_static`
    return decorated


class _NotToStaticDecorator(Protocol):
    @overload
    def __call__(
        self, func: Callable[_InputT, _RetT]
    ) -> Callable[_InputT, _RetT]: ...

    @overload
    def __call__(self, func: None = ...) -> _NotToStaticDecorator: ...


@overload
def not_to_static(
    func: Callable[_InputT, _RetT]
) -> Callable[_InputT, _RetT]: ...


@overload
def not_to_static(func: None = ...) -> _NotToStaticDecorator: ...


def not_to_static(func=None):
    """
    A Decorator to suppresses the convention of a function.

    Args:
        func(callable): The function to decorate.

    Returns:
        callable: A function which won't be converted in Dynamic-to-Static.

    Examples:
        .. code-block:: python

            >>> # doctest: +SKIP('`paddle.jit.to_static` can not run in xdoctest')
            >>> import paddle

            >>> @paddle.jit.not_to_static
            ... def func_not_to_static(x):
            ...     res = x - 1
            ...     return res

            >>> @paddle.jit.to_static
            ... def func(x):
            ...     if paddle.mean(x) < 0:
            ...         out = func_not_to_static(x)
            ...     else:
            ...         out = x + 1
            ...     return out
            ...
            >>> x = paddle.ones([1, 2], dtype='float32')
            >>> out = func(x)
            >>> print(out)
            Tensor(shape=[1, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[2., 2.]])
    """
    if func is None:
        return not_to_static

    options = ConversionOptions(not_convert=True)
    options.attach(func)
    return func


class _SaveLoadConfig:
    def __init__(self):
        self._output_spec = None
        self._model_filename = None
        self._params_filename = None
        self._separate_params = False
        # used for `paddle.load`
        self._keep_name_table = False

        # NOTE: Users rarely use following configs, so these configs are not open to users,
        # reducing user learning costs, but we retain the configuration capabilities

        # If True, programs are modified to only support direct inference deployment.
        # Otherwise,more information will be stored for flexible optimization and re-training.
        # Currently, only True is supported
        self._export_for_deployment = True

        # If True, It will save inference program only, and do not save params of Program
        self._program_only = False
        self.with_hook = False

        # if True, multi `StaticFunction` will share params in one file.
        self.combine_params = False

        # when need to save a prune model, use input_names_after_prune to specify the inputs left after pruning
        self.input_names_after_prune = None

        # in the scene of llm-inference, pruning program can cause unexpectable result, an option to skip prune is necessary
        self.skip_prune_program = False

        # if True, the params will be saved separately in multiple files.
        self.separate_parameters = False

    @property
    def output_spec(self):
        return self._output_spec

    @output_spec.setter
    def output_spec(self, spec):
        if spec is None:
            return
        if not isinstance(spec, list):
            raise TypeError(
                f"The config `output_spec` should be 'list', but received input type is {type(input)}."
            )
            for var in spec:
                if not isinstance(var, core.eager.Tensor):
                    raise TypeError(
                        f"The element in config `output_spec` list should be 'Variable', but received element's type is {type(var)}."
                    )
        self._output_spec = spec

    @property
    def model_filename(self):
        return self._model_filename

    @model_filename.setter
    def model_filename(self, filename):
        if filename is None:
            return
        if not isinstance(filename, str):
            raise TypeError(
                f"The config `model_filename` should be str, but received input's type is {type(filename)}."
            )
        if len(filename) == 0:
            raise ValueError("The config `model_filename` is empty string.")
        self._model_filename = filename

    @property
    def params_filename(self):
        return self._params_filename

    @params_filename.setter
    def params_filename(self, filename):
        if filename is None:
            return
        if not isinstance(filename, str):
            raise TypeError(
                f"The config `params_filename` should be str, but received input's type is {type(filename)}."
            )
        if len(filename) == 0:
            raise ValueError("The config `params_filename` is empty string.")
        self._params_filename = filename

    @property
    def keep_name_table(self):
        return self._keep_name_table

    @keep_name_table.setter
    def keep_name_table(self, value):
        if value is None:
            return
        if not isinstance(value, bool):
            raise TypeError(
                f"The config `keep_name_table` should be bool value, but received input's type is {type(value)}."
            )
        self._keep_name_table = value


def _parse_save_configs(configs: _SaveOptions) -> _SaveLoadConfig:
    supported_configs = [
        "output_spec",
        "with_hook",
        "combine_params",
        "clip_extra",
        "skip_forward",
        "input_names_after_prune",
        "skip_prune_program",
        "separate_parameters",
    ]

    # input check
    for key in configs:
        if key not in supported_configs:
            raise ValueError(
                f"The additional config ({key}) of `paddle.jit.save` is not supported."
            )

    # construct inner config
    inner_config = _SaveLoadConfig()
    inner_config.output_spec = configs.get("output_spec", None)
    inner_config.with_hook = configs.get("with_hook", False)
    inner_config.combine_params = configs.get("combine_params", False)
    inner_config.clip_extra = configs.get("clip_extra", True)
    inner_config.skip_forward = configs.get("skip_forward", False)
    inner_config.input_names_after_prune = configs.get(
        "input_names_after_prune", None
    )
    inner_config.skip_prune_program = configs.get("skip_prune_program", False)
    inner_config.separate_parameters = configs.get("separate_parameters", False)

    return inner_config


def _parse_load_config(configs: _LoadOptions) -> _SaveLoadConfig:
    supported_configs = ['model_filename', 'params_filename']

    # input check
    for key in configs:
        if key not in supported_configs:
            raise ValueError(
                f"The additional config ({key}) of `paddle.jit.load` is not supported."
            )

    # construct inner config
    inner_config = _SaveLoadConfig()
    inner_config.model_filename = configs.get('model_filename', None)
    inner_config.params_filename = configs.get('params_filename', None)

    return inner_config


def _get_input_var_and_names(inputs, input_spec, input_names_after_prune):
    name_none_error = (
        "The %s's name is None. "
        "When using jit.save, please set InputSpec's name in "
        "to_static(input_spec=[]) and jit.save(input_spec=[]) "
        "and make sure they are consistent."
    )
    name_no_exists_error = (
        "The tensor `%s` does not exists. "
        "Please make sure the name of InputSpec or example Tensor "
        "in input_spec is the same as the name of InputSpec in "
        "`to_static` decorated on the Layer.forward method."
    )
    if input_names_after_prune is not None:
        input_spec = [
            x
            for x in input_spec
            if isinstance(x, paddle.static.InputSpec)
            and x.name in input_names_after_prune
        ]

    input_vars = [
        var
        for var in paddle.utils.flatten(inputs)
        if isinstance(var, (Variable, paddle.pir.Value))
    ]
    input_var_names = [
        var.name
        for var in paddle.utils.flatten(inputs)
        if isinstance(var, (Variable, paddle.pir.Value))
    ]

    if input_spec is None:
        # no prune
        return input_vars, input_var_names
    else:
        # filter out non-tensor type spec infos.
        input_spec = [
            spec
            for spec in input_spec
            if isinstance(spec, paddle.static.InputSpec)
        ]
    result_var_list = []
    result_name_list = []
    if len(input_spec) == len(input_var_names):
        # no prune
        result_var_list = input_vars
        result_name_list = input_var_names
        # if input spec name not in input_var_names, only raise warning
        for spec in input_spec:
            if spec.name is None:
                warnings.warn(name_none_error % spec)
            elif spec.name not in input_var_names:
                warnings.warn(name_no_exists_error % spec.name)
            else:
                # do nothing
                pass
    else:
        # prune
        for spec in input_spec:
            if spec.name is None:
                # name is None, the input_spec only can be InputSpec
                raise ValueError(name_none_error % spec)
            elif spec.name not in input_var_names:
                # the input_spec can be `InputSpec` or `Tensor`
                raise ValueError(name_no_exists_error % spec.name)
            else:
                result_var_list.append(spec)
                result_name_list.append(spec.name)

    return result_var_list, result_name_list


def _contains_dict(output):
    if isinstance(output, dict):
        return True
    if isinstance(output, Sequence) and not isinstance(output, str):
        return any(_contains_dict(i) for i in output)
    return False


def _get_output_vars(outputs, output_spec, with_hook=False):
    name_no_exists_error = (
        "The tensor `%s` does not exists. "
        "Please make sure the name of example Tensor "
        "in configs.output_spec is the output tensor of "
        "Layer.forward method."
    )
    output_spec_is_not_value_error = (
        "tensor `%s` is not support in pir mode, "
        "because pir value has no name sometimes, especially as output,"
        "so we can't check tensor's name with output var name, please"
        "change as pir.value(to_static layer's output)"
        "or int(the position of to_static layer's output)"
    )
    if output_spec and with_hook:
        raise RuntimeError(
            "Currently not support specify output_spec while founding pre/post hooks in your outermost layer."
        )
    result_list = []
    if _contains_dict(outputs):
        warnings.warn(
            "Found 'dict' in given outputs, the values will be returned in a sequence sorted in lexicographical order by their keys."
        )
    if use_pir_api():
        from paddle.autograd.backward_utils import ValueSet

        for var in paddle.utils.flatten(outputs):
            if isinstance(var, paddle.pir.Value) and var not in ValueSet(
                result_list
            ):
                result_list.append(var)

        if output_spec is not None:
            output_size = len(result_list)
            if len(output_spec) == output_size:
                for var in output_spec:
                    if not isinstance(var, (paddle.pir.Value, int)):
                        warnings.warn(output_spec_is_not_value_error % var.name)
                    else:
                        if var not in ValueSet(result_list):
                            warnings.warn(name_no_exists_error % var.name)
            else:
                result_set = ValueSet(result_list)
                part_result_list = []
                for var in output_spec:
                    if isinstance(var, paddle.pir.Value):
                        if var not in result_set:
                            raise ValueError(name_no_exists_error % var.name)
                        else:
                            part_result_list.append(var)
                    elif isinstance(var, int):
                        if var >= output_size:
                            raise ValueError(
                                "position %d should smaller than output's size % d",
                                var,
                                output_size,
                            )
                        else:
                            part_result_list.append(result_list[var])

                    else:
                        raise ValueError(
                            output_spec_is_not_value_error % var.name
                        )

                return part_result_list
    else:
        output_vars_dict = OrderedDict()
        for var in paddle.utils.flatten(outputs):
            if isinstance(var, (Variable)):
                output_vars_dict[var.name] = var
        if output_spec is None:
            result_list = list(output_vars_dict.values())
        elif output_spec is not None and len(output_spec) == len(
            output_vars_dict
        ):
            result_list = list(output_vars_dict.values())
            for var in output_spec:
                if var.name not in output_vars_dict:
                    warnings.warn(name_no_exists_error % var.name)
        else:
            for var in output_spec:
                if var.name not in output_vars_dict:
                    raise ValueError(name_no_exists_error % var.name)
                else:
                    result_list.append(output_vars_dict[var.name])

    return result_list


# NOTE(chenweihang): [ Handling of use cases of API paddle.jit.load ]
# `paddle.jit.load` may be used to load saved results of:
# 1. Expected cases:
#   - paddle.jit.save
#   - paddle.static.save_inference_model
# 2. Error cases:
#   - paddle.save: no .pdmodel for prefix
#   - paddle.static.save: no .pdiparams but .pdparams exists
#   - paddle.base.io.save_params/save_persistables: no __model__
# TODO(chenweihang): polish error message in above error cases
def _build_load_path_and_config(path, config):
    # NOTE(chenweihang): If both [prefix save format] and [directory save format] exist,
    # raise error, avoid confusing behavior
    if use_pir_api():
        model_suffix = PIR_INFER_MODEL_SUFFIX
    else:
        model_suffix = INFER_MODEL_SUFFIX
    prefix_format_path = path + model_suffix
    prefix_format_exist = os.path.exists(prefix_format_path)
    directory_format_exist = os.path.isdir(path)
    if prefix_format_exist and directory_format_exist:
        raise ValueError(
            f"The {path}.pdmodel and {path} directory exist at the same time, "
            "don't know which one to load, please make sure that the specified target "
            "of ``path`` is unique."
        )
    elif not prefix_format_exist and not directory_format_exist:
        raise ValueError(
            f"The ``path`` ({path}) to load model not exists. "
            "Please make sure that *.pdmodel exists or "
            "don't using ``skip_forward=True`` to jit.save."
        )
    else:
        if prefix_format_exist:
            file_prefix = os.path.basename(path)
            model_path = os.path.dirname(path)
            if config.model_filename is not None:
                warnings.warn(
                    "When loading the result saved with the "
                    "specified file prefix, the ``model_filename`` config does "
                    "not take effect."
                )
            config.model_filename = file_prefix + model_suffix
            if config.params_filename is not None:
                warnings.warn(
                    "When loading the result saved with the "
                    "specified file prefix, the ``params_filename`` config does "
                    "not take effect."
                )
            config.params_filename = file_prefix + INFER_PARAMS_SUFFIX
        else:
            # Compatible with the old save_inference_model format
            model_path = path

    return model_path, config


_save_pre_hooks_lock = threading.Lock()
_save_pre_hooks = []


class HookRemoveHelper:
    """A HookRemoveHelper that can be used to remove hook."""

    def __init__(self, hook):
        self._hook = hook

    def remove(self):
        _remove_save_pre_hook(self._hook)


def _register_save_pre_hook(hook):
    """
    Register a save pre-hook for `paddle.jit.save`.
    This hook will be executed before `save` function has been invoked.

    hook(layer, input_spec, configs) -> None
    - layer (Layer|function): This argument is corresponding to `layer` in `paddle.jit.save`.
    - input_spec (list or tuple[InputSpec|Tensor|Python built-in variable]): This argument is corresponding to `input_spec` in `paddle.jit.save`.
    - configs (dict): This argument is corresponding to `configs` in `paddle.jit.save`.

    Args:
        hook(function): a function registered as a save pre-hook

    Returns:
        HookRemoveHelper: a HookRemoveHelper object that can be used to remove the added hook by calling `hook_remove_helper.remove()`.

    Examples:
        .. code-block:: python

            >>> # doctest: +SKIP('`paddle.jit.api.to_static` can not run in xdoctest')
            >>> import numpy as np
            >>> import paddle

            >>> IMAGE_SIZE = 256
            >>> CLASS_NUM = 10

            >>> class LinearNet(paddle.nn.Layer):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self._linear = paddle.nn.Linear(IMAGE_SIZE, CLASS_NUM)
            ...
            ...     def forward(self, x):
            ...         return self._linear(x)
            ...
            >>> saving_count = 0
            >>> def save_pre_hook(layer, input_spec, configs):
            ...     global saving_count
            ...     saving_count += 1
            ...
            >>> remove_handler = paddle.jit.api._register_save_pre_hook(save_pre_hook)

            >>> layer = LinearNet()
            >>> paddle.jit.save(layer, "/tmp", [paddle.static.InputSpec(shape=[-1, IMAGE_SIZE])])
            >>> print(saving_count)
            1

            >>> remove_handler.remove()
            >>> paddle.jit.save(layer, "/tmp", [paddle.static.InputSpec(shape=[-1, IMAGE_SIZE])])
            >>> print(saving_count)
            1
    """
    global _save_pre_hooks_lock
    global _save_pre_hooks
    _save_pre_hooks_lock.acquire()
    if hook not in _save_pre_hooks:
        _save_pre_hooks.append(hook)
    _save_pre_hooks_lock.release()
    return HookRemoveHelper(hook)


def _clear_save_pre_hooks():
    global _save_pre_hooks_lock
    global _save_pre_hooks
    _save_pre_hooks_lock.acquire()
    _save_pre_hooks.clear()
    _save_pre_hooks_lock.release()


def _remove_save_pre_hook(hook):
    global _save_pre_hooks_lock
    global _save_pre_hooks
    _save_pre_hooks_lock.acquire()
    if hook in _save_pre_hooks:
        _save_pre_hooks.remove(hook)
    _save_pre_hooks_lock.release()


class _SaveFunction(Protocol):
    def __call__(
        self,
        layer: Layer | Callable[..., Any],
        path: str,
        input_spec: Sequence[InputSpec | Tensor | object] | None = ...,
        **configs: Unpack[_SaveOptions],
    ) -> None: ...


@wrap_decorator
def _run_save_pre_hooks(func: _SaveFunction) -> _SaveFunction:
    def wrapper(
        layer: Layer | Callable[..., Any],
        path: str,
        input_spec: Sequence[InputSpec | Tensor | object] | None = None,
        **configs: Unpack[_SaveOptions],
    ) -> None:
        global _save_pre_hooks
        for hook in _save_pre_hooks:
            hook(layer, input_spec, configs)
        func(layer, path, input_spec, **configs)

    return wrapper


def _save_property(filename: str, property_vals: list[tuple[Any, str]]):
    """class property serialization.

    Args:
        filename (str): *.meta
        property_vals (list[tuple[Any, str]]): class property.
    """

    def set_property(meta, key, val):
        if isinstance(val, float):
            meta.set_float(key, val)
        elif isinstance(val, int):
            meta.set_int(key, val)
        elif isinstance(val, str):
            meta.set_string(key, val)
        elif isinstance(val, (tuple, list)):
            if isinstance(val[0], float):
                meta.set_floats(key, val)
            elif isinstance(val[0], int):
                meta.set_ints(key, val)
            elif isinstance(val[0], str):
                meta.set_strings(key, val)
        else:
            raise ValueError(f"Note support val type: {type(val)}")

    with open(filename, 'wb') as f:
        meta = paddle.framework.core.Property()
        for item in property_vals:
            val, key = item[0], item[1]
            set_property(meta, key, val)
        f.write(meta.serialize_to_string())


def _get_function_names_from_layer(layer: Layer) -> list[str]:
    cls = layer.__class__
    return [
        member_name
        for member_name, member in inspect.getmembers(cls)
        if (
            inspect.isfunction(member)
            or inspect.ismethod(member)
            or inspect.ismethoddescriptor(member)
        )
    ]


@_run_save_pre_hooks
@switch_to_static_graph
def save(
    layer: Layer | Callable[..., Any],
    path: str,
    input_spec: Sequence[InputSpec | Tensor | object] | None = None,
    **configs: Unpack[_SaveOptions],
) -> None:
    """
    Saves input Layer or function as ``paddle.jit.TranslatedLayer``
    format model, which can be used for inference or fine-tuning after loading.

    It will save the translated program and all related persistable
    variables of input Layer to given ``path`` .

    ``path`` is the prefix of saved objects, and the saved translated program file
    suffix is ``.pdmodel`` , the saved persistable variables file suffix is ``.pdiparams`` ,
    and here also saved some additional variable description information to a file,
    its suffix is ``.pdiparams.info``, these additional information is used in fine-tuning.

    The saved model can be loaded by follow APIs:
      - ``paddle.jit.load``
      - ``paddle.static.load_inference_model``
      - Other C++ inference APIs

    .. note::
        When using ``paddle.jit.save`` to save a function, parameters will not be saved. If you have to
        save the parameter, please pass the Layer containing function and parameter to ``paddle.jit.save``.

    Args:
        layer (Layer|function): The Layer or function to be saved.
        path (str): The path prefix to save model. The format is ``dirname/file_prefix`` or ``file_prefix``.
        input_spec (list or tuple[InputSpec|Tensor|Python built-in variable], optional): Describes the input of the saved model's forward
            method, which can be described by InputSpec or example Tensor. Moreover, we support to specify non-tensor type argument,
            such as int, float, string, or list/dict of them.If None, all input variables of
            the original Layer's forward method would be the inputs of the saved model. Default None.
        **configs (dict, optional): Other save configuration options for compatibility. We do not
            recommend using these configurations, they may be removed in the future. If not necessary,
            DO NOT use them. Default None.
            The following options are currently supported:
            (1) output_spec (list[Tensor|Value|int]): Selects the output targets of the saved model,
            By default, all return variables of original Layer's forward method are kept as the
            output of the saved model. If the provided ``output_spec`` list is not all output variables,
            the saved model will be pruned according to the given ``output_spec`` list.
            in pir mode, Tensor is not supported, because value has no name in most cases,
            which can't be used to judge which tensor corresponds to which value; the value can't be found
            if the saved program is not the same as the program that includes output_spec, so we need to
            use the position of the output.

    Returns:
        None

    Examples:
        .. code-block:: python

            >>> # doctest: +SKIP('`paddle.jit.to_static` can not run in xdoctest')
            >>> # example 1: save layer
            >>> import numpy as np
            >>> import paddle
            >>> import paddle.nn as nn
            >>> import paddle.optimizer as opt

            >>> BATCH_SIZE = 16
            >>> BATCH_NUM = 4
            >>> EPOCH_NUM = 4

            >>> IMAGE_SIZE = 784
            >>> CLASS_NUM = 10

            >>> # define a random dataset
            >>> class RandomDataset(paddle.io.Dataset): # type: ignore[type-arg]
            ...     def __init__(self, num_samples):
            ...         self.num_samples = num_samples
            ...
            ...     def __getitem__(self, idx):
            ...         image = np.random.random([IMAGE_SIZE]).astype('float32')
            ...         label = np.random.randint(0, CLASS_NUM - 1, (1, )).astype('int64')
            ...         return image, label
            ...
            ...     def __len__(self):
            ...         return self.num_samples

            >>> class LinearNet(nn.Layer):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self._linear = nn.Linear(IMAGE_SIZE, CLASS_NUM)
            ...
            ...     @paddle.jit.to_static
            ...     def forward(self, x):
            ...         return self._linear(x)

            >>> def train(layer, loader, loss_fn, opt):
            ...     for epoch_id in range(EPOCH_NUM):
            ...         for batch_id, (image, label) in enumerate(loader()):
            ...             out = layer(image)
            ...             loss = loss_fn(out, label)
            ...             loss.backward()
            ...             opt.step()
            ...             opt.clear_grad()
            ...             print("Epoch {} batch {}: loss = {}".format(
            ...                 epoch_id, batch_id, np.mean(loss.numpy())))

            >>> # 1. train & save model.

            >>> # create network
            >>> layer = LinearNet()
            >>> loss_fn = nn.CrossEntropyLoss()
            >>> adam = opt.Adam(learning_rate=0.001, parameters=layer.parameters())

            >>> # create data loader
            >>> dataset = RandomDataset(BATCH_NUM * BATCH_SIZE)
            >>> loader = paddle.io.DataLoader(dataset,
            ...     batch_size=BATCH_SIZE,
            ...     shuffle=True,
            ...     drop_last=True,
            ...     num_workers=2
            ... )

            >>> # train
            >>> train(layer, loader, loss_fn, adam)

            >>> # save
            >>> path = "example_model/linear"
            >>> paddle.jit.save(layer, path)

            >>> # example 2: save function
            >>> import paddle
            >>> from paddle.static import InputSpec


            >>> def save_function():
            ...     @paddle.jit.to_static
            ...     def fun(inputs):
            ...         return paddle.tanh(inputs)
            ...
            ...     path = 'test_jit_save_load_function_1/func'
            ...     inps = paddle.rand([3, 6])
            ...     origin = fun(inps)
            ...
            ...     paddle.jit.save(fun, path)
            ...     load_func = paddle.jit.load(path)
            ...
            ...     load_result = load_func(inps)
            ...     print((load_result - origin).abs().max() < 1e-10)

            >>> save_function()
    """
    # 1. input build & check
    prog_translator = ProgramTranslator()
    is_prim_infer = core._is_fwd_prim_enabled() and core._is_bwd_prim_enabled()
    if not prog_translator.enable_to_static:
        raise RuntimeError(
            "The paddle.jit.save doesn't work when setting 'paddle.jit.enable_to_static' to False."
        )

    if not (
        isinstance(layer, (Layer, StaticFunction)) or inspect.isfunction(layer)
    ):
        raise TypeError(
            f"The input of paddle.jit.save should be 'Layer' or 'Function', but received input type is {type(layer)}."
        )
    elif inspect.isfunction(layer) or isinstance(layer, StaticFunction):
        warnings.warn(
            'What you save is a function, and `jit.save` will generate the name of the model file according to `path` you specify. When loading these files with `jit.load`, you get a `TranslatedLayer` whose inference result is the same as the inference result of the function you saved.'
        )

    # NOTE(chenweihang): If the input layer be wrapped by DataParallel,
    # the args and kwargs of forward method will can't be parsed by
    # function_spec, so here we save DataParallel._layers instead
    # DataParallel it self
    # NOTE(chenweihang): using inner_layer, do not change input layer
    if isinstance(layer, paddle.DataParallel):
        inner_layer = layer._layers
    else:
        inner_layer = layer

    # path check
    file_prefix = os.path.basename(path)
    if file_prefix == "":
        raise ValueError(
            "The input path MUST be format of dirname/file_prefix "
            "[dirname\\file_prefix in Windows system], but received "
            "file_prefix is empty string."
        )

    dirname = os.path.dirname(path)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname)

    # avoid change user given input_spec
    inner_input_spec = None
    if input_spec is not None:
        if isinstance(layer, Layer):
            for member_name in _get_function_names_from_layer(inner_layer):
                static_func = getattr(inner_layer, member_name, None)
                if (
                    isinstance(static_func, StaticFunction)
                    and 'forward' != member_name
                ):
                    raise ValueError(
                        f"If there are static functions other than 'forward' that need to be saved, the input 'input_spec' should be None, but received the type of 'input_spec' is {type(input_spec)}."
                    )

        if not isinstance(input_spec, (list, tuple)):
            raise TypeError(
                f"The input input_spec should be 'list', but received input_spec's type is {type(input_spec)}."
            )
        inner_input_spec = []
        for var in paddle.utils.flatten(input_spec):
            if isinstance(var, paddle.static.InputSpec):
                inner_input_spec.append(var)
            elif isinstance(
                var, (core.eager.Tensor, Variable, paddle.pir.Value)
            ):
                inner_input_spec.append(
                    paddle.static.InputSpec.from_tensor(var)
                )
            else:
                # NOTE(Aurelius84): Support non-Tensor type in `input_spec`.
                inner_input_spec.append(var)

    # parse configs
    configs = _parse_save_configs(configs)
    # whether outermost layer has pre/post hook, if does, we need also save
    # these operators in program.
    with_hook = configs.with_hook
    combine_params = configs.combine_params
    if combine_params:
        configs._program_only = True

    scope = core.Scope()
    extra_var_info = {}
    if isinstance(layer, Layer):
        functions = list(set(_get_function_names_from_layer(inner_layer)))
        functions = sorted(functions)
        if inner_layer._forward_pre_hooks or inner_layer._forward_post_hooks:
            with_hook = True
    else:
        # layer is function
        functions = [layer]

    combine_vars = {}
    combine_program = []
    property_vals = []  # (value, key)
    concrete_program = None
    for attr_func in functions:
        if isinstance(layer, Layer):
            static_func = get_ast_static_function(
                getattr(inner_layer, attr_func, None)
            )
            if isinstance(static_func, StaticFunction):
                if static_func.is_property:
                    # property method to be exported
                    immediate_val = static_func()
                    property_vals.append(
                        (
                            immediate_val,
                            layer.__class__.__name__ + '.' + attr_func,
                        )
                    )
                    continue

                concrete_program = (
                    static_func.concrete_program_specify_input_spec(
                        inner_input_spec,
                        with_hook=with_hook,
                        is_prim_infer=is_prim_infer,
                    )
                )

            elif 'forward' == attr_func:
                if configs.skip_forward:
                    # do not jit.save forward function
                    continue

                # transform in jit.save, if input_spec is incomplete, declarative will throw error
                # inner_input_spec is list[InputSpec], it should be packed with same structure
                # as original input_spec here.
                if inner_input_spec:
                    inner_input_spec = paddle.utils.pack_sequence_as(
                        input_spec, inner_input_spec
                    )
                static_forward = to_static(
                    inner_layer.forward,
                    input_spec=inner_input_spec,
                    full_graph=True,
                )

                concrete_program = (
                    static_forward.concrete_program_specify_input_spec(
                        with_hook=with_hook, is_prim_infer=is_prim_infer
                    )
                )
                # the input_spec has been used in declarative, which is equal to
                # @to_static with input_spec and jit.save without input_spec,
                # avoid needless warning
                inner_input_spec = None
            else:
                continue
        else:
            # When layer is a function
            if isinstance(attr_func, StaticFunction):
                static_func = get_ast_static_function(attr_func)

                if static_func.is_property:
                    # property method to be exported
                    immediate_val = static_func()
                    property_vals.append((immediate_val, static_func))
                    continue

                concrete_program = (
                    static_func.concrete_program_specify_input_spec(
                        inner_input_spec, is_prim_infer=is_prim_infer
                    )
                )
            else:
                static_func = get_ast_static_function(attr_func)
                if inner_input_spec:
                    inner_input_spec = paddle.utils.pack_sequence_as(
                        input_spec, inner_input_spec
                    )
                static_function = to_static(
                    static_func,
                    input_spec=inner_input_spec,
                    full_graph=True,
                )
                concrete_program = static_function.concrete_program

                if static_function.class_instance is None:
                    warnings.warn(
                        f'`jit.save` will only save the `Program`, not the parameters. If you have to save the parameters, please make sure that {layer} is a member function of `paddle.nn.Layer` and the saved parameters are in `state_dict`'
                    )

        # when save multi `StaticFunction`, all `StaticFunction` share params.
        dygraph_state_dict = None
        if isinstance(inner_layer, Layer):
            dygraph_state_dict = inner_layer.to_static_state_dict()
        elif isinstance(attr_func, StaticFunction):
            if static_func.class_instance:
                dygraph_state_dict = (
                    static_func.class_instance.to_static_state_dict()
                )

        if dygraph_state_dict:
            # NOTE(chenweihang): we maintain the mapping of variable name to
            # structured name, the buffer variable (non-persistable)
            # saved to inference program may not need by dygraph Layer,
            # we only record the state_dict variable's structured name
            state_names_dict = {}
            state_var_dict = {}
            for structured_name, var in dygraph_state_dict.items():
                state_names_dict[var.name] = structured_name
                state_var_dict[var.name] = var
        # 3. share parameters from Layer to scope & record var info
        with dygraph.guard():
            if use_pir_api():
                for tensor, value in zip(*concrete_program.parameters):
                    if not value.persistable:
                        continue
                    param_or_buffer_tensor = scope.var(value.name).get_tensor()
                    src_tensor = (
                        state_var_dict[tensor.name].value().get_tensor()
                    )
                    param_or_buffer_tensor._share_data_with(src_tensor)

            else:
                for param_or_buffer in concrete_program.parameters:
                    # share to scope
                    if param_or_buffer.type == core.VarDesc.VarType.VOCAB:
                        scr_tensor = param_or_buffer.value().get_map_tensor()
                        tgt_var = scope.var(param_or_buffer.name)
                        tgt_var.set_vocab(scr_tensor)
                    else:
                        param_or_buffer_tensor = scope.var(
                            param_or_buffer.name
                        ).get_tensor()
                        # src_tensor = param_or_buffer.value().get_tensor()
                        src_tensor = (
                            state_var_dict[param_or_buffer.name]
                            .value()
                            .get_tensor()
                        )
                        param_or_buffer_tensor._share_data_with(src_tensor)
                    # record var info
                    if param_or_buffer.name not in extra_var_info:
                        extra_info_dict = {}
                        if param_or_buffer.name in state_names_dict:
                            extra_info_dict['structured_name'] = (
                                state_names_dict[param_or_buffer.name]
                            )
                        extra_info_dict['stop_gradient'] = (
                            param_or_buffer.stop_gradient
                        )
                        if isinstance(param_or_buffer, EagerParamBase):
                            extra_info_dict['trainable'] = (
                                param_or_buffer.trainable
                            )
                        extra_var_info[param_or_buffer.name] = extra_info_dict
        # 4. build input & output of save_inference_model
        # NOTE(chenweihang): [ Get input variables name ]
        # There are two cases, whether to prune the inputs or not
        # - not prune inputs (recommend):
        #   - the len(input_spec) == len((concrete_program.inputs) - 1
        #   - here can use concrete_program.inputs directly
        # - prune inputs:
        #   - the input_spec length < len((concrete_program.inputs) - 1
        #   - the input_spec's name should be in concrete_program.inputs

        input_vars, input_var_names = _get_input_var_and_names(
            concrete_program.inputs,
            inner_input_spec,
            configs.input_names_after_prune,
        )

        # NOTE(chenweihang): [ Get output variables ]
        # the rule is like [ Get input variables name ]. For output var,
        # we only support Tensor spec, and actually, we only need the
        # var name of output, and we don't recommended to use output_spec
        # NOTE(Ruting): in pir mode, Tensor is not supported, because value has no name in most cases,
        # which can't be used to judge which tensor corresponds to which value; the value can't be found
        # if the saved program is not the same as the program that includes output_spec, so we need to
        # use the position of the output.

        output_vars = _get_output_vars(
            concrete_program.outputs, configs.output_spec, with_hook
        )

        # 5. save inference model
        # construct new save_inference_model arguments
        model_path = dirname
        # NOTE(chenweihang): because prefix contains model and params filename,
        # so we don't support set model_filename & params_filename
        if 'forward' == attr_func or not isinstance(layer, Layer):
            model_filename = file_prefix + INFER_MODEL_SUFFIX
            params_filename = file_prefix + INFER_PARAMS_SUFFIX
            path_prefix = file_prefix
        else:
            model_filename = file_prefix + '.' + attr_func + INFER_MODEL_SUFFIX
            params_filename = (
                file_prefix + '.' + attr_func + INFER_PARAMS_SUFFIX
            )
            path_prefix = file_prefix + '.' + attr_func
        file_path = os.path.join(model_path, path_prefix)
        with scope_guard(scope):
            if use_pir_api():
                value_map = paddle.pir.IrMapping()
                clone_program = concrete_program.main_program.clone(value_map)
                clone_input_vars = []
                for v in input_vars:
                    if type(v) is paddle.static.InputSpec:
                        name = v.name
                        for op in clone_program.global_block().ops:
                            if (
                                op.name() == 'pd_op.data'
                                and op.attrs()["name"] == name
                            ):
                                clone_input_vars.append(op.result(0))
                    else:
                        clone_input_vars.append(value_map.look_up(v))

                clone_output_vars = [value_map.look_up(v) for v in output_vars]

            else:
                input_vars = [
                    concrete_program.main_program.global_block().var(name)
                    for name in input_var_names
                ]
                clone_program = concrete_program.main_program.clone()
                clone_input_vars = input_vars
                clone_output_vars = output_vars
            save_inference_model(
                path_prefix=file_path,
                feed_vars=clone_input_vars,
                fetch_vars=clone_output_vars,
                executor=Executor(_current_expected_place()),
                program=clone_program,
                clip_extra=configs.clip_extra,
                skip_prune_program=configs.skip_prune_program,
                separate_parameters=configs.separate_parameters,
            )

        if combine_params:
            if use_pir_api():
                # NOTE(Ruting): concrete_program has been pruned when init partialProgramLayer,
                # so we do not need to prune again.

                for var in concrete_program.main_program.list_vars():
                    if var.persistable:
                        combine_vars[var.name] = var
                # NOTE(Ruting): concrete_program will delete after this loop item,
                # value delete at the same time, so we use list to Extend its lifecycle
                combine_program.append(concrete_program.main_program)
            else:
                clone_main_program = concrete_program.main_program.clone()
                clone_main_program = clone_main_program._prune_with_input(
                    input_var_names, output_vars
                )
                for block in clone_main_program.blocks:
                    combine_vars.update(block.vars)

    # save shared params
    if combine_params:
        # sort vars by name
        combine_vars = sorted(combine_vars.items(), key=lambda item: item[0])
        ordered_vars = []
        for name, var in combine_vars:
            ordered_vars.append(var)

        params_filename = file_prefix + INFER_PARAMS_SUFFIX
        with scope_guard(scope):
            if use_pir_api():
                paddle.static.save_vars(
                    Executor(_current_expected_place()),
                    dirname=model_path,
                    vars=ordered_vars,
                    filename=params_filename,
                )
            else:
                paddle.static.save_vars(
                    Executor(_current_expected_place()),
                    dirname=model_path,
                    vars=list(
                        filter(
                            paddle.framework.io_utils.is_persistable,
                            ordered_vars,
                        )
                    ),
                    filename=params_filename,
                )
        # save property
        property_save_path = os.path.join(
            os.path.normpath(model_path), file_prefix + INFER_PROPERTY_SUFFIX
        )
        _save_property(property_save_path, property_vals)

    # NOTE(chenweihang): [ Save extra variable info ]
    # save_inference_model will lose some important variable information, including:
    #   - Variable name and correspondence (when saved variables as one file)
    #   - Variable.stop_gradient information
    #   - Which persistent variable are parameter and which are not
    #   - Parameter.trainable information
    #
    # The lost information cannot be recovered when it is loaded again,
    # so if we want to perform fine-tune after loading, we may need to
    # configure redundant information to proceed.
    #
    # Due to compatibility issues, we cannot change the original storage structure,
    # but we can save these information in `jit.save` without changing the original
    # storage to improve user experience. So we save extra information into
    # file `***.pdiparams.info`

    # "layer" can only be Layer or function or StaticFunction.
    contain_parameter = False
    if concrete_program is not None:
        for var in concrete_program.main_program.list_vars():
            if use_pir_api():
                is_persistable = (
                    var.get_defining_op().has_attr("persistable")
                    and var.get_defining_op().attrs()["persistable"] is True
                )
                contain_parameter |= is_persistable
            else:
                contain_parameter |= isinstance(var, Parameter)

    if (isinstance(layer, Layer) or contain_parameter) and extra_var_info:
        with scope_guard(scope):
            extra_var_info_path = path + INFER_PARAMS_INFO_SUFFIX
            with open(extra_var_info_path, 'wb') as f:
                pickle.dump(extra_var_info, f, protocol=2)
    scope.erase(scope.local_var_names())


@dygraph_only
def load(
    path: str, **configs: Unpack[_LoadOptions]
) -> TranslatedLayer | PirTranslatedLayer:
    """
    :api_attr: imperative

    Load model saved by ``paddle.jit.save`` or ``paddle.static.save_inference_model`` or
    paddle 1.x API ``paddle.static.save_inference_model`` as ``paddle.jit.TranslatedLayer``,
    then performing inference or fine-tune training.

    .. note::
        If you load model saved by ``paddle.static.save_inference_model`` ,
        there will be the following limitations when using it in fine-tuning:
        1. Imperative mode do not support DenseTensor. All original model's feed targets or parameters that depend on LoD are temporarily unavailable.
        2. All saved model's feed targets need to be passed into TranslatedLayer's forward function.
        3. The variable's ``stop_gradient`` information is lost and can not be recovered.
        4. The parameter's ``trainable`` information is lost and can not be recovered.

    Args:
        path (str): The path prefix to load model. The format is ``dirname/file_prefix`` or ``file_prefix`` .
        **configs (dict, optional): Other load configuration options for compatibility. We do not
            recommend using these configurations, they may be removed in the future. If not necessary,
            DO NOT use them. Default None.
            The following options are currently supported:
            (1) model_filename (str): The inference model file name of the paddle 1.x
            ``save_inference_model`` save format. Default file name is :code:`__model__` .
            (2) params_filename (str): The persistable variables file name of the paddle 1.x
            ``save_inference_model`` save format. No default file name, save variables separately
            by default.


    Returns:
        TranslatedLayer: A Layer object can run saved translated model.

    Examples:
        1. Load model saved by ``paddle.jit.save`` then performing inference and fine-tune training.

            .. code-block:: python
                :name: code-example1

                >>> # doctest: +SKIP('`paddle.jit.to_static` can not run in xdoctest')
                >>> import numpy as np
                >>> import paddle
                >>> import paddle.nn as nn
                >>> import paddle.optimizer as opt

                >>> BATCH_SIZE = 16
                >>> BATCH_NUM = 4
                >>> EPOCH_NUM = 4

                >>> IMAGE_SIZE = 784
                >>> CLASS_NUM = 10

                >>> # define a random dataset
                >>> class RandomDataset(paddle.io.Dataset): # type: ignore[type-arg]
                ...     def __init__(self, num_samples):
                ...         self.num_samples = num_samples
                ...
                ...     def __getitem__(self, idx):
                ...         image = np.random.random([IMAGE_SIZE]).astype('float32')
                ...         label = np.random.randint(0, CLASS_NUM - 1, (1, )).astype('int64')
                ...         return image, label
                ...
                ...     def __len__(self):
                ...         return self.num_samples

                >>> class LinearNet(nn.Layer):
                ...     def __init__(self):
                ...         super().__init__()
                ...         self._linear = nn.Linear(IMAGE_SIZE, CLASS_NUM)
                ...
                ...     @paddle.jit.to_static
                ...     def forward(self, x):
                ...         return self._linear(x)
                ...
                >>> def train(layer, loader, loss_fn, opt):
                ...     for epoch_id in range(EPOCH_NUM):
                ...         for batch_id, (image, label) in enumerate(loader()):
                ...             out = layer(image)
                ...             loss = loss_fn(out, label)
                ...             loss.backward()
                ...             opt.step()
                ...             opt.clear_grad()
                ...             print("Epoch {} batch {}: loss = {}".format(
                ...                 epoch_id, batch_id, np.mean(loss.numpy())))

                >>> # 1. train & save model.

                >>> # create network
                >>> layer = LinearNet()
                >>> loss_fn = nn.CrossEntropyLoss()
                >>> adam = opt.Adam(learning_rate=0.001, parameters=layer.parameters())

                >>> # create data loader
                >>> dataset = RandomDataset(BATCH_NUM * BATCH_SIZE)
                >>> loader = paddle.io.DataLoader(
                ...     dataset,
                ...     batch_size=BATCH_SIZE,
                ...     shuffle=True,
                ...     drop_last=True,
                ...     num_workers=2
                ... )

                >>> # train
                >>> train(layer, loader, loss_fn, adam)

                >>> # save
                >>> path = "example_model/linear"
                >>> paddle.jit.save(layer, path)

                >>> # 2. load model

                >>> # load
                >>> loaded_layer = paddle.jit.load(path)

                >>> # inference
                >>> loaded_layer.eval()
                >>> x = paddle.randn([1, IMAGE_SIZE], 'float32')
                >>> pred = loaded_layer(x)

                >>> # fine-tune
                >>> loaded_layer.train()
                >>> adam = opt.Adam(learning_rate=0.001, parameters=loaded_layer.parameters())
                >>> train(loaded_layer, loader, loss_fn, adam)


        2. Load model saved by ``paddle.static.save_inference_model`` then performing and fine-tune training.

            .. code-block:: python
                :name: code-example2

                >>> # doctest: +SOLO('can not use multiprocessing testing `DataLoader`')
                >>> import numpy as np
                >>> import paddle
                >>> import paddle.static as static
                >>> import paddle.nn as nn
                >>> import paddle.optimizer as opt
                >>> import paddle.nn.functional as F

                >>> BATCH_SIZE = 16
                >>> BATCH_NUM = 4
                >>> EPOCH_NUM = 4

                >>> IMAGE_SIZE = 784
                >>> CLASS_NUM = 10

                >>> # define a random dataset
                >>> class RandomDataset(paddle.io.Dataset): # type: ignore[type-arg]
                ...     def __init__(self, num_samples):
                ...         self.num_samples = num_samples
                ...
                ...     def __getitem__(self, idx):
                ...         image = np.random.random([IMAGE_SIZE]).astype('float32')
                ...         label = np.random.randint(0, CLASS_NUM - 1, (1, )).astype('int64')
                ...         return image, label
                ...
                ...     def __len__(self):
                ...         return self.num_samples

                >>> paddle.enable_static()

                >>> image = static.data(name='image', shape=[None, 784], dtype='float32')
                >>> label = static.data(name='label', shape=[None, 1], dtype='int64')
                >>> pred = static.nn.fc(x=image, size=10, activation='softmax')
                >>> loss = F.cross_entropy(input=pred, label=label)
                >>> avg_loss = paddle.mean(loss)

                >>> optimizer = paddle.optimizer.SGD(learning_rate=0.001)
                >>> optimizer.minimize(avg_loss)

                >>> place = paddle.CPUPlace()
                >>> exe = static.Executor(place)
                >>> exe.run(static.default_startup_program())

                >>> # create data loader
                >>> dataset = RandomDataset(BATCH_NUM * BATCH_SIZE)
                >>> loader = paddle.io.DataLoader(dataset,
                ...     feed_list=[image, label],
                ...     places=place,
                ...     batch_size=BATCH_SIZE,
                ...     shuffle=True,
                ...     drop_last=True,
                ...     return_list=False,
                ...     num_workers=2
                ... )

                >>> # 1. train and save inference model
                >>> for data in loader():
                ...     exe.run(
                ...         static.default_main_program(),
                ...         feed=data,
                ...         fetch_list=[avg_loss]
                ...     )

                >>> model_path = "fc.example.model"
                >>> paddle.static.save_inference_model(
                ...     model_path,
                ...     [image],
                ...     [pred],
                ...     exe
                ... )

                >>> # 2. load model

                >>> # enable dygraph mode
                >>> paddle.disable_static(place)

                >>> # load
                >>> fc = paddle.jit.load(model_path)

                >>> # inference
                >>> fc.eval()
                >>> x = paddle.randn([1, IMAGE_SIZE], 'float32')
                >>> pred = fc(x)

                >>> # fine-tune
                >>> fc.train()
                >>> loss_fn = nn.CrossEntropyLoss()
                >>> adam = opt.Adam(learning_rate=0.001, parameters=fc.parameters())
                >>> loader = paddle.io.DataLoader(dataset,
                ...     places=place,
                ...     batch_size=BATCH_SIZE,
                ...     shuffle=True,
                ...     drop_last=True,
                ...     num_workers=2
                ... )
                >>> for epoch_id in range(EPOCH_NUM):
                ...     for batch_id, (image, label) in enumerate(loader()):
                ...         out = fc(image)
                ...         loss = loss_fn(out, label)
                ...         loss.backward()
                ...         adam.step()
                ...         adam.clear_grad()
                ...         print("Epoch {} batch {}: loss = {}".format(
                ...             epoch_id, batch_id, np.mean(loss.numpy())))
    """
    # 1. construct correct config
    config = _parse_load_config(configs)
    model_path, config = _build_load_path_and_config(path, config)

    if use_pir_api():
        return PirTranslatedLayer._construct(model_path, config)
    else:
        return TranslatedLayer._construct(model_path, config)


def set_dynamic_shape(variable, shape_list):
    if paddle.base.dygraph.base.in_to_static_mode():
        if isinstance(variable, paddle.base.framework.Variable):
            variable.desc.set_shape(shape_list)
        elif isinstance(variable, paddle.pir.Value):
            variable.set_shape(shape_list)
        else:
            raise TypeError(
                "In to_static mode, variable must be a Variable or Value"
            )
    else:
        # in dygraph mode, dynamic shape is not needed, just do nothing.
        return


def get_ast_static_function(function):
    if isinstance(function, SymbolicStaticFunction):
        if function.class_instance:
            dygraph_function = types.MethodType(
                function._dygraph_function, function.class_instance
            )
        else:
            dygraph_function = function._dygraph_function

        if function._function_spec._input_spec is None:
            ast_static_function = ASTStaticFunction(
                dygraph_function,
                function.last_call_input_spec,
                **function._kwargs,
            )
            return ast_static_function
        else:
            ast_static_function = ASTStaticFunction(
                dygraph_function,
                function._function_spec._input_spec,
                **function._kwargs,
            )
            return ast_static_function
    return function


def json_to_pdmodel(net, input_spec, load_path, save_path):
    net1 = paddle.jit.load(load_path)
    state_dict = {}
    for val in net1.state_dict().values():
        name = val.name[: val.name.rfind('_')]
        state_dict[name] = val

    name_state_dict = {}
    for name, var in net.to_static_state_dict().items():
        name_state_dict[name] = state_dict[var.name]

    net.set_state_dict(name_state_dict)
    with paddle.pir_utils.OldIrGuard():
        paddle.jit.save(net, save_path, input_spec)
