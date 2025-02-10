# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import collections
import inspect
import os
import threading
import warnings
import weakref
from typing import TYPE_CHECKING, Any, Callable, Generic, TypeVar

from typing_extensions import ParamSpec, Self

import paddle
import paddle.pir.core as ir_static
from paddle import decomposition, get_flags
from paddle.base import core, framework
from paddle.base.data_feeder import check_type
from paddle.base.dygraph.base import (
    param_guard,
    switch_to_static_graph,
    to_static_mode_guard,
)
from paddle.framework import in_dynamic_mode, use_pir_api
from paddle.nn.layer import layers
from paddle.pir import Value
from paddle.pir.core import _convert_into_value, static_op_arg_cast_guard
from paddle.utils import flatten, gast

from . import error, logging_utils
from .function_spec import (
    FunctionSpec,
    _hash_spec_names,
    get_buffers,
    get_parameters,
)
from .origin_info import (
    attach_origin_info,
    create_and_update_origin_info_map,
    update_op_callstack_with_origin_info,
)
from .partial_program import PartialProgramLayer, PartialProgramLayerHook
from .pir_partial_program import (
    PartialProgramLayer as PirPartialProgramLayer,
    PartialProgramLayerHook as PirPartialProgramLayerHook,
)
from .transformers import DygraphToStaticAst
from .utils import (
    ALREADY_D2S,
    NO_SHAPE_VAR_TYPE,
    ast_to_func,
    backend_guard,
    cuda_pinned_tensors_move_to_excepted_place,
    func_to_source_code,
    input_specs_compatible,
    is_paddle_func,
    make_hashable,
    prim_or_cinn_is_enabled,
    type_name,
)

if TYPE_CHECKING:
    from paddle._typing import NestedSequence
    from paddle.static import InputSpec, Program
    from paddle.static.amp.fp16_utils import AmpOptions

_RetT = TypeVar("_RetT")
_InputT = ParamSpec("_InputT")

__all__ = []

# For each traced function, we set `max_traced_program_count` = 10 to consider caching performance.
# Once exceeding the threshold, we will raise warning to users to make sure the conversion is as expected.
MAX_TRACED_PROGRAM_COUNT = 10

CONVERSION_OPTIONS = "__jst_not_to_static"


def synchronized(func):
    func.__lock__ = threading.Lock()

    def lock_func(*args, **kwargs):
        with func.__lock__:
            return func(*args, **kwargs)

    return lock_func


def show_op_callstack(op):
    op_callstack = op.callstack
    target_lines = {
        "outputs = static_func(*inputs)",
        "outputs = static_func(*inputs, **_kwargs)",
    }
    op_callstack_message = ""
    for index, line in enumerate(op_callstack):
        if line.strip() in target_lines:
            op_callstack_result = '\n'.join(op_callstack[index + 1 :])
            op_callstack_message = (
                f"In transformed code:\n\n{op_callstack_result}\n\n"
            )
    raise ValueError(
        f"{op_callstack_message}Sorry about what's happened. In to_static mode, {op.name()}'s output variable is a viewed Tensor in dygraph. "
        f"This will result in inconsistent calculation behavior between dynamic and static graphs. "
        f"You must find the location of the strided ops be called, and call paddle.assign() before inplace input. "
        f"If you certainly make sure it's safe, you can set env stride_in_no_check_dy2st_diff to 1."
    )


def check_view_api_used_by_inplace(program: paddle.pir.Program) -> None:
    """
    check viewed value used by inplace op in pir mode.

    Two scenarios will raise ValueError:
        # one
        a = transpose(b)
        a.add_(c)
        # two
        a = transpose(b)
        b.add_(c)
    """
    # TODO(ooooo): Deal with these inplace ops
    skipped_inplace_ops = [
        "pd_op.set_value_",
        "pd_op.set_value_with_tensor_",
        # It willn't change tensor imdeiately,but it's output is dangerous.
        "pd_op.share_data_",
    ]

    def val_is_used_by_stride_op(op, val):
        return op.name() in framework.stride_ops and op.operand_source(
            0
        ).is_same(val)

    def is_used_by_inplace_op(op, val, info):
        return op.name().endswith("_") and any(
            op.operand_source(index).is_same(val) for index in info.values()
        )

    all_vars_list = program.list_vars()
    for value in all_vars_list:
        used_by_stride_ops = []
        for op in reversed(value.all_used_ops()):
            inplace_info = paddle.core.pir.get_op_inplace_info(op)
            if val_is_used_by_stride_op(op, value):
                used_by_stride_ops.append(op)
            if is_used_by_inplace_op(op, value, inplace_info):
                if op.name() in skipped_inplace_ops:
                    continue
                if value.get_defining_op().name() in framework.stride_ops:
                    show_op_callstack(op)
                if len(used_by_stride_ops) == 0:
                    continue
                show_op_callstack(op)


class FunctionCache:
    """
    Caches the transformed functions to avoid redundant conversions of the same function.
    """

    def __init__(self):
        # Caches the converted static functions. {dygraph_func: static_func}
        self._converted_static_func_caches = weakref.WeakKeyDictionary()
        # Caches the converted ast node for same source code. {source_code: ast_root}
        self._code_to_ast_caches = {}
        self._dygraph_to_static = DygraphToStaticAst()

    def convert_with_cache(self, func):
        """
        Returns the cached static function or converts it when first encounters the function.
        """
        # If hit cache, return it directly.
        static_func = self._converted_static_func_caches.get(func, None)

        if static_func is None:
            static_func = self._convert(func)
            self._converted_static_func_caches[func] = static_func

        return static_func

    def _convert(self, func):
        """
        Converts dygraph function into static function. For two functions with same dedent code,
        the second function will reuse the transformed ast node of previous one.

        For example:
            # A.py
            def foo(x, y):
                z = x + y
                return z

            # B.py
            def foo(x, y):
                z = x + y
                return z

        If the conversion of A.foo happens after B.foo, it will reuse the transformed ast node of B.foo
        to speed up the conversion.
        """
        func = inspect.unwrap(func)
        source_code = func_to_source_code(func)

        # TODO(liym27):
        #  Consider this case: source_code in self._code_to_ast_caches,
        #  but actually they are methods in different classes.
        #  Maybe use (__class__, source_code) as key
        if source_code in self._code_to_ast_caches:
            root = self._code_to_ast_caches[source_code]
        else:
            root = gast.parse(source_code)
            root = attach_origin_info(root, func)
            root = self._dygraph_to_static.get_static_ast(root)
            self._code_to_ast_caches[source_code] = root

        # Get static function from AST
        static_func, file_name = ast_to_func(root, func)

        create_and_update_origin_info_map(root, static_func)
        return static_func

    def exist(self, func):
        return func in self._converted_static_func_caches


_CACHE_LOCK = threading.Lock()
_FUNCTION_CACHE = FunctionCache()


def convert_to_static(function):
    """
    Transforms function of dygraph into static function using the cache mechanism.

    Note(dev): It will return function.__func__ if encountering class method.

    Args:
        function(callable): The function with dygraph layers that will be converted into static layers.
    """
    if getattr(function, ALREADY_D2S, None):
        return function

    # Return directly if decorated with @not_to_static and DO NOT Cache it
    options = getattr(function, CONVERSION_OPTIONS, None)
    # or ignore paddle api
    need_skip = (options is not None and options.not_convert) or is_paddle_func(
        function
    )
    if need_skip:
        return function.__func__ if inspect.ismethod(function) else function

    with _CACHE_LOCK:
        static_func = _FUNCTION_CACHE.convert_with_cache(function)
        setattr(static_func, ALREADY_D2S, True)
        return static_func


class CacheKey:
    """
    Cached key for ProgramCache.
    """

    __slots__ = [
        'function_spec',
        'input_args_with_spec',
        'input_kwargs_with_spec',
        'class_instance',
        'kwargs',
        '_spec_names_id',
        '_pir_flags',
    ]

    def __init__(
        self,
        function_spec,
        input_args_with_spec,
        input_kwargs_with_spec,
        class_instance,
        **kwargs,
    ):
        """
        Initializes a cache key.

        Args:
            functions_spec(FunctionSpec): a FunctionSpec instance of decorated function.
            input_args_with_spec(list[InputSpec]): actual input args with some arguments replaced by InputSpec.
            input_kwargs_with_spec(list[{string:InputSpec}]): actual input kwargs with some arguments replaced by InputSpec.
            class_instance(object): a instance of class `Layer`.
            **kwargs(dict): manage other arguments used for better scalability
        """
        self.function_spec = function_spec
        self.input_args_with_spec = input_args_with_spec
        self.input_kwargs_with_spec = input_kwargs_with_spec
        self.class_instance = class_instance
        # NOTE: `kwargs` is usually not considered as basic member for `__hash__`
        self.kwargs = kwargs
        self._spec_names_id = _hash_spec_names(
            input_args_with_spec, input_kwargs_with_spec
        )
        self._pir_flags = (
            get_flags('FLAGS_enable_pir_in_executor')[
                'FLAGS_enable_pir_in_executor'
            ]
            or get_flags('FLAGS_enable_pir_with_pt_in_dy2st')[
                'FLAGS_enable_pir_with_pt_in_dy2st'
            ]
        )

    @classmethod
    def from_func_and_args(cls, function_spec, args, kwargs, class_instance):
        """
        Generated a CacheKey instance by given inputs.

        Args:
            functions_spec(FunctionSpec): a FunctionSpec instance of decorated function.
            args(tuple): tuple of actual inputs arguments.
            kwargs(dict): dict of actual inputs keyword arguments.
            class_instance(object): a instance of class `Layer`.
        """
        # 1. filter `self` in args
        if args and isinstance(args[0], layers.Layer):
            args = args[1:]
        # 2. convert tensor and numpy array into InputSpec
        _args, _kwargs = function_spec.unified_args_and_kwargs(args, kwargs)
        (
            input_args_with_spec,
            input_kwargs_with_spec,
        ) = function_spec.args_to_input_spec(_args, _kwargs)

        # 3. check whether hit the cache or build a new program for the input arguments
        return CacheKey(
            function_spec,
            input_args_with_spec,
            input_kwargs_with_spec,
            class_instance,
        )

    def __hash__(self):
        error_msg = "Arguments to a `@paddle.jit.to_static` must be a hashable Python objects (or nested structures of these types)."
        with_hook = self.kwargs.get("with_hook", False)
        is_train = self.kwargs.get("is_train", False)
        return hash(
            (
                id(self.function_spec),
                make_hashable(self.input_args_with_spec, error_msg),
                make_hashable(self.input_kwargs_with_spec, error_msg),
                self._spec_names_id,
                self.class_instance,
                with_hook,
                is_train,
                self._pir_flags,
                use_pir_api(),
            )
        )

    def __eq__(self, other):
        return (type(self) is type(other)) and hash(self) == hash(other)

    def __neq__(self, other):
        return not self == other

    def __repr__(self):
        return f"id(function_spec): {id(self.function_spec)}, input_args_with_spec: {self.input_args_with_spec}, input_kwargs_with_spec: {self.input_kwargs_with_spec}, class_instance: {self.class_instance}"


def unwrap_decorators(func):
    """
    Unwraps a decorated function and returns the decorator list and inner target.
    """
    decorators = []
    cur = func
    while True:
        if isinstance(cur, StaticFunction):
            decorators.append(cur)
            # Note: if `cur` is a method, keep it as bound method of class.
            instance = cur.class_instance
            if instance is not None:
                cur = cur.dygraph_function.__get__(instance)
            else:
                cur = cur.dygraph_function
        else:
            break
    return decorators, cur


class StaticFunction(Generic[_InputT, _RetT]):
    def __init__(self, function, input_spec=None, **kwargs):
        """
        Initializes a `StaticFunction`.

        Args:
            function(callable): A function or method that will be converted into static program.
            input_spec(list[InputSpec]): list of InputSpec to specify the `shape/dtype/name` information for each input argument, default None.
            **kwargs(dict): other arguments like `build_strategy` et.al.
        """
        # save the instance `self` while decorating a method of class.

        if inspect.ismethod(function):
            self._dygraph_function = function.__func__
            self._class_instance = weakref.ref(function.__self__)

            if not hasattr(self.class_instance, '_original_funcs'):
                raise TypeError(
                    "When using 'to_static' to convert method of a class, "
                    "please ensure the class inherits from nn.Layer"
                )
            self.class_instance._original_funcs[function.__name__] = (
                self._dygraph_function
            )
        else:
            self._dygraph_function = function
            self._class_instance = None

        self._input_spec = input_spec
        self._function_spec = FunctionSpec(function, input_spec)
        self._program_cache = ProgramCache()
        self._descriptor_cache = weakref.WeakKeyDictionary()
        # Note: Hold a reference to ProgramTranslator for switching `enable_to_static`.
        self._program_trans = ProgramTranslator()
        self._kwargs = kwargs
        self._training = True
        self._cuda_graph_capture_mode = ""
        self._cuda_graph_pool_id = 0
        self._property = kwargs.get("property", False)
        # Note: Record the patched method name for rollback.
        self._patched_name = None

    @property
    def is_property(self) -> bool:
        # whether is class proproty to be exported.
        return self._property

    def train(self) -> None:
        if (
            isinstance(self.class_instance, layers.Layer)
            and self.class_instance.training is False
        ):
            raise RuntimeError(
                f"Failed to switch train mode. {self.dygraph_function} is a Layer's method, "
                "please use Layer.train() to switch train mode."
            )
        self._training = True

    def eval(self) -> None:
        if (
            isinstance(self.class_instance, layers.Layer)
            and self.class_instance.training is True
        ):
            raise RuntimeError(
                f"Failed to switch eval mode. {self.dygraph_function} is a Layer's method, "
                "please use Layer.eval() to switch eval mode."
            )
        self._training = False

    def __get__(self, instance, owner):
        """
        Overrides this method to parse the class instance and call bound method correctly.

        For example:

            '''
            class Net(Layer):
                def __init__(self):
                    pass

                @paddle.jit.to_static
                def forward(self, x, y):
                    return x + y

            net = Net()
            out = net(x, y)
            '''

        In above case, `net(x, y)` will call `net.forward(x, y)` firstly that is a bound method
        of `Net` instance. After decorated by `@paddle.jit.to_static`, it will firstly to call `__get__`
        to parse the class instance correctly instead of the `StaticFunction` instance.
        """
        if instance not in self._descriptor_cache:
            if instance is None:
                return self
            # Note(Aurelius84): To construct new instance of StaticFunction when we
            # first encounter the bound function of layer and cache it.
            new_static_layer = self._clone()
            if (
                isinstance(instance, layers.Layer)
                and self._dygraph_function.__name__
                not in instance._original_funcs.keys()
            ):
                instance._original_funcs[self._dygraph_function.__name__] = (
                    self._dygraph_function
                )
            new_static_layer._class_instance = weakref.ref(instance)
            self._descriptor_cache[instance] = new_static_layer

        return self._descriptor_cache[instance]

    def _clone(self) -> Self:
        return self.__class__(
            self.dygraph_function, self._input_spec, **self._kwargs
        )

    def __call__(self, *args: _InputT.args, **kwargs: _InputT.kwargs) -> _RetT:
        """
        Supports to call the returned instance with input `args` and `kwargs` directly.

        Args:
            *args(tuple): tuple of all input arguments from original decorated function.
            **kwargs(dict): dict of all input keyword arguments from original decorated function.

        Return:
            Outputs of decorated function.
        """
        if self._property:
            return self._call_dygraph_function(*args, **kwargs)

        # 1. call dygraph function directly if not enable `declarative`
        if not self._program_trans.enable_to_static:
            # NOTE(liym27):
            # Here calls `warnings.warn` but not `logging_utils.warn` because by default warnings.warn(message)
            # will show up **only once**. StaticFunction.__call__ will run many times, it is appropriate to
            # display this warning message only once.
            logging_utils.warn(
                "The decorator '@paddle.jit.to_static' does NOT work when setting 'paddle.jit.enable_to_static' to False. "
                "We will just return dygraph output. If you would like to get static graph output, please call API "
                "paddle.jit.enable_to_static(True)"
            )
            return self._call_dygraph_function(*args, **kwargs)

        if not in_dynamic_mode():
            raise RuntimeError(
                f"Failed to run the callable object {self.dygraph_function} decorated by '@paddle.jit.to_static', "
                "because it is NOT in dynamic mode. Please disable the static graph mode to enter dynamic mode with the "
                "following API: paddle.disable_static()."
            )

        return self._perform_call(*args, **kwargs)

    def _is_train_mode(self) -> bool:
        if self.class_instance is not None:
            if not hasattr(self.class_instance, 'training'):
                raise TypeError(
                    "When using 'to_static' to convert method of a class, "
                    "please ensure the class inherits from nn.Layer"
                )
            return self.class_instance.training
        else:
            return self._training

    def _call_dygraph_function(
        self, *args: _InputT.args, **kwargs: _InputT.kwargs
    ) -> _RetT:
        """
        Calls dygraph function directly and returns the outputs.

        Args:
            *args(tuple): tuple of all input arguments from original decorated function.
            **kwargs(dict): dict of all input keyword arguments from original decorated function.

        Return:
            Outputs of dygraph function.
        """
        return self.dygraph_function(*args, **kwargs)

    def _raise_when_property(self):
        """raise RuntimeError when property=True

        Raises:
            RuntimeError: can not call this func when property=True
        """
        if self.is_property:
            raise RuntimeError("Can not call the func when property=True.")

    def get_concrete_program(
        self, *args: _InputT.args, **kwargs: _InputT.kwargs
    ) -> tuple[ConcreteProgram, PirPartialProgramLayer]:
        raise NotImplementedError("Not implemented yet.")

    def get_concrete_program_with_cache_key(self, cached_key):
        raise NotImplementedError("Not implemented yet.")

    def get_traced_count(self):
        raise NotImplementedError("Not implemented yet.")

    @property
    def code(self) -> str:
        raise NotImplementedError("Not implemented yet.")

    @property
    def class_instance(self):
        if self._class_instance is None:
            return None
        if self._class_instance() is None:
            raise RuntimeError(
                "The instance of class has been deleted, please re-create the instance."
            )
        return self._class_instance()

    @property
    def dygraph_function(self) -> Callable[_InputT, _RetT]:
        """
        Returns the original decorated function.
        """
        if self.class_instance is not None:
            return self._dygraph_function.__get__(self.class_instance)
        else:
            return self._dygraph_function

    @property
    def concrete_program(self) -> ConcreteProgram:
        raise NotImplementedError("Not implemented yet.")

    def concrete_program_specify_input_spec(
        self,
        input_spec: NestedSequence[InputSpec] | None = None,
        with_hook: bool = False,
        is_prim_infer: bool = False,
    ):
        raise NotImplementedError("Not implemented yet.")

    def rollback(self) -> Callable[_InputT, _RetT]:
        """
        Rollback into original dygraph functions for current class instance.

        Returns:
            Function or Method

        Example::
            .. code-block:: python

                >>> # doctest: +SKIP('`paddle.jit.to_static` can not run in xdoctest')
                >>> import paddle

                >>> class Net(paddle.nn.Layer):
                ...     def __init__(self):
                ...         super().__init__()
                ...
                ...     def forward(self, x, flag=True):
                ...         if flag:
                ...             out = x + 1
                ...         else:
                ...             out = x - 1
                ...         return out
                ...
                >>> x = paddle.randn([10, 1], 'float32')
                >>> net = paddle.jit.to_static(Net())  # convert into static graph mode
                >>> out = net(x)

                >>> net.forward.rollback()  # rollback into dygraph mode
                >>> out = net(x)
        """

        def rollback_impl(class_instance):
            for name, func in class_instance._original_funcs.items():
                setattr(class_instance, name, func.__get__(class_instance))

            for sublayer in class_instance.sublayers(include_self=False):
                rollback_impl(sublayer)

        if self.class_instance is None:
            return self._dygraph_function

        # only rollback sub-functions on path of top _dygraph_function
        fn_name = (
            self._patched_name
            if self._patched_name is not None
            else self._dygraph_function.__name__
        )
        assert (
            fn_name in self.class_instance._original_funcs
        ), f"Not Found function '{fn_name}' in class '{self.class_instance.__class__}'."
        func = self.class_instance._original_funcs[fn_name]
        setattr(self.class_instance, fn_name, func.__get__(self.class_instance))

        for sublayer in self.class_instance.sublayers(include_self=False):
            rollback_impl(sublayer)

        return getattr(self.class_instance, fn_name)

    def __deepcopy__(self, memo):
        """
        Customized behavior for copy.deepcopy, return original decorated function instead
        of a new StaticFunction Object. StaticFunction itself is not copyable because it's
        associated with class_instance.

        We add __deepcopy__ here only for the following usage:

        Example::
            .. code-block:: python

                >>> import copy
                >>> import paddle

                >>> class Net(paddle.nn.Layer):
                ...     def __init__(self):
                ...         super().__init__()
                ...
                ...     def forward(self, x, flag=True):
                ...         if flag:
                ...             out = x + 1
                ...         else:
                ...             out = x - 1
                ...         return out
                ...
                >>> x = paddle.randn([10, 1], 'float32')
                >>> net = paddle.jit.to_static(Net())  # convert into static graph mode

                >>> copy_net = copy.deepcopy(net)      # deepcopy a new net without @to_static

        Please attention that original 'net' will unwrap @to_static and rollback into simple Layer.
        """
        if self.class_instance is not None:
            net_name = type(self.class_instance).__name__
            logging_utils.log(
                level=-1,
                msg=f"Not recommend to deepcopy '{net_name}' decorated with @to_static, it has side effect that will"
                f" rollback into original state before @to_static. Please deepcopy '{net_name}' before applying @to_static.",
            )
            self.rollback()
            return self._dygraph_function.__get__(memo[id(self.class_instance)])
        else:
            return self._dygraph_function

    @property
    def inputs(self) -> list[Any]:
        raise NotImplementedError("Not implemented yet.")

    @property
    def outputs(self) -> list[Any]:
        raise NotImplementedError("Not implemented yet.")

    @property
    def main_program(self) -> Program:
        raise NotImplementedError("Not implemented yet.")

    @property
    def program_cache(self) -> ProgramCache:
        raise NotImplementedError("Not implemented yet.")

    @property
    def function_spec(self) -> FunctionSpec:
        raise NotImplementedError("Not implemented yet.")


def raise_error_template(func_str):
    def _raise_error(*args, **kwargs):
        error_template = (
            "Can't call {func} when full_graph=False. "
            "Use paddle.jit.to_static(full_graph=True) instead."
        )
        raise RuntimeError(error_template.format(func=func_str))

    return _raise_error


class SymbolicStaticFunction(StaticFunction):
    def __init__(self, function, input_spec=None, **kwargs):
        if input_spec is not None:
            warnings.warn(
                "full_graph=False don't support input_spec arguments. It will not produce any effect.\n"
                "You can set full_graph=True, then you can assign input spec.\n"
            )
        super().__init__(function, input_spec, **kwargs)
        self.last_call_input_spec = None

    def _perform_call(self, *args, **kwargs):
        from ..sot import symbolic_translate

        args, kwargs = self._function_spec.unified_args_and_kwargs(args, kwargs)
        cuda_pinned_tensors_move_to_excepted_place(args)

        (
            input_args_with_spec,
            input_kwargs_with_spec,
        ) = self._function_spec.args_to_input_spec(args, kwargs)
        self.last_call_input_spec = input_args_with_spec

        build_strategy = self._kwargs.get("build_strategy", None)
        backend = self._kwargs.get("backend", None)
        traced_fun = symbolic_translate(
            self._dygraph_function,
            build_strategy=build_strategy,
            training=self._is_train_mode(),
            backend=backend,
        )
        if self.class_instance is not None:
            args = (self.class_instance, *args)
        return traced_fun(*args, **kwargs)

    @property
    def code(self):
        raise_error_template("code")()

    @property
    def concrete_program(self):
        raise_error_template("concrete_program")()

    concrete_program_specify_input_spec = raise_error_template(
        "concrete_program_specify_input_spec"
    )
    get_concrete_program = raise_error_template("get_concrete_program")
    get_concrete_program_with_cache_key = raise_error_template(
        "get_concrete_program_with_cache_key"
    )
    get_traced_count = raise_error_template("get_traced_count")

    @property
    def inputs(self):
        raise_error_template("inputs")()

    @property
    def outputs(self):
        raise_error_template("outputs")()

    @property
    def main_program(self):
        raise_error_template("main_program")()

    @property
    def program_cache(self):
        raise_error_template("program_cache")()

    @property
    def function_spec(self):
        raise_error_template("function_spec")()


class ASTStaticFunction(StaticFunction[_InputT, _RetT]):
    """
    Wrapper class to Manage program conversion of decorated function.

    """

    def __init__(self, function, input_spec=None, **kwargs):
        super().__init__(function, input_spec, **kwargs)

    def _perform_call(self, *args, **kwargs):
        # 1. trace ops from dygraph layers and cache the generated program.
        args, kwargs = self._function_spec.unified_args_and_kwargs(args, kwargs)

        try:
            _, partial_program_layer = self.get_concrete_program(
                *args, **kwargs, is_train=self._is_train_mode()
            )
            # 2. synchronize self.training attribute.
            if isinstance(self.class_instance, layers.Layer):
                partial_program_layer.training = self.class_instance.training
            else:
                partial_program_layer.training = self._training

            partial_program_layer._cuda_graph_capture_mode = (
                self._cuda_graph_capture_mode
            )
            partial_program_layer._cuda_graph_pool_id = self._cuda_graph_pool_id

            # 3. return outputs.
            try:
                return partial_program_layer(args)
            except Exception as e:
                if not hasattr(e, error.ERROR_DATA):
                    # runtime error
                    error.attach_error_data(e, in_runtime=True)
                    raise
        except Exception as e:
            error_data = getattr(e, error.ERROR_DATA, None)
            if error_data:
                error_data.raise_new_exception()
            else:
                logging_utils.warn(
                    "Please file an issue at 'https://github.com/PaddlePaddle/Paddle/issues'"
                    f" if you can't handle this {type(e)} yourself."
                )
                raise e

    def get_concrete_program(
        self, *args: _InputT.args, **kwargs: _InputT.kwargs
    ) -> tuple[ConcreteProgram, PirPartialProgramLayer]:
        """
        Returns traced concrete program and inner executable partial layer.

        Args:
            *args(tuple): input arguments values or InputSpec
            **kwargs(dict) : input kwargs values.

        Returns:
            Traced ConcreteProgram and executable translated Layer.
        """
        self._raise_when_property()

        with_hook = kwargs.get("with_hook", False)
        is_train = kwargs.get("is_train", True)
        is_prim_infer = kwargs.get("is_prim_infer", False)
        if "is_train" in kwargs:
            kwargs.pop("is_train")
        if "with_hook" in kwargs:
            kwargs.pop("with_hook")
        if "is_prim_infer" in kwargs:
            kwargs.pop("is_prim_infer")
        # 1. unify args/kwargs and replace Tensor with InputSpec
        if len(args) != len(self._function_spec.args_name):
            args, kwargs = self._function_spec.unified_args_and_kwargs(
                args, kwargs
            )
        (
            input_args_with_spec,
            input_kwargs_with_spec,
        ) = self._function_spec.args_to_input_spec(args, kwargs)

        # 2. generate cache key
        cache_key = CacheKey(
            self._function_spec,
            input_args_with_spec,
            input_kwargs_with_spec,
            self.class_instance,
            **self._kwargs,
            with_hook=with_hook,
            is_train=is_train,
        )
        if is_prim_infer:
            (
                concrete_program,
                partial_program_layer,
            ) = self._program_cache.get_program_without_cache(cache_key)
        else:
            # 3. check whether hit the cache or build a new program for the input arguments
            concrete_program, partial_program_layer = self._program_cache[
                cache_key
            ]
        return concrete_program, partial_program_layer

    def get_concrete_program_with_cache_key(
        self, cached_key: CacheKey
    ) -> tuple[ConcreteProgram, PartialProgramLayer | PirPartialProgramLayer]:
        """
        Returns traced concrete program and inner executable partial layer by cached key.

        Args:
            cached_key(CacheKey): The cached key use to get concrete program.

        Returns:
            Traced ConcreteProgram and executable translated Layer.
        """
        self._raise_when_property()
        (
            concrete_program,
            partial_program_layer,
        ) = self._program_cache.get_program_without_cache(cached_key)
        return concrete_program, partial_program_layer

    def get_traced_count(self) -> int:
        """
        Returns the number of traced programs for the decorated function.
        """
        return len(self._program_cache)

    @property
    def code(self) -> str:
        """
        Returns the source code of transformed static function for debugging.
        """
        static_func = convert_to_static(self.dygraph_function)
        source_code = func_to_source_code(static_func)
        return source_code

    @property
    def concrete_program(self) -> ConcreteProgram:
        """
        Returns recent ConcreteProgram instance of decorated function.

        Examples:
            .. code-block:: python

                >>> # doctest: +SKIP('`paddle.jit.to_static` can not run in xdoctest')
                >>> import paddle
                >>> from paddle.jit import to_static
                >>> from paddle.static import InputSpec

                >>> paddle.disable_static()

                >>> def foo(x, y):
                ...     z = x + y
                ...     return z
                ...
                >>> # usage 1:
                >>> decorated_foo = to_static(foo, input_spec=[InputSpec([10], name='x'), InputSpec([10], name='y')])
                >>> print(decorated_foo.concrete_program)

                >>> # usage 2:
                >>> decorated_foo = to_static(foo)
                >>> out_foo = decorated_foo(paddle.rand([10]), paddle.rand([10]))
                >>> print(decorated_foo.concrete_program)
        """
        return self.concrete_program_specify_input_spec(input_spec=None)

    def concrete_program_specify_input_spec(
        self,
        input_spec: NestedSequence[InputSpec] | None = None,
        with_hook: bool = False,
        is_prim_infer: bool = False,
    ) -> ConcreteProgram:
        """
        Returns recent ConcreteProgram instance of decorated function while
        specifying input_spec. If the self._function_spec already has
        input_spec, it will check the compatibility of input input_spec and
        the self._function_spec.input_spec. If input input_spec=None, then
        this method uses self._function_spec.input_spec

        args:
            input_spec (list[InputSpec], optional): Describes the input of
                the translate function.
        """
        self._raise_when_property()
        # if specific the `input_spec`, the length of program_cache will always 1,
        # else, return the last one.
        cached_program_len = len(self._program_cache)
        # If specific `input_spec`, apply conversion from dygraph layers into static Program.
        # NOTE(jiabin): is_prim_infer indicates this method called by paddle.jit.save and it is worked in prim mode

        desired_input_spec = input_spec
        if self._function_spec.input_spec is not None:
            if input_spec is not None and not input_specs_compatible(
                flatten(input_spec), flatten(self._function_spec.input_spec)
            ):
                raise ValueError(
                    f"The `input_spec`: {input_spec} used to construct concrete_program is conflict with the `input_spec`: {self._function_spec.input_spec} in `@paddle.jit.to_static`"
                )
            # NOTE(chenweihang): we should always translated program based on the `input_spec`
            # decorated on forward if it is valid
            desired_input_spec = self._function_spec.input_spec
            if input_spec is not None:
                logging_utils.warn(
                    f"\n\nYou have specified `input_spec` both in function definition (higher priority) and `paddle.jit.save` (will be ignored.)\n\n\t Using: {desired_input_spec}\n\n\t Ignore: {input_spec}\n"
                )

        has_input_spec = desired_input_spec is not None
        if has_input_spec:
            concrete_program, _ = self.get_concrete_program(
                *desired_input_spec,
                with_hook=with_hook,
                is_train=self._is_train_mode(),
                is_prim_infer=is_prim_infer,
            )
            return concrete_program
        else:
            if cached_program_len != 0:
                logging_utils.warn(
                    "No input_spec is found, save cached program instead"
                )
                if cached_program_len > 1:
                    logging_utils.warn(
                        f"Current {self._function_spec} has more than one cached programs: {cached_program_len}, the last traced program will be return by default."
                    )

                cache_key = self._program_cache._recent_cache_key

                if with_hook:
                    cache_key.kwargs["with_hook"] = True

                if is_prim_infer:
                    (
                        concrete_program,
                        _,
                    ) = self.get_concrete_program_with_cache_key(cache_key)
                    return concrete_program
                else:
                    concrete_program, _ = self._program_cache[cache_key]
                    return concrete_program

            else:
                raise ValueError(
                    f"No valid transformed program for {self._function_spec}.\n\t    Please specific `input_spec` in `@paddle.jit.to_static` or feed input tensor to call the decorated function at once.\n"
                )

    @property
    def inputs(self) -> list[Any]:
        """
        Returns input tensors of recent converted static program.
        """
        self._raise_when_property()
        concrete_program = self.concrete_program
        inputs = [
            var
            for var in flatten(concrete_program.inputs)
            if isinstance(var, (framework.Variable, Value))
        ]
        return inputs

    @property
    def outputs(self) -> list[Any]:
        """
        Returns output tensors of recent converted static program.
        """
        self._raise_when_property()
        concrete_program = self.concrete_program
        outputs = [
            var
            for var in flatten(concrete_program.outputs)
            if isinstance(var, (framework.Variable, Value))
        ]

        return outputs

    @property
    def main_program(self) -> Program:
        """
        Returns recent converted static main program.
        """
        self._raise_when_property()
        concrete_program = self.concrete_program
        main_program = concrete_program.main_program
        return main_program

    @property
    def program_cache(self) -> ProgramCache:
        return self._program_cache

    @property
    def function_spec(self) -> FunctionSpec:
        return self._function_spec


def _verify_init_in_dynamic_mode(class_instance):
    """
    Verifies the instance is initialized in dynamic mode.
    """
    if isinstance(class_instance, layers.Layer):
        if not class_instance._init_in_dynamic_mode:
            raise RuntimeError(
                " `paddle.jit.to_static` is only available in dynamic mode. Please call `paddle.disable_static()` before "
                f"initializing your Layer class `{class_instance}` . Because parameters of Layer class should be initialized firstly "
                "in dynamic mode while applying transformation."
            )


class HookHelper:
    """
    Only For converting pre/post hooks operation in outermost layer while jit.save.
    Because hooks in sublayer have been processed automatically.
    """

    def __init__(self, func, class_instance, with_hook=False):
        self.func = func
        self.class_instance = class_instance
        self.with_hook = with_hook
        self.need_apply_hook = (
            with_hook
            and isinstance(self.class_instance, layers.Layer)
            and func.__name__ == "forward"
        )

    def apply_pre_hooks(self, inputs):
        """
        Apply _forward_pre_hooks from outermost layer
        """
        if not self.need_apply_hook:
            return inputs

        inputs = inputs[1:]
        for forward_pre_hook in self.class_instance._forward_pre_hooks.values():
            hook_result = forward_pre_hook(self.class_instance, inputs)
            if hook_result is not None:
                if not isinstance(hook_result, tuple):
                    hook_result = (hook_result,)
                inputs = hook_result

        return [self.class_instance, *list(inputs)]

    def apply_post_hooks(self, inputs, outputs):
        """
        Apply _forward_post_hooks from outermost layer
        """
        if not self.need_apply_hook:
            return outputs

        inputs = inputs[1:]
        for (
            forward_post_hook
        ) in self.class_instance._forward_post_hooks.values():
            hook_result = forward_post_hook(
                self.class_instance, inputs, outputs
            )
            if hook_result is not None:
                outputs = hook_result

        inputs.insert(0, self.class_instance)
        return outputs


class ConcreteProgram:
    __slots__ = [
        'inputs',
        'outputs',
        'main_program',
        "startup_program",
        "parameters",
        "function",
        'kwargs',
    ]

    def __init__(
        self,
        inputs,
        outputs,
        parameters,
        function,
        main_program,
        startup_program=None,
        **kwargs,
    ):
        self.inputs = inputs
        self.outputs = outputs
        self.main_program = main_program
        self.startup_program = startup_program
        self.parameters = parameters
        self.function = function
        self.kwargs = kwargs

    @staticmethod
    @switch_to_static_graph
    def pir_from_func_spec(
        func_spec, input_spec, input_kwargs_spec, class_instance, **kwargs
    ):
        """
        Builds the main_program with specialized inputs and returns outputs
        of program as fetch_list.

        Args:
            func_spec(FunctionSpec): A FunctionSpec instance for decorated function.
            input_spec(list[InputSpec]):
        """
        # verify the instance is initialized in imperative mode.
        _verify_init_in_dynamic_mode(class_instance)

        # Transforms dygraph function into static function and caches it.
        dygraph_function = func_spec.dygraph_function
        static_func = convert_to_static(dygraph_function)
        # apply pre\post hook for outermost layer
        hook_helper = HookHelper(
            dygraph_function, class_instance, kwargs.get("with_hook", False)
        )

        main_program, startup_program = ir_static.Program(), ir_static.Program()
        # Note: The random seed should be synchronized into cached program
        # if set in `fluid.dygraph_guard` because some ops rely on it, such as
        # `fluid.layers.dropout`.
        main_program.random_seed = (
            paddle.static.default_main_program().random_seed
        )
        startup_program.random_seed = (
            paddle.static.default_startup_program().random_seed
        )

        with ir_static.program_guard(main_program, startup_program):
            with to_static_mode_guard(
                is_to_static=True
            ), static_op_arg_cast_guard(_convert_into_value):
                # 1. Adds `paddle.static.data` layers for input if needed
                static_inputs, program_inputs = (
                    func_spec.pir_to_static_inputs_with_spec(
                        input_spec, main_program
                    )
                )
                _kwargs, _ = func_spec.pir_to_static_inputs_with_spec(
                    input_kwargs_spec, main_program
                )
                if class_instance:
                    static_inputs = (
                        class_instance,
                        *list(static_inputs),
                    )
                    program_inputs = (
                        class_instance,
                        *list(program_inputs),
                    )

                # 2. Builds program only once and returns the output Variables.
                with param_guard(
                    get_parameters(class_instance, True)
                ), param_guard(get_buffers(class_instance, True)):
                    try:
                        # only for jit.save, do nothing while train and eval process
                        inputs = hook_helper.apply_pre_hooks(static_inputs)
                        if _kwargs:
                            outputs = static_func(*inputs, **_kwargs)
                        else:
                            outputs = static_func(*inputs)
                        outputs = hook_helper.apply_post_hooks(inputs, outputs)
                    except BaseException as e:
                        # NOTE: If e is raised in compile time, e should be attached to ERROR_DATA here.
                        error.attach_error_data(e)
                        error_data = getattr(e, error.ERROR_DATA, None)
                        if error_data:
                            error_data.raise_new_exception()
                        raise

                # 3. Gets all ParamBases and buffered VarBases in the function
                from ..pir_dy2static.parameter_recorder import (
                    _global_parameter_recorder,
                )

                all_parameters_and_buffers = _global_parameter_recorder.pop(
                    main_program
                )
                if outputs is not None:
                    need_wrap_into_list = (
                        not isinstance(outputs, (tuple, list))
                        or len(outputs) == 1
                    )
                    if need_wrap_into_list:
                        outputs = [outputs]

        main_program = update_op_callstack_with_origin_info(main_program)
        if not os.environ.get("stride_in_no_check_dy2st_diff", "0") == "1":
            check_view_api_used_by_inplace(main_program)

        return ConcreteProgram(
            inputs=program_inputs,
            outputs=outputs,
            parameters=all_parameters_and_buffers,
            function=dygraph_function,
            main_program=main_program,
            startup_program=startup_program,
            **kwargs,
        )

    # TODO(@xiongkun): remove after new ir is switch
    @staticmethod
    @switch_to_static_graph
    def from_func_spec(
        func_spec, input_spec, input_kwargs_spec, class_instance, **kwargs
    ):
        """
        Builds the main_program with specialized inputs and returns outputs
        of program as fetch_list.

        Args:
            func_spec(FunctionSpec): A FunctionSpec instance for decorated function.
            input_spec(list[InputSpec]):
        """
        # verify the instance is initialized in imperative mode.
        _verify_init_in_dynamic_mode(class_instance)

        # Transforms dygraph function into static function and caches it.
        dygraph_function = func_spec.dygraph_function
        static_func = convert_to_static(dygraph_function)
        # apply pre\post hook for outermost layer
        hook_helper = HookHelper(
            dygraph_function, class_instance, kwargs.get("with_hook", False)
        )

        main_program, startup_program = framework.Program(), framework.Program()
        # Note: The random seed should be synchronized into cached program
        # if set in `base.dygraph_guard` because some ops rely on it, such as
        # `base.layers.dropout`.
        main_program.random_seed = (
            paddle.static.default_main_program().random_seed
        )
        startup_program.random_seed = (
            paddle.static.default_startup_program().random_seed
        )

        ProgramTranslator.get_instance()._amp_records.clear()

        with framework.program_guard(main_program, startup_program):
            with to_static_mode_guard(is_to_static=True):
                # 1. Adds `paddle.static.data` layers for input if needed
                static_inputs = func_spec.to_static_inputs_with_spec(
                    input_spec, main_program
                )
                _kwargs = func_spec.to_static_inputs_with_spec(
                    input_kwargs_spec, main_program
                )
                if class_instance:
                    static_inputs = (
                        class_instance,
                        *list(static_inputs),
                    )

                # 2. Builds program only once and returns the output Variables.
                with param_guard(
                    get_parameters(class_instance, True)
                ), param_guard(get_buffers(class_instance, True)):
                    try:
                        # only for jit.save, do nothing while train and eval process
                        inputs = hook_helper.apply_pre_hooks(static_inputs)
                        if _kwargs:
                            outputs = static_func(*inputs, **_kwargs)
                        else:
                            outputs = static_func(*inputs)
                        outputs = hook_helper.apply_post_hooks(inputs, outputs)
                    except BaseException as e:
                        # NOTE: If e is raised in compile time, e should be attached to ERROR_DATA here.
                        error.attach_error_data(e)
                        error_data = getattr(e, error.ERROR_DATA, None)
                        if error_data:
                            error_data.raise_new_exception()
                        raise

                # 3. Gets all ParamBases and buffered VarBases in the function
                all_parameters_and_buffers = (
                    ProgramTranslator.get_instance()._params_recorder.pop(
                        main_program
                    )
                )

                if outputs is not None:
                    need_wrap_into_list = (
                        not isinstance(outputs, (tuple, list))
                        or len(outputs) == 1
                    )
                    if need_wrap_into_list:
                        outputs = [outputs]

        main_program = update_op_callstack_with_origin_info(main_program)

        return ConcreteProgram(
            inputs=static_inputs,
            outputs=outputs,
            parameters=all_parameters_and_buffers,
            function=dygraph_function,
            main_program=main_program,
            startup_program=startup_program,
            **kwargs,
        )


def _program_hash(program):
    """
    because program is not deleted while calling from_func_spec.
    so it's ok to use id(program)
    """
    return id(program)


class ParametersRecorder:
    def __init__(self):
        self.params_dict = {}

    @synchronized
    def add(self, program, param):
        """use the default_program as key, append param the parameter list."""
        key = _program_hash(program)
        if key not in self.params_dict:
            self.params_dict[key] = set()
        params = self.params_dict[key]
        params.add(param)

    def pop(self, program):
        params = self.params_dict.get(_program_hash(program))
        if params is None:
            return []
        del self.params_dict[_program_hash(program)]
        params = list(params)
        params.sort(key=lambda x: x.name)
        return params


class InplaceMap:
    def __init__(self):
        self.params_dict = {}

    @synchronized
    def add(self, program, id, param):
        """use the default_program as key, append param the parameter list."""
        key = _program_hash(program)
        if key not in self.params_dict:
            self.params_dict[key] = {}

        params = self.params_dict[key]
        params[id] = param

    def get(self, program, id):
        params = self.params_dict.get(_program_hash(program))
        if params is None:
            return None
        if id not in params:
            return None
        root_var = params[id]
        saved = []
        while root_var.desc.id() in params.keys():
            saved.append(root_var)
            root_var = params[root_var.desc.id()]
        for var in saved:
            params[var.desc.id()] = root_var
        return root_var

    def restore_checkpoint(self, checkpoint):
        # InplaceMap is a nested effect.
        # when enter a block, we should save a checkpoint
        # when exit a block, we should restore a checkpoint
        # for example:
        # if cond > 0:
        #    x [:] = 0
        # return x
        # x[:] only effect current cond block, we should restore in false block.
        self.params_dict = checkpoint

    def save_checkpoint(self):
        ckp = {}
        for program_id, params in self.params_dict.items():
            new_params = dict(params.items())
            ckp[program_id] = new_params
        return ckp


class PirPrimHooker(PirPartialProgramLayerHook):
    def __init__(self, original_program, backend):
        self.backend = backend
        self.custom_vjps = set()
        with backend_guard(self.backend):
            if core._is_all_prim_enabled():
                self.custom_vjps = {
                    op.name()
                    for op in original_program.global_block().ops
                    if core.has_custom_vjp(op)
                }

    def before_append_backward(self, forward_program, src_vars):
        with backend_guard(self.backend):
            if core._is_fwd_prim_enabled():
                dst_vars = decomposition.decompose(
                    forward_program, src_vars, blacklist=self.custom_vjps
                )
                return forward_program, dst_vars
            return forward_program, src_vars

    def after_append_backward(
        self,
        whole_program,
        inputs,
        src_vars,
        grad_outputs,
        forward_end_idx,
        backward_start_idx,
    ):
        with backend_guard(self.backend):
            if core._is_fwd_prim_enabled() and len(self.custom_vjps) != 0:
                backward_length = (
                    len(whole_program.global_block().ops) - forward_end_idx
                )
                # decompose forward program
                dst_vars = decomposition.decompose(
                    whole_program,
                    src_vars,
                    whitelist=self.custom_vjps,
                    end_index=forward_end_idx,
                )
                new_start_index = (
                    len(whole_program.global_block().ops) - backward_length
                )
                # decompose backward program
                dst_vars = decomposition.decompose(
                    whole_program,
                    dst_vars,
                    whitelist=self.custom_vjps,
                    start_index=new_start_index,
                )
                return whole_program, new_start_index, dst_vars
            return whole_program, forward_end_idx, src_vars

    def after_infer(self, infer_program):
        with backend_guard(self.backend):
            if core._is_fwd_prim_enabled():
                targets = decomposition.decompose(
                    infer_program.program, infer_program.out_values
                )
                infer_program.out_values = targets
                infer_program.forward_range = (
                    0,
                    len(infer_program.program.global_block().ops),
                )
            return


class PirAutoRecomputeHooker(PirPartialProgramLayerHook):
    def __init__(self, recompute_ops=None):
        self.recompute_ops = recompute_ops

    def before_append_backward(self, forward_program, src_vars):
        return forward_program, src_vars

    def after_append_backward(
        self,
        whole_program,
        inputs,
        src_vars,
        grad_outputs,
        forward_end_idx,
        backward_start_idx,
    ):
        if core._enable_auto_recompute():
            whole_program, forward_end_idx = decomposition.auto_recompute(
                whole_program,
                inputs,
                src_vars,
                grad_outputs,
                forward_end_idx,
                backward_start_idx,
            )
        return whole_program, forward_end_idx, src_vars


class ProgramCache:
    """
    Wrapper class for the program functions defined by dygraph function.
    """

    def __init__(self):
        # {hash_id : (concrete_program, partial_layer)}
        self._caches = collections.OrderedDict()
        # trace mostly recent used program
        self._recent_key = None
        self._recent_cache_key = None

    def _build_once(self, cache_key):
        # TODO(Aurelius84): Need a global FLAGS to enable/disable to_prim
        enable_prim = cache_key.kwargs['build_strategy'].build_cinn_pass

        if use_pir_api():
            concrete_program = ConcreteProgram.pir_from_func_spec(
                func_spec=cache_key.function_spec,
                input_spec=cache_key.input_args_with_spec,
                input_kwargs_spec=cache_key.input_kwargs_with_spec,
                class_instance=cache_key.class_instance,
                **cache_key.kwargs,
            )
        else:
            concrete_program = ConcreteProgram.from_func_spec(
                func_spec=cache_key.function_spec,
                input_spec=cache_key.input_args_with_spec,
                input_kwargs_spec=cache_key.input_kwargs_with_spec,
                class_instance=cache_key.class_instance,
                **cache_key.kwargs,
            )

        backend = cache_key.kwargs['backend']
        if not use_pir_api():
            # decrease prim_is_enable() call to decrease print log
            if prim_or_cinn_is_enabled(
                cache_key.kwargs['build_strategy'], backend
            ):
                for var in concrete_program.main_program.list_vars():
                    if var.type not in NO_SHAPE_VAR_TYPE and -1 in var.shape:
                        warnings.warn(
                            f"Now prim and cinn do not support -1 shape, but the shape of var {var.name} is {var.shape}"
                        )

        if use_pir_api():
            from .pir_partial_program import partial_program_from

            partial_program = partial_program_from(
                concrete_program, cache_key.class_instance is not None
            )
        else:  # TODO(pir): remove later.
            from .partial_program import partial_program_from

            partial_program = partial_program_from(
                concrete_program, cache_key.class_instance is not None
            )
        with backend_guard(backend):
            if core._is_fwd_prim_enabled():
                if use_pir_api():
                    partial_program.add_hooker(
                        PirPrimHooker(concrete_program.main_program, backend)
                    )
                else:
                    partial_program.set_hooker(
                        PrimHooker(concrete_program.main_program, backend)
                    )
        if use_pir_api() and core._enable_auto_recompute():
            partial_program.add_hooker(PirAutoRecomputeHooker())
        return concrete_program, partial_program

    def __getitem__(self, item):
        if not isinstance(item, CacheKey):
            raise ValueError(
                f'type(item) should be CacheKey, but received {type_name(item)}'
            )
        item_id = hash(item)
        self._recent_cache_key = item
        self._recent_key = item_id
        if item_id not in self._caches:
            self._caches[item_id] = self._build_once(item)
            # Note: raise warnings if number of traced program is more than `max_tracing_count`
            current_tracing_count = len(self._caches)
            if current_tracing_count > MAX_TRACED_PROGRAM_COUNT:
                logging_utils.warn(
                    f"Current traced program number: {current_tracing_count} > `max_tracing_count`:{MAX_TRACED_PROGRAM_COUNT}. Too much cached programs will bring expensive overhead. "
                    "The reason may be: (1) passing tensors with different shapes, (2) passing python objects instead of tensors."
                )

        return self._caches[item_id]

    def get_program_without_cache(self, cache_key):
        return self._build_once(cache_key=cache_key)

    def get_program(self, item):
        if not isinstance(item, CacheKey):
            raise ValueError(
                f"Input item's type should be FunctionSpec, but received {type_name(item)}"
            )
        item_id = hash(item)
        if item_id not in self._caches:
            raise RuntimeError(
                "Failed to find program for input item, please decorate input function by `@paddle.jit.to_static`."
            )
        return self._caches[item_id]

    def last(self):
        assert (
            len(self._caches) >= 1
        ), "No valid cached program in ProgramCache."
        assert self._recent_key is not None
        return self._recent_key, self._caches[self._recent_key]

    def __len__(self):
        return len(self._caches)

    def concrete_programs(self):
        return [cp for key, (cp, _) in self._caches.items()]

    def clear(self):
        self._caches = collections.OrderedDict()


class PrimHooker(PartialProgramLayerHook):
    def __init__(self, original_program, backend):
        self.backend = backend
        self.custom_vjps = set()
        with backend_guard(self.backend):
            if core._is_all_prim_enabled():
                self.custom_vjps = {
                    op.type
                    for op in original_program.block(0).ops
                    if core.has_comp_grad_op_maker(op.type)
                }

    def before_append_backward(self, forward_program):
        with backend_guard(self.backend):
            if core._is_fwd_prim_enabled():
                _to_prim(forward_program.blocks, blacklist=self.custom_vjps)
            return forward_program

    def after_append_backward(self, whole_program, backward_start_idx):
        with backend_guard(self.backend):
            backward_length = (
                len(whole_program.block(0).ops) - backward_start_idx
            )
            if core._is_fwd_prim_enabled() and len(self.custom_vjps) != 0:
                # only process backward part of block
                _to_prim(whole_program.blocks, backward_length=backward_length)
            new_start_index = len(whole_program.block(0).ops) - backward_length
            if backward_length > 0:
                # only process forward part of block
                _to_prim(whole_program.blocks, start_idx=new_start_index)
            return whole_program, new_start_index

    def after_infer(self, infer_program):
        with backend_guard(self.backend):
            if core._is_fwd_prim_enabled():
                _to_prim(infer_program.block(0))
            return infer_program


class ProgramTranslator:
    """
    Class to translate dygraph function into static graph function. The object
    of this class is a singleton.

    Args:
        None.

    Returns:
        ProgramTranslator: the singleton object.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> # Two methods get same object because ProgramTranslator is a singleton
            >>> paddle.jit.dy2static.program_translator.ProgramTranslator()
            >>> paddle.jit.dy2static.program_translator.ProgramTranslator.get_instance()

    """

    _singleton_lock = threading.Lock()
    _instance = None

    @synchronized
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = object.__new__(cls, *args, **kwargs)
            cls._instance._initialized = False
        return cls._instance

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._singleton_lock:
                cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls):
        if cls._instance is not None:
            cls._instance._initialized = False
            cls._instance.__init__()

    def __init__(self):
        # To make sure that calls __init__ only once.
        if self._initialized:
            return
        self._initialized = True
        self._program_cache = ProgramCache()
        self._params_recorder = ParametersRecorder()
        self._inplace_map = InplaceMap()
        self._amp_records: dict[int, list[tuple[AmpOptions, int, int]]] = {}
        self.enable_to_static = True

    def enable(self, enable_to_static):
        check_type(
            enable_to_static,
            "enable_to_static",
            bool,
            "ProgramTranslator.enable",
        )
        self.enable_to_static = enable_to_static


def enable_to_static(enable_to_static_bool: bool) -> None:
    """
    Enable or disable the converting from imperative to static graph by
    ProgramTranslator globally.

    Args:
        enable_to_static_bool (bool): True or False to enable or disable converting to static.

    Returns:
        None.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> @paddle.jit.to_static
            >>> def func(x):
            ...     if paddle.mean(x) > 0:
            ...         x_v = x - 1
            ...     else:
            ...         x_v = x + 1
            ...     return x_v
            ...
            >>> paddle.jit.enable_to_static(False)

            >>> x = paddle.ones([1, 2])
            >>> # ProgramTranslator is disabled so the func is run in dygraph
            >>> print(func(x))
            Tensor(shape=[1, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[0., 0.]])

    """
    check_type(
        enable_to_static_bool,
        "enable_to_static_bool",
        bool,
        "paddle.jit.enable_to_static",
    )
    _program_trans = ProgramTranslator()
    _program_trans.enable(enable_to_static_bool)


@switch_to_static_graph
def _to_prim(
    blocks,
    blacklist=frozenset(),
    whitelist=frozenset(),
    start_idx=-1,
    backward_length=-1,
):
    """Switch to static graph and call to_prim."""
    # TODO(Aurelius84): Fix this cycle import problem
    from paddle.incubate.autograd import primapi

    primapi.to_prim(
        blocks,
        blacklist=blacklist,
        whitelist=whitelist,
        start_idx=start_idx,
        backward_length=backward_length,
    )
