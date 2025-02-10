#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import copy
import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ContextManager,
    List,
    Literal,
    Protocol,
    Set,
    Tuple,
    TypeVar,
    Union,
    overload,
)

import paddle
from paddle.base import core
from paddle.base.framework import (
    _current_expected_place,
    _dygraph_tracer,
    dygraph_only,
    in_dynamic_or_pir_mode,
    in_pir_mode,
)
from paddle.base.wrapped_decorator import signature_safe_contextmanager
from paddle.static.amp.decorator import OptimizerWithMixedPrecision

from .amp_lists import black_list, white_list

if TYPE_CHECKING:
    from typing import Generator

    from typing_extensions import TypeAlias, TypeGuard

    from paddle import Tensor
    from paddle._typing.dtype_like import _DTypeLiteral
    from paddle.nn import Layer
    from paddle.nn.layer.layers import _StateDict
    from paddle.static import Operator, Program

    _AmpLevelLiteral = Literal["O0", "OD", "O1", "O2"]
    _CustomList: TypeAlias = Union[List[str], Tuple[str, ...], Set[str]]

    class _OptimizerLike(Protocol):
        def minimize(
            self,
            loss: Tensor,
            startup_program: Program,
            parameters: list[Tensor],
            no_grad_set: set[Tensor],
        ) -> tuple[list[Operator], list[tuple[Tensor, Tensor]]]: ...

        def step(self) -> None: ...

        def set_state_dict(self, state_dict: dict[str, Tensor]) -> None: ...

        def clear_grad(self, set_to_zero: bool) -> None: ...


_ModelsT = TypeVar("_ModelsT", "Layer", List["Layer"])
_OptimizersT = TypeVar("_OptimizersT", "_OptimizerLike", List["_OptimizerLike"])


AMP_RELATED_FLAGS = [
    'FLAGS_cudnn_exhaustive_search',
    'FLAGS_conv_workspace_size_limit',
    'FLAGS_cudnn_batchnorm_spatial_persistent',
]

AMP_RELATED_FLAGS_SETTING = {
    'FLAGS_cudnn_exhaustive_search': 1,
    'FLAGS_conv_workspace_size_limit': 1000,
    'FLAGS_cudnn_batchnorm_spatial_persistent': 1,
}

AMP_LEVEL = core.AmpLevel
_g_amp_state_ = None


def amp_state():
    global _g_amp_state_
    return _g_amp_state_


class AMPGlobalState:
    model_parameters: list[Tensor]
    use_master_grad: bool
    already_register_final_backward_hook: bool
    already_classify_params_meshes: bool
    mesh2params: dict[paddle.distributed.ProcessMesh | None, list[Tensor]]
    amp_dtype: _DTypeLiteral

    def __init__(self) -> None:
        self.model_parameters = []
        self.use_master_grad = False
        self.already_register_final_backward_hook = False
        self.already_classify_params_meshes = False  # For dist
        self.mesh2params = {}  # For dist
        self.amp_dtype = 'float32'

    def __setattr__(self, name: str, val: Any) -> None:
        self.__dict__[name] = val


_amp_global_state = AMPGlobalState()


def amp_global_state() -> AMPGlobalState:
    return _amp_global_state


# NOTE(zhiqiu): similar as paddle.static.amp.fp16_lists.AutoMixedPrecisionLists._update_list
# The reason why not use AutoMixedPrecisionLists is that custom_black_varnames is not suitable for imperative mode.
def _update_list(
    custom_white_list: _CustomList,
    custom_black_list: _CustomList,
    level: _AmpLevelLiteral = 'O1',
    dtype: _DTypeLiteral = 'float16',
) -> tuple[set[str], set[str]]:
    """
    Update black and white list according to users' custom list.
    """
    if level == 'O0':
        _white_list = set()
        _black_list = set()
        return _white_list, _black_list
    _white_list = copy.copy(white_list()[dtype][level])
    _black_list = copy.copy(black_list()[dtype][level])
    if custom_white_list and custom_black_list:
        for op_name in custom_white_list:
            if op_name in custom_black_list:
                raise ValueError(
                    "Custom white list overlap " "custom black list"
                )
    if custom_white_list:
        for op_name in custom_white_list:
            if op_name in _black_list:
                _black_list.remove(op_name)
            _white_list.add(op_name)
    if custom_black_list:
        for op_name in custom_black_list:
            if op_name in _white_list:
                _white_list.remove(op_name)
            _black_list.add(op_name)
    return _white_list, _black_list


def _in_amp_guard() -> bool:
    """
    Judge whether current code block is in `amp_guard` context.
    """
    tracer = _dygraph_tracer()
    if tracer:
        if tracer._amp_level == core.AmpLevel.O1:
            return True
        else:
            return False
    else:
        return False


def _in_pure_fp16_guard() -> bool:
    tracer = _dygraph_tracer()
    return tracer and tracer._amp_level == core.AmpLevel.O2


def _is_gpu_float16_supported() -> bool:
    """
    Judge whether current gpu support float16 amp.
    """
    prop = paddle.device.cuda.get_device_capability()
    return prop[0] >= 7 or paddle.is_compiled_with_rocm()


def _is_gpu_bfloat16_supported() -> bool:
    """
    Judge whether current gpu support bfloat16 amp.
    """
    prop = paddle.device.cuda.get_device_capability()
    cuda_version = paddle.version.cuda()
    if cuda_version is not None and cuda_version != 'False':
        cuda_version_check = int(cuda_version.split('.')[0]) >= 11
    else:
        cuda_version_check = False
    return prop[0] >= 8 and cuda_version_check or paddle.is_compiled_with_rocm()


def _is_xpu_float16_supported() -> bool:
    """
    Judge whether current xpu device support float16 amp.
    Only XPU2 and XPU3 support float16 amp.
    """
    place = _current_expected_place()
    return (
        core.get_xpu_device_version(place.get_device_id())
        >= core.XPUVersion.XPU2
    )


def _is_xpu_bfloat16_supported() -> bool:
    """
    Judge whether current xpu device support bfloat16 amp.
    Only XPU3 support bfloat16 amp.
    Although XPU2 supports bfloat16 computing, but XPU2's bfloat16 operators haven't been widely covered.
    """
    place = _current_expected_place()
    return (
        core.get_xpu_device_version(place.get_device_id())
        >= core.XPUVersion.XPU3
    )


def _is_custom_device_bfloat16_supported() -> bool:
    """
    Judge whether current custom device support bfloat16 amp.
    """
    place = _current_expected_place()
    return (
        place.get_device_type() == 'npu'
        or place.get_device_type() == 'intel_hpu'
    )


def need_keep_fp32(layer: Layer, dtype: str) -> bool:
    need_keep_fp32 = False
    # Highest priority. Because all the layers except BN will use bfloat16 params in bfloat16 training,
    # here we provide a option to keep fp32 param.
    if not layer._cast_to_low_precision:
        need_keep_fp32 = True
    # The BN layers will keep fp32
    elif isinstance(
        layer,
        (
            paddle.nn.BatchNorm,
            paddle.nn.BatchNorm1D,
            paddle.nn.BatchNorm2D,
            paddle.nn.BatchNorm3D,
            paddle.nn.SyncBatchNorm,
        ),
    ):
        need_keep_fp32 = True
    # layer._dtype is used to set params dtype. BF16 will use bf16 params.
    elif (layer._dtype == 'float16') or (
        (dtype == 'float16')
        and isinstance(
            layer,
            (
                paddle.nn.LayerNorm,
                paddle.nn.InstanceNorm1D,
                paddle.nn.InstanceNorm2D,
                paddle.nn.InstanceNorm3D,
            ),
        )
    ):
        need_keep_fp32 = True

    return need_keep_fp32


def set_excluded_layers(
    models: list[Layer],
    excluded_layers: Layer | list[Layer | type[Layer]] | type[Layer],
) -> None:
    excluded_layers_instances = []
    excluded_layers_types = []
    error_message = "excluded_layers must be either a nn.Layer instance/type or a list of nn.Layer instances/types."
    if excluded_layers is None:
        excluded_layers = []
    elif isinstance(excluded_layers, paddle.nn.Layer):
        excluded_layers_instances = [excluded_layers]
    elif isinstance(excluded_layers, type) and issubclass(
        excluded_layers, paddle.nn.Layer
    ):
        excluded_layers_types = [excluded_layers]
    elif isinstance(excluded_layers, list):
        for item in excluded_layers:
            if isinstance(item, paddle.nn.Layer):
                excluded_layers_instances.append(item)
            elif issubclass(item, paddle.nn.Layer):
                excluded_layers_types.append(item)
            else:
                raise TypeError(error_message)
    else:
        raise TypeError(error_message)

    for idx in range(len(excluded_layers_instances)):
        for layer in excluded_layers_instances[idx].sublayers(
            include_self=True
        ):
            layer._cast_to_low_precision = False
    excluded_layers_types = tuple(excluded_layers_types)
    for idx in range(len(models)):
        for layer in models[idx].sublayers(include_self=True):
            if isinstance(layer, excluded_layers_types):
                layer._cast_to_low_precision = False


def _pir_apply(
    self: Layer,
    func: Callable[[Tensor, _DTypeLiteral], Tensor | None],
    dtype: _DTypeLiteral,
    include_sublayers: bool = True,
) -> None:
    if include_sublayers:
        for layer in self.children():
            _pir_apply(layer, func, dtype, include_sublayers)

    for key, param in self._parameters.items():
        if param is not None:
            param_applied = func(param, dtype)

    for key, buf in self._buffers.items():
        if buf is not None:
            self._buffers[key] = func(buf, dtype)

    self._dtype = dtype


def _pir_transform(t: Tensor, dtype: str) -> None:
    main = paddle.static.default_main_program()
    startup = paddle.static.default_startup_program()
    with paddle.static.program_guard(startup):
        block = startup.global_block()
        for op in block.ops:
            if (
                op.name() == 'builtin.set_parameter'
                and op.attrs()['parameter_name'] == t.name
            ):
                param = op.operand(0).source()
                cast_param = paddle.cast(param, dtype)
                cast_param.persistable = True
                paddle._pir_ops.update_parameter(cast_param, t.name)
                block.remove_op(op)
                break
    main.set_parameters_from(startup)
    with paddle.static.program_guard(main):
        paddle.pir.reset_insertion_point_to_start()
        block = main.global_block()
        cast_param = paddle._pir_ops.parameter(t.name)
        cast_param.trainable = t.trainable
        cast_param.stop_gradient = t.stop_gradient
        cast_param.persistable = t.persistable
        cast_param.optimize_attr = t.optimize_attr
        cast_param.regularizer = t.regularizer
        cast_param.do_model_average = t.do_model_average
        cast_param.need_clip = t.need_clip
        cast_param.is_distributed = t.is_distributed
        cast_param.is_parameter = t.is_parameter
        op = t.get_defining_op()
        t.replace_all_uses_with(cast_param)
        block.remove_op(op)
        t.value_assign(cast_param)


def _pir_to_impl(
    self: Layer,
    dtype: _DTypeLiteral,
    include_sublayers: bool,
    floating_only: bool,
) -> Layer:
    def transform(t: Tensor, dtype: _DTypeLiteral) -> Tensor | None:
        if floating_only and (not paddle.is_floating_point(t)):
            return t
        return _pir_transform(t, dtype)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        _pir_apply(self, transform, dtype, include_sublayers)

    self._dtype = dtype
    return self


def amp_initialize(
    models: list[Layer],
    dtype: _DTypeLiteral,
    excluded_layers: Layer | list[Layer | type[Layer]] | type[Layer],
) -> list[Layer]:
    set_excluded_layers(models, excluded_layers)
    for idx in range(len(models)):
        for layer in models[idx].sublayers(include_self=True):
            if need_keep_fp32(layer, dtype):
                continue
            if dtype == "float16" and isinstance(
                layer,
                (
                    paddle.incubate.nn.FusedFeedForward,
                    paddle.incubate.nn.FusedMultiHeadAttention,
                ),
            ):
                layer._amp_decorate(dtype=dtype)
                continue

            if in_pir_mode():
                _pir_to_impl(
                    layer,
                    dtype=dtype,
                    include_sublayers=False,
                    floating_only=True,
                )
            else:
                layer._to_impl(
                    dtype=dtype, include_sublayers=False, floating_only=True
                )
    return models


def check_models(models: list[Layer]) -> None:
    for model in models:
        if not isinstance(model, paddle.nn.Layer):
            raise RuntimeError(
                f"Current train mode is pure fp16, models should be paddle.nn.Layer, but receive {type(model)}."
            )
        if isinstance(model, paddle.DataParallel):
            raise RuntimeError(
                "For distributed AMP training, you should first use paddle.amp.decorate() to decorate origin model, and then call paddle.DataParallel get distributed model."
            )


def _is_valid_optimizer(optimizer: Any) -> TypeGuard[_OptimizerLike]:
    from paddle.distributed.fleet.meta_optimizers.dygraph_optimizer.dygraph_sharding_optimizer import (
        DygraphShardingOptimizer,
        DygraphShardingOptimizerV2,
    )

    return isinstance(
        optimizer,
        (
            paddle.optimizer.Optimizer,
            DygraphShardingOptimizer,
            DygraphShardingOptimizerV2,
        ),
    )


def check_optimizers(optimizers: list[Any]) -> None:
    for optimizer in optimizers:
        if not _is_valid_optimizer(optimizer):
            raise RuntimeError(
                f"Current train mode is pure fp16, optimizers should be paddle.optimizer.Optimizer or DygraphShardingOptimizer, but receive {type(optimizer)}."
            )


@signature_safe_contextmanager
def amp_guard(
    enable: bool = True,
    custom_white_list: _CustomList | None = None,
    custom_black_list: _CustomList | None = None,
    level: _AmpLevelLiteral = 'O1',
    dtype: _DTypeLiteral = 'float16',
    use_promote: bool = True,
) -> Generator[None, None, None]:
    """
    Create a context which enables auto-mixed-precision(AMP) of operators executed in dynamic graph mode.
    If enabled, the input data type (float32 or float16) of each operator is decided
    by autocast algorithm for better performance.

    Commonly, it is used together with `GradScaler` to achieve Auto-Mixed-Precision in
    imperative mode. It is used together with `decorator` to achieve Pure fp16 in imperative mode.

    Args:
        enable(bool, optional): Enable auto-mixed-precision or not. Default is True.
        custom_white_list(set|list|tuple|None, optional): The custom white_list. It's the set of ops that support
             fp16 calculation and are considered numerically-safe and performance-critical. These ops
             will be converted to fp16.
        custom_black_list(set|list|tuple|None, optional): The custom black_list. The set of ops that support fp16
             calculation and are considered numerically-dangerous and whose effects may also be
             observed in downstream ops. These ops will not be converted to fp16.
        level(str, optional): Auto mixed precision level. Accepted values are "O1" and "O2": O1 represent mixed precision, the input data type of each operator will be casted by white_list and black_list;
             O2 represent Pure fp16, all operators parameters and input data will be casted to fp16, except operators in black_list, don't support fp16 kernel and batchnorm. Default is O1(amp).
        dtype(str, optional): Whether to use 'float16' or 'bfloat16'. Default is 'float16'.
        use_promote(bool, optional): Whether op's dtype is 'float32', accord 'Promote to the Widest' principle, use 'float32' to calculate.
             Only active on 'AMP-02'. Default is True.

    Examples:

        .. code-block:: python

            >>> # doctest: +REQUIRES(env:GPU)
            >>> import paddle

            >>> data = paddle.uniform([10, 3, 32, 32], paddle.float32, -1, 1)
            >>> conv2d = paddle.nn.Conv2D(3, 2, 3, bias_attr=False)
            >>> conv2d = paddle.amp.amp_decorate(models=conv2d, level='O2')
            >>> with paddle.amp.amp_guard():
            ...     conv = conv2d(data)
            ...     print(conv.dtype)
            >>> # doctest: +SKIP("This has diff in xdoctest env")
            paddle.float16
            >>> # doctest: -SKIP
            ...
            >>> with paddle.amp.amp_guard(enable=False):
            ...     conv = conv2d(data)
            ...     print(conv.dtype)
            >>> # doctest: +SKIP("This has diff in xdoctest env")
            paddle.float32
            >>> # doctest: -SKIP
    """
    assert (
        in_dynamic_or_pir_mode()
    ), "We only support 'amp_guard' in dynamic or pir mode."

    amp_state = locals()
    global _g_amp_state_
    original_state = _g_amp_state_
    _g_amp_state_ = amp_state

    # check amp_level: O0-O2
    level = level.upper()
    if level not in ['O0', 'OD', 'O1', 'O2']:
        raise ValueError("level should be O0, OD, O1 or O2.")

    # check amp_dtype: float16 or bfloat16
    dtype = dtype.lower()
    if enable:
        if dtype not in ['float16', 'bfloat16']:
            raise ValueError(
                "If enable amp, dtype should be 'float16' or 'bfloat16'."
            )

    amp_dtype = dtype
    amp_global_state().amp_dtype = amp_dtype

    if level == 'OD':
        amp_level = AMP_LEVEL.OD
    elif level == 'O1':
        amp_level = AMP_LEVEL.O1
    elif level == 'O2':
        amp_level = AMP_LEVEL.O2
    elif level == 'O0':
        amp_level = AMP_LEVEL.O0

    _white_list, _black_list = _update_list(
        custom_white_list, custom_black_list, level, dtype
    )

    if in_pir_mode():
        if not enable:
            amp_level = AMP_LEVEL.O0
            amp_dtype = "float32"
        amp_attrs = core._get_amp_attrs()
        # set amp level
        original_amp_level = amp_attrs._amp_level
        amp_attrs._amp_level = amp_level
        # set amp op list
        original_white_list, original_black_list = core._get_amp_op_list()
        core._set_amp_op_list(_white_list, _black_list)
        # set amp dtype
        original_amp_dtype = amp_attrs._amp_dtype
        amp_attrs._amp_dtype = amp_dtype
        # switch promote
        if amp_level == AMP_LEVEL.O2:
            original_use_promote = amp_attrs._use_promote
            amp_attrs._use_promote = use_promote

        try:
            yield
        finally:
            _g_amp_state_ = original_state
            amp_attrs._amp_level = original_amp_level
            core._set_amp_op_list(original_white_list, original_black_list)
            amp_attrs._amp_dtype = original_amp_dtype
            if amp_level == AMP_LEVEL.O2:
                amp_attrs._use_promote = original_use_promote

    else:
        # check tracer
        tracer = _dygraph_tracer()
        if not tracer:
            raise ValueError(
                "current_tracer is None, maybe it is not in imperative mode."
            )
        # check device_type:
        # NOTE: Now, amp only support gpu for float16 and bfloat16, xpu for float16, npu for float16 and bfloat16.
        # Maybe we will support cpu for bfloat16.
        if enable and not (
            tracer._expected_place.is_gpu_place()
            or tracer._expected_place.is_xpu_place()
            or tracer._expected_place.is_custom_place()
        ):
            warnings.warn(
                f'amp_guard can only be enabled on CUDAPlace, XPUPlace, and CustomPlace, current place is {tracer._expected_place}, so it makes no effect.'
            )
            enable = False
        if enable:
            # For xpu:
            if tracer._expected_place.is_xpu_place():
                if (dtype == 'float16') and not _is_xpu_float16_supported():
                    xpu_version = core.get_xpu_device_version(
                        _current_expected_place().get_device_id()
                    )
                    warnings.warn(
                        f'{core.XPUVersion(xpu_version)} does not support float16 amp.'
                    )
                    enable = False
                elif (dtype == 'bfloat16') and not _is_xpu_bfloat16_supported():
                    xpu_version = core.get_xpu_device_version(
                        _current_expected_place().get_device_id()
                    )
                    warnings.warn(
                        f'{core.XPUVersion(xpu_version)} does not support bfloat16 amp.'
                    )
                    enable = False
            # For custom device:
            if (
                tracer._expected_place.is_custom_place()
                and not _is_custom_device_bfloat16_supported()
                and (dtype == 'bfloat16')
            ):
                warnings.warn('CustomPlace only support float16 amp.')
                enable = False
            # For gpu float16: Compute Capability should >= 7.
            # For gpu bfloat16: Compute Capability should >= 8 & CUDA Version should >= 11.
            if tracer._expected_place.is_gpu_place():
                if (dtype == 'float16') and not _is_gpu_float16_supported():
                    prop = paddle.device.cuda.get_device_capability()
                    warnings.warn(
                        f"For float16, amp only support NVIDIA GPU with Compute Capability 7.0 or higher, current GPU is: {paddle.device.cuda.get_device_name()}, with Compute Capability: {prop[0]}.{prop[1]}."
                    )
                    enable = False
                elif (dtype == 'bfloat16') and not _is_gpu_bfloat16_supported():
                    prop = paddle.device.cuda.get_device_capability()
                    cuda_version = paddle.version.cuda()
                    warnings.warn(
                        f"For bfloat16, amp only support NVIDIA GPU with Compute Capability 8.0 or higher and CUDA Version 11.0 or higher, current GPU is: {paddle.device.cuda.get_device_name()}, with Compute Capability: {prop[0]}.{prop[1]}, current CUDA Version is: {cuda_version}."
                    )
                    enable = False

        if not enable:
            amp_level = AMP_LEVEL.O0
            amp_dtype = "float32"

        # master_grad_hook will run at the end of backward.
        # Since backward_final_hook will be cleared once they have been
        # done, we should register the hook every step.
        if (
            amp_global_state().use_master_grad
            and not amp_global_state().already_register_final_backward_hook
        ):

            def master_grad_hook():
                # NOTE(lizhiyu): To support semi-auto of dygraph mode, we must
                # classify the params of model into different classes according to their process_mesh.
                # Otherwise, fault will occur.
                if not amp_global_state().already_classify_params_meshes:
                    for param in amp_global_state().model_parameters:
                        if param is not None and param.process_mesh is not None:
                            if (
                                param.process_mesh
                                not in amp_global_state().mesh2params
                            ):
                                amp_global_state().mesh2params[
                                    param.process_mesh
                                ] = [param]
                            else:
                                amp_global_state().mesh2params[
                                    param.process_mesh
                                ].append(param)
                    amp_global_state().already_classify_params_meshes = True

                if len(amp_global_state().mesh2params):
                    for _, params in amp_global_state().mesh2params.items():
                        core.eager.set_master_grads(params)
                else:
                    core.eager.set_master_grads(
                        amp_global_state().model_parameters
                    )

                amp_global_state().already_register_final_backward_hook = False

            core.eager._add_backward_final_hook(master_grad_hook)
            amp_global_state().already_register_final_backward_hook = True

        if tracer:
            # enable auto_cast
            original_amp_level = tracer._amp_level
            tracer._amp_level = amp_level

            # set amp op list
            original_white_list, original_black_list = tracer._get_amp_op_list()
            tracer._set_amp_op_list(_white_list, _black_list)

            # TODO(zhiqiu) set amp related flags automatically in this guard
            # Currently, if FLAGS_cudnn_batchnorm_spatial_persistent is set True in amp_guard,
            # batch_norm can run in fast mode, but batch_norm_grad can not if backward if not executed inside amp_guard.
            # So, users need to set related flags manually.

            # original_flags = get_flags(AMP_RELATED_FLAGS)
            # set_flags(AMP_RELATED_FLAGS_SETTING)

            # set amp dtype
            original_amp_dtype = tracer._amp_dtype
            tracer._amp_dtype = amp_dtype

            # switch promote
            if amp_level == AMP_LEVEL.O2:
                original_use_promote = tracer._use_promote
                tracer._use_promote = use_promote

        # restore status
        try:
            yield
        finally:
            if tracer:
                _g_amp_state_ = original_state
                tracer._amp_level = original_amp_level
                tracer._set_amp_op_list(
                    original_white_list, original_black_list
                )
                # set_flags(original_flags)
                tracer._amp_dtype = original_amp_dtype
                if amp_level == AMP_LEVEL.O2:
                    tracer._use_promote = original_use_promote


class StateDictHook:
    def __init__(self, save_dtype: str) -> None:
        self._save_dtype = save_dtype

    def __call__(self, state_dict: _StateDict) -> None:
        for key in state_dict:
            param = state_dict[key]
            if paddle.is_floating_point(param):
                param_applied = paddle.cast(param, self._save_dtype)
                param_applied.name = param.name
                state_dict[key] = param_applied


def _set_multi_precision(
    optimizer: _OptimizerLike, multi_precision: bool
) -> None:
    from paddle.distributed.fleet.meta_optimizers.dygraph_optimizer.dygraph_sharding_optimizer import (
        DygraphShardingOptimizer,
        DygraphShardingOptimizerV2,
    )

    optimizer = (
        optimizer._inner_opt
        if isinstance(
            optimizer, (DygraphShardingOptimizer, DygraphShardingOptimizerV2)
        )
        else optimizer
    )
    if hasattr(optimizer, "_multi_precision"):
        optimizer._multi_precision = multi_precision


@overload
def amp_decorate(
    models: _ModelsT,
    optimizers: _OptimizersT = ...,
    level: _AmpLevelLiteral = ...,
    dtype: _DTypeLiteral = ...,
    master_weight: bool | None = ...,
    save_dtype: _DTypeLiteral | None = ...,
    master_grad: bool = ...,
    excluded_layers: (
        Layer | list[Layer | type[Layer]] | type[Layer] | None
    ) = ...,
) -> tuple[_ModelsT, _OptimizersT]: ...


@overload
def amp_decorate(
    models: _ModelsT,
    optimizers: Literal[None] = ...,
    level: _AmpLevelLiteral = ...,
    dtype: _DTypeLiteral = ...,
    master_weight: bool | None = ...,
    save_dtype: _DTypeLiteral | None = ...,
    master_grad: bool = ...,
    excluded_layers: (
        Layer | list[Layer | type[Layer]] | type[Layer] | None
    ) = ...,
) -> _ModelsT: ...


@dygraph_only
def amp_decorate(
    models: _ModelsT,
    optimizers: _OptimizersT | None = None,
    level: _AmpLevelLiteral = 'O1',
    dtype: _DTypeLiteral = 'float16',
    master_weight: bool | None = None,
    save_dtype: _DTypeLiteral | None = None,
    master_grad: bool = False,
    excluded_layers: (
        Layer | list[Layer | type[Layer]] | type[Layer] | None
    ) = None,
) -> tuple[_ModelsT, _OptimizersT] | _ModelsT:
    """
    Decorate models and optimizers for auto-mixed-precision. When level is O1(amp), the decorate will do nothing.
    When level is O2(pure fp16), the decorate will cast all parameters of models to FP16, except BatchNorm, InstanceNorm and LayerNorm.

    Commonly, it is used together with `amp_guard` to achieve Pure fp16 in imperative mode.

    Args:
        models(Layer|list of Layer, optional): The defined models by user, models must be either a single model or a list of models. Default is None.
        optimizers(Optimizer|list of Optimizer|None, optional): The defined optimizers by user, optimizers must be either a single optimizer or a list of optimizers. Default is None.
        level(str, optional): Auto mixed precision level. Accepted values are "O1" and "O2": O1 represent mixed precision, the decorator will do nothing;
             O2 represent Pure fp16/bf16, the decorator will cast all parameters of models to FP16/BF16, except BatchNorm, InstanceNorm and LayerNorm. Default is O1(amp)
        dtype(str, optional): Whether to use 'float16' or 'bfloat16'. Default is 'float16'.
        master_weight(bool|None, optional): For level='O2', whether to use multi-precision during weight updating. If master_weight is None, in O2 level optimizer will use multi-precision. Default is None.
        save_dtype(str|None, optional): The save model parameter dtype when use `paddle.save` or `paddle.jit.save`,it should be float16, bfloat16, float32, float64 or None.
             The save_dtype will not change model parameters dtype, it just change the state_dict dtype. When save_dtype is None, the save dtype is same as model dtype. Default is None.
        master_grad(bool, optional): For level='O2', whether to use float32 weight gradients for calculations such as gradient clipping, weight decay, and weight updates. If master_grad is enabled, the weight
             gradients will be float32 dtype after the back propagation. Default is False, there is only float16 weight gradients.
        excluded_layers(Layer|list of Layer, optional): Specify the layers not to be decorated. The weights of these layers will always keep float32 when level is O2. `excluded_layers` can be specified as
             an Layer instance/type or a list of Layer instances/types. Default is None, the weights of the whole model will be casted to float16 or bfloat16.

    Examples:

        .. code-block:: python

            >>> # doctest: +REQUIRES(env:GPU)
            >>> # Demo1: single model and optimizer:
            >>> import paddle
            >>> paddle.device.set_device('gpu')

            >>> model = paddle.nn.Conv2D(3, 2, 3, bias_attr=False)
            >>> optimizer = paddle.optimizer.SGD(parameters=model.parameters())

            >>> model, optimizer = paddle.amp.amp_decorate(models=model, optimizers=optimizer, level='O2')

            >>> data = paddle.rand([10, 3, 32, 32])

            >>> with paddle.amp.amp_guard(enable=True, custom_white_list=None, custom_black_list=None, level='O2'):
            ...     output = model(data)
            ...     print(output.dtype)
            paddle.float16

            >>> # Demo2: multi models and optimizers:
            >>> model2 = paddle.nn.Conv2D(3, 2, 3, bias_attr=False)
            >>> optimizer2 = paddle.optimizer.Adam(parameters=model2.parameters())

            >>> models, optimizers = paddle.amp.amp_decorate(models=[model, model2], optimizers=[optimizer, optimizer2], level='O2')

            >>> data = paddle.rand([10, 3, 32, 32])

            >>> with paddle.amp.amp_guard(enable=True, custom_white_list=None, custom_black_list=None, level='O2'):
            ...     output = models[0](data)
            ...     output2 = models[1](data)
            ...     print(output.dtype)
            ...     print(output2.dtype)
            paddle.float16
            paddle.float16

            >>> # Demo3: optimizers is None:
            >>> model3 = paddle.nn.Conv2D(3, 2, 3, bias_attr=False)
            >>> optimizer3 = paddle.optimizer.Adam(parameters=model2.parameters())

            >>> model = paddle.amp.amp_decorate(models=model3, level='O2')

            >>> data = paddle.rand([10, 3, 32, 32])

            >>> with paddle.amp.amp_guard(enable=True, custom_white_list=None, custom_black_list=None, level='O2'):
            ...     output = model(data)
            ...     print(output.dtype)
            paddle.float16
    """
    if level not in ['O1', 'O2']:
        raise ValueError(
            "level should be O1 or O2, O1 represent AMP train mode, O2 represent Pure fp16 train mode."
        )
    if dtype not in ['float16', 'bfloat16']:
        raise ValueError("dtype only support float16 or bfloat16.")

    if level == 'O1':
        if optimizers is None:
            return models
        else:
            return models, optimizers

    # check tracer
    tracer = _dygraph_tracer()
    if not tracer:
        raise ValueError(
            "current_tracer is None, maybe it is not in imperative mode."
        )

    # check device_type:
    if not (
        tracer._expected_place.is_gpu_place()
        or tracer._expected_place.is_xpu_place()
        or tracer._expected_place.is_custom_place()
    ):
        if optimizers is None:
            return models
        else:
            return models, optimizers
    # For xpu:
    if tracer._expected_place.is_xpu_place():
        if (dtype == 'float16' and not _is_xpu_float16_supported()) or (
            dtype == 'bfloat16' and not _is_xpu_bfloat16_supported()
        ):
            if optimizers is None:
                return models
            else:
                return models, optimizers
    # For custom device:
    if (
        tracer._expected_place.is_custom_place()
        and not _is_custom_device_bfloat16_supported()
        and (dtype == 'bfloat16')
    ):
        if optimizers is None:
            return models
        else:
            return models, optimizers
    # For gpu float16: Compute Capability should >= 7.
    # For gpu bfloat16: Compute Capability should >= 8 & CUDA Version should >= 11.
    if tracer._expected_place.is_gpu_place():
        if (dtype == 'float16' and not _is_gpu_float16_supported()) or (
            dtype == 'bfloat16' and not _is_gpu_bfloat16_supported()
        ):
            if optimizers is None:
                return models
            else:
                return models, optimizers

    models_is_list = False
    if isinstance(models, paddle.nn.Layer):
        models_is_list = False
        models = [models]
        check_models(models)
    elif isinstance(models, list):
        check_models(models)
        models_is_list = True
    else:
        raise TypeError(
            "models must be either a single model or a list of models."
        )

    # initialize parameters of the model.
    amp_initialize(models=models, dtype=dtype, excluded_layers=excluded_layers)

    if optimizers is not None:
        # check optimizers
        optimizers_is_list = False
        if _is_valid_optimizer(optimizers):
            optimizers_is_list = False
            optimizers = [optimizers]
            check_optimizers(optimizers)
        elif isinstance(optimizers, list):
            check_optimizers(optimizers)
            optimizers_is_list = True
        else:
            raise TypeError(
                "optimizers must be either a single optimizer or a list of optimizers."
            )
        # support master_weight
        use_multi_precision = master_weight is not False
        for opt in optimizers:
            _set_multi_precision(opt, use_multi_precision)

    # support master_grad
    if master_grad:
        amp_global_state().use_master_grad = True
        for idx in range(len(models)):
            amp_global_state().model_parameters.extend(models[idx].parameters())

    if save_dtype is not None:
        if save_dtype not in ['float16', 'bfloat16', 'float32', 'float64']:
            raise ValueError(
                f"save_dtype can only be float16 float32 or float64, but your input save_dtype is {save_dtype}."
            )
        for idx in range(len(models)):
            for layer in models[idx].sublayers(include_self=True):
                layer.register_state_dict_hook(StateDictHook(save_dtype))

    if models_is_list:
        if optimizers is not None:
            if optimizers_is_list:
                return models, optimizers
            else:
                return models, optimizers[0]
        else:
            return models
    else:
        if optimizers is not None:
            if optimizers_is_list:
                return models[0], optimizers
            else:
                return models[0], optimizers[0]
        else:
            return models[0]


def auto_cast(
    enable: bool = True,
    custom_white_list: _CustomList | None = None,
    custom_black_list: _CustomList | None = None,
    level: _AmpLevelLiteral = 'O1',
    dtype: _DTypeLiteral = 'float16',
    use_promote: bool = True,
) -> ContextManager:
    """
    Create a context which enables auto-mixed-precision(AMP) of operators executed in dynamic graph mode.
    If enabled, the input data type (float32, float16 or bfloat16) of each operator is decided
    by autocast algorithm for better performance.

    Commonly, it is used together with `GradScaler` and `decorator` to achieve Auto-Mixed-Precision in
    imperative mode.

    Args:
        enable(bool, optional): Enable auto-mixed-precision or not. Default is True.
        custom_white_list(set|list|tuple|None, optional): A default white list is already set. Usually there is no need to set custom white list.
             The set of ops should be considered numerically-safe and performance-critical. These ops will be converted to float16/bfloat16.
        custom_black_list(set|list|tuple|None, optional): A default black list is already set. You can set a custom black list according to the model.
             The set of ops are considered numerically-dangerous and whose effects may also be observed in downstream ops. These ops will not be
             converted to float16/bfloat16.
        level(str, optional): Auto mixed precision level. Accepted values are "O1", "O2" and "OD": At the O1 level, operators in the white list
             will use float16/bfloat16 inputs for calculations, and operators in the black list will use float32 inputs for calculations. At the O2
             level, model's parameters will be casted to float16/bfloat16 by using `decorator`, and operators that have all float16/bfloat16 inputs
             will be converted to float16/bfloat16, and that have any float32 input will be converted to float32. For the OD level, operators in
             default white list will compute in float16/bfloat16, and the others will compute in float32. Default is O1.
        dtype(str, optional): Whether to use 'float16' or 'bfloat16'. Default is 'float16'.
        use_promote(bool, optional): Whether to promotes to fp32 when op has any float32 inputs. It is only supported when amp level is O2. Default is True.

    Examples:

        .. code-block:: python

            >>> # doctest: +REQUIRES(env:GPU)
            >>> import paddle

            >>> conv2d = paddle.nn.Conv2D(3, 2, 3, bias_attr=False)
            >>> data = paddle.rand([10, 3, 32, 32])

            >>> with paddle.amp.auto_cast():
            ...     conv = conv2d(data)
            ...     print(conv.dtype)
            >>> # doctest: +SKIP("This has diff in xdoctest env")
            paddle.float16
            >>> # doctest: -SKIP

            >>> with paddle.amp.auto_cast(enable=False):
            ...     conv = conv2d(data)
            ...     print(conv.dtype)
            >>> # doctest: +SKIP("This has diff in xdoctest env")
            paddle.float32
            >>> # doctest: -SKIP

            >>> with paddle.amp.auto_cast(custom_black_list={'conv2d'}):
            ...     conv = conv2d(data)
            ...     print(conv.dtype)
            >>> # doctest: +SKIP("This has diff in xdoctest env")
            paddle.float32
            >>> # doctest: -SKIP

            >>> a = paddle.rand([2, 3])
            >>> b = paddle.rand([2, 3])
            >>> with paddle.amp.auto_cast(custom_white_list={'elementwise_add'}):
            ...     c = a + b
            ...     print(c.dtype)
            >>> # doctest: +SKIP("This has diff in xdoctest env")
            paddle.float16
            >>> # doctest: -SKIP

            >>> with paddle.amp.auto_cast(custom_white_list={'elementwise_add'}, level='O2'):
            ...     d = a + b
            ...     print(d.dtype)
            >>> # doctest: +SKIP("This has diff in xdoctest env")
            paddle.float16
            >>> # doctest: -SKIP

    """
    return amp_guard(
        enable, custom_white_list, custom_black_list, level, dtype, use_promote
    )


@overload
def decorate(
    models: _ModelsT,
    optimizers: _OptimizersT,
    level: _AmpLevelLiteral = ...,
    dtype: _DTypeLiteral = ...,
    master_weight: bool | None = ...,
    save_dtype: _DTypeLiteral | None = ...,
    master_grad: bool = ...,
    excluded_layers: (
        Layer | list[Layer | type[Layer]] | type[Layer] | None
    ) = ...,
) -> tuple[_ModelsT, _OptimizersT]: ...


@overload
def decorate(
    models: _ModelsT,
    optimizers: None = ...,
    level: _AmpLevelLiteral = ...,
    dtype: _DTypeLiteral = ...,
    master_weight: bool | None = ...,
    save_dtype: _DTypeLiteral | None = ...,
    master_grad: bool = ...,
    excluded_layers: (
        Layer | list[Layer | type[Layer]] | type[Layer] | None
    ) = ...,
) -> _ModelsT: ...


def decorate(
    models: _ModelsT,
    optimizers: _OptimizersT | None = None,
    level: _AmpLevelLiteral = 'O1',
    dtype: _DTypeLiteral = 'float16',
    master_weight: bool | None = None,
    save_dtype: _DTypeLiteral | None = None,
    master_grad: bool = False,
    excluded_layers: (
        Layer | list[Layer | type[Layer]] | type[Layer] | None
    ) = None,
) -> tuple[_ModelsT, _OptimizersT] | _ModelsT:
    """
    Decorate models and optimizers for auto-mixed-precision. When level is O1(amp), the decorate will do nothing.
    When level is O2(pure float16/bfloat16), the decorate will cast all parameters of models to float16/bfloat16, except BatchNorm, InstanceNorm and LayerNorm.

    Commonly, it is used together with `auto_cast` to achieve Pure float16/bfloat16 in imperative mode.

    Args:
        models(Layer|list of Layer): The defined models by user, models must be either a single model or a list of models. Default is None.
        optimizers(Optimizer|list of Optimizer|None, optional): The defined optimizers by user, optimizers must be either a single optimizer or a list of optimizers. Default is None.
        level(str, optional): Auto mixed precision level. Accepted values are 'O1' and 'O2': O1 represent mixed precision, the decorator will do nothing;
             O2 represent Pure float16/bfloat16, the decorator will cast all parameters of models to float16/bfloat16, except BatchNorm, InstanceNorm and LayerNorm. Default is O1(amp)
        dtype(str, optional): Whether to use 'float16' or 'bfloat16'. Default is 'float16'.
        master_weight(bool, optional): For level='O2', whether to use multi-precision during weight updating. If master_weight is None, in O2 level optimizer will use multi-precision. Default is None.
        save_dtype(str|None, optional): The save model parameter dtype when use `paddle.save` or `paddle.jit.save`,it should be float16, bfloat16, float32, float64 or None.
             The save_dtype will not change model parameters dtype, it just change the state_dict dtype. When save_dtype is None, the save dtype is same as model dtype. Default is None.
        master_grad(bool, optional): For level='O2', whether to use float32 weight gradients for calculations such as gradient clipping, weight decay, and weight updates. If master_grad is enabled, the weight
             gradients will be float32 dtype after the back propagation. Default is False, there is only float16 weight gradients.
        excluded_layers(Layer|list of Layer, optional): Specify the layers not to be decorated. The weights of these layers will always keep float32 when level is O2. `excluded_layers` can be specified as
             an Layer instance/type or a list of Layer instances/types. Default is None, the weights of the whole model will be casted to float16 or bfloat16.

    Examples:

        .. code-block:: python

            >>> # doctest: +REQUIRES(env:GPU)
            >>> # Demo1: single model and optimizer:
            >>> import paddle
            >>> paddle.device.set_device('gpu')

            >>> model = paddle.nn.Conv2D(3, 2, 3, bias_attr=False)
            >>> optimizer = paddle.optimizer.SGD(parameters=model.parameters())

            >>> model, optimizer = paddle.amp.decorate(models=model, optimizers=optimizer, level='O2')

            >>> data = paddle.rand([10, 3, 32, 32])

            >>> with paddle.amp.auto_cast(enable=True, custom_white_list=None, custom_black_list=None, level='O2'):
            ...     output = model(data)
            ...     print(output.dtype)
            paddle.float16

            >>> # Demo2: multi models and optimizers:
            >>> model2 = paddle.nn.Conv2D(3, 2, 3, bias_attr=False)
            >>> optimizer2 = paddle.optimizer.Adam(parameters=model2.parameters())

            >>> models, optimizers = paddle.amp.decorate(models=[model, model2], optimizers=[optimizer, optimizer2], level='O2')

            >>> data = paddle.rand([10, 3, 32, 32])

            >>> with paddle.amp.auto_cast(enable=True, custom_white_list=None, custom_black_list=None, level='O2'):
            ...     output = models[0](data)
            ...     output2 = models[1](data)
            ...     print(output.dtype)
            ...     print(output2.dtype)
            paddle.float16
            paddle.float16

            >>> # Demo3: optimizers is None:
            >>> model3 = paddle.nn.Conv2D(3, 2, 3, bias_attr=False)
            >>> optimizer3 = paddle.optimizer.Adam(parameters=model3.parameters())

            >>> model = paddle.amp.decorate(models=model3, level='O2')

            >>> data = paddle.rand([10, 3, 32, 32])

            >>> with paddle.amp.auto_cast(enable=True, custom_white_list=None, custom_black_list=None, level='O2'):
            ...     output = model(data)
            ...     print(output.dtype)
            paddle.float16

    """

    if paddle.framework.in_pir_mode():
        assert not isinstance(models, (list, tuple))
        assert not isinstance(optimizers, (list, tuple))
        amp_global_state().use_master_grad = master_grad
        if level in ['O0', 'OD', 'O1']:
            if optimizers is None:
                return models
            else:
                optimizers = OptimizerWithMixedPrecision(
                    optimizer=optimizers,
                    amp_lists=None,
                    level=level,
                    dtype=dtype,
                    init_loss_scaling=1.0,
                    incr_every_n_steps=None,
                    decr_every_n_nan_or_inf=None,
                    incr_ratio=None,
                    decr_ratio=None,
                    use_dynamic_loss_scaling=False,
                    use_amp_guard=None,
                    use_master_grad=master_grad,
                    use_promote=None,
                )
                return models, optimizers
        elif level == 'O2':
            amp_initialize(
                models=[models], dtype=dtype, excluded_layers=excluded_layers
            )
            use_multi_precision = master_weight is not False
            _set_multi_precision(optimizers, use_multi_precision)
            if optimizers is None:
                return models
            else:
                optimizers = OptimizerWithMixedPrecision(
                    optimizer=optimizers,
                    amp_lists=None,
                    level=level,
                    dtype=dtype,
                    init_loss_scaling=1.0,
                    incr_every_n_steps=None,
                    decr_every_n_nan_or_inf=None,
                    incr_ratio=None,
                    decr_ratio=None,
                    use_dynamic_loss_scaling=False,
                    use_amp_guard=None,
                    use_master_grad=master_grad,
                    use_promote=None,
                )
                return models, optimizers
        else:
            raise ValueError("level should be O0, OD, O1 or O2.")
    else:
        return amp_decorate(
            models,
            optimizers,
            level,
            dtype,
            master_weight,
            save_dtype,
            master_grad,
            excluded_layers,
        )
