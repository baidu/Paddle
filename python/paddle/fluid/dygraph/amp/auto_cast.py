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

from __future__ import print_function
from paddle.fluid.wrapped_decorator import signature_safe_contextmanager, wrap_decorator
from paddle.fluid import core
import contextlib
from paddle.fluid.framework import Variable, in_dygraph_mode, OpProtoHolder, Parameter, _dygraph_tracer, dygraph_only, set_flags, get_flags
import warnings
import copy
import functools
import paddle
import operator
import types

__all__ = ['amp_guard', 'amp_decorator']

# The set of ops that support fp16 calculation and are considered numerically-
# safe and performance-critical. These ops are always converted to fp16.
WHITE_LIST = {
    'conv2d',
    'matmul',
    'matmul_v2',
    'mul',
    'fake_quantize_dequantize_abs_max',
    'fake_quantize_dequantize_moving_average_abs_max',
}

# The set of ops that support fp16 calculation and are considered numerically-
# dangerous and whose effects may also be observed in downstream ops.
BLACK_LIST = {
    'exp',
    'square',
    'log',
    'mean',
    'sum',
    'cos_sim',
    'softmax',
    'softmax_with_cross_entropy',
    'sigmoid_cross_entropy_with_logits',
    'cross_entropy',
    'cross_entropy2',
    # default fp32 can avoid return inf when the sum value large than 65504
    'reduce_sum',
}

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


#NOTE(zhiqiu): similar as paddle.fluid.contrib.mixed_precision.fp16_lists.AutoMixedPrecisionLists._update_list
# The reason why not use AutoMixedPrecisionLists is that custom_black_varnames is not suitable for imperative mode.
def _update_list(custom_white_list, custom_black_list):
    """
    Update black and white list according to users' custom list.
    """
    _white_list = copy.copy(WHITE_LIST)
    _black_list = copy.copy(BLACK_LIST)
    if custom_white_list and custom_black_list:
        for op_name in custom_white_list:
            if op_name in custom_black_list:
                raise ValueError("Custom white list overlap "
                                 "custom black list")
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


def _in_amp_guard():
    """
    Judge whether current code block is in `amp_guard` context.
    """
    tracer = _dygraph_tracer()
    if tracer:
        return tracer._enable_autocast
    else:
        return False


@dygraph_only
def fp16_initialize(enable_pure_fp16, models, optimizers):
    if not enable_pure_fp16:
        return models, optimizers

    if len(models) != len(optimizers):
        raise RuntimeError(
            "Current models num should be equal to optimizers num, but receive {} != {}.".
            format(len(models), len(optimizers)))

    for idx in range(len(models)):
        if getattr(optimizers[idx], '_param_groups', None) and isinstance(
                optimizers[idx]._param_groups[0], dict):
            for p in models[idx].parameters():
                contains = False
                for param_group in optimizers[idx]._param_groups:
                    for q in param_group['params']:
                        if p is q:
                            contains = True
                if not contains:
                    raise RuntimeError(
                        "Current the order of models should be consistent with that of optimizers, but receive models_{} not corresponding to optimizers_{}.".
                        format(idx, idx))
        else:
            for p in models[idx].parameters():
                contains = False
                for q in optimizers[idx]._parameter_list:
                    if p is q:
                        contains = True
                if not contains:
                    raise RuntimeError(
                        "Current the order of models should be consistent with that of optimizers, but receive models_{} not corresponding to optimizers_{}.".
                        format(idx, idx))

    for idx in range(len(models)):
        for layer in models[idx].sublayers(include_self=True):
            if len(layer._sub_layers) is 0:
                if (layer._dtype is 'float16') or isinstance(layer, (
                        paddle.nn.BatchNorm, paddle.nn.LayerNorm)):
                    continue
                layer.to(dtype='float16')

                #以group的dict形式输入的参数
                if getattr(optimizers[idx], '_param_groups',
                           None) and isinstance(
                               optimizers[idx]._param_groups[0], dict):
                    #更新group
                    for param_group in optimizers[idx]._param_groups:
                        for i, param in enumerate(param_group['params']):
                            if id(param) in layer._parameters_transform_map:
                                param_group['params'][
                                    i] = layer._parameters_transform_map[id(
                                        param)][0]
                    #更新list
                    for param_group in optimizers[idx]._parameter_list:
                        params = param_group['params']
                        for i, param in enumerate(params):
                            if id(param) in layer._parameters_transform_map:
                                params[i] = layer._parameters_transform_map[id(
                                    param)][0]
                #以list的形式输入的参数
                else:
                    for i, param in enumerate(optimizers[idx]._parameter_list):
                        if id(param) in layer._parameters_transform_map:
                            optimizers[idx]._parameter_list[
                                i] = layer._parameters_transform_map[id(param)][
                                    0]
                            if hasattr(optimizers[idx], '_param_groups'):
                                optimizers[idx]._param_groups[
                                    i] = layer._parameters_transform_map[id(
                                        param)][0]
    '''
    for idx in range(len(optimizers)):
        if hasattr(optimizers[idx], '_multi_precision'):
            optimizers[idx]._multi_precision = True
    '''

    return models, optimizers


def check_models(models):
    for model in models:
        if not isinstance(model, paddle.nn.Layer):
            raise RuntimeError(
                "Current train mode is pure fp16, models should be paddle.nn.Layer, but receive {}.".
                format(type(model)))


def check_optimizers(optimizers):
    for optimizer in optimizers:
        if not isinstance(optimizer, (paddle.optimizer.Optimizer,
                                      paddle.fluid.optimizer.Optimizer)):
            raise RuntimeError(
                "Current train mode is pure fp16, optimizers should be paddle.optimizer.Optimizer or paddle.fluid.optimizer.Optimizer, but receive {}.".
                format(type(optimizer)))


@signature_safe_contextmanager
@dygraph_only
def amp_guard(enable=True,
              custom_white_list=None,
              custom_black_list=None,
              enable_pure_fp16=False,
              models=None,
              optimizers=None):
    """
    :api_attr: imperative

    Create a context which enables auto-mixed-precision(AMP) of operators executed in imperative mode.
    If enabled, the input data type (float32 or float16) of each operator is decided 
    by autocast algorithm for better performance. 
    
    Commonly, it is used together with `AmpScaler` to achieve Auto-Mixed-Precision in 
    imperative mode.

    Args:
        enable(bool, optional): Enable auto-mixed-precision or not. Default is True.
        custom_white_list(set|list, optional): The custom white_list.
        custom_black_list(set|list, optional): The custom black_list.
        
    Examples:

     .. code-block:: python

        import numpy as np
        import paddle.fluid as fluid

        data = np.random.uniform(-1, 1, [10, 3, 32, 32]).astype('float32')
        with fluid.dygraph.guard():
            conv2d = fluid.dygraph.Conv2D(3, 2, 3)
            data = fluid.dygraph.to_variable(data)
            with fluid.dygraph.amp_guard():
                conv = conv2d(data)
                print(conv.dtype) # FP16
            with fluid.dygraph.amp_guard(enable=False):
                conv = conv2d(data)
                print(conv.dtype) # FP32

    """
    tracer = _dygraph_tracer()
    if not tracer:
        raise ValueError(
            "current_tracer is None, maybe it is not in imperative mode.")

    if enable and not (tracer._expected_place.is_gpu_place() or
                       tracer._expected_place.is_xpu_place()):
        warnings.warn(
            'amp_guard can only be enabled on CUDAPlace and XPUPlace, current place is %s, so it makes no effect.'
            % tracer._expected_place)
        enable = False

    if (not enable) and enable_pure_fp16:
        warnings.warn(
            'When enable autocast is False, enable_pure_fp16 should be False, but current is %s, so it makes no effect.'
            % enable_pure_fp16)
        enable_pure_fp16 = False

    if enable_pure_fp16:
        if isinstance(models, paddle.nn.Layer):
            models = [models]
            check_models(models)
        elif isinstance(models, list):
            check_models(models)
        else:
            raise TypeError(
                "models must be either a single model or a list of models.")

        if isinstance(optimizers, (paddle.optimizer.Optimizer,
                                   paddle.fluid.optimizer.Optimizer)):
            optimizers = [optimizers]
            check_optimizers(optimizers)
        elif isinstance(optimizers, list):
            check_optimizers(optimizers)
        else:
            raise TypeError(
                "optimizers must be either a single optimizer or a list of optimizers."
            )

        models, optimizers = fp16_initialize(enable_pure_fp16, models,
                                             optimizers)
        '''
        if isinstance(models, paddle.nn.Layer) and isinstance(
                optimizers,
            (paddle.optimizer.Optimizer, paddle.fluid.optimizer.Optimizer)):
            model, optimizer = fp16_initialize(enable_pure_fp16, models,
                                               optimizers)
        else:
            raise ValueError(
                "Current train mode is pure fp16, model and optimizer should be paddle.nn.Layer and (paddle.optimizer.Optimizer or paddle.fluid.optimizer.Optimizer), but receive {} and {}.".
                format(type(model), type(optimizer)))
        '''

    # use default white_list and black_list if no custom lists provided
    _white_list = WHITE_LIST
    _black_list = BLACK_LIST
    if custom_white_list or custom_black_list:
        _white_list, _black_list = _update_list(custom_white_list,
                                                custom_black_list)

    if tracer:
        # enable auto_cast
        original_enable = tracer._enable_autocast
        tracer._enable_autocast = enable
        # set amp op list
        original_white_list, original_black_list = tracer._get_amp_op_list()
        tracer._set_amp_op_list(_white_list, _black_list)

        # TODO(zhiqiu) set amp related flags automatically in this guard
        # Currently, if FLAGS_cudnn_batchnorm_spatial_persistent is set True in amp_guard,
        # batch_norm can run in fast mode, but batch_norm_grad can not if backward if not executed insise amp_guard.
        # So, users need to set related flags manually.

        # original_flags = get_flags(AMP_RELATED_FLAGS)
        # set_flags(AMP_RELATED_FLAGS_SETTING)

        original_pure_fp16_enable = tracer._enable_pure_fp16
        tracer._enable_pure_fp16 = enable_pure_fp16

    # restore status
    try:
        yield
    finally:
        if tracer:
            tracer._enable_autocast = original_enable
            tracer._set_amp_op_list(original_white_list, original_black_list)
            # set_flags(original_flags)
            tracer._enable_pure_fp16 = original_pure_fp16_enable


@dygraph_only
def amp_decorator(mode='pure_fp16',
                  models=None,
                  optimizers=None,
                  custom_black_list=None,
                  custom_white_list=None,
                  enable_loss_scaling=True,
                  init_loss_scaling=2.**15,
                  incr_ratio=2.0,
                  decr_ratio=0.5,
                  incr_every_n_steps=1000,
                  decr_every_n_nan_or_inf=2,
                  use_dynamic_loss_scaling=True):
    if mode == 'fp32':
        return models, optimizers

    if not (mode in ['amp', 'pure_fp16']):
        raise ValueError(
            "the input parameter mode should be fp32 or amp or pure_fp16.")

    tracer = _dygraph_tracer()
    if not tracer:
        raise ValueError(
            "current_tracer is None, maybe it is not in imperative mode.")

    if not (tracer._expected_place.is_gpu_place() or
            tracer._expected_place.is_xpu_place()):
        warnings.warn(
            'amp or pure_fp16 mode can only be enabled on CUDAPlace and XPUPlace, current place is %s, so it makes no effect.'
            % tracer._expected_place)
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
            "models must be either a single model or a list of models.")

    optimizers_is_list = False
    if isinstance(optimizers, (paddle.optimizer.Optimizer,
                               paddle.fluid.optimizer.Optimizer)):
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

    _white_list = WHITE_LIST
    _black_list = BLACK_LIST
    if custom_white_list or custom_black_list:
        _white_list, _black_list = _update_list(custom_white_list,
                                                custom_black_list)

    if tracer:
        if mode == 'pure_fp16':
            tracer._enable_autocast = True
            tracer._enable_pure_fp16 = True
            tracer._set_amp_op_list(_white_list, _black_list)
            # 改写model、optimizer、加入scaler策略
            models, optimizers = fp16_initialize(
                enable_pure_fp16=True, models=models, optimizers=optimizers)

        elif mode == 'amp':
            tracer._enable_autocast = True
            tracer._enable_pure_fp16 = False
            tracer._set_amp_op_list(_white_list, _black_list)

    scalers = optimizers
    if enable_loss_scaling:
        for i, optimizer in enumerate(optimizers):
            scalers[i] = paddle.amp.GradScaler(
                enable=enable_loss_scaling,
                init_loss_scaling=init_loss_scaling,
                incr_ratio=incr_ratio,
                decr_ratio=decr_ratio,
                incr_every_n_steps=incr_every_n_steps,
                decr_every_n_nan_or_inf=decr_every_n_nan_or_inf,
                use_dynamic_loss_scaling=use_dynamic_loss_scaling)
            scalers[i].user_defined_optimizer = optimizer

            @dygraph_only
            def scaler_minimize(self, *args, **kwargs):
                tracer = _dygraph_tracer()
                if not tracer:
                    raise ValueError(
                        "current_tracer is None, maybe it is not in imperative mode."
                    )
                origin_enable_autocast = tracer._enable_autocast
                origin_enable_pure_fp16 = tracer._enable_pure_fp16
                tracer._enable_autocast = False
                tracer._enable_pure_fp16 = False

                if not self._enable:
                    return self.user_defined_optimizer.minimize(*args, **kwargs)
                self._unscale(self.user_defined_optimizer)

                optimize_ops, params_grads = (None, None)
                if self._found_inf:
                    self._cache_founf_inf = True
                else:
                    optimize_ops, params_grads = self.user_defined_optimizer.minimize(
                        *args, **kwargs)
                    self._cache_founf_inf = False
                if self._use_dynamic_loss_scaling:
                    self._update()

                tracer._enable_autocast = origin_enable_autocast
                tracer._enable_pure_fp16 = origin_enable_pure_fp16

                return optimize_ops, params_grads

            scalers[i].minimize = types.MethodType(scaler_minimize, scalers[i])

            @dygraph_only
            def scaler_clear_grad(self):
                tracer = _dygraph_tracer()
                if not tracer:
                    raise ValueError(
                        "current_tracer is None, maybe it is not in imperative mode."
                    )
                origin_enable_autocast = tracer._enable_autocast
                origin_enable_pure_fp16 = tracer._enable_pure_fp16
                tracer._enable_autocast = False
                tracer._enable_pure_fp16 = False

                self.user_defined_optimizer.clear_grad()

                tracer._enable_autocast = origin_enable_autocast
                tracer._enable_pure_fp16 = origin_enable_pure_fp16

            scalers[i].clear_grad = types.MethodType(scaler_clear_grad,
                                                     scalers[i])

    if models_is_list:
        if optimizers_is_list:
            return models, scalers
        else:
            return models, scalers[0]
    else:
        if optimizers_is_list:
            return models[0], scalers
        else:
            return models[0], scalers[0]
