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
import paddle.fluid.framework as framework
from paddle.fluid.framework import Variable, in_dygraph_mode, OpProtoHolder, Parameter, _dygraph_tracer, dygraph_only, set_flags
import warnings
import copy

__all__ = ['amp_guard']

# The set of ops that support fp16 calculation and are considered numerically-
# safe and performance-critical. These ops are always converted to fp16.
WHITE_LIST = {
    'conv2d',
    'matmul',
    'mul',
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


@signature_safe_contextmanager
@dygraph_only
def amp_guard(enable=True, custom_white_list=None, custom_black_list=None):
    """
    :api_attr: imperative

    Create a context which enables autocast of operators executed in imperative mode.
    If enabled, the input data type (float32 or float16) of each operator is decided 
    by autocast algorithm for better performance. 
    
    Commonly, it is used together with `AmpScaler` to achieve Auto-Mixed-Precision in 
    imperative mode.

    Args:
        enable(bool, optional): Enable loss scaling or not. Default is True.
        custom_white_list(set|list, optional): The custom white_list.
        custom_black_list(set|list, optional): The custom black_list.
        
    Examples:

     .. code-block:: python

        import numpy as np
        import paddle.fluid as fluid

        data = np.random.uniform(-1, 1, [10, 3, 32, 32]).astype('float32')
        with fluid.dygraph.guard():
            conv2d = Conv2D(3, 2, 3)
            data = to_variable(data)
            with fluid.dygraph.amp_guard():
                conv = conv2d(data)
                print(conv.dtype) # FP16
            with fluid.dygraph.amp_guard(enable=False):
                conv = conv2d(data)
                print(conv.dtype) # FP32

    """

    if enable and not core.is_compiled_with_cuda():
        warnings.warn(
            'Auto Mixed Precision can only be enabled with Paddle compiled with CUDA.'
        )
        enable = False
    tracer = _dygraph_tracer()
    if not tracer:
        raise Exception(
            "current_tracer is None, maybe it is not in imperative mode.")

    # use default white_list and black_list if no custom lists provided
    _white_list = WHITE_LIST
    _black_list = BLACK_LIST
    if custom_white_list or custom_black_list:
        _white_list, _black_list = _update_list(custom_white_list,
                                                custom_black_list)

    if tracer:
        # enable auto_cast
        original_enable = tracer._enable_autocast
        tracer._enable_autocast = original_enable
        # set amp op list
        original_white_list, original_black_list = tracer._get_amp_op_list()
        tracer._set_amp_op_list(_white_list, _black_list)
        # set amp related flags
        original_flags = framework.get_flags(AMP_RELATED_FLAGS)
        framework.set_flags(AMP_RELATED_FLAGS_SETTING)

    # restore status
    try:
        yield
    finally:
        if tracer:
            tracer._enable_autocast = original_enable
            tracer._set_amp_op_list(original_white_list, original_black_list)
            framework.set_flags(original_flags)
