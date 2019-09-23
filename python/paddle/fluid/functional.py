# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import sys

from .layer_helper import LayerHelper
from .layers.utils import *
from . import core
from .framework import Variable
import numpy as np
import os
import six
import multiprocessing
from six.moves import zip, range, xrange
import warnings
from . import compiler
from .. import compat as cpt
from .trainer_factory import TrainerFactory

__all__ = ['conv2d']


def conv2d(input,
           filter,
           stride=1,
           padding=0,
           dilation=1,
           groups=None,
           use_cudnn=True,
           name=None):
    """
    Similar with conv2d, this is a convolution2D layers. Difference
    is filter can be token as input directly instead of setting filter size
    and number of fliters. Filter is a  4-D tensor with shape
    [out_channels(num_filters), in_channels, filter_size_h, filter_size_w].
     Args:
        input (Variable): The input image with [N, C, H, W] format.
        filter(Variable): The input filter with [out_channels, in_channels, H, W] format.
        stride (int|tuple): The stride size. If stride is a tuple, it must
            contain two integers, (stride_H, stride_W). Otherwise, the
            stride_H = stride_W = stride. Default: stride = 1.
        padding (int|tuple): The padding size. If padding is a tuple, it must
            contain two integers, (padding_H, padding_W). Otherwise, the
            padding_H = padding_W = padding. Default: padding = 0.
        dilation (int|tuple): The dilation size. If dilation is a tuple, it must
            contain two integers, (dilation_H, dilation_W). Otherwise, the
            dilation_H = dilation_W = dilation. Default: dilation = 1.
        use_cudnn (bool): Use cudnn kernel or not, it is valid only when the cudnn
            library is installed. Default: True
        name (str|None): A name for this layer(optional). If set None, the layer
            will be named automatically. Default: None
    Returns:
        Variable: The tensor variable storing the convolution and \
                  non-linearity activation result.
    Raises:
        ValueError: If the shapes of input, filter_size, stride, padding and
                    groups mismatch.
    Examples:
        .. code-block:: python
          data = fluid.layers.data(name='data', shape=[3, 32, 32], \
                                  dtype='float32')
          filter = fluid.layers.data(name='filter',shape=[10,3,3,3], \
                                    dtype='float32',append_batch_size=False)
          conv2d = fluid.functional.conv2d(input=data,
                                       filter=filter)

        .. code-block:: python for dygraph paddle
          data = np.random.random((2, 3, 4, 4)).astype(np.float32)
          filter = np.random.random((8, 3, 3, 3)).astype(np.float32)
          with fluid.dygraph.guard():
              data = fluid.dygraph.to_variable(data)
              filter = fluid.dygraph.to_variable(filter)
              res = fluid.functional.conv2d(data, filter)
    """
    helper = LayerHelper("conv2d_with_filter", **locals())
    num_channels = input.shape[1]
    num_filters = filter.shape[0]
    num_filter_channels = filter.shape[1]
    l_type = 'conv2d'

    if (num_channels == groups and num_filters % num_channels == 0 and
            not use_cudnn):
        l_type = 'depthwise_conv2d'

    if groups is None:
        groups = 1
        assert num_filter_channels == num_channels
    else:
        #num_channels must be divisible by groups.
        assert num_channels % groups == 0
        # num_filter_channels must equal to num_channels divided by groups.
        assert num_channels // groups == num_filter_channels

    stride = convert_to_list(stride, 2, 'stride')
    padding = convert_to_list(padding, 2, 'padding')
    dilation = convert_to_list(dilation, 2, 'dilation')

    if not isinstance(use_cudnn, bool):
        raise ValueError("use_cudnn should be True or False")

    pre_bias = helper.create_variable_for_type_inference(dtype=input.dtype)

    helper.append_op(
        type=l_type,
        inputs={
            'Input': input,
            'Filter': filter,
        },
        outputs={"Output": pre_bias},
        attrs={
            'strides': stride,
            'paddings': padding,
            'dilations': dilation,
            'groups': groups,
            'use_cudnn': use_cudnn,
            'use_mkldnn': False
        })
    return pre_bias
