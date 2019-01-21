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

from __future__ import print_function

from six.moves import reduce

from .. import core
from ..layers import utils
from . import layers
from ..framework import Variable, OpProtoHolder
from ..param_attr import ParamAttr
from ..initializer import Normal, Constant

__all__ = [
    'Conv2D',
    'Pool2D',
    'FC',
    'BatchNorm',
]


class Conv2D(layers.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=None,
                 use_cudnn=True,
                 act=None,
                 param_attr=None,
                 bias_attr=None,
                 name=None,
                 dtype=core.VarDesc.VarType.FP32):
        assert param_attr is not False, "param_attr should not be False here."
        super(Conv2D, self).__init__(name=name, dtype=dtype)

        from ..layer_helper import LayerHelper
        self._helper = LayerHelper(
            type(self).__name__,
            param_attr=param_attr,
            bias_attr=bias_attr,
            dtype=dtype,
            name=name,
            act=act)

        self._groups = groups
        self._stride = utils.convert_to_list(stride, 2, 'stride')
        self._padding = utils.convert_to_list(padding, 2, 'padding')
        self._dilation = utils.convert_to_list(dilation, 2, 'dilation')
        if not isinstance(use_cudnn, bool):
            raise ValueError("use_cudnn should be True or False")
        self._use_cudnn = use_cudnn
        self._num_channels = num_channels
        if (self._num_channels == self._groups and
                num_filters % self._num_channels == 0 and not self._use_cudnn):
            self._l_type = 'depthwise_conv2d'
        else:
            self._l_type = 'conv2d'

        if groups is None:
            num_filter_channels = num_channels
        else:
            if num_channels % groups != 0:
                raise ValueError("num_channels must be divisible by groups.")
            num_filter_channels = num_channels // groups
        filter_size = utils.convert_to_list(filter_size, 2, 'filter_size')
        filter_shape = [num_filters, int(num_filter_channels)] + filter_size

        def _get_default_param_initializer():
            filter_elem_num = filter_size[0] * filter_size[1] * num_channels
            std = (2.0 / filter_elem_num)**0.5
            return Normal(0.0, std, 0)

        self._filter_param = self._helper.create_parameter(
            attr=self._helper.param_attr,
            shape=filter_shape,
            dtype=self._dtype,
            default_initializer=_get_default_param_initializer())

        if self._use_cudnn:
            self._helper.create_variable(
                name="kCUDNNFwdAlgoCache",
                persistable=True,
                type=core.VarDesc.VarType.RAW)
            self._helper.create_variable(
                name="kCUDNNBwdDataAlgoCache",
                persistable=True,
                type=core.VarDesc.VarType.RAW)
            self._helper.create_variable(
                name="kCUDNNBwdFilterAlgoCache",
                persistable=True,
                type=core.VarDesc.VarType.RAW)

        self._bias_param = self._helper.create_parameter(
            attr=self._helper.bias_attr,
            shape=[num_filters],
            dtype=self._dtype,
            is_bias=True)

    def forward(self, input):
        pre_bias = self._helper.create_variable_for_type_inference(
            dtype=self._dtype)

        self._helper.append_op(
            type=self._l_type,
            inputs={
                'Input': input,
                'Filter': self._filter_param,
            },
            outputs={"Output": pre_bias},
            attrs={
                'strides': self._stride,
                'paddings': self._padding,
                'dilations': self._dilation,
                'groups': self._groups,
                'use_cudnn': self._use_cudnn,
                'use_mkldnn': False,
            })

        pre_act = self._helper.create_variable_for_type_inference(
            dtype=self._dtype)

        self._helper.append_op(
            type='elementwise_add',
            inputs={'X': [pre_bias],
                    'Y': [self._bias_param]},
            outputs={'Out': [pre_act]},
            attrs={'axis': 1})

        # Currently, we don't support inplace in imperative mode
        return self._helper.append_activation(pre_act)


class Pool2D(layers.Layer):
    def __init__(self,
                 pool_size=-1,
                 pool_type="max",
                 pool_stride=1,
                 pool_padding=0,
                 global_pooling=False,
                 use_cudnn=True,
                 ceil_mode=False,
                 exclusive=True,
                 name=None,
                 dtype=core.VarDesc.VarType.FP32):
        if pool_type not in ["max", "avg"]:
            raise ValueError(
                "Unknown pool_type: '%s'. It can only be 'max' or 'avg'.",
                str(pool_type))

        if global_pooling is False and pool_size == -1:
            raise ValueError(
                "When the global_pooling is False, pool_size must be passed "
                "and be a valid value. Received pool_size: " + str(pool_size))

        if not isinstance(use_cudnn, bool):
            raise ValueError("use_cudnn should be True or False")

        super(Pool2D, self).__init__(name=name, dtype=dtype)

        from ..layer_helper import LayerHelper
        self._helper = LayerHelper(type(self).__name__, dtype=dtype, name=name)

        self._pool_type = pool_type
        self._pool_size = utils.convert_to_list(pool_size, 2, 'pool_size')
        self._pool_padding = utils.convert_to_list(pool_padding, 2,
                                                   'pool_padding')
        self._pool_stride = utils.convert_to_list(pool_stride, 2, 'pool_stride')
        self._global_pooling = global_pooling
        self._use_cudnn = use_cudnn
        self._ceil_mode = ceil_mode
        self._exclusive = exclusive
        self._l_type = 'pool2d'

    def forward(self, input):
        pool_out = self._helper.create_variable_for_type_inference(self._dtype)

        self._helper.append_op(
            type=self._l_type,
            inputs={"X": input},
            outputs={"Out": pool_out},
            attrs={
                "pooling_type": self._pool_type,
                "ksize": self._pool_size,
                "global_pooling": self._global_pooling,
                "strides": self._pool_stride,
                "paddings": self._pool_padding,
                "use_cudnn": self._use_cudnn,
                "ceil_mode": self._ceil_mode,
                "use_mkldnn": False,
                "exclusive": self._exclusive,
            })
        return pool_out


class FC(layers.Layer):
    def __init__(self,
                 size,
                 param_attr=None,
                 bias_attr=None,
                 dtype=core.VarDesc.VarType.FP32,
                 num_flatten_dims=1,
                 act=None,
                 is_test=False,
                 name=None):
        super(FC, self).__init__()

        self._size = size
        self._num_flatten_dims = num_flatten_dims
        self._dtype = dtype
        from ..layer_helper import LayerHelper
        self._helper = LayerHelper(
            'FC',
            param_attr=param_attr,
            bias_attr=bias_attr,
            act=act,
            name=name)

    def _build_once(self, input):
        input_shape = input.shape
        param_shape = [
            reduce(lambda a, b: a * b, input_shape[self._num_flatten_dims:], 1)
        ] + [self._size]
        self._w = self._helper.create_parameter(
            attr=self._helper.param_attr,
            shape=param_shape,
            dtype=self._dtype,
            is_bias=False)

        if self._helper.bias_attr:
            size = list([self._size])
            self._b = self._helper.create_parameter(
                attr=self._helper.bias_attr,
                shape=size,
                dtype=self._dtype,
                is_bias=True)
        else:
            self._b = None

    def forward(self, input):
        tmp = self._helper.create_variable_for_type_inference(self._dtype)
        self._helper.append_op(
            type="mul",
            inputs={"X": input,
                    "Y": self._w},
            outputs={"Out": tmp},
            attrs={
                "x_num_col_dims": self._num_flatten_dims,
                "y_num_col_dims": 1
            })

        pre_bias = self._helper.create_variable_for_type_inference(self._dtype)
        self._helper.append_op(
            type="sum",
            inputs={"X": [tmp]},
            outputs={"Out": pre_bias},
            attrs={"use_mkldnn": False})

        if self._b:
            pre_activation = self._helper.create_variable_for_type_inference(
                dtype=self._dtype)
            self._helper.append_op(
                type='elementwise_add',
                inputs={'X': [pre_bias],
                        'Y': [self._b]},
                outputs={'Out': [pre_activation]},
                attrs={'axis': self._num_flatten_dims})
        else:
            pre_activation = pre_bias
        # Currently, we don't support inplace in imperative mode
        return self._helper.append_activation(pre_activation)


class BatchNorm(layers.Layer):
    def __init__(self,
                 num_channels,
                 act=None,
                 is_test=False,
                 momentum=0.9,
                 epsilon=1e-05,
                 param_attr=None,
                 bias_attr=None,
                 dtype=core.VarDesc.VarType.FP32,
                 data_layout='NCHW',
                 in_place=False,
                 name=None,
                 moving_mean_name=None,
                 moving_variance_name=None,
                 do_model_average_for_mean_and_var=False,
                 fuse_with_relu=False,
                 use_global_stats=False):
        super(BatchNorm, self).__init__()

        assert bias_attr is not False, "bias_attr should not be False in batch_norm."

        from ..layer_helper import LayerHelper
        self._helper = LayerHelper(
            'batch_norm',
            param_attr=param_attr,
            bias_attr=bias_attr,
            name=name,
            act=act)

        if dtype == core.VarDesc.VarType.FP16:
            self._dtype = core.VarDesc.VarType.FP32
        else:
            self._dtype = dtype

        param_shape = [num_channels]

        # create parameter
        self._scale = self._helper.create_parameter(
            attr=self._helper.param_attr,
            shape=param_shape,
            dtype=self._dtype,
            default_initializer=Constant(1.0))

        # TODO(minqiyang): change stop_gradient sign to trainable to align with static graph
        #  # setting stop_gradient=True to reduce computation
        #  if use_global_stats and self._helper.param_attr.learning_rate == 0.:
        #  self._scale.stop_gradient = True

        self._bias = self._helper.create_parameter(
            attr=self._helper.bias_attr,
            shape=param_shape,
            dtype=self._dtype,
            is_bias=True)
        # TODO(minqiyang): change stop_gradient sign to trainable to align with static graph
        #  # setting stop_gradient=True to reduce computation
        #  if use_global_stats and self._helper.bias_attr.learning_rate == 0.:
        #  self._bias.stop_gradient = True

        self._mean = self._helper.create_parameter(
            attr=ParamAttr(
                name=moving_mean_name,
                initializer=Constant(0.0),
                trainable=False,
                do_model_average=do_model_average_for_mean_and_var),
            shape=param_shape,
            dtype=self._dtype)
        self._mean.stop_gradient = True

        self._variance = self._helper.create_parameter(
            attr=ParamAttr(
                name=moving_variance_name,
                initializer=Constant(1.0),
                trainable=False,
                do_model_average=do_model_average_for_mean_and_var),
            shape=param_shape,
            dtype=self._dtype)
        self._variance.stop_gradient = True

        self._in_place = in_place
        self._momentum = momentum
        self._epsilon = epsilon
        self._is_test = is_test
        self._fuse_with_relu = fuse_with_relu
        self._use_global_stats = use_global_stats

    def _build_once(self, input):
        pass

    def forward(self, input):
        # create output
        # mean and mean_out share the same memory
        mean_out = self._mean
        # variance and variance out share the same memory
        variance_out = self._variance

        saved_mean = self._helper.create_variable_for_type_inference(
            dtype=self._dtype, stop_gradient=True)
        saved_variance = self._helper.create_variable_for_type_inference(
            dtype=self._dtype, stop_gradient=True)
        batch_norm_out = input if self._in_place else self._helper.create_variable_for_type_inference(
            self._dtype)

        self._helper.append_op(
            type="batch_norm",
            inputs={
                "X": input,
                "Scale": self._scale,
                "Bias": self._bias,
                "Mean": self._mean,
                "Variance": self._variance
            },
            outputs={
                "Y": batch_norm_out,
                "MeanOut": mean_out,
                "VarianceOut": variance_out,
                "SavedMean": saved_mean,
                "SavedVariance": saved_variance
            },
            attrs={
                "momentum": self._momentum,
                "epsilon": self._epsilon,
                "is_test": self._is_test,
                "use_mkldnn": False,
                "fuse_with_relu": self._fuse_with_relu,
                "use_global_stats": self._use_global_stats
            })

        # Currently, we don't support inplace in imperative mode
        return self._helper.append_activation(batch_norm_out)
