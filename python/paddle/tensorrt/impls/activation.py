# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved
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

import numpy as np
import tensorrt as trt

from paddle.tensorrt.converter_utils import (
    add_constant_layer,
    trt_div,
    trt_min,
    trt_pow,
    trt_prod,
    trt_sub,
    trt_sum,
)
from paddle.tensorrt.register import converter_registry

activation_type_map = {
    "pd_op.tanh": trt.ActivationType.TANH,
    "pd_op.relu": trt.ActivationType.RELU,
    "pd_op.sigmoid": trt.ActivationType.SIGMOID,
    "pd_op.silu": trt.ActivationType.SIGMOID,
    "pd_op.swish": trt.ActivationType.SIGMOID,
}


@converter_registry.register("pd_op.relu", trt_version="trt_version_ge=8.0")
@converter_registry.register("pd_op.tanh", trt_version="trt_version_ge=8.0")
@converter_registry.register("pd_op.sigmoid", trt_version="trt_version_ge=8.0")
def activation_converter(network, paddle_op, inputs):
    layer = network.add_activation(
        inputs[0], activation_type_map[paddle_op.name()]
    )
    return layer.get_output(0)


@converter_registry.register(
    "pd_op.logsigmoid", trt_version="trt_version_ge=8.0"
)
def logsigmoid_converter(network, paddle_op, inputs):
    sigmoid_layer = network.add_activation(
        inputs[0], trt.ActivationType.SIGMOID
    )
    layer = network.add_unary(
        sigmoid_layer.get_output(0), trt.UnaryOperation.LOG
    )
    return layer.get_output(0)


@converter_registry.register("pd_op.relu6", trt_version="trt_version_ge=8.0")
def relu6_converter(network, paddle_op, inputs):
    layer = network.add_activation(inputs[0], trt.ActivationType.CLIP)
    layer.alpha = 0.0
    layer.beta = 6.0
    return layer.get_output(0)


@converter_registry.register("pd_op.softmax", trt_version="trt_version_ge=8.0")
def softmax_converter(network, paddle_op, inputs):
    input1 = inputs[0]
    input_shape = input1.shape
    input_dims = len(input_shape)
    axis = paddle_op.attrs().get("axis", -1)

    # support 0 or 1 dims input
    is_0_dims = input_dims == 0
    is_1_dims = input_dims == 1
    if is_0_dims or is_1_dims:
        reshaped_layer = network.add_shuffle(input1)
        reshaped_dims = (1, 1 if is_0_dims else input_shape[0])
        reshaped_layer.reshape_dims = reshaped_dims
        input1 = reshaped_layer.get_output(0)
        input_shape = input1.shape
        input_dims = len(input_shape)
        axis = -1

    layer = network.add_softmax(input1)
    axes = max(0, input_dims - 3)

    # Handle padded dimensions
    padded_dims = 0
    explicit_batch = 1
    for i in range(input_dims - 1, explicit_batch, -1):
        if input_shape[i] == 1:
            padded_dims += 1
        else:
            break

    if axis < 0:
        axes = input_dims + axis
    else:
        axes = axis

    layer.axes = 1 << axes

    # Support 0 or 1 dims input
    if is_0_dims or is_1_dims:
        reshaped_layer = network.add_shuffle(layer.get_output(0))
        reshaped_layer.reshape_dims = inputs[0].shape
        layer = reshaped_layer

    return layer.get_output(0)


@converter_registry.register("pd_op.gelu", trt_version="trt_version_ge=8.0")
def gelu_converter(network, paddle_op, inputs):
    input_val = inputs[0]
    approximate = paddle_op.attrs()["approximate"]

    const_shape = [1] * len(input_val.shape)

    def create_constant(network, const_shape, value, dtype=np.float32):
        arr = np.array([value], dtype=dtype)
        const_layer = network.add_constant(const_shape, arr)
        return const_layer.get_output(0)

    if approximate:
        constant_layer_pow = create_constant(network, const_shape, 3.0)
        constant_layer_multiply = create_constant(
            network, const_shape, 0.044715
        )
        constant_layer_sqrt = create_constant(
            network, const_shape, 0.7978845608028654
        )
        constant_layer_one = create_constant(network, const_shape, 1.0)
        constant_layer_half = create_constant(network, const_shape, 0.5)

        layer_pow = trt_pow(network, input_val, constant_layer_pow)
        layer_mul = trt_prod(network, layer_pow, constant_layer_multiply)
        layer_add = trt_sum(network, layer_mul, input_val)
        layer_sqrt = trt_prod(network, layer_add, constant_layer_sqrt)

        layer_tanh = network.add_activation(
            layer_sqrt, trt.ActivationType.TANH
        ).get_output(0)
        layer_one = trt_sum(network, layer_tanh, constant_layer_one)
        layer_cdf = trt_prod(network, layer_one, constant_layer_half)
        y = trt_prod(network, layer_cdf, input_val)

        return y
    else:
        constant_layer_one = create_constant(network, const_shape, 1.0)
        constant_layer_half = create_constant(network, const_shape, 0.5)
        constant_layer_rsqrt2 = create_constant(
            network, const_shape, 0.70710678118
        )

        layer_mul = trt_prod(network, input_val, constant_layer_rsqrt2)
        layer_erf = network.add_unary(
            layer_mul, trt.UnaryOperation.ERF
        ).get_output(0)
        layer_add = trt_sum(network, layer_erf, constant_layer_one)
        layer_cdf = trt_prod(network, layer_add, constant_layer_half)
        y = trt_prod(network, layer_cdf, input_val)

        return y


@converter_registry.register(
    "pd_op.hardsigmoid", trt_version="trt_version_ge=8.0"
)
def hardsigmoid_converter(network, paddle_op, inputs):
    x = inputs[0]
    slope = paddle_op.attrs()["slope"]
    offset = paddle_op.attrs()["offset"]
    hardsigmoid_layer = network.add_activation(
        x, trt.ActivationType.HARD_SIGMOID
    )
    hardsigmoid_layer.alpha = slope
    hardsigmoid_layer.beta = offset
    return hardsigmoid_layer.get_output(0)


@converter_registry.register(
    "pd_op.hardswish", trt_version="trt_version_ge=8.0"
)
def hardswish_converter(network, paddle_op, inputs):
    x = inputs[0]
    threshold = 6.0
    scale = 6.0
    offset = 3.0
    hardsigmoid_layer = network.add_activation(
        x, trt.ActivationType.HARD_SIGMOID
    )
    hardsigmoid_layer.alpha = 1.0 / scale
    hardsigmoid_layer.beta = offset / scale
    hardswish_layer = network.add_elementwise(
        x, hardsigmoid_layer.get_output(0), trt.ElementWiseOperation.PROD
    )
    return hardswish_layer.get_output(0)


@converter_registry.register("pd_op.elu", trt_version="8.x")
@converter_registry.register("pd_op.elu_", trt_version="8.x")
def elu_converter(network, paddle_op, inputs):
    x = inputs[0]
    alpha = paddle_op.attrs()["alpha"]
    elu_layer = network.add_activation(x, trt.ActivationType.ELU)
    elu_layer.alpha = alpha
    return elu_layer.get_output(0)


@converter_registry.register("pd_op.softplus", trt_version="8.x")
def softplus_converter(network, paddle_op, inputs):
    x = inputs[0]
    beta = paddle_op.attrs()["beta"]
    threshold = paddle_op.attrs()["threshold"]
    layer_clip = network.add_activation(x, trt.ActivationType.CLIP)
    layer_clip.alpha = -3.40282e038
    layer_clip.beta = threshold / beta

    softplus_layer = network.add_activation(
        layer_clip.get_output(0), trt.ActivationType.SOFTPLUS
    )
    softplus_layer.alpha = 1.0 / beta
    softplus_layer.beta = beta
    return softplus_layer.get_output(0)


@converter_registry.register("pd_op.swish", trt_version="8.x")
@converter_registry.register("pd_op.silu", trt_version="8.x")
def swish_silu_converter(network, paddle_op, inputs):
    layer_output = network.add_activation(
        inputs[0], activation_type_map[paddle_op.name()]
    ).get_output(0)
    return trt_prod(network, inputs[0], layer_output)


@converter_registry.register("pd_op.tanh_shrink", trt_version="8.x")
def tanh_shrink_converter(network, paddle_op, inputs):
    x = inputs[0]
    tanh_layer = network.add_activation(x, trt.ActivationType.TANH)
    subtract_layer = network.add_elementwise(
        x, tanh_layer.get_output(0), trt.ElementWiseOperation.SUB
    )
    return subtract_layer.get_output(0)


@converter_registry.register("pd_op.stanh", trt_version="8.x")
def stanh_converter(network, paddle_op, inputs):
    x = inputs[0]
    scale_a = paddle_op.attrs()["scale_a"]
    scale_b = paddle_op.attrs()["scale_b"]
    stanh_layer = network.add_activation(x, trt.ActivationType.SCALED_TANH)
    stanh_layer.alpha = scale_b
    stanh_layer.beta = scale_a
    return stanh_layer.get_output(0)


@converter_registry.register("pd_op.mish", trt_version="8.x")
def mish_converter(network, paddle_op, inputs):
    x = inputs[0]
    softplus_layer = network.add_activation(x, trt.ActivationType.SOFTPLUS)
    softplus_output = softplus_layer.get_output(0)

    tanh_layer = network.add_activation(
        softplus_output, trt.ActivationType.TANH
    )
    tanh_output = tanh_layer.get_output(0)

    return trt_prod(network, x, tanh_output)


@converter_registry.register("pd_op.celu", trt_version="8.x")
def celu_converter(network, paddle_op, inputs):
    input_tensor = inputs[0]
    alpha = paddle_op.attrs()["alpha"]
    input_rank = len(input_tensor.shape)
    constant_shape = trt.Dims([1] * input_rank)
    alpha_data = add_constant_layer(
        network, [alpha], constant_shape, dtype="float32"
    )
    constant_zero_data = add_constant_layer(
        network, [0.0], constant_shape, dtype="float32"
    )
    constant_one_data = add_constant_layer(
        network, [1.0], constant_shape, dtype="float32"
    )
    input_div_with_alpha = trt_div(network, input_tensor, alpha_data)
    input_exp_layer = network.add_unary(
        input_div_with_alpha, trt.UnaryOperation.EXP
    )
    input_sub_with_one = trt_sub(
        network, input_exp_layer.get_output(0), constant_one_data
    )
    input_prod_with_alpha = trt_prod(network, input_sub_with_one, alpha_data)
    min_input = trt_min(network, input_prod_with_alpha, constant_zero_data)
    relu_layer = network.add_activation(input_tensor, trt.ActivationType.RELU)
    output_tensor = trt_sum(network, relu_layer.get_output(0), min_input)
    return output_tensor


@converter_registry.register("pd_op.thresholded_relu", trt_version="8.x")
def thresholded_relu_converter(network, paddle_op, inputs):
    x = inputs[0]
    threshold = paddle_op.attrs()["threshold"]
    thresholded_relu_layer = network.add_activation(
        x, trt.ActivationType.THRESHOLDED_RELU
    )
    thresholded_relu_layer.alpha = threshold
    return thresholded_relu_layer.get_output(0)


@converter_registry.register("pd_op.selu", trt_version="8.x")
def selu_converter(network, paddle_op, inputs):
    x = inputs[0]
    alpha = paddle_op.attrs()["alpha"]
    scale = paddle_op.attrs()["scale"]
    selu_layer = network.add_activation(x, trt.ActivationType.SELU)
    selu_layer.alpha = alpha
    selu_layer.beta = scale
    return selu_layer.get_output(0)
