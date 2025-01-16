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
    append_ones,
    get_axes_for_reduce_op,
    get_dynamic_dims,
    get_trt_plugin,
    has_dynamic_shape,
    replenish_layer_and_output,
)
from paddle.tensorrt.register import converter_registry


@converter_registry.register(
    "pd_op.layer_norm", trt_version="trt_version_ge=8.6"
)
def layernorm_converter(network, paddle_op, inputs):
    input_a, scale, bias = inputs

    begin_norm_axis = paddle_op.attrs().get("begin_norm_axis", 0)
    epsilon = paddle_op.attrs().get("epsilon", 1e-5)
    assert len(paddle_op.operands()) == 3
    scale_shape = paddle_op.operands()[1].source().shape

    scale_tensor = network.add_constant(scale_shape, scale).get_output(0)
    bias_shape = paddle_op.operands()[2].source().shape
    bias_tensor = network.add_constant(bias_shape, bias).get_output(0)

    # dims = list(range( len(input_a.shape) - len(normalized_shape), len(input_a.shape)))
    dims = list(range(len(input_a.shape)))[begin_norm_axis:]
    axes = get_axes_for_reduce_op(dims)

    scale_tensor = append_ones(
        network,
        scale_tensor,
        f"{scale_tensor.name}_broadcast",
        len(input_a.shape) - len(scale_tensor.shape),
    )

    bias_tensor = append_ones(
        network,
        bias_tensor,
        f"{bias_tensor.name}_broadcast",
        len(input_a.shape) - len(bias_tensor.shape),
    )

    layer_norm = network.add_normalization(
        input_a, scale_tensor, bias_tensor, axes
    )
    layer_norm.epsilon = epsilon
    layer_norm.compute_precision = trt.float32

    replenish_layer_and_output(
        layer_norm, paddle_op.name(), paddle_op.get_output_names()
    )
    return layer_norm.get_output(0)


@converter_registry.register(
    "pd_op.batch_norm", trt_version="trt_version_ge=8.0"
)
@converter_registry.register(
    "pd_op.batch_norm_", trt_version="trt_version_ge=8.0"
)
def batch_norm_converter(network, paddle_op, inputs):
    input_tensor, mean, variance, scale, bias = inputs
    scale_shape = paddle_op.operands()[3].source().shape
    eps = paddle_op.attrs().get("epsilon", 1e-8)
    mean_np = mean.numpy()
    variance_np = variance.numpy()
    scale_np = scale.numpy()
    bias_np = bias.numpy()

    actual_scale_np = scale_np / np.sqrt(variance_np + eps)
    actual_bias_np = bias_np - mean_np * actual_scale_np
    bias = trt.Weights(actual_bias_np)
    scale = trt.Weights(actual_scale_np)
    power = np.ones(scale_shape, dtype='float32')
    power = trt.Weights(power)
    input_tensor_shape = paddle_op.operands()[0].source().shape
    if has_dynamic_shape(input_tensor_shape):
        assert (
            input_tensor.shape[1] != -1
        ), "Channel dim can't be dynamic for batch norm."
    # For BatchNorm1d ,reshape 1d to 2d
    output_shape = input_tensor_shape

    if not network.has_implicit_batch_dimension and len(input_tensor_shape) < 4:
        assert (
            len(get_dynamic_dims(input_tensor.shape)) <= 1
        ), "BatchNorm1D with more than one dynamic dims is not currently supported."
        reshape_layer = network.add_shuffle(input_tensor)
        if len(input_tensor_shape) == 2:
            reshape_layer.reshape_dims = (
                input_tensor_shape[0],
                input_tensor_shape[1],
                1,
                1,
            )
        else:  # len(input_tensor_shape) ==3
            reshape_layer.reshape_dims = (
                input_tensor_shape[0],
                input_tensor_shape[1],
                input_tensor_shape[2],
                1,
            )
        input_tensor = reshape_layer.get_output(0)
    # (self: tensorrt.tensorrt.INetworkDefinition, input: tensorrt.tensorrt.ITensor, mode: tensorrt.tensorrt.ScaleMode, shift: tensorrt.tensorrt.Weights = None, scale: tensorrt.tensorrt.Weights = None, power: tensorrt.tensorrt.Weights = None) -> tensorrt.tensorrt.IScaleLayer
    batch_norm_layer = network.add_scale(
        input_tensor, trt.ScaleMode.CHANNEL, bias, scale, power
    )
    # For BatchNorm1d,reshape output back to 1d
    if not network.has_implicit_batch_dimension and len(output_shape) < 4:
        reshape_output_layer = network.add_shuffle(
            batch_norm_layer.get_output(0)
        )
        reshape_output_layer.reshape_dims = tuple(output_shape)
        batch_norm_layer = reshape_output_layer

    replenish_layer_and_output(
        batch_norm_layer, paddle_op.name(), paddle_op.get_output_names()
    )
    return batch_norm_layer.get_output(0)


@converter_registry.register(
    "pd_op.instance_norm", trt_version="trt_version_ge=8.0"
)
def instance_norm_converter(network, paddle_op, inputs):
    eps = paddle_op.attrs().get("epsilon", 1e-8)
    instance_norm_inputs = [inputs[0], inputs[1], inputs[2]]
    plugin_fields = [
        trt.PluginField(
            "epsilon",
            np.array(eps, dtype=np.float32),
            trt.PluginFieldType.FLOAT32,
        ),
    ]
    plugin_field_collection = trt.PluginFieldCollection(plugin_fields)
    plugin_name = "pir_instance_norm"
    plugin_version = "1"
    plugin = get_trt_plugin(
        plugin_name, plugin_field_collection, plugin_version
    )
    instance_norm_layer = network.add_plugin_v2(instance_norm_inputs, plugin)
    replenish_layer_and_output(
        instance_norm_layer, paddle_op.name(), paddle_op.get_output_names()
    )
    return instance_norm_layer.get_output(0)
