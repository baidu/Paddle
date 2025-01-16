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
    add_elementwise_layer,
    replenish_layer_and_output,
    unary_op_converter,
)
from paddle.tensorrt.register import converter_registry

logic_type_map = {
    "pd_op.greater_than": trt.ElementWiseOperation.GREATER,
    "pd_op.less_than": trt.ElementWiseOperation.LESS,
    "pd_op.equal": trt.ElementWiseOperation.EQUAL,
    "pd_op.bitwise_and": trt.ElementWiseOperation.AND,
    "pd_op.bitwise_or": trt.ElementWiseOperation.OR,
    "pd_op.logical_xor": trt.ElementWiseOperation.XOR,
    "pd_op.logical_or": trt.ElementWiseOperation.OR,
    "pd_op.logical_or_": trt.ElementWiseOperation.OR,
    "pd_op.logical_and": trt.ElementWiseOperation.AND,
}


@converter_registry.register("pd_op.greater_than", trt_version="8.x")
@converter_registry.register("pd_op.less_than", trt_version="8.x")
@converter_registry.register("pd_op.equal", trt_version="8.x")
@converter_registry.register("pd_op.bitwise_and", trt_version="8.x")
@converter_registry.register("pd_op.bitwise_or", trt_version="8.x")
@converter_registry.register("pd_op.logical_xor", trt_version="8.x")
@converter_registry.register("pd_op.logical_or", trt_version="8.x")
@converter_registry.register("pd_op.logical_or_", trt_version="8.x")
@converter_registry.register("pd_op.logical_and", trt_version="8.x")
def logic_converter(network, paddle_op, inputs):
    layer_output = add_elementwise_layer(
        network, paddle_op, inputs, logic_type_map[paddle_op.name()]
    )
    return layer_output


@converter_registry.register("pd_op.not_equal", trt_version="8.x")
def not_equal_converter(network, paddle_op, inputs):
    layer_output = add_elementwise_layer(
        network, paddle_op, inputs, trt.ElementWiseOperation.EQUAL
    )
    not_layer = network.add_unary(layer_output, trt.UnaryOperation.NOT)
    layer_output = not_layer.get_output(0)
    replenish_layer_and_output(
        not_layer, paddle_op.name(), paddle_op.get_output_names()
    )
    return layer_output


@converter_registry.register("pd_op.bitwise_not", trt_version="8.x")
def bitwise_not_converter(network, paddle_op, inputs):
    input_tensor = inputs[0]
    if input_tensor.dtype == trt.bool:
        bitwise_not_layer = network.add_unary(
            input_tensor, trt.UnaryOperation.NOT
        )
        layer_output = bitwise_not_layer.get_output(0)
        replenish_layer_and_output(
            bitwise_not_layer, paddle_op.name(), paddle_op.get_output_names()
        )
    else:
        neg_one_tensor_dims = trt.Dims([1] * len(input_tensor.shape))
        neg_one_value = np.array([-1], dtype=np.int32)
        neg_one_weights = trt.Weights(neg_one_value)
        neg_one_tensor = network.add_constant(
            neg_one_tensor_dims, neg_one_weights
        ).get_output(0)
        mul_neg_one = network.add_elementwise(
            input_tensor, neg_one_tensor, trt.ElementWiseOperation.PROD
        ).get_output(0)
        sum_layer = network.add_elementwise(
            mul_neg_one, neg_one_tensor, trt.ElementWiseOperation.SUM
        )
        layer_output = sum_layer.get_output(0)
        replenish_layer_and_output(
            sum_layer, paddle_op.name(), paddle_op.get_output_names()
        )

    return layer_output


@converter_registry.register("pd_op.logical_not", trt_version="8.x")
@converter_registry.register("pd_op.logical_not_", trt_version="8.x")
def logic_not_converter(network, paddle_op, inputs):
    layer_output = unary_op_converter(network, paddle_op, inputs)
    return layer_output
