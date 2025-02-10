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


import tensorrt as trt

from paddle.tensorrt.converter_utils import (
    add_1D_constant_layer,
    broadcast,
    get_shape_tensor_element,
    trt_shape,
    trt_sum,
)
from paddle.tensorrt.register import converter_registry


@converter_registry.register("pd_op.matmul", trt_version="trt_version_ge=8.0")
def matmul_converter(network, paddle_op, inputs):
    weight_shape = paddle_op.operands()[1].source().shape
    transpose_x = paddle_op.attrs()["transpose_x"]
    transpose_y = paddle_op.attrs()["transpose_y"]
    self_matrix_op = (
        trt.MatrixOperation.TRANSPOSE
        if transpose_x
        else trt.MatrixOperation.NONE
    )
    other_matrix_op = (
        trt.MatrixOperation.TRANSPOSE
        if transpose_y
        else trt.MatrixOperation.NONE
    )

    weight_tensor = inputs[1]
    if type(inputs[1]) == trt.Weights:
        weight_tensor = network.add_constant(
            weight_shape, inputs[1]
        ).get_output(0)

    if len(weight_shape) == 1:
        layer = network.add_shuffle(weight_tensor)
        layer.reshape_dims = (*tuple(weight_shape), 1)
        weight_tensor = layer.get_output(0)

    lhs_val, rhs_val = broadcast(
        network, inputs[0], weight_tensor, inputs[0].name, weight_tensor.name
    )
    out = network.add_matrix_multiply(
        lhs_val, self_matrix_op, rhs_val, other_matrix_op
    )
    return out.get_output(0)


@converter_registry.register(
    "pd_op.transpose", trt_version="trt_version_ge=8.0"
)
def transpose_converter(network, paddle_op, inputs):
    perm = paddle_op.attrs()["perm"]
    transposed_tensor = network.add_shuffle(inputs[0])
    transposed_tensor.second_transpose = perm
    return transposed_tensor.get_output(0)


@converter_registry.register("pd_op.bmm", trt_version="8.x")
def bmm_converter(network, paddle_op, inputs):
    out = network.add_matrix_multiply(
        inputs[0], trt.MatrixOperation.NONE, inputs[1], trt.MatrixOperation.NONE
    )
    return out.get_output(0)


@converter_registry.register("pd_op.flip", trt_version="8.x")
def flip_converter(network, paddle_op, inputs):
    input_tensor = inputs[0]
    input_dims = input_tensor.shape
    rank = len(input_dims)
    axis = paddle_op.attrs()["axis"]
    axis = [a + rank if a < 0 else a for a in axis]
    shape_tensor = trt_shape(network, input_tensor)

    def get_axis_length(axis_idx):
        dim_val = input_dims[axis_idx]
        if dim_val >= 0:
            return add_1D_constant_layer(network, [dim_val], is_scalar=True)
        else:
            return get_shape_tensor_element(
                network, shape_tensor, axis_idx, is_scalar=True
            )

    for axis_idx in axis:
        loop_layer = network.add_loop()
        trip_limit = get_axis_length(axis_idx)
        loop_layer.add_trip_limit(trip_limit, trt.TripLimit.COUNT)
        iterator = loop_layer.add_iterator(input_tensor, axis_idx, reverse=True)
        zero_tensor = add_1D_constant_layer(network, [0])
        one_tensor = add_1D_constant_layer(network, [1])
        iRec_layer = loop_layer.add_recurrence(zero_tensor)
        iCur = iRec_layer.get_output(0)
        iNext_layer = trt_sum(network, iCur, one_tensor)
        iRec_layer.set_input(1, iNext_layer)
        loop_out_layer = loop_layer.add_loop_output(
            iterator.get_output(0), trt.LoopOutput.CONCATENATE, axis_idx
        )
        loop_out_layer.set_input(1, trip_limit)
        input_tensor = loop_out_layer.get_output(0)

    identity_layer = network.add_identity(input_tensor)
    return identity_layer.get_output(0)


@converter_registry.register("pd_op.p_norm", trt_version="8.x")
def p_norm_converter(network, paddle_op, inputs):
    input_tensor = inputs[0]
    input_dims = input_tensor.shape

    axis = paddle_op.attrs().get("axis", -1)
    keepdim = paddle_op.attrs().get("keepdim", False)
    axis = axis if axis >= 0 else axis + len(input_dims)
    axis_mask = 1 << axis

    prod_layer = network.add_elementwise(
        input_tensor, input_tensor, trt.ElementWiseOperation.PROD
    )
    prod_tensor = prod_layer.get_output(0)

    reduce_layer = network.add_reduce(
        prod_tensor, trt.ReduceOperation.SUM, axis_mask, keepdim
    )
    reduced_tensor = reduce_layer.get_output(0)

    sqrt_layer = network.add_unary(reduced_tensor, trt.UnaryOperation.SQRT)
    output_tensor = sqrt_layer.get_output(0)

    return output_tensor
