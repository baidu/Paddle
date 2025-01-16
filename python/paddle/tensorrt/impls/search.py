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
    get_input_constant_value,
    get_shape_tensor_element,
    replenish_layer_and_output,
    squeeze_trt,
    trt_cast,
    trt_gather,
    trt_reshape,
    trt_shape,
    trt_unsqueeze,
)
from paddle.tensorrt.register import converter_registry


@converter_registry.register("pd_op.nonzero", trt_version="8.x")
def non_zero_converter(network, paddle_op, inputs):
    input_tensor = inputs[0]
    cast_layer = network.add_cast(input_tensor, trt.float32)
    non_zero_layer = network.add_non_zero(cast_layer.get_output(0))

    replenish_layer_and_output(
        non_zero_layer, paddle_op.name(), paddle_op.get_output_names()
    )
    return non_zero_layer.get_output(0)


@converter_registry.register("pd_op.argmax", trt_version="trt_version_ge=8.0")
def argmax_converter(network, paddle_op, inputs):
    x = inputs[0]
    input_dims = x.shape
    rank = len(input_dims)
    axis = int(get_input_constant_value(paddle_op, inputs, 1)[0])
    keepdims = paddle_op.attrs()["keepdims"]

    if axis < 0:
        axis += rank

    topk_layer = network.add_topk(
        input=x, op=trt.TopKOperation.MAX, k=1, axes=(1 << axis)
    )

    if keepdims:
        replenish_layer_and_output(
            topk_layer, paddle_op.name(), paddle_op.get_output_names()[1]
        )
        return topk_layer.get_output(1)
    else:
        topk_out = topk_layer.get_output(1)
        topk_out_shape_size = len(topk_out.shape)
        # Mark which dimensions to squeeze
        should_squeeze = [False] * topk_out_shape_size
        should_squeeze[axis] = True

        # Get dimensions to keep
        gather_indices = [
            i for i, squeeze in enumerate(should_squeeze) if not squeeze
        ]

        # Add Shuffle layer
        layer = network.add_shuffle(topk_out)
        shape_tensor = trt_shape(network, topk_out)
        real_shape_tensor = trt_gather(network, shape_tensor, gather_indices)
        layer.set_input(1, real_shape_tensor)
        replenish_layer_and_output(
            layer, paddle_op.name(), paddle_op.get_output_names()
        )
        return layer.get_output(0)


@converter_registry.register("pd_op.argmin", trt_version="8.x")
def argmin_converter(network, paddle_op, inputs):
    x = inputs[0]
    input_dims = x.shape
    rank = len(input_dims)
    axis = int(get_input_constant_value(paddle_op, inputs, 1)[0])
    keepdims = paddle_op.attrs()["keepdims"]

    if axis < 0:
        axis += rank

    topk_layer = network.add_topk(
        input=x, op=trt.TopKOperation.MIN, k=1, axes=(1 << axis)
    )

    if keepdims:
        replenish_layer_and_output(
            topk_layer, paddle_op.name(), paddle_op.get_output_names()
        )
        return topk_layer.get_output(1)
    else:
        squeeze_layer = network.add_shuffle(topk_layer.get_output(1))
        output_dims = []
        for i in range(len(input_dims)):
            if i == axis:
                continue
            output_dims.append(input_dims[i])
        squeeze_layer.reshape_dims = tuple(output_dims)
        replenish_layer_and_output(
            squeeze_layer, paddle_op.name(), paddle_op.get_output_names()
        )
        return squeeze_layer.get_output(0)


@converter_registry.register("pd_op.argsort", trt_version="8.x")
def argsort_converter(network, paddle_op, inputs):
    input_tensor = inputs[0]
    input_shape = input_tensor.shape
    in_type = input_tensor.dtype
    in_rank = len(input_shape)
    axis = paddle_op.attrs()["axis"]
    descending = paddle_op.attrs()["descending"]
    if axis < 0:
        axis += len(input_shape)
    topk_op = trt.TopKOperation.MAX if descending else trt.TopKOperation.MIN
    need_cast = True if in_type != trt.DataType.FLOAT else False
    if in_rank == 1:
        unsqueeze_shape = trt.Dims([1, -1])
        input_tensor = trt_reshape(
            network, input_tensor, unsqueeze_shape, is_shape_tensor=False
        )
        axis = 1
    if need_cast:
        input_tensor = trt_cast(network, input_tensor, trt.DataType.FLOAT)
    topk_layer = network.add_topk(input_tensor, topk_op, 1, 1 << axis)
    shape = trt_shape(network, input_tensor)
    k_tensor = get_shape_tensor_element(network, shape, axis, True)
    topk_layer.set_input(1, k_tensor)
    out = topk_layer.get_output(0)
    indices = topk_layer.get_output(1)
    if in_rank == 1:
        squeeze_shape = trt.Dims([-1])
        out = trt_reshape(network, out, squeeze_shape, is_shape_tensor=False)
        indices = trt_reshape(
            network, indices, squeeze_shape, is_shape_tensor=False
        )
    out_tensor = trt_cast(network, out, in_type)
    indices_tensor = trt_cast(network, indices, indices.dtype)
    replenish_layer_and_output(
        topk_layer, paddle_op.name(), paddle_op.get_output_names()
    )
    return out_tensor, indices_tensor


@converter_registry.register("pd_op.where", trt_version="8.x")
def where_converter(network, paddle_op, inputs):
    condition = inputs[0]
    x = inputs[1]
    y = inputs[2]

    select_layer = network.add_select(condition, x, y)

    replenish_layer_and_output(
        select_layer, paddle_op.name(), paddle_op.get_output_names()
    )
    return select_layer.get_output(0)


@converter_registry.register("pd_op.topk", trt_version="8.x")
def topk_converter(network, paddle_op, inputs):
    input_tensor = inputs[0]

    input_shape = input_tensor.shape

    axis = paddle_op.attrs().get("axis", -1)
    largest = paddle_op.attrs().get("largest", True)
    flag = trt.TopKOperation.MAX if largest else trt.TopKOperation.MIN

    k_list = get_input_constant_value(paddle_op, inputs, 1)
    if k_list is None:
        raise NotImplementedError("Dynamic k is not supported in TensorRT.")
    k = k_list[0]
    input_rank = len(input_shape)

    expand_to_2d = input_rank == 1
    if expand_to_2d:
        input_tensor = trt_unsqueeze(network, input_tensor, [1])

    input_type = input_tensor.dtype
    if input_type == trt.DataType.INT32:
        input_tensor = trt_cast(network, input_tensor, trt.DataType.FLOAT)

    if axis < 0:
        axis += input_rank

    layer = network.add_topk(input_tensor, flag, int(k), 1 << axis)
    values = layer.get_output(0)
    indices = layer.get_output(1)

    if expand_to_2d:
        values = squeeze_trt(network, values, [1])
        indices = squeeze_trt(network, indices, [1])

    if input_type == trt.DataType.INT32:
        values = trt_cast(network, values, trt.DataType.INT32)

    replenish_layer_and_output(
        layer, paddle_op.name(), paddle_op.get_output_names()
    )
    return values, indices


@converter_registry.register("pd_op.index_select", trt_version="8.x")
def index_select_converter(network, paddle_op, inputs):
    input_tensor = inputs[0]
    index_tensor = inputs[1]
    axis = paddle_op.attrs().get("axis", 0)

    reshape_layer = network.add_shuffle(index_tensor)
    reshape_layer.reshape_dims = (-1,)

    gather_layer = network.add_gather(
        input_tensor, reshape_layer.get_output(0), axis
    )

    replenish_layer_and_output(
        gather_layer, paddle_op.name(), paddle_op.get_output_names()
    )
    return gather_layer.get_output(0)
