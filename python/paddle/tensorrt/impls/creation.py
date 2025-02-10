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

import paddle
from paddle.pir.core import _PADDLE_PIR_DTYPE_2_NUMPY_DTYPE
from paddle.tensorrt.converter_utils import (
    add_1D_constant_layer,
    cast_tensor,
    get_input_constant_value,
    resize_to_1d,
    trt_cast,
    trt_floor_div,
    trt_max,
    trt_min,
    trt_reduce_to_scalar,
    trt_reshape,
    trt_shape,
    trt_sub,
)
from paddle.tensorrt.register import converter_registry


@converter_registry.register(
    "pd_op.full_int_array", trt_version="trt_version_ge=8.0"
)
def full_int_array_converter(network, paddle_op, inputs):
    value = paddle_op.attrs()["value"]
    if len(value) == 0:
        return ()
    value_weight = trt.Weights(np.array(value, dtype=np.int32))
    full_int_array_layer = network.add_constant([len(value)], value_weight)
    return full_int_array_layer.get_output(0)


@converter_registry.register("pd_op.full", trt_version="trt_version_ge=8.0")
def full_converter(network, paddle_op, inputs):
    shape = paddle_op.attrs()["shape"]
    value = paddle_op.attrs().get("value", 1.0)
    dtype = paddle_op.attrs().get("dtype")
    out_dtype = np.dtype(_PADDLE_PIR_DTYPE_2_NUMPY_DTYPE[dtype])
    if out_dtype == np.dtype("float64"):
        out_dtype = np.dtype("float32")
    if out_dtype == np.dtype("int64"):
        out_dtype = np.dtype("int32")
    full_layer = network.add_constant(
        shape, np.full(shape, value, dtype=out_dtype)
    )
    return full_layer.get_output(0)


@converter_registry.register("pd_op.assign", trt_version="8.x")
@converter_registry.register("pd_op.assign_out_", trt_version="8.x")
def assign_converter(network, paddle_op, inputs):
    input_tensor = inputs[0]
    identity_layer = network.add_identity(input_tensor)
    return identity_layer.get_output(0)


@converter_registry.register("pd_op.assign_value", trt_version="8.x")
@converter_registry.register("pd_op.assign_value_", trt_version="8.x")
def assign_value_converter(network, paddle_op, inputs):
    attrs = paddle_op.attrs()
    shape = attrs['shape']
    dtype = attrs['dtype']
    values = attrs['values']

    paddle_to_np_dtype_map = {
        paddle.float16: np.float16,
        paddle.float32: np.float32,
        paddle.float64: np.float64,
        paddle.int32: np.int32,
        paddle.int64: np.int64,
    }

    if dtype not in paddle_to_np_dtype_map:
        raise ValueError(
            f"Unsupported dtype {dtype} for assign_value op in TRT converter."
        )

    np_dtype = paddle_to_np_dtype_map[dtype]

    arr = np.array(values, dtype=np_dtype).reshape(shape)
    if np_dtype == np.int64:
        arr = arr.astype(np.int32)
    const_layer = network.add_constant(tuple(shape), arr)
    if const_layer is None:
        raise RuntimeError("Failed to create constant layer for assign_value.")

    return const_layer.get_output(0)


@converter_registry.register("pd_op.arange", trt_version="8.x")
def arange_converter(network, paddle_op, inputs):
    start, end, step = inputs
    zero_tensor = add_1D_constant_layer(network, 0, np.int32)

    delta = trt_sub(network, end, start)

    f_quotient_tensor = trt_floor_div(network, delta, step)

    if start.dtype == trt.DataType.FLOAT:
        quotient_tensor = cast_tensor(network, f_quotient_tensor, trt.int32)
    else:
        quotient_tensor = f_quotient_tensor

    number_tensor = trt_max(network, quotient_tensor, zero_tensor)

    start_tensor = trt_reshape(network, start, ())

    fill_layer = network.add_fill(shape=(), op=trt.FillOperation.LINSPACE)
    fill_layer.set_input(0, number_tensor)
    fill_layer.set_input(1, start_tensor)
    fill_layer.set_input(2, step)

    return fill_layer.get_output(0)


@converter_registry.register("pd_op.full_like", trt_version="8.x")
def full_like_converter(network, paddle_op, inputs):
    input_tensor = inputs[0]
    shape = input_tensor.shape
    ndims = len(shape)

    dtype = int(paddle_op.attrs().get("dtype", -1))

    dtype_map = {
        0: None,  # Undefined
        1: trt.bool,  # bool
        2: trt.int32,  # int32
        3: trt.int32,  # int64 -> int32
        4: trt.int32,  # int16 -> int32
        5: trt.float32,  # float16 -> float32
        6: trt.float32,  # float64 -> float32
        7: trt.float32,  # float32
        8: trt.int32,  # uint8 -> int32
        11: trt.float32,  # float32
    }

    target_dtype = dtype_map.get(dtype, None)
    if target_dtype is None:
        target_dtype = input_tensor.dtype

    value = get_input_constant_value(paddle_op, inputs, 1)
    if value is not None:
        if isinstance(value, (list, tuple)):
            value = value[0] if value else 0

        if target_dtype == trt.int32:
            value_tensor = add_1D_constant_layer(network, int(value), np.int32)
        else:
            value_tensor = add_1D_constant_layer(
                network, float(value), np.float32
            )
    else:
        value_tensor = inputs[1]
        if value_tensor.dtype != target_dtype:
            value_tensor = trt_cast(network, value_tensor, target_dtype)

    shape_tensor = trt_shape(network, input_tensor)
    one_rank_tensor = add_1D_constant_layer(network, [1] * ndims)
    input_shape_tensor = one_rank_tensor

    shuffle_layer = network.add_shuffle(value_tensor)
    shuffle_layer.set_input(1, input_shape_tensor)
    start = trt.Dims([0] * ndims)
    size = trt.Dims([1] * ndims)
    stride = trt.Dims([1] * ndims)

    starts_tensor = add_1D_constant_layer(network, [0] * ndims)
    one_tensor = add_1D_constant_layer(network, 1)
    sizes_tensor = trt_max(network, input_shape_tensor, shape_tensor)
    input_sub_tensor = trt_sub(network, input_shape_tensor, one_tensor)
    strides_tensor = trt_min(network, one_tensor, input_sub_tensor)

    layer = network.add_slice(shuffle_layer.get_output(0), start, size, stride)
    layer.set_input(1, starts_tensor)
    layer.set_input(2, sizes_tensor)
    layer.set_input(3, strides_tensor)

    output = layer.get_output(0)

    if output.dtype != target_dtype:
        output = trt_cast(network, output, target_dtype)

    return output


@converter_registry.register("pd_op.full_with_tensor", trt_version="8.x")
def full_with_tensor_converter(network, paddle_op, inputs):
    value_input = inputs[0]

    shape_tensor = None
    dtype = paddle_op.attrs()["dtype"]

    operands = paddle_op.operands()
    num_operands = len(operands)

    if num_operands >= 2:
        shape_tensor = inputs[1]
        if isinstance(shape_tensor, list):
            shape_tensor_list = shape_tensor
        else:
            shape_tensor_list = [shape_tensor]

    shape_val = get_input_constant_value(paddle_op, inputs, 1)
    if shape_val is not None:
        shape_tensor = shape_val
        is_static_shape = True
    else:
        shape_tensor = inputs[1]
        is_static_shape = False

    shape_nbDims = 0
    tensor_rank = 0
    if isinstance(shape_tensor, trt.ITensor):
        shape_x = shape_tensor.shape
        shape_nbDims = len(shape_x)
        shapes_tensor = shape_tensor
    elif isinstance(shape_tensor, (list, tuple)):
        shape_nbDims = len(shape_tensor)
        shapes_tensor = shape_tensor
    else:
        raise TypeError(f"Unsupported shape_tensor type: {type(shape_tensor)}")

    if shape_tensor is not None and len(shape_tensor_list) == 1:
        is_dynamic_shape = True
    elif len(shape_tensor_list) >= 1:
        is_dynamic_shape = True
    else:
        is_dynamic_shape = False

    if is_dynamic_shape:
        if len(shape_tensor_list) == 1:
            shape_tensor = shape_tensor_list[0]
            if not isinstance(shape_tensor, trt.ITensor):
                raise TypeError("shape_tensor must be an ITensor")
            tensor_rank = shape_tensor.shape[0]
            shapes_tensor = shape_tensor
        else:
            shape_tensors = []
            for tensor in shape_tensor_list:
                if len(tensor.shape) == 0:
                    tensor = trt_reshape(network, tensor, (1,))
                shape_tensors.append(tensor)

            concat_layer = network.add_concatenation(shape_tensors)
            shapes_tensor = concat_layer.get_output(0)
            tensor_rank = len(shape_tensors)

        shapes_tensor = resize_to_1d(network, shapes_tensor)
        fill_layer = network.add_fill(shape=(), op=trt.FillOperation.LINSPACE)
        fill_layer.set_input(0, shapes_tensor)

    if dtype == paddle.int32 or dtype == paddle.int64:
        beta_vec = [0] * tensor_rank
        value_input = trt_reduce_to_scalar(network, value_input)
        fill_layer.set_input(1, value_input)
        fill_layer.set_input(
            2, add_1D_constant_layer(network, beta_vec, np.int32)
        )
    elif dtype == paddle.float32:
        beta_vec = [0.0] * tensor_rank
        value_input = trt_reduce_to_scalar(network, value_input, trt.float32)
        fill_layer.set_input(1, value_input)
        fill_layer.set_input(
            2, add_1D_constant_layer(network, beta_vec, np.float32)
        )
    else:
        raise ValueError(f"Unsupported dtype for full_with_tensor: {dtype}")

    output_tensor = fill_layer.get_output(0)
    return output_tensor
