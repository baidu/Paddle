# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import logging
import os
import sys

import numpy as np
import tensorrt as trt

from paddle.tensorrt.util import TensorRTConfigManager, TensorRTConstantManager

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)


from tensorrt import INetworkDefinition, ITensor

from paddle.base.log_helper import get_logger

_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s'
)

version = trt.__version__
version_list = list(map(int, version.split('.')))


def has_dynamic_shape(shape):
    return any(s == -1 for s in shape)


def append_ones(network, input, name, num_prepend_ones):
    layer = network.add_shuffle(input)

    if has_dynamic_shape(input.shape):
        input_shape_layer = network.add_shape(input)
        input_shape_layer.name = f"{name}_broadcast_orig_shape"
        prepend_shape_layer = network.add_constant(
            (num_prepend_ones,), np.ones((num_prepend_ones,), dtype=np.int32)
        )
        prepend_shape_layer.name = f"{name}_broadcast_prepend_ones"
        reshape_dim_layer = network.add_concatenation(
            [prepend_shape_layer.get_output(0), input_shape_layer.get_output(0)]
        )
        reshape_dim_layer.axis = 0
        reshape_dim_layer.name = f"{name}_broadcast_final_shape"
        layer.set_input(1, reshape_dim_layer.get_output(0))
    else:
        layer.reshape_dims = (1,) * num_prepend_ones + tuple(input.shape)

    layer.name = name
    return layer.get_output(0)


def broadcast(network, a, b, a_name, b_name, preset_diff=0):
    a_shape = tuple(a.shape)
    b_shape = tuple(b.shape)

    diff = len(a_shape) - len(b_shape) - preset_diff
    if diff > 0:
        b = append_ones(network, b, f"{b_name}_broadcast", diff)
    elif diff < 0:
        a = append_ones(network, a, f"{a_name}_broadcast", -diff)

    return a, b


def get_axes_for_reduce_op(
    dim,
    has_implicit_batch_dimension=False,
):
    if isinstance(dim, int):
        dim = (dim,)

    if has_implicit_batch_dimension:
        assert (
            0 not in dim
        ), "Can't reduce over batch dimension when it's implicit."

    axes = 0
    for d in dim:
        axes |= 1 << (d - (1 if has_implicit_batch_dimension else 0))

    return axes


def get_dynamic_dims(shape):
    """
    This function finds the dynamic dimensions in the given
    shape. A dimension is dynamic if it's -1.

    Args:
        shape (Shape): A sequence of integer that represents
            the shape of a tensor.

    Returns:
        A list of integers contains all the dynamic dimensions
        in the given shape
    """
    dynamic_dims = []
    for i, s in enumerate(shape):
        if s == -1:
            dynamic_dims.append(i)
    return dynamic_dims


def get_trt_plugin(plugin_name, field_collection, version, plugin_namespace=""):
    plugin_registry = trt.get_plugin_registry()
    plugin_creator = plugin_registry.get_plugin_creator(
        plugin_name, version, plugin_namespace
    )
    assert (
        plugin_creator
    ), f"Unable to found plugin creator with name {plugin_name}"
    plugin = plugin_creator.create_plugin(
        name=plugin_name, field_collection=field_collection
    )
    assert plugin is not None, f"Plugin:{plugin_name} could not be fetched"
    return plugin


def get_positive_dim(dim, dim_size):
    if dim < 0:
        return dim % dim_size
    return dim


def add_elementwise_layer(network, paddle_op, inputs, op_type):
    from paddle.tensorrt.util import support_fp32_mix_precision

    weight_shape = paddle_op.operands()[1].source().shape
    input_shape = paddle_op.operands()[0].source().shape

    weight_tensor = inputs[1]
    input_tensor = inputs[0]
    if type(inputs[1]) == trt.Weights:
        weight_tensor = network.add_constant(
            weight_shape, inputs[1]
        ).get_output(0)
    if type(inputs[0]) == trt.Weights:
        input_tensor = network.add_constant(input_shape, inputs[0]).get_output(
            0
        )
    lhs_val, rhs_val = broadcast(
        network,
        input_tensor,
        weight_tensor,
        input_tensor.name,
        weight_tensor.name,
    )
    layer = network.add_elementwise(lhs_val, rhs_val, op_type)
    support_fp32_mix_precision(paddle_op.name(), layer)
    return layer.get_output(0)


# Create and add 1D constant layer
def add_1D_constant_layer(network, data, dtype=np.int32, is_scalar=False):
    if not isinstance(data, list):
        data = [data]
    shape = () if is_scalar else (len(data),)
    constant_data = np.array(data, dtype=dtype)
    constant_layer = network.add_constant(shape, constant_data)
    return constant_layer.get_output(0)


# Create and add ND constant layer
def add_constant_layer(network, data, shape, dtype=np.int32):
    constant_data = np.array(data, dtype=dtype)
    constant_data = np.resize(constant_data, shape)
    constant_layer = network.add_constant(shape, constant_data)
    return constant_layer.get_output(0)


# Create an constant layer with shape_tensor and value
def fill_constant_layer(network, shape_tensor, tensor_rank, data, trt_dtype):
    fill_layer = network.add_fill(
        trt.Dims([tensor_rank]), trt.FillOperation.LINSPACE
    )
    np_dtype = map_trt_dtype(trt_dtype)
    fill_layer.set_input(0, shape_tensor)
    fill_layer.set_input(
        1, add_1D_constant_layer(network, data, np_dtype, is_scalar=True)
    )
    beta = [0] * tensor_rank
    fill_layer.set_input(
        2, add_1D_constant_layer(network, beta, np_dtype, is_scalar=False)
    )
    return fill_layer.get_output(0)


def trt_expand(network, input, rank, shape_tensor, shape_rank):
    if rank < shape_rank:
        one_rank_tensor = add_1D_constant_layer(
            network, [1] * (shape_rank - rank)
        )
        in_shape_tensor = trt_shape(network, input)
        itensors = [one_rank_tensor, in_shape_tensor]
        input_shape_tensor = trt_concat(network, itensors)
    else:
        input_shape_tensor = trt_shape(network, input)

    new_input_tensor = trt_reshape(network, input, input_shape_tensor, "", True)

    start = [0] * shape_rank
    starts_tensor = add_1D_constant_layer(network, start)
    one_tensor = add_1D_constant_layer(network, 1)
    sizes_tensor = trt_max(network, input_shape_tensor, shape_tensor)
    input_sub_tensor = trt_sub(network, input_shape_tensor, one_tensor)
    strides_tensor = trt_min(network, one_tensor, input_sub_tensor)

    slice_layer = network.add_slice(
        new_input_tensor, start, [0] * len(start), [0] * len(start)
    )
    slice_layer.set_input(1, starts_tensor)
    slice_layer.set_input(2, sizes_tensor)
    slice_layer.set_input(3, strides_tensor)

    return slice_layer.get_output(0)


# Concat not make rank changed
def trt_concat(network, inputs, axis=0):
    concat_layer = network.add_concatenation(inputs=inputs)
    if axis != 0:
        concat_layer.axis = axis
    return concat_layer.get_output(0)


def trt_cast(network, input, dtype):
    identity_layer = network.add_identity(input)
    identity_layer.set_output_type(0, dtype)
    identity_layer.get_output(0).dtype = dtype
    return identity_layer.get_output(0)


def trt_shape(network: INetworkDefinition, input: ITensor) -> ITensor:
    """
    Add a IShapeLayer to get the shape of `input` ITensor.
    This includes a workaround that casting the shape result(int64) from TRT10 back to int32.
    Many existing paddle op kernels only support input shape tensor as int32
    , to make TRT op more compatible with other paddle op, we cast back to int32.
    NOTE: please remove this workaround when all paddle op supports shape tensor in int64
    """
    shape_layer = network.add_shape(input)
    if version_list[0] >= 10:  # trt_version >=10
        # workaround
        return trt_cast(network, shape_layer.get_output(0), trt.int32)
    return shape_layer.get_output(0)


def trt_reshape(network, input, new_shape, name="", is_shape_tensor=False):
    reshape_layer = network.add_shuffle(input)
    if is_shape_tensor:
        reshape_layer.set_input(1, new_shape)
    else:
        reshape_layer.reshape_dims = new_shape
    if name != "":
        reshape_layer.name = name
    return reshape_layer.get_output(0)


# resize shape tensor's shape to 1dim
def resize_to_1d(network, shape_tensor):
    if shape_tensor is None:
        return shape_tensor
    if len(shape_tensor.shape) > 1:
        # shape_tensor need 1-dim in trt
        shape_tensor_layer = network.add_shuffle(shape_tensor)
        numel = 1
        for ele in shape_tensor.shape:
            numel *= ele
        shape_tensor_layer.reshape_dims = [numel]
        shape_tensor = shape_tensor_layer.get_output(0)
    return shape_tensor


# Get element tensor of 1D shape tensor
def get_shape_tensor_element(network, x, index, is_scalar=False):
    assert (
        index >= 0
    ), f"The index should be greater or equal than 0, but got {index}"
    index_tensor = add_1D_constant_layer(network, index, is_scalar=is_scalar)
    gather_layer = network.add_gather(input=x, indices=index_tensor, axis=0)
    shape_tensor = resize_to_1d(network, gather_layer.get_output(0))
    return shape_tensor


def trt_less(network, a, b):
    layer = network.add_elementwise(a, b, trt.ElementWiseOperation.LESS)
    return layer.get_output(0)


def trt_sum(network, a, b):
    layer = network.add_elementwise(a, b, trt.ElementWiseOperation.SUM)
    return layer.get_output(0)


def trt_max(network, a, b):
    layer = network.add_elementwise(a, b, trt.ElementWiseOperation.MAX)
    return layer.get_output(0)


def trt_sub(network, a, b):
    layer = network.add_elementwise(a, b, trt.ElementWiseOperation.SUB)
    return layer.get_output(0)


def trt_min(network, a, b):
    layer = network.add_elementwise(a, b, trt.ElementWiseOperation.MIN)
    return layer.get_output(0)


def trt_div(network, a, b):
    layer = network.add_elementwise(a, b, trt.ElementWiseOperation.DIV)
    return layer.get_output(0)


def trt_floor_div(network, a, b):
    layer = network.add_elementwise(a, b, trt.ElementWiseOperation.FLOOR_DIV)
    return layer.get_output(0)


def trt_equal(network, a, b):
    layer = network.add_elementwise(a, b, trt.ElementWiseOperation.EQUAL)
    return layer.get_output(0)


def trt_gather(network, input, indices, axis=0):
    indices_tensor = add_1D_constant_layer(network, indices)
    result = network.add_gather(input, indices_tensor, axis).get_output(0)
    return result


def trt_prod(network, a, b):
    layer = network.add_elementwise(a, b, trt.ElementWiseOperation.PROD)
    return layer.get_output(0)


def trt_pow(network, a, b):
    layer = network.add_elementwise(a, b, trt.ElementWiseOperation.POW)
    return layer.get_output(0)


def cast_tensor(network, input_tensor, dtype):
    layer = network.add_identity(input_tensor)
    layer.set_output_type(0, dtype)
    return layer.get_output(0)


def build_start_tensor(network, rank, axis_tensor, offset):
    # Create indices_tensor [0, 1, ..., rank-1]
    indices = np.arange(rank, dtype=np.int32)
    indices_tensor = network.add_constant([rank], indices).get_output(0)

    # Create mask: mask = (indices == axis_tensor)
    mask = network.add_elementwise(
        indices_tensor, axis_tensor, trt.ElementWiseOperation.EQUAL
    ).get_output(0)
    mask_int = cast_tensor(network, mask, trt.int32)

    # Calculate start_tensor = mask_int * offset
    start_tensor = network.add_elementwise(
        mask_int, offset, trt.ElementWiseOperation.PROD
    ).get_output(0)

    return start_tensor


def build_size_tensor(
    network, rank, axis_tensor, size_value, input_shape_tensor
):
    # Create indices_tensor [0, 1, ..., rank-1]
    indices = np.arange(rank, dtype=np.int32)
    indices_tensor = network.add_constant([rank], indices).get_output(0)

    # Create mask: mask = (indices == axis_tensor)
    mask = network.add_elementwise(
        indices_tensor, axis_tensor, trt.ElementWiseOperation.EQUAL
    ).get_output(0)
    mask_int = cast_tensor(network, mask, trt.int32)

    # Create ones_tensor
    ones_tensor = network.add_constant(
        [rank], np.ones([rank], dtype=np.int32)
    ).get_output(0)

    # Calculate inverse_mask = ones_tensor - mask_int
    inverse_mask = network.add_elementwise(
        ones_tensor, mask_int, trt.ElementWiseOperation.SUB
    ).get_output(0)

    # Calculate size_tensor = mask_int * size_value + inverse_mask * input_shape_tensor
    size_value_broadcast = network.add_elementwise(
        mask_int, size_value, trt.ElementWiseOperation.PROD
    ).get_output(0)

    input_shape_broadcast = network.add_elementwise(
        inverse_mask, input_shape_tensor, trt.ElementWiseOperation.PROD
    ).get_output(0)

    size_tensor = network.add_elementwise(
        size_value_broadcast,
        input_shape_broadcast,
        trt.ElementWiseOperation.SUM,
    ).get_output(0)

    return size_tensor


# convert trt_dtype to numpy dtype
def map_trt_dtype(trt_dtype):
    dtype_map = {
        trt.DataType.FLOAT: np.float32,
        trt.DataType.HALF: np.float16,
        trt.DataType.INT32: np.int32,
        trt.DataType.INT8: np.int8,
        trt.DataType.BOOL: bool,
    }
    if trt_dtype in dtype_map:
        return dtype_map[trt_dtype]
    else:
        raise TypeError(f"Unsupported trt_dtype: {trt_dtype}")


# Reduce the given tensor in the TensorRT network to a scalar
def trt_reduce_to_scalar(network, tensor, dtype=trt.int32):
    if len(tensor.shape) == 0:
        return tensor
    axes = 0
    for i in range(len(tensor.shape)):
        axes |= 1 << i
    reduce_layer = network.add_reduce(
        tensor, trt.ReduceOperation.SUM, axes, keep_dims=False
    )
    scalar = trt_cast(network, reduce_layer.get_output(0), dtype)
    return scalar


def convert_conv2d(network, paddle_op, inputs):
    from paddle.tensorrt.util import support_fp32_mix_precision

    bias = None
    if (
        paddle_op.name() == "pd_op.conv2d"
        or paddle_op.name() == "pd_op.depthwise_conv2d"
    ):
        input_tensor, filter = inputs
    elif (
        paddle_op.name() == "pd_op.conv2d_transpose"
        or paddle_op.name() == "pd_op.depthwise_conv2d_transpose"
    ):
        if len(inputs) == 3:
            input_tensor, filter, output_size = inputs
        elif len(inputs) == 2:
            input_tensor, filter = inputs
            output_size = None
        else:
            raise ValueError("Invalid number of inputs for conv2d_transpose")
    if paddle_op.name() == "pd_op.fused_conv2d_add_act":
        input_tensor, filter, bias, _ = inputs
    input_shape = paddle_op.operands()[0].source().shape
    filter_shape = paddle_op.operands()[1].source().shape

    if len(filter_shape) != 4:
        raise ValueError(
            f"filter's dims size should be 4, but got {len(filter_shape)}"
        )

    n_output = filter_shape[0]
    n_input = filter_shape[1]
    filter_h = filter_shape[2]
    filter_w = filter_shape[3]

    paddings = paddle_op.attrs().get("paddings", [0, 0])
    stride = paddle_op.attrs().get("strides", [1, 1])
    dilation = paddle_op.attrs().get("dilations", [1, 1])
    groups = paddle_op.attrs().get("groups", 1)

    if has_dynamic_shape(input_shape):
        assert (
            input_shape[1] != -1
        ), "Channel dim can't be dynamic for transpose convolution."

    output_padding = paddle_op.attrs().get("output_padding", [0, 0])
    padding_algorithm = paddle_op.attrs().get("padding_algorithm", "EXPLICIT")
    if padding_algorithm == "VALID":
        paddings = [0] * len(paddings)

    nv_ksize = trt.DimsHW(filter_h, filter_w)
    nv_dilations = trt.DimsHW(dilation[0], dilation[1])
    nv_strides = trt.DimsHW(stride[0], stride[1])

    pre_paddings = [0, 0]
    post_paddings = [0, 0]

    if len(paddings) == 2:
        pre_paddings[0] = paddings[0]
        pre_paddings[1] = paddings[1]
        post_paddings[0] = paddings[0]
        post_paddings[1] = paddings[1]
    elif len(paddings) == 4:
        pre_paddings[0] = paddings[0]
        pre_paddings[1] = paddings[2]
        post_paddings[0] = paddings[1]
        post_paddings[1] = paddings[3]
    else:
        raise ValueError(f"Unsupported paddings size: {len(paddings)}")

    if (
        paddle_op.name() == "pd_op.conv2d"
        or paddle_op.name() == "pd_op.depthwise_conv2d"
        or paddle_op.name() == "pd_op.fused_conv2d_add_act"
    ):
        layer = network.add_convolution_nd(
            input=input_tensor,
            num_output_maps=n_output,
            kernel_shape=nv_ksize,
            kernel=filter,
            bias=bias,
        )
    elif (
        paddle_op.name() == "pd_op.conv2d_transpose"
        or paddle_op.name() == "pd_op.depthwise_conv2d_transpose"
    ):
        layer = network.add_deconvolution_nd(
            input=input_tensor,
            num_output_maps=n_input * groups,
            kernel_shape=nv_ksize,
            kernel=filter,
            bias=None,
        )

    layer.stride_nd = nv_strides
    layer.pre_padding = pre_paddings

    if output_padding:
        post_paddings[0] -= output_padding[0]
        post_paddings[1] -= output_padding[1]

    if post_paddings[0] < 0 or post_paddings[1] < 0:
        raise ValueError("The value PostPadding should be >= 0.")

    layer.post_padding = post_paddings
    layer.num_groups = groups

    if padding_algorithm == "SAME":
        layer.padding_mode = trt.PaddingMode.SAME_UPPER
        nv_dilations = trt.DimsHW(1, 1)

    layer.dilation_nd = nv_dilations
    support_fp32_mix_precision(paddle_op.name(), layer)

    return layer.get_output(0)


def get_input_constant_value(paddle_op, inputs, input_index):
    input_op = paddle_op.operands()[input_index].source().get_defining_op()
    constant_manager = TensorRTConstantManager()
    if input_op.name() == "builtin.constant":
        return constant_manager.get_constant_value(
            input_op.attrs()["value"]
        ).tolist()
    elif input_op.name() == "pd_op.full_int_array":
        return input_op.attrs()["value"]
    elif input_op.name() == "pd_op.full":
        return [input_op.attrs()["value"]]
    else:
        return None


def add_reduce_layer(network, paddle_op, inputs, op_type):
    input_tensor = inputs[0]
    axis = get_input_constant_value(paddle_op, inputs, 1)
    input_shape = paddle_op.operands()[0].source().shape
    keepdim = paddle_op.attrs()["keepdim"]
    if network.has_implicit_batch_dimension:
        assert (
            axis != 0
        ), "can't reduce on axis == 0 when network has implicit batch dimension"
    output_shape = []
    if len(axis) == 0:
        axis = list(range(len(input_shape)))
    for i in range(len(axis)):
        if axis[i] < 0:
            axis[i] = len(input_shape) + axis[i]
    layer = network.add_reduce(
        input_tensor,
        op_type,
        axes=get_axes_for_reduce_op(axis),
        keep_dims=keepdim,
    )
    layer.get_output(0).dtype = layer.get_input(0).dtype
    return layer.get_output(0)


def add_cast_reduce_layer(network, paddle_op, inputs, op_type):
    input_tensor = inputs[0]
    cast_layer = network.add_identity(input_tensor)
    cast_layer.set_output_type(0, trt.int32)
    cast_layer.get_output(0).dtype = trt.int32

    axis = paddle_op.attrs().get("axis")
    input_shape = paddle_op.operands()[0].source().shape
    keepdim = paddle_op.attrs()["keepdim"]
    if network.has_implicit_batch_dimension:
        assert (
            axis != 0
        ), "can't reduce on axis == 0 when network has implicit batch dimension"
    output_shape = []
    if len(axis) == 0:
        axis = list(range(len(input_shape)))
    for i in range(len(axis)):
        if axis[i] < 0:
            axis[i] = len(input_shape) + axis[i]
    layer = network.add_reduce(
        cast_layer.get_output(0),
        op_type,
        axes=get_axes_for_reduce_op(axis),
        keep_dims=keepdim,
    )
    layer.set_output_type(0, trt.bool)
    layer.get_output(0).dtype = cast_layer.get_output(0).dtype
    return layer.get_output(0)


def fix_negative_indices(network, input_shape, indices):
    rank = len(input_shape.shape)
    zero_tensor = add_1D_constant_layer(network, [0] * rank)
    minus_one_tensor = add_1D_constant_layer(network, [-1] * rank)

    min_indices_zero = trt_min(network, indices, zero_tensor)
    sign = trt_max(network, min_indices_zero, minus_one_tensor)
    sub = trt_prod(network, sign, input_shape)
    fixed_indices = trt_sub(network, indices, sub)
    return fixed_indices


def trt_unsqueeze(network, input_tensor, axes):
    input_shape = network.add_shape(input_tensor).get_output(0)

    axis_set = set(axes)

    subscripts = list(range(len(input_tensor.shape)))

    for axis in sorted(axis_set):
        subscripts.insert(axis, len(input_tensor.shape))

    one_tensor = network.add_constant(
        (1,), np.array([1], dtype=np.int32)
    ).get_output(0)
    extended_shape = network.add_concatenation(
        [input_shape, one_tensor]
    ).get_output(0)

    gather_layer = network.add_gather(
        extended_shape,
        network.add_constant(
            (len(subscripts),), np.array(subscripts, dtype=np.int32)
        ).get_output(0),
        axis=0,
    )
    new_shape_tensor = gather_layer.get_output(0)

    reshaped_tensor = network.add_shuffle(input_tensor)
    reshaped_tensor.set_input(1, new_shape_tensor)

    return reshaped_tensor.get_output(0)


def squeeze_trt(network, input_tensor, axes):
    input_shape = network.add_shape(input_tensor).get_output(0)
    input_shape = input_tensor.shape
    all_dims = list(range(len(input_shape)))
    remaining_dims = [dim for dim in all_dims if dim not in axes]

    input_shape_tensor = network.add_shape(input_tensor).get_output(0)

    remaining_dims_tensor = network.add_constant(
        (len(remaining_dims),), np.array(remaining_dims, dtype=np.int32)
    ).get_output(0)

    new_shape_tensor = network.add_gather(
        input_shape_tensor, remaining_dims_tensor, axis=0
    ).get_output(0)
    reshape_layer = network.add_shuffle(input_tensor)
    reshape_layer.set_input(1, new_shape_tensor)
    return reshape_layer.get_output(0)


def unary_op_converter(network, paddle_op, inputs):
    from paddle.tensorrt import PrecisionMode

    ops_type_map = {
        "pd_op.sqrt": [trt.UnaryOperation.SQRT],
        "pd_op.sqrt_": [trt.UnaryOperation.SQRT],
        "pd_op.floor": [trt.UnaryOperation.FLOOR],
        "pd_op.exp": [trt.UnaryOperation.EXP],
        "pd_op.abs": [trt.UnaryOperation.ABS],
        "pd_op.abs_": [trt.UnaryOperation.ABS],
        "pd_op.sin": [trt.UnaryOperation.SIN],
        "pd_op.cos": [trt.UnaryOperation.COS],
        "pd_op.sinh": [trt.UnaryOperation.SINH],
        "pd_op.cosh": [trt.UnaryOperation.COSH],
        "pd_op.asinh": [trt.UnaryOperation.ASINH],
        "pd_op.acosh": [trt.UnaryOperation.ACOSH],
        "pd_op.atanh": [trt.UnaryOperation.ATANH],
        "pd_op.ceil": [trt.UnaryOperation.CEIL],
        "pd_op.reciprocal": [trt.UnaryOperation.RECIP],
        "pd_op.erf": [trt.UnaryOperation.ERF],
        "pd_op.sign": [trt.UnaryOperation.SIGN],
        "pd_op.round": [trt.UnaryOperation.ROUND],
        "pd_op.logical_not": [trt.UnaryOperation.NOT],
        "pd_op.rsqrt": [trt.UnaryOperation.SQRT, trt.UnaryOperation.RECIP],
        "pd_op.tan": [trt.UnaryOperation.TAN],
        "pd_op.asin": [trt.UnaryOperation.ASIN],
        "pd_op.acos": [trt.UnaryOperation.ACOS],
        "pd_op.atan": [trt.UnaryOperation.ATAN],
    }

    input_tensor = inputs[0]
    layer = None
    org_type = input_tensor.dtype

    trt_type_mapping = {
        trt.DataType.INT8: trt.int8,
        trt.DataType.INT32: trt.int32,
    }

    trt_manager = TensorRTConfigManager()
    precision_mode = trt_manager.get_precision_mode()

    need_cast = org_type in [trt.DataType.INT8, trt.DataType.INT32]
    if need_cast:
        identity_layer = network.add_identity(input_tensor)
        if precision_mode == PrecisionMode.FP32:
            identity_layer.set_output_type(0, trt.float32)
        else:
            identity_layer.set_output_type(0, trt.float16)
        input_tensor = identity_layer.get_output(0)

    if paddle_op.name() in ops_type_map:
        for trt_op in ops_type_map[paddle_op.name()]:
            layer = network.add_unary(input_tensor, trt_op)
            input_tensor = layer.get_output(0)
    else:
        raise NotImplementedError(
            f"Unsupported unary operation: {paddle_op.name()}"
        )
    if need_cast:
        restore_layer = network.add_identity(input_tensor)
        restore_layer.set_output_type(0, trt_type_mapping[org_type])
        input_tensor = restore_layer.get_output(0)

    return input_tensor


# get the length of the specified axis for input_tensor
def get_axis_length(network, input_tensor, axis, is_scalar=False):
    input_shape = input_tensor.shape
    if input_shape[axis] >= 0:
        output_tensor = add_1D_constant_layer(
            network, input_shape[axis], is_scalar=is_scalar
        )
    else:
        dynamic_shape = trt_shape(network, input_tensor)
        output_tensor = get_shape_tensor_element(
            network, dynamic_shape, axis, is_scalar
        )
    return output_tensor


def WithFp16():
    from paddle.tensorrt import PrecisionMode

    trt_manager = TensorRTConfigManager()
    precision_mode = trt_manager.get_precision_mode()
    enable_fp16 = False
    if precision_mode == PrecisionMode.FP16:
        enable_fp16 = True
    # TODO(lizexu123) WithInt8() and use_dla are not yet implemented
    return enable_fp16
