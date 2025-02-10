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

from paddle import pir
from paddle.tensorrt.converter_utils import (
    add_1D_constant_layer,
    get_input_constant_value,
    get_shape_tensor_element,
    trt_concat,
    trt_reshape,
    trt_shape,
)
from paddle.tensorrt.register import converter_registry
from paddle.tensorrt.util import get_trt_version_list


@converter_registry.register("pd_op.dropout", trt_version="8.x")
def dropout_converter(network, paddle_op, inputs):
    input_x = inputs[0]
    dropout_prob = get_input_constant_value(paddle_op, inputs, 2)[0]
    downgrade_in_infer = paddle_op.attrs().get("mode")

    if downgrade_in_infer == "upscale_in_train":
        shuffle_layer = network.add_shuffle(input_x)
        return shuffle_layer.get_output(0)

    weight_data = np.array([1 - dropout_prob]).astype("float32")
    scale_weights = trt.Weights(weight_data)
    shift_weights = trt.Weights(np.array([0]).astype("float32"))
    power_weights = trt.Weights(np.array([1]).astype("float32"))

    scale_layer = network.add_scale(
        input_x,
        mode=trt.ScaleMode.UNIFORM,
        shift=shift_weights,
        scale=scale_weights,
        power=power_weights,
    )

    return scale_layer.get_output(0)


@converter_registry.register(
    "pd_op.bilinear_interp", trt_version="trt_version_ge=8.0"
)
def bilinear_interp_converter(network, paddle_op, inputs):
    input_tensor = inputs[0]
    input_shape_tensor = network.add_shape(input_tensor).get_output(0)
    input_rank = (
        input_shape_tensor.shape
    )  # The reason is unknown that adding this unused code make input_shape_tensor maintain the correct result.
    data_format = paddle_op.attrs().get("data_format")
    interp_method = paddle_op.attrs().get("interp_method")
    align_corners = paddle_op.attrs().get("align_corners")
    align_mode = paddle_op.attrs().get("align_mode")
    out_h = paddle_op.attrs().get("out_h")
    out_w = paddle_op.attrs().get("out_w")
    out_d = paddle_op.attrs().get("out_d")
    scale_attr = paddle_op.attrs().get("scale")

    trt_major = get_trt_version_list()[0]
    trt_minor = get_trt_version_list()[1]
    trt_version_float = float(f"{trt_major}.{trt_minor}")

    resize_layer = network.add_resize(input_tensor)
    # Set resize mode to LINEAR unconditionally
    if trt_version_float >= 8.6:
        resize_layer.resize_mode = trt.InterpolationMode.LINEAR
    else:
        resize_layer.resize_mode = trt.ResizeMode.LINEAR

    # Set coordinate transformation based on align_corners and align_mode
    if align_corners:
        resize_layer.coordinate_transformation = (
            trt.ResizeCoordinateTransformation.ALIGN_CORNERS
        )
    else:
        if align_mode == 0:
            resize_layer.coordinate_transformation = (
                trt.ResizeCoordinateTransformation.HALF_PIXEL
            )
        else:  # align_mode == 1
            resize_layer.coordinate_transformation = (
                trt.ResizeCoordinateTransformation.ASYMMETRIC
            )

    if data_format == "NCHW":
        h_axis = 2
        w_axis = 3
    elif data_format == "NHWC":
        h_axis = 1
        w_axis = 2

    in_dim = input_tensor.shape

    outsize_tensor = None
    if trt_version_float >= 8.2:
        if not pir.is_fake_value(paddle_op.operands()[1].source()):
            size_tensor_operand = paddle_op.operands()[1].source()
            if len(inputs) > 1 and inputs[1] is not None:
                output_tensor_operand = paddle_op.operands()[1].source()
                outsize_tensor = inputs[1]
        elif not pir.is_fake_value(paddle_op.operands()[2].source()):
            size_tensor_operand = paddle_op.operands()[2].source()
            size_tensor = inputs[2]
            if size_tensor_operand.is_combine():
                size_tensors = []
                if not isinstance(size_tensor, list):
                    size_tensors = [size_tensor]
                else:
                    size_tensors = size_tensor
                if len(size_tensors) >= 2:
                    # Extract the first two elements representing height and width
                    outsize_h = size_tensors[0]
                    outsize_w = size_tensors[1]
                    outsize_tensor = network.add_concatenation(
                        [outsize_h, outsize_w]
                    ).get_output(0)
            else:
                size_tensor_shape = size_tensor_operand.source().shape
                if size_tensor_shape.size >= 2:
                    outsize_h = network.add_slice(
                        size_tensor, start=[0], shape=[1], stride=[1]
                    ).get_output(0)
                    outsize_w = network.add_slice(
                        size_tensor, start=[1], shape=[1], stride=[1]
                    ).get_output(0)
                    outsize_tensor = network.add_concatenation(
                        [outsize_h, outsize_w]
                    ).get_output(0)
    use_scales = True
    if outsize_tensor is not None:
        use_scales = False
    elif out_h > 0 and out_w > 0 and scale_attr is not None:
        use_scales = True

    if use_scales:
        scale_h = -1.0
        scale_w = -1.0

        if scale_attr and len(scale_attr) > 1:
            scale_h = scale_attr[0]
            scale_w = scale_attr[1]
        elif scale_attr and len(scale_attr) == 1:
            scale_h = scale_w = scale_attr[0]

        if scale_w > 0 and scale_h > 0:
            if in_dim[h_axis] > 0 and in_dim[w_axis] > 0:
                out_h = int(in_dim[h_axis] * scale_h)
                out_w = int(in_dim[w_axis] * scale_w)
        else:
            if out_h > 0 and out_w > 0 and not (scale_w > 0 and scale_h > 0):
                if in_dim[h_axis] > 0 and in_dim[w_axis] > 0:
                    scale_h = float(out_h) / float(in_dim[h_axis])
                    scale_w = float(out_w) / float(in_dim[w_axis])

        scales = [1.0] * len(input_tensor.shape)
        if data_format == "NCHW":
            scales[2] = scale_h
            scales[3] = scale_w
        elif data_format == "NHWC":
            scales[1] = scale_h
            scales[2] = scale_w

        resize_layer.scales = scales
    else:
        if outsize_tensor is not None:
            outsize_itensors = []
            batch_dim = get_shape_tensor_element(network, input_shape_tensor, 0)
            outsize_itensors.append(batch_dim)
            if data_format == "NCHW":
                channel_dim = get_shape_tensor_element(
                    network, input_shape_tensor, 1
                )
                outsize_itensors.append(channel_dim)
                outsize_itensors.append(outsize_tensor)
            elif data_format == "NHWC":
                channel_dim = get_shape_tensor_element(
                    network, input_shape_tensor, 3
                )
                outsize_itensors.append(outsize_tensor)
                outsize_itensors.append(channel_dim)
            output_size_tensor = network.add_concatenation(
                outsize_itensors
            ).get_output(0)
            resize_layer.set_input(1, output_size_tensor)

    return resize_layer.get_output(0)


@converter_registry.register(
    "pd_op.embedding", trt_version="trt_version_ge=8.0"
)
def embedding_converter(network, paddle_op, inputs):
    x = inputs[0]
    weight = inputs[1]
    gather_layer = network.add_gather(weight, x, 0)
    return gather_layer.get_output(0)


@converter_registry.register("pd_op.unbind", trt_version="trt_version_ge=8.0")
def unbind_converter(network, paddle_op, inputs):
    x = inputs[0]
    input_shape = x.shape
    axis = paddle_op.attrs().get("axis")
    rank = len(input_shape)
    if axis < 0:
        axis += rank
    axis = int(axis)
    # Input for the add_slice layer
    start_tensors = []
    size_tensors = []
    # Input for the add_shuffle layer
    new_shape_tensors = []
    for i in range(rank):
        if axis == i:
            size_tensors.append(add_1D_constant_layer(network, 1))
        else:
            size_tensors.append(
                get_shape_tensor_element(network, trt_shape(network, x), i)
            )
            new_shape_tensors.append(
                get_shape_tensor_element(network, trt_shape(network, x), i)
            )
        start_tensors.append(add_1D_constant_layer(network, 0))

    new_shape_tensor = trt_concat(network, new_shape_tensors)
    stride = trt.Dims([1] * rank)
    outputs = []
    output_size = len(paddle_op.results()[0].type().as_vec_type().as_list())
    for i in range(output_size):
        start_tensors[axis] = add_1D_constant_layer(network, i)
        # Create Slice layer
        slice_layer = network.add_slice(
            x,
            stride,
            stride,
            stride,
        )
        slice_layer.set_input(1, trt_concat(network, start_tensors))
        slice_layer.set_input(2, trt_concat(network, size_tensors))
        shuffle_layer = trt_reshape(
            network,
            slice_layer.get_output(0),
            new_shape_tensor,
            is_shape_tensor=True,
        )
        outputs.append(shuffle_layer)
    return outputs


@converter_registry.register(
    "pd_op.nearest_interp", trt_version="trt_version_ge=8.0"
)
def nearest_interp_converter(network, paddle_op, inputs):
    input_tensor = inputs[0]
    input_shape_tensor = network.add_shape(input_tensor).get_output(0)
    input_rank = (
        input_shape_tensor.shape
    )  # The reason is unknown that adding this unused code make input_shape_tensor maintain the correct result.
    data_format = paddle_op.attrs().get("data_format")
    interp_method = paddle_op.attrs().get("interp_method")
    align_corners = paddle_op.attrs().get("align_corners")
    out_h = paddle_op.attrs().get("out_h")
    out_w = paddle_op.attrs().get("out_w")
    out_d = paddle_op.attrs().get("out_d")
    scale_attr = paddle_op.attrs().get("scale")

    # Parse TensorRT version
    trt_major = get_trt_version_list()[0]
    trt_minor = get_trt_version_list()[1]
    trt_version_float = float(f"{trt_major}.{trt_minor}")

    # Create Resize layer
    resize_layer = network.add_resize(input_tensor)

    if trt_version_float >= 8.6:
        if align_corners:
            resize_layer.coordinate_transformation = (
                trt.ResizeCoordinateTransformation.ASYMMETRIC
            )
    else:
        resize_layer.coordinate_transformation = (
            trt.ResizeCoordinateTransformation.ASYMMETRIC
        )

    in_dim = input_tensor.shape
    scale_h = 1.0
    scale_w = 1.0

    if scale_attr is not None and len(scale_attr) >= 2:
        scale_h = scale_attr[0]
        scale_w = scale_attr[1]
    else:
        if out_h > 0 and out_w > 0:
            if data_format == "NCHW":
                h_axis = 2
                w_axis = 3
            elif data_format == "NHWC":
                h_axis = 1
                w_axis = 2

            scale_h = float(out_h) / float(in_dim[h_axis])
            scale_w = float(out_w) / float(in_dim[w_axis])

    outsize_tensor = None
    if inputs[2] is not None:
        outsize_tensor = network.add_concatenation(inputs[2]).get_output(0)

    scales = [1.0] * len(input_tensor.shape)
    if data_format == "NCHW":
        scales[1] = 1.0
        scales[2] = scale_h
        scales[3] = scale_w
    elif data_format == "NHWC":
        scales[1] = scale_h
        scales[2] = scale_w
        scales[3] = 1.0
    else:
        raise ValueError(
            f"Unsupported data format {data_format}, only NCHW or NHWC are supported."
        )
    if outsize_tensor is not None:
        outsize_itensors = []
        batch_dim = get_shape_tensor_element(network, input_shape_tensor, 0)
        outsize_itensors.append(batch_dim)
        if data_format == "NCHW":
            channel_dim = get_shape_tensor_element(
                network, input_shape_tensor, 1
            )
            outsize_itensors.append(channel_dim)
            outsize_itensors.append(outsize_tensor)
        elif data_format == "NHWC":
            channel_dim = get_shape_tensor_element(
                network, input_shape_tensor, 3
            )
            outsize_itensors.append(outsize_tensor)
            outsize_itensors.append(channel_dim)
        resize_layer.set_input(
            1, network.add_concatenation(outsize_itensors).get_output(0)
        )
    else:
        resize_layer.scales = scales

    return resize_layer.get_output(0)


@converter_registry.register(
    "pd_op.linear_interp", trt_version="trt_version_ge=8.0"
)
def linear_interp_converter(network, paddle_op, inputs):
    input_tensor = inputs[0]
    data_layout = paddle_op.attrs().get("data_format")
    interp_method = paddle_op.attrs().get("interp_method")
    align_corners = paddle_op.attrs().get("align_corners")
    out_w = paddle_op.attrs().get("out_w")
    scale_attr = paddle_op.attrs().get("scale")
    layer = network.add_resize(input_tensor)
    trt_major = get_trt_version_list()[0]
    trt_minor = get_trt_version_list()[1]
    trt_version_float = float(f"{trt_major}.{trt_minor}")

    if trt_version_float >= 8.6:
        layer.resize_mode = trt.InterpolationMode.LINEAR
    else:
        layer.resize_mode = trt.ResizeMode.LINEAR

    if align_corners:
        layer.coordinate_transformation = (
            trt.ResizeCoordinateTransformation.ALIGN_CORNERS
        )
    else:
        layer.coordinate_transformation = (
            trt.ResizeCoordinateTransformation.HALF_PIXEL
        )

    in_dim = input_tensor.shape
    scale_w = -1.0

    if scale_attr and len(scale_attr) > 0:
        scale_w = scale_attr[0]

    w_axis = 2 if data_layout == "NCHW" else 1

    if float(scale_w) > 0.0:
        out_w = int(in_dim[w_axis] * scale_w)

    outsize_tensor = None
    if len(inputs) > 1 and inputs[1] is not None:
        outsize_tensor = inputs[1]

    if outsize_tensor is None:
        if len(inputs) > 2 and inputs[2] is not None:
            outsize_tensor = inputs[2][0]

    if out_w > 0 and scale_w <= 0:
        scale_w = float(out_w) / float(in_dim[w_axis])

    scales = [1.0]
    if data_layout == "NCHW":
        scales.append(1.0)
        scales.append(scale_w)
    elif data_layout == "NHWC":
        scales.append(scale_w)
        scales.append(1.0)

    if outsize_tensor is not None:
        outsize_itensors = []
        input_shape = trt_shape(network, input_tensor)
        batch_dim = get_shape_tensor_element(network, input_shape, 0)
        outsize_itensors.append(batch_dim)

        if data_layout == "NCHW":
            channel_dim = get_shape_tensor_element(network, input_shape, 1)
            outsize_itensors.append(channel_dim)
            outsize_itensors.append(outsize_tensor)
        elif data_layout == "NHWC":
            outsize_itensors.append(outsize_tensor)
            channel_dim = get_shape_tensor_element(network, input_shape, 2)
            outsize_itensors.append(channel_dim)

        layer.set_input(1, trt_concat(network, outsize_itensors))
    else:
        layer.scales = scales

    return layer.get_output(0)
