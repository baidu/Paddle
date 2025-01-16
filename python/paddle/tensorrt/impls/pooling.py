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
    get_input_constant_value,
    replenish_layer_and_output,
)
from paddle.tensorrt.register import converter_registry


@converter_registry.register("pd_op.pool2d", trt_version="trt_version_ge=8.0")
def pool2d_converter(network, paddle_op, inputs):
    input_tensor = inputs[0]

    input_shape = paddle_op.operands()[0].source().shape
    input_dims = len(input_shape)

    global_pooling = paddle_op.attrs().get("global_pooling", False)
    pool_type = paddle_op.attrs().get("pooling_type", "avg")
    strides = paddle_op.attrs().get("strides", [1, 1])
    paddings = paddle_op.attrs().get("paddings", [0, 0])
    exclusive = paddle_op.attrs().get("exclusive", True)
    ceil_mode = paddle_op.attrs().get("ceil_mode", False)
    adaptive = paddle_op.attrs().get("adaptive", False)
    padding_algorithm = paddle_op.attrs().get("padding_algorithm", "EXPLICIT")

    if not paddle_op.attrs().get("kernel_size") and len(inputs) == 2:
        kernel_size = get_input_constant_value(paddle_op, inputs, 1)
        if kernel_size is None:
            raise Exception(
                "The defining op of kernel size must be builtin.constant/pd_op.full_int_array"
            )
    else:
        kernel_size = paddle_op.attrs().get("kernel_size", [1, 1])

    if pool_type == "max":
        nv_pool_type = trt.PoolingType.MAX
    elif pool_type == "avg":
        nv_pool_type = trt.PoolingType.AVERAGE
    else:
        raise ValueError(f"Unsupported pooling type: {pool_type}")

    if global_pooling or adaptive:
        paddings = [0, 0, 0, 0]

    if padding_algorithm == "VALID":
        paddings = [0, 0, 0, 0]

    if len(paddings) == 2:
        paddings = [paddings[0], paddings[0], paddings[1], paddings[1]]
    elif len(paddings) != 4:
        raise ValueError(f"Unsupported paddings size: {len(paddings)}")

    nv_paddings = trt.DimsHW(paddings[0], paddings[1])
    nv_ksize = trt.DimsHW(kernel_size[0], kernel_size[1])
    nv_strides = trt.DimsHW(strides[0], strides[1])

    layer = None
    g_pre_pad = trt.DimsHW(0, 0)
    g_post_pad = trt.DimsHW(0, 0)

    if (
        input_shape[input_dims - 2] > 0
        and input_shape[input_dims - 2] + paddings[0] + paddings[2]
        < kernel_size[0]
    ):
        g_post_pad.h = strides[0] - 1
    if (
        input_shape[input_dims - 1] > 0
        and input_shape[input_dims - 1] + paddings[1] + paddings[3]
        < kernel_size[1]
    ):
        g_post_pad.w = strides[1] - 1

    real_paddings = paddings.copy()
    for i in range(2):
        copy_pad = paddings[i]
        real_paddings.insert(2 * i + 1, copy_pad)

    if padding_algorithm == "SAME":
        for i in range(2):
            copy_pad = paddings[2 * i]
            paddings.insert(2 * i + 1, copy_pad)

        for i in range(2):
            out_size = (input_shape[2 + i] + strides[i] - 1) // strides[i]
            pad_sum = max(
                (out_size - 1) * strides[i]
                + kernel_size[i]
                - input_shape[2 + i],
                0,
            )
            pad_0 = pad_sum // 2
            pad_1 = pad_sum - pad_0
            paddings[2 * i] = pad_0
            paddings[2 * i + 1] = pad_1
        real_paddings = paddings.copy()

    paddings = [paddings[i] for i in range(len(paddings)) if i % 2 == 0]

    if padding_algorithm == "VALID":
        read_paddings = [0] * len(real_paddings)

    if adaptive and pool_type == "avg":
        output_h, output_w = kernel_size
        if output_h == 1 and output_w == 1:
            reduce_axes = (1 << (input_dims - 2)) | (1 << (input_dims - 1))
            reduce_layer = network.add_reduce(
                input=input_tensor,
                op=trt.ReduceOperation.AVG,
                axes=reduce_axes,
                keep_dims=True,
            )
            if reduce_layer is None:
                raise RuntimeError("Failed to add reduce layer in TensorRT.")
            layer = reduce_layer
        else:
            input_h = input_shape[input_dims - 2]
            input_w = input_shape[input_dims - 1]
            if input_h < 0 or input_w < 0:
                raise ValueError(
                    "Adaptive pooling with dynamic input dimensions is not supported."
                )

            stride_h = input_h // output_h
            stride_w = input_w // output_w
            kernel_h = input_h - (output_h - 1) * stride_h
            kernel_w = input_w - (output_w - 1) * stride_w

            if stride_h <= 0 or stride_w <= 0:
                raise ValueError(
                    "Calculated stride is non-positive, which is invalid."
                )

            nv_ksize = trt.DimsHW(kernel_h, kernel_w)
            nv_strides = trt.DimsHW(stride_h, stride_w)
            nv_paddings = trt.DimsHW(0, 0)
            pooling_layer = network.add_pooling_nd(
                input=input_tensor,
                type=nv_pool_type,
                window_size=nv_ksize,
            )
            if pooling_layer is None:
                raise RuntimeError("Failed to add pooling layer in TensorRT.")
            pooling_layer.stride_nd = nv_strides
            pooling_layer.padding_nd = nv_paddings
            pooling_layer.average_count_excludes_padding = exclusive
            layer = pooling_layer
    elif global_pooling and not adaptive:
        reduce_axes = (1 << (input_dims - 2)) | (1 << (input_dims - 1))
        reduce_layer = network.add_reduce(
            input=input_tensor,
            op=(
                trt.ReduceOperation.AVG
                if pool_type == "avg"
                else trt.ReduceOperation.MAX
            ),
            axes=reduce_axes,
            keep_dims=True,
        )
        if reduce_layer is None:
            raise RuntimeError("Failed to add reduce layer in TensorRT.")
        layer = reduce_layer
    elif not adaptive and not global_pooling and not ceil_mode:
        if padding_algorithm != "SAME" and (
            (g_post_pad.h > 0 and input_shape[input_dims - 2] > 0)
            or (g_post_pad.w > 0 and input_shape[input_dims - 1] > 0)
        ):
            pad_layer = network.add_padding_nd(
                input=input_tensor,
                pre_padding=(g_pre_pad.h, g_pre_pad.w),
                post_padding=(g_post_pad.h, g_post_pad.w),
            )
            if pad_layer is None:
                raise RuntimeError("Failed to add padding layer in TensorRT.")
            input_tensor = pad_layer.get_output(0)
        pooling_layer = network.add_pooling_nd(
            input=input_tensor, type=nv_pool_type, window_size=nv_ksize
        )
        if pooling_layer is None:
            raise RuntimeError("Failed to add pooling layer in TensorRT.")
        pooling_layer.stride_nd = nv_strides
        pooling_layer.padding_nd = nv_paddings
        pooling_layer.average_count_excludes_padding = exclusive
        if padding_algorithm == "SAME":
            pooling_layer.padding_mode = trt.PaddingMode.SAME_UPPER

        layer = pooling_layer
    elif not adaptive and not global_pooling and ceil_mode:
        pooling_layer = network.add_pooling_nd(
            input=input_tensor, type=nv_pool_type, window_size=nv_ksize
        )
        if pooling_layer is None:
            raise RuntimeError("Failed to add pooling layer in TensorRT.")
        pooling_layer.stride_nd = nv_strides
        pooling_layer.padding_nd = nv_paddings
        pooling_layer.average_count_excludes_padding = exclusive
        if padding_algorithm == "SAME":
            pooling_layer.padding_mode = trt.PaddingMode.SAME_UPPER
        else:
            pooling_layer.padding_mode = trt.PaddingMode.EXPLICIT_ROUND_UP
        layer = pooling_layer
    else:
        need_to_expand_dims = input_dims == 3
        if need_to_expand_dims:
            axes = [3]
            axes_tensor = network.add_constant(
                shape=(len(axes),),
                weights=np.array(axes, dtype=np.int32),
            ).get_output(0)
            unsqueeze_layer = network.add_unsqueeze(
                input=input_tensor, axes=axes_tensor
            )
            if unsqueeze_layer is None:
                raise RuntimeError("Failed to add unsqueeze layer in TensorRT.")
            input_tensor = unsqueeze_layer.get_output(0)
            input_shape = unsqueeze_layer.get_output(0).shape
            input_dims = len(input_shape)

        nbSpatialDims = len(kernel_size)
        if not (
            (nbSpatialDims == 1 and need_to_expand_dims)
            or nbSpatialDims == 2
            or nbSpatialDims == 3
        ):
            raise RuntimeError(
                f"kernel_shape ({nbSpatialDims}D) misaligns with the input tensor shape ({input_dims}D)."
            )

        begPadding = [0] * nbSpatialDims
        endPadding = [0] * nbSpatialDims

        if ceil_mode:
            padding_mode = trt.PaddingMode.EXPLICIT_ROUND_UP
        else:
            padding_mode = trt.PaddingMode.EXPLICIT_ROUND_DOWN

        countExcludePadding = True
        if pool_type == "avg":
            if exclusive:
                countExcludePadding = True
            else:
                countExcludePadding = False

        auto_pad = "NOTSET"
        if padding_algorithm == "SAME":
            auto_pad = "SAME_UPPER"
        elif padding_algorithm == "VALID":
            auto_pad = "VALID"

        if auto_pad != "SAME_LOWER" and auto_pad != "SAME_UPPER":
            ndim = len(paddings) // 2
            for i in range(nbSpatialDims):
                if i < ndim:
                    begPadding[i] = paddings[i]
                    endPadding[i] = paddings[i + ndim]
                else:
                    begPadding[i] = 0
                    endPadding[i] = 0
            if auto_pad == "EXPLICIT_ROUND_UP":
                padding_mode = trt.PaddingMode.EXPLICIT_ROUND_UP

        if nbSpatialDims == 2:
            nv_begPadding = trt.DimsHW(begPadding[0], begPadding[1])
            nv_endPadding = trt.DimsHW(endPadding[0], endPadding[1])

        pooling_layer = network.add_pooling_nd(
            input=input_tensor,
            type=nv_pool_type,
            window_size=nv_ksize,
        )
        if pooling_layer is None:
            raise RuntimeError("Failed to add pooling layer in TensorRT.")
        pooling_layer.stride_nd = nv_strides
        pooling_layer.pre_padding = nv_begPadding
        pooling_layer.post_padding = nv_endPadding
        pooling_layer.average_count_excludes_padding = countExcludePadding
        pooling_layer.padding_mode = padding_mode

        layer = pooling_layer

        if need_to_expand_dims:
            axes = [3]
            axes_tensor = network.add_constant(
                shape=(len(axes),),
                weights=np.array(axes, dtype=np.int32),
            ).get_output(0)
            squeeze_layer = network.add_squeeze(
                input=layer.get_output(0), axes=axes_tensor
            )
            if squeeze_layer is None:
                raise RuntimeError("Failed to add squeeze layer in TensorRT.")
            layer = squeeze_layer

    if layer is None:
        raise RuntimeError("Failed to create pooling layer in TensorRT.")

    replenish_layer_and_output(
        layer, paddle_op.name(), paddle_op.get_output_names()
    )
    return layer.get_output(0)
