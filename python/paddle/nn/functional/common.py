#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy

import paddle
from paddle import _C_ops, pir
from paddle.base.layer_helper import LayerHelper
from paddle.common_ops_import import Variable, default_main_program
from paddle.framework import (
    core,
    in_dynamic_mode,
    in_dynamic_or_pir_mode,
    in_pir_mode,
)
from paddle.tensor.creation import full
from paddle.utils import deprecated

from ...base.data_feeder import (
    check_dtype,
    check_type,
    check_variable_and_dtype,
)
from ...tensor import clip, concat, sqrt, sum
from ...tensor.creation import zeros

# TODO: define the common functions to build a neural network
from ...tensor.manipulation import squeeze, unsqueeze
from ...utils.layers_utils import NotSupportedTensorArgumentError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from typing_extensions import TypeAlias

    from paddle import Tensor
    from paddle._typing import (
        DataLayout1DVariant,
        DataLayout2D,
        DataLayout3D,
        DataLayoutND,
        ShapeLike,
        Size2,
        Size4,
    )
    from paddle.distributed.communication.group import Group

    _InterpolateMode: TypeAlias = Literal[
        'linear', 'area', 'nearest', 'bilinear', 'bicubic', 'trilinear'
    ]
    _DropoutMode: TypeAlias = Literal['upscale_in_train', 'downscale_in_infer']
    _PaddingTensorMode: TypeAlias = Literal[
        "zeros", "constant", "reflect", "replicate", "circular"
    ]
    _PaddingSizeMode: TypeAlias = Literal[  # noqa: PYI047
        'valid', 'same', 'VALID', 'SAME'
    ]

__all__ = []


def unfold(
    x: Tensor,
    kernel_sizes: Size2,
    strides: Size2 = 1,
    paddings: Size2 | Size4 = 0,
    dilations: Size2 = 1,
    name: str | None = None,
) -> Tensor:
    r"""

    Return a col buffer of sliding local blocks of input x, also known
    as im2col for batched 2D image tensors. For each block under the convolution filter,
    all element will be rearranged as a column. While the convolution filter sliding over
    the input feature map, a series of such columns will be formed.

    For each input :math:`x` with shape [N, C, H, W], the output shape [N, Cout, Lout]
    can be calculated as following.

    .. math::

        dkernel[0] &= dilations[0] \times (kernel\_sizes[0] - 1) + 1

        dkernel[1] &= dilations[1] \times (kernel\_sizes[1] - 1) + 1

        hout &= \frac{H + paddings[0] + paddings[2] - dkernel[0]}{strides[0]} + 1

        wout &= \frac{W + paddings[1] + paddings[3] - dkernel[1]}{strides[1]} + 1

        Cout &= C \times kernel\_sizes[0] \times kernel\_sizes[1]

        Lout &= hout \times wout


    Parameters:
        x(Tensor):              4-D Tensor, input tensor of format [N, C, H, W],
                                  data type can be float32 or float64
        kernel_sizes(int|list|tuple):   The size of convolution kernel, should be [k_h, k_w]
                                  or an integer k treated as [k, k].
        strides(int|list|tuple, optional):        The strides, should be [stride_h, stride_w]
                                  or an integer stride treated as [stride, stride].
                                  For default, strides will be [1, 1].
        paddings(int|list|tuple, optional):       The paddings of each dimension, should be
                                  [padding_top, padding_left, padding_bottom, padding_right]
                                  or [padding_h, padding_w] or an integer padding.
                                  If [padding_h, padding_w] was given, it will expanded to
                                  [padding_h, padding_w, padding_h, padding_w]. If an integer
                                  padding was given, [padding, padding, padding, padding] will
                                  be used. For default, paddings will be [0, 0, 0, 0]
        dilations(int|list|tuple, optional):      the dilations of convolution kernel, should be
                                  [dilation_h, dilation_w], or an integer dilation treated as
                                  [dilation, dilation]. For default, it will be [1, 1].
        name(str, optional): The default value is None.
                             Normally there is no need for user to set this property.
                             For more information, please refer to :ref:`api_guide_Name`


    Returns:
        Tensor, The tensor corresponding to the sliding local blocks.
        The output shape is [N, Cout, Lout] as described above.
        Cout is the  total number of values within each block,
        and Lout is the total number of such blocks.
        The data type of output is the same as the input :math:`x`

    Examples:

        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn.functional as F

            >>> x = paddle.randn((100,3,224,224))
            >>> y = F.unfold(x, [3, 3], 1, 1, 1)
    """

    helper = LayerHelper("unfold", **locals())

    check_variable_and_dtype(
        x, 'x', ['uint16', 'float16', 'float32', 'float64'], 'unfold'
    )

    assert len(x.shape) == 4, "input should be the format of [N, C, H, W]"

    if isinstance(kernel_sizes, int):
        kernel_sizes = [kernel_sizes, kernel_sizes]
    else:
        assert isinstance(kernel_sizes, (list, tuple)) and (
            len(kernel_sizes) == 2
        ), "kernel_sizes should either be an integer or a list/tuple of two integers"
        kernel_sizes = list(kernel_sizes)

    if isinstance(strides, int):
        strides = [strides, strides]
    else:
        assert isinstance(strides, (list, tuple)) and (
            len(strides) == 2
        ), "strides should either be an integer or a list/tuple of two integers"
        strides = list(strides)

    if isinstance(dilations, int):
        dilations = [dilations, dilations]
    else:
        assert isinstance(dilations, (list, tuple)) and (
            len(dilations) == 2
        ), "dilations should either be an integer or a list/tuple of two integers"
        dilations = list(dilations)

    if isinstance(paddings, int):
        paddings = [paddings] * 4
    elif isinstance(paddings, (list, tuple)):
        paddings = list(paddings)
        if len(paddings) == 2:
            paddings = paddings * 2
        elif len(paddings) == 4:
            pass
        else:
            raise ValueError(
                "paddings should either be an integer or a list/tuple of 2 or 4 integers"
            )
    else:
        raise ValueError(
            "Unexpected type of paddings, it should be either an integer or a list/tuple"
            "of 2 or 4 integers"
        )

    if in_dynamic_or_pir_mode():
        return _C_ops.unfold(x, kernel_sizes, strides, paddings, dilations)

    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type="unfold",
        inputs={"X": x},
        outputs={"Y": out},
        attrs={
            "kernel_sizes": kernel_sizes,
            "strides": strides,
            "paddings": paddings,
            "dilations": dilations,
        },
    )
    return out


def interpolate(
    x: Tensor,
    size: ShapeLike | None = None,
    scale_factor: ShapeLike | float | None = None,
    mode: _InterpolateMode = 'nearest',
    align_corners: bool = False,
    align_mode: int = 0,
    data_format: (
        DataLayout1DVariant | DataLayout2D | DataLayout3D | None
    ) = None,
    name: str | None = None,
) -> Tensor:
    """

    This API resizes a batch of images.

    The input must be a 3-D Tensor of the shape (num_batches, channels, in_w)
    or (num_batches, in_w, channels), or 4-D (num_batches, channels, in_h, in_w) or
    (num_batches, in_h, in_w, channels), or a 5-D Tensor of the shape
    (num_batches, channels, in_d, in_h, in_w) or (num_batches, in_d, in_h, in_w, channels),
    Where in_w is width of the input tensor, in_h is the height of the input tensor,
    in_d is the depth of the input tensor.
    and the resizing only applies on the three dimensions(depth, height and width).

    Supporting resample methods:

    - 'linear' : Linear interpolation
    - 'bilinear' : Bilinear interpolation
    - 'trilinear' : Trilinear interpolation
    - 'nearest' : Nearest neighbor interpolation
    - 'bicubic' : Bicubic interpolation
    - 'area': Area interpolation

    Linear interpolation is the method of using a line connecting two known quantities
    to determine the value of an unknown quantity between the two known quantities.

    Nearest neighbor interpolation is to perform nearest neighbor interpolation
    in both the 3rd dimension(in height direction) and the 4th dimension(in width
    direction) on input tensor.

    Bilinear interpolation is an extension of linear interpolation for
    interpolating functions of two variables (e.g. H-direction and
    W-direction in this op) on a rectilinear 2D grid. The key idea is
    to perform linear interpolation first in one direction, and then
    again in the other direction.

    Trilinear interpolation is an extension of linear interpolation for
    interpolating functions of three variables (e.g. D-direction,
    H-direction and W-direction in this op) on a rectilinear 3D grid.
    The linear interpolation is performed on three directions.
    align_corners and align_mode are optional parameters,the calculation method
    of interpolation can be selected by them.

    Bicubic interpolation is an extension of cubic interpolation for interpolating
    data points on a two-dimensional regular grid. The interpolated surface is
    smoother than corresponding surfaces obtained by bilinear interpolation or
    nearest-neighbor interpolation.

    Area interpolation is to perform area interpolation
    in both the 3rd dimension(in height direction) , the 4th dimension(in width
    direction) and the 5th dimension(in depth direction) on input tensor. Set to
    area will directly call `paddle.nn.functional.adaptive_avg_pool1d` or
    `paddle.nn.functional.adaptive_avg_pool2d` or `paddle.nn.functional.adaptive_avg_pool3d`.

    Example:

    .. code-block:: text

        # For scale_factor:
            if align_corners = True && out_size > 1 :
              scale_factor = (in_size-1.0)/(out_size-1.0)
            else:
              scale_factor = float(in_size/out_size)

        # Linear interpolation:
            if:
                align_corners = False , align_mode = 0
                input : (N,C,W_in)
                output: (N,C,W_out) where:
                W_out = (W_{in}+0.5) * scale_{factor} - 0.5
            else:
                input : (N,C,W_in)
                output: (N,C,W_out) where:
                W_out = W_{in} * scale_{factor}

        # Nearest neighbor interpolation:

              align_corners = False
              input : (N,C,H_in,W_in)
              output: (N,C,H_out,W_out) where:
              H_out = floor (H_{in} * scale_{factor})
              W_out = floor (W_{in} * scale_{factor})

        # Bilinear interpolation:
          if:
              align_corners = False , align_mode = 0
              input : (N,C,H_in,W_in)
              output: (N,C,H_out,W_out) where:
              H_out = (H_{in}+0.5) * scale_{factor} - 0.5
              W_out = (W_{in}+0.5) * scale_{factor} - 0.5
          else:
              input : (N,C,H_in,W_in)
              output: (N,C,H_out,W_out) where:
              H_out = H_{in} * scale_{factor}
              W_out = W_{in} * scale_{factor}

        # Bicubic interpolation:
          if:
              align_corners = False
              input : (N,C,H_in,W_in)
              output: (N,C,H_out,W_out) where:
              H_out = (H_{in}+0.5) * scale_{factor} - 0.5
              W_out = (W_{in}+0.5) * scale_{factor} - 0.5
          else:
              input : (N,C,H_in,W_in)
              output: (N,C,H_out,W_out) where:
              H_out = H_{in} * scale_{factor}
              W_out = W_{in} * scale_{factor}

        # Trilinear interpolation:
          if:
              align_corners = False , align_mode = 0
              input : (N,C,D_in,H_in,W_in)
              output: (N,C,D_out,H_out,W_out) where:
              D_out = (D_{in}+0.5) * scale_{factor} - 0.5
              H_out = (H_{in}+0.5) * scale_{factor} - 0.5
              W_out = (W_{in}+0.5) * scale_{factor} - 0.5
          else:
              input : (N,C,D_in,H_in,W_in)
              output: (N,C,D_out,H_out,W_out) where:
              D_out = D_{in} * scale_{factor}
              H_out = H_{in} * scale_{factor}
              W_out = W_{in} * scale_{factor}

    For details of linear interpolation, please refer to Wikipedia:
    https://en.wikipedia.org/wiki/Linear_interpolation.

    For details of nearest neighbor interpolation, please refer to Wikipedia:
    https://en.wikipedia.org/wiki/Nearest-neighbor_interpolation.

    For details of bilinear interpolation, please refer to Wikipedia:
    https://en.wikipedia.org/wiki/Bilinear_interpolation.

    For details of trilinear interpolation, please refer to Wikipedia:
    https://en.wikipedia.org/wiki/Trilinear_interpolation.

    For details of bicubic interpolation, please refer to Wikipedia:
    https://en.wikipedia.org/wiki/Bicubic_interpolation

    Parameters:
        x (Tensor): 3-D, 4-D or 5-D Tensor, its data type is float32, float64, or uint8, its data format is
             specified by :attr:`data_format`. If :attr:`data_format` is not provided, the data format will
             be presumed according to its dimension. See details in :attr:`data_format`.
        size (list|tuple|Tensor|None): Output shape of image resize
             layer, the shape is (out_w, ) when input is a 3-D Tensor, the shape is (out_h, out_w)
             when input is a 4-D Tensor and is (out_d, out_h, out_w) when input is a 5-D Tensor.
             Default: None. If a list/tuple, each element can be an integer or a Tensor of shape: [1] or [].
             If a Tensor, its dimensions size should be a 1.
        scale_factor (float|Tensor|list|tuple|None): The multiplier for the input height or width. At
             least one of :attr:`size` or :attr:`scale_factor` must be set.
             And :attr:`size` has a higher priority than :attr:`scale_factor`.Has to match input size if it is either
             a list or a tuple or a Tensor. If a list/tuple, each element can be an integer or a Tensor of shape: [1] or [].
             Default: None.
        mode (str): The resample method. It supports 'linear', 'area', 'nearest', 'bilinear',
                       'bicubic' and 'trilinear' currently. Default: 'nearest'
        align_corners(bool) :  An optional bool, If True, the centers of the 4 corner pixels of the
                               input and output tensors are aligned, preserving the values at the
                               corner pixels.This only has an effect when 'linear', 'bilinear', 'bicubic' or 'trilinear'.
                               Default: False
        align_mode(int)  :  An optional for linear/bilinear/trilinear interpolation. Refer to the formula in the example above,
                            it can be \'0\' for src_idx = scale_factor*(dst_index+0.5)-0.5 , can be \'1\' for
                            src_idx = scale_factor*dst_index.
        data_format (str, optional): Specify the data format of the input, and the data format of
             the output will be consistent with that of the input. An optional string from:`"NCW"`,
             `"NWC"`,  `"NCHW"`, `"NHWC"`, `"NCDHW"`, `"NDHWC"`. The default value is None.
             When :attr:`data_format` is not specified, it will be automatically inferred from the
             input dimension of :attr:`x`. When :attr:`x` is a 3-D Tensor, :attr:`data_format` will be
             set to `"NCW"`; When :attr:`x` is a 4-D Tensor, :attr:`data_format` will be set to
             `"NCHW"`; When :attr:`x` is a 5-D Tensor, :attr:`data_format` will be set to `"NCDHW"`.
             When it is `"NCHW"`, the data should be stored in the order of:
             `[batch_size, input_channels, input_height, input_width]`. When it is `"NCDHW"`, the
             data should be stored in the order of: `[batch_size, input_channels, input_depth, input_height, input_width]`.
        name(str, optional): The default value is None.
                             Normally there is no need for user to set this property.
                             For more information, please refer to :ref:`api_guide_Name`
    Returns:
        A 3-D, 4-D or 5-D Tensor, with the same data format of the input :attr:`x`.


    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn.functional as F

            >>> input_data = paddle.randn(shape=(2,3,6,10)).astype(paddle.float32)
            >>> output_1 = F.interpolate(x=input_data, size=[12,12])
            >>> print(output_1.shape)
            [2, 3, 12, 12]
            >>> # given scale
            >>> output_2 = F.interpolate(x=input_data, scale_factor=[2,1])
            >>> print(output_2.shape)
            [2, 3, 12, 10]
            >>> # bilinear interp
            >>> output_3 = F.interpolate(x=input_data, scale_factor=[2,1], mode="bilinear")
            >>> print(output_2.shape)
            [2, 3, 12, 10]
    """
    if data_format is None:
        dim_size = len(x.shape)
        if dim_size == 3:
            data_format = 'NCW'
        elif dim_size == 4:
            data_format = 'NCHW'
        elif dim_size == 5:
            data_format = 'NCDHW'
        else:
            raise ValueError(
                f"The dimension of the input tensor should only be 3-D, 4-D or 5-D, but the received dimension is {dim_size}."
            )
    data_format = data_format.upper()
    resample = mode.upper()
    resample_type = mode.lower()

    resample_methods = [
        'LINEAR',
        'BILINEAR',
        'TRILINEAR',
        'NEAREST',
        'BICUBIC',
        'AREA',
    ]
    if resample not in resample_methods:
        raise ValueError(
            "The 'resample' of image_resize can only be 'area', 'linear', 'bilinear', 'trilinear', "
            " 'bicubic' or 'nearest' currently."
        )

    if resample in ['LINEAR'] and len(x.shape) != 3:
        raise ValueError("'linear' only support 3-D tensor.")

    if resample in ['NEAREST'] and len(x.shape) != 4 and len(x.shape) != 5:
        raise ValueError("'NEAREST' only support 4-D  or 5-D tensor.")

    if resample in ['BILINEAR', 'BICUBIC'] and len(x.shape) != 4:
        raise ValueError("'bilinear' and 'bicubic' only support 4-D tensor.")
    if resample == 'TRILINEAR' and len(x.shape) != 5:
        raise ValueError("'trilinear'only support 5-D tensor.")

    if size is None and scale_factor is None:
        raise ValueError("One of size and scale_factor must not be None.")

    if isinstance(size, (tuple, list)) and (len(size) != x.ndim - 2):
        raise ValueError(
            'The x and size should satisfy rank(x) - 2 == len(size).'
        )

    if isinstance(size, (Variable, paddle.pir.Value)):
        size = size.cast("int32")  # static mode only support int32
        if size.ndim != 1:
            raise ValueError(
                f"If size is a tensor, it's rank must be 1, but received {size.ndim}."
            )
        if size.shape[0] != x.ndim - 2:
            raise ValueError(
                'The x and size should satisfy rank(x) - 2 == size.shape[0].'
            )

    if not isinstance(align_corners, bool):
        raise TypeError("Attr align_corners should be a bool value")

    if align_mode != 0 and align_mode != 1:
        raise ValueError("align_mode can only be 0 or 1")
    if align_corners != 0 and resample == 'NEAREST':
        raise ValueError(
            "align_corners option can only be set with the interpolating modes: linear | bilinear | bicubic | trilinear"
        )

    if resample == 'AREA':
        if isinstance(size, (list, tuple, Variable, paddle.pir.Value)):
            if len(size) == 0:
                raise ValueError("output size can not be empty")
        if size is None:
            raise ValueError("output size can not be None in AREA mode")
        if len(x.shape) == 3:
            return paddle.nn.functional.adaptive_avg_pool1d(x, size)
        elif len(x.shape) == 4:
            print("size :", size)
            return paddle.nn.functional.adaptive_avg_pool2d(x, size)
        elif len(x.shape) == 5:
            return paddle.nn.functional.adaptive_avg_pool3d(x, size)
    helper = LayerHelper(f'{resample_type}_interp_v2', **locals())
    if len(x.shape) == 3 and data_format not in ['NCW', 'NWC']:
        raise ValueError(
            "Got wrong value for param `data_format`: "
            + data_format
            + " received but only `NCW` or `NWC` supported for 3-D input."
        )
    elif len(x.shape) == 4 and data_format not in ['NCHW', 'NHWC']:
        raise ValueError(
            "Got wrong value for param `data_format`: "
            + data_format
            + " received but only `NCHW` or `NHWC` supported for 4-D input."
        )
    elif len(x.shape) == 5 and data_format not in ['NCDHW', 'NDHWC']:
        raise ValueError(
            "Got wrong value for param `data_format`: "
            + data_format
            + " received but only `NCDHW` or `NDHWC` supported for 5-D input."
        )

    def _is_list_or_tuple_(data):
        return isinstance(data, (list, tuple))

    if data_format == 'NCHW' or data_format == 'NCDHW' or data_format == 'NCW':
        data_layout = 'NCHW'
    if data_format == 'NHWC' or data_format == 'NDHWC' or data_format == 'NWC':
        data_layout = 'NHWC'

    if resample == 'NEAREST':
        align_corners = False

    inputs = {"X": x}
    attrs = {
        "out_d": -1,
        "out_h": -1,
        "out_w": -1,
        "interp_method": resample_type,
        "align_corners": align_corners,
        "align_mode": align_mode,
        "data_layout": data_layout,
    }

    out_shape = size
    scale = scale_factor
    if out_shape is not None and scale is not None:
        raise ValueError("Only one of size or scale_factor should be defined.")
    if out_shape is not None:
        if (
            isinstance(out_shape, (Variable, paddle.pir.Value))
            and not in_dynamic_mode()
        ):
            out_shape.stop_gradient = True
            inputs['OutSize'] = out_shape
        else:
            if in_dynamic_mode():
                if isinstance(out_shape, Variable):
                    out_shape = list(out_shape.numpy(False))
                else:
                    out_shape = list(out_shape)

                for i, dim in enumerate(out_shape):
                    if isinstance(dim, Variable):
                        out_shape[i] = dim.item()
            if not (_is_list_or_tuple_(out_shape)):
                raise TypeError("size should be a list or tuple or Variable.")
            # Validate the shape
            contain_var = False
            for dim_idx, dim_size in enumerate(out_shape):
                if isinstance(dim_size, (Variable, paddle.pir.Value)):
                    contain_var = True
                    continue
                assert (
                    dim_size > 0
                ), "Each dimension size given in out_shape must be greater than 0."

            if contain_var:
                new_size_tensor = []
                size_list = []
                for dim in out_shape:
                    if isinstance(dim, (Variable, paddle.pir.Value)):
                        dim.stop_gradient = True
                        new_size_tensor.append(dim)
                        size_list.append(-1)
                    else:
                        assert isinstance(dim, int)
                        if in_pir_mode():
                            temp_out = paddle.tensor.fill_constant(
                                [1], 'int32', dim, force_cpu=True
                            )
                        else:
                            temp_out = (
                                helper.create_variable_for_type_inference(
                                    'int32'
                                )
                            )
                            paddle.tensor.fill_constant(
                                [1], 'int32', dim, force_cpu=True, out=temp_out
                            )
                        new_size_tensor.append(temp_out)
                        size_list.append(dim)
                inputs['SizeTensor'] = new_size_tensor

            if len(x.shape) == 3:
                if len(out_shape) != 1:
                    raise ValueError(
                        "size length should be 2 for input 3-D tensor"
                    )
                if contain_var:
                    attrs['out_w'] = size_list[0]
                else:
                    out_shape = list(map(int, out_shape))
                    attrs['out_w'] = out_shape[0]
            if len(x.shape) == 4:
                if len(out_shape) != 2:
                    raise ValueError(
                        "size length should be 2 for " "input 4-D tensor."
                    )
                if contain_var:
                    attrs['out_h'] = size_list[0]
                    attrs['out_w'] = size_list[1]
                else:
                    out_shape = list(map(int, out_shape))
                    attrs['out_h'] = out_shape[0]
                    attrs['out_w'] = out_shape[1]
            if len(x.shape) == 5:
                if len(out_shape) != 3:
                    raise ValueError(
                        "size length should be 3 for " "input 5-D tensor."
                    )
                if contain_var:
                    attrs['out_d'] = size_list[0]
                    attrs['out_h'] = size_list[1]
                    attrs['out_w'] = size_list[2]
                else:
                    out_shape = list(map(int, out_shape))
                    attrs['out_d'] = out_shape[0]
                    attrs['out_h'] = out_shape[1]
                    attrs['out_w'] = out_shape[2]

    else:
        if in_dynamic_mode() and isinstance(scale, Variable):
            if scale.shape == []:
                scale = float(scale)
            else:
                scale = list(scale.numpy())
        if isinstance(scale, (Variable, paddle.pir.Value)):
            scale.stop_gradient = True
            inputs["Scale"] = scale
        elif isinstance(scale, (float, int, numpy.ndarray)):
            if scale <= 0:
                raise ValueError("Attr(scale) should be greater than zero.")
            scale_list = []
            for i in range(len(x.shape) - 2):
                scale_list.append(scale)
            attrs['scale'] = list(map(float, scale_list))
        elif isinstance(scale, (list, tuple)):
            if len(scale) != len(x.shape) - 2:
                raise ValueError(
                    f"scale_shape length should be {len(x.shape) - 2} for "
                    f"input {len(x.shape)}-D tensor."
                )
            for value in scale:
                if value <= 0:
                    raise ValueError("Attr(scale) should be greater than zero.")
            attrs['scale'] = list(map(float, scale))
        else:
            raise TypeError(
                "Attr(scale)'s type should be float, int, list, tuple, or Tensor."
            )

    if in_dynamic_or_pir_mode():
        attr_list = []
        for k, v in attrs.items():
            attr_list.append(k)
            attr_list.append(v)
        dy_attr = tuple(attr_list)

        if resample_type == "linear":
            out = _C_ops.linear_interp(
                x,
                inputs['OutSize'] if 'OutSize' in inputs else None,
                inputs['SizeTensor'] if 'SizeTensor' in inputs else None,
                inputs['Scale'] if 'Scale' in inputs else None,
                attrs['data_layout'],
                attrs['out_d'],
                attrs['out_h'],
                attrs['out_w'],
                attrs['scale'] if 'scale' in attrs else [],
                attrs['interp_method'],
                attrs['align_corners'],
                attrs['align_mode'],
            )
        elif resample_type == "bilinear":
            out = _C_ops.bilinear_interp(
                x,
                inputs['OutSize'] if 'OutSize' in inputs else None,
                inputs['SizeTensor'] if 'SizeTensor' in inputs else None,
                inputs['Scale'] if 'Scale' in inputs else None,
                attrs['data_layout'],
                attrs['out_d'],
                attrs['out_h'],
                attrs['out_w'],
                attrs['scale'] if 'scale' in attrs else [],
                attrs['interp_method'],
                attrs['align_corners'],
                attrs['align_mode'],
            )
        elif resample_type == "trilinear":
            out = _C_ops.trilinear_interp(
                x,
                inputs['OutSize'] if 'OutSize' in inputs else None,
                inputs['SizeTensor'] if 'SizeTensor' in inputs else None,
                inputs['Scale'] if 'Scale' in inputs else None,
                attrs['data_layout'],
                attrs['out_d'],
                attrs['out_h'],
                attrs['out_w'],
                attrs['scale'] if 'scale' in attrs else [],
                attrs['interp_method'],
                attrs['align_corners'],
                attrs['align_mode'],
            )
        elif resample_type == "nearest":
            out = _C_ops.nearest_interp(
                x,
                inputs['OutSize'] if 'OutSize' in inputs else None,
                inputs['SizeTensor'] if 'SizeTensor' in inputs else None,
                inputs['Scale'] if 'Scale' in inputs else None,
                attrs['data_layout'],
                attrs['out_d'],
                attrs['out_h'],
                attrs['out_w'],
                attrs['scale'] if 'scale' in attrs else [],
                attrs['interp_method'],
                attrs['align_corners'],
                attrs['align_mode'],
            )
        elif resample_type == "bicubic":
            out = _C_ops.bicubic_interp(
                x,
                inputs['OutSize'] if 'OutSize' in inputs else None,
                inputs['SizeTensor'] if 'SizeTensor' in inputs else None,
                inputs['Scale'] if 'Scale' in inputs else None,
                attrs['data_layout'],
                attrs['out_d'],
                attrs['out_h'],
                attrs['out_w'],
                attrs['scale'] if 'scale' in attrs else [],
                attrs['interp_method'],
                attrs['align_corners'],
                attrs['align_mode'],
            )
        return out

    dtype = helper.input_dtype(input_param_name='x')

    out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type=f'{resample_type}_interp_v2',
        inputs=inputs,
        outputs={"Out": out},
        attrs=attrs,
    )
    return out


def upsample(
    x: Tensor,
    size: ShapeLike | None = None,
    scale_factor: ShapeLike | None = None,
    mode: _InterpolateMode = 'nearest',
    align_corners: bool = False,
    align_mode: int = 0,
    data_format: (
        DataLayout1DVariant | DataLayout2D | DataLayout3D | None
    ) = None,
    name: str | None = None,
) -> Tensor:
    """

    This API resizes a batch of images.

    The input must be a 3-D Tensor of the shape (num_batches, channels, in_w)
    or (num_batches, in_w, channels), or 4-D (num_batches, channels, in_h, in_w) or
    (num_batches, in_h, in_w, channels), or a 5-D Tensor of the shape
    (num_batches, channels, in_d, in_h, in_w) or (num_batches, in_d, in_h, in_w, channels),
    Where in_w is width of the input tensor, in_h is the height of the input tensor,
    in_d is the depth of the input tensor.
    and the resizing only applies on the three dimensions(depth, height and width).

    Supporting resample methods:
    - 'linear' : Linear interpolation
    - 'bilinear' : Bilinear interpolation
    - 'trilinear' : Trilinear interpolation
    - 'nearest' : Nearest neighbor interpolation
    - 'bicubic' : Bicubic interpolation
    - 'area': Area interpolation

    Linear interpolation is the method of using a line connecting two known quantities
    to determine the value of an unknown quantity between the two known quantities.

    Nearest neighbor interpolation is to perform nearest neighbor interpolation
    in both the 3rd dimension(in height direction) and the 4th dimension(in width
    direction) on input tensor.
    Bilinear interpolation is an extension of linear interpolation for
    interpolating functions of two variables (e.g. H-direction and
    W-direction in this op) on a rectilinear 2D grid. The key idea is
    to perform linear interpolation first in one direction, and then
    again in the other direction.

    Bicubic interpolation is an extension of cubic interpolation for interpolating
    data points on a two-dimensional regular grid. The interpolated surface is
    smoother than corresponding surfaces obtained by bilinear interpolation or
    nearest-neighbor interpolation.

    Trilinear interpolation is an extension of linear interpolation for
    interpolating functions of three variables (e.g. D-direction,
    H-direction and W-direction in this op) on a rectilinear 3D grid.

    The linear interpolation is performed on three directions.
    align_corners and align_mode are optional parameters,the calculation method
    of interpolation can be selected by them.

    Area interpolation is to perform area interpolation
    in both the 3rd dimension(in height direction) , the 4th dimension(in width
    direction) and the 5th dimension(in depth direction) on input tensor. Set to
    area will directly call `paddle.nn.functional.adaptive_avg_pool1d` or
    `paddle.nn.functional.adaptive_avg_pool2d` or `paddle.nn.functional.adaptive_avg_pool3d`.

    Example:
        .. code-block:: text

            For scale_factor:
                if align_corners = True && out_size > 1 :
                scale_factor = (in_size-1.0)/(out_size-1.0)
                else:
                scale_factor = float(in_size/out_size)
            Linear interpolation:
                if:
                    align_corners = False , align_mode = 0
                    input : (N,C,W_in)
                    output: (N,C,W_out) where:
                    W_out = (W_{in}+0.5) * scale_{factor} - 0.5
                else:
                    input : (N,C,W_in)
                    output: (N,C,W_out) where:
                    W_out = W_{in} * scale_{factor}
            Nearest neighbor interpolation:
            if:
                align_corners = False
                input : (N,C,H_in,W_in)
                output: (N,C,H_out,W_out) where:
                H_out = floor (H_{in} * scale_{factor})
                W_out = floor (W_{in} * scale_{factor})
            else:
                align_corners = True
                input : (N,C,H_in,W_in)
                output: (N,C,H_out,W_out) where:
                H_out = round(H_{in} * scale_{factor})
                W_out = round(W_{in} * scale_{factor})

            Bilinear interpolation:
            if:
                align_corners = False , align_mode = 0
                input : (N,C,H_in,W_in)
                output: (N,C,H_out,W_out) where:
                H_out = (H_{in}+0.5) * scale_{factor} - 0.5
                W_out = (W_{in}+0.5) * scale_{factor} - 0.5
            else:
                input : (N,C,H_in,W_in)
                output: (N,C,H_out,W_out) where:
                H_out = H_{in} * scale_{factor}
                W_out = W_{in} * scale_{factor}
            Bicubic interpolation:
            if:
                align_corners = False
                input : (N,C,H_in,W_in)
                output: (N,C,H_out,W_out) where:
                H_out = (H_{in}+0.5) * scale_{factor} - 0.5
                W_out = (W_{in}+0.5) * scale_{factor} - 0.5
            else:
                input : (N,C,H_in,W_in)
                output: (N,C,H_out,W_out) where:
                H_out = H_{in} * scale_{factor}
                W_out = W_{in} * scale_{factor}
            Trilinear interpolation:
            if:
                align_corners = False , align_mode = 0
                input : (N,C,D_in,H_in,W_in)
                output: (N,C,D_out,H_out,W_out) where:
                D_out = (D_{in}+0.5) * scale_{factor} - 0.5
                H_out = (H_{in}+0.5) * scale_{factor} - 0.5
                W_out = (W_{in}+0.5) * scale_{factor} - 0.5
            else:
                input : (N,C,D_in,H_in,W_in)
                output: (N,C,D_out,H_out,W_out) where:
                D_out = D_{in} * scale_{factor}
                H_out = H_{in} * scale_{factor}
                W_out = W_{in} * scale_{factor}

    For details of linear interpolation, please refer to Wikipedia:
    https://en.wikipedia.org/wiki/Linear_interpolation.

    For details of nearest neighbor interpolation, please refer to Wikipedia:
    https://en.wikipedia.org/wiki/Nearest-neighbor_interpolation.

    For details of bilinear interpolation, please refer to Wikipedia:
    https://en.wikipedia.org/wiki/Bilinear_interpolation.

    For details of bicubic interpolation, please refer to Wikipedia:
    https://en.wikipedia.org/wiki/Bicubic_interpolation

    For details of trilinear interpolation, please refer to Wikipedia:
    https://en.wikipedia.org/wiki/Trilinear_interpolation.

    Parameters:
        x (Tensor): 3-D, 4-D or 5-D Tensor, its data type is float32, float64, or uint8, its data format is
             specified by :attr:`data_format`. If :attr:`data_format` is not provided, the data format will
             be presumed according to its dimension. See details in :attr:`data_format`.
        size (list|tuple|Tensor|None, optional): Output shape of image resize
             layer, the shape is (out_w, ) when input is a 3-D Tensor, the shape is (out_h, out_w)
             when input is a 4-D Tensor and is (out_d, out_h, out_w) when input is a 5-D Tensor.
             Default: None. If a list/tuple, each element can be an integer or a Tensor of shape: [1] or [].
             If a Tensor , its dimensions size should be a 1.
        scale_factor (float|Tensor|list|tuple|None, optional): The multiplier for the input height or width. At
             least one of :attr:`size` or :attr:`scale_factor` must be set.
             And :attr:`size` has a higher priority than :attr:`scale_factor`.Has to match input size if
             it is either a list or a tuple or a Tensor. If a list/tuple, each element can be an integer or a Tensor of shape: [1] or [].
             Default: None.
        mode (str, optional): The resample method. It supports 'linear', 'nearest', 'bilinear', 'area',
                       'bicubic' and 'trilinear' currently. Default: 'nearest'
        align_corners(bool, optional) :  An optional bool, If True, the centers of the 4 corner pixels of the
                               input and output tensors are aligned, preserving the values at the
                               corner pixels.
                               Default: False
        align_mode(int, optional)  :  An optional for linear/bilinear/trilinear interpolation. Refer to the formula in the example above,
                            it can be \'0\' for src_idx = scale_factor*(dst_index+0.5)-0.5 , can be \'1\' for
                            src_idx = scale_factor*dst_index.
        data_format (str, optional): Specify the data format of the input, and the data format of
             the output will be consistent with that of the input. An optional string from:`"NCW"`,
             `"NWC"`,  `"NCHW"`, `"NHWC"`, `"NCDHW"`, `"NDHWC"`. The default value is None.
             When :attr:`data_format` is not specified, it will be automatically inferred from the
             input dimension of :attr:`x`. When :attr:`x` is a 3-D Tensor, :attr:`data_format` will be
             set to `"NCW"`; When :attr:`x` is a 4-D Tensor, :attr:`data_format` will be set to
             `"NCHW"`; When :attr:`x` is a 5-D Tensor, :attr:`data_format` will be set to `"NCDHW"`.
             When it is `"NCHW"`, the data should be stored in the order of:
             `[batch_size, input_channels, input_height, input_width]`. When it is `"NCDHW"`, the
             data should be stored in the order of: `[batch_size, input_channels, input_depth, input_height, input_width]`.
        name(str, optional): The default value is None.
                             Normally there is no need for user to set this property.
                             For more information, please refer to :ref:`api_guide_Name`

    Returns:
        A 3-D, 4-D or 5-D Tensor, with the same data format of the input :attr:`x`.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn as nn

            >>> input_data = paddle.randn(shape=(2,3,6,10)).astype(paddle.float32)
            >>> upsample_out = paddle.nn.Upsample(size=[12,12])
            >>> output = upsample_out(x=input_data)
            >>> print(output.shape)
            [2, 3, 12, 12]

    """

    return interpolate(
        x, size, scale_factor, mode, align_corners, align_mode, data_format
    )


def bilinear(
    x1: Tensor,
    x2: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    name: str | None = None,
) -> Tensor:
    """

    This layer performs bilinear on two inputs.
    See :ref:`api_paddle_nn_Bilinear` for details and output shape.

    Parameters:
        x1 (Tensor): the first input tensor, it's data type should be float32, float64.
        x2 (Tensor): the second input tensor, it's data type should be float32, float64.
        weight (Tensor): The learnable weights of this layer, shape is [out_features, in1_features, in2_features].
        bias (Tensor, optional): The learnable bias(Bias) of this layer, shape is [1, out_features]. If it is set to None, no bias will be added to the output units. The default value is None.
        name (str, optional): The default value is None. Normally there is no need for user
            to set this property. For more information, please refer to :ref:`api_guide_Name`. Default: None.

    Returns:
        Tensor: A 2-D Tensor of shape [batch_size, out_features].

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn.functional as F

            >>> x1 = paddle.randn((5, 5)).astype(paddle.float32)
            >>> x2 = paddle.randn((5, 4)).astype(paddle.float32)
            >>> w = paddle.randn((1000, 5, 4)).astype(paddle.float32)
            >>> b = paddle.randn((1, 1000)).astype(paddle.float32)
            >>> result = F.bilinear(x1, x2, w, b)
            >>> print(result.shape)
            [5, 1000]
    """

    if in_dynamic_or_pir_mode():
        return _C_ops.bilinear(x1, x2, weight, bias)
    else:
        check_variable_and_dtype(x1, 'x1', ['float32', 'float64'], 'bilinear')
        check_variable_and_dtype(x2, 'x2', ['float32', 'float64'], 'bilinear')

        inputs = {"X": x1, "Y": x2, "Weight": weight}
        if bias is not None:
            inputs["Bias"] = bias

        helper = LayerHelper("bilinear", **locals())
        out = helper.create_variable_for_type_inference(dtype=x1.dtype)

        helper.append_op(
            type="bilinear_tensor_product", inputs=inputs, outputs={"Out": out}
        )

        return out


def dropout(
    x: Tensor,
    p: float = 0.5,
    axis: int | Sequence[int] | None = None,
    training: bool = True,
    mode: _DropoutMode = "upscale_in_train",
    name: str | None = None,
) -> Tensor:
    r"""
    Dropout is a regularization technique for reducing overfitting by preventing
    neuron co-adaption during training. The dropout operator randomly sets the
    outputs of some units to zero, while upscale others according to the given
    dropout probability.

    Args:
        x (Tensor): The input tensor. The data type is float16, float32 or float64.
        p (float|int, optional): Probability of setting units to zero. Default: 0.5.
        axis (int|list|tuple, optional): The axis along which the dropout is performed. Default: None.
        training (bool, optional): A flag indicating whether it is in train phrase or not. Default: True.
        mode(str, optional): ['upscale_in_train'(default) | 'downscale_in_infer'].

            1. upscale_in_train (default), upscale the output at training time

                - train: :math:`out = input \times \frac{mask}{(1.0 - dropout\_prob)}`
                - inference: :math:`out = input`

            2. downscale_in_infer, downscale the output at inference

                - train: :math:`out = input \times mask`
                - inference: :math:`out = input \times (1.0 - dropout\_prob)`

        name (str, optional): Name for the operation, Default: None. For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        A Tensor representing the dropout, has same shape and data type as `x` .


    Examples:
        We use ``p=0.5`` in the following description for simplicity.

        1. When ``axis=None`` , this is commonly used dropout, which dropout each element of x randomly.

        ..  code-block:: text

            Let's see a simple case when x is a 2d tensor with shape 2*3:
            [[1 2 3]
             [4 5 6]]
            we generate mask with the same shape as x, which is 2*3. The value of mask is
            sampled from a Bernoulli distribution randomly. For example, we may get such mask:
            [[0 1 0]
             [1 0 1]]
            So the output is obtained from elementwise multiply of x and mask:
            [[0 2 0]
             [4 0 6]]
            Using default setting, i.e. ``mode='upscale_in_train'`` ,
            if in training phase, the final upscale output is:
            [[0 4 0 ]
             [8 0 12]]
            if in test phase, the output is the same as input:
            [[1 2 3]
             [4 5 6]]
            we can also set ``mode='downscale_in_infer'`` , then
            if in training phase, the final output is:
            [[0 2 0]
             [4 0 6]]
            if in test phase, the scale output is:
            [[0.5 1.  1.5]
             [2.  2.5 3. ]]



        2. When ``axis!=None`` , this is useful for dropping whole channels from an image or sequence.

        ..  code-block:: text

            Let's see the simple case when x is a 2d tensor with shape 2*3 again:
            [[1 2 3]
             [4 5 6]]
            (1) If ``axis=0`` , this means the dropout is only performed in axis `0` .
                we generate mask with the shape 2*1. Only in axis `0` the value is randomly selected.
                For example, we may get such mask:
                [[1]
                 [0]]
                The output is obtained from elementwise multiply of x and mask. Doing that the mask will be
                broadcast from 2*1 to 2*3:
                [[1 1 1]
                 [0 0 0]]
                and the result after elementwise multiply is:
                [[1 2 3]
                 [0 0 0]]
                then we can do upscale or downscale according to the setting of other arguments.
            (2) If ``axis=1`` , this means the dropout is only performed in axis `1` .
                we generate mask with the shape 1*3. Only in axis `1` the value is randomly selected.
                For example, we may get such mask:
                [[1 0 1]]
                Doing elementwise multiply the mask will be broadcast from 1*3 to 2*3:
                [[1 0 1]
                 [1 0 1]]
                and the result after elementwise multiply is:
                [[1 0 3]
                 [4 0 6]]
            (3) What about ``axis=[0, 1]`` ? This means the dropout is performed in all axes of x,
                which is the same case as default setting ``axis=None`` .
            (4) You may note that logically `axis=None` means the dropout is performed in none axis of x,
                We generate mask with the shape 1*1. Whole input is randomly selected or dropped.
                For example, we may get such mask:
                [[0]]
                Doing elementwise multiply the mask will be broadcast from 1*1 to 2*3:
                [[0 0 0]
                 [0 0 0]]
                and the result after elementwise multiply is:
                [[0 0 0]
                 [0 0 0]]
                Actually this is not what we want because all elements may set to zero~

        When x is a 4d tensor with shape `NCHW`, where `N` is batch size, `C` is the number of channels, H and W are the height and width of the feature, we can set ``axis=[0,1]`` and the dropout will be performed in channel `N` and `C`, `H` and `W` is tied, i.e. paddle.nn.dropout(x, p, axis=[0,1]) . Please refer to ``paddle.nn.functional.dropout2d`` for more details.
        Similarly, when x is a 5d tensor with shape `NCDHW`, where `D` is the depth of the feature, we can set ``axis=[0,1]`` to perform dropout3d. Please refer to ``paddle.nn.functional.dropout3d`` for more details.

        .. code-block:: python

            >>> import paddle
            >>> paddle.seed(2023)
            >>> x = paddle.to_tensor([[1,2,3], [4,5,6]]).astype(paddle.float32)
            >>> y_train = paddle.nn.functional.dropout(x, 0.5)
            >>> y_test = paddle.nn.functional.dropout(x, 0.5, training=False)
            >>> y_0 = paddle.nn.functional.dropout(x, axis=0)
            >>> y_1 = paddle.nn.functional.dropout(x, axis=1)
            >>> y_01 = paddle.nn.functional.dropout(x, axis=[0,1])
            >>> print(x)
            Tensor(shape=[2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[1., 2., 3.],
             [4., 5., 6.]])
            >>> print(y_train)
            Tensor(shape=[2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[2., 4., 0.],
            [8., 0., 0.]])
            >>> print(y_test)
            Tensor(shape=[2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[1., 2., 3.],
             [4., 5., 6.]])
            >>> print(y_0)
            Tensor(shape=[2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[2., 4., 6.],
             [8. , 10., 12.]])
            >>> print(y_1)
            Tensor(shape=[2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[2. , 4. , 6. ],
             [8. , 10., 12.]])
            >>> print(y_01)
            Tensor(shape=[2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[0., 0., 6.],
             [0., 0., 0.]])
    """
    if not isinstance(p, (float, int, Variable, pir.Value)):
        raise TypeError("p argument should be a number or Variable")

    if isinstance(p, (int, float)):
        # fast return for p == 0
        if p == 0:
            return x
        elif p < 0 or p > 1:
            raise ValueError("p argument should between 0 and 1")
    if mode not in ('downscale_in_infer', 'upscale_in_train'):
        raise ValueError(
            "mode argument should be 'downscale_in_infer' or 'upscale_in_train'"
        )
    if axis and not isinstance(axis, (int, list, tuple)):
        raise TypeError("datatype of axis argument should be int or list")

    if axis is None:  # commonly used dropout
        seed = None
        mode = (
            'downgrade_in_infer' if mode == 'downscale_in_infer' else mode
        )  # semantic transfer

        if in_dynamic_or_pir_mode():
            if paddle.static.default_main_program().random_seed != 0:
                seed = paddle.static.default_main_program().random_seed
            out = _C_ops.dropout(
                x,
                None,
                p,
                not training,
                mode,
                seed if seed is not None else 0,
                seed is not None,
            )

            return out
        else:
            helper = LayerHelper('dropout', **locals())
            check_variable_and_dtype(
                x, 'x', ['float16', 'uint16', 'float32', 'float64'], 'dropout'
            )

            out = helper.create_variable_for_type_inference(dtype=x.dtype)
            mask = helper.create_variable_for_type_inference(
                dtype=core.VarDesc.VarType.UINT8, stop_gradient=True
            )

            def get_attrs(prog, dropout_prob, is_test, seed):
                if (seed is None or seed == 0) and prog.random_seed != 0:
                    seed = prog.random_seed

                if isinstance(
                    dropout_prob, Variable
                ) and not dropout_prob.shape != [1]:
                    raise TypeError(
                        f"Required p.shape == [1] if type(p) is Variable, but received p.shape = {p.shape}"
                    )
                attrs = {
                    'dropout_prob': dropout_prob,
                    'is_test': is_test,
                    'fix_seed': seed is not None,
                    'seed': seed if seed is not None else 0,
                    'dropout_implementation': mode,
                }
                return attrs

            attrs = get_attrs(helper.main_program, p, not training, seed)

            helper.append_op(
                type='dropout',
                inputs={'X': [x]},
                outputs={'Out': [out], 'Mask': [mask]},
                attrs=attrs,
            )
            return out
    else:  # sometimes called dropout_nd #TODO: optimize with c++
        if not in_dynamic_mode():
            check_variable_and_dtype(
                x, 'x', ['float16', 'uint16', 'float32', 'float64'], 'dropout'
            )
        dtype = x.dtype
        keep_prob = 1 - p
        if training:
            if in_dynamic_mode() and p == 1.0:
                return paddle.scale(x, scale=0.0)
            elif in_pir_mode() and isinstance(p, (float, int)) and p == 1.0:
                return paddle.scale(x, scale=0.0)

            scale_input = (
                paddle.scale(x, scale=1 / keep_prob)
                if mode == 'upscale_in_train'
                else x
            )

            # get mask shape
            input_shape = x.shape
            if not in_dynamic_mode():
                input_shape_tensor = paddle.shape(x)
            drop_axes = [axis] if isinstance(axis, int) else list(axis)
            if min(drop_axes) < 0 or max(drop_axes) > len(input_shape) - 1:
                raise ValueError(
                    f"axis value should be greater than or equal to 0 and less than dimensions of x:{len(input_shape)}, but get axis value:{max(drop_axes)} "
                )
            if len(drop_axes) > len(input_shape):
                raise ValueError(
                    f"length of axis should not be greater than dimensions of x:{len(input_shape)}, but get length of axis: {len(drop_axes)}"
                )
            mask_shape = [1] * len(input_shape)
            if not in_dynamic_mode():
                for i in drop_axes:
                    mask_shape[i] = input_shape_tensor[i]
            else:
                for i in drop_axes:
                    mask_shape[i] = input_shape[i]

            # get mask
            random_tensor = paddle.uniform(
                mask_shape, dtype='float32', min=0.0, max=1.0
            )
            p = full(shape=[1], fill_value=p, dtype='float32')
            keep_mask = paddle.greater_equal(random_tensor, p)

            scale_input = paddle.cast(scale_input, dtype)
            keep_mask = paddle.cast(keep_mask, dtype)
            ret = paddle.multiply(scale_input, keep_mask, name=name)
            return ret
        else:  # test
            ret = (
                paddle.scale(x, scale=keep_prob)
                if mode == 'downscale_in_infer'
                else x
            )
            return ret


def dropout2d(
    x: Tensor,
    p: float = 0.5,
    training: bool = True,
    data_format: DataLayout2D = 'NCHW',
    name: str | None = None,
) -> Tensor:
    """
    Randomly zero out entire channels (in the batched input 4d tensor with the shape `NCHW` ,
    a channel is a 2D feature map with the shape `HW` ). Each channel will be zeroed out independently
    on every forward call with probability `p` using samples from a Bernoulli distribution.

    See :ref:`api_paddle_nn_functional_dropout` for more details.

    Args:
        x (Tensor):  The input is 4-D Tensor with shape [N, C, H, W] or [N, H, W, C].
                     The data type is float16, float32 or float64.
        p (float, optional): Probability of setting units to zero. Default: 0.5.
        training (bool, optional): A flag indicating whether it is in train phrase or not. Default: True.
        data_format (str, optional): Specify the data format of the input, and the data format of the output will be consistent with that of the input. An optional string from `NCHW` or `NHWC` . When it is `NCHW` , the data is stored in the order of: [batch_size, input_channels, input_height, input_width]. Default: `NCHW` .
        name (str, optional): Name for the operation, Default: None. For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        A Tensor representing the dropout2d, has same shape and data type as `x` .


    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.seed(1)
            >>> x = paddle.randn(shape=(2, 3, 4, 5)).astype(paddle.float32)
            >>> y_train = paddle.nn.functional.dropout2d(x)  #train
            >>> y_test = paddle.nn.functional.dropout2d(x, training=False) #test
            >>> for i in range(2):
            ...     for j in range(3):
            ...         print(x[i,j,:,:])
            ...         print(y_train[i,j,:,:]) # may all 0
            ...         print(y_test[i,j,:,:])
            Tensor(shape=[4, 5], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[-0.30557564,  0.11855337,  0.41220093, -0.09968963,  1.50014710],
             [ 1.24004936, -0.92485696,  0.08612321,  1.15149164, -0.09276631],
             [ 1.22873247, -1.46587241, -1.30802727,  0.19496460,  1.73776841],
             [ 0.40092674,  0.67630458,  0.72265440,  1.31720388, -1.41899264]])
            Tensor(shape=[4, 5], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[-0.61115128,  0.23710674,  0.82440186, -0.19937925,  3.00029421],
             [ 2.48009872, -1.84971392,  0.17224643,  2.30298328, -0.18553263],
             [ 2.45746493, -2.93174481, -2.61605453,  0.38992921,  3.47553682],
             [ 0.80185348,  1.35260916,  1.44530880,  2.63440776, -2.83798528]])
            Tensor(shape=[4, 5], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[-0.30557564,  0.11855337,  0.41220093, -0.09968963,  1.50014710],
             [ 1.24004936, -0.92485696,  0.08612321,  1.15149164, -0.09276631],
             [ 1.22873247, -1.46587241, -1.30802727,  0.19496460,  1.73776841],
             [ 0.40092674,  0.67630458,  0.72265440,  1.31720388, -1.41899264]])
            Tensor(shape=[4, 5], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[ 0.88350385, -1.14767575,  0.51043051, -0.10051888, -0.61305630],
             [-0.12084112,  0.48506257, -1.13189507,  0.62806708, -0.80003673],
             [ 0.51513153, -0.08890446,  0.22753835,  0.11557858,  0.78117645],
             [ 1.47505593,  0.84618902, -0.38528305, -1.05887091,  0.16592593]])
            Tensor(shape=[4, 5], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[ 1.76700771, -2.29535151,  1.02086103, -0.20103776, -1.22611260],
             [-0.24168225,  0.97012514, -2.26379013,  1.25613415, -1.60007346],
             [ 1.03026307, -0.17780893,  0.45507669,  0.23115715,  1.56235290],
             [ 2.95011187,  1.69237804, -0.77056611, -2.11774182,  0.33185187]])
            Tensor(shape=[4, 5], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[ 0.88350385, -1.14767575,  0.51043051, -0.10051888, -0.61305630],
             [-0.12084112,  0.48506257, -1.13189507,  0.62806708, -0.80003673],
             [ 0.51513153, -0.08890446,  0.22753835,  0.11557858,  0.78117645],
             [ 1.47505593,  0.84618902, -0.38528305, -1.05887091,  0.16592593]])
            Tensor(shape=[4, 5], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[-1.46668839, -0.38117948,  1.18678427,  0.38740095,  0.29117522],
             [-0.13538910, -0.14527084, -0.04912176, -0.26063353,  0.23640174],
             [ 0.45643106,  0.60587281, -1.03242552, -0.45319262, -1.57911122],
             [-0.08732958, -0.75898546,  0.14563090, -1.73751652, -0.89109969]])
            Tensor(shape=[4, 5], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[-0., -0., 0. , 0. , 0. ],
             [-0., -0., -0., -0., 0. ],
             [0. , 0. , -0., -0., -0.],
             [-0., -0., 0. , -0., -0.]])
            Tensor(shape=[4, 5], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[-1.46668839, -0.38117948,  1.18678427,  0.38740095,  0.29117522],
             [-0.13538910, -0.14527084, -0.04912176, -0.26063353,  0.23640174],
             [ 0.45643106,  0.60587281, -1.03242552, -0.45319262, -1.57911122],
             [-0.08732958, -0.75898546,  0.14563090, -1.73751652, -0.89109969]])
            Tensor(shape=[4, 5], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[-0.32110816, -0.76044011,  0.34456784, -0.39410326,  0.37896338],
             [ 0.52747023,  0.72711533,  0.29204839,  0.72493637,  0.31128070],
             [ 0.58046782, -1.78499067, -1.67504823, -0.38590902, -0.26243693],
             [ 0.96669912,  0.43670532, -0.38109761,  0.78405094, -2.17882323]])
            Tensor(shape=[4, 5], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[-0., -0., 0. , -0., 0. ],
             [0. , 0. , 0. , 0. , 0. ],
             [0. , -0., -0., -0., -0.],
             [0. , 0. , -0., 0. , -0.]])
            Tensor(shape=[4, 5], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[-0.32110816, -0.76044011,  0.34456784, -0.39410326,  0.37896338],
             [ 0.52747023,  0.72711533,  0.29204839,  0.72493637,  0.31128070],
             [ 0.58046782, -1.78499067, -1.67504823, -0.38590902, -0.26243693],
             [ 0.96669912,  0.43670532, -0.38109761,  0.78405094, -2.17882323]])
            Tensor(shape=[4, 5], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[ 0.17168395,  0.45112833,  0.63307828,  2.38763475, -1.27247131],
             [ 0.56171960, -1.09584677,  0.38300961, -0.57512099,  0.31011426],
             [-0.95336407, -1.04852903, -0.21312937, -0.53549880, -0.00074209],
             [ 2.22819090,  1.12403083, -0.04198794, -1.51167727, -0.42699185]])
            Tensor(shape=[4, 5], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[0. , 0. , 0. , 0. , -0.],
             [0. , -0., 0. , -0., 0. ],
             [-0., -0., -0., -0., -0.],
             [0. , 0. , -0., -0., -0.]])
            Tensor(shape=[4, 5], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[ 0.17168395,  0.45112833,  0.63307828,  2.38763475, -1.27247131],
             [ 0.56171960, -1.09584677,  0.38300961, -0.57512099,  0.31011426],
             [-0.95336407, -1.04852903, -0.21312937, -0.53549880, -0.00074209],
             [ 2.22819090,  1.12403083, -0.04198794, -1.51167727, -0.42699185]])
            Tensor(shape=[4, 5], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[ 0.62503546, -0.20989063, -0.22046235, -0.38679042, -1.02590704],
             [ 1.04561794,  1.08428383, -0.52219963, -1.56003857,  0.89213932],
             [-0.16578521,  0.14524542, -0.45563069,  0.48180851,  1.35843253],
             [ 1.07669640, -0.84535235, -1.18651557,  0.79144061, -0.45565742]])
            Tensor(shape=[4, 5], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[0. , -0., -0., -0., -0.],
             [0. , 0. , -0., -0., 0. ],
             [-0., 0. , -0., 0. , 0. ],
             [0. , -0., -0., 0. , -0.]])
            Tensor(shape=[4, 5], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[ 0.62503546, -0.20989063, -0.22046235, -0.38679042, -1.02590704],
             [ 1.04561794,  1.08428383, -0.52219963, -1.56003857,  0.89213932],
             [-0.16578521,  0.14524542, -0.45563069,  0.48180851,  1.35843253],
             [ 1.07669640, -0.84535235, -1.18651557,  0.79144061, -0.45565742]])
    """
    input_shape = x.shape
    if len(input_shape) != 4:
        raise ValueError(
            f"dimensions of x should be 4, but received {len(input_shape)} != 4"
        )

    if data_format not in ["NCHW", "NHWC"]:
        raise ValueError(
            "Attr(data_format) should be 'NCHW' or 'NHWC'. Received "
            f"Attr(data_format): {data_format}."
        )

    return dropout(
        x,
        p=p,
        axis=[0, 1] if data_format == 'NCHW' else [0, 3],
        training=training,
        mode="upscale_in_train",
        name=name,
    )


def dropout3d(
    x: Tensor,
    p: float = 0.5,
    training: bool = True,
    data_format: DataLayout3D = 'NCDHW',
    name: str | None = None,
) -> Tensor:
    """
    Randomly zero out entire channels (in the batched input 5d tensor with the shape `NCDHW` ,
    a channel is a 3D feature map with the shape `DHW` ). Each channel will be zeroed out independently
    on every forward call with probability `p` using samples from a Bernoulli distribution.

    See :ref:`api_paddle_nn_functional_dropout` for more details.

    Args:
        x (Tensor):  The input is 5-D Tensor with shape [N, C, D, H, W] or [N, D, H, W, C].
                     The data type is float32 or float64.
        p (float, optional): Probability of setting units to zero. Default: 0.5.
        training (bool, optional): A flag indicating whether it is in train phrase or not. Default: True.
        data_format (str, optional): Specify the data format of the input, and the data format of the output will be consistent with that of the input. An optional string from ``NCDHW`` or ``NDHWC``. When it is ``NCDHW`` , the data is stored in the order of: [batch_size, input_channels, input_depth, input_height, input_width]. Default: ``NCDHW`` .
        name (str, optional): Name for the operation, Default: None. For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        A Tensor representing the dropout3d, has same shape and data type with `x` .


    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.randn(shape=(2, 3, 4, 5, 6)).astype(paddle.float32)
            >>> y_train = paddle.nn.functional.dropout3d(x)  #train
            >>> y_test = paddle.nn.functional.dropout3d(x, training=False) #test
            >>> print(x[0,0,:,:,:])
            >>> print(y_train[0,0,:,:,:]) # may all 0
            >>> print(y_test[0,0,:,:,:])

    """

    input_shape = x.shape
    if len(input_shape) != 5:
        raise ValueError(
            f"dimensions of x should be 5, but received {len(input_shape)} != 5"
        )

    if data_format not in ["NCDHW", "NDHWC"]:
        raise ValueError(
            "Attr(data_format) should be 'NCDHW' or 'NDHWC'. Received "
            f"Attr(data_format): {data_format}."
        )

    return dropout(
        x,
        p=p,
        axis=[0, 1] if data_format == 'NCDHW' else [0, 4],
        training=training,
        mode="upscale_in_train",
        name=name,
    )


def _feature_alpha_dropout_impl(
    x: Tensor,
    feature_dropout: bool,
    p: float,
    training: bool = True,
    name: str | None = None,
) -> Tensor:
    if not isinstance(p, (float, int)):
        raise TypeError("p argument should be a float or int")
    if p < 0 or p > 1:
        raise ValueError("p argument should between 0 and 1")

    if not in_dynamic_mode():
        check_variable_and_dtype(
            x, 'x', ['float16', 'uint16', 'float32', 'float64'], 'alpha_dropout'
        )

    if training:
        if p == 1:
            return paddle.scale(x, scale=0.0)
        # get transformation params
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        alpha_p = -alpha * scale
        a = ((1 - p) * (1 + p * alpha_p**2)) ** -0.5
        b = -a * alpha_p * p

        dtype = x.dtype
        if not feature_dropout:
            input_shape = x.shape
        else:
            if x.ndim < 2:
                raise ValueError(
                    'Feature alpha dropout needs at least 2D input.'
                )
            input_shape = list(x.shape[:2]) + [1] * len(x.shape[2:])

        # get mask
        random_tensor = paddle.uniform(
            input_shape, dtype='float32', min=0.0, max=1.0
        )
        p = full(shape=input_shape, fill_value=p, dtype='float32')
        keep_mask = paddle.greater_equal(random_tensor, p)
        keep_mask = paddle.cast(keep_mask, dtype)
        drop_mask = paddle.subtract(
            full(shape=input_shape, fill_value=1.0, dtype=dtype), keep_mask
        )

        # apply mask
        b = full(shape=input_shape, fill_value=b, dtype=dtype)
        y = paddle.add(
            paddle.multiply(x, keep_mask),
            paddle.scale(drop_mask, scale=alpha_p),
        )
        res = paddle.add(paddle.scale(y, scale=a), b, name=name)
        return res
    else:  # test
        return x


def alpha_dropout(
    x: Tensor,
    p: float = 0.5,
    training: bool = True,
    name: str | None = None,
) -> Tensor:
    """
    Alpha Dropout is a type of Dropout that maintains the self-normalizing property.
    For an input with zero mean and unit standard deviation, the output of Alpha Dropout
    maintains the original mean and standard deviation of the input.
    Alpha Dropout fits well to SELU activate function by randomly setting activations to the negative saturation value.

    Args:
        x (Tensor): The input tensor. The data type is bfloat16, float16, float32 or float64.
        p (float | int, optional): Probability of setting units to zero. Default 0.5.
        training (bool, optional): A flag indicating whether it is in train phrase or not. Default True.
        name (str | None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: A Tensor representing the dropout, has same shape and data type as `x`.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.seed(1)
            >>> x = paddle.to_tensor([[-1, 1], [-1, 1]]).astype(paddle.float32)
            >>> y_train = paddle.nn.functional.alpha_dropout(x, 0.5)
            >>> y_test = paddle.nn.functional.alpha_dropout(x, 0.5, training=False)
            >>> print(y_train)
            Tensor(shape=[2, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[-0.77919382,  1.66559887],
            [-0.10721093, -0.77919382]])
            >>> print(y_test)
            Tensor(shape=[2, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[-1.,  1.],
            [-1.,  1.]])
    """
    return _feature_alpha_dropout_impl(
        x, feature_dropout=False, p=p, training=training, name=name
    )


def feature_alpha_dropout(
    x: Tensor,
    p: float = 0.5,
    training: bool = True,
    name: str | None = None,
) -> Tensor:
    """
    A channel is a feature map, Feature Alpha Dropout randomly masks out entire channels.
    Alpha Dropout is a type of Dropout that maintains the self-normalizing property.
    For an input with zero mean and unit standard deviation, the output of Alpha Dropout
    maintains the original mean and standard deviation of the input.
    Alpha Dropout fits well to SELU activate function by randomly setting activations to the negative saturation value.

    Args:
        x (Tensor): The input tensor. The data type is bfloat16, float16, float32 or float64.
        p (float | int, optional): Probability of setting units to zero. Default 0.5.
        training (bool, optional): A flag indicating whether it is in train phrase or not. Default True.
        name (str | None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: A Tensor representing the dropout, has same shape and data type as `x`.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.seed(1)
            >>> x = paddle.to_tensor([[-1, 1], [-1, 1]]).astype(paddle.float32)
            >>> y_train = paddle.nn.functional.feature_alpha_dropout(x, 0.5)
            >>> y_test = paddle.nn.functional.feature_alpha_dropout(x, 0.5, training=False)
            >>> print(y_train)
            Tensor(shape=[2, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[-0.77919382,  1.66559887],
            [-0.10721093, -0.77919382]])
            >>> print(y_test)
            Tensor(shape=[2, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[-1.,  1.],
            [-1.,  1.]])
    """
    return _feature_alpha_dropout_impl(
        x, feature_dropout=True, p=p, training=training, name=name
    )


def pad(
    x: Tensor,
    pad: ShapeLike,
    mode: _PaddingTensorMode = 'constant',
    value: float = 0.0,
    data_format: DataLayoutND | None = None,
    pad_from_left_axis: bool = True,
    name: str | None = None,
) -> Tensor:
    """
    Pad tensor according to ``'pad'`` and ``'mode'``.

    Note:
        1. Denote ``'x'``'s dimension as N (same in the following). If mode is ``'constant'``, the length
        of ``'pad'`` can be any even number less than or equal to 2*N.

        2. When mode is ``'constant'``, and ``'pad'`` is a list or tuple, and the length of ``'pad'`` is not
        equal to 2*(N - 2):
        2.1. If the length of ``'pad'`` is 2*N, the order of padding can be customized by ``'pad_from_left_axis'``.
        if ``'pad_from_left_axis'`` is True, then the padding order will be started from the first dimension of
        ``'x'`` and moving backward according to ``'pad'``; else if ``'pad_from_left_axis'`` is False, then the
        padding order will be started from the last dimension of ``'x'`` and moving forward according to ``'pad'``.
        2.2. Otherwise, the padding will be started from the last dimension.

        3. When mode is any of ``'reflect'``, ``'replicate'``, ``'circular'``, or ``'pad'`` is a tensor, or the
        length of ``'pad'`` is 2*(N - 2), and the dimension of ``'x'`` only supports 3-D, 4-D and 5-D.
        In these cases, input ``'x'`` will be padded on [D, H, W] axes according to ``'data_format'``. It will pad
        from the last dimension to the first dimension of [D, H, W] axes.
        Specifically, if N = 3, then the pad has the form (pad_left, pad_right); if N = 4, then the pad has the form
        (pad_left, pad_right, pad_top, pad_bottom); if N = 5, then the pad has the form (pad_left, pad_right,
        pad_top, pad_bottom, pad_front, pad_back).

        4. If mode is ``'reflect'``, pad[0] and pad[1] must be no greater than width-1. The height and depth
        dimension has the same condition.

    Args:
        x (Tensor): The input tensor with data type float32, float64, int32, int64, complex64 or complex128.
        pad (Tensor|list[int]|tuple[int]): The padding size with data type int. Refer to Note for details.
        mode (str, optional): Four modes: ``'constant'`` (default), ``'reflect'``, ``'replicate'``, ``'circular'``. Default is ``'constant'``.

           - 'constant' mode, uses a constant value to pad the input tensor.
           - 'reflect' mode, uses reflection of the input boundaries to pad the input tensor.
           - 'replicate' mode, uses input boundaries to pad the input tensor.
           - 'circular' mode, uses circular input to pad the input tensor.

        value (float, optional): The value to fill the padded areas in 'constant' mode . Default is :math:`0.0`.
        data_format (str, optional): An string from: ``'NCL'``, ``'NLC'``, ``'NHWC'``, ``'NCHW'``, ``'NCDHW'``, ``'NDHWC'``. Specify the data format of
           the input data when: 1. mode is any of ``'reflect'``, ``'replicate'`` or ``'circular'``; or 2. the input ``'pad'`` is a tensor;
           or 3. the length of ``'pad'`` is ``2*(x.ndim - 2)``. The default value is None, which means it will be automatically inferred from the
           input dimension of ``'x'``. When ``'x'`` is a 3-D Tensor, data_format will be set to ``'NCL'``; When ``'x'`` is a 4-D Tensor,
           data_format will be set to ``'NCHW'``; When ``'x'`` is a 5-D Tensor, data_format will be set to ``'NCDHW'``.
        pad_from_left_axis (bool, optional): The parameter is only valid when mode is ``'constant'`` and the input ``'pad'`` is
           length of ``'pad'`` is ``2*x.ndim``, the order of padding can be customized. If True, the padding will be started from
           the first axis of ``'x'``; if False, it will be started from the last axis of ``'x'``. Default: True.
        name (str|None, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: ``'None'``.

    Returns:
        Tensor, a Tensor padded according to pad and mode and data type is same as input.

    Example:

        .. code-block:: text

            x = [[[[[1., 2., 3.],
                    [4., 5., 6.]]]]]

            Case 0:
                pad = [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                mode = 'constant'
                value = 0
                pad_from_left_axis = True
                Out = [[[[[0., 0., 0.],
                          [1., 2., 3.],
                          [4., 5., 6.],
                          [0., 0., 0.]]]]]
                Out.shape = [1, 1, 1, 4, 3]

            Case 1:
                pad = [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                mode = 'constant'
                value = 0
                pad_from_left_axis = False
                Out = [[[[[0., 0., 0.],
                          [0., 0., 0.]]],
                        [[[1., 2., 3.],
                          [4., 5., 6.]]],
                        [[[0., 0., 0.],
                          [0., 0., 0.]]]]]
                Out.shape = [1, 3, 1, 2, 3]

            Case 3:
                pad = [1, 0, 0, 1],
                mode = 'constant'
                value = 0
                Out = [[[[[0., 1., 2., 3.],
                          [0., 4., 5., 6.],
                          [0., 0., 0., 0.]]]]]
                Out.shape = [1, 1, 1, 3, 4]

            Case 4:
                pad = [2, 2, 1, 1, 0, 0],
                mode = 'constant'
                value = 0
                Out = [[[[[0. 0. 0. 0. 0. 0. 0.]
                          [0. 0. 1. 2. 3. 0. 0.]
                          [0. 0. 4. 5. 6. 0. 0.]
                          [0. 0. 0. 0. 0. 0. 0.]]]]]
                Out.shape = [1, 1, 1, 4, 7]

            Case 5:
                pad = [2, 2, 1, 1, 0, 0],
                mode = 'reflect'
                Out = [[[[[6. 5. 4. 5. 6. 5. 4.]
                          [3. 2. 1. 2. 3. 2. 1.]
                          [6. 5. 4. 5. 6. 5. 4.]
                          [3. 2. 1. 2. 3. 2. 1.]]]]]
                Out.shape = [1, 1, 1, 4, 7]

            Case 6:
                pad = [2, 2, 1, 1, 0, 0],
                mode = 'replicate'
                Out = [[[[[1. 1. 1. 2. 3. 3. 3.]
                          [1. 1. 1. 2. 3. 3. 3.]
                          [4. 4. 4. 5. 6. 6. 6.]
                          [4. 4. 4. 5. 6. 6. 6.]]]]]
                Out.shape = [1, 1, 1, 4, 7]

            Case 7:
                pad = [2, 2, 1, 1, 0, 0],
                mode = 'circular'
                Out = [[[[[5. 6. 4. 5. 6. 4. 5.]
                          [2. 3. 1. 2. 3. 1. 2.]
                          [5. 6. 4. 5. 6. 4. 5.]
                          [2. 3. 1. 2. 3. 1. 2.]]]]]
                Out.shape = [1, 1, 1, 4, 7]

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn.functional as F

            >>> # example 1
            >>> x_shape = (1, 1, 3)
            >>> x = paddle.arange(paddle.prod(paddle.to_tensor(x_shape)), dtype="float32").reshape(x_shape) + 1
            >>> y = F.pad(x, [0, 0, 0, 0, 2, 3], value=1, mode='constant', data_format="NCL")
            >>> print(y)
            Tensor(shape=[1, 1, 8], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[[1., 1., 1., 2., 3., 1., 1., 1.]]])

            >>> # example 2
            >>> x_shape = (1, 1, 3)
            >>> x = paddle.arange(paddle.prod(paddle.to_tensor(x_shape)), dtype="float32").reshape(x_shape) + 1
            >>> y = F.pad(x, [2, 3], value=1, mode='constant', data_format="NCL")
            >>> print(y)
            Tensor(shape=[1, 1, 8], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[[1., 1., 1., 2., 3., 1., 1., 1.]]])

            >>> # example 3
            >>> x_shape = (1, 1, 2, 3)
            >>> x = paddle.arange(paddle.prod(paddle.to_tensor(x_shape)), dtype="float32").reshape(x_shape) + 1
            >>> y = F.pad(x, [1, 2, 1, 1], value=1, mode='circular')
            >>> print(y)
            Tensor(shape=[1, 1, 4, 6], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[[[6., 4., 5., 6., 4., 5.],
               [3., 1., 2., 3., 1., 2.],
               [6., 4., 5., 6., 4., 5.],
               [3., 1., 2., 3., 1., 2.]]]])

            >>> # example 4
            >>> x_shape = (1, 1, 3)
            >>> x = paddle.arange(paddle.prod(paddle.to_tensor(x_shape)), dtype="float32").reshape(x_shape) + 1
            >>> y = F.pad(x, [1, 0, 0, 1, 0, 0], value=0, mode='constant', pad_from_left_axis=True)
            >>> print(y)
            Tensor(shape=[2, 2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[[0., 0., 0.],
              [0., 0., 0.]],
             [[1., 2., 3.],
              [0., 0., 0.]]])

            >>> # example 5
            >>> x_shape = (1, 1, 3)
            >>> x = paddle.arange(paddle.prod(paddle.to_tensor(x_shape)), dtype="float32").reshape(x_shape) + 1
            >>> y = F.pad(x, [1, 0, 0, 1, 0, 0], value=0, mode='constant', pad_from_left_axis=False)
            >>> print(y)
            Tensor(shape=[1, 2, 4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[[0., 1., 2., 3.],
              [0., 0., 0., 0.]]])

            >>> # example 6
            >>> x_shape = (1, 1, 3)
            >>> x = paddle.arange(paddle.prod(paddle.to_tensor(x_shape)), dtype="float32").reshape(x_shape) + 1
            >>> y = F.pad(x, [1, 0, 0, 1], value=0, mode='constant')
            >>> print(y)
            Tensor(shape=[1, 2, 4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[[0., 1., 2., 3.],
              [0., 0., 0., 0.]]])
    """
    assert mode in [
        'reflect',
        'replicate',
        'constant',
        'circular',
    ], f"mode should be one of constant, reflect, replicate, circular, but got {mode}."

    x_dim = len(x.shape)

    if (
        mode == "constant"
        and isinstance(pad, (list, tuple))
        and len(pad) != (x_dim - 2) * 2
    ):
        paddings = pad
        pad_value = value

        padding_len = len(paddings)
        # pad the length of paddings to 2*x_dim
        if padding_len < 2 * x_dim:
            pad_len_for_paddings = 2 * x_dim - padding_len
            paddings = paddings + ([0] if isinstance(pad, list) else (0,)) * (
                pad_len_for_paddings
            )

        # since the kernel pad from left axis, if we want to pad from right axis, we need to reverse the paddings
        if not (len(pad) == x_dim * 2 and pad_from_left_axis):
            paddings = [
                paddings[i - 1] if i % 2 == 1 else paddings[i + 1]
                for i in range(2 * x_dim - 1, -1, -1)
            ]

        if in_dynamic_mode():
            out = _C_ops.pad(x, paddings, float(pad_value))
            return out

        if in_pir_mode():
            if isinstance(paddings, paddle.pir.Value):
                raise NotSupportedTensorArgumentError(
                    "`pad` must be list of int, but received Tensor.",
                    "pad",
                )
            if isinstance(paddings, list) and any(
                not isinstance(i, int) for i in pad
            ):
                raise NotSupportedTensorArgumentError(
                    f"`pad` must be list of int, but received list of {type(pad[0])}.",
                    "pad",
                )

            if isinstance(pad_value, paddle.pir.Value):
                return _C_ops.pad(x, paddings, pad_value)
            else:
                return _C_ops.pad(x, paddings, float(pad_value))

        check_variable_and_dtype(
            x,
            'x',
            [
                'float16',
                'float32',
                'float64',
                'int32',
                'int64',
                'complex64',
                'complex128',
                'uint16',
            ],
            "pad",
        )

        check_type(pad_value, 'pad_value', (float, int, Variable), 'pad')
        if isinstance(pad_value, int):
            pad_value = float(pad_value)

        helper = LayerHelper('pad', **locals())
        dtype = helper.input_dtype(input_param_name='x')
        out = helper.create_variable_for_type_inference(dtype)
        helper.append_op(
            type='pad',
            inputs={'X': x},
            outputs={'Out': out},
            attrs={'paddings': paddings, 'pad_value': pad_value},
        )
        return out

    assert x_dim in [
        3,
        4,
        5,
    ], f"input tensor dimension must be in [3, 4, 5] but got {x_dim}"

    if data_format is None:
        if x_dim == 3:
            data_format = "NCL"
        elif x_dim == 4:
            data_format = "NCHW"
        elif x_dim == 5:
            data_format = "NCDHW"

    data_format = data_format.upper()
    assert data_format in ["NCL", "NCHW", "NCDHW", "NLC", "NHWC", "NDHWC"], (
        "data_format should be in one of [NCL, NCHW, NCDHW, NLC, NHWC, NDHWC], "
        f"but got {data_format}"
    )
    supported_format_map = {
        3: ["NCL", "NLC"],
        4: ["NCHW", "NHWC"],
        5: ["NCDHW", "NDHWC"],
    }
    assert (
        data_format in supported_format_map[x_dim]
    ), f"input tensor dimension is {x_dim}, it's data format should be in {supported_format_map[x_dim]} but got {data_format}"

    unsqueezed_dim = []

    if isinstance(pad, (Variable, pir.Value)):
        if data_format in ["NCL", "NCHW", "NCDHW"]:
            data_format = "NCDHW"
            if x_dim == 3:
                pad = concat([zeros((4,), dtype="int32"), pad], axis=0)
                unsqueezed_dim = [3, 4]
                x = unsqueeze(x, axis=unsqueezed_dim)
            elif x_dim == 4:
                pad = concat([pad, zeros((2,), dtype="int32")], axis=0)
                unsqueezed_dim = [2]
                x = unsqueeze(x, axis=unsqueezed_dim)
        elif data_format in ["NLC", "NHWC", "NDHWC"]:
            data_format = "NDHWC"
            if x_dim == 3:
                pad = concat([zeros((4,), dtype="int32"), pad], axis=0)
                unsqueezed_dim = [2, 3]
                x = unsqueeze(x, axis=unsqueezed_dim)
            elif x_dim == 4:
                pad = concat([pad, zeros((2,), dtype="int32")], axis=0)
                unsqueezed_dim = [1]
                x = unsqueeze(x, axis=unsqueezed_dim)
    else:
        pad = list(pad)
        if data_format in ["NCL", "NCHW", "NCDHW"]:
            data_format = "NCDHW"
            if x_dim == 3:
                pad = [0, 0, 0, 0, *pad]
                unsqueezed_dim = [3, 4]
                x = unsqueeze(x, axis=unsqueezed_dim)
            elif x_dim == 4:
                pad = [*pad, 0, 0]
                unsqueezed_dim = [2]
                x = unsqueeze(x, axis=unsqueezed_dim)
        elif data_format in ["NLC", "NHWC", "NDHWC"]:
            data_format = "NDHWC"
            if x_dim == 3:
                pad = [0, 0, 0, 0, *pad]
                unsqueezed_dim = [2, 3]
                x = unsqueeze(x, axis=unsqueezed_dim)
            elif x_dim == 4:
                pad = [*pad, 0, 0]
                unsqueezed_dim = [1]
                x = unsqueeze(x, axis=unsqueezed_dim)

    if in_dynamic_or_pir_mode():
        if isinstance(pad, Variable):
            pad = pad.tolist()
        if isinstance(pad, paddle.pir.Value):
            raise NotSupportedTensorArgumentError(
                "`pad` must be list of int, but received Tensor.",
                "pad",
            )
        if isinstance(pad, list) and any(not isinstance(i, int) for i in pad):
            raise NotSupportedTensorArgumentError(
                f"`pad` must be list of int, but received list of {type(pad[0])}.",
                "pad",
            )
        out = _C_ops.pad3d(x, pad, mode, value, data_format)
    else:
        attrs = {'mode': mode, 'value': value, 'data_format': data_format}
        inputs = {'X': [x]}
        if isinstance(pad, Variable):
            inputs['Paddings'] = [pad]
            attrs['paddings'] = []
        else:
            attrs['paddings'] = pad

        helper = LayerHelper('pad3d', **locals())

        dtype = helper.input_dtype(input_param_name='input')
        out = helper.create_variable_for_type_inference(dtype)
        helper.append_op(
            type='pad3d', inputs=inputs, outputs={"Out": out}, attrs=attrs
        )

    if len(unsqueezed_dim) != 0:
        out = squeeze(out, axis=unsqueezed_dim)

    return out


@deprecated(
    since="3.0.0",
    update_to="paddle.nn.ZeroPad2D",
    level=1,
    reason="Please use class ZeroPad2D",
)
def zeropad2d(
    x: Tensor,
    padding: ShapeLike,
    data_format: DataLayout2D = "NCHW",
    name: str | None = None,
) -> Tensor:
    """
    Pads the input tensor boundaries with zero according to 'pad'.

    Args:
        x(Tensor): The input tensor with data type float16/float32/float64/int32/int64.
        padding(int | Tensor | List[int] | Tuple[int]): The padding size with data type int.
            The input dimension should be 4 and pad has the form (pad_left, pad_right,
            pad_top, pad_bottom).
        data_format(str, optional): An string from: "NHWC", "NCHW". Specify the data format of
            the input data. Default: "NCHW".
        name(str, optional): The default value is None. Normally there is no need for user
            to set this property. For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, padded with 0 according to pad and data type is same as input.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn.functional as F

            >>> x_shape = paddle.to_tensor([1, 1, 2, 3])
            >>> x = paddle.arange(paddle.prod(x_shape), dtype="float32").reshape(x_shape) + 1
            >>> y = F.zeropad2d(x, [1, 2, 1, 1])
            >>> print(y)
            Tensor(shape=[1, 1, 4, 6], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[[[0., 0., 0., 0., 0., 0.],
               [0., 1., 2., 3., 0., 0.],
               [0., 4., 5., 6., 0., 0.],
               [0., 0., 0., 0., 0., 0.]]]])
    """

    return pad(
        x,
        pad=padding,
        mode='constant',
        value=0,
        data_format=data_format,
        name=name,
    )


def cosine_similarity(
    x1: Tensor, x2: Tensor, axis: int = 1, eps: float = 1e-8
) -> Tensor:
    """
    Compute cosine similarity between x1 and x2 along axis.

    Parameters:
        x1 (Tensor): First input. float32/double.
        x2 (Tensor): Second input. float32/double.
        axis (int, optional): Dimension of vectors to compute cosine similarity. Default is 1.
        eps(float, optional): Small value to avoid division by zero. Default is 1e-8.

    Returns:
        Tensor, a Tensor representing cosine similarity between x1 and x2 along axis.

    Examples:
        .. code-block:: text

            Case 0:
                x1 = [[0.8024077  0.9927354  0.27238318 0.8344984 ]
                     [0.48949873 0.5797396  0.65444374 0.66510963]
                     [0.1031398  0.9614342  0.08365563 0.6796464 ]
                     [0.10760343 0.7461209  0.7726148  0.5801006 ]]
                x2 = [[0.62913156 0.1536727  0.9847992  0.04591406]
                     [0.9098952  0.15715368 0.8671125  0.3156102 ]
                     [0.4427798  0.54136837 0.5276275  0.32394758]
                     [0.3769419  0.8535014  0.48041078 0.9256797 ]]
                axis = 1
                eps = 1e-8
                Out: [0.5275037  0.8368967  0.75037485 0.9245899]

    Code Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn as nn

            >>> paddle.seed(1)
            >>> x1 = paddle.randn(shape=[2, 3])
            >>> x2 = paddle.randn(shape=[2, 3])

            >>> result = paddle.nn.functional.cosine_similarity(x1, x2, axis=0)
            >>> print(result)
            Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [ 0.97689527,  0.99996042, -0.55138415])

    """
    w12 = sum(paddle.multiply(x1, x2), axis=axis)
    w1 = sum(paddle.multiply(x1, x1), axis=axis)
    w2 = sum(paddle.multiply(x2, x2), axis=axis)
    n12 = sqrt(clip(w1 * w2, min=eps * eps))
    cos_sim = w12 / n12
    return cos_sim


def linear(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    name: str | None = None,
) -> Tensor:
    r"""

    Fully-connected linear transformation operator. For each input :math:`X` ,
    the equation is:

    .. math::

        Out = XW + b

    where :math:`W` is the weight and :math:`b` is the bias.

    If the weight is a 2-D tensor of shape :math:`[in\_features, out\_features]` ,
    input should be a multi-dimensional tensor of shape
    :math:`[batch\_size, *, in\_features]` , where :math:`*` means any number of
    additional dimensions. The linear operator multiplies input tensor with
    weight and produces an output tensor of shape :math:`[batch\_size, *, out\_features]` ,
    If :math:`bias` is not None, the bias should be a 1-D tensor of shape
    :math:`[out\_features]` and will be added to the output.

    Parameters:
        x (Tensor): Input tensor. The data type should be bfloat16, float16, float32 or float64.
        weight (Tensor): Weight tensor. The data type should be float16, float32 or float64.
        bias (Tensor, optional): Bias tensor. The data type should be float16, float32 or float64.
                                 If it is set to None, no bias will be added to the output units.
        name (str, optional): Normally there is no need for user to set this parameter.
                              For detailed information, please refer to :ref:`api_guide_Name` .

    Returns:
        Tensor, the shape is :math:`[batch\_size, *, out\_features]` and the
        data type is the same with input :math:`x` .

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.seed(2023)
            >>> x = paddle.randn((3, 2), dtype="float32")
            >>> print(x)
            Tensor(shape=[3, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[ 0.06132207,  1.11349595],
             [ 0.41906244, -0.24858207],
             [-1.85169315, -1.50370061]])
            >>> weight = paddle.full(shape=[2, 4], fill_value=0.5, dtype="float32", name="weight")
            >>> print(weight)
            Tensor(shape=[2, 4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[0.50000000, 0.50000000, 0.50000000, 0.50000000],
             [0.50000000, 0.50000000, 0.50000000, 0.50000000]])
            >>> bias = paddle.ones(shape=[4], dtype="float32", name="bias")
            >>> print(bias)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [1., 1., 1., 1.])
            >>> y = paddle.nn.functional.linear(x, weight, bias)
            >>> print(y)
            Tensor(shape=[3, 4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[ 1.58740902,  1.58740902,  1.58740902,  1.58740902],
             [ 1.08524013,  1.08524013,  1.08524013,  1.08524013],
             [-0.67769694, -0.67769694, -0.67769694, -0.67769694]])
    """
    if in_dynamic_mode():
        # TODO(jiabin): using addmm for fast forward route
        return _C_ops.linear(x, weight, bias)

    elif in_pir_mode():
        out = _C_ops.matmul(x, weight, False, False)
        if bias is not None:
            return _C_ops.add(out, bias)
        else:
            return out
    else:
        helper = LayerHelper('linear', **locals())
        dtype = x.dtype

        check_variable_and_dtype(
            x, 'x', ["uint16", 'float16', 'float32', 'float64'], 'linear'
        )
        check_dtype(
            dtype,
            'dtype',
            ["uint16", 'float16', 'float32', 'float64'],
            'linear',
        )

        inputs = {'X': [x], 'Y': [weight]}
        attrs = {'trans_x': False, 'trans_y': False}
        tmp = helper.create_variable_for_type_inference(dtype)
        helper.append_op(
            type='matmul_v2',
            inputs=inputs,
            outputs={'Out': tmp},
            attrs=attrs,
        )
        if bias is not None:
            res = helper.create_variable_for_type_inference(dtype)
            helper.append_op(
                type='elementwise_add',
                inputs={'X': [tmp], 'Y': [bias]},
                outputs={'Out': [res]},
                attrs={'axis': -1},
            )
        else:
            res = tmp
        return res


def label_smooth(
    label: Tensor,
    prior_dist: Tensor | None = None,
    epsilon: float = 0.1,
    name: str | None = None,
) -> Tensor:
    r"""
    Label smoothing is a mechanism to regularize the classifier layer and is called
    label-smoothing regularization (LSR).Label smoothing is proposed to encourage
    the model to be less confident, since optimizing the log-likelihood of the
    correct label directly may cause overfitting and reduce the ability of the
    model to adapt.

    Label smoothing replaces the ground-truth label :math:`y` with the weighted sum
    of itself and some fixed distribution :math:`\mu`. For class :math:`k`,
    i.e.

    .. math::

        \\tilde{y_k} = (1 - \epsilon) * y_k + \epsilon * \mu_k,

    where :math:`1 - \epsilon` and :math:`\epsilon` are the weights
    respectively, and :math:`\\tilde{y}_k` is the smoothed label. Usually
    uniform distribution is used for :math:`\mu`.

    See more details about label smoothing in https://arxiv.org/abs/1512.00567.

    Parameters:
        label(Tensor): The input variable containing the label data. The
                        label data should use one-hot representation. It's
                        a multidimensional tensor with a shape of
                        :math:`[N_1, ..., Depth]`, where Depth is class number. The dtype can be "float16" "float32" and "float64".
        prior_dist(Tensor, optional): The prior distribution to be used to smooth
                        labels. If not provided, an uniform distribution
                        is used. It's a multidimensional tensor with a shape of
                        :math:`[1, class\_num]` . The default value is None.
        epsilon(float, optional): The weight used to mix up the original ground-truth
                        distribution and the fixed distribution. The default value is
                        0.1.
        name(str, optional): The default value is None. Normally there is no need for user
                        to set this property. For more information, please refer to
                        :ref:`api_guide_Name`.

    Returns:
        Tensor: The tensor containing the smoothed labels.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.disable_static()

            >>> x = paddle.to_tensor([[[0, 1, 0],
            >>>                     [ 1,  0, 1]]], dtype="float32", stop_gradient=False)

            >>> output = paddle.nn.functional.label_smooth(x)
            >>> print(output)
            Tensor(shape=[1, 2, 3], dtype=float32, place=Place(cpu), stop_gradient=False,
            [[[0.03333334, 0.93333334, 0.03333334],
            [0.93333334, 0.03333334, 0.93333334]]])
    """
    if epsilon > 1.0 or epsilon < 0.0:
        raise ValueError("The value of epsilon must be between 0 and 1.")

    if in_dynamic_or_pir_mode():
        return _C_ops.label_smooth(label, prior_dist, float(epsilon))

    check_variable_and_dtype(
        label,
        'label',
        ['uint16', 'float16', 'float32', 'float64'],
        'label_smooth',
    )

    helper = LayerHelper("label_smooth", **locals())
    label.stop_gradient = True
    smooth_label = helper.create_variable_for_type_inference(label.dtype)
    helper.append_op(
        type="label_smooth",
        inputs=(
            {"X": label, "PriorDist": prior_dist}
            if prior_dist
            else {"X": label}
        ),
        outputs={"Out": smooth_label},
        attrs={"epsilon": float(epsilon)},
    )
    return smooth_label


def class_center_sample(
    label: Tensor,
    num_classes: int,
    num_samples: int,
    group: Group | bool | None = None,
) -> tuple[Tensor, Tensor]:
    """
    Class center sample method is proposed from the paper PartialFC that only sample a subset of the class centers.
    The process of sampling subset class centers is straightforward:

    1. First select the positive class centers;
    2. Then randomly sample negative class centers.

    Specifically, given a label tensor, shape [batch_size], select all the positive class centers and randomly
    sample negative class centers, then remap the input label tensor using the sampled class centers.

    For more information, Partial FC: Training 10 Million Identities on a Single Machine
    arxiv: https://arxiv.org/abs/2010.05222

    Note:
        If the number of the positive class centers is greater than the input num_samples, it keeps all the positive
        class centers and the shape of sampled_class_center will be [num_positive_class_centers].

        The API supports CPU, single GPU and multi GPU.

        For data parallel mode, set ``group=False``.

        For model parallel mode, set ``group=None`` or the group instance return by paddle.distributed.new_group.

    Args:
        label (Tensor): 1-D tensor with shape [N], each label in [0, num_classes)
        num_classes (int): A positive integer to specify the number of classes at local rank.
            Note that num_classes of each GPU can be different.
        num_samples (int): A positive integer to specify the number of class center to sample.
        group (Group, optional): The group instance return by paddle.distributed.new_group
            or ``None`` for global default group or ``False`` for data parallel (do not communication cross ranks).
            Default is ``None``.

    Returns:
        Tuple of two ``Tensor`` : (remapped_label, sampled_class_center), remapped label using sampled class center,
        sampled class center from [0, num_classes).

    Examples:

    .. code-block:: python
        :name: code-example1

        >>> # CPU or single GPU
        >>> import paddle
        >>> num_classes = 20
        >>> batch_size = 10
        >>> num_samples = 6
        >>> paddle.seed(2023)
        >>> label = paddle.randint(low=0, high=num_classes, shape=[batch_size], dtype='int64')
        >>> remapped_label, sampled_class_index = paddle.nn.functional.class_center_sample(label, num_classes, num_samples)
        >>> print(label)
        Tensor(shape=[10], dtype=int64, place=Place(cpu), stop_gradient=True,
        [17, 10, 5 , 18, 8 , 8 , 19, 14, 10, 14])
        >>> print(remapped_label)
        Tensor(shape=[10], dtype=int64, place=Place(cpu), stop_gradient=True,
        [4, 2, 0, 5, 1, 1, 6, 3, 2, 3])
        >>> print(sampled_class_index)
        Tensor(shape=[7], dtype=int64, place=Place(cpu), stop_gradient=True,
        [5 , 8 , 10, 14, 17, 18, 19])

    .. code-block:: python
        :name: code-example2

        >>> # doctest: +REQUIRES(env:DISTRIBUTED)
        >>> # Multi GPU, test_class_center_sample.py
        >>> import paddle
        >>> import paddle.distributed as dist
        >>> strategy = dist.fleet.DistributedStrategy()
        >>> dist.fleet.init(is_collective=True, strategy=strategy)
        >>> batch_size = 10
        >>> num_samples = 6
        >>> rank_id = dist.get_rank()
        >>> # num_classes of each GPU can be different, e.g num_classes_list = [10, 8]
        >>> num_classes_list = [10, 10]
        >>> num_classes = paddle.sum(paddle.to_tensor(num_classes_list))
        >>> label = paddle.randint(low=0, high=num_classes.item(), shape=[batch_size], dtype='int64') # type: ignore
        >>> label_list = [] # type: ignore
        >>> dist.all_gather(label_list, label)
        >>> label = paddle.concat(label_list, axis=0)
        >>> remapped_label, sampled_class_index = paddle.nn.functional.class_center_sample(label, num_classes_list[rank_id], num_samples)

        >>> print(label)
        >>> print(remapped_label)
        >>> print(sampled_class_index)
        >>> #python -m paddle.distributed.launch --gpus=0,1 test_class_center_sample.py
        >>> # rank 0 output:
        Tensor(shape=[20], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
        [10, 17, 15, 11, 9 , 12, 18, 18, 17, 18, 19, 2 , 8 , 13, 11, 13, 9 , 10, 0 , 4 ])
        Tensor(shape=[20], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
        [6 , 11, 10, 7 , 4 , 8 , 12, 12, 11, 12, 13, 1 , 3 , 9 , 7 , 9 , 4 , 6 , 0 , 2 ])
        Tensor(shape=[6], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
        [0, 2, 4, 8, 9, 3])
        >>> # rank 1 output:
        Tensor(shape=[20], dtype=int64, place=CUDAPlace(1), stop_gradient=True,
        [10, 17, 15, 11, 9 , 12, 18, 18, 17, 18, 19, 2 , 8 , 13, 11, 13, 9 , 10, 0 , 4 ])
        Tensor(shape=[20], dtype=int64, place=CUDAPlace(1), stop_gradient=True,
        [6 , 11, 10, 7 , 4 , 8 , 12, 12, 11, 12, 13, 1 , 3 , 9 , 7 , 9 , 4 , 6 , 0 , 2 ])
        Tensor(shape=[7], dtype=int64, place=CUDAPlace(1), stop_gradient=True,
        [0, 1, 2, 3, 5, 7, 8])
    """
    if not (group is False or group is None or hasattr(group, 'is_member')):
        raise ValueError(
            f'Expected group is False, None or instance of paddle.distributed.collective.Group \
             (got group: {group})'
        )
        return

    if hasattr(group, 'is_member') and not group.is_member():
        return

    ring_id = 0
    rank = 0
    nranks = 1
    if group is not False:
        if core.is_compiled_with_dist():
            parallel_env = paddle.distributed.ParallelEnv()
            global_rank = parallel_env.rank
            rank = (
                global_rank
                if group is None
                else group.get_group_rank(global_rank)
            )
            nranks = parallel_env.world_size if group is None else group.nranks

    if num_samples > num_classes:
        raise ValueError(
            f'Expected num_samples less than or equal to {num_classes}, got num_samples {num_samples}'
        )

    label_size = 1
    for dim in list(label.shape):
        label_size *= dim
    if label_size != -1 and label_size < 1:
        raise ValueError(
            f'Expected label_size > 0 \
             (got label_size: {label_size})'
        )

    label_dims = len(list(label.shape))
    if label_dims != 1:
        raise ValueError(
            f'Expected label_dims == 1 \
             (got label_dims: {label_dims})'
        )

    seed = None
    if (seed is None or seed == 0) and default_main_program().random_seed != 0:
        seed = default_main_program().random_seed

    if in_dynamic_or_pir_mode():
        return _C_ops.class_center_sample(
            label,
            num_classes,
            num_samples,
            ring_id,
            rank,
            nranks,
            seed is not None,
            seed if seed is not None else 0,
        )

    check_variable_and_dtype(
        label, 'label', ['int64', 'int32'], 'class_center_sample'
    )
    op_type = 'class_center_sample'
    helper = LayerHelper(op_type, **locals())
    remapped_label = helper.create_variable_for_type_inference(
        dtype=label.dtype
    )
    sampled_class_center = helper.create_variable_for_type_inference(
        dtype=label.dtype
    )
    helper.append_op(
        type=op_type,
        inputs={'Label': label},
        outputs={
            'RemappedLabel': remapped_label,
            'SampledLocalClassCenter': sampled_class_center,
        },
        attrs={
            'num_classes': num_classes,
            'num_samples': num_samples,
            'ring_id': ring_id,
            'nranks': nranks,
            'rank': rank,
            'fix_seed': seed is not None,
            'seed': seed if seed is not None else 0,
        },
    )
    return remapped_label, sampled_class_center


def fold(
    x: Tensor,
    output_sizes: Size2,
    kernel_sizes: Size2,
    strides: Size2 = 1,
    paddings: Size2 | Size4 = 0,
    dilations: Size2 = 1,
    name: str | None = None,
) -> Tensor:
    r"""

    Combines an array of sliding local blocks into a large containing
    tensor. also known as col2im when operated on batched 2D image tensor. Fold calculates each
    combined value in the resulting large tensor by summing all values from all containing blocks.


    For each input :math:`x` with shape [N, C_in , L], the output shape [N, C_out, H_out, W_out]
    can be calculated as following.

    .. math::

        H_{out} &= output\_size[0] \\
        W_{out} &= output\_size[1] \\
        C_{out} &= \frac{C_{in}}{kernel\_sizes[0]\times kernel\_sizes[1]} \\

    Parameters:
        x(Tensor):                3-D Tensor, input tensor of format [N, C, L],
                                  data type can be float32, float64, complex64 or complex128
        output_sizes(int|list|tuple):       The size of output size, should be [output_size_h, output_size_w]
                                  or an integer o treated as [o, o].
        kernel_sizes(int|list|tuple):   The size of convolution kernel, should be [k_h, k_w]
                                  or an integer k treated as [k, k].
        strides(int|list|tuple, optional):        The strides, should be [stride_h, stride_w]
                                  or an integer stride treated as [stride, stride].
                                  For default, strides will be [1, 1].
        paddings(int|list|tuple, optional):       The paddings of each dimension, should be
                                  [padding_top, padding_left, padding_bottom, padding_right]
                                  or [padding_h, padding_w] or an integer padding.
                                  If [padding_h, padding_w] was given, it will expanded to
                                  [padding_h, padding_w, padding_h, padding_w]. If an integer
                                  padding was given, [padding, padding, padding, padding] will
                                  be used. For default, paddings will be [0, 0, 0, 0]
        dilations(int|list|tuple, optional):      the dilations of convolution kernel, should be
                                  [dilation_h, dilation_w], or an integer dilation treated as
                                  [dilation, dilation]. For default, it will be [1, 1].
        name(str, optional): The default value is None.
                             Normally there is no need for user to set this property.
                             For more information, please refer to :ref:`api_guide_Name`


    Returns:
        The tensor formed by combining a group of sliding local blocks
        The output shape is [N, Cout, H, W] as described above.

    Examples:

        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn.functional as F

            >>> x = paddle.randn([2,3*2*2,12])
            >>> y = F.fold(x, output_sizes=[4, 5], kernel_sizes=2)
            >>> x = paddle.randn([2,3*2*2,12])
            >>> y = F.fold(x, output_sizes=[4, 5], kernel_sizes=2)
            >>> print(y.shape)
            [2, 3, 4, 5]

    """

    helper = LayerHelper("fold", **locals())

    check_variable_and_dtype(
        x, 'x', ['float32', 'float64', 'complex64', 'complex128'], 'fold'
    )

    assert len(x.shape) == 3, "input should be the format of [N, C, L]"

    def _is_list_or_tuple_(data):
        return isinstance(data, (list, tuple))

    if isinstance(output_sizes, int):
        output_sizes = [output_sizes, output_sizes]
    else:
        assert _is_list_or_tuple_(output_sizes) and (
            len(output_sizes) == 2
        ), "output_sizes should either be an integer or a list/tuple of two integers"

    if isinstance(kernel_sizes, int):
        kernel_sizes = [kernel_sizes, kernel_sizes]
    else:
        assert _is_list_or_tuple_(kernel_sizes) and (
            len(kernel_sizes) == 2
        ), "kernel_sizes should either be an integer or a list/tuple of two integers"

    if isinstance(strides, int):
        strides = [strides, strides]
    else:
        assert _is_list_or_tuple_(strides) and (
            len(strides) == 2
        ), "strides should either be an integer or a list/tuple of two integers"

    if isinstance(dilations, int):
        dilations = [dilations, dilations]
    else:
        assert _is_list_or_tuple_(dilations) and (
            len(dilations) == 2
        ), "dilations should either be an integer or a list/tuple of two integers"

    if isinstance(paddings, int):
        paddings = [paddings] * 4
    elif isinstance(paddings, list):
        if len(paddings) == 2:
            paddings = paddings * 2
        elif len(paddings) == 4:
            pass
        else:
            raise ValueError(
                "paddings should either be an integer or a list of 2 or 4 integers"
            )
    else:
        raise ValueError(
            "Unexpected type of paddings, it should be either an integer or a list"
            "of 2 or 4 integers"
        )

    if in_dynamic_or_pir_mode():
        out = _C_ops.fold(
            x, output_sizes, kernel_sizes, strides, paddings, dilations
        )
    else:
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(
            type="fold",
            inputs={"X": x},
            outputs={"Y": out},
            attrs={
                "output_sizes": output_sizes,
                "kernel_sizes": kernel_sizes,
                "strides": strides,
                "paddings": paddings,
                "dilations": dilations,
            },
        )
    return out
