# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar, overload

import paddle
from paddle import in_dynamic_mode

from .. import functional as F
from ..quant.quantized_linear import weight_only_linear, weight_quantize
from .layers import Layer

if TYPE_CHECKING:
    from collections.abc import Sequence

    from paddle import Tensor
    from paddle._typing import (
        DataLayout1D,
        DataLayout1DVariant,
        DataLayout2D,
        DataLayout3D,
        ParamAttrLike,
        ShapeLike,
        Size2,
        Size4,
    )

    from ..functional.common import (
        _DropoutMode,
        _InterpolateMode,
        _PaddingTensorMode,
    )

    _T_Padding = TypeVar("_T_Padding", Tensor, Sequence[int])

__all__ = []


@overload
def _npairs(x: _T_Padding, n: int) -> _T_Padding: ...


@overload
def _npairs(x: int, n: int) -> int: ...


def _npairs(x, n):
    if isinstance(x, (paddle.Tensor, list, tuple)):
        return x
    x = [x] * (n * 2)
    return x


class Identity(Layer):
    r"""

    A placeholder identity operator that is argument-insensitive. For each input :math:`X` ,
    the output :math:`Out` is:

    .. math::

        Out = X

    Parameters:
        args: any argument (unused)
        kwargs: any keyword argument (unused)

    Shape:
        - input: Multi-dimensional tensor with shape :math:`[batch\_size, n1, n2, ...]` .
        - output: Multi-dimensional tensor with shape :math:`[batch\_size, n1, n2, ...]` .

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.seed(100)

            >>> input_tensor = paddle.randn(shape=[3, 2])
            >>> layer = paddle.nn.Identity()
            >>> out = layer(input_tensor)
            >>> print(input_tensor)
            Tensor(shape=[3, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[-1.41661501,  0.25904641],
             [ 0.00979547, -0.30324230],
             [-1.34256756, -0.76540256]])
            >>> print(out)
            Tensor(shape=[3, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[-1.41661501,  0.25904641],
             [ 0.00979547, -0.30324230],
             [-1.34256756, -0.76540256]])

    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        return input


class Linear(Layer):
    r"""

    Fully-connected linear transformation layer. For each input :math:`X` ,
    the equation is:

    .. math::

        Out = XW + b

    where :math:`W` is the weight and :math:`b` is the bias.

    Linear layer takes only one multi-dimensional tensor as input with the
    shape :math:`[batch\_size, *, in\_features]` , where :math:`*` means any
    number of additional dimensions. It multiplies input tensor with the weight
    (a 2-D tensor of shape :math:`[in\_features, out\_features]` ) and produces
    an output tensor of shape :math:`[batch\_size, *, out\_features]` .
    If :math:`bias\_attr` is not False, the bias (a 1-D tensor of
    shape :math:`[out\_features]` ) will be created and added to the output.

    Parameters:
        in_features (int): The number of input units.
        out_features (int): The number of output units.
        weight_attr (ParamAttr|None, optional): The attribute for the learnable
            weight of this layer. The default value is None. If the Initializer of the
            param_attr is not set, the parameter is initialized with Xavier.
            For detailed information, please refer to paddle.ParamAttr.
        bias_attr (ParamAttr|bool|None, optional): The attribute for the learnable bias
            of this layer. If it is set to False, no bias will be added to the output.
            If it is set to None or one kind of ParamAttr, a bias parameter will
            be created according to ParamAttr. For detailed information, please refer
            to paddle.ParamAttr. The default value is None and the bias will be
            initialized to zero.
        name (str|None, optional): Normally there is no need for user to set this parameter.
            For detailed information, please refer to :ref:`api_guide_Name` .

    Attribute:
        **weight** (Parameter): the learnable weight of this layer.

        **bias** (Parameter): the learnable bias of this layer.

    Shape:
        - input: Multi-dimensional tensor with shape :math:`[batch\_size, *, in\_features]` . Its data types are float16, float32, float64 ,The default is float32 .
        - output: Multi-dimensional tensor with shape :math:`[batch\_size, *, out\_features]` . The data type is the same as the input .

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.seed(100)

            >>> # Define the linear layer.
            >>> weight_attr = paddle.ParamAttr(
            ...     name="weight",
            ...     initializer=paddle.nn.initializer.Constant(value=0.5))
            >>> bias_attr = paddle.ParamAttr(
            ...     name="bias",
            ...     initializer=paddle.nn.initializer.Constant(value=1.0))
            >>> linear = paddle.nn.Linear(2, 4, weight_attr=weight_attr, bias_attr=bias_attr)
            >>> print(linear.weight)
            Parameter containing:
            Tensor(shape=[2, 4], dtype=float32, place=Place(cpu), stop_gradient=False,
            [[0.50000000, 0.50000000, 0.50000000, 0.50000000],
             [0.50000000, 0.50000000, 0.50000000, 0.50000000]])

            >>> print(linear.bias)
            Parameter containing:
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=False,
            [1., 1., 1., 1.])

            >>> x = paddle.randn((3, 2), dtype="float32")
            >>> y = linear(x)
            >>> print(y)
            Tensor(shape=[3, 4], dtype=float32, place=Place(cpu), stop_gradient=False,
            [[ 0.42121571,  0.42121571,  0.42121571,  0.42121571],
             [ 0.85327661,  0.85327661,  0.85327661,  0.85327661],
             [-0.05398512, -0.05398512, -0.05398512, -0.05398512]])
    """

    weight: Tensor
    bias: Tensor
    name: str | None

    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight_attr: ParamAttrLike | None = None,
        bias_attr: ParamAttrLike | None = None,
        name: str | None = None,
        use_wint8=False,
    ) -> None:
        super().__init__()
        self._dtype = self._helper.get_default_dtype()
        self._weight_attr = weight_attr
        self._bias_attr = bias_attr
        self.weight = self.create_parameter(
            shape=[in_features, out_features],
            attr=self._weight_attr,
            dtype=self._dtype,
            is_bias=False,
        )
        self.bias = self.create_parameter(
            shape=[out_features],
            attr=self._bias_attr,
            dtype=self._dtype,
            is_bias=True,
        )
        self.name = name
        self.use_wint8 = use_wint8
        self.first_run = True
        self.weight8 = None
        self.scale8 = None

    def forward(self, input: Tensor) -> Tensor:
        if self.use_wint8:
            if self.first_run:
                self.weight8, self.scale8 = weight_quantize(self.weight)
                del self.weight
                paddle.device.cuda.empty_cache()
                self.first_run = False
            out = weight_only_linear(
                input, self.weight8, self.bias, self.scale8
            )
        else:
            out = F.linear(
                x=input, weight=self.weight, bias=self.bias, name=self.name
            )
        return out

    def extra_repr(self) -> str:
        name_str = f', name={self.name}' if self.name else ''
        return f'in_features={self.weight.shape[0]}, out_features={self.weight.shape[1]}, dtype={self._dtype}{name_str}'


class Upsample(Layer):
    """
    This op resizes a batch of images.

    The input must be a 3-D Tensor of the shape (num_batches, channels, in_w)
    or (num_batches, in_w, channels), or 4-D (num_batches, channels, in_h, in_w) or
    (num_batches, in_h, in_w, channels), or a 5-D Tensor of the shape
    (num_batches, channels, in_d, in_h, in_w) or (num_batches, in_d, in_h, in_w, channels),
    Where in_w is width of the input tensor, in_h is the height of the input tensor,
    in_d is the depth of the input tensor.
    and the resizing only applies on the three dimensions(depth, height and width).

    Supporting resample methods:
        'linear' : Linear interpolation
        'bilinear' : Bilinear interpolation
        'trilinear' : Trilinear interpolation
        'nearest' : Nearest neighbor interpolation
        'bicubic' : Bicubic interpolation
        'area' : Area interpolation

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

    https://en.wikipedia.org/wiki/Linear_interpolation.
    For details of linear interpolation, please refer to Wikipedia:

    For details of nearest neighbor interpolation, please refer to Wikipedia:
    https://en.wikipedia.org/wiki/Nearest-neighbor_interpolation.

    For details of bilinear interpolation, please refer to Wikipedia:
    https://en.wikipedia.org/wiki/Bilinear_interpolation.

    For details of bicubic interpolation, please refer to Wikipedia:
    https://en.wikipedia.org/wiki/Bicubic_interpolation

    For details of trilinear interpolation, please refer to Wikipedia:
    https://en.wikipedia.org/wiki/Trilinear_interpolation.

    Parameters:

        size (list|tuple|Tensor|None): Output shape of image resize
             layer, the shape is (out_w, ) when input is a 3-D Tensor, the shape is (out_h, out_w)
             when input is a 4-D Tensor and is (out_d, out_h, out_w) when input is a 5-D Tensor.
             Default: None. If a list/tuple, each element can be an integer or a Tensor of shape: [1].
             If a Tensor , its dimensions size should be a 1.
        scale_factor (float|Tensor|list|tuple|None): The multiplier for the input height or width. At
             least one of :attr:`size` or :attr:`scale_factor` must be set.
             And :attr:`size` has a higher priority than :attr:`scale_factor`. Has to match input size if it is either a list or a tuple or a Tensor.
             Default: None.
        mode (str): The resample method. It supports 'linear', 'nearest', 'bilinear', 'area',
                       'bicubic' and 'trilinear' currently. Default: 'nearest'
        align_corners(bool) :  An optional bool, If True, the centers of the 4 corner pixels of the
                               input and output tensors are aligned, preserving the values at the
                               corner pixels.
                               Default: False
        align_mode(int)  :  An optional for linear/bilinear/trilinear interpolation. Refer to the formula in the example above,
                            it can be \'0\' for src_idx = scale_factor*(dst_indx+0.5)-0.5 , can be \'1\' for
                            src_idx = scale_factor*dst_index.
        data_format (str|None, optional): Specify the data format of the input, and the data format of
             the output will be consistent with that of the input. An optional string from:`"NCW"`,
             `"NWC"`,  `"NCHW"`, `"NHWC"`, `"NCDHW"`, `"NDHWC"`. The default value is None.
             When :attr:`data_format` is not specified, it will be automatically inferred from the
             input dimension of :attr:`x`. When :attr:`x` is a 3-D Tensor, :attr:`data_format` will be
             set to `"NCW"`; When :attr:`x` is a 4-D Tensor, :attr:`data_format` will be set to
             `"NCHW"`; When :attr:`x` is a 5-D Tensor, :attr:`data_format` will be set to `"NCDHW"`.
             When it is `"NCHW"`, the data should be stored in the order of:
             `[batch_size, input_channels, input_height, input_width]`. When it is `"NCDHW"`, the
             data should be stored in the order of: `[batch_size, input_channels, input_depth, input_height, input_width]`.
        name(str|None, optional): The default value is None.
                             Normally there is no need for user to set this property.
                             For more information, please refer to :ref:`api_guide_Name`
    Returns:
        A callable object of Upsample.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> input = paddle.rand([2, 3, 6, 10], dtype="float32")
            >>> upsample_out = paddle.nn.Upsample(size=[12, 12])

            >>> output = upsample_out(x=input)
            >>> print(output.shape)
            [2, 3, 12, 12]

    """

    size: ShapeLike | None
    scale_factor: ShapeLike | float | None
    mode: _InterpolateMode
    align_corners: bool
    align_mode: int
    data_format: DataLayout1DVariant | DataLayout2D | DataLayout3D | None
    name: str | None

    def __init__(
        self,
        size: ShapeLike | None = None,
        scale_factor: ShapeLike | float | None = None,
        mode: _InterpolateMode = 'nearest',
        align_corners: bool = False,
        align_mode: int = 0,
        data_format: (
            DataLayout1DVariant | DataLayout2D | DataLayout3D | None
        ) = None,
        name: str | None = None,
    ) -> None:
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode.lower()
        self.align_corners = align_corners
        self.align_mode = align_mode
        self.data_format = data_format
        self.name = name

    def forward(self, x: Tensor) -> Tensor:
        if self.data_format is None:
            dim_size = len(x.shape)
            if dim_size == 3:
                self.data_format = 'NCW'
            elif dim_size == 4:
                self.data_format = 'NCHW'
            elif dim_size == 5:
                self.data_format = 'NCDHW'
            else:
                raise ValueError(
                    f"The dimension of the input tensor should only be 3-D, 4-D or 5-D, but the received dimension is {dim_size}."
                )
        out = F.interpolate(
            x,
            size=self.size,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
            align_mode=self.align_mode,
            data_format=self.data_format,
            name=self.name,
        )

        return out

    def extra_repr(self) -> str:
        if self.scale_factor is not None:
            main_str = f'scale_factor={self.scale_factor}'
        else:
            main_str = f'size={self.size}'
        name_str = f', name={self.name}' if self.name else ''
        return f'{main_str}, mode={self.mode}, align_corners={self.align_corners}, align_mode={self.align_mode}, data_format={self.data_format}{name_str}'


class UpsamplingNearest2D(Layer):
    """
    This op upsamples a batch of images, using nearest neighbours' pixel values.
    The input must be a 4-D Tensor of the shape (num_batches, channels, in_h, in_w),
    where in_w is width of the input tensor, in_h is the height of the input tensor.
    And the upsampling only applies on the two dimensions(height and width).
    Nearest neighbor interpolation is to perform nearest neighbor interpolation
    in both the 3rd dimension(in height direction) and the 4th dimension(in width
    direction) on input tensor.

    For details of nearest neighbor interpolation, please refer to Wikipedia:
    https://en.wikipedia.org/wiki/Nearest-neighbor_interpolation.

    Parameters:
        x (Tensor): 4-D Tensor, its data type is float32, float64, or uint8,
                          its data format is specified by :attr:`data_format`.
        size (int|list|tuple|Tensor|None): Output shape of image resize
             layer, the shape is (out_h, out_w) when input is a 4-D Tensor.
             Default: None. If an int value, the `out_h` and `out_w` will be set as the number.
             If a list/tuple, each element can be an integer or a Tensor of shape: [1].
             If a Tensor, its dimensions size should be a 1.
        scale_factor (float|int|list|tuple|Tensor|None): The multiplier for the input height or width. At
             least one of :attr:`size` or :attr:`scale_factor` must be set.
             And :attr:`size` has a higher priority than :attr:`scale_factor`.
             Has to match input size if it is either a list or a tuple or a Tensor.
             Default: None.
        data_format (str, optional): Specify the data format of the input, and the data format of the output
            will be consistent with that of the input. An optional string from:`NCW`, `NWC`, `"NCHW"`, `"NHWC"`, `"NCDHW"`,
            `"NDHWC"`. The default is `"NCHW"`. When it is `"NCHW"`, the data is stored in the order of:
            `[batch_size, input_channels, input_height, input_width]`. When it is `"NCHW"`, the data is stored
            in the order of: `[batch_size, input_channels, input_depth, input_height, input_width]`.
        name(str|None, optional): The default value is None.
                             Normally there is no need for user to set this property.
                             For more information, please refer to :ref:`api_guide_Name`
    Returns:
        A 4-D Tensor of the shape (num_batches, channels, out_h, out_w) or (num_batches, out_h, out_w, channels),


    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn as nn

            >>> input_data = paddle.rand(shape=(2, 3, 6, 10)).astype("float32")
            >>> upsample_out  = paddle.nn.UpsamplingNearest2D(size=[12, 12])
            >>> input = paddle.to_tensor(input_data)
            >>> output = upsample_out(x=input)
            >>> print(output.shape)
            [2, 3, 12, 12]
    """

    size: ShapeLike | None
    scale_factor: ShapeLike | float | None
    data_format: DataLayout1DVariant | DataLayout2D | DataLayout3D
    name: str | None

    def __init__(
        self,
        size: ShapeLike | None = None,
        scale_factor: ShapeLike | float | None = None,
        data_format: DataLayout1DVariant | DataLayout2D | DataLayout3D = 'NCHW',
        name: str | None = None,
    ) -> None:
        super().__init__()
        if isinstance(size, int):
            size = [size, size]
        self.size = size
        self.scale_factor = scale_factor
        self.data_format = data_format
        self.name = name

    def forward(self, x: Tensor) -> Tensor:
        out = F.interpolate(
            x,
            size=self.size,
            scale_factor=self.scale_factor,
            mode='nearest',
            align_corners=False,
            align_mode=0,
            data_format=self.data_format,
            name=self.name,
        )

        return out

    def extra_repr(self) -> str:
        if self.scale_factor is not None:
            main_str = f'scale_factor={self.scale_factor}'
        else:
            main_str = f'size={self.size}'
        name_str = f', name={self.name}' if self.name else ''
        return f'{main_str}, data_format={self.data_format}{name_str}'


class UpsamplingBilinear2D(Layer):
    """
    This op upsamples a batch of images, using bilinear' pixel values.
    The input must be a 4-D Tensor of the shape (num_batches, channels, in_h, in_w),
    where in_w is width of the input tensor, in_h is the height of the input tensor.
    And the upsampling only applies on the two dimensions(height and width).
    Bilinear interpolation is an extension of linear interpolation for
    interpolating functions of two variables (e.g. H-direction and
    W-direction in this op) on a rectilinear 2D grid. The key idea is
    to perform linear interpolation first in one direction, and then
    again in the other direction.

    For details of bilinear interpolation, please refer to Wikipedia:
    https://en.wikipedia.org/wiki/Bilinear_interpolation.

    Parameters:
        x (Tensor): 4-D Tensor, its data type is float32, float64, or uint8,
                          its data format is specified by :attr:`data_format`.
        size (int|list|tuple|Tensor|None): Output shape of image resize
             layer, the shape is (out_h, out_w) when input is a 4-D Tensor.
             Default: None. If an int value, the `out_h` and `out_w` will be set as the number.
             If a list/tuple, each element can be an integer or a Tensor  of shape: [1].
             If a Tensor , its dimensions size should be a 1.
        scale_factor (float|int|list|tuple|Tensor|None): The multiplier for the input height or width. At
             least one of :attr:`size` or :attr:`scale_factor` must be set.
             And :attr:`size` has a higher priority than :attr:`scale_factor`.
             Has to match input size if it is either a list or a tuple or a Tensor.
             Default: None.
        data_format (str, optional): Specify the data format of the input, and the data format of the output
            will be consistent with that of the input. An optional string from:`NCW`, `NWC`, `"NCHW"`, `"NHWC"`, `"NCDHW"`,
            `"NDHWC"`. The default is `"NCHW"`. When it is `"NCHW"`, the data is stored in the order of:
            `[batch_size, input_channels, input_height, input_width]`. When it is `"NCHW"`, the data is stored
            in the order of: `[batch_size, input_channels, input_depth, input_height, input_width]`.
        name(str|None, optional): The default value is None.
                             Normally there is no need for user to set this property.
                             For more information, please refer to :ref:`api_guide_Name`
    Returns:
        A 4-D Tensor of the shape (num_batches, channels, out_h, out_w) or (num_batches, out_h, out_w, channels),

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn as nn

            >>> input_data = paddle.rand(shape=(2, 3, 6, 10)).astype("float32")
            >>> upsample_out  = paddle.nn.UpsamplingBilinear2D(size=[12, 12])
            >>> input = paddle.to_tensor(input_data)
            >>> output = upsample_out(x=input)
            >>> print(output.shape)
            [2, 3, 12, 12]
    """

    size: ShapeLike | None
    scale_factor: ShapeLike | float
    data_format: DataLayout1DVariant | DataLayout2D | DataLayout3D
    name: str | None

    def __init__(
        self,
        size: ShapeLike | None = None,
        scale_factor: ShapeLike | float = None,
        data_format: DataLayout1DVariant | DataLayout2D | DataLayout3D = 'NCHW',
        name: str | None = None,
    ) -> None:
        super().__init__()
        if isinstance(size, int):
            size = [size, size]
        self.size = size
        self.scale_factor = scale_factor
        self.data_format = data_format
        self.name = name

    def forward(self, x: Tensor) -> Tensor:
        out = F.interpolate(
            x,
            size=self.size,
            scale_factor=self.scale_factor,
            mode='bilinear',
            align_corners=True,
            align_mode=0,
            data_format=self.data_format,
            name=self.name,
        )

        return out

    def extra_repr(self) -> str:
        if self.scale_factor is not None:
            main_str = f'scale_factor={self.scale_factor}'
        else:
            main_str = f'size={self.size}'
        name_str = f', name={self.name}' if self.name else ''
        return f'{main_str}, data_format={self.data_format}{name_str}'


class Bilinear(Layer):
    r"""

    This layer performs bilinear on two inputs.

    .. math::

      out_{i} = x1 * W_{i} * {x2^\mathrm{T}}, i=0,1,...,outfeatures-1

      out = out + b

    In this formula:
     - :math:`x1`: the first input contains in1_features elements, shape is [batch_size, in1_features].
     - :math:`x2`: the second input contains in2_features elements, shape is [batch_size, in2_features].
     - :math:`W_{i}`: the i-th learned weight, shape is [in1_features, in2_features], and learned weight's shape is [out_features, in1_features, in2_features].
     - :math:`out_{i}`: the i-th element of out, shape is [batch_size], and out's shape is [batch_size, out_features].
     - :math:`b`: the learned bias, shape is [1, out_features].
     - :math:`x2^\mathrm{T}`: the transpose of :math:`x2`.

    Parameters:
       in1_features (int): The dimension of each first input(`x1`).
       in2_features (int): The dimension of each second input(`x2`).
       out_features (int): The dimension of output of this layer.
       weight_attr (ParamAttr|None, optional): The parameter attribute for the learnable w, parameters/weights of
       this layer. The default value is None.
       bias_attr (ParamAttr|bool|None, optional): The parameter attribute for the bias
           of this layer. If it is set to False, no bias will be added to the output units.
           If it is set to None, the bias is initialized zero. The default value is None.
       name (str|None, optional): The default value is None. Normally there is no need for user
           to set this property. For more information, please refer to :ref:`api_guide_Name`. Default: None.

    Attribute:
        **weight** (Parameter): the learnable weights of this layer.

        **bias** (Parameter): the learnable bias of this layer.

    Returns:
       Tensor: A 2-D Tensor of shape [batch_size, out_features].

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> layer1 = paddle.rand((5, 5)).astype('float32')
            >>> layer2 = paddle.rand((5, 4)).astype('float32')
            >>> bilinear = paddle.nn.Bilinear(in1_features=5,
            ...                               in2_features=4,
            ...                               out_features=1000)

            >>> result = bilinear(layer1,layer2)
            >>> print(result.shape)
            [5, 1000]

    """

    weight: Tensor
    bias: Tensor

    def __init__(
        self,
        in1_features: int,
        in2_features: int,
        out_features: int,
        weight_attr: ParamAttrLike | None = None,
        bias_attr: ParamAttrLike | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__()
        self._weight_attr = weight_attr
        self._bias_attr = bias_attr
        self._name = name
        self._in1_features = in1_features
        self._in2_features = in2_features
        self._out_features = out_features
        self._dtype = self._helper.get_default_dtype()

        weight_shape = [
            self._out_features,
            self._in1_features,
            self._in2_features,
        ]
        self.weight = self.create_parameter(
            attr=self._weight_attr,
            shape=weight_shape,
            dtype=self._dtype,
            is_bias=False,
        )
        bias_shape = [1, self._out_features]
        self.bias = self.create_parameter(
            attr=self._bias_attr,
            shape=bias_shape,
            dtype=self._dtype,
            is_bias=True,
        )

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        return F.bilinear(x1, x2, self.weight, self.bias, self._name)

    def extra_repr(self) -> str:
        name_str = f', name={self._name}' if self._name else ''
        return f'in1_features={self._in1_features}, in2_features={self._in2_features}, out_features={self._out_features}, dtype={self._dtype}{name_str}'


class Dropout(Layer):
    r"""
    Dropout is a regularization technique for reducing overfitting by preventing
    neuron co-adaption during training as described in the paper:
    `Improving neural networks by preventing co-adaptation of feature detectors <https://arxiv.org/abs/1207.0580>`_
    The dropout operator randomly sets the outputs of some units to zero, while upscale others
    according to the given dropout probability.

    See :ref:`api_paddle_nn_functional_dropout` for more details.

    In dygraph mode, please use ``eval()`` to switch to evaluation mode, where dropout is disabled.

    Warning:
        The corresponding `functional methods` please reference :ref:`api_paddle_nn_functional_dropout`.

    Parameters:
        p (float|int, optional): Probability of setting units to zero. Default: 0.5
        axis (int|list|tuple|None, optional): The axis along which the dropout is performed. Default: None.
        mode(str, optional): ['upscale_in_train'(default) | 'downscale_in_infer']

                               1. upscale_in_train (default), upscale the output at training time

                                  - train: :math:`out = input \times \frac{mask}{(1.0 - p)}`
                                  - inference: :math:`out = input`

                               2. downscale_in_infer, downscale the output at inference

                                  - train: :math:`out = input \times mask`
                                  - inference: :math:`out = input \times (1.0 - p)`
        name (str|None, optional): Name for the operation, Default: None. For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - input: N-D tensor.
        - output: N-D tensor, the same shape as input.


    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.seed(2023)

            >>> x = paddle.to_tensor([[1, 2, 3], [4, 5, 6]], dtype="float32")
            >>> m = paddle.nn.Dropout(p=0.5)

            >>> y_train = m(x)
            >>> print(y_train)
            Tensor(shape=[2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[2., 4., 0.],
             [8., 0., 0.]])

            >>> m.eval()  # switch the model to test phase
            >>> y_test = m(x)
            >>> print(y_test)
            Tensor(shape=[2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[1., 2., 3.],
             [4., 5., 6.]])
    """

    p: float
    axis: int | Sequence[int] | None
    mode: _DropoutMode
    name: str | None

    def __init__(
        self,
        p: float = 0.5,
        axis: int | Sequence[int] | None = None,
        mode: _DropoutMode = "upscale_in_train",
        name: str | None = None,
    ) -> None:
        super().__init__()

        self.p = p
        self.axis = axis
        self.mode = mode
        self.name = name

    def forward(self, input: Tensor) -> Tensor:
        out = F.dropout(
            input,
            p=self.p,
            axis=self.axis,
            training=self.training,
            mode=self.mode,
            name=self.name,
        )
        return out

    def extra_repr(self) -> str:
        name_str = f', name={self.name}' if self.name else ''
        return f'p={self.p}, axis={self.axis}, mode={self.mode}{name_str}'


class Dropout2D(Layer):
    """
    Randomly zero out entire channels (in the batched input 4d tensor with the shape `NCHW` ,
    a channel is a 2D feature map with the shape `HW`). Each channel will be zeroed out independently
    on every forward call with probability `p` using samples from a Bernoulli distribution.
    Dropout2D will help promote independence between feature maps as described in the paper:
    `Efficient Object Localization Using Convolutional Networks <https://arxiv.org/abs/1411.4280>`_

    See :ref:`api_paddle_nn_functional_dropout2d` for more details.

    In dygraph mode, please use ``eval()`` to switch to evaluation mode, where dropout is disabled.

    Parameters:
        p (float, optional): Probability of setting units to zero. Default: 0.5.
        data_format (str, optional): Specify the data format of the input, and the data format of the output will be consistent with that of the input. An optional string from `NCHW` or `NHWC`. When it is `NCHW`, the data is stored in the order of: [batch_size, input_channels, input_height, input_width]. Default: `NCHW`.
        name (str|None, optional): Name for the operation, Default: None. For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - input: 4-D tensor.
        - output: 4-D tensor, the same shape as input.


    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.seed(100)
            >>> x = paddle.rand([2, 2, 1, 3], dtype="float32")
            >>> print(x)
            Tensor(shape=[2, 2, 1, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[[[0.55355281, 0.20714243, 0.01162981]],
              [[0.51577556, 0.36369765, 0.26091650]]],
             [[[0.18905126, 0.56219709, 0.00808361]],
              [[0.78120756, 0.32112977, 0.90572405]]]])

            >>> m = paddle.nn.Dropout2D(p=0.5)
            >>> y_train = m(x)
            >>> print(y_train)
            Tensor(shape=[2, 2, 1, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[[[1.10710561, 0.41428486, 0.02325963]],
              [[1.03155112, 0.72739530, 0.52183300]]],
             [[[0.        , 0.        , 0.        ]],
              [[0.        , 0.        , 0.        ]]]])

            >>> m.eval()  # switch the model to test phase
            >>> y_test = m(x)
            >>> print(y_test)
            Tensor(shape=[2, 2, 1, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[[[0.55355281, 0.20714243, 0.01162981]],
              [[0.51577556, 0.36369765, 0.26091650]]],
             [[[0.18905126, 0.56219709, 0.00808361]],
              [[0.78120756, 0.32112977, 0.90572405]]]])
    """

    p: float
    data_format: DataLayout2D
    name: str | None

    def __init__(
        self,
        p: float = 0.5,
        data_format: DataLayout2D = 'NCHW',
        name: str | None = None,
    ) -> None:
        super().__init__()

        self.p = p
        self.data_format = data_format
        self.name = name

    def forward(self, input: Tensor) -> Tensor:
        out = F.dropout2d(
            input,
            p=self.p,
            training=self.training,
            data_format=self.data_format,
            name=self.name,
        )
        return out

    def extra_repr(self) -> str:
        name_str = f', name={self.name}' if self.name else ''
        return f'p={self.p}, data_format={self.data_format}{name_str}'


class Dropout3D(Layer):
    """
    Randomly zero out entire channels (in the batched input 5d tensor with the shape `NCDHW` ,
    a channel is a 3D feature map with the shape `DHW` ). Each channel will be zeroed out independently
    on every forward call with probability `p` using samples from a Bernoulli distribution.
    Dropout3D will help promote independence between feature maps as described in the paper:
    `Efficient Object Localization Using Convolutional Networks <https://arxiv.org/abs/1411.4280>`_

    See :ref:`api_paddle_nn_functional_dropout3d` for more details.

    In dygraph mode, please use ``eval()`` to switch to evaluation mode, where dropout is disabled.

    Parameters:
        p (float|int, optional): Probability of setting units to zero. Default: 0.5.
        data_format (str, optional): Specify the data format of the input, and the data format of the output will be consistent with that of the input. An optional string from `NCDHW` or `NDHWC`. When it is `NCDHW`, the data is stored in the order of: [batch_size, input_channels, input_depth, input_height, input_width]. Default: `NCDHW`.
        name (str|None, optional): Name for the operation, Default: None. For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - input: 5-D tensor.
        - output: 5-D tensor, the same shape as input.


    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.arange(24, dtype="float32").reshape((1, 2, 2, 2, 3))
            >>> print(x)
            Tensor(shape=[1, 2, 2, 2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[[[[0. , 1. , 2. ],
                [3. , 4. , 5. ]],
               [[6. , 7. , 8. ],
                [9. , 10., 11.]]],
              [[[12., 13., 14.],
                [15., 16., 17.]],
               [[18., 19., 20.],
                [21., 22., 23.]]]]])

            >>> m = paddle.nn.Dropout3D(p=0.5)
            >>> y_train = m(x)

            >>> m.eval()  # switch the model to test phase
            >>> y_test = m(x)
            >>> print(y_test)
            Tensor(shape=[1, 2, 2, 2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[[[[0. , 1. , 2. ],
                [3. , 4. , 5. ]],
               [[6. , 7. , 8. ],
                [9. , 10., 11.]]],
              [[[12., 13., 14.],
                [15., 16., 17.]],
               [[18., 19., 20.],
                [21., 22., 23.]]]]])
    """

    p: float
    data_format: DataLayout3D
    name: str | None

    def __init__(
        self,
        p: float = 0.5,
        data_format: DataLayout3D = 'NCDHW',
        name: str | None = None,
    ) -> None:
        super().__init__()

        self.p = p
        self.data_format = data_format
        self.name = name

    def forward(self, input: Tensor) -> Tensor:
        out = F.dropout3d(
            input,
            p=self.p,
            training=self.training,
            data_format=self.data_format,
            name=self.name,
        )
        return out

    def extra_repr(self) -> str:
        name_str = f', name={self.name}' if self.name else ''
        return f'p={self.p}, data_format={self.data_format}{name_str}'


class AlphaDropout(Layer):
    """
    Alpha Dropout is a type of Dropout that maintains the self-normalizing property. For an input with
    zero mean and unit standard deviation, the output of Alpha Dropout maintains the original mean and
    standard deviation of the input. Alpha Dropout fits well to SELU activate function by randomly setting
    activations to the negative saturation value.

    For more information, please refer to:
    `Self-Normalizing Neural Networks <https://arxiv.org/abs/1706.02515>`_

    In dygraph mode, please use ``eval()`` to switch to evaluation mode, where dropout is disabled.

    Parameters:
        p (float|int, optional): Probability of setting units to zero. Default: 0.5
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - input: N-D tensor.
        - output: N-D tensor, the same shape as input.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.seed(2023)

            >>> x = paddle.to_tensor([[-1, 1], [-1, 1]], dtype="float32")
            >>> m = paddle.nn.AlphaDropout(p=0.5)
            >>> y_train = m(x)
            >>> print(y_train)
            Tensor(shape=[2, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[-0.10721093,  1.66559887],
             [-0.77919382,  1.66559887]])

            >>> m.eval()  # switch the model to test phase
            >>> y_test = m(x)
            >>> print(y_test)
            Tensor(shape=[2, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[-1.,  1.],
             [-1.,  1.]])
    """

    p: float
    name: str | None

    def __init__(self, p: float = 0.5, name: str | None = None) -> None:
        super().__init__()
        self.p = p
        self.name = name

    def forward(self, input: Tensor) -> Tensor:
        out = F.alpha_dropout(
            input, p=self.p, training=self.training, name=self.name
        )
        return out

    def extra_repr(self) -> str:
        name_str = f', name={self.name}' if self.name else ''
        return f'p={self.p}{name_str}'


class FeatureAlphaDropout(Layer):
    """
    A channel is a feature map, Feature Alpha Dropout randomly masks out entire channels.
    Alpha Dropout is a type of Dropout that maintains the self-normalizing property. For an input with
    zero mean and unit standard deviation, the output of Alpha Dropout maintains the original mean and
    standard deviation of the input. Alpha Dropout fits well to SELU activate function by randomly setting
    activations to the negative saturation value.

    For more information, please refer to:
    `Self-Normalizing Neural Networks <https://arxiv.org/abs/1706.02515>`_

    In dygraph mode, please use ``eval()`` to switch to evaluation mode, where dropout is disabled.

    Parameters:
        p (float | int, optional): Probability of setting units to zero. Default: 0.5
        name (str | None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - input: N-D tensor.
        - output: N-D tensor, the same shape as input.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.seed(2023)

            >>> x = paddle.to_tensor([[-1, 1], [-1, 1]], dtype="float32")
            >>> m = paddle.nn.FeatureAlphaDropout(p=0.5)
            >>> y_train = m(x)
            >>> print(y_train)
            Tensor(shape=[2, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[-0.10721093,  1.66559887],
             [-0.77919382,  1.66559887]])

            >>> m.eval()  # switch the model to test phase
            >>> y_test = m(x)
            >>> print(y_test)
            Tensor(shape=[2, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[-1.,  1.],
             [-1.,  1.]])
    """

    p: float
    name: str | None

    def __init__(self, p: float = 0.5, name: str | None = None) -> None:
        super().__init__()
        self.p = p
        self.name = name

    def forward(self, input: Tensor) -> Tensor:
        out = F.feature_alpha_dropout(
            input, p=self.p, training=self.training, name=self.name
        )
        return out

    def extra_repr(self) -> str:
        name_str = f', name={self.name}' if self.name else ''
        return f'p={self.p}{name_str}'


class Pad1D(Layer):
    """
    This interface is used to construct a callable object of the ``Pad1D`` class.
    Pad tensor according to ``pad``, ``mode`` and ``value``.
    If mode is ``reflect``, pad[0] and pad[1] must be no greater than width-1.

    Parameters:
        padding (Tensor|list[int]|tuple[int]|int): The padding size with data type ``'int'``. If is ``'int'``, use the
            same padding in both dimensions. Else [len(padding)/2] dimensions
            of input will be padded. The pad has the form (pad_left, pad_right).
        mode (str, optional): Four modes: ``'constant'`` (default), ``'reflect'``, ``'replicate'``, ``'circular'``. Default: ``'constant'``.

           - 'constant' mode, uses a constant value to pad the input tensor.
           - 'reflect' mode, uses reflection of the input boundaries to pad the input tensor.
           - 'replicate' mode, uses input boundaries to pad the input tensor.
           - 'circular' mode, uses circular input to pad the input tensor.

        value (float, optional): The value to fill the padded areas. Default is :math:`0.0`.
        data_format (str, optional): An string from: ``'NCL'``, ``'NLC'``. Specify the data format of the input data.
           Default: ``'NCL'``.
        name (str|None, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: ``'None'``.

    Returns:
        None

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn as nn

            >>> input_shape = (1, 2, 3)
            >>> pad = [1, 2]
            >>> data = paddle.arange(paddle.prod(paddle.to_tensor(input_shape)), dtype="float32").reshape(input_shape) + 1
            >>> my_pad = nn.Pad1D(padding=pad, mode="constant")
            >>> result = my_pad(data)
            >>> print(result)
            Tensor(shape=[1, 2, 6], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[[0., 1., 2., 3., 0., 0.],
              [0., 4., 5., 6., 0., 0.]]])
    """

    def __init__(
        self,
        padding: Tensor | Sequence[int] | int,
        mode: _PaddingTensorMode = 'constant',
        value: float = 0.0,
        data_format: DataLayout1D = "NCL",
        name: str | None = None,
    ) -> None:
        super().__init__()
        self._pad = _npairs(padding, 1)
        self._mode = mode
        self._value = value
        self._data_format = data_format
        self._name = name

    def forward(self, x: Tensor) -> Tensor:
        return F.pad(
            x,
            pad=self._pad,
            mode=self._mode,
            value=self._value,
            data_format=self._data_format,
            name=self._name,
        )

    def extra_repr(self) -> str:
        name_str = f', name={self._name}' if self._name else ''
        return f'padding={self._pad}, mode={self._mode}, value={self._value}, data_format={self._data_format}{name_str}'


class ZeroPad1D(Layer):
    """
    This interface is used to construct a callable object of the ``ZeroPad1D`` class.
    Pads the input tensor boundaries with zero.

    Parameters:
        padding (Tensor|list[int]|tuple[int]|int): The padding size with data type int. If is int, use the
            same padding in all dimensions. Else [len(padding)/2] dimensions of input will be padded.
            The pad has the form (pad_left, pad_right).
        data_format (str): An string from: "NCL", "NLC". Specify the data format of the input data.
           Default is  "NCL"
        name (str|None, optional) : The default value is None.  Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - x(Tensor): The input tensor of zeropad1d operator, which is a 3-D tensor.
          The data type can be float32, float64.
        - output(Tensor): The output tensor of zeropad1d operator, which is a 3-D tensor.
          The data type is same as input x.

    Examples:

        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn as nn

            >>> input_shape = (1, 2, 3)
            >>> pad = [1, 2]
            >>> data = paddle.arange(paddle.prod(paddle.to_tensor(input_shape)), dtype="float32").reshape(input_shape) + 1
            >>> my_pad = nn.ZeroPad1D(padding=pad)
            >>> result = my_pad(data)
            >>> print(result)
            Tensor(shape=[1, 2, 6], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[[0., 1., 2., 3., 0., 0.],
              [0., 4., 5., 6., 0., 0.]]])
    """

    def __init__(
        self,
        padding: Tensor | Sequence[int] | int,
        data_format: DataLayout1D = "NCL",
        name: str | None = None,
    ) -> None:
        super().__init__()
        self._pad = _npairs(padding, 1)
        self._mode = 'constant'
        self._value = 0.0
        self._data_format = data_format
        self._name = name

    def forward(self, x: Tensor) -> Tensor:
        return F.pad(
            x,
            pad=self._pad,
            mode=self._mode,
            value=self._value,
            data_format=self._data_format,
            name=self._name,
        )

    def extra_repr(self) -> str:
        name_str = f', name={self._name}' if self._name else ''
        return f'padding={self._pad}, data_format={self._data_format}{name_str}'


class Pad2D(Layer):
    """
    This interface is used to construct a callable object of the ``Pad2D`` class.
    Pad tensor according to ``pad``, ``mode`` and ``value``.
    If mode is ``'reflect'``, pad[0] and pad[1] must be no greater
    than width-1. The height dimension has the same condition.

    Parameters:
        padding (Tensor|list[int]|tuple[int]|int): The padding size with data type ``'int'``. If is ``'int'``, use the
            same padding in all dimensions. Else [len(padding)/2] dimensions of input will be padded.
            The pad has the form (pad_left, pad_right, pad_top, pad_bottom).
        mode (str, optional): Four modes: ``'constant'`` (default), ``'reflect'``, ``'replicate'``, ``'circular'``. Default: ``'constant'``.

           - 'constant' mode, uses a constant value to pad the input tensor.
           - 'reflect' mode, uses reflection of the input boundaries to pad the input tensor.
           - 'replicate' mode, uses input boundaries to pad the input tensor.
           - 'circular' mode, uses circular input to pad the input tensor.

        value (float, optional): The value to fill the padded areas. Default is :math:`0.0`.
        data_format (str, optional): An string from: ``'NCHW'``, ``'NHWC'``. Specify the data format of the input data.
           Default: ``'NCHW'``.
        name (str|None, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: ``'None'``.

    Returns:
        None

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn as nn

            >>> input_shape = (1, 1, 2, 3)
            >>> pad = [1, 0, 1, 2]
            >>> data = paddle.arange(paddle.prod(paddle.to_tensor(input_shape)), dtype="float32").reshape(input_shape) + 1
            >>> my_pad = nn.Pad2D(padding=pad, mode="constant")
            >>> result = my_pad(data)
            >>> print(result)
            Tensor(shape=[1, 1, 5, 4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[[[0., 0., 0., 0.],
               [0., 1., 2., 3.],
               [0., 4., 5., 6.],
               [0., 0., 0., 0.],
               [0., 0., 0., 0.]]]])
    """

    def __init__(
        self,
        padding: Tensor | Sequence[int] | int,
        mode: _PaddingTensorMode = 'constant',
        value: float = 0.0,
        data_format: DataLayout2D = "NCHW",
        name: str | None = None,
    ) -> None:
        super().__init__()
        self._pad = _npairs(padding, 2)
        self._mode = mode
        self._value = value
        self._data_format = data_format
        self._name = name

    def forward(self, x: Tensor) -> Tensor:
        return F.pad(
            x,
            pad=self._pad,
            mode=self._mode,
            value=self._value,
            data_format=self._data_format,
            name=self._name,
        )

    def extra_repr(self) -> str:
        name_str = f', name={self._name}' if self._name else ''
        return f'padding={self._pad}, mode={self._mode}, value={self._value}, data_format={self._data_format}{name_str}'


class ZeroPad2D(Layer):
    """
    This interface is used to construct a callable object of the ``ZeroPad2D`` class.
    Pads the input tensor boundaries with zero.

    Parameters:
        padding (Tensor|list[int]|tuple[int]|int): The padding size with data type int. If is int, use the
            same padding in all dimensions. Else [len(padding)/2] dimensions of input will be padded.
            The pad has the form (pad_left, pad_right, pad_top, pad_bottom).
        data_format (str): An string from: "NCHW", "NHWC". Specify the data format of the input data.
           Default is  "NCHW"
        name (str|None, optional) : The default value is None.  Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - x(Tensor): The input tensor of zeropad2d operator, which is a 4-D tensor.
          The data type can be float32, float64.
        - output(Tensor): The output tensor of zeropad2d operator, which is a 4-D tensor.
          The data type is same as input x.

    Examples:

        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn as nn

            >>> input_shape = paddle.to_tensor([1, 1, 2, 3])
            >>> pad = [1, 0, 1, 2]
            >>> data = paddle.arange(paddle.prod(input_shape), dtype="float32").reshape(input_shape) + 1
            >>> my_pad = nn.ZeroPad2D(padding=pad)
            >>> result = my_pad(data)
            >>> print(result)
            Tensor(shape=[1, 1, 5, 4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[[[0., 0., 0., 0.],
               [0., 1., 2., 3.],
               [0., 4., 5., 6.],
               [0., 0., 0., 0.],
               [0., 0., 0., 0.]]]])
    """

    def __init__(
        self,
        padding: Tensor | Sequence[int] | int,
        data_format: DataLayout2D = "NCHW",
        name: str | None = None,
    ) -> None:
        super().__init__()
        self._pad = _npairs(padding, 2)
        self._mode = 'constant'
        self._value = 0.0
        self._data_format = data_format
        self._name = name

    def forward(self, x: Tensor) -> Tensor:
        return F.pad(
            x,
            pad=self._pad,
            mode=self._mode,
            value=self._value,
            data_format=self._data_format,
            name=self._name,
        )

    def extra_repr(self) -> str:
        name_str = f', name={self._name}' if self._name else ''
        return f'padding={self._pad}, data_format={self._data_format}{name_str}'


class Pad3D(Layer):
    """
    This interface is used to construct a callable object of the ``Pad3D`` class.
    Pad tensor according to ``'pad'``, ``'mode'`` and ``'value'``.
    If mode is ``'reflect'``, pad[0] and pad[1] must be no greater
    than width-1. The height and depth dimension has the same condition.

    Parameters:
        padding (Tensor|list[int]|tuple[int]|int): The padding size with data type ``'int'``. If is ``'int'``, use the
            same padding in all dimensions. Else [len(padding)/2] dimensions
            of input will be padded. The pad has the form (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back).
        mode (str, optional): Four modes: ``'constant'`` (default), ``'reflect'``, ``'replicate'``, ``'circular'``. Default: ``'constant'``.

           - 'constant' mode, uses a constant value to pad the input tensor.
           - 'reflect' mode, uses reflection of the input boundaries to pad the input tensor.
           - 'replicate' mode, uses input boundaries to pad the input tensor.
           - 'circular' mode, uses circular input to pad the input tensor.

        value (float, optional): The value to fill the padded areas. Default is :math:`0.0`.
        data_format (str, optional): An string from: ``'NCDHW'``, ``'NDHWC'``. Specify the data format of the input data.
           Default:  ``'NCDHW'``。
        name (str|None, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: ``'None'``.

    Returns:
        None

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn as nn

            >>> input_shape = (1, 1, 1, 2, 3)
            >>> pad = [1, 0, 1, 2, 0, 0]
            >>> data = paddle.arange(paddle.prod(paddle.to_tensor(input_shape)), dtype="float32").reshape(input_shape) + 1
            >>> my_pad = nn.Pad3D(padding=pad, mode="constant")
            >>> result = my_pad(data)
            >>> print(result)
            Tensor(shape=[1, 1, 1, 5, 4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[[[[0., 0., 0., 0.],
                [0., 1., 2., 3.],
                [0., 4., 5., 6.],
                [0., 0., 0., 0.],
                [0., 0., 0., 0.]]]]])
    """

    def __init__(
        self,
        padding: Tensor | Sequence[int] | int,
        mode: _PaddingTensorMode = 'constant',
        value: float = 0.0,
        data_format: DataLayout3D = "NCDHW",
        name: str | None = None,
    ) -> None:
        super().__init__()
        self._pad = _npairs(padding, 3)
        self._mode = mode
        self._value = value
        self._data_format = data_format
        self._name = name

    def forward(self, x: Tensor) -> Tensor:
        return F.pad(
            x,
            pad=self._pad,
            mode=self._mode,
            value=self._value,
            data_format=self._data_format,
            name=self._name,
        )

    def extra_repr(self) -> str:
        name_str = f', name={self._name}' if self._name else ''
        return f'padding={self._pad}, mode={self._mode}, value={self._value}, data_format={self._data_format}{name_str}'


class ZeroPad3D(Layer):
    """
    This interface is used to construct a callable object of the ``ZeroPad3D`` class.
    Pads the input tensor boundaries with zero.

    Parameters:
        padding (Tensor|list[int]|tuple[int]|int): The padding size with data type int. If is int, use the
            same padding in all dimensions. Else [len(padding)/2] dimensions of input will be padded.
            The pad has the form (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back).
        data_format (str): An string from: "NCDHW", "NDHWC". Specify the data format of the input data.
           Default is  "NCDHW"
        name (str|None, optional) : The default value is None.  Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - x(Tensor): The input tensor of zeropad3d operator, which is a 5-D tensor.
          The data type can be float32, float64.
        - output(Tensor): The output tensor of zeropad3d operator, which is a 5-D tensor.
          The data type is same as input x.

    Examples:

        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn as nn

            >>> input_shape = (1, 1, 1, 2, 3)
            >>> pad = [1, 0, 1, 2, 0, 0]
            >>> data = paddle.arange(paddle.prod(paddle.to_tensor(input_shape)), dtype="float32").reshape(input_shape) + 1
            >>> my_pad = nn.ZeroPad3D(padding=pad)
            >>> result = my_pad(data)
            >>> print(result)
            Tensor(shape=[1, 1, 1, 5, 4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[[[[0., 0., 0., 0.],
                [0., 1., 2., 3.],
                [0., 4., 5., 6.],
                [0., 0., 0., 0.],
                [0., 0., 0., 0.]]]]])
    """

    def __init__(
        self,
        padding: Tensor | Sequence[int] | int,
        data_format: DataLayout3D = "NCDHW",
        name: str | None = None,
    ) -> None:
        super().__init__()
        self._pad = _npairs(padding, 3)
        self._mode = 'constant'
        self._value = 0.0
        self._data_format = data_format
        self._name = name

    def forward(self, x: Tensor) -> Tensor:
        return F.pad(
            x,
            pad=self._pad,
            mode=self._mode,
            value=self._value,
            data_format=self._data_format,
            name=self._name,
        )

    def extra_repr(self) -> str:
        name_str = f', name={self._name}' if self._name else ''
        return f'padding={self._pad}, data_format={self._data_format}{name_str}'


class CosineSimilarity(Layer):
    """
    This interface is used to compute cosine similarity between x1 and x2 along axis.

    Parameters:
        axis (int): Dimension of vectors to compute cosine similarity. Default is 1.
        eps (float): Small value to avoid division by zero. Default is 1e-8.
    Returns:
        None

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

            >>> x1 = paddle.to_tensor([[1., 2., 3.],
            ...                        [2., 3., 4.]], dtype="float32")
            >>> x2 = paddle.to_tensor([[8., 3., 3.],
            ...                        [2., 3., 4.]], dtype="float32")

            >>> cos_sim_func = nn.CosineSimilarity(axis=0)
            >>> result = cos_sim_func(x1, x2)
            >>> print(result)
            Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [0.65079135, 0.98058069, 1.        ])
    """

    def __init__(self, axis: int = 1, eps: float = 1e-8) -> None:
        super().__init__()
        self._axis = axis
        self._eps = eps

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        return F.cosine_similarity(x1, x2, axis=self._axis, eps=self._eps)

    def extra_repr(self) -> str:
        return 'axis={_axis}, eps={_eps}'.format(**self.__dict__)


class Embedding(Layer):
    r"""

    Embedding Layer, used to construct a callable object of the ``Embedding`` class.
    For specific usage, refer to code examples. It implements the function of the Embedding Layer.
    This layer is used to lookup embeddings vector of ids provided by :attr:`x` .
    It automatically constructs a 2D embedding matrix based on the
    input :attr:`num_embeddings` and :attr:`embedding_dim`.

    The shape of output Tensor is generated by appending an emb_size dimension to the
    last dimension of the input Tensor shape.

    Note:
        The id in :attr:`x` must satisfy :math:`0 =< id < num_embeddings` ,
        otherwise the program will throw an exception and exit.

    .. code-block:: text

        Case 1:

        x is a Tensor. padding_idx = -1
            x.data = [[1, 3], [2, 4], [4, 127]
            x.shape = [3, 2]
        Given size = [128, 16]
        output is a Tensor:
            out.shape = [3, 2, 16]
            out.data = [[[0.129435295, 0.244512452, ..., 0.436322452],
                         [0.345421456, 0.524563927, ..., 0.144534654]],
                        [[0.345249859, 0.124939536, ..., 0.194353745],
                         [0.945345345, 0.435394634, ..., 0.435345365]],
                        [[0.945345345, 0.435394634, ..., 0.435345365],
                         [0.0,         0.0,         ..., 0.0        ]]]  # padding data
        The input padding_idx is less than 0, it is automatically converted to padding_idx = -1 + 128 = 127
        It will pad all-zero data when ids is 127.

    Parameters:
        num_embeddings (int): Just one element which indicate the size of the dictionary of embeddings.
        embedding_dim (int):  Just one element which indicate the size of each embedding vector respectively.
        padding_idx(int|float|None, optional): padding_idx needs to be in the interval [-num_embeddings, num_embeddings).
            If :math:`padding\_idx < 0`, the :math:`padding\_idx` will automatically be converted
            to :math:`vocab\_size + padding\_idx` . It will output all-zero padding data whenever lookup
            encounters :math:`padding\_idx` in id. And the padding data will not be updated while training.
            If set None, it makes no effect to output. Default: None.
        max_norm(float, optional): If provided, will renormalize the embedding vectors to have a norm larger than
            :attr:`max\_norm` . It will inplace update the input embedding weight in dynamic graph mode. Default: None.
        norm_type(float, optional): The p of the p-norm to compute for the max_norm option. Default: 2.0.
        sparse(bool, optional): The flag indicating whether to use sparse update. This parameter only
            affects the performance of the backwards gradient update. It is recommended to set
            True because sparse update is faster. But some optimizer does not support sparse update,
            such as :ref:`api_paddle_optimizer_adadelta_Adadelta` , :ref:`api_paddle_optimizer_adamax_Adamax` , :ref:`api_paddle_optimizer_lamb_Lamb`.
            In these case, sparse must be False. Default: False.
        weight_attr(ParamAttr|None, optional): To specify the weight parameter property. Default: None, which means the
            default weight parameter property is used. See usage for details in :ref:`api_paddle_ParamAttr` . In addition,
            user-defined or pre-trained word vectors can be loaded with the :attr:`param_attr` parameter.
            The local word vector needs to be transformed into numpy format, and the shape of local word
            vector should be consistent with :attr:`num_embeddings` . Then :ref:`api_paddle_nn_initializer_Assign`
            is used to load custom or pre-trained word vectors. See code example for details.
        scale_grad_by_freq(bool, optional): Indicating whether to scale the gradients by the inverse frequency of the
            word ids in input `x`. Default: False.
        name(str|None, optional): For detailed information, please refer to :ref:`api_guide_Name`. Usually name is no need to set and
            None by default.

    Attribute:
        **weight** (Parameter): the learnable weights of this layer.

    Returns:
        None

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([[0], [1], [3]], dtype="int64", stop_gradient=False)
            >>> embedding = paddle.nn.Embedding(4, 3, sparse=True)

            >>> w0 = paddle.to_tensor([[0., 0., 0.],
            ...                        [1., 1., 1.],
            ...                        [2., 2., 2.],
            ...                        [3., 3., 3.]], dtype="float32")
            >>> embedding.weight.set_value(w0)
            >>> print(embedding.weight)
            Parameter containing:
            Tensor(shape=[4, 3], dtype=float32, place=Place(cpu), stop_gradient=False,
            [[0., 0., 0.],
             [1., 1., 1.],
             [2., 2., 2.],
             [3., 3., 3.]])

            >>> adam = paddle.optimizer.Adam(parameters=[embedding.weight], learning_rate=0.01)
            >>> adam.clear_grad()

            >>> out = embedding(x)
            >>> print(out)
            Tensor(shape=[3, 1, 3], dtype=float32, place=Place(cpu), stop_gradient=False,
            [[[0., 0., 0.]],
             [[1., 1., 1.]],
             [[3., 3., 3.]]])

            >>> out.backward()
            >>> adam.step()

    """

    weight: Tensor

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: float | None = None,
        max_norm: float | None = None,
        norm_type: float = 2.0,
        sparse: bool = False,
        weight_attr: ParamAttrLike | None = None,
        scale_grad_by_freq: bool = False,
        name: str | None = None,
    ) -> None:
        super().__init__()
        self._num_embeddings = num_embeddings
        self._embedding_dim = embedding_dim
        self._sparse = sparse
        self._is_distributed = False
        self._max_norm = max_norm
        self._norm_type = norm_type
        self._padding_idx = padding_idx
        self._scale_grad_by_freq = scale_grad_by_freq

        if self._num_embeddings <= 0:
            raise ValueError("num_embeddings must be gather than 0")

        if self._embedding_dim <= 0:
            raise ValueError("embedding_dim must be gather than 0")

        padding_idx = (
            -1
            if padding_idx is None
            else (
                padding_idx
                if padding_idx >= 0
                else (num_embeddings + padding_idx)
            )
        )

        if padding_idx >= num_embeddings or padding_idx < -num_embeddings:
            raise ValueError(
                f"padding_idx must be within [-{num_embeddings}, {num_embeddings})"
            )

        self._dtype = self._helper.get_default_dtype()
        self._size = [self._num_embeddings, self._embedding_dim]

        self._weight_attr = weight_attr
        self._remote_prefetch = False
        self._name = name
        self.weight = self.create_parameter(
            attr=self._weight_attr,
            shape=self._size,
            dtype=self._dtype,
            is_bias=False,
        )

        if in_dynamic_mode() and padding_idx != -1:
            with paddle.no_grad():
                self.weight[padding_idx] = 0.0

    def forward(self, x: Tensor) -> Tensor:
        return F.embedding(
            x,
            weight=self.weight,
            padding_idx=self._padding_idx,
            max_norm=self._max_norm,
            norm_type=self._norm_type,
            sparse=self._sparse,
            scale_grad_by_freq=self._scale_grad_by_freq,
            name=self._name,
        )

    def extra_repr(self) -> str:
        main_str = '{_num_embeddings}, {_embedding_dim}'
        if self._padding_idx is not None:
            main_str += ', padding_idx={_padding_idx}'
        main_str += ', sparse={_sparse}'
        main_str += ', scale_grad_by_freq={_scale_grad_by_freq}'
        if self._name is not None:
            main_str += ', name={_name}'
        return main_str.format(**self.__dict__)


class Unfold(Layer):
    """
    Returns a col buffer of sliding local blocks of input x, also known
    as im2col for batched 2D image tensors. For each block under the convolution filter,
    all element will be rearranged as a column. While the convolution filter sliding over
    the input feature map, a series of such columns will be formed.

    For each input :math:`x` with shape [N, C, H, W], the output shape [N, Cout, Lout]
    can be calculated as following.

    See ``paddle.nn.functional.unfold`` for more details.


    Parameters:
        kernel_sizes(int|list|tuple): The size of convolution kernel, should be [k_h, k_w]
            or an integer k treated as [k, k].
        strides(int|list|tuple, optional): The strides, should be [stride_h, stride_w]
            or an integer stride treated as [sride, stride]. For default, strides will be [1, 1].
        paddings(int|list|tuple, optional): The paddings of each dimension, should be
            [padding_top, padding_left, padding_bottom, padding_right] or [padding_h, padding_w]
            or an integer padding. If [padding_h, padding_w] was given, it will expanded to
            [padding_h, padding_w, padding_h, padding_w]. If an integer padding was given,
            [padding, padding, padding, padding] will be used. For default,
            paddings will be [0, 0, 0, 0].
        dilations(int|list|tuple, optional): The dilations of convolution kernel, should be
            [dilation_h, dilation_w], or an integer dilation treated as [dilation, dilation].
            For default, it will be [1, 1].
        name(str|None, optional): The default value is None. Normally there is no need for user to
            set this property. For more information, please refer to :ref:`api_guide_Name`


    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn as nn

            >>> x = paddle.randn((100, 3, 224, 224))
            >>> unfold = nn.Unfold(kernel_sizes=[3, 3])
            >>> result = unfold(x)
            >>> print(result.shape)
            [100, 27, 49284]

    """

    kernel_sizes: Size2
    dilations: Size2
    paddings: Size2 | Size4
    strides: Size2
    name: str | None

    def __init__(
        self,
        kernel_sizes: Size2,
        dilations: Size2 = 1,
        paddings: Size2 | Size4 = 0,
        strides: Size2 = 1,
        name: str | None = None,
    ) -> None:
        super().__init__()

        self.kernel_sizes = kernel_sizes
        self.dilations = dilations
        self.paddings = paddings
        self.strides = strides
        self.name = name

    def forward(self, input: Tensor) -> Tensor:
        return F.unfold(
            input,
            kernel_sizes=self.kernel_sizes,
            strides=self.strides,
            paddings=self.paddings,
            dilations=self.dilations,
            name=self.name,
        )

    def extra_repr(self) -> str:
        name_str = f', name={self.name}' if self.name else ''
        return f'kernel_size={self.kernel_sizes}, dilation={self.dilations}, padding={self.paddings}, stride={self.strides}{name_str}'


class Fold(Layer):
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
        output_sizes(list):  The size of output size, should be [output_size_h, output_size_w]
                                  or an integer o treated as [o, o].
        kernel_sizes(int|list|tuple):   The size of convolution kernel, should be [k_h, k_w]
                                  or an integer k treated as [k, k].
        strides(int|list|tuple, optional):  The strides, should be [stride_h, stride_w]
                                  or an integer stride treated as [stride, stride].
                                  For default, strides will be [1, 1].
        paddings(int|list|tuple, optional):  The paddings of each dimension, should be
                                  [padding_top, padding_left, padding_bottom, padding_right]
                                  or [padding_h, padding_w] or an integer padding.
                                  If [padding_h, padding_w] was given, it will expanded to
                                  [padding_h, padding_w, padding_h, padding_w]. If an integer
                                  padding was given, [padding, padding, padding, padding] will
                                  be used. For default, paddings will be [0, 0, 0, 0]
        dilations(int|list|tuple, optional): The dilations of convolution kernel, should be
                                  [dilation_h, dilation_w], or an integer dilation treated as
                                  [dilation, dilation]. For default, it will be [1, 1].
        name(str|None, optional): The default value is None.
                             Normally there is no need for user to set this property.
                             For more information, please refer to :ref:`api_guide_Name`


    Returns:
        The tensor formed by combining a group of sliding local blocks
        The output shape is [N, Cout, H, W] as decriabled above.

    Examples:

        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn as nn

            >>> x = paddle.randn([2, 3*2*2, 12])
            >>> fold = nn.Fold(output_sizes=[4, 5], kernel_sizes=2)
            >>> y = fold(x)
            >>> print(y.shape)
            [2, 3, 4, 5]
   """

    output_sizes: Size2
    kernel_sizes: Size2
    dilations: Size2
    paddings: Size2 | Size4
    strides: Size2
    name: str | None

    def __init__(
        self,
        output_sizes: Size2,
        kernel_sizes: Size2,
        dilations: Size2 = 1,
        paddings: Size2 | Size4 = 0,
        strides: Size2 = 1,
        name: str | None = None,
    ) -> None:
        super().__init__()

        self.output_sizes = output_sizes
        self.kernel_sizes = kernel_sizes
        self.dilations = dilations
        self.paddings = paddings
        self.strides = strides
        self.name = name

    def forward(self, input: Tensor) -> Tensor:
        return F.fold(
            input,
            output_sizes=self.output_sizes,
            kernel_sizes=self.kernel_sizes,
            strides=self.strides,
            paddings=self.paddings,
            dilations=self.dilations,
            name=self.name,
        )

    def extra_repr(self) -> str:
        name_str = f', name={self.name}' if self.name else ''
        return f'kernel_size={self.kernel_sizes}, dilation={self.dilations}, padding={self.paddings}, stride={self.strides}{name_str}'


class Flatten(Layer):
    """
    This interface is used to construct a callable object of the ``FLatten`` class.
    For more details, refer to code examples.
    It implements flatten a contiguous range of dims into a tensor.

    Parameters:
        start_axis(int): first dim to flatten (default = 1)
        stop_axis(int): last dim to flatten (default = -1).

    Returns:
        None

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> inp = paddle.ones([5, 2, 3, 4]).astype('float32')
            >>> flatten = paddle.nn.Flatten(start_axis=1, stop_axis=2)
            >>> y = flatten(inp)
            >>> print(y.shape)
            [5, 6, 4]

    """

    start_axis: int
    stop_axis: int

    def __init__(self, start_axis: int = 1, stop_axis: int = -1) -> None:
        super().__init__()
        self.start_axis = start_axis
        self.stop_axis = stop_axis

    def forward(self, input: Tensor) -> Tensor:
        out = paddle.flatten(
            input, start_axis=self.start_axis, stop_axis=self.stop_axis
        )
        return out


class Unflatten(Layer):
    """
    This interface is used to construct a callable object of the ``Unflatten`` class.
    For more details, refer to code examples.
    It a certain dimension of the input x Tensor into a desired shape.

    Parameters:
        axis (int): :attr:`axis` to be unflattened, specified as an index into `x.shape`.
        shape (list|tuple|Tensor): Unflatten :attr:`shape` on the specified :attr:`axis`. At most one dimension of the target :attr:`shape` can be -1.
            If the input :attr:`shape` does not contain -1 , the product of all elements in ``shape`` should be equal to ``x.shape[axis]``.
            The data type is `int` . If :attr:`shape` is a list or tuple, the elements of it should be integers or Tensors with shape [].
            If :attr:`shape` is an Tensor, it should be an 1-D Tensor.
        name(str|None, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        None

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> x = paddle.randn(shape=[4, 6, 8])
            >>> shape = [2, 3]
            >>> axis = 1
            >>> unflatten = paddle.nn.Unflatten(axis, shape)
            >>> res = unflatten(x)
            >>> print(res.shape)
            [4, 2, 3, 8]

    """

    axis: int
    shape: ShapeLike
    name: str | None

    def __init__(
        self, axis: int, shape: ShapeLike, name: str | None = None
    ) -> None:
        super().__init__()
        self.axis = axis
        self.shape = shape
        self.name = name

    def forward(self, input: Tensor) -> Tensor:
        out = paddle.unflatten(
            input, axis=self.axis, shape=self.shape, name=self.name
        )
        return out

    def extra_repr(self) -> str:
        name_str = f', name={self.name}' if self.name else ''
        return f'axis={self.axis}, shape={self.shape}{name_str}'
