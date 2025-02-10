# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

import paddle

from . import _C_ops
from .base.data_feeder import check_variable_and_dtype
from .base.layer_helper import LayerHelper
from .framework import in_dynamic_or_pir_mode
from .tensor.attribute import is_floating_point, is_integer
from .tensor.creation import _complex_to_real_dtype, _real_to_complex_dtype

if TYPE_CHECKING:
    from typing import Literal

    from paddle import Tensor
    from paddle._typing import DTypeLike

    _NormalizeMode = Literal["forward", "backward", "ortho"]
__all__ = [
    'fft',
    'ifft',
    'rfft',
    'irfft',
    'hfft',
    'ihfft',
    'fft2',
    'ifft2',
    'rfft2',
    'irfft2',
    'hfft2',
    'ihfft2',
    'fftn',
    'ifftn',
    'rfftn',
    'irfftn',
    'hfftn',
    'ihfftn',
    'fftfreq',
    'rfftfreq',
    'fftshift',
    'ifftshift',
]


def _check_normalization(norm):
    if norm not in ['forward', 'backward', 'ortho']:
        raise ValueError(
            f"Unexpected norm: {norm}. Norm should be forward, backward or ortho"
        )


def _check_fft_n(n):
    if not isinstance(n, int):
        raise ValueError(
            f"Invalid FFT argument n({n}), it should be an integer."
        )
    if n <= 0:
        raise ValueError(f"Invalid FFT argument n({n}), it should be positive.")


def _check_fft_shape(x, s):
    ndim = x.ndim
    if not isinstance(s, Sequence):
        raise ValueError(
            "Invalid FFT argument s({}), it should be a sequence of integers."
        )

    if len(s) > ndim:
        raise ValueError(
            "Length of FFT argument s should not be larger than the rank of input. "
            f"Received s: {s}, rank of x: {ndim}"
        )
    for size in s:
        if not isinstance(size, int) or size <= 0:
            raise ValueError(f"FFT sizes {s} contains invalid value ({size})")


def _check_fft_axis(x, axis):
    ndim = x.ndim
    if not isinstance(axis, int):
        raise ValueError(f"Invalid FFT axis ({axis}), it should be an integer.")
    if axis < -ndim or axis >= ndim:
        raise ValueError(
            f"Invalid FFT axis ({axis}), it should be in range [-{ndim}, {ndim})"
        )


def _check_fft_axes(x, axes):
    ndim = x.ndim
    if not isinstance(axes, Sequence):
        raise ValueError(
            f"Invalid FFT axes ({axes}), it should be a sequence of integers."
        )
    if len(axes) > ndim:
        raise ValueError(
            "Length of fft axes should not be larger than the rank of input. "
            f"Received, len of axes: {len(axes)}, rank of x: {ndim}"
        )
    for axis in axes:
        if not isinstance(axis, int) or axis < -ndim or axis >= ndim:
            raise ValueError(
                f"FFT axes {axes} contains invalid value ({axis}), it should be in range [-{ndim}, {ndim})"
            )


def _resize_fft_input(x, s, axes):
    if len(s) != len(axes):
        raise ValueError("length of `s` should equals length of `axes`.")
    shape = x.shape
    ndim = x.ndim

    axes_to_pad = []
    paddings = []
    axes_to_slice = []
    slices = []
    for i, axis in enumerate(axes):
        if shape[axis] < s[i]:
            axes_to_pad.append(axis)
            paddings.append(s[i] - shape[axis])
        elif shape[axis] > s[i]:
            axes_to_slice.append(axis)
            slices.append((0, s[i]))

    if axes_to_slice:
        x = paddle.slice(
            x,
            axes_to_slice,
            starts=[item[0] for item in slices],
            ends=[item[1] for item in slices],
        )
    if axes_to_pad:
        padding_widths = [0] * (2 * ndim)
        for axis, pad in zip(axes_to_pad, paddings):
            padding_widths[2 * axis + 1] = pad
        x = paddle.nn.functional.pad(x, padding_widths)
    return x


def _normalize_axes(x, axes):
    ndim = x.ndim
    return [item if item >= 0 else (item + ndim) for item in axes]


def _check_at_least_ndim(x, rank):
    if x.ndim < rank:
        raise ValueError(f"The rank of the input ({x.ndim}) should >= {rank}")


# public APIs 1d
def fft(
    x: Tensor,
    n: int | None = None,
    axis: int = -1,
    norm: _NormalizeMode = "backward",
    name: str | None = None,
) -> Tensor:
    """
    Calculate one-dimensional discrete Fourier transform.

    This function uses the efficient fast Fourier transform (FFT) algorithm [1] to
    calculate the 1-D * n * point discrete Fourier transform (DFT).

    Args:
        x (Tensor): The input data. It's a Tensor type. It's a complex.
        n (int|None, optional): The length of the output transform axis. If `n` is less than
            the length input, the input will be cropped. If larger, the input is filled
            with zeros. If `n` is not given, the input length along the axis specified
            by `axis` is used.
        axis (int, optional): Axis used to calculate FFT. If not specified, the last axis
            is used by default.
        norm (str, optional): Indicates which direction to scale the `forward` or `backward` transform
            pair and what normalization factor to use. The parameter value must be one
            of "forward" or "backward" or "ortho". Default is "backward", meaning no normalization on
            the forward transforms and scaling by ``1/n`` on the `ifft`. "forward" instead applies
            the ``1/n`` factor on the forward transform. For ``norm="ortho"``, both directions are
            scaled by ``1/sqrt(n)``.
        name (str|None, optional): The default value is None.  Normally there is no need for user to set
            this property. For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        complex tensor. The truncated or zero-padded input, transformed along the axis indicated
        by `axis`, or the last one if `axis` is not specified.

    Examples:

        .. code-block:: python

            >>> import numpy as np
            >>> import paddle

            >>> x = np.exp(3j * np.pi * np.arange(7) / 7)
            >>> xp = paddle.to_tensor(x)
            >>> fft_xp = paddle.fft.fft(xp).numpy().round(3)
            >>> print(fft_xp)
            [1.+1.254j 1.+4.381j 1.-4.381j 1.-1.254j 1.-0.482j 1.+0.j 1.+0.482j]


    """
    if is_integer(x) or is_floating_point(x):
        return fft_r2c(
            x, n, axis, norm, forward=True, onesided=False, name=name
        )
    else:
        return fft_c2c(x, n, axis, norm, forward=True, name=name)


def ifft(
    x: Tensor,
    n: int | None = None,
    axis: int = -1,
    norm: _NormalizeMode = "backward",
    name: str | None = None,
) -> Tensor:
    """
    Compute the 1-D inverse discrete Fourier Transform.

    This function computes the inverse of the 1-D *n*-point discrete Fourier transform
    computed by `fft`.  In other words, ``ifft(fft(x)) == x`` to within numerical accuracy.

    The input should be ordered in the same way as is returned by `fft`,
    i.e.,

    * ``x[0]`` should contain the zero frequency term,
    * ``x[1:n//2]`` should contain the positive-frequency terms,
    * ``x[n//2 + 1:]`` should contain the negative-frequency terms, in
      increasing order starting from the most negative frequency.

    For an even number of input points, ``x[n//2]`` represents the sum of
    the values at the positive and negative Nyquist frequencies, as the two
    are aliased together.

    Args:
        x (Tensor): The input data. It's a Tensor type. It's a complex.
        n (int, optional): The length of the output transform axis. If `n` is less than
            the length input, the input will be cropped. If larger, the input is filled
            with zeros. If `n` is not given, the input length along the axis specified
            by `axis` is used.
        axis (int, optional): Axis used to calculate FFT. If not specified, the last axis
            is used by default.
        norm (str, optional): Indicates which direction to scale the `forward` or `backward` transform
            pair and what normalization factor to use. The parameter value must be one
            of "forward" or "backward" or "ortho". Default is "backward", meaning no normalization on
            the forward transforms and scaling by ``1/n`` on the `ifft`. "forward" instead applies
            the ``1/n`` factor on the forward transform. For ``norm="ortho"``, both directions are
            scaled by ``1/sqrt(n)``.
        name (str, optional): The default value is None.  Normally there is no need for user to set
            this property. For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        complex tensor. The truncated or zero-padded input, transformed along the axis indicated
        by `axis`, or the last one if `axis` is not specified.

    Examples:

        .. code-block:: python

            >>> import numpy as np
            >>> import paddle

            >>> x = np.exp(3j * np.pi * np.arange(7) / 7)
            >>> xp = paddle.to_tensor(x)
            >>> ifft_xp = paddle.fft.ifft(xp).numpy().round(3)
            >>> print(ifft_xp)
            [0.143+0.179j 0.143+0.069j 0.143+0.j 0.143-0.069j 0.143-0.179j 0.143-0.626j 0.143+0.626j]

    """
    if is_integer(x) or is_floating_point(x):
        return fft_r2c(
            x, n, axis, norm, forward=False, onesided=False, name=name
        )
    else:
        return fft_c2c(x, n, axis, norm, forward=False, name=name)


def rfft(
    x: Tensor,
    n: int | None = None,
    axis: int = -1,
    norm: _NormalizeMode = "backward",
    name: str | None = None,
) -> Tensor:
    """
    The one dimensional FFT for real input.

    This function computes the one dimensional *n*-point discrete Fourier
    Transform (DFT) of a real-valued tensor by means of an efficient algorithm
    called the Fast Fourier Transform (FFT).

    When the DFT is computed for purely real input, the output is
    Hermitian-symmetric. This function does not compute the negative frequency
    terms, and the length of the transformed axis of the output is therefore
    ``n//2 + 1``.

    Args:
        x (Tensor) : Real-valued input tensor
        n (int, optional): Number of points along transformation axis in the
            input to use. If `n` is smaller than the length of the input, the
            input is cropped. If it is larger, the input is padded with zeros.
            If `n` is not given, the length of the input along the axis
            specified by `axis` is used.
        axis (int, optional): Axis over which to compute the FFT. Default value
            is last axis.
        norm (str, optional) : Normalization mode, indicates which direction of
            the forward/backward  pair of transforms is scaled and with what
            normalization factor. Include {"backward", "ortho", "forward"},
            default value is "backward".

                - "backward": The factor of forward direction and backward direction are ``1`` and ``1/n`` respectively;
                - "forward": The factor of forward direction and backward direction are ``1/n`` and ``1`` respectively;
                - "ortho": The factor of forward direction and backward direction are both ``1/sqrt(n)``.

            Where ``n`` is the multiplication of each element in  ``s`` .
        name (str|None, optional): The default value is None.  Normally there is no
            need for user to set this property. For more information, please
            refer to :ref:`api_guide_Name` .

    Returns:
        out(Tensor) : complex tensor

    Examples:

    .. code-block:: python

        >>> import paddle

        >>> x = paddle.to_tensor([0.0, 1.0, 0.0, 0.0])
        >>> print(paddle.fft.rfft(x))
        Tensor(shape=[3], dtype=complex64, place=Place(cpu), stop_gradient=True,
        [(1+0j), -1j, (-1+0j)])
    """
    return fft_r2c(x, n, axis, norm, forward=True, onesided=True, name=name)


def irfft(
    x: Tensor,
    n: int | None = None,
    axis: int = -1,
    norm: _NormalizeMode = "backward",
    name: str | None = None,
) -> Tensor:
    """
    Computes the inverse of `rfft`.

    This function calculates the inverse of the one-dimensional *n* point discrete
    Fourier transform of the actual input calculated by "rfft". In other words,
    ``irfft(rfft(a),len(a)) == a`` is within the numerical accuracy range.

    The input shall be in the form of "rfft", i.e. the actual zero frequency term,
    followed by the complex positive frequency term, in the order of increasing frequency.
    Because the discrete Fourier transform of the actual input is Hermite symmetric,
    the negative frequency term is regarded as the complex conjugate term of the corresponding
    positive frequency term.

    Args:
        x (Tensor): The input data. It's a Tensor type. It's a complex.
        n (int, optional): The length of the output transform axis. For `n` output
            points, ``n//2 + 1``input points are necessary. If the length of the input tensor is greater
            than `n`, it will be cropped, if it is shorter than this, fill in zero. If `n` is not given,
            it is considered to be ``2 * (k-1)``, where ``k`` is the length of the input axis specified
            along the ` axis'.
        axis (int, optional): Axis used to calculate FFT. If not specified, the last axis
            is used by default.
        norm (str, optional): Indicates which direction to scale the `forward` or `backward` transform
            pair and what normalization factor to use. The parameter value must be one
            of "forward" or "backward" or "ortho". Default is "backward".
        name (str|None, optional): The default value is None.  Normally there is no need for user to set
            this property. For more information, please refer to :ref:`api_guide_Name` .

    Returns:
        Real tensor. Truncated or zero fill input for the transformation along the axis indicated by
        `axis`, or the last input if `axis` is not specified. The length of the conversion axis
        is `n`, or ``2 * k-2``, if `k` is None, where `k` is the length of the input conversion axis.
        If the output is an odd number, you need to specify the value of 'n', such as ``2 * k-1``
        in some cases.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([1, -1j, -1])
            >>> irfft_x = paddle.fft.irfft(x)
            >>> print(irfft_x)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [0., 1., 0., 0.])
    """
    return fft_c2r(x, n, axis, norm, forward=False, name=name)


def hfft(
    x: Tensor,
    n: int | None = None,
    axis: int = -1,
    norm: _NormalizeMode = "backward",
    name: str | None = None,
) -> Tensor:
    """
    Compute the FFT of a signal that has Hermitian symmetry, a real
    spectrum.

    Args:
        x (Tensor): The input data. It's a Tensor type. It's a complex.
        n (int, optional): The length of the output transform axis. For `n` output
            points, ``n//2 + 1`` input points are necessary. If the length of the input tensor is greater
            than `n`, it will be cropped, if it is shorter than this, fill in zero. If `n` is not given,
            it is considered to be ``2 * (k-1)``, where ``k`` is the length of the input axis specified
            along the ` axis'.
        axis (int,optional): Axis used to calculate FFT. If not specified, the last axis
            is used by default.
        norm (str, optional): Indicates which direction to scale the `forward` or `backward` transform
            pair and what normalization factor to use. The parameter value must be one
            of "forward" or "backward" or "ortho". Default is "backward".
        name (str, optional): The default value is None.  Normally there is no need for user to set
            this property. For more information, please refer to :ref:`api_guide_Name` .

    Returns:
        Real tensor. Truncated or zero fill input for the transformation along the axis indicated by
        `axis`, or the last input if `axis` is not specified. The length of the conversion axis
        is `n`, or ``2 * k-2``, if `k` is None, where `k` is the length of the input conversion axis.
        If the output is an odd number, you need to specify the value of 'n', such as ``2 * k-1`` in
        some cases.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([1, -1j, -1])
            >>> hfft_x = paddle.fft.hfft(x)
            >>> print(hfft_x)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [0., 0., 0., 4.])
    """

    return fft_c2r(x, n, axis, norm, forward=True, name=name)


def ihfft(
    x: Tensor,
    n: int | None = None,
    axis: int = -1,
    norm: _NormalizeMode = "backward",
    name: str | None = None,
) -> Tensor:
    """
    The inverse FFT of a signal that has Hermitian symmetry.

    This function computes the one dimensional *n*-point inverse FFT of a signal
    that has Hermitian symmetry by means of an efficient algorithm called
    the Fast Fourier Transform (FFT).

    When the DFT is computed for purely real input, the output is
    Hermitian-symmetric. This function does not compute the negative frequency
    terms, and the length of the transformed axis of the output is therefore
    ``n//2 + 1``.

    Args:
        x (Tensor): Input tensor.
        n (int|None, optional): The number of points along transformation axis in the
            input to use.  If `n` is smaller than the length of the input, the
            input is cropped.  If it is larger, the input is padded with zeros.
            If `n` is not given, the length of the input along the axis
            specified by `axis` is used.
        axis (int, optional) : Axis over which to compute the inverse FFT. If not
            given, the last axis is used.
        norm (str, optional) : Normalization mode, indicates which direction of
            the forward/backward pair of transforms is scaled and with what
            normalization factor. Include {"backward", "ortho", "forward"},
            default value is "backward".
        name (str|None, optional): The default value is None.  Normally there is no
            need for user to set this property. For more information, please
            refer to :ref:`api_guide_Name` .

    Returns:
        out(Tensor) : complex tensor.

    Examples:

    .. code-block:: python

        >>> import paddle

        >>> spectrum = paddle.to_tensor([10.0, -5.0, 0.0, -1.0, 0.0, -5.0])
        >>> print(paddle.fft.ifft(spectrum))
        Tensor(shape=[6], dtype=complex64, place=Place(cpu), stop_gradient=True,
        [(-0.1666666716337204+0j), (1-0j),
        (2.3333334922790527-0j), (3.5+0j),
        (2.3333334922790527+0j), (1+0j)])

        >>> print(paddle.fft.ihfft(spectrum))
        Tensor(shape=[4], dtype=complex64, place=Place(cpu), stop_gradient=True,
        [(-0.1666666716337204+0j), (1-0j),
        (2.3333334922790527-0j), (3.5+0j)])

    """
    return fft_r2c(x, n, axis, norm, forward=False, onesided=True, name=name)


# public APIs nd
def fftn(
    x: Tensor,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: _NormalizeMode = "backward",
    name: str | None = None,
) -> Tensor:
    """
    Compute the N-D discrete Fourier Transform.

    This function calculates the n-D discrete Fourier transform on any number of axes
    in the M-D array by fast Fourier transform (FFT).

    Args:
        x (Tensor): The input data. It's a Tensor type. It's a complex.
        s (sequence of ints, optional): Shape (length of each transformed axis) of the output
            (``s[0]`` refers to axis 0, ``s[1]`` to axis 1, etc.).
            This corresponds to ``n`` for ``fft(x, n)``.
            Along any axis, if the given shape is smaller than that of the input,
            the input is cropped. If it is larger, the input is padded with zeros.
            if `s` is not given, the shape of the input along the axes specified
            by `axes` is used.
        axes (sequence of ints, optional): Axes used to calculate FFT. If not given, the last ``len(s)``
            axes are used, or all axes if `s` is also not specified.
        norm (str, optional): Indicates which direction to scale the `forward` or `backward` transform
            pair and what normalization factor to use. The parameter value must be one
            of "forward" or "backward" or "ortho". Default is "backward", meaning no normalization on
            the forward transforms and scaling by ``1/n`` on the `ifft`. "forward" instead applies
            the ``1/n`` factor on the forward transform. For ``norm="ortho"``, both directions are
            scaled by ``1/sqrt(n)``.
        name (str, optional): The default value is None.  Normally there is no need for user to set
            this property. For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        complex tensor. The truncated or zero-padded input, transformed along the axes indicated by
        `axes`, or by a combination of `s` and `x`, as explained in the parameters section above.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> arr = paddle.arange(4, dtype="float64")
            >>> x = paddle.meshgrid(arr, arr, arr)[1]

            >>> fftn_xp = paddle.fft.fftn(x, axes=(1, 2))
            >>> print(fftn_xp)
            Tensor(shape=[4, 4, 4], dtype=complex128, place=Place(cpu), stop_gradient=True,
            [[[(24+0j), 0j, 0j, -0j],
              [(-8+8j), 0j, 0j, -0j],
              [(-8+0j), 0j, 0j, -0j],
              [(-8-8j), 0j, 0j, -0j]],
             [[(24+0j), 0j, 0j, -0j],
              [(-8+8j), 0j, 0j, -0j],
              [(-8+0j), 0j, 0j, -0j],
              [(-8-8j), 0j, 0j, -0j]],
             [[(24+0j), 0j, 0j, -0j],
              [(-8+8j), 0j, 0j, -0j],
              [(-8+0j), 0j, 0j, -0j],
              [(-8-8j), 0j, 0j, -0j]],
             [[(24+0j), 0j, 0j, -0j],
              [(-8+8j), 0j, 0j, -0j],
              [(-8+0j), 0j, 0j, -0j],
              [(-8-8j), 0j, 0j, -0j]]])
    """
    if is_integer(x) or is_floating_point(x):
        return fftn_r2c(
            x, s, axes, norm, forward=True, onesided=False, name=name
        )
    else:
        return fftn_c2c(x, s, axes, norm, forward=True, name=name)


def ifftn(
    x: Tensor,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: _NormalizeMode = "backward",
    name: str | None = None,
) -> Tensor:
    """
    Compute the N-D inverse discrete Fourier Transform.

    This function computes the inverse of the N-D discrete
    Fourier Transform over any number of axes in an M-D array by
    means of the Fast Fourier Transform (FFT).  In other words,
    ``ifftn(fftn(x)) == x`` to within numerical accuracy.

    The input, analogously to `ifft`, should be ordered in the same way as is
    returned by `fftn`, i.e., it should have the term for zero frequency
    in all axes in the low-order corner, the positive frequency terms in the
    first half of all axes, the term for the Nyquist frequency in the middle
    of all axes and the negative frequency terms in the second half of all
    axes, in order of decreasingly negative frequency.

    Args:
        x (Tensor): The input data. It's a Tensor type. It's a complex.
        s (sequence of ints|None, optional): Shape (length of each transformed axis) of the output
            (``s[0]`` refers to axis 0, ``s[1]`` to axis 1, etc.).
            This corresponds to ``n`` for ``fft(x, n)``.
            Along any axis, if the given shape is smaller than that of the input,
            the input is cropped. If it is larger, the input is padded with zeros.
            if `s` is not given, the shape of the input along the axes specified
            by `axes` is used.
        axes (sequence of ints|None, optional): Axes used to calculate FFT. If not given, the last ``len(s)``
            axes are used, or all axes if `s` is also not specified.
        norm (str, optional): Indicates which direction to scale the `forward` or `backward` transform
            pair and what normalization factor to use. The parameter value must be one
            of "forward" or "backward" or "ortho". Default is "backward", meaning no normalization on
            the forward transforms and scaling by ``1/n`` on the `ifft`. "forward" instead applies
            the ``1/n`` factor on the forward transform. For ``norm="ortho"``, both directions are
            scaled by ``1/sqrt(n)``.
        name (str|None, optional): The default value is None.  Normally there is no need for user to set
            this property. For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        complex tensor. The truncated or zero-padded input, transformed along the axes indicated by
        `axes`, or by a combination of `s` and `x`, as explained in the parameters section above.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> x = paddle.eye(3)
            >>> ifftn_x = paddle.fft.ifftn(x, axes=(1,))
            >>> print(ifftn_x)
            Tensor(shape=[3, 3], dtype=complex64, place=Place(cpu), stop_gradient=True,
            [[(0.3333333432674408+0j),
              (0.3333333432674408-0j),
              (0.3333333432674408+0j)],
             [(0.3333333432674408+0j),
              (-0.1666666716337204+0.28867512941360474j),
              (-0.1666666716337204-0.28867512941360474j)],
             [(0.3333333432674408+0j),
              (-0.1666666716337204-0.28867512941360474j),
              (-0.1666666716337204+0.28867512941360474j)]])
    """
    if is_integer(x) or is_floating_point(x):
        return fftn_r2c(
            x, s, axes, norm, forward=False, onesided=False, name=name
        )
    else:
        return fftn_c2c(x, s, axes, norm, forward=False, name=name)


def rfftn(
    x: Tensor,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: _NormalizeMode = "backward",
    name: str | None = None,
) -> Tensor:
    """

    The N dimensional FFT for real input.

    This function computes the N-dimensional discrete Fourier Transform over
    any number of axes in an M-dimensional real array by means of the Fast
    Fourier Transform (FFT).  By default, all axes are transformed, with the
    real transform performed over the last axis, while the remaining
    transforms are complex.

    The transform for real input is performed over the last transformation
    axis, as by `rfft`, then the transform over the remaining axes is
    performed as by `fftn`.  The order of the output is as for `rfft` for the
    final transformation axis, and as for `fftn` for the remaining
    transformation axes.

    Args:
        x (Tensor) : Input tensor, taken to be real.
        s (Sequence[int]|None, optional) : Shape to use from the exec fft. The final element of
            `s` corresponds to `n` for ``rfft(x, n)``, while for the remaining
            axes, it corresponds to `n` for ``fft(x, n)``. Along any axis, if
            the given shape is smaller than that of the input, the input is
            cropped.  If it is larger, the input is padded with zeros. if `s` is
            not given, the shape of the input along the axes specified by `axes`
            is used.
        axes(Sequence[int]|None, optional) : Axes over which to compute the FFT.  If not given,
            the last ``len(s)`` axes are used, or all axes if `s` is also not
            specified.
        norm(str, optional) : Normalization mode, indicates which direction of
            the forward/backward pair of transforms is scaled and with what
            normalization factor. Include {"backward", "ortho", "forward"},
            default value is "backward". The details of
            three operations are shown below:

                - "backward": The factor of forward direction and backward direction are ``1``
                  and ``1/n`` respectively;
                - "forward": The factor of forward direction and backward direction are ``1/n``
                  and ``1`` respectively;
                - "ortho": The factor of forward direction and backward direction are both ``1/sqrt(n)``.

            Where ``n`` is the multiplication of each element in  ``s`` .
        name(str|None, optional): The default value is None.  Normally there is no
            need for user to set this property. For more information, please
            refer to :ref:`api_guide_Name` .

    Returns:
        out(Tensor), complex tensor

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> # default, all axis will be used to exec fft
            >>> x = paddle.ones((2, 3, 4))
            >>> print(paddle.fft.rfftn(x))
            Tensor(shape=[2, 3, 3], dtype=complex64, place=Place(cpu), stop_gradient=True,
            [[[(24+0j), 0j, 0j],
              [0j, 0j, 0j],
              [0j, 0j, 0j]],
             [[0j, 0j, 0j],
              [0j, 0j, 0j],
              [0j, 0j, 0j]]])

            >>> # use axes(2, 0)
            >>> print(paddle.fft.rfftn(x, axes=(2, 0)))
            Tensor(shape=[2, 3, 4], dtype=complex64, place=Place(cpu), stop_gradient=True,
            [[[(8+0j), 0j, 0j, 0j],
              [(8+0j), 0j, 0j, 0j],
              [(8+0j), 0j, 0j, 0j]],
             [[0j, 0j, 0j, 0j],
              [0j, 0j, 0j, 0j],
              [0j, 0j, 0j, 0j]]])

    """
    return fftn_r2c(x, s, axes, norm, forward=True, onesided=True, name=name)


def irfftn(
    x: Tensor,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: _NormalizeMode = "backward",
    name: str | None = None,
) -> Tensor:
    """
    Computes the inverse of `rfftn`.

    This function computes the inverse of the N-D discrete
    Fourier Transform for real input over any number of axes in an
    M-D array by means of the Fast Fourier Transform (FFT). In
    other words, ``irfftn(rfftn(x), x.shape) == x`` to within numerical
    accuracy. (The ``x.shape`` is necessary like ``len(x)`` is for `irfft`,
    and for the same reason.)

    The input should be ordered in the same way as is returned by `rfftn`,
    i.e., as for `irfft` for the final transformation axis, and as for `ifftn`
    along all the other axes.

    Args:
        x (Tensor): The input data. It's a Tensor type.
        s (sequence of ints|None, optional): The length of the output transform axis.
            (``s[0]`` refers to axis 0, ``s[1]`` to axis 1, etc.).

            - `s` is also the number of input points used along this axis, except for the last axis, where ``s[-1]//2+1`` points of the input are used.
            - Along any axis, if the shape indicated by `s` is smaller than that of the input, the input is cropped. If it is larger, the input is padded with zeros.
            - If `s` is not given, the shape of the input along the axes specified by axes is used. Except for the last axis which is taken to be ``2*(k-1)``

            where ``k`` is the length of the input along that axis.

        axes (sequence of ints|None, optional): Axes over which to compute the inverse FFT. If not given, the last
            `len(s)` axes are used, or all axes if `s` is also not specified.
        norm (str): Indicates which direction to scale the `forward` or `backward` transform
            pair and what normalization factor to use. The parameter value must be one
            of "forward" or "backward" or "ortho". Default is "backward". The details of
            three operations are shown below:

                - "backward": The factor of forward direction and backward direction are ``1`` and ``1/n`` respectively;
                - "forward": The factor of forward direction and backward direction are ``1/n`` and ``1`` respectively;
                - "ortho": The factor of forward direction and backward direction are both ``1/sqrt(n)``.

            Where ``n`` is the multiplication of each element in  ``s`` .
        name (str|None, optional): The default value is None.  Normally there is no need for user to set
            this property. For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Real tensor. The truncated or zero-padded input, transformed along the axes indicated by `axes`,
        or by a combination of `s` or `x`, as explained in the parameters section above. The length of
        each transformed axis is as given by the corresponding element of `s`, or the length of the input
        in every axis except for the last one if `s` is not given. In the final transformed axis the length
        of the output when `s` is not given is ``2*(m-1)``, where ``m`` is the length of the final
        transformed axis of the input. To get an odd number of output points in the final axis,
        `s` must be specified.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([2.+2.j, 2.+2.j, 3.+3.j]).astype(paddle.complex128)
            >>> print(x)
            Tensor(shape=[3], dtype=complex128, place=Place(cpu), stop_gradient=True,
            [(2+2j), (2+2j), (3+3j)])

            >>> irfftn_x = paddle.fft.irfftn(x)
            >>> print(irfftn_x)
            Tensor(shape=[4], dtype=float64, place=Place(cpu), stop_gradient=True,
            [2.25000000, -1.25000000, 0.25000000, 0.75000000])

    """
    return fftn_c2r(x, s, axes, norm, forward=False, name=name)


def hfftn(
    x: Tensor,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: _NormalizeMode = "backward",
    name: str | None = None,
) -> Tensor:
    """
    Compute the N-D FFT of Hermitian symmetric complex input, i.e., a
    signal with a real spectrum.

    This function calculates the n-D discrete Fourier transform of Hermite symmetric
    complex input on any axis in M-D array by fast Fourier transform (FFT).
    In other words, ``ihfftn(hfftn(x, s)) == x`` is within the numerical accuracy range.
    (``s`` here are ``x.shape`` and ``s[-1] = x.shape[- 1] * 2 - 1``. This is necessary
    for the same reason that ``irfft`` requires ``x.shape``.)

    Args:
        x (Tensor): The input data. It's a Tensor type.
        s (sequence of ints|None, optional): The length of the output transform axis.
            (``s[0]`` refers to axis 0, ``s[1]`` to axis 1, etc.). `s` is also the
            number of input points used along this axis, except for the last axis,
            where ``s[-1]//2+1`` points of the input are used. Along any axis, if
            the shape indicated by `s` is smaller than that of the input, the input
            is cropped. If it is larger, the input is padded with zeros.
            If `s` is not given, the shape of the input along the axes specified by axes
            is used. Except for the last axis which is taken to be ``2*(k-1)`` where
            ``k`` is the length of the input along that axis.
        axes (sequence of ints|None, optional): Axes over which to compute the inverse FFT. If not given, the last
            `len(s)` axes are used, or all axes if `s` is also not specified.
        norm (str, optional): Indicates which direction to scale the `forward` or `backward` transform
            pair and what normalization factor to use. The parameter value must be one
            of "forward" or "backward" or "ortho". Default is "backward".
        name (str|None, optional): The default value is None.  Normally there is no need for user to set
            this property. For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Real tensor. Truncate or zero fill input, transforming along the axis indicated by axis or
        a combination of `s` or `X`.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([(2+2j), (2+2j), (3+3j)])
            >>> hfftn_x = paddle.fft.hfftn(x)
            >>> print(hfftn_x)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [9., 3., 1., -5.])
    """
    return fftn_c2r(x, s, axes, norm, forward=True, name=name)


def ihfftn(
    x: Tensor,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: _NormalizeMode = "backward",
    name: str | None = None,
) -> Tensor:
    """
    The n dimensional inverse FFT of a signal that has Hermitian symmetry.

    This function computes the n dimensional inverse FFT over any number of axes
    in an M-dimensional of a signal that has Hermitian symmetry by means of an
    efficient algorithm called the Fast Fourier Transform (FFT).

    Args:
        x (Tensor): Input tensor.
        s (Sequence[int]|None, optional) : Shape (length along each transformed axis)
            to use from the input. (``s[0]`` refers to axis 0, ``s[1]`` to axis
            1, etc.). Along any axis, if the given shape is smaller than that
            of the input, the input is cropped. If it is larger, the input is
            padded with zeros. if `s` is not given, the shape of the input
            along the axes specified by `axes` is used.
        axes (Sequence[int]|None, optional) : Axis over which to compute the inverse FFT. If not
            given, the last axis is used.
        norm(str, optional) : Normalization mode, indicates which direction of
            the forward/backward pair of transforms is scaled and with what
            normalization factor. Include {"backward", "ortho", "forward"},
            default value is "backward".
        name(str|None, optional): The default value is None.  Normally there is no
            need for user to set this property. For more information, please
            refer to :ref:`api_guide_Name` .

    Returns:
        out(Tensor) : complex tensor.

    Examples:

    .. code-block:: python

        >>> import paddle

        >>> spectrum = paddle.to_tensor([10.0, -5.0, 0.0, -1.0, 0.0, -5.0])
        >>> print(paddle.fft.ifft(spectrum))
        Tensor(shape=[6], dtype=complex64, place=Place(cpu), stop_gradient=True,
        [(-0.1666666716337204+0j), (1-0j),
         (2.3333334922790527-0j), (3.5+0j),
         (2.3333334922790527+0j), (1+0j)])

        >>> print(paddle.fft.ihfft(spectrum))
        Tensor(shape=[4], dtype=complex64, place=Place(cpu), stop_gradient=True,
        [(-0.1666666716337204+0j), (1-0j),
         (2.3333334922790527-0j), (3.5+0j)])
    """
    return fftn_r2c(x, s, axes, norm, forward=False, onesided=True, name=name)


# public APIs 2d
def fft2(
    x: Tensor,
    s: list[int] | tuple[int, int] | None = None,
    axes: list[int] | tuple[int, int] = (-2, -1),
    norm: _NormalizeMode = "backward",
    name: str | None = None,
) -> Tensor:
    """
    Compute the 2-D discrete Fourier Transform

    This function computes the N-D discrete Fourier Transform
    over any axes in an M-D array by means of the
    Fast Fourier Transform (FFT). By default, the transform is computed over
    the last two axes of the input array, i.e., a 2-dimensional FFT.

    Args:
        x (Tensor): The input data. It's a Tensor type.
        s (sequence of ints|None, optional): Shape (length of each transformed axis) of the output.
            It should be a sequence of 2 integers. This corresponds to ``n`` for ``fft(x, n)``.
            Along each axis, if the given shape is smaller than that of the input,
            the input is cropped. If it is larger, the input is padded with zeros.
            if `s` is not given, the shape of the input along the axes specified
            by `axes` is used. Default is None.
        axes (sequence of ints, optional):  Axes over which to compute the FFT. It should be a
            sequence of 2 integers. If not specified, the last two axes are used by default.
        norm (str, optional): Indicates which direction to scale the `forward` or `backward` transform
            pair and what normalization factor to use. The parameter value must be one
            of "forward" or "backward" or "ortho". Default is "backward".
        name (str|None, optional): The default value is None.  Normally there is no need for user to set
            this property. For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Complex tensor. The truncated or zero-padded input, transformed along the axes indicated by `axes`,
        or the last two axes if `axes` is not given.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> arr = paddle.arange(2, dtype="float64")
            >>> x = paddle.meshgrid(arr, arr)[0]

            >>> fft2_xp = paddle.fft.fft2(x)
            >>> print(fft2_xp)
            Tensor(shape=[2, 2], dtype=complex128, place=Place(cpu), stop_gradient=True,
            [[(2+0j), 0j],
             [(-2+0j), 0j]])

    """
    _check_at_least_ndim(x, 2)
    if s is not None:
        if not isinstance(s, Sequence) or len(s) != 2:
            raise ValueError(
                f"Invalid FFT argument s ({s}), it should be a sequence of 2 integers."
            )
    if axes is not None:
        if not isinstance(axes, Sequence) or len(axes) != 2:
            raise ValueError(
                f"Invalid FFT argument axes ({axes}), it should be a sequence of 2 integers."
            )
    return fftn(x, s, axes, norm, name)


def ifft2(
    x: Tensor,
    s: list[int] | tuple[int, int] | None = None,
    axes: list[int] | tuple[int, int] = (-2, -1),
    norm: _NormalizeMode = "backward",
    name: str | None = None,
) -> Tensor:
    """
    Compute the 2-D inverse discrete Fourier Transform.

    This function computes the inverse of the 2-D discrete Fourier
    Transform over any number of axes in an M-D array by means of
    the Fast Fourier Transform (FFT). In other words, ``ifft2(fft2(x)) == x``
    to within numerical accuracy. By default, the inverse transform is
    computed over the last two axes of the input array.

    The input, analogously to `ifft`, should be ordered in the same way as is
    returned by `fft2`, i.e., it should have the term for zero frequency
    in the low-order corner of the two axes, the positive frequency terms in
    the first half of these axes, the term for the Nyquist frequency in the
    middle of the axes and the negative frequency terms in the second half of
    both axes, in order of decreasingly negative frequency.

    Args:
        x (Tensor): The input data. It's a Tensor type.
        s (sequence of ints|None, optional): Shape (length of each transformed axis) of the output.
            It should be a sequence of 2 integers. This corresponds to ``n`` for ``ifft(x, n)``.
            Along each axis, if the given shape is smaller than that of the input,
            the input is cropped. If it is larger, the input is padded with zeros.
            if `s` is not given, the shape of the input along the axes specified
            by `axes` is used. Default is None.
        axes (sequence of ints, optional):  Axes over which to compute the inverse FFT. It should be a
            sequence of 2 integers. If not specified, the last two axes are used by default.
        norm (str, optional): Indicates which direction to scale the `forward` or `backward` transform
            pair and what normalization factor to use. The parameter value must be one
            of "forward" or "backward" or "ortho". Default is "backward".
        name (str|None, optional): The default value is None.  Normally there is no need for user to set
            this property. For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Complex tensor. The truncated or zero-padded input, transformed along the axes indicated by `axes`,
        or the last two axes if `axes` is not given.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> arr = paddle.arange(2, dtype="float64")
            >>> x = paddle.meshgrid(arr, arr)[0]

            >>> ifft2_xp = paddle.fft.ifft2(x)
            >>> print(ifft2_xp)
            Tensor(shape=[2, 2], dtype=complex128, place=Place(cpu), stop_gradient=True,
            [[(0.5+0j), 0j],
             [(-0.5+0j), 0j]])
    """
    _check_at_least_ndim(x, 2)
    if s is not None:
        if not isinstance(s, Sequence) or len(s) != 2:
            raise ValueError(
                f"Invalid FFT argument s ({s}), it should be a sequence of 2 integers."
            )
    if axes is not None:
        if not isinstance(axes, Sequence) or len(axes) != 2:
            raise ValueError(
                f"Invalid FFT argument axes ({axes}), it should be a sequence of 2 integers."
            )
    return ifftn(x, s, axes, norm, name)


def rfft2(
    x: Tensor,
    s: list[int] | tuple[int, int] | None = None,
    axes: list[int] | tuple[int, int] = (-2, -1),
    norm: _NormalizeMode = "backward",
    name: str | None = None,
) -> Tensor:
    """
    The two dimensional FFT with real tensor input.

    This is really just `rfftn` with different default behavior.
    For more details see `rfftn`.

    Args:
        x (Tensor): Input tensor, taken to be real.
        s (sequence[int]|None, optional): Shape (length of each transformed axis) of the output.
            It should be a sequence of 2 integers. This corresponds to ``n`` for ``rfft(x, n)``.
            Along each axis, if the given shape is smaller than that of the input,
            the input is cropped. If it is larger, the input is padded with zeros.
            if `s` is not given, the shape of the input along the axes specified
            by `axes` is used. Default is None.
        axes (sequence[int], optional):  Axes over which to compute the FFT. It should be a
            sequence of 2 integers. If not specified, the last two axes are used by default.
        norm (str, optional) : {"backward", "ortho", "forward"},
            default is "backward". Indicates which direction of the
            forward/backward pair of transforms is scaled and with what
            normalization factor. The details of
            three operations are shown below:

                - "backward": The factor of forward direction and backward direction are ``1`` and ``1/n`` respectively;
                - "forward": The factor of forward direction and backward direction are ``1/n`` and ``1`` respectively;
                - "ortho": The factor of forward direction and backward direction are both ``1/sqrt(n)``.

            Where ``n`` is the multiplication of each element in  ``s`` .
        name (str|None, optional): The default value is None.  Normally there is no
            need for user to set this property. For more information, please
            refer to :ref:`api_guide_Name` .

    Returns:
        out(Tensor): The result of the real 2-D FFT.

    Examples:

    .. code-block:: python

        >>> import paddle

        >>> arr = paddle.arange(5, dtype="float64")
        >>> x = paddle.meshgrid(arr, arr)[0]

        >>> result = paddle.fft.rfft2(x)
        >>> print(result.numpy())
        [[50. +0.j 0. +0.j 0. +0.j]
         [-12.5+17.20477401j 0. +0.j 0. +0.j]
         [-12.5 +4.0614962j 0. +0.j 0. +0.j]
         [-12.5 -4.0614962j 0. +0.j 0. +0.j]
         [-12.5-17.20477401j 0. +0.j 0. +0.j]]
    """
    _check_at_least_ndim(x, 2)
    if s is not None:
        if not isinstance(s, Sequence) or len(s) != 2:
            raise ValueError(
                f"Invalid FFT argument s ({s}), it should be a sequence of 2 integers."
            )
    if axes is not None:
        if not isinstance(axes, Sequence) or len(axes) != 2:
            raise ValueError(
                f"Invalid FFT argument axes ({axes}), it should be a sequence of 2 integers."
            )
    return rfftn(x, s, axes, norm, name)


def irfft2(
    x: Tensor,
    s: list[int] | tuple[int, int] | None = None,
    axes: list[int] | tuple[int, int] = (-2, -1),
    norm: _NormalizeMode = "backward",
    name: str | None = None,
) -> Tensor:
    """
    Computes the inverse of `rfft2`.

    Args:
        x (Tensor): The input data. It's a Tensor type.
        s (sequence[int]|None, optional): Shape (length of each transformed axis) of the output.
            It should be a sequence of 2 integers. This corresponds to ``n`` for ``irfft(x, n)``.
            Along each axis, if the given shape is smaller than that of the input,
            the input is cropped. If it is larger, the input is padded with zeros.
            if `s` is not given, the shape of the input along the axes specified
            by `axes` is used. Default is None.
        axes (sequence[int], optional):  Axes over which to compute the inverse FFT. It should be a
            sequence of 2 integers. If not specified, the last two axes are used by default.
        norm (str, optional): Indicates which direction to scale the `forward` or `backward` transform
            pair and what normalization factor to use. The parameter value must be one
            of "forward" or "backward" or "ortho". Default is "backward". The details of
            three operations are shown below:

                - "backward": The factor of forward direction and backward direction are ``1`` and ``1/n`` respectively;
                - "forward": The factor of forward direction and backward direction are ``1/n`` and ``1`` respectively;
                - "ortho": The factor of forward direction and backward direction are both ``1/sqrt(n)``.

            Where ``n`` is the multiplication of each element in  ``s`` .
        name (str|None, optional): The default value is None.  Normally there is no need for user to set
            this property. For more information, please refer to :ref:`api_guide_Name` .

    Returns:
        Real tensor. The result of the inverse real 2-D FFT.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([[3.+3.j, 2.+2.j, 3.+3.j], [2.+2.j, 2.+2.j, 3.+3.j]])
            >>> irfft2_x = paddle.fft.irfft2(x)
            >>> print(irfft2_x)
            Tensor(shape=[2, 4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[2.37500000, -1.12500000, 0.37500000, 0.87500000],
             [0.12500000, 0.12500000, 0.12500000, 0.12500000]])
    """
    _check_at_least_ndim(x, 2)
    if s is not None:
        if not isinstance(s, Sequence) or len(s) != 2:
            raise ValueError(
                f"Invalid FFT argument s ({s}), it should be a sequence of 2 integers."
            )
    if axes is not None:
        if not isinstance(axes, Sequence) or len(axes) != 2:
            raise ValueError(
                f"Invalid FFT argument axes ({axes}), it should be a sequence of 2 integers."
            )
    return irfftn(x, s, axes, norm, name)


def hfft2(
    x: Tensor,
    s: list[int] | tuple[int, int] | None = None,
    axes: list[int] | tuple[int, int] = (-2, -1),
    norm: _NormalizeMode = "backward",
    name: str | None = None,
) -> Tensor:
    """
    Compute the 2-D FFT of a Hermitian complex array.

    Args:
        x (Tensor): The input data. It's a Tensor type.
        s (sequence[int]|None, optional): Shape (length of each transformed axis) of the output.
            It should be a sequence of 2 integers. This corresponds to ``n`` for ``hfft(x, n)``.
            Along each axis, if the given shape is smaller than that of the input,
            the input is cropped. If it is larger, the input is padded with zeros.
            if `s` is not given, the shape of the input along the axes specified
            by `axes` is used. Default is None.
        axes (sequence[int], optional):  Axes over which to compute the FFT. It should be a
            sequence of 2 integers. If not specified, the last two axes are used by default.
        norm (str): Indicates which direction to scale the `forward` or `backward` transform
            pair and what normalization factor to use. The parameter value must be one
            of "forward" or "backward" or "ortho". Default is "backward".
        name (str|None, optional): The default value is None.  Normally there is no need for user to set
            this property. For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Real tensor. The real result of the 2-D Hermitian complex real FFT.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([[3.+3.j, 2.+2.j, 3.+3.j], [2.+2.j, 2.+2.j, 3.+3.j]])
            >>> hfft2_x = paddle.fft.hfft2(x)
            >>> print(hfft2_x)
            Tensor(shape=[2, 4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[19., 7., 3., -9.],
             [1., 1., 1., 1.]])
    """
    _check_at_least_ndim(x, 2)
    if s is not None:
        if not isinstance(s, Sequence) or len(s) != 2:
            raise ValueError(
                f"Invalid FFT argument s ({s}), it should be a sequence of 2 integers."
            )
    if axes is not None:
        if not isinstance(axes, Sequence) or len(axes) != 2:
            raise ValueError(
                f"Invalid FFT argument axes ({axes}), it should be a sequence of 2 integers."
            )
    return hfftn(x, s, axes, norm, name)


def ihfft2(
    x: Tensor,
    s: list[int] | tuple[int, int] | None = None,
    axes: list[int] | tuple[int, int] = (-2, -1),
    norm: _NormalizeMode = "backward",
    name: str | None = None,
) -> Tensor:
    """
    Compute the two dimensional inverse FFT of a real spectrum.

    This is really `ihfftn` with different defaults.
    For more details see `ihfftn`.

    Args:
        x (Tensor): Input tensor.
        s (sequence[int]|None, optional): Shape (length of each transformed axis) of the output.
            It should be a sequence of 2 integers. This corresponds to ``n`` for ``ihfft(x, n)``.
            Along each axis, if the given shape is smaller than that of the input,
            the input is cropped. If it is larger, the input is padded with zeros.
            if `s` is not given, the shape of the input along the axes specified
            by `axes` is used. Default is None.
        axes (sequence[int], optional):  Axes over which to compute the inverse FFT. It should be a
            sequence of 2 integers. If not specified, the last two axes are used by default.
        norm (str, optional): {"backward", "ortho", "forward"}. Default is
            "backward".
        name (str, optional): The default value is None.  Normally there is no
            need for user to set this property. For more information, please
            refer to :ref:`api_guide_Name` .

    Returns:
        out(Tensor) : The result of the inverse hermitian 2-D FFT.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> arr = paddle.arange(5, dtype="float64")
            >>> x = paddle.meshgrid(arr, arr)[0]
            >>> print(x)
            Tensor(shape=[5, 5], dtype=float64, place=Place(cpu), stop_gradient=True,
            [[0., 0., 0., 0., 0.],
             [1., 1., 1., 1., 1.],
             [2., 2., 2., 2., 2.],
             [3., 3., 3., 3., 3.],
             [4., 4., 4., 4., 4.]])

            >>> ihfft2_xp = paddle.fft.ihfft2(x)
            >>> print(ihfft2_xp.numpy())
            [[2. +0.j 0. -0.j 0. -0.j]
             [-0.5-0.68819096j 0. +0.j 0. +0.j]
             [-0.5-0.16245985j 0. +0.j 0. +0.j]
             [-0.5+0.16245985j 0. +0.j 0. +0.j]
             [-0.5+0.68819096j 0. +0.j 0. +0.j]]
    """
    _check_at_least_ndim(x, 2)
    if s is not None:
        if not isinstance(s, Sequence) or len(s) != 2:
            raise ValueError(
                f"Invalid FFT argument s ({s}), it should be a sequence of 2 integers."
            )
    if axes is not None:
        if not isinstance(axes, Sequence) or len(axes) != 2:
            raise ValueError(
                f"Invalid FFT argument axes ({axes}), it should be a sequence of 2 integers."
            )
    return ihfftn(x, s, axes, norm, name)


# public APIs utilities
def fftfreq(
    n: int,
    d: float = 1.0,
    dtype: DTypeLike | None = None,
    name: str | None = None,
) -> Tensor:
    """
    Return the Discrete Fourier Transform sample frequencies.

    The returned float array `f` contains the frequency bin centers in cycles
    per unit of the sample spacing (with zero at the start).  For instance, if
    the sample spacing is in seconds, then the frequency unit is cycles/second.

    Given input length `n` and a sample spacing `d`::

      f = [0, 1, ...,   n/2-1,     -n/2, ..., -1] / (d*n)   if n is even
      f = [0, 1, ..., (n-1)/2, -(n-1)/2, ..., -1] / (d*n)   if n is odd

    Args:
        n (int): Dimension inputted.
        d (float, optional): Sample spacing (inverse of the sampling rate). Defaults is 1.
        dtype (str, optional): The data type of returns. Defaults is the data type of returns
            of ``paddle.get_default_dtype()``.
        name (str, optional): The default value is None.  Normally there is no need for user to set
            this property. For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor. A tensor of length 'n' containing the sampling frequency.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> scalar_temp = 0.5
            >>> fftfreq_xp = paddle.fft.fftfreq(5, d=scalar_temp)
            >>> print(fftfreq_xp)
            Tensor(shape=[5], dtype=float32, place=Place(cpu), stop_gradient=True,
            [0., 0.40000001, 0.80000001, -0.80000001, -0.40000001])
    """
    if d * n == 0:
        raise ValueError("d or n should not be 0.")

    dtype = paddle.framework.get_default_dtype()
    val = 1.0 / (n * d)
    pos_max = (n + 1) // 2
    neg_max = n // 2
    indices = paddle.arange(-neg_max, pos_max, dtype=dtype, name=name)
    indices = paddle.roll(indices, -neg_max, name=name)
    return indices * val


def rfftfreq(
    n: int,
    d: float = 1.0,
    dtype: DTypeLike | None = None,
    name: str | None = None,
) -> Tensor:
    """
    Return the Discrete Fourier Transform sample frequencies.

    The returned floating-point array "F" contains the center of the frequency unit,
    and the unit is the number of cycles of the sampling interval (the starting point is zero).

    Given input length `n` and a sample spacing `d`::

      f = [0, 1, ...,     n/2-1,     n/2] / (d*n)   if n is even
      f = [0, 1, ..., (n-1)/2-1, (n-1)/2] / (d*n)   if n is odd

    the Nyquist frequency component is considered to be positive.

    Args:
        n (int): Dimension inputted.
        d (float, optional): Sample spacing (inverse of the sampling rate). Defaults is 1.
        dtype (str, optional): The data type of returns. Defaults is the data type of returns
            of ``paddle.get_default_dtype()``.
        name (str, optional): The default value is None.  Normally there is no need for user to set
            this property. For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor. A tensor of length ``n//2 + 1`` containing the sample frequencies.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> scalar_temp = 0.3
            >>> rfftfreq_xp = paddle.fft.rfftfreq(5, d=scalar_temp)
            >>> print(rfftfreq_xp)
            Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [0., 0.66666669, 1.33333337])

    """
    if d * n == 0:
        raise ValueError("d or n should not be 0.")

    dtype = paddle.framework.get_default_dtype()
    val = 1.0 / (n * d)
    pos_max = 1 + n // 2
    indices = paddle.arange(0, pos_max, dtype=dtype, name=name)
    return indices * val


def fftshift(
    x: Tensor, axes: Sequence[int] | None = None, name: str | None = None
) -> Tensor:
    """
    Shift the zero-frequency component to the center of the spectrum.

    This function swaps half spaces for all the axes listed (all by default).
    Note that ``y[0]`` is the Nyquist component only if ``len(x)`` is even.

    Args:
        n (int): Dimension inputted.
        axes (int|tuple|None, optional): The axis on which to move. The default is none, which moves all axes.
            Default is None.
        name (str|None, optional): The default value is None.  Normally there is no need for user to set
            this property. For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor. The shifted tensor.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> fftfreq_xp = paddle.fft.fftfreq(5, d=0.3)
            >>> print(fftfreq_xp)
            Tensor(shape=[5], dtype=float32, place=Place(cpu), stop_gradient=True,
            [0., 0.66666669, 1.33333337, -1.33333337, -0.66666669])

            >>> res = paddle.fft.fftshift(fftfreq_xp)
            >>> print(res)
            Tensor(shape=[5], dtype=float32, place=Place(cpu), stop_gradient=True,
            [-1.33333337, -0.66666669, 0., 0.66666669, 1.33333337])

    """
    shape = paddle.shape(x)
    if axes is None:
        # shift all axes
        rank = len(x.shape)
        axes = list(range(0, rank))
        shifts = shape // 2
    elif isinstance(axes, int):
        shifts = shape[axes] // 2
    else:
        shifts = paddle.concat([shape[ax : ax + 1] // 2 for ax in axes])
    return paddle.roll(x, shifts, axes, name=name)


def ifftshift(
    x: Tensor, axes: Sequence[int] | None = None, name: str | None = None
) -> Tensor:
    """
    The inverse of `fftshift`. Although the even length 'x' is the same, the function of the
    odd length 'x' is different. An example.

    Args:
        n (int): Dimension inputted.
        axes (int|tuple|None, optional): The axis on which to move. The default is none, which moves all axes.
            Default is None.
        name (str|None, optional): The default value is None.  Normally there is no need for user to set
            this property. For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor. The shifted tensor.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> fftfreq_xp = paddle.fft.fftfreq(5, d=0.3)
            >>> print(fftfreq_xp)
            Tensor(shape=[5], dtype=float32, place=Place(cpu), stop_gradient=True,
            [0., 0.66666669, 1.33333337, -1.33333337, -0.66666669])

            >>> res = paddle.fft.ifftshift(fftfreq_xp)
            >>> print(res)
            Tensor(shape=[5], dtype=float32, place=Place(cpu), stop_gradient=True,
            [1.33333337, -1.33333337, -0.66666669, 0., 0.66666669])

    """
    shape = paddle.shape(x)
    if axes is None:
        # shift all axes
        rank = len(x.shape)
        axes = list(range(0, rank))
        shifts = (shape + 1) // 2
    elif isinstance(axes, int):
        shifts = (shape[axes] + 1) // 2
    else:
        shifts = paddle.concat([(shape[ax : ax + 1] + 1) // 2 for ax in axes])
    return paddle.roll(x, shifts, axes, name=name)


# internal functions
def fft_c2c(x, n, axis, norm, forward, name):
    if is_integer(x):
        x = paddle.cast(x, _real_to_complex_dtype(paddle.get_default_dtype()))
    elif is_floating_point(x):
        x = paddle.cast(x, _real_to_complex_dtype(x.dtype))
    _check_normalization(norm)

    axis = axis if axis is not None else -1
    _check_fft_axis(x, axis)
    axes = [axis]
    axes = _normalize_axes(x, axes)
    if n is not None:
        _check_fft_n(n)
        s = [n]
        x = _resize_fft_input(x, s, axes)

    if in_dynamic_or_pir_mode():
        out = _C_ops.fft_c2c(x, axes, norm, forward)
    else:
        op_type = 'fft_c2c'
        check_variable_and_dtype(x, 'x', ['complex64', 'complex128'], op_type)
        inputs = {
            'X': [x],
        }
        attrs = {'axes': axes, 'normalization': norm, 'forward': forward}
        helper = LayerHelper(op_type, **locals())
        dtype = helper.input_dtype(input_param_name='x')
        out = helper.create_variable_for_type_inference(dtype)
        outputs = {"Out": [out]}
        helper.append_op(
            type=op_type, inputs=inputs, outputs=outputs, attrs=attrs
        )
    return out


def fft_r2c(x, n, axis, norm, forward, onesided, name):
    if is_integer(x):
        x = paddle.cast(x, paddle.get_default_dtype())
    _check_normalization(norm)
    axis = axis if axis is not None else -1
    _check_fft_axis(x, axis)
    axes = [axis]
    axes = _normalize_axes(x, axes)
    if n is not None:
        _check_fft_n(n)
        s = [n]
        x = _resize_fft_input(x, s, axes)
    if in_dynamic_or_pir_mode():
        out = _C_ops.fft_r2c(x, axes, norm, forward, onesided)
    else:
        op_type = 'fft_r2c'
        check_variable_and_dtype(
            x, 'x', ['float16', 'float32', 'float64'], op_type
        )
        inputs = {
            'X': [x],
        }
        attrs = {
            'axes': axes,
            'normalization': norm,
            'forward': forward,
            'onesided': onesided,
        }
        helper = LayerHelper(op_type, **locals())
        dtype = helper.input_dtype(input_param_name='x')
        out = helper.create_variable_for_type_inference(
            _real_to_complex_dtype(dtype)
        )
        outputs = {"Out": [out]}
        helper.append_op(
            type=op_type, inputs=inputs, outputs=outputs, attrs=attrs
        )
    return out


def fft_c2r(x, n, axis, norm, forward, name):
    if is_integer(x):
        x = paddle.cast(x, _real_to_complex_dtype(paddle.get_default_dtype()))
    elif is_floating_point(x):
        x = paddle.cast(x, _real_to_complex_dtype(x.dtype))
    _check_normalization(norm)
    axis = axis if axis is not None else -1
    _check_fft_axis(x, axis)
    axes = [axis]
    axes = _normalize_axes(x, axes)
    if n is not None:
        _check_fft_n(n)
        s = [n // 2 + 1]
        x = _resize_fft_input(x, s, axes)

    if in_dynamic_or_pir_mode():
        if n is not None:
            out = _C_ops.fft_c2r(x, axes, norm, forward, n)
        else:
            out = _C_ops.fft_c2r(x, axes, norm, forward, 0)
    else:
        op_type = 'fft_c2r'
        check_variable_and_dtype(x, 'x', ['complex64', 'complex128'], op_type)
        inputs = {
            'X': [x],
        }
        attrs = {'axes': axes, 'normalization': norm, 'forward': forward}
        if n is not None:
            attrs['last_dim_size'] = n
        helper = LayerHelper(op_type, **locals())
        dtype = helper.input_dtype(input_param_name='x')
        out = helper.create_variable_for_type_inference(
            _complex_to_real_dtype(dtype)
        )
        outputs = {"Out": [out]}
        helper.append_op(
            type=op_type, inputs=inputs, outputs=outputs, attrs=attrs
        )
    return out


def fftn_c2c(x, s, axes, norm, forward, name):
    if is_integer(x):
        x = paddle.cast(x, _real_to_complex_dtype(paddle.get_default_dtype()))
    elif is_floating_point(x):
        x = paddle.cast(x, _real_to_complex_dtype(x.dtype))
    _check_normalization(norm)
    if s is not None:
        _check_fft_shape(x, s)

    rank = x.ndim
    if axes is None:
        if s is None:
            axes = list(range(rank))
        else:
            fft_ndims = len(s)
            axes = list(range(rank - fft_ndims, rank))
    else:
        _check_fft_axes(x, axes)
        axes = _normalize_axes(x, axes)
        axes_argsoft = np.argsort(axes).tolist()
        axes = [axes[i] for i in axes_argsoft]
        if s is not None:
            if len(s) != len(axes):
                raise ValueError(
                    f"Length of s ({len(s)}) and length of axes ({len(axes)}) does not match."
                )
            s = [s[i] for i in axes_argsoft]

    if s is not None:
        x = _resize_fft_input(x, s, axes)

    if in_dynamic_or_pir_mode():
        out = _C_ops.fft_c2c(x, axes, norm, forward)
    else:
        op_type = 'fft_c2c'
        check_variable_and_dtype(x, 'x', ['complex64', 'complex128'], op_type)
        inputs = {
            'X': [x],
        }
        attrs = {'axes': axes, 'normalization': norm, 'forward': forward}
        helper = LayerHelper(op_type, **locals())
        dtype = helper.input_dtype(input_param_name='x')
        out = helper.create_variable_for_type_inference(dtype)
        outputs = {"Out": [out]}
        helper.append_op(
            type=op_type, inputs=inputs, outputs=outputs, attrs=attrs
        )
    return out


def fftn_r2c(x, s, axes, norm, forward, onesided, name):
    if is_integer(x):
        x = paddle.cast(x, paddle.get_default_dtype())
    _check_normalization(norm)
    if s is not None:
        _check_fft_shape(x, s)

    rank = x.ndim
    if axes is None:
        if s is None:
            axes = list(range(rank))
        else:
            fft_ndims = len(s)
            axes = list(range(rank - fft_ndims, rank))
    else:
        _check_fft_axes(x, axes)
        axes = _normalize_axes(x, axes)
        axes_argsoft = np.argsort(axes[:-1]).tolist()
        axes = [axes[i] for i in axes_argsoft] + [axes[-1]]
        if s is not None:
            if len(s) != len(axes):
                raise ValueError(
                    f"Length of s ({len(s)}) and length of axes ({len(axes)}) does not match."
                )
            s = [s[i] for i in axes_argsoft] + [s[-1]]

    if s is not None:
        x = _resize_fft_input(x, s, axes)

    if in_dynamic_or_pir_mode():
        out = _C_ops.fft_r2c(x, axes, norm, forward, onesided)
    else:
        op_type = 'fft_r2c'
        check_variable_and_dtype(
            x, 'x', ['float16', 'float32', 'float64'], op_type
        )
        inputs = {
            'X': [x],
        }
        attrs = {
            'axes': axes,
            'normalization': norm,
            'forward': forward,
            'onesided': onesided,
        }
        helper = LayerHelper(op_type, **locals())
        dtype = helper.input_dtype(input_param_name='x')
        out = helper.create_variable_for_type_inference(
            _real_to_complex_dtype(dtype)
        )
        outputs = {"Out": [out]}
        helper.append_op(
            type=op_type, inputs=inputs, outputs=outputs, attrs=attrs
        )

    return out


def fftn_c2r(x, s, axes, norm, forward, name):
    if is_integer(x):
        x = paddle.cast(x, _real_to_complex_dtype(paddle.get_default_dtype()))
    elif is_floating_point(x):
        x = paddle.cast(x, _real_to_complex_dtype(x.dtype))
    _check_normalization(norm)
    if s is not None:
        _check_fft_shape(x, s)

    rank = x.ndim
    if axes is None:
        if s is None:
            axes = list(range(rank))
        else:
            fft_ndims = len(s)
            axes = list(range(rank - fft_ndims, rank))
    else:
        _check_fft_axes(x, axes)
        axes = _normalize_axes(x, axes)
        axes_argsoft = np.argsort(axes[:-1]).tolist()
        axes = [axes[i] for i in axes_argsoft] + [axes[-1]]
        if s is not None:
            if len(s) != len(axes):
                raise ValueError(
                    f"Length of s ({len(s)}) and length of axes ({len(axes)}) does not match."
                )
            s = [s[i] for i in axes_argsoft] + [s[-1]]

    if s is not None:
        fft_input_shape = list(s)
        fft_input_shape[-1] = fft_input_shape[-1] // 2 + 1
        x = _resize_fft_input(x, fft_input_shape, axes)

    if in_dynamic_or_pir_mode():
        if s is not None:
            out = _C_ops.fft_c2r(x, axes, norm, forward, s[-1])
        else:
            out = _C_ops.fft_c2r(x, axes, norm, forward, 0)
    else:
        op_type = 'fft_c2r'
        check_variable_and_dtype(x, 'x', ['complex64', 'complex128'], op_type)
        inputs = {
            'X': [x],
        }
        attrs = {'axes': axes, 'normalization': norm, 'forward': forward}
        if s:
            attrs["last_dim_size"] = s[-1]
        helper = LayerHelper(op_type, **locals())
        dtype = helper.input_dtype(input_param_name='x')
        out = helper.create_variable_for_type_inference(
            _complex_to_real_dtype(dtype)
        )
        outputs = {"Out": [out]}
        helper.append_op(
            type=op_type, inputs=inputs, outputs=outputs, attrs=attrs
        )
    return out
