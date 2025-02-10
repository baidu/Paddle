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

import numpy as np
from typing_extensions import TypeAlias, overload

import paddle
from paddle import _C_ops
from paddle.base.libpaddle import DataType
from paddle.common_ops_import import VarDesc
from paddle.utils.inplace_utils import inplace_apis_in_dygraph_only

from ..base.data_feeder import (
    check_dtype,
    check_type,
    check_variable_and_dtype,
    convert_dtype,
)
from ..common_ops_import import Variable
from ..framework import (
    LayerHelper,
    in_dynamic_mode,
    in_dynamic_or_pir_mode,
    in_pir_mode,
)
from .creation import full
from .manipulation import cast
from .math import _get_reduce_axis

if TYPE_CHECKING:
    from collections.abc import Sequence

    from paddle import Tensor

    _POrder: TypeAlias = Literal['fro', 'nuc']

__all__ = []


# Consistent with kDefaultDim from C++ Backend
K_DEFAULT_DIM = 9


def transpose(
    x: Tensor, perm: Sequence[int], name: str | None = None
) -> Tensor:
    """
    Permute the data dimensions of `input` according to `perm`.

    The `i`-th dimension  of the returned tensor will correspond to the
    perm[i]-th dimension of `input`.

    Args:
        x (Tensor): The input Tensor. It is a N-D Tensor of data types bool, float16, bfloat16, float32, float64, int8, int16, int32, int64, uint8, uint16, complex64, complex128.
        perm (list|tuple): Permute the input according to the data of perm.
        name (str|None, optional): The name of this layer. For more information, please refer to :ref:`api_guide_Name`. Default is None.

    Returns:
        Tensor: A transposed n-D Tensor, with data type being bool, float32, float64, int32, int64.

    Examples:

        .. code-block:: text

            # The following codes in this code block are pseudocode, designed to show the execution logic and results of the function.

            x = to_tensor([[[ 1  2  3  4] [ 5  6  7  8] [ 9 10 11 12]]
                           [[13 14 15 16] [17 18 19 20] [21 22 23 24]]])
            shape(x): return [2,3,4]

            # Example 1
            perm0 = [1,0,2]
            y_perm0 = transpose(x, perm0) # Permute x by perm0

            # dim:0 of y_perm0 is dim:1 of x
            # dim:1 of y_perm0 is dim:0 of x
            # dim:2 of y_perm0 is dim:2 of x
            # The above two lines can also be understood as exchanging the zeroth and first dimensions of x

            y_perm0.data = [[[ 1  2  3  4]  [13 14 15 16]]
                            [[ 5  6  7  8]  [17 18 19 20]]
                            [[ 9 10 11 12]  [21 22 23 24]]]
            shape(y_perm0): return [3,2,4]

            # Example 2
            perm1 = [2,1,0]
            y_perm1 = transpose(x, perm1) # Permute x by perm1

            # dim:0 of y_perm1 is dim:2 of x
            # dim:1 of y_perm1 is dim:1 of x
            # dim:2 of y_perm1 is dim:0 of x
            # The above two lines can also be understood as exchanging the zeroth and second dimensions of x

            y_perm1.data = [[[ 1 13]  [ 5 17]  [ 9 21]]
                            [[ 2 14]  [ 6 18]  [10 22]]
                            [[ 3 15]  [ 7 19]  [11 23]]
                            [[ 4 16]  [ 8 20]  [12 24]]]
            shape(y_perm1): return [4,3,2]

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> x = paddle.randn([2, 3, 4])
            >>> x_transposed = paddle.transpose(x, perm=[1, 0, 2])
            >>> print(x_transposed.shape)
            [3, 2, 4]

    """
    if in_dynamic_or_pir_mode():
        return _C_ops.transpose(x, perm)
    else:
        check_variable_and_dtype(
            x,
            'x',
            [
                'bool',
                'float16',
                'bfloat16',
                'float32',
                'float64',
                'int8',
                'uint8',
                'int16',
                'int32',
                'int64',
                'uint16',
                'complex64',
                'complex128',
                'float8_e5m2',
                'float8_e4m3fn',
            ],
            'transpose',
        )
        check_type(perm, 'perm', (list, tuple), 'transpose')
        if isinstance(perm, tuple):
            perm = list(perm)
        if len(perm) != len(x.shape):
            raise ValueError(
                "Input(perm) is the permutation of dimensions of Input(x), "
                "its length should be equal to dimensions of Input(x), "
                f"but received dimension of Input(x) is {len(x.shape)}, "
                f"the length of Input(perm) is {len(perm)}."
            )
        for idx, dim in enumerate(perm):
            if dim >= len(x.shape):
                raise ValueError(
                    "Each element in Input(perm) should be less than Input(x)'s dimension, "
                    f"but {idx}-th element in Input(perm) is {perm[idx]} which exceeds Input(x)'s "
                    f"dimension {len(x.shape)}."
                )

        helper = LayerHelper('transpose', **locals())
        out = helper.create_variable_for_type_inference(x.dtype)
        x_shape = helper.create_variable_for_type_inference(x.dtype)
        helper.append_op(
            type='transpose2',
            inputs={'X': [x]},
            outputs={'Out': [out], 'XShape': [x_shape]},
            attrs={'axis': perm},
        )
        return out


@inplace_apis_in_dygraph_only
def transpose_(x, perm, name=None):
    r"""
    Inplace version of ``transpose`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_paddle_transpose`.
    """
    if in_dynamic_mode():
        return _C_ops.transpose_(x, perm)


def matrix_transpose(
    x: paddle.Tensor,
    name: str | None = None,
) -> paddle.Tensor:
    """
    Transpose the last two dimensions of the input tensor `x`.

    Note:
        If `n` is the number of dimensions of `x`, `paddle.matrix_transpose(x)` is equivalent to `x.transpose([0, 1, ..., n-2, n-1])`.

    Args:
        x (Tensor): The input tensor to be transposed. `x` must be an N-dimensional tensor (N >= 2) of any data type supported by Paddle.
        name (str|None, optional): The name of this layer. For more information, please refer to :ref:`api_guide_Name`. Default is None.

    Returns:
        Tensor: A new tensor with the same shape as `x`, except that the last two dimensions are transposed.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> x = paddle.ones(shape=[2, 3, 5])
            >>> x_transposed = paddle.matrix_transpose(x)
            >>> print(x_transposed.shape)
            [2, 5, 3]
    """
    return x.mT


def matmul(
    x: Tensor,
    y: Tensor,
    transpose_x: bool = False,
    transpose_y: bool = False,
    name: str | None = None,
) -> Tensor:
    """
    Applies matrix multiplication to two tensors. `matmul` follows
    the complete broadcast rules,
    and its behavior is consistent with `np.matmul`.

    Currently, the input tensors' number of dimensions can be any, `matmul` can be used to
    achieve the `dot`, `matmul` and `batchmatmul`.

    The actual behavior depends on the shapes of :math:`x`, :math:`y` and the
    flag values of :attr:`transpose_x`, :attr:`transpose_y`. Specifically:

    - If a transpose flag is specified, the last two dimensions of the tensor
      are transposed. If the tensor is ndim-1 of shape, the transpose is invalid. If the tensor
      is ndim-1 of shape :math:`[D]`, then for :math:`x` it is treated as :math:`[1, D]`, whereas
      for :math:`y` it is the opposite: It is treated as :math:`[D, 1]`.

    The multiplication behavior depends on the dimensions of `x` and `y`. Specifically:

    - If both tensors are 1-dimensional, the dot product result is obtained.

    - If both tensors are 2-dimensional, the matrix-matrix product is obtained.

    - If the `x` is 1-dimensional and the `y` is 2-dimensional,
      a `1` is prepended to its dimension in order to conduct the matrix multiply.
      After the matrix multiply, the prepended dimension is removed.

    - If the `x` is 2-dimensional and `y` is 1-dimensional,
      the matrix-vector product is obtained.

    - If both arguments are at least 1-dimensional and at least one argument
      is N-dimensional (where N > 2), then a batched matrix multiply is obtained.
      If the first argument is 1-dimensional, a 1 is prepended to its dimension
      in order to conduct the batched matrix multiply and removed after.
      If the second argument is 1-dimensional, a 1 is appended to its
      dimension for the purpose of the batched matrix multiple and removed after.
      The non-matrix (exclude the last two dimensions) dimensions are
      broadcasted according the broadcast rule.
      For example, if input is a (j, 1, n, m) tensor and the other is a (k, m, p) tensor,
      out will be a (j, k, n, p) tensor.

    Args:
        x (Tensor): The input tensor which is a Tensor.
        y (Tensor): The input tensor which is a Tensor.
        transpose_x (bool, optional): Whether to transpose :math:`x` before multiplication. Default is False.
        transpose_y (bool, optional): Whether to transpose :math:`y` before multiplication. Default is False.
        name (str|None, optional): If set None, the layer will be named automatically. For more information, please refer to :ref:`api_guide_Name`. Default is None.

    Returns:
        Tensor: The output Tensor.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> # vector * vector
            >>> x = paddle.rand([10])
            >>> y = paddle.rand([10])
            >>> z = paddle.matmul(x, y)
            >>> print(z.shape)
            []

            >>> # matrix * vector
            >>> x = paddle.rand([10, 5])
            >>> y = paddle.rand([5])
            >>> z = paddle.matmul(x, y)
            >>> print(z.shape)
            [10]

            >>> # batched matrix * broadcasted vector
            >>> x = paddle.rand([10, 5, 2])
            >>> y = paddle.rand([2])
            >>> z = paddle.matmul(x, y)
            >>> print(z.shape)
            [10, 5]

            >>> # batched matrix * batched matrix
            >>> x = paddle.rand([10, 5, 2])
            >>> y = paddle.rand([10, 2, 5])
            >>> z = paddle.matmul(x, y)
            >>> print(z.shape)
            [10, 5, 5]

            >>> # batched matrix * broadcasted matrix
            >>> x = paddle.rand([10, 1, 5, 2])
            >>> y = paddle.rand([1, 3, 2, 5])
            >>> z = paddle.matmul(x, y)
            >>> print(z.shape)
            [10, 3, 5, 5]

    """
    if in_dynamic_or_pir_mode():
        return _C_ops.matmul(x, y, transpose_x, transpose_y)
    else:
        attrs = {
            'trans_x': transpose_x,
            'trans_y': transpose_y,
        }

        def __check_input(x, y):
            var_names = {'x': x, 'y': y}
            for name, val in var_names.items():
                check_variable_and_dtype(
                    val,
                    name,
                    [
                        'int8',
                        'uint16',
                        'float16',
                        'float32',
                        'float64',
                        'complex64',
                        'complex128',
                    ],
                    'matmul',
                )

        __check_input(x, y)

        helper = LayerHelper('matmul_v2', **locals())
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(
            type='matmul_v2',
            inputs={'X': x, 'Y': y},
            outputs={'Out': out},
            attrs=attrs,
        )
        return out


def fp8_fp8_half_gemm_fused(
    x,
    y,
    transpose_x=False,
    transpose_y=False,
    bias=None,
    scale=1.0,
    output_dtype="float16",
    act="identity",
    name=None,
):
    if in_dynamic_or_pir_mode():
        return _C_ops.fp8_fp8_half_gemm_fused(
            x, y, bias, transpose_x, transpose_y, scale, output_dtype, act
        )
    else:
        attrs = {
            'transpose_x': transpose_x,
            'transpose_y': transpose_y,
            'scale': scale,
            'output_dtype': output_dtype,
            'act': act,
        }
        if bias is None:

            def __check_input(x, y):
                var_names = {'x': x, 'y': y}
                for name, val in var_names.items():
                    check_variable_and_dtype(
                        val,
                        name,
                        [
                            'float8_e5m2',
                            'float8_e4m3fn',
                        ],
                        'fp8_fp8_half_gemm_fused',
                    )

            __check_input(x, y)

            helper = LayerHelper('fp8_fp8_half_gemm_fused', **locals())
            if output_dtype == 'float16':
                out = helper.create_variable_for_type_inference(dtype='float16')
            elif output_dtype == 'bfloat16':
                out = helper.create_variable_for_type_inference(
                    dtype='bfloat16'
                )
            else:
                raise ValueError("The output_dtype must be float16 or bfloat16")

            helper.append_op(
                type='fp8_fp8_half_gemm_fused',
                inputs={'x': x, 'y': y},
                outputs={'out': out},
                attrs=attrs,
            )
            return out
        else:

            def __check_input(x, y):
                var_names = {'x': x, 'y': y}
                for name, val in var_names.items():
                    check_variable_and_dtype(
                        val,
                        name,
                        [
                            'float8_e5m2',
                            'float8_e4m3fn',
                        ],
                        'fp8_fp8_half_gemm_fused',
                    )

            __check_input(x, y)
            if output_dtype == 'float16':
                check_variable_and_dtype(
                    bias, 'bias', ['float16'], 'fp8_fp8_half_gemm_fused'
                )
            elif output_dtype == 'bfloat16':
                check_variable_and_dtype(
                    bias, 'bias', ['bfloat16'], 'fp8_fp8_half_gemm_fused'
                )
            else:
                raise ValueError("The output_dtype must be float16 or bfloat16")

            helper = LayerHelper('fp8_fp8_half_gemm_fused', **locals())

            if output_dtype == 'float16':
                out = helper.create_variable_for_type_inference(dtype='float16')
            elif output_dtype == 'bfloat16':
                out = helper.create_variable_for_type_inference(
                    dtype='bfloat16'
                )
            else:
                raise ValueError("The output_dtype must be float16 or bfloat16")

            helper.append_op(
                type='fp8_fp8_half_gemm_fused',
                inputs={'x': x, 'y': y, 'bias': bias},
                outputs={'out': out},
                attrs=attrs,
            )
            return out


def vector_norm(
    x: Tensor,
    p: float = 2.0,
    axis: int | Sequence[int] | None = None,
    keepdim: bool = False,
    name: str | None = None,
) -> Tensor:
    """
    Calculate the p-order vector norm for certain  dimension of Tensor `input`.
    Returns the vector norm (the 1-norm, the Euclidean or 2-norm, and in general the p-norm)
    of a given tensor.

    Args:
        x (Tensor): Tensor, data type float32, float64.
        p (int|float, optional): None for porder=2.0. Default None.
        axis (int|list|tuple, optional): None for last dimension. Default None.
        keepdim (bool, optional): Whether keep the dimensions as the `input`, Default False.
        name (str|None, optional): The default value is None. Normally there is no need for
            user to set this property. For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: results of vector_norm operation on the specified axis of input tensor,
        it's data type is the same as input's Tensor.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import numpy as np
            >>> x = paddle.arange(24, dtype="float32").reshape([2, 3, 4]) - 12
            >>> print(x)
            Tensor(shape=[2, 3, 4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[[-12., -11., -10., -9. ],
              [-8. , -7. , -6. , -5. ],
              [-4. , -3. , -2. , -1. ]],
             [[ 0. ,  1. ,  2. ,  3. ],
              [ 4. ,  5. ,  6. ,  7. ],
              [ 8. ,  9. ,  10.,  11.]]])
            >>> out_vector_norm = paddle.linalg.vector_norm(x=x,p=2,axis=None,keepdim=False)
            >>> print(out_vector_norm)
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
            34.)
            >>> out_vector_norm = paddle.linalg.vector_norm(x=x,p=0,axis=[0,1],keepdim=False)
            >>> print(out_vector_norm)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [5., 6., 6., 6.])
            >>> out_vector_norm = paddle.linalg.vector_norm(x=x,p=float("inf"),axis=[1,2],keepdim=False)
            >>> print(out_vector_norm)
            Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [12., 11.])
            >>> out_vector_norm = paddle.linalg.vector_norm(x=x,p=1,axis=1,keepdim=False)
            >>> print(out_vector_norm)
            Tensor(shape=[2, 4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[24., 21., 18., 15.],
             [12., 15., 18., 21.]])
    """

    def zero_norm(
        input, porder=None, axis=axis, keepdim=False, asvector=False, name=None
    ):
        return paddle.count_nonzero(
            input, axis=axis, keepdim=keepdim, name=name
        ).astype(input.dtype)

    def inf_norm(
        input, porder=None, axis=axis, keepdim=False, asvector=False, name=None
    ):
        if in_dynamic_or_pir_mode():
            out = _C_ops.abs(input)
            if porder == np.float64('inf'):
                return _C_ops.max(out, axis, keepdim)
            else:
                return _C_ops.min(out, axis, keepdim)
        else:
            helper = LayerHelper('inf_norm', **locals())
            out = helper.create_variable_for_type_inference(
                dtype=helper.input_dtype()
            )
            helper.append_op(
                type='abs', inputs={'X': input}, outputs={'Out': out}
            )
            reduce_out = helper.create_variable_for_type_inference(
                dtype=helper.input_dtype()
            )
            reduce_all, axis = _get_reduce_axis(axis, x)
            reduce_type = (
                'reduce_max' if porder == np.float64('inf') else 'reduce_min'
            )
            helper.append_op(
                type=reduce_type,
                inputs={'X': out},
                outputs={'Out': reduce_out},
                attrs={
                    'dim': axis,
                    'keep_dim': keepdim,
                    'reduce_all': reduce_all,
                },
            )

            return reduce_out

    def vector_norm_axis_tuple(
        input, porder=2, axis=None, keepdim=False, asvector=False, name=None
    ):
        """
        NOTE:
            This function calculates the vector norm for dim >= 2.
        """
        if in_dynamic_or_pir_mode():
            abs_out = _C_ops.abs(input)
            pow_out = _C_ops.pow(abs_out, porder)
            sum_out = _C_ops.sum(pow_out, axis, None, keepdim)
            out = _C_ops.pow(sum_out, float(1.0 / porder))
            return out

        block = LayerHelper('norm', **locals())
        out = block.create_variable_for_type_inference(
            dtype=block.input_dtype()
        )
        abs_out = block.create_variable_for_type_inference(
            dtype=block.input_dtype()
        )
        block.append_op(
            type='abs', inputs={'X': input}, outputs={'Out': abs_out}
        )
        pow_out = block.create_variable_for_type_inference(
            dtype=block.input_dtype()
        )

        block.append_op(
            type='pow',
            inputs={'X': abs_out},
            outputs={'Out': pow_out},
            attrs={'factor': porder},
        )
        sum_out = block.create_variable_for_type_inference(
            dtype=block.input_dtype()
        )
        reduce_all, axis = _get_reduce_axis(axis, x)
        block.append_op(
            type='reduce_sum',
            inputs={'X': pow_out},
            outputs={'Out': sum_out},
            attrs={
                'dim': axis,
                'keep_dim': keepdim,
                'reduce_all': reduce_all,
            },
        )
        block.append_op(
            type='pow',
            inputs={'X': sum_out},
            outputs={'Out': out},
            attrs={'factor': float(1.0 / porder)},
        )
        return out

    def vector_norm_axis_int(
        input, porder=2, axis=None, keepdim=False, asvector=False, name=None
    ):
        """
        NOTE:
            This function calculates the vector norm for len(axis) == 1.
        """
        if in_dynamic_or_pir_mode():
            if axis is None:
                axis = -1
            return _C_ops.p_norm(input, porder, axis, 1e-12, keepdim, asvector)
        else:
            if porder is not None:
                check_type(porder, 'porder', (float, int), 'p_norm')
            if axis is not None:
                check_type(axis, 'axis', (int), 'p_norm')
            check_variable_and_dtype(
                input,
                'input',
                ['float16', 'uint16', 'float32', 'float64'],
                'p_norm',
            )

            attrs = {
                'axis': axis if axis is not None else -1,
                'porder': float(porder) if porder is not None else 2.0,
                'keepdim': keepdim,
                'asvector': asvector,
                'epsilon': 1e-12,
            }
            helper = LayerHelper('p_norm', **locals())
            out = helper.create_variable_for_type_inference(
                dtype=helper.input_dtype()
            )

            helper.append_op(
                type='p_norm',
                inputs={'X': input},
                outputs={'Out': out},
                attrs=attrs,
            )
            return out

    if not isinstance(p, (int, float)):
        raise ValueError(f"only valid p type is int and float, found {type(p)}")

    asvector = False
    if axis is None:
        axis = -1
        asvector = True

    if isinstance(axis, tuple):
        axis = list(axis)
    if isinstance(axis, list) and len(axis) == 1:
        axis = axis[0]

    if paddle.is_complex(x):
        abs_x = paddle.abs(x)
    else:
        abs_x = x

    # when len(axis) == 1, use the original op to calculate
    if isinstance(axis, int):
        return vector_norm_axis_int(
            abs_x,
            axis=axis,
            porder=p,
            keepdim=keepdim,
            asvector=asvector,
            name=name,
        )

    # when len(axis) >= 1, calculate by combining other Python apis
    elif isinstance(axis, list):
        if p == np.inf or p == -np.inf:
            return inf_norm(
                abs_x, porder=p, axis=axis, keepdim=keepdim, name=name
            )
        elif p == 0:
            return zero_norm(
                abs_x, porder=p, axis=axis, keepdim=keepdim, name=name
            )
        else:
            return vector_norm_axis_tuple(
                abs_x, porder=p, axis=axis, keepdim=keepdim, name=name
            )


def matrix_norm(
    x: Tensor,
    p: float | _POrder = 'fro',
    axis: int | list[int] | tuple[int, int] = [-2, -1],
    keepdim: bool = False,
    name: str | None = None,
) -> Tensor:
    """
    Calculate the p-order matrix norm for certain  dimension of Tensor `input`.

    Args:
        x (Tensor): Tensor, data type float32, float64.
        p (int|float|string, optional): Default 'fro'.
        axis (int|list|tuple, optional): The axis is a list(int)/tuple(int) with two elements. Default last two dimensions.
        keepdim (bool, optional): Whether keep the dimensions as the `input`, Default False.
        name (str|None, optional): The default value is None. Normally there is no need for
            user to set this property. For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: results of matrix_norm operation on the specified axis of input tensor,
        it's data type is the same as input's Tensor.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> x = paddle.arange(24, dtype="float32").reshape([2, 3, 4]) - 12
            >>> print(x)
            Tensor(shape=[2, 3, 4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[[-12., -11., -10., -9. ],
              [-8. , -7. , -6. , -5. ],
              [-4. , -3. , -2. , -1. ]],
             [[ 0. ,  1. ,  2. ,  3. ],
              [ 4. ,  5. ,  6. ,  7. ],
              [ 8. ,  9. ,  10.,  11.]]])

            >>> out_matrix_norm = paddle.linalg.matrix_norm(x=x,p=2,axis=[0,1],keepdim=False)
            >>> print(out_matrix_norm)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [15.75857544, 14.97978878, 14.69693947, 14.97978973])

            >>> out_matrix_norm = paddle.linalg.matrix_norm(x=x,p='fro',axis=[0,1],keepdim=False)
            >>> print(out_matrix_norm)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [17.43559647, 16.91153526, 16.73320007, 16.91153526])

            >>> out_matrix_norm = paddle.linalg.matrix_norm(x=x,p=float('inf'),axis=[1,2],keepdim=False)
            >>> print(out_matrix_norm)
            Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [42., 38.])

            >>> out_matrix_norm = paddle.linalg.matrix_norm(x=x,p=-1,axis=[0,1],keepdim=False)
            >>> print(out_matrix_norm)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [12., 12., 12., 12.])

            >>> out_matrix_norm = paddle.linalg.matrix_norm(x=x,p='nuc',axis=[0,1],keepdim=False)
            >>> print(out_matrix_norm)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [23.21962357, 22.82873154, 22.69693947, 22.82873154])

    """

    def _backshift_permutation(dim0, dim1, dimn):
        """
        Auxiliary function for matrix_norm
        Computes the permutation that moves the two given dimensions to the back
        """
        ret = [i for i in range(dimn) if i != dim0 and i != dim1]
        ret.extend((dim0, dim1))
        return ret

    def _inverse_permutation(perm):
        """
        Given a permutation, returns its inverse. It's equivalent to argsort on an array
        """
        return [i for i, j in sorted(enumerate(perm), key=lambda ij: ij[1])]

    def frobenius_norm(
        input: Tensor,
        dim: list[int] | None = None,
        keepdim: bool = False,
        name: str | None = None,
    ) -> Tensor:
        """
        The frobenius norm OP is to calculate the frobenius norm of certain two dimensions of Tensor `input`.
        Args:
          input (Variable): Tensor, data type float32, float64.
          dim (list, optional): None for last two dimensions. Default None.
          keepdim (bool, optional): Whether keep the dimensions as the `input`, Default False.
          name (str, optional): The default value is None. Normally there is no need for
              user to set this property. For more information, please refer to :ref:`api_guide_Name`.
        """
        if dim is not None and not (isinstance(dim, list) and len(dim) == 2):
            raise ValueError(
                "The dim of frobenius norm op should be None or two elements list!"
            )

        if in_dynamic_or_pir_mode():
            if dim is None:
                return _C_ops.frobenius_norm(input, [], keepdim, True)
            return _C_ops.frobenius_norm(input, dim, keepdim, False)
        else:
            attrs = {'dim': dim, 'keep_dim': keepdim, 'reduce_all': False}
            if dim is None:
                attrs['reduce_all'] = True
            check_variable_and_dtype(
                input, 'input', ['float32', 'float64'], 'frobenius_norm'
            )

            helper = LayerHelper('frobenius_norm', **locals())
            out = helper.create_variable_for_type_inference(
                dtype=helper.input_dtype()
            )

            helper.append_op(
                type='frobenius_norm',
                inputs={'X': input},
                outputs={'Out': out},
                attrs=attrs,
            )
            return out

    def nuclear_norm(
        input: Tensor,
        axis: int | list[int] | tuple[int, int] = axis,
        keepdim: bool = False,
        name: str | None = None,
    ) -> Tensor:
        """
        The nuclear norm OP is to calculate the nuclear norm of certain two dimensions of Tensor `input`.
        Args:
          input (Variable): Tensor, data type float32, float64.
          axis (list): Two dimensions.
          keepdim (bool, optional): Whether keep the dimensions as the `input`, Default False.
          name (str|None, optional): The default value is None. Normally there is no need for
              user to set this property. For more information, please refer to :ref:`api_guide_Name`.
        """

        perm = _backshift_permutation(axis[0], axis[1], len(input.shape))
        inv_perm = _inverse_permutation(perm)

        if in_dynamic_or_pir_mode():
            transposed = _C_ops.transpose(input, perm)
            u, s, vh = _C_ops.svd(transposed, False)
            result = _C_ops.sum(s, -1, None, keepdim)
            if keepdim:
                result = _C_ops.transpose(
                    _C_ops.unsqueeze(result, -1), inv_perm
                )
            return result

        attrs = {'axis': axis, 'keepdim': keepdim}

        check_variable_and_dtype(
            input, 'input', ['float32', 'float64'], 'nuclear_norm'
        )

        block = LayerHelper('nuclear_norm', **locals())
        out = block.create_variable_for_type_inference(
            dtype=block.input_dtype()
        )

        transpose_out = block.create_variable_for_type_inference(
            dtype=block.input_dtype()
        )
        input_shape = block.create_variable_for_type_inference(
            dtype=block.input_dtype()
        )

        block.append_op(
            type='transpose2',
            inputs={'X': [input]},
            outputs={'Out': [transpose_out], 'XShape': [input_shape]},
            attrs={'axis': perm},
        )

        u = block.create_variable_for_type_inference(dtype=block.input_dtype())
        s = block.create_variable_for_type_inference(dtype=block.input_dtype())
        vt = block.create_variable_for_type_inference(dtype=block.input_dtype())
        block.append_op(
            type='svd',
            inputs={'X': [transpose_out]},
            outputs={'U': u, 'VH': vt, 'S': s},
            attrs={'full_matrices': False},
        )

        reduce_all, sum_axis = _get_reduce_axis(-1, s)
        block.append_op(
            type='reduce_sum',
            inputs={'X': s},
            outputs={'Out': out},
            attrs={
                'dim': sum_axis,
                'keep_dim': keepdim,
                'reduce_all': reduce_all,
            },
        )

        if keepdim:
            unsqueeze_out = block.create_variable_for_type_inference(
                dtype=block.input_dtype()
            )

            block.append_op(
                type='unsqueeze2',
                inputs={'X': [out]},
                outputs={'Out': [unsqueeze_out], 'XShape': [input_shape]},
                attrs={'axes': [-1]},
            )

            block.append_op(
                type='transpose2',
                inputs={'X': [unsqueeze_out]},
                outputs={'Out': [out], 'XShape': [input_shape]},
                attrs={'axis': inv_perm},
            )

        return out

    def p_matrix_norm(
        input: Tensor,
        porder: float | _POrder = 1.0,
        axis: int | list[int] | tuple[int, int] = axis,
        keepdim: bool = False,
        name: str | None = None,
    ) -> Tensor:
        """
        Calculate the p-order matrix norm for certain  dimension of Tensor `input`.
        Args:
          input (Variable): Tensor, data type float32, float64.
          porder (int|float,str): p in ['fro', 'nuc', ±1, ±2, ±inf] Default 1.
          axis (list): Two dimensions.
          keepdim (bool, optional): Whether keep the dimensions as the `input`, Default False.
          name (str, optional): The default value is None. Normally there is no need for
              user to set this property. For more information, please refer to :ref:`api_guide_Name`.
        """

        perm = _backshift_permutation(axis[0], axis[1], len(input.shape))
        inv_perm = _inverse_permutation(perm)

        if in_dynamic_or_pir_mode():
            abs_ord = abs(porder)

            max_min = _C_ops.max if porder > 0.0 else _C_ops.min

            if abs_ord == 2.0:
                transpose_out = _C_ops.transpose(input, perm)
                u, s, vh = _C_ops.svd(transpose_out, False)
                result = max_min(s, -1, keepdim)
                if keepdim:
                    result = _C_ops.transpose(
                        _C_ops.unsqueeze(result, -1), inv_perm
                    )
                return result
            else:  # 1,-1,inf,-inf
                dim0, dim1 = axis
                if abs_ord == np.float64("inf"):
                    dim0, dim1 = dim1, dim0
                if not keepdim and (dim0 < dim1):
                    dim1 -= 1
                return max_min(
                    vector_norm(input, 1.0, axis=dim0, keepdim=keepdim),
                    dim1,
                    keepdim,
                )

        check_variable_and_dtype(
            input,
            'input',
            ['float16', 'uint16', 'float32', 'float64'],
            'p_matrix_norm',
        )

        block = LayerHelper('p_matrix_norm', **locals())
        out = block.create_variable_for_type_inference(
            dtype=block.input_dtype()
        )

        abs_ord = abs(porder)

        if abs_ord == 2.0:
            transpose_out = block.create_variable_for_type_inference(
                dtype=block.input_dtype()
            )
            input_shape = block.create_variable_for_type_inference(
                dtype=block.input_dtype()
            )

            block.append_op(
                type='transpose2',
                inputs={'X': [input]},
                outputs={'Out': [transpose_out], 'XShape': [input_shape]},
                attrs={'axis': perm},
            )

            u = block.create_variable_for_type_inference(
                dtype=block.input_dtype()
            )
            s = block.create_variable_for_type_inference(
                dtype=block.input_dtype()
            )
            vt = block.create_variable_for_type_inference(
                dtype=block.input_dtype()
            )
            block.append_op(
                type='svd',
                inputs={'X': [transpose_out]},
                outputs={'U': u, 'VH': vt, 'S': s},
                attrs={'full_matrices': False},
            )

            reduce_type = 'reduce_max' if porder > 0 else 'reduce_min'
            reduce_out = block.create_variable_for_type_inference(
                dtype=block.input_dtype()
            )
            reduce_all, max_min_axis = _get_reduce_axis(-1, s)
            block.append_op(
                type=reduce_type,
                inputs={'X': s},
                outputs={'Out': reduce_out},
                attrs={
                    'dim': max_min_axis,
                    'keep_dim': keepdim,
                    'reduce_all': reduce_all,
                },
            )

            if keepdim:
                unsqueeze_out = block.create_variable_for_type_inference(
                    dtype=block.input_dtype()
                )

                block.append_op(
                    type='unsqueeze2',
                    inputs={'X': [reduce_out]},
                    outputs={'Out': [unsqueeze_out], 'XShape': [input_shape]},
                    attrs={'axes': [-1]},
                )

                block.append_op(
                    type='transpose2',
                    inputs={'X': [unsqueeze_out]},
                    outputs={'Out': [out], 'XShape': [input_shape]},
                    attrs={'axis': inv_perm},
                )
                return out

            return reduce_out

        else:
            dim0, dim1 = axis
            if abs_ord == np.float64("inf"):
                dim0, dim1 = dim1, dim0
            if not keepdim and (dim0 < dim1):
                dim1 -= 1

            vector_out = block.create_variable_for_type_inference(
                dtype=block.input_dtype()
            )

            attrs = {
                'axis': dim0,
                'porder': 1,
                'keepdim': keepdim,
                'asvector': False,
                'epsilon': 1e-12,
            }

            block.append_op(
                type='p_norm',
                inputs={'X': input},
                outputs={'Out': vector_out},
                attrs=attrs,
            )

            reduce_type = 'reduce_max' if porder > 0 else 'reduce_min'
            reduce_out = block.create_variable_for_type_inference(
                dtype=block.input_dtype()
            )
            reduce_all, max_min_axis = _get_reduce_axis(dim1, vector_out)
            block.append_op(
                type=reduce_type,
                inputs={'X': vector_out},
                outputs={'Out': reduce_out},
                attrs={
                    'dim': max_min_axis,
                    'keep_dim': keepdim,
                    'reduce_all': reduce_all,
                },
            )
            return reduce_out

    if isinstance(axis, tuple):
        axis = list(axis)

    if isinstance(axis, list) and len(axis) == 2:
        if p == "fro":
            return frobenius_norm(x, dim=axis, keepdim=keepdim, name=name)
        elif p == "nuc":
            return nuclear_norm(x, axis=axis, keepdim=keepdim, name=name)
        elif (
            p == np.inf
            or p == -np.inf
            or p == 1
            or p == -1
            or p == 2
            or p == -2
        ):
            return p_matrix_norm(
                x, porder=p, axis=axis, keepdim=keepdim, name=name
            )
        else:
            raise ValueError(
                f"just support p value 'fro','nuc',1,-1,inf,-inf,2,-2 if axis is 2D, found {p}"
            )

    else:
        raise ValueError(
            f"except axis type int or list (length of list == 2), found {len(axis)}"
        )


def norm(
    x: Tensor,
    p: float | _POrder | None = None,
    axis: int | list[int] | tuple[int, int] | None = None,
    keepdim: bool = False,
    name: str | None = None,
) -> Tensor:
    """

    Returns the matrix norm (the Frobenius norm, the nuclear norm and p-norm) or vector norm (the 1-norm, the Euclidean
    or 2-norm, and in general the p-norm) of a given tensor.

    Whether the function calculates the vector norm or the matrix norm is determined as follows:

    - If axis is of type int, calculate the vector norm.

    - If axis is a two-dimensional array, calculate the matrix norm.

    - If axis is None, x is compressed into a one-dimensional vector and the vector norm is calculated.

    Paddle supports the following norms:

    +----------------+--------------------------------+--------------------------------+
    |     porder     |        norm for matrices       |        norm for vectors        |
    +================+================================+================================+
    |  None(default) |         frobenius norm         |            2_norm              |
    +----------------+--------------------------------+--------------------------------+
    |       fro      |         frobenius norm         |          not support           |
    +----------------+--------------------------------+--------------------------------+
    |       nuc      |          nuclear norm          |          not support           |
    +----------------+--------------------------------+--------------------------------+
    |       inf      |     max(sum(abs(x), dim=1))    |          max(abs(x))           |
    +----------------+--------------------------------+--------------------------------+
    |      -inf      |     min(sum(abs(x), dim=1))    |          min(abs(x))           |
    +----------------+--------------------------------+--------------------------------+
    |       0        |          not support           |          sum(x != 0)           |
    +----------------+--------------------------------+--------------------------------+
    |       1        |     max(sum(abs(x), dim=0))    |           as below             |
    +----------------+--------------------------------+--------------------------------+
    |      -1        |     min(sum(abs(x), dim=0))    |           as below             |
    +----------------+--------------------------------+--------------------------------+
    |       2        |The maximum singular value      |           as below             |
    |                |of a matrix consisting of axis. |                                |
    +----------------+--------------------------------+--------------------------------+
    |      -2        |The minimum singular value      |           as below             |
    |                |of a matrix consisting of axis. |                                |
    +----------------+--------------------------------+--------------------------------+
    |    other int   |           not support          | sum(abs(x)^{porder})^          |
    |     or float   |                                | {(1 / porder)}                 |
    +----------------+--------------------------------+--------------------------------+

    Args:
        x (Tensor): The input tensor could be N-D tensor, and the input data
            type could be float32 or float64.
        p (int|float|string|None, optional): Order of the norm. Supported values are `fro`, `nuc`, `0`, `±1`, `±2`,
            `±inf` and any real number yielding the corresponding p-norm.
            Default value is None.
        axis (int|list|tuple, optional): The axis on which to apply norm operation. If axis is int
            or list(int)/tuple(int)  with only one element, the vector norm is computed over the axis.
            If `axis < 0`, the dimension to norm operation is rank(input) + axis.
            If axis is a list(int)/tuple(int) with two elements, the matrix norm is computed over the axis.
            Default value is `None`.
        keepdim (bool, optional): Whether to reserve the reduced dimension in the
            output Tensor. The result tensor will have fewer dimension
            than the :attr:`input` unless :attr:`keepdim` is true, default
            value is False.
        name (str|None, optional): The default value is None. Normally there is no need for
            user to set this property. For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: results of norm operation on the specified axis of input tensor,
        it's data type is the same as input's Tensor.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> x = paddle.arange(24, dtype="float32").reshape([2, 3, 4]) - 12
            >>> print(x)
            Tensor(shape=[2, 3, 4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[[-12., -11., -10., -9. ],
              [-8. , -7. , -6. , -5. ],
              [-4. , -3. , -2. , -1. ]],
             [[ 0. ,  1. ,  2. ,  3. ],
              [ 4. ,  5. ,  6. ,  7. ],
              [ 8. ,  9. ,  10.,  11.]]])

            >>> # compute frobenius norm along last two dimensions.
            >>> out_fro = paddle.linalg.norm(x, p='fro', axis=[0,1])
            >>> print(out_fro)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [17.43559647, 16.91153526, 16.73320007, 16.91153526])

            >>> # compute 2-order vector norm along last dimension.
            >>> out_pnorm = paddle.linalg.norm(x, p=2, axis=-1)
            >>> print(out_pnorm)
            Tensor(shape=[2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[21.11871147, 13.19090557, 5.47722578 ],
             [3.74165750 , 11.22497177, 19.13112640]])

            >>> # compute 2-order  norm along [0,1] dimension.
            >>> out_pnorm = paddle.linalg.norm(x, p=2, axis=[0,1])
            >>> print(out_pnorm)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [15.75857544, 14.97978878, 14.69693947, 14.97978973])

            >>> # compute inf-order  norm
            >>> out_pnorm = paddle.linalg.norm(x, p=float("inf"))
            >>> print(out_pnorm)
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
            12.)

            >>> out_pnorm = paddle.linalg.norm(x, p=float("inf"), axis=0)
            >>> print(out_pnorm)
            Tensor(shape=[3, 4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[12., 11., 10., 9. ],
             [8. , 7. , 6. , 7. ],
             [8. , 9. , 10., 11.]])

            >>> # compute -inf-order  norm
            >>> out_pnorm = paddle.linalg.norm(x, p=-float("inf"))
            >>> print(out_pnorm)
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
            0.)

            >>> out_pnorm = paddle.linalg.norm(x, p=-float("inf"), axis=0)
            >>> print(out_pnorm)
            Tensor(shape=[3, 4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[0., 1., 2., 3.],
             [4., 5., 6., 5.],
             [4., 3., 2., 1.]])
    """

    if isinstance(axis, tuple):
        axis = list(axis)
    elif isinstance(axis, list) and len(axis) == 1:
        axis = axis[0]

    # calculate vector norm, where axis is None, int or list with only one integer
    if axis is None or (isinstance(axis, int)):
        # 'fro' is used to adapt previous usage
        if p is None or p == 'fro':
            p = 2.0
        if isinstance(p, (int, float)):
            return vector_norm(
                x,
                p=p,
                axis=axis,
                keepdim=keepdim,
                name=name,
            )
        else:
            raise ValueError(
                f"only valid p type is int or float for vector_norm, found {type(p)} and{p}"
            )

    # calculate matrix norm, where axis is list with two integers
    elif isinstance(axis, list) and len(axis) == 2:
        if p is None:
            p = 'fro'
        return matrix_norm(x=x, p=p, axis=axis, keepdim=keepdim, name=name)

    else:
        raise ValueError(
            f"except axis type int or list (length of list <=2), found {axis}"
        )


def dist(x: Tensor, y: Tensor, p: float = 2, name: str | None = None) -> Tensor:
    r"""

    Returns the p-norm of (x - y). It is not a norm in a strict sense, only as a measure
    of distance. The shapes of x and y must be broadcastable. The definition is as follows, for
    details, please refer to the `Introduction to Tensor <../../guides/beginner/tensor_en.html#chapter5-broadcasting-of-tensor>`_:

    - Each input has at least one dimension.
    - Match the two input dimensions from back to front, the dimension sizes must either be equal, one of them is 1, or one of them does not exist.

    Where, z = x - y, the shapes of x and y are broadcastable, then the shape of z can be
    obtained as follows:

    1. If the number of dimensions of x and y are not equal, prepend 1 to the dimensions of the
    tensor with fewer dimensions.

    For example, The shape of x is [8, 1, 6, 1], the shape of y is [7, 1, 5], prepend 1 to the
    dimension of y.

    x (4-D Tensor):  8 x 1 x 6 x 1

    y (4-D Tensor):  1 x 7 x 1 x 5

    2. Determine the size of each dimension of the output z: choose the maximum value from the
    two input dimensions.

    z (4-D Tensor):  8 x 7 x 6 x 5

    If the number of dimensions of the two inputs are the same, the size of the output can be
    directly determined in step 2. When p takes different values, the norm formula is as follows:

    When p = 0, defining $0^0=0$, the zero-norm of z is simply the number of non-zero elements of z.

    .. math::

        ||z||_{0}=\lim_{p \\rightarrow 0}\sum_{i=1}^{m}|z_i|^{p}

    When p = inf, the inf-norm of z is the maximum element of the absolute value of z.

    .. math::

        ||z||_\infty=\max_i |z_i|

    When p = -inf, the negative-inf-norm of z is the minimum element of the absolute value of z.

    .. math::

        ||z||_{-\infty}=\min_i |z_i|

    Otherwise, the p-norm of z follows the formula,

    .. math::

        ||z||_{p}=(\sum_{i=1}^{m}|z_i|^p)^{\\frac{1}{p}}

    Args:
        x (Tensor): 1-D to 6-D Tensor, its data type is bfloat16, float16, float32 or float64.
        y (Tensor): 1-D to 6-D Tensor, its data type is bfloat16, float16, float32 or float64.
        p (float, optional): The norm to be computed, its data type is float32 or float64. Default: 2.
        name (str|None, optional): The default value is `None`. Normally there is no need for
            user to set this property. For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: Tensor that is the p-norm of (x - y).

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([[3, 3],[3, 3]], dtype="float32")
            >>> y = paddle.to_tensor([[3, 3],[3, 1]], dtype="float32")
            >>> out = paddle.dist(x, y, 0)
            >>> print(out)
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
            1.)

            >>> out = paddle.dist(x, y, 2)
            >>> print(out)
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
            2.)

            >>> out = paddle.dist(x, y, float("inf"))
            >>> print(out)
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
            2.)

            >>> out = paddle.dist(x, y, float("-inf"))
            >>> print(out)
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
            0.)
    """
    if in_dynamic_or_pir_mode():
        return _C_ops.dist(x, y, p)

    check_variable_and_dtype(
        x, 'dtype', ['bfloat16', 'float16', 'float32', 'float64'], 'dist'
    )
    check_variable_and_dtype(
        y, 'dtype', ['bfloat16', 'float16', 'float32', 'float64'], 'dist'
    )
    check_type(p, 'p', (float, int), 'dist')
    helper = LayerHelper("dist", **locals())
    out = helper.create_variable_for_type_inference(x.dtype)

    inputs = {"X": [x], "Y": [y]}
    outputs = {'Out': [out]}
    attrs = {"p": float(p)}
    helper.append_op(
        type='dist', inputs=inputs, outputs={'Out': out}, attrs=attrs
    )
    return out


def cond(
    x: Tensor,
    p: float | _POrder | None = None,
    name: str | None = None,
) -> Tensor:
    """

    Computes the condition number of a matrix or batches of matrices with respect to a matrix norm ``p``.

    Args:
        x (Tensor): The input tensor could be tensor of shape ``(*, m, n)`` where ``*`` is zero or more batch dimensions
            for ``p`` in ``(2, -2)``, or of shape ``(*, n, n)`` where every matrix is invertible for any supported ``p``.
            And the input data type could be ``float32`` or ``float64``.
        p (float|string, optional): Order of the norm. Supported values are `fro`, `nuc`, `1`, `-1`, `2`, `-2`,
            `inf`, `-inf`. Default value is `None`, meaning that the order of the norm is `2`.
        name (str, optional): The default value is `None`. Normally there is no need for
            user to set this property. For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: computing results of condition number, its data type is the same as input Tensor ``x``.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.seed(2023)
            >>> x = paddle.to_tensor([[1., 0, -1], [0, 1, 0], [1, 0, 1]])

            >>> # compute conditional number when p is None
            >>> out = paddle.linalg.cond(x)
            >>> print(out)
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
            1.41421378)

            >>> # compute conditional number when order of the norm is 'fro'
            >>> out_fro = paddle.linalg.cond(x, p='fro')
            >>> print(out_fro)
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
            3.16227770)

            >>> # compute conditional number when order of the norm is 'nuc'
            >>> out_nuc = paddle.linalg.cond(x, p='nuc')
            >>> print(out_nuc)
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
            9.24264145)

            >>> # compute conditional number when order of the norm is 1
            >>> out_1 = paddle.linalg.cond(x, p=1)
            >>> print(out_1)
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
            2.)

            >>> # compute conditional number when order of the norm is -1
            >>> out_minus_1 = paddle.linalg.cond(x, p=-1)
            >>> print(out_minus_1)
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
            1.)

            >>> # compute conditional number when order of the norm is 2
            >>> out_2 = paddle.linalg.cond(x, p=2)
            >>> print(out_2)
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
            1.41421378)

            >>> # compute conditional number when order of the norm is -1
            >>> out_minus_2 = paddle.linalg.cond(x, p=-2)
            >>> print(out_minus_2)
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
            0.70710671)

            >>> # compute conditional number when order of the norm is inf
            >>> out_inf = paddle.linalg.cond(x, p=float("inf"))
            >>> print(out_inf)
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
            2.)

            >>> # compute conditional number when order of the norm is -inf
            >>> out_minus_inf = paddle.linalg.cond(x, p=-float("inf"))
            >>> print(out_minus_inf)
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
            1.)

            >>> a = paddle.randn([2, 4, 4])
            >>> print(a)
            Tensor(shape=[2, 4, 4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[[ 0.06132207,  1.11349595,  0.41906244, -0.24858207],
              [-1.85169315, -1.50370061,  1.73954511,  0.13331604],
              [ 1.66359663, -0.55764782, -0.59911072, -0.57773495],
              [-1.03176904, -0.33741450, -0.29695082, -1.50258386]],
             [[ 0.67233968, -1.07747352,  0.80170447, -0.06695852],
              [-1.85003340, -0.23008066,  0.65083790,  0.75387722],
              [ 0.61212337, -0.52664012,  0.19209868, -0.18707706],
              [-0.00711021,  0.35236868, -0.40404350,  1.28656745]]])

            >>> a_cond_fro = paddle.linalg.cond(a, p='fro')
            >>> print(a_cond_fro)
            Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [6.37173700 , 35.15114594])

            >>> b = paddle.randn([2, 3, 4])
            >>> print(b)
            Tensor(shape=[2, 3, 4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[[ 0.03306439,  0.70149767,  0.77064633, -0.55978841],
              [-0.84461296,  0.99335045, -1.23486686,  0.59551388],
              [-0.63035583, -0.98797107,  0.09410731,  0.47007179]],
             [[ 0.85850012, -0.98949534, -1.63086998,  1.07340240],
              [-0.05492965,  1.04750168, -2.33754158,  1.16518629],
              [ 0.66847134, -1.05326962, -0.05703246, -0.48190674]]])

            >>> b_cond_2 = paddle.linalg.cond(b, p=2)
            >>> print(b_cond_2)
            Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [2.86566353, 6.85834455])

    """

    def mat_norm(
        input: Tensor, porder: float = 1.0, axis: list[int] | None = None
    ) -> Tensor:
        """
        NOTE:
            Calculate the matrix norm of a square matrix or batches of square matrices,
            when porder is in (1, -1, inf, -inf)
        """
        if in_dynamic_or_pir_mode():
            abs_out = _C_ops.abs(input)
            sum_out = _C_ops.sum(abs_out, axis, None, False)

            if porder == 1 or porder == np.inf:
                return _C_ops.max(sum_out, [-1], False)
            if porder == -1 or porder == -np.inf:
                return _C_ops.min(sum_out, [-1], False)
        else:
            block = LayerHelper('norm', **locals())
            abs_out = block.create_variable_for_type_inference(
                dtype=block.input_dtype()
            )
            sum_out = block.create_variable_for_type_inference(
                dtype=block.input_dtype()
            )
            out = block.create_variable_for_type_inference(
                dtype=block.input_dtype()
            )
            block.append_op(
                type='abs', inputs={'X': input}, outputs={'Out': abs_out}
            )

            reduce_all, axis = _get_reduce_axis(axis, x)
            block.append_op(
                type='reduce_sum',
                inputs={'X': abs_out},
                outputs={'Out': sum_out},
                attrs={
                    'dim': axis,
                    'keep_dim': False,
                    'reduce_all': reduce_all,
                },
            )
            if porder == 1 or porder == np.inf:
                block.append_op(
                    type='reduce_max',
                    inputs={'X': sum_out},
                    outputs={'Out': out},
                    attrs={
                        'dim': [-1],
                        'keep_dim': False,
                        'reduce_all': reduce_all,
                    },
                )
            if porder == -1 or porder == -np.inf:
                block.append_op(
                    type='reduce_min',
                    inputs={'X': sum_out},
                    outputs={'Out': out},
                    attrs={
                        'dim': [-1],
                        'keep_dim': False,
                        'reduce_all': reduce_all,
                    },
                )
            return out

    def fro_norm(
        input: Tensor, porder: float = 2, axis: list[int] = [-1]
    ) -> Tensor:
        """
        NOTE:
            Calculate the frobenius norm of a square matrix or batches of square matrices.
        """
        if in_dynamic_or_pir_mode():
            pow_out = _C_ops.pow(input, porder)
            sum_out_1 = _C_ops.sum(pow_out, axis, None, False)
            sum_out_2 = _C_ops.sum(sum_out_1, axis, None, False)
            return _C_ops.pow(sum_out_2, float(1.0 / porder))
        else:
            block = LayerHelper('norm', **locals())
            pow_out = block.create_variable_for_type_inference(
                dtype=block.input_dtype()
            )
            sum_out_1 = block.create_variable_for_type_inference(
                dtype=block.input_dtype()
            )
            sum_out_2 = block.create_variable_for_type_inference(
                dtype=block.input_dtype()
            )
            out = block.create_variable_for_type_inference(
                dtype=block.input_dtype()
            )
            block.append_op(
                type='pow',
                inputs={'X': input},
                outputs={'Out': pow_out},
                attrs={'factor': porder},
            )

            reduce_all, axis = _get_reduce_axis(axis, x)
            block.append_op(
                type='reduce_sum',
                inputs={'X': pow_out},
                outputs={'Out': sum_out_1},
                attrs={
                    'dim': axis,
                    'keep_dim': False,
                    'reduce_all': reduce_all,
                },
            )
            block.append_op(
                type='reduce_sum',
                inputs={'X': sum_out_1},
                outputs={'Out': sum_out_2},
                attrs={
                    'dim': axis,
                    'keep_dim': False,
                    'reduce_all': reduce_all,
                },
            )
            block.append_op(
                type='pow',
                inputs={'X': sum_out_2},
                outputs={'Out': out},
                attrs={'factor': float(1.0 / porder)},
            )
            return out

    def svd_norm(
        input: Tensor, porder: float, axis: list[int] = [-1]
    ) -> Tensor:
        """
        NOTE:
            Calculate the matrix norm, which is related to singular values, of a matrix
            or batches of matrices, including nuclear norm, 2-norm and (-2)-norm.
        """
        u, s, vh = svd(input, full_matrices=False)

        if in_dynamic_or_pir_mode():
            if porder == "nuc":
                return _C_ops.sum(s, axis, None, False)
            max_out = _C_ops.max(s, axis, False)
            min_out = _C_ops.min(s, axis, False)
            if porder == 2:
                return _C_ops.divide(max_out, min_out)
            if porder == -2:
                return _C_ops.divide(min_out, max_out)
        else:
            reduce_all, axis = _get_reduce_axis(axis, x)
            block = LayerHelper('norm', **locals())
            out = block.create_variable_for_type_inference(
                dtype=block.input_dtype()
            )
            if porder == "nuc":
                block.append_op(
                    type='reduce_sum',
                    inputs={'X': s},
                    outputs={'Out': out},
                    attrs={
                        'dim': axis,
                        'keep_dim': False,
                        'reduce_all': reduce_all,
                    },
                )
                return out
            max_out = block.create_variable_for_type_inference(
                dtype=block.input_dtype()
            )
            min_out = block.create_variable_for_type_inference(
                dtype=block.input_dtype()
            )
            block.append_op(
                type='reduce_max',
                inputs={'X': s},
                outputs={'Out': max_out},
                attrs={
                    'dim': axis,
                    'keep_dim': False,
                    'reduce_all': reduce_all,
                },
            )
            block.append_op(
                type='reduce_min',
                inputs={'X': s},
                outputs={'Out': min_out},
                attrs={
                    'dim': axis,
                    'keep_dim': False,
                    'reduce_all': reduce_all,
                },
            )
            if porder == 2:
                block.append_op(
                    type='elementwise_div',
                    inputs={'X': max_out, 'Y': min_out},
                    outputs={'Out': out},
                    attrs={'axis': -1},
                )
                return out
            if porder == -2:
                block.append_op(
                    type='elementwise_div',
                    inputs={'X': min_out, 'Y': max_out},
                    outputs={'Out': out},
                    attrs={'axis': -1},
                )
                return out

    def empty_tensor(input, shape):
        if in_dynamic_or_pir_mode():
            if in_pir_mode():
                raise ValueError(
                    "only support x is nonempty tensor in static graph mode"
                )
            return input.reshape(shape)
        raise ValueError(
            "only support x is nonempty tensor in static graph mode"
        )

    x_shape = list(x.shape)
    if not len(x_shape) >= 2:
        raise ValueError(
            "input should be a matrix or batches of matrices, "
            + f"but the dimension of received input is {len(x_shape)}"
        )
    if p is None:
        p = 2
    x_size = 0 if (0 in x_shape) else 1
    if p in ("fro", "nuc", 1, -1, np.inf, -np.inf):
        if x_shape[len(x_shape) - 1] == x_shape[len(x_shape) - 2]:
            if x_size == 0:
                return empty_tensor(x, x_shape[:-2])
            x_inv = x.inverse()
            if p == "fro":
                return fro_norm(x) * fro_norm(x_inv)
            if p == "nuc":
                return svd_norm(x, p) * svd_norm(x_inv, p)
            if p in (1, -1):
                return mat_norm(x, porder=p, axis=[-2]) * mat_norm(
                    x_inv, porder=p, axis=[-2]
                )
            if p in (np.inf, -np.inf):
                return mat_norm(x, porder=p, axis=[-1]) * mat_norm(
                    x_inv, porder=p, axis=[-1]
                )
        else:
            raise ValueError(
                f"only support p is {p} when input is a "
                + "square matrix or batches of square matrices"
            )
    elif p in (2, -2):
        if x_size == 0:
            return empty_tensor(x, x_shape[:-2])
        return svd_norm(x, porder=p)
    else:
        raise ValueError(
            f"unsupported {p} for p, only supporting ('fro', 'nuc', "
            + "1, -1, 2, -2, inf, -inf) or none"
        )


def dot(x: Tensor, y: Tensor, name: str | None = None) -> Tensor:
    """
    This operator calculates inner product for vectors.

    Note:
       Support 1-d and 2-d Tensor. When it is 2d, the first dimension of this matrix
       is the batch dimension, which means that the vectors of multiple batches are dotted.

    Parameters:
        x(Tensor): 1-D or 2-D ``Tensor``. Its dtype should be ``float32``, ``float64``, ``int32``, ``int64``, ``complex64``, ``complex128``
        y(Tensor): 1-D or 2-D ``Tensor``. Its dtype should be ``float32``, ``float64``, ``int32``, ``int64``, ``complex64``, ``complex128``
        name(str|None, optional): Name of the output. Default is None. It's used to print debug info for developers. Details: :ref:`api_guide_Name`

    Returns:
        Tensor: the calculated result Tensor.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> # 1-D Tensor * 1-D Tensor
            >>> x = paddle.to_tensor([1, 2, 3])
            >>> y = paddle.to_tensor([4, 5, 6])
            >>> z = paddle.dot(x, y)
            >>> print(z)
            Tensor(shape=[], dtype=int64, place=Place(cpu), stop_gradient=True,
            32)

            >>> # 2-D Tensor * 2-D Tensor
            >>> x = paddle.to_tensor([[1, 2, 3], [2, 4, 6]])
            >>> y = paddle.to_tensor([[4, 5, 6], [4, 5, 6]])
            >>> z = paddle.dot(x, y)
            >>> print(z)
            Tensor(shape=[2], dtype=int64, place=Place(cpu), stop_gradient=True,
            [32, 64])

    """
    if in_dynamic_or_pir_mode():
        return _C_ops.dot(x, y)
    else:
        op_type = 'dot'

        assert x is not None, f'x cannot be None in {op_type}'
        assert y is not None, f'y cannot be None in {op_type}'

        check_variable_and_dtype(
            x,
            'x',
            [
                'float16',
                'uint16',
                'float32',
                'float64',
                'int32',
                'int64',
                'complex64',
                'complex128',
            ],
            op_type,
        )
        check_variable_and_dtype(
            y,
            'y',
            [
                'float16',
                'uint16',
                'float32',
                'float64',
                'int32',
                'int64',
                'complex64',
                'complex128',
            ],
            op_type,
        )

        helper = LayerHelper(op_type, **locals())
        if name is None:
            out = helper.create_variable_for_type_inference(dtype=x.dtype)
        else:
            out = helper.create_variable(
                name=name, dtype=x.dtype, persistable=False
            )
        helper.append_op(
            type="dot", inputs={'X': x, 'Y': y}, attrs={}, outputs={"Out": out}
        )
        return out


def vecdot(
    x: Tensor,
    y: Tensor,
    axis: int = -1,
    name: str | None = None,
) -> Tensor:
    """
    Computes the dot product of two tensors along a specified axis.

    This function multiplies two tensors element-wise and sums them along a specified axis to compute their dot product. It supports tensors of any dimensionality, including 0-D tensors, as long as the shapes of `x` and `y` are broadcastable along the specified axis.

    Args:
        x (Tensor): The first input tensor. It should be a tensor with dtype of float32, float64, int32, int64, complex64, or complex128.
        y (Tensor): The second input tensor. Its shape must be broadcastable with `x` along the specified `axis`, and it must have the same dtype as `x`.
        axis (int, optional): The axis along which to compute the dot product. Default is -1, which indicates the last axis.
        name (str|None, optional): Name of the output. Default is None. It's used to print debug info for developers. Details: :ref:`api_guide_Name`

    Returns:
        Tensor: A tensor containing the dot product of `x` and `y` along the specified axis.

    Examples:

        .. code-block:: python

            >>> import paddle
            >>> x = paddle.to_tensor([[1, 2, 3], [4, 5, 6]], dtype='float32')
            >>> y = paddle.to_tensor([[1, 2, 3], [4, 5, 6]], dtype='float32')
            >>> result = paddle.linalg.vecdot(x, y, axis=1)
            >>> print(result)
            Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [14.0, 77.0])
    """
    out = (x.conj() * y).sum(axis=axis)
    return out


def cov(
    x: Tensor,
    rowvar: bool = True,
    ddof: bool = True,
    fweights: Tensor | None = None,
    aweights: Tensor | None = None,
    name: str | None = None,
) -> Tensor:
    """
    Estimate the covariance matrix of the input variables, given data and weights.

    A covariance matrix is a square matrix, indicate the covariance of each pair variables in the input matrix.
    For example, for an N-dimensional samples X=[x1,x2,…xN]T, then the covariance matrix
    element Cij is the covariance of xi and xj. The element Cii is the variance of xi itself.

    Parameters:
        x (Tensor): A N-D(N<=2) Tensor containing multiple variables and observations. By default, each row of x represents a variable. Also see rowvar below.
        rowvar (bool, optional): If rowvar is True (default), then each row represents a variable, with observations in the columns. Default: True.
        ddof (bool, optional): If ddof=True will return the unbiased estimate, and ddof=False will return the simple average. Default: True.
        fweights (Tensor, optional): 1-D Tensor of integer frequency weights; The number of times each observation vector should be repeated. Default: None.
        aweights (Tensor, optional): 1-D Tensor of observation vector weights. How important of the observation vector, larger data means this element is more important. Default: None.
        name (str|None, optional): Name of the output. Default is None. It's used to print debug info for developers. Details: :ref:`api_guide_Name` .

    Returns:
        Tensor: The covariance matrix Tensor of the variables.

    Examples:

        .. code-block:: python

            >>> import paddle
            >>> paddle.seed(2023)

            >>> xt = paddle.rand((3, 4))
            >>> paddle.linalg.cov(xt)
            >>> print(xt)
            Tensor(shape=[3, 4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[0.86583614, 0.52014720, 0.25960937, 0.90525323],
             [0.42400089, 0.40641287, 0.97020894, 0.74437362],
             [0.51785129, 0.73292869, 0.97786582, 0.04315904]])
    """
    op_type = 'cov'
    if len(x.shape) > 2 or len(x.shape) < 1:
        raise ValueError(
            "Input(x) only support N-D (1<=N<=2) tensor in cov, but received "
            f"length of Input(input) is {len(x.shape)}."
        )
    check_variable_and_dtype(x, 'dtype', ['float32', 'float64'], 'cov')
    nx = x
    if len(x.shape) == 1:
        nx = x.reshape((1, -1))
    if not rowvar and nx.shape[0] != 1:
        nx = nx.t()
    w = None
    observation_num = nx.shape[1]
    if fweights is not None:
        w = fweights.astype(nx.dtype)
        if len(w.shape) > 1:
            raise ValueError(
                "Input(fweights) only support N-D (N<=1) tensor in cov, but received "
                f"shape of Input(input) is {len(fweights.shape)}."
            )
        if fweights.shape[0] != observation_num:
            raise ValueError(
                f"The number of Input(fweights) should equal to x's dim[1]: {observation_num}, but received "
                f"size of Input(fweights) is {fweights.shape[0]}."
            )
        if fweights.min() < 0:
            raise ValueError(
                "The value of Input(fweights) cannot be negative, but received "
                f"min of Input(fweights) is {fweights.min()}."
            )
        if not paddle.all(
            fweights
            == paddle.round(fweights.astype('float64')).astype(fweights.dtype)
        ):
            raise ValueError("Input(fweights) must be integer ")

    if aweights is not None:
        aw = aweights.astype(nx.dtype)
        if len(aw.shape) > 1:
            raise ValueError(
                "Input(aweights) only support N-D (N<=1) tensor in cov, but received "
                f"length of Input(input) is {len(aweights.shape)}."
            )
        check_variable_and_dtype(
            aweights, 'dtype', ['float32', 'float64'], 'cov'
        )
        if aweights.shape[0] != observation_num:
            raise ValueError(
                f"The number of Input(aweights) should equal to x's dim[1]: {observation_num}, but received "
                f"size of Input(aweights) is {aweights.shape[0]}."
            )
        if aweights.min() < 0:
            raise ValueError(
                "The value of Input(aweights) cannot be negative, but received "
                f"min of Input(aweights) is {aweights.min()}."
            )
        if w is not None:
            w = w * aw
        else:
            w = aw

    w_sum = paddle.to_tensor(observation_num, dtype=nx.dtype)
    if fweights is not None or aweights is not None:
        w_sum = w.sum()
        if w_sum.item() == 0:
            raise ValueError("The sum of weights is zero, can't be normalized.")

    if w is not None:
        nx_w = nx * w
        avg = (nx_w).sum(axis=1) / w_sum
    else:
        avg = nx.sum(axis=1) / w_sum
        nx_w = nx

    if w is not None and aweights is not None and ddof:
        norm_factor = w_sum - (w * aweights.astype(w.dtype)).sum() / w_sum
    else:
        norm_factor = w_sum - ddof
    norm_factor = paddle.clip(norm_factor, min=0)
    nx = nx - avg.unsqueeze(1)
    xxt = paddle.mm(nx, nx_w.t().conj())
    cov = paddle.divide(xxt, norm_factor).squeeze()
    return cov


def t(input: Tensor, name: str | None = None) -> Tensor:
    """
    Transpose <=2-D tensor.
    0-D and 1-D tensors are returned as it is and 2-D tensor is equal to
    the paddle.transpose function which perm dimensions set 0 and 1.

    Args:
        input (Tensor): The input Tensor. It is a N-D (N<=2) Tensor of data types float32, float64, int32, int64.
        name (str|None, optional): The default value is None.  Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name` .

    Returns:
        Tensor: A transposed n-D Tensor, with data type being float16, float32, float64, int32, int64.

    Examples:

        .. code-block:: python
            :name: code-example

            >>> import paddle

            >>> # Example 1 (0-D tensor)
            >>> x = paddle.to_tensor([0.79])
            >>> out = paddle.t(x)
            >>> print(out)
            Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True,
            [0.79000002])

            >>> # Example 2 (1-D tensor)
            >>> x = paddle.to_tensor([0.79, 0.84, 0.32])
            >>> out2 = paddle.t(x)
            >>> print(out2)
            Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [0.79000002, 0.83999997, 0.31999999])
            >>> print(paddle.t(x).shape)
            [3]

            >>> # Example 3 (2-D tensor)
            >>> x = paddle.to_tensor([[0.79, 0.84, 0.32],
            ...                       [0.64, 0.14, 0.57]])
            >>> print(x.shape)
            [2, 3]
            >>> out3 = paddle.t(x)
            >>> print(out3)
            Tensor(shape=[3, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[0.79000002, 0.63999999],
             [0.83999997, 0.14000000],
             [0.31999999, 0.56999999]])
            >>> print(paddle.t(x).shape)
            [3, 2]

    """
    if len(input.shape) > 2:
        raise ValueError(
            "Input(input) only support N-D (N<=2) tensor, but received "
            f"length of Input(input) is {len(input.shape)}. Perhaps you can use paddle."
            "tensor.transpose() instead."
        )
    if in_dynamic_or_pir_mode():
        if len(input.shape) <= 1:
            return input
        # 2-D tensor
        perm = [1, 0]
        out = _C_ops.transpose(input, perm)
        return out
    else:
        check_variable_and_dtype(
            input,
            'input',
            ['float16', 'float32', 'float64', 'int32', 'int64', 'uint16'],
            'transpose',
        )

        helper = LayerHelper('t', **locals())
        out = helper.create_variable_for_type_inference(input.dtype)
        input_shape = helper.create_variable_for_type_inference(input.dtype)
        if len(input.shape) <= 1:
            out = input
        else:
            helper.append_op(
                type='transpose2',
                inputs={'X': [input]},
                outputs={'Out': [out], 'XShape': [input_shape]},
                attrs={'axis': [1, 0]},
            )
        return out


@inplace_apis_in_dygraph_only
def t_(input, name=None):
    r"""
    Inplace version of ``t`` API, the output Tensor will be inplaced with input ``input``.
    Please refer to :ref:`api_paddle_t`.
    """
    if len(input.shape) > 2:
        raise ValueError(
            "Input(input) only support N-D (N<=2) tensor, but received "
            f"length of Input(input) is {len(input.shape)}. Perhaps you can use paddle."
            "tensor.transpose() instead."
        )
    if in_dynamic_mode():
        if len(input.shape) <= 1:
            return input
        # 2-D tensor
        perm = [1, 0]
        out = _C_ops.transpose_(input, perm)
        return out


def cross(
    x: Tensor,
    y: Tensor,
    axis: int = 9,
    name: str | None = None,
) -> Tensor:
    """
    Computes the cross product between two tensors along an axis.

    Inputs must have the same shape, and the length of their axes should be equal to 3.
    If `axis` is not given, it defaults to the first axis found with the length 3.

    Args:
        x (Tensor): The first input tensor, the data type is float16, float32, float64, int32, int64, complex64, complex128.
        y (Tensor): The second input tensor, the data type is float16, float32, float64, int32, int64, complex64, complex128.
        axis (int, optional): The axis along which to compute the cross product. It defaults to be 9 which indicates using the first axis found with the length 3.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor. A Tensor with same data type as `x`.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([[1.0, 1.0, 1.0],
            ...                         [2.0, 2.0, 2.0],
            ...                         [3.0, 3.0, 3.0]])
            >>> y = paddle.to_tensor([[1.0, 1.0, 1.0],
            ...                         [1.0, 1.0, 1.0],
            ...                         [1.0, 1.0, 1.0]])
            ...
            >>> z1 = paddle.cross(x, y)
            >>> print(z1)
            Tensor(shape=[3, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[-1., -1., -1.],
             [ 2.,  2.,  2.],
             [-1., -1., -1.]])

            >>> z2 = paddle.cross(x, y, axis=1)
            >>> print(z2)
            Tensor(shape=[3, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[0., 0., 0.],
             [0., 0., 0.],
             [0., 0., 0.]])
    """
    if in_dynamic_or_pir_mode():
        axis = K_DEFAULT_DIM if axis is None else axis
        return _C_ops.cross(x, y, axis)
    else:
        check_variable_and_dtype(
            x,
            'x',
            [
                'float16',
                'uint16',
                'float32',
                'float64',
                "int32",
                "int64",
                "complex64",
                "complex128",
            ],
            'cross',
        )
        check_variable_and_dtype(
            y,
            'y',
            [
                'float16',
                'uint16',
                'float32',
                'float64',
                "int32",
                "int64",
                "complex64",
                "complex128",
            ],
            'cross',
        )
        helper = LayerHelper("cross", **locals())
        out = helper.create_variable_for_type_inference(x.dtype)
        attrs = {}
        attrs['dim'] = axis

        helper.append_op(
            type='cross',
            inputs={'X': x, 'Y': y},
            outputs={'Out': out},
            attrs=attrs,
        )
        return out


def cholesky(x: Tensor, upper: bool = False, name: str | None = None) -> Tensor:
    r"""
    Computes the Cholesky decomposition of one symmetric positive-definite
    matrix or batches of symmetric positive-definite matrices.

    If `upper` is `True`, the decomposition has the form :math:`A = U^{T}U` ,
    and the returned matrix :math:`U` is upper-triangular. Otherwise, the
    decomposition has the form  :math:`A = LL^{T}` , and the returned matrix
    :math:`L` is lower-triangular.

    Args:
        x (Tensor): The input tensor. Its shape should be `[*, M, M]`,
            where * is zero or more batch dimensions, and matrices on the
            inner-most 2 dimensions all should be symmetric positive-definite.
            Its data type should be float32 or float64.
        upper (bool, optional): The flag indicating whether to return upper or lower
            triangular matrices. Default: False.
        name (str|None, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, A Tensor with same shape and data type as `x`. It represents
        triangular matrices generated by Cholesky decomposition.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.seed(2023)

            >>> a = paddle.rand([3, 3], dtype="float32")
            >>> a_t = paddle.transpose(a, [1, 0])
            >>> x = paddle.matmul(a, a_t) + 1e-03

            >>> out = paddle.linalg.cholesky(x, upper=False)
            >>> print(out)
            Tensor(shape=[3, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[1.04337072, 0.        , 0.        ],
             [1.06467664, 0.17859250, 0.        ],
             [1.30602181, 0.08326444, 0.22790681]])
    """
    if in_dynamic_or_pir_mode():
        return _C_ops.cholesky(x, upper)
    else:
        check_variable_and_dtype(x, 'dtype', ['float32', 'float64'], 'cholesky')
        check_type(upper, 'upper', bool, 'cholesky')
        helper = LayerHelper('cholesky', **locals())
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(
            type='cholesky',
            inputs={'X': [x]},
            outputs={'Out': out},
            attrs={'upper': upper},
        )
        return out


def matrix_rank(
    x: Tensor,
    tol: float | Tensor | None = None,
    hermitian: bool = False,
    atol: float | Tensor | None = None,
    rtol: float | Tensor | None = None,
    name: str | None = None,
) -> Tensor:
    r"""
    Computes the rank of a matrix.

    Notes:
        1. Support the use of attribute `tol` alone or the use of attributes `atol` and `rtol` together without `tol`.

        2. When `tol` is used alone, it will return the rank of a matrix is the number of singular values that are greater than the specified `tol`
        threshold when hermitian=False, or the number of eigenvalues in absolute value that are greater than the specified `tol` threshold
        when hermitian=True. It is compatible with numpy API.

        3. When `atol` and `rtol` are used, the tolerance value is computed as `max(atol, sigma_1 * rtol)`, where sigma_1 is largest
        singular value (or eigenvalues in absolute value).

        4. When `atol` and `rtol` are used: If `rtol` is not specified, then it is set to be `max(m,n) * eps`, where `x` has dimension(m, n) and
        `eps` is the epsilon value for the dtype of `x`; If `rtol` is not specified and `atol` is specified to be greater than 0, then it
        is set to be 0.

    Args:
        x (Tensor): The input tensor. Its shape should be `[..., m, n]`, where `...` is zero or more batch dimensions. If `x` is a batch
            of matrices then the output has the same batch dimensions. The data type of `x` should be float32 or float64.
        tol (float|Tensor, optional): The tolerance value. If `tol` is not specified, and `sigma` is the largest singular value
            (or eigenvalues in absolute value), and `eps` is the epsilon value for the dtype of `x`, then `tol` is computed with formula
            `tol=sigma * max(m,n) * eps`. Note that if `x` is a batch of matrices, `tol` is computed this way for every batch. Default: None.
        hermitian (bool, optional): Indicates whether `x` is Hermitian. Default: False. When hermitian=True, `x` is assumed to be Hermitian,
            enabling a more efficient method for finding eigenvalues, but `x` is not checked inside the function. Instead, We just use
            the lower triangular of the matrix to compute. Default: False.
        atol (float|Tensor, optional): The absolute tolerance value. When None it is considered to be 0. Default: None.
        rtol (float|Tensor, optional): The relative tolerance value. See above Notes for the value it takes when None. Default: None.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: Rank of tensor x.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> a = paddle.eye(10)
            >>> b = paddle.linalg.matrix_rank(a)
            >>> print(b)
            Tensor(shape=[], dtype=int32, place=Place(cpu), stop_gradient=True,
            10)

            >>> c = paddle.ones(shape=[3, 4, 5, 5])
            >>> d = paddle.linalg.matrix_rank(c, tol=0.01, hermitian=True)
            >>> print(d)
            Tensor(shape=[3, 4], dtype=int32, place=Place(cpu), stop_gradient=True,
            [[1, 1, 1, 1],
             [1, 1, 1, 1],
             [1, 1, 1, 1]])

    """
    use_atol_rtol = False
    if (atol is not None) or (rtol is not None):
        if tol is not None:
            raise ValueError(
                "Only support to use tol alone or use atol and rtol without tol."
            )
        use_atol_rtol = True

    if use_atol_rtol:
        if atol is None:
            atol = full([], 0.0, x.dtype)
        if isinstance(atol, (float, int)):
            atol = full([], atol, x.dtype)
        if atol.dtype != x.dtype:
            atol = cast(atol, x.dtype)

        if rtol is not None:
            if isinstance(rtol, (float, int)):
                rtol = full([], rtol, x.dtype)
            if rtol.dtype != x.dtype:
                rtol = cast(rtol, x.dtype)

            atol, rtol = paddle.broadcast_tensors([atol, rtol])

        if in_dynamic_or_pir_mode():
            return _C_ops.matrix_rank_atol_rtol(x, atol, rtol, hermitian)
        else:
            inputs = {}
            attrs = {}
            check_variable_and_dtype(
                x, 'x', ['float32', 'float64'], 'matrix_rank_atol_rtol'
            )
            inputs['x'] = x
            inputs['atol'] = atol
            inputs['rtol'] = rtol
            check_type(hermitian, 'hermitian', bool, 'matrix_rank_atol_rtol')
            attrs['hermitian'] = hermitian

            helper = LayerHelper('matrix_rank_atol_rtol', **locals())
            out = helper.create_variable_for_type_inference(dtype='int32')
            helper.append_op(
                type='matrix_rank_atol_rtol',
                inputs=inputs,
                outputs={'out': out},
                attrs=attrs,
            )
            return out
    else:
        if in_dynamic_or_pir_mode():
            if isinstance(tol, (Variable, paddle.pir.Value)):
                if tol.dtype != x.dtype:
                    tol_tensor = cast(tol, x.dtype)
                else:
                    tol_tensor = tol
                use_default_tol = False
                return _C_ops.matrix_rank_tol(
                    x, tol_tensor, use_default_tol, hermitian
                )

            if tol is None:
                tol_attr = 0.0
                use_default_tol = True
            else:
                tol_attr = float(tol)
                use_default_tol = False
            return _C_ops.matrix_rank(x, tol_attr, use_default_tol, hermitian)
        else:
            inputs = {}
            attrs = {}
            check_variable_and_dtype(
                x, 'x', ['float32', 'float64'], 'matrix_rank'
            )
            inputs['X'] = x
            if tol is None:
                attrs['use_default_tol'] = True
            elif isinstance(tol, Variable):
                attrs['use_default_tol'] = False
                if tol.dtype != x.dtype:
                    inputs['TolTensor'] = cast(tol, x.dtype)
                else:
                    inputs['TolTensor'] = tol
            else:
                check_type(tol, 'tol', float, 'matrix_rank')
                attrs['use_default_tol'] = False
                attrs['tol'] = tol
            check_type(hermitian, 'hermitian', bool, 'matrix_rank')
            attrs['hermitian'] = hermitian

            helper = LayerHelper('matrix_rank', **locals())
            out = helper.create_variable_for_type_inference(dtype='int32')
            helper.append_op(
                type='matrix_rank',
                inputs=inputs,
                outputs={'Out': out},
                attrs=attrs,
            )
            return out


def bmm(x: Tensor, y: Tensor, name: str | None = None) -> Tensor:
    """
    Applies batched matrix multiplication to two tensors.

    Both of the two input tensors must be three-dimensional and share the same batch size.

    If x is a (b, m, k) tensor, y is a (b, k, n) tensor, the output will be a (b, m, n) tensor.

    Args:
        x (Tensor): The input Tensor.
        y (Tensor): The input Tensor.
        name (str|None): A name for this layer(optional). If set None, the layer
            will be named automatically. Default: None.

    Returns:
        Tensor: The product Tensor.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> # In imperative mode:
            >>> # size x: (2, 2, 3) and y: (2, 3, 2)
            >>> x = paddle.to_tensor([[[1.0, 1.0, 1.0],
            ...                     [2.0, 2.0, 2.0]],
            ...                     [[3.0, 3.0, 3.0],
            ...                     [4.0, 4.0, 4.0]]])
            >>> y = paddle.to_tensor([[[1.0, 1.0],[2.0, 2.0],[3.0, 3.0]],
            ...                     [[4.0, 4.0],[5.0, 5.0],[6.0, 6.0]]])
            >>> out = paddle.bmm(x, y)
            >>> print(out)
            Tensor(shape=[2, 2, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[[6. , 6. ],
              [12., 12.]],
             [[45., 45.],
              [60., 60.]]])

    """
    if in_dynamic_or_pir_mode():
        return _C_ops.bmm(x, y)
    else:
        x_shape = x.shape
        y_shape = y.shape
        if not len(x_shape) == len(y_shape) == 3:
            raise ValueError(
                f"x and y should be 3-dimensional. But received x's dimension: {x_shape}, y's dimension: {y_shape}"
            )
        if x_shape[2] != -1 and y_shape[1] != -1 and x_shape[2] != y_shape[1]:
            raise ValueError(
                f"x's width must be equal with y's height. But received x's shape: {x_shape}, y's shape: {y_shape}"
            )
        if x_shape[0] != -1 and y_shape[0] != -1 and x_shape[0] != y_shape[0]:
            raise ValueError(
                f"x's batch (shape[0]) must be equal with y's batch (shape[0]). But received x's shape: {x_shape}, y's shape: {y_shape}"
            )
        helper = LayerHelper('bmm', **locals())
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(
            type='bmm', inputs={'X': x, 'Y': y}, outputs={'Out': out}
        )
        return out


def histogram(
    input: Tensor,
    bins: int = 100,
    min: float = 0.0,
    max: float = 0.0,
    weight: Tensor | None = None,
    density: bool = False,
    name: str | None = None,
) -> Tensor:
    """
    Computes the histogram of a tensor. The elements are sorted into equal width bins between min and max.
    If min and max are both zero, the minimum and maximum values of the data are used.

    Args:
        input (Tensor): A Tensor with shape :math:`[N_1, N_2,..., N_k]` . The data type of the input Tensor
            should be float32, float64, int32, int64.
        bins (int, optional): number of histogram bins. Default: 100.
        min (float, optional): lower end of the range (inclusive). Default: 0.0.
        max (float, optional): upper end of the range (inclusive). Default: 0.0.
        weight (Tensor, optional): If provided, it must have the same shape as input. Each value in input contributes its associated
            weight towards the bin count (instead of 1). Default: None.
        density (bool, optional): If False, the result will contain the count (or total weight) in each bin. If True, the result is the
            value of the probability density function over the bins, normalized such that the integral over the range of the bins is 1.
        name (str|None, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        Tensor, shape is (nbins,), the counts or density of the histogram.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> inputs = paddle.to_tensor([1, 2, 1])
            >>> result = paddle.histogram(inputs, bins=4, min=0, max=3)
            >>> print(result)
            Tensor(shape=[4], dtype=int64, place=Place(cpu), stop_gradient=True,
            [0, 2, 1, 0])
    """
    if isinstance(min, int):
        min = float(min)
    if isinstance(max, int):
        max = float(max)

    if in_dynamic_or_pir_mode():
        return _C_ops.histogram(input, weight, bins, min, max, density)
    else:
        helper = LayerHelper('histogram', **locals())
        check_variable_and_dtype(
            input, 'X', ['int32', 'int64', 'float32', 'float64'], 'histogram'
        )

        if weight or density:
            if weight:
                check_variable_and_dtype(
                    weight,
                    'Weight',
                    ['int32', 'int64', 'float32', 'float64'],
                    'histogram',
                )
            out = helper.create_variable_for_type_inference(
                dtype=VarDesc.VarType.FP32
            )
        else:
            out = helper.create_variable_for_type_inference(
                dtype=VarDesc.VarType.INT64
            )

        helper.append_op(
            type='histogram',
            inputs={'X': input, 'Weight': weight},
            outputs={'Out': out},
            attrs={
                'bins': bins,
                'min': min,
                'max': max,
                'density': density,
            },
        )
        return out


def histogram_bin_edges(
    input: Tensor,
    bins: int = 100,
    min: float = 0.0,
    max: float = 0.0,
    name: str | None = None,
) -> Tensor:
    """
    Computes only the edges of the bins used by the histogram function.
    If min and max are both zero, the minimum and maximum values of the data are used.

    Args:
        input (Tensor): The data type of the input Tensor should be float32, float64, int32, int64.
        bins (int, optional): number of histogram bins.
        min (float, optional): lower end of the range (inclusive). Default: 0.0.
        max (float, optional): upper end of the range (inclusive). Default: 0.0.
        name (str|None, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        Tensor, the values of the bin edges. The output data type will be float32.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> inputs = paddle.to_tensor([1, 2, 1])
            >>> result = paddle.histogram_bin_edges(inputs, bins=4, min=0, max=3)
            >>> print(result)
            Tensor(shape=[5], dtype=float32, place=Place(cpu), stop_gradient=True,
            [0.        , 0.75000000, 1.50000000, 2.25000000, 3.        ])
    """
    if isinstance(min, int):
        min = float(min)
    if isinstance(max, int):
        max = float(max)

    check_type(input, 'input', (Variable), 'histogram_bin_edges')
    check_dtype(
        input.dtype,
        'input',
        ['float32', 'float64', 'int32', 'int64'],
        'histogram_bin_edges',
    )
    check_type(bins, 'bins', int, 'histogram_bin_edges')
    if max == 0.0 and min == 0.0:
        min = paddle.min(input)
        max = paddle.max(input)
    else:
        if max < min:
            raise ValueError("max must be larger than min in range parameter")
    if (min - max) == 0.0:
        max = max + 0.5
        min = min - 0.5
    return paddle.linspace(min, max, bins + 1, name=name)


def bincount(
    x: Tensor,
    weights: Tensor | None = None,
    minlength: int = 0,
    name: str | None = None,
) -> Tensor:
    """
    Computes frequency of each value in the input tensor.

    Args:
        x (Tensor): A Tensor with non-negative integer. Should be 1-D tensor.
        weights (Tensor, optional): Weight for each value in the input tensor. Should have the same shape as input. Default is None.
        minlength (int, optional): Minimum number of bins. Should be non-negative integer. Default is 0.
        name (str|None, optional): Normally there is no need for user to set this property.
            For more information, please refer to :ref:`api_guide_Name`. Default is None.

    Returns:
        Tensor: The tensor of frequency.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([1, 2, 1, 4, 5])
            >>> result1 = paddle.bincount(x)
            >>> print(result1)
            Tensor(shape=[6], dtype=int64, place=Place(cpu), stop_gradient=True,
            [0, 2, 1, 0, 1, 1])

            >>> w = paddle.to_tensor([2.1, 0.4, 0.1, 0.5, 0.5])
            >>> result2 = paddle.bincount(x, weights=w)
            >>> print(result2)
            Tensor(shape=[6], dtype=float32, place=Place(cpu), stop_gradient=True,
            [0.        , 2.19999981, 0.40000001, 0.        , 0.50000000, 0.50000000])
    """
    if x.dtype not in [
        paddle.int32,
        paddle.int64,
        DataType.INT32,
        DataType.INT64,
    ]:
        raise TypeError("Elements in Input(x) should all be integers")

    if in_dynamic_or_pir_mode():
        return _C_ops.bincount(x, weights, minlength)
    else:
        helper = LayerHelper('bincount', **locals())

        check_variable_and_dtype(x, 'X', ['int32', 'int64'], 'bincount')

        if weights is not None:
            check_variable_and_dtype(
                weights,
                'Weights',
                ['int32', 'int64', 'float32', 'float64'],
                'bincount',
            )
            out = helper.create_variable_for_type_inference(dtype=weights.dtype)
        else:
            out = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(
            type='bincount',
            inputs={'X': x, 'Weights': weights},
            outputs={'Out': out},
            attrs={'minlength': minlength},
        )
        return out


def mv(x: Tensor, vec: Tensor, name: str | None = None) -> Tensor:
    """
    Performs a matrix-vector product of the matrix x and the vector vec.

    Args:
        x (Tensor): A tensor with shape :math:`[M, N]` , The data type of the input Tensor x
            should be one of float32, float64.
        vec (Tensor): A tensor with shape :math:`[N]` , The data type of the input Tensor x
            should be one of float32, float64.
        name (str|None, optional): Normally there is no need for user to set this property.
            For more information, please refer to :ref:`api_guide_Name`. Default is None.

    Returns:
        Tensor: The tensor which is produced by x and vec.

    Examples:
        .. code-block:: python

            >>> # x: [M, N], vec: [N]
            >>> # paddle.mv(x, vec)  # out: [M]

            >>> import paddle

            >>> x = paddle.to_tensor([[2, 1, 3], [3, 0, 1]]).astype("float64")
            >>> vec = paddle.to_tensor([3, 5, 1]).astype("float64")
            >>> out = paddle.mv(x, vec)
            >>> print(out)
            Tensor(shape=[2], dtype=float64, place=Place(cpu), stop_gradient=True,
            [14., 10.])
    """
    if in_dynamic_or_pir_mode():
        return _C_ops.mv(x, vec)
    else:

        def __check_input(x, vec):
            var_names = {'x': x, 'vec': vec}
            for name, val in var_names.items():
                check_variable_and_dtype(
                    val, name, ['float32', 'float64'], 'mv'
                )
            x_shape = list(x.shape)
            vec_shape = list(vec.shape)
            if len(x_shape) != 2:
                raise ValueError(
                    f"x should be 2-dimensional. But received x's dimension: {x_shape}"
                )
            if len(vec_shape) != 1:
                raise ValueError(
                    f"vec should be 1-dimensional. But received vec's dimension: {vec_shape}"
                )

        __check_input(x, vec)

        helper = LayerHelper('mv', **locals())
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(
            type='mv', inputs={'X': x, 'Vec': vec}, outputs={'Out': out}
        )
        return out


def det(x: Tensor, name: str | None = None) -> Tensor:
    """

    Calculates determinant value of a square matrix or batches of square matrices.

    Args:
        x (Tensor): the input matrix of size `(n, n)` or the
            batch of matrices of size `(*, n, n)` where `*` is one or more
            batch dimensions.
        name (str|None, optional): Name of the output.It's used to print debug info for
            developers. Details: :ref:`api_guide_Name`. Default is None.

    Returns:
        Tensor, the determinant value of a square matrix or batches of square matrices.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.seed(2023)
            >>> x =  paddle.randn([3,3,3])
            >>> A = paddle.linalg.det(x)
            >>> print(A)
            Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [-1.29280925,  0.77832544,  0.89754158])


    """
    if in_dynamic_or_pir_mode():
        return _C_ops.det(x)
    else:
        check_dtype(
            x.dtype,
            'Input',
            ['float16', 'float32', 'float64', 'complex64', 'complex128'],
            'det',
        )

        input_shape = list(x.shape)
        assert len(input_shape) >= 2, (
            "The x must be at least 2-dimensional, "
            f"but received Input x's dimensional: {len(input_shape)}.\n"
        )

        assert input_shape[-1] == input_shape[-2], (
            "Expect squared input,"
            f"but received {input_shape[-2]} by {input_shape[-1]} matrix.\n"
        )
        helper = LayerHelper('determinant', **locals())
        out = helper.create_variable_for_type_inference(dtype=x.dtype)

        helper.append_op(
            type='determinant', inputs={'Input': [x]}, outputs={'Out': [out]}
        )
        return out


def slogdet(x: Tensor, name: str | None = None) -> Tensor:
    """

    Calculates the sign and natural logarithm of the absolute value of a square matrix's or batches square matrices' determinant.
    The determinant can be computed with ``sign * exp`` (logabsdet).

    Supports input of float, double, complex64, complex128.

    Notes:
        1. For matrices that have zero determinant, this returns ``(0, -inf)``.

        2. For matrices with complex value, the :math:`abs(det)` is the modulus of the determinant,
        and therefore :math:`sign = det / abs(det)`.

    Args:
        x (Tensor): the batch of matrices of size :math:`(*, n, n)`
            where math:`*` is one or more batch dimensions.
        name (str|None, optional): Name of the output.It's used to print debug info for
            developers. Details: :ref:`api_guide_Name`. Default is None.

    Returns:
        y (Tensor), A tensor containing the sign of the determinant and the natural logarithm
        of the absolute value of determinant, respectively. The output shape is :math:`(2, *)`,
        where math:`*` is one or more batch dimensions of the input `x`.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.seed(2023)
            >>> x = paddle.randn([3, 3, 3])
            >>> A = paddle.linalg.slogdet(x)
            >>> print(A)
            Tensor(shape=[2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[-1.        ,  1.        ,  1.        ],
             [ 0.25681755, -0.25061053, -0.10809582]])

    """
    if in_dynamic_or_pir_mode():
        return _C_ops.slogdet(x)
    else:
        check_dtype(
            x.dtype,
            'Input',
            ['float32', 'float64', 'complex64', 'complex128'],
            'slogdet',
        )

        input_shape = list(x.shape)
        assert len(input_shape) >= 2, (
            "The x must be at least 2-dimensional, "
            f"but received Input x's dimensional: {len(input_shape)}.\n"
        )

        assert input_shape[-1] == input_shape[-2], (
            "Expect squared input,"
            f"but received {input_shape[-2]} by {input_shape[-1]} matrix.\n"
        )
        helper = LayerHelper('slogdeterminant', **locals())
        out = helper.create_variable_for_type_inference(dtype=x.dtype)

        helper.append_op(
            type='slogdeterminant',
            inputs={'Input': [x]},
            outputs={'Out': [out]},
        )
        return out


def svd(
    x: Tensor, full_matrices: bool = False, name: str | None = None
) -> tuple[Tensor, Tensor, Tensor]:
    r"""
    Computes the singular value decomposition of one matrix or a batch of regular matrices.

    Let :math:`X` be the input matrix or a batch of input matrices, the output should satisfies:

    .. math::
        X = U * diag(S) * V^{H}

    Args:
        x (Tensor): The input tensor. Its shape should be `[..., N, M]`,
            where `...` is zero or more batch dimensions. N and M can be arbitrary
            positive number. Note that if x is singular matrices, the grad is numerical
            instable. The data type of x should be float32 or float64.
        full_matrices (bool, optional): A flag to control the behavior of svd.
            If full_matrices = True, svd op will compute full U and V matrices,
            which means shape of U is `[..., N, N]`, shape of V is `[..., M, M]`. K = min(M, N).
            If full_matrices = False, svd op will use a economic method to store U and V.
            which means shape of U is `[..., N, K]`, shape of V is `[..., M, K]`. K = min(M, N).
            Default value is False.
        name (str|None, optional): Name for the operation. For more information,
            please refer to :ref:`api_guide_Name`. Default value is None.

    Returns:
        - U (Tensor), is the singular value decomposition result U.
        - S (Tensor), is the singular value decomposition result S.
        - VH (Tensor), VH is the conjugate transpose of V, which is the singular value decomposition result V.

        Tuple of 3 tensors(U, S, VH): VH is the conjugate transpose of V. S is the singular value vectors of matrices with shape `[..., K]`

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([[1.0, 2.0], [1.0, 3.0], [4.0, 6.0]]).astype('float64')
            >>> x = x.reshape([3, 2])
            >>> u, s, vh = paddle.linalg.svd(x)
            >>> print (u)
            Tensor(shape=[3, 2], dtype=float64, place=Place(cpu), stop_gradient=True,
            [[-0.27364809, -0.21695147],
             [-0.37892198, -0.87112408],
             [-0.88404460,  0.44053933]])

            >>> print (s)
            Tensor(shape=[2], dtype=float64, place=Place(cpu), stop_gradient=True,
            [8.14753743, 0.78589688])

            >>> print (vh)
            Tensor(shape=[2, 2], dtype=float64, place=Place(cpu), stop_gradient=True,
            [[-0.51411221, -0.85772294],
             [ 0.85772294, -0.51411221]])

            >>> # one can verify : U * S * VT == X
            >>> #                  U * UH == I
            >>> #                  V * VH == I
    """

    if in_dynamic_or_pir_mode():
        return _C_ops.svd(x, full_matrices)
    else:
        check_variable_and_dtype(x, 'dtype', ['float32', 'float64'], 'svd')
        check_type(full_matrices, 'full_matrices', bool, 'svd')
        helper = LayerHelper('svd', **locals())
        u = helper.create_variable_for_type_inference(dtype=x.dtype)
        vh = helper.create_variable_for_type_inference(dtype=x.dtype)
        s = helper.create_variable_for_type_inference(dtype=x.dtype)
        attrs = {}
        attrs['full_matrices'] = full_matrices
        helper.append_op(
            type='svd',
            inputs={'X': [x]},
            outputs={'U': u, 'VH': vh, 'S': s},
            attrs=attrs,
        )
        return u, s, vh


def svdvals(x: Tensor, name: str | None = None) -> Tensor:
    r"""
    Computes the singular values of one matrix or a batch of matrices.

    Let :math:`X` be the input matrix or a batch of input matrices,
    the output singular values :math:`S` are the diagonal elements of the matrix
    produced by singular value decomposition:

    .. math::
        X = U * diag(S) * V^{H}

    Args:
        x (Tensor): The input tensor. Its shape should be `[..., M, N]`, where
            `...` is zero or more batch dimensions. The data type of x should
            be float32 or float64.
        name (str|None, optional): Name for the operation. For more
            information, please refer to :ref:`api_guide_Name`.
            Default: None.

    Returns:
        Tensor: Singular values of x. The shape is `[..., K]`, where `K = min(M, N)`.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([[1.0, 2.0], [1.0, 3.0], [4.0, 6.0]])
            >>> s = paddle.linalg.svdvals(x)
            >>> print(s)
            Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [8.14753819, 0.78589684])
    """
    return _C_ops.svdvals(x)


def _conjugate(x):
    if x.is_complex():
        return x.conj()
    return x


def _transpose(x):
    shape = x.shape
    perm = list(range(0, len(shape)))
    perm = perm[:-2] + [perm[-1]] + [perm[-2]]
    return paddle.transpose(x, perm)


def _transjugate(x):
    return _conjugate(_transpose(x))


def _get_approximate_basis(x, q, niter=2, M=None):
    niter = 2 if niter is None else niter
    m, n = x.shape[-2:]
    qr = paddle.linalg.qr

    R = paddle.randn((n, q), dtype=x.dtype)

    A_t = _transpose(x)
    A_H = _conjugate(A_t)
    if M is None:
        Q = qr(paddle.matmul(x, R))[0]
        for i in range(niter):
            Q = qr(paddle.matmul(A_H, Q))[0]
            Q = qr(paddle.matmul(x, Q))[0]
    else:
        M_H = _transjugate(M)
        Q = qr(paddle.matmul(x, R) - paddle.matmul(M, R))[0]
        for i in range(niter):
            Q = qr(paddle.matmul(A_H, Q) - paddle.matmul(M_H, Q))[0]
            Q = qr(paddle.matmul(x, Q) - paddle.matmul(M, Q))[0]

    return Q


def svd_lowrank(
    x: Tensor,
    q: int | None = None,
    niter: int = 2,
    M: Tensor | None = None,
    name: str | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    r"""
    Return the singular value decomposition (SVD) on a low-rank matrix or batches of such matrices.

    If :math:`X` is the input matrix or a batch of input matrices, the output should satisfies:

    .. math::
        X \approx U * diag(S) * V^{H}

    When :math:`M` is given, the output should satisfies:

    .. math::
        X - M \approx U * diag(S) * V^{H}

    Args:
        x (Tensor): The input tensor. Its shape should be `[..., N, M]`, where `...` is
            zero or more batch dimensions. N and M can be arbitrary positive number.
            The data type of ``x`` should be float32 or float64.
        q (int, optional): A slightly overestimated rank of :math:`X`.
            Default value is None, which means the overestimated rank is 6.
        niter (int, optional): The number of iterations to perform. Default: 2.
        M (Tensor, optional): The input tensor's mean. Its shape should be `[..., 1, M]`.
            Default value is None.
        name (str|None, optional): Name for the operation. For more information, please
            refer to :ref:`api_guide_Name`. Default: None.

    Returns:
        - Tensor U, is N x q matrix.
        - Tensor S, is a vector with length q.
        - Tensor V, is M x q matrix.

        tuple (U, S, V): which is the nearly optimal approximation of a singular value decomposition of the matrix :math:`X` or :math:`X - M`.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.seed(2024)

            >>> x = paddle.randn((5, 5), dtype='float64')
            >>> U, S, V = paddle.linalg.svd_lowrank(x)
            >>> print(U)
            Tensor(shape=[5, 5], dtype=float64, place=Place(cpu), stop_gradient=True,
            [[-0.03586982, -0.17211503,  0.31536566, -0.38225676, -0.85059629],
             [-0.38386839,  0.67754925,  0.23222694,  0.51777188, -0.26749766],
             [-0.85977150, -0.28442378, -0.41412094, -0.08955629, -0.01948348],
             [ 0.18611503,  0.56047358, -0.67717019, -0.39286761, -0.19577062],
             [ 0.27841082, -0.34099254, -0.46535957,  0.65071250, -0.40770727]])

            >>> print(S)
            Tensor(shape=[5], dtype=float64, place=Place(cpu), stop_gradient=True,
            [4.11253399, 3.03227120, 2.45499752, 1.25602436, 0.45825337])

            >>> print(V)
            Tensor(shape=[5, 5], dtype=float64, place=Place(cpu), stop_gradient=True,
            [[ 0.46401347,  0.50977695, -0.08742316, -0.11140428, -0.71046833],
             [-0.48927226, -0.35047624,  0.07918771,  0.45431083, -0.65200463],
             [-0.20494730,  0.67097011, -0.05427719,  0.66510472,  0.24997083],
             [-0.69645001,  0.40237917,  0.09360970, -0.58032322, -0.08666357],
             [ 0.13512270,  0.07199989,  0.98710572,  0.04529277,  0.01134594]])
    """
    if not paddle.is_tensor(x):
        raise ValueError(f'Input must be tensor, but got {type(x)}')

    m, n = x.shape[-2:]
    if q is None:
        q = min(6, m, n)
    elif not (q >= 0 and q <= min(m, n)):
        raise ValueError(
            f'q(={q}) must be non-negative integer'
            f' and not greater than min(m, n)={min(m, n)}'
        )

    if not (niter >= 0):
        raise ValueError(f'niter(={niter}) must be non-negative integer')

    if M is None:
        M_t = None
    else:
        M = M.broadcast_to(x.shape)
        M_t = _transpose(M)
    A_t = _transpose(x)

    if m < n or n > q:
        Q = _get_approximate_basis(A_t, q, niter=niter, M=M_t)
        Q_c = _conjugate(Q)
        if M is None:
            B_t = paddle.matmul(x, Q_c)
        else:
            B_t = paddle.matmul(x, Q_c) - paddle.matmul(M, Q_c)
        assert B_t.shape[-2] == m, (B_t.shape, m)
        assert B_t.shape[-1] == q, (B_t.shape, q)
        assert B_t.shape[-1] <= B_t.shape[-2], B_t.shape
        U, S, Vh = paddle.linalg.svd(B_t, full_matrices=False)
        V = _transjugate(Vh)
        V = Q.matmul(V)
    else:
        Q = _get_approximate_basis(x, q, niter=niter, M=M)
        Q_c = _conjugate(Q)
        if M is None:
            B = paddle.matmul(A_t, Q_c)
        else:
            B = paddle.matmul(A_t, Q_c) - paddle.matmul(M_t, Q_c)
        B_t = _transpose(B)
        assert B_t.shape[-2] == q, (B_t.shape, q)
        assert B_t.shape[-1] == n, (B_t.shape, n)
        assert B_t.shape[-1] <= B_t.shape[-2], B_t.shape
        U, S, Vh = paddle.linalg.svd(B_t, full_matrices=False)
        V = _transjugate(Vh)
        U = Q.matmul(U)

    return U, S, V


def pca_lowrank(
    x: Tensor,
    q: int | None = None,
    center: bool = True,
    niter: int = 2,
    name: str | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    r"""
    Performs linear Principal Component Analysis (PCA) on a low-rank matrix or batches of such matrices.

    Let :math:`X` be the input matrix or a batch of input matrices, the output should satisfies:

    .. math::
        X = U * diag(S) * V^{T}

    Args:
        x (Tensor): The input tensor. Its shape should be `[..., N, M]`,
            where `...` is zero or more batch dimensions. N and M can be arbitrary
            positive number. The data type of x should be float32 or float64.
        q (int, optional): a slightly overestimated rank of :math:`X`.
            Default value is :math:`q=min(6,N,M)`.
        center (bool, optional): if True, center the input tensor.
            Default value is True.
        niter (int, optional): number of iterations to perform. Default: 2.
        name (str|None, optional): Name for the operation. For more information,
            please refer to :ref:`api_guide_Name`. Default: None.

    Returns:
        - Tensor U, is N x q matrix.
        - Tensor S, is a vector with length q.
        - Tensor V, is M x q matrix.

        tuple (U, S, V): which is the nearly optimal approximation of a singular value decomposition of a centered matrix :math:`X`.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.seed(2023)

            >>> x = paddle.randn((5, 5), dtype='float64')
            >>> U, S, V = paddle.linalg.pca_lowrank(x)
            >>> print(U)
           Tensor(shape=[5, 5], dtype=float64, place=Place(cpu), stop_gradient=True,
           [[ 0.80131563,  0.11962647,  0.27667179, -0.25891214,  0.44721360],
            [-0.12642301,  0.69917551, -0.17899393,  0.51296394,  0.44721360],
            [ 0.08997135, -0.69821706, -0.20059228,  0.51396579,  0.44721360],
            [-0.23871837, -0.02815453, -0.59888153, -0.61932365,  0.44721360],
            [-0.52614559, -0.09243040,  0.70179595, -0.14869394,  0.44721360]])

            >>> print(S)
            Tensor(shape=[5], dtype=float64, place=Place(cpu), stop_gradient=True,
            [2.60101614, 2.40554940, 1.49768346, 0.19064830, 0.00000000])

            >>> print(V)
            Tensor(shape=[5, 5], dtype=float64, place=Place(cpu), stop_gradient=True,
            [[ 0.58339481, -0.17143771,  0.00522143,  0.57976310,  0.54231640],
             [ 0.22334335,  0.72963474, -0.30148399, -0.39388750,  0.41438019],
             [ 0.05416913,  0.34666487,  0.93549758,  0.00063507,  0.04162998],
             [-0.39519094,  0.53074980, -0.16687419,  0.71175586, -0.16638919],
             [-0.67131070, -0.19071018,  0.07795789, -0.04615811,  0.71046714]])
    """

    if not paddle.is_tensor(x):
        raise ValueError(f'Input must be tensor, but got {type(x)}')

    (m, n) = x.shape[-2:]

    if q is None:
        q = min(6, m, n)
    elif not (q >= 0 and q <= min(m, n)):
        raise ValueError(
            f'q(={q}) must be non-negative integer'
            f' and not greater than min(m, n)={min(m, n)}'
        )
    if not (niter >= 0):
        raise ValueError(f'niter(={niter}) must be non-negative integer')

    if not center:
        return svd_lowrank(x, q, niter=niter, M=None)

    C = x.mean(axis=-2, keepdim=True)
    return svd_lowrank(x - C, q, niter=niter, M=None)


def matrix_power(
    x: Tensor, n: int, name: str | None = None
) -> tuple[Tensor, int]:
    r"""

    Computes the n-th power of a square matrix or a batch of square matrices.

    Let :math:`X` be a square matrix or a batch of square matrices, :math:`n` be
    an exponent, the equation should be:

    .. math::
        Out = X ^ {n}

    Specifically,

    - If `n > 0`, it returns the matrix or a batch of matrices raised to the power of `n`.

    - If `n = 0`, it returns the identity matrix or a batch of identity matrices.

    - If `n < 0`, it returns the inverse of each matrix (if invertible) raised to the power of `abs(n)`.

    Args:
        x (Tensor): A square matrix or a batch of square matrices to be raised
            to power `n`. Its shape should be `[*, M, M]`, where `*` is zero or
            more batch dimensions. Its data type should be float32 or float64.
        n (int): The exponent. It can be any positive, negative integer or zero.
        name (str|None, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        - Tensor, The n-th power of the matrix (or the batch of matrices) `x`. Its
          data type should be the same as that of `x`.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([[1, 2, 3],
            ...                       [1, 4, 9],
            ...                       [1, 8, 27]], dtype='float64')
            >>> print(paddle.linalg.matrix_power(x, 2))
            Tensor(shape=[3, 3], dtype=float64, place=Place(cpu), stop_gradient=True,
            [[6.  , 34. , 102.],
             [14. , 90. , 282.],
             [36. , 250., 804.]])

            >>> print(paddle.linalg.matrix_power(x, 0))
            Tensor(shape=[3, 3], dtype=float64, place=Place(cpu), stop_gradient=True,
            [[1., 0., 0.],
             [0., 1., 0.],
             [0., 0., 1.]])

            >>> print(paddle.linalg.matrix_power(x, -2))
            Tensor(shape=[3, 3], dtype=float64, place=Place(cpu), stop_gradient=True,
            [[ 12.91666667, -12.75000000,  2.83333333 ],
             [-7.66666667 ,  8.         , -1.83333333 ],
             [ 1.80555556 , -1.91666667 ,  0.44444444 ]])
    """
    if in_dynamic_or_pir_mode():
        return _C_ops.matrix_power(x, n)
    else:
        check_variable_and_dtype(
            x, 'dtype', ['float32', 'float64'], 'matrix_power'
        )
        check_type(n, 'n', int, 'matrix_power')
        helper = LayerHelper('matrix_power', **locals())
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(
            type='matrix_power',
            inputs={'X': x},
            outputs={'Out': out},
            attrs={'n': n},
        )
        return out


@overload
def qr(
    x: Tensor,
    mode: Literal['reduced', 'complete'] = ...,
    name: str | None = ...,
) -> tuple[Tensor, Tensor]: ...


@overload
def qr(
    x: Tensor,
    mode: Literal['r'] = ...,
    name: str | None = ...,
) -> Tensor: ...


def qr(
    x,
    mode="reduced",
    name=None,
) -> Tensor | tuple[Tensor, Tensor]:
    r"""
    Computes the QR decomposition of one matrix or batches of matrices (backward is unsupported now).

    Args:
        x (Tensor): The input tensor. Its shape should be `[..., M, N]`,
            where ... is zero or more batch dimensions. M and N can be arbitrary
            positive number. The data type of x supports float, double, complex64, complex128.
        mode (str, optional): A flag to control the behavior of qr.
            Suppose x's shape is `[..., M, N]` and denoting `K = min(M, N)`:
            If mode = "reduced", qr op will return reduced Q and R matrices,
            which means Q's shape is `[..., M, K]` and R's shape is `[..., K, N]`.
            If mode = "complete", qr op will return complete Q and R matrices,
            which means Q's shape is `[..., M, M]` and R's shape is `[..., M, N]`.
            If mode = "r", qr op will only return reduced R matrix, which means
            R's shape is `[..., K, N]`. Default: "reduced".
        name (str|None, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        If mode = "reduced" or mode = "complete", qr will return a two tensor-tuple, which represents Q and R.
        If mode = "r", qr will return a tensor which represents R.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).astype('float64')
            >>> q, r = paddle.linalg.qr(x)
            >>> print(q)
            Tensor(shape=[3, 2], dtype=float64, place=Place(cpu), stop_gradient=True,
            [[-0.16903085,  0.89708523],
             [-0.50709255,  0.27602622],
             [-0.84515425, -0.34503278]])
            >>> print(r)
            Tensor(shape=[2, 2], dtype=float64, place=Place(cpu), stop_gradient=True,
            [[-5.91607978, -7.43735744],
             [ 0.        ,  0.82807867]])

            >>> # one can verify : X = Q * R ;
    """
    if in_dynamic_or_pir_mode():
        q, r = _C_ops.qr(x, mode)
        if mode == "r":
            return r
        else:
            return q, r
    else:
        check_variable_and_dtype(
            x, 'dtype', ['float32', 'float64', 'complex64', 'complex128'], 'qr'
        )
        check_type(mode, 'mode', str, 'qr')
        helper = LayerHelper('qr', **locals())
        q = helper.create_variable_for_type_inference(dtype=x.dtype)
        r = helper.create_variable_for_type_inference(dtype=x.dtype)
        attrs = {}
        attrs['mode'] = mode
        helper.append_op(
            type='qr', inputs={'X': [x]}, outputs={'Q': q, 'R': r}, attrs=attrs
        )
        if mode == "r":
            return r
        else:
            return q, r


@overload
def lu(
    x: Tensor,
    pivot: bool = ...,
    get_infos: Literal[False] = ...,
    name: str | None = ...,
) -> tuple[Tensor, Tensor]: ...


@overload
def lu(
    x: Tensor,
    pivot: bool = ...,
    get_infos: Literal[True] = ...,
    name: str | None = ...,
) -> tuple[Tensor, Tensor, Tensor]: ...


@overload
def lu(
    x: Tensor, pivot: bool = ..., get_infos: bool = ..., name: str | None = ...
) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]: ...


def lu(
    x, pivot=True, get_infos=False, name=None
) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
    r"""
    Computes the LU factorization of an N-D(N>=2) matrix x.

    Returns the LU factorization(inplace x) and Pivots. low triangular matrix L and
    upper triangular matrix U are combined to a single LU matrix.

    Pivoting is done if pivot is set to True.
    P mat can be get by pivots:

    .. code-block:: text

        ones = eye(rows) #eye matrix of rank rows
        for i in range(cols):
            swap(ones[i], ones[pivots[i]])
        return ones

    Args:

        X (Tensor): the tensor to factor of N-dimensions(N>=2).

        pivot (bool, optional): controls whether pivoting is done. Default: True.

        get_infos (bool, optional): if set to True, returns an info IntTensor. Default: False.

        name (str|None, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        factorization (Tensor), LU matrix, the factorization of input X.

        pivots (IntTensor), the pivots of size(\*(N-2), min(m,n)). `pivots` stores all the
        intermediate transpositions of rows. The final permutation `perm` could be
        reconstructed by this, details refer to upper example.

        infos (IntTensor, optional), if `get_infos` is `True`, this is a tensor of size (\*(N-2))
        where non-zero values indicate whether factorization for the matrix or each minibatch
        has succeeded or failed.


    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).astype('float64')
            >>> lu,p,info = paddle.linalg.lu(x, get_infos=True)

            >>> print(lu)
            Tensor(shape=[3, 2], dtype=float64, place=Place(cpu), stop_gradient=True,
            [[5.        , 6.        ],
             [0.20000000, 0.80000000],
             [0.60000000, 0.50000000]])
            >>> print(p)
            Tensor(shape=[2], dtype=int32, place=Place(cpu), stop_gradient=True,
            [3, 3])
            >>> print(info)
            Tensor(shape=[], dtype=int32, place=Place(cpu), stop_gradient=True,
            0)

            >>> P,L,U = paddle.linalg.lu_unpack(lu,p)

            >>> print(P)
            Tensor(shape=[3, 3], dtype=float64, place=Place(cpu), stop_gradient=True,
            [[0., 1., 0.],
             [0., 0., 1.],
             [1., 0., 0.]])
            >>> print(L)
            Tensor(shape=[3, 2], dtype=float64, place=Place(cpu), stop_gradient=True,
            [[1.        , 0.        ],
             [0.20000000, 1.        ],
             [0.60000000, 0.50000000]])
            >>> print(U)
            Tensor(shape=[2, 2], dtype=float64, place=Place(cpu), stop_gradient=True,
            [[5.        , 6.        ],
             [0.        , 0.80000000]])

            >>> # one can verify : X = P @ L @ U ;
    """

    if in_dynamic_or_pir_mode():
        lu, p, info = _C_ops.lu(x, pivot)
    else:
        check_variable_and_dtype(x, 'dtype', ['float32', 'float64'], 'lu')
        helper = LayerHelper('lu', **locals())
        lu = helper.create_variable_for_type_inference(dtype=x.dtype)
        p = helper.create_variable_for_type_inference(dtype='int')
        info = helper.create_variable_for_type_inference(dtype='int')
        attrs = {}
        attrs['pivot'] = pivot
        helper.append_op(
            type='lu',
            inputs={'X': x},
            outputs={'Out': lu, 'Pivots': p, 'Infos': info},
            attrs=attrs,
        )
    if get_infos:
        return lu, p, info
    else:
        return lu, p


def lu_unpack(
    x: Tensor,
    y: Tensor,
    unpack_ludata: bool = True,
    unpack_pivots: bool = True,
    name: str | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    r"""
    Unpack L U and P to single matrix tensor .
    unpack L and U matrix from LU, unpack permutation matrix P from Pivtos .

    P mat can be get by pivots:

    .. code-block:: text

        ones = eye(rows) #eye matrix of rank rows
        for i in range(cols):
            swap(ones[i], ones[pivots[i]])


    Args:
        x (Tensor): The LU tensor get from paddle.linalg.lu, which is combined by L and U.

        y (Tensor): Pivots get from paddle.linalg.lu.

        unpack_ludata (bool, optional): whether to unpack L and U from x. Default: True.

        unpack_pivots (bool, optional): whether to unpack permutation matrix P from Pivtos. Default: True.

        name (str|None, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        P (Tensor), Permutation matrix P of lu factorization.

        L (Tensor), The lower triangular matrix tensor of lu factorization.

        U (Tensor), The upper triangular matrix tensor of lu factorization.


    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).astype('float64')
            >>> lu,p,info = paddle.linalg.lu(x, get_infos=True)

            >>> print(lu)
            Tensor(shape=[3, 2], dtype=float64, place=Place(cpu), stop_gradient=True,
            [[5.        , 6.        ],
             [0.20000000, 0.80000000],
             [0.60000000, 0.50000000]])
            >>> print(p)
            Tensor(shape=[2], dtype=int32, place=Place(cpu), stop_gradient=True,
            [3, 3])
            >>> print(info)
            Tensor(shape=[1], dtype=int32, place=Place(cpu), stop_gradient=True,
            [0])

            >>> P,L,U = paddle.linalg.lu_unpack(lu,p)

            >>> print(P)
            Tensor(shape=[3, 3], dtype=float64, place=Place(cpu), stop_gradient=True,
            [[0., 1., 0.],
             [0., 0., 1.],
             [1., 0., 0.]])
            >>> print(L)
            Tensor(shape=[3, 2], dtype=float64, place=Place(cpu), stop_gradient=True,
            [[1.        , 0.        ],
             [0.20000000, 1.        ],
             [0.60000000, 0.50000000]])
            >>> print(U)
            Tensor(shape=[2, 2], dtype=float64, place=Place(cpu), stop_gradient=True,
            [[5.        , 6.        ],
             [0.        , 0.80000000]])

            >>> # one can verify : X = P @ L @ U ;
    """
    if x.ndim < 2:
        raise ValueError(
            f"The shape of x should be (*, M, N), but received ndim is [{x.ndim} < 2]"
        )
    if y.ndim < 1:
        raise ValueError(
            f"The shape of Pivots should be (*, K), but received ndim is [{y.ndim} < 1]"
        )
    if in_dynamic_or_pir_mode():
        P, L, U = _C_ops.lu_unpack(x, y, unpack_ludata, unpack_pivots)
        return P, L, U
    else:
        check_variable_and_dtype(
            x, 'dtype', ['float32', 'float64'], 'lu_unpack'
        )
        helper = LayerHelper('lu_unpack', **locals())
        p = helper.create_variable_for_type_inference(dtype=x.dtype)
        l = helper.create_variable_for_type_inference(dtype=x.dtype)
        u = helper.create_variable_for_type_inference(dtype=x.dtype)

        attrs = {}
        attrs['unpack_ludata'] = unpack_ludata
        attrs['unpack_pivots'] = unpack_pivots
        helper.append_op(
            type='lu_unpack',
            inputs={'X': x, 'Pivots': y},
            outputs={'Pmat': p, 'L': l, 'U': u},
            attrs=attrs,
        )
        return p, l, u


def eig(x: Tensor, name: str | None = None) -> tuple[Tensor, Tensor]:
    """
    Performs the eigenvalue decomposition of a square matrix or a batch of square matrices.

    Note:
        - If the matrix is a Hermitian or a real symmetric matrix, please use :ref:`api_paddle_linalg_eigh` instead, which is much faster.
        - If only eigenvalues is needed, please use :ref:`api_paddle_linalg_eigvals` instead.
        - If the matrix is of any shape, please use :ref:`api_paddle_linalg_svd`.
        - This API is only supported on CPU device.
        - The output datatype is always complex for both real and complex input.

    Args:
        x (Tensor): A tensor with shape math:`[*, N, N]`, The data type of the x should be one of ``float32``,
            ``float64``, ``compplex64`` or ``complex128``.
        name (str|None, optional): The default value is `None`. Normally there is no need for user to set
            this property. For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Eigenvalues(Tensor): A tensor with shape math:`[*, N]` refers to the eigen values.
        Eigenvectors(Tensor): A tensor with shape math:`[*, N, N]` refers to the eigen vectors.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([[1.6707249, 7.2249975, 6.5045543],
            ...                       [9.956216,  8.749598,  6.066444 ],
            ...                       [4.4251957, 1.7983172, 0.370647 ]])
            >>> w, v = paddle.linalg.eig(x)
            >>> print(v)
            Tensor(shape=[3, 3], dtype=complex64, place=Place(cpu), stop_gradient=True,
            [[ (0.5061365365982056+0j) ,  (0.7971761226654053+0j) ,
               (0.1851806491613388+0j) ],
             [ (0.8308236598968506+0j) , (-0.3463813066482544+0j) ,
               (-0.6837005615234375+0j) ],
             [ (0.23142573237419128+0j), (-0.49449989199638367+0j),
               (0.7058765292167664+0j) ]])

            >>> print(w)
            Tensor(shape=[3], dtype=complex64, place=Place(cpu), stop_gradient=True,
            [ (16.50470733642578+0j)  , (-5.503481388092041+0j)  ,
              (-0.21026138961315155+0j)])
    """

    if in_dynamic_or_pir_mode():
        return _C_ops.eig(x)
    else:
        check_variable_and_dtype(
            x, 'X', ['float32', 'float64', 'complex64', 'complex128'], 'eig'
        )
        helper = LayerHelper('eig', **locals())

        w = helper.create_variable_for_type_inference(x.dtype)
        v = helper.create_variable_for_type_inference(x.dtype)

        inputs = {'X': x}
        outputs = {'Eigenvalues': w, 'Eigenvectors': v}
        helper.append_op(type='eig', inputs=inputs, outputs=outputs)

        return w, v


def eigvals(x: Tensor, name: str | None = None) -> Tensor:
    """
    Compute the eigenvalues of one or more general matrices.

    Warning:
        The gradient kernel of this operator does not yet developed.
        If you need back propagation through this operator, please replace it with paddle.linalg.eig.

    Args:
        x (Tensor): A square matrix or a batch of square matrices whose eigenvalues will be computed.
            Its shape should be `[*, M, M]`, where `*` is zero or more batch dimensions.
            Its data type should be float32, float64, complex64, or complex128.
        name (str|None, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, A tensor containing the unsorted eigenvalues which has the same batch
        dimensions with `x`. The eigenvalues are complex-valued even when `x` is real.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.seed(2023)

            >>> x = paddle.rand(shape=[3, 3], dtype='float64')
            >>> print(x)
            Tensor(shape=[3, 3], dtype=float64, place=Place(cpu), stop_gradient=True,
            [[0.86583615, 0.52014721, 0.25960938],
             [0.90525323, 0.42400090, 0.40641288],
             [0.97020893, 0.74437359, 0.51785128]])

            >>> print(paddle.linalg.eigvals(x))
            Tensor(shape=[3], dtype=complex128, place=Place(cpu), stop_gradient=True,
            [ (1.788956694280852+0j)  ,  (0.16364484879581526+0j),
              (-0.14491322408727625+0j)])
    """

    x_shape = list(x.shape)
    if len(x_shape) < 2:
        raise ValueError(
            f"The dimension of Input(x) should be at least 2, but received x's dimension = {len(x_shape)}, x's shape = {x_shape}"
        )

    if x_shape[-1] != x_shape[-2]:
        raise ValueError(
            f"The last two dimensions of Input(x) should be equal, but received x's shape = {x_shape}"
        )

    if in_dynamic_or_pir_mode():
        return _C_ops.eigvals(x)
    else:
        check_variable_and_dtype(
            x,
            'dtype',
            ['float32', 'float64', 'complex64', 'complex128'],
            'eigvals',
        )
        helper = LayerHelper('eigvals', **locals())
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(type='eigvals', inputs={'X': x}, outputs={'Out': out})
        return out


def multi_dot(x: list[Tensor], name: str | None = None) -> Tensor:
    """
    Multi_dot is an operator that calculates multiple matrix multiplications.

    Supports inputs of float16(only GPU support), float32 and float64 dtypes. This function does not
    support batched inputs.

    The input tensor in [x] must be 2-D except for the first and last can be 1-D.
    If the first tensor is a 1-D vector of shape(n, ) it is treated as row vector
    of shape(1, n), similarly if the last tensor is a 1D vector of shape(n, ), it
    is treated as a column vector of shape(n, 1).

    If the first and last tensor are 2-D matrix, then the output is also 2-D matrix,
    otherwise the output is a 1-D vector.

    Multi_dot will select the lowest cost multiplication order for calculation. The
    cost of multiplying two matrices with shapes (a, b) and (b, c) is a * b * c.
    Given matrices A, B, C with shapes (20, 5), (5, 100), (100, 10) respectively,
    we can calculate the cost of different multiplication orders as follows:
    - Cost((AB)C) = 20x5x100 + 20x100x10 = 30000
    - Cost(A(BC)) = 5x100x10 + 20x5x10 = 6000

    In this case, multiplying B and C first, then multiply A, which is 5 times faster
    than sequential calculation.

    Args:
        x (list[Tensor]): The input tensors which is a list Tensor.
        name (str|None, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: The output Tensor.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> # A * B
            >>> A = paddle.rand([3, 4])
            >>> B = paddle.rand([4, 5])
            >>> out = paddle.linalg.multi_dot([A, B])
            >>> print(out.shape)
            [3, 5]

            >>> # A * B * C
            >>> A = paddle.rand([10, 5])
            >>> B = paddle.rand([5, 8])
            >>> C = paddle.rand([8, 7])
            >>> out = paddle.linalg.multi_dot([A, B, C])
            >>> print(out.shape)
            [10, 7]

    """
    if in_dynamic_or_pir_mode():
        return _C_ops.multi_dot(x)
    else:
        check_type(x, 'x', (list, tuple), 'multi_dot')
        for id, item in enumerate(x):
            check_variable_and_dtype(
                item,
                'x[' + str(id) + ']',
                ['float16', 'float32', 'float64', 'uint16'],
                'multi_dot',
            )
            if item.dtype != x[0].dtype:
                raise TypeError(
                    "All the Tensors in the input must have the same data type."
                )

        helper = LayerHelper('multi_dot', **locals())
        dtype = helper.input_dtype(input_param_name='x')
        out = helper.create_variable_for_type_inference(dtype)
        helper.append_op(
            type='multi_dot', inputs={"X": x}, outputs={"Out": out}
        )
        return out


def eigh(
    x: Tensor, UPLO: Literal['L', 'U'] = 'L', name: str | None = None
) -> tuple[Tensor, Tensor]:
    """
    Compute the eigenvalues and eigenvectors of a
    complex Hermitian (conjugate symmetric) or a real symmetric matrix.

    Args:
        x (Tensor): A tensor with shape :math:`[*, N, N]` , The data type of the input Tensor x
            should be one of float32, float64, complex64, complex128.
        UPLO (str, optional): (string, default 'L'), 'L' represents the lower triangular matrix,
            "'U' represents the upper triangular matrix.". Default: 'L'.
        name (str|None, optional): The default value is None. Normally there is no need for user to set this
            property.  For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        2-element tuple containing

        - out_value(Tensor): A Tensor with shape :math:`[*, N]` and data type of float32 and float64.
          The eigenvalues of eigh op.
        - out_vector(Tensor): A Tensor with shape :math:`[*, N, N]` and data type of float32, float64,
          complex64 and complex128. The eigenvectors of eigh op.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([[1, -2j], [2j, 5]])
            >>> out_value, out_vector = paddle.linalg.eigh(x, UPLO='L')
            >>> print(out_value)
            Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [0.17157286, 5.82842731])
            >>> print(out_vector)
            Tensor(shape=[2, 2], dtype=complex64, place=Place(cpu), stop_gradient=True,
            [[(-0.9238795042037964+0j), (-0.3826833963394165+0j)],
             [ 0.3826833963394165j    , -0.9238795042037964j    ]])

    """
    if in_dynamic_mode():
        return _C_ops.eigh(x, UPLO)

    def __check_input(x, UPLO):
        x_shape = list(x.shape)
        if len(x.shape) < 2:
            raise ValueError(
                "Input(input) only support >=2 tensor, but received "
                f"length of Input(input) is {len(x.shape)}."
            )
        if x_shape[-1] != x_shape[-2]:
            raise ValueError(
                f"The input matrix must be batches of square matrices. But received x's dimension: {x_shape}"
            )
        if UPLO != 'L' and UPLO != 'U':
            raise ValueError(
                f"UPLO must be L or U. But received UPLO is: {UPLO}"
            )

    if in_pir_mode():
        __check_input(x, UPLO)
        return _C_ops.eigh(x, UPLO)

    else:
        __check_input(x, UPLO)

        helper = LayerHelper('eigh', **locals())
        check_variable_and_dtype(
            x,
            'dtype',
            ['float32', 'float64', 'complex64', 'complex128'],
            'eigh',
        )

        out_value = helper.create_variable_for_type_inference(dtype=x.dtype)
        out_vector = helper.create_variable_for_type_inference(dtype=x.dtype)

        helper.append_op(
            type='eigh',
            inputs={'X': x},
            outputs={'Eigenvalues': out_value, 'Eigenvectors': out_vector},
            attrs={'UPLO': UPLO},
        )
        return out_value, out_vector


def pinv(
    x: Tensor,
    rcond: float | Tensor = 1e-15,
    hermitian: bool = False,
    name: str | None = None,
) -> Tensor:
    r"""
    Calculate pseudo inverse via SVD(singular value decomposition)
    of one matrix or batches of regular matrix.

    .. math::

        if hermitian == False:
            x = u * s * vt  (SVD)
            out = v * 1/s * ut
        else:
            x = u * s * ut  (eigh)
            out = u * 1/s * u.conj().transpose(-2,-1)

    If x is hermitian or symmetric matrix, svd will be replaced with eigh.

    Args:
        x (Tensor): The input tensor. Its shape should be (*, m, n)
            where * is zero or more batch dimensions. m and n can be
            arbitrary positive number. The data type of x should be
            float32 or float64 or complex64 or complex128. When data
            type is complex64 or complex128, hermitian should be set
            True.
        rcond (Tensor|float, optional): the tolerance value to determine
            when is a singular value zero. Default:1e-15.
        hermitian (bool, optional): indicates whether x is Hermitian
            if complex or symmetric if real. Default: False.
        name (str|None, optional): The default value is None. Normally there is no need for user to set this
            property. For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: The tensor with same data type with x. it represents
        pseudo inverse of x. Its shape should be (*, n, m).

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.arange(15).reshape((3, 5)).astype('float64')
            >>> input = paddle.to_tensor(x)
            >>> out = paddle.linalg.pinv(input)
            >>> print(input)
            Tensor(shape=[3, 5], dtype=float64, place=Place(cpu), stop_gradient=True,
            [[0. , 1. , 2. , 3. , 4. ],
             [5. , 6. , 7. , 8. , 9. ],
             [10., 11., 12., 13., 14.]])

            >>> print(out)
            Tensor(shape=[5, 3], dtype=float64, place=Place(cpu), stop_gradient=True,
            [[-0.22666667, -0.06666667,  0.09333333],
             [-0.12333333, -0.03333333,  0.05666667],
             [-0.02000000, -0.00000000,  0.02000000],
             [ 0.08333333,  0.03333333, -0.01666667],
             [ 0.18666667,  0.06666667, -0.05333333]])

            # one can verify : x * out * x = x ;
            # or              out * x * out = x ;
    """
    if in_dynamic_or_pir_mode():
        if not hermitian:
            # combine svd and matmul op
            u, s, vt = _C_ops.svd(x, False)
            max_singular_val = _C_ops.max(s, [-1], True)
            rcond = paddle.to_tensor(rcond, dtype=x.dtype)
            cutoff = rcond * max_singular_val
            y = float('inf')
            y = paddle.to_tensor(y, dtype=x.dtype)

            singular = paddle.where(s > cutoff, 1 / s, 1 / y)
            st = _C_ops.unsqueeze(singular, [-2])

            dims = list(range(len(vt.shape)))
            perm = dims[:-2] + [dims[-1]] + [dims[-2]]
            v = _C_ops.transpose(vt, perm)

            out_1 = v * st
            out_2 = _C_ops.matmul(out_1, u, False, True)
            return out_2
        else:
            # combine eigh and matmul op
            s, u = _C_ops.eigh(x, 'UPLO')
            s_abs = paddle.abs(s)
            max_singular_val = _C_ops.max(s_abs, [-1], True)
            rcond = paddle.to_tensor(rcond, dtype=s.dtype)
            cutoff = rcond * max_singular_val
            y = float('inf')
            y = paddle.to_tensor(y, dtype=s.dtype)

            singular = paddle.where(s_abs > cutoff, 1 / s, 1 / y)
            st = _C_ops.unsqueeze(singular, [-2])

            out_1 = u * st
            u_conj = _C_ops.conj(u)
            out_2 = _C_ops.matmul(out_1, u_conj, False, True)
            return out_2
    else:
        if not hermitian:
            helper = LayerHelper('pinv', **locals())
            dtype = x.dtype
            check_variable_and_dtype(x, 'x', ['float32', 'float64'], 'pinv')

            u = helper.create_variable_for_type_inference(dtype)
            s = helper.create_variable_for_type_inference(dtype)
            vt = helper.create_variable_for_type_inference(dtype)
            helper.append_op(
                type='svd',
                inputs={'X': [x]},
                outputs={'U': u, 'VH': vt, 'S': s},
                attrs={'full_matrices': False},
            )

            max_singular_val = helper.create_variable_for_type_inference(dtype)
            helper.append_op(
                type='reduce_max',
                inputs={'X': s},
                outputs={'Out': max_singular_val},
                attrs={'dim': [-1], 'keep_dim': True, 'reduce_all': False},
            )

            rcond = full(shape=[1], fill_value=rcond, dtype=dtype)
            cutoff = rcond * max_singular_val
            y = float('inf')
            y = full(shape=[1], fill_value=y, dtype=dtype)

            singular = paddle.where(s > cutoff, 1 / s, 1 / y)

            st = helper.create_variable_for_type_inference(dtype=dtype)
            st_shape = helper.create_variable_for_type_inference(dtype=dtype)
            helper.append_op(
                type='unsqueeze2',
                inputs={'X': singular},
                attrs={'axes': [-2]},
                outputs={'Out': st, 'XShape': st_shape},
            )

            dims = list(range(len(vt.shape)))
            perm = dims[:-2] + [dims[-1]] + [dims[-2]]
            v = helper.create_variable_for_type_inference(dtype)
            v_shape = helper.create_variable_for_type_inference(dtype)
            helper.append_op(
                type='transpose2',
                inputs={'X': [vt]},
                outputs={'Out': [v], 'XShape': [v_shape]},
                attrs={'axis': perm},
            )

            out_1 = helper.create_variable_for_type_inference(dtype)
            helper.append_op(
                type='elementwise_mul',
                inputs={'X': v, 'Y': st},
                outputs={'Out': out_1},
                attrs={'axis': -1},
            )
            out_1 = helper.append_activation(out_1)

            out_2 = helper.create_variable_for_type_inference(dtype)
            helper.append_op(
                type='matmul_v2',
                inputs={'X': out_1, 'Y': u},
                outputs={'Out': out_2},
                attrs={'trans_x': False, 'trans_y': True},
            )
            return out_2
        else:
            helper = LayerHelper('pinv', **locals())
            dtype = x.dtype
            check_variable_and_dtype(
                x,
                'dtype',
                ['float32', 'float64', 'complex64', 'complex128'],
                'pinv',
            )

            if dtype == paddle.complex128:
                s_type = 'float64'
            elif dtype == paddle.complex64:
                s_type = 'float32'
            else:
                s_type = dtype

            u = helper.create_variable_for_type_inference(dtype)
            s = helper.create_variable_for_type_inference(s_type)
            helper.append_op(
                type='eigh',
                inputs={'X': x},
                outputs={'Eigenvalues': s, 'Eigenvectors': u},
                attrs={'UPLO': 'L'},
            )
            s_abs = helper.create_variable_for_type_inference(s_type)
            helper.append_op(
                type='abs', inputs={'X': s}, outputs={'Out': s_abs}
            )
            max_singular_val = helper.create_variable_for_type_inference(s_type)
            helper.append_op(
                type='reduce_max',
                inputs={'X': s_abs},
                outputs={'Out': max_singular_val},
                attrs={'dim': [-1], 'keep_dim': True, 'reduce_all': False},
            )

            rcond = full(shape=[1], fill_value=rcond, dtype=s_type)
            cutoff = rcond * max_singular_val
            y = float('inf')
            y = full(shape=[1], fill_value=y, dtype=s_type)

            singular = paddle.where(s_abs > cutoff, 1 / s, 1 / y)

            st = helper.create_variable_for_type_inference(dtype=s_type)
            st_shape = helper.create_variable_for_type_inference(dtype=s_type)
            helper.append_op(
                type='unsqueeze2',
                inputs={'X': singular},
                attrs={'axes': [-2]},
                outputs={'Out': st, 'XShape': st_shape},
            )

            out_1 = helper.create_variable_for_type_inference(dtype)
            helper.append_op(
                type='elementwise_mul',
                inputs={'X': u, 'Y': st},
                outputs={'Out': out_1},
                attrs={'axis': -1},
            )
            out_1 = helper.append_activation(out_1)

            u_conj = helper.create_variable_for_type_inference(dtype)
            helper.append_op(
                type='conj', inputs={'X': u}, outputs={'Out': [u_conj]}
            )

            out_2 = helper.create_variable_for_type_inference(dtype)
            helper.append_op(
                type='matmul_v2',
                inputs={'X': out_1, 'Y': u_conj},
                outputs={'Out': out_2},
                attrs={'trans_x': False, 'trans_y': True},
            )
            return out_2


def _check_right_solve_shape(x, y):
    """check the input shape of x and y for solve when left is False"""
    x_shape = x.shape[-2:]
    if len(y.shape) == 1:
        raise ValueError(
            "Incompatible shapes of X and Y for the equation Out * X = Y, "
            f"where input X's matrix shape is {x_shape} and"
            f"input Y's matrix shape is {list(y.shape).append(1)}"
        )
    else:
        y_shape = y.shape[-2:]
        if x_shape[0] != y_shape[1]:
            raise ValueError(
                "Incompatible shapes of X and Y for the equation Out * X = Y, "
                f"where input X's matrix shape is {x_shape} and"
                f"input Y's matrix shape is {y_shape}"
            )


def _transpose_last_2dim(x):
    """transpose the last 2 dimension of a tensor"""
    x_new_dims = list(range(len(x.shape)))
    x_new_dims[-1], x_new_dims[-2] = x_new_dims[-2], x_new_dims[-1]
    x = transpose(x, x_new_dims)
    return x


def solve(
    x: Tensor, y: Tensor, left: bool = True, name: str | None = None
) -> Tensor:
    r"""

    Computes the solution of a square system of linear equations with a unique solution for input 'X' and 'Y'.
    Let :math:`X` be a square matrix or a batch of square matrices, :math:`Y` be
    a vector/matrix or a batch of vectors/matrices. When `left` is True, the equation should be:

    .. math::
        Out = X^-1 * Y

    When `left` is False, the equation should be:

    .. math::
        Out = Y * X^-1

    Specifically, this system of linear equations has one solution if and only if input 'X' is invertible.

    Args:
        x (Tensor): A square matrix or a batch of square matrices. Its shape should be ``[*, M, M]``, where ``*`` is zero or
            more batch dimensions. Its data type should be float32 or float64.
        y (Tensor): A vector/matrix or a batch of vectors/matrices. Its shape should be ``[*, M, K]``, where ``*`` is zero or
            more batch dimensions. Its data type should be float32 or float64.
        left (bool, optional): Whether to solve the system :math:`X * Out = Y` or :math:`Out * X = Y`. Default: True.
        name (str|None, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: The solution of a square system of linear equations with a unique solution for input 'x' and 'y'.
        Its data type should be the same as that of `x`.

    Examples:

        .. code-block:: python

            >>> # a square system of linear equations:
            >>> # 3*X0 + X1 = 9
            >>> # X0 + 2*X1 = 8

            >>> import paddle

            >>> x = paddle.to_tensor([[3, 1],[1, 2]], dtype="float64")
            >>> y = paddle.to_tensor([9, 8], dtype="float64")
            >>> out = paddle.linalg.solve(x, y)

            >>> print(out)
            Tensor(shape=[2], dtype=float64, place=Place(cpu), stop_gradient=True,
            [2., 3.])
    """
    if not left:
        _check_right_solve_shape(x, y)
        x = _transpose_last_2dim(x)
        y = _transpose_last_2dim(y)

    if in_dynamic_or_pir_mode():
        out = _C_ops.solve(x, y)
    else:
        inputs = {"X": [x], "Y": [y]}
        helper = LayerHelper("solve", **locals())
        check_variable_and_dtype(x, 'x', ['float32', 'float64'], 'solve')
        check_variable_and_dtype(y, 'y', ['float32', 'float64'], 'solve')
        out = helper.create_variable_for_type_inference(dtype=x.dtype)

        helper.append_op(
            type="solve", inputs={"X": x, "Y": y}, outputs={"Out": out}
        )

    if not left:
        out = _transpose_last_2dim(out)
    return out


def triangular_solve(
    x: Tensor,
    y: Tensor,
    upper: bool = True,
    transpose: bool = False,
    unitriangular: bool = False,
    name: str | None = None,
) -> Tensor:
    r"""
    Computes the solution of a system of equations with a triangular coefficient. `x` is coefficient matrix
    `y` is multiple right-hand sides of equations.

    Input `x` and `y` is 2D matrices or batches of 2D matrices. If the inputs are batches, the outputs is also
    batches.

    Equations can be described as:

    .. math::
        x * Out = y

    Solution of Equations is:

    .. math::
        Out = x ^ {-1} * y

    Args:
        x (Tensor): The input triangular coefficient matrix. Its shape should be `[*, M, M]`, where `*` is zero or
            more batch dimensions. Its data type should be float32, float64, complex64, complex128.
        y (Tensor): Multiple right-hand sides of system of equations. Its shape should be `[*, M, K]`, where `*` is
            zero or more batch dimensions. Its data type should be float32, float64, complex64, complex128.
        upper (bool, optional): Whether to solve the upper-triangular system of equations (default) or the lower-triangular
            system of equations. Default: True.
        transpose (bool, optional): whether `x` should be transposed before calculation. Default: False.
        unitriangular (bool, optional): whether `x` is unit triangular. If True, the diagonal elements of `x` are assumed
            to be 1 and not referenced from `x` . Default: False.
        name (str|None, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: The solution of the system of equations. Its data type should be the same as that of `x`.

    Examples:
        .. code-block:: python

            >>> # a square system of linear equations:
            >>> # x1 +   x2  +   x3 = 0
            >>> #      2*x2  +   x3 = -9
            >>> #               -x3 = 5

            >>> import paddle
            >>> x = paddle.to_tensor([[1, 1, 1],
            ...                       [0, 2, 1],
            ...                       [0, 0,-1]], dtype="float64")
            >>> y = paddle.to_tensor([[0], [-9], [5]], dtype="float64")
            >>> out = paddle.linalg.triangular_solve(x, y, upper=True)

            >>> print(out)
            Tensor(shape=[3, 1], dtype=float64, place=Place(cpu), stop_gradient=True,
            [[ 7.],
             [-2.],
             [-5.]])
    """
    if in_dynamic_or_pir_mode():
        return _C_ops.triangular_solve(x, y, upper, transpose, unitriangular)
    else:
        inputs = {"X": [x], "Y": [y]}
        helper = LayerHelper("triangular_solve", **locals())
        check_variable_and_dtype(
            x,
            'x',
            ['float32', 'float64', 'complex64', 'complex128'],
            'triangular_solve',
        )
        check_variable_and_dtype(
            y,
            'y',
            ['float32', 'float64', 'complex64', 'complex128'],
            'triangular_solve',
        )
        out = helper.create_variable_for_type_inference(dtype=x.dtype)

        helper.append_op(
            type='triangular_solve',
            inputs={'X': x, 'Y': y},
            outputs={'Out': out},
            attrs={
                'upper': upper,
                'transpose': transpose,
                'unitriangular': unitriangular,
            },
        )
        return out


def cholesky_solve(
    x: Tensor, y: Tensor, upper: bool = False, name: str | None = None
) -> Tensor:
    r"""
    Solves a linear system of equations A @ X = B, given A's Cholesky factor matrix u and  matrix B.

    Input `x` and `y` is 2D matrices or batches of 2D matrices. If the inputs are batches, the outputs
    is also batches.

    Args:
        x (Tensor): Multiple right-hand sides of system of equations. Its shape should be `[*, M, K]`, where `*` is
            zero or more batch dimensions. Its data type should be float32 or float64.
        y (Tensor): The input matrix which is upper or lower triangular Cholesky factor of square matrix A. Its shape should be `[*, M, M]`, where `*` is zero or
            more batch dimensions. Its data type should be float32 or float64.
        upper (bool, optional): whether to consider the Cholesky factor as a lower or upper triangular matrix. Default: False.
        name (str|None, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: The solution of the system of equations. Its data type is the same as that of `x`.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> u = paddle.to_tensor([[1, 1, 1],
            ...                       [0, 2, 1],
            ...                       [0, 0,-1]], dtype="float64")
            >>> b = paddle.to_tensor([[0], [-9], [5]], dtype="float64")
            >>> out = paddle.linalg.cholesky_solve(b, u, upper=True)

            >>> print(out)
            Tensor(shape=[3, 1], dtype=float64, place=Place(cpu), stop_gradient=True,
            [[-2.50000000],
             [-7.        ],
             [ 9.50000000]])
    """
    if in_dynamic_or_pir_mode():
        return _C_ops.cholesky_solve(x, y, upper)
    else:
        helper = LayerHelper("cholesky_solve", **locals())
        check_variable_and_dtype(
            x, 'x', ['float32', 'float64'], 'cholesky_solve'
        )
        check_variable_and_dtype(
            y, 'y', ['float32', 'float64'], 'cholesky_solve'
        )
        out = helper.create_variable_for_type_inference(dtype=x.dtype)

        helper.append_op(
            type='cholesky_solve',
            inputs={'X': x, 'Y': y},
            outputs={'Out': out},
            attrs={'upper': upper},
        )
        return out


def eigvalsh(
    x: Tensor, UPLO: Literal['L', 'U'] = 'L', name: str | None = None
) -> Tensor:
    """
    Computes the eigenvalues of a
    complex Hermitian (conjugate symmetric) or a real symmetric matrix.

    Args:
        x (Tensor): A tensor with shape :math:`[*, M, M]` , where * is zero or greater batch dimension. The data type of the input Tensor x
            should be one of float32, float64, complex64, complex128.
        UPLO(str, optional): Lower triangular part of a ('L', default) or the upper triangular part ('U').
        name(str|None, optional): The default value is None.  Normally there is no need for user to set this
            property.  For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: The tensor eigenvalues in ascending order.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([[1, -2j], [2j, 5]])
            >>> out_value = paddle.eigvalsh(x, UPLO='L')
            >>> print(out_value)
            Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [0.17157286, 5.82842731])
    """
    if in_dynamic_mode():
        values, _ = _C_ops.eigvalsh(x, UPLO, x.stop_gradient)
        return values

    def __check_input(x, UPLO):
        x_shape = list(x.shape)
        if len(x.shape) < 2:
            raise ValueError(
                "Input(input) only support >=2 tensor, but received "
                f"length of Input(input) is {len(x.shape)}."
            )
        if x_shape[-1] != x_shape[-2]:
            raise ValueError(
                f"The input matrix must be batches of square matrices. But received x's dimension: {x_shape}"
            )
        if UPLO != 'L' and UPLO != 'U':
            raise ValueError(
                f"UPLO must be L or U. But received UPLO is: {UPLO}"
            )

    if in_pir_mode():
        __check_input(x, UPLO)
        values, _ = _C_ops.eigvalsh(x, UPLO, x.stop_gradient)
        return values

    else:
        __check_input(x, UPLO)

        helper = LayerHelper('eigvalsh', **locals())
        check_variable_and_dtype(
            x,
            'dtype',
            ['float32', 'float64', 'complex64', 'complex128'],
            'eigvalsh',
        )

        out_value = helper.create_variable_for_type_inference(dtype=x.dtype)
        out_vector = helper.create_variable_for_type_inference(dtype=x.dtype)

        is_test = x.stop_gradient
        helper.append_op(
            type='eigvalsh',
            inputs={'X': x},
            outputs={'Eigenvalues': out_value, 'Eigenvectors': out_vector},
            attrs={'UPLO': UPLO, 'is_test': is_test},
        )
        return out_value


def lstsq(
    x: Tensor,
    y: Tensor,
    rcond: float | None = None,
    driver: Literal['gels', 'gelsy', 'gelsd', 'gelss'] | None = None,
    name: str | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Computes a solution to
    the least squares problem of a system of linear equations.

    Args:
        x (Tensor): A tensor with shape ``(*, M, N)`` , the data type of the input Tensor ``x``
            should be one of float32, float64.
        y (Tensor): A tensor with shape ``(*, M, K)`` , the data type of the input Tensor ``y``
            should be one of float32, float64.
        rcond(float, optional): The default value is None. A float pointing number used to determine
            the effective rank of ``x``. If ``rcond`` is None, it will be set to max(M, N) times the
            machine precision of x_dtype.
        driver(str, optional): The default value is None. The name of LAPACK method to be used. For
            CPU inputs the valid values are 'gels', 'gelsy', 'gelsd, 'gelss'. For CUDA input, the only
            valid driver is 'gels'. If ``driver`` is None, 'gelsy' is used for CPU inputs and 'gels'
            for CUDA inputs.
        name(str, optional): The default value is None. Normally there is no need for user to set
            this property. For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tuple: A tuple of 4 Tensors which is (``solution``, ``residuals``, ``rank``, ``singular_values``).
        ``solution`` is a tensor with shape ``(*, N, K)``, meaning the least squares solution. ``residuals``
        is a tensor with shape ``(*, K)``, meaning the squared residuals of the solutions, which is computed
        when M > N and every matrix in ``x`` is full-rank, otherwise return an empty tensor. ``rank`` is a tensor
        with shape ``(*)``, meaning the ranks of the matrices in ``x``, which is computed when ``driver`` in
        ('gelsy', 'gelsd', 'gelss'), otherwise return an empty tensor. ``singular_values`` is a tensor with
        shape ``(*, min(M, N))``, meaning singular values of the matrices in ``x``, which is computed when
        ``driver`` in ('gelsd', 'gelss'), otherwise return an empty tensor.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([[1, 3], [3, 2], [5, 6.]])
            >>> y = paddle.to_tensor([[3, 4, 6], [5, 3, 4], [1, 2, 1.]])
            >>> results = paddle.linalg.lstsq(x, y, driver="gelsd")
            >>> print(results[0])
            Tensor(shape=[2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[ 0.78350395, -0.22165027, -0.62371236],
             [-0.11340097,  0.78866047,  1.14948535]])
            >>> print(results[1])
            Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [19.81443405, 10.43814468, 30.56185532])
            >>> print(results[2])
            Tensor(shape=[], dtype=int32, place=Place(cpu), stop_gradient=True,
            2)
            >>> print(results[3])
            Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [9.03455734, 1.54167950])

            >>> x = paddle.to_tensor([[10, 2, 3], [3, 10, 5], [5, 6, 12.]])
            >>> y = paddle.to_tensor([[4, 2, 9], [2, 0, 3], [2, 5, 3.]])
            >>> results = paddle.linalg.lstsq(x, y, driver="gels")
            >>> print(results[0])
            Tensor(shape=[3, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[ 0.39386186,  0.10230169,  0.93606132],
             [ 0.10741688, -0.29028130,  0.11892584],
             [-0.05115093,  0.51918161, -0.19948851]])
            >>> print(results[1])
            Tensor(shape=[0], dtype=float32, place=Place(cpu), stop_gradient=True,
            [])
    """
    device = paddle.get_device()
    if device == "cpu":
        if driver not in (None, "gels", "gelss", "gelsd", "gelsy"):
            raise ValueError(
                f"Only support valid driver is 'gels', 'gelss', 'gelsd', 'gelsy' or None for CPU inputs. But got {driver}"
            )
        driver = "gelsy" if driver is None else driver
    elif "gpu" in device:
        if driver not in (None, "gels"):
            raise ValueError(
                f"Only support valid driver is 'gels' or None for CUDA inputs. But got {driver}"
            )
        driver = "gels" if driver is None else driver
    else:
        raise RuntimeError("Only support lstsq api for CPU or CUDA device.")

    if not (
        x.dtype == y.dtype
        and x.dtype
        in (
            paddle.float32,
            paddle.float64,
            paddle.base.core.DataType.FLOAT32,
            paddle.base.core.DataType.FLOAT64,
        )
    ):
        raise ValueError(
            "Only support x and y have the same dtype such as 'float32' and 'float64'."
        )

    if x.ndim < 2:
        raise ValueError(
            f"The shape of x should be (*, M, N), but received ndim is [{x.ndim} < 2]"
        )

    if y.ndim < 2:
        raise ValueError(
            f"The shape of y should be (*, M, K), but received ndim is [{y.ndim} < 2]"
        )

    if x.shape[-2] != y.shape[-2]:
        raise ValueError(
            f"x with shape (*, M = {x.shape[-2]}, N) and y with shape (*, M = {y.shape[-2]}, K) should have same M."
        )

    if rcond is None:
        if (
            x.dtype == paddle.float32
            or x.dtype == paddle.base.core.DataType.FLOAT32
        ):
            rcond = 1e-7 * max(x.shape[-2], x.shape[-1])
        elif (
            x.dtype == paddle.float64
            or x.dtype == paddle.base.core.DataType.FLOAT64
        ):
            rcond = 1e-15 * max(x.shape[-2], x.shape[-1])

    if in_dynamic_or_pir_mode():
        solution, residuals, rank, singular_values = _C_ops.lstsq(
            x, y, rcond, driver
        )
        if driver == "gels":
            rank = paddle.empty(shape=[0], dtype="int32")
            singular_values = paddle.empty(shape=[0], dtype=x.dtype)
        elif driver == "gelsy":
            singular_values = paddle.empty(shape=[0], dtype=x.dtype)

        return solution, residuals, rank, singular_values
    else:
        helper = LayerHelper('lstsq', **locals())
        check_variable_and_dtype(
            x,
            'dtype',
            ['float32', 'float64', 'complex64', 'complex128'],
            'lstsq',
        )
        check_variable_and_dtype(
            y,
            'dtype',
            ['float32', 'float64', 'complex64', 'complex128'],
            'lstsq',
        )

        solution = helper.create_variable_for_type_inference(dtype=x.dtype)
        residuals = helper.create_variable_for_type_inference(dtype=x.dtype)
        rank = helper.create_variable_for_type_inference(dtype=paddle.int32)
        singular_values = helper.create_variable_for_type_inference(
            dtype=x.dtype
        )

        helper.append_op(
            type='lstsq',
            inputs={'X': x, 'Y': y},
            outputs={
                'Solution': solution,
                'Residuals': residuals,
                'Rank': rank,
                'SingularValues': singular_values,
            },
            attrs={'rcond': rcond, 'driver': driver},
        )

        if driver == "gels":
            rank = paddle.static.data(name='rank', shape=[0])
            singular_values = paddle.static.data(
                name='singular_values', shape=[0]
            )
        elif driver == "gelsy":
            singular_values = paddle.static.data(
                name='singular_values', shape=[0]
            )

        return solution, residuals, rank, singular_values


def corrcoef(x: Tensor, rowvar: bool = True, name: str | None = None) -> Tensor:
    """

    A correlation coefficient matrix indicate the correlation of each pair variables in the input matrix.
    For example, for an N-dimensional samples X=[x1,x2,…xN]T, then the correlation coefficient matrix
    element Rij is the correlation of xi and xj. The element Rii is the covariance of xi itself.

    The relationship between the correlation coefficient matrix `R` and the
    covariance matrix `C`, is

    .. math:: R_{ij} = \\frac{ C_{ij} } { \\sqrt{ C_{ii} * C_{jj} } }

    The values of `R` are between -1 and 1.

    Args:

        x (Tensor): A N-D(N<=2) Tensor containing multiple variables and observations. By default, each row of x represents a variable. Also see rowvar below.
        rowvar (bool, optional): If rowvar is True (default), then each row represents a variable, with observations in the columns. Default: True.
        name (str|None, optional): Name of the output. It's used to print debug info for developers. Details: :ref:`api_guide_Name`. Default: None.

    Returns:

        The correlation coefficient matrix of the variables.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.seed(2023)

            >>> xt = paddle.rand((3,4))
            >>> print(paddle.linalg.corrcoef(xt))
            Tensor(shape=[3, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[ 0.99999988, -0.47689581, -0.89559376],
             [-0.47689593,  1.        ,  0.16345492],
             [-0.89559382,  0.16345496,  1.        ]])

    """
    if len(x.shape) > 2 or len(x.shape) < 1:
        raise ValueError(
            "Input(x) only support N-D (1<=N<=2) tensor in corrcoef, but received "
            f"length of Input(input) is {len(x.shape)}."
        )
    check_variable_and_dtype(x, 'dtype', ['float32', 'float64'], 'corrcoef')

    c = cov(x, rowvar)
    if c.ndim == 0:
        # scalar covariance
        # nan if incorrect value (nan, inf, 0), 1 otherwise
        return c / c

    d = paddle.diag(c)

    if paddle.is_complex(d):
        d = d.real()
    stddev = paddle.sqrt(d)
    c /= stddev[:, None]
    c /= stddev[None, :]

    # Clip to [-1, 1].  This does not guarantee
    if paddle.is_complex(c):
        return paddle.complex(
            paddle.clip(c.real(), -1, 1), paddle.clip(c.imag(), -1, 1)
        )
    else:
        c = paddle.clip(c, -1, 1)

    return c


def cdist(
    x: Tensor,
    y: Tensor,
    p: float = 2.0,
    compute_mode: Literal[
        'use_mm_for_euclid_dist_if_necessary',
        'use_mm_for_euclid_dist',
        'donot_use_mm_for_euclid_dist',
    ] = "use_mm_for_euclid_dist_if_necessary",
    name: str | None = None,
) -> Tensor:
    r"""

    Compute the p-norm distance between each pair of the two collections of inputs.

    This function is equivalent to `scipy.spatial.distance.cdist(input,'minkowski', p=p)`
    if :math:`p \in (0, \infty)`. When :math:`p = 0` it is equivalent to `scipy.spatial.distance.cdist(input, 'hamming') * M`.
    When :math:`p = \infty`, the closest scipy function is `scipy.spatial.distance.cdist(xn, lambda x, y: np.abs(x - y).max())`.

    Args:
        x (Tensor): A tensor with shape :math:`B \times P \times M`.
        y (Tensor): A tensor with shape :math:`B \times R \times M`.
        p (float, optional): The value for the p-norm distance to calculate between each vector pair. Default: :math:`2.0`.
        compute_mode (str, optional): The mode for compute distance.

            - ``use_mm_for_euclid_dist_if_necessary`` , for p = 2.0 and (P > 25 or R > 25), it will use matrix multiplication to calculate euclid distance if possible.
            - ``use_mm_for_euclid_dist`` , for p = 2.0, it will use matrix multiplication to calculate euclid distance.
            - ``donot_use_mm_for_euclid_dist`` , it will not use matrix multiplication to calculate euclid distance.

            Default: ``use_mm_for_euclid_dist_if_necessary``.
        name (str|None, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        Tensor, the dtype is same as input tensor.

        If x has shape :math:`B \times P \times M` and y has shape :math:`B \times R \times M` then
        the output will have shape :math:`B \times P \times R`.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> x = paddle.to_tensor([[0.9041,  0.0196], [-0.3108, -2.4423], [-0.4821,  1.059]], dtype=paddle.float32)
            >>> y = paddle.to_tensor([[-2.1763, -0.4713], [-0.6986,  1.3702]], dtype=paddle.float32)
            >>> distance = paddle.cdist(x, y)
            >>> print(distance)
            Tensor(shape=[3, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[3.11927032, 2.09589314],
             [2.71384072, 3.83217239],
             [2.28300953, 0.37910119]])
    """

    check_variable_and_dtype(x, 'x', ('float32', 'float64'), 'cdist')
    check_variable_and_dtype(y, 'y', ('float32', 'float64'), 'cdist')
    check_type(p, 'p', (float, int), 'cdist')

    if compute_mode not in [
        'use_mm_for_euclid_dist_if_necessary',
        'use_mm_for_euclid_dist',
        'donot_use_mm_for_euclid_dist',
    ]:
        raise ValueError(
            "The compute_mode should be 'use_mm_for_euclid_dist_if_necessary', "
            "'use_mm_for_euclid_dist' or 'donot_use_mm_for_euclid_dist', "
            f"but received compute_mode is {compute_mode}."
        )

    mode = 0
    if compute_mode == 'use_mm_for_euclid_dist_if_necessary':
        mode = 0
    elif compute_mode == 'use_mm_for_euclid_dist':
        mode = 1
    elif compute_mode == 'donot_use_mm_for_euclid_dist':
        mode = 2

    x_shape = list(x.shape)
    assert len(x_shape) >= 2, (
        "The x must be at least 2-dimensional, "
        f"But received Input x's dimensional is {len(x_shape)}.\n"
    )
    y_shape = list(y.shape)
    assert len(y_shape) >= 2, (
        "The y must be at least 2-dimensional, "
        f"But received Input y's dimensional is {len(y_shape)}.\n"
    )
    assert x_shape[-1] == y_shape[-1], (
        "The x and y must have same last dimension, "
        f"But received Input x's last dimension is {x_shape[-1]}, "
        f"Input y's last dimension is {y_shape[-1]}.\n"
    )
    assert p >= 0, (
        "The p must be greater than or equal to 0, " f"But received p is {p}.\n"
    )

    r1 = x.shape[-2]
    r2 = y.shape[-2]
    c1 = x.shape[-1]

    p = float(p)

    if r1 == 0 or r2 == 0:
        return paddle.empty((r1, r2), dtype=x.dtype)

    if c1 == 0:
        return paddle.zeros((r1, r2), dtype=x.dtype)

    if p == 2.0 and (mode == 1 or (mode == 0 and (r1 > 25 or r2 > 25))):
        x_norm = paddle.sum(x.pow(2), axis=-1, keepdim=True)
        y_norm = paddle.sum(y.pow(2), axis=-1, keepdim=True)
        y_transposed = paddle.transpose(
            y, perm=[*range(y.ndim - 2), y.ndim - 1, y.ndim - 2]
        )
        y_norm_transposed = paddle.transpose(
            y_norm,
            perm=[*range(y_norm.ndim - 2), y_norm.ndim - 1, y_norm.ndim - 2],
        )
        res = paddle.matmul(x, y_transposed) * -2 + y_norm_transposed + x_norm
        res = paddle.clip(res, min=0.0).sqrt()
        return res

    return paddle.linalg.norm(
        x[..., None, :] - y[..., None, :, :], p=p, axis=-1
    )


def householder_product(
    x: Tensor, tau: Tensor, name: str | None = None
) -> Tensor:
    r"""

    Computes the first n columns of a product of Householder matrices.

    This function can get the vector :math:`\omega_{i}` from matrix `x` (m x n), the :math:`i-1` elements are zeros, and the i-th is `1`, the rest of the elements are from i-th column of `x`.
    And with the vector `tau` can calculate the first n columns of a product of Householder matrices.

    :math:`H_i = I_m - \tau_i \omega_i \omega_i^H`

    Args:
        x (Tensor): A tensor with shape (*, m, n) where * is zero or more batch dimensions.
        tau (Tensor): A tensor with shape (*, k) where * is zero or more batch dimensions.
        name (str|None, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        Tensor, the dtype is same as input tensor, the Q in QR decomposition.

        :math:`out = Q = H_1H_2H_3...H_k`

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> x = paddle.to_tensor([[-1.1280,  0.9012, -0.0190],
            ...         [ 0.3699,  2.2133, -1.4792],
            ...         [ 0.0308,  0.3361, -3.1761],
            ...         [-0.0726,  0.8245, -0.3812]])
            >>> tau = paddle.to_tensor([1.7497, 1.1156, 1.7462])
            >>> Q = paddle.linalg.householder_product(x, tau)
            >>> print(Q)
            Tensor(shape=[4, 3], dtype=float32, place=Place(gpu:0), stop_gradient=True,
                   [[-0.74969995, -0.02181768,  0.31115776],
                    [-0.64721400, -0.12367040, -0.21738708],
                    [-0.05389076, -0.37562513, -0.84836429],
                    [ 0.12702821, -0.91822827,  0.36892807]])
    """

    check_dtype(
        x.dtype,
        'x',
        [
            'float16',
            'float32',
            'float64',
            'complex64',
            'complex128',
        ],
        'householder_product',
    )
    check_dtype(
        tau.dtype,
        'tau',
        [
            'float16',
            'float32',
            'float64',
            'complex64',
            'complex128',
        ],
        'householder_product',
    )
    assert (
        x.dtype == tau.dtype
    ), "The input x must have the same dtype with input tau.\n"
    assert (
        len(x.shape) >= 2
        and len(tau.shape) >= 1
        and len(x.shape) == len(tau.shape) + 1
    ), (
        "The input x must have more than 2 dimensions, and input tau must have more than 1 dimension,"
        "and the dimension of x is 1 larger than the dimension of tau\n"
    )
    assert (
        x.shape[-2] >= x.shape[-1]
    ), "The rows of input x must be greater than or equal to the columns of input x.\n"
    assert (
        x.shape[-1] >= tau.shape[-1]
    ), "The last dim of x must be greater than tau.\n"
    for idx, _ in enumerate(x.shape[:-2]):
        assert (
            x.shape[idx] == tau.shape[idx]
        ), "The input x must have the same batch dimensions with input tau.\n"

    def _householder_product(x, tau):
        m, n = x.shape[-2:]
        k = tau.shape[-1]
        Q = paddle.eye(m).astype(x.dtype)
        for i in range(min(k, n)):
            w = x[i:, i]
            if in_dynamic_mode():
                w[0] = 1
            else:
                w = paddle.static.setitem(w, 0, 1)
            w = w.reshape([-1, 1])
            if in_dynamic_mode():
                if x.dtype in [paddle.complex128, paddle.complex64]:
                    Q[:, i:] = Q[:, i:] - (
                        Q[:, i:] @ w @ paddle.conj(w).T * tau[i]
                    )
                else:
                    Q[:, i:] = Q[:, i:] - (Q[:, i:] @ w @ w.T * tau[i])
            else:
                Q = paddle.static.setitem(
                    Q,
                    (slice(None), slice(i, None)),
                    (
                        Q[:, i:] - (Q[:, i:] @ w @ w.T * tau[i])
                        if x.dtype in [paddle.complex128, paddle.complex64]
                        else Q[:, i:] - (Q[:, i:] @ w @ w.T * tau[i])
                    ),
                )
        return Q[:, :n]

    if len(x.shape) == 2:
        return _householder_product(x, tau)
    m, n = x.shape[-2:]
    org_x_shape = x.shape
    org_tau_shape = tau.shape
    x = x.reshape((-1, org_x_shape[-2], org_x_shape[-1]))
    tau = tau.reshape((-1, org_tau_shape[-1]))
    n_batch = x.shape[0]
    out = paddle.zeros([n_batch, m, n], dtype=x.dtype)
    for i in range(n_batch):
        if in_dynamic_mode():
            out[i] = _householder_product(x[i], tau[i])
        else:
            out = paddle.static.setitem(
                out, i, _householder_product(x[i], tau[i])
            )
    out = out.reshape(org_x_shape)
    return out


# Reference: MatrixExponential, https://eigen.tuxfamily.org/dox/unsupported/MatrixExponential_8h_source.html
def _matrix_exp_pade3(mat_a, mat_i=None, mat_a2=None, *, dtype=None):
    """3rd-order Pade approximant."""
    b = [120.0, 60.0, 12.0]
    if not paddle.framework.in_dynamic_mode():
        b = [paddle.full((), x, dtype) for x in b]

    if mat_a2 is None:
        mat_a2, *_ = _matrix_mats(mat_a, 2, dtype)

    tmp = mat_a2 + b[1] * mat_i
    mat_u = paddle.matmul(mat_a, tmp)
    mat_v = b[2] * mat_a2 + b[0] * mat_i
    return mat_u, mat_v


def _matrix_exp_pade5(
    mat_a, mat_i=None, mat_a2=None, mat_a4=None, *, dtype=None
):
    """5th-order Pade approximant."""
    b = [30240.0, 15120.0, 3360.0, 420.0, 30.0]
    if not paddle.framework.in_dynamic_mode():
        b = [paddle.full((), x, dtype) for x in b]

    if mat_a4 is None:
        mat_a2, mat_a4, *_ = _matrix_mats(mat_a, 4, dtype)

    tmp = mat_a4 + b[3] * mat_a2 + b[1] * mat_i
    mat_u = paddle.matmul(mat_a, tmp)
    mat_v = b[4] * mat_a4 + b[2] * mat_a2 + b[0] * mat_i
    return mat_u, mat_v


def _matrix_exp_pade7(
    mat_a, mat_i=None, mat_a2=None, mat_a4=None, mat_a6=None, *, dtype=None
):
    """7th-order Pade approximant."""
    b = [17297280.0, 8648640.0, 1995840.0, 277200.0, 25200.0, 1512.0, 56.0]
    if not paddle.framework.in_dynamic_mode():
        b = [paddle.full((), x, dtype) for x in b]

    if mat_a6 is None:
        mat_a2, mat_a4, mat_a6, *_ = _matrix_mats(mat_a, 6, dtype)

    tmp = mat_a6 + b[5] * mat_a4 + b[3] * mat_a2 + b[1] * mat_i
    mat_u = paddle.matmul(mat_a, tmp)
    mat_v = b[6] * mat_a6 + b[4] * mat_a4 + b[2] * mat_a2 + b[0] * mat_i
    return mat_u, mat_v


def _matrix_exp_pade9(
    mat_a,
    mat_i=None,
    mat_a2=None,
    mat_a4=None,
    mat_a6=None,
    mat_a8=None,
    *,
    dtype=None,
):
    """9th-order Pade approximant."""
    b = [
        17643225600.0,
        8821612800.0,
        2075673600.0,
        302702400.0,
        30270240.0,
        2162160.0,
        110880.0,
        3960.0,
        90.0,
    ]
    if not paddle.framework.in_dynamic_mode():
        b = [paddle.full((), x, dtype) for x in b]

    if mat_a8 is None:
        mat_a2, mat_a4, mat_a6, mat_a8, *_ = _matrix_mats(mat_a, 8, dtype)

    tmp = mat_a8 + b[7] * mat_a6 + b[5] * mat_a4 + b[3] * mat_a2 + b[1] * mat_i
    mat_u = paddle.matmul(mat_a, tmp)
    mat_v = (
        b[8] * mat_a8
        + b[6] * mat_a6
        + b[4] * mat_a4
        + b[2] * mat_a2
        + b[0] * mat_i
    )
    return mat_u, mat_v


def _matrix_exp_pade13(
    mat_a, mat_i=None, mat_a2=None, mat_a4=None, mat_a6=None, *, dtype=None
):
    """13th-order Pade approximant."""
    b = [
        64764752532480000.0,
        32382376266240000.0,
        7771770303897600.0,
        1187353796428800.0,
        129060195264000.0,
        10559470521600.0,
        670442572800.0,
        33522128640.0,
        1323241920.0,
        40840800.0,
        960960.0,
        16380.0,
        182.0,
    ]
    if not paddle.framework.in_dynamic_mode():
        b = [paddle.full((), x, dtype) for x in b]

    if mat_a6 is None:
        mat_a2, mat_a4, mat_a6, *_ = _matrix_mats(mat_a, 6, dtype)

    tmp_u = (
        paddle.matmul(mat_a6, mat_a6 + b[11] * mat_a4 + b[9] * mat_a2)
        + b[7] * mat_a6
        + b[5] * mat_a4
        + b[3] * mat_a2
        + b[1] * mat_i
    )
    mat_u = paddle.matmul(mat_a, tmp_u)
    tmp_v = b[12] * mat_a6 + b[10] * mat_a4 + b[8] * mat_a2
    mat_v = (
        paddle.matmul(mat_a6, tmp_v)
        + b[6] * mat_a6
        + b[4] * mat_a4
        + b[2] * mat_a2
        + b[0] * mat_i
    )
    return mat_u, mat_v


def _matrix_uv_where(vals, cases, l1_norm):
    if len(vals) == 1:
        return paddle.where(
            paddle.less_than(l1_norm, vals[0]), cases[0], cases[1]
        )
    else:
        return paddle.where(
            paddle.less_than(l1_norm, vals[0]),
            cases[0],
            _matrix_uv_where(vals[1:], cases[1:], l1_norm),
        )


def _matrix_mats(mat_a, total, dtype):
    mat_a2 = paddle.matmul(mat_a, mat_a)
    mat_a4 = None
    mat_a6 = None
    mat_a8 = None

    if total > 2:
        mat_a4 = paddle.matmul(mat_a2, mat_a2)

    if total > 4:
        mat_a6 = paddle.matmul(mat_a4, mat_a2)

    if total > 6:
        mat_a8 = paddle.matmul(mat_a6, mat_a2)

    return mat_a2, mat_a4, mat_a6, mat_a8


def _matrix_uv_float32(mat_a, l1_norm, squarings, dtype):
    mat_i = paddle.eye(mat_a.shape[-1], dtype=dtype)
    mat_a2, mat_a4, *_ = _matrix_mats(mat_a, 4, dtype)

    u3, v3 = _matrix_exp_pade3(mat_a, mat_i, mat_a2, dtype=dtype)
    u5, v5 = _matrix_exp_pade5(mat_a, mat_i, mat_a2, mat_a4, dtype=dtype)
    u7, v7 = _matrix_exp_pade7(
        mat_a
        / paddle.cast(
            paddle.pow(paddle.full((), 2.0, dtype), squarings),
            dtype,
        ),
        mat_i,
        dtype=dtype,
    )
    conds = (
        paddle.full((), 4.258730016922831e-001, dtype),
        paddle.full((), 1.880152677804762e000, dtype),
    )

    u = _matrix_uv_where(conds, (u3, u5, u7), l1_norm)
    v = _matrix_uv_where(conds, (v3, v5, v7), l1_norm)

    return u, v


def _matrix_uv_float64(mat_a, l1_norm, squarings, dtype):
    mat_i = paddle.eye(mat_a.shape[-1], dtype=dtype)
    mat_a2, mat_a4, mat_a6, mat_a8, *_ = _matrix_mats(mat_a, 8, dtype)

    u3, v3 = _matrix_exp_pade3(mat_a, mat_i, mat_a2, dtype=dtype)
    u5, v5 = _matrix_exp_pade5(mat_a, mat_i, mat_a2, mat_a4, dtype=dtype)
    u7, v7 = _matrix_exp_pade7(
        mat_a, mat_i, mat_a2, mat_a4, mat_a6, dtype=dtype
    )
    u9, v9 = _matrix_exp_pade9(
        mat_a, mat_i, mat_a2, mat_a4, mat_a6, mat_a8, dtype=dtype
    )
    u13, v13 = _matrix_exp_pade13(
        mat_a
        / paddle.cast(
            paddle.pow(paddle.full((), 2.0, dtype), squarings),
            dtype,
        ),
        mat_i,
        dtype=dtype,
    )

    conds = (
        paddle.full((), 1.495585217958292e-002, dtype),
        paddle.full((), 2.539398330063230e-001, dtype),
        paddle.full((), 9.504178996162932e-001, dtype),
        paddle.full((), 2.097847961257068e000, dtype),
    )

    u = _matrix_uv_where(conds, (u3, u5, u7, u9, u13), l1_norm)
    v = _matrix_uv_where(conds, (v3, v5, v7, v9, v13), l1_norm)

    return u, v


def matrix_exp(x: Tensor, name: str | None = None) -> Tensor:
    r"""
    Computes the matrix exponential of square matrices.

    .. math::

        exp(A) = \sum_{n=0}^\infty A^n/n!

    The input tensor x should be of square matrices with shape like :math:`(*, M, M)`, and the
    exponential output is computed by Pade approximation of the scaling and squaring method.

    [1] Nicholas J. Higham, The scaling and squaring method for the matrix exponential revisited.

    Args:
        x (Tensor): A tensor with shape :math:`(*, M, M)` where :math:`*` is zero or more batch dimensions. The data type should be one of float32, float64.
        name (str|None, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        Tensor, the shape and dtype are same as input tensor.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> mat_a = paddle.empty((2, 2, 2))
            >>> mat_a[0, :, :] = paddle.eye(2, 2)
            >>> mat_a[1, :, :] = 2 * paddle.eye(2, 2)
            >>> print(mat_a)
            Tensor(shape=[2, 2, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[[1., 0.],
              [0., 1.]],
             [[2., 0.],
              [0., 2.]]])

            >>> out = paddle.linalg.matrix_exp(mat_a)
            >>> print(out)
            Tensor(shape=[2, 2, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[[2.71828198, 0.        ],
              [0.        , 2.71828198]],
             [[7.38905621, 0.        ],
              [0.        , 7.38905621]]])

            >>> import math
            >>> mat_a = paddle.to_tensor([[0, math.pi/3], [-math.pi/3, 0]])
            >>> out = paddle.linalg.matrix_exp(mat_a)
            >>> print(out)
            Tensor(shape=[2, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[ 0.49999994,  0.86602545],
             [-0.86602551,  0.50000000]])

    """

    # convert to tensor if necessary
    if not isinstance(
        x,
        (
            paddle.Tensor,
            paddle.base.framework.Variable,
            paddle.base.libpaddle.pir.Value,
        ),
    ):
        mat_a = paddle.to_tensor(x)
    else:
        mat_a = x

    dtype = convert_dtype(mat_a.dtype)

    # check dtype, shape
    if dtype not in ['float32', 'float64']:
        raise ValueError(
            f"The input tensor's dtype must be float32 or float64, but got {dtype}"
        )

    # 0-dim
    if mat_a.ndim == 0:
        return paddle.exp(mat_a)

    # check tensor dim
    if mat_a.ndim < 2:
        raise ValueError('The input tensor must be at least two-dimensional')

    if mat_a.shape[-1] != mat_a.shape[-2]:
        raise ValueError('Last 2 dimensions of the tensor must be square')

    # scalar case
    if list(mat_a.shape[-2:]) == [1, 1]:
        return paddle.exp(mat_a)

    # compute uv
    l1_norm = paddle.unsqueeze(
        paddle.max(paddle.sum(paddle.abs(mat_a), axis=mat_a.ndim - 2), axis=-1),
        axis=[-1, -2],
    )

    squarings = paddle.full(mat_a.shape, 0, dtype)
    _matrix_uv_func = None

    # dtype already checked before, we use `if-elif` only
    if dtype == 'float32':
        maxnorm = paddle.full((), 3.925724783138660, dtype)
        squarings = paddle.floor(
            paddle.log(l1_norm / maxnorm)
            / paddle.log(paddle.full((), 2.0, dtype))
        )
        squarings = paddle.maximum(squarings, paddle.zeros_like(squarings))

        _matrix_uv_func = _matrix_uv_float32

    elif dtype == 'float64':
        maxnorm = paddle.full((), 5.371920351148152, dtype)
        squarings = paddle.floor(
            paddle.log(l1_norm / maxnorm)
            / paddle.log(paddle.full((), 2.0, dtype))
        )
        squarings = paddle.maximum(squarings, paddle.zeros_like(squarings))

        _matrix_uv_func = _matrix_uv_float64

    u, v = _matrix_uv_func(mat_a, l1_norm, squarings, dtype)

    # compute result
    is_finite = paddle.isfinite(paddle.max(l1_norm))
    result = paddle.static.nn.cond(
        is_finite,
        lambda: paddle.linalg.solve(-u + v, u + v),
        lambda: paddle.full(mat_a.shape, np.nan, dtype),
    )

    max_squaring = paddle.max(squarings)
    i = paddle.full((), 0, dtype)

    def cond(i, _):
        return paddle.static.nn.cond(
            is_finite,
            lambda: paddle.less_than(i, max_squaring),
            lambda: paddle.full((), False, dtype=paddle.bool),
        )

    def body(i, result):
        return i + 1, paddle.where(
            paddle.less_than(i, squarings),
            paddle.matmul(result, result),
            result,
        )

    _, result = paddle.static.nn.while_loop(cond, body, [i, result])

    return result


def histogramdd(
    x: Tensor,
    bins: Tensor | list[int] | int = 10,
    ranges: Sequence[float] | None = None,
    density: bool = False,
    weights: Tensor | None = None,
    name: str | None = None,
) -> tuple[Tensor, list[Tensor]]:
    r"""
    Computes a multi-dimensional histogram of the values in a tensor.

    Interprets the elements of an input tensor whose innermost dimension has size `N` as a collection of N-dimensional points. Maps each of the points into a set of N-dimensional bins and returns the number of points (or total weight) in each bin.

    input `x` must be a tensor with at least 2 dimensions. If input has shape `(M, N)`, each of its `M` rows defines a point in N-dimensional space. If input has three or more dimensions, all but the last dimension are flattened.

    Each dimension is independently associated with its own strictly increasing sequence of bin edges. Bin edges may be specified explicitly by passing a sequence of 1D tensors. Alternatively, bin edges may be constructed automatically by passing a sequence of integers specifying the number of equal-width bins in each dimension.

    Args:
        x (Tensor): The input tensor.
        bins (list[Tensor], list[int], or int): If list[Tensor], defines the sequences of bin edges. If list[int], defines the number of equal-width bins in each dimension. If int, defines the number of equal-width bins for all dimensions.
        ranges (sequence[float]|None, optional): Defines the leftmost and rightmost bin edges in each dimension. If is None, set the minimum and maximum as leftmost and rightmost edges for each dimension.
        density (bool, optional): If False (default), the result will contain the count (or total weight) in each bin. If True, each count (weight) is divided by the total count (total weight), then divided by the volume of its associated bin.
        weights (Tensor, optional): By default, each value in the input has weight 1. If a weight tensor is passed, each N-dimensional coordinate in input contributes its associated weight towards its bin's result. The weight tensor should have the same shape as the input tensor excluding its innermost dimension N.
        name (str|None, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        N-dimensional Tensor containing the values of the histogram. ``bin_edges(Tensor[])``,  sequence of N 1D Tensors containing the bin edges.

    Examples:
        .. code-block:: python
            :name: exampl

            >>> import paddle
            >>> x = paddle.to_tensor([[0., 1.], [1., 0.], [2.,0.], [2., 2.]])
            >>> bins = [3,3]
            >>> weights = paddle.to_tensor([1., 2., 4., 8.])
            >>> paddle.histogramdd(x, bins=bins, weights=weights)
            (Tensor(shape=[3, 3], dtype=float32, place=Place(gpu:0), stop_gradient=True,
                   [[0., 1., 0.],
                    [2., 0., 0.],
                    [4., 0., 8.]]), [Tensor(shape=[4], dtype=float32, place=Place(gpu:0), stop_gradient=True,
                   [0.        , 0.66666669, 1.33333337, 2.        ]), Tensor(shape=[4], dtype=float32, place=Place(gpu:0), stop_gradient=True,
                   [0.        , 0.66666669, 1.33333337, 2.        ])])

        .. code-block:: python
            :name: examp2

            >>> import paddle
            >>> y = paddle.to_tensor([[0., 0.], [1., 1.], [2., 2.]])
            >>> bins = [2,2]
            >>> ranges = [0., 1., 0., 1.]
            >>> density = True
            >>> paddle.histogramdd(y, bins=bins, ranges=ranges, density=density)
            (Tensor(shape=[2, 2], dtype=float32, place=Place(gpu:0), stop_gradient=True,
                   [[2., 0.],
                    [0., 2.]]), [Tensor(shape=[3], dtype=float32, place=Place(gpu:0), stop_gradient=True,
                   [0.        , 0.50000000, 1.        ]), Tensor(shape=[3], dtype=float32, place=Place(gpu:0), stop_gradient=True,
                   [0.        , 0.50000000, 1.        ])])


    """

    def __check_x(x):
        assert (
            len(x.shape) >= 2
        ), "input x must be a tensor with at least 2 dimensions."
        check_variable_and_dtype(
            x,
            'x',
            [
                'float32',
                'float64',
            ],
            'histogramdd',
        )

    def __check_bins(bins, x):  # when Tensor[], check dtype
        for bins_tensor in bins:
            bins_tensor = paddle.to_tensor(bins_tensor)
            check_variable_and_dtype(
                bins_tensor,
                'bins',
                [
                    'float32',
                    'float64',
                ],
                'histogramdd',
            )
            assert (
                bins_tensor.dtype == x.dtype
            ), "When bins is Tensor[], the dtype of bins must be the same as x.\n"

    def __check_weights(x, weights):
        if weights is None:
            return
        x_shape, weights_shape = x.shape, weights.shape
        assert len(x_shape) == len(weights_shape) + 1, (
            "if weight tensor is provided,"
            "it should have the same shape as the input tensor excluding its innermost dimension.\n"
        )
        for i, _ in enumerate(weights_shape):
            assert weights_shape[i] == x_shape[i], (
                "if weight tensor is provided,"
                "it should have the same shape as the input tensor excluding its innermost dimension.\n"
            )
        check_variable_and_dtype(
            weights,
            'weights',
            [
                'float32',
                'float64',
            ],
            'histogramdd',
        )
        assert (
            weights.dtype == x.dtype
        ), "The dtype of weights must be the same as x.\n"

    def __check_ranges(D, ranges):
        if ranges is None:
            return
        check_type(ranges, 'ranges', (list, tuple), 'histogramdd')
        assert D * 2 == len(
            ranges
        ), f"The length of ranges list must be {D * 2}\n"

    check_type(density, 'density', bool, 'histogramdd')

    __check_x(x)
    # weights
    __check_weights(x, weights)
    D = x.shape[-1]
    reshaped_input = x.reshape([-1, D])
    N = reshaped_input.shape[0]
    reshaped_weights = None
    if weights is not None:
        weights = weights.astype(x.dtype)
        reshaped_weights = weights.reshape([N])
        assert reshaped_weights.shape[0] == N, f"The size of weight must be {N}"
    # ranges
    __check_ranges(D, ranges)
    if ranges is None:
        ranges = paddle.zeros([D, 2], dtype=x.dtype)
        maxv = paddle.max(reshaped_input, axis=0).reshape([-1])
        minv = paddle.min(reshaped_input, axis=0).reshape([-1])

        if paddle.in_dynamic_mode():
            ranges[:, 0] = minv
            ranges[:, 1] = maxv
        else:
            ranges = paddle.static.setitem(ranges, (slice(None), 0), minv)
            ranges = paddle.static.setitem(ranges, (slice(None), 1), maxv)
    else:
        ranges = paddle.to_tensor(ranges, dtype=x.dtype).reshape([D, 2])
    # bins to edges
    edges = []
    hist_shape = []
    dedges = []
    if isinstance(bins, (int, list)):  # int or int[]
        if isinstance(bins, int):
            bins = [bins] * D
        assert (
            len(bins) == D
        ), f"The length of bins must be {D} when bins is a list.\n"
        for idx, r in enumerate(ranges):
            if not isinstance(bins[idx], int):
                raise ValueError(
                    f"The type of {idx}-th element in bins list must be int."
                )
            e = paddle.linspace(r[0], r[1], bins[idx] + 1, x.dtype)
            edges.append(e)
            dedges.append(e.diff())
            hist_shape.append(bins[idx] + 2)
    elif isinstance(
        bins, tuple
    ):  # tuple with D tensors for each innermost dimension
        __check_bins(bins, x)
        for bin in bins:
            bin = paddle.to_tensor(bin)
            edges.append(bin)
            dedges.append(bin.diff())
            hist_shape.append(bin.shape[0] + 1)
    else:
        raise ValueError("Input bins must be Tensor[], int[], or int.")
    index_list = []
    # edges shape: [D, linspaced]
    # index_list shape: [D, N]
    for idx, edge in enumerate(edges):
        edge = paddle.to_tensor(edge)
        index_list.append(
            paddle.searchsorted(edge, reshaped_input[:, idx], right=True)
        )
    index_list = paddle.to_tensor(index_list)
    for i in range(D):
        on_edge = reshaped_input[:, i] == edges[i][-1]
        if paddle.in_dynamic_mode():
            index_list[i] = paddle.where(
                on_edge, index_list[i] - 1, index_list[i]
            )
        else:
            index_list_i = paddle.where(
                on_edge, index_list[i] - 1, index_list[i]
            )
            index_list = paddle.static.setitem(index_list, i, index_list_i)
    index_list = tuple(index_list)
    lut = paddle.arange(
        paddle.to_tensor(hist_shape).prod(),
    ).reshape(hist_shape)
    flattened_index = lut[index_list]
    hist = paddle.bincount(
        flattened_index,
        reshaped_weights,
        minlength=paddle.to_tensor(hist_shape).prod(),
    )
    hist = hist.reshape(hist_shape)
    hist = hist.astype(x.dtype)

    core = D * (slice(1, -1),)
    hist = hist[core]

    if density:
        s = hist.sum()
        for i in range(D):
            shape = D * [1]
            shape[i] = hist_shape[i] - 2
            hist = hist / dedges[i].reshape(shape)
        hist /= s

    return (hist, edges)


def ormqr(
    x: Tensor,
    tau: Tensor,
    y: Tensor,
    left: bool = True,
    transpose: bool = False,
    name: str | None = None,
) -> Tensor:
    r'''
    Calculate the product of a normal matrix and a householder matrix.
    Compute the product of the matrix C (given by y) with dimensions (m, n) and a matrix Q,
    where Q is generated by the Householder reflection coefficient (x, tau). Returns a Tensor.

    Args:
        x (Tensor): Shape(\*,mn, k), when left is True, the value of mn is equal to m, otherwise the value of mn is equal to n. \* indicates that the length of the tensor on axis 0 is 0 or greater.
        tau (Tensor): Shape (\*, min(mn, k)), where \* indicates that the length of the Tensor on axis 0 is 0 or greater, and its type is the same as input.
        y (Tensor): Shape (\*m,n), where \* indicates that the length of the Tensor on axis 0 is 0 or greater, and its type is the same as input.
        left (bool, optional): Determines the order in which the matrix product operations are operated. If left is true, the order of evaluation is op(Q) \* y, otherwise, the order of evaluation is y \* op(Q). Default value: True.
        transpose (bool, optional): If true, the matrix Q is conjugated and transposed, otherwise, the conjugate transpose transformation is not performed. Default value: False.
        name (str|None, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        Tensor. Data type and dimension are equals with :attr:`y`.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import numpy as np
            >>> from paddle import  linalg

            >>> input = paddle.to_tensor([[-114.6, 10.9, 1.1], [-0.304, 38.07, 69.38], [-0.45, -0.17, 62]])
            >>> tau = paddle.to_tensor([1.55, 1.94, 3.0])
            >>> y = paddle.to_tensor([[-114.6, 10.9, 1.1], [-0.304, 38.07, 69.38], [-0.45, -0.17, 62]])
            >>> output = linalg.ormqr(input, tau, y)
            >>> print(output)
            Tensor(shape=[3, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
                [[ 63.82712936 , -13.82312393 , -116.28614044],
                [-53.65926361 , -28.15783691 , -70.42700958 ],
                [-79.54292297 ,  24.00182915 , -41.34253311 ]])
    '''

    check_dtype(
        y.dtype,
        'y',
        [
            'float16',
            'float32',
            'float64',
            'complex64',
            'complex128',
        ],
        'ormqr',
    )
    check_type(left, 'left', bool, 'ormqr')
    check_type(transpose, 'transpose', bool, 'ormqr')
    assert (
        x.dtype == tau.dtype and x.dtype == y.dtype
    ), "The input tau and y must have the same dtype with the x.\n"
    assert (
        len(x.shape) >= 2 and len(y.shape) >= 2 and len(tau.shape) >= 1
    ), "The input x and y must have more than 2 dimensions, and input tau must have more than 1 dimension"
    assert len(x.shape) == len(tau.shape) + 1 and len(x.shape) == len(
        y.shape
    ), "the dimension of x is 1 larger than the dimension of tau\n and the dimension of x is equal to the dimension of input"
    assert (
        x.shape[-1] == tau.shape[-1]
    ), "The innermost dimension of x and tau should be the same"
    if transpose and left:
        assert (
            x.shape[-2] == y.shape[-2]
        ), "The row dimensions of x and y should be the same"
    elif not transpose and left:
        assert (
            x.shape[-1] == y.shape[-2]
        ), "The column dimension of x and the row dimension of y should be the same"
    elif transpose and not left:
        assert (
            x.shape[-2] == y.shape[-1]
        ), "The row dimension of x and the column dimension of y should be the same"
    else:
        assert (
            x.shape[-1] == y.shape[-1]
        ), "The column dimensions of Impt and Osser's should be the same"
    if len(x.shape) == 3:
        assert (
            x.shape[0] == y.shape[0] and x.shape[0] == tau.shape[0]
        ), "The input and tau and y parameters should have the same batch"
    Q = householder_product(x, tau)
    if len(x.shape) == 2:
        Q = Q.T if transpose else Q
    else:
        Q = paddle.transpose(Q, [0, 2, 1]) if transpose else Q
    result = matmul(Q, y) if left else matmul(y, Q)

    return result


def cholesky_inverse(
    x: Tensor, upper: bool = False, name: str | None = None
) -> Tensor:
    r"""
    Using the Cholesky factor `U` to calculate the inverse matrix of a symmetric positive definite matrix, returns the matrix `inv`.

    If `upper` is `False`, `U` is lower triangular matrix:

    .. math::

        inv = (UU^{T})^{-1}

    If `upper` is `True`, `U` is upper triangular matrix:

    .. math::

        inv = (U^{T}U)^{-1}

    Args:
        x (Tensor): A tensor of lower or upper triangular Cholesky decompositions of symmetric matrix with shape `[N, N]`.
            The data type of the `x` should be one of ``float32``, ``float64``.
        upper (bool, optional): If `upper` is `False`, `x` is lower triangular matrix, or is upper triangular matrix. Default: `False`.
        name (str|None, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        Tensor. Computes the inverse matrix.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> # lower triangular matrix
            >>> x = paddle.to_tensor([[3.,.0,.0], [5.,3.,.0], [-1.,1.,2.]])
            >>> out = paddle.linalg.cholesky_inverse(x)
            >>> print(out)
            Tensor(shape=[3, 3], dtype=float32, place=Place(gpu:0), stop_gradient=True,
            [[ 0.61728382, -0.25925916,  0.22222219],
             [-0.25925916,  0.13888884, -0.08333331],
             [ 0.22222218, -0.08333331,  0.25000000]])

            >>> # upper triangular matrix
            >>> out = paddle.linalg.cholesky_inverse(x.T, upper=True)
            >>> print(out)
            Tensor(shape=[3, 3], dtype=float32, place=Place(gpu:0), stop_gradient=True,
            [[ 0.61728382, -0.25925916,  0.22222219],
             [-0.25925916,  0.13888884, -0.08333331],
             [ 0.22222218, -0.08333331,  0.25000000]])

    """
    if x.ndim != 2:
        raise ValueError('The input tensor must be 2-dimensional')

    if x.shape[0] != x.shape[1]:
        raise ValueError('The input tensor must be square matrix')

    if upper:
        A = x.T @ x
    else:
        A = x @ x.T
    return paddle.linalg.inv(A)


def diagonal(
    x: Tensor,
    offset: int = 0,
    axis1: int = 0,
    axis2: int = 1,
    name: str | None = None,
) -> Tensor:
    """
    Computes the diagonals of the input tensor x.

    If ``x`` is 2D, returns the diagonal.
    If ``x`` has larger dimensions, diagonals be taken from the 2D planes specified by axis1 and axis2.
    By default, the 2D planes formed by the first and second axis of the input tensor x.

    The argument ``offset`` determines where diagonals are taken from input tensor x:

    - If offset = 0, it is the main diagonal.
    - If offset > 0, it is above the main diagonal.
    - If offset < 0, it is below the main diagonal.

    Args:
        x (Tensor): The input tensor x. Must be at least 2-dimensional. The input data type should be bool, int32,
            int64, bfloat16, float16, float32, float64.
        offset (int, optional): Which diagonals in input tensor x will be taken. Default: 0 (main diagonals).
        axis1 (int, optional): The first axis with respect to take diagonal. Default: 0.
        axis2 (int, optional): The second axis with respect to take diagonal. Default: 1.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: a partial view of input tensor in specify two dimensions, the output data type is the same as input data type.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> paddle.seed(2023)
            >>> x = paddle.rand([2, 2, 3],'float32')
            >>> print(x)
            Tensor(shape=[2, 2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[[0.86583614, 0.52014720, 0.25960937],
              [0.90525323, 0.42400089, 0.40641287]],
             [[0.97020894, 0.74437362, 0.51785129],
              [0.73292869, 0.97786582, 0.04315904]]])

            >>> out1 = paddle.diagonal(x)
            >>> print(out1)
            Tensor(shape=[3, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[0.86583614, 0.73292869],
             [0.52014720, 0.97786582],
             [0.25960937, 0.04315904]])

            >>> out2 = paddle.diagonal(x, offset=0, axis1=2, axis2=1)
            >>> print(out2)
            Tensor(shape=[2, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[0.86583614, 0.42400089],
             [0.97020894, 0.97786582]])

            >>> out3 = paddle.diagonal(x, offset=1, axis1=0, axis2=1)
            >>> print(out3)
            Tensor(shape=[3, 1], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[0.90525323],
             [0.42400089],
             [0.40641287]])

            >>> out4 = paddle.diagonal(x, offset=0, axis1=1, axis2=2)
            >>> print(out4)
            Tensor(shape=[2, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[0.86583614, 0.42400089],
             [0.97020894, 0.97786582]])

    """
    if in_dynamic_or_pir_mode():
        return _C_ops.diagonal(x, offset, axis1, axis2)
    else:

        def __check_input(x, offset, axis1, axis2):
            check_dtype(
                x.dtype,
                'Input',
                [
                    'bool',
                    'int32',
                    'int64',
                    'float16',
                    'uint16',
                    'float32',
                    'float64',
                ],
                'diagonal',
            )

            input_shape = list(x.shape)
            assert len(input_shape) >= 2, (
                "The x must be at least 2-dimensional, "
                f"But received Input x's dimensional: {len(input_shape)}.\n"
            )

            axis1_ = axis1 if axis1 >= 0 else len(input_shape) + axis1
            axis2_ = axis2 if axis2 >= 0 else len(input_shape) + axis2

            assert axis1_ < len(
                input_shape
            ), f"The argument axis1 is out of range (expected to be in range of [{-(len(input_shape))}, {len(input_shape) - 1}], but got {axis1}).\n"

            assert axis2_ < len(
                input_shape
            ), f"The argument axis2 is out of range (expected to be in range of [{-(len(input_shape))}, {len(input_shape) - 1}], but got {axis2}).\n"

            assert axis1_ != axis2_, (
                "axis1 and axis2 cannot be the same axis."
                f"But received axis1 = {axis1}, axis2 = {axis2}\n"
            )

        __check_input(x, offset, axis1, axis2)
        helper = LayerHelper('diagonal', **locals())
        out = helper.create_variable_for_type_inference(dtype=x.dtype)

        helper.append_op(
            type='diagonal',
            inputs={'Input': [x]},
            attrs={'offset': offset, 'axis1': axis1, 'axis2': axis2},
            outputs={'Out': [out]},
        )

        return out
