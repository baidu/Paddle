#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from typing_extensions import overload

import paddle
from paddle import _C_ops
from paddle.common_ops_import import VarDesc, Variable
from paddle.utils.inplace_utils import inplace_apis_in_dygraph_only

from ..base.data_feeder import check_dtype, check_variable_and_dtype
from ..framework import (
    LayerHelper,
    convert_np_dtype_to_dtype_,
    core,
    in_dynamic_mode,
    in_dynamic_or_pir_mode,
    in_pir_mode,
)

if TYPE_CHECKING:
    from paddle import Tensor
    from paddle._typing import DTypeLike

# from ..base.layers import has_inf  #DEFINE_ALIAS
# from ..base.layers import has_nan  #DEFINE_ALIAS

__all__ = []


def argsort(
    x: Tensor,
    axis: int = -1,
    descending: bool = False,
    stable: bool = False,
    name: str | None = None,
) -> Tensor:
    """
    Sorts the input along the given axis, and returns the corresponding index tensor for the sorted output values. The default sort algorithm is ascending, if you want the sort algorithm to be descending, you must set the :attr:`descending` as True.

    Args:
        x (Tensor): An input N-D Tensor with type bfloat16, float16, float32, float64, int16,
            int32, int64, uint8.
        axis (int, optional): Axis to compute indices along. The effective range
            is [-R, R), where R is Rank(x). when axis<0, it works the same way
            as axis+R. Default is -1.
        descending (bool, optional) : Descending is a flag, if set to true,
            algorithm will sort by descending order, else sort by
            ascending order. Default is false.
        stable (bool, optional): Whether to use stable sorting algorithm or not.
            When using stable sorting algorithm, the order of equivalent elements
            will be preserved. Default is False.
        name (str|None, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        Tensor, sorted indices(with the same shape as ``x``
        and with data type int64).

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([[[5,8,9,5],
            ...                        [0,0,1,7],
            ...                        [6,9,2,4]],
            ...                       [[5,2,4,2],
            ...                        [4,7,7,9],
            ...                        [1,7,0,6]]],
            ...                      dtype='float32')
            >>> out1 = paddle.argsort(x, axis=-1)
            >>> out2 = paddle.argsort(x, axis=0)
            >>> out3 = paddle.argsort(x, axis=1)

            >>> print(out1)
            Tensor(shape=[2, 3, 4], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[[0, 3, 1, 2],
              [0, 1, 2, 3],
              [2, 3, 0, 1]],
             [[1, 3, 2, 0],
              [0, 1, 2, 3],
              [2, 0, 3, 1]]])

            >>> print(out2)
            Tensor(shape=[2, 3, 4], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[[0, 1, 1, 1],
              [0, 0, 0, 0],
              [1, 1, 1, 0]],
             [[1, 0, 0, 0],
              [1, 1, 1, 1],
              [0, 0, 0, 1]]])

            >>> print(out3)
            Tensor(shape=[2, 3, 4], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[[1, 1, 1, 2],
              [0, 0, 2, 0],
              [2, 2, 0, 1]],
             [[2, 0, 2, 0],
              [1, 1, 0, 2],
              [0, 2, 1, 1]]])

            >>> x = paddle.to_tensor([1, 0]*40, dtype='float32')
            >>> out1 = paddle.argsort(x, stable=False)
            >>> out2 = paddle.argsort(x, stable=True)

            >>> print(out1)
            Tensor(shape=[80], dtype=int64, place=Place(cpu), stop_gradient=True,
            [55, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 1 , 57, 59, 61,
             63, 65, 67, 69, 71, 73, 75, 77, 79, 17, 11, 13, 25, 7 , 3 , 27, 23, 19,
             15, 5 , 21, 9 , 10, 64, 62, 68, 60, 58, 8 , 66, 14, 6 , 70, 72, 4 , 74,
             76, 2 , 78, 0 , 20, 28, 26, 30, 32, 24, 34, 36, 22, 38, 40, 12, 42, 44,
             18, 46, 48, 16, 50, 52, 54, 56])

            >>> print(out2)
            Tensor(shape=[80], dtype=int64, place=Place(cpu), stop_gradient=True,
            [1 , 3 , 5 , 7 , 9 , 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35,
             37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71,
             73, 75, 77, 79, 0 , 2 , 4 , 6 , 8 , 10, 12, 14, 16, 18, 20, 22, 24, 26,
             28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62,
             64, 66, 68, 70, 72, 74, 76, 78])
    """
    if in_dynamic_or_pir_mode():
        _, ids = _C_ops.argsort(x, axis, descending, stable)
        return ids
    else:
        check_variable_and_dtype(
            x,
            'x',
            [
                'uint16',
                'float16',
                'float32',
                'float64',
                'int16',
                'int32',
                'int64',
                'uint8',
            ],
            'argsort',
        )

        helper = LayerHelper("argsort", **locals())
        out = helper.create_variable_for_type_inference(
            dtype=x.dtype, stop_gradient=True
        )
        ids = helper.create_variable_for_type_inference(
            VarDesc.VarType.INT64, stop_gradient=True
        )
        helper.append_op(
            type='argsort',
            inputs={'X': x},
            outputs={'Out': out, 'Indices': ids},
            attrs={'axis': axis, 'descending': descending, 'stable': stable},
        )
        return ids


def argmax(
    x: Tensor,
    axis: int | None = None,
    keepdim: bool = False,
    dtype: DTypeLike = "int64",
    name: str | None = None,
) -> Tensor:
    """
    Computes the indices of the max elements of the input tensor's
    element along the provided axis.

    Args:
        x (Tensor): An input N-D Tensor with type float16, float32, float64, int16,
            int32, int64, uint8.
        axis (int|None, optional): Axis to compute indices along. The effective range
            is [-R, R), where R is x.ndim. when axis < 0, it works the same way
            as axis + R. Default is None, the input `x` will be into the flatten tensor, and selecting the min value index.
        keepdim (bool, optional): Whether to keep the given axis in output. If it is True, the dimensions will be same as input x and with size one in the axis. Otherwise the output dimensions is one fewer than x since the axis is squeezed. Default is False.
        dtype (str|np.dtype, optional): Data type of the output tensor which can
                    be int32, int64. The default value is ``int64`` , and it will
                    return the int64 indices.
        name (str|None, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        Tensor, return the tensor of int32 if set :attr:`dtype` is int32, otherwise return the tensor of int64.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([[5,8,9,5],
            ...                       [0,0,1,7],
            ...                       [6,9,2,4]])
            >>> out1 = paddle.argmax(x)
            >>> print(out1.numpy())
            2
            >>> out2 = paddle.argmax(x, axis=0)
            >>> print(out2.numpy())
            [2 2 0 1]
            >>> out3 = paddle.argmax(x, axis=-1)
            >>> print(out3.numpy())
            [2 3 1]
            >>> out4 = paddle.argmax(x, axis=0, keepdim=True)
            >>> print(out4.numpy())
            [[2 2 0 1]]
    """
    if axis is not None and not isinstance(
        axis, (int, Variable, paddle.pir.Value)
    ):
        raise TypeError(
            f"The type of 'axis'  must be int or Tensor or None in argmax, but received {type(axis)}."
        )

    if dtype is None:
        raise ValueError(
            "the value of 'dtype' in argmax could not be None, but received None"
        )

    var_dtype = convert_np_dtype_to_dtype_(dtype)
    flatten = False
    if axis is None:
        flatten = True
        axis = 0

    if in_dynamic_mode():
        return _C_ops.argmax(x, axis, keepdim, flatten, var_dtype)
    elif in_pir_mode():
        check_dtype(var_dtype, 'dtype', ['int32', 'int64'], 'argmax')
        return _C_ops.argmax(x, axis, keepdim, flatten, var_dtype)
    else:
        helper = LayerHelper("argmax", **locals())
        check_variable_and_dtype(
            x,
            'x',
            [
                'uint16',
                'float16',
                'float32',
                'float64',
                'int16',
                'int32',
                'int64',
                'uint8',
            ],
            'paddle.argmax',
        )
        check_dtype(var_dtype, 'dtype', ['int32', 'int64'], 'argmax')
        attrs = {}
        out = helper.create_variable_for_type_inference(var_dtype)
        attrs['keepdims'] = keepdim
        attrs['axis'] = axis
        attrs['flatten'] = flatten
        attrs['dtype'] = var_dtype
        helper.append_op(
            type='arg_max', inputs={'X': x}, outputs={'Out': [out]}, attrs=attrs
        )
        out.stop_gradient = True
        return out


def argmin(
    x: Tensor,
    axis: int | None = None,
    keepdim: bool = False,
    dtype: DTypeLike = "int64",
    name: str | None = None,
) -> Tensor:
    """
    Computes the indices of the min elements of the input tensor's
    element along the provided axis.

    Args:
        x (Tensor): An input N-D Tensor with type float16, float32, float64, int16,
            int32, int64, uint8.
        axis (int|None, optional): Axis to compute indices along. The effective range
            is [-R, R), where R is x.ndim. when axis < 0, it works the same way
            as axis + R. Default is None, the input `x` will be into the flatten tensor, and selecting the min value index.
        keepdim (bool, optional): Whether to keep the given axis in output. If it is True, the dimensions will be same as input x and with size one in the axis. Otherwise the output dimensions is one fewer than x since the axis is squeezed. Default is False.
        dtype (str|np.dtype, optional): Data type of the output tensor which can
                    be int32, int64. The default value is 'int64', and it will
                    return the int64 indices.
        name (str|None, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        Tensor, return the tensor of `int32` if set :attr:`dtype` is `int32`, otherwise return the tensor of `int64`.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x =  paddle.to_tensor([[5,8,9,5],
            ...                        [0,0,1,7],
            ...                        [6,9,2,4]])
            >>> out1 = paddle.argmin(x)
            >>> print(out1.numpy())
            4
            >>> out2 = paddle.argmin(x, axis=0)
            >>> print(out2.numpy())
            [1 1 1 2]
            >>> out3 = paddle.argmin(x, axis=-1)
            >>> print(out3.numpy())
            [0 0 2]
            >>> out4 = paddle.argmin(x, axis=0, keepdim=True)
            >>> print(out4.numpy())
            [[1 1 1 2]]
    """
    if axis is not None and not isinstance(
        axis, (int, Variable, paddle.pir.Value)
    ):
        raise TypeError(
            f"The type of 'axis'  must be int or Tensor or None in argmin, but received {type(axis)}."
        )

    if dtype is None:
        raise ValueError(
            "the value of 'dtype' in argmin could not be None, but received None"
        )

    var_dtype = convert_np_dtype_to_dtype_(dtype)
    flatten = False
    if axis is None:
        flatten = True
        axis = 0

    if in_dynamic_mode():
        return _C_ops.argmin(x, axis, keepdim, flatten, var_dtype)
    elif in_pir_mode():
        check_dtype(var_dtype, 'dtype', ['int32', 'int64'], 'argmin')
        return _C_ops.argmin(x, axis, keepdim, flatten, var_dtype)
    else:
        helper = LayerHelper("argmin", **locals())
        check_variable_and_dtype(
            x,
            'x',
            [
                'uint16',
                'float16',
                'float32',
                'float64',
                'int16',
                'int32',
                'int64',
                'uint8',
            ],
            'paddle.argmin',
        )
        check_dtype(var_dtype, 'dtype', ['int32', 'int64'], 'argmin')
        out = helper.create_variable_for_type_inference(var_dtype)
        attrs = {}
        attrs['keepdims'] = keepdim
        attrs['axis'] = axis
        attrs['flatten'] = flatten
        attrs['dtype'] = var_dtype
        helper.append_op(
            type='arg_min', inputs={'X': x}, outputs={'Out': [out]}, attrs=attrs
        )
        out.stop_gradient = True
        return out


def index_select(
    x: Tensor, index: Tensor, axis: int = 0, name: str | None = None
) -> Tensor:
    """

    Returns a new tensor which indexes the ``input`` tensor along dimension ``axis`` using
    the entries in ``index`` which is a Tensor. The returned tensor has the same number
    of dimensions as the original ``x`` tensor. The dim-th dimension has the same
    size as the length of ``index``; other dimensions have the same size as in the ``x`` tensor.

    Args:
        x (Tensor): The input Tensor to be operated. The data of ``x`` can be one of float16, float32, float64, int32, int64, complex64 and complex128.
        index (Tensor): The 1-D Tensor containing the indices to index. The data type of ``index`` must be int32 or int64.
        axis (int, optional): The dimension in which we index. Default: if None, the ``axis`` is 0.
        name (str|None, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        Tensor, A Tensor with same data type as ``x``.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([[1.0, 2.0, 3.0, 4.0],
            ...                       [5.0, 6.0, 7.0, 8.0],
            ...                       [9.0, 10.0, 11.0, 12.0]])
            >>> index = paddle.to_tensor([0, 1, 1], dtype='int32')
            >>> out_z1 = paddle.index_select(x=x, index=index)
            >>> print(out_z1.numpy())
            [[1. 2. 3. 4.]
             [5. 6. 7. 8.]
             [5. 6. 7. 8.]]
            >>> out_z2 = paddle.index_select(x=x, index=index, axis=1)
            >>> print(out_z2.numpy())
            [[ 1.  2.  2.]
             [ 5.  6.  6.]
             [ 9. 10. 10.]]
    """

    if in_dynamic_or_pir_mode():
        return _C_ops.index_select(x, index, axis)
    else:
        helper = LayerHelper("index_select", **locals())
        check_variable_and_dtype(
            x,
            'x',
            [
                'uint16',
                'float16',
                'float32',
                'float64',
                'int32',
                'int64',
                'complex64',
                'complex128',
            ],
            'paddle.tensor.search.index_select',
        )
        check_variable_and_dtype(
            index,
            'index',
            ['int32', 'int64'],
            'paddle.tensor.search.index_select',
        )

        out = helper.create_variable_for_type_inference(x.dtype)

        helper.append_op(
            type='index_select',
            inputs={'X': x, 'Index': index},
            outputs={'Out': out},
            attrs={'dim': axis},
        )
        return out


@overload
def nonzero(x: Tensor, as_tuple: Literal[False] = ...) -> Tensor: ...


@overload
def nonzero(x: Tensor, as_tuple: Literal[True] = ...) -> tuple[Tensor, ...]: ...


@overload
def nonzero(x: Tensor, as_tuple: bool = ...) -> Tensor | tuple[Tensor, ...]: ...


def nonzero(x: Tensor, as_tuple=False):
    """
    Return a tensor containing the indices of all non-zero elements of the `input`
    tensor. If as_tuple is True, return a tuple of 1-D tensors, one for each dimension
    in `input`, each containing the indices (in that dimension) of all non-zero elements
    of `input`. Given a n-Dimensional `input` tensor with shape [x_1, x_2, ..., x_n], If
    as_tuple is False, we can get a output tensor with shape [z, n], where `z` is the
    number of all non-zero elements in the `input` tensor. If as_tuple is True, we can get
    a 1-D tensor tuple of length `n`, and the shape of each 1-D tensor is [z, 1].

    Args:
        x (Tensor): The input tensor variable.
        as_tuple (bool, optional): Return type, Tensor or tuple of Tensor.

    Returns:
        Tensor or tuple of Tensor, The data type is int64.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> x1 = paddle.to_tensor([[1.0, 0.0, 0.0],
            ...                        [0.0, 2.0, 0.0],
            ...                        [0.0, 0.0, 3.0]])
            >>> x2 = paddle.to_tensor([0.0, 1.0, 0.0, 3.0])
            >>> out_z1 = paddle.nonzero(x1)
            >>> print(out_z1)
            Tensor(shape=[3, 2], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[0, 0],
             [1, 1],
             [2, 2]])

            >>> out_z1_tuple = paddle.nonzero(x1, as_tuple=True)
            >>> for out in out_z1_tuple:
            ...     print(out)
            Tensor(shape=[3, 1], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[0],
             [1],
             [2]])
            Tensor(shape=[3, 1], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[0],
             [1],
             [2]])

            >>> out_z2 = paddle.nonzero(x2)
            >>> print(out_z2)
            Tensor(shape=[2, 1], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[1],
             [3]])

            >>> out_z2_tuple = paddle.nonzero(x2, as_tuple=True)
            >>> for out in out_z2_tuple:
            ...     print(out)
            Tensor(shape=[2, 1], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[1],
             [3]])

    """
    list_out = []
    shape = x.shape
    rank = len(shape)

    if in_dynamic_or_pir_mode():
        outs = _C_ops.nonzero(x)
    else:
        check_variable_and_dtype(
            x,
            'x',
            [
                'int16',
                'int32',
                'int64',
                'uint16',
                'float16',
                'float32',
                'float64',
                'bool',
            ],
            'where_index',
        )

        helper = LayerHelper("where_index", **locals())

        outs = helper.create_variable_for_type_inference(
            dtype=core.VarDesc.VarType.INT64
        )

        helper.append_op(
            type='where_index', inputs={'Condition': x}, outputs={'Out': [outs]}
        )

    if not as_tuple:
        return outs
    elif rank == 1:
        return (outs,)
    else:
        for i in range(rank):
            list_out.append(
                paddle.slice(outs, axes=[1], starts=[i], ends=[i + 1])
            )
        return tuple(list_out)


def sort(
    x: Tensor,
    axis: int = -1,
    descending: bool = False,
    stable: bool = False,
    name: str | None = None,
) -> Tensor:
    """

    Sorts the input along the given axis, and returns the sorted output tensor. The default sort algorithm is ascending, if you want the sort algorithm to be descending, you must set the :attr:`descending` as True.

    Args:
        x (Tensor): An input N-D Tensor with type float32, float64, int16,
            int32, int64, uint8.
        axis (int, optional): Axis to compute indices along. The effective range
            is [-R, R), where R is Rank(x). when axis<0, it works the same way
            as axis+R. Default is -1.
        descending (bool, optional) : Descending is a flag, if set to true,
            algorithm will sort by descending order, else sort by
            ascending order. Default is false.
        stable (bool, optional): Whether to use stable sorting algorithm or not.
            When using stable sorting algorithm, the order of equivalent elements
            will be preserved. Default is False.
        name (str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        Tensor, sorted tensor(with the same shape and data type as ``x``).

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([[[5,8,9,5],
            ...                        [0,0,1,7],
            ...                        [6,9,2,4]],
            ...                       [[5,2,4,2],
            ...                        [4,7,7,9],
            ...                        [1,7,0,6]]],
            ...                      dtype='float32')
            >>> out1 = paddle.sort(x=x, axis=-1)
            >>> out2 = paddle.sort(x=x, axis=0)
            >>> out3 = paddle.sort(x=x, axis=1)
            >>> print(out1.numpy())
            [[[5. 5. 8. 9.]
              [0. 0. 1. 7.]
              [2. 4. 6. 9.]]
             [[2. 2. 4. 5.]
              [4. 7. 7. 9.]
              [0. 1. 6. 7.]]]
            >>> print(out2.numpy())
            [[[5. 2. 4. 2.]
              [0. 0. 1. 7.]
              [1. 7. 0. 4.]]
             [[5. 8. 9. 5.]
              [4. 7. 7. 9.]
              [6. 9. 2. 6.]]]
            >>> print(out3.numpy())
            [[[0. 0. 1. 4.]
              [5. 8. 2. 5.]
              [6. 9. 9. 7.]]
             [[1. 2. 0. 2.]
              [4. 7. 4. 6.]
              [5. 7. 7. 9.]]]
    """
    if in_dynamic_or_pir_mode():
        outs, _ = _C_ops.argsort(x, axis, descending, stable)
        return outs
    else:
        helper = LayerHelper("sort", **locals())
        out = helper.create_variable_for_type_inference(
            dtype=x.dtype, stop_gradient=False
        )
        ids = helper.create_variable_for_type_inference(
            VarDesc.VarType.INT64, stop_gradient=True
        )
        helper.append_op(
            type='argsort',
            inputs={'X': x},
            outputs={'Out': out, 'Indices': ids},
            attrs={'axis': axis, 'descending': descending, 'stable': stable},
        )
        return out


def mode(
    x: Tensor, axis: int = -1, keepdim: bool = False, name: str | None = None
) -> tuple[Tensor, Tensor]:
    """
    Used to find values and indices of the modes at the optional axis.

    Args:
        x (Tensor): Tensor, an input N-D Tensor with type float32, float64, int32, int64.
        axis (int, optional): Axis to compute indices along. The effective range
            is [-R, R), where R is x.ndim. when axis < 0, it works the same way
            as axis + R. Default is -1.
        keepdim (bool, optional): Whether to keep the given axis in output. If it is True, the dimensions will be same as input x and with size one in the axis. Otherwise the output dimensions is one fewer than x since the axis is squeezed. Default is False.
        name (str|None, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        tuple (Tensor), return the values and indices. The value data type is the same as the input `x`. The indices data type is int64.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> tensor = paddle.to_tensor([[[1,2,2],[2,3,3]],[[0,5,5],[9,9,0]]], dtype=paddle.float32)
            >>> res = paddle.mode(tensor, 2)
            >>> print(res)
            (Tensor(shape=[2, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[2., 3.],
             [5., 9.]]), Tensor(shape=[2, 2], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[2, 2],
             [2, 1]]))

    """
    if in_dynamic_or_pir_mode():
        return _C_ops.mode(x, axis, keepdim)
    else:
        helper = LayerHelper("mode", **locals())
        inputs = {"X": [x]}
        attrs = {}
        attrs['axis'] = axis
        attrs['keepdim'] = keepdim

        values = helper.create_variable_for_type_inference(dtype=x.dtype)
        indices = helper.create_variable_for_type_inference(dtype="int64")

        helper.append_op(
            type="mode",
            inputs=inputs,
            outputs={"Out": [values], "Indices": [indices]},
            attrs=attrs,
        )
        indices.stop_gradient = True
        return values, indices


def where(
    condition: Tensor,
    x: Tensor | float | None = None,
    y: Tensor | float | None = None,
    name: str | None = None,
) -> Tensor:
    r"""
    Return a Tensor of elements selected from either :attr:`x` or :attr:`y` according to corresponding elements of :attr:`condition`. Concretely,

    .. math::

        out_i =
        \begin{cases}
        x_i, & \text{if}  \ condition_i \  \text{is} \ True \\
        y_i, & \text{if}  \ condition_i \  \text{is} \ False \\
        \end{cases}.

    Notes:
        ``numpy.where(condition)`` is identical to ``paddle.nonzero(condition, as_tuple=True)``, please refer to :ref:`api_paddle_nonzero`.

    Args:
        condition (Tensor): The condition to choose x or y. When True (nonzero), yield x, otherwise yield y, must have a dtype of bool if used as mask.
        x (Tensor|scalar|None, optional): A Tensor or scalar to choose when the condition is True with data type of bfloat16, float16, float32, float64, int32 or int64. Either both or neither of x and y should be given.
        y (Tensor|scalar|None, optional): A Tensor or scalar to choose when the condition is False with data type of bfloat16, float16, float32, float64, int32 or int64. Either both or neither of x and y should be given.
        name (str|None, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        Tensor, A Tensor with the same shape as :attr:`condition` and same data type as :attr:`x` and :attr:`y`.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([0.9383, 0.1983, 3.2, 1.2])
            >>> y = paddle.to_tensor([1.0, 1.0, 1.0, 1.0])

            >>> out = paddle.where(x>1, x, y)
            >>> print(out)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [1.        , 1.        , 3.20000005, 1.20000005])

            >>> out = paddle.where(x>1)
            >>> print(out)
            (Tensor(shape=[2, 1], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[2],
             [3]]),)
    """
    if np.isscalar(x):
        x = paddle.full([1], x, np.array([x]).dtype.name)

    if np.isscalar(y):
        y = paddle.full([1], y, np.array([y]).dtype.name)

    if x is None and y is None:
        return nonzero(condition, as_tuple=True)

    if x is None or y is None:
        raise ValueError("either both or neither of x and y should be given")

    # NOTE: We might need to adapt the broadcast_shape and broadcast_to for dynamic shape
    # so dynamic and pir branch can be merged into one code block
    condition_shape = list(condition.shape)
    x_shape = list(x.shape)
    y_shape = list(y.shape)

    if in_dynamic_mode():
        # NOTE: `condition` must be a bool Tensor as required in
        # https://data-apis.org/array-api/latest/API_specification/generated/array_api.where.html#array_api.where
        if condition.dtype != paddle.bool:
            raise ValueError(
                "The `condition` is expected to be a boolean Tensor, "
                f"but got a Tensor with dtype {condition.dtype}"
            )
        broadcast_shape = paddle.broadcast_shape(x_shape, y_shape)
        broadcast_shape = paddle.broadcast_shape(
            broadcast_shape, condition_shape
        )

        broadcast_x = x
        broadcast_y = y
        broadcast_condition = condition

        if condition_shape != broadcast_shape:
            broadcast_condition = paddle.broadcast_to(
                broadcast_condition, broadcast_shape
            )
        if x_shape != broadcast_shape:
            broadcast_x = paddle.broadcast_to(broadcast_x, broadcast_shape)
        if y_shape != broadcast_shape:
            broadcast_y = paddle.broadcast_to(broadcast_y, broadcast_shape)

        return _C_ops.where(broadcast_condition, broadcast_x, broadcast_y)

    else:
        # for PIR and old IR
        if x_shape == y_shape and condition_shape == x_shape:
            broadcast_condition = condition
            broadcast_x = x
            broadcast_y = y
        else:
            zeros_like_x = paddle.zeros_like(x)
            zeros_like_y = paddle.zeros_like(y)
            zeros_like_condition = paddle.zeros_like(condition)
            zeros_like_condition = paddle.cast(zeros_like_condition, x.dtype)
            cast_cond = paddle.cast(condition, x.dtype)

            broadcast_zeros = paddle.add(zeros_like_x, zeros_like_y)
            broadcast_zeros = paddle.add(broadcast_zeros, zeros_like_condition)
            broadcast_x = paddle.add(x, broadcast_zeros)
            broadcast_y = paddle.add(y, broadcast_zeros)
            broadcast_condition = paddle.add(cast_cond, broadcast_zeros)
            broadcast_condition = paddle.cast(broadcast_condition, 'bool')

        if in_pir_mode():
            return _C_ops.where(broadcast_condition, broadcast_x, broadcast_y)
        else:
            check_variable_and_dtype(condition, 'condition', ['bool'], 'where')
            check_variable_and_dtype(
                x,
                'x',
                ['uint16', 'float16', 'float32', 'float64', 'int32', 'int64'],
                'where',
            )
            check_variable_and_dtype(
                y,
                'y',
                ['uint16', 'float16', 'float32', 'float64', 'int32', 'int64'],
                'where',
            )
            helper = LayerHelper("where", **locals())
            out = helper.create_variable_for_type_inference(dtype=x.dtype)

            helper.append_op(
                type='where',
                inputs={
                    'Condition': broadcast_condition,
                    'X': broadcast_x,
                    'Y': broadcast_y,
                },
                outputs={'Out': [out]},
            )

            return out


@inplace_apis_in_dygraph_only
def where_(
    condition: Tensor,
    x: Tensor | float | None = None,
    y: Tensor | float | None = None,
    name: str | None = None,
) -> Tensor:
    r"""
    Inplace version of ``where`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_paddle_where`.
    """
    if np.isscalar(x) or np.isscalar(y):
        raise ValueError("either both or neither of x and y should be given")

    if x is None or y is None:
        raise ValueError("either both or neither of x and y should be given")

    # NOTE: `condition` must be a bool Tensor as required in
    # https://data-apis.org/array-api/latest/API_specification/generated/array_api.where.html#array_api.where
    if condition.dtype != paddle.bool:
        raise ValueError(
            "The `condition` is expected to be a boolean Tensor, "
            f"but got a Tensor with dtype {condition.dtype}"
        )

    condition_shape = list(condition.shape)
    x_shape = list(x.shape)
    y_shape = list(y.shape)

    broadcast_shape = paddle.broadcast_shape(x_shape, y_shape)
    broadcast_shape = paddle.broadcast_shape(broadcast_shape, condition_shape)

    broadcast_x = x
    broadcast_y = y
    broadcast_condition = condition

    if condition_shape != broadcast_shape:
        broadcast_condition = paddle.broadcast_to(
            broadcast_condition, broadcast_shape
        )
    if x_shape != broadcast_shape:
        broadcast_x = paddle.broadcast_to(broadcast_x, broadcast_shape)
    if y_shape != broadcast_shape:
        broadcast_y = paddle.broadcast_to(broadcast_y, broadcast_shape)

    if in_dynamic_mode():
        return _C_ops.where_(broadcast_condition, broadcast_x, broadcast_y)


def index_sample(x: Tensor, index: Tensor) -> Tensor:
    """
    **IndexSample Layer**

    IndexSample OP returns the element of the specified location of X,
    and the location is specified by Index.

    .. code-block:: text


                Given:

                X = [[1, 2, 3, 4, 5],
                     [6, 7, 8, 9, 10]]

                Index = [[0, 1, 3],
                         [0, 2, 4]]

                Then:

                Out = [[1, 2, 4],
                       [6, 8, 10]]

    Args:
        x (Tensor): The source input tensor with 2-D shape. Supported data type is
            int32, int64, bfloat16, float16, float32, float64, complex64, complex128.
        index (Tensor): The index input tensor with 2-D shape, first dimension should be same with X.
            Data type is int32 or int64.

    Returns:
        Tensor, The output is a tensor with the same shape as index.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([[1.0, 2.0, 3.0, 4.0],
            ...                       [5.0, 6.0, 7.0, 8.0],
            ...                       [9.0, 10.0, 11.0, 12.0]], dtype='float32')
            >>> index = paddle.to_tensor([[0, 1, 2],
            ...                           [1, 2, 3],
            ...                           [0, 0, 0]], dtype='int32')
            >>> target = paddle.to_tensor([[100, 200, 300, 400],
            ...                            [500, 600, 700, 800],
            ...                            [900, 1000, 1100, 1200]], dtype='int32')
            >>> out_z1 = paddle.index_sample(x, index)
            >>> print(out_z1.numpy())
            [[1. 2. 3.]
             [6. 7. 8.]
             [9. 9. 9.]]

            >>> # Use the index of the maximum value by topk op
            >>> # get the value of the element of the corresponding index in other tensors
            >>> top_value, top_index = paddle.topk(x, k=2)
            >>> out_z2 = paddle.index_sample(target, top_index)
            >>> print(top_value.numpy())
            [[ 4.  3.]
             [ 8.  7.]
             [12. 11.]]

            >>> print(top_index.numpy())
            [[3 2]
             [3 2]
             [3 2]]

            >>> print(out_z2.numpy())
            [[ 400  300]
             [ 800  700]
             [1200 1100]]

    """
    if in_dynamic_or_pir_mode():
        return _C_ops.index_sample(x, index)
    else:
        helper = LayerHelper("index_sample", **locals())
        check_variable_and_dtype(
            x,
            'x',
            [
                'uint16',
                'float16',
                'float32',
                'float64',
                'int32',
                'int64',
                'complex64',
                'complex128',
            ],
            'paddle.tensor.search.index_sample',
        )
        check_variable_and_dtype(
            index,
            'index',
            ['int32', 'int64'],
            'paddle.tensor.search.index_sample',
        )
        out = helper.create_variable_for_type_inference(dtype=x.dtype)

        helper.append_op(
            type='index_sample',
            inputs={'X': x, 'Index': index},
            outputs={'Out': out},
        )
        return out


def masked_select(x: Tensor, mask: Tensor, name: str | None = None) -> Tensor:
    """
    Returns a new 1-D tensor which indexes the input tensor according to the ``mask``
    which is a tensor with data type of bool.

    Note:
        ``paddle.masked_select`` supports broadcasting. If you want know more about broadcasting, please refer to `Introduction to Tensor`_ .

        .. _Introduction to Tensor: ../../guides/beginner/tensor_en.html#chapter5-broadcasting-of-tensor

    Args:
        x (Tensor): The input Tensor, the data type can be int32, int64, uint16, float16, float32, float64.
        mask (Tensor): The Tensor containing the binary mask to index with, it's data type is bool.
        name (str|None, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        Tensor, A 1-D Tensor which is the same data type  as ``x``.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([[1.0, 2.0, 3.0, 4.0],
            ...                       [5.0, 6.0, 7.0, 8.0],
            ...                       [9.0, 10.0, 11.0, 12.0]])
            >>> mask = paddle.to_tensor([[True, False, False, False],
            ...                          [True, True, False, False],
            ...                          [True, False, False, False]])
            >>> out = paddle.masked_select(x, mask)
            >>> print(out.numpy())
            [1. 5. 6. 9.]
    """
    if in_dynamic_or_pir_mode():
        if in_pir_mode():
            check_variable_and_dtype(
                x,
                'x',
                ['float16', 'float32', 'float64', 'int32', 'int64', 'uint16'],
                'paddle.tensor.search.mask_select',
            )
            check_variable_and_dtype(
                mask, 'mask', ['bool'], 'paddle.tensor.search.masked_select'
            )
        return _C_ops.masked_select(x, mask)
    else:
        check_variable_and_dtype(
            x,
            'x',
            ['float16', 'float32', 'float64', 'int32', 'int64', 'uint16'],
            'paddle.tensor.search.mask_select',
        )
        check_variable_and_dtype(
            mask, 'mask', ['bool'], 'paddle.tensor.search.masked_select'
        )
        helper = LayerHelper("masked_select", **locals())
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(
            type='masked_select',
            inputs={'X': x, 'Mask': mask},
            outputs={'Y': out},
        )
        return out


def topk(
    x: Tensor,
    k: int | Tensor,
    axis: int | None = None,
    largest: bool = True,
    sorted: bool = True,
    name: str | None = None,
) -> tuple[Tensor, Tensor]:
    """
    Return values and indices of the k largest or smallest at the optional axis.
    If the input is a 1-D Tensor, finds the k largest or smallest values and indices.
    If the input is a Tensor with higher rank, this operator computes the top k values and indices along the :attr:`axis`.

    Args:
        x (Tensor): Tensor, an input N-D Tensor with type float32, float64, int32, int64.
        k (int, Tensor): The number of top elements to look for along the axis.
        axis (int|None, optional): Axis to compute indices along. The effective range
            is [-R, R), where R is x.ndim. when axis < 0, it works the same way
            as axis + R. Default is -1.
        largest (bool, optional) : largest is a flag, if set to true,
            algorithm will sort by descending order, otherwise sort by
            ascending order. Default is True.
        sorted (bool, optional): controls whether to return the elements in sorted order, default value is True.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        tuple(Tensor), return the values and indices. The value data type is the same as the input `x`. The indices data type is int64.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> data_1 = paddle.to_tensor([1, 4, 5, 7])
            >>> value_1, indices_1 = paddle.topk(data_1, k=1)
            >>> print(value_1)
            Tensor(shape=[1], dtype=int64, place=Place(cpu), stop_gradient=True,
            [7])
            >>> print(indices_1)
            Tensor(shape=[1], dtype=int64, place=Place(cpu), stop_gradient=True,
            [3])

            >>> data_2 = paddle.to_tensor([[1, 4, 5, 7], [2, 6, 2, 5]])
            >>> value_2, indices_2 = paddle.topk(data_2, k=1)
            >>> print(value_2)
            Tensor(shape=[2, 1], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[7],
             [6]])
            >>> print(indices_2)
            Tensor(shape=[2, 1], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[3],
             [1]])

            >>> value_3, indices_3 = paddle.topk(data_2, k=1, axis=-1)
            >>> print(value_3)
            Tensor(shape=[2, 1], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[7],
             [6]])
            >>> print(indices_3)
            Tensor(shape=[2, 1], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[3],
             [1]])

            >>> value_4, indices_4 = paddle.topk(data_2, k=1, axis=0)
            >>> print(value_4)
            Tensor(shape=[1, 4], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[2, 6, 5, 7]])
            >>> print(indices_4)
            Tensor(shape=[1, 4], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[1, 1, 0, 0]])


    """

    if in_dynamic_or_pir_mode():
        if axis is None:
            axis = -1
        out, indices = _C_ops.topk(x, k, axis, largest, sorted)
        return out, indices
    else:
        helper = LayerHelper("top_k_v2", **locals())
        inputs = {"X": [x]}
        attrs = {}
        if isinstance(k, Variable):
            inputs['K'] = [k]
        else:
            attrs = {'k': k}
        attrs['largest'] = largest
        attrs['sorted'] = sorted
        if axis is not None:
            attrs['axis'] = axis

        values = helper.create_variable_for_type_inference(dtype=x.dtype)
        indices = helper.create_variable_for_type_inference(dtype="int64")

        helper.append_op(
            type="top_k_v2",
            inputs=inputs,
            outputs={"Out": [values], "Indices": [indices]},
            attrs=attrs,
        )
        indices.stop_gradient = True
        return values, indices


def bucketize(
    x: Tensor,
    sorted_sequence: Tensor,
    out_int32: bool = False,
    right: bool = False,
    name: str | None = None,
) -> Tensor:
    """
    This API is used to find the index of the corresponding 1D tensor `sorted_sequence` in the innermost dimension based on the given `x`.

    Args:
        x (Tensor): An input N-D tensor value with type int32, int64, float32, float64.
        sorted_sequence (Tensor): An input 1-D tensor with type int32, int64, float32, float64. The value of the tensor monotonically increases in the innermost dimension.
        out_int32 (bool, optional): Data type of the output tensor which can be int32, int64. The default value is False, and it indicates that the output data type is int64.
        right (bool, optional): Find the upper or lower bounds of the sorted_sequence range in the innermost dimension based on the given `x`. If the value of the sorted_sequence is nan or inf, return the size of the innermost dimension.
                               The default value is False and it shows the lower bounds.
        name (str|None, optional): The default value is None. Normally there is no need for user to set this property. For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor (the same sizes of the `x`), return the tensor of int32 if set :attr:`out_int32` is True, otherwise return the tensor of int64.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> sorted_sequence = paddle.to_tensor([2, 4, 8, 16], dtype='int32')
            >>> x = paddle.to_tensor([[0, 8, 4, 16], [-1, 2, 8, 4]], dtype='int32')
            >>> out1 = paddle.bucketize(x, sorted_sequence)
            >>> print(out1)
            Tensor(shape=[2, 4], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[0, 2, 1, 3],
             [0, 0, 2, 1]])
            >>> out2 = paddle.bucketize(x, sorted_sequence, right=True)
            >>> print(out2)
            Tensor(shape=[2, 4], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[0, 3, 2, 4],
             [0, 1, 3, 2]])
            >>> out3 = x.bucketize(sorted_sequence)
            >>> print(out3)
            Tensor(shape=[2, 4], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[0, 2, 1, 3],
             [0, 0, 2, 1]])
            >>> out4 = x.bucketize(sorted_sequence, right=True)
            >>> print(out4)
            Tensor(shape=[2, 4], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[0, 3, 2, 4],
             [0, 1, 3, 2]])

    """
    check_variable_and_dtype(
        sorted_sequence,
        'SortedSequence',
        ['float32', 'float64', 'int32', 'int64'],
        'paddle.searchsorted',
    )
    if sorted_sequence.dim() != 1:
        raise ValueError(
            f"sorted_sequence tensor must be 1 dimension, but got dim {sorted_sequence.dim()}"
        )
    return searchsorted(sorted_sequence, x, out_int32, right, name)


def searchsorted(
    sorted_sequence: Tensor,
    values: Tensor,
    out_int32: bool = False,
    right: bool = False,
    name: str | None = None,
) -> Tensor:
    """
    Find the index of the corresponding `sorted_sequence` in the innermost dimension based on the given `values`.

    Args:
        sorted_sequence (Tensor): An input N-D or 1-D tensor with type int32, int64, float16, float32, float64, bfloat16. The value of the tensor monotonically increases in the innermost dimension.
        values (Tensor): An input N-D tensor value with type int32, int64, float16, float32, float64, bfloat16.
        out_int32 (bool, optional): Data type of the output tensor which can be int32, int64. The default value is False, and it indicates that the output data type is int64.
        right (bool, optional): Find the upper or lower bounds of the sorted_sequence range in the innermost dimension based on the given `values`. If the value of the sorted_sequence is nan or inf, return the size of the innermost dimension.
                               The default value is False and it shows the lower bounds.
        name (str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        Tensor (the same sizes of the `values`), return the tensor of int32 if set :attr:`out_int32` is True, otherwise return the tensor of int64.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> sorted_sequence = paddle.to_tensor([[1, 3, 5, 7, 9, 11],
            ...                                     [2, 4, 6, 8, 10, 12]], dtype='int32')
            >>> values = paddle.to_tensor([[3, 6, 9, 10], [3, 6, 9, 10]], dtype='int32')
            >>> out1 = paddle.searchsorted(sorted_sequence, values)
            >>> print(out1)
            Tensor(shape=[2, 4], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[1, 3, 4, 5],
             [1, 2, 4, 4]])
            >>> out2 = paddle.searchsorted(sorted_sequence, values, right=True)
            >>> print(out2)
            Tensor(shape=[2, 4], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[2, 3, 5, 5],
             [1, 3, 4, 5]])
            >>> sorted_sequence_1d = paddle.to_tensor([1, 3, 5, 7, 9, 11, 13])
            >>> out3 = paddle.searchsorted(sorted_sequence_1d, values)
            >>> print(out3)
            Tensor(shape=[2, 4], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[1, 3, 4, 5],
             [1, 3, 4, 5]])

    """
    if in_dynamic_or_pir_mode():
        return _C_ops.searchsorted(sorted_sequence, values, out_int32, right)
    else:
        check_variable_and_dtype(
            sorted_sequence,
            'SortedSequence',
            ['uint16', 'float16', 'float32', 'float64', 'int32', 'int64'],
            'paddle.searchsorted',
        )
        check_variable_and_dtype(
            values,
            'Values',
            ['uint16', 'float16', 'float32', 'float64', 'int32', 'int64'],
            'paddle.searchsorted',
        )

        helper = LayerHelper('searchsorted', **locals())
        out_type = 'int32' if out_int32 else 'int64'
        out = helper.create_variable_for_type_inference(dtype=out_type)
        helper.append_op(
            type='searchsorted',
            inputs={'SortedSequence': sorted_sequence, "Values": values},
            outputs={'Out': out},
            attrs={"out_int32": out_int32, "right": right},
        )

        return out


def kthvalue(
    x: Tensor,
    k: int,
    axis: int | None = None,
    keepdim: bool = False,
    name: str | None = None,
) -> tuple[Tensor, Tensor]:
    """
    Find values and indices of the k-th smallest at the axis.

    Args:
        x (Tensor): A N-D Tensor with type float16, float32, float64, int32, int64.
        k (int): The k for the k-th smallest number to look for along the axis.
        axis (int, optional): Axis to compute indices along. The effective range
            is [-R, R), where R is x.ndim. when axis < 0, it works the same way
            as axis + R. The default is None. And if the axis is None, it will computed as -1 by default.
        keepdim (bool, optional): Whether to keep the given axis in output. If it is True, the dimensions will be same as input x and with size one in the axis. Otherwise the output dimensions is one fewer than x since the axis is squeezed. Default is False.
        name (str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        tuple(Tensor), return the values and indices. The value data type is the same as the input `x`. The indices data type is int64.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> x = paddle.randn((2,3,2))
            >>> print(x)
            >>> # doctest: +SKIP('Different environments yield different output.')
            Tensor(shape=[2, 3, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[[ 0.11855337, -0.30557564],
              [-0.09968963,  0.41220093],
              [ 1.24004936,  1.50014710]],
             [[ 0.08612321, -0.92485696],
              [-0.09276631,  1.15149164],
              [-1.46587241,  1.22873247]]])
            >>> # doctest: -SKIP
            >>> y = paddle.kthvalue(x, 2, 1)
            >>> print(y)
            >>> # doctest: +SKIP('Different environments yield different output.')
            (Tensor(shape=[2, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[ 0.11855337,  0.41220093],
             [-0.09276631,  1.15149164]]), Tensor(shape=[2, 2], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[0, 1],
             [1, 1]]))
            >>> # doctest: -SKIP
    """
    if in_dynamic_or_pir_mode():
        if axis is not None:
            return _C_ops.kthvalue(x, k, axis, keepdim)
        else:
            return _C_ops.kthvalue(x, k, -1, keepdim)

    helper = LayerHelper("kthvalue", **locals())
    inputs = {"X": [x]}
    attrs = {'k': k}
    if axis is not None:
        attrs['axis'] = axis
    values = helper.create_variable_for_type_inference(dtype=x.dtype)
    indices = helper.create_variable_for_type_inference(dtype="int64")

    helper.append_op(
        type="kthvalue",
        inputs=inputs,
        outputs={"Out": [values], "Indices": [indices]},
        attrs=attrs,
    )
    indices.stop_gradient = True
    return values, indices


def top_p_sampling(
    x: Tensor,
    ps: Tensor,
    threshold: Tensor | None = None,
    topp_seed: Tensor | None = None,
    seed: int = -1,
    k: int = 0,
    mode: Literal['truncated', 'non-truncated'] = "truncated",
    return_top: bool = False,
    name: str | None = None,
) -> tuple[Tensor, Tensor]:
    """
    Get the TopP scores and ids.

    Args:
        x(Tensor): An input 2-D Tensor with type float32, float16 and bfloat16.
        ps(Tensor): A 1-D Tensor with type float32, float16 and bfloat16,
            used to specify the top_p corresponding to each query.
        threshold(Tensor|None, optional): A 1-D Tensor with type float32, float16 and bfloat16,
            used to avoid sampling low score tokens.
        topp_seed(Tensor|None, optional): A 1-D Tensor with type int64,
            used to specify the random seed for each query.
        seed(int, optional): the random seed. Default is -1,
        k(int): the number of top_k scores/ids to be returned. Default is 0.
        mode(str): The mode to choose sampling strategy. If the mode is `truncated`, sampling will truncate the probability at top_p_value.
            If the mode is `non-truncated`, it will not be truncated. Default is `truncated`.
        return_top(bool): Whether to return the top_k scores and ids. Default is False.
        name (str|None, optional): For details, please refer to :ref:`api_guide_Name`.
            Generally, no setting is required. Default: None.

    Returns:
        tuple(Tensor), return the values and indices. The value data type is the same as the input `x`. The indices data type is int64.

    Examples:

        .. code-block:: python

            >>> # doctest: +REQUIRES(env:GPU)
            >>> import paddle

            >>> paddle.device.set_device('gpu')
            >>> paddle.seed(2023)
            >>> x = paddle.randn([2,3])
            >>> print(x)
            Tensor(shape=[2, 3], dtype=float32, place=Place(gpu:0), stop_gradient=True,
             [[-0.32012719, -0.07942779,  0.26011357],
              [ 0.79003978, -0.39958701,  1.42184138]])
            >>> paddle.seed(2023)
            >>> ps = paddle.randn([2])
            >>> print(ps)
            Tensor(shape=[2], dtype=float32, place=Place(gpu:0), stop_gradient=True,
             [-0.32012719, -0.07942779])
            >>> value, index = paddle.tensor.top_p_sampling(x, ps)
            >>> print(value)
            Tensor(shape=[2, 1], dtype=float32, place=Place(gpu:0), stop_gradient=True,
             [[0.26011357],
              [1.42184138]])
            >>> print(index)
            Tensor(shape=[2, 1], dtype=int64, place=Place(gpu:0), stop_gradient=True,
             [[2],
              [2]])
    """

    if in_dynamic_or_pir_mode():
        res = _C_ops.top_p_sampling(x, ps, threshold, topp_seed, seed, k, mode)
        if return_top:
            return res
        else:
            return res[0], res[1]

    inputs = {"x": x, "ps": ps, "threshold": threshold, "topp_seed": topp_seed}
    attrs = {"seed": seed, "k": k, "mode": mode}

    helper = LayerHelper('top_p_sampling', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    ids = helper.create_variable_for_type_inference(dtype="int64")
    topk_scores = helper.create_variable_for_type_inference(dtype=x.dtype)
    topk_ids = helper.create_variable_for_type_inference(dtype="int64")
    helper.append_op(
        type='top_p_sampling',
        inputs=inputs,
        outputs={
            'out': out,
            'ids': ids,
            "topk_scores": topk_scores,
            "topk_ids": topk_ids,
        },
        attrs=attrs,
    )
    if return_top:
        return out, ids, topk_scores, topk_ids
    else:
        return out, ids
