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

import functools
import math
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from typing_extensions import overload

import paddle
from paddle import _C_ops
from paddle.tensor import fill_constant
from paddle.utils.inplace_utils import inplace_apis_in_dygraph_only

from ..base.data_feeder import (
    check_dtype,
    check_type,
    check_variable_and_dtype,
    convert_dtype,
)
from ..base.framework import Variable, default_main_program
from ..framework import (
    LayerHelper,
    convert_np_dtype_to_dtype_,
    core,
    dygraph_only,
    in_dynamic_mode,
    in_dynamic_or_pir_mode,
    in_pir_mode,
)
from .creation import _complex_to_real_dtype, _real_to_complex_dtype, zeros

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from paddle import Tensor
    from paddle._typing import (
        DTypeLike,
        NestedList,
        NestedSequence,
        Numeric,
        ShapeLike,
        TensorOrTensors,
    )

__all__ = []


def tensor_array_to_tensor(
    input: Tensor | list[Tensor],
    axis: int = 1,
    use_stack: bool = False,
    name: str | None = None,
) -> tuple[Tensor, Tensor]:
    r"""
    This function concatenates or stacks all tensors in the input DenseTensorArray
    along the axis mentioned and returns that as the output.

    For Example:

    .. code-block:: text

        Case 1:

            Given:

                input.data = {[[0.6, 0.1, 0.3],
                               [0.5, 0.3, 0.2]],
                              [[1.3],
                               [1.8]],
                              [[2.3, 2.1],
                               [2.5, 2.4]]}

                axis = 1, use_stack = False

            Then:

                output.data = [[0.6, 0.1, 0.3, 1.3, 2.3, 2.1],
                               [0.5, 0.3, 0.2, 1.8, 2.5, 2.4]]

                output_index.data = [3, 1, 2]

        Case 2:

            Given:

                input.data = {[[0.6, 0.1],
                               [0.5, 0.3]],
                              [[0.3, 1.3],
                               [0.2, 1.8]],
                              [[2.3, 2.1],
                               [2.5, 2.4]]}

                axis = 1, use_stack = True

            Then:

                output.data = [[[0.6, 0.1]
                                [0.3, 1.3]
                                [2.3, 2.1],
                               [[0.5, 0.3]
                                [0.2, 1.8]
                                [2.5, 2.4]]]

                output_index.data = [2, 2, 2]

    Args:
        input(Tensor|list[Tensor]): A TensorArray variable.
        axis(int, optional): The axis along which the tensors in attr::`input` will be
            concatenated or stacked.
        use_stack(bool, optional): Act as concat_op or stack_op. For stack mode, all
            tensors in the tensor array must have the same shape.
        name(str|None, optional): A name for this layer(optional). If set None, the layer
                       will be named automatically.

    Returns:
        Tensor: The concatenated or stacked tensor variable.
        Tensor: A 1-D tensor variable with int32 data type. The data in this \
            tensor contains all input including tensors' sizes along the axis.

    Examples:
        .. code-block:: python

            >>> import numpy
            >>> import paddle
            >>> x0 = paddle.assign(numpy.random.rand(2, 2).astype("float32"))
            >>> x1 = paddle.assign(numpy.random.rand(2, 2).astype("float32"))
            >>> i = paddle.full(shape=[1], dtype="int64", fill_value=0)
            >>> array = paddle.tensor.array.create_array(dtype='float32')
            >>> paddle.tensor.array.array_write(x0, i, array)
            >>> paddle.tensor.array.array_write(x1, i + 1, array)
            >>> output, output_index = paddle.tensor.manipulation.tensor_array_to_tensor(input=array)
    """
    if in_dynamic_mode():
        assert isinstance(
            input, list
        ), "The 'input' in tensor_array_to_tensor must be list"
        from paddle import concat, stack

        op = stack if use_stack else concat
        res = op(input, axis=axis)
        sizes = paddle.to_tensor(np.array([int(x.shape[axis]) for x in input]))
        return res, sizes
    elif in_pir_mode():
        check_type(
            input,
            'input',
            (list, paddle.pir.Value),
            'tensor_array_to_tensor',
        )
        if isinstance(input, list):
            for i, input_x in enumerate(input):
                check_type(
                    input_x,
                    'input[' + str(i) + ']',
                    paddle.pir.Value,
                    'tensor_array_to_tensor',
                )
                if not input_x.is_dense_tensor_array_type():
                    raise TypeError("input should be tensor array variable")
        else:
            if not input.is_dense_tensor_array_type():
                raise TypeError("input should be tensor array variable")
        return paddle._pir_ops.array_to_tensor(input, axis, use_stack)
    else:
        check_type(input, 'input', (list, Variable), 'tensor_array_to_tensor')
        if isinstance(input, list):
            for i, input_x in enumerate(input):
                check_type(
                    input_x,
                    'input[' + str(i) + ']',
                    Variable,
                    'tensor_array_to_tensor',
                )
        helper = LayerHelper('tensor_array_to_tensor', **locals())
        out = helper.create_variable_for_type_inference(
            dtype=helper.input_dtype()
        )
        out_index = helper.create_variable_for_type_inference(dtype="int32")
        helper.append_op(
            type='tensor_array_to_tensor',
            inputs={'X': input},
            outputs={'Out': [out], 'OutIndex': [out_index]},
            attrs={'axis': axis, 'use_stack': use_stack},
        )
        return out, out_index


def cast(x: Tensor, dtype: DTypeLike) -> Tensor:
    """

    Take in the Tensor :attr:`x` with :attr:`x.dtype` and cast it
    to the output with :attr:`dtype`. It's meaningless if the output dtype
    equals the input dtype, but it's fine if you do so.

    The following picture shows an example where a tensor of type float64 is cast to a tensor of type uint8.

    .. image:: https://githubraw.cdn.bcebos.com/PaddlePaddle/docs/develop/docs/images/api_legend/cast.png
        :width: 800
        :alt: legend of reshape API
        :align: center

    Args:
        x (Tensor): An input N-D Tensor with data type bool, float16,
            float32, float64, int32, int64, uint8.
        dtype (paddle.dtype|np.dtype|str): Data type of the output:
            bool, float16, float32, float64, int8, int32, int64, uint8.

    Returns:
        Tensor, A Tensor with the same shape as input's.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([2, 3, 4], 'float64')
            >>> y = paddle.cast(x, 'uint8')
    """
    if not isinstance(dtype, (core.VarDesc.VarType, core.DataType)):
        dtype = convert_np_dtype_to_dtype_(dtype)
    if in_dynamic_or_pir_mode():
        return _C_ops.cast(x, dtype)
    else:
        check_variable_and_dtype(
            x,
            'x',
            [
                'bool',
                'float16',
                'float32',
                'float64',
                'int16',
                'int32',
                'int64',
                'uint8',
                'uint16',
                'float8_e4m3fn',
                'float8_e5m2',
            ],
            'cast',
        )
        check_dtype(
            dtype,
            'dtype',
            [
                'bool',
                'float16',
                'float32',
                'float64',
                'int8',
                'int16',
                'int32',
                'int64',
                'uint8',
                'uint16',
                'float8_e4m3fn',
                'float8_e5m2',
            ],
            'cast',
        )

        helper = LayerHelper('cast', **locals())
        out = helper.create_variable_for_type_inference(
            dtype=dtype, stop_gradient=x.stop_gradient
        )
        helper.append_op(
            type='cast',
            inputs={'X': [x]},
            outputs={'Out': [out]},
            attrs={'in_dtype': x.dtype, 'out_dtype': out.dtype},
        )
        return out


@inplace_apis_in_dygraph_only
def cast_(x: Tensor, dtype: DTypeLike) -> Tensor:
    """
    Inplace version of ``cast`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_paddle_cast`.
    """
    if in_dynamic_mode():
        if not isinstance(dtype, (core.VarDesc.VarType, core.DataType)):
            dtype = convert_np_dtype_to_dtype_(dtype)
        return _C_ops.cast_(x, dtype)


def slice(
    input: Tensor,
    axes: Sequence[int | Tensor],
    starts: Sequence[int | Tensor] | Tensor,
    ends: Sequence[int | Tensor] | Tensor,
) -> Tensor:
    """
    This operator produces a slice of ``input`` along multiple axes. Similar to numpy:
    https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
    Slice uses ``axes``, ``starts`` and ``ends`` attributes to specify the start and
    end dimension for each axis in the list of axes and Slice uses this information
    to slice the input data tensor. If a negative value is passed to
    ``starts`` or ``ends`` such as :math:`-i`,  it represents the reverse position of the
    axis :math:`i-1` (here 0 is the initial position).
    If the value passed to ``starts`` or ``ends`` is greater than n
    (the number of elements in this dimension), it represents n.
    For slicing to the end of a dimension with unknown size, it is recommended
    to pass in INT_MAX. The size of ``axes`` must be equal to ``starts`` and ``ends``.
    Following examples will explain how slice works:

    .. code-block:: text

        Case1:
            Given:
                data = [ [1, 2, 3, 4], [5, 6, 7, 8], ]
                axes = [0, 1]
                starts = [1, 0]
                ends = [2, 3]
            Then:
                result = [ [5, 6, 7], ]

        Case2:
            Given:
                data = [ [1, 2, 3, 4], [5, 6, 7, 8], ]
                axes = [0, 1]
                starts = [0, 1]
                ends = [-1, 1000]       # -1 denotes the reverse 0th position of dimension 0.
            Then:
                result = [ [2, 3, 4], ] # result = data[0:1, 1:4]

    The following figure illustrates the first case -- a 2D tensor of shape [2, 4] is transformed into a 2D tensor of shape [1, 3] through a slicing operation.

    .. image:: https://githubraw.cdn.bcebos.com/PaddlePaddle/docs/develop/docs/images/api_legend/slice.png
        :width: 500
        :alt: legend of slice API
        :align: center

    Args:
        input (Tensor): A ``Tensor`` . The data type is ``float16``, ``float32``, ``float64``, ``int32`` or ``int64``.
        axes (list|tuple): The data type is ``int32`` . Axes that `starts` and `ends` apply to .
        starts (list|tuple|Tensor): The data type is ``int32`` . If ``starts`` is a list or tuple, each element of
                it should be integer or 0-D int Tensor with shape []. If ``starts`` is an Tensor, it should be an 1-D Tensor.
                It represents starting indices of corresponding axis in ``axes``.
        ends (list|tuple|Tensor): The data type is ``int32`` . If ``ends`` is a list or tuple, each element of
                it should be integer or 0-D int Tensor with shape []. If ``ends`` is an Tensor, it should be an 1-D Tensor .
                It represents ending indices of corresponding axis in ``axes``.

    Returns:
        Tensor, A ``Tensor``. The data type is same as ``input``.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> input = paddle.rand(shape=[4, 5, 6], dtype='float32')
            >>> # example 1:
            >>> # attr starts is a list which doesn't contain tensor.
            >>> axes = [0, 1, 2]
            >>> starts = [-3, 0, 2]
            >>> ends = [3, 2, 4]
            >>> sliced_1 = paddle.slice(input, axes=axes, starts=starts, ends=ends)
            >>> # sliced_1 is input[1:3, 0:2, 2:4].

            >>> # example 2:
            >>> # attr starts is a list which contain tensor.
            >>> minus_3 = paddle.full([1], -3, "int32")
            >>> sliced_2 = paddle.slice(input, axes=axes, starts=[minus_3, 0, 2], ends=ends)
            >>> # sliced_2 is input[1:3, 0:2, 2:4].
    """
    if isinstance(axes, (list, tuple)):
        axes = list(axes)
        if len(axes) == 0:
            raise ValueError("Input axes should not be an empty list/tuple.")
        for i in range(len(axes)):
            if axes[i] < 0:
                axes[i] = max(0, axes[i] + len(input.shape))
            else:
                axes[i] = min(len(input.shape) - 1, axes[i])

    else:
        raise ValueError(
            f"Input axes must be a python list or tuple, but received {type(axes)}"
        )

    if in_dynamic_mode():
        attrs = ()
        starts_tensor = None
        ends_tensor = None

        infer_flags = [1 for i in range(len(axes))]

        if isinstance(starts, (list, tuple)):
            starts = [
                item.item(0) if isinstance(item, core.eager.Tensor) else item
                for item in starts
            ]
        elif isinstance(starts, core.eager.Tensor):
            tensor_t = starts.numpy(False)
            starts = list(tensor_t)
            infer_flags = [-1 for i in range(len(axes))]

        if isinstance(ends, (list, tuple)):
            ends = [
                item.item(0) if isinstance(item, core.eager.Tensor) else item
                for item in ends
            ]
        elif isinstance(ends, core.eager.Tensor):
            tensor_t = ends.numpy(False)
            ends = list(tensor_t)
            infer_flags = [-1 for i in range(len(axes))]

        return _C_ops.slice(input, axes, starts, ends, infer_flags, [])
    elif in_pir_mode():
        if not isinstance(starts, (list, tuple, paddle.pir.Value)):
            raise ValueError(
                "Input starts must be an Value, python list or tuple."
            )
        if not isinstance(ends, (list, tuple, paddle.pir.Value)):
            raise ValueError(
                "Input ends must be an Value, python list or tuple."
            )
        infer_flags = [1 for i in range(len(axes))]
        # starts
        if isinstance(starts, paddle.pir.Value):
            starts.stop_gradient = True
            infer_flags = [-1 for i in range(len(axes))]
        elif isinstance(starts, (list, tuple)):
            if paddle.utils._contain_var(starts):
                for i, dim in enumerate(starts):
                    if isinstance(dim, paddle.pir.Value):
                        infer_flags[i] = -1
                starts = paddle.utils.get_int_tensor_list(starts)

        # ends
        if isinstance(ends, paddle.pir.Value):
            ends.stop_gradient = True
            infer_flags = [-1 for i in range(len(axes))]
        elif isinstance(ends, (list, tuple)):
            if paddle.utils._contain_var(ends):
                for i, dim in enumerate(ends):
                    if isinstance(dim, paddle.pir.Value):
                        infer_flags[i] = -1
                ends = paddle.utils.get_int_tensor_list(ends)
        return _C_ops.slice(input, axes, starts, ends, infer_flags, [])
    else:
        if not isinstance(starts, (list, tuple, Variable)):
            raise ValueError(
                "Input starts must be an Variable, python list or tuple."
            )
        if not isinstance(ends, (list, tuple, Variable)):
            raise ValueError(
                "Input ends must be an Variable, python list or tuple."
            )

        helper = LayerHelper('slice', **locals())

        inputs = {'Input': input}
        attrs = {'axes': axes}
        infer_flags = [1 for i in range(len(axes))]

        # starts
        if isinstance(starts, Variable):
            starts.stop_gradient = True
            inputs['StartsTensor'] = starts
            infer_flags = [-1 for i in range(len(axes))]
        elif isinstance(starts, (list, tuple)):
            attrs['starts'] = []
            if paddle.utils._contain_var(starts):
                inputs['StartsTensorList'] = (
                    paddle.utils._convert_to_tensor_list(starts)
                )
                for i, dim in enumerate(starts):
                    if isinstance(dim, Variable):
                        attrs['starts'].append(-1)
                        infer_flags[i] = -1
                    else:
                        attrs['starts'].append(dim)
            else:
                attrs['starts'] = starts

        # ends
        if isinstance(ends, Variable):
            ends.stop_gradient = True
            inputs['EndsTensor'] = ends
            infer_flags = [-1 for i in range(len(axes))]
        elif isinstance(ends, (list, tuple)):
            attrs['ends'] = []
            if paddle.utils._contain_var(ends):
                inputs['EndsTensorList'] = paddle.utils._convert_to_tensor_list(
                    ends
                )
                for i, dim in enumerate(ends):
                    if isinstance(dim, Variable):
                        attrs['ends'].append(-1)
                        infer_flags[i] = -1
                    else:
                        attrs['ends'].append(dim)
            else:
                attrs['ends'] = ends

        # infer_flags
        attrs['infer_flags'] = infer_flags
        out = helper.create_variable_for_type_inference(
            dtype=helper.input_dtype('input')
        )
        helper.append_op(
            type='slice', inputs=inputs, attrs=attrs, outputs={'Out': out}
        )

        return out


def transpose(
    x: Tensor, perm: Sequence[int], name: str | None = None
) -> Tensor:
    """
    Permute the data dimensions of `input` according to `perm`.

    The `i`-th dimension  of the returned tensor will correspond to the
    perm[i]-th dimension of `input`.

    The image illustrates the second example of the transpose operation.

    .. image:: https://githubraw.cdn.bcebos.com/PaddlePaddle/docs/develop/docs/images/api_legend/transpose.png
        :width: 500
        :alt: legend of transpose API

    Args:
        x (Tensor): The input Tensor. It is a N-D Tensor of data types bool, float32, float64, int32.
        perm (list|tuple): Permute the input according to the data of perm.
        name (str|None, optional): The name of this layer. It is optional.

    Returns:
        Tensor, A transposed n-D Tensor, with data type being bool, float32, float64, int32, int64.

    For Example:

        .. code-block:: text

         x = [[[ 1  2  3  4] [ 5  6  7  8] [ 9 10 11 12]]
             [[13 14 15 16] [17 18 19 20] [21 22 23 24]]]
         shape(x) =  [2,3,4]

         # Example 1
         perm0 = [1,0,2]
         y_perm0 = [[[ 1  2  3  4] [13 14 15 16]]
                   [[ 5  6  7  8]  [17 18 19 20]]
                   [[ 9 10 11 12]  [21 22 23 24]]]
         shape(y_perm0) = [3,2,4]

         # Example 2
         perm1 = [2,1,0]
         y_perm1 = [[[ 1 13] [ 5 17] [ 9 21]]
                   [[ 2 14] [ 6 18] [10 22]]
                   [[ 3 15]  [ 7 19]  [11 23]]
                   [[ 4 16]  [ 8 20]  [12 24]]]
         shape(y_perm1) = [4,3,2]

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
                'float32',
                'float64',
                'int32',
                'int64',
                'uint16',
                'complex64',
                'complex128',
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


def unstack(x: Tensor, axis: int = 0, num: int | None = None) -> Tensor:
    """
    This layer unstacks input Tensor :code:`x` into several Tensors along :code:`axis`.

    If :code:`axis` < 0, it would be replaced with :code:`axis+rank(x)`.
    If :code:`num` is None, it would be inferred from :code:`x.shape[axis]`,
    and if :code:`x.shape[axis]` <= 0 or is unknown, :code:`ValueError` is
    raised.

    Args:
        x (Tensor): Input Tensor. It is a N-D Tensors of data types float32, float64, int32, int64, complex64, complex128.
        axis (int, optional): The axis along which the input is unstacked.
        num (int|None, optional): The number of output variables.

    Returns:
        list(Tensor), The unstacked Tensors list. The list elements are N-D Tensors of data types float32, float64, int32, int64, complex64, complex128.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> x = paddle.ones(name='x', shape=[2, 3, 5], dtype='float32')  # create a tensor with shape=[2, 3, 5]
            >>> y = paddle.unstack(x, axis=1)  # unstack with second axis, which results 3 tensors with shape=[2, 5]

    """
    if not (-x.ndim <= axis < x.ndim):
        raise ValueError(f'`axis` must be in the range [-{x.ndim}, {x.ndim})')
    if num is not None and (num < 0 or num > x.shape[axis]):
        raise ValueError(f'`num` must be in the range [0, {x.shape[axis]})')
    if in_dynamic_or_pir_mode():
        if num is None:
            num = x.shape[axis]
        if num == 0:
            return []
        return _C_ops.unstack(x, axis, num)
    else:
        helper = LayerHelper('unstack', **locals())
        if num is None:
            if axis is None or x.shape[axis] <= 0:
                raise ValueError('unknown unstack number')
            else:
                num = x.shape[axis]

        outs = []
        for _ in range(num):
            outs.append(helper.create_variable_for_type_inference(x.dtype))

        helper.append_op(
            type='unstack',
            inputs={'X': [x]},
            outputs={'Y': outs},
            attrs={'axis': axis, 'num': num},
        )
        return outs


def shard_index(
    input: Tensor,
    index_num: int,
    nshards: int,
    shard_id: int,
    ignore_value: int = -1,
) -> Tensor:
    """
    Reset the values of `input` according to the shard it belongs to.
    Every value in `input` must be a non-negative integer, and
    the parameter `index_num` represents the integer above the maximum
    value of `input`. Thus, all values in `input` must be in the range
    [0, index_num) and each value can be regarded as the offset to the beginning
    of the range. The range is further split into multiple shards. Specifically,
    we first compute the `shard_size` according to the following formula,
    which represents the number of integers each shard can hold. So for the
    i'th shard, it can hold values in the range [i*shard_size, (i+1)*shard_size).
    ::

        shard_size = (index_num + nshards - 1) // nshards

    For each value `v` in `input`, we reset it to a new value according to the
    following formula:
    ::

        v = v - shard_id * shard_size if shard_id * shard_size <= v < (shard_id+1) * shard_size else ignore_value

    That is, the value `v` is set to the new offset within the range represented by the shard `shard_id`
    if it in the range. Otherwise, we reset it to be `ignore_value`.

    As shown below, a ``[2, 1]`` 2D tensor is updated with the ``shard_index`` operation. Given ``index_num = 20``, ``nshards = 2``, and ``shard_id = 0``, the shard size is ``shard_size = (20 + 2 - 1) // 2 = 10``.
    For each label element: if its value is in [0, 10), it's adjusted to its offset; e.g., 1 becomes 1 - 0 * 10 = 1. Otherwise, it's set to the default ignore_value of -1, like 16 becoming -1.

    .. image:: https://githubraw.cdn.bcebos.com/PaddlePaddle/docs/develop/docs/images/api_legend/shard_index.png
       :width: 500
       :alt: Illustration of Case 2
       :align: center

    Args:
        input (Tensor): Input tensor with data type int64 or int32. It's last dimension must be 1.
        index_num (int): An integer represents the integer above the maximum value of `input`.
        nshards (int): The number of shards.
        shard_id (int): The index of the current shard.
        ignore_value (int, optional): An integer value out of sharded index range. The default value is -1.

    Returns:
        Tensor.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> label = paddle.to_tensor([[16], [1]], "int64")
            >>> shard_label = paddle.shard_index(input=label,
            ...                                  index_num=20,
            ...                                  nshards=2,
            ...                                  shard_id=0)
            >>> print(shard_label.numpy())
            [[-1]
             [ 1]]


    """
    if in_dynamic_or_pir_mode():
        return _C_ops.shard_index(
            input, index_num, nshards, shard_id, ignore_value
        )

    check_variable_and_dtype(input, 'input', ['int64', 'int32'], 'shard_index')
    op_type = 'shard_index'
    helper = LayerHelper(op_type, **locals())
    if shard_id < 0 or shard_id >= nshards:
        raise ValueError(
            f'The shard_id({shard_id}) should be in [0, {nshards})'
        )

    out = helper.create_variable_for_type_inference(dtype=input.dtype)
    helper.append_op(
        type=op_type,
        inputs={'X': [input]},
        outputs={'Out': out},
        attrs={
            'index_num': index_num,
            'nshards': nshards,
            'shard_id': shard_id,
            'ignore_value': ignore_value,
        },
        stop_gradient=True,
    )
    return out


def crop(
    x: Tensor,
    shape: ShapeLike | None = None,
    offsets: Sequence[int] | Tensor | None = None,
    name: str | None = None,
) -> Tensor:
    """
    Crop input into output, as specified by offsets and shape.

    .. code-block:: text

        * Case 1 (input is a 2-D Tensor):
            Input:
                X.shape = [3, 5]
                X.data = [[0, 1, 2, 0, 0],
                          [0, 3, 4, 0, 0],
                          [0, 0, 0, 0, 0]]
            Parameters:
                shape = [2, 2]
                offsets = [0, 1]
            Output:
                Out.shape = [2, 2]
                Out.data = [[1, 2],
                            [3, 4]]
        * Case 2 (input is a 3-D Tensor):
            Input:
                X.shape = [2, 3, 4]
                X.data =  [[[0, 1, 2, 3],
                            [0, 5, 6, 7],
                            [0, 0, 0, 0]],
                           [[0, 3, 4, 5],
                            [0, 6, 7, 8],
                            [0, 0, 0, 0]]]
            Parameters:
                shape = [2, 2, -1]
                offsets = [0, 0, 1]
            Output:
                Out.shape = [2, 2, 3]
                Out.data  = [[[1, 2, 3],
                              [5, 6, 7]],
                             [[3, 4, 5],
                              [6, 7, 8]]]

    The image below demonstrates the  Case 2 that a 3D tensor with shape [2,3,4] is cropped into a 3D tensor with shape [2,2,3]

    .. image:: https://githubraw.cdn.bcebos.com/PaddlePaddle/docs/develop/docs/images/api_legend/crop.png
       :width: 500
       :alt: Illustration of Case 2
       :align: center

    Parameters:
        x (Tensor): 1-D to 6-D Tensor, the data type is float32, float64, int32 or int64.
        shape (list|tuple|Tensor, optional): The output shape is specified
            by `shape`. Its data type is int32. If a list/tuple, it's length must be
            the same as the dimension size of `x`. If a Tensor, it should be a 1-D Tensor.
            When it is a list, each element can be an integer or a Tensor of shape: [1].
            If Variable contained, it is suitable for the case that the shape may
            be changed each iteration.
        offsets (list|tuple|Tensor, optional): Specifies the cropping
            offsets at each dimension. Its data type is int32. If a list/tuple, it's length
            must be the same as the dimension size of `x`. If a Tensor, it should be a 1-D
            Tensor. When it is a list, each element can be an integer or a Tensor of shape: [1].
            If Variable contained, it is suitable for the case that the offsets may be changed
            each iteration. Default: None, the offsets are 0 at each dimension.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, The cropped Tensor has same data type with `x`.

    Examples:

        .. code-block:: python

            >>> import paddle
            >>> x = paddle.to_tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
            >>> # x.shape = [3, 3]
            >>> # x = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

            >>> # shape can be a 1-D Tensor or list or tuple.
            >>> shape = paddle.to_tensor([2, 2], dtype='int32')
            >>> # shape = [2, 2]
            >>> # shape = (2, 2)
            >>> out = paddle.crop(x, shape)
            >>> # out.shape = [2, 2]
            >>> # out = [[1,2], [4,5]]

            >>> # offsets can be a 1-D Tensor or list or tuple.
            >>> offsets = paddle.to_tensor([0, 1], dtype='int32')
            >>> # offsets = [1, 0]
            >>> # offsets = (1, 1)
            >>> out = paddle.crop(x, shape, offsets)
            >>> # out.shape = [2, 2]
            >>> # if offsets = [0, 0], out = [[1,2], [4,5]]
            >>> # if offsets = [0, 1], out = [[2,3], [5,6]]
            >>> # if offsets = [1, 0], out = [[4,5], [7,8]]
            >>> # if offsets = [1, 1], out = [[5,6], [8,9]]

    """

    helper = LayerHelper('crop_tensor', **locals())
    check_variable_and_dtype(
        x, 'x', ['float32', 'float64', 'int32', 'int64'], 'crop_tensor'
    )
    check_type(
        shape,
        'shape',
        (list, tuple, Variable, type(None), paddle.pir.Value),
        'crop_tensor',
    )
    check_type(
        offsets,
        'offsets',
        (list, tuple, Variable, type(None), paddle.pir.Value),
        'crop_tensor',
    )

    if offsets is None:
        offsets = [0] * len(x.shape)

    if shape is None:
        shape = x.shape

    if in_dynamic_mode():
        return _C_ops.crop(x, shape, offsets)

    def _attr_shape_check(shape_val):
        if not isinstance(shape_val, int):
            raise TypeError(
                f"Attr(shape)'s dtype of Op(crop_tensor) should be int32, but received: {type(shape_val)}."
            )
        if shape_val == 0:
            raise ValueError(
                f"Attr(shape) of Op(crop_tensor) should not be zero, but received: {shape_val}."
            )
        if shape_val < -1:
            raise ValueError(
                f"When the element in Attr(shape) of Op(crop_tensor) is negative, only -1 is supported, but received: {shape_val}."
            )

    def _attr_offsets_check(offset_val):
        if not isinstance(offset_val, int):
            raise TypeError(
                f"Attr(offsets)'s dtype of Op(crop_tensor) should be int32, but received: {type(offset_val)}."
            )
        if offset_val < 0:
            raise ValueError(
                f"Attr(offsets) of Op(crop_tensor) should be greater or equal to zero, but received: {offset_val}."
            )

    if in_pir_mode():
        if isinstance(offsets, paddle.pir.Value):
            offsets.stop_gradient = True
        elif paddle.utils._contain_var(offsets):
            new_offsets_tensor = []
            for dim in offsets:
                if isinstance(dim, paddle.pir.Value):
                    dim.stop_gradient = True
                    new_offsets_tensor.append(dim)
                else:
                    _attr_offsets_check(dim)
                    temp_out = fill_constant([1], 'int32', dim, force_cpu=True)
                    new_offsets_tensor.append(temp_out)
            offsets = new_offsets_tensor
        else:
            for offset in offsets:
                _attr_offsets_check(offset)

        if isinstance(shape, paddle.pir.Value):
            shape.stop_gradient = True
        elif paddle.utils._contain_var(shape):
            new_shape_tensor = []
            for dim_size in shape:
                if isinstance(dim_size, paddle.pir.Value):
                    dim_size.stop_gradient = True
                    new_shape_tensor.append(dim_size)
                else:
                    _attr_shape_check(dim_size)
                    temp_out = fill_constant(
                        [1], 'int32', dim_size, force_cpu=True
                    )
                    new_shape_tensor.append(temp_out)
            shape = new_shape_tensor
        else:
            for dim_size in shape:
                _attr_shape_check(dim_size)
        return _C_ops.crop(x, shape, offsets)

    out = helper.create_variable_for_type_inference(x.dtype)
    ipts = {'X': x}
    attrs = {}

    if isinstance(offsets, Variable):
        offsets.stop_gradient = True
        ipts['Offsets'] = offsets
        attrs['offsets'] = [-1] * len(x.shape)
    elif paddle.utils._contain_var(offsets):
        new_offsets_tensor = []
        offsets_attr = []
        for dim in offsets:
            if isinstance(dim, Variable):
                dim.stop_gradient = True
                new_offsets_tensor.append(dim)
                offsets_attr.append(-1)
            else:
                _attr_offsets_check(dim)
                temp_out = helper.create_variable_for_type_inference('int32')
                fill_constant([1], 'int32', dim, force_cpu=True, out=temp_out)
                new_offsets_tensor.append(temp_out)
                offsets_attr.append(dim)
        ipts['OffsetsTensor'] = new_offsets_tensor
        attrs['offsets'] = offsets_attr
    else:
        for offset in offsets:
            _attr_offsets_check(offset)
        attrs['offsets'] = offsets

    if isinstance(shape, Variable):
        shape.stop_gradient = True
        ipts['Shape'] = shape
    elif paddle.utils._contain_var(shape):
        new_shape_tensor = []
        shape_attr = []
        for dim_size in shape:
            if isinstance(dim_size, Variable):
                dim_size.stop_gradient = True
                new_shape_tensor.append(dim_size)
                shape_attr.append(0)
            else:
                _attr_shape_check(dim_size)
                temp_out = helper.create_variable_for_type_inference('int32')
                fill_constant(
                    [1], 'int32', dim_size, force_cpu=True, out=temp_out
                )
                new_shape_tensor.append(temp_out)
                shape_attr.append(dim_size)
        ipts['ShapeTensor'] = new_shape_tensor
        attrs['shape'] = shape_attr
    else:
        for dim_size in shape:
            _attr_shape_check(dim_size)
        attrs['shape'] = shape

    helper.append_op(
        type='crop_tensor',
        inputs=ipts,
        outputs={'Out': out},
        attrs=None if len(attrs) == 0 else attrs,
    )
    return out


@dygraph_only
def fill_(x: Tensor, value: float) -> Tensor:
    """
    **Notes**:
        **This API is ONLY available in Dygraph mode**

    This function fill the Tensor with value inplace.

    Args:
        x (Tensor): ``x`` is the Tensor we want to filled data inplace
        value (int|float): ``value`` is the value to be filled in x

    Returns:
        x(Tensor), Tensor x filled with value inplace

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> tensor = paddle.to_tensor([0, 1, 2, 3, 4])

            >>> tensor.fill_(0)
            >>> print(tensor.tolist())
            [0, 0, 0, 0, 0]

    """
    if not isinstance(value, (float, int)):
        raise TypeError(
            f"The type of 'value'  must be int or float, but received {type(value)}."
        )
    return _C_ops.fill_(x, value)


@dygraph_only
def zero_(x: Tensor) -> Tensor:
    """
    **Notes**:
        **This API is ONLY available in Dygraph mode**

    This function fill the Tensor with zero inplace.

    Args:
        x (Tensor): ``x`` is the Tensor we want to filled with zero inplace

    Returns:
        x (Tensor), Tensor x filled with zero inplace

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> tensor = paddle.to_tensor([0, 1, 2, 3, 4])

            >>> tensor.zero_()
            >>> print(tensor.tolist())
            [0, 0, 0, 0, 0]

    """
    return _C_ops.fill_(x, 0.0)


@dygraph_only
def fill_diagonal_(
    x: Tensor,
    value: float,
    offset: int = 0,
    wrap: bool = False,
    name: str | None = None,
) -> Tensor:
    """
    Note:
        This API is ONLY available in Dygraph mode.

    This function fill the value into the x Tensor's diagonal inplace.

    Args:
        x(Tensor): ``x`` is the original Tensor
        value(int|float): ``value`` is the value to filled in x
        offset(int,optional): the offset to the main diagonal. Default: 0 (main diagonal).
        wrap(bool,optional): the diagonal 'wrapped' after N columns for tall matrices.
        name(str|None,optional): Name for the operation (optional, default is None)

    Returns:
        Tensor, Tensor with diagonal filled with value.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> x = paddle.ones((4, 3)) * 2
            >>> x.fill_diagonal_(1.0)
            >>> print(x.tolist())
            [[1.0, 2.0, 2.0], [2.0, 1.0, 2.0], [2.0, 2.0, 1.0], [2.0, 2.0, 2.0]]
    """
    if in_dynamic_mode():
        if len(x.shape) == 2:
            return _C_ops.fill_diagonal_(x, value, offset, wrap)
        return _C_ops.fill_diagonal_(x, value, offset, True)


def _fill_diagonal_tensor_impl(
    x: Tensor,
    y: Tensor,
    offset: int = 0,
    dim1: int = 0,
    dim2: int = 1,
    inplace: bool = False,
) -> Tensor:
    inshape = x.shape
    assert dim1 < len(inshape) and dim1 >= -len(
        inshape
    ), 'dim1 should between [-rank,rank) in fill_diagonal_tensor_'
    assert dim2 < len(inshape) and dim2 >= -len(
        inshape
    ), 'dim2 should between [-rank,rank) in fill_diagonal_tensor_'
    assert len(inshape) >= 2, 'Tensor dims should >= 2 in fill_diagonal_tensor_'
    dim1 %= len(inshape)
    dim2 %= len(inshape)

    predshape = []
    for i in range(len(inshape)):
        if i != dim1 and i != dim2:
            predshape.append(inshape[i])
    diaglen = min(
        inshape[dim1],
        inshape[dim1] + offset,
        inshape[dim2],
        inshape[dim2] - offset,
    )
    predshape.append(diaglen)
    assert tuple(predshape) == tuple(
        y.shape
    ), f"the y shape should be {predshape}"
    if len(y.shape) == 1:
        y = y.reshape([1, -1])

    if in_dynamic_or_pir_mode():
        if inplace:
            return _C_ops.fill_diagonal_tensor_(x, y, offset, dim1, dim2)
        else:
            return _C_ops.fill_diagonal_tensor(x, y, offset, dim1, dim2)
    else:
        check_variable_and_dtype(
            x,
            'X',
            [
                'float16',
                'float32',
                'float64',
                'uint16',
                'uint8',
                'int8',
                'int16',
                'int32',
                'int64',
                'bool',
                'complex64',
                'complex128',
            ],
            'paddle.tensor.manipulation.fill_diagonal_tensor',
        )
        check_variable_and_dtype(
            y,
            'Y',
            [
                'float16',
                'float32',
                'float64',
                'uint16',
                'uint8',
                'int8',
                'int16',
                'int32',
                'int64',
                'bool',
                'complex64',
                'complex128',
            ],
            'paddle.tensor.manipulation.fill_diagonal_tensor',
        )
        helper = LayerHelper('fill_diagonal_tensor', **locals())
        out = helper.create_variable_for_type_inference(x.dtype)
        helper.append_op(
            type='fill_diagonal_tensor',
            inputs={
                'X': x,
                'Y': y,
            },
            outputs={'Out': out},
            attrs={
                'offset': offset,
                'dim1': dim1,
                'dim2': dim2,
            },
        )
        return out


def fill_diagonal_tensor_(
    x: Tensor,
    y: Tensor,
    offset: int = 0,
    dim1: int = 0,
    dim2: int = 1,
    name: str | None = None,
) -> Tensor:
    """
    Note:
        This API is ONLY available in Dygraph mode.

    This function fill the source Tensor y into the x Tensor's diagonal inplace.

    Args:
        x (Tensor): ``x`` is the original Tensor
        y (Tensor): ``y`` is the Tensor to filled in x
        dim1 (int,optional): first dimension with respect to which to fill diagonal. Default: 0.
        dim2 (int,optional): second dimension with respect to which to fill diagonal. Default: 1.
        offset (int,optional): the offset to the main diagonal. Default: 0 (main diagonal).
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, Tensor with diagonal filled with y.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.ones((4, 3)) * 2
            >>> y = paddle.ones((3,))
            >>> x.fill_diagonal_tensor_(y)
            >>> print(x.tolist())
            [[1.0, 2.0, 2.0], [2.0, 1.0, 2.0], [2.0, 2.0, 1.0], [2.0, 2.0, 2.0]]

    """
    return _fill_diagonal_tensor_impl(
        x, y, offset=offset, dim1=dim1, dim2=dim2, inplace=True
    )


def fill_diagonal_tensor(
    x: Tensor,
    y: Tensor,
    offset: int = 0,
    dim1: int = 0,
    dim2: int = 1,
    name: str | None = None,
) -> Tensor:
    """
    This function fill the source Tensor y into the x Tensor's diagonal.

    Args:
        x (Tensor): ``x`` is the original Tensor
        y (Tensor): ``y`` is the Tensor to filled in x
        dim1 (int,optional): first dimension with respect to which to fill diagonal. Default: 0.
        dim2 (int,optional): second dimension with respect to which to fill diagonal. Default: 1.
        offset (int,optional): the offset to the main diagonal. Default: 0 (main diagonal).
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, Tensor with diagonal filled with y.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.ones((4, 3)) * 2
            >>> y = paddle.ones((3,))
            >>> nx = x.fill_diagonal_tensor(y)
            >>> print(nx.tolist())
            [[1.0, 2.0, 2.0], [2.0, 1.0, 2.0], [2.0, 2.0, 1.0], [2.0, 2.0, 2.0]]

    """
    return _fill_diagonal_tensor_impl(
        x, y, offset=offset, dim1=dim1, dim2=dim2, inplace=False
    )


@dygraph_only
def tolist(x: Tensor) -> NestedList[int | float | complex]:
    """
    Note:
        This API is ONLY available in Dygraph mode.

    This function translate the paddle.Tensor to python list.

    Args:
        x (Tensor): ``x`` is the Tensor we want to translate to list.

    Returns:
        list, A list that contain the same value of current Tensor.


    Examples:
        .. code-block:: python

            >>> import paddle

            >>> t = paddle.to_tensor([0,1,2,3,4])
            >>> expectlist = t.tolist()
            >>> print(expectlist)
            [0, 1, 2, 3, 4]

            >>> expectlist = paddle.tolist(t)
            >>> print(expectlist)
            [0, 1, 2, 3, 4]

    """
    # TODO(zhouwei): will remove 0-D Tensor.numpy() hack
    return x.numpy(False).tolist()


def concat(
    x: Sequence[Tensor], axis: int | Tensor = 0, name: str | None = None
) -> Tensor:
    """

    Concatenates the input along the axis. It doesn't support 0-D Tensor because it requires a certain axis, and 0-D Tensor
    doesn't have any axis.

    The image illustrates a typical case of the concat operation.
    Two three-dimensional tensors with shapes [2, 3, 4] are concatenated along different axes, resulting in tensors of different shapes.
    The effects of concatenation along various dimensions are clearly visible.

    .. image:: https://githubraw.cdn.bcebos.com/PaddlePaddle/docs/develop/docs/images/api_legend/concat.png
        :width: 500
        :alt: legend of concat API
        :align: center

    Args:
        x (list|tuple): ``x`` is a Tensor list or Tensor tuple which is with data type bool, float16, bfloat16,
            float32, float64, int8, int16, int32, int64, uint8, uint16, complex64, complex128. All the Tensors in ``x`` must have same data type.
        axis (int|Tensor, optional): Specify the axis to operate on the input Tensors.
            Tt should be integer or 0-D int Tensor with shape []. The effective range is [-R, R), where R is Rank(x). When ``axis < 0``,
            it works the same way as ``axis+R``. Default is 0.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, A Tensor with the same data type as ``x``.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x1 = paddle.to_tensor([[1, 2, 3],
            ...                        [4, 5, 6]])
            >>> x2 = paddle.to_tensor([[11, 12, 13],
            ...                        [14, 15, 16]])
            >>> x3 = paddle.to_tensor([[21, 22],
            ...                        [23, 24]])
            >>> zero = paddle.full(shape=[1], dtype='int32', fill_value=0)
            >>> # When the axis is negative, the real axis is (axis + Rank(x))
            >>> # As follow, axis is -1, Rank(x) is 2, the real axis is 1
            >>> out1 = paddle.concat(x=[x1, x2, x3], axis=-1)
            >>> out2 = paddle.concat(x=[x1, x2], axis=0)
            >>> out3 = paddle.concat(x=[x1, x2], axis=zero)
            >>> print(out1)
            Tensor(shape=[2, 8], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[1 , 2 , 3 , 11, 12, 13, 21, 22],
             [4 , 5 , 6 , 14, 15, 16, 23, 24]])
            >>> print(out2)
            Tensor(shape=[4, 3], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[1 , 2 , 3 ],
             [4 , 5 , 6 ],
             [11, 12, 13],
             [14, 15, 16]])
            >>> print(out3)
            Tensor(shape=[4, 3], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[1 , 2 , 3 ],
             [4 , 5 , 6 ],
             [11, 12, 13],
             [14, 15, 16]])
    """
    input = x
    if in_dynamic_mode():
        if isinstance(axis, Variable):
            axis = axis.item(0)
        if not isinstance(input, (Variable, paddle.pir.Value)):
            input = [t for t in input if t.shape.count(0) == 0]
        return _C_ops.concat(input, axis)
    elif in_pir_mode():

        def is_in_amp_mode():
            amp_attrs = core._get_amp_attrs()
            amp_level = amp_attrs._amp_level
            apply_amp_level_list = [
                core.AmpLevel.O1,
                core.AmpLevel.O2,
            ]
            return amp_level in apply_amp_level_list

        is_in_amp = is_in_amp_mode()
        check_type(input, 'input', (list, tuple, paddle.pir.Value), 'concat')
        if not isinstance(input, paddle.pir.Value):
            for id, x in enumerate(input):
                check_variable_and_dtype(
                    x,
                    'input[' + str(id) + ']',
                    [
                        'bool',
                        'float16',
                        'bfloat16',
                        'float32',
                        'float64',
                        'int16',
                        'int32',
                        'int64',
                        'int8',
                        'unit8',
                        'uint16',
                        'complex64',
                        'complex128',
                    ],
                    'concat',
                )
                if (not is_in_amp) and x.dtype != input[0].dtype:
                    raise TypeError(
                        "All the Tensors in the input must have the same data type."
                    )
        elif (
            isinstance(input, paddle.pir.Value)
            and input.is_dense_tensor_array_type()
        ):
            out, _ = _C_ops.array_to_tensor(input, axis, False)
            return out
        else:
            input = [input]

        if not isinstance(input, paddle.pir.Value):
            input = [t for t in input if t.shape.count(0) == 0]
        return _C_ops.concat(input, axis)
    else:
        check_type(input, 'input', (list, tuple, Variable), 'concat')
        if not isinstance(input, Variable):
            for id, x in enumerate(input):
                check_variable_and_dtype(
                    x,
                    'input[' + str(id) + ']',
                    [
                        'bool',
                        'float16',
                        'bfloat16',
                        'float32',
                        'float64',
                        'int16',
                        'int32',
                        'int64',
                        'int8',
                        'unit8',
                        'uint16',
                        'complex64',
                        'complex128',
                    ],
                    'concat',
                )
                if x.dtype != input[0].dtype:
                    raise TypeError(
                        "All the Tensors in the input must have the same data type."
                    )
        else:
            input = [input]
        check_type(axis, 'axis', (int, Variable), 'concat')

        if isinstance(axis, Variable):
            check_dtype(
                axis.dtype,
                'axis',
                ['int32', 'int64'],
                'concat',
                "The data type of axis must be int32 or int64 when axis is a Tensor",
            )

        helper = LayerHelper('concat', **locals())
        out = helper.create_variable_for_type_inference(
            dtype=helper.input_dtype()
        )

        if input[0].desc.type() == core.VarDesc.VarType.DENSE_TENSOR_ARRAY:
            # NOTE(liym27): Don't remove this if branch!
            # This feature is supported for Dynamic-to-Static, because after transformed, the type of inputs[0]
            # is DENSE_TENSOR_ARRAY in some scenarios. And this feature can be used in static graph mode.

            assert len(input) == 1, (
                "If the elements of 'input' in concat are Variable(DenseTensorArray), "
                f"number of the elements must be 1, but received {len(input)}."
            )
            out_index = helper.create_variable_for_type_inference(dtype="int32")
            helper.append_op(
                type='tensor_array_to_tensor',
                inputs={'X': input[0]},
                outputs={'Out': [out], 'OutIndex': [out_index]},
                attrs={'axis': axis, 'use_stack': False},
            )
        else:
            inputs = {'X': input}
            attrs = {}
            if isinstance(axis, Variable):
                axis.stop_gradient = True
                inputs['AxisTensor'] = axis
            else:
                attrs['axis'] = axis

            helper.append_op(
                type='concat',
                inputs=inputs,
                outputs={'Out': [out]},
                attrs=attrs,
            )
        return out


def broadcast_tensors(
    input: Sequence[Tensor], name: str | None = None
) -> list[Tensor]:
    """
    Broadcast a list of tensors following broadcast semantics

    Note:
        If you want know more about broadcasting, please refer to `Introduction to Tensor`_ .

        .. _Introduction to Tensor: ../../guides/beginner/tensor_en.html#chapter5-broadcasting-of-tensor

    The following figure illustrates the process of broadcasting three tensors to the same dimensions.
    The dimensions of the three tensors are [4, 1, 3], [2, 3], and [4, 2, 1], respectively. During broadcasting,
    alignment starts from the last dimension, and for each dimension, either the sizes of the two tensors in that dimension are equal,
    or one of the tensors has a dimension of 1, or one of the tensors lacks that dimension. In the figure below, in the last dimension,
    Tensor3 has a size of 1, while Tensor1 and Tensor2 have sizes of 3; thus, this dimension is expanded to 3 for all tensors.
    In the second-to-last dimension, Tensor1 has a size of 2, and Tensor2 and Tensor3 both have sizes of 2; hence, this dimension is expanded to 2 for all tensors.
    In the third-to-last dimension, Tensor2 lacks this dimension, while Tensor1 and Tensor3 have sizes of 4; consequently,
    this dimension is expanded to 4 for all tensors. Ultimately, all tensors are expanded to [4, 2, 3].

    .. image:: https://githubraw.cdn.bcebos.com/PaddlePaddle/docs/develop/docs/images/api_legend/broadcast.png
       :width: 800
       :alt: Illustration of BroadCast
       :align: center

    Args:
        input (list|tuple): ``input`` is a Tensor list or Tensor tuple which is with data type bool,
            float16, float32, float64, int32, int64, complex64, complex128. All the Tensors in ``input`` must have same data type.
            Currently we only support tensors with rank no greater than 5.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        list(Tensor), The list of broadcasted tensors following the same order as ``input``.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> x1 = paddle.rand([1, 2, 3, 4]).astype('float32')
            >>> x2 = paddle.rand([1, 2, 1, 4]).astype('float32')
            >>> x3 = paddle.rand([1, 1, 3, 1]).astype('float32')
            >>> out1, out2, out3 = paddle.broadcast_tensors(input=[x1, x2, x3])
            >>> # out1, out2, out3: tensors broadcasted from x1, x2, x3 with shape [1,2,3,4]
    """

    num_inputs = len(input)
    if in_dynamic_or_pir_mode():
        return _C_ops.broadcast_tensors(input)
    else:
        check_type(input, 'input', (list, tuple), 'broadcast_tensors')
        if num_inputs < 1:
            raise TypeError(
                "At least 1 tensor is needed to perform broadcast_tensors"
            )

        # Check input types
        for id, x in enumerate(input):
            check_variable_and_dtype(
                x,
                'input[' + str(id) + ']',
                [
                    'bool',
                    'float16',
                    'float32',
                    'float64',
                    'int32',
                    'int64',
                    'uint16',
                    'complex64',
                    'complex128',
                ],
                'broadcast_tensors',
            )
            if x.dtype != input[0].dtype:
                raise TypeError(
                    "All the Tensors in the input must have the same data type."
                )

        # Check bcast semantics
        output_shape_r_last_tensor_index = []
        output_shape_r = []

        # Use while loop due to weird behaviour of "range()"
        j = 0
        while j < len(input):
            tensor = input[j]
            shape = list(reversed(tensor.shape))

            i = 0
            while i < len(shape):
                if len(output_shape_r) <= i:
                    output_shape_r.append(shape[i])
                    output_shape_r_last_tensor_index.append(j)
                else:
                    invalid = (
                        output_shape_r[i] != shape[i]
                        and output_shape_r[i] != 1
                        and shape[i] != 1
                    )
                    if invalid:
                        last_index = output_shape_r_last_tensor_index[i]
                        raise TypeError(
                            "Input tensors to broadcast_tensors does not follow bcast semantics"
                            f"Tensor {last_index} conflicts with Tensor {j} in reversed dimension {i}"
                        )
                    if output_shape_r[i] <= shape[i]:
                        output_shape_r[i] = shape[i]
                        output_shape_r_last_tensor_index[i] = j
                i += 1  # while i < len(shape)
            j += 1  # while j < len(input)

        helper = LayerHelper('broadcast_tensors', **locals())
        i = 0
        out = []
        while i < num_inputs:
            out.append(
                helper.create_variable_for_type_inference(
                    dtype=helper.input_dtype()
                )
            )
            i += 1

        inputs = {'X': input}
        helper.append_op(
            type='broadcast_tensors',
            inputs=inputs,
            outputs={'Out': out},
            attrs={},
        )

        return out


def flip(
    x: Tensor, axis: Sequence[int] | int, name: str | None = None
) -> Tensor:
    """
    Reverse the order of a n-D tensor along given axis in axis.

    The image below illustrates how ``flip`` works.

    .. image:: https://githubraw.cdn.bcebos.com/PaddlePaddle/docs/develop/docs/images/api_legend/flip.png
        :width: 500
        :alt: legend of flip API
        :align: center

    Args:
        x (Tensor): A Tensor with shape :math:`[N_1, N_2,..., N_k]` . The data type of the input Tensor x
            should be float32, float64, int32, int64, bool.
        axis (list|tuple|int): The axis(axes) to flip on. Negative indices for indexing from the end are accepted.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, Tensor or DenseTensor calculated by flip layer. The data type is same with input x.

    Examples:
        .. code-block:: python

            >>> # doctest: +SKIP("This has diff in xdoctest env")
            >>> import paddle

            >>> image_shape=(3, 2, 2)
            >>> img = paddle.arange(image_shape[0] * image_shape[1] * image_shape[2]).reshape(image_shape)
            >>> tmp = paddle.flip(img, [0,1])
            >>> print(tmp)
            Tensor(shape=[3, 2, 2], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[[10, 11],
              [8 , 9 ]],
             [[6 , 7 ],
              [4 , 5 ]],
             [[2 , 3 ],
              [0 , 1 ]]])

            >>> out = paddle.flip(tmp,-1)
            >>> print(out)
            Tensor(shape=[3, 2, 2], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[[11, 10],
              [9 , 8 ]],
             [[7 , 6 ],
              [5 , 4 ]],
             [[3 , 2 ],
              [1 , 0 ]]])
    """
    if isinstance(axis, int):
        axis = [axis]

    if in_dynamic_or_pir_mode():
        return _C_ops.flip(x, axis)
    else:
        helper = LayerHelper("flip", **locals())
        check_type(x, 'X', (Variable), 'flip')
        dtype = helper.input_dtype('x')
        check_dtype(
            dtype,
            'X',
            ['float16', 'float32', 'float64', 'int32', 'int64', 'bool'],
            'flip',
        )
        check_type(axis, 'axis', (list, tuple), 'flip')
        if name is None:
            out = helper.create_variable_for_type_inference(dtype)
        else:
            out = helper.create_variable(
                name=name, dtype=dtype, persistable=False
            )

        helper.append_op(
            type="flip",
            inputs={"X": x},
            outputs={"Out": out},
            attrs={"axis": axis},
        )
        return out


def rot90(
    x: Tensor, k: int = 1, axes: Sequence[int] = [0, 1], name: str | None = None
) -> Tensor:
    """
    Rotate a n-D tensor by 90 degrees. The rotation direction and times are specified by axes and the absolute value of k. Rotation direction is from axes[0] towards axes[1] if k > 0, and from axes[1] towards axes[0] for k < 0.

    Args:
        x (Tensor): The input Tensor. The data type of the input Tensor x
            should be float16, float32, float64, int32, int64, bool. float16 is only supported on gpu.
        k (int, optional): Direction and number of times to rotate, default value: 1.
        axes (list|tuple, optional): Axes to rotate, dimension must be 2. default value: [0, 1].
        name (str|None, optional): The default value is None.  Normally there is no need for user to set this property.
            For more information, please refer to :ref:`api_guide_Name` .

    Returns:
        Tensor, Tensor or DenseTensor calculated by rot90 layer. The data type is same with input x.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> data = paddle.arange(4)
            >>> data = paddle.reshape(data, (2, 2))
            >>> print(data.numpy())
            [[0 1]
             [2 3]]

            >>> y = paddle.rot90(data, 1, [0, 1])
            >>> print(y.numpy())
            [[1 3]
             [0 2]]

            >>> y= paddle.rot90(data, -1, [0, 1])
            >>> print(y.numpy())
            [[2 0]
             [3 1]]

            >>> data2 = paddle.arange(8)
            >>> data2 = paddle.reshape(data2, (2,2,2))
            >>> print(data2.numpy())
            [[[0 1]
              [2 3]]
             [[4 5]
              [6 7]]]

            >>> y = paddle.rot90(data2, 1, [1, 2])
            >>> print(y.numpy())
            [[[1 3]
              [0 2]]
             [[5 7]
              [4 6]]]
    """

    helper = LayerHelper("rot90", **locals())
    check_type(x, 'X', (Variable, paddle.pir.Value), 'rot90')
    dtype = helper.input_dtype('x')
    check_dtype(
        dtype,
        'X',
        ['float16', 'float32', 'float64', 'int32', 'int64', 'bool'],
        'rot90',
    )
    check_type(axes, 'axes', (list, tuple), 'rot90')

    input_total_dims = len(x.shape)
    total_rot_dims = len(axes)
    if total_rot_dims != 2:
        raise ValueError(
            f"expected total rotation axes == 2, but got axes = {total_rot_dims}"
        )
    if input_total_dims < 2:
        raise ValueError(
            f"expected total dims >= 2, but got total dims = {input_total_dims}"
        )

    if not (axes[0] != axes[1] and abs(axes[0] - axes[1]) != input_total_dims):
        raise ValueError(
            f"expected rotation axes to be different, but got axis0 = {axes[0]}, and axis1 = {axes[1]}"
        )

    if not (axes[0] < input_total_dims and axes[0] >= -input_total_dims):
        raise ValueError(f"Rotation axis0 out of range, axis0 = {axes[0]}")
    if not (axes[1] < input_total_dims and axes[1] >= -input_total_dims):
        raise ValueError(f"Rotation axis1 out of range, axis1 = {axes[1]}")

    k %= 4
    if k == 0:
        return x
    if k == 2:
        return flip(flip(x, axes[0]), axes[1])

    axes_list = list(range(0, input_total_dims))
    (axes_list[axes[0]], axes_list[axes[1]]) = (
        axes_list[axes[1]],
        axes_list[axes[0]],
    )
    if k == 1:
        return transpose(flip(x, axes[1]), axes_list)
    else:
        # k == 3
        return flip(transpose(x, axes_list), axes[1])


def flatten(
    x: Tensor, start_axis: int = 0, stop_axis: int = -1, name: str | None = None
) -> Tensor:
    r"""
    Flattens a contiguous range of axes in a tensor according to start_axis and stop_axis.

    Note:
        The output Tensor will share data with origin Tensor and doesn't have a Tensor copy in ``dygraph`` mode.
        If you want to use the Tensor copy version, please use `Tensor.clone` like ``flatten_clone_x = x.flatten().clone()``.

    For Example:

    .. code-block:: text

        Case 1:

          Given
            X.shape = (3, 100, 100, 4)

          and
            start_axis = 1
            end_axis = 2

          We get:
            Out.shape = (3, 100 * 100, 4)

        Case 2:

          Given
            X.shape = (3, 100, 100, 4)

          and
            start_axis = 0
            stop_axis = -1

          We get:
            Out.shape = (3 * 100 * 100 * 4)

    Args:
        x (Tensor): A tensor of number of dimensions >= axis. A tensor with data type float16, float32,
                      float64, int8, int32, int64, uint8.
        start_axis (int): the start axis to flatten
        stop_axis (int): the stop axis to flatten
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, A tensor with the contents of the input tensor, whose input axes are flattened by indicated :attr:`start_axis` and :attr:`end_axis`, and data type is the same as input :attr:`x`.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> image_shape=(2, 3, 4, 4)

            >>> x = paddle.arange(end=image_shape[0] * image_shape[1] * image_shape[2] * image_shape[3])
            >>> img = paddle.reshape(x, image_shape)

            >>> out = paddle.flatten(img, start_axis=1, stop_axis=2)
            >>> print(out.shape)
            [2, 12, 4]

            >>> # out shares data with img in dygraph mode
            >>> img[0, 0, 0, 0] = -1
            >>> print(out[0, 0, 0])
            Tensor(shape=[], dtype=int64, place=Place(cpu), stop_gradient=True,
            -1)
    """
    if not (isinstance(x, (Variable, paddle.pir.Value))):
        raise ValueError("The input x should be a Tensor")

    x_dim = len(x.shape)
    if x_dim == 0:
        if not (isinstance(start_axis, int)) or start_axis not in [0, -1]:
            raise ValueError(
                "The start_axis should be int, and should be 0 or -1 when the input tensor is a 0-D-Tensor"
            )
        if not (isinstance(stop_axis, int)) or stop_axis not in [0, -1]:
            raise ValueError(
                "The stop_axis should be int, and should be 0 or -1 when the input tensor is a 0-D-Tensor"
            )
    else:
        if (
            not (isinstance(start_axis, int))
            or (start_axis > x_dim - 1)
            or start_axis < -x_dim
        ):
            raise ValueError(
                "The start_axis should be a int, and in range [-rank(x), rank(x))"
            )
        if (
            not (isinstance(stop_axis, int))
            or (stop_axis > x_dim - 1)
            or stop_axis < -x_dim
        ):
            raise ValueError(
                "The stop_axis should be a int, and in range [-rank(x), rank(x))"
            )
        if start_axis < 0:
            start_axis = start_axis + x_dim
        if stop_axis < 0:
            stop_axis = stop_axis + x_dim
        if start_axis > stop_axis:
            raise ValueError("The stop_axis should be larger than stat_axis")

    if in_dynamic_or_pir_mode():
        return _C_ops.flatten(x, start_axis, stop_axis)
    else:
        check_variable_and_dtype(
            x,
            'x',
            [
                'float16',
                'float32',
                'float64',
                'int8',
                'int16',
                'int32',
                'int64',
                'uint8',
                'uint16',
            ],
            'flatten',
        )
        helper = LayerHelper('flatten', **locals())
        out = helper.create_variable_for_type_inference(x.dtype)
        x_shape = helper.create_variable_for_type_inference(x.dtype)
        helper.append_op(
            type='flatten_contiguous_range',
            inputs={"X": x},
            outputs={'Out': out, 'XShape': x_shape},
            attrs={"start_axis": start_axis, "stop_axis": stop_axis},
        )
        return out


@inplace_apis_in_dygraph_only
def flatten_(
    x: Tensor, start_axis: int = 0, stop_axis: int = -1, name: str | None = None
) -> Tensor:
    """
    Inplace version of ``flatten`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_paddle_flatten`.
    """
    if not (isinstance(x, Variable)):
        raise ValueError("The input x should be a Tensor")

    x_dim = len(x.shape)
    if (
        not (isinstance(start_axis, int))
        or (start_axis > x_dim - 1)
        or start_axis < -x_dim
    ):
        raise ValueError(
            "The start_axis should be a int, and in range [-rank(x), rank(x))"
        )
    if (
        not (isinstance(stop_axis, int))
        or (stop_axis > x_dim - 1)
        or stop_axis < -x_dim
    ):
        raise ValueError(
            "The stop_axis should be a int, and in range [-rank(x), rank(x))"
        )
    if start_axis < 0:
        start_axis = start_axis + x_dim
    if stop_axis < 0:
        stop_axis = stop_axis + x_dim
    if start_axis > stop_axis:
        raise ValueError("The stop_axis should be larger than stat_axis")

    if in_dynamic_mode():
        return _C_ops.flatten_(x, start_axis, stop_axis)


def roll(
    x: Tensor,
    shifts: int | Sequence[int],
    axis: int | Sequence[int] | None = None,
    name: str | None = None,
) -> Tensor:
    """
    Roll the `x` tensor along the given axis(axes). With specific 'shifts', Elements that
    roll beyond the last position are re-introduced at the first according to 'shifts'.
    If a axis is not specified,
    the tensor will be flattened before rolling and then restored to the original shape.

    Args:
        x (Tensor): The x tensor as input.
        shifts (int|list|tuple): The number of places by which the elements
                           of the `x` tensor are shifted.
        axis (int|list|tuple, optional): axis(axes) along which to roll. Default: None
        name(str|None, optional): The default value is None.  Normally there is no need for user to set this property.
                For more information, please refer to :ref:`api_guide_Name` .

    The image below shows a 2D tensor `[[1,2,3],[4,5,6],[7,8,9]]` being transformed into tensors with
    different shapes through the roll operation.

    .. image:: https://githubraw.cdn.bcebos.com/PaddlePaddle/docs/develop/docs/images/api_legend/roll.png
        :width: 700
        :align: center
        :alt: legend of roll API

    Returns:
        Tensor, A Tensor with same data type as `x`.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([[1.0, 2.0, 3.0],
            ...                       [4.0, 5.0, 6.0],
            ...                       [7.0, 8.0, 9.0]])
            >>> out_z1 = paddle.roll(x, shifts=1)
            >>> print(out_z1.numpy())
            [[9. 1. 2.]
             [3. 4. 5.]
             [6. 7. 8.]]
            >>> out_z2 = paddle.roll(x, shifts=1, axis=0)
            >>> print(out_z2.numpy())
            [[7. 8. 9.]
             [1. 2. 3.]
             [4. 5. 6.]]
            >>> out_z3 = paddle.roll(x, shifts=1, axis=1)
            >>> print(out_z3.numpy())
            [[3. 1. 2.]
             [6. 4. 5.]
             [9. 7. 8.]]
    """
    origin_shape = x.shape
    if type(shifts) == int:
        shifts = [shifts]
    if type(axis) == int:
        axis = [axis]

    len_origin_shape = len(origin_shape)
    if axis is not None:
        for i in range(len(axis)):
            if axis[i] >= len_origin_shape or axis[i] < -len_origin_shape:
                raise ValueError(
                    f"axis is out of range, it should be in range [{-len_origin_shape}, {len_origin_shape}), but received {axis}"
                )
    else:
        axis = []

    if in_dynamic_or_pir_mode():
        return _C_ops.roll(x, shifts, axis)
    else:
        check_variable_and_dtype(
            x,
            'dtype',
            [
                'bool',
                'float16',
                'float32',
                'uint16',
                'float64',
                'int32',
                'int64',
                'complex64',
                'complex128',
            ],
            'roll',
        )
        helper = LayerHelper("roll", **locals())
        check_type(axis, 'axis', (list, tuple), 'roll')

        out = helper.create_variable_for_type_inference(x.dtype)

        if isinstance(shifts, Variable):
            helper.append_op(
                type='roll',
                inputs={'X': x, "ShiftsTensor": shifts},
                outputs={'Out': out},
                attrs={'axis': axis},
            )
        else:
            check_type(shifts, 'shifts', (list, tuple), 'roll')
            helper.append_op(
                type='roll',
                inputs={'X': x},
                outputs={'Out': out},
                attrs={'axis': axis, 'shifts': shifts},
            )
        return out


def stack(
    x: Sequence[Tensor], axis: int = 0, name: str | None = None
) -> Tensor:
    """
    Stacks all the input tensors ``x`` along ``axis`` dimension.
    All tensors must be of the same shape and same dtype.

    For example, given N tensors of shape [A, B], if ``axis == 0``, the shape of stacked
    tensor is [N, A, B]; if ``axis == 1``, the shape of stacked
    tensor is [A, N, B], etc.

    It also supports the operation with zero-size tensors which contain 0 in their shape.
    See the examples below.

    .. code-block:: text

        Case 1:

          Input:
            x[0].shape = [1, 2]
            x[0].data = [ [1.0 , 2.0 ] ]
            x[1].shape = [1, 2]
            x[1].data = [ [3.0 , 4.0 ] ]
            x[2].shape = [1, 2]
            x[2].data = [ [5.0 , 6.0 ] ]

          Attrs:
            axis = 0

          Output:
            Out.dims = [3, 1, 2]
            Out.data =[ [ [1.0, 2.0] ],
                        [ [3.0, 4.0] ],
                        [ [5.0, 6.0] ] ]


        Case 2:

          Input:
            x[0].shape = [1, 2]
            x[0].data = [ [1.0 , 2.0 ] ]
            x[1].shape = [1, 2]
            x[1].data = [ [3.0 , 4.0 ] ]
            x[2].shape = [1, 2]
            x[2].data = [ [5.0 , 6.0 ] ]


          Attrs:
            axis = 1 or axis = -2  # If axis = -2, axis = axis+ndim(x[0])+1 = -2+2+1 = 1.

          Output:
            Out.shape = [1, 3, 2]
            Out.data =[ [ [1.0, 2.0]
                          [3.0, 4.0]
                          [5.0, 6.0] ] ]


        Case 3:

            Input:
                x[0].shape = [0, 1, 2]
                x[0].data = []
                x[1].shape = [0, 1, 2]
                x[1].data = []

            Attrs:
                axis = 0

            Output:
                Out.shape = [2, 0, 1, 2]
                Out.data = []


        Case 4:

            Input:
                x[0].shape = [0, 1, 2]
                x[0].data = []
                x[1].shape = [0, 1, 2]
                x[1].data = []

            Attrs:
                axis = 1

            Output:
                Out.shape = [0, 2, 1, 2]
                Out.data = []

    The image below demonstrates the Case 1: three 2-dimensional tensors with shape [1, 2] are stacked in the dimension of axis=0 to form a 3-dimensional tensor with shape [3, 1, 2] .

    .. figure:: https://githubraw.cdn.bcebos.com/PaddlePaddle/docs/develop/docs/images/api_legend/stack/stack-0.png
       :width: 1000
       :alt: Legend 1
       :align: center

    Args:
        x (list[Tensor]|tuple[Tensor]): Input ``x`` can be a ``list`` or ``tuple`` of tensors, the Tensors in ``x``
                                     must be of the same shape and dtype. Supported data types: float32, float64, int32, int64.
        axis (int, optional): The axis along which all inputs are stacked. ``axis`` range is ``[-(R+1), R+1)``,
                              where ``R`` is the number of dimensions of the first input tensor ``x[0]``.
                              If ``axis < 0``, ``axis = axis+R+1``. The default value of axis is 0.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, The stacked tensor with same data type as input.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x1 = paddle.to_tensor([[1.0, 2.0]])
            >>> x2 = paddle.to_tensor([[3.0, 4.0]])
            >>> x3 = paddle.to_tensor([[5.0, 6.0]])

            >>> out = paddle.stack([x1, x2, x3], axis=0)
            >>> print(out.shape)
            [3, 1, 2]
            >>> print(out)
            Tensor(shape=[3, 1, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[[1., 2.]],
             [[3., 4.]],
             [[5., 6.]]])

            >>> out = paddle.stack([x1, x2, x3], axis=-2)
            >>> print(out.shape)
            [1, 3, 2]
            >>> print(out)
            Tensor(shape=[1, 3, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[[1., 2.],
              [3., 4.],
              [5., 6.]]])

            >>> # zero-size tensors
            >>> x1 = paddle.ones([0, 1, 2])
            >>> x2 = paddle.ones([0, 1, 2])

            >>> out = paddle.stack([x1, x2], axis=0)
            >>> print(out.shape)
            [2, 0, 1, 2]
            >>> print(out)
            Tensor(shape=[2, 0, 1, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[],
             []])

            >>> out = paddle.stack([x1, x2], axis=1)
            >>> print(out.shape)
            [0, 2, 1, 2]
            >>> print(out)
            Tensor(shape=[0, 2, 1, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [])
    """
    axis = 0 if axis is None else axis

    if in_dynamic_mode():
        return _C_ops.stack(x, axis)

    if not isinstance(x, list) and not isinstance(x, tuple):
        # NOTE:(zhiqiu) Only support Variable as input if the Variable is a DENSE_TENSOR_ARRAY create by create_array, array_write, array_read, etc.
        # In that case, Variable is array of tensors indeed.
        if (
            isinstance(x, Variable)
            and x.desc.type() == core.VarDesc.VarType.DENSE_TENSOR_ARRAY
        ) or (
            isinstance(x, paddle.pir.Value) and x.is_dense_tensor_array_type()
        ):
            x = [x]
        else:
            raise TypeError(
                "The type of '{}' in {} must be {}, but received {}".format(
                    'x',
                    'stack',
                    'list[Tensor], tuple[Tensor] or TensorArray',
                    type(x),
                )
            )

    if in_pir_mode():
        if x[0].is_dense_tensor_array_type():
            assert len(x) == 1, (
                "If the elements of 'x' in stack are Variable(DenseTensorArray), "
                f"number of the elements must be 1, but received {len(x)}."
            )
            out, _ = _C_ops.array_to_tensor(x, axis, True)
            return out
        else:
            return _C_ops.stack(x, axis)

    helper = LayerHelper('stack', **locals())

    out = helper.create_variable_for_type_inference(x[0].dtype)
    if x[0].desc.type() == core.VarDesc.VarType.DENSE_TENSOR_ARRAY:
        assert len(x) == 1, (
            "If the elements of 'x' in stack are Variable(DenseTensorArray), "
            f"number of the elements must be 1, but received {len(x)}."
        )
        out_index = helper.create_variable_for_type_inference(dtype="int32")

        for i in x:
            check_variable_and_dtype(
                i,
                'x',
                [
                    'float16',
                    'float32',
                    'float64',
                    'int32',
                    'int64',
                    'uint16',
                ],
                'stack',
            )

        helper.append_op(
            type='tensor_array_to_tensor',
            inputs={'X': x[0]},
            outputs={'Out': [out], 'OutIndex': [out_index]},
            attrs={'axis': axis, 'use_stack': True},
        )
    else:
        helper.append_op(
            type='stack',
            inputs={'X': x},
            outputs={'Y': out},
            attrs={'axis': axis},
        )

    return out


def hstack(x: Sequence[Tensor], name: str | None = None) -> Tensor:
    """
    Stacks all the input tensors ``x`` along horizontal axis.
    All tensors must be of the same dtype.

    The image below illustrates how ``hstack`` works.

    .. image:: https://githubraw.cdn.bcebos.com/PaddlePaddle/docs/develop/docs/images/api_legend/hstack.png
        :width: 500
        :alt: legend of hstack API
        :align: center

    Args:
        x (list[Tensor]|tuple[Tensor]): Input ``x`` can be a ``list`` or ``tuple`` of tensors, the Tensors in ``x`` must be of the same
            shape and dtype. Supported data types: ``float16``, ``float32``, ``float64``, ``int8``, ``int32``, ``int64`` or ``bfloat16``.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, The stacked tensor with same data type as input.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> # hstack with 0-D tensors
            >>> x1 = paddle.to_tensor(1.0)
            >>> x2 = paddle.to_tensor(2.0)
            >>> out = paddle.hstack((x1, x2))
            >>> print(out)
            Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [1., 2.])

            >>> # hstack with 1-D tensors
            >>> x1 = paddle.to_tensor([1.0, 2.0])
            >>> x2 = paddle.to_tensor([3.0, 4.0, 5.0])
            >>> out = paddle.hstack((x1, x2))
            >>> print(out)
            Tensor(shape=[5], dtype=float32, place=Place(cpu), stop_gradient=True,
            [1., 2., 3., 4., 5.])

            >>> # hstack mix with 0-D & 1-D tensors
            >>> x1 = paddle.to_tensor(1.0)
            >>> x2 = paddle.to_tensor([3.0, 4.0, 5.0])
            >>> out = paddle.hstack((x1, x2))
            >>> print(out)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [1., 3., 4., 5.])

            >>> # hstack with 2-D tensors
            >>> x1 = paddle.to_tensor([[1.0, 2.0]])
            >>> x2 = paddle.to_tensor([[3.0, 4.0, 5.0]])
            >>> out = paddle.hstack((x1, x2))
            >>> print(out)
            Tensor(shape=[1, 5], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[1., 2., 3., 4., 5.]])

    """
    arrays = paddle.atleast_1d(*x)
    if not isinstance(arrays, list):
        arrays = [arrays]

    if arrays and arrays[0].ndim == 1:
        return paddle.concat(arrays, axis=0, name=name)
    else:
        return paddle.concat(arrays, axis=1, name=name)


def vstack(x: Sequence[Tensor], name: str | None = None) -> Tensor:
    """
    Stacks all the input tensors ``x`` along vertical axis.
    All tensors must be of the same dtype.

    The image below illustrates how ``vstack`` works.

    .. image:: https://githubraw.cdn.bcebos.com/PaddlePaddle/docs/develop/docs/images/api_legend/vstack.png
        :width: 500
        :alt: legend of vstack API
        :align: center

    Args:
        x (list[Tensor]|tuple[Tensor]): Input ``x`` can be a ``list`` or ``tuple`` of tensors, the Tensors in ``x`` must be of the same
            shape and dtype. Supported data types: ``float16``, ``float32``, ``float64``, ``int8``, ``int32``, ``int64`` or ``bfloat16``.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, The stacked tensor with same data type as input.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> # vstack with 0-D tensors
            >>> x1 = paddle.to_tensor(1.0)
            >>> x2 = paddle.to_tensor(2.0)
            >>> out = paddle.vstack((x1, x2))
            >>> print(out)
            Tensor(shape=[2, 1], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[1.],
             [2.]])

            >>> # vstack with 1-D tensors
            >>> x1 = paddle.to_tensor([1.0, 2.0, 3.0])
            >>> x2 = paddle.to_tensor([3.0, 4.0, 5.0])
            >>> out = paddle.vstack((x1, x2))
            >>> print(out)
            Tensor(shape=[2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[1., 2., 3.],
             [3., 4., 5.]])

            >>> # vstack mix with 1-D & 2-D tensors
            >>> x1 = paddle.to_tensor([1.0, 2.0, 3.0])
            >>> x2 = paddle.to_tensor([[3.0, 4.0, 5.0]])
            >>> out = paddle.vstack((x1, x2))
            >>> print(out)
            Tensor(shape=[2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[1., 2., 3.],
             [3., 4., 5.]])

            >>> # vstack with 2-D tensors
            >>> x1 = paddle.to_tensor([[1.0, 2.0, 3.0]])
            >>> x2 = paddle.to_tensor([[3.0, 4.0, 5.0]])
            >>> out = paddle.vstack((x1, x2))
            >>> print(out)
            Tensor(shape=[2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[1., 2., 3.],
             [3., 4., 5.]])

    """
    arrays = paddle.atleast_2d(*x)
    if not isinstance(arrays, list):
        arrays = [arrays]

    return paddle.concat(arrays, axis=0, name=name)


def dstack(x: Sequence[Tensor], name: str | None = None) -> Tensor:
    """
    Stacks all the input tensors ``x`` along depth axis.
    All tensors must be of the same dtype.

    Args:
        x (list[Tensor]|tuple[Tensor]): Input ``x`` can be a ``list`` or ``tuple`` of tensors, the Tensors in ``x`` must be of the same
            shape and dtype. Supported data types: ``float16``, ``float32``, ``float64``, ``int8``, ``int32``, ``int64`` or ``bfloat16``.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, The stacked tensor with same data type as input.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> # dstack with 0-D tensors
            >>> x1 = paddle.to_tensor(1.0)
            >>> x2 = paddle.to_tensor(2.0)
            >>> out = paddle.dstack((x1, x2))
            >>> print(out)
            Tensor(shape=[1, 1, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[[1., 2.]]])

            >>> # dstack with 1-D tensors
            >>> x1 = paddle.to_tensor([1.0, 2.0, 3.0])
            >>> x2 = paddle.to_tensor([3.0, 4.0, 5.0])
            >>> out = paddle.dstack((x1, x2))
            >>> print(out)
            Tensor(shape=[1, 3, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[[1., 3.],
              [2., 4.],
              [3., 5.]]])

            >>> # dstack with 3-D tensors
            >>> x1 = paddle.to_tensor([[[1.0, 2.0], [3.0, 4.0]]])
            >>> x2 = paddle.to_tensor([[[3.0, 4.0], [5.0, 6.0]]])
            >>> out = paddle.dstack((x1, x2))
            >>> print(out)
            Tensor(shape=[1, 2, 4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[[1., 2., 3., 4.],
              [3., 4., 5., 6.]]])

    """
    arrays = paddle.atleast_3d(*x)
    if not isinstance(arrays, list):
        arrays = [arrays]

    return paddle.concat(arrays, axis=2, name=name)


def column_stack(x: Sequence[Tensor], name: str | None = None) -> Tensor:
    """
    Stacks all the input tensors ``x`` along horizontal axis. Each tensor in ``x`` will be first reshaped into ``(tensor.numel(), 1)``
    if ``tensor.ndim < 2`` before being stacked.
    All tensors must be of the same dtype.

    Args:
        x (list[Tensor]|tuple[Tensor]): Input ``x`` can be a ``list`` or ``tuple`` of tensors, the Tensors in ``x`` must be of the same
            shape and dtype. Supported data types: ``float16``, ``float32``, ``float64``, ``int32``, ``int64`` or ``bfloat16``.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, The stacked tensor with same data type as input.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> # column_stack with 0-D tensors
            >>> x1 = paddle.to_tensor(1.0)
            >>> x2 = paddle.to_tensor(2.0)
            >>> out = paddle.column_stack((x1, x2))
            >>> print(out)
            Tensor(shape=[1, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[1., 2.]])

            >>> # column_stack mix with 1-D & 2-D tensors
            >>> x1 = paddle.to_tensor([[1.0], [2.0], [3.0]])
            >>> x2 = paddle.to_tensor([3.0, 4.0, 5.0])
            >>> out = paddle.column_stack((x1, x2))
            >>> print(out)
            Tensor(shape=[3, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[1., 3.],
             [2., 4.],
             [3., 5.]])

            >>> # column_stack with 3-D tensors
            >>> x1 = paddle.to_tensor([[[1.0, 2.0], [3.0, 4.0]]])
            >>> x2 = paddle.to_tensor([[[3.0, 4.0], [5.0, 6.0]]])
            >>> out = paddle.column_stack((x1, x2))
            >>> print(out)
            Tensor(shape=[1, 4, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[[1., 2.],
              [3., 4.],
              [3., 4.],
              [5., 6.]]])

    """
    arrays = []

    for tensor in x:
        if tensor.ndim < 2:
            arrays.append(tensor.reshape((tensor.numel(), 1)))
        else:
            arrays.append(tensor)

    return paddle.concat(arrays, axis=1, name=name)


def row_stack(x: Sequence[Tensor], name: str | None = None) -> Tensor:
    """
    Alias of `paddle.vstack()`.
    Stacks all the input tensors ``x`` along vertical axis.
    All tensors must be of the same dtype.

    Args:
        x (list[Tensor]|tuple[Tensor]): Input ``x`` can be a ``list`` or ``tuple`` of tensors, the Tensors in ``x`` must be of the same
            shape and dtype. Supported data types: ``float16``, ``float32``, ``float64``, ``int8``, ``int32``, ``int64`` or ``bfloat16``.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, The stacked tensor with same data type as input.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> # row_stack with 0-D tensors
            >>> x1 = paddle.to_tensor(1.0)
            >>> x2 = paddle.to_tensor(2.0)
            >>> out = paddle.row_stack((x1, x2))
            >>> print(out)
            Tensor(shape=[2, 1], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[1.],
             [2.]])

            >>> # row_stack with 1-D tensors
            >>> x1 = paddle.to_tensor([1.0, 2.0, 3.0])
            >>> x2 = paddle.to_tensor([3.0, 4.0, 5.0])
            >>> out = paddle.row_stack((x1, x2))
            >>> print(out)
            Tensor(shape=[2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[1., 2., 3.],
             [3., 4., 5.]])

            >>> # row_stack mix with 1-D & 2-D tensors
            >>> x1 = paddle.to_tensor([1.0, 2.0, 3.0])
            >>> x2 = paddle.to_tensor([[3.0, 4.0, 5.0]])
            >>> out = paddle.row_stack((x1, x2))
            >>> print(out)
            Tensor(shape=[2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[1., 2., 3.],
             [3., 4., 5.]])

            >>> # row_stack with 2-D tensors
            >>> x1 = paddle.to_tensor([[1.0, 2.0, 3.0]])
            >>> x2 = paddle.to_tensor([[3.0, 4.0, 5.0]])
            >>> out = paddle.row_stack((x1, x2))
            >>> print(out)
            Tensor(shape=[2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[1., 2., 3.],
             [3., 4., 5.]])

    """
    return paddle.vstack(x, name=name)


def split(
    x: Tensor,
    num_or_sections: int | Sequence[int],
    axis: int | Tensor = 0,
    name: str | None = None,
) -> list[Tensor]:
    """
    Split the input tensor into multiple sub-Tensors.

    Args:
        x (Tensor): A N-D Tensor. The data type is bool, bfloat16, float16, float32, float64, uint8, int8, int32 or int64.
        num_or_sections (int|list|tuple): If ``num_or_sections`` is an int, then ``num_or_sections``
            indicates the number of equal sized sub-Tensors that the ``x`` will be divided into.
            If ``num_or_sections`` is a list or tuple, the length of it indicates the number of
            sub-Tensors and the elements in it indicate the sizes of sub-Tensors'  dimension orderly.
            The length of the list must not  be larger than the ``x`` 's size of specified ``axis``.
        axis (int|Tensor, optional): The axis along which to split, it can be a integer or a ``0-D Tensor``
            with shape [] and data type  ``int32`` or ``int64``.
            If :math::`axis < 0`, the axis to split along is :math:`rank(x) + axis`. Default is 0.
        name (str|None, optional): The default value is None.  Normally there is no need for user to set this property.
            For more information, please refer to :ref:`api_guide_Name` .
    Returns:
        list(Tensor), The list of segmented Tensors.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> # x is a Tensor of shape [3, 9, 5]
            >>> x = paddle.rand([3, 9, 5])

            >>> out0, out1, out2 = paddle.split(x, num_or_sections=3, axis=1)
            >>> print(out0.shape)
            [3, 3, 5]
            >>> print(out1.shape)
            [3, 3, 5]
            >>> print(out2.shape)
            [3, 3, 5]

            >>> out0, out1, out2 = paddle.split(x, num_or_sections=[2, 3, 4], axis=1)
            >>> print(out0.shape)
            [3, 2, 5]
            >>> print(out1.shape)
            [3, 3, 5]
            >>> print(out2.shape)
            [3, 4, 5]

            >>> out0, out1, out2 = paddle.split(x, num_or_sections=[2, 3, -1], axis=1)
            >>> print(out0.shape)
            [3, 2, 5]
            >>> print(out1.shape)
            [3, 3, 5]
            >>> print(out2.shape)
            [3, 4, 5]

            >>> # axis is negative, the real axis is (rank(x) + axis)=1
            >>> out0, out1, out2 = paddle.split(x, num_or_sections=3, axis=-2)
            >>> print(out0.shape)
            [3, 3, 5]
            >>> print(out1.shape)
            [3, 3, 5]
            >>> print(out2.shape)
            [3, 3, 5]
    """

    input = x
    dim = axis
    if in_dynamic_mode():
        if isinstance(dim, Variable):
            dim = dim.item(0)
        assert dim + len(input.shape) >= 0, "(rank(x) + axis) must >= 0"
        dim = (dim + len(input.shape)) if dim < 0 else dim

        if isinstance(num_or_sections, (list, tuple)):
            if paddle.utils._contain_var(num_or_sections):
                for index, item in enumerate(num_or_sections):
                    if isinstance(item, Variable):
                        num_or_sections[index] = num_or_sections[index].item()
        elif not isinstance(num_or_sections, int):
            raise TypeError(
                "The type of 'num_or_sections' in split must be int, list or tuple in imperative mode, but "
                f"received {type(num_or_sections)}."
            )

        if isinstance(num_or_sections, int):
            return _C_ops.split_with_num(input, num_or_sections, dim)
        else:
            return _C_ops.split(input, num_or_sections, dim)
    elif in_pir_mode():
        if isinstance(dim, paddle.pir.Value):
            dim.stop_gradient = True
        if isinstance(dim, int):
            assert len(input.shape) + dim >= 0, "(rank(x) + axis) must >= 0"
            dim = (len(input.shape) + dim) if dim < 0 else dim

        input_shape = input.shape

        if not isinstance(num_or_sections, (int, list, tuple)):
            raise TypeError(
                "The type of 'num_or_sections' in split must be int, list or tuple in imperative mode, but "
                f"received {type(num_or_sections)}."
            )
        if isinstance(num_or_sections, int):
            assert num_or_sections > 0, 'num_or_sections must be than 0.'
            if isinstance(dim, int) and input_shape[dim] > 0:
                assert input_shape[dim] % num_or_sections == 0, (
                    "The input's size along the split dimension "
                    "must be evenly divisible by Attr(num_or_sections). "
                    f"But {num_or_sections} is not evenly divisible by {input_shape[dim]}. "
                )
            return _C_ops.split_with_num(input, num_or_sections, dim)
        else:
            if isinstance(dim, int) and input_shape[dim] > 0:
                assert (
                    len(num_or_sections) <= input_shape[dim]
                ), 'len(num_or_sections) must not be more than input.shape[dim].'
            if paddle.utils._contain_var(num_or_sections):
                num_or_sections = paddle.utils.get_int_tensor_list(
                    num_or_sections
                )
            return _C_ops.split(input, num_or_sections, dim)

    else:
        check_variable_and_dtype(
            input,
            'input',
            [
                'bool',
                'bfloat16',
                'float16',
                'uint16',
                'float32',
                'float64',
                'int32',
                'int64',
                'uint8',
                'int8',
            ],
            'split',
        )
        check_type(
            num_or_sections, 'num_or_sections', (list, int, tuple), 'split'
        )
        check_type(dim, 'dim', (int, Variable), 'split')
        if isinstance(dim, Variable):
            check_dtype(dim.dtype, 'dim', ['int32', 'int64'], 'split')

        helper = LayerHelper('split', **locals())

        input_shape = input.shape
        inputs = {'X': input}
        attrs = {
            'num': num_or_sections if isinstance(num_or_sections, int) else 0
        }

        def _get_SectionsTensorList(one_list):
            tensor_list = []
            unk_dim_idx = -1
            for idx, dim_size in enumerate(one_list):
                if isinstance(dim_size, Variable):
                    dim_size.stop_gradient = True
                    tensor_list.append(dim_size)
                else:
                    assert isinstance(dim_size, int)
                    if dim_size == -1:
                        assert unk_dim_idx == -1, (
                            "Only one value of 'num_or_section' in split can "
                            f"be -1. But received num_or_section[{idx}] is also -1."
                        )
                        unk_dim_idx = idx
                    temp_out = helper.create_variable_for_type_inference(
                        'int32'
                    )
                    fill_constant(
                        [1], 'int32', dim_size, force_cpu=True, out=temp_out
                    )
                    tensor_list.append(temp_out)
            return tensor_list

        if isinstance(dim, Variable):
            dim.stop_gradient = True
            inputs['AxisTensor'] = dim
        else:
            assert len(input.shape) + dim >= 0, "(rank(x) + axis) must >= 0"
            dim = (len(input_shape) + dim) if dim < 0 else dim
            attrs['axis'] = dim

        if isinstance(num_or_sections, int):
            assert num_or_sections > 0, 'num_or_sections must be than 0.'
            if isinstance(dim, int) and input_shape[dim] > 0:
                assert input_shape[dim] % num_or_sections == 0, (
                    "The input's size along the split dimension "
                    "must be evenly divisible by Attr(num_or_sections). "
                    f"But {num_or_sections} is not evenly divisible by {input_shape[dim]}. "
                )
            num = num_or_sections
        else:
            if isinstance(dim, int) and input_shape[dim] > 0:
                assert (
                    len(num_or_sections) <= input_shape[dim]
                ), 'len(num_or_sections) must not be more than input.shape[dim].'
            num = len(num_or_sections)
            attrs['sections'] = [
                -1 if isinstance(ele, Variable) else ele
                for ele in num_or_sections
            ]
            if paddle.utils._contain_var(num_or_sections):
                inputs['SectionsTensorList'] = _get_SectionsTensorList(
                    num_or_sections
                )

        outs = [
            helper.create_variable_for_type_inference(
                dtype=helper.input_dtype()
            )
            for i in range(num)
        ]
        helper.append_op(
            type='split', inputs=inputs, outputs={'Out': outs}, attrs=attrs
        )
        return outs


def tensor_split(
    x: Tensor,
    num_or_indices: int | Sequence[int],
    axis: int | Tensor = 0,
    name: str | None = None,
) -> list[Tensor]:
    """
    Split the input tensor into multiple sub-Tensors along ``axis``, allowing not being of equal size.

    In the following figure, the shape of Tenser x is [6], and after paddle.tensor_split(x, num_or_indices=4) transformation, we get four sub-Tensors out0, out1, out2, and out3 :

    .. image:: https://githubraw.cdn.bcebos.com/PaddlePaddle/docs/develop/docs/images/api_legend/tensor_split/tensor_split-1_en.png

    since the length of x in axis = 0 direction 6 is not divisible by num_or_indices = 4,
    the size of the first int(6 % 4) part after splitting will be int(6 / 4) + 1
    and the size of the remaining parts will be int(6 / 4).

    Args:
        x (Tensor): A Tensor whose dimension must be greater than 0. The data type is bool, bfloat16, float16, float32, float64, uint8, int32 or int64.
        num_or_indices (int|list|tuple): If ``num_or_indices`` is an int ``n``, ``x`` is split into ``n`` sections along ``axis``.
            If ``x`` is divisible by ``n``, each section will be ``x.shape[axis] / n``. If ``x`` is not divisible by ``n``, the first
            ``int(x.shape[axis] % n)`` sections will have size ``int(x.shape[axis] / n) + 1``, and the rest will be ``int(x.shape[axis] / n).
            If ``num_or_indices`` is a list or tuple of integer indices, ``x`` is split along ``axis`` at each of the indices. For instance,
            ``num_or_indices=[2, 4]`` with ``axis=0`` would split ``x`` into ``x[:2]``, ``x[2:4]`` and ``x[4:]`` along axis 0.
        axis (int|Tensor, optional): The axis along which to split, it can be a integer or a ``0-D Tensor``
            with shape [] and data type  ``int32`` or ``int64``.
            If :math::`axis < 0`, the axis to split along is :math:`rank(x) + axis`. Default is 0.
        name (str|None, optional): The default value is None.  Normally there is no need for user to set this property.
            For more information, please refer to :ref:`api_guide_Name` .
    Returns:
        list[Tensor], The list of segmented Tensors.

    Examples:
        .. code-block:: python
            :name: tensor-split-example-1

            >>> import paddle

            >>> # evenly split
            >>> # x is a Tensor of shape [8]
            >>> x = paddle.rand([8])
            >>> out0, out1 = paddle.tensor_split(x, num_or_indices=2)
            >>> print(out0.shape)
            [4]
            >>> print(out1.shape)
            [4]


        .. image:: https://githubraw.cdn.bcebos.com/PaddlePaddle/docs/develop/docs/images/api_legend/tensor_split/tensor_split-2.png

        .. code-block:: python
            :name: tensor-split-example-2

            >>> import paddle

            >>> # not evenly split
            >>> # x is a Tensor of shape [8]
            >>> x = paddle.rand([8])
            >>> out0, out1, out2 = paddle.tensor_split(x, num_or_indices=3)
            >>> print(out0.shape)
            [3]
            >>> print(out1.shape)
            [3]
            >>> print(out2.shape)
            [2]

        .. image:: https://githubraw.cdn.bcebos.com/PaddlePaddle/docs/develop/docs/images/api_legend/tensor_split/tensor_split-3_en.png

        .. code-block:: python
            :name: tensor-split-example-3

            >>> import paddle

            >>> # split with indices
            >>> # x is a Tensor of shape [8]
            >>> x = paddle.rand([8])
            >>> out0, out1, out2 = paddle.tensor_split(x, num_or_indices=[2, 3])
            >>> print(out0.shape)
            [2]
            >>> print(out1.shape)
            [1]
            >>> print(out2.shape)
            [5]

        .. image:: https://githubraw.cdn.bcebos.com/PaddlePaddle/docs/develop/docs/images/api_legend/tensor_split/tensor_split-4.png

        .. code-block:: python
            :name: tensor-split-example-4

            >>> import paddle

            >>> # split along axis
            >>> # x is a Tensor of shape [7, 8]
            >>> x = paddle.rand([7, 8])
            >>> out0, out1 = paddle.tensor_split(x, num_or_indices=2, axis=1)
            >>> print(out0.shape)
            [7, 4]
            >>> print(out1.shape)
            [7, 4]

        .. image:: https://githubraw.cdn.bcebos.com/PaddlePaddle/docs/develop/docs/images/api_legend/tensor_split/tensor_split-5.png

        .. code-block:: python
            :name: tensor-spilt-example-5

            >>> import paddle

            >>> # split along axis with indices
            >>> # x is a Tensor of shape [7, 8]
            >>> x = paddle.rand([7, 8])
            >>> out0, out1, out2 = paddle.tensor_split(x, num_or_indices=[2, 3], axis=1)
            >>> print(out0.shape)
            [7, 2]
            >>> print(out1.shape)
            [7, 1]
            >>> print(out2.shape)
            [7, 5]

        .. image:: https://githubraw.cdn.bcebos.com/PaddlePaddle/docs/develop/docs/images/api_legend/tensor_split/tensor_split-6.png

    """
    if x.ndim <= 0 or x.ndim <= axis:
        raise ValueError(
            f"The input tensor's dimension must be greater than 0 or axis which is {axis}, but got {x.ndim}"
        )

    total_n = x.shape[axis]

    def _tensor_split_indices(x, total_n, indices, axis):
        splits = []

        starts = 0
        ends = 0
        for idx in [*list(indices), total_n]:
            ends = idx
            # convert index < 0 to positive
            starts_index = starts if starts >= 0 else total_n + starts
            ends_index = ends if ends >= 0 else total_n + ends
            # ends index should equal or larger than starts
            ends_index = max(starts_index, ends_index)

            sub_array = paddle.slice(
                x, axes=[axis], starts=[starts_index], ends=[ends_index]
            )
            splits.append(sub_array)
            starts = ends

        return splits

    def _tensor_split_sections(x, total_n, sections, axis):
        if sections <= 0:
            raise ValueError('num_or_indices must be larger than 0.')

        base, mod = divmod(total_n, sections)
        num_or_sections = [base + 1] * mod + [base] * (sections - mod)
        return split(x, num_or_sections, axis)

    if isinstance(num_or_indices, int):
        return _tensor_split_sections(x, total_n, num_or_indices, axis)

    elif isinstance(num_or_indices, (list, tuple)):
        return _tensor_split_indices(x, total_n, num_or_indices, axis)

    else:
        raise ValueError(
            f"The num_or_indices should be int, list or tuple of ints, but got {type(num_or_indices)}"
        )


def hsplit(
    x: Tensor, num_or_indices: int | Sequence[int], name: str | None = None
) -> list[Tensor]:
    """

    ``hsplit`` Full name Horizontal Split, splits the input Tensor into multiple sub-Tensors along the horizontal axis, in the following two cases:

    1. When the dimension of x is equal to 1, it is equivalent to ``paddle.tensor_split`` with ``axis=0``;

        .. image:: https://githubraw.cdn.bcebos.com/PaddlePaddle/docs/develop/docs/images/api_legend/hsplit/hsplit-1.png

    2. when the dimension of x is greater than 1, it is equivalent to ``paddle.tensor_split`` with ``axis=1``.

        .. image:: https://githubraw.cdn.bcebos.com/PaddlePaddle/docs/develop/docs/images/api_legend/hsplit/hsplit-2.png


    Args:
        x (Tensor): A Tensor whose dimension must be greater than 0. The data type is bool, bfloat16, float16, float32, float64, uint8, int32 or int64.
        num_or_indices (int|list|tuple): If ``num_or_indices`` is an int ``n``, ``x`` is split into ``n`` sections.
            If ``num_or_indices`` is a list or tuple of integer indices, ``x`` is split at each of the indices.
        name (str|None, optional): The default value is None.  Normally there is no need for user to set this property.
            For more information, please refer to :ref:`api_guide_Name` .
    Returns:
        list[Tensor], The list of segmented Tensors.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> # x is a Tensor of shape [8]
            >>> x = paddle.rand([8])
            >>> out0, out1 = paddle.hsplit(x, num_or_indices=2)
            >>> print(out0.shape)
            [4]
            >>> print(out1.shape)
            [4]

            >>> # x is a Tensor of shape [7, 8]
            >>> x = paddle.rand([7, 8])
            >>> out0, out1 = paddle.hsplit(x, num_or_indices=2)
            >>> print(out0.shape)
            [7, 4]
            >>> print(out1.shape)
            [7, 4]

            >>> out0, out1, out2 = paddle.hsplit(x, num_or_indices=[1, 4])
            >>> print(out0.shape)
            [7, 1]
            >>> print(out1.shape)
            [7, 3]
            >>> print(out2.shape)
            [7, 4]

    """
    if x.ndim < 1:
        raise ValueError(
            f"The input tensor's dimension must be greater than 0, but got {x.ndim}"
        )
    if x.ndim > 1:
        return tensor_split(x, num_or_indices, axis=1, name=name)
    else:
        return tensor_split(x, num_or_indices, axis=0, name=name)


def dsplit(
    x: Tensor, num_or_indices: int | Sequence[int], name: str | None = None
) -> list[Tensor]:
    """

    ``dsplit`` Full name Depth Split, splits the input Tensor into multiple sub-Tensors along the depth axis, which is equivalent to ``paddle.tensor_split`` with ``axis=2``.

    Note:
        Make sure that the number of Tensor dimensions transformed using ``paddle.dsplit`` must be no less than 3.

    In the following figure, Tenser ``x`` has shape [4, 4, 4], and after ``paddle.dsplit(x, num_or_indices=2)`` transformation, we get ``out0`` and ``out1`` sub-Tensors whose shapes are both [4, 4, 2] :

    .. image:: https://githubraw.cdn.bcebos.com/PaddlePaddle/docs/develop/docs/images/api_legend/dsplit/dsplit.png


    Args:
        x (Tensor): A Tensor whose dimension must be greater than 2. The data type is bool, bfloat16, float16, float32, float64, uint8, int32 or int64.
        num_or_indices (int|list|tuple): If ``num_or_indices`` is an int ``n``, ``x`` is split into ``n`` sections.
            If ``num_or_indices`` is a list or tuple of integer indices, ``x`` is split at each of the indices.
        name (str|None, optional): The default value is None.  Normally there is no need for user to set this property.
            For more information, please refer to :ref:`api_guide_Name` .
    Returns:
        list[Tensor], The list of segmented Tensors.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> # x is a Tensor of shape [7, 6, 8]
            >>> x = paddle.rand([7, 6, 8])
            >>> out0, out1 = paddle.dsplit(x, num_or_indices=2)
            >>> print(out0.shape)
            [7, 6, 4]
            >>> print(out1.shape)
            [7, 6, 4]

            >>> out0, out1, out2 = paddle.dsplit(x, num_or_indices=[1, 4])
            >>> print(out0.shape)
            [7, 6, 1]
            >>> print(out1.shape)
            [7, 6, 3]
            >>> print(out2.shape)
            [7, 6, 4]

    """
    if x.ndim < 3:
        raise ValueError(
            f"The input tensor's dimension must be greater than 2, but got {x.ndim}"
        )
    return tensor_split(x, num_or_indices, axis=2, name=name)


def vsplit(
    x: Tensor, num_or_indices: int | Sequence[int], name: str | None = None
) -> list[Tensor]:
    """

    ``vsplit`` Full name Vertical Split, splits the input Tensor into multiple sub-Tensors along the vertical axis, which is equivalent to ``paddle.tensor_split`` with ``axis=0``.

    1. When the number of Tensor dimensions is equal to 2:

    .. image:: https://githubraw.cdn.bcebos.com/PaddlePaddle/docs/develop/docs/images/api_legend/vsplit/vsplit-1.png

    2. When the number of Tensor dimensions is greater than 2:

    .. image:: https://githubraw.cdn.bcebos.com/PaddlePaddle/docs/develop/docs/images/api_legend/vsplit/vsplit-2.png


    Note:
        Make sure that the number of Tensor dimensions transformed using ``paddle.vsplit`` must be not less than 2.

    Args:
        x (Tensor): A Tensor whose dimension must be greater than 1. The data type is bool, bfloat16, float16, float32, float64, uint8, int32 or int64.
        num_or_indices (int|list|tuple): If ``num_or_indices`` is an int ``n``, ``x`` is split into ``n`` sections.
            If ``num_or_indices`` is a list or tuple of integer indices, ``x`` is split at each of the indices.
        name (str, optional): The default value is None.  Normally there is no need for user to set this property.
            For more information, please refer to :ref:`api_guide_Name` .

    Returns:
        list[Tensor], The list of segmented Tensors.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> # x is a Tensor of shape [8, 6, 7]
            >>> x = paddle.rand([8, 6, 7])
            >>> out0, out1 = paddle.vsplit(x, num_or_indices=2)
            >>> print(out0.shape)
            [4, 6, 7]
            >>> print(out1.shape)
            [4, 6, 7]

            >>> out0, out1, out2 = paddle.vsplit(x, num_or_indices=[1, 4])
            >>> print(out0.shape)
            [1, 6, 7]
            >>> print(out1.shape)
            [3, 6, 7]
            >>> print(out2.shape)
            [4, 6, 7]

    """
    if x.ndim < 2:
        raise ValueError(
            f"The input tensor's dimension must be greater than 1, but got {x.ndim}"
        )
    return tensor_split(x, num_or_indices, axis=0, name=name)


def squeeze(
    x: Tensor, axis: int | Sequence[int] | None = None, name: str | None = None
) -> Tensor:
    """
    Squeeze the dimension(s) of size 1 of input tensor x's shape.

    Note that the output Tensor will share data with origin Tensor and doesn't have a
    Tensor copy in ``dygraph`` mode. If you want to use the Tensor copy version,
    please use `Tensor.clone` like ``squeeze_clone_x = x.squeeze().clone()``.

    If axis is provided, it will remove the dimension(s) by given axis that of size 1.
    If the dimension of given axis is not of size 1, the dimension remain unchanged.
    If axis is not provided, all dims equal of size 1 will be removed.

    .. code-block:: text

        Case1:

          Input:
            x.shape = [1, 3, 1, 5]  # If axis is not provided, all dims equal of size 1 will be removed.
            axis = None
          Output:
            out.shape = [3, 5]

        Case2:

          Input:
            x.shape = [1, 3, 1, 5]  # If axis is provided, it will remove the dimension(s) by given axis that of size 1.
            axis = 0
          Output:
            out.shape = [3, 1, 5]

        Case4:

          Input:
            x.shape = [1, 3, 1, 5]  # If the dimension of one given axis (3) is not of size 1, the dimension remain unchanged.
            axis = [0, 2, 3]
          Output:
            out.shape = [3, 5]

        Case4:

          Input:
            x.shape = [1, 3, 1, 5]  # If axis is negative, axis = axis + ndim (number of dimensions in x).
            axis = [-2]
          Output:
            out.shape = [1, 3, 5]

    Args:
        x (Tensor): The input Tensor. Supported data type: float32, float64, bool, int8, int32, int64.
        axis (int|list|tuple, optional): An integer or list/tuple of integers, indicating the dimensions to be squeezed. Default is None.
                          The range of axis is :math:`[-ndim(x), ndim(x))`.
                          If axis is negative, :math:`axis = axis + ndim(x)`.
                          If axis is None, all the dimensions of x of size 1 will be removed.
        name (str|None, optional): Please refer to :ref:`api_guide_Name`, Default None.

    Returns:
        Tensor, Squeezed Tensor with the same data type as input Tensor.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.rand([5, 1, 10])
            >>> output = paddle.squeeze(x, axis=1)

            >>> print(x.shape)
            [5, 1, 10]
            >>> print(output.shape)
            [5, 10]

            >>> # output shares data with x in dygraph mode
            >>> x[0, 0, 0] = 10.
            >>> print(output[0, 0])
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
            10.)

    """
    if axis is None:
        axis = []
    elif isinstance(axis, int):
        axis = [axis]
    elif isinstance(axis, tuple):
        axis = list(axis)

    input = x
    axes = axis
    if in_dynamic_mode():
        return _C_ops.squeeze(input, axes)
    elif in_pir_mode():
        if isinstance(axes, int):
            axes = [axes]
        if isinstance(axes, paddle.pir.Value):
            axes.stop_gradient = True
        elif isinstance(axes, (list, tuple)):
            if paddle.utils._contain_var(axes):
                axes = paddle.utils.get_int_tensor_list(
                    axes, default_dtype='int64'
                )
        return _C_ops.squeeze(input, axes)
    else:
        helper = LayerHelper("squeeze", **locals())
        check_variable_and_dtype(
            input,
            'input',
            [
                'float16',
                'uint16',
                'float32',
                'float64',
                'bool',
                'int8',
                'int32',
                'int64',
                'complex64',
                'complex128',
            ],
            'squeeze',
        )

        check_type(axes, 'axis/axes', (int, list, tuple, Variable), 'squeeze')
        attrs = {}
        if isinstance(axes, Variable):
            axes.stop_gradient = True
            attrs["axes"] = axes
        elif isinstance(axes, (list, tuple)):
            if paddle.utils._contain_var(axes):
                attrs["axes"] = paddle.utils._convert_to_tensor_list(axes)
            else:
                attrs["axes"] = axes

        out = helper.create_variable_for_type_inference(dtype=input.dtype)
        x_shape = helper.create_variable_for_type_inference(dtype=input.dtype)
        helper.append_op(
            type="squeeze2",
            inputs={"X": input},
            attrs=attrs,
            outputs={"Out": out, "XShape": x_shape},
        )

        return out


@inplace_apis_in_dygraph_only
def squeeze_(
    x: Tensor, axis: int | Sequence[int] | None = None, name: str | None = None
) -> Tensor:
    """
    Inplace version of ``squeeze`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_paddle_tensor_squeeze`.
    """
    if axis is None:
        axis = []
    elif isinstance(axis, int):
        axis = [axis]
    elif isinstance(axis, tuple):
        axis = list(axis)

    input = x
    axes = axis
    if in_dynamic_mode():
        return _C_ops.squeeze_(input, axes)


def unique_consecutive(
    x: Tensor,
    return_inverse: bool = False,
    return_counts: bool = False,
    axis: int | None = None,
    dtype: DTypeLike = 'int64',
    name: str | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Eliminates all but the first element from every consecutive group of equivalent elements.

    Note:
        This function is different from :ref:`api_paddle_unique` in the sense that this function
        only eliminates consecutive duplicate values. This semantics is similar to :ref:`api_paddle_unique` in C++.

    Args:
        x(Tensor): the input tensor, it's data type should be float32, float64, int32, int64.
        return_inverse(bool, optional): If True, also return the indices for where elements in
            the original input ended up in the returned unique consecutive tensor. Default is False.
        return_counts(bool, optional): If True, also return the counts for each unique consecutive element.
            Default is False.
        axis(int, optional): The axis to apply unique consecutive. If None, the input will be flattened.
            Default is None.
        dtype(np.dtype|str, optional): The data type `inverse` tensor: int32 or int64.
            Default: int64.
        name(str|None, optional): Name for the operation. For more information, please refer to
            :ref:`api_guide_Name`. Default is None.

    Returns:
        - out (Tensor), the unique consecutive tensor for x.
        - inverse (Tensor), the element of the input tensor corresponds to
            the index of the elements in the unique consecutive tensor for x.
            inverse is provided only if return_inverse is True.
        - counts (Tensor), the counts of the every unique consecutive element in the input tensor.
            counts is provided only if return_counts is True.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([1, 1, 2, 2, 3, 1, 1, 2])
            >>> output = paddle.unique_consecutive(x) #
            >>> print(output)
            Tensor(shape=[5], dtype=int64, place=Place(cpu), stop_gradient=True,
            [1, 2, 3, 1, 2])

            >>> _, inverse, counts = paddle.unique_consecutive(x, return_inverse=True, return_counts=True)
            >>> print(inverse)
            Tensor(shape=[8], dtype=int64, place=Place(cpu), stop_gradient=True,
            [0, 0, 1, 1, 2, 3, 3, 4])
            >>> print(counts)
            Tensor(shape=[5], dtype=int64, place=Place(cpu), stop_gradient=True,
             [2, 2, 1, 2, 1])

            >>> x = paddle.to_tensor([[2, 1, 3], [3, 0, 1], [2, 1, 3], [2, 1, 3]])
            >>> output = paddle.unique_consecutive(x, axis=0) #
            >>> print(output)
            Tensor(shape=[3, 3], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[2, 1, 3],
             [3, 0, 1],
             [2, 1, 3]])

            >>> x = paddle.to_tensor([[2, 1, 3], [3, 0, 1], [2, 1, 3], [2, 1, 3]])
            >>> output = paddle.unique_consecutive(x, axis=0) #
            >>> print(output)
            Tensor(shape=[3, 3], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[2, 1, 3],
             [3, 0, 1],
             [2, 1, 3]])
    """

    if axis is None:
        axis = []
    else:
        axis = [axis]

    attr_dtype = dtype
    if not isinstance(
        attr_dtype, (core.VarDesc.VarType, paddle.pir.core.DataType)
    ):
        attr_dtype = convert_np_dtype_to_dtype_(dtype)

    if in_dynamic_or_pir_mode():
        out, inverse, counts = _C_ops.unique_consecutive(
            x, return_inverse, return_counts, axis, attr_dtype
        )
        outs = [out]
        if return_inverse:
            outs.append(inverse)
        if return_counts:
            outs.append(counts)
        if len(outs) == 1:
            return outs[0]
        return tuple(outs)
    else:
        check_variable_and_dtype(
            x,
            "input",
            ['float32', 'float64', 'int32', 'int64'],
            'unique_consecutive',
        )
        check_type(return_inverse, 'return_inverse', bool, 'unique_consecutive')
        check_type(return_counts, 'return_counts', bool, 'unique_consecutive')
        check_dtype(dtype, 'dtype', ['int32', 'int64'], 'unique_consecutive')
        if len(axis) != 0:
            check_type(axis[0], 'axis', int, 'unique_consecutive')
        helper = LayerHelper('unique_consecutive', **locals())
        attrs = {
            'dtype': attr_dtype,
            "return_inverse": return_inverse,
            "return_counts": return_counts,
            "axis": axis,
        }
        out = helper.create_variable_for_type_inference(
            dtype=x.dtype, stop_gradient=True
        )
        inverse = helper.create_variable_for_type_inference(
            dtype=attr_dtype, stop_gradient=True
        )
        counts = helper.create_variable_for_type_inference(
            dtype=attr_dtype, stop_gradient=True
        )
        outputs = {"Out": out, "Index": inverse, "Counts": counts}
        outs = [out]
        if return_inverse:
            outs.append(inverse)
        if return_counts:
            outs.append(counts)
        helper.append_op(
            type="unique_consecutive",
            inputs={"X": x},
            attrs=attrs,
            outputs=outputs,
        )
        if len(outs) == 1:
            return outs[0]
        return tuple(outs)


@overload
def unique(
    x: Tensor,
    return_index: Literal[True] = ...,
    return_inverse: Literal[True] = ...,
    return_counts: Literal[True] = ...,
    axis: int | None = ...,
    dtype: DTypeLike = ...,
    name: str | None = ...,
) -> tuple[Tensor, Tensor, Tensor, Tensor]: ...


@overload
def unique(
    x: Tensor,
    return_index: Literal[False] = ...,
    return_inverse: Literal[True] = ...,
    return_counts: Literal[True] = ...,
    axis: int | None = ...,
    dtype: DTypeLike = ...,
    name: str | None = ...,
) -> tuple[Tensor, Tensor, Tensor]: ...


@overload
def unique(
    x: Tensor,
    return_index: Literal[True] = ...,
    return_inverse: Literal[False] = ...,
    return_counts: Literal[True] = ...,
    axis: int | None = ...,
    dtype: DTypeLike = ...,
    name: str | None = ...,
) -> tuple[Tensor, Tensor, Tensor]: ...


@overload
def unique(
    x: Tensor,
    return_index: Literal[True] = ...,
    return_inverse: Literal[True] = ...,
    return_counts: Literal[False] = ...,
    axis: int | None = ...,
    dtype: DTypeLike = ...,
    name: str | None = ...,
) -> tuple[Tensor, Tensor, Tensor]: ...


@overload
def unique(
    x: Tensor,
    return_index: Literal[False] = ...,
    return_inverse: Literal[False] = ...,
    return_counts: Literal[True] = ...,
    axis: int | None = ...,
    dtype: DTypeLike = ...,
    name: str | None = ...,
) -> tuple[Tensor, Tensor]: ...


@overload
def unique(
    x: Tensor,
    return_index: Literal[False] = ...,
    return_inverse: Literal[True] = ...,
    return_counts: Literal[False] = ...,
    axis: int | None = ...,
    dtype: DTypeLike = ...,
    name: str | None = ...,
) -> tuple[Tensor, Tensor]: ...


@overload
def unique(
    x: Tensor,
    return_index: Literal[True] = ...,
    return_inverse: Literal[False] = ...,
    return_counts: Literal[False] = ...,
    axis: int | None = ...,
    dtype: DTypeLike = ...,
    name: str | None = ...,
) -> tuple[Tensor, Tensor]: ...


@overload
def unique(
    x: Tensor,
    return_index: Literal[False] = ...,
    return_inverse: Literal[False] = ...,
    return_counts: Literal[False] = ...,
    axis: int | None = ...,
    dtype: DTypeLike = ...,
    name: str | None = ...,
) -> Tensor: ...


@overload
def unique(
    x: Tensor,
    return_index: bool = False,
    return_inverse: bool = False,
    return_counts: bool = False,
    axis: int | None = ...,
    dtype: DTypeLike = ...,
    name: str | None = ...,
) -> Tensor | tuple[Tensor, ...]: ...


def unique(
    x,
    return_index=False,
    return_inverse=False,
    return_counts=False,
    axis=None,
    dtype="int64",
    name=None,
):
    r"""
    Returns the unique elements of `x` in ascending order.

    Args:
        x(Tensor): The input tensor, it's data type should be float32, float64, int32, int64.
        return_index(bool, optional): If True, also return the indices of the input tensor that
            result in the unique Tensor.
        return_inverse(bool, optional): If True, also return the indices for where elements in
            the original input ended up in the returned unique tensor.
        return_counts(bool, optional): If True, also return the counts for each unique element.
        axis(int, optional): The axis to apply unique. If None, the input will be flattened.
            Default: None.
        dtype(np.dtype|str, optional): The date type of `indices` or `inverse` tensor: int32 or int64.
            Default: int64.
        name(str|None, optional): Name for the operation. For more information, please refer to
            :ref:`api_guide_Name`. Default: None.

    Returns:
        tuple (out, indices, inverse, counts). `out` is the unique tensor for `x`. `indices` is \
            provided only if `return_index` is True. `inverse` is provided only if `return_inverse` \
            is True. `counts` is provided only if `return_counts` is True.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([2, 3, 3, 1, 5, 3])
            >>> unique = paddle.unique(x)
            >>> print(unique)
            Tensor(shape=[4], dtype=int64, place=Place(cpu), stop_gradient=True,
            [1, 2, 3, 5])

            >>> _, indices, inverse, counts = paddle.unique(x, return_index=True, return_inverse=True, return_counts=True)
            >>> print(indices)
            Tensor(shape=[4], dtype=int64, place=Place(cpu), stop_gradient=True,
            [3, 0, 1, 4])
            >>> print(inverse)
            Tensor(shape=[6], dtype=int64, place=Place(cpu), stop_gradient=True,
            [1, 2, 2, 0, 3, 2])
            >>> print(counts)
            Tensor(shape=[4], dtype=int64, place=Place(cpu), stop_gradient=True,
            [1, 1, 3, 1])

            >>> x = paddle.to_tensor([[2, 1, 3], [3, 0, 1], [2, 1, 3]])
            >>> unique = paddle.unique(x)
            >>> print(unique)
            Tensor(shape=[4], dtype=int64, place=Place(cpu), stop_gradient=True,
            [0, 1, 2, 3])

            >>> unique = paddle.unique(x, axis=0)
            >>> print(unique)
            Tensor(shape=[2, 3], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[2, 1, 3],
             [3, 0, 1]])
    """
    if axis is None:
        axis = []
    else:
        axis = [axis]
    attr_dtype = convert_np_dtype_to_dtype_(dtype)
    if in_dynamic_mode():
        out, indices, inverse, counts = _C_ops.unique(
            x, return_index, return_inverse, return_counts, axis, attr_dtype
        )
        outs = [out]
        if return_index:
            outs.append(indices)
        if return_inverse:
            outs.append(inverse)
        if return_counts:
            outs.append(counts)

        if len(outs) == 1:
            return outs[0]

        return tuple(outs)
    elif in_pir_mode():
        out, indices, inverse, counts = _C_ops.unique(
            x,
            return_index,
            return_inverse,
            return_counts,
            axis,
            attr_dtype,
            True,
        )
        outs = [out]
        if return_index:
            outs.append(indices)
        if return_inverse:
            outs.append(inverse)
        if return_counts:
            outs.append(counts)

        if len(outs) == 1:
            return outs[0]

        return tuple(outs)
    else:
        check_variable_and_dtype(
            x,
            "input",
            ['float16', 'uint16', 'float32', 'float64', 'int32', 'int64'],
            'unique',
        )
        check_type(return_index, 'return_index', bool, 'unique')
        check_type(return_inverse, 'return_inverse', bool, 'unique')
        check_type(return_counts, 'return_counts', bool, 'unique')
        check_dtype(dtype, 'dtype', ['int32', 'int64'], 'unique')
        if len(axis) != 0:
            check_type(axis[0], 'axis', int, 'unique')

        helper = LayerHelper('unique', **locals())
        attrs = {
            'dtype': attr_dtype,
            "return_index": return_index,
            "return_inverse": return_inverse,
            "return_counts": return_counts,
            "axis": axis,
            "is_sorted": True,
        }
        out = helper.create_variable_for_type_inference(
            dtype=x.dtype, stop_gradient=True
        )
        indices = helper.create_variable_for_type_inference(
            dtype=attr_dtype, stop_gradient=True
        )
        inverse = helper.create_variable_for_type_inference(
            dtype=attr_dtype, stop_gradient=True
        )
        counts = helper.create_variable_for_type_inference(
            dtype=attr_dtype, stop_gradient=True
        )
        outputs = {
            "Out": out,
            "Indices": indices,
            "Index": inverse,
            "Counts": counts,
        }
        outs = [out]
        if return_index:
            outs.append(indices)
        if return_inverse:
            outs.append(inverse)
        if return_counts:
            outs.append(counts)

        helper.append_op(
            type="unique", inputs={"X": x}, attrs=attrs, outputs=outputs
        )

        if len(outs) == 1:
            return outs[0]

        return tuple(outs)


def unsqueeze(
    x: Tensor,
    axis: int | Sequence[Tensor | int] | Tensor,
    name: str | None = None,
) -> Tensor:
    """
    Insert single-dimensional entries to the shape of input Tensor ``x``. Takes one
    required argument axis, a dimension or list of dimensions that will be inserted.
    Dimension indices in axis are as seen in the output tensor.

    Note that the output Tensor will share data with origin Tensor and doesn't have a
    Tensor copy in ``dygraph`` mode. If you want to use the Tensor copy version,
    please use `Tensor.clone` like ``unsqueeze_clone_x = x.unsqueeze(-1).clone()``.

    Args:
        x (Tensor): The input Tensor to be unsqueezed. Supported data type: bfloat16, float16, float32, float64, bool, int8, int32, int64.
        axis (int|list|tuple|Tensor): Indicates the dimensions to be inserted. The data type is ``int32`` .
                                    If ``axis`` is a list or tuple, each element of it should be integer or 0-D Tensor with shape [].
                                    If ``axis`` is a Tensor, it should be an 1-D Tensor .
                                    If ``axis`` is negative, ``axis = axis + ndim(x) + 1``.
        name (str|None, optional): Name for this layer. Please refer to :ref:`api_guide_Name`, Default None.

    Returns:
        Tensor, Unsqueezed Tensor with the same data type as input Tensor.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.rand([5, 10])
            >>> print(x.shape)
            [5, 10]

            >>> out1 = paddle.unsqueeze(x, axis=0)
            >>> print(out1.shape)
            [1, 5, 10]

            >>> out2 = paddle.unsqueeze(x, axis=[0, 2])
            >>> print(out2.shape)
            [1, 5, 1, 10]

            >>> axis = paddle.to_tensor([0, 1, 2])
            >>> out3 = paddle.unsqueeze(x, axis=axis)
            >>> print(out3.shape)
            [1, 1, 1, 5, 10]

            >>> # out1, out2, out3 share data with x in dygraph mode
            >>> x[0, 0] = 10.
            >>> print(out1[0, 0, 0])
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
            10.)
            >>> print(out2[0, 0, 0, 0])
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
            10.)
            >>> print(out3[0, 0, 0, 0, 0])
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
            10.)

    """
    input = x
    axes = axis
    if in_dynamic_mode():
        if isinstance(axes, int):
            axes = [axes]
        elif isinstance(axes, Variable):
            axes = axes.tolist()
        elif isinstance(axes, (list, tuple)):
            axes = [
                item.item(0) if isinstance(item, Variable) else item
                for item in axes
            ]
        return _C_ops.unsqueeze(input, axes)
    elif in_pir_mode():
        if isinstance(axes, int):
            axes = [axes]
        if isinstance(axes, paddle.pir.Value):
            axes.stop_gradient = True
        elif isinstance(axes, (list, tuple)):
            if paddle.utils._contain_var(axes):
                axes = paddle.utils.get_int_tensor_list(
                    axes, default_dtype='int64'
                )
        return _C_ops.unsqueeze(input, axes)
    else:
        check_type(axes, 'axis/axes', (int, list, tuple, Variable), 'unsqueeze')
        check_variable_and_dtype(
            input,
            'input',
            [
                'uint16',
                'float16',
                'uint16',
                'float32',
                'float64',
                'bool',
                'int8',
                'int16',
                'int32',
                'int64',
                'complex64',
                'complex128',
            ],
            'unsqueeze',
        )
        helper = LayerHelper("unsqueeze2", **locals())
        inputs = {"X": input}
        attrs = {}

        if isinstance(axes, int):
            axes = [axes]
        if isinstance(axes, Variable):
            axes.stop_gradient = True
            inputs["AxesTensor"] = axes
        elif isinstance(axes, (list, tuple)):
            if paddle.utils._contain_var(axes):
                inputs["AxesTensorList"] = paddle.utils._convert_to_tensor_list(
                    axes
                )
            else:
                attrs["axes"] = axes

        out = helper.create_variable_for_type_inference(dtype=input.dtype)
        x_shape = helper.create_variable_for_type_inference(dtype=input.dtype)
        helper.append_op(
            type="unsqueeze2",
            inputs=inputs,
            attrs=attrs,
            outputs={"Out": out, "XShape": x_shape},
        )

        return out


@inplace_apis_in_dygraph_only
def unsqueeze_(
    x: Tensor, axis: int | Sequence[int] | Tensor, name: str | None = None
) -> Tensor:
    """
    Inplace version of ``unsqueeze`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_paddle_tensor_unsqueeze`.
    """
    input = x
    axes = axis
    if isinstance(axes, int):
        axes = [axes]
    elif isinstance(axes, Variable):
        axes = axes.tolist()
    elif isinstance(axes, (list, tuple)):
        axes = [
            item.item(0) if isinstance(item, Variable) else item
            for item in axes
        ]
    return _C_ops.unsqueeze_(input, axes)


def gather(
    x: Tensor,
    index: Tensor,
    axis: Tensor | int | None = None,
    name: str | None = None,
) -> Tensor:
    """
    Output is obtained by gathering entries of ``axis``
    of ``x`` indexed by ``index`` and concatenate them together.

    .. code-block:: text


                Given:

                x = [[1, 2],
                     [3, 4],
                     [5, 6]]

                index = [1, 2]
                axis=[0]

                Then:

                out = [[3, 4],
                       [5, 6]]

    Args:
        x (Tensor): The source input tensor with rank>=1. Supported data type is
            int32, int64, float32, float64, complex64, complex128 and uint8 (only for CPU),
            float16 (only for GPU).
        index (Tensor): The index input tensor with rank=0 or rank=1. Data type is int32 or int64.
        axis (Tensor|int|None, optional): The axis of input to be gathered, it's can be int or a Tensor with data type is int32 or int64. The default value is None, if None, the ``axis`` is 0.
        name (str|None, optional): The default value is None.  Normally there is no need for user to set this property.
            For more information, please refer to :ref:`api_guide_Name` .

    Returns:
        output (Tensor), If the index is a 1-D tensor, the output is a tensor with the same shape as ``x``. If the index is a 0-D tensor, the output will reduce the dimension where the axis pointing.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> input = paddle.to_tensor([[1,2],[3,4],[5,6]])
            >>> index = paddle.to_tensor([0,1])
            >>> output = paddle.gather(input, index, axis=0)
            >>> print(output)
            Tensor(shape=[2, 2], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[1, 2],
             [3, 4]])
    """
    if axis is None:
        axis = 0

    if in_dynamic_or_pir_mode():
        return _C_ops.gather(x, index, axis)
    else:
        check_variable_and_dtype(
            x,
            'x',
            [
                'float16',
                'float32',
                'float64',
                'int16',
                'int32',
                'int64',
                'uint8',
                'uint16',
                'complex64',
                'complex128',
            ],
            'gather',
        )
        check_variable_and_dtype(index, 'index', ['int32', 'int64'], 'gather')

        if isinstance(axis, Variable):
            check_variable_and_dtype(axis, 'axis', ['int32', 'int64'], 'gather')

        helper = LayerHelper('gather', **locals())
        dtype = helper.input_dtype('x')
        out = helper.create_variable_for_type_inference(dtype)
        if not isinstance(axis, Variable):
            helper.append_op(
                type="gather",
                inputs={"X": x, "Index": index},
                attrs={'axis': axis, 'overwrite': False},
                outputs={"Out": out},
            )
        else:
            helper.append_op(
                type="gather",
                inputs={"X": x, "Index": index, "Axis": axis},
                attrs={"overwrite": False},
                outputs={"Out": out},
            )

        return out


def unbind(input: Tensor, axis: int = 0) -> list[Tensor]:
    """

    Removes a tensor dimension, then split the input tensor into multiple sub-Tensors.

    Args:
        input (Tensor): The input variable which is an N-D Tensor, data type being bool, float16, float32, float64, int32, int64, complex64 or complex128.
        axis (int, optional): A 0-D Tensor with shape [] and type is ``int32|int64``. The dimension along which to unbind.
            If :math:`axis < 0`, the dimension to unbind along is :math:`rank(input) + axis`. Default is 0.
    Returns:
        list(Tensor), The list of segmented Tensor variables.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> # input is a Tensor which shape is [3, 4, 5]
            >>> input = paddle.rand([3, 4, 5])

            >>> [x0, x1, x2] = paddle.unbind(input, axis=0)
            >>> # x0.shape [4, 5]
            >>> # x1.shape [4, 5]
            >>> # x2.shape [4, 5]

            >>> [x0, x1, x2, x3] = paddle.unbind(input, axis=1)
            >>> # x0.shape [3, 5]
            >>> # x1.shape [3, 5]
            >>> # x2.shape [3, 5]
            >>> # x3.shape [3, 5]
    """
    if not isinstance(axis, (int)):
        raise TypeError(
            f"The type of 'axis'  must be int, but received {type(axis)}."
        )

    if axis not in range(-input.ndim, input.ndim):
        raise ValueError(
            f'The axis must in range({-input.ndim}, {input.ndim}).'
        )

    if in_dynamic_or_pir_mode():
        return _C_ops.unbind(input, axis)
    else:
        if isinstance(axis, np.generic):
            axis = np.asscalar(axis)
        input_shape = input.shape
        axis_ = axis if axis >= 0 else len(input_shape) + axis
        num = input_shape[axis_]
        helper = LayerHelper("unbind", **locals())
        check_type(input, 'input', (Variable), 'unbind')
        dtype = helper.input_dtype()
        check_dtype(
            dtype,
            'unbind',
            [
                'bool',
                'float16',
                'uint16',
                'float32',
                'float64',
                'int32',
                'int64',
                'complex64',
                'complex128',
            ],
            'unbind',
        )
        outs = [
            helper.create_variable_for_type_inference(
                dtype=helper.input_dtype()
            )
            for i in range(num)
        ]
        helper.append_op(
            type="unbind",
            inputs={"X": input},
            outputs={"Out": outs},
            attrs={"axis": axis},
        )
        return outs


def scatter(
    x: Tensor,
    index: Tensor,
    updates: Tensor,
    overwrite: bool = True,
    name: str | None = None,
) -> Tensor:
    """
    **Scatter Layer**
    Output is obtained by updating the input on selected indices based on updates.

    As shown in the figure, when ``overwrite`` is set to ``True``, the output for the same index is updated in overwrite mode, where ``x[index[i]]`` is directly replaced with ``update[i]`` sequentially; When ``overwrite`` is set to ``False``, the output for the same index is updated in accumulation mode. In this mode, ``x[index[i]]`` is first initialized with elements set to 0. Then, ``update[i]`` is sequentially added to ``x[index[i]]`` to produce the output.

    .. image:: https://githubraw.cdn.bcebos.com/PaddlePaddle/docs/develop/docs/images/api_legend/scatter.png
        :alt: Legend - scatter behavior display

    .. code-block:: python
        :name: scatter-example-1

        >>> import paddle
        >>> #input:
        >>> x = paddle.to_tensor([[1, 1], [2, 2], [3, 3]], dtype='float32')
        >>> index = paddle.to_tensor([2, 1, 0, 1], dtype='int64')
        >>> # shape of updates should be the same as x
        >>> # shape of updates with dim > 1 should be the same as input
        >>> updates = paddle.to_tensor([[1, 1], [2, 2], [3, 3], [4, 4]], dtype='float32')
        >>> overwrite = False
        >>> # calculation:
        >>> if not overwrite:
        ...     for i in range(len(index)):
        ...         x[index[i]] = paddle.zeros([2])
        >>> for i in range(len(index)):
        ...     if (overwrite):
        ...         x[index[i]] = updates[i]
        ...     else:
        ...         x[index[i]] += updates[i]
        >>> # output:
        >>> out = paddle.to_tensor([[3, 3], [6, 6], [1, 1]])
        >>> print(out.shape)
        [3, 2]

    **NOTICE**: The order in which updates are applied is nondeterministic,
    so the output will be nondeterministic if index contains duplicates.

    Args:
        x (Tensor): The input N-D Tensor with ndim>=1. Data type can be float32, float64.
        index (Tensor): The index is a 1-D or 0-D Tensor. Data type can be int32, int64. The length of index cannot exceed updates's length, and the value in index cannot exceed input's length.
        updates (Tensor): Update input with updates parameter based on index. When the index is a 1-D tensor, the updates shape should be the same as input, and dim value with dim > 1 should be the same as input. When the index is a 0-D tensor, the updates should be a (N-1)-D tensor, the ith dim of the updates should be equal with the (i+1)th dim of the input.
        overwrite (bool, optional): The mode that updating the output when there are same indices.If True, use the overwrite mode to update the output of the same index,if False, use the accumulate mode to update the output of the same index. Default value is True.
        name(str|None, optional): The default value is None. Normally there is no need for user to set this property.  For more information, please refer to :ref:`api_guide_Name` .

    Returns:
        Tensor, The output is a Tensor with the same shape as x.

    Examples:
        .. code-block:: python
            :name: scatter-example-2

            >>> import paddle

            >>> x = paddle.to_tensor([[1, 1], [2, 2], [3, 3]], dtype='float32')
            >>> index = paddle.to_tensor([2, 1, 0, 1], dtype='int64')
            >>> updates = paddle.to_tensor([[1, 1], [2, 2], [3, 3], [4, 4]], dtype='float32')

            >>> output1 = paddle.scatter(x, index, updates, overwrite=False)
            >>> print(output1)
            Tensor(shape=[3, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[3., 3.],
             [6., 6.],
             [1., 1.]])

            >>> output2 = paddle.scatter(x, index, updates, overwrite=True)
            >>> # CPU device:
            >>> # [[3., 3.],
            >>> #  [4., 4.],
            >>> #  [1., 1.]]
            >>> # GPU device maybe have two results because of the repeated numbers in index
            >>> # result 1:
            >>> # [[3., 3.],
            >>> #  [4., 4.],
            >>> #  [1., 1.]]
            >>> # result 2:
            >>> # [[3., 3.],
            >>> #  [2., 2.],
            >>> #  [1., 1.]]
    """
    if in_dynamic_or_pir_mode():
        return _C_ops.scatter(x, index, updates, overwrite)
    else:
        check_variable_and_dtype(
            x,
            'dtype',
            ['float32', 'float64', 'float16', 'int32', 'int64', 'uint16'],
            'scatter',
        )
        check_type(overwrite, 'overwrite', bool, 'scatter')
        helper = LayerHelper('scatter', **locals())
        out = helper.create_variable_for_type_inference(x.dtype)
        helper.append_op(
            type="scatter",
            inputs={"X": x, "Ids": index, "Updates": updates},
            attrs={'overwrite': overwrite},
            outputs={"Out": out},
        )
        return out


@inplace_apis_in_dygraph_only
def scatter_(
    x: Tensor,
    index: Tensor,
    updates: Tensor,
    overwrite: bool = True,
    name: str | None = None,
) -> Tensor:
    """
    Inplace version of ``scatter`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_paddle_tensor_scatter`.
    """
    return _C_ops.scatter_(x, index, updates, overwrite)


def scatter_nd_add(
    x: Tensor, index: Tensor, updates: Tensor, name: str | None = None
) -> Tensor:
    r"""

    Output is obtained by applying sparse addition to a single value
    or slice in a Tensor.

    :attr:`x` is a Tensor with ndim :math:`R`
    and :attr:`index` is a Tensor with ndim :math:`K` . Thus, :attr:`index`
    has shape :math:`[i_0, i_1, ..., i_{K-2}, Q]` where :math:`Q \leq R` . :attr:`updates`
    is a Tensor with ndim :math:`K - 1 + R - Q` and its
    shape is :math:`index.shape[:-1] + x.shape[index.shape[-1]:]` .

    According to the :math:`[i_0, i_1, ..., i_{K-2}]` of :attr:`index` ,
    add the corresponding :attr:`updates` slice to the :attr:`x` slice
    which is obtained by the last one dimension of :attr:`index` .

    .. code-block:: text

        Given:

        * Case 1:
            x = [0, 1, 2, 3, 4, 5]
            index = [[1], [2], [3], [1]]
            updates = [9, 10, 11, 12]

          we get:

            output = [0, 22, 12, 14, 4, 5]

        * Case 2:
            x = [[65, 17], [-14, -25]]
            index = [[], []]
            updates = [[[-1, -2], [1, 2]],
                       [[3, 4], [-3, -4]]]
            x.shape = (2, 2)
            index.shape = (2, 0)
            updates.shape = (2, 2, 2)

          we get:

            output = [[67, 19], [-16, -27]]

    Args:
        x (Tensor): The x input. Its dtype should be int32, int64, float32, float64.
        index (Tensor): The index input with ndim > 1 and index.shape[-1] <= x.ndim.
                          Its dtype should be int32 or int64 as it is used as indexes.
        updates (Tensor): The updated value of scatter_nd_add op, and it must have the same dtype
                            as x. It must have the shape index.shape[:-1] + x.shape[index.shape[-1]:].
        name (str|None, optional): The output tensor name. If set None, the layer will be named automatically.

    Returns:
        output (Tensor), The output is a tensor with the same shape and dtype as x.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> x = paddle.rand(shape=[3, 5, 9, 10], dtype='float32')
            >>> updates = paddle.rand(shape=[3, 9, 10], dtype='float32')
            >>> index = paddle.to_tensor([[1, 1],
            ...                           [0, 1],
            ...                           [1, 3]], dtype='int64')

            >>> output = paddle.scatter_nd_add(x, index, updates)
            >>> print(output.shape)
            [3, 5, 9, 10]
    """
    if x.dtype != updates.dtype:
        raise TypeError(
            f"x and updates must have same data type but x.dtype={convert_dtype(x.dtype)}, updates.dtype={convert_dtype(updates.dtype)}"
        )

    if in_dynamic_or_pir_mode():
        return _C_ops.scatter_nd_add(x, index, updates)
    else:
        helper = LayerHelper('scatter_nd_add', **locals())
        dtype = helper.input_dtype(input_param_name='x')
        output = helper.create_variable_for_type_inference(dtype)
        helper.append_op(
            type="scatter_nd_add",
            inputs={"X": x, "Index": index, "Updates": updates},
            outputs={"Out": output},
        )
        return output


def scatter_nd(
    index: Tensor, updates: Tensor, shape: ShapeLike, name: str | None = None
) -> Tensor:
    """
    **Scatter_nd Layer**

    Output is obtained by scattering the :attr:`updates` in a new tensor according
    to :attr:`index` . This op is similar to :code:`scatter_nd_add`, except the
    tensor of :attr:`shape` is zero-initialized. Correspondingly, :code:`scatter_nd(index, updates, shape)`
    is equal to :code:`scatter_nd_add(paddle.zeros(shape, updates.dtype), index, updates)` .
    If :attr:`index` has repeated elements, then the corresponding updates are accumulated.
    Because of the numerical approximation issues, the different order of repeated elements
    in :attr:`index` may cause different results. The specific calculation method can be
    seen :code:`scatter_nd_add` . This op is the inverse of the :code:`gather_nd` op.

    Args:
        index (Tensor): The index input with ndim >= 1 and index.shape[-1] <= len(shape).
                          Its dtype should be int32 or int64 as it is used as indexes.
        updates (Tensor): The updated value of scatter_nd op. Its dtype should be float32, float64.
                            It must have the shape index.shape[:-1] + shape[index.shape[-1]:]
        shape(tuple|list|Tensor): Shape of output tensor.
        name (str|None, optional): The output Tensor name. If set None, the layer will be named automatically.

    Returns:
        output (Tensor), The output is a tensor with the same type as :attr:`updates` .

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> index = paddle.to_tensor([[1, 1],
            ...                           [0, 1],
            ...                           [1, 3]], dtype="int64")
            >>> updates = paddle.rand(shape=[3, 9, 10], dtype='float32')
            >>> shape = [3, 5, 9, 10]

            >>> output = paddle.scatter_nd(index, updates, shape)
    """
    return scatter_nd_add(zeros(shape, updates.dtype), index, updates, name)


def chunk(
    x: Tensor, chunks: int, axis: int | Tensor = 0, name: str | None = None
) -> list[Tensor]:
    """
    Split the input tensor into multiple sub-Tensors.

    Here are some examples to explain it.

        - 1. Given a 3-D tensor x with a shape [3, 3, 3], if we split the first dimension into three equal parts, it will output a list containing three 3-D tensors with a shape of [1, 3, 3].
        - 2. Given a 3-D tensor x with a shape [3, 3, 3], if we split the second dimension into three equal parts, it will output a list containing three 3-D tensors with a shape of [3, 1, 3].
        - 3. Given a 3-D tensor x with a shape [3, 3, 3], if we split the third dimension into three equal parts, it will output a list containing three 3-D tensors with a shape of [3, 3, 1].

    The following figure illustrates the first example.

    .. image:: https://githubraw.cdn.bcebos.com/PaddlePaddle/docs/develop/docs/images/chunk.png
        :width: 800
        :alt: legend of reshape API
        :align: center

    Args:
        x (Tensor): A N-D Tensor. The data type is bool, float16, float32, float64, int32 or int64.
        chunks(int): The number of tensor to be split along the certain axis.
        axis (int|Tensor, optional): The axis along which to split, it can be a integer or a ``0-D Tensor``
            with shape [] and data type  ``int32`` or ``int64``.
            If :math::`axis < 0`, the axis to split along is :math:`rank(x) + axis`. Default is 0.
        name (str|None, optional): The default value is None.  Normally there is no need for user to set this property.
            For more information, please refer to :ref:`api_guide_Name` .
    Returns:
        list(Tensor), The list of segmented Tensors.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.rand([3, 9, 5])

            >>> out0, out1, out2 = paddle.chunk(x, chunks=3, axis=1)
            >>> # out0.shape [3, 3, 5]
            >>> # out1.shape [3, 3, 5]
            >>> # out2.shape [3, 3, 5]


            >>> # axis is negative, the real axis is (rank(x) + axis) which real
            >>> # value is 1.
            >>> out0, out1, out2 = paddle.chunk(x, chunks=3, axis=-2)
            >>> # out0.shape [3, 3, 5]
            >>> # out1.shape [3, 3, 5]
            >>> # out2.shape [3, 3, 5]
    """
    check_type(chunks, 'chunks', (int), 'chunk')
    return split(x, num_or_sections=chunks, axis=axis, name=name)


def tile(
    x: Tensor,
    repeat_times: TensorOrTensors | Sequence[int],
    name: str | None = None,
) -> Tensor:
    """

    Construct a new Tensor by repeating ``x`` the number of times given by ``repeat_times``.
    After tiling, the value of the i'th dimension of the output is equal to ``x.shape[i]*repeat_times[i]``.

    Both the number of dimensions of ``x`` and the number of elements in ``repeat_times`` should be less than or equal to 6.

    Args:
        x (Tensor): The input tensor, its data type should be bool, float16, float32, float64, int32, int64, complex64 or complex128.
        repeat_times (list|tuple|Tensor): The number of repeating times. If repeat_times is a list or tuple, all its elements
            should be integers or 1-D Tensors with the data type int32. If repeat_times is a Tensor, it should be an 1-D Tensor with the data type int32.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        N-D Tensor. The data type is the same as ``x``. The size of the i-th dimension is equal to ``x[i] * repeat_times[i]``.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> data = paddle.to_tensor([1, 2, 3], dtype='int32')
            >>> out = paddle.tile(data, repeat_times=[2, 1])
            >>> print(out)
            Tensor(shape=[2, 3], dtype=int32, place=Place(cpu), stop_gradient=True,
            [[1, 2, 3],
             [1, 2, 3]])

            >>> out = paddle.tile(data, repeat_times=(2, 2))
            >>> print(out)
            Tensor(shape=[2, 6], dtype=int32, place=Place(cpu), stop_gradient=True,
            [[1, 2, 3, 1, 2, 3],
             [1, 2, 3, 1, 2, 3]])

            >>> repeat_times = paddle.to_tensor([1, 2], dtype='int32')
            >>> out = paddle.tile(data, repeat_times=repeat_times)
            >>> print(out)
            Tensor(shape=[1, 6], dtype=int32, place=Place(cpu), stop_gradient=True,
            [[1, 2, 3, 1, 2, 3]])
    """

    def check_input(x, repeat_times):
        check_type(
            repeat_times,
            'repeat_times',
            (list, tuple, Variable, paddle.pir.Value),
            'tile',
        )
        if isinstance(repeat_times, (Variable, paddle.pir.Value)):
            assert (
                len(repeat_times.shape) == 1
            ), 'repeat_times must be a Tensor with ndim == 1.'
        else:
            for elem in repeat_times:
                if isinstance(elem, (Variable, paddle.pir.Value)):
                    numel = functools.reduce(lambda x, y: x * y, elem.shape, 1)
                    assert (
                        numel == 1
                    ), 'Elements in repeat_times must be Tensor with one element or integers.'
                else:
                    type_tuple = (int, np.int32, np.int64)
                    assert isinstance(
                        elem, type_tuple
                    ), 'Elements in repeat_times must be Tensor with one element or integers.'

        check_variable_and_dtype(
            x,
            'x',
            [
                'bool',
                'float16',
                'uint16',
                'float32',
                'float64',
                'int32',
                'int64',
                'complex64',
                'complex128',
            ],
            'tile',
        )
        if convert_dtype(x.dtype) == 'bool' and not x.stop_gradient:
            raise ValueError(
                "When the date type is bool for the input 'x' of tile op, you "
                "must set its stop_gradient to be True by "
                "some_var.stop_gradient == True supporting some_var is the input."
            )

    if in_dynamic_mode():
        if isinstance(repeat_times, core.eager.Tensor):
            assert (
                repeat_times.ndim == 1
            ), "Only support ndim == 1 while repeat_times is a Tensor."
            repeat_times = repeat_times.tolist()

        return _C_ops.tile(x, repeat_times)
    elif in_pir_mode():
        check_input(x, repeat_times)
        if isinstance(repeat_times, (list, tuple)):
            if paddle.utils._contain_var(repeat_times):
                repeat_times = paddle.utils.get_int_tensor_list(repeat_times)
        return _C_ops.tile(x, repeat_times)
    else:
        check_input(x, repeat_times)

        def get_attr_repeat_times(list_repeat_times):
            attrs_repeat_times = []
            for idx, times in enumerate(list_repeat_times):
                if isinstance(times, Variable):
                    attrs_repeat_times.append(-1)
                else:
                    attrs_repeat_times.append(times)
                    assert (
                        times > 0
                    ), "All elements in repeat_times must be positive for tile."
            return attrs_repeat_times

        helper = LayerHelper('tile', **locals())

        inputs = {"X": [x]}
        attrs = {}

        if isinstance(repeat_times, Variable):
            repeat_times.stop_gradient = True
            inputs['RepeatTimes'] = repeat_times
            attrs['repeat_times'] = [-1]
        elif isinstance(repeat_times, (list, tuple)):
            attrs['repeat_times'] = get_attr_repeat_times(repeat_times)
            if paddle.utils._contain_var(repeat_times):
                inputs['repeat_times_tensor'] = (
                    paddle.utils._convert_to_tensor_list(repeat_times)
                )

        dtype = helper.input_dtype(input_param_name='x')
        out = helper.create_variable_for_type_inference(dtype)
        helper.append_op(
            type='tile', inputs=inputs, outputs={'Out': out}, attrs=attrs
        )
        return out


def expand_as(x: Tensor, y: Tensor, name: str | None = None) -> Tensor:
    """

    Expand the input tensor ``x`` to the same shape as the input tensor ``y``.

    Both the number of dimensions of ``x`` and ``y`` must be less than or equal to 6, and the number of dimensions of ``y`` must be greater than or equal to that of ``x``. The dimension to expand must have a value of 0.

    The following diagram illustrates how a one-dimensional tensor is transformed into a tensor with a shape of [2,3] through the expand_as operation. The target tensor has a shape of [2,3], and through expand_as, the one-dimensional tensor is expanded into a tensor with a shape of [2,3].

    .. image:: https://githubraw.cdn.bcebos.com/PaddlePaddle/docs/develop/docs/images/api_legend/expand_as.png
        :width: 800
        :alt: expand_as API
        :align: center

    Args:
        x (Tensor): The input tensor, its data type is bool, float32, float64, int32 or int64.
        y (Tensor): The input tensor that gives the shape to expand to.
        name (str|None, optional): The default value is None. Normally there is no need for user to set this property. For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        N-D Tensor, A Tensor with the same shape as ``y``. The data type is the same as ``x``.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> data_x = paddle.to_tensor([1, 2, 3], 'int32')
            >>> data_y = paddle.to_tensor([[1, 2, 3], [4, 5, 6]], 'int32')
            >>> out = paddle.expand_as(data_x, data_y)
            >>> print(out)
            Tensor(shape=[2, 3], dtype=int32, place=Place(cpu), stop_gradient=True,
            [[1, 2, 3],
             [1, 2, 3]])
    """
    if in_dynamic_mode():
        return _C_ops.expand_as(x, None, y.shape)
    elif in_pir_mode():
        if convert_dtype(x.dtype) == 'bool' and not x.stop_gradient:
            raise ValueError(
                "When the data type of input 'x' for expand_as is bool, "
                "you must set its stop_gradient to be False by "
                "some_var.stop_gradient = True, supporting "
                "some_var as the input 'x'."
            )
        return _C_ops.expand_as(x, y, y.shape)
    else:
        check_variable_and_dtype(
            x,
            'x',
            [
                'bool',
                'float32',
                'float64',
                'int32',
                'int64',
                'float16',
                'uint16',
            ],
            'expand_as',
        )
        check_type(y, 'y', Variable, 'expand_as')

        if convert_dtype(x.dtype) == 'bool' and not x.stop_gradient:
            raise ValueError(
                "When the data type of input 'x' for expand_as is bool, "
                "you must set its stop_gradient to be False by "
                "some_var.stop_gradient = True, supporting "
                "some_var as the input 'x'."
            )
        inputs = {"X": [x], "Y": [y]}

        helper = LayerHelper('expand_as', **locals())
        dtype = helper.input_dtype(input_param_name='x')
        out = helper.create_variable_for_type_inference(dtype)
        helper.append_op(
            type='expand_as_v2',
            inputs=inputs,
            attrs={'target_shape': y.shape},
            outputs={'Out': out},
        )
        return out


def broadcast_to(
    x: Tensor, shape: ShapeLike, name: str | None = None
) -> Tensor:
    """

    Broadcast the input tensor to a given shape.

    Both the number of dimensions of ``x`` and the number of elements in ``shape`` should be less than or equal to 6. The dimension to broadcast to must have a value 0.

    The following figure shows the process of broadcasting a one-dimensional tensor of shape [3] to a two-dimensional tensor of shape [2,3] based on the shape specified by 'shape'.

    .. image:: https://githubraw.cdn.bcebos.com/PaddlePaddle/docs/develop/docs/images/api_legend/broadcast_to.png
        :width: 500
        :alt: broadcast_to API
        :align: center

    Args:
        x (Tensor): The input tensor, its data type is bool, float16, float32, float64, int32, int64, uint8 or uint16.
        shape (list|tuple|Tensor): The result shape after broadcasting. The data type is int32. If shape is a list or tuple, all its elements
            should be integers or 0-D or 1-D Tensors with the data type int32. If shape is a Tensor, it should be an 1-D Tensor with the data type int32.
            The value -1 in shape means keeping the corresponding dimension unchanged.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
    Returns:
        N-D Tensor, A Tensor with the given shape. The data type is the same as ``x``.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> data = paddle.to_tensor([1, 2, 3], dtype='int32')
            >>> out = paddle.broadcast_to(data, shape=[2, 3])
            >>> print(out)
            Tensor(shape=[2, 3], dtype=int32, place=Place(cpu), stop_gradient=True,
            [[1, 2, 3],
             [1, 2, 3]])
    """
    return expand(x, shape, name)


def expand(x: Tensor, shape: ShapeLike, name: str | None = None) -> Tensor:
    """

    Expand the input tensor to a given shape.

    Both the number of dimensions of ``x`` and the number of elements in ``shape`` should be less than or equal to 6. And the number of dimensions of ``x`` should be less than the number of elements in ``shape``. The dimension to expand must have a value 0.

    The image illustrates a typical case of the expand operation.
    The Original Tensor is a 1D tensor with shape ``[3]`` and values [1, 2, 3]. Using the ``paddle.expand`` method with the parameter ``shape = [2, 3]``, it is broadcasted and expanded into a 2D tensor with shape ``[2, 3]``

    .. image:: https://githubraw.cdn.bcebos.com/PaddlePaddle/docs/develop/docs/images/api_legend/expand.png
        :width: 500
        :alt: legend of expand API
        :align: center


    Args:
        x (Tensor): The input Tensor, its data type is bool, float16, float32, float64, int32, int64, uint8, uint16, complex64 or complex128.
        shape (list|tuple|Tensor): The result shape after expanding. The data type is int32. If shape is a list or tuple, all its elements
            should be integers or 0-D or 1-D Tensors with the data type int32. If shape is a Tensor, it should be an 1-D Tensor with the data type int32.
            The value -1 in shape means keeping the corresponding dimension unchanged.
        name (str|None, optional): The default value is None. Normally there is no need for user to set this property. For more information, please refer to :ref:`api_guide_Name` .

    Returns:
        N-D Tensor, A Tensor with the given shape. The data type is the same as ``x``.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> data = paddle.to_tensor([1, 2, 3], dtype='int32')
            >>> out = paddle.expand(data, shape=[2, 3])
            >>> print(out)
            Tensor(shape=[2, 3], dtype=int32, place=Place(cpu), stop_gradient=True,
            [[1, 2, 3],
             [1, 2, 3]])
    """
    if in_dynamic_mode():
        return _C_ops.expand(x, shape)
    elif in_pir_mode():
        if convert_dtype(x.dtype) == 'bool' and not x.stop_gradient:
            raise ValueError(
                "When the data type of input 'x' for expand is bool, "
                "you must set its stop_gradient to be False by "
                "some_var.stop_gradient = True, supporting "
                "some_var as the input."
            )
        if isinstance(shape, paddle.pir.Value):
            shape.stop_gradient = True
        elif isinstance(shape, (list, tuple)):
            if paddle.utils._contain_var(shape):
                shape = paddle.utils.get_int_tensor_list(shape)
        else:
            raise TypeError("Shape only supports Value, or list, or tuple.")
        return _C_ops.expand(x, shape)
    else:
        if isinstance(shape, Variable):
            assert len(shape.shape) == 1, 'shape must be a 1-D Tensor.'
        else:
            for elem in shape:
                if isinstance(elem, Variable):
                    assert (
                        elem.numel() == 1
                    ), 'Elements in shape must be Tensor with one element or integers.'
                else:
                    type_tuple = (int, np.int32, np.int64)
                    assert isinstance(
                        elem, type_tuple
                    ), 'Elements in shape must be Tensor with one element or integers.'

        check_variable_and_dtype(
            x,
            'x',
            [
                'bool',
                'float16',
                'float32',
                'float64',
                'int32',
                'int64',
                'uint8',
                'uint16',
                'complex64',
                'complex128',
            ],
            'expand',
        )
        check_type(shape, 'shape', (list, tuple, Variable), 'expand')
        if convert_dtype(x.dtype) == 'bool' and not x.stop_gradient:
            raise ValueError(
                "When the data type of input 'x' for expand is bool, "
                "you must set its stop_gradient to be False by "
                "some_var.stop_gradient = True, supporting "
                "some_var as the input."
            )

        inputs = {"X": [x]}
        attrs = {}

        helper = LayerHelper('expand', **locals())

        def get_attr_expand_shape(list_expand_shape):
            attrs_expand_shape = []
            for idx, shape in enumerate(list_expand_shape):
                if isinstance(shape, Variable):
                    attrs_expand_shape.append(-2)
                else:
                    attrs_expand_shape.append(shape)
                    assert (
                        shape > 0 or shape == -1
                    ), "All elements in shape of expand must be positive or -1."
            return attrs_expand_shape

        if isinstance(shape, Variable):
            shape.stop_gradient = True
            inputs['Shape'] = shape
        elif isinstance(shape, (list, tuple)):
            attrs['shape'] = get_attr_expand_shape(shape)
            if paddle.utils._contain_var(shape):
                inputs['expand_shapes_tensor'] = (
                    paddle.utils._convert_to_tensor_list(shape)
                )

        dtype = helper.input_dtype(input_param_name='x')
        out = helper.create_variable_for_type_inference(dtype)
        helper.append_op(
            type='expand_v2', inputs=inputs, outputs={'Out': out}, attrs=attrs
        )
        return out


def reshape(x: Tensor, shape: ShapeLike, name: str | None = None) -> Tensor:
    """
    Changes the shape of ``x`` without changing its data.

    Note that the output Tensor will share data with origin Tensor and doesn't
    have a Tensor copy in ``dygraph`` mode.
    If you want to use the Tensor copy version, please use `Tensor.clone` like
    ``reshape_clone_x = x.reshape([-1]).clone()``.

    Some tricks exist when specifying the target shape.

        - 1. -1 means the value of this dimension is inferred from the total element number of x and remaining dimensions. Thus one and only one dimension can be set -1.

        - 2. 0 means the actual dimension value is going to be copied from the corresponding dimension of x. The index of 0s in shape can not exceed the dimension of x.

    Here are some examples to explain it.

        - 1. Given a 3-D tensor x with a shape [2, 4, 6], and the target shape is [6, 8], the reshape operator will transform x into a 2-D tensor with shape [6, 8] and leaving x's data unchanged.

        - 2. Given a 3-D tensor x with a shape [2, 4, 6], and the target shape specified is [2, 3, -1, 2], the reshape operator will transform x into a 4-D tensor with shape [2, 3, 4, 2] and leaving x's data unchanged. In this case, one dimension of the target shape is set to -1, the value of this dimension is inferred from the total element number of x and remaining dimensions.

        - 3. Given a 3-D tensor x with a shape [2, 4, 6], and the target shape is [-1, 0, 3, 2], the reshape operator will transform x into a 4-D tensor with shape [2, 4, 3, 2] and leaving x's data unchanged. In this case, besides -1, 0 means the actual dimension value is going to be copied from the corresponding dimension of x.

    The following figure illustrates the first example -- a 3D tensor of shape [2, 4, 6] is transformed into a 2D tensor of shape [6, 8], during which the order and values of the elements in the tensor remain unchanged. The elements in the two subdiagrams correspond to each other, clearly demonstrating how the reshape API works.

    .. image:: https://githubraw.cdn.bcebos.com/PaddlePaddle/docs/develop/docs/images/api_legend/reshape.png
        :width: 800
        :alt: legend of reshape API
        :align: center

    Args:
        x (Tensor): An N-D Tensor. The data type is ``float16``, ``float32``, ``float64``, ``int16``, ``int32``, ``int64``, ``int8``, ``uint8``, ``complex64``, ``complex128``, ``bfloat16`` or ``bool``.
        shape (list|tuple|Tensor): Define the target shape. At most one dimension of the target shape can be -1.
                        The data type is ``int32`` . If ``shape`` is a list or tuple, each element of it should be integer or Tensor with shape [].
                        If ``shape`` is a Tensor, it should be an 1-D Tensor .
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, A reshaped Tensor with the same data type as ``x``.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.rand([2, 4, 6], dtype="float32")
            >>> positive_four = paddle.full([1], 4, "int32")

            >>> out = paddle.reshape(x, [-1, 0, 3, 2])
            >>> print(out.shape)
            [2, 4, 3, 2]

            >>> out = paddle.reshape(x, shape=[positive_four, 12])
            >>> print(out.shape)
            [4, 12]

            >>> shape_tensor = paddle.to_tensor([8, 6], dtype=paddle.int32)
            >>> out = paddle.reshape(x, shape=shape_tensor)
            >>> print(out.shape)
            [8, 6]
            >>> # out shares data with x in dygraph mode
            >>> x[0, 0, 0] = 10.
            >>> print(out[0, 0])
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
            10.)

    """

    def get_attr_shape(list_shape):
        unk_dim_idx = -1
        attrs_shape = []
        for dim_idx, dim_size in enumerate(list_shape):
            if isinstance(dim_size, (Variable, paddle.pir.Value)):
                attrs_shape.append(-1)
            else:
                attrs_shape.append(dim_size)
                if dim_size == -1:
                    assert unk_dim_idx == -1, (
                        "Only one dimension value of 'shape' in reshape can "
                        f"be -1. But received shape[{dim_idx}] is also -1.\n"
                        "\n\t# N = x.shape()[2]\t\t# N is an int. "
                        "(NOT recommend under @to_static)\n\tN = paddle.shape(x)[2]\t\t"
                        "# N is a Tensor. (Recommend)\n\tz = paddle.reshape([N, -1, 4])"
                        "\t# z.shape is [-1, -1, 4]\n\n"
                        "    If your target shape in Reshape represents dynamic shape, "
                        "please turn it into a Tensor under @to_static. See above example for details."
                    )
                    unk_dim_idx = dim_idx
                elif dim_size == 0:
                    if math.prod(x.shape):
                        assert dim_idx < len(x.shape), (
                            "The index of 0 in `shape` must be less than "
                            "the input tensor X's dimensions. "
                            f"But received shape[{dim_idx}] = 0, X's dimensions = {len(x.shape)}."
                        )
                else:
                    assert dim_size > 0, (
                        "Each dimension value of 'shape' in reshape must not "
                        "be negative except one unknown dimension. "
                        f"But received shape[{dim_idx}] = {dim_size!s}."
                    )
        return attrs_shape

    if in_dynamic_mode():
        if isinstance(shape, (list, tuple)):
            new_shape = []
            for ele in shape:
                if isinstance(ele, core.eager.Tensor):
                    new_shape.append(ele.item())
                else:
                    new_shape.append(ele)
            if new_shape == x.shape:
                out = x
            else:
                out = _C_ops.reshape(x, new_shape)
        elif isinstance(shape, core.eager.Tensor):
            shape.stop_gradient = True
            out = _C_ops.reshape(x, shape)
        else:
            raise ValueError(
                "shape must be an instance of `list`, `tuple` `Variable`,"
                f" got '{type(shape)}.'"
            )
        return out
    elif in_pir_mode():
        check_variable_and_dtype(
            x,
            'x',
            [
                'float16',
                'float32',
                'float64',
                'int8',
                'uint8',
                'int16',
                'int32',
                'int64',
                'bool',
                'uint16',
                'complex64',
                'complex128',
            ],
            'reshape',
        )
        check_type(shape, 'shape', (list, tuple, paddle.pir.Value), 'reshape')
        if isinstance(shape, (list, tuple)):
            if paddle.utils._contain_var(shape):
                new_shape = paddle.utils.get_int_tensor_list(shape)
            else:
                new_shape = get_attr_shape(shape)
            out = _C_ops.reshape(x, new_shape)
        elif isinstance(shape, paddle.pir.Value):
            shape.stop_gradient = True
            out = _C_ops.reshape(x, shape)
        else:
            raise ValueError(
                "shape must be an instance of `list`, `tuple` `Value(in pir mode)`,"
                f" got '{type(shape)}.'"
            )

        return out
    else:
        check_variable_and_dtype(
            x,
            'x',
            [
                'float16',
                'float32',
                'float64',
                'int16',
                'int32',
                'int64',
                'bool',
                'uint16',
                'int8',
                'uint8',
                'complex64',
                'complex128',
            ],
            'reshape',
        )
        check_type(shape, 'shape', (list, tuple, Variable), 'reshape')

        inputs = {"X": x}
        attrs = {}
        if isinstance(shape, Variable):
            shape.stop_gradient = True
            inputs["Shape"] = shape
        elif isinstance(shape, (list, tuple)):
            attrs["shape"] = get_attr_shape(shape)
            if paddle.utils._contain_var(shape):
                inputs['ShapeTensor'] = paddle.utils._convert_to_tensor_list(
                    shape
                )

        helper = LayerHelper("reshape2", **locals())
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
        x_shape = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(
            type="reshape2",
            inputs=inputs,
            attrs=attrs,
            outputs={"Out": out, "XShape": x_shape},
        )

        return out


def masked_scatter(
    x: Tensor, mask: Tensor, value: Tensor, name: str | None = None
) -> Tensor:
    """
    Copies elements from `value` into `x` tensor at positions where the `mask` is True.

    Elements from source are copied into `x` starting at position 0 of `value` and continuing in order one-by-one for
    each occurrence of `mask` being True. The shape of `mask` must be broadcastable with the shape of the underlying tensor.
    The `value` should have at least as many elements as the number of ones in `mask`.

    The image illustrates a typical case of the masked_scatter operation.

      1. Tensor  ``value``: Contains the data to be filled into the target tensor. Only the parts where the mask is True will take values from the value tensor, while the rest will be ignored;
      2. Tensor  ``mask``: Specifies which positions should extract values from the value tensor and update the target tensor. True indicates the corresponding position needs to be updated;
      3. Tensor  ``origin``: The input tensor, where only the parts satisfying the mask will be replaced, and the rest remains unchanged;

    Result: After the ``masked_scatter`` operation, the parts of the ``origin`` tensor where the ``mask`` is ``True`` are updated with the corresponding values from the ``value`` tensor, while the parts where the ``mask`` is ``False`` remain unchanged, forming the final updated tensor.

    .. image:: https://githubraw.cdn.bcebos.com/PaddlePaddle/docs/develop/docs/images/api_legend/masked_scatter.png
        :width: 500
        :alt: legend of masked_scatter API
        :align: center

    Args:
        x (Tensor): An N-D Tensor. The data type is ``float16``, ``float32``, ``float64``, ``int32``,
            ``int64`` or ``bfloat16``.
        mask (Tensor): The boolean tensor indicate the position to be filled.
            The data type of mask must be bool.
        value (Tensor): The value used to fill the target tensor.
            Supported data types are same as x.
        name (str|None, optional): Name for the operation (optional, default is None). For more information,
            please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, A reshaped Tensor with the same data type as ``x``.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.seed(2048)
            >>> x = paddle.randn([2, 2])
            >>> print(x)
            Tensor(shape=[2, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
                [[-1.24725831,  0.03843464],
                [-0.31660911,  0.04793844]])

            >>> mask = paddle.to_tensor([[True, True], [False, False]])
            >>> value = paddle.to_tensor([1, 2, 3, 4, 5,], dtype="float32")

            >>> out = paddle.masked_scatter(x, mask, value)
            >>> print(out)
            Tensor(shape=[2, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
                [[1,  2],
                [-0.31660911,  0.04793844]])

    """
    # make sure the dtype of x and value is the same
    assert (
        x.dtype == value.dtype
    ), f'x and value must have the same dtype, but got x dtype is {x.dtype}, value dtype is {value.dtype}'
    assert mask.dtype == paddle.bool

    zeros_like_x = paddle.zeros_like(x, dtype=int)
    mask = paddle.add(paddle.cast(mask, dtype="int"), zeros_like_x)
    mask_prefix = paddle.clip(mask.cumsum() - 1, min=0)
    if in_dynamic_mode():
        assert (
            mask_prefix[-1] <= value.numel()
        ), f'mask true nums must be <= value size, but got mask true nums is {mask_prefix[-1].item()}, value size is {value.numel().item()}'

    value = value.flatten()[mask_prefix].reshape(mask.shape)
    mask = paddle.logical_not(mask.astype(bool))
    return paddle.where(mask, x, value)


@inplace_apis_in_dygraph_only
def masked_scatter_(
    x: Tensor, mask: Tensor, value: Tensor, name: str | None = None
) -> Tensor:
    """
    Inplace version of ``masked_scatter`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_paddle_masked_scatter`.
    """
    assert (
        x.dtype == value.dtype
    ), f'x and value must have the same dtype, but got x dtype is {x.dtype}, value dtype is {value.dtype}'
    assert mask.dtype == paddle.bool
    zeros_like_x = paddle.zeros_like(x, dtype=int)
    mask = paddle.add(paddle.cast(mask, dtype="int"), zeros_like_x)
    mask_prefix = paddle.clip(mask.cumsum() - 1, min=0)
    assert (
        mask_prefix[-1] <= value.numel()
    ), f'mask true nums must be <= value size, but got mask true nums is {mask_prefix[-1].item()}, value size is {value.numel().item()}'

    value = value.flatten()[mask_prefix].reshape(mask.shape)
    mask = paddle.logical_not(mask.astype(bool))
    out = paddle.where_(mask, x, value)
    return out


@inplace_apis_in_dygraph_only
def reshape_(x: Tensor, shape: ShapeLike, name: str | None = None) -> Tensor:
    """
    Inplace version of ``reshape`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_paddle_tensor_reshape`.
    """
    if in_dynamic_mode():
        tmp_tensor_type = core.eager.Tensor
        if isinstance(shape, (list, tuple)):
            shape = [
                item.item(0) if isinstance(item, tmp_tensor_type) else item
                for item in shape
            ]
            if shape == x.shape:
                out = x
            else:
                out = _C_ops.reshape_(x, shape)
        elif isinstance(shape, tmp_tensor_type):
            shape.stop_gradient = True
            out = _C_ops.reshape_(x, shape)
        else:
            raise ValueError(
                "shape must be an instance of `list`, `tuple` or `Variable`,"
                f" got '{type(shape)}.'"
            )

        return out


@overload
def atleast_1d(inputs: Tensor, name: str | None = ...) -> Tensor: ...


@overload
def atleast_1d(*inputs: Tensor, name: str | None = ...) -> list[Tensor]: ...


def atleast_1d(*inputs, name=None):
    """
    Convert inputs to tensors and return the view with at least 1-dimension. Scalar inputs are converted,
    one or high-dimensional inputs are preserved.

    Args:
        inputs (list[Tensor]): One or more tensors. The data type is ``float16``, ``float32``, ``float64``, ``int16``, ``int32``, ``int64``, ``int8``, ``uint8``, ``complex64``, ``complex128``, ``bfloat16`` or ``bool``.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        One Tensor, if there is only one input.
        List of Tensors, if there are more than one inputs.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> # one input
            >>> x = paddle.to_tensor(123, dtype='int32')
            >>> out = paddle.atleast_1d(x)
            >>> print(out)
            Tensor(shape=[1], dtype=int32, place=Place(cpu), stop_gradient=True,
            [123])

            >>> # more than one inputs
            >>> x = paddle.to_tensor(123, dtype='int32')
            >>> y = paddle.to_tensor([1.23], dtype='float32')
            >>> out = paddle.atleast_1d(x, y)
            >>> print(out)
            [Tensor(shape=[1], dtype=int32, place=Place(cpu), stop_gradient=True,
            [123]), Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True,
            [1.23000002])]

            >>> # more than 1-D input
            >>> x = paddle.to_tensor(123, dtype='int32')
            >>> y = paddle.to_tensor([[1.23]], dtype='float32')
            >>> out = paddle.atleast_1d(x, y)
            >>> print(out)
            [Tensor(shape=[1], dtype=int32, place=Place(cpu), stop_gradient=True,
            [123]), Tensor(shape=[1, 1], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[1.23000002]])]
    """
    out = []
    for input in inputs:
        if not isinstance(
            input,
            (
                paddle.Tensor,
                paddle.base.framework.Variable,
                paddle.base.libpaddle.pir.Value,
            ),
        ):
            tensor = paddle.to_tensor(input)
        else:
            tensor = input

        if tensor.dim() == 0:
            result = tensor.reshape((1,))
        else:
            result = tensor
        out.append(result)

    if len(out) == 1:
        return out[0]
    else:
        return out


@overload
def atleast_2d(inputs: Tensor, name: str | None = ...) -> Tensor: ...


@overload
def atleast_2d(*inputs: Tensor, name: str | None = ...) -> list[Tensor]: ...


def atleast_2d(*inputs, name=None):
    """
    Convert inputs to tensors and return the view with at least 2-dimension. Two or high-dimensional inputs are preserved.

    The following diagram illustrates the behavior of atleast_2d on different dimensional inputs for the following cases:

        1. A 0-dim tensor input.
        2. A 0-dim tensor and a 1-dim tensor input.
        3. A 0-dim tensor and a 3-dim tensor input.

    .. image:: https://githubraw.cdn.bcebos.com/PaddlePaddle/docs/develop/docs/images/api_legend/atleast_2d.png
        :width: 600
        :alt: legend of atleast_2d API
        :align: center

    In each case, the function returns the tensors (or a list of tensors) in views with at least 2 dimensions.

    Args:
        inputs (Tensor|list(Tensor)): One or more tensors. The data type is ``float16``, ``float32``, ``float64``, ``int16``, ``int32``, ``int64``, ``int8``, ``uint8``, ``complex64``, ``complex128``, ``bfloat16`` or ``bool``.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        One Tensor, if there is only one input.
        List of Tensors, if there are more than one inputs.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> # one input
            >>> x = paddle.to_tensor(123, dtype='int32')
            >>> out = paddle.atleast_2d(x)
            >>> print(out)
            Tensor(shape=[1, 1], dtype=int32, place=Place(cpu), stop_gradient=True,
            [[123]])

            >>> # more than one inputs
            >>> x = paddle.to_tensor(123, dtype='int32')
            >>> y = paddle.to_tensor([1.23], dtype='float32')
            >>> out = paddle.atleast_2d(x, y)
            >>> print(out)
            [Tensor(shape=[1, 1], dtype=int32, place=Place(cpu), stop_gradient=True,
            [[123]]), Tensor(shape=[1, 1], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[1.23000002]])]

            >>> # more than 2-D input
            >>> x = paddle.to_tensor(123, dtype='int32')
            >>> y = paddle.to_tensor([[[1.23]]], dtype='float32')
            >>> out = paddle.atleast_2d(x, y)
            >>> print(out)
            [Tensor(shape=[1, 1], dtype=int32, place=Place(cpu), stop_gradient=True,
            [[123]]), Tensor(shape=[1, 1, 1], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[[1.23000002]]])]
    """
    out = []
    for input in inputs:
        if not isinstance(
            input,
            (
                paddle.Tensor,
                paddle.base.framework.Variable,
                paddle.base.libpaddle.pir.Value,
            ),
        ):
            tensor = paddle.to_tensor(input)
        else:
            tensor = input

        if tensor.dim() == 0:
            result = tensor.reshape((1, 1))
        elif tensor.dim() == 1:
            result = paddle.unsqueeze(tensor, axis=0)
        else:
            result = tensor
        out.append(result)

    if len(out) == 1:
        return out[0]
    else:
        return out


@overload
def atleast_3d(inputs: Tensor, name: str | None = ...) -> Tensor: ...


@overload
def atleast_3d(*inputs: Tensor, name: str | None = ...) -> list[Tensor]: ...


def atleast_3d(*inputs, name=None):
    """
    Convert inputs to tensors and return the view with at least 3-dimension. Three or high-dimensional inputs are preserved.

    Args:
        inputs (Tensor|list(Tensor)): One or more tensors. The data type is ``float16``, ``float32``, ``float64``, ``int16``, ``int32``, ``int64``, ``int8``, ``uint8``, ``complex64``, ``complex128``, ``bfloat16`` or ``bool``.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        One Tensor, if there is only one input.
        List of Tensors, if there are more than one inputs.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> # one input
            >>> x = paddle.to_tensor(123, dtype='int32')
            >>> out = paddle.atleast_3d(x)
            >>> print(out)
            Tensor(shape=[1, 1, 1], dtype=int32, place=Place(cpu), stop_gradient=True,
            [[[123]]])

            >>> # more than one inputs
            >>> x = paddle.to_tensor(123, dtype='int32')
            >>> y = paddle.to_tensor([1.23], dtype='float32')
            >>> out = paddle.atleast_3d(x, y)
            >>> print(out)
            [Tensor(shape=[1, 1, 1], dtype=int32, place=Place(cpu), stop_gradient=True,
            [[[123]]]), Tensor(shape=[1, 1, 1], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[[1.23000002]]])]

            >>> # more than 3-D input
            >>> x = paddle.to_tensor(123, dtype='int32')
            >>> y = paddle.to_tensor([[[[1.23]]]], dtype='float32')
            >>> out = paddle.atleast_3d(x, y)
            >>> print(out)
            [Tensor(shape=[1, 1, 1], dtype=int32, place=Place(cpu), stop_gradient=True,
            [[[123]]]), Tensor(shape=[1, 1, 1, 1], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[[[1.23000002]]]])]
    """
    out = []
    for input in inputs:
        if not isinstance(
            input,
            (
                paddle.Tensor,
                paddle.base.framework.Variable,
                paddle.base.libpaddle.pir.Value,
            ),
        ):
            tensor = paddle.to_tensor(input)
        else:
            tensor = input

        if tensor.dim() == 0:
            result = tensor.reshape((1, 1, 1))
        elif tensor.dim() == 1:
            result = paddle.unsqueeze(tensor, axis=[0, 2])
        elif tensor.dim() == 2:
            result = paddle.unsqueeze(tensor, axis=2)
        else:
            result = tensor
        out.append(result)

    if len(out) == 1:
        return out[0]
    else:
        return out


def gather_nd(x: Tensor, index: Tensor, name: str | None = None) -> Tensor:
    """

    This function is actually a high-dimensional extension of :code:`gather`
    and supports for simultaneous indexing by multiple axes. :attr:`index` is a
    K-dimensional integer tensor, which is regarded as a (K-1)-dimensional
    tensor of :attr:`index` into :attr:`input`, where each element defines
    a slice of params:

    .. math::

        output[(i_0, ..., i_{K-2})] = input[index[(i_0, ..., i_{K-2})]]

    Obviously, :code:`index.shape[-1] <= input.rank` . And, the output tensor has
    shape :code:`index.shape[:-1] + input.shape[index.shape[-1]:]` .

    .. code-block:: text

            Given:
                x =  [[[ 0,  1,  2,  3],
                       [ 4,  5,  6,  7],
                       [ 8,  9, 10, 11]],
                      [[12, 13, 14, 15],
                       [16, 17, 18, 19],
                       [20, 21, 22, 23]]]
                x.shape = (2, 3, 4)

            * Case 1:
                index = [[1]]

                gather_nd(x, index)
                         = [x[1, :, :]]
                         = [[12, 13, 14, 15],
                            [16, 17, 18, 19],
                            [20, 21, 22, 23]]

            * Case 2:
                index = [[0,2]]

                gather_nd(x, index)
                         = [x[0, 2, :]]
                         = [8, 9, 10, 11]

            * Case 3:
                index = [[1, 2, 3]]

                gather_nd(x, index)
                         = [x[1, 2, 3]]
                         = [23]

    Args:
        x (Tensor): The input Tensor which it's data type should be bool, float16, float32, float64, int32, int64.
        index (Tensor): The index input with rank > 1, index.shape[-1] <= input.rank.
                        Its dtype should be int32, int64.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        output (Tensor), A tensor with the shape index.shape[:-1] + input.shape[index.shape[-1]:]

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([[[1, 2], [3, 4], [5, 6]],
            ...                       [[7, 8], [9, 10], [11, 12]]])
            >>> index = paddle.to_tensor([[0, 1]])

            >>> output = paddle.gather_nd(x, index)
            >>> print(output)
            Tensor(shape=[1, 2], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[3, 4]])

    """
    if in_dynamic_or_pir_mode():
        check_dtype(index.dtype, "index", ['int32', 'int64'], 'gather_nd')
        return _C_ops.gather_nd(x, index)
    else:
        check_variable_and_dtype(
            x,
            'x',
            [
                'bool',
                'float16',
                'uint16',
                'float32',
                'float64',
                'int16',
                'int32',
                'int64',
            ],
            'gather_nd',
        )
        check_variable_and_dtype(
            index, 'index', ['int32', 'int64'], 'gather_nd'
        )
        helper = LayerHelper('gather_nd', **locals())
        dtype = helper.input_dtype()
        output = helper.create_variable_for_type_inference(dtype)
        helper.append_op(
            type="gather_nd",
            inputs={"X": x, "Index": index},
            outputs={"Out": output},
        )
        return output


def strided_slice(
    x: Tensor,
    axes: Sequence[int | Tensor],
    starts: Sequence[int | Tensor] | Tensor,
    ends: Sequence[int | Tensor] | Tensor,
    strides: Sequence[int | Tensor] | Tensor,
    name: str | None = None,
) -> Tensor:
    """
    This operator produces a slice of ``x`` along multiple axes. Similar to numpy:
    https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
    Slice uses ``axes``, ``starts`` and ``ends`` attributes to specify the start and
    end dimension for each axis in the list of axes and Slice uses this information
    to slice the input data tensor. If a negative value is passed to
    ``starts`` or ``ends`` such as :math:`-i`,  it represents the reverse position of the
    axis :math:`i-1` th(here 0 is the initial position). The ``strides`` represents steps of
    slicing and if the ``strides`` is negative, slice operation is in the opposite direction.
    If the value passed to ``starts`` or ``ends`` is greater than n
    (the number of elements in this dimension), it represents n.
    For slicing to the end of a dimension with unknown size, it is recommended
    to pass in INT_MAX. The size of ``axes`` must be equal to ``starts`` , ``ends`` and ``strides``.
    Following examples will explain how strided_slice works:

    .. code-block:: text

        Case1:
            Given:
                data = [ [1, 2, 3, 4], [5, 6, 7, 8], ]
                axes = [0, 1]
                starts = [1, 0]
                ends = [2, 3]
                strides = [1, 1]
            Then:
                result = [ [5, 6, 7], ]

        Case2:
            Given:
                data = [ [1, 2, 3, 4], [5, 6, 7, 8], ]
                axes = [0, 1]
                starts = [0, 1]
                ends = [2, 0]
                strides = [1, -1]
            Then:
                result = [ [8, 7, 6], ]
        Case3:
            Given:
                data = [ [1, 2, 3, 4], [5, 6, 7, 8], ]
                axes = [0, 1]
                starts = [0, 1]
                ends = [-1, 1000]
                strides = [1, 3]
            Then:
                result = [ [2], ]

    Args:
        x (Tensor): An N-D ``Tensor``. The data type is ``bool``, ``float16``, ``float32``, ``float64``, ``int32`` or ``int64``.
        axes (list|tuple): The data type is ``int32`` . Axes that `starts` and `ends` apply to.
                            It's optional. If it is not provides, it will be treated as :math:`[0,1,...,len(starts)-1]`.
        starts (list|tuple|Tensor): The data type is ``int32`` . If ``starts`` is a list or tuple, the elements of it should be
            integers or Tensors with shape []. If ``starts`` is an Tensor, it should be an 1-D Tensor.
            It represents starting indices of corresponding axis in ``axes``.
        ends (list|tuple|Tensor): The data type is ``int32`` . If ``ends`` is a list or tuple, the elements of it should be
            integers or Tensors with shape []. If ``ends`` is an Tensor, it should be an 1-D Tensor.
            It represents ending indices of corresponding axis in ``axes``.
        strides (list|tuple|Tensor): The data type is ``int32`` . If ``strides`` is a list or tuple, the elements of it should be
            integers or Tensors with shape []. If ``strides`` is an Tensor, it should be an 1-D Tensor.
            It represents slice step of corresponding axis in ``axes``.
        name(str|None, optional): The default value is None.  Normally there is no need for user to set this property.
                        For more information, please refer to :ref:`api_guide_Name` .

    Returns:
        Tensor, A ``Tensor`` with the same dimension as ``x``. The data type is same as ``x``.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> x = paddle.zeros(shape=[3,4,5,6], dtype="float32")
            >>> # example 1:
            >>> # attr starts is a list which doesn't contain Tensor.
            >>> axes = [1, 2, 3]
            >>> starts = [-3, 0, 2]
            >>> ends = [3, 2, 4]
            >>> strides_1 = [1, 1, 1]
            >>> strides_2 = [1, 1, 2]
            >>> sliced_1 = paddle.strided_slice(x, axes=axes, starts=starts, ends=ends, strides=strides_1)
            >>> # sliced_1 is x[:, 1:3:1, 0:2:1, 2:4:1].
            >>> # example 2:
            >>> # attr starts is a list which contain tensor Tensor.
            >>> minus_3 = paddle.full(shape=[1], fill_value=-3, dtype='int32')
            >>> sliced_2 = paddle.strided_slice(x, axes=axes, starts=[minus_3, 0, 2], ends=ends, strides=strides_2)
            >>> # sliced_2 is x[:, 1:3:1, 0:2:1, 2:4:2].
    """
    if in_dynamic_or_pir_mode():
        return _C_ops.strided_slice(x, axes, starts, ends, strides)
    else:
        helper = LayerHelper('strided_slice', **locals())

        check_variable_and_dtype(
            x,
            'x',
            [
                'bool',
                'float16',
                'uint16',
                'float32',
                'float64',
                'int32',
                'int64',
            ],
            'strided_slice',
        )
        check_type(axes, 'axes', (list, tuple), 'strided_slice')
        check_type(starts, 'starts', (list, tuple, Variable), 'strided_slice')
        check_type(ends, 'ends', (list, tuple, Variable), 'strided_slice')
        check_type(strides, 'strides', (list, tuple, Variable), 'strided_slice')

        def check_list_elements_dtype(list_input, input_name):
            if isinstance(list_input, Variable):
                check_dtype(
                    list_input.dtype,
                    input_name,
                    ['int32', 'int64'],
                    'strided_slice',
                )
            else:
                for i, var in enumerate(list_input):
                    var_name = input_name + '[' + str(i) + ']'
                    if isinstance(var, Variable):
                        check_dtype(
                            var.dtype, var_name, ['int32'], 'strided_slice'
                        )

        check_list_elements_dtype(axes, 'axes')
        check_list_elements_dtype(starts, 'starts')
        check_list_elements_dtype(ends, 'ends')
        check_list_elements_dtype(strides, 'strides')

        def get_new_list_tensor(old_list):
            new_list_tensor = []
            for dim in old_list:
                if isinstance(dim, Variable):
                    dim.stop_gradient = True
                    new_list_tensor.append(dim)
                else:
                    assert isinstance(dim, int)
                    temp_out = helper.create_variable_for_type_inference(
                        'int32'
                    )
                    fill_constant(
                        [1], 'int32', dim, force_cpu=True, out=temp_out
                    )
                    new_list_tensor.append(temp_out)
            return new_list_tensor

        inputs = {'Input': x}
        attrs = {'axes': axes}
        infer_flags = [1 for i in range(len(axes))]
        # starts
        if isinstance(starts, Variable):
            starts.stop_gradient = True
            inputs['StartsTensor'] = starts
        elif isinstance(starts, (list, tuple)):
            attrs['starts'] = []
            if paddle.utils._contain_var(starts):
                inputs['StartsTensorList'] = get_new_list_tensor(starts)
                for i, dim in enumerate(starts):
                    if isinstance(dim, Variable):
                        attrs['starts'].append(-1)
                        infer_flags[i] = -1
                    else:
                        attrs['starts'].append(dim)
            else:
                attrs['starts'] = starts

        # ends
        if isinstance(ends, Variable):
            ends.stop_gradient = True
            inputs['EndsTensor'] = ends
        elif isinstance(ends, (list, tuple)):
            attrs['ends'] = []
            if paddle.utils._contain_var(ends):
                inputs['EndsTensorList'] = get_new_list_tensor(ends)
                for i, dim in enumerate(ends):
                    if isinstance(dim, Variable):
                        attrs['ends'].append(-1)
                        infer_flags[i] = -1
                    else:
                        attrs['ends'].append(dim)
            else:
                attrs['ends'] = ends

        # strides
        if isinstance(strides, Variable):
            strides.stop_gradient = True
            inputs['StridesTensor'] = strides
        elif isinstance(strides, (list, tuple)):
            attrs['strides'] = []
            if paddle.utils._contain_var(strides):
                inputs['StridesTensorList'] = get_new_list_tensor(strides)
                for i, dim in enumerate(strides):
                    if isinstance(dim, Variable):
                        attrs['strides'].append(-1)
                        infer_flags[i] = -1
                    else:
                        attrs['strides'].append(dim)
            else:
                attrs['strides'] = strides
        attrs['infer_flags'] = infer_flags
        out = helper.create_variable_for_type_inference(
            dtype=helper.input_dtype('x')
        )
        helper.append_op(
            type='strided_slice',
            inputs=inputs,
            attrs=attrs,
            outputs={'Out': out},
        )

        return out


def tensordot(
    x: Tensor,
    y: Tensor,
    axes: int | NestedSequence[int] | Tensor = 2,
    name: str | None = None,
) -> Tensor:
    r"""
    This function computes a contraction, which sum the product of elements from two tensors along the given axes.

    Args:
        x (Tensor): The left tensor for contraction with data type ``float16`` or ``float32`` or ``float64``.
        y (Tensor): The right tensor for contraction with the same data type as ``x``.
        axes (int|tuple|list|Tensor, optional):  The axes to contract for ``x`` and ``y``, defaulted to integer ``2``.

            1. It could be a non-negative integer ``n``,
               in which the function will sum over the last ``n`` axes of ``x`` and the first ``n`` axes of ``y`` in order.

            2. It could be a 1-d tuple or list with data type ``int``, in which ``x`` and ``y`` will be contracted along the same given axes.
               For example, ``axes`` =[0, 1] applies contraction along the first two axes for ``x`` and the first two axes for ``y``.

            3. It could be a tuple or list containing one or two 1-d tuple|list|Tensor with data type ``int``.
               When containing one tuple|list|Tensor, the data in tuple|list|Tensor specified the same axes for ``x`` and ``y`` to contract.
               When containing two tuple|list|Tensor, the first will be applied to ``x`` and the second to ``y``.
               When containing more than two tuple|list|Tensor, only the first two axis sequences will be used while the others will be ignored.

            4. It could be a tensor, in which the ``axes`` tensor will be translated to a python list
               and applied the same rules described above to determine the contraction axes.
               Note that the ``axes`` with Tensor type is ONLY available in Dygraph mode.
        name(str|None, optional): The default value is None.  Normally there is no need for user to set this property.
                             For more information, please refer to :ref:`api_guide_Name` .

    Return:
        Output (Tensor), The contraction result with the same data type as ``x`` and ``y``.
        In general, :math:`output.ndim = x.ndim + y.ndim - 2 \times n_{axes}`, where :math:`n_{axes}` denotes the number of axes to be contracted.

    NOTES:
        1. This function supports tensor broadcast,
           the size in the corresponding dimensions of ``x`` and ``y`` should be equal, or applies to the broadcast rules.
        2. This function also supports axes expansion,
           when the two given axis sequences for ``x`` and ``y`` are of different lengths,
           the shorter sequence will expand the same axes as the longer one at the end.
           For example, if ``axes`` =[[0, 1, 2, 3], [1, 0]],
           the axis sequence for ``x`` is [0, 1, 2, 3],
           while the corresponding axis sequences for ``y`` will be expanded from [1, 0] to [1, 0, 2, 3].

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> from typing import Literal

            >>> data_type: Literal["float64"] = 'float64'

            >>> # For two 2-d tensor x and y, the case axes=0 is equivalent to outer product.
            >>> # Note that tensordot supports empty axis sequence, so all the axes=0, axes=[], axes=[[]], and axes=[[],[]] are equivalent cases.
            >>> x = paddle.arange(4, dtype=data_type).reshape([2, 2])
            >>> y = paddle.arange(4, dtype=data_type).reshape([2, 2])
            >>> z = paddle.tensordot(x, y, axes=0)
            >>> print(z)
            Tensor(shape=[2, 2, 2, 2], dtype=float64, place=Place(cpu), stop_gradient=True,
             [[[[0., 0.],
                [0., 0.]],
               [[0., 1.],
                [2., 3.]]],
              [[[0., 2.],
                [4., 6.]],
               [[0., 3.],
                [6., 9.]]]])

            >>> # For two 1-d tensor x and y, the case axes=1 is equivalent to inner product.
            >>> x = paddle.arange(10, dtype=data_type)
            >>> y = paddle.arange(10, dtype=data_type)
            >>> z1 = paddle.tensordot(x, y, axes=1)
            >>> z2 = paddle.dot(x, y)
            >>> print(z1)
            Tensor(shape=[], dtype=float64, place=Place(cpu), stop_gradient=True,
            285.)
            >>> print(z2)
            Tensor(shape=[], dtype=float64, place=Place(cpu), stop_gradient=True,
            285.)


            >>> # For two 2-d tensor x and y, the case axes=1 is equivalent to matrix multiplication.
            >>> x = paddle.arange(6, dtype=data_type).reshape([2, 3])
            >>> y = paddle.arange(12, dtype=data_type).reshape([3, 4])
            >>> z1 = paddle.tensordot(x, y, axes=1)
            >>> z2 = paddle.matmul(x, y)
            >>> print(z1)
            Tensor(shape=[2, 4], dtype=float64, place=Place(cpu), stop_gradient=True,
            [[20., 23., 26., 29.],
             [56., 68., 80., 92.]])
            >>> print(z2)
            Tensor(shape=[2, 4], dtype=float64, place=Place(cpu), stop_gradient=True,
            [[20., 23., 26., 29.],
             [56., 68., 80., 92.]])

            >>> # When axes is a 1-d int list, x and y will be contracted along the same given axes.
            >>> # Note that axes=[1, 2] is equivalent to axes=[[1, 2]], axes=[[1, 2], []], axes=[[1, 2], [1]], and axes=[[1, 2], [1, 2]].
            >>> x = paddle.arange(24, dtype=data_type).reshape([2, 3, 4])
            >>> y = paddle.arange(36, dtype=data_type).reshape([3, 3, 4])
            >>> z = paddle.tensordot(x, y, axes=[1, 2])
            >>> print(z)
            Tensor(shape=[2, 3], dtype=float64, place=Place(cpu), stop_gradient=True,
            [[506. , 1298., 2090.],
             [1298., 3818., 6338.]])

            >>> # When axes is a list containing two 1-d int list, the first will be applied to x and the second to y.
            >>> x = paddle.arange(60, dtype=data_type).reshape([3, 4, 5])
            >>> y = paddle.arange(24, dtype=data_type).reshape([4, 3, 2])
            >>> z = paddle.tensordot(x, y, axes=([1, 0], [0, 1]))
            >>> print(z)
            Tensor(shape=[5, 2], dtype=float64, place=Place(cpu), stop_gradient=True,
            [[4400., 4730.],
             [4532., 4874.],
             [4664., 5018.],
             [4796., 5162.],
             [4928., 5306.]])

            >>> # Thanks to the support of axes expansion, axes=[[0, 1, 3, 4], [1, 0, 3, 4]] can be abbreviated as axes= [[0, 1, 3, 4], [1, 0]].
            >>> x = paddle.arange(720, dtype=data_type).reshape([2, 3, 4, 5, 6])
            >>> y = paddle.arange(720, dtype=data_type).reshape([3, 2, 4, 5, 6])
            >>> z = paddle.tensordot(x, y, axes=[[0, 1, 3, 4], [1, 0]])
            >>> print(z)
            Tensor(shape=[4, 4], dtype=float64, place=Place(cpu), stop_gradient=True,
            [[23217330., 24915630., 26613930., 28312230.],
             [24915630., 26775930., 28636230., 30496530.],
             [26613930., 28636230., 30658530., 32680830.],
             [28312230., 30496530., 32680830., 34865130.]])
    """
    op_type = 'tensordot'
    input_dtype = ['float16', 'float32', 'float64']

    check_variable_and_dtype(x, 'x', input_dtype, op_type)
    check_variable_and_dtype(y, 'y', input_dtype, op_type)
    check_type(
        axes, 'axes', (int, tuple, list, Variable, paddle.pir.Value), op_type
    )

    def _var_to_list(var):
        if in_dynamic_mode():
            return tolist(var)
        raise TypeError(
            "The 'axes' with type 'Tensor' in "
            + op_type
            + " is not available in static graph mode, "
            "please convert its type to int|Tuple|List, or use dynamic graph mode."
        )

    axes_x = []
    axes_y = []
    if np.issubdtype(type(axes), np.integer):
        assert axes >= 0, (
            "The 'axes' in "
            + op_type
            + f" should not be negative, but received axes={axes}."
        )
        axes_x = range(x.ndim - axes, x.ndim)
        axes_y = range(axes)
    else:
        if isinstance(axes, Variable):
            axes = _var_to_list(axes)

        if not axes or np.issubdtype(type(axes[0]), np.integer):
            axes_x = axes
        else:
            axes_x = axes[0]
            if len(axes) > 1:
                axes_y = axes[1]

            if isinstance(axes_x, Variable):
                axes_x = _var_to_list(axes_x)
            if isinstance(axes_y, Variable):
                axes_y = _var_to_list(axes_y)

    axes_x, axes_y = list(axes_x), list(axes_y)
    len_axes_x, len_axes_y = len(axes_x), len(axes_y)
    if len_axes_x < len_axes_y:
        axes_x.extend(axes_y[len_axes_x:])
    elif len_axes_y < len_axes_x:
        axes_y.extend(axes_x[len_axes_y:])

    shape_x, shape_y = list(x.shape), list(y.shape)
    need_contracted_dim_x = np.zeros((x.ndim), dtype=bool)
    need_contracted_dim_y = np.zeros((y.ndim), dtype=bool)
    contraction_size = 1
    for i in range(len(axes_x)):
        dim_x, dim_y = axes_x[i], axes_y[i]
        sx, sy = shape_x[dim_x], shape_y[dim_y]
        if sx == 1:
            shape_y[dim_y] = 1
            y = y.sum(dim_y).reshape(shape_y)
        elif sy == 1:
            shape_x[dim_x] = 1
            x = x.sum(dim_x).reshape(shape_x)
        else:
            assert sx == sy, (
                "The dimensional size for 'x' and 'y' in "
                + op_type
                + f" should match each other, but 'x' has size {sx} in dim {dim_x} while 'y' has size {sy} in dim {dim_y}."
            )

        need_contracted_dim_x[dim_x] = True
        need_contracted_dim_y[dim_y] = True
        contraction_size *= shape_x[dim_x]

    perm_x = []
    perm_y = []
    shape_out = []
    not_contraction_size_x = 1
    not_contraction_size_y = 1
    for i in range(x.ndim):
        if not need_contracted_dim_x[i]:
            perm_x.append(i)
            shape_out.append(shape_x[i])
            not_contraction_size_x *= shape_x[i]
    perm_x.extend(axes_x)
    perm_y.extend(axes_y)
    for i in range(y.ndim):
        if not need_contracted_dim_y[i]:
            perm_y.append(i)
            shape_out.append(shape_y[i])
            not_contraction_size_y *= shape_y[i]

    x = x.transpose(perm=perm_x).reshape(
        [not_contraction_size_x, contraction_size]
    )
    y = y.transpose(perm=perm_y).reshape(
        [contraction_size, not_contraction_size_y]
    )
    out = x.matmul(y).reshape(shape_out)
    return out


def as_complex(x: Tensor, name: str | None = None) -> Tensor:
    """Transform a real tensor to a complex tensor.

    The data type of the input tensor is 'float32' or 'float64', and the data
    type of the returned tensor is 'complex64' or 'complex128', respectively.

    The shape of the input tensor is ``(* ,2)``, (``*`` means arbitrary shape), i.e.
    the size of the last axis should be 2, which represent the real and imag part
    of a complex number. The shape of the returned tensor is ``(*,)``.

    The image below demonstrates the case that a real 3D-tensor with shape [2, 3, 2] is transformed into a complex 2D-tensor with shape [2, 3].

    .. image:: https://githubraw.cdn.bcebos.com/PaddlePaddle/docs/develop/docs/images/api_legend/as_complex.png
       :width: 500
       :alt: Illustration of as_complex
       :align: center

    Args:
        x (Tensor): The input tensor. Data type is 'float32' or 'float64'.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, The output. Data type is 'complex64' or 'complex128', with the same precision as the input.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> x = paddle.arange(12, dtype=paddle.float32).reshape([2, 3, 2])
            >>> y = paddle.as_complex(x)
            >>> print(y)
            Tensor(shape=[2, 3], dtype=complex64, place=Place(cpu), stop_gradient=True,
            [[1j      , (2+3j)  , (4+5j)  ],
             [(6+7j)  , (8+9j)  , (10+11j)]])
    """
    if in_dynamic_or_pir_mode():
        return _C_ops.as_complex(x)
    else:
        check_variable_and_dtype(x, 'x', ['float32', 'float64'], 'as_complex')
        op_type = "as_complex"
        helper = LayerHelper(op_type, **locals())
        inputs = {"X": x}
        out = helper.create_variable_for_type_inference(
            dtype=_real_to_complex_dtype(x.dtype)
        )
        outputs = {"Out": out}
        attrs = {}
        helper.append_op(
            type=op_type, inputs=inputs, attrs=attrs, outputs=outputs
        )
        return out


def as_real(x: Tensor, name: str | None = None) -> Tensor:
    """Transform a complex tensor to a real tensor.

    The data type of the input tensor is 'complex64' or 'complex128', and the data
    type of the returned tensor is 'float32' or 'float64', respectively.

    When the shape of the input tensor is ``(*, )``, (``*`` means arbitrary shape),
    the shape of the output tensor is ``(*, 2)``, i.e. the shape of the output is
    the shape of the input appended by an extra ``2``.

    Args:
        x (Tensor): The input tensor. Data type is 'complex64' or 'complex128'.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, The output. Data type is 'float32' or 'float64', with the same precision as the input.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> x = paddle.arange(12, dtype=paddle.float32).reshape([2, 3, 2])
            >>> y = paddle.as_complex(x)
            >>> z = paddle.as_real(y)
            >>> print(z)
            Tensor(shape=[2, 3, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[[0. , 1. ],
             [2. , 3. ],
             [4. , 5. ]],
            [[6. , 7. ],
             [8. , 9. ],
             [10., 11.]]])
    """
    if in_dynamic_or_pir_mode():
        return _C_ops.as_real(x)
    else:
        check_variable_and_dtype(x, 'x', ['complex64', 'complex128'], 'as_real')
        op_type = "as_real"
        helper = LayerHelper(op_type, **locals())
        inputs = {"X": x}
        out = helper.create_variable_for_type_inference(
            dtype=_complex_to_real_dtype(x.dtype)
        )
        outputs = {"Out": out}
        helper.append_op(type=op_type, inputs=inputs, outputs=outputs)
        return out


def repeat_interleave(
    x: Tensor,
    repeats: int | Tensor,
    axis: int | None = None,
    name: str | None = None,
) -> Tensor:
    """

    Returns a new tensor which repeats the ``x`` tensor along dimension ``axis`` using
    the entries in ``repeats`` which is a int or a Tensor.

    The image illustrates a typical case of the repeat_interleave operation.
    Given a tensor ``[[1, 2, 3], [4, 5, 6]]``, with the repeat counts ``repeats = [3, 2, 1]`` and parameter ``axis = 1``, it means that the elements in the 1st column are repeated 3 times, the 2nd column is repeated 2 times, and the 3rd column is repeated 1 time.

    The final output is a 2D tensor: ``[[1, 1, 1, 2, 2, 3], [4, 4, 4, 5, 5, 6]]``.

    .. image:: https://githubraw.cdn.bcebos.com/PaddlePaddle/docs/develop/docs/images/api_legend/repeat_interleave.png
        :width: 500
        :alt: legend of repeat_interleave API
        :align: center


    Args:
        x (Tensor): The input Tensor to be operated. The data of ``x`` can be one of float32, float64, int32, int64.
        repeats (Tensor|int): The number of repetitions for each element. repeats is broadcasted to fit the shape of the given axis.
        axis (int|None, optional): The dimension in which we manipulate. Default: None, the output tensor is flatten.
        name(str|None, optional): The default value is None. Normally there is no
            need for user to set this property. For more information, please
            refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, A Tensor with same data type as ``x``.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([[1, 2, 3], [4, 5, 6]])
            >>> repeats = paddle.to_tensor([3, 2, 1], dtype='int32')

            >>> out = paddle.repeat_interleave(x, repeats, 1)
            >>> print(out)
            Tensor(shape=[2, 6], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[1, 1, 1, 2, 2, 3],
             [4, 4, 4, 5, 5, 6]])

            >>> out = paddle.repeat_interleave(x, 2, 0)
            >>> print(out)
            Tensor(shape=[4, 3], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[1, 2, 3],
             [1, 2, 3],
             [4, 5, 6],
             [4, 5, 6]])

            >>> out = paddle.repeat_interleave(x, 2, None)
            >>> print(out)
            Tensor(shape=[12], dtype=int64, place=Place(cpu), stop_gradient=True,
            [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6])
    """
    if isinstance(repeats, (Variable, paddle.pir.Value)) and not repeats.shape:
        repeats = paddle.reshape(repeats, [1])
    if axis is None:
        x = paddle.flatten(x)
        axis = 0
    if in_dynamic_or_pir_mode():
        if isinstance(repeats, (Variable, paddle.pir.Value)):
            return _C_ops.repeat_interleave_with_tensor_index(x, repeats, axis)
        return _C_ops.repeat_interleave(x, repeats, axis)

    helper = LayerHelper("repeat_interleave", **locals())
    check_variable_and_dtype(
        x,
        'x',
        ['float32', 'float64', 'int32', 'int64'],
        'paddle.tensor.manipulation.repeat_interleave',
    )

    out = helper.create_variable_for_type_inference(x.dtype)

    helper.append_op(
        type='repeat_interleave',
        inputs={
            'X': x,
            'RepeatsTensor': repeats if isinstance(repeats, Variable) else None,
        },
        outputs={'Out': out},
        attrs={
            'dim': axis,
            'Repeats': repeats if isinstance(repeats, int) else 0,
        },
    )
    return out


def moveaxis(
    x: Tensor,
    source: int | Sequence[int],
    destination: int | Sequence[int],
    name: str | None = None,
) -> Tensor:
    """
    Move the axis of tensor from ``source`` position to ``destination`` position.

    Other axis that have not been moved remain their original order.

    Args:
        x (Tensor): The input Tensor. It is a N-D Tensor of data types bool, int32, int64, float32, float64, complex64, complex128.
        source(int|tuple|list): ``source`` position of axis that will be moved. Each element must be unique and integer.
        destination(int|tuple|list): ``destination`` position of axis that has been moved. Each element must be unique and integer.
        name(str|None, optional): The default value is None.  Normally there is no need for user to set this
            property. For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, A new tensor whose axis have been moved.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.ones([3, 2, 4])
            >>> outshape = paddle.moveaxis(x, [0, 1], [1, 2]).shape
            >>> print(outshape)
            [4, 3, 2]

            >>> x = paddle.ones([2, 3])
            >>> outshape = paddle.moveaxis(x, 0, 1).shape # equivalent to paddle.t(x)
            >>> print(outshape)
            [3, 2]
    """
    src = [source] if isinstance(source, int) else source
    dst = [destination] if isinstance(destination, int) else destination
    if isinstance(source, tuple):
        src = list(source)
    if isinstance(destination, tuple):
        dst = list(destination)
    assert len(src) == len(
        dst
    ), "'source' must have the same number with 'destination'"

    if len(src) != len(set(src)):
        raise ValueError("Each element of 'source' must be unique!")
    if len(dst) != len(set(dst)):
        raise ValueError("Each element of 'destination' must be unique!")

    ndim = len(x.shape)

    # perm is the new order after move axis
    perm = list(range(ndim))
    src_dims = list(range(ndim))
    dst_dims = list(range(ndim))

    for i, axis in enumerate(zip(src, dst)):
        assert isinstance(
            axis[0], int
        ), "Each element of 'source' must be integer."
        if axis[0] < 0:
            assert (
                axis[0] >= -ndim
            ), f"'source' must be in the range of [-{ndim}, {ndim})"
            src[i] += ndim
        else:
            assert (
                axis[0] < ndim
            ), f"'source' must be in the range of [-{ndim}, {ndim})"

        assert isinstance(
            axis[1], int
        ), "Each element of 'source' must be integer."
        if axis[1] < 0:
            assert (
                axis[1] >= -ndim
            ), f"'source' must be in the range of [-{ndim}, {ndim})"
            dst[i] += ndim
        else:
            assert (
                axis[1] < ndim
            ), f"'source' must be in the range of [-{ndim}, {ndim})"
        perm[dst[i]] = src[i]
        src_dims.remove(src[i])
        dst_dims.remove(dst[i])

    for i in range(len(src_dims)):
        perm[dst_dims[i]] = src_dims[i]

    if in_dynamic_or_pir_mode():
        out = _C_ops.transpose(x, perm)
        return out
    else:
        check_variable_and_dtype(
            x,
            'x',
            [
                'bool',
                'float16',
                'float32',
                'float64',
                'int32',
                'int64',
                'complex64',
                'complex128',
            ],
            'moveaxis',
        )

        helper = LayerHelper('moveaxis', **locals())
        out = helper.create_variable_for_type_inference(x.dtype)
        x_shape = helper.create_variable_for_type_inference(x.dtype)
        helper.append_op(
            type='transpose2',
            inputs={'X': [x]},
            outputs={'Out': [out], 'XShape': [x_shape]},
            attrs={'axis': perm},
        )
        return out


def masked_fill(
    x, mask: Tensor, value: Numeric, name: str | None = None
) -> Tensor:
    """
    Fills elements of self tensor with value where mask is True. The shape of mask must be broadcastable with the shape of the underlying tensor.

    The following figure shows an example: consider a 3x3 matrix `x`,where all elements have a value of 1, and a matrix `mask` of the same size, and `value` is 3.

    .. image:: https://githubraw.cdn.bcebos.com/PaddlePaddle/docs/develop/docs/images/api_legend/masked_fill.png
       :width: 700
       :align: center

    Args:
        x (Tensor) : The Destination Tensor. Supported data types are float,
            double, int, int64_t,float16 and bfloat16.
        mask (Tensor): The boolean tensor indicate the position to be filled.
            The data type of mask must be bool.
        value (Scalar or 0-D Tensor): The value used to fill the target tensor.
            Supported data types are float, double, int, int64_t,float16 and bfloat16.
        name(str|None, optional): The default value is None. Normally there is no
            need for user to set this property. For more information, please
            refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, same dimension and dtype with x.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:GPU)
            >>> import paddle
            >>> x = paddle.ones((3, 3), dtype="float32")
            >>> mask = paddle.to_tensor([[True, True, False]])
            >>> print(mask)
            Tensor(shape=[1, 3], dtype=bool, place=Place(gpu:0), stop_gradient=True,
                   [[True , True , False]])
            >>> out = paddle.masked_fill(x, mask, 2)
            >>> print(out)
            Tensor(shape=[3, 3], dtype=float32, place=Place(gpu:0), stop_gradient=True,
                   [[2., 2., 1.],
                    [2., 2., 1.],
                    [2., 2., 1.]])
    """
    if np.isscalar(value):
        value = paddle.full([], value, x.dtype)

    mask = paddle.logical_not(mask)
    out = paddle.where(mask, x, value)
    return out


@inplace_apis_in_dygraph_only
def masked_fill_(
    x, mask: Tensor, value: Numeric, name: str | None = None
) -> Tensor:
    """
    Inplace version of ``masked_fill`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_paddle_masked_fill`.
    """
    if np.isscalar(value):
        value = paddle.full([], value, x.dtype)

    mask = paddle.logical_not(mask)
    out = paddle.where_(mask, x, value)
    return out


@overload
def non_negative_axis(arr: Tensor, axis: int) -> int: ...


@overload
def non_negative_axis(arr: Tensor, axis: Tensor) -> Tensor: ...


def non_negative_axis(arr, axis):
    ndim = len(arr.shape)
    if axis >= 0:
        assert axis < ndim, f"'axis'  must be in the range of [-{ndim}, {ndim})"
    else:
        assert (
            axis >= -ndim
        ), f"'axis'  must be in the range of [-{ndim}, {ndim})"
        axis += ndim

    return axis


def infer_dynamic_broadcast_shape(
    arr_shape: Tensor, arr_shape_dim: int, indices_shape: Tensor, axis: int
) -> Tensor:
    """
    Find the broadcast shape for indices when `arr` has a dynamic shape.

    Args:
        arr_shape (Tensor): Shape tensor of arr.
        arr_shape_dim (int): Dimensions of arr.
        indices_shape (Tensor): Shape tensor of indices.
        axis (int): The axis to put 1d slices along.

    Returns:
        Tensor: The shape tensor for later broadcasting
    """
    new_shapes = [
        arr_shape[:axis],
        indices_shape[axis : axis + 1],
        arr_shape[axis + 1 :],
    ]
    return paddle.concat(new_shapes)


def infer_broadcast_shape(
    arr: Tensor, indices: Tensor, axis: int | Tensor
) -> tuple[int] | None:
    # This function is used in take/put_along_axis
    broadcast_shape_list = list(arr.shape)
    broadcast_shape_list[axis] = list(indices.shape)[axis]
    broadcast_shape = tuple(broadcast_shape_list)
    for i in range(len(arr.shape)):
        if arr.shape[i] < indices.shape[i]:
            # if indices matrix has larger size than arr matrix, do not broadcast.
            return None
    return broadcast_shape


def take_along_axis(
    arr: Tensor, indices: Tensor, axis: int, broadcast: bool = True
) -> Tensor:
    """
    Take values from the input array by given indices matrix along the designated axis.

    Args:
        arr (Tensor) : The input Tensor. Supported data types are bfloat16, float16, float32, float64,
            int32, int64, uint8.
        indices (Tensor) : Indices to take along each 1d slice of arr. This must match the dimension of arr,
            and need to broadcast against arr. Supported data type are int32 and int64.
        axis (int) : The axis to take 1d slices along.
        broadcast (bool, optional): whether the indices broadcast.

    Returns:
        Tensor, The indexed element, same dtype with arr

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([[1, 2, 3], [4, 5, 6], [7,8,9]])
            >>> index = paddle.to_tensor([[0]])
            >>> axis = 0
            >>> result = paddle.take_along_axis(x, index, axis)
            >>> print(result)
            Tensor(shape=[1, 3], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[1, 2, 3]])
    """
    if len(arr.shape) != len(indices.shape):
        raise ValueError(
            "`indices` and `arr` must have the same number of dimensions!"
        )
    axis = non_negative_axis(arr, axis)
    if broadcast:
        broadcast_shape = infer_broadcast_shape(arr, indices, axis)
        if not broadcast_shape:
            # if indices matrix have larger size than arr, arr should broadcast into indices shape.
            broadcast_shape = indices.shape
        indices = paddle.broadcast_to(indices, broadcast_shape)
        broadcast_shape_list = list(broadcast_shape)
        broadcast_shape_list[axis] = list(arr.shape)[axis]
        broadcast_shape = tuple(broadcast_shape_list)
        arr = paddle.broadcast_to(arr, broadcast_shape)
    else:
        for i in range(len(arr.shape)):
            if i != axis and arr.shape[i] < indices.shape[i]:
                raise RuntimeError(
                    f"Size does not match at dimension {i} expected index {indices.shape} to be smaller than self {arr.shape} apart from dimension {axis}"
                )

    if in_dynamic_or_pir_mode():
        return _C_ops.take_along_axis(arr, indices, axis)
    else:
        check_variable_and_dtype(
            arr,
            'x',
            [
                'float16',
                'float32',
                'float64',
                'int32',
                'int64',
                'uint8',
                'uint16',
            ],
            'take_along_axis',
        )
        check_variable_and_dtype(
            indices, 'index', ['int32', 'int64'], 'take_along_axis'
        )
        helper = LayerHelper('take_along_axis', **locals())
        dtype = helper.input_dtype()
        result = helper.create_variable_for_type_inference(dtype)
        helper.append_op(
            type="take_along_axis",
            inputs={"Input": arr, "Index": indices},
            attrs={"Axis": axis},
            outputs={"Result": result},
        )
        return result


def put_along_axis(
    arr: Tensor,
    indices: Tensor,
    values: float | Tensor,
    axis: int,
    reduce: Literal[
        'assign', 'add', 'mul', 'multiply', 'mean', 'amin', 'amax'
    ] = "assign",
    include_self: bool = True,
    broadcast: bool = True,
) -> Tensor:
    """
    Put values into the destination array by given indices matrix along the designated axis.

    Args:
        arr (Tensor) : The Destination Tensor. Supported data types are bfloat16, float16, float32, float64,
            int32, int64, uint8.
        indices (Tensor) : Indices to put along each 1d slice of arr. This must match the dimension of arr,
            and need to broadcast against arr if broadcast is 'True'. Supported data type are int32 and int64.
        values (scalar|Tensor) : The value element(s) to put. The data types should be same as arr.
        axis (int) : The axis to put 1d slices along.
        reduce (str, optional): The reduce operation, default is 'assign', support 'add', 'assign', 'mul', 'multiply', 'mean', 'amin' and 'amax'.
        include_self (bool, optional): whether to reduce with the elements of arr, default is 'True'.
        broadcast (bool, optional): whether to broadcast indices, default is 'True'.

    Returns:
        Tensor, The indexed element, same dtype with arr

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([[10, 30, 20], [60, 40, 50]])
            >>> index = paddle.to_tensor([[0]])
            >>> value = 99
            >>> axis = 0
            >>> result = paddle.put_along_axis(x, index, value, axis)
            >>> print(result)
            Tensor(shape=[2, 3], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[99, 99, 99],
             [60, 40, 50]])

            >>> index = paddle.zeros((2,2)).astype("int32")
            >>> value=paddle.to_tensor([[1,2],[3,4]]).astype(x.dtype)
            >>> result = paddle.put_along_axis(x, index, value, 0, "add", True, False)
            >>> print(result)
            Tensor(shape=[2, 3], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[14, 36, 20],
             [60, 40, 50]])

            >>> result = paddle.put_along_axis(x, index, value, 0, "mul", True, False)
            >>> print(result)
            Tensor(shape=[2, 3], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[30 , 240, 20 ],
             [60 , 40 , 50 ]])

            >>> result = paddle.put_along_axis(x, index, value, 0, "mean", True, False)
            >>> print(result)
            Tensor(shape=[2, 3], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[4 , 12, 20],
             [60, 40, 50]])

            >>> result = paddle.put_along_axis(x, index, value, 0, "amin", True, False)
            >>> print(result)
            Tensor(shape=[2, 3], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[1 , 2 , 20],
             [60, 40, 50]])

            >>> result = paddle.put_along_axis(x, index, value, 0, "amax", True, False)
            >>> print(result)
            Tensor(shape=[2, 3], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[10, 30, 20],
             [60, 40, 50]])

            >>> result = paddle.put_along_axis(x, index, value, 0, "add", False, False)
            >>> print(result)
            Tensor(shape=[2, 3], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[4 , 6 , 20],
             [60, 40, 50]])

    """

    def has_dynamic_shape(shapes):
        return any(shape < 0 for shape in shapes)

    if len(arr.shape) != len(indices.shape):
        raise ValueError(
            "`indices` and `arr` must have the same number of dimensions!"
        )
    axis = non_negative_axis(arr, axis)
    if broadcast:
        if has_dynamic_shape(arr.shape) or has_dynamic_shape(indices.shape):
            arr_shape = paddle.shape(arr)
            indices_shape = paddle.shape(indices)
            broadcast_shape = infer_dynamic_broadcast_shape(
                arr_shape, arr.ndim, indices_shape, axis
            )
            values = (
                paddle.to_tensor(values)
                if not isinstance(
                    values, (paddle.Tensor, paddle.pir.Value, Variable)
                )
                else values
            )
            indices = paddle.broadcast_to(indices, broadcast_shape)
            values = paddle.broadcast_to(values, broadcast_shape)
        else:
            broadcast_shape = infer_broadcast_shape(arr, indices, axis)
            values = (
                paddle.to_tensor(values)
                if not isinstance(
                    values, (paddle.Tensor, paddle.pir.Value, Variable)
                )
                else values
            )
            if broadcast_shape:
                indices = paddle.broadcast_to(indices, broadcast_shape)
                values = paddle.broadcast_to(values, broadcast_shape)
            else:
                values = paddle.broadcast_to(values, indices.shape)
    else:
        if isinstance(values, (paddle.Tensor, paddle.pir.Value)):
            if len(indices.shape) != len(values.shape):
                raise ValueError(
                    "`indices` and `values` must have the same number of dimensions!"
                )
            for i in range(len(arr.shape)):
                if (
                    i != axis and arr.shape[i] < indices.shape[i]
                ) or indices.shape[i] > values.shape[i]:
                    raise RuntimeError(
                        f"Size does not match at dimension {i} expected index {indices.shape} to be smaller than self {arr.shape} apart from dimension {axis} and to be smaller size than values {values.shape}"
                    )
        else:
            values = paddle.to_tensor(values).astype(arr.dtype)
            elements = 1
            for num in values.shape:
                elements *= num
            if elements == 1:  # paddle.pir.Value has no attribute 'size'
                values = paddle.broadcast_to(values, indices.shape)
        axis_max_size = arr.shape[axis]
        if in_dynamic_mode() and not (indices < axis_max_size).all():
            raise RuntimeError(
                f"one of element of indices is out of bounds for dimension {axis} with size {axis_max_size}"
            )
    if in_dynamic_or_pir_mode():
        if convert_dtype(indices.dtype) not in ['int32', 'int64']:
            raise TypeError(
                f"The data type of indices should be one of ['int32', 'int64'], but got {convert_dtype(indices.dtype)}"
            )
        return _C_ops.put_along_axis(
            arr, indices, values, axis, reduce, include_self
        )
    else:
        check_variable_and_dtype(
            arr,
            'x',
            [
                'float16',
                'float32',
                'float64',
                'int32',
                'int64',
                'uint8',
                'uint16',
            ],
            'put_along_axis',
        )
        check_variable_and_dtype(
            indices, 'index', ['int32', 'int64'], 'put_along_axis'
        )
        check_type(include_self, 'include_self', bool, 'put_along_axis')
        helper = LayerHelper('put_along_axis', **locals())
        dtype = helper.input_dtype()
        result = helper.create_variable_for_type_inference(dtype)
        helper.append_op(
            type="put_along_axis",
            inputs={"Input": arr, "Index": indices, "Value": values},
            attrs={
                "Axis": axis,
                "Reduce": reduce,
                "Include_self": include_self,
            },
            outputs={"Result": result},
        )
        return result


@inplace_apis_in_dygraph_only
def put_along_axis_(
    arr: Tensor,
    indices: Tensor,
    values: float | Tensor,
    axis: int,
    reduce: Literal[
        'assign', 'add', 'mul', 'multiply', 'mean', 'amin', 'amax'
    ] = "assign",
    include_self: bool = True,
    broadcast: bool = True,
):
    r"""
    Inplace version of ``put_along_axis`` API, the output Tensor will be inplaced with input ``arr``.
    Please refer to :ref:`api_paddle_put_along_axis`.
    """
    if len(arr.shape) != len(indices.shape):
        raise ValueError(
            "`indices` and `arr` must have the same number of dimensions!"
        )
    axis = non_negative_axis(arr, axis)
    if broadcast:
        broadcast_shape = infer_broadcast_shape(arr, indices, axis)
        values = (
            paddle.to_tensor(values)
            if not isinstance(values, paddle.Tensor)
            else values
        )
        if broadcast_shape:
            indices = paddle.broadcast_to(indices, broadcast_shape)
        values = paddle.broadcast_to(values, indices.shape)
    else:
        if isinstance(values, (paddle.Tensor, paddle.pir.Value)):
            if len(indices.shape) != len(values.shape):
                raise ValueError(
                    "`indices` and `values` must have the same number of dimensions!"
                )
            for i in range(len(arr.shape)):
                if (
                    i != axis and arr.shape[i] < indices.shape[i]
                ) or indices.shape[i] > values.shape[i]:
                    raise RuntimeError(
                        f"Size does not match at dimension {i} expected index {indices.shape} to be smaller than self {arr.shape} apart from dimension {axis} and to be smaller size than values {values.shape}"
                    )
        else:
            values = paddle.to_tensor(values).astype(arr.dtype)
            elements = 1
            for num in values.shape:
                elements *= num
            if elements == 1:  # paddle.pir.Value has no attribute 'size'
                values = paddle.broadcast_to(values, indices.shape)
        axis_max_size = arr.shape[axis]
        if not (indices < axis_max_size).all():
            raise RuntimeError(
                f"one of element of indices is out of bounds for dimension {axis} with size {axis_max_size}"
            )
    return _C_ops.put_along_axis_(
        arr, indices, values, axis, reduce, include_self
    )


def index_add(
    x: Tensor, index: Tensor, axis: int, value: Tensor, name: str | None = None
) -> Tensor:
    """
    Adds the elements of the input tensor with value tensor by selecting the indices in the order given in index.

    Args:
        x (Tensor) : The Destination Tensor. Supported data types are int32, int64, float16, float32, float64.
        index (Tensor): The 1-D Tensor containing the indices to index.
            The data type of ``index`` must be int32 or int64.
        axis (int): The dimension in which we index.
        value (Tensor): The tensor used to add the elements along the target axis.
        name(str|None, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        Tensor, same dimension and dtype with x.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:GPU)
            >>> import paddle
            >>> paddle.device.set_device('gpu')

            >>> input_tensor = paddle.to_tensor(paddle.ones((3, 3)), dtype="float32")
            >>> index = paddle.to_tensor([0, 2], dtype="int32")
            >>> value = paddle.to_tensor([[1, 1, 1], [1, 1, 1]], dtype="float32")
            >>> outplace_res = paddle.index_add(input_tensor, index, 0, value)
            >>> print(outplace_res)
            Tensor(shape=[3, 3], dtype=float32, place=Place(gpu:0), stop_gradient=True,
            [[2., 2., 2.],
             [1., 1., 1.],
             [2., 2., 2.]])
    """
    if in_dynamic_or_pir_mode():
        return _C_ops.index_add(x, index, value, axis)

    helper = LayerHelper("index_add", **locals())
    check_variable_and_dtype(
        x,
        'x',
        ['float16', 'float32', 'float64', 'int32', 'int64', 'uint16'],
        'paddle.tensor.manipulation.index_add',
    )
    check_variable_and_dtype(
        index,
        'index',
        ['int32', 'int64'],
        'paddle.tensor.manipulation.index_add',
    )
    check_variable_and_dtype(
        value,
        'add_value',
        ['float16', 'float32', 'float64', 'int32', 'int64', 'uint16'],
        'paddle.tensor.manipulation.index_add',
    )

    out = helper.create_variable_for_type_inference(x.dtype)

    helper.append_op(
        type='index_add',
        inputs={
            'X': x,
            'Index': index,
            'AddValue': value,
        },
        outputs={'Out': out},
        attrs={'axis': axis},
    )
    return out


@inplace_apis_in_dygraph_only
def index_add_(
    x: Tensor, index: Tensor, axis: int, value: Tensor, name: str | None = None
) -> Tensor:
    """
    Inplace version of ``index_add`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_paddle_index_add`.
    """
    return _C_ops.index_add_(x, index, value, axis)


@inplace_apis_in_dygraph_only
def index_put_(
    x: Tensor,
    indices: Sequence[Tensor],
    value: Tensor,
    accumulate: bool = False,
    name: str | None = None,
) -> Tensor:
    """
    Inplace version of ``index_put`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_paddle_index_put`.
    """
    return _C_ops.index_put_(x, indices, value, accumulate)


def index_put(
    x: Tensor,
    indices: Sequence[Tensor],
    value: Tensor,
    accumulate: bool = False,
    name: str | None = None,
) -> Tensor:
    """
    Puts values from the tensor values into the tensor x using the indices specified in indices (which is a tuple of Tensors).
    The expression paddle.index_put_(x, indices, values) is equivalent to tensor[indices] = values. Returns x.
    If accumulate is True, the elements in values are added to x. If accumulate is False, the behavior is undefined if indices contain duplicate elements.

    Args:
        x (Tensor) : The Source Tensor. Supported data types are int32, int64, float16, float32, float64, bool.
        indices (list[Tensor]|tuple[Tensor]): The tuple of Tensor containing the indices to index.
            The data type of ``tensor in indices`` must be int32, int64 or bool.
        value (Tensor): The tensor used to be assigned to x.
        accumulate (bool, optional): Whether the elements in values are added to x. Default: False.
        name(str|None, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        Tensor, same dimension and dtype with x.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.zeros([3, 3])
            >>> value = paddle.ones([3])
            >>> ix1 = paddle.to_tensor([0,1,2])
            >>> ix2 = paddle.to_tensor([1,2,1])
            >>> indices=(ix1,ix2)

            >>> out = paddle.index_put(x,indices,value)
            >>> print(x)
            Tensor(shape=[3, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[0., 0., 0.],
             [0., 0., 0.],
             [0., 0., 0.]])
            >>> print(out)
            Tensor(shape=[3, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[0., 1., 0.],
             [0., 0., 1.],
             [0., 1., 0.]])
    """
    if in_dynamic_or_pir_mode():
        return _C_ops.index_put(x, indices, value, accumulate)

    helper = LayerHelper("index_put", **locals())
    check_variable_and_dtype(
        x,
        'x',
        ['float16', 'float32', 'float64', 'int32', 'int64', 'bool'],
        'paddle.tensor.manipulation.index_put',
    )
    check_variable_and_dtype(
        value,
        'value',
        ['float16', 'float32', 'float64', 'int32', 'int64', 'bool'],
        'paddle.tensor.manipulation.index_put',
    )

    out = helper.create_variable_for_type_inference(x.dtype)

    helper.append_op(
        type='index_put',
        inputs={
            'x': x,
            'indices': indices,
            'value': value,
        },
        outputs={'out': out},
        attrs={'accumulate': accumulate},
    )
    return out


def unflatten(
    x: Tensor, axis: int, shape: ShapeLike, name: str | None = None
) -> Tensor:
    """
    Expand a certain dimension of the input x Tensor into a desired shape.

    The figure below shows the shape of a [2, 6] Tensor after applying ``unflatten(X, axis=1, shape=(2, 3))``, with data ranging from 0 to 11 in sequence.

    .. image:: https://githubraw.cdn.bcebos.com/PaddlePaddle/docs/develop/docs/images/api_legend/unflatten.png
       :width: 500
       :alt: Illustration of unflatten
       :align: center

    Args:
        x (Tensor) : An N-D Tensor. The data type is float16, float32, float64, int16, int32, int64, bool, uint16.
        axis (int): :attr:`axis` to be unflattened, specified as an index into `x.shape`.
        shape (list|tuple|Tensor): Unflatten :attr:`shape` on the specified :attr:`axis`. At most one dimension of the target :attr:`shape` can be -1.
            If the input :attr:`shape` does not contain -1 , the product of all elements in ``shape`` should be equal to ``x.shape[axis]``.
            The data type is `int` . If :attr:`shape` is a list or tuple, the elements of it should be integers or Tensors with shape [].
            If :attr:`shape` is an Tensor, it should be an 1-D Tensor.
        name(str|None, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        Tensor, return the unflatten tensor of :attr:`x`.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.randn(shape=[4, 6, 8])
            >>> shape = [2, 3]
            >>> axis = 1
            >>> res = paddle.unflatten(x, axis, shape)
            >>> print(res.shape)
            [4, 2, 3, 8]

            >>> x = paddle.randn(shape=[4, 6, 8])
            >>> shape = (-1, 2)
            >>> axis = -1
            >>> res = paddle.unflatten(x, axis, shape)
            >>> print(res.shape)
            [4, 6, 4, 2]

            >>> x = paddle.randn(shape=[4, 6, 8])
            >>> shape = paddle.to_tensor([2, 2])
            >>> axis = 0
            >>> res = paddle.unflatten(x, axis, shape)
            >>> print(res.shape)
            [2, 2, 6, 8]
    """

    # determine whether the input axis is valid.
    axis = non_negative_axis(x, axis)
    if isinstance(shape, (list, tuple)):
        new_shape = (
            list(x.shape[:axis]) + list(shape) + list(x.shape[axis + 1 :])
        )
    elif isinstance(shape, (Variable, paddle.pir.Value)):
        # The data type returned by `paddle.shape` is only 'int32'.
        new_shape = paddle.concat(
            [
                paddle.shape(x)[:axis],
                paddle.cast(shape, 'int64'),
                paddle.shape(x)[axis + 1 :],
            ]
        )
    else:
        raise TypeError(
            f"The data type of x should be one of ['List', 'Tuple', 'Tensor'], but got {type(shape)}"
        )
    x = x.reshape(new_shape)
    return x


@dygraph_only
def as_strided(
    x: Tensor,
    shape: Sequence[int],
    stride: Sequence[int],
    offset: int = 0,
    name: str | None = None,
) -> Tensor:
    """
    View x with specified shape, stride and offset.

    Note that the output Tensor will share data with origin Tensor and doesn't
    have a Tensor copy in ``dygraph`` mode.

    The following image illustrates an example: transforming an input Tensor with shape [2,4,6] into a Tensor with ``shape [8,6]`` and ``stride [6,1]``.


    .. image:: https://githubraw.cdn.bcebos.com/PaddlePaddle/docs/develop/docs/images/api_legend/as_strided.png
         :alt: Legend

    Args:
        x (Tensor): An N-D Tensor. The data type is ``float32``, ``float64``, ``int32``, ``int64`` or ``bool``
        shape (list|tuple): Define the target shape. Each element of it should be integer.
        stride (list|tuple): Define the target stride. Each element of it should be integer.
        offset (int, optional): Define the target Tensor's offset from x's holder. Default: 0.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, A as_strided Tensor with the same data type as ``x``.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.base.set_flags({"FLAGS_use_stride_kernel": True})

            >>> x = paddle.rand([2, 4, 6], dtype="float32")

            >>> out = paddle.as_strided(x, [8, 6], [6, 1])
            >>> print(out.shape)
            [8, 6]
            >>> # the stride is [6, 1].
    """
    return _C_ops.as_strided(x, shape, stride, offset)


@dygraph_only
def view(
    x: Tensor,
    shape_or_dtype: Sequence[int] | DTypeLike,
    name: str | None = None,
) -> Tensor:
    """
    View x with specified shape or dtype.

    Note that the output Tensor will share data with origin Tensor and doesn't
    have a Tensor copy in ``dygraph`` mode.

    Args:
        x (Tensor): An N-D Tensor. The data type is ``float32``, ``float64``, ``int32``, ``int64`` or ``bool``
        shape_or_dtype (list|tuple|np.dtype|str|VarType): Define the target shape or dtype. If list or tuple, shape_or_dtype represents shape, each element of it should be integer. If np.dtype or str or VarType, shape_or_dtype represents dtype, it can be bool, float16, float32, float64, int8, int32, int64, uint8.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, A viewed Tensor with the same data as ``x``.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.base.set_flags({"FLAGS_use_stride_kernel": True})

            >>> x = paddle.rand([2, 4, 6], dtype="float32")

            >>> out = paddle.view(x, [8, 6])
            >>> print(out.shape)
            [8, 6]

            >>> import paddle
            >>> paddle.base.set_flags({"FLAGS_use_stride_kernel": True})

            >>> x = paddle.rand([2, 4, 6], dtype="float32")

            >>> out = paddle.view(x, "uint8")
            >>> print(out.shape)
            [2, 4, 24]

            >>> import paddle
            >>> paddle.base.set_flags({"FLAGS_use_stride_kernel": True})

            >>> x = paddle.rand([2, 4, 6], dtype="float32")

            >>> out = paddle.view(x, [8, -1])
            >>> print(out.shape)
            [8, 6]

            >>> import paddle
            >>> paddle.base.set_flags({"FLAGS_use_stride_kernel": True})

            >>> x = paddle.rand([2, 4, 6], dtype="float32")

            >>> out = paddle.view(x, paddle.uint8)
            >>> print(out.shape)
            [2, 4, 24]

    """
    if isinstance(shape_or_dtype, (list, tuple)):
        return _C_ops.view_shape(x, shape_or_dtype)
    else:
        if not isinstance(
            shape_or_dtype, (core.VarDesc.VarType, core.DataType)
        ):
            shape_or_dtype = convert_np_dtype_to_dtype_(shape_or_dtype)
        return _C_ops.view_dtype(x, shape_or_dtype)


@dygraph_only
def view_as(x: Tensor, other: Tensor, name: str | None = None) -> Tensor:
    """
    View x with other's shape.

    Note that the output Tensor will share data with origin Tensor and doesn't
    have a Tensor copy in ``dygraph`` mode.

    The following figure shows a view_as operation - a three-dimensional tensor with a shape of [2, 4, 6]
    is transformed into a two-dimensional tensor with a shape of [8, 6] through the view_as operation.
    We can clearly see the corresponding relationship between the elements before and after the transformation.

    .. image:: https://githubraw.cdn.bcebos.com/PaddlePaddle/docs/develop/docs/images/api_legend/view_as.png
        :width: 800
        :alt: legend of view_as API
        :align: center

    Args:
        x (Tensor): An N-D Tensor. The data type is ``float32``, ``float64``, ``int32``, ``int64`` or ``bool``
        other (Tensor): The result tensor has the same size as other.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, A viewed Tensor with the same shape as ``other``.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.base.set_flags({"FLAGS_use_stride_kernel": True})

            >>> x = paddle.rand([2, 4, 6], dtype="float32")
            >>> y = paddle.rand([8, 6], dtype="float32")

            >>> out = paddle.view_as(x, y)
            >>> print(out.shape)
            [8, 6]
    """
    return _C_ops.view_shape(x, other.shape)


@dygraph_only
def unfold(
    x: Tensor, axis: int, size: int, step: int, name: str | None = None
) -> Tensor:
    """
    View x with specified shape, stride and offset, which contains all slices of size from x in the dimension axis.

    Note that the output Tensor will share data with origin Tensor and doesn't
    have a Tensor copy in ``dygraph`` mode.

    Args:
        x (Tensor): An N-D Tensor. The data type is ``float32``, ``float64``, ``int32``, ``int64`` or ``bool``
        axis (int): The axis along which the input is unfolded.
        size (int): The size of each slice that is unfolded.
        step (int): The step between each slice.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, A unfold Tensor with the same data type as ``x``.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.base.set_flags({"FLAGS_use_stride_kernel": True})

            >>> x = paddle.arange(9, dtype="float64")

            >>> out = paddle.unfold(x, 0, 2, 4)
            >>> print(out)
            Tensor(shape=[2, 2], dtype=float64, place=Place(cpu), stop_gradient=True,
            [[0., 1.],
             [4., 5.]])
    """
    return _C_ops.tensor_unfold(x, axis, size, step)


# TODO(dev): We need avoid implementing it by this way.
__METHODS: dict[str, Callable[..., Any]] = {
    'fill_': fill_,
    'zero_': zero_,
    'fill_diagonal_': fill_diagonal_,
    'fill_diagonal_tensor_': fill_diagonal_tensor_,
    "fill_diagonal_tensor": fill_diagonal_tensor,
    'tolist': tolist,
}
for name, func in __METHODS.items():
    setattr(core.eager.Tensor, name, func)


def _index_fill_impl(
    x: Tensor, index: Tensor, axis: int, value: Tensor, inplace: bool
) -> Tensor:
    if not isinstance(index, (Variable, paddle.pir.Value)):
        raise ValueError("index must be Tensor")

    if not isinstance(value, Variable):
        value = paddle.to_tensor(value, dtype=x.dtype)
    else:
        if len(value.shape) > 0:
            raise ValueError("value must be scalar or 0-D tensor")

    x_dim = len(x.shape)
    if not (isinstance(axis, int)) or (axis > x_dim - 1) or axis < -x_dim:
        raise ValueError(
            "The axis should be int, and in range [-rank(x), rank(x))"
        )

    if axis < 0:
        axis = axis + x_dim

    perm = list(range(len(x.shape)))
    perm[0] = axis
    perm[axis] = 0

    if inplace:
        paddle.transpose_(x, perm)
        paddle.index_put_(x, (index,), value)
        paddle.transpose_(x, perm)
        return x
    else:
        out = paddle.transpose(x, perm)
        out = paddle.index_put(out, (index,), value)
        out = paddle.transpose(out, perm)
        return out


def index_fill(
    x: Tensor, index: Tensor, axis: int, value: float, name: str | None = None
):
    """
    Fill the elements of the input tensor with value by the specific axis and index.

    As shown below, a ``[3, 3]`` 2D tensor is updated via the index_fill operation. With ``axis=0``, ``index=[0, 2]`` and ``value=-1``, the 1st and 3rd row elements become ``-1``. The resulting tensor, still [3, 3], has updated values.

    .. image:: https://githubraw.cdn.bcebos.com/PaddlePaddle/docs/develop/docs/images/api_legend/index_fill.png
       :width: 500
       :alt: Illustration of Case 2
       :align: center

    Args:
        x (Tensor) : The Destination Tensor. Supported data types are int32, int64, float16, float32, float64.
        index (Tensor): The 1-D Tensor containing the indices to index.
            The data type of ``index`` must be int32 or int64.
        axis (int): The dimension along which to index.
        value (int|float): The tensor used to fill with.
        name(str|None, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        Tensor, same dimension and dtype with x.


    Examples:
        .. code-block:: python

            >>> import paddle
            >>> input_tensor = paddle.to_tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype='int64')
            >>> index = paddle.to_tensor([0, 2], dtype="int32")
            >>> value = -1
            >>> res = paddle.index_fill(input_tensor, index, 0, value)
            >>> print(input_tensor)
            Tensor(shape=[3, 3], dtype=int64, place=Place(gpu:0), stop_gradient=True,
                   [[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])
            >>> print(res)
            Tensor(shape=[3, 3], dtype=int64, place=Place(gpu:0), stop_gradient=True,
                   [[-1, -1, -1],
                    [ 4,  5,  6],
                    [-1, -1, -1]])

    """
    return _index_fill_impl(x, index, axis, value, False)


@inplace_apis_in_dygraph_only
def index_fill_(
    x: Tensor, index: Tensor, axis: int, value: float, name: str | None = None
):
    """
    Inplace version of ``index_fill`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_paddle_index_fill`.
    """
    return _index_fill_impl(x, index, axis, value, True)


def diagonal_scatter(
    x: Tensor,
    y: Tensor,
    offset: int = 0,
    axis1: int = 0,
    axis2: int = 1,
    name: str | None = None,
) -> Tensor:
    """
    Embed the values of Tensor ``y`` into Tensor ``x`` along the diagonal elements
    of Tensor ``x``, with respect to ``axis1`` and ``axis2``.

    This function returns a tensor with fresh storage.

    The argument ``offset`` controls which diagonal to consider:

    - If ``offset`` = 0, it is the main diagonal.
    - If ``offset`` > 0, it is above the main diagonal.
    - If ``offset`` < 0, it is below the main diagonal.

    Note:
        ``y`` should have the same shape as :ref:`paddle.diagonal <api_paddle_diagonal>`.

    The image below demonstrates the example: A 2D tensor with a shape of [2, 3] is ``diagonal_scatter`` along its main diagonal (``offset = 0``)  within ``axis1 = 0`` and ``axis2 = 1`` using a 1D tensor filled with ones.

    .. image:: https://githubraw.cdn.bcebos.com/PaddlePaddle/docs/develop/docs/images/api_legend/diagonal_scatter.png
       :width: 500
       :alt: legend of diagonal_scatter API

    Args:
        x (Tensor): ``x`` is the original Tensor. Must be at least 2-dimensional.
        y (Tensor): ``y`` is the Tensor to embed into ``x``
        offset (int, optional): which diagonal to consider. Default: 0 (main diagonal).
        axis1 (int, optional): first axis with respect to which to take diagonal. Default: 0.
        axis2 (int, optional): second axis with respect to which to take diagonal. Default: 1.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, Tensor with diagonal embedded with ``y``.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> x = paddle.arange(6.0).reshape((2, 3))
            >>> y = paddle.ones((2,))
            >>> out = x.diagonal_scatter(y)
            >>> print(out)
            Tensor(shape=[2, 3], dtype=float32, place=Place(gpu:0), stop_gradient=True,
                   [[1., 1., 2.],
                    [3., 1., 5.]])

    """
    return fill_diagonal_tensor(x, y, offset, axis1, axis2, name)


def select_scatter(
    x: Tensor, values: Tensor, axis: int, index: int, name: str | None = None
) -> Tensor:
    """
    Embeds the values of the values tensor into x at the given index of axis.

    Args:
        x (Tensor) : The Destination Tensor. Supported data types are `bool`, `float16`, `float32`, `float64`, `uint8`, `int8`, `int16`, `int32`, `int64`, `bfloat16`, `complex64`, `complex128`.
        values (Tensor) : The tensor to embed into x. Supported data types are `bool`, `float16`, `float32`, `float64`, `uint8`, `int8`, `int16`, `int32`, `int64`, `bfloat16`, `complex64`, `complex128`.
        axis (int) : the dimension to insert the slice into.
        index (int) : the index to select with.
        name (str|None, optional): Name for the operation (optional, default is None).

    Returns:
        Tensor, same dtype and shape with x

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.zeros((2,3,4)).astype("float32")
            >>> values = paddle.ones((2,4)).astype("float32")
            >>> res = paddle.select_scatter(x,values,1,1)
            >>> print(res)
            Tensor(shape=[2, 3, 4], dtype=float32, place=Place(cpu), stop_gradient=True,
                   [[[0., 0., 0., 0.],
                     [1., 1., 1., 1.],
                     [0., 0., 0., 0.]],
                    [[0., 0., 0., 0.],
                     [1., 1., 1., 1.],
                     [0., 0., 0., 0.]]])

    """
    x_shape = x.shape
    value_shape = values.shape
    if not isinstance(x_shape, list):
        x_shape = list(x_shape)
    if index < 0:
        index += x_shape[axis]
    if axis < 0:
        axis += len(x_shape)
    del x_shape[axis]
    if len(x_shape) != len(value_shape):
        raise RuntimeError(
            "expected values to have a size equal to the slice of x. value size = "
            + str(value_shape)
            + " slice size = "
            + str(x_shape)
        )
    for i in range(len(x_shape)):
        if x_shape[i] != value_shape[i]:
            raise RuntimeError(
                "expected values to have a size equal to the slice of x. value size = "
                + str(value_shape)
                + " slice size = "
                + str(x_shape)
            )
    from ..base.framework import default_main_program

    starts = [index]
    ends = [index + 1]
    steps = [1]
    axes = [axis]
    none_axes = []
    decrease_axes = [axis]
    inputs = {'Input': x}
    attrs = {
        'axes': axes,
        'starts': starts,
        'ends': ends,
        'steps': steps,
        'decrease_axes': decrease_axes,
        'none_axes': none_axes,
    }

    dtype = x.dtype
    attrs['dtype'] = dtype

    values = values.astype(dtype)
    inputs["ValueTensor"] = values

    if in_dynamic_or_pir_mode():
        return _C_ops.set_value_with_tensor(
            x,
            values,
            starts,
            ends,
            steps,
            axes,
            decrease_axes,
            none_axes,
        )
    else:
        helper = LayerHelper('select_scatter', **locals())
        output = helper.create_variable_for_type_inference(dtype=x.dtype)
        cur_block = default_main_program().current_block()
        cur_block.append_op(
            type="set_value",
            inputs=inputs,
            outputs={'Out': output},
            attrs=attrs,
            inplace_map={"Input": "Out"},
        )

        return output


def slice_scatter(
    x: Tensor,
    value: Tensor,
    axes: Sequence[int],
    starts: Sequence[int],
    ends: Sequence[int],
    strides: Sequence[int],
    name: str | None = None,
) -> Tensor:
    """
    Embeds the `value` tensor into `x` along multiple axes. Returns a new tensor instead of a view.
    The size of `axes` must be equal to `starts` , `ends` and `strides`.

    Args:
        x (Tensor) : The input Tensor. Supported data types are `bool`, `float16`, `float32`, `float64`, `uint8`, `int8`, `int16`, `int32`, `int64`, `bfloat16`, `complex64`, `complex128`.
        value (Tensor) : The tensor to embed into x. Supported data types are `bool`, `float16`, `float32`, `float64`, `uint8`, `int8`, `int16`, `int32`, `int64`, `bfloat16`, `complex64`, `complex128`.
        axes (list|tuple) : the dimensions to insert the value.
        starts (list|tuple) : the start indices of where to insert.
        ends (list|tuple) : the stop indices of where to insert.
        strides (list|tuple) : the steps for each insert.
        name (str|None, optional): Name for the operation (optional, default is None).

    Returns:
        Tensor, same dtype and shape with x

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.zeros((3, 9))
            >>> value = paddle.ones((3, 2))
            >>> res = paddle.slice_scatter(x, value, axes=[1], starts=[2], ends=[6], strides=[2])
            >>> print(res)
            Tensor(shape=[3, 9], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[0., 0., 1., 0., 1., 0., 0., 0., 0.],
             [0., 0., 1., 0., 1., 0., 0., 0., 0.],
             [0., 0., 1., 0., 1., 0., 0., 0., 0.]])

            >>> # broadcast `value` got the same result
            >>> x = paddle.zeros((3, 9))
            >>> value = paddle.ones((3, 1))
            >>> res = paddle.slice_scatter(x, value, axes=[1], starts=[2], ends=[6], strides=[2])
            >>> print(res)
            Tensor(shape=[3, 9], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[0., 0., 1., 0., 1., 0., 0., 0., 0.],
             [0., 0., 1., 0., 1., 0., 0., 0., 0.],
             [0., 0., 1., 0., 1., 0., 0., 0., 0.]])

            >>> # broadcast `value` along multiple axes
            >>> x = paddle.zeros((3, 3, 5))
            >>> value = paddle.ones((1, 3, 1))
            >>> res = paddle.slice_scatter(x, value, axes=[0, 2], starts=[1, 0], ends=[3, 4], strides=[1, 2])
            >>> print(res)
            Tensor(shape=[3, 3, 5], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[[0., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0.]],
             [[1., 0., 1., 0., 0.],
              [1., 0., 1., 0., 0.],
              [1., 0., 1., 0., 0.]],
             [[1., 0., 1., 0., 0.],
              [1., 0., 1., 0., 0.],
              [1., 0., 1., 0., 0.]]])

    """
    none_axes = []
    decrease_axes = []
    dtype = x.dtype
    value = value.astype(dtype)

    if in_dynamic_or_pir_mode():
        return _C_ops.set_value_with_tensor(
            x,
            value,
            starts,
            ends,
            strides,
            axes,
            decrease_axes,
            none_axes,
        )
    else:
        attrs = {
            'axes': axes,
            'starts': starts,
            'ends': ends,
            'steps': strides,
            'decrease_axes': decrease_axes,
            'none_axes': none_axes,
            'dtype': dtype,
        }

        inputs = {
            'Input': x,
            'ValueTensor': value,
        }

        helper = LayerHelper('slice_scatter', **locals())
        output = helper.create_variable_for_type_inference(dtype=x.dtype)
        cur_block = default_main_program().current_block()
        cur_block.append_op(
            type="set_value",
            inputs=inputs,
            outputs={'Out': output},
            attrs=attrs,
            inplace_map={"Input": "Out"},
        )

        return output


def block_diag(inputs: Sequence[Tensor], name: str | None = None) -> Tensor:
    """
    Create a block diagonal matrix from provided tensors.

    Args:
        inputs (list|tuple): ``inputs`` is a Tensor list or Tensor tuple, one or more tensors with 0, 1, or 2 dimensions. The data type: ``bool``, ``float16``, ``float32``, ``float64``, ``uint8``, ``int8``, ``int16``, ``int32``, ``int64``, ``bfloat16``, ``complex64``, ``complex128``.
        name (str|None, optional): Name for the operation (optional, default is None).

    Returns:
        Tensor, A ``Tensor``. The data type is same as ``inputs``.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> A = paddle.to_tensor([[4], [3], [2]])
            >>> B = paddle.to_tensor([7, 6, 5])
            >>> C = paddle.to_tensor(1)
            >>> D = paddle.to_tensor([[5, 4, 3], [2, 1, 0]])
            >>> E = paddle.to_tensor([[8, 7], [7, 8]])
            >>> out = paddle.block_diag([A, B, C, D, E])
            >>> print(out)
            Tensor(shape=[9, 10], dtype=int64, place=Place(gpu:0), stop_gradient=True,
                [[4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 7, 6, 5, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 5, 4, 3, 0, 0],
                [0, 0, 0, 0, 0, 2, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 8, 7],
                [0, 0, 0, 0, 0, 0, 0, 0, 7, 8]])
    """

    def to_col_block(arys, i, a):
        return [
            (
                a
                if idx == i
                else paddle.zeros([ary.shape[0], a.shape[1]], dtype=a.dtype)
            )
            for idx, ary in enumerate(arys)
        ]

    def to_2d(ary):
        if ary.ndim == 0:
            return ary.unsqueeze(axis=0).unsqueeze(axis=0)
        if ary.ndim == 1:
            return ary.unsqueeze(axis=0)
        if ary.ndim == 2:
            return ary
        raise ValueError(
            "For 'block_diag', the dimension of each elements in 'inputs' must be 0, 1, or 2, but got "
            f"{ary.ndim}"
        )

    arys = [to_2d(ary) for ary in inputs]

    matrix = [
        paddle.concat(to_col_block(arys, idx, ary), axis=0)
        for idx, ary in enumerate(arys)
    ]
    return paddle.concat(matrix, axis=1)
