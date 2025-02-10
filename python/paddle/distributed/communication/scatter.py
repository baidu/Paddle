# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    from paddle import Tensor
    from paddle.base.core import task
    from paddle.distributed.communication.group import Group

import numpy as np

import paddle
import paddle.distributed as dist
from paddle import framework
from paddle.distributed.communication import stream

from .serialization_utils import (
    convert_object_to_tensor,
    convert_tensor_to_object,
)


def scatter(
    tensor: Tensor,
    tensor_list: Sequence[Tensor] | None = None,
    src: int = 0,
    group: Group | None = None,
    sync_op: bool = True,
) -> task | None:
    """

    Scatter a tensor to all participators. As shown below, one process is started with a GPU and the source of the scatter
    is GPU0. Through scatter operator, the data in GPU0 will be sent to all GPUs averagely.

    .. image:: https://githubraw.cdn.bcebos.com/PaddlePaddle/docs/develop/docs/api/paddle/distributed/img/scatter.png
        :width: 800
        :alt: scatter
        :align: center

    Args:
        tensor (Tensor): The output Tensor. Its data type
            should be float16, float32, float64, int32, int64, int8, uint8, bool or bfloat16.
        tensor_list (list|tuple): A list/tuple of Tensors to scatter. Every element in the list must be a Tensor whose data type
            should be float16, float32, float64, int32, int64, int8, uint8, bool or bfloat16. Default value is None.
        src (int): The source rank id. Default value is 0.
        group (Group, optional): The group instance return by new_group or None for global default group.
        sync_op (bool, optional): Whether this op is a sync op. The default value is True.

    Returns:
        None.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env: DISTRIBUTED)
            >>> import paddle
            >>> import paddle.distributed as dist

            >>> dist.init_parallel_env()
            >>> if dist.get_rank() == 0:
            ...     data1 = paddle.to_tensor([7, 8, 9])
            ...     data2 = paddle.to_tensor([10, 11, 12])
            ...     dist.scatter(data1, src=1)
            >>> else:
            ...     data1 = paddle.to_tensor([1, 2, 3])
            ...     data2 = paddle.to_tensor([4, 5, 6])
            ...     dist.scatter(data1, tensor_list=[data1, data2], src=1)
            >>> print(data1, data2)
            >>> # [1, 2, 3] [10, 11, 12] (2 GPUs, out for rank 0)
            >>> # [4, 5, 6] [4, 5, 6] (2 GPUs, out for rank 1)
    """
    return stream.scatter(tensor, tensor_list, src, group, sync_op)


def scatter_object_list(
    out_object_list: list[Any],
    in_object_list: list[Any] | None = None,
    src: int = 0,
    group: Group | None = None,
) -> None:
    """

    Scatter picklable objects from the source to all others. Similar to scatter(), but python object can be passed in.

    Args:
        out_object_list (list): The list of objects to store the scattered objects.
        in_object_list (list): The list of objects to scatter. Only objects on the src rank will be scattered.
        src (int): The source rank in global view.
        group (Group): The group instance return by new_group or None for global default group.

    Returns:
        None.

    Warning:
        This API only supports the dygraph mode.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env: DISTRIBUTED)
            >>> import paddle.distributed as dist

            >>> dist.init_parallel_env()
            >>> out_object_list = [] # type: ignore
            >>> if dist.get_rank() == 0:
            ...     in_object_list = [{'foo': [1, 2, 3]}, {'foo': [4, 5, 6]}]
            >>> else:
            ...     in_object_list = [{'bar': [1, 2, 3]}, {'bar': [4, 5, 6]}]
            >>> dist.scatter_object_list(out_object_list, in_object_list, src=1)
            >>> print(out_object_list)
            >>> # [{'bar': [1, 2, 3]}] (2 GPUs, out for rank 0)
            >>> # [{'bar': [4, 5, 6]}] (2 GPUs, out for rank 1)
    """
    assert (
        framework.in_dynamic_mode()
    ), "scatter_object_list doesn't support static graph mode."

    rank = dist.get_rank()
    in_obj_tensors = []
    in_obj_sizes = []

    if rank == src:
        for obj in in_object_list:
            obj_tensor, obj_size = convert_object_to_tensor(obj)
            in_obj_tensors.append(obj_tensor)
            in_obj_sizes.append(obj_size)
        max_obj_size_tensor = max(in_obj_sizes)
    else:
        max_obj_size_tensor = paddle.empty([], dtype="int64")
    stream.broadcast(max_obj_size_tensor, src)
    max_obj_size = int(max_obj_size_tensor.item())

    # resize to the same size
    in_tensor_list = []
    for tensor in in_obj_tensors:
        numpy_data = tensor.numpy()
        numpy_data = np.resize(numpy_data, [max_obj_size])
        in_tensor = paddle.to_tensor(numpy_data)
        in_tensor_list.append(in_tensor)
    out_tensor = paddle.empty([max_obj_size], dtype="uint8")
    scatter(out_tensor, in_tensor_list if rank == src else None, src, group)

    out_tensor_size = paddle.empty([], dtype="int64")
    scatter(out_tensor_size, in_obj_sizes if rank == src else None, src, group)

    out_object_list.clear()
    out_object_list.append(
        convert_tensor_to_object(out_tensor, out_tensor_size.item())
    )
