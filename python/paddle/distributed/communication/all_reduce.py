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

from typing import TYPE_CHECKING

import paddle
from paddle.distributed.communication import stream
from paddle.distributed.communication.reduce import ReduceOp

if TYPE_CHECKING:
    from paddle import Tensor
    from paddle.base.core import task
    from paddle.distributed.communication.group import Group
    from paddle.distributed.communication.reduce import _ReduceOp


def all_reduce(
    tensor: Tensor,
    op: _ReduceOp = ReduceOp.SUM,
    group: Group | None = None,
    sync_op: bool = True,
) -> task:
    """

    Reduce a tensor over all ranks so that all get the result.
    As shown below, one process is started with a GPU and the data of this process is represented
    by its group rank. The reduce operator is sum. Through all_reduce operator,
    each GPU will have the sum of the data from all GPUs.

    .. image:: https://githubraw.cdn.bcebos.com/PaddlePaddle/docs/develop/docs/api/paddle/distributed/img/allreduce.png
        :width: 800
        :alt: all_reduce
        :align: center

    Args:
        tensor (Tensor): The input Tensor. It also works as the output Tensor. Its data type
            should be float16, float32, float64, int32, int64, int8, uint8 or bool.
        op (ReduceOp.SUM|ReduceOp.MAX|ReduceOp.MIN|ReduceOp.PROD|ReduceOp.AVG, optional): The operation used. Default value is ReduceOp.SUM.
        group (Group|None, optional): The group instance return by new_group or None for global default group.
        sync_op (bool, optional): Whether this op is a sync op. Default value is True.

    Returns:
        Return a task object.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env: DISTRIBUTED)
            >>> import paddle
            >>> import paddle.distributed as dist

            >>> dist.init_parallel_env()
            >>> if dist.get_rank() == 0:
            ...     data = paddle.to_tensor([[4, 5, 6], [4, 5, 6]])
            >>> else:
            ...     data = paddle.to_tensor([[1, 2, 3], [1, 2, 3]])
            >>> dist.all_reduce(data)
            >>> print(data)
            >>> # [[5, 7, 9], [5, 7, 9]] (2 GPUs)
    """
    # AVG is only supported when nccl >= 2.10
    if op == ReduceOp.AVG and paddle.base.core.nccl_version() < 21000:
        group = (
            paddle.distributed.collective._get_global_group()
            if group is None
            else group
        )
        tensor.scale_(1.0 / group.nranks)
        return stream.all_reduce(
            tensor,
            op=ReduceOp.SUM,
            group=group,
            sync_op=sync_op,
            use_calc_stream=False,
        )

    return stream.all_reduce(
        tensor, op=op, group=group, sync_op=sync_op, use_calc_stream=False
    )
