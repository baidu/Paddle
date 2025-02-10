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

from paddle import _legacy_C_ops
from paddle.common_ops_import import check_variable_and_dtype
from paddle.framework import LayerHelper, in_dynamic_mode


def global_scatter(
    x, local_count, global_count, group=None, use_calc_stream=True
):
    """
    The global_scatter operator distributes the data of x to n_expert * world_size experts according to local_count,
    and then receives data according to global_count. The expert refers to a user-defined expert network,
    n_expert refers to the number of expert networks owned by each card, and world_size refers to the number of graphics cards running the network.

    As shown below, the value of the world size is 2, n_expert 2, the batch size of the x 4 and local_count is [2, 0, 2, 0].
    The global_count of the rank 0 is [2, 0, , ], rank 1 is [2, 0, ,](Due to the limited space, only the data calculated on rank 0 is shown here).
    In the global_scatter operator, local_count[i] represents sending local_count[i] data to the (i % n_expert)th expert of the (i // n_expert)th card,
    global_count[i] represents receiving global_count[i] data from the (i // n_expert)th card to the (i % n_expert)th expert of this card. The rank in the
    figure represent the rank of the current card in all cards.

    The process of global_scatter sending data is as follows:

    local_count[0] represents taking out 2 batches from x and sending 2 batches to the 0th expert of the 0th card;

    local_count[1] represents taking out 0 batches from x and sending 0 batches to the 1st expert of the 0th card;

    local_count[2] represents taking out 2 batches from x and sending 2 batches to the 0th expert of the 1st card;

    local_count[3] represents taking out 0 batches from x and sending 0 batches to the 1st expert of the 1st card;

    Therefore, the global_count[0] of the 0th card is equal to 2, which means that 2 batches of data are received from the 0th card to the 0th expert;

    the global_count[1] of the 0th card is equal to 0, which means that 0 batches of data are received from the 0th card to the 1st expert;

    the global_count[0] of the 1st card is equal to 2, which means that 2 batches of data are received from the 0th card to the 0th expert;

    the global_count[1] of the 1st card is equal to 0, which means that 0 batches of data are received from the 0th card to the 1st expert.

    .. image:: https://githubraw.cdn.bcebos.com/PaddlePaddle/docs/develop/docs/api/paddle/distributed/img/global_scatter_gather.png
        :width: 800
        :alt: global_scatter_gather
        :align: center

    Args:
        x (Tensor): Tensor. The tensor data type should be float16, float32, float64, int32 or int64.
        local_count (Tensor): Tensor which have n_expert * world_size elements that indicates
            how many data needed to be sent. The tensor data type should be int64.
        global_count (Tensor): Tensor which have n_expert * world_size elements that indicates
            how many data needed to be received. The tensor data type should be int64.
        group (Group, optional): The group instance return by new_group or None for global default group. Default: None.
        use_calc_stream (bool, optional): Whether to use calculation stream (True) or communication stream. Default: True.

    Returns:
        out (Tensor): The data received from all experts.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:DISTRIBUTED)
            >>> import paddle
            >>> from paddle.distributed import init_parallel_env
            >>> from paddle.distributed.utils import moe_utils
            >>> init_parallel_env()
            >>> n_expert = 2
            >>> world_size = 2
            >>> d_model = 2
            >>> in_feat = d_model
            >>> local_input_buf = paddle.to_tensor(
            ...     [[1, 2],[3, 4],[5, 6],[7, 8],[9, 10]],
            ...     dtype='float32',
            ...     stop_gradient=False
            ... )
            >>> if paddle.distributed.ParallelEnv().local_rank == 0:
            ...     local_count = paddle.to_tensor([2, 1, 1, 1], dtype="int64")
            ...     global_count = paddle.to_tensor([2, 1, 1, 1], dtype="int64")
            >>> else:
            ...     local_count = paddle.to_tensor([1, 1, 2, 1], dtype="int64")
            ...     global_count = paddle.to_tensor([1, 1, 2, 1], dtype="int64")
            >>> a = moe_utils.global_scatter(local_input_buf,
            ...     local_count,
            ...     global_count
            ... )
            >>> a.stop_gradient = False
            >>> print(a)
            >>> # out for rank 0: [[1, 2], [3, 4], [1, 2], [5, 6], [3, 4]]
            >>> # out for rank 1: [[7, 8], [5, 6], [7, 8], [9, 10], [9, 10]]
            >>> # backward test
            >>> c = a * a
            >>> c.backward()
            >>> print("local_input_buf.grad: ", local_input_buf.grad)
            >>> # out for rank 0: [[2, 4], [6, 8], [10, 12], [14, 16], [18, 20]]
            >>> # out for rank 1: [[2, 4], [6, 8], [10, 12], [14, 16], [18, 20]]

    """
    if group is not None and not group.is_member():
        return

    ring_id = 0 if group is None else group.id
    if in_dynamic_mode():
        return _legacy_C_ops.global_scatter(
            x,
            local_count,
            global_count,
            'use_calc_stream',
            use_calc_stream,
            'ring_id',
            ring_id,
        )
    else:
        op_type = 'global_scatter'
        check_variable_and_dtype(
            x,
            'x',
            ['float16', 'float32', 'float64', 'int32', 'int64', 'uint16'],
            'global_scatter',
        )
        check_variable_and_dtype(
            local_count, 'local_count', ['int64'], 'global_scatter'
        )
        check_variable_and_dtype(
            global_count, 'global_count', ['int64'], 'global_scatter'
        )

        helper = LayerHelper(op_type, **locals())
        out = helper.create_variable_for_type_inference(dtype=x.dtype)

        helper.append_op(
            type=op_type,
            inputs={
                'X': [x],
                'local_count': [local_count],
                'global_count': [global_count],
            },
            outputs={'Out': [out]},
            attrs={'ring_id': ring_id, 'use_calc_stream': use_calc_stream},
        )
        return out


def global_gather(
    x, local_count, global_count, group=None, use_calc_stream=True
):
    """
    The global_gather operator gathers the data of x into n_expert * world_size experts according to global_count, and then receives data according to local_count.
    The expert refers to a user-defined expert network, n_expert refers to the number of expert networks owned by each card, and world_size refers to the number of graphics cards running the network.

    As shown below, the value of the world size is 2, n_expert 2, the batch size of the x 4 and local_count is [2, 0, 2, 0].
    The global_count of the rank 0 is [2, 0, , ], rank 1 is [2, 0, ,](Due to the limited space, only the data calculated on rank 0 is shown here).
    In the global_gather operator, the meaning of the global_count and local_count is opposed to global_scatter, global_count[i] represents sending global_count[i] data to the (i % n_expert)th expert of the (i // n_expert)th card,
    local_count[i] represents receiving local_count[i] data from the (i // n_expert)th card to the (i % n_expert)th expert of this card. The data sent will be arranged according to the experts of each card.
    The rank in the figure represent the rank of the current card in all cards.

    The process of global_gather sending data is as follows:

    The global_count[0] of the 0th card represents sending 2 data to the 0th expert of the 0th card;

    The global_count[1] of the 0th card represents sending 0 data to the 1st expert of the 0th card;

    The global_count[0] of the 1st card represents sending 2 data to the 0th expert of the 0th card;

    The global_count[1] of the 1st card represents sending 0 data to the 1st expert of the 0th card.

    .. image:: https://githubraw.cdn.bcebos.com/PaddlePaddle/docs/develop/docs/api/paddle/distributed/img/global_scatter_gather.png
        :width: 800
        :alt: global_scatter_gather
        :align: center


    Args:
        x (Tensor): Tensor. Tensor whose data type should be float16, float32, float64, int32 or int64.
        local_count (Tensor): Tensor which have n_expert * world_size elements that indicates
            how many data needed to be received. Tensor data type should be int64.
        global_count (Tensor): Tensor which have n_expert * world_size elements that indicates
            how many data needed to be sent. Tensor data type should be int64.
        group (Group, optional): The group instance return by new_group or None for global default group. Default: None.
        use_calc_stream (bool, optional): Whether to use calculation stream (True) or communication stream. Default: True.

    Returns:
        out (Tensor): The data received from all experts.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:DISTRIBUTED)
            >>> import paddle
            >>> from paddle.distributed import init_parallel_env
            >>> from paddle.distributed.utils import moe_utils
            >>> init_parallel_env()
            >>> n_expert = 2
            >>> world_size = 2
            >>> d_model = 2
            >>> in_feat = d_model
            >>> local_input_buf = paddle._to_tensor(
            ...     [[1, 2],[3, 4],[5, 6],[7, 8],[9, 10]],
            ...     dtype='float32',
            ...     stop_gradient=False
            ... )
            >>> if paddle.distributed.ParallelEnv().local_rank == 0:
            ...     local_count = paddle.to_tensor([2, 1, 1, 1], dtype="int64")
            ...     global_count = paddle.to_tensor([2, 1, 1, 1], dtype="int64")
            >>> else:
            ...     local_count = paddle.to_tensor([1, 1, 2, 1], dtype="int64")
            ...     global_count = paddle.to_tensor([1, 1, 2, 1], dtype="int64")
            >>> a = moe_utils.global_gather(
            ...     local_input_buf,
            ...     local_count,
            ...     global_count
            ... )
            >>> print(a)
            >>> # out for rank 0: [[1, 2], [3, 4], [7, 8], [1, 2], [7, 8]]
            >>> # out for rank 1: [[5, 6], [9, 10], [3, 4], [5, 6], [9, 10]]
            >>> a.stop_gradient = False
            >>> c = a * a
            >>> c.backward()
            >>> print("local_input_buf.grad", local_input_buf.grad)
            >>> # out for rank 0: [[2, 4], [6, 8], [10, 12], [14, 16], [18, 20]]
            >>> # out for rank 1: [[2, 4], [6, 8], [10, 12], [14, 16], [18, 20]]

    """
    if group is not None and not group.is_member():
        return

    ring_id = 0 if group is None else group.id
    if in_dynamic_mode():
        return _legacy_C_ops.global_gather(
            x,
            local_count,
            global_count,
            'use_calc_stream',
            use_calc_stream,
            'ring_id',
            ring_id,
        )
    else:
        op_type = 'global_gather'
        check_variable_and_dtype(
            x,
            'x',
            ['float16', 'float32', 'float64', 'int32', 'int64', 'uint16'],
            'global_gather',
        )

        check_variable_and_dtype(
            local_count, 'local_count', ['int64'], 'global_gather'
        )

        check_variable_and_dtype(
            global_count, 'global_count', ['int64'], 'global_gather'
        )
        helper = LayerHelper(op_type, **locals())
        out = helper.create_variable_for_type_inference(dtype=x.dtype)

        helper.append_op(
            type=op_type,
            inputs={
                'X': [x],
                'local_count': [local_count],
                'global_count': [global_count],
            },
            outputs={'Out': [out]},
            attrs={
                'ring_id': group,
                'use_calc_stream': use_calc_stream,
            },
        )
        return out
