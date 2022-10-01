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

import paddle
import paddle.distributed as dist
import paddle.fluid.framework as framework
from paddle.distributed import collective


def _check_tensor_shape(tensor, shape, nranks=1):
    expect_shape = list(shape)
    expect_shape[0] //= nranks
    if list(tensor.shape) != expect_shape:
        raise RuntimeError(
            "The in_tensor for reduce_scatter is not correctly-sized.")


def _check_tensor_list_shape(tensor_list, shape, nranks=1):
    if len(tensor_list) != nranks:
        raise RuntimeError(
            f"The tensor_list for reduce_scatter is not correctly-sized.")
    for tensor in tensor_list:
        if tensor.shape != shape:
            raise RuntimeError(
                f"The tensor_list for reduce_scatter is not correctly-sized.")


def _reduce_scatter_base_in_dygraph(out_tensor, in_tensor, op, group, sync_op,
                                    use_calc_stream):
    op_type = collective._get_reduce_op(op, "_reduce_scatter_base")
    group = collective._get_default_group() if group is None else group

    _check_tensor_shape(out_tensor, in_tensor.shape, group.nranks)

    if use_calc_stream:
        return group.process_group._reduce_scatter_base_on_calc_stream(
            in_tensor, out_tensor, op_type)

    task = group.process_group._reduce_scatter_base(in_tensor, out_tensor,
                                                    op_type, sync_op)
    if sync_op:
        task.wait()

    return task


def _reduce_scatter_base(out_tensor,
                         in_tensor,
                         op=collective.ReduceOp.SUM,
                         group=None,
                         sync_op=True,
                         use_calc_stream=False):
    """

    Reduce, then scatter a flattened tensor across devices.

    Args:
        out_tensor (Tensor): The output tensor on each rank. The result will overwrite this tenor after communication. Support
            float16, float32, float64, int32 or int64 as the input data type.
        in_tensor (Tensor): The input tensor to reduce and scatter.
        op (ReduceOp.SUM|ReduceOp.MAX|ReduceOp.MIN|ReduceOp.PROD, optional): The reduction used. If none is given, use ReduceOp.SUM as default.
        group (Group, optional): Communicate in which group. If none is given, use the global group as default.
        sync_op (bool, optional): Indicate whether the communication is sync or not. If none is given, use true as default.
        use_calc_stream (bool, optional): Indicate whether the communication is done on calculation stream. If none is given, use false as default. This
            option is designed for high performance demand, be careful to turn it on except you are clearly know its meaning.

    Returns:
        Return a task object.

    Warning:
        This API only supports the dygraph mode now.

    Examples:
        .. code-block:: python

            # required: distributed
            import paddle
            import paddle.distributed as dist

            dist.init_parallel_env()
            if dist.get_rank() == 0:
                data1 = paddle.to_tensor([7, 8, 9])
                data2 = paddle.to_tensor([10, 11, 12])
                dist.stream.scatter(data1, src=1)
            else:
                data1 = paddle.to_tensor([1, 2, 3])
                data2 = paddle.to_tensor([4, 5, 6])
                dist.stream.scatter(data1, [data1, data2], src=1)
            out = data1.numpy()
            # [1, 2, 3] (2 GPUs, out for rank 0)
            # [4, 5, 6] (2 GPUs, out for rank 1)
    """
    if group is not None and not group.is_member():
        raise RuntimeError(
            "The group should not be None and all ranks which invoke this operation should be the member of this group."
        )

    if not sync_op and use_calc_stream:
        raise RuntimeError(
            "use_calc_stream can only be true in sync op behavior.")

    if framework.in_dygraph_mode():
        return _reduce_scatter_base_in_dygraph(out_tensor, in_tensor, op, group,
                                               sync_op, use_calc_stream)

    raise RuntimeError(
        "paddle.distributed.stream._reduce_scatter_base is only supported in dygraph mode now."
    )


def _reduce_scatter_in_dygraph(tensor, tensor_list, op, group, sync_op,
                               use_calc_stream):
    op_type = collective._get_reduce_op(op, "reduce_scatter")
    group = collective._get_default_group() if group is None else group

    _check_tensor_list_shape(tensor_list, tensor.shape, group.nranks)

    if use_calc_stream:
        return group.process_group.reduce_scatter_on_calc_stream(
            tensor_list, tensor, op_type)

    task = group.process_group.reduce_scatter(tensor_list, tensor, op_type,
                                              sync_op)
    if sync_op:
        task.wait()

    return task


def reduce_scatter(tensor,
                   tensor_list,
                   op=collective.ReduceOp.SUM,
                   group=None,
                   sync_op=True,
                   use_calc_stream=False):
    """

    Scatter a tensor (or a tensor list) across devices.

    Args:
        tensor (Tensor): The output tensor on each rank. The result will overwrite this tenor after communication. Support
            float16, float32, float64, int32 or int64 as the input data type.
        tensor_list (List[Tensor]]): The input to scatter (default is `None`, must be specified on the source rank).
            If it is a tensor, it should be correctly-sized. If it is a list, it should contain correctly-sized tensors.
        op (ReduceOp.SUM|ReduceOp.MAX|ReduceOp.MIN|ReduceOp.PROD, optional): The reduction used. If none is given, use ReduceOp.SUM as default.
        group (Group, optional): Communicate in which group. If none is given, use the global group as default.
        sync_op (bool, optional): Indicate whether the communication is sync or not. If none is given, use true as default.
        use_calc_stream (bool, optional): Indicate whether the communication is done on calculation stream. If none is given, use false as default. This
            option is designed for high performance demand, be careful to turn it on except you are clearly know its meaning.

    Returns:
        Return a task object.

    Warning:
        This API only supports the dygraph mode now.

    Examples:
        .. code-block:: python

            # required: distributed
            import paddle
            import paddle.distributed as dist

            dist.init_parallel_env()
            if dist.get_rank() == 0:
                data1 = paddle.to_tensor([7, 8, 9])
                data2 = paddle.to_tensor([10, 11, 12])
                dist.stream.scatter(data1, src=1)
            else:
                data1 = paddle.to_tensor([1, 2, 3])
                data2 = paddle.to_tensor([4, 5, 6])
                dist.stream.scatter(data1, [data1, data2], src=1)
            out = data1.numpy()
            # [1, 2, 3] (2 GPUs, out for rank 0)
            # [4, 5, 6] (2 GPUs, out for rank 1)
    """
    if group is not None and not group.is_member():
        raise RuntimeError(
            "The group should not be None and all ranks which invoke this operation should be the member of this group."
        )

    if not sync_op and use_calc_stream:
        raise RuntimeError(
            "use_calc_stream can only be true in sync op behavior.")

    if framework.in_dygraph_mode():
        return _reduce_scatter_in_dygraph(tensor, tensor_list, op, group,
                                          sync_op, use_calc_stream)

    raise RuntimeError(
        "paddle.distributed.stream.reduce_scatter is only supported in dygraph mode now."
    )
