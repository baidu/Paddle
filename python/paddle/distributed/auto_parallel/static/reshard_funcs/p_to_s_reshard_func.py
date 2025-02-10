# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.distributed.utils.stream_utils import ExecutionStreamType

from ..process_group import new_process_group
from .base_reshard_func import (
    ReshardFunction,
    copy_dist_attr_with_new_member,
    is_partial,
    is_shard,
)


class PToSReshardFunction(ReshardFunction):
    def is_suitable(self, src_dist_attr, dst_dist_attr):
        if not is_partial(src_dist_attr):
            return False

        if not is_shard(dst_dist_attr):
            return False

        in_mesh = src_dist_attr.process_mesh
        out_mesh = dst_dist_attr.process_mesh

        if in_mesh.ndim != 1:
            return False
        if out_mesh.ndim != 1:
            return False
        if in_mesh != out_mesh:
            return False
        return True

    def reshard(self, src_dist_attr, dst_dist_attr, src_value, dst_type):
        src_mesh = src_dist_attr.process_mesh
        src_reduce_type = src_dist_attr.partial_status[0]
        assert (
            src_reduce_type == paddle.base.core.ReduceType.kRedSum
        ), f"The p to s reshard func only support sum op, but received {src_reduce_type}"

        chunk_id = -1
        if src_value.get_defining_op().dist_attr:
            chunk_id = src_value.get_defining_op().dist_attr.chunk_id

        split_axis = dst_dist_attr.dims_mapping.index(0)
        permute = False
        if split_axis != 0:
            perm = list(range(0, len(src_value.shape)))
            perm[0] = split_axis
            perm[split_axis] = 0
            src_value = paddle._C_ops.transpose(src_value, perm)
            permute = True
            tmp_dims_mapping = dst_dist_attr.dims_mapping
            tmp_dims_mapping[split_axis] = -1
            tmp_dims_mapping[0] = 0
            dst_dist_attr = copy_dist_attr_with_new_member(
                dst_dist_attr, new_dims_mapping=tmp_dims_mapping
            )
            global_dst_attr = dst_type.as_dist_type().dist_attr()
            global_dims_mapping = global_dst_attr.dims_mapping
            axis = global_dims_mapping[0]
            global_dims_mapping[0] = global_dims_mapping[split_axis]
            global_dims_mapping[split_axis] = axis
            global_dist_attr = copy_dist_attr_with_new_member(
                global_dst_attr, new_dims_mapping=global_dims_mapping
            )
            dst_type = paddle.base.libpaddle.pir.cvt_to_dist_type(
                src_value.type(), global_dist_attr
            )

        num_of_process = len(src_dist_attr.process_mesh.process_ids)
        remainder_of_padding = src_value.shape[split_axis] % num_of_process
        is_balanced_split = remainder_of_padding == 0

        if is_balanced_split:
            group = new_process_group(sorted(src_mesh.process_ids))
            dst_value = paddle._C_ops.reduce_scatter(
                src_value, group.id, num_of_process
            )
            dst_value.get_defining_op().set_execution_stream(
                ExecutionStreamType.DefaultStream.value
            )

            # set dist type and dist attr
            dst_value.set_type(dst_type)
            dst_value.get_defining_op().dist_attr = (
                paddle.base.libpaddle.pir.create_op_dist_attribute(
                    src_mesh, [src_dist_attr], [dst_dist_attr], chunk_id
                )
            )

            if split_axis != 0:
                dst_value = paddle._C_ops.transpose(dst_value, perm)
            return dst_value
        else:
            dst_type = paddle.base.libpaddle.pir.cvt_to_dist_type(
                src_value.type(), dst_dist_attr
            )
            reduce_scatter_axis = 0
            avg_size_on_reduce_scatter_axis = int(
                (src_value.shape[reduce_scatter_axis] + num_of_process - 1)
                / num_of_process
            )
            padding_num = (
                avg_size_on_reduce_scatter_axis * num_of_process
                - src_value.shape[reduce_scatter_axis]
            )
            padding_shape = src_value._local_shape
            padding_shape[reduce_scatter_axis] = padding_num
            padding_tensor = paddle.full(
                padding_shape,
                0.0,
                src_value.dtype,
            )
            tmp_src_type = paddle.base.libpaddle.pir.cvt_to_dist_type(
                padding_tensor.type(), src_dist_attr
            )
            padding_tensor.set_type(tmp_src_type)
            padding_tensor.get_defining_op().dist_attr = (
                paddle.base.libpaddle.pir.create_op_dist_attribute(
                    src_dist_attr.process_mesh, [], [src_dist_attr], chunk_id
                )
            )
            concat_value = paddle._C_ops.concat(
                [src_value, padding_tensor], reduce_scatter_axis
            )
            axis_dist_attr = (
                paddle.base.libpaddle.pir.create_tensor_dist_attribute(
                    src_dist_attr.process_mesh,
                    [-1],
                    {0: paddle.base.core.ReduceType.kRedSum},
                )
            )
            concat_value.get_defining_op().dist_attr = (
                paddle.base.libpaddle.pir.create_op_dist_attribute(
                    src_dist_attr.process_mesh,
                    [
                        paddle.base.libpaddle.pir.create_array_attribute(
                            [src_dist_attr, src_dist_attr]
                        ),
                        axis_dist_attr,
                    ],
                    [src_dist_attr],
                    chunk_id,
                )
            )

            concat_global_shape = list(src_value.shape)
            concat_global_shape[reduce_scatter_axis] = (
                avg_size_on_reduce_scatter_axis * num_of_process
            )
            concat_type = paddle.pir.create_shaped_type(
                src_value.type(), concat_global_shape
            )
            concat_type = paddle.base.libpaddle.pir.cvt_to_dist_type(
                concat_type, src_dist_attr
            )
            concat_value.set_type(concat_type)

            dst_value = self.reshard_p_to_s_with_padding(
                concat_value,
                reduce_scatter_axis,
                src_dist_attr,
                dst_dist_attr,
                dst_type,
                padding_num,
            )
            if permute:
                dst_value = paddle._C_ops.transpose(dst_value, perm)
            return dst_value

    def reshard_p_to_s_with_padding(
        self,
        src_value,
        split_axis,
        src_dist_attr,
        dst_dist_attr,
        dst_type,
        padding_num=0,
    ):
        group = new_process_group(
            sorted(src_dist_attr.process_mesh.process_ids)
        )
        dst_value = paddle._C_ops.reduce_scatter(
            src_value, group.id, len(src_dist_attr.process_mesh.process_ids)
        )
        out_global_shape = dst_type.shape
        out_global_shape[split_axis] = (
            padding_num + out_global_shape[split_axis]
        )
        dst_tmp_type = paddle.pir.create_shaped_type(
            dst_value.type(), out_global_shape
        )
        dst_tmp_type = paddle.base.libpaddle.pir.cvt_to_dist_type(
            dst_tmp_type, dst_dist_attr
        )
        dst_value.set_type(dst_tmp_type)
        dst_value.get_defining_op().set_execution_stream(
            ExecutionStreamType.DefaultStream.value
        )
        dst_value.get_defining_op().dist_attr = (
            paddle.base.libpaddle.pir.create_op_dist_attribute(
                src_dist_attr.process_mesh,
                [src_dist_attr],
                [dst_dist_attr],
                src_value.get_defining_op().dist_attr.chunk_id,
            )
        )
        if padding_num != 0:
            if dist.get_rank() == dst_dist_attr.process_mesh.process_ids[-1]:
                dst_value = paddle._C_ops.split(
                    dst_value,
                    [
                        dst_value.shape[split_axis] - padding_num,
                        padding_num,
                    ],
                    0,
                )[0]
                dst_value.get_defining_op().dist_attr = (
                    paddle.base.libpaddle.pir.create_op_dist_attribute(
                        dst_dist_attr.process_mesh,
                        [dst_dist_attr],
                        [dst_dist_attr],
                        src_value.get_defining_op().dist_attr.chunk_id,
                    )
                )
            else:
                dst_value.set_type(dst_type)
        return dst_value
