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


import copy

import paddle
from paddle.distributed.utils.stream_utils import ExecutionStreamType

from ..process_group import new_process_group
from .base_reshard_func import (
    ReshardFunction,
    is_shard,
)


class SToSReshardFunction(ReshardFunction):
    def is_suitable(self, src_dist_attr, dst_dist_attr):
        if not is_shard(src_dist_attr):
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
        """
        Reshard from shard to shard status on 1D mesh.
        E.g. tensor shape: [B, S, H], mesh = [0, 1]
        1. [Shard(0)] --> [Shard(1)], N ranks:
          1). reshape from [B, S, H] -> [B, N, S/N, H]
          2). transpose from [B, N, S/N, H] -> [N, B, S/N, H]
          3). reshape from [N, B, S/N, H] -> [N*B, S/N, H]
          4). all to all communicate
        2. [Shard(1)] --> [Shard(0)], N ranks:
          1). all to all communicate
          2). reshape from [B, S, H] -> [N, B/N, S, H]
          3). transpose from [N, B/N, S, H] -> [B/N, N, S/N, H]
          4). reshape from [B/N, N, S/N, H] -> [B, S, H]
        """
        in_split_axis = src_dist_attr.dims_mapping.index(0)
        out_split_axis = dst_dist_attr.dims_mapping.index(0)
        nranks = len(src_dist_attr.process_mesh.process_ids)

        if out_split_axis != 0:
            pre_shape = copy.copy(src_value.shape)
            pre_shape[out_split_axis] = pre_shape[out_split_axis] // nranks
            pre_shape.insert(out_split_axis, nranks)
            out_reshape1 = paddle._C_ops.reshape(src_value, pre_shape)

            axes = [out_split_axis]
            for i in range(len(pre_shape)):
                if i != out_split_axis:
                    axes.append(i)
            out_transpose = paddle._C_ops.transpose(out_reshape1, axes)

            pre_shape.pop(out_split_axis)
            pre_shape[in_split_axis] *= nranks
            in_all2all = paddle._C_ops.reshape(out_transpose, pre_shape)
            in_all2all_type = paddle.base.libpaddle.pir.cvt_to_dist_type(
                src_value.type(), dst_dist_attr
            )
            in_all2all.set_type(in_all2all_type)
        else:
            in_all2all = paddle._C_ops.share_data_(src_value)

        src_mesh = src_dist_attr.process_mesh
        group = new_process_group(sorted(src_mesh.process_ids))
        dst_value = paddle._C_ops.all_to_all(in_all2all, group.id)
        dst_value.get_defining_op().set_execution_stream(
            ExecutionStreamType.DefaultStream.value
        )
        out_all2all_type = paddle.base.libpaddle.pir.cvt_to_dist_type(
            in_all2all.type(), src_dist_attr
        )
        dst_value.set_type(out_all2all_type)
        dst_value.get_defining_op().dist_attr = (
            paddle.base.libpaddle.pir.create_op_dist_attribute(
                src_mesh, [src_dist_attr], [dst_dist_attr], -1
            )
        )

        if in_split_axis != 0:
            post_shape = copy.copy(src_value.shape)
            post_shape[0] = post_shape[0] // nranks
            post_shape.insert(0, nranks)
            dst_value = paddle.reshape(dst_value, post_shape)

            axes = list(range(1, len(post_shape)))
            axes.insert(in_split_axis, 0)
            dst_value = paddle._C_ops.transpose(dst_value, axes)

            post_shape.pop(0)
            post_shape[in_split_axis] *= nranks
            dst_value = paddle._C_ops.reshape(dst_value, post_shape)

        dst_value.set_type(dst_type)

        return dst_value
