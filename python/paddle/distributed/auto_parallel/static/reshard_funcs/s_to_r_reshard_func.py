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

from ..process_group import new_process_group
from .base_reshard_func import (
    ReshardFunction,
    copy_op_attr_with_new_member,
    is_replicated,
    is_shard,
)
from .same_status_reshard_func import SameStatusReshardFunction


class SToRReshardFunction(ReshardFunction):
    def is_suitable(self, src_dist_attr, dst_dist_attr):
        if not is_shard(src_dist_attr):
            return False

        if not is_replicated(dst_dist_attr):
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

    def infer_allgather_dist_type(self, in_value, split_axis):
        tensor_ndim = len(in_value.shape)
        in_dist_attr = in_value.dist_attr()
        split_mesh_dim = in_dist_attr.dims_mapping[split_axis]
        mesh = in_dist_attr.process_mesh

        # Calculate local shape. In nd_mesh_reshard, multiple tensor axis
        # may be shard and it will call this 1-D s_to_r function on each
        # axis. In this case, we should recompute the local and global shape.
        out_local_shape = list(in_value.shape)
        out_local_shape[split_axis] = int(
            (in_value.shape[split_axis] + mesh.shape[split_mesh_dim] - 1)
            / mesh.shape[split_mesh_dim]
        )
        out_global_shape = list(out_local_shape)
        out_global_shape[0] *= mesh.shape[split_mesh_dim]
        out_type = paddle.pir.create_shaped_type(
            in_value.type(), out_global_shape
        )

        out_dims_mapping = list(in_dist_attr.dims_mapping)
        out_dims_mapping[split_axis] = -1
        out_dist_attr = paddle.base.libpaddle.pir.create_tensor_dist_attribute(
            mesh, out_dims_mapping, in_dist_attr.partial_status
        )
        out_type = paddle.base.libpaddle.pir.cvt_to_dist_type(
            out_type, out_dist_attr
        )
        return out_type

    def reshard(self, src_dist_attr, dst_dist_attr, src_value, dst_type):
        if src_dist_attr.process_mesh.size == 1:
            dst_value = paddle._C_ops.share_data_(src_value)
            share_data_op = dst_value.get_defining_op()
            # set dist type and dist attr
            dst_value.set_type(dst_type)

            chunk_id = -1
            if src_value.get_defining_op().dist_attr:
                chunk_id = src_value.get_defining_op().dist_attr.chunk_id
            share_data_op.dist_attr = (
                paddle.base.libpaddle.pir.create_op_dist_attribute(
                    src_dist_attr.process_mesh,
                    [src_dist_attr],
                    [dst_dist_attr],
                    chunk_id,
                )
            )
            return dst_value

        def get_split_axis_with_dims_mapping(dims_mapping):
            split_axis = {}
            for idx, v in enumerate(dims_mapping):
                if v != -1:
                    split_axis[idx] = v
            return split_axis

        split_axis_map = get_split_axis_with_dims_mapping(
            src_dist_attr.dims_mapping
        )

        split_axis = -1
        for k, v in split_axis_map.items():
            split_axis = k
            break
        num_of_process = src_dist_attr.process_mesh.size
        num_of_padding = src_value.shape[split_axis] % num_of_process
        is_balanced_split = num_of_padding == 0

        if is_balanced_split:
            new_value = self.reshard_s_to_r_with_padding(
                src_value,
                split_axis,
                src_dist_attr,
                dst_dist_attr,
                dst_type,
                num_of_padding,
            )
            return new_value
        else:
            # find the last one
            need_padding = (
                paddle.distributed.get_rank()
                == src_dist_attr.process_mesh.process_ids[-1]
            )

            # get padding_num
            avg_size_on_split_axis = int(
                (src_value.shape[split_axis] + num_of_process - 1)
                / num_of_process
            )
            padding_num = (
                avg_size_on_split_axis * num_of_process
                - src_value.shape[split_axis]
            )
            if need_padding:
                # set right _local_shape
                local_shape_at_split_axis = src_value.shape[
                    split_axis
                ] - avg_size_on_split_axis * (num_of_process - 1)
                local_shape = src_value._local_shape
                local_shape[split_axis] = local_shape_at_split_axis
                tmp_src_type = paddle.base.libpaddle.pir.cvt_to_dist_type(
                    src_value.type(), src_dist_attr, list(local_shape)
                )
                src_value.set_type(tmp_src_type)
                padding_shape = src_value._local_shape
                padding_shape[split_axis] = padding_num
                padding_tensor = paddle.full(
                    padding_shape,
                    0.0,
                    src_value.dtype,
                )
                tmp_src_type1 = paddle.base.libpaddle.pir.cvt_to_dist_type(
                    padding_tensor.type(), dst_dist_attr
                )
                padding_tensor.set_type(tmp_src_type1)
                padding_tensor.get_defining_op().dist_attr = (
                    paddle.base.libpaddle.pir.create_op_dist_attribute(
                        dst_dist_attr.process_mesh, [], [dst_dist_attr]
                    )
                )

                concat_value = paddle._C_ops.concat(
                    [src_value, padding_tensor], split_axis
                )
                # set concat dist_attr
                axis_dist_attr = (
                    paddle.base.libpaddle.pir.create_tensor_dist_attribute(
                        src_dist_attr.process_mesh, [-1], {}
                    )
                )
                concat_value.get_defining_op().dist_attr = (
                    paddle.base.libpaddle.pir.create_op_dist_attribute(
                        src_dist_attr.process_mesh,
                        [
                            paddle.base.libpaddle.pir.create_array_attribute(
                                [src_dist_attr, dst_dist_attr]
                            ),
                            axis_dist_attr,
                        ],
                        [src_dist_attr],
                    )
                )
                # set concat_value type
                concat_global_shape = list(src_value.shape)
                concat_global_shape[split_axis] = (
                    avg_size_on_split_axis * num_of_process
                )
                concat_type = paddle.pir.create_shaped_type(
                    src_value.type(), concat_global_shape
                )
                concat_type = paddle.base.libpaddle.pir.cvt_to_dist_type(
                    concat_type, src_dist_attr
                )
                concat_value.set_type(concat_type)

                new_value = self.reshard_s_to_r_with_padding(
                    concat_value,
                    split_axis,
                    src_dist_attr,
                    dst_dist_attr,
                    dst_type,
                    padding_num,
                )
                return new_value
            else:
                new_value = self.reshard_s_to_r_with_padding(
                    src_value,
                    split_axis,
                    src_dist_attr,
                    dst_dist_attr,
                    dst_type,
                    padding_num,
                )
                return new_value

    def reshard_s_to_r_with_padding(
        self,
        src_value,
        split_axis,
        src_dist_attr,
        dst_dist_attr,
        dst_type,
        padding_num=0,
    ):
        src_mesh = src_dist_attr.process_mesh
        num_of_process = len(src_mesh.process_ids)
        chunk_id = -1
        if src_value.get_defining_op().dist_attr:
            chunk_id = src_value.get_defining_op().dist_attr.chunk_id

        group = new_process_group(sorted(src_mesh.process_ids))
        allgather_value = paddle._C_ops.all_gather(
            src_value, group.id, num_of_process
        )
        allgather_type = self.infer_allgather_dist_type(src_value, split_axis)
        allgather_value.set_type(allgather_type)

        # set op_dist_attr
        new_dist_attr = paddle.base.libpaddle.pir.create_tensor_dist_attribute(
            dst_dist_attr.process_mesh,
            [-1] * len(dst_dist_attr.dims_mapping),
            dst_dist_attr.partial_status,
        )
        allgather_value.get_defining_op().dist_attr = (
            paddle.base.libpaddle.pir.create_op_dist_attribute(
                src_mesh, [src_dist_attr], [new_dist_attr], chunk_id
            )
        )

        if split_axis != 0 or padding_num != 0:
            allgather_op = allgather_value.get_defining_op()
            split_values = paddle._C_ops.split_with_num(
                allgather_op.result(0), num_of_process, 0
            )
            builtin_split_op = split_values[0].get_defining_op()
            pd_split_op = builtin_split_op.operand_source(0).get_defining_op()
            pd_split_op.dist_attr = copy_op_attr_with_new_member(
                pd_split_op.dist_attr, new_chunk_id=chunk_id
            )

            # fix the split_with_num dist attribute.
            new_inner_types = []
            for sub_value in split_values:
                new_inner_type = paddle.base.libpaddle.pir.cvt_to_dist_type(
                    sub_value.type(), allgather_value.dist_attr()
                )
                new_inner_types.append(new_inner_type)
                sub_value.set_type(new_inner_type)
            vec_type = paddle.base.libpaddle.pir.create_vec_type(
                new_inner_types
            )
            pd_split_op.result(0).set_type(vec_type)

            if padding_num != 0:
                tmp_split_values = paddle._C_ops.split(
                    split_values[-1],
                    [
                        split_values[-1].shape[split_axis] - padding_num,
                        padding_num,
                    ],
                    split_axis,
                )
                split_op = tmp_split_values.get_defining_op()
                split_op.dist_attr = copy_op_attr_with_new_member(
                    split_op.dist_attr, new_chunk_id=chunk_id
                )
                split_values[-1] = tmp_split_values[0]
                concat_value = paddle._C_ops.concat(split_values, split_axis)
                concat_op = concat_value.get_defining_op()
                concat_op.dist_attr = copy_op_attr_with_new_member(
                    concat_op.dist_attr, new_chunk_id=chunk_id
                )
                return concat_value
            else:
                concat_value = paddle._C_ops.concat(split_values, split_axis)
                # fold builtin.split op and builtin.combine op
                concat_op = concat_value.get_defining_op()
                concat_op.dist_attr = copy_op_attr_with_new_member(
                    concat_op.dist_attr, new_chunk_id=chunk_id
                )
                builtin_combine_op = concat_op.operand_source(
                    0
                ).get_defining_op()
                concat_op.operand(0).set_source(pd_split_op.result(0))
                builtin_combine_op.erase()
                builtin_split_op.erase()
                return concat_value

        return allgather_value


class SToRReshardFunctionCrossMesh(ReshardFunction):
    def is_suitable(self, src_dist_attr, dst_dist_attr):
        if not is_shard(src_dist_attr):
            return False

        if not is_replicated(dst_dist_attr):
            return False

        in_mesh = src_dist_attr.process_mesh
        out_mesh = dst_dist_attr.process_mesh

        if (
            in_mesh.ndim != 1
            or out_mesh.ndim != 1
            or in_mesh.shape != out_mesh.shape
        ):
            return False

        if in_mesh == out_mesh:
            return False

        return True

    def reshard(self, src_dist_attr, dst_dist_attr, src_value, dst_type):
        same_status_func = SameStatusReshardFunction()
        tmp_dist_attr = paddle.base.libpaddle.pir.create_tensor_dist_attribute(
            dst_dist_attr.process_mesh,
            src_dist_attr.dims_mapping,
            src_dist_attr.partial_status,
        )
        tmp_dst_type = paddle.base.libpaddle.pir.cvt_to_dist_type(
            src_value.type(), tmp_dist_attr
        )
        out_value = same_status_func.reshard(
            src_dist_attr, tmp_dist_attr, src_value, tmp_dst_type
        )

        s_to_r_func = SToRReshardFunction()
        assert s_to_r_func.is_suitable(
            tmp_dist_attr, dst_dist_attr
        ), f"Invoke the p to r reshard function is not valid from {tmp_dist_attr} to {dst_dist_attr}"
        return s_to_r_func.reshard(
            tmp_dist_attr, dst_dist_attr, out_value, dst_type
        )
