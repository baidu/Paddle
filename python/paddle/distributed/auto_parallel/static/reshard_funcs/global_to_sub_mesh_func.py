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

from .base_reshard_func import (
    ReshardFunction,
    is_replicated,
)
from .nd_mesh_reshard_func import NdMeshReshardFunction


class GlobaleToSubMeshFunction(ReshardFunction):
    def is_suitable(self, src_dist_attr, dst_dist_attr):

        # NOTE we could allow the src_dist_attr is not replicated and reshard it as replicated before go through the global_to_sub logic
        # but the dst_dist_attr should be replicated otherwise there will be un-defined result when change the mesh.
        if not is_replicated(dst_dist_attr):
            return False
        in_mesh = src_dist_attr.process_mesh
        out_mesh = dst_dist_attr.process_mesh

        sub_mesh_dim = paddle.base.core.sub_mesh_dim(in_mesh, out_mesh)
        if sub_mesh_dim == -1:
            return False
        return src_dist_attr.placements_attr[sub_mesh_dim].is_replicated()
        # if in_mesh.ndim > out_mesh.ndim + 1:
        #     return False
        # if in_mesh.ndim == out_mesh.ndim:
        #     return set(out_mesh.process_ids) < set(in_mesh.process_ids)
        # else:
        #     sub_meshes = paddle.base.libpaddle.pir.get_sub_meshes(in_mesh)
        #     return out_mesh in sub_meshes

    def reshard(self, src_dist_attr, dst_dist_attr, src_value, dst_type):

        # reshard operand as replicated before change the mesh.
        if not is_replicated(src_dist_attr):
            tmp_dist_attr = (
                paddle.base.libpaddle.pir.create_tensor_dist_attribute(
                    src_dist_attr.process_mesh,
                    [-1] * len(src_dist_attr.dims_mapping),
                    {},
                )
            )
            tmp_dst_type = paddle.base.libpaddle.pir.cvt_to_dist_type(
                src_value.type(), tmp_dist_attr
            )

            pre_reshard_func = NdMeshReshardFunction()
            src_value = pre_reshard_func.reshard(
                src_dist_attr,
                tmp_dist_attr,
                src_value,
                tmp_dst_type,
            )
            src_dist_attr = tmp_dist_attr

        if src_value.has_one_use():
            src_value.update_dist_attr(dst_dist_attr)
            prev_op = src_value.get_defining_op()
            op_dist_attr = prev_op.dist_attr
            op_mesh = op_dist_attr.process_mesh
            operands = op_dist_attr.operands()
            results = op_dist_attr.results()
            chunk_id = op_dist_attr.chunk_id
            results[src_value.index()] = dst_dist_attr
            prev_op.dist_attr = (
                paddle.base.libpaddle.pir.create_op_dist_attribute(
                    op_mesh, operands, results, chunk_id
                )
            )
            return src_value
        else:
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
