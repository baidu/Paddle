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
import paddle.distributed as dist
from paddle.distributed.auto_parallel.placement_type import (
    check_placements_equal,
)

from ..process_group import new_process_group
from ..utils import mesh_equal_ignore_shape_one, split_mesh
from .base_reshard_func import ReshardFunction, copy_dist_attr_with_new_member


class SubToGlobalMeshFunction(ReshardFunction):
    """
    Reshard from sub-mesh to global mesh, now only supports
    both input and output values are replicated, e.g.
    1. input: mesh:[0], placements:[Replicate()]
       output: mesh:[0,1], placements:[Replicate()]
    2. input: mesh:[0,1], placements:[Replicate()]
       output: mesh:[[0,1],[2,3]], placements:[Replicate(), Replicate()]
    """

    def is_suitable(self, src_dist_attr, dst_dist_attr):
        # only supports replicated input and output
        # if 0 in src_dist_attr.dims_mapping or 0 in src_dist_attr.partial_status:
        #     return False
        # if 0 in dst_dist_attr.dims_mapping or 0 in dst_dist_attr.partial_status:
        #     return False
        # in_mesh = src_dist_attr.process_mesh
        # out_mesh = dst_dist_attr.process_mesh
        # if out_mesh.ndim > in_mesh.ndim + 1:
        #     return False
        # if out_mesh.ndim == in_mesh.ndim:
        #     return set(in_mesh.process_ids) < set(out_mesh.process_ids)
        # else:
        #     sub_meshes = paddle.base.libpaddle.pir.get_sub_meshes(in_mesh)
        #     return out_mesh in sub_meshes
        in_mesh = src_dist_attr.process_mesh
        out_mesh = dst_dist_attr.process_mesh
        sub_mesh_dim = paddle.base.core.sub_mesh_dim(out_mesh, in_mesh)
        if sub_mesh_dim == -1:
            return False
        sub_meshes, sub_placements = (
            dist.auto_parallel.api._get_sub_meshes_and_local_placements(
                out_mesh, dst_dist_attr.placements_attr, sub_mesh_dim
            )
        )
        if not check_placements_equal(
            src_dist_attr.placements_attr, sub_placements
        ):
            return False
        return True

    def reshard(self, src_dist_attr, dst_dist_attr, src_value, dst_type):
        src_mesh = src_dist_attr.process_mesh
        dst_mesh = dst_dist_attr.process_mesh

        sub_mesh_dim = paddle.base.core.sub_mesh_dim(dst_mesh, src_mesh)
        sub_meshes = split_mesh(dst_mesh, sub_mesh_dim)
        dst_meshes = [
            mesh
            for mesh in sub_meshes
            if not mesh_equal_ignore_shape_one(mesh, src_mesh, sub_mesh_dim)
        ]

        comm_group_ids = []
        root_ranks = []
        for p_id in src_mesh.process_ids:
            comm_group_ids.append([p_id])
            root_ranks.append(p_id)
        for i, group_ids in enumerate(comm_group_ids):
            for mesh in dst_meshes:
                group_ids.append(mesh.process_ids[i])

        other_ranks = copy.copy(dst_mesh.process_ids)
        for rank in other_ranks:
            if rank in src_mesh.process_ids:
                other_ranks.remove(rank)

        cur_rank = paddle.distributed.get_rank()

        if cur_rank in src_mesh.process_ids:
            # the root rank will broadcast the src_value to other ranks
            chunk_id = -1
            if src_value.get_defining_op().dist_attr:
                chunk_id = src_value.get_defining_op().dist_attr.chunk_id
            tmp_value = paddle._C_ops.share_data_(src_value)
            value_type = paddle.base.libpaddle.pir.cvt_to_dist_type(
                src_value.type(), src_value.dist_attr()
            )
            tmp_value.set_type(value_type)
            op = tmp_value.get_defining_op()
            op.dist_attr = paddle.base.libpaddle.pir.create_op_dist_attribute(
                src_mesh, [src_dist_attr], [src_dist_attr], chunk_id
            )
        elif cur_rank in other_ranks:
            # create the buffer on other ranks for receiving the data
            tmp_value = paddle.zeros(dst_type.shape, dst_type.dtype)
            op = tmp_value.get_defining_op()
            mesh = paddle.distributed.ProcessMesh(other_ranks)
            value_dist_attr = copy_dist_attr_with_new_member(
                dst_dist_attr, new_process_mesh=mesh
            )
            value_type = paddle.base.libpaddle.pir.cvt_to_dist_type(
                dst_type, value_dist_attr
            )
            tmp_value.set_type(value_type)
            op.dist_attr = paddle.base.libpaddle.pir.create_op_dist_attribute(
                mesh, [], [value_dist_attr]
            )
        else:
            # do nothing if the current rank is not in src_mesh and dst_mesh.
            # use reshard_op to create and return a fake value, and the fake
            # value will be removed 'remove_other_rank_op_pass'.
            fake_var = paddle._C_ops.reshard_v2(src_value, dst_dist_attr)
            return fake_var

        # create communication groups
        for i, group_ids in enumerate(comm_group_ids):
            comm_group_ids[i] = sorted(group_ids)
            # the root arg in broadcast is the local index
            # of the rank in the communication group
            root_ranks[i] = comm_group_ids[i].index(root_ranks[i])

        comm_groups = []
        for i, group_ids in enumerate(comm_group_ids):
            comm_groups.append(new_process_group(group_ids))
            if cur_rank in group_ids:
                cur_group_id = i

        broadcast_value = paddle._C_ops.broadcast(
            tmp_value, comm_groups[cur_group_id].id, root_ranks[cur_group_id]
        )
        broadcast_value.set_type(dst_type)

        broadcast_op = broadcast_value.get_defining_op()
        broadcast_op.dist_attr = (
            paddle.base.libpaddle.pir.create_op_dist_attribute(
                dst_mesh, [src_dist_attr], [dst_dist_attr]
            )
        )

        return broadcast_value
