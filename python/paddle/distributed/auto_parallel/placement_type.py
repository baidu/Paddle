#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from typing import cast

import paddle
from paddle.base.core import Partial, Replicate, Shard


def to_placements(dim_map, mesh, partial_idx=[]):
    """
    convert dim_map to placements.

    Args:
        dim_map(List[int]): a list of integer that represents sharding on each tensor dimension.
        mesh(paddle.distributed.ProcessMesh): The `ProcessMesh` object describes the Cartesian topology of the used processes.
        partial_idx(List[int], Optional): a list of integer that represents the DTensor have pending sum on which device mesh dimension

    Returns:
        List[Placement]: a list contains some `paddle.distributed.Placement`.
    """
    if isinstance(mesh, paddle.base.libpaddle.ProcessMesh):
        shape = mesh.shape
    else:
        shape = mesh.mesh.shape
    placements = [Replicate() for _ in range(len(shape))]

    for s in partial_idx:
        placements[s] = Partial()

    for i, m in enumerate(dim_map):
        if m >= 0:
            p = placements[m]
            if p.is_shard():
                p = cast(Shard, p)
                raise Exception(
                    f"ProcessMesh dimension can not be mapped to two dimension of same tensor: {i} and {p.get_dim()}."
                )
            elif p.is_partial():
                raise Exception(
                    f"ProcessMesh dimension {m} can not be both shard and partial!"
                )
            placements[m] = Shard(i)

    return placements


def check_placements_equal(this, that):
    assert isinstance(this, list) and isinstance(that, list)
    small_placements = this if len(this) < len(that) else that
    large_placements = that if len(this) < len(that) else this
    for i in range(len(large_placements)):
        if i < len(small_placements):
            if small_placements[i] != large_placements[i]:
                return False
        else:
            if large_placements[i] != Replicate():
                return False
    return True


def to_dim_map(placements, tensor_dims):
    """
    convert placements to dim_map.

    Args:
        placements(List[Placement]): a list contains some `paddle.distributed.Placement`.
        tensor_dims(int): the dimension of dist_tensor.

    Returns:
        List[int]: a list of integer that represents sharding on each tensor dimension.
    """
    dim_map = [-1] * tensor_dims
    partial_status = {}
    for i, placement in enumerate(placements):
        if placement.is_shard():
            shard_dim = cast(Shard, placement).get_dim()
            # if dim_map[shard_dim] > -1:
            #     raise Exception(
            #         "Tensor dim {shard_dim} is already sharded on mesh dim {dim_map[shard_dim]}"
            #     )

            dim_map[shard_dim] = i
        if placement.is_partial():
            partial_status[i] = cast(Partial, placement).reduce_type()

    return dim_map, partial_status


def get_shard_spec(mesh, placements, tensor_dims):
    """to get shard_spec for construct DistAttr for static API."""
    dim_map, _ = to_dim_map(placements, tensor_dims)
    mesh_dim_names = mesh.dim_names
    shard_spec = [None] * len(dim_map)
    for i, d in enumerate(dim_map):
        if d > -1:
            shard_spec[i] = mesh_dim_names[d]

    return shard_spec
