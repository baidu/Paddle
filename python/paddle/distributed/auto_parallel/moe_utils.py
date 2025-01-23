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


from __future__ import annotations

import copy
import os
from typing import TYPE_CHECKING

import numpy as np

import paddle
import paddle.distributed as dist
from paddle import Tensor
from paddle.autograd import PyLayer

from .placement_type import check_placements_equal, to_dim_map
from .static.reshard_funcs.base_reshard_func import choose_reshard_func
from .static.reshard_funcs.nd_mesh_reshard_func import get_1D_sub_process_mesh

if TYPE_CHECKING:
    from paddle.distributed import Placement
    from paddle.distributed.auto_parallel.process_mesh import ProcessMesh


def _specific_alltoall_dim(
    dist_tensor: Tensor, mesh: ProcessMesh, placements: list[Placement]
):
    """
    Get the specific dimension for alltoall communication in nd_mesh reshard.
    """
    if not os.getenv("FLAGS_enable_moe_utils") == "true":
        return None

    mesh_dim = None
    if paddle.in_dynamic_mode():
        src_mesh = dist_tensor.process_mesh
        src_placements = dist_tensor.placements
    elif paddle.framework.in_pir_mode():
        src_mesh = dist_tensor.process_mesh
        src_placements = dist_tensor.dist_attr().placements_attr

    if src_mesh != mesh or src_mesh.ndim == 1:
        return None

    if any(p.is_partial() for p in src_placements):
        return None
    if any(p.is_partial() for p in placements):
        return None

    for i in range(min(len(src_placements), len(placements))):
        src_p = src_placements[i]
        dst_p = placements[i]
        if src_p.is_shard() and dst_p.is_shard() and src_p != dst_p:
            # reshard from shard to shard, needs alltoall
            # now only supports reshard on one dimension
            src_dim = src_p.get_dim()
            dst_dim = dst_p.get_dim()
            if mesh_dim is not None or abs(src_dim - dst_dim) != 1:
                return None
            else:
                mesh_dim = i

    return mesh_dim


def _dtensor_from_local(
    local_tensor, mesh, placements, local_tensor_shape=None
):
    # assume the each rank has the same tensor shape for now, just use the local shape to calculate the global shape
    global_dims = list(local_tensor.shape)
    if local_tensor_shape is not None:
        global_dims = local_tensor_shape
    for idx, placement in enumerate(placements):
        if placement.is_shard():
            shard_dim = placement.get_dim()
            local_dim_size = global_dims[shard_dim]
            global_dims[shard_dim] = local_dim_size * mesh.shape[idx]

    if paddle.in_dynamic_mode():
        place = paddle.framework._current_expected_place()
        place = paddle.framework._get_paddle_place(place)

        return paddle.Tensor(
            local_tensor,
            dims=global_dims,
            process_mesh=mesh,
            placements=placements,
            place=place,
        )

    # TODO Adopt Mix2Dist Pass to allow the program could be executed actually.
    elif paddle.framework.in_pir_mode():
        assert isinstance(
            local_tensor, (type(None), paddle.pir.Value)
        ), "input tensor is not pir value."
        assert (
            local_tensor.is_dense_tensor_type()
        ), "dtensor_from_local() are only supported dense tensor type right."
        sharding_specs = (
            paddle.distributed.auto_parallel.placement_type.get_shard_spec(
                mesh, placements, local_tensor.ndim
            )
        )
        dims_mapping = paddle.distributed.auto_parallel.static.utils.convert_to_dims_mapping(
            sharding_specs, mesh
        )
        local_shape = local_tensor.shape
        global_tensor_type = paddle.pir.create_shaped_type(
            local_tensor.type(), global_dims
        )
        dist_dense_tensor_type = paddle.base.libpaddle.pir.create_dist_dense_tensor_type_by_dense_tensor(
            global_tensor_type, local_shape, mesh, dims_mapping
        )
        local_tensor.set_type(dist_dense_tensor_type)
        return local_tensor
    else:
        raise RuntimeError(
            "dtensor_from_local() are only supported in dynamic or pir mode."
        )


def _pir_nd_mesh_all2all(src_value, dst_type, mesh, placements, dim):
    """
    Use all to all communication in nd_mesh reshard.
    """
    # create value on sub 1D mesh
    sub_value = paddle._C_ops.share_data(src_value)
    sub_mesh = get_1D_sub_process_mesh(mesh, dim)
    sub_placements = [src_value.dist_attr().placements_attr[dim]]
    sub_value_shape = dist.auto_parallel.api._cal_global_shape(
        src_value._local_shape, sub_mesh, sub_placements
    )
    sub_value_type = paddle.pir.create_shaped_type(
        sub_value.type(), sub_value_shape
    )
    sub_dims_mapping, partial_status = to_dim_map(
        sub_placements, len(sub_value_shape)
    )
    sub_value_dist_attr = (
        paddle.base.libpaddle.pir.create_tensor_dist_attribute(
            sub_mesh, sub_dims_mapping, partial_status
        )
    )
    sub_value_dist_type = paddle.base.libpaddle.pir.cvt_to_dist_type(
        sub_value_type, sub_value_dist_attr
    )
    sub_value.set_type(sub_value_dist_type)

    # 1D mesh reshard
    dst_placements = [placements[dim]]
    sub_dst_dims_mapping, partial_status = to_dim_map(
        dst_placements, len(sub_value_shape)
    )
    sub_dst_type = paddle.pir.create_shaped_type(
        sub_value.type(), sub_value_shape
    )
    sub_dst_dist_attr = paddle.base.libpaddle.pir.create_tensor_dist_attribute(
        sub_mesh, sub_dst_dims_mapping, partial_status
    )
    sub_dst_type = paddle.base.libpaddle.pir.cvt_to_dist_type(
        sub_dst_type, sub_dst_dist_attr
    )
    reshard_func = choose_reshard_func(sub_value_dist_attr, sub_dst_dist_attr)
    out = reshard_func.reshard(
        sub_value_dist_attr, sub_dst_dist_attr, sub_value, sub_dst_type
    )

    # set the type of the output value with global mesh
    if out is not None:
        out.set_type(dst_type)

    return out


class _NdMeshAlltoAll(PyLayer):
    @staticmethod
    def forward(
        ctx,
        dist_tensor: Tensor,
        mesh: ProcessMesh,
        placements: list[Placement],
        dim: int,
    ):
        sub_mesh = get_1D_sub_process_mesh(mesh, dim)
        ctx.alltoall_dim = dim
        ctx.x_mesh = copy.deepcopy(dist_tensor.process_mesh)
        ctx.x_placements = copy.deepcopy(dist_tensor.placements)
        ctx.out_mesh = copy.deepcopy(mesh)
        ctx.out_placements = copy.deepcopy(placements)

        local_shape = _cal_local_shape(
            dist_tensor.shape, mesh, dist_tensor.placements
        )
        out = _dtensor_from_local(
            dist_tensor._local_value(),
            sub_mesh,
            [dist_tensor.placements[dim]],
            local_shape,
        )
        out = dist.reshard(out, sub_mesh, [placements[dim]])
        local_shape = _cal_local_shape(out.shape, sub_mesh, out.placements)
        out = _dtensor_from_local(
            out._local_value(), mesh, placements, local_shape
        )
        out.stop_gradient = dist_tensor.stop_gradient
        return out

    @staticmethod
    def backward(ctx, out_grad):
        if not check_placements_equal(ctx.out_placements, out_grad.placements):
            out = dist.reshard(out_grad, ctx.out_mesh, ctx.out_placements)
        out = _NdMeshAlltoAll.apply(
            out_grad, ctx.x_mesh, ctx.x_placements, ctx.alltoall_dim
        )
        return out


def _cal_local_shape(global_shape, mesh, placements):
    local_shape = list(global_shape)
    for idx, placement in enumerate(placements):
        if placement.is_shard():
            shard_dim = placement.get_dim()
            local_shape[shard_dim] = local_shape[shard_dim] // mesh.shape[idx]
    return local_shape


def infer_positive_shape(src_shape, tgt_shape):
    if isinstance(tgt_shape, (list, tuple)):
        ret_shape = np.array(tgt_shape)
    else:
        ret_shape = tgt_shape.copy()

    minus_one_idx = np.where(ret_shape == -1)[0]
    if minus_one_idx.size > 0:
        assert (
            minus_one_idx.size <= 1
        ), "At most one -1 is allowed in target shape."

        nelem = np.prod(src_shape)
        ret_shape[minus_one_idx[0]] = 1
        ret_shape[minus_one_idx[0]] = nelem // np.prod(ret_shape)

    return list(ret_shape)


class _local_reshape(PyLayer):
    @staticmethod
    def forward(
        ctx,
        dist_tensor: Tensor,
        global_shape: list,
        local_shape: list,
        mesh: ProcessMesh,
        placements: list[Placement],
    ):
        place = paddle.framework._current_expected_place()
        place = paddle.framework._get_paddle_place(place)

        if dist_tensor._local_value()._is_initialized():
            local_tensor = dist_tensor._local_value().clone()
        else:
            local_tensor = dist_tensor._local_value()
        ctx.x_global_shape = copy.deepcopy(dist_tensor.shape)
        ctx.x_local_shape = copy.deepcopy(local_tensor.shape)
        ctx.x_mesh = copy.deepcopy(dist_tensor.process_mesh)
        ctx.x_placements = copy.deepcopy(dist_tensor.placements)

        local_tensor = local_tensor.reshape(local_shape)
        out = paddle.Tensor(
            local_tensor,
            dims=global_shape,
            process_mesh=mesh,
            placements=placements,
            place=place,
        )
        out.stop_gradient = dist_tensor.stop_gradient
        return out

    @staticmethod
    def backward(ctx, out_grad):
        place = paddle.framework._current_expected_place()
        place = paddle.framework._get_paddle_place(place)

        if out_grad._local_value()._is_initialized():
            local_grad = out_grad._local_value().clone()
            x_local_shape = ctx.x_local_shape
        else:
            local_grad = out_grad._local_value()
            x_local_shape = [0]
        local_grad = local_grad.reshape(x_local_shape)
        ret = paddle.Tensor(
            local_grad,
            dims=ctx.x_global_shape,
            process_mesh=ctx.x_mesh,
            placements=ctx.x_placements,
            place=place,
        )
        return ret


def _dist_reshape(
    dist_tensor: Tensor,
    global_shape: list,
    mesh: ProcessMesh,
    placements: list[Placement],
):
    """
    Reshape the local tensors of the dist tensor on each rank,
    and manually set the process_mesh and placements of the output.
    """
    tgt_global_shape = infer_positive_shape(dist_tensor.shape, global_shape)
    tgt_local_shape = _cal_local_shape(tgt_global_shape, mesh, placements)
    if paddle.in_dynamic_mode():
        src_local_shape = dist_tensor._local_value().shape
        if not dist_tensor._local_value()._is_initialized():
            tgt_local_shape = dist_tensor._local_value().shape
    elif paddle.framework.in_pir_mode():
        # src_local_shape = dist_tensor._local_shape
        src_local_shape = _cal_local_shape(
            dist_tensor.shape,
            dist_tensor.dist_attr().process_mesh,
            dist_tensor.dist_attr().placements_attr,
        )
    else:
        raise NotImplementedError(
            "dist_reshape is only supported in dynamic and pir mode."
        )

    assert np.prod(tgt_local_shape) == np.prod(
        src_local_shape
    ), f"The local shapes {src_local_shape} and {tgt_local_shape} are mismatched."

    if paddle.in_dynamic_mode():
        return _local_reshape.apply(
            dist_tensor, tgt_global_shape, tgt_local_shape, mesh, placements
        )
    elif paddle.framework.in_pir_mode():
        return paddle._C_ops.dist_reshape(
            dist_tensor,
            dist_tensor.placements,
            tgt_global_shape,
            tgt_local_shape,
            mesh,
            placements,
        )


def _replace_dist_reshape_pass(dist_program):
    for op in dist_program.global_block().ops:
        if op.name() == "dist_op.dist_reshape":
            paddle.pir.set_insertion_point(op)
            in_var = op.operand_source(0)
            out = paddle._C_ops.reshape(in_var, op.result(0).shape)
            reshape_op = out.get_defining_op()
            dist_type = op.result(0).type().as_dist_type()
            out.set_type(dist_type.local_type())
            op.result(0).replace_all_uses_with(out)
            assert op.result(0).use_empty() is True
            op.erase()


def _only_reshard_mesh_shape(
    dist_tensor: Tensor, mesh: ProcessMesh, placements: list[Placement]
):
    if not os.getenv("FLAGS_enable_moe_utils") == "true":
        return False

    if paddle.in_dynamic_mode():
        src_placements = dist_tensor.placements
        src_mesh = dist_tensor.process_mesh
    elif paddle.framework.in_pir_mode():
        if hasattr(dist_tensor, "dist_attr"):
            src_placements = dist_tensor.dist_attr().placements_attr
            src_mesh = dist_tensor.dist_attr().process_mesh
        else:
            src_placements = dist_tensor.placements
            src_mesh = dist_tensor.process_mesh
    else:
        raise NotImplementedError(
            "_only_reshard_mesh_shape is only supported in dynamic and pir mode."
        )
    if src_mesh == mesh or src_mesh.process_ids != mesh.process_ids:
        return False

    # only the mesh shapes are different,
    # if the placements are all replicate or partial,
    # then we can reshard the mesh shapes
    if any(p.is_shard() for p in src_placements + placements):
        return False

    if src_placements[0].is_partial():
        for p in src_placements + placements:
            if p != src_placements[0]:
                return False

    return True


def _reshard_mesh_shape(
    dist_tensor: Tensor, mesh: ProcessMesh, placements: list[Placement]
):
    """
    Reshard the mesh shape of the dist tensor. This is used to only modify
    the mesh shape of the input, and its data is not changed.

    E.g.
      1. [0,1,2,3], [Replicate()]  --> [[0,1],[2,3]], [Replicate(), Replicate()]
      2. [[0,1],[2,3]] [Partial(),Partial()]  --> [0,1,2,3], [Partial()]
    """
    if paddle.in_dynamic_mode():
        src_placements = dist_tensor.placements
        src_mesh = dist_tensor.process_mesh
    elif paddle.framework.in_pir_mode():
        src_placements = dist_tensor.dist_attr().placements_attr
        src_mesh = dist_tensor.dist_attr().process_mesh
    else:
        raise NotImplementedError(
            "_only_reshard_mesh_shape is only supported in dynamic and pir mode."
        )

    dst_placements = [placements[0]] * mesh.ndim
    return _dist_reshape(dist_tensor, dist_tensor.shape, mesh, dst_placements)
