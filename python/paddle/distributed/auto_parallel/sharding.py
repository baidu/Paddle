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
import operator
from collections import OrderedDict
from functools import reduce
from itertools import product

import numpy as np

import paddle
import paddle.distributed as dist
from paddle import core
from paddle.autograd import no_grad
from paddle.base.libpaddle import pir
from paddle.distributed import fleet
from paddle.distributed.auto_parallel.static.process_group import (
    new_process_group,
)
from paddle.distributed.auto_parallel.static.reshard_funcs.nd_mesh_reshard_func import (
    get_1D_sub_process_mesh,
)
from paddle.distributed.auto_parallel.static.utils import split_mesh
from paddle.distributed.fleet.meta_optimizers.common import OpRole
from paddle.distributed.fleet.utils.tensor_fusion_helper import (
    align,
    get_current_device_type,
)
from paddle.distributed.passes.pass_utils import AutoParallelStreamType
from paddle.framework import _current_expected_place_ as _get_device
from paddle.optimizer import Optimizer

from .moe_utils import _dtensor_from_local
from .static.reshard_funcs.base_reshard_func import copy_op_attr_with_new_member
from .strategy import Strategy


def get_placement_with_sharding(param, sharding_axis, param_placements=None):
    shard_axis = -1
    if param_placements is None:
        param_placements = param.placements

    for placement in param_placements:
        if isinstance(placement, dist.Shard):
            # the parameter can't be shard twice with sharding on different mesh now
            # for example, [Shard(0), Shard(1)], assert here in case
            assert (
                shard_axis == -1
            ), "The parameter can't be shard twice with sharding strategy even in different mesh now."
            shard_axis = placement.get_dim()

    placement_with_sharding = None
    for dim in range(param.ndim):
        if dim != shard_axis:
            placement_with_sharding = dist.Shard(dim)
            break

    new_placements = copy.deepcopy(param_placements)
    if placement_with_sharding is not None:
        new_placements[sharding_axis] = placement_with_sharding

    return new_placements


def get_mesh_comm_list(mesh, axis_name):
    assert axis_name in mesh.dim_names
    axis_index = mesh.dim_names.index(axis_name)

    ranges = []
    for dim_num in mesh._shape:
        ranges.append(range(dim_num))
    ranges[axis_index] = [0]

    all_result = []
    for x in product(*ranges):
        result = []
        for i in range(0, mesh.get_dim_size(axis_name)):
            coord = x[0:axis_index] + (i,) + x[axis_index + 1 :]
            result.append(mesh.mesh[coord])
        all_result.append(result)
    return all_result


class ShardingOptimizerStage1(Optimizer):
    """
    .. ZeRO: https://arxiv.org/abs/1910.02054

    """

    def __init__(self, optimizer, shard_fn=None, strategy=None):
        assert (
            optimizer is not None
        ), "The argument `optimizer` cannot be empty."
        assert isinstance(
            optimizer, (paddle.optimizer.AdamW, paddle.optimizer.SGD)
        ), "`paddle.distributed.ShardOptimizer` only supports AdamW and SGD optimizer for now."
        self.__dict__["_inner_opt"] = optimizer
        self._shard_fn = shard_fn
        self._strategy = strategy or Strategy()
        self._slice_param_group_info = []
        self._dy_shard_group = None

        paddle.enable_static()
        if self._shard_fn._mesh is None:
            mesh = dist.auto_parallel.get_mesh()
        else:
            mesh = self._shard_fn._mesh
        dp_groups = get_mesh_comm_list(mesh, "dp")
        for group in dp_groups:
            comm_group = new_process_group(sorted(group))
            if dist.get_rank() in group:
                self._sharding_group = comm_group

        self._mp_group = None
        if "mp" in mesh._dim_names:
            mp_groups = get_mesh_comm_list(mesh, "mp")
            for group in mp_groups:
                comm_group = new_process_group(sorted(group))
                if dist.get_rank() in group:
                    self._mp_group = comm_group

        self.pp_meshes = set()
        if "pp" in mesh.dim_names:
            pp_rank = mesh.get_rank_by_dim_and_process_id("pp", dist.get_rank())
            for idx in range(0, mesh.get_dim_size("pp")):
                self.pp_meshes.add(mesh.get_mesh_with_dim("pp", index=idx))
            mesh = mesh.get_mesh_with_dim("pp", index=pp_rank)
        else:
            self.pp_meshes.add(mesh)

        self._sharding_axis = mesh._dim_names.index("dp")

        self._sharding_degree = mesh._shape[self._sharding_axis]
        self._mp_mesh_axis = -1
        self._mp_degree = 1
        if "mp" in mesh._dim_names:
            self._mp_mesh_axis = mesh._dim_names.index("mp")
            self._mp_degree = mesh._shape[self._mp_mesh_axis]
            pp_meshes = set()
            for pp_mesh in self.pp_meshes:
                pp_meshes.add(pp_mesh)
                for sub_pp_mesh in split_mesh(
                    global_mesh=pp_mesh, sub_mesh_dim=self._mp_mesh_axis
                ):
                    pp_meshes.add(sub_pp_mesh)
            self.pp_meshes = pp_meshes

        paddle.disable_static()

    def apply_gradients(self, params_grads):
        place = _get_device()
        if isinstance(place, paddle.framework.CUDAPlace):
            place = paddle.framework.CUDAPlace(
                paddle.distributed.ParallelEnv().dev_id
            )
        self._place = paddle.base.libpaddle.Place()
        self._place.set_place(place)

        comm_buffer_size_MB = self._strategy.sharding.comm_buffer_size_MB
        if comm_buffer_size_MB < 0:
            comm_buffer_size_MB = 256

        parameters_dict = {}
        grads_dict = {}
        has_dist_param = False
        has_not_dist_param = False
        new_params_grads = []
        for param, grad in params_grads:
            if grad is None:
                continue
            param_dist_attr = param.dist_attr()
            grad_dist_attr = grad.dist_attr()
            assert (
                param_dist_attr is not None
            ), f"parameter dist attribute must not None. but received {param.name} : {param}."
            assert (
                grad_dist_attr is not None
            ), f"gradient dist attribute must not None. but received {param.name} grad : {grad}."
            assert (
                param_dist_attr.process_mesh == grad_dist_attr.process_mesh
            ), f"Parameter and grad should have same process_mesh. but received name:{param.name}, parameter:{param}, grad: {grad}."

            if self._sharding_axis not in grad_dist_attr.partial_dims:
                new_params_grads.append((param, grad))
                if param.optimize_attr is None:
                    param.optimize_attr = {'no_fusion': True}
                else:
                    param.optimize_attr["no_fusion"] = True
                continue
            else:
                if param.optimize_attr is None:
                    param.optimize_attr = {'no_fusion': False}
                else:
                    param.optimize_attr["no_fusion"] = False

            assert (
                param_dist_attr.process_mesh in self.pp_meshes
            ), f"parameter mesh mush be in pp_meshes. but received parameter name:{param.name}, mesh:{param_dist_attr.process_mesh}, pp_meshes: {self.pp_meshes}."

            if dist.get_rank() in param_dist_attr.process_mesh.process_ids:
                sub_mesh = get_1D_sub_process_mesh(
                    param_dist_attr.process_mesh, self._sharding_axis
                )
                assert (
                    sorted(sub_mesh.process_ids) == self._sharding_group.ranks
                ), f" all parameter must have the same sharding group. but received {param.name} sharding group is : {sub_mesh.process_ids}, global sharding group is: {self._sharding_group.ranks}"

            assert (
                param_dist_attr.partial_dims == set()
            ), f"Sharding fusion do not support partial parameter. but received {param.name} : {param}."
            assert (
                param_dist_attr.dims_mapping == grad_dist_attr.dims_mapping
            ), f"Parameter and grad should have same dims_mapping. but received name:{param.name}, parameter:{param}, grad: {grad}."
            assert (
                param.shape == grad.shape
            ), f"Parameter and grad should have same global shape. but received name:{param.name}, parameter:{param}, grad: {grad}."
            assert (
                param._local_shape == grad._local_shape
            ), f"Parameter and grad should have same local shape. but received name:{param.name}, parameter:{param}, grad: {grad}."

            if (
                self._mp_degree > 1
                and self._mp_mesh_axis in param_dist_attr.dims_mapping
            ):
                param.is_distributed = True
                has_dist_param = True

            else:
                param.is_distributed = False
                has_not_dist_param = True
            parameters_dict.setdefault(param_dist_attr.process_mesh, []).append(
                param
            )
            grads_dict.setdefault(param_dist_attr.process_mesh, []).append(grad)

        main_program = paddle.static.default_main_program()
        target_block = main_program.global_block()
        last_op = target_block.ops[-1]

        group_size = comm_buffer_size_MB * 1024 * 1024
        all_gather_param_info_list = []
        for mesh, parameters in parameters_dict.items():
            grads = grads_dict[mesh]
            var_groups = OrderedDict()
            group_indices = pir.assign_value_group_by_size(
                parameters, [group_size, group_size]
            )

            if dist.get_rank() in mesh.process_ids:
                self._cache_slice_param_group_info(parameters, group_indices)

            for group_idx, indices in enumerate(group_indices):
                group_param_list = []
                group_grad_list = []
                for index in indices:
                    var_groups.setdefault(group_idx, []).append(
                        parameters[index].name
                    )
                    group_param_list.append(parameters[index])
                    group_grad_list.append(grads[index])

                if self._strategy.sharding.enable_overlap:
                    self._reduce_scatter_overlap(group_grad_list, target_block)

                (
                    slice_param_dict,
                    padded_size_dict,
                    main_shard_fused_param,
                    main_fused_param,
                ) = self._fuse_group_param(group_idx, group_param_list)

                dtype = group_grad_list[0].dtype
                align_size = (
                    fleet.utils.tensor_fusion_helper.alignment[
                        get_current_device_type()
                    ]
                    // align[group_param_list[0].dtype]
                )
                align_size = (
                    align_size
                    * self._sharding_degree
                    * core.size_of_dtype(dtype)
                    // core.size_of_dtype(group_param_list[0].dtype)
                )

                if dist.get_rank() in mesh.process_ids:
                    self._cache_slice_param_range_and_size(
                        group_idx,
                        slice_param_dict,
                        padded_size_dict,
                        align_size,
                    )

                if not self._strategy.sharding.release_gradients:
                    _, fused_grad = paddle._C_ops.coalesce_tensor_(
                        group_grad_list,
                        dtype,
                        True,
                        False,
                        False,
                        0.0,
                        True,
                        align_size,
                        -1,
                        [],
                        [],
                    )
                    if not self._strategy.pipeline.enable:
                        for grad in group_grad_list:
                            grad.persistable = True
                    fused_grad.persistable = True
                    fused_type = paddle.pir.create_shaped_type(
                        fused_grad.type(), main_fused_param._local_shape
                    )
                    fused_grad.set_type(
                        pir.cvt_to_dist_type(fused_type, fused_grad.dist_attr())
                    )
                else:
                    first_grad_op = None
                    first_index = None
                    for grad in group_grad_list:
                        grad_op = grad.get_defining_op()
                        index = target_block.ops.index(grad_op)
                        if first_index is None or index < first_index:
                            first_index = index
                            first_grad_op = grad_op
                    pir.set_insertion_point(first_grad_op)
                    fused_grad = paddle._C_ops.empty(
                        main_fused_param._local_shape,
                        dtype,
                        self._place,
                    )
                    dist_attr = pir.create_tensor_dist_attribute(mesh, [-1], {})
                    fused_grad.set_type(
                        pir.cvt_to_dist_type(fused_grad.type(), dist_attr)
                    )
                    prev_var = fused_grad.get_defining_op().operand_source(0)
                    prev_var.set_type(
                        pir.cvt_to_dist_type(prev_var.type(), dist_attr)
                    )

                    grad_begin = 0
                    for grad in group_grad_list:
                        grad_op = grad.get_defining_op()
                        size = np.prod(grad._local_shape)
                        pir.set_insertion_point(grad_op)
                        grad_buffer = paddle._C_ops.view_slice(
                            fused_grad, grad_begin, grad_begin + size
                        )
                        grad_buffer = paddle._C_ops.view_shape(
                            grad_buffer, grad._local_shape
                        )
                        pir.set_insertion_point_after(grad_op)
                        paddle._C_ops.share_var([grad, grad_buffer])
                        grad_begin += (
                            (
                                (
                                    size * core.size_of_dtype(dtype)
                                    + align_size
                                    - 1
                                )
                                // align_size
                            )
                            * align_size
                            // core.size_of_dtype(dtype)
                        )

                if not self._strategy.sharding.enable_overlap:
                    pir.reset_insertion_point_to_end()

                shard_size = fused_grad._local_shape[0] // self._sharding_degree
                rank = self._sharding_group.ranks.index(dist.get_rank())
                rank_begin = rank * shard_size
                rank_end = rank_begin + shard_size
                view_shard_fused_grad = paddle._C_ops.view_slice(
                    fused_grad, rank_begin, rank_end
                )

                shard_fused_grad = paddle._C_ops.reduce_scatter(
                    fused_grad, self._sharding_group.id, self._sharding_degree
                )

                if self._strategy.sharding.enable_overlap:
                    shard_fused_grad.get_defining_op().set_execution_stream(
                        AutoParallelStreamType.SHARDING_STREAM.value
                    )
                pir.reset_insertion_point_to_end()

                paddle._C_ops.share_var(
                    [view_shard_fused_grad, shard_fused_grad]
                )

                slice_param_list = []
                for slice_param, param_info in slice_param_dict.items():
                    slice_param_list.append(slice_param)

                all_gather_param_info_list.append(
                    (
                        slice_param_list,
                        main_shard_fused_param,
                        main_fused_param,
                    )
                )

                for slice_param, param_info in slice_param_dict.items():
                    index, param_begin, param_end = param_info
                    slice_grad = paddle._C_ops.view_slice(
                        shard_fused_grad, param_begin, param_end
                    )
                    partail_status = (
                        group_grad_list[index].dist_attr().partial_status
                    )
                    partail_status.pop(self._sharding_axis)
                    slice_grad_dist_attr = pir.create_tensor_dist_attribute(
                        slice_grad.process_mesh, [-1], partail_status
                    )
                    slice_grad.set_type(
                        pir.cvt_to_dist_type(
                            slice_grad.type(), slice_grad_dist_attr
                        )
                    )

                    slice_grad_out_dist_attr = (
                        slice_grad.get_defining_op().dist_attr.results()
                    )
                    slice_grad_out_dist_attr[0] = slice_grad_out_dist_attr[
                        0
                    ].as_tensor_dist_attr()
                    slice_grad_out_dist_attr[0] = (
                        pir.create_tensor_dist_attribute(
                            slice_grad.process_mesh, [-1], partail_status
                        )
                    )
                    slice_grad.get_defining_op().dist_attr = (
                        copy_op_attr_with_new_member(
                            slice_grad.get_defining_op().dist_attr,
                            new_results=slice_grad_out_dist_attr,
                        )
                    )

                    new_params_grads.append((slice_param, slice_grad))

        if self._inner_opt._grad_clip is not None:
            self._inner_opt._grad_clip.should_comm_on_shard_dim = True
            self._inner_opt._grad_clip.sharding_group = self._sharding_group
            self._inner_opt._grad_clip.mp_group = self._mp_group
            self._inner_opt._grad_clip.has_dist_param = has_dist_param
            self._inner_opt._grad_clip.has_not_dist_param = has_not_dist_param
        self._inner_opt.apply_gradients(new_params_grads)

        pir.reset_insertion_point_to_end()
        for (
            slice_param_list,
            shard_param,
            fused_param,
        ) in all_gather_param_info_list:
            if self._strategy.sharding.enable_overlap:
                last_idx = None
                last_op = None
                for op in slice_param_list[-1].all_used_ops():
                    idx = target_block.ops.index(op)
                    if last_idx is None or idx > last_idx:
                        last_idx = idx
                        last_op = op

                # NOTE: add dependency between opt op and allgather_value for correctness
                tmp = paddle._C_ops.nop(last_op.results()[0])
                tmp.get_defining_op().set_execution_stream(
                    AutoParallelStreamType.SHARDING_STREAM.value
                )

                allgather_value = paddle._C_ops.all_gather(
                    shard_param, self._sharding_group.id, self._sharding_degree
                )
                allgather_value.get_defining_op().set_execution_stream(
                    AutoParallelStreamType.SHARDING_STREAM.value
                )
            else:
                allgather_value = paddle._C_ops.all_gather(
                    shard_param, self._sharding_group.id, self._sharding_degree
                )
            paddle._C_ops.share_var([fused_param, allgather_value])

        start_index = target_block.ops.index(last_op) + 1
        return target_block.ops[start_index:]

    def _cache_slice_param_group_info(self, parameters, group_indices):
        self._slice_param_group_info = [{} for _ in range(len(group_indices))]
        for group_idx, indices in enumerate(group_indices):
            for index in indices:
                param = parameters[index]
                self._slice_param_group_info[group_idx][param.name] = {}
                self._slice_param_group_info[group_idx][param.name][
                    "shape"
                ] = param.shape
                self._slice_param_group_info[group_idx][param.name][
                    "param_start"
                ] = -1
                self._slice_param_group_info[group_idx][param.name][
                    "param_end"
                ] = -1
                self._slice_param_group_info[group_idx][param.name][
                    "placements"
                ] = param.placements
                self._slice_param_group_info[group_idx][param.name][
                    "process_mesh"
                ] = param.process_mesh

    def _cache_slice_param_range_and_size(
        self, group_idx, slice_param_dict, padded_size_dict, align_size
    ):
        for slice_param, param_info in slice_param_dict.items():
            slice_param_name = slice_param.name.replace("slice@", "")
            _, param_begin, param_end = param_info
            self._slice_param_group_info[group_idx][slice_param_name][
                "param_start"
            ] = param_begin
            self._slice_param_group_info[group_idx][slice_param_name][
                "param_end"
            ] = param_end

        for name, padded_size in padded_size_dict.items():
            self._slice_param_group_info[group_idx][name][
                "padded_size"
            ] = padded_size

        for name, _ in self._slice_param_group_info[group_idx].items():
            self._slice_param_group_info[group_idx][name][
                "align_size"
            ] = align_size

    def _reduce_scatter_overlap(self, group_grad_list, target_block):
        '''
        In order to overlap computation and reduce_scatter communication, we need to:
          a. place reduce_scatter in communication stream
          b. place reduce_scatter op and its producer ops after the last grad define op
        This function will complete the item b.
        '''
        insertion_info = {"idx": None, "op": None}
        # 1. move ops after the grad op
        for grad in group_grad_list:
            stack = [grad.get_defining_op()]
            grad_op = None
            advance_ops = []
            # 1.1 get the grad define op
            while len(stack) > 0:
                op = stack.pop()
                if op.op_role == int(OpRole.Backward):
                    grad_op = op
                    break
                if op.num_operands() == 1:  # only one operand
                    stack.append(op.operand_source(0).get_defining_op())
                    if op.op_role != int(OpRole.Backward):
                        advance_ops.append(op)
                else:
                    break
            if grad_op is not None:
                new_idx = target_block.ops.index(grad_op) + 1
                # 1.2 move ops
                for op in advance_ops:
                    old_idx = target_block.ops.index(op)
                    if new_idx != old_idx:
                        target_block.move_op(op, new_idx)
                # 2.1 get insertion point
                if (
                    insertion_info["idx"] is None
                    or new_idx > insertion_info["idx"]
                ):
                    insertion_info["idx"] = new_idx
                    if len(advance_ops) > 0:
                        insertion_info["op"] = advance_ops[-1]
                    else:
                        insertion_info["op"] = grad_op

        # 2.2 set insertion point
        if insertion_info["op"] is not None:
            pir.set_insertion_point_after(insertion_info["op"])

    def _fuse_group_param(self, group_index, group_param_list):
        startup_program = paddle.static.default_startup_program()
        main_program = paddle.static.default_main_program()
        with paddle.static.program_guard(startup_program):

            def get_param_from_startup(startup, name):
                for op in startup.global_block().ops:
                    if (
                        op.name() == 'builtin.set_parameter'
                        and name == op.attrs()['parameter_name']
                    ):
                        return op.operand(0).source()
                raise ValueError(
                    f"can't find param ({name}) in startup program"
                )

            startup_param_list = []
            fuse_param_name = "fused@"
            for param in group_param_list:
                startup_param = get_param_from_startup(
                    startup_program, param.name
                )
                startup_param_list.append(startup_param)
                fuse_param_name = fuse_param_name + "-" + param.name
            dtype = startup_param_list[0].dtype
            align_size = (
                fleet.utils.tensor_fusion_helper.alignment[
                    get_current_device_type()
                ]
                // align[dtype]
            )
            align_size = align_size * self._sharding_degree
            _, fused_param = paddle._C_ops.coalesce_tensor_(
                startup_param_list,
                dtype,
                True,
                False,
                False,
                0.0,
                True,
                align_size,
                -1,
                [],
                [],
            )

            group_size = 0
            for param in group_param_list:
                size = np.prod(param._local_shape) * core.size_of_dtype(dtype)
                padded_size = (
                    ((size + align_size - 1) // align_size)
                    * align_size
                    // core.size_of_dtype(dtype)
                )
                group_size += padded_size
            fused_type = paddle.pir.create_shaped_type(
                fused_param.type(), [group_size]
            )
            dist_attr = pir.create_tensor_dist_attribute(
                group_param_list[0].process_mesh, [-1], {}
            )
            fused_param.set_type(pir.cvt_to_dist_type(fused_type, dist_attr))
            fused_param.persistable = True
            paddle._pir_ops.set_persistable_value(fused_param, fuse_param_name)
            main_fused_param = main_program.global_block().add_kwarg(
                fuse_param_name, fused_param.type()
            )
            main_fused_param.place_attr = self._place
            main_fused_param.persistable = True

            shard_size = group_size // self._sharding_degree
            rank = self._sharding_group.ranks.index(dist.get_rank())
            rank_begin = rank * shard_size
            shard_fused_param = paddle._C_ops.view_slice(
                fused_param, rank_begin, rank_begin + shard_size
            )
            shard_fused_param.persistable = True
            paddle._pir_ops.set_persistable_value(
                shard_fused_param, "shard@" + fuse_param_name
            )
            main_shard_fused_param = main_program.global_block().add_kwarg(
                "shard@" + fuse_param_name, shard_fused_param.type()
            )
            main_shard_fused_param.place_attr = self._place
            main_shard_fused_param.persistable = True
            total_buffer_size = 0
            slice_param_dict = {}
            padded_size_dict = {}

            for index, param in enumerate(group_param_list):
                size = np.prod(param._local_shape) * core.size_of_dtype(dtype)
                padded_size = (
                    ((size + align_size - 1) // align_size)
                    * align_size
                    // core.size_of_dtype(dtype)
                )
                padded_size_dict[param.name] = padded_size

                param_begin = max(total_buffer_size - rank_begin, 0)
                total_buffer_size += padded_size
                param_end = min(total_buffer_size - rank_begin, shard_size)
                if param_begin < param_end:
                    init_slice_param = paddle._C_ops.view_slice(
                        shard_fused_param, param_begin, param_end
                    )
                    init_slice_param.persistable = True
                    slice_param_name = "slice@" + param.name
                    paddle._pir_ops.set_parameter(
                        init_slice_param, slice_param_name
                    )
                    main_program.set_parameters_from(startup_program)
                    with paddle.static.program_guard(main_program):
                        pir.reset_insertion_point_to_start()
                        slice_param = paddle._pir_ops.parameter(
                            slice_param_name
                        )
                        slice_param.persistable = True
                        slice_param.set_type(init_slice_param.type())
                        slice_param.trainable = param.trainable
                        slice_param.stop_gradient = param.stop_gradient
                        slice_param.optimize_attr = param.optimize_attr
                        slice_param.regularizer = param.regularizer
                        slice_param.do_model_average = param.do_model_average
                        slice_param.need_clip = param.need_clip
                        slice_param.is_distributed = param.is_distributed
                        slice_param.is_parameter = param.is_parameter

                    slice_param_dict[slice_param] = (
                        index,
                        param_begin,
                        param_end,
                    )
        return (
            slice_param_dict,
            padded_size_dict,
            main_shard_fused_param,
            main_fused_param,
        )

    def _apply_optimize(
        self, loss, startup_program, params_grads, param_group_idx=0
    ):
        return self.apply_gradients(params_grads)

    def __getattr__(self, item):
        if "_inner_opt" in self.__dict__:
            if item == "_inner_opt":
                return self.__dict__[item]
            return getattr(self.__dict__["_inner_opt"], item)
        else:
            raise AttributeError

    def __setattr__(self, item, value):
        if item == '_inner_opt':
            msg = f'{type(self).__name__}._inner_opt is READ ONLY'
            raise AttributeError(msg)
        return setattr(self._inner_opt, item, value)

    @no_grad()
    def convert_state_dict_without_tensor_fusion_param(self, state_dict):
        master_opt_param_names = []
        moment_opt_param_names = []
        pow_acc_opt_param_names = []
        slice_param_names = []

        for name, tensor in state_dict.items():
            if not tensor.is_dist():
                continue
            if "slice@" not in name:
                continue

            if "_moment" in name:
                moment_opt_param_names.append(name)
            elif "_pow_acc" in name:
                pow_acc_opt_param_names.append(name)
            elif "_master" in name:
                master_opt_param_names.append(name)
            else:
                slice_param_names.append(name)

        # slice@ parameters share the same memory with the original parameters
        # when the model is saved, we no need to save the slice@ parameters
        for name in slice_param_names:
            del state_dict[name]
        paddle.device.cuda.empty_cache()

        if self._dy_shard_group is None:
            self._create_dy_sharding_group()

        for group_info in self._slice_param_group_info:
            self._all_gather_master_opt_params(
                state_dict, group_info, master_opt_param_names
            )
            self._all_gather_moment_opt_params(
                state_dict, group_info, moment_opt_param_names
            )
            # The pow_acc parameter is a scalar and doesn't require
            # sharding, so it is simply broadcast to all devices.
            self._broadcast_pow_acc_opt_params(
                state_dict, group_info, pow_acc_opt_param_names
            )
            paddle.device.cuda.empty_cache()

    def _create_dy_sharding_group(self):
        mesh = self._shard_fn._mesh
        if mesh is None:
            mesh = dist.auto_parallel.get_mesh()

        shard_groups = get_mesh_comm_list(mesh, "dp")
        for group in shard_groups:
            comm_group = dist.new_group(sorted(group))
            if dist.get_rank() in group:
                self._dy_shard_group = comm_group

    @no_grad()
    def convert_state_dict_with_tensor_fusion_param(self, state_dict):
        moment_suffixs = []
        pow_acc_suffixs = []
        master_suffixs = []

        for name in state_dict.keys():
            if "_moment" in name:
                moment_suffixs.append(name.split(".dist")[-1])
            elif "_pow_acc" in name:
                pow_acc_suffixs.append(name.split(".dist")[-1])
            elif "_master" in name:
                master_suffixs.append(name.split(".dist")[-1])

        moment_suffixs = sorted(set(moment_suffixs))
        pow_acc_suffixs = sorted(set(pow_acc_suffixs))
        master_suffixs = sorted(set(master_suffixs))

        if self._dy_shard_group is None:
            self._create_dy_sharding_group()

        for group_info in self._slice_param_group_info:
            group_size = 0
            for param_name, param_info in group_info.items():
                group_size = max(group_size, param_info["param_end"])

            bucket_info = self._bucket_tensors_with_group_size(
                group_info, group_size
            )

            self._re_slicing_opt_param(
                state_dict, group_info, bucket_info, master_suffixs
            )
            self._re_slicing_opt_param(
                state_dict, group_info, bucket_info, moment_suffixs
            )
            self._remove_pow_acc_opt_params(
                state_dict, group_info, bucket_info, pow_acc_suffixs
            )

    def _remove_pow_acc_opt_params(
        self, state_dict, group_info, bucket_info, pow_acc_suffixs
    ):
        group_rank_mapping, size_mapping = bucket_info
        cur_rank = self._sharding_group.ranks.index(dist.get_rank())

        for idx, (param_name, param_info) in enumerate(group_info.items()):
            for pow_acc_suffix in pow_acc_suffixs:
                if cur_rank in group_rank_mapping[idx]:
                    state_dict["slice@" + param_name + pow_acc_suffix] = (
                        state_dict[param_name + pow_acc_suffix]
                    )
                del state_dict[param_name + pow_acc_suffix]

    def _re_slicing_opt_param(
        self, state_dict, group_info, bucket_info, param_suffixs
    ):
        group_rank_mapping, size_mapping = bucket_info
        cur_rank = self._sharding_group.ranks.index(dist.get_rank())

        for param_suffix in param_suffixs:
            # Step1: Gather the optimizer parameters across sharding groups
            opt_param_list = []
            for idx, (param_name, param_info) in enumerate(group_info.items()):
                opt_param = state_dict[param_name + param_suffix]
                param_list = []
                dist.all_gather(
                    param_list,
                    opt_param._local_value().contiguous(),
                    group=self._dy_shard_group,
                )
                param_sharding_axis = opt_param.placements[
                    self._sharding_axis
                ].get_dim()
                global_opt_param = paddle.concat(
                    param_list, axis=param_sharding_axis
                )
                global_opt_param = global_opt_param.view([-1])
                opt_param_list.append(global_opt_param)

                # process gaps generated by coalesce_tensor
                if global_opt_param.shape[0] < param_info["padded_size"]:
                    opt_param_list.append(
                        paddle.zeros(
                            [
                                param_info["padded_size"]
                                - global_opt_param.shape[0]
                            ],
                            dtype=global_opt_param.dtype,
                        )
                    )

                del param_list, global_opt_param
                paddle.device.cuda.empty_cache()

            # Step2: Fuse the optimizer parameters using coalesce_tensor
            fused_opt_param = paddle.concat(opt_param_list, axis=0)

            del opt_param_list
            paddle.device.cuda.empty_cache()

            # Step3: Slice the current rank's optimizer parameters
            param_index = 0
            for idx, (param_name, param_info) in enumerate(group_info.items()):
                if cur_rank in group_rank_mapping[idx]:
                    # param tensor may be sliced into multiple devices
                    # we need calculate the start index of the current rank
                    opt_param = state_dict[param_name + param_suffix]
                    cur_rank_start_index = param_index
                    for i, rank_id in enumerate(group_rank_mapping[idx]):
                        if rank_id == cur_rank:
                            break
                        cur_rank_start_index += size_mapping[idx][i]

                    shard_opt_param = fused_opt_param[
                        cur_rank_start_index : cur_rank_start_index
                        + param_info["param_end"]
                        - param_info["param_start"]
                    ]
                    shard_opt_param_placements = [
                        dist.Replicate()
                        for _ in range(len(opt_param.process_mesh.shape))
                    ]
                    shard_opt_param = _dtensor_from_local(
                        shard_opt_param,
                        opt_param.process_mesh,
                        shard_opt_param_placements,
                    )
                    state_dict["slice@" + param_name + param_suffix] = (
                        shard_opt_param
                    )

                param_index += param_info["padded_size"]
                del state_dict[param_name + param_suffix]
                paddle.device.cuda.empty_cache()

            # release memory
            del fused_opt_param
            paddle.device.cuda.empty_cache()

    @no_grad()
    def _all_gather_opt_params(
        self, state_dict, group_info, opt_param_names, opt_suffix
    ):
        # Retrieve the optimizer parameters for the current device.
        opt_param_list = []
        for param_name, param_info in group_info.items():
            opt_param_name = "slice@" + param_name + opt_suffix
            if opt_param_name not in state_dict:
                continue
            if opt_param_name not in opt_param_names:
                continue
            opt_param_list.append(
                state_dict[opt_param_name]._local_value().clone()
            )

        if len(opt_param_list) == 0:
            return

        fused_opt_param = paddle.concat(opt_param_list, axis=0)
        fused_opt_param_list = []
        # All-gather the optimizer parameters across sharding groups
        dist.all_gather(
            fused_opt_param_list, fused_opt_param, group=self._dy_shard_group
        )

        fused_opt_param_list = [item.cpu() for item in fused_opt_param_list]
        fused_opt_param = paddle.concat(fused_opt_param_list, axis=0)

        for param_name, param_info in group_info.items():
            opt_param_name = "slice@" + param_name + opt_suffix
            if opt_param_name not in state_dict:
                continue
            if opt_param_name not in opt_param_names:
                continue

            local_tensor = state_dict[opt_param_name]._local_value()
            del state_dict[opt_param_name]

        paddle.device.cuda.empty_cache()

        param_index = 0
        for param_name, param_info in group_info.items():
            opt_param_name = "slice@" + param_name + opt_suffix

            global_shape = copy.deepcopy(param_info["shape"])
            if self._mp_group is not None:
                mp_placement = param_info["placements"][self._mp_mesh_axis]
                if isinstance(mp_placement, dist.Shard):
                    param_tensor_parallel_axis = mp_placement.get_dim()
                    global_shape[param_tensor_parallel_axis] /= self._mp_degree
                    global_shape[param_tensor_parallel_axis] = int(
                        global_shape[param_tensor_parallel_axis]
                    )

            global_size = reduce(operator.mul, global_shape, 1)
            # retrieve the global parameters.
            global_param = fused_opt_param[
                param_index : param_index + global_size
            ]

            shard_opt_param = global_param.reshape(global_shape)

            opt_param_mesh = param_info["process_mesh"]
            opt_param_placements = get_placement_with_sharding(
                shard_opt_param, self._sharding_axis, param_info["placements"]
            )

            # slice the global parameter into local parameter based on the sharding axis
            shard_index = [slice(None)] * len(shard_opt_param.shape)
            rank = self._sharding_group.ranks.index(dist.get_rank())
            param_sharding_axis = opt_param_placements[
                self._sharding_axis
            ].get_dim()

            shard_slice_start_idx = (
                rank / self._sharding_degree
            ) * shard_opt_param.shape[param_sharding_axis]
            shard_slice_end_idx = (
                shard_slice_start_idx
                + shard_opt_param.shape[param_sharding_axis]
                / self._sharding_degree
            )

            shard_slice = slice(
                int(shard_slice_start_idx), int(shard_slice_end_idx)
            )
            shard_index[param_sharding_axis] = shard_slice
            shard_opt_param = shard_opt_param[tuple(shard_index)]

            shard_opt_param = _dtensor_from_local(
                shard_opt_param.cuda(),
                opt_param_mesh,
                opt_param_placements,
                shard_opt_param.shape,
            )

            state_dict[param_name + opt_suffix] = shard_opt_param
            padded_size = param_info["padded_size"]
            param_index += padded_size

        paddle.device.cuda.empty_cache()

    def _all_gather_moment_opt_params(
        self, state_dict, group_info, moment_opt_param_names
    ):
        if len(moment_opt_param_names) == 0:
            return

        moments = {}
        for name in moment_opt_param_names:
            moment_suffix = name.split(".dist")[-1]
            if moment_suffix not in moments:
                moments[moment_suffix] = []
            moments[moment_suffix].append(name)

        moments = dict(sorted(moments.items()))
        for moment_suffix, moment_names in moments.items():
            self._all_gather_opt_params(
                state_dict, group_info, moment_names, moment_suffix
            )

    def _all_gather_master_opt_params(
        self, state_dict, group_info, master_opt_param_names
    ):
        if len(master_opt_param_names) == 0:
            return

        master_suffix = master_opt_param_names[0].split(".dist")[-1]
        self._all_gather_opt_params(
            state_dict,
            group_info,
            master_opt_param_names,
            master_suffix,
        )

    def _broadcast_pow_acc_opt_params(
        self, state_dict, group_info, pow_acc_opt_param_names
    ):
        if len(pow_acc_opt_param_names) == 0:
            return

        pow_acc_suffixs = []
        for name in pow_acc_opt_param_names:
            pow_acc_suffix = name.split(".dist")[-1]
            pow_acc_suffixs.append(pow_acc_suffix)
        pow_acc_suffixs = sorted(set(pow_acc_suffixs))

        group_size = 0
        for param_name, param_info in group_info.items():
            group_size = max(group_size, param_info["param_end"])

        # Bucket the parameters according to the group size, with the
        # number of buckets equal to the size of the sharding group.
        group_rank_mapping, _ = self._bucket_tensors_with_group_size(
            group_info, group_size
        )
        cur_rank = self._sharding_group.ranks.index(dist.get_rank())

        for idx, (param_name, param_info) in enumerate(group_info.items()):
            root_rank = group_rank_mapping[idx][0]
            for pow_acc_suffix in pow_acc_suffixs:
                pow_acc_name = "slice@" + param_name + pow_acc_suffix
                if cur_rank == root_rank:
                    pow_acc_tensor = state_dict[pow_acc_name]
                    pow_acc_local_tensor = pow_acc_tensor._local_value()
                    dist.broadcast(
                        pow_acc_local_tensor,
                        src=self._sharding_group.ranks[root_rank],
                        group=self._dy_shard_group,
                    )
                    state_dict[param_name + pow_acc_suffix] = pow_acc_tensor
                    state_dict.pop(pow_acc_name)
                else:
                    tmp_mesh = param_info["process_mesh"]
                    tmp_placements = [
                        dist.Replicate() for _ in range(len(tmp_mesh.shape))
                    ]
                    tmp_data = paddle.zeros([1])

                    dist.broadcast(
                        tmp_data,
                        src=self._sharding_group.ranks[root_rank],
                        group=self._dy_shard_group,
                    )
                    pow_acc_tensor = _dtensor_from_local(
                        tmp_data, tmp_mesh, tmp_placements
                    )
                    state_dict[param_name + pow_acc_suffix] = pow_acc_tensor

    def _bucket_tensors_with_group_size(self, group_info, group_size):
        group_mapping = [[] for _ in group_info]
        size_mapping = [[] for _ in group_info]
        current_size = 0
        current_bucket_index = 0

        for idx, param_info in enumerate(group_info.values()):
            tensor_size = param_info["padded_size"]

            while tensor_size > 0:
                available_space = group_size - current_size

                if tensor_size <= available_space:
                    group_mapping[idx].append(current_bucket_index)
                    size_mapping[idx].append(tensor_size)
                    current_size += tensor_size
                    tensor_size = 0
                else:
                    # tensor will be split into two buckets
                    if available_space > 0:
                        group_mapping[idx].append(current_bucket_index)
                        size_mapping[idx].append(available_space)
                        tensor_size -= available_space
                        current_size += available_space

                    current_bucket_index += 1
                    current_size = 0

        return group_mapping, size_mapping

    def convert_state_dict_with_rank_unique_name(self, state_dict):
        cur_rank = dist.get_rank()
        tensor_names = list(state_dict.keys())

        for name in tensor_names:
            tensor = state_dict[name]
            if not tensor.is_dist():
                continue
            if "slice@" not in name:
                continue

            if "_moment" in name or "_pow_acc" in name or "_master" in name:
                rank_name = f"{name}_rank{cur_rank}"
                state_dict[rank_name] = state_dict[name]

            del state_dict[name]

    def convert_state_dict_with_origin_name(self, state_dict):
        tensor_names = list(state_dict.keys())
        for name in list(state_dict.keys()):
            if "_rank" in name:
                no_rank_name = name.split("_rank")[0]
                state_dict[no_rank_name] = state_dict[name]
                del state_dict[name]
