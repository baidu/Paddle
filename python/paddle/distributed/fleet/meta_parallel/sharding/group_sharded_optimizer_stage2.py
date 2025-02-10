#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

# The file has been adapted from fairscale file:
# https://github.com/facebookresearch/fairscale/blob/main/fairscale/optim/oss.py
# Git commit hash: 8acbec718f3c70a6b9785470bb9e05cd84fc3f8e
# We retain the following license from the original files:

# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import logging
import warnings
from collections import OrderedDict

import paddle
import paddle.distributed as dist
from paddle.distributed import ParallelMode, fleet
from paddle.framework import core
from paddle.nn import ClipGradByGlobalNorm
from paddle.optimizer import Optimizer

HybridParallelClipGrad = (
    fleet.meta_optimizers.dygraph_optimizer.hybrid_parallel_optimizer.HybridParallelClipGrad
)
from paddle.distributed.collective import _get_global_group, new_group

from .group_sharded_storage import GradStorage, ParamStorage
from .group_sharded_utils import GroupShardedClipGrad, Type, device_guard

# CUDA alignment 256 bytes, cpu alignment 4096 bytes
alignment = {"gpu": 256, "cpu": 4096, "xpu": 256}
align = {
    Type.fp16.value: 2,
    Type.bf16.value: 2,
    Type.fp32.value: 4,
}


class GroupShardedOptimizerStage2(Optimizer):
    """
    A wrapper for Sharding Stage2 Optimizer in Dygraph.

    .. warning: ShardingOptimizer encapsulates the optimization strategy and integrates it into the optimizer.

    .. ZeRO: 1.https://arxiv.org/pdf/1910.02054.pdf 2.https://arxiv.org/pdf/1910.02054.pdf.

    """

    # TODO (Baibaifan)
    # Feature Notes:
    # 1. Unified memory for parameters and parameters.grad to InternalStorage.
    # 2. Support the segmentation of optimizer parameters and partial updating of parameters.
    # 3. Dynamically adjust training parameters and models.
    # 4. Support offload function.
    # 5. Support the establishment of independent communication groups.
    # 6. Broadcast_fp16 is not supported now.
    def __init__(
        self,
        params,
        optim,
        group=None,
        offload=False,
        device="xpu" if core.is_compiled_with_xpu() else "gpu",
        pretrain_sync_models=True,
        dp_group=None,
        **kw,
    ):
        super().__init__(learning_rate=optim._learning_rate, parameters=params)
        assert (
            core.is_compiled_with_cuda()
            or core.is_compiled_with_xpu()
            or (device in core.get_all_custom_device_type())
        ), "Only GPU and XPU and CustomDevice is supported now"

        # Segmentation information
        self._dtype_rank_params = (
            OrderedDict()
        )  # {dtype:[param1,param2]} device, rank, params
        self._param2rank = {}
        self.__segment_params = []
        self._rank_buffer_size = {}  # {dtype: {rank: numel+alignment}}
        self._param2align = {}  # {param.name: align}

        # Default information
        self._optim = optim

        # sharing stage 2 comm overlap flag
        self._reduce_overlap = False
        # record the last task used for comm overlap for sharding stage 2
        self._comm_task = None

        assert hasattr(
            self._optim, "_master_weights"
        ), "Must use optimizer with _master_weights attribute"

        # Support parameter group and parameter list
        self._local_params = []
        if isinstance(params[0], dict):
            for param_group in params:
                self._local_params.extend(list(param_group["params"]))
        else:
            self._local_params.extend(list(params))

        self.use_main_grad = None
        for param in self._local_params:
            if self.use_main_grad is None and hasattr(param, "main_grad"):
                self.use_main_grad = True
            if self.use_main_grad:
                assert hasattr(
                    param, "main_grad"
                ), "Params have different main grad attributes."
        if self.use_main_grad:
            assert not offload, "offload not support main_grad for now"

        self._default_device = device
        self._pfp16 = (
            len(
                list(
                    filter(
                        lambda x: x.trainable and x.dtype == Type.fp16.value,
                        self._local_params,
                    )
                )
            )
            > 0
        )
        self._pbf16 = (
            len(
                list(
                    filter(
                        lambda x: x.trainable and x.dtype == Type.bf16.value,
                        self._local_params,
                    )
                )
            )
            > 0
        )

        self._broadcast_overlap = False
        self._forward_pre_hook_remove_helper = []
        try:
            # The fp32 params such as layer_norm_0.w_0 will be at the end of param_list.
            # Have to sort the params to make sure all params are in the forward using order.
            self._broadcast_order_params = sorted(
                self.local_params,
                key=lambda x: int(x.name.split('.')[0].split('_')[-1]),
            )
        except ValueError:
            self._broadcast_order_params = None

        self._group = (
            new_group(_get_global_group().ranks) if group is None else group
        )

        # only support to combine stage2 and dp hybrid parallel now.
        self._dp_group = dp_group
        self.world_size = self._group.nranks
        self._rank = self._group.rank
        self._global_root_rank = self._group.ranks[0]

        if self._dp_group is not None and self._dp_group.nranks > 1:
            assert (
                not offload
            ), "Not support! when using offload with sharding stage2, please use pure sharding stage2, exclude data parallel."

        # Synchronous all ranks models
        if pretrain_sync_models:
            self._sync_params_and_buffers()

        self.param_storages = {}  # {dtype: {rank: InternalStorage}}

        if isinstance(self._optim._grad_clip, ClipGradByGlobalNorm):
            logging.warning(
                "While using ClipGradByGlobalNorm in GroupShardedOptimizerStage2, the grad clip of original optimizer will be changed."
            )

            hcg = fleet.fleet._hcg if hasattr(fleet.fleet, "_hcg") else None
            if (
                hcg
                and hcg.get_parallel_mode() is not ParallelMode.DATA_PARALLEL
                and not offload
            ):
                if self.use_main_grad:
                    self._optim._inner_opt._grad_clip = HybridParallelClipGrad(
                        self._optim._inner_opt._grad_clip, hcg
                    )
                else:
                    self._optim._grad_clip = HybridParallelClipGrad(
                        self._optim._grad_clip, hcg
                    )
            else:
                if self.use_main_grad:
                    self._optim._inner_opt._grad_clip = GroupShardedClipGrad(
                        self._optim._inner_opt._grad_clip,
                        paddle.get_device(),
                        self._group,
                    )
                else:
                    self._optim._grad_clip = GroupShardedClipGrad(
                        self._optim._grad_clip, paddle.get_device(), self._group
                    )

            if self._optim._parameter_list and isinstance(
                self._optim._parameter_list[0], dict
            ):
                for item in self._optim._param_groups:
                    if "grad_clip" in item.keys():
                        item["grad_clip"] = self._optim._grad_clip

        if offload:
            assert (
                self._pfp16
            ), "Only support offload strategy while using 'Adam', 'AdamW' and 'Momentum' optimizer with AMP/Pure FP16"

        self.offload = offload  # Using for offload
        self.offload_device = "cpu"
        self.offload_buffer_size = 0
        self.offload_param2align = {}
        self.offload_params = None
        self.offload_grads = None
        self.dev_id = int(paddle.get_device().split(":")[1])

        self._master_params = {}

        # Update optimizer parameters and adjust parameter storage and use according to rank.
        self._update_opt_status()

    def _set_auxiliary_var(self, key, val):
        super()._set_auxiliary_var(key, val)
        self._optim._set_auxiliary_var(key, val)

    @paddle.autograd.no_grad()
    def _sync_params_and_buffers(self):
        """
        Sync all model states for all ranks
        """

        for p in self._local_params:
            dist.broadcast(
                p, src=self._global_root_rank, group=self._group, sync_op=True
            )

            if self._dp_group:
                dist.broadcast(
                    p,
                    src=self._dp_group.ranks[0],
                    group=self._dp_group,
                    sync_op=True,
                )

    def _update_task(self, task):
        if self._reduce_overlap:
            assert task is not None
        # Only track of the last reduce task.
        # Since all tasks are on the same stream, only need to wait the last one.
        # After waiting for the last reduce task, all reduce tasks before have already finished.
        self._comm_task = task

    def _set_reduce_overlap(self, reduce_overlap):
        # Enable gradients' reduces overlap with backward calculation.
        self._reduce_overlap = reduce_overlap

    def _set_broadcast_overlap(
        self, broadcast_overlap, layers=None, num_groups=None
    ):
        # Enable post optimizer broadcasts overlap with the forward calculation of next batch.
        self._broadcast_overlap = broadcast_overlap
        if self._broadcast_overlap:
            assert (
                layers is not None
            ), "To enable broadcast overlap forward, please pass the module to the function."
            self._layers = layers
            warnings.warn(
                "Setting overlap broadcast means the `paddle.device.cuda.synchronize()` "
                "must be called manually before calling `paddle.save()` and before and inference."
            )
            if self._broadcast_order_params is None:
                # Params' names should be like column_linear_32.w_0 patter to get the best performance.
                warnings.warn(
                    r"The param name passed to the optimizer doesn't follow .+_[0-9]+\..+ patter, "
                    "overlap broadcast may harm the performance."
                )
                self._broadcast_order_params = self._local_params

        if num_groups is None or num_groups > len(self._broadcast_order_params):
            warnings.warn(
                "The num_groups for broadcast is larger than the number of params to be broadcast. "
                "It will set to default value: 1 (use the default sharding group)."
            )
            num_groups = 1

        assert (
            isinstance(num_groups, int) and num_groups > 0
        ), "num_groups should be a positive integer"

        self._number_of_broadcast_groups = num_groups
        self._broadcast_groups = [
            None for _ in range(self._number_of_broadcast_groups)
        ]
        self._broadcast_groups[0] = self._group

        ranks = self._group.ranks
        for i in range(1, self._number_of_broadcast_groups):
            self._broadcast_groups[i] = new_group(ranks)

    def _generate_master_params(self, trainable_params):
        if self.offload:
            for param in trainable_params:
                if param.name not in self._master_params.keys():
                    self._master_params[param.name] = core.eager.Tensor(
                        name=param.name,
                        value=param.cast(dtype=Type.fp32.value).numpy(),
                        place=core.CPUPlace(),
                        stop_gradient=param.stop_gradient,
                    )
        else:
            for param in trainable_params:
                if (
                    param.dtype == Type.fp16.value
                    or param.dtype == Type.bf16.value
                ):
                    master_tensor = paddle.cast(param, Type.fp32.value)
                    master_tensor.name = param.name
                    self._optim._master_weights[param.name] = master_tensor

    def _update_opt_status(self):
        """Update optimizer status and parameter storage information, and special functions to be developed."""
        # func 1
        self._integration_params()

    # Segment helpers

    def _segment_params(self):
        """
        Divide all optimizer parameters equally into rank.
        """
        if len(self.__segment_params) == 0:
            self.__segment_params, param_lists = [
                [] for _ in range(self.world_size)
            ], [[] for _ in range(self.world_size)]
            sizes = [0] * self.world_size
            for param in self._local_params:
                # Add this param to rank with smallest size.
                rank = sizes.index(min(sizes))
                param_lists[rank].append(param)

                # Statistical real numels
                sizes[rank] += param._numel() if param.trainable else 0

            for rank, params in enumerate(param_lists):
                self.__segment_params[rank].extend(params)
        return self.__segment_params

    @property
    def local_params(self):
        return self._local_params

    @property
    def param2rank(self):
        """Map the params to the rank which owns them"""
        if len(self._param2rank) == 0:
            for rank, params in enumerate(self._segment_params()):
                for param in params:
                    self._param2rank[param.name] = rank
        return self._param2rank

    @property
    def dtype_rank_params(self):
        """
        Divide the parameters into groups according to rank and dtype.
        """
        if len(self._dtype_rank_params) == 0:
            # Assign the parameters of each rank according to the type
            trainable_params = list(
                filter(lambda x: x.trainable, self._local_params)
            )
            for param in trainable_params:
                if param.dtype not in self._dtype_rank_params.keys():
                    self._dtype_rank_params[param.dtype] = [
                        [] for _ in range(self.world_size)
                    ]
                self._dtype_rank_params[param.dtype][
                    self.param2rank[param.name]
                ].append(param)

            # Sort per rank params by size
            for dtype in self._dtype_rank_params.keys():
                for rank_params in self._dtype_rank_params[dtype]:
                    rank_params.sort(key=lambda x: x._numel())

        return self._dtype_rank_params

    @property
    def rank_buffer_size(self):
        """
        Count the memory size of the parameters corresponding to rank under the corresponding dtype.
        """
        # CUDA alignment 256 bytes
        if self._default_device in core.get_all_custom_device_type():
            device_alignment = core.libpaddle._get_device_min_chunk_size(
                self._default_device
            )
        else:
            device_alignment = alignment[self._default_device]

        if len(self._rank_buffer_size) == 0:
            for dtype in self.dtype_rank_params.keys():
                if dtype not in self._rank_buffer_size.keys():
                    self._rank_buffer_size[dtype] = {}
                for dst_rank, per_rank_params in enumerate(
                    self.dtype_rank_params[dtype]
                ):
                    if dst_rank not in self._rank_buffer_size[dtype].keys():
                        self._rank_buffer_size[dtype][dst_rank] = 0
                    for param in per_rank_params:
                        if not param.trainable:
                            continue
                        size = param._numel() * align[dtype]
                        remaining = size % device_alignment
                        ali = (
                            0
                            if remaining == 0
                            else device_alignment - remaining
                        )
                        align_ = ali // align[dtype]
                        self._rank_buffer_size[dtype][dst_rank] += (
                            param._numel() + align_
                        )
                        self._param2align[param.name] = align_

        return self._rank_buffer_size

    def _integration_params(self):
        """
        Integrate the parameters into a continuous memory according to rank, and support the update of training parameters.
        """

        for dtype, per_rank_params in self.dtype_rank_params.items():
            if dtype not in self.param_storages.keys():
                self.param_storages[dtype] = {}

            for dst_rank, params in enumerate(per_rank_params):
                if len(params) > 0:
                    # Merge all the trainable params in a single InternalStorage
                    trainable_params = list(
                        filter(lambda x: x.trainable, params)
                    )
                    if (self._pfp16 or self._pbf16) and dst_rank == self._rank:
                        self._generate_master_params(trainable_params)
                    if trainable_params:
                        param_storage = ParamStorage(
                            size=self.rank_buffer_size[dtype][dst_rank],
                            dtype=dtype,
                            device=self._default_device,
                        )

                        param_storage.add_rank_params(
                            trainable_params, self._param2align
                        )
                        self.param_storages[dtype][dst_rank] = param_storage

        # Clear the InternalStorage keys which are not in use anymore
        dtype_in_use = list(self.dtype_rank_params.keys())
        dtype_to_pop = list(
            filter(lambda x: x not in dtype_in_use, self.param_storages.keys())
        )
        for d in dtype_to_pop:
            self.param_storages.pop(d)

        if self.offload:
            self._optim._master_weights = self._master_params
            cpu_master_params = list(self._master_params.values())
            if self._default_device in core.get_all_custom_device_type():
                device_alignment = core.libpaddle._get_device_min_chunk_size(
                    self._default_device
                )
            else:
                device_alignment = alignment[self._default_device]

            for param in cpu_master_params:
                size = param._numel() * align[Type.fp32.value]
                remaining = size % device_alignment
                ali = 0 if remaining == 0 else device_alignment - remaining
                align_ = ali // align[Type.fp32.value]
                self.offload_buffer_size += param._numel() + align_
                self.offload_param2align[param.name] = align_

            if cpu_master_params:
                with device_guard(self._rank, self.offload_device):
                    self.offload_params = ParamStorage(
                        size=self.offload_buffer_size,
                        dtype=Type.fp32.value,
                        device=self.offload_device,
                    )
                    self.offload_params.buffer.name = "offload_buffer"
                    self.offload_params.add_rank_params(
                        cpu_master_params, self.offload_param2align, False
                    )
                    self.offload_params.buffer.stop_gradient = False

                    self.offload_grads = GradStorage(
                        size=self.offload_buffer_size,
                        dtype=Type.fp32.value,
                        device=self.offload_device,
                        destination=self._rank,
                        param2align=self.offload_param2align,
                        convert_cpu=True,
                    )
                    for p in cpu_master_params:
                        self.offload_grads.add_grad(
                            p, self.offload_param2align[p.name]
                        )

                    self._optim._master_weights[
                        self.offload_params.buffer.name
                    ] = self.offload_params.buffer

    def _offload_acc_grad(self, param_name, grad_fp32_cpu):
        """accumulate grads with offload strategy"""
        with device_guard(self._rank, self.offload_device):
            if param_name in self._master_params.keys():
                if self._master_params[param_name].grad is None:
                    self._master_params[param_name]._copy_gradient_from(
                        grad_fp32_cpu
                    )
                else:
                    self._master_params[param_name].grad.add_(grad_fp32_cpu)

        self.offload_params.buffer._copy_gradient_from(
            self.offload_grads.buffer
        )

    def _offload_scale_grad(self, scale_size):
        """scale grads with offload strategy"""
        with device_guard(self._rank, self.offload_device):
            self.offload_grads.buffer.scale_(scale=scale_size)

    def _offload_clear_grad(self):
        """clear grads with offload strategy"""
        with device_guard(self._rank, self.offload_device):
            self.offload_grads.buffer.zero_()

    def _step(self):
        if self._broadcast_overlap:
            # Clear the pre forward hook in the optimizer step.
            for hook_remove in self._forward_pre_hook_remove_helper:
                hook_remove.remove()
            self._forward_pre_hook_remove_helper = []

        if self.offload:
            params_list = [self.offload_params.buffer]

            # TODO(Baibaifan): Offload will support param_groups later
            if not isinstance(self._optim._param_groups[0], dict):
                self._optim._parameter_list = params_list
                self._optim._param_groups = params_list

        # Run the optimizer of the current rank step
        if self.offload:
            with device_guard(device=self.offload_device):
                self._optim.step()

            for param in self._local_params:
                if param.name in self._master_params.keys():
                    if (
                        self._default_device
                        in core.get_all_custom_device_type()
                    ):
                        param.set_value(
                            self._master_params[param.name]
                            ._copy_to(
                                paddle.CustomPlace(
                                    self._default_device, self.dev_id
                                ),
                                True,
                            )
                            .cast(dtype=param.dtype)
                        )
                    elif self._default_device == "xpu":
                        param.set_value(
                            self._master_params[param.name]
                            .to("xpu:" + str(self.dev_id))
                            .cast(dtype=param.dtype)
                        )
                    else:
                        param.set_value(
                            self._master_params[param.name]
                            .cuda(self.dev_id)
                            .cast(dtype=param.dtype)
                        )
        else:
            self._optim.step()

        # Synchronize all the updated shards in between the ranks
        self._broadcast_params()

    def step(self):
        """
        A wrapper for Optimizer's step function to finish the update operation of the optimizer.
        """
        # This method won't be called directly by opt.step()!
        # The _redefine_opt_step() in class GroupShardedStage2 will wrap this function.
        self._step()

    def minimize(self):
        raise RuntimeError(
            "optimizer.minimize() not support now, please use optimizer.step()"
        )

    def set_state_dict(self, state_dict):
        self._optim.set_state_dict(state_dict)

    def state_dict(self):
        return self._optim.state_dict()

    def _clear_cache(self):
        self.__segment_params.clear()
        self._dtype_rank_params.clear()
        self._param2rank.clear()

    @paddle.autograd.no_grad()
    def _broadcast_params(self):
        """Broadcast the parameters of the current rank to each rank"""

        # Exchange all the shards with the other ranks
        if self._broadcast_overlap:
            self._broadcast_params_overlap_forward()
        else:
            for dtype_per_rank in self.param_storages.values():
                for dst_rank, internal_storage in dtype_per_rank.items():
                    dist.broadcast(
                        tensor=internal_storage.buffer,
                        src=self._group.ranks[dst_rank],
                        group=self._group,
                        sync_op=True,
                    )

    def _forward_pre_hook_function(self, tasks):
        # Since the layers will call pre hook by `forward_pre_hook(self, inputs)`,
        # the helper functions needs the x and y to take those params.
        def __impl__(x, y):
            for task in tasks:
                # Wait for broadcast task before using the result of the broadcast.
                task.wait()

        return __impl__

    def set_lr(self, lr):
        super().set_lr(lr)
        self._optim.set_lr(lr)

    def get_lr(self):
        return self._optim.get_lr()

    @paddle.autograd.no_grad()
    def _broadcast_params_overlap_forward(self):
        # Exchange all the shards with the other ranks,
        # but overlap the broadcast with next batch's calculation.
        group_idx = 0

        param2task = {}
        for x in self._broadcast_order_params:
            if x.trainable:
                group = self._broadcast_groups[group_idx]
                group_idx = (group_idx + 1) % self._number_of_broadcast_groups
                task = dist.broadcast(
                    tensor=x,
                    src=group.ranks[self._param2rank[x.name]],
                    group=group,
                    sync_op=False,
                )
                assert x.name not in param2task
                param2task[x.name] = task

        for layer in self._layers.sublayers():
            if len(layer.sublayers()) == 0:
                # Register forward pre hood for leaf layers. This will get the best performance.
                tasks = []
                for param in layer.parameters():
                    if param.trainable:
                        if param.name in param2task:
                            tasks.append(param2task[param.name])
                self._forward_pre_hook_remove_helper.append(
                    layer.register_forward_pre_hook(
                        self._forward_pre_hook_function(tasks)
                    )
                )
