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
# https://github.com/facebookresearch/fairscale/blob/main/fairscale/nn/data_parallel/sharded_ddp.py
# Git commit hash: 8acbec718f3c70a6b9785470bb9e05cd84fc3f8e
# We retain the following license from the original files:

# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import logging
from functools import reduce
from types import MethodType

import paddle
import paddle.distributed as dist
from paddle import nn
from paddle.distributed import collective
from paddle.distributed.utils.log_utils import get_logger
from paddle.framework import core

from .group_sharded_optimizer_stage2 import GroupShardedOptimizerStage2
from .group_sharded_storage import GradStorage
from .group_sharded_utils import Type, device_guard

logger_ = get_logger(logging.WARNING)


def _trainable(param):
    return param.trainable


class GroupShardedStage2(nn.Layer):
    """
    A wrapper for Sharding Stage2 Layer in Dygraph.
    .. warning: GroupShardedStage2 encapsulates the layer strategy and integrates it into the nn.Layer.
    .. ZeRO: https://arxiv.org/pdf/1910.02054.pdf.
    """

    # TODO (Baibaifan)
    # Feature Notes::
    # 1. Unified memory for param and param.grad to InternalStorage.
    # 2. Divide param.grad according to rank to centrally apply for and release GPU memory.
    # 3. Dynamically adjust training parameters and models.
    # 4. Support offload function.
    # 5. Support the establishment of independent communication groups.

    def __init__(
        self,
        layer,
        sharding_optimizer,
        group=None,
        sync_buffers=False,
        buffer_max_size=2**23,  # 8MB
        auto_refresh_trainable=True,
        device="xpu" if core.is_compiled_with_xpu() else "gpu",
        dp_group=None,
    ):
        super().__init__()

        # training options
        self._layer = layer
        self._sharding_optimizers = (
            [sharding_optimizer]
            if not isinstance(sharding_optimizer, list)
            else sharding_optimizer
        )
        assert all(
            isinstance(opt, GroupShardedOptimizerStage2)
            for opt in self._sharding_optimizers
        ), "Please use GroupShardedOptimizerStage2 optimizer"
        self._sync_buffers = sync_buffers
        self._auto_refresh_trainable = auto_refresh_trainable

        # Communication related attributes
        self._group = (
            collective.new_group(collective._get_global_group().ranks)
            if group is None
            else group
        )
        self._world_size_scaling = 1.0 / self._group.nranks
        assert (
            self._group.nranks > 1
        ), "Training must be distributed, ranks must be greater than 1"
        self._rank = self._group.rank
        self._global_root_rank = self._group.ranks[
            0
        ]  # picking ranks index 0 as the reference
        self._default_device = device

        self._dp_group = dp_group

        # Global statistical parameters
        self._all_params = []
        for optim in self._sharding_optimizers:
            self._all_params.extend(list(optim.local_params))
        self.use_main_grad = None
        for param in self._all_params:
            if self.use_main_grad is None and hasattr(param, "main_grad"):
                self.use_main_grad = True
            if self.use_main_grad:
                assert hasattr(
                    param, "main_grad"
                ), "Params have different main grad attributes."

        # sharing stage 2 comm overlap flag
        self._reduce_overlap = False

        self._grad_reduced = []
        self._trainable_param2rank = {}
        self._trainable_param2align = {}
        self._trainable_params = list(
            filter(lambda x: x.trainable, self._all_params)
        )
        self._trainable_mask = list(map(_trainable, self._trainable_params))
        self._param_grads = []

        # Set grad storage size & Display param sizes and model sizes
        model_size = sum([p._numel() for p in self._layer.parameters()])
        assert buffer_max_size >= 0, "buffer_max_size must be GE than 0."
        self._buffer_max_size = self._rank_buffer_size(
            buffer_max_size, model_size
        )
        self._use_grad_storage = buffer_max_size > 0
        self._grad_storages = {}  # {dtype: {rank: GradStorage}}
        self._has_grad_storage = []
        self._grad_storage_list = []

        # Offload
        # TODO(haohongxiang): Now it's not be supported for multi-optimizers using Offload strategy
        self._offload_optims = list(
            filter(lambda optim: optim.offload, self._sharding_optimizers)
        )
        if len(self._offload_optims) > 0:
            assert (
                len(self._sharding_optimizers) == 1
            ), "Only support offload strategy for single optimizer"

        self._offload = len(self._offload_optims) > 0
        self._offload_device = "cpu"

        # Set backward pass hooks
        self._bw_hooks = []

        self.scale_in_opt = False

        # TODO (Baibaifan) Set tasks flow support asynchronous communicate
        # self._tasks_flow = deque()

        # Define optimizer step and clear_grad
        self._redefine_opt_step()
        self._redefine_opt_clear()

    def forward(self, *inputs, **kwargs):
        """
        A wrapper for Sharding Stage2 layer.
        - Fresh trainable params or rebuild grad storage
        - Sync layer's buffer params
        - Clear all flags states
        - Forward for origin layers
        """

        # Whether to need to reset trainable parameters
        needs_fresh = len(self._bw_hooks) == 0 and self.training

        if self._auto_refresh_trainable:
            needs_fresh |= self._detect_train_change()

        # Front hook
        self._init_internal_storage(needs_fresh)

        # Sync layer's buffers state
        if self._sync_buffers:
            self.__sync_buffers()

        # Normal FW on the base model
        fw = self._layer(*inputs, **kwargs)

        return fw

    def set_state_dict(self, state_dict, use_structured_name=True):
        self._layer.set_state_dict(
            state_dict, use_structured_name=use_structured_name
        )

    def state_dict(
        self,
        destination=None,
        include_sublayers=True,
        structured_name_prefix="",
    ):
        return self._layer.state_dict(
            destination=destination,
            include_sublayers=include_sublayers,
            structured_name_prefix=structured_name_prefix,
        )

    def _clear_gradients(self):
        """
        Set zero to the gradient of the optimizer's current rank trainable parameters.
        """
        # Release grad storages
        for dtype in self._grad_storages.keys():
            if (
                not self._offload
                and self._rank in self._grad_storages[dtype].keys()
            ):
                self._grad_storages[dtype][self._rank].buffer.zero_()

        # Release grads of params
        for param in self._trainable_params:
            if param.name in self._param_grads:
                if self.use_main_grad and param.main_grad is not None:
                    param.main_grad.zero_()
                elif param.grad is not None:
                    param._zero_grads()

        # Release grads of master params with offload strategy
        if self._offload:
            self._sharding_optimizers[0]._offload_clear_grad()

    def _grad_scale(self):
        """
        this function will do 2 things:
        1.  Before the optimization, scale main_grad to support gradient merge if param has main_grad, or to support fused_linear_param_grad_add gradient merge.
        2.  Before the optimization, scale the gradients before allreduce of dp_group.
        """

        need_dp_scale = self._dp_group is not None and self._dp_group.nranks > 1
        if self.scale_in_opt:
            scale_factor = self._world_size_scaling
        else:
            scale_factor = 1.0

        if need_dp_scale:
            dp_scale_factor = 1.0 / (self._dp_group.nranks)
            scale_factor = scale_factor * dp_scale_factor

        # Scale grad storages
        for dtype in self._grad_storages.keys():
            if (
                not self._offload
                and self._rank in self._grad_storages[dtype].keys()
            ):
                self._grad_storages[dtype][self._rank].buffer.scale_(
                    scale=scale_factor
                )
        # Scale grads of params
        with paddle.no_grad():
            for param in self._trainable_params:
                if param.name in self._param_grads:
                    if self.use_main_grad and param.main_grad is not None:
                        param.main_grad.scale_(scale=scale_factor)
                    elif param.grad is not None:
                        param.grad.scale_(scale=scale_factor)

        # Scale grads of master params with offload strategy
        if self._offload:
            if need_dp_scale is False:
                return
            self._sharding_optimizers[0]._offload_scale_grad(
                scale=1.0 / (self._dp_group.nranks)
            )

    def _init_internal_storage(self, needs_fresh):
        """
        Judge Fresh trainable params or rebuild grad storage.
        """
        if needs_fresh:
            self._fresh_trainable()
        else:
            self._build_grad_storages()

        # Clear all flags state
        self._clear_counters()

    def to(self, device=None, dtype=None, blocking=True):
        """
        Synchronously or asynchronously convert the data type of the layer, the device is not supported now.
        """
        assert isinstance(device, str), "Device must be type str"
        assert (
            device == self._default_device
        ), "New devices are not supported, because of the optimizer state is not sync"

        self._layer.to(device=device, dtype=dtype, blocking=blocking)

        # Re-build the buckets, hooks, etc..
        self._fresh_trainable()

    def _fresh_trainable(self):
        """Whether to update training parameters."""

        # Make sure that this is not done while gradients are waiting to be reduced (if no_sync context for instance)
        if reduce(lambda x, y: x or y, self._grad_reduced, False):
            logging.warning("Grads waiting to be reduced.")

        self._trainable_params = list(
            filter(lambda x: x.trainable, self._all_params)
        )
        self._trainable_params.sort(key=lambda x: x._numel())

        self._trainable_param2rank = {}
        for optim in self._sharding_optimizers:
            # Need to be wrapped for Sharding Stage2 Optimizer
            if len(optim.param_storages.keys()) == 0:
                optim._update_opt_status()

            # Get the parameters split by the optimizer according to rank
            for (
                per_rank_params
            ) in (
                optim.dtype_rank_params.values()
            ):  # all the params from all ranks
                for params in per_rank_params:
                    for param in filter(lambda x: x.trainable, params):
                        self._trainable_param2rank[param.name] = (
                            optim.param2rank[param.name]
                        )
                        self._trainable_param2align[param.name] = (
                            optim._param2align[param.name]
                        )

        # Create grad_storage
        self._setup_use_grad_storage()
        # setup backward hooks
        self._setup_backward_hooks()

    @paddle.autograd.no_grad()
    def __sync_buffers(self):
        """
        Sync all the param buffers from all ranks (exp: batch norm statistics).
        """

        for buffer in self._layer.buffers(include_sublayers=True):
            dist.broadcast(
                buffer, self._global_root_rank, self._group, sync_op=True
            )

            if self._dp_group and self._dp_group.nranks > 1:
                dist.broadcast(
                    buffer,
                    self._dp_group.ranks[0],
                    self._dp_group,
                    sync_op=True,
                )

    def __getattr__(self, name):
        """Forward missing attributes to wrapped layer."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._layer, name)

    @paddle.autograd.no_grad()
    def _clear_counters(self):
        """Reset all the grad reduce and call counters."""
        if self.training:
            self._grad_reduced = [True for _ in self._trainable_params]

        if self._use_grad_storage:
            for grad_storage in self._grad_storage_list:
                grad_storage.reset_checked_in()

    def _set_reduce_overlap(self, reduce_overlap):
        # Hacky way to not add an extra parameter to the `group_sharded_parallel` funct.
        # User should use this like:
        # model, optimizer, scaler = group_sharded_parallel(...)
        # model._set_reduce_overlap(True)
        self._reduce_overlap = reduce_overlap
        if self._reduce_overlap:
            assert (
                len(self._sharding_optimizers) == 1
            ), "Only support comm overlap strategy for single optimizer"
        self._sharding_optimizers[0]._set_reduce_overlap(reduce_overlap)

    def _get_scaled_grad_fn(self, param):
        @paddle.autograd.no_grad()
        def scale(grad):
            # do gradient scale separately
            # For grad scale, we need to do it in the backward hook due to fp16 may overflow if we first add grad and then scale
            # For main_grad scale and fused_linear_param_grad_add, we do scale in the optimizer.
            if not self.scale_in_opt:
                if (
                    not hasattr(param, "main_grad")
                    and grad is not None
                    and grad.dtype == Type.fp16.value
                ):
                    assert (
                        grad._is_initialized()
                    ), "grad should be initialized in stage2"
                    grad.scale_(self._world_size_scaling)
                else:
                    self.scale_in_opt = True

        return scale

    def _get_reduce_fn(self, index, param, dst_rank):
        """
        There are two ways to reduce gradient.
        - 1. Do not use self._use_grad_storage or exceeded buffer_max_size will be reduced separately.
        - 2. Use grad_storage Reduce the storage to get the full gradient from different ranks.
        """

        if not self._use_grad_storage or not self._has_grad_storage[index]:
            # Direct reduction
            @paddle.autograd.no_grad()
            def reduce(*_):
                # Skip gradient reduction, do not change status information
                if self._grad_reduced[index]:
                    assert (
                        param.grad is not None or param.main_grad is not None
                    ), "Parameter should have grad or main grad"

                    # Change reduce information
                    self._grad_reduced[index] = False

                    # Clear the gradient that does not belong to the current rank through the callback function
                    def cleanup():
                        if dst_rank != self._rank:
                            if self.use_main_grad:
                                param.main_grad._clear_data()
                                param.main_grad = None
                            else:
                                param.clear_gradient(False)
                        elif self._offload:
                            tmp_grad = param.grad.cast(
                                dtype=Type.fp32.value
                            ).cpu()

                            self._sharding_optimizers[0]._offload_acc_grad(
                                param.name, tmp_grad
                            )
                            del tmp_grad
                            param.clear_gradient(False)

                    # Synchronize the reduce parameter gradient asynchronize
                    self._sharding_optimizers[0]._update_task(
                        dist.reduce(
                            tensor=(
                                param.grad
                                if not self.use_main_grad
                                else param.main_grad
                            ),
                            dst=self._group.ranks[dst_rank],
                            group=self._group,
                            sync_op=not self._reduce_overlap,
                        )
                    )

                    # Clear the task flow and trigger callback to clear the redundant gradient
                    # self._clear_task_flow()

                    cleanup()

        else:
            # Buffer reduction
            @paddle.autograd.no_grad()
            def reduce(*_):
                # Skip gradient reduction, do not change status information
                if self._grad_reduced[index]:
                    assert (
                        param.grad is not None or param.main_grad is not None
                    ), "Parameter should have grad or main grad"

                    # Change reduce information
                    self._grad_reduced[index] = False
                    grad_storage = self._grad_storages[param.dtype][dst_rank]
                    grad_storage.params_checked_in += 1

                    if grad_storage.all_checked_in:
                        assert grad_storage.buffer is not None

                        # Clearing up the grad_storage buffer
                        def cleanup():
                            if dst_rank != self._rank:
                                for p in grad_storage._params:
                                    if self.use_main_grad:
                                        p.main_grad._clear_data()
                                        p.main_grad = None
                                    else:
                                        p.clear_gradient(False)

                                grad_storage.buffer._clear_data()
                            elif self._offload:
                                grad_storage.to(device=self._offload_device)
                                for p in grad_storage._params:
                                    with device_guard():
                                        tmp_grad = p.grad.cast(
                                            dtype=Type.fp32.value
                                        )
                                    self._sharding_optimizers[
                                        0
                                    ]._offload_acc_grad(p.name, tmp_grad)
                                    p.clear_gradient(False)
                                grad_storage._device = self._default_device
                                grad_storage.buffer._clear_data()

                        # Reduce the bucket
                        grad_storage.sent = True
                        # Synchronize the reduce parameter gradient asynchronize
                        self._sharding_optimizers[0]._update_task(
                            dist.reduce(
                                tensor=grad_storage.buffer,
                                dst=self._group.ranks[grad_storage.destination],
                                group=self._group,
                                sync_op=not self._reduce_overlap,
                            )
                        )

                        cleanup()

                    # Clear the task flow and trigger callback to clear the redundant gradient
                    # self._clear_task_flow()

        return reduce

    def _setup_backward_hooks(self):
        """
        Set the backward hook to synchronize the gradients of all rank by reduce group ranks.
        """

        # Remove previous backward hooks
        while len(self._bw_hooks) > 0:
            self._bw_hooks.pop().remove()

        # Go through the parameters, attach the hook
        if not self.training:
            return

        for index, param in enumerate(self._trainable_params):
            param._register_grad_hook(self._get_scaled_grad_fn(param))

            dst_rank = self._trainable_param2rank[param.name]

            reduce_function = self._get_reduce_fn(index, param, dst_rank)

            self._bw_hooks.append(
                param._register_backward_hook(reduce_function)
            )

    def _setup_use_grad_storage(self):
        """
        Integrate the parameters gradient into a continuous memory according to rank, and support the update of training parameters.
        """

        # According to parameters's numel sort, allocate memory of parameter gradient to continuous memory according to rank
        self._grad_storages = {}
        self._has_grad_storage = [False for _ in self._trainable_params]

        for index, param in enumerate(self._trainable_params):
            dst_rank = self._trainable_param2rank[param.name]

            if param.dtype not in self._grad_storages.keys():
                self._grad_storages[param.dtype] = {}

            if dst_rank not in self._grad_storages[param.dtype].keys():
                self._grad_storages[param.dtype][dst_rank] = GradStorage(
                    self._buffer_max_size[param.dtype],
                    dtype=(
                        param.dtype
                        if not self.use_main_grad
                        else paddle.float32
                    ),
                    device=self._default_device,
                    destination=dst_rank,
                    param2align=self._trainable_param2align,
                )

            # Criteria to decide whether this parameter is to be put in GradStorage
            if self._grad_storages[param.dtype][dst_rank].can_add_grad_view(
                param, self._trainable_param2align[param.name]
            ):
                self._grad_storages[param.dtype][dst_rank].add_grad(
                    param, self._trainable_param2align[param.name]
                )
                self._has_grad_storage[index] = True
            else:
                self._param_grads.append(param.name)

        for dtype in self._grad_storages.keys():
            self._grad_storage_list.extend(
                list(self._grad_storages[dtype].values())
            )

    # def _clear_task_flow(self):
    #     """Try to consume the previous tasks."""
    #     while len(self._tasks_flow) > 0:
    #         task = self._tasks_flow.popleft()
    #         task.wait()
    #         if task.callback is not None:
    #             task.callback()

    def _detect_train_change(self):
        # Current trainable parameters
        trainable_mask = list(map(_trainable, self._trainable_params))

        # Whether parameters trainability changed
        trainability_changed = trainable_mask != self._trainable_mask

        if trainability_changed:
            logging.warning(
                "Trainable params changed, because of eval/train mode or parameter freezing/unfreeze."
            )
            self._trainable_mask = trainable_mask

        return trainability_changed

    def _build_grad_storages(self):
        """
        Rebuild grad storages.
        """
        # Rebuild fp16/fp32 grad storages
        for dtype in self._grad_storages.keys():
            for dst_rank, grad_storage in self._grad_storages[dtype].items():
                if self._offload or dst_rank != self._rank:
                    grad_storage.manual_release()
                    grad_storage.rebuild()

    def _rank_buffer_size(self, buffer_max_size, model_size):
        """
        Generate the minimum buffer size for each rank & Display param sizes and model sizes.
        """

        # Initialize buffer size
        rank_buffer_size = {}
        for shard_opt in self._sharding_optimizers:
            if shard_opt.rank_buffer_size:
                for dtype in shard_opt.rank_buffer_size.keys():
                    sizes = max(shard_opt.rank_buffer_size[dtype].values())
                    rank_buffer_size[dtype] = min(sizes, buffer_max_size)

        if Type.fp16.value in rank_buffer_size.keys():
            # FP16 GradStorage and model size
            logger_.info(
                f"====== FP16 GradStorage size: {rank_buffer_size[Type.fp16.value] / 2**19:.2f}M parameters, Model size {model_size / 2**19:.2f}M parameters ======"
            )
        if Type.bf16.value in rank_buffer_size.keys():
            # FP16 GradStorage and model size
            logger_.info(
                f"====== BF16 GradStorage size: {rank_buffer_size[Type.bf16.value] / 2**19:.2f}M parameters, Model size {model_size / 2**19:.2f}M parameters ======"
            )
        if Type.fp32.value in rank_buffer_size.keys():
            # FP32 GradStorage and model size
            logger_.info(
                f"====== FP32 GradStorage size: {rank_buffer_size[Type.fp32.value] / 2**18:.2f}M parameters, Model size {model_size / 2**18:.2f}M parameters ======"
            )
        return rank_buffer_size

    def _dp_allreduce(self):
        # do dp allreduce here for gradient merge.
        if self._dp_group and self._dp_group.nranks > 1:
            for dtype in self._grad_storages.keys():
                for rank, g in sorted(
                    self._grad_storages[dtype].items(), key=lambda x: x[0]
                ):
                    if g.destination == self._rank:
                        assert g.buffer._is_initialized()
                        dist.all_reduce(
                            tensor=g.buffer,
                            group=self._dp_group,
                            sync_op=True,
                        )
            for param in self._trainable_params:
                if param.name in self._param_grads:
                    if self.use_main_grad and param.main_grad is None:
                        continue
                    elif param.grad is None:
                        continue
                    dst_rank = self._trainable_param2rank[param.name]
                    if dst_rank == self._rank:
                        dist.all_reduce(
                            tensor=(
                                param.grad
                                if not self.use_main_grad
                                else param.main_grad
                            ),
                            group=self._dp_group,
                            sync_op=True,
                        )

    def _redefine_opt_step(self):
        grad_func = self._grad_scale
        dp_allreduce_func = self._dp_allreduce

        for opt in self._sharding_optimizers:
            opt_step = opt.step

            def _opt_step(self):
                if self._reduce_overlap:
                    # Wait for the last reduce task. This wait must before grad scale function.
                    assert self._comm_task is not None
                    self._comm_task.wait()

                grad_func()
                dp_allreduce_func()
                opt_step()

            opt.step = MethodType(_opt_step, opt)

    def _redefine_opt_clear(self):
        clear_func = self._clear_gradients

        def _opt_clear(self):
            clear_func()

        for opt in self._sharding_optimizers:
            opt.clear_grad = MethodType(_opt_clear, opt)
