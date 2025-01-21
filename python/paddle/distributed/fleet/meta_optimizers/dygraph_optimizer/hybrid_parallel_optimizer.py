# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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


import os

import paddle
from paddle import framework
from paddle.autograd import no_grad
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_optimizers.dygraph_optimizer.dygraph_sharding_optimizer import (
    DygraphShardingOptimizer,
    DygraphShardingOptimizerV2,
)
from paddle.distributed.fleet.utils.hybrid_parallel_util import (
    obtain_optimizer_parameters_list,
)
from paddle.framework import core
from paddle.nn import ClipGradByGlobalNorm, clip

from ...base.topology import ParallelMode
from ...utils import timer_helper as timer
from ...utils.hybrid_parallel_util import (
    fused_allreduce_gradients,
    unwrap_optimizer,
)
from ...utils.log_util import get_sync_logger, logger
from ...utils.mix_precision_utils import MixPrecisionOptimizer

g_profile_optimizer_details_steps = int(
    os.getenv("FLAGS_profile_optimizer_details_steps", "0")
)

__all__ = []


class HybridParallelClipGrad:
    def __init__(self, clip, hcg, timers=None):
        self._clip = clip
        self._hcg = hcg
        self.not_sharding_stage1 = True
        self._timers = timers
        self.processed_steps = 0

    def _global_norm(self, global_norm_var_dist, global_norm_var_not_dist):
        if self.processed_steps < g_profile_optimizer_details_steps:
            get_sync_logger().info("Starting to calculate global norm.")
        # sharding first
        sharding_flag = self._hcg.get_sharding_parallel_world_size() > 1
        dp_flag = self._hcg.get_data_parallel_world_size() > 1
        mp_flag = self._hcg.get_model_parallel_world_size() > 1
        pp_flag = self._hcg.get_pipe_parallel_world_size() > 1

        # add all reduce to get global norm of distributed params_and_grads
        if sharding_flag:
            # norm of mp distributed variable
            if mp_flag:
                if self._timers:
                    self._timers("dygraph-clip-sd-group-dist").start()
                # dist should reduce among sharding group、mp group、pp group
                paddle.distributed.all_reduce(
                    global_norm_var_dist,
                    group=self._hcg.get_sharding_parallel_group(),
                )
                if self._timers:
                    self._timers("dygraph-clip-sd-group-dist").stop()
            # not dist only reduce among sharding group and pp group later
            if self._timers:
                self._timers("dygraph-clip-sd-group-non-dist").start()
            paddle.distributed.all_reduce(
                global_norm_var_not_dist,
                group=self._hcg.get_sharding_parallel_group(),
            )
            if self._timers:
                self._timers("dygraph-clip-sd-group-non-dist").stop()

        # norm of mp distributed variable
        if self._timers:
            self._timers("dygraph-clip-check-group").start()
        if mp_flag:
            # dist should reduce among sharding group、mp group、pp group

            # the else branch would suffice, but this branch remains here for number precision backward compatibility
            if not (dp_flag and sharding_flag):
                paddle.distributed.all_reduce(
                    global_norm_var_dist,
                    group=self._hcg.get_check_parallel_group(sharding_flag),
                )
            else:
                paddle.distributed.all_reduce(
                    global_norm_var_dist,
                    group=self._hcg.get_model_parallel_group(),
                )
                if pp_flag:
                    paddle.distributed.all_reduce(
                        global_norm_var_dist,
                        group=self._hcg.get_pipe_parallel_group(),
                    )
        if self._timers:
            self._timers("dygraph-clip-check-group").stop()

        # add all reduce to get global norm of non-distributed params_and_grads in groups of pp
        if self._timers:
            self._timers("dygraph-clip-pp-group").start()
        if pp_flag:
            paddle.distributed.all_reduce(
                global_norm_var_not_dist,
                group=self._hcg.get_pipe_parallel_group(),
            )
        if self._timers:
            self._timers("dygraph-clip-pp-group").stop()

        if self.processed_steps < g_profile_optimizer_details_steps:
            get_sync_logger().info("Finished calculating global norm.")
        self.processed_steps += 1

    @no_grad()
    def _dygraph_clip(self, params_grads):
        if self._timers:
            self._timers("dygraph-clip").start()
        sum_square_dist_fp16 = []
        sum_square_dist_bf16 = []
        sum_square_dist_fp32 = []
        sum_square_not_dist_fp16 = []
        sum_square_not_dist_bf16 = []
        sum_square_not_dist_fp32 = []

        for p, g in params_grads:
            if g is None:
                continue
            if getattr(p, 'need_clip', True) is False:
                continue
            merge_grad = g
            if g.type == core.VarDesc.VarType.SELECTED_ROWS:
                merge_grad = clip.merge_selected_rows(g)
                merge_grad = clip.get_tensor_from_selected_rows(merge_grad)
            sum_square = clip._squared_l2_norm(merge_grad)

            not_shared_enable = (not hasattr(p, 'is_firstly_shared')) or (
                hasattr(p, 'is_firstly_shared')
                and getattr(p, 'is_firstly_shared', True)
            )

            if not_shared_enable:
                if p.is_distributed:
                    if g.dtype == paddle.float16:
                        sum_square_dist_fp16.append(sum_square)
                    elif g.dtype == paddle.bfloat16:
                        sum_square_dist_bf16.append(sum_square)
                    elif g.dtype == paddle.float32:
                        sum_square_dist_fp32.append(sum_square)
                else:
                    if g.dtype == paddle.float16:
                        sum_square_not_dist_fp16.append(sum_square)
                    if g.dtype == paddle.bfloat16:
                        sum_square_not_dist_bf16.append(sum_square)
                    elif g.dtype == paddle.float32:
                        sum_square_not_dist_fp32.append(sum_square)

        def async_add_n(var_list):
            return paddle.stack(var_list).sum()

        # global norm of distributed FP16 params_and_grads
        if len(sum_square_dist_fp16) == 0:
            global_norm_dist_fp16 = paddle.zeros((1,), dtype=paddle.float32)
        else:
            global_norm_dist_fp16 = async_add_n(sum_square_dist_fp16)
            global_norm_dist_fp16 = paddle.cast(
                global_norm_dist_fp16, dtype=paddle.float32
            )

        # global norm of non-distributed FP16 params_and_grads
        if len(sum_square_not_dist_fp16) == 0:
            global_norm_not_dist_fp16 = paddle.zeros((1,), dtype=paddle.float32)
        else:
            global_norm_not_dist_fp16 = async_add_n(sum_square_not_dist_fp16)
            global_norm_not_dist_fp16 = paddle.cast(
                global_norm_not_dist_fp16, dtype=paddle.float32
            )

        # global norm of distributed BF16 params_and_grads
        if len(sum_square_dist_bf16) == 0:
            global_norm_dist_bf16 = paddle.zeros((1,), dtype=paddle.float32)
        else:
            global_norm_dist_bf16 = async_add_n(sum_square_dist_bf16)
            global_norm_dist_bf16 = paddle.cast(
                global_norm_dist_bf16, dtype=paddle.float32
            )

        # global norm of non-distributed FP16 params_and_grads
        if len(sum_square_not_dist_bf16) == 0:
            global_norm_not_dist_bf16 = paddle.zeros((1,), dtype=paddle.float32)
        else:
            global_norm_not_dist_bf16 = async_add_n(sum_square_not_dist_bf16)
            global_norm_not_dist_bf16 = paddle.cast(
                global_norm_not_dist_bf16, dtype=paddle.float32
            )

        # global norm of distributed FP32 params_and_grads
        if len(sum_square_dist_fp32) == 0:
            global_norm_dist_fp32 = paddle.zeros((1,), dtype=paddle.float32)
        else:
            global_norm_dist_fp32 = async_add_n(sum_square_dist_fp32)

        # global norm of non-distributed FP32 params_and_grads
        if len(sum_square_not_dist_fp32) == 0:
            global_norm_not_dist_fp32 = paddle.zeros((1,), dtype=paddle.float32)
        else:
            global_norm_not_dist_fp32 = async_add_n(sum_square_not_dist_fp32)

        global_norm_var_dist = (
            global_norm_dist_fp16
            + global_norm_dist_bf16
            + global_norm_dist_fp32
        )
        global_norm_var_not_dist = (
            global_norm_not_dist_fp16
            + global_norm_not_dist_bf16
            + global_norm_not_dist_fp32
        )

        result = self._comm_and_clip(
            params_grads, global_norm_var_dist, global_norm_var_not_dist
        )
        if self._timers:
            self._timers("dygraph-clip").stop()

        return result

    def _comm_and_clip(
        self, params_grads, global_norm_var_dist, global_norm_var_not_dist
    ):
        self._global_norm(global_norm_var_dist, global_norm_var_not_dist)

        global_norm_var_fp32 = paddle.sqrt(
            global_norm_var_dist + global_norm_var_not_dist
        )

        max_global_norm = paddle.full(
            shape=[],
            dtype=global_norm_var_fp32.dtype,
            fill_value=self.clip_norm,
        )
        clip_var = paddle.divide(
            x=max_global_norm,
            y=paddle.maximum(x=global_norm_var_fp32, y=max_global_norm)
            + paddle.full(shape=[], dtype=paddle.float32, fill_value=1.0e-6),
        )
        clip_var_fp16 = paddle.cast(clip_var, paddle.float16)

        if (
            not isinstance(
                paddle.framework._current_expected_place(), paddle.CustomPlace
            )
            or paddle.framework._current_expected_place().get_device_type()
            == 'npu'
        ):
            clip_var_bf16 = paddle.cast(clip_var, paddle.bfloat16)
        for p, g in params_grads:
            if g is None:
                continue
            if getattr(p, 'need_clip', True) is False:
                continue
            if g.dtype == paddle.float16:
                g.multiply_(clip_var_fp16)
            elif g.dtype == paddle.bfloat16:
                g.multiply_(clip_var_bf16)
            else:
                g.multiply_(clip_var)
            p._reset_grad_inplace_version(True)

        return params_grads

    def __getattr__(self, item):
        return getattr(self._clip, item)

    def __call__(self, params_grads):
        return self._dygraph_clip(params_grads)


class HybridParallelOptimizer:
    # adapter wrapper for optimizer
    def __init__(self, optimizer, hcg, strategy):
        # Note: Only sharding stage 1 is considered in HybridParallelOptimizer.
        # The sharding stage2 and stage3 optimizers are invoked in other api.
        if hcg.get_sharding_parallel_world_size() > 1:
            split_param = strategy.hybrid_configs[
                'sharding_configs'
            ].split_param
            ShardingOptimizer = (
                DygraphShardingOptimizerV2
                if split_param
                else DygraphShardingOptimizer
            )
            optimizer = ShardingOptimizer(optimizer, hcg)

        self._enable_timer = strategy.hybrid_configs["enable_optimizer_timer"]

        if self._enable_timer:
            if not timer.is_timer_initialized():
                timer.set_timers()
            self._timers = timer.get_timers()
        else:
            self._timers = None

        self._inner_opt = optimizer
        self._strategy = strategy
        self._hcg = hcg

        self._use_dp_mode = (
            self._hcg.get_parallel_mode() == ParallelMode.DATA_PARALLEL
        )

        self._need_dp = self._hcg.get_data_parallel_world_size() > 1

        # NOTE(shenliang03): Because of the pure DataParallel mode, the gradient synchronization
        # is achieved through reducer, so there is no need to call fuse_allreduce in optimizer.
        self._dp_enable = not self._use_dp_mode and self._need_dp

        self._sharding_enable = self._hcg.get_sharding_parallel_world_size() > 1

        self._sep_enable = self._hcg.get_sep_parallel_world_size() > 1

        if (
            isinstance(self._inner_opt._grad_clip, ClipGradByGlobalNorm)
            and not self._use_dp_mode
        ):
            logger.warning(
                "While using ClipGradByGlobalNorm in TensorParallel, PipelineParallel "
                "or Sharding, the grad clip of original optimizer will be changed."
            )

            inner_opt = unwrap_optimizer(
                self._inner_opt,
                (
                    MixPrecisionOptimizer,
                    DygraphShardingOptimizer,
                    DygraphShardingOptimizerV2,
                ),
            )

            if (
                inner_opt._parameter_list
                and not isinstance(inner_opt._parameter_list[0], dict)
                and len(
                    [
                        p
                        for p in inner_opt._parameter_list
                        if hasattr(p, "main_grad")
                    ]
                )
                > 0
            ):
                inner_opt._grad_clip = HybridParallelClipGrad(
                    inner_opt._grad_clip, hcg, self._timers
                )
            else:
                inner_opt._grad_clip = HybridParallelClipGrad(
                    inner_opt._grad_clip, hcg, self._timers
                )
                if inner_opt._parameter_list and isinstance(
                    inner_opt._parameter_list[0], dict
                ):
                    for item in inner_opt._param_groups:
                        if "grad_clip" in item.keys():
                            item["grad_clip"] = HybridParallelClipGrad(
                                inner_opt._grad_clip, hcg, self._timers
                            )
        self.processed_steps = 0

    def _set_all_gather_overlap_forward(
        self, all_gather_overlap_forward, layers=None
    ):
        self._all_gather_overlap_forward = all_gather_overlap_forward
        if self._all_gather_overlap_forward:
            self._layers = layers
            self._inner_opt._set_all_gather_overlap_forward(
                self._all_gather_overlap_forward, self._layers
            )

    def _set_broadcast_overlap(self, broadcast_overlap, layers=None):
        self._broadcast_overlap = broadcast_overlap
        if self._broadcast_overlap:
            self._layers = layers
            self._inner_opt._set_broadcast_overlap(
                self._broadcast_overlap, self._layers
            )

    def _insert_sync(self, sync_var, src, mp_group, sync_mode):
        if sync_mode == "broadcast":
            paddle.distributed.broadcast(
                sync_var, src=src, group=mp_group, sync_op=True
            )
        else:
            paddle.distributed.all_reduce(
                sync_var, group=mp_group, sync_op=True
            )
            sync_var.multiply_(
                paddle.full(
                    shape=[],
                    dtype=sync_var.dtype,
                    fill_value=(1.0 / mp_group.nranks),
                )
            )

    def _filter_fn(self, param, strategy):
        p_name = param.name
        tar_param = strategy.sync_param_name
        if param.is_distributed is False:
            for tar in tar_param:
                if tar in p_name:
                    return True
        return False

    def _step(self, parameters_list):
        if self.processed_steps < g_profile_optimizer_details_steps:
            get_sync_logger().info("Starting hybridoptimizer step")

        mp_group = self._hcg.get_model_parallel_group()
        src_rank = self._hcg.get_model_parallel_group_src_rank()
        params = None
        mp_configs = None

        if mp_group.nranks > 1:
            mp_configs = fleet.fleet._user_defined_strategy.hybrid_configs[
                "mp_configs"
            ]

        if mp_configs and (
            mp_configs.sync_param
            or mp_configs.sync_grad
            or mp_configs.sync_moment
        ):
            params = sorted(
                [
                    p
                    for p in parameters_list
                    if self._filter_fn(p, fleet.fleet._user_defined_strategy)
                ],
                key=lambda p: p.name,
            )

        def syc_grad(p):
            if hasattr(p, "main_grad") and p.main_grad is not None:
                assert p.grad is None
                self._insert_sync(
                    p.main_grad, src_rank, mp_group, mp_configs.sync_mode
                )
            elif p.grad is not None:
                self._insert_sync(
                    p.grad, src_rank, mp_group, mp_configs.sync_mode
                )

        if self.processed_steps < g_profile_optimizer_details_steps:
            get_sync_logger().info("Starting mp grad sync")

        # Grad sync before opt
        if mp_group.nranks > 1 and mp_configs and mp_configs.sync_grad:
            for p in params:
                syc_grad(p)

        if self.processed_steps < g_profile_optimizer_details_steps:
            get_sync_logger().info("Finished mp grad sync")

        self._inner_opt.step()

        def syc_param(p):
            # Param sync after opt
            self._insert_sync(p, src_rank, mp_group, mp_configs.sync_mode)

        def syc_master_weight(p):
            # Master param sync after opt
            if (
                hasattr(self._inner_opt, "_multi_precision")
                and self._inner_opt._multi_precision
                and p.name in self._inner_opt._master_weights
            ):
                self._insert_sync(
                    self._inner_opt._master_weights[p.name],
                    src_rank,
                    mp_group,
                    mp_configs.sync_mode,
                )

        # syc param and master weight after opt
        if mp_group.nranks > 1 and mp_configs and mp_configs.sync_param:
            for p in params:
                syc_param(p)
                syc_master_weight(p)

        def syc_moment(p):
            if isinstance(
                self._inner_opt,
                (paddle.optimizer.Adam, paddle.optimizer.AdamW),
            ):
                if (
                    p.name
                    in self._inner_opt._accumulators[
                        self._inner_opt._moment1_acc_str
                    ]
                ):
                    moment1 = self._inner_opt._get_accumulator(
                        self._inner_opt._moment1_acc_str, p
                    )
                    self._insert_sync(
                        moment1, src_rank, mp_group, mp_configs.sync_mode
                    )

                if (
                    p.name
                    in self._inner_opt._accumulators[
                        self._inner_opt._moment2_acc_str
                    ]
                ):
                    moment2 = self._inner_opt._get_accumulator(
                        self._inner_opt._moment2_acc_str, p
                    )
                    self._insert_sync(
                        moment2, src_rank, mp_group, mp_configs.sync_mode
                    )

        # Moment sync after opt
        if mp_group.nranks > 1 and mp_configs and mp_configs.sync_moment:
            for p in params:
                syc_moment(p)
        if self.processed_steps < g_profile_optimizer_details_steps:
            get_sync_logger().info("Finishing hybridoptimizer step")
        self.processed_steps += 1

    def _hybrid_sync_grad(self, parameter_list):
        dp_parameter_list = parameter_list
        if self._sharding_enable:
            assert isinstance(
                self._inner_opt,
                (DygraphShardingOptimizer, DygraphShardingOptimizerV2),
            )
            self._inner_opt.reduce_gradients(parameter_list, self._hcg)
            dp_parameter_list = self._inner_opt.filter_parameters(
                parameter_list, self._hcg
            )
        if self._dp_enable or self._sep_enable:
            fused_allreduce_gradients(dp_parameter_list, self._hcg)

    @no_grad()
    @framework.dygraph_only
    def step(self):
        parameter_list = list(obtain_optimizer_parameters_list(self._inner_opt))
        self._hybrid_sync_grad(parameter_list)
        self._step(parameter_list)

    @no_grad()
    def minimize(
        self, loss, startup_program=None, parameters=None, no_grad_set=None
    ):
        # minimize does not support parameters in the form of param_group,
        # so no need use _obtain_optimizer_parameters_list
        parameter_list = (
            parameters
            if parameters
            else obtain_optimizer_parameters_list(self._inner_opt)
        )
        parameter_list = list(parameter_list)
        self._hybrid_sync_grad(parameter_list)
        return self._inner_opt.minimize(
            loss, startup_program, parameter_list, no_grad_set
        )

    def __getattr__(self, item):
        return getattr(self._inner_opt, item)
