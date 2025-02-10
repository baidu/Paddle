#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from __future__ import annotations

import os
import queue
import sys
import time
import warnings
from collections import defaultdict, deque
from enum import Enum
from typing import Callable

import paddle
from paddle import framework

from ..meta_optimizers.dygraph_optimizer import HybridParallelOptimizer
from ..utils import timer_helper as timer
from ..utils.hybrid_parallel_util import (
    broadcast_dp_parameters,
    broadcast_mp_parameters,
    broadcast_sep_parameters,
    broadcast_sharding_parameters,
)
from ..utils.log_util import get_sync_logger, logger
from .meta_parallel_base import MetaParallelBase
from .parallel_layers.pp_layers import PipelineLayer

_use_four_directions = os.environ.get(
    'PADDLE_USE_FOUR_DIRECTIONS_P2P', paddle.base.core.is_compiled_with_xpu()
)
_use_four_directions = False  # xpu use the same p2p method as gpu
if _use_four_directions:
    from .pp_utils import four_directions_p2p_communication as p2p
else:
    from .pp_utils import p2p_communication as p2p

from paddle.distributed import fleet
from paddle.distributed.fleet.utils.tensor_fusion_helper import (
    HOOK_ACTION,
    FusedCommBuffer,
    assign_group_by_size,
)

from .pipeline_hooks import (
    PipelineHook,
)

g_profile_pipeline_details_steps = int(
    os.getenv("FLAGS_profile_pipeline_details_steps", "0")
)

__all__ = []


def get_action(is_dp, shard_split_param=False):
    if is_dp:
        return HOOK_ACTION.ALL_REDUCE
    if shard_split_param:
        return HOOK_ACTION.REDUCE_SCATTER
    return HOOK_ACTION.REDUCE


def _get_align_mode_scale():
    hcg = fleet.get_hybrid_communicate_group()
    data_parallel_world_size = hcg.get_data_parallel_world_size()
    sharding_parallel_world_size = hcg.get_sharding_parallel_world_size()
    return max(data_parallel_world_size, 1) * max(
        sharding_parallel_world_size, 1
    )


# assume only the first stage and last stage need data, and data consumption is ordered
# to be replaced by real micro dataset from reader
class FakeMicroDataset:
    def __init__(
        self, data, is_first_stage, is_last_stage, acc_steps, micro_batch_size
    ):
        self._data = data
        self._index = 0
        self._acc_steps = acc_steps
        self._is_first_stage = is_first_stage
        self._is_last_stage = is_last_stage
        self._micro_batch_size = micro_batch_size

    def __iter__(self):
        return self

    def __next__(self):
        if self._index >= self._acc_steps:
            raise StopIteration
        assert self._is_first_stage or self._is_last_stage
        micro_batch_data = self._load_micro_batch(self._index)
        self._index += 1
        return micro_batch_data

    def _load_micro_batch(self, micro_step):
        inputs = self._data

        data = None
        label = None
        if self._is_first_stage:
            assert len(inputs) == 2, "length of input should be 2"
            data = self._load_micro_batch_impl(inputs[0], micro_step)

        if self._is_last_stage:
            assert len(inputs) == 2, "length of input should be 2"
            label = self._load_micro_batch_impl(inputs[1], micro_step)

        return (data, label)

    def _load_micro_batch_impl(self, inputs, micro_step):
        begin = micro_step * self._micro_batch_size
        end = begin + self._micro_batch_size

        if isinstance(inputs, tuple):
            output = []
            for data in inputs:
                if isinstance(data, list):
                    assert (
                        len(data) == self._acc_steps
                    ), f"length of data should be {self._acc_steps}, but it is {len(data)}"
                    output.append(
                        data[micro_step].detach()
                        if data[micro_step] is not None
                        else None
                    )
                elif data is not None:
                    self._check_data_valid(data)
                    output.append(data[begin:end, :].detach())
                else:
                    output.append(None)
            return tuple(output)

        elif isinstance(inputs, list):
            assert (
                len(inputs) == self._acc_steps
            ), f"length of data should be {self._acc_steps}, but it is {len(inputs)}"
            return inputs[micro_step].detach()
        elif inputs is not None:
            self._check_data_valid(inputs)
            return inputs[begin:end, :].detach()
        else:
            return None

    def _check_data_valid(self, data):
        batch_size = data.shape[0]
        assert self._micro_batch_size * self._acc_steps == batch_size, (
            "batch_size needs to be divisible by micro_batch_size. Currently, "
            f"batch_size = {batch_size}, micro_batch_size = {self._micro_batch_size}, accumulate_steps = {self._acc_steps}."
        )


# Enum for specifying the pipeline parallel micro-step locations.
class PipelineParallelMicroStepLocations(Enum):
    FORWARD_BEGIN = 'forward_begin'
    FORWARD_END = 'forward_end'
    BACKWARD_BEGIN = 'backward_begin'
    BACKWARD_END = 'backward_end'


# A callback class for managing hooks at different stages of a pipeline parallel process.
class PipelineParallelMicroStepCallback:
    def __init__(self):
        # Initializes a dictionary to store hooks for each micro-step location in the pipeline.
        self.hooks: dict[PipelineParallelMicroStepLocations, list[Callable]] = {
            PipelineParallelMicroStepLocations.FORWARD_BEGIN: [],
            PipelineParallelMicroStepLocations.FORWARD_END: [],
            PipelineParallelMicroStepLocations.BACKWARD_BEGIN: [],
            PipelineParallelMicroStepLocations.BACKWARD_END: [],
        }

    def register_hook(
        self, location: PipelineParallelMicroStepLocations, hook: Callable
    ):
        """
        Registers a hook function to be called at a specified pipeline parallel micro-step location.

        Args:
            location (PipelineParallelMicroStepLocations): The micro-step location where the hook should be registered.
            hook (Callable): The hook function to be registered. The function should accept the following optional keyword arguments:
                - input_tensor (paddle.Tensor): The input tensor to the current micro-step.
                - output_tensor (paddle.Tensor): The output tensor from the current micro-step.
                - input_tensor_grad (paddle.Tensor): The gradient of the input tensor.
                - output_tensor_grad (paddle.Tensor): The gradient of the output tensor.
                - step_id (paddle.Tensor): An identifier for the current step in the pipeline.

        Raises:
            AssertionError: If the specified location is not a valid micro-step location.
        """
        assert (
            location in self.hooks
        ), f"Invalid location '{location}'. Valid locations are 'forward_begin', 'forward_end', 'backward_begin', or 'backward_end'."
        self.hooks[location].append(hook)

    def on_location(
        self, location: PipelineParallelMicroStepLocations, **kwargs
    ):
        """
        Triggers all registered hooks at a specified pipeline parallel micro-step location.

        Args:
            location (PipelineParallelMicroStepLocations): The micro-step location where the hooks should be triggered.
            kwargs: Additional keyword arguments to be passed to the hook functions.

        Raises:
            AssertionError: If the specified location is not a valid micro-step location.
        """
        assert (
            location in self.hooks
        ), f"Invalid location '{location}'. Valid locations are 'forward_begin', 'forward_end', 'backward_begin', or 'backward_end'."
        for hook in self.hooks[location]:
            hook(**kwargs)


pipeline_parallel_callbacks_ = PipelineParallelMicroStepCallback()


# It is typically very difficult for us to directly access the PipelineParallel object.
# Users may use fleet.distributed_model to wrap a model into a pipeline parallel model (PP model).
# We may not have access to the wrapped model when we want to register hooks, for example, when using PaddleNLP trainer to wrap around the PP model.
# Additionally, we usually have only one `PipelineParallel` model, so the callbacks are registered globally.
def register_global_pipeline_parallel_hook(
    location: PipelineParallelMicroStepLocations, hook: Callable
):
    """
    Registering global hooks for pipeline parallelism.
    """
    pipeline_parallel_callbacks_.register_hook(location, hook)


class PipelineParallel(MetaParallelBase):
    def __init__(self, layers, hcg, strategy):
        if not isinstance(layers, PipelineLayer):
            raise TypeError(
                "The Layer should be a derived class of PipelineLayer."
            )
        super().__init__(layers, hcg, strategy)
        self.use_data_parallel = self._hcg.get_data_parallel_world_size() > 1
        self.use_model_parallel = self._hcg.get_model_parallel_world_size() > 1
        self.use_sep_parallel = self._hcg.get_sep_parallel_world_size() > 1
        self.use_sharding_parallel = (
            self._hcg.get_sharding_parallel_world_size() > 1
        )

        self.total_loss = None

        self.micro_batch_size = self._strategy.pipeline_configs[
            'micro_batch_size'
        ]
        self.accumulate_steps = self._strategy.pipeline_configs[
            'accumulate_steps'
        ]
        # If sent tensor are not the same from different hosts,
        # they shouldn't been sent partially and then concatenated as a whole tensor.
        self._enable_partial_send_recv = self._strategy.pipeline_configs[
            'enable_partial_send_recv'
        ]
        self._using_cache = self._strategy.pipeline_configs['p2p_cache_shape']

        self.num_stages = self._hcg.get_pipe_parallel_world_size()
        self.stage_id = self._hcg.get_stage_id()
        self.global_rank = self._hcg.get_global_rank()
        self.pp_group = self._hcg.get_pipe_parallel_group()

        self.dp_group = self._hcg.get_data_parallel_group()

        # fused sep and dp
        if self.use_sep_parallel:
            self.dp_group = self._hcg.get_dp_sep_parallel_group()

        self.sharding_group = self._hcg.get_sharding_parallel_group()

        self._virtual_pp_world_size = None
        self._virtual_pp_rank = None
        self._real_pp_world_size = self.num_stages
        self._real_pp_rank = self.stage_id

        self._delay_scale_loss = self._strategy.hybrid_configs[
            "pp_configs"
        ].delay_scale_loss
        # TODO(PP Dev): support dp_comm_overlap without use_main_grad training.
        # This combination will trigger inplace check error during `reshape_` in function `_split_tensors`.
        self._dp_comm_overlap = self._strategy.hybrid_configs[
            "pp_configs"
        ].dp_comm_overlap
        self._sharding_comm_overlap = self._strategy.hybrid_configs[
            "pp_configs"
        ].sharding_comm_overlap
        self._enable_timer = self._strategy.hybrid_configs[
            "pp_configs"
        ].enable_timer
        self._release_gradients = self._strategy.hybrid_configs[
            "pp_configs"
        ].release_gradients

        self._sharding_split_param = self._strategy.hybrid_configs[
            "sharding_configs"
        ].split_param

        self._overlap_p2p_comm = self._strategy.hybrid_configs[
            "pp_configs"
        ].overlap_p2p_comm

        self._clear_every_step_cache = self._strategy.hybrid_configs[
            "pp_configs"
        ].clear_every_step_cache

        self._use_batch_p2p_comm = self._strategy.hybrid_configs[
            "pp_configs"
        ].use_batch_p2p_comm
        if self._use_batch_p2p_comm and self._overlap_p2p_comm:
            warnings.warn(
                "non_batch_p2p_comm should be enabled when overlap_p2p_comm is activated, setting non_batch_p2p_comm=True."
            )
            self._use_batch_p2p_comm = False

        logger.info(
            f"dp_comm_overlap {self._dp_comm_overlap}; \
            sharding_comm_overlap {self._sharding_comm_overlap}; \
            sharding_split_param {self._sharding_split_param};"
        )

        self._profiling = self._strategy.hybrid_configs["pp_configs"].profiling
        self._records = []
        self._record_format = (
            '"name": "{}{}", "cat": "pipeline timeline", "ph": {}, "pid": 0, "tid": '
            + str(self.stage_id + 1)
            + ', "ts": {}, "cname": "{}"'
        )
        self._forward_color = "thread_state_running"  # RGB: 126, 200, 148
        self._backward_color = "rail_idle"  # RGB: 238, 142, 0
        if self._profiling:
            logger.info(
                "If enable pp profiling, the max training steps should be restricted "
                "to a reasonable value (such as 5) to avoid generating large profile files. "
                "The profiler will generate a profile file 'profile_record_tmp_file_for_rank_*' "
                "for each rank. Users should gather all profile files for one entire pipeline "
                "to one node (rank 0 is recommended) to get the full view of the pipeline profile. "
                "[DONT CHANGE THE NAME OF THE PROFILE FILES!]. "
                "Then get the profile parser from this url: "
                "https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/distributed/fleet/meta_parallel/pp_utils/profiler_helper.py "
                "and save the script to the same directory of all profile files."
                "Parse those files by this command: `python profiler_helper.py`. "
                "After parsing, a new file 'pipeline_profile.json' will be generated. "
                "Users can inspect this file by chrome://tracing website."
            )

        if self._dp_comm_overlap:
            assert self.use_data_parallel and self.num_stages > 1

        if self._sharding_comm_overlap:
            assert self.use_sharding_parallel and self.num_stages > 1

        assert not (
            self._dp_comm_overlap and self._sharding_comm_overlap
        ), "Cannot use dp pp overlap and sharding pp overlap at the same time."

        self._chunk_2_comm_buffers = defaultdict(list)
        self._comm_overlap = (
            self._dp_comm_overlap or self._sharding_comm_overlap
        )

        if self._enable_timer:
            if not timer.is_timer_initialized():
                timer.set_timers()
            self.timers = timer.get_timers()

        p2p.initialize_p2p_groups(
            hcg,
            self._enable_partial_send_recv,
            self._enable_timer,
        )

        # construct pipeline meta info
        self._p2p_helper = p2p.P2pHelper(self._using_cache)

        self.global_rank = self._hcg.get_global_rank()
        self.micro_batch_id = 0

        # default loss function index
        self.loss_fn_idx = 0

        self._compute_loss = True
        self.callbacks = pipeline_parallel_callbacks_

        logger.info(
            f"Pipeline Info -- num_stages: {self.num_stages}, stage_id: {self.stage_id}"
        )

        if self.use_model_parallel:
            logger.info("start broadcast mp parameters")
            broadcast_mp_parameters(self._layers, self._hcg)

        if self.use_sep_parallel:
            logger.info("start broadcast sep parameters")
            broadcast_sep_parameters(self._layers, self._hcg)

        if self.use_sharding_parallel:
            logger.info("start broadcast sharding parameters")
            broadcast_sharding_parameters(self._layers, self._hcg)

        if self.use_data_parallel:
            logger.info("start broadcast dp parameters")
            broadcast_dp_parameters(self._layers, self._hcg)

        if self._dp_comm_overlap:
            self.register_allreduce_overlap_hook(
                self._layers, self.dp_group, self.accumulate_steps, True
            )

        self.processed_steps = 0

        self._init_user_hooks()
        # only support user hooks during training
        self.user_hooks_enabled = True

    def register_hook(
        self, location: PipelineParallelMicroStepLocations, hook: Callable
    ):
        self.callbacks.register_hook(location, hook)

    def _init_user_hooks(self):
        self._init_user_forward_backward_hooks()
        self._init_user_bubble_hooks()

    def _init_user_forward_backward_hooks(self):
        # initialize forward hooks
        self.forward_hooks = PipelineHook()
        self.forward_hooks.set_hooks_capacity(
            (
                self._virtual_pp_world_size
                if self._virtual_pp_world_size is not None
                else 1
            )
            * self.accumulate_steps
        )

        # initialize backward hooks
        self.backward_hooks = PipelineHook()
        self.backward_hooks.set_hooks_capacity(
            (
                self._virtual_pp_world_size
                if self._virtual_pp_world_size is not None
                else 1
            )
            * self.accumulate_steps
        )

    def _init_user_bubble_hooks(self):
        # (TODO:gexiao) support bubble hooks if needed
        self.bubble_hooks = None
        # self.bubble_hooks = PipelineHook()
        # self.bubble_hooks.set_hooks_capacity(2 * self.num_stages - 2)

    def _reset_user_hooks_status(self):
        if self.bubble_hooks:
            self.bubble_hooks.reset_current_id()
        if self.forward_hooks:
            self.forward_hooks.reset_current_id()
        if self.backward_hooks:
            self.backward_hooks.reset_current_id()

    def _check_user_hooks_status_at_step_end(self):
        if not self.user_hooks_enabled:
            return
        expected_bubble_step = 2 * self.num_stages - 2
        expected_forward_step = (
            self._virtual_pp_world_size
            if self._virtual_pp_world_size is not None
            else 1
        ) * self.accumulate_steps
        expected_backward_step = (
            self._virtual_pp_world_size
            if self._virtual_pp_world_size is not None
            else 1
        ) * self.accumulate_steps

        if self.bubble_hooks:
            assert (
                self.bubble_hooks.current_id
            ) == expected_bubble_step, f"bubble hooks status is not correct, current id is {self.bubble_hooks.current_id}, expected id is {expected_bubble_step}"
        if self.forward_hooks:
            assert (
                self.forward_hooks.current_id
            ) == expected_forward_step, f"forward hooks status is not correct, current id is {self.forward_hooks.current_id}, expected id is {expected_forward_step}"
        if self.backward_hooks:
            assert (
                self.backward_hooks.current_id
            ) == expected_backward_step, f"backward hooks status is not correct, current id is {self.backward_hooks.current_id}, expected id is {expected_backward_step}"

    def register_bubble_pipeline_parallel_hook(
        self, location: int, hook: Callable
    ):
        """
        Registering bubble hooks for pipeline parallelism.
        """
        if not self.bubble_hooks:
            raise ValueError("Bubble hooks are not supported yet.")
        self.bubble_hooks.register_hook(location, hook)

    def register_forward_pipeline_parallel_hook(
        self, location: int, hook: Callable
    ):
        """
        Registering forward hooks for pipeline parallelism.
        """
        if not self.forward_hooks:
            raise ValueError("Forward hooks are not supported yet.")
        self.forward_hooks.register_hook(location, hook)

    def register_backward_pipeline_parallel_hook(
        self, location: int, hook: Callable
    ):
        """
        Registering backward hooks for pipeline parallelism.
        """
        if not self.backward_hooks:
            raise ValueError("Backward hooks are not supported yet.")
        self.backward_hooks.register_hook(location, hook)

    @property
    def bubble_pipeline_parallel_hook_capacity(self):
        capacity = 0
        if self.bubble_hooks:
            capacity = self.bubble_hooks.hooks_capacity
        return capacity

    @property
    def forward_pipeline_parallel_hook_capacity(self):
        capacity = 0
        if self.forward_hooks:
            capacity = self.forward_hooks.hooks_capacity
        return capacity

    @property
    def backward_pipeline_parallel_hook_capacity(self):
        capacity = 0
        if self.backward_hooks:
            capacity = self.backward_hooks.hooks_capacity
        return capacity

    def is_pipeline_first_stage(self, ignore_virtual=False):
        if not ignore_virtual:
            if self._virtual_pp_world_size is not None:
                assert self._virtual_pp_rank is not None
                if self._virtual_pp_rank != 0:
                    return False
        assert self._real_pp_rank is not None
        return self._real_pp_rank == 0

    def is_pipeline_last_stage(self, ignore_virtual=False):
        if not ignore_virtual:
            if self._virtual_pp_world_size is not None:
                assert self._virtual_pp_rank is not None
                if self._virtual_pp_rank != (self._virtual_pp_world_size - 1):
                    return False
        assert self._real_pp_rank is not None
        assert self._real_pp_world_size is not None
        return self._real_pp_rank == (self._real_pp_world_size - 1)

    def set_virtual_pipeline_rank(self, rank):
        self._virtual_pp_rank = rank

    def fused_gradient(
        self, model, comm_group, acc_steps, dp, group_size=128 * 1024 * 1024
    ):
        if model.get_num_virtual_stages() > 1:
            models = model.get_model_chunks()
        else:
            models = [model]

        act = get_action(dp, self._sharding_split_param)

        if act == HOOK_ACTION.REDUCE:
            assert hasattr(self, "optimizer")
            assert hasattr(self.optimizer, "_param2rank")
            _param2rank = self.optimizer._param2rank

        for chunk_idx, model in enumerate(models):
            # For virtual pipeline. Will separate parameters in different chunk into
            # different groups to get the best performance.

            fused_parameter_group = {}
            parameter_list = [
                p for p in model.parameters() if not p.stop_gradient
            ]
            if len(parameter_list) < 1:
                return

            if act == HOOK_ACTION.REDUCE:
                # Sort parameters for sharding, since they have different dst rank
                for p in parameter_list:
                    assert p.name in _param2rank
                    dst_rank = _param2rank[p.name]
                    if dst_rank in fused_parameter_group:
                        fused_parameter_group[dst_rank].append(p)
                    else:
                        fused_parameter_group[dst_rank] = [p]
            else:
                fused_parameter_group[-1] = parameter_list

            for dst in fused_parameter_group:
                parameter_list = fused_parameter_group[dst]
                if act == HOOK_ACTION.REDUCE:
                    # parse the relative dst rank to absolute dst rank for sharding
                    dst = comm_group.ranks[dst]
                var_groups = assign_group_by_size(parameter_list, group_size)

                for group_idx, parameters in var_groups.items():
                    buffer = FusedCommBuffer(
                        group_idx,
                        parameters,
                        comm_group,
                        acc_steps,
                        act,
                        dst,
                        release_grads=self._release_gradients,
                    )
                    self._chunk_2_comm_buffers[chunk_idx].append(buffer)

        return self._chunk_2_comm_buffers

    def bw_hook_func(self, buffer, param):
        @paddle.autograd.no_grad()
        def fused_allreduce(*_):
            buffer.add_grad(param)

        return fused_allreduce

    def register_allreduce_overlap_hook(
        self, model, comm_group, acc_steps, dp, group_size=128 * 1024 * 1024
    ):
        # register hook
        self.fused_gradient(model, comm_group, acc_steps, dp, group_size)
        for _, buffers in self._chunk_2_comm_buffers.items():
            for buffer in buffers:
                for param in buffer._params:
                    param._register_backward_hook(
                        self.bw_hook_func(buffer, param)
                    )

    def timer_printer(self):
        if not self._enable_timer:
            return
        all_flag_names = self.timers.timers.keys()
        self.timers.log(all_flag_names)

    def _record_stamp(self, name, step, phase, color):
        if self._profiling:
            paddle.device.synchronize()
            self._records.append(
                '{'
                + self._record_format.format(
                    name,
                    step,
                    phase,
                    int(time.time() * 1000),
                    color,
                )
                + '}'
            )

    def _flush_records(self):
        if self._profiling:
            with open(
                f'./profile_record_tmp_file_for_rank_{self.global_rank}',
                'a+',
            ) as f:
                for record in self._records:
                    f.write(record + '\n')
            self._records = []

    def forward_backward_pipeline(
        self,
        data,
        scaler=None,
        static_scheduler=False,
        return_micro_batch_loss=False,
    ):
        # use the 1f1b scheduling strategy.
        # this strategy is inspired by:
        # https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/schedules.py

        self._reset_user_hooks_status()
        # no _forward_only mode
        self.user_hooks_enabled = True

        if self.processed_steps < g_profile_pipeline_details_steps:
            get_sync_logger().info("start forward_backward_pipeline")
        if static_scheduler:
            assert (
                not self._profiling
            ), "While _profiling, static scheduler is not available"
            if data is not None:
                warnings.warn(
                    "Static scheduler run won't real run the model, but data has been provided"
                )
            logger.info(
                "enable static_scheduler will return the pp schedule instead of the loss"
            )
            schedule = ""

        self.scaler = scaler

        # store total loss of entire batch
        self.total_loss = None

        # store data id for micro_batch
        self.micro_batch_id = 0

        startup_steps = self.num_stages - self.stage_id - 1
        startup_steps = min(startup_steps, self.accumulate_steps)
        steady_steps = self.accumulate_steps - startup_steps

        input_buffers = []
        output_buffers = []

        micro_dataset = self._wrap_data(data)

        for step_id in range(startup_steps):
            if static_scheduler:
                schedule += f"f{step_id};"
                logger.info(f"forward step for micro step {step_id}")
                continue
            input_tensor = self._p2p_helper.recv_forward(
                self.is_pipeline_first_stage(),
                batch_p2p_comm=self._use_batch_p2p_comm,
            )

            self._record_stamp("F", step_id, '"B"', self._forward_color)
            output_tensor = self._forward_step(
                input_tensor, micro_dataset, step_id=step_id
            )
            self._record_stamp("F", step_id, '"E"', self._forward_color)
            self._p2p_helper.send_forward(
                output_tensor,
                self.is_pipeline_last_stage(),
                batch_p2p_comm=self._use_batch_p2p_comm,
            )

            input_buffers.append(input_tensor)
            output_buffers.append(output_tensor)

            if not self.is_pipeline_last_stage():
                self._release_output(output_tensor)

        if steady_steps > 0 and not static_scheduler:
            input_tensor = self._p2p_helper.recv_forward(
                self.is_pipeline_first_stage(),
                batch_p2p_comm=self._use_batch_p2p_comm,
            )

        for i in range(steady_steps):
            if static_scheduler:
                schedule += f"f{startup_steps + i};"
                schedule += f"b{i};"
                logger.info(f"forward step for micro step {startup_steps + i}")
                logger.info(f"backward step for micro step {i}")
                continue
            last_iter = i == (steady_steps - 1)

            self._record_stamp(
                "F", startup_steps + i, '"B"', self._forward_color
            )
            output_tensor = self._forward_step(
                input_tensor, micro_dataset, step_id=startup_steps + i
            )
            self._record_stamp(
                "F", startup_steps + i, '"E"', self._forward_color
            )

            output_tensor_grad = self._p2p_helper.send_forward_recv_backward(
                output_tensor,
                self.is_pipeline_last_stage(),
                batch_p2p_comm=self._use_batch_p2p_comm,
            )

            input_buffers.append(input_tensor)
            output_buffers.append(output_tensor)

            if not self.is_pipeline_last_stage():
                self._release_output(output_tensor)

            input_tensor, output_tensor = input_buffers.pop(
                0
            ), output_buffers.pop(0)

            self._record_stamp("B", i, '"B"', self._backward_color)
            input_tensor_grad = self._backward_step(
                input_tensor, output_tensor, output_tensor_grad, step_id=i
            )
            self._record_stamp("B", i, '"E"', self._backward_color)

            if last_iter:
                input_tensor = None
                self._p2p_helper.send_backward(
                    input_tensor_grad,
                    self.is_pipeline_first_stage(),
                    batch_p2p_comm=self._use_batch_p2p_comm,
                )
            else:
                input_tensor = self._p2p_helper.send_backward_recv_forward(
                    input_tensor_grad,
                    self.is_pipeline_first_stage(),
                    batch_p2p_comm=self._use_batch_p2p_comm,
                )

        for i in range(startup_steps):
            if static_scheduler:
                schedule += f"b{steady_steps + i};"
                logger.info(f"backward step for micro step {steady_steps + i}")
                continue
            input_tensor = input_buffers.pop(0)
            output_tensor = output_buffers.pop(0)

            output_tensor_grad = self._p2p_helper.recv_backward(
                self.is_pipeline_last_stage(),
                batch_p2p_comm=self._use_batch_p2p_comm,
            )

            self._record_stamp(
                "B", steady_steps + i, '"B"', self._backward_color
            )
            input_tensor_grad = self._backward_step(
                input_tensor,
                output_tensor,
                output_tensor_grad,
                step_id=steady_steps + i,
            )
            self._record_stamp(
                "B", steady_steps + i, '"E"', self._backward_color
            )
            self._p2p_helper.send_backward(
                input_tensor_grad,
                self.is_pipeline_first_stage(),
                batch_p2p_comm=self._use_batch_p2p_comm,
            )

        if static_scheduler:
            return schedule

        self._flush_records()

        if self._comm_overlap:
            assert (
                len(self._chunk_2_comm_buffers) > 0
            ), "comm buffers should be created"
            for _, buffers in self._chunk_2_comm_buffers.items():
                for buffer in buffers:
                    buffer.scale_grads()

        if self._enable_timer:
            self.timers("allreduce_shared_weight_gradients").start()
        self._layers.allreduce_shared_weight_gradients()
        if self._enable_timer:
            self.timers("allreduce_shared_weight_gradients").stop()
            self.timers("broadcast_final_loss").start()
        with paddle.amp.auto_cast(enable=False):
            train_loss = self._broadcast_final_loss(return_micro_batch_loss)
        if self._enable_timer:
            self.timers("broadcast_final_loss").stop()

        if self._clear_every_step_cache:
            self._p2p_helper.clear_meta_cache()

        self.timer_printer()

        if self.processed_steps < g_profile_pipeline_details_steps:
            get_sync_logger().info("end forward_backward_pipeline")
        self.processed_steps += 1
        self._check_user_hooks_status_at_step_end()
        return train_loss

    def register_sharding_comm_overlap_hook(self, optimizer):
        """for delayed hook register until we get optimizer"""
        assert isinstance(
            optimizer, HybridParallelOptimizer
        ), 'optimizer should be HybridParallelOptimizer subclass.'
        self.optimizer = optimizer
        if self._sharding_comm_overlap and len(self._chunk_2_comm_buffers) == 0:
            self.register_allreduce_overlap_hook(
                self._layers, self.sharding_group, self.accumulate_steps, False
            )

    def _prepare_training(self, data, optimizer, lr_scheduler):
        # reset the virtual pp rank for each run
        self.set_virtual_pipeline_rank(0)

        assert isinstance(
            optimizer, HybridParallelOptimizer
        ), 'optimizer should be HybridParallelOptimizer subclass.'

        assert (
            framework._dygraph_tracer()._has_grad
        ), 'Please enable the generation of gradients.'

        if self.is_pipeline_first_stage(
            ignore_virtual=True
        ) or self.is_pipeline_last_stage(ignore_virtual=True):
            assert (
                data is not None
            ), "For the first and the last stage, the data must be set."
        else:
            data = None

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self._layers.train()
        self.register_sharding_comm_overlap_hook(optimizer)

        return data

    def _wrap_data(self, data):
        """
        for backward compatibility, wrap data to Fake FakeMicroDataset if it is of type list or tuple
        """
        if (not isinstance(data, tuple)) and (not isinstance(data, list)):
            return data

        micro_dataset = FakeMicroDataset(
            data,
            self.is_pipeline_first_stage(ignore_virtual=True),
            self.is_pipeline_last_stage(ignore_virtual=True),
            self.accumulate_steps,
            self.micro_batch_size,
        )
        return micro_dataset

    def train_batch(
        self,
        data,
        optimizer,
        lr_scheduler=None,
        scaler=None,
        loss_fn_idx=0,
        return_micro_batch_loss=False,
    ):
        data = self._prepare_training(data, optimizer, lr_scheduler)

        # check loss_fn_idx is valid and loss_fn exists
        assert (
            loss_fn_idx in range(len(self._layers._loss_fn))
            and self._layers._loss_fn[loss_fn_idx] is not None
        ), f"loss function {loss_fn_idx} should exist to compute loss"
        self.loss_fn_idx = loss_fn_idx

        # 1f1b scheduler for pipeline parallel
        train_loss = self.forward_backward_pipeline(
            data, scaler, return_micro_batch_loss=return_micro_batch_loss
        )

        # optimizer
        with paddle.amp.auto_cast(enable=False):
            self._optimizer_step()

        return train_loss

    def eval_batch(self, data, compute_loss=False, loss_fn_idx=0):
        self.user_hooks_enabled = False
        # reset the virtual pp rank for each run
        self.set_virtual_pipeline_rank(0)

        self._layers.eval()
        self._compute_loss = compute_loss

        # store data id for micro_batch
        self.micro_batch_id = 0

        # store total loss of entire batch
        self.total_loss = None

        # check loss_fn_idx is valid and loss_fn exists
        assert (
            loss_fn_idx in range(len(self._layers._loss_fn))
            and self._layers._loss_fn[loss_fn_idx] is not None
        ), f"loss function {loss_fn_idx} should exist to compute loss"
        self.loss_fn_idx = loss_fn_idx

        startup_steps = self.num_stages - self.stage_id - 1
        startup_steps = min(startup_steps, self.accumulate_steps)
        steady_steps = self.accumulate_steps - startup_steps

        input_buffers = []
        output_buffers = []

        # convert to micro dataset
        micro_dataset = self._wrap_data(data)

        for step_id in range(startup_steps):
            input_tensor = self._p2p_helper.recv_forward(
                self.is_pipeline_first_stage(),
                batch_p2p_comm=self._use_batch_p2p_comm,
            )

            output_tensor = self._forward_step(
                input_tensor, micro_dataset, step_id=None
            )
            self._p2p_helper.send_forward(
                output_tensor,
                self.is_pipeline_last_stage(),
                skip_check_meta=True,
                batch_p2p_comm=self._use_batch_p2p_comm,
            )

            input_buffers.append(input_tensor)
            output_buffers.append(output_tensor)

        if steady_steps > 0:
            input_tensor = self._p2p_helper.recv_forward(
                self.is_pipeline_first_stage(),
                batch_p2p_comm=self._use_batch_p2p_comm,
            )

        for i in range(steady_steps):
            last_iter = i == (steady_steps - 1)

            output_tensor = self._forward_step(
                input_tensor, micro_dataset, step_id=None
            )
            self._p2p_helper.send_forward(
                output_tensor,
                self.is_pipeline_last_stage(),
                skip_check_meta=True,
                batch_p2p_comm=self._use_batch_p2p_comm,
            )

            input_buffers.append(input_tensor)
            output_buffers.append(output_tensor)

            if not last_iter:
                input_tensor = self._p2p_helper.recv_forward(
                    self.is_pipeline_first_stage(),
                    batch_p2p_comm=self._use_batch_p2p_comm,
                )

        if self._compute_loss:
            self.train_loss = self._broadcast_final_loss()
        else:
            self.train_loss = output_buffers

        return self.train_loss

    def _forward_step(
        self, input_tensor, micro_dataset, chunk_id=None, step_id=None
    ):
        if self.user_hooks_enabled:
            self.forward_hooks.run_hook()
        if self.processed_steps < g_profile_pipeline_details_steps:
            get_sync_logger().info("Before forward_step")
        if self._enable_timer:
            self.timers("forward_step").start()
        if self.is_pipeline_first_stage():
            input_tensor = next(micro_dataset)[0]
            self._check_micro_batch_data_valid(input_tensor)

        assert chunk_id is None or isinstance(chunk_id, int)

        self.callbacks.on_location(
            PipelineParallelMicroStepLocations.FORWARD_BEGIN,
            input_tensor=input_tensor,
            step_id=step_id,
        )
        output_tensor = self._layers.forward(input_tensor, chunk_id=chunk_id)
        self.callbacks.on_location(
            PipelineParallelMicroStepLocations.FORWARD_END,
            input_tensor=input_tensor,
            output_tensor=output_tensor,
            step_id=step_id,
        )

        if self.is_pipeline_last_stage():
            # train calculate loss for train
            if self._compute_loss:
                assert (
                    self._layers._loss_fn[self.loss_fn_idx] is not None
                ), "loss function should exist to compute loss"
                labels = next(micro_dataset)[1]
                self._check_micro_batch_data_valid(labels)
                for idx, loss_fn in enumerate(self._layers._loss_fn):
                    loss_tensor = loss_fn(output_tensor, labels)
                    assert isinstance(
                        loss_tensor, paddle.Tensor
                    ), "Currently, loss_fn should obtain Paddle.Tensor dtype"

                    with paddle.amp.auto_cast(enable=False):
                        if (
                            self.accumulate_steps > 1
                            and not self._delay_scale_loss
                        ):
                            loss_tensor = loss_tensor / self.accumulate_steps

                        if self.total_loss is None:
                            self.total_loss = []
                        # when self.total_loss length is less than idx, append a new tensor
                        if len(self.total_loss) <= idx:
                            self.total_loss.append([])
                        self.total_loss[idx].append(loss_tensor.detach())

                    if idx == self.loss_fn_idx:
                        backward_loss_tensor = loss_tensor
        if self.is_pipeline_first_stage() or self.is_pipeline_last_stage():
            # Only increase micro batch id at virtual first/last pp stage.
            # The micro batch id is used to load data, therefore, only increase it when load data.
            self.micro_batch_id += 1
        if self._enable_timer:
            self.timers("forward_step").stop()
        if self.processed_steps < g_profile_pipeline_details_steps:
            get_sync_logger().info("After forward_step")
        if self.is_pipeline_last_stage() and self._compute_loss:
            return backward_loss_tensor
        return output_tensor

    def _backward_step(
        self, input_tensor, output_tensor, output_tensor_grad, step_id=None
    ):
        if self.user_hooks_enabled:
            self.backward_hooks.run_hook()
        if self._enable_timer:
            self.timers("backward_step").start()
        if self.processed_steps < g_profile_pipeline_details_steps:
            get_sync_logger().info("Before backward_step")
        with paddle.amp.auto_cast(enable=False):
            self.callbacks.on_location(
                PipelineParallelMicroStepLocations.BACKWARD_BEGIN,
                input_tensor=input_tensor,
                output_tensor=output_tensor,
                output_tensor_grad=output_tensor_grad,
                step_id=step_id,
            )
            if self.is_pipeline_last_stage():
                assert output_tensor_grad is None
                # In align mode, we scale the grad directly after forward
                if paddle.distributed.in_auto_parallel_align_mode():
                    output_tensor = output_tensor / _get_align_mode_scale()
                if self.scaler:
                    paddle.autograd.backward(self.scaler.scale(output_tensor))
                else:
                    paddle.autograd.backward(output_tensor)
            else:
                if isinstance(output_tensor, tuple):
                    outputs = [t for t in output_tensor if not t.stop_gradient]
                    assert len(outputs) == len(output_tensor_grad)
                    paddle.autograd.backward(
                        tensors=outputs,
                        grad_tensors=list(output_tensor_grad),
                    )
                else:
                    paddle.autograd.backward(
                        tensors=[output_tensor],
                        grad_tensors=[output_tensor_grad],
                    )

            input_tensor_grad = None
            if input_tensor is not None:
                if isinstance(input_tensor, tuple):
                    input_tensor_grad = tuple(
                        [t.grad for t in input_tensor if not t.stop_gradient]
                    )
                else:
                    input_tensor_grad = input_tensor.grad
            if self._enable_timer:
                self.timers("backward_step").stop()
            self.callbacks.on_location(
                PipelineParallelMicroStepLocations.BACKWARD_END,
                input_tensor=input_tensor,
                output_tensor=output_tensor,
                input_tensor_grad=input_tensor_grad,
                output_tensor_grad=output_tensor_grad,
                step_id=step_id,
            )

            if self.processed_steps < g_profile_pipeline_details_steps:
                get_sync_logger().info("After backward_step")
            return input_tensor_grad

    def _check_micro_batch_data_valid(self, micro_batch_data):
        if isinstance(micro_batch_data, (tuple, list)):
            for data in micro_batch_data:
                self._check_micro_batch_data_valid(data)
        elif micro_batch_data is not None:
            assert isinstance(micro_batch_data, paddle.Tensor)

    def _broadcast_final_loss(self, return_micro_batch_loss=False):
        # Since the last backward run in interleave will set the virtual rank to 0,
        # here we need to check last stage ignoring virtual stage.
        if self.is_pipeline_last_stage(ignore_virtual=True):
            assert (
                self.total_loss is not None
            ), "train_batch() in last stage should obtain valid loss"
            losses = []
            for idx in range(len(self._layers._loss_fn)):
                self.total_loss[idx] = paddle.to_tensor(self.total_loss[idx])
                if not return_micro_batch_loss:
                    # TODO(shenliang03): it will use mean/sum to calculate loss
                    tmp = paddle.zeros_like(self.total_loss[idx][0])
                    for loss in self.total_loss[idx]:
                        tmp += loss.detach()
                    if not self._delay_scale_loss:
                        losses.append(tmp)
                    else:
                        losses.append(tmp / self.accumulate_steps)
                else:
                    losses.append(self.total_loss[idx].detach())

            for idx in range(len(self._layers._loss_fn)):
                is_fp32 = (
                    paddle.full([], 1, 'int64')
                    if losses[idx].dtype == paddle.float32
                    else paddle.full([], 0, 'int64')
                )
                paddle.distributed.broadcast(
                    is_fp32,
                    src=self.global_rank,
                    sync_op=True,
                    group=self.pp_group,
                )
                paddle.distributed.broadcast(
                    losses[idx],
                    src=self.global_rank,
                    sync_op=True,
                    group=self.pp_group,
                )
        else:
            losses = []
            for idx in range(len(self._layers._loss_fn)):
                is_fp32 = paddle.full([], 1, 'int64')
                paddle.distributed.broadcast(
                    is_fp32,
                    src=self._hcg.get_rank_from_stage(self.num_stages - 1),
                    sync_op=True,
                    group=self.pp_group,
                )
                if return_micro_batch_loss:
                    loss_shape = [self.accumulate_steps]
                else:
                    loss_shape = [1]
                losses.append(
                    paddle.zeros(shape=loss_shape, dtype="float32")
                    if is_fp32.item()
                    else paddle.zeros(shape=loss_shape, dtype="float16")
                )
                paddle.distributed.broadcast(
                    losses[idx],
                    src=self._hcg.get_rank_from_stage(self.num_stages - 1),
                    sync_op=True,
                    group=self.pp_group,
                )
        return losses[0] if len(losses) == 1 else losses

    def _optimizer_step(self):
        if self._delay_scale_loss:
            for p in self._layers.parameters():
                if hasattr(p, "main_grad") and p.main_grad is not None:
                    assert p.grad is None
                    p.main_grad = p.main_grad.scale(1.0 / self.accumulate_steps)
                elif p.grad is not None:
                    p.grad = p.grad.scale(1.0 / self.accumulate_steps)

        if self.scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        if self._release_gradients:
            self.optimizer.clear_grad(set_to_zero=False)
            for _, buffers in self._chunk_2_comm_buffers.items():
                for buffer in buffers:
                    buffer._clear_grad_storage()
        else:
            self.optimizer.clear_grad()

        if self.lr_scheduler:
            self.lr_scheduler.step()

    def _release_output(self, output):
        def can_free(t):
            return (
                t is not None
                and isinstance(t, paddle.Tensor)
                and t._is_initialized()
                and t.inplace_version == 0
            )

        if isinstance(output, (tuple, list)):
            for t in output:
                if can_free(t):
                    t._clear_dataptr()

        elif can_free(output):
            output._clear_dataptr()

    def get_static_scheduler(self):
        return self.forward_backward_pipeline(data=None, static_scheduler=True)


class PipelineParallelWithInterleave(PipelineParallel):
    # pipeline parallel with interleave scheduler

    def __init__(self, layers, hcg, strategy):
        super().__init__(layers=layers, hcg=hcg, strategy=strategy)
        self._record_format = (
            '"name": "{}{}_VP{}", "cat": "virtual pipeline timeline", "ph": {}, "pid": 0, "tid": '
            + str(self.stage_id + 1)
            + ', "ts": {}, "cname": "{}"'
        )
        self._forward_colors = [
            "thread_state_running",  # RGB: 126, 200, 148
            "thread_state_unknown",  # RGB: 199, 155, 125
        ]
        self._backward_colors = [
            "rail_load",  # RGB: 13, 168, 97
            "rail_idle",  # RGB: 238, 142, 0
        ]
        # Structures to record the micro step for each layer chunk
        self._forward_micro_step_counter = {}
        self._backward_micro_step_counter = {}

        assert layers.get_num_virtual_stages() > 1

        # setup for interleave scheduler
        self._check_sanity()
        self.num_model_chunks = layers.get_num_virtual_stages()
        self.model_chunks = layers.get_model_chunks()
        assert self.model_chunks is not None
        assert len(self.model_chunks) == self.num_model_chunks
        self._virtual_pp_world_size = self.num_model_chunks
        self._virtual_pp_rank = 0
        self._reset_counter()
        self._best_unbalanced_scheduler = self._strategy.hybrid_configs[
            "pp_configs"
        ].best_unbalanced_scheduler
        if self._best_unbalanced_scheduler:
            assert (
                not self._comm_overlap
            ), "pp best unbalaced scheduler can not run together with dp/sharding overlap"

        self._enable_offload_queue = self._strategy.hybrid_configs[
            "pp_configs"
        ].enable_offload_queue

        # reinit user hook since now we have virtual stages
        self._init_user_hooks()

    def _init_user_bubble_hooks(self):
        # initialize bubble hooks
        self.bubble_hooks = PipelineHook()
        self.bubble_hooks.set_hooks_capacity(2 * self.num_stages - 2)

    def _check_sanity(self):
        assert (
            framework.in_dynamic_mode()
        ), "virtual pipeline stage with interleave only support eager dygraph mode"

        assert (
            self.num_stages > 2
        ), "virtual pipeline must run under pp degree > 2"

        assert (
            self.accumulate_steps >= 2 * self.num_stages
        ), f"accumulate_steps({self.accumulate_steps}) should be greater than or equal to 2 * num_stages({self.num_stages}) for pipeline with interleave"

    def _reset_counter(self):
        for i in range(self.num_model_chunks):
            self._forward_micro_step_counter[i] = 0
            self._backward_micro_step_counter[i] = 0

    def _record_stamp(self, name, step, phase, forward=True):
        if self._profiling:
            paddle.device.synchronize()
            virtual_pp_rank = self._get_virtual_pp_rank(step, forward=forward)
            color_idx = virtual_pp_rank % 2
            # Get the profile color and micro step for current layer chunk
            if forward:
                color = self._forward_colors[color_idx]
                micro_step = self._forward_micro_step_counter[virtual_pp_rank]
                if phase == '"E"':
                    self._forward_micro_step_counter[virtual_pp_rank] += 1
            else:
                color = self._backward_colors[color_idx]
                micro_step = self._backward_micro_step_counter[virtual_pp_rank]
                if phase == '"E"':
                    self._backward_micro_step_counter[virtual_pp_rank] += 1
            self._records.append(
                '{'
                + self._record_format.format(
                    name,
                    micro_step,
                    virtual_pp_rank,
                    phase,
                    int(time.time() * 1000),
                    color,
                )
                + '}'
            )

    def _flush_records(self):
        if self._profiling:
            with open(
                f'./profile_record_tmp_file_for_rank_{self.global_rank}',
                'a+',
            ) as f:
                for record in self._records:
                    f.write(record + '\n')
            self._records = []
            self._reset_counter()

    def _get_virtual_pp_rank(self, micro_step, forward):
        first_chunk_acc = (
            self.accumulate_steps % self.num_stages + self.num_stages
        )
        first_chunk_steps = first_chunk_acc * self.num_model_chunks
        if self._best_unbalanced_scheduler:
            num_group_last_chunk_forward = (
                (micro_step - first_chunk_acc) // self.num_stages
            ) // self.num_model_chunks
            misplace_start = (
                first_chunk_acc
                + self.num_model_chunks
                * self.num_stages
                * num_group_last_chunk_forward
            )
            misplace_end = (
                self.accumulate_steps % self.num_stages
                + num_group_last_chunk_forward * self.num_stages
            ) * self.num_model_chunks + self.num_stages
            forward_virtual_pp_stage = (
                (micro_step - first_chunk_acc) // self.num_stages
            ) % self.num_model_chunks

        if micro_step < first_chunk_steps:
            virtual_pp_stage = micro_step // first_chunk_acc
            if not forward and self._best_unbalanced_scheduler:
                if (
                    micro_step
                    >= first_chunk_acc
                    + (self.num_model_chunks - 1) * self.num_stages
                ):
                    if forward_virtual_pp_stage == self.num_model_chunks - 1:
                        virtual_pp_stage = 0
                    elif (
                        micro_step >= misplace_start
                        and micro_step < misplace_end
                    ):
                        virtual_pp_stage = (
                            micro_step - self.num_stages
                        ) // first_chunk_acc
        else:
            origin_micro_step = micro_step
            micro_step -= first_chunk_steps
            virtual_pp_stage = micro_step % (
                self.num_stages * self.num_model_chunks
            )
            virtual_pp_stage = virtual_pp_stage // self.num_stages
            if not forward and self._best_unbalanced_scheduler:
                total_num_forward_step_from_steady = (
                    first_chunk_acc
                    + (self.accumulate_steps - first_chunk_acc)
                    * self.num_model_chunks
                )
                if (
                    origin_micro_step <= total_num_forward_step_from_steady
                    and forward_virtual_pp_stage == self.num_model_chunks - 1
                ):
                    virtual_pp_stage = 0
                elif (
                    misplace_start <= total_num_forward_step_from_steady
                    and origin_micro_step >= misplace_start
                    and origin_micro_step < misplace_end
                ):
                    if origin_micro_step < first_chunk_steps + self.num_stages:
                        virtual_pp_stage = (
                            origin_micro_step - self.num_stages
                        ) // first_chunk_acc
                    else:
                        virtual_pp_stage = (micro_step - self.num_stages) % (
                            self.num_stages * self.num_model_chunks
                        )
                        virtual_pp_stage = virtual_pp_stage // self.num_stages

        if not forward:
            virtual_pp_stage = self.num_model_chunks - virtual_pp_stage - 1

        return virtual_pp_stage

    def _forward_step_helper(self, micro_dataset, micro_step):
        virtual_pp_rank = self._get_virtual_pp_rank(micro_step, forward=True)
        self.set_virtual_pipeline_rank(virtual_pp_rank)

        # some checkers
        assert hasattr(self, 'input_tensors')
        assert hasattr(self, 'output_tensors')
        if not self._forward_only:
            assert hasattr(self, 'output_tensor_grads')
        assert len(self.input_tensors[virtual_pp_rank]) == (
            len(self.output_tensors[virtual_pp_rank]) + 1
        )
        input_tensor = self.input_tensors[virtual_pp_rank][-1]
        output_tensor = self._forward_step(
            input_tensor, micro_dataset, virtual_pp_rank, step_id=micro_step
        )
        self.output_tensors[virtual_pp_rank].append(output_tensor)

        if self._forward_only:
            # no need to store tensor for backward
            self.input_tensors[virtual_pp_rank].pop()
            self.output_tensors[virtual_pp_rank].pop()

        return output_tensor

    def _overlap_comm_grads(self):
        if self._comm_overlap:
            self._backward_step_count += 1
            sync_step = self._backward_step_count - self.stage_id
            if sync_step > 0 and sync_step % self.num_stages == 0:
                chunk_idx = self._virtual_pp_world_size - (
                    sync_step // self.num_stages
                )
                for buffer in self._chunk_2_comm_buffers[chunk_idx]:
                    buffer.comm_grads()

            if self.stage_id != 0:
                if (
                    self._backward_step_count
                    == self.num_stages * self.num_model_chunks
                ):
                    for buffer in self._chunk_2_comm_buffers[0]:
                        buffer.comm_grads()

    def _sync_overlap_grads(self):
        if self._comm_overlap:
            assert (
                self._backward_step_count
                == self.num_stages * self.num_model_chunks
            ), (
                "backward step count should be equal to accumulate steps * virtual pp world size,"
                f" but get {self._backward_step_count}, excepted result is {self.num_stages * self.num_model_chunks}"
            )

            for _, buffers in self._chunk_2_comm_buffers.items():
                for buffer in buffers:
                    buffer.scale_grads()

    def _backward_step_helper(self, micro_step):
        virtual_pp_rank = self._get_virtual_pp_rank(micro_step, forward=False)
        self.set_virtual_pipeline_rank(virtual_pp_rank)

        # some checkers
        assert hasattr(self, 'input_tensors')
        assert hasattr(self, 'output_tensors')
        assert hasattr(self, 'output_tensor_grads')

        assert (
            len(self.output_tensor_grads[virtual_pp_rank]) > 0
        ), f"output_tensor_grads is empty for virtual_pp_rank {virtual_pp_rank}"

        assert len(self.input_tensors[virtual_pp_rank]) > 0
        assert len(self.output_tensors[virtual_pp_rank]) > 0

        input_tensor = self.input_tensors[virtual_pp_rank].pop(0)
        output_tensor = self.output_tensors[virtual_pp_rank].pop(0)
        output_tensor_grad = self.output_tensor_grads[virtual_pp_rank].pop(0)
        input_tensor_grad = self._backward_step(
            input_tensor, output_tensor, output_tensor_grad, step_id=micro_step
        )

        self._overlap_comm_grads()

        return input_tensor_grad

    def bw_hook_func(self, buffer, param):
        # For pipeline with interleave, we need to add grad to buffer without communication.
        # Use communication where appropriate to avoid dp communication and pp scheduling conflicts.
        # all reduce hook
        @paddle.autograd.no_grad()
        def fused_allreduce(*_):
            buffer.add_grad(param, use_comm=False)

        return fused_allreduce

    def register_allreduce_overlap_hook(self, model, comm_group, acc_steps, dp):
        super().register_allreduce_overlap_hook(
            model, comm_group, acc_steps, dp, group_size=sys.maxsize
        )

    def forward_backward_pipeline(
        self,
        data,
        scaler,
        forward_only=False,
        compute_loss=True,
        static_scheduler=False,
        return_micro_batch_loss=False,
    ):
        self._reset_user_hooks_status()
        if self.processed_steps < g_profile_pipeline_details_steps:
            get_sync_logger().info("start forward_backward_pipeline")
        # use interleave scheduling strategy.
        # this strategy is inspired by:
        # https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/schedules.py
        if not compute_loss:
            assert (
                not forward_only
            ), "compute_loss can only be set to False when forward_only is set to True"

        if static_scheduler:
            assert (
                not forward_only
            ), "static_scheduler only for training not for eval"
            assert (
                not self._profiling
            ), "While _profiling, static scheduler is not available"
            if data is not None:
                warnings.warn(
                    "Static scheduler run won't real run the model, but data has been provided"
                )
            logger.info(
                "enable static_scheduler will return the pp schedule instead of the loss"
            )
            schedule = ""
        # NOTE(shenliang03): Due to ring_exchange for pipeline with interleave, cache should be enabled
        assert (
            self._using_cache
        ), "cache should be enabled for pipeline with interleave"

        # init some attributes for this batch run
        self.scaler = scaler
        self.total_loss = None
        self.micro_batch_id = 0
        self._forward_only = forward_only
        self.user_hooks_enabled = not self._forward_only

        first_chunk_acc = (
            self.accumulate_steps % self.num_stages + self.num_stages
        )
        first_chunk_steps = first_chunk_acc * self.num_model_chunks
        fwd_buffer_queue = queue.Queue()
        bwd_buffer_queue = queue.Queue()
        skip_steps = self.accumulate_steps % self.num_stages
        last_stage_recv_queue = deque()

        left_id = skip_steps
        right_id = left_id + first_chunk_acc * (self.num_model_chunks - 1)

        def _process_fwd_buffer(step_id, tensor):
            if step_id < first_chunk_steps:
                if not self.is_pipeline_last_stage():
                    fwd_buffer_queue.put(tensor)
                if left_id <= step_id < right_id:
                    tensor = fwd_buffer_queue.get()
                else:
                    tensor = None
            else:
                if self.is_pipeline_last_stage():
                    tensor = None
            return tensor

        def _last_stage_need_recv_next(micro_step):
            if micro_step >= first_chunk_acc:
                if len(last_stage_recv_queue) == 0:
                    return False
                else:
                    res = last_stage_recv_queue[0]
                    if micro_step - res[0] < self.num_stages:
                        return False
                    else:
                        return True
            else:
                return False

        def _last_stage_recv_pp_rank(micro_step):
            if micro_step >= first_chunk_acc:
                assert (
                    len(last_stage_recv_queue) != 0
                ), "last_stage_recv_queue can't be empty"
                virtual_pp_stage = (last_stage_recv_queue.popleft())[1]
                return virtual_pp_stage - 1
            else:
                return self.num_model_chunks - 1

        def _process_bwd_buffer(step_id, tensor):
            if self._best_unbalanced_scheduler:
                if not self.is_pipeline_first_stage():
                    bwd_buffer_queue.put(tensor)
                if step_id >= left_id and not bwd_buffer_queue.empty():
                    tensor = bwd_buffer_queue.get()
                else:
                    tensor = None
            else:
                if step_id < first_chunk_steps:
                    if not self.is_pipeline_first_stage():
                        bwd_buffer_queue.put(tensor)
                    if left_id <= step_id < right_id:
                        tensor = bwd_buffer_queue.get()
                    else:
                        tensor = None
                else:
                    if self.is_pipeline_first_stage():
                        tensor = None
            return tensor

        per_stage_accumulate_steps = self.accumulate_steps // self.num_stages
        self._backward_step_count = -(
            first_chunk_steps
            + (per_stage_accumulate_steps - 2)
            * self.num_stages
            * self.num_model_chunks
        )

        # init some data buffers for interleave scheduler
        self.input_tensors = [[] for _ in range(self.num_model_chunks)]
        self.output_tensors = [[] for _ in range(self.num_model_chunks)]
        self.output_tensor_grads = [[] for _ in range(self.num_model_chunks)]

        micro_dataset = self._wrap_data(data)

        num_steps = self.accumulate_steps * self.num_model_chunks
        if forward_only:
            # If only forward, since there is no backward during running, all steps are startup steps
            startup_steps = num_steps
        else:
            # actually startup_steps is calculated from two number:
            # first_forward_cross_to_end = (self.num_stages - self.stage_id - 1) + (self.num_model_chunks - 1) * self.num_stages
            # end_to_first_backward_cross = (self.num_stages - self.stage_id - 1)
            # startup_steps = first_forward_cross_to_end + end_to_first_backward_cross
            startup_steps = (self.num_stages - self.stage_id - 1) * 2
            startup_steps += (self.num_model_chunks - 1) * first_chunk_acc
            startup_steps = min(startup_steps, num_steps)

        steady_steps = num_steps - startup_steps

        for location in range(self.stage_id):
            if self.user_hooks_enabled:
                self.bubble_hooks.run_hook()

        rest_bubble_times = self.num_stages - 1 - self.stage_id

        self.set_virtual_pipeline_rank(0)
        if not static_scheduler:
            self.input_tensors[0].append(
                self._p2p_helper.recv_forward(
                    self.is_pipeline_first_stage(),
                    sync_recv=False,
                    batch_p2p_comm=self._use_batch_p2p_comm,
                )
            )

        fwd_wait_handles = None
        bwd_wait_handles = None

        # run startup steps
        for micro_step in range(startup_steps):
            if fwd_wait_handles is not None:
                for req in fwd_wait_handles:
                    req.wait()

            if static_scheduler:
                virtual_pp_rank = self._get_virtual_pp_rank(
                    micro_step, forward=True
                )
                real_micro_step = self._forward_micro_step_counter[
                    virtual_pp_rank
                ]
                self._forward_micro_step_counter[virtual_pp_rank] += 1
                schedule += f"f{real_micro_step}_vp{virtual_pp_rank};"
                logger.info(
                    f"forward step for {real_micro_step} with virtual pp rank {virtual_pp_rank}"
                )
                continue

            self._record_stamp("F", micro_step, '"B"', forward=True)
            output_tensor = self._forward_step_helper(micro_dataset, micro_step)
            self._record_stamp("F", micro_step, '"E"', forward=True)

            if micro_step >= startup_steps - rest_bubble_times:
                if self.user_hooks_enabled:
                    self.bubble_hooks.run_hook()

            # determine whether recv forward tensor or not
            next_virtual_pp_rank = self._get_virtual_pp_rank(
                micro_step + 1, forward=True
            )
            recv_prev = True
            if self.is_pipeline_first_stage(ignore_virtual=True):
                if next_virtual_pp_rank == 0:
                    # next chunk is the first chunk, not need to pre recv an input tensor
                    recv_prev = False
            # last micro step, no next run
            if micro_step == (num_steps - 1):
                recv_prev = False

            # last stage shouldn't send tensor to downstream
            if self.is_pipeline_last_stage(ignore_virtual=True):
                output_tensor = _process_fwd_buffer(micro_step, output_tensor)

            if not self._overlap_p2p_comm:
                # prepare for the first steady step
                if (
                    micro_step == (startup_steps - 1)
                    and (not forward_only)
                    and steady_steps
                ):
                    input_tensor_grad = None
                    recv_next = True
                    if self.is_pipeline_last_stage(ignore_virtual=True):
                        recv_next = False

                    # the last startup step needs on four direction comm to set up for steady 1f1b
                    (
                        input_tensor,
                        output_tensor_grad,
                    ) = self._p2p_helper.send_forward_backward_recv_forward_backward(
                        output_tensor,
                        input_tensor_grad,
                        recv_prev=recv_prev,
                        recv_next=recv_next,
                        batch_p2p_comm=self._use_batch_p2p_comm,
                        skip_check_meta=not self.training,
                    )
                    # output_tensor_grad is not none if recv_next
                    # append output_tensor_grad no matter none or not
                    self.output_tensor_grads[self.num_model_chunks - 1].append(
                        output_tensor_grad
                    )
                else:
                    input_tensor = self._p2p_helper.send_forward_recv_forward(
                        output_tensor,
                        recv_prev=recv_prev,
                        batch_p2p_comm=self._use_batch_p2p_comm,
                        skip_check_meta=not self.training,
                    )
                # append input_tensor no matter none or not
                self.input_tensors[next_virtual_pp_rank].append(input_tensor)
            else:
                (
                    input_tensor,
                    fwd_wait_handles,
                ) = self._p2p_helper.send_forward_recv_forward(
                    output_tensor,
                    recv_prev=recv_prev,
                    batch_p2p_comm=self._use_batch_p2p_comm,
                    overlap_p2p_comm=True,
                    skip_check_meta=not self.training,
                )
                if (
                    micro_step == (startup_steps - 1)
                    and (not forward_only)
                    and steady_steps
                ):
                    input_tensor_grad = None
                    recv_next = True
                    if self.is_pipeline_last_stage(ignore_virtual=True):
                        recv_next = False

                    (
                        output_tensor_grad,
                        bwd_wait_handles,
                    ) = self._p2p_helper.send_backward_recv_backward(
                        input_tensor_grad,
                        recv_next=recv_next,
                        batch_p2p_comm=self._use_batch_p2p_comm,
                        overlap_p2p_comm=True,
                    )
                    self.output_tensor_grads[self.num_model_chunks - 1].append(
                        output_tensor_grad
                    )

                # append input_tensor no matter none or not
                self.input_tensors[next_virtual_pp_rank].append(input_tensor)
            self._release_output(output_tensor)

        # run 1f1b steady steps
        for micro_step in range(steady_steps):
            if static_scheduler:
                forward_micro_step_id = micro_step + startup_steps
                forward_virtual_pp_rank = self._get_virtual_pp_rank(
                    forward_micro_step_id, forward=True
                )
                backward_micro_step_id = micro_step
                backward_virtual_pp_rank = self._get_virtual_pp_rank(
                    backward_micro_step_id, forward=False
                )
                real_forward_micro_step = self._forward_micro_step_counter[
                    forward_virtual_pp_rank
                ]
                self._forward_micro_step_counter[forward_virtual_pp_rank] += 1
                real_backward_micro_step = self._backward_micro_step_counter[
                    backward_virtual_pp_rank
                ]
                self._backward_micro_step_counter[backward_virtual_pp_rank] += 1
                schedule += (
                    f"f{real_forward_micro_step}_vp{forward_virtual_pp_rank};"
                )
                schedule += (
                    f"b{real_backward_micro_step}_vp{backward_virtual_pp_rank};"
                )
                logger.info(
                    f"forward step for {real_forward_micro_step} with virtual pp rank {forward_virtual_pp_rank}"
                )
                logger.info(
                    f"backward step for {real_backward_micro_step} with virtual pp rank {backward_virtual_pp_rank}"
                )
                continue
            # forward
            forward_micro_step_id = micro_step + startup_steps

            if self._overlap_p2p_comm:
                if fwd_wait_handles is not None:
                    for req in fwd_wait_handles:
                        req.wait()

                self._release_output(output_tensor)
                output_tensor = self._forward_step_helper(
                    micro_dataset, forward_micro_step_id
                )

                forward_virtual_pp_rank = self._get_virtual_pp_rank(
                    forward_micro_step_id, forward=True
                )
                self.set_virtual_pipeline_rank(forward_virtual_pp_rank)
                if self.is_pipeline_last_stage(ignore_virtual=True):
                    output_tensor = _process_fwd_buffer(
                        forward_micro_step_id, output_tensor
                    )

                # determine whether to recv input tensor from upstream
                recv_prev = True
                if self.is_pipeline_first_stage(ignore_virtual=True):
                    next_forward_virtual_pp_rank = self._get_virtual_pp_rank(
                        forward_micro_step_id + 1, forward=True
                    )
                    if next_forward_virtual_pp_rank == 0:
                        # next chunk is the first chunk, not need to pre recv an input tensor
                        recv_prev = False
                else:
                    next_forward_virtual_pp_rank = self._get_virtual_pp_rank(
                        forward_micro_step_id + 1, forward=True
                    )

                # last iteration doesn't need recv from upstream
                if micro_step == (steady_steps - 1):
                    recv_prev = False

                # Send activation tensor to the next stage and receive activation tensor from the
                # previous stage
                (
                    input_tensor,
                    fwd_wait_handles,
                ) = self._p2p_helper.send_forward_recv_forward(
                    output_tensor,
                    recv_prev=recv_prev,
                    batch_p2p_comm=self._use_batch_p2p_comm,
                    overlap_p2p_comm=True,
                    skip_check_meta=not self.training,
                )

                if bwd_wait_handles is not None:
                    for req in bwd_wait_handles:
                        req.wait()

                # backward pass
                backward_micro_step_id = micro_step
                input_tensor_grad = self._backward_step_helper(
                    backward_micro_step_id
                )

                if (
                    self._best_unbalanced_scheduler
                    and self.is_pipeline_last_stage(ignore_virtual=True)
                ):
                    cur_pp_rank = self._get_virtual_pp_rank(
                        backward_micro_step_id, forward=False
                    )
                    if cur_pp_rank != 0:
                        last_stage_recv_queue.append(
                            (backward_micro_step_id, cur_pp_rank)
                        )

                # first stage doesn't send grad to upstream
                backward_virtual_pp_rank = self._get_virtual_pp_rank(
                    backward_micro_step_id, forward=False
                )
                self.set_virtual_pipeline_rank(backward_virtual_pp_rank)
                if self.is_pipeline_first_stage(ignore_virtual=True):
                    input_tensor_grad = _process_bwd_buffer(
                        backward_micro_step_id, input_tensor_grad
                    )

                recv_next = True
                if self.is_pipeline_last_stage(ignore_virtual=True):
                    if self._best_unbalanced_scheduler:
                        next_backward_virtual_pp_rank = (
                            self._get_virtual_pp_rank(
                                backward_micro_step_id + 1,
                                forward=False,
                            )
                        )
                        if self.is_pipeline_last_stage(ignore_virtual=True):
                            recv_next = _last_stage_need_recv_next(
                                backward_micro_step_id + 1
                            )
                    else:
                        next_backward_virtual_pp_rank = (
                            self._get_virtual_pp_rank(
                                backward_micro_step_id + 1,
                                forward=False,
                            )
                        )
                        if next_backward_virtual_pp_rank == (
                            self.num_model_chunks - 1
                        ):
                            # next chunk is the last chunk, not need to pre recv an output tensor grad
                            recv_next = False
                else:
                    next_backward_virtual_pp_rank = self._get_virtual_pp_rank(
                        backward_micro_step_id + 1,
                        forward=False,
                    )

                (
                    output_tensor_grad,
                    bwd_wait_handles,
                ) = self._p2p_helper.send_backward_recv_backward(
                    input_tensor_grad,
                    recv_next=recv_next,
                    batch_p2p_comm=self._use_batch_p2p_comm,
                    overlap_p2p_comm=True,
                )
            else:
                self._record_stamp(
                    "F", forward_micro_step_id, '"B"', forward=True
                )
                output_tensor = self._forward_step_helper(
                    micro_dataset, forward_micro_step_id
                )
                self._record_stamp(
                    "F", forward_micro_step_id, '"E"', forward=True
                )

                # backward
                backward_micro_step_id = micro_step
                self._record_stamp(
                    "B", backward_micro_step_id, '"B"', forward=False
                )
                input_tensor_grad = self._backward_step_helper(
                    backward_micro_step_id
                )
                self._record_stamp(
                    "B", backward_micro_step_id, '"E"', forward=False
                )
                if (
                    self._best_unbalanced_scheduler
                    and self.is_pipeline_last_stage(ignore_virtual=True)
                ):
                    cur_pp_rank = self._get_virtual_pp_rank(
                        backward_micro_step_id, forward=False
                    )
                    if cur_pp_rank != 0:
                        last_stage_recv_queue.append(
                            (backward_micro_step_id, cur_pp_rank)
                        )

                # four directions comm
                # send output tensor to downstream
                # send input tensor grad to upstream
                # recv input tensor from upstream
                # recv output tensor grad from downstream

                # last stage doesn't send rst to downstream
                forward_virtual_pp_rank = self._get_virtual_pp_rank(
                    forward_micro_step_id, forward=True
                )
                self.set_virtual_pipeline_rank(forward_virtual_pp_rank)
                if self.is_pipeline_last_stage(ignore_virtual=True):
                    output_tensor = _process_fwd_buffer(
                        forward_micro_step_id, output_tensor
                    )

                # first stage doesn't send grad to upstream
                backward_virtual_pp_rank = self._get_virtual_pp_rank(
                    backward_micro_step_id, forward=False
                )
                self.set_virtual_pipeline_rank(backward_virtual_pp_rank)
                if self.is_pipeline_first_stage(ignore_virtual=True):
                    input_tensor_grad = _process_bwd_buffer(
                        backward_micro_step_id, input_tensor_grad
                    )

                # determine whether to recv input tensor from upstream
                recv_prev = True
                next_forward_virtual_pp_rank = self._get_virtual_pp_rank(
                    forward_micro_step_id + 1, forward=True
                )
                if self.is_pipeline_first_stage(ignore_virtual=True) and (
                    next_forward_virtual_pp_rank == 0
                ):
                    # first pp stage and first virtual stage
                    recv_prev = False

                # last iteration doesn't need recv from upstream
                if micro_step == (steady_steps - 1):
                    recv_prev = False

                # determine whether to recv grad from downstream
                recv_next = True
                if self._best_unbalanced_scheduler:
                    next_backward_virtual_pp_rank = self._get_virtual_pp_rank(
                        backward_micro_step_id + 1,
                        forward=False,
                    )
                    if self.is_pipeline_last_stage(ignore_virtual=True):
                        recv_next = _last_stage_need_recv_next(
                            backward_micro_step_id + 1
                        )
                else:
                    next_backward_virtual_pp_rank = self._get_virtual_pp_rank(
                        backward_micro_step_id + 1, forward=False
                    )
                    if self.is_pipeline_last_stage(ignore_virtual=True) and (
                        next_backward_virtual_pp_rank
                        == (self.num_model_chunks - 1)
                    ):
                        # last pp stage and last virtual stage
                        recv_next = False

                (
                    input_tensor,
                    output_tensor_grad,
                ) = self._p2p_helper.send_forward_backward_recv_forward_backward(
                    output_tensor,
                    input_tensor_grad,
                    recv_prev=recv_prev,
                    recv_next=recv_next,
                    batch_p2p_comm=self._use_batch_p2p_comm,
                    skip_check_meta=not self.training,
                )
            # append input_tensor no matter none or not
            self.input_tensors[next_forward_virtual_pp_rank].append(
                input_tensor
            )

            # append output_tensor_grad no matter none or not
            if self._best_unbalanced_scheduler:
                if self.is_pipeline_last_stage(ignore_virtual=True):
                    if recv_next:
                        recv_next_virtual_pp_rank = _last_stage_recv_pp_rank(
                            backward_micro_step_id + 1
                        )
                        self.output_tensor_grads[
                            recv_next_virtual_pp_rank
                        ].append(output_tensor_grad)
                        if (
                            next_backward_virtual_pp_rank
                            == self.num_model_chunks - 1
                            and recv_next_virtual_pp_rank
                            != next_backward_virtual_pp_rank
                        ):
                            self.output_tensor_grads[
                                self.num_model_chunks - 1
                            ].append(None)
                    elif (
                        next_backward_virtual_pp_rank
                        == self.num_model_chunks - 1
                    ):
                        self.output_tensor_grads[
                            self.num_model_chunks - 1
                        ].append(None)
                else:
                    self.output_tensor_grads[
                        next_backward_virtual_pp_rank
                    ].append(output_tensor_grad)
            else:
                self.output_tensor_grads[next_backward_virtual_pp_rank].append(
                    output_tensor_grad
                )

            self._release_output(output_tensor)

        assert fwd_buffer_queue.empty(), "forward buffer should be empty"
        if not static_scheduler:
            self._release_output(output_tensor)

        # remaining backward steps
        if not forward_only:
            if self._overlap_p2p_comm and bwd_wait_handles is not None:
                for wait_handles in bwd_wait_handles:
                    wait_handles.wait()

            # no steady steps, which only occurs when accumulate_step == num_stage
            if not steady_steps:
                output_tensor_grad = p2p.recv_backward(
                    self.is_pipeline_last_stage(),
                    batch_p2p_comm=self._use_batch_p2p_comm,
                )
                self.output_tensor_grads[self.num_model_chunks - 1].append(
                    output_tensor_grad
                )
            for micro_step in range(steady_steps, num_steps):
                if static_scheduler:
                    virtual_pp_rank = self._get_virtual_pp_rank(
                        micro_step, forward=False
                    )
                    real_micro_step = self._backward_micro_step_counter[
                        virtual_pp_rank
                    ]
                    self._backward_micro_step_counter[virtual_pp_rank] += 1
                    schedule += f"b{real_micro_step}_vp{virtual_pp_rank};"
                    logger.info(
                        f"backward step for {real_micro_step} with virtual pp rank {virtual_pp_rank}"
                    )
                    continue

                if (
                    micro_step
                    < steady_steps + self.num_stages - 1 - self.stage_id
                ) and self.user_hooks_enabled:
                    self.bubble_hooks.run_hook()

                # cooldown loop
                self._record_stamp("B", micro_step, '"B"', forward=False)
                input_tensor_grad = self._backward_step_helper(micro_step)
                self._record_stamp("B", micro_step, '"E"', forward=False)
                next_backward_virtual_pp_rank = self._get_virtual_pp_rank(
                    micro_step + 1,
                    forward=False,
                )
                if (
                    self._best_unbalanced_scheduler
                    and self.is_pipeline_last_stage(ignore_virtual=True)
                ):
                    cur_pp_rank = self._get_virtual_pp_rank(
                        micro_step, forward=False
                    )
                    if cur_pp_rank != 0:
                        last_stage_recv_queue.append((micro_step, cur_pp_rank))

                recv_next = True
                if self.is_pipeline_last_stage(ignore_virtual=True):
                    if self._best_unbalanced_scheduler:
                        recv_next = _last_stage_need_recv_next(micro_step + 1)
                    else:
                        if next_backward_virtual_pp_rank == (
                            self.num_model_chunks - 1
                        ):
                            recv_next = False

                if micro_step == (num_steps - 1):
                    recv_next = False

                if self.is_pipeline_first_stage(ignore_virtual=True):
                    input_tensor_grad = _process_bwd_buffer(
                        micro_step, input_tensor_grad
                    )

                # append output_tensor_grad no matter none or not
                if self._best_unbalanced_scheduler:
                    if self.is_pipeline_last_stage(ignore_virtual=True):
                        output_tensor_grad = (
                            self._p2p_helper.send_backward_recv_backward(
                                input_tensor_grad,
                                recv_next=recv_next,
                                batch_p2p_comm=self._use_batch_p2p_comm,
                            )
                        )
                        if recv_next:
                            recv_next_virtual_pp_rank = (
                                _last_stage_recv_pp_rank(micro_step + 1)
                            )
                            self.output_tensor_grads[
                                recv_next_virtual_pp_rank
                            ].append(output_tensor_grad)
                        else:
                            self.output_tensor_grads[
                                next_backward_virtual_pp_rank
                            ].append(output_tensor_grad)
                    else:
                        self.output_tensor_grads[
                            next_backward_virtual_pp_rank
                        ].append(
                            self._p2p_helper.send_backward_recv_backward(
                                input_tensor_grad,
                                recv_next=recv_next,
                                batch_p2p_comm=self._use_batch_p2p_comm,
                            )
                        )
                else:
                    self.output_tensor_grads[
                        next_backward_virtual_pp_rank
                    ].append(
                        self._p2p_helper.send_backward_recv_backward(
                            input_tensor_grad,
                            recv_next=recv_next,
                            batch_p2p_comm=self._use_batch_p2p_comm,
                        )
                    )

            self._sync_overlap_grads()

            for _ in range(self.stage_id):
                self.bubble_hooks.run_hook()

            if static_scheduler:
                self._reset_counter()
                return schedule

            if self._enable_timer:
                self.timers("allreduce_shared_weight_gradients").start()
            self._layers.allreduce_shared_weight_gradients()
            if self._enable_timer:
                self.timers("allreduce_shared_weight_gradients").stop()

        self._flush_records()

        assert bwd_buffer_queue.empty(), "backward buffer should be empty"
        if compute_loss:
            # return loss if compute loss
            if self._enable_timer:
                self.timers("broadcast_final_loss").start()
            with paddle.amp.auto_cast(enable=False):
                train_loss = self._broadcast_final_loss(return_micro_batch_loss)
            if self._enable_timer:
                self.timers("broadcast_final_loss").stop()
        else:
            # else just return all intermediate output tensor for all micro steps
            train_loss = self.output_tensors

        if self._clear_every_step_cache:
            self._p2p_helper.clear_meta_cache()

        self.timer_printer()
        if self.processed_steps < g_profile_pipeline_details_steps:
            get_sync_logger().info("end forward_backward_pipeline")
        self.processed_steps += 1
        self._check_user_hooks_status_at_step_end()

        return train_loss

    def train_batch(
        self,
        data,
        optimizer,
        lr_scheduler=None,
        scaler=None,
        loss_fn_idx=0,
        return_micro_batch_loss=False,
    ):
        data = self._prepare_training(data, optimizer, lr_scheduler)

        # check loss_fn_idx is valid and loss_fn exists
        assert (
            loss_fn_idx in range(len(self._layers._loss_fn))
            and self._layers._loss_fn[loss_fn_idx] is not None
        ), f"loss function {loss_fn_idx} should exist to compute loss"
        self.loss_fn_idx = loss_fn_idx

        # interleave scheduler for pipeline parallel
        train_loss = self.forward_backward_pipeline(
            data, scaler, return_micro_batch_loss=return_micro_batch_loss
        )

        # optimizer
        with paddle.amp.auto_cast(enable=False):
            self._optimizer_step()

        return train_loss

    def eval_batch(self, data, compute_loss=False, loss_fn_idx=0):
        self.user_hooks_enabled = False
        # reset the virtual pp rank for each run
        self.set_virtual_pipeline_rank(0)

        self._layers.eval()
        self._compute_loss = compute_loss

        # check loss_fn_idx is valid and loss_fn exists
        assert (
            loss_fn_idx in range(len(self._layers._loss_fn))
            and self._layers._loss_fn[loss_fn_idx] is not None
        ), f"loss function {loss_fn_idx} should exist to compute loss"
        self.loss_fn_idx = loss_fn_idx

        return self.forward_backward_pipeline(data, None, forward_only=True)

    def get_static_scheduler(self):
        return self.forward_backward_pipeline(
            data=None, scaler=None, static_scheduler=True
        )


class PipelineParallelWithInterleaveFthenB(PipelineParallelWithInterleave):
    def __init__(self, layers, hcg, strategy):
        super().__init__(layers=layers, hcg=hcg, strategy=strategy)

    def _init_user_bubble_hooks(self):
        # (TODO:gexiao) support bubble hooks if needed
        self.bubble_hooks = None
        # self.bubble_hooks = PipelineHook()
        # self.bubble_hooks.set_hooks_capacity(2 * self.num_stages - 2)

    def _check_sanity(self):
        assert (
            framework.in_dynamic_mode()
        ), "virtual pipeline stage with interleave only support eager dygraph mode"

        assert (
            self.num_stages > 2
        ), "virtual pipeline must run under pp degree > 2"

    def _get_virtual_pp_rank(self, micro_step, forward):
        virtual_pp_stage = micro_step % (
            self.accumulate_steps * self.num_model_chunks
        )
        virtual_pp_stage = virtual_pp_stage // self.accumulate_steps
        if not forward:
            virtual_pp_stage = self.num_model_chunks - virtual_pp_stage - 1

        return virtual_pp_stage

    def _overlap_comm_grads(self):
        if not self._comm_overlap:
            return
        self._backward_step_count += 1
        sync_step = self._backward_step_count - self.stage_id

        if sync_step > 0 and sync_step % self.accumulate_steps == 0:
            chunk_idx = self._virtual_pp_world_size - (
                sync_step // self.accumulate_steps
            )
            for buffer in self._chunk_2_comm_buffers[chunk_idx]:
                buffer.comm_grads()

        if self.stage_id == 0:
            return

        if (
            self._backward_step_count
            == self.accumulate_steps * self._virtual_pp_world_size
        ):
            for buffer in self._chunk_2_comm_buffers[0]:
                buffer.comm_grads()

    def _sync_overlap_grads(self):
        if not self._comm_overlap:
            return

        expected_count = self.accumulate_steps * self._virtual_pp_world_size
        assert self._backward_step_count == expected_count, (
            f"backward step count should be equal to accumulate steps * virtual pp world size, "
            f"but got {self._backward_step_count}, expected result is {expected_count}"
        )

        for buffers in self._chunk_2_comm_buffers.values():
            for buffer in buffers:
                buffer.scale_grads()

    def forward_backward_pipeline(
        self,
        data,
        scaler,
        forward_only=False,
        compute_loss=True,
        return_micro_batch_loss=False,
    ):
        self._reset_user_hooks_status()
        if self.processed_steps < g_profile_pipeline_details_steps:
            get_sync_logger().info("start forward_backward_pipeline")
        if not compute_loss:
            assert (
                not forward_only
            ), "compute_loss can only be set to False when forward_only is set to True"

        # NOTE(shenliang03): Due to ring_exchange for pipeline with interleave, cache should be enabled
        assert (
            self._using_cache
        ), "cache should be enabled for pipeline with interleave"

        # init some attributes for this batch run
        self.scaler = scaler
        self.total_loss = None
        self.micro_batch_id = 0
        self._forward_only = forward_only
        self.user_hooks_enabled = not self._forward_only

        assert (
            self.accumulate_steps == self.num_stages
            or self.accumulate_steps % self.num_stages != 0
        ), f"accumulate_steps({self.accumulate_steps}) and num_stages({self.num_stages}) should be a multiple or accumulate_steps % num_stages == 0"

        self._backward_step_count = 0
        skip_steps = self.accumulate_steps - self.num_stages
        send_recv_buffer_queue = queue.Queue()

        # init some data buffers for interleave scheduler
        self.input_tensors = [[] for _ in range(self.num_model_chunks)]
        self.output_tensors = [[] for _ in range(self.num_model_chunks)]
        self.output_tensor_grads = [[] for _ in range(self.num_model_chunks)]

        micro_dataset = self._wrap_data(data)
        num_steps = self.accumulate_steps * self.num_model_chunks

        self.set_virtual_pipeline_rank(0)
        self.input_tensors[0].append(
            self._p2p_helper.recv_forward(
                self.is_pipeline_first_stage(),
                sync_recv=False,
                batch_p2p_comm=self._use_batch_p2p_comm,
            )
        )

        # run startup steps
        for micro_step in range(num_steps):
            output_tensor = self._forward_step_helper(micro_dataset, micro_step)
            # determine whether recv forward tensor or not
            next_virtual_pp_rank = self._get_virtual_pp_rank(
                micro_step + 1, forward=True
            )

            recv_prev = True
            if self.is_pipeline_first_stage(ignore_virtual=True):
                if next_virtual_pp_rank == 0:
                    # next chunk is the first chunk, not need to pre recv an input tensor
                    recv_prev = False

            # last micro step, no next run
            if micro_step == (num_steps - 1):
                recv_prev = False

            if self.is_pipeline_last_stage(ignore_virtual=True):
                # last stage skip send/recv
                if not self.is_pipeline_last_stage():
                    send_recv_buffer_queue.put(output_tensor)

                if micro_step < skip_steps or (
                    self.is_pipeline_last_stage()
                    and micro_step % self.accumulate_steps >= skip_steps
                ):
                    output_tensor = None
                else:
                    output_tensor = send_recv_buffer_queue.get()

            input_tensor = self._p2p_helper.send_forward_recv_forward(
                output_tensor,
                recv_prev=recv_prev,
                batch_p2p_comm=self._use_batch_p2p_comm,
                skip_check_meta=not self.training,
            )
            self.input_tensors[next_virtual_pp_rank].append(input_tensor)

            self._release_output(output_tensor)

        assert (
            send_recv_buffer_queue.empty()
        ), "send_recv buffer should be empty"

        # remaining backward steps
        if not forward_only:
            self.output_tensor_grads[self.num_model_chunks - 1].append(
                self._p2p_helper.recv_backward(
                    self.is_pipeline_last_stage(),
                    sync_recv=False,
                    batch_p2p_comm=self._use_batch_p2p_comm,
                )
            )

            for micro_step in range(num_steps):
                # cooldown loop
                input_tensor_grad = self._backward_step_helper(micro_step)
                next_backward_virtual_pp_rank = self._get_virtual_pp_rank(
                    micro_step + 1, forward=False
                )

                recv_next = True
                if self.is_pipeline_last_stage(ignore_virtual=True):
                    if next_backward_virtual_pp_rank == (
                        self.num_model_chunks - 1
                    ):
                        recv_next = False

                if micro_step == (num_steps - 1):
                    recv_next = False

                if self.is_pipeline_first_stage(ignore_virtual=True):
                    if not self.is_pipeline_first_stage():
                        send_recv_buffer_queue.put(input_tensor_grad)

                    if micro_step < skip_steps or (
                        self.is_pipeline_first_stage()
                        and micro_step % self.accumulate_steps >= skip_steps
                    ):
                        input_tensor_grad = None
                    else:
                        input_tensor_grad = send_recv_buffer_queue.get()

                self.output_tensor_grads[next_backward_virtual_pp_rank].append(
                    self._p2p_helper.send_backward_recv_backward(
                        input_tensor_grad,
                        recv_next=recv_next,
                        batch_p2p_comm=self._use_batch_p2p_comm,
                    )
                )

            assert (
                send_recv_buffer_queue.empty()
            ), "send_recv buffer should be empty"

            self._sync_overlap_grads()

            if self._enable_timer:
                self.timers("allreduce_shared_weight_gradients").start()
            self._layers.allreduce_shared_weight_gradients()
            if self._enable_timer:
                self.timers("allreduce_shared_weight_gradients").stop()

        if compute_loss:
            # return loss if compute loss
            if self._enable_timer:
                self.timers("broadcast_final_loss").start()
            with paddle.amp.auto_cast(enable=False):
                train_loss = self._broadcast_final_loss(return_micro_batch_loss)
            if self._enable_timer:
                self.timers("broadcast_final_loss").stop()
        else:
            # else just return all intermediate output tensor for all micro steps
            train_loss = self.output_tensors

        if self._clear_every_step_cache:
            self._p2p_helper.clear_meta_cache()

        self.timer_printer()

        if self.processed_steps < g_profile_pipeline_details_steps:
            get_sync_logger().info("end forward_backward_pipeline")
        self.processed_steps += 1
        self._check_user_hooks_status_at_step_end()
        return train_loss


class OffloadQueue(queue.Queue):
    def __init__(self, offload=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger.info(f"Using OffloadQueue: {offload}")
        self.offload = offload

    def put(self, tensor, *args, **kwargs):
        if self.offload and isinstance(
            tensor, (paddle.Tensor, paddle.base.framework.core.eager.Tensor)
        ):
            tensor_cpu = tensor.pin_memory()
            tensor_cpu._share_buffer_to(tensor)
        elif self.offload and isinstance(tensor, tuple):
            for t in tensor:
                if isinstance(
                    t, (paddle.Tensor, paddle.base.framework.core.eager.Tensor)
                ):
                    t_cpu = t.pin_memory()
                    t_cpu._share_buffer_to(t)
        super().put(tensor, *args, **kwargs)

    def get(self, *args, **kwargs):
        tensor = super().get(*args, **kwargs)
        if self.offload and isinstance(
            tensor, (paddle.Tensor, paddle.base.framework.core.eager.Tensor)
        ):
            tensor = tensor.to(paddle.base.framework._current_expected_place())
        elif self.offload and isinstance(tensor, tuple):
            for t in tensor:
                if isinstance(
                    t, (paddle.Tensor, paddle.base.framework.core.eager.Tensor)
                ):
                    t_dev = t.to(
                        paddle.base.framework._current_expected_place()
                    )
                    t_dev._share_buffer_to(t)
        return tensor


class VPPFhenBInBalancedMemory(PipelineParallelWithInterleaveFthenB):
    def __init__(self, layers, hcg, strategy):
        super().__init__(layers=layers, hcg=hcg, strategy=strategy)
        logger.info("Using VPPFhenBInBalancedMemory")

    def _init_user_bubble_hooks(self):
        # (TODO:gexiao) support bubble hooks if needed
        self.bubble_hooks = None
        # self.bubble_hooks = PipelineHook()
        # self.bubble_hooks.set_hooks_capacity(2 * self.num_stages - 2)

    def forward_backward_pipeline(
        self,
        data,
        scaler,
        forward_only=False,
        compute_loss=True,
        return_micro_batch_loss=False,
    ):
        self._reset_user_hooks_status()
        if not compute_loss:
            assert (
                not forward_only
            ), "compute_loss can only be set to False when forward_only is set to True"
        assert (
            self._using_cache
        ), "cache should be enabled for pipeline with interleave"
        self.user_hooks_enabled = not forward_only
        if forward_only:
            return super().forward_backward_pipeline(
                data,
                scaler,
                forward_only,
                compute_loss,
                return_micro_batch_loss,
            )

        if self.processed_steps < g_profile_pipeline_details_steps:
            get_sync_logger().info("start forward_backward_pipeline")

        # init some attributes for this batch run
        self.scaler = scaler
        self.total_loss = None
        self.micro_batch_id = 0
        self._forward_only = forward_only

        # init some data buffers for interleave scheduler
        self.input_tensors = [[] for _ in range(self.num_model_chunks)]
        self.output_tensors = [[] for _ in range(self.num_model_chunks)]
        self.output_tensor_grads = [[] for _ in range(self.num_model_chunks)]
        backward_send_recv_buffer_queue = OffloadQueue(
            offload=self._enable_offload_queue
        )
        forward_send_recv_buffer_queue = OffloadQueue(
            offload=self._enable_offload_queue
        )

        skip_steps = self.accumulate_steps - self.num_stages
        micro_dataset = self._wrap_data(data)
        num_steps = self.accumulate_steps * self.num_model_chunks

        # the whole pipeline is splited into 3 parse:
        # startup_steps, steady_1f1b_steps, cooldown_steps
        startup_steps = (
            self.accumulate_steps * (self.num_model_chunks - 1)
            + self.num_stages
            - self.stage_id
            - 1
        )
        steady_1f1b_steps = self.accumulate_steps - (
            self.num_stages - self.stage_id - 1
        )
        cooldown_steps = startup_steps

        self.set_virtual_pipeline_rank(0)
        self.input_tensors[0].append(
            self._p2p_helper.recv_forward(
                self.is_pipeline_first_stage(),
                sync_recv=False,
                batch_p2p_comm=self._use_batch_p2p_comm,
            )
        )

        # In startup_steps, we send every output_tensor of last stage,
        # to simplify the code logic of stage 1F1B.
        for micro_step in range(startup_steps):
            self._record_stamp("F", micro_step, '"B"', forward=True)
            output_tensor = self._forward_step_helper(micro_dataset, micro_step)
            self._record_stamp("F", micro_step, '"E"', forward=True)
            next_forward_virtual_pp_rank = self._get_virtual_pp_rank(
                micro_step + 1, forward=True
            )
            recv_prev = True
            if self.is_pipeline_first_stage(ignore_virtual=True) and (
                micro_step < self.num_stages - 1
            ):
                recv_prev = False

            input_tensor = self._p2p_helper.send_forward_recv_forward(
                output_tensor,
                recv_prev=recv_prev,
                batch_p2p_comm=self._use_batch_p2p_comm,
                skip_check_meta=not self.training,
            )
            if self.is_pipeline_first_stage(ignore_virtual=True):
                if input_tensor is not None:
                    # stash the input_tensor and it will be used in the next chunk later
                    forward_send_recv_buffer_queue.put(input_tensor)
                if next_forward_virtual_pp_rank == 0:
                    input_tensor = None
                else:
                    # when a input_tensor is needed, get one from the queue
                    input_tensor = forward_send_recv_buffer_queue.get()

            self.input_tensors[next_forward_virtual_pp_rank].append(
                input_tensor
            )
            self._release_output(output_tensor)

        if self.is_pipeline_first_stage(ignore_virtual=True):
            assert (
                forward_send_recv_buffer_queue.qsize()
                == num_steps - startup_steps - 1
            ), forward_send_recv_buffer_queue.qsize()

        input_tensor_grad = None
        for micro_step in range(steady_1f1b_steps):
            last_iter = micro_step == (steady_1f1b_steps - 1)
            forward_micro_step_id = micro_step + startup_steps
            backward_micro_step_id = micro_step

            self._record_stamp("F", forward_micro_step_id, '"B"', forward=True)
            output_tensor = self._forward_step_helper(
                micro_dataset, forward_micro_step_id
            )
            self._record_stamp("F", forward_micro_step_id, '"E"', forward=True)

            output_tensor_grad = self._p2p_helper.send_forward_recv_backward(
                output_tensor,
                self.is_pipeline_last_stage(ignore_virtual=True),
                batch_p2p_comm=self._use_batch_p2p_comm,
            )
            # Unlike normal FthenB, in 1F1B steps, we recv output_tensor_grad
            # for the current step, but not for the next step
            cur_backward_virtual_pp_rank = self._get_virtual_pp_rank(
                backward_micro_step_id, forward=False
            )
            self.output_tensor_grads[cur_backward_virtual_pp_rank].append(
                output_tensor_grad
            )
            self._record_stamp(
                "B", backward_micro_step_id, '"B"', forward=False
            )
            input_tensor_grad = self._backward_step_helper(
                backward_micro_step_id
            )
            self._record_stamp(
                "B", backward_micro_step_id, '"E"', forward=False
            )

            # stash the input_tensor_grad and it will be sent to ths last stage later
            if self.is_pipeline_first_stage(ignore_virtual=True):
                backward_send_recv_buffer_queue.put(input_tensor_grad)

            if not last_iter:
                input_tensor = self._p2p_helper.send_backward_recv_forward(
                    input_tensor_grad,
                    self.is_pipeline_first_stage(ignore_virtual=True),
                    batch_p2p_comm=self._use_batch_p2p_comm,
                )
                next_forward_virtual_pp_rank = self._get_virtual_pp_rank(
                    forward_micro_step_id + 1, forward=True
                )
                if self.is_pipeline_first_stage(ignore_virtual=True):
                    input_tensor = forward_send_recv_buffer_queue.get()
                self.input_tensors[next_forward_virtual_pp_rank].append(
                    input_tensor
                )

        assert (
            forward_send_recv_buffer_queue.qsize() == 0
        ), forward_send_recv_buffer_queue.qsize()

        next_backward_virtual_pp_rank = self._get_virtual_pp_rank(
            steady_1f1b_steps, forward=False
        )

        # no more fwd, but we need to send the input_tensor_grad.
        if self.is_pipeline_first_stage(ignore_virtual=True):
            input_tensor_grad = backward_send_recv_buffer_queue.get()
        self.output_tensor_grads[next_backward_virtual_pp_rank].append(
            self._p2p_helper.send_backward_recv_backward(
                input_tensor_grad,
                recv_next=True,
                batch_p2p_comm=self._use_batch_p2p_comm,
            )
        )

        # run cooldown
        for micro_step in range(cooldown_steps):
            backward_micro_step_id = micro_step + steady_1f1b_steps

            self._record_stamp(
                "B", backward_micro_step_id, '"B"', forward=False
            )
            input_tensor_grad = self._backward_step_helper(
                backward_micro_step_id
            )
            self._record_stamp(
                "B", backward_micro_step_id, '"E"', forward=False
            )
            next_backward_virtual_pp_rank = self._get_virtual_pp_rank(
                backward_micro_step_id + 1, forward=False
            )

            recv_next = True
            if backward_micro_step_id == (num_steps - 1):
                recv_next = False
            if self.is_pipeline_first_stage(ignore_virtual=True):
                if not self.is_pipeline_first_stage():
                    backward_send_recv_buffer_queue.put(input_tensor_grad)

                if (
                    self.is_pipeline_first_stage()
                    and backward_micro_step_id % self.accumulate_steps
                    >= skip_steps
                ):
                    # no need to send the input_tensor_grad anymore
                    input_tensor_grad = None
                else:
                    input_tensor_grad = backward_send_recv_buffer_queue.get()
            self.output_tensor_grads[next_backward_virtual_pp_rank].append(
                self._p2p_helper.send_backward_recv_backward(
                    input_tensor_grad,
                    recv_next=recv_next,
                    batch_p2p_comm=self._use_batch_p2p_comm,
                )
            )

        assert (
            backward_send_recv_buffer_queue.empty()
        ), "send_recv buffer should be empty"

        self._flush_records()
        self._sync_overlap_grads()

        if self._enable_timer:
            self.timers("allreduce_shared_weight_gradients").start()
        self._layers.allreduce_shared_weight_gradients()
        if self._enable_timer:
            self.timers("allreduce_shared_weight_gradients").stop()

        if compute_loss:
            # return loss if compute loss
            if self._enable_timer:
                self.timers("broadcast_final_loss").start()
            with paddle.amp.auto_cast(enable=False):
                train_loss = self._broadcast_final_loss(return_micro_batch_loss)
            if self._enable_timer:
                self.timers("broadcast_final_loss").stop()
        else:
            # else just return all intermediate output tensor for all micro steps
            train_loss = self.output_tensors

        if self._clear_every_step_cache:
            self._p2p_helper.clear_meta_cache()

        self.timer_printer()

        if self.processed_steps < g_profile_pipeline_details_steps:
            get_sync_logger().info("end forward_backward_pipeline")
        self.processed_steps += 1
        self._check_user_hooks_status_at_step_end()
        return train_loss
