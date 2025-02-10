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
import warnings
from collections import defaultdict
from functools import reduce

import paddle
from paddle import framework
from paddle.base.dygraph import base as imperative_base
from paddle.base.framework import EagerParamBase
from paddle.distributed import fleet
from paddle.distributed.communication.reduce import (
    ReduceOp,
    is_avg_reduce_op_supported,
)
from paddle.framework.recall_error import (
    SHARDING_PAD_NON_ZERO_ERROR,
    check_naninf,
)
from paddle.utils import strtobool

from ...utils import timer_helper as timer
from ...utils.log_util import logger
from ...utils.tensor_fusion_helper import (
    HOOK_ACTION,
    FusedCommBuffer,
    assign_group_by_size,
    fused_parameters,
)

g_sharding_v2_check_zero_padding = int(
    os.getenv("FLAGS_sharding_v2_check_zero_padding", "0")
)


def _is_trainable(param):
    return not param.stop_gradient


class DygraphShardingOptimizer:
    """
    A wrapper for Sharding Optimizer in Dygraph.

    .. warning: DygraphShardingOptimizer is experimental and subject to change.

    .. ZeRO: https://arxiv.org/abs/1910.02054

    """

    # TODO (JZ-LIANG)
    # TO support following features in future:
    # 1. fused update parameter sync
    # 2. parameters_groups
    # 3. dynamic trainable params, which is the case between pretraining and finetuning
    # 4. option to choose fuse comm (more GPU MEM need) or un-fuse comm

    def __init__(self, optimizer, hcg):
        logger.info("init DygraphShardingOptimizer")
        # TODO(pangengzheng): support param_groups
        if isinstance(optimizer._parameter_list[0], dict):
            raise TypeError(
                "Do not support param_groups now, please set optimizer._parameter_list as a list of Parameter"
            )
        if not hasattr(optimizer, '_apply_optimize') or not callable(
            optimizer._apply_optimize
        ):
            raise ValueError(
                "the optimizer object should have _apply_optimize function"
            )
        # the self._parameter_list holds the whole model parameters
        self._parameter_list = optimizer._parameter_list
        self._origin_parameter_list = self._parameter_list
        self._inner_opt = optimizer
        self._hcg = hcg
        self._sharding_world_size = self._hcg.get_sharding_parallel_world_size()
        self._sharding_rank = self._hcg.get_sharding_parallel_rank()

        strategy = fleet.fleet._user_defined_strategy
        sharding_configs = strategy.hybrid_configs['sharding_configs']

        self.tensor_fusion = sharding_configs.tensor_fusion
        self.accumulate_steps = sharding_configs.accumulate_steps
        self.comm_overlap = sharding_configs.comm_overlap
        self.comm_buffer_size_MB = sharding_configs.comm_buffer_size_MB
        self.fuse_optimizer = sharding_configs.fuse_optimizer
        self.use_reduce_avg = sharding_configs.use_reduce_avg
        self.enable_fuse_optimizer_states = (
            sharding_configs.enable_fuse_optimizer_states
        )
        assert (
            not self.enable_fuse_optimizer_states
        ), "enable_fuse_optimizer_states is not supported on sharding optimizer V1 now."

        if self.use_reduce_avg and (not is_avg_reduce_op_supported()):
            self.use_reduce_avg = False
            warnings.warn(
                "nccl reduce_avg requires paddle compiled with cuda and nccl>=2.10.0, please check compilation setups."
            )

        pp_overlap = strategy.hybrid_configs['pp_configs'].sharding_comm_overlap
        if self.tensor_fusion or self.comm_overlap:
            assert (
                not pp_overlap
            ), "Can not enable pp's sharding_comm_overlap and sharding's tensor_fusion at the same time."

        self._use_main_grad = hasattr(self._parameter_list[0], "main_grad")
        self._rank2decay = {}
        self._rank2fused = {}
        self._comm_buffers = []

        self._rank2params = self._partition_parameters()
        self._param2rank = self._map_param_to_rank()

        self._broadcast_overlap = False
        self._forward_pre_hook_remove_helper = []

        if (
            paddle.is_compiled_with_xpu()
            and os.getenv("XPU_CDNN_CLUSTER_PARALLEL") is not None
        ):
            assert (
                not self.comm_overlap
            ), "comm overlap not support when use xpu cdnn_cluster parallel."

        try:
            # The fp32 params such as layer_norm_0.w_0 will be at the end of param_list.
            # Have to sort the params to make sure all params are in the forward using order.
            self._broadcast_order_params = sorted(
                self._parameter_list,
                key=lambda x: int(x.name.split('.')[0].split('_')[-1]),
            )

        except ValueError:
            self._broadcast_order_params = None

        if not self.tensor_fusion and not self.comm_overlap:
            local_params = self._rank2params[self._sharding_rank]
            self._set_inner_opt_attr('_parameter_list', local_params)
            self._set_inner_opt_attr('_param_groups', local_params)
        else:
            if self.fuse_optimizer:
                lr = None
                for param in self._origin_parameter_list:
                    if hasattr(param, "optimize_attr"):
                        param_lr = param.optimize_attr['learning_rate']
                        if lr is None:
                            lr = param_lr
                        elif lr != param_lr:
                            warnings.warn(
                                "Parameters have different learning rate, "
                                "won't do fusion on the optimizer."
                            )
                            self.fuse_optimizer = False
                            break
            self.origin_decay_param_fun = getattr(
                self._inner_opt, '_apply_decay_param_fun', None
            )
            self._tensor_fusion()

            decay_params = [
                p.name for p in self._rank2decay[self._sharding_rank]
            ]
            local_fused_params = self._rank2fused[self._sharding_rank]
            apply_decay_param_fun = lambda x: x in decay_params

            all_fused_params = []
            for v in self._rank2fused.values():
                all_fused_params += v
            self._parameter_list = all_fused_params
            self._param_groups = all_fused_params

            self._set_inner_opt_attr('_parameter_list', local_fused_params)
            self._set_inner_opt_attr('_param_groups', local_fused_params)
            if self.comm_overlap:
                # Only set local param for check finite when comm overlap.
                # Under comm overlap, all grads will be communicated before check_finite.
                # Therefore, each sharding rank can get all grads' info at check_finite.
                # Without comm overlap, all grads will be communicated after check_finite,
                # which means each sharding rank should do check_finite to all grads.
                self._local_parameter_list = local_fused_params
            if self.origin_decay_param_fun is not None:
                self._set_inner_opt_attr(
                    '_apply_decay_param_fun', apply_decay_param_fun
                )
            # Note: during the tensor fusion for parameters, the allocator will apply for
            # some extra GPU memory for the fused big parameters. This extra GPU memory will
            # be useless at once the fusion has done. But the Paddle's allocator won't
            # release those memory, it will hold that part in the memory poll. So after
            # tensor fusion, the 'reserved' memory will increase but the 'allocate' memory
            # won't change. To avoid failure on some other applications (such as some nvtx
            # operations), here we manually let the allocator release the cached memory.
            paddle.device.cuda.empty_cache()

    def clear_grad(self, set_to_zero=True):
        """
        should clear grad for all parameters in model
        """
        for p in self._parameter_list:
            if hasattr(p, "main_grad") and p.main_grad is not None:
                assert p._grad_ivar() is None
                if set_to_zero:
                    p.main_grad.zero_()
                else:
                    p.main_grad._clear()
                    p.main_grad = None
            elif not hasattr(p, "main_grad"):
                if self.tensor_fusion:
                    if set_to_zero:
                        p.grad.zero_()
                    else:
                        p.grad._clear()
                        p.grad = None
                else:
                    p.clear_gradient(set_to_zero)

    def _tensor_fusion(self):
        comm_group = self._hcg.get_sharding_parallel_group()
        for i in range(self._sharding_world_size):
            params = self._rank2params[i]
            dst = comm_group.ranks[i]
            # TODO(sharding dev): make scale_after_comm a field to be configured by user
            decay_fused, all_fused, all_buffer = fused_parameters(
                params,
                use_main_grad=self._use_main_grad,
                fuse_param=True,
                comm_overlap=self.comm_overlap,
                comm_group=comm_group,
                dst=dst,
                acc_step=self.accumulate_steps,
                scale_after_comm=False,
                apply_decay_param_fun=self.origin_decay_param_fun,
                use_reduce_avg=self.use_reduce_avg,
                group_size=self.comm_buffer_size_MB * 1024 * 1024,
            )
            if self.comm_overlap:
                self._comm_buffers += all_buffer
            self._rank2decay[i] = decay_fused
            self._rank2fused[i] = all_fused
            for p in all_fused:
                self._param2rank[p.name] = i

    def _partition_parameters(self):
        """
        Partitions parameters among sharding ranks.

        Return:
        Dict[int, List]
        """
        # TODO(JZ-LIANG) support multiple partition methods
        # method1: greedy even but unorder
        # method2: roughly even with order

        mapping = {}
        for rank_ in range(self._sharding_world_size):
            mapping[rank_] = []
        sizes = [0] * self._sharding_world_size

        parameters = list(self._parameter_list)
        need_sort_parameters = strtobool(
            os.getenv('FLAGS_sharding_sort_parameters', '1')
        )
        if need_sort_parameters:
            parameters.sort(
                key=lambda p: reduce(lambda x, y: x * y, p.shape), reverse=True
            )

        for param in parameters:
            rank = sizes.index(min(sizes))
            mapping[rank].append(param)
            numel = reduce(lambda x, y: x * y, param.shape, 1)
            assert (
                numel > 0
            ), f"param [{param.name}] should larger than 0, but it is [{numel}]"
            sizes[rank] += numel

        return mapping

    def _map_param_to_rank(self):
        """
        mapping parameters to the shard which holds it.

        Return:
        Dict[str, int]
        """
        mapping = {}
        for rank, params in self._rank2params.items():
            for param in params:
                mapping[param.name] = rank
        return mapping

    def filter_parameters(self, parameter_list, hcg):
        sharding_parallel_rank = hcg.get_sharding_parallel_rank()
        parameter_list = [
            param
            for param in parameter_list
            if self._param2rank[param.name] == sharding_parallel_rank
        ]
        return parameter_list

    def _get_param_grad(self, param):
        if not param.trainable:
            return None

        if hasattr(param, "main_grad"):
            assert (
                param._grad_ivar() is None
            ), "param.grad should be None when using main_grad"
            return param.main_grad

        return param._grad_ivar()

    def reduce_gradients(self, parameter_list, hcg):
        # TODO merge grad / nrank with dp
        logger.debug("sharding start gradients sync")
        if self.comm_overlap:
            for buffer in self._comm_buffers:
                buffer.scale_grads()
            return

        # sync here to guarantee cdnn_cluster parallel correct.
        if (
            paddle.is_compiled_with_xpu()
            and os.getenv("XPU_CDNN_CLUSTER_PARALLEL") is not None
        ):
            paddle.device.synchronize()

        with framework.no_grad():
            for param in parameter_list:
                g_var = self._get_param_grad(param)
                if g_var is not None:
                    reduce_op = ReduceOp.AVG
                    if not self.use_reduce_avg:
                        sharding_nrank = (
                            hcg.get_sharding_parallel_group().nranks
                        )
                        g_var.scale_(1.0 / sharding_nrank)
                        reduce_op = ReduceOp.SUM

                    # In align mode, we scale the grad in advance, so we need a SUM here
                    if paddle.distributed.in_auto_parallel_align_mode():
                        reduce_op = ReduceOp.SUM

                    param_rank = self._param2rank[param.name]

                    need_check = strtobool(
                        os.getenv('FLAGS_pp_check_naninf', '0')
                    )
                    if need_check:
                        err_msg = check_naninf(g_var)
                        if err_msg is not None:
                            raise ValueError(
                                f"{err_msg}. Tensor contains inf or nan values at rank {paddle.distributed.get_rank()} before gradient communication"
                            )

                    paddle.distributed.reduce(
                        g_var,
                        dst=hcg.get_sharding_parallel_group().ranks[param_rank],
                        op=reduce_op,
                        group=hcg.get_sharding_parallel_group(),
                        sync_op=True,
                    )

    def _forward_pre_hook_function(self, tasks):
        def __impl__(x, y):
            for task in tasks:
                task.wait()

        return __impl__

    def _sharding_sync_parameters(self):
        """
        Synchronize parameter across sharding group efficiently.
        """
        with framework.no_grad():
            # Choose appropriate parameters collection based on whether tensor fusion is enabled.
            valid_rank_to_params = (
                self._rank2params
                if not self.tensor_fusion
                else self._rank2fused
            )

            # Pre-compute sharding group ranks for efficiency
            sharding_group_ranks = self._hcg.get_sharding_parallel_group().ranks

            broadcast_tasks = []
            if self._broadcast_overlap:
                param2task = {}

                group = self._hcg.get_sharding_parallel_group()
                for param in self._broadcast_order_params:
                    if param.trainable:
                        task = paddle.distributed.broadcast(
                            tensor=param,
                            src=group.ranks[self._param2rank[param.name]],
                            group=group,
                            sync_op=False,
                        )
                        assert param.name not in param2task
                        param2task[param.name] = task

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

            else:
                for rank, params in valid_rank_to_params.items():
                    # Compute the global source rank only once per each rank's set of parameters
                    src_rank = sharding_group_ranks[rank]

                    for param in params:
                        # NOTE: We should check if the parameter is trainable, because some parameters
                        # (e.g., freeze the parameters for training) are not trainable and should
                        # not be broadcasted.
                        g_var = self._get_param_grad(param)
                        if g_var is not None:
                            task = paddle.distributed.broadcast(
                                param,
                                src=src_rank,
                                group=self._hcg.get_sharding_parallel_group(),
                                sync_op=False,
                            )
                            broadcast_tasks.append(task)

                # Wait for all async broadcast tasks to complete
                for task in broadcast_tasks:
                    task.wait()

    def _update_trainable(self):
        """
        allow user to update trainable parameters list during training
        """
        raise NotImplementedError

    def minimize(
        self, loss, startup_program=None, parameters=None, no_grad_set=None
    ):
        # NOTE in dygraph mode, the only different between step and minimize is that minimize
        # allow user to customize the parameters for updating on each step

        input_param_names = {param.name for param in parameters}
        parameters = list(
            filter(
                lambda x: x.name in input_param_names,
                self._rank2params[self._sharding_rank],
            )
        )
        result = self._inner_opt.minimize(
            loss, startup_program, parameters, no_grad_set
        )

        # sync parameters across sharding ranks
        self._sharding_sync_parameters()

        return result

    def _set_broadcast_overlap(self, broadcast_overlap, layers=None):
        self._broadcast_overlap = broadcast_overlap
        if self._broadcast_overlap:
            assert (
                layers is not None
            ), "To Enable Stage1 Optimizer Broadcast Overlap Forward, layers cannot be None"
            self._layers = layers
            warnings.warn(
                r"Setting overlap broadcast implies that `paddle.device.cuda.synchronize()` must be manually invoked before calling `paddle.save()` and prior to inference"
            )

            if self._broadcast_order_params is None:
                warnings.warn(
                    r"The param name passed to the optimizer doesn't follow .+_[0-9]+\..+ patter, "
                    "overlap broadcast may harm the performance."
                )
                self._broadcast_order_params = self._parameter_list

    @imperative_base.no_grad
    @framework.dygraph_only
    def step(self):
        # TODO Check whether the model trainable param changed and update state accordingly
        if self._broadcast_overlap:
            # Clear the pre forward hook in the optimizer step.
            for hook_remove in self._forward_pre_hook_remove_helper:
                hook_remove.remove()
            self._forward_pre_hook_remove_helper = []

        target_param_list = (
            self._origin_parameter_list
            if (not self.tensor_fusion or not self.fuse_optimizer)
            else self._parameter_list
        )
        if not isinstance(target_param_list[0], dict):
            params_grads = []
            for param in target_param_list:
                if (
                    hasattr(param, "regularizer")
                    and param.regularizer is not None
                ):
                    raise ValueError(
                        f"param {param.name} should not has the regularizer attribute"
                    )
                if param.stop_gradient:
                    continue
                grad_var = param._grad_ivar()
                if hasattr(param, "main_grad") and param.main_grad is not None:
                    grad_var = param.main_grad
                params_grads.append((param, grad_var))
            rank_params = (
                self._rank2params[self._sharding_rank]
                if (not self.tensor_fusion or not self.fuse_optimizer)
                else self._rank2fused[self._sharding_rank]
            )
            update_param_names = [p.name for p in rank_params]
            update_params_grads = [
                (p, g) for p, g in params_grads if p.name in update_param_names
            ]
            self._apply_optimize(
                loss=None,
                startup_program=None,
                params_grads=update_params_grads,
            )

        # sync parameters across sharding ranks
        self._sharding_sync_parameters()

    @framework.dygraph_only
    def set_state_dict(self, state_dict):
        inner_state = {}
        parameters = self._rank2params[self._sharding_rank]

        if "LR_Scheduler" in state_dict:
            inner_state["LR_Scheduler"] = state_dict.pop("LR_Scheduler")

        if "master_weights" in state_dict:
            master = state_dict.pop("master_weights")
            inner_state["master_weights"] = {}
            for p in parameters:
                for k, v in master.items():
                    if p.name == k:
                        v.name = self._inner_opt._gen_master_weight_var_name(p)
                        inner_state["master_weights"][k] = v

        for p in parameters:
            for k, v in state_dict.items():
                if p.name in k:
                    inner_state[k] = v

        self._inner_opt.set_state_dict(inner_state)

    def _set_inner_opt_attr(self, attr_name, value):
        inner_opt = self._inner_opt
        inner_opt_name = '_inner_opt'
        if not isinstance(attr_name, str):
            raise TypeError(
                f"attr_name should be str type, but is {type(attr_name)}"
            )
        while hasattr(inner_opt, attr_name):
            setattr(inner_opt, attr_name, value)
            if (
                hasattr(inner_opt, inner_opt_name)
                and getattr(inner_opt, inner_opt_name, None) is not None
            ):
                inner_opt = getattr(inner_opt, inner_opt_name, None)
            else:
                break

    def __getattr__(self, item):
        return getattr(self._inner_opt, item)


class DygraphShardingOptimizerV2:
    """
    A wrapper for Sharding Optimizer in Dygraph, which split params

    .. warning: DygraphShardingOptimizer is experimental and subject to change.

    .. ZeRO: https://arxiv.org/abs/1910.02054

    """

    def __init__(self, optimizer, hcg):
        logger.info("init DygraphShardingOptimizerV2")

        # TODO(pangengzheng): support param_groups
        if isinstance(optimizer._parameter_list[0], dict):
            raise TypeError(
                "Do not support param_groups now, please set optimizer._parameter_list as a list of Parameter"
            )
        if not hasattr(optimizer, '_apply_optimize') or not callable(
            optimizer._apply_optimize
        ):
            raise ValueError(
                "the optimizer object should have _apply_optimize function"
            )

        self._inner_opt = optimizer
        self._hcg = hcg
        self._sharding_world_size = self._hcg.get_sharding_parallel_world_size()
        self._sharding_rank = self._hcg.get_sharding_parallel_rank()

        self._parameter_list = optimizer._parameter_list

        # param name -> slice_param
        self._slice_params = {}
        # comm_buffer_list = []
        self._comm_buffer_list = []

        # slice parameter list
        self._local_parameter_list = [
            self._create_slice_param(p) for p in optimizer._parameter_list
        ]

        # Accessing user defined strategy
        strategy = fleet.fleet._user_defined_strategy
        sharding_config = strategy.hybrid_configs['sharding_configs']
        pp_config = strategy.hybrid_configs['pp_configs']

        # Asserting tensor fusion not supported
        self.tensor_fusion = sharding_config.tensor_fusion
        assert not self.tensor_fusion, "not supported yet"

        # Setting accumulate steps and communication overlap
        acc_steps = sharding_config.accumulate_steps
        self.comm_overlap = sharding_config.comm_overlap

        comm_buffer_size_MB = sharding_config.comm_buffer_size_MB
        free_grads_in_comm = sharding_config.free_grads_in_comm

        self._enable_timer = strategy.hybrid_configs["enable_optimizer_timer"]

        if self._enable_timer:
            if not timer.is_timer_initialized():
                timer.set_timers()
            self.timers = timer.get_timers()

        # Setting pipeline parallelism overlap
        self.pp_overlap = pp_config.sharding_comm_overlap
        self.sd_release_grads = (
            pp_config.release_gradients or sharding_config.release_gradients
        )

        # Check nccl reduce_avg setting
        self.use_reduce_avg = sharding_config.use_reduce_avg
        if self.use_reduce_avg and (not is_avg_reduce_op_supported()):
            self.use_reduce_avg = False
            warnings.warn(
                "nccl reduce_avg requires paddle compiled with cuda and nccl>=2.10.0, please check compilation setups."
            )

        self.enable_fuse_optimizer_states = (
            sharding_config.enable_fuse_optimizer_states
        )
        self._build_comm_buffers(
            acc_steps, comm_buffer_size_MB * 1024 * 1024, free_grads_in_comm
        )
        if self.enable_fuse_optimizer_states:
            self._inner_opt.use_fusion_storage()
        # NOTE(shenliang03): Sort the comm_buffers by dst rank,
        # it will improve the performance in reduce communicate. Default
        # g_shard_sort_reduce_root is True.

        self._comm_buffer_list.sort(key=lambda x: x._dst)

        self._set_inner_opt_attr('_parameter_list', self._local_parameter_list)
        self._set_inner_opt_attr('_param_groups', self._local_parameter_list)

        if (
            paddle.is_compiled_with_xpu()
            and os.getenv("XPU_CDNN_CLUSTER_PARALLEL") is not None
        ):
            assert (
                not self.comm_overlap
            ), "comm overlap not support when use xpu cdnn_cluster parallel."

        # Ensure acc_steps is greater than 0 when comm_overlap is used
        if self.comm_overlap:
            assert (
                acc_steps > 0
            ), "acc_steps should be larger than 0 when using comm_overlap in sharding"

        # Ensure pp_overlap and comm_overlap are not both True
        assert not (
            self.pp_overlap and self.comm_overlap
        ), "pp_overlap and comm_overlap should not be True at the same time"

        # Determine the use of pipeline parallelism
        self._use_pipeline_parallel = strategy.hybrid_configs["pp_degree"] > 1

        # Ensure pipeline parallel and comm_overlap are not used together
        if self._use_pipeline_parallel:
            assert (
                not self.comm_overlap
            ), "You should not use pipeline parallel and comm_overlap at the same time"

        # Register reduce overlap hook if comm_overlap is used without pp_overlap
        if not self.pp_overlap and self.comm_overlap:
            self.register_reduce_overlap_hook(use_comm=True)

        self._all_gather_overlap_forward = False
        self._forward_pre_hook_remove_helper = []

    def _set_all_gather_overlap_forward(
        self, all_gather_overlap_forward, layers
    ):
        self._all_gather_overlap_forward = all_gather_overlap_forward
        if self._all_gather_overlap_forward:
            self._layers = layers

    def register_reduce_overlap_hook(self, use_comm):
        # Register backward hooks for each parameter in the buffer
        for buffer in self._comm_buffer_list:
            for param in buffer._params:
                # Directly register the hook function with necessary parameters
                param._register_backward_hook(
                    self._create_backward_hook(buffer, param, use_comm)
                )

    def _create_backward_hook(self, buffer, param, use_comm):
        """Creates a backward hook function for autograd."""

        @paddle.autograd.no_grad()
        def fused_allreduce(*_):
            # Directly add gradient to the buffer
            buffer.add_grad(param, use_comm=use_comm)

        return fused_allreduce

    def _build_comm_buffers(
        self, acc_steps, group_size=256 * 1024 * 1024, free_grads_in_comm=False
    ):
        if self.pp_overlap:
            return
        # NOTE(lijin23): for XPU, we fuse all params to a single comm buffer to
        # improve the communication bandwidth of BKCL.
        if (
            paddle.is_compiled_with_xpu()
            and os.getenv("XPU_PADDLE_FUSE_SHARDING_BUFFER") is not None
        ):
            group_size = 2**62

        comm_group = self._hcg.get_sharding_parallel_group()

        color_dict = defaultdict(list)
        for param in self._parameter_list:
            color = getattr(param, 'color', -1)
            color_color = -1
            color_group = comm_group
            if isinstance(color, dict):
                # if color is dict: param.color = {'color': "1", 'group': group}
                color_color = color.get('color', -1)
                color_group = color.get('group', comm_group)
            else:
                # if color is not a dict: param.color = 1
                color_color = color
            color_dict[(color_color, color_group)].append(param)

        # NOTE(shenliang03): If comm_overlap is not used, the parameter list is sorted by data type to
        # to reduce communication overhead.
        if not self.comm_overlap:
            for color, params in color_dict.items():
                params.sort(key=lambda x: str(x.dtype))

        all_var_groups = []
        group_idx = 0
        for color, params in color_dict.items():
            g_color = color[0]
            g_group = color[1]
            logger.info(f"Tensor Fusion Color {g_color} and Group {g_group}: ")
            var_groups = assign_group_by_size(params, group_size)
            for _, parameters in var_groups.items():
                buffer = FusedCommBuffer(
                    group_idx,
                    parameters,
                    g_group,
                    acc_steps,
                    act=HOOK_ACTION.REDUCE_SCATTER,
                    release_grads=self.sd_release_grads,
                    use_reduce_avg=self.use_reduce_avg,
                    free_grads_in_comm=free_grads_in_comm,
                    init_slice_param=self.enable_fuse_optimizer_states,
                    slice_params=self._slice_params,
                )
                group_idx += 1
                self._comm_buffer_list.append(buffer)

    def clear_grad(self, set_to_zero=True):
        """
        should clear grad for all parameters in model
        """
        if not self.sd_release_grads:
            assert set_to_zero, "should not erase grad buffer"

        def clear_grad_func(p):
            if hasattr(p, "main_grad") and p.main_grad is not None:
                assert p._grad_ivar() is None
                if set_to_zero:
                    p.main_grad.zero_()
                else:
                    p.main_grad._clear()
                    p.main_grad = None
            elif not hasattr(p, "main_grad"):
                if self.tensor_fusion:
                    if set_to_zero:
                        p.grad.zero_()
                    else:
                        p.grad._clear()
                        p.grad = None
                else:
                    p.clear_gradient(set_to_zero)

        for p in self._parameter_list:
            clear_grad_func(p)

        if self.sd_release_grads and not self.pp_overlap:
            for comm_buffer in self._comm_buffer_list:
                if comm_buffer.need_reduce_scale_sync():
                    comm_buffer._clear_grad_storage()

    def filter_parameters(self, parameter_list, hcg):
        parameter_list = [
            self._slice_params[param.name] for param in parameter_list
        ]
        parameter_list = [
            param for param in parameter_list if param._is_initialized()
        ]
        return parameter_list

    def reduce_gradients(self, parameter_list, hcg):
        # TODO merge grad / nrank with dp
        logger.debug("sharding start gradients sync")
        # sync here to guarantee cdnn_cluster parallel correct.
        if (
            paddle.is_compiled_with_xpu()
            and os.getenv("XPU_CDNN_CLUSTER_PARALLEL") is not None
        ):
            paddle.device.synchronize()

        with framework.no_grad():
            for comm_buffer in self._comm_buffer_list:
                if self.sd_release_grads and comm_buffer.grad_storage is None:
                    if comm_buffer.need_reduce_scale_sync():
                        for param in comm_buffer.params:
                            comm_buffer._copy_grad_to_buffer(param)

            if g_sharding_v2_check_zero_padding:
                self._check_padding_zero()

            if self._enable_timer:
                self.timers("reduce-gradients").start()
            for comm_buffer in self._comm_buffer_list:
                if not self.comm_overlap:
                    comm_buffer._comm_grads()

                comm_buffer.scale_grads()

            if self._enable_timer:
                self.timers("reduce-gradients").stop()

    def _check_padding_zero(self):
        if self._enable_timer:
            self.timers("check-padding-zero").start()
        for comm_buffer in self._comm_buffer_list:
            for k, v in comm_buffer._sharding_param_grad_view.items():
                pad_tensor = v._get_padding()
                if pad_tensor is not None:
                    assert paddle.all(
                        pad_tensor == 0
                    ).item(), f"{SHARDING_PAD_NON_ZERO_ERROR}. The padding of Tensor {k} is not zero"
        if self._enable_timer:
            self.timers("check-padding-zero").stop()

    def _forward_pre_hook_function(self, tasks):
        def __impl__(x, y):
            for task in tasks:
                task.wait()

        return __impl__

    def _sharding_sync_parameters(self):
        """
        sync parameter across sharding group
        """
        if self._enable_timer:
            self.timers("sync-parameters").start()

        logger.debug("sharding start sync parameters")
        with framework.no_grad():
            if self._all_gather_overlap_forward:
                param2task = {}
                for comm_buffer in self._comm_buffer_list:
                    comm_buffer.sync_params(sync=False, param2task=param2task)

                for layer in self._layers.sublayers():
                    if len(layer.sublayers()) == 0:
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
            else:
                for comm_buffer in self._comm_buffer_list:
                    comm_buffer.sync_params()

        if self._enable_timer:
            self.timers("sync-parameters").stop()

    def _update_trainable(self):
        """
        allow user to update trainable parameters list during training
        """
        raise NotImplementedError

    def minimize(
        self, loss, startup_program=None, parameters=None, no_grad_set=None
    ):
        # NOTE in dygraph mode, the only different between step and minimize is that minimize
        # allow user to customize the parameters for updating on each step
        raise AssertionError("not supported yet")

    def _create_slice_param(self, param):
        # not initialized yet
        slice_param = EagerParamBase(shape=[1], dtype=param.dtype)
        slice_param.name = param.name

        def copy_attr(attr_name):
            if hasattr(param, attr_name):
                setattr(slice_param, attr_name, getattr(param, attr_name))

        copy_attr("is_distributed")
        copy_attr("optimize_attr")
        copy_attr("do_model_average")
        copy_attr("need_clip")
        copy_attr("no_sync")

        self._slice_params[param.name] = slice_param
        return slice_param

    def _collect_comm_buffers(self):
        if self._comm_buffer_list:
            return
        for param in self._parameter_list:
            if not hasattr(param, "comm_buffer_ref"):
                continue
            comm_buffer_ref = param.comm_buffer_ref
            del param.comm_buffer_ref
            comm_buffer = comm_buffer_ref()
            self._comm_buffer_list.append(comm_buffer)

        assert self._comm_buffer_list

    def _assign_slice_grad(self):
        param_num = 0
        for comm_buffer in self._comm_buffer_list:
            param_num = param_num + len(comm_buffer.params)
            for param in comm_buffer.params:
                assert param.name in self._slice_params
                slice_param = self._slice_params[param.name]
                if self.sd_release_grads and hasattr(slice_param, "main_grad"):
                    assert not slice_param.main_grad._is_initialized()
                    del slice_param.main_grad
                comm_buffer.assign_slice_grad(param, slice_param)

        assert param_num == len(self._parameter_list)

    def step(self):
        # TODO Check whether the model trainable param changed and update state accordingly
        # hack for pp comm overlap

        if self._all_gather_overlap_forward:
            # Clear the pre forward hook in the optimizer step.
            for hook_remove in self._forward_pre_hook_remove_helper:
                hook_remove.remove()
            self._forward_pre_hook_remove_helper = []

        self._collect_comm_buffers()
        self._assign_slice_grad()

        if not isinstance(self._parameter_list[0], dict):
            params_grads = []
            for param in self._parameter_list:
                if (
                    hasattr(param, "regularizer")
                    and param.regularizer is not None
                ):
                    raise ValueError(
                        f"param {param.name} should not has the regularizer attribute"
                    )
                if param.stop_gradient:
                    continue
                # update on slice
                assert param.name in self._slice_params
                param = self._slice_params[param.name]
                grad_var = param._grad_ivar()
                if hasattr(param, "main_grad") and param.main_grad is not None:
                    grad_var = param.main_grad
                if grad_var is not None:
                    params_grads.append((param, grad_var))

            if self._enable_timer:
                self.timers("apply-optimize").start()

            self._apply_optimize(
                loss=None,
                startup_program=None,
                params_grads=params_grads,
            )
            if self._enable_timer:
                self.timers("apply-optimize").stop()

        # sync parameters across sharding ranks
        self._sharding_sync_parameters()

    @framework.dygraph_only
    def set_state_dict(self, state_dict):
        inner_state = {}
        parameters = self._parameter_list

        if "LR_Scheduler" in state_dict:
            inner_state["LR_Scheduler"] = state_dict.pop("LR_Scheduler")

        if "master_weights" in state_dict:
            master = state_dict.pop("master_weights")
            inner_state["master_weights"] = {}
            for p in parameters:
                for k, v in master.items():
                    if p.name == k:
                        v.name = self._inner_opt._gen_master_weight_var_name(p)
                        inner_state["master_weights"][k] = v

        for p in parameters:
            for k, v in state_dict.items():
                if p.name in k:
                    inner_state[k] = v

        self._inner_opt.set_state_dict(inner_state)

    def _set_inner_opt_attr(self, attr_name, value):
        inner_opt = self._inner_opt
        inner_opt_name = '_inner_opt'
        if not isinstance(attr_name, str):
            raise TypeError(
                f"attr_name should be str type, but is {type(attr_name)}"
            )
        while hasattr(inner_opt, attr_name):
            setattr(inner_opt, attr_name, value)
            inner_opt = getattr(inner_opt, inner_opt_name, None)
            if inner_opt is None:
                break

    def __getattr__(self, item):
        return getattr(self._inner_opt, item)
