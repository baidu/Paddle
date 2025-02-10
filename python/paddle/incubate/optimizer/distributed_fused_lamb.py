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
from paddle.base import core, unique_name
from paddle.base.executor import global_scope
from paddle.base.framework import Variable, name_scope
from paddle.base.layer_helper import LayerHelper
from paddle.nn import ClipGradByGlobalNorm
from paddle.optimizer import Optimizer


def init_communicator(block, rank, ranks, ring_id):
    eps = os.environ['PADDLE_TRAINER_ENDPOINTS']
    eps = [ep.strip() for ep in eps.split(",") if ep.strip()]
    cur_ep = eps[rank]
    other_eps = [eps[r] for r in ranks if r != rank]

    local_rank = ranks.index(rank)
    comm_var_name = unique_name.generate('comm_id')
    comm_id_var = block.create_var(
        name=comm_var_name, persistable=True, type=core.VarDesc.VarType.RAW
    )
    if core.is_compiled_with_cuda():
        block.append_op(
            type='c_gen_nccl_id',
            inputs={},
            outputs={'Out': comm_id_var},
            attrs={
                'rank': local_rank,
                'endpoint': cur_ep,
                'other_endpoints': other_eps,
                'ring_id': ring_id,
            },
        )
    elif core.is_compiled_with_xpu():
        block.append_op(
            type='c_gen_bkcl_id',
            inputs={},
            outputs={'Out': comm_id_var},
            attrs={
                'rank': local_rank,
                'endpoint': cur_ep,
                'other_endpoints': other_eps,
                'ring_id': ring_id,
            },
        )
    elif (
        paddle.distributed.ParallelEnv().device_type
        in paddle.device.get_all_custom_device_type()
    ):
        block.append_op(
            type='c_gen_xccl_id',
            inputs={},
            outputs={'Out': comm_id_var},
            attrs={
                'rank': local_rank,
                'endpoint': cur_ep,
                'other_endpoints': other_eps,
                'ring_id': ring_id,
            },
        )
    block.append_op(
        type='c_comm_init',
        inputs={'X': comm_id_var},
        outputs={},
        attrs={
            'nranks': len(ranks),
            'rank': local_rank,
            'ring_id': ring_id,
            'endpoints': ','.join(eps),
        },
    )
    tmp_var = block.create_var(name=unique_name.generate('tmp'))
    block.append_op(
        type='fill_constant', outputs={'Out': tmp_var}, attrs={'value': 1}
    )
    block.append_op(
        type='all_reduce',
        inputs={'x': tmp_var},
        outputs={'out': tmp_var},
        attrs={
            'ring_id': ring_id,
            'reduce_type': paddle.distributed.ReduceOp.SUM,
        },
    )
    block.append_op(
        type='c_sync_calc_stream',
        inputs={'X': tmp_var},
        outputs={'Out': tmp_var},
    )
    return ring_id


def broadcast_parameters(block, parameters, ring_id):
    for p in parameters:
        block.append_op(
            type='broadcast',
            inputs={'x': p},
            outputs={'out': p},
            attrs={
                'ring_id': ring_id,
            },
        )


class DistributedFusedLamb(Optimizer):
    def __init__(
        self,
        learning_rate=0.001,
        lamb_weight_decay=0.01,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-6,
        parameters=None,
        grad_clip=None,
        exclude_from_weight_decay_fn=None,
        clip_after_allreduce=True,
        is_grad_scaled_by_nranks=True,
        alignment=128,
        use_master_param_norm=True,
        gradient_accumulation_steps=1,
        use_master_acc_grad=True,
        nproc_per_node=None,
        use_hierarchical_allreduce=False,
        name=None,
    ):
        assert (
            not paddle.in_dynamic_mode()
        ), "DistributedFusedLamb does not support dygraph mode"
        super().__init__(learning_rate=learning_rate, grad_clip=None, name=name)

        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon
        self._weight_decay = (
            lamb_weight_decay if lamb_weight_decay is not None else 0.0
        )
        if grad_clip is not None:
            assert isinstance(
                grad_clip, ClipGradByGlobalNorm
            ), "Only ClipGradByGlobalNorm is supported in DistributedFusedLamb"
            max_global_grad_norm = grad_clip.clip_norm
        else:
            max_global_grad_norm = -1.0
        self._max_global_grad_norm = max_global_grad_norm
        self._alignment = alignment if alignment is not None else -1
        self._clip_after_allreduce = clip_after_allreduce
        self._is_grad_scaled_by_nranks = is_grad_scaled_by_nranks
        self._exclude_from_weight_decay_fn = exclude_from_weight_decay_fn
        self._scale = None
        self._use_master_param_norm = use_master_param_norm
        self._gradient_accumulation_steps = gradient_accumulation_steps
        self._use_master_acc_grad = use_master_acc_grad
        self._nproc_per_node = nproc_per_node
        self._use_hierarchical_allreduce = use_hierarchical_allreduce
        assert self._gradient_accumulation_steps >= 1

        self.helper = LayerHelper('distributed_fused_lamb')
        self._supports_check_nan_inf = True  # very import flag for AMP

        main_block = self.helper.main_program.global_block()
        self._found_inf = main_block.create_var(
            name=unique_name.generate('found_inf'),
            shape=[1],
            dtype=core.VarDesc.VarType.BOOL,
        )
        self._step = None

        if self._gradient_accumulation_steps > 1:
            self._stop_update = main_block.create_var(
                name=unique_name.generate('stop_update'),
                shape=[1],
                dtype=core.VarDesc.VarType.BOOL,
            )
        else:
            self._stop_update = None

        self._param_to_master_param = {}

    def _get_stop_update_var(self):
        return self._stop_update if self._stop_update is not None else False

    def _set_step(self, step):
        self._step = step

    def _get_or_create_step(self):
        if self._step is None:
            self._step = self._create_persistable_var('step', dtype='int64')
        return self._step

    def _set_scale(self, scale):
        assert scale is not None
        if not isinstance(scale, Variable):
            scale = self._create_scale_from_constant(scale)
        self._scale = scale

    def _create_scale_from_constant(self, value):
        name = unique_name.generate('global_scale')
        return paddle.static.create_global_var(
            name=name,
            shape=[1],
            dtype='float32',
            value=float(value),
            persistable=True,
        )

    def _get_or_create_scale(self):
        if self._scale is None:
            self._scale = self._create_scale_from_constant(1.0)
        return self._scale

    def _create_persistable_var(self, name=None, shape=[-1], dtype='float32'):
        startup_block = self.helper.startup_program.global_block()
        if name is not None:
            name = unique_name.generate(name)
        startup_var = startup_block.create_var(
            name=name,
            shape=shape,
            dtype=dtype,
            persistable=True,
            stop_gradient=True,
        )
        main_block = self.helper.main_program.global_block()
        main_var = main_block.create_var(
            name=startup_var.name,
            shape=startup_var.shape,
            dtype=startup_var.dtype,
            persistable=True,
            stop_gradient=True,
        )
        return main_var

    def _get_parameter(self, name, scope=None):
        if scope is None:
            scope = global_scope()

        master_param = self._param_to_master_param.get(name)
        assert master_param is not None

        master_param_t = scope.find_var(master_param).get_tensor()
        assert master_param_t._dtype() == paddle.float32

        param_t = scope.find_var(name).get_tensor()
        if param_t._dtype() == paddle.float32:
            assert param_t._ptr() == master_param_t._ptr()
            return param_t, None
        else:
            assert param_t._dtype() == paddle.float16
            assert param_t.shape() == master_param_t.shape()
            return param_t, master_param_t

    def apply_optimize(self, params_grads):
        self.apply_gradients(params_grads)

    def apply_gradients(self, params_grads):
        flattened = []
        for p, g in params_grads:
            flattened.extend([p, g])
        with flattened[0].block.program._optimized_guard(flattened), name_scope(
            "optimizer"
        ):
            self._apply_gradients_impl(params_grads)

    def _apply_gradients_impl(self, params_grads):
        for p, g in params_grads:
            assert (
                g.type == core.VarDesc.VarType.DENSE_TENSOR
            ), "Only support dense gradient"
            g.persistable = True  # the gradient must be persistable for fusion

        fp32_fused_param = self._create_persistable_var('fp32_fused_param')
        fp32_fused_grad = self._create_persistable_var('fp32_fused_grad')
        fp16_fused_param = self._create_persistable_var(
            'fp16_fused_param', dtype='float16'
        )
        fp16_fused_grad = self._create_persistable_var(
            'fp16_fused_grad', dtype='float16'
        )

        master_params = []
        for p, g in params_grads:
            master_p = self._create_persistable_var('master_weight')
            self._param_to_master_param[p.name] = master_p.name
            master_params.append(master_p)

        moment1 = self._create_persistable_var('moment1')
        moment1.is_distributed = True
        moment2 = self._create_persistable_var('moment2')
        moment2.is_distributed = True
        beta1pow = self._create_persistable_var('beta1pow')
        beta2pow = self._create_persistable_var('beta2pow')

        param_info = self._create_persistable_var('param_info', dtype='int32')
        param_info.is_distributed = True

        fused_offsets = self._create_persistable_var(
            'fused_offsets', dtype='int32'
        )

        fp32_partial_fused_offsets = self._create_persistable_var(
            'fp32_partial_fused_offsets', dtype='int32'
        )
        fp32_partial_fused_offsets.is_distributed = True

        fp16_partial_fused_offsets = self._create_persistable_var(
            'fp16_partial_fused_offsets', dtype='int32'
        )
        fp16_partial_fused_offsets.is_distributed = True

        param_order = self._create_persistable_var('param_order', dtype='int32')
        param_order.is_distributed = True

        if self._gradient_accumulation_steps > 1:
            fp32_acc_fused_grad = [
                self._create_persistable_var('fp32_acc_fused_grad')
            ]
            fp16_acc_fused_grad = [
                self._create_persistable_var(
                    'fp16_acc_fused_grad', dtype='float16'
                )
            ]
            acc_step = [self._create_persistable_var('acc_step', dtype='int64')]
        else:
            fp32_acc_fused_grad = []
            fp16_acc_fused_grad = []
            acc_step = []

        step = self._get_or_create_step()

        rank = paddle.distributed.get_rank()
        nranks = paddle.distributed.get_world_size()
        if self._nproc_per_node is None:
            nproc_per_node = nranks
        else:
            nproc_per_node = self._nproc_per_node
        assert (
            nranks % nproc_per_node == 0
        ), "nranks should be exactly divided by nproc_per_node"

        shard_inside_node = nranks > nproc_per_node
        local_rank = rank % nproc_per_node
        node_id = int(rank / nproc_per_node)
        node_num = int(nranks / nproc_per_node)
        ring_ids = []
        startup_block = self.helper.startup_program.global_block()
        if nranks > 1:
            ring_id = init_communicator(
                startup_block, rank, list(range(nranks)), 0
            )
            ring_ids.append(ring_id)

        use_hierarchical_allreduce = False
        if node_num > 1 and len(ring_ids) <= 1 and shard_inside_node:
            local_group_ranks = list(
                range(node_id * nproc_per_node, (node_id + 1) * nproc_per_node)
            )
            ring_id = init_communicator(
                startup_block, rank, local_group_ranks, 1
            )
            ring_ids.append(ring_id)

            if self._use_hierarchical_allreduce and nranks > nproc_per_node:
                use_hierarchical_allreduce = True
                outer_group_ranks = list(
                    range(rank % nproc_per_node, nranks, nproc_per_node)
                )
                ring_id = init_communicator(
                    startup_block, rank, outer_group_ranks, ring_ids[-1] + 1
                )
                ring_ids.append(ring_id)

        scale = self._get_or_create_scale()

        params = [p for p, _ in params_grads]
        grads = [g for _, g in params_grads]
        apply_weight_decay = [1] * len(params)
        if self._exclude_from_weight_decay_fn is not None:
            for i, p in enumerate(params):
                if self._exclude_from_weight_decay_fn(p):
                    apply_weight_decay[i] = 0

        for g in grads:
            startup_block.create_var(
                name=g.name,
                type=g.type,
                dtype=g.dtype,
                persistable=g.persistable,
                shape=g.shape,
            )

        if nranks > 1:
            broadcast_parameters(startup_block, params, ring_ids[0])

        startup_block.append_op(
            type='distributed_fused_lamb_init',
            inputs={
                'Param': params,
                'Grad': grads,
            },
            outputs={
                'FP32FusedParam': [fp32_fused_param],
                'FP32FusedGrad': [fp32_fused_grad],
                'FP16FusedParam': [fp16_fused_param],
                'FP16FusedGrad': [fp16_fused_grad],
                'Moment1': [moment1],
                'Moment2': [moment2],
                'Beta1Pow': [beta1pow],
                'Beta2Pow': [beta2pow],
                'GlobalScale': [scale],
                'ParamInfo': [param_info],
                'ParamOut': params,
                'MasterParamOut': master_params,
                'GradOut': grads,
                'FP32ShardFusedParamOffsets': [fp32_partial_fused_offsets],
                'FP16ShardFusedParamOffsets': [fp16_partial_fused_offsets],
                'FusedParamOffsets': [fused_offsets],
                'ParamOrder': [param_order],
                'Step': [step],
            },
            attrs={
                'alignment': self._alignment,
                'rank': local_rank if shard_inside_node else rank,
                'nranks': nproc_per_node if shard_inside_node else nranks,
                'apply_weight_decay': apply_weight_decay,
                'moment1': 0.0,
                'moment2': 0.0,
                'beta1': self._beta1,
                'beta2': self._beta2,
            },
        )

        main_block = self.helper.main_program.global_block()
        self._create_global_learning_rate()
        lr = None
        for p_g in params_grads:
            if lr is None:
                lr = self._create_param_lr(p_g)
            else:
                new_lr = self._create_param_lr(p_g)
                assert id(lr) == id(
                    new_lr
                ), "The learning rate for each parameter should be the same"
        assert lr is not None

        lamb_op = main_block.append_op(
            type='distributed_fused_lamb',
            inputs={
                'FP32FusedParam': [fp32_fused_param],
                'FP32FusedGrad': [fp32_fused_grad],
                'FP16FusedParam': [fp16_fused_param],
                'FP16FusedGrad': [fp16_fused_grad],
                'LearningRate': [lr],
                'Moment1': [moment1],
                'Moment2': [moment2],
                'Beta1Pow': [beta1pow],
                'Beta2Pow': [beta2pow],
                'GlobalScale': [scale],
                'ParamInfo': [param_info],
                'Param': params,
                'Grad': grads,
                'FusedParamOffsets': [fused_offsets],
                'FP32ShardFusedParamOffsets': [fp32_partial_fused_offsets],
                'FP16ShardFusedParamOffsets': [fp16_partial_fused_offsets],
                'ParamOrder': [param_order],
            },
            outputs={
                'FP32FusedParamOut': [fp32_fused_param],
                'FP16FusedParamOut': [fp16_fused_param],
                'Moment1Out': [moment1],
                'Moment2Out': [moment2],
                'Beta1PowOut': [beta1pow],
                'Beta2PowOut': [beta2pow],
                'ParamOut': params,
                'GradOut': grads,
                'FoundInf': [self._found_inf],
                'FP32AccFusedGrad': fp32_acc_fused_grad,
                'FP16AccFusedGrad': fp16_acc_fused_grad,
                'AccStep': acc_step,
                'StopUpdate': (
                    self._stop_update if self._stop_update is not None else []
                ),
                'Step': [step],
            },
            attrs={
                'weight_decay': self._weight_decay,
                'beta1': self._beta1,
                'beta2': self._beta2,
                'epsilon': self._epsilon,
                'max_global_grad_norm': self._max_global_grad_norm,
                'clip_after_allreduce': self._clip_after_allreduce,
                'rank': rank,
                'nranks': nranks,
                'ring_ids': ring_ids,
                'use_master_param_norm': self._use_master_param_norm,
                'is_grad_scaled_by_nranks': self._is_grad_scaled_by_nranks,
                'acc_steps': self._gradient_accumulation_steps,
                'use_master_acc_grad': self._use_master_acc_grad,
                'use_hierarchical_allreduce': use_hierarchical_allreduce,
            },
        )
        return [lamb_op]
