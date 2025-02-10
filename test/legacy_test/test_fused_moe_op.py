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

import unittest

import numpy as np
from op_test import OpTest
from test_sparse_attention_op import get_cuda_version

import paddle
import paddle.nn.functional as F
from paddle import base, nn
from paddle.incubate.nn.functional import fused_moe, swiglu
from paddle.nn.layer.common import Linear
from paddle.nn.quant import weight_quantize
from paddle.static import Program, program_guard

paddle.seed(42)


class Expert(nn.Layer):
    def __init__(self, d_model, d_feedforward):
        super().__init__()
        self.fc1 = nn.Linear(
            d_model, d_feedforward * 2
        )  # Swiglu expects twice the hidden_dim
        self.swiglu = swiglu
        self.fc2 = nn.Linear(d_feedforward, d_model)

    def forward(self, x, idx):
        x = self.fc1(x)
        x = swiglu(x)
        x = self.fc2(x)
        return x


@unittest.skipIf(
    not paddle.is_compiled_with_cuda()
    or get_cuda_version() < 11030
    or paddle.device.cuda.get_device_capability()[0] < 8,
    "FusedMoe requires CUDA >= 11.2 and CUDA_ARCH >= 8",
)
class TestFusedMoEOp(OpTest):
    def setUp(self):
        self.config()
        self.rtol = 1e-2
        self.atol = 1e-2

        paddle.set_default_dtype(self.x_type)
        self.__class__.op_type = "fused_moe"
        # Since it's only used in inference.
        self.__class__.no_need_check_grad = True

        paddle.disable_static(place=paddle.CUDAPlace(0))

        self.experts = nn.LayerList(
            [
                Expert(self.d_model, self.d_feedforward)
                for _ in range(self.num_expert)
            ]
        )

        self.bmm_w0 = paddle.to_tensor(
            np.array([expert.fc1.weight.numpy() for expert in self.experts]),
            dtype=self.x_type,
        )
        self.bmm_b0 = paddle.to_tensor(
            np.array(
                [expert.fc1.bias.numpy() for expert in self.experts]
            ).reshape(self.num_expert, 1, -1),
            dtype=self.x_type,
        )

        # d_model * 2 for swiglu
        self.bmm_w1 = paddle.to_tensor(
            np.array([expert.fc2.weight.numpy() for expert in self.experts]),
            dtype=self.x_type,
        )
        self.bmm_b1 = paddle.to_tensor(
            np.array(
                [expert.fc2.bias.numpy() for expert in self.experts]
            ).reshape(self.num_expert, 1, -1),
            dtype=self.x_type,
        )

        self.tensor_x = paddle.to_tensor(
            np.random.randn(self.batch_size, self.seq_len, self.d_model) * 0.1,
            dtype=self.x_type,
        )

        self.bmm_w0.stop_gradient = True
        self.bmm_b0.stop_gradient = True
        self.bmm_w1.stop_gradient = True
        self.bmm_b1.stop_gradient = True
        self.tensor_x.stop_gradient = True

        self.gate = Linear(self.d_model, self.num_expert)

        self.gate_weight = self.gate.weight.cast(paddle.float32)
        self.gate_weight.stop_gradient = True

        paddle.set_default_dtype(self.x_type)
        self.activation = swiglu
        self.quanted = False

    def config(self):
        self.x_type = np.float16
        self.batch_size = 8
        self.seq_len = 6000
        self.num_expert = 32
        self.d_model = 768
        self.d_feedforward = 3072
        self.top_k = 2
        self.quant_method = "None"
        self.norm_topk_prob = False
        self.group_moe = False

    def GetWintData(self):
        if self.quant_method == "None" or self.quanted:
            return
        self.quanted = True
        fc0_expert_weights_for_ref_list = []
        scale0 = []
        for i in range(self.num_expert):
            fc0_expert_weights_for_ref_i, fc0_expert_weights_scale_for_ref_i = (
                weight_quantize(self.bmm_w0[i], algo=self.quant_method)
            )
            fc0_expert_weights_for_ref_list.append(
                fc0_expert_weights_for_ref_i.reshape(
                    [self.d_model, self.d_feedforward * 2]
                    if self.quant_method == "weight_only_int8"
                    else [self.d_model, self.d_feedforward]
                )
            )
            scale0.append(fc0_expert_weights_scale_for_ref_i)
        self.bmm_w0 = paddle.to_tensor(fc0_expert_weights_for_ref_list)
        self.scale0 = paddle.to_tensor(scale0)

        fc1_expert_weights_for_ref_list = []
        scale1 = []
        for i in range(self.num_expert):
            fc1_expert_weights_for_ref_i, fc1_expert_weights_scale_for_ref_i = (
                weight_quantize(self.bmm_w1[i], algo=self.quant_method)
            )
            fc1_expert_weights_for_ref_list.append(
                fc1_expert_weights_for_ref_i.reshape(
                    [self.d_feedforward, self.d_model]
                    if self.quant_method == "weight_only_int8"
                    else [self.d_feedforward, self.d_model // 2]
                )
            )
            scale1.append(fc1_expert_weights_scale_for_ref_i)
        self.bmm_w1 = paddle.to_tensor(fc1_expert_weights_for_ref_list)
        self.scale1 = paddle.to_tensor(scale1)

    def GetFusedMoeOut(self, tensor_x):
        if self.quant_method != "None":
            self.GetWintData()
        paddle.disable_static(place=paddle.CUDAPlace(0))
        fused_out = fused_moe(
            tensor_x,
            self.gate_weight,
            self.bmm_w0,
            self.bmm_w1,
            self.bmm_b0,
            None if self.quant_method == "None" else self.scale0,
            self.bmm_b1,
            None if self.quant_method == "None" else self.scale1,
            self.quant_method,
            self.top_k,
            self.norm_topk_prob,
        )
        fused_out = fused_out.reshape([-1, self.seq_len, self.d_model])

        return fused_out

    def GetSplitMoeOut(self, tensor_x):
        if self.quant_method != "None":
            self.GetWintData()

        paddle.disable_static(place=paddle.CUDAPlace(0))

        # Reshape input tensor
        tensor_x = tensor_x.reshape([-1, self.d_model])  # Shape: [1280, 768]

        # Gate output
        gate_out = paddle.matmul(tensor_x.cast("float32"), self.gate_weight)
        gate_out = gate_out.reshape([-1, self.num_expert])

        from paddle.incubate.nn.functional import (
            moe_dispatch,
            moe_ffn,
            moe_reduce,
        )

        # Dispatch tokens to experts
        (
            permute_input,
            token_nums_per_expert,
            permute_indices_per_token,
            expert_scales_float,
            top_k_indices,
        ) = moe_dispatch(tensor_x, gate_out, self.top_k, self.group_moe)
        # Apply feed-forward networks for experts
        ffn_out = moe_ffn(
            permute_input,
            token_nums_per_expert,
            self.bmm_w0,
            self.bmm_w1,
            self.bmm_b0,
            None if self.quant_method == "None" else self.scale0,
            None if self.quant_method == "None" else self.scale1,
            self.quant_method,
        )

        # Reduce results back to the original token order
        final_out = moe_reduce(
            ffn_out,
            expert_scales_float,
            permute_indices_per_token,
            top_k_indices,
            self.bmm_b1,
            norm_topk_prob=self.norm_topk_prob,
        )

        final_out = final_out.reshape([-1, self.seq_len, self.d_model])
        return final_out

    def GetBaselineOut(self, hidden_states):
        paddle.disable_static(place=paddle.CUDAPlace(0))
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = paddle.reshape(hidden_states, [-1, hidden_dim])
        router_logits = self.gate(hidden_states).cast(paddle.float32)
        routing_weights = F.softmax(router_logits, axis=-1, dtype='float32')
        routing_weights, selected_experts = paddle.topk(
            routing_weights, self.top_k, axis=-1
        )
        # mixtral true, qwen_moe false
        if self.norm_topk_prob:
            routing_weights /= paddle.sum(
                routing_weights, axis=-1, keepdim=True
            )
        # we cast back to the input dtype
        routing_weights = routing_weights.cast(np.float32)

        final_hidden_states = paddle.zeros_like(hidden_states)

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = paddle.transpose(
            F.one_hot(selected_experts, num_classes=self.num_expert), [2, 1, 0]
        )

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_expert):
            expert_layer = self.experts[expert_idx]
            idx, top_x = paddle.where(expert_mask[expert_idx])

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = paddle.index_select(
                hidden_states, top_x, axis=0
            ).reshape([-1, hidden_dim])
            current_hidden_states = (
                expert_layer(current_state, expert_idx)
                * routing_weights[top_x, idx]
            )
            # Use scatter to accumulate the results
            paddle.index_add_(
                x=final_hidden_states,
                index=top_x.squeeze(),
                axis=0,
                value=current_hidden_states.to(hidden_states.dtype),
            )

        final_hidden_states = paddle.reshape(
            final_hidden_states, [batch_size, sequence_length, hidden_dim]
        )
        return final_hidden_states

    def test_fused_moe_op_new(self):
        ref_out = self.GetBaselineOut(self.tensor_x).cast(np.float32)

        fused_moe_out = self.GetFusedMoeOut(self.tensor_x).cast(np.float32)

        split_moe_out = self.GetSplitMoeOut(self.tensor_x).cast(np.float32)

        np.testing.assert_allclose(
            ref_out, fused_moe_out, rtol=self.rtol, atol=self.atol
        )

        np.testing.assert_allclose(
            fused_moe_out, fused_moe_out, rtol=self.rtol, atol=self.atol
        )


@unittest.skipIf(
    not paddle.is_compiled_with_cuda()
    or get_cuda_version() < 11030
    or paddle.device.cuda.get_device_capability()[0] < 8,
    "FusedMoe requires CUDA >= 11.2 and CUDA_ARCH >= 8",
)
class TestFusedGoupeMoEOp(TestFusedMoEOp):
    def config(self):
        super().config()
        self.x_type = paddle.bfloat16
        self.rtol = 1e-2
        self.atol = 1e-2
        self.norm_topk_prob = False
        self.group_moe = True
        self.num_expert = 48
        self.top_k = 6

    def GetMoeOut(self, tensor_x):
        paddle.disable_static(place=paddle.CUDAPlace(0))
        # Gate output
        tensor_x = tensor_x.reshape([-1, self.d_model])  # Shape: [1280, 768]

        gate_out = paddle.matmul(tensor_x.cast("float32"), self.gate_weight)
        gate_out = gate_out.reshape([-1, self.num_expert])

        from paddle.incubate.nn.functional import (
            moe_dispatch,
            moe_ffn,
            moe_reduce,
        )

        # Dispatch tokens to experts
        (
            permute_input,
            token_nums_per_expert,
            permute_indices_per_token,
            expert_scales_float,
            top_k_indices,
        ) = moe_dispatch(tensor_x, gate_out, self.top_k, self.group_moe)

        # Apply feed-forward networks for experts
        ffn_out = moe_ffn(
            permute_input,
            token_nums_per_expert,
            self.bmm_w0,
            self.bmm_w1,
            self.bmm_b0,
        )

        # Reduce results back to the original token order
        final_out = moe_reduce(
            ffn_out,
            expert_scales_float,
            permute_indices_per_token,
            top_k_indices,
            self.bmm_b1,
            norm_topk_prob=True,
        )
        return final_out

    def GetBaselineOut(self, hidden_states):
        paddle.disable_static(place=paddle.CUDAPlace(0))
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = paddle.reshape(hidden_states, [-1, hidden_dim])

        gate_out = self.gate(hidden_states).cast(paddle.float32)
        group_size = self.num_expert // self.top_k
        gate_out = gate_out.reshape([-1, self.top_k, group_size])
        gate_out = F.softmax(gate_out)
        max_prob = gate_out.max(-1, keepdim=True)  # [s, k, 1]
        gate_out /= max_prob
        gate_out = gate_out.reshape([0, -1])
        max_prob = max_prob.squeeze(-1)

        routing_weights, selected_experts = paddle.topk(
            gate_out, self.top_k, axis=-1
        )
        # norm routing_weights
        expert_scales_float = max_prob * routing_weights
        expert_scales_float = expert_scales_float / paddle.clip(
            expert_scales_float.sum(-1, keepdim=True), min=1e-12
        )
        routing_weights = expert_scales_float.cast(np.float32)

        final_hidden_states = paddle.zeros_like(hidden_states)

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = paddle.transpose(
            F.one_hot(selected_experts, num_classes=self.num_expert), [2, 1, 0]
        )

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_expert):
            expert_layer = self.experts[expert_idx]
            idx, top_x = paddle.where(expert_mask[expert_idx])

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = paddle.index_select(
                hidden_states, top_x, axis=0
            ).reshape([-1, hidden_dim])
            current_hidden_states = (
                expert_layer(current_state, expert_idx)
                * routing_weights[top_x, idx]
            )
            # Use scatter to accumulate the results
            paddle.index_add_(
                x=final_hidden_states,
                index=top_x.squeeze(),
                axis=0,
                value=current_hidden_states.to(hidden_states.dtype),
            )

        final_hidden_states = paddle.reshape(
            final_hidden_states, [batch_size * sequence_length, hidden_dim]
        )
        return final_hidden_states

    def test_fused_moe_op_new(self):
        ref_out = self.GetBaselineOut(self.tensor_x).cast(np.float32)
        fused_moe_out = self.GetMoeOut(self.tensor_x).cast(np.float32)

        np.testing.assert_allclose(
            ref_out, fused_moe_out, rtol=self.rtol, atol=self.atol
        )
        print("success !!!")


@unittest.skipIf(
    not paddle.is_compiled_with_cuda()
    or get_cuda_version() < 11030
    or paddle.device.cuda.get_device_capability()[0] < 8,
    "FusedMoe requires CUDA >= 11.2 and CUDA_ARCH >= 8",
)
class TestFusedMoEOpBf16(TestFusedMoEOp):
    def config(self):
        super().config()
        self.x_type = paddle.bfloat16
        self.rtol = 1e-2
        self.atol = 1e-2


@unittest.skipIf(
    not paddle.is_compiled_with_cuda()
    or get_cuda_version() < 11030
    or paddle.device.cuda.get_device_capability()[0] < 8,
    "FusedMoe requires CUDA >= 11.2 and CUDA_ARCH >= 8",
)
class TestFusedMoEOpWint8(TestFusedMoEOp):
    def config(self):
        super().config()
        self.rtol = 1e-2
        self.atol = 1e-2
        self.quant_method = "weight_only_int8"


@unittest.skipIf(
    not paddle.is_compiled_with_cuda()
    or get_cuda_version() < 11030
    or paddle.device.cuda.get_device_capability()[0] < 8,
    "FusedMoe requires CUDA >= 11.2 and CUDA_ARCH >= 8",
)
class TestFusedMoEOpWint4(TestFusedMoEOp):
    def config(self):
        super().config()
        self.rtol = 1e-2
        self.atol = 1e-2
        self.quant_method = "weight_only_int4"


@unittest.skipIf(
    not paddle.is_compiled_with_cuda()
    or get_cuda_version() < 11030
    or paddle.device.cuda.get_device_capability()[0] < 8,
    "FusedMoe requires CUDA >= 11.2 and CUDA_ARCH >= 8",
)
class TestFusedMoEOpNonNorm(TestFusedMoEOp):
    def config(self):
        super().config()
        self.rtol = 1e-2
        self.atol = 1e-2
        self.norm_topk_prob = False


@unittest.skipIf(
    not paddle.is_compiled_with_cuda()
    or get_cuda_version() < 11030
    or paddle.device.cuda.get_device_capability()[0] < 8,
    "FusedMoe requires CUDA >= 11.2 and CUDA_ARCH >= 8",
)
class TestFusedMoEOpStatic(OpTest):
    def setUp(self):
        self.config()
        self.rtol = 1e-2
        self.atol = 1e-2

        paddle.set_default_dtype(self.x_type)
        self.__class__.op_type = "fused_moe"
        # Since it's only used in inference.
        self.__class__.no_need_check_grad = True

        paddle.disable_static(place=paddle.CUDAPlace(0))

        self.experts = nn.LayerList(
            [
                Expert(self.d_model, self.d_feedforward)
                for _ in range(self.num_expert)
            ]
        )

        self.bmm_w0 = paddle.to_tensor(
            np.array([expert.fc1.weight.numpy() for expert in self.experts]),
            dtype=self.x_type,
        )
        self.bmm_b0 = paddle.to_tensor(
            np.array(
                [expert.fc1.bias.numpy() for expert in self.experts]
            ).reshape(self.num_expert, 1, -1),
            dtype=self.x_type,
        )

        # d_model*2 for swiglu
        self.bmm_w1 = paddle.to_tensor(
            np.array([expert.fc2.weight.numpy() for expert in self.experts]),
            dtype=self.x_type,
        )
        self.bmm_b1 = paddle.to_tensor(
            np.array(
                [expert.fc2.bias.numpy() for expert in self.experts]
            ).reshape(self.num_expert, 1, -1),
            dtype=self.x_type,
        )

        self.tensor_x = paddle.to_tensor(
            np.random.randn(self.batch_size, self.seq_len, self.d_model) * 0.1,
            dtype=self.x_type,
        )

        self.bmm_w0.stop_gradient = True
        self.bmm_b0.stop_gradient = True
        self.bmm_w1.stop_gradient = True
        self.bmm_b1.stop_gradient = True
        self.tensor_x.stop_gradient = True

        self.gate = Linear(self.d_model, self.num_expert)

        self.gate_weight = self.gate.weight.cast(paddle.float32)
        self.gate_weight.stop_gradient = True

        paddle.set_default_dtype(self.x_type)
        self.activation = swiglu

    def config(self):
        self.x_type = np.float16
        self.batch_size = 10
        self.seq_len = 128
        self.num_expert = 32
        self.d_model = 768
        self.d_feedforward = 3072
        self.top_k = 2
        self.quant_method = "None"
        self.norm_topk_prob = True

    def GetFusedMoeOut(self, tensor_x):
        tensor_x_numpy = tensor_x.numpy()
        gate_weight_numpy = self.gate_weight.numpy()
        bmm_w0_numpy = self.bmm_w0.numpy()
        bmm_b0_numpy = self.bmm_b0.numpy()
        bmm_w1_numpy = self.bmm_w1.numpy()
        bmm_b1_numpy = self.bmm_b1.numpy()

        paddle.enable_static()
        with program_guard(Program(), Program()):
            tensor_x = paddle.static.data(
                name="tensor_x",
                shape=(self.batch_size, self.seq_len, self.d_model),
                dtype=self.x_type,
            )
            gate_weight = paddle.static.data(
                name="gate_weight",
                shape=(self.d_model, self.num_expert),
                dtype=np.float32,
            )
            bmm_w0 = paddle.static.data(
                name="bmm_w0",
                shape=(self.num_expert, self.d_model, self.d_feedforward * 2),
                dtype=self.x_type,
            )
            bmm_b0 = paddle.static.data(
                name="bmm_b0",
                shape=(self.num_expert, 1, self.d_feedforward * 2),
                dtype=self.x_type,
            )
            bmm_w1 = paddle.static.data(
                name="bmm_w1",
                shape=(self.num_expert, self.d_feedforward, self.d_model),
                dtype=self.x_type,
            )
            bmm_b1 = paddle.static.data(
                name="bmm_b1",
                shape=(self.num_expert, 1, self.d_model),
                dtype=self.x_type,
            )

            fused_out = fused_moe(
                tensor_x,
                gate_weight,
                bmm_w0,
                bmm_w1,
                bmm_b0,
                None,
                bmm_b1,
                None,
                self.quant_method,
                self.top_k,
                self.norm_topk_prob,
            )
            exe = base.Executor()
            feed = {
                "tensor_x": tensor_x_numpy,
                "gate_weight": gate_weight_numpy,
                "bmm_w0": bmm_w0_numpy,
                "bmm_b0": bmm_b0_numpy,
                "bmm_w1": bmm_w1_numpy,
                "bmm_b1": bmm_b1_numpy,
            }
            res = exe.run(
                feed=feed,
                fetch_list=[fused_out],
            )
        paddle.disable_static()
        final_hidden_states = np.reshape(
            res[0], [self.batch_size, self.seq_len, self.d_model]
        )
        return final_hidden_states

    def GetBaselineOut(self, hidden_states):
        paddle.disable_static(place=paddle.CUDAPlace(0))
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = paddle.reshape(hidden_states, [-1, hidden_dim])
        router_logits = self.gate(hidden_states).cast(paddle.float32)

        routing_weights = F.softmax(router_logits, axis=-1, dtype='float32')
        routing_weights, selected_experts = paddle.topk(
            routing_weights, self.top_k, axis=-1
        )
        # mixtral true, qwen_moe false
        if self.norm_topk_prob:
            routing_weights /= paddle.sum(
                routing_weights, axis=-1, keepdim=True
            )
        # we cast back to the input dtype
        routing_weights = routing_weights.cast(np.float32)

        final_hidden_states = paddle.zeros_like(hidden_states)

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = paddle.transpose(
            F.one_hot(selected_experts, num_classes=self.num_expert), [2, 1, 0]
        )

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_expert):
            expert_layer = self.experts[expert_idx]
            idx, top_x = paddle.where(expert_mask[expert_idx])

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = paddle.index_select(
                hidden_states, top_x, axis=0
            ).reshape([-1, hidden_dim])
            current_hidden_states = (
                expert_layer(current_state, expert_idx)
                * routing_weights[top_x, idx]
            )
            # Use scatter to accumulate the results
            paddle.index_add_(
                x=final_hidden_states,
                index=top_x.squeeze(),
                axis=0,
                value=current_hidden_states.to(hidden_states.dtype),
            )

        final_hidden_states = paddle.reshape(
            final_hidden_states, [batch_size, sequence_length, hidden_dim]
        )
        return final_hidden_states

    def test_fused_moe_op_new(self):
        ref_out = self.GetBaselineOut(self.tensor_x).cast(np.float32)
        fused_moe_out = self.GetFusedMoeOut(self.tensor_x)
        np.testing.assert_allclose(
            ref_out, fused_moe_out, rtol=self.rtol, atol=self.atol
        )


@unittest.skipIf(
    not paddle.is_compiled_with_cuda()
    or get_cuda_version() < 11030
    or paddle.device.cuda.get_device_capability()[0] < 8,
    "FusedMoe requires CUDA >= 11.2 and CUDA_ARCH >= 8",
)
class TestFusedMoEOpStaticWint8(TestFusedMoEOpStatic):
    def config(self):
        super().config()
        self.rtol = 1e-2
        self.atol = 1e-2
        self.quant_method = "weight_only_int8"

    def GetWintData(self):
        if self.quant_method == "None":
            return
        fc0_expert_weights_for_ref_list = []
        scale0 = []
        for i in range(self.num_expert):
            fc0_expert_weights_for_ref_i, fc0_expert_weights_scale_for_ref_i = (
                weight_quantize(self.bmm_w0[i], algo=self.quant_method)
            )
            fc0_expert_weights_for_ref_list.append(
                fc0_expert_weights_for_ref_i.reshape(
                    [self.d_model, self.d_feedforward * 2]
                )
            )
            scale0.append(fc0_expert_weights_scale_for_ref_i)
        self.bmm_w0 = paddle.to_tensor(fc0_expert_weights_for_ref_list)
        self.scale0 = paddle.to_tensor(scale0)

        fc1_expert_weights_for_ref_list = []
        scale1 = []
        for i in range(self.num_expert):
            fc1_expert_weights_for_ref_i, fc1_expert_weights_scale_for_ref_i = (
                weight_quantize(self.bmm_w1[i], algo=self.quant_method)
            )
            fc1_expert_weights_for_ref_list.append(
                fc1_expert_weights_for_ref_i.reshape(
                    [self.d_feedforward, self.d_model]
                )
            )
            scale1.append(fc1_expert_weights_scale_for_ref_i)
        self.bmm_w1 = paddle.to_tensor(fc1_expert_weights_for_ref_list)
        self.scale1 = paddle.to_tensor(scale1)

    def GetFusedMoeOut(self, tensor_x):
        self.GetWintData()
        tensor_x_numpy = tensor_x.numpy()
        gate_weight_numpy = self.gate_weight.numpy()
        bmm_w0_numpy = self.bmm_w0.numpy()
        bmm_b0_numpy = self.bmm_b0.numpy()
        bmm_w1_numpy = self.bmm_w1.numpy()
        bmm_b1_numpy = self.bmm_b1.numpy()
        scale0_numpy = self.scale0.numpy()
        scale1_numpy = self.scale1.numpy()

        paddle.enable_static()
        with program_guard(Program(), Program()):
            tensor_x = paddle.static.data(
                name="tensor_x",
                shape=(self.batch_size, self.seq_len, self.d_model),
                dtype=self.dtype,
            )
            gate_weight = paddle.static.data(
                name="gate_weight",
                shape=(self.d_model, self.num_expert),
                dtype=np.float32,
            )
            bmm_w0 = paddle.static.data(
                name="bmm_w0",
                shape=(self.num_expert, self.d_model, self.d_feedforward * 2),
                dtype=np.int8,
            )
            bmm_b0 = paddle.static.data(
                name="bmm_b0",
                shape=(self.num_expert, 1, self.d_feedforward * 2),
                dtype=self.dtype,
            )
            bmm_w1 = paddle.static.data(
                name="bmm_w1",
                shape=(self.num_expert, self.d_feedforward, self.d_model),
                dtype=np.int8,
            )
            bmm_b1 = paddle.static.data(
                name="bmm_b1",
                shape=(self.num_expert, 1, self.d_model),
                dtype=self.dtype,
            )
            scale0 = paddle.static.data(
                name="scale0",
                shape=(self.num_expert, self.d_feedforward * 2),
                dtype=self.dtype,
            )
            scale1 = paddle.static.data(
                name="scale1",
                shape=(self.num_expert, self.d_model),
                dtype=self.dtype,
            )

            fused_out = fused_moe(
                tensor_x,
                gate_weight,
                bmm_w0,
                bmm_w1,
                bmm_b0,
                scale0,
                bmm_b1,
                scale1,
                self.quant_method,
                self.top_k,
                self.norm_topk_prob,
            )
            exe = base.Executor()
            feed = {
                "tensor_x": tensor_x_numpy,
                "gate_weight": gate_weight_numpy,
                "bmm_w0": bmm_w0_numpy,
                "bmm_b0": bmm_b0_numpy,
                "bmm_w1": bmm_w1_numpy,
                "bmm_b1": bmm_b1_numpy,
                "scale0": scale0_numpy,
                "scale1": scale1_numpy,
            }
            res = exe.run(
                feed=feed,
                fetch_list=[fused_out],
            )
        paddle.disable_static()
        final_hidden_states = np.reshape(
            res[0], [self.batch_size, self.seq_len, self.d_model]
        )
        return final_hidden_states


if __name__ == "__main__":
    unittest.main()
