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

import math
import os

import numpy as np

import paddle
import paddle.distributed as dist
import paddle.nn.functional as F
from paddle import nn
from paddle.distributed import to_distributed
from paddle.distributed.auto_parallel.high_level_api import ToDistributedConfig

EPOCHS = 1
VOCAB_SIZE = 8000
BATCH_NUM = 2
BATCH_SIZE = 4
HIDDEN_SIZE = 2048
INTERMEDIATE_SIZE = 4096
SEQ_LENGTH = 1024
N_HEAD = 32
NUM_HIDDEN_LAYERS = 4


def create_numpy_like_random(name):
    return paddle.ParamAttr(
        name=name, initializer=paddle.nn.initializer.Uniform(-0.1, 0.1)
    )


class LlamaRotaryEmbedding(nn.Layer):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        # [dim / 2]
        self.inv_freq = 1.0 / (
            self.base
            ** (
                paddle.cast(paddle.arange(0, self.dim, 2), dtype="float32")
                / self.dim
            )
        )
        self._set_cos_sin_cache(seq_len=max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len):
        self.max_seq_len_cached = seq_len
        # [seq_len]
        t = paddle.arange(seq_len, dtype="float32")
        # [seq_len, dim/2]
        freqs = paddle.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        # [seq_len, dim]
        emb = paddle.concat([freqs, freqs], axis=-1)
        # [1, seqlen, 1, dim]
        self.cos_cached = emb.cos()[None, :, None, :]
        self.sin_cached = emb.sin()[None, :, None, :]

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        cos = self.cos_cached[:, :seq_len, :, :]
        sin = self.sin_cached[:, :seq_len, :, :]
        return (
            cos.cast(x.dtype) if cos.dtype != x.dtype else cos,
            sin.cast(x.dtype) if sin.dtype != x.dtype else sin,
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return paddle.concat([-x2, x1], axis=-1)  # shape is the same as x


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    if position_ids is None:
        # Note: Only for LlamaForCausalLMPipe model pretraining
        cos = cos[:, : q.shape[1], :, :]  # [bs, seq_len, 1, dim]
        sin = sin[:, : q.shape[1], :, :]  # [bs, seq_len, 1, dim]
    else:
        cos = cos.squeeze(axis=[0, 2])  # [seq_len, dim]
        sin = sin.squeeze(axis=[0, 2])  # [seq_len, dim]
        cos = cos[position_ids].unsqueeze(2)  # [bs, seq_len, 1, dim]
        sin = sin[position_ids].unsqueeze(2)  # [bs, seq_len, 1, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def scaled_dot_product_attention(
    query_states,
    key_states,
    value_states,
    attention_mask,
):
    bsz, q_len, num_heads, head_dim = query_states.shape
    _, kv_seq_len, _, _ = value_states.shape

    #  [ bz, seqlen, nhead, head_dim] -> [bs, nhead, seq_len, head_dim]
    query_states = paddle.transpose(query_states, [0, 2, 1, 3])
    # merge with the next transpose
    key_states = paddle.transpose(key_states, [0, 2, 1, 3])
    value_states = paddle.transpose(value_states, [0, 2, 1, 3])

    # matmul and divide by sqrt(head_dim)
    attn_weights = paddle.matmul(
        query_states / math.sqrt(head_dim), key_states.transpose([0, 1, 3, 2])
    )

    attention_mask = attention_mask.reshape([bsz, 1, q_len, kv_seq_len])

    attn_weights = attn_weights + attention_mask
    if not paddle.in_dynamic_mode():
        attn_weights = F.softmax(attn_weights, axis=-1, dtype="float32").astype(
            query_states.dtype
        )
    else:
        with paddle.amp.auto_cast(False):
            attn_weights = F.softmax(
                attn_weights, axis=-1, dtype="float32"
            ).astype(query_states.dtype)

    attn_output = paddle.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose([0, 2, 1, 3])

    attn_output = attn_output.reshape([bsz, q_len, head_dim * num_heads])

    return attn_output


class LlamaAttention(nn.Layer):
    def __init__(self, param_prefix="", hidden_size=HIDDEN_SIZE, n_head=N_HEAD):
        super().__init__()
        weight_attr_0 = create_numpy_like_random(param_prefix + "_0")
        weight_attr_1 = create_numpy_like_random(param_prefix + "_1")
        weight_attr_2 = create_numpy_like_random(param_prefix + "_2")
        weight_attr_3 = create_numpy_like_random(param_prefix + "_3")
        self.hidden_size = hidden_size
        self.num_heads = n_head
        self.head_dim = hidden_size // n_head
        self.q_proj = nn.Linear(
            hidden_size, hidden_size, weight_attr_0, bias_attr=False
        )
        self.k_proj = nn.Linear(
            hidden_size, hidden_size, weight_attr_1, bias_attr=False
        )
        self.v_proj = nn.Linear(
            hidden_size, hidden_size, weight_attr_2, bias_attr=False
        )
        self.o_proj = nn.Linear(
            hidden_size, hidden_size, weight_attr_3, bias_attr=False
        )
        self.rotary_emb = LlamaRotaryEmbedding(
            self.head_dim, max_position_embeddings=SEQ_LENGTH, base=10000
        )

    def forward(
        self,
        hidden_states,
        position_ids=None,
        attention_mask=None,
    ):
        # mix_layer = self.qkv_proj(x)
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # target_shape = [0, 0, self.num_heads, 3 * self.head_dim]
        target_query_shape = [0, 0, self.num_heads, self.head_dim]
        target_key_value_shape = [0, 0, self.num_heads, self.head_dim]

        # mix_layer = paddle.reshape(mix_layer, target_shape)
        query_states = query_states.reshape(shape=target_query_shape)
        key_states = key_states.reshape(shape=target_key_value_shape)
        value_states = value_states.reshape(shape=target_key_value_shape)
        kv_seq_len = key_states.shape[-3]

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )

        output = scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attention_mask,
        )

        attn_output = output
        attn_output = self.o_proj(attn_output)

        return attn_output


class LlamaMlp(nn.Layer):
    def __init__(
        self,
        param_prefix="",
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        weight_attr_0 = create_numpy_like_random(param_prefix + "_0")
        weight_attr_1 = create_numpy_like_random(param_prefix + "_1")
        weight_attr_2 = create_numpy_like_random(param_prefix + "_2")

        self.gate_proj = nn.Linear(
            hidden_size, intermediate_size, weight_attr_1, bias_attr=False
        )
        self.up_proj = nn.Linear(
            hidden_size, intermediate_size, weight_attr_0, bias_attr=False
        )
        self.down_proj = nn.Linear(
            intermediate_size, hidden_size, weight_attr_2, bias_attr=False
        )

    def forward(self, x):
        x = paddle.incubate.nn.functional.swiglu(
            self.gate_proj(x), self.up_proj(x)
        )
        out = self.down_proj(x)
        return out


class LlamaRMSNorm(nn.Layer):
    def __init__(self, hidden_size=HIDDEN_SIZE):
        super().__init__()
        self.hidden_size = hidden_size
        self.weight = paddle.create_parameter(
            shape=[self.hidden_size],
            dtype=paddle.get_default_dtype(),
            default_initializer=nn.initializer.Constant(1.0),
        )
        self.variance_epsilon = 1.0

    def forward(self, hidden_states):
        with paddle.amp.auto_cast(False):
            hidden_states = hidden_states.astype("float32")
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = (
                paddle.rsqrt(variance + self.variance_epsilon) * hidden_states
            )
        if self.weight.dtype in [paddle.float16, paddle.bfloat16]:
            hidden_states = paddle.cast(hidden_states, self.weight.dtype)
        return hidden_states * self.weight


class LlamaDecoderLayer(nn.Layer):
    def __init__(
        self,
        param_prefix="",
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.self_attn = LlamaAttention(param_prefix + "_att", hidden_size)
        self.mlp = LlamaMlp(param_prefix + "_mlp")
        self.input_layernorm = LlamaRMSNorm(hidden_size)
        self.post_attn_layernorm = LlamaRMSNorm(hidden_size)

    def forward(
        self,
        hidden_states,
        position_ids=None,
        attention_mask=None,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states, position_ids, attention_mask
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attn_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


def _expand_2d_mask(mask, dtype, tgt_length):
    """
    Expands attention_mask from `[batch_size, src_length]` to `[batch_size, 1, tgt_length, src_length]`.
    """
    batch_size, src_length = mask.shape[0], mask.shape[-1]
    tgt_length = tgt_length if tgt_length is not None else src_length

    mask = mask[:, None, None, :].astype("bool")
    mask.stop_gradient = True
    expanded_mask = mask.expand([batch_size, 1, tgt_length, src_length])

    return expanded_mask


def _make_causal_mask(input_ids_shape, past_key_values_length):
    """
    Make casual mask used for self-attention
    """
    batch_size, target_length = input_ids_shape  # target_length: seq_len

    mask = paddle.tril(
        paddle.ones((target_length, target_length), dtype="bool")
    )

    if past_key_values_length > 0:
        # [tgt_len, tgt_len + past_len]
        mask = paddle.concat(
            [
                paddle.ones(
                    [target_length, past_key_values_length], dtype="bool"
                ),
                mask,
            ],
            axis=-1,
        )

    # [bs, 1, tgt_len, tgt_len + past_len]
    return mask[None, None, :, :].expand(
        [batch_size, 1, target_length, target_length + past_key_values_length]
    )


def _prepare_decoder_attention_mask(
    attention_mask, input_shape, past_key_values_length, dtype
):
    if attention_mask is not None:
        if len(attention_mask.shape) == 2:
            expanded_attn_mask = _expand_2d_mask(
                attention_mask, dtype, tgt_length=input_shape[-1]
            )
            if input_shape[-1] > 1:
                combined_attention_mask = _make_causal_mask(
                    input_shape,
                    past_key_values_length=past_key_values_length,
                )
                expanded_attn_mask = (
                    expanded_attn_mask & combined_attention_mask
                )
        # [bsz, seq_len, seq_len] -> [bsz, 1, seq_len, seq_len]
        elif len(attention_mask.shape) == 3:
            expanded_attn_mask = attention_mask.unsqueeze(1).astype("bool")
        else:
            expanded_attn_mask = attention_mask
    else:
        expanded_attn_mask = _make_causal_mask(
            input_shape, past_key_values_length=past_key_values_length
        )
    expanded_attn_mask = paddle.where(
        expanded_attn_mask, 0.0, paddle.finfo(dtype).min
    ).astype(dtype)
    return expanded_attn_mask


class LlamaModel(nn.Layer):
    def __init__(
        self,
        param_prefix="",
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.embed_tokens = nn.Embedding(
            vocab_size,
            hidden_size,
        )

        self.layers = nn.LayerList(
            [
                LlamaDecoderLayer(param_prefix + "_decoder_" + str(i))
                for i in range(NUM_HIDDEN_LAYERS)
            ]
        )
        self.norm = LlamaRMSNorm(hidden_size)

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        attention_mask=None,
    ):
        batch_size, seq_length = input_ids.shape

        inputs_embeds = self.embed_tokens(input_ids)

        # embed positions
        attention_mask = paddle.ones(
            (batch_size, seq_length), dtype=paddle.bool
        )
        if position_ids is None:
            position_ids = paddle.arange(seq_length, dtype="int64").expand(
                (batch_size, seq_length)
            )

        attention_mask = _prepare_decoder_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            0,
            inputs_embeds.dtype,
        )  # [bs, 1, seq_len, seq_len]

        hidden_states = inputs_embeds

        for idx, (decoder_layer) in enumerate(self.layers):

            layer_outputs = decoder_layer(
                hidden_states,
                position_ids,
                attention_mask,
            )

            hidden_states = layer_outputs

        hidden_states = self.norm(hidden_states)

        return hidden_states


class LlamaPretrainingCriterion(paddle.nn.Layer):
    """
    Criterion for Llama.
    It calculates the final loss.
    """

    def __init__(
        self,
        param_prefix="",
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
    ):
        super().__init__()
        self.ignore_index = -100
        self.loss_func = paddle.nn.CrossEntropyLoss(
            reduction="none", ignore_index=self.ignore_index
        )

    def forward(self, prediction_scores, masked_lm_labels):
        with paddle.amp.auto_cast(False):
            masked_lm_loss = self.loss_func(
                prediction_scores.astype("float32"),
                masked_lm_labels.unsqueeze(2),
            )

            binary_sequence = paddle.where(
                masked_lm_loss > 0,
                paddle.ones_like(masked_lm_loss),
                paddle.zeros_like(masked_lm_loss),
            )
            count = paddle.sum(binary_sequence)
            if count == 0:
                loss = paddle.sum(masked_lm_loss * binary_sequence)
            else:
                loss = paddle.sum(masked_lm_loss * binary_sequence) / count

        return loss


class LlamaLMHead(paddle.nn.Layer):
    def __init__(
        self,
        param_prefix="",
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.weight = self.create_parameter(
            shape=[hidden_size, vocab_size],
            dtype=paddle.get_default_dtype(),
        )

    def forward(self, hidden_states, tensor_parallel_output=None):
        logits = paddle.matmul(hidden_states, self.weight, transpose_y=False)
        return logits


class LlamaForCausalLM(paddle.nn.Layer):

    def __init__(
        self,
        param_prefix="",
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
    ):
        super().__init__()
        self.llama = LlamaModel(
            param_prefix + "_llama", vocab_size, hidden_size, intermediate_size
        )
        self.lm_head = LlamaLMHead(
            param_prefix + "_lm_head",
            vocab_size,
            hidden_size,
            intermediate_size,
        )
        self.criterion = LlamaPretrainingCriterion(
            param_prefix + "_criterion",
            vocab_size,
            hidden_size,
            intermediate_size,
        )

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        attention_mask=None,
        labels=None,
    ):

        outputs = self.llama(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )

        logits = self.lm_head(outputs)

        loss = None
        if labels is not None:
            loss = self.criterion(logits, labels)

        return (loss, logits)


class RandomDataset(paddle.io.Dataset):
    def __init__(self, inputs, labels, num_samples):
        self.inputs = inputs
        self.labels = labels
        self.num_samples = num_samples

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

    def __len__(self):
        return self.num_samples


class TestLlamaDecoderForSemiAutoParallel:
    def __init__(self):
        self._dtype = os.getenv("dtype", "float32")
        self._backend = os.getenv("backend", "gpu")
        self._seed = eval(os.getenv("seed", "2023"))

        self._device_num = os.getenv("num_of_devices", 8)
        self._node_num = 1

        np.random.seed(self._seed)
        paddle.seed(self._seed)
        self._model = LlamaForCausalLM("demo_llama")

        # ensure that input data between dp is different and data within dp is the same
        self._mesh = dist.ProcessMesh(
            [[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dim_names=["pp", "dp", "mp"]
        )
        if "dp" in self._mesh.dim_names:
            dp_seed = self._mesh.get_rank_by_dim_and_process_id(
                "dp", dist.get_rank()
            )
        else:
            dp_seed = 0
        np.random.seed(self._seed + dp_seed)
        paddle.seed(self._seed + dp_seed)
        self._input_seqs = np.random.randint(
            low=0, high=1024, size=(BATCH_SIZE * BATCH_NUM, SEQ_LENGTH)
        ).astype("int64")
        self._labels = np.random.randint(
            low=0, high=1024, size=(BATCH_SIZE * BATCH_NUM, SEQ_LENGTH)
        ).astype("int64")
        self._dataset = RandomDataset(
            self._input_seqs, self._labels, BATCH_SIZE * BATCH_NUM
        )
        self._sampler = paddle.io.BatchSampler(
            self._dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True
        )
        self._loader = paddle.io.DataLoader(
            self._dataset, batch_sampler=self._sampler
        )
        self._opt = paddle.optimizer.SGD(
            learning_rate=0.1, parameters=self._model.parameters()
        )

        paddle.set_device(self._backend)

    def test_to_distributed_api(self):
        # # config: sequence_parallel
        dist_config = ToDistributedConfig()
        dist_config.sequence_parallel = True

        # # wrap model by using **to_distributed**
        dist_model, dist_opt, dist_loader = to_distributed(
            self._model,
            self._opt,
            self._loader,
            self._device_num,
            self._node_num,
            dist_config,
        )

        for epoch in range(EPOCHS):
            dist_model.train()
            for i, data in enumerate(dist_loader()):
                inputs, labels = data
                loss, _ = dist_model(inputs, labels=labels)
                loss.backward()
                dist_opt.step()
                dist_opt.clear_grad()

    def run_test_case(self):
        if self._backend == "gpu":
            cuda_version_main = int(paddle.version.cuda().split(".")[0])
            device_prop_main = paddle.device.cuda.get_device_capability()[0]
            if cuda_version_main >= 11 and device_prop_main >= 8:
                self.test_to_distributed_api()


if __name__ == '__main__':
    TestLlamaDecoderForSemiAutoParallel().run_test_case()
