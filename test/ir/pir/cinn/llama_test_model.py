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

import math
import sys
from os.path import dirname

import paddle
import paddle.nn.functional as F
from paddle import nn
from paddle.incubate.nn.functional import swiglu

sys.path.append(dirname(__file__))


class LlamaConfig:
    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        max_position_embeddings=2048,
        seq_length=2048,
        num_hidden_layers=1,
        num_attention_heads=32,
        num_key_value_heads=32,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.seq_length = seq_length
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache


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


def _make_causal_mask(input_ids_shape, past_key_values_length):
    """
    Make causal mask used for self-attention
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


def get_triangle_upper_mask(x, mask=None):
    if mask is not None:
        return mask
    # [bsz, n_head, q_len, kv_seq_len]
    shape = x.shape
    #  [bsz, 1, q_len, kv_seq_len]
    shape[1] = 1
    mask = paddle.full(shape, paddle.finfo(x.dtype).min, dtype=x.dtype)
    mask = paddle.triu(mask, diagonal=1)
    mask.stop_gradient = True
    return mask


def scaled_dot_product_attention(
    query_states,
    config,
    key_states,
    value_states,
    attention_mask,
    output_attentions,
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

    # NOTE: we only call get_triangle_upper_mask under PP setup
    # FIXME ZHUI when we use pipeline parallel, the attention_mask can be None
    # we just make it triangle_upper_mask
    if attention_mask is None:
        attention_mask = get_triangle_upper_mask(attn_weights)
    attention_mask = attention_mask.reshape([bsz, 1, q_len, kv_seq_len])

    attn_weights = attn_weights + attention_mask
    attn_weights = F.softmax(attn_weights, axis=-1, dtype="float32").astype(
        query_states.dtype
    )

    attn_output = paddle.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose([0, 2, 1, 3])

    attn_output = attn_output.reshape([bsz, q_len, head_dim * num_heads])
    return (attn_output, attn_weights) if output_attentions else attn_output


class LlamaMLP(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias_attr=False
        )
        self.up_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias_attr=False
        )
        self.down_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias_attr=False
        )

    def forward(self, x):
        x = swiglu(self.gate_proj(x), self.up_proj(x))
        out = self.down_proj(x)
        return out


class LlamaRMSNorm(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.weight = paddle.create_parameter(
            shape=[self.hidden_size],
            dtype=paddle.get_default_dtype(),
            default_initializer=nn.initializer.Constant(0.2),
        )
        self.variance_epsilon = config.rms_norm_eps
        self.config = config

    def forward(self, hidden_states):
        hidden_states = hidden_states.astype("float32")
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = (
            paddle.rsqrt(variance + self.variance_epsilon) * hidden_states
        )

        if self.weight.dtype in [paddle.float16, paddle.bfloat16]:
            hidden_states = paddle.cast(hidden_states, self.weight.dtype)
        return hidden_states * self.weight


class LlamaAttention(nn.Layer):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig):
        super().__init__()

        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads

        self.head_dim = self.hidden_size // config.num_attention_heads

        self.num_key_value_heads = config.num_key_value_heads
        assert config.num_attention_heads // config.num_key_value_heads
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.gqa_or_mqa = (
            config.num_attention_heads != config.num_key_value_heads
        )

        self.max_position_embeddings = config.max_position_embeddings
        self.seq_length = config.seq_length

        self.q_proj = nn.Linear(
            self.hidden_size,
            self.hidden_size,
            bias_attr=False,
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.config.num_key_value_heads * self.head_dim,
            bias_attr=False,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.config.num_key_value_heads * self.head_dim,
            bias_attr=False,
        )

        self.o_proj = nn.Linear(
            self.hidden_size,
            self.hidden_size,
            bias_attr=False,
        )

        self._init_rope()

    def _init_rope(self):
        self.rotary_emb = LlamaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
        )

    def forward(
        self,
        hidden_states,
        position_ids: tuple[paddle.Tensor] | None = None,
        past_key_value: tuple[paddle.Tensor] | None = None,
        attention_mask: paddle.Tensor | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> tuple[
        paddle.Tensor, paddle.Tensor | None, tuple[paddle.Tensor] | None
    ]:
        """Input shape: Batch x Time x Channel"""
        # [bs, seq_len, num_head * head_dim] -> [seq_len / n, bs, num_head * head_dim] (n is model parallelism)

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        target_query_shape = [0, 0, self.num_heads, self.head_dim]
        target_key_value_shape = [0, 0, self.num_key_value_heads, self.head_dim]
        query_states = query_states.reshape(shape=target_query_shape)
        key_states = key_states.reshape(shape=target_key_value_shape)
        value_states = value_states.reshape(shape=target_key_value_shape)

        kv_seq_len = key_states.shape[-3]

        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-3]

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )

        # [bs, seq_len, num_head, head_dim]
        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = paddle.concat([past_key_value[0], key_states], axis=1)
            value_states = paddle.concat(
                [past_key_value[1], value_states], axis=1
            )

        past_key_value = (key_states, value_states) if use_cache else None

        outputs = scaled_dot_product_attention(
            query_states,
            self.config,
            key_states,
            value_states,
            attention_mask,
            output_attentions,
        )
        if output_attentions:
            attn_output, attn_weights = outputs
        else:
            attn_output = outputs

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        outputs = (attn_output,)

        if output_attentions:
            outputs += (attn_weights,)

        if use_cache:
            outputs += (past_key_value,)

        if type(outputs) is tuple and len(outputs) == 1:
            outputs = outputs[0]

        return outputs


class LlamaDecoderLayer(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config)
        self.post_attention_layernorm = LlamaRMSNorm(config)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        position_ids: tuple[paddle.Tensor] | None = None,
        attention_mask: paddle.Tensor | None = None,
        output_attentions: bool | None = False,
        past_key_value: tuple[paddle.Tensor] | None = None,
        use_cache: bool | None = False,
    ) -> tuple[paddle.Tensor, tuple[paddle.Tensor, paddle.Tensor] | None]:
        """
        Args:
            hidden_states (`paddle.Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`paddle.Tensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `cache` key value states are returned and can be used to speed up decoding
                (see `cache`).
            cache (`Tuple(paddle.Tensor)`, *optional*): cached past key and value projection states
        """

        # [bs * seq_len, embed_dim] -> [seq_len * bs / n, embed_dim] (sequence_parallel)
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        outputs = self.self_attn(
            hidden_states,
            position_ids,
            past_key_value,
            attention_mask,
            output_attentions,
            use_cache,
        )

        if type(outputs) is tuple:
            hidden_states = outputs[0]
        else:
            hidden_states = outputs

        if output_attentions:
            self_attn_weights = outputs[1]

        if use_cache:
            present_key_value = outputs[2 if output_attentions else 1]

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        # remove empty tuple for pipeline parallel
        if type(outputs) is tuple and len(outputs) == 1:
            outputs = outputs[0]

        return outputs


class LlamaModel(nn.Layer):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size

        self.embed_tokens = nn.Embedding(
            self.vocab_size,
            self.hidden_size,
        )

        self.layers = nn.LayerList(
            [LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config)

    @staticmethod
    def _prepare_decoder_attention_mask(
        attention_mask, input_shape, past_key_values_length, dtype
    ):
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            if len(attention_mask.shape) == 2:
                expanded_attn_mask = _expand_2d_mask(
                    attention_mask, dtype, tgt_length=input_shape[-1]
                )
                # For decoding phase in generation, seq_length = 1, we don't need to add causal mask
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
            # if attention_mask is already 4-D, do nothing
            else:
                expanded_attn_mask = attention_mask
        else:
            expanded_attn_mask = _make_causal_mask(
                input_shape, past_key_values_length=past_key_values_length
            )
        # Convert bool attention_mask to float attention mask, which will be added to attention_scores later
        expanded_attn_mask = paddle.where(
            expanded_attn_mask, 0.0, paddle.finfo(dtype).min
        ).astype(dtype)
        return expanded_attn_mask

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        attention_mask=None,
        use_cache=None,
    ):
        output_attentions = False
        output_hidden_states = False
        use_cache = (
            use_cache if use_cache is not None else self.config.use_cache
        )

        # retrieve input_ids
        if input_ids is not None:
            batch_size, seq_length = input_ids.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids")

        past_key_values = tuple([None] * len(self.layers))
        # NOTE: to make cache can be clear in-time
        past_key_values = list(past_key_values)

        seq_length_with_past = seq_length
        cache_length = 0
        if past_key_values[0] is not None:
            cache_length = paddle.shape(past_key_values[0][0])[1]
            seq_length_with_past += cache_length
        inputs_embeds = self.embed_tokens(input_ids)

        # embed positions
        if attention_mask is None:
            # [bs, seq_len]
            attention_mask = paddle.ones(
                (batch_size, seq_length_with_past), dtype=paddle.bool
            )

        if position_ids is None:
            position_ids = paddle.arange(seq_length, dtype="int64").expand(
                (batch_size, seq_length)
            )

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            cache_length,
            inputs_embeds.dtype,
        )  # [bs, 1, seq_len, seq_len]

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, (decoder_layer) in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )

            has_gradient = not hidden_states.stop_gradient

            layer_outputs = decoder_layer(
                hidden_states,
                position_ids,
                attention_mask,
                output_attentions,
                past_key_value,
                use_cache,
            )

            # NOTE: clear outdate cache after it has been used for memory saving
            past_key_value = past_key_values[idx] = None
            if type(layer_outputs) is tuple:
                hidden_states = layer_outputs[0]
            else:
                hidden_states = layer_outputs

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if use_cache:
                next_decoder_cache += (
                    layer_outputs[2 if output_attentions else 1],
                )

        hidden_states = self.norm(hidden_states)

        return hidden_states
