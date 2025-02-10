# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.distributed as dist
from paddle import nn
from paddle.distributed import fleet


class FusedLlamaRMSNorm(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.weight = paddle.create_parameter(
            shape=[self.hidden_size],
            dtype=paddle.get_default_dtype(),
            default_initializer=nn.initializer.Constant(1.0),
        )
        self.variance_epsilon = config.rms_norm_eps
        self.config = config

    def forward(self, hidden_states):
        return paddle.incubate.nn.functional.fused_rms_norm(
            hidden_states,
            self.weight,
            None,
            self.variance_epsilon,
            begin_norm_axis=1,
        )[0]


def _set_var_distributed(var):
    if var is None:
        return
    var.is_distributed = True
    if not paddle.in_dynamic_mode():
        # NOTE: use current_block and find_var_recursive to support while_loop
        startup_block = paddle.static.default_startup_program().current_block()
        main_block = paddle.static.default_main_program().current_block()
        startup_block._find_var_recursive(var.name).is_distributed = True
        main_block._find_var_recursive(var.name).is_distributed = True


class FusedMultiTransformerConfig:
    def __init__(
        self,
        embed_dim,
        num_heads,
        dim_feedforward,
        quant_type="",
        dropout_rate=0.0,
        activation="gelu",
        norm_type="layernorm",
        use_neox_rotary_style=False,
        rope_theta=10000.0,
        normalize_before=True,
        ln_scale_attrs=None,
        ln_bias_attrs=None,
        qkv_weight_attrs=None,
        qkv_weight_scale_attrs=None,
        qkv_bias_attrs=None,
        linear_weight_attrs=None,
        linear_weight_scale_attrs=None,
        linear_bias_attrs=None,
        ffn_ln_scale_attrs=None,
        ffn_ln_bias_attrs=None,
        gate_weight_attrs=None,
        gate_bias_attrs=None,
        up_weight_attrs=None,
        up_bias_attrs=None,
        ffn1_weight_attrs=None,
        ffn1_weight_scale_attrs=None,
        ffn1_bias_attrs=None,
        ffn1_0_weight_attrs=None,
        ffn1_1_weight_attrs=None,
        ffn1_0_bias_attrs=None,
        ffn1_1_bias_attrs=None,
        ffn2_weight_attrs=None,
        ffn2_weight_scale_attrs=None,
        ffn2_bias_attrs=None,
        linear_shift_attrs=None,
        linear_smooth_attrs=None,
        ffn2_shift_attrs=None,
        ffn2_smooth_attrs=None,
        quant_round_type=0,
        quant_max_bound=127.0,
        quant_min_bound=-127.0,
        epsilon=1e-5,
        residual_alpha=1.0,
        num_layers=-1,
        nranks=1,
        trans_qkvw=True,
        ring_id=-1,
        kv_num_heads=-1,
        rank_id=-1,
    ):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if kv_num_heads > 0:
            self.kv_num_heads = kv_num_heads
        else:
            self.kv_num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.norm_type = norm_type
        self.rope_theta = rope_theta
        self.use_neox_rotary_style = use_neox_rotary_style
        self.normalize_before = normalize_before
        self.ln_scale_attrs = ln_scale_attrs
        self.ln_bias_attrs = ln_bias_attrs
        self.qkv_weight_attrs = qkv_weight_attrs
        self.qkv_weight_scale_attrs = qkv_weight_scale_attrs
        self.qkv_bias_attrs = qkv_bias_attrs
        self.linear_weight_attrs = linear_weight_attrs
        self.linear_weight_scale_attrs = linear_weight_scale_attrs
        self.linear_bias_attrs = linear_bias_attrs
        self.ffn_ln_scale_attrs = ffn_ln_scale_attrs
        self.ffn_ln_bias_attrs = ffn_ln_bias_attrs
        self.gate_weight_attrs = gate_weight_attrs
        self.gate_bias_attrs = gate_bias_attrs
        self.up_weight_attrs = up_weight_attrs
        self.up_bias_attrs = up_bias_attrs
        self.ffn1_weight_attrs = ffn1_weight_attrs
        self.ffn1_weight_scale_attrs = ffn1_weight_scale_attrs
        self.ffn1_bias_attrs = ffn1_bias_attrs
        self.ffn2_weight_attrs = ffn2_weight_attrs
        self.ffn2_weight_scale_attrs = ffn2_weight_scale_attrs
        self.ffn2_bias_attrs = ffn2_bias_attrs

        self.linear_shift_attrs = linear_shift_attrs
        self.linear_smooth_attrs = linear_smooth_attrs
        self.ffn2_shift_attrs = ffn2_shift_attrs

        self.epsilon = epsilon
        self.residual_alpha = residual_alpha
        self.num_layers = num_layers
        self.nranks = nranks
        self.rank_id = rank_id
        self.trans_qkvw = trans_qkvw
        self.ring_id = ring_id


class FusedMultiTransformerBase(nn.Layer):
    def __init__(self, config: FusedMultiTransformerConfig):
        super().__init__()

        self.config = config
        self._dtype = self._helper.get_default_dtype()
        if self._dtype == "bfloat16":
            self._fuse_kernel_compute_dtype = "bf16"
        elif self._dtype == "float16":
            self._fuse_kernel_compute_dtype = "fp16"
        elif self._dtype == "float32":
            self._fuse_kernel_compute_dtype = "fp32"
        else:
            raise ValueError(
                f"FusedMultiTransformer just support float32, float16 and bfloat16 as default dtype, but received {self._dtype}"
            )
        self._epsilon = config.epsilon
        self._residual_alpha = config.residual_alpha
        self.nranks = config.nranks
        self.norm_type = config.norm_type
        if self.norm_type == "layernorm":
            self.norm_func = paddle.incubate.nn.functional.fused_layer_norm
        elif self.norm_type == "rmsnorm":
            self.norm_func = paddle.incubate.nn.functional.fused_rms_norm
        else:
            raise NotImplementedError(
                "Only support norm type of [layernorm, rmsnorm]"
            )
        self.use_neox_rotary_style = config.use_neox_rotary_style
        self._norm_weight_dtype = (
            "float32" if self.norm_type == "layernorm" else self._dtype
        )

        self.activation = config.activation

        self.embed_dim = config.embed_dim
        self.head_dim = config.embed_dim // config.num_heads
        assert (
            self.head_dim * config.num_heads == config.embed_dim
        ), "embed_dim must be divisible by num_heads"

        # tensor model parallel
        if config.nranks > 1:
            assert config.ring_id != -1
        assert config.num_heads % config.nranks == 0
        assert config.dim_feedforward % config.nranks == 0
        self.num_heads = config.num_heads // config.nranks
        self.kv_num_heads = config.kv_num_heads // config.nranks
        dim_feedforward = config.dim_feedforward // config.nranks
        self.dim_feedforward = dim_feedforward

        self.num_layers = config.num_layers
        assert self.num_layers > 0
        if isinstance(config.qkv_weight_attrs, (list, tuple)):
            assert self.num_layers == len(config.qkv_weight_attrs)

        self.weight_dtype = self._dtype
        self.create_params_type = self.get_weight_create_dype()

        self.ln_scales, self.ln_biases = [], []
        self.qkv_biases = []
        self.linear_biases = []
        self.ffn_ln_scales, self.ffn_ln_biases = [], []
        self.ffn1_biases = []
        self.ffn2_biases = []

        self.init_weight_shape(config)

        for i in range(self.num_layers):
            ln_scale_attr = self.get_attr(config.ln_scale_attrs, i)
            ln_bias_attr = self.get_attr(config.ln_bias_attrs, i)

            qkv_bias_attr = self.get_attr(config.qkv_bias_attrs, i)
            linear_bias_attr = self.get_attr(config.linear_bias_attrs, i)

            ffn_ln_scale_attr = self.get_attr(config.ffn_ln_scale_attrs, i)
            ffn_ln_bias_attr = self.get_attr(config.ffn_ln_bias_attrs, i)
            ffn1_bias_attr = self.get_attr(config.ffn1_bias_attrs, i)
            ffn2_bias_attr = self.get_attr(config.ffn2_bias_attrs, i)

            ln_scale = self.create_parameter(
                attr=ln_scale_attr,
                shape=[config.embed_dim],
                default_initializer=paddle.nn.initializer.Constant(value=1.0),
                dtype=self._norm_weight_dtype,
            )
            ln_bias = None
            if ln_bias_attr:
                ln_bias = self.create_parameter(
                    attr=ln_bias_attr,
                    shape=[config.embed_dim],
                    is_bias=True,
                    dtype=self._norm_weight_dtype,
                )

            qkv_bias = None
            if qkv_bias_attr:
                qkv_bias = self.create_parameter(
                    shape=[
                        (self.num_heads + 2 * self.kv_num_heads) * self.head_dim
                    ],
                    attr=qkv_bias_attr,
                    dtype=self._dtype,
                    is_bias=True,
                )

            linear_bias = None
            if linear_bias_attr:
                linear_bias = self.create_parameter(
                    shape=[config.embed_dim],
                    attr=linear_bias_attr,
                    dtype=self._dtype,
                    is_bias=True,
                )

            ffn_ln_scale = self.create_parameter(
                shape=[config.embed_dim],
                attr=ffn_ln_scale_attr,
                is_bias=False,
                default_initializer=paddle.nn.initializer.Constant(1.0),
                dtype=self._norm_weight_dtype,
            )

            ffn_ln_bias = None
            if ffn_ln_bias_attr:
                ffn_ln_bias = self.create_parameter(
                    shape=[config.embed_dim],
                    attr=ffn_ln_bias_attr,
                    is_bias=True,
                    dtype=self._norm_weight_dtype,
                )

            ffn1_bias = None
            if ffn1_bias_attr:
                ffn1_bias = self.create_parameter(
                    shape=(
                        [dim_feedforward * 2]
                        if self.activation.endswith("glu")
                        else [dim_feedforward]
                    ),
                    attr=ffn1_bias_attr,
                    dtype=self._dtype,
                    is_bias=True,
                )

            ffn2_bias = None
            if ffn2_bias_attr:
                ffn2_bias = self.create_parameter(
                    shape=[config.embed_dim],
                    attr=ffn2_bias_attr,
                    dtype=self._dtype,
                    is_bias=True,
                )

            # tensor model parallel
            if config.nranks > 1:
                # column parallel
                _set_var_distributed(qkv_bias)
                _set_var_distributed(ffn1_bias)

            self.ln_scales.append(ln_scale)
            self.ln_biases.append(ln_bias)
            self.qkv_biases.append(qkv_bias)
            self.linear_biases.append(linear_bias)

            self.ffn_ln_scales.append(ffn_ln_scale)
            self.ffn_ln_biases.append(ffn_ln_bias)
            self.ffn1_biases.append(ffn1_bias)
            self.ffn2_biases.append(ffn2_bias)

            self._add_parameter(ln_scale)
            self._add_parameter(ln_bias)
            self._add_parameter(qkv_bias)
            self._add_parameter(linear_bias)

            self._add_parameter(ffn_ln_scale)
            self._add_parameter(ffn_ln_bias)
            self._add_parameter(ffn1_bias)
            self._add_parameter(ffn2_bias)

        self.dropout_rate = config.dropout_rate
        self.linear = paddle.incubate.nn.functional.fused_linear

    def init_weight(self):
        self.qkv_weights = []
        self.linear_weights = []
        self.gate_weights = []
        self.ffn1_weights = []
        self.ffn2_weights = []

        for i in range(self.num_layers):
            qkv_weight_attr = self.get_attr(self.config.qkv_weight_attrs, i)
            linear_weight_attr = self.get_attr(
                self.config.linear_weight_attrs, i
            )
            gate_weight_attr = self.get_attr(self.config.gate_weight_attrs, i)
            ffn1_weight_attr = self.get_attr(self.config.ffn1_weight_attrs, i)
            ffn2_weight_attr = self.get_attr(self.config.ffn2_weight_attrs, i)

            qkv_weight = self.create_parameter(
                shape=self.qkv_weight_shape,
                attr=qkv_weight_attr,
                dtype=self.create_params_type,
                is_bias=False,
            )
            linear_weight = self.create_parameter(
                shape=self.linear_weight_shape,
                attr=linear_weight_attr,
                dtype=self.create_params_type,
                is_bias=False,
            )
            gate_weight = None
            ffn1_weight = self.create_parameter(
                shape=self.ffn1_weight_shape,
                attr=ffn1_weight_attr,
                dtype=self.create_params_type,
                is_bias=False,
            )

            ffn2_weight = self.create_parameter(
                shape=self.ffn2_weight_shape,
                attr=ffn2_weight_attr,
                dtype=self.create_params_type,
                is_bias=False,
            )

            # tensor model parallel
            if self.config.nranks > 1:
                # column parallel
                _set_var_distributed(qkv_weight)
                _set_var_distributed(ffn1_weight)
                # row parallel
                _set_var_distributed(linear_weight)
                _set_var_distributed(ffn2_weight)

            self.qkv_weights.append(qkv_weight)
            self.linear_weights.append(linear_weight)

            if gate_weight is not None:
                self.gate_weights.append(gate_weight)
            self.ffn1_weights.append(ffn1_weight)
            self.ffn2_weights.append(ffn2_weight)

            self._add_parameter(qkv_weight)
            self._add_parameter(linear_weight)
            if gate_weight is not None:
                self._add_parameter(gate_weight)
            self._add_parameter(ffn1_weight)
            self._add_parameter(ffn2_weight)

    def get_attr(self, attrs, idx):
        if isinstance(attrs, (list, tuple)):
            assert (
                len(attrs) == self.num_layers
            ), f"length of attrs is {len(attrs)} is not equal to self.num_layers {self.num_layers}"
            return attrs[idx]
        return attrs

    def _add_parameter(self, param):
        if param is None:
            return
        assert param.name not in self._parameters
        self._parameters[param.name] = param

    def init_weight_shape(self, config):
        self.qkv_weight_shape = (
            [
                (self.num_heads + 2 * self.kv_num_heads) * self.head_dim,
                self.embed_dim,
            ]
            if config.trans_qkvw
            else [
                self.embed_dim,
                (self.num_heads + 2 * self.kv_num_heads) * self.head_dim,
            ]
        )
        self.linear_weight_shape = [
            self.num_heads * self.head_dim,
            self.embed_dim,
        ]

        self.ffn1_weight_shape = (
            [self.embed_dim, self.dim_feedforward * 2]
            if self.activation.endswith("glu")
            else [self.embed_dim, self.dim_feedforward]
        )
        self.ffn2_weight_shape = [self.dim_feedforward, self.embed_dim]

    def skip_quant(self, layer_name, layer_idx):
        return False

    def get_weight_create_dype(self):
        return self._dtype

    def compute_layernorm_before_qkv(self, src, i):
        if i == 0:
            ln_out = self.norm_func(
                src,
                self.ln_scales[i],
                self.ln_biases[i],
                self._epsilon,
                begin_norm_axis=1,
            )[0]
        else:
            ln_out = src

        return ln_out

    def compute_qkv_linear(self, ln_out, i):
        if (
            paddle.version.cuda() == "False"
            or float(paddle.version.cuda()) < 11.6
        ):
            qkv_out = paddle.matmul(ln_out, self.qkv_weights[i], False, True)
            if self.qkv_biases[i] is not None:
                qkv_out = paddle.add(qkv_out, self.qkv_biases[i])
            return qkv_out
        else:
            # This method requires CUDA version >= 11.6.
            return self.linear(
                ln_out,
                self.qkv_weights[i],
                self.qkv_biases[i],
                transpose_weight=True,
            )

    def compute_qkv(self, src, residual_input, i):
        ln_out = self.compute_layernorm_before_qkv(src, i)
        qkv_out = self.compute_qkv_linear(ln_out, i)
        return qkv_out, residual_input

    def compute_max_len(self, seq_lens_encoder, seq_lens_decoder, cum_offsets):
        if (
            seq_lens_encoder is None
            or seq_lens_decoder is None
            or cum_offsets is None
        ):
            return None, None
        return paddle.incubate.nn.functional.blha_get_max_len(
            seq_lens_encoder,
            seq_lens_decoder,
            cum_offsets,
        )

    def compute_fmha(
        self,
        qkv_out,
        padding_offset,
        seq_lens,
        input_ids,
        rotary_embs,
        rotary_emb_dims,
        caches,
        pre_caches,
        pre_caches_length,
        attn_mask,
        i,
    ):
        bsz = input_ids.shape[0]
        qkv_out = qkv_out.reshape(
            (3 * (paddle.shape(seq_lens)[0]), self.num_heads, -1, self.head_dim)
        )
        q_out, k_out, v_out = paddle.split(qkv_out, 3, axis=0)
        qktv_out = paddle.incubate.nn.functional.variable_length_memory_efficient_attention(
            q_out,
            k_out,
            v_out,
            seq_lens,
            seq_lens + pre_caches_length,
            mask=attn_mask,
            scale=float(self.head_dim**-0.5),
        )

        qktv_out_shape = paddle.shape(qktv_out)
        offset_shape = paddle.shape(padding_offset)
        return paddle.reshape(
            qktv_out, (offset_shape, qktv_out_shape[1] * qktv_out_shape[3])
        )

    def compute_mmha(
        self,
        qkv_out,
        caches,
        attn_mask,
        seq_lens,
        rotary_embs,
        rotary_emb_dims,
        i,
    ):
        return paddle.incubate.nn.functional.masked_multihead_attention(
            x=qkv_out,
            cache_kv=caches[i],
            src_mask=attn_mask,
            sequence_lengths=seq_lens,
            rotary_tensor=rotary_embs,
            rotary_emb_dims=rotary_emb_dims,
            use_neox_rotary_style=self.use_neox_rotary_style,
        )[0]

    def compute_out_linear(self, fmha_out, i):
        return paddle.matmul(fmha_out, self.linear_weights[i])

    def compute_attn(
        self,
        time_step,
        qkv_out,
        padding_offset,
        seq_lens,
        input_ids,
        rotary_embs,
        rotary_emb_dims,
        caches,
        pre_caches,
        pre_caches_length,
        attn_mask,
        i,
        **kwargs,
    ):
        # fmha compute
        if time_step is None:  # context
            fmha_out = self.compute_fmha(
                qkv_out,
                padding_offset,
                seq_lens,
                input_ids,
                rotary_embs,
                rotary_emb_dims,
                caches,
                pre_caches,
                pre_caches_length,
                attn_mask,
                i,
            )
        else:
            fmha_out = self.compute_mmha(
                qkv_out,
                caches,
                attn_mask,
                seq_lens,
                rotary_embs,
                rotary_emb_dims,
                i,
            )
        out_linear_out = self.compute_out_linear(fmha_out, i)

        return out_linear_out

    def compute_ffn_layernorm(self, out_linear_out, residual_input, i):
        norm_out = self.norm_func(
            out_linear_out,
            norm_weight=self.ffn_ln_scales[i],
            norm_bias=self.ffn_ln_biases[i],
            epsilon=self._epsilon,
            begin_norm_axis=1,
            bias=self.linear_biases[i],
            residual=residual_input,
        )
        tmp_out, residual_input = norm_out[0], norm_out[1]

        return tmp_out, residual_input

    def compute_activation(self, ffn1_out, i):
        return paddle.incubate.nn.functional.fused_bias_act(
            ffn1_out, self.ffn1_biases[i], act_method=self.activation
        )

    def compute_ffn1(self, tmp_out, i):
        return paddle.matmul(tmp_out, self.ffn1_weights[i])

    def compute_ffn2(self, ffn1_out, i):
        return paddle.matmul(ffn1_out, self.ffn2_weights[i])

    def compute_bias_residual_layernorm(
        self, ffn2_out, residual_input, i, num_layers
    ):
        if i != num_layers - 1:
            norm_out = self.norm_func(
                ffn2_out,
                norm_weight=self.ln_scales[i + 1],
                norm_bias=self.ln_biases[i + 1],
                epsilon=self._epsilon,
                begin_norm_axis=1,
                bias=self.ffn2_biases[i],
                residual=residual_input,
            )
            tmp_out, residual_input = norm_out[0], norm_out[1]
        else:
            tmp_out = paddle.incubate.nn.functional.fused_layer_norm(
                ffn2_out,
                norm_weight=None,
                norm_bias=None,
                epsilon=self._epsilon,
                begin_norm_axis=1,
                bias=self.ffn2_biases[i],
                residual=residual_input,
            )[0]
        return tmp_out, residual_input

    def post_process(self, **kwargs):
        time_step = kwargs.get("time_step", None)
        multi_block_output = kwargs.get("multi_block_output", None)
        cum_offsets = kwargs.get("cum_offsets", None)
        seq_lens = kwargs.get("seq_lens", None)
        input_ids = kwargs.get("input_ids", None)

        out = multi_block_output

        return out

    def forward(
        self,
        input_ids,
        src,
        cum_offsets=None,
        padding_offset=None,
        attn_mask=None,
        caches=None,
        pre_caches=None,
        pre_caches_length=0,
        rotary_embs=None,
        rotary_emb_dims=0,
        seq_lens=None,
        time_step=None,
        **kwargs,
    ):
        kwargs["cum_offsets"] = cum_offsets
        if caches is not None:
            assert len(caches) == len(self.qkv_weights) or len(
                caches
            ) == 2 * len(self.qkv_weights)

        assert self.num_layers == len(self.qkv_weights)

        max_enc_len_this_time, max_dec_len_this_time = self.compute_max_len(
            kwargs.get("seq_lens_encoder", None),
            kwargs.get("seq_lens_decoder", None),
            cum_offsets,
        )
        kwargs["max_enc_len_this_time"] = max_enc_len_this_time
        kwargs["max_dec_len_this_time"] = max_dec_len_this_time

        residual_input = src
        for i in range(self.num_layers):
            qkv_out, residual_input = self.compute_qkv(src, residual_input, i)
            out_linear_out = self.compute_attn(
                time_step,
                qkv_out,
                padding_offset,
                seq_lens,
                input_ids,
                rotary_embs,
                rotary_emb_dims,
                caches,
                pre_caches,
                pre_caches_length,
                attn_mask,
                i,
                **kwargs,
            )
            # all_reduce
            if self.nranks > 1:
                dist.all_reduce(out_linear_out)

            # ffn layernorm
            tmp_out, residual_input = self.compute_ffn_layernorm(
                out_linear_out, residual_input, i
            )

            # ffn1 matmul
            ffn1_out = self.compute_ffn1(tmp_out, i)
            ffn1_out = self.compute_activation(ffn1_out, i)

            # ffn2 matmul
            ffn2_out = self.compute_ffn2(ffn1_out, i)

            # all_reduce
            if self.nranks > 1:
                dist.all_reduce(ffn2_out)

            # norm + residual_add_bias
            tmp_out, residual_input = self.compute_bias_residual_layernorm(
                ffn2_out, residual_input, i, self.num_layers
            )
            src = tmp_out

        kwargs["time_step"] = time_step
        kwargs["multi_block_output"] = tmp_out
        kwargs["seq_lens"] = seq_lens
        kwargs["input_ids"] = input_ids
        out = self.post_process(**kwargs)
        return out, caches


class LlamaInferenceModel(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_size = self.hidden_size // self.num_attention_heads
        self.intermediate_size = config.intermediate_size
        self.num_layers = config.num_hidden_layers
        self.epsilon = config.rms_norm_eps
        self.max_position_embeddings = config.max_position_embeddings
        self.quant_type = ""
        self.rope_theta = config.rope_theta
        self.use_neox = True

        if (
            config.tensor_parallel_degree > 1
            and config.vocab_size % config.tensor_parallel_degree == 0
        ):
            self.embed_tokens = fleet.meta_parallel.VocabParallelEmbedding(
                self.vocab_size,
                self.hidden_size,
                weight_attr=paddle.ParamAttr(
                    initializer=nn.initializer.XavierNormal()
                ),
            )
        else:
            self.embed_tokens = nn.Embedding(
                self.vocab_size,
                self.hidden_size,
            )
        ring_id = -1
        try:
            hcg = fleet.get_hybrid_communicate_group()
            model_parallel_group = hcg.get_model_parallel_group()
            ring_id = model_parallel_group.id
        except:
            pass
        linear_shift_attrs = None
        linear_smooth_attrs = None
        ffn2_shift_attrs = None
        ffn2_smooth_attrs = None

        ln_bias_attrs = None
        qkv_bias_attrs = None
        out_proj_bias_attrs = None
        ffn_ln_bias_attrs = None
        ffn1_bias_attrs = None
        ffn2_bias_attrs = None

        ffn1_0_weight_attrs = None
        ffn1_1_weight_attrs = None
        ffn1_0_bias_attrs = None
        ffn1_1_bias_attrs = None

        ffn1_weight_attrs = None
        ffn2_weight_attrs = None

        ln_scale_attrs = [
            paddle.ParamAttr(name=f"fusellama.{i}.ln_scale")
            for i in range(self.num_layers)
        ]
        qkv_weight_attrs = [
            paddle.ParamAttr(
                name=f"fusellama.{i}.qkv_weight",
                initializer=paddle.nn.initializer.Constant(value=0),
            )
            for i in range(self.num_layers)
        ]
        out_proj_weight_attrs = [
            paddle.ParamAttr(
                name=f"fusellama.{i}.out_proj_weight",
                initializer=paddle.nn.initializer.Constant(value=0),
            )
            for i in range(self.num_layers)
        ]
        ffn_ln_scale_attrs = [
            paddle.ParamAttr(name=f"fusellama.{i}.ffn_ln_scale")
            for i in range(self.num_layers)
        ]

        ffn1_weight_attrs = [
            paddle.ParamAttr(
                name=f"fusellama.{i}.ffn1_weight",
                initializer=paddle.nn.initializer.Constant(value=0),
            )
            for i in range(self.num_layers)
        ]
        ffn2_weight_attrs = [
            paddle.ParamAttr(
                name=f"fusellama.{i}.ffn2_weight",
                initializer=paddle.nn.initializer.Constant(value=0),
            )
            for i in range(self.num_layers)
        ]

        qkv_weight_scale_attrs = None
        out_proj_weight_scale_attrs = None
        ffn1_weight_scale_attrs = None
        ffn2_weight_scale_attrs = None

        transformer_config = FusedMultiTransformerConfig(
            embed_dim=self.hidden_size,
            num_heads=self.num_attention_heads,
            kv_num_heads=self.num_key_value_heads,
            dim_feedforward=self.intermediate_size,
            quant_type=self.quant_type,
            activation="swiglu",
            num_layers=config.num_hidden_layers,
            nranks=config.tensor_parallel_degree,
            ring_id=ring_id,
            ln_scale_attrs=ln_scale_attrs,
            qkv_weight_attrs=qkv_weight_attrs,
            qkv_weight_scale_attrs=qkv_weight_scale_attrs,
            linear_weight_attrs=out_proj_weight_attrs,
            linear_weight_scale_attrs=out_proj_weight_scale_attrs,
            ffn_ln_scale_attrs=ffn_ln_scale_attrs,
            ffn1_weight_attrs=ffn1_weight_attrs,
            ffn1_weight_scale_attrs=ffn1_weight_scale_attrs,
            ffn1_0_weight_attrs=ffn1_0_weight_attrs,
            ffn1_1_weight_attrs=ffn1_1_weight_attrs,
            ffn2_weight_attrs=ffn2_weight_attrs,
            ffn2_weight_scale_attrs=ffn2_weight_scale_attrs,
            linear_shift_attrs=linear_shift_attrs,
            linear_smooth_attrs=linear_smooth_attrs,
            ffn2_shift_attrs=ffn2_shift_attrs,
            ffn2_smooth_attrs=ffn2_smooth_attrs,
            ln_bias_attrs=ln_bias_attrs,
            qkv_bias_attrs=qkv_bias_attrs,
            linear_bias_attrs=out_proj_bias_attrs,
            ffn_ln_bias_attrs=ffn_ln_bias_attrs,
            ffn1_bias_attrs=ffn1_bias_attrs,
            ffn1_0_bias_attrs=ffn1_0_bias_attrs,
            ffn1_1_bias_attrs=ffn1_1_bias_attrs,
            ffn2_bias_attrs=ffn2_bias_attrs,
            epsilon=self.epsilon,
            norm_type="rmsnorm",
            use_neox_rotary_style=self.use_neox,
            rank_id=config.tensor_parallel_rank,
            trans_qkvw=(
                False
                if paddle.is_compiled_with_rocm() and "a8w8" in self.quant_type
                else True
            ),
        )
        self.transformer_block = FusedMultiTransformerBase(transformer_config)
        self.transformer_block.init_weight()
        self.norm = FusedLlamaRMSNorm(config)
        self.cache_kvs = None
        self.head_dim_shape_tensor = paddle.ones(
            (self.hidden_size // self.num_attention_heads), dtype="int8"
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        use_cache=None,
        cache_kvs=None,
        pre_caches=None,
        seq_len_encoder=None,
        seq_len_decoder=None,
        past_key_values=None,
        output_attentions=False,
        output_hidden_states=None,
        return_dict=False,
        **kwargs,
    ):
        past_key_values = kwargs.get("cache", None)
        is_decoder = past_key_values is not None

        if inputs_embeds is not None:
            batch, seq_len, hidden_dim = inputs_embeds.shape
            inputs_embeds = inputs_embeds.reshape([batch * seq_len, hidden_dim])

        output_attentions = (
            output_attentions if output_attentions is not None else False
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else False
        )
        use_cache = (
            use_cache if use_cache is not None else self.config.use_cache
        )
        cache_kvs = cache_kvs if cache_kvs is not None else self.cache_kvs
        return_dict = (
            return_dict
            if return_dict is not None
            else self.config.use_return_dict
        )

        if past_key_values is None:
            past_key_values = tuple([None] * self.config.num_hidden_layers)

        ids_remove_padding = input_ids.reshape((-1,))
        padding_offset = ids_remove_padding
        cum_offsets = paddle.ones(paddle.shape(input_ids)[0])

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(ids_remove_padding)

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        seq_lens = seq_len_decoder if is_decoder else seq_len_encoder

        position_offset = 0
        if not is_decoder and pre_caches is not None:
            position_offset = 128

        hidden_states, _ = self.transformer_block(
            input_ids,
            hidden_states,
            cum_offsets=cum_offsets,
            padding_offset=padding_offset,
            attn_mask=attention_mask,
            caches=cache_kvs,
            pre_caches=pre_caches,
            pre_caches_length=position_offset,
            seq_lens=seq_lens,
            rotary_embs=None,
            rotary_emb_dims=1,
            time_step=None,
        )
        hidden_states = self.norm(hidden_states)
        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    None,
                    all_hidden_states,
                    all_self_attns,
                ]
                if v is not None
            )
