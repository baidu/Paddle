# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import logging
import math
from abc import abstractmethod

import paddle
import paddle.nn
import paddle.nn.functional as F

_ALL_PATTERNS = {}
_USED_PATTERNS = []

logger = logging.getLogger(__name__)


def register_pattern(cls):
    """Register a pattern"""

    def register():
        global _ALL_PATTERNS
        pattern = cls()
        _ALL_PATTERNS[pattern.name] = pattern
        logger.debug(
            f'register pattern : {pattern.name}, pattern program: {pattern.program}'
        )

    register()

    return cls


def register_used_patterns(names):
    if isinstance(names, list):
        for name in names:
            _USED_PATTERNS.append(name)
    else:
        _USED_PATTERNS.append(names)


def clear_used_patterns():
    _USED_PATTERNS.clear()


def get_pattern(name):
    return _ALL_PATTERNS[name]


class ShapeConfigs:
    def __init__(self):
        self.batch_size = 4
        self.seq_length = 16
        self.hidden_size = 32
        self.intermediate_size = 64
        self.num_heads = 4
        self.head_dim = 8
        self.variance_epsilon = 1e-6
        self.input_shape = [self.batch_size, self.seq_length]
        self.input_embedding_shape = [
            self.batch_size,
            self.seq_length,
            self.hidden_size,
        ]
        self.hidden_states_shape = [
            self.batch_size,
            self.seq_length,
            self.num_heads,
            self.head_dim,
        ]
        self.hidden_shape = [self.hidden_size]
        self.hidden_hidden_weight_shape = [self.hidden_size, self.hidden_size]
        self.hidden_intermediate_weight_shape = [
            self.hidden_size,
            self.intermediate_size,
        ]
        self.intermediate_hidden_weight_shape = [
            self.intermediate_size,
            self.hidden_size,
        ]
        self.cos_shape = [1, self.seq_length, 1, self.head_dim]
        self.sin_shape = [1, self.seq_length, 1, self.head_dim]
        self.attention_mask_shape = [
            self.batch_size,
            1,
            self.seq_length,
            self.seq_length,
        ]
        self.target_q_shape = [0, 0, self.num_heads, self.head_dim]
        self.target_kv_shape = [0, 0, self.num_heads, self.head_dim]


class BasePattern:
    """
    Base class of pattern.
    """

    _name = "base"

    def __init__(self):
        super().__init__()
        self.shape_configs = ShapeConfigs()
        self.build()

    @property
    def name(self):
        return self.__class__._name

    @abstractmethod
    def build(self):
        pass


class DistInfos:
    def __init__(self):
        self.dist_infos = {}  # containing weight and bias
        self.dist_infos[1] = [
            [paddle.distributed.Replicate()],
            [paddle.distributed.Replicate()],
        ]
        self.dist_infos[2] = [
            [paddle.distributed.Replicate(), paddle.distributed.Replicate()],
            [paddle.distributed.Replicate(), paddle.distributed.Replicate()],
        ]

    def get_dist_info(self, mesh_num_dims=1):
        assert mesh_num_dims in [
            1,
            2,
        ], "mesh dims must be 1 or 2, for [dp], [mp] or [dp, mp]"
        return self.dist_infos[mesh_num_dims]


class MpDistInfos(DistInfos):
    def __init__(self, mp_type=None):
        super().__init__()
        if mp_type == "column":
            self.dist_infos[1] = [
                [paddle.distributed.Shard(1)],
                [paddle.distributed.Shard(0)],
            ]
            self.dist_infos[2] = [
                [paddle.distributed.Replicate(), paddle.distributed.Shard(1)],
                [paddle.distributed.Replicate(), paddle.distributed.Shard(0)],
            ]
        elif mp_type == "row":
            self.dist_infos[1] = [
                [paddle.distributed.Shard(0)],
                [paddle.distributed.Replicate()],
            ]
            self.dist_infos[2] = [
                [paddle.distributed.Replicate(), paddle.distributed.Shard(0)],
                [
                    paddle.distributed.Replicate(),
                    paddle.distributed.Replicate(),
                ],
            ]


@register_pattern
class EmbeddingPattern(BasePattern):
    """Embedding pattern"""

    name = "embedding"

    def __init__(self):
        super().__init__()

    def build(self):
        # # # build program # # #
        paddle.enable_static()
        # program init
        start_program, main_program = (
            paddle.static.Program(),
            paddle.static.Program(),
        )
        # data init
        x = paddle.randn(self.shape_configs.input_shape)
        # program construction
        with paddle.static.program_guard(main_program, start_program):
            x = paddle.static.data('x', self.shape_configs.input_shape, x.dtype)
            weight = paddle.create_parameter(
                shape=self.shape_configs.hidden_intermediate_weight_shape,
                dtype=paddle.get_default_dtype(),
                default_initializer=paddle.nn.initializer.Constant(1.0),
            )
            self.apply(x, weight)

        self.program = main_program
        self.ops_dist_infos = None
        paddle.disable_static()

    @staticmethod
    def apply(x, weight):
        return paddle.nn.functional.embedding(x, weight)


@register_pattern
class RMSNormPattern(BasePattern):
    """RMSNorm pattern"""

    name = "rmsnorm"

    def __init__(self):
        super().__init__()

    def build(self):
        # # # build program # # #
        paddle.enable_static()
        # program init
        start_program, main_program = (
            paddle.static.Program(),
            paddle.static.Program(),
        )
        # data init
        x = paddle.randn(self.shape_configs.input_embedding_shape)
        # program construction
        with paddle.static.program_guard(main_program, start_program):
            x = paddle.static.data(
                'x', self.shape_configs.input_embedding_shape, x.dtype
            )
            weight = paddle.create_parameter(
                shape=self.shape_configs.hidden_shape,
                dtype=paddle.get_default_dtype(),
                default_initializer=paddle.nn.initializer.Constant(1.0),
            )
            self.apply(x, weight, self.shape_configs)

        self.program = main_program
        self.ops_dist_infos = None
        paddle.disable_static()

    @staticmethod
    def apply(x, norm_weight, shape_configs):
        with paddle.amp.auto_cast(False):
            x = x.astype("float32")
            variance = x.pow(2).mean(-1, keepdim=True)
            x = paddle.rsqrt(variance + shape_configs.variance_epsilon) * x
            out = x * norm_weight
            return out


@register_pattern
class RotateHalfPattern(BasePattern):
    """Rotate Half pattern"""

    name = "rotate_half"

    def __init__(self):
        super().__init__()

    def build(self):
        # # # build program # # #
        paddle.enable_static()
        # program init
        start_program, main_program = (
            paddle.static.Program(),
            paddle.static.Program(),
        )
        # data init
        tmp_x = paddle.randn(self.shape_configs.hidden_states_shape)
        tmp = paddle.static.data(
            'tmp_x', self.shape_configs.hidden_states_shape, tmp_x.dtype
        )
        # program construction
        with paddle.static.program_guard(main_program, start_program):
            x = paddle.reshape(tmp, self.shape_configs.hidden_states_shape)
            self.apply(x)

        self.program = main_program
        self.ops_dist_infos = None
        paddle.disable_static()

    @staticmethod
    def apply(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return paddle.concat([-x2, x1], axis=-1)  # shape is the same as x


@register_pattern
class ApplyRotaryPosEmbPattern(BasePattern):
    """ApplyRotaryPosEmb pattern"""

    name = "apply_rotary_pos_emb"

    def __init__(self):
        super().__init__()

    def build(self):
        # # # build program # # #
        paddle.enable_static()
        # program init
        start_program, main_program = (
            paddle.static.Program(),
            paddle.static.Program(),
        )
        # data init
        q = paddle.randn(self.shape_configs.hidden_states_shape)
        k = paddle.randn(self.shape_configs.hidden_states_shape)
        cos = paddle.randn(self.shape_configs.cos_shape)
        sin = paddle.randn(self.shape_configs.sin_shape)
        position_ids = paddle.randint(
            low=1, shape=self.shape_configs.input_shape
        )
        # program construction
        with paddle.static.program_guard(main_program, start_program):
            q = paddle.static.data(
                'q', self.shape_configs.hidden_states_shape, q.dtype
            )
            k = paddle.static.data(
                'k', self.shape_configs.hidden_states_shape, k.dtype
            )
            cos = paddle.static.data(
                'cos', self.shape_configs.cos_shape, cos.dtype
            )
            sin = paddle.static.data(
                'sin', self.shape_configs.sin_shape, sin.dtype
            )
            position_ids = paddle.static.data(
                'position_ids',
                self.shape_configs.input_shape,
                position_ids.dtype,
            )

            self.apply(q, k, cos, sin, position_ids)

        self.program = main_program
        self.ops_dist_infos = None
        paddle.disable_static()

    @staticmethod
    def apply(q, k, cos, sin, position_ids):
        if position_ids is None:
            cos = cos[:, : q.shape[1], :, :]  # [bs, seq_len, 1, dim]
            sin = sin[:, : q.shape[1], :, :]  # [bs, seq_len, 1, dim]
        else:
            cos = cos.squeeze(axis=[0, 2])  # [seq_len, dim]
            sin = sin.squeeze(axis=[0, 2])  # [seq_len, dim]
            cos = cos[position_ids].unsqueeze(2)  # [bs, seq_len, 1, dim]
            sin = sin[position_ids].unsqueeze(2)  # [bs, seq_len, 1, dim]
        q_embed = (q * cos) + (RotateHalfPattern.apply(q) * sin)
        k_embed = (k * cos) + (RotateHalfPattern.apply(k) * sin)
        return q_embed, k_embed


@register_pattern
class QKVReshapePattern(BasePattern):
    """QKV(not fused) and reshape pattern"""

    name = "qkv_reshape"

    def __init__(self):
        super().__init__()

    def build(self):
        # # # build program # # #
        paddle.enable_static()
        # program init
        start_program, main_program = (
            paddle.static.Program(),
            paddle.static.Program(),
        )
        # data init
        hidden_states = paddle.randn(self.shape_configs.input_embedding_shape)
        # program construction
        with paddle.static.program_guard(main_program, start_program):
            hidden_states = paddle.static.data(
                'hidden_states',
                self.shape_configs.input_embedding_shape,
                hidden_states.dtype,
            )
            q_weight = paddle.create_parameter(
                self.shape_configs.hidden_hidden_weight_shape, "float32"
            )
            k_weight = paddle.create_parameter(
                self.shape_configs.hidden_hidden_weight_shape, "float32"
            )
            v_weight = paddle.create_parameter(
                self.shape_configs.hidden_hidden_weight_shape, "float32"
            )

            self.apply(
                q_weight, k_weight, v_weight, hidden_states, self.shape_configs
            )

        self.program = main_program
        self.ops_dist_infos = None
        paddle.disable_static()

    @staticmethod
    def apply(q_weight, k_weight, v_weight, hidden_states, shape_configs):
        q = paddle.matmul(hidden_states, q_weight)
        k = paddle.matmul(hidden_states, k_weight)
        v = paddle.matmul(hidden_states, v_weight)
        q = q.reshape(shape=shape_configs.target_q_shape)
        k = k.reshape(shape=shape_configs.target_kv_shape)
        v = v.reshape(shape=shape_configs.target_kv_shape)
        return q, k, v


@register_pattern
class QKVRopePattern(BasePattern):
    """QKV(not fused) and Rope pattern"""

    name = "qkv_rope"

    def __init__(self):
        super().__init__()

    def build(self):
        # # # build program # # #
        paddle.enable_static()
        # program init
        start_program, main_program = (
            paddle.static.Program(),
            paddle.static.Program(),
        )
        # data init
        hidden_states = paddle.randn(self.shape_configs.input_embedding_shape)
        cos_cached = paddle.randn(self.shape_configs.cos_shape)
        sin_cached = paddle.randn(self.shape_configs.sin_shape)
        position_ids = paddle.randint(
            low=1, shape=self.shape_configs.input_shape
        )
        # program construction
        with paddle.static.program_guard(main_program, start_program):
            hidden_states = paddle.static.data(
                'hidden_states',
                self.shape_configs.input_embedding_shape,
                hidden_states.dtype,
            )
            q_weight = paddle.create_parameter(
                self.shape_configs.hidden_hidden_weight_shape, "float32"
            )
            k_weight = paddle.create_parameter(
                self.shape_configs.hidden_hidden_weight_shape, "float32"
            )
            v_weight = paddle.create_parameter(
                self.shape_configs.hidden_hidden_weight_shape, "float32"
            )
            cos_cached = paddle.static.data(
                'cos_cached', self.shape_configs.cos_shape, cos_cached.dtype
            )
            sin_cached = paddle.static.data(
                'sin_cached', self.shape_configs.sin_shape, cos_cached.dtype
            )
            position_ids = paddle.static.data(
                'position_ids',
                self.shape_configs.input_shape,
                position_ids.dtype,
            )

            self.apply(
                hidden_states,
                q_weight,
                k_weight,
                v_weight,
                cos_cached,
                sin_cached,
                position_ids,
                self.shape_configs,
            )

        self.program = main_program
        self.ops_dist_infos = None
        paddle.disable_static()

    @staticmethod
    def apply(
        hidden_states,
        q_weight,
        k_weight,
        v_weight,
        cos_cached,
        sin_cached,
        position_ids,
        shape_configs,
    ):
        q, k, v = QKVReshapePattern.apply(
            q_weight, k_weight, v_weight, hidden_states, shape_configs
        )
        # rope_emb
        seq_length = shape_configs.cos_shape[1]
        cos = cos_cached[:, :seq_length, :, :]
        sin = sin_cached[:, :seq_length, :, :]
        q_embed, k_embed = ApplyRotaryPosEmbPattern.apply(
            q, k, cos, sin, position_ids
        )
        return q_embed, k_embed, v


@register_pattern
class ScaleDotProductPattern(BasePattern):
    """Scale dot product attention pattern"""

    name = "scale_dot_product_attention"

    def __init__(self):
        super().__init__()

    def build(self):
        # # # build program # # #
        paddle.enable_static()
        # program init
        start_program, main_program = (
            paddle.static.Program(),
            paddle.static.Program(),
        )
        # data init
        q = paddle.randn(self.shape_configs.hidden_states_shape)
        k = paddle.randn(self.shape_configs.hidden_states_shape)
        v = paddle.randn(self.shape_configs.hidden_states_shape)
        attention_mask = paddle.randn(self.shape_configs.attention_mask_shape)

        # program construction
        with paddle.static.program_guard(main_program, start_program):
            q = paddle.static.data(
                'q', self.shape_configs.hidden_states_shape, q.dtype
            )
            k = paddle.static.data(
                'k', self.shape_configs.hidden_states_shape, k.dtype
            )
            v = paddle.static.data(
                'v', self.shape_configs.hidden_states_shape, v.dtype
            )
            attention_mask = paddle.static.data(
                'attention_mask',
                self.shape_configs.attention_mask_shape,
                attention_mask.dtype,
            )
            outputs = self.apply(q, k, v, attention_mask)

        self.program = main_program
        self.ops_dist_infos = None
        paddle.disable_static()

    @staticmethod
    def apply(
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
            query_states / math.sqrt(head_dim),
            key_states.transpose([0, 1, 3, 2]),
        )

        attention_mask = attention_mask.reshape([bsz, 1, q_len, kv_seq_len])

        attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, axis=-1, dtype="float32").astype(
            query_states.dtype
        )

        attn_output = paddle.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose([0, 2, 1, 3])

        attn_output = attn_output.reshape([bsz, q_len, head_dim * num_heads])

        return attn_output


@register_pattern
class AttentionPattern(BasePattern):
    """Attention pattern"""

    name = "attention"

    def __init__(self):
        super().__init__()

    def build(self):
        # # # build program # # #
        paddle.enable_static()
        # program init
        start_program, main_program = (
            paddle.static.Program(),
            paddle.static.Program(),
        )
        # data init
        hidden_states = paddle.randn(self.shape_configs.input_embedding_shape)
        cos_cached = paddle.randn(self.shape_configs.cos_shape)
        sin_cached = paddle.randn(self.shape_configs.sin_shape)
        position_ids = paddle.randint(
            low=1, shape=self.shape_configs.input_shape
        )
        attention_mask = paddle.randn(self.shape_configs.attention_mask_shape)
        # program construction
        with paddle.static.program_guard(main_program, start_program):
            hidden_states = paddle.static.data(
                'hidden_states',
                self.shape_configs.input_embedding_shape,
                hidden_states.dtype,
            )
            q_weight = paddle.create_parameter(
                self.shape_configs.hidden_hidden_weight_shape, "float32"
            )
            k_weight = paddle.create_parameter(
                self.shape_configs.hidden_hidden_weight_shape, "float32"
            )
            v_weight = paddle.create_parameter(
                self.shape_configs.hidden_hidden_weight_shape, "float32"
            )
            out_weight = paddle.create_parameter(
                self.shape_configs.hidden_hidden_weight_shape, "float32"
            )
            cos_cached = paddle.static.data(
                'cos_cached', self.shape_configs.cos_shape, cos_cached.dtype
            )
            sin_cached = paddle.static.data(
                'sin_cached', self.shape_configs.sin_shape, cos_cached.dtype
            )
            position_ids = paddle.static.data(
                'position_ids',
                self.shape_configs.input_shape,
                position_ids.dtype,
            )
            attention_mask = paddle.static.data(
                'attention_mask',
                self.shape_configs.attention_mask_shape,
                attention_mask.dtype,
            )

            self.apply(
                hidden_states,
                q_weight,
                k_weight,
                v_weight,
                out_weight,
                cos_cached,
                sin_cached,
                position_ids,
                attention_mask,
                self.shape_configs,
            )

        self.program = main_program

        qkv_linear_dist_infos = MpDistInfos("column")
        out_linear_dist_infos = MpDistInfos("row")
        ops_dist_infos = {
            (9,): qkv_linear_dist_infos,
            (10,): qkv_linear_dist_infos,
            (11,): qkv_linear_dist_infos,
            (76,): out_linear_dist_infos,
        }
        self.ops_dist_infos = ops_dist_infos

        paddle.disable_static()

    @staticmethod
    def apply(
        hidden_states,
        q_weight,
        k_weight,
        v_weight,
        out_weight,
        cos_cached,
        sin_cached,
        position_ids,
        attention_mask,
        shape_configs,
    ):
        q_embed, k_embed, v = QKVRopePattern.apply(
            hidden_states,
            q_weight,
            k_weight,
            v_weight,
            cos_cached,
            sin_cached,
            position_ids,
            shape_configs,
        )
        tmp = ScaleDotProductPattern.apply(q_embed, k_embed, v, attention_mask)
        output = paddle.matmul(tmp, out_weight)
        return output


@register_pattern
class MLP3Pattern(BasePattern):
    """MLP pattern"""

    name = "mlp_3_with_swiglu"

    def __init__(self):
        super().__init__()

    def build(self):
        # # # build program # # #
        paddle.enable_static()
        # program init
        start_program, main_program = (
            paddle.static.Program(),
            paddle.static.Program(),
        )
        # data init
        hidden_states = paddle.randn(self.shape_configs.input_embedding_shape)
        # program construction
        with paddle.static.program_guard(main_program, start_program):
            hidden_states = paddle.static.data(
                'hidden_states',
                self.shape_configs.input_embedding_shape,
                hidden_states.dtype,
            )
            gate_weight = paddle.create_parameter(
                self.shape_configs.hidden_intermediate_weight_shape, "float32"
            )
            up_weight = paddle.create_parameter(
                self.shape_configs.hidden_intermediate_weight_shape, "float32"
            )
            down_weight = paddle.create_parameter(
                self.shape_configs.intermediate_hidden_weight_shape, "float32"
            )

            self.apply(hidden_states, gate_weight, up_weight, down_weight)

        self.program = main_program

        up_linear_dist_infos = MpDistInfos("column")
        down_linear_dist_infos = MpDistInfos("row")
        # # # build ops dist infos # # #
        ops_dist_infos = {
            (4,): up_linear_dist_infos,
            (5,): up_linear_dist_infos,
            (7,): down_linear_dist_infos,
        }
        self.ops_dist_infos = ops_dist_infos

        paddle.disable_static()

    @staticmethod
    def apply(hidden_states, gate_weight, up_weight, down_weight):
        gate = paddle.matmul(hidden_states, gate_weight)
        up = paddle.matmul(hidden_states, up_weight)
        tmp = paddle.incubate.nn.functional.swiglu(gate, up)
        out = paddle.matmul(tmp, down_weight)
        return out


@register_pattern
class DecoderLayerPattern(BasePattern):
    """Decoder layer pattern"""

    name = "decoder_layer"

    def __init__(self):
        super().__init__()

    def build(self):
        # # # build program # # #
        paddle.enable_static()
        # program init
        start_program, main_program = (
            paddle.static.Program(),
            paddle.static.Program(),
        )
        # data init
        hidden_states = paddle.randn(self.shape_configs.input_embedding_shape)
        cos_cached = paddle.randn(self.shape_configs.cos_shape)
        sin_cached = paddle.randn(self.shape_configs.sin_shape)
        position_ids = paddle.randint(
            low=1, shape=self.shape_configs.input_shape
        )
        attention_mask = paddle.randn(self.shape_configs.attention_mask_shape)
        # program construction
        with paddle.static.program_guard(main_program, start_program):
            hidden_states = paddle.static.data(
                'hidden_states',
                self.shape_configs.input_embedding_shape,
                hidden_states.dtype,
            )
            q_weight = paddle.create_parameter(
                self.shape_configs.hidden_hidden_weight_shape, "float32"
            )
            k_weight = paddle.create_parameter(
                self.shape_configs.hidden_hidden_weight_shape, "float32"
            )
            v_weight = paddle.create_parameter(
                self.shape_configs.hidden_hidden_weight_shape, "float32"
            )
            out_weight = paddle.create_parameter(
                self.shape_configs.hidden_hidden_weight_shape, "float32"
            )
            input_rms_norm_weight = paddle.create_parameter(
                self.shape_configs.hidden_shape, "float32"
            )
            post_attention_rms_norm_weight = paddle.create_parameter(
                self.shape_configs.hidden_shape, "float32"
            )
            gate_weight = paddle.create_parameter(
                self.shape_configs.hidden_intermediate_weight_shape, "float32"
            )
            up_weight = paddle.create_parameter(
                self.shape_configs.hidden_intermediate_weight_shape, "float32"
            )
            down_weight = paddle.create_parameter(
                self.shape_configs.intermediate_hidden_weight_shape, "float32"
            )
            cos_cached = paddle.static.data(
                'cos_cached', self.shape_configs.cos_shape, cos_cached.dtype
            )
            sin_cached = paddle.static.data(
                'sin_cached', self.shape_configs.sin_shape, cos_cached.dtype
            )
            position_ids = paddle.static.data(
                'position_ids',
                self.shape_configs.input_shape,
                position_ids.dtype,
            )
            attention_mask = paddle.static.data(
                'attention_mask',
                self.shape_configs.attention_mask_shape,
                attention_mask.dtype,
            )

            self.apply(
                hidden_states,
                q_weight,
                k_weight,
                v_weight,
                out_weight,
                gate_weight,
                up_weight,
                down_weight,
                input_rms_norm_weight,
                post_attention_rms_norm_weight,
                cos_cached,
                sin_cached,
                position_ids,
                attention_mask,
                self.shape_configs,
            )

        self.program = main_program

        qkv_linear_dist_infos = MpDistInfos("column")
        out_linear_dist_infos = MpDistInfos("row")
        up_linear_dist_infos = MpDistInfos("column")
        down_linear_dist_infos = MpDistInfos("row")
        # # # build ops dist infos # # #
        ops_dist_infos = {
            (22,): qkv_linear_dist_infos,
            (23,): qkv_linear_dist_infos,
            (24,): qkv_linear_dist_infos,
            (89,): out_linear_dist_infos,
            (99,): up_linear_dist_infos,
            (100,): up_linear_dist_infos,
            (102,): down_linear_dist_infos,
        }
        self.ops_dist_infos = ops_dist_infos

        paddle.disable_static()

    @staticmethod
    def apply(
        hidden_states,
        q_weight,
        k_weight,
        v_weight,
        out_weight,
        gate_weight,
        up_weight,
        down_weight,
        input_rms_norm_weight,
        post_attention_rms_norm_weight,
        cos_cached,
        sin_cached,
        position_ids,
        attention_mask,
        shape_configs,
    ):
        residual = hidden_states
        hidden_states = RMSNormPattern.apply(
            hidden_states, input_rms_norm_weight, shape_configs
        )
        output = AttentionPattern.apply(
            hidden_states,
            q_weight,
            k_weight,
            v_weight,
            out_weight,
            cos_cached,
            sin_cached,
            position_ids,
            attention_mask,
            shape_configs,
        )
        hidden_states = residual + output
        residual = hidden_states
        hidden_states = RMSNormPattern.apply(
            hidden_states, post_attention_rms_norm_weight, shape_configs
        )
        hidden_states = MLP3Pattern.apply(
            hidden_states, gate_weight, up_weight, down_weight
        )
        hidden_states = residual + hidden_states
        return hidden_states


@register_pattern
class QKVReshapePattern_2(BasePattern):
    """QKV(not fused) and reshape pattern"""

    name = "qkv_reshape_2"

    def __init__(self):
        super().__init__()

    def build(self):
        # # # build program # # #
        paddle.enable_static()
        # program init
        start_program, main_program = (
            paddle.static.Program(),
            paddle.static.Program(),
        )
        # data init
        hidden_states = paddle.randn(self.shape_configs.input_embedding_shape)
        # program construction
        with paddle.static.program_guard(main_program, start_program):
            hidden_states = paddle.static.data(
                'hidden_states',
                self.shape_configs.input_embedding_shape,
                hidden_states.dtype,
            )
            q_weight = paddle.create_parameter(
                self.shape_configs.hidden_hidden_weight_shape, "float32"
            )
            k_weight = paddle.create_parameter(
                self.shape_configs.hidden_hidden_weight_shape, "float32"
            )
            v_weight = paddle.create_parameter(
                self.shape_configs.hidden_hidden_weight_shape, "float32"
            )
            q_bias = paddle.create_parameter(
                self.shape_configs.hidden_shape, "float32"
            )
            k_bias = paddle.create_parameter(
                self.shape_configs.hidden_shape, "float32"
            )
            v_bias = paddle.create_parameter(
                self.shape_configs.hidden_shape, "float32"
            )

            self.apply(
                hidden_states,
                q_weight,
                k_weight,
                v_weight,
                q_bias,
                k_bias,
                v_bias,
                self.shape_configs,
            )

        self.program = main_program
        self.ops_dist_infos = None
        paddle.disable_static()

    @staticmethod
    def apply(
        hidden_states,
        q_weight,
        k_weight,
        v_weight,
        q_bias,
        k_bias,
        v_bias,
        shape_configs,
    ):
        q_tmp = paddle.matmul(hidden_states, q_weight)
        q = paddle.add(q_tmp, q_bias)
        k_tmp = paddle.matmul(hidden_states, k_weight)
        k = paddle.add(k_tmp, k_bias)
        v_tmp = paddle.matmul(hidden_states, v_weight)
        v = paddle.add(v_tmp, v_bias)
        q = q.reshape(shape=shape_configs.target_q_shape)
        k = k.reshape(shape=shape_configs.target_kv_shape)
        v = v.reshape(shape=shape_configs.target_kv_shape)
        return q, k, v


@register_pattern
class CoreAttnPattern(BasePattern):
    """Core attention pattern"""

    name = "core_attn"

    def __init__(self):
        super().__init__()

    def build(self):
        # # # build program # # #
        paddle.enable_static()
        # program init
        start_program, main_program = (
            paddle.static.Program(),
            paddle.static.Program(),
        )
        # data init
        q = paddle.randn(self.shape_configs.hidden_states_shape)
        k = paddle.randn(self.shape_configs.hidden_states_shape)
        v = paddle.randn(self.shape_configs.hidden_states_shape)
        attention_mask = paddle.randn(self.shape_configs.attention_mask_shape)
        # program construction
        with paddle.static.program_guard(main_program, start_program):
            q = paddle.static.data(
                'q', self.shape_configs.hidden_states_shape, q.dtype
            )
            k = paddle.static.data(
                'k', self.shape_configs.hidden_states_shape, k.dtype
            )
            v = paddle.static.data(
                'v', self.shape_configs.hidden_states_shape, v.dtype
            )
            attention_mask = paddle.static.data(
                'attention_mask',
                self.shape_configs.attention_mask_shape,
                attention_mask.dtype,
            )
            outputs = self.apply(q, k, v, attention_mask)

        self.program = main_program
        self.ops_dist_infos = None
        paddle.disable_static()

    @staticmethod
    def apply(
        q,
        k,
        v,
        attention_mask,
    ):
        bsz, q_len, num_heads, head_dim = q.shape
        perm = [0, 2, 1, 3]
        q = paddle.transpose(x=q, perm=perm)
        k = paddle.transpose(x=k, perm=perm)
        v = paddle.transpose(x=v, perm=perm)
        # scale dot product attention
        product = paddle.matmul(x=q * (head_dim**-0.5), y=k, transpose_y=True)

        if attention_mask is not None:
            product = product + attention_mask.astype(product.dtype)
            weights = F.softmax(product)
        else:
            weights = paddle.incubate.softmax_mask_fuse_upper_triangle(product)
        weights = F.dropout(
            weights, 0.5, training=True, mode="upscale_in_train"
        )

        out = paddle.matmul(weights, v)

        # combine heads
        out = paddle.transpose(
            out, perm=[0, 2, 1, 3]
        )  # bs, seq_len, num_head, head_dim
        out = paddle.reshape(x=out, shape=[0, 0, -1])  # bs, seq_len, dim

        return out


@register_pattern
class Attentio2nPattern(BasePattern):
    """Attention2 pattern"""

    name = "attention2"

    def __init__(self):
        super().__init__()

    def build(self):
        # # # build program # # #
        paddle.enable_static()
        # program init
        start_program, main_program = (
            paddle.static.Program(),
            paddle.static.Program(),
        )
        # data init
        hidden_states = paddle.randn(self.shape_configs.input_embedding_shape)
        attention_mask = paddle.randn(self.shape_configs.attention_mask_shape)
        # program construction
        with paddle.static.program_guard(main_program, start_program):
            hidden_states = paddle.static.data(
                'hidden_states',
                self.shape_configs.input_embedding_shape,
                hidden_states.dtype,
            )
            q_weight = paddle.create_parameter(
                self.shape_configs.hidden_hidden_weight_shape, "float32"
            )
            k_weight = paddle.create_parameter(
                self.shape_configs.hidden_hidden_weight_shape, "float32"
            )
            v_weight = paddle.create_parameter(
                self.shape_configs.hidden_hidden_weight_shape, "float32"
            )
            out_weight = paddle.create_parameter(
                self.shape_configs.hidden_hidden_weight_shape, "float32"
            )
            q_bias = paddle.create_parameter(
                self.shape_configs.hidden_shape, "float32"
            )
            k_bias = paddle.create_parameter(
                self.shape_configs.hidden_shape, "float32"
            )
            v_bias = paddle.create_parameter(
                self.shape_configs.hidden_shape, "float32"
            )
            out_bias = paddle.create_parameter(
                self.shape_configs.hidden_shape, "float32"
            )
            attention_mask = paddle.static.data(
                'attention_mask',
                self.shape_configs.attention_mask_shape,
                attention_mask.dtype,
            )

            self.apply(
                hidden_states,
                q_weight,
                k_weight,
                v_weight,
                out_weight,
                q_bias,
                k_bias,
                v_bias,
                out_bias,
                attention_mask,
                self.shape_configs,
            )

        self.program = main_program

        qkv_linear_dist_infos = MpDistInfos("column")
        out_linear_dist_infos = MpDistInfos("row")
        # # # build ops dist infos # # #
        ops_dist_infos = {
            (10, 11): qkv_linear_dist_infos,
            (12, 13): qkv_linear_dist_infos,
            (14, 15): qkv_linear_dist_infos,
            (37, 38): out_linear_dist_infos,
        }
        self.ops_dist_infos = ops_dist_infos

        paddle.disable_static()

    @staticmethod
    def apply(
        hidden_states,
        q_weight,
        k_weight,
        v_weight,
        out_weight,
        q_bias,
        k_bias,
        v_bias,
        out_bias,
        attention_mask,
        shape_configs,
    ):
        q, k, v = QKVReshapePattern_2.apply(
            hidden_states,
            q_weight,
            k_weight,
            v_weight,
            q_bias,
            k_bias,
            v_bias,
            shape_configs,
        )
        tmp = CoreAttnPattern.apply(q, k, v, attention_mask)
        out_tmp = paddle.matmul(tmp, out_weight)
        output = paddle.add(out_tmp, out_bias)
        return output


@register_pattern
class MLP2Pattern(BasePattern):
    """FFN pattern"""

    name = "mlp_2_with_gelu"

    def __init__(self):
        super().__init__()

    def build(self):
        # # # build program # # #
        paddle.enable_static()
        # program init
        start_program, main_program = (
            paddle.static.Program(),
            paddle.static.Program(),
        )
        # data init
        x = paddle.randn(self.shape_configs.input_embedding_shape)
        # program construction
        with paddle.static.program_guard(main_program, start_program):
            x = paddle.static.data(
                'x', self.shape_configs.input_embedding_shape, x.dtype
            )
            w1 = paddle.create_parameter(
                self.shape_configs.hidden_hidden_weight_shape, "float32"
            )
            b1 = paddle.create_parameter(
                self.shape_configs.hidden_shape, "float32"
            )
            w2 = paddle.create_parameter(
                self.shape_configs.hidden_hidden_weight_shape, "float32"
            )
            b2 = paddle.create_parameter(
                self.shape_configs.hidden_shape, "float32"
            )

            self.apply(x, w1, b1, w2, b2)

        self.program = main_program

        linear_1_dist_infos = MpDistInfos("column")
        linear_2_dist_infos = MpDistInfos("row")
        # # # build ops dist infos # # #
        ops_dist_infos = {
            (5, 6): linear_1_dist_infos,
            (8, 9): linear_2_dist_infos,
        }
        self.ops_dist_infos = ops_dist_infos

        paddle.disable_static()

    @staticmethod
    def apply(x, w1, b1, w2, b2):
        tmp_1 = paddle.matmul(x, w1)
        tmp_2 = paddle.add(tmp_1, b1)
        tmp_3 = paddle.nn.functional.gelu(tmp_2)
        tmp_4 = paddle.matmul(tmp_3, w2)
        out = paddle.add(tmp_4, b2)
        return out


def match_pattern(pattern, program):

    def _compare_op_node(src, tgt):
        """Compare whether two op nodes are equivalent."""
        if src.name() != tgt.name():
            return False

        return True

    def _match_core(src, tgt, is_op):
        nonlocal not_matched
        # not support one input name or output name corresponding to multiple vars
        if not_matched:
            return

        if is_op:
            logger.debug(
                f'comparing src op {src.name()} with tgt op {tgt.name()}'
            )
            # skip comparing data_op
            if src.name() == "pd_op.data" or src.name() == "builtin.parameter":
                return
            # compare op
            if not _compare_op_node(src, tgt):
                not_matched = True
                return

            src_id = src.get_parent_block().ops.index(src)
            tgt_id = tgt.get_parent_block().ops.index(tgt)
            result[src_id] = tgt_id

            # compare input operands num
            if src.num_operands() != tgt.num_operands():
                not_matched = True
                return
            # compare output results num
            if src.num_results() != tgt.num_results():
                not_matched = True
                return

            # compare input operands
            src_operands = src.operands_source()
            for idx, src_operand in enumerate(src_operands):
                tgt_operand = tgt.operand_source(idx)
                _match_core(src_operand, tgt_operand, is_op=False)

            # compare output results
            src_results = src.results()
            for idx, src_result in enumerate(src_results):
                tgt_result = tgt.result(idx)
                _match_core(src_result, tgt_result, is_op=False)

        else:
            logger.debug('comparing operands')
            # as input for op node
            src_as_input_ops = src.all_used_ops()
            tgt_as_input_ops = tgt.all_used_ops()
            # todo: process src_as_input_ops < tgt_as_input_ops
            if len(src_as_input_ops) > len(tgt_as_input_ops):
                not_matched = True
                return
            if len(src_as_input_ops) == len(tgt_as_input_ops):
                for idx, src_as_input_op in enumerate(src_as_input_ops):
                    src_as_input_op_id = (
                        src_as_input_op.get_parent_block().ops.index(
                            src_as_input_op
                        )
                    )
                    if src_as_input_op_id in result.keys():
                        continue

                    tgt_as_input_op = tgt_as_input_ops[idx]
                    _match_core(src_as_input_op, tgt_as_input_op, is_op=True)

            # as output for op node
            src_as_output_op = src.get_defining_op()
            tgt_as_output_op = tgt.get_defining_op()
            if src_as_output_op is not None and tgt_as_output_op is not None:
                src_as_output_op_id = (
                    src_as_output_op.get_parent_block().ops.index(
                        src_as_output_op
                    )
                )
                if src_as_output_op_id not in result.keys():
                    _match_core(src_as_output_op, tgt_as_output_op, is_op=True)

    results = []
    result = {}
    matched_ids = set()
    matched_op_node_ids = set()

    # starts with a op node
    src_ops = pattern.program.global_block().ops
    for op in src_ops:
        if op.name() != "pd_op.data" and op.name() != "builtin.parameter":
            src_start_op = op
            break
    # src_start_op = src_ops[0] # to be done, need to check pattern start op
    assert src_start_op is not None, "src_start_op is none"
    logger.debug(
        f'in match_pattern func, Matching Pattern {pattern.name}, start op is {src_start_op.name()}'
    )

    tgt_ops = program.global_block().ops
    for idx, tgt_op in enumerate(tgt_ops):
        if tgt_op.name() == src_start_op.name():
            tgt_op_id = tgt_op.get_parent_block().ops.index(tgt_op)
            if tgt_op_id in matched_op_node_ids:
                continue
            logger.debug(
                f'in match_pattern func, Matching Pattern {pattern.name}, tgt_op is {tgt_op.name()}'
            )
            not_matched = False
            result = {}
            _match_core(src_start_op, tgt_op, is_op=True)
            if not not_matched:
                logger.debug(
                    f'in match_pattern func, Matched Pattern {pattern.name}, pattern.program is {pattern.program} result is {result}'
                )
                need_to_append = True
                for value in result.values():
                    if value in matched_op_node_ids:
                        result = {}
                        need_to_append = False
                        break
                if need_to_append:
                    results.append(result)
                    for value in result.values():
                        matched_ids.add(value)
                        matched_op_node_ids.add(value)
                    result = {}
            else:
                not_matched = False
                result = {}

    return results, matched_ids


def match_all_patterns(program):
    matched_results = {}
    matched_ids = set()
    for pattern_name in _ALL_PATTERNS:
        if pattern_name in _USED_PATTERNS:
            logger.debug(
                f'in match_all_patterns func, Matching Pattern {pattern_name}'
            )
            pattern = _ALL_PATTERNS[pattern_name]
            results, matched = match_pattern(pattern, program)
            for result in results:
                has_matched = False
                for id in result:
                    if result[id] in matched_ids:
                        has_matched = True
                        break
                if not has_matched:
                    for item in result:
                        matched_ids.add(result[id])
                    if pattern.name not in matched_results:
                        matched_results[pattern.name] = []
                    matched_results[pattern.name].append(result)

    return matched_results
