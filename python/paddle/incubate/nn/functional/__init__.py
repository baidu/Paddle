# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved
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

from .blha_get_max_len import blha_get_max_len
from .block_multihead_attention import (
    block_multihead_attention,
    block_multihead_attention_xpu,  # noqa: F401
)
from .fused_bias_act import fused_bias_act
from .fused_dot_product_attention import (
    cudnn_flash_attention,  # noqa: F401
    fused_dot_product_attention,  # noqa: F401
)
from .fused_dropout_add import fused_dropout_add
from .fused_gate_attention import fused_gate_attention  # noqa: F401
from .fused_layer_norm import fused_layer_norm
from .fused_matmul_bias import (
    fused_linear,
    fused_linear_activation,
    fused_matmul_bias,
)
from .fused_moe import fused_moe, moe_dispatch, moe_ffn, moe_reduce
from .fused_rms_norm import fused_rms_norm
from .fused_rotary_position_embedding import fused_rotary_position_embedding
from .fused_transformer import (
    fused_bias_dropout_residual_layer_norm,
    fused_feedforward,
    fused_multi_head_attention,
    fused_multi_transformer,
)
from .masked_multihead_attention import masked_multihead_attention
from .swiglu import swiglu
from .variable_length_memory_efficient_attention import (
    variable_length_memory_efficient_attention,
)

__all__ = [
    'fused_multi_head_attention',
    'fused_feedforward',
    'fused_multi_transformer',
    'fused_matmul_bias',
    'fused_linear',
    'fused_linear_activation',
    'fused_bias_dropout_residual_layer_norm',
    'fused_moe',
    "moe_dispatch",
    "moe_ffn",
    "moe_reduce",
    'fused_dropout_add',
    'fused_rotary_position_embedding',
    'variable_length_memory_efficient_attention',
    "fused_rms_norm",
    "fused_layer_norm",
    "fused_bias_act",
    "masked_multihead_attention",
    "blha_get_max_len",
    "block_multihead_attention",
    "swiglu",
]
