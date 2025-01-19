# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

# The set of ops that support fp16 and bf16 calculation and are considered numerically-
# safe and performance-critical. These ops are always converted to fp16 or bf16.
WHITE_LIST = {
    'conv2d',
    'einsum',
    'matmul',
    'matmul_v2',
    'max_pool2d_with_index',
    'mul',
    'fused_gemm_epilogue',
    "fused_rotary_position_embedding",
    "flash_attn",
}

# The set of ops that support fp16, and bf16 was unsupported.
ONLY_FP16_WHITE_LIST = {
    'fake_quantize_dequantize_abs_max',
    'fake_quantize_dequantize_moving_average_abs_max',
    'fused_attention',
    'fused_feedforward',
}

FP16_WHITE_LIST = WHITE_LIST | ONLY_FP16_WHITE_LIST

# The set of ops that support fp16 calculation and are considered numerically-
# dangerous and whose effects may also be observed in downstream ops.
FP16_BLACK_LIST = {
    'tan',
    'acos',
    'asin',
    'sinh',
    'cosh',
    'atanh',
    'tanh_shrink',
    'erfinv',
    'exp',
    'expm1',
    'log',
    'log10',
    'log2',
    'reciprocal',
    'rsqrt',
    'pow',
    'square',
    'reduce_sum',
    'mean',
    'reduce_mean',
    'reduce_prod',
    'cumprod',
    'cumsum',
    'dist',
    'pnorm',
    'frobenius_norm',
    'renorm',
    'group_norm',
    'layer_norm',
    'softmax',
    'softmin',
    'softplus',
    'log_softmax',
    'softmax_with_cross_entropy',
    'sigmoid_cross_entropy_with_logits',
    'c_softmax_with_cross_entropy',
    'c_softmax_with_multi_label_cross_entropy',
    'cross_entropy',
    'cross_entropy2',
    'nll_loss',
    'huber_loss',
    'triplet_margin_loss',
    'log_loss',
    'hsigmoid_loss',
    'margin_cross_entropy',
}

# FP16/BF16 performance of grad op is worse than that of FP32. Use FP32 by default.
EXTRA_BLACK_LIST = {
    'linear_interp_v2',
    'nearest_interp_v2',
    'bilinear_interp_v2',
    'bicubic_interp_v2',
    'trilinear_interp_v2',
    'lookup_table',
    'lookup_table_v2',
    'scatter',
}

BF16_WHITE_LIST = WHITE_LIST
BF16_BLACK_LIST = FP16_BLACK_LIST


# At OD level, ops in WHITE_LIST will use FP16/BF16 and the others will use FP32.
def white_list():
    white_list = {
        "float16": {
            "OD": FP16_WHITE_LIST,
            "O1": FP16_WHITE_LIST,
            "O2": FP16_WHITE_LIST,
        },
        "bfloat16": {
            "OD": BF16_WHITE_LIST,
            "O1": BF16_WHITE_LIST,
            "O2": BF16_WHITE_LIST,
        },
    }
    return white_list


def black_list():
    black_list = {
        "float16": {
            "OD": set(),
            "O1": FP16_BLACK_LIST | EXTRA_BLACK_LIST,
            "O2": EXTRA_BLACK_LIST,
        },
        "bfloat16": {
            "OD": set(),
            "O1": BF16_BLACK_LIST | EXTRA_BLACK_LIST,
            "O2": EXTRA_BLACK_LIST,
        },
    }
    return black_list
