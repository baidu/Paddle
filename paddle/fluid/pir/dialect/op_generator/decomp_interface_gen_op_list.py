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

# =====================================
# DecompInterface gen op list
# =====================================


##################### decomp_rule ####################
# come into effect in generated file op_decomp_rule.cc
# manual decomp interface implementation are located in manual_op_decomp.cc
######################################################
MANUAL_IMPL_DECOMP = [
    "batch_norm",
    "batch_norm_",
    "clip",
    "one_hot",
]

GENERATE_IMPL_DECOMP = [
    "add_n",
    "addmm",
    "any",
    "baddbmm",
    "bce_loss",
    "bmm",
    "diag",
    "dropout",
    "elu",
    "embedding",
    "eye",
    "flatten",
    "full_like",
    "gelu",
    "group_norm",
    "hardsigmoid",
    "hardswish",
    "heaviside",
    "huber_loss",
    "index_sample",
    "index_select",
    "instance_norm",
    "kldiv_loss",
    "layer_norm",
    "leaky_relu",
    "lerp",
    "log_loss",
    "log_softmax",
    "mean",
    "mean_all",
    "meshgrid",
    "numel",
    "p_norm",
    "reciprocal",
    "relu",
    "relu6",
    "sigmoid_cross_entropy_with_logits",
    "silu",
    "softmax",
    "softsign",
    "square",
    "squared_l2_norm",
    "squeeze",
    "stack",
    "swiglu",
    "swish",
    "unbind",
    "unsqueeze",
    "unstack",
]
decomp_rule_interface_declare_gen_op_list = (
    GENERATE_IMPL_DECOMP + MANUAL_IMPL_DECOMP
)
decomp_rule_interface_implementation_gen_op_list = GENERATE_IMPL_DECOMP
# xshape output will no longer used after decomp, but return none to keep output num the same as origin op
decomp_ops_contain_unused_output = ["squeeze", "unsqueeze"]

##################### decomp_vjp ####################
# come into effect in generated file op_decomp_vjp.cc
# manual decomp interface implementation are located in manual_op_decomp_vjp.cc
####################################################

GENERATE_IMPL_VJP = [
    'abs_grad',
    'add_grad',
    'angle_grad',
    'bce_loss_grad',
    'cos_grad',
    'divide_grad',
    'elementwise_pow_grad',
    'erf_grad',
    'exp_grad',
    'floor_grad',
    'gelu_grad',
    'group_norm_grad',
    'hardsigmoid_grad',
    'hardswish_grad',
    'leaky_relu_grad',
    'layer_norm_grad',
    'log_grad',
    'matmul_grad',
    'max_grad',
    'maximum_grad',
    'mean_grad',
    'minimum_grad',
    'multiply_grad',
    'pow_grad',
    'reduce_as_grad',
    'relu_grad',
    'relu6_grad',
    'reshape_grad',
    'roll_grad',
    'rsqrt_grad',
    'sigmoid_grad',
    'silu_grad',
    'sin_grad',
    'softmax_grad',
    'sqrt_grad',
    'squeeze_grad',
    'subtract_grad',
    'sum_grad',
    'swiglu_grad',
    'tanh_grad',
    'transpose_grad',
    'unsqueeze_grad',
    'p_norm_grad',
]

# declare belongs to codegen, but implementation not
MANUAL_IMPL_VJP = ["concat_grad", "stack_grad", 'slice_grad']

decomp_vjp_interface_declare_gen_op_list = GENERATE_IMPL_VJP + MANUAL_IMPL_VJP

decomp_vjp_interface_implementation_gen_op_list = GENERATE_IMPL_VJP
