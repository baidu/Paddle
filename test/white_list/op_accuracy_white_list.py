# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

# For op in NO_FP64_CHECK_GRAD_OP_LIST, the op test requires check_grad with fp64 precision
NO_FP64_CHECK_GRAD_OP_LIST = [
    'instance_norm',
    'affine_grid',
    'clip',
    'clip_tensor',
    'conv2d',
    'conv2d_transpose',
    'conv3d',
    'conv3d_transpose',
    'cudnn_lstm',
    'cvm',
    'data_norm',
    'deformable_conv',
    'deformable_conv_v1',
    'deformable_psroi_pooling',
    'depthwise_conv2d',
    'depthwise_conv2d_transpose',
    'dropout',
    'fused_elemwise_activation',
    'hinge_loss',
    'huber_loss',
    'im2sequence',
    'increment',
    'l1_norm',
    'log_loss',
    'logit',
    'lrn',
    'match_matrix_tensor',
    'matmul',
    'max_pool2d_v2',
    'max_pool2d_with_index',
    'max_pool3d_with_index',
    'fractional_max_pool2d',
    'fractional_max_pool3d',
    'minus',
    'nce',
    'pool2d',
    'pool3d',
    'prroi_pool',
    'reduce_max',
    'reduce_min',
    'reshape2',
    'row_conv',
    'scatter',
    'sequence_conv',
    'sequence_pool',
    'sequence_reverse',
    'sequence_slice',
    'shuffle_channel',
    'sigmoid',
    'smooth_l1_loss',
    'softmax',
    'spectral_norm',
    'squared_l2_norm',
    'tanh',
    'mish',
    'transpose2',
    'trilinear_interp',
    'trilinear_interp_v2',
    'var_conv_2d',
    'warpctc',
    'warprnnt',
    'bilateral_slice',
    'cast',
    'fake_channel_wise_quantize_dequantize_abs_max',
    'fake_quantize_dequantize_abs_max',
    'fake_quantize_dequantize_moving_average_abs_max',
]

NO_FP16_CHECK_GRAD_OP_LIST = [
    'fused_elemwise_activation',
    'pool2d',
    'pool3d',
    'softmax',
    'conv2d_transpose',
]

NO_FP16_COMPARED_WITH_FP32_OP_LIST = [
    'fake_quantize_moving_average_abs_max',
    'fused_scale_bias_relu_conv_bn',
    'fused_scale_bias_add_relu',
    'p_norm',
]

NO_BF16_COMPARED_WITH_FP32_OP_LIST = [
    'dequantize',
    'fusion_lstm',
]
