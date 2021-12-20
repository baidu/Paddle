/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#pragma once

#ifdef PADDLE_WITH_XPU
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "paddle/fluid/framework/op_kernel_type.h"

namespace paddle {
namespace platform {

using vartype = paddle::framework::proto::VarType;
using pOpKernelType = paddle::framework::OpKernelType;
using XPUKernelSet =
    std::unordered_set<pOpKernelType, paddle::framework::OpKernelType::Hash>;
using XPUOpMap = std::unordered_map<std::string, XPUKernelSet>;

XPUOpMap& get_kl2_ops() {
  // KL1支持的op，通过op_name, data_type, place来索引
  static XPUOpMap s_xpu2_kernels{
      {"adamw", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"adam", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"arg_max", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"assign_value",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"batch_norm_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"batch_norm", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"cast", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                             pOpKernelType(vartype::FP16, XPUPlace()),
                             pOpKernelType(vartype::BOOL, XPUPlace()),
                             pOpKernelType(vartype::INT64, XPUPlace()),
                             pOpKernelType(vartype::INT32, XPUPlace())})},
      {"clip", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"concat_grad", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"concat", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"conv2d_grad", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"conv2d", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"depthwise_conv2d_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"depthwise_conv2d",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"dropout_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"dropout", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"elementwise_add_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"elementwise_add",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"elementwise_div_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"elementwise_div_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"elementwise_div",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"elementwise_div",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"elementwise_floordiv",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"elementwise_max_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"elementwise_max",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"elementwise_min_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"elementwise_min",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"elementwise_mul_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"elementwise_mul",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"elementwise_pow",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"elementwise_sub_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"elementwise_sub",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"equal", XPUKernelSet({pOpKernelType(vartype::INT64, XPUPlace()),
                              pOpKernelType(vartype::INT32, XPUPlace()),
                              pOpKernelType(vartype::FP32, XPUPlace())})},
      {"expand_as_v2",
       XPUKernelSet({pOpKernelType(vartype::INT32, XPUPlace()),
                     pOpKernelType(vartype::INT64, XPUPlace()),
                     pOpKernelType(vartype::BOOL, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace()),
                     pOpKernelType(vartype::FP32, XPUPlace())})},
      {"expand_v2", XPUKernelSet({pOpKernelType(vartype::INT32, XPUPlace()),
                                  pOpKernelType(vartype::INT64, XPUPlace()),
                                  pOpKernelType(vartype::BOOL, XPUPlace()),
                                  pOpKernelType(vartype::FP16, XPUPlace()),
                                  pOpKernelType(vartype::FP32, XPUPlace())})},
      {"fill_any_like",
       XPUKernelSet({pOpKernelType(vartype::INT64, XPUPlace()),
                     pOpKernelType(vartype::INT32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace()),
                     pOpKernelType(vartype::FP32, XPUPlace())})},
      {"fill_constant",
       XPUKernelSet({pOpKernelType(vartype::INT64, XPUPlace()),
                     pOpKernelType(vartype::INT32, XPUPlace()),
                     pOpKernelType(vartype::INT16, XPUPlace()),
                     pOpKernelType(vartype::INT8, XPUPlace()),
                     pOpKernelType(vartype::BOOL, XPUPlace()),
                     pOpKernelType(vartype::FP64, XPUPlace()),
                     pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace()),
                     pOpKernelType(vartype::BF16, XPUPlace()),
                     pOpKernelType(vartype::COMPLEX64, XPUPlace()),
                     pOpKernelType(vartype::COMPLEX128, XPUPlace())})},
      {"flatten2_grad",
       XPUKernelSet({pOpKernelType(vartype::INT64, XPUPlace()),
                     pOpKernelType(vartype::INT32, XPUPlace()),
                     pOpKernelType(vartype::INT8, XPUPlace()),
                     pOpKernelType(vartype::FP32, XPUPlace())})},
      {"flatten2", XPUKernelSet({pOpKernelType(vartype::INT64, XPUPlace()),
                                 pOpKernelType(vartype::INT32, XPUPlace()),
                                 pOpKernelType(vartype::INT8, XPUPlace()),
                                 pOpKernelType(vartype::FP32, XPUPlace())})},
      {"flatten_contiguous_range_grad",
       XPUKernelSet({pOpKernelType(vartype::INT64, XPUPlace()),
                     pOpKernelType(vartype::INT32, XPUPlace()),
                     pOpKernelType(vartype::INT8, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace()),
                     pOpKernelType(vartype::FP32, XPUPlace())})},
      {"flatten_contiguous_range",
       XPUKernelSet({pOpKernelType(vartype::INT64, XPUPlace()),
                     pOpKernelType(vartype::INT32, XPUPlace()),
                     pOpKernelType(vartype::INT8, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace()),
                     pOpKernelType(vartype::FP32, XPUPlace())})},
      {"flatten_grad",
       XPUKernelSet({pOpKernelType(vartype::INT64, XPUPlace()),
                     pOpKernelType(vartype::INT32, XPUPlace()),
                     pOpKernelType(vartype::INT8, XPUPlace()),
                     pOpKernelType(vartype::FP32, XPUPlace())})},
      {"flatten", XPUKernelSet({pOpKernelType(vartype::INT64, XPUPlace()),
                                pOpKernelType(vartype::INT32, XPUPlace()),
                                pOpKernelType(vartype::INT8, XPUPlace()),
                                pOpKernelType(vartype::FP32, XPUPlace())})},
      {"gather_grad", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                                    pOpKernelType(vartype::FP16, XPUPlace())})},
      {"gather_nd", XPUKernelSet({pOpKernelType(vartype::INT32, XPUPlace()),
                                  pOpKernelType(vartype::INT64, XPUPlace()),
                                  pOpKernelType(vartype::FP32, XPUPlace())})},
      {"gather", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                               pOpKernelType(vartype::FP16, XPUPlace())})},
      {"gaussian_random",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"gelu_grad", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                                  pOpKernelType(vartype::FP16, XPUPlace())})},
      {"gelu", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                             pOpKernelType(vartype::FP16, XPUPlace())})},
      {"greater_equal",
       XPUKernelSet({pOpKernelType(vartype::INT64, XPUPlace()),
                     pOpKernelType(vartype::INT32, XPUPlace()),
                     pOpKernelType(vartype::FP32, XPUPlace())})},
      {"greater_than",
       XPUKernelSet({pOpKernelType(vartype::INT64, XPUPlace()),
                     pOpKernelType(vartype::INT32, XPUPlace()),
                     pOpKernelType(vartype::FP32, XPUPlace())})},
      {"iou_similarity",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"label_smooth",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"layer_norm_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"layer_norm_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"layer_norm", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"layer_norm", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                                   pOpKernelType(vartype::FP16, XPUPlace())})},
      {"less_equal", XPUKernelSet({pOpKernelType(vartype::INT64, XPUPlace()),
                                   pOpKernelType(vartype::INT32, XPUPlace()),
                                   pOpKernelType(vartype::FP32, XPUPlace())})},
      {"less_than", XPUKernelSet({pOpKernelType(vartype::INT64, XPUPlace()),
                                  pOpKernelType(vartype::INT32, XPUPlace()),
                                  pOpKernelType(vartype::FP32, XPUPlace())})},
      {"log", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"lookup_table_v2_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"lookup_table_v2",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"masked_select",
       XPUKernelSet({pOpKernelType(vartype::INT32, XPUPlace()),
                     pOpKernelType(vartype::INT64, XPUPlace()),
                     pOpKernelType(vartype::FP32, XPUPlace())})},
      {"matmul_grad", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"matmul_v2_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"matmul_v2", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"matmul", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"mean_grad", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                                  pOpKernelType(vartype::FP16, XPUPlace())})},
      {"mean", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                             pOpKernelType(vartype::FP16, XPUPlace())})},
      {"momentum", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"mul", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                            pOpKernelType(vartype::FP16, XPUPlace())})},
      {"not_equal", XPUKernelSet({pOpKernelType(vartype::INT64, XPUPlace()),
                                  pOpKernelType(vartype::INT32, XPUPlace()),
                                  pOpKernelType(vartype::FP32, XPUPlace())})},
      {"one_hot_v2", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                                   pOpKernelType(vartype::INT64, XPUPlace())})},
      {"pool2d_grad", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                                    pOpKernelType(vartype::FP16, XPUPlace())})},
      {"pool2d", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                               pOpKernelType(vartype::FP16, XPUPlace())})},
      {"prior_box", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"range", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                              pOpKernelType(vartype::INT64, XPUPlace())})},
      {"reduce_max_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"reduce_max", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"reduce_mean_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"reduce_mean", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"reduce_sum_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"reduce_sum", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"relu_grad", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"relu", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"reshape2_grad",
       XPUKernelSet({pOpKernelType(vartype::FP64, XPUPlace()),
                     pOpKernelType(vartype::INT64, XPUPlace()),
                     pOpKernelType(vartype::INT32, XPUPlace()),
                     pOpKernelType(vartype::BOOL, XPUPlace()),
                     pOpKernelType(vartype::FP32, XPUPlace())})},
      {"reshape2", XPUKernelSet({pOpKernelType(vartype::FP64, XPUPlace()),
                                 pOpKernelType(vartype::INT64, XPUPlace()),
                                 pOpKernelType(vartype::INT32, XPUPlace()),
                                 pOpKernelType(vartype::BOOL, XPUPlace()),
                                 pOpKernelType(vartype::FP32, XPUPlace())})},
      {"scale", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"scale", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                              pOpKernelType(vartype::FP16, XPUPlace()),
                              pOpKernelType(vartype::INT64, XPUPlace())})},
      {"shape", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                              pOpKernelType(vartype::INT64, XPUPlace())})},
      {"slice_grad", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                                   pOpKernelType(vartype::FP16, XPUPlace()),
                                   pOpKernelType(vartype::INT32, XPUPlace())})},
      {"slice", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                              pOpKernelType(vartype::FP16, XPUPlace()),
                              pOpKernelType(vartype::INT32, XPUPlace())})},
      {"softmax_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"softmax_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"softmax_with_cross_entropy_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"softmax_with_cross_entropy",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"softmax", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"softmax", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                                pOpKernelType(vartype::FP16, XPUPlace())})},
      {"split", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                              pOpKernelType(vartype::INT32, XPUPlace())})},
      {"squeeze2_grad",
       XPUKernelSet({pOpKernelType(vartype::FP64, XPUPlace()),
                     pOpKernelType(vartype::INT64, XPUPlace()),
                     pOpKernelType(vartype::INT32, XPUPlace()),
                     pOpKernelType(vartype::BOOL, XPUPlace()),
                     pOpKernelType(vartype::INT8, XPUPlace()),
                     pOpKernelType(vartype::UINT8, XPUPlace()),
                     pOpKernelType(vartype::FP32, XPUPlace())})},
      {"squeeze2", XPUKernelSet({pOpKernelType(vartype::FP64, XPUPlace()),
                                 pOpKernelType(vartype::INT64, XPUPlace()),
                                 pOpKernelType(vartype::INT32, XPUPlace()),
                                 pOpKernelType(vartype::BOOL, XPUPlace()),
                                 pOpKernelType(vartype::INT8, XPUPlace()),
                                 pOpKernelType(vartype::UINT8, XPUPlace()),
                                 pOpKernelType(vartype::FP32, XPUPlace())})},
      {"stack", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                              pOpKernelType(vartype::INT64, XPUPlace()),
                              pOpKernelType(vartype::INT32, XPUPlace())})},
      {"sum", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                            pOpKernelType(vartype::FP16, XPUPlace())})},
      {"tanh_grad", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                                  pOpKernelType(vartype::FP16, XPUPlace())})},
      {"tanh", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                             pOpKernelType(vartype::FP16, XPUPlace())})},
      {"tile", XPUKernelSet({pOpKernelType(vartype::INT32, XPUPlace()),
                             pOpKernelType(vartype::INT64, XPUPlace()),
                             pOpKernelType(vartype::BOOL, XPUPlace()),
                             pOpKernelType(vartype::FP32, XPUPlace())})},
      {"transpose2_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"transpose2", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                                   pOpKernelType(vartype::FP16, XPUPlace())})},
      {"transpose_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"transpose", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                                  pOpKernelType(vartype::FP16, XPUPlace())})},
      {"unsqueeze2_grad",
       XPUKernelSet({pOpKernelType(vartype::FP64, XPUPlace()),
                     pOpKernelType(vartype::INT64, XPUPlace()),
                     pOpKernelType(vartype::INT32, XPUPlace()),
                     pOpKernelType(vartype::BOOL, XPUPlace()),
                     pOpKernelType(vartype::INT8, XPUPlace()),
                     pOpKernelType(vartype::UINT8, XPUPlace()),
                     pOpKernelType(vartype::FP32, XPUPlace())})},
      {"unsqueeze2", XPUKernelSet({pOpKernelType(vartype::FP64, XPUPlace()),
                                   pOpKernelType(vartype::INT64, XPUPlace()),
                                   pOpKernelType(vartype::INT32, XPUPlace()),
                                   pOpKernelType(vartype::BOOL, XPUPlace()),
                                   pOpKernelType(vartype::INT8, XPUPlace()),
                                   pOpKernelType(vartype::UINT8, XPUPlace()),
                                   pOpKernelType(vartype::FP32, XPUPlace())})},
      {"where_index", XPUKernelSet({pOpKernelType(vartype::INT32, XPUPlace()),
                                    pOpKernelType(vartype::BOOL, XPUPlace()),
                                    pOpKernelType(vartype::FP32, XPUPlace())})},
      {"where", XPUKernelSet({pOpKernelType(vartype::INT32, XPUPlace()),
                              pOpKernelType(vartype::INT64, XPUPlace()),
                              pOpKernelType(vartype::FP32, XPUPlace())})},
      // AddMore
  };

  return s_xpu2_kernels;
}

}  // namespace platform
}  // namespace paddle
#endif
