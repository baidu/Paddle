// Copyright (c) 2021 CINN Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <string>
#include <vector>

#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/lang/compute.h"
#include "paddle/cinn/poly/stage.h"

namespace cinn {
namespace hlir {
namespace pe {

/**
 * @brief Rectified Linear Unit.
 *
 * @param A The first Tensor
 * @param threshold The relu threshold (default: 0)
 * @param output_name The name of the output Tensor
 *
 * @return The result Tensor.
 */
ir::Tensor Relu(const ir::Tensor &A,
                double threshold = 0.0,
                const std::string &output_name = UniqName("T_Relu_out"));

/**
 * @brief Rectified Linear Unit bounded by six.
 *
 * @param A The first Tensor
 * @param threshold The relu threshold (default: 0)
 * @param output_name The name of the output Tensor
 *
 * @return The result Tensor.
 */
ir::Tensor Relu6(const ir::Tensor &A,
                 double threshold = 0.0,
                 const std::string &output_name = UniqName("T_Relu6_out"));

/**
 * @brief Leaky Rectified Linear Unit.
 *
 * @param A The first Tensor
 * @param alpha The slope for the small gradient when t < 0
 * @param output_name The name of the output Tensor
 *
 * @return The result Tensor.
 */
ir::Tensor LeakyRelu(
    const ir::Tensor &A,
    double alpha = 0.1,
    const std::string &output_name = UniqName("T_LeakyRelu_out"));

/**
 * @brief Leaky Rectified Linear Unit.
 *
 * @param A The first Tensor
 * @param slope The channel-wise slope tensor
 * @param axis The axis where the channel data needs to be applied
 * @param output_name The name of the output Tensor
 *
 * @return The result Tensor.
 */
ir::Tensor PRelu(const ir::Tensor &A,
                 const ir::Tensor &slope,
                 const int axis = 1,
                 const std::string &output_name = UniqName("T_PRelu_out"));

/**
 * @brief Perform a 2-D convolution with an NCHW-layout using winograd
 * algorithm.
 *
 * @param input The 4-D input tensor {N, C_in, H, W}
 * @param weights The 4-D weight tensor {C_out, C_in/group, filter_h, filter_w}
 * @param pad_h padding applied to the height of the image, default is 0
 * @param pad_w padding applied to the width of the image, default is 0
 * @param stride_h striding applied to the height of the image, default is 1
 * @param stride_w striding applied to the width of the image, default is 1
 * @param dilation_h dilation applied to the height of the image, default is 1
 * @param dilation_w dilation applied to the width of the image, default is 1
 * @param output_name The name of the output tensors
 *
 * @return the output tensor
 */
std::vector<ir::Tensor> Conv2d_winograd_NCHW(
    const ir::Tensor &input,
    const ir::Tensor &weights,
    int pad_h,
    int pad_w,
    int stride_h,
    int stride_w,
    int dilation_h,
    int dilation_w,
    const std::string &output_name = UniqName("T_Conv2d_winograd_NCHW_out"));

/**
 * @brief Perform a 2-D convolution with an NCHW-layout and support group and
 * depthwise convolution.
 *
 * @param input The 4-D input tensor {N, C_in, H, W}
 * @param weights The 4-D weight tensor {C_out, C_in/group, filter_h, filter_w}
 * @param pad_h padding applied to the height of the image, default is 0
 * @param pad_w padding applied to the width of the image, default is 0
 * @param stride_h striding applied to the height of the image, default is 1
 * @param stride_w striding applied to the width of the image, default is 1
 * @param dilation_h dilation applied to the height of the image, default is 1
 * @param dilation_w dilation applied to the width of the image, default is 1
 * @param output_name The name of the output tensors
 *
 * @return the output tensor
 */
std::vector<ir::Tensor> Conv2d_NCHW(
    const ir::Tensor &input,
    const ir::Tensor &weights,
    int pad_h,
    int pad_w,
    int stride_h,
    int stride_w,
    int dilation_h,
    int dilation_w,
    const std::string &output_name = UniqName("T_Conv2d_NCHW_out"),
    bool choose_direct_compute = false);

std::vector<ir::Tensor> Conv2d_NCHW_5D(
    const ir::Tensor &input,
    const ir::Tensor &weights,
    int pad_h,
    int pad_w,
    int stride_h,
    int stride_w,
    int dilation_h,
    int dilation_w,
    std::string key,
    const std::string &output_name = UniqName("T_Conv2d_NCHW_5D_out"),
    const cinn::common::Target &target = cinn::common::DefaultHostTarget());

/**
 * @brief Perform a 2-D convolution with an NCHWc-layout.
 *
 * @param input The 5-D input tensor {N, C_in_outer, H, W, C_in_inner}
 * @param weight The 6-D weight tensor {C_out_outer, C_filter_outer, filter_h,
 * filter_w, C_filter_inner, C_out_inner}
 * @param pad_h padding applied to the height of the image, default is 0
 * @param pad_w padding applied to the width of the image, default is 0
 * @param stride_h striding applied to the height of the image, default is 1
 * @param stride_w striding applied to the width of the image, default is 1
 * @param dilation_h dilation applied to the height of the image, default is 1
 * @param dilation_w dilation applied to the width of the image, default is 1
 * @param output_shapes The shape of the output tensors
 * @param output_name The name of the output tensors
 *
 * @return the output tensor
 */
std::vector<ir::Tensor> Conv2d_NCHWc(
    const ir::Tensor &input,
    const ir::Tensor &weights,
    int pad_h,
    int pad_w,
    int stride_h,
    int stride_w,
    int dilation_h,
    int dilation_w,
    const std::string &output_name = UniqName("T_Conv2d_NCHWc_out"),
    const cinn::common::Target &target = cinn::common::DefaultHostTarget());

#ifdef CINN_WITH_DNNL
std::vector<ir::Tensor> Conv2d_NCHW_ONEDNN(
    const ir::Tensor &input,
    const ir::Tensor &weights,
    int pad_h,
    int pad_w,
    int stride_h,
    int stride_w,
    int dilation_h,
    int dilation_w,
    const std::string &output_name = UniqName("T_Conv2d_NCHW_out"));
#endif

/**
 * @brief Perform a 2-D convolution with an NHWC-layout and support group and
 * depthwise convolution.
 *
 * @param input The 4-D input tensor {N, H, W, C_in}
 * @param weight The 4-D weight tensor {C_out, C_in/group, filter_h, filter_w}
 * @param pad_h padding applied to the height of the image, default is 0
 * @param pad_w padding applied to the width of the image, default is 0
 * @param stride_h striding applied to the height of the image, default is 1
 * @param stride_w striding applied to the width of the image, default is 1
 * @param dilation_h dilation applied to the height of the image, default is 1
 * @param dilation_w dilation applied to the width of the image, default is 1
 * @param output_shapes The shape of the output tensors
 * @param output_name The name of the output tensor
 *
 * @return the output tensors
 */
std::vector<ir::Tensor> Conv2d_NHWC(
    const ir::Tensor &input,
    const ir::Tensor &weights,
    int pad_h,
    int pad_w,
    int stride_h,
    int stride_w,
    int dilation_h,
    int dilation_w,
    const std::string &output_name = UniqName("T_Conv2d_NHWC_out"));

/**
 * @brief Perform a 2-D depthwise convolution with an NCHW-layout
 *
 * @param input The 4-D input tensor {N, C_in, H, W}
 * @param weight The 4-D weight tensor {C_in, channel_multiplier, filter_h,
 * filter_w}
 * @param pad_h padding counts applied to the height of the image, before and
 * after (symmetric padding)
 * @param pad_w padding counts applied to the width of the image, before and
 * after (symmetric padding)
 * @param stride_h striding counts applied to the height of the image, before
 * and after (symmetric padding)
 * @param stride_w striding counts applied to the width of the image, before and
 * after (symmetric padding)
 * @param output_shapes The shape of the output tensors
 * @param output_name The name of the output tensors
 *
 * @return the output tensor
 */
std::vector<ir::Tensor> Depthwise_Conv2d_NCHW(
    const ir::Tensor &input,
    const ir::Tensor &weight,
    int pad_h,
    int pad_w,
    int stride_h,
    int stride_w,
    const std::string output_name = UniqName("T_depthwise_conv2d_nchw"));

/**
 * @brief Perform a 2-D depthwise convolution with an NHWC-layout
 *
 * @param input The 4-D input tensor {N, H, W, C_in}
 * @param weight The 4-D weight tensor {C_in, channel_multiplier, filter_h,
 * filter_w}
 * @param pad_h padding counts applied to the height of the image, before and
 * after (symmetric padding)
 * @param pad_w padding counts applied to the width of the image, before and
 * after (symmetric padding)
 * @param stride_h striding counts applied to the height of the image, before
 * and after (symmetric padding)
 * @param stride_w striding counts applied to the width of the image, before and
 * after (symmetric padding)
 * @param output_shapes The shape of the output tensors
 * @param output_name The name of the output tensor
 *
 * @return the output tensors
 */
std::vector<ir::Tensor> Depthwise_Conv2d_NHWC(
    const ir::Tensor &input,
    const ir::Tensor &weight,
    int pad_h,
    int pad_w,
    int stride_h,
    int stride_w,
    const std::string output_name = UniqName("T_depthwise_conv2d_nhwc"));

ir::Tensor BatchNorm_NCHW(
    const ir::Tensor &input,
    const ir::Tensor &scale,
    const ir::Tensor &bias,
    const ir::Tensor &mean,
    const ir::Tensor &variance,
    float epsilon,
    const std::string &output_name = UniqName("T_BatchNorm_NCHW_out"));

ir::Tensor BatchNorm_NCHWc(
    const ir::Tensor &input,
    const ir::Tensor &scale,
    const ir::Tensor &bias,
    const ir::Tensor &mean,
    const ir::Tensor &variance,
    float epsilon,
    const std::string &output_name = UniqName("T_BatchNorm_NCHWc_out"));

/**
 * @brief Perform padding operation.
 * @param tensor The input tensor.
 * @param pad_before Vector of Exprs describing the padding before the
 * respective dimension
 * @param pad_after Vector of Exprs describing the padding after the respective
 * dimension
 * @param pad_value The value to fill padding elements with. Default is zero.
 * @param name The name of the output padding tensor
 * @param pad_mode Padding type to use: "constant" pads with constant_value;
 * "edge" pads using the edge values of the input array; "reflect" pads by
 * reflecting values with respect to the edges.
 *
 * @return the output tensor after padding.
 *
 * @note
 *  The pad_after vector must either be empty or have the same length as
 * pad_before When pad_after is empty, it takes the same values as pad_before
 * (symmetric padding) The pad vector applies from the leading dimensions and
 * skips missing trailing dimensions: e.g. pad(t(i, j, k), {1}, {1}) returns the
 * equivalent operation for the following pseudocode: for i in [0, t.shape[0] +
 * 2): for j in [0, t.shape[0] + 2): for k in [0, t.shape[0] + 2): name(i,j,k) =
 *                             i < 1 ? 0 :
 *                               ((1 <= i < t.shape[0] + 1) ?
 *                                 t(i-1, j, k) : 0));
 *
 */
ir::Tensor Pad(const ir::Tensor &tensor,
               const std::vector<Expr> &pad_before,
               std::vector<Expr> pad_after = std::vector<Expr>(),
               Expr pad_value = Expr(),
               const std::string &name = UniqName("T_pad_out"),
               const std::string &pad_mode = "constant");

std::vector<ir::Tensor> Softmax(
    const ir::Tensor &A,
    int axis = -1,
    const std::string &output_name = UniqName("T_softmax_out"));

#ifdef CINN_WITH_DNNL
std::vector<ir::Tensor> SoftmaxONEDNN(
    const ir::Tensor &A,
    int axis = -1,
    const std::string &output_name = UniqName("T_softmax_out"));
#endif

/**
 * @brief Perform pooling on the width dimension of the tensor.
 *        Width axis is determined by the data_format string in which 'W' means
 * width. Only support NCW and NWC data_format.
 * @param tensor The input tensor with shape of {N, C, W} or {N, W, C}
 * @param kernel_size Vector of ints: {pool_kernel_width}
 * @param stride_size Vector of ints: {pool_stride_width}
 * @param padding_size Vector of ints: {head_pad_width, tail_pad_width}
 * @param pool_type The type of pooling operator, currently support "max" and
 * "avg". Default is "max".
 * @param ceil_mode Whether to use ceil when calculating the output size.
 * Default is false.
 * @param exclusive Whether include padding in the calculation. Default is True.
 * @param data_format The input data format. Only support NCW and NWC
 * data_format.
 * @param output_name the name of the output tensor after padding and pooling.
 *
 * @return the vector of padding tensor and pooling tensor.
 */
std::vector<ir::Tensor> Pool1d(
    const ir::Tensor &tensor,
    const std::vector<int> &kernel_size,
    const std::vector<int> &stride_size,
    const std::vector<int> &padding_size,
    const std::string &pool_type = "max",
    bool ceil_mode = false,
    bool exclusive = true,
    const std::string &data_format = "NCW",
    const std::string &output_name = UniqName("T_Pool1d_out"));

/**
 * @brief Perform pooling on the height and width dimension of the tensor.
 *        Height and width axes are determined by the data_format string in
 * which 'H' means height and 'W' means width. Only support NCHW and NHWC
 * data_format.
 * @param tensor The input tensor with shape of {N, C, H, W} or {N, H, W, C}
 * @param kernel_size Vector of ints: {pool_kernel_height, pool_kernel_width}
 * @param stride_size Vector of ints: {pool_stride_height, pool_stride_width}
 * @param padding_size Vector of ints: {head_pad_height, head_pad_width,
 * tail_pad_height, tail_pad_width}
 * @param pool_type The type of pooling operator, currently support "max" and
 * "avg". Default is "max".
 * @param ceil_mode Whether to use ceil when calculating the output size.
 * Default is false.
 * @param exclusive Whether include padding in the calculation. Default is True.
 * @param data_format The input data format. Only support NCHW and NHWC
 * data_format.
 * @param output_name the name of the output tensor after padding and pooling.
 *
 * @return the vector of padding tensor and pooling tensor.
 */
std::vector<ir::Tensor> Pool2d(
    const ir::Tensor &tensor,
    const std::vector<int> &kernel_size,
    const std::vector<int> &stride_size,
    const std::vector<int> &padding_size,
    const std::string &pool_type = "max",
    bool ceil_mode = false,
    bool exclusive = true,
    const std::string &data_format = "NCHW",
    bool adaptive = false,
    const std::string &output_name = UniqName("T_Pool2d_out"));
std::vector<ir::Tensor> GlobalPool2d(const ir::Tensor &tensor,
                                     const std::string &pool_type,
                                     const std::string &output_name);
/**
 * @brief Perform pooling on the depth, height and width dimension of the
 * tensor. Depth, height and width axis is determined by the data_format string
 * in which 'D' means depth, 'H' means height and 'W' means width. Only support
 * NCDHW and NDHWC data_format.
 * @param tensor The input tensor with shape of {N, C, D, H, W} or {N, D, H, W,
 * C}
 * @param kernel_size Vector of ints: {pool_kernel_depth, pool_kernel_height,
 * pool_kernel_width}
 * @param stride_size Vector of ints: {pool_stride_depth, pool_stride_height,
 * pool_stride_width}
 * @param padding_size Vector of ints: {head_pad_depth, head_pad_height,
 * head_pad_width, tail_pad_depth, tail_pad_height, tail_pad_width}
 * @param pool_type The type of pooling operator, currently support "max" and
 * "avg". Default is "max".
 * @param ceil_mode Whether to use ceil when calculating the output size.
 * Default is false.
 * @param exclusive Whether include padding in the calculation. Default is True.
 * @param data_format The input data format. Only support NCDHW and NDHWC
 * data_format.
 * @param output_name the name of the output tensor after padding and pooling.
 */
std::vector<ir::Tensor> Pool3d(
    const ir::Tensor &tensor,
    const std::vector<int> &kernel_size,
    const std::vector<int> &stride_size,
    const std::vector<int> &padding_size,
    const std::string &pool_type = "max",
    bool ceil_mode = false,
    bool exclusive = true,
    const std::string &data_format = "NCDHW",
    const std::string &output_name = UniqName("T_Pool3d_out"));

/**
 * @brief Perform dropout in the inference which will downgrade the outcome at
 * inference or keep the same.
 * @param tensor The input tensor
 * @param dropout_prob float. Probability of setting units to zero.
 * @param dropout_implementation
 * ['downgrade_in_infer'(default)|'upscale_in_train']
 * 1. downgrade_in_infer(default), downgrade the outcome at inference
 *      out = input * (1.0 - dropout_prob)
 * 2. upscale_in_train, keep the same
 *      out = input
 * @param output_name the name of the output tensor.
 */
ir::Tensor DropoutInfer(
    const ir::Tensor &tensor,
    float dropout_prob,
    const std::string &dropout_implementation = "downgrade_in_infer",
    const std::string &output_name = UniqName("T_Dropout_infer_out"));

/**
 * @brief Perform Select for meta op 'Select'.
 * @param condition : the condition tensor for select value.
 * @param true_value : value for condition value = true.
 * @param false_value : value for condition value = false.
 * 1. condition expr = true
 *     out = true_value
 * 2. condition expr = false
 *     out = false_value
 * @param output_name : the name of the output tensor.
 */
ir::Tensor Select(const ir::Tensor &condition,
                  const ir::Tensor &true_value,
                  const ir::Tensor &false_value,
                  const std::string &output_name = UniqName("T_Select_out"));

}  // namespace pe
}  // namespace hlir
}  // namespace cinn
