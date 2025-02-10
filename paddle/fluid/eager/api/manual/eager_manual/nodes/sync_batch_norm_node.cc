// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "glog/logging.h"
#include "paddle/common/flags.h"
#include "paddle/fluid/eager/api/generated/eager_generated/forwards/dygraph_functions.h"
#include "paddle/fluid/eager/api/manual/eager_manual/nodes/nodes.h"
#include "paddle/fluid/eager/api/utils/global_utils.h"
#include "paddle/fluid/eager/nan_inf_utils.h"
#include "paddle/fluid/eager/utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/imperative/tracer.h"
#include "paddle/phi/api/all.h"
#include "paddle/phi/api/backward/backward_api.h"
#include "paddle/phi/api/backward/sparse_bw_api.h"
#include "paddle/phi/api/include/sparse_api.h"
#include "paddle/phi/api/lib/api_custom_impl.h"
#include "paddle/phi/core/platform/profiler/event_tracing.h"

COMMON_DECLARE_bool(check_nan_inf);

paddle::small_vector<std::vector<paddle::Tensor>, egr::kSlotSmallVectorSize>
SyncBatchNormGradNode::operator()(
    paddle::small_vector<std::vector<paddle::Tensor>,
                         egr::kSlotSmallVectorSize>& grads,
    bool create_graph,
    bool is_new_grad) {
  VLOG(3) << "Running AD API GRAD: "
          << "sync_batch_norm_grad";
  // This 'Local_XXXGradNode' record event is different with
  // 'Global_XXXGradNode' event.
  // * 'Local_XXXGradNode' will only cover execution time of this function.
  // * 'Global_XXXGradNode' will not only cover execution time of this function,
  // but also include gradient
  //    accumulation when the output(s) of corresponding forward OP are shared
  //    by other OP(s), which may have extra accumulation overhead than
  //    'Local_XXXGradNode'.
  phi::RecordEvent node_execution_inner(
      "Local_SyncBatchNormGradNode", phi::TracerEventType::OperatorInner, 1);

  // Fill Zero For GradIn Tensors

  // Apply Gradient Hooks
  auto hooked_grads = ApplyGradientHooks(grads);

  // Collect GradIn Tensors, Attrs and Recovered TensorWrappers
  auto x = egr::EagerUtils::RecoverTensorWrapper(&this->x_);
  auto scale = egr::EagerUtils::RecoverTensorWrapper(&this->scale_);
  auto bias = egr::EagerUtils::RecoverTensorWrapper(&this->bias_);
  auto saved_mean = egr::EagerUtils::RecoverTensorWrapper(&this->saved_mean_);
  auto saved_variance =
      egr::EagerUtils::RecoverTensorWrapper(&this->saved_variance_);
  auto reserve_space =
      egr::EagerUtils::RecoverTensorWrapper(&this->reserve_space_);

  paddle::optional<paddle::Tensor> reserve_space_optional;
  if (reserve_space.impl())
    reserve_space_optional =
        paddle::make_optional<paddle::Tensor>(reserve_space);

  auto& out_grad = hooked_grads[0][0];
  auto& momentum = this->momentum_;
  auto& epsilon = this->epsilon_;
  auto& data_layout = this->data_layout_;
  auto& is_test = this->is_test_;
  auto& use_global_stats = this->use_global_stats_;
  auto& trainable_statistics = this->trainable_statistics_;
  // Prepare Grad function call

  const auto& out_metas = OutputMeta();
  paddle::small_vector<std::vector<paddle::Tensor>, egr::kSlotSmallVectorSize>
      returns(5);
  for (int i = 0; i < 5; ++i) {
    out_metas[i].empty() ? returns[i].resize(1)
                         : returns[i].resize(out_metas[i].size());
  }

  auto* api_output_0 =
      (out_metas[0].empty() || out_metas[0][0].IsStopGradient())
          ? nullptr
          : &returns[0][0];
  auto* api_output_1 =
      (out_metas[3].empty() || out_metas[3][0].IsStopGradient())
          ? nullptr
          : &returns[3][0];
  auto* api_output_2 =
      (out_metas[4].empty() || out_metas[4][0].IsStopGradient())
          ? nullptr
          : &returns[4][0];
  // Runtime check if we need next grad
  bool trace_backward = egr::Controller::Instance().HasGrad() && create_graph;

  // Inplace Check

  // Inplace Strategy

  VLOG(5) << "Running C++ API: "
          << "sync_batch_norm_grad";
  // Before log info

  if (VLOG_IS_ON(3)) {
    const char* INPUT_PRINT_TEMPLATE = "{ Input: [%s]} ";

    std::string input_str = "";
    std::string output_str = "";
    const char* TENSOR_OUT_GRAD_TEMPLATE = " \n( out_grad , [%s]), ";
    std::string input_out_grad_str = paddle::string::Sprintf(
        TENSOR_OUT_GRAD_TEMPLATE, egr::EagerUtils::TensorStr(out_grad));
    input_str += input_out_grad_str;
    const char* TENSOR_X_TEMPLATE = " \n( x , [%s]), ";
    std::string input_x_str = paddle::string::Sprintf(
        TENSOR_X_TEMPLATE, egr::EagerUtils::TensorStr(x));
    input_str += input_x_str;
    const char* TENSOR_SCALE_TEMPLATE = " \n( scale , [%s]), ";
    std::string input_scale_str = paddle::string::Sprintf(
        TENSOR_SCALE_TEMPLATE, egr::EagerUtils::TensorStr(scale));
    input_str += input_scale_str;
    const char* TENSOR_BIAS_TEMPLATE = " \n( bias , [%s]), ";
    std::string input_bias_str = paddle::string::Sprintf(
        TENSOR_BIAS_TEMPLATE, egr::EagerUtils::TensorStr(bias));
    input_str += input_bias_str;
    const char* TENSOR_SAVED_MEAN_TEMPLATE = " \n( saved_mean , [%s]), ";
    std::string input_saved_mean_str = paddle::string::Sprintf(
        TENSOR_SAVED_MEAN_TEMPLATE, egr::EagerUtils::TensorStr(saved_mean));
    input_str += input_saved_mean_str;
    const char* TENSOR_SAVED_VARIANCE_TEMPLATE =
        " \n( saved_variance , [%s]), ";
    std::string input_saved_variance_str =
        paddle::string::Sprintf(TENSOR_SAVED_VARIANCE_TEMPLATE,
                                egr::EagerUtils::TensorStr(saved_variance));
    input_str += input_saved_variance_str;
    const char* TENSOR_RESERVE_SPACE_TEMPLATE = " \n( reserve_space , [%s]), ";
    std::string input_reserve_space_str =
        paddle::string::Sprintf(TENSOR_RESERVE_SPACE_TEMPLATE,
                                egr::EagerUtils::TensorStr(reserve_space));
    input_str += input_reserve_space_str;
    VLOG(3) << paddle::string::Sprintf(INPUT_PRINT_TEMPLATE, input_str);
  }

  // Call grad_api function

  paddle::experimental::sync_batch_norm_grad(x,
                                             scale,
                                             bias,
                                             saved_mean,
                                             saved_variance,
                                             reserve_space_optional,
                                             out_grad,
                                             momentum,
                                             epsilon,
                                             data_layout,
                                             is_test,
                                             use_global_stats,
                                             trainable_statistics,
                                             api_output_0,
                                             api_output_1,
                                             api_output_2);
  // Check NaN and Inf id needed
  if (FLAGS_check_nan_inf) {
    egr::CheckTensorHasNanOrInf("sync_batch_norm_grad", returns);
  }

  // Get GradOut autograd_meta

  auto& x_grad = returns[0][0];
  egr::AutogradMeta* x_grad_autograd_meta =
      returns[0][0].initialized() ? egr::EagerUtils::autograd_meta(&x_grad)
                                  : nullptr;
  if (x_grad_autograd_meta) x_grad_autograd_meta->SetStopGradient(false);

  auto& scale_grad = returns[3][0];
  egr::AutogradMeta* scale_grad_autograd_meta =
      returns[3][0].initialized() ? egr::EagerUtils::autograd_meta(&scale_grad)
                                  : nullptr;
  if (scale_grad_autograd_meta)
    scale_grad_autograd_meta->SetStopGradient(false);

  auto& bias_grad = returns[4][0];
  egr::AutogradMeta* bias_grad_autograd_meta =
      returns[4][0].initialized() ? egr::EagerUtils::autograd_meta(&bias_grad)
                                  : nullptr;
  if (bias_grad_autograd_meta) bias_grad_autograd_meta->SetStopGradient(false);

  // Create Grad Node
  if (trace_backward) {
    PADDLE_THROW(common::errors::Unavailable(
        "The Op sync_batch_norm_grad doesn't have any grad"
        "op. If you don't intend calculating higher order"
        "derivatives, please set `create_graph`to False."));
  }
  VLOG(4) << "Finish AD API GRAD: sync_batch_norm_grad";
  // LOG IF DEBUG

  if (VLOG_IS_ON(4)) {
    const char* INPUT_PRINT_TEMPLATE = "{ Input: [%s],  \n Output: [%s] } ";

    std::string input_str = "";
    std::string output_str = "";
    const char* TENSOR_OUT_GRAD_TEMPLATE = " \n( out_grad , [%s]), ";
    std::string input_out_grad_str = paddle::string::Sprintf(
        TENSOR_OUT_GRAD_TEMPLATE, egr::EagerUtils::TensorStr(out_grad));
    input_str += input_out_grad_str;
    const char* TENSOR_X_TEMPLATE = " \n( x , [%s]), ";
    std::string input_x_str = paddle::string::Sprintf(
        TENSOR_X_TEMPLATE, egr::EagerUtils::TensorStr(x));
    input_str += input_x_str;
    const char* TENSOR_SCALE_TEMPLATE = " \n( scale , [%s]), ";
    std::string input_scale_str = paddle::string::Sprintf(
        TENSOR_SCALE_TEMPLATE, egr::EagerUtils::TensorStr(scale));
    input_str += input_scale_str;
    const char* TENSOR_BIAS_TEMPLATE = " \n( bias , [%s]), ";
    std::string input_bias_str = paddle::string::Sprintf(
        TENSOR_BIAS_TEMPLATE, egr::EagerUtils::TensorStr(bias));
    input_str += input_bias_str;
    const char* TENSOR_SAVED_MEAN_TEMPLATE = " \n( saved_mean , [%s]), ";
    std::string input_saved_mean_str = paddle::string::Sprintf(
        TENSOR_SAVED_MEAN_TEMPLATE, egr::EagerUtils::TensorStr(saved_mean));
    input_str += input_saved_mean_str;
    const char* TENSOR_SAVED_VARIANCE_TEMPLATE =
        " \n( saved_variance , [%s]), ";
    std::string input_saved_variance_str =
        paddle::string::Sprintf(TENSOR_SAVED_VARIANCE_TEMPLATE,
                                egr::EagerUtils::TensorStr(saved_variance));
    input_str += input_saved_variance_str;
    const char* TENSOR_RESERVE_SPACE_TEMPLATE = " \n( reserve_space , [%s]), ";
    std::string input_reserve_space_str =
        paddle::string::Sprintf(TENSOR_RESERVE_SPACE_TEMPLATE,
                                egr::EagerUtils::TensorStr(reserve_space));
    input_str += input_reserve_space_str;
    const char* TENSOR_X_GRAD_TEMPLATE = " \n ( x_grad , [%s]), ";
    std::string output_x_grad_str = paddle::string::Sprintf(
        TENSOR_X_GRAD_TEMPLATE, egr::EagerUtils::TensorStr(x_grad));
    output_str += output_x_grad_str;
    const char* TENSOR_SCALE_GRAD_TEMPLATE = " \n ( scale_grad , [%s]), ";
    std::string output_scale_grad_str = paddle::string::Sprintf(
        TENSOR_SCALE_GRAD_TEMPLATE, egr::EagerUtils::TensorStr(scale_grad));
    output_str += output_scale_grad_str;
    const char* TENSOR_BIAS_GRAD_TEMPLATE = " \n ( bias_grad , [%s]), ";
    std::string output_bias_grad_str = paddle::string::Sprintf(
        TENSOR_BIAS_GRAD_TEMPLATE, egr::EagerUtils::TensorStr(bias_grad));
    output_str += output_bias_grad_str;
    VLOG(4) << paddle::string::Sprintf(
        INPUT_PRINT_TEMPLATE, input_str, output_str);
  }

  if (HasNodePostHook()) {
    returns = ApplyNodePostHooks(returns, hooked_grads);
  }

  // Return
  if (NeedComplexToRealConversion()) HandleComplexGradToRealGrad(&returns);
  return returns;
}

namespace sparse {
paddle::small_vector<std::vector<paddle::Tensor>, egr::kSlotSmallVectorSize>
SyncBatchNormGradNode::operator()(
    paddle::small_vector<std::vector<paddle::Tensor>,
                         egr::kSlotSmallVectorSize>& grads,
    bool create_graph,
    bool is_new_grad) {
  VLOG(3) << "Running AD API GRAD: "
          << "sync_batch_norm_grad";
  // This 'Local_XXXGradNode' record event is different with
  // 'Global_XXXGradNode' event.
  // * 'Local_XXXGradNode' will only cover execution time of this function.
  // * 'Global_XXXGradNode' will not only cover execution time of this function,
  // but also include gradient
  //    accumulation when the output(s) of corresponding forward OP are shared
  //    by other OP(s), which may have extra accumulation overhead than
  //    'Local_XXXGradNode'.
  phi::RecordEvent node_execution_inner(
      "Local_SyncBatchNormGradNode", phi::TracerEventType::OperatorInner, 1);

  // Fill Zero For GradIn Tensors

  // Apply Gradient Hooks
  auto hooked_grads = ApplyGradientHooks(grads);

  // Collect GradIn Tensors, Attrs and Recovered TensorWrappers
  auto x = egr::EagerUtils::RecoverTensorWrapper(&this->x_);
  auto scale = egr::EagerUtils::RecoverTensorWrapper(&this->scale_);
  auto bias = egr::EagerUtils::RecoverTensorWrapper(&this->bias_);
  auto saved_mean = egr::EagerUtils::RecoverTensorWrapper(&this->saved_mean_);
  auto saved_variance =
      egr::EagerUtils::RecoverTensorWrapper(&this->saved_variance_);
  auto reserve_space =
      egr::EagerUtils::RecoverTensorWrapper(&this->reserve_space_);

  paddle::optional<paddle::Tensor> reserve_space_optional;
  if (reserve_space.impl())
    reserve_space_optional =
        paddle::make_optional<paddle::Tensor>(reserve_space);

  auto& out_grad = hooked_grads[0][0];
  auto& momentum = this->momentum_;
  auto& epsilon = this->epsilon_;
  auto& data_layout = this->data_layout_;
  auto& is_test = this->is_test_;
  auto& use_global_stats = this->use_global_stats_;
  auto& trainable_statistics = this->trainable_statistics_;
  // Prepare Grad function call

  const auto& out_metas = OutputMeta();
  paddle::small_vector<std::vector<paddle::Tensor>, egr::kSlotSmallVectorSize>
      returns(5);
  for (int i = 0; i < 5; ++i) {
    out_metas[i].empty() ? returns[i].resize(1)
                         : returns[i].resize(out_metas[i].size());
  }

  auto* api_output_0 =
      (out_metas[0].empty() || out_metas[0][0].IsStopGradient())
          ? nullptr
          : &returns[0][0];
  auto* api_output_1 =
      (out_metas[3].empty() || out_metas[3][0].IsStopGradient())
          ? nullptr
          : &returns[3][0];
  auto* api_output_2 =
      (out_metas[4].empty() || out_metas[4][0].IsStopGradient())
          ? nullptr
          : &returns[4][0];
  // Runtime check if we need next grad
  bool trace_backward = egr::Controller::Instance().HasGrad() && create_graph;

  // Inplace Check

  // Inplace Strategy

  VLOG(5) << "Running C++ API: "
          << "sync_batch_norm_grad";
  // Before log info

  if (VLOG_IS_ON(3)) {
    const char* INPUT_PRINT_TEMPLATE = "{ Input: [%s]} ";

    std::string input_str = "";
    std::string output_str = "";
    const char* TENSOR_OUT_GRAD_TEMPLATE = " \n( out_grad , [%s]), ";
    std::string input_out_grad_str = paddle::string::Sprintf(
        TENSOR_OUT_GRAD_TEMPLATE, egr::EagerUtils::TensorStr(out_grad));
    input_str += input_out_grad_str;
    const char* TENSOR_X_TEMPLATE = " \n( x , [%s]), ";
    std::string input_x_str = paddle::string::Sprintf(
        TENSOR_X_TEMPLATE, egr::EagerUtils::TensorStr(x));
    input_str += input_x_str;
    const char* TENSOR_SCALE_TEMPLATE = " \n( scale , [%s]), ";
    std::string input_scale_str = paddle::string::Sprintf(
        TENSOR_SCALE_TEMPLATE, egr::EagerUtils::TensorStr(scale));
    input_str += input_scale_str;
    const char* TENSOR_BIAS_TEMPLATE = " \n( bias , [%s]), ";
    std::string input_bias_str = paddle::string::Sprintf(
        TENSOR_BIAS_TEMPLATE, egr::EagerUtils::TensorStr(bias));
    input_str += input_bias_str;
    const char* TENSOR_SAVED_MEAN_TEMPLATE = " \n( saved_mean , [%s]), ";
    std::string input_saved_mean_str = paddle::string::Sprintf(
        TENSOR_SAVED_MEAN_TEMPLATE, egr::EagerUtils::TensorStr(saved_mean));
    input_str += input_saved_mean_str;
    const char* TENSOR_SAVED_VARIANCE_TEMPLATE =
        " \n( saved_variance , [%s]), ";
    std::string input_saved_variance_str =
        paddle::string::Sprintf(TENSOR_SAVED_VARIANCE_TEMPLATE,
                                egr::EagerUtils::TensorStr(saved_variance));
    input_str += input_saved_variance_str;
    const char* TENSOR_RESERVE_SPACE_TEMPLATE = " \n( reserve_space , [%s]), ";
    std::string input_reserve_space_str =
        paddle::string::Sprintf(TENSOR_RESERVE_SPACE_TEMPLATE,
                                egr::EagerUtils::TensorStr(reserve_space));
    input_str += input_reserve_space_str;
    VLOG(3) << paddle::string::Sprintf(INPUT_PRINT_TEMPLATE, input_str);
  }

  // Call grad_api function

  paddle::experimental::sparse::sync_batch_norm_grad(x,
                                                     scale,
                                                     bias,
                                                     saved_mean,
                                                     saved_variance,
                                                     reserve_space_optional,
                                                     out_grad,
                                                     momentum,
                                                     epsilon,
                                                     data_layout,
                                                     is_test,
                                                     use_global_stats,
                                                     trainable_statistics,
                                                     api_output_0,
                                                     api_output_1,
                                                     api_output_2);
  // Check NaN and Inf id needed
  if (FLAGS_check_nan_inf) {
    egr::CheckTensorHasNanOrInf("sync_batch_norm_grad", returns);
  }

  // Get GradOut autograd_meta

  auto& x_grad = returns[0][0];
  egr::AutogradMeta* x_grad_autograd_meta =
      returns[0][0].initialized() ? egr::EagerUtils::autograd_meta(&x_grad)
                                  : nullptr;
  if (x_grad_autograd_meta) x_grad_autograd_meta->SetStopGradient(false);

  auto& scale_grad = returns[3][0];
  egr::AutogradMeta* scale_grad_autograd_meta =
      returns[3][0].initialized() ? egr::EagerUtils::autograd_meta(&scale_grad)
                                  : nullptr;
  if (scale_grad_autograd_meta)
    scale_grad_autograd_meta->SetStopGradient(false);

  auto& bias_grad = returns[4][0];
  egr::AutogradMeta* bias_grad_autograd_meta =
      returns[4][0].initialized() ? egr::EagerUtils::autograd_meta(&bias_grad)
                                  : nullptr;
  if (bias_grad_autograd_meta) bias_grad_autograd_meta->SetStopGradient(false);

  // Create Grad Node
  if (trace_backward) {
    PADDLE_THROW(common::errors::Unavailable(
        "The Op sync_batch_norm_grad doesn't have any grad"
        "op. If you don't intend calculating higher order"
        "derivatives, please set `create_graph`to False."));
  }
  VLOG(4) << "Finish AD API GRAD: sync_batch_norm_grad";
  // LOG IF DEBUG

  if (VLOG_IS_ON(4)) {
    const char* INPUT_PRINT_TEMPLATE = "{ Input: [%s],  \n Output: [%s] } ";

    std::string input_str = "";
    std::string output_str = "";
    const char* TENSOR_OUT_GRAD_TEMPLATE = " \n( out_grad , [%s]), ";
    std::string input_out_grad_str = paddle::string::Sprintf(
        TENSOR_OUT_GRAD_TEMPLATE, egr::EagerUtils::TensorStr(out_grad));
    input_str += input_out_grad_str;
    const char* TENSOR_X_TEMPLATE = " \n( x , [%s]), ";
    std::string input_x_str = paddle::string::Sprintf(
        TENSOR_X_TEMPLATE, egr::EagerUtils::TensorStr(x));
    input_str += input_x_str;
    const char* TENSOR_SCALE_TEMPLATE = " \n( scale , [%s]), ";
    std::string input_scale_str = paddle::string::Sprintf(
        TENSOR_SCALE_TEMPLATE, egr::EagerUtils::TensorStr(scale));
    input_str += input_scale_str;
    const char* TENSOR_BIAS_TEMPLATE = " \n( bias , [%s]), ";
    std::string input_bias_str = paddle::string::Sprintf(
        TENSOR_BIAS_TEMPLATE, egr::EagerUtils::TensorStr(bias));
    input_str += input_bias_str;
    const char* TENSOR_SAVED_MEAN_TEMPLATE = " \n( saved_mean , [%s]), ";
    std::string input_saved_mean_str = paddle::string::Sprintf(
        TENSOR_SAVED_MEAN_TEMPLATE, egr::EagerUtils::TensorStr(saved_mean));
    input_str += input_saved_mean_str;
    const char* TENSOR_SAVED_VARIANCE_TEMPLATE =
        " \n( saved_variance , [%s]), ";
    std::string input_saved_variance_str =
        paddle::string::Sprintf(TENSOR_SAVED_VARIANCE_TEMPLATE,
                                egr::EagerUtils::TensorStr(saved_variance));
    input_str += input_saved_variance_str;
    const char* TENSOR_RESERVE_SPACE_TEMPLATE = " \n( reserve_space , [%s]), ";
    std::string input_reserve_space_str =
        paddle::string::Sprintf(TENSOR_RESERVE_SPACE_TEMPLATE,
                                egr::EagerUtils::TensorStr(reserve_space));
    input_str += input_reserve_space_str;
    const char* TENSOR_X_GRAD_TEMPLATE = " \n ( x_grad , [%s]), ";
    std::string output_x_grad_str = paddle::string::Sprintf(
        TENSOR_X_GRAD_TEMPLATE, egr::EagerUtils::TensorStr(x_grad));
    output_str += output_x_grad_str;
    const char* TENSOR_SCALE_GRAD_TEMPLATE = " \n ( scale_grad , [%s]), ";
    std::string output_scale_grad_str = paddle::string::Sprintf(
        TENSOR_SCALE_GRAD_TEMPLATE, egr::EagerUtils::TensorStr(scale_grad));
    output_str += output_scale_grad_str;
    const char* TENSOR_BIAS_GRAD_TEMPLATE = " \n ( bias_grad , [%s]), ";
    std::string output_bias_grad_str = paddle::string::Sprintf(
        TENSOR_BIAS_GRAD_TEMPLATE, egr::EagerUtils::TensorStr(bias_grad));
    output_str += output_bias_grad_str;
    VLOG(6) << "gradnode_ptr = " << this;
    VLOG(4) << paddle::string::Sprintf(
        INPUT_PRINT_TEMPLATE, input_str, output_str);
  }

  if (HasNodePostHook()) {
    returns = ApplyNodePostHooks(returns, hooked_grads);
  }

  // Return
  if (NeedComplexToRealConversion()) HandleComplexGradToRealGrad(&returns);
  return returns;
}

}  // namespace sparse
