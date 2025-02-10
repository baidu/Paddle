// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/prim/api/all.h"
#include "paddle/fluid/prim/api/composite_backward/composite_backward_api.h"
#include "paddle/fluid/prim/utils/utils.h"
#include "paddle/phi/api/all.h"
#include "paddle/phi/api/backward/backward_api.h"
#include "paddle/phi/api/backward/sparse_bw_api.h"
#include "paddle/phi/api/include/sparse_api.h"
#include "paddle/phi/api/lib/api_custom_impl.h"
#include "paddle/phi/core/platform/profiler/event_tracing.h"

using egr::ConvertAllInputsToDistTensor;
using egr::InputsContainDistTensor;

COMMON_DECLARE_bool(check_nan_inf);

paddle::small_vector<std::vector<paddle::Tensor>, egr::kSlotSmallVectorSize>
MultiplyGradNode::operator()(
    paddle::small_vector<std::vector<paddle::Tensor>,
                         egr::kSlotSmallVectorSize>& grads,
    bool create_graph,
    bool is_new_grad) {
  VLOG(3) << "Running AD API GRAD: "
          << "multiply_grad";
  // This 'Local_XXXGradNode' record event is different with
  // 'Global_XXXGradNode' event.
  // * 'Local_XXXGradNode' will only cover execution time of this function.
  // * 'Global_XXXGradNode' will not only cover execution time of this function,
  // but also include gradient
  //    accumulation when the output(s) of corresponding forward OP are shared
  //    by other OP(s), which may have extra accumulation overhead than
  //    'Local_XXXGradNode'.
  phi::RecordEvent node_execution_inner(
      "Local_MultiplyGradNode", phi::TracerEventType::OperatorInner, 1);

  // Fill Zero For GradIn Tensors
  const auto& input_metas = this->InputMeta();
  egr::EagerUtils::FillZeroForEmptyGradInput(&grads[0][0], input_metas[0][0]);

  // Apply Gradient Hooks
  auto hooked_grads = ApplyGradientHooks(grads);

  // Collect GradIn Tensors, Attrs and Recovered TensorWrappers
  auto x = egr::EagerUtils::RecoverTensorWrapper(&this->x_);
  auto y = egr::EagerUtils::RecoverTensorWrapper(&this->y_);
  auto& grad_out = hooked_grads[0][0];
  auto& axis = this->axis_;

  // Convert All Inputs to DistTensor if Necessary
  const phi::distributed::ProcessMesh* mesh = nullptr;
  bool inputs_contain_dist_tensor = InputsContainDistTensor(&mesh, grad_out);
  if (inputs_contain_dist_tensor) {
    ConvertAllInputsToDistTensor(mesh, x, y);
  }

  // Prepare Grad function call

  const auto& out_metas = OutputMeta();
  paddle::small_vector<std::vector<paddle::Tensor>, egr::kSlotSmallVectorSize>
      returns(2);
  for (int i = 0; i < 2; ++i) {
    out_metas[i].empty() ? returns[i].resize(1)
                         : returns[i].resize(out_metas[i].size());
  }

  auto* api_output_0 =
      (out_metas[0].empty() || out_metas[0][0].IsStopGradient())
          ? nullptr
          : &returns[0][0];
  auto* api_output_1 =
      (out_metas[1].empty() || out_metas[1][0].IsStopGradient())
          ? nullptr
          : &returns[1][0];
  // Runtime check if we need next grad
  bool trace_backward = egr::Controller::Instance().HasGrad() && create_graph;

  // Set DistAttr of Out Tensor for semi-auto parallel
  if (IsRunAutoParallel() || inputs_contain_dist_tensor) {
    egr::EagerUtils::SetGradOutputDistAttr(
        out_metas, {0, 1}, *mesh, api_output_0, api_output_1);
  }

  // Inplace Check

  // Inplace Strategy

  VLOG(5) << "Running C++ API: "
          << "multiply_grad";
  // Before log info

  if (VLOG_IS_ON(3)) {
    const char* INPUT_PRINT_TEMPLATE = "{ Input: [%s]} ";

    std::string input_str = "";
    std::string output_str = "";
    const char* TENSOR_GRAD_OUT_TEMPLATE = " \n( grad_out , [%s]), ";
    std::string input_grad_out_str = paddle::string::Sprintf(
        TENSOR_GRAD_OUT_TEMPLATE, egr::EagerUtils::TensorStr(grad_out));
    input_str += input_grad_out_str;
    const char* TENSOR_X_TEMPLATE = " \n( x , [%s]), ";
    std::string input_x_str = paddle::string::Sprintf(
        TENSOR_X_TEMPLATE, egr::EagerUtils::TensorStr(x));
    input_str += input_x_str;
    const char* TENSOR_Y_TEMPLATE = " \n( y , [%s]), ";
    std::string input_y_str = paddle::string::Sprintf(
        TENSOR_Y_TEMPLATE, egr::EagerUtils::TensorStr(y));
    input_str += input_y_str;
    VLOG(3) << paddle::string::Sprintf(INPUT_PRINT_TEMPLATE, input_str);
  }

  // Call grad_api function

  std::string grad_op_name = "multiply_grad";
  auto need_skip =
      paddle::prim::StaticCompositeContext::Instance().CheckSkipCompOps(
          grad_op_name);
  if (paddle::prim::PrimCommonUtils::IsEagerPrimEnabled() && !need_skip) {
    bool original_global_grad = egr::Controller::Instance().HasGrad();
    if (!create_graph) {
      egr::Controller::Instance().SetHasGrad(create_graph);
    }
    paddle::prim::multiply_grad<paddle::Tensor>(
        x, y, grad_out, axis, api_output_0, api_output_1);
    VLOG(4) << "Composite api multiply_grad is called ";
    if (!create_graph) {
      egr::Controller::Instance().SetHasGrad(original_global_grad);
    }
  } else {
    paddle::experimental::multiply_grad(
        x, y, grad_out, axis, api_output_0, api_output_1);
    VLOG(4) << "Fused api multiply_grad is called ";
  }

  // Check NaN and Inf id needed

  if (FLAGS_check_nan_inf) {
    try {
      egr::CheckTensorHasNanOrInf("multiply_grad", returns);
    } catch (...) {
      LOG(WARNING) << "There are nan/inf in (multiply_grad)";
      auto forward_trace = GetForwardTrace();
      std::cout << forward_trace << std::endl;
      std::rethrow_exception(std::current_exception());
    }
  }

  // Get GradOut autograd_meta

  auto& grad_x = returns[0][0];
  egr::AutogradMeta* grad_x_autograd_meta =
      returns[0][0].initialized() ? egr::EagerUtils::autograd_meta(&grad_x)
                                  : nullptr;
  if (grad_x_autograd_meta) grad_x_autograd_meta->SetStopGradient(false);

  auto& grad_y = returns[1][0];
  egr::AutogradMeta* grad_y_autograd_meta =
      returns[1][0].initialized() ? egr::EagerUtils::autograd_meta(&grad_y)
                                  : nullptr;
  if (grad_y_autograd_meta) grad_y_autograd_meta->SetStopGradient(false);

  // Create Grad Node

  if (!paddle::prim::PrimCommonUtils::IsEagerPrimEnabled() || need_skip) {
    if (trace_backward) {
      phi::RecordEvent node_creation_record_event(
          "multiply_grad node_creation",
          phi::TracerEventType::OperatorInner,
          1);

      // Node Construction
      auto grad_node = std::shared_ptr<MultiplyDoubleGradNode>(  // NOLINT
          new MultiplyDoubleGradNode(2, 3));
      // SetAttributes if needed
      grad_node->SetAttribute_axis(axis);
      // Set TensorWrappers for Forward Inputs if needed
      grad_node->SetTensorWrapper_x(x);
      grad_node->SetTensorWrapper_y(y);
      grad_node->SetTensorWrapper_grad_out(grad_out);
      // SetGradOutMeta & SetEdges
      grad_node->SetGradOutMeta(x, 0);
      grad_node->SetGradOutMeta(y, 1);
      grad_node->SetGradOutMeta(grad_out, 2);
      // SetOutRank & SetHistory & SetGradInMeta
      if (grad_x_autograd_meta) {
        egr::EagerUtils::SetOutRankWithSlot(grad_x_autograd_meta, 0);
      }
      if (grad_y_autograd_meta) {
        egr::EagerUtils::SetOutRankWithSlot(grad_y_autograd_meta, 1);
      }
      if (grad_x_autograd_meta) {
        egr::EagerUtils::SetHistory(grad_x_autograd_meta, grad_node);
      }
      if (grad_y_autograd_meta) {
        egr::EagerUtils::SetHistory(grad_y_autograd_meta, grad_node);
      }
      grad_node->SetGradInMeta(grad_x, 0);
      grad_node->SetGradInMeta(grad_y, 1);
      // Set TensorWrappers for Forward Outputs if needed
    }
  }

  VLOG(4) << "Finish AD API GRAD: multiply_grad";
  VLOG(6) << "gradnode_ptr = " << this;
  // LOG IF DEBUG

  if (VLOG_IS_ON(4)) {
    const char* INPUT_PRINT_TEMPLATE = "{ Input: [%s],  \n Output: [%s] } ";

    std::string input_str = "";
    std::string output_str = "";
    const char* TENSOR_GRAD_OUT_TEMPLATE = " \n( grad_out , [%s]), ";
    std::string input_grad_out_str = paddle::string::Sprintf(
        TENSOR_GRAD_OUT_TEMPLATE, egr::EagerUtils::TensorStr(grad_out));
    input_str += input_grad_out_str;
    const char* TENSOR_X_TEMPLATE = " \n( x , [%s]), ";
    std::string input_x_str = paddle::string::Sprintf(
        TENSOR_X_TEMPLATE, egr::EagerUtils::TensorStr(x));
    input_str += input_x_str;
    const char* TENSOR_Y_TEMPLATE = " \n( y , [%s]), ";
    std::string input_y_str = paddle::string::Sprintf(
        TENSOR_Y_TEMPLATE, egr::EagerUtils::TensorStr(y));
    input_str += input_y_str;
    const char* TENSOR_GRAD_X_TEMPLATE = " \n ( grad_x , [%s]), ";
    std::string output_grad_x_str = paddle::string::Sprintf(
        TENSOR_GRAD_X_TEMPLATE, egr::EagerUtils::TensorStr(grad_x));
    output_str += output_grad_x_str;
    const char* TENSOR_GRAD_Y_TEMPLATE = " \n ( grad_y , [%s]), ";
    std::string output_grad_y_str = paddle::string::Sprintf(
        TENSOR_GRAD_Y_TEMPLATE, egr::EagerUtils::TensorStr(grad_y));
    output_str += output_grad_y_str;
    VLOG(4) << paddle::string::Sprintf(
        INPUT_PRINT_TEMPLATE, input_str, output_str);
  }

  // Return
  if (NeedComplexToRealConversion()) HandleComplexGradToRealGrad(&returns);
  return returns;
}

paddle::small_vector<std::vector<paddle::Tensor>, egr::kSlotSmallVectorSize>
MultiplyDoubleGradNode::operator()(
    paddle::small_vector<std::vector<paddle::Tensor>,
                         egr::kSlotSmallVectorSize>& grads,
    bool create_graph,
    bool is_new_grad) {
  VLOG(3) << "Running AD API GRAD: "
          << "multiply_double_grad";
  // This 'Local_XXXGradNode' record event is different with
  // 'Global_XXXGradNode' event.
  // * 'Local_XXXGradNode' will only cover execution time of this function.
  // * 'Global_XXXGradNode' will not only cover execution time of this function,
  // but also include gradient
  //    accumulation when the output(s) of corresponding forward OP are shared
  //    by other OP(s), which may have extra accumulation overhead than
  //    'Local_XXXGradNode'.
  phi::RecordEvent node_execution_inner(
      "Local_MultiplyDoubleGradNode", phi::TracerEventType::OperatorInner, 1);

  // Fill Zero For GradIn Tensors
  const auto& input_metas = this->InputMeta();
  egr::EagerUtils::FillZeroForEmptyOptionalGradInput(&grads[0][0],
                                                     input_metas[0][0]);
  egr::EagerUtils::FillZeroForEmptyOptionalGradInput(&grads[1][0],
                                                     input_metas[1][0]);

  // Apply Gradient Hooks
  auto hooked_grads = ApplyGradientHooks(grads);

  // Collect GradIn Tensors, Attrs and Recovered TensorWrappers
  auto x = egr::EagerUtils::RecoverTensorWrapper(&this->x_);
  auto y = egr::EagerUtils::RecoverTensorWrapper(&this->y_);
  auto fwd_grad_out = egr::EagerUtils::RecoverTensorWrapper(&this->grad_out_);
  auto& fwd_grad_grad_x = hooked_grads[0][0];

  paddle::optional<paddle::Tensor> fwd_grad_grad_x_optional;
  if (fwd_grad_grad_x.initialized())
    fwd_grad_grad_x_optional =
        paddle::make_optional<paddle::Tensor>(fwd_grad_grad_x);

  auto& fwd_grad_grad_y = hooked_grads[1][0];

  paddle::optional<paddle::Tensor> fwd_grad_grad_y_optional;
  if (fwd_grad_grad_y.initialized())
    fwd_grad_grad_y_optional =
        paddle::make_optional<paddle::Tensor>(fwd_grad_grad_y);

  auto& axis = this->axis_;
  // Prepare Grad function call

  const auto& out_metas = OutputMeta();
  paddle::small_vector<std::vector<paddle::Tensor>, egr::kSlotSmallVectorSize>
      returns(3);
  for (int i = 0; i < 3; ++i) {
    out_metas[i].empty() ? returns[i].resize(1)
                         : returns[i].resize(out_metas[i].size());
  }

  auto* api_output_0 =
      (out_metas[0].empty() || out_metas[0][0].IsStopGradient())
          ? nullptr
          : &returns[0][0];
  auto* api_output_1 =
      (out_metas[1].empty() || out_metas[1][0].IsStopGradient())
          ? nullptr
          : &returns[1][0];
  auto* api_output_2 =
      (out_metas[2].empty() || out_metas[2][0].IsStopGradient())
          ? nullptr
          : &returns[2][0];
  // Runtime check if we need next grad
  bool trace_backward = egr::Controller::Instance().HasGrad() && create_graph;

  // Inplace Check

  bool can_be_inplaced = false;
  if (fwd_grad_grad_x.initialized()) {
    VLOG(10) << fwd_grad_grad_x.name() << "(grad_x_grad) use_count: "
             << fwd_grad_grad_x.impl().use_count();
    if (fwd_grad_grad_x.impl().use_count() == 1 ||
        (fwd_grad_grad_x.impl().use_count() == 2 &&
         fwd_grad_grad_x.impl().get() == grads[0][0].impl().get())) {
      can_be_inplaced = true;
    }
  }
  // Inplace Strategy

  if (trace_backward) {
    VLOG(6) << "No Inplace should happened for wrapped input: "
               "{inplace_grad_input_str}";
  } else {
    if (api_output_2 != nullptr && can_be_inplaced) {
      egr::EagerUtils::HandleViewBetweenInputAndOutput(fwd_grad_grad_x,
                                                       api_output_2);
    }
  }

  VLOG(5) << "Running C++ API: "
          << "multiply_double_grad";
  // Before log info

  if (VLOG_IS_ON(3)) {
    const char* INPUT_PRINT_TEMPLATE = "{ Input: [%s]} ";

    std::string input_str = "";
    std::string output_str = "";
    const char* TENSOR_FWD_GRAD_GRAD_X_TEMPLATE =
        " \n( fwd_grad_grad_x , [%s]), ";
    std::string input_fwd_grad_grad_x_str =
        paddle::string::Sprintf(TENSOR_FWD_GRAD_GRAD_X_TEMPLATE,
                                egr::EagerUtils::TensorStr(fwd_grad_grad_x));
    input_str += input_fwd_grad_grad_x_str;
    const char* TENSOR_FWD_GRAD_GRAD_Y_TEMPLATE =
        " \n( fwd_grad_grad_y , [%s]), ";
    std::string input_fwd_grad_grad_y_str =
        paddle::string::Sprintf(TENSOR_FWD_GRAD_GRAD_Y_TEMPLATE,
                                egr::EagerUtils::TensorStr(fwd_grad_grad_y));
    input_str += input_fwd_grad_grad_y_str;
    const char* TENSOR_X_TEMPLATE = " \n( x , [%s]), ";
    std::string input_x_str = paddle::string::Sprintf(
        TENSOR_X_TEMPLATE, egr::EagerUtils::TensorStr(x));
    input_str += input_x_str;
    const char* TENSOR_Y_TEMPLATE = " \n( y , [%s]), ";
    std::string input_y_str = paddle::string::Sprintf(
        TENSOR_Y_TEMPLATE, egr::EagerUtils::TensorStr(y));
    input_str += input_y_str;
    const char* TENSOR_FWD_GRAD_OUT_TEMPLATE = " \n( fwd_grad_out , [%s]), ";
    std::string input_fwd_grad_out_str = paddle::string::Sprintf(
        TENSOR_FWD_GRAD_OUT_TEMPLATE, egr::EagerUtils::TensorStr(fwd_grad_out));
    input_str += input_fwd_grad_out_str;
    VLOG(3) << paddle::string::Sprintf(INPUT_PRINT_TEMPLATE, input_str);
  }

  // Call grad_api function

  std::string grad_op_name = "multiply_double_grad";
  auto need_skip =
      paddle::prim::StaticCompositeContext::Instance().CheckSkipCompOps(
          grad_op_name);
  if (!need_skip) {
    bool original_global_grad = egr::Controller::Instance().HasGrad();
    if (!create_graph) {
      egr::Controller::Instance().SetHasGrad(create_graph);
    }
    paddle::prim::multiply_double_grad<paddle::Tensor>(x,
                                                       y,
                                                       fwd_grad_out,
                                                       fwd_grad_grad_x_optional,
                                                       fwd_grad_grad_y_optional,
                                                       axis,
                                                       api_output_0,
                                                       api_output_1,
                                                       api_output_2);
    VLOG(4) << "Composite api multiply_double_grad is called ";
    if (!create_graph) {
      egr::Controller::Instance().SetHasGrad(original_global_grad);
    }
  } else {
    paddle::experimental::multiply_double_grad(x,
                                               y,
                                               fwd_grad_out,
                                               fwd_grad_grad_x_optional,
                                               fwd_grad_grad_y_optional,
                                               axis,
                                               api_output_0,
                                               api_output_1,
                                               api_output_2);
    VLOG(4) << "Fused api multiply_double_grad is called";
  }

  // Check NaN and Inf id needed

  if (FLAGS_check_nan_inf) {
    try {
      egr::CheckTensorHasNanOrInf("multiply_double_grad", returns);
    } catch (...) {
      LOG(WARNING) << "There are nan/inf in (multiply_double_grad)";
      auto forward_trace = GetForwardTrace();
      std::cout << forward_trace << std::endl;
      std::rethrow_exception(std::current_exception());
    }
  }

  // Get GradOut autograd_meta

  auto& grad_x = returns[0][0];
  egr::AutogradMeta* grad_x_autograd_meta =
      returns[0][0].initialized() ? egr::EagerUtils::autograd_meta(&grad_x)
                                  : nullptr;
  if (grad_x_autograd_meta) grad_x_autograd_meta->SetStopGradient(false);

  auto& grad_y = returns[1][0];
  egr::AutogradMeta* grad_y_autograd_meta =
      returns[1][0].initialized() ? egr::EagerUtils::autograd_meta(&grad_y)
                                  : nullptr;
  if (grad_y_autograd_meta) grad_y_autograd_meta->SetStopGradient(false);

  auto& grad_grad_out = returns[2][0];
  egr::AutogradMeta* grad_grad_out_autograd_meta =
      returns[2][0].initialized()
          ? egr::EagerUtils::autograd_meta(&grad_grad_out)
          : nullptr;
  if (grad_grad_out_autograd_meta)
    grad_grad_out_autograd_meta->SetStopGradient(false);

  // Create Grad Node

  if (need_skip) {
    if (trace_backward) {
      PADDLE_THROW(common::errors::Unavailable(
          "The Op multiply_double_grad doesn't have any grad"
          "op. If you don't intend calculating higher order"
          "derivatives, please set `create_graph`to False."));
    }
  }
  VLOG(4) << "Finish AD API GRAD: multiply_double_grad";
  VLOG(6) << "gradnode_ptr = " << this;
  // LOG IF DEBUG

  if (VLOG_IS_ON(4)) {
    const char* INPUT_PRINT_TEMPLATE = "{ Input: [%s],  \n Output: [%s] } ";

    std::string input_str = "";
    std::string output_str = "";
    const char* TENSOR_FWD_GRAD_GRAD_X_TEMPLATE =
        " \n( fwd_grad_grad_x , [%s]), ";
    std::string input_fwd_grad_grad_x_str =
        paddle::string::Sprintf(TENSOR_FWD_GRAD_GRAD_X_TEMPLATE,
                                egr::EagerUtils::TensorStr(fwd_grad_grad_x));
    input_str += input_fwd_grad_grad_x_str;
    const char* TENSOR_FWD_GRAD_GRAD_Y_TEMPLATE =
        " \n( fwd_grad_grad_y , [%s]), ";
    std::string input_fwd_grad_grad_y_str =
        paddle::string::Sprintf(TENSOR_FWD_GRAD_GRAD_Y_TEMPLATE,
                                egr::EagerUtils::TensorStr(fwd_grad_grad_y));
    input_str += input_fwd_grad_grad_y_str;
    const char* TENSOR_X_TEMPLATE = " \n( x , [%s]), ";
    std::string input_x_str = paddle::string::Sprintf(
        TENSOR_X_TEMPLATE, egr::EagerUtils::TensorStr(x));
    input_str += input_x_str;
    const char* TENSOR_Y_TEMPLATE = " \n( y , [%s]), ";
    std::string input_y_str = paddle::string::Sprintf(
        TENSOR_Y_TEMPLATE, egr::EagerUtils::TensorStr(y));
    input_str += input_y_str;
    const char* TENSOR_FWD_GRAD_OUT_TEMPLATE = " \n( fwd_grad_out , [%s]), ";
    std::string input_fwd_grad_out_str = paddle::string::Sprintf(
        TENSOR_FWD_GRAD_OUT_TEMPLATE, egr::EagerUtils::TensorStr(fwd_grad_out));
    input_str += input_fwd_grad_out_str;
    const char* TENSOR_GRAD_X_TEMPLATE = " \n ( grad_x , [%s]), ";
    std::string output_grad_x_str = paddle::string::Sprintf(
        TENSOR_GRAD_X_TEMPLATE, egr::EagerUtils::TensorStr(grad_x));
    output_str += output_grad_x_str;
    const char* TENSOR_GRAD_Y_TEMPLATE = " \n ( grad_y , [%s]), ";
    std::string output_grad_y_str = paddle::string::Sprintf(
        TENSOR_GRAD_Y_TEMPLATE, egr::EagerUtils::TensorStr(grad_y));
    output_str += output_grad_y_str;
    const char* TENSOR_GRAD_GRAD_OUT_TEMPLATE = " \n ( grad_grad_out , [%s]), ";
    std::string output_grad_grad_out_str =
        paddle::string::Sprintf(TENSOR_GRAD_GRAD_OUT_TEMPLATE,
                                egr::EagerUtils::TensorStr(grad_grad_out));
    output_str += output_grad_grad_out_str;
    VLOG(4) << paddle::string::Sprintf(
        INPUT_PRINT_TEMPLATE, input_str, output_str);
  }

  // Return
  if (NeedComplexToRealConversion()) HandleComplexGradToRealGrad(&returns);
  return returns;
}

namespace sparse {
paddle::small_vector<std::vector<paddle::Tensor>, egr::kSlotSmallVectorSize>
MultiplyGradNode::operator()(
    paddle::small_vector<std::vector<paddle::Tensor>,
                         egr::kSlotSmallVectorSize>& grads,
    bool create_graph,
    bool is_new_grad) {
  VLOG(3) << "Running AD API GRAD: "
          << "multiply_grad";
  // This 'Local_XXXGradNode' record event is different with
  // 'Global_XXXGradNode' event.
  // * 'Local_XXXGradNode' will only cover execution time of this function.
  // * 'Global_XXXGradNode' will not only cover execution time of this function,
  // but also include gradient
  //    accumulation when the output(s) of corresponding forward OP are shared
  //    by other OP(s), which may have extra accumulation overhead than
  //    'Local_XXXGradNode'.
  phi::RecordEvent node_execution_inner(
      "Local_MultiplyGradNode", phi::TracerEventType::OperatorInner, 1);

  // Fill Zero For GradIn Tensors
  const auto& input_metas = this->InputMeta();
  egr::EagerUtils::FillZeroForEmptyGradInput(&grads[0][0], input_metas[0][0]);

  // Apply Gradient Hooks
  auto hooked_grads = ApplyGradientHooks(grads);

  // Collect GradIn Tensors, Attrs and Recovered TensorWrappers
  auto x = egr::EagerUtils::RecoverTensorWrapper(&this->x_);
  auto y = egr::EagerUtils::RecoverTensorWrapper(&this->y_);
  auto& out_grad = hooked_grads[0][0];
  // Prepare Grad function call

  const auto& out_metas = OutputMeta();
  paddle::small_vector<std::vector<paddle::Tensor>, egr::kSlotSmallVectorSize>
      returns(2);
  for (int i = 0; i < 2; ++i) {
    out_metas[i].empty() ? returns[i].resize(1)
                         : returns[i].resize(out_metas[i].size());
  }

  auto* api_output_0 =
      (out_metas[0].empty() || out_metas[0][0].IsStopGradient())
          ? nullptr
          : &returns[0][0];
  auto* api_output_1 =
      (out_metas[1].empty() || out_metas[1][0].IsStopGradient())
          ? nullptr
          : &returns[1][0];
  // Runtime check if we need next grad
  bool trace_backward = egr::Controller::Instance().HasGrad() && create_graph;

  // Inplace Check

  // Inplace Strategy

  VLOG(5) << "Running C++ API: "
          << "multiply_grad";
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
    const char* TENSOR_Y_TEMPLATE = " \n( y , [%s]), ";
    std::string input_y_str = paddle::string::Sprintf(
        TENSOR_Y_TEMPLATE, egr::EagerUtils::TensorStr(y));
    input_str += input_y_str;
    VLOG(3) << paddle::string::Sprintf(INPUT_PRINT_TEMPLATE, input_str);
  }

  // Call grad_api function

  paddle::experimental::sparse::multiply_grad(
      x, y, out_grad, api_output_0, api_output_1);
  // Check NaN and Inf id needed

  if (FLAGS_check_nan_inf) {
    try {
      egr::CheckTensorHasNanOrInf("multiply_grad", returns);
    } catch (...) {
      LOG(WARNING) << "There are nan/inf in (multiply_grad)";
      auto forward_trace = GetForwardTrace();
      std::cout << forward_trace << std::endl;
      std::rethrow_exception(std::current_exception());
    }
  }

  // Get GradOut autograd_meta

  auto& x_grad = returns[0][0];
  egr::AutogradMeta* x_grad_autograd_meta =
      returns[0][0].initialized() ? egr::EagerUtils::autograd_meta(&x_grad)
                                  : nullptr;
  if (x_grad_autograd_meta) x_grad_autograd_meta->SetStopGradient(false);

  auto& y_grad = returns[1][0];
  egr::AutogradMeta* y_grad_autograd_meta =
      returns[1][0].initialized() ? egr::EagerUtils::autograd_meta(&y_grad)
                                  : nullptr;
  if (y_grad_autograd_meta) y_grad_autograd_meta->SetStopGradient(false);

  // Create Grad Node
  if (trace_backward) {
    PADDLE_THROW(common::errors::Unavailable(
        "The Op multiply_grad doesn't have any grad"
        "op. If you don't intend calculating higher order"
        "derivatives, please set `create_graph`to False."));
  }
  VLOG(4) << "Finish AD API GRAD: multiply_grad";
  VLOG(6) << "gradnode_ptr = " << this;
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
    const char* TENSOR_Y_TEMPLATE = " \n( y , [%s]), ";
    std::string input_y_str = paddle::string::Sprintf(
        TENSOR_Y_TEMPLATE, egr::EagerUtils::TensorStr(y));
    input_str += input_y_str;
    const char* TENSOR_X_GRAD_TEMPLATE = " \n ( x_grad , [%s]), ";
    std::string output_x_grad_str = paddle::string::Sprintf(
        TENSOR_X_GRAD_TEMPLATE, egr::EagerUtils::TensorStr(x_grad));
    output_str += output_x_grad_str;
    const char* TENSOR_Y_GRAD_TEMPLATE = " \n ( y_grad , [%s]), ";
    std::string output_y_grad_str = paddle::string::Sprintf(
        TENSOR_Y_GRAD_TEMPLATE, egr::EagerUtils::TensorStr(y_grad));
    output_str += output_y_grad_str;
    VLOG(6) << "gradnode_ptr = " << this;
    VLOG(4) << paddle::string::Sprintf(
        INPUT_PRINT_TEMPLATE, input_str, output_str);
  }

  // Return
  if (NeedComplexToRealConversion()) HandleComplexGradToRealGrad(&returns);
  return returns;
}

}  // namespace sparse
