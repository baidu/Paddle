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
#include "paddle/phi/api/all.h"
#include "paddle/phi/api/lib/api_custom_impl.h"
#include "paddle/phi/core/platform/profiler/event_tracing.h"

COMMON_DECLARE_bool(check_nan_inf);

paddle::small_vector<std::vector<paddle::Tensor>, egr::kSlotSmallVectorSize>
AddNGradNodeFinal::operator()(
    paddle::small_vector<std::vector<paddle::Tensor>, egr::kSlotSmallVectorSize>
        &grads,
    bool create_graph,
    bool is_new_grad) {
  // Fill Zero For GradIn Tensors

  // This 'Local_XXXGradNode' record event is different with
  // 'Global_XXXGradNode' event.
  // * 'Local_XXXGradNode' will only cover execution time of this function.
  // * 'Global_XXXGradNode' will not only cover execution time of this function,
  // but also include gradient
  //    accumulation when the output(s) of corresponding forward OP are shared
  //    by other OP(s), which may have extra accumulation overhead than
  //    'Local_XXXGradNode'.
  phi::RecordEvent node_execution_inner(
      "Local_AddNGradNodeFinal", phi::TracerEventType::OperatorInner, 1);

  // Apply Gradient Hooks
  auto hooked_grads = ApplyGradientHooks(grads);

  // Collect GradIn Tensors, Attrs and Recovered TensorWrappers
  auto x = egr::EagerUtils::RecoverTensorWrapper(&this->x_);
  auto &out_grad = hooked_grads[0][0];
  // Prepare Grad function call

  const auto &out_metas = OutputMeta();
  paddle::small_vector<std::vector<paddle::Tensor>, egr::kSlotSmallVectorSize>
      returns(1);
  for (int i = 0; i < 1; ++i) {
    out_metas[i].empty() ? returns[i].resize(1)
                         : returns[i].resize(out_metas[i].size());
  }

  std::vector<paddle::Tensor *> api_output_0;
  api_output_0.reserve(returns[0].size());
  for (size_t i = 0; i < returns[0].size(); ++i) {
    if (out_metas[0].empty() || out_metas[0][i].IsStopGradient()) {
      api_output_0.push_back(nullptr);
    } else {
      api_output_0.push_back(&returns[0][i]);
    }
  }
  // Call grad_api function
  VLOG(3) << "Final State Running: AddNGradNodeFinal";

  // dygraph function
  for (auto &item : returns[0]) {
    item = ::scale_ad_func(out_grad, phi::Scalar(1.0), 0.0, true);
  }

  // Check NaN and Inf id needed
  if (FLAGS_check_nan_inf) {
    egr::CheckTensorHasNanOrInf("add_n_grad", returns);
  }

  if (VLOG_IS_ON(4)) {
    const char *INPUT_PRINT_TEMPLATE = "{ Input: [%s],  \n Output: [%s] } ";
    std::string input_str = "";
    std::string output_str = "";

    const char *TENSOR_INPUT_TEMPLATE = " \n( x , [%s]), ";
    std::string input_x_str = paddle::string::Sprintf(
        TENSOR_INPUT_TEMPLATE, egr::EagerUtils::TensorStr(x));
    input_str += input_x_str;

    const char *TENSOR_OUT_GRAD_TEMPLATE = " \n( out_grad , [%s]), ";
    std::string input_out_grad_str = paddle::string::Sprintf(
        TENSOR_OUT_GRAD_TEMPLATE, egr::EagerUtils::TensorStr(out_grad));
    input_str += input_out_grad_str;

    const char *TENSOR_OUTPUT_TEMPLATE = " \n ( returns , [%s]), ";
    std::string output_returns_str = paddle::string::Sprintf(
        TENSOR_OUTPUT_TEMPLATE, egr::EagerUtils::TensorStr(returns[0][0]));
    output_str += output_returns_str;

    VLOG(6) << "gradnode_ptr = " << this;
    VLOG(4) << paddle::string::Sprintf(
        INPUT_PRINT_TEMPLATE, input_str, output_str);
  }

  if (HasNodePostHook()) {
    returns = ApplyNodePostHooks(returns, hooked_grads);
  }

  if (NeedComplexToRealConversion()) HandleComplexGradToRealGrad(&returns);
  return returns;
}
