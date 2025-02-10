/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/unused_var_check.h"

#include <glog/logging.h>

#include <string>

#include "paddle/common/flags.h"
#include "paddle/fluid/framework/no_need_buffer_vars_inference.h"
#include "paddle/fluid/framework/op_info.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/enforce.h"
PHI_DEFINE_EXPORTED_bool(
    enable_unused_var_check,
    false,
    "Checking whether operator contains unused inputs, "
    "especially for grad operator. It should be in unittest.");

namespace paddle::framework {

std::unordered_set<std::string> *GetThreadLocalUsedVarNameSet() {
  thread_local std::unordered_set<std::string> used_var_name_set;
  return &used_var_name_set;
}

void LogVarUsageIfUnusedVarCheckEnabled(const std::string &name) {
  if (FLAGS_enable_unused_var_check) {
    VLOG(6) << "Variable used:" << name;
    GetThreadLocalUsedVarNameSet()->insert(name);
  }
}

static const std::unordered_set<std::string> &GetOpWithUnusedVarAllowSet() {
  // NOTE(zhiqiu): Currently, there are some operators which involves unused
  // inputs and cannot be removed from the allow_list below.
  // They can be mainly divided into four categories:
  // 0: the inputs of which are only used in if branch, or used in cuda kernel
  // but not in cpu kernel; 1: the inputs of which are used to indicate dtype of
  // outputs; 2: the inputs of which are used in fused operators. The category
  // number is presented in the comments after each operator.
  // Use pointer here for safe static deinitialization
  static auto *allow_set = new std::unordered_set<std::string>({
      // called once
      "batch_norm",                         // 0
      "batch_norm_grad",                    // 0
      "sync_batch_norm",                    // 0
      "sync_batch_norm_grad",               // 0
      "dgc_momentum",                       // 0
      "fake_quantize_range_abs_max",        // 0
      "rmsprop",                            // 0
      "sequence_conv_grad",                 // 0
      "roi_perspective_transform_grad",     // 0
      "fill_zeros_like",                    // 1
      "fill_any_like",                      // 1
      "nce_grad",                           // 1
      "precision_recall",                   // 1
      "fusion_seqpool_cvm_concat",          // 2
      "fused_batch_norm_act",               // 2
      "fused_batch_norm_act_grad",          // 2
      "data_norm",                          // 0
      "data_norm_grad",                     // 0
      "update_loss_scaling",                // 0
      "fused_embedding_eltwise_layernorm",  // 0
      "trunc_grad",                         // 1
  });
  return *allow_set;
}

void CheckUnusedVar(const OperatorBase &op, const Scope &scope) {
  // skip op in allow list.
  if (GetOpWithUnusedVarAllowSet().count(op.Type()) != 0) {
    return;
  }
  auto *used_set = GetThreadLocalUsedVarNameSet();
  std::vector<std::string> unused_input_var_names;
  auto &inferer = op.Info().NoNeedBufferVarsInferer();
  std::unordered_set<std::string> no_need_buffer_ins = {};
  if (inferer) {
    no_need_buffer_ins = inferer(op.Inputs(), op.Outputs(), op.Attrs());
  }

  for (auto &pair : op.Inputs()) {
    // skip no need buffer vars declared
    if (no_need_buffer_ins.count(pair.first) != 0) {
      VLOG(6) << op.Type() << " " << pair.first;
      continue;
    }
    if (used_set->count(pair.first) == 0) {
      for (auto &in_var_name : pair.second) {
        auto *in_var = scope.FindVar(in_var_name);
        if (in_var != nullptr && in_var->IsInitialized()) {
          auto *tensor = &in_var->Get<phi::DenseTensor>();
          if (tensor != nullptr && tensor->IsInitialized()) {
            unused_input_var_names.emplace_back(pair.first);
            break;
          }
        }
      }
    }
  }
  if (!unused_input_var_names.empty()) {
    std::string err_msg =
        "Operator " + op.Type() + " has input(s) not unused: ";
    for (auto &in_var_name : unused_input_var_names) {
      err_msg += in_var_name;
      err_msg += ", ";
    }
    err_msg +=
        "please make sure it(they) is(are) needed. If not, remove it(them) "
        "from inputs of the operator; if yes, register "
        "NoNeedBufferVarsInference or add "
        "the operator to "
        "allow list in unused_var_check.cc. See more details at "
        "[https://github.com/PaddlePaddle/Paddle/wiki/"
        "OP-Should-Not-Have-Unused-Input]";
    PADDLE_ENFORCE_EQ(unused_input_var_names.size(),
                      0,
                      common::errors::PermissionDenied(
                          "Unused input variables check failed: %s", err_msg));
  }
}

}  // namespace paddle::framework
