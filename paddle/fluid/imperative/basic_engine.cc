// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/imperative/basic_engine.h"

#include <algorithm>
#include <memory>
#include <queue>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "paddle/fluid/imperative/gradient_accumulator.h"
#include "paddle/fluid/imperative/layer.h"
#include "paddle/fluid/imperative/op_base.h"
#include "paddle/fluid/imperative/tracer.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/platform/profiler.h"

namespace paddle {
namespace imperative {

static std::string GetVariableWrapperStr(const VariableWrapper* v) {
  std::stringstream ss;
  ss << v;
  if (v && v->Var().IsType<framework::LoDTensor>() &&
      v->Var().Get<framework::LoDTensor>().IsInitialized()) {
    auto& tensor = v->Var().Get<framework::LoDTensor>();
    framework::Tensor cpu_tensor;
    framework::TensorCopySync(tensor, platform::CPUPlace(), &cpu_tensor);
    if (cpu_tensor.type() == framework::proto::VarType::FP32) {
      auto* p = cpu_tensor.data<float>();
      int64_t numel = cpu_tensor.numel();
      ss << "[";
      for (int64_t i = 0; i < numel; ++i) {
        ss << p[i] << ", ";
      }
      ss << "]";
    }
  }
  return ss.str();
}

BasicEngine::BasicEngine(VarBase* var,
                         const detail::BackwardStrategy& strategy) {
  backward_strategy_ = strategy;
  init_node_ = var->GradVarBase()->GradNode();
  var->GradVarBase()->ClearGradNode();

  if (init_node_ == nullptr || var->OverridedStopGradient()) {
    VLOG(3) << "Skip auto grad since there is no grad op for var or loss is "
               "stop_gradient=True: "
            << var->Name();
    return;
  }

  VLOG(3) << "start backward";

  PADDLE_ENFORCE_EQ(var->HasGradVar(), true,
                    "Grad variable not exist for variable %s", var->Name());

  auto& fwd_var = var->Var().Get<framework::LoDTensor>();
  auto* grad_var =
      var->GradVarBase()->MutableVar()->GetMutable<framework::LoDTensor>();
  VLOG(6) << "init loss grad:" << var->GradVarBase()->Name()
          << " as stop_gradient false";
  var->GradVarBase()->InnerSetOverridedStopGradient(false);
  auto* dev_ctx = platform::DeviceContextPool::Instance().Get(fwd_var.place());
  grad_var->Resize(fwd_var.dims());
  grad_var->mutable_data(fwd_var.place(), fwd_var.type());
  operators::math::set_constant(*dev_ctx, grad_var, 1.0);
}

void BasicEngine::CheckBackwardInputs(const OpBase& op,
                                      const platform::Place& place) {
  for (auto& pair : op.GetInsMap()) {
    if (!pair.second.IsGrad()) {
      continue;
    }

    for (auto& var : pair.second) {
      if (!var) {
        continue;
      }

      auto* inner_var = var->MutableVar();
      framework::Tensor* tensor = nullptr;
      if (!inner_var->IsInitialized() ||
          inner_var->IsType<framework::LoDTensor>()) {
        tensor = inner_var->GetMutable<framework::LoDTensor>();
      }

      if (tensor && !tensor->IsInitialized()) {
        // if grad var has OverridedStopGradient skip this Op
        VLOG(6) << "Set ungenerated Grad: " << var->Name() << " as zero";
        auto* dev_ctx = platform::DeviceContextPool::Instance().Get(place);
        tensor->mutable_data(place, var->DataType());
        operators::math::set_constant(*dev_ctx, tensor, 0.0);
      }
    }
  }
}

void BasicEngine::PrepareGradAccumulators(const GradOpNode& node) {
  for (auto& op : node) {
    for (const auto& pair : op.GetOutsMap()) {
      if (!pair.second.IsGrad()) {
        continue;
      }

      for (const auto& var : pair.second) {
        if (!var) continue;

        auto& accumulator = accumulators_[var.get()];
        if (!accumulator) {
          if (backward_strategy_.sorted_sum_gradient_) {
            accumulator.reset(new SortedGradientAccumulator(var.get()));
          } else {
            accumulator.reset(new EagerGradientAccumulator(var.get()));
          }
        }

        accumulator->IncreaseRefCnt();

        VLOG(1) << "Prepare to acccumulate variable grad " << var->Name() << "("
                << var.get() << ")  with reference count "
                << accumulator->RefCnt();
      }
    }
  }
}

void BasicEngine::PrepareDeps() {
  PADDLE_ENFORCE_EQ(node_deps_.empty(), true,
                    "Op deps must be initialized here");
  PADDLE_ENFORCE_EQ(accumulators_.empty(), true,
                    "Accumulators must be initialized here");

  std::queue<GradOpNode*> q;
  std::unordered_set<GradOpNode*> visited;

  q.push(init_node_.get());
  visited.insert(init_node_.get());

  while (!q.empty()) {
    auto* cur_node = q.front();
    q.pop();

    for (auto& cur_op : *cur_node) {
      PADDLE_ENFORCE_NE(
          cur_op.GetInsMap().empty() && cur_op.GetOutsMap().empty(), true,
          platform::errors::NotFound(
              "Inputs and outputs of %s do not exist. "
              "This may be because you call \"backward()\" twice for the same "
              "subgraph. Please try to call \"stop_gradient = True\" or "
              "\"detach()\" if you use some same vars between two "
              "\"backward()\" "
              "calls.",
              cur_op.Type()));
    }

    PrepareGradAccumulators(*cur_node);

    const auto& grad_pending_nodes = cur_node->GradPendingNodes();
    for (auto& grad_pending_node : grad_pending_nodes) {
      PADDLE_ENFORCE_NOT_NULL(grad_pending_node);
      ++node_deps_[grad_pending_node.get()];
      if (visited.count(grad_pending_node.get()) == 0) {
        visited.insert(grad_pending_node.get());
        q.push(grad_pending_node.get());
      }
    }
  }
}

void BasicEngine::SumGradient(const OpBase& op,
                              std::shared_ptr<VariableWrapper> src,
                              VariableWrapper* dst) {
  auto iter = accumulators_.find(dst);

  PADDLE_ENFORCE_EQ(iter != accumulators_.end(), true,
                    "Cannot find gradient of variable %s", dst->Name());
  iter->second->Add(std::move(src), op.id());
}

void BasicEngine::Execute() {
  if (init_node_ == nullptr) {
    return;
  }

  PrepareDeps();
  // Start execute Computation graph
  std::queue<std::shared_ptr<GradOpNode>> q;
  q.push(std::move(init_node_));

  size_t op_num = 0;

  while (!q.empty()) {
    auto shared_cur_node = std::move(q.front());
    q.pop();

    for (auto& cur_op : *shared_cur_node) {
      ++op_num;

      // CheckBackWardInput
      CheckBackwardInputs(cur_op, cur_op.place());

      // Step 1: Run Backward
      auto& bwd_ins = cur_op.GetInsMap();
      auto& bwd_outs = cur_op.GetOutsMap();

      NameVarMap<VariableWrapper> tmp_outs(bwd_outs);
      // 1. construct the output map 2. replace the element in the map
      // A var may be coresponding to several grad var in one op
      for (auto it = tmp_outs.begin(); it != tmp_outs.end(); ++it) {
        if (!it->second.IsGrad()) {
          continue;
        }
        for (size_t i = 0; i < it->second.size(); ++i) {
          auto var = it->second[i];
          if (!var) {
            continue;
          }
          auto tmp_var = std::make_shared<VariableWrapper>(var->Name());
          it->second[i] = tmp_var;
          need_accu_var_list_.emplace_back(var.get(), std::move(tmp_var));
        }
      }

      {
        VLOG(3) << "Start to execute grad op " << cur_op.Type();
        OpBase::Run(cur_op.InnerOp(), bwd_ins, tmp_outs, cur_op.Attrs(),
                    cur_op.place());
      }

      // Step 2: Sum Gradient
      for (auto& pair : need_accu_var_list_) {
        SumGradient(cur_op, pair.second, pair.first);
        VLOG(1) << "Sum gradient of variable " << pair.first->Name()
                << " after op " << cur_op.Type() << " : " << pair.second.get()
                << " -> " << pair.first << " "
                << GetVariableWrapperStr(pair.first);
      }

      need_accu_var_list_.clear();

      VLOG(3) << "Remove op after op " << cur_op.Type() << " runs";
      cur_op.ClearBackwardTrace();
    }

    // Step 3: Collect ready ops

    for (auto& grad_pending_node : shared_cur_node->GradPendingNodes()) {
      PADDLE_ENFORCE_NOT_NULL(grad_pending_node);
      auto iter = node_deps_.find(grad_pending_node.get());
      if (iter == node_deps_.end()) {
        continue;
      }

      if (--(iter->second) == 0) {
        q.push(grad_pending_node);
      }
    }
  }
  Clear();

  VLOG(1) << "Backward op number: " << op_num;
}

void BasicEngine::Clear() {
  init_node_.reset();
  node_deps_.clear();
  accumulators_.clear();
  need_accu_var_list_.clear();
}

}  // namespace imperative
}  // namespace paddle
