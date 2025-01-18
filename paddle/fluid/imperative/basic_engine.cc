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

#include "paddle/common/flags.h"
#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/imperative/gradient_accumulator.h"
#include "paddle/fluid/imperative/layer.h"
#include "paddle/fluid/imperative/op_base.h"
#include "paddle/fluid/imperative/tracer.h"
#include "paddle/fluid/platform/profiler.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/kernels/autotune/switch_autotune.h"
#include "paddle/phi/kernels/funcs/math_function.h"

COMMON_DECLARE_bool(sort_sum_gradient);

namespace paddle {
namespace imperative {

void BasicEngine::Init(
    const std::vector<std::shared_ptr<VarBase>>& tensors,
    const std::vector<std::shared_ptr<VarBase>>& grad_tensors,
    bool retain_graph) {
  retain_graph_ = retain_graph;

  PADDLE_ENFORCE_EQ(
      tensors.size(),
      grad_tensors.size(),
      platform::errors::Unavailable(
          "The size of tensors do not equal the size of grad_tensors,"
          "the size of tensors is %s, but the size of grad_tensors is %s.",
          tensors.size(),
          grad_tensors.size()));

  PADDLE_ENFORCE_EQ(accumulators_.empty(),
                    true,
                    platform::errors::AlreadyExists(
                        "Accumulators are not empty before preparing it for "
                        "backward network execution."));
  PADDLE_ENFORCE_EQ(accumulators_with_grad_node_.empty(),
                    true,
                    platform::errors::AlreadyExists(
                        "Accumulators with grad_node as the key are not empty "
                        "before preparing it for backward network execution."));

  for (size_t i = 0; i < tensors.size(); ++i) {
    auto var = tensors[i];
    auto grad_tensor = grad_tensors[i];

    auto init_node = var->GradVarBase()->GradNode();

    PADDLE_ENFORCE_EQ(
        var->GradVarBase()->GraphIsFreed(),
        false,
        platform::errors::Unavailable(
            "%s trying to backward through the same graph a second "
            "time, but this graph have already been freed. Please "
            "specify Tensor.backward(retain_graph=True) when "
            "calling backward at the first time.",
            var->Name()));

    if (!retain_graph) {
      VLOG(5) << "Clear the auto-grad graph from grad var " << var->Name()
              << " because of retain_graph=False when calling backward";
      var->GradVarBase()->SetGraphIsFreed(true);
    }

    if (init_node == nullptr || var->OverriddenStopGradient()) {
      VLOG(3) << "Skip auto grad since there is no grad op for var or loss is "
                 "stop_gradient=True: "
              << var->Name();
      continue;
    }

    VLOG(3) << "Init node of backward";

    PADDLE_ENFORCE_EQ(
        var->HasGradVar(),
        true,
        platform::errors::NotFound("Tensor %s has no gradient", var->Name()));

    auto& fwd_var = var->Var().Get<phi::DenseTensor>();
    auto* grad_var =
        var->GradVarBase()->MutableVar()->GetMutable<phi::DenseTensor>();
    VLOG(6) << "init loss grad:" << var->GradVarBase()->Name()
            << " as stop_gradient false";
    var->GradVarBase()->InnerSetOverriddenStopGradient(false);
    auto* dev_ctx =
        platform::DeviceContextPool::Instance().Get(fwd_var.place());
    if (grad_tensor == nullptr) {
      grad_var->Resize(fwd_var.dims());
      grad_var->mutable_data(fwd_var.place(), fwd_var.type());
      phi::funcs::set_constant(*dev_ctx, grad_var, 1.0f);
    } else {
      paddle::framework::TensorCopy(grad_tensor->Var().Get<phi::DenseTensor>(),
                                    fwd_var.place(),
                                    *dev_ctx,
                                    grad_var);
    }

    VariableWrapper* init_grad_var = var->GradVarBase()->SharedVar().get();
    auto& accumulator =
        accumulators_with_grad_node_[init_grad_var->GetGradNode()]
                                    [init_grad_var];
    if (!accumulator) {
      if (FLAGS_sort_sum_gradient) {
        accumulator =
            std::make_unique<SortedGradientAccumulator>(init_grad_var);
      } else {
        accumulator = std::make_unique<EagerGradientAccumulator>(init_grad_var);
      }
    }
    accumulator->IncreaseRefCnt();
    accumulator->IncreaseCurCnt();

    init_nodes_.push_back(init_node);
  }
}

void BasicEngine::CheckBackwardInputs(const OpBase& op) {
  for (auto& pair : op.GetInsMap()) {
    if (!pair.second.IsGrad()) {
      continue;
    }

    for (auto& var : pair.second) {
      if (!var) {
        continue;
      }

      auto* inner_var = var->MutableVar();
      phi::DenseTensor* tensor = nullptr;
      if (!inner_var->IsInitialized() ||
          inner_var->IsType<phi::DenseTensor>()) {
        tensor = inner_var->GetMutable<phi::DenseTensor>();
      }

      if (tensor && !tensor->IsInitialized()) {
        auto* dev_ctx = platform::DeviceContextPool::Instance().Get(op.place());
        // NOTE(zhiqiu): since grad variable is ungenerated, so the dtype is not
        // correct. var->DataType() returns the default dtype, which is float32.
        // Here, we use the type of the corresponding forward datatype.

        tensor->mutable_data(
            op.place(), framework::TransToPhiDataType(var->ForwardDataType()));
        VLOG(6) << "Set ungenerated Grad: " << var->Name()
                << " as zero with dtype "
                << framework::DataTypeToString(var->ForwardDataType());
        phi::funcs::set_constant(*dev_ctx, tensor, 0.0f);
      }
    }
  }
}

void BasicEngine::PrepareGradAccumulators(
    const OpBase& op,
    const std::vector<std::shared_ptr<GradOpNode>>& grad_pending_nodes) {
  for (const auto& pair : op.GetOutsMap()) {
    if (!pair.second.IsGrad()) {
      continue;
    }

    for (const auto& var : pair.second) {
      if (!var) continue;

      bool find_grad_node_of_var = false;
      if (!grad_pending_nodes.empty()) {
        // Because Inplace op overwrites the grad_node of the input grad_var. So
        // only the information of grad_pending_node can be used to find the
        // grad_node of grad_var.
        for (auto& grad_pending_node : grad_pending_nodes) {
          PADDLE_ENFORCE_NOT_NULL(
              grad_pending_node,
              platform::errors::NotFound("Grad pending node is nullptr."));
          for (auto& grad_pending_op : *grad_pending_node) {
            VLOG(6) << "Determine whether var (" << var->Name()
                    << ") is the input var of grad_pending_op ("
                    << grad_pending_op.Type() << ").";
            grad_pending_op.EnforceHasInOut();
            for (const auto& grad_pending_op_ins_pair :
                 grad_pending_op.GetInsMap()) {
              if (!grad_pending_op_ins_pair.second.IsGrad()) {
                continue;
              }
              for (const auto& pending_in_var :
                   grad_pending_op_ins_pair.second) {
                if (var == pending_in_var) {
                  VLOG(6) << "Var (" << var->Name()
                          << ") is the input var of grad_pending_op ("
                          << grad_pending_op.Type() << ").";
                  find_grad_node_of_var = true;
                  break;
                }
              }
              if (find_grad_node_of_var) {
                break;
              }
            }
          }

          if (find_grad_node_of_var) {
            auto& accumulator =
                accumulators_with_grad_node_[grad_pending_node][var.get()];

            if (!accumulator) {
              if (FLAGS_sort_sum_gradient) {
                accumulator =
                    std::make_unique<SortedGradientAccumulator>(var.get());
              } else {
                accumulator =
                    std::make_unique<EagerGradientAccumulator>(var.get());
              }
            }

            accumulator->IncreaseRefCnt();

            VLOG(3) << "Prepare to accumulate variable grad " << var->Name()
                    << "(" << var.get()
                    << ") that has grad node with reference count "
                    << accumulator->RefCnt();
            break;
          }
        }
        if (!find_grad_node_of_var) {
          // Special case: `set_value` is inplace op, and it can change
          // the var with `stop_gradient=True` to the var with
          // `stop_gradient=False `.
          // This inplace var has grad_node (the inplace op), but it
          // isn't the input of grad_pending_op.
          VLOG(6) << "No grad node corresponding to grad Tensor ("
                  << var->Name() << ") was found.";
        }
      }

      if (grad_pending_nodes.empty() || !find_grad_node_of_var) {
        auto& accumulator = accumulators_[var.get()];
        if (!accumulator) {
          if (FLAGS_sort_sum_gradient) {
            accumulator =
                std::make_unique<SortedGradientAccumulator>(var.get());
          } else {
            accumulator = std::make_unique<EagerGradientAccumulator>(var.get());
          }
        }

        accumulator->IncreaseRefCnt();

        VLOG(3) << "Prepare to accumulate variable grad " << var->Name() << "("
                << var.get()
                << ") that don't have grad node  with reference count "
                << accumulator->RefCnt();
      }
    }
  }
}

void BasicEngine::PrepareDeps() {
  PADDLE_ENFORCE_EQ(
      node_deps_.empty(),
      true,
      platform::errors::AlreadyExists("Op deps are not empty before preparing "
                                      "it for backward network execution."));

  std::queue<GradOpNode*> q;
  std::unordered_set<GradOpNode*> visited;

  for (auto& init_node : init_nodes_) {
    q.push(init_node.get());
    visited.insert(init_node.get());
  }

  while (!q.empty()) {
    auto* cur_node = q.front();
    q.pop();

    const auto& grad_pending_nodes = cur_node->GradPendingNodes();

    for (auto& cur_op : *cur_node) {
      cur_op.EnforceHasInOut();
      PrepareGradAccumulators(cur_op, grad_pending_nodes);
    }

    for (auto& grad_pending_node : grad_pending_nodes) {
      PADDLE_ENFORCE_NOT_NULL(
          grad_pending_node,
          platform::errors::NotFound("Grad pending node is nullptr."));
      ++node_deps_[grad_pending_node.get()];
      if (visited.count(grad_pending_node.get()) == 0) {
        visited.insert(grad_pending_node.get());
        q.push(grad_pending_node.get());
      }
    }
  }
}

static std::shared_ptr<NameVarMap<VariableWrapper>> CallGradientHooks(
    const NameVarMap<VariableWrapper>& bwd_ins, const std::string& op_type) {
  std::shared_ptr<NameVarMap<VariableWrapper>> tmp_ins_ptr = nullptr;
  for (const auto& pair : bwd_ins) {
    for (size_t i = 0; i < pair.second.size(); ++i) {
      auto& var = pair.second[i];
      if (var->HasVariableWrapperHook()) {
        if (tmp_ins_ptr == nullptr) {
          tmp_ins_ptr = std::make_shared<NameVarMap<VariableWrapper>>(bwd_ins);
        }
        VLOG(3) << "Call " << var->GetVariableWrapperHooks().size()
                << " hooks of " << op_type << "'s input `" << pair.first
                << "`'s var `" << var->Name() << "`.";
        auto tmp_var = var;
        for (const auto& hook_pair : var->GetVariableWrapperHooks()) {
          tmp_var = (*hook_pair.second)(tmp_var);
          CheckVar(var, tmp_var);
        }
        (*tmp_ins_ptr)[pair.first][i] = tmp_var;
      }
    }
  }
  return tmp_ins_ptr;
}

static bool IsInputCanInplace(const std::shared_ptr<VariableWrapper>& var) {
  auto* inner_var = var->MutableVar();
  if (inner_var->IsInitialized() && inner_var->IsType<phi::DenseTensor>()) {
    auto tensor = inner_var->GetMutable<phi::DenseTensor>();
    if (tensor->IsInitialized()) {
      return true;
    }
  }
  return false;
}

static void PerformBackwardInplace(const std::string& op_type,
                                   const NameVarMap<VariableWrapper>& ins,
                                   NameVarMap<VariableWrapper>* outs) {
  auto& infer_inplace =
      paddle::framework::OpInfoMap::Instance().Get(op_type).infer_inplace_;

  if (infer_inplace) {
    auto in_to_outs = infer_inplace(true);
    for (auto& pair : in_to_outs) {
      phi::DenseTensor *in_tensor = nullptr, *out_tensor = nullptr;
      for (auto& p : ins) {
        if (p.first == pair.first) {
          // has at least one var
          if (!p.second.empty() && p.second[0]) {
            auto& in_var = p.second[0];
            VLOG(10) << p.first << " use_count: " << in_var.use_count();
            // the refcount of var to be inplaced should be 1
            if (in_var.use_count() == 1) {
              if (IsInputCanInplace(in_var)) {
                in_tensor =
                    in_var->MutableVar()->GetMutable<phi::DenseTensor>();
              }
            }
          }
        }
      }
      if (!in_tensor) {
        continue;
      }
      for (auto& p : *outs) {
        if (p.first == pair.second) {
          if (!p.second.empty() && p.second[0]) {
            auto& out_var = p.second[0];
            if (out_var->Type() == framework::proto::VarType::LOD_TENSOR) {
              out_tensor =
                  out_var->MutableVar()->GetMutable<phi::DenseTensor>();
            }
          }
        }
      }
      if (!out_tensor) {
        continue;
      }
      out_tensor->ShareBufferWith(*in_tensor);
      out_tensor->Resize(in_tensor->dims());
      VLOG(4) << "Inplace performed in op " << op_type << ": " << pair.second
              << " -> " << pair.first;
    }
  }
}

void BasicEngine::Execute() {
  platform::RecordEvent backward_record_event(
      "backward", platform::TracerEventType::UserDefined, 1);

  if (init_nodes_.empty()) {
    return;
  }

  PrepareDeps();
  // Start execute Computation graph
  std::queue<std::shared_ptr<GradOpNode>> q;
  for (auto& init_node : init_nodes_) {
    if (node_deps_[init_node.get()] == 0) {
      q.push(std::move(init_node));
    }
  }

  size_t op_num = 0;

  while (!q.empty()) {
    auto shared_cur_node = std::move(q.front());
    q.pop();

    auto& inplace_grad_name_map = shared_cur_node->InplaceGradNameMap();

    for (auto& cur_op : *shared_cur_node) {
      platform::RecordEvent op_type_record_event(
          cur_op.Type() + " grad_node", platform::TracerEventType::Operator, 1);

      ++op_num;

      // CheckBackWardInput
      CheckBackwardInputs(cur_op);

      // Step 1: Run Backward OP
      auto& bwd_ins = cur_op.GetInsMap();
      auto& bwd_outs = cur_op.GetOutsMap();

      /**
       * [ Why need temporary outputs here? ]
       *
       * - construct the temp output map, avoid to disrupt graph
       * - replace the element in the map by temp var, because a
       *   var may be corresponding to several grad var in one op
       */
      NameVarMap<VariableWrapper> tmp_outs(bwd_outs);

      for (auto& pair : tmp_outs) {
        if (!pair.second.IsGrad()) {
          continue;
        }

        for (auto& var : pair.second) {
          if (!var) {
            continue;
          }

          const auto& grad_pending_nodes = shared_cur_node->GradPendingNodes();
          std::unordered_map<VariableWrapper*,
                             std::unique_ptr<GradientAccumulator>>::iterator
              iter;
          bool flag_find_grad = false;
          if (!grad_pending_nodes.empty()) {
            VLOG(10) << "Find gradient of var (" << var->Name()
                     << ") with grad_node.";
            for (auto& grad_pending_node : grad_pending_nodes) {
              const auto& iter_grad_node =
                  accumulators_with_grad_node_.find(grad_pending_node);
              if (iter_grad_node != accumulators_with_grad_node_.end()) {
                iter = iter_grad_node->second.find(var.get());
                if (iter != iter_grad_node->second.end()) {
                  flag_find_grad = true;
                  break;
                }
              }
            }
            if (!flag_find_grad) {
              VLOG(6) << "Cannot find gradient of variable " << var->Name()
                      << " in accumulators_with_grad_node_";
            }
          }
          if (grad_pending_nodes.empty() || !flag_find_grad) {
            VLOG(10) << "Find gradient of var (" << var->Name()
                     << ") with no grad_node.";
            iter = accumulators_.find(var.get());
            PADDLE_ENFORCE_EQ(
                iter != accumulators_.end(),
                true,
                platform::errors::NotFound(
                    "Cannot find gradient of variable %s", var->Name()));
          }

          // leaf_accumulators_ : hooks and accumulate-grad for leaf tensor,
          // it should be orderly and not repeated.
          if (var->IsLeafGrad()) {
            if (std::find(leaf_accumulators_.begin(),
                          leaf_accumulators_.end(),
                          iter->second.get()) == leaf_accumulators_.end()) {
              leaf_accumulators_.push_back(iter->second.get());
            }

            if (iter->second->HasInnerVar()) {
              var = iter->second->InnerVar();
            }
          }

          if (var->OverriddenStopGradient() || iter->second->RefCnt() > 1) {
            auto tmp_var = std::make_shared<VariableWrapper>(var->Name());
            tmp_var->SetType(var->Type());
            tmp_var->SetForwardDataType(var->ForwardDataType());
            var = tmp_var;
            need_accu_var_list_.emplace_back(iter->second.get(), var);
            VLOG(10) << "create temporary var of " << var->Name()
                     << " for sum gradient within this graph!";
          } else if (!inplace_grad_name_map.empty() &&
                     inplace_grad_name_map.count(pair.first) &&
                     bwd_ins.count(inplace_grad_name_map.at(pair.first))) {
            // When calculate Inplace grad op, create a new output var.
            // If a tmp var has been created, there is no need to create it
            // again.
            for (auto& in_var :
                 bwd_ins.at(inplace_grad_name_map.at(pair.first))) {
              if (in_var == var) {
                auto tmp_var = std::make_shared<VariableWrapper>(var->Name());
                tmp_var->SetType(var->Type());
                tmp_var->SetForwardDataType(var->ForwardDataType());
                inplace_output_grad_var_list_.emplace_back(var, tmp_var);
                var = tmp_var;
                VLOG(10) << "Inplace grad op does not use the Inplace "
                            "strategy, a temporary output var ("
                         << var->Name() << ") will be created.";
                break;
              }
            }
          }
        }
      }

      VLOG(4) << "Check whether there is any inplace operation affecting "
                 "gradient calculation.";
      for (auto& pair : bwd_ins) {
        for (auto& var_wrapper : pair.second) {
          auto wrapper_version_snapshot = var_wrapper->InplaceVersionSnapshot();
          auto tensor_version =
              var_wrapper->MutableVar()->CurrentInplaceVersion();
          PADDLE_ENFORCE_EQ(
              tensor_version,
              wrapper_version_snapshot,
              platform::errors::PermissionDenied(
                  "Tensor '%s' used in gradient computation in grad op '%s' "
                  "has been "
                  "modified by an inplace operation. "
                  "Its version is %s but the expected version is %s. "
                  "Please fix your code to void calling an inplace operator "
                  "after using the Tensor which will used in gradient "
                  "computation.",
                  var_wrapper->Name(),
                  cur_op.Type(),
                  tensor_version,
                  wrapper_version_snapshot));

          VLOG(6) << " The version of Tensor '" << var_wrapper->Name()
                  << "' is [ " << wrapper_version_snapshot << " ]";
        }
      }

      /**
       * [ Why need temporary inputs here? ]
       *
       * - Hook execution should not change original input tensor.
       *   User can register hook for Tensor's gradient, It is expected
       *   that the hook only affects the gradient of the backward
       *   propagation, and does not affect the gradient value input
       *   as the hook.
       * - use `tmp_ins_ptr`, only copy bwd_ins when the var in bwd_ins
       *   hold hooks
       */
      auto tmp_ins_ptr = CallGradientHooks(bwd_ins, cur_op.Type());

      if (!tmp_ins_ptr) {
        PerformBackwardInplace(cur_op.Type(), bwd_ins, &tmp_outs);
      }

      {
        VLOG(3) << "Start to execute grad op " << cur_op.Type();
        phi::memory_utils::MemoryDebugger debugger(cur_op.Type());
        try {
          if (tmp_ins_ptr == nullptr) {
            OpBase::Run(cur_op.InnerOp(),
                        bwd_ins,
                        tmp_outs,
                        cur_op.Attrs(),
                        cur_op.DefaultAttrsMap(),
                        cur_op.place());
          } else {
            OpBase::Run(cur_op.InnerOp(),
                        *tmp_ins_ptr,
                        tmp_outs,
                        cur_op.Attrs(),
                        cur_op.DefaultAttrsMap(),
                        cur_op.place());
          }
        } catch (platform::EnforceNotMet& exception) {
          Clear();
          throw exception;
        } catch (std::exception& ex) {
          Clear();
          PADDLE_THROW(platform::errors::External("%s", ex.what()));
        }
      }

      // Function Post Hook
      if (cur_op.HasVoidFunctionPostHook()) {
        for (const auto& hook : cur_op.GetVoidFunctionPostHooks()) {
          (*hook)();
        }
      }

      for (auto& pair : inplace_output_grad_var_list_) {
        *pair.first = *pair.second;
      }

      // Step 2: Sum Gradient of This graph
      for (auto& pair : need_accu_var_list_) {
        pair.first->SumGrad(std::move(pair.second), cur_op.id());
      }

      // Step 3: Call Hooks && Sum Gradient with Pre-Graph && Call BackwardHooks
      for (auto* accumulator : leaf_accumulators_) {
        if (!accumulator->SumGradCompleted()) {
          continue;
        }
        // 1. Call Hooks for `inner_var_`
        accumulator->CallGradientHooks();

        // 2. Sum Gradient `inner_var_` to `var_` of Current or Previous Graph
        accumulator->AccumulateGrad();

        // 3. Call backward Hooks for `var_`
        accumulator->CallReduceHooks();
      }

      need_accu_var_list_.clear();
      inplace_output_grad_var_list_.clear();
      leaf_accumulators_.clear();

      if (!retain_graph_) {
        VLOG(3) << "Remove op after op " << cur_op.Type() << " runs";
        cur_op.ClearBackwardTrace();
      }
    }

    // Step 3: Collect ready ops
    for (auto& grad_pending_node : shared_cur_node->GradPendingNodes()) {
      PADDLE_ENFORCE_NOT_NULL(
          grad_pending_node,
          platform::errors::NotFound("Grad pending node is nullptr."));
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

  phi::autotune::AutoTuneStatus::Instance().Update();
}

void BasicEngine::Clear() {
  init_nodes_.clear();
  node_deps_.clear();
  accumulators_.clear();
  accumulators_with_grad_node_.clear();
  need_accu_var_list_.clear();
  leaf_accumulators_.clear();
}

}  // namespace imperative
}  // namespace paddle
