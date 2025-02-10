/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/ir/constant_folding_pass.h"

#include <string>
#include <vector>
#include "glog/logging.h"

#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle::framework::ir {
class Node;
}  // namespace paddle::framework::ir

/*
 * When a op's inputs and outputs is determined before feeding data to the
 * model, we can remove this op from the model. This ConstantFolding pass can
 * remove all these like ops.
 *
 */

namespace paddle::framework::ir::patterns {

struct ConstantFolding : public PatternBase {
  ConstantFolding(PDPattern *pattern, const std::string &name_scope)
      : PatternBase(pattern, name_scope, "constant_folding_pass") {}
};
}  // namespace paddle::framework::ir::patterns
namespace paddle::framework::ir {

namespace {
std::unordered_set<std::string> GetControlFlowVarNames(ir::Graph *graph) {
  std::unordered_set<std::string> control_flow_ops{"while",
                                                   "conditional_block"};
  std::unordered_set<std::string> control_flow_var_names;
  for (auto *node : graph->Nodes()) {
    if (!node->IsOp() || control_flow_ops.count(node->Op()->Type()) == 0)
      continue;
    for (auto const &in_names : node->Op()->Inputs()) {
      auto var_names = in_names.second;
      control_flow_var_names.insert(var_names.begin(), var_names.end());
    }
    for (auto const &out_names : node->Op()->Outputs()) {
      auto var_names = out_names.second;
      control_flow_var_names.insert(var_names.begin(), var_names.end());
    }
  }
  return control_flow_var_names;
}

bool OutputUsedByControlFlow(ir::Node *node,
                             const std::unordered_set<std::string> &cf_vars) {
  for (auto out_node : node->outputs) {
    if (cf_vars.count(out_node->Name())) {
      return true;
    }
  }
  return false;
}
}  // namespace

ConstantFoldingPass::ConstantFoldingPass() = default;

void ConstantFoldingPass::ApplyImpl(ir::Graph *graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::PreconditionNotMet("graph should not be null."));
  FusePassBase::Init("constant_folding", graph);
  auto *scope = param_scope();

  PADDLE_ENFORCE_NOT_NULL(
      scope,
      common::errors::Fatal(
          "scope must not be null when applying constant folding."));

  std::vector<std::string> blacklist{"feed",
                                     "matrix_multiply",
                                     "save",
                                     "quantize_linear",
                                     "dequantize_linear"};
  const auto cf_vars = GetControlFlowVarNames(graph);
  int folded_op_num = 0;

  auto op_node_sorted = framework::ir::TopologyVariantSort(
      *graph, static_cast<framework::ir::SortKind>(0));
  for (auto *op_node : op_node_sorted) {
    if (!op_node->IsOp()) continue;
    if (std::find(blacklist.begin(), blacklist.end(), op_node->Name()) !=
        blacklist.end())
      continue;
    if (OutputUsedByControlFlow(op_node, cf_vars)) {
      continue;
    }
    bool input_persis = true;
    // map is used to record how many time a name string occurs in the whole
    // graph's nodes
    std::unordered_map<std::string, int> map;
    for (auto in_node : op_node->inputs) {
      map[in_node->Name()] = 0;
      if (in_node->Var() == nullptr || !in_node->Var()->Persistable() ||
          !in_node->inputs.empty()) {
        input_persis = false;
      }
    }
    for (auto out_node : op_node->outputs) {
      map[out_node->Name()] = 0;
      if (out_node->Var() == nullptr) {
        input_persis = false;
      }
    }
    // Forbid other node in graph having the same name with nodes in map
    for (auto const &iter : map) {
      for (auto node : graph->Nodes()) {
        if (node->IsVar() && node->Name() == iter.first) {
          map[node->Name()]++;
          if (map[node->Name()] > 1) {
            input_persis = false;
          }
        }
      }
    }

    framework::Scope *local_scope = new framework::Scope();
    std::unordered_set<const paddle::framework::ir::Node *> remove_nodes;
    std::unique_ptr<OperatorBase> op;

    if (input_persis) {
      for (auto in_node : op_node->inputs) {
        local_scope->Var(in_node->Var()->Name());
        local_scope->FindVar(in_node->Var()->Name())
            ->GetMutable<phi::DenseTensor>();
        // This persistable input node is exclusive, and can be removed
        if (in_node->outputs.size() == 1L) remove_nodes.emplace(in_node);

        auto in_shape = in_node->Var()->GetShape();
        auto *global_persis_x_tensor =
            scope->FindVar(in_node->Name())->GetMutable<phi::DenseTensor>();
        auto *local_x_tensor = local_scope->FindVar(in_node->Name())
                                   ->GetMutable<phi::DenseTensor>();
        local_x_tensor->Resize(global_persis_x_tensor->dims());
        *local_x_tensor = *global_persis_x_tensor;
      }

      op = paddle::framework::OpRegistry::CreateOp(*op_node->Op());
      remove_nodes.emplace(op_node);
      for (auto out_node : op_node->outputs) {
        local_scope->Var(out_node->Var()->Name());
        local_scope->FindVar(out_node->Var()->Name())
            ->GetMutable<phi::DenseTensor>();
        // useless out_node can be removed, not need set it persistable !
        if (out_node->outputs.empty()) remove_nodes.emplace(out_node);
      }
      op->Run(*local_scope, phi::CPUPlace());
      folded_op_num++;
      for (auto out_node : op_node->outputs) {
        // this out_node is useless, do not set it persistable
        if (out_node->outputs.empty()) continue;
        auto out_desc = out_node->Var();
        auto out_name = out_desc->Name();
        auto *local_out_tensor =
            local_scope->FindVar(out_name)->GetMutable<phi::DenseTensor>();
        std::vector<int64_t> out_shape;
        for (int64_t i = 0; i < local_out_tensor->dims().size(); i++) {
          out_shape.push_back(local_out_tensor->dims()[static_cast<int>(i)]);
        }
        out_desc->SetShape(out_shape);
        out_desc->SetPersistable(true);
        auto *var_desc_out = op_node->Op()->Block()->Var(out_name);
        var_desc_out->SetShape(out_shape);
        var_desc_out->SetPersistable(true);
        var_desc_out->Flush();
        auto *global_out_tensor =
            scope->Var(out_name)->GetMutable<phi::DenseTensor>();
        *global_out_tensor = *local_out_tensor;
      }
      GraphSafeRemoveNodes(graph, remove_nodes);
    }
    delete local_scope;
  }
  AddStatis(folded_op_num);
}

}  // namespace paddle::framework::ir

REGISTER_PASS(constant_folding_pass,
              paddle::framework::ir::ConstantFoldingPass);
