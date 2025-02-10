// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
// Copyright (c) 2023 NVIDIA Authors. All Rights Reserved.
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
#include "paddle/fluid/framework/ir/fuse_adamw_op_pass.h"
#include <string>
#include "paddle/fluid/framework/details/multi_devices_helper.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle::framework::ir {

std::vector<std::string> GetNodeNames(const std::vector<Node *> &node_vector) {
  std::vector<std::string> out_vector;
  out_vector.reserve(node_vector.size());
  for (auto i : node_vector) {
    out_vector.emplace_back(i->Name());
  }
  return out_vector;
}

Node *GetInputNode(const Node *op, const std::string &name) {
  Node *out = nullptr;

  for (auto &node : op->inputs) {
    if (node->Name() == op->Op()->Input(name)[0]) {
      out = node;
      break;
    }
  }

  PADDLE_ENFORCE_NOT_NULL(
      out, common::errors::InvalidArgument("Input's name cannot be found."));

  return out;
}

Node *GetOutputNode(const Node *op, const std::string &name) {
  Node *out = nullptr;

  for (auto &node : op->outputs) {
    if (node->Name() == op->Op()->Output(name)[0]) {
      out = node;
      break;
    }
  }

  PADDLE_ENFORCE_NOT_NULL(
      out, common::errors::InvalidArgument("Output's name cannot be found."));

  return out;
}

void SaveInOutNodes(std::vector<std::vector<Node *>> *inout_node_vectors,
                    const AdamWConfig &config,
                    const Node *op) {
  size_t i = 0;

  for (auto &name : config.inputs_name) {
    (*inout_node_vectors)[i].emplace_back(GetInputNode(op, name));
    i++;
  }
  for (auto &name : config.outputs_name) {
    (*inout_node_vectors)[i].emplace_back(GetOutputNode(op, name));
    i++;
  }
  if (config.multi_precision) {
    (*inout_node_vectors)[i++].emplace_back(GetInputNode(op, "MasterParam"));
    (*inout_node_vectors)[i].emplace_back(GetOutputNode(op, "MasterParamOut"));
  }
}

void InsertOpToGraph(const std::vector<std::vector<Node *>> &inout_node_vectors,
                     const AdamWConfig &config,
                     ir::Graph *graph) {
  float weight_decay = static_cast<float>(0.0);
  bool use_adamw = false;

  if (config.with_decay) {
    weight_decay = config.first_coeff;
    use_adamw = true;
  }
  if (!inout_node_vectors[0].empty() && config.replace_adamw) {
    OpDesc fuse_adamw_op_desc(config.block);
    fuse_adamw_op_desc.SetType("fused_adam");

    size_t i = 0;

    for (auto &name : config.replace_inputs_name) {
      fuse_adamw_op_desc.SetInput(name, GetNodeNames(inout_node_vectors[i]));
      i++;
    }

    fuse_adamw_op_desc.SetInput("LearningRate",
                                {config.first_lr->Name()});  // NOLINT
    if (config.use_skip_update) {
      fuse_adamw_op_desc.SetInput("SkipUpdate",
                                  {config.first_skip_update->Name()});
    } else {
      fuse_adamw_op_desc.SetInput("SkipUpdate", {});
    }

    for (auto &name : config.replace_outputs_name) {
      fuse_adamw_op_desc.SetOutput(name, GetNodeNames(inout_node_vectors[i]));
      i++;
    }

    if (config.multi_precision) {
      fuse_adamw_op_desc.SetInput("MasterParams",
                                  GetNodeNames(inout_node_vectors[i++]));
      fuse_adamw_op_desc.SetOutput("MasterParamsOut",
                                   GetNodeNames(inout_node_vectors[i]));
    } else {
      fuse_adamw_op_desc.SetInput("MasterParams", {});
    }

    fuse_adamw_op_desc.SetAttr("beta1", config.beta1);
    fuse_adamw_op_desc.SetAttr("beta2", config.beta2);
    fuse_adamw_op_desc.SetAttr("op_role", config.op_role);
    fuse_adamw_op_desc.SetAttr("epsilon", config.epsilon);
    fuse_adamw_op_desc.SetAttr("chunk_size", 16 * 2048);
    fuse_adamw_op_desc.SetAttr("weight_decay", weight_decay);
    fuse_adamw_op_desc.SetAttr("use_adamw", use_adamw);
    fuse_adamw_op_desc.SetAttr("multi_precision", config.multi_precision);
    fuse_adamw_op_desc.SetAttr("use_global_beta_pow",
                               config.use_global_beta_pow);

    auto fuse_adamw_node = graph->CreateOpNode(&fuse_adamw_op_desc);

    IR_NODE_LINK_TO(config.first_lr, fuse_adamw_node);
    if (config.use_skip_update) {
      IR_NODE_LINK_TO(config.first_skip_update, fuse_adamw_node);
    }

    for (size_t k = 0; k < inout_node_vectors[0].size(); k++) {
      size_t j = 0;

      for (; j < config.replace_inputs_name.size(); j++) {
        IR_NODE_LINK_TO(inout_node_vectors[j][k], fuse_adamw_node);
      }
      for (; j < config.replace_inputs_name.size() +
                     config.replace_outputs_name.size();
           j++) {
        IR_NODE_LINK_TO(fuse_adamw_node, inout_node_vectors[j][k]);
      }
      if (config.multi_precision) {
        IR_NODE_LINK_TO(inout_node_vectors[j][k], fuse_adamw_node);
        j++;
        IR_NODE_LINK_TO(fuse_adamw_node, inout_node_vectors[j][k]);
      }
    }
  }
}

bool InitAndCheckAttrs(const size_t &found_adamw_count,
                       AdamWConfig *config,
                       const Node *op,
                       bool *is_continue) {
  const Node *adamw_op = op;
  Node *skip_update = nullptr;
  Node *learning_rate = GetInputNode(adamw_op, "LearningRate");
  auto adamw_op_desc = adamw_op->Op();

  // Initialize variables
  float coeff = 0.0, lr_ratio = 1.0;
  bool lazy_mode = false;
  int64_t min_row_size_to_use_multithread = 1000;

  // Get skip_update and coeff, these will be used to check whether we can
  // use fuse_adamw.
  for (auto &node : adamw_op->inputs) {
    auto in_name = adamw_op_desc->Input("SkipUpdate");
    if (!in_name.empty()) {
      if (node->Name() == in_name[0]) {
        config->use_skip_update = true;
        skip_update = node;
        break;
      }
    }
  }
  coeff = PADDLE_GET_CONST(float, adamw_op_desc->GetAttr("coeff"));

  // Get attrs and block
  if (found_adamw_count == 0) {
    // Get block
    config->block = adamw_op_desc->Block();
    // Get attrs
    config->beta1 = PADDLE_GET_CONST(float, adamw_op_desc->GetAttr("beta1"));
    config->beta2 = PADDLE_GET_CONST(float, adamw_op_desc->GetAttr("beta2"));
    config->op_role = PADDLE_GET_CONST(int, adamw_op_desc->GetAttr("op_role"));
    config->epsilon =
        PADDLE_GET_CONST(float, adamw_op_desc->GetAttr("epsilon"));
    config->use_global_beta_pow =
        PADDLE_GET_CONST(bool, adamw_op_desc->GetAttr("use_global_beta_pow"));

    lazy_mode = PADDLE_GET_CONST(bool, adamw_op_desc->GetAttr("lazy_mode"));
    min_row_size_to_use_multithread = PADDLE_GET_CONST(
        int64_t, adamw_op_desc->GetAttr("min_row_size_to_use_multithread"));
    lr_ratio = PADDLE_GET_CONST(float, adamw_op_desc->GetAttr("lr_ratio"));

    config->first_lr = learning_rate;
    config->first_coeff = coeff;
    if (config->use_skip_update) {
      config->first_skip_update = skip_update;
    }

    // We do not support these patterns
    if (lazy_mode != false || lr_ratio != 1.0 ||
        min_row_size_to_use_multithread != 1000) {
      return false;
    }
  }

  // Check whether with_decay and multi_precision are matched
  if (config->with_decay !=
          PADDLE_GET_CONST(bool, adamw_op_desc->GetAttr("with_decay")) ||
      config->multi_precision !=
          PADDLE_GET_CONST(bool, adamw_op_desc->GetAttr("multi_precision"))) {
    *is_continue = true;
    return true;
  }

  // We do not support these patterns
  if ((learning_rate->Name() != config->first_lr->Name()) ||
      (coeff != config->first_coeff) ||
      (config->use_skip_update &&
       skip_update->Name() != config->first_skip_update->Name())) {
    return false;
  }

  return true;
}

void FuseAdamWPass::ApplyImpl(ir::Graph *graph) const {
  graph = FuseAdamWFun(graph, true, true);
  graph = FuseAdamWFun(graph, true, false);
  graph = FuseAdamWFun(graph, false, true);
  graph = FuseAdamWFun(graph, false, false);  // NOLINT
}

ir::Graph *FuseAdamWPass::FuseAdamWFun(ir::Graph *graph,
                                       const bool with_decay,
                                       const bool multi_precision) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::InvalidArgument("Graph cannot be nullptr."));

  VLOG(4) << "handle fuse AdadW";

  const std::string scope_name("fuse_adamw");
  FusePassBase::Init(scope_name, graph);

  size_t found_adamw_count = 0;

  AdamWConfig config;

  config.with_decay = with_decay;
  config.multi_precision = multi_precision;

  // Used to store Nodes of input and output for each pattern
  std::vector<std::vector<Node *>> inout_node_vectors(13);

  std::unordered_set<const Node *> adamw_op_del_set;

  for (auto &node : graph->Nodes()) {
    if (node->Name() == "adamw") {
      const Node *adamw_op = node;
      bool is_continue = false;

      // Initialize attrs and check attrs to determine whether we support this
      // pattern.
      if (!InitAndCheckAttrs(found_adamw_count, &config, node, &is_continue)) {
        config.replace_adamw = false;
        return graph;
      }

      if (is_continue) {
        continue;
      }

      adamw_op_del_set.insert(adamw_op);

      // Save input and output Nodes
      SaveInOutNodes(&inout_node_vectors, config, adamw_op);

      found_adamw_count++;
    }
  }

  // Remove old op
  if (config.replace_adamw && !inout_node_vectors[0].empty()) {
    GraphSafeRemoveNodes(graph, adamw_op_del_set);
  }

  // Insert new op to graph
  InsertOpToGraph(inout_node_vectors, config, graph);

  VLOG(4) << "replace adamw with fuse_adamw";

  AddStatis(static_cast<int>(found_adamw_count));
  return graph;
}

}  // namespace paddle::framework::ir

REGISTER_PASS(fuse_adamw_op_pass, paddle::framework::ir::FuseAdamWPass);
