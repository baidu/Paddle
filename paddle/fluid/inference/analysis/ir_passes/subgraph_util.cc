/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

/*
 * This file defines the class to partition a graph.
 */

#include "paddle/fluid/inference/analysis/ir_passes/subgraph_util.h"

#include "paddle/fluid/platform/enforce.h"

namespace paddle::framework::ir {
class Node;
}  // namespace paddle::framework::ir

namespace paddle::inference::analysis {
using framework::ir::Node;

std::vector<std::string> ExtractParameters(
    const std::unordered_set<Node *> &nodes, bool sorted) {
  // We can judge whether a variable is a parameter by
  // its persistable property, but sometimes the persistable
  // of the feed op output is true, so we have to identify it.
  std::vector<std::string> feed_outputs;
  for (const auto &node : nodes) {
    if (!node->IsOp()) continue;
    std::string op_type = node->Op()->Type();
    if (op_type == "feed" || op_type == "fetch") {
      std::vector<std::string> output_names = node->Op()->OutputArgumentNames();
      std::copy(output_names.begin(),
                output_names.end(),
                std::back_inserter(feed_outputs));
    }
  }

  std::vector<std::string> parameters;
  for (const auto &node : nodes) {
    if (!node->IsVar()) continue;
    if (node->Var()->Persistable() &&
        std::find(feed_outputs.begin(), feed_outputs.end(), node->Name()) ==
            feed_outputs.end()) {
      parameters.push_back(node->Name());
    }
  }
  if (sorted) {
    std::sort(parameters.begin(), parameters.end());
    parameters.erase(std::unique(parameters.begin(), parameters.end()),
                     parameters.end());
  }
  return parameters;
}

std::unordered_set<Node *> GetRelatedIOVarNodes(
    const std::vector<Node *> &nodes) {
  std::unordered_set<Node *> io_nodes;
  for (const auto &node : nodes) {
    if (!node->IsOp()) continue;
    for (const auto &in : node->inputs) {
      io_nodes.insert(in);
    }
    for (const auto &out : node->outputs) {
      io_nodes.insert(out);
    }
  }
  return io_nodes;
}

void PrependFeedOps(framework::BlockDesc *global_block,
                    const std::vector<std::string> &feed_target_names,
                    std::string feed_holder_name) {
  framework::VarDesc *feed_var = global_block->Var(feed_holder_name);
  feed_var->SetType(paddle::framework::proto::VarType::FEED_MINIBATCH);
  feed_var->SetPersistable(true);
  for (size_t i = 0; i < feed_target_names.size(); i++) {
    framework::OpDesc *feed_op = global_block->AppendOp();
    feed_op->SetType("feed");
    feed_op->SetInput("X", {feed_holder_name});
    feed_op->SetOutput("Out", {feed_target_names[i]});
    feed_op->SetAttr("col", static_cast<int>(i));
  }
}

void PrependFetchOps(framework::BlockDesc *global_block,
                     const std::vector<std::string> &fetch_target_names,
                     std::string fetch_holder_name) {
  framework::VarDesc *fetch_var = global_block->Var(fetch_holder_name);
  fetch_var->SetType(paddle::framework::proto::VarType::FETCH_LIST);
  fetch_var->SetPersistable(true);
  for (size_t i = 0; i < fetch_target_names.size(); i++) {
    framework::OpDesc *fetch_op = global_block->AppendOp();
    fetch_op->SetType("fetch");
    fetch_op->SetInput("X", {fetch_target_names[i]});
    fetch_op->SetOutput("Out", {fetch_holder_name});
    fetch_op->SetAttr("col", static_cast<int>(i));
  }
}

void RenameAndGetOutputs(
    const std::vector<framework::ir::Node *> &subgraph_nodes,
    framework::BlockDesc *block_desc,
    const std::set<std::string> &input_names_with_id,
    std::set<std::string> *output_names_with_id,
    std::set<std::string> *output_names,
    std::unordered_map<std::string, std::string> *output_name_map,
    const std::unordered_map<std::string, framework::ir::Node *> &graph_var_map,
    bool trt_and_not_int8) {
  //// In the normal case, the paddle-trt exists bug when running the googlenet.
  // When there are more than two convolutions of 1 * 1 with the same input, the
  // paddle-tensorrt will do the merging optimization, which fuse those conv
  // into one conv, and then trigger bug. So,  We should use strategy to avoid
  // this optimization for the time being. This bug will be fixed in the future.
  std::unordered_map<std::string /*name*/, int /*ITensor_quote_num*/>
      same_hierarchy_conv2d_num_map;

  auto add_block_var = [&](const std::string &graph_arg,
                           const std::string &block_arg) {
    auto arg_var_node = graph_var_map.find(graph_arg);
    PADDLE_ENFORCE_NE(arg_var_node,
                      graph_var_map.end(),
                      common::errors::InvalidArgument(
                          "Can not find %s in graph_var_map", graph_arg));
    auto *var_t = block_desc->Var(block_arg);
    var_t->SetShape(arg_var_node->second->Var()->GetShape());
    var_t->SetDataType(arg_var_node->second->Var()->GetDataType());
  };

  for (size_t index = 0; index < block_desc->OpSize(); ++index) {
    framework::proto::OpDesc *op =
        block_desc->Op(static_cast<int>(index))->Proto();
    framework::OpDesc op_desc(*op, nullptr);
    auto correspond_node = subgraph_nodes[index];
    PADDLE_ENFORCE_EQ(
        correspond_node->Name(),
        op->type(),
        common::errors::PreconditionNotMet("We should get %s, but get %s",
                                           op->type(),
                                           correspond_node->Name()));

    std::unordered_map<std::string, size_t> var2id;
    std::unordered_map<std::string, framework::ir::Node *> in_vars;
    for (auto *in_var : correspond_node->inputs) {
      var2id[in_var->Name()] = in_var->id();
      in_vars[in_var->Name()] = in_var;
    }
    // rename for the input variables of op inside subgraph
    for (int i = 0; i < op->inputs_size(); i++) {
      // one input
      auto *in_var = op->mutable_inputs(i);
      std::vector<std::string> replaced_names;
      for (int k = 0; k < in_var->arguments_size(); k++) {  // all the arguments
        const std::string arg_value = in_var->arguments(k);
        const std::string arg_value_with_id =
            RenameVarBeUnique(arg_value, std::to_string(var2id[arg_value]));
        if (input_names_with_id.count(arg_value_with_id)) {
          replaced_names.push_back(arg_value);
          if (graph_var_map.count(arg_value)) {
            add_block_var(arg_value, arg_value);
          }
        } else {
          replaced_names.push_back(arg_value_with_id);
          if (graph_var_map.count(arg_value)) {
            add_block_var(arg_value, arg_value_with_id);
          }
        }
      }
      in_var->clear_arguments();
      for (auto &replaced_name : replaced_names) {
        in_var->add_arguments(replaced_name);
      }
    }
    var2id.clear();
    for (auto out_var : correspond_node->outputs) {
      var2id[out_var->Name()] = out_var->id();
    }
    if (op_desc.Type() == "conv2d" && trt_and_not_int8) {
      auto input_var_name = op_desc.Input("Input").front();
      auto filter_var_name = op_desc.Input("Filter").front();
      auto out_var_name = op_desc.Output("Output").front();
      auto filter_shape = in_vars[filter_var_name]->Var()->GetShape();
      const std::vector<int> strides =
          PADDLE_GET_CONST(std::vector<int>, op_desc.GetAttr("strides"));
      const std::vector<int> paddings =
          PADDLE_GET_CONST(std::vector<int>, op_desc.GetAttr("paddings"));
      if (same_hierarchy_conv2d_num_map[input_var_name] > 0) {
        (*output_names_with_id)
            .insert(RenameVarBeUnique(out_var_name,
                                      std::to_string(var2id[out_var_name])));
        (*output_names).insert(out_var_name);
      } else if (filter_shape[2] == 1 && filter_shape[3] == 1 &&
                 strides[0] == 1 && strides[1] == 1 && paddings[0] == 0 &&
                 paddings[1] == 0) {
        same_hierarchy_conv2d_num_map[input_var_name] += 1;
      }
    }
    // rename for the output variables of op inside subgraph
    for (int i = 0; i < op->outputs_size(); i++) {
      framework::proto::OpDesc_Var *out_var = op->mutable_outputs(i);
      std::vector<std::string> replaced_names;
      for (int k = 0; k < out_var->arguments_size(); k++) {
        const std::string arg_value = out_var->arguments(k);
        const std::string arg_value_with_id =
            RenameVarBeUnique(arg_value, std::to_string(var2id[arg_value]));
        if (graph_var_map.count(arg_value)) {
          add_block_var(arg_value, arg_value_with_id);
        }
        if (output_names_with_id->count(arg_value_with_id)) {
          (*output_name_map)[arg_value] = arg_value_with_id;
        }
        replaced_names.push_back(arg_value_with_id);
      }
      out_var->clear_arguments();
      for (auto &replaced_name : replaced_names) {
        out_var->add_arguments(replaced_name);
      }
    }
  }
}

std::string RenameVarBeUnique(std::string original_var_name,
                              std::string var_id) {
  return original_var_name + "_subgraph_" + var_id;
}

}  // namespace paddle::inference::analysis
