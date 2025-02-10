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

#include "paddle/fluid/framework/ir/fuse_multi_transformer_layer_pass.h"

#include <string>

#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_version_registry.h"

namespace paddle {
namespace framework {
class Scope;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace framework {
namespace ir {
namespace patterns {

std::unordered_map<std::string, std::string>
MultiTransformerLayerPattern::operator()(bool enable_int8,
                                         int num_fused_op,
                                         bool is_decoder) {
  std::string fused_multi_transformer_name =
      enable_int8 ? "fused_multi_transformer_int8" : "fused_multi_transformer";

  std::unordered_map<std::string, std::string> node_reprs;

  // x0 and src_mask is unique input of subgraph
  auto* x0 = pattern->NewNode(x0_repr());
  x0->assert_is_op_input(fused_multi_transformer_name, "X")->AsInput();
  auto* src_mask = pattern->NewNode(src_mask_repr());
  src_mask->assert_is_op_input(fused_multi_transformer_name, "SrcMask")
      ->AsInput();

  for (int i = 0; i < num_fused_op; ++i) {
    auto fuse_op_repr =
        PDNodeName(name_scope_, repr_, id_, "fuse_op_" + std::to_string(i));
    node_reprs["fuse_op_" + std::to_string(i)] = fuse_op_repr;
    auto* fused_multi_transformer =
        pattern->NewNode(fuse_op_repr)
            ->assert_is_op(fused_multi_transformer_name);

    auto out_repr =
        PDNodeName(name_scope_, repr_, id_, "out_" + std::to_string(i));
    node_reprs["out_" + std::to_string(i)] = out_repr;
    auto* out = pattern->NewNode(out_repr)->assert_is_op_output(
        fused_multi_transformer_name, "Out");

    if (is_decoder) {
      fused_multi_transformer->LinksFrom({x0, src_mask}).LinksTo({out});
    } else {
      auto cache_kv_repr =
          PDNodeName(name_scope_, repr_, id_, "cache_kv_" + std::to_string(i));
      node_reprs["cache_kv_" + std::to_string(i)] = cache_kv_repr;
      auto* cache_kv = pattern->NewNode(cache_kv_repr);
      cache_kv->assert_is_op_input(fused_multi_transformer_name, "CacheKV");
      cache_kv->AsInput();

      auto fill_const_op_repr =
          PDNodeName(name_scope_, repr_, id_, "fill_op_" + std::to_string(i));
      node_reprs["fill_op_" + std::to_string(i)] = fill_const_op_repr;
      auto fill_const_op = pattern->NewNode(fill_const_op_repr)
                               ->assert_is_op("fill_constant_batch_size_like");

      fused_multi_transformer->LinksFrom({x0, src_mask, cache_kv})
          .LinksTo({out});
      fill_const_op->LinksFrom({x0}).LinksTo({cache_kv});
    }
    x0 = out;
  }
  x0->AsOutput();
  return node_reprs;
}
}  // namespace patterns

inline void MergeInput(OpDesc* op,
                       const std::vector<VariableNameMap>& input_name_maps,
                       const std::string& input_name) {
  std::vector<std::string> tmp = input_name_maps[0].at(input_name);
  for (size_t i = 1; i < input_name_maps.size(); ++i) {
    tmp.insert(tmp.end(),
               input_name_maps[i].at(input_name).begin(),
               input_name_maps[i].at(input_name).end());
  }
  op->SetInput(input_name, tmp);
}

template <typename T>
inline void MergeAttrs(const std::vector<OpDesc*>& ops,
                       const std::string& attr_name) {
  std::vector<T> res;
  for (auto op : ops) {
    auto scale_vec = PADDLE_GET_CONST(std::vector<T>, op->GetAttr(attr_name));
    res.insert(res.end(), scale_vec.begin(), scale_vec.end());
  }
  ops[0]->SetAttr(attr_name, res);
}

int FuseMultiTransformerLayerPass::BuildFusion(Graph* graph,
                                               const std::string& name_scope,
                                               Scope* scope) const {
  GraphPatternDetector gpd;
  auto* pattern = gpd.mutable_pattern();

  bool enable_int8 = false;
  if (graph->Has("enable_int8")) {
    enable_int8 = graph->Get<bool>("enable_int8");
  }
  if (!enable_int8) {
    VLOG(4)
        << "fuse_multi_layer_transformer_pass will match float transformer op "
           "cause enable_int8 is not been set or set to false";
  }

  int num_fuse_op = 0;
  bool is_decoder = false;

  if (graph->Has(kFusedMultiTransformerEncoderFusionCount)) {
    num_fuse_op = graph->Get<int>(kFusedMultiTransformerEncoderFusionCount);
    is_decoder = false;
  } else if (graph->Has(kFusedMultiTransformerDecoderFusionCount)) {
    num_fuse_op = graph->Get<int>(kFusedMultiTransformerDecoderFusionCount);
    is_decoder = true;
  }
  if (num_fuse_op == 0) {
    VLOG(4) << "fuse_multi_transformer_layer_pass will be skipped "
               "cause num_fuse_op is not been set or set to 0";
    return 0;
  }
  if (!is_decoder) {
    VLOG(4) << "fuse_multi_transformer_layer_pass will match encoder pattern";
  } else {
    VLOG(4) << "fuse_multi_transformer_layer_pass will match decoder pattern";
  }

  patterns::MultiTransformerLayerPattern multi_layer_pattern(pattern,
                                                             name_scope);
  auto node_reprs = multi_layer_pattern(enable_int8, num_fuse_op, is_decoder);

  int fusion_count{0};
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    ///////////////////
    //// Get nodes ////
    ///////////////////

    GET_IR_NODE_FROM_SUBGRAPH(src_mask, src_mask, multi_layer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(x0, x0, multi_layer_pattern);

    std::vector<Node*> fuse_op_nodes;
    std::vector<Node*> out_nodes;

    std::vector<OpDesc*> fuse_op_descs;
    std::vector<VariableNameMap> fuse_op_input_var_name_maps;
    std::vector<VariableNameMap> fuse_op_output_var_name_maps;

    for (int i = 0; i < num_fuse_op; ++i) {
      PDNode* fuse_op_pdnode =
          multi_layer_pattern.PatternBase::pattern->RetrieveNode(
              node_reprs["fuse_op_" + std::to_string(i)]);
      Node* fuse_op_node = subgraph.at(fuse_op_pdnode);
      fuse_op_nodes.push_back(fuse_op_node);
      fuse_op_descs.push_back(fuse_op_node->Op());
      fuse_op_input_var_name_maps.emplace_back(fuse_op_node->Op()->Inputs());
      fuse_op_output_var_name_maps.emplace_back(fuse_op_node->Op()->Outputs());

      PDNode* out_pdnode =
          multi_layer_pattern.PatternBase::pattern->RetrieveNode(
              node_reprs["out_" + std::to_string(i)]);
      out_nodes.push_back(subgraph.at(out_pdnode));

      // fill_const op use x0 as input
      if (!is_decoder && i != 0) {
        PDNode* fill_op_pdnode =
            multi_layer_pattern.PatternBase::pattern->RetrieveNode(
                node_reprs["fill_op_" + std::to_string(i)]);
        Node* fill_op_node = subgraph.at(fill_op_pdnode);
        fill_op_node->Op()->SetInput("Input", {x0->Name()});
        IR_NODE_UNLINK(out_nodes[i - 1], fill_op_node);
        IR_NODE_LINK_TO(x0, fill_op_node);
      }
    }

    ///////////////
    //// Merge ////
    ///////////////

    // Merge inputs
    std::vector<std::string> inputs_names = {"CacheKV",
                                             "FFN1Bias",
                                             "FFN1Weight",
                                             "FFN2Bias",
                                             "FFN2Weight",
                                             "FFNLnBias",
                                             "FFNLnScale",
                                             "LnBias",
                                             "LnScale",
                                             "OutLinearBias",
                                             "OutLinearW",
                                             "QKVBias",
                                             "QKVW"};
    if (enable_int8) {
      std::vector<std::string> inputs_names_int8_supp = {
          "FFN1OutScale", "FFN2OutScale", "OutLinearOutScale", "QKVOutScale"};
      inputs_names.insert(inputs_names.end(),
                          inputs_names_int8_supp.begin(),
                          inputs_names_int8_supp.end());
    }
    for (const auto& input_name : inputs_names) {
      MergeInput(fuse_op_descs[0], fuse_op_input_var_name_maps, input_name);
    }

    // Merge outputs
    fuse_op_descs[0]->SetOutput(
        "Out", fuse_op_output_var_name_maps[num_fuse_op - 1]["Out"]);
    auto& merged_cache_kv_out_names =
        fuse_op_output_var_name_maps[0]["CacheKVOut"];
    for (int i = 1; i < num_fuse_op; ++i) {
      const auto& out_var_names = fuse_op_output_var_name_maps[i]["CacheKVOut"];
      merged_cache_kv_out_names.insert(merged_cache_kv_out_names.end(),
                                       out_var_names.begin(),
                                       out_var_names.end());
    }
    fuse_op_descs[0]->SetOutput("CacheKVOut", merged_cache_kv_out_names);

    if (enable_int8) {
      // Merge inputs scale
      std::vector<std::string> attr_names = {"qkv_in_scale",
                                             "out_linear_in_scale",
                                             "ffn1_in_scale",
                                             "ffn2_in_scale"};
      for (const auto& name : attr_names) {
        MergeAttrs<float>(fuse_op_descs, name);
      }
    }

    ////////////////
    //// ReLink ////
    ////////////////
    // Before relink, out nodes (0 -> num_layer-1) should be removed
    std::unordered_set<const Node*> marked_out_nodes(out_nodes.begin(),
                                                     out_nodes.end() - 1);
    GraphSafeRemoveNodes(graph, marked_out_nodes);

    // Relink all input nodes of fused_multi_transformer ops to the first op
    auto& merged_inputs = fuse_op_nodes[0]->inputs;
    for (int i = 1; i < num_fuse_op; ++i) {
      merged_inputs.insert(merged_inputs.end(),
                           fuse_op_nodes[i]->inputs.begin(),
                           fuse_op_nodes[i]->inputs.end());
    }

    // Relink fuse op -> out
    IR_NODE_UNLINK(fuse_op_nodes[num_fuse_op - 1], out_nodes[num_fuse_op - 1]);
    IR_NODE_LINK_TO(fuse_op_nodes[0], out_nodes[num_fuse_op - 1]);

    /////////////////////////////
    //// Delete unused nodes ////
    /////////////////////////////
    // Delete fused_multi_transformer op expect for the first one
    std::unordered_set<const Node*> marked_fuse_op_nodes(
        fuse_op_nodes.begin() + 1, fuse_op_nodes.end());

    GraphSafeRemoveNodes(graph, marked_fuse_op_nodes);
    ++fusion_count;
  };

  gpd(graph, handler);
  return fusion_count;
}

void FuseMultiTransformerLayerPass::ApplyImpl(Graph* graph) const {
  FusePassBase::Init(name_scope_, graph);
  auto* scope = param_scope();
  PADDLE_ENFORCE_NOT_NULL(
      scope,
      common::errors::Fatal("During the fuse_multi_transformer_layer pass, "
                            "The scope should not be null."));

  VLOG(3) << "Running fuse_multi_transformer_layer_pass.";
  if (graph->IsMainGraph()) {
    VLOG(3) << "The ID of block running fuse_multi_transformer_layer_pass is: "
               "0(main_graph)";
  } else {
    VLOG(3) << "The ID of block running fuse_multi_transformer_layer_pass is: "
            << graph->GetBlockId();
  }

  int fusion_count = BuildFusion(graph, name_scope_, scope);

  AddStatis(fusion_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(fuse_multi_transformer_layer_pass,
              paddle::framework::ir::FuseMultiTransformerLayerPass);
