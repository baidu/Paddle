/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. */

#include "paddle/fluid/framework/ir/squared_mat_sub_fuse_pass.h"

#include <string>

#include "paddle/fluid/framework/op_version_registry.h"

namespace paddle::framework::ir {

PDNode* BuildSquaredMatSubPattern(PDPattern* pattern,
                                  const std::string& name_scope) {
  auto var_is_op_input = [=](Node* x,
                             const std::string& op_type,
                             const std::string& arg_name = "") -> bool {
    if (!(x && x->IsVar())) {
      return false;
    }
    for (auto* op : x->outputs) {
      if (op && op->IsOp() && op->Op() && op->Op()->Type() == op_type) {
        if (arg_name.empty()) {
          return true;
        }
        for (auto& name : op->Op()->Input(arg_name)) {
          if (name == x->Name()) {
            return true;
          }
        }
      }
    }
    return false;
  };

  auto var_is_op_only_output = [](Node* x, const std::string& op_type) -> bool {
    return x && x->IsVar() && x->inputs.size() == 1 && x->inputs[0] &&
           x->inputs[0]->IsOp() && x->inputs[0]->Op()->Type() == op_type &&
           x->inputs[0]->outputs.size() == 1;
  };

  auto next_op = [=](Node* x, const std::string& op_type) -> Node* {
    if (!(x && x->IsVar())) {
      return nullptr;
    }
    for (auto* op : x->outputs) {
      if (op && op->IsOp() && op->Op() && op->Op()->Type() == op_type) {
        return op;
      }
    }
    return nullptr;
  };

  auto get_op_input_var = [=](Node* x, const std::string& arg_name) -> Node* {
    if (!(x && x->IsOp())) {
      return nullptr;
    }
    for (auto* var : x->inputs) {
      for (auto const& name : x->Op()->Input(arg_name)) {
        if (var->Name() == name) {
          return var;
        }
      }
    }
    return nullptr;
  };

  auto is_fusion_input_var = [=](Node* x, const std::string& arg_name) {
    bool basic = (var_is_op_input(x, "matmul_v2", arg_name) ||
                  var_is_op_input(x, "matmul", arg_name)) &&
                 var_is_op_input(x, "square", "X");
    if (!basic) {
      return false;
    }
    auto* squared_x_op = next_op(x, "square");
    if (!(squared_x_op && squared_x_op->outputs.size() == 1)) {
      return false;
    }
    auto* squared_x = squared_x_op->outputs[0];
    bool next_is_matmul_from_arg =
        (var_is_op_input(squared_x, "matmul_v2", arg_name) ||
         var_is_op_input(squared_x, "matmul", arg_name)) &&
        squared_x->outputs.size() == 1 &&
        squared_x->outputs[0]->outputs.size() == 1;
    if (!next_is_matmul_from_arg) {
      return false;
    }
    auto* sub_y_in = squared_x->outputs[0]->outputs[0];
    return var_is_op_input(sub_y_in, "elementwise_sub", "Y") &&
           sub_y_in->outputs[0]->outputs.size() == 1 &&
           var_is_op_input(sub_y_in->outputs[0]->outputs[0], "elementwise_mul");
  };

  auto is_fusion_first_mul_out = [=](Node* x) -> bool {
    bool input_is_matmul_op = x && x->inputs.size() == 1 &&
                              x->inputs[0]->IsOp() &&
                              (x->inputs[0]->Op()->Type() == "matmul_v2" ||
                               x->inputs[0]->Op()->Type() == "matmul");
    if (!input_is_matmul_op) {
      return false;
    }
    auto* mat_x = get_op_input_var(x->inputs[0], "X");
    auto* mat_y = get_op_input_var(x->inputs[0], "Y");
    bool input_mul_is_valid = mat_x && is_fusion_input_var(mat_x, "X") &&
                              mat_y && is_fusion_input_var(mat_y, "Y");
    if (!input_mul_is_valid) {
      return false;
    }

    bool next_is_square = var_is_op_input(x, "square", "X") &&
                          x->outputs.size() == 1 &&
                          x->outputs[0]->outputs.size() == 1;
    if (!next_is_square) {
      return false;
    }
    auto* sub_x_in = x->outputs[0]->outputs[0];
    return var_is_op_input(sub_x_in, "elementwise_sub", "X") &&
           sub_x_in->outputs[0]->outputs.size() == 1 &&
           var_is_op_input(sub_x_in->outputs[0]->outputs[0], "elementwise_mul");
  };

  auto* x = pattern->NewNode(
      [=](Node* x) { return is_fusion_input_var(x, "X"); }, name_scope + "/x");

  auto* y = pattern->NewNode(
      [=](Node* x) { return is_fusion_input_var(x, "Y"); }, name_scope + "/y");

  auto* square_x_op = pattern->NewNode(
      [=](Node* x) {
        return x && x->IsOp() && x->Op()->Type() == "square" &&
               is_fusion_input_var(x->inputs[0], "X");
      },
      name_scope + "/squared_x_op");

  auto* square_y_op = pattern->NewNode(
      [=](Node* x) {
        return x && x->IsOp() && x->Op()->Type() == "square" &&
               is_fusion_input_var(x->inputs[0], "Y");
      },
      name_scope + "/squared_y_op");

  auto* squared_x = pattern->NewNode(
      [=](Node* x) {
        return x && x->inputs.size() == 1 && x->inputs[0]->inputs.size() == 1 &&
               is_fusion_input_var(x->inputs[0]->inputs[0], "X");
      },
      name_scope + "/squared_x");

  auto* squared_y = pattern->NewNode(
      [=](Node* x) {
        return x && x->inputs.size() == 1 && x->inputs[0]->inputs.size() == 1 &&
               is_fusion_input_var(x->inputs[0]->inputs[0], "Y");
      },
      name_scope + "/squared_y");

  auto* matmuled_xy =
      pattern->NewNode([=](Node* x) { return is_fusion_first_mul_out(x); },
                       name_scope + "/matmuled_xy");

  auto* matmul_xy_op = pattern->NewNode(
      [=](Node* x) {
        return x && x->IsOp() &&
               (x->Op()->Type() == "matmul_v2" ||
                x->Op()->Type() == "matmul") &&
               is_fusion_first_mul_out(x->outputs[0]);
      },
      name_scope + "/matmul_xy_op");

  auto* square_matmuled_xy_op = pattern->NewNode(
      [=](Node* x) {
        return x && x->IsOp() && x->Op()->Type() == "square" &&
               is_fusion_first_mul_out(x->inputs[0]);
      },
      name_scope + "/square_matmuled_xy_op");

  auto* squared_xmuly = pattern->NewNode(
      [=](Node* x) {
        return x && x->IsVar() && x->inputs.size() == 1 &&
               x->inputs[0]->IsOp() && x->inputs[0]->Op()->Type() == "square" &&
               is_fusion_first_mul_out(x->inputs[0]->inputs[0]);
      },
      name_scope + "/squared_xmuly");

  auto is_fusion_mat_squared_x_y_op_out = [=](Node* x) -> bool {
    bool basic = x && x->IsVar() && x->inputs.size() == 1 &&
                 x->inputs[0]->IsOp() &&
                 (x->inputs[0]->Op()->Type() == "matmul_v2" ||
                  x->inputs[0]->Op()->Type() == "matmul");
    if (!basic) {
      return false;
    }
    auto* sqx = get_op_input_var(x->inputs[0], "X");
    auto* sqy = get_op_input_var(x->inputs[0], "Y");

    return var_is_op_only_output(sqx, "square") &&
           var_is_op_only_output(sqy, "square") && sqx->inputs[0] &&
           sqx->inputs[0]->inputs.size() == 1 &&
           is_fusion_input_var(sqx->inputs[0]->inputs[0], "X") &&
           sqy->inputs[0] && sqy->inputs[0]->inputs.size() == 1 &&
           is_fusion_input_var(sqy->inputs[0]->inputs[0], "Y");
  };

  auto* matmul_squared_x_y_op = pattern->NewNode(
      [=](Node* x) {
        return x && x->IsOp() &&
               (x->Op()->Type() == "matmul_v2" ||
                x->Op()->Type() == "matmul") &&
               is_fusion_mat_squared_x_y_op_out(x->outputs[0]);
      },
      name_scope + "/matmul_squared_x_y_op");

  auto* mat_squared_x_y_op_out = pattern->NewNode(
      [=](Node* x) { return is_fusion_mat_squared_x_y_op_out(x); },
      name_scope + "/mat_squared_x_y_op_out");

  auto is_fusion_sub_op = [=](Node* x) -> bool {
    bool is_sub_op = x && x->IsOp() && x->Op()->Type() == "elementwise_sub";
    if (!is_sub_op) {
      return false;
    }
    auto* matmul_sqx_sqy_var = get_op_input_var(x, "Y");
    return is_fusion_mat_squared_x_y_op_out(matmul_sqx_sqy_var);
  };

  auto* sub_op = pattern->NewNode([=](Node* x) { return is_fusion_sub_op(x); },
                                  name_scope + "/sub_op");

  auto* sub_op_out = pattern->NewNode(
      [=](Node* x) {
        return x && x->IsVar() && x->inputs.size() == 1 &&
               is_fusion_sub_op(x->inputs[0]);
      },
      name_scope + "/sub_op_out");

  auto is_fusion_element_op = [=](Node* x) -> bool {
    bool is_elemul_op = x && x->IsOp() && x->Op()->Type() == "elementwise_mul";
    if (!is_elemul_op) {
      return false;
    }
    for (auto* in : x->inputs) {
      if (in && !in->inputs.empty() && in->inputs[0] &&
          is_fusion_sub_op(in->inputs[0])) {
        return true;
      }
    }
    return false;
  };

  auto* elementmul_op =
      pattern->NewNode([=](Node* x) { return is_fusion_element_op(x); },
                       name_scope + "/elementmul_op");

  auto* constant_op = pattern->NewNode(
      [=](Node* x) {
        return x && x->IsOp() && x->Op()->Type() == "fill_constant" &&
               x->outputs.size() == 1 &&
               is_fusion_element_op(x->outputs[0]->outputs[0]);
      },
      name_scope + "/fill_constant_op");

  auto* constant_op_out = pattern->NewNode(
      [=](Node* x) {
        return x && x->IsVar() && var_is_op_input(x, "elementwise_mul") &&
               !x->inputs.empty() && x->inputs[0] && x->inputs[0]->IsOp() &&
               x->inputs[0]->Op()->Type() == "fill_constant" && x->outputs[0] &&
               is_fusion_element_op(x->outputs[0]);
      },
      name_scope + "/constant_op_out");

  auto* last_out_var = pattern->NewNode(
      [=](Node* x) {
        return var_is_op_only_output(x, "elementwise_mul") &&
               is_fusion_element_op(x->inputs[0]);
      },
      name_scope + "/out");

  square_x_op->LinksFrom({x}).LinksTo({squared_x});
  square_y_op->LinksFrom({y}).LinksTo({squared_y});
  matmul_xy_op->LinksFrom({x, y}).LinksTo({matmuled_xy});
  matmul_squared_x_y_op->LinksFrom({squared_x, squared_y})
      .LinksTo({mat_squared_x_y_op_out});
  square_matmuled_xy_op->LinksFrom({matmuled_xy}).LinksTo({squared_xmuly});
  sub_op->LinksFrom({squared_xmuly, mat_squared_x_y_op_out})
      .LinksTo({sub_op_out});
  constant_op->LinksFrom({}).LinksTo({constant_op_out});
  elementmul_op->LinksFrom({constant_op_out, sub_op_out})
      .LinksTo({last_out_var});

  return last_out_var;
}

static int BuildFusion(Graph* graph,
                       const std::string& name_scope,
                       const SquaredMatSubFusePass* pass) {
  GraphPatternDetector gpd;
  auto* pattern = gpd.mutable_pattern();

  BuildSquaredMatSubPattern(pattern, name_scope);

  auto retrieve_node = [](const std::string& name,
                          const GraphPatternDetector::subgraph_t& subgraph,
                          const PDPattern& pat) -> Node* {
    PADDLE_ENFORCE_GT(subgraph.count(pat.RetrieveNode(name)),
                      0,
                      common::errors::NotFound("Pattern has no node called %s.",
                                               name.c_str()));
    Node* p = subgraph.at(pat.RetrieveNode(name));
    PADDLE_ENFORCE_NOT_NULL(
        p, common::errors::NotFound("Subgraph has no node %s.", name.c_str()));
    return p;
  };

  int fusion_count{0};
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    LOG(INFO) << "handle square mat sub fuse";
    if (!pass->IsAcceptable(subgraph, g)) {
      LOG(WARNING) << "Pass in op compat failed.";
      return;
    }

    auto& fused_pattern = gpd.pattern();

    auto* matx = retrieve_node(name_scope + "/x", subgraph, fused_pattern);
    auto* maty = retrieve_node(name_scope + "/y", subgraph, fused_pattern);
    auto* squaredx =
        retrieve_node(name_scope + "/squared_x", subgraph, fused_pattern);
    auto* squaredy =
        retrieve_node(name_scope + "/squared_y", subgraph, fused_pattern);
    auto* squaredxy =
        retrieve_node(name_scope + "/squared_xmuly", subgraph, fused_pattern);
    auto* last_out_var =
        retrieve_node(name_scope + "/out", subgraph, fused_pattern);
    auto* fill_constant_op = retrieve_node(
        name_scope + "/fill_constant_op", subgraph, fused_pattern);

    // Create New OpDesc
    OpDesc op_desc;
    op_desc.SetType("fusion_squared_mat_sub");
    op_desc.SetInput("X", {matx->Name()});
    op_desc.SetInput("Y", {maty->Name()});
    op_desc.SetOutput("SquaredX", {squaredx->Name()});
    op_desc.SetOutput("SquaredY", {squaredy->Name()});
    op_desc.SetOutput("SquaredXY", {squaredxy->Name()});
    op_desc.SetOutput("Out", {last_out_var->Name()});
    op_desc.SetAttr("scalar", fill_constant_op->Op()->GetAttr("value"));

    auto* op = graph->CreateOpNode(&op_desc);
    IR_NODE_LINK_TO(matx, op);
    IR_NODE_LINK_TO(maty, op);
    IR_NODE_LINK_TO(op, squaredx);
    IR_NODE_LINK_TO(op, squaredy);
    IR_NODE_LINK_TO(op, squaredxy);
    IR_NODE_LINK_TO(op, last_out_var);

    std::unordered_set<const Node*> marked_nodes;
    for (auto& item : subgraph) {
      marked_nodes.insert(item.second);
    }

    marked_nodes.erase(matx);
    marked_nodes.erase(maty);
    marked_nodes.erase(squaredx);
    marked_nodes.erase(squaredy);
    marked_nodes.erase(squaredxy);
    marked_nodes.erase(last_out_var);
    GraphSafeRemoveNodes(graph, marked_nodes);
    ++fusion_count;
  };
  gpd(graph, handler);
  return fusion_count;
}

SquaredMatSubFusePass::SquaredMatSubFusePass() {
  AddOpCompat(OpCompat("square"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End();

  AddOpCompat(OpCompat("matmul"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("Y")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("alpha")
      .IsNumEQ(1.0f)
      .End()
      .AddAttr("transpose_X")
      .IsBoolEQ(false)
      .End()
      .AddAttr("transpose_Y")
      .IsBoolEQ(false)
      .End();

  AddOpCompat(OpCompat("matmul_v2"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("Y")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("trans_x")
      .IsBoolEQ(false)
      .End()
      .AddAttr("trans_y")
      .IsBoolEQ(false)
      .End();

  AddOpCompat(OpCompat("elementwise_sub"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("Y")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("axis")
      .IsIntIn({-1, 0})
      .End();

  AddOpCompat(OpCompat("elementwise_mul"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("Y")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("axis")
      .IsIntIn({-1, 0})
      .End();

  AddOpCompat(OpCompat("fill_constant"))
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("dtype")
      .IsNumGE(0)
      .IsNumLE(25)
      .End()
      .AddAttr("shape")
      .End()
      // type:float, there is no restriction
      .AddAttr("value")
      .End()
      .AddAttr("str_value")
      .IsStringEQ("")
      .IsOptional()
      .End();
}

// to use IsCompat
bool SquaredMatSubFusePass::IsAcceptable(
    const GraphPatternDetector::subgraph_t& subgraph, Graph* g) const {
  return IsCompat(subgraph, g);
}

void SquaredMatSubFusePass::ApplyImpl(ir::Graph* graph) const {
  FusePassBase::Init(name_scope_, graph);
  int fusion_count = BuildFusion(graph, name_scope_, this);
  AddStatis(fusion_count);
}

}  // namespace paddle::framework::ir

REGISTER_PASS(squared_mat_sub_fuse_pass,
              paddle::framework::ir::SquaredMatSubFusePass);
REGISTER_PASS_CAPABILITY(squared_mat_sub_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .LE("matmul", 1)
            .EQ("matmul_v2", 0)
            .EQ("square", 0)
            .LE("elementwise_mul", 1)
            .LE("elementwise_sub", 1)
            .LE("fill_constant", 2)
            .EQ("fusion_squared_mat_sub", 0));
