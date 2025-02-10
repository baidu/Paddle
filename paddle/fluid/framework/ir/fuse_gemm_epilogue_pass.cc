// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
// Copyright (c) 2022 NVIDIA Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/fuse_gemm_epilogue_pass.h"

#include <string>
#include "paddle/fluid/framework/details/multi_devices_helper.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle::framework::ir {

static void GetTransposeAttrsFromOp(const OpDesc &op,
                                    bool *trans_x,
                                    bool *trans_y) {
  *trans_x = PADDLE_GET_CONST(bool, op.GetAttr("trans_x"));
  *trans_y = PADDLE_GET_CONST(bool, op.GetAttr("trans_y"));
}

void FuseGemmEpiloguePass::ApplyImpl(ir::Graph *graph) const {
  EpiloguePassActivationCache cache;

  graph = FuseLinearActFwd(graph, {"relu", "gelu"}, false, false, &cache);
  graph = FuseLinearActFwd(graph, {"relu"}, true, true, &cache);
  graph = FuseLinearActFwd(graph, {"gelu"}, true, false, &cache);
  graph = FuseLinearFwd(graph, false);
  graph = FuseLinearFwd(graph, true);
  graph = FuseLinearActBwd(graph, {"relu_grad"}, true, &cache);
  graph = FuseLinearActBwd(graph, {"gelu_grad"}, false, &cache);
  graph = FuseLinearBwd(graph, false);
  graph = FuseLinearBwd(graph, true);  // NOLINT
}

ir::Graph *FuseGemmEpiloguePass::FuseLinearFwd(ir::Graph *graph,
                                               bool is_training) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::InvalidArgument("Graph cannot be nullptr."));
  const std::string scope_name("gemm_epilogue");
  FusePassBase::Init(scope_name, graph);

  GraphPatternDetector gpd;
  auto *x = gpd.mutable_pattern()
                ->NewNode(patterns::PDNodeName(scope_name, "x"))
                ->AsInput()
                ->assert_is_op_input("matmul_v2", "X");
  patterns::LinearAct linear_act_pattern(gpd.mutable_pattern(), "linear_act");

  linear_act_pattern(x, {}, is_training, false);

  int found_linear_count = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t &subgraph,
                     Graph *g) {
    VLOG(4) << "handle LinearAct fuse";

    GET_IR_NODE_FROM_SUBGRAPH(matmul_op, matmul, linear_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(matmul_w, matmul_w, linear_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(matmul_out, matmul_out, linear_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(ele_add_op, ele_add, linear_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(ele_bias, ele_bias, linear_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(ele_out, elewise_add_out, linear_act_pattern);

    std::vector<int64_t> matmul_x_shape = subgraph.at(x)->Var()->GetShape();
    std::vector<int64_t> matmul_w_shape = matmul_w->Var()->GetShape();

    // Note (Ming Huang): We only support matmul_v2 from paddle.nn.Linear
    // currently. The conditions below are used to verify whether matmul_v2
    // is created by paddle.nn.Linear
    auto matmul_op_desc = matmul_op->Op();
    if (!IsGemmFromLinear_(matmul_x_shape, matmul_w_shape)) return;

    bool trans_x = false, trans_y = false;
    GetTransposeAttrsFromOp(*matmul_op_desc, &trans_x, &trans_y);

    OpDesc fused_gemm_epilogue_op_desc(matmul_op->Op()->Block());
    std::string activation = "none";
    fused_gemm_epilogue_op_desc.SetType("fused_gemm_epilogue");
    fused_gemm_epilogue_op_desc.SetInput("X", {subgraph.at(x)->Name()});
    fused_gemm_epilogue_op_desc.SetInput("Y", {matmul_w->Name()});
    fused_gemm_epilogue_op_desc.SetInput("Bias", {ele_bias->Name()});
    fused_gemm_epilogue_op_desc.SetOutput("Out", {ele_out->Name()});
    fused_gemm_epilogue_op_desc.SetAttr("activation", activation);
    fused_gemm_epilogue_op_desc.SetAttr("op_role",
                                        matmul_op_desc->GetAttr("op_role"));
    fused_gemm_epilogue_op_desc.SetAttr("trans_x", trans_x);
    fused_gemm_epilogue_op_desc.SetAttr("trans_y", trans_y);
    auto gemm_epilogue_node = g->CreateOpNode(&fused_gemm_epilogue_op_desc);

    IR_NODE_LINK_TO(subgraph.at(x), gemm_epilogue_node);
    IR_NODE_LINK_TO(matmul_w, gemm_epilogue_node);
    IR_NODE_LINK_TO(ele_bias, gemm_epilogue_node);
    IR_NODE_LINK_TO(gemm_epilogue_node, ele_out);

    VLOG(4) << "\n\t " << subgraph.at(x)->Name() << " and " << matmul_w->Name()
            << " -> " << matmul_op->Name() << " -> " << matmul_out->Name()
            << "\n\t " << matmul_out->Name() << " and " << ele_bias->Name()
            << " -> " << ele_add_op->Name() << " -> " << ele_out->Name()
            << "\n\t " << ele_out->Name();

    GraphSafeRemoveNodes(g, {matmul_op, matmul_out, ele_add_op});
    found_linear_count++;
  };

  gpd(graph, handler);

  AddStatis(found_linear_count);
  return graph;
}

ir::Graph *FuseGemmEpiloguePass::FuseLinearActFwd(
    ir::Graph *graph,
    const std::unordered_set<std::string> &act_types,
    bool is_training,
    bool is_act_grad_x_from_act,
    EpiloguePassActivationCache *cache) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::InvalidArgument("Graph cannot be nullptr."));

  const std::string scope_name("gemm_epilogue");
  FusePassBase::Init(scope_name, graph);

  GraphPatternDetector gpd;
  auto *x = gpd.mutable_pattern()
                ->NewNode(patterns::PDNodeName(scope_name, "x"))
                ->AsInput()
                ->assert_is_op_input("matmul_v2", "X");
  patterns::LinearAct linear_act_pattern(gpd.mutable_pattern(), "linear_act");

  linear_act_pattern(x, act_types, is_training, is_act_grad_x_from_act);

  int found_linear_act_count = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t &subgraph,
                     Graph *g) {
    VLOG(4) << "handle LinearAct fuse";

    GET_IR_NODE_FROM_SUBGRAPH(matmul_op, matmul, linear_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(matmul_w, matmul_w, linear_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(matmul_out, matmul_out, linear_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(ele_add_op, ele_add, linear_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(ele_bias, ele_bias, linear_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(ele_out, elewise_add_out, linear_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(act_op, act, linear_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(act_out, act_out, linear_act_pattern);

    std::vector<int64_t> matmul_x_shape = subgraph.at(x)->Var()->GetShape();
    std::vector<int64_t> matmul_w_shape = matmul_w->Var()->GetShape();

    // Note (Ming Huang): We only support matmul_v2 from paddle.nn.Linear
    // currently. The conditions below are used to verify whether matmul_v2
    // is created by paddle.nn.Linear
    auto matmul_op_desc = matmul_op->Op();
    if (!IsGemmFromLinear_(matmul_x_shape, matmul_w_shape)) return;

    auto activation = act_op->Op()->Type();

    bool trans_x = false, trans_y = false;
    GetTransposeAttrsFromOp(*matmul_op_desc, &trans_x, &trans_y);

    OpDesc fused_gemm_epilogue_op_desc(matmul_op->Op()->Block());
    fused_gemm_epilogue_op_desc.SetType("fused_gemm_epilogue");
    fused_gemm_epilogue_op_desc.SetInput("X", {subgraph.at(x)->Name()});
    fused_gemm_epilogue_op_desc.SetInput("Y", {matmul_w->Name()});
    fused_gemm_epilogue_op_desc.SetInput("Bias", {ele_bias->Name()});
    fused_gemm_epilogue_op_desc.SetOutput("Out", {act_out->Name()});
    fused_gemm_epilogue_op_desc.SetAttr("activation", activation);
    fused_gemm_epilogue_op_desc.SetAttr("op_role",
                                        matmul_op_desc->GetAttr("op_role"));
    fused_gemm_epilogue_op_desc.SetAttr("trans_x", trans_x);
    fused_gemm_epilogue_op_desc.SetAttr("trans_y", trans_y);

    auto gemm_epilogue_node = g->CreateOpNode(&fused_gemm_epilogue_op_desc);

    IR_NODE_LINK_TO(subgraph.at(x), gemm_epilogue_node);
    IR_NODE_LINK_TO(matmul_w, gemm_epilogue_node);
    IR_NODE_LINK_TO(ele_bias, gemm_epilogue_node);
    IR_NODE_LINK_TO(gemm_epilogue_node, act_out);

    // Only need to check weight.shape[1] for auxiliary pointer
    // and mark it the act op is fused for backward epilogue fusion.
    // That because cuBlasLt epilogue's restriction.
    if (is_training) {
      int divisor_of_n = activation == "relu" ? 128 : 8;
      if (matmul_w_shape[1] % divisor_of_n) return;

      VarDesc reserve_space(patterns::PDNodeName(scope_name, "ReserveSpace"));
      auto *reserve_space_node = g->CreateVarNode(&reserve_space);

      cache->InsertFusedActivation(
          GetReserveSpaceCacheKey(act_out->Var()->Name(), g->GetBlockId()),
          reserve_space_node);

      gemm_epilogue_node->Op()->SetOutput("ReserveSpace",
                                          {reserve_space_node->Name()});

      if (!is_act_grad_x_from_act) {
        GET_IR_NODE_FROM_SUBGRAPH(act_grad_op, act_grad, linear_act_pattern);
        act_grad_op->Op()->RenameInput(ele_out->Name(),
                                       reserve_space_node->Name());
        IR_NODE_LINK_TO(reserve_space_node, act_grad_op);
      }
      IR_NODE_LINK_TO(gemm_epilogue_node, reserve_space_node);
    }

    VLOG(4) << "\n\t " << subgraph.at(x)->Name() << " and " << matmul_w->Name()
            << " -> " << matmul_op->Name() << " -> " << matmul_out->Name()
            << "\n\t " << matmul_out->Name() << " and " << ele_bias->Name()
            << " -> " << ele_add_op->Name() << " -> " << ele_out->Name()
            << "\n\t " << ele_out->Name() << " -> " << act_op->Name() << " -> "
            << act_out->Name();

    GraphSafeRemoveNodes(g,
                         {matmul_op, matmul_out, ele_add_op, ele_out, act_op});
    found_linear_act_count++;
  };

  gpd(graph, handler);

  AddStatis(found_linear_act_count);
  return graph;
}

ir::Graph *FuseGemmEpiloguePass::FuseLinearBwd(ir::Graph *graph,
                                               bool without_x_gradient) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::InvalidArgument("Graph cannot be nullptr."));
  const std::string scope_name("gemm_epilogue");
  FusePassBase::Init(scope_name, graph);

  GraphPatternDetector gpd;
  auto *dout =
      gpd.mutable_pattern()
          ->NewNode(patterns::PDNodeName(scope_name, "dout"))
          ->AsInput()
          ->assert_is_op_input("elementwise_add_grad", GradVarName("Out"));

  patterns::ElewiseAddMatmulAct ele_add_matmul_act_pattern(
      gpd.mutable_pattern(), "ele_add_matmul_act");
  ele_add_matmul_act_pattern(dout, {}, without_x_gradient, false);

  int found_ele_add_matmul_act_count = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t &subgraph,
                     Graph *g) {
    VLOG(4) << "handle ElewiseAddMatmulAct fuse";

    GET_IR_NODE_FROM_SUBGRAPH(
        ele_add_grad_op, ele_add_grad, ele_add_matmul_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        ele_grad_bias, ele_grad_bias, ele_add_matmul_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        ele_grad_dx, ele_grad_dx, ele_add_matmul_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        ele_grad_dbias, ele_grad_dbias, ele_add_matmul_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        matmul_grad_op, matmul_grad, ele_add_matmul_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        matmul_grad_x, matmul_grad_x, ele_add_matmul_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        matmul_grad_w, matmul_grad_w, ele_add_matmul_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        matmul_grad_dw, matmul_grad_dw, ele_add_matmul_act_pattern);

    Node *matmul_grad_dx = nullptr;
    if (!without_x_gradient) {
      GET_IR_NODE_FROM_SUBGRAPH(
          matmul_grad_dx_ptr, matmul_grad_dx, ele_add_matmul_act_pattern);
      matmul_grad_dx = matmul_grad_dx_ptr;
    }

    std::vector<int64_t> matmul_grad_x_shape = matmul_grad_x->Var()->GetShape();
    std::vector<int64_t> matmul_grad_w_shape = matmul_grad_w->Var()->GetShape();

    // Note (Ming Huang): We only support matmul_v2_grad from paddle.nn.Linear
    // currently. The conditions below are used to verify whether matmul_v2
    // is created by paddle.nn.Linear
    auto matmul_grad_op_desc = matmul_grad_op->Op();
    if (!IsGemmFromLinear_(matmul_grad_x_shape, matmul_grad_w_shape)) return;

    bool trans_x = false, trans_y = false;
    GetTransposeAttrsFromOp(*matmul_grad_op_desc, &trans_x, &trans_y);

    OpDesc fused_gemm_epilogue_grad_op_desc(ele_add_grad_op->Op()->Block());
    std::string activation_grad = "none";
    fused_gemm_epilogue_grad_op_desc.SetType("fused_gemm_epilogue_grad");
    fused_gemm_epilogue_grad_op_desc.SetInput("DOut",
                                              {subgraph.at(dout)->Name()});
    fused_gemm_epilogue_grad_op_desc.SetInput("X", {matmul_grad_x->Name()});
    fused_gemm_epilogue_grad_op_desc.SetInput("Y", {matmul_grad_w->Name()});
    if (matmul_grad_dx) {
      fused_gemm_epilogue_grad_op_desc.SetOutput("DX",
                                                 {matmul_grad_dx->Name()});
    }
    fused_gemm_epilogue_grad_op_desc.SetOutput("DY", {matmul_grad_dw->Name()});
    fused_gemm_epilogue_grad_op_desc.SetOutput("DBias",
                                               {ele_grad_dbias->Name()});
    fused_gemm_epilogue_grad_op_desc.SetAttr("activation_grad",
                                             activation_grad);
    fused_gemm_epilogue_grad_op_desc.SetAttr(
        "op_role", matmul_grad_op_desc->GetAttr("op_role"));
    fused_gemm_epilogue_grad_op_desc.SetAttr("trans_x", trans_x);
    fused_gemm_epilogue_grad_op_desc.SetAttr("trans_y", trans_y);
    auto matmul_grad_op_role_val =
        details::GetOpRoleVarsOrEmpty(*(matmul_grad_op->Op()));
    auto ele_add_grad_op_role_val =
        details::GetOpRoleVarsOrEmpty(*(ele_add_grad_op->Op()));
    std::vector<std::string> fused_gemm_epilogue_grad_op_role_var;
    fused_gemm_epilogue_grad_op_role_var.reserve(
        matmul_grad_op_role_val.size());
    for (auto const &i : matmul_grad_op_role_val) {
      fused_gemm_epilogue_grad_op_role_var.push_back(i);
    }
    for (auto const &i : ele_add_grad_op_role_val) {
      fused_gemm_epilogue_grad_op_role_var.push_back(i);
    }
    fused_gemm_epilogue_grad_op_desc.SetAttr(
        "op_role_var", fused_gemm_epilogue_grad_op_role_var);

    auto gemm_epilogue_grad_node =
        g->CreateOpNode(&fused_gemm_epilogue_grad_op_desc);

    IR_NODE_LINK_TO(subgraph.at(dout), gemm_epilogue_grad_node);
    IR_NODE_LINK_TO(matmul_grad_x, gemm_epilogue_grad_node);
    IR_NODE_LINK_TO(matmul_grad_w, gemm_epilogue_grad_node);
    IR_NODE_LINK_TO(ele_grad_bias, gemm_epilogue_grad_node);
    IR_NODE_LINK_TO(gemm_epilogue_grad_node, matmul_grad_dw);
    IR_NODE_LINK_TO(gemm_epilogue_grad_node, ele_grad_dbias);
    if (matmul_grad_dx) {
      IR_NODE_LINK_TO(gemm_epilogue_grad_node, matmul_grad_dx);
    }

    std::string matmul_grad_dx_name =
        matmul_grad_dx != nullptr ? matmul_grad_dx->Name() : " ";
    VLOG(4) << "\n\t " << subgraph.at(dout)->Name() << " and "
            << ele_grad_bias->Name() << " -> " << ele_add_grad_op->Name()
            << " -> " << ele_grad_dx->Name() << " and "
            << ele_grad_dbias->Name() << "\n\t " << ele_grad_dx->Name() << ", "
            << matmul_grad_x->Name() << " and " << matmul_grad_w->Name()
            << " -> " << matmul_grad_op->Name() << " -> "
            << matmul_grad_w->Name() << " and " << matmul_grad_dx_name;

    GraphSafeRemoveNodes(g, {ele_add_grad_op, ele_grad_dx, matmul_grad_op});
    found_ele_add_matmul_act_count++;
  };

  gpd(graph, handler);

  AddStatis(found_ele_add_matmul_act_count);
  return graph;
}

ir::Graph *FuseGemmEpiloguePass::FuseLinearActBwd(
    ir::Graph *graph,
    const std::unordered_set<std::string> &act_grad_types,
    bool is_act_grad_x_from_act,
    EpiloguePassActivationCache *cache) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::InvalidArgument("Graph cannot be nullptr."));
  const std::string scope_name("gemm_epilogue");
  FusePassBase::Init(scope_name, graph);

  GraphPatternDetector gpd;
  auto *dout =
      gpd.mutable_pattern()
          ->NewNode(patterns::PDNodeName(scope_name, "dout"))
          ->AsInput()
          ->assert_is_op_input("elementwise_add_grad", GradVarName("Out"));

  patterns::ElewiseAddMatmulAct ele_add_matmul_act_pattern(
      gpd.mutable_pattern(), "ele_add_matmul_act");
  ele_add_matmul_act_pattern(
      dout, act_grad_types, false, is_act_grad_x_from_act);

  int found_ele_add_matmul_act_count = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t &subgraph,
                     Graph *g) {
    VLOG(4) << "handle ElewiseAddMatmulAct fuse";

    GET_IR_NODE_FROM_SUBGRAPH(
        ele_add_grad_op, ele_add_grad, ele_add_matmul_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        ele_grad_bias, ele_grad_bias, ele_add_matmul_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        ele_grad_dx, ele_grad_dx, ele_add_matmul_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        ele_grad_dbias, ele_grad_dbias, ele_add_matmul_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        matmul_grad_op, matmul_grad, ele_add_matmul_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        matmul_grad_x, matmul_grad_x, ele_add_matmul_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        matmul_grad_w, matmul_grad_w, ele_add_matmul_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        matmul_grad_dx, matmul_grad_dx, ele_add_matmul_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        matmul_grad_dw, matmul_grad_dw, ele_add_matmul_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        act_grad_op, act_grad, ele_add_matmul_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        act_grad_dx, act_grad_dx, ele_add_matmul_act_pattern);

    auto key =
        GetReserveSpaceCacheKey(matmul_grad_x->Var()->Name(), g->GetBlockId());
    if (!cache->HasFusedActivation(key)) {
      return;
    }
    auto *reserve_space_node = cache->GetFusedActivationSpace(key);

    std::vector<int64_t> matmul_grad_x_shape = matmul_grad_x->Var()->GetShape();
    std::vector<int64_t> matmul_grad_w_shape = matmul_grad_w->Var()->GetShape();

    // Note (Ming Huang): We only support matmul_v2_grad from paddle.nn.Linear
    // currently. The conditions below are used to verify whether matmul_v2
    // is created by paddle.nn.Linear
    auto matmul_grad_op_desc = matmul_grad_op->Op();
    if (!IsGemmFromLinear_(matmul_grad_x_shape, matmul_grad_w_shape)) return;

    auto activation_grad = act_grad_op->Op()->Type();

    bool trans_x = false, trans_y = false;
    GetTransposeAttrsFromOp(*matmul_grad_op_desc, &trans_x, &trans_y);
    OpDesc fused_gemm_epilogue_grad_op_desc(ele_add_grad_op->Op()->Block());
    fused_gemm_epilogue_grad_op_desc.SetType("fused_gemm_epilogue_grad");
    fused_gemm_epilogue_grad_op_desc.SetInput("DOut",
                                              {subgraph.at(dout)->Name()});
    fused_gemm_epilogue_grad_op_desc.SetInput("X", {matmul_grad_x->Name()});
    fused_gemm_epilogue_grad_op_desc.SetInput("Y", {matmul_grad_w->Name()});
    fused_gemm_epilogue_grad_op_desc.SetInput("ReserveSpace",
                                              {reserve_space_node->Name()});
    fused_gemm_epilogue_grad_op_desc.SetOutput("DX", {act_grad_dx->Name()});
    fused_gemm_epilogue_grad_op_desc.SetOutput("DY", {matmul_grad_dw->Name()});
    fused_gemm_epilogue_grad_op_desc.SetOutput("DBias",
                                               {ele_grad_dbias->Name()});
    fused_gemm_epilogue_grad_op_desc.SetAttr("activation_grad",
                                             activation_grad);
    fused_gemm_epilogue_grad_op_desc.SetAttr(
        "op_role", matmul_grad_op_desc->GetAttr("op_role"));
    fused_gemm_epilogue_grad_op_desc.SetAttr("trans_x", trans_x);
    fused_gemm_epilogue_grad_op_desc.SetAttr("trans_y", trans_y);
    auto matmul_grad_op_role_val =
        details::GetOpRoleVarsOrEmpty(*(matmul_grad_op->Op()));
    auto ele_add_grad_op_role_val =
        details::GetOpRoleVarsOrEmpty(*(ele_add_grad_op->Op()));
    std::vector<std::string> fused_gemm_epilogue_grad_op_role_var;
    fused_gemm_epilogue_grad_op_role_var.reserve(
        matmul_grad_op_role_val.size());
    for (auto const &i : matmul_grad_op_role_val) {
      fused_gemm_epilogue_grad_op_role_var.push_back(i);
    }
    for (auto const &i : ele_add_grad_op_role_val) {
      fused_gemm_epilogue_grad_op_role_var.push_back(i);
    }
    fused_gemm_epilogue_grad_op_desc.SetAttr(
        "op_role_var", fused_gemm_epilogue_grad_op_role_var);

    auto gemm_epilogue_grad_node =
        g->CreateOpNode(&fused_gemm_epilogue_grad_op_desc);

    IR_NODE_LINK_TO(subgraph.at(dout), gemm_epilogue_grad_node);
    IR_NODE_LINK_TO(matmul_grad_x, gemm_epilogue_grad_node);
    IR_NODE_LINK_TO(matmul_grad_w, gemm_epilogue_grad_node);
    IR_NODE_LINK_TO(ele_grad_bias, gemm_epilogue_grad_node);
    IR_NODE_LINK_TO(gemm_epilogue_grad_node, act_grad_dx);
    IR_NODE_LINK_TO(gemm_epilogue_grad_node, matmul_grad_dw);
    IR_NODE_LINK_TO(gemm_epilogue_grad_node, ele_grad_dbias);
    IR_NODE_LINK_TO(reserve_space_node, gemm_epilogue_grad_node);

    VLOG(4) << "\n\t " << subgraph.at(dout)->Name() << " and "
            << ele_grad_bias->Name() << " -> " << ele_add_grad_op->Name()
            << " -> " << ele_grad_dx->Name() << " and "
            << ele_grad_dbias->Name() << "\n\t " << ele_grad_dx->Name() << ", "
            << matmul_grad_x->Name() << " and " << matmul_grad_w->Name()
            << " -> " << matmul_grad_op->Name() << " -> "
            << matmul_grad_dx->Name() << " and " << matmul_grad_w->Name()
            << "\n\t " << matmul_grad_dx->Name() << " -> "
            << act_grad_op->Name() << " -> " << act_grad_dx->Name();

    GraphSafeRemoveNodes(g,
                         {ele_add_grad_op,
                          ele_grad_dx,
                          matmul_grad_op,
                          matmul_grad_dx,
                          act_grad_op});
    found_ele_add_matmul_act_count++;
  };

  gpd(graph, handler);

  AddStatis(found_ele_add_matmul_act_count);
  return graph;
}

bool FuseGemmEpiloguePass::IsGemmFromLinear_(
    const std::vector<int64_t> &x_shape,
    const std::vector<int64_t> &w_shape) const {
  return (w_shape.size() == 2 && x_shape.size() >= 2);
}

}  // namespace paddle::framework::ir

REGISTER_PASS(fuse_gemm_epilogue_pass,
              paddle::framework::ir::FuseGemmEpiloguePass);
