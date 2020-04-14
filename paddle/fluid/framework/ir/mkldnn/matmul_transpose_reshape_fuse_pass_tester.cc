// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/mkldnn/matmul_transpose_reshape_fuse_pass.h"
#include <gtest/gtest.h>

namespace paddle {
namespace framework {
namespace ir {

void SetOp(ProgramDesc *prog, const std::string &type,
           const std::vector<std::string> &inputs,
           const std::vector<std::string> &outputs) {
  auto *op = prog->MutableBlock(0)->AppendOp();
  op->SetType(type);
  op->SetInput("X", {inputs[0]});
  op->SetOutput("Out", outputs);
  if (type == "matmul") {
    op->SetInput("Y", {inputs[1]});
    op->SetAttr("use_mkldnn", true);
    op->SetAttr("shape_Out", std::vector<int64_t>({1, 2, 3}));
  }
}

ProgramDesc BuildProgramDesc() {
  ProgramDesc prog;
  for (auto &v :
       std::initializer_list<std::string>({"a1", "a2", "b", "c", "d", "e"})) {
    auto *var = prog.MutableBlock(0)->Var(v);
    var->SetType(proto::VarType::SELECTED_ROWS);
  }

  SetOp(&prog, "matmul", {"a1", "a2"}, {"b"});
  SetOp(&prog, "transpose2", {"b"}, {"c"});
  SetOp(&prog, "reshape2", {"c"}, {"d"});
  SetOp(&prog, "fc", {"d"}, {"e"});

  return prog;
}

void MainTest(const ProgramDesc &prog) {
  std::unique_ptr<ir::Graph> graph(new ir::Graph(prog));

  int original_nodes_num = graph->Nodes().size();

  auto pass =
      PassRegistry::Instance().Get("matmul_transpose_reshape_fuse_pass");
  graph.reset(pass->Apply(graph.release()));

  int current_nodes_num = graph->Nodes().size();
  EXPECT_EQ(original_nodes_num - 4, current_nodes_num);

  for (auto *node : graph->Nodes()) {
    if (node->IsOp()) {
      auto *op = node->Op();
      if (op->Type() == "matmul") {
        ASSERT_TRUE(op->HasAttr("shape_Out"));
      }
    }
  }
}

TEST(MatmulTransposeReshapeFusePass, matmul_inputs) {
  auto prog = BuildProgramDesc();
  MainTest(prog);
}
}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(matmul_transpose_reshape_fuse_pass);
