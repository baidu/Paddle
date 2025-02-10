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
#include <gtest/gtest.h>

#include "paddle/fluid/framework/ir/is_test_pass.h"
#if defined _WIN32 || defined __APPLE__
#undef FALSE
#undef TRUE
#endif
namespace paddle::framework::ir {

enum class ISTEST_STATE { FALSE, TRUE, UNSET };

void SetOp(ProgramDesc* prog,
           const std::string& type,
           const std::string& name,
           const std::vector<std::string>& inputs,
           const std::vector<std::string>& outputs,
           bool use_mkldnn = false,
           ISTEST_STATE is_test = ISTEST_STATE::UNSET) {
  auto* op = prog->MutableBlock(0)->AppendOp();
  op->SetType(type);
  op->SetAttr("name", name);
  op->SetInput("X", inputs);
  op->SetOutput("Out", outputs);
  op->SetAttr("use_mkldnn", use_mkldnn);
  if (is_test == ISTEST_STATE::UNSET)
    op->MutableAttrMap()->erase("is_test");
  else if (is_test == ISTEST_STATE::FALSE)
    op->SetAttr("is_test", false);
  else
    op->SetAttr("is_test", true);
}

// a->pool2d->b
// b->relu->c
// c,weights1)->conv2d->d
//
// d->pool2d->e
// e->hard_sigmoid->f
// (f,weights2)->conv2d->g
//
// g->pool2d->h
// h->tanh->i
// (i,weights3)->conv2d->j
ProgramDesc BuildProgramDesc() {
  ProgramDesc prog;
  for (auto& v : std::vector<std::string>({"a",
                                           "b",
                                           "c",
                                           "d",
                                           "e",
                                           "f",
                                           "g",
                                           "h",
                                           "i",
                                           "j",
                                           "weights1",
                                           "weights2",
                                           "weights3"})) {
    auto* var = prog.MutableBlock(0)->Var(v);
    var->SetType(proto::VarType::SELECTED_ROWS);
    if (v == "weights1" || v == "weights2" || v == "weights3") {
      var->SetPersistable(true);
    }
  }

  SetOp(&prog,
        "pool2d",
        "pooling1",
        std::vector<std::string>({"a"}),
        std::vector<std::string>({"b"}),
        true,
        ISTEST_STATE::TRUE);
  SetOp(&prog,
        "relu",
        "activation1",
        std::vector<std::string>({"b"}),
        std::vector<std::string>({"c"}),
        true,
        ISTEST_STATE::TRUE);
  SetOp(&prog,
        "conv2d",
        "conv1",
        std::vector<std::string>({"c", "weights1"}),
        std::vector<std::string>({"d"}),
        true,
        ISTEST_STATE::TRUE);

  SetOp(&prog,
        "pool2d",
        "pooling2",
        std::vector<std::string>({"d"}),
        std::vector<std::string>({"e"}),
        false,
        ISTEST_STATE::FALSE);
  SetOp(&prog,
        "hard_sigmoid",
        "activation2",
        std::vector<std::string>({"e"}),
        std::vector<std::string>({"f"}),
        false,
        ISTEST_STATE::FALSE);
  SetOp(&prog,
        "conv2d",
        "conv2",
        std::vector<std::string>({"f", "weights2"}),
        std::vector<std::string>({"g"}),
        false,
        ISTEST_STATE::FALSE);

  SetOp(&prog,
        "pool2d",
        "pooling3",
        std::vector<std::string>({"g"}),
        std::vector<std::string>({"h"}),
        false,
        ISTEST_STATE::UNSET);
  SetOp(&prog,
        "tanh",
        "activation3",
        std::vector<std::string>({"h"}),
        std::vector<std::string>({"i"}),
        true,
        ISTEST_STATE::UNSET);
  SetOp(&prog,
        "conv2d",
        "conv3",
        std::vector<std::string>({"i", "weights3"}),
        std::vector<std::string>({"j"}),
        false,
        ISTEST_STATE::UNSET);

  return prog;
}

TEST(IsTestPass, basic) {
  auto prog = BuildProgramDesc();

  std::unique_ptr<ir::Graph> graph(new ir::Graph(prog));

  auto pass = PassRegistry::Instance().Get("is_test_pass");

  graph.reset(pass->Apply(graph.release()));

  for (auto* node : graph->Nodes()) {
    if (node->IsOp()) {
      auto* op = node->Op();
      auto op_name = PADDLE_GET_CONST(std::string, op->GetAttr("name"));
      if (op_name == "conv3") {
        ASSERT_FALSE(op->HasAttr("is_test"));
      } else {
        ASSERT_TRUE(op->HasAttr("is_test"));
        EXPECT_TRUE(PADDLE_GET_CONST(bool, op->GetAttr("is_test")));
      }
    }
  }
}

}  // namespace paddle::framework::ir

USE_PASS(is_test_pass);
