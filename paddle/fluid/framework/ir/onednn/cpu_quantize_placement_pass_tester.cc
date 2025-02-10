// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/onednn/cpu_quantize_placement_pass.h"
#include "paddle/fluid/platform/onednn_helper.h"

namespace paddle::framework::ir {

void SetOp(ProgramDesc* prog,
           const std::string& type,
           const std::string& name,
           const std::vector<std::string>& inputs,
           const std::vector<std::string>& outputs,
           const std::string& mkldnn_data_type = "float32") {
  auto* op = prog->MutableBlock(0)->AppendOp();

  op->SetType(type);
  op->SetAttr("use_mkldnn", true);
  op->SetAttr("mkldnn_data_type", mkldnn_data_type);

  if (type == "conv2d") {
    op->SetAttr("name", name);
    op->SetInput("Input", {inputs[0]});
    op->SetInput("Filter", {inputs[1]});
    op->SetInput("Bias", {inputs[2]});
  } else if (type == "relu") {
    op->SetInput("X", inputs);
  } else if (type == "concat") {
    op->SetAttr("axis", 1);
    op->SetInput("X", {inputs[0], inputs[1]});
  } else if (type == "pool2d") {
    op->SetInput("X", {inputs[0]});
  } else {
    FAIL() << "Unexpected operator type.";
  }
  op->SetOutput("Out", {outputs[0]});
}

// operator                      mkldnn_data_type
// ---------------------------------------
// (a,b)->concat->c              none
// (c,weights,bias)->conv->f     false
// f->relu->g                    none
// g->pool->h                    false
// (h,weights2,bias2)->conv->k   false
// k->pool->l                    false
ProgramDesc BuildProgramDesc() {
  ProgramDesc prog;

  for (auto& v : std::vector<std::string>({"a",
                                           "b",
                                           "c",
                                           "weights",
                                           "bias",
                                           "f",
                                           "g",
                                           "h",
                                           "weights2",
                                           "bias2",
                                           "k",
                                           "l"})) {
    auto* var = prog.MutableBlock(0)->Var(v);
    var->SetType(proto::VarType::SELECTED_ROWS);
    if (v == "weights" || v == "bias") {
      var->SetPersistable(true);
    }
  }

  SetOp(&prog, "concat", "concat1", {"a", "b"}, {"c"}, "float32");
  SetOp(&prog, "conv2d", "conv1", {"c", "weights", "bias"}, {"f"}, "float32");
  SetOp(&prog, "relu", "relu1", {"f"}, {"g"}, "float32");
  SetOp(&prog, "pool2d", "pool1", {"g"}, {"h"}, "float32");
  SetOp(&prog, "conv2d", "conv2", {"h", "weights2", "bias2"}, {"k"}, "float32");
  SetOp(&prog, "pool2d", "pool2", {"k"}, {"l"}, "float32");

  return prog;
}

void MainTest(std::initializer_list<std::string> quantize_enabled_op_types,
              std::initializer_list<int> quantize_excluded_op_ids,
              unsigned expected_int8_data_type_count) {
  auto prog = BuildProgramDesc();

  std::unique_ptr<ir::Graph> graph(new ir::Graph(prog));

  auto pass = PassRegistry::Instance().Get("cpu_quantize_placement_pass");
  pass->Set("quantize_enabled_op_types",
            new std::unordered_set<std::string>(quantize_enabled_op_types));
  pass->Set("quantize_excluded_op_ids",
            new std::unordered_set<int>(quantize_excluded_op_ids));

  graph.reset(pass->Apply(graph.release()));

  unsigned int8_data_type_count = 0;

  for (auto* node : graph->Nodes()) {
    if (node->IsOp()) {
      if (platform::HasOpINT8DataType(node->Op())) {
        ++int8_data_type_count;
      }
    }
  }

  EXPECT_EQ(int8_data_type_count, expected_int8_data_type_count);
}

void DefaultAttrTest(unsigned expected_int8_data_type_count) {
  auto prog = BuildProgramDesc();
  std::unique_ptr<ir::Graph> graph(new ir::Graph(prog));
  auto pass = PassRegistry::Instance().Get("cpu_quantize_placement_pass");
  graph.reset(pass->Apply(graph.release()));

  unsigned int8_data_type_count = 0;
  for (auto* node : graph->Nodes()) {
    if (node->IsOp()) {
      if (platform::HasOpINT8DataType(node->Op())) {
        ++int8_data_type_count;
      }
    }
  }
  EXPECT_EQ(int8_data_type_count, expected_int8_data_type_count);
}

TEST(QuantizerPlacementPass, enabled_pool) { MainTest({"pool2d"}, {}, 2); }

TEST(QuantizerPlacementPass, enabled_conv_excluded_one) {
  MainTest({"conv2d"}, {4}, 1);
}

TEST(QuantizerPlacementPass, empty_list) {
  // all operators except relu should be quantized
  MainTest({}, {}, 5);
}

TEST(QuantizerPlacementPass, default_attr_value) {
  // all operators except relu should be quantized
  DefaultAttrTest(5);
}

void EnabledOpTypesTest(
    std::initializer_list<std::string> quantize_enabled_op_types,
    std::string missing_op) {
  auto prog = BuildProgramDesc();
  std::unique_ptr<ir::Graph> graph(new ir::Graph(prog));

  auto pass = PassRegistry::Instance().Get("cpu_quantize_placement_pass");
  pass->Set("quantize_enabled_op_types",
            new std::unordered_set<std::string>(quantize_enabled_op_types));

  try {
    graph.reset(pass->Apply(graph.release()));
  } catch (paddle::platform::EnforceNotMet& err) {
    std::string ex_msg = err.what();
    std::string expected_msg =
        "Pass attribute quantize_enabled_op_types contains operator " +
        missing_op + " that is not supported by OneDNN quantization.";
    EXPECT_TRUE(ex_msg.find(expected_msg) != std::string::npos);
  }
}

TEST(QuantizerPlacementPass, unsupported_op_type) {
  // Dropout op is not supported by OneDNN quantization
  EnabledOpTypesTest({"conv2d", "dropout"}, "dropout");
}

}  // namespace paddle::framework::ir

USE_PASS(cpu_quantize_placement_pass);
