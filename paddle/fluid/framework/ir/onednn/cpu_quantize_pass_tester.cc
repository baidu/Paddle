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

#include <unordered_map>

#include "paddle/fluid/framework/ir/onednn/cpu_quantize_pass.h"  // NOLINT
#include "paddle/fluid/framework/naive_executor.h"
#include "paddle/fluid/imperative/type_defs.h"
#include "paddle/phi/common/place.h"

namespace paddle::framework::ir {

static float const SCALE = 2.f;
static int const S8_MAX = 127;
static int const U8_MAX = 255;

void SetOp(ProgramDesc* prog,
           const std::string& type,
           const std::string& name,
           const std::vector<std::string>& inputs,
           const std::vector<std::string>& outputs,
           bool use_mkldnn,
           const std::string& mkldnn_data_type = "float32") {
  auto* op = prog->MutableBlock(0)->AppendOp();
  op->SetType(type);
  op->SetAttr("use_mkldnn", use_mkldnn);
  op->SetAttr("name", name);
  if (type != "dropout" && type != "quantize" && type != "dequantize") {
    op->SetAttr("mkldnn_data_type", mkldnn_data_type);
  }

  if (type == "conv2d") {
    op->SetInput("Input", {inputs[0]});
    op->SetInput("Filter", {inputs[1]});
    if (inputs.size() > 2)
      op->SetInput("Bias", {inputs[2]});
    else
      op->SetInput("Bias", {});
    if (inputs.size() > 3) {
      op->SetInput("ResidualData", {inputs[3]});
      op->SetAttr("fuse_residual_connection", true);
    } else {
      op->SetInput("ResidualData", {});
      op->SetAttr("fuse_residual_connection", false);
    }
    op->SetOutput("Output", {outputs[0]});
  } else if (type == "pool2d" || type == "fused_transpose" ||
             type == "reshape2" || type == "nearest_interp" ||
             type == "nearest_interp_v2" || type == "dropout") {
    op->SetInput("X", {inputs[0]});
    op->SetOutput("Out", {outputs[0]});
  } else if (type == "slice") {
    op->SetInput("Input", {inputs[0]});
    op->SetOutput("Out", {outputs[0]});
  } else if (type == "split") {
    op->SetInput("X", {inputs[0]});
    op->SetOutput("Out", {outputs});
  } else if (type == "fc") {
    op->SetInput("Input", {inputs[0]});
    if (inputs.size() > 1) op->SetInput("W", {inputs[1]});
    if (inputs.size() > 2) op->SetInput("Bias", {inputs[2]});
    op->SetOutput("Out", {outputs[0]});
    op->SetAttr("Scale_in", 1.0f);
    op->SetAttr("Scale_out", 1.0f);
    op->SetAttr("Scale_weights", std::vector<float>{1.0f});
  } else if (type == "concat") {
    op->SetInput("X", inputs);
    op->SetOutput("Out", outputs);
  } else if (type == "dequantize") {
    op->SetInput("Input", {inputs[0]});
    op->SetOutput("Output", {outputs[0]});
    op->SetAttr("Scale", 1.0f);
  } else if (type == "matmul" || type == "matmul_v2" ||
             type == "fused_matmul") {
    op->SetInput("X", {inputs[0]});
    if (inputs.size() > 1) op->SetInput("Y", {inputs[1]});
    if (inputs.size() > 2) op->SetInput("ResidualData", {inputs[2]});
    op->SetOutput("Out", {outputs[0]});
    op->SetAttr("Scale_x", 1.0f);
    op->SetAttr("Scale_y", 1.0f);
    op->SetAttr("Scale_out", 1.0f);
  } else if (type == "fused_elementwise_add" ||
             type == "fused_elementwise_sub" ||
             type == "fused_elementwise_mul") {
    op->SetInput("X", {inputs[0]});
    if (inputs.size() > 1) op->SetInput("Y", {inputs[1]});
    op->SetOutput("Out", {outputs[0]});
    op->SetAttr("scale_x", 1.0f);
    op->SetAttr("scale_y", 1.0f);
    op->SetAttr("scale_out", 1.0f);
  } else if (type == "fusion_gru") {
    op->SetInput("X", {inputs[0]});
    op->SetInput("Bias", {inputs[1]});
    op->SetInput("WeightX", {inputs[2]});
    op->SetInput("WeightH", {inputs[3]});
    op->SetOutput("Hidden", {outputs[0]});
    op->SetAttr("Scale_data", 1.0f);
    op->SetAttr("Shift_data", 0.0f);
    op->SetAttr("Weight_scale", std::vector<float>{1.0f});
  } else if (type == "fusion_lstm") {
    op->SetInput("X", {inputs[0]});
    op->SetInput("Bias", {inputs[1]});
    op->SetInput("WeightX", {inputs[2]});
    op->SetInput("WeightH", {inputs[3]});

    op->SetOutput("Hidden", {outputs[0]});
    op->SetOutput("Cell", {outputs[1]});

    op->SetAttr("Scale_data", 1.0f);
    op->SetAttr("Shift_data", 0.0f);
    op->SetAttr("Weight_scale", std::vector<float>{1.0f});
  }
}

void InitTensorHolder(Scope* scope,
                      const phi::Place& place,
                      const char* var_name) {
  auto x = scope->Var(var_name);
  auto tensor = x->GetMutable<phi::DenseTensor>();
  tensor->mutable_data(place, phi::TransToPhiDataType(proto::VarType::FP32), 1);
}

void PreparePass(std::unique_ptr<ir::Graph>* graph,
                 const ProgramDesc& prog,
                 const std::vector<std::string> variable_names,
                 int* original_nodes_num,
                 int* current_nodes_num,
                 std::string var_without_scale = "",
                 std::string var_signed = "") {
  auto place = phi::CPUPlace();
  NaiveExecutor exe{place};
  Scope scope;
  exe.CreateVariables(prog, 0, true, &scope);
  auto* scales = new VarQuantScale();
  for (auto& v : variable_names) {
    if (v.compare(var_without_scale) == 0) continue;
    InitTensorHolder(&scope, place, v.c_str());
    phi::DenseTensor tensor;
    tensor.Resize({1});
    auto* ptr = tensor.mutable_data<double>(place);
    ptr[0] = SCALE;
    (*scales)[v] = std::make_pair(v == var_signed, std::move(tensor));
  }

  (*graph)->SetNotOwned(kParamScopeAttr, &scope);
  std::unique_ptr<Pass> pass =
      PassRegistry::Instance().Get("cpu_quantize_pass");
  pass->Set("quant_var_scales", scales);

  *original_nodes_num = (*graph)->Nodes().size();
  (*graph).reset(pass->Apply((*graph).release()));
  *current_nodes_num = (*graph)->Nodes().size();
}

void CheckScales(const OpDesc* op, float scale, float shift) {
  std::string type = op->Type();
  std::vector<std::string> scale_names;
  if (type == "conv2d" || type == "fused_conv2d" || type == "fc") {
    EXPECT_EQ(op->GetAttrIfExists<std::vector<float>>("Scale_weights")[0],
              scale);
    scale_names.push_back("Scale_in");
    scale_names.push_back("Scale_out");
  } else if (type == "fused_matmul") {
    scale_names.push_back("Scale_x");
    scale_names.push_back("Scale_y");
    scale_names.push_back("Scale_out");
    auto const& names = op->InputNames();
    if (std::find(names.begin(), names.end(), "ResidualData") != names.end())
      scale_names.push_back("Scale_in_eltwise");
  } else if (type == "fused_elementwise_add" ||
             type == "fused_elementwise_sub" ||
             type == "fused_elementwise_mul") {
    scale_names.push_back("scale_x");
    scale_names.push_back("scale_y");
    scale_names.push_back("scale_out");
  } else if (type == "fusion_gru" || type == "fusion_lstm") {
    EXPECT_EQ(op->GetAttrIfExists<float>("Shift_data"), shift);
    EXPECT_EQ(op->GetAttrIfExists<std::vector<float>>("Scale_weights")[0],
              scale);
    EXPECT_EQ(op->GetAttrIfExists<bool>("force_fp32_output"), true);
    scale_names.push_back("Scale_data");
  }

  for (auto const& scale_name : scale_names) {
    EXPECT_EQ(op->GetAttrIfExists<float>(scale_name), scale);
  }
}

void MainTest(const ProgramDesc& prog,
              const std::vector<std::string> variable_names,
              std::unordered_map<std::string, int> expected_operators,
              const int added_nodes_count,
              float scale = 1.f,
              float shift = 1.f,
              std::string var_without_scale = "",
              std::string var_signed = "") {
  std::unique_ptr<ir::Graph> graph(new ir::Graph(prog));
  int original_nodes_num, current_nodes_num;
  PreparePass(&graph,
              prog,
              variable_names,
              &original_nodes_num,
              &current_nodes_num,
              var_without_scale,
              var_signed);
  for (auto* node : graph->Nodes()) {
    if (node->IsOp()) {
      auto* op = node->Op();
      if (expected_operators.count(op->Type()) > 0) {
        expected_operators[op->Type()]--;
        if (op->GetAttrIfExists<std::string>("mkldnn_data_type") == "int8")
          CheckScales(op, scale, shift);
      }
    }
  }
  for (auto const& pair : expected_operators) {
    EXPECT_EQ(pair.second, 0);
  }
  EXPECT_EQ(original_nodes_num + added_nodes_count, current_nodes_num);
}

static const std::initializer_list<std::string> variable_names{"a",
                                                               "w1",
                                                               "c",
                                                               "d",
                                                               "w2",
                                                               "e",
                                                               "f",
                                                               "g",
                                                               "h",
                                                               "w3",
                                                               "b1",
                                                               "i",
                                                               "j",
                                                               "w4",
                                                               "b2",
                                                               "w5",
                                                               "b3"};
// (a,w1)->Conv1->c and c->Pool1->d
//
// (d,w2)->Conv2->e and e->Pool2->f
//
// d->Dropout1->g and (g, w5, b3)->Fc1->h and (h,w3,b1,i)->Conv3->j
//
// (d,w4, b2)->Conv4->i
ProgramDesc BuildProgramDesc(bool use_mkldnn,
                             const std::string& mkldnn_data_type) {
  ProgramDesc prog;
  for (auto& v : variable_names) {
    auto* var = prog.MutableBlock(0)->Var(v);
    if (v.find("w") == 0 || v.find("b") == 0) {
      var->SetPersistable(true);
    }
  }

  SetOp(&prog,
        "conv2d",
        "Conv1",
        {"a", "w1"},
        {"c"},
        use_mkldnn,
        mkldnn_data_type);
  SetOp(&prog, "pool2d", "Pool1", {"c"}, {"d"}, use_mkldnn, mkldnn_data_type);

  SetOp(&prog,
        "conv2d",
        "Conv2",
        {"d", "w2"},
        {"e"},
        use_mkldnn,
        mkldnn_data_type);
  SetOp(&prog, "pool2d", "Pool2", {"e"}, {"f"}, use_mkldnn, mkldnn_data_type);

  SetOp(&prog, "dropout", "Dropout1", {"d"}, {"g"}, use_mkldnn);
  SetOp(&prog,
        "fc",
        "Fc1",
        {"g", "w5", "b3"},
        {"h"},
        use_mkldnn,
        mkldnn_data_type);
  SetOp(&prog,
        "conv2d",
        "Conv3",
        {"h", "w3", "b1", "i"},
        {"j"},
        use_mkldnn,
        mkldnn_data_type);

  SetOp(&prog,
        "conv2d",
        "Conv4",
        {"c", "w4", "b2"},
        {"i"},
        use_mkldnn,
        mkldnn_data_type);

  return prog;
}

TEST(CpuQuantizePass, quantize) {
  bool use_mkldnn = true;
  std::string mkldnn_data_type = "int8";
  // (a->QUANT1->IN1,w1)->Conv1->OUT1->DEQUANT1->c and
  // c->QUANT2->IN2->Pool1->OUT2->DEQUANT2->d
  //
  // (d->QUANT3->IN3,w2)->Conv2->OUT3->DEQUANT3->e and
  // e->QUANT4->IN4->Pool2->OUT4->DEQUANT4->f
  //
  // d->Dropout1->g and (g->QUANT8->IN8,w5,b3)->Fc1->OUT7->DEQUANT7->h and
  // (h->QUANT5->IN5,w3,b1,i->QUANT6->IN6)->Conv3->OUT5->DEQUANT5->j
  //
  // (d->QUANT7->IN7,w4, b2)->Conv4->DEQUANT6->OUT6->i
  // Insert nodes: 8 Quant + 8 IN + 7 OUT + 7 DEQUANT
  int added_nodes = 8 + 8 + 7 + 7;
  std::unordered_map<std::string, int> expected_operators = {
      {"fused_conv2d", 4}, {"pool2d", 2}, {"quantize", 8}, {"dequantize", 7}};
  MainTest(BuildProgramDesc(use_mkldnn, mkldnn_data_type),
           variable_names,
           expected_operators,
           added_nodes,
           SCALE * S8_MAX);
}

TEST(CpuQuantizePass, do_not_quantize) {
  bool use_mkldnn = true;
  std::string mkldnn_data_type = "float32";
  int added_nodes = 0;
  std::unordered_map<std::string, int> expected_operators = {
      {"fused_conv2d", 4}, {"pool2d", 2}, {"quantize", 0}, {"dequantize", 0}};
  MainTest(BuildProgramDesc(use_mkldnn, mkldnn_data_type),
           variable_names,
           expected_operators,
           added_nodes,
           1.0f);
}

static const std::initializer_list<std::string> variable_names_concat = {
    "a1", "b1", "a2", "b2", "c", "d"};

// a1->Pool1->b1
// a2->Pool2->b2
// (b1,b2)->Concat->c
// c->Pool3->d
ProgramDesc BuildProgramDescConcat() {
  ProgramDesc prog;

  SetOp(&prog, "pool2d", "Pool1", {"a1"}, {"b1"}, true, "float32");
  SetOp(&prog, "pool2d", "Pool2", {"a2"}, {"b2"}, true, "float32");
  SetOp(&prog, "concat", "Concat", {"b1", "b2"}, {"c"}, true, "int8");
  SetOp(&prog, "pool2d", "Pool3", {"c"}, {"d"}, true, "float32");

  return prog;
}

TEST(CpuQuantizePass, concat) {
  // a1->Pool1->b1
  // a2->Pool2->b2
  // (b1->QUANT1->IN1, b2->QUANT2->IN2)->Concat->c
  // c->OUT1->DEQUANT1->Pool3->d
  int added_nodes = 6;
  std::unordered_map<std::string, int> expected_operators = {
      {"pool2d", 3}, {"concat", 1}, {"quantize", 2}, {"dequantize", 1}};
  MainTest(BuildProgramDescConcat(),
           variable_names_concat,
           expected_operators,
           added_nodes);
}

static const std::initializer_list<std::string> variable_names_fusion_gru = {
    "x", "wx", "wh", "b", "h"};

// (x, wx, wh, b)->Fusion_gru->h
ProgramDesc BuildProgramDescFusionGru() {
  ProgramDesc prog;
  for (auto& v : variable_names_fusion_gru) {
    auto* var = prog.MutableBlock(0)->Var(v);
    if (v.find("wx") == 0 || v.find("wh") || v.find("b")) {
      var->SetPersistable(true);
    }
  }

  SetOp(&prog,
        "fusion_gru",
        "Fusion_gru",
        {"x", "wx", "wh", "b"},
        {"h"},
        true,
        "int8");

  return prog;
}

static const std::initializer_list<std::string> variable_names_fusion_lstm = {
    "x", "wx", "wh", "b", "h", "c"};

// (x, wx, wh, b)->Fusion_lstm_1->h
ProgramDesc BuildProgramDescFusionLSTM() {
  ProgramDesc prog;
  for (auto& v : variable_names_fusion_lstm) {
    auto* var = prog.MutableBlock(0)->Var(v);
    if (v.find("wx") == 0 || v.find("wh") || v.find("b")) {
      var->SetPersistable(true);
    }
  }

  SetOp(&prog,
        "fusion_lstm",
        "Fusion_lstm_1",
        {"x", "wx", "wh", "b"},
        {"h", "c"},
        true,
        "int8");

  return prog;
}

TEST(CpuQuantizePass, fusion_gru) {
  // (x, wx, wh, b)->Fusion_gru->h

  // 1 Quant + 1 IN + 0 DeQuant + 0 OUT
  int added_nodes = 1 + 1 + 0 + 0;
  std::unordered_map<std::string, int> expected_operators = {
      {"fusion_gru", 1}, {"quantize", 1}, {"dequantize", 0}};
  MainTest(BuildProgramDescFusionGru(),
           variable_names_fusion_gru,
           expected_operators,
           added_nodes,
           SCALE * S8_MAX,
           128);
}

TEST(CpuQuantizePass, fusion_lstm) {
  // (x, wx, wh, b)->Fusion_lstm->h

  // 1 Quant + 1 IN + 0 DeQuant + 0 OUT
  int added_nodes = 1 + 1 + 0 + 0;
  std::unordered_map<std::string, int> expected_operators = {
      {"fusion_lstm", 1}, {"quantize", 1}, {"dequantize", 0}};
  MainTest(BuildProgramDescFusionLSTM(),
           variable_names_fusion_lstm,
           expected_operators,
           added_nodes,
           SCALE * S8_MAX,
           128.);
}

static const std::initializer_list<std::string> variable_names_immutable_ops = {
    "a", "w1", "b", "c", "d", "e", "f", "g"};

// a->Dequantize->b
// b->Tested Op->c
// c->Dropout->d
void TestImmutableOp(const std::string tested_op) {
  ProgramDesc prog;
  for (auto& v : variable_names_immutable_ops) {
    prog.MutableBlock(0)->Var(v)->SetDataType(proto::VarType::FP32);
  }
  SetOp(&prog, "dequantize", "Dequantize1", {"a"}, {"b"}, true);
  SetOp(&prog, tested_op, tested_op, {"b"}, {"c"}, true, "int8");
  SetOp(&prog, "dropout", "Dropout", {"c"}, {"d"}, true, "float32");

  // a->Dequantize->b
  // b2->Quant->b3->Tested Op->c1->Dequant->c2
  // c2->Dropout->d
  // 1 Quant + 1 IN + 1 DeQuant + 1 OUT
  int added_nodes = 4;
  std::unordered_map<std::string, int> expected_operators = {
      {tested_op, 1}, {"quantize", 1}, {"dequantize", 2}};
  MainTest(prog,
           variable_names_immutable_ops,
           expected_operators,
           added_nodes,
           SCALE * S8_MAX);
}

// a->Dropout1->b
// b->Tested Op->c
// c->Dropout2->d
void TestImmutableOpBetweenNonQuantizedOp(const std::string tested_op) {
  ProgramDesc prog;
  for (auto& v : variable_names_immutable_ops) {
    prog.MutableBlock(0)->Var(v);
  }

  SetOp(&prog, "dropout", "Dropout1", {"a"}, {"b"}, true, "float32");
  SetOp(&prog, tested_op, tested_op, {"b"}, {"c"}, true, "int8");
  SetOp(&prog, "dropout", "Dropout2", {"c"}, {"d"}, true, "float32");

  // 0 Quant + 0 IN + 0 DeQuant + 0 OUT
  int added_nodes = 0;
  std::unordered_map<std::string, int> expected_operators = {
      {tested_op, 1}, {"dropout", 2}, {"quantize", 0}, {"dequantize", 0}};
  MainTest(prog,
           variable_names_immutable_ops,
           expected_operators,
           added_nodes,
           SCALE * S8_MAX);
}

// a->Dropout1->b
// b->TestedOp1(won't be quantized)->c
//    c->Dropout2->d
//    c->TestedOp2(will be quantized)->e
//        e->Pool2d1(will be quantized)->f
//        e->Pool2d2(will be quantized)->g
void TestImmutableOpWithManyOutputs(const std::string tested_op) {
  ProgramDesc prog;
  for (auto& v : variable_names_immutable_ops) {
    prog.MutableBlock(0)->Var(v)->SetDataType(proto::VarType::FP32);
  }

  SetOp(&prog, "dropout", "Dropout1", {"a"}, {"b"}, true, "float32");
  SetOp(&prog,
        tested_op,
        std::string(tested_op + "1"),
        {"b"},
        {"c"},
        true,
        "int8");
  SetOp(&prog, "dropout", "Dropout2", {"c"}, {"d"}, true, "float32");
  SetOp(&prog,
        tested_op,
        std::string(tested_op + "2"),
        {"c"},
        {"e"},
        true,
        "int8");
  SetOp(&prog, "pool2d", "Pool2d1", {"e"}, {"f"}, true, "int8");
  SetOp(&prog, "pool2d", "Pool2d2", {"e"}, {"g"}, true, "int8");

  // 3 Quant + 3 IN + 3 DeQuant + 3 OUT
  int added_nodes = 12;
  std::unordered_map<std::string, int> expected_operators = {{tested_op, 2},
                                                             {"dropout", 2},
                                                             {"pool2d", 2},
                                                             {"quantize", 3},
                                                             {"dequantize", 3}};
  MainTest(prog,
           variable_names_immutable_ops,
           expected_operators,
           added_nodes,
           SCALE * S8_MAX);
}

const std::vector<std::string> immutables = {"reshape2",
                                             "fused_transpose",
                                             "slice",
                                             "nearest_interp",
                                             "nearest_interp_v2",
                                             "split"};

class TestImmutables : public testing::TestWithParam<std::string> {};

TEST_P(TestImmutables, immutable_basic) {  // NOLINT
  TestImmutableOp(GetParam());
}

TEST_P(TestImmutables, immutable_between_non_quantized) {  // NOLINT
  TestImmutableOpBetweenNonQuantizedOp(GetParam());
}

TEST_P(TestImmutables, immutable_many_outputs) {  // NOLINT
  TestImmutableOpWithManyOutputs(GetParam());
}

INSTANTIATE_TEST_CASE_P(
    CpuQuantizePass,
    TestImmutables,
    testing::ValuesIn(immutables),
    [](const ::testing::TestParamInfo<TestImmutables::ParamType>& info) {
      std::string name = info.param;
      return name;
    });

static const std::initializer_list<std::string> variable_names_matmul = {
    "a", "b", "c", "d", "e", "f", "g", "h"};

ProgramDesc BuildProgramDescMatmul() {
  ProgramDesc prog;
  for (auto& v : variable_names_matmul) {
    prog.MutableBlock(0)->Var(v);
  }
  SetOp(&prog, "dequantize", "Dequantize1", {"a"}, {"b"}, true);
  SetOp(&prog, "dequantize", "Dequantize2", {"c"}, {"d"}, true);
  SetOp(&prog, "fused_matmul", "FusedMatmul", {"b", "d"}, {"e"}, true, "int8");
  SetOp(&prog, "dropout", "Dropout", {"e"}, {"f"}, true, "float32");

  return prog;
}

ProgramDesc BuildProgramDescMatmulResidual() {
  ProgramDesc prog;
  for (auto& v : variable_names_matmul) {
    prog.MutableBlock(0)->Var(v);
  }
  SetOp(&prog, "dequantize", "Dequantize1", {"a"}, {"b"}, true);
  SetOp(&prog, "dequantize", "Dequantize2", {"c"}, {"d"}, true);
  SetOp(&prog, "dequantize", "Dequantize3", {"e"}, {"f"}, true);
  SetOp(&prog,
        "fused_matmul",
        "FusedMatmul",
        {"b", "d", "f"},
        {"g"},
        true,
        "int8");
  SetOp(&prog, "dropout", "Dropout", {"g"}, {"h"}, true, "float32");

  return prog;
}

TEST(CpuQuantizePass, matmul) {
  // 2 Quant + 2 IN + 1 DeQuant + 1 OUT
  int added_nodes = 6;
  std::unordered_map<std::string, int> expected_operators = {
      {"fused_matmul", 1}, {"quantize", 2}, {"dequantize", 3}};
  MainTest(BuildProgramDescMatmul(),
           variable_names_matmul,
           expected_operators,
           added_nodes,
           SCALE * S8_MAX);
}

TEST(CpuQuantizePass, matmul_residual) {
  // 3 Quant + 3 IN + 1 DeQuant + 1 OUT
  int added_nodes = 8;
  std::unordered_map<std::string, int> expected_operators = {
      {"fused_matmul", 1}, {"quantize", 3}, {"dequantize", 4}};
  MainTest(BuildProgramDescMatmulResidual(),
           variable_names_matmul,
           expected_operators,
           added_nodes,
           SCALE * S8_MAX);
}

static const std::initializer_list<std::string> variable_names_elementwise = {
    "a", "b", "c", "d", "e", "f"};

ProgramDesc BuildProgramDescElementwise(const std::string elementwise_type,
                                        const std::string elementwise_name) {
  ProgramDesc prog;
  for (auto& v : variable_names_elementwise) {
    prog.MutableBlock(0)->Var(v);
  }
  SetOp(&prog, "dequantize", "Dequantize1", {"a"}, {"b"}, true);
  SetOp(&prog, "dequantize", "Dequantize2", {"c"}, {"d"}, true);
  SetOp(&prog,
        elementwise_type,
        elementwise_name,
        {"b", "d"},
        {"e"},
        true,
        "int8");
  SetOp(&prog, "dropout", "Dropout", {"e"}, {"f"}, true, "float32");

  return prog;
}

void TestElementwise(std::vector<std::string> elementwise) {
  // 2 Quant + 2 IN + 1 DeQuant + 1 OUT
  int added_nodes = 6;
  std::unordered_map<std::string, int> expected_operators = {
      {elementwise[0], 1}, {"quantize", 2}, {"dequantize", 3}};
  MainTest(BuildProgramDescElementwise(elementwise[0], elementwise[1]),
           variable_names_elementwise,
           expected_operators,
           added_nodes,
           SCALE * S8_MAX);
}

void TestElementwiseOutputScaleMissing(std::vector<std::string> elementwise) {
  int added_nodes = 0;
  std::unordered_map<std::string, int> expected_operators = {
      {elementwise[0], 1}, {"quantize", 0}, {"dequantize", 2}};
  MainTest(BuildProgramDescElementwise(elementwise[0], elementwise[1]),
           variable_names_elementwise,
           expected_operators,
           added_nodes,
           1.f,
           1.f,
           "e");
}

void TestElementwiseUnsignedAndSignedInput(
    std::vector<std::string> elementwise) {
  int added_nodes = 0;
  std::unordered_map<std::string, int> expected_operators = {
      {elementwise[0], 1}, {"quantize", 0}, {"dequantize", 2}};
  MainTest(BuildProgramDescElementwise(elementwise[0], elementwise[1]),
           variable_names_elementwise,
           expected_operators,
           added_nodes,
           1.f,
           1.f,
           "",
           "b");
}

const std::vector<std::vector<std::string>> elementwises = {
    {"fused_elementwise_add", "FusedElementwiseAdd"},
    {"fused_elementwise_mul", "FusedElementwiseMul"},
    {"fused_elementwise_sub", "FusedElementwiseSub"}};

class TestElementwises
    : public testing::TestWithParam<std::vector<std::string>> {};

TEST_P(TestElementwises, elementwise_basic) {  // NOLIN
  TestElementwise(GetParam());
}

TEST_P(TestElementwises, elementwise_output_scale_missing) {  // NOLINT
  TestElementwiseOutputScaleMissing(GetParam());
}

TEST_P(TestElementwises, elementwise_unsigned_and_signed_input) {  // NOLINT
  TestElementwiseUnsignedAndSignedInput(GetParam());
}

INSTANTIATE_TEST_CASE_P(
    CpuQuantizePass,
    TestElementwises,
    testing::ValuesIn(elementwises),
    [](const ::testing::TestParamInfo<TestElementwises::ParamType>& info) {
      std::string name = info.param[0];
      return name;
    });

const std::vector<std::string> churn_out_vars(ProgramDesc* prog,
                                              const std::string& prefix,
                                              int number) {
  auto v = std::vector<std::string>();
  for (int i = 0; i < number; ++i) {
    auto name = prefix + std::to_string(i);
    prog->MutableBlock(0)->Var(name);
    v.push_back(name);
  }
  return v;
}

void create_vars(ProgramDesc* prog,
                 const std::initializer_list<std::string>& names) {
  for (auto const& name : names) prog->MutableBlock(0)->Var(name);
}

void SetMultiGruOp(ProgramDesc* prog,
                   const std::string x,
                   const std::vector<std::string> wx,
                   const std::vector<std::string> wh,
                   const std::vector<std::string> b,
                   const std::string h,
                   int layers) {
  auto* op = prog->MutableBlock(0)->AppendOp();
  op->SetType("multi_gru");
  op->SetInput("X", {x});
  op->SetInput("WeightX", wx);
  op->SetInput("WeightH", wh);
  op->SetInput("Bias", b);
  op->SetOutput("Hidden", {h});
  op->SetAttr("layers", layers);
  op->SetAttr("origin_mode", false);
  op->SetAttr("use_mkldnn", true);
  op->SetAttr("name", std::string("Multi_gru"));
  op->SetAttr("mkldnn_data_type", std::string("int8"));
  op->SetAttr("Scale_data", 1.0f);
  op->SetAttr("Shift_data", 0.0f);
}

void MainTestMultiGru(int layers) {
  ProgramDesc prog;

  // Create variables
  create_vars(&prog, {"x", "h"});
  const std::vector<std::string> wx = churn_out_vars(&prog, "wx", 2 * layers);
  const std::vector<std::string> wh = churn_out_vars(&prog, "wh", 2 * layers);
  const std::vector<std::string> b = churn_out_vars(&prog, "b", 2 * layers);

  std::vector<std::string> all_vars;
  all_vars.reserve(wx.size() + wh.size() + b.size() + 2);
  all_vars.insert(all_vars.end(), wx.begin(), wx.end());
  all_vars.insert(all_vars.end(), wh.begin(), wh.end());
  all_vars.insert(all_vars.end(), b.begin(), b.end());
  all_vars.push_back("x");
  all_vars.push_back("h");

  // Prepare program descriptor
  SetMultiGruOp(&prog, "x", wx, wh, b, "h", layers);

  // Prepare and run the pass
  std::unique_ptr<ir::Graph> graph(new ir::Graph(prog));
  int original_nodes_num, current_nodes_num;
  PreparePass(&graph, prog, all_vars, &original_nodes_num, &current_nodes_num);

  // Verify graph after quantization
  float scale = 2 * 127;
  float shift = 128;
  int quantize_nodes_count = 0;
  int dequantize_nodes_count = 0;
  int multi_gru_nodes_count = 0;
  for (auto* node : graph->Nodes()) {
    if (node->IsOp()) {
      auto* op = node->Op();
      if (op->Type() == "multi_gru") {
        multi_gru_nodes_count++;

        auto op_name = PADDLE_GET_CONST(std::string, op->GetAttr("name"));
        EXPECT_EQ(PADDLE_GET_CONST(float, op->GetAttr("Scale_data")), scale)
            << "Scale_data for node '" + op_name + "'.";
        EXPECT_EQ(PADDLE_GET_CONST(float, op->GetAttr("Shift_data")), shift)
            << "Shift_data for node '" + op_name + "'.";
        EXPECT_EQ(op->Input("Scale_weights").size(), 2u * layers)
            << "Scale_weights for node '" + op_name + "'.";
        EXPECT_EQ(PADDLE_GET_CONST(bool, op->GetAttr("force_fp32_output")),
                  true)
            << "force_fp32_output for node '" + op_name + "'.";
      } else if (op->Type() == "quantize") {
        quantize_nodes_count++;
      } else if (op->Type() == "dequantize") {
        dequantize_nodes_count++;
      }
    }
  }

  int multi_gru_count = 1;
  int quant_count = 1;
  int quant_out_count = 1;
  int dequant_count = 0;
  int dequant_out_count = 0;
  int scale_weights_count = 2 * layers;
  int added_nodes_count = quant_count + quant_out_count + scale_weights_count +
                          dequant_count + dequant_out_count;

  EXPECT_EQ(multi_gru_nodes_count, multi_gru_count);
  EXPECT_EQ(quantize_nodes_count, quant_count);
  EXPECT_EQ(dequantize_nodes_count, dequant_count);
  EXPECT_EQ(original_nodes_num + added_nodes_count, current_nodes_num);
}

TEST(CpuQuantizePass, multi_gru_1) {
  int layers = 1;
  MainTestMultiGru(layers);
}

TEST(CpuQuantizePass, multi_gru_2) {
  int layers = 2;
  MainTestMultiGru(layers);
}

TEST(CpuQuantizePass, multi_gru_3) {
  int layers = 3;
  MainTestMultiGru(layers);
}

static const std::initializer_list<std::string>
    variable_names_multi_inputs_outputs = {"a", "b", "c1", "c2", "d", "e"};

// a->Pool->b
// b->Split->c1, c2
// (c1, c2, c1, c2)->Concat->d
// d->Pool->e
ProgramDesc BuildProgramDescMulti() {
  ProgramDesc prog;
  for (auto& v : variable_names_multi_inputs_outputs) {
    prog.MutableBlock(0)->Var(v)->SetDataType(proto::VarType::FP32);
  }

  SetOp(&prog, "pool2d", "Pool", {"a"}, {"b"}, true, "float32");
  SetOp(&prog, "split", "Split", {"b"}, {"c1", "c2"}, true, "int8");
  SetOp(
      &prog, "concat", "Concat", {"c1", "c2", "c1", "c2"}, {"d"}, true, "int8");
  SetOp(&prog, "pool2d", "Pool2", {"d"}, {"e"}, true, "float32");

  return prog;
}

TEST(CpuQuantizePass, multi_inputs_outputs_ops) {
  // a->QUANT1->Split
  //               b1->DEQUANT->OUT->QUANT
  //               b2->DEQUANT->OUT->QUANT
  // (b1, b2, b1, b2)->Concat->c->DEQUANT->Pool->d
  int added_nodes = 6 * 2;
  std::unordered_map<std::string, int> expected_operators = {{"pool2d", 2},
                                                             {"concat", 1},
                                                             {"split", 1},
                                                             {"quantize", 3},
                                                             {"dequantize", 3}};
  MainTest(BuildProgramDescMulti(),
           variable_names_multi_inputs_outputs,
           expected_operators,
           added_nodes);
}

}  // namespace paddle::framework::ir

USE_PASS(cpu_quantize_pass);
