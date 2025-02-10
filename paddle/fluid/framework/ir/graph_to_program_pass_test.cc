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

#include "paddle/fluid/framework/ir/graph_to_program_pass.h"

#include <algorithm>

#include "gtest/gtest.h"
#include "paddle/fluid/framework/details/build_strategy.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/var_desc.h"

namespace paddle::framework::ir {

class Node;

void BuildNoCircleGraph(Graph* g) {
  OpDesc op1;
  op1.SetType("op1");
  OpDesc op2;
  op2.SetType("op2");
  OpDesc op3;
  op3.SetType("op3");
  OpDesc op4;
  op4.SetType("op4");
  OpDesc op5;
  op5.SetType("op5");
  VarDesc var1("var1");
  VarDesc var2("var2");
  VarDesc var3("var3");
  VarDesc var4("var4");

  ir::Node* o1 = g->CreateOpNode(&op1);
  ir::Node* o2 = g->CreateOpNode(&op2);
  ir::Node* o3 = g->CreateOpNode(&op3);
  ir::Node* o4 = g->CreateOpNode(&op4);
  ir::Node* o5 = g->CreateOpNode(&op5);
  ir::Node* v1 = g->CreateVarNode(&var1);
  ir::Node* v2 = g->CreateVarNode(&var2);
  ir::Node* v3 = g->CreateVarNode(&var3);
  ir::Node* v4 = g->CreateVarNode(&var4);

  // o1->v1->o2
  o1->outputs.push_back(v1);
  o2->inputs.push_back(v1);
  v1->inputs.push_back(o1);
  v1->outputs.push_back(o2);
  // o2->v2->o3
  // o2->v2->o4
  o2->outputs.push_back(v2);
  o3->inputs.push_back(v2);
  o4->inputs.push_back(v2);
  v2->outputs.push_back(o3);
  v2->outputs.push_back(o4);
  v2->inputs.push_back(o2);
  // o4->v3->o5
  o4->outputs.push_back(v3);
  o5->inputs.push_back(v3);
  v3->inputs.push_back(o4);
  v3->outputs.push_back(o5);
  // o3-v4->o5
  o3->outputs.push_back(v4);
  o5->inputs.push_back(v4);
  v4->inputs.push_back(o3);
  v4->outputs.push_back(o5);
}

TEST(GraphToProgramPass, Basic) {
  ProgramDesc prog;
  std::unique_ptr<Graph> g(new Graph(prog));
  BuildNoCircleGraph(g.get());

  auto pass = paddle::framework::ir::PassRegistry::Instance().Get(
      "graph_to_program_pass");

  ProgramDesc compiled_prog;
  pass->SetNotOwned<paddle::framework::ProgramDesc>("program", &compiled_prog);
  pass->Apply(g.get());
  std::vector<OpDesc*> ops = compiled_prog.Block(0).AllOps();
  EXPECT_EQ(ops[0]->Type(), "op1");
  EXPECT_EQ(ops[1]->Type(), "op2");
  if (ops[2]->Type() == "op3") {
    EXPECT_EQ(ops[3]->Type(), "op4");
  } else if (ops[2]->Type() == "op4") {
    EXPECT_EQ(ops[3]->Type(), "op3");
  }
  EXPECT_EQ(ops[4]->Type(), "op5");

  std::unordered_set<std::string> vars;
  for (VarDesc* v : compiled_prog.Block(0).AllVars()) {
    vars.insert(v->Name());
  }
  EXPECT_TRUE(vars.find("var1") != vars.end());
  EXPECT_TRUE(vars.find("var2") != vars.end());
  EXPECT_TRUE(vars.find("var3") != vars.end());
}

void BuildProgramWithMultiBlock(ProgramDesc* program) {
  auto* global_block = program->MutableBlock(0);
  auto* mul_1_x = global_block->Var("Mul_1_X");
  mul_1_x->SetType(proto::VarType::DENSE_TENSOR);
  mul_1_x->SetLoDLevel(0);
  mul_1_x->SetDataType(proto::VarType::FP32);
  mul_1_x->SetShape({1000, 784});

  auto* mul_1_y = global_block->Var("Mul_1_Y");
  mul_1_y->SetType(proto::VarType::DENSE_TENSOR);
  mul_1_y->SetLoDLevel(0);
  mul_1_y->SetDataType(proto::VarType::FP32);
  mul_1_y->SetShape({784, 100});

  auto* mul_1_out = global_block->Var("Mul_1_Out");
  mul_1_out->SetType(proto::VarType::DENSE_TENSOR);
  auto* mul_op_1 = global_block->AppendOp();

  mul_op_1->SetType("mul");
  mul_op_1->SetInput("X", {mul_1_x->Name()});
  mul_op_1->SetInput("Y", {mul_1_y->Name()});
  mul_op_1->SetOutput("Y", {mul_1_out->Name()});

  // building cond op such as less_than
  auto* less_than_op_1 = global_block->AppendOp();
  less_than_op_1->SetType("less_than");
  auto* less_than_1_x = global_block->Var("Less_than_1_X");
  less_than_1_x->SetType(proto::VarType::DENSE_TENSOR);
  less_than_1_x->SetLoDLevel(0);
  less_than_1_x->SetDataType(proto::VarType::FP32);
  less_than_1_x->SetShape({1});

  auto* less_than_1_y = global_block->Var("Less_than_1_Y");
  less_than_1_y->SetType(proto::VarType::DENSE_TENSOR);
  less_than_1_y->SetLoDLevel(0);
  less_than_1_y->SetDataType(proto::VarType::FP32);
  less_than_1_y->SetShape({1});

  auto* less_than_1_out = global_block->Var("Less_than_1_Out");
  less_than_1_out->SetType(proto::VarType::BOOL);

  less_than_op_1->SetInput("X", {less_than_1_x->Name()});
  less_than_op_1->SetInput("Y", {less_than_1_y->Name()});
  less_than_op_1->SetOutput("Out", {less_than_1_out->Name()});

  BlockDesc* sub_block = program->AppendBlock(*global_block);
  std::vector<BlockDesc*> sub_blocks;
  sub_blocks.push_back(sub_block);

  BlockDesc* sub_block2 =
      program->AppendBlock(*sub_block);  // for testing nested case.
  sub_blocks.push_back(sub_block2);

  // building while op in sub_block
  auto* while_op = global_block->AppendOp();
  while_op->SetType("while");
  while_op->SetAttr("sub_block", sub_blocks[0]);

  auto* while_x = global_block->Var("While_X");
  while_x->SetType(proto::VarType::DENSE_TENSOR);
  while_x->SetLoDLevel(0);
  while_x->SetDataType(proto::VarType::FP32);
  while_x->SetShape({1});

  while_op->SetInput("kX", {while_x->Name()});
  while_op->SetInput("kCondition", {less_than_1_out->Name()});

  auto* while_out = global_block->Var("While_Out");
  while_out->SetType(proto::VarType::DENSE_TENSOR);
  while_out->SetLoDLevel(0);
  while_out->SetDataType(proto::VarType::FP32);
  while_out->SetShape({1});

  auto* steps = global_block->Var("StepScopes");

  while_op->SetOutput("kOutputs", {while_out->Name()});
  while_op->SetOutput("kStepScopes", {steps->Name()});

  auto* mul_2_x = global_block->Var("Mul_2_X");
  mul_2_x->SetType(proto::VarType::DENSE_TENSOR);
  mul_2_x->SetLoDLevel(0);
  mul_2_x->SetDataType(proto::VarType::FP32);
  mul_2_x->SetShape({1000, 784});

  auto* mul_2_y = global_block->Var("Mul_2_Y");
  mul_2_y->SetType(proto::VarType::DENSE_TENSOR);
  mul_2_y->SetLoDLevel(0);
  mul_2_y->SetDataType(proto::VarType::FP32);
  mul_2_y->SetShape({784, 100});

  auto* mul_op_2 = sub_blocks[0]->AppendOp();
  mul_op_2->SetType("mul");
  mul_op_2->SetInput("X", {mul_2_x->Name()});
  mul_op_2->SetInput("Y", {mul_2_y->Name()});

  auto* mul_2_out = global_block->Var("Mul_2_Out");
  mul_2_out->SetType(proto::VarType::DENSE_TENSOR);
  mul_op_2->SetOutput("Y", {mul_2_out->Name()});

  auto* less_than_op_2 = sub_blocks[0]->AppendOp();
  less_than_op_2->SetType("less_than");
  auto* less_than_2_x = global_block->Var("Less_than_2_X");
  less_than_2_x->SetType(proto::VarType::DENSE_TENSOR);
  less_than_2_x->SetLoDLevel(0);
  less_than_2_x->SetDataType(proto::VarType::FP32);
  less_than_2_x->SetShape({1});

  auto* less_than_2_y = global_block->Var("Less_than_2_Y");
  less_than_2_y->SetType(proto::VarType::DENSE_TENSOR);
  less_than_2_y->SetLoDLevel(0);
  less_than_2_y->SetDataType(proto::VarType::FP32);
  less_than_2_y->SetShape({1});

  less_than_op_2->SetInput("X", {less_than_2_x->Name()});
  less_than_op_2->SetInput("Y", {less_than_2_y->Name()});

  auto* less_than_2_out = global_block->Var("Less_than_2_Out");
  less_than_2_out->SetType(proto::VarType::BOOL);
  less_than_op_2->SetOutput("Out", {less_than_2_out->Name()});

  auto* cond_op = sub_blocks[0]->AppendOp();
  cond_op->SetType("conditional_block");
  cond_op->SetAttr("sub_block", sub_blocks[1]);

  auto* cond_x = sub_blocks[0]->Var("Cond_X");
  cond_x->SetType(proto::VarType::DENSE_TENSOR);
  cond_x->SetLoDLevel(0);
  cond_x->SetDataType(proto::VarType::FP32);
  cond_x->SetShape({1});

  cond_op->SetInput("kInputs", {cond_x->Name()});
  cond_op->SetInput("kCondition", {less_than_2_out->Name()});

  auto* cond_out = sub_blocks[0]->Var("Cond_Out");
  cond_out->SetType(proto::VarType::DENSE_TENSOR);
  cond_out->SetLoDLevel(0);
  cond_out->SetDataType(proto::VarType::FP32);
  cond_out->SetShape({1});

  auto* scope = sub_blocks[0]->Var("Scope");
  scope->SetType(proto::VarType::STEP_SCOPES);

  cond_op->SetOutput("kOutputs", {cond_out->Name()});
  cond_op->SetOutput("kScope", {scope->Name()});

  auto* mul_3_x = global_block->Var("Mul_3_X");
  mul_3_x->SetType(proto::VarType::DENSE_TENSOR);
  mul_3_x->SetLoDLevel(0);
  mul_3_x->SetDataType(proto::VarType::FP32);
  mul_3_x->SetShape({1000, 784});

  auto* mul_3_y = global_block->Var("Mul_3_Y");
  mul_3_y->SetType(proto::VarType::DENSE_TENSOR);
  mul_3_y->SetLoDLevel(0);
  mul_3_y->SetDataType(proto::VarType::FP32);
  mul_3_y->SetShape({784, 100});

  auto* mul_3_out = global_block->Var("Mul_3_Out");
  mul_3_out->SetType(proto::VarType::DENSE_TENSOR);

  auto* mul_op_3 = sub_blocks[1]->AppendOp();
  mul_op_3->SetType("mul");
  mul_op_3->SetInput("X", {mul_3_x->Name()});
  mul_op_3->SetInput("Y", {mul_3_y->Name()});
  mul_op_3->SetOutput("Y", {mul_3_out->Name()});
}

bool VarComparator(const VarDesc* a, const VarDesc* b) {
  return a->Name() < b->Name();
}

void CheckBlockVarsEqual(const BlockDesc& before_block,
                         const BlockDesc& after_block) {
  auto before_vars = before_block.AllVars();
  auto after_vars = after_block.AllVars();

  EXPECT_EQ(before_vars.size(), after_vars.size());

  // var's order is unimportant
  std::sort(before_vars.begin(), before_vars.end(), VarComparator);
  std::sort(after_vars.begin(), after_vars.end(), VarComparator);

  for (size_t var_idx = 0; var_idx < before_vars.size(); ++var_idx) {
    const auto& before_var = before_vars.at(var_idx);
    const auto& after_var = after_vars.at(var_idx);

    EXPECT_EQ(before_var->Name(), after_var->Name());
    EXPECT_EQ(before_var->GetType(), after_var->GetType());
  }
}

void CheckOpInputsEqual(const OpDesc* before_op, const OpDesc* after_op) {
  const auto& before_inputs = before_op->InputNames();
  const auto& after_inputs = after_op->InputNames();

  EXPECT_EQ(before_inputs.size(), after_inputs.size());
  for (size_t in_idx = 0; in_idx < before_inputs.size(); ++in_idx) {
    const auto& before_in_arg = before_inputs[in_idx];
    const auto& after_in_arg = after_inputs[in_idx];
    EXPECT_EQ(before_in_arg, after_in_arg);

    const auto& before_in_vars = before_op->Input(before_in_arg);
    const auto& after_in_vars = after_op->Input(after_in_arg);
    EXPECT_EQ(before_in_vars, after_in_vars);
  }
}

void CheckOpOutputsEqual(const OpDesc* before_op, const OpDesc* after_op) {
  const auto& before_outputs = before_op->OutputNames();
  const auto& after_outputs = after_op->OutputNames();

  EXPECT_EQ(before_outputs.size(), after_outputs.size());
  for (size_t out_idx = 0; out_idx < before_outputs.size(); ++out_idx) {
    const auto& before_out_arg = before_outputs[out_idx];
    const auto& after_out_arg = after_outputs[out_idx];
    EXPECT_EQ(before_out_arg, after_out_arg);

    const auto& before_out_vars = before_op->Output(before_out_arg);
    const auto& after_out_vars = after_op->Output(after_out_arg);
    EXPECT_EQ(before_out_vars, after_out_vars);
  }
}

void CheckOpAttrsEqual(const OpDesc* before_op, const OpDesc* after_op) {
  const auto& before_attrs = before_op->AttrNames();
  const auto& after_attrs = after_op->AttrNames();

  EXPECT_EQ(before_attrs.size(), after_attrs.size());
  for (size_t attr_idx = 0; attr_idx < before_attrs.size(); ++attr_idx) {
    const auto& before_attr = before_attrs[attr_idx];
    const auto& after_attr = after_attrs[attr_idx];
    EXPECT_EQ(before_attr, after_attr);

    EXPECT_EQ(before_op->GetAttrType(before_attr),
              after_op->GetAttrType(after_attr));
  }
}

void CheckBlockOpsEqual(const BlockDesc& before_block,
                        const BlockDesc& after_block) {
  EXPECT_EQ(before_block.OpSize(), after_block.OpSize());

  // op's order must be the same
  for (size_t op_idx = 0; op_idx < before_block.OpSize(); ++op_idx) {
    const auto& before_op = before_block.Op(static_cast<int>(op_idx));
    const auto& after_op = after_block.Op(static_cast<int>(op_idx));

    EXPECT_EQ(before_op->Type(), after_op->Type());

    // Step4.2.1 : check each op's input
    CheckOpInputsEqual(before_op, after_op);

    // Step4.2.2 : check each op's output
    CheckOpOutputsEqual(before_op, after_op);

    // Step4.2.3 : check each op's attribute
    CheckOpAttrsEqual(before_op, after_op);
  }
}

TEST(GraphToProgramPass, MultiBlock) {
  // Set FLAGS_convert_all_blocks to true to make sure this test works.
  bool flag_temp = FLAGS_convert_all_blocks;
  FLAGS_convert_all_blocks = true;

  // Step1: Build a program with multi block
  ProgramDesc before_prog;
  BuildProgramWithMultiBlock(&before_prog);

  // Step2: Convert program into graph
  std::unique_ptr<Graph> g(new ir::Graph(before_prog));

  // Step3 : Convert graph back to program
  auto pass = paddle::framework::ir::PassRegistry::Instance().Get(
      "graph_to_program_pass");

  ProgramDesc after_prog;
  pass->SetNotOwned<paddle::framework::ProgramDesc>("program", &after_prog);
  pass->Apply(g.get());

  // Step4 : Check tow program equal
  EXPECT_EQ(before_prog.Size(), after_prog.Size());

  for (size_t block_idx = 0; block_idx < before_prog.Size(); ++block_idx) {
    const auto& before_block = before_prog.Block(block_idx);
    const auto& after_block = after_prog.Block(block_idx);

    EXPECT_EQ(before_block.ID(), after_block.ID());

    // Step4.1 : check each block's var
    CheckBlockVarsEqual(before_block, after_block);

    // Step4.2 : check each block's op
    CheckBlockOpsEqual(before_block, after_block);
  }

  // Recover FLAGS_convert_all_blocks.
  FLAGS_convert_all_blocks = flag_temp;
}

}  // namespace paddle::framework::ir

USE_PASS(graph_to_program_pass);
