// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
#include <sstream>

#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/phi/core/tensor_meta.h"
#include "paddle/pir/core/block.h"
#include "paddle/pir/core/builder.h"
#include "paddle/pir/core/builtin_attribute.h"
#include "paddle/pir/core/builtin_op.h"
#include "paddle/pir/core/builtin_type.h"
#include "paddle/pir/core/dialect.h"
#include "paddle/pir/core/enforce.h"
#include "paddle/pir/core/ir_context.h"
#include "paddle/pir/core/ir_printer.h"
#include "paddle/pir/core/op_base.h"
#include "paddle/pir/core/program.h"
#include "paddle/pir/core/region.h"
#include "test/cpp/pir/tools/test_dialect.h"
#include "test/cpp/pir/tools/test_op.h"

pir::AttributeMap CreateAttributeMap(
    const std::vector<std::string> &attribute_names,
    const std::vector<std::string> &attributes) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  pir::AttributeMap attr_map;
  for (size_t i = 0; i < attribute_names.size(); i++) {
    pir::Attribute attr_value = pir::StrAttribute::get(ctx, attributes[i]);
    attr_map.insert(
        std::pair<std::string, pir::Attribute>(attribute_names[i], attr_value));
  }
  return attr_map;
}

pir::Operation *CreateDenseTensorOp(
    pir::IrContext *ctx,
    const phi::DDim &dims,
    const std::vector<std::string> &attribute_names,
    const std::vector<std::string> &attributes,
    const pir::Type &dtype =
        pir::Float32Type::get(pir::IrContext::Instance())) {
  std::vector<pir::Value> op_inputs = {};
  phi::DataLayout data_layout = phi::DataLayout::NCHW;
  phi::LoD lod = {{0, 1, 2}};
  size_t offset = 0;
  std::vector<pir::Type> op_output_types = {
      pir::DenseTensorType::get(ctx, dtype, dims, data_layout, lod, offset)};
  pir::Operation *op =
      pir::Operation::Create(op_inputs,
                             CreateAttributeMap(attribute_names, attributes),
                             op_output_types,
                             pir::OpInfo());
  return op;
}

TEST(op_test, region_test) {
  // (1) Register Dialect, Operation1, Operation2 into IrContext.
  pir::IrContext *ctx = pir::IrContext::Instance();
  pir::Dialect *test_dialect = ctx->GetOrRegisterDialect<test::TestDialect>();
  EXPECT_EQ(test_dialect != nullptr, true);

  // (2) Get registered operations.
  pir::OpInfo op1_info = ctx->GetRegisteredOpInfo(test::Operation1::name());
  pir::OpInfo op2_info = ctx->GetRegisteredOpInfo(test::Operation2::name());

  pir::Operation *op1 =
      pir::Operation::Create({},
                             CreateAttributeMap({"op1_attr1", "op1_attr2"},
                                                {"op1_attr1", "op1_attr2"}),
                             {pir::Float32Type::get(ctx)},
                             op1_info);
  pir::Operation *op_2 =
      pir::Operation::Create({}, {}, {pir::Float32Type::get(ctx)}, op2_info);

  pir::OperationArgument argument(op2_info);
  argument.output_types = {pir::Float32Type::get(ctx)};
  argument.num_regions = 1;

  pir::Operation *op3 = pir::Operation::Create(argument);

  pir::Region &region = op3->region(0);
  EXPECT_EQ(region.empty(), true);

  // (3) Test custom operation printer
  std::stringstream ss;
  op1->Print(ss);
  EXPECT_EQ(ss.str(), " (%0) = \"test.operation1\" ()");

  region.push_back(new pir::Block());
  region.push_front(new pir::Block());
  region.insert(region.begin(), new pir::Block());
  pir::Block *block = region.front();
  block->push_front(op1);
  block->insert(block->begin(), op_2);
  op3->Destroy();
}

TEST(op_test, module_op_death) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  pir::OpInfo op_info = ctx->GetRegisteredOpInfo(pir::ModuleOp::name());

  std::vector<pir::Value> inputs{pir::Value()};
  pir::AttributeMap attrs{{"program", pir::Int32Attribute::get(ctx, 1)}};
  std::vector<pir::Type> output_types = {pir::Float32Type::get(ctx)};

  EXPECT_THROW(pir::Operation::Create(inputs, {}, {}, op_info),
               pir::IrNotMetException);
  EXPECT_THROW(pir::Operation::Create({}, attrs, {}, op_info),
               pir::IrNotMetException);
  EXPECT_THROW(pir::Operation::Create({}, {}, output_types, op_info),
               pir::IrNotMetException);

  pir::Program program(ctx);

  EXPECT_EQ(program.module_op().program(), &program);
  EXPECT_EQ(program.module_op().ir_context(), ctx);

  program.module_op()->set_attribute("program",
                                     pir::PointerAttribute::get(ctx, &program));
}

TEST(op_test, trait_and_interface) {
  pir::IrContext ctx;
  ctx.GetOrRegisterDialect<test::TestDialect>();
  pir::Program program(&ctx);
  auto block = program.block();
  pir::Builder builder(&ctx, block);
  auto op1 = builder.Build<test::Operation1>();
  auto op2 = builder.Build<test::Operation2>();

  EXPECT_EQ(op1->HasTrait<test::ReadOnlyTrait>(), false);
  EXPECT_EQ(op1->HasInterface<test::InferShapeInterface>(), false);
  EXPECT_EQ(op2->HasTrait<test::ReadOnlyTrait>(), true);
  EXPECT_EQ(op2->HasInterface<test::InferShapeInterface>(), true);

  pir::OperationArgument argument(&ctx, "test.region");
  argument.num_regions = 2u;
  EXPECT_THROW(builder.Build(argument), pir::IrNotMetException);
}

TEST(op_test, op_traits_test) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<test::TestDialect>();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();

  pir::Program program(ctx);
  auto block = program.block();
  pir::Builder builder(ctx, block);

  pir::Type dtype = pir::Float32Type::get(ctx);
  phi::DDim dims = {2, 2};
  phi::DataLayout data_layout = phi::DataLayout::NCHW;
  phi::LoD lod = {{0, 1, 2}};
  size_t offset = 0;

  pir::DenseTensorType dense_tensor_dtype =
      pir::DenseTensorType::get(ctx, dtype, dims, data_layout, lod, offset);

  pir::Operation *op1 =
      CreateDenseTensorOp(ctx, dims, {"op1_temp"}, {"op1_attr"}, dtype);
  pir::Operation *op2 =
      CreateDenseTensorOp(ctx, dims, {"op2_temp"}, {"op2_attr"}, dtype);

  auto op3 = builder.Build<test::Operation3>(
      op1->result(0), op2->result(0), dense_tensor_dtype);

  EXPECT_EQ(op3->HasTrait<pir::op_trait::SameOperandsShapeTrait>(), true);
  EXPECT_EQ(op3->HasTrait<pir::op_trait::SameOperandsAndResultShapeTrait>(),
            true);
  EXPECT_EQ(op3->HasTrait<pir::op_trait::SameOperandsElementTypeTrait>(), true);
  EXPECT_EQ(
      op3->HasTrait<pir::op_trait::SameOperandsAndResultElementTypeTrait>(),
      true);
  EXPECT_EQ(op3->HasTrait<pir::op_trait::SameOperandsAndResultTypeTrait>(),
            true);
  EXPECT_EQ(op3->HasTrait<pir::op_trait::SameTypeOperandsTrait>(), true);
}
