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

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <memory>
#include <sstream>

#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/op_dialect.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/add_broadcast_to_elementwise_pass.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/add_store_in_group_op_pass.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/cinn_group_cluster_pass.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/lowering_pass/lower_cinn_fusion_op_pass.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/merge_reshape_with_broadcast_pass.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/pd_to_cinn_pass.h"
#include "paddle/fluid/framework/new_executor/interpretercore.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_api.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/transforms/build_cinn_pass.h"
#include "paddle/fluid/pir/transforms/general/dead_code_elimination_pass.h"
#include "paddle/fluid/pir/transforms/pd_op_to_kernel_pass.h"
#include "paddle/pir/include/core/builtin_dialect.h"
#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/core/ir_context.h"
#include "paddle/pir/include/core/program.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_dialect.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_op.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_manager.h"

bool simple_cmp(float a, float b) { return std::abs((a - b) / a) < 1e-5; }

std::vector<::pir::Type> CreateDenseTensorTypes(const phi::DDim& dims) {
  ::pir::IrContext* ctx = ::pir::IrContext::Instance();
  ::pir::Type fp32_dtype = ::pir::Float32Type::get(ctx);
  phi::DataLayout data_layout = phi::DataLayout::NCHW;
  phi::LegacyLoD lod = {};
  size_t offset = 0;
  std::vector<::pir::Type> op_output_types = {::pir::DenseTensorType::get(
      ctx, fp32_dtype, dims, data_layout, lod, offset)};
  return op_output_types;
}

static void RunAndCheckResult(::pir::Program* program,
                              const bool check_result = true,
                              const float gt_val = 2.0) {
  ::pir::IrContext* ctx = ::pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<cinn::dialect::OperatorDialect>();

  pir::PassManager stage_1_pm(ctx);
  stage_1_pm.AddPass(cinn::dialect::ir::CreatePdOpToCinnOpPass());
  stage_1_pm.AddPass(
      std::make_unique<cinn::dialect::ir::MergeReshapeWithBroadcastPass>());

  stage_1_pm.AddPass(pir::CreateDeadCodeEliminationPass());
  stage_1_pm.AddPass(pir::CreateBuildCinnPass());
  stage_1_pm.AddPass(cinn::dialect::ir::CreateAddBroadcastToElementwisePass());

  PADDLE_ENFORCE_EQ(
      stage_1_pm.Run(program),
      true,
      common::errors::Unavailable("stage_1_pm fail to run program"));

  pir::PassManager stage_2_pm(ctx);
  stage_2_pm.AddPass(cinn::dialect::ir::CreateAddStoreInGroupOpPass());
  stage_2_pm.AddPass(cinn::dialect::ir::CreateCinnGroupClusterPass());
  stage_2_pm.AddPass(pir::CreateDeadCodeEliminationPass());
  stage_2_pm.AddPass(cinn::dialect::ir::CreateLowerCinnFusionOpPass());

  PADDLE_ENFORCE_EQ(
      stage_2_pm.Run(program),
      true,
      common::errors::Unavailable("stage_2_pm fail to run program"));

  phi::Place place = phi::GPUPlace(0);

  auto kernel_program = paddle::dialect::PdOpLowerToKernelPass(program, place);

  paddle::framework::Scope exe_scope;

  paddle::framework::InterpreterCore executor(
      place, {"out@fetch"}, kernel_program->block(), &exe_scope);

  executor.Run({}, true);

  auto out_tensor =
      executor.local_scope()->FindVar("out@fetch")->Get<phi::DenseTensor>();

  if (check_result) {
    bool res0 = simple_cmp(out_tensor.data<float>()[0], gt_val);
    EXPECT_EQ(res0, true);
  }
}

std::shared_ptr<::pir::Program> BuildGroupProgram() {
  ::pir::IrContext* ctx = ::pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();

  auto program = std::make_shared<::pir::Program>(ctx);
  ::pir::Builder builder = ::pir::Builder(ctx, program->block());

  // full -> softmax(max -> subtract -> exp -> sum -> divide)
  const float value_one = 1.0;
  const std::vector<int64_t> shape = {128, 128, 768};
  auto x = builder
               .Build<paddle::dialect::FullOp>(
                   shape, value_one, phi::DataType::FLOAT32, phi::GPUPlace())
               .result(0);

  auto max =
      builder.Build<paddle::dialect::MaxOp>(x, std::vector<int64_t>{-1}, true)
          .result(0);
  auto sub = builder.Build<paddle::dialect::SubtractOp>(x, max).result(0);
  auto exp = builder.Build<paddle::dialect::ExpOp>(sub).result(0);
  auto sum =
      builder
          .Build<paddle::dialect::SumOp>(
              exp, std::vector<int64_t>{-1}, phi::DataType::FLOAT32, true)
          .result(0);
  auto out = builder.Build<paddle::dialect::DivideOp>(exp, sum).result(0);

  builder.Build<paddle::dialect::FetchOp>(out, "out", 0);
  return program;
}

// TEST(GroupOp, TestBuild) {
//   // Step 1: Construct pir::Program
//   ::pir::IrContext* ctx = ::pir::IrContext::Instance();
//   std::shared_ptr<::pir::Program> program = BuildGroupProgram();

//   RunAndCheckResult(program.get(), true, 1.0 / 768);
// }

// std::shared_ptr<::pir::Program> BuildLayerNormProgram() {
//   ::pir::IrContext* ctx = ::pir::IrContext::Instance();
//   ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
//   auto program = std::make_shared<::pir::Program>(ctx);
//   ::pir::Builder builder = ::pir::Builder(ctx, program->block());

//   std::vector<int64_t> axes{-1};
//   auto x =
//       builder
//           .Build<paddle::dialect::FullOp>(std::vector<int64_t>({128, 128,
//           768}),
//                                           1.0,
//                                           phi::DataType::FLOAT32,
//                                           phi::GPUPlace())
//           .result(0);

//   auto bias = builder
//                   .Build<paddle::dialect::FullOp>(std::vector<int64_t>({768}),
//                                                   1.0,
//                                                   phi::DataType::FLOAT32,
//                                                   phi::GPUPlace())
//                   .result(0);

//   auto scale = builder
//                    .Build<paddle::dialect::FullOp>(std::vector<int64_t>({768}),
//                                                    1.0,
//                                                    phi::DataType::FLOAT32,
//                                                    phi::GPUPlace())
//                    .result(0);

//   auto num = builder
//                  .Build<paddle::dialect::FullOp>(std::vector<int64_t>{1},
//                                                  768.0,
//                                                  phi::DataType::FLOAT32,
//                                                  phi::CPUPlace())
//                  .result(0);
//   auto eps = builder
//                  .Build<paddle::dialect::FullOp>(std::vector<int64_t>{1},
//                                                  1e-5,
//                                                  phi::DataType::FLOAT32,
//                                                  phi::CPUPlace())
//                  .result(0);

//   auto sum =
//       builder
//           .Build<paddle::dialect::SumOp>(x, axes, phi::DataType::FLOAT32,
//           true) .result(0);

//   auto mean = builder.Build<paddle::dialect::DivideOp>(sum, num).result(0);
//   auto power = builder.Build<paddle::dialect::MultiplyOp>(x, x).result(0);
//   auto power_sum = builder
//                        .Build<paddle::dialect::SumOp>(
//                            power, axes, phi::DataType::FLOAT32, true)
//                        .result(0);
//   auto mean2 =
//       builder.Build<paddle::dialect::DivideOp>(power_sum, num).result(0);
//   auto power_mean =
//       builder.Build<paddle::dialect::MultiplyOp>(mean, mean).result(0);

//   auto var =
//       builder.Build<paddle::dialect::SubtractOp>(mean2,
//       power_mean).result(0);

//   auto sub = builder.Build<paddle::dialect::SubtractOp>(x, mean).result(0);
//   auto t1 = builder.Build<paddle::dialect::AddOp>(var, eps).result(0);
//   auto t2 = builder.Build<paddle::dialect::SqrtOp>(t1).result(0);
//   auto t3 = builder.Build<paddle::dialect::DivideOp>(sub, t2).result(0);
//   auto t5 = builder.Build<paddle::dialect::MultiplyOp>(t3, scale).result(0);
//   auto out = builder.Build<paddle::dialect::MultiplyOp>(t5, bias).result(0);

//   builder.Build<paddle::dialect::FetchOp>(out, "out", 0);
//   return program;
// }

// TEST(GroupOp, TestBuildLayerNorm) {
//   // Step 1: Construct pir::Program
//   ::pir::IrContext* ctx = ::pir::IrContext::Instance();
//   std::shared_ptr<::pir::Program> program = BuildLayerNormProgram();

//   RunAndCheckResult(program.get(), false);
// }

// std::shared_ptr<::pir::Program> BuildDropOutProgram() {
//   ::pir::IrContext* ctx = ::pir::IrContext::Instance();
//   ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
//   auto program = std::make_shared<::pir::Program>(ctx);
//   ::pir::Builder builder = ::pir::Builder(ctx, program->block());

//   auto x =
//       builder
//           .Build<paddle::dialect::FullOp>(std::vector<int64_t>({128, 128,
//           768}),
//                                           1.0,
//                                           phi::DataType::FLOAT32,
//                                           phi::GPUPlace())
//           .result(0);

//   auto prob = builder
//                   .Build<paddle::dialect::FullOp>(std::vector<int64_t>({1}),
//                                                   0.5,
//                                                   phi::DataType::FLOAT32,
//                                                   phi::GPUPlace())
//                   .result(0);

//   auto random = builder
//                     .Build<paddle::dialect::UniformOp>(
//                         std::vector<int64_t>({128, 128, 768}),
//                         phi::DataType::FLOAT32,
//                         0.0,
//                         1.0,
//                         0,
//                         phi::GPUPlace())
//                     .result(0);

//   auto mask =
//       builder.Build<paddle::dialect::GreaterThanOp>(random, prob).result(0);
//   auto mask1 =
//       builder.Build<paddle::dialect::CastOp>(mask, phi::DataType::FLOAT32)
//           .result(0);
//   auto mul = builder.Build<paddle::dialect::MultiplyOp>(x, mask1).result(0);
//   auto neg_prob = prob =
//       builder
//           .Build<paddle::dialect::FullOp>(std::vector<int64_t>({1}),
//                                           0.5,
//                                           phi::DataType::FLOAT32,
//                                           phi::GPUPlace())
//           .result(0);
//   auto out = builder.Build<paddle::dialect::DivideOp>(mul,
//   neg_prob).result(0);

//   builder.Build<paddle::dialect::FetchOp>(out, "out", 0);
//   return program;
// }

// TEST(GroupOp, TestBuildDropout) {
//   // Step 1: Construct pir::Program
//   ::pir::IrContext* ctx = ::pir::IrContext::Instance();
//   std::shared_ptr<::pir::Program> program = BuildDropOutProgram();

//   RunAndCheckResult(program.get(), false);
// }

std::shared_ptr<::pir::Program> BuildScaleGroupProgram() {
  ::pir::IrContext* ctx = ::pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();

  auto program = std::make_shared<::pir::Program>(ctx);
  ::pir::Builder builder = ::pir::Builder(ctx, program->block());

  // full -> softmax(max -> subtract -> exp -> sum -> divide)
  const float value_one = 1.0;
  const std::vector<int64_t> shape = {16, 16};
  auto x = builder
               .Build<paddle::dialect::FullOp>(
                   shape, value_one, phi::DataType::FLOAT32, phi::GPUPlace())
               .result(0);

  auto out =
      builder.Build<paddle::dialect::ScaleOp>(x, 0.5, 0.0, false).result(0);

  builder.Build<paddle::dialect::FetchOp>(out, "out", 0);
  return program;
}

TEST(GroupOp, TestBuildScale) {
  // Step 1: Construct pir::Program
  ::pir::IrContext* ctx = ::pir::IrContext::Instance();
  std::shared_ptr<::pir::Program> program = BuildScaleGroupProgram();

  RunAndCheckResult(program.get(), true, 0.5);
}

// std::shared_ptr<::pir::Program> BuildScaleTensorGroupProgram() {
//   ::pir::IrContext* ctx = ::pir::IrContext::Instance();
//   ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();

//   auto program = std::make_shared<::pir::Program>(ctx);
//   ::pir::Builder builder = ::pir::Builder(ctx, program->block());

//   // full -> softmax(max -> subtract -> exp -> sum -> divide)
//   const float value_one = 0.5;
//   const std::vector<int64_t> shape = {16, 16};
//   auto x = builder
//                .Build<paddle::dialect::FullOp>(
//                    shape, value_one, phi::DataType::FLOAT32, phi::GPUPlace())
//                .result(0);
//   auto scale = builder
//                    .Build<paddle::dialect::FullOp>(std::vector<int64_t>({1}),
//                                                    0.0,
//                                                    phi::DataType::FLOAT32,
//                                                    phi::GPUPlace())
//                    .result(0);
//   auto factor = builder.Build<paddle::dialect::CosOp>(scale).result(0);
//   auto out =
//       builder.Build<paddle::dialect::ScaleOp>(x, factor, 0.0,
//       false).result(0);

//   builder.Build<paddle::dialect::FetchOp>(out, "out", 0);
//   return program;
// }

// TEST(GroupOp, TestBuildScaleTensor) {
//   // Step 1: Construct pir::Program
//   ::pir::IrContext* ctx = ::pir::IrContext::Instance();
//   std::shared_ptr<::pir::Program> program = BuildScaleTensorGroupProgram();

//   RunAndCheckResult(program.get(), true, 0.5);
// }

// std::shared_ptr<::pir::Program> BuildPowerProgram() {
//   ::pir::IrContext* ctx = ::pir::IrContext::Instance();
//   ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
//   auto program = std::make_shared<::pir::Program>(ctx);
//   ::pir::Builder builder = ::pir::Builder(ctx, program->block());

//   auto x = builder
//                .Build<paddle::dialect::FullOp>(std::vector<int64_t>({16,
//                16}),
//                                                2.0,
//                                                phi::DataType::FLOAT32,
//                                                phi::GPUPlace())
//                .result(0);

//   auto factor =
//       builder
//           .Build<paddle::dialect::FullOp>(std::vector<int64_t>({16, 16}),
//                                           2.0,
//                                           phi::DataType::FLOAT32,
//                                           phi::GPUPlace())
//           .result(0);

//   auto power1 =
//       builder.Build<paddle::dialect::ElementwisePowOp>(x, factor).result(0);

//   auto power2 = builder.Build<paddle::dialect::PowOp>(power1, 2.0).result(0);
//   auto out =
//       builder
//           .Build<paddle::dialect::ReshapeOp>(power2,
//           std::vector<int64_t>({-1})) .result(0);

//   builder.Build<paddle::dialect::FetchOp>(out, "out", 0);
//   return program;
// }

// TEST(GroupOp, TestBuildPower) {
//   // Step 1: Construct pir::Program
//   ::pir::IrContext* ctx = ::pir::IrContext::Instance();
//   std::shared_ptr<::pir::Program> program = BuildPowerProgram();

//   RunAndCheckResult(program.get(), true, 16.0);
// }

// std::shared_ptr<::pir::Program> BuildLayerNorm2Program() {
//   ::pir::IrContext* ctx = ::pir::IrContext::Instance();
//   ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
//   auto program = std::make_shared<::pir::Program>(ctx);
//   ::pir::Builder builder = ::pir::Builder(ctx, program->block());

//   std::vector<int64_t> axes{-1};
//   auto x =
//       builder
//           .Build<paddle::dialect::FullOp>(std::vector<int64_t>({128, 128,
//           768}),
//                                           1.0,
//                                           phi::DataType::FLOAT32,
//                                           phi::GPUPlace())
//           .result(0);

//   auto bias = builder
//                   .Build<paddle::dialect::FullOp>(std::vector<int64_t>({768}),
//                                                   1.0,
//                                                   phi::DataType::FLOAT32,
//                                                   phi::GPUPlace())
//                   .result(0);

//   auto scale = builder
//                    .Build<paddle::dialect::FullOp>(std::vector<int64_t>({768}),
//                                                    1.0,
//                                                    phi::DataType::FLOAT32,
//                                                    phi::GPUPlace())
//                    .result(0);

//   auto num =
//       builder
//           .Build<paddle::dialect::FullOp>(std::vector<int64_t>{128, 128, 1},
//                                           768.0,
//                                           phi::DataType::FLOAT32,
//                                           phi::CPUPlace())
//           .result(0);
//   auto sum =
//       builder
//           .Build<paddle::dialect::SumOp>(x, axes, phi::DataType::FLOAT32,
//           true) .result(0);

//   auto mean = builder.Build<paddle::dialect::DivideOp>(sum, num).result(0);

//   auto diff = builder.Build<paddle::dialect::SubtractOp>(x, mean).result(0);

//   auto power = builder.Build<paddle::dialect::MultiplyOp>(diff,
//   diff).result(0); auto power_sum = builder
//                        .Build<paddle::dialect::SumOp>(
//                            power, axes, phi::DataType::FLOAT32, true)
//                        .result(0);
//   auto num2 =
//       builder
//           .Build<paddle::dialect::FullOp>(std::vector<int64_t>{128, 128, 1},
//                                           768.0,
//                                           phi::DataType::FLOAT32,
//                                           phi::CPUPlace())
//           .result(0);
//   auto var2 =
//       builder.Build<paddle::dialect::DivideOp>(power_sum, num2).result(0);

//   auto t1 = builder.Build<paddle::dialect::ScaleOp>(var2, 1.0,
//   1e-5).result(0); auto factor = builder
//                     .Build<paddle::dialect::FullOp>(std::vector<int64_t>{1},
//                                                     -0.5,
//                                                     phi::DataType::FLOAT32,
//                                                     phi::CPUPlace())
//                     .result(0);
//   auto t2 =
//       builder.Build<paddle::dialect::ElementwisePowOp>(t1, factor).result(0);
//   // auto t2 = builder.Build<paddle::dialect::RsqrtOp>(t1).result(0);
//   auto t3 = builder.Build<paddle::dialect::MultiplyOp>(diff, t2).result(0);
//   auto t5 = builder.Build<paddle::dialect::MultiplyOp>(t3, scale).result(0);
//   auto out = builder.Build<paddle::dialect::AddOp>(t5, bias).result(0);
//   auto mean_out =
//       builder
//           .Build<paddle::dialect::ReshapeOp>(mean,
//           std::vector<int64_t>({-1})) .result(0);
//   auto mean2_out =
//       builder
//           .Build<paddle::dialect::ReshapeOp>(var2,
//           std::vector<int64_t>({-1})) .result(0);

//   builder.Build<paddle::dialect::FetchOp>(out, "out", 0);
//   builder.Build<paddle::dialect::FetchOp>(mean_out, "mean", 0);
//   builder.Build<paddle::dialect::FetchOp>(mean2_out, "var", 0);
//   return program;
// }

// TEST(GroupOp, TestBuildLayerNorm2) {
//   // Step 1: Construct pir::Program
//   ::pir::IrContext* ctx = ::pir::IrContext::Instance();
//   std::shared_ptr<::pir::Program> program = BuildLayerNorm2Program();

//   RunAndCheckResult(program.get(), false);
// }

// std::shared_ptr<::pir::Program> BuildSum2GroupProgram() {
//   ::pir::IrContext* ctx = ::pir::IrContext::Instance();
//   ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
//   auto program = std::make_shared<::pir::Program>(ctx);
//   ::pir::Builder builder = ::pir::Builder(ctx, program->block());

//   auto x = builder
//                .Build<paddle::dialect::FullOp>(std::vector<int64_t>({16,
//                16}),
//                                                0.0,
//                                                phi::DataType::FLOAT32,
//                                                phi::GPUPlace())
//                .result(0);

//   auto cos = builder.Build<paddle::dialect::CosOp>(x).result(0);

//   auto y = builder
//                .Build<paddle::dialect::FullOp>(std::vector<int64_t>({8, 8}),
//                                                0.0,
//                                                phi::DataType::FLOAT32,
//                                                phi::GPUPlace())
//                .result(0);

//   auto sin = builder.Build<paddle::dialect::SinOp>(y).result(0);

//   builder.Build<paddle::dialect::FetchOp>(cos, "out", 0);
//   builder.Build<paddle::dialect::FetchOp>(sin, "out2", 0);
//   return program;
// }

// TEST(GroupOp, TestBuildSum2Group) {
//   // Step 1: Construct pir::Program
//   ::pir::IrContext* ctx = ::pir::IrContext::Instance();
//   std::shared_ptr<::pir::Program> program = BuildSum2GroupProgram();

//   RunAndCheckResult(program.get(), true, 1.0);
// }

// std::shared_ptr<::pir::Program> BuildConcatProgram() {
//   ::pir::IrContext* ctx = ::pir::IrContext::Instance();
//   ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
//   auto program = std::make_shared<::pir::Program>(ctx);
//   ::pir::Builder builder = ::pir::Builder(ctx, program->block());

//   auto x = builder
//                .Build<paddle::dialect::FullOp>(std::vector<int64_t>({16,
//                16}),
//                                                2.0,
//                                                phi::DataType::FLOAT32,
//                                                phi::GPUPlace())
//                .result(0);

//   auto y = builder
//                .Build<paddle::dialect::FullOp>(std::vector<int64_t>({16,
//                16}),
//                                                2.0,
//                                                phi::DataType::FLOAT32,
//                                                phi::GPUPlace())
//                .result(0);

//   auto t1 =
//       builder.Build<pir::CombineOp>(std::vector<pir::Value>({x,
//       y})).result(0);

//   auto out = builder.Build<paddle::dialect::ConcatOp>(t1, 1).result(0);

//   builder.Build<paddle::dialect::FetchOp>(out, "out", 0);
//   return program;
// }

// TEST(GroupOp, TestBuildConcat) {
//   // Step 1: Construct pir::Program
//   ::pir::IrContext* ctx = ::pir::IrContext::Instance();
//   std::shared_ptr<::pir::Program> program = BuildConcatProgram();

//   RunAndCheckResult(program.get(), true, 2.0);
// }

// std::shared_ptr<::pir::Program> BuildSliceProgram() {
//   ::pir::IrContext* ctx = ::pir::IrContext::Instance();
//   ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
//   auto program = std::make_shared<::pir::Program>(ctx);
//   ::pir::Builder builder = ::pir::Builder(ctx, program->block());

//   auto x = builder
//                .Build<paddle::dialect::FullOp>(std::vector<int64_t>({16,
//                16}),
//                                                2.0,
//                                                phi::DataType::FLOAT32,
//                                                phi::GPUPlace())
//                .result(0);

//   auto out = builder
//                  .Build<paddle::dialect::SliceOp>(x,
//                                                   std::vector<int64_t>({1}),
//                                                   std::vector<int64_t>({0}),
//                                                   std::vector<int64_t>({2}),
//                                                   std::vector<int64_t>({}),
//                                                   std::vector<int64_t>({}))
//                  .result(0);

//   builder.Build<paddle::dialect::FetchOp>(out, "out", 0);
//   return program;
// }

// TEST(GroupOp, TestBuildSlice) {
//   // Step 1: Construct pir::Program
//   ::pir::IrContext* ctx = ::pir::IrContext::Instance();
//   std::shared_ptr<::pir::Program> program = BuildSliceProgram();

//   RunAndCheckResult(program.get(), true, 2.0);
// }

// std::shared_ptr<::pir::Program> BuildSplitProgram() {
//   ::pir::IrContext* ctx = ::pir::IrContext::Instance();
//   ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
//   auto program = std::make_shared<::pir::Program>(ctx);
//   ::pir::Builder builder = ::pir::Builder(ctx, program->block());

//   auto x = builder
//                .Build<paddle::dialect::FullOp>(std::vector<int64_t>({16,
//                16}),
//                                                2.0,
//                                                phi::DataType::FLOAT32,
//                                                phi::GPUPlace())
//                .result(0);

//   auto out_arr =
//       builder.Build<paddle::dialect::SplitWithNumOp>(x, 4, 1).result(0);
//   auto out = builder.Build<pir::SliceOp>(out_arr, 0).result(0);
//   builder.Build<paddle::dialect::FetchOp>(out, "out", 0);
//   return program;
// }

// TEST(GroupOp, TestBuildSplit) {
//   // Step 1: Construct pir::Program
//   ::pir::IrContext* ctx = ::pir::IrContext::Instance();
//   std::shared_ptr<::pir::Program> program = BuildSplitProgram();

//   RunAndCheckResult(program.get(), true, 2.0);
// }

// std::shared_ptr<::pir::Program> BuildAddNProgram() {
//   ::pir::IrContext* ctx = ::pir::IrContext::Instance();
//   ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
//   auto program = std::make_shared<::pir::Program>(ctx);
//   ::pir::Builder builder = ::pir::Builder(ctx, program->block());

//   auto x = builder
//                .Build<paddle::dialect::FullOp>(std::vector<int64_t>({16,
//                16}),
//                                                2.0,
//                                                phi::DataType::FLOAT32,
//                                                phi::GPUPlace())
//                .result(0);

//   auto y = builder
//                .Build<paddle::dialect::FullOp>(std::vector<int64_t>({16,
//                16}),
//                                                2.0,
//                                                phi::DataType::FLOAT32,
//                                                phi::GPUPlace())
//                .result(0);

//   auto z = builder
//                .Build<paddle::dialect::FullOp>(std::vector<int64_t>({16,
//                16}),
//                                                2.0,
//                                                phi::DataType::FLOAT32,
//                                                phi::GPUPlace())
//                .result(0);

//   auto t1 = builder.Build<pir::CombineOp>(std::vector<pir::Value>({x, y, z}))
//                 .result(0);

//   auto out = builder.Build<paddle::dialect::AddNOp>(t1).result(0);

//   builder.Build<paddle::dialect::FetchOp>(out, "out", 0);
//   return program;
// }

// TEST(GroupOp, TestBuildAddN) {
//   // Step 1: Construct pir::Program
//   ::pir::IrContext* ctx = ::pir::IrContext::Instance();
//   std::shared_ptr<::pir::Program> program = BuildAddNProgram();

//   RunAndCheckResult(program.get(), true, 6.0);
// }

// std::shared_ptr<::pir::Program> BuildSplitSectionProgram() {
//   ::pir::IrContext* ctx = ::pir::IrContext::Instance();
//   ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
//   auto program = std::make_shared<::pir::Program>(ctx);
//   ::pir::Builder builder = ::pir::Builder(ctx, program->block());

//   auto x = builder
//                .Build<paddle::dialect::FullOp>(std::vector<int64_t>({16,
//                16}),
//                                                2.0,
//                                                phi::DataType::FLOAT32,
//                                                phi::GPUPlace())
//                .result(0);

//   auto split_arr = builder
//                        .Build<paddle::dialect::SplitOp>(
//                            x, std::vector<int64_t>({3, 5, 8}), -1)
//                        .out();
//   auto out = builder.Build<pir::SliceOp>(split_arr, 0).result(0);
//   builder.Build<paddle::dialect::FetchOp>(out, "out", 0);
//   return program;
// }

// TEST(GroupOp, TestBuildSplitSection) {
//   // Step 1: Construct pir::Program
//   ::pir::IrContext* ctx = ::pir::IrContext::Instance();
//   std::shared_ptr<::pir::Program> program = BuildSplitSectionProgram();

//   RunAndCheckResult(program.get(), 2.0);
// }

// std::shared_ptr<::pir::Program> BuildReshapeSumProgram() {
//   ::pir::IrContext* ctx = ::pir::IrContext::Instance();
//   ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
//   auto program = std::make_shared<::pir::Program>(ctx);
//   ::pir::Builder builder = ::pir::Builder(ctx, program->block());

//   auto x = builder
//                .Build<paddle::dialect::FullOp>(
//                    std::vector<int64_t>({128 * 128, 768}),
//                    1.0,
//                    phi::DataType::FLOAT32,
//                    phi::GPUPlace())
//                .result(0);
//   auto sum = builder
//                  .Build<paddle::dialect::SumOp>(
//                      x, std::vector<int64_t>{0}, phi::DataType::FLOAT32,
//                      true)
//                  .result(0);

//   builder.Build<paddle::dialect::FetchOp>(sum, "out", 0);
//   return program;
// }

// TEST(GroupOp, TestBuildReshapeSum) {
//   // Step 1: Construct pir::Program
//   ::pir::IrContext* ctx = ::pir::IrContext::Instance();
//   std::shared_ptr<::pir::Program> program = BuildReshapeSumProgram();

//   RunAndCheckResult(program.get(), true, 128 * 128);
// }
