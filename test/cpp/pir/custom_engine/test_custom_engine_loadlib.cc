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

#include <gtest/gtest.h>

#include <array>
#include <string>

#include "paddle/fluid/custom_engine/custom_engine_manager.h"
#include "paddle/fluid/platform/init.h"
#include "paddle/phi/backends/custom/fake_cpu_device.h"
#include "paddle/phi/backends/device_manager.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/memory/allocation/allocator_facade.h"
#include "paddle/phi/core/platform/device_context.h"

#include "paddle/fluid/framework/new_executor/interpretercore.h"
#include "paddle/fluid/framework/new_executor/pir_interpreter.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/transforms/pd_op_to_kernel_pass.h"
#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/core/builtin_dialect.h"
#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/core/ir_context.h"
#include "paddle/pir/include/core/program.h"
#include "paddle/pir/include/core/type_id.h"
#include "test/cpp/pir/custom_engine/fake_cpu_engine.h"

#define OUT_NAME "program_out"

void RegisterDevice() {
  CustomRuntimeParams runtime_params;
  runtime_params.size = sizeof(CustomRuntimeParams);
  auto device_interface = std::make_unique<C_DeviceInterface>();
  runtime_params.interface = device_interface.get();
  std::memset(runtime_params.interface, 0, sizeof(C_DeviceInterface));
  runtime_params.interface->size = sizeof(C_DeviceInterface);

  InitFakeCPUDevice(&runtime_params);
  phi::LoadCustomRuntimeLib(
      runtime_params, std::move(device_interface), "", nullptr);
}

void RegisterEngine() {
  CustomEngineParams engine_params;
  std::memset(&engine_params, 0, sizeof(CustomEngineParams));
  engine_params.size = sizeof(CustomEngineParams);
  auto engine_interface = new (C_CustomEngineInterface);
  engine_params.interface = engine_interface;
  std::memset(engine_params.interface, 0, sizeof(C_CustomEngineInterface));
  engine_params.interface->size = sizeof(C_CustomEngineInterface);

  InitPluginCustomEngine(&engine_params);
  paddle::custom_engine::LoadCustomEngineLib("", &engine_params);
}

void InitCustom() {
  RegisterDevice();
  EXPECT_GT(static_cast<int>(phi::DeviceManager::GetAllDeviceTypes().size()),
            0);
  auto place = phi::CustomPlace(DEVICE_TYPE, 0);
  auto device = phi::DeviceManager::GetDeviceWithPlace(place);
  EXPECT_NE(device, nullptr);

  std::vector<phi::Place> places;
  auto device_types = phi::DeviceManager::GetAllDeviceTypes();
  for (auto dev_type : device_types) {
    auto devices = phi::DeviceManager::GetDeviceList(dev_type);
    for (auto dev_id : devices) {
      places.push_back(phi::PlaceHelper::CreatePlace(dev_type, dev_id));
    }
  }
  EXPECT_GT(static_cast<int>(places.size()), 0);

  places.emplace_back(phi::CPUPlace());
  phi::DeviceContextPool::Init(places);
  RegisterEngine();
}

void CreateProgram(pir::Program *program) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<pir::BuiltinDialect>();

  pir::Block *block = program->block();
  pir::Builder builder(ctx, block);

  auto full_op1 =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{2, 2}, 100);
  auto full_op2 =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{2, 2}, 10);

  auto buildin_combine_op = builder.Build<pir::CombineOp>(
      std::vector<pir::Value>{full_op1.result(0), full_op2.result(0)});

  auto engine_op = builder.Build<paddle::dialect::FakeEngineOp>(
      buildin_combine_op.result(0),
      std::vector<std::string>{"input_0", "input_1"},
      std::vector<std::string>{"output_0"},
      std::vector<std::vector<int64_t>>{{2, 2}},
      std::vector<phi::DataType>{phi::DataType::FLOAT32});
  engine_op->region(0).emplace_back();

  auto output = builder.Build<pir::SplitOp>(engine_op.result(0)).outputs()[0];
  builder.Build<pir::ShadowOutputOp>(output, OUT_NAME);

  return;
}

TEST(CustomDevice, Tensor) {
  paddle::framework::InitMemoryMethod();
  InitCustom();
  pir::IrContext *ctx = pir::IrContext::Instance();
  pir::Program *program = new pir::Program(ctx);
  CreateProgram(program);

  EXPECT_EQ(program->block()->size(), 6u);

  auto kernel_program = paddle::dialect::PdOpLowerToKernelPass(program);

  auto place = phi::CustomPlace(DEVICE_TYPE, 0);
  paddle::framework::Scope scope;
  paddle::framework::InterpreterCore test_core(
      place, {}, kernel_program->block(), &scope);
  test_core.SetSkipGcVars({OUT_NAME});
  test_core.Run({});

  auto out_tensor =
      test_core.local_scope() == nullptr
          ? scope.FindVar(OUT_NAME)->Get<phi::DenseTensor>()
          : test_core.local_scope()->FindVar(OUT_NAME)->Get<phi::DenseTensor>();
  bool res0 = out_tensor.data<float>()[0] == 100;
  EXPECT_EQ(res0, true);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
