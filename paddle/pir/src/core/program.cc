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

#include "paddle/pir/include/core/program.h"
#include <limits>
#include <mutex>
#include <random>
#include <unordered_set>
#include "glog/logging.h"
#include "paddle/pir/include/core/ir_context.h"

namespace pir {

namespace {

int64_t GetRandomId() {
  std::random_device rd{};
  std::mt19937_64 gen(rd());
  std::uniform_int_distribution<int64_t> dis(
      0, std::numeric_limits<int64_t>::max());
  return dis(gen);
}

bool InsertGlobalStorageSuccess(int64_t random_id) {
  static std::unordered_set<int64_t> storage;
  static std::mutex mutex;
  std::unique_lock<std::mutex> lock(mutex);
  return storage.emplace(random_id).second;
}

int64_t GetUniqueRandomId() {
  int kLimit = 100;
  for (int i = 0; i < kLimit; ++i) {
    int64_t random_id = GetRandomId();
    if (InsertGlobalStorageSuccess(random_id)) {
      return random_id;
    }
  }
  LOG(FATAL) << "Fatal bug occurred in GetUniqueRandomId().";
}

}  // namespace

Program::Program(IrContext* context) {
  module_ = ModuleOp::Create(context, this);
  id_ = GetUniqueRandomId();
}

Program::~Program() {
  if (module_) {
    module_.Destroy();
  }
}

std::shared_ptr<Program> Program::Clone(IrMapping& ir_mapping) const {
  pir::IrContext* ctx = pir::IrContext::Instance();
  auto new_program = std::make_shared<Program>(ctx);
  auto clone_options = CloneOptions::All();

  // deal kwargs
  for (auto [key, value] : block()->kwargs()) {
    auto new_arg = new_program->block()->AddKwarg(key, value.type());
    auto tmp_block_arg = value.dyn_cast<BlockArgument>();
    for (auto [name, attr_value] : tmp_block_arg.attributes()) {
      new_arg.set_attribute(name, attr_value);
    }
    ir_mapping.Add(value, new_arg);
  }

  for (const auto& op : *block()) {
    auto* new_op = op.Clone(ir_mapping, clone_options);
    new_program->block()->push_back(new_op);

    if (new_op->name() == "builtin.set_parameter") {
      std::unique_ptr<pir::Parameter> param_new(
          new Parameter(nullptr, 0, new_op->operand(0).type()));
      new_program->SetParameter(
          new_op->attribute<StrAttribute>("parameter_name").AsString(),
          std::move(param_new));
    }
  }
  return new_program;
}

void Program::CopyToBlock(IrMapping& ir_mapping, Block* insert_block) const {
  auto clone_options = CloneOptions::All();
  for (const auto& op : *block()) {
    bool skip_op = false;
    for (uint32_t i = 0; i < op.num_results(); i++) {
      if (ir_mapping.GetMutableMap<pir::Value>().count(op.result(i))) {
        skip_op = true;
        break;
      }
    }
    if (skip_op || op.isa<pir::ShadowOutputOp>()) {
      continue;
    }

    auto* new_op = op.Clone(ir_mapping, clone_options);
    insert_block->push_back(new_op);
  }
  return;
}

Parameter* Program::GetParameter(const std::string& name) const {
  if (parameters_.count(name) != 0) {
    return parameters_.at(name).get();
  }
  return nullptr;
}

void Program::SetParameter(const std::string& name,
                           std::shared_ptr<Parameter> parameter) {
  parameters_[name] = parameter;
}

}  // namespace pir
