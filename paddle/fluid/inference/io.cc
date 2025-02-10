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

#include "paddle/fluid/inference/io.h"

#include <algorithm>
#include <fstream>
#include <vector>

#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/feed_fetch_type.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/version.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/pybind/pybind.h"
#include "paddle/phi/core/platform/cpu_helper.h"

// phi
#include "paddle/phi/kernels/declarations.h"

PD_DEFINE_string(devices,  // NOLINT
                 "",
                 "The devices to be used which is joined by comma.");
PD_DEFINE_int32(math_num_threads,
                1,
                "Number of threads used to run math functions.");

namespace paddle::inference {

void Init(const std::vector<std::string> argv) {
  framework::InitGflags(argv);
  platform::SetNumThreads(FLAGS_math_num_threads);
  // init devices
  std::vector<int> devices;
  std::string token;
  std::istringstream tokenStream(FLAGS_devices);
  while (std::getline(tokenStream, token, ',')) {
    devices.push_back(std::stoi(token));
  }
  framework::InitDevices(devices);
}

void ReadBinaryFile(const std::string& filename, std::string* contents) {
  std::ifstream fin(filename, std::ios::in | std::ios::binary);
  PADDLE_ENFORCE_EQ(
      fin.is_open(),
      true,
      common::errors::Unavailable("Failed to open file %s.", filename));
  fin.seekg(0, std::ios::end);
  contents->clear();
  contents->resize(fin.tellg());
  fin.seekg(0, std::ios::beg);
  fin.read(&(contents->at(0)), contents->size());  // NOLINT
  fin.close();
}

bool IsPersistable(const framework::VarDesc* var) {
  if (var->Persistable() &&
      var->GetType() != framework::proto::VarType::FEED_MINIBATCH &&
      var->GetType() != framework::proto::VarType::FETCH_LIST &&
      var->GetType() != framework::proto::VarType::RAW) {
    return true;
  }
  return false;
}

void LoadCombinePersistables(framework::Executor* executor,
                             framework::Scope* scope,
                             const framework::ProgramDesc& main_program,
                             const std::string& dirname,
                             const std::string& param_filename,
                             bool model_from_memory = false) {
  if (param_filename.empty()) {
    VLOG(4)
        << "param_filename is empty when load combine params. Return directly.";
    return;
  }
  const framework::BlockDesc& global_block = main_program.Block(0);

  auto load_program = std::make_unique<framework::ProgramDesc>();
  auto load_block = load_program->MutableBlock(0);
  std::vector<std::string> param_list;

  for (auto* var : global_block.AllVars()) {
    if (IsPersistable(var)) {
      VLOG(4) << "persistable variable's name: " << var->Name();

      framework::VarDesc* new_var = load_block->Var(var->Name());
      new_var->SetShape(var->GetShape());
      new_var->SetDataType(var->GetDataType());
      auto var_type = var->GetType();
      new_var->SetType(var_type);

      if ((var_type !=
           framework::proto::VarType::Type::VarType_Type_SELECTED_ROWS) &&
          (var_type != framework::proto::VarType::VOCAB)) {
        new_var->SetLoDLevel(var->GetLoDLevel());
      }

      new_var->SetPersistable(true);
      param_list.push_back(new_var->Name());
    }
  }

  // sort param_list to have consistent ordering
  std::sort(param_list.begin(), param_list.end());
  // append just the load_combine op
  framework::OpDesc* op = load_block->AppendOp();
  op->SetType("load_combine");
  op->SetOutput("Out", param_list);
  op->SetAttr("file_path", {param_filename});
  op->SetAttr("model_from_memory", {model_from_memory});
  op->CheckAttrs();

  executor->Run(*load_program, scope, 0, true, true);
}

void LoadSeparatePersistables(framework::Executor* executor,
                              framework::Scope* scope,
                              const framework::ProgramDesc& main_program,
                              const std::string& dirname) {
  const framework::BlockDesc& global_block = main_program.Block(0);
  std::vector<framework::VarDesc*> persistable_vars;
  for (auto* var : global_block.AllVars()) {
    if (IsPersistable(var)) {
      persistable_vars.push_back(var);
    }
  }
  if (persistable_vars.empty()) {
    return;
  }

  size_t num_threads = 8;
  size_t chunk_size =
      std::max(static_cast<size_t>(1), persistable_vars.size() / num_threads);
  num_threads = std::min(num_threads, persistable_vars.size() / chunk_size);
  size_t remains_size = persistable_vars.size() % num_threads;
  VLOG(4) << "Start Load with multi-thread: " << num_threads
          << " chund size: " << chunk_size;

  auto load_handler = [&](const std::vector<framework::VarDesc*>& vars) {
    if (vars.empty()) {
      return;
    }
    auto load_program = std::make_unique<framework::ProgramDesc>();
    auto load_block = load_program->MutableBlock(0);

    for (auto* var : vars) {
      VLOG(4) << "persistable variable's name: " << var->Name();

      framework::VarDesc* new_var = load_block->Var(var->Name());
      new_var->SetShape(var->GetShape());
      new_var->SetDataType(var->GetDataType());
      auto var_type = var->GetType();
      new_var->SetType(var_type);

      if ((var_type !=
           framework::proto::VarType::Type::VarType_Type_SELECTED_ROWS) &&
          (var_type != framework::proto::VarType::VOCAB)) {
        new_var->SetLoDLevel(var->GetLoDLevel());
      }

      new_var->SetPersistable(true);

      // append_op
      framework::OpDesc* op = load_block->AppendOp();
      op->SetType("load");
      op->SetOutput("Out", {new_var->Name()});
      op->SetAttr("file_path", {dirname + "/" + new_var->Name()});
      op->CheckAttrs();
    }
    executor->Run(*load_program, scope, 0, true, true);
  };

  std::vector<std::future<void>> futures;
  for (size_t i = 0; i < num_threads; ++i) {
    auto start_it = persistable_vars.begin() + i * chunk_size;
    auto end_it = start_it + chunk_size;
    futures.push_back(
        std::async(std::launch::async,
                   load_handler,
                   std::vector<framework::VarDesc*>(start_it, end_it)));
  }
  if (remains_size > 0) {
    futures.push_back(
        std::async(std::launch::async,
                   load_handler,
                   std::vector<framework::VarDesc*>(
                       persistable_vars.rbegin(),
                       persistable_vars.rbegin() + remains_size)));
  }
  for (auto& future : futures) {
    future.wait();
  }
}

std::unique_ptr<framework::ProgramDesc> Load(framework::Executor* executor,
                                             framework::Scope* scope,
                                             const std::string& dirname) {
  std::string model_filename = dirname + "/__model__";
  std::string program_desc_str;
  VLOG(3) << "loading model from " << model_filename;
  ReadBinaryFile(model_filename, &program_desc_str);

  std::unique_ptr<framework::ProgramDesc> main_program(
      new framework::ProgramDesc(program_desc_str));

  // model_from_memory is false in separate parameters.
  LoadSeparatePersistables(executor, scope, *main_program, dirname);
  return main_program;
}

std::unique_ptr<framework::ProgramDesc> Load(framework::Executor* executor,
                                             framework::Scope* scope,
                                             const std::string& prog_filename,
                                             const std::string& param_filename,
                                             bool load_params) {
  std::string program_desc_str;
  ReadBinaryFile(prog_filename, &program_desc_str);

  std::unique_ptr<framework::ProgramDesc> main_program(
      new framework::ProgramDesc(program_desc_str));

  if (load_params) {
    LoadCombinePersistables(executor,
                            scope,
                            *main_program,
                            "",
                            param_filename,
                            false /* model_from_memory */);
  }
  return main_program;
}

std::unique_ptr<framework::ProgramDesc> LoadFromMemory(
    framework::Executor* executor,
    framework::Scope* scope,
    const std::string& prog_buffer,
    const std::string& param_buffer) {
  std::unique_ptr<framework::ProgramDesc> main_program(
      new framework::ProgramDesc(prog_buffer));

  LoadCombinePersistables(executor,
                          scope,
                          *main_program,
                          "",
                          param_buffer,
                          true /* model_filename */);
  return main_program;
}

void SaveVars(const framework::Scope& scope,
              const std::vector<std::string>& vars,
              const std::string& dirname,
              bool predicate) {
  framework::ProgramDesc prog;
  auto* block = prog.MutableBlock(0);
  auto* op = block->AppendOp();
  op->SetType("save_combine");
  op->SetInput("X", vars);
  op->SetAttr("file_path", dirname + "/param");
  op->CheckAttrs();

  phi::CPUPlace place;
  framework::Executor exe(place);
  exe.Run(prog, const_cast<framework::Scope*>(&scope), 0, true, true);
}

}  // namespace paddle::inference
