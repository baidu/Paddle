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

#pragma once

#include <optional>

#include "paddle/cinn/common/common.h"
#include "paddle/cinn/ir/lowered_func.h"

namespace cinn {
namespace hlir {
namespace framework {

// An enum class used to control the compilation stage.
enum class CompilationStage {
  // Fully compiled by default, the following compilation result can be
  // obtained: lowered_function, source_code, source_ptx, instruction and
  // runtime_program.
  DEFAULT = 0,
  // Just do lowering, we can only get lowered_function from compilation result.
  LOWERING = 1,
  // Stop after codegen and jit, we can get: lowered_function, source_code and
  // source_ptx from compilation result.
  CODEGEN_AND_JIT = 2,
  // Stop after build instruction, we can get: lowered_function, source_code,
  // source_ptx and runtime_program from compilation result.
  BUILD_INSTRUCTION = 3,
};

// An enum class used to represent the compilation status.
enum class CompilationStatus {
  // An unknown error occurred during compilation.
  UNKNOWN_FAIL = 0,
  // An error occurred during lowering.
  LOWERING_FAIL = 1,
  // An error occurred during codegen and jit.
  CODEGEN_JIT_FAIL = 2,
  // An error occurred during build instruction.
  INSTRUCTION_FAIL = 3,
  // An error occurred during build runtime program.
  PROGRAM_FAIL = 4,
  // Compile successfully.
  SUCCESS = 5,
};

class GraphCompiler;

class CompilationResult {
  friend class GraphCompiler;

 public:
  void InitCompilationResult(int group_size);

  // Setters
  void SetStatus(int idx, const CompilationStatus& status);
  void SetMessage(int idx, const std::string& message);
  void SetLoweredFuncs(int idx, const std::vector<ir::LoweredFunc>& funcs);
  void SetSourceCode(int idx, const std::string& source_code);
  void SetSourcePtx(int idx, const std::string& source_ptx);

  // Getters
  bool IsSuccess() const;
  int Size() const { return size_; }
  CompilationStatus Status() const;
  CompilationStatus Status(int idx) const;
  std::string Message() const;
  std::string Message(int idx) const;
  std::vector<std::vector<ir::LoweredFunc>> LoweredFuncs() const;
  std::vector<ir::LoweredFunc> LoweredFuncs(int idx) const;
  std::vector<std::string> SourceCodes() const;
  std::string SourceCode(int idx) const;
  std::vector<std::string> SourcePtxs() const;
  std::string SourcePtx(int idx) const;

 private:
  std::vector<CompilationStatus> status_;
  std::vector<std::string> messages_;
  std::vector<std::optional<std::vector<ir::LoweredFunc>>> lowered_funcs_;
  std::vector<std::optional<std::string>> source_codes_;
  std::vector<std::optional<std::string>> source_ptxs_;
  int size_;
};

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
