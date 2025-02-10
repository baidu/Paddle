// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include <string>
#include <unordered_set>
#include <vector>

#include "paddle/cinn/backends/codegen_gpu_dev.h"

namespace cinn::ir {
class Module;
}  // namespace cinn::ir

namespace cinn {
namespace backends {
namespace sycl {

/**
 * SYCL device code generator.
 *
 * It generates the device function, e.g, the function called "myadd" will have
 * a __global__ function called "myadd_kernel", different from codegen_c, the
 * declaration of the "myadd_kernel" function has an expanded argument list,
 * which finally similar to `__global__ void myadd(float* __restrict__ A, float*
 * __restrict__ B, int n);`
 */
class CodeGenSyclDevice : public CodeGenGpuDev {
 public:
  explicit CodeGenSyclDevice(Target target);

  /**
   * Compile the \p module to \p outputs.
   */
  void Compile(const ir::Module& module, const Outputs& outputs);

  //! Compile on syclrtc.
  std::string Compile(const ir::Module& module, bool for_syclrtc = true);

  std::string Compile(const ir::Module& module, OutputKind output_kind);
  void Compile(const ir::LoweredFunc& func);
  static const std::string& GetSourceHeader();

 protected:
  void Visit(const ir::_Var_* op) override;
  void Visit(const ir::_LoweredFunc_* op) override;
  void Visit(const ir::Min* op) override;
  void Visit(const ir::Max* op) override;
  void Visit(const ir::Call* op) override;

  void PrintIncludes() override;

  void PrintTempBufferCreation(const ir::Buffer& buffer);

  /**
   * Print the function declaration, this is different from C, we expand the
   * arguments and get something like
   * `__global__ void myadd(float* __restrict__ A, float* __restrict__ B, int
   * n);`
   */
  void PrintFunctionDeclaration(const ir::_LoweredFunc_* op);
  void PrintFunctionBody(const ir::_LoweredFunc_* op);

 private:
  // generate unique kernel name, which is namespace + op->name.
  std::string GenerateKernelName(const ir::_LoweredFunc_* op);
  bool for_syclrtc_{false};
  static const std::string source_header_;
};

}  // namespace sycl
}  // namespace backends
}  // namespace cinn
