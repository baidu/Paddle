// Copyright (c) 2024 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/backends/codegen_c.h"
#include "paddle/cinn/common/common.h"
#include "paddle/cinn/common/ir_util.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/lowered_func.h"
#include "paddle/cinn/ir/module.h"
#include "paddle/cinn/lang/packed_func.h"
#include "paddle/cinn/runtime/cinn_runtime.h"

namespace cinn::ir {
class Module;
}  // namespace cinn::ir

namespace cinn {
namespace backends {

/**
 * CUDA/HIP device code generator.
 *
 * It generates the device function, e.g, the function called "myadd" will have
 * a __global__ function called "myadd_kernel", different from codegen_c, the
 * declaration of the "myadd_kernel" function has an expanded argument list,
 * which finally similar to `__global__ void myadd(float* __restrict__ A, float*
 * __restrict__ B, int n);`
 */
class CodeGenGpuDev : public CodeGenC {
 public:
  explicit CodeGenGpuDev(Target target);

  /**
   * Compile the \p module to \p outputs.
   */
  void Compile(const ir::Module& module, const Outputs& outputs);

  //! Compile on RTC (RunTime Compilation).
  std::string Compile(const ir::Module& module, bool use_rtc = true);

  void Compile(const ir::LoweredFunc& func);

  /**
   * \brief Print a function argument in cuda/hip syntax. Currently, just some
   * decoration of __restrict__.
   * @param arg the argument.
   * @return the representation in cuda/hip syntax.
   *
   * We make it a static to make the test easier.
   */
  virtual void PrintFuncArg(const ir::Argument& arg);

  std::string Compile(const ir::Module& module, OutputKind output_kind);

  ir::Expr GetDynSharedMemOffset() const {
    if (MathEqual(dyn_shared_mem_offset_, Expr(-1))) {
      return Expr(0);
    }
    return dyn_shared_mem_offset_;
  }

 protected:
  void Visit(const ir::_Var_* op) override;
  void Visit(const ir::_LoweredFunc_* op) override;
  void Visit(const ir::Min* op) override;
  void Visit(const ir::Max* op) override;
  void Visit(const ir::Call* op) override;
  void Visit(const ir::Load* op) override;
  void VisitStmt(const ir::stmt::Free& stmt) override;
  void VisitStmt(const ir::stmt::Alloc& stmt) override;
  void VisitStmt(const ir::stmt::Store& op) override;
  void VisitStmt(const ir::stmt::Let& stmt) override;

  // Print element access at a cuda/hip built-in vector on a load/store node
  bool PrintBuiltinVectorAccess(const ir::LoadStoreAddrMnger* op,
                                ir::Expr index,
                                bool is_store);
  // Print element access at a cuda/hip built-in vector on a store node
  bool PrintBuiltinVectorAccess(const ir::stmt::Store& stmt,
                                ir::Expr index,
                                bool is_store);

  void PrintBuiltinCodes();

  virtual void PrintIncludes() = 0;

  virtual void PrintTempBufferCreation(const ir::Buffer& buffer);

  void PrintTempBufferAliasDefinition(const ir::Buffer& buffer);

  std::vector<ir::stmt::StmtRef> GenerateBufferAliasStmts(
      const ir::_LoweredFunc_* op, const std::vector<ir::Buffer>& temp_buffers);

  /**
   * Print the function declaration, this is different from C, we expand the
   * arguments and get something like
   * `__global__ void myadd(float* __restrict__ A, float* __restrict__ B, int
   * n);`
   */
  virtual void PrintFunctionDeclaration(const ir::_LoweredFunc_* op);

 private:
  Target target_;
  bool use_rtc_{false};
  // names of vectorized tensors from `Let` statements where dtypes of the
  // tensors are customized_type with customized_type::k_builtin_vector_t
  // prefix
  std::unordered_set<std::string> vectorized_tensor_names_;

  ir::Expr dyn_shared_mem_offset_{-1};
  std::vector<ir::Buffer> dynamic_alloc_buffers_;
};

ir::Expr CalculateSharedMemory(const ir::LoweredFunc& func);

}  // namespace backends
}  // namespace cinn
