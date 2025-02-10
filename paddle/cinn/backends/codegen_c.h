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
#include <vector>

#include "paddle/cinn/common/common.h"
#include "paddle/cinn/ir/intrinsic_ops.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/lowered_func.h"
#include "paddle/cinn/ir/module.h"
#include "paddle/cinn/lang/packed_func.h"
#include "paddle/cinn/runtime/cinn_runtime.h"
#include "paddle/common/flags.h"

namespace cinn {

namespace ir {
class Module;
}  // namespace ir

namespace backends {

//! keyword of __restrict__.
extern const char* kCKeywordRestrict;

class CodeGenC : public ir::IrPrinter {
 public:
  enum class OutputKind {
    CHeader,  //! output the C header file.
    CImpl,    //! output the C implementation file.
  };

  explicit CodeGenC(Target target);

  void Compile(const ir::Module& module, const Outputs& outputs);

  virtual std::string Compile(const ir::Module& module, OutputKind output_kind);

  //! Disable inline the builtin codes(too large) for simpler string comparison.
  void SetInlineBuiltinCodes(bool x = true) { inline_builtin_codes_ = x; }

 protected:
  void Compile(const ir::LoweredFunc& function);

  void GenerateHeaderFile(const ir::Module& module);

  std::string GetTypeName(Type type);

  std::string GetTypeRepr(Type type);
  //! type cast, print like "int(x)"
  // @{
  void PrintCastExpr(const Type& type, Expr e);
  void PrintCastExpr(const std::string& type, Expr e);
  // @}

  void PrintFunctionDeclaration(const ir::_LoweredFunc_* op) {
    str_ += "void ";
    str_ += op->name;
    str_ += "(";
    str_ += "void* _args, int32_t num_args";
    str_ += ")";
  }

  void PrintShape(const std::vector<Expr>& shape,
                  char leftb = '{',
                  char rightb = '}');

  virtual void PrintIncludes();
  void PrintBuiltinCodes();
  void PrintFileGuardOpen(const std::string& module_name);
  void PrintFileGuardClose(const std::string& module_name);

  //! Create the buffers in global scope(just creation without allocating them).
  void PrintBufferCreation(const std::vector<ir::Buffer>& buffers);
  void PrintBufferDestroy(const std::vector<ir::Buffer>& buffers);
  void PrintRuntimeType(const cinn_type_t& type);

  //! Print different kinds of Calls.
  // @{
  void PrintCallArgs(const ir::Call* call);
  void PrintCall_buffer_malloc(const ir::Call* op);
  void PrintCall_cinn_pod_value_to_(const ir::Call* op);
  void PrintCall_get_address(const ir::Call* op);
  void PrintCall_pod_values_to_array(const ir::Call* op);
  // @}

#define __DEFINE_VISIT(op__) void Visit(const ir::op__* op) override;
  NODETY_FORALL(__DEFINE_VISIT)
#undef __DEFINE_VISIT
  void Visit(const ir::_Module_* op) override;
  void Visit(const ir::_LoweredFunc_* op) override;

  void VisitBlock(const ir::stmt::BlockRef& block) override;
  void VisitStmt(const ir::stmt::Let& stmt) override;
  void VisitStmt(const ir::stmt::Store& stmt) override;
  void VisitStmt(const ir::stmt::Alloc& stmt) override;
  void VisitStmt(const ir::stmt::Free& stmt) override;
  void VisitStmt(const ir::stmt::IfThenElse& stmt) override;
  void VisitStmt(const ir::stmt::For& stmt) override;
  void VisitStmt(const ir::stmt::Schedule& stmt) override;

#define __DEFINE_VISIT(op__) \
  void Visit(const ir::intrinsics::op__* op) override;
  INTRINSIC_KIND_FOR_EACH(__DEFINE_VISIT)
#undef __DEFINE_VISIT

  void PrintFuncArg(const ir::Argument& arg);

  void PrintStackVecType(Type type, int lanes);

  friend class ExternFunctionEmitter;

  struct StoreHash {
    size_t operator()(const ir::stmt::Store& stmt) const {
      return std::hash<const Object*>()(stmt.get());
    }
  };

 protected:
  Target target_;
  std::stringstream ss_;
  bool inline_builtin_codes_{true};
  std::unordered_map<const ir::Store*, ir::Expr> store_to_offset_;
  std::unordered_map<const ir::stmt::Store, ir::Expr, StoreHash>
      store_stmt_to_offset_;
  std::unordered_map<const ir::Load*, ir::Expr> load_to_offset_;
};

namespace detail {

Expr StridedRampBase(Expr e, int stride);

}  // namespace detail

}  // namespace backends
}  // namespace cinn
