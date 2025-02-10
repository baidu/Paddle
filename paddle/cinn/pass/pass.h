// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/lowered_func.h"
#include "paddle/cinn/ir/stmt.h"

namespace cinn {
namespace optim {

enum class PassKind { PK_FUNC, PK_BLOCK, PK_STMT, PK_EXPR };

class LogicalResult {
 public:
  static LogicalResult success() { return LogicalResult(true); }
  static LogicalResult failure() { return LogicalResult(false); }
  bool succeeded() const { return success_; }
  bool failed() const { return !success_; }

 private:
  explicit LogicalResult(bool success) : success_(success) {}
  bool success_;
};

template <typename IRScopeRefT>
class Pass {
 public:
  explicit Pass(PassKind kind, const std::string& name)
      : kind_(kind), name_(name) {}
  virtual ~Pass() {}

  virtual LogicalResult Run(IRScopeRefT scope) = 0;

  PassKind kind() const { return kind_; }
  const std::string& name() const { return name_; }

 private:
  PassKind kind_;
  std::string name_;
};

class FuncPass : public Pass<ir::LoweredFunc> {
 public:
  explicit FuncPass(const std::string& name) : Pass(PassKind::PK_FUNC, name) {}

  // Run on the whole function.
  virtual LogicalResult Run(ir::LoweredFunc f) = 0;
  // Only run on function body.
  virtual LogicalResult Run(ir::stmt::BlockRef block) {
    LOG(WARNING) << name()
                 << "should run on the whole function, not on the body block.";
    return LogicalResult::failure();
  }
};

class BlockPass : public Pass<ir::stmt::BlockRef> {
 public:
  explicit BlockPass(const std::string& name)
      : Pass(PassKind::PK_BLOCK, name) {}
  virtual LogicalResult Run(ir::stmt::BlockRef block) = 0;
};

class StmtPass : public Pass<ir::stmt::StmtRef> {
 public:
  explicit StmtPass(const std::string& name) : Pass(PassKind::PK_STMT, name) {}
  virtual LogicalResult Run(ir::stmt::StmtRef stmt) = 0;
};

class ExprPass : public Pass<ir::Expr*> {
 public:
  explicit ExprPass(const std::string& name) : Pass(PassKind::PK_STMT, name) {}
  virtual LogicalResult Run(ir::Expr* expr) = 0;
};

}  // namespace optim
}  // namespace cinn
