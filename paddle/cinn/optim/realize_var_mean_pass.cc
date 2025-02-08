// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/cinn/optim/realize_var_mean_pass.h"
#include "paddle/cinn/ir/ir_mutator.h"
#include "paddle/cinn/ir/utils/ir_copy.h"
#include "paddle/cinn/lang/compute.h"
#include "paddle/phi/core/enforce.h"

namespace cinn {
namespace optim {

using ir::stmt::Alloc;
using ir::stmt::BlockRef;
using ir::stmt::Evaluate;
using ir::stmt::For;
using ir::stmt::Free;
using ir::stmt::IfThenElse;
using ir::stmt::Let;
using ir::stmt::Schedule;
using ir::stmt::StmtRef;
using ir::stmt::Store;

namespace {

std::set<ir::Buffer> CollectReduceVarBuffers(const BlockRef& body) {
  std::set<ir::Buffer> buffers;

  const auto VisitFn = [&](const StmtRef& stmt) {
    if (!stmt.isa<Store>()) return;
    Store store_stmt = stmt.as<Store>();
    auto* call_node = store_stmt->value().As<ir::Call>();
    if (call_node && call_node->name == "cinn_reduce_variance") {
      buffers.insert(store_stmt->tensor().as_tensor()->buffer);
    }
  };

  ir::stmt::Visit(body, VisitFn, [](auto) {});
  return buffers;
}

// Get the corresponding Welford type of this element type.
Type GetWelfordType(const Type& elem_type) {
  Type welford_type(
      ir::Type::type_t::Customized, /* bits = */ 128, /* width = */ 1);
  welford_type.set_customized_type("welford_fp32");
  welford_type.set_cpp_const(false);
  return welford_type;
}

struct LoadTypeMutator : public ir::IRMutator<> {
  explicit LoadTypeMutator(const std::set<ir::Buffer>& buffers)
      : buffers_(buffers) {}

  void operator()(ir::Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

 private:
  void Visit(const ir::Load* op, ir::Expr* expr) override {
    auto* node = expr->As<ir::Load>();
    auto& buffer = node->tensor.as_tensor()->buffer;
    if (buffers_.count(buffer) > 0) {
      ir::Type new_type = GetWelfordType(buffer->dtype);
      node->tensor.as_tensor()->set_type(new_type);
      buffer->dtype = new_type;
    }
  }

  const std::set<ir::Buffer>& buffers_;
};

void SetWelfordBufferType(const BlockRef& body,
                          const std::set<ir::Buffer>& buffers) {
  const auto VisitFn = [&](StmtRef stmt) {
    if (!stmt.isa<Store>()) return;
    Store store_stmt = stmt.as<Store>();
    auto& buffer = store_stmt->tensor().as_tensor()->buffer;

    // Set store buffer type
    if (buffers.count(buffer) > 0) {
      ir::Expr new_tensor = ir::ir_utils::IRCopy(store_stmt->tensor());
      ir::Type new_type = GetWelfordType(buffer->dtype);
      new_tensor.as_tensor()->set_type(new_type);
      new_tensor.as_tensor()->buffer->dtype = new_type;
      store_stmt->set_tensor(new_tensor);
    }

    // Set load buffer type
    ir::Expr new_value = ir::ir_utils::IRCopy(store_stmt->value());
    LoadTypeMutator load_type_mutator(buffers);
    load_type_mutator(&new_value);
    store_stmt->set_value(new_value);
  };

  ir::stmt::Mutate(body, VisitFn, [](auto) {});
}

struct WelfordExternCallMutator : public ir::IRMutator<> {
  void operator()(ir::Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

 private:
  void Visit(const ir::Call* op, ir::Expr* expr) override {
    ir::IRMutator<>::Visit(op, expr);
    if (op->name != "cinn_reduce_variance") return;

    if (op->read_args[0].type() == op->read_args[1].type()) {
      *expr = ir::Add::Make(op->read_args[0], op->read_args[1]);
    } else {
      *expr = lang::CallExtern("cinn_welford_add_fp32", op->read_args);
    }
  }
};

void ReplaceWelfordExternCall(const BlockRef& body) {
  const auto VisitFn = [&](StmtRef stmt) {
    if (!stmt.isa<Store>()) return;
    Store store_stmt = stmt.as<Store>();
    ir::Expr new_value = ir::ir_utils::IRCopy(store_stmt->value());
    WelfordExternCallMutator()(&new_value);
    store_stmt->set_value(new_value);
  };

  ir::stmt::Mutate(body, VisitFn, [](auto) {});
}

}  // namespace

LogicalResult RealizeVarMeanPass::Run(ir::LoweredFunc func) {
  BlockRef body = func->body_block;

  std::set<ir::Buffer> buffers = CollectReduceVarBuffers(body);

  for (auto& buffer : func->temp_bufs) {
    if (buffers.count(buffer) > 0) {
      buffer->dtype = GetWelfordType(buffer->dtype);
    }
  }

  SetWelfordBufferType(body, buffers);

  ReplaceWelfordExternCall(body);

  return LogicalResult::success();
}

std::unique_ptr<FuncPass> CreateRealizeVarMeanPass() {
  return std::make_unique<RealizeVarMeanPass>();
}

}  // namespace optim
}  // namespace cinn
