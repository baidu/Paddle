// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/optim/update_buffer_axis_pass.h"

#include <unordered_map>

#include "paddle/cinn/common/cas.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_mutator.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/stmt_visitors.h"
#include "paddle/cinn/ir/utils/ir_copy.h"
#include "paddle/cinn/ir/utils/ir_replace.h"
#include "paddle/cinn/optim/replace_var_with_expr.h"
#include "paddle/cinn/utils/string.h"

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
using ir::stmt::Store;

bool ExprMathEqual(const Expr& expr1, const Expr& expr2) {
  ir::Expr cmp_expr = common::AutoSimplify(ir::Sub::Make(expr1, expr2));
  // This is ugly code since AutoSimplify is not powerful enough. Modify it
  // after we make auto simplify better
  ir::Expr simplified = common::AutoSimplify(cmp_expr);
  int count = 0;
  while (simplified != cmp_expr) {
    cmp_expr = simplified;
    simplified = common::AutoSimplify(cmp_expr);
    ++count;
    // Control dead loop
    if (count >= 5) {
      break;
    }
  }
  return simplified.is_constant() && simplified.get_constant() == 0;
}

void FormalizeSingleIndex(const ir::Tensor& tensor,
                          std::vector<ir::Expr>* indices) {
  if (tensor->shape.size() > 1 && indices->size() == 1) {
    ir::Expr origin_index_expr = (*indices)[0];
    ir::Expr mul = Expr(1);
    (*indices)[0] = ir::Mod::Make(origin_index_expr, tensor->shape.back());
    for (int i = static_cast<int>(tensor->shape.size()) - 2; i >= 0; --i) {
      mul = ir::Mul::Make(tensor->shape[i + 1], mul);
      ir::Expr div_expr = ir::Div::Make(origin_index_expr, mul);
      ir::Expr index_expr = ir::Mod::Make(div_expr, tensor->shape[i]);
      indices->insert(indices->begin(), common::AutoSimplify(index_expr));
    }
  }
}

class AnalyzeBufferAxis : public ir::IRMutator<>,
                          public ir::stmt::StmtMutator<> {
 public:
  void operator()(ir::Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

  void operator()(BlockRef block) {
    ir::stmt::StmtMutator<>::VisitBlock(block);
  }

 private:
  void VisitStmt(For stmt) override {
    if (stmt->is_gpu_block_binded()) {
      var_bind_threads.insert(stmt->loop_var()->name);
      VisitBlock(stmt->body());
      var_bind_threads.erase(stmt->loop_var()->name);
      return;
    }
    VisitBlock(stmt->body());
  }

  // Analyze the buffer access inside store
  void VisitStmt(Store stmt) override {
    const ir::Tensor& tensor = stmt->tensor().as_tensor_ref();

    if (!tensor->buffer.defined() ||
        tensor->buffer->memory_type == ir::MemoryType::Heap) {
      ir::Expr value = stmt->value();
      ir::IRMutator<>::Visit(&value, &value);
      stmt->set_value(value);
      return;
    }

    std::vector<ir::Expr> indices = stmt->indices();
    FormalizeSingleIndex(tensor, &indices);
    stmt->set_indices(indices);
    AnalyzeTensorAxis(indices, tensor);
    ir::Expr value = stmt->value();
    ir::IRMutator<>::Visit(&value, &value);
    stmt->set_value(value);
  }

  void VisitStmt(Schedule stmt) override {
    const std::vector<ir::Var>& iter_vars = stmt->iter_vars();
    const std::vector<ir::Expr>& iter_values = stmt->iter_values();
    for (int i = 0; i < iter_vars.size(); ++i) {
      iter_var_to_bind_expr_[iter_vars[i]->name] = iter_values[i];
    }
    VisitBlock(stmt->body());
  }

  void VisitStmt(IfThenElse stmt) override {
    VisitBlock(stmt->true_case());
    if (stmt->false_case().defined()) {
      VisitBlock(stmt->false_case());
    }
  }

  void VisitStmt(Let stmt) override {
    ir::Expr expr = stmt->body();
    ir::IRMutator<>::Visit(&expr, &expr);
    stmt->set_body(expr);
  }

  void VisitStmt(Alloc) override {}

  void VisitStmt(Evaluate) override {}

  void VisitStmt(Free) override {}

  // Analyze the buffer access inside load
  void Visit(const ir::Load* op, Expr* expr) override {
    ir::Load* load = expr->As<ir::Load>();
    ir::Tensor tensor = load->tensor.as_tensor_ref();
    if (!tensor->buffer.defined() ||
        tensor->buffer->memory_type == ir::MemoryType::Heap) {
      ir::IRMutator<>::Visit(op, expr);
      return;
    }
    FormalizeSingleIndex(tensor, &(load->indices));
    AnalyzeTensorAxis(load->indices, tensor);
    ir::IRMutator<>::Visit(op, expr);
  }

  void AnalyzeTensorAxis(const std::vector<Expr>& indices,
                         const ir::Tensor& tensor) {
    if (!tensor->buffer.defined() ||
        tensor->buffer->memory_type == ir::MemoryType::Heap) {
      return;
    }

    const std::string& buffer_name = tensor->buffer->name;
    if (!buffer_name_access_same_index_expr.count(buffer_name)) {
      for (int i = 0; i < indices.size(); ++i) {
        if (tensor->buffer->memory_type == ir::MemoryType::GPUShared) {
          // In GPUShared case, the thread vars cannot be simplified
          std::vector<ir::Expr> var_nodes =
              ir::ir_utils::CollectIRNodesWithoutTensor(
                  indices[i], [&](const Expr* x) {
                    const ir::_Var_* var = x->As<ir::_Var_>();
                    return var != nullptr && var_bind_threads.count(var->name);
                  });
          if (var_nodes.empty()) {
            buffer_name_access_same_index_expr[buffer_name][i] =
                GetIndexBindExpr(indices[i]);
          }

        } else {
          buffer_name_access_same_index_expr[buffer_name][i] =
              GetIndexBindExpr(indices[i]);
        }
      }
      return;
    }

    std::map<int, ir::Expr>& index_expr =
        buffer_name_access_same_index_expr[buffer_name];
    for (int i = 0; i < indices.size(); ++i) {
      if (index_expr.count(i)) {
        if (!ExprMathEqual(index_expr[i], GetIndexBindExpr(indices[i]))) {
          index_expr.erase(i);
        }
      }
    }
    if (index_expr.empty()) {
      buffer_name_access_same_index_expr.erase(buffer_name);
    }
  }

  ir::Expr GetIndexBindExpr(ir::Expr index) {
    if (index.as_var() && iter_var_to_bind_expr_.count(index.as_var()->name)) {
      return iter_var_to_bind_expr_[index.as_var()->name];
    }
    return index;
  }

 public:
  // Stores the buffer names, and its indice where always using same Expr to
  // access For example:
  //   _A[i * 3][j] = ...
  //   ... = _A[k][j]
  // The buffer name _A will map to {1 : j}, where 1 is the indice
  // having same expr j.
  std::unordered_map<std::string, std::map<int, ir::Expr>>
      buffer_name_access_same_index_expr;

 private:
  std::unordered_map<std::string, ir::Expr> iter_var_to_bind_expr_;
  std::unordered_set<std::string> var_bind_threads;
};

class ReplaceSameAxisToZero : public ir::IRMutator<>,
                              public ir::stmt::StmtMutator<> {
 public:
  ReplaceSameAxisToZero(
      const std::unordered_map<std::string, std::map<int, ir::Expr>>&
          buffer_name_access_same_index_expr)
      : buffer_name_access_same_index_expr_(
            buffer_name_access_same_index_expr) {}

  void operator()(ir::Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

  void operator()(BlockRef block) {
    ir::stmt::StmtMutator<>::VisitBlock(block);
  }

 private:
  // Analyze the buffer access inside store
  void VisitStmt(Store stmt) override {
    ir::Tensor tensor = stmt->tensor().as_tensor_ref();
    std::vector<Expr> expr = stmt->indices();
    ReplaceIndices(tensor, &expr);
    stmt->set_indices(expr);
  }

  void VisitStmt(IfThenElse stmt) override {
    VisitBlock(stmt->true_case());
    if (stmt->false_case().defined()) {
      VisitBlock(stmt->false_case());
    }
  }

  void VisitStmt(Let stmt) override {
    ir::Expr expr = stmt->body();
    ir::IRMutator<>::Visit(&expr, &expr);
    stmt->set_body(expr);
  }

  void VisitStmt(For stmt) override { VisitBlock(stmt->body()); }

  void VisitStmt(Schedule stmt) override { VisitBlock(stmt->body()); }

  void VisitStmt(Alloc) override {}

  void VisitStmt(Evaluate) override {}

  void VisitStmt(Free) override {}

  // Analyze the buffer access inside load
  void Visit(const ir::Load* op, Expr* expr) override {
    ir::Load* load = expr->As<ir::Load>();
    ir::Tensor tensor = load->tensor.as_tensor_ref();
    ReplaceIndices(tensor, &(load->indices));
    ir::IRMutator<>::Visit(op, expr);
  }

  void ReplaceIndices(const ir::Tensor& tensor, std::vector<Expr>* indices) {
    if (!tensor->buffer.defined() ||
        tensor->buffer->memory_type == ir::MemoryType::Heap) {
      return;
    }
    const std::string& buffer_name = tensor->buffer->name;
    if (buffer_name_access_same_index_expr_.count(buffer_name)) {
      for (const auto& p :
           buffer_name_access_same_index_expr_.at(buffer_name)) {
        int r = p.first;
        // After optimization, some load indice may be removed, so we need this
        // condition
        if (indices->size() > r) {
          ir::ir_utils::IrReplace(
              &(indices->at(r)), indices->at(r), ir::Expr(0));
        }
      }
      return;
    }
  }

  const std::unordered_map<std::string, std::map<int, ir::Expr>>&
      buffer_name_access_same_index_expr_;
};

void UpdateBufferAxis(BlockRef block) {
  VLOG(6) << "Before UpdateBufferAxisPass, Block = \n" << block;

  AnalyzeBufferAxis buffer_axis_analyzer;
  buffer_axis_analyzer(block);
  for (const auto& p :
       buffer_axis_analyzer.buffer_name_access_same_index_expr) {
    VLOG(6) << "Buffer name: " << p.first;
    for (const auto& q : p.second) {
      VLOG(6) << "Index: " << q.first << " Expr: " << q.second;
    }
  }

  ReplaceSameAxisToZero replacer(
      buffer_axis_analyzer.buffer_name_access_same_index_expr);
  replacer(block);
  VLOG(6) << "After UpdateBufferAxisPass, Block = \n" << block;
}

LogicalResult UpdateBufferAxisPass::Run(BlockRef block) {
  UpdateBufferAxis(block);
  return LogicalResult::success();
}

std::unique_ptr<BlockPass> CreateUpdateBufferAxisPass() {
  return std::make_unique<UpdateBufferAxisPass>();
}

}  // namespace optim
}  // namespace cinn
