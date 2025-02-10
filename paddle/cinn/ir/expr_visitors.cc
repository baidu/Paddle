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
#include "paddle/cinn/ir/expr_visitors.h"

namespace cinn {
namespace ir {

void VisitExpr(const stmt::Let &stmt,
               const std::function<void(const Expr &)> &callback) {
  const auto &symbol = stmt->symbol();
  const auto &body = stmt->body();
  callback(symbol);
  if (body.defined()) {
    callback(body);
  }
}

void VisitExpr(const stmt::Store &stmt,
               const std::function<void(const Expr &)> &callback) {
  const auto &value = stmt->value();
  const auto &tensor = stmt->tensor();
  const auto &indices = stmt->indices();
  callback(value);
  callback(tensor);
  for (const auto &indice : indices) {
    callback(indice);
  }
}

void VisitExpr(const stmt::Alloc &stmt,
               const std::function<void(const Expr &)> &callback) {
  const auto &destination = stmt->destination();
  const auto &extents = stmt->extents();
  const auto &condition = stmt->condition();
  const auto &body = stmt->body();
  callback(destination);
  for (const auto &extent : extents) {
    callback(extent);
  }
  if (condition.defined()) {
    callback(condition);
  }
  if (body.defined()) {
    callback(body);
  }
}

void VisitExpr(const stmt::Free &stmt,
               const std::function<void(const Expr &)> &callback) {
  const auto &destination = stmt->destination();
  callback(destination);
}

void VisitExpr(const stmt::IfThenElse &stmt,
               const std::function<void(const Expr &)> &callback) {
  const auto &condition = stmt->condition();
  callback(condition);
}

void VisitExpr(const stmt::For &stmt,
               const std::function<void(const Expr &)> &callback) {
  const auto &min = stmt->min();
  const auto &extent = stmt->extent();
  callback(min);
  callback(extent);
}

void VisitExpr(const stmt::Schedule &stmt,
               const std::function<void(const Expr &)> &callback) {
  const auto &iter_vars = stmt->iter_vars();
  const auto &iter_values = stmt->iter_values();
  const auto &read_buffers = stmt->read_buffers();
  const auto &write_buffers = stmt->write_buffers();

  for (const auto &iter_var : iter_vars) {
    if (iter_var->lower_bound.defined()) {
      callback(iter_var->lower_bound);
    }
    if (iter_var->upper_bound.defined()) {
      callback(iter_var->upper_bound);
    }
  }
  for (const auto &iter_value : iter_values) {
    callback(iter_value);
  }
  for (const auto &read_buffer : read_buffers) {
    callback(read_buffer);
  }
  for (const auto &write_buffer : write_buffers) {
    callback(write_buffer);
  }
}

void VisitExpr(const stmt::Evaluate &stmt,
               const std::function<void(const Expr &)> &callback) {
  const auto &value = stmt->value();
  callback(value);
}

void MutateExpr(stmt::Let stmt, const std::function<void(Expr *)> &callback) {
  ir::Expr symbol = stmt->symbol();
  ir::Expr body = stmt->body();
  callback(&symbol);
  if (body.defined()) {
    callback(&body);
  }
  stmt->set_symbol(symbol);
  stmt->set_body(body);
}

void MutateExpr(stmt::Store stmt, const std::function<void(Expr *)> &callback) {
  ir::Expr value = stmt->value();
  ir::Expr tensor = stmt->tensor();
  std::vector<ir::Expr> indices = stmt->indices();
  callback(&value);
  callback(&tensor);
  for (ir::Expr &indice : indices) {
    callback(&indice);
  }
  stmt->set_value(value);
  stmt->set_tensor(tensor);
  stmt->set_indices(indices);
}

void MutateExpr(stmt::Alloc stmt, const std::function<void(Expr *)> &callback) {
  ir::Expr destination = stmt->destination();
  std::vector<ir::Expr> extents = stmt->extents();
  ir::Expr condition = stmt->condition();
  ir::Expr body = stmt->body();
  callback(&destination);
  for (ir::Expr &extent : extents) {
    callback(&extent);
  }
  if (condition.defined()) {
    callback(&condition);
  }
  if (body.defined()) {
    callback(&body);
  }
  stmt->set_destination(destination);
  stmt->set_extents(extents);
  stmt->set_condition(condition);
  stmt->set_body(body);
}

void MutateExpr(stmt::Free stmt, const std::function<void(Expr *)> &callback) {
  ir::Expr destination = stmt->destination();
  callback(&destination);
  stmt->set_destination(destination);
}

void MutateExpr(stmt::IfThenElse stmt,
                const std::function<void(Expr *)> &callback) {
  ir::Expr condition = stmt->condition();
  callback(&condition);
  stmt->set_condition(condition);
}

void MutateExpr(stmt::For stmt, const std::function<void(Expr *)> &callback) {
  ir::Expr min = stmt->min();
  ir::Expr extent = stmt->extent();
  callback(&min);
  callback(&extent);
  stmt->set_min(min);
  stmt->set_extent(extent);
}

void MutateExpr(stmt::Schedule stmt,
                const std::function<void(Expr *)> &callback) {
  std::vector<ir::Var> iter_vars = stmt->iter_vars();
  std::vector<ir::Expr> iter_values = stmt->iter_values();
  std::vector<ir::Expr> read_buffers = stmt->read_buffers();
  std::vector<ir::Expr> write_buffers = stmt->write_buffers();

  for (ir::Var iter_var : iter_vars) {
    if (iter_var->lower_bound.defined()) {
      callback(&(iter_var->lower_bound));
    }
    if (iter_var->upper_bound.defined()) {
      callback(&(iter_var->upper_bound));
    }
  }
  for (ir::Expr &iter_value : iter_values) {
    callback(&iter_value);
  }
  for (ir::Expr &read_buffer : read_buffers) {
    callback(&read_buffer);
  }
  for (ir::Expr &write_buffer : write_buffers) {
    callback(&write_buffer);
  }

  stmt->set_iter_vars(iter_vars);
  stmt->set_iter_values(iter_values);
  stmt->set_read_buffers(read_buffers);
  stmt->set_write_buffers(write_buffers);
}

void MutateExpr(stmt::Evaluate stmt,
                const std::function<void(Expr *)> &callback) {
  ir::Expr value = stmt->value();
  callback(&value);
  stmt->set_value(value);
}

}  // namespace ir
}  // namespace cinn
