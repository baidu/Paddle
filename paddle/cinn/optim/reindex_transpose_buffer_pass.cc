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

#include "paddle/cinn/optim/reindex_transpose_buffer_pass.h"
#include "paddle/cinn/ir/op/ir_operators.h"
#include "paddle/cinn/ir/utils/ir_copy.h"
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

std::set<ir::Buffer> CollectTransposeBuffers(const BlockRef& body) {
  std::set<ir::Buffer> buffers;

  const auto VisitFn = [&](const StmtRef& stmt) {
    if (!stmt.isa<Schedule>()) return;
    Schedule schedule = stmt.as<Schedule>();
    auto attr_it = schedule->attrs().find("transpose_stage");
    if (attr_it == schedule->attrs().end()) return;

    StmtRef store = schedule->body()->stmts().front();
    PADDLE_ENFORCE(
        store.isa<Store>(),
        ::common::errors::PreconditionNotMet(
            "The Schedule of transpose buffer must have a pure Store."));

    Store store_stmt = store.as<Store>();
    PADDLE_ENFORCE_NOT_NULL(
        store_stmt->value().As<ir::Load>(),
        ::common::errors::PreconditionNotMet(
            "The store value of transpose buffer must be a pure Load."));

    if (attr_it->second == ir::attr_t(std::string("write"))) {
      ir::Buffer buffer = store_stmt->tensor().as_tensor()->buffer;
      buffers.insert(buffer);
    }
  };

  ir::stmt::Visit(body, VisitFn, [](auto) {});
  return buffers;
}

void ReplaceTransposeBuffersWithUnionBuffer(
    ir::LoweredFunc func,
    const std::set<ir::Buffer>& old_buffers,
    ir::Buffer new_buffer) {
  std::vector<ir::Buffer> new_temp_bufs;
  for (auto& buffer : func->temp_bufs) {
    if (old_buffers.count(buffer) > 0) continue;
    new_temp_bufs.push_back(buffer);
  }
  new_temp_bufs.push_back(new_buffer);
  func->temp_bufs = std::move(new_temp_bufs);
}

struct TransposeBufferIndicesMutator : public ir::stmt::StmtMutator<> {
  explicit TransposeBufferIndicesMutator(ir::Buffer union_buffer)
      : union_buffer_(union_buffer) {}

  void operator()(BlockRef block) { VisitBlock(block); }

 private:
  void VisitStmt(Schedule stmt) override {
    Schedule schedule = stmt.as<Schedule>();
    auto attr_it = stmt->attrs().find("transpose_stage");
    if (attr_it == stmt->attrs().end()) {
      ir::stmt::StmtMutator<>::VisitBlock(stmt->body());
      return;
    }

    StmtRef store = stmt->body()->stmts().front();
    Store store_stmt = store.as<Store>();

    // Note: we currently use constant tiling configs for transpose, i.e.
    // inner_loop = 4, blockDim.y = 8, blockDim.x = 32. Therefore, the shape
    // and indices of the transpose buffer are also fixed.
    std::vector<ir::Expr> shape = {ir::Expr(32), ir::Expr(32)};

    if (attr_it->second == ir::attr_t(std::string("write"))) {
      // at buffer write stage, re-index the store buffer
      store_stmt->set_indices({GetIndexY(), GetIndexX() ^ GetIndexY()});
      ir::Expr new_tensor = ir::ir_utils::IRCopy(store_stmt->tensor());
      new_tensor.as_tensor()->shape = shape;
      new_tensor.as_tensor()->buffer = union_buffer_;
      store_stmt->set_tensor(new_tensor);
    } else {
      // at buffer read stage, re-index the load buffer
      ir::Expr new_value = ir::ir_utils::IRCopy(store_stmt->value());
      auto* load = new_value.As<ir::Load>();
      load->indices = {GetIndexX(), GetIndexY() ^ GetIndexX()};
      load->tensor.as_tensor()->shape = shape;
      load->tensor.as_tensor()->buffer = union_buffer_;
      store_stmt->set_value(new_value);
    }
  }

  ir::Expr GetIndexX() { return ir::Var("threadIdx.x"); }

  ir::Expr GetIndexY() {
    return inner_loop_var_ * ir::Expr(8) + ir::Var("threadIdx.y");
  }

  void VisitStmt(For stmt) override {
    if (stmt->is_serial()) {
      inner_loop_var_ = stmt->loop_var();
    }
    VisitBlock(stmt->body());
  }

  void VisitStmt(IfThenElse stmt) override {
    ir::stmt::BlockRef true_case = stmt->true_case();
    VisitBlock(true_case);
    stmt->set_true_case(true_case);
    if (stmt->false_case().defined()) {
      ir::stmt::BlockRef false_case = stmt->false_case();
      VisitBlock(false_case);
      stmt->set_false_case(false_case);
    }
  }

  void VisitStmt(Let stmt) override { return; }
  void VisitStmt(Store stmt) override { return; }
  void VisitStmt(Alloc stmt) override { return; }
  void VisitStmt(Free stmt) override { return; }
  void VisitStmt(Evaluate stmt) override { return; }

 private:
  ir::Buffer union_buffer_;
  ir::Var inner_loop_var_;
};

}  // namespace

LogicalResult ReindexTransposeBufferPass::Run(ir::LoweredFunc func) {
  BlockRef body = func->body_block;

  // Step 1. Collect all transpose buffers in the function, also verify that the
  // transpose buffers are used properly.
  std::set<ir::Buffer> transpose_buffers = CollectTransposeBuffers(body);
  if (transpose_buffers.empty()) {
    return LogicalResult::success();
  }

  // Step 2. Create a union buffer to replace all transpose buffers.
  // The union buffer's size is the size of the largest data type among the
  // transpose buffers multiplied by 1024 (the block size).
  int max_dtype_bytes = 0;
  for (auto& buffer : transpose_buffers) {
    max_dtype_bytes = std::max(max_dtype_bytes, buffer->dtype.bytes());
  }

  ir::Buffer union_buffer = ir::_Buffer_::Make(
      "transpose_union_shm", {ir::Expr(max_dtype_bytes * 1024)});
  union_buffer->dtype = common::UInt(8);
  union_buffer->memory_type = ir::MemoryType::GPUShared;

  ReplaceTransposeBuffersWithUnionBuffer(func, transpose_buffers, union_buffer);

  // Step 3. Swizzle the load & store indices of transpose buffers.
  TransposeBufferIndicesMutator mutator(union_buffer);
  mutator(body);

  return LogicalResult::success();
}

std::unique_ptr<FuncPass> CreateReindexTransposeBufferPass() {
  return std::make_unique<ReindexTransposeBufferPass>();
}

}  // namespace optim
}  // namespace cinn
