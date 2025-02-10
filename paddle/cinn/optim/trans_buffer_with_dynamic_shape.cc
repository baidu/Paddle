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

#include "paddle/cinn/optim/trans_buffer_with_dynamic_shape.h"

#include <numeric>
#include <unordered_set>

#include "paddle/cinn/common/cas.h"
#include "paddle/cinn/common/dev_info_manager.h"
#include "paddle/cinn/common/integer_set.h"
#include "paddle/cinn/common/ir_util.h"
#include "paddle/cinn/ir/ir_mutator.h"
#include "paddle/cinn/ir/op/ir_operators.h"
#include "paddle/cinn/ir/utils/ir_compare.h"
#include "paddle/cinn/ir/utils/ir_copy.h"
#include "paddle/cinn/runtime/backend_api.h"
#include "paddle/cinn/utils/string.h"

namespace cinn::optim {

namespace {

common::cas_intervals_t var_intervals = {};
cinn::common::SymbolicExprAnalyzer analyzer(var_intervals);

struct Mutator : public ir::IRMutator<>, public ir::stmt::StmtMutator<> {
  using ir::IRMutator<>::Visit;

  Mutator() : shared_mem_size_used_(0) {}

  void operator()(ir::stmt::BlockRef block) { VisitBlock(block); }

  size_t shared_mem_size_used() const { return shared_mem_size_used_; }

 private:
  void Visit(const ir::_Tensor_* tensor, Expr* expr) override {
    if (!tensor->buffer.defined()) return;
    auto buf = tensor->buffer.As<ir::_Buffer_>();
    if (!visited_buf_.count(buf->name)) {
      visited_buf_.insert(buf->name);
      auto buf_size = ir::Expr(1);

      size_t max_dim = std::max(buf->shape.size(), tensor->shape.size());
      size_t min_dim = std::min(buf->shape.size(), tensor->shape.size());
      size_t i = 0;
      for (; i < min_dim; ++i) {
        Expr e = expr->as_tensor()->shape[i];
        Expr buf_e = buf->shape[i];
        if (buf->memory_type == ir::MemoryType::GPULocal) {
          e = cinn::common::AutoSimplify(e);
          buf_e = cinn::common::AutoSimplify(buf_e);
          if (!e.is_constant()) {
            auto new_shape = ir::ir_utils::IRCopy(e);
            new_shape = analyzer.UpperBound(new_shape);
            PADDLE_ENFORCE_EQ(
                new_shape.is_constant(),
                true,
                ::common::errors::InvalidArgument("new_shape is not constant"));
            e = new_shape;
          }
          if (!buf_e.is_constant()) {
            auto new_shape = ir::ir_utils::IRCopy(buf_e);
            new_shape = analyzer.UpperBound(new_shape);
            PADDLE_ENFORCE_EQ(
                new_shape.is_constant(),
                true,
                ::common::errors::InvalidArgument("new_shape is not constant"));
            buf_e = new_shape;
          }
        }
        buf_size = buf_size * buf_e;
      }
      for (; i < max_dim; i++) {
        auto e = buf->shape.size() > tensor->shape.size() ? buf->shape[i]
                                                          : tensor->shape[i];
        if (buf->memory_type == ir::MemoryType::GPULocal) {
          e = cinn::common::AutoSimplify(e);
          if (!e.is_constant()) {
            auto new_shape = ir::ir_utils::IRCopy(e);
            new_shape = analyzer.UpperBound(new_shape);
            PADDLE_ENFORCE_EQ(
                new_shape.is_constant(),
                true,
                ::common::errors::InvalidArgument("new_shape is not constant"));
            e = new_shape;
          }
        }
        buf_size = buf_size *
                   (buf->shape.size() > tensor->shape.size() ? e : ir::Expr(1));
      }
      if (buf->memory_type == ir::MemoryType::GPUShared) {
        buf_size = analyzer.UpperBound(buf_size);
        PADDLE_ENFORCE_EQ(
            buf_size.is_constant(),
            true,
            ::common::errors::InvalidArgument("buf_size is not constant"));
        shared_mem_size_used_ += static_cast<size_t>(buf_size.get_constant()) *
                                 static_cast<size_t>(buf->dtype.bits()) / 8;
      }
      for (auto& e : expr->as_tensor()->shape) {
        Visit(&e, &e);
      }
    }
  }

  void VisitStmt(ir::stmt::Let stmt) override {
    Expr body = stmt->body();
    Visit(&body, &body);
  }

  void VisitStmt(ir::stmt::Store stmt) override {
    Expr tensor = stmt->tensor();
    Visit(&tensor, &tensor);
  }

  void VisitStmt(ir::stmt::For stmt) override {
    Expr min = stmt->min();
    Expr extent = stmt->extent();
    Visit(&min, &min);
    Visit(&extent, &extent);
    VisitBlock(stmt->body());
  }

  void VisitStmt(ir::stmt::IfThenElse stmt) override {
    Expr condition = stmt->condition();
    Visit(&condition, &condition);
    VisitBlock(stmt->true_case());
    if (stmt->false_case().defined()) {
      VisitBlock(stmt->false_case());
    }
  }

  void VisitStmt(ir::stmt::Schedule stmt) override {
    for (Expr read_buffer : stmt->read_buffers()) {
      Visit(&read_buffer, &read_buffer);
    }
    for (Expr write_buffer : stmt->write_buffers()) {
      Visit(&write_buffer, &write_buffer);
    }
    VisitBlock(stmt->body());
  }

  void VisitStmt(ir::stmt::Evaluate stmt) override {
    Expr value = stmt->value();
    Visit(&value, &value);
  }

  void VisitStmt(ir::stmt::Alloc stmt) override { return; }

  void VisitStmt(ir::stmt::Free stmt) override { return; }

  size_t shared_mem_size_used_;
  std::unordered_set<std::string> visited_buf_;
};
}  // namespace

LogicalResult TransBufferWithDynamicShapePass::Run(ir::LoweredFunc func) {
  Mutator mutator;
  mutator(func->body_block);
  cinn::common::DefaultDeviceTarget().arch.Match(
      [&](std::variant<common::UnknownArch, common::X86Arch, common::ARMArch>) {
      },
      [&](common::NVGPUArch) {
#ifdef CINN_WITH_CUDA
        auto cur_dev_info =
            common::DevInfoMgr<common::NVGPUArch>::GetDevInfo(0);
        if (cur_dev_info->IsValid()) {
          size_t max_shm_per_block = cur_dev_info->GetMaxSharedMemPerBlock();
          PADDLE_ENFORCE_EQ(
              (mutator.shared_mem_size_used() <= max_shm_per_block),
              true,
              ::common::errors::InvalidArgument(
                  "The shared memory size used by current kernel is greater "
                  "than the max shared memory per block"));
        }
#endif
      },
      [&](common::HygonDCUArchHIP) {
        using cinn::runtime::BackendAPI;
        size_t max_shm_per_block =
            BackendAPI::get_backend(common::HygonDCUArchHIP{})
                ->get_device_property(
                    BackendAPI::DeviceProperty::MaxSharedMemoryPerBlock);
        PADDLE_ENFORCE_LE(
            mutator.shared_mem_size_used(),
            max_shm_per_block,
            ::common::errors::InvalidArgument(
                "The shared memory size used by current kernel is greater "
                "than the max shared memory per block"));
      },
      [&](common::HygonDCUArchSYCL) { CINN_NOT_IMPLEMENTED });
  return LogicalResult::success();
}

std::unique_ptr<FuncPass> CreateTransBufferWithDynamicShapePass() {
  return std::make_unique<TransBufferWithDynamicShapePass>();
}
}  // namespace cinn::optim
