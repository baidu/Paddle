// Copyright (c) 2022 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/auto_schedule/analysis/analyze_ir.h"

#include <glog/logging.h>

#include <algorithm>
#include <string>
#include <unordered_set>

#include "paddle/cinn/ir/buffer.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/lowered_func.h"
#include "paddle/cinn/ir/schedule/ir_schedule.h"
#include "paddle/cinn/ir/tensor.h"
#include "paddle/cinn/ir/utils/ir_copy.h"
#include "paddle/cinn/ir/utils/ir_nodes_collector.h"
#include "paddle/cinn/lang/lower.h"
#include "paddle/cinn/optim/optimize.h"
#include "paddle/cinn/optim/transform_gpu_forloop.h"
#include "paddle/common/enforce.h"
namespace cinn {
namespace auto_schedule {

std::vector<ir::Var> IndicesToVars(const std::vector<ir::Expr>& indices) {
  std::vector<ir::Var> result;
  for (const ir::Expr& e : indices) {
    // Whether we have to convert other types, like const numbers to Var?
    if (e.As<ir::_Var_>() != nullptr) {
      ir::Expr copy_e = ir::ir_utils::IRCopy(e);
      ir::_Var_* var_ref = copy_e.As<ir::_Var_>();
      result.emplace_back(ir::Var(var_ref));
    }
  }
  return result;
}

void AnalyzeScheduleBlockReadWriteBuffer(ir::ScheduleBlock* sche_block) {
  if (!sche_block->read_buffers.empty() || !sche_block->write_buffers.empty()) {
    return;
  }

  ir::ir_utils::CollectIRNodesWithoutTensor(
      sche_block->body, [&](const Expr* x) {
        const ir::Load* load_expr = x->As<ir::Load>();
        if (load_expr != nullptr) {
          const ir::Tensor t = load_expr->tensor.as_tensor_ref();
          sche_block->read_buffers.emplace_back(
              ir::BufferRange(t->buffer, IndicesToVars(load_expr->indices)));
          return false;
        }
        const ir::Store* store_expr = x->As<ir::Store>();
        if (store_expr != nullptr) {
          const ir::Tensor t = store_expr->tensor.as_tensor_ref();
          sche_block->write_buffers.emplace_back(
              ir::BufferRange(t->buffer, IndicesToVars(store_expr->indices)));
          return false;
        }
        return false;
      });
}

bool ContainsNodeType(ir::Expr expr,
                      const std::unordered_set<ir::IrNodeTy>& node_types) {
  std::vector<ir::Expr> collection =
      ir::ir_utils::CollectIRNodesWithoutTensor(expr, [&](const Expr* x) {
        return node_types.find(x->node_type()) != node_types.end();
      });
  return !collection.empty();
}

std::unordered_set<std::string> GetOutputNamesFromLoweredFunc(
    const std::vector<ir::LoweredFunc>& lowered_funcs) {
  std::unordered_set<std::string> result;
  for (const ir::LoweredFunc& func : lowered_funcs) {
    for (const ir::Argument& arg : func->args) {
      if (arg.is_output()) {
        result.insert(arg.name());
      }
    }
  }
  return result;
}

bool NeedsMultiLevelTiling(const ir::ScheduleBlockRealize& sche_block_realize) {
  const ir::ScheduleBlock* sche_block =
      sche_block_realize.schedule_block.As<ir::ScheduleBlock>();
  if (sche_block->write_buffers.size() != 1 ||
      sche_block->read_buffers.empty()) {
    return false;
  }
  const ir::Expr& write_buffer =
      sche_block->write_buffers[0].As<ir::_BufferRange_>()->buffer;

  // Enumerate each read region, get the number of schedule block iter vars
  // which  are not used to index the read region
  int total_unused_iter_vars = 0;

  for (const ir::Expr& read_buffer_expr : sche_block->read_buffers) {
    const ir::_BufferRange_* read_buffer =
        read_buffer_expr.As<ir::_BufferRange_>();
    // Skip the reduction buffer
    if (read_buffer->buffer == write_buffer) {
      continue;
    }
    // Collect the vars in schedule block that are used to index the read region
    std::unordered_set<std::string> vars_index_read;
    for (const Var& range : read_buffer->ranges) {
      vars_index_read.insert(range->name);
    }
    // Check the block iter vars are not used to index the read region
    int n_unused_block_vars = 0;
    for (const ir::Var& block_iter_var : sche_block->iter_vars) {
      if (!block_iter_var->is_reduce_axis) {
        bool iter_var_in_read = false;
        for (const std::string& var : vars_index_read) {
          if (var == block_iter_var->name) {
            iter_var_in_read = true;
            break;
          }
        }
        if (!iter_var_in_read) {
          ++n_unused_block_vars;
        }
      }
    }
    total_unused_iter_vars += n_unused_block_vars;
  }

  return total_unused_iter_vars >= 1;
}

std::unordered_set<std::string> GetReduceLoopVarNames(const ir::Expr block) {
  const ir::ScheduleBlockRealize* block_realize =
      block.As<ir::ScheduleBlockRealize>();
  PADDLE_ENFORCE_NOT_NULL(block_realize,
                          ::common::errors::InvalidArgument(
                              "The block is not a ScheduleBlockRealize"));
  const ir::ScheduleBlock* block_node =
      block_realize->schedule_block.As<ir::ScheduleBlock>();
  PADDLE_ENFORCE_NOT_NULL(
      block_node,
      ::common::errors::InvalidArgument("The block is not a ScheduleBlock"));
  std::vector<ir::Expr> iter_values = block_realize->iter_values;
  std::vector<ir::Var> iter_vars = block_node->iter_vars;

  std::unordered_set<std::string> reduce_loop_var;
  for (int i = 0; i < iter_vars.size(); ++i) {
    if (iter_vars[i]->is_reduce_axis) {
      ir::ir_utils::CollectIRNodesWithoutTensor(
          iter_values[i], [&](const ir::Expr* x) {
            if (x->as_var()) {
              reduce_loop_var.insert(x->as_var_ref()->name);
            }
            return false;
          });
    }
  }
  return reduce_loop_var;
}

std::string GetBlockName(const ir::Expr block) {
  const ir::ScheduleBlockRealize* block_realize =
      block.As<ir::ScheduleBlockRealize>();
  PADDLE_ENFORCE_NOT_NULL(block_realize,
                          ::common::errors::InvalidArgument(
                              "The block is not a ScheduleBlockRealize"));
  const ir::ScheduleBlock* block_node =
      block_realize->schedule_block.As<ir::ScheduleBlock>();
  PADDLE_ENFORCE_NOT_NULL(
      block_node,
      ::common::errors::InvalidArgument("The block is not a ScheduleBlock"));
  return block_node->name;
}

}  // namespace auto_schedule
}  // namespace cinn
