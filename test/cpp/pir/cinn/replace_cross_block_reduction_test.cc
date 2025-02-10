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

#include "paddle/cinn/optim/replace_cross_block_reduction.h"

#include <gtest/gtest.h>

#include "paddle/cinn/cinn.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/op/ir_operators.h"
#include "paddle/cinn/ir/schedule/ir_schedule.h"
#include "paddle/cinn/ir/utils/stmt_converter.h"
#include "paddle/cinn/utils/string.h"

namespace cinn {
namespace optim {

TEST(CrossBlockReductionReplacer, SRLayout) {
  Context::Global().ResetNameId();

  Placeholder<float> A("A", {Expr(8), Expr(16)});
  Var reduce_k(8, "reduce_k");
  ir::Tensor B = Compute(
      {Expr(16)},
      [&](Var i) { return lang::ReduceSum(A(reduce_k, i), {reduce_k}); },
      "B");
  ir::Tensor C = Compute(
      {Expr(16)}, [&](Var i) { return lang::Sqrt(B(i)); }, "C");

  ast_gen_ius::TensorGroup tensor_group({A, B, C});
  auto func = lang::LowerToAst("reduce_sum_sqrt", {C}, &tensor_group);
  ir::Expr expr_func_body = ir::ConvertStmtBlockToExprBlock(func->body_block);
  ir::ModuleExpr mod_expr({expr_func_body});
  ir::IRSchedule ir_sch(mod_expr);

  ir_sch.Bind(ir_sch.GetLoops("B")[0], "blockIdx.x");
  ir_sch.Bind(ir_sch.GetLoops("B")[1], "blockIdx.y");
  ir_sch.Bind(ir_sch.GetLoops("C")[0], "blockIdx.x");

  func->body = ir_sch.GetModule().GetExprs()[0];
  A->WithBuffer("global", "_A");
  B->WithBuffer("local", "_B_temp_buffer");
  func->temp_bufs = {A->buffer, B->buffer};

  VLOG(6) << "Before ReplaceCrossBlockReduction: " << func;
  ReplaceCrossBlockReduction(func);
  VLOG(6) << "After ReplaceCrossBlockReduction: " << func;

  EXPECT_EQ(utils::GetStreamCnt(func),
            utils::Trim(R"ROC(function reduce_sum_sqrt (_C, _A, _semaphore)
{
  ScheduleBlock(root)
  {
    {
      thread_bind[blockIdx.x] for (i, 0, 16)
      {
        ScheduleBlock(B__reduce_init)
        {
          i0 = axis.bind(i)
          {
            B__reduce_init[i0] = 0.00000000f
          }
        }
        thread_bind[blockIdx.y] for (reduce_k, 0, 8)
        {
          is_last_block_done[0] = cinn_grid_reduce_update_semaphore(Tensor(semaphore, [16]))
          if (is_last_block_done[0]) {
            ScheduleBlock(B)
            {
              i0_0, i1 = axis.bind(i, reduce_k)
              {
                B[i0_0] = cinn_grid_reduce_sum_fp32(Tensor(A, [8,16]), 16, i0_0)
              }
            }
          }
        }
      }
      thread_bind[blockIdx.x] for (i, 0, 16)
      {
        if (is_last_block_done[0]) {
          ScheduleBlock(C)
          {
            i0_1 = axis.bind(i)
            {
              C[i0_1] = sqrt(B[i0_1])
            }
          }
        }
      }
    }
  }
}
)ROC"));
  EXPECT_EQ(func->temp_spaces.size(), 2);
  EXPECT_EQ(func->temp_spaces[0].size().as_int64(), 512);
  EXPECT_EQ(func->temp_spaces[0].arg_idx(), 1);
  EXPECT_EQ(func->temp_spaces[0].need_zero_init(), false);
  EXPECT_EQ(func->temp_spaces[1].size().as_int64(), 64);
  EXPECT_EQ(func->temp_spaces[1].arg_idx(), 2);
  EXPECT_EQ(func->temp_spaces[1].need_zero_init(), true);
}

TEST(CrossBlockReductionReplacer, RSLayout) {
  Context::Global().ResetNameId();

  Placeholder<float> A("A", {Expr(8), Expr(4), Expr(32)});
  Var reduce_k(8, "reduce_k");
  ir::Tensor B = Compute(
      {Expr(4), Expr(32)},
      [&](Var i, Var j) {
        return lang::ReduceMax(A(reduce_k, i, j), {reduce_k});
      },
      "B");
  ir::Tensor C = Compute(
      {Expr(4), Expr(32)},
      [&](Var i, Var j) { return lang::Exp(B(i, j)); },
      "C");

  ast_gen_ius::TensorGroup tensor_group({A, B, C});
  auto func = lang::LowerToAst("reduce_max_exp", {C}, &tensor_group);

  ir::Expr expr_func_body = ir::ConvertStmtBlockToExprBlock(func->body_block);
  ir::ModuleExpr mod_expr({expr_func_body});
  ir::IRSchedule ir_sch(mod_expr);

  ir_sch.Bind(ir_sch.GetLoops("B")[0], "blockIdx.x");
  ir_sch.Bind(ir_sch.GetLoops("B")[1], "threadIdx.x");
  ir_sch.Bind(ir_sch.GetLoops("B")[2], "blockIdx.y");
  ir_sch.Bind(ir_sch.GetLoops("C")[0], "blockIdx.x");
  ir_sch.Bind(ir_sch.GetLoops("C")[1], "threadIdx.x");

  func->body = ir_sch.GetModule().GetExprs()[0];
  A->WithBuffer("global", "_A");
  B->WithBuffer("local", "_B_temp_buffer");
  func->temp_bufs = {A->buffer, B->buffer};

  VLOG(6) << "Before ReplaceCrossBlockReduction: " << func;
  ReplaceCrossBlockReduction(func);
  VLOG(6) << "After ReplaceCrossBlockReduction: " << func;

  EXPECT_EQ(utils::GetStreamCnt(func),
            utils::Trim(R"ROC(function reduce_max_exp (_C, _A, _semaphore)
{
  ScheduleBlock(root)
  {
    {
      thread_bind[blockIdx.x] for (i, 0, 4)
      {
        thread_bind[threadIdx.x] for (j, 0, 32)
        {
          ScheduleBlock(B__reduce_init)
          {
            i0, i1 = axis.bind(i, j)
            {
              B__reduce_init[i0, i1] = -3.40282347e+38f
            }
          }
          thread_bind[blockIdx.y] for (reduce_k, 0, 8)
          {
            is_last_block_done[0] = cinn_grid_reduce_update_semaphore(Tensor(semaphore, [4]))
            if (is_last_block_done[0]) {
              ScheduleBlock(B)
              {
                i0_0, i1_0, i2 = axis.bind(i, j, reduce_k)
                {
                  B[i0_0, i1_0] = cinn_grid_reduce_max_fp32(Tensor(A, [8,4,32]), 128, ((i0_0 * 32) + i1_0))
                }
              }
            }
          }
        }
      }
      thread_bind[blockIdx.x] for (i, 0, 4)
      {
        thread_bind[threadIdx.x] for (j, 0, 32)
        {
          if (is_last_block_done[0]) {
            ScheduleBlock(C)
            {
              i0_1, i1_1 = axis.bind(i, j)
              {
                C[i0_1, i1_1] = exp(B[i0_1, i1_1])
              }
            }
          }
        }
      }
    }
  }
}
)ROC"));
  EXPECT_EQ(func->temp_spaces.size(), 2);
  EXPECT_EQ(func->temp_spaces[0].size().as_int64(), 4096);
  EXPECT_EQ(func->temp_spaces[0].arg_idx(), 1);
  EXPECT_EQ(func->temp_spaces[0].need_zero_init(), false);
  EXPECT_EQ(func->temp_spaces[1].size().as_int64(), 16);
  EXPECT_EQ(func->temp_spaces[1].arg_idx(), 2);
  EXPECT_EQ(func->temp_spaces[1].need_zero_init(), true);
}

}  // namespace optim
}  // namespace cinn
