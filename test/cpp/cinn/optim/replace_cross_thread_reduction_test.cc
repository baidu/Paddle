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

#include "paddle/cinn/optim/replace_cross_thread_reduction.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "paddle/cinn/cinn.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/op/ir_operators.h"
#include "paddle/cinn/ir/schedule/ir_schedule.h"
#include "paddle/cinn/ir/utils/stmt_converter.h"
#include "paddle/cinn/utils/string.h"

namespace cinn {
namespace optim {

TEST(CrossThreadReductionReplacer, basic) {
#ifdef CINN_WITH_CUDA
  Context::Global().ResetNameId();
  Placeholder<float> A("A", {Expr(64), Expr(128)});
  Target target = cinn::common::DefaultNVGPUTarget();
  Module::Builder builder("reduce_sum", target);
  Var reduce_j(128, "reduce_j");
  ir::Tensor B = Compute(
      {Expr(64)},
      [&](Var i) { return lang::ReduceSum(A(i, reduce_j), {reduce_j}); },
      "B");
  ast_gen_ius::TensorGroup tensor_group({A, B});
  auto func = lang::LowerToAst("reduce_sum", {A, B}, &tensor_group);
  VLOG(6) << "original func\n" << func;

  ir::Expr expr_func_body = ir::ConvertStmtBlockToExprBlock(func->body_block);
  ir::ModuleExpr mod_expr({expr_func_body});
  ir::IRSchedule ir_sch(mod_expr);

  ir_sch.Bind(ir_sch.GetLoops("B")[0], "blockIdx.x");
  ir_sch.Bind(ir_sch.GetLoops("B")[1], "threadIdx.x");

  ir::Expr func_body = ir_sch.GetModule().GetExprs()[0];
  std::vector<ir::Argument> args{
      ir::Argument(ir::Var("A"), ir::Argument::IO::kInput),
      ir::Argument(ir::Var("B"), ir::Argument::IO::kOutput)};
  auto new_func = ir::_LoweredFunc_::Make("test_func", args, func_body, {});
  VLOG(6) << "After Bind: " << new_func->body;

  ReplaceCrossThreadReduction(new_func);
  VLOG(6) << "After ReplaceCrossThreadReduction: " << new_func->body;

  EXPECT_EQ(utils::GetStreamCnt(new_func->body), utils::Trim(R"ROC({
  ScheduleBlock(root)
  {
    {
      thread_bind[blockIdx.x] for (i, 0, 64)
      {
        ScheduleBlock(B__reduce_init)
        {
          i0 = axis.bind(i)
          {
            B__reduce_init[i0] = 0.00000000f
          }
        }
        thread_bind[threadIdx.x] for (reduce_j, 0, 128)
        {
          ScheduleBlock(B)
          {
            i0_0, i1 = axis.bind(i, reduce_j)
            {
              B[i0_0] = cinn_block_reduce_sum_fp32(A[i0_0, i1], _Buffer_<cinn_buffer_t*: 32>(shm32__fp32_reduce), false)
            }
          }
        }
      }
    }
  }
}
)ROC"));
#endif
}

}  // namespace optim
}  // namespace cinn
