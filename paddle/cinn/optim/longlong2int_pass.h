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

#pragma once
#include "paddle/cinn/ir/stmt.h"
#include "paddle/cinn/pass/pass.h"

namespace cinn {
namespace optim {
class LongLong2IntStmtPass : public StmtPass {
 public:
  LongLong2IntStmtPass() : StmtPass("longlong2int_stmt") {}
  LogicalResult Run(ir::stmt::StmtRef stmt) override;
};

class LongLong2IntExprPass : public ExprPass {
 public:
  LongLong2IntExprPass() : ExprPass("longlong2int_expr") {}
  LogicalResult Run(ir::Expr expr) override;
};

/**
 * Converts int64 (long long) types to int32 in a Stmt where possible.
 *
 * IMPORTANT: Before applying this pass, it is MANDATORY to use
 * `CanApplyLongLong2Int` to check for potential overflow issues.
 *
 * This pass is applicable in scenarios where the IR contains int64 types that
 * can be safely represented as int32 without overflow.
 *
 * When applied, this pass will convert int64 expression to int32
 * in various constructs, including:
 * - Tensor shapes and indices
 * - Loop variables and bounds
 * - Buffer metadata (shapes, strides, offsets)
 * - Comparison operations (index only)
 *
 * Examples:
 * 1. Loop variable conversion:
 * Before conversion:
 * {
 *   ScheduleBlock(root_12)
 *   {
 *     attrs(tile_method:TileFirstGeneralTactic)
 *     thread_bind[blockIdx.x] for (blockIdx.x, 0, 352)
 *     {
 *       thread_bind[threadIdx.x] for (threadIdx.x, 0, 256)
 *       {
 *         ScheduleBlock(var_2)
 *         {
 *           i0, i1, i2, i3 = axis.bind(idx / 4096, (idx % 4096) / 256, (idx %
 * 256) / 16, idx % 16) read_buffers(_var[i0(0:22ll), i2(0:16ll)])
 *           write_buffers(_var_2[i0(0:22ll), i1(0:16ll), i2(0:16ll),
 * i3(0:16ll)])
 *         var_2[i0, i1, i2, i3] = var[i0, i2, i3 + i1 * 16ll]
 *         }
 *       }
 *     }
 *   }
 * }
 *
 * After conversion:
 * {
 *   ScheduleBlock(root_12)
 *   {
 *     attrs(tile_method:TileFirstGeneralTactic)
 *     thread_bind[blockIdx.x] for (blockIdx.x, 0, 352)
 *     {
 *       thread_bind[threadIdx.x] for (threadIdx.x, 0, 256)
 *       {
 *         ScheduleBlock(var_2)
 *         {
 *           i0, i1, i2, i3 = axis.bind(idx / 4096, (idx % 4096) / 256, (idx %
 * 256) / 16, idx % 16) read_buffers(_var[i0(0:22), i2(0:16)])
 *           write_buffers(_var_2[i0(0:22), i1(0:16), i2(0:16),i3(0:16)])
 *           var_2[i0, i1, i2, i3] = var[i0, i2, i3 + i1 * 16ll]
 *         }
 *       }
 *     }
 *   }
 * }
 *
 * The 16ll in var[i0, i2, i3 + i1 * 16ll] is not converted for it is part of
 * Load Exoression, which will be converted in LongLong2IntExprPass.
 */
std::unique_ptr<StmtPass> CreateLongLong2IntStmtPass();

/**
 * Converts int64 (long long) types to int32 in a Expr where possible.
 *
 * IMPORTANT: Before applying this pass, it is MANDATORY to use
 * `CanApplyLongLong2Int` to check for potential overflow issues.
 *
 * This pass is applicable in scenarios where the IR contains int64 types that
 * can be safely represented as int32 without overflow.
 *
 * When applied, this pass will convert int64 expression to int32
 * in various constructs, including:
 * - Tensor shapes and indices
 * - Loop variables and bounds
 * - Buffer metadata (shapes, strides, offsets)
 * - Comparison operations (index only)
 *
 * Examples:
 * 1. Loop variable conversion:
 * Before conversion:
 * {
 *   ScheduleBlock(root_12)
 *   {
 *     attrs(tile_method:TileFirstGeneralTactic)
 *     thread_bind[blockIdx.x] for (blockIdx.x, 0, 352)
 *     {
 *       thread_bind[threadIdx.x] for (threadIdx.x, 0, 256)
 *       {
 *         ScheduleBlock(var_2)
 *         {
 *           i0, i1, i2, i3 = axis.bind(idx / 4096, (idx % 4096) / 256, (idx %
 * 256) / 16, idx % 16) read_buffers(_var[i0(0:22ll), i2(0:16ll)])
 *           write_buffers(_var_2[i0(0:22ll), i1(0:16ll), i2(0:16ll),
 * i3(0:16ll)])
 *         var_2[i0, i1, i2, i3] = var[i0, i2, i3 + i1 * 16ll]
 *         }
 *       }
 *     }
 *   }
 * }
 *
 * After conversion:
 * {
 *   ScheduleBlock(root_12)
 *   {
 *     attrs(tile_method:TileFirstGeneralTactic)
 *     thread_bind[blockIdx.x] for (blockIdx.x, 0, 352)
 *     {
 *       thread_bind[threadIdx.x] for (threadIdx.x, 0, 256)
 *       {
 *         ScheduleBlock(var_2)
 *         {
 *           i0, i1, i2, i3 = axis.bind(idx / 4096ll, (idx % 4096ll) / 256ll,
 * (idx % 256ll) / 16ll, idx % 16ll) read_buffers(_var[i0(0:22ll), i2(0:16ll)])
 *           write_buffers(_var_2[i0(0:22ll), i1(0:16ll),
 * i2(0:16ll),i3(0:16ll)]) var_2[i0, i1, i2, i3] = var[i0, i2, i3 + i1 * 16]
 *         }
 *       }
 *     }
 *   }
 * }
 *
 * Only 16ll in var[i0, i2, i3 + i1 * 16ll] is converted for other longlong
 * Exprs are components of ScheduleBlock, which will be converted in
 * LongLong2IntStmtPass.
 */
std::unique_ptr<ExprPass> CreateLongLong2IntExprPass();

// Check if the given block can be converted from long long to int,
// A.K.A. the product of the extents of all possible nested loops is within
// INT_MAX
bool CanApplyLongLong2Int(ir::stmt::BlockRef block);

/**
 * Converts int64 (long long) types to int32 in a block where possible.
 *
 * This pass is applicable in scenarios where the IR contains int64 types that
 * can be safely represented as int32 without overflow.
 *
 * When applied, this pass will traverse the IR and convert int64 types to int32
 * in various constructs, including:
 * - Tensor shapes and indices
 * - Loop variables and bounds
 * - Buffer metadata (shapes, strides, offsets)
 * - Comparison operations
 *
 * Overflow checking:
 * The pass performs overflow checking primarily for nested for-loops. This
 * focus on nested loops is based on the assumption that they are the most
 * common source of potential overflows in typical computational kernels. The
 * check considers:
 * - The product of loop extents (iteration counts)
 * - Whether loop bounds are constant and of index type
 *
 *
 * Examples:
 * 1. Loop variable conversion:
 * Before conversion:
 * {
 *   ScheduleBlock(root_12)
 *   {
 *     attrs(tile_method:TileFirstGeneralTactic)
 *     thread_bind[blockIdx.x] for (blockIdx.x, 0, 352)
 *     {
 *       thread_bind[threadIdx.x] for (threadIdx.x, 0, 256)
 *       {
 *         ScheduleBlock(var_2)
 *         {
 *           i0, i1, i2, i3 = axis.bind(idx / 4096, (idx % 4096) / 256, (idx %
 * 256) / 16, idx % 16) read_buffers(_var[i0(0:22ll), i2(0:16ll)])
 *           write_buffers(_var_2[i0(0:22ll), i1(0:16ll), i2(0:16ll),
 * i3(0:16ll)])
 *         var_2[i0, i1, i2, i3] = var[i0, i2, i3 + i1 * 16ll]
 *         }
 *       }
 *     }
 *   }
 * }
 *
 * After conversion:
 * {
 *   ScheduleBlock(root_12)
 *   {
 *     attrs(tile_method:TileFirstGeneralTactic)
 *     thread_bind[blockIdx.x] for (blockIdx.x, 0, 352)
 *     {
 *       thread_bind[threadIdx.x] for (threadIdx.x, 0, 256)
 *       {
 *         ScheduleBlock(var_2)
 *         {
 *           i0, i1, i2, i3 = axis.bind(idx / 4096, (idx % 4096) / 256, (idx %
 * 256) / 16, idx % 16) read_buffers(_var[i0(0:22), i2(0:16)])
 *           write_buffers(_var_2[i0(0:22), i1(0:16), i2(0:16),i3(0:16)])
 *           var_2[i0, i1, i2, i3] = var[i0, i2, i3 + i1 * 16]
 *         }
 *       }
 *     }
 *   }
 * }
 */
void CastLonglong2Int(ir::stmt::BlockRef block);

}  // namespace optim
}  // namespace cinn
