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

#include "paddle/cinn/hlir/dialect/operator/transforms/reduce_as_to_sum_pass.h"

#include <regex>
#include "paddle/cinn/hlir/dialect/operator/ir/cinn_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/group_merge/op_with_group_merge_util.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/refresh_combine_pattern.h"
#include "paddle/cinn/hlir/framework/pir/utils.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"
#include "paddle/pir/include/core/builtin_dialect.h"
#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_manager.h"
#include "paddle/pir/include/pattern_rewrite/pattern_rewrite_driver.h"

PD_DECLARE_string(deny_cinn_ops);

namespace cinn {
namespace dialect {
namespace ir {
using CompatibleInfo = cinn::hlir::framework::pir::CompatibleInfo;
using paddle::dialect::FullIntArrayOp;
using paddle::dialect::FullOp;

class ReduceAsOpPattern
    : public pir::OpRewritePattern<paddle::dialect::ReduceAsOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::ReduceAsOp>::OpRewritePattern;

  bool MatchAndRewrite(paddle::dialect::ReduceAsOp op,
                       pir::PatternRewriter &rewriter) const override {
    auto x_shape =
        phi::vectorize(op->operand_source(0)
                           .type()
                           .dyn_cast<paddle::dialect::DenseTensorType>()
                           .dims());

    auto y_shape =
        phi::vectorize(op->operand_source(1)
                           .type()
                           .dyn_cast<paddle::dialect::DenseTensorType>()
                           .dims());

    size_t x_rank = x_shape.size();
    size_t y_rank = y_shape.size();
    if (y_rank == 1 && y_shape[0] == 1) {
      // TODO(phrain): reduce to shape [1] will failed in codegen
      return false;
    }
    int64_t compare_offset = x_rank - y_rank;
    std::vector<int64_t> reduce_axis;

    for (int64_t i = 0; i < compare_offset; ++i) {
      reduce_axis.push_back(i);
    }

    bool x_y_shape_equal = false;
    bool is_static_shape = IsStaticShape(x_shape, y_shape);
    bool keep_dim = true;
    bool need_squeeze = false;
    if (is_static_shape) {
      x_y_shape_equal = (x_shape == y_shape);
      ProcessStaticShape(
          x_shape, y_shape, &reduce_axis, &keep_dim, &need_squeeze);
    } else {
      bool can_replace = ProcessDynamicShape(
          op, &reduce_axis, &x_y_shape_equal, &keep_dim, &need_squeeze);
      if (!can_replace) {
        return true;
      }
    }
    if (x_y_shape_equal) {
      rewriter.ReplaceAllUsesWith(op.result(0), op.operand_source(0));
      return false;
    }

    auto pir_dtype =
        op->operand_source(0).type().dyn_cast<pir::DenseTensorType>().dtype();
    auto phi_dtype = paddle::dialect::TransToPhiDataType(pir_dtype);
    auto sum_op = rewriter.Build<paddle::dialect::SumOp>(
        op.operand_source(0), reduce_axis, phi_dtype, keep_dim);

    auto new_output = sum_op.result(0);

    if (need_squeeze) {
      std::vector<int64_t> sequeeze_dim;
      for (int64_t i = 0; i < compare_offset; ++i) {
        sequeeze_dim.push_back(i);
      }

      auto squeeze_op =
          rewriter.Build<paddle::dialect::SqueezeOp>(new_output, sequeeze_dim);
      new_output = squeeze_op.result(0);
    }

    rewriter.ReplaceAllUsesWith(op.result(0), new_output);

    rewriter.EraseOp(op);

    return true;
  }

 private:
  bool IsStaticShape(const std::vector<int64_t> &x_shape,
                     const std::vector<int64_t> &y_shape) const {
    bool x_has_dynamic_shape =
        std::find(x_shape.begin(), x_shape.end(), -1) != x_shape.end();
    bool y_has_dynamic_shape =
        std::find(y_shape.begin(), y_shape.end(), -1) != y_shape.end();

    return (!x_has_dynamic_shape) && (!y_has_dynamic_shape);
  }

  void GetKeepDimAndSqueezeInfo(size_t x_rank,
                                size_t y_rank,
                                bool no_keep_dim_eq_output,
                                bool *keep_dim,
                                bool *need_squeeze) const {
    if (x_rank == y_rank) {
      *keep_dim = true;
      *need_squeeze = false;
    } else if (no_keep_dim_eq_output) {
      *keep_dim = false;
      *need_squeeze = false;
    } else {
      *keep_dim = true;
      *need_squeeze = true;
    }
  }

  void ProcessStaticShape(const std::vector<int64_t> &x_shape,
                          const std::vector<int64_t> &y_shape,
                          std::vector<int64_t> *reduce_axis,
                          bool *keep_dim,
                          bool *need_squeeze) const {
    size_t x_rank = x_shape.size();
    size_t y_rank = y_shape.size();

    // Get reduc axis and
    int64_t compare_offset = x_rank - y_rank;

    for (size_t i = 0; i < y_rank; ++i) {
      if (y_shape[i] == 1 && x_shape[i + compare_offset] != 1) {
        reduce_axis->push_back(compare_offset + i);
      }
    }

    std::set<int64_t> reduce_axis_set(reduce_axis->begin(), reduce_axis->end());
    std::vector<int64_t> new_shape;
    for (size_t i = 0; i < x_shape.size(); ++i) {
      if (!reduce_axis_set.count(i)) {
        new_shape.push_back(x_shape[i]);
      }
    }

    GetKeepDimAndSqueezeInfo(
        x_rank, y_rank, (new_shape == y_shape), keep_dim, need_squeeze);
  }
  bool ProcessDynamicShape(paddle::dialect::ReduceAsOp op,
                           std::vector<int64_t> *reduce_axis,
                           bool *x_y_shape_equal,
                           bool *keep_dim,
                           bool *need_squeeze) const {
    auto &shape_analysis =
        pir::ShapeAnalysisManager::Instance().Get(op->GetParentProgram());

    const auto &x_shape =
        shape_analysis.GetShapeOrDataForValue(op->operand_source(0)).shape();
    const auto &y_shape =
        shape_analysis.GetShapeOrDataForValue(op->operand_source(1)).shape();

    if (x_shape == y_shape) {
      *x_y_shape_equal = true;
      return true;
    } else {
      size_t x_rank = x_shape.size();
      size_t y_rank = y_shape.size();

      int64_t compare_offset = x_rank - y_rank;
      bool can_replace_with_sum = true;

      for (size_t i = 0; i < y_rank; ++i) {
        bool x_dim_i_eq_one = x_shape[i + compare_offset].isa<int64_t>() &&
                              x_shape[i + compare_offset].Get<int64_t>() == 1;
        bool y_dim_i_eq_one =
            y_shape[i].isa<int64_t>() && y_shape[i].Get<int64_t>() == 1;
        if (y_dim_i_eq_one && (!x_dim_i_eq_one)) {
          reduce_axis->push_back(compare_offset + i);
        } else if (x_shape[i + compare_offset] != y_shape[i]) {
          can_replace_with_sum = false;
          break;
        }
      }

      std::set<int64_t> reduce_axis_set(reduce_axis->begin(),
                                        reduce_axis->end());
      std::vector<symbol::DimExpr> new_shape;
      for (size_t i = 0; i < x_shape.size(); ++i) {
        if (!reduce_axis_set.count(i)) {
          new_shape.push_back(x_shape[i]);
        }
      }
      GetKeepDimAndSqueezeInfo(
          x_rank, y_rank, (new_shape == y_shape), keep_dim, need_squeeze);

      return can_replace_with_sum;
    }
  }
};

ReduceAsToSumPass::ReduceAsToSumPass()
    : pir::PatternRewritePass("reduce_as_to_sum_pass", 1) {}

pir::RewritePatternSet ReduceAsToSumPass::InitializePatterns(
    pir::IrContext *context) {
  pir::RewritePatternSet ps(context);
  ps.Add<ReduceAsOpPattern>(context);
  ps.Add<RefreshCombineOpPattern>(context);

  return ps;
}

bool ReduceAsToSumPass::CanApplyOn(pir::Operation *op) const {
  return op->num_regions() > 0;
}

std::unique_ptr<pir::Pass> CreateReduceAsToSumPass() {
  return std::make_unique<ReduceAsToSumPass>();
}

}  // namespace ir
}  // namespace dialect
}  // namespace cinn
