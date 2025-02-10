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

#include "paddle/cinn/hlir/dialect/operator/transforms/transpose_optimize_pass.h"

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

template <typename T = int>
std::vector<T> GetVectorFromIntArrayAttribute(
    const pir::ArrayAttribute &array_attr) {
  const auto &vector_attr = array_attr.AsVector();

  std::vector<T> result;
  if (vector_attr.size() > 0) {
    PADDLE_ENFORCE_EQ(vector_attr[0].isa<::pir::Int32Attribute>(),
                      true,
                      ::common::errors::Unimplemented(
                          "the 0th elementwise MUST be ir::Int64Attribute"));
    for (size_t i = 0; i < vector_attr.size(); ++i) {
      result.push_back(vector_attr[i].dyn_cast<::pir::Int32Attribute>().data());
    }
  }
  return result;
}

bool CanRemove(const std::vector<int32_t> &pre_axis,
               const std::vector<int32_t> &base_axis) {
  if (base_axis.size() != pre_axis.size()) {
    return false;
  }

  std::vector<int32_t> out_dim(base_axis.size(), 0);

  std::vector<int32_t> base_dim;
  for (size_t i = 0; i < pre_axis.size(); ++i) {
    out_dim[i] = pre_axis[base_axis[i]];

    base_dim.push_back(i);
  }

  return base_dim == out_dim;
}

class TransposeOptimizePattern
    : public pir::OpRewritePattern<paddle::dialect::TransposeOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::TransposeOp>::OpRewritePattern;

  bool MatchAndRewrite(paddle::dialect::TransposeOp op,
                       pir::PatternRewriter &rewriter) const override {
    std::set<std::string> can_process_transpose_set{"pd_op.cos", "pd_op.sin"};

    const std::vector<int32_t> base_axis =
        GetVectorFromIntArrayAttribute<int32_t>(
            op.attribute("perm").template dyn_cast<pir::ArrayAttribute>());

    auto pre_op = op->operand_source(0).defining_op();

    if (can_process_transpose_set.count(pre_op->name())) {
      if (pre_op->operand_source(0)
              .defining_op()
              ->isa<paddle::dialect::TransposeOp>()) {
        auto pre_transpose = pre_op->operand_source(0)
                                 .defining_op()
                                 ->dyn_cast<paddle::dialect::TransposeOp>();

        const std::vector<int32_t> pre_axis =
            GetVectorFromIntArrayAttribute<int32_t>(
                pre_transpose.attribute("perm")
                    .template dyn_cast<pir::ArrayAttribute>());

        if (CanRemove(pre_axis, base_axis)) {
          pre_op->operand(0).set_source(pre_transpose.operand_source(0));
          rewriter.ReplaceAllUsesWith(op->result(0), op->operand_source(0));

          rewriter.EraseOp(pre_transpose);
          rewriter.EraseOp(op);
        }
      }
    }

    return true;
  }
};

TransposeOptimizePass::TransposeOptimizePass()
    : pir::PatternRewritePass("transpose_optimize", 1) {}

pir::RewritePatternSet TransposeOptimizePass::InitializePatterns(
    pir::IrContext *context) {
  pir::RewritePatternSet ps(context);
  ps.Add<TransposeOptimizePattern>(context);

  return ps;
}

bool TransposeOptimizePass::CanApplyOn(pir::Operation *op) const {
  return op->num_regions() > 0;
}

std::unique_ptr<pir::Pass> CreateTransposeOptimizePass() {
  return std::make_unique<TransposeOptimizePass>();
}

}  // namespace ir
}  // namespace dialect
}  // namespace cinn
