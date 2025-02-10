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

#include "paddle/cinn/hlir/dialect/operator/transforms/replace_zero_scale_to_full_pass.h"

#include "paddle/cinn/hlir/dialect/operator/ir/cinn_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"
#include "paddle/pir/include/pattern_rewrite/pattern_applicator.h"
#include "paddle/pir/include/pattern_rewrite/pattern_match.h"
#include "paddle/pir/include/pattern_rewrite/pattern_rewrite_driver.h"

namespace cinn {
namespace dialect {
namespace ir {

bool IsGeneByFullOp(pir::Operation* op, int32_t input_idx) {
  return input_idx < op->num_operands() && op->operand_source(input_idx) &&
         op->operand_source(input_idx).defining_op() &&
         op->operand_source(input_idx)
             .defining_op()
             ->isa<paddle::dialect::FullOp>();
}

float GetFullValue(paddle::dialect::FullOp full_op) {
  return full_op.attribute("value")
      .dyn_cast<paddle::dialect::ScalarAttribute>()
      .data()
      .to<float>();
}

bool ReplaceWithFullOp(pir::Operation* op,
                       pir::PatternRewriter* rewriter,
                       int32_t align_input_idx) {
  auto out_type = op->result(0).type();
  if (!out_type.isa<paddle::dialect::DenseTensorType>()) {
    return false;
  }
  auto tensor_type = out_type.dyn_cast<paddle::dialect::DenseTensorType>();
  if (!(out_type.dyn_cast<pir::ShapedTypeInterface>().IsDynamicShape())) {
    auto phi_dtype = paddle::dialect::TransToPhiDataType(tensor_type.dtype());
    auto full_op = rewriter->Build<paddle::dialect::FullOp>(
        phi::vectorize(tensor_type.dims()), 0.0, phi_dtype);

    rewriter->ReplaceAllUsesWith(op->result(0), full_op.result(0));

    return true;
  }

  return false;
}

class ReplaceZeroScaleToFullPattern
    : public pir::OpRewritePattern<paddle::dialect::ScaleOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::ScaleOp>::OpRewritePattern;

  bool MatchAndRewrite(paddle::dialect::ScaleOp op,
                       pir::PatternRewriter& rewriter) const override {
    if (IsGeneByFullOp(op, 1)) {
      auto full_op = op.operand_source(1)
                         .defining_op()
                         ->dyn_cast<paddle::dialect::FullOp>();
      auto scale_value = GetFullValue(full_op);
      auto bias_value =
          op->attributes().at("bias").dyn_cast<pir::FloatAttribute>().data();

      if (scale_value == 0.0f && bias_value == 0.0f) {
        // replace to full(0)
        return ReplaceWithFullOp(op, &rewriter, 0);
      }
    }

    return false;
  }
};

class ReplaceMultiplyToFullPattern
    : public pir::OpRewritePattern<paddle::dialect::MultiplyOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::MultiplyOp>::OpRewritePattern;

  bool MatchAndRewrite(paddle::dialect::MultiplyOp op,
                       pir::PatternRewriter& rewriter) const override {
    if (IsGeneByFullOp(op, 0)) {
      auto full_op = op.operand_source(0)
                         .defining_op()
                         ->dyn_cast<paddle::dialect::FullOp>();
      auto full_value = GetFullValue(full_op);

      if (full_value == 0.0f) {
        return ReplaceWithFullOp(op, &rewriter, 1);
      }
    }

    if (IsGeneByFullOp(op, 1)) {
      auto full_op = op.operand_source(1)
                         .defining_op()
                         ->dyn_cast<paddle::dialect::FullOp>();
      auto full_value = GetFullValue(full_op);

      if (full_value == 0.0f) {
        return ReplaceWithFullOp(op, &rewriter, 0);
      }
    }

    return false;
  }
};

class ReplaceAddToFullPattern
    : public pir::OpRewritePattern<paddle::dialect::AddOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::AddOp>::OpRewritePattern;

  bool MatchAndRewrite(paddle::dialect::AddOp op,
                       pir::PatternRewriter& rewriter) const override {
    if (IsGeneByFullOp(op, 0) && IsGeneByFullOp(op, 1)) {
      auto x_full_op = op.operand_source(0)
                           .defining_op()
                           ->dyn_cast<paddle::dialect::FullOp>();
      auto x_full_value = GetFullValue(x_full_op);

      auto y_full_op = op.operand_source(1)
                           .defining_op()
                           ->dyn_cast<paddle::dialect::FullOp>();
      auto y_full_value = GetFullValue(y_full_op);

      if (x_full_value == 0.0f && y_full_value == 0.0f) {
        return ReplaceWithFullOp(op, &rewriter, 0);
      }
    }
    return false;
  }
};

class ReplaceZeroScaleToFullPass : public pir::PatternRewritePass {
 public:
  ReplaceZeroScaleToFullPass()
      : pir::PatternRewritePass("replace_zero_scale_to_full_pass", 1) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext* context) override {
    pir::RewritePatternSet ps(context);

    // replace x * 0 to full(0)
    ps.Add<ReplaceZeroScaleToFullPattern>(context);
    ps.Add<ReplaceMultiplyToFullPattern>(context);
    ps.Add<ReplaceAddToFullPattern>(context);
    return ps;
  }

  bool CanApplyOn(pir::Operation* op) const override {
    return op->num_regions() > 0;
  }
};

std::unique_ptr<pir::Pass> CreateReplaceZeroScaleToFullPass() {
  return std::make_unique<ReplaceZeroScaleToFullPass>();
}
}  // namespace ir
}  // namespace dialect
}  // namespace cinn
