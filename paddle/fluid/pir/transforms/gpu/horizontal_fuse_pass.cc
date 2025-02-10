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

#include "paddle/fluid/pir/transforms/gpu/horizontal_fuse_pass.h"

#include <string>
#include <vector>

#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/utils/general_functions.h"

#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/core/ir_context.h"
#include "paddle/pir/include/core/operation.h"
#include "paddle/pir/include/core/program.h"
#include "paddle/pir/include/core/region.h"
#include "paddle/pir/include/core/value.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"
#include "paddle/pir/include/pattern_rewrite/frozen_rewrite_pattern_set.h"
#include "paddle/pir/include/pattern_rewrite/pattern_match.h"
#include "paddle/pir/include/pattern_rewrite/pattern_rewrite_driver.h"

namespace {

class HorizontalFusePattern : public pir::RewritePattern {
 public:
  explicit HorizontalFusePattern(pir::IrContext* context)
      : RewritePattern(MatchAnyOpTypeTag(),
                       1 /*benefit*/,
                       context,
                       {} /*generated_names*/) {}

  // distinguish input is X or Y, where to get weight
  // 1 means w from Y, 0 means w from x
  mutable uint32_t w_from_source_index;

  bool MatchAndRewrite(
      pir::Operation* op,
      pir::PatternRewriter& rewriter) const override {  // NOLINT
    bool match_flag = false;
    return false;
    for (size_t i = 0; i < op->num_results(); i++) {
      // only fuse three or more op
      if (GetOpCntUseValue(op->result(i)) >= 3) {
        match_flag = true;
        RewriteOpsbyValue(op, &rewriter, i);
        VLOG(4) << "horizontal_fuse_pass applied rewrite on [" << op->name()
                << "] op";
      }
    }
    return match_flag;
  }

 private:
  uint32_t operands_source_index(pir::Operation* op,
                                 const pir::Value& target_value) const {
    std::vector<pir::Value> sources = op->operands_source();
    auto it = std::find(sources.begin(), sources.end(), target_value);

    if (it != sources.end()) {
      return std::distance(sources.begin(), it);
    } else {
      return static_cast<uint32_t>(-1);  // use -1 to indicate not found
    }
  }

  bool IsValidOp(pir::Operation* curr_op, const pir::Value& val) const {
    // judge prev_op is a ParameterOp\ConstantTensorOp\DataOp
    auto areOperandsValid = [this, curr_op, val]() -> bool {
      uint32_t curr_op_input_index = operands_source_index(curr_op, val);
      for (uint32_t i = 0; i < curr_op->num_operands(); i++) {
        auto* prev_op = pir::GetDefiningOpForInput(curr_op, i);
        if (i != curr_op_input_index) {
          return prev_op && (prev_op->isa<pir::ParameterOp>() ||
                             prev_op->isa<pir::ConstantTensorOp>() ||
                             prev_op->isa<paddle::dialect::DataOp>());
        }
      }
      return false;
    };

    if (curr_op->isa<paddle::dialect::GemmEpilogueOp>() ||
        curr_op->isa<paddle::dialect::FcOp>()) {
      return areOperandsValid();
    }

    // check Attribute
    if (curr_op->isa<paddle::dialect::MatmulOp>()) {
      if (curr_op->HasAttribute("transpose_x") &&
          curr_op->attribute<pir::BoolAttribute>("transpose_x").data() ==
              true) {
        return false;
      }
      if (curr_op->HasAttribute("transpose_y") &&
          curr_op->attribute<pir::BoolAttribute>("transpose_y").data() ==
              true) {
        return false;
      }
      return areOperandsValid();
    }

    return false;
  }

  bool AreOperationsEqual(const pir::Operation* op1,
                          const pir::Operation* op2) const {
    if (op1->name() != op2->name() ||
        op1->num_operands() != op2->num_operands() ||
        op1->num_results() != op2->num_results()) {
      return false;
    }

    // when pdmodel has sub_graph, the shape of y1 and y2 may be different
    // Check if dimensions sizes are the same and first dimensions are the same
    const pir::Value& w1 = op1->operand_source(w_from_source_index);
    const pir::Value& w2 = op2->operand_source(w_from_source_index);
    auto w1_dims = pir::GetShapeFromValue(w1);
    auto w2_dims = pir::GetShapeFromValue(w2);

    // Determine which dimension to compare based on w_from_source_index
    std::size_t compare_dim = (w_from_source_index == 0) ? 1 : 0;

    // Check if dimensions sizes are different or the specified dimension value
    // is different
    if (w1_dims.size() != w2_dims.size() ||
        w1_dims[compare_dim] != w2_dims[compare_dim]) {
      return false;
    }
    return true;
  }

  int GetOpCntUseValue(const pir::Value& val) const {
    int op_cnt_use_x = 0;
    pir::Operation* op_example = nullptr;

    for (auto it = val.use_begin(); it != val.use_end(); ++it) {
      pir::Operation* curr_op = it.owner();
      if (!IsValidOp(curr_op, val)) {
        continue;
      }
      if (op_example == nullptr) {
        op_example = curr_op;
        op_cnt_use_x++;
        w_from_source_index = operands_source_index(curr_op, val) == 0 ? 1 : 0;
      } else {
        if (AreOperationsEqual(op_example, curr_op)) {
          op_cnt_use_x++;
        }
      }
    }
    return op_cnt_use_x;
  }
  template <typename OpTy, typename... Args>
  pir::Value InsertOpAfter(pir::Operation** prev_op_ptr,
                           pir::PatternRewriter* rewriter,
                           Args&&... args) const {
    rewriter->SetInsertionPointAfter(*prev_op_ptr);
    pir::Operation* new_op = rewriter->Build<OpTy>(std::forward<Args>(args)...);
    *prev_op_ptr = new_op;
    return new_op->result(0);
  }

  template <typename OpTy, typename... Args>
  pir::Value InsertOpWithOrder(pir::Operation** prev_op,
                               pir::PatternRewriter* rewriter,
                               int w_from_source_index,
                               const pir::Value& val,
                               const pir::Value& w_qkv,
                               Args&&... args) const {
    if (w_from_source_index == 0) {
      return InsertOpAfter<OpTy>(
          prev_op, rewriter, w_qkv, val, std::forward<Args>(args)...);
    } else {
      return InsertOpAfter<OpTy>(
          prev_op, rewriter, val, w_qkv, std::forward<Args>(args)...);
    }
  }

  pir::Value InsertFusedOp(const std::string& fused_matmul_op_name,
                           pir::Operation** prev_op,
                           pir::PatternRewriter* rewriter,
                           const pir::Value& val,
                           const pir::Value& w_qkv,
                           const pir::Value& bias_qkv,
                           const pir::AttributeMap& fused_matmul_op_attrs,
                           int w_from_source_index) const {
    if (fused_matmul_op_name == paddle::dialect::MatmulOp::name()) {
      return InsertOpWithOrder<paddle::dialect::MatmulOp>(
          prev_op,
          rewriter,
          w_from_source_index,
          val,
          w_qkv,
          fused_matmul_op_attrs);
    } else if (fused_matmul_op_name ==
               paddle::dialect::GemmEpilogueOp::name()) {
      return InsertOpWithOrder<paddle::dialect::GemmEpilogueOp>(
          prev_op,
          rewriter,
          w_from_source_index,
          val,
          w_qkv,
          bias_qkv,
          fused_matmul_op_attrs);
    } else {
      return InsertOpWithOrder<paddle::dialect::FcOp>(prev_op,
                                                      rewriter,
                                                      w_from_source_index,
                                                      val,
                                                      w_qkv,
                                                      bias_qkv,
                                                      fused_matmul_op_attrs);
    }
  }

  void RewriteOpsbyValue(pir::Operation* op,
                         pir::PatternRewriter* rewriter,
                         size_t idx) const {
    /// val is used by the op, which belongs to a kind of
    /// MatmulOp/GemmEpilogueOp/FcOp

    // prepare val
    pir::Value val = op->result(idx);
    std::vector<int64_t> x_dims = pir::GetShapeFromValue(val);

    int64_t x_last_axis =
        w_from_source_index == 1 ? x_dims.size() - 1 : x_dims.size() - 2;

    // prepare src ops
    std::vector<pir::Operation*> fused_matmul_ops;
    std::string fused_matmul_op_name;
    pir::AttributeMap fused_matmul_op_attrs;
    // prepare weight
    std::vector<int64_t> w_shapes;
    std::vector<pir::Value> combine_op_inputs_weight;
    // prepare bias
    std::vector<pir::Value> combine_op_inputs_bias;
    // prepare outs
    std::vector<pir::Value> src_outs;

    for (auto it = val.use_begin(); it != val.use_end(); ++it) {
      pir::Operation* curr_op = it.owner();

      if (!IsValidOp(curr_op, val)) {
        continue;
      }
      fused_matmul_ops.push_back(curr_op);
      combine_op_inputs_weight.push_back(
          curr_op->operand_source(w_from_source_index));
      auto w_dims =
          pir::GetShapeFromValue(curr_op->operand_source(w_from_source_index));

      // input is X
      if (w_from_source_index) {
        w_shapes.push_back(w_dims[w_dims.size() - 1]);
      } else {
        // input is Y
        w_shapes.push_back(w_dims[0]);
      }

      src_outs.push_back(curr_op->result(0));

      if (fused_matmul_op_name.empty()) {
        fused_matmul_op_name = curr_op->name();
        fused_matmul_op_attrs = curr_op->attributes();
      }
      if (curr_op->num_operands() > 2) {
        combine_op_inputs_bias.push_back(curr_op->operand_source(2));
      }
    }
    /// build new graph
    /// insert new ops
    pir::Operation* prev_op = op;
    auto weight_combined = InsertOpAfter<pir::CombineOp>(
        &prev_op, rewriter, combine_op_inputs_weight);

    auto w_qkv = InsertOpAfter<paddle::dialect::ConcatOp>(
        &prev_op, rewriter, weight_combined, w_from_source_index);

    pir::Value bias_qkv;
    if (combine_op_inputs_bias.size()) {
      auto bias_combined = InsertOpAfter<pir::CombineOp>(
          &prev_op, rewriter, combine_op_inputs_bias);
      bias_qkv = InsertOpAfter<paddle::dialect::ConcatOp>(
          &prev_op, rewriter, bias_combined, 0);
    }

    pir::Value xw_fused = InsertFusedOp(fused_matmul_op_name,
                                        &prev_op,
                                        rewriter,
                                        val,
                                        w_qkv,
                                        bias_qkv,
                                        fused_matmul_op_attrs,
                                        w_from_source_index);

    pir::Value xw_splitted = InsertOpAfter<paddle::dialect::SplitOp>(
        &prev_op, rewriter, xw_fused, w_shapes, x_last_axis);

    rewriter->SetInsertionPointAfter(prev_op);

    auto split_builtin_op = rewriter->Build<pir::SplitOp>(xw_splitted);

    std::vector<pir::Value> xw = split_builtin_op.outputs();

    // replace res Value
    for (size_t k = 0; k < src_outs.size(); k++) {
      rewriter->ReplaceAllUsesWith(src_outs[k], xw[k]);
    }

    // delete fused matmul op
    for (auto fused_matmul_op : fused_matmul_ops) {
      rewriter->EraseOp(fused_matmul_op);
    }
  }
};

class HorizontalFusePass : public pir::Pass {
 public:
  HorizontalFusePass() : pir::Pass("horizontal_fuse_pass", 2) {}

  bool Initialize(pir::IrContext* context) override {
    pir::RewritePatternSet ps(context);
    ps.Add<HorizontalFusePattern>(context);
    patterns_ = pir::FrozenRewritePatternSet(std::move(ps));
    return true;
  }

  void Run(pir::Operation* op) override {
    pir::GreedyRewriteConfig cfg;
    cfg.use_top_down_traversal = true;

    cfg.max_iterations = 10;
    auto [_, num_rewrites] = pir::ApplyPatternsGreedily(op, patterns_, cfg);
    AddStatistics(num_rewrites);
  }

 private:
  pir::FrozenRewritePatternSet patterns_;
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateHorizontalFusePass() {
  return std::make_unique<HorizontalFusePass>();
}

}  // namespace pir

REGISTER_IR_PASS(horizontal_fuse_pass, HorizontalFusePass);
