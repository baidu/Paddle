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

#include "paddle/cinn/hlir/dialect/operator/transforms/add_store_in_fusion_op_pass.h"

#include "paddle/cinn/hlir/dialect/operator/ir/cinn_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/include/core/builtin_type_interfaces.h"
#include "paddle/pir/include/core/dialect.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_op.h"
#include "paddle/pir/include/dialect/shape/utils/shape_analysis.h"
#include "paddle/pir/include/pattern_rewrite/pattern_rewrite_driver.h"

namespace cinn {
namespace dialect {
namespace ir {

class AddYieldStoreInFusionOpPattern
    : public pir::OpRewritePattern<::pir::YieldOp> {
 public:
  using pir::OpRewritePattern<::pir::YieldOp>::OpRewritePattern;

  bool MatchAndRewrite(::pir::YieldOp op,
                       pir::PatternRewriter& rewriter) const override {
    for (auto i = 0; i < op->num_operands(); ++i) {
      rewriter.SetInsertionPointAfter(op->operand_source(i).defining_op());
      auto store_op = rewriter.Build<cinn::dialect::YieldStoreOp>(
          op->operand_source(i), op->operand_source(i).type());
      auto original_base = op->operand_source(i);
      op->operand(i).set_source(store_op.result(0));
    }

    return true;
  }
};

class AddStoreInFusionOpPass : public pir::Pass {
 public:
  AddStoreInFusionOpPass()
      : pir::Pass("add_store_in_fusion_op", /*opt_level=*/1) {}

  bool Initialize(pir::IrContext* context) override {
    pir::RewritePatternSet ps(context);
    ps.Add<AddYieldStoreInFusionOpPattern>(context);

    patterns_ = pir::FrozenRewritePatternSet(std::move(ps));
    return true;
  }

  void Run(pir::Operation* op) override {
    pir::GreedyRewriteConfig cfg;
    cfg.use_top_down_traversal = true;
    cfg.max_iterations = 1;
    for (uint32_t i = 0; i < op->num_regions(); ++i) {
      for (auto& block : op->region(i)) {
        for (auto& op : block) {
          if (op.isa<cinn::dialect::FusionOp>()) {
            auto [_, num_rewrites] =
                pir::ApplyPatternsGreedily(&op, patterns_, cfg);
            AddStatistics(num_rewrites);
          }
        }
      }
    }
  }

  bool CanApplyOn(pir::Operation* op) const override {
    return op->num_regions() > 0;
  }

 private:
  pir::FrozenRewritePatternSet patterns_;
};

class ProcessShadowOutputPattern
    : public pir::OpRewritePattern<::pir::ShadowOutputOp> {
 public:
  using pir::OpRewritePattern<::pir::ShadowOutputOp>::OpRewritePattern;

  bool MatchAndRewrite(::pir::ShadowOutputOp op,
                       pir::PatternRewriter& rewriter) const override {
    auto& shape_ana =
        pir::ShapeAnalysisManager::Instance().Get(op->GetParentProgram());

    auto shape_data = shape_ana.GetShapeOrDataForValue(op->operand_source(0));

    std::cerr << "shape data" << shape_data << std::endl;

    return true;
  }
};

using FullShapeMap =
    std::unordered_map<symbol::ShapeOrDataDimExprs, pir::Value>;
using PartialShapeMap =
    std::unordered_map<symbol::DimExpr, std::pair<pir::Value, size_t>>;
class ShadowOutputAnalysis final {
 public:
  ShadowOutputAnalysis(FullShapeMap* full_shape_map,
                       PartialShapeMap* partial_shape_map)
      : full_shape_map_(full_shape_map),
        partial_shape_map_(partial_shape_map) {}
  void Run(pir::Operation* module_op) { RunImpl(module_op); }

 protected:
  void RunImpl(pir::Operation* op) {
    if (op->isa<pir::ShadowOutputOp>() &&
        (!op->attributes().count("no_need_buffer"))) {
      // std::cerr << "insert shadow output\n";
      if (op->operand_source(0).defining_op()) {
        ProcessShadowOutput(op->operand_source(0));
      }
      return;
    }

    // if( op->isa<paddle::dialect::da>)

    if ((op->dialect()->name() == "pd_op") &&
        (!cinn::hlir::framework::pir::CompatibleInfo::IsSupportForCinn(*op))) {
      // std::cerr << "insert " << op->name() << std::endl;;
      for (size_t i = 0; i < op->num_results(); ++i) {
        ProcessShadowOutput(op->result(i));
      }
      return;
    }

    for (uint32_t i = 0; i < op->num_regions(); ++i) {
      for (auto& block : op->region(i)) {
        for (auto& op : block) {
          RunImpl(&op);
        }
      }
    }
  }

  void ProcessShadowOutput(pir::Value val) {
    if (!val.type().isa<paddle::dialect::DenseTensorType>()) {
      return;
    }
    if (!(val.defining_op())) {
      return;
    }
    auto& shape_analysis = pir::ShapeAnalysisManager::Instance().Get(
        val.defining_op()->GetParentProgram());

    auto shape_data = shape_analysis.GetShapeOrDataForValue(val);

    if (!shape_data.isa<symbol::TensorShapeOrDataDimExprs>()) {
      return;
    }

    if (!full_shape_map_->count(shape_data)) {
      // std::cerr << "insert full shape " << shape_data << std::endl;
      full_shape_map_->emplace(shape_data, val);
    }

    auto shape = shape_data.shape();

    for (size_t i = 0; i < shape.size(); ++i) {
      if (!partial_shape_map_->count(shape[i])) {
        // std::cerr << "insert shape  !! " << shape[i] << std::endl;
        partial_shape_map_->emplace(shape[i], std::make_pair(val, i));
      }
    }
  }

 private:
  FullShapeMap* full_shape_map_;        // not_owned
  PartialShapeMap* partial_shape_map_;  // not_owned
};

class ShadowOutOpPattern : public pir::OpRewritePattern<pir::ShadowOutputOp> {
 public:
  ShadowOutOpPattern(::pir::IrContext* context,
                     const FullShapeMap& full_shape_map,
                     const PartialShapeMap& partial_shape_map)
      : pir::OpRewritePattern<pir::ShadowOutputOp>(context),
        full_shape_map_(full_shape_map),
        partial_shape_map_(partial_shape_map) {}

  bool MatchAndRewrite(pir::ShadowOutputOp shadow_out_op,
                       pir::PatternRewriter& rewriter) const override {
    ::pir::IrContext* ctx = ::pir::IrContext::Instance();

    if (!shadow_out_op->attributes().count("no_need_buffer")) {
      return false;
    }

    auto out_name = shadow_out_op.attribute("output_name")
                        .dyn_cast<pir::StrAttribute>()
                        .AsString();
    if (FindNewBindShape(shadow_out_op, out_name, &rewriter)) {
      rewriter.EraseOp(shadow_out_op);
    }
    return true;
  }

 protected:
  bool FindNewBindShape(pir::ShadowOutputOp shadow_out_op,
                        const std::string& shadow_output_name,
                        pir::PatternRewriter* rewriter) const {
    auto* program = shadow_out_op->GetParentProgram();
    auto& shape_analysis = pir::ShapeAnalysisManager::Instance().Get(program);

    auto shape_data =
        shape_analysis.GetShapeOrDataForValue(shadow_out_op->operand_source(0));

    std::vector<pir::Value> vec_value;
    std::vector<int> tensor_idx;
    std::vector<int> dim_idx;
    if (full_shape_map_.count(shape_data)) {
      std::cerr << "found full shape" << std::endl;
      int rank = shape_data.shape().size();

      for (int i = 0; i < rank; ++i) {
        tensor_idx.push_back(0);
        dim_idx.push_back(i);
      }
      auto combine_op = rewriter->Build<pir::CombineOp>(
          std::vector<pir::Value>({full_shape_map_.at(shape_data)}));
      auto gs_op = rewriter->Build<paddle::dialect::GenerateShapeOp>(
          combine_op.result(0), tensor_idx, dim_idx);
      rewriter->Build<pir::ShadowOutputOp>(gs_op->result(0),
                                           shadow_output_name);

      return true;
    } else if (GetPartialShape(shape_data.shape(),
                               rewriter,
                               &vec_value,
                               &tensor_idx,
                               &dim_idx)) {
      std::cerr << "found partial shape " << std::endl;

      auto combine_op = rewriter->Build<pir::CombineOp>(vec_value);
      auto gs_op = rewriter->Build<paddle::dialect::GenerateShapeOp>(
          combine_op.result(0), tensor_idx, dim_idx);
      rewriter->Build<pir::ShadowOutputOp>(gs_op->result(0),
                                           shadow_output_name);

      return true;
    } else {
      return false;
    }
  }

  bool GetPartialShape(const std::vector<symbol::DimExpr>& vec_dims,
                       pir::PatternRewriter* rewriter,
                       std::vector<pir::Value>* vec_value,
                       std::vector<int>* tensor_idx,
                       std::vector<int>* dim_idx) const {
    bool all_found = true;
    // std::cerr << "partial map size " << partial_shape_map_.size() <<
    // std::endl;
    int index = 0;
    for (auto& d : vec_dims) {
      if (!partial_shape_map_.count(d)) {
        if (d.isa<int64_t>()) {
          auto d_int = d.dyn_cast<int64_t>();
          auto full_op = rewriter->Build<paddle::dialect::FullOp>(
              std::vector<int64_t>({d_int}),
              0.0,
              phi::DataType::BOOL,
              phi::CPUPlace());

          vec_value->emplace_back(full_op.result(0));
          tensor_idx->emplace_back(index++);
          dim_idx->emplace_back(0);
        } else {
          std::cerr << "!!!!!!!!!!!!!! can not found " << d << std::endl;
          all_found = false;
        }
      } else {
        vec_value->emplace_back(partial_shape_map_.at(d).first);
        tensor_idx->emplace_back(index++);
        dim_idx->emplace_back(partial_shape_map_.at(d).second);
      }
    }

    return all_found;
  }

 private:
  const FullShapeMap& full_shape_map_;        // not owned
  const PartialShapeMap& partial_shape_map_;  // not owned
};

class ProcessShadowOutputPass : public pir::PatternRewritePass {
 public:
  ProcessShadowOutputPass()
      : pir::PatternRewritePass("process shadow output pass", 1) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext* context) override {
    pir::RewritePatternSet ps(context);
    ps.Add<ShadowOutOpPattern>(context, full_shape_value_, partial_shape_value);

    return ps;
  }

  bool CanApplyOn(pir::Operation* op) const override {
    if (op->isa<pir::ModuleOp>()) {
      // begin to process all the shadow output
      ShadowOutputAnalysis(&full_shape_value_, &partial_shape_value).Run(op);
    }
    return op->num_regions() > 0;
  }

 private:
  mutable FullShapeMap full_shape_value_;
  mutable PartialShapeMap partial_shape_value;
};

std::unique_ptr<pir::Pass> CreateAddStoreInFusionOpPass() {
  return std::make_unique<AddStoreInFusionOpPass>();
}

std::unique_ptr<pir::Pass> CreateProcessShadowOutputPass() {
  return std::make_unique<ProcessShadowOutputPass>();
}

}  // namespace ir
}  // namespace dialect
}  // namespace cinn
