// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/pir/transforms/onednn/cpu_bfloat16_placement_pass.h"

#include "paddle/fluid/pir/dialect/operator/interface/op_yaml_info.h"
#include "paddle/fluid/pir/dialect/operator/ir/onednn_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/dialect/operator/utils/op_yaml_info_parser.h"
#include "paddle/fluid/pir/dialect/operator/utils/op_yaml_info_util.h"
#include "paddle/fluid/pir/utils/general_functions.h"

#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"
#include "paddle/pir/include/pattern_rewrite/pattern_match.h"

#include "paddle/fluid/pir/dialect/kernel/ir/kernel_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"
#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/core/ir_context.h"
#include "paddle/pir/include/core/operation.h"

namespace {

bool CheckIfknownShape(pir::Operation* op, size_t index) {
  bool is_from_tensor = false;
  std::vector<int64_t> shape = paddle::dialect::ParseValueShape(
      op->operand_source(index), &is_from_tensor);
  size_t num_minus = 0;
  for (auto i : shape) {
    if (i == -1) num_minus++;
  }
  // If all dims are -1, then the shape is actually unknown.
  if (num_minus == shape.size()) return false;
  return true;
}

class OneDNNBf16PlacementPattern : public pir::RewritePattern {
 public:
  explicit OneDNNBf16PlacementPattern(pir::IrContext* context)
      : pir::RewritePattern(MatchAnyOpTypeTag(),
                            5 /*benefit*/,
                            context,
                            {} /*generated_names*/) {}

  bool Match(pir::Operation* op) const override {  // NOLINT
    if (!op->isa<paddle::onednn::dialect::BilinearInterpOp>() &&
        !op->isa<paddle::onednn::dialect::CastOp>() &&
        !op->isa<paddle::onednn::dialect::Cast_Op>() &&
        !op->isa<paddle::onednn::dialect::ClipOp>() &&
        !op->isa<paddle::onednn::dialect::Clip_Op>() &&
        !op->isa<paddle::onednn::dialect::ConcatOp>() &&
        !op->isa<paddle::onednn::dialect::Conv2dOp>() &&
        !op->isa<paddle::onednn::dialect::Conv2dTransposeOp>() &&
        !op->isa<paddle::onednn::dialect::Conv2dTransposeBiasOp>() &&
        !op->isa<paddle::onednn::dialect::AddOp>() &&
        !op->isa<paddle::onednn::dialect::Add_Op>() &&
        !op->isa<paddle::onednn::dialect::MultiplyOp>() &&
        !op->isa<paddle::onednn::dialect::Multiply_Op>() &&
        !op->isa<paddle::onednn::dialect::FcOp>() &&
        !op->isa<paddle::onednn::dialect::FusionGruOp>() &&
        !op->isa<paddle::onednn::dialect::GeluOp>() &&
        !op->isa<paddle::onednn::dialect::LayerNormOp>() &&
        !op->isa<paddle::onednn::dialect::MatmulOp>() &&
        !op->isa<paddle::onednn::dialect::Pool2dOp>() &&
        !op->isa<paddle::onednn::dialect::PreluOp>() &&
        !op->isa<paddle::onednn::dialect::ReluOp>() &&
        !op->isa<paddle::onednn::dialect::Relu_Op>() &&
        !op->isa<paddle::onednn::dialect::Reshape_Op>() &&
        !op->isa<paddle::onednn::dialect::ReshapeOp>() &&
        !op->isa<paddle::onednn::dialect::ScaleOp>() &&
        !op->isa<paddle::onednn::dialect::Scale_Op>() &&
        !op->isa<paddle::onednn::dialect::SigmoidOp>() &&
        !op->isa<paddle::onednn::dialect::Sigmoid_Op>() &&
        !op->isa<paddle::onednn::dialect::SliceOp>() &&
        !op->isa<paddle::onednn::dialect::SoftmaxOp>() &&
        !op->isa<paddle::onednn::dialect::Softmax_Op>() &&
        !op->isa<paddle::onednn::dialect::SqueezeOp>() &&
        !op->isa<paddle::onednn::dialect::Squeeze_Op>() &&
        !op->isa<paddle::onednn::dialect::SumOp>() &&
        !op->isa<paddle::onednn::dialect::TransposeOp>() &&
        !op->isa<paddle::onednn::dialect::SplitOp>() &&
        !op->isa<paddle::onednn::dialect::Transpose_Op>() &&
        !op->isa<paddle::onednn::dialect::FusedConv2dOp>() &&
        !op->isa<paddle::onednn::dialect::FusedMatmulOp>()) {
      return false;
    }

    // The pass use HasOpINT8DataType to skip int8 op
    auto op_attr = op->attributes();
    if (op_attr.find("mkldnn_data_type") != op_attr.end()) {
      auto mkldnn_data_type = op_attr.at("mkldnn_data_type")
                                  .dyn_cast<pir::StrAttribute>()
                                  .AsString();
      // Reduce repetitive match
      if (mkldnn_data_type != "float32") {
        return false;
      }
    }

    if (op_attr.find("use_quantizer") != op_attr.end()) {
      if (op_attr.at("use_quantizer").dyn_cast<pir::BoolAttribute>().data()) {
        return false;
      }
    }
    if (op->name() == "onednn_op.scale" || op->name() == "onednn_op.scale_") {
      bool bias_after_scale =
          op_attr.at("bias_after_scale").dyn_cast<pir::BoolAttribute>().data();
      if (bias_after_scale) {
        // If bias after scale, add quant/dequant for scale will cause some
        // error
        return false;
      }
    }

    const std::vector<std::string> permitted_input_names = {
        "x", "y", "input", "residual_param", "residual_data"};
    auto op_name = op->name();
    auto op_info = pir::IrContext::Instance()->GetRegisteredOpInfo(op_name);
    if (!op_info) return false;
    paddle::dialect::OpYamlInfoParser yaml_parser(
        op_info.GetInterfaceImpl<paddle::dialect::OpYamlInfoInterface>()
            ->get_op_info_(op_name),
        paddle::dialect::IsLegacyOp(op_name));
    auto input_names = yaml_parser.InputNames();

    for (size_t i = 0; i < op->num_operands(); i++) {
      pir::Value value = op->operand_source(i);
      if (!value) continue;
      std::string input_name = input_names[i];
      auto iter = std::find(permitted_input_names.begin(),
                            permitted_input_names.end(),
                            input_name);
      if (iter == permitted_input_names.end()) {
        continue;
      }
      pir::Type type = op->operand_type(i);
      if (!type) continue;
      if (type.isa<pir::VectorType>()) {
        // Support pir::VectorType in bf16
        // Special op will do detailed check in its pattern
        pir::VectorType vector_type = value.type().dyn_cast<pir::VectorType>();
        for (size_t idx = 0; idx < static_cast<size_t>(vector_type.size());
             idx++) {
          auto input_type =
              vector_type[idx].isa<paddle::dialect::DenseTensorType>();
          // We don't precess nested VectorType
          if (!input_type) return false;
          pir::Type input_dtype =
              vector_type[idx]
                  .dyn_cast<paddle::dialect::DenseTensorType>()
                  .dtype();
          // Only float input can be converted to bfloat16
          if (!input_dtype.isa<pir::Float32Type>()) return false;
        }
      } else if (type.isa<paddle::dialect::DenseTensorType>()) {
        pir::Type op_dtype = pir::GetDataTypeFromValue(value);
        // Only float input can be converted to bfloat16
        if (!op_dtype.isa<pir::Float32Type>()) return false;
      } else {
        return false;
      }
    }

    // Workaround for reshape & slice when shape is unknown
    // TODO(Xinyi): Since we can't distinguish when IntArray is produced by
    // Combine, currently we fix it in a specific way. In future, we may think
    // out a more generalized method
    if (op_name == "onednn_op.reshape_" || op_name == "onednn_op.reshape") {
      return CheckIfknownShape(op, 1);
    } else if (op_name == "onednn_op.slice") {
      return CheckIfknownShape(op, 1) && CheckIfknownShape(op, 2);
    }

    return true;
  }

  void Rewrite(pir::Operation* op,
               pir::PatternRewriter& rewriter) const override {  // NOLINT
    std::string target_op_name = op->name();

    auto op_info =
        pir::IrContext::Instance()->GetRegisteredOpInfo(target_op_name);
    if (op_info) {
      std::vector<pir::Type> op_item_inner_output_types;
      for (size_t i = 0; i < op->num_results(); ++i) {
        op_item_inner_output_types.push_back(op->result_type(i));
      }
      auto attributes = op->attributes();
      for (auto& attr : attributes) {
        if (attr.first == "mkldnn_data_type") {
          VLOG(8) << "mkldnn_data_type set to bf16, op:" << target_op_name;
          attributes[attr.first] =
              pir::StrAttribute::get(pir::IrContext::Instance(), "bfloat16");
        }
      }

      pir::Operation* op_item_inner = rewriter.Build(op->operands_source(),
                                                     attributes,
                                                     op_item_inner_output_types,
                                                     op_info);
      rewriter.ReplaceOp(op, op_item_inner->results());
    }
  }
};

class RemoveOrphanedPattern : public pir::RewritePattern {
 public:
  explicit RemoveOrphanedPattern(pir::IrContext* context)
      : pir::RewritePattern(MatchAnyOpTypeTag(),
                            1 /*benefit*/,
                            context,
                            {} /*generated_names*/) {}

  // find orphaned bfloat16 operator that is between two float32 operators
  // revert mkldnn_data_type attr to float32
  bool Match(pir::Operation* op) const override {  // NOLINT
    if (!op->isa<paddle::onednn::dialect::BilinearInterpOp>() &&
        !op->isa<paddle::onednn::dialect::CastOp>() &&
        !op->isa<paddle::onednn::dialect::Cast_Op>() &&
        !op->isa<paddle::onednn::dialect::ClipOp>() &&
        !op->isa<paddle::onednn::dialect::Clip_Op>() &&
        !op->isa<paddle::onednn::dialect::ConcatOp>() &&
        !op->isa<paddle::onednn::dialect::Conv2dOp>() &&
        !op->isa<paddle::onednn::dialect::Conv2dTransposeOp>() &&
        !op->isa<paddle::onednn::dialect::Conv2dTransposeBiasOp>() &&
        !op->isa<paddle::onednn::dialect::AddOp>() &&
        !op->isa<paddle::onednn::dialect::Add_Op>() &&
        !op->isa<paddle::onednn::dialect::MultiplyOp>() &&
        !op->isa<paddle::onednn::dialect::Multiply_Op>() &&
        !op->isa<paddle::onednn::dialect::FcOp>() &&
        !op->isa<paddle::onednn::dialect::FusionGruOp>() &&
        !op->isa<paddle::onednn::dialect::GeluOp>() &&
        !op->isa<paddle::onednn::dialect::LayerNormOp>() &&
        !op->isa<paddle::onednn::dialect::MatmulOp>() &&
        !op->isa<paddle::onednn::dialect::Pool2dOp>() &&
        !op->isa<paddle::onednn::dialect::PreluOp>() &&
        !op->isa<paddle::onednn::dialect::ReluOp>() &&
        !op->isa<paddle::onednn::dialect::Relu_Op>() &&
        !op->isa<paddle::onednn::dialect::Reshape_Op>() &&
        !op->isa<paddle::onednn::dialect::ReshapeOp>() &&
        !op->isa<paddle::onednn::dialect::ScaleOp>() &&
        !op->isa<paddle::onednn::dialect::Scale_Op>() &&
        !op->isa<paddle::onednn::dialect::SigmoidOp>() &&
        !op->isa<paddle::onednn::dialect::Sigmoid_Op>() &&
        !op->isa<paddle::onednn::dialect::SliceOp>() &&
        !op->isa<paddle::onednn::dialect::SoftmaxOp>() &&
        !op->isa<paddle::onednn::dialect::Softmax_Op>() &&
        !op->isa<paddle::onednn::dialect::SqueezeOp>() &&
        !op->isa<paddle::onednn::dialect::Squeeze_Op>() &&
        !op->isa<paddle::onednn::dialect::SumOp>() &&
        !op->isa<paddle::onednn::dialect::TransposeOp>() &&
        !op->isa<paddle::onednn::dialect::SplitOp>() &&
        !op->isa<paddle::onednn::dialect::Transpose_Op>() &&
        !op->isa<paddle::onednn::dialect::FusedConv2dOp>() &&
        !op->isa<paddle::onednn::dialect::FusedMatmulOp>()) {
      return false;
    }
    auto op_attr = op->attributes();
    if (op_attr.find("mkldnn_data_type") != op_attr.end()) {
      auto mkldnn_data_type = op_attr.at("mkldnn_data_type")
                                  .dyn_cast<pir::StrAttribute>()
                                  .AsString();
      if (mkldnn_data_type != "bfloat16") {
        return false;
      }
    }

    bool prev_fp32 = false;
    bool next_fp32 = false;
    const std::set<std::string> constant_op({"pd_op.data",
                                             "builtin.parameter",
                                             "pd_op.feed",
                                             "pd_op.fetch",
                                             "pd_op.assign"});

    const std::vector<std::string> permitted_input_names = {
        "x", "y", "input", "residual_param", "residual_data"};
    auto op_name = op->name();
    auto op_info = pir::IrContext::Instance()->GetRegisteredOpInfo(op_name);
    if (!op_info) return false;
    paddle::dialect::OpYamlInfoParser yaml_parser(
        op_info.GetInterfaceImpl<paddle::dialect::OpYamlInfoInterface>()
            ->get_op_info_(op_name),
        paddle::dialect::IsLegacyOp(op_name));
    auto input_names = yaml_parser.InputNames();

    if (op->num_operands()) {
      for (uint32_t i = 0; i < op->num_operands(); i++) {
        if (!op->operand_source(i) || !op->operand_source(i).type()) {
          continue;
        }
        std::string input_name = input_names[i];
        auto iter = std::find(permitted_input_names.begin(),
                              permitted_input_names.end(),
                              input_name);
        if (iter == permitted_input_names.end()) {
          // The input in permitted_input, it must be bf16, others can be fp32
          continue;
        }
        auto* prev_op = pir::GetDefiningOpForInput(op, i);
        // Some ops do not need to be processed
        std::string prev_name = prev_op->name();
        if (constant_op.count(prev_name)) {
          continue;
        }

        auto op_attr = prev_op->attributes();
        if (op_attr.find("mkldnn_data_type") == op_attr.end()) {
          // data_type_is_missing
          prev_fp32 = true;
          break;
        }
        auto mkldnn_data_type = op_attr.at("mkldnn_data_type")
                                    .dyn_cast<pir::StrAttribute>()
                                    .AsString();

        if (mkldnn_data_type == "float32") {
          prev_fp32 = true;
          break;
        }
      }
    } else {
      // The first op in graph should be treated as prev_fp32 = true
      prev_fp32 = true;
    }

    size_t num_useops = 0;
    for (uint32_t i = 0; i < op->num_results(); i++) {
      if (!op->result(i) || !op->result(i).type()) {
        continue;
      }
      auto next_op_list = pir::GetUseOpsForOutput(op, i);
      num_useops += next_op_list.size();
      for (auto const& [next_op, op_index] : next_op_list) {
        // Some ops do not need to be processed
        std::string next_op_name = next_op->name();
        if (constant_op.count(next_op_name)) {
          continue;
        }
        auto op_next_attr = next_op->attributes();
        if (op_next_attr.find("mkldnn_data_type") == op_next_attr.end()) {
          // data_type_is_missing
          VLOG(8) << "data_type_is_missing:" << next_op->name();
          next_fp32 = true;
          break;
        }
        auto mkldnn_data_type = op_next_attr.at("mkldnn_data_type")
                                    .dyn_cast<pir::StrAttribute>()
                                    .AsString();
        if (mkldnn_data_type == "float32") {
          VLOG(8) << "mkldnn_data_type is fp32:" << next_op->name();
          next_fp32 = true;
          break;
        }
      }
    }

    // Check if it's the last op on graph. If it is, this op can be seen as a
    // fp32 op down here
    if (num_useops == 0) next_fp32 = true;

    return prev_fp32 && next_fp32;
  }

  void Rewrite(pir::Operation* op,
               pir::PatternRewriter& rewriter) const override {  // NOLINT
    std::string target_op_name = op->name();
    auto op_info =
        pir::IrContext::Instance()->GetRegisteredOpInfo(target_op_name);
    if (op_info) {
      std::vector<pir::Type> op_item_inner_output_types;
      for (size_t i = 0; i < op->num_results(); ++i) {
        op_item_inner_output_types.push_back(op->result_type(i));
      }
      auto attributes = op->attributes();

      if (attributes.find("mkldnn_data_type") != attributes.end()) {
        attributes["mkldnn_data_type"] =
            pir::StrAttribute::get(pir::IrContext::Instance(), "float32");
      }

      pir::Operation* op_item_inner = rewriter.Build(op->operands_source(),
                                                     attributes,
                                                     op_item_inner_output_types,
                                                     op_info);
      rewriter.ReplaceOp(op, op_item_inner->results());
    }
  }
};

class OneDNNPlacementBf16Pass : public pir::PatternRewritePass {
 public:
  OneDNNPlacementBf16Pass()
      : pir::PatternRewritePass("cpu_bfloat16_placement_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext* context) override {
    pir::RewritePatternSet ps(context);
    ps.Add<OneDNNBf16PlacementPattern>(context);
    ps.Add<RemoveOrphanedPattern>(context);

    return ps;
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateCpuBf16PlacementPass() {
  return std::make_unique<OneDNNPlacementBf16Pass>();
}

}  // namespace pir

REGISTER_IR_PASS(cpu_bfloat16_placement_pass, OneDNNPlacementBf16Pass);
