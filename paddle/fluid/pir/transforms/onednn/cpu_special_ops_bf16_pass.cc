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

#include "paddle/fluid/pir/transforms/onednn/cpu_special_ops_bf16_pass.h"

#include "paddle/fluid/pir/dialect/kernel/ir/kernel_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/onednn_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"
#include "paddle/fluid/pir/utils/general_functions.h"

#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace {

template <class IrType1, class IrType2>
static pir::Type create_type(pir::Type type,
                             pir::Type out_dtype,
                             pir::IrContext *ctx) {
  auto input_type = type.dyn_cast<IrType1>();
  return IrType2::get(ctx,
                      out_dtype,
                      input_type.dims(),
                      input_type.data_layout(),
                      input_type.lod(),
                      input_type.offset());
}

// Move cast quantize to cpu_special_op_bf16_pass.h
template <typename OpType>
class CastBf16Pattern : public pir::OpRewritePattern<OpType> {
 public:
  using pir::OpRewritePattern<OpType>::OpRewritePattern;

  bool MatchAndRewrite(
      OpType op,
      pir::PatternRewriter &rewriter) const override {  // NOLINT
    std::string target_op_name = op->name();
    if (!(target_op_name == "onednn_op.cast" ||
          target_op_name == "onednn_op.cast_"))
      return false;
    auto *pre_op = pir::GetDefiningOpForInput(op, 0);
    if (pre_op && pre_op->name() == "onednn_op.quantize") return false;

    auto attributes = op->attributes();
    auto onednn_data_type = attributes["mkldnn_data_type"];
    std::string onednn_dtype =
        onednn_data_type.template dyn_cast<pir::StrAttribute>().AsString();
    if (onednn_dtype != "bfloat16") return false;

    pir::IrContext *ctx = rewriter.ir_context();

    auto dtype_attr = attributes["dtype"];
    phi::DataType dtype =
        dtype_attr.template dyn_cast<paddle::dialect::DataTypeAttribute>()
            .data();
    if (dtype == phi::DataType::FLOAT32) {
      pir::Attribute new_dtype =
          paddle::dialect::DataTypeAttribute::get(ctx, phi::DataType::BFLOAT16);
      attributes["dtype"] = new_dtype;
    } else {
      return false;
    }

    std::unordered_map<std::string, pir::Attribute> q_attributes;
    q_attributes["scale"] = rewriter.float_attr(1.0f);
    q_attributes["shift"] = rewriter.float_attr(0.0f);
    q_attributes["is_negative_input"] = rewriter.bool_attr(false);
    q_attributes["output_format"] = rewriter.str_attr("NCHW");
    q_attributes["bfloat16"] = rewriter.bool_attr(true);

    paddle::onednn::dialect::QuantizeOp q_op =
        rewriter.Build<paddle::onednn::dialect::QuantizeOp>(
            op->operand_source(0), q_attributes);

    auto type = q_op->result_type(0);
    pir::Type new_type =
        create_type<pir::DenseTensorType, paddle::dialect::DenseTensorType>(
            type, pir::BFloat16Type::get(ctx), ctx);
    q_op->result(0).set_type(new_type);

    OpType new_cast = rewriter.Build<OpType>(q_op.output(), attributes);

    std::unordered_map<std::string, pir::Attribute> dq_attributes;
    dq_attributes["scale"] = rewriter.float_attr(1.0f);
    dq_attributes["shift"] = rewriter.float_attr(0.0f);
    paddle::onednn::dialect::DequantizeOp dq_op =
        rewriter.Build<paddle::onednn::dialect::DequantizeOp>(new_cast.out(),
                                                              dq_attributes);
    rewriter.ReplaceAllUsesWith(op->result(0), dq_op.output());
    rewriter.EraseOp(op);
    return true;
  }
};

// For ops like conv and concat, their input is sometimes packed as VectorType,
// hence current quantization doesn't work. Here we deal with them specifically.
class ConcatBf16QuantizePattern
    : public pir::OpRewritePattern<paddle::onednn::dialect::ConcatOp> {
 public:
  using pir::OpRewritePattern<
      paddle::onednn::dialect::ConcatOp>::OpRewritePattern;
  bool MatchAndRewrite(
      paddle::onednn::dialect::ConcatOp op,
      pir::PatternRewriter &rewriter) const override {  // NOLINT
    // The input should come from combine.
    pir::CombineOp pre_op =
        pir::GetDefiningOpForInput(op, 0)->dyn_cast<pir::CombineOp>();
    if (!pre_op) return false;
    if (!pre_op.out().HasOneUse()) return false;

    auto op_attributes = op->attributes();
    auto onednn_data_type = op_attributes.at("mkldnn_data_type")
                                .dyn_cast<pir::StrAttribute>()
                                .AsString();
    if (onednn_data_type != "bfloat16") return false;

    auto combine_inputs = pre_op.inputs();

    for (size_t idx = 0; idx < combine_inputs.size(); idx++) {
      // Check if it's already quantized
      auto pre_pre_op = pir::GetDefiningOpForInput(pre_op, idx);
      if (pre_pre_op && pre_pre_op->name() == "onednn_op.quantize")
        return false;
    }

    pir::IrContext *ctx = rewriter.ir_context();

    std::unordered_map<std::string, pir::Attribute> q_attributes;
    q_attributes["scale"] = rewriter.float_attr(1.0f);
    q_attributes["shift"] = rewriter.float_attr(0.0f);
    q_attributes["is_negative_input"] = rewriter.bool_attr(false);
    q_attributes["output_format"] = rewriter.str_attr("NCHW");
    q_attributes["bfloat16"] = rewriter.bool_attr(true);

    // Insert quantize before combine
    std::vector<pir::Value> new_combine_inputs(combine_inputs.size());
    for (size_t idx = 0; idx < combine_inputs.size(); idx++) {
      paddle::onednn::dialect::QuantizeOp quant_op =
          rewriter.Build<paddle::onednn::dialect::QuantizeOp>(
              combine_inputs[idx], q_attributes);
      auto type = quant_op->result_type(0);
      pir::Type new_type =
          create_type<pir::DenseTensorType, paddle::dialect::DenseTensorType>(
              type, pir::BFloat16Type::get(ctx), ctx);
      quant_op->result(0).set_type(new_type);
      new_combine_inputs[idx] = quant_op.output();
    }

    // Create new combine
    pir::CombineOp new_combine =
        rewriter.Build<pir::CombineOp>(new_combine_inputs);
    rewriter.ReplaceAllUsesWith(pre_op.out(), new_combine.out());
    rewriter.EraseOp(pre_op);

    // Create new concat
    auto concat_info =
        ctx->GetRegisteredOpInfo(paddle::onednn::dialect::ConcatOp::name());
    if (!concat_info) return false;

    std::vector<pir::Type> op_item_inner_output_types;
    auto type = op->result_type(0);
    pir::Type new_type =
        create_type<pir::DenseTensorType, paddle::dialect::DenseTensorType>(
            type, pir::BFloat16Type::get(ctx), ctx);
    op_item_inner_output_types.push_back(new_type);

    paddle::onednn::dialect::ConcatOp new_concat =
        rewriter
            .Build({new_combine.out(), op.axis()},
                   op_attributes,
                   op_item_inner_output_types,
                   concat_info)
            ->dyn_cast<paddle::onednn::dialect::ConcatOp>();

    // Insert dequant op under concat
    std::unordered_map<std::string, pir::Attribute> dq_attributes;
    dq_attributes["scale"] = rewriter.float_attr(1.0f);
    dq_attributes["shift"] = rewriter.float_attr(0.0f);
    paddle::onednn::dialect::DequantizeOp dequant_op =
        rewriter.Build<paddle::onednn::dialect::DequantizeOp>(new_concat.out(),
                                                              dq_attributes);

    rewriter.ReplaceAllUsesWith(op.out(), dequant_op.output());
    rewriter.EraseOp(op);
    return true;
  }
};

class SplitSliceBf16QuantizePattern
    : public pir::OpRewritePattern<paddle::onednn::dialect::SplitOp> {
 public:
  using pir::OpRewritePattern<
      paddle::onednn::dialect::SplitOp>::OpRewritePattern;
  bool MatchAndRewrite(
      paddle::onednn::dialect::SplitOp op,
      pir::PatternRewriter &rewriter) const override {  // NOLINT
    /**
     * In op_translator, all op with vector as input need add SliceOp
     * The output op should be builtin.slice to deal vector.
     * split(out:vector) -> slice(in: vector, out: dense tensor) -> dequant
     *                   split
     *  vec[tensor<3x6x6xbf16>,tensor<3x6x6xbf16>]
     *               |                  |
     *        builtin.slice         builtin.slice
     *     tensor<3x6x6xbf16>     tensor<3x6x6xbf16>
     *            |                     |
     *         other op               other op
     */

    auto next_op_list = pir::GetUseOpsForOutput(op, 0);
    for (auto i = 0; i < static_cast<int>(next_op_list.size()); i++) {
      pir::SliceOp next_op = (next_op_list[i].first)->dyn_cast<pir::SliceOp>();
      if (!next_op) {
        return false;
      }
    }

    paddle::onednn::dialect::QuantizeOp pre_op =
        pir::GetDefiningOpForInput(op, 0)
            ->dyn_cast<paddle::onednn::dialect::QuantizeOp>();
    if (pre_op) return false;

    auto op_attributes = op->attributes();
    auto onednn_data_type = op_attributes.at("mkldnn_data_type")
                                .dyn_cast<pir::StrAttribute>()
                                .AsString();
    if (onednn_data_type != "bfloat16") return false;

    pir::IrContext *ctx = rewriter.ir_context();

    std::unordered_map<std::string, pir::Attribute> q_attributes;
    q_attributes["scale"] = rewriter.float_attr(1.0f);
    q_attributes["shift"] = rewriter.float_attr(0.0f);
    q_attributes["is_negative_input"] = rewriter.bool_attr(false);
    q_attributes["output_format"] = rewriter.str_attr("NCHW");
    q_attributes["bfloat16"] = rewriter.bool_attr(true);

    // Insert quantize before split
    pir::Value split_input = op.x();
    auto type = op->result_type(0);
    if (!type.isa<pir::VectorType>()) {
      return false;
    }
    paddle::onednn::dialect::QuantizeOp quant_op =
        rewriter.Build<paddle::onednn::dialect::QuantizeOp>(split_input,
                                                            q_attributes);
    auto vec_type = type.dyn_cast<pir::VectorType>();
    auto quantize_type_ = quant_op->result_type(0);
    pir::Type new_type_quantize =
        create_type<pir::DenseTensorType, paddle::dialect::DenseTensorType>(
            quantize_type_, pir::BFloat16Type::get(ctx), ctx);

    quant_op->result(0).set_type(new_type_quantize);

    auto output_num = vec_type.size();
    std::vector<pir::Type> results_type(output_num);
    for (size_t idx = 0; idx < output_num; ++idx) {
      auto dense_type =
          vec_type[idx].dyn_cast<paddle::dialect::DenseTensorType>();
      pir::Type new_type = create_type<paddle::dialect::DenseTensorType,
                                       paddle::dialect::DenseTensorType>(
          dense_type, pir::BFloat16Type::get(ctx), ctx);
      results_type[idx] = new_type;
    }

    pir::Value new_split_input = quant_op.output();
    auto split_info =
        ctx->GetRegisteredOpInfo(paddle::onednn::dialect::SplitOp::name());

    std::vector<pir::Type> split_op_output_types;
    split_op_output_types.push_back(type);
    // Only can use this build func to build split for getting vector output
    pir::Operation *split_op = rewriter.Build(
        std::vector<pir::Value>{
            new_split_input, op->operand_source(1), op->operand_source(2)},
        op_attributes,
        split_op_output_types,
        split_info);
    auto new_vec_type =
        pir::VectorType::get(rewriter.ir_context(), results_type);
    split_op->result(0).set_type(new_vec_type);

    for (auto i = 0; i < static_cast<int>(next_op_list.size()); i++) {
      pir::SliceOp next_op = (next_op_list[i].first)->dyn_cast<pir::SliceOp>();

      auto index =
          next_op->attribute("index").dyn_cast<pir::Int32Attribute>().data();
      auto input_type = next_op->operand(0).type().dyn_cast<pir::VectorType>();
      auto new_type_ = input_type[index];
      pir::Type new_type_slice =
          create_type<pir::DenseTensorType, paddle::dialect::DenseTensorType>(
              new_type_, pir::BFloat16Type::get(ctx), ctx);

      auto slice_info = ctx->GetRegisteredOpInfo(pir::SliceOp::name());

      std::vector<pir::Type> op_item_inner_output_types;
      op_item_inner_output_types.push_back(new_type_slice);

      auto slice_attributes = next_op->attributes();
      pir::Operation *new_slice = rewriter
                                      .Build({split_op->result(0)},
                                             slice_attributes,
                                             op_item_inner_output_types,
                                             slice_info)
                                      ->dyn_cast<pir::SliceOp>();

      std::unordered_map<std::string, pir::Attribute> dq_attributes;
      dq_attributes["scale"] = rewriter.float_attr(1.0f);
      dq_attributes["shift"] = rewriter.float_attr(0.0f);
      paddle::onednn::dialect::DequantizeOp dequant_op =
          rewriter.Build<paddle::onednn::dialect::DequantizeOp>(
              new_slice->result(0), dq_attributes);

      rewriter.ReplaceAllUsesWith(next_op->result(0), dequant_op->result(0));
      rewriter.EraseOp(next_op);
    }
    rewriter.EraseOp(op);
    return true;
  }
};

class SplitdoubleBf16QuantizePattern
    : public pir::OpRewritePattern<paddle::onednn::dialect::SplitOp> {
 public:
  using pir::OpRewritePattern<
      paddle::onednn::dialect::SplitOp>::OpRewritePattern;
  bool MatchAndRewrite(
      paddle::onednn::dialect::SplitOp op,
      pir::PatternRewriter &rewriter) const override {  // NOLINT
    // The output op should be builtin.split to deal vector.
    // split(out:vector) -> split(in: vector, out: vector tensor) -> dequant
    /**
     *                   split
     *  vec[tensor<3x6x6xbf16>,tensor<3x6x6xbf16>]
     *                    |
     *              builtin.split
     *  tensor<3x6x6xbf16>   tensor<3x6x6xbf16>
     *          |                   |
     *       other op
     */
    auto next_op_list = pir::GetUseOpsForOutput(op, 0);
    for (auto i = 0; i < static_cast<int>(next_op_list.size()); i++) {
      pir::SplitOp next_op = (next_op_list[i].first)->dyn_cast<pir::SplitOp>();
      if (!next_op) {
        return false;
      }
    }

    paddle::onednn::dialect::QuantizeOp pre_op =
        pir::GetDefiningOpForInput(op, 0)
            ->dyn_cast<paddle::onednn::dialect::QuantizeOp>();
    if (pre_op) return false;

    auto op_attributes = op->attributes();
    auto onednn_data_type = op_attributes.at("mkldnn_data_type")
                                .dyn_cast<pir::StrAttribute>()
                                .AsString();
    if (onednn_data_type != "bfloat16") return false;

    pir::IrContext *ctx = rewriter.ir_context();

    std::unordered_map<std::string, pir::Attribute> q_attributes;
    q_attributes["scale"] = rewriter.float_attr(1.0f);
    q_attributes["shift"] = rewriter.float_attr(0.0f);
    q_attributes["is_negative_input"] = rewriter.bool_attr(false);
    q_attributes["output_format"] = rewriter.str_attr("NCHW");
    q_attributes["bfloat16"] = rewriter.bool_attr(true);

    // Insert quantize before split
    auto type = op->result_type(0);
    if (!type.isa<pir::VectorType>()) {
      return false;
    }
    pir::Value split_input = op.x();
    paddle::onednn::dialect::QuantizeOp quant_op =
        rewriter.Build<paddle::onednn::dialect::QuantizeOp>(split_input,
                                                            q_attributes);
    auto vec_type = type.dyn_cast<pir::VectorType>();
    auto quantize_type_ = quant_op->result_type(0);
    pir::Type new_type_quantize =
        create_type<pir::DenseTensorType, paddle::dialect::DenseTensorType>(
            quantize_type_, pir::BFloat16Type::get(ctx), ctx);

    quant_op->result(0).set_type(new_type_quantize);

    auto output_num = vec_type.size();
    std::vector<pir::Type> results_type(output_num);
    for (size_t idx = 0; idx < output_num; ++idx) {
      auto dense_type =
          vec_type[idx].dyn_cast<paddle::dialect::DenseTensorType>();
      auto new_type = paddle::dialect::DenseTensorType::get(
          rewriter.ir_context(),
          paddle::dialect::TransToIrDataType(phi::DataType::BFLOAT16,
                                             rewriter.ir_context()),
          dense_type.dims(),
          dense_type.data_layout(),
          dense_type.lod(),
          dense_type.offset());
      results_type[idx] = new_type;
    }

    pir::Value new_split_input = quant_op.output();
    auto split_info =
        ctx->GetRegisteredOpInfo(paddle::onednn::dialect::SplitOp::name());

    std::vector<pir::Type> split_op_output_types;
    split_op_output_types.push_back(type);

    pir::Operation *split_op = rewriter.Build(
        std::vector<pir::Value>{
            new_split_input, op->operand_source(1), op->operand_source(2)},
        op_attributes,
        split_op_output_types,
        split_info);
    auto new_vec_type =
        pir::VectorType::get(rewriter.ir_context(), results_type);
    split_op->result(0).set_type(new_vec_type);

    for (size_t i = 0; i < next_op_list.size(); i++) {
      pir::SplitOp next_op = (next_op_list[i].first)->dyn_cast<pir::SplitOp>();

      pir::SplitOp next_split_op =
          rewriter.Build<pir::SplitOp>(split_op->result(0));
      auto next_op_outputs = next_split_op.outputs();
      for (size_t idx = 0; idx < next_op_outputs.size(); idx++) {
        if (!next_op->result(idx).HasOneUse()) {
          // Some output in vector not use anymore, no need add dq
          rewriter.ReplaceAllUsesWith(next_op->result(idx),
                                      next_split_op->result(idx));
          continue;
        }
        std::unordered_map<std::string, pir::Attribute> dq_attributes;
        dq_attributes["scale"] = rewriter.float_attr(1.0f);
        dq_attributes["shift"] = rewriter.float_attr(0.0f);
        paddle::onednn::dialect::DequantizeOp dequant_op =
            rewriter.Build<paddle::onednn::dialect::DequantizeOp>(
                next_split_op->result(idx), dq_attributes);

        rewriter.ReplaceAllUsesWith(next_op->result(idx),
                                    dequant_op->result(0));
      }
      rewriter.EraseOp(next_op);
    }
    rewriter.EraseOp(op);
    return true;
  }
};

class CPUSpecialOpsBf16Pass : public pir::PatternRewritePass {
 public:
  CPUSpecialOpsBf16Pass()
      : pir::PatternRewritePass("cpu_special_ops_bf16_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    uint32_t benefit = 100;

    auto cast_bf16_pattern =
        std::make_unique<CastBf16Pattern<paddle::onednn::dialect::CastOp>>(
            context,
            benefit--,
            std::vector<std::string>{
                paddle::onednn::dialect::QuantizeOp::name(),
                paddle::onednn::dialect::DequantizeOp::name(),
            });
    ps.Add(std::move(cast_bf16_pattern));

    auto cast_bf16_pattern_2 =
        std::make_unique<CastBf16Pattern<paddle::onednn::dialect::Cast_Op>>(
            context,
            benefit--,
            std::vector<std::string>{
                paddle::onednn::dialect::QuantizeOp::name(),
                paddle::onednn::dialect::DequantizeOp::name(),
            });
    ps.Add(std::move(cast_bf16_pattern_2));

    auto concat_bf16_quant_pattern =
        std::make_unique<ConcatBf16QuantizePattern>(
            context,
            benefit--,
            std::vector<std::string>{
                paddle::onednn::dialect::QuantizeOp::name(),
                paddle::onednn::dialect::DequantizeOp::name(),
            });
    ps.Add(std::move(concat_bf16_quant_pattern));

    auto split_bf16_quant_pattern =
        std::make_unique<SplitSliceBf16QuantizePattern>(
            context, benefit--, std::vector<std::string>{});
    ps.Add(std::move(split_bf16_quant_pattern));

    auto split_double_bf16_quant_pattern =
        std::make_unique<SplitdoubleBf16QuantizePattern>(
            context, benefit--, std::vector<std::string>{});
    ps.Add(std::move(split_double_bf16_quant_pattern));

    return ps;
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateCPUSpecialOpsBf16Pass() {
  return std::make_unique<CPUSpecialOpsBf16Pass>();
}

}  // namespace pir

REGISTER_IR_PASS(cpu_special_ops_bf16_pass, CPUSpecialOpsBf16Pass);
