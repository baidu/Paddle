// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/pir/transforms/gpu/fused_gemm_epilogue_pass.h"

#include "paddle/fluid/pir/dialect/operator/interface/infer_symbolic_shape/infer_sym_utils.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"
#include "paddle/fluid/pir/utils/general_functions.h"

#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace {

//  %2 = pd_op.matmul( %0, %1 )
//  %4 = pd_op.add( %2, %3 )
//  fused to
//  %4, %5 = pd_op.fused_gemm_epilogue( %0, %1, %3 )
class FusedLinearPattern
    : public pir::OpRewritePattern<paddle::dialect::MatmulOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::MatmulOp>::OpRewritePattern;

  bool MatchAndRewrite(paddle::dialect::MatmulOp matmul,
                       pir::PatternRewriter &rewriter) const override {
    auto matmul_out = matmul->result(0);
    // The result of matmul can only be uniquely used by an add OP.
    if (matmul_out.use_count() != 1) {
      return false;
    }
    if (!matmul_out.use_begin()->owner()->dyn_cast<paddle::dialect::AddOp>()) {
      return false;
    }
    auto add =
        matmul_out.use_begin()->owner()->dyn_cast<paddle::dialect::AddOp>();

    // The data rank of matmul should be >= 2.
    // The weight rank of matmul should be = 2.
    // The bias rank of add should be = 1.
    if (pir::GetShapeFromValue(matmul->operand_source(0)).size() < 2 ||
        pir::GetShapeFromValue(matmul->operand_source(1)).size() != 2 ||
        pir::GetShapeFromValue(add->operand_source(1)).size() != 1) {
      return false;
    }

    pir::AttributeMap attr_map;
    attr_map.emplace("trans_x", matmul->attribute("transpose_x"));
    attr_map.emplace("trans_y", matmul->attribute("transpose_y"));
    attr_map.emplace(
        "activation",
        pir::StrAttribute::get(pir::IrContext::Instance(), "none"));

    rewriter.SetInsertionPointAfter(add);

    auto fuse_gemm = rewriter.Build<paddle::dialect::FusedGemmEpilogueOp>(
        matmul->operand_source(0),
        matmul->operand_source(1),
        add->operand_source(1),
        attr_map);

    if (matmul->HasAttribute("op_role")) {
      fuse_gemm->set_attribute("op_role", matmul->attribute("op_role"));
    }
    if (matmul->HasAttribute("chunk_id")) {
      fuse_gemm->set_attribute("chunk_id", matmul->attribute("chunk_id"));
    }

    rewriter.ReplaceAllUsesWith(add->result(0), fuse_gemm.result(0));
    rewriter.ReplaceAllUsesWith(matmul_out, fuse_gemm.result(0));

    rewriter.EraseOp(add);
    rewriter.EraseOp(matmul);
    return true;
  }
};

//  %3, %4 = pd_op.add_grad( %0, %1，%2 )
//  %7, %8 = pd_op.matmul_grad( %5, %6, %3)
//  fused to
//  %7, %8, %4 = pd_op.fused_gemm_epilogue_grad( %5, %6, none, %2)
class FusedLinearGradPattern
    : public pir::OpRewritePattern<paddle::dialect::MatmulGradOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::MatmulGradOp>::OpRewritePattern;

  bool MatchAndRewrite(paddle::dialect::MatmulGradOp matmul_grad,
                       pir::PatternRewriter &rewriter) const override {
    auto matmul_grad_out = matmul_grad->operand_source(2);

    paddle::dialect::AddGradOp add_grad;
    if (add_grad = matmul_grad_out.defining_op()
                       ->dyn_cast<paddle::dialect::AddGradOp>()) {
      if (matmul_grad_out != add_grad->result(0)) {
        return false;
      }
    } else {
      return false;
    }
    // The data gradient of add_grad can only be uniquely used by a matmul_grad
    // OP.
    if (add_grad.result(0).use_count() != 1) {
      return false;
    }

    // The data rank of matmul_grad should be >= 2.
    // The weight rank of matmul_grad should be = 2.
    // The bias rank of add_grad should be = 1.
    if (pir::GetShapeFromValue(matmul_grad->operand_source(0)).size() < 2 ||
        pir::GetShapeFromValue(matmul_grad->operand_source(1)).size() != 2 ||
        pir::GetShapeFromValue(add_grad->operand_source(1)).size() != 1) {
      return false;
    }

    pir::AttributeMap attr_map;
    attr_map.emplace("trans_x", matmul_grad.attribute("transpose_x"));
    attr_map.emplace("trans_y", matmul_grad.attribute("transpose_y"));
    attr_map.emplace(
        "activation_grad",
        pir::StrAttribute::get(pir::IrContext::Instance(), "none"));

    rewriter.SetInsertionPointAfter(add_grad);
    auto fuse_gemm_grad =
        rewriter.Build<paddle::dialect::FusedGemmEpilogueGradOp>(
            matmul_grad->operand_source(0),
            matmul_grad->operand_source(1),
            pir::Value(),
            add_grad->operand_source(2),
            attr_map);

    if (matmul_grad->HasAttribute("op_role")) {
      fuse_gemm_grad->set_attribute("op_role",
                                    matmul_grad->attribute("op_role"));
    }
    if (matmul_grad->HasAttribute("chunk_id")) {
      fuse_gemm_grad->set_attribute("chunk_id",
                                    matmul_grad->attribute("chunk_id"));
    }

    rewriter.ReplaceAllUsesWith(matmul_grad.result(0),
                                fuse_gemm_grad.result(0));
    rewriter.ReplaceAllUsesWith(matmul_grad.result(1),
                                fuse_gemm_grad.result(1));
    rewriter.ReplaceAllUsesWith(add_grad.result(1), fuse_gemm_grad.result(2));

    rewriter.EraseOp(matmul_grad);
    rewriter.EraseOp(add_grad);

    return true;
  }
};

//  %1 = pd_op.sum( %0, %2)
//  %3 = pd_op.assign( %0 )
//  %4, %5 = pd_op.matmul_grad( %6, %7, %3)
//  fused to
//  %4, %5 = pd_op.fused_gemm_epilogue_grad( %6, %7, none, %0)
class FusedLinearGradSinglePattern
    : public pir::OpRewritePattern<paddle::dialect::MatmulGradOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::MatmulGradOp>::OpRewritePattern;

  bool MatchAndRewrite(paddle::dialect::MatmulGradOp matmul_grad,
                       pir::PatternRewriter &rewriter) const override {
    auto dout = matmul_grad->operand_source(2);

    if (pir::GetShapeFromValue(matmul_grad->operand_source(1)).size() != 2) {
      return false;
    }

    if (auto assign_op =
            dout.defining_op()->dyn_cast<paddle::dialect::AssignOp>()) {
      dout = assign_op->operand_source(0);
    }

    bool can_fuse_sum = false;
    pir::Value sum_output;
    pir::Value sum_input;
    for (auto user_it = dout.use_begin(); user_it != dout.use_end();
         ++user_it) {
      if (!user_it->owner()) {
        continue;
      }
      if (auto sum_op = user_it->owner()->dyn_cast<paddle::dialect::SumOp>()) {
        sum_input = sum_op->operand_source(0);
        int64_t input_rank = -1;
        if (sum_input.type() &&
            sum_input.type().isa<paddle::dialect::DenseTensorType>()) {
          input_rank = sum_input.type()
                           .dyn_cast<paddle::dialect::DenseTensorType>()
                           .dims()
                           .size();
        }
        if (input_rank == -1) {
          break;
        }

        if (sum_op->operand_source(1)
                .defining_op()
                ->isa<paddle::dialect::FullIntArrayOp>()) {
          auto axis_full_op = sum_op->operand_source(1)
                                  .defining_op()
                                  ->dyn_cast<paddle::dialect::FullIntArrayOp>();
          const std::vector<int64_t> axis =
              paddle::dialect::details::GetVectorAttr<int64_t>(axis_full_op,
                                                               "value");

          std::set<int64_t> reduce_set;
          for (auto d : axis) {
            if (d < 0) {
              d += input_rank;
            }
            reduce_set.insert(d);
          }
          if ((reduce_set.size() == static_cast<size_t>(input_rank - 1)) &&
              (!reduce_set.count(input_rank - 1))) {
            can_fuse_sum = true;
          }
        }

        sum_output = sum_op->result(0);
        rewriter.SetInsertionPointAfter(sum_op);
        break;
      }
    }

    if (!can_fuse_sum) {
      return false;
    }

    pir::AttributeMap attr_map;
    attr_map.emplace("trans_x", matmul_grad.attribute("transpose_x"));
    attr_map.emplace("trans_y", matmul_grad.attribute("transpose_y"));
    attr_map.emplace(
        "activation_grad",
        pir::StrAttribute::get(pir::IrContext::Instance(), "none"));

    auto fuse_gemm = rewriter.Build<paddle::dialect::FusedGemmEpilogueGradOp>(
        matmul_grad->operand_source(0),
        matmul_grad->operand_source(1),
        pir::Value(),
        sum_input,
        attr_map);

    rewriter.ReplaceAllUsesWith(matmul_grad.result(0), fuse_gemm.result(0));
    rewriter.ReplaceAllUsesWith(matmul_grad.result(1), fuse_gemm.result(1));
    rewriter.ReplaceAllUsesWith(sum_output, fuse_gemm.result(2));

    rewriter.EraseOp(matmul_grad);

    return true;
  }
};

class FusedLinearGeluPattern : public paddle::drr::DrrPatternBase {
 public:
  std::string name() const override { return "FusedLinearGeluPattern"; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    // Source pattern
    const auto &fused_gemm_epilogue =
        pat.Op(paddle::dialect::FusedGemmEpilogueOp::name(),
               {{{"trans_x", pat.Attr("trans_x")},
                 {"trans_y", pat.Attr("trans_y")},
                 {"activation", pat.Attr("act")}}});
    const auto &gelu = pat.Op(paddle::dialect::GeluOp::name());
    fused_gemm_epilogue(
        {&pat.Tensor("x"), &pat.Tensor("w"), &pat.Tensor("bias")},
        {&pat.Tensor("fuse_out"), &pat.Tensor("reserve_space")});
    pat.Tensor("out") = gelu(pat.Tensor("fuse_out"));

    // Constrains the activation is none
    pat.AddConstraint([&](const paddle::drr::MatchContext &match_ctx) {
      return (match_ctx.Attr<std::string>("act") == "none");
    });

    // Result pattern
    paddle::drr::ResultPattern res = pat.ResultPattern();
    const auto &fused_gemm_epilogue_gelu =
        res.Op(paddle::dialect::FusedGemmEpilogueOp::name(),
               {{{"trans_x", pat.Attr("trans_x")},
                 {"trans_y", pat.Attr("trans_y")},
                 {"activation", res.StrAttr("gelu")}}});
    fused_gemm_epilogue_gelu(
        {&res.Tensor("x"), &res.Tensor("w"), &res.Tensor("bias")},
        {&res.Tensor("out"), &res.Tensor("reserve_space")});
  }
};

class FusedLinearReluPattern : public paddle::drr::DrrPatternBase {
 public:
  std::string name() const override { return "FusedLinearReluPattern"; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    // Source pattern
    const auto &fused_gemm_epilogue =
        pat.Op(paddle::dialect::FusedGemmEpilogueOp::name(),
               {{{"trans_x", pat.Attr("trans_x")},
                 {"trans_y", pat.Attr("trans_y")},
                 {"activation", pat.Attr("act")}}});
    const auto &relu = pat.Op(paddle::dialect::ReluOp::name());
    fused_gemm_epilogue(
        {&pat.Tensor("x"), &pat.Tensor("w"), &pat.Tensor("bias")},
        {&pat.Tensor("fuse_out"), &pat.Tensor("reserve_space")});
    pat.Tensor("out") = relu(pat.Tensor("fuse_out"));

    // Constrains the activation is none
    pat.AddConstraint([&](const paddle::drr::MatchContext &match_ctx) {
      return (match_ctx.Attr<std::string>("act") == "none");
    });

    // Result pattern
    paddle::drr::ResultPattern res = pat.ResultPattern();
    const auto &fused_gemm_epilogue_relu =
        res.Op(paddle::dialect::FusedGemmEpilogueOp::name(),
               {{{"trans_x", pat.Attr("trans_x")},
                 {"trans_y", pat.Attr("trans_y")},
                 {"activation", res.StrAttr("relu")}}});
    fused_gemm_epilogue_relu(
        {&res.Tensor("x"), &res.Tensor("w"), &res.Tensor("bias")},
        {&res.Tensor("out"), &res.Tensor("reserve_space")});
  }
};

class FusedLinearGeluGradPattern : public paddle::drr::DrrPatternBase {
 public:
  std::string name() const override { return "FusedLinearGeluGradPattern"; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &fused_gemm_epilogue =
        pat.Op(paddle::dialect::FusedGemmEpilogueOp::name(),
               {{{"trans_x", pat.Attr("trans_x1")},
                 {"trans_y", pat.Attr("trans_y1")},
                 {"activation", pat.Attr("act1")}}});
    const auto &fused_gemm_epilogue_grad1 =
        pat.Op(paddle::dialect::FusedGemmEpilogueGradOp::name(),
               {{{"trans_x", pat.Attr("trans_x2")},
                 {"trans_y", pat.Attr("trans_y2")},
                 {"activation_grad", pat.Attr("act2")}}});
    fused_gemm_epilogue(
        {&pat.Tensor("x"), &pat.Tensor("w"), &pat.Tensor("bias")},
        {&pat.Tensor("fuse_out"), &pat.Tensor("reserve_space")});
    pat.Tensor("out") =
        pat.Op(paddle::dialect::GeluOp::name())(pat.Tensor("fuse_out"));

    fused_gemm_epilogue_grad1({&pat.Tensor("x1"),
                               &pat.Tensor("w1"),
                               &pat.Tensor("reserve_space1"),
                               &pat.Tensor("out_grad")},
                              {&pat.Tensor("x1_grad"),
                               &pat.Tensor("w1_grad"),
                               &pat.Tensor("bias1_grad")});
    pat.Tensor("gelu_dx") = pat.Op(paddle::dialect::GeluGradOp::name())(
        pat.Tensor("fuse_out"), pat.Tensor("x1_grad"));

    pat.AddConstraint([&](const paddle::drr::MatchContext &match_ctx) {
      return match_ctx.Attr<std::string>("act1") == "none" &&
             match_ctx.Attr<std::string>("act2") == "none";
    });

    paddle::drr::ResultPattern res = pat.ResultPattern();
    const auto &fused_gemm_epilogue_new =
        res.Op(paddle::dialect::FusedGemmEpilogueOp::name(),
               {{{"trans_x", pat.Attr("trans_x1")},
                 {"trans_y", pat.Attr("trans_y1")},
                 {"activation", res.StrAttr("gelu")}}});
    const auto &fused_gemm_epilogue_grad_new =
        res.Op(paddle::dialect::FusedGemmEpilogueGradOp::name(),
               {{{"trans_x", pat.Attr("trans_x2")},
                 {"trans_y", pat.Attr("trans_y2")},
                 {"activation_grad", res.StrAttr("gelu_grad")}}});
    fused_gemm_epilogue_new(
        {&res.Tensor("x"), &res.Tensor("w"), &res.Tensor("bias")},
        {&res.Tensor("out"), &res.Tensor("reserve_space2")});
    fused_gemm_epilogue_grad_new({&res.Tensor("x1"),
                                  &res.Tensor("w1"),
                                  &res.Tensor("reserve_space2"),
                                  &res.Tensor("out_grad")},
                                 {&res.Tensor("gelu_dx"),
                                  &res.Tensor("w1_grad"),
                                  &res.Tensor("bias1_grad")});
  }
};

class FusedLinearReluGradPattern : public paddle::drr::DrrPatternBase {
 public:
  std::string name() const override { return "FusedLinearReluGradPattern"; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &fused_gemm_epilogue =
        pat.Op(paddle::dialect::FusedGemmEpilogueOp::name(),
               {{{"trans_x", pat.Attr("trans_x1")},
                 {"trans_y", pat.Attr("trans_y1")},
                 {"activation", pat.Attr("act1")}}});
    const auto &fused_gemm_epilogue_grad =
        pat.Op(paddle::dialect::FusedGemmEpilogueGradOp::name(),
               {{{"trans_x", pat.Attr("trans_x2")},
                 {"trans_y", pat.Attr("trans_y2")},
                 {"activation_grad", pat.Attr("act2")}}});
    const auto &fused_gemm_epilogue_grad1 =
        pat.Op(paddle::dialect::FusedGemmEpilogueGradOp::name(),
               {{{"trans_x", pat.Attr("trans_x3")},
                 {"trans_y", pat.Attr("trans_y3")},
                 {"activation_grad", pat.Attr("act3")}}});

    fused_gemm_epilogue(
        {&pat.Tensor("x"), &pat.Tensor("w"), &pat.Tensor("bias")},
        {&pat.Tensor("fuse_out"), &pat.Tensor("reserve_space")});
    fused_gemm_epilogue_grad1({&pat.Tensor("x1"),
                               &pat.Tensor("w1"),
                               &pat.Tensor("reserve_space2"),
                               &pat.Tensor("out_grad")},
                              {&pat.Tensor("x1_grad"),
                               &pat.Tensor("w1_grad"),
                               &pat.Tensor("bias1_grad")});

    pat.Tensor("relu_dx") = pat.Op(paddle::dialect::ReluGradOp::name())(
        pat.Tensor("x1"), pat.Tensor("x1_grad"));
    fused_gemm_epilogue_grad({&pat.Tensor("x"),
                              &pat.Tensor("w"),
                              &pat.Tensor("reserve_space1"),
                              &pat.Tensor("relu_dx")},
                             {&pat.Tensor("x_grad"),
                              &pat.Tensor("w_grad"),
                              &pat.Tensor("bias_grad")});

    pat.AddConstraint([&](const paddle::drr::MatchContext &match_ctx) {
      return match_ctx.Attr<std::string>("act1") == "relu" &&
             match_ctx.Attr<std::string>("act3") == "none";
    });

    paddle::drr::ResultPattern res = pat.ResultPattern();
    const auto &res_fused_gemm_epilogue_grad1 =
        res.Op(paddle::dialect::FusedGemmEpilogueGradOp::name(),
               {{{"trans_x", pat.Attr("trans_x3")},
                 {"trans_y", pat.Attr("trans_y3")},
                 {"activation_grad", res.StrAttr("relu_grad")}}});

    res_fused_gemm_epilogue_grad1({&res.Tensor("x1"),
                                   &res.Tensor("w1"),
                                   &res.Tensor("reserve_space"),
                                   &res.Tensor("out_grad")},
                                  {&res.Tensor("relu_dx"),
                                   &res.Tensor("w1_grad"),
                                   &res.Tensor("bias1_grad")});
  }
};

class FusedGemmEpiloguePass : public pir::PatternRewritePass {
 public:
  FusedGemmEpiloguePass()
      : pir::PatternRewritePass("fused_gemm_epilogue_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    ps.Add<FusedLinearPattern>(context);
    ps.Add<FusedLinearGradPattern>(context);
    ps.Add(paddle::drr::Create<FusedLinearGeluPattern>(context));
    ps.Add(paddle::drr::Create<FusedLinearReluPattern>(context));
    ps.Add(paddle::drr::Create<FusedLinearGeluGradPattern>(context));
    ps.Add(paddle::drr::Create<FusedLinearReluGradPattern>(context));
    ps.Add<FusedLinearGradSinglePattern>(context);

    return ps;
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateFusedGemmEpiloguePass() {
  return std::make_unique<FusedGemmEpiloguePass>();
}

}  // namespace pir

REGISTER_IR_PASS(fused_gemm_epilogue_pass, FusedGemmEpiloguePass);
