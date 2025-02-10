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

#include "paddle/fluid/pir/transforms/tensorrt/trt_op_marker_pass.h"
#include <glog/logging.h>
#include <bitset>
#include <vector>
#include "paddle/fluid/inference/tensorrt/helper.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/utils/general_functions.h"

#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/fluid/platform/tensorrt/helper.h"
#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace {

inline auto kCanRunTrtAttr = paddle::dialect::kCanRunTrtAttr;

#define DEFINE_GENERAL_PATTERN(OpName, OpType)                            \
  class OpName##OpPattern : public pir::OpRewritePattern<OpType> {        \
   public:                                                                \
    using pir::OpRewritePattern<OpType>::OpRewritePattern;                \
    bool MatchAndRewrite(OpType op,                                       \
                         pir::PatternRewriter &rewriter) const override { \
      if (op->HasAttribute(kCanRunTrtAttr) &&                             \
          op->attribute<pir::BoolAttribute>(kCanRunTrtAttr).data()) {     \
        return false;                                                     \
      }                                                                   \
      op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));        \
      return true;                                                        \
    }                                                                     \
  };

DEFINE_GENERAL_PATTERN(Matmul, paddle::dialect::MatmulOp)
DEFINE_GENERAL_PATTERN(BatchNorm, paddle::dialect::BatchNormOp)
DEFINE_GENERAL_PATTERN(BatchNorm_, paddle::dialect::BatchNorm_Op)
DEFINE_GENERAL_PATTERN(Softmax, paddle::dialect::SoftmaxOp)
DEFINE_GENERAL_PATTERN(Relu, paddle::dialect::ReluOp)
DEFINE_GENERAL_PATTERN(FullIntArray, paddle::dialect::FullIntArrayOp)
DEFINE_GENERAL_PATTERN(Reshape, paddle::dialect::ReshapeOp)
DEFINE_GENERAL_PATTERN(Dropout, paddle::dialect::DropoutOp)
DEFINE_GENERAL_PATTERN(Bmm, paddle::dialect::BmmOp)
DEFINE_GENERAL_PATTERN(Concat, paddle::dialect::ConcatOp)
DEFINE_GENERAL_PATTERN(Nonzero, paddle::dialect::NonzeroOp)
DEFINE_GENERAL_PATTERN(Gelu, paddle::dialect::GeluOp)
DEFINE_GENERAL_PATTERN(Relu6, paddle::dialect::Relu6Op)
DEFINE_GENERAL_PATTERN(Fused_gemm_epilogue,
                       paddle::dialect::FusedGemmEpilogueOp)
DEFINE_GENERAL_PATTERN(Layer_norm, paddle::dialect::LayerNormOp)
DEFINE_GENERAL_PATTERN(Add, paddle::dialect::AddOp)
DEFINE_GENERAL_PATTERN(Full, paddle::dialect::FullOp)
DEFINE_GENERAL_PATTERN(Silu, paddle::dialect::SiluOp)
DEFINE_GENERAL_PATTERN(Conv2d, paddle::dialect::Conv2dOp)
DEFINE_GENERAL_PATTERN(FusedConv2dAddAct, paddle::dialect::FusedConv2dAddActOp)
DEFINE_GENERAL_PATTERN(DepthwiseConv2d, paddle::dialect::DepthwiseConv2dOp)
DEFINE_GENERAL_PATTERN(Shape, paddle::dialect::ShapeOp)
DEFINE_GENERAL_PATTERN(Shape64, paddle::dialect::Shape64Op)
DEFINE_GENERAL_PATTERN(Expand, paddle::dialect::ExpandOp)
DEFINE_GENERAL_PATTERN(ExpandAs, paddle::dialect::ExpandAsOp)
DEFINE_GENERAL_PATTERN(Sigmoid, paddle::dialect::SigmoidOp)
DEFINE_GENERAL_PATTERN(Sqrt, paddle::dialect::SqrtOp)
DEFINE_GENERAL_PATTERN(Hardsigmoid, paddle::dialect::HardsigmoidOp)
DEFINE_GENERAL_PATTERN(Hardswish, paddle::dialect::HardswishOp)
DEFINE_GENERAL_PATTERN(Assign, paddle::dialect::AssignOp)
DEFINE_GENERAL_PATTERN(Tile, paddle::dialect::TileOp)
DEFINE_GENERAL_PATTERN(Share_Data, paddle::dialect::ShareDataOp)
DEFINE_GENERAL_PATTERN(AssignOut, paddle::dialect::AssignOut_Op)
DEFINE_GENERAL_PATTERN(Swish, paddle::dialect::SwishOp)
DEFINE_GENERAL_PATTERN(Log, paddle::dialect::LogOp)
DEFINE_GENERAL_PATTERN(Floor, paddle::dialect::FloorOp)
DEFINE_GENERAL_PATTERN(Roll, paddle::dialect::RollOp)
DEFINE_GENERAL_PATTERN(Elu, paddle::dialect::EluOp)
DEFINE_GENERAL_PATTERN(Selu, paddle::dialect::SeluOp)
DEFINE_GENERAL_PATTERN(Stanh, paddle::dialect::StanhOp)
DEFINE_GENERAL_PATTERN(Softplus, paddle::dialect::SoftplusOp)
DEFINE_GENERAL_PATTERN(ThresholdedRelu, paddle::dialect::ThresholdedReluOp)
DEFINE_GENERAL_PATTERN(Flip, paddle::dialect::FlipOp)
DEFINE_GENERAL_PATTERN(Mish, paddle::dialect::MishOp)
DEFINE_GENERAL_PATTERN(AssignValue, paddle::dialect::AssignValueOp)
DEFINE_GENERAL_PATTERN(AssignValue_, paddle::dialect::AssignValue_Op)
DEFINE_GENERAL_PATTERN(LeakyRelu, paddle::dialect::LeakyReluOp)
DEFINE_GENERAL_PATTERN(LeakyRelu_, paddle::dialect::LeakyRelu_Op)
DEFINE_GENERAL_PATTERN(Anchor_Generator, paddle::dialect::AnchorGeneratorOp)
DEFINE_GENERAL_PATTERN(Exp, paddle::dialect::ExpOp)
DEFINE_GENERAL_PATTERN(Abs, paddle::dialect::AbsOp)
DEFINE_GENERAL_PATTERN(Abs_, paddle::dialect::Abs_Op)
DEFINE_GENERAL_PATTERN(Sin, paddle::dialect::SinOp)
DEFINE_GENERAL_PATTERN(Logsigmoid, paddle::dialect::LogsigmoidOp)
DEFINE_GENERAL_PATTERN(Embedding, paddle::dialect::EmbeddingOp)
DEFINE_GENERAL_PATTERN(Unbind, paddle::dialect::UnbindOp)
DEFINE_GENERAL_PATTERN(Cos, paddle::dialect::CosOp)
DEFINE_GENERAL_PATTERN(Sinh, paddle::dialect::SinhOp)
DEFINE_GENERAL_PATTERN(Cosh, paddle::dialect::CoshOp)
DEFINE_GENERAL_PATTERN(Asinh, paddle::dialect::AsinhOp)
DEFINE_GENERAL_PATTERN(Acosh, paddle::dialect::AcoshOp)
DEFINE_GENERAL_PATTERN(Atanh, paddle::dialect::AtanhOp)
DEFINE_GENERAL_PATTERN(Ceil, paddle::dialect::CeilOp)
DEFINE_GENERAL_PATTERN(Rsqrt, paddle::dialect::RsqrtOp)
DEFINE_GENERAL_PATTERN(Reciprocal, paddle::dialect::ReciprocalOp)
DEFINE_GENERAL_PATTERN(Erf, paddle::dialect::ErfOp)
DEFINE_GENERAL_PATTERN(Isnan, paddle::dialect::IsnanOp)
DEFINE_GENERAL_PATTERN(Sign, paddle::dialect::SignOp)
DEFINE_GENERAL_PATTERN(Round, paddle::dialect::RoundOp)
DEFINE_GENERAL_PATTERN(Numel, paddle::dialect::NumelOp)
DEFINE_GENERAL_PATTERN(Pool3d, paddle::dialect::Pool3dOp)
DEFINE_GENERAL_PATTERN(ShuffleChannel, paddle::dialect::ShuffleChannelOp)

#undef DEFINE_GENERAL_PATTERN

// Add ReduceCommonOpPattern base class to simplify code
template <typename OpType>
class ReduceCommonOpPattern : public pir::OpRewritePattern<OpType> {
 public:
  using pir::OpRewritePattern<OpType>::OpRewritePattern;
  bool MatchAndRewrite(OpType op,
                       pir::PatternRewriter &rewriter) const override {
    if (op->HasAttribute(kCanRunTrtAttr) &&
        op->template attribute<pir::BoolAttribute>(kCanRunTrtAttr).data()) {
      return false;
    }

    if (!op->HasAttribute("keepdim")) {
      VLOG(3) << "the max does not have attr keep_dim ";
      return false;
    }

    if constexpr (std::is_same_v<OpType, paddle::dialect::AnyOp> ||
                  std::is_same_v<OpType, paddle::dialect::AllOp>) {
      if (!op->HasAttribute("axis")) {
        VLOG(3) << "The axis attribute does not exist";
        return false;
      }
    }

    pir::Value x = op.operand_source(0);
    auto x_dtype = pir::GetDataTypeFromValue(x);
    if constexpr (std::is_same_v<OpType, paddle::dialect::AnyOp>) {
      if (!x_dtype.isa<pir::BoolType>()) {
        VLOG(3) << "any op input data type must be bool";
        return false;
      }
    } else if constexpr (std::is_same_v<OpType, paddle::dialect::AllOp>) {
      if (!x_dtype.isa<pir::BoolType>()) {
        VLOG(3) << "all op input data type must be bool";
        return false;
      }
    } else {
      if (!(x_dtype.isa<pir::Float32Type>() ||
            x_dtype.isa<pir::Float64Type>() || x_dtype.isa<pir::Int32Type>() ||
            x_dtype.isa<pir::Int64Type>())) {
        if constexpr (std::is_same_v<OpType, paddle::dialect::MinOp>) {
          VLOG(3) << "min input data type must be int32 or int64 or "
                     "float32 or "
                     "float64";
        } else if constexpr (std::is_same_v<OpType, paddle::dialect::MaxOp>) {
          VLOG(3) << "max input data type must be int32 or int64 or "
                     "float32 or "
                     "float64";
        } else if constexpr (std::is_same_v<OpType, paddle::dialect::MeanOp>) {
          VLOG(3) << "mean input data type must be int32 or int64 or "
                     "float32 or "
                     "float64";
        }
        return false;
      }
    }
    op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));
    return true;
  }
};

// use type aliases to simplify usage
using MinOpPattern = ReduceCommonOpPattern<paddle::dialect::MinOp>;
using MaxOpPattern = ReduceCommonOpPattern<paddle::dialect::MaxOp>;
using MeanOpPattern = ReduceCommonOpPattern<paddle::dialect::MeanOp>;
using AnyOpPattern = ReduceCommonOpPattern<paddle::dialect::AnyOp>;
using AllOpPattern = ReduceCommonOpPattern<paddle::dialect::AllOp>;
using SumOpPattern = ReduceCommonOpPattern<paddle::dialect::SumOp>;

// Add ElementwiseCommonOpPattern base class to simplify code
template <typename OpType>
class ElementwiseCommonOpPattern : public pir::OpRewritePattern<OpType> {
 public:
  using pir::OpRewritePattern<OpType>::OpRewritePattern;
  bool MatchAndRewrite(OpType op,
                       pir::PatternRewriter &rewriter) const override {
    if (op->HasAttribute(kCanRunTrtAttr) &&
        op->template attribute<pir::BoolAttribute>(kCanRunTrtAttr).data()) {
      return false;
    }

    pir::Value x = op.operand_source(0);
    pir::Value y = op.operand_source(1);
    auto x_dtype = pir::GetDataTypeFromValue(x);
    auto y_dtype = pir::GetDataTypeFromValue(y);

    if constexpr (std::is_same_v<OpType, paddle::dialect::ElementwisePowOp>) {
      if (x_dtype.isa<pir::Int32Type>() || y_dtype.isa<pir::Int32Type>()) {
        VLOG(3) << "elementwise_pow do not support int32 datatype.";
        return false;
      }
    }

    if (x_dtype.isa<pir::BoolType>() || y_dtype.isa<pir::BoolType>()) {
      if constexpr (std::is_same_v<OpType, paddle::dialect::MultiplyOp>) {
        VLOG(3) << "elementwise_mul do not support boolean datatype.";
      } else if constexpr (std::is_same_v<OpType,  // NOLINT
                                          paddle::dialect::SubtractOp>) {
        VLOG(3) << "elementwise_sub do not support boolean datatype.";
      } else if constexpr (std::is_same_v<OpType, paddle::dialect::DivideOp>) {
        VLOG(3) << "elementwise_div do not support boolean datatype.";
      } else if constexpr (std::is_same_v<OpType,  // NOLINT
                                          paddle::dialect::ElementwisePowOp>) {
        VLOG(3) << "elementwise_pow do not support boolean datatype.";
      } else if constexpr (std::is_same_v<OpType, paddle::dialect::MinimumOp>) {
        VLOG(3) << "elementwise_min do not support boolean datatype.";
      } else if constexpr (std::is_same_v<OpType, paddle::dialect::MaximumOp>) {
        VLOG(3) << "elementwise_max do not support boolean datatype.";
      } else if constexpr (std::is_same_v<OpType,  // NOLINT
                                          paddle::dialect::FloorDivideOp>) {
        VLOG(3) << "elementwise_floordiv do not support boolean datatype.";
      } else if constexpr (std::is_same_v<OpType,  // NOLINT
                                          paddle::dialect::RemainderOp>) {
        VLOG(3) << "elementwise_mod do not support boolean datatype.";
      } else {
        VLOG(3) << "elementwise other do not support boolean datatype.";
      }
      return false;
    }

    op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));
    return true;
  }
};

using MultiplyOpPattern =
    ElementwiseCommonOpPattern<paddle::dialect::MultiplyOp>;
using SubtractOpPattern =
    ElementwiseCommonOpPattern<paddle::dialect::SubtractOp>;
using DivideOpPattern = ElementwiseCommonOpPattern<paddle::dialect::DivideOp>;
using ElementwisePowOpPattern =
    ElementwiseCommonOpPattern<paddle::dialect::ElementwisePowOp>;
using MinimumOpPattern = ElementwiseCommonOpPattern<paddle::dialect::MinimumOp>;
using MaximumOpPattern = ElementwiseCommonOpPattern<paddle::dialect::MaximumOp>;
using FloorDivideOpPattern =
    ElementwiseCommonOpPattern<paddle::dialect::FloorDivideOp>;
using RemainderOpPattern =
    ElementwiseCommonOpPattern<paddle::dialect::RemainderOp>;

template <typename OpType>
class ActOpPattern : public pir::OpRewritePattern<OpType> {
 public:
  using pir::OpRewritePattern<OpType>::OpRewritePattern;
  bool MatchAndRewrite(OpType op,
                       pir::PatternRewriter &rewriter) const override {
    if (op->HasAttribute(kCanRunTrtAttr) &&
        op->template attribute<pir::BoolAttribute>(kCanRunTrtAttr).data()) {
      return false;
    }
#if IS_TRT_VERSION_LT(8600)
    pir::Value x = op.operand_source(0);
    auto x_type = x.type().dyn_cast<paddle::dialect::DenseTensorType>();
    auto x_shape = x_type.dims();
    int dims = x_shape.size();
    if (dims < 1) {
      VLOG(3) << op->name()
              << " op does not support 0 dim input when TensorRT < 8.6.";
      return false;
    }
#endif

    op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));
    return true;
  }
};
using TanhOpPattern = ActOpPattern<paddle::dialect::TanhOp>;
using CeluOpPattern = ActOpPattern<paddle::dialect::CeluOp>;
using TanhShrinkOpPattern = ActOpPattern<paddle::dialect::TanhShrinkOp>;

template <typename OpType>
class Logical_NotOpPattern : public pir::OpRewritePattern<OpType> {
 public:
  using pir::OpRewritePattern<OpType>::OpRewritePattern;
  bool MatchAndRewrite(OpType op,
                       pir::PatternRewriter &rewriter) const override {
    if (op->HasAttribute(kCanRunTrtAttr) &&
        op->template attribute<pir::BoolAttribute>(kCanRunTrtAttr).data()) {
      return false;
    }
    pir::Value x = op.operand_source(0);
    auto x_dtype = pir::GetDataTypeFromValue(x);
    if (!x_dtype.isa<pir::BoolType>()) {
      VLOG(3) << " logical_not op only support bool input in tensorrt.";
      return false;
    }
    op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));
    return true;
  }
};
using LogicalNotOpPattern = Logical_NotOpPattern<paddle::dialect::LogicalNotOp>;
using LogicalNot_OpPattern =
    Logical_NotOpPattern<paddle::dialect::LogicalNot_Op>;

class Pool2dOpPattern
    : public pir::OpRewritePattern<paddle::dialect::Pool2dOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::Pool2dOp>::OpRewritePattern;
  bool MatchAndRewrite(paddle::dialect::Pool2dOp op,
                       pir::PatternRewriter &rewriter) const override {
    if (op->HasAttribute(kCanRunTrtAttr) &&
        op->attribute<pir::BoolAttribute>(kCanRunTrtAttr).data()) {
      return false;
    }
    paddle::dialect::FullIntArrayOp full_int_array_op =
        pir::GetDefiningOpForInput(op, 1)
            ->dyn_cast<paddle::dialect::FullIntArrayOp>();
    if (!full_int_array_op) {
      VLOG(3) << "Cannot find FullIntArrayOp";
      return false;
    }
    auto attr_value =
        full_int_array_op->attribute<pir::ArrayAttribute>("value");
    std::vector<int64_t> kernel_size;
    for (const auto &attr : attr_value.AsVector()) {
      kernel_size.push_back(attr.dyn_cast<pir::Int64Attribute>().data());
    }

    auto padding_attr = op->attribute<pir::ArrayAttribute>("paddings");
    std::vector<int32_t> paddings;
    for (const auto &attr : padding_attr.AsVector()) {
      paddings.push_back(attr.dyn_cast<pir::Int32Attribute>().data());
    }
    if (paddings.size() > 2) {
      VLOG(3) << "The padding size should be less than 2";
      return false;
    }
    if (op->HasAttribute("data_format")) {
      auto data_format =
          op->attribute<pir::StrAttribute>("data_format").AsString();
      if (data_format == "NHWC" || data_format == "NDHWC") {
        VLOG(3) << "Pool2d not support NHWC or NDHWC into trt ";
        return false;
      }
    }
    if (!op->HasAttribute("pooling_type")) {
      VLOG(3) << "The pooling_type attribute does not exist";
      return false;
    }
    std::string pool_type =
        op->attribute<pir::StrAttribute>("pooling_type").AsString();
    if (pool_type != "max" && pool_type != "avg") {
      VLOG(3) << "Wrong pool op type, the trt do not support the " << pool_type
              << " pool type.";
      return false;
    }
    if (pool_type == "avg") {
      if (op->HasAttribute("global_pooling")) {
        if (!op->attribute<pir::BoolAttribute>("global_pooling").data()) {
          if (op->HasAttribute("exclusive")) {
            if (op->attribute<pir::BoolAttribute>("exclusive").data()) {
              for (size_t i = 0; i < kernel_size.size(); ++i) {
                if (kernel_size[i] <= paddings[i]) {
                  VLOG(3) << "the padding size should be less than the "
                             "filter size "
                             "for exclusive-counting pooling.";
                  return false;
                }
              }
            }
          }
        }
      }
    }

    auto ceil_mode = op->attribute<pir::BoolAttribute>("ceil_mode").data();
    auto global_pooling =
        op->attribute<pir::BoolAttribute>("global_pooling").data();
    std::string padding_algorithm =
        op->attribute<pir::StrAttribute>("padding_algorithm").AsString();

    auto adaptive = op->attribute<pir::BoolAttribute>("adaptive").data();
    // TODO(Lizexu): This piece of code exists in the old IR-TRT implementation
    // but is not covered by unit tests, raising suspicions about its
    // correctness. In the PIR-TRT implementation, following the same approach
    // causes precision issues. For now, we will exclude it from entering
    // TensorRT.
    pir::Value input = op.operand_source(0);
    auto input_type = input.type().dyn_cast<paddle::dialect::DenseTensorType>();
    auto input_dims = input_type.dims();
    int g_post_pad_h = 0;
    int g_post_pad_w = 0;
    int input_height = input_dims[input_dims.size() - 2];
    int input_width = input_dims[input_dims.size() - 1];
    std::vector<int32_t> strides;
    auto strides_attr = op->attribute<pir::ArrayAttribute>("strides");
    for (const auto &attr : strides_attr.AsVector()) {
      strides.push_back(attr.dyn_cast<pir::Int32Attribute>().data());
    }
    if (input_height > 0 &&
        input_height - kernel_size[0] + 2 * paddings[0] < 0) {
      g_post_pad_h = strides[0] - 1;
    }
    if (input_width > 0 && input_width - kernel_size[1] + 2 * paddings[1] < 0) {
      g_post_pad_w = strides[1] - 1;
    }
    if (!adaptive && !global_pooling && !ceil_mode) {
      if (padding_algorithm != "SAME" &&
          ((g_post_pad_h > 0 && input_height > 0) ||
           (g_post_pad_w > 0 && input_width > 0))) {
        VLOG(3) << "The pool2d op meets the condition that may cause precision "
                   "issues in TRT. Skip TRT conversion.";
        return false;
      }
    }
    op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));
    return true;
  }
};

class Conv2dTransposeOpPattern
    : public pir::OpRewritePattern<paddle::dialect::Conv2dTransposeOp> {
 public:
  using pir::OpRewritePattern<
      paddle::dialect::Conv2dTransposeOp>::OpRewritePattern;
  bool MatchAndRewrite(paddle::dialect::Conv2dTransposeOp op,
                       pir::PatternRewriter &rewriter) const override {
    if (op->HasAttribute(kCanRunTrtAttr) &&
        op->attribute<pir::BoolAttribute>(kCanRunTrtAttr).data()) {
      return false;
    }
    if (!op->HasAttribute("dilations")) {
      VLOG(3) << "In conv2d_transpose, dilations attribute does not exist";
      return false;
    } else {
      auto dilation_attr = op->attribute<pir::ArrayAttribute>("dilations");
      std::vector<int32_t> dilations;
      for (const auto &attr : dilation_attr.AsVector()) {
        dilations.push_back(attr.dyn_cast<pir::Int32Attribute>().data());
      }
      if (dilations[0] != 1 || dilations[1] != 1) {
        VLOG(3) << "In conv2d_transpose, Dilations must be (1, 1) for "
                   "tensorRT, but given ("
                << dilations[0] << ", " << dilations[1] << ")";
        return false;
      }
    }
    op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));
    return true;
  }
};

class DepthwiseConv2dTransposeOpPattern
    : public pir::OpRewritePattern<
          paddle::dialect::DepthwiseConv2dTransposeOp> {
 public:
  using pir::OpRewritePattern<
      paddle::dialect::DepthwiseConv2dTransposeOp>::OpRewritePattern;
  bool MatchAndRewrite(paddle::dialect::DepthwiseConv2dTransposeOp op,
                       pir::PatternRewriter &rewriter) const override {
    if (op->HasAttribute(kCanRunTrtAttr) &&
        op->attribute<pir::BoolAttribute>(kCanRunTrtAttr).data()) {
      return false;
    }
    if (!op->HasAttribute("dilations")) {
      VLOG(3) << "In depthwise_conv2d_transpose, dilations attribute does not "
                 "exist";
      return false;
    } else {
      auto dilation_attr = op->attribute<pir::ArrayAttribute>("dilations");
      std::vector<int32_t> dilations;
      for (const auto &attr : dilation_attr.AsVector()) {
        dilations.push_back(attr.dyn_cast<pir::Int32Attribute>().data());
      }
      if (dilations[0] != 1 || dilations[1] != 1) {
        VLOG(3)
            << "In depthwise_conv2d_transpose, Dilations must be (1, 1) for "
               "tensorRT, but given ("
            << dilations[0] << ", " << dilations[1] << ")";
        return false;
      }
    }

    op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));
    return true;
  }
};

class DeformableConvOpPattern
    : public pir::OpRewritePattern<paddle::dialect::DeformableConvOp> {
 public:
  using pir::OpRewritePattern<
      paddle::dialect::DeformableConvOp>::OpRewritePattern;
  bool MatchAndRewrite(paddle::dialect::DeformableConvOp op,
                       pir::PatternRewriter &rewriter) const override {
    if (op->HasAttribute(kCanRunTrtAttr) &&
        op->attribute<pir::BoolAttribute>(kCanRunTrtAttr).data()) {
      return false;
    }
    if (!op->HasAttribute("groups") || !op->HasAttribute("strides") ||
        !op->HasAttribute("paddings")) {
      VLOG(3) << "In deformable_conv, groups or strides or paddings attributes "
                 "do not exist";
      return false;
    }
    pir::Value input = op.operand_source(0);
    auto input_type = input.type().dyn_cast<paddle::dialect::DenseTensorType>();
    auto input_shape = input_type.dims();
    if (input_shape.size() != 4) {
      VLOG(3) << "Input of deformable conv should be 4-D Tensor, but got "
              << input_shape.size() << "-D Tensor";
      return false;
    }
    pir::Value filter = op.operand_source(2);
    auto filter_type =
        filter.type().dyn_cast<paddle::dialect::DenseTensorType>();
    auto filter_shape = filter_type.dims();
    int groups = op->attribute<pir::Int32Attribute>("groups").data();
    if (input_shape[1] != filter_shape[1] * groups) {
      VLOG(3) << "The number of input channels should be equal to filter "
              << "channels * groups. But got input channels " << input_shape[1]
              << "filter channels " << filter_shape[1];
      return false;
    }
    std::vector<int32_t> strides;
    auto stride_attr = op->attribute<pir::ArrayAttribute>("strides");
    for (const auto &attr : stride_attr.AsVector()) {
      strides.push_back(attr.dyn_cast<pir::Int32Attribute>().data());
    }
    if (strides.size() != 2) {
      VLOG(3) << "The size of strides should be 2, but got " << strides.size();
      return false;
    }
    std::vector<int32_t> paddings;
    auto padding_attr = op->attribute<pir::ArrayAttribute>("paddings");
    for (const auto &attr : padding_attr.AsVector()) {
      paddings.push_back(attr.dyn_cast<pir::Int32Attribute>().data());
    }
    if (paddings.size() != 2) {
      VLOG(3) << "The size of paddings should be 2, but got "
              << paddings.size();
      return false;
    }
    op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));
    return true;
  }
};

class ArangeOpPattern
    : public pir::OpRewritePattern<paddle::dialect::ArangeOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::ArangeOp>::OpRewritePattern;
  bool MatchAndRewrite(paddle::dialect::ArangeOp op,
                       pir::PatternRewriter &rewriter) const override {
    if (op->HasAttribute(kCanRunTrtAttr) &&
        op->attribute<pir::BoolAttribute>(kCanRunTrtAttr).data()) {
      return false;
    }
#if IS_TRT_VERSION_LT(8400)
    pir::Value start = op.operand_source(0);
    auto start_type = pir::GetDataTypeFromValue(start);
    if (!start_type.isa<pir::Float32Type>() ||
        !start_type.isa<pir::Float64Type>()) {
      VLOG(3) << "The type of start is not float32 or float64";
      return false;
    }
#endif
    op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));
    return true;
  }
};

class GroupNormOpPattern
    : public pir::OpRewritePattern<paddle::dialect::GroupNormOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::GroupNormOp>::OpRewritePattern;
  bool MatchAndRewrite(paddle::dialect::GroupNormOp op,
                       pir::PatternRewriter &rewriter) const override {
    if (op->HasAttribute(kCanRunTrtAttr) &&
        op->attribute<pir::BoolAttribute>(kCanRunTrtAttr).data()) {
      return false;
    }
    if (!op->HasAttribute("epsilon") || !op->HasAttribute("groups") ||
        !op->HasAttribute("data_format")) {
      VLOG(3) << "In group_norm, epsilon or groups or data_format attributes "
                 "do not exist";
      return false;
    }
    std::string layout_str =
        op->attribute<pir::StrAttribute>("data_format").AsString();
    if (layout_str != "NCHW") {
      VLOG(3) << "Group norm trt plugin only support NCHW layout, but got "
              << layout_str;
      return false;
    }
    op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));
    return true;
  }
};

class TransposeOpPattern
    : public pir::OpRewritePattern<paddle::dialect::TransposeOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::TransposeOp>::OpRewritePattern;
  bool MatchAndRewrite(paddle::dialect::TransposeOp op,
                       pir::PatternRewriter &rewriter) const override {
    if (op->HasAttribute(kCanRunTrtAttr) &&
        op->attribute<pir::BoolAttribute>(kCanRunTrtAttr).data()) {
      return false;
    }
    pir::Value x = op.operand_source(0);
    auto x_type = x.type().dyn_cast<paddle::dialect::DenseTensorType>();
    auto x_shape = x_type.dims();
    int dims = x_shape.size();
    std::vector<int> perm;
    auto perm_attr = op->attribute<pir::ArrayAttribute>("perm");
    for (const auto &attr : perm_attr.AsVector()) {
      perm.push_back(attr.dyn_cast<pir::Int32Attribute>().data());
    }
    auto is_valid_permutation = [&](int dims,
                                    const std::vector<int> &permutation) {
      std::bitset<nvinfer1::Dims::MAX_DIMS> found;
      for (int i = 0; i < dims; ++i) {
        const int x = permutation[i];
        if ((x < 0) || (x >= dims) || found[x])
          return false;  // Out of bounds or duplicate
        found.set(x);
      }
      return true;
    };
    if (!is_valid_permutation(dims, perm)) {
      VLOG(3) << "Invalid permutation dimensions for trt transpose op "
                 "converter: duplicate or out of bound.";
      return false;
    }
    op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));
    return true;
  }
};

class GatherOpPattern
    : public pir::OpRewritePattern<paddle::dialect::GatherOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::GatherOp>::OpRewritePattern;
  bool MatchAndRewrite(paddle::dialect::GatherOp op,
                       pir::PatternRewriter &rewriter) const override {
    if (op->HasAttribute(kCanRunTrtAttr) &&
        op->attribute<pir::BoolAttribute>(kCanRunTrtAttr).data()) {
      return false;
    }
    if (!op.axis().defining_op()->isa<paddle::dialect::FullOp>()) {
      VLOG(3) << "When axis is not a constant "
                 "Skip to convert into TRT.";

      return false;
    }
    op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));
    return true;
  }
};

class GatherNdOpPattern
    : public pir::OpRewritePattern<paddle::dialect::GatherNdOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::GatherNdOp>::OpRewritePattern;
  bool MatchAndRewrite(paddle::dialect::GatherNdOp op,
                       pir::PatternRewriter &rewriter) const override {
    if (op->HasAttribute(kCanRunTrtAttr) &&
        op->attribute<pir::BoolAttribute>(kCanRunTrtAttr).data()) {
      return false;
    }
#if IS_TRT_VERSION_LT(8200)
    pir::Value index_var_name = op.operand_source(1);
    auto index_var_name_type =
        index_var_name.type().dyn_cast<paddle::dialect::DenseTensorType>();
    auto index_shape = index_var_name_type.dims();
    pir::Value x_var_name = op.operand_source(0);
    auto x_var_name_type =
        x_var_name.type().dyn_cast<paddle::dialect::DenseTensorType>();
    auto x_shape = x_var_name_type.dims();
    if (x_shape.size() <= 2) {
      VLOG(3) << "gather_nd op requires the input's dimension to be greater "
                 "than 2";
      return false;
    }
    if (x_shape.size() != index_shape.size()) {
      VLOG(3) << "gather_nd op Index input dims size [" << index_shape.size()
              << " ] not equal to x dims size [" << x_shape.size() << "]";
      return false;
    }
#endif
    op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));
    return true;
  }
};

class ScaleOpPattern : public pir::OpRewritePattern<paddle::dialect::ScaleOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::ScaleOp>::OpRewritePattern;
  bool MatchAndRewrite(paddle::dialect::ScaleOp op,
                       pir::PatternRewriter &rewriter) const override {
    if (op->HasAttribute(kCanRunTrtAttr) &&
        op->attribute<pir::BoolAttribute>(kCanRunTrtAttr).data()) {
      return false;
    }
    pir::Value x = op.operand_source(0);
    auto x_dtype = pir::GetDataTypeFromValue(x);

    if (!(x_dtype.isa<pir::Float32Type>() || x_dtype.isa<pir::Float64Type>() ||
          x_dtype.isa<pir::Float16Type>() || x_dtype.isa<pir::Int32Type>() ||
          x_dtype.isa<pir::Int64Type>())) {
      VLOG(3) << "At present, ScaleOp only support float32 or float16 or "
                 "float64 or int32 or int64 into trt.";
      return false;
    }
    op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));
    return true;
  }
};

class UnsqueezeOpPattern
    : public pir::OpRewritePattern<paddle::dialect::UnsqueezeOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::UnsqueezeOp>::OpRewritePattern;
  bool MatchAndRewrite(paddle::dialect::UnsqueezeOp op,
                       pir::PatternRewriter &rewriter) const override {
    if (op->HasAttribute(kCanRunTrtAttr) &&
        op->attribute<pir::BoolAttribute>(kCanRunTrtAttr).data()) {
      return false;
    }
    paddle::dialect::FullIntArrayOp full_int_array_op =
        pir::GetDefiningOpForInput(op, 1)
            ->dyn_cast<paddle::dialect::FullIntArrayOp>();
    auto axis = full_int_array_op->attribute<pir::ArrayAttribute>("value");

    if (!axis) {
      VLOG(3) << "The necessary attributes of the unsuqeeze axis is missing";
      return false;
    }
    pir::Value x = op.operand_source(0);
    auto x_type = x.type().dyn_cast<paddle::dialect::DenseTensorType>();
    auto x_shape = x_type.dims();

    std::vector<int32_t> dynamic_dims;
    for (int i = 0; i < x_shape.size(); ++i) {
      if (x_shape[i] == -1) {
        dynamic_dims.push_back(i);
      }
    }
    if (dynamic_dims.size() > 1) {
      VLOG(3) << "Currently we don't support unsqueeze with more than one "
                 "dynamic dims";
      return false;
    }

    op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));
    return true;
  }
};

class Unsqueeze_OpPattern
    : public pir::OpRewritePattern<paddle::dialect::Unsqueeze_Op> {
 public:
  using pir::OpRewritePattern<paddle::dialect::Unsqueeze_Op>::OpRewritePattern;
  bool MatchAndRewrite(paddle::dialect::Unsqueeze_Op op,
                       pir::PatternRewriter &rewriter) const override {
    if (op->HasAttribute(kCanRunTrtAttr) &&
        op->attribute<pir::BoolAttribute>(kCanRunTrtAttr).data()) {
      return false;
    }
    paddle::dialect::FullIntArrayOp full_int_array_op =
        pir::GetDefiningOpForInput(op, 1)
            ->dyn_cast<paddle::dialect::FullIntArrayOp>();
    auto axis = full_int_array_op->attribute<pir::ArrayAttribute>("value");

    if (!axis) {
      VLOG(3) << "The necessary attributes of the unsuqeeze axis is missing";
      return false;
    }
    pir::Value x = op.operand_source(0);
    auto x_type = x.type().dyn_cast<paddle::dialect::DenseTensorType>();
    auto x_shape = x_type.dims();

    std::vector<int32_t> dynamic_dims;
    for (int i = 0; i < x_shape.size(); ++i) {
      if (x_shape[i] == -1) {
        dynamic_dims.push_back(i);
      }
    }
    if (dynamic_dims.size() > 1) {
      VLOG(3) << "Currently we don't support unsqueeze with more than one "
                 "dynamic dims";
      return false;
    }

    op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));
    return true;
  }
};

class SqueezeOpPattern
    : public pir::OpRewritePattern<paddle::dialect::SqueezeOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::SqueezeOp>::OpRewritePattern;
  bool MatchAndRewrite(paddle::dialect::SqueezeOp op,
                       pir::PatternRewriter &rewriter) const override {
    if (op->HasAttribute(kCanRunTrtAttr) &&
        op->attribute<pir::BoolAttribute>(kCanRunTrtAttr).data()) {
      return false;
    }

    paddle::dialect::FullIntArrayOp full_int_array_op =
        pir::GetDefiningOpForInput(op, 1)
            ->dyn_cast<paddle::dialect::FullIntArrayOp>();

    auto axis = full_int_array_op->attribute<pir::ArrayAttribute>("value");
    std::vector<int64_t> axes;
    for (const auto &attr : axis.AsVector()) {
      axes.push_back(attr.dyn_cast<pir::Int64Attribute>().data());
    }
    if (axes.empty()) {
      auto input_var_name = op.operand_source(0);
      auto input_var_name_type =
          input_var_name.type().dyn_cast<paddle::dialect::DenseTensorType>();
      auto input_var_name_shape = input_var_name_type.dims();

      for (int i = 0; i < input_var_name_shape.size(); ++i) {
        int64_t s = input_var_name_shape[i];
        if (s == -1) {
          VLOG(3) << "The necessary attributes of the squeeze operator axis is "
                     "missing. ss =====-1";
          return false;
        } else if (s == 1) {
          axes.push_back(s);
        }
      }

      if (axes.empty()) {
        VLOG(3) << "The necessary attributes of the squeeze2 operator axes is "
                   "missing.";
        return false;
      }
    }

    op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));
    return true;
  }
};

class SliceOpPattern : public pir::OpRewritePattern<paddle::dialect::SliceOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::SliceOp>::OpRewritePattern;
  bool MatchAndRewrite(paddle::dialect::SliceOp op,
                       pir::PatternRewriter &rewriter) const override {
    if (op->HasAttribute(kCanRunTrtAttr) &&
        op->attribute<pir::BoolAttribute>(kCanRunTrtAttr).data()) {
      return false;
    }

    if (!op->HasAttribute("axes")) {
      VLOG(3)
          << "The necessary attribute of the slice operator axes are missing.";
      return false;
    }

    auto axes_attr = op->attribute<pir::ArrayAttribute>("axes");
    std::vector<int64_t> axes;
    for (const auto &attr : axes_attr.AsVector()) {
      axes.push_back(attr.dyn_cast<pir::Int64Attribute>().data());
    }

    size_t starts_size = axes.size();
    size_t ends_size = axes.size();
    if (pir::GetDefiningOpForInput(op, 1)
            ->isa<paddle::dialect::FullIntArrayOp>()) {
      paddle::dialect::FullIntArrayOp full_int_array_op_start =
          pir::GetDefiningOpForInput(op, 1)
              ->dyn_cast<paddle::dialect::FullIntArrayOp>();
      auto starts_attr =
          full_int_array_op_start->attribute<pir::ArrayAttribute>("value");
      std::vector<int64_t> starts;
      for (const auto &attr : starts_attr.AsVector()) {
        starts.push_back(attr.dyn_cast<pir::Int64Attribute>().data());
      }
      starts_size = starts.size();
    }

    if (pir::GetDefiningOpForInput(op, 2)
            ->isa<paddle::dialect::FullIntArrayOp>()) {
      paddle::dialect::FullIntArrayOp full_int_array_op_end =
          pir::GetDefiningOpForInput(op, 2)
              ->dyn_cast<paddle::dialect::FullIntArrayOp>();
      auto ends_attr =
          full_int_array_op_end->attribute<pir::ArrayAttribute>("value");
      std::vector<int64_t> ends;
      for (const auto &attr : ends_attr.AsVector()) {
        ends.push_back(attr.dyn_cast<pir::Int64Attribute>().data());
      }
      ends_size = ends.size();
    }
    if (starts_size != axes.size() || ends_size != axes.size()) {
      VLOG(3) << "The size of axes and starts are not equal. "
                 "Axes size: "
              << axes.size() << ", Starts size: " << starts_size
              << ", Ends size: " << ends_size;
      return false;
    }

    op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));
    return true;
  }
};

class IndexSelectOpPattern
    : public pir::OpRewritePattern<paddle::dialect::IndexSelectOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::IndexSelectOp>::OpRewritePattern;
  bool MatchAndRewrite(paddle::dialect::IndexSelectOp op,
                       pir::PatternRewriter &rewriter) const override {
    if (op->HasAttribute(kCanRunTrtAttr) &&
        op->attribute<pir::BoolAttribute>(kCanRunTrtAttr).data()) {
      return false;
    }
#if IS_TRT_VERSION_LT(8200)
    VLOG(3) << "index_select op is only supported by tensorrt8.2 above ";
    return false;
#endif
    pir::Value x = op.operand_source(1);
    auto x_dtype = pir::GetDataTypeFromValue(x);
    if (!(x_dtype.isa<pir::Int32Type>() || x_dtype.isa<pir::Int64Type>())) {
      VLOG(3) << "Index select op Index input data type must be int32 or int64";
      return false;
    }
    op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));
    return true;
  }
};

class FlattenOpPattern
    : public pir::OpRewritePattern<paddle::dialect::FlattenOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::FlattenOp>::OpRewritePattern;
  bool MatchAndRewrite(paddle::dialect::FlattenOp op,
                       pir::PatternRewriter &rewriter) const override {
    if (op->HasAttribute(kCanRunTrtAttr) &&
        op->attribute<pir::BoolAttribute>(kCanRunTrtAttr).data()) {
      return false;
    }
    if (!op->HasAttribute("start_axis") && !op->HasAttribute("stop_axis")) {
      VLOG(3) << "flatten op must has start_axis and stop_axis attributes";
      return false;
    }

    op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));
    return true;
  }
};
class CastOpPattern : public pir::OpRewritePattern<paddle::dialect::CastOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::CastOp>::OpRewritePattern;
  bool MatchAndRewrite(paddle::dialect::CastOp op,
                       pir::PatternRewriter &rewriter) const override {
    if (op->HasAttribute(kCanRunTrtAttr) &&
        op->attribute<pir::BoolAttribute>(kCanRunTrtAttr).data()) {
      return false;
    }
    if (!op->HasAttribute("dtype")) {
      VLOG(3) << "the cast op does not have attr dtype ";
      return false;
    }
    auto dtype =
        op->attribute<paddle::dialect::DataTypeAttribute>("dtype").data();
    if (dtype == phi::DataType::BOOL) {
#if IS_TRT_VERSION_LT(8400)
      VLOG(3)
          << "the cast op supports inputs and outputs of BOOL by trt8.4 above ";
      return false;
#endif
    }
    op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));
    return true;
  }
};

class SplitOpPattern : public pir::OpRewritePattern<paddle::dialect::SplitOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::SplitOp>::OpRewritePattern;

  bool MatchAndRewrite(paddle::dialect::SplitOp op,
                       pir::PatternRewriter &rewriter) const override {
    if (op->HasAttribute(kCanRunTrtAttr) &&
        op->attribute<pir::BoolAttribute>(kCanRunTrtAttr).data()) {
      return false;
    }

    pir::Value axis_tensor = op.operand_source(2);
    if (!axis_tensor) {
      VLOG(3) << "pd_op.split can not find axis input";
      return false;
    }
    auto out_vector_type = op.result(0).type().dyn_cast<pir::VectorType>();
    if (pir::GetDefiningOpForInput(op, 2)->isa<paddle::dialect::FullOp>()) {
      paddle::dialect::FullOp full_op =
          pir::GetDefiningOpForInput(op, 2)
              ->dyn_cast<paddle::dialect::FullOp>();
      auto axis = full_op->attribute<paddle::dialect::ScalarAttribute>("value")
                      .data()
                      .to<int>();
      auto x_shape = op.operand_source(0)
                         .type()
                         .dyn_cast<paddle::dialect::DenseTensorType>()
                         .dims();

      axis += (axis < 0) ? x_shape.size() : 0;

      if (x_shape[axis] == -1) {
        VLOG(3) << "The (" << axis << ") dim of input should not be -1";
        return false;
      }
    }

    if (pir::GetDefiningOpForInput(op, 1)
            ->isa<paddle::dialect::FullIntArrayOp>()) {
      paddle::dialect::FullIntArrayOp full_sections_op =
          pir::GetDefiningOpForInput(op, 1)
              ->dyn_cast<paddle::dialect::FullIntArrayOp>();
      auto sections = full_sections_op->attribute<pir::ArrayAttribute>("value");
      std::vector<int64_t> output_lengths;
      for (const auto &attr : sections.AsVector()) {
        output_lengths.push_back(attr.dyn_cast<pir::Int64Attribute>().data());
      }
      if (output_lengths.size() != out_vector_type.size()) {
        VLOG(3) << "The output_length should be equal to the output size.";
        return false;
      }
    }

    op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));
    return true;
  }
};

class SplitWithNumOpPattern
    : public pir::OpRewritePattern<paddle::dialect::SplitWithNumOp> {
 public:
  using pir::OpRewritePattern<
      paddle::dialect::SplitWithNumOp>::OpRewritePattern;
  bool MatchAndRewrite(paddle::dialect::SplitWithNumOp op,
                       pir::PatternRewriter &rewriter) const override {
    if (op->HasAttribute(kCanRunTrtAttr) &&
        op->attribute<pir::BoolAttribute>(kCanRunTrtAttr).data()) {
      return false;
    }

    pir::Value axis_tensor = op.operand_source(1);
    if (!axis_tensor) {
      VLOG(3) << "pd_op.split_with_num can not find axis input";
      return false;
    }
    if (pir::GetDefiningOpForInput(op, 1)
            ->isa<paddle::dialect::FullIntArrayOp>()) {
      paddle::dialect::FullIntArrayOp full_int_array_op =
          pir::GetDefiningOpForInput(op, 1)
              ->dyn_cast<paddle::dialect::FullIntArrayOp>();
      auto axis = full_int_array_op
                      ->attribute<paddle::dialect::ScalarAttribute>("value")
                      .data()
                      .to<int>();
      auto x_shape = op.operand_source(0)
                         .type()
                         .dyn_cast<paddle::dialect::DenseTensorType>()
                         .dims();

      axis += (axis < 0) ? x_shape.size() : 0;
      if (x_shape[axis] == -1) {
        VLOG(3) << "The (" << axis << ") dim of input should not be -1";
        return false;
      }
      if (!op->HasAttribute("num")) {
        VLOG(3) << "split_with_num op must has num attributes";
        return false;
      }
      int num = op->attribute<pir::Int32Attribute>("num").data();
      std::vector<int64_t> output_lengths;

      if (num > 0) {
        int64_t in_axis_dim = x_shape[axis];
        if (in_axis_dim % num != 0) {
          VLOG(3) << "Invalid number to split. Tensor split does not result"
                     " in an equal division of dimensions. Axis dim = "
                  << in_axis_dim << " num = " << num << "!= 0";
          return false;
        }
        size_t out_axis_dim = in_axis_dim / num;
        for (int i = 0; i < num; ++i) {
          output_lengths.push_back(out_axis_dim);
        }
      }
      auto out_vector_type = op.result(0).type().dyn_cast<pir::VectorType>();
      if (out_vector_type.size() != output_lengths.size()) {
        VLOG(3) << "The output_length should be equal to the output size.";
        return false;
      }
    }

    op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));
    return true;
  }
};

class GreaterThanOpPattern
    : public pir::OpRewritePattern<paddle::dialect::GreaterThanOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::GreaterThanOp>::OpRewritePattern;
  bool MatchAndRewrite(paddle::dialect::GreaterThanOp op,
                       pir::PatternRewriter &rewriter) const override {
    if (op->HasAttribute(kCanRunTrtAttr) &&
        op->attribute<pir::BoolAttribute>(kCanRunTrtAttr).data()) {
      return false;
    }
#if IS_TRT_VERSION_LT(8400)
    VLOG(3) << "pd_op.greater_than op is not supported when TensorRT < 8.4";
    return false;
#else
    pir::Value x = op.operand_source(0);
    pir::Value y = op.operand_source(1);
    auto x_dtype = pir::GetDataTypeFromValue(x);
    auto y_dtype = pir::GetDataTypeFromValue(y);
    if (x_dtype.isa<pir::BoolType>() || y_dtype.isa<pir::BoolType>()) {
      VLOG(3) << "pd_op.greater_than op do not support bool datatype";
      return false;
    }
#endif
    op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));
    return true;
  }
};

class LessThanOpPattern
    : public pir::OpRewritePattern<paddle::dialect::LessThanOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::LessThanOp>::OpRewritePattern;
  bool MatchAndRewrite(paddle::dialect::LessThanOp op,
                       pir::PatternRewriter &rewriter) const override {
    if (op->HasAttribute(kCanRunTrtAttr) &&
        op->attribute<pir::BoolAttribute>(kCanRunTrtAttr).data()) {
      return false;
    }
#if IS_TRT_VERSION_LT(8400)
    VLOG(3) << "pd_op.less_than op is not supported when TensorRT < 8.4";
    return false;
#else
    pir::Value x = op.operand_source(0);
    pir::Value y = op.operand_source(1);
    auto x_dtype = pir::GetDataTypeFromValue(x);
    auto y_dtype = pir::GetDataTypeFromValue(y);
    if (x_dtype.isa<pir::BoolType>() || y_dtype.isa<pir::BoolType>()) {
      VLOG(3) << "pd_op.less_than op do not support bool datatype";
      return false;
    }
#endif
    op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));
    return true;
  }
};

template <typename OpType>
class LogicalCommonOpPattern : public pir::OpRewritePattern<OpType> {
 public:
  using pir::OpRewritePattern<OpType>::OpRewritePattern;
  bool MatchAndRewrite(OpType op,
                       pir::PatternRewriter &rewriter) const override {
    if (op->HasAttribute(kCanRunTrtAttr) &&
        op->template attribute<pir::BoolAttribute>(kCanRunTrtAttr).data()) {
      return false;
    }
    pir::Value x = op.operand_source(0);
    pir::Value y = op.operand_source(1);
    auto x_dtype = pir::GetDataTypeFromValue(x);
    auto y_dtype = pir::GetDataTypeFromValue(y);
    if (!(x_dtype.isa<pir::BoolType>() && y_dtype.isa<pir::BoolType>())) {
      VLOG(3) << op->name() << " op only supports bool datatype";
      return false;
    }
    op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));
    return true;
  }
};
using LogicalXorOpPattern =
    LogicalCommonOpPattern<paddle::dialect::LogicalXorOp>;
using LogicalOrOpPattern = LogicalCommonOpPattern<paddle::dialect::LogicalOrOp>;
using LogicalOr_OpPattern =
    LogicalCommonOpPattern<paddle::dialect::LogicalOr_Op>;
using LogicalAndOpPattern =
    LogicalCommonOpPattern<paddle::dialect::LogicalAndOp>;

template <typename OpType>
class ComparisonCommonOpPattern : public pir::OpRewritePattern<OpType> {
 public:
  using pir::OpRewritePattern<OpType>::OpRewritePattern;
  bool MatchAndRewrite(OpType op,
                       pir::PatternRewriter &rewriter) const override {
    if (op->HasAttribute(kCanRunTrtAttr) &&
        op->template attribute<pir::BoolAttribute>(kCanRunTrtAttr).data()) {
      return false;
    }
    pir::Value x = op.operand_source(0);
    pir::Value y = op.operand_source(1);
    auto x_dtype = pir::GetDataTypeFromValue(x);
    auto y_dtype = pir::GetDataTypeFromValue(y);
    if (x_dtype.isa<pir::BoolType>() || y_dtype.isa<pir::BoolType>()) {
      VLOG(3) << "ElementWiseOperation::kLESS/ElementWiseOperation::kGREATER "
                 "do not support boolean datatype.";
      return false;
    }
    op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));
    return true;
  }
};
using LessEqualOpPattern =
    ComparisonCommonOpPattern<paddle::dialect::LessEqualOp>;
using LessEqual_OpPattern =
    ComparisonCommonOpPattern<paddle::dialect::LessEqual_Op>;
using GreaterEqualOpPattern =
    ComparisonCommonOpPattern<paddle::dialect::GreaterEqualOp>;
using GreaterEqual_OpPattern =
    ComparisonCommonOpPattern<paddle::dialect::GreaterEqual_Op>;

class MulticlassNms3OpPattern
    : public pir::OpRewritePattern<paddle::dialect::MulticlassNms3Op> {
 public:
  using pir::OpRewritePattern<
      paddle::dialect::MulticlassNms3Op>::OpRewritePattern;
  bool MatchAndRewrite(paddle::dialect::MulticlassNms3Op op,
                       pir::PatternRewriter &rewriter) const override {
    if (op->HasAttribute(kCanRunTrtAttr) &&
        op.attribute<pir::BoolAttribute>(kCanRunTrtAttr).data()) {
      return false;
    }
    auto rois_num = op.operand_source(2);
    if (rois_num.impl() != nullptr) {
      return false;
    }
    for (auto operand : op->operands()) {
      auto operand_source = operand.source();
      if (operand_source.impl() == nullptr) {
        continue;
      }
      auto shape = operand_source.type()
                       .dyn_cast<paddle::dialect::DenseTensorType>()
                       .dims();
      if (shape.size() != 3) {
        VLOG(3) << "multiclass_nms op dims != 3 not supported in tensorrt, "
                   "but got dims "
                << shape.size() << ", so jump it.";
        return false;
      }
    }
    bool has_attrs =
        (op->HasAttribute("background_label") &&
         op->HasAttribute("score_threshold") && op->HasAttribute("nms_top_k") &&
         op->HasAttribute("keep_top_k") && op->HasAttribute("normalized"));
    if (has_attrs == false) return false;

    // TODO(wangxinxin08): tricky solution because the outputs of batchedNMS
    // plugin are not constient with those of multiclass_nms3
    if (op->HasAttribute("nms_eta") == false) return false;
    auto nms_eta = op.attribute<pir::FloatAttribute>("nms_eta").data();
    if (nms_eta <= 1.0) return false;

    auto nms_top_k = op.attribute<pir::Int32Attribute>("nms_top_k").data();
    if (nms_top_k < 0) return false;

    auto keep_top_k = op.attribute<pir::Int32Attribute>("keep_top_k").data();
    if (keep_top_k < 0) return false;

    auto registry = paddle::platform::GetPluginRegistry();

    if (registry == nullptr) return false;
    op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));
    return true;
  }
};

class ArgmaxOpPattern
    : public pir::OpRewritePattern<paddle::dialect::ArgmaxOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::ArgmaxOp>::OpRewritePattern;
  bool MatchAndRewrite(paddle::dialect::ArgmaxOp op,
                       pir::PatternRewriter &rewriter) const override {
    if (op->HasAttribute(kCanRunTrtAttr) &&
        op.attribute<pir::BoolAttribute>(kCanRunTrtAttr).data()) {
      return false;
    }
    if (!op.axis().defining_op()->isa<paddle::dialect::FullOp>()) {
      VLOG(3) << "Skip to convert into TRT while found axis is not a constant "
                 "data in arg_max.";
      return false;
    }
    pir::Value x = op.x();
    auto data_type = pir::GetDataTypeFromValue(x);
    if (!(data_type.isa<pir::Float32Type>() ||
          data_type.isa<pir::Float16Type>() ||
          data_type.isa<pir::Float64Type>())) {
      VLOG(3) << "At present, pd_op.argmax only support float32 or float16 or "
                 "float64 into trt.";
      return false;
    }
    int axis = static_cast<int>(op.axis()
                                    .defining_op()
                                    ->attribute<pir::DoubleAttribute>("value")
                                    .data());

    bool flatten = op.attribute<pir::BoolAttribute>("flatten").data();
    phi::DataType dtype =
        op.attribute<paddle::dialect::DataTypeAttribute>("dtype").data();
    if (axis == 0 || flatten ||
        (dtype != phi::DataType::INT32 && dtype != phi::DataType::INT64)) {
      VLOG(3) << "Skipping TRT conversion in pd_op.argmax: axis is zero, "
                 "flatten is True, or dtype isn't int32/int64";
      return false;
    }
    op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));
    return true;
  }
};

class ArgminOpPattern
    : public pir::OpRewritePattern<paddle::dialect::ArgminOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::ArgminOp>::OpRewritePattern;
  bool MatchAndRewrite(paddle::dialect::ArgminOp op,
                       pir::PatternRewriter &rewriter) const override {
    if (op->HasAttribute(kCanRunTrtAttr) &&
        op.attribute<pir::BoolAttribute>(kCanRunTrtAttr).data()) {
      return false;
    }
    if (!op.axis().defining_op()->isa<paddle::dialect::FullOp>()) {
      VLOG(3) << "Skip to convert into TRT while found axis is not a constant "
                 "data in arg_mix.";
      return false;
    }
    pir::Value x = op.x();
    auto data_type = pir::GetDataTypeFromValue(x);
    if (!(data_type.isa<pir::Float32Type>() ||
          data_type.isa<pir::Float16Type>() ||
          data_type.isa<pir::Float64Type>())) {
      VLOG(3) << "At present, pd_op.argmin only support float32 or float16 or "
                 "float64 into trt.";
      return false;
    }
    int axis = static_cast<int>(op.axis()
                                    .defining_op()
                                    ->attribute<pir::DoubleAttribute>("value")
                                    .data());

    bool flatten = op.attribute<pir::BoolAttribute>("flatten").data();
    phi::DataType dtype =
        op.attribute<paddle::dialect::DataTypeAttribute>("dtype").data();
    if (axis == 0 || flatten ||
        (dtype != phi::DataType::INT32 && dtype != phi::DataType::INT64)) {
      VLOG(3) << "Skipping TRT conversion in pd_op.argmin: axis is zero, "
                 "flatten is True, or dtype isn't int32/int64";
      return false;
    }

    op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));
    return true;
  }
};

class ArgsortOpPattern
    : public pir::OpRewritePattern<paddle::dialect::ArgsortOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::ArgsortOp>::OpRewritePattern;
  bool MatchAndRewrite(paddle::dialect::ArgsortOp op,
                       pir::PatternRewriter &rewriter) const override {
    if (op->HasAttribute(kCanRunTrtAttr) &&
        op.attribute<pir::BoolAttribute>(kCanRunTrtAttr).data()) {
      return false;
    }
    const std::vector<std::string> required_attrs = {"axis", "descending"};
    for (const auto &attr : required_attrs) {
      if (!op->HasAttribute(attr)) {
        VLOG(3) << "pd_op.argsort " << attr << " attribute does not exist";
        return false;
      }
    }
    pir::Value x = op.x();
    auto x_type = x.type().dyn_cast<paddle::dialect::DenseTensorType>();
    auto x_shape = x_type.dims();
    int axis = op->attribute<pir::Int32Attribute>("axis").data();
    if (axis < 0) {
      axis += x_shape.size();
    }
    if (x_shape[axis] > 3840) {
      VLOG(3)
          << "In pd_op.argsort,the axis dim of input should be less than 3840";
      return false;
    }
    op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));
    return true;
  }
};
class BilinearInterpV2Pattern
    : public pir::OpRewritePattern<paddle::dialect::BilinearInterpOp> {
 public:
  using pir::OpRewritePattern<
      paddle::dialect::BilinearInterpOp>::OpRewritePattern;
  bool MatchAndRewrite(paddle::dialect::BilinearInterpOp op,
                       pir::PatternRewriter &rewriter) const override {
    if (op->HasAttribute(kCanRunTrtAttr) &&
        op->attribute<pir::BoolAttribute>(kCanRunTrtAttr).data()) {
      return false;
    }
    const std::vector<std::string> required_attrs = {"data_format",
                                                     "interp_method",
                                                     "align_corners",
                                                     "scale",
                                                     "out_h",
                                                     "out_w"};
    for (const auto &attr : required_attrs) {
      if (!op->HasAttribute(attr)) {
        VLOG(3) << "BilinearInterpV2 " << attr << " attribute does not exist";
        return false;
      }
    }
    auto data_format =
        op->attribute<pir::StrAttribute>("data_format").AsString();
    if (data_format != "NCHW" && data_format != "NHWC") {
      VLOG(3) << "BilinearInterpV2: data format must be NCHW or NHWC";
      return false;
    }
    auto interp_method =
        op->attribute<pir::StrAttribute>("interp_method").AsString();
    if (interp_method != "bilinear") {
      VLOG(3) << "The interp_method of BilinearInterpV2 is not bilinear";
      return false;
    }
#if IS_TRT_VERSION_GE(8200)
    auto size_tensor = op.operand_source(2);
    if (size_tensor.impl()) {
      auto size_tensor_type = size_tensor.type();
      if (size_tensor_type.isa<pir::VectorType>()) {
        auto vector_type = size_tensor.type().dyn_cast<pir::VectorType>();
        if (vector_type.size() == 2) {
          op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));
          return true;
        }
      }
    }
#else
    auto size_tensor = op.operand_source(2);
    if (size_tensor.impl() != nullptr) {
      VLOG(3) << "The Paddle-TRT doesn't support the SizeTensor for "
                 "BilinearInterpV2";
      return false;
    }
#endif
    pir::Value scale_tensor = op.operand_source(3);

    bool has_scale_input = false;
    if (scale_tensor) {
      has_scale_input = true;
    }

    if (has_scale_input) {
      VLOG(3) << "BilinearInterpV2 has scale input can not into trt,support "
                 "scale attribute into trt";
      return false;
    }
    if (!has_scale_input && op->HasAttribute("scale")) {
      std::vector<float> scale;
      auto scale_attr = op->attribute<pir::ArrayAttribute>("scale");
      for (const auto &attr : scale_attr.AsVector()) {
        scale.push_back(attr.dyn_cast<pir::FloatAttribute>().data());
      }
      if (scale.size() <= 1) {
        if (!op->HasAttribute("out_h") || !op->HasAttribute("out_w")) {
          VLOG(3) << "BilinearInterpV2 doesn't have scale_tensor and the scale "
                     "size <=1 and without"
                     "out_h / out_w, it will return false";
          return false;
        }
        auto out_h = op->attribute<pir::Int32Attribute>("out_h").data();
        auto out_w = op->attribute<pir::Int32Attribute>("out_w").data();
        if (!(out_h <= 0 && out_w <= 0)) {
          if (out_h <= 0) {
            VLOG(3) << "BilinearInterpV2 out_h must be greater than 0 if scale "
                       "is not set.";
            return false;
          }
          if (out_w <= 0) {
            VLOG(3) << "BilinearInterpV2 out_w must be greater than 0 if scale "
                       "is not set.";
            return false;
          }
        }
      } else {
        for (size_t i = 0; i < scale.size(); i++) {
          if (scale[i] <= 0) {
            VLOG(3) << "BilinearInterpV2  dynamic shape not support Attr(scale["
                    << i << "]" << scale[i]
                    << " less than 1 and Input(Scale) Vector not set.";
            return false;
          }
        }
      }
    }

    op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));
    return true;
  }
};

class NearestInterV2Pattern
    : public pir::OpRewritePattern<paddle::dialect::NearestInterpOp> {
 public:
  using pir::OpRewritePattern<
      paddle::dialect::NearestInterpOp>::OpRewritePattern;
  bool MatchAndRewrite(paddle::dialect::NearestInterpOp op,
                       pir::PatternRewriter &rewriter) const override {
    if (op->HasAttribute(kCanRunTrtAttr) &&
        op->attribute<pir::BoolAttribute>(kCanRunTrtAttr).data()) {
      return false;
    }
    const std::vector<std::string> required_attrs = {"data_format",
                                                     "interp_method",
                                                     "align_corners",
                                                     "scale",
                                                     "out_h",
                                                     "out_w"};
    for (const auto &attr : required_attrs) {
      if (!op->HasAttribute(attr)) {
        VLOG(3) << "NearestInterV2 " << attr << " attribute does not exist";
        return false;
      }
    }

    auto data_format =
        op->attribute<pir::StrAttribute>("data_format").AsString();
    if (data_format != "NCHW" && data_format != "NHWC") {
      VLOG(3) << "NearestInterV2: data format must be NCHW or NHWC";
      return false;
    }
    auto interp_method =
        op->attribute<pir::StrAttribute>("interp_method").AsString();
    if (interp_method != "nearest") {
      VLOG(3) << "The interp_method of NearestInterV2 is not nearest";
      return false;
    }

#if IS_TRT_VERSION_GE(8200)
    pir::Value size_tensor = op.operand_source(2);
    if (size_tensor) {
      auto size_tensor_type = size_tensor.type();
      if (size_tensor_type.isa<pir::VectorType>()) {
        auto vector_type = size_tensor.type().dyn_cast<pir::VectorType>();
        if (vector_type.size() == 2) {
          op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));
          return true;
        }
      }
    }
#endif

    if (op->HasAttribute("scale")) {
      std::vector<float> scale;
      auto scale_attr = op->attribute<pir::ArrayAttribute>("scale");
      for (const auto &attr : scale_attr.AsVector()) {
        scale.push_back(attr.dyn_cast<pir::FloatAttribute>().data());
      }
      auto out_h = op->attribute<pir::Int32Attribute>("out_h").data();
      auto out_w = op->attribute<pir::Int32Attribute>("out_w").data();
      if (!(out_h > 0 && out_w > 0)) {
        if (scale.size() < 2) {
          VLOG(3) << "NearestInterV2 scale attribute size < 2";
          return false;
        }
        if (scale[0] <= 0.f || scale[1] <= 0.f) {
          VLOG(3) << "scale factor must be greater than 0 if out_h or out_w is "
                     "not set.";
          return false;
        }
      }
    }

    op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));
    return true;
  }
};

class ClipPattern : public pir::OpRewritePattern<paddle::dialect::ClipOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::ClipOp>::OpRewritePattern;
  bool MatchAndRewrite(paddle::dialect::ClipOp op,
                       pir::PatternRewriter &rewriter) const override {
    pir::Value x = op.operand_source(0);
    auto x_shape = pir::GetShapeFromValue(x);
    if (x_shape.size() == 0) {
      VLOG(3) << " clip op does not support input's dim is 0 in tensorrt.";
      return false;
    }
    auto min_tensor = op.operand_source(1);
    if (!min_tensor) {
      VLOG(3) << "clip op does not have input min tensor";
      return false;
    }
    auto max_tensor = op.operand_source(2);
    if (!max_tensor) {
      VLOG(3) << "clip op does not have input max tensor";
      return false;
    }
    op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));
    return true;
  }
};

class GridSampleOpPattern
    : public pir::OpRewritePattern<paddle::dialect::GridSampleOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::GridSampleOp>::OpRewritePattern;
  bool MatchAndRewrite(paddle::dialect::GridSampleOp op,
                       pir::PatternRewriter &rewriter) const override {
    if (op->HasAttribute(kCanRunTrtAttr) &&
        op->attribute<pir::BoolAttribute>(kCanRunTrtAttr).data()) {
      return false;
    }
#if IS_TRT_VERSION_LT(8510)
    VLOG(3) << "grid_sample is not supported when TensorRT < 8.5.1";
    return false;
#else
    if (!op->HasAttribute("mode") || !op->HasAttribute("padding_mode") ||
        !op->HasAttribute("align_corners")) {
      VLOG(3)
          << "grid_sample need attributes: mode, padding_mode, align_corners";
      return false;
    }
    auto x = op.operand_source(0);
    auto grid = op.operand_source(1);
    auto x_shape = pir::GetShapeFromValue(x);
    auto grid_shape = pir::GetShapeFromValue(grid);

    if (x_shape.size() != 4 || grid_shape.size() != 4) {
      VLOG(3) << "The input and grid tensors must be shape tensors of rank 4 "
                 "when using TRT GridSample layer.";
      return false;
    }
#endif
    op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));
    return true;
  }
};

class StackOpPattern : public pir::OpRewritePattern<paddle::dialect::StackOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::StackOp>::OpRewritePattern;
  bool MatchAndRewrite(paddle::dialect::StackOp op,
                       pir::PatternRewriter &rewriter) const override {
    if (op->HasAttribute(kCanRunTrtAttr) &&
        op->attribute<pir::BoolAttribute>(kCanRunTrtAttr).data()) {
      return false;
    }

    pir::Value x = op.operand_source(0);
    int rank = 1;
    auto x_type = x.type();
    if (x_type.isa<pir::VectorType>()) {
      rank = x_type.dyn_cast<pir::VectorType>().size();
    } else {
      auto x_shape = pir::GetShapeFromValue(x);
      rank = x_shape.size();
    }

    int axis = 1;
    if (op->HasAttribute("axis")) {
      axis = op->attribute<pir::Int32Attribute>("axis").data();
    } else {
      axis = -1;
    }
    if (axis > rank || axis < -(rank + 1)) {
      VLOG(3) << "Invalid axis value: " << axis
              << ". Axis should be in range [-" << (rank + 1) << ", " << rank
              << "], where rank is " << rank << ".";
      return false;
    }
    op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));
    return true;
  }
};

class WherePattern : public pir::OpRewritePattern<paddle::dialect::WhereOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::WhereOp>::OpRewritePattern;
  bool MatchAndRewrite(paddle::dialect::WhereOp op,
                       pir::PatternRewriter &rewriter) const override {
    if (op->HasAttribute(kCanRunTrtAttr) &&
        op.attribute<pir::BoolAttribute>(kCanRunTrtAttr).data()) {
      return false;
    }
    pir::Value x = op.operand_source(1);
    pir::Value y = op.operand_source(2);
    if (x == nullptr || y == nullptr) {
      VLOG(3) << "pd_op.where x or y tensor value is null";
      return false;
    }
#if IS_TRT_VERSION_LT(8400)
    VLOG(3) << "where is not supported when TensorRT < 8.4";
    return false;
#endif

    op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));
    return true;
  }
};

class EqualOpPattern : public pir::OpRewritePattern<paddle::dialect::EqualOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::EqualOp>::OpRewritePattern;
  bool MatchAndRewrite(paddle::dialect::EqualOp op,
                       pir::PatternRewriter &rewriter) const override {
    if (op->HasAttribute(kCanRunTrtAttr) &&
        op.attribute<pir::BoolAttribute>(kCanRunTrtAttr).data()) {
      return false;
    }
#if IS_TRT_VERSION_LT(8600)
    pir::Value x = op.operand_source(0);
    auto x_type = x.type().dyn_cast<paddle::dialect::DenseTensorType>();
    auto x_shape = x_type.dims();
    int dims = x_shape.size();
    if (dims < 1) {
      VLOG(3)
          << "pd_op.equal op does not support 0 dim input when TensorRT < 8.6.";
      return false;
    }
#endif

    op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));
    return true;
  }
};

class NotEqualOpPattern
    : public pir::OpRewritePattern<paddle::dialect::NotEqualOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::NotEqualOp>::OpRewritePattern;
  bool MatchAndRewrite(paddle::dialect::NotEqualOp op,
                       pir::PatternRewriter &rewriter) const override {
    if (op->HasAttribute(kCanRunTrtAttr) &&
        op.attribute<pir::BoolAttribute>(kCanRunTrtAttr).data()) {
      return false;
    }
#if IS_TRT_VERSION_LT(8600)
    pir::Value x = op.operand_source(0);
    auto x_type = x.type().dyn_cast<paddle::dialect::DenseTensorType>();
    auto x_shape = x_type.dims();
    int dims = x_shape.size();
    if (dims < 1) {
      VLOG(3) << "pd_op.not_equal op does not support 0 dim input when "
                 "TensorRT < 8.6.";
      return false;
    }
#endif

    op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));
    return true;
  }
};

template <typename OpType>
class BitwiseCommonOpPattern : public pir::OpRewritePattern<OpType> {
 public:
  using pir::OpRewritePattern<OpType>::OpRewritePattern;

  bool MatchAndRewrite(OpType op,
                       pir::PatternRewriter &rewriter) const override {
    if (op->HasAttribute(kCanRunTrtAttr) &&
        op->template attribute<pir::BoolAttribute>(kCanRunTrtAttr).data()) {
      return false;
    }
    pir::Value input_operand = op.operand_source(0);
    auto input_type = pir::GetDataTypeFromValue(input_operand);
    if (!input_type.isa<pir::BoolType>()) {
      if constexpr (std::is_same_v<OpType, paddle::dialect::BitwiseAndOp>) {
        VLOG(3) << "the bitwise_and only supports input of BOOL.";
      } else {
        VLOG(3) << "the bitwise_or only supports input of BOOL.";
      }
      return false;
    }
    op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));
    return true;
  }
};
using BitwiseAndOpPattern =
    BitwiseCommonOpPattern<paddle::dialect::BitwiseAndOp>;
using BitwiseOrOpPattern = BitwiseCommonOpPattern<paddle::dialect::BitwiseOrOp>;

class BitwiseNotOpPattern
    : public pir::OpRewritePattern<paddle::dialect::BitwiseNotOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::BitwiseNotOp>::OpRewritePattern;

  bool MatchAndRewrite(paddle::dialect::BitwiseNotOp op,
                       pir::PatternRewriter &rewriter) const override {
    if (op->HasAttribute(kCanRunTrtAttr) &&
        op->attribute<pir::BoolAttribute>(kCanRunTrtAttr).data()) {
      return false;
    }

    pir::Value input_operand = op.operand_source(0);
    auto input_type = pir::GetDataTypeFromValue(input_operand);
    if (input_type.isa<pir::Int8Type>() || input_type.isa<pir::UInt8Type>()) {
      VLOG(3) << "INT8 / UINT8 type convert to TRT is not supported.";
      return false;
    }
#if IS_TRT_VERSION_LT(8600)
    pir::Value x = op.operand_source(0);
    auto x_type = x.type().dyn_cast<paddle::dialect::DenseTensorType>();
    auto x_shape = x_type.dims();
    int dims = x_shape.size();
    if (dims < 1) {
      VLOG(3) << "BOOL type does not support 0-dim input when TensorRT < 8.6.";
      return false;
    }
#endif
    op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));
    return true;
  }
};

class FullLikeOpPattern
    : public pir::OpRewritePattern<paddle::dialect::FullLikeOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::FullLikeOp>::OpRewritePattern;
  bool MatchAndRewrite(paddle::dialect::FullLikeOp op,
                       pir::PatternRewriter &rewriter) const override {
    if (op->HasAttribute(kCanRunTrtAttr) &&
        op->attribute<pir::BoolAttribute>(kCanRunTrtAttr).data()) {
      return false;
    }
    pir::Value x = op.operand_source(0);
    auto x_dtype = pir::GetDataTypeFromValue(x);
    bool hasAttr = op->HasAttribute("dtype");
    auto dtype =
        op->attribute<paddle::dialect::DataTypeAttribute>("dtype").data();

    if (dtype == phi::DataType::BOOL ||
        (!hasAttr && x_dtype.isa<pir::BoolType>())) {
      op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));
      VLOG(3) << "the pd_op.full_like supports input of BOOL by trt8.4 above";
      return true;
    }

    op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));
    return true;
  }
};

class FullWithTensorPattern
    : public pir::OpRewritePattern<paddle::dialect::FullWithTensorOp> {
 public:
  using pir::OpRewritePattern<
      paddle::dialect::FullWithTensorOp>::OpRewritePattern;
  bool MatchAndRewrite(paddle::dialect::FullWithTensorOp op,
                       pir::PatternRewriter &rewriter) const override {
    if (op->HasAttribute(kCanRunTrtAttr) &&
        op.attribute<pir::BoolAttribute>(kCanRunTrtAttr).data()) {
      return false;
    }
    pir::Value value = op.operand_source(0);
    if (value == nullptr) {
      VLOG(3) << "pd_op.full_with_tensor value is null";
      return false;
    }
#if IS_TRT_VERSION_LT(8500)
    if (pir::GetDefiningOpForInput(op, 1)
            ->isa<paddle::dialect::FullIntArrayOp>()) {
      paddle::dialect::FullIntArrayOp full_int_array =
          pir::GetDefiningOpForInput(op, 1)
              ->dyn_cast<paddle::dialect::FullIntArrayOp>();
      auto shape_attr = full_int_array->attribute<pir::ArrayAttribute>("value");
      if (shape_attr.size() == 1) {
        VLOG(3) << "pd_op.full_with_tensor shape is not support when TensorRT "
                   "< 8.5.0";
        return false;
      }
    } else {
      pir::Value shape = op.operand_source(1);
      if (shape != nullptr) {
        VLOG(3) << "pd_op.full_with_tensor shape is not support when TensorRT "
                   "< 8.5.0";
        return false;
      }
    }
#endif
    auto dtype =
        op->attribute<paddle::dialect::DataTypeAttribute>("dtype").data();
    if (dtype != phi::DataType::INT32 && dtype != phi::DataType::INT64 &&
        dtype != phi::DataType::FLOAT32) {
      VLOG(3) << "pd_op.full_with_tensor only support int32, int64, float32";
      return false;
    }

    op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));
    return true;
  }
};

class TakeAlongAxisOpPattern
    : public pir::OpRewritePattern<paddle::dialect::TakeAlongAxisOp> {
 public:
  using pir::OpRewritePattern<
      paddle::dialect::TakeAlongAxisOp>::OpRewritePattern;

  bool MatchAndRewrite(paddle::dialect::TakeAlongAxisOp op,
                       pir::PatternRewriter &rewriter) const override {
    if (op->HasAttribute(kCanRunTrtAttr) &&
        op->attribute<pir::BoolAttribute>(kCanRunTrtAttr).data()) {
      return false;
    }
    pir::Value x_value = op.operand_source(0);
    auto x_shape = pir::GetShapeFromValue(x_value);
    pir::Value index_value = op.operand_source(1);
    auto index_shape = pir::GetShapeFromValue(index_value);
    if (x_shape.size() != index_shape.size()) {
      VLOG(3) << "TakeAlongAxis op Index input dims size ["
              << index_shape.size() << "] is not equal to input dims size ["
              << x_shape.size() << "].";
      return false;
    }

    op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));
    return true;
  }
};

class StridedSliceOpPattern
    : public pir::OpRewritePattern<paddle::dialect::StridedSliceOp> {
 public:
  using pir::OpRewritePattern<
      paddle::dialect::StridedSliceOp>::OpRewritePattern;
  bool MatchAndRewrite(paddle::dialect::StridedSliceOp op,
                       pir::PatternRewriter &rewriter) const override {
    if (op->HasAttribute(kCanRunTrtAttr) &&
        op.attribute<pir::BoolAttribute>(kCanRunTrtAttr).data()) {
      return false;
    }
    if (!op->HasAttribute("axes")) {
      VLOG(3) << "The necessary attribute of the pd_op.strided_slice operator "
                 "axes are missing.";
      return false;
    }
    if (!op.operand_source(1) || !op.operand_source(2) ||
        !op.operand_source(3)) {
      VLOG(3) << "pd_op.strided_slice must has starts,ends and strides input";
      return false;
    }
    if (!pir::GetDefiningOpForInput(op, 1)
             ->isa<paddle::dialect::FullIntArrayOp>() ||
        !pir::GetDefiningOpForInput(op, 2)
             ->isa<paddle::dialect::FullIntArrayOp>() ||
        !pir::GetDefiningOpForInput(op, 3)
             ->isa<paddle::dialect::FullIntArrayOp>()) {
      VLOG(3) << "pd_op.strided_slice's starts/ends/strides input must be "
                 "constant value";
      return false;
    }
    op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));
    return true;
  }
};

class TopkOpPattern : public pir::OpRewritePattern<paddle::dialect::TopkOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::TopkOp>::OpRewritePattern;
  bool MatchAndRewrite(paddle::dialect::TopkOp op,
                       pir::PatternRewriter &rewriter) const override {
    if (op->HasAttribute(kCanRunTrtAttr) &&
        op.attribute<pir::BoolAttribute>(kCanRunTrtAttr).data()) {
      return false;
    }
    if (!op->HasAttribute("axis")) {
      VLOG(3) << "pd_op.topk must has axis attribute";
      return false;
    }
    if (!pir::GetDefiningOpForInput(op, 1)->isa<paddle::dialect::FullOp>()) {
      VLOG(3) << "The 'k' input of pd_op.topk must be an integer";
      return false;
    }

    if (op->HasAttribute("sorted")) {
      bool sorted = op->attribute<pir::BoolAttribute>("sorted").data();
      if (!sorted) {
        VLOG(3)
            << "pd_op.topk does not support results not sorted in tensorrt.";
        return false;
      }
    }
    op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));
    return true;
  }
};

class CumsumOpPattern
    : public pir::OpRewritePattern<paddle::dialect::CumsumOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::CumsumOp>::OpRewritePattern;
  bool MatchAndRewrite(paddle::dialect::CumsumOp op,
                       pir::PatternRewriter &rewriter) const override {
    if (op->HasAttribute(kCanRunTrtAttr) &&
        op.attribute<pir::BoolAttribute>(kCanRunTrtAttr).data()) {
      return false;
    }

    if (!pir::GetDefiningOpForInput(op, 1)->isa<paddle::dialect::FullOp>()) {
      VLOG(3) << "The 'axis' input of pd_op.cumsum must be an integer";
      return false;
    }

    op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));
    return true;
  }
};

bool CheckSetValue(const pir::Operation *op, int starts_input_loc = 1) {
  paddle::dialect::FullIntArrayOp starts_defining_op =
      pir::GetDefiningOpForInput(op, starts_input_loc)
          ->dyn_cast<paddle::dialect::FullIntArrayOp>();
  paddle::dialect::FullIntArrayOp ends_defining_op =
      pir::GetDefiningOpForInput(op, starts_input_loc + 1)
          ->dyn_cast<paddle::dialect::FullIntArrayOp>();
  paddle::dialect::FullIntArrayOp steps_defining_op =
      pir::GetDefiningOpForInput(op, starts_input_loc + 2)
          ->dyn_cast<paddle::dialect::FullIntArrayOp>();
  if (!starts_defining_op || !ends_defining_op || !steps_defining_op) {
    VLOG(3) << "SetValueOp Input : starts/ends/steps, is not initialized with "
               "constant "
               "value, this op will not be translated to TRT Layer.";
    return false;
  }
  auto starts = starts_defining_op->attribute<pir::ArrayAttribute>("value");
  auto ends = ends_defining_op->attribute<pir::ArrayAttribute>("value");
  auto steps = steps_defining_op->attribute<pir::ArrayAttribute>("value");
  auto axes = op->attribute<pir::ArrayAttribute>("axes");
  if (starts.size() != 1UL || ends.size() != 1UL || steps.size() != 1UL ||
      axes.size() != 1UL) {
    VLOG(3) << "the set_value op"
            << "has more than one element in starts/ends/steps/axes/values, it "
               "can not "
               "enter into trt.";
    return false;
  }
  auto decrease_axes = op->attribute<pir::ArrayAttribute>("decrease_axes");
  if (decrease_axes.size() != 0) {
    VLOG(3) << "the set_value op doesn't support decrease_axes attribute "
               "currently, it can not "
               "enter into trt.";
    return false;
  }
  return true;
}

bool SetValueOpMatchAndRewrite(const pir::Operation *op) {
  if (op->HasAttribute(kCanRunTrtAttr) &&
      op->attribute<pir::BoolAttribute>(kCanRunTrtAttr).data()) {
    return false;
  }
  bool in_trt = CheckSetValue(op);
  if (!in_trt) {
    return false;
  }

  auto values = op->attribute<pir::ArrayAttribute>("values").AsVector();
  if (values.size() != 1UL) {
    VLOG(3) << "the set_value op"
            << "has more than one element in values, it can not "
               "enter into trt.";
    return false;
  }
  auto value = values[0];
  auto value_dtype =
      value.dyn_cast<paddle::dialect::ScalarAttribute>().data().dtype();
  if (value_dtype != phi::DataType::FLOAT32 &&
      value_dtype != phi::DataType::FLOAT64) {
    VLOG(3) << "SetValueOp only support float32/float64 value when translate "
               "to trt.";
    return false;
  }
  return true;
}

class SetValueOpPattern
    : public pir::OpRewritePattern<paddle::dialect::SetValueOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::SetValueOp>::OpRewritePattern;
  bool MatchAndRewrite(paddle::dialect::SetValueOp op,
                       pir::PatternRewriter &rewriter) const override {
    bool in_trt = SetValueOpMatchAndRewrite(op);
    if (!in_trt) {
      return false;
    }
    op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));
    return true;
  }
};

class SetValue_OpPattern
    : public pir::OpRewritePattern<paddle::dialect::SetValue_Op> {
 public:
  using pir::OpRewritePattern<paddle::dialect::SetValue_Op>::OpRewritePattern;
  bool MatchAndRewrite(paddle::dialect::SetValue_Op op,
                       pir::PatternRewriter &rewriter) const override {
    bool in_trt = SetValueOpMatchAndRewrite(op);
    if (!in_trt) {
      return false;
    }
    op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));
    return true;
  }
};

class SetValueWithTensorOpPattern
    : public pir::OpRewritePattern<paddle::dialect::SetValueWithTensorOp> {
 public:
  using pir::OpRewritePattern<
      paddle::dialect::SetValueWithTensorOp>::OpRewritePattern;
  bool MatchAndRewrite(paddle::dialect::SetValueWithTensorOp op,
                       pir::PatternRewriter &rewriter) const override {
    if (op->HasAttribute(kCanRunTrtAttr) &&
        op.attribute<pir::BoolAttribute>(kCanRunTrtAttr).data()) {
      return false;
    }
    bool in_trt = CheckSetValue(op, 2);
    if (!in_trt) {
      return false;
    }
    pir::Value values = op.operand_source(1);
    auto values_dtype = pir::GetDataTypeFromValue(values);
    if (!values_dtype.isa<pir::Float32Type>() &&
        !values_dtype.isa<pir::Float64Type>()) {
      VLOG(3) << "SetValueWithTensorOp only support float32/float64 value when "
                 "translate "
                 "to trt.";
      return false;
    }
    op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));
    return true;
  }
};

class SetValueWithTensor_OpPattern
    : public pir::OpRewritePattern<paddle::dialect::SetValueWithTensor_Op> {
 public:
  using pir::OpRewritePattern<
      paddle::dialect::SetValueWithTensor_Op>::OpRewritePattern;
  bool MatchAndRewrite(paddle::dialect::SetValueWithTensor_Op op,
                       pir::PatternRewriter &rewriter) const override {
    if (op->HasAttribute(kCanRunTrtAttr) &&
        op.attribute<pir::BoolAttribute>(kCanRunTrtAttr).data()) {
      return false;
    }
    bool in_trt = CheckSetValue(op, 2);
    if (!in_trt) {
      return false;
    }
    pir::Value values = op.operand_source(1);
    auto values_dtype = pir::GetDataTypeFromValue(values);
    if (!values_dtype.isa<pir::Float32Type>() &&
        !values_dtype.isa<pir::Float64Type>()) {
      VLOG(3) << "SetValueWithTensorOp only support float32/float64 value when "
                 "translate "
                 "to trt.";
      return false;
    }
    op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));
    return true;
  }
};

class OneHotOpPattern
    : public pir::OpRewritePattern<paddle::dialect::OneHotOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::OneHotOp>::OpRewritePattern;
  bool MatchAndRewrite(paddle::dialect::OneHotOp op,
                       pir::PatternRewriter &rewriter) const override {
    if (op->HasAttribute(kCanRunTrtAttr) &&
        op.attribute<pir::BoolAttribute>(kCanRunTrtAttr).data()) {
      return false;
    }
#if IS_TRT_VERSION_LT(8510)
    VLOG(3) << "pd_op.one_hot is not supported when TensorRT<8.5.1";
    return false;
    pir::Value input = op.operand_source(0);
    auto input_type = pir::GetDataTypeFromValue(input);
    if (!input_type.isa<pir::Float32Type>() ||
        !input_type.isa<pir::Int32Type>() ||
        !input_type.isa<pir::Int64Type>()) {
      VLOG(3) << "pd_op.one_hot only support int32,int64,float.";
      return false;
    }
#endif

    op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));
    return true;
  }
};

class TemporalShiftOpPattern
    : public pir::OpRewritePattern<paddle::dialect::TemporalShiftOp> {
 public:
  using pir::OpRewritePattern<
      paddle::dialect::TemporalShiftOp>::OpRewritePattern;

  bool MatchAndRewrite(paddle::dialect::TemporalShiftOp op,
                       pir::PatternRewriter &rewriter) const override {
    if (op->HasAttribute(kCanRunTrtAttr) &&
        op.attribute<pir::BoolAttribute>(kCanRunTrtAttr).data()) {
      return false;
    }
    if (!op->HasAttribute("shift_ratio") || !op->HasAttribute("seg_num")) {
      VLOG(3) << "temporal shift need attributes : shift_ratio and seg_num";
      return false;
    }
    op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));
    return true;
  }
};

class FusedBiasDropoutResidualLayerNormOpPattern
    : public pir::OpRewritePattern<
          paddle::dialect::FusedBiasDropoutResidualLayerNormOp> {
 public:
  using pir::OpRewritePattern<
      paddle::dialect::FusedBiasDropoutResidualLayerNormOp>::OpRewritePattern;

  bool MatchAndRewrite(paddle::dialect::FusedBiasDropoutResidualLayerNormOp op,
                       pir::PatternRewriter &rewriter) const override {
    if (op->HasAttribute(kCanRunTrtAttr) &&
        op->attribute<pir::BoolAttribute>(kCanRunTrtAttr).data()) {
      return false;
    }
    auto dropout_rate_attr = op->attribute<pir::FloatAttribute>("dropout_rate");
    float dropout_rate = dropout_rate_attr.data();
    if (dropout_rate != 0.0f) {
      VLOG(3) << "preln_residual_bias trt layer can not work with "
                 "fused_bias_dropout_residual_layer_norm op in which the "
                 "dropout_rate != 0, stop convert";
      return false;
    }
    op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));
    return true;
  }
};

class InstanceNormOpPattern
    : public pir::OpRewritePattern<paddle::dialect::InstanceNormOp> {
 public:
  using pir::OpRewritePattern<
      paddle::dialect::InstanceNormOp>::OpRewritePattern;
  bool MatchAndRewrite(paddle::dialect::InstanceNormOp op,
                       pir::PatternRewriter &rewriter) const override {
    if (op->HasAttribute(kCanRunTrtAttr) &&
        op->attribute<pir::BoolAttribute>(kCanRunTrtAttr).data()) {
      return false;
    }
    pir::Value x = op.operand_source(0);
    auto x_type = x.type().dyn_cast<paddle::dialect::DenseTensorType>();
    auto x_shape = x_type.dims();
    if (x_shape.size() != 4) {
      VLOG(3) << "The instance_norm op only support 4-dimensional input in "
                 "tensorrt.";
      return false;
    }

    pir::Value scale = op.operand_source(1);
    pir::Value bias = op.operand_source(2);
    if (scale.impl() == nullptr || bias.impl() == nullptr) {
      VLOG(3) << "instance_norm op's scale and bias should not be null in "
                 "tensorrt.";
      return false;
    }

    op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));
    return true;
  }
};

class EinsumOpPattern
    : public pir::OpRewritePattern<paddle::dialect::EinsumOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::EinsumOp>::OpRewritePattern;
  bool MatchAndRewrite(paddle::dialect::EinsumOp op,
                       pir::PatternRewriter &rewriter) const override {
    if (op->HasAttribute(kCanRunTrtAttr) &&
        op.attribute<pir::BoolAttribute>(kCanRunTrtAttr).data()) {
      return false;
    }
    std::string equation =
        op->attribute<pir::StrAttribute>("equation").AsString();
    if (equation.empty()) {
      VLOG(3) << "Einsum equation is empty";
      return false;
    }

    auto operands = op->operands();
    if (operands.size() > 2) {
      VLOG(3) << "TensorRT currently supports up to 2 input tensors"
              << "to einsum but operation had" << operands.size()
              << "input tensors !";
      return false;
    }

    if (equation.find("...") != std::string::npos) {
      VLOG(3) << "TensorRT currently does not support ellipses !";
      return false;
    }
    op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));
    return true;
  }
};

class PNormOpPattern : public pir::OpRewritePattern<paddle::dialect::PNormOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::PNormOp>::OpRewritePattern;

  bool MatchAndRewrite(paddle::dialect::PNormOp op,
                       pir::PatternRewriter &rewriter) const override {
    if (op->HasAttribute(kCanRunTrtAttr) &&
        op->attribute<pir::BoolAttribute>(kCanRunTrtAttr).data()) {
      return false;
    }
    if (!op->HasAttribute("asvector") || !op->HasAttribute("axis") ||
        !op->HasAttribute("porder") || !op->HasAttribute("keepdim")) {
      VLOG(3) << "p_norm op needs attributes: asvector, porder, axis, keepdim.";
      return false;
    }
    bool asvector = op->attribute<pir::BoolAttribute>("asvector").data();
    int axis = op->attribute<pir::Int32Attribute>("axis").data();
    float porder = op->attribute<pir::FloatAttribute>("porder").data();

    if (asvector || porder != 2.0f || axis != -1) {
      VLOG(3) << "p_norm op only supports asvector=False, porder=2, axis=-1.";
      return false;
    }
    op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));
    return true;
  }
};

class AffineChannelOpPattern
    : public pir::OpRewritePattern<paddle::dialect::AffineChannelOp> {
 public:
  using pir::OpRewritePattern<
      paddle::dialect::AffineChannelOp>::OpRewritePattern;
  bool MatchAndRewrite(paddle::dialect::AffineChannelOp op,
                       pir::PatternRewriter &rewriter) const override {
    if (op->HasAttribute(kCanRunTrtAttr) &&
        op->attribute<pir::BoolAttribute>(kCanRunTrtAttr).data()) {
      return false;
    }
    if (!op->HasAttribute("data_layout")) {
      VLOG(3) << "pd_op.affine_channel must has data_layout";
      return false;
    }
    pir::Value x = op.operand_source(0);
    auto x_shape = pir::GetShapeFromValue(x);
    if (x_shape.size() == 2) {
      VLOG(3) << "pd_op.affine_channel x.shape can not be 2";
      return false;
    }

    op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));
    return true;
  }
};

class PreluOpPattern : public pir::OpRewritePattern<paddle::dialect::PreluOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::PreluOp>::OpRewritePattern;
  bool MatchAndRewrite(paddle::dialect::PreluOp op,
                       pir::PatternRewriter &rewriter) const override {
    if (op->HasAttribute(kCanRunTrtAttr) &&
        op.attribute<pir::BoolAttribute>(kCanRunTrtAttr).data()) {
      return false;
    }
    pir::Value alpha_var = op.operand_source(1);
    if (!alpha_var) {
      VLOG(3) << "Variable Alpha of prelu TRT converter not found.";
      return false;
    }
    op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));
    return true;
  }
};

class YoloBoxOpPattern
    : public pir::OpRewritePattern<paddle::dialect::YoloBoxOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::YoloBoxOp>::OpRewritePattern;
  bool MatchAndRewrite(paddle::dialect::YoloBoxOp op,
                       pir::PatternRewriter &rewriter) const override {
    if (op->HasAttribute(kCanRunTrtAttr) &&
        op->attribute<pir::BoolAttribute>(kCanRunTrtAttr).data()) {
      return false;
    }
    if (!op->HasAttribute("iou_aware") &&
        !op->HasAttribute("iou_aware_factor")) {
      VLOG(3)
          << "pd_op.yolo_box must has iou_aware and iou_aware_factor attribute";
      return false;
    }
    op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));
    return true;
  }
};

class FullBatchSizeLikeOpPattern
    : public pir::OpRewritePattern<paddle::dialect::FullBatchSizeLikeOp> {
 public:
  using pir::OpRewritePattern<
      paddle::dialect::FullBatchSizeLikeOp>::OpRewritePattern;
  bool MatchAndRewrite(paddle::dialect::FullBatchSizeLikeOp op,
                       pir::PatternRewriter &rewriter) const override {
    if (op->HasAttribute(kCanRunTrtAttr) &&
        op->attribute<pir::BoolAttribute>(kCanRunTrtAttr).data()) {
      return false;
    }
    if (!op->HasAttribute("input_dim_idx")) {
      VLOG(3) << "pd_op.full_batch_size_like must has input_dim_idx attribute";
      return false;
    }
    if (!op->HasAttribute("output_dim_idx")) {
      VLOG(3) << "pd_op.full_batch_size_like must has output_dim_idx attribute";
      return false;
    }
    if (!op->HasAttribute("shape")) {
      VLOG(3) << "pd_op.full_batch_size_like must has shape attribute";
      return false;
    }
    pir::Value input = op.operand_source(0);
    auto input_type = pir::GetDataTypeFromValue(input);
    if (!input_type.isa<pir::Float32Type>()) {
      VLOG(3) << "pd_op.full_batch_size_like only support float32.";
      return false;
    }

    op->set_attribute(kCanRunTrtAttr, rewriter.bool_attr(true));
    return true;
  }
};

class TrtOpMarkerPass : public pir::PatternRewritePass {
 public:
  TrtOpMarkerPass() : pir::PatternRewritePass("trt_op_marker_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);

#define ADD_PATTERN(OpName) \
  ps.Add(std::make_unique<OpName##OpPattern>(context));
    ADD_PATTERN(Matmul)
    ADD_PATTERN(BatchNorm)
    ADD_PATTERN(BatchNorm_)
    ADD_PATTERN(Softmax)
    ADD_PATTERN(Relu)
    ADD_PATTERN(FullIntArray)
    ADD_PATTERN(Reshape)
    ADD_PATTERN(Dropout)
    ADD_PATTERN(Bmm)
    ADD_PATTERN(Concat)
    ADD_PATTERN(Full)
    ADD_PATTERN(Fused_gemm_epilogue)
    ADD_PATTERN(Add)
    ADD_PATTERN(Silu)
    ADD_PATTERN(Conv2d)
    ADD_PATTERN(FusedConv2dAddAct)
    ADD_PATTERN(DepthwiseConv2d)
    ADD_PATTERN(Nonzero)
    ADD_PATTERN(Gelu)
    ADD_PATTERN(Relu6)
    ADD_PATTERN(Shape)
    ADD_PATTERN(Shape64)
    ADD_PATTERN(Expand)
    ADD_PATTERN(ExpandAs)
    ADD_PATTERN(Sigmoid)
    ADD_PATTERN(Sqrt)
    ADD_PATTERN(Hardsigmoid)
    ADD_PATTERN(Hardswish)
    ADD_PATTERN(AssignOut)
    ADD_PATTERN(Assign)
    ADD_PATTERN(Tile)
    ADD_PATTERN(Share_Data)
    ADD_PATTERN(Swish)
    ADD_PATTERN(Log)
    ADD_PATTERN(Floor)
    ADD_PATTERN(Roll)
    ADD_PATTERN(Elu)
    ADD_PATTERN(Selu)
    ADD_PATTERN(Stanh)
    ADD_PATTERN(Softplus)
    ADD_PATTERN(ThresholdedRelu)
    ADD_PATTERN(Flip)
    ADD_PATTERN(Mish)
    ADD_PATTERN(AssignValue)
    ADD_PATTERN(AssignValue_)
    ADD_PATTERN(LeakyRelu)
    ADD_PATTERN(LeakyRelu_)
    ADD_PATTERN(Anchor_Generator)
    ADD_PATTERN(Exp)
    ADD_PATTERN(Abs)
    ADD_PATTERN(Abs_)
    ADD_PATTERN(Cos)
    ADD_PATTERN(Sin)
    ADD_PATTERN(Logsigmoid)
    ADD_PATTERN(Embedding)
    ADD_PATTERN(Unbind)
    ADD_PATTERN(Cos)
    ADD_PATTERN(Sinh)
    ADD_PATTERN(Cosh)
    ADD_PATTERN(Asinh)
    ADD_PATTERN(Acosh)
    ADD_PATTERN(Atanh)
    ADD_PATTERN(Ceil)
    ADD_PATTERN(Rsqrt)
    ADD_PATTERN(Reciprocal)
    ADD_PATTERN(Erf)
    ADD_PATTERN(Isnan)
    ADD_PATTERN(Sign)
    ADD_PATTERN(Round)
    ADD_PATTERN(Numel)
    ADD_PATTERN(Pool3d)
    ADD_PATTERN(ShuffleChannel)
#if IS_TRT_VERSION_GE(8600)
    ADD_PATTERN(Layer_norm)
#endif
#undef ADD_PATTERN
    ps.Add(std::make_unique<Pool2dOpPattern>(context));
    ps.Add(std::make_unique<Conv2dTransposeOpPattern>(context));
    ps.Add(std::make_unique<DepthwiseConv2dTransposeOpPattern>(context));
    ps.Add(std::make_unique<DeformableConvOpPattern>(context));
    ps.Add(std::make_unique<ArangeOpPattern>(context));
    ps.Add(std::make_unique<LogicalNotOpPattern>(context));
    ps.Add(std::make_unique<BitwiseAndOpPattern>(context));
    ps.Add(std::make_unique<BitwiseOrOpPattern>(context));
    ps.Add(std::make_unique<BitwiseNotOpPattern>(context));
    ps.Add(std::make_unique<LogicalNot_OpPattern>(context));
    ps.Add(std::make_unique<LogicalOrOpPattern>(context));
    ps.Add(std::make_unique<LogicalOr_OpPattern>(context));
    ps.Add(std::make_unique<LogicalAndOpPattern>(context));
    ps.Add(std::make_unique<GroupNormOpPattern>(context));
    ps.Add(std::make_unique<TransposeOpPattern>(context));
    ps.Add(std::make_unique<GatherOpPattern>(context));
    ps.Add(std::make_unique<GatherNdOpPattern>(context));
    ps.Add(std::make_unique<ScaleOpPattern>(context));
    ps.Add(std::make_unique<UnsqueezeOpPattern>(context));
    ps.Add(std::make_unique<SqueezeOpPattern>(context));
    ps.Add(std::make_unique<Unsqueeze_OpPattern>(context));
    ps.Add(std::make_unique<EmbeddingOpPattern>(context));
    ps.Add(std::make_unique<UnbindOpPattern>(context));
    ps.Add(std::make_unique<SliceOpPattern>(context));
    ps.Add(std::make_unique<IndexSelectOpPattern>(context));
    ps.Add(std::make_unique<FlattenOpPattern>(context));
    ps.Add(std::make_unique<CastOpPattern>(context));
    ps.Add(std::make_unique<SplitOpPattern>(context));
    ps.Add(std::make_unique<SplitWithNumOpPattern>(context));
    ps.Add(std::make_unique<GreaterEqualOpPattern>(context));
    ps.Add(std::make_unique<GreaterEqual_OpPattern>(context));
    ps.Add(std::make_unique<GreaterThanOpPattern>(context));
    ps.Add(std::make_unique<LessEqualOpPattern>(context));
    ps.Add(std::make_unique<LessEqual_OpPattern>(context));
    ps.Add(std::make_unique<LessThanOpPattern>(context));
    ps.Add(std::make_unique<MultiplyOpPattern>(context));
    ps.Add(std::make_unique<SubtractOpPattern>(context));
    ps.Add(std::make_unique<DivideOpPattern>(context));
    ps.Add(std::make_unique<MinimumOpPattern>(context));
    ps.Add(std::make_unique<MaximumOpPattern>(context));
    ps.Add(std::make_unique<FloorDivideOpPattern>(context));
    ps.Add(std::make_unique<ElementwisePowOpPattern>(context));
    ps.Add(std::make_unique<MeanOpPattern>(context));
    ps.Add(std::make_unique<RemainderOpPattern>(context));
    ps.Add(std::make_unique<MulticlassNms3OpPattern>(context));
    ps.Add(std::make_unique<ArgmaxOpPattern>(context));
    ps.Add(std::make_unique<ArgminOpPattern>(context));
    ps.Add(std::make_unique<ArgsortOpPattern>(context));
    ps.Add(std::make_unique<MaxOpPattern>(context));
    ps.Add(std::make_unique<MinOpPattern>(context));
    ps.Add(std::make_unique<AllOpPattern>(context));
    ps.Add(std::make_unique<AnyOpPattern>(context));
    ps.Add(std::make_unique<SumOpPattern>(context));
    ps.Add(std::make_unique<BilinearInterpV2Pattern>(context));
    ps.Add(std::make_unique<NearestInterV2Pattern>(context));
    ps.Add(std::make_unique<ClipPattern>(context));
    ps.Add(std::make_unique<GridSampleOpPattern>(context));
    ps.Add(std::make_unique<StackOpPattern>(context));
    ps.Add(std::make_unique<TanhShrinkOpPattern>(context));
    ps.Add(std::make_unique<WherePattern>(context));
    ps.Add(std::make_unique<FullLikeOpPattern>(context));
    ps.Add(std::make_unique<FullWithTensorPattern>(context));
    ps.Add(std::make_unique<TakeAlongAxisOpPattern>(context));
    ps.Add(std::make_unique<StridedSliceOpPattern>(context));
    ps.Add(std::make_unique<TopkOpPattern>(context));
    ps.Add(std::make_unique<CumsumOpPattern>(context));
    ps.Add(std::make_unique<SetValueOpPattern>(context));
    ps.Add(std::make_unique<SetValue_OpPattern>(context));
    ps.Add(std::make_unique<SetValueWithTensorOpPattern>(context));
    ps.Add(std::make_unique<SetValueWithTensor_OpPattern>(context));
    ps.Add(std::make_unique<EqualOpPattern>(context));
    ps.Add(std::make_unique<NotEqualOpPattern>(context));
    ps.Add(std::make_unique<LogicalXorOpPattern>(context));
    ps.Add(std::make_unique<TanhOpPattern>(context));
    ps.Add(std::make_unique<CeluOpPattern>(context));
    ps.Add(std::make_unique<OneHotOpPattern>(context));
    ps.Add(std::make_unique<TemporalShiftOpPattern>(context));
    ps.Add(std::make_unique<InstanceNormOpPattern>(context));
    ps.Add(std::make_unique<EinsumOpPattern>(context));
    ps.Add(std::make_unique<PNormOpPattern>(context));
    ps.Add(std::make_unique<AffineChannelOpPattern>(context));
    ps.Add(std::make_unique<PreluOpPattern>(context));
    ps.Add(
        std::make_unique<FusedBiasDropoutResidualLayerNormOpPattern>(context));
    ps.Add(std::make_unique<YoloBoxOpPattern>(context));
    ps.Add(std::make_unique<FullBatchSizeLikeOpPattern>(context));
    return ps;
  }
};
}  // namespace

namespace pir {
std::unique_ptr<Pass> CreateTrtOpMarkerPass() {
  return std::make_unique<TrtOpMarkerPass>();
}
}  // namespace pir

REGISTER_IR_PASS(trt_op_marker_pass, TrtOpMarkerPass);
