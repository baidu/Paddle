// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,affine
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/pir/dialect/operator/interface/infer_symbolic_shape/unary_infer_sym.h"
#include "paddle/fluid/pir/dialect/operator/interface/infer_symbolic_shape/infer_sym_slice_utils.h"
#include "paddle/fluid/pir/dialect/operator/interface/infer_symbolic_shape/infer_sym_utils.h"

namespace {
std::vector<symbol::DimExpr> GetRealPadding(
    const std::vector<int> &origin_paddings,
    const bool global_pooling,
    const bool adaptive,
    const std::string padding_algorithm,
    const std::vector<symbol::DimExpr> data_dims,
    const std::vector<int> &strides,
    const std::vector<symbol::DimExpr> &kernel_size) {
  const auto &GetInitPadding = [&]() -> std::vector<symbol::DimExpr> {
    std::vector<symbol::DimExpr> res;
    // set padding size == data_dims.size() * 2
    if (origin_paddings.size() == data_dims.size()) {
      for (std::size_t i = 0; i < origin_paddings.size(); ++i) {
        res.emplace_back(symbol::DimExpr{origin_paddings.at(i)});
        res.emplace_back(symbol::DimExpr{origin_paddings.at(i)});
      }
    } else {
      PADDLE_ENFORCE_EQ(
          data_dims.size() * 2,
          origin_paddings.size(),
          common::errors::InvalidArgument(
              "Paddings size %d should be the same or twice as the "
              "pooling size %d.",
              origin_paddings.size(),
              data_dims.size() * 2));
      for (std::size_t i = 0; i < origin_paddings.size(); ++i) {
        res.emplace_back(symbol::DimExpr{origin_paddings.at(i)});
      }
    }
    return res;
  };

  std::vector<symbol::DimExpr> real_padding = GetInitPadding();

  const auto &UpdataPadding = [&]() {
    symbol::DimExpr one_dimexpr{1};
    symbol::DimExpr zero_dimexpr{0};
    // when padding_algorithm is "VALID" or "SAME"
    if (padding_algorithm == "SAME") {
      for (std::size_t i = 0; i < data_dims.size(); ++i) {
        symbol::DimExpr stride_dimexpr = symbol::DimExpr{strides[i]};

        symbol::DimExpr out_size =
            (data_dims[i] + stride_dimexpr - one_dimexpr) / stride_dimexpr;
        symbol::DimExprBuilder builder;
        symbol::DimExpr pad_sum =
            builder.Max((out_size - one_dimexpr) * stride_dimexpr +
                            kernel_size[i] - data_dims[i],
                        zero_dimexpr);
        symbol::DimExpr pad_0 = pad_sum / symbol::DimExpr{2};
        symbol::DimExpr pad_1 = pad_sum - pad_0;
        real_padding[i * 2] = pad_0;
        real_padding[i * 2 + 1] = pad_1;
      }
    } else if (padding_algorithm == "VALID") {
      real_padding.assign(real_padding.size(), zero_dimexpr);
    }

    // if global_pooling == true or adaptive == true, padding will be ignore
    if (global_pooling || adaptive) {
      real_padding.assign(real_padding.size(), zero_dimexpr);
    }
  };

  UpdataPadding();
  return real_padding;
}

symbol::ShapeOrDataDimExprs Pool2dRawInferSymbolicShape(
    pir::Operation *op,
    const std::vector<symbol::DimExpr> &kernel_size,
    pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));

  const auto &x_dims = x_shape_or_data.shape();
  PADDLE_ENFORCE_EQ(
      x_dims.size() == 4 || x_dims.size() == 5,
      true,
      common::errors::InvalidArgument(
          "the input of Op(pool) should be 4-D or 5-D Tensor. But "
          "received: %u-D Tensor.",
          x_dims.size()));

  PADDLE_ENFORCE_EQ(x_dims.size() - kernel_size.size(),
                    2U,
                    common::errors::InvalidArgument(
                        "the rank of input minus the size of kernel_size "
                        "must be equal to 2 in Op(pool). "
                        "But received: the rank of input is %d and the "
                        "rank of kernel_size is %d.",
                        x_dims.size(),
                        kernel_size.size()));

  std::vector<int> strides = [&]() {
    std::vector<int> res;
    const auto &stride_attr =
        op->attributes().at("strides").dyn_cast<pir::ArrayAttribute>();
    for (size_t i = 0; i < stride_attr.size(); i++) {
      res.emplace_back(
          stride_attr.at(i).dyn_cast<pir::Int32Attribute>().data());
    }
    return res;
  }();

  PADDLE_ENFORCE_EQ(
      kernel_size.size(),
      strides.size(),
      common::errors::InvalidArgument(
          "the rank of kernel_size and strides in Op(pool) must be equal. "
          "But received: the rank of kernel_size is %d and the rank of stride "
          "is %d.",
          kernel_size.size(),
          strides.size()));

  const std::string &data_format =
      op->attribute<pir::StrAttribute>("data_format").AsString();
  const bool channel_last = data_format == "NHWC" || data_format == "NDHWC";

  const auto &data_dims = [&]() -> std::vector<symbol::DimExpr> {
    if (channel_last) {
      return std::vector<symbol::DimExpr>(x_dims.begin() + 1, x_dims.end() - 1);
    } else {
      return std::vector<symbol::DimExpr>(x_dims.begin() + 2, x_dims.end());
    }
  }();

  bool global_pooling =
      op->attribute<pir::BoolAttribute>("global_pooling").data();
  bool adaptive = op->attribute<pir::BoolAttribute>("adaptive").data();
  std::string padding_algorithm =
      op->attribute<pir::StrAttribute>("padding_algorithm").AsString();

  const auto &real_paddings = [&]() -> std::vector<symbol::DimExpr> {
    std::vector<int> paddings;
    const auto &padding_attr =
        op->attributes().at("paddings").dyn_cast<pir::ArrayAttribute>();
    for (size_t i = 0; i < padding_attr.size(); i++) {
      paddings.emplace_back(
          padding_attr.at(i).dyn_cast<pir::Int32Attribute>().data());
    }
    return GetRealPadding(paddings,
                          global_pooling,
                          adaptive,
                          padding_algorithm,
                          data_dims,
                          strides,
                          kernel_size

    );
  }();

  const auto &real_kernel_size = [&]() -> std::vector<symbol::DimExpr> {
    if (global_pooling) {
      return data_dims;
    }
    return kernel_size;
  }();

  const auto &output_shape_or_data = [&]() -> symbol::ShapeOrDataDimExprs {
    std::vector<symbol::DimExpr> output_shape;
    bool ceil_mode = op->attribute<pir::BoolAttribute>("ceil_mode").data();
    if (adaptive) {
      output_shape.insert(
          output_shape.end(), real_kernel_size.begin(), real_kernel_size.end());
    } else {
      for (size_t i = 0; i < data_dims.size(); ++i) {
        symbol::DimExpr stride_dimexpr{strides[i]};
        symbol::DimExpr one_dimexpr{1};
        if (!ceil_mode) {
          output_shape.emplace_back((data_dims[i] - real_kernel_size[i] +
                                     real_paddings[2 * i] +
                                     real_paddings[2 * i + 1]) /
                                        stride_dimexpr +
                                    one_dimexpr);
        } else {
          output_shape.emplace_back(
              (data_dims[i] - real_kernel_size[i] + real_paddings[2 * i] +
               real_paddings[2 * i + 1] + stride_dimexpr - one_dimexpr) /
                  stride_dimexpr +
              one_dimexpr);
        }
      }
    }

    // output_N = input_N
    output_shape.insert(output_shape.begin(), x_dims[0]);
    // output_C = input_C
    if (channel_last) {
      output_shape.push_back(x_dims[x_dims.size() - 1]);
    } else {
      output_shape.insert(output_shape.begin() + 1, x_dims[1]);
    }
    return symbol::ShapeOrDataDimExprs{
        symbol::TensorShapeOrDataDimExprs(output_shape)};
  }();

  return output_shape_or_data;
}
}  // namespace

namespace paddle::dialect {
using paddle::dialect::details::CreateShapeOrDataForXShape;

bool AffineGridOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &input_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  std::vector<symbol::DimExpr> input_dims = input_shape_or_data.shape();

  const auto &attributes = op->attributes();
  int output_shape_size;
  std::vector<symbol::DimExpr> output_shape_data;
  if (attributes.find("output_shape") != attributes.end()) {
    std::vector<int64_t> output_shape =
        op->attribute<paddle::dialect::IntArrayAttribute>("output_shape")
            .data()
            .GetData();
    output_shape_size = output_shape.size();
    for (const auto &i : output_shape) {
      output_shape_data.push_back(symbol::DimExpr{i});
    }
  } else if (op->operand_source(1)) {
    const auto &output_shape_or_data =
        infer_context->GetShapeOrDataForValue(op->operand_source(1));

    output_shape_data = details::GetOrCreateExprVecFromData(
        output_shape_or_data, infer_context);
    output_shape_size = output_shape_data.size();
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "The input arguments must have the shape of output, please check!"));
  }

  std::vector<symbol::DimExpr> output_dims;
  output_dims.push_back(input_dims[0]);  // N

  if (output_shape_size == 4) {
    // N * H * W * 2
    output_dims.push_back(output_shape_data[2]);  // H
    output_dims.push_back(output_shape_data[3]);  // W
    output_dims.push_back(symbol::DimExpr(2));    // 2
  } else {
    // N * D * H * W * 3
    output_dims.push_back(output_shape_data[2]);  // D
    output_dims.push_back(output_shape_data[3]);  // H
    output_dims.push_back(output_shape_data[4]);  // W
    output_dims.push_back(symbol::DimExpr(3));    // 3
  }

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(output_dims)});

  return true;
}

bool AllOpInferSymbolicShape(pir::Operation *op,
                             pir::InferSymbolicShapeContext *infer_context) {
  const auto &axis = details::GetVectorAttr(op, "axis");
  return details::ReduceInferDim(op,
                                 infer_context,
                                 axis,
                                 GetBoolAttr(op, "keepdim"), /*keepdim*/
                                 axis.size() == 0 /*reduce_all*/);
}

bool AmaxOpInferSymbolicShape(pir::Operation *op,
                              pir::InferSymbolicShapeContext *infer_context) {
  const auto &axis = details::GetVectorAttr(op, "axis");
  return details::ReduceInferDim(op,
                                 infer_context,
                                 axis,
                                 GetBoolAttr(op, "keepdim"), /*keepdim*/
                                 axis.size() == 0 /*reduce_all*/);
}

bool AminOpInferSymbolicShape(pir::Operation *op,
                              pir::InferSymbolicShapeContext *infer_context) {
  const auto &axis = details::GetVectorAttr(op, "axis");
  return details::ReduceInferDim(op,
                                 infer_context,
                                 axis,
                                 GetBoolAttr(op, "keepdim"), /*keepdim*/
                                 axis.size() == 0 /*reduce_all*/);
}

bool AnyOpInferSymbolicShape(pir::Operation *op,
                             pir::InferSymbolicShapeContext *infer_context) {
  const auto &axis = details::GetVectorAttr(op, "axis");
  return details::ReduceInferDim(op,
                                 infer_context,
                                 axis,
                                 GetBoolAttr(op, "keepdim"), /*keepdim*/
                                 axis.size() == 0 /*reduce_all*/);
}

bool ArgmaxOpInferSymbolicShape(pir::Operation *op,
                                pir::InferSymbolicShapeContext *infer_context) {
  bool flatten = GetBoolAttr(op, "flatten");
  bool keepdims = GetBoolAttr(op, "keepdims");

  const auto &input_sym_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(0)).shape();
  int rank = input_sym_shape.size();

  const auto &axis_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));
  int axis =
      static_cast<int>(axis_shape_or_data.data().value().at(0).Get<int64_t>());
  if (axis < 0) axis += rank;

  const auto &out_sym_shape = [&] {
    std::vector<symbol::DimExpr> out_sym_shape;
    if (flatten) {
      if (keepdims) {
        out_sym_shape.emplace_back(std::int64_t(rank));
      } else {
        out_sym_shape = {};
      }
    } else {
      for (int i = 0; i < axis; i++) {
        out_sym_shape.emplace_back(input_sym_shape.at(i));
      }
      if (keepdims) {
        out_sym_shape.emplace_back(std::int64_t(1));
      }

      for (int i = axis + 1; i < rank; i++) {
        out_sym_shape.emplace_back(input_sym_shape.at(i));
      }
    }
    return out_sym_shape;
  }();

  symbol::ShapeOrDataDimExprs shape_data{
      symbol::TensorShapeOrDataDimExprs(out_sym_shape)};

  infer_context->SetShapeOrDataForValue(op->result(0), shape_data);
  return true;
}

bool ArgminOpInferSymbolicShape(pir::Operation *op,
                                pir::InferSymbolicShapeContext *infer_context) {
  return ArgmaxOpInferSymbolicShape(op, infer_context);
}

bool AsComplexOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  pir::Value operand_source = op->operand_source(0);
  const symbol::ShapeOrDataDimExprs &operand_shape_or_data =
      infer_context->GetShapeOrDataForValue(operand_source);

  const std::vector<symbol::DimExpr> out_dims = [&] {
    std::vector<symbol::DimExpr> out_dims = operand_shape_or_data.shape();
    out_dims.pop_back();
    return out_dims;
  }();

  symbol::ShapeOrDataDimExprs shape_data{
      symbol::TensorShapeOrDataDimExprs(out_dims)};

  infer_context->SetShapeOrDataForValue(op->result(0), shape_data);
  return true;
}
bool AsRealOpInferSymbolicShape(pir::Operation *op,
                                pir::InferSymbolicShapeContext *infer_context) {
  pir::Value operand_source = op->operand_source(0);
  const symbol::ShapeOrDataDimExprs &operand_shape_or_data =
      infer_context->GetShapeOrDataForValue(operand_source);

  const std::vector<symbol::DimExpr> out_dims = [&] {
    std::vector<symbol::DimExpr> out_dims = operand_shape_or_data.shape();
    out_dims.push_back(symbol::DimExpr(2));
    return out_dims;
  }();

  symbol::ShapeOrDataDimExprs shape_data{
      symbol::TensorShapeOrDataDimExprs(out_dims)};

  infer_context->SetShapeOrDataForValue(op->result(0), shape_data);
  return true;
}

bool AssignOpInferSymbolicShape(pir::Operation *op,
                                pir::InferSymbolicShapeContext *infer_context) {
  infer_context->SetShapeOrDataForValue(
      op->result(0),
      infer_context->GetShapeOrDataForValue(op->operand_source(0)));
  return true;
}

// bool AllReduceOpInferSymbolicShape(pir::Operation *op,
//                                    pir::InferSymbolicShapeContext
//                                    *infer_context) {
//   // pass
//   return true;
// }

// bool AllReduce_OpInferSymbolicShape(pir::Operation *op,
//                                     pir::InferSymbolicShapeContext
//                                     *infer_context) {
//   return AllReduceOpInferSymbolicShape(op, infer_context);
// }

// bool BarrierOpInferSymbolicShape(pir::Operation *op,
//                                  pir::InferSymbolicShapeContext
//                                  *infer_context) {
//   // pass
//   return true;
// }

bool Assign_OpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  return AssignOpInferSymbolicShape(op, infer_context);
}

bool AsStridedOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const std::vector<int> &shape =
      paddle::dialect::details::GetVectorAttr<int>(op, "dims");

  int rank = shape.size();
  std::vector<symbol::DimExpr> out_shape;
  for (int i = 0; i < rank; ++i) {
    symbol::DimExpr out_unknown = infer_context->GetNextSymName();
    if (shape[i] == -1) {
      out_shape.push_back(out_unknown);
    } else {
      out_shape.push_back(symbol::DimExpr(shape[i]));
    }
  }

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(out_shape)});

  return true;
}

bool BatchSizeLikeInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const symbol::ShapeOrDataDimExprs &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const auto &x_shape = x_shape_or_data.shape();
  std::vector<int> shape_attr =
      paddle::dialect::details::GetVectorAttr<int>(op, "shape");
  int input_dim_idx =
      op->attribute<pir::Int32Attribute>("input_dim_idx").data();
  int output_dim_idx =
      op->attribute<pir::Int32Attribute>("output_dim_idx").data();
  PADDLE_ENFORCE_GT(shape_attr.size(),
                    0,
                    common::errors::InvalidArgument(
                        "Shape size must be larger than 0, but received: %d.",
                        shape_attr.size()));
  std::vector<symbol::DimExpr> out_shape;
  for (size_t i = 0; i < shape_attr.size(); ++i) {
    out_shape.emplace_back(symbol::DimExpr(shape_attr[i]));
  }

  PADDLE_ENFORCE_GE(
      input_dim_idx,
      0,
      common::errors::InvalidArgument(
          "Input dimension index must be larger than or equal to 0."));
  size_t input_dim_size = x_shape.size();

  PADDLE_ENFORCE_GE(
      output_dim_idx,
      0,
      common::errors::InvalidArgument(
          "Output dimension index must be larger than or equal to 0."));
  PADDLE_ENFORCE(static_cast<int>(input_dim_size) > input_dim_idx ||
                     static_cast<int>(input_dim_size) == -1,
                 common::errors::InvalidArgument(
                     "Input dimension size must be larger than "
                     "input dimension index, but received input "
                     "dimension size: %s, input dimension index: %s.",
                     static_cast<int>(input_dim_size),
                     input_dim_idx));

  size_t output_shape_size = shape_attr.size();
  PADDLE_ENFORCE_GT(
      output_shape_size,
      output_dim_idx,
      common::errors::InvalidArgument(
          "Output dimension size must be larger than output dimension index."));
  // NOTE(gongshaotian):The Python API for this operator has been discontinued
  // in version 2.6. Currently, only the situation where one -1 appears in the
  // shape parameter (shape[input_id_idx] == -1) is supported.
  out_shape[output_dim_idx] = x_shape[input_dim_idx];

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(out_shape)});

  return true;
}

bool BipartiteMatchOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &dist_mat_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(0)).shape();

  PADDLE_ENFORCE_EQ(
      dist_mat_shape.size(),
      2,
      common::errors::InvalidArgument("The rank of Input(DistMat) must be 2."));

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(dist_mat_shape)});

  infer_context->SetShapeOrDataForValue(
      op->result(1),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(dist_mat_shape)});

  return true;
}

bool CastOpInferSymbolicShape(pir::Operation *op,
                              pir::InferSymbolicShapeContext *infer_context) {
  infer_context->SetShapeOrDataForValue(
      op->result(0),
      infer_context->GetShapeOrDataForValue(op->operand_source(0)));

  return true;
}

bool Cast_OpInferSymbolicShape(pir::Operation *op,
                               pir::InferSymbolicShapeContext *infer_context) {
  return CastOpInferSymbolicShape(op, infer_context);
}

bool CheckNumericsOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs({symbol::DimExpr(3)})});

  infer_context->SetShapeOrDataForValue(
      op->result(1),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs({symbol::DimExpr(3)})});

  return true;
}

bool CholeskyOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(0)).shape();

  auto rank = x_shape.size();
  PADDLE_ENFORCE_GE(rank,
                    2,
                    common::errors::InvalidArgument(
                        "The Input(X) should have at least 2 dimensions. But "
                        "received a %d dimension tensor.",
                        rank));

  infer_context->AddEqualCstr(x_shape[rank - 2], x_shape[rank - 1]);

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(x_shape)});

  return true;
}

bool ClassCenterSampleOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &label_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));

  PADDLE_ENFORCE_EQ(label_shape_or_data.shape().size(),
                    1,
                    common::errors::InvalidArgument(
                        "Rank of Input(Label) should be equal to 1, "
                        "but the value given is %d.",
                        label_shape_or_data.shape().size()));

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::TensorShapeOrDataDimExprs(label_shape_or_data.shape()));

  symbol::DimExpr out_unknown = infer_context->GetNextSymName();
  const std::vector<symbol::DimExpr> out_dims = {out_unknown};
  const symbol::ShapeOrDataDimExprs sampled_local_class_center_dims{
      symbol::TensorShapeOrDataDimExprs(out_dims)};
  infer_context->SetShapeOrDataForValue(op->result(1),
                                        sampled_local_class_center_dims);

  return true;
}

bool ClipByNormOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &input_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(0)).shape();
  float max_norm = op->attribute<pir::FloatAttribute>("max_norm").data();
  PADDLE_ENFORCE_GT(
      max_norm,
      0,
      common::errors::InvalidArgument("max_norm should be greater than 0. "
                                      "Received max_norm is %f.",
                                      max_norm));

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(input_shape)});
  return true;
}

bool ClipByNormSrOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  return ClipByNormOpInferSymbolicShape(op, infer_context);
}

bool CummaxOpInferSymbolicShape(pir::Operation *op,
                                pir::InferSymbolicShapeContext *infer_context) {
  pir::Value operand_source = op->operand_source(0);
  const symbol::ShapeOrDataDimExprs &operand_shape_or_data =
      infer_context->GetShapeOrDataForValue(operand_source);
  const auto &operand_shape = operand_shape_or_data.shape();

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(operand_shape)});
  infer_context->SetShapeOrDataForValue(
      op->result(1),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(operand_shape)});
  return true;
}
bool CumminOpInferSymbolicShape(pir::Operation *op,
                                pir::InferSymbolicShapeContext *infer_context) {
  return CummaxOpInferSymbolicShape(op, infer_context);
}
bool CumprodOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  pir::Value operand_source = op->operand_source(0);
  const symbol::ShapeOrDataDimExprs &operand_shape_or_data =
      infer_context->GetShapeOrDataForValue(operand_source);

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(operand_shape_or_data.shape())});
  return true;
}
bool Cumprod_OpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  return CumprodOpInferSymbolicShape(op, infer_context);
}
bool CumsumOpInferSymbolicShape(pir::Operation *op,
                                pir::InferSymbolicShapeContext *infer_context) {
  pir::Value operand_source = op->operand_source(0);

  const symbol::ShapeOrDataDimExprs &operand_shape_or_data =
      infer_context->GetShapeOrDataForValue(operand_source);

  bool flatten = GetBoolAttr(op, "flatten");
  if (flatten) {
    symbol::DimExpr product{1};
    const auto &dim_exprs = operand_shape_or_data.shape();
    for (const auto &dim_expr : dim_exprs) {
      product = product * dim_expr;
    }
    const std::vector<symbol::DimExpr> out_dims = {product};
    symbol::ShapeOrDataDimExprs shape_data{
        symbol::TensorShapeOrDataDimExprs(out_dims)};
    infer_context->SetShapeOrDataForValue(op->result(0), shape_data);

  } else {
    infer_context->SetShapeOrDataForValue(
        op->result(0),
        symbol::ShapeOrDataDimExprs{
            symbol::TensorShapeOrDataDimExprs(operand_shape_or_data.shape())});
  }
  return true;
}
bool Cumsum_OpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  return CumsumOpInferSymbolicShape(op, infer_context);
}
bool ChannelShuffleOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const auto &x_shape = x_shape_or_data.shape();

  int groups = op->attribute<pir::Int32Attribute>("groups").data();
  std::string data_format =
      op->attribute<pir::StrAttribute>("data_format").AsString();

  PADDLE_ENFORCE_EQ(x_shape.size(),
                    4,
                    common::errors::InvalidArgument(
                        "Input should be a 4-D tensor of format [N, "
                        "C, H, W] or [N, H, W, C], but got %u.",
                        x_shape.size()));
  PADDLE_ENFORCE_GE(
      groups,
      1,
      common::errors::InvalidArgument("groups should be larger than 0."));
  PADDLE_ENFORCE_EQ(data_format == "NCHW" || data_format == "NHWC",
                    true,
                    common::errors::InvalidArgument(
                        "data_format must be one of NCHW and NHWC. "
                        "But received data_format: %s",
                        data_format));

  const bool channel_last = (data_format == "NHWC");

  symbol::DimExpr channels;
  if (!channel_last) {
    channels = x_shape[1];
  } else {
    channels = x_shape[3];
  }

  symbol::DimExpr groups_expr = symbol::DimExpr(groups);
  symbol::DimExpr expected_channels = groups_expr * (channels / groups_expr);

  infer_context->AddEqualCstr(channels, expected_channels);

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(x_shape)});

  return true;
}

bool CropOpInferSymbolicShape(pir::Operation *op,
                              pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const std::vector<symbol::DimExpr> &x_shape = x_shape_or_data.shape();

  // GetIntArrayFromAttrOrOperand is used to get vector from IntArray[].
  // Sometimes from attribute and other from operand Enter name and index of
  // IntArray[]
  std::vector<symbol::DimExpr> offsets =
      details::GetIntArrayFromAttrOrOperand(op, infer_context, "offsets", 2);
  std::vector<symbol::DimExpr> in_shape =
      details::GetIntArrayFromAttrOrOperand(op, infer_context, "shape", 1);
  std::vector<symbol::DimExpr> out_dims;

  PADDLE_ENFORCE_EQ(in_shape.size(),
                    x_shape.size(),
                    common::errors::InvalidArgument(
                        "The number of elements (%d) of attribute 'shape' for "
                        "CropTensor must be equal to the number of "
                        "dimensions (%d) of the input.",
                        in_shape.size(),
                        x_shape.size()));
  PADDLE_ENFORCE_EQ(
      offsets.size(),
      x_shape.size(),
      common::errors::InvalidArgument(
          "The number of elements (%d) of attribute 'offsets' for "
          "CropTensor must be equal to the number of "
          "dimensions (%d) of the input.",
          offsets.size(),
          x_shape.size()));

  for (size_t i = 0; i < in_shape.size(); ++i) {
    if (in_shape[i].isa<int64_t>()) {
      if (in_shape[i].Get<int64_t>() == -1) {
        out_dims.push_back(symbol::DimExpr(x_shape[i] - offsets[i]));
      } else {
        out_dims.push_back(symbol::DimExpr(in_shape[i]));
      }
    } else {
      out_dims.push_back(in_shape[i]);
    }
  }

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(out_dims)});
  return true;
}

bool DecodeJpegOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const std::string &mode = op->attribute<pir::StrAttribute>("mode").AsString();

  std::vector<symbol::DimExpr> out_shape;

  if (mode == "unchanged") {
    out_shape = {infer_context->GetNextSymName(),
                 infer_context->GetNextSymName(),
                 infer_context->GetNextSymName()};
  } else if (mode == "gray") {
    out_shape = {symbol::DimExpr(1),
                 infer_context->GetNextSymName(),
                 infer_context->GetNextSymName()};
  } else if (mode == "rgb") {
    out_shape = {symbol::DimExpr(3),
                 infer_context->GetNextSymName(),
                 infer_context->GetNextSymName()};
  } else {
    PADDLE_THROW(common::errors::Fatal(
        "The provided mode is not supported for JPEG files on GPU: ", mode));
  }

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(out_shape)});

  return true;
}

bool DetOpInferSymbolicShape(pir::Operation *op,
                             pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const auto &x_shape = x_shape_or_data.shape();
  int x_shape_size = x_shape.size();
  PADDLE_ENFORCE_GE(
      x_shape_size,
      2,
      common::errors::InvalidArgument("the input matrix dimension size should "
                                      "greater than or equal to 2."));
  infer_context->AddEqualCstr(x_shape[x_shape_size - 2],
                              x_shape[x_shape_size - 1]);
  std::vector<symbol::DimExpr> out_shape = x_shape;
  out_shape.pop_back();
  out_shape.pop_back();
  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(out_shape)});
  return true;
}

bool DiagOpInferSymbolicShape(pir::Operation *op,
                              pir::InferSymbolicShapeContext *infer_context) {
  const auto x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const auto x_shape = x_shape_or_data.shape();
  const int offset_data = op->attribute<pir::Int32Attribute>("offset").data();
  auto offset = symbol::DimExpr(offset_data);

  if (x_shape.size() <= 1) {
    symbol::DimExpr size_ =
        (x_shape.size() == 1UL ? x_shape[0] : symbol::DimExpr(1)) +
        symbol::DimExpr(std::abs(offset_data));
    infer_context->SetShapeOrDataForValue(
        op->result(0), symbol::TensorShapeOrDataDimExprs({size_, size_}));
  } else if (x_shape.size() == 2UL) {
    if (x_shape[0].isa<int64_t>() && x_shape[1].isa<int64_t>()) {
      int64_t size_ = 0;
      if (offset_data >= 0) {
        if (x_shape[0].dyn_cast<int64_t>() <
            x_shape[1].dyn_cast<int64_t>() - offset_data) {
          size_ = x_shape[0].dyn_cast<int64_t>();
        } else {
          size_ = x_shape[1].dyn_cast<int64_t>() - offset_data;
        }
      } else {
        if (x_shape[0].dyn_cast<int64_t>() + offset_data <
            x_shape[1].dyn_cast<int64_t>()) {
          size_ = x_shape[0].dyn_cast<int64_t>() + offset_data;
        } else {
          size_ = x_shape[1].dyn_cast<int64_t>();
        }
      }
      infer_context->SetShapeOrDataForValue(
          op->result(0), symbol::TensorShapeOrDataDimExprs({size_}));
    } else {
      symbol::DimExpr out_unknown =
          infer_context->GetNextSymName();  // unknown until runtime
      infer_context->SetShapeOrDataForValue(
          op->result(0), symbol::TensorShapeOrDataDimExprs({out_unknown}));
    }
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "diag only support 1D/2D matrix, but input has %u dims",
        x_shape.size()));
  }
  return true;
}

bool DiagEmbedOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  pir::Value operand_source = op->operand_source(0);
  const symbol::ShapeOrDataDimExprs &operand_shape_or_data =
      infer_context->GetShapeOrDataForValue(operand_source);
  const auto &attributes = op->attributes();
  int dim1 = attributes.at("dim1").dyn_cast<pir::Int32Attribute>().data();
  int dim2 = attributes.at("dim2").dyn_cast<pir::Int32Attribute>().data();
  int offset = attributes.at("offset").dyn_cast<pir::Int32Attribute>().data();

  const auto &x_dims = operand_shape_or_data.shape();
  int dim1_ = dim1 < 0 ? x_dims.size() + dim1 + 1 : dim1;
  int dim2_ = dim2 < 0 ? x_dims.size() + dim2 + 1 : dim2;
  int64_t offset_ = static_cast<int64_t>(std::abs(offset));
  symbol::DimExpr new_dim_len =
      symbol::DimExpr(offset_) + x_dims.at(x_dims.size() - 1);

  const auto &out_dims = [&] {
    std::vector<symbol::DimExpr> out_dims = x_dims;
    out_dims.pop_back();
    out_dims.insert(out_dims.begin() + std::min(dim1_, dim2_), new_dim_len);
    out_dims.insert(out_dims.begin() + std::max(dim1_, dim2_), new_dim_len);
    return out_dims;
  }();
  symbol::ShapeOrDataDimExprs shape_data{
      symbol::TensorShapeOrDataDimExprs(out_dims)};
  infer_context->SetShapeOrDataForValue(op->result(0), shape_data);
  return true;
}
bool DiagonalOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  pir::Value operand_source = op->operand_source(0);
  const symbol::ShapeOrDataDimExprs &operand_shape_or_data =
      infer_context->GetShapeOrDataForValue(operand_source);
  const auto &attributes = op->attributes();
  int axis1 = attributes.at("axis1").dyn_cast<pir::Int32Attribute>().data();
  int axis2 = attributes.at("axis2").dyn_cast<pir::Int32Attribute>().data();
  int offset = attributes.at("offset").dyn_cast<pir::Int32Attribute>().data();

  const auto &x_dims = operand_shape_or_data.shape();
  int axis1_ = axis1 < 0 ? x_dims.size() + axis1 : axis1;
  int axis2_ = axis2 < 0 ? x_dims.size() + axis2 : axis2;

  auto out_dims = x_dims;
  auto axis1_size = out_dims.at(axis1_);
  auto axis2_size = out_dims.at(axis2_);
  out_dims.erase(out_dims.begin() + std::max(axis1_, axis2_));
  out_dims.erase(out_dims.begin() + std::min(axis1_, axis2_));

  symbol::DimExprBuilder builder;
  symbol::DimExpr zero{0};
  symbol::DimExpr res_shape;
  symbol::DimExpr offset_sym{offset};
  if (offset == 0) {
    res_shape = builder.Min(axis1_size, axis2_size);
  } else if (offset > 0) {
    if (axis2_size.isa<int64_t>()) {
      res_shape = (axis2_size.dyn_cast<int64_t>() - offset) > 0
                      ? builder.Min(axis1_size, axis2_size - offset_sym)
                      : zero;
    } else {
      res_shape = infer_context->GetNextSymName();
    }
  } else {
    if (axis1_size.isa<int64_t>()) {
      res_shape = (axis1_size.dyn_cast<int64_t>() + offset) > 0
                      ? builder.Min(axis1_size + offset_sym, axis2_size)
                      : zero;
    } else {
      res_shape = infer_context->GetNextSymName();
    }
  }
  out_dims.push_back(symbol::SimplifyDimExpr(res_shape));

  symbol::ShapeOrDataDimExprs shape_data{
      symbol::TensorShapeOrDataDimExprs(out_dims)};
  infer_context->SetShapeOrDataForValue(op->result(0), shape_data);
  return true;
}

bool DistributeFpnProposalsOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &attributes = op->attributes();
  int32_t min_level =
      attributes.at("min_level").dyn_cast<pir::Int32Attribute>().data();
  int32_t max_level =
      attributes.at("max_level").dyn_cast<pir::Int32Attribute>().data();
  int32_t num_levels = max_level - min_level + 1;
  int64_t batch_size = 1;

  symbol::DimExpr num_rois =
      infer_context->GetShapeOrDataForValue(op->operand_source(0))
          .shape()
          .at(0);

  const auto &multi_rois_out_shape = [&]() {
    symbol::TensorListShapeOrDataDimExprs multi_rois_out_shape;
    if (num_levels == 1) {
      multi_rois_out_shape.emplace_back(
          symbol::TensorShapeOrDataDimExprs({num_rois, 4}));
    } else {
      symbol::DimExpr last_dim = num_rois;
      for (int i = 0; i < num_levels - 1; i++) {
        const auto &next_sym_name = infer_context->GetNextSymName();
        std::vector<symbol::DimExpr> level_dim = {next_sym_name, 4};
        multi_rois_out_shape.emplace_back(
            symbol::TensorShapeOrDataDimExprs(level_dim));
        last_dim = last_dim - level_dim.at(0);
      }
      multi_rois_out_shape.emplace_back(symbol::TensorShapeOrDataDimExprs(
          {infer_context->GetNextSymName(), 4}));
    }

    return multi_rois_out_shape;
  }();

  const auto &rois_num_per_level_out_shape = [&]() {
    symbol::TensorListShapeOrDataDimExprs rois_num_per_level_out_shape;
    rois_num_per_level_out_shape.resize(
        num_levels, symbol::TensorShapeOrDataDimExprs({batch_size}));
    return rois_num_per_level_out_shape;
  }();

  const auto &restore_ind = [&]() {
    if (op->operand_source(1)) {
      return symbol::TensorShapeOrDataDimExprs(
          {infer_context->GetNextSymName(), 1});
    }
    return symbol::TensorShapeOrDataDimExprs({num_rois, 1});
  }();

  infer_context->SetShapeOrDataForValue(op->result(0), multi_rois_out_shape);
  infer_context->SetShapeOrDataForValue(op->result(1),
                                        rois_num_per_level_out_shape);
  infer_context->SetShapeOrDataForValue(op->result(2), restore_ind);
  return true;
}

bool FakeChannelWiseQuantizeDequantizeAbsMaxOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));

  int bit_length = op->attribute<pir::Int32Attribute>("bit_length").data();
  int quant_axis = op->attribute<pir::Int32Attribute>("quant_axis").data();

  PADDLE_ENFORCE_EQ(bit_length >= 1 && bit_length <= 16,
                    true,
                    common::errors::InvalidArgument(
                        "'bit_length' should be between 1 and 16, but "
                        "the received is %d",
                        bit_length));
  PADDLE_ENFORCE_EQ(
      quant_axis == 0 || quant_axis == 1,
      true,
      common::errors::InvalidArgument("'quant_axis' should be 0 or 1, but "
                                      "the received is %d",
                                      quant_axis));

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(x_shape_or_data.shape())});

  std::vector<symbol::DimExpr> out_scale_shape = {
      x_shape_or_data.shape()[quant_axis]};
  infer_context->SetShapeOrDataForValue(
      op->result(1),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(out_scale_shape)});

  return true;
}

bool FrameOpInferSymbolicShape(pir::Operation *op,
                               pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const auto &x_shape = x_shape_or_data.shape();
  size_t x_rank = x_shape.size();
  PADDLE_ENFORCE_GE(x_rank,
                    1,
                    common::errors::InvalidArgument(
                        "Input(X) of FrameOp should be a tensor which contains "
                        "at least 1 dimension, but got rank %s.",
                        x_rank));
  int hop_length = op->attribute<pir::Int32Attribute>("hop_length").data();
  int frame_length = op->attribute<pir::Int32Attribute>("frame_length").data();
  int axis = op->attribute<pir::Int32Attribute>("axis").data();
  PADDLE_ENFORCE_GT(hop_length,
                    0,
                    common::errors::InvalidArgument(
                        "Attribute(hop_length) of FrameOp should be greater "
                        "than 0, but got %s.",
                        hop_length));
  PADDLE_ENFORCE_EQ(
      (axis == 0 || axis == -1),
      true,
      common::errors::InvalidArgument(
          "Attribute(axis) of FrameOp should 0 or -1, but got %s.", axis));
  std::vector<symbol::DimExpr> out_shape;
  symbol::DimExpr seq_length;
  symbol::DimExpr n_frames = 0;
  int start_axis = 0;
  int end_axis = 0;

  if (axis == 0) {
    seq_length = x_shape[0];
    start_axis = 1;
    end_axis = x_rank - 1;
  } else {
    seq_length = x_shape[x_rank - 1];
    start_axis = 0;
    end_axis = x_rank - 2;
  }
  if (seq_length.isa<int64_t>()) {
    PADDLE_ENFORCE_LE(frame_length,
                      seq_length.dyn_cast<int64_t>(),
                      common::errors::InvalidArgument(
                          "Attribute(frame_length) of FrameOp should be less "
                          "equal than sequence length, but got (%s) > (%s).",
                          frame_length,
                          seq_length.dyn_cast<int64_t>()));
  }

  // It won't go into for loop when x_rank == 1U.
  for (int i = start_axis; i <= end_axis; i++) {
    out_shape.push_back(x_shape[i]);
  }

  n_frames = symbol::DimExpr((seq_length - frame_length) / hop_length + 1);
  if (axis == 0) {
    out_shape.insert(out_shape.begin(), symbol::DimExpr(frame_length));
    out_shape.insert(out_shape.begin(), n_frames);
  } else {
    out_shape.push_back(symbol::DimExpr(frame_length));
    out_shape.push_back(n_frames);
  }
  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(out_shape)});
  return true;
}

bool EigOpInferSymbolicShape(pir::Operation *op,
                             pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const auto &x_shape = x_shape_or_data.shape();

  size_t rank = x_shape.size();

  symbol::DimExpr last_dim = x_shape[rank - 1];
  infer_context->AddEqualCstr(x_shape[rank - 2], last_dim);
  std::vector<symbol::DimExpr> batch_shape(x_shape.begin(), x_shape.end() - 2);
  symbol::ShapeOrDataDimExprs out_w_shape{
      symbol::TensorShapeOrDataDimExprs(batch_shape)};

  infer_context->SetShapeOrDataForValue(op->result(0), out_w_shape);

  infer_context->SetShapeOrDataForValue(
      op->result(1),
      symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(x_shape)});

  return true;
}

bool EighOpInferSymbolicShape(pir::Operation *op,
                              pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(0)).shape();
  std::vector<symbol::DimExpr> out_shape;
  for (size_t i = 0; i < x_shape.size() - 1; ++i) {
    out_shape.emplace_back(x_shape.at(i));
  }
  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(out_shape)});
  infer_context->SetShapeOrDataForValue(
      op->result(1),
      symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(x_shape)});
  return true;
}

bool EigvalshOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  return EighOpInferSymbolicShape(op, infer_context);
}

bool FullBatchSizeLikeOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  return BatchSizeLikeInferSymbolicShape(op, infer_context);
}

bool FractionalMaxPoolOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const std::vector<symbol::DimExpr> &x_dims = x_shape_or_data.shape();

  PADDLE_ENFORCE_EQ(
      (x_dims.size() == 4 || x_dims.size() == 5),
      true,
      common::errors::InvalidArgument(
          "Pooling input should be 4-D or 5-D tensor but received %dD-Tensor",
          x_dims.size()));

  std::vector<int> output_size =
      paddle::dialect::details::GetVectorAttr<int>(op, "output_size");
  std::vector<int> kernel_size =
      paddle::dialect::details::GetVectorAttr<int>(op, "kernel_size");

  std::vector<symbol::DimExpr> output_shape = {x_dims[0], x_dims[1]};
  for (const auto &dim : output_size) {
    output_shape.emplace_back(symbol::DimExpr(dim));
  }

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(output_shape)});

  infer_context->SetShapeOrDataForValue(
      op->result(1),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(output_shape)});

  return true;
}

bool FractionalMaxPool3dOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  return FractionalMaxPoolOpInferSymbolicShape(op, infer_context);
}

bool FractionalMaxPool2dOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  return FractionalMaxPoolOpInferSymbolicShape(op, infer_context);
}

bool FakeQuantizeAbsMaxOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const std::vector<symbol::DimExpr> &x_dims = x_shape_or_data.shape();

  int bit_length = op->attribute<pir::Int32Attribute>("bit_length").data();

  PADDLE_ENFORCE_EQ(bit_length >= 1 && bit_length <= 16,
                    true,
                    common::errors::InvalidArgument(
                        "'bit_length' should be between 1 and 16, but "
                        "the received is %d",
                        bit_length));

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(x_dims)});
  infer_context->SetShapeOrDataForValue(
      op->result(1),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs({symbol::DimExpr(1)})});

  return true;
}

bool FakeChannelWiseQuantizeAbsMaxOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));

  int bit_length = op->attribute<pir::Int32Attribute>("bit_length").data();
  int quant_axis = op->attribute<pir::Int32Attribute>("quant_axis").data();

  PADDLE_ENFORCE_EQ(bit_length >= 1 && bit_length <= 16,
                    true,
                    common::errors::InvalidArgument(
                        "'bit_length' should be between 1 and 16, but "
                        "the received is %d",
                        bit_length));

  PADDLE_ENFORCE_EQ(
      quant_axis == 0 || quant_axis == 1,
      true,
      common::errors::InvalidArgument("'quant_axis' should be 0 or 1, but "
                                      "the received is %d",
                                      quant_axis));

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(x_shape_or_data.shape())});

  std::vector<symbol::DimExpr> out_scale_shape = {
      x_shape_or_data.shape()[quant_axis]};
  infer_context->SetShapeOrDataForValue(
      op->result(1),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(out_scale_shape)});

  return true;
}

bool EigvalsOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(0)).shape();
  std::vector<symbol::DimExpr> out_shape;
  for (size_t i = 0; i < x_shape.size() - 1; ++i) {
    out_shape.push_back(x_shape.at(i));
  }
  infer_context->SetShapeOrDataForValue(
      op->result(0), symbol::TensorShapeOrDataDimExprs(out_shape));
  return true;
}

bool FftC2cOpInferSymbolicShape(pir::Operation *op,
                                pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  std::vector<symbol::DimExpr> x_dims = x_shape_or_data.shape();

  // Set the output shape to be the same as the input shape
  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(x_dims)});

  return true;
}

bool FftC2rOpInferSymbolicShape(pir::Operation *op,
                                pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  std::vector<symbol::DimExpr> x_dims = x_shape_or_data.shape();

  auto axes = paddle::dialect::details::GetVectorAttr<int64_t>(op, "axes");
  int64_t last_dim_size =
      op->attribute<pir::Int64Attribute>("last_dim_size").data();
  int last_fft_axis = static_cast<int>(axes.back());

  std::vector<symbol::DimExpr> out_dims = x_dims;

  if (last_dim_size > 0) {
    out_dims[last_fft_axis] = symbol::DimExpr(last_dim_size);
  } else {
    symbol::DimExprBuilder builder;
    out_dims[last_fft_axis] =
        builder.Mul(x_dims[last_fft_axis], 2) - symbol::DimExpr{1};
  }

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(out_dims)});

  return true;
}

bool FftR2cOpInferSymbolicShape(pir::Operation *op,
                                pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  std::vector<symbol::DimExpr> x_dims = x_shape_or_data.shape();

  auto axes = paddle::dialect::details::GetVectorAttr<int64_t>(op, "axes");
  bool onesided = op->attribute<pir::BoolAttribute>("onesided").data();

  std::vector<symbol::DimExpr> out_dims = x_dims;

  if (onesided) {
    int last_fft_axis = static_cast<int>(axes.back());
    symbol::DimExprBuilder builder;
    out_dims[last_fft_axis] =
        builder.Add(builder.Div(x_dims[last_fft_axis], 2), 1);
  }

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(out_dims)});

  return true;
}

bool FillDiagonalOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  std::vector<symbol::DimExpr> x_dims = x_shape_or_data.shape();

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(x_dims)});

  return true;
}

bool FillDiagonal_OpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  return FillDiagonalOpInferSymbolicShape(op, infer_context);
}

bool FlattenOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &attributes = op->attributes();
  int start_axis =
      attributes.at("start_axis").dyn_cast<pir::Int32Attribute>().data();
  int stop_axis =
      attributes.at("stop_axis").dyn_cast<pir::Int32Attribute>().data();

  const auto &x_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(0)).shape();
  int in_dims_size = x_shape.size();

  if (in_dims_size == 0) {
    PADDLE_ENFORCE_EQ(
        start_axis == 0 || start_axis == -1,
        true,
        common::errors::InvalidArgument("The start_axis should be 0 or -1 when "
                                        "the input tensor is a 0D-Tensor"));
    PADDLE_ENFORCE_EQ(stop_axis == 0 || stop_axis == -1,
                      true,
                      common::errors::InvalidArgument(
                          "The stop_axis should be 0 or -1 when the "
                          "input tensor is a 0D-Tensor"));
    // this can ensure out shape {1}
    start_axis = 0;
    stop_axis = -1;
  }

  if (start_axis < 0) {
    start_axis = start_axis + in_dims_size;
  }
  if (stop_axis < 0) {
    stop_axis = stop_axis + in_dims_size;
  }
  if (in_dims_size > 0) {
    PADDLE_ENFORCE_GE(
        stop_axis,
        start_axis,
        common::errors::InvalidArgument("The stop_axis should be greater"
                                        "than or equal to start_axis."));
  }

  symbol::DimExpr outer{1};
  std::vector<symbol::DimExpr> out_shape;
  out_shape.reserve(in_dims_size - stop_axis + start_axis + 1);
  for (int i = 0; i < start_axis; ++i) {
    out_shape.push_back(x_shape.at(i));
  }
  for (int i = start_axis; i <= stop_axis; i++) {
    outer = outer * x_shape.at(i);
  }
  out_shape.push_back(outer);
  for (int i = stop_axis + 1; i < in_dims_size; i++) {
    out_shape.push_back(x_shape.at(i));
  }

  symbol::ShapeOrDataDimExprs out_shape_data{
      symbol::TensorShapeOrDataDimExprs(out_shape)};
  infer_context->SetShapeOrDataForValue(op->result(0), out_shape_data);
  return true;
}

bool FakeQuantizeDequantizeAbsMaxOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const symbol::ShapeOrDataDimExprs &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const auto &x_shape = x_shape_or_data.shape();

  // Validate the bit_length attribute
  int bit_length = op->attribute<pir::Int32Attribute>("bit_length").data();
  PADDLE_ENFORCE_EQ(bit_length >= 1 && bit_length <= 16,
                    true,
                    common::errors::InvalidArgument(
                        "'bit_length' should be between 1 and 16, but "
                        "the received is %d",
                        bit_length));

  // Set the shape for the output tensor 'out', same as input tensor 'x'
  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(x_shape)});

  // Set the shape for the output tensor 'out_scale' as a scalar {1}
  symbol::TensorShapeOrDataDimExprs scalar_shape(
      std::vector<symbol::DimExpr>{symbol::DimExpr(1)});
  infer_context->SetShapeOrDataForValue(op->result(1), scalar_shape);

  return true;
}

bool Flatten_OpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  return FlattenOpInferSymbolicShape(op, infer_context);
}

bool FrobeniusNormOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  bool keepdim = op->attribute<pir::BoolAttribute>("keep_dim").data();
  bool reduce_all = op->attribute<pir::BoolAttribute>("reduce_all").data();

  std::vector<int64_t> axis;
  if (paddle::dialect::details::GetAxisFromOpInput(
          op->operand_source(1), infer_context, &axis)) {
    if (axis.size() == 0) {
      reduce_all = true;
    }

    return details::ReduceInferDim(
        op, infer_context, axis, keepdim, reduce_all);
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "Reduction[Sum|Max|Prod|Mean..] OpInferSymbolicShape: 'axis' only "
        "support FullIntArrayOp's result or constant DimExpr now."));
  }
  return false;
}

bool FoldOpInferSymbolicShape(pir::Operation *op,
                              pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(0)).shape();

  std::vector<symbol::DimExpr> out_shape;
  out_shape.push_back(x_shape[0]);

  const std::vector<int> &output_sizes =
      paddle::dialect::details::GetVectorAttr<int>(op, "output_sizes");
  PADDLE_ENFORCE_EQ(
      output_sizes.size(),
      2,
      common::errors::InvalidArgument(
          "It is expected output_size equals to 2, but got size %d",
          output_sizes.size()));
  infer_context->AddGreatThanOneCstr(output_sizes[0]);
  infer_context->AddGreatThanOneCstr(output_sizes[1]);

  const std::vector<int> &kernel_sizes =
      paddle::dialect::details::GetVectorAttr<int>(op, "kernel_sizes");
  const std::vector<int> &dilations =
      paddle::dialect::details::GetVectorAttr<int>(op, "dilations");
  const std::vector<int> &strides =
      paddle::dialect::details::GetVectorAttr<int>(op, "strides");
  const std::vector<int> &paddings =
      paddle::dialect::details::GetVectorAttr<int>(op, "paddings");

  PADDLE_ENFORCE_EQ(
      kernel_sizes.size(),
      2,
      common::errors::InvalidArgument(
          "It is expected kernel_size equals to 2, but got size %d",
          kernel_sizes.size()));
  PADDLE_ENFORCE_EQ(
      strides.size(),
      2,
      common::errors::InvalidArgument(
          "It is expected strides_size equals to 2, but got size %d",
          strides.size()));
  PADDLE_ENFORCE_EQ(
      paddings.size(),
      4,
      common::errors::InvalidArgument(
          "It is expected paddings_size equals to 4, but got size %d",
          paddings.size()));
  PADDLE_ENFORCE_EQ(
      dilations.size(),
      2,
      common::errors::InvalidArgument(
          "It is expected dilations_size equals to 2, but got size %d",
          dilations.size()));

  int blocks_height = (output_sizes[0] + 2 * paddings[0] -
                       (dilations[0] * (kernel_sizes[0] - 1) + 1)) /
                          strides[0] +
                      1;
  int blocks_width = (output_sizes[1] + 2 * paddings[1] -
                      (dilations[1] * (kernel_sizes[1] - 1) + 1)) /
                         strides[1] +
                     1;

  infer_context->AddEqualCstr((blocks_height * blocks_width), x_shape[2]);

  out_shape.push_back(x_shape[1] / (kernel_sizes[0] * kernel_sizes[1]));

  out_shape.push_back(symbol::DimExpr(output_sizes[0]));
  out_shape.push_back(symbol::DimExpr(output_sizes[1]));

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(out_shape)});

  return true;
}

bool IdentityLossOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &input_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(0)).shape();
  int reduction = op->attribute<pir::Int32Attribute>("reduction").data();
  if (reduction == 2) {
    infer_context->SetShapeOrDataForValue(
        op->result(0),
        symbol::ShapeOrDataDimExprs{
            symbol::TensorShapeOrDataDimExprs(input_shape)});
  } else {
    std::vector<symbol::DimExpr> out_shape = {};
    infer_context->SetShapeOrDataForValue(
        op->result(0),
        symbol::ShapeOrDataDimExprs{
            symbol::TensorShapeOrDataDimExprs(out_shape)});
  }

  return true;
}

bool IsEmptyOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const std::vector<symbol::DimExpr> out_shape = {};
  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(out_shape)});
  return true;
}

bool GumbelSoftmaxOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const std::vector<symbol::DimExpr> &x_shape = x_shape_or_data.shape();
  size_t rank = x_shape.size();

  int axis = op->attribute<pir::Int32Attribute>("axis").data();

  if (rank > 0) {
    PADDLE_ENFORCE_EQ(
        axis >= -static_cast<int>(rank) && axis < static_cast<int>(rank),
        true,
        common::errors::InvalidArgument(
            "Attr(axis) value should be in range [-R, R-1], "
            "R is the rank of Input(X)."));
  } else if (rank == 0) {
    PADDLE_ENFORCE_EQ(axis >= -1 && axis <= 0,
                      true,
                      common::errors::InvalidArgument(
                          "Attr(axis) value should be in range [-1, "
                          "0] when input is 0D Tensor "));
  }

  // Set output shape to be the same as input shape
  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(x_shape)});

  return true;
}

bool IdentityLoss_OpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  return IdentityLossOpInferSymbolicShape(op, infer_context);
}

bool KthvalueOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  pir::Value operand_source = op->operand_source(0);
  const symbol::ShapeOrDataDimExprs &operand_shape_or_data =
      infer_context->GetShapeOrDataForValue(operand_source);
  const auto &attributes = op->attributes();
  int axis = attributes.at("axis").dyn_cast<pir::Int32Attribute>().data();
  bool keepdim = GetBoolAttr(op, "keepdim");

  const auto &input_dims = operand_shape_or_data.shape();
  const int &dim_size = input_dims.size();
  if (axis < 0) axis += dim_size;
  std::vector<symbol::DimExpr> out_dims;
  for (int i = 0; i < axis; i++) {
    out_dims.emplace_back(input_dims.at(i));
  }
  if (keepdim && dim_size > 0) {
    out_dims.emplace_back(symbol::DimExpr(1));
  }
  for (int i = axis + 1; i < dim_size; i++) {
    out_dims.emplace_back(input_dims.at(i));
  }
  symbol::ShapeOrDataDimExprs shape_data{
      symbol::TensorShapeOrDataDimExprs(out_dims)};
  infer_context->SetShapeOrDataForValue(op->result(0), shape_data);
  infer_context->SetShapeOrDataForValue(op->result(1), shape_data);
  return true;
}

bool L1NormOpInferSymbolicShape(pir::Operation *op,
                                pir::InferSymbolicShapeContext *infer_context) {
  // The output is a scalar, set the output shape accordingly
  std::vector<symbol::DimExpr> output_shape;
  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(output_shape)});
  return true;
}

bool L1Norm_OpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  return L1NormOpInferSymbolicShape(op, infer_context);
}

bool InverseOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &input_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  std::vector<symbol::DimExpr> input_dims = input_shape.shape();
  int input_rank = input_dims.size();

  infer_context->AddEqualCstr(input_dims[input_rank - 2],
                              input_dims[input_rank - 1]);

  std::vector<symbol::DimExpr> output_dims = input_dims;

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(output_dims)});
  return true;
}

bool LpPool2dOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &kernel_size = [&]() -> std::vector<symbol::DimExpr> {
    std::vector<int64_t> kernel_size_int_vec =
        op->attribute<paddle::dialect::IntArrayAttribute>("kernel_size")
            .data()
            .GetData();
    return details::VecInt642Expr(kernel_size_int_vec);
  }();
  infer_context->SetShapeOrDataForValue(
      op->result(0),
      Pool2dRawInferSymbolicShape(op, kernel_size, infer_context));
  return true;
}

bool LogcumsumexpOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  // same as CumsumOpInferSymbolicShape
  return CumsumOpInferSymbolicShape(op, infer_context);
}

bool LogsumexpOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  bool keepdim = GetBoolAttr(op, "keepdim");
  std::vector<int> axis_in = details::GetVectorAttr<int>(op, "axis");
  std::vector<int64_t> axis;
  axis.reserve(axis_in.size());
  std::for_each(axis_in.begin(), axis_in.end(), [&axis](const int &t) {
    axis.push_back(static_cast<int64_t>(t));
  });
  bool reduce_all = axis.size() == 0 ? true : false;
  return details::ReduceInferDim(op, infer_context, axis, keepdim, reduce_all);
}

bool LrnOpInferSymbolicShape(pir::Operation *op,
                             pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const std::vector<symbol::DimExpr> &x_shape = x_shape_or_data.shape();
  int x_size = x_shape.size();
  PADDLE_ENFORCE_EQ(
      x_size,
      4,
      common::errors::InvalidArgument("Input(input) rank should be 4, "
                                      "but received input rank (%d) != 4",
                                      x_size));
  int n_value = op->attribute<pir::Int32Attribute>("n").data();
  PADDLE_ENFORCE_GT(
      n_value,
      0UL,
      common::errors::InvalidArgument("Argument(n) should be positive, "
                                      "but received n(%d) not greater than 0",
                                      n_value));
  PADDLE_ENFORCE_EQ(
      n_value % 2,
      1UL,
      common::errors::InvalidArgument("Argument(n) should be odd value, "
                                      "but received n(%d) is not an odd value",
                                      n_value));
  infer_context->SetShapeOrDataForValue(
      op->result(0), symbol::TensorShapeOrDataDimExprs(x_shape));
  infer_context->SetShapeOrDataForValue(
      op->result(1), symbol::TensorShapeOrDataDimExprs(x_shape));
  return true;
}

bool LuOpInferSymbolicShape(pir::Operation *op,
                            pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const std::vector<symbol::DimExpr> &x_shape = x_shape_or_data.shape();
  int x_rank = x_shape.size();

  PADDLE_ENFORCE_GE(
      x_rank,
      2,
      common::errors::InvalidArgument(
          "The rank of input must be greater than or equal to 2."));

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(x_shape)});

  const auto &m = x_shape[x_rank - 1];
  const auto &n = x_shape[x_rank - 2];
  symbol::DimExprBuilder builder;
  symbol::DimExpr min_mn = builder.Min(m, n);

  if (x_rank == 2) {
    infer_context->SetShapeOrDataForValue(
        op->result(2),
        symbol::ShapeOrDataDimExprs{
            symbol::TensorShapeOrDataDimExprs({symbol::DimExpr(1)})});
  } else {
    std::vector<symbol::DimExpr> infos_shape(x_shape.begin(),
                                             x_shape.begin() + x_rank - 2);
    infer_context->SetShapeOrDataForValue(
        op->result(2),
        symbol::ShapeOrDataDimExprs{
            symbol::TensorShapeOrDataDimExprs(infos_shape)});
  }

  bool pivot = op->attribute<pir::BoolAttribute>("pivot").data();
  if (pivot) {
    std::vector<symbol::DimExpr> pivots_shape(x_shape.begin(),
                                              x_shape.begin() + x_rank - 1);
    pivots_shape[x_rank - 2] = min_mn;
    infer_context->SetShapeOrDataForValue(
        op->result(1),
        symbol::ShapeOrDataDimExprs{
            symbol::TensorShapeOrDataDimExprs(pivots_shape)});
  }

  return true;
}

bool Lu_OpInferSymbolicShape(pir::Operation *op,
                             pir::InferSymbolicShapeContext *infer_context) {
  return LuOpInferSymbolicShape(op, infer_context);
}

bool MaxOpInferSymbolicShape(pir::Operation *op,
                             pir::InferSymbolicShapeContext *infer_context) {
  bool keepdim = GetBoolAttr(op, "keepdim");

  std::vector<int64_t> axis;
  if (paddle::dialect::details::GetAxisFromOpInput(
          op->operand_source(1), infer_context, &axis)) {
    bool reduce_all = axis.size() == 0;

    return details::ReduceInferDim(
        op, infer_context, axis, keepdim, reduce_all);
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "Reduction[Sum|Max|Prod|Mean..] OpInferSymbolicShape: 'axis' only "
        "support FullIntArrayOp's result or constant DimExpr now."));
  }
  return false;
}

bool ModeOpInferSymbolicShape(pir::Operation *op,
                              pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const auto &x_shape = x_shape_or_data.shape();

  int axis = op->attribute<pir::Int32Attribute>("axis").data();
  bool keepdim = op->attribute<pir::BoolAttribute>("keepdim").data();

  int dim_size = x_shape.size();

  if (axis < 0) {
    axis += dim_size;
  }

  std::vector<symbol::DimExpr> out_dims;
  for (int i = 0; i < axis; i++) {
    out_dims.emplace_back(x_shape[i]);
  }
  if (keepdim && dim_size > 0) {
    out_dims.emplace_back(symbol::DimExpr(1));
  }
  for (int i = axis + 1; i < dim_size; i++) {
    out_dims.emplace_back(x_shape[i]);
  }

  symbol::TensorShapeOrDataDimExprs out_shape(out_dims);

  infer_context->SetShapeOrDataForValue(op->result(0),
                                        symbol::ShapeOrDataDimExprs{out_shape});

  infer_context->SetShapeOrDataForValue(op->result(1),
                                        symbol::ShapeOrDataDimExprs{out_shape});

  return true;
}

bool MaxoutOpInferSymbolicShape(pir::Operation *op,
                                pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const std::vector<symbol::DimExpr> &x_shape = x_shape_or_data.shape();

  int groups = op->attribute<pir::Int32Attribute>("groups").data();
  int axis = op->attribute<pir::Int32Attribute>("axis").data();

  if (axis < 0) {
    axis += x_shape.size();
  }

  std::vector<symbol::DimExpr> output_shape = x_shape;
  output_shape[axis] = x_shape[axis] / groups;
  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(output_shape)});

  return true;
}

bool MinOpInferSymbolicShape(pir::Operation *op,
                             pir::InferSymbolicShapeContext *infer_context) {
  return MaxOpInferSymbolicShape(op, infer_context);
}

bool MaxPoolWithIndexOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const std::vector<symbol::DimExpr> &x_shape = x_shape_or_data.shape();
  std::vector<int> paddings =
      paddle::dialect::details::GetVectorAttr<int>(op, "paddings");
  std::vector<int> strides =
      paddle::dialect::details::GetVectorAttr<int>(op, "strides");
  std::vector<int> kernel_size =
      paddle::dialect::details::GetVectorAttr<int>(op, "kernel_size");
  bool adaptive = op->attribute<pir::BoolAttribute>("adaptive").data();
  bool ceil_mode = op->attribute<pir::BoolAttribute>("ceil_mode").data();
  bool global_pooling =
      op->attribute<pir::BoolAttribute>("global_pooling").data();

  std::vector<symbol::DimExpr> kernel_size_;
  for (size_t i = 0; i < kernel_size.size(); ++i) {
    kernel_size_.emplace_back(kernel_size[i]);
  }
  if (global_pooling) {
    kernel_size_.resize(x_shape.size() - 2);
    for (size_t i = 0; i < kernel_size_.size(); ++i) {
      paddings[i] = 0;
      kernel_size_[i] = x_shape[i + 2];
    }
  }

  PADDLE_ENFORCE_EQ(
      x_shape.size() - kernel_size_.size(),
      2U,
      common::errors::InvalidArgument(
          "The input size %d minus the kernel size %d should equal to 2.",
          x_shape.size(),
          kernel_size_.size()));

  std::vector<symbol::DimExpr> out_shape = {x_shape[0], x_shape[1]};

  if (adaptive) {
    out_shape.insert(out_shape.end(), kernel_size_.begin(), kernel_size_.end());
  } else {
    for (size_t i = 0; i < kernel_size_.size(); ++i) {
      PADDLE_ENFORCE_NE(
          strides[i],
          0,
          common::errors::InvalidArgument(
              "The stride of MaxPool shall not be 0, but received %d.",
              strides[i]));

      if (ceil_mode) {
        out_shape.push_back(symbol::DimExpr((x_shape[i + 2] - kernel_size_[i] +
                                             2 * paddings[i] + strides[i] - 1) /
                                                strides[i] +
                                            1));
      } else {
        out_shape.push_back(symbol::DimExpr(
            (x_shape[i + 2] - kernel_size_[i] + 2 * paddings[i]) / strides[i] +
            1));
      }
    }
  }

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(out_shape)});
  infer_context->SetShapeOrDataForValue(
      op->result(1),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(out_shape)});

  return true;
}

bool MaxPool2dWithIndexOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const std::vector<symbol::DimExpr> &x_shape = x_shape_or_data.shape();

  PADDLE_ENFORCE_EQ(
      x_shape.size(),
      4,
      common::errors::InvalidArgument("Pooling input should be 4-D Tensor"
                                      "but received %dD-Tensor",
                                      x_shape.size()));

  std::vector<int> paddings =
      paddle::dialect::details::GetVectorAttr<int>(op, "paddings");
  std::vector<int> strides =
      paddle::dialect::details::GetVectorAttr<int>(op, "strides");

  PADDLE_ENFORCE_EQ(
      paddings.size(),
      2,
      common::errors::InvalidArgument(
          "It is expected paddings size equals to 2, but got size %d",
          paddings.size()));
  PADDLE_ENFORCE_EQ(
      strides.size(),
      2,
      common::errors::InvalidArgument(
          "It is expected strides_size equals to 2, but got size %d",
          strides.size()));

  return MaxPoolWithIndexOpInferSymbolicShape(op, infer_context);
}

bool MaxPool3dWithIndexOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const std::vector<symbol::DimExpr> &x_shape = x_shape_or_data.shape();

  PADDLE_ENFORCE_EQ(
      x_shape.size(),
      5,
      common::errors::InvalidArgument("Pooling input should be 5-D Tensor"
                                      "but received %dD-Tensor",
                                      x_shape.size()));

  std::vector<int> paddings =
      paddle::dialect::details::GetVectorAttr<int>(op, "paddings");
  std::vector<int> strides =
      paddle::dialect::details::GetVectorAttr<int>(op, "strides");

  PADDLE_ENFORCE_EQ(
      paddings.size(),
      3,
      common::errors::InvalidArgument(
          "It is expected paddings size equals to 3, but got size %d",
          paddings.size()));
  PADDLE_ENFORCE_EQ(
      strides.size(),
      3,
      common::errors::InvalidArgument(
          "It is expected strides_size equals to 3, but got size %d",
          strides.size()));

  return MaxPoolWithIndexOpInferSymbolicShape(op, infer_context);
}

bool MeanOpInferSymbolicShape(pir::Operation *op,
                              pir::InferSymbolicShapeContext *infer_context) {
  bool keepdim = GetBoolAttr(op, "keepdim");
  std::vector<int64_t> axis;
  if (op->num_operands() == 1) {
    const auto attributes = op->attributes();
    if (op->attributes().find("axis") != attributes.end()) {
      axis = op->attribute<paddle::dialect::IntArrayAttribute>("axis")
                 .data()
                 .GetData();
      bool reduce_all = axis.size() == 0;

      return details::ReduceInferDim(
          op, infer_context, axis, keepdim, reduce_all);
    }
  } else if (paddle::dialect::details::GetAxisFromOpInput(
                 op->operand_source(1), infer_context, &axis)) {
    bool reduce_all = axis.size() == 0;

    return details::ReduceInferDim(
        op, infer_context, axis, keepdim, reduce_all);
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "Reduction[Sum|Max|Prod|Mean..] OpInferSymbolicShape: 'axis' only "
        "support FullIntArrayOp's result or constant DimExpr now."));
  }
  return false;
}

bool MeanAllOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const std::vector<symbol::DimExpr> &x_dims = x_shape_or_data.shape();

  PADDLE_ENFORCE_GT(
      x_dims.size(),
      0,
      common::errors::InvalidArgument("Input(x) of MeanAllOp must have rank "
                                      "greater than 0, but received rank 0."));

  std::vector<symbol::DimExpr> output_shape = {};

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(output_shape)});

  return true;
}

bool MatrixPowerOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const std::vector<symbol::DimExpr> &x_shape = x_shape_or_data.shape();
  const int n_dim = x_shape.size();

  PADDLE_ENFORCE_GE(n_dim,
                    2,
                    common::errors::InvalidArgument(
                        "The Input(X) should have at least 2 dimensions. But "
                        "received a %d dimension tensor.",
                        n_dim));
  infer_context->AddEqualCstr(x_shape[n_dim - 2], x_shape[n_dim - 1]);
  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(x_shape)});

  return true;
}

bool MatrixRankOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  // 获取输入x的符号形状
  const symbol::ShapeOrDataDimExprs &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const std::vector<symbol::DimExpr> &x_shape = x_shape_or_data.shape();

  // 确保输入x的维度大于等于2
  PADDLE_ENFORCE_GE(x_shape.size(),
                    2,
                    common::errors::InvalidArgument(
                        "The dims of input must be greater than 2."));

  // 获取Hermitian属性
  bool hermitian = op->attribute<pir::BoolAttribute>("hermitian").data();

  // 如果hermitian为true，确保输入x是方阵
  if (hermitian) {
    infer_context->AddEqualCstr(x_shape[x_shape.size() - 2],
                                x_shape[x_shape.size() - 1]);
  }

  std::vector<symbol::DimExpr> x_batch_dims = {};

  if (x_shape.size() != 2) {
    x_batch_dims = x_shape;
    x_batch_dims.erase(x_batch_dims.end() - 2, x_batch_dims.end());
  }

  // 推断输出的形状，设置批次维度
  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(x_batch_dims)});

  return true;
}

// bool MaxPool2DWithIndexOpInferSymbolicShape(pir::Operation *op,
//                                             pir::InferSymbolicShapeContext
//                                             *infer_context) {
//   // pass
//   return true;
// }

// bool MaxPool3DWithIndexOpInferSymbolicShape(pir::Operation *op,
//                                             pir::InferSymbolicShapeContext
//                                             *infer_context) {
//   // pass
//   return true;
// }

bool MultinomialOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  ExprVec x_shape = x_shape_or_data.shape();
  size_t x_rank = x_shape.size();
  PADDLE_ENFORCE_EQ(x_rank > 0 && x_rank <= 2,
                    true,
                    common::errors::InvalidArgument(
                        "The number of dimensions of the input probability "
                        "distribution should be > 0 and <= 2, but got %d.",
                        x_rank));
  ExprVec out_dims(x_rank);
  for (size_t i = 0; i < x_rank - 1; i++) {
    out_dims[i] = x_shape[i];
  }
  if (op->HasAttribute("num_samples")) {
    const auto &int_num_samples =
        op->attribute<paddle::dialect::ScalarAttribute>("num_samples").data();
    out_dims[x_rank - 1] = symbol::DimExpr(int_num_samples.to<int64_t>());
  } else if (op->operand_source(1)) {
    const auto &num_samples_shape_or_data =
        infer_context->GetShapeOrDataForValue(op->operand_source(1));
    const auto &data_vec = paddle::dialect::details::GetOrCreateExprVecFromData(
        num_samples_shape_or_data, infer_context);
    out_dims[x_rank - 1] = data_vec[0];
  }

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(out_dims)});

  return true;
}

bool NanmedianOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  std::vector<int> axis_list;
  bool keep_dim = false;
  std::string mode;
  if (op->HasAttribute("axes")) {
    axis_list = paddle::dialect::details::GetVectorAttr<int>(op, "axes");
  }
  if (op->HasAttribute("keep_dim")) {
    keep_dim = op->attribute<pir::BoolAttribute>("keep_dim").data();
  }
  if (op->HasAttribute("mode")) {
    mode = op->attribute<pir::StrAttribute>("mode").AsString();
  }
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const auto &x_shape = x_shape_or_data.shape();
  int64_t x_rank = x_shape.size();
  ExprVec out_shape;
  if (axis_list.empty()) {
    if (keep_dim) {
      for (int64_t i = 0; i < x_rank; i++) {
        out_shape.emplace_back(1);
      }
    }
  } else {
    std::vector<int64_t> formatted_axis;
    for (size_t i = 0; i < axis_list.size(); i++) {
      if (axis_list[i] < 0) {
        axis_list[i] += x_rank;
      }
      if (x_rank == 0) {
        infer_context->AddEqualCstr(axis_list[i], symbol::DimExpr(0));
      } else {
        PADDLE_ENFORCE_LT(axis_list[i],
                          x_rank,
                          common::errors::InvalidArgument(
                              "each element of the axis should be in the "
                              "range [ -dimension(X), dimension(X) ) "
                              "which dimension = %d. But received axis = %d.",
                              x_rank,
                              axis_list[i]));
      }
      PADDLE_ENFORCE_EQ(
          std::find(formatted_axis.begin(), formatted_axis.end(), axis_list[i]),
          formatted_axis.end(),
          common::errors::InvalidArgument(
              "Attr(axes) has duplicated elements: %d.", axis_list[i]));
      formatted_axis.emplace_back(axis_list[i]);
    }

    for (int64_t i = 0; i < x_rank; i++) {
      if (std::find(formatted_axis.begin(), formatted_axis.end(), i) ==
          formatted_axis.end()) {
        out_shape.emplace_back(x_shape[i]);
      } else if (keep_dim) {
        out_shape.emplace_back(1);
      }
    }
  }

  auto median_shape = out_shape;

  if (mode == "avg") {
    median_shape.emplace_back(2);
  }
  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(out_shape)});
  infer_context->SetShapeOrDataForValue(
      op->result(1),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(median_shape)});

  return true;
}

bool NormOpInferSymbolicShape(pir::Operation *op,
                              pir::InferSymbolicShapeContext *infer_context) {
  auto x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const auto &x_shape = x_shape_or_data.shape();

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(x_shape)});

  int axis = op->attribute<pir::Int32Attribute>("axis").data();
  bool is_test = op->attribute<pir::BoolAttribute>("is_test").data();

  if (!is_test) {
    if (axis < 0) axis += x_shape.size();

    auto norm_shape = x_shape;
    norm_shape[axis] = symbol::DimExpr(1);
    infer_context->SetShapeOrDataForValue(
        op->result(1),
        symbol::ShapeOrDataDimExprs{
            symbol::TensorShapeOrDataDimExprs(norm_shape)});
  }

  return true;
}

bool NonzeroOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const auto &x_shape = x_shape_or_data.shape();
  int rank = x_shape.size();

  PADDLE_ENFORCE_GE(
      rank,
      1UL,
      common::errors::InvalidArgument(
          "Input(x) should have number of dimension at least 1."));

  std::string sym_name = infer_context->GetNextSymName();
  std::vector<symbol::DimExpr> out_shape{symbol::DimExpr{sym_name},
                                         symbol::DimExpr{rank}};

  symbol::ShapeOrDataDimExprs shape_data{
      symbol::TensorShapeOrDataDimExprs(out_shape)};
  infer_context->SetShapeOrDataForValue(op->result(0), shape_data);
  return true;
}

bool NumelOpInferSymbolicShape(pir::Operation *op,
                               pir::InferSymbolicShapeContext *infer_context) {
  std::vector<symbol::DimExpr> out_shape = {};
  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(out_shape)});

  return true;
}

bool OneHotOpInferSymbolicShape(pir::Operation *op,
                                pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const std::vector<symbol::DimExpr> &x_shape = x_shape_or_data.shape();
  const auto &num_classes_shape_or_date =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));

  const auto &attributes = op->attributes();
  int64_t num_classes;
  symbol::DimExpr num_classes_expr;

  if (attributes.find("num_classes") != attributes.end()) {
    num_classes = op->attribute<pir::Int64Attribute>("num_classes").data();
    num_classes_expr = symbol::DimExpr(num_classes);
  } else if (num_classes_shape_or_date.data().has_value()) {
    num_classes_expr = num_classes_shape_or_date.data().value().at(0);
  } else {
    num_classes_expr = infer_context->GetNextSymName();
  }

  const std::vector<symbol::DimExpr> &out_shape = [&] {
    std::vector<symbol::DimExpr> out_shape = x_shape;
    out_shape.push_back(num_classes_expr);
    return out_shape;
  }();

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(out_shape)});

  return true;
}

// bool P_NormOpInferSymbolicShape(pir::Operation *op,
//                                 pir::InferSymbolicShapeContext
//                                 *infer_context) {

bool OverlapAddOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const std::vector<symbol::DimExpr> &x_dims = x_shape_or_data.shape();
  const int x_rank = x_dims.size();

  int hop_length = op->attribute<pir::Int32Attribute>("hop_length").data();
  int axis = op->attribute<pir::Int32Attribute>("axis").data();

  PADDLE_ENFORCE_GE(
      x_rank,
      2,
      common::errors::InvalidArgument(
          "Input(X) of OverlapAddOp should be a tensor which contains "
          "at least 2 dimensions, but got rank %s.",
          x_rank));

  PADDLE_ENFORCE_GT(
      hop_length,
      0,
      common::errors::InvalidArgument(
          "Attribute(hop_length) of OverlapAddOp should be greater "
          "than 0, but got %s.",
          hop_length));

  PADDLE_ENFORCE_EQ(
      (axis == 0 || axis == -1),
      true,
      common::errors::InvalidArgument(
          "Attribute(axis) of OverlapAddOp should be 0 or -1, but got %s.",
          axis));

  std::vector<symbol::DimExpr> output_shape;
  symbol::DimExpr n_frames;
  symbol::DimExpr frame_length;
  symbol::DimExpr seq_length;

  int start_axis = 0;
  int end_axis = 0;
  if (axis == 0) {
    n_frames = x_dims[0];
    frame_length = x_dims[0];
    start_axis = 2;
    end_axis = x_rank - 1;
  } else {
    n_frames = x_dims[x_rank - 1];
    frame_length = x_dims[x_rank - 2];
    start_axis = 0;
    end_axis = x_rank - 3;
  }

  seq_length = (n_frames - symbol::DimExpr(1)) * symbol::DimExpr(hop_length) +
               frame_length;

  for (int i = start_axis; i <= end_axis; i++) {
    output_shape.push_back(x_dims[i]);
  }

  if (axis == 0) {
    // (seq_length, ...)
    output_shape.insert(output_shape.begin(), seq_length);
  } else {
    // (..., seq_length)
    output_shape.push_back(seq_length);
  }

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(output_shape)});

  return true;
}

bool PixelShuffleOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const auto x_shape = x_shape_or_data.shape();

  const auto attributes = op->attributes();
  const int upscale_factor =
      attributes.at("upscale_factor").dyn_cast<pir::Int32Attribute>().data();
  const std::string &data_format =
      op->attribute<pir::StrAttribute>("data_format").AsString();

  PADDLE_ENFORCE_EQ(x_shape.size(),
                    4,
                    common::errors::InvalidArgument(
                        "Input should be a 4-D tensor of format [N, C, H, W] "
                        "or [N, H, W, C], but got %u.",
                        x_shape.size()));

  PADDLE_ENFORCE_NE(
      upscale_factor,
      0,
      common::errors::InvalidArgument("upscale_factor should not be 0."));

  PADDLE_ENFORCE_EQ(
      data_format == "NCHW" || data_format == "NHWC",
      true,
      common::errors::InvalidArgument("data_format must be one of NCHW and "
                                      "NHWC. But received data_format: %s",
                                      data_format));

  const bool channel_last = (data_format == "NHWC");

  // the number of channels should be able to be divided by the upscale_factor
  // ^ 2.
  // TODO(Lans1ot, Buaa): add constrain for the channel number and
  // upscale_factor

  auto output_shape = x_shape;
  output_shape[0] = x_shape[0];

  const auto upscale_factor_ = symbol::DimExpr(upscale_factor);

  if (!channel_last) {
    output_shape[1] = x_shape[1] / (upscale_factor_ * upscale_factor_);
    output_shape[2] = x_shape[2] * upscale_factor_;
    output_shape[3] = x_shape[3] * upscale_factor_;
  } else {
    output_shape[1] = x_shape[1] * upscale_factor_;
    output_shape[2] = x_shape[2] * upscale_factor_;
    output_shape[3] = x_shape[3] / (upscale_factor_ * upscale_factor_);
  }
  infer_context->SetShapeOrDataForValue(
      op->result(0), symbol::TensorShapeOrDataDimExprs(output_shape));
  return true;
}

bool PixelUnshuffleOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const symbol::ShapeOrDataDimExprs &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const std::vector<symbol::DimExpr> x_shape = x_shape_or_data.shape();

  const pir::AttributeMap attributes = op->attributes();
  const int downscale_factor =
      attributes.at("downscale_factor").dyn_cast<pir::Int32Attribute>().data();
  const std::string &data_format =
      op->attribute<pir::StrAttribute>("data_format").AsString();

  PADDLE_ENFORCE_EQ(x_shape.size(),
                    4,
                    common::errors::InvalidArgument(
                        "Input should be a 4-D tensor of format [N, C, H, W] "
                        "or [N, H, W, C], but got %u.",
                        x_shape.size()));

  PADDLE_ENFORCE_GE(downscale_factor,
                    1,
                    common::errors::InvalidArgument(
                        "downscale_factor should be larger than 0."));

  PADDLE_ENFORCE_EQ(
      data_format == "NCHW" || data_format == "NHWC",
      true,
      common::errors::InvalidArgument("data_format must be one of NCHW and "
                                      "NHWC. But received data_format: %s",
                                      data_format));

  // the number of height and width should be able to be divided by the
  // upscale_factor ^ 2.
  // TODO(Lans1ot, Buaa): add constrain for the height, width and upscale_factor

  const bool channel_last = (data_format == "NHWC");

  std::vector<symbol::DimExpr> output_shape = x_shape;
  const symbol::DimExpr downscale_factor_(downscale_factor);
  output_shape[0] = x_shape[0];
  if (!channel_last) {
    output_shape[1] = x_shape[1] * (downscale_factor_ * downscale_factor_);
    output_shape[2] = x_shape[2] / downscale_factor_;
    output_shape[3] = x_shape[3] / downscale_factor_;
  } else {
    output_shape[1] = x_shape[1] / downscale_factor_;
    output_shape[2] = x_shape[2] / downscale_factor_;
    output_shape[3] = x_shape[3] * (downscale_factor_ * downscale_factor_);
  }
  infer_context->SetShapeOrDataForValue(
      op->result(0), symbol::TensorShapeOrDataDimExprs(output_shape));

  return true;
}

bool PNormOpInferSymbolicShape(pir::Operation *op,
                               pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const auto &x_shape = x_shape_or_data.shape();
  int x_rank = x_shape.size();

  int axis = op->attribute<pir::Int32Attribute>("axis").data();
  bool keepdim = op->attribute<pir::BoolAttribute>("keepdim").data();
  bool asvector = op->attribute<pir::BoolAttribute>("asvector").data();

  if (axis < 0) {
    axis += x_rank;
  }

  bool axis_valid = (axis >= 0) && (axis < x_rank);

  PADDLE_ENFORCE_EQ(
      axis_valid,
      true,
      common::errors::InvalidArgument(
          "Attr(axis) value should be in range [-R, R-1], R is the rank of "
          "Input(X). "
          "But received axis: %d, R: %d. Current Input(X)'s shape is=[%s].",
          axis,
          x_rank,
          x_shape));

  std::vector<symbol::DimExpr> out_shape;

  if (asvector) {
    if (keepdim) {
      for (int i = 0; i < x_rank; ++i) {
        out_shape.emplace_back(symbol::DimExpr(1));
      }
    } else {
      out_shape = {};
    }
  } else {
    if (keepdim) {
      for (int i = 0; i < x_rank; ++i) {
        if (i == axis) {
          out_shape.emplace_back(symbol::DimExpr(1));
        } else {
          out_shape.emplace_back(x_shape[i]);
        }
      }
    } else {
      for (int i = 0; i < x_rank; ++i) {
        if (i != axis) {
          out_shape.emplace_back(x_shape[i]);
        }
      }
    }
  }

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(out_shape)});

  return true;
}

bool PartialSumOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const symbol::TensorListShapeOrDataDimExprs &xs_shapes =
      infer_context->GetShapeOrDataForValue(op->operand_source(0))
          .dyn_cast<symbol::TensorListShapeOrDataDimExprs>();

  int inputs_num = xs_shapes.size();
  PADDLE_ENFORCE_GT(inputs_num,
                    0,
                    common::errors::InvalidArgument(
                        "ShapeError: Input tensors count should > 0. But "
                        "received inputs' length is 0."));
  if (inputs_num == 1) {
    VLOG(3) << "Warning: partial_sum op have only one input, may be useless";
  }

  symbol::DimExpr batch_size = xs_shapes[0].shape()[0];
  symbol::DimExpr input_len = xs_shapes[1].shape()[1];

  for (int i = 0; i < inputs_num; i++) {
    const std::vector<symbol::DimExpr> x_shape = xs_shapes[i].shape();
    PADDLE_ENFORCE_EQ(x_shape.size(),
                      2,
                      common::errors::InvalidArgument(
                          "Only support two dimensions input now."));

    if (i > 0) {
      infer_context->AddEqualCstr(x_shape[0], batch_size);
      infer_context->AddEqualCstr(x_shape[1], input_len);
    }
  }

  int start_index = op->attribute<pir::Int32Attribute>("start_index").data();
  int length = op->attribute<pir::Int32Attribute>("length").data();

  std::vector<symbol::DimExpr> output_shape(2);
  output_shape[0] = batch_size;
  output_shape[1] = (length == -1) ? input_len - symbol::DimExpr(start_index)
                                   : symbol::DimExpr(length);

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(output_shape)});

  return true;
}

bool PadOpInferSymbolicShape(pir::Operation *op,
                             pir::InferSymbolicShapeContext *infer_context) {
  // input(0): Tensor x
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  PADDLE_ENFORCE_EQ(x_shape_or_data.data().has_value(),
                    false,
                    common::errors::InvalidArgument(
                        "InferSymbolicShape of PadOp only support input with "
                        "value now."));
  const auto &x_dims_sym = x_shape_or_data.shape();
  const size_t rank = x_dims_sym.size();

  // input(1): int[] paddings
  std::vector<int> paddings =
      paddle::dialect::details::GetVectorAttr<int>(op, "paddings");
  PADDLE_ENFORCE_EQ(rank * 2,
                    paddings.size(),
                    common::errors::InvalidArgument(
                        "The size of paddings should be 2 * input's rank. But "
                        "got paddings.size() = %d, input's rank = %d.",
                        paddings.size(),
                        rank));

  // output
  const auto &out_dims = [&] {
    std::vector<symbol::DimExpr> out_dims;
    out_dims.reserve(rank);
    for (size_t i = 0; i < rank; ++i) {
      out_dims.push_back(x_dims_sym.at(i) + paddings.at(2 * i) +
                         paddings.at(2 * i + 1));
    }
    return out_dims;
  }();

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(out_dims)});

  return true;
}

bool Pad3dOpInferSymbolicShape(pir::Operation *op,
                               pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(0)).shape();
  PADDLE_ENFORCE_EQ(x_shape.size(),
                    5,
                    common::errors::InvalidArgument(
                        "The size of Input(X)'s dimension should be equal to "
                        "5, but received %d. ",
                        x_shape.size()));
  const auto &paddings_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));
  const std::string &data_format =
      op->attribute<pir::StrAttribute>("data_format").AsString();
  const std::vector<symbol::DimExpr> &paddings =
      paddle::dialect::details::GetDataFromTensorOrTensorList(paddings_shape);
  const std::vector<symbol::DimExpr> &out_dims = [&] {
    std::vector<symbol::DimExpr> out_dims = x_shape;
    PADDLE_ENFORCE_EQ(paddings.size(),
                      6,
                      common::errors::InvalidArgument(
                          "Shape of Input(Paddings) should be equal to "
                          "[6], but received [%d].",
                          paddings.size()));
    if (data_format == "NCDHW") {
      out_dims.at(1) = x_shape.at(1);
      out_dims.at(2) = x_shape.at(2) + paddings.at(4) + paddings.at(5);
      out_dims.at(3) = x_shape.at(3) + paddings.at(2) + paddings.at(3);
      out_dims.at(4) = x_shape.at(4) + paddings.at(0) + paddings.at(1);
    } else {
      out_dims.at(1) = x_shape.at(1) + paddings.at(4) + paddings.at(5);
      out_dims.at(2) = x_shape.at(2) + paddings.at(2) + paddings.at(3);
      out_dims.at(3) = x_shape.at(3) + paddings.at(0) + paddings.at(1);
      out_dims.at(4) = x_shape.at(4);
    }
    return out_dims;
  }();

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(out_dims)});

  return true;
}

bool Pool2dOpInferSymbolicShape(pir::Operation *op,
                                pir::InferSymbolicShapeContext *infer_context) {
  const auto &kernel_size_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));
  const auto &kernel_size =
      paddle::dialect::details::GetExprVecFromData(kernel_size_shape_or_data);
  infer_context->SetShapeOrDataForValue(
      op->result(0),
      Pool2dRawInferSymbolicShape(op, kernel_size, infer_context));
  return true;
}

bool Pool3dOpInferSymbolicShape(pir::Operation *op,
                                pir::InferSymbolicShapeContext *infer_context) {
  std::vector<int> kernel_size_ =
      paddle::dialect::details::GetVectorAttr<int>(op, "kernel_size");
  std::vector<symbol::DimExpr> kernel_size;
  for (size_t i = 0; i < kernel_size_.size(); ++i) {
    kernel_size.push_back(symbol::DimExpr(kernel_size_[i]));
  }
  infer_context->SetShapeOrDataForValue(
      op->result(0),
      Pool2dRawInferSymbolicShape(op, kernel_size, infer_context));
  return true;
}

// bool PoolOpInferSymbolicShape(pir::Operation *op,
//                               pir::InferSymbolicShapeContext *infer_context)
//                               {
//   // pass
//   return true;
// }

bool ProdOpInferSymbolicShape(pir::Operation *op,
                              pir::InferSymbolicShapeContext *infer_context) {
  bool keepdim = GetBoolAttr(op, "keepdim");
  bool reduce_all = GetBoolAttr(op, "reduce_all");

  std::vector<int64_t> axis;
  if (paddle::dialect::details::GetAxisFromOpInput(
          op->operand_source(1), infer_context, &axis)) {
    if (axis.size() == 0) {
      reduce_all = true;
    }

    return paddle::dialect::details::ReduceInferDim(
        op, infer_context, axis, keepdim, reduce_all);
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "Reduction[Sum|Max|Prod|Mean..] OpInferSymbolicShape: 'axis' only "
        "support FullIntArrayOp's result or constant DimExpr now."));
  }
  return false;
}

bool QrOpInferSymbolicShape(pir::Operation *op,
                            pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const std::vector<symbol::DimExpr> &x_shape = x_shape_or_data.shape();
  int x_rank = x_shape.size();

  PADDLE_ENFORCE_GE(
      x_rank,
      2,
      common::errors::InvalidArgument("the rank of input must greater than 2"));

  bool compute_q = false;
  bool reduced = false;
  const std::string &mode = op->attribute<pir::StrAttribute>("mode").AsString();
  if (mode == "reduced") {
    compute_q = true;
    reduced = true;
  } else if (mode == "complete") {
    compute_q = true;
    reduced = false;
  } else if (mode == "r") {
    compute_q = false;
    reduced = true;
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "QR received unrecognized mode '%s'"
        " but expected one of 'reduced' (default), 'r', or 'complete'",
        mode));
  }

  symbol::DimExpr m = x_shape[x_rank - 2];
  symbol::DimExpr n = x_shape[x_rank - 1];
  symbol::DimExprBuilder builder;
  symbol::DimExpr min_mn = builder.Min(m, n);

  if (compute_q) {
    symbol::DimExpr k = reduced ? min_mn : m;
    std::vector<symbol::DimExpr> q_shape = x_shape;
    q_shape[x_rank - 1] = k;
    infer_context->SetShapeOrDataForValue(
        op->result(0),
        symbol::ShapeOrDataDimExprs{
            symbol::TensorShapeOrDataDimExprs(q_shape)});
  } else {
    std::vector<symbol::DimExpr> q_shape = {0};
    infer_context->SetShapeOrDataForValue(
        op->result(0),
        symbol::ShapeOrDataDimExprs{
            symbol::TensorShapeOrDataDimExprs(q_shape)});
  }

  symbol::DimExpr k = reduced ? min_mn : m;
  std::vector<symbol::DimExpr> r_shape = x_shape;
  r_shape[x_rank - 2] = k;
  r_shape[x_rank - 1] = n;
  infer_context->SetShapeOrDataForValue(
      op->result(1),
      symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(r_shape)});

  return true;
}

bool RepeatInterleaveOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  pir::Value operand_source = op->operand_source(0);
  const symbol::ShapeOrDataDimExprs &operand_shape_or_data =
      infer_context->GetShapeOrDataForValue(operand_source);

  const auto &attributes = op->attributes();
  int repeats = attributes.at("repeats").dyn_cast<pir::Int32Attribute>().data();
  // what should I do if axis is null
  int axis = attributes.at("axis").dyn_cast<pir::Int32Attribute>().data();

  const std::vector<symbol::DimExpr> &in_dims_sym = [&] {
    std::vector<symbol::DimExpr> dims;
    if (operand_shape_or_data.data().has_value()) {
      dims = operand_shape_or_data.data().value();
    } else {
      dims = operand_shape_or_data.shape();
    }
    return dims;
  }();

  int x_rank = in_dims_sym.size();
  if (axis < 0) axis += x_rank;

  const auto &out_sym_shape = [&] {
    std::vector<symbol::DimExpr> out_sym_shape;
    for (int i = 0; i < x_rank; i++) {
      if (i == axis) {
        out_sym_shape.push_back(in_dims_sym.at(i) * repeats);
      } else {
        out_sym_shape.push_back(in_dims_sym.at(i));
      }
    }
    return out_sym_shape;
  }();

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(out_sym_shape)});

  return true;
}

bool ReshapeOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const symbol::ShapeOrDataDimExprs &x_dim_expr =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const symbol::ShapeOrDataDimExprs &shape_dim_expr =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));

  const auto &GetProduct = [&](const auto &dim_exprs, const auto &Filter) {
    symbol::DimExpr product{1};
    for (const auto &dim_expr : dim_exprs) {
      if (Filter(dim_expr)) {
        product = product * dim_expr;
      }
    }
    return product;
  };

  const auto &IsNotMinusOne = [&](const symbol::DimExpr &dim_expr) {
    if (dim_expr.isa<int64_t>()) {
      return dim_expr.dyn_cast<int64_t>() != static_cast<int64_t>(-1);
    }
    return true;
  };

  const auto &IsPositiveInteger = [&](const symbol::DimExpr &dim_expr) {
    if (dim_expr.isa<int64_t>()) {
      return dim_expr.dyn_cast<int64_t>() > static_cast<int64_t>(0);
    }
    return true;
  };

  const auto &IsZero = [&](const symbol::DimExpr &dim_expr) {
    if (dim_expr.isa<int64_t>()) {
      return dim_expr.dyn_cast<int64_t>() == static_cast<int64_t>(0);
    }
    return false;
  };

  const auto &original_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(0)).shape();
  const auto &input_numel =
      GetProduct(original_shape, [](const auto &) { return true; });

  const std::vector<symbol::DimExpr> out_dims = [&] {
    ExprVec target_shape = paddle::dialect::details::GetOrCreateExprVecFromData(
        shape_dim_expr, infer_context);

    // replace '0' with original shape
    for (size_t i = 0; i < target_shape.size(); i++) {
      if (i < original_shape.size() && IsZero(target_shape.at(i))) {
        target_shape.at(i) = original_shape.at(i);
      }
    }

    // replace '-1' with inferred shape

    const auto &product_exclude_minus_one =
        GetProduct(target_shape, IsPositiveInteger);
    const auto &input_dims = target_shape;

    std::vector<symbol::DimExpr> out_dims;
    out_dims.reserve(input_dims.size());
    for (size_t i = 0; i < input_dims.size(); ++i) {
      auto out_dim_expr = IsNotMinusOne(input_dims.at(i))
                              ? input_dims.at(i)
                              : (input_numel / product_exclude_minus_one);
      out_dims.emplace_back(out_dim_expr);
    }
    return out_dims;
  }();

  symbol::ShapeOrDataDimExprs shape_data = [&] {
    if (x_dim_expr.data().has_value()) {
      return symbol::TensorShapeOrDataDimExprs(out_dims,
                                               x_dim_expr.data().value());
    }
    return symbol::TensorShapeOrDataDimExprs(out_dims);
  }();

  const auto &output_numel =
      GetProduct(out_dims, [](const auto &) { return true; });

  infer_context->AddEqualCstr(input_numel, output_numel);

  infer_context->SetShapeOrDataForValue(op->result(0), shape_data);
  return true;
}

bool Reshape_OpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  return ReshapeOpInferSymbolicShape(op, infer_context);
}

bool ShapeOpInferSymbolicShape(pir::Operation *op,
                               pir::InferSymbolicShapeContext *infer_context) {
  const symbol::ShapeOrDataDimExprs &operand_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const auto &out_data = operand_shape_or_data.shape();
  const std::vector<symbol::DimExpr> shape{std::int64_t(out_data.size())};
  symbol::ShapeOrDataDimExprs shape_or_data{
      symbol::TensorShapeOrDataDimExprs(shape, out_data)};

  infer_context->SetShapeOrDataForValue(op->result(0), shape_or_data);
  return true;
}

bool Shape64OpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const symbol::ShapeOrDataDimExprs &operand_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const auto &out_data = operand_shape_or_data.shape();
  const std::vector<symbol::DimExpr> shape{std::int64_t(out_data.size())};
  symbol::ShapeOrDataDimExprs shape_or_data{
      symbol::TensorShapeOrDataDimExprs(shape, out_data)};

  infer_context->SetShapeOrDataForValue(op->result(0), shape_or_data);
  return true;
}

bool ShardIndexOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &in_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const std::vector<symbol::DimExpr> &in_shape = in_shape_or_data.shape();
  PADDLE_ENFORCE_GE(
      in_shape.size(),
      2,
      common::errors::InvalidArgument("Rank of Input(X) should be at least 2, "
                                      "but the value given is %d.",
                                      in_shape.size()));
  infer_context->AddEqualCstr(in_shape[in_shape.size() - 1],
                              symbol::DimExpr{1});
  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(in_shape)});
  return true;
}

bool RreluOpInferSymbolicShape(pir::Operation *op,
                               pir::InferSymbolicShapeContext *infer_context) {
  float lower = op->attribute<pir::FloatAttribute>("lower").data();
  float upper = op->attribute<pir::FloatAttribute>("upper").data();
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const auto &x_shape = x_shape_or_data.shape();

  // Check constraints for the attributes lower and upper
  PADDLE_ENFORCE_GE(lower,
                    0,
                    common::errors::InvalidArgument(
                        "The lower value should be greater than or equal to 0. "
                        "But received lower value = %f.",
                        lower));
  PADDLE_ENFORCE_LE(upper,
                    1,
                    common::errors::InvalidArgument(
                        "The upper value should be less than or equal to 1. "
                        "But received upper value = %f.",
                        upper));
  PADDLE_ENFORCE_GE(
      upper,
      lower,
      common::errors::InvalidArgument(
          "The upper value should be greater than or equal to lower value. "
          "But received upper value = %f, lower value = %f.",
          upper,
          lower));

  // Set the shape for the output tensor out
  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(x_shape)});

  // Set the shape for the output tensor noise if it exists
  if (op->num_results() > 1 && op->result(1) != nullptr) {
    infer_context->SetShapeOrDataForValue(
        op->result(1),
        symbol::ShapeOrDataDimExprs{
            symbol::TensorShapeOrDataDimExprs(x_shape)});
  }

  return true;
}

bool SequencePoolOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const symbol::ShapeOrDataDimExprs &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const std::vector<symbol::DimExpr> &x_shape = x_shape_or_data.shape();
  std::string pooltype =
      op->attribute<pir::StrAttribute>("pooltype").AsString();

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(x_shape)});

  if (pooltype == "MAX") {
    infer_context->SetShapeOrDataForValue(
        op->result(1),
        symbol::ShapeOrDataDimExprs{
            symbol::TensorShapeOrDataDimExprs(x_shape)});
  } else {
    infer_context->SetSymbolForValueByStaticShape(op->result(1));
  }

  return true;
}

bool ShapeSrOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  return ShapeOpInferSymbolicShape(op, infer_context);
}

bool Shape64SrOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  return Shape64OpInferSymbolicShape(op, infer_context);
}
// bool ShardIndexOpInferSymbolicShape(pir::Operation *op,
//                                     pir::InferSymbolicShapeContext
//                                     *infer_context) {
//   // pass
//   return true;
// }

bool ShuffleChannelOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const auto &x_dims = x_shape_or_data.shape();

  PADDLE_ENFORCE_EQ(
      x_dims.size(),
      4,
      common::errors::InvalidArgument("The layout of input is NCHW."));

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs(symbol::TensorShapeOrDataDimExprs(x_dims)));

  return true;
}

bool SliceOpInferSymbolicShape(pir::Operation *op,
                               pir::InferSymbolicShapeContext *infer_context) {
  pir::Value operand_source = op->operand_source(0);
  pir::Value res = op->result(0);

  std::vector<int64_t> axes_vec = details::GetVectorAttr(op, "axes");
  std::vector<int64_t> infer_flags = details::GetVectorAttr(op, "infer_flags");
  const std::vector<int64_t> decrease_axis =
      details::GetVectorAttr(op, "decrease_axis");

  auto GetExprVec = [&](std::vector<symbol::DimExpr> *expr_vec,
                        const int &operand_idx,
                        const std::string &attr_name) -> bool {
    if (op->operand_source(operand_idx)) {
      const symbol::ShapeOrDataDimExprs &se_shape_data =
          infer_context->GetShapeOrDataForValue(
              op->operand_source(operand_idx));
      if (slice_utils::GetExprVecOfStartEnd(se_shape_data, expr_vec)) {
        return true;
      }
      PADDLE_ENFORCE_EQ(
          se_shape_data.shape().at(0).isa<std::int64_t>() &&
              (static_cast<int64_t>(axes_vec.size()) ==
               se_shape_data.shape().at(0).dyn_cast<std::int64_t>()),
          true,
          common::errors::InvalidArgument(
              "The size of axes must equal size of starts and ends."));
      return false;
    } else {
      if (op->attributes().find(attr_name) != op->attributes().end()) {
        const std::vector<int64_t> se_raw =
            paddle::dialect::details::GetVectorAttr(op, attr_name);
        for (const int64_t &se : se_raw) {
          expr_vec->push_back(symbol::DimExpr{se});
        }
        return true;
      }
      return false;
    }
  };

  std::vector<symbol::DimExpr> starts;
  std::vector<symbol::DimExpr> ends;
  if (!GetExprVec(&starts, 1, "starts") || !GetExprVec(&ends, 2, "ends")) {
    const auto &in_shapeordata =
        infer_context->GetShapeOrDataForValue(op->operand_source(0));
    // NOTE(gongshaotian): When there is no data value in the starts and ends
    // parameters, only the shape value is processed regardless of whether the
    // input has a data value, and the  data value is no longer processed.
    std::vector<symbol::DimExpr> out_shape = in_shapeordata.shape();
    for (size_t i = 0; i < axes_vec.size(); i++) {
      int64_t axis = axes_vec[i];
      out_shape[axis] = infer_context->GetNextSymName();
    }
    ExprVec out_dims = paddle::dialect::slice_utils::GetDecreasedDims(
        out_shape, decrease_axis);
    infer_context->SetShapeOrDataForValue(
        res,
        symbol::ShapeOrDataDimExprs{
            symbol::TensorShapeOrDataDimExprs(out_dims)});
    return true;
  }

  infer_context->SetShapeOrDataForValue(
      res,
      slice_utils::SliceRawInferSymbolicShape(operand_source,
                                              res,
                                              starts,
                                              ends,
                                              axes_vec,
                                              infer_flags,
                                              decrease_axis,
                                              infer_context));

  return true;
}

bool SlogdetOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const auto &x_shape = x_shape_or_data.shape();
  size_t x_shape_size = x_shape.size();
  PADDLE_ENFORCE_GE(
      x_shape_size,
      2,
      common::errors::InvalidArgument("the input matrix dimension size should "
                                      "greater than or equal to 2."));
  infer_context->AddEqualCstr(x_shape[x_shape_size - 1],
                              x_shape[x_shape_size - 2]);
  std::vector<symbol::DimExpr> out_shape = {2};
  size_t additional_dims = x_shape.size() - 2;
  for (size_t i = 0; i < additional_dims; i++) {
    out_shape.push_back(x_shape[i]);
  }
  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(out_shape)});
  return true;
}

bool SplitOpInferSymbolicShape(pir::Operation *op,
                               pir::InferSymbolicShapeContext *infer_context) {
  // input
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  PADDLE_ENFORCE_EQ(x_shape_or_data.data().has_value(),
                    false,
                    common::errors::InvalidArgument(
                        "InferSymbolicShape of SplitOp only support input with "
                        "value now."));
  const auto &x_dims_sym = x_shape_or_data.shape();

  // axis
  PADDLE_ENFORCE_EQ(
      op->operand_source(2).defining_op()->isa<paddle::dialect::FullOp>(),
      true,
      common::errors::InvalidArgument(
          "Invalid input args : axis, please check"));

  int64_t axis = op->operand_source(2)
                     .defining_op<paddle::dialect::FullOp>()
                     .attributes()
                     .at("value")
                     .dyn_cast<paddle::dialect::ScalarAttribute>()
                     .data()
                     .to<int64_t>();
  size_t rank = x_dims_sym.size();
  axis = axis >= 0 ? axis : std::max(int64_t(0), int64_t(axis + rank));

  // sections
  const std::vector<symbol::DimExpr> &sections_sym =
      details::GetExprVecFromData(
          infer_context->GetShapeOrDataForValue(op->operand_source(1)));

  // output
  const symbol::TensorListShapeOrDataDimExprs &output_shape_data_list = [&] {
    const auto &GetSum = [&](const auto &dim_exprs, const auto &Filter) {
      symbol::DimExpr sum{0};
      for (const auto &dim_expr : dim_exprs) {
        if (Filter(dim_expr)) {
          sum = sum + dim_expr;
        }
      }
      return sum;
    };
    const auto &All = [&](const auto &dim_exprs, const auto &Cond) {
      for (const auto &dim_expr : dim_exprs) {
        if (!Cond(dim_expr)) {
          return false;
        }
      }
      return true;
    };
    const auto &IsNotMinusOne = [&](const symbol::DimExpr &dim_expr) {
      if (dim_expr.isa<int64_t>()) {
        return dim_expr.dyn_cast<int64_t>() != static_cast<int64_t>(-1);
      }
      return true;
    };
    const auto &sum_exclude_minus_one = GetSum(sections_sym, IsNotMinusOne);

    const bool &all_sections_sym_not_minus_one =
        All(sections_sym, IsNotMinusOne);
    if (all_sections_sym_not_minus_one) {
      infer_context->AddEqualCstr(x_dims_sym.at(axis), sum_exclude_minus_one);
    }

    symbol::TensorListShapeOrDataDimExprs shape_data_list;
    std::vector<symbol::DimExpr> output_dims_sym = x_dims_sym;
    if (!all_sections_sym_not_minus_one && sections_sym.size() == 1) {
      VLOG(3) << "[SplitOp]-1 is the only split section. The output shape is "
                 "identical to the input shape.";
      shape_data_list.push_back(
          symbol::TensorShapeOrDataDimExprs(output_dims_sym));
      return shape_data_list;
    }
    for (uint32_t idx = 0; idx < sections_sym.size(); idx++) {
      const auto &section_sym = sections_sym.at(idx);
      output_dims_sym.at(axis) =
          IsNotMinusOne(section_sym)
              ? section_sym
              : x_dims_sym.at(axis) - sum_exclude_minus_one;

      shape_data_list.push_back(
          symbol::TensorShapeOrDataDimExprs(output_dims_sym));
    }
    return shape_data_list;
  }();

  infer_context->SetShapeOrDataForValue(
      op->result(0), symbol::ShapeOrDataDimExprs{output_shape_data_list});

  return true;
}

bool SplitWithNumOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const symbol::ShapeOrDataDimExprs &axis_shape_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));
  PADDLE_ENFORCE_EQ(
      axis_shape_data.data().has_value(),
      true,
      common::errors::InvalidArgument(
          "In InferSymbolicShape, axis of SplitWithNumOp is null"));
  const std::vector<symbol::DimExpr> &axis_data =
      axis_shape_data.data().value();
  PADDLE_ENFORCE_EQ(
      axis_data.size() == 1,
      true,
      common::errors::InvalidArgument(
          "In SplitWithNumOp, data of axis should be one dimension"));

  const auto &attributes = op->attributes();
  int num = attributes.at("num").dyn_cast<pir::Int32Attribute>().data();

  const auto &x_s_or_d =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  int rank = x_s_or_d.shape().size();

  const auto &out_s_d = [&](int64_t split_axis, int64_t res_num) {
    symbol::DimExpr input_axis_dim = x_s_or_d.shape().at(split_axis);
    symbol::DimExpr axis_shape = input_axis_dim / symbol::DimExpr{res_num};
    if (res_num > 1) {
      infer_context->AddGreatThanOneCstr(input_axis_dim);
    }

    std::vector<symbol::DimExpr> res_s_d;
    for (size_t i = 0; i < x_s_or_d.shape().size(); ++i) {
      const auto &sym_dim = split_axis == static_cast<int64_t>(i)
                                ? axis_shape
                                : x_s_or_d.shape().at(i);
      res_s_d.push_back(sym_dim);
    }
    return symbol::TensorShapeOrDataDimExprs(res_s_d);
  };

  if (axis_data.at(0).isa<int64_t>()) {
    // case 1: DimExpr of axis is int.  axis_shape_or_data: {shape:{1},
    // data:{3}} eg: axis generator op is full_op and assign_op
    int64_t axis = axis_data[0].dyn_cast<int64_t>();
    axis = axis < 0 ? axis + rank : axis;
    symbol::TensorListShapeOrDataDimExprs res_list_s_d(num, out_s_d(axis, num));
    infer_context->SetShapeOrDataForValue(
        op->result(0), symbol::ShapeOrDataDimExprs{res_list_s_d});
  } else if (axis_data.at(0).isa<std::string>()) {
    // case 2: DimExpr of axis is a symbol(string).  axis_shape_or_data:
    // {shape:{1}, data:{s0}} eg: axis generator op is data_op
    int candidate_axis = -1;
    int count = 0;
    for (size_t i = 0; i < x_s_or_d.shape().size(); ++i) {
      if (x_s_or_d.shape().at(i).isa<int64_t>()) {
        if (x_s_or_d.shape().at(i).dyn_cast<int64_t>() % num == 0) {
          count++;
          candidate_axis = i;
        }
      } else {
        PADDLE_THROW(
            common::errors::InvalidArgument("The type of X must be int64_t."));
      }
    }
    if (count == 1) {
      // calculate the axis of split_with_num_op
      symbol::TensorListShapeOrDataDimExprs res_list_s_d(
          num, out_s_d(candidate_axis, num));
      infer_context->SetShapeOrDataForValue(
          op->result(0), symbol::ShapeOrDataDimExprs{res_list_s_d});
    } else {
      // create new Symbol
      std::vector<symbol::DimExpr> res_s;
      for (size_t i = 0; i < x_s_or_d.shape().size(); ++i) {
        const auto &s_dim =
            x_s_or_d.shape().at(i).dyn_cast<std::int64_t>() % num == 0
                ? symbol::DimExpr(infer_context->GetNextSymName())
                : x_s_or_d.shape().at(i);
        res_s.emplace_back(s_dim);
      }
      const symbol::TensorShapeOrDataDimExprs &res_s_d =
          symbol::TensorShapeOrDataDimExprs(res_s);
      symbol::TensorListShapeOrDataDimExprs res_list_s_d(num, res_s_d);
      infer_context->SetShapeOrDataForValue(
          op->result(0), symbol::ShapeOrDataDimExprs{res_list_s_d});
    }
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "The type of axis must be int64_t or string."));
  }
  return true;
}

// bool StridedSliceOpInferSymbolicShape(pir::Operation *op,
//                                       pir::InferSymbolicShapeContext
//                                       *infer_context) {
//   // pass
//   return true;
// }

bool SumOpInferSymbolicShape(pir::Operation *op,
                             pir::InferSymbolicShapeContext *infer_context) {
  bool keepdim = GetBoolAttr(op, "keepdim");

  std::vector<int64_t> axis;
  if (paddle::dialect::details::GetAxisFromOpInput(
          op->operand_source(1), infer_context, &axis)) {
    bool reduce_all = (axis.size() == 0);

    return details::ReduceInferDim(
        op, infer_context, axis, keepdim, reduce_all);
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "Reduction[Sum|Max|Prod|Mean..] OpInferSymbolicShape: 'axis' only "
        "support FullIntArrayOp's result or constant DimExpr now."));
  }

  return false;
}

bool SvdOpInferSymbolicShape(pir::Operation *op,
                             pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const std::vector<symbol::DimExpr> &x_shape = x_shape_or_data.shape();
  bool full_matrices =
      op->attribute<pir::BoolAttribute>("full_matrices").data();

  const int x_rank = x_shape.size();
  PADDLE_ENFORCE_GE(
      x_rank,
      2,
      common::errors::InvalidArgument(
          "the rank of input must be greater than or equal to 2"));

  const symbol::DimExpr m = x_shape[x_rank - 2];
  const symbol::DimExpr n = x_shape[x_rank - 1];
  symbol::DimExprBuilder builder;
  const symbol::DimExpr k = builder.Min(m, n);

  auto UDDim = [&](const std::vector<symbol::DimExpr> &x_shape,
                   const symbol::DimExpr &k) {
    std::vector<symbol::DimExpr> x_vec = x_shape;
    x_vec[x_vec.size() - 1] = k;
    return x_vec;
  };

  auto VHDDim = [&](const std::vector<symbol::DimExpr> &x_shape,
                    const symbol::DimExpr &k) {
    std::vector<symbol::DimExpr> x_vec = x_shape;
    x_vec[x_vec.size() - 2] = k;
    return x_vec;
  };

  auto SDDim = [&](const std::vector<symbol::DimExpr> &x_shape,
                   const symbol::DimExpr &k) {
    std::vector<symbol::DimExpr> x_vec = x_shape;
    x_vec[x_vec.size() - 2] = k;
    x_vec.erase(x_vec.end() - 1);  // rank - 1
    return x_vec;
  };

  std::vector<symbol::DimExpr> u_shape =
      !full_matrices ? UDDim(x_shape, k) : UDDim(x_shape, m);
  std::vector<symbol::DimExpr> vh_shape =
      !full_matrices ? VHDDim(x_shape, k) : VHDDim(x_shape, n);
  std::vector<symbol::DimExpr> s_shape = SDDim(x_shape, k);

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(u_shape)});
  infer_context->SetShapeOrDataForValue(
      op->result(1),
      symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(s_shape)});
  infer_context->SetShapeOrDataForValue(
      op->result(2),
      symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(vh_shape)});
  return true;
}

bool SetValueOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &input_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const auto &input_shape = input_shape_or_data.shape();
  PADDLE_ENFORCE_LT(
      input_shape.size(),
      7,
      common::errors::InvalidArgument("The SetValueOp's rank of input should "
                                      "be less than 7, but received %d.",
                                      input_shape.size()));

  infer_context->SetShapeOrDataForValue(
      op->result(0), symbol::TensorShapeOrDataDimExprs(input_shape));
  return true;
}

bool SetValue_OpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  return SetValueOpInferSymbolicShape(op, infer_context);
}

bool SetValueWithTensorOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  return SetValueOpInferSymbolicShape(op, infer_context);
}

bool SetValueWithTensor_OpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  return SetValueOpInferSymbolicShape(op, infer_context);
}

// bool TensorUnfoldOpInferSymbolicShape(
//     pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
//   // pass
//   return true;
// }

// bool TraceOpInferSymbolicShape(pir::Operation *op,
//                                pir::InferSymbolicShapeContext *infer_context)
//                                {
//   // pass
//   return true;
// }

bool SquaredL2NormOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  auto dtype = infer_context->GetShapeOrDataForValue(op->operand_source(0));
  std::vector<symbol::DimExpr> batch_dims;
  batch_dims.push_back(symbol::DimExpr(1));
  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs(
          symbol::TensorShapeOrDataDimExprs(batch_dims)));

  return true;
}

bool TemporalShiftOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const symbol::ShapeOrDataDimExprs &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const auto &x_dims = x_shape_or_data.shape();

  PADDLE_ENFORCE_EQ(
      x_dims.size(),
      4,
      common::errors::InvalidArgument("Input(X) rank should be 4 in shape of "
                                      "[N*T, C, H, W], but received X rank(%d)",
                                      x_dims.size()));

  int seg_num = op->attribute<pir::Int32Attribute>("seg_num").data();
  float shift_ratio = op->attribute<pir::FloatAttribute>("shift_ratio").data();

  PADDLE_ENFORCE_GT(
      seg_num,
      0,
      common::errors::InvalidArgument(
          "Attr(seg_num) should be greater than 0, but received %d", seg_num));
  PADDLE_ENFORCE_GT(
      shift_ratio,
      0.0f,
      common::errors::InvalidArgument(
          "Attr(shift_ratio) should be greater than 0, but received %f",
          shift_ratio));
  PADDLE_ENFORCE_LT(
      shift_ratio,
      0.5f,
      common::errors::InvalidArgument(
          "Attr(shift_ratio) should be less than 0.5, but received %f",
          shift_ratio));

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs(symbol::TensorShapeOrDataDimExprs(x_dims)));

  return true;
}

bool TileOpInferSymbolicShape(pir::Operation *op,
                              pir::InferSymbolicShapeContext *infer_context) {
  pir::Value operand_x = op->operand_source(0);
  symbol::ShapeOrDataDimExprs x_shape_or_data =
      infer_context->GetShapeOrDataForValue(operand_x);
  pir::Value operand_repeat_times = op->operand_source(1);
  symbol::ShapeOrDataDimExprs repeat_times_shape_or_data =
      infer_context->GetShapeOrDataForValue(operand_repeat_times);

  std::vector<symbol::DimExpr> x_dimexpr = x_shape_or_data.shape();
  std::vector<symbol::DimExpr> repeat_times_dimexpr =
      details::GetExprVecFromData(repeat_times_shape_or_data);
  if (repeat_times_dimexpr.empty()) {
    repeat_times_dimexpr = std::vector<symbol::DimExpr>(x_dimexpr.size(), 1);
  }

  auto out_rank = std::max(static_cast<size_t>(x_dimexpr.size()),
                           repeat_times_dimexpr.size());
  std::vector<symbol::DimExpr> out_shape(out_rank);
  if (x_dimexpr.size() > repeat_times_dimexpr.size()) {
    auto diff = x_dimexpr.size() - repeat_times_dimexpr.size();
    repeat_times_dimexpr.insert(repeat_times_dimexpr.begin(), diff, 1);
  } else {
    auto diff = repeat_times_dimexpr.size() - x_dimexpr.size();
    x_dimexpr.insert(x_dimexpr.begin(), diff, 1);
  }

  for (size_t i = 0; i < repeat_times_dimexpr.size(); ++i) {
    out_shape.at(i) = x_dimexpr.at(i) * repeat_times_dimexpr.at(i);
  }

  symbol::ShapeOrDataDimExprs shape_data{
      symbol::TensorShapeOrDataDimExprs(out_shape)};

  pir::Value res = op->result(0);
  infer_context->SetShapeOrDataForValue(res, shape_data);

  return true;
}

bool TopkOpInferSymbolicShape(pir::Operation *op,
                              pir::InferSymbolicShapeContext *infer_context) {
  symbol::ShapeOrDataDimExprs x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  symbol::ShapeOrDataDimExprs k_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));
  const auto &attributes = op->attributes();
  int axis = attributes.at("axis").dyn_cast<pir::Int32Attribute>().data();
  const std::vector<symbol::DimExpr> &in_dims_sym = [&] {
    std::vector<symbol::DimExpr> dims;
    if (x_shape_or_data.data().has_value()) {
      dims = x_shape_or_data.data().value();
    } else {
      dims = x_shape_or_data.shape();
    }
    return dims;
  }();

  int x_rank = in_dims_sym.size();

  symbol::DimExpr k = k_shape_or_data.data().value().at(0);

  if (axis < 0) axis += x_rank;
  const auto &out_sym_shape = [&] {
    std::vector<symbol::DimExpr> out_sym_shape;
    for (int i = 0; i < x_rank; ++i) {
      if (i == axis) {
        out_sym_shape.push_back(k);
      } else {
        out_sym_shape.push_back(in_dims_sym.at(i));
      }
    }
    return out_sym_shape;
  }();

  symbol::ShapeOrDataDimExprs shape_data{
      symbol::TensorShapeOrDataDimExprs(out_sym_shape)};

  infer_context->SetShapeOrDataForValue(op->result(0), shape_data);
  infer_context->SetShapeOrDataForValue(op->result(1), shape_data);

  return true;
}

bool TopkV1OpInferSymbolicShape(pir::Operation *op,
                                pir::InferSymbolicShapeContext *infer_context) {
  return TopkOpInferSymbolicShape(op, infer_context);
}

bool TraceOpInferSymbolicShape(pir::Operation *op,
                               pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const std::vector<symbol::DimExpr> &x_shape = x_shape_or_data.shape();
  int rank = x_shape.size();
  int axis1 = op->attribute<pir::Int32Attribute>("axis1").data();
  int axis2 = op->attribute<pir::Int32Attribute>("axis2").data();
  int dim1_ = axis1 < 0 ? rank + axis1 : axis1;
  int dim2_ = axis2 < 0 ? rank + axis2 : axis2;
  PADDLE_ENFORCE_GE(
      rank,
      2,
      common::errors::OutOfRange(
          "Input(x)'s dim is out of range (expected at least 2, but got %ld).",
          rank));
  PADDLE_ENFORCE_LT(
      dim1_,
      rank,
      common::errors::OutOfRange(
          "axis1 is out of range (expected to be in range of [%ld, "
          "%ld], but got %ld).",
          -(rank),
          (rank - 1),
          axis1));
  PADDLE_ENFORCE_GE(
      dim1_,
      0,
      common::errors::OutOfRange(
          "axis1 is out of range (expected to be in range of [%ld, "
          "%ld], but got %ld).",
          -(rank),
          (rank - 1),
          axis1));
  PADDLE_ENFORCE_LT(
      dim2_,
      rank,
      common::errors::OutOfRange(
          "axis2 is out of range (expected to be in range of [%ld, "
          "%ld], but got %ld).",
          -(rank),
          (rank - 1),
          axis2));
  PADDLE_ENFORCE_GE(
      dim2_,
      0,
      common::errors::OutOfRange(
          "axis2 is out of range (expected to be in range of [%ld, "
          "%ld], but got %ld).",
          -(rank),
          (rank - 1),
          axis2));
  PADDLE_ENFORCE_NE(
      dim1_,
      dim2_,
      common::errors::InvalidArgument("The dimensions should not be identical "
                                      "%ld vs %ld.",
                                      axis1,
                                      axis2));
  std::vector<symbol::DimExpr> x_dims = x_shape;
  if (x_shape.size() == 2) {
    x_dims.clear();
  } else {
    x_dims.erase(x_dims.begin() + std::max(dim1_, dim2_));
    x_dims.erase(x_dims.begin() + std::min(dim1_, dim2_));
  }
  infer_context->SetShapeOrDataForValue(
      op->result(0), symbol::TensorShapeOrDataDimExprs(x_dims));
  return true;
}

bool TransposeOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  std::vector<pir::Attribute> perm =
      op->attributes().at("perm").dyn_cast<pir::ArrayAttribute>().AsVector();
  if (perm.size() == 1) {
    // perm must be [0], which means nothing to do with input, just copy the
    // info from input
    infer_context->SetShapeOrDataForValue(
        op->result(0),
        infer_context->GetShapeOrDataForValue(op->operand_source(0)));
    return true;
  }
  const std::vector<symbol::DimExpr> &x_dims = [&] {
    std::vector<symbol::DimExpr> dims;
    const auto &x_shape_or_data =
        infer_context->GetShapeOrDataForValue(op->operand_source(0));
    dims = x_shape_or_data.shape();
    return dims;
  }();

  int x_rank = x_dims.size();

  const std::vector<int32_t> formatted_axis = [x_rank, &perm] {
    std::vector<int32_t> out(perm.size(), 0);
    std::transform(perm.begin(),
                   perm.end(),
                   out.begin(),
                   [](pir::Attribute &p) -> int32_t {
                     return p.dyn_cast<pir::Int32Attribute>().data();
                   });

    // format the negative axis
    std::for_each(out.begin(), out.end(), [x_rank](int32_t &v) {
      if (v < 0) {
        v += x_rank;
      }
    });
    return out;
  }();

  int axis_size = static_cast<int>(formatted_axis.size());

  std::vector<symbol::DimExpr> out_dims(x_dims);
  for (int i = 0; i < axis_size; ++i) {
    out_dims.at(i) = x_dims.at(formatted_axis.at(i));
  }

  infer_context->SetShapeOrDataForValue(op->result(0),
                                        ShapeOrData{TensorExprs(out_dims)});

  return true;
}

bool Transpose_OpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  return TransposeOpInferSymbolicShape(op, infer_context);
}

bool TransLayoutOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  return TransposeOpInferSymbolicShape(op, infer_context);
}
bool SqueezeOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  PADDLE_ENFORCE_EQ(
      op->num_operands(),
      2,
      common::errors::InvalidArgument(
          "SqueezeOpInferSymbolicShape ONLY support num_operands() == 2 "
          "now, but got %d operands",
          op->num_operands()));

  auto x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  auto axes_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));

  const auto &in_dims_sym = x_shape_or_data.shape();
  const std::vector<symbol::DimExpr> &squeeze_dims_sym =
      details::GetExprVecFromData(axes_shape_or_data);

  std::vector<int> squeeze_dims;
  for (auto squeeze_dim : squeeze_dims_sym) {
    PADDLE_ENFORCE_EQ(
        squeeze_dim.Has<std::int64_t>(),
        true,
        common::errors::InvalidArgument(
            "in SqueezeOpInferSymbolicShape, axes must be known int type, "
            "but got: %s",
            symbol::ToString(squeeze_dim)));
    squeeze_dims.emplace_back(
        static_cast<int>(squeeze_dim.Get<std::int64_t>()));
  }

  // GetOutputSqueezeShape
  size_t num_squeeze_dims = squeeze_dims.size();
  std::vector<bool> should_squeeze(in_dims_sym.size(), false);
  // Mark dimensions need to be squeezed.
  if (num_squeeze_dims == 0) {
    for (size_t i = 0; i < in_dims_sym.size(); ++i) {
      // TODO(lanxianghit): if symbol here, maybe we need the result of dim expr
      // simplification
      if (in_dims_sym.at(i) == 1) {
        should_squeeze.at(i) = true;
      }
    }
  } else {
    for (size_t i = 0; i < num_squeeze_dims; ++i) {
      if (in_dims_sym.size() == 0) {
        continue;
      }
      int current = squeeze_dims.at(i) < 0
                        ? squeeze_dims.at(i) + in_dims_sym.size()
                        : squeeze_dims.at(i);

      if (!should_squeeze.at(current)) {
        // At compile time, dim of SYMBOL is allowed to squeeze?
        if (!in_dims_sym.at(current).Has<std::int64_t>()) {
          should_squeeze[current] = true;
          continue;
        }
        if (in_dims_sym.at(current) == 1) {
          should_squeeze[current] = true;
          continue;
        }
        should_squeeze[current] = false;
      }
    }
  }

  // Make output dimensions
  std::vector<symbol::DimExpr> output_shape_sym;
  for (size_t i = 0; i < in_dims_sym.size(); ++i) {
    if (!should_squeeze.at(i)) {
      output_shape_sym.emplace_back(in_dims_sym.at(i));
    }
  }

  symbol::ShapeOrDataDimExprs shape_data{
      symbol::TensorShapeOrDataDimExprs(output_shape_sym)};

  pir::Value res = op->result(0);
  infer_context->SetShapeOrDataForValue(res, shape_data);

  return true;
}
bool Squeeze_OpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  return SqueezeOpInferSymbolicShape(op, infer_context);
}

bool UnbindOpInferSymbolicShape(pir::Operation *op,
                                pir::InferSymbolicShapeContext *infer_context) {
  // input
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  PADDLE_ENFORCE_EQ(
      x_shape_or_data.data().has_value(),
      false,
      common::errors::InvalidArgument(
          "InferSymbolicShape of UnbindOp only support input with "
          "value now."));
  const auto &x_dims_sym = x_shape_or_data.shape();

  // axis
  int axis = op->attributes().at("axis").dyn_cast<pir::Int32Attribute>().data();
  int rank = x_dims_sym.size();
  axis = axis >= 0 ? axis : axis + rank;

  // output
  const symbol::TensorListShapeOrDataDimExprs &output_shape_data_list = [&] {
    symbol::TensorListShapeOrDataDimExprs shape_data_list;
    std::vector<symbol::DimExpr> output_dims_sym = x_dims_sym;

    const symbol::DimExpr &unbound_dim = x_dims_sym.at(axis);
    PADDLE_ENFORCE_EQ(unbound_dim.isa<int64_t>(),
                      true,
                      common::errors::InvalidArgument(
                          "InferSymbolicShape of UnbindOp only support unbound "
                          "dim with constant length!"));
    output_dims_sym.erase(output_dims_sym.begin() + axis);
    const int64_t unbound_dim_length = unbound_dim.dyn_cast<int64_t>();

    for (uint32_t idx = 0; idx < unbound_dim_length; idx++) {
      shape_data_list.push_back(
          symbol::TensorShapeOrDataDimExprs(output_dims_sym));
    }
    return shape_data_list;
  }();

  infer_context->SetShapeOrDataForValue(
      op->result(0), symbol::ShapeOrDataDimExprs{output_shape_data_list});

  return true;
}

bool UniformInplaceOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const std::vector<symbol::DimExpr> &x_shape = x_shape_or_data.shape();
  float min = op->attribute<pir::FloatAttribute>("min").data();
  float max = op->attribute<pir::FloatAttribute>("max").data();
  int diag_num = op->attribute<pir::Int32Attribute>("diag_num").data();
  int diag_step = op->attribute<pir::Int32Attribute>("diag_step").data();

  PADDLE_ENFORCE_LT(
      min,
      max,
      common::errors::InvalidArgument(
          "The uniform_random's min must less then max. But received min = "
          "%f great than or equal max = %f.",
          min,
          max));
  PADDLE_ENFORCE_GE(diag_num,
                    0,
                    common::errors::InvalidArgument(
                        "The uniform_random's diag_num must greater than or "
                        "equal 0. But received diag_num (%d) < 0.",
                        diag_num));
  PADDLE_ENFORCE_GE(diag_step,
                    0,
                    common::errors::InvalidArgument(
                        "The uniform_random's diag_step must greater than or "
                        "equal 0. But received diag_step (%d) < 0.",
                        diag_step));
  PADDLE_ENFORCE_NE(op->result(0),
                    nullptr,
                    common::errors::InvalidArgument(
                        "uniform_random should have output tensor out."));

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(x_shape)});
  return true;
}

bool UniformInplace_OpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  return UniformInplaceOpInferSymbolicShape(op, infer_context);
}

bool UniformRandomBatchSizeLikeOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  return BatchSizeLikeInferSymbolicShape(op, infer_context);
}

bool UniformRandomBatchSizeLikeSrOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  return UniformRandomBatchSizeLikeOpInferSymbolicShape(op, infer_context);
}

bool UniqueOpInferSymbolicShape(pir::Operation *op,
                                pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  PADDLE_ENFORCE_EQ(
      x_shape_or_data.data().has_value(),
      false,
      common::errors::InvalidArgument(
          "InferSymbolicShape of UniqueOp only support input with "
          "value now."));
  const auto &x_dims_sym = x_shape_or_data.shape();
  const size_t rank = x_dims_sym.size();
  std::vector<int> axes =
      paddle::dialect::details::GetVectorAttr<int>(op, "axis");

  symbol::DimExpr unique_dim_sym =
      infer_context->GetNextSymName();  // unknown until runtime

  const std::vector<symbol::DimExpr> &counts_dims = [&] {
    std::vector<symbol::DimExpr> out_dims;
    out_dims.push_back(unique_dim_sym);
    return out_dims;
  }();

  const std::vector<symbol::DimExpr> &index_dims = counts_dims;

  const std::vector<symbol::DimExpr> &out_dims = [&] {
    if (axes.empty()) {
      return counts_dims;
    }
    std::vector<symbol::DimExpr> out_dims = x_dims_sym;
    int axis = axes.at(0);
    axis = axis >= 0 ? axis : axis + rank;
    out_dims.at(axis) = unique_dim_sym;
    return out_dims;
  }();

  const std::vector<symbol::DimExpr> &inverse_dims = [&] {
    std::vector<symbol::DimExpr> inverse_dims;
    if (axes.empty()) {
      // flatten before unique
      symbol::DimExpr product{1};
      for (const auto &x_dim : x_dims_sym) {
        product = product * x_dim;
      }
      inverse_dims.push_back(product);
    } else {
      int axis = axes.at(0);
      axis = axis >= 0 ? axis : axis + rank;
      inverse_dims.push_back(x_dims_sym.at(axis));
    }
    return inverse_dims;
  }();

  bool return_index = GetBoolAttr(op, "return_index");
  bool return_inverse = GetBoolAttr(op, "return_inverse");
  bool return_counts = GetBoolAttr(op, "return_counts");

  symbol::ShapeOrDataDimExprs empty{symbol::TensorShapeOrDataDimExprs{}};
  infer_context->SetShapeOrDataForValue(
      op->result(0), symbol::TensorShapeOrDataDimExprs{out_dims});
  infer_context->SetShapeOrDataForValue(
      op->result(1),
      return_index ? symbol::TensorShapeOrDataDimExprs{index_dims} : empty);
  infer_context->SetShapeOrDataForValue(
      op->result(2),
      return_inverse ? symbol::TensorShapeOrDataDimExprs{inverse_dims} : empty);
  infer_context->SetShapeOrDataForValue(
      op->result(3),
      return_counts ? symbol::TensorShapeOrDataDimExprs{counts_dims} : empty);

  return true;
}

bool UniqueConsecutiveOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  PADDLE_ENFORCE_EQ(
      x_shape_or_data.data().has_value(),
      false,
      common::errors::InvalidArgument(
          "InferSymbolicShape of UniqueConsecutiveOp only support input with "
          "value now."));
  const auto &x_dims_sym = x_shape_or_data.shape();
  const size_t rank = x_dims_sym.size();
  std::vector<int> axes =
      paddle::dialect::details::GetVectorAttr<int>(op, "axis");

  symbol::DimExpr unique_dim_sym =
      infer_context->GetNextSymName();  // unknown until runtime

  const std::vector<symbol::DimExpr> &counts_dims = [&] {
    std::vector<symbol::DimExpr> out_dims;
    out_dims.push_back(unique_dim_sym);
    return out_dims;
  }();

  const std::vector<symbol::DimExpr> &out_dims = [&] {
    if (axes.empty()) {
      return counts_dims;
    }
    std::vector<symbol::DimExpr> out_dims = x_dims_sym;
    int axis = axes.at(0);
    axis = axis >= 0 ? axis : axis + rank;
    out_dims.at(axis) = unique_dim_sym;
    return out_dims;
  }();

  const std::vector<symbol::DimExpr> &inverse_dims = [&] {
    std::vector<symbol::DimExpr> inverse_dims;
    if (axes.empty()) {
      // flatten before unique
      symbol::DimExpr product{1};
      for (const auto &x_dim : x_dims_sym) {
        product = product * x_dim;
      }
      inverse_dims.push_back(product);
    } else {
      int axis = axes.at(0);
      axis = axis >= 0 ? axis : axis + rank;
      inverse_dims.push_back(x_dims_sym.at(axis));
    }
    return inverse_dims;
  }();

  bool return_inverse = GetBoolAttr(op, "return_inverse");
  bool return_counts = GetBoolAttr(op, "return_counts");

  symbol::ShapeOrDataDimExprs empty{symbol::TensorShapeOrDataDimExprs{}};
  infer_context->SetShapeOrDataForValue(
      op->result(0), symbol::TensorShapeOrDataDimExprs{out_dims});
  infer_context->SetShapeOrDataForValue(
      op->result(1),
      return_inverse ? symbol::TensorShapeOrDataDimExprs{inverse_dims} : empty);
  infer_context->SetShapeOrDataForValue(
      op->result(2),
      return_counts ? symbol::TensorShapeOrDataDimExprs{counts_dims} : empty);

  return true;
}

bool UnsqueezeOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  PADDLE_ENFORCE_EQ(
      op->num_operands(),
      2,
      common::errors::InvalidArgument(
          "UnsqueezeOp InferSymbolicShape ONLY support num_operands() == 2 "
          "now, but got %d operands",
          op->num_operands()));

  auto x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  auto axis_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));

  const auto &x_sym_shape = x_shape_or_data.shape();
  int x_dims_size = x_sym_shape.size();

  std::vector<symbol::DimExpr> axis_sym;
  axis_sym =
      details::GetOrCreateExprVecFromData(axis_shape_or_data, infer_context);
  int axis_sym_size = axis_sym.size();

  // GetUnsqueezeShape
  int output_rank = x_dims_size + axis_sym_size;
  std::vector<symbol::DimExpr> result_sym_dims(output_rank, 0);

  int cur_output_rank = x_dims_size;
  bool is_new_sym = false;
  for (auto axis_expr : axis_sym) {
    if (axis_expr.Has<std::int64_t>()) {
      int axis = static_cast<int>(axis_expr.Get<std::int64_t>());
      int cur = axis < 0 ? axis + cur_output_rank + 1 : axis;

      // Move old axis, and insert new axis
      for (int i = cur_output_rank; i >= cur; --i) {
        if (result_sym_dims.at(i) == 1) {
          // Move axis
          result_sym_dims.at(i + 1) = 1;
          result_sym_dims.at(i) = 0;
        }
      }
      result_sym_dims.at(cur) = 1;
      // Add the output size.
      cur_output_rank++;
    } else {
      is_new_sym = true;
      break;
    }
  }

  // Make output shape
  if (is_new_sym) {
    for (int out_idx = 0; out_idx < output_rank; ++out_idx) {
      result_sym_dims.at(out_idx) = infer_context->GetNextSymName();
    }
  } else {
    for (int in_idx = 0, out_idx = 0; out_idx < output_rank; ++out_idx) {
      if (result_sym_dims.at(out_idx) == 0) {
        result_sym_dims.at(out_idx) = x_sym_shape.at(in_idx++);
      }
    }
  }

  symbol::ShapeOrDataDimExprs shape_data{
      symbol::TensorShapeOrDataDimExprs(result_sym_dims)};

  pir::Value res = op->result(0);
  infer_context->SetShapeOrDataForValue(res, shape_data);

  return true;
}
bool Unsqueeze_OpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  return UnsqueezeOpInferSymbolicShape(op, infer_context);
}

bool UnfoldOpInferSymbolicShape(pir::Operation *op,
                                pir::InferSymbolicShapeContext *infer_context) {
  const symbol::ShapeOrDataDimExprs &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));

  const auto &x_shape = x_shape_or_data.shape();

  PADDLE_ENFORCE_EQ(
      x_shape.size(),
      4UL,
      common::errors::InvalidArgument(
          "Input should be 4-D tensor of format [N, C, H, W], but get %u",
          x_shape.size()));

  const std::vector<int> &kernel_sizes =
      paddle::dialect::details::GetVectorAttr<int>(op, "kernel_sizes");
  const std::vector<int> &strides =
      paddle::dialect::details::GetVectorAttr<int>(op, "strides");
  const std::vector<int> &paddings =
      paddle::dialect::details::GetVectorAttr<int>(op, "paddings");
  const std::vector<int> &dilations =
      paddle::dialect::details::GetVectorAttr<int>(op, "dilations");

  PADDLE_ENFORCE_EQ(
      x_shape.size() - kernel_sizes.size(),
      2UL,
      common::errors::InvalidArgument(
          "The dims of X should be larger than that of kernel_sizes "
          "by a number of 2, due to the batch size and input channel dim. "
          "But received dims(X:%u) - dims(kernel_sizes:%u) != 2",
          x_shape.size(),
          kernel_sizes.size()));
  PADDLE_ENFORCE_EQ(
      strides.size(),
      kernel_sizes.size(),
      common::errors::InvalidArgument(
          "The dims of strides should be the same with that of kernel_sizes. "
          "But received dims(strides: %u) != dims(kernel_sizes: %u).",
          strides.size(),
          kernel_sizes.size()));
  PADDLE_ENFORCE_EQ(
      paddings.size(),
      2 * strides.size(),
      common::errors::InvalidArgument(
          "The dims of paddings should be 2 times of that of strides. "
          "But received dims(paddings: %u) != 2*dims(strides: %u).",
          paddings.size(),
          strides.size()));
  PADDLE_ENFORCE_EQ(
      strides.size(),
      dilations.size(),
      common::errors::InvalidArgument(
          "The dims of strides should be the same with that of dilations. "
          "But received dims(strides: %u) != dims(dilations: %u).",
          strides.size(),
          dilations.size()));

  std::vector<symbol::DimExpr> out_shapes;

  out_shapes.push_back(x_shape[0]);
  out_shapes.push_back(x_shape[1] *
                       symbol::DimExpr(kernel_sizes[0] * kernel_sizes[1]));

  const auto &calculate_output_dim = [&](symbol::DimExpr input_size,
                                         int padding1,
                                         int padding2,
                                         int dilation,
                                         int kernel_size,
                                         int stride) {
    const symbol::DimExpr dkernel_size =
        symbol::DimExpr(dilation * (kernel_size - 1) + 1);
    return (input_size + symbol::DimExpr(padding1 + padding2) - dkernel_size) /
               symbol::DimExpr(stride) +
           symbol::DimExpr(1);
  };

  symbol::DimExpr output_height = calculate_output_dim(x_shape[2],
                                                       paddings[0],
                                                       paddings[2],
                                                       dilations[0],
                                                       kernel_sizes[0],
                                                       strides[0]);
  symbol::DimExpr output_width = calculate_output_dim(x_shape[3],
                                                      paddings[1],
                                                      paddings[3],
                                                      dilations[1],
                                                      kernel_sizes[1],
                                                      strides[1]);

  symbol::DimExpr output_col_length = output_height * output_width;
  out_shapes.push_back(output_col_length);

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(out_shapes)});

  return true;
}

bool UnstackOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const std::vector<symbol::DimExpr> &x_shape = x_shape_or_data.shape();
  int rank = x_shape.size();

  int axis = op->attribute<pir::Int32Attribute>("axis").data();
  int num = op->attribute<pir::Int32Attribute>("num").data();

  PADDLE_ENFORCE_GE(axis,
                    -rank,
                    common::errors::InvalidArgument(
                        "The attribute axis is out of range, it must be inside "
                        "[-rank, rank), where rank = %d",
                        rank));
  PADDLE_ENFORCE_LT(axis,
                    rank,
                    common::errors::InvalidArgument(
                        "The attribute axis is out of range, it must be inside "
                        "[-rank, rank), where rank = %d",
                        rank));
  if (axis < 0) axis += rank;

  infer_context->AddEqualCstr(x_shape[axis], num);

  symbol::TensorListShapeOrDataDimExprs out_list_shape_or_data;

  std::vector<symbol::DimExpr> out_shape = x_shape;
  out_shape.erase(out_shape.begin() + axis);

  symbol::TensorShapeOrDataDimExprs out_shape_or_data =
      symbol::TensorShapeOrDataDimExprs(out_shape);
  for (int i = 0; i < num; i++) {
    out_list_shape_or_data.push_back(out_shape_or_data);
  }
  infer_context->SetShapeOrDataForValue(op->result(0), out_list_shape_or_data);
  return true;
}

bool WeightQuantizeOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(0)).shape();
  PADDLE_ENFORCE_EQ(
      x_shape.size(),
      2UL,
      common::errors::InvalidArgument(
          "The x tensor of quant op must be 2D, but got[%d]", x_shape.size()));
  if (x_shape[0].isa<int64_t>()) {
    int64_t x_shape_0 = x_shape[0].dyn_cast<int64_t>();
    PADDLE_ENFORCE_EQ(
        x_shape_0 % 64,
        0,
        common::errors::InvalidArgument(
            "The first dimension of input must be divisible by 64, but got[%d]",
            x_shape_0));
  }
  if (x_shape[1].isa<int64_t>()) {
    int64_t x_shape_1 = x_shape[1].dyn_cast<int64_t>();
    PADDLE_ENFORCE_EQ(
        x_shape_1 % 16,
        0,
        common::errors::InvalidArgument("The second dimension of input must be "
                                        "divisible by 16, but got[%d]",
                                        x_shape_1));
  }

  const int group_size =
      op->attribute<pir::Int32Attribute>("group_size").data();
  const std::string algo = op->attribute<pir::StrAttribute>("algo").AsString();
  PADDLE_ENFORCE_EQ(
      ((group_size == -1) || (group_size == 64) || (group_size == 128)),
      true,
      common::errors::InvalidArgument(
          "Currently, group_size only support -1, 64 or 128."));
  std::vector<symbol::DimExpr> scale_shape;
  std::vector<symbol::DimExpr> out_shape;
  if (group_size != -1) {
    symbol::DimExpr scale_shape_0 =
        (x_shape[0] + (group_size - 1)) / group_size;
    scale_shape = {scale_shape_0, x_shape[1]};
  } else {
    scale_shape = {x_shape[1]};
  }
  if (algo == "weight_only_int8" || algo == "llm.int8") {
    out_shape = {x_shape[1], x_shape[0]};
  } else if (algo == "weight_only_int4") {
    out_shape = {x_shape[1] / 2, x_shape[0]};
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "The algo must be in ['weight_only_int8', 'weight_only_int4', "
        "'llm.int8'], but got[%s]",
        algo));
  }
  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(out_shape)});

  infer_context->SetShapeOrDataForValue(
      op->result(1),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(scale_shape)});
  return true;
}

}  // namespace paddle::dialect
