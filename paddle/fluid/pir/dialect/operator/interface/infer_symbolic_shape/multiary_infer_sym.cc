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

#include "paddle/fluid/pir/dialect/operator/interface/infer_symbolic_shape/multiary_infer_sym.h"
#include "paddle/common/ddim.h"
#include "paddle/common/layout.h"
#include "paddle/fluid/pir/dialect/operator/interface/infer_symbolic_shape/infer_sym_slice_utils.h"
#include "paddle/fluid/pir/dialect/operator/interface/infer_symbolic_shape/infer_sym_utils.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"

namespace paddle::dialect {

bool AccuracyOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const symbol::ShapeOrDataDimExprs &out_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const symbol::ShapeOrDataDimExprs &label_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(2));

  // Assume indices has same shape as inference, because
  // it's the output of topk.
  PADDLE_ENFORCE_EQ(
      label_shape.shape().size(),
      2UL,
      common::errors::InvalidArgument(
          "ShapeError: label's dimensions of AccuracyOp must be 2. "
          "But received label's dimensions = %d",
          label_shape.shape().size()));

  infer_context->AddEqualCstr(label_shape.shape()[1], symbol::DimExpr{1});
  infer_context->AddEqualCstr(out_shape.shape()[0], label_shape.shape()[0]);

  std::vector<symbol::DimExpr> accuracy_shape = {};
  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(accuracy_shape)});

  std::vector<symbol::DimExpr> correct_shape = {};
  infer_context->SetShapeOrDataForValue(
      op->result(1),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(correct_shape)});

  std::vector<symbol::DimExpr> total_shape = {};
  infer_context->SetShapeOrDataForValue(
      op->result(2),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(total_shape)});

  return true;
}

bool AddNOpInferSymbolicShape(pir::Operation *op,
                              pir::InferSymbolicShapeContext *infer_context) {
  const auto &input_list_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  PADDLE_ENFORCE_EQ(
      input_list_shape_or_data.isa<symbol::TensorListShapeOrDataDimExprs>(),
      true,
      common::errors::InvalidArgument(
          "The type of inputs shape should be TensorListShapeOrDataDimExprs"));
  const auto &inputs_shape =
      input_list_shape_or_data
          .dyn_cast<symbol::TensorListShapeOrDataDimExprs>();
  PADDLE_ENFORCE_GT(
      inputs_shape.size(),
      0,
      common::errors::InvalidArgument(
          "The input tensor X's dimensions of AddNOp "
          "should be larger than 0. But received X's dimensions %d.",
          inputs_shape.size()));
  symbol::TensorShapeOrDataDimExprs candidate_shape = inputs_shape.front();
  for (size_t i = 1; i < inputs_shape.size(); ++i) {
    // 0D tensor
    if (inputs_shape[i].shape().size() == 0) {
      continue;
    }
    if (candidate_shape.shape().size() == 0) {
      candidate_shape = inputs_shape[i];
      continue;
    }
    for (size_t j = 0; j < candidate_shape.shape().size(); ++j) {
      infer_context->AddEqualCstr(candidate_shape.shape()[j],
                                  inputs_shape[i].shape()[j]);
    }
  }
  infer_context->SetShapeOrDataForValue(
      op->result(0), symbol::ShapeOrDataDimExprs{candidate_shape});

  return true;
}

bool AddmmOpInferSymbolicShape(pir::Operation *op,
                               pir::InferSymbolicShapeContext *infer_context) {
  const auto &input_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const auto &x_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));
  const auto &y_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(2));

  auto ndim_input = input_shape.shape().size();
  auto ndim_x = x_shape.shape().size();
  auto ndim_y = y_shape.shape().size();

  PADDLE_ENFORCE_EQ(ndim_input == 2 || ndim_input == 1,
                    true,
                    common::errors::InvalidArgument(
                        "The input tensor input's dimension must be 2 or 1. "
                        "But received input's dimension = [%d].",
                        ndim_input));
  PADDLE_ENFORCE_EQ(ndim_x,
                    2,
                    common::errors::InvalidArgument(
                        "The input tensor x's dimension must be 2. "
                        "But received x's dimension = [%d].",
                        ndim_x));
  PADDLE_ENFORCE_EQ(ndim_y,
                    2,
                    common::errors::InvalidArgument(
                        "The input tensor y's dimension must be 2. "
                        "But received y's dimension = [%d].",
                        ndim_y));

  std::vector<symbol::DimExpr> output_shape;
  output_shape.push_back(x_shape.shape()[0]);
  output_shape.push_back(y_shape.shape()[1]);

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(output_shape)});

  infer_context->AddEqualCstr(x_shape.shape()[1], y_shape.shape()[0]);

  if (ndim_input == 2) {
    infer_context->AddBroadcastableCstr(input_shape.shape()[0],
                                        x_shape.shape()[0]);
    infer_context->AddBroadcastableCstr(input_shape.shape()[1],
                                        y_shape.shape()[1]);
  } else if (ndim_input == 1) {
    infer_context->AddBroadcastableCstr(input_shape.shape()[0],
                                        y_shape.shape()[1]);
  }

  return true;
}

bool Addmm_OpInferSymbolicShape(pir::Operation *op,
                                pir::InferSymbolicShapeContext *infer_context) {
  return AddmmOpInferSymbolicShape(op, infer_context);
}

bool AucOpInferSymbolicShape(pir::Operation *op,
                             pir::InferSymbolicShapeContext *infer_context) {
  const auto &predict_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const auto &label_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));

  PADDLE_ENFORCE_GE(
      predict_shape.shape().size(),
      2,
      common::errors::InvalidArgument(
          "The Input(Predict) has not been initialized properly. The "
          "shape of Input(Predict) = [%s], the shape size must be "
          "greater_equal 2.",
          predict_shape.shape()));

  const auto &predict_height = predict_shape.shape()[0];
  const auto &label_height = label_shape.shape()[0];

  infer_context->AddEqualCstr(predict_height, label_height);

  int num_thresholds =
      op->attribute<pir::Int32Attribute>("num_thresholds").data();
  int slide_steps = op->attribute<pir::Int32Attribute>("slide_steps").data();

  int num_pred_buckets = num_thresholds + 1;

  PADDLE_ENFORCE_GE(
      num_pred_buckets,
      1,
      common::errors::InvalidArgument("num_thresholds must larger than 1"));
  PADDLE_ENFORCE_GE(
      slide_steps,
      0,
      common::errors::InvalidArgument("slide_steps must be natural number"));

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(std::vector<symbol::DimExpr>{})});

  if (slide_steps) {
    infer_context->SetShapeOrDataForValue(
        op->result(1),
        symbol::ShapeOrDataDimExprs{
            symbol::TensorShapeOrDataDimExprs(std::vector<symbol::DimExpr>{
                (1 + slide_steps) * num_pred_buckets + 1})});
    infer_context->SetShapeOrDataForValue(
        op->result(2),
        symbol::ShapeOrDataDimExprs{
            symbol::TensorShapeOrDataDimExprs(std::vector<symbol::DimExpr>{
                (1 + slide_steps) * num_pred_buckets + 1})});
  } else {
    infer_context->SetShapeOrDataForValue(
        op->result(1),
        symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(
            std::vector<symbol::DimExpr>{1, num_pred_buckets})});
    infer_context->SetShapeOrDataForValue(
        op->result(2),
        symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(
            std::vector<symbol::DimExpr>{1, num_pred_buckets})});
  }

  return true;
}

bool BatchFcOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &input_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const auto &w_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));
  const auto &bias_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(2));

  const std::vector<symbol::DimExpr> &input_dims = input_shape_or_data.shape();
  const std::vector<symbol::DimExpr> &w_dims = w_shape_or_data.shape();
  const std::vector<symbol::DimExpr> &bias_dims = bias_shape_or_data.shape();

  PADDLE_ENFORCE_EQ(
      input_dims.size(),
      3,
      common::errors::InvalidArgument("Input of BatchFcOp should have 3D."));
  PADDLE_ENFORCE_EQ(
      w_dims.size(),
      3,
      common::errors::InvalidArgument("W of BatchFcOp should have 3D."));
  infer_context->AddEqualCstr(input_dims[0], w_dims[0]);
  infer_context->AddEqualCstr(input_dims[2], w_dims[1]);
  infer_context->AddEqualCstr(bias_dims[0], input_dims[0]);
  infer_context->AddEqualCstr(bias_dims[1], w_dims[2]);

  std::vector<symbol::DimExpr> out_dims = {
      input_dims[0], input_dims[1], w_dims[2]};

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(out_dims)});

  return true;
}

bool BatchNormOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const auto &scale_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(3));
  const auto &bias_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(4));

  std::vector<symbol::DimExpr> x_dims = x_shape_or_data.shape();

  std::string data_layout_str =
      op->attribute<pir::StrAttribute>("data_format").AsString();
  const DataLayout data_layout = common::StringToDataLayout(data_layout_str);

  PADDLE_ENFORCE_GE(
      x_dims.size(),
      2,
      common::errors::InvalidArgument(
          "ShapeError: the dimension of input "
          "X must greater than or equal to 2. But received: the shape of input "
          "X = [%s], the dimension of input X =[%d]",
          x_dims,
          x_dims.size()));
  PADDLE_ENFORCE_LE(
      x_dims.size(),
      5,
      common::errors::InvalidArgument(
          "ShapeError: the dimension of input X "
          "must smaller than or equal to 5. But received: the shape of input X "
          "= [%s], the dimension of input X = [%d]",
          x_dims,
          x_dims.size()));

  symbol::DimExpr C = (data_layout == DataLayout::kNCHW)
                          ? x_dims[1]
                          : x_dims[x_dims.size() - 1];

  if (!scale_shape_or_data.isa<symbol::NullShapeOrDataDimExpr>()) {
    std::vector<symbol::DimExpr> scale_dims = scale_shape_or_data.shape();
    PADDLE_ENFORCE_EQ(scale_dims.size(),
                      1UL,
                      common::errors::InvalidArgument(
                          "ShapeError: the dimension of scale must equal to 1."
                          "But received: the dimension of scale is [%d]",
                          scale_dims.size()));
    infer_context->AddEqualCstr(scale_dims[0], C);
  }

  if (!bias_shape_or_data.isa<symbol::NullShapeOrDataDimExpr>()) {
    std::vector<symbol::DimExpr> bias_dims = bias_shape_or_data.shape();
    PADDLE_ENFORCE_EQ(bias_dims.size(),
                      1UL,
                      common::errors::InvalidArgument(
                          "ShapeError: the dimension of bias must equal to 1."
                          "But received: the dimension of bias is [%d]",
                          bias_dims.size()));
    infer_context->AddEqualCstr(bias_dims[0], C);
  }

  // Set output shapes
  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(x_dims)});

  std::vector<symbol::DimExpr> param_dims = {C};
  infer_context->SetShapeOrDataForValue(
      op->result(1),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(param_dims)});
  infer_context->SetShapeOrDataForValue(
      op->result(2),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(param_dims)});

  if (op->result(3) && op->result(3).type()) {
    infer_context->SetShapeOrDataForValue(
        op->result(3),
        symbol::ShapeOrDataDimExprs{
            symbol::TensorShapeOrDataDimExprs(param_dims)});
  }
  if (op->result(4) && op->result(4).type()) {
    infer_context->SetShapeOrDataForValue(
        op->result(4),
        symbol::ShapeOrDataDimExprs{
            symbol::TensorShapeOrDataDimExprs(param_dims)});
  }
  if (op->result(5) && op->result(5).type()) {
    std::vector<symbol::DimExpr> reserve_space_dims{
        symbol::DimExpr{infer_context->GetNextSymName()}};
    infer_context->SetShapeOrDataForValue(
        op->result(5),
        symbol::ShapeOrDataDimExprs{
            symbol::TensorShapeOrDataDimExprs(reserve_space_dims)});
  }

  return true;
}

bool BatchNorm_OpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  return BatchNormOpInferSymbolicShape(op, infer_context);
}

bool BicubicInterpOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const symbol::ShapeOrDataDimExprs &x =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));

  const auto &attributes = op->attributes();

  const std::string data_format =
      attributes.at("data_format").dyn_cast<pir::StrAttribute>().AsString();
  int out_d = attributes.at("out_d").dyn_cast<pir::Int32Attribute>().data();
  int out_h = attributes.at("out_h").dyn_cast<pir::Int32Attribute>().data();
  int out_w = attributes.at("out_w").dyn_cast<pir::Int32Attribute>().data();
  const std::vector<float> &scale = details::GetVectorAttr<float>(op, "scale");

  const bool has_size_tensor = [&] {
    pir::Value size_tensor = op->operand_source(2);
    if (!size_tensor || !size_tensor.type()) {
      return false;
    }
    const auto &list_size_tensor =
        size_tensor.type().dyn_cast<pir::VectorType>();
    return list_size_tensor && !list_size_tensor.empty();
  }();
  auto GetSizeTensorDataExpr =
      [&](pir::Value value) -> std::vector<symbol::DimExpr> {
    const symbol::ShapeOrDataDimExprs &size_tensor_shape =
        infer_context->GetShapeOrDataForValue(value);
    PADDLE_ENFORCE_EQ(
        size_tensor_shape.isa<symbol::TensorListShapeOrDataDimExprs>(),
        true,
        common::errors::InvalidArgument(
            "The size_tensor of Interpolation should be type of "
            "TensorListShapeOrDataDimExprs"));
    return details::GetOrCreateExprVecFromData(size_tensor_shape,
                                               infer_context);
  };
  auto GetOutSizeDataExpr =
      [&](pir::Value value) -> std::vector<symbol::DimExpr> {
    const symbol::ShapeOrDataDimExprs &out_size_tensor_shape =
        infer_context->GetShapeOrDataForValue(value);
    return details::GetOrCreateExprVecFromData(out_size_tensor_shape,
                                               infer_context);
  };
  auto GetOutDimByScale = [&](const symbol::DimExpr &in_dim,
                              float scale) -> symbol::DimExpr {
    PADDLE_ENFORCE_GT(scale,
                      0,
                      common::errors::InvalidArgument(
                          "The scale in Attr(scale) of Operator(interpolate) "
                          "should be greater than 0, but received value is %d.",
                          scale));
    if (in_dim.isa<int64_t>()) {
      return symbol::DimExpr{
          static_cast<int64_t>(in_dim.dyn_cast<int64_t>() * scale)};
    }
    return symbol::DimExpr{infer_context->GetNextSymName()};
  };

  std::vector<int> size_tensor;
  if (out_d != -1) size_tensor.push_back(out_d);
  if (out_h != -1) size_tensor.push_back(out_h);
  if (out_w != -1) size_tensor.push_back(out_w);

  const DataLayout data_layout = common::StringToDataLayout(data_format);

  if (x.shape().size() == 3) {
    // shape check for 1D interpolate for input tensor shape NCHW
    if (!size_tensor.empty()) {
      // top priority size
      std::vector<symbol::DimExpr> dim_out;
      if (data_layout == DataLayout::kNCHW) {
        dim_out = {x.shape()[0], x.shape()[1], symbol::DimExpr{out_w}};
      } else {
        dim_out = {x.shape()[0], symbol::DimExpr{out_w}, x.shape()[2]};
      }

      symbol::ShapeOrDataDimExprs shape_data{
          symbol::TensorShapeOrDataDimExprs(dim_out)};

      pir::Value res = op->result(0);
      infer_context->SetShapeOrDataForValue(res, shape_data);
      return true;
    }

    symbol::DimExpr out_w_tmp{0};
    const auto &next_sym = infer_context->GetNextSymName();
    out_w_tmp = symbol::DimExpr(next_sym);

    std::vector<symbol::DimExpr> dim_out;
    if (data_layout == DataLayout::kNCHW) {
      dim_out = {x.shape()[0], x.shape()[1], out_w_tmp};
    } else {
      dim_out = {x.shape()[0], out_w_tmp, x.shape()[2]};
    }

    symbol::ShapeOrDataDimExprs shape_data{
        symbol::TensorShapeOrDataDimExprs(dim_out)};

    pir::Value res = op->result(0);
    infer_context->SetShapeOrDataForValue(res, shape_data);
    return true;
  } else if (x.shape().size() == 4) {
    // shape check for 2D interpolate for input tensor shape NCHW
    auto GetOutHW = [&]() -> std::tuple<symbol::DimExpr, symbol::DimExpr> {
      // top priority size
      if (has_size_tensor) {
        const auto &size_tensor_list_shape =
            GetSizeTensorDataExpr(op->operand_source(2));
        PADDLE_ENFORCE_EQ(size_tensor_list_shape.size(),
                          2,
                          common::errors::InvalidArgument(
                              "The size of size_tensor list should be 2."));
        return std::make_tuple(size_tensor_list_shape.at(0),
                               size_tensor_list_shape.at(1));
      }
      // has out_size tensor
      if (op->operand_source(1)) {
        const auto &out_size_shape_or_data =
            infer_context->GetShapeOrDataForValue(op->operand_source(1));
        PADDLE_ENFORCE_EQ(
            out_size_shape_or_data.shape().size(),
            1,
            common::errors::InvalidArgument(
                "The rank of input out_size tensor should be 1."));
        infer_context->AddEqualCstr(out_size_shape_or_data.shape()[0],
                                    symbol::DimExpr{2});
        const auto &out_size_data = GetOutSizeDataExpr(op->operand_source(1));
        return std::make_tuple(symbol::DimExpr{out_size_data[0]},
                               symbol::DimExpr{out_size_data[1]});
      }
      // has scale
      if (scale.size() == 2) {
        float scale_h = scale[0];
        float scale_w = scale[1];
        const auto &in_h =
            data_layout == DataLayout::kNCHW ? x.shape()[2] : x.shape()[1];
        const auto &in_w =
            data_layout == DataLayout::kNCHW ? x.shape()[3] : x.shape()[2];
        return std::make_tuple(GetOutDimByScale(in_h, scale_h),
                               GetOutDimByScale(in_w, scale_w));
      }

      return std::make_tuple(symbol::DimExpr{out_h}, symbol::DimExpr{out_w});
    };

    const std::vector<symbol::DimExpr> dim_out = [&] {
      const auto &[out_h_sym, out_w_sym] = GetOutHW();
      if (data_layout == DataLayout::kNCHW) {
        return std::vector<symbol::DimExpr>{
            x.shape()[0], x.shape()[1], out_h_sym, out_w_sym};
      } else {
        return std::vector<symbol::DimExpr>{
            x.shape()[0], out_h_sym, out_w_sym, x.shape()[3]};
      }
    }();

    symbol::ShapeOrDataDimExprs shape_data{
        symbol::TensorShapeOrDataDimExprs(dim_out)};
    infer_context->SetShapeOrDataForValue(op->result(0), shape_data);

    return true;
  } else if (x.shape().size() == 5) {
    auto GetOutDHW =
        [&]() -> std::tuple<symbol::DimExpr, symbol::DimExpr, symbol::DimExpr> {
      // top priority size
      if (has_size_tensor) {
        const auto &size_tensor_list_shape =
            GetSizeTensorDataExpr(op->operand_source(2));
        PADDLE_ENFORCE_EQ(size_tensor_list_shape.size(),
                          3,
                          common::errors::InvalidArgument(
                              "The size of size_tensor list should be 3."));
        return std::make_tuple(size_tensor_list_shape.at(0),
                               size_tensor_list_shape.at(1),
                               size_tensor_list_shape.at(2));
      }
      // has out_size tensor
      if (op->operand_source(1)) {
        const auto &out_size_data = GetOutSizeDataExpr(op->operand_source(1));
        return std::make_tuple(symbol::DimExpr{out_size_data[0]},
                               symbol::DimExpr{out_size_data[1]},
                               symbol::DimExpr{out_size_data[2]});
      }
      // has scale
      if (scale.size() == 3) {
        float scale_d = scale[0];
        float scale_h = scale[1];
        float scale_w = scale[2];
        const auto &in_d =
            data_layout == DataLayout::kNCHW ? x.shape()[2] : x.shape()[1];
        const auto &in_h =
            data_layout == DataLayout::kNCHW ? x.shape()[3] : x.shape()[2];
        const auto &in_w =
            data_layout == DataLayout::kNCHW ? x.shape()[4] : x.shape()[3];
        return std::make_tuple(GetOutDimByScale(in_d, scale_d),
                               GetOutDimByScale(in_h, scale_h),
                               GetOutDimByScale(in_w, scale_w));
      }

      return std::make_tuple(symbol::DimExpr{out_d},
                             symbol::DimExpr{out_h},
                             symbol::DimExpr{out_w});
    };

    const std::vector<symbol::DimExpr> dim_out = [&] {
      const auto &[out_d_sym, out_h_sym, out_w_sym] = GetOutDHW();
      if (data_layout == DataLayout::kNCHW) {
        return std::vector<symbol::DimExpr>{
            x.shape()[0], x.shape()[1], out_d_sym, out_h_sym, out_w_sym};
      } else {
        return std::vector<symbol::DimExpr>{
            x.shape()[0], out_d_sym, out_h_sym, out_w_sym, x.shape()[4]};
      }
    }();

    symbol::ShapeOrDataDimExprs shape_data{
        symbol::TensorShapeOrDataDimExprs(dim_out)};
    infer_context->SetShapeOrDataForValue(op->result(0), shape_data);
    return true;
  } else {
    PADDLE_THROW(
        common::errors::Fatal("Input(X) dimension must be 3, 4 or 5!"));
  }

  return true;
}

bool BilinearOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const auto &y_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));
  const auto &weight_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(2));

  PADDLE_ENFORCE_EQ(
      x_shape.shape().size(),
      2UL,
      common::errors::InvalidArgument("The input(X) must be a 2D Tensor."));
  PADDLE_ENFORCE_EQ(
      y_shape.shape().size(),
      2UL,
      common::errors::InvalidArgument("The input(Y) must be a 2D Tensor."));
  PADDLE_ENFORCE_EQ(
      weight_shape.shape().size(),
      3UL,
      common::errors::InvalidArgument(
          "Expected the input(Weight) is a 3D tensor. But received %dD tensor.",
          weight_shape.shape().size()));

  infer_context->AddEqualCstr(x_shape.shape()[0], y_shape.shape()[0]);

  infer_context->AddEqualCstr(x_shape.shape()[1], weight_shape.shape()[1]);
  infer_context->AddEqualCstr(y_shape.shape()[1], weight_shape.shape()[2]);

  if (op->operand_source(3)) {  // has bias
    const auto &bias_shape =
        infer_context->GetShapeOrDataForValue(op->operand_source(3));
    PADDLE_ENFORCE_EQ(bias_shape.shape().size(),
                      2UL,
                      common::errors::InvalidArgument(
                          "The Input(Bias) must be a 2-D tensor with "
                          "the 2nd dimension fixed to 1 (a row vector)."));
    infer_context->AddEqualCstr(bias_shape.shape()[0], symbol::DimExpr{1});
    infer_context->AddEqualCstr(bias_shape.shape()[1], weight_shape.shape()[0]);
  }

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(
          {x_shape.shape()[0], weight_shape.shape()[0]})});

  return true;
}

bool AssignPosOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &eff_num_len_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(2));
  if (eff_num_len_shape_or_data.data()
          .has_value()) {  // according to the kernel code
    infer_context->SetShapeOrDataForValue(
        op->result(0),
        symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(
            {eff_num_len_shape_or_data.data()->at(0)})});
  } else {
    infer_context->SetShapeOrDataForValue(
        op->result(0),
        symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(
            {infer_context->GetNextSymName()})});
  }
  return true;
}

bool BroadcastTensorsOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &input_shape_or_data_list =
      infer_context->GetShapeOrDataForValue(op->operand_source(0))
          .dyn_cast<symbol::TensorListShapeOrDataDimExprs>();
  // 1. Find Output rank = max(Inputs rank)
  int target_rank = 0;
  for (const auto &input_shape_or_data : input_shape_or_data_list) {
    int tmp_rank = input_shape_or_data.shape().size();
    target_rank = std::max(target_rank, tmp_rank);
  }
  // 2. Output dim(axis=x) = max(Inputs dim(axis=x))
  std::vector<symbol::DimExpr> out_shape;
  symbol::DimExprBuilder builder;
  for (int i = 0; i < target_rank; i++) {
    auto tmp_dim = symbol::DimExpr{1};
    for (const auto &input_shape_or_data : input_shape_or_data_list) {
      int axis = input_shape_or_data.shape().size();
      axis = i - target_rank + axis;
      if (axis >= 0) {
        infer_context->AddBroadcastableCstr(input_shape_or_data.shape()[axis],
                                            tmp_dim);
        tmp_dim = builder.Broadcast(input_shape_or_data.shape()[axis], tmp_dim);
      }
    }
    out_shape.emplace_back(tmp_dim);
  }
  symbol::TensorListShapeOrDataDimExprs out_shapes;
  for (size_t i = 0; i < input_shape_or_data_list.size(); i++) {
    out_shapes.emplace_back(out_shape);
  }
  infer_context->SetShapeOrDataForValue(
      op->result(0), symbol::ShapeOrDataDimExprs{out_shapes});
  return true;
}

bool BilinearInterpOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  return BicubicInterpOpInferSymbolicShape(op, infer_context);
}

bool BoxCoderOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const symbol::ShapeOrDataDimExprs &prior_box_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const std::vector<symbol::DimExpr> &prior_box_shape =
      prior_box_shape_or_data.shape();

  const symbol::ShapeOrDataDimExprs &target_box_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(2));
  const std::vector<symbol::DimExpr> &target_box_shape =
      target_box_shape_or_data.shape();

  const std::string &code_type =
      op->attribute<pir::StrAttribute>("code_type").AsString();
  int axis = op->attribute<pir::Int32Attribute>("axis").data();

  PADDLE_ENFORCE_EQ(prior_box_shape.size(),
                    2,
                    common::errors::InvalidArgument(
                        "The rank of Input PriorBox in BoxCoder operator "
                        "must be 2. But received rank = %d",
                        prior_box_shape.size()));
  infer_context->AddEqualCstr(prior_box_shape[1], symbol::DimExpr{4});

  if (!paddle::dialect::details::IsFakeValue(op->operand_source(1))) {
    const symbol::ShapeOrDataDimExprs &prior_box_var_shape_or_data =
        infer_context->GetShapeOrDataForValue(op->operand_source(1));
    const std::vector<symbol::DimExpr> &prior_box_var_shape =
        prior_box_var_shape_or_data.shape();

    PADDLE_ENFORCE_EQ(prior_box_var_shape.size(),
                      2,
                      common::errors::InvalidArgument(
                          "The rank of Input(PriorBoxVar) in BoxCoder operator "
                          "should be 2. But received rank = %d",
                          prior_box_var_shape.size()));

    for (size_t i = 0; i < prior_box_shape.size(); i++) {
      infer_context->AddEqualCstr(prior_box_shape[i], prior_box_var_shape[i]);
    }
  }

  if (code_type == "encode_center_size") {
    PADDLE_ENFORCE_EQ(target_box_shape.size(),
                      2,
                      common::errors::InvalidArgument(
                          "The rank of Input TargetBox in BoxCoder operator "
                          "must be 2. But received rank is %d",
                          target_box_shape.size()));

    infer_context->AddEqualCstr(target_box_shape[1], symbol::DimExpr{4});
    infer_context->SetShapeOrDataForValue(
        op->result(0),
        symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(
            {target_box_shape[0], prior_box_shape[0], symbol::DimExpr(4)})});
  } else if (code_type == "decode_center_size") {
    PADDLE_ENFORCE_EQ(target_box_shape.size(),
                      3,
                      common::errors::InvalidArgument(
                          "The rank of Input TargetBox in BoxCoder operator "
                          "must be 3. But received rank is %d",
                          target_box_shape.size()));
    PADDLE_ENFORCE_EQ(axis == 0 || axis == 1,
                      true,
                      common::errors::InvalidArgument(
                          "axis in BoxCoder operator must be 0 or 1. "
                          "But received axis = %d",
                          axis));
    if (axis == 0) {
      infer_context->AddEqualCstr(target_box_shape[1], prior_box_shape[0]);
    } else if (axis == 1) {
      infer_context->AddEqualCstr(target_box_shape[0], prior_box_shape[0]);
    }
    infer_context->AddEqualCstr(target_box_shape[2], prior_box_shape[1]);
    infer_context->SetShapeOrDataForValue(
        op->result(0),
        symbol::ShapeOrDataDimExprs{
            symbol::TensorShapeOrDataDimExprs(target_box_shape)});
  }

  return true;
}

bool CheckFiniteAndUnscaleOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  // Retrieve the shape information of the input list
  pir::Value operand_source = op->operand_source(0);
  const symbol::TensorListShapeOrDataDimExprs &xs_shapes =
      infer_context->GetShapeOrDataForValue(operand_source)
          .dyn_cast<symbol::TensorListShapeOrDataDimExprs>();

  // Set the shapes for the output tensor list
  symbol::TensorListShapeOrDataDimExprs outs_shapes;
  outs_shapes.reserve(xs_shapes.size());
  for (const auto &input_shape : xs_shapes) {
    outs_shapes.emplace_back(input_shape.shape());
  }
  infer_context->SetShapeOrDataForValue(
      op->result(0), symbol::ShapeOrDataDimExprs{outs_shapes});

  // Set the shape for the found_infinite output
  symbol::TensorShapeOrDataDimExprs found_infinite_shape({symbol::DimExpr(1)});
  infer_context->SetShapeOrDataForValue(
      op->result(1), symbol::ShapeOrDataDimExprs{found_infinite_shape});

  return true;
}

bool CheckFiniteAndUnscale_OpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  return CheckFiniteAndUnscaleOpInferSymbolicShape(op, infer_context);
}

bool ChunkEvalOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const symbol::ShapeOrDataDimExprs &inference_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const symbol::ShapeOrDataDimExprs &label_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));

  const std::vector<symbol::DimExpr> &inference_shape =
      inference_shape_or_data.shape();
  const std::vector<symbol::DimExpr> &label_shape = label_shape_or_data.shape();

  PADDLE_ENFORCE_EQ(
      inference_shape.size(),
      label_shape.size(),
      common::errors::InvalidArgument(
          "Input(Inference)'s rank must be the same as Input(Label)'s rank, "
          "but received [%s] (Inference) vs [%s] (Label).",
          inference_shape.size(),
          label_shape.size()));

  for (size_t i = 0; i < inference_shape.size(); ++i) {
    infer_context->AddEqualCstr(inference_shape[i], label_shape[i]);
  }

  if (op->operand_source(2)) {
    const symbol::ShapeOrDataDimExprs &seq_length_shape_or_data =
        infer_context->GetShapeOrDataForValue(op->operand_source(2));
    const std::vector<symbol::DimExpr> &seq_length_shape =
        seq_length_shape_or_data.shape();

    PADDLE_ENFORCE_EQ(
        inference_shape.size() == 3 || inference_shape.size() == 2,
        true,
        common::errors::InvalidArgument(
            "when Input(SeqLength) is provided, Input(Inference) "
            "should be of dim 3 or dim 2 "
            "but received [%s].",
            inference_shape));

    if (inference_shape.size() == 3) {
      infer_context->AddEqualCstr(inference_shape[2], symbol::DimExpr(1));
    }

    PADDLE_ENFORCE_LE(
        seq_length_shape.size(),
        2,
        common::errors::InvalidArgument("Input(SeqLength)'s rank should not be "
                                        "greater than 2, but received %d.",
                                        seq_length_shape.size()));
  }

  std::vector<symbol::DimExpr> scalar_shape{symbol::DimExpr(1)};
  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(scalar_shape)});
  infer_context->SetShapeOrDataForValue(
      op->result(1),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(scalar_shape)});
  infer_context->SetShapeOrDataForValue(
      op->result(2),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(scalar_shape)});
  infer_context->SetShapeOrDataForValue(
      op->result(3),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(scalar_shape)});
  infer_context->SetShapeOrDataForValue(
      op->result(4),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(scalar_shape)});
  infer_context->SetShapeOrDataForValue(
      op->result(5),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(scalar_shape)});

  return true;
}

bool CrfDecodingOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &emission_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const std::vector<symbol::DimExpr> emission_dims =
      emission_shape_or_data.shape();

  const auto &transition_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));
  const std::vector<symbol::DimExpr> transition_dims =
      transition_shape_or_data.shape();

  const auto &label_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(2));
  const std::vector<symbol::DimExpr> label_dims = label_shape_or_data.shape();

  const auto &length_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(3));

  const auto one = symbol::DimExpr{1};

  if (!length_shape_or_data.isa<symbol::NullShapeOrDataDimExpr>()) {
    PADDLE_ENFORCE_EQ(emission_dims.size(),
                      3,
                      common::errors::InvalidArgument(
                          "The Input(Emission) should be a 3-D tensor. But "
                          "received: input rank %u, input shape [%s]. ",
                          emission_dims.size(),
                          emission_dims));
  } else {
    PADDLE_ENFORCE_EQ(emission_dims.size(),
                      2,
                      common::errors::InvalidArgument(
                          "The Input(Emission) should be a 2-D tensor. But "
                          "received: input rank %u, input shape [%s].",
                          emission_dims.size(),
                          emission_dims));
  }

  PADDLE_ENFORCE_EQ(transition_dims.size(),
                    2UL,
                    common::errors::InvalidArgument(
                        "The Input(Transition) should be a 2-D tensor. But "
                        "received: input rank %u, input shape [%s].",
                        transition_dims.size(),
                        transition_dims));
  infer_context->AddEqualCstr(transition_dims[0] - 2, transition_dims[1]);

  infer_context->AddEqualCstr(emission_dims[emission_dims.size() - 1],
                              transition_dims[transition_dims.size() - 1]);

  if (!label_dims.empty()) {
    if (!length_shape_or_data.isa<symbol::NullShapeOrDataDimExpr>()) {
      if (label_dims.size() == 3UL) {
        infer_context->AddEqualCstr(label_dims[2], one);
      } else {
        PADDLE_ENFORCE_EQ(
            label_dims.size(),
            2UL,
            common::errors::InvalidArgument(
                "The Input(Label) should be a 3-D tensor with last dimension "
                "fixed to 1 or a 2-D tensor in padding mode. But received: "
                "input "
                "rank %u, input shape [%s].",
                label_dims.size(),
                label_dims));
      }
    } else {
      if (label_dims.size() == 2UL) {
        infer_context->AddEqualCstr(label_dims[2], one);
      } else {
        PADDLE_ENFORCE_EQ(
            label_dims.size(),
            1UL,
            common::errors::InvalidArgument(
                "The Input(Label) should be a 2-D tensor with last "
                "dimension fixed to 1 or a 1-D tensor. But received: "
                "input rank %u, input shape [%s].",
                label_dims.size(),
                label_dims));
      }
    }

    infer_context->AddEqualCstr(emission_dims[0], label_dims[0]);

    std::vector<symbol::DimExpr> viterbi_path_dims;
    viterbi_path_dims.push_back(emission_dims[0]);
    if (!length_shape_or_data.isa<symbol::NullShapeOrDataDimExpr>()) {
      viterbi_path_dims.push_back(emission_dims[1]);
    } else {
      viterbi_path_dims.push_back(symbol::DimExpr(1));
    }

    infer_context->SetShapeOrDataForValue(
        op->result(0), symbol::TensorShapeOrDataDimExprs(viterbi_path_dims));
  }
  return true;
}

bool CoalesceTensorOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  pir::Value operand_source = op->operand_source(0);
  const symbol::TensorListShapeOrDataDimExprs &input_shapes =
      infer_context->GetShapeOrDataForValue(operand_source)
          .dyn_cast<symbol::TensorListShapeOrDataDimExprs>();

  auto dtype = op->attribute("dtype")
                   .dyn_cast<paddle::dialect::DataTypeAttribute>()
                   .data();
  const auto &attributes = op->attributes();
  bool use_align =
      attributes.at("use_align").dyn_cast<pir::BoolAttribute>().data();
  int align_size =
      attributes.at("align_size").dyn_cast<pir::Int32Attribute>().data();
  int size_of_dtype =
      attributes.at("size_of_dtype").dyn_cast<pir::Int32Attribute>().data();

  if (size_of_dtype == -1) {
    size_of_dtype = static_cast<int>(phi::SizeOf(dtype));
  }

  auto alignment = [](size_t size, size_t align_size) {
    size_t remaining = size % align_size;
    auto aligned_size = remaining == 0 ? size : size + (align_size - remaining);
    return aligned_size;
  };

  if (use_align && align_size > 0) {
    int64_t numel = 0;
    for (const auto &item_shape : input_shapes) {
      const std::vector<symbol::DimExpr> dims = item_shape.shape();
      auto size = dims.size();
      auto len = use_align
                     ? alignment(static_cast<size_t>(size) * size_of_dtype,
                                 align_size) /
                           size_of_dtype
                     : static_cast<size_t>(size);
      numel += static_cast<int64_t>(len);
    }
    infer_context->SetShapeOrDataForValue(
        op->result(1),
        symbol::ShapeOrDataDimExprs{
            symbol::TensorShapeOrDataDimExprs({numel})});
  }

  symbol::TensorListShapeOrDataDimExprs outs_shapes;
  outs_shapes.reserve(input_shapes.size());
  for (const auto &input_shape : input_shapes) {
    outs_shapes.emplace_back(input_shape.shape());
  }
  infer_context->SetShapeOrDataForValue(
      op->result(0), symbol::TensorListShapeOrDataDimExprs(outs_shapes));

  return true;
}

bool CoalesceTensor_OpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  return CoalesceTensorOpInferSymbolicShape(op, infer_context);
}

bool CrossEntropyWithSoftmaxOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const symbol::ShapeOrDataDimExprs &input_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const symbol::ShapeOrDataDimExprs &index_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));

  const auto &input_dim = input_shape.shape();
  const auto &index_dim = index_shape.shape();
  const auto &attributes = op->attributes();
  int axis = attributes.at("axis").dyn_cast<pir::Int32Attribute>().data();
  if (axis < 0) axis += input_shape.shape().size();
  bool soft_label =
      attributes.at("soft_label").dyn_cast<pir::BoolAttribute>().data();
  PADDLE_ENFORCE(!soft_label || input_dim.size() == index_dim.size(),
                 common::errors::InvalidArgument(
                     "The input and index should have the same rank when "
                     "soft_label is true. But received input rank(%d) and "
                     "index rank(%d)",
                     input_dim.size(),
                     index_dim.size()));

  auto softmax_dim = index_dim;
  auto out_dim = index_dim;

  if (index_dim.size() == input_dim.size()) {
    if (soft_label) {
      out_dim[axis] = 1;
    }
    softmax_dim[axis] = input_dim[axis];
  } else {
    softmax_dim.insert(softmax_dim.begin() + axis, input_dim[axis]);
    if (soft_label) {
      out_dim.insert(out_dim.begin() + axis, 1);
    }
  }

  infer_context->SetShapeOrDataForValue(
      op->result(0), symbol::TensorShapeOrDataDimExprs(softmax_dim));
  infer_context->SetShapeOrDataForValue(
      op->result(1), symbol::TensorShapeOrDataDimExprs(out_dim));

  return true;
}

bool CrossEntropyWithSoftmax_OpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  return CrossEntropyWithSoftmaxOpInferSymbolicShape(op, infer_context);
}

bool ConcatOpInferSymbolicShape(pir::Operation *op,
                                pir::InferSymbolicShapeContext *infer_context) {
  const auto &axis_expr =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));
  if (!axis_expr.data() || !axis_expr.data()->at(0).isa<int64_t>()) {
    pir::Value res = op->result(0);
    infer_context->SetSymbolForValueByStaticShape(res);
    return true;
  }

  pir::Value operand_source = op->operand_source(0);
  const auto &x_shape = infer_context->GetShapeOrDataForValue(operand_source);
  const auto &shape_data_list =
      x_shape.dyn_cast<symbol::TensorListShapeOrDataDimExprs>();

  size_t rank = shape_data_list.at(0).shape().size();
  const int64_t axis = [&] {
    int64_t axis = axis_expr.data()->at(0).dyn_cast<int64_t>();
    return axis >= 0 ? axis : std::max(int64_t(0), int64_t(axis + rank));
  }();

  if (details::HasCompleteData(x_shape) && (rank == 1)) {
    ExprVec data = details::GetExprVecFromData(x_shape);
    const std::vector<symbol::DimExpr> shape{std::int64_t(data.size())};
    symbol::ShapeOrDataDimExprs shape_data{
        symbol::TensorShapeOrDataDimExprs(shape, data)};
    pir::Value res = op->result(0);
    infer_context->SetShapeOrDataForValue(res, shape_data);

    return true;
  }

  const std::vector<symbol::DimExpr> &out_dims = [&] {
    std::vector<symbol::DimExpr> out_dims = shape_data_list.at(0).shape();
    for (size_t i = 0; i < rank; ++i) {
      if (i != static_cast<size_t>(axis)) {
        details::BuildCstrEqForTensorListAlongAxis(
            infer_context, shape_data_list, i);
        continue;
      }
      for (size_t j = 1; j < shape_data_list.size(); ++j) {
        out_dims.at(axis) =
            out_dims.at(axis) + shape_data_list.at(j).shape().at(axis);
      }
    }
    return out_dims;
  }();

  symbol::ShapeOrDataDimExprs shape_data{
      symbol::TensorShapeOrDataDimExprs(out_dims)};

  pir::Value res = op->result(0);
  infer_context->SetShapeOrDataForValue(res, shape_data);

  return true;
}

bool DetectionMapOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &detect_res_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const auto &label_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));
  const auto &pos_count_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(3));
  const auto &true_pos_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(4));
  const auto &false_pos_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(5));

  const auto &detect_res_shape = detect_res_shape_or_data.shape();
  const auto &label_shape = label_shape_or_data.shape();

  PADDLE_ENFORCE_EQ(detect_res_shape.size(),
                    2UL,
                    common::errors::InvalidArgument(
                        "Input(DetectRes) ndim must be 2, the shape is [N, 6],"
                        "but received the ndim is %d",
                        detect_res_shape.size()));
  infer_context->AddEqualCstr(detect_res_shape[1], symbol::DimExpr(6));
  PADDLE_ENFORCE_EQ(label_shape.size(),
                    2,
                    common::errors::InvalidArgument(
                        "The ndim of Input(Label) must be 2, but received %d",
                        label_shape.size()));

  if (!pos_count_shape_or_data.isa<symbol::NullShapeOrDataDimExpr>()) {
    PADDLE_ENFORCE_EQ(
        !true_pos_shape_or_data.isa<symbol::NullShapeOrDataDimExpr>(),
        true,
        common::errors::InvalidArgument(
            "Input(TruePos) of DetectionMAPOp should not be null when "
            "Input(PosCount) is not null."));
    PADDLE_ENFORCE_EQ(
        !false_pos_shape_or_data.isa<symbol::NullShapeOrDataDimExpr>(),
        true,
        common::errors::InvalidArgument(
            "Input(FalsePos) of DetectionMAPOp should not be null when "
            "Input(PosCount) is not null."));
  }

  std::vector<symbol::DimExpr> output_pos_count = {
      infer_context->GetNextSymName(), symbol::DimExpr(1)};
  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(output_pos_count)});

  std::vector<symbol::DimExpr> output_true_pos = {
      infer_context->GetNextSymName(), symbol::DimExpr(2)};
  infer_context->SetShapeOrDataForValue(
      op->result(1),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(output_true_pos)});

  std::vector<symbol::DimExpr> output_false_pos = {
      infer_context->GetNextSymName(), symbol::DimExpr(2)};
  infer_context->SetShapeOrDataForValue(
      op->result(2),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(output_false_pos)});

  std::vector<symbol::DimExpr> m_ap_shape = {symbol::DimExpr(1)};
  infer_context->SetShapeOrDataForValue(
      op->result(3),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(m_ap_shape)});

  return true;
}
// bool CudnnLstmOpInferSymbolicShape(pir::Operation *op,
//                                    pir::InferSymbolicShapeContext
//                                    *infer_context) {
//   // pass
//   return true;
// }

bool DeformableConvOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const std::vector<symbol::DimExpr> &x_shape = x_shape_or_data.shape();
  const auto &offset_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));
  const std::vector<symbol::DimExpr> &offset_shape =
      offset_shape_or_data.shape();
  const auto &filter_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(2));
  const std::vector<symbol::DimExpr> &filter_shape =
      filter_shape_or_data.shape();

  const std::vector<int> &strides = details::GetVectorAttr<int>(op, "strides");
  const std::vector<int> &paddings =
      details::GetVectorAttr<int>(op, "paddings");
  const std::vector<int> &dilations =
      details::GetVectorAttr<int>(op, "dilations");
  const int groups = op->attribute<pir::Int32Attribute>("groups").data();
  const int deformable_groups =
      op->attribute<pir::Int32Attribute>("deformable_groups").data();

  infer_context->AddEqualCstr(x_shape[1],
                              filter_shape[1] * symbol::DimExpr(groups));

  std::vector<symbol::DimExpr> output_shape = {x_shape[0], filter_shape[0]};
  for (size_t i = 0; i < strides.size(); ++i) {
    symbol::DimExpr conv_output_size, dkernel;
    dkernel = symbol::DimExpr(dilations[i]) *
                  (filter_shape[i + 2] - symbol::DimExpr(1)) +
              symbol::DimExpr(1);
    conv_output_size =
        (x_shape[i + 2] + symbol::DimExpr(2 * paddings[i]) - dkernel) /
            symbol::DimExpr(strides[i]) +
        symbol::DimExpr(1);
    output_shape.emplace_back(conv_output_size);
  }

  infer_context->AddEqualCstr(output_shape[2], offset_shape[2]);
  infer_context->AddEqualCstr(output_shape[3], offset_shape[3]);
  infer_context->AddEqualCstr(offset_shape[1],
                              symbol::DimExpr(2 * deformable_groups) *
                                  filter_shape[2] * filter_shape[3]);
  if (op->operand_source(3)) {
    const auto &mask_shape_or_data =
        infer_context->GetShapeOrDataForValue(op->operand_source(3));
    const std::vector<symbol::DimExpr> &mask_shape = mask_shape_or_data.shape();
    infer_context->AddEqualCstr(output_shape[2], mask_shape[2]);
    infer_context->AddEqualCstr(output_shape[3], mask_shape[3]);
    infer_context->AddEqualCstr(
        mask_shape[1],
        symbol::DimExpr(deformable_groups) * filter_shape[2] * filter_shape[3]);
  }

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(output_shape)});

  return true;
}

bool FakeQuantizeRangeAbsMaxOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));

  int window_size = op->attribute<pir::Int32Attribute>("window_size").data();
  int bit_length = op->attribute<pir::Int32Attribute>("bit_length").data();

  PADDLE_ENFORCE_EQ(bit_length >= 1 && bit_length <= 16,
                    true,
                    common::errors::InvalidArgument(
                        "'bit_length' should be between 1 and 16, but "
                        "the received is %d",
                        bit_length));
  const auto &out_scale = symbol::DimExpr(1);
  const auto &out_scales = symbol::DimExpr(window_size);
  infer_context->SetShapeOrDataForValue(op->result(0), x_shape_or_data);
  infer_context->SetShapeOrDataForValue(
      op->result(1),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs({out_scale})});
  if (op->result(2)) {
    infer_context->SetShapeOrDataForValue(
        op->result(2),
        symbol::ShapeOrDataDimExprs{
            symbol::TensorShapeOrDataDimExprs({out_scales})});
  }
  return true;
}

bool EditDistanceOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &hyps_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const auto &refs_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));

  const auto &hyps_dims = hyps_shape_or_data.shape();
  const auto &refs_dims = refs_shape_or_data.shape();

  PADDLE_ENFORCE_EQ(
      hyps_dims.size(),
      2,
      common::errors::InvalidArgument(
          "Input(Hyps) must be a 2-D Tensor, but received rank %u.",
          hyps_dims.size()));
  PADDLE_ENFORCE_EQ(
      refs_dims.size(),
      2,
      common::errors::InvalidArgument(
          "Input(Refs) must be a 2-D Tensor, but received rank %u.",
          refs_dims.size()));

  infer_context->AddEqualCstr(hyps_dims[0], refs_dims[0]);

  bool has_lengths = op->operand_source(2) && op->operand_source(3);
  if (has_lengths) {
    const auto &hypslength_shape_or_data =
        infer_context->GetShapeOrDataForValue(op->operand_source(2));
    const auto &refslength_shape_or_data =
        infer_context->GetShapeOrDataForValue(op->operand_source(3));

    infer_context->AddEqualCstr(hypslength_shape_or_data.shape()[0],
                                hyps_dims[0]);

    infer_context->AddEqualCstr(refslength_shape_or_data.shape()[0],
                                refs_dims[0]);

    infer_context->AddEqualCstr(hypslength_shape_or_data.shape()[0],
                                refslength_shape_or_data.shape()[0]);
  } else {
    symbol::DimExpr one = symbol::DimExpr(1);

    infer_context->AddEqualCstr(hyps_dims[1], one);
    infer_context->AddEqualCstr(refs_dims[1], one);
  }

  symbol::ShapeOrDataDimExprs out_shape_or_data_exprs(
      symbol::TensorShapeOrDataDimExprs(
          std::vector<symbol::DimExpr>{refs_dims}));
  infer_context->SetShapeOrDataForValue(op->result(0), out_shape_or_data_exprs);

  symbol::ShapeOrDataDimExprs single_dim_expr(symbol::TensorShapeOrDataDimExprs(
      std::vector<symbol::DimExpr>{symbol::DimExpr(1)}));
  infer_context->SetShapeOrDataForValue(
      op->result(1), symbol::ShapeOrDataDimExprs({single_dim_expr}));

  return true;
}

bool FakeQuantizeMovingAverageAbsMaxOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(0)).shape();

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
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs({x_shape})});
  // Create a scalar shape for the other output tensors
  symbol::TensorShapeOrDataDimExprs scalar_shape(
      std::vector<symbol::DimExpr>{symbol::DimExpr(1)});

  // Set the shape for all scalar output tensors: 'out_scale', 'out_state',
  // 'out_accum'
  for (size_t i = 1; i < op->num_results(); ++i) {
    infer_context->SetShapeOrDataForValue(op->result(i), scalar_shape);
  }

  return true;
}

bool FakeQuantizeRangeAbsMax_OpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  return FakeQuantizeRangeAbsMaxOpInferSymbolicShape(op, infer_context);
}

bool FakeQuantizeMovingAverageAbsMax_OpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  return FakeQuantizeMovingAverageAbsMaxOpInferSymbolicShape(op, infer_context);
}

bool FakeQuantizeDequantizeMovingAverageAbsMaxOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  return FakeQuantizeMovingAverageAbsMaxOpInferSymbolicShape(op, infer_context);
}

bool FakeQuantizeDequantizeMovingAverageAbsMax_OpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  return FakeQuantizeMovingAverageAbsMaxOpInferSymbolicShape(op, infer_context);
}

bool FullWithTensorOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  pir::Value operand_source = op->operand_source(1);
  const symbol::ShapeOrDataDimExprs &operand_shape_or_data =
      infer_context->GetShapeOrDataForValue(operand_source);

  const auto &out_shape =
      details::GetOrCreateExprVecFromData(operand_shape_or_data, infer_context);

  infer_context->SetShapeOrDataForValue(
      op->result(0), symbol::TensorShapeOrDataDimExprs(out_shape));
  return true;
}

bool FlashAttnOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  pir::Value operand_source = op->operand_source(0);
  const symbol::ShapeOrDataDimExprs &q =
      infer_context->GetShapeOrDataForValue(operand_source);

  const symbol::ShapeOrDataDimExprs &k =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));

  const symbol::ShapeOrDataDimExprs &v =
      infer_context->GetShapeOrDataForValue(op->operand_source(2));

  PADDLE_ENFORCE_EQ(q.shape().size(),
                    4,
                    common::errors::InvalidArgument(
                        "flash_attn receive input with dim "
                        "[batch_size, seq_len, num_heads, head_dim]"));

  infer_context->AddEqualCstr(q.shape()[0], k.shape()[0]);
  infer_context->AddEqualCstr(q.shape()[0], v.shape()[0]);
  infer_context->AddEqualCstr(k.shape()[1], v.shape()[1]);

  if (op->operand_source(4)) {
    const symbol::ShapeOrDataDimExprs &attn_mask =
        infer_context->GetShapeOrDataForValue(op->operand_source(4));
    infer_context->AddEqualCstr(attn_mask.shape()[0], q.shape()[0]);
    infer_context->AddEqualCstr(attn_mask.shape()[2], q.shape()[1]);
    infer_context->AddEqualCstr(attn_mask.shape()[3], k.shape()[1]);
  }

  std::vector<symbol::DimExpr> out_shape = q.shape();

  out_shape.back() = v.shape().back();

  infer_context->SetShapeOrDataForValue(
      op->result(0), symbol::TensorShapeOrDataDimExprs(out_shape));

  // GPU has round for seqlen, but XPU has not. Here we align with the GPU
  // version.
  auto round_multiple = [](symbol::DimExpr x) {
    auto m = symbol::DimExpr{128};
    auto m_minus_one = symbol::DimExpr{127};
    return (x + m_minus_one) / m * m;
  };
  auto batch_size_expr = q.shape()[0];
  auto num_heads_expr = q.shape()[2];
  auto seqlen_q_rounded_expr = round_multiple(q.shape()[1]);
  auto seqlen_k_rounded_expr = round_multiple(k.shape()[1]);
  if (op->result(1)) {
    std::vector<symbol::DimExpr> softmax_shape{batch_size_expr,
                                               num_heads_expr,
                                               seqlen_q_rounded_expr,
                                               seqlen_k_rounded_expr};
    infer_context->SetShapeOrDataForValue(
        op->result(1), symbol::TensorShapeOrDataDimExprs(softmax_shape));
  }
  if (op->result(2)) {
    std::vector<symbol::DimExpr> softmax_lse_shape{
        batch_size_expr, num_heads_expr, seqlen_q_rounded_expr};
    infer_context->SetShapeOrDataForValue(
        op->result(2), symbol::TensorShapeOrDataDimExprs(softmax_lse_shape));
  }
  if (op->result(3)) {
    std::vector<symbol::DimExpr> seed_offset_shape{symbol::DimExpr{2}};
    infer_context->SetShapeOrDataForValue(
        op->result(3), symbol::TensorShapeOrDataDimExprs(out_shape));
  }
  return true;
}

bool FusedFeedforwardOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  std::vector<symbol::DimExpr> x_shape = x_shape_or_data.shape();

  auto RowMatrixFromVector = [](const std::vector<symbol::DimExpr> &x_shape)
      -> std::vector<symbol::DimExpr> {
    if (x_shape.size() > 1) {
      return x_shape;
    }
    return {symbol::DimExpr(1), x_shape[0]};
  };
  std::vector<symbol::DimExpr> tensor_dim = RowMatrixFromVector(x_shape);
  symbol::DimExpr height_;
  symbol::DimExpr width_;
  symbol::DimExpr stride_;
  symbol::DimExpr batch_size_ = 1;

  if (tensor_dim.size() == 2) {
    height_ = tensor_dim[0];
    width_ = tensor_dim[1];
  } else {
    std::vector<symbol::DimExpr> dim_vec = tensor_dim;
    for (size_t i = 0; i < dim_vec.size() - 2; ++i) {
      batch_size_ = batch_size_ * dim_vec[i];
    }
    height_ = dim_vec[dim_vec.size() - 2];
    width_ = dim_vec[dim_vec.size() - 1];
    stride_ = height_ * width_;
  }

  const auto &linear1_weight_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(3));
  std::vector<symbol::DimExpr> linear1_weight_dims =
      linear1_weight_shape_or_data.shape();

  std::vector<symbol::DimExpr> tmp_dim_x = x_shape;
  tmp_dim_x.back() = linear1_weight_dims.back();

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(x_shape)});

  bool is_test = op->attribute<pir::BoolAttribute>("is_test").data();

  if (!is_test) {
    infer_context->SetShapeOrDataForValue(
        op->result(1),
        symbol::ShapeOrDataDimExprs{
            symbol::TensorShapeOrDataDimExprs(tmp_dim_x)});
  }
  infer_context->SetShapeOrDataForValue(
      op->result(9),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(tmp_dim_x)});
  infer_context->SetShapeOrDataForValue(
      op->result(7),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(tmp_dim_x)});
  infer_context->SetShapeOrDataForValue(
      op->result(10),
      symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(x_shape)});

  if (!is_test) {
    infer_context->SetShapeOrDataForValue(
        op->result(2),
        symbol::ShapeOrDataDimExprs{
            symbol::TensorShapeOrDataDimExprs(x_shape)});
  }

  std::vector<symbol::DimExpr> mean_dim =
      std::vector<symbol::DimExpr>{batch_size_ * height_};

  bool pre_layer_norm =
      op->attribute<pir::BoolAttribute>("pre_layer_norm").data();

  if (pre_layer_norm) {
    if (!paddle::dialect::details::IsFakeValue(op->result(8))) {
      infer_context->SetShapeOrDataForValue(
          op->result(8),
          symbol::ShapeOrDataDimExprs{
              symbol::TensorShapeOrDataDimExprs(x_shape)});
    }

    if (!paddle::dialect::details::IsFakeValue(op->result(3))) {
      infer_context->SetShapeOrDataForValue(
          op->result(3),
          symbol::ShapeOrDataDimExprs{
              symbol::TensorShapeOrDataDimExprs(mean_dim)});
    }

    if (!paddle::dialect::details::IsFakeValue(op->result(4))) {
      infer_context->SetShapeOrDataForValue(
          op->result(4),
          symbol::ShapeOrDataDimExprs{
              symbol::TensorShapeOrDataDimExprs(mean_dim)});
    }

  } else {
    if (!paddle::dialect::details::IsFakeValue(op->result(5))) {
      infer_context->SetShapeOrDataForValue(
          op->result(5),
          symbol::ShapeOrDataDimExprs{
              symbol::TensorShapeOrDataDimExprs(mean_dim)});
    }

    if (!paddle::dialect::details::IsFakeValue(op->result(6))) {
      infer_context->SetShapeOrDataForValue(
          op->result(6),
          symbol::ShapeOrDataDimExprs{
              symbol::TensorShapeOrDataDimExprs(mean_dim)});
    }
  }

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(x_shape)});

  return true;
}

bool FusedAttentionOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const auto &qkv_weight_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(3));
  const auto &cache_kv_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(5));
  const auto &src_mask_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(6));
  const auto &qkv_bias_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(4));
  const std::vector<symbol::DimExpr> &x_shape = x_shape_or_data.shape();
  const std::vector<symbol::DimExpr> &qkv_weight_shape =
      qkv_weight_shape_or_data.shape();
  symbol::DimExpr dim_head = 0;
  symbol::DimExpr hidden_size = 0;
  symbol::DimExpr nranks = 1;
  const bool transpose_qkv_wb =
      op->attribute<pir::BoolAttribute>("transpose_qkv_wb").data();
  const int num_heads_ = op->attribute<pir::Int32Attribute>("num_heads").data();
  symbol::DimExpr num_heads = symbol::DimExpr(num_heads_);
  const int ring_id = op->attribute<pir::Int32Attribute>("ring_id").data();
  const bool pre_layer_norm =
      op->attribute<pir::BoolAttribute>("pre_layer_norm").data();
  const bool is_test = op->attribute<pir::BoolAttribute>("is_test").data();
  if (transpose_qkv_wb) {
    PADDLE_ENFORCE_EQ(qkv_weight_shape.size(),
                      2,
                      common::errors::InvalidArgument(
                          "The dimensions of qkv_weight must be 2 if enable"
                          "transpose_qkv_wb: (dim_embed, 3 * dim_embed),"
                          "but received dimensions of"
                          "Input is [%d]",
                          qkv_weight_shape.size()));
    PADDLE_ENFORCE_GT(num_heads_,
                      0,
                      common::errors::InvalidArgument(
                          "The num_heads must be provided and greater than 0 "
                          "if enable transpose_qkv_wb, but we got %d.",
                          num_heads));
    infer_context->AddEqualCstr((qkv_weight_shape[0] / num_heads) * num_heads,
                                qkv_weight_shape[0]);
    if (ring_id == -1) {
      infer_context->AddEqualCstr(qkv_weight_shape[0] * symbol::DimExpr(3),
                                  qkv_weight_shape[1]);
    } else {
      nranks = (qkv_weight_shape[0] * symbol::DimExpr(3)) / qkv_weight_shape[1];
    }
    dim_head = qkv_weight_shape[0] / (num_heads * nranks);
    hidden_size = qkv_weight_shape[0];
  } else {
    PADDLE_ENFORCE_EQ(qkv_weight_shape.size(),
                      4,
                      common::errors::InvalidArgument(
                          "The dimensions of qkv_weight must be 4 if not"
                          "enable transpose_qkv_wb: (3, num_head, dim_head, "
                          "dim_embed), but received [%d]",
                          qkv_weight_shape.size()));
    infer_context->AddEqualCstr(qkv_weight_shape[0], symbol::DimExpr(3));
    if (ring_id == -1) {
      infer_context->AddEqualCstr(qkv_weight_shape[1] * qkv_weight_shape[2],
                                  qkv_weight_shape[3]);
    }
    num_heads = qkv_weight_shape[1];
    dim_head = qkv_weight_shape[2];
    hidden_size = qkv_weight_shape[3];
  }
  PADDLE_ENFORCE_EQ(x_shape.size(),
                    3,
                    common::errors::InvalidArgument(
                        "The dimensions of x must be 3 (batch_size, seq_len, "
                        "dim_embed), but received dimensions of Input is [%d]",
                        x_shape.size()));
  infer_context->AddEqualCstr(x_shape[2], hidden_size);
  if (pre_layer_norm) {
    infer_context->SetShapeOrDataForValue(
        op->result(0),
        symbol::ShapeOrDataDimExprs{
            symbol::TensorShapeOrDataDimExprs({x_shape[0] * x_shape[1]})});
    infer_context->SetShapeOrDataForValue(
        op->result(1),
        symbol::ShapeOrDataDimExprs{
            symbol::TensorShapeOrDataDimExprs({x_shape[0] * x_shape[1]})});
    infer_context->SetShapeOrDataForValue(
        op->result(2),
        symbol::ShapeOrDataDimExprs{
            symbol::TensorShapeOrDataDimExprs(x_shape)});
  } else {
    // The following three code used to set unoptional output value.
    // Now it's result related to the infermeta.
    infer_context->SetSymbolForValueByStaticShape(op->result(0));
    infer_context->SetSymbolForValueByStaticShape(op->result(1));
    infer_context->SetSymbolForValueByStaticShape(op->result(2));
    if (paddle::dialect::details::IsFakeValue(op->result(15))) {
      infer_context->SetSymbolForValueByStaticShape(op->result(15));
    } else {
      infer_context->SetShapeOrDataForValue(
          op->result(15),
          symbol::ShapeOrDataDimExprs{
              symbol::TensorShapeOrDataDimExprs({x_shape[0] * x_shape[1]})});
    }
    if (paddle::dialect::details::IsFakeValue(op->result(16))) {
      infer_context->SetSymbolForValueByStaticShape(op->result(16));
    } else {
      infer_context->SetShapeOrDataForValue(
          op->result(16),
          symbol::ShapeOrDataDimExprs{
              symbol::TensorShapeOrDataDimExprs({x_shape[0] * x_shape[1]})});
    }
    if (paddle::dialect::details::IsFakeValue(op->result(17))) {
      infer_context->SetSymbolForValueByStaticShape(op->result(17));
    } else {
      infer_context->SetShapeOrDataForValue(
          op->result(17),
          symbol::ShapeOrDataDimExprs{
              symbol::TensorShapeOrDataDimExprs(x_shape)});
    }
  }
  if (transpose_qkv_wb) {
    // [batch_size, seq_len, 3 * num_heads * dim_head]
    infer_context->SetShapeOrDataForValue(
        op->result(3),
        symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(
            {x_shape[0],
             x_shape[1],
             symbol::DimExpr(3) * num_heads * dim_head})});
    if (!qkv_bias_shape_or_data.isa<symbol::NullShapeOrDataDimExpr>()) {
      infer_context->SetShapeOrDataForValue(
          op->result(4),
          symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(
              {x_shape[0],
               x_shape[1],
               symbol::DimExpr(3) * num_heads * dim_head})});
    } else {
      // The following code used to set unoptional output value.
      // Now it's result related to the infermeta.
      infer_context->SetSymbolForValueByStaticShape(op->result(4));
    }
  } else {
    // [batch_size, seq_len, 3, num_head, head_size]
    infer_context->SetShapeOrDataForValue(
        op->result(3),
        symbol::ShapeOrDataDimExprs{
            symbol::TensorShapeOrDataDimExprs({x_shape[0],
                                               x_shape[1],
                                               symbol::DimExpr(3),
                                               num_heads,
                                               dim_head})});
    if (!qkv_bias_shape_or_data.isa<symbol::NullShapeOrDataDimExpr>()) {
      infer_context->SetShapeOrDataForValue(
          op->result(4),
          symbol::ShapeOrDataDimExprs{
              symbol::TensorShapeOrDataDimExprs({x_shape[0],
                                                 x_shape[1],
                                                 symbol::DimExpr(3),
                                                 num_heads,
                                                 dim_head})});
    } else {
      // The following code used to set unoptional output value.
      // Now it's result related to the infermeta.
      infer_context->SetSymbolForValueByStaticShape(op->result(4));
    }
  }
  // [3, batch_size, num_head, seq_len, head_size]
  infer_context->SetShapeOrDataForValue(
      op->result(5),
      symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(
          {symbol::DimExpr(3), x_shape[0], num_heads, x_shape[1], dim_head})});
  // cache_seq_len + seq_len if cache else seq_len
  symbol::DimExpr out_seq_len = x_shape[1];
  if (!cache_kv_shape_or_data.isa<symbol::NullShapeOrDataDimExpr>()) {
    const std::vector<symbol::DimExpr> &cache_kv_shape =
        cache_kv_shape_or_data.shape();
    PADDLE_ENFORCE_EQ(
        cache_kv_shape.size(),
        5,
        common::errors::InvalidArgument(
            "The CacheKV must be 5 dims, but got %d", cache_kv_shape.size()));
    infer_context->AddEqualCstr(cache_kv_shape[0], symbol::DimExpr(2));
    infer_context->AddEqualCstr(cache_kv_shape[1], x_shape[0]);
    infer_context->AddEqualCstr(cache_kv_shape[2], num_heads);
    infer_context->AddEqualCstr(cache_kv_shape[4], dim_head);
    out_seq_len = out_seq_len + cache_kv_shape[3];
    // [3, batch_size, num_head, cache_seq_len + seq_len, head_size]
    if (paddle::dialect::details::IsFakeValue(op->result(18))) {
      infer_context->SetSymbolForValueByStaticShape(op->result(18));
    } else {
      infer_context->SetShapeOrDataForValue(
          op->result(18),
          symbol::ShapeOrDataDimExprs{
              symbol::TensorShapeOrDataDimExprs({cache_kv_shape[0],
                                                 cache_kv_shape[1],
                                                 cache_kv_shape[2],
                                                 out_seq_len,
                                                 cache_kv_shape[4]})});
    }
  }
  // [batch, num_head, seq_len, out_seq_len]
  infer_context->SetShapeOrDataForValue(
      op->result(6),
      symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(
          {x_shape[0], num_heads, x_shape[1], out_seq_len})});
  if (!src_mask_shape_or_data.isa<symbol::NullShapeOrDataDimExpr>()) {
    infer_context->SetShapeOrDataForValue(
        op->result(11),
        symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(
            {x_shape[0], num_heads, x_shape[1], out_seq_len})});
  } else {
    // The following code used to set unoptional output value.
    // Now it's result related to the infermeta.
    infer_context->SetSymbolForValueByStaticShape(op->result(11));
  }
  // the same as QKOut's shape
  infer_context->SetShapeOrDataForValue(
      op->result(10),
      symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(
          {x_shape[0], num_heads, x_shape[1], out_seq_len})});
  if (!is_test) {
    infer_context->SetShapeOrDataForValue(
        op->result(9),
        symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(
            {x_shape[0], num_heads, x_shape[1], out_seq_len})});
  } else {
    // The following code used to set unoptional output value.
    // Now it's result related to the infermeta.
    infer_context->SetSymbolForValueByStaticShape(op->result(9));
  }

  infer_context->SetShapeOrDataForValue(
      op->result(8),
      symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(
          {x_shape[0], num_heads, x_shape[1], out_seq_len})});

  // [batch_size, num_heads, seq_len, head_dim]
  infer_context->SetShapeOrDataForValue(
      op->result(7),
      symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(
          {x_shape[0], num_heads, x_shape[1], dim_head})});

  // [batch_size, seq_len, number of heads*head size]
  infer_context->SetShapeOrDataForValue(
      op->result(12),
      symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(
          {x_shape[0], x_shape[1], num_heads, dim_head})});

  infer_context->SetShapeOrDataForValue(
      op->result(13),
      symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(x_shape)});

  if (!is_test) {
    infer_context->SetShapeOrDataForValue(
        op->result(14),
        symbol::ShapeOrDataDimExprs{
            symbol::TensorShapeOrDataDimExprs(x_shape)});
  } else {
    // The following code used to set unoptional output value.
    // Now it's result related to the infermeta.
    infer_context->SetSymbolForValueByStaticShape(op->result(14));
  }

  infer_context->SetShapeOrDataForValue(
      op->result(19),
      symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(x_shape)});
  return true;
}

bool FlashAttnVarlenQkvpackedOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &qkv_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const std::vector<symbol::DimExpr> &qkv_shape = qkv_shape_or_data.shape();

  auto round_multiple = [](symbol::DimExpr x) {
    auto m = symbol::DimExpr{128};
    auto m_minus_one = symbol::DimExpr{127};
    return (x + m_minus_one) / m * m;
  };

  size_t rank = qkv_shape.size();
  PADDLE_ENFORCE_EQ((rank == 4UL || rank == 5UL),
                    true,
                    common::errors::InvalidArgument(
                        "qkv dims must be 4(unpadded) or 5(padded batch)"));
  std::vector<symbol::DimExpr> out_dims;
  std::vector<symbol::DimExpr> softmax_lse_shape;
  std::vector<symbol::DimExpr> softmax_shape;
  if (rank == 4UL) {
    // qkv [total_*,nheads/nheads_k+2,nheads_k,headdim]
    out_dims = {qkv_shape[0], (qkv_shape[1] - 2) * qkv_shape[2], qkv_shape[3]};
    softmax_shape = {qkv_shape[0],
                     (qkv_shape[1] - 2) * qkv_shape[2],
                     infer_context->GetNextSymName(),
                     infer_context->GetNextSymName()};
    softmax_lse_shape = {qkv_shape[0],
                         (qkv_shape[1] - 2) * qkv_shape[2],
                         infer_context->GetNextSymName()};
  } else if (rank == 5UL) {
    // qkv [batchsize,seqlen,nheads/nheads_k+2,nheads_k,headdim]
    out_dims = {qkv_shape[0],
                qkv_shape[1],
                (qkv_shape[2] - 2) * qkv_shape[3],
                qkv_shape[4]};
    softmax_shape = {qkv_shape[0],
                     (qkv_shape[2] - 2) * qkv_shape[3],
                     round_multiple(qkv_shape[1]),
                     infer_context->GetNextSymName()};
    softmax_lse_shape = {qkv_shape[0],
                         (qkv_shape[2] - 2) * qkv_shape[3],
                         round_multiple(qkv_shape[1])};
  }

  // Set output tensor shapes
  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(out_dims)});

  infer_context->SetShapeOrDataForValue(
      op->result(1),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs({softmax_shape})});

  infer_context->SetShapeOrDataForValue(
      op->result(2),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(softmax_lse_shape)});

  if (!paddle::dialect::details::IsFakeValue(op->result(3))) {
    std::vector<symbol::DimExpr> seed_offset_dims = {symbol::DimExpr(2)};
    infer_context->SetShapeOrDataForValue(
        op->result(3),
        symbol::ShapeOrDataDimExprs{
            symbol::TensorShapeOrDataDimExprs(seed_offset_dims)});
  }
  return true;
}

bool FlashAttnQkvpackedOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  return FlashAttnVarlenQkvpackedOpInferSymbolicShape(op, infer_context);
}

// bool FlashAttnUnpaddedOpInferSymbolicShape(pir::Operation *op,
//                                            pir::InferSymbolicShapeContext
//                                            *infer_context) {
//   // pass
//   return true;
// }

bool FlashmaskAttentionOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const symbol::ShapeOrDataDimExprs &q =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const symbol::ShapeOrDataDimExprs &k =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));
  const symbol::ShapeOrDataDimExprs &v =
      infer_context->GetShapeOrDataForValue(op->operand_source(2));

  PADDLE_ENFORCE_EQ(q.shape().size(),
                    4,
                    common::errors::InvalidArgument(
                        "flash_attn receive input with dim "
                        "[batch_size, seq_len, num_heads, head_dim]"));

  infer_context->AddEqualCstr(q.shape()[0], k.shape()[0]);
  infer_context->AddEqualCstr(q.shape()[0], v.shape()[0]);
  infer_context->AddEqualCstr(k.shape()[1], v.shape()[1]);

  if (op->operand_source(3)) {
    const std::vector<symbol::DimExpr> &startend_row_indices =
        infer_context->GetShapeOrDataForValue(op->operand_source(4)).shape();
    PADDLE_ENFORCE_EQ(
        startend_row_indices.size(),
        4,
        common::errors::InvalidArgument(
            "flashmask_attention receive startend_row_indices with dim "
            "[batch_size, num_heads,seq_len, mask_bounds]"));
  }
  std::vector<symbol::DimExpr> out_shape = q.shape();

  out_shape.back() = v.shape().back();

  infer_context->SetShapeOrDataForValue(
      op->result(0), symbol::TensorShapeOrDataDimExprs(out_shape));

  // GPU has round for seqlen, but XPU has not. Here we align with the GPU
  // version.
  auto round_multiple = [](symbol::DimExpr x) {
    auto m = symbol::DimExpr{128};
    auto m_minus_one = symbol::DimExpr{127};
    return (x + m_minus_one) / m * m;
  };
  auto batch_size_expr = q.shape()[0];
  auto num_heads_expr = q.shape()[2];
  auto seqlen_q_rounded_expr = round_multiple(q.shape()[1]);
  auto seqlen_k_rounded_expr = round_multiple(k.shape()[1]);

  if (op->result(1)) {
    std::vector<symbol::DimExpr> softmax_shape{batch_size_expr,
                                               num_heads_expr,
                                               seqlen_q_rounded_expr,
                                               seqlen_k_rounded_expr};
    infer_context->SetShapeOrDataForValue(
        op->result(1), symbol::TensorShapeOrDataDimExprs(softmax_shape));
  }
  if (op->result(2)) {
    std::vector<symbol::DimExpr> softmax_lse_shape{
        batch_size_expr, num_heads_expr, seqlen_q_rounded_expr};
    infer_context->SetShapeOrDataForValue(
        op->result(2), symbol::TensorShapeOrDataDimExprs(softmax_lse_shape));
  }
  if (op->result(3)) {
    std::vector<symbol::DimExpr> seed_offset_shape{symbol::DimExpr{2}};
    infer_context->SetShapeOrDataForValue(
        op->result(3), symbol::TensorShapeOrDataDimExprs(out_shape));
  }
  return true;
}
bool FusedBatchNormActOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  return BatchNormOpInferSymbolicShape(op, infer_context);
}

bool FusedBatchNormAct_OpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  return BatchNormOpInferSymbolicShape(op, infer_context);
}

bool FusedBnAddActivationOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  return BatchNormOpInferSymbolicShape(op, infer_context);
}

bool FusedBnAddActivation_OpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  return BatchNormOpInferSymbolicShape(op, infer_context);
}

bool FusedGemmEpilogueOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const auto &x_dims = x_shape_or_data.shape();

  const auto &y_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));
  const auto &y_dims = y_shape_or_data.shape();

  const auto &bias_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));
  const auto &bias_dims = bias_shape_or_data.shape();

  size_t x_rank = x_dims.size();
  size_t y_rank = y_dims.size();

  std::vector<symbol::DimExpr> out_shape;
  out_shape.reserve(x_rank);

  for (size_t i = 0; i + 2 < x_rank; ++i) {
    out_shape.emplace_back(x_dims[i]);
  }

  bool transpose_x_attr = GetBoolAttr(op, "trans_x");
  bool transpose_y_attr = GetBoolAttr(op, "trans_y");

  symbol::DimExpr out_M =
      transpose_x_attr ? x_dims[x_rank - 1] : x_dims[x_rank - 2];
  symbol::DimExpr out_N =
      transpose_y_attr ? y_dims[y_rank - 2] : y_dims[y_rank - 1];

  out_shape.emplace_back(out_M);
  out_shape.emplace_back(out_N);

  symbol::DimExpr x_K =
      transpose_x_attr ? x_dims[x_rank - 2] : x_dims[x_rank - 1];
  symbol::DimExpr y_K =
      transpose_y_attr ? y_dims[y_rank - 1] : y_dims[y_rank - 2];

  infer_context->AddEqualCstr(x_K, y_K);
  // bias_dims[0] equal to out_N
  infer_context->AddEqualCstr(out_N, bias_dims[0]);

  infer_context->SetShapeOrDataForValue(op->result(0),
                                        ShapeOrData{TensorExprs(out_shape)});

  // process reserve space
  if (!paddle::dialect::details::IsFakeValue(op->result(1))) {
    infer_context->SetShapeOrDataForValue(op->result(1),
                                          ShapeOrData{TensorExprs(out_shape)});
  } else {
    infer_context->SetSymbolForValueByStaticShape(op->result(1));
  }

  return true;
}

bool FusedMultiTransformerOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const std::vector<symbol::DimExpr> &x_shape = x_shape_or_data.shape();

  const auto &qkv_weight_data_list =
      infer_context->GetShapeOrDataForValue(op->operand_source(3))
          .dyn_cast<symbol::TensorListShapeOrDataDimExprs>();

  const std::vector<symbol::DimExpr> &y_shape =
      qkv_weight_data_list.at(0).shape();

  PADDLE_ENFORCE_EQ(
      x_shape.size(),
      3,
      common::errors::InvalidArgument("The dimensions of x must be 3"
                                      "(batch_size, seq_len, dim_embed),"
                                      "but received dimensions of"
                                      "Input is [%d]",
                                      x_shape.size()));
  PADDLE_ENFORCE_EQ(
      y_shape.size(),
      4,
      common::errors::InvalidArgument("The dimensions of qkv_weight must be 4"
                                      "(3, num_head, dim_head, dim_embed),"
                                      "but received dimensions of"
                                      "Input is [%d]",
                                      y_shape.size()));

  bool trans_qkvw = op->attribute<pir::BoolAttribute>("trans_qkvw").data();

  if (trans_qkvw) {
    infer_context->AddEqualCstr(x_shape[2], y_shape[3]);
  } else {
    infer_context->AddEqualCstr(x_shape[2], y_shape[0]);
  }
  const auto &cache_kv_data_list =
      infer_context->GetShapeOrDataForValue(op->operand_source(5))
          .dyn_cast<symbol::TensorListShapeOrDataDimExprs>();

  if (cache_kv_data_list.size() > 0) {
    const std::vector<symbol::DimExpr> &c_shape =
        cache_kv_data_list.at(0).shape();

    PADDLE_ENFORCE_EQ(
        c_shape.size(),
        5,
        common::errors::InvalidArgument(
            "The CacheKV must be 5 dims, but got %d", c_shape.size()));
    infer_context->AddEqualCstr(c_shape[0], symbol::DimExpr{2});
    infer_context->AddEqualCstr(c_shape[1], x_shape[0]);

    if (trans_qkvw) {
      infer_context->AddEqualCstr(c_shape[2], y_shape[1]);
      infer_context->AddEqualCstr(c_shape[4], y_shape[2]);
    } else {
      infer_context->AddEqualCstr(c_shape[2], y_shape[2]);
      infer_context->AddEqualCstr(c_shape[4], y_shape[3]);
    }
  }

  infer_context->SetShapeOrDataForValue(
      op->result(0), symbol::TensorShapeOrDataDimExprs(x_shape));
  return true;
}

bool GenerateProposalsOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  symbol::DimExpr out_unknown = infer_context->GetNextSymName();
  std::vector<symbol::DimExpr> rpn_rois_shape = {out_unknown,
                                                 symbol::DimExpr(4)};
  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(rpn_rois_shape)});

  std::vector<symbol::DimExpr> rpn_roi_probs_shape = {out_unknown,
                                                      symbol::DimExpr(1)};
  infer_context->SetShapeOrDataForValue(
      op->result(1),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(rpn_roi_probs_shape)});

  // NOTE(gongshaotian): In the training task, the isFakeValue() interface can
  // be used to determine whether rpn_rois_num needs to be output. However, in
  // the inference task of executing the old model that has already been
  // trained, since the InferMeta() function will no longer be executed, it is
  // impossible to determine whether rpn_rois_num is a Fake value through
  // isFakeValue(), so it is necessary to judge based on the dimension of
  // DenseTensor.
  if (paddle::dialect::details::IsFakeValue(op->result(2)) ||
      op->result(2)
              .type()
              .dyn_cast<paddle::dialect::DenseTensorType>()
              .dims()
              .size() == 0) {
    infer_context->SetSymbolForValueByStaticShape(op->result(2));
  } else {
    const std::vector<symbol::DimExpr> &score_shape =
        infer_context->GetShapeOrDataForValue(op->operand_source(0)).shape();
    std::vector<symbol::DimExpr> rpn_rois_num_shape = {score_shape[0]};

    infer_context->SetShapeOrDataForValue(
        op->result(2),
        symbol::ShapeOrDataDimExprs{
            symbol::TensorShapeOrDataDimExprs(rpn_rois_num_shape)});
  }

  return true;
}

bool GraphKhopSamplerOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const symbol::ShapeOrDataDimExprs &row_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const symbol::ShapeOrDataDimExprs &col_ptr_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));
  const symbol::ShapeOrDataDimExprs &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(2));
  const symbol::ShapeOrDataDimExprs &eids_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(3));

  const std::vector<symbol::DimExpr> &row_shape = row_shape_or_data.shape();
  const std::vector<symbol::DimExpr> &col_ptr_shape =
      col_ptr_shape_or_data.shape();
  const std::vector<symbol::DimExpr> &x_shape = x_shape_or_data.shape();
  const std::vector<symbol::DimExpr> &eids_shape = eids_shape_or_data.shape();

  auto GKSShapeCheck = [&](const std::vector<symbol::DimExpr> &shape,
                           const std::string &tensor_name) {
    if (shape.size() == 2)
      infer_context->AddEqualCstr(shape[1], symbol::DimExpr(1));
    else
      PADDLE_ENFORCE_EQ(
          shape.size(),
          1,
          common::errors::InvalidArgument(
              "The %s should be 1D, when it is not 2D, but we get %d",
              tensor_name,
              shape.size()));
  };

  GKSShapeCheck(row_shape, "row");
  GKSShapeCheck(col_ptr_shape, "col_ptr");
  GKSShapeCheck(x_shape, "x");

  std::vector<int> sample_sizes =
      paddle::dialect::details::GetVectorAttr<int>(op, "sample_sizes");
  PADDLE_ENFORCE_EQ(
      !sample_sizes.empty(),
      true,
      common::errors::InvalidArgument(
          "The parameter 'sample_sizes' in GraphSampleOp must be set. "
          "But received 'sample_sizes' is empty."));

  bool return_eids = op->attribute<pir::BoolAttribute>("return_eids").data();
  if (return_eids) {
    GKSShapeCheck(eids_shape, "eids");
    symbol::DimExpr out_unknown_4 = infer_context->GetNextSymName();
    infer_context->SetShapeOrDataForValue(
        op->result(4),
        symbol::ShapeOrDataDimExprs{
            symbol::TensorShapeOrDataDimExprs({out_unknown_4})});
  } else {
    infer_context->SetSymbolForValueByStaticShape(op->result(4));
  }

  symbol::DimExpr out_unknown_0_1 = infer_context->GetNextSymName();
  symbol::DimExpr out_unknown_2 = infer_context->GetNextSymName();
  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs({out_unknown_0_1, 1})});
  infer_context->SetShapeOrDataForValue(
      op->result(1),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs({out_unknown_0_1, 1})});
  infer_context->SetShapeOrDataForValue(
      op->result(2),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs({out_unknown_2})});
  infer_context->SetShapeOrDataForValue(
      op->result(3),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs({x_shape[0]})});

  return true;
}

bool ReindexGraphOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const symbol::ShapeOrDataDimExprs &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const symbol::ShapeOrDataDimExprs &neighbors_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));
  const symbol::ShapeOrDataDimExprs &count_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(2));
  const symbol::ShapeOrDataDimExprs &hashtable_value_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(3));
  const symbol::ShapeOrDataDimExprs &hashtable_index_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(4));

  const std::vector<symbol::DimExpr> &x_shape = x_shape_or_data.shape();
  const std::vector<symbol::DimExpr> &neighbors_shape =
      neighbors_shape_or_data.shape();
  const std::vector<symbol::DimExpr> &count_shape = count_shape_or_data.shape();
  const std::vector<symbol::DimExpr> &hashtable_value_shape =
      hashtable_value_shape_or_data.shape();
  const std::vector<symbol::DimExpr> &hashtable_index_shape =
      hashtable_index_shape_or_data.shape();

  auto GraphReindexShapeCheck = [&](const std::vector<symbol::DimExpr> &shape,
                                    const std::string &tensor_name) {
    if (shape.size() == 2) {
      infer_context->AddEqualCstr(shape[1], symbol::DimExpr{1});
    } else {
      PADDLE_ENFORCE_EQ(
          shape.size(),
          1,
          common::errors::InvalidArgument(
              "The %s should be 1D, when it is not 2D, but we get %d",
              tensor_name,
              shape.size()));
    }
  };

  GraphReindexShapeCheck(x_shape, "X");
  GraphReindexShapeCheck(neighbors_shape, "Neighbors");
  GraphReindexShapeCheck(count_shape, "Count");

  if (!paddle::dialect::details::IsFakeValue(op->operand_source(3)) &&
      !paddle::dialect::details::IsFakeValue(op->operand_source(4))) {
    GraphReindexShapeCheck(hashtable_value_shape, "HashTable_Value");
    GraphReindexShapeCheck(hashtable_index_shape, "HashTable_Index");
  }

  VLOG(3) << "ReindexGraphOpInferSymbolicShape";

  symbol::DimExpr unique_dim_sym = infer_context->GetNextSymName();
  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs({neighbors_shape[0]})});
  infer_context->SetShapeOrDataForValue(
      op->result(1),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs({neighbors_shape[0]})});
  infer_context->SetShapeOrDataForValue(
      op->result(2),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs({unique_dim_sym})});

  return true;
}

bool GraphSampleNeighborsOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const symbol::ShapeOrDataDimExprs &row_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const symbol::ShapeOrDataDimExprs &col_ptr_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));
  const symbol::ShapeOrDataDimExprs &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(2));
  const symbol::ShapeOrDataDimExprs &eids_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(3));
  const symbol::ShapeOrDataDimExprs &perm_buffer_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(4));

  const std::vector<symbol::DimExpr> &row_shape = row_shape_or_data.shape();
  const std::vector<symbol::DimExpr> &col_ptr_shape =
      col_ptr_shape_or_data.shape();
  const std::vector<symbol::DimExpr> &x_shape = x_shape_or_data.shape();
  const std::vector<symbol::DimExpr> &eids_shape = eids_shape_or_data.shape();
  const std::vector<symbol::DimExpr> &perm_buffer_shape =
      perm_buffer_shape_or_data.shape();

  auto GSNShapeCheck = [&](const std::vector<symbol::DimExpr> &shape,
                           const std::string &tensor_name) {
    if (shape.size() == 2)
      infer_context->AddEqualCstr(shape[1], symbol::DimExpr{1});
    else
      PADDLE_ENFORCE_EQ(
          shape.size(),
          1,
          common::errors::InvalidArgument(
              "The %s should be 1D, when it is not 2D, but we get %d",
              tensor_name,
              shape.size()));
  };

  GSNShapeCheck(row_shape, "Row");
  GSNShapeCheck(col_ptr_shape, "Col_Ptr");
  GSNShapeCheck(x_shape, "X");

  bool return_eids = op->attribute<pir::BoolAttribute>("return_eids").data();
  bool flag_perm_buffer =
      op->attribute<pir::BoolAttribute>("flag_perm_buffer").data();

  if (return_eids) {
    GSNShapeCheck(eids_shape, "Eids");
    symbol::DimExpr out_unknown_2 = infer_context->GetNextSymName();
    infer_context->SetShapeOrDataForValue(
        op->result(2),
        symbol::ShapeOrDataDimExprs{
            symbol::TensorShapeOrDataDimExprs({out_unknown_2})});
  } else {
    infer_context->SetSymbolForValueByStaticShape(op->result(2));
  }

  if (flag_perm_buffer) {
    GSNShapeCheck(perm_buffer_shape, "Perm_Buffer");
  }

  symbol::DimExpr out_unknown_0 = infer_context->GetNextSymName();
  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs({out_unknown_0})});
  infer_context->SetShapeOrDataForValue(
      op->result(1),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs({x_shape[0]})});

  return true;
}

bool GruOpInferSymbolicShape(pir::Operation *op,
                             pir::InferSymbolicShapeContext *infer_context) {
  const symbol::ShapeOrDataDimExprs &input_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const symbol::ShapeOrDataDimExprs &weight_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(2));
  const symbol::ShapeOrDataDimExprs &hidden_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->result(3));

  std::vector<symbol::DimExpr> input_shape = input_shape_or_data.shape();
  std::vector<symbol::DimExpr> weight_shape = weight_shape_or_data.shape();
  std::vector<symbol::DimExpr> hidden_shape = hidden_shape_or_data.shape();

  bool is_test = op->attribute<pir::BoolAttribute>("is_test").data();

  symbol::DimExpr input_size = input_shape[1];
  symbol::DimExpr frame_size = weight_shape[0];

  // Check if input_size is 3 times frame_size
  infer_context->AddEqualCstr(input_size, frame_size * 3);

  // Check if weight matrix has size [frame_size, frame_size * 3]
  infer_context->AddEqualCstr(weight_shape[1], frame_size * 3);

  if (op->operand(1)) {  // Check if H0 is given
    const symbol::ShapeOrDataDimExprs &h0_shape_or_data =
        infer_context->GetShapeOrDataForValue(op->operand_source(1));
    std::vector<symbol::DimExpr> h0_shape = h0_shape_or_data.shape();
    infer_context->AddEqualCstr(h0_shape[1], frame_size);
  }

  if (op->operand(3)) {  // Check if Bias is given
    const symbol::ShapeOrDataDimExprs &bias_shape_or_data =
        infer_context->GetShapeOrDataForValue(op->operand_source(3));
    std::vector<symbol::DimExpr> bias_shape = bias_shape_or_data.shape();
    infer_context->AddEqualCstr(bias_shape[0], 1);
    infer_context->AddEqualCstr(bias_shape[1], frame_size * 3);
  }

  if (is_test) {
    symbol::TensorShapeOrDataDimExprs batch_gate_shape(input_shape);
    infer_context->SetShapeOrDataForValue(op->result(0), batch_gate_shape);

    symbol::TensorShapeOrDataDimExprs batch_reset_hidden_prev_shape(
        {hidden_shape});
    infer_context->SetShapeOrDataForValue(op->result(1),
                                          batch_reset_hidden_prev_shape);

    symbol::TensorShapeOrDataDimExprs batch_hidden_shape({hidden_shape});
    infer_context->SetShapeOrDataForValue(op->result(2), batch_hidden_shape);
  } else {
    symbol::TensorShapeOrDataDimExprs batch_gate_shape(input_shape);
    infer_context->SetShapeOrDataForValue(op->result(0), batch_gate_shape);

    symbol::TensorShapeOrDataDimExprs batch_reset_hidden_prev_shape(
        {input_shape[0], frame_size});
    infer_context->SetShapeOrDataForValue(op->result(1),
                                          batch_reset_hidden_prev_shape);

    symbol::TensorShapeOrDataDimExprs batch_hidden_shape(
        {input_shape[0], frame_size});
    infer_context->SetShapeOrDataForValue(op->result(2), batch_hidden_shape);
  }

  symbol::TensorShapeOrDataDimExprs hidden_shape_output(
      {input_shape[0], frame_size});
  infer_context->SetShapeOrDataForValue(op->result(3), hidden_shape_output);

  return true;
}

bool GruUnitOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  // Get symbolic shapes of the input tensors
  const symbol::ShapeOrDataDimExprs &input_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const symbol::ShapeOrDataDimExprs &hidden_prev_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));
  const symbol::ShapeOrDataDimExprs &weight_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(2));
  const symbol::ShapeOrDataDimExprs &bias_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(3));

  auto input_shape = input_shape_or_data.shape();
  auto hidden_prev_shape = hidden_prev_shape_or_data.shape();
  auto weight_shape = weight_shape_or_data.shape();

  // Validate input dimensions
  symbol::DimExpr batch_size = input_shape[0];
  symbol::DimExpr input_size = input_shape[1];
  symbol::DimExpr frame_size = hidden_prev_shape[1];
  symbol::DimExpr weight_height = weight_shape[0];
  symbol::DimExpr weight_width = weight_shape[1];

  // Enforce dimension constraints using symbolic dimensions
  infer_context->AddEqualCstr(input_size, frame_size * 3);
  infer_context->AddEqualCstr(weight_height, frame_size);
  infer_context->AddEqualCstr(weight_width, frame_size * 3);

  // If bias is used, check its dimensions
  if (!bias_shape_or_data.isa<symbol::NullShapeOrDataDimExpr>()) {
    auto bias_shape = bias_shape_or_data.shape();
    symbol::DimExpr bias_height = bias_shape[0];
    symbol::DimExpr bias_width = bias_shape[1];
    infer_context->AddEqualCstr(bias_height, 1);
    infer_context->AddEqualCstr(bias_width, frame_size * 3);
  }

  // Set output dimensions
  std::vector<symbol::DimExpr> gate_dims = {batch_size, frame_size * 3};
  std::vector<symbol::DimExpr> reset_hidden_prev_dims = {batch_size,
                                                         frame_size};
  std::vector<symbol::DimExpr> hidden_dims = {batch_size, frame_size};

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(gate_dims)});
  infer_context->SetShapeOrDataForValue(
      op->result(1),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(reset_hidden_prev_dims)});
  infer_context->SetShapeOrDataForValue(
      op->result(2),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(hidden_dims)});

  return true;
}

bool GroupNormOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const symbol::ShapeOrDataDimExprs &x_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));

  const auto &scale_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));
  const auto &bias_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(2));

  infer_context->SetShapeOrDataForValue(op->result(0), x_shape);

  int64_t channel_idx;
  std::string data_format =
      op->attribute<pir::StrAttribute>("data_format").AsString();
  if (data_format == "NHWC") {
    channel_idx = x_shape.shape().size() - 1;
  } else if (data_format == "NCHW") {
    channel_idx = 1;
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "GroupNorm only support NHWC and NCHW data format"));
  }

  symbol::DimExpr channel_dim = x_shape.shape()[channel_idx];

  if (!scale_shape_or_data.isa<symbol::NullShapeOrDataDimExpr>()) {
    std::vector<symbol::DimExpr> scale_dims = scale_shape_or_data.shape();
    infer_context->AddEqualCstr(scale_dims[0], channel_dim);
  }
  if (!bias_shape_or_data.isa<symbol::NullShapeOrDataDimExpr>()) {
    std::vector<symbol::DimExpr> bias_dims = bias_shape_or_data.shape();
    infer_context->AddEqualCstr(bias_dims[0], channel_dim);
  }

  const symbol::DimExpr &batch_size = x_shape.shape()[0];
  int groups = op->attribute<pir::Int32Attribute>("groups").data();
  symbol::TensorShapeOrDataDimExprs mean_shape(
      std::vector<symbol::DimExpr>{batch_size, groups});
  if (op->result(1)) {
    infer_context->SetShapeOrDataForValue(op->result(1), mean_shape);
  }
  if (op->result(2)) {
    infer_context->SetShapeOrDataForValue(op->result(2), mean_shape);
  }
  return true;
}

bool InstanceNormOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const std::vector<symbol::DimExpr> &x_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(0)).shape();
  PADDLE_ENFORCE_GE(
      x_shape.size(),
      2,
      common::errors::InvalidArgument(
          "ShapeError: the dimension of input X must "
          "greater than or equal to 2. But received: the shape of input "
          "X = [%s], the dimension of input X =[%d]",
          x_shape,
          x_shape.size()));
  PADDLE_ENFORCE_LE(
      x_shape.size(),
      5,
      common::errors::InvalidArgument(
          "ShapeError: the dimension of input X must "
          "smaller than or equal to 5, But received: the shape of input "
          "X = [%s], the dimension of input X = [%d]",
          x_shape,
          x_shape.size()));
  symbol::DimExpr N = x_shape[0];
  symbol::DimExpr C = x_shape[1];
  symbol::DimExpr NxC = N * C;
  if (op->operand_source(1)) {
    const std::vector<symbol::DimExpr> &scale_shape =
        infer_context->GetShapeOrDataForValue(op->operand_source(1)).shape();
    PADDLE_ENFORCE_EQ(
        scale_shape.size(),
        1UL,
        common::errors::InvalidArgument(
            "ShapeError: the dimension of scale must equal to 1."
            "But received: the shape of scale is [%s], the dimension "
            "of scale is [%d]",
            scale_shape,
            scale_shape.size()));
    infer_context->AddEqualCstr(C, scale_shape[0]);
  }
  if (op->operand(2)) {
    const std::vector<symbol::DimExpr> &bias_shape =
        infer_context->GetShapeOrDataForValue(op->operand_source(2)).shape();
    PADDLE_ENFORCE_EQ(
        bias_shape.size(),
        1UL,
        common::errors::InvalidArgument(
            "ShapeError: the dimension of bias must equal to 1."
            "But received: the shape of bias is [%s],the dimension "
            "of bias is [%d]",
            bias_shape,
            bias_shape.size()));
    infer_context->AddEqualCstr(C, bias_shape[0]);
  }
  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(x_shape)});
  if (paddle::dialect::details::IsFakeValue(op->result(1))) {
    infer_context->SetSymbolForValueByStaticShape(op->result(1));
  } else {
    infer_context->SetShapeOrDataForValue(
        op->result(1),
        symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs({NxC})});
  }
  if (paddle::dialect::details::IsFakeValue(op->result(2))) {
    infer_context->SetSymbolForValueByStaticShape(op->result(2));
  } else {
    infer_context->SetShapeOrDataForValue(
        op->result(2),
        symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs({NxC})});
  }
  return true;
}

bool LerpOpInferSymbolicShape(pir::Operation *op,
                              pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const auto &y_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));
  const auto &w_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(2));
  std::vector<symbol::DimExpr> x_shape = x_shape_or_data.shape();
  std::vector<symbol::DimExpr> y_shape = y_shape_or_data.shape();
  std::vector<symbol::DimExpr> w_shape = w_shape_or_data.shape();
  int x_ndims = x_shape.size();
  int y_ndims = y_shape.size();
  int w_ndims = w_shape.size();
  std::vector<symbol::DimExpr> out1_shape;
  std::vector<symbol::DimExpr> out2_shape;
  int diffxy = x_ndims - y_ndims;
  if (diffxy > 0) {
    for (int i = 0; i < diffxy; ++i) {
      y_shape.emplace(y_shape.begin(), 1);
    }
  } else {
    for (int i = 0; i < -diffxy; ++i) {
      x_shape.emplace(x_shape.begin(), 1);
    }
  }
  symbol::DimExprBuilder builder;
  for (size_t i = 0; i < x_shape.size(); ++i) {
    out1_shape.emplace_back(builder.Broadcast(x_shape[i], y_shape[i]));
    infer_context->AddBroadcastableCstr(x_shape[i], y_shape[i]);
  }
  int out1_ndims = out1_shape.size();
  int diffxyw = w_ndims - out1_ndims;
  if (diffxyw > 0) {
    for (int i = 0; i < diffxyw; ++i) {
      out1_shape.emplace(out1_shape.begin(), 1);
    }
  } else {
    for (int i = 0; i < -diffxyw; ++i) {
      w_shape.emplace(w_shape.begin(), 1);
    }
  }
  for (size_t i = 0; i < w_shape.size(); ++i) {
    out2_shape.emplace_back(builder.Broadcast(w_shape[i], out1_shape[i]));
    infer_context->AddBroadcastableCstr(w_shape[i], out1_shape[i]);
  }
  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(out2_shape)});
  return true;
}

bool Lerp_OpInferSymbolicShape(pir::Operation *op,
                               pir::InferSymbolicShapeContext *infer_context) {
  return LerpOpInferSymbolicShape(op, infer_context);
}

bool LayerNormOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  // Get the shapes of input tensors
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const auto &scale_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));
  const auto &bias_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(2));

  std::vector<symbol::DimExpr> x_dims = x_shape_or_data.shape();
  int begin_norm_axis =
      op->attribute<pir::Int32Attribute>("begin_norm_axis").data();

  // Flatten x_dims to 2D and get dim[1]
  symbol::DimExpr matrix_dim_1 = x_dims[begin_norm_axis];
  for (std::size_t i = begin_norm_axis + 1; i < x_dims.size(); ++i) {
    matrix_dim_1 = matrix_dim_1 * x_dims[i];
  }

  if (!scale_shape_or_data.isa<symbol::NullShapeOrDataDimExpr>()) {
    std::vector<symbol::DimExpr> scale_dims = scale_shape_or_data.shape();
    infer_context->AddEqualCstr(scale_dims[0], matrix_dim_1);
  }
  if (!bias_shape_or_data.isa<symbol::NullShapeOrDataDimExpr>()) {
    std::vector<symbol::DimExpr> bias_dims = bias_shape_or_data.shape();
    infer_context->AddEqualCstr(bias_dims[0], matrix_dim_1);
  }

  // Set output shapes
  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(x_dims)});

  // Set mean and variance shapes
  std::vector<symbol::DimExpr> before_norm_dims(
      x_dims.begin(), x_dims.begin() + begin_norm_axis);
  infer_context->SetShapeOrDataForValue(
      op->result(1),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(before_norm_dims)});
  infer_context->SetShapeOrDataForValue(
      op->result(2),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(before_norm_dims)});

  return true;
}

bool LinspaceOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &num_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(2));
  const auto step = [&] {
    symbol::DimExpr expr;
    if (num_shape_or_data.data().has_value()) {
      expr = num_shape_or_data.data().value()[0];
    } else {
      expr = num_shape_or_data.shape()[0];
    }
    return expr;
  }();
  const symbol::ShapeOrDataDimExprs &shape_data = [&] {
    std::vector<symbol::DimExpr> out_dims{step};
    return symbol::ShapeOrDataDimExprs{
        symbol::TensorShapeOrDataDimExprs(out_dims)};
  }();
  infer_context->SetShapeOrDataForValue(op->result(0), shape_data);
  return true;
}

bool LinearInterpOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  return BicubicInterpOpInferSymbolicShape(op, infer_context);
}

bool LlmInt8LinearOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const std::vector<symbol::DimExpr> &x_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(0)).shape();
  const std::vector<symbol::DimExpr> &weight_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(1)).shape();
  const std::vector<symbol::DimExpr> &weight_scale_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(3)).shape();
  size_t x_rank = x_shape.size();

  PADDLE_ENFORCE_EQ(weight_shape.size(),
                    2UL,
                    common::errors::InvalidArgument(
                        "The input(weight) must be a 2D Tensor."));
  infer_context->AddEqualCstr(weight_shape[1], x_shape[x_rank - 1]);
  if (weight_shape[0].isa<int64_t>()) {
    int64_t w_shape_0 = weight_shape[0].Get<int64_t>();
    PADDLE_ENFORCE_EQ(
        w_shape_0 % 16,
        0,
        common::errors::InvalidArgument(
            "The first dimension of input must be divisible by 16, but got[%d]",
            w_shape_0));
  }
  if (weight_shape[1].isa<int64_t>()) {
    int64_t w_shape_1 = weight_shape[1].Get<int64_t>();
    PADDLE_ENFORCE_EQ(
        w_shape_1 % 16,
        0,
        common::errors::InvalidArgument("The second dimension of input must be "
                                        "divisible by 16, but got[%d]",
                                        w_shape_1));
  }

  infer_context->AddEqualCstr(weight_scale_shape[0], weight_shape[0]);
  std::vector<symbol::DimExpr> out_shape = x_shape;
  out_shape[x_rank - 1] = weight_shape[0];
  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(out_shape)});
  return true;
}

bool LogspaceOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  return LinspaceOpInferSymbolicShape(op, infer_context);
}

bool NearestInterpOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  return BicubicInterpOpInferSymbolicShape(op, infer_context);
}

bool MatchMatrixTensorOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const auto &y_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));
  const auto &w_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(2));

  const auto &x_shape = x_shape_or_data.shape();
  const auto &y_shape = y_shape_or_data.shape();
  const auto &w_shape = w_shape_or_data.shape();

  PADDLE_ENFORCE_EQ(x_shape.size(),
                    2,
                    common::errors::InvalidArgument(
                        "The dimensions of Input(X) should be equal to 2, "
                        "but received %d.",
                        x_shape.size()));
  PADDLE_ENFORCE_EQ(y_shape.size(),
                    2,
                    common::errors::InvalidArgument(
                        "The dimensions of Input(Y) should be equal to 2, "
                        "but received %d.",
                        y_shape.size()));
  PADDLE_ENFORCE_EQ(w_shape.size(),
                    3,
                    common::errors::InvalidArgument(
                        "The dimensions of Input(W) should be equal to 3, "
                        "but received %d.",
                        w_shape.size()));

  infer_context->AddEqualCstr(w_shape[0], x_shape[1]);
  int dim_t = op->attribute<pir::Int32Attribute>("dim_t").data();
  infer_context->AddEqualCstr(w_shape[1], symbol::DimExpr(dim_t));
  infer_context->AddEqualCstr(w_shape[2], y_shape[1]);

  std::vector<symbol::DimExpr> out_dims = {infer_context->GetNextSymName(),
                                           symbol::DimExpr(1)};
  std::vector<symbol::DimExpr> tmp_dims = {infer_context->GetNextSymName(),
                                           symbol::DimExpr(1)};

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(out_dims)});
  infer_context->SetShapeOrDataForValue(
      op->result(1),
      symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(tmp_dims)});

  return true;
}

bool MaskedMultiheadAttention_OpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const symbol::ShapeOrDataDimExprs &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const symbol::ShapeOrDataDimExprs &cache_kv_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));

  const std::vector<symbol::DimExpr> &x_shape = x_shape_or_data.shape();
  const std::vector<symbol::DimExpr> &cache_kv_shape =
      cache_kv_shape_or_data.shape();

  std::string compute_dtype =
      op->attribute<pir::StrAttribute>("compute_dtype").AsString();

  PADDLE_ENFORCE_EQ(
      cache_kv_shape.size(),
      5,
      common::errors::InvalidArgument("The cache_kv must be 5 dims."));
  infer_context->AddEqualCstr(cache_kv_shape[0], symbol::DimExpr(2));
  // TODO(Luohongzhige, Buaa): add constrain for the num_head and k_num_head

  symbol::DimExpr bsz = x_shape[0];
  symbol::DimExpr dim_head = cache_kv_shape[4];
  symbol::DimExpr k_num_head = cache_kv_shape[2];
  symbol::DimExpr v_num_head = k_num_head;
  symbol::DimExpr num_head =
      (x_shape[x_shape.size() - 1] / dim_head - k_num_head - v_num_head);
  std::vector<symbol::DimExpr> out_shape = {bsz, num_head * dim_head};

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(out_shape)});

  infer_context->SetShapeOrDataForValue(
      op->result(1),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(cache_kv_shape)});

  if (op->operand_source(7) != nullptr) {
    const symbol::ShapeOrDataDimExprs &beam_cache_offset_shape_or_data =
        infer_context->GetShapeOrDataForValue(op->operand_source(7));
    const std::vector<symbol::DimExpr> &beam_cache_offset_shape =
        beam_cache_offset_shape_or_data.shape();
    infer_context->SetShapeOrDataForValue(
        op->result(2),
        symbol::ShapeOrDataDimExprs{
            symbol::TensorShapeOrDataDimExprs(beam_cache_offset_shape)});
  }

  return true;
}

bool MemoryEfficientAttentionOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &q_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(0)).shape();
  const auto &k_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(1)).shape();
  const auto &v_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(2)).shape();
  PADDLE_ENFORCE_EQ(
      q_shape.size(),
      4,
      common::errors::InvalidArgument("Query should be a 4-D tensor"
                                      "But received Query dimension(%d)",
                                      q_shape.size()));
  PADDLE_ENFORCE_EQ(
      k_shape.size(),
      4,
      common::errors::InvalidArgument("Key should be a 4-D tensor"
                                      "But received Key dimension(%d)",
                                      k_shape.size()));
  PADDLE_ENFORCE_EQ(
      v_shape.size(),
      4,
      common::errors::InvalidArgument("Value should be a 4-D tensor"
                                      "But received Value dimension(%d)",
                                      v_shape.size()));

  const auto &query_batch_size = q_shape[0];
  const auto &query_seq_length = q_shape[1];
  const auto &query_num_head = q_shape[2];
  const auto &query_head_size = q_shape[3];

  const auto &key_batch_size = k_shape[0];
  const auto &key_seq_length = k_shape[1];
  const auto &key_num_head = k_shape[2];
  const auto &key_head_size = k_shape[3];

  const auto &value_batch_size = v_shape[0];
  const auto &value_seq_length = v_shape[1];
  const auto &value_num_head = v_shape[2];
  const auto &value_head_size = v_shape[3];

  infer_context->AddEqualCstr(query_batch_size, key_batch_size);
  infer_context->AddEqualCstr(key_batch_size, value_batch_size);

  infer_context->AddEqualCstr(query_num_head, key_num_head);
  infer_context->AddEqualCstr(key_num_head, value_num_head);

  infer_context->AddEqualCstr(query_head_size, key_head_size);

  infer_context->AddEqualCstr(key_seq_length, value_seq_length);

  const std::vector<symbol::DimExpr> out_dims{
      query_batch_size, query_seq_length, query_num_head, value_head_size};
  const std::vector<symbol::DimExpr> logsumexp_dims{query_num_head,
                                                    query_batch_size};
  const std::vector<symbol::DimExpr> seed_and_offset_dims{2};

  infer_context->SetShapeOrDataForValue(
      op->result(0), symbol::TensorShapeOrDataDimExprs(out_dims));
  infer_context->SetShapeOrDataForValue(
      op->result(1), symbol::TensorShapeOrDataDimExprs(logsumexp_dims));
  infer_context->SetShapeOrDataForValue(
      op->result(2), symbol::TensorShapeOrDataDimExprs(seed_and_offset_dims));

  return true;
}

bool NllLossOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const symbol::ShapeOrDataDimExprs &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const std::vector<symbol::DimExpr> &x_shape = x_shape_or_data.shape();
  const symbol::ShapeOrDataDimExprs &label_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));
  const std::vector<symbol::DimExpr> &label_shape = label_shape_or_data.shape();
  PADDLE_ENFORCE_EQ(x_shape.size() == 2 || x_shape.size() == 4,
                    true,
                    common::errors::InvalidArgument(
                        "The tensor rank of Input(X) must be 2 or 4."));
  infer_context->AddEqualCstr(x_shape[0], label_shape[0]);

  if (op->operand_source(2)) {
    const symbol::ShapeOrDataDimExprs &w_shape_or_data =
        infer_context->GetShapeOrDataForValue(op->operand_source(2));
    const std::vector<symbol::DimExpr> &w_shape = w_shape_or_data.shape();
    PADDLE_ENFORCE_EQ(w_shape.size(),
                      1,
                      common::errors::InvalidArgument(
                          "Input(Weight) should be a 1D tensor."));

    infer_context->AddEqualCstr(x_shape[1], w_shape[0]);
  }

  const std::string &reduction =
      op->attribute<pir::StrAttribute>("reduction").AsString();

  std::vector<symbol::DimExpr> out_shape;
  if (x_shape.size() == 2) {
    if (reduction == "none") {
      out_shape = {x_shape[0]};
    } else {
      out_shape = std::vector<symbol::DimExpr>{};
    }
    infer_context->SetShapeOrDataForValue(
        op->result(0),
        symbol::ShapeOrDataDimExprs{
            symbol::TensorShapeOrDataDimExprs(out_shape)});
  } else if (x_shape.size() == 4) {
    PADDLE_ENFORCE_EQ(label_shape.size(),
                      3,
                      common::errors::InvalidArgument(
                          "Expected Input(Label) dimensions=3, received %d.",
                          label_shape.size()));

    infer_context->AddEqualCstr(x_shape[0], label_shape[0]);
    infer_context->AddEqualCstr(x_shape[2], label_shape[1]);
    infer_context->AddEqualCstr(x_shape[3], label_shape[2]);

    if (reduction == "none") {
      out_shape = {x_shape[0], x_shape[2], x_shape[3]};
    } else {
      out_shape = std::vector<symbol::DimExpr>{};
    }
    infer_context->SetShapeOrDataForValue(
        op->result(0),
        symbol::ShapeOrDataDimExprs{
            symbol::TensorShapeOrDataDimExprs(out_shape)});
  }
  infer_context->SetShapeOrDataForValue(
      op->result(1),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(std::vector<symbol::DimExpr>{})});
  return true;
}

bool RoiPoolOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(0)).shape();
  const auto &rois_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(1)).shape();
  const auto &rois_num_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(2));
  if (!rois_num_shape_or_data.isa<symbol::NullShapeOrDataDimExpr>()) {
    const auto &rois_num_shape = rois_num_shape_or_data.shape();
    PADDLE_ENFORCE_EQ(
        rois_num_shape.size(),
        1,
        common::errors::InvalidArgument(
            "The number of rois should be a 1-D tensor with shape (num_rois), "
            "but received the number of rois with %d dimension",
            rois_num_shape.size()));
  }

  int pooled_height =
      op->attribute<pir::Int32Attribute>("pooled_height").data();
  int pooled_width = op->attribute<pir::Int32Attribute>("pooled_width").data();
  PADDLE_ENFORCE_EQ(
      x_shape.size(),
      4,
      common::errors::InvalidArgument(
          "The input data should be a four-dimensional tensor with [N,C,H,W], "
          "but received input data with %d dimension",
          x_shape.size()));
  PADDLE_ENFORCE_EQ(
      rois_shape.size(),
      2,
      common::errors::InvalidArgument(
          "rois should be a 2-D DenseTensor with shape (num_rois, "
          "4) given as [[x1, y1, x2, y2], ...], but received "
          "rois is %d-dimensional DenseTensor",
          rois_shape.size()));
  const auto &four = symbol::DimExpr(4);
  infer_context->AddEqualCstr(rois_shape[1], four);

  auto out_dims = x_shape;

  out_dims[0] = rois_shape[0];
  out_dims[1] = x_shape[1];
  out_dims[2] = symbol::DimExpr(pooled_height);
  out_dims[3] = symbol::DimExpr(pooled_width);

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(out_dims)});
  infer_context->SetShapeOrDataForValue(
      op->result(1),
      symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(out_dims)});

  return true;
}

bool QuantizeLinearOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(0)).shape();
  const auto &scale_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(1)).shape();
  const auto &in_accum_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(3)).shape();
  const auto &in_state_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(4)).shape();

  int quant_axis = op->attribute<pir::Int32Attribute>("quant_axis").data();
  infer_context->SetShapeOrDataForValue(
      op->result(0), symbol::TensorShapeOrDataDimExprs(x_shape));

  if (op->result(1)) {
    if (quant_axis < 0) {
      infer_context->SetShapeOrDataForValue(
          op->result(1), symbol::TensorShapeOrDataDimExprs(scale_shape));
    } else {
      infer_context->SetShapeOrDataForValue(
          op->result(1),
          symbol::TensorShapeOrDataDimExprs(
              std::vector<symbol::DimExpr>{x_shape[quant_axis]}));
    }
  }

  if (op->result(2)) {
    infer_context->SetShapeOrDataForValue(
        op->result(2), symbol::TensorShapeOrDataDimExprs(in_accum_shape));
  }

  if (op->result(3)) {
    infer_context->SetShapeOrDataForValue(
        op->result(3), symbol::TensorShapeOrDataDimExprs(in_state_shape));
  }

  return true;
}

bool QuantizeLinear_OpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  return QuantizeLinearOpInferSymbolicShape(op, infer_context);
}

bool DequantizeLinearOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  return QuantizeLinearOpInferSymbolicShape(op, infer_context);
}

bool DequantizeLinear_OpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  return QuantizeLinearOpInferSymbolicShape(op, infer_context);
}
bool RoiAlignOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &x = op->operand_source(0);
  const auto &boxes = op->operand_source(1);

  const auto &num_boxes =
      infer_context->GetShapeOrDataForValue(boxes).shape()[0];
  symbol::DimExpr channel_num =
      infer_context->GetShapeOrDataForValue(x).shape()[1];

  int32_t out_h = op->attribute<pir::Int32Attribute>("pooled_height").data();
  int32_t out_w = op->attribute<pir::Int32Attribute>("pooled_width").data();

  std::vector<symbol::DimExpr> output_shape = {
      num_boxes, channel_num, out_h, out_w};
  infer_context->SetShapeOrDataForValue(
      op->result(0), symbol::TensorShapeOrDataDimExprs(output_shape));
  return true;
}

bool SpectralNormOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &weight_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(0)).shape();
  const auto &u_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(1)).shape();
  const auto &v_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(2)).shape();

  size_t rank_weight = weight_shape.size();

  PADDLE_ENFORCE_GE(rank_weight,
                    2,
                    common::errors::InvalidArgument(
                        "The rank of Input(Weights) should be greater equal "
                        "than 2, but received Weight rank(%d)",
                        rank_weight));
  PADDLE_ENFORCE_LE(rank_weight,
                    5,
                    common::errors::InvalidArgument(
                        "The rank of Input(Weights) should be less equal than "
                        "5, but received Weight rank(%d)",
                        rank_weight));

  int dim = op->attribute<pir::Int32Attribute>("dim").data();
  int power_iters = op->attribute<pir::Int32Attribute>("power_iters").data();

  PADDLE_ENFORCE_EQ(dim == 0 || dim == 1,
                    true,
                    common::errors::InvalidArgument(
                        "Attr(dim) can only be 0 or 1, but received %d", dim));
  PADDLE_ENFORCE_GE(
      power_iters,
      0,
      common::errors::InvalidArgument(
          "Attr(power_iters) should be greater equal then 0, but received %d",
          power_iters));

  symbol::DimExpr weight = 1;
  for (size_t i = 0; i < rank_weight; i++) {
    if (i != static_cast<size_t>(dim)) {
      weight = weight * weight_shape[i];
    }
  }
  infer_context->AddEqualCstr(u_shape[0], weight_shape[dim]);
  infer_context->AddEqualCstr(v_shape[0], weight);
  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(weight_shape)});

  return true;
}
bool LstmOpInferSymbolicShape(pir::Operation *op,
                              pir::InferSymbolicShapeContext *infer_context) {
  const auto &input_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(0)).shape();
  const auto &weight_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(3)).shape();
  const symbol::ShapeOrDataDimExprs &bias_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(4));
  const auto &bias_shape = bias_shape_or_data.shape();
  const bool use_peepholes =
      op->attribute<pir::BoolAttribute>("use_peepholes").data();
  PADDLE_ENFORCE_EQ(
      input_shape.size(),
      2,
      common::errors::InvalidArgument(
          "Input(X)'s rank must be 2, but received %d.", input_shape.size()));
  if (op->operand_source(1)) {
    const auto &h0_shape =
        infer_context->GetShapeOrDataForValue(op->operand_source(1)).shape();
    const auto &c0_shape =
        infer_context->GetShapeOrDataForValue(op->operand_source(2)).shape();
    size_t dims = h0_shape.size();
    for (size_t i = 0; i < dims; i++) {
      infer_context->AddEqualCstr(h0_shape[i], c0_shape[i]);
    }
  }

  PADDLE_ENFORCE_EQ(
      weight_shape.size(),
      2,
      common::errors::InvalidArgument(
          "The rank of Input(Weight) should be 2, but received %d.",
          weight_shape.size()));
  symbol::DimExpr frame_size = input_shape[1] / symbol::DimExpr(4);
  infer_context->AddEqualCstr(weight_shape[0], frame_size);
  infer_context->AddEqualCstr(weight_shape[1], frame_size * symbol::DimExpr(4));
  PADDLE_ENFORCE_EQ(bias_shape.size(),
                    2,
                    common::errors::InvalidArgument(
                        "The rank of Input(Bias) should be 2, but received %d.",
                        bias_shape.size()));
  infer_context->AddEqualCstr(bias_shape[0], symbol::DimExpr{1});
  if (use_peepholes) {
    infer_context->AddEqualCstr(bias_shape[1], frame_size * symbol::DimExpr{7});
  } else {
    infer_context->AddEqualCstr(bias_shape[1], frame_size * symbol::DimExpr{4});
  }
  std::vector<symbol::DimExpr> out_shape = {input_shape[0], frame_size};
  symbol::ShapeOrDataDimExprs out_shape_or_data =
      symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(out_shape)};
  infer_context->SetShapeOrDataForValue(op->result(0), out_shape_or_data);
  infer_context->SetShapeOrDataForValue(op->result(1), out_shape_or_data);

  // Based on the kernel and infermeta, the inferred results are the same
  // regardless of whether is_test is true or false.
  infer_context->SetShapeOrDataForValue(
      op->result(2),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(input_shape)});
  infer_context->SetShapeOrDataForValue(op->result(3), out_shape_or_data);

  return true;
}

// bool MergedAdamOpInferSymbolicShape(pir::Operation *op,
//                                     pir::InferSymbolicShapeContext
//                                     *infer_context) {
//   // pass
//   return true;
// }

// bool MergedAdam_OpInferSymbolicShape(pir::Operation *op,
//                                      pir::InferSymbolicShapeContext
//                                      *infer_context) {
//   return MergedAdamOpInferSymbolicShape(op, infer_context);
// }

// bool MergedMomentumOpInferSymbolicShape(pir::Operation *op,
//                                         pir::InferSymbolicShapeContext
//                                         *infer_context) {
//   // pass
//   return true;
// }

// bool MergedMomentum_OpInferSymbolicShape(pir::Operation *op,
//                                          pir::InferSymbolicShapeContext
//                                          *infer_context) {
//   return MergedMomentumOpInferSymbolicShape(op, infer_context);
// }

bool MoeOpInferSymbolicShape(pir::Operation *op,
                             pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(0)).shape();
  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(x_shape)});
  return true;
}

bool MulticlassNms3OpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &bboxes_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const auto &scores_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));

  const std::vector<symbol::DimExpr> &box_dims = bboxes_shape_or_data.shape();
  const std::vector<symbol::DimExpr> &score_dims = scores_shape_or_data.shape();
  const size_t score_size = score_dims.size();
  // TODO(Jeff114514):
  // add constraint for score_size == 2 || score_size == 3
  // add constraint for box_dims.size() == 3
  if (score_size == 3) {
    // TODO(Jeff114514):
    // add constraint for box_dims[2] == 4 || box_dims[2] == 8 || box_dims[2] ==
    // 16 || box_dims[2] == 24 || box_dims[2] == 32
    infer_context->AddEqualCstr(box_dims[1], score_dims[2]);
  } else {
    infer_context->AddEqualCstr(box_dims[2], symbol::DimExpr(4));
    infer_context->AddEqualCstr(box_dims[1], score_dims[1]);
  }

  const auto &next_symbol_out_and_index = infer_context->GetNextSymName();

  std::vector<symbol::DimExpr> out_shape;
  out_shape.emplace_back(next_symbol_out_and_index);
  out_shape.emplace_back(box_dims[2] + 2);

  std::vector<symbol::DimExpr> index_shape;
  index_shape.emplace_back(next_symbol_out_and_index);
  index_shape.emplace_back(1);

  std::vector<symbol::DimExpr> nms_rois_num_shape;
  nms_rois_num_shape.emplace_back(infer_context->GetNextSymName());

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(out_shape)});
  infer_context->SetShapeOrDataForValue(
      op->result(1),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(index_shape)});
  infer_context->SetShapeOrDataForValue(
      op->result(2),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(nms_rois_num_shape)});

  return true;
}

bool MeshgridOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const symbol::TensorListShapeOrDataDimExprs &shape_data_list =
      infer_context->GetShapeOrDataForValue(op->operand_source(0))
          .dyn_cast<symbol::TensorListShapeOrDataDimExprs>();

  const symbol::ShapeOrDataDimExprs sym_shape_dim_exprs = [&] {
    symbol::TensorListShapeOrDataDimExprs shape_dim_exprs_list;
    std::vector<symbol::DimExpr> vec;

    for (auto &shape_data : shape_data_list) {
      if (shape_data.shape().size() == 0) {
        vec.emplace_back(1);
      } else {
        vec.emplace_back(shape_data.shape()[0]);
      }
    }

    auto shape_dim_exprs = symbol::TensorShapeOrDataDimExprs(vec);

    for (size_t i = 0; i < shape_data_list.size(); i++) {
      shape_dim_exprs_list.emplace_back(shape_dim_exprs);
    }

    return symbol::ShapeOrDataDimExprs(shape_dim_exprs_list);
  }();

  pir::Value res = op->result(0);
  infer_context->SetShapeOrDataForValue(res, sym_shape_dim_exprs);
  return true;
}

bool NceOpInferSymbolicShape(pir::Operation *op,
                             pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const std::vector<symbol::DimExpr> &x_shape = x_shape_or_data.shape();
  const auto &label_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));
  const std::vector<symbol::DimExpr> &label_shape = label_shape_or_data.shape();

  infer_context->AddEqualCstr(x_shape[0], label_shape[0]);

  symbol::DimExpr num_true_classes =
      (label_shape.size() == 2 ? label_shape[1] : 1);

  const auto &weight_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(2));
  const std::vector<symbol::DimExpr> &weight_shape =
      weight_shape_or_data.shape();

  if (op->operand_source(3) != nullptr) {
    const auto &bias_shape_or_data =
        infer_context->GetShapeOrDataForValue(op->operand_source(3));
    const std::vector<symbol::DimExpr> &bias_shape = bias_shape_or_data.shape();
    infer_context->AddEqualCstr(weight_shape[0], bias_shape[0]);
  }

  int num_total_classes =
      op->attribute<pir::Int64Attribute>("num_total_classes").data();
  infer_context->AddEqualCstr(symbol::DimExpr(num_total_classes),
                              weight_shape[0]);

  std::vector<symbol::DimExpr> out_shape = {x_shape[0], 1};
  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(out_shape)});

  bool is_test = op->attribute<pir::BoolAttribute>("is_test").data();
  int num_neg_samples =
      op->attribute<pir::Int64Attribute>("num_neg_samples").data();
  if (!is_test) {
    std::vector<symbol::DimExpr> sample_out_shape = {x_shape[0]};
    sample_out_shape.push_back(num_true_classes + num_neg_samples);
    infer_context->SetShapeOrDataForValue(
        op->result(1),
        symbol::ShapeOrDataDimExprs{
            symbol::TensorShapeOrDataDimExprs(sample_out_shape)});
    infer_context->SetShapeOrDataForValue(
        op->result(2),
        symbol::ShapeOrDataDimExprs{
            symbol::TensorShapeOrDataDimExprs(sample_out_shape)});
  }
  return true;
}

bool MovingAverageAbsMaxScaleOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  // Get shapes of input tensors
  const auto &x_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(0)).shape();
  const auto &in_state_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(2)).shape();
  const auto &in_accum_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(1)).shape();

  // Set shapes for output tensors
  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs({x_shape})});
  infer_context->SetShapeOrDataForValue(
      op->result(1), symbol::TensorShapeOrDataDimExprs({symbol::DimExpr(1)}));
  infer_context->SetShapeOrDataForValue(
      op->result(2),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs({in_state_shape})});
  infer_context->SetShapeOrDataForValue(
      op->result(3),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs({in_accum_shape})});

  return true;
}

bool MovingAverageAbsMaxScale_OpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  return MovingAverageAbsMaxScaleOpInferSymbolicShape(op, infer_context);
}

// bool NceOpInferSymbolicShape(pir::Operation *op,
//                              pir::InferSymbolicShapeContext *infer_context){
//   // pass
//   return true;
// }

bool PsroiPoolOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const std::vector<symbol::DimExpr> &input_dims = x_shape_or_data.shape();
  const auto &rois_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));
  const std::vector<symbol::DimExpr> &rois_dims = rois_shape_or_data.shape();
  PADDLE_ENFORCE_EQ(
      input_dims.size(),
      4,
      common::errors::InvalidArgument("The format of input tensor is NCHW"));
  PADDLE_ENFORCE_EQ(
      rois_dims.size(),
      2,
      common::errors::InvalidArgument(
          "ROIs should be a 2-D DenseTensor of shape (num_rois, 4) "
          "given as [(x1, y1, x2, y2), ...]"));
  infer_context->AddEqualCstr(rois_dims[1], symbol::DimExpr(4));
  if (op->operand_source(2)) {
    auto &rois_num_shape_or_data =
        infer_context->GetShapeOrDataForValue(op->operand_source(2));
    const std::vector<symbol::DimExpr> &rois_num_dims =
        rois_num_shape_or_data.shape();
    PADDLE_ENFORCE_EQ(rois_num_dims.size(),
                      1,
                      common::errors::InvalidArgument(
                          "The second dimension of RoisNum should "
                          "be 1, but received dimension is %d",
                          rois_num_dims.size()));
  }
  int pooled_height =
      op->attribute<pir::Int32Attribute>("pooled_height").data();
  int pooled_width = op->attribute<pir::Int32Attribute>("pooled_width").data();
  int output_channels =
      op->attribute<pir::Int32Attribute>("output_channels").data();
  auto divisor =
      symbol::DimExpr(output_channels * pooled_height * pooled_width);
  infer_context->AddEqualCstr(input_dims[1], divisor);
  PADDLE_ENFORCE_GT(pooled_height,
                    0,
                    common::errors::InvalidArgument(
                        "The pooled output height must be greater than 0"));
  PADDLE_ENFORCE_GT(pooled_width,
                    0,
                    common::errors::InvalidArgument(
                        "The pooled output width must be greater than 0"));
  PADDLE_ENFORCE_GT(output_channels,
                    1,
                    common::errors::InvalidArgument(
                        "The pooled output channels must greater than 1"));
  std::vector<symbol::DimExpr> out_dims = {
      rois_dims[0], output_channels, pooled_height, pooled_width};

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(out_dims)});

  return true;
}

bool PyramidHashOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const auto &w_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));

  const std::vector<symbol::DimExpr> &x_shape = x_shape_or_data.shape();
  const std::vector<symbol::DimExpr> &w_shape = w_shape_or_data.shape();

  PADDLE_ENFORCE_EQ((x_shape.size() == 2) && (w_shape.size() == 2),
                    true,
                    common::errors::InvalidArgument(
                        "The rank of Input(X) and Input(W) of PyramidHashOP "
                        "is invalid. It should be 2, but got %d",
                        x_shape.size()));

  int num_emb = op->attribute<pir::Int32Attribute>("num_emb").data();
  int space_len = op->attribute<pir::Int32Attribute>("space_len").data();
  int rand_len = op->attribute<pir::Int32Attribute>("rand_len").data();
  int white_list_len =
      op->attribute<pir::Int32Attribute>("white_list_len").data();
  int black_list_len =
      op->attribute<pir::Int32Attribute>("black_list_len").data();

  infer_context->AddEqualCstr(w_shape[0],
                              symbol::DimExpr(space_len + rand_len));
  infer_context->AddEqualCstr(w_shape[1], symbol::DimExpr(1));

  PADDLE_ENFORCE_EQ(num_emb % rand_len,
                    0,
                    common::errors::InvalidArgument(
                        "The PyramidHashOP’s Attr(num_emb) should mod "
                        "Attr(rand_len), but num_emb is %d, rand_len is %d",
                        num_emb,
                        rand_len));

  // Handle white_list constraints if present
  if (white_list_len > 0) {
    const auto &white_list_shape_or_data =
        infer_context->GetShapeOrDataForValue(op->operand_source(2));
    if (white_list_shape_or_data.isa<symbol::NullShapeOrDataDimExpr>()) {
      PADDLE_THROW(
          "Input(WhiteList) of PyramidHashOP is not "
          "found but white_list_len > 0.");
    } else {
      std::vector<symbol::DimExpr> wl_shape = white_list_shape_or_data.shape();
      PADDLE_ENFORCE_EQ(wl_shape.size(),
                        2,
                        common::errors::InvalidArgument(
                            "The rank of Input(WhiteList) of PyramidHashOP is "
                            "invalid. It should be 2, but got %d",
                            wl_shape.size()));

      infer_context->AddEqualCstr(wl_shape[0], symbol::DimExpr(white_list_len));
      infer_context->AddEqualCstr(wl_shape[1], symbol::DimExpr(1));
    }
  }

  // Handle black_list constraints if present
  if (black_list_len > 0) {
    const auto &black_list_shape_or_data =
        infer_context->GetShapeOrDataForValue(op->operand_source(3));
    std::vector<symbol::DimExpr> bl_dims = black_list_shape_or_data.shape();

    PADDLE_ENFORCE_EQ(bl_dims.size(),
                      2,
                      common::errors::InvalidArgument(
                          "The rank of Input(BlackList) of PyramidHashOP is "
                          "invalid. It should be 2, but got %d",
                          bl_dims.size()));

    infer_context->AddEqualCstr(bl_dims[0], symbol::DimExpr(black_list_len));
    infer_context->AddEqualCstr(bl_dims[1], symbol::DimExpr(1));
  }

  // Set the output shapes
  std::vector<symbol::DimExpr> out_shape = {infer_context->GetNextSymName(),
                                            symbol::DimExpr(num_emb)};
  std::vector<symbol::DimExpr> drop_pos = {infer_context->GetNextSymName(), 1};

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(out_shape)});
  infer_context->SetShapeOrDataForValue(
      op->result(1),
      symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(drop_pos)});
  infer_context->SetShapeOrDataForValue(
      op->result(2),
      symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(x_shape)});

  return true;
}

// bool QuantizeLinearOpInferSymbolicShape(pir::Operation *op,
//                                         pir::InferSymbolicShapeContext
//                                         *infer_context) {
//   // pass
//   return true;
// }

// bool QuantizeLinear_OpInferSymbolicShape(pir::Operation *op,
//                                         pir::InferSymbolicShapeContext
//                                         *infer_context) {
//   return QuantizeLinearOpInferSymbolicShape(op, infer_context);
// }

bool RankAttentionOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const auto &rank_offset_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));
  const auto &rank_param_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(2));

  const std::vector<symbol::DimExpr> &x_shape = x_shape_or_data.shape();
  const std::vector<symbol::DimExpr> &offset_shape =
      rank_offset_shape_or_data.shape();
  const std::vector<symbol::DimExpr> &param_shape =
      rank_param_shape_or_data.shape();

  int max_rank = op->attribute<pir::Int32Attribute>("max_rank").data();
  infer_context->AddEqualCstr((offset_shape[1] - 1) / 2,
                              symbol::DimExpr(max_rank));

  std::vector<symbol::DimExpr> out_shape = {x_shape[0], param_shape[1]};
  infer_context->SetShapeOrDataForValue(
      op->result(1),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(out_shape)});

  if (details::IsFakeValue(op->result(0))) {
    infer_context->SetSymbolForValueByStaticShape(op->result(0));
  } else {
    std::vector<symbol::DimExpr> x_help_shape = {x_shape[0],
                                                 x_shape[1] * max_rank};
    infer_context->SetShapeOrDataForValue(
        op->result(0),
        symbol::ShapeOrDataDimExprs{
            symbol::TensorShapeOrDataDimExprs(x_help_shape)});
  }

  if (details::IsFakeValue(op->result(2))) {
    infer_context->SetSymbolForValueByStaticShape(op->result(2));
  } else {
    std::vector<symbol::DimExpr> ins_rank_shape = {x_shape[0], 1};
    infer_context->SetShapeOrDataForValue(
        op->result(2),
        symbol::ShapeOrDataDimExprs{
            symbol::TensorShapeOrDataDimExprs(ins_rank_shape)});
  }

  return true;
}

bool RandomRoutingOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &top_value_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(1)).shape();
  const auto &prob_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(0)).shape();
  const auto &top_idx_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(2)).shape();
  infer_context->AddEqualCstr(prob_shape[0], top_value_shape[0]);
  infer_context->AddEqualCstr(top_idx_shape[0], top_value_shape[0]);
  infer_context->AddEqualCstr(top_idx_shape[1], top_value_shape[1]);
  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(top_idx_shape)});
  return true;
}

bool RandomRouting_OpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  return RandomRoutingOpInferSymbolicShape(op, infer_context);
}

bool RmsNormOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const symbol::ShapeOrDataDimExprs &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  std::vector<symbol::DimExpr> x_shape = x_shape_or_data.shape();
  size_t x_shape_size = x_shape.size();

  symbol::DimExpr normalized_dims(1);
  int begin_norm_axis =
      op->attribute<pir::Int32Attribute>("begin_norm_axis").data();
  for (size_t i = begin_norm_axis; i < x_shape_size; ++i) {
    normalized_dims = normalized_dims * x_shape[i];
  }

  const auto &norm_weight_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(3));
  const std::vector<symbol::DimExpr> &norm_weight_dims =
      norm_weight_shape.shape();

  infer_context->AddEqualCstr(normalized_dims, norm_weight_dims[0]);

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(x_shape)});

  if (op->result(2)) {
    std::vector<symbol::DimExpr> inv_var_dims(
        x_shape.begin(), x_shape.begin() + begin_norm_axis);
    infer_context->SetShapeOrDataForValue(
        op->result(2),
        symbol::ShapeOrDataDimExprs{
            symbol::TensorShapeOrDataDimExprs(inv_var_dims)});
  }

  infer_context->SetShapeOrDataForValue(
      op->result(1),
      symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(x_shape)});

  return true;
}

bool RnnOpInferSymbolicShape(pir::Operation *op,
                             pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const auto &pre_state_shape_or_data_list =
      infer_context->GetShapeOrDataForValue(op->operand_source(1))
          .dyn_cast<symbol::TensorListShapeOrDataDimExprs>();
  const auto &sequence_length_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(3));

  const std::string &mode = op->attribute<pir::StrAttribute>("mode").AsString();
  const bool &is_bidirec =
      op->attribute<pir::BoolAttribute>("is_bidirec").data();
  const int &hidden_size =
      op->attribute<pir::Int32Attribute>("hidden_size").data();

  const auto &x_shape = x_shape_or_data.shape();
  PADDLE_ENFORCE_EQ(x_shape.size(),
                    3,
                    common::errors::InvalidArgument(
                        "The rank of Input in RNN  must be 3. But "
                        "received Input's rank is %d.",
                        x_shape.size()));

  if (!sequence_length_shape_or_data.isa<symbol::NullShapeOrDataDimExpr>()) {
    const auto &sequence_length_shape = sequence_length_shape_or_data.shape();
    infer_context->AddEqualCstr(x_shape[1], sequence_length_shape[0]);
  }

  PADDLE_ENFORCE_EQ(pre_state_shape_or_data_list[0].shape().size(),
                    3,
                    common::errors::InvalidArgument(
                        "The rank of PreState in RNN  must be 3. But "
                        "the received rank is %d.",
                        pre_state_shape_or_data_list[0].shape().size()));
  for (size_t i = 0; i < 3; ++i) {
    details::BuildCstrEqForTensorListAlongAxis(
        infer_context, pre_state_shape_or_data_list, i);
  }
  size_t i = 0;
  for (; i < pre_state_shape_or_data_list.size(); ++i) {
    infer_context->AddEqualCstr(x_shape[1],
                                pre_state_shape_or_data_list[i].shape()[1]);
  }
  size_t num_state = mode == "LSTM" ? 2 : 1;
  PADDLE_ENFORCE_EQ(i,
                    num_state,
                    common::errors::InvalidArgument(
                        "The number of tensors in PreState of %s should be %d, "
                        "but received %d.",
                        mode,
                        2,
                        i));
  std::vector<symbol::DimExpr> out_shape = x_shape;
  out_shape[2] = is_bidirec
                     ? symbol::DimExpr(static_cast<int64_t>(hidden_size) * 2)
                     : symbol::DimExpr(static_cast<int64_t>(hidden_size));
  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(out_shape)});

  size_t state_num = pre_state_shape_or_data_list.size();
  symbol::TensorListShapeOrDataDimExprs state_shape_or_data_list;
  for (size_t i = 0; i < state_num; ++i) {
    state_shape_or_data_list.emplace_back(
        pre_state_shape_or_data_list[i].shape());
  }
  infer_context->SetShapeOrDataForValue(
      op->result(2), symbol::ShapeOrDataDimExprs{state_shape_or_data_list});

  int gate_num = 4;
  if (mode == "RNN_RELU" || mode == "RNN_TANH") {
    gate_num = 1;
  } else if (mode == "GRU") {
    gate_num = 3;
  }
  const int &num_layers =
      op->attribute<pir::Int32Attribute>("num_layers").data();

  int hidden_date_idx = num_layers - 1;
  if (mode == "LSTM") {
    hidden_date_idx += (gate_num + 2) * num_layers;
  } else if (mode == "GRU") {
    hidden_date_idx += (gate_num + 1) * num_layers;
  } else {
    hidden_date_idx += gate_num * num_layers;
  }
  symbol::DimExpr block_size =
      symbol::DimExpr(static_cast<int64_t>(num_state)) * x_shape[0] *
      x_shape[1] * symbol::DimExpr(hidden_size);
  std::vector<symbol::DimExpr> reserve_shape = {symbol::DimExpr(hidden_size),
                                                block_size};
  infer_context->SetShapeOrDataForValue(
      op->result(3),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(reserve_shape)});

  symbol::DimExpr dropout_state_shape = infer_context->GetNextSymName();

  infer_context->SetShapeOrDataForValue(
      op->result(1),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs({dropout_state_shape})});
  return true;
}

bool Rnn_OpInferSymbolicShape(pir::Operation *op,
                              pir::InferSymbolicShapeContext *infer_context) {
  return RnnOpInferSymbolicShape(op, infer_context);
}

// bool RoiPoolOpInferSymbolicShape(pir::Operation *op,
//                                  pir::InferSymbolicShapeContext
//                                  *infer_context) {
//   // pass
//   return true;
// }

bool SequenceConvOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  // 获取输入张量的符号形状和数据类型
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const auto &padding_data_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));
  const auto &filter_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(2));
  const std::vector<symbol::DimExpr> &x_dims = x_shape_or_data.shape();
  const std::vector<symbol::DimExpr> &filter_dims =
      filter_shape_or_data.shape();

  int context_length =
      op->attribute<pir::Int32Attribute>("context_length").data();
  bool padding_trainable =
      op->attribute<pir::BoolAttribute>("padding_trainable").data();
  int context_start =
      op->attribute<pir::Int32Attribute>("context_start").data();
  int context_stride =
      op->attribute<pir::Int32Attribute>("context_stride").data();

  PADDLE_ENFORCE_EQ(
      context_stride,
      1,
      common::errors::InvalidArgument(
          "Currently, SequenceConvOp only supports contextStride=1. But "
          "received contextStride = %u.",
          context_stride));
  PADDLE_ENFORCE_EQ(
      x_dims.size() == 2 && filter_dims.size() == 2,
      true,
      common::errors::InvalidArgument(
          "Input(X, Filter) should be 2-D tensor. But received Input(X): "
          "input rank %u, input shape [%s]; received Input(Filter): "
          "input rank %u, input shape [%s].",
          x_dims.size(),
          x_dims,
          filter_dims.size(),
          filter_dims));
  auto divisor = symbol::DimExpr(context_length);
  infer_context->AddEqualCstr(filter_dims[0], divisor * x_dims[1]);

  if (padding_trainable) {
    const std::vector<symbol::DimExpr> &padding_dims =
        padding_data_shape_or_data.shape();
    int up_pad = std::max(0, -context_start);
    int down_pad = std::max(0, context_start + context_length - 1);
    int total_pad = up_pad + down_pad;
    const auto &input_width = x_dims[1];
    bool start_equals_zero = context_start == 0;
    bool length_equals_one = context_length == 1;
    bool start_length = start_equals_zero && length_equals_one;

    PADDLE_ENFORCE_EQ(
        start_length,
        false,
        common::errors::InvalidArgument(
            "If context_start is 0 and context_length is 1, paddingTrainable "
            "should be false."));
    PADDLE_ENFORCE_EQ(
        padding_dims.size(),
        2,
        common::errors::InvalidArgument(
            "Input(PaddingData) should be 2-D tensor. But received: "
            "input rank %u, input shape [%s].",
            padding_dims.size(),
            padding_dims));
    infer_context->AddEqualCstr(padding_dims[0], symbol::DimExpr(total_pad));
    infer_context->AddEqualCstr(padding_dims[1], input_width);
  }

  std::vector<symbol::DimExpr> out_dims = x_dims;
  out_dims[1] = filter_dims[1];
  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(out_dims)});

  return true;
}

bool SparseAttentionOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  // Get symbolic shapes of the input tensors
  const auto &q_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const auto &k_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));
  const auto &v_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(2));
  const auto &offset_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(3));
  const auto &columns_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(4));

  std::vector<symbol::DimExpr> q_dims = q_shape_or_data.shape();
  std::vector<symbol::DimExpr> k_dims = k_shape_or_data.shape();
  std::vector<symbol::DimExpr> v_dims = v_shape_or_data.shape();
  std::vector<symbol::DimExpr> offset_dims = offset_shape_or_data.shape();
  std::vector<symbol::DimExpr> columns_dims = columns_shape_or_data.shape();

  // Ensure the input tensors have the expected rank
  PADDLE_ENFORCE_EQ(q_dims.size(),
                    4UL,
                    common::errors::InvalidArgument(
                        "Dimension in query's shapes should be 4."));
  PADDLE_ENFORCE_EQ(k_dims.size(),
                    4UL,
                    common::errors::InvalidArgument(
                        "Dimension in key's shapes should be 4."));
  PADDLE_ENFORCE_EQ(v_dims.size(),
                    4UL,
                    common::errors::InvalidArgument(
                        "Dimension in value's shapes should be 4."));
  PADDLE_ENFORCE_EQ(offset_dims.size(),
                    3UL,
                    common::errors::InvalidArgument(
                        "Dimension in offset's shapes should be 3."));
  PADDLE_ENFORCE_EQ(columns_dims.size(),
                    3UL,
                    common::errors::InvalidArgument(
                        "Dimension in columns' shapes should be 3."));

  // Add equality constraints between corresponding dimensions
  infer_context->AddEqualCstr(q_dims[0], k_dims[0]);  // batch_size
  infer_context->AddEqualCstr(q_dims[1], k_dims[1]);  // num_heads
  infer_context->AddEqualCstr(q_dims[2], k_dims[2]);  // M (seq_len)
  infer_context->AddEqualCstr(q_dims[3], k_dims[3]);  // N (head_dim)
  infer_context->AddEqualCstr(v_dims[0], k_dims[0]);  // batch_size
  infer_context->AddEqualCstr(v_dims[1], k_dims[1]);  // num_heads
  infer_context->AddEqualCstr(v_dims[3], k_dims[3]);  // head_dim

  // Ensure that offset and columns dimensions are consistent with the input
  // tensors
  infer_context->AddEqualCstr(offset_dims[0], q_dims[0]);      // batch_size
  infer_context->AddEqualCstr(offset_dims[1], q_dims[1]);      // num_heads
  infer_context->AddEqualCstr(offset_dims[2], q_dims[2] + 1);  // seq_len + 1

  infer_context->AddEqualCstr(columns_dims[0], q_dims[0]);  // batch_size
  infer_context->AddEqualCstr(columns_dims[1], q_dims[1]);  // num_heads

  // Prepare the output tensor shapes
  auto batch_size = q_dims[0];
  auto num_heads = q_dims[1];
  auto M = q_dims[2];
  auto N = q_dims[3];
  auto sparse_nnz = columns_dims[2];

  // Set output tensor shapes
  std::vector<symbol::DimExpr> out_dims = {batch_size, num_heads, M, N};
  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(out_dims)});

  std::vector<symbol::DimExpr> sparse_dot_sdd_dims = {
      batch_size, num_heads, sparse_nnz};
  infer_context->SetShapeOrDataForValue(
      op->result(1),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(sparse_dot_sdd_dims)});

  std::vector<symbol::DimExpr> softmax_dims = {
      batch_size, num_heads, sparse_nnz};
  infer_context->SetShapeOrDataForValue(
      op->result(2),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(softmax_dims)});

  return true;
}

// bool SpectralNormOpInferSymbolicShape(pir::Operation *op,
//                                       pir::InferSymbolicShapeContext
//                                       *infer_context) {
//   // pass
//   return true;
// }

bool StackOpInferSymbolicShape(pir::Operation *op,
                               pir::InferSymbolicShapeContext *infer_context) {
  pir::Value operand_source = op->operand_source(0);

  const auto &attributes = op->attributes();
  int axis = attributes.at("axis").dyn_cast<pir::Int32Attribute>().data();
  const symbol::TensorListShapeOrDataDimExprs &shape_data_list =
      infer_context->GetShapeOrDataForValue(operand_source)
          .dyn_cast<symbol::TensorListShapeOrDataDimExprs>();

  size_t rank = shape_data_list.at(0).shape().size();
  if (axis < 0) axis += rank + 1;
  const symbol::ShapeOrDataDimExprs shape_data = [&] {
    std::vector<symbol::DimExpr> result_shape = {};
    std::vector<symbol::DimExpr> result_data = {};
    const symbol::TensorShapeOrDataDimExprs &x_shape_data =
        shape_data_list.at(0);

    const bool data_flag = [&] {
      for (const auto &shape_data : shape_data_list) {
        if (!shape_data.data().has_value()) {
          return false;
        }
      }
      return true;
    }();

    if (data_flag) {
      // case 1: data is not empty, eg: shape_data_list =
      // [[shape:{3},data:{S0,6,7}],...]
      if (axis == 0 && x_shape_data.data().value().size() <= 1) {
        for (const auto &shape_data : shape_data_list) {
          result_data.emplace_back(shape_data.data().value().at(0));
        }
      } else {
        PADDLE_THROW(common::errors::Unimplemented(
            op->name() +
            " 's InferSymbolicShape can NOT deal with data size > 1 now."));
      }
      result_shape.emplace_back(
          static_cast<std::int64_t>(shape_data_list.size()));
    } else {
      // case 2: data is empty, eg: shape_data_list =
      // [[shape:{5,6,7},data:{}],...]
      for (size_t i = 0; i < rank; ++i) {
        details::BuildCstrEqForTensorListAlongAxis(
            infer_context, shape_data_list, i);
      }
      for (const symbol::DimExpr &dim : x_shape_data.shape()) {
        result_shape.emplace_back(dim);
      }
      result_shape.insert(result_shape.begin() + axis,
                          static_cast<std::int64_t>(shape_data_list.size()));
    }

    if (result_data.empty()) {
      return symbol::ShapeOrDataDimExprs(
          symbol::TensorShapeOrDataDimExprs(result_shape));
    }
    return symbol::ShapeOrDataDimExprs(
        symbol::TensorShapeOrDataDimExprs(result_shape, result_data));
  }();

  pir::Value res = op->result(0);
  infer_context->SetShapeOrDataForValue(res, shape_data);
  return true;
}

// bool SaveCombineOpInferSymbolicShape(pir::Operation *op,
//                                      pir::InferSymbolicShapeContext
//                                      *infer_context) {
//   // pass
//   return true;
// }

bool SigmoidCrossEntropyWithLogitsOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &input_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const auto &label_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));

  size_t rank = input_shape_or_data.shape().size();
  PADDLE_ENFORCE_EQ(rank,
                    label_shape_or_data.shape().size(),
                    common::errors::InvalidArgument(
                        "Input(X) and Input(Label) shall have the same rank."
                        "But received: the rank of Input(X) is [%d], "
                        "the rank of Input(Label) is [%d].",
                        rank,
                        label_shape_or_data.shape().size()));

  for (size_t i = 0; i < rank; ++i) {
    infer_context->AddEqualCstr(input_shape_or_data.shape()[i],
                                label_shape_or_data.shape()[i]);
  }
  if (op->operand_source(2)) {
    const auto &pos_shape_or_data =
        infer_context->GetShapeOrDataForValue(op->operand_source(2));
    for (size_t i = 0; i < rank; ++i) {
      infer_context->AddEqualCstr(input_shape_or_data.shape()[i],
                                  pos_shape_or_data.shape()[i]);
    }
  }
  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(input_shape_or_data.shape())});
  return true;
}

bool SigmoidCrossEntropyWithLogits_OpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  return SigmoidCrossEntropyWithLogitsOpInferSymbolicShape(op, infer_context);
}

bool SyncBatchNorm_OpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  return BatchNormOpInferSymbolicShape(op, infer_context);
}

bool TdmSamplerOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const symbol::ShapeOrDataDimExprs &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  std::vector<symbol::DimExpr> x_dims = x_shape_or_data.shape();

  bool output_positive =
      op->attribute<pir::BoolAttribute>("output_positive").data();
  std::vector<int> neg_samples_num_list =
      paddle::dialect::details::GetVectorAttr<int>(op, "neg_samples_num_list");

  int64_t sample_res_length = 0;
  for (int sample_nums : neg_samples_num_list) {
    sample_res_length += sample_nums + static_cast<int64_t>(output_positive);
  }

  symbol::DimExpr batch_size = x_dims[0];
  symbol::DimExpr sample_res_dim(sample_res_length);

  std::vector<symbol::DimExpr> output_dims = {batch_size, sample_res_dim};
  symbol::TensorShapeOrDataDimExprs output_shape(output_dims);

  infer_context->SetShapeOrDataForValue(
      op->result(0), symbol::ShapeOrDataDimExprs{output_shape});
  infer_context->SetShapeOrDataForValue(
      op->result(1), symbol::ShapeOrDataDimExprs{output_shape});
  infer_context->SetShapeOrDataForValue(
      op->result(2), symbol::ShapeOrDataDimExprs{output_shape});

  return true;
}

bool TrilinearInterpOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  return BicubicInterpOpInferSymbolicShape(op, infer_context);
}

bool HsigmoidLossOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &input =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const auto &label =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));
  const std::vector<symbol::DimExpr> &input_dims = input.shape();
  const std::vector<symbol::DimExpr> &label_dims = label.shape();

  infer_context->AddEqualCstr(input_dims[0], label_dims[0]);

  std::vector<symbol::DimExpr> out_shape = {input_dims[0], 1};

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(out_shape)});
  infer_context->SetShapeOrDataForValue(
      op->result(1),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(out_shape)});
  infer_context->SetShapeOrDataForValue(
      op->result(2),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(out_shape)});

  return true;
}

bool ViterbiDecodeOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &input_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const auto &transition_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));
  const auto &length_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(2));
  const std::vector<symbol::DimExpr> &input_shape = input_shape_or_data.shape();
  const std::vector<symbol::DimExpr> &transition_shape =
      transition_shape_or_data.shape();
  const std::vector<symbol::DimExpr> &length_shape =
      length_shape_or_data.shape();

  infer_context->AddEqualCstr(input_shape[0], length_shape[0]);
  infer_context->AddEqualCstr(input_shape[2], transition_shape[0]);

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(length_shape)});

  symbol::DimExpr batch_size = input_shape[0];

  std::vector<symbol::DimExpr> path_shape = {batch_size,
                                             infer_context->GetNextSymName()};
  infer_context->SetShapeOrDataForValue(
      op->result(1),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(path_shape)});

  return true;
}

bool WarpctcOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &logits_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const std::vector<symbol::DimExpr> &logits_shape =
      logits_shape_or_data.shape();

  symbol::DimExpr max_sequence_length, num_sequences;
  symbol::DimExpr sequence_width = symbol::DimExpr(1);

  if (op->operand_source(2) && op->operand_source(3)) {
    max_sequence_length = logits_shape[0];
    num_sequences = logits_shape[1];
    sequence_width = logits_shape[2];
  } else {
    max_sequence_length = infer_context->GetNextSymName();
    num_sequences = infer_context->GetNextSymName();
    for (size_t i = 1; i < logits_shape.size(); ++i) {
      sequence_width = sequence_width * logits_shape[i];
    }
  }

  std::vector<symbol::DimExpr> loss_shape = {num_sequences, symbol::DimExpr(1)};
  std::vector<symbol::DimExpr> warpctcgrad_shape = {
      max_sequence_length, num_sequences, sequence_width};

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(loss_shape)});
  infer_context->SetShapeOrDataForValue(
      op->result(1),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(warpctcgrad_shape)});

  return true;
}

bool WarprnntOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &input_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const std::vector<symbol::DimExpr> &input_shape = input_shape_or_data.shape();

  std::vector<symbol::DimExpr> loss_shape = {input_shape[0]};
  std::vector<symbol::DimExpr> warpctcgrad_shape = input_shape;

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(loss_shape)});
  infer_context->SetShapeOrDataForValue(
      op->result(1),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(warpctcgrad_shape)});
  return true;
}

bool WeightOnlyLinearOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const auto &weight_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));
  const auto &bias_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(2));
  const auto &weight_scale_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(3));
  const std::string &weight_dtype =
      op->attribute<pir::StrAttribute>("weight_dtype").AsString();
  const int group_size =
      op->attribute<pir::Int32Attribute>("group_size").data();

  PADDLE_ENFORCE(
      (group_size == -1 || group_size == 64 || group_size == 128),
      common::errors::InvalidArgument("group_size must be -1, 64 or 128."));

  ExprVec weight_scale_shape = weight_scale_shape_or_data.shape();
  ExprVec x_shape = x_shape_or_data.shape();
  ExprVec weight_shape = weight_shape_or_data.shape();
  symbol::DimExpr n =
      group_size == -1 ? weight_scale_shape[0] : weight_scale_shape[1];
  PADDLE_ENFORCE(weight_dtype == "int8" || weight_dtype == "int4",
                 common::errors::InvalidArgument(
                     "quant_method must be 'int8' or 'int4'."));
  PADDLE_ENFORCE_EQ(weight_shape.size(),
                    2UL,
                    common::errors::InvalidArgument(
                        "The input(weight) must be a 2D Tensor."));
  // TODO(Jeff114514): can not use % between symbol::DimExpr and int, need to
  // make sure weight_shape[0] and weight_shape[1] is divisible by 16
  infer_context->AddEqualCstr(x_shape[x_shape.size() - 1], weight_shape[1]);
  if (!bias_shape_or_data.isa<symbol::NullShapeOrDataDimExpr>()) {
    ExprVec bias_shape = bias_shape_or_data.shape();
    PADDLE_ENFORCE_EQ(
        bias_shape.size(),
        1UL,
        common::errors::InvalidArgument(
            "The size of Input(Bias)'s dimension should equal to 1UL.",
            bias_shape.size()));
  }

  if (group_size == -1) {
    PADDLE_ENFORCE_EQ(weight_scale_shape.size(),
                      1UL,
                      common::errors::InvalidArgument(
                          "The input(weight_scale) must be a 1D Tensor."
                          "in per-channel mode."));
  } else {
    PADDLE_ENFORCE_EQ(weight_scale_shape.size(),
                      2UL,
                      common::errors::InvalidArgument(
                          "The input(weight_scale) must be a 2D Tensor"
                          " in groupwise mode."));
    infer_context->AddEqualCstr(
        weight_scale_shape[0],
        (weight_shape[1] + (group_size - 1)) / group_size);
  }
  ExprVec out_shape = x_shape;
  out_shape[out_shape.size() - 1] = n;
  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(out_shape)});
  return true;
}

bool WeightedSampleNeighborsOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  auto GSNShapeCheck = [](const ExprVec &input_shape,
                          std::string tensor_name,
                          pir::InferSymbolicShapeContext *infer_context) {
    if (input_shape.size() == 2) {
      infer_context->AddEqualCstr(input_shape[1], symbol::DimExpr(1));
    } else {
      PADDLE_ENFORCE_EQ(
          input_shape.size(),
          1,
          common::errors::InvalidArgument(
              "The %s should be 1D, when it is not 2D, but we get %d",
              tensor_name,
              input_shape.size()));
    }
  };

  const auto &row_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(0)).shape();
  const auto &col_ptr_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(1)).shape();
  const auto &edge_weight_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(2)).shape();
  const auto &x_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(3)).shape();
  const auto &eids_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(4)).shape();
  bool return_eids = op->attribute<pir::BoolAttribute>("return_eids").data();

  GSNShapeCheck(row_shape, "row", infer_context);
  GSNShapeCheck(col_ptr_shape, "col_ptr", infer_context);
  GSNShapeCheck(edge_weight_shape, "edge_weight", infer_context);
  GSNShapeCheck(x_shape, "input_nodes", infer_context);

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(
          {infer_context->GetNextSymName()})});
  infer_context->SetShapeOrDataForValue(
      op->result(1),
      symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(
          {infer_context->GetNextSymName()})});
  if (return_eids) {
    GSNShapeCheck(eids_shape, "eids", infer_context);
    infer_context->SetShapeOrDataForValue(
        op->result(2),
        symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(
            {infer_context->GetNextSymName()})});
  } else {
    infer_context->SetSymbolForValueByStaticShape(op->result(2));
  }
  return true;
}

bool WhereOpInferSymbolicShape(pir::Operation *op,
                               pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(0)).shape();
  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(x_shape)});

  const std::vector<pir::Value> &operands = {
      op->operand_source(0), op->operand_source(1), op->operand_source(2)};

  size_t rank = x_shape.size();

  for (size_t i = 0; i < rank; ++i) {
    paddle::dialect::details::BuildCstrEqForTensorListAlongAxis(
        infer_context, operands, i);
  }

  return true;
}

bool Where_OpInferSymbolicShape(pir::Operation *op,
                                pir::InferSymbolicShapeContext *infer_context) {
  return WhereOpInferSymbolicShape(op, infer_context);
}

bool MultiplexOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &inputs_shape_or_data_list =
      infer_context->GetShapeOrDataForValue(op->operand_source(0))
          .dyn_cast<symbol::TensorListShapeOrDataDimExprs>();

  PADDLE_ENFORCE_NE(
      inputs_shape_or_data_list.empty(),
      true,
      common::errors::InvalidArgument("MultiInput(X) shouldn't be empty."));

  const auto &ids_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));

  PADDLE_ENFORCE_EQ(ids_shape_or_data.shape().size(),
                    2,
                    common::errors::PreconditionNotMet(
                        "The index tensor must be a vector with 2 dimensions"));

  infer_context->AddEqualCstr(ids_shape_or_data.shape()[1], symbol::DimExpr(1));

  PADDLE_ENFORCE_GT(
      inputs_shape_or_data_list.size(),
      1,
      common::errors::InvalidArgument("multiplex operator should have more "
                                      "than one candidate input tensors."));

  size_t num_inputs = inputs_shape_or_data_list.size();
  std::vector<symbol::DimExpr> first_input_shape =
      inputs_shape_or_data_list[0].shape();
  PADDLE_ENFORCE_GE(
      first_input_shape.size(),
      2,
      common::errors::InvalidArgument(
          "The rank of candidate tensors must be not less than 2."));

  for (size_t i = 1; i < num_inputs; ++i) {
    std::vector<symbol::DimExpr> element_shape =
        inputs_shape_or_data_list[i].shape();

    PADDLE_ENFORCE_EQ(first_input_shape.size(),
                      element_shape.size(),
                      common::errors::PreconditionNotMet(
                          "All the candidate tensors must have the same dim."));

    for (size_t j = 0; j < first_input_shape.size(); ++j)
      infer_context->AddEqualCstr(first_input_shape[j], element_shape[j]);
    // all of the input Tensors should have the same shape
  }

  if (first_input_shape[0].isa<int64_t>() &&
      ids_shape_or_data.shape()[0].isa<int64_t>()) {
    PADDLE_ENFORCE_GE(first_input_shape[0].dyn_cast<int64_t>(),
                      ids_shape_or_data.shape()[0].dyn_cast<int64_t>(),
                      common::errors::InvalidArgument(
                          "The 2nd-dim of input cannot be smaller than "
                          "batchSize of the index tensor."));
  }
  std::vector<symbol::DimExpr> &output_shape = first_input_shape;
  output_shape[0] = ids_shape_or_data.shape()[0];
  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrData{symbol::TensorShapeOrDataDimExprs(output_shape)});
  return true;
}
bool YoloLossOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(0)).shape();
  const auto &box_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(1)).shape();
  const auto &label_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(2)).shape();
  const std::vector<int> &anchors_mask =
      paddle::dialect::details::GetVectorAttr<int>(op, "anchor_mask");
  int mask_num = anchors_mask.size();
  int class_num = op->attribute<pir::Int32Attribute>("class_num").data();

  PADDLE_ENFORCE_EQ(x_shape.size(),
                    4,
                    common::errors::InvalidArgument(
                        "Input(X) should be a 4-D tensor. But received "
                        "X dimension size(%s)",
                        x_shape.size()));
  PADDLE_ENFORCE_EQ(box_shape.size(),
                    3,
                    common::errors::InvalidArgument(
                        "Input(GTBox) should be a 3-D tensor, but "
                        "received gtbox dimension size(%s)",
                        box_shape.size()));
  PADDLE_ENFORCE_EQ(label_shape.size(),
                    2,
                    common::errors::InvalidArgument(
                        "Input(GTLabel) should be a 2-D tensor,"
                        "But received Input(GTLabel) dimension size(%s) != 2.",
                        label_shape.size()));
  infer_context->AddEqualCstr(box_shape[2], symbol::DimExpr(4));
  infer_context->AddEqualCstr(x_shape[2], x_shape[3]);
  infer_context->AddEqualCstr(x_shape[1],
                              symbol::DimExpr(mask_num * (5 + class_num)));
  infer_context->AddEqualCstr(label_shape[0], box_shape[0]);
  infer_context->AddEqualCstr(label_shape[1], box_shape[1]);

  if (op->operand_source(3) != nullptr) {
    const auto &score_shape =
        infer_context->GetShapeOrDataForValue(op->operand_source(3)).shape();
    PADDLE_ENFORCE_EQ(
        score_shape.size(),
        2,
        common::errors::InvalidArgument("Input(GTScore) should be a 2-D tensor"
                                        "But received GTScore dimension(%s)",
                                        box_shape.size()));
    infer_context->AddEqualCstr(score_shape[0], box_shape[0]);
    infer_context->AddEqualCstr(score_shape[1], box_shape[1]);
  }

  std::vector<symbol::DimExpr> out_shape = {x_shape[0]};
  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(out_shape)});

  std::vector<symbol::DimExpr> obj_mask_shape = {
      x_shape[0], symbol::DimExpr(mask_num), x_shape[2], x_shape[3]};
  infer_context->SetShapeOrDataForValue(
      op->result(1),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(obj_mask_shape)});

  std::vector<symbol::DimExpr> match_mask_shape = {box_shape[0], box_shape[1]};
  infer_context->SetShapeOrDataForValue(
      op->result(2),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(match_mask_shape)});

  return true;
}

bool FakeChannelWiseDequantizeMaxAbsOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));

  int quant_axis = op->attribute<pir::Int32Attribute>("quant_axis").data();
  int x_num_col_dims =
      op->attribute<pir::Int32Attribute>("x_num_col_dims").data();

  PADDLE_ENFORCE_EQ(
      quant_axis == 0 || quant_axis == 1,
      true,
      common::errors::InvalidArgument("'quant_axis' should be 0 or 1, but "
                                      "the received is %d",
                                      quant_axis));
  PADDLE_ENFORCE_EQ(x_num_col_dims == 0,
                    false,
                    common::errors::InvalidArgument(
                        "'x_num_col_dims' should be larger than 0, but "
                        "the received is %d",
                        x_num_col_dims));

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(x_shape_or_data.shape())});

  return true;
}

bool MultiDotOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const symbol::TensorListShapeOrDataDimExprs &input_values =
      infer_context->GetShapeOrDataForValue(op->operand_source(0))
          .dyn_cast<symbol::TensorListShapeOrDataDimExprs>();
  const size_t input_num = input_values.size();
  PADDLE_ENFORCE_GT(
      input_num,
      1,
      common::errors::InvalidArgument(
          "The number of input tensors in multi_dot op should > 1"));

  bool is_vector = false;
  std::vector<symbol::DimExpr> output_shape;

  auto first_value_shape = input_values[0].shape();
  PADDLE_ENFORCE_LT(
      first_value_shape.size(),
      3,
      common::errors::InvalidArgument(
          "multi_dot: the first input tensor must be 1D or 2D but got[%d]!",
          static_cast<int>(first_value_shape.size())));
  // If the first tensor is 1D of size n view it as a row vector (1, n)

  if (first_value_shape.size() == 1) {
    first_value_shape =
        std::vector<symbol::DimExpr>{symbol::DimExpr(1), first_value_shape[0]};
    is_vector = true;
  }

  auto last_value_shape = input_values[input_num - 1].shape();
  PADDLE_ENFORCE_LT(
      last_value_shape.size(),
      3,
      common::errors::InvalidArgument(
          "the last input tensor of multi_dot must be 1D or 2D but got[%d]!",
          static_cast<int>(first_value_shape.size())));

  // If the last tensor is 1D of size n view it as a column vector (n, 1)
  if (last_value_shape.size() == 1) {
    last_value_shape =
        std::vector<symbol::DimExpr>{last_value_shape[0], symbol::DimExpr(1)};
    output_shape = is_vector
                       ? std::vector<symbol::DimExpr>{}
                       : std::vector<symbol::DimExpr>{first_value_shape[0]};
  } else {
    output_shape = is_vector ? std::vector<symbol::DimExpr>{last_value_shape[1]}
                             : std::vector<symbol::DimExpr>{
                                   first_value_shape[0], last_value_shape[1]};
  }

  auto width = first_value_shape[1];
  for (size_t i = 1; i < input_num - 1; ++i) {
    auto &input_dim = input_values[i].shape();
    PADDLE_ENFORCE_EQ(input_dim.size(),
                      2,
                      common::errors::InvalidArgument(
                          "the input tensor of multi_dot op must be 2D."));

    infer_context->AddEqualCstr(input_dim[0], width);
    width = input_dim[1];
  }

  infer_context->AddEqualCstr(last_value_shape[0], width);

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(output_shape)});
  return true;
}
bool UpdateLossScaling_OpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const symbol::TensorListShapeOrDataDimExprs &shape_data_list =
      infer_context->GetShapeOrDataForValue(op->operand_source(0))
          .dyn_cast<symbol::TensorListShapeOrDataDimExprs>();

  const symbol::ShapeOrDataDimExprs sym_shape_dim_exprs = [&] {
    symbol::TensorListShapeOrDataDimExprs shape_dim_exprs_list;

    for (auto &shape_data : shape_data_list) {
      auto shape_dim_exprs =
          symbol::TensorShapeOrDataDimExprs(shape_data.shape());
      shape_dim_exprs_list.emplace_back(shape_dim_exprs);
    }

    return symbol::ShapeOrDataDimExprs(shape_dim_exprs_list);
  }();

  infer_context->SetShapeOrDataForValue(op->result(0), sym_shape_dim_exprs);

  infer_context->SetShapeOrDataForValue(
      op->result(1),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs({symbol::DimExpr(1)})});

  infer_context->SetShapeOrDataForValue(
      op->result(2),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs({symbol::DimExpr(1)})});

  infer_context->SetShapeOrDataForValue(
      op->result(3),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs({symbol::DimExpr(1)})});

  return true;
}

bool LuSolveOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &lu_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(0)).shape();
  const auto &b_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(1)).shape();

  // LU tensor must be at least 2D (square matrix)
  PADDLE_ENFORCE_GE(
      lu_shape.size(),
      2,
      common::errors::InvalidArgument(
          "The LU tensor must be at least 2D, but got %d dimensions",
          lu_shape.size()));

  // Last two dimensions of LU must be equal (square matrix)
  infer_context->AddEqualCstr(lu_shape[lu_shape.size() - 1],
                              lu_shape[lu_shape.size() - 2]);

  // b tensor must have compatible shape with LU
  PADDLE_ENFORCE_GE(
      b_shape.size(),
      1,
      common::errors::InvalidArgument(
          "The b tensor must be at least 1D, but got %d dimensions",
          b_shape.size()));

  // For broadcasting rules, b should be compatible with lu along batch
  // dimensions
  size_t lu_batch_dims = lu_shape.size() - 2;
  size_t b_batch_dims = b_shape.size() - 1;
  size_t max_batch_dims = std::max(lu_batch_dims, b_batch_dims);

  // Output shape is same as b shape
  std::vector<symbol::DimExpr> out_shape = b_shape;

  // Add broadcast constraints for batch dimensions
  for (size_t i = 0; i < max_batch_dims; ++i) {
    symbol::DimExpr lu_dim =
        i < lu_batch_dims ? lu_shape[i] : symbol::DimExpr(1);
    symbol::DimExpr b_dim = i < b_batch_dims ? b_shape[i] : symbol::DimExpr(1);
    infer_context->AddBroadcastableCstr(lu_dim, b_dim);
  }

  // Last dimension of b must be equal to last dimension of LU
  infer_context->AddEqualCstr(b_shape[b_shape.size() - 1],
                              lu_shape[lu_shape.size() - 1]);

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(out_shape)});

  return true;
}

bool YoloBoxPostOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &image_shape_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(3));

  const symbol::DimExpr &batch = image_shape_shape_or_data.shape()[0];

  std::vector<symbol::DimExpr> dim_out = {symbol::DimExpr(1),
                                          symbol::DimExpr(6)};
  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(dim_out)});

  std::vector<symbol::DimExpr> dim_nms_rois_num = {batch};
  infer_context->SetShapeOrDataForValue(
      op->result(1),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(dim_nms_rois_num)});

  return true;
}

}  // namespace paddle::dialect
