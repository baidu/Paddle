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

#include "paddle/fluid/pir/dialect/operator/interface/infer_symbolic_shape/nullary_infer_sym.h"
#include "paddle/fluid/pir/dialect/operator/interface/infer_symbolic_shape/infer_sym_utils.h"

namespace paddle::dialect {

bool ArangeOpInferSymbolicShape(pir::Operation *op,
                                pir::InferSymbolicShapeContext *infer_context) {
  const auto &start_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const auto &end_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));
  const auto &step_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(2));

  const auto result = op->result(0);
  bool contain_unknown_dim = [&]() {
    bool check = result && result.type() &&
                 result.type().isa<paddle::dialect::DenseTensorType>();
    PADDLE_ENFORCE_EQ(check,
                      true,
                      phi::errors::PreconditionNotMet(
                          "result for arange must be DenseTensorType"));
    const auto dims =
        result.type().dyn_cast<paddle::dialect::DenseTensorType>().dims();
    return ::common::contain_unknown_dim(dims);
  }();

  if (!contain_unknown_dim) {
    infer_context->SetSymbolForValueByStaticShape(result);
    return true;
  }

  const symbol::ShapeOrDataDimExprs &shape_data = [&] {
    if (!start_shape_or_data.data().has_value() ||
        !end_shape_or_data.data().has_value() ||
        !step_shape_or_data.data().has_value()) {
      return symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(std::vector<symbol::DimExpr>{
              symbol::DimExpr(infer_context->GetNextSymName())})};
    }
    const auto &start = start_shape_or_data.data()->at(0);
    const auto &end = end_shape_or_data.data()->at(0);
    const auto &step = step_shape_or_data.data()->at(0);
    std::vector<symbol::DimExpr> out_dims;
    // TODO(lanxianghit, jiahy0825): here should be ceil((end - start) / step),
    // but DimExpr doesn't support ceil and float now
    out_dims.emplace_back((end - start) / step);
    return symbol::ShapeOrDataDimExprs{
        symbol::TensorShapeOrDataDimExprs(out_dims)};
  }();

  infer_context->SetShapeOrDataForValue(op->result(0), shape_data);

  return true;
}

bool AssignValueOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const std::vector<int> shape =
      paddle::dialect::details::GetVectorAttr<int>(op, "shape");
  std::vector<symbol::DimExpr> sym_dims;
  sym_dims.reserve(shape.size());
  for (const int &dim : shape) {
    sym_dims.emplace_back(symbol::DimExpr(static_cast<int64_t>(dim)));
  }

  bool result_is_int_type = [&]() {
    const auto &dtype =
        op->result(0).type().dyn_cast<pir::DenseTensorType>().dtype();
    return dtype.isa<pir::Int32Type>() || dtype.isa<pir::Int64Type>();
  }();

  std::vector<int64_t> values;
  if (result_is_int_type) {
    const auto &attributes = op->attributes();
    for (size_t i = 0;
         i < attributes.at("values").dyn_cast<pir::ArrayAttribute>().size();
         i++) {
      values.push_back(attributes.at("values")
                           .dyn_cast<pir::ArrayAttribute>()
                           .at(i)
                           .dyn_cast<paddle::dialect::ScalarAttribute>()
                           .data()
                           .to<int64_t>());
    }
  }

  if (values.size() > 0 && sym_dims.size() <= 1) {
    std::vector<symbol::DimExpr> data;
    for (const auto &value : values) {
      data.emplace_back(symbol::DimExpr(value));
    }
    symbol::ShapeOrDataDimExprs shape_data{
        symbol::TensorShapeOrDataDimExprs(sym_dims, data)};
    infer_context->SetShapeOrDataForValue(op->result(0), shape_data);
    return true;
  }

  symbol::ShapeOrDataDimExprs shape_data{
      symbol::TensorShapeOrDataDimExprs(sym_dims)};
  infer_context->SetShapeOrDataForValue(op->result(0), shape_data);
  return true;
}

bool AssignValue_OpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  return AssignValueOpInferSymbolicShape(op, infer_context);
}

bool CudnnLstmOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const std::vector<symbol::DimExpr> &x_shape = x_shape_or_data.shape();
  const auto &init_h_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));
  const std::vector<symbol::DimExpr> &init_h_shape =
      init_h_shape_or_data.shape();
  const auto &init_c_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(2));
  const std::vector<symbol::DimExpr> &init_c_shape =
      init_c_shape_or_data.shape();
  bool is_bidirec = op->attribute<pir::BoolAttribute>("is_bidirec").data();
  int hidden_size = op->attribute<pir::Int32Attribute>("hidden_size").data();
  size_t ndims = init_c_shape.size();

  PADDLE_ENFORCE_EQ(x_shape.size(),
                    3,
                    common::errors::InvalidArgument(
                        "The rank of Input in CudnnLSTM  must be 3. But "
                        "received Input's rank is %d.",
                        x_shape.size()));
  PADDLE_ENFORCE_EQ(init_h_shape.size(),
                    3,
                    common::errors::InvalidArgument(
                        "The rank of InitH in CudnnLSTM  must be 3. But "
                        "received InitH's rank is %d.",
                        init_h_shape.size()));

  if (op->operand_source(5)) {
    const auto &sequence_length_shape_or_data =
        infer_context->GetShapeOrDataForValue(op->operand_source(5));
    const std::vector<symbol::DimExpr> &sequence_length_shape =
        sequence_length_shape_or_data.shape();
    infer_context->AddEqualCstr(x_shape[1], sequence_length_shape[0]);
  }

  infer_context->AddEqualCstr(x_shape[1], init_h_shape[1]);
  for (size_t i = 0; i < ndims; ++i) {
    infer_context->AddEqualCstr(init_c_shape[i], init_h_shape[i]);
  }
  std::vector<symbol::DimExpr> out_shape = x_shape;
  out_shape[2] = is_bidirec ? symbol::DimExpr{hidden_size * 2}
                            : symbol::DimExpr{hidden_size};
  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(out_shape)});
  infer_context->SetShapeOrDataForValue(
      op->result(1),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(init_c_shape)});
  infer_context->SetShapeOrDataForValue(
      op->result(2),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(init_h_shape)});
  return true;
}

bool DataOpInferSymbolicShape(pir::Operation *op,
                              pir::InferSymbolicShapeContext *infer_context) {
  std::string name =
      op->attributes().at("name").dyn_cast<pir::StrAttribute>().AsString();
  const std::vector<symbol::DimExpr> sym_dims =
      details::GetSymShapeForInputValue(name, op->result(0), infer_context);

  auto IsNumelLEKMaxRank = [](pir::Value value) {
    const pir::DenseTensorType &tensor_type =
        value.type().dyn_cast<pir::DenseTensorType>();
    const auto &dims = tensor_type.dims();
    if (dims.size() == 0) return true;
    if (dims.size() != 1) return false;
    if (dims[0] >= 1 && dims[0] <= ::common::DDim::kMaxRank) {
      return true;
    }
    return false;
  };

  auto IsIntType = [&](pir::Value value) {
    const auto &dtype = value.type().dyn_cast<pir::DenseTensorType>().dtype();
    return dtype.isa<pir::Int32Type>() || dtype.isa<pir::Int64Type>();
  };

  auto CreateSymForEachNumel = [&](pir::Value value) -> decltype(auto) {
    const pir::DenseTensorType &tensor_type =
        value.type().dyn_cast<pir::DenseTensorType>();
    const auto &dims = tensor_type.dims();
    const int64_t numel = ::common::product(dims);
    std::vector<symbol::DimExpr> data;
    for (int64_t i = 0; i < numel; ++i) {
      data.push_back(symbol::DimExpr(infer_context->GetNextSymName()));
    }
    return data;
  };

  const auto &shape_or_data = [&]() {
    if (IsNumelLEKMaxRank(op->result(0)) && IsIntType(op->result(0))) {
      return symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(
          sym_dims, CreateSymForEachNumel(op->result(0)))};
    } else {
      return symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(sym_dims)};
    }
  }();

  infer_context->SetShapeOrDataForValue(op->result(0), shape_or_data);

  return true;
}

bool EmptyOpInferSymbolicShape(pir::Operation *op,
                               pir::InferSymbolicShapeContext *infer_context) {
  const auto &shape_gen_op = op->operand_source(0).defining_op();
  if (shape_gen_op->isa<paddle::dialect::FullIntArrayOp>()) {
    std::vector<int64_t> shape = details::GetVectorAttr(
        shape_gen_op->dyn_cast<paddle::dialect::FullIntArrayOp>(), "value");
    std::vector<symbol::DimExpr> sym_dims;
    sym_dims.reserve(shape.size());
    for (const int64_t &dim : shape) {
      sym_dims.emplace_back(symbol::DimExpr(dim));
    }

    symbol::ShapeOrDataDimExprs shape_data{
        symbol::TensorShapeOrDataDimExprs(sym_dims)};
    infer_context->SetShapeOrDataForValue(op->result(0), shape_data);
    return true;

  } else {
    pir::Value operand_source = op->operand_source(0);
    const symbol::ShapeOrDataDimExprs &operand_shape_or_data =
        infer_context->GetShapeOrDataForValue(operand_source);
    PADDLE_ENFORCE_EQ(
        operand_shape_or_data.data().has_value(),
        true,
        common::errors::InvalidArgument(
            "The data of input dim_expr shape is null. When input of empty op "
            "is a tensor, the data of input dim_expr shape must have value."));

    infer_context->SetShapeOrDataForValue(
        op->result(0),
        symbol::TensorShapeOrDataDimExprs{
            operand_shape_or_data.data().value()});
    return true;
  }
}

bool EyeOpInferSymbolicShape(pir::Operation *op,
                             pir::InferSymbolicShapeContext *infer_context) {
  std::vector<symbol::DimExpr> out_shape;
  symbol::DimExpr num_columns_dim;
  symbol::DimExpr num_rows_dim;
  if (op->HasAttribute("num_rows")) {
    int num_rows_int = op->attribute<pir::Int64Attribute>("num_rows").data();
    num_rows_dim = symbol::DimExpr(num_rows_int);
  } else if (op->operand_source(0)) {
    const auto &num_rows_shape_or_data =
        infer_context->GetShapeOrDataForValue(op->operand_source(0));
    if (num_rows_shape_or_data.data().has_value()) {
      num_rows_dim =
          symbol::DimExpr(num_rows_shape_or_data.data().value().at(0));
    } else {
      num_rows_dim = symbol::DimExpr(infer_context->GetNextSymName());
    }
  }

  if (op->HasAttribute("num_columns")) {
    int num_columns_int =
        op->attribute<pir::Int64Attribute>("num_columns").data();
    if (num_columns_int == -1) {
      num_columns_dim = num_rows_dim;
    } else {
      num_columns_dim = symbol::DimExpr(num_columns_int);
    }
  } else if (op->operand_source(1)) {
    const auto &num_columns_shape_or_data =
        infer_context->GetShapeOrDataForValue(op->operand_source(1));
    if (num_columns_shape_or_data.data().has_value()) {
      num_columns_dim =
          symbol::DimExpr(num_columns_shape_or_data.data().value().at(0));
    } else {
      num_columns_dim = symbol::DimExpr(infer_context->GetNextSymName());
    }
  }
  out_shape.emplace_back(num_rows_dim);
  out_shape.emplace_back(num_columns_dim);

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(out_shape)});
  return true;
}
bool FeedOpInferSymbolicShape(pir::Operation *op,
                              pir::InferSymbolicShapeContext *infer_context) {
  std::string name =
      op->attributes().at("name").dyn_cast<pir::StrAttribute>().AsString();
  const auto &symbolic_shape =
      details::GetSymShapeForInputValue(name, op->result(0), infer_context);
  infer_context->SetShapeOrDataForValue(
      op->result(0), symbol::TensorShapeOrDataDimExprs(symbolic_shape));
  return true;
}

bool FullOpInferSymbolicShape(pir::Operation *op,
                              pir::InferSymbolicShapeContext *infer_context) {
  const auto &attributes = op->attributes();

  const std::vector<symbol::DimExpr> shape = [&] {
    pir::Attribute attr_shape = attributes.at("shape");
    const paddle::dialect::IntArrayAttribute &attr =
        attr_shape.dyn_cast<paddle::dialect::IntArrayAttribute>();
    const auto &shape_vec = attr.data().GetData();
    std::vector<symbol::DimExpr> shape(shape_vec.begin(), shape_vec.end());
    return shape;
  }();

  const auto shape_data = [&]() -> symbol::TensorShapeOrDataDimExprs {
    const auto &value_scalar = attributes.at("value")
                                   .dyn_cast<paddle::dialect::ScalarAttribute>()
                                   .data();
    // NOTE(Aurelius84): to<int64_t> is a risky operation when Scalar's dtype is
    // not int32/int64. However, we found Full's Value could be like '3.0' but
    // used as int.
    const int64_t value = value_scalar.to<int64_t>();
    // skip computing data when value is not equal a integer
    if (value_scalar.to<float>() - value > 1e-6) {
      return symbol::TensorShapeOrDataDimExprs(shape);
    }

    const size_t shape_size = shape.size();
    // NOTE(Aurelius84): When shape.size()==1, a new std::vector<int64_t> with
    // length = shape[0] will be constructed, but not all cases are used for
    // ShapeAnalysis. Considering MAX_RANK < 9 in Paddle, we limit it below
    // DATA_MAX_LENGTH = 128 and will not create this vector once length >
    // DATA_MAX_LENGTH.
    constexpr int64_t DATA_MAX_LENGTH = 128;
    if (shape_size == 0U) {
      std::vector<symbol::DimExpr> data{value};
      return symbol::TensorShapeOrDataDimExprs(shape, data);
    } else if (shape_size == 1U &&
               shape[0].template Get<int64_t>() <= DATA_MAX_LENGTH) {
      std::vector<symbol::DimExpr> data(shape[0].template Get<int64_t>(),
                                        symbol::DimExpr(value));
      return symbol::TensorShapeOrDataDimExprs(shape, data);
    } else {
      return symbol::TensorShapeOrDataDimExprs(shape);
    }
  }();

  infer_context->SetShapeOrDataForValue(
      op->result(0), symbol::ShapeOrDataDimExprs(shape_data));
  return true;
}

bool Full_OpInferSymbolicShape(pir::Operation *op,
                               pir::InferSymbolicShapeContext *infer_context) {
  return FullOpInferSymbolicShape(op, infer_context);
}

bool FullIntArrayOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &attributes = op->attributes();
  pir::Attribute attr_value = attributes.at("value");
  const auto &vec = attr_value.dyn_cast<pir::ArrayAttribute>().AsVector();

  const std::vector<symbol::DimExpr> data = [&] {
    std::vector<symbol::DimExpr> data;
    for (auto item : vec) {
      int64_t i = item.dyn_cast<pir::Int64Attribute>().data();
      data.push_back(symbol::DimExpr(i));
    }
    return data;
  }();

  const std::vector<symbol::DimExpr> shape{std::int64_t(vec.size())};

  symbol::ShapeOrDataDimExprs shape_data{
      symbol::TensorShapeOrDataDimExprs(shape, data)};

  pir::Value res = op->result(0);
  infer_context->SetShapeOrDataForValue(res, shape_data);
  return true;
}

bool GaussianOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &shape_gen_op = op->operand_source(0).defining_op();

  if (shape_gen_op->isa<paddle::dialect::FullIntArrayOp>()) {
    std::vector<int64_t> shape = details::GetVectorAttr(
        shape_gen_op->dyn_cast<paddle::dialect::FullIntArrayOp>(), "value");
    std::vector<symbol::DimExpr> sym_dims;
    sym_dims.reserve(shape.size());
    for (const int64_t &dim : shape) {
      sym_dims.emplace_back(symbol::DimExpr(dim));
    }

    symbol::ShapeOrDataDimExprs shape_data{
        symbol::TensorShapeOrDataDimExprs(sym_dims)};
    infer_context->SetShapeOrDataForValue(op->result(0), shape_data);
    return true;

  } else {
    pir::Value operand_source = op->operand_source(0);
    const symbol::ShapeOrDataDimExprs &operand_shape_or_data =
        infer_context->GetShapeOrDataForValue(operand_source);
    PADDLE_ENFORCE_EQ(
        operand_shape_or_data.data().has_value(),
        true,
        common::errors::InvalidArgument(
            "The data of input dim_expr shape is null. When input of empty op "
            "is a tensor, the data of input dim_expr shape must have value."));

    infer_context->SetShapeOrDataForValue(
        op->result(0),
        symbol::TensorShapeOrDataDimExprs{
            operand_shape_or_data.data().value()});
    return true;
  }
}

bool RandpermOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  int n = op->attribute<pir::Int32Attribute>("n").data();
  std::vector<symbol::DimExpr> out_shape = {n};
  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(out_shape)});

  return true;
}

bool RandintOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &shape_gen_op = op->operand_source(0).defining_op();

  if (shape_gen_op->isa<paddle::dialect::FullIntArrayOp>()) {
    std::vector<int64_t> shape = details::GetVectorAttr(
        shape_gen_op->dyn_cast<paddle::dialect::FullIntArrayOp>(), "value");
    std::vector<symbol::DimExpr> sym_dims;
    sym_dims.reserve(shape.size());
    for (const int64_t &dim : shape) {
      sym_dims.emplace_back(symbol::DimExpr(dim));
    }

    symbol::ShapeOrDataDimExprs shape_data{
        symbol::TensorShapeOrDataDimExprs(sym_dims)};
    infer_context->SetShapeOrDataForValue(op->result(0), shape_data);
    return true;

  } else {
    PADDLE_ENFORCE_EQ(
        infer_context->HasShapeOrDataForValue(op->operand_source(0)),
        true,
        common::errors::PreconditionNotMet(
            "Shape is not comes from FullIntArrayOp "
            "should have shape or data"));

    auto shape_dim_expr =
        infer_context->GetShapeOrDataForValue(op->operand_source(0));
    ExprVec target_shape = paddle::dialect::details::GetOrCreateExprVecFromData(
        shape_dim_expr, infer_context);

    symbol::ShapeOrDataDimExprs shape_data{
        symbol::TensorShapeOrDataDimExprs(target_shape)};

    infer_context->SetShapeOrDataForValue(op->result(0), shape_data);
    return true;
  }
}

bool ReadFileOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  symbol::DimExpr unique_dim_sym = infer_context->GetNextSymName();

  const std::vector<symbol::DimExpr> &out_shape = [&] {
    std::vector<symbol::DimExpr> shape;
    shape.emplace_back(symbol::DimExpr(1));
    shape.emplace_back(unique_dim_sym);
    return shape;
  }();

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(out_shape)});

  return true;
}

bool PRecvOpInferSymbolicShape(pir::Operation *op,
                               pir::InferSymbolicShapeContext *infer_context) {
  const int ring_id = op->attribute<pir::Int32Attribute>("ring_id").data();
  const bool dynamic_shape =
      op->attribute<pir::BoolAttribute>("dynamic_shape").data();
  const int peer = op->attribute<pir::Int32Attribute>("peer").data();

  PADDLE_ENFORCE_GE(
      peer,
      0,
      common::errors::InvalidArgument(
          "The peer (%d) for p_recv op must be non-negative.", peer));

  PADDLE_ENFORCE_GE(
      ring_id,
      0,
      common::errors::InvalidArgument(
          "The ring_id (%d) for p_recv op must be non-negative.", ring_id));

  const std::vector<int> out_shape =
      paddle::dialect::details::GetVectorAttr<int>(op, "out_shape");
  if (!dynamic_shape) {
    PADDLE_ENFORCE_GE(out_shape.size(),
                      1,
                      common::errors::InvalidArgument(
                          "The size of the output shape must be greater than 0 "
                          "but the value given is %d.",
                          out_shape.size()));

    std::vector<symbol::DimExpr> output_shape;
    for (size_t i = 0; i < out_shape.size(); ++i) {
      PADDLE_ENFORCE_GE(out_shape[i],
                        1,
                        common::errors::InvalidArgument(
                            "The shape attribute for p_recv must be set "
                            "explicitly, but the %dth element is %d which "
                            "is less than 1. Or dynamic_shape should be set to "
                            "True for both p_send and p_recv.",
                            i,
                            out_shape[i]));
      output_shape.push_back(symbol::DimExpr(out_shape[i]));
    }

    infer_context->SetShapeOrDataForValue(
        op->result(0),
        symbol::ShapeOrDataDimExprs{
            symbol::TensorShapeOrDataDimExprs(output_shape)});
  }

  return true;
}

bool SeedOpInferSymbolicShape(pir::Operation *op,
                              pir::InferSymbolicShapeContext *infer_context) {
  std::vector<symbol::DimExpr> dims = {symbol::DimExpr(1)};

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(dims)});

  return true;
}

bool TrilIndicesOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &attributes = op->attributes();
  int rows = attributes.at("rows").dyn_cast<pir::Int32Attribute>().data();
  int cols = attributes.at("cols").dyn_cast<pir::Int32Attribute>().data();
  int offset = attributes.at("offset").dyn_cast<pir::Int32Attribute>().data();

  const auto &out_sym_shape = [&] {
    std::vector<symbol::DimExpr> out_sym_shape;
    auto n_first_row =
        offset > 0 ? std::min<int64_t>(cols, 1 + offset) : rows + offset > 0;
    auto n_last_row =
        std::max<int64_t>(0, std::min<int64_t>(cols, rows + offset));
    auto n_row_all =
        std::max<int64_t>(0, std::min<int64_t>(rows, rows + offset));
    auto n_row_trapezoid = (n_last_row - n_first_row + 1);
    auto tril_size = (n_first_row + n_last_row) * n_row_trapezoid >> 1;
    auto diff_row = n_row_all - n_row_trapezoid;
    if (diff_row > 0) {
      tril_size += diff_row * cols;
    }
    out_sym_shape.emplace_back(std::int64_t(2));
    out_sym_shape.emplace_back(std::int64_t(tril_size));
    return out_sym_shape;
  }();

  symbol::ShapeOrDataDimExprs shape_data{
      symbol::TensorShapeOrDataDimExprs(out_sym_shape)};
  infer_context->SetShapeOrDataForValue(op->result(0), shape_data);
  return true;
}
bool TriuIndicesOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &attributes = op->attributes();
  int row = attributes.at("row").dyn_cast<pir::Int32Attribute>().data();
  int col = attributes.at("col").dyn_cast<pir::Int32Attribute>().data();
  int offset = attributes.at("offset").dyn_cast<pir::Int32Attribute>().data();

  const auto &out_sym_shape = [&] {
    std::vector<symbol::DimExpr> out_sym_shape;
    offset = offset - 1;
    auto n_first_row =
        offset > 0 ? std::min<int64_t>(col, 1 + offset) : row + offset > 0;
    auto n_last_row =
        std::max<int64_t>(0, std::min<int64_t>(col, row + offset));
    auto n_row_all = std::max<int64_t>(0, std::min<int64_t>(row, row + offset));
    auto n_row_trapezoid = (n_last_row - n_first_row + 1);
    auto tril_size = (n_first_row + n_last_row) * n_row_trapezoid >> 1;
    auto diff_row = n_row_all - n_row_trapezoid;
    if (diff_row > 0) {
      tril_size += diff_row * col;
    }
    out_sym_shape.emplace_back(std::int64_t(2));
    out_sym_shape.emplace_back(std::int64_t(row * col - tril_size));
    return out_sym_shape;
  }();

  symbol::ShapeOrDataDimExprs shape_data{
      symbol::TensorShapeOrDataDimExprs(out_sym_shape)};
  infer_context->SetShapeOrDataForValue(op->result(0), shape_data);
  return true;
}

bool TruncatedGaussianRandomOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const std::vector<int> shape =
      paddle::dialect::details::GetVectorAttr<int>(op, "shape");
  std::vector<symbol::DimExpr> out_shape;
  out_shape.reserve(shape.size());
  for (const int &dim : shape) {
    out_shape.emplace_back(symbol::DimExpr(static_cast<int64_t>(dim)));
  }

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(out_shape)});
  return true;
}

bool UniformOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  return GaussianOpInferSymbolicShape(op, infer_context);
}

}  // namespace paddle::dialect
