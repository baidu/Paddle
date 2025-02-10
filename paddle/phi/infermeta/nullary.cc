/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/infermeta/nullary.h"

namespace phi {

void ArangeInferMeta(const Scalar& start,
                     const Scalar& end,
                     const Scalar& step,
                     DataType dtype,
                     MetaTensor* out) {
  if (!start.FromTensor() && !end.FromTensor() && !step.FromTensor()) {
    double start_value = start.to<double>();
    double end_value = end.to<double>();
    double step_value = step.to<double>();
    int numel =
        static_cast<int>(std::ceil((end_value - start_value) / step_value));
    out->set_dims(common::make_ddim(std::vector<int64_t>(1, numel)));
  } else {
    out->set_dims({-1});
  }
  out->set_dtype(dtype);
}

void AssignValueInferMeta(const std::vector<int>& shape,
                          DataType dtype,
                          MetaTensor* out) {
  out->set_dims(common::make_ddim(shape));
  out->set_dtype(dtype);
}

void CommInitAllInferMeta(const std::vector<int>& devices, int ring_id) {}

void CreateArrayInferMeta(DataType dtype, MetaTensor* out) {
  out->set_dtype(dtype);
}

void CreateInferMeta(const IntArray& shape,
                     DataType dtype,
                     MetaTensor* out,
                     MetaConfig config) {
  if (config.is_runtime || !shape.FromTensor()) {
    const auto& data = shape.GetData();
    for (size_t i = 0; i < data.size(); ++i) {
      PADDLE_ENFORCE_GE(
          data[i],
          0,
          common::errors::InvalidArgument(
              "Each value of attribute 'shape' is expected to be no less "
              "than 0. But received: shape[%u] = %d; shape = [%s].",
              i,
              data[i],
              common::make_ddim(data)));
    }
  }
  CreateInferMetaBase(shape.GetData(), dtype, DataLayout::NCHW, out);
}

void CreateVecShapeInferMeta(const std::vector<int64_t>& shape,
                             DataType dtype,
                             MetaTensor* out) {
  CreateInferMetaBase(
      {static_cast<int64_t>(shape.size())}, dtype, DataLayout::NCHW, out);
}

void CreateInferMetaBase(const std::vector<int64_t>& shape,
                         DataType dtype,
                         DataLayout layout,
                         MetaTensor* out) {
  auto out_dims = common::make_ddim(shape);
  out->set_dims(out_dims);
  out->set_dtype(dtype);
  out->set_layout(layout);
}

void DataInferMeta(const std::string& name,
                   const phi::IntArray& shape,
                   phi::DataType data_type,
                   MetaTensor* out) {
  auto out_dims = common::make_ddim(shape.GetData());
  out->set_dims(out_dims);
  out->set_dtype(data_type);
}

void EyeInferMeta(const Scalar& num_rows,
                  const Scalar& num_columns,
                  DataType dtype,
                  MetaTensor* out,
                  MetaConfig config) {
  int64_t rows = 0, columns = 0;
  if (!config.is_runtime && num_rows.FromTensor()) {
    rows = -1;
  } else {
    rows = num_rows.to<int64_t>();
  }

  if (!config.is_runtime && num_columns.FromTensor()) {
    columns = -1;
  } else {
    columns = num_columns.to<int64_t>();
    if (columns == -1) columns = rows;
  }
  out->set_dims({rows, columns});
  out->set_dtype(dtype);
}

void GaussianInferMeta(const IntArray& shape,
                       float mean,
                       float std,
                       int seed,
                       DataType dtype,
                       MetaTensor* out) {
  auto out_dims = common::make_ddim(shape.GetData());
  out->set_dims(out_dims);
  out->set_dtype(dtype);
  out->set_layout(DataLayout::NCHW);
}

void PartialRecvInferMeta(int peer,
                          DataType dtype,
                          const std::vector<int>& out_shape,
                          int num,
                          int id,
                          MetaTensor* out) {
  PADDLE_ENFORCE_GE(
      peer,
      0,
      common::errors::InvalidArgument(
          "The peer (%d) for partial_recv op must be non-negative.", peer));
  PADDLE_ENFORCE_GE(num,
                    1,
                    common::errors::InvalidArgument(
                        "The num (%d) for partial_send op must >=1", num));
  PADDLE_ENFORCE_EQ(
      (id >= 0 && id < num),
      true,
      common::errors::InvalidArgument(
          "The id (%d) for partial_send op must >=0 and <num (%d)", id, num));
  PADDLE_ENFORCE_GE(out_shape.size(),
                    1,
                    common::errors::InvalidArgument(
                        "The size of the output shape must be greater than 0 "
                        "but the value given is %d.",
                        out_shape.size()));

  for (size_t i = 0; i < out_shape.size(); ++i) {
    PADDLE_ENFORCE_GE(out_shape[i],
                      1,
                      common::errors::InvalidArgument(
                          "The shape attribute for partial_recv must be set "
                          "explicitly, but the %dth element is %d which "
                          "is less than 1.",
                          i,
                          out_shape[i]));
  }
  auto out_dims = common::make_ddim(out_shape);
  int64_t numel = common::product(out_dims);
  PADDLE_ENFORCE_EQ(
      (numel % num),
      0,
      common::errors::InvalidArgument(
          "The output numel (%d) must be divisible by num(%d)", numel, num));

  out->set_dims(common::make_ddim(out_shape));
  out->set_dtype(dtype);
}

void LoadInferMeta(MetaTensor* out, MetaConfig config) {}

void RandpermInferMeta(int n, DataType dtype, MetaTensor* out) {
  out->set_dims(common::make_ddim({n}));
  out->set_dtype(dtype);
}

void UniformRandomInferMeta(const IntArray& shape,
                            DataType dtype,
                            MetaTensor* out) {
  auto out_dims = common::make_ddim(shape.GetData());
  out->set_dims(out_dims);
  out->set_dtype(dtype);
  out->set_layout(DataLayout::NCHW);
}

void RandintInferMeta(
    int low, int high, const IntArray& shape, DataType dtype, MetaTensor* out) {
  PADDLE_ENFORCE_NOT_NULL(
      out, errors::InvalidArgument("Output(Out) of RandintOp is null."));
  PADDLE_ENFORCE_LT(
      low,
      high,
      errors::InvalidArgument("randint's low must less then high, "
                              "but received: low = %d, high = %d.",
                              low,
                              high));

  auto& shape_vector = shape.GetData();

  std::vector<int64_t> tensor_shape;
  tensor_shape.reserve(shape_vector.size());
  for (auto dim : shape_vector) {
    tensor_shape.push_back(static_cast<int64_t>(dim));
  }
  out->set_dims(common::make_ddim(tensor_shape));
  out->set_dtype(dtype);
}

void PRecvInferMeta(const int peer,
                    DataType dtype,
                    const std::vector<int>& out_shape,
                    const bool dynamic_shape,
                    MetaTensor* out) {
  PADDLE_ENFORCE_GE(
      peer,
      0,
      errors::InvalidArgument(
          "The peer (%d) for p_recv op must be non-negative.", peer));

  if (!dynamic_shape) {
    PADDLE_ENFORCE_GE(out_shape.size(),
                      1,
                      errors::InvalidArgument(
                          "The size of the output shape must be greater than 0 "
                          "but the value given is %d.",
                          out_shape.size()));
    for (size_t i = 0; i < out_shape.size(); ++i) {
      PADDLE_ENFORCE_GE(out_shape[i],
                        1,
                        errors::InvalidArgument(
                            "The shape attribute for p_recv must be set "
                            "explicitly, but the %dth element is %d which "
                            "is less than 1. Or dynamic_shape should be "
                            "set to True for both p_send and p_recv.",
                            i,
                            out_shape[i]));
    }
    out->set_dims(common::make_ddim(out_shape));
  }
  out->set_dtype(dtype);
}

void PRecvArrayInferMeta(int peer,
                         DataType dtype,
                         const std::vector<int>& out_shape,
                         MetaTensor* out) {
  PADDLE_ENFORCE_GE(
      peer,
      0,
      errors::InvalidArgument(
          "The peer (%d) for p_recv op must be non-negative.", peer));

  PADDLE_ENFORCE_GE(out_shape.size(),
                    1,
                    errors::InvalidArgument(
                        "The size of the output shape must be greater than 0 "
                        "but the value given is %d.",
                        out_shape.size()));

  for (size_t i = 0; i < out_shape.size(); ++i) {
    PADDLE_ENFORCE_GE(
        out_shape[i],
        1,
        errors::InvalidArgument("The shape attribute for recv must be set "
                                "explicitly, but the %dth element is %d which "
                                "is less than 1. Or dynamic_shape should be "
                                "set to True for both p_send and p_recv.",
                                i,
                                out_shape[i]));
  }
  out->set_dtype(dtype);
}

void SeedInferMeta(int seed, MetaTensor* out) {
  out->set_dims(common::make_ddim({1}));
  out->set_dtype(DataType::INT32);
}

void TruncatedGaussianRandomInferMeta(const std::vector<int>& shape,
                                      float mean,
                                      float std,
                                      int seed,
                                      float a,
                                      float b,
                                      DataType dtype,
                                      MetaTensor* out) {
  auto out_dims = common::make_ddim(shape);
  out->set_dims(out_dims);
  out->set_dtype(dtype);
  out->set_layout(DataLayout::NCHW);
}

void TrilIndicesInferMeta(
    int rows, int cols, int offset, DataType dtype, MetaTensor* out) {
  // number of elements in the first row of the tril,bounded by [0, cols]
  auto n_first_row =
      offset > 0 ? std::min<int64_t>(cols, 1 + offset) : rows + offset > 0;
  // number of elements in the last row of the tril, bounded by [0, cols]
  auto n_last_row =
      std::max<int64_t>(0, std::min<int64_t>(cols, rows + offset));
  // number of rows, bounded by [0, rows]
  auto n_row_all = std::max<int64_t>(0, std::min<int64_t>(rows, rows + offset));
  auto n_row_trapezoid = (n_last_row - n_first_row + 1);
  // calculate # of elements in the top trapezoid
  auto tril_size = (n_first_row + n_last_row) * n_row_trapezoid >> 1;
  // calculate # of elements in the bottom rectangle if there is any
  auto diff_row = n_row_all - n_row_trapezoid;
  if (diff_row > 0) {
    tril_size += diff_row * cols;
  }
  std::vector<int64_t> tmp = {2, tril_size};
  auto out_dims = common::make_ddim(tmp);
  out->set_dims(out_dims);
  out->set_dtype(dtype);
}

void TriuIndicesInferMeta(
    int row, int col, int offset, DataType dtype, MetaTensor* out) {
  // number of elements in the first row of the tril,bounded by [0, cols]
  // use total item number minus bottom rectangle item number to get
  // the above rectangle item number
  //     triu_size = rows * cols - tril_size
  // so the `offset` need to be set as `offset-1` in order to include
  // the item on the diagonal line
  offset = offset - 1;
  auto n_first_row =
      offset > 0 ? std::min<int64_t>(col, 1 + offset) : row + offset > 0;
  // number of elements in the last row of the tril, bounded by [0, cols]
  auto n_last_row = std::max<int64_t>(0, std::min<int64_t>(col, row + offset));
  // number of rows, bounded by [0, rows]
  auto n_row_all = std::max<int64_t>(0, std::min<int64_t>(row, row + offset));
  auto n_row_trapezoid = (n_last_row - n_first_row + 1);
  // calculate # of elements in the top trapezoid
  auto tril_size = (n_first_row + n_last_row) * n_row_trapezoid >> 1;
  // calculate # of elements in the bottom rectangle if there is any
  auto diff_row = n_row_all - n_row_trapezoid;
  if (diff_row > 0) {
    tril_size += diff_row * col;
  }
  std::vector<int64_t> tmp = {2, row * col - tril_size};
  auto out_dims = common::make_ddim(tmp);
  out->set_dims(out_dims);
  out->set_dtype(dtype);
}

void ReadFileInferMeta(const std::string& filename, MetaTensor* out) {
  auto out_dims = std::vector<int>(1, -1);
  out->set_dims(phi::make_ddim(out_dims));
  out->set_dtype(phi::DataType::UINT8);
}

}  // namespace phi
