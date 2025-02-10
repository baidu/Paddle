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

#include "paddle/phi/infermeta/unary.h"

#include <algorithm>
#include <set>

#include "paddle/common/flags.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/type_traits.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/core/utils/data_type.h"
#include "paddle/phi/kernels/funcs/flatten2_utils.h"
#include "paddle/phi/kernels/funcs/hash_utils.h"
#include "paddle/phi/kernels/funcs/parse_qr_mode.h"
#include "paddle/phi/kernels/funcs/pooling.h"
#include "paddle/phi/kernels/funcs/slice_utils.h"
#include "paddle/phi/kernels/funcs/strided_slice.h"
#include "paddle/phi/kernels/funcs/unfold_functor.h"
#include "paddle/phi/kernels/funcs/unsqueeze.h"
#include "paddle/phi/kernels/impl/einsum_impl.h"

namespace phi {

namespace detail {
// Used in MatrixRankInferMeta
static DDim CheckAndGetOutputDim(const DDim& dim_x) {
  auto x_vec = common::vectorize(dim_x);
  if (x_vec.size() == 2) {
    return common::make_ddim({});
  }
  x_vec.erase(x_vec.end() - 2, x_vec.end());
  return common::make_ddim(x_vec);
}
}  // namespace detail

void AddPositionEncodingInferMeta(const MetaTensor& x,
                                  float alpha,
                                  float beta,
                                  MetaTensor* out) {
  const auto& x_dims = x.dims();
  out->set_dims(x_dims);
  out->share_lod(x);
  out->set_dtype(x.dtype());
}

void AffineGridInferMeta(const MetaTensor& input,
                         const IntArray& outputShape,
                         bool align_corners,
                         MetaTensor* output) {
  auto theta_dims = input.dims();
  bool is_from_tensor = outputShape.FromTensor();
  if (!is_from_tensor) {
    PADDLE_ENFORCE_EQ(
        theta_dims.size(),
        3,
        common::errors::InvalidArgument(
            "The input Theta's dimensions size should be 3. But received "
            "Theta's dimensions size=[%d],  Theta's dimensions=[%s].",
            theta_dims.size(),
            theta_dims));

    PADDLE_ENFORCE_GE(
        outputShape.GetData().size(),
        4,
        common::errors::InvalidArgument(
            "The size of attribute 'output_shape' in AffineGridOp should be >= "
            "4. But received output_shape's size=[%d].",
            outputShape.GetData().size()));

    PADDLE_ENFORCE_LE(
        outputShape.GetData().size(),
        5,
        common::errors::InvalidArgument(
            "The size of attribute 'output_shape' in AffineGridOp should be <= "
            "5. But received output_shape's size=[%d].",
            outputShape.GetData().size()));
  }
  PADDLE_ENFORCE_GE(theta_dims[1],
                    2,
                    common::errors::InvalidArgument(
                        "The second dimension of input 'theta' in AffineGridOp "
                        "should be >= 2. "
                        "But received second dimension=[%d], dimensions=[%s]",
                        theta_dims[1],
                        theta_dims));

  PADDLE_ENFORCE_LE(theta_dims[1],
                    3,
                    common::errors::InvalidArgument(
                        "The second dimension of input 'theta' in AffineGridOp "
                        "should be <= 3. "
                        "But received second dimension=[%d], dimensions=[%s]",
                        theta_dims[1],
                        theta_dims));

  PADDLE_ENFORCE_GE(theta_dims[2],
                    3,
                    common::errors::InvalidArgument(
                        "The third dimension of input 'theta' in AffineGridOp "
                        "should be >= 3. "
                        "But received third dimension=[%d], dimensions=[%s]",
                        theta_dims[2],
                        theta_dims));

  PADDLE_ENFORCE_LE(theta_dims[2],
                    4,
                    common::errors::InvalidArgument(
                        "The third dimension of input 'theta' in AffineGridOp "
                        "should be <= 4. "
                        "But received third dimension=[%d], dimensions=[%s]",
                        theta_dims[2],
                        theta_dims));
  if (outputShape.GetData().size() == 4 && !is_from_tensor) {
    // N * H * W * 2
    output->set_dims(common::make_ddim({theta_dims[0], -1, -1, 2}));
  } else {
    // N * D * H * W * 3
    output->set_dims(common::make_ddim({theta_dims[0], -1, -1, -1, 3}));
  }
  output->set_dtype(input.dtype());
  output->share_lod(input);
}

void AllGatherInferMeta(const MetaTensor& x, int nranks, MetaTensor* out) {
  auto dim = x.dims();
  dim[0] = dim[0] * nranks;
  if (dim[0] < 0) dim[0] = -1;
  out->set_dtype(x.dtype());
  out->set_dims(dim);
}

void AllReduceInferMeta(const MetaTensor& x, MetaTensor* out) {
  out->set_dtype(x.dtype());
  out->set_dims(x.dims());
}

void AllToAllInferMeta(const MetaTensor& x, MetaTensor* out) {
  auto dim = x.dims();
  if (dim[0] < 0) dim[0] = -1;
  out->set_dtype(x.dtype());
  out->set_dims(dim);
}

void AnchorGeneratorInferMeta(const MetaTensor& input,
                              const std::vector<float>& anchor_sizes,
                              const std::vector<float>& aspect_ratios,
                              const std::vector<float>& variances,
                              const std::vector<float>& stride,
                              float offset,
                              MetaTensor* anchors,
                              MetaTensor* variances_out) {
  const auto& input_dims = input.dims();
  PADDLE_ENFORCE_EQ(
      input_dims.size(),
      4,
      common::errors::InvalidArgument("The layout of input is NCHW."));

  size_t num_anchors = aspect_ratios.size() * anchor_sizes.size();
  std::vector<int64_t> dim_vec(4);
  dim_vec[0] = input_dims[2];
  dim_vec[1] = input_dims[3];
  dim_vec[2] = static_cast<int64_t>(num_anchors);
  dim_vec[3] = 4;
  anchors->set_dims(common::make_ddim(dim_vec));
  anchors->set_dtype(input.dtype());
  variances_out->set_dims(common::make_ddim(dim_vec));
  variances_out->set_dtype(input.dtype());
}

void ArrayLengthInferMeta(const MetaTensor& x, MetaTensor* out) {
  out->set_dtype(phi::DataType::INT64);
  out->set_dims(common::make_ddim({1}));
}

void ArrayToTensorInferMeta(const MetaTensor& x,
                            int axis,
                            bool use_stack,
                            MetaTensor* out,
                            MetaTensor* out_index,
                            MetaConfig config) {
  if (config.is_runtime) {
    PADDLE_ENFORCE_EQ(
        (x.is_tensor_array()),
        true,
        common::errors::InvalidArgument(
            "The dtype of 1st input in ArrayToTensor must be TensorArray, but "
            "received [%s]",
            x.dtype()));
    size_t x_size = x.size();
    std::vector<phi::DDim> vec_dims;
    for (size_t i = 0; i < x_size; ++i) {
      phi::DDim dims = x.dims(i);
      if (!vec_dims.empty()) {
        PADDLE_ENFORCE_EQ(
            (vec_dims.back() == dims),
            true,
            common::errors::InvalidArgument(
                "The dims of input in ArrayToTensor must be equal, but "
                "received"));
      }
      vec_dims.push_back(dims);
    }
    if (vec_dims.empty()) return;
    auto dim_vec = common::vectorize<int>(vec_dims[0]);
    if (use_stack) {
      dim_vec.insert(dim_vec.begin() + axis, dim_vec.size() * dim_vec[axis]);
    } else {
      dim_vec[axis] = dim_vec.size() * dim_vec[axis];
    }
    phi::DDim dims = common::make_ddim(dim_vec);
    out->set_dims(dims);
    return;
  }
  auto dims = x.dims();
  // if the shape is empty
  if (dims == common::make_ddim({0UL})) return;
  // otherwise, suppose the shape of array is the shape of tensor in the
  // array, which is consistent with what tensor_array_read_write dose
  if (use_stack) {
    auto dim_vec = common::vectorize<int>(dims);
    // use -1 for the stack dim size
    dim_vec.insert(dim_vec.begin() + axis, -1);
    dims = common::make_ddim(dim_vec);
  } else {
    // use -1 for the concat dim size
    dims[axis] = -1;
  }
  out->set_dims(dims);
  out->set_dtype(x.dtype());
  out_index->set_dtype(DataType::INT32);
  out_index->set_dims(common::make_ddim({-1}));
}

void TensorToArrayInferMeta(const MetaTensor& x,
                            const MetaTensor& out_grad,
                            int axis,
                            bool use_stack,
                            MetaTensor* x_grad) {
  x_grad->set_dtype(x.dtype());
  x_grad->set_layout(x.layout());
}

void ArgMinMaxInferMeta(const MetaTensor& x,
                        const Scalar& axis,
                        bool keepdims,
                        bool flatten,
                        DataType dtype,
                        MetaTensor* out,
                        MetaConfig config) {
  PADDLE_ENFORCE_EQ(
      (dtype == DataType::UNDEFINED || dtype == DataType::INT32 ||
       dtype == DataType::INT64),
      true,
      common::errors::InvalidArgument(
          "The attribute of dtype in argmin/argmax must be [%s] or [%s], but "
          "received [%s]",
          DataTypeToString(DataType::INT32),
          DataTypeToString(DataType::INT64),
          DataTypeToString(dtype)));

  if (!config.is_runtime && axis.FromTensor()) {
    std::vector<int64_t> vec;
    if (flatten) {
      if (keepdims) {  // NOLINT
        vec = std::vector<int64_t>(x.dims().size(), -1);
      } else {
        vec = {};
      }
    } else {
      if (keepdims) {
        vec = std::vector<int64_t>(x.dims().size(), -1);
      } else {
        vec = std::vector<int64_t>(x.dims().size() - 1, -1);
      }
    }
    out->set_dims(common::make_ddim(vec));
    if (dtype == DataType::INT32 || dtype == DataType::INT64) {
      out->set_dtype(dtype);
    }
    return;
  }
  auto int_axis = axis.to<int64_t>();
  const auto& x_dims = x.dims();

  auto x_rank = x.dims().size();
  if (x_rank > 0) {
    PADDLE_ENFORCE_GE(int_axis,
                      -x_rank,
                      common::errors::InvalidArgument(
                          "'axis'(%d) must be greater than or equal to"
                          " -Rank(X)(%d).",
                          int_axis,
                          -x_rank));
    PADDLE_ENFORCE_LT(
        int_axis,
        x_rank,
        common::errors::InvalidArgument(
            "'axis'(%d) must be less than Rank(X)(%d) of Input(X).",
            int_axis,
            x_rank));
  } else {
    // 0-dim tensor
    PADDLE_ENFORCE_EQ(int_axis == 0 || int_axis == -1,
                      true,
                      common::errors::InvalidArgument(
                          "'axis'(%d) must be 0 or -1 if input tensor is "
                          "0-dim.",
                          int_axis));
  }

  if (int_axis < 0) int_axis += x_rank;

  if (config.is_runtime) {
    if (dtype == DataType::INT32) {
      int64_t all_element_num = 0;
      if (flatten) {
        all_element_num = common::product(x_dims);
      } else {
        all_element_num = x_dims[static_cast<int>(int_axis)];
      }
      PADDLE_ENFORCE_LE(
          all_element_num,
          INT_MAX,
          common::errors::InvalidArgument(
              "The element num of the argmin/argmax input at axis is "
              "%d, is larger than int32 maximum value:%d, you must "
              "set the dtype of argmin/argmax to 'int64'.",
              all_element_num,
              INT_MAX));
    }
  }

  std::vector<int64_t> vec;
  if (flatten) {
    if (keepdims) {  // NOLINT
      vec = std::vector<int64_t>(x.dims().size(), 1);
    } else {
      vec = {};
    }
  } else {
    for (int64_t i = 0; i < int_axis; i++)
      vec.emplace_back(x_dims[static_cast<int>(i)]);
    if (keepdims) {
      vec.emplace_back(static_cast<int64_t>(1));
    }
    for (int64_t i = int_axis + 1; i < x_rank; i++)
      vec.emplace_back(x_dims[static_cast<int>(i)]);
  }

  out->set_dims(common::make_ddim(vec));
  if (dtype == DataType::INT32 || dtype == DataType::INT64) {
    out->set_dtype(dtype);
  }
}

void ArgsortInferMeta(const MetaTensor& input,
                      int axis,
                      bool descending,
                      bool stable,
                      MetaTensor* output,
                      MetaTensor* indices) {
  auto in_dims = input.dims();
  auto num_dims = in_dims.size();
  if (num_dims > 0) {
    PADDLE_ENFORCE_GE(axis,
                      -num_dims,
                      common::errors::InvalidArgument(
                          "'axis'(%d) must be greater than or equal to"
                          " -num_dims(%d).",
                          axis,
                          -num_dims));
    PADDLE_ENFORCE_LT(
        axis,
        num_dims,
        common::errors::InvalidArgument(
            "'axis'(%d) must be less than num_dims(%d).", axis, num_dims));
  } else {  // 0-dim tensor
    PADDLE_ENFORCE_EQ(
        axis == 0 || axis == -1,
        1,
        common::errors::InvalidArgument(
            "'axis'(%d) must be 0 or -1 if input tensor is 0-dim.", axis));
  }

  output->share_dims(input);
  output->set_dtype(input.dtype());
  indices->share_dims(input);
  indices->set_dtype(DataType::INT64);
  output->share_lod(input);
  indices->share_lod(input);
}

void AsRealInferMeta(const MetaTensor& input, MetaTensor* output) {
  auto out_dims_v = common::vectorize(input.dims());
  out_dims_v.push_back(2);
  auto out_dims = common::make_ddim(out_dims_v);
  output->set_dims(out_dims);
  output->share_lod(input);
  output->set_dtype(dtype::ToReal(input.dtype()));
}

void AsComplexInferMeta(const MetaTensor& input, MetaTensor* output) {
  auto in_dims = input.dims();
  const int input_rank = in_dims.size();
  PADDLE_ENFORCE_GE(
      input_rank,
      1,
      common::errors::InvalidArgument(
          "The rank of input(X) is less than 1. "
          "Expected the rank of input(X) to be equal to or greater than 1."
          "But received rank of input(X) = %d",
          input_rank));
  const int last_dim_size = static_cast<int>(in_dims[input_rank - 1]);
  PADDLE_ENFORCE_EQ(
      last_dim_size,
      2,
      common::errors::InvalidArgument(
          "The size of the last dimension of input(X)"
          "does not equals 2."
          "Expected the size of last dimension of input(X) to be 2."
          "But received %d",
          last_dim_size));

  const phi::DDim out_dims(in_dims.Get(), input_rank - 1);
  output->set_dims(out_dims);
  output->share_lod(input);
  output->set_dtype(dtype::ToComplex(input.dtype()));
}

void BarrierInferMeta(const MetaTensor& x, MetaTensor* out) {}

void BatchSizeLikeInferMeta(const MetaTensor& x,
                            const std::vector<int>& shape,
                            int x_batch_size_dim,
                            int out_batch_size_dim,
                            MetaTensor* out) {
  PADDLE_ENFORCE_GT(
      shape.size(),
      0UL,
      common::errors::InvalidArgument(
          "Shape size must be larger than 0, but received: %s.", shape.size()));
  std::vector<int64_t> shape_int64(shape.size(), 0);
  std::transform(shape.begin(), shape.end(), shape_int64.begin(), [](int a) {
    return static_cast<int64_t>(a);
  });
  auto output_dim = common::make_ddim(shape_int64);

  int input_dim_size = static_cast<int>(x.dims().size());
  PADDLE_ENFORCE_GE(
      x_batch_size_dim,
      0,
      common::errors::InvalidArgument("Input dimension index must be larger "
                                      "equal than 0, but received: %s.",
                                      x_batch_size_dim));
  PADDLE_ENFORCE(input_dim_size > x_batch_size_dim || input_dim_size == -1,
                 common::errors::InvalidArgument(
                     "Input dimension size must be larger than "
                     "input dimension index, but received input "
                     "dimension size: %s, input dimension index: %s.",
                     input_dim_size,
                     x_batch_size_dim));

  int output_dim_size = static_cast<int>(shape.size());
  PADDLE_ENFORCE_GE(
      out_batch_size_dim,
      0,
      common::errors::InvalidArgument("Output dimension index must be larger "
                                      "equal than 0, but received: %s.",
                                      out_batch_size_dim));
  PADDLE_ENFORCE_GT(
      output_dim_size,
      out_batch_size_dim,
      common::errors::InvalidArgument(
          "Output dimension size must be larger than output dimension index, "
          "but received output dimension size: %s, output dimension index: "
          "%s.",
          output_dim_size,
          out_batch_size_dim));

  output_dim[out_batch_size_dim] = x.dims()[x_batch_size_dim];
  out->set_dims(output_dim);
}

void BipartiteMatchInferMeta(const MetaTensor& dist_mat,
                             const std::string& match_type,
                             float dist_threshold,
                             MetaTensor* col_to_row_match_indices,
                             MetaTensor* col_to_row_match_dist) {
  const auto& dims = dist_mat.dims();
  PADDLE_ENFORCE_EQ(
      dims.size(),
      2,
      common::errors::InvalidArgument("The rank of Input(DistMat) must be 2."));

  col_to_row_match_indices->set_dims(dims);
  col_to_row_match_indices->set_dtype(DataType::INT32);
  col_to_row_match_dist->set_dims(dims);
  col_to_row_match_dist->set_dtype(dist_mat.dtype());
}

void CastInferMeta(const MetaTensor& x, DataType out_dtype, MetaTensor* out) {
  out->set_dims(x.dims());
  out->set_layout(x.layout());
  out->share_lod(x);
  // In inplace case, setting the dtype of out will reset the dtype of x at the
  // same time, which will cause bugs, so move the dtype setting of out to the
  // kernel
  if (!(out->is_same_tensor(x))) {
    out->set_dtype(out_dtype);
  }
}

void CConcatInferMeta(const MetaTensor& x, int nranks, MetaTensor* out) {
  phi::DDim dim = x.dims();
  dim[dim.size() - 1] = dim[dim.size() - 1] * nranks;
  if (dim[dim.size() - 1] < 0) dim[dim.size() - 1] = -1;
  out->set_dims(dim);
  out->set_layout(x.layout());
  out->set_dtype(x.dtype());
}

void CholeskyInferMeta(const MetaTensor& x, bool upper, MetaTensor* out) {
  auto dims = x.dims();
  auto rank = dims.size();
  PADDLE_ENFORCE_GE(rank,
                    2,
                    errors::InvalidArgument(
                        "The Input(X) should have at least 2 dimensions. But "
                        "received a %d dimension tensor.",
                        rank));
  PADDLE_ENFORCE_EQ(
      dims[rank - 2],
      dims[rank - 1],
      errors::InvalidArgument(
          "The inner-most 2 dimensions of Input(X) all should be symmetric "
          "positive-definite matrices and have the same size. But received "
          "X's shape[-2] = %d and shape[-1] = %d.",
          dims[rank - 2],
          dims[rank - 1]));
  out->set_dims(x.dims());
  out->set_dtype(x.dtype());
}

void CINNBroadcastInferMeta(const MetaTensor& x,
                            const std::vector<int64_t>& axes,
                            const std::vector<int64_t>& out_shape,
                            MetaTensor* out) {
  out->set_dims(common::make_ddim(out_shape));
  out->set_dtype(x.dtype());
}

void ClassCenterSampleInferMeta(const MetaTensor& label,
                                int num_classes,
                                int num_samples,
                                int ring_id,
                                int rank,
                                int nranks,
                                bool fix_seed,
                                int seed,
                                MetaTensor* remapped_label,
                                MetaTensor* sampled_local_class_center) {
  PADDLE_ENFORCE_EQ(
      label.dims().size(),
      1,
      errors::InvalidArgument("Rank of Input(Label) should be equal to 1, "
                              "but the value given is %d.",
                              label.dims().size()));
  PADDLE_ENFORCE_NOT_NULL(remapped_label,
                          common::errors::InvalidArgument(
                              "output of remapped label should not be null."));
  PADDLE_ENFORCE_NOT_NULL(
      sampled_local_class_center,
      common::errors::InvalidArgument(
          "output of sampled local class center should not be null."));
  remapped_label->set_dims(label.dims());
  remapped_label->set_dtype(label.dtype());
  sampled_local_class_center->set_dims(common::make_ddim({-1}));
  sampled_local_class_center->set_dtype(label.dtype());
}

void ClipByNormInferMeta(const MetaTensor& x, float max_norm, MetaTensor* out) {
  PADDLE_ENFORCE_GT(
      max_norm,
      0,
      common::errors::InvalidArgument("max_norm should be greater than 0. "
                                      "Received max_norm is %f.",
                                      max_norm));
  out->set_dims(x.dims());
  out->set_dtype(x.dtype());
  out->share_lod(x);
}

void CIdentityInferMeta(const MetaTensor& x,
                        int ring_id,
                        bool use_calc_stream,
                        bool use_model_parallel,
                        MetaTensor* out) {
  PADDLE_ENFORCE_GE(
      ring_id,
      0,
      errors::InvalidArgument(
          "The ring_id (%d) for c_identity must be non-negative.", ring_id));
  out->set_dims(x.dims());
  out->set_dtype(x.dtype());
}

void CreateLikeInferMeta(const MetaTensor& x, DataType dtype, MetaTensor* out) {
  out->set_dims(x.dims());
  out->set_dtype(dtype == DataType::UNDEFINED ? x.dtype() : dtype);
  out->set_layout(x.layout());
}

void CreateArrayLikeInferMeta(const MetaTensor& x, MetaTensor* out) {
  out->set_dtype(x.dtype());
  out->set_layout(x.layout());
}

void CumInferMeta(const MetaTensor& x,
                  int axis,
                  bool flatten,
                  bool exclusive,
                  bool reverse,
                  MetaTensor* out) {
  auto x_dims = x.dims();
  if (flatten) {
    out->set_dims(common::make_ddim({common::product(x_dims)}));
    out->set_dtype(x.dtype());
  } else {
    if (x_dims.size() > 0) {
      PADDLE_ENFORCE_GE(
          axis,
          -x_dims.size(),
          common::errors::OutOfRange(
              "axis is out of range (expected to be in range of [%ld, "
              "%ld), but got %ld).",
              -(x_dims.size()),
              x_dims.size(),
              axis));
      PADDLE_ENFORCE_LT(
          axis,
          x_dims.size(),
          common::errors::OutOfRange(
              "axis is out of range (expected to be in range of [%ld, "
              "%ld), but got %ld).",
              -(x_dims.size()),
              x_dims.size(),
              axis));
    } else {
      PADDLE_ENFORCE_EQ(
          (axis == 0 || axis == -1),
          true,
          errors::InvalidArgument("The axis must be -1 or 0 in 0D Tensor, "
                                  "but the value given is %d.",
                                  axis));
    }
    out->set_dims(x_dims);
    out->set_dtype(x.dtype());
  }

  out->share_lod(x);
}

void CumScalarAxisInferMeta(const MetaTensor& x,
                            const Scalar& axis,
                            bool flatten,
                            bool exclusive,
                            bool reverse,
                            MetaTensor* out) {
  CumInferMeta(x, axis.to<int>(), flatten, exclusive, reverse, out);
}

void CumWithIndicesInferMeta(const MetaTensor& x,
                             int axis,
                             DataType dtype,
                             MetaTensor* out,
                             MetaTensor* indices) {
  auto x_dims = x.dims();
  PADDLE_ENFORCE_EQ(
      (dtype == DataType::INT32 || dtype == DataType::INT64),
      true,
      common::errors::InvalidArgument(
          "dtype of indices must be DataType::INT32 or DataType::INT64"));

  if (x_dims.size() > 0) {
    PADDLE_ENFORCE_GE(
        axis,
        -x_dims.size(),
        common::errors::OutOfRange(
            "axis is out of range (expected to be in range of [%ld, "
            "%ld), but got %ld).",
            -(x_dims.size()),
            x_dims.size(),
            axis));
    PADDLE_ENFORCE_LT(
        axis,
        x_dims.size(),
        common::errors::OutOfRange(
            "axis is out of range (expected to be in range of [%ld, "
            "%ld), but got %ld).",
            -(x_dims.size()),
            x_dims.size(),
            axis));
  } else {
    PADDLE_ENFORCE_EQ(
        (axis == 0 || axis == -1),
        true,
        errors::InvalidArgument("The axis must be -1 or 0 in 0D Tensor, "
                                "but the value given is %d.",
                                axis));
  }

  if (dtype == DataType::INT32) {
    int _axis = 0;
    if (axis < 0) {
      _axis = axis + x_dims.size();
    } else {
      _axis = axis;
    }
    PADDLE_ENFORCE_LT(
        common::vectorize(x_dims)[_axis],
        INT32_MAX,
        common::errors::OutOfRange(
            "cummax with axis %ld may be overflow, set dtype int64 to continue",
            axis));
  }

  out->set_dims(x_dims);
  out->set_dtype(x.dtype());
  out->share_lod(x);
  indices->set_dims(x_dims);
  indices->set_dtype(dtype);
  indices->share_lod(x);
}

void CropInferMeta(const MetaTensor& x,
                   const IntArray& shape,
                   const IntArray& offsets,
                   MetaTensor* out,
                   MetaConfig config) {
  PADDLE_ENFORCE_NE(
      out,
      nullptr,
      errors::InvalidArgument("CropTensor should have output tensor out."));

  auto x_dim = x.dims();
  auto shape_dims = shape.GetData();
  auto offsets_vec = offsets.GetData();

  PADDLE_ENFORCE_EQ(shape_dims.size(),
                    x_dim.size(),
                    errors::InvalidArgument(
                        "The number of elements (%d) of attribute 'shape' for "
                        "CropTensor must be equal to the number of "
                        "dimensions (%d) of the input.",
                        shape_dims.size(),
                        x_dim.size()));

  PADDLE_ENFORCE_EQ(
      offsets_vec.size(),
      x_dim.size(),
      errors::InvalidArgument(
          "The number of elements (%d) of attribute 'offsets' for "
          "CropTensor must be equal to the number of "
          "dimensions (%d) of the input.",
          offsets_vec.size(),
          x_dim.size()));

  if (config.is_runtime) {
    out->share_lod(x);
  }

  auto out_dims = std::vector<int64_t>(shape.size(), -1);
  for (int i = 0; i < static_cast<int>(shape_dims.size()); ++i) {
    if (shape_dims[i] > 0) {
      out_dims[i] = static_cast<int64_t>(shape_dims[i]);
    }
  }
  out->set_dims(common::make_ddim(out_dims));
  out->set_dtype(x.dtype());
}

void CScatterInferMeta(const MetaTensor& x,
                       int ring_id,
                       int root_id,
                       int nranks,
                       MetaTensor* out) {
  PADDLE_ENFORCE_GE(nranks,
                    2,
                    common::errors::InvalidArgument(
                        "The number of ranks (%d) must be greater than 1 "
                        "to use collective op (c_scatter op).",
                        nranks));
  PADDLE_ENFORCE_GE(
      root_id,
      0,
      common::errors::InvalidArgument(
          "The root_id (%d) for c_scatter_op must be non-negative.", root_id));
  PADDLE_ENFORCE_GE(
      ring_id,
      0,
      common::errors::InvalidArgument(
          "The ring_id (%d) for c_scatter_op must be non-negative.", ring_id));
  auto dim = x.dims();
  dim[0] = dim[0] / nranks;
  if (dim[0] < 0) dim[0] = -1;
  out->set_dims(dim);
  out->set_dtype(x.dtype());
}

void CSplitInferMeta(const MetaTensor& x, int nranks, MetaTensor* out) {
  phi::DDim dim = x.dims();
  dim[dim.size() - 1] = dim[dim.size() - 1] / nranks;
  if (dim[0] < 0) dim[0] = -1;
  out->set_dims(dim);
  out->set_layout(x.layout());
  out->set_dtype(x.dtype());
}

void DecodeJpegInferMeta(const MetaTensor& x,
                         const std::string& mode,
                         MetaTensor* out) {
  std::vector<int> out_dims;

  if (mode == "unchanged") {
    out_dims = {-1, -1, -1};
  } else if (mode == "gray") {
    out_dims = {1, -1, -1};
  } else if (mode == "rgb") {
    out_dims = {3, -1, -1};
  } else {
    errors::Fatal("The provided mode is not supported for JPEG files on GPU: ",
                  mode);
  }
  if (out != nullptr) {
    out->set_dims(common::make_ddim(out_dims));
    out->set_dtype(x.dtype());
  }
}

void DeQuantizeXPUInferMeta(const MetaTensor& x,
                            DataType out_dtype,
                            float scale,
                            MetaTensor* y) {
  auto x_dims = x.dims();
  y->set_dims(x_dims);
  y->set_dtype(out_dtype);
}

void DiagEmbedInferMeta(
    const MetaTensor& x, int offset, int dim1, int dim2, MetaTensor* out) {
  auto x_dims = x.dims();

  PADDLE_ENFORCE_GE(
      dim1,
      -(x_dims.size() + 1),
      common::errors::OutOfRange(
          "Dim1 is out of range (expected to be in range of [%ld, "
          "%ld], but got %ld).",
          -(x_dims.size() + 1),
          x_dims.size(),
          dim1));
  PADDLE_ENFORCE_LE(
      dim1,
      x_dims.size(),
      common::errors::OutOfRange(
          "Dim1 is out of range (expected to be in range of [%ld, "
          "%ld], but got %ld).",
          -(x_dims.size() + 1),
          x_dims.size(),
          dim1));

  PADDLE_ENFORCE_GE(
      dim2,
      -(x_dims.size() + 1),
      common::errors::OutOfRange(
          "Dim2 is out of range (expected to be in range of [%ld, "
          "%ld], but got %ld).",
          -(x_dims.size() + 1),
          x_dims.size(),
          dim2));
  PADDLE_ENFORCE_LE(
      dim2,
      x_dims.size(),
      common::errors::OutOfRange(
          "Dim2 is out of range (expected to be in range of [%ld, "
          "%ld], but got %ld).",
          -(x_dims.size() + 1),
          x_dims.size(),
          dim2));

  int dim1_ = dim1 < 0 ? x_dims.size() + dim1 + 1 : dim1;
  int dim2_ = dim2 < 0 ? x_dims.size() + dim2 + 1 : dim2;
  int offset_ = std::abs(offset);

  PADDLE_ENFORCE_NE(dim1_,
                    dim2_,
                    common::errors::InvalidArgument(
                        "diagonal dimensions should not be identical "
                        "%ld vs %ld.",
                        dim1,
                        dim2));

  int x_last_dim = x_dims[x_dims.size() - 1];
  int new_dim_len =
      (x_last_dim == -1) ? -1 : static_cast<int>(offset_ + x_last_dim);
  auto sizes = common::vectorize(x_dims);
  sizes.pop_back();
  sizes.insert(sizes.begin() + std::min(dim1_, dim2_), new_dim_len);
  sizes.insert(sizes.begin() + std::max(dim1_, dim2_), new_dim_len);
  out->set_dims(common::make_ddim(sizes));
  out->set_dtype(x.dtype());
}

void DiagInferMeta(const MetaTensor& x,
                   int offset,
                   float padding_value,
                   MetaTensor* out) {
  auto x_dims = x.dims();

  if (x_dims.size() <= 1) {
    int64_t size_ = (x_dims.size() == 1UL ? x_dims[0] : 1) + std::abs(offset);
    out->set_dims({size_, size_});
    out->set_dtype(x.dtype());
  } else if (x_dims.size() == 2UL) {
    int64_t size_ = 0;
    if (offset >= 0) {
      // Note(LutaoChu): Do not use std::min here, otherwise the calculation
      // of `size_` will have unexpected result on Windows Python3.8
      if (x_dims[0] < x_dims[1] - offset) {
        size_ = x_dims[0];
      } else {
        size_ = x_dims[1] - offset;
      }
    } else {
      // Note(LutaoChu): Do not use std::min here, otherwise the calculation
      // of `size_` will have unexpected result on Windows Python3.8
      if (x_dims[0] + offset < x_dims[1]) {
        size_ = x_dims[0] + offset;
      } else {
        size_ = x_dims[1];
      }
    }
    out->set_dims({size_});
    out->set_dtype(x.dtype());
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "The input tensor X's dimensions of DiagV2Op should be either 1 or "
        "2, but received %d.",
        x_dims.size()));
  }
}

void DiagonalInferMeta(const MetaTensor& input,
                       int offset,
                       int axis1,
                       int axis2,
                       MetaTensor* out) {
  auto x_dims = input.dims();
  int offset_ = offset;
  int axis1_ = axis1 < 0 ? x_dims.size() + axis1 : axis1;
  int axis2_ = axis2 < 0 ? x_dims.size() + axis2 : axis2;
  PADDLE_ENFORCE_GE(
      x_dims.size(),
      2,
      common::errors::OutOfRange("Input's dim is out of range (expected at "
                                 "least 2 dimensions, but got %ld).",
                                 x_dims.size()));
  PADDLE_ENFORCE_LT(
      axis1_,
      x_dims.size(),
      common::errors::OutOfRange(
          "Attr(axis1) is out of range (expected to be in range of [%ld, "
          "%ld], but got %ld).",
          -(x_dims.size()),
          (x_dims.size() - 1),
          axis1));
  PADDLE_ENFORCE_GE(
      axis1_,
      0,
      common::errors::OutOfRange(
          "Attr(axis1) is out of range (expected to be in range of [%ld, "
          "%ld], but got %ld).",
          -(x_dims.size()),
          (x_dims.size() - 1),
          axis1));
  PADDLE_ENFORCE_LT(
      axis2_,
      x_dims.size(),
      common::errors::OutOfRange(
          "Attr(axis2) is out of range (expected to be in range of [%ld, "
          "%ld], but got %ld).",
          -(x_dims.size()),
          (x_dims.size() - 1),
          axis2));
  PADDLE_ENFORCE_GE(
      axis2_,
      0,
      common::errors::OutOfRange(
          "Attr(axis2) is out of range (expected to be in range of [%ld, "
          "%ld], but got %ld).",
          -(x_dims.size()),
          (x_dims.size() - 1),
          axis2));
  PADDLE_ENFORCE_NE(
      axis1_,
      axis2_,
      common::errors::InvalidArgument("The dimensions should not be identical "
                                      "%d vs %d.",
                                      axis1,
                                      axis2));

  auto out_dims = common::vectorize(x_dims);
  // from out_dims get the dim size of axis1_.
  auto axis1_size = out_dims[axis1_];
  auto axis2_size = out_dims[axis2_];
  // delete two dims by attr axis1 and axis2 from out_dims.
  /* example:
     out_dim = [2, 3, 4];
     axis1 = 0;
     axis2 = 1;
     according to the attr of axis1 and axis2, we get:
     out_dim = [4].
  */
  out_dims.erase(out_dims.begin() + std::max(axis1_, axis2_));
  out_dims.erase(out_dims.begin() + std::min(axis1_, axis2_));

  if (axis1_size != -1 && axis2_size != -1) {
    if (offset_ == 0) {
      out_dims.push_back(std::min(axis1_size, axis2_size));
    } else if (offset_ > 0) {
      if ((axis2_size - offset_) > 0) {
        out_dims.push_back(std::min(axis1_size, axis2_size - offset_));
      } else {
        out_dims.push_back(0);
      }
    } else {
      if ((axis1_size + offset_) > 0) {
        out_dims.push_back(std::min(axis1_size + offset_, axis2_size));
      } else {
        out_dims.push_back(0);
      }
    }
  } else {
    out_dims.push_back(-1);
  }
  out->set_dims(common::make_ddim(out_dims));
  out->set_dtype(input.dtype());
}

void DirichletInferMeta(const MetaTensor& alpha, MetaTensor* out) {
  const auto alpha_dim = alpha.dims();
  PADDLE_ENFORCE_GE(alpha_dim.size(),
                    1,
                    common::errors::InvalidArgument(
                        "ShapeError: The number of dimensions of 'Alpha' "
                        "must be greater than or equal to 1. "
                        "But received Alpha's dimensions = %d,",
                        alpha_dim.size()));
  out->set_dims(alpha_dim);
  out->set_dtype(alpha.dtype());
}

void DistConcatInferMeta(const MetaTensor& x, int nranks, MetaTensor* out) {
  auto dim = x.dims();
  dim[dim.size() - 1] = dim[dim.size() - 1] * nranks;
  if (dim[dim.size() - 1] < 0) dim[dim.size() - 1] = -1;
  out->set_dtype(x.dtype());
  out->set_dims(dim);
}

void DistReduceInferMeta(const MetaTensor& x, MetaTensor* out) {
  out->set_dtype(x.dtype());
  out->set_dims(x.dims());
}

void DistBroadcastInferMeta(const MetaTensor& x, MetaTensor* out) {
  out->set_dtype(x.dtype());
  out->set_dims(x.dims());
}

void EigInferMeta(const MetaTensor& x, MetaTensor* out_w, MetaTensor* out_v) {
  phi::DDim x_dims = x.dims();
  int rank = x_dims.size();
  PADDLE_ENFORCE_GE(rank,
                    2,
                    common::errors::InvalidArgument(
                        "Expects input tensor x to be not less than "
                        "2 dimensions, but got dimension %d",
                        rank));
  PADDLE_ENFORCE_EQ(x_dims[rank - 2],
                    x_dims[rank - 1],
                    common::errors::InvalidArgument(
                        "The input matrix must be a square matrix, "
                        "but receive a matrix with %d rows and %d columns",
                        x_dims[rank - 2],
                        x_dims[rank - 1]));

  std::vector<int> batch_dims_vec{};
  for (int i = 0; i < rank - 1; ++i) {
    batch_dims_vec.emplace_back(x_dims[i]);
  }
  const DataType& x_dtype = x.dtype();
  const DataType& out_dtype =
      IsComplexType(x_dtype) ? x_dtype : ToComplexType(x_dtype);
  out_w->set_dims(common::make_ddim(batch_dims_vec));
  out_w->set_dtype(out_dtype);
  out_v->set_dims(x_dims);
  out_v->set_dtype(out_dtype);
}

void EmbeddingGradSparseInferMeta(const MetaTensor& x,
                                  const MetaTensor& weight,
                                  MetaTensor* out) {
  if (weight) {
    if (out->is_selected_rows()) {
      out->set_dims(x.dims());
      out->set_dtype(x.dtype());
    } else {
      out->share_dims(weight);
      out->set_dtype(weight.dtype());
    }
  }
}

void EighInferMeta(const MetaTensor& x,
                   const std::string& uplo,
                   MetaTensor* out_w,
                   MetaTensor* out_v) {
  auto input_dim = x.dims();
  auto rank = input_dim.size();

  PADDLE_ENFORCE_GE(rank,
                    2,
                    common::errors::InvalidArgument(
                        "The Input(X) should have at least 2 dimensions."
                        "But received a %d dimension tensor.",
                        rank));
  PADDLE_ENFORCE_EQ(
      input_dim[rank - 2],
      input_dim[rank - 1],
      common::errors::InvalidArgument(
          "Eigh op is designed for square matrix, consequently"
          "inner-most 2 dimensions of Input(X) should be symmetric."
          "But received X's shape[-2] = %d and shape[-1] = %d.",
          input_dim[rank - 2],
          input_dim[rank - 1]));

  std::vector<int64_t> values_dim;

  for (auto i = 0; i < rank - 1; i++) {
    values_dim.emplace_back(input_dim[i]);
  }
  out_w->set_dims(common::make_ddim(values_dim));
  out_w->set_dtype(dtype::ToReal(x.dtype()));
  out_v->set_dims(input_dim);
  out_v->set_dtype(x.dtype());
}

void EigvalsInferMeta(const MetaTensor& x, MetaTensor* out, MetaConfig config) {
  auto x_dims = x.dims();
  PADDLE_ENFORCE_GE(x_dims.size(),
                    2,
                    errors::InvalidArgument(
                        "The dimensions of Input(X) for Eigvals operator "
                        "should be at least 2, "
                        "but received X's dimension = %d, X's shape = [%s].",
                        x_dims.size(),
                        x_dims));

  if (config.is_runtime || !common::contain_unknown_dim(x_dims)) {
    int last_dim = x_dims.size() - 1;
    PADDLE_ENFORCE_EQ(x_dims[last_dim],
                      x_dims[last_dim - 1],
                      errors::InvalidArgument(
                          "The last two dimensions of Input(X) for Eigvals "
                          "operator should be equal, "
                          "but received X's shape = [%s].",
                          x_dims));
  }

  auto out_dims = common::vectorize(x_dims);
  out_dims.resize(x_dims.size() - 1);

  const DataType& x_dtype = x.dtype();
  const DataType& out_dtype =
      IsComplexType(x_dtype) ? x_dtype : ToComplexType(x_dtype);

  out->set_dims(common::make_ddim(out_dims));
  out->set_dtype(out_dtype);
}

void EigvalshInferMeta(const MetaTensor& x,
                       const std::string& uplo,
                       bool is_test,
                       MetaTensor* out_w,
                       MetaTensor* out_v) {
  auto input_dim = x.dims();
  auto rank = input_dim.size();

  PADDLE_ENFORCE_GE(
      rank,
      2,
      errors::InvalidArgument("The Input(X) should have at least 2 dimensions."
                              "But received a %d dimension tensor.",
                              rank));
  PADDLE_ENFORCE_EQ(
      input_dim[rank - 2],
      input_dim[rank - 1],
      errors::InvalidArgument(
          "Eigvalsh op is designed for square matrix, consequently"
          "inner-most 2 dimensions of Input(X) should be symmetric."
          "But received X's shape[-2] = %d and shape[-1] = %d.",
          input_dim[rank - 2],
          input_dim[rank - 1]));

  std::vector<int64_t> values_dim;

  for (auto i = 0; i < rank - 1; i++) {
    values_dim.emplace_back(input_dim[i]);
  }

  if (out_w != nullptr) {
    out_w->set_dims(common::make_ddim(values_dim));
    out_w->set_dtype(dtype::ToReal(x.dtype()));
  }
  if (out_v != nullptr) {
    out_v->set_dims(input_dim);
    out_v->set_dtype(x.dtype());
  }
}

void EinsumInferMeta(const std::vector<const MetaTensor*>& inputs,
                     const std::string& equation,
                     MetaTensor* out) {
  // collect the following information to prepare einsum.
  LabelMap labelshape(0);
  LabelMap labeltype(LabelType::Reduction);
  std::vector<LabelMap> label2perms(inputs.size(), LabelMap(-1));
  std::vector<char> all_labels;
  std::vector<int> output_dims;
  std::vector<std::vector<int>> broadcast_shapes(2);

  std::vector<DDim> input_dims;
  for (auto& i : inputs) {
    input_dims.push_back(i->dims());
  }
  std::vector<std::string> input_strs;
  std::string right;
  ParseEinsumEquation(equation,
                      input_dims,
                      &labelshape,
                      &labeltype,
                      &all_labels,
                      &label2perms,
                      &broadcast_shapes,
                      &output_dims,
                      &right,
                      &input_strs);

  VLOG(3) << "Einsum Infershape: input dims:"
          << paddle::string::join_strings(input_dims, "\n");
  VLOG(3) << "Einsum Infershape: equation:" << equation;
  VLOG(3) << "Einsum Infershape: all_labels:"
          << paddle::string::join_strings(all_labels, ",");
  VLOG(3) << "Einsum Infershape: output dims:"
          << paddle::string::join_strings(output_dims, ",");
  VLOG(3) << "Label Type is : " << label_to_string(all_labels, labeltype);
  VLOG(3) << "Label Shape is : " << label_to_string(all_labels, labelshape);
  out->set_dims(common::make_ddim(output_dims));
  out->set_dtype(inputs[0]->dtype());
}

void EinsumRawInferMeta(const std::vector<const MetaTensor*>& inputs,
                        const std::string& equation,
                        MetaTensor* out,
                        std::vector<MetaTensor*> inner_cache,
                        std::vector<MetaTensor*> xshape) {
  EinsumInferMeta(inputs, equation, out);
  for (size_t i = 0; i < xshape.size(); ++i) {
    if (xshape[i] != nullptr) {
      xshape[i]->set_dims(inputs[i]->dims());
      xshape[i]->set_dtype(inputs[i]->dtype());
    }
  }
}

void ExpandInferMeta(const MetaTensor& x,
                     const IntArray& shape,
                     MetaTensor* out) {
#define EXPAND_MAX_RANK_SUPPORTED 8
  auto x_dims = x.dims();
  auto expand_shape = shape.GetData();

  if (expand_shape.empty()) {
    expand_shape = std::vector<int64_t>(x_dims.size(), -1);
  }

  PADDLE_ENFORCE_GE(
      expand_shape.size(),
      static_cast<size_t>(x_dims.size()),
      common::errors::InvalidArgument(
          "The number of elements (%d) of 'shape' for "
          "expand_v2 op must be greater than or equal to the rank "
          "(%d) of the input.",
          expand_shape.size(),
          static_cast<size_t>(x_dims.size())));
  PADDLE_ENFORCE_LE(expand_shape.size(),
                    EXPAND_MAX_RANK_SUPPORTED,
                    common::errors::InvalidArgument(
                        "The number of elements (%d) of 'shape' for "
                        "must not be greater than %d.",
                        expand_shape.size(),
                        EXPAND_MAX_RANK_SUPPORTED));
  PADDLE_ENFORCE_GE(expand_shape.size(),
                    0,
                    common::errors::InvalidArgument(
                        "The number of elements (%d) of 'shape' for "
                        "must be a positive integer.",
                        expand_shape.size()));

  int out_rank = expand_shape.size();
  const auto& out_shape = [&]() -> std::vector<int64_t> {
    std::vector<int64_t> res = expand_shape;
    int x_rank = x_dims.size();
    const auto& DealWithMinusOne = [&]() {
      for (int x_idx = x_rank - 1, out_idx = out_rank - 1; x_idx >= 0;
           x_idx--, out_idx--) {
        if (res[out_idx] == -1) {
          res[out_idx] = x_dims[x_idx];
        }
      }
    };
    const auto& DealWithMinusTwo = [&]() {
      // We use -2 to represent the element in expand_shape is a var.
      for (int x_idx = x_rank - 1, out_idx = out_rank - 1; out_idx >= 0;
           x_idx--, out_idx--) {
        if (res[out_idx] == -2) {
          res[out_idx] = -1;
        }
      }
    };
    DealWithMinusOne();
    DealWithMinusTwo();
    return res;
  }();

  out->set_dims(common::make_ddim(out_shape));
  out->set_dtype(x.dtype());
  if (out_rank > 0 && out_shape[0] == x_dims[0]) {
    out->share_lod(x);
  }
#undef EXPAND_MAX_RANK_SUPPORTED
}

void FakeChannelWiseQuantizeAbsMaxInferMeta(const MetaTensor& x,
                                            int bit_length,
                                            int round_type,
                                            int quant_axis,
                                            bool is_test,
                                            MetaTensor* out,
                                            MetaTensor* out_scale) {
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
  out->set_dtype(x.dtype());
  out->set_dims(x.dims());
  out_scale->set_dims({x.dims()[quant_axis]});
  out->share_lod(x);
}

void FakeChannelWiseQuantizeDequantizeAbsMaxInferMeta(const MetaTensor& x,
                                                      int bit_length,
                                                      int round_type,
                                                      int quant_axis,
                                                      MetaTensor* out,
                                                      MetaTensor* out_scale) {
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
  out->set_dtype(x.dtype());
  out->set_dims(x.dims());
  out_scale->set_dims({x.dims()[quant_axis]});
  out->share_lod(x);
}

void FakeQuantizeAbsMaxInferMeta(const MetaTensor& x,
                                 int bit_length,
                                 int round_type,
                                 MetaTensor* out,
                                 MetaTensor* out_scale) {
  PADDLE_ENFORCE_EQ(bit_length >= 1 && bit_length <= 16,
                    true,
                    common::errors::InvalidArgument(
                        "'bit_length' should be between 1 and 16, but "
                        "the received is %d",
                        bit_length));
  out->set_dtype(x.dtype());
  out->set_dims(x.dims());
  out_scale->set_dims({1});
  out->share_lod(x);
}

void FillAnyLikeInferMeta(const MetaTensor& x,
                          const Scalar& value,
                          DataType dtype,
                          MetaTensor* out) {
  out->set_dims(x.dims());
  out->set_dtype(dtype == DataType::UNDEFINED ? x.dtype() : dtype);
  out->share_lod(x);
}

void FetchBarrierInferMeta(const std::vector<const MetaTensor*>& x,
                           int trainer_id,
                           const std::vector<std::string>& endpoints,
                           std::vector<MetaTensor*> out) {}

void FillDiagonalInferMeta(
    const MetaTensor& x, float value, int offset, bool wrap, MetaTensor* out) {
  PADDLE_ENFORCE_NE(
      out,
      nullptr,
      common::errors::InvalidArgument("Tensor out should not be null if "));
  auto x_dims = x.dims();
  out->set_dims(x_dims);
  out->set_dtype(x.dtype());
}

void FFTC2CInferMeta(const MetaTensor& x,
                     const std::vector<int64_t>& axes,
                     const std::string& normalization,
                     bool forward,
                     MetaTensor* out,
                     MetaConfig config) {
  PADDLE_ENFORCE_NOT_NULL(
      out,
      common::errors::InvalidArgument("Output of fft_c2c should not be null."));
  // only ensure that fft axes' size greater than zero at runtime
  // they might be -1 to indicate unknown size ar compile time
  if (config.is_runtime) {
    const phi::DDim x_dim = x.dims();
    for (auto axis : axes) {
      PADDLE_ENFORCE_GT(x_dim[axis],
                        0,
                        common::errors::InvalidArgument(
                            "Invalid fft n-point (%d).", x_dim[axis]));
    }
  }
  out->share_meta(x);
}

void FFTC2RInferMeta(const MetaTensor& x,
                     const std::vector<int64_t>& axes,
                     const std::string& normalization,
                     bool forward,
                     int64_t last_dim_size,
                     MetaTensor* out,
                     MetaConfig config) {
  PADDLE_ENFORCE_NOT_NULL(
      out,
      common::errors::InvalidArgument("Output of fft_c2r should not be null."));
  const phi::DDim x_dim = x.dims();
  const int last_fft_axis = static_cast<int>(axes.back());

  // only ensure that fft axes' size greater than zero at runtime
  // they might be -1 to indicate unknown size ar compile time
  if (config.is_runtime) {
    size_t signal_dims = axes.size();
    for (size_t i = 0; i < signal_dims - 1; i++) {
      PADDLE_ENFORCE_GT(x_dim[axes[i]],
                        0,
                        common::errors::InvalidArgument(
                            "Invalid fft n-point (%d).", x_dim[axes[i]]));
    }
  }

  out->set_layout(x.layout());
  out->set_dtype(ToRealType(x.dtype()));
  phi::DDim out_dim = x_dim;

  if (last_dim_size > 0) {
    out_dim.at(last_fft_axis) = last_dim_size;
  } else if (config.is_runtime) {
    const int64_t input_last_dim_size = x_dim[last_fft_axis];
    const int64_t fft_n_point = (input_last_dim_size - 1) * 2;
    PADDLE_ENFORCE_GT(fft_n_point,
                      0,
                      common::errors::InvalidArgument(
                          "Invalid fft n-point (%d).", fft_n_point));
    out_dim.at(last_fft_axis) = fft_n_point;
  } else {
    const int64_t input_last_dim_size = x_dim[last_fft_axis];
    out_dim.at(last_fft_axis) =
        input_last_dim_size == -1 ? -1 : (input_last_dim_size - 1) * 2;
  }
  out->set_dims(out_dim);
}

void FFTR2CInferMeta(const MetaTensor& x,
                     const std::vector<int64_t>& axes,
                     const std::string& normalization,
                     bool forward,
                     bool onesided,
                     MetaTensor* out,
                     MetaConfig config) {
  PADDLE_ENFORCE_NOT_NULL(
      out,
      common::errors::InvalidArgument("Output of fft_r2c should not be null."));
  const phi::DDim x_dim = x.dims();

  // only ensure that fft axes' size greater than zero at runtime
  // they might be -1 to indicate unknown size ar compile time
  if (config.is_runtime) {
    for (auto axis : axes) {
      PADDLE_ENFORCE_GT(x_dim[axis],
                        0,
                        common::errors::InvalidArgument(
                            "Invalid fft n-point (%d).", x_dim[axis]));
    }
  }

  out->set_layout(x.layout());
  out->set_dtype(ToComplexType(x.dtype()));
  if (!onesided) {
    out->share_dims(x);
  } else {
    phi::DDim out_dim = x.dims();
    const int last_fft_axis = static_cast<int>(axes.back());
    const int64_t last_fft_dim_size = x_dim[last_fft_axis];
    out_dim.at(last_fft_axis) = last_fft_dim_size / 2 + 1;
    out->set_dims(out_dim);
  }
}

void FlattenWithXShapeInferMeta(const MetaTensor& x,
                                int start_axis,
                                int stop_axis,
                                MetaTensor* out,
                                MetaTensor* xshape) {
  FlattenInferMeta(x, start_axis, stop_axis, out);
  if (xshape == nullptr) return;
  const auto& x_dims = x.dims();
  std::vector<int64_t> xshape_dims(x_dims.size() + 1);
  xshape_dims[0] = 0;
  for (int i = 0; i < x_dims.size(); ++i) {
    xshape_dims[i + 1] = x_dims[i];
  }
  xshape->set_dims(common::make_ddim(xshape_dims));
  xshape->share_lod(x);
}

void FlattenInferMeta(const MetaTensor& x,
                      int start_axis,
                      int stop_axis,
                      MetaTensor* out) {
  auto x_dims = x.dims();
  int in_dims_size = x_dims.size();

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

  int64_t outer = 1;
  std::vector<int32_t> out_shape;
  out_shape.reserve(in_dims_size - stop_axis + start_axis + 1);

  for (int i = 0; i < start_axis; ++i) {
    out_shape.push_back(x_dims[i]);  // NOLINT
  }
  for (int i = start_axis; i <= stop_axis; i++) {
    if (x_dims[i] == -1 || outer == -1) {
      outer = -1;
    } else {
      outer *= x_dims[i];
    }
  }
  out_shape.push_back(outer);  // NOLINT
  for (int i = stop_axis + 1; i < in_dims_size; i++) {
    out_shape.push_back(x_dims[i]);  // NOLINT
  }
  const auto& out_dims = common::make_ddim(out_shape);
  out->set_dims(out_dims);
  out->set_dtype(x.dtype());
  out->set_layout(x.layout());

  if (x_dims[0] == out_dims[0]) {
    // Only pass LoD when the first dimension of output and Input(X)
    // are the same.
    out->share_lod(x);
  }
}

void Flatten2InferMeta(const MetaTensor& x,
                       int axis,
                       MetaTensor* out,
                       MetaTensor* x_shape) {
  const auto& in_dims = x.dims();
  PADDLE_ENFORCE_GE(axis,
                    0,
                    common::errors::InvalidArgument(
                        "The axis should be greater than or equal to 0."));
  PADDLE_ENFORCE_LE(
      axis,
      in_dims.size(),
      common::errors::InvalidArgument(
          "The axis should be less than or equal to input tensor's rank"));

  const auto& out_dims = phi::funcs::GetOutputShape(axis, in_dims);
  out->set_dims(common::make_ddim(out_dims));
  if (in_dims[0] == out_dims[0]) {
    // Only pass LoD when the first dimension of output and Input(X)
    // are the same.
    out->share_lod(x);
  }
  out->set_dtype(x.dtype());
  if (!x_shape->initialized()) return;
  std::vector<int64_t> xshape_dims(in_dims.size() + 1);
  xshape_dims[0] = 0;
  for (int i = 0; i < in_dims.size(); ++i) {
    xshape_dims[i + 1] = in_dims[i];
  }
  x_shape->set_dims(common::make_ddim(xshape_dims));
  x_shape->share_lod(x);
  x_shape->set_dtype(x.dtype());
}

void FlipInferMeta(const MetaTensor& x,
                   const std::vector<int>& axis,
                   MetaTensor* out) {
  auto x_dims = x.dims();
  auto flip_dims = axis;
  size_t flip_dims_size = axis.size();

  if (flip_dims_size > 0) {
    // check if dims axis within range
    auto min_max_d = std::minmax_element(flip_dims.begin(), flip_dims.end());
    PADDLE_ENFORCE_LT(*min_max_d.first,
                      x_dims.size(),
                      common::errors::InvalidArgument(
                          "min(axes) should be less than the input tensor X's "
                          "axes of FlipOp. But received min(axes) = %d,  "
                          "X's axes = %d, X's shape = [%s]",
                          *min_max_d.first,
                          x_dims.size(),
                          x_dims));
    PADDLE_ENFORCE_GE(*min_max_d.first,
                      x_dims.size() * -1,
                      common::errors::InvalidArgument(
                          "min(axes) should be greater than or equal to the "
                          "input tensor X's "
                          "axes of FlipOp times -1. But received "
                          "min(axes) = %d,  X's "
                          "axes = %d, X's shape = [%s]",
                          *min_max_d.first,
                          x_dims.size() * -1,
                          x_dims));
    PADDLE_ENFORCE_LT(*min_max_d.second,
                      x_dims.size(),
                      common::errors::InvalidArgument(
                          "max(axes) should be less than the input tensor X's "
                          "axes of FlipOp. But received max(axes) = %d,  "
                          "X's axes = %d, X's shape = [%s]",
                          *min_max_d.second,
                          x_dims.size(),
                          x_dims));
    PADDLE_ENFORCE_GE(*min_max_d.second,
                      x_dims.size() * -1,
                      common::errors::InvalidArgument(
                          "max(axes) should be greater than or equal to the "
                          "input tensor X's "
                          "axes of FlipOp times -1. But received "
                          "max(axes) = %d,  X's "
                          "axes = %d, X's shape = [%s]",
                          *min_max_d.second,
                          x_dims.size() * -1,
                          x_dims));

    // check duplicates in dims
    flip_dims.erase(std::unique(flip_dims.begin(), flip_dims.end()),
                    flip_dims.end());
    PADDLE_ENFORCE_EQ(flip_dims.size(),
                      flip_dims_size,
                      common::errors::InvalidArgument(
                          "axes has duplicates, original flip axes size=%d, "
                          "but unique flip axes size=%d.)",
                          flip_dims_size,
                          flip_dims.size()));
  }

  VLOG(3) << "flip operator x.shape=" << x_dims;

  std::vector<int64_t> output_dims(x_dims.size());
  for (int i = 0; i < x_dims.size(); ++i) {
    output_dims[i] = x_dims[i];
  }

  out->set_dims(common::make_ddim(output_dims));
  out->set_dtype(x.dtype());
  out->share_lod(x);
}

void FoldInferMeta(const MetaTensor& x,
                   const std::vector<int>& output_sizes,
                   const std::vector<int>& kernel_sizes,
                   const std::vector<int>& strides,
                   const std::vector<int>& paddings,
                   const std::vector<int>& dilations,
                   MetaTensor* out) {
  auto in_dims = x.dims();

  PADDLE_ENFORCE_EQ(
      output_sizes.size(),
      2,
      common::errors::InvalidArgument(
          "It is expected output_size equals to 2, but got size %d",
          output_sizes.size()));
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

  int output_height = output_sizes[0];
  int output_width = output_sizes[1];
  int kernel_height = kernel_sizes[0];
  int kernel_width = kernel_sizes[1];
  int dilation_height = dilations[0];
  int dilation_width = dilations[1];
  int stride_height = strides[0];
  int stride_width = strides[1];

  // check kernel_sizes
  PADDLE_ENFORCE_GT(kernel_height,
                    0,
                    common::errors::InvalidArgument(
                        "The `kernel_sizes` should be greater than zero, "
                        "but received kernel_height: %d kernel_width: %d.",
                        kernel_sizes[0],
                        kernel_sizes[1]));
  PADDLE_ENFORCE_GT(kernel_width,
                    0,
                    common::errors::InvalidArgument(
                        "The `kernel_sizes` should be greater than zero, "
                        "but received kernel_height: %d kernel_width: %d.",
                        kernel_sizes[0],
                        kernel_sizes[1]));
  // check strides
  PADDLE_ENFORCE_GT(stride_height,
                    0,
                    common::errors::InvalidArgument(
                        "The `strides` should be greater than zero, "
                        "but received strides_height: %d strides_width: %d.",
                        strides[0],
                        strides[1]));
  PADDLE_ENFORCE_GT(stride_width,
                    0,
                    common::errors::InvalidArgument(
                        "The `strides` should be greater than zero, "
                        "but received strides_height: %d strides_width: %d.",
                        strides[0],
                        strides[1]));
  // check output size
  PADDLE_ENFORCE_GT(output_height,
                    0,
                    common::errors::InvalidArgument(
                        "The `output_height` should be greater than zero, "
                        "but received output_height: %d .",
                        output_height));
  PADDLE_ENFORCE_GT(output_width,
                    0,
                    common::errors::InvalidArgument(
                        "The `output_width` should be greater than zero, "
                        "but received output_width: %d .",
                        output_width));
  // check dilations
  PADDLE_ENFORCE_GT(
      dilation_height,
      0,
      common::errors::InvalidArgument(
          "The `dilations` should be greater than zero, "
          "but received dilations_height: %d dilations_width: %d.",
          dilations[0],
          dilations[1]));
  PADDLE_ENFORCE_GT(
      dilation_width,
      0,
      common::errors::InvalidArgument(
          "The `dilations` should be greater than zero, "
          "but received dilations_height: %d dilations_width: %d.",
          dilations[0],
          dilations[1]));

  std::vector<int> out_dims;
  // batch_size
  out_dims.push_back(in_dims[0]);  // NOLINT
  // output_plane
  int output_channels =
      static_cast<int>(in_dims[1] / (kernel_width * kernel_height));
  out_dims.push_back(output_channels);

  int blocks_height = (output_sizes[0] + 2 * paddings[0] -
                       (dilations[0] * (kernel_sizes[0] - 1) + 1)) /
                          strides[0] +
                      1;
  int blocks_width = (output_sizes[1] + 2 * paddings[1] -
                      (dilations[1] * (kernel_sizes[1] - 1) + 1)) /
                         strides[1] +
                     1;

  // check output height and width
  PADDLE_ENFORCE_GT(
      blocks_height,
      0,
      common::errors::InvalidArgument(
          "The sliding blocks calculated from input spatial size (%d, %d), "
          "kernel_sizes (%d, %d), strides (%d, %d), dilations (%d, %d), "
          "is (%d, %d), which should be a positive integer.",
          in_dims[2],
          in_dims[3],
          kernel_sizes[0],
          kernel_sizes[1],
          strides[0],
          strides[1],
          dilations[0],
          dilations[1],
          output_height,
          output_width));

  PADDLE_ENFORCE_GT(
      blocks_width,
      0,
      common::errors::InvalidArgument(
          "The sliding blocks calculated from input spatial size (%d, %d), "
          "kernel_sizes (%d, %d), strides (%d, %d), dilations (%d, %d), "
          "is (%d, %d), which should be a positive integer.",
          in_dims[2],
          in_dims[3],
          kernel_sizes[0],
          kernel_sizes[1],
          strides[0],
          strides[1],
          dilations[0],
          dilations[1],
          output_height,
          output_width));

  PADDLE_ENFORCE_EQ(
      blocks_height * blocks_width,
      in_dims[2],
      common::errors::InvalidArgument(
          "Given input output_size (%d, %d), "
          "kernel_sizes (%d, %d), strides (%d, %d), dilations (%d, %d), "
          "which should be expected size of input's dimension "
          "2 to match the calculated number of %d * %d = %d, but got %d",
          output_height,
          output_width,
          kernel_sizes[0],
          kernel_sizes[1],
          strides[0],
          strides[1],
          dilations[0],
          dilations[1],
          blocks_height,
          blocks_width,
          blocks_height * blocks_width,
          in_dims[2]));

  PADDLE_ENFORCE_EQ(
      in_dims[1] % (kernel_sizes[0] * kernel_sizes[1]),
      0,
      common::errors::InvalidArgument(
          "Expected size of input's dimension 1 to be divisible by the"
          "product of kernel_size, but got input.size(1)=%d and "
          "kernel_size=( %d"
          ", %d).",
          in_dims[1],
          kernel_sizes[0],
          kernel_sizes[1]));

  out_dims.push_back(output_height);
  out_dims.push_back(output_width);
  if (out != nullptr) {
    out->set_dims(common::make_ddim(out_dims));
    out->set_dtype(x.dtype());
  }
}

void FractionalMaxPoolInferMeta(const MetaTensor& x,
                                const std::vector<int>& output_size,
                                const std::vector<int>& kernel_size,
                                float random_u,
                                bool return_mask,
                                MetaTensor* out,
                                MetaTensor* mask,
                                MetaConfig config) {
  std::vector<int> output_size_ = output_size;

  auto x_dims = x.dims();

  PADDLE_ENFORCE_EQ(
      (x_dims.size() == 4 || x_dims.size() == 5),
      true,
      errors::InvalidArgument("Pooling input should be 4-D or "
                              "5-D tensor but received %dD-Tensor",
                              x_dims.size()));

  PADDLE_ENFORCE_EQ(
      x_dims.size() - output_size_.size(),
      2U,
      errors::InvalidArgument(
          "The input size %d minus the output size %d should equal to 2.",
          x_dims.size(),
          output_size_.size()));

  std::vector<int64_t> output_shape({x_dims[0], x_dims[1]});
  output_shape.insert(
      output_shape.end(), output_size_.begin(), output_size_.end());

  out->set_dims(common::make_ddim(output_shape));
  out->set_dtype(x.dtype());

  mask->set_dims(common::make_ddim(output_shape));
  mask->set_dtype(phi::CppTypeToDataType<int>::Type());
}

void FrameInferMeta(const MetaTensor& x,
                    int frame_length,
                    int hop_length,
                    int axis,
                    MetaTensor* out,
                    MetaConfig config) {
  PADDLE_ENFORCE_NOT_NULL(out,
                          common::errors::InvalidArgument(
                              "Output(Out) of FrameOp should not be null."));
  const auto x_dims = x.dims();
  const int x_rank = x_dims.size();

  PADDLE_ENFORCE_GE(x_rank,
                    1,
                    common::errors::InvalidArgument(
                        "Input(X) of FrameOp should be a tensor which contains "
                        "at least 1 dimension, but got rank %s.",
                        x_rank));
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

  std::vector<int64_t> output_shape;
  int seq_length = 0;
  int n_frames = 0;

  int start_axis = 0;
  int end_axis = 0;

  if (axis == 0) {
    seq_length = static_cast<int>(x_dims[0]);
    start_axis = 1;
    end_axis = x_rank - 1;
  } else {
    seq_length = static_cast<int>(x_dims[x_rank - 1]);
    start_axis = 0;
    end_axis = x_rank - 2;
  }

  bool contain_unknown_dim = common::contain_unknown_dim(x_dims);
  bool check = config.is_runtime || !contain_unknown_dim;
  if (check) {
    PADDLE_ENFORCE_LE(frame_length,
                      seq_length,
                      common::errors::InvalidArgument(
                          "Attribute(frame_length) of FrameOp should be less "
                          "equal than sequence length, but got (%s) > (%s).",
                          frame_length,
                          seq_length));
  }

  // It won't go into for loop when x_rank == 1U.
  for (int i = start_axis; i <= end_axis; i++) {
    output_shape.push_back(x_dims[i]);
  }

  if (seq_length == -1) {
    n_frames = -1;
  } else {
    n_frames = 1 + (seq_length - frame_length) / hop_length;
  }

  if (axis == 0) {
    // (n_frames, frame_length, ...)
    output_shape.insert(output_shape.begin(), frame_length);
    output_shape.insert(output_shape.begin(), n_frames);
  } else {
    // (..., frame_length, n_frames)
    output_shape.push_back(frame_length);
    output_shape.push_back(n_frames);
  }

  out->set_dims(common::make_ddim(output_shape));
  out->set_dtype(x.dtype());
}

void FullBatchSizeLikeInferMeta(const MetaTensor& x,
                                const std::vector<int>& shape,
                                const Scalar& val,
                                DataType dtype,
                                int x_batch_size_dim,
                                int out_batch_size_dim,
                                MetaTensor* out) {
  BatchSizeLikeInferMeta(x, shape, x_batch_size_dim, out_batch_size_dim, out);
  out->set_dtype(dtype);
}

void GumbelSoftmaxInferMeta(const MetaTensor& x,
                            float temperature,
                            bool hard,
                            int axis,
                            MetaTensor* out) {
  UnchangedInferMetaCheckAxis(x, axis, out);
}

void HashInferMeta(const MetaTensor& x,
                   int num_hash,
                   int64_t mod_by,
                   MetaTensor* out) {
  const auto& dims = x.dims();
  PADDLE_ENFORCE_EQ(dims.size(),
                    2UL,
                    common::errors::InvalidArgument(
                        "The input of hash_op's dimensions must be 2"));
  std::vector<int64_t> out_dims;
  phi::funcs::HashOutputSize(dims, out_dims, num_hash);

  out->set_dims(common::make_ddim(out_dims));
  out->share_lod(x);
  out->set_dtype(x.dtype());
}

void IdentityLossInferMeta(const MetaTensor& x,
                           int reduction,
                           MetaTensor* out) {
  if (reduction == 2) {
    out->set_dtype(x.dtype());
    out->set_dims(x.dims());
  } else {
    out->set_dims(common::make_ddim({}));
    out->set_dtype(x.dtype());
  }
}

void IncrementInferMeta(const MetaTensor& x, float value, MetaTensor* out) {
  PADDLE_ENFORCE_EQ(
      product(x.dims()),
      1UL,
      errors::InvalidArgument("The number of elements in Input(X) should be 1."
                              "Now the number is %d.",
                              product(x.dims())));
  out->set_dims(x.dims());
  out->share_lod(x);
  out->set_layout(x.layout());
  out->set_dtype(x.dtype());
}

static phi::DDim ValidateShape(const std::vector<int64_t> shape,
                               const phi::DDim& in_dims) {
  const int64_t in_size = common::product(in_dims);
  auto in_dims_vec = common::vectorize(in_dims);
  std::vector<int64_t> output_shape(shape.size(), 0);
  int64_t capacity = 1;
  int unk_dim_idx = -1;

  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] == -1) {
      // only one dimension can be set to -1, whose size will be inferred.
      PADDLE_ENFORCE_EQ(
          unk_dim_idx,
          -1,
          common::errors::InvalidArgument(
              "Only one dimension value of 'shape' in ReshapeOp can "
              "be -1. But received shape = [%s], shape[%d] is also -1.",
              common::make_ddim(shape),
              i));
      unk_dim_idx = static_cast<int>(i);
      output_shape[i] = shape[i];
    } else if (shape[i] == 0) {
      if (static_cast<int>(i) < in_dims.size()) {
        if (in_size == 0) {
          // such as [3, 2, 0] -> [0, 0] is [0, 0]; [3, 2, 0] -> [10, 0] is [10,
          // 0]
          output_shape[i] = 0;
        } else {
          // such as [3, 2, 1] -> [0, 0] is [3, 2]; [3, 2, 1] -> [3, 2, 0] is
          // [3, 2, 1]
          output_shape[i] = in_dims[static_cast<int>(i)];
        }
      } else {
        PADDLE_ENFORCE_EQ(in_size,
                          0,
                          common::errors::InvalidArgument(
                              "If The index of 0 in `shape` >= "
                              "the input tensor X's dimensions, "
                              "It can only be Zero-Sized Tensor"));
      }
      capacity *= output_shape[i];
    } else {
      PADDLE_ENFORCE_GT(
          shape[i],
          0,
          common::errors::InvalidArgument(
              "Each dimension value of 'shape' in ReshapeOp must not "
              "be negative except one unknown dimension. "
              "But received  shape = [%s], shape[%d] = %d.",
              common::make_ddim(shape),
              i,
              shape[i]));
      output_shape[i] = shape[i];
      capacity *= output_shape[i];
    }
  }

  if (capacity == 0) {
    PADDLE_ENFORCE_EQ(in_size,
                      0,
                      common::errors::InvalidArgument(
                          "Only Zero-Size Tensor'shape can contain 0"));
    PADDLE_ENFORCE_EQ(unk_dim_idx,
                      -1,
                      common::errors::InvalidArgument(
                          "can not reshape %s to %s, because the unspecified "
                          "dimension %i can be any number and is ambiguous",
                          in_dims,
                          common::make_ddim(shape),
                          unk_dim_idx));
  }

  bool no_negative = std::all_of(in_dims_vec.cbegin(),
                                 in_dims_vec.cend(),
                                 [](int64_t i) { return i >= 0; });
  if (unk_dim_idx != -1) {
    // in compile time, no_negative may be False.
    if (no_negative) {
      output_shape[unk_dim_idx] = in_size / capacity;
      PADDLE_ENFORCE_EQ(
          output_shape[unk_dim_idx] * capacity,
          in_size,
          common::errors::InvalidArgument(
              "The 'shape' attribute in ReshapeOp is invalid. "
              "The input tensor X'size must be divisible by known "
              "capacity of 'shape'. "
              "But received X's shape = [%s], X's size = %d, "
              "'shape' is [%s], known capacity of 'shape' is %d.",
              in_dims,
              in_size,
              common::make_ddim(shape),
              capacity));
    } else {
      // such as [-1, 8, 3]->[-1, 8], out_shape will remain [-1, 8]
      output_shape[unk_dim_idx] = -1;
    }
  } else {
    if (no_negative) {
      PADDLE_ENFORCE_EQ(
          capacity,
          in_size,
          common::errors::InvalidArgument(
              "The 'shape' in ReshapeOp is invalid. "
              "The input tensor X'size must be equal to the capacity of "
              "'shape'. "
              "But received X's shape = [%s], X's size = %d, 'shape' is "
              "[%s], the capacity of 'shape' is %d.",
              in_dims,
              in_size,
              common::make_ddim(shape),
              capacity));
    }
  }

  return common::make_ddim(output_shape);
}

void InferMetaFromVecValue(const MetaTensor& x,
                           const std::vector<int64_t>& shape,
                           MetaTensor* out) {
  auto x_dims = x.dims();
  auto out_dims = ValidateShape(shape, x_dims);
  out->set_dims(out_dims);
  out->set_dtype(x.dtype());
  out->set_layout(x.layout());
  if (x_dims.size() > 0 && (x_dims[0] == out_dims[0])) {
    // Only pass LoD when the first dimension of output and Input(X)
    // are the same.
    out->share_lod(x);
  }
}

void InverseInferMeta(const MetaTensor& x, MetaTensor* out) {
  auto input_dims = x.dims();
  int input_rank = static_cast<int>(input_dims.size());
  PADDLE_ENFORCE_GE(
      input_rank,
      2,
      errors::InvalidArgument(
          "The dimension of Input(Input) is expected to be no less than 2. "
          "But received: Input(Input)'s dimension = %d, shape = [%s].",
          input_rank,
          input_dims));
  for (int i = 0; i < input_rank; ++i) {
    PADDLE_ENFORCE_EQ(
        (input_dims[i] == -1) || (input_dims[i] > 0),
        true,
        errors::InvalidArgument(
            "Each dimension of input tensor is expected to be -1 or a "
            "positive number, but received %d. Input's shape is [%s].",
            input_dims[i],
            input_dims));
  }
  if (input_dims[input_rank - 2] > 0 && input_dims[input_rank - 1] > 0) {
    PADDLE_ENFORCE_EQ(input_dims[input_rank - 2],
                      input_dims[input_rank - 1],
                      errors::InvalidArgument(
                          "The last two dimensions are expected to be equal. "
                          "But received: %d and %d; "
                          "Input(Input)'s shape = [%s].",
                          input_dims[input_rank - 2],
                          input_dims[input_rank - 1],
                          input_dims));
  }

  out->set_dims(input_dims);
  out->set_dtype(x.dtype());
  out->share_lod(x);
}

void IsEmptyInferMeta(const MetaTensor& x, MetaTensor* out) {
  out->set_dims(common::make_ddim({}));
  out->set_dtype(DataType::BOOL);
}

void IsfiniteInferMeta(const MetaTensor& x, MetaTensor* out) {
  out->set_dims(x.dims());
  out->set_dtype(DataType::BOOL);
}

void KthvalueInferMeta(const MetaTensor& x,
                       int k,
                       int axis,
                       bool keepdim,
                       MetaTensor* out,
                       MetaTensor* indices,
                       MetaConfig config) {
  auto input_dims = x.dims();
  const int& dim_size = input_dims.size();
  if (dim_size > 0) {
    PADDLE_ENFORCE_LT(axis,
                      dim_size,
                      common::errors::InvalidArgument(
                          "the axis must be [-%d, %d), but received %d .",
                          dim_size,
                          dim_size,
                          axis));
    PADDLE_ENFORCE_GE(axis,
                      -dim_size,
                      common::errors::InvalidArgument(
                          "the axis must be [-%d, %d), but received %d .",
                          dim_size,
                          dim_size,
                          axis));
  } else if (dim_size == 0) {
    // 0-dim tensor
    PADDLE_ENFORCE_EQ(axis == 0 || axis == -1,
                      true,
                      common::errors::InvalidArgument(
                          "'axis'(%d) must be 0 or -1 if input tensor is "
                          "0-dim.",
                          axis));
  }
  if (axis < 0) axis += dim_size;
  PADDLE_ENFORCE_GE(
      k,
      1,
      common::errors::InvalidArgument(
          "the k in the kthvalue must >= 1, but received %d .", k));
  PADDLE_ENFORCE_GE(input_dims.size(),
                    0,
                    common::errors::InvalidArgument(
                        "input of kthvalue must have >= 0d shape"));
  if (dim_size > 0 && config.is_runtime) {
    PADDLE_ENFORCE_GE(
        input_dims[axis],
        k,
        common::errors::InvalidArgument(
            "input of kthvalue must have >= %d columns in axis of %d",
            k,
            axis));
  }
  std::vector<int64_t> dimvec;
  for (int i = 0; i < axis; i++) {
    dimvec.emplace_back(input_dims[i]);
  }
  if (keepdim && dim_size > 0) {
    dimvec.emplace_back(static_cast<int64_t>(1));
  }
  for (int i = axis + 1; i < dim_size; i++) {
    dimvec.emplace_back(input_dims[i]);
  }
  DDim dims = common::make_ddim(dimvec);
  out->set_dims(dims);
  out->share_lod(x);
  out->set_dtype(x.dtype());
  indices->set_dims(dims);
  indices->share_lod(x);
  indices->set_dtype(DataType::INT64);
}

void LogicalNotInferMeta(const MetaTensor& x, MetaTensor* out) {
  UnchangedInferMeta(x, out);
  if (!(out->is_same_tensor(x))) {
    out->set_dtype(DataType::BOOL);
  }
}

void LogsumexpInferMeta(const MetaTensor& input,
                        const std::vector<int>& axis_in,
                        bool keepdim,
                        bool reduce_all,
                        MetaTensor* out) {
  auto input_rank = input.dims().size();
  // only support 0~4D, due to eigen template compile slow
  PADDLE_ENFORCE_LE(
      input_rank,
      4,
      errors::InvalidArgument("The input tensor X's dimensions of logsumexp "
                              "should be less or equal than 4. "));
  std::vector<int64_t> axis;
  axis.reserve(axis_in.size());
  std::for_each(axis_in.begin(), axis_in.end(), [&axis](const int& t) {
    axis.push_back(static_cast<int64_t>(t));
  });
  ReduceInferMetaBase(input, axis, keepdim, reduce_all, out);
}

void MatrixPowerInferMeta(const MetaTensor& x, int n, MetaTensor* out) {
  auto dims = x.dims();
  auto n_dim = dims.size();
  PADDLE_ENFORCE_GE(n_dim,
                    2,
                    common::errors::InvalidArgument(
                        "The Input(X) should have at least 2 dimensions. But "
                        "received a %d dimension tensor.",
                        n_dim));
  for (int i = 0; i < n_dim; ++i)
    PADDLE_ENFORCE_GE(
        dims[i],
        0,
        common::errors::InvalidArgument(
            "The size of Input(X) should greater than or equal to 0."));
  PADDLE_ENFORCE_EQ(dims[n_dim - 2],
                    dims[n_dim - 1],
                    common::errors::InvalidArgument(
                        "The inner-most 2 dimensions of Input(X) all should "
                        "be square matrices "
                        "But received X's shape[-2] = %d and shape[-1] = %d.",
                        dims[n_dim - 2],
                        dims[n_dim - 1]));
  out->set_dims(dims);
  out->share_lod(x);
  out->set_dtype(x.dtype());
}

void LUInferMeta(const MetaTensor& x,
                 bool pivot,
                 MetaTensor* out,
                 MetaTensor* pivots,
                 MetaTensor* infos) {
  auto x_dims = x.dims();
  int x_rank = x_dims.size();

  PADDLE_ENFORCE_NOT_NULL(
      out,
      common::errors::InvalidArgument("Output(Out) should not be nullptr."));
  PADDLE_ENFORCE_GE(x_rank,
                    2,
                    common::errors::InvalidArgument(
                        "The rank of input must greater than 2."));
  out->set_dims(x_dims);
  out->set_dtype(x.dtype());
  int m = static_cast<int>(x_dims[x_rank - 1]);
  int n = static_cast<int>(x_dims[x_rank - 2]);
  int min_mn = std::min(m, n);
  auto dims_vec = common::vectorize(x_dims);
  PADDLE_ENFORCE_NOT_NULL(
      infos,
      common::errors::InvalidArgument("Output(Infos) should not be nullptr."));
  if (x_rank == 2) {
    infos->set_dims(common::make_ddim({}));
  } else {
    auto Infos_dim =
        std::vector<int>(dims_vec.begin(), dims_vec.begin() + x_rank - 2);
    infos->set_dims(common::make_ddim(Infos_dim));
  }
  infos->set_dtype(DataType::INT32);
  if (pivot) {
    PADDLE_ENFORCE_NOT_NULL(pivots,
                            common::errors::InvalidArgument(
                                "Output(Pivots) should not be nullptr."));
    auto Pivots_dim =
        std::vector<int>(dims_vec.begin(), dims_vec.begin() + x_rank - 1);
    Pivots_dim[x_rank - 2] = min_mn;
    pivots->set_dims(common::make_ddim(Pivots_dim));
    pivots->set_dtype(DataType::INT32);
  }
}

void MatrixRankInferMeta(const MetaTensor& x,
                         bool use_default_tol,
                         bool hermitian,
                         MetaTensor* out) {
  auto dim_x = x.dims();
  PADDLE_ENFORCE_GE(dim_x.size(),
                    2,
                    common::errors::InvalidArgument(
                        "The dims of input must be greater than 2."));

  if (hermitian) {
    int rows = static_cast<int>(dim_x[dim_x.size() - 2]);
    int cols = static_cast<int>(dim_x[dim_x.size() - 1]);
    PADDLE_ENFORCE_EQ(rows,
                      cols,
                      common::errors::InvalidArgument(
                          "if hermitian == true, matrix should be n*n"));
  }
  DDim dim_x_batch = detail::CheckAndGetOutputDim(dim_x);
  out->set_dims(dim_x_batch);
  out->share_lod(x);
}

void MaxOutInferMeta(const MetaTensor& x,
                     int groups,
                     int axis,
                     MetaTensor* out) {
  auto in_x_dims = x.dims();
  // check groups > 1
  PADDLE_ENFORCE_GT(
      groups,
      1,
      common::errors::InvalidArgument("Attr(groups) of Op(maxout) should be "
                                      "larger than 1. But received %d.",
                                      groups));
  PADDLE_ENFORCE_EQ(
      axis == 1 || axis == -1 || axis == 3,
      true,
      common::errors::InvalidArgument(
          "axis only supported 1, -1 or 3, but received axis is: %d.", axis));
  PADDLE_ENFORCE_EQ(in_x_dims.size(),
                    4,
                    common::errors::InvalidArgument(
                        "x's dims should be 4, but received x's dims is: %d.",
                        in_x_dims.size()));

  if (axis < 0) {
    axis += in_x_dims.size();
  }
  PADDLE_ENFORCE_EQ(
      in_x_dims[axis] % groups,
      0,
      common::errors::InvalidArgument(
          "The number of input channels for Op(maxout) "
          "should be divisible by Attr(groups). But received: the "
          "input's channels is [%d], the shape of input is [%s], "
          "the Attr(groups) is [%d], the Attr(axis) is [%d]. The "
          "error may come from wrong Attr(groups) or Attr(axis) setting.",
          in_x_dims[axis],
          in_x_dims,
          groups,
          axis));
  std::vector<int64_t> output_shape(
      {in_x_dims[0], in_x_dims[1], in_x_dims[2], in_x_dims[3]});
  output_shape[axis] = in_x_dims[axis] / groups;
  out->set_dims(common::make_ddim(output_shape));
  out->set_dtype(x.dtype());
}

void MaxPoolWithIndexInferMeta(const MetaTensor& x,
                               const std::vector<int>& kernel_size,
                               const std::vector<int>& strides,
                               const std::vector<int>& paddings,
                               bool global_pooling,
                               bool adaptive,
                               bool ceil_mode,
                               MetaTensor* out,
                               MetaTensor* mask,
                               MetaConfig config) {
  std::vector<int> paddings_ = paddings;
  std::vector<int> kernel_size_ = kernel_size;

  auto x_dims = x.dims();

  PADDLE_ENFORCE_EQ(
      (x_dims.size() == 4 || x_dims.size() == 5),
      true,
      errors::InvalidArgument("Pooling input should be 4-D or "
                              "5-D tensor but received %dD-Tensor",
                              x_dims.size()));

  if (global_pooling) {
    kernel_size_.resize(static_cast<size_t>(x_dims.size()) - 2);
    for (int i = 0; i < static_cast<int>(kernel_size_.size()); ++i) {
      paddings_[i] = 0;
      kernel_size_[i] = static_cast<int>(x_dims[i + 2]);
    }
  }

  PADDLE_ENFORCE_EQ(
      x_dims.size() - kernel_size_.size(),
      2U,
      errors::InvalidArgument(
          "The input size %d minus the kernel size %d should equal to 2.",
          x_dims.size(),
          kernel_size_.size()));
  PADDLE_ENFORCE_EQ(
      kernel_size_.size(),
      strides.size(),
      errors::InvalidArgument(
          "Strides size %d and pooling size %d should be the same.",
          strides.size(),
          kernel_size_.size()));
  PADDLE_ENFORCE_EQ(
      kernel_size_.size(),
      paddings_.size(),
      errors::InvalidArgument(
          "Paddings size %d and pooling size %d should be the same.",
          paddings_.size(),
          kernel_size_.size()));

  std::vector<int64_t> output_shape({x_dims[0], x_dims[1]});
  if (adaptive) {
    output_shape.insert(
        output_shape.end(), kernel_size_.begin(), kernel_size_.end());
  } else {
    for (int i = 0; i < static_cast<int>(kernel_size_.size()); ++i) {
      if ((!config.is_runtime) && (x_dims[i + 2] < 0)) {
        output_shape.push_back(x_dims[i + 2]);
      } else {
        output_shape.push_back(
            funcs::MaxPoolOutputSize(static_cast<int>(x_dims[i + 2]),
                                     kernel_size_[i],
                                     paddings_[i],
                                     strides[i],
                                     ceil_mode));
      }
    }
  }

  out->set_dims(common::make_ddim(output_shape));
  out->set_dtype(x.dtype());

  mask->set_dims(common::make_ddim(output_shape));
  mask->set_dtype(phi::CppTypeToDataType<int>::Type());
}

void MaxPoolV2InferMeta(const MetaTensor& x,
                        const std::vector<int>& kernel_size,
                        const std::vector<int>& strides,
                        const std::vector<int>& paddings,
                        const std::string& data_format,
                        bool global_pooling,
                        bool adaptive,
                        MetaTensor* out,
                        MetaTensor* saved_idx,
                        MetaConfig config) {
  PADDLE_ENFORCE_EQ(adaptive,
                    false,
                    common::errors::InvalidArgument(
                        "max_pool2d_v2 op does not support adaptive."));
  Pool2DInferMeta(x,
                  kernel_size,
                  strides,
                  paddings,
                  false,
                  false,
                  data_format,
                  "max",
                  global_pooling,
                  adaptive,
                  "EXPLICIT",
                  out,
                  config);
  saved_idx->set_dims(out->dims());
  saved_idx->set_dtype(phi::CppTypeToDataType<int>::Type());
}

void MeanAllInferMeta(const MetaTensor& x, MetaTensor* out) {
  out->set_dims(common::make_ddim({}));
  out->set_dtype(x.dtype());
  out->set_layout(x.layout());
}

void ModeInferMeta(const MetaTensor& x,
                   int axis,
                   bool keepdim,
                   MetaTensor* out,
                   MetaTensor* indices) {
  auto input_dims = x.dims();
  const int& dim_size = input_dims.size();
  PADDLE_ENFORCE_LE(axis,
                    dim_size,
                    common::errors::InvalidArgument(
                        "the axis must be [-%d, %d), but received %d .",
                        dim_size,
                        dim_size,
                        axis));
  if (dim_size > 0) {
    PADDLE_ENFORCE_GE(axis,
                      -dim_size,
                      common::errors::InvalidArgument(
                          "the axis must be [-%d, %d), but received %d .",
                          dim_size,
                          dim_size,
                          axis));
  }
  PADDLE_ENFORCE_GE(
      input_dims.size(),
      0,
      errors::InvalidArgument("input of ModeOp must have >= 0d shape"));
  if (axis < 0) {
    axis += dim_size;
    if (dim_size != 0) {
      PADDLE_ENFORCE_GE(axis,
                        0,
                        common::errors::InvalidArgument(
                            "the axis must be [-%d, %d), but received %d .",
                            dim_size,
                            dim_size,
                            axis - dim_size));
    }
  }

  std::vector<int64_t> dimvec;
  for (int i = 0; i < axis; i++) {
    dimvec.emplace_back(input_dims[i]);
  }
  if (keepdim && dim_size > 0) {
    dimvec.emplace_back(static_cast<int64_t>(1));
  }
  for (int i = axis + 1; i < dim_size; i++) {
    dimvec.emplace_back(input_dims[i]);
  }
  DDim dims = common::make_ddim(dimvec);
  out->set_dims(dims);
  out->share_lod(x);
  out->set_dtype(x.dtype());

  indices->set_dims(dims);
  indices->share_lod(x);
  indices->set_dtype(DataType::INT64);
}

void MultinomialInferMeta(const MetaTensor& x,
                          const Scalar& num_samples,
                          bool replacement,
                          MetaTensor* out,
                          MetaConfig config) {
  auto int_num_samples = num_samples.to<int>();
  auto x_dim = x.dims();
  int x_rank = static_cast<int>(x_dim.size());
  PADDLE_ENFORCE_GT(x_rank,
                    0,
                    errors::InvalidArgument(
                        "The number of dimensions of the input probability "
                        "distribution should be > 0, but got %d.",
                        x_rank));
  PADDLE_ENFORCE_LE(x_rank,
                    2,
                    errors::InvalidArgument(
                        "The number of dimensions of the input probability "
                        "distribution should be <= 2, but got %d.",
                        x_rank));

  std::vector<int64_t> out_dims(x_rank);
  for (int i = 0; i < x_rank - 1; i++) {
    out_dims[i] = x_dim[i];
  }

  if (config.is_runtime || !num_samples.FromTensor()) {
    PADDLE_ENFORCE_GT(int_num_samples,
                      0,
                      errors::InvalidArgument(
                          "The number of samples should be > 0, but got %d.",
                          int_num_samples));
    out_dims[x_rank - 1] = int_num_samples;
  } else {
    out_dims[x_rank - 1] = -1;
  }

  out->set_dims(common::make_ddim(out_dims));
  out->set_dtype(DataType::INT64);
}

void NanmedianInferMeta(const MetaTensor& x,
                        const IntArray& axes,
                        bool keep_dim,
                        const std::string& mode,
                        MetaTensor* out,
                        MetaTensor* median_index) {
  std::vector<int64_t> axis_list = axes.GetData();
  auto x_dim = x.dims();
  int64_t x_rank = x_dim.size();

  std::vector<int32_t> out_dim;
  if (axis_list.empty()) {
    if (keep_dim) {
      for (int64_t i = 0; i < x_rank; i++) {
        out_dim.push_back(1);
      }
    }
  } else {
    std::vector<int64_t> formatted_axis;
    for (auto& axis : axis_list) {
      if (x_rank == 0) {
        PADDLE_ENFORCE_EQ(axis == 0 || axis == -1,
                          true,
                          common::errors::InvalidArgument(
                              "When input 0D Tensor, each element of the axis "
                              "can only be -1, 0, None"));
      } else {
        PADDLE_ENFORCE_LT(axis,
                          x_rank,
                          errors::InvalidArgument(
                              "each element of the axis should be in the "
                              "range [ -dimension(X), dimension(X) ) "
                              "which dimension = %d. But received axis = %d.",
                              x_rank,
                              axis));
        PADDLE_ENFORCE_GE(axis,
                          -x_rank,
                          errors::InvalidArgument(
                              "each element of the axis should be in the "
                              "range [ -dimension(X), dimension(X) ) "
                              "which dimension = %d. But received axis = %d.",
                              x_rank,
                              axis));
      }
      if (axis < 0) axis += x_rank;
      PADDLE_ENFORCE_EQ(
          std::find(formatted_axis.begin(), formatted_axis.end(), axis),
          formatted_axis.end(),
          errors::InvalidArgument("Attr(axes) has duplicated elements: %d.",
                                  static_cast<int>(axis)));

      formatted_axis.push_back(axis);
    }

    for (int64_t i = 0; i < x_rank; i++) {
      if (std::find(formatted_axis.begin(), formatted_axis.end(), i) ==
          formatted_axis.end()) {
        out_dim.push_back(x_dim[i]);  // NOLINT
      } else if (keep_dim) {
        out_dim.push_back(1);
      }
    }
  }
  out->set_dtype(x.dtype());
  out->set_dims(make_ddim(out_dim));

  auto median_dim = out_dim;
  if (mode == "avg") {
    median_dim.push_back(2);
  }
  median_index->set_dtype(DataType::INT64);
  median_index->set_dims(make_ddim(median_dim));
}

void NMSInferMeta(const MetaTensor& x, float threshold, MetaTensor* out) {
  auto boxes_dim = x.dims();
  PADDLE_ENFORCE_EQ(boxes_dim.size(),
                    2,
                    common::errors::InvalidArgument(
                        "The Input Boxes must be 2-dimension "
                        "whose shape must be [N, 4] "
                        "N is the number of boxes "
                        "in last dimension in format [x1, x2, y1, y2]. "));
  out->set_dims(common::make_ddim({-1}));
  out->set_dtype(DataType::INT64);
}

void NonZeroInferMeta(const MetaTensor& condition, MetaTensor* out) {
  auto rank = condition.dims().size();
  PADDLE_ENFORCE_GE(
      rank,
      1UL,
      common::errors::InvalidArgument(
          "Input(Condition) should have number of dimension at least 1"));
  out->set_dims(common::make_ddim({-1, rank}));
  out->set_dtype(DataType::INT64);
}

void NormInferMeta(const MetaTensor& x,
                   int axis,
                   float epsilon,
                   bool is_test,
                   MetaTensor* out,
                   MetaTensor* norm) {
  auto xdim = x.dims();
  out->set_dims(x.dims());
  out->set_dtype(x.dtype());

  if (is_test == false) {
    if (axis < 0) axis = xdim.size() + axis;
    xdim[axis] = 1;
    norm->set_dims(xdim);
    norm->set_dtype(x.dtype());
  }
}

void OneHotRawInferMeta(const MetaTensor& x,
                        const Scalar& depth,
                        DataType dtype,
                        bool allow_out_of_range,
                        MetaTensor* out) {
  auto x_dims = x.dims();
  PADDLE_ENFORCE_GE(x_dims.size(),
                    0,
                    common::errors::InvalidArgument(
                        "Rank of Input(X) should be at least 0."));
  auto out_dims_vec = common::vectorize(x_dims);
  out_dims_vec.push_back(depth.to<int>());
  auto out_dims = common::make_ddim(out_dims_vec);
  out->set_dims(out_dims);
  out->share_lod(x);
  out->set_dtype(dtype);
}

void OneHotInferMeta(const MetaTensor& x,
                     const Scalar& num_classes,
                     MetaTensor* out) {
  auto x_dims = x.dims();
  PADDLE_ENFORCE_GE(x_dims.size(),
                    0,
                    common::errors::InvalidArgument(
                        "Rank of Input(X) should be at least 0."));

  int num_classes_int = num_classes.to<int>();
  auto out_dims_vec = common::vectorize(x_dims);
  out_dims_vec.push_back(num_classes_int);
  auto out_dims = common::make_ddim(out_dims_vec);

  out->set_dims(out_dims);
  out->share_lod(x);
  out->set_dtype(phi::DataType::FLOAT32);
}

void OverlapAddInferMeta(const MetaTensor& x,
                         int hop_length,
                         int axis,
                         MetaTensor* out,
                         MetaConfig config) {
  const auto x_dims = x.dims();
  const int x_rank = x_dims.size();

  PADDLE_ENFORCE_GE(
      x_rank,
      2,
      errors::InvalidArgument(
          "Input(X) of OverlapAddOp should be a tensor which contains "
          "at least 2 dimensions, but got rank %s.",
          x_rank));

  PADDLE_ENFORCE_GT(
      hop_length,
      0,
      errors::InvalidArgument(
          "Attribute(hop_length) of OverlapAddOp should be greater "
          "than 0, but got %s.",
          hop_length));

  PADDLE_ENFORCE_EQ(
      (axis == 0 || axis == -1),
      true,
      errors::InvalidArgument(
          "Attribute(axis) of OverlapAddOp should 0 or -1, but got %s.", axis));

  std::vector<int64_t> output_shape;
  int n_frames = 0;
  int frame_length = 0;
  int seq_length = 0;

  int start_axis = 0;
  int end_axis = 0;
  if (axis == 0) {
    n_frames = static_cast<int>(x_dims[0]);
    frame_length = static_cast<int>(x_dims[1]);
    start_axis = 2;
    end_axis = x_rank - 1;
  } else {
    n_frames = static_cast<int>(x_dims[x_rank - 1]);
    frame_length = static_cast<int>(x_dims[x_rank - 2]);
    start_axis = 0;
    end_axis = x_rank - 3;
  }

  bool contain_unknown_dim = common::contain_unknown_dim(x_dims);
  bool check = config.is_runtime || !contain_unknown_dim;
  if (check) {
    PADDLE_ENFORCE_LE(
        hop_length,
        frame_length,
        errors::InvalidArgument(
            "Attribute(hop_length) of OverlapAddOp should be less or equal "
            "than frame_length, but got hop_length(%s) > frame_length(%s).",
            hop_length,
            frame_length));
  }

  if (n_frames == -1) {
    seq_length = -1;
  } else {
    seq_length = (n_frames - 1) * hop_length + frame_length;
  }

  // It won't go into for loop when x_rank == 2U.
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

  out->set_dims(common::make_ddim(output_shape));
}

void PadInferMeta(const MetaTensor& input,
                  const std::vector<int>& paddings,
                  const Scalar& padding_value,
                  MetaTensor* out,
                  MetaConfig config) {
  auto x_dim = input.dims();
  PADDLE_ENFORCE_EQ(
      static_cast<int>(paddings.size()),
      x_dim.size() * 2,
      common::errors::InvalidArgument(
          "Size of 'paddings' dimension should be equal to 2 * size of "
          "Input(X)'s dimension, but received (size of 'paddings' dimension "
          "is) %d vs (2 * size of Input(X)'s dimension is) %d.",
          static_cast<int>(paddings.size()),
          x_dim.size() * 2));
  for (size_t i = 0; i < paddings.size(); ++i) {
    PADDLE_ENFORCE_GE(paddings[i],
                      0,
                      common::errors::InvalidArgument(
                          "The element of 'paddings' should >= 0, but "
                          "received %d for index %d.",
                          paddings[i],
                          static_cast<int>(i)));
  }
  std::vector<int64_t> out_dims(x_dim.size());
  for (int i = 0; i < x_dim.size(); ++i) {
    if ((!config.is_runtime) && (x_dim[i] == -1)) {
      out_dims[i] = -1;
    } else {
      out_dims[i] = x_dim[i] + paddings[i * 2] + paddings[i * 2 + 1];
    }
  }
  out->set_dims(common::make_ddim(out_dims));
  if (out_dims[0] == x_dim[0]) {
    // Only pass LoD when the first dimension is equal between
    // output and input.
    out->share_lod(input);
  }
  out->set_dtype(input.dtype());
}

void Pad3dInferMeta(const MetaTensor& x,
                    const IntArray& paddings_int_array,
                    const std::string& mode,
                    float value,
                    const std::string& data_format,
                    MetaTensor* out,
                    MetaConfig config) {
  auto x_dim = x.dims();
  PADDLE_ENFORCE_EQ(x_dim.size(),
                    5,
                    errors::InvalidArgument(
                        "The size of Input(X)'s dimension should be equal to "
                        "5, but received %d. ",
                        x_dim.size()));

  std::vector<int64_t> out_dims(x_dim.size(), -1);
  out_dims[0] = x_dim[0];
  auto& paddings = paddings_int_array.GetData();
  if (data_format == "NCDHW") {
    out_dims[1] = x_dim[1];
  } else {
    out_dims[4] = x_dim[4];
  }
  if (paddings_int_array.FromTensor()) {
    if (config.is_runtime) {
      PADDLE_ENFORCE_EQ(
          paddings.size(),
          6,
          errors::InvalidArgument("Shape of Input(Paddings) should be equal to "
                                  "[6], but received [%d].",
                                  paddings.size()));
      if (data_format == "NCDHW") {
        out_dims[2] = x_dim[2] + paddings[4] + paddings[5];
        out_dims[3] = x_dim[3] + paddings[2] + paddings[3];
        out_dims[4] = x_dim[4] + paddings[0] + paddings[1];
      } else {
        out_dims[1] = x_dim[1] + paddings[4] + paddings[5];
        out_dims[2] = x_dim[2] + paddings[2] + paddings[3];
        out_dims[3] = x_dim[3] + paddings[0] + paddings[1];
      }
    }
  } else {
    PADDLE_ENFORCE_EQ(
        paddings.size(),
        6,
        errors::InvalidArgument(
            "Size of paddings should be equal to 6, but received %d.",
            static_cast<int>(paddings.size())));
    if (data_format == "NCDHW") {
      out_dims[2] = ((!config.is_runtime) && (x_dim[2] < 0))
                        ? x_dim[2]
                        : (x_dim[2] + paddings[4] + paddings[5]);  // depth

      out_dims[3] = ((!config.is_runtime) && (x_dim[3] < 0))
                        ? x_dim[3]
                        : (x_dim[3] + paddings[2] + paddings[3]);  // height

      out_dims[4] = ((!config.is_runtime) && (x_dim[4] < 0))
                        ? x_dim[4]
                        : (x_dim[4] + paddings[0] + paddings[1]);  // width
    } else {                                                       // NDHWC
      out_dims[1] = ((!config.is_runtime) && (x_dim[1] < 0))
                        ? x_dim[1]
                        : (x_dim[1] + paddings[4] + paddings[5]);  // depth
      out_dims[2] = ((!config.is_runtime) && (x_dim[2] < 0))
                        ? x_dim[2]
                        : (x_dim[2] + paddings[2] + paddings[3]);  // height
      out_dims[3] = ((!config.is_runtime) && (x_dim[3] < 0))
                        ? x_dim[3]
                        : (x_dim[3] + paddings[0] + paddings[1]);  // width
    }
  }

  out->set_dims(common::make_ddim(out_dims));
  out->set_dtype(x.dtype());
  out->share_lod(x);
}

void PartialAllgatherInferMeta(const MetaTensor& x,
                               int nranks,
                               int rank,
                               MetaTensor* out) {
  PADDLE_ENFORCE_GE(
      nranks,
      2,
      common::errors::InvalidArgument("The value of nranks should be >=2."));
  PADDLE_ENFORCE_EQ(
      (rank >= 0 && rank < nranks),
      true,
      common::errors::InvalidArgument(
          "The rank (%d) for partial_allgather op must >=0 and <nranks (%d)",
          rank,
          nranks));

  auto x_dims = x.dims();
  out->set_dims(x_dims);
  out->set_dtype(x.dtype());
}

void PartialSendInferMeta(const MetaTensor& x, int peer, int num, int id) {
  PADDLE_ENFORCE_GE(
      peer,
      0,
      common::errors::InvalidArgument(
          "The peer (%d) for partial_send op must be non-negative.", peer));
  PADDLE_ENFORCE_GE(num,
                    1,
                    common::errors::InvalidArgument(
                        "The num (%d) for partial_send op must >=1", num));
  PADDLE_ENFORCE_EQ(
      (id >= 0 && id < num),
      true,
      common::errors::InvalidArgument(
          "The id (%d) for partial_send op must >=0 and <num (%d)", id, num));
}

void PixelShuffleInferMeta(const MetaTensor& x,
                           int upscale_factor,
                           const std::string& data_format,
                           MetaTensor* out) {
  auto input_dims = x.dims();
  PADDLE_ENFORCE_EQ(input_dims.size(),
                    4,
                    common::errors::InvalidArgument(
                        "Input should be a 4-D tensor of format [N, C, H, W] "
                        "or [N, H, W, C], but got %u.",
                        input_dims.size()));
  PADDLE_ENFORCE_NE(
      upscale_factor,
      0,
      common::errors::InvalidArgument("upscale_factor should not be 0."));

  const bool channel_last = (data_format == "NHWC");

  if (!channel_last) {
    PADDLE_ENFORCE_EQ(input_dims[1] % (upscale_factor * upscale_factor),
                      0,
                      common::errors::InvalidArgument(
                          "The square of upscale_factor[%u] should divide the "
                          "number of channel[%u]",
                          upscale_factor * upscale_factor,
                          input_dims[1]));
  } else {
    PADDLE_ENFORCE_EQ(input_dims[3] % (upscale_factor * upscale_factor),
                      0,
                      common::errors::InvalidArgument(
                          "The square of upscale_factor[%u] should divide the "
                          "number of channel[%u]",
                          upscale_factor * upscale_factor,
                          input_dims[3]));
  }
  auto output_dims = input_dims;
  output_dims[0] = input_dims[0];
  if (!channel_last) {
    output_dims[1] = input_dims[1] / (upscale_factor * upscale_factor);
    output_dims[2] = input_dims[2] * upscale_factor;
    output_dims[3] = input_dims[3] * upscale_factor;
  } else {
    output_dims[1] = input_dims[1] * upscale_factor;
    output_dims[2] = input_dims[2] * upscale_factor;
    output_dims[3] = input_dims[3] / (upscale_factor * upscale_factor);
  }
  out->set_dtype(x.dtype());
  out->set_dims(output_dims);
}

void PixelShuffleGradInferMeta(const MetaTensor& out_grad,
                               int upscale_factor,
                               const std::string& data_format,
                               MetaTensor* x_grad) {
  auto do_dims = out_grad.dims();
  PADDLE_ENFORCE_EQ(do_dims.size(),
                    4,
                    common::errors::InvalidArgument(
                        "Input should be a 4-D tensor of format [N, C, H, W] "
                        "or [N, H, W, C], but got %u.",
                        do_dims.size()));

  const bool channel_last = (data_format == "NHWC");

  auto dx_dims = do_dims;
  dx_dims[0] = do_dims[0];

  if (!channel_last) {
    dx_dims[1] = do_dims[1] * (upscale_factor * upscale_factor);
    dx_dims[2] = do_dims[2] / upscale_factor;
    dx_dims[3] = do_dims[3] / upscale_factor;
  } else {
    dx_dims[1] = do_dims[1] / upscale_factor;
    dx_dims[2] = do_dims[2] / upscale_factor;
    dx_dims[3] = do_dims[3] * (upscale_factor * upscale_factor);
  }
  x_grad->set_dims(dx_dims);
  x_grad->set_dtype(out_grad.dtype());
}

void PixelUnshuffleInferMeta(const MetaTensor& x,
                             int downscale_factor,
                             const std::string& data_format,
                             MetaTensor* out) {
  auto input_dims = x.dims();
  PADDLE_ENFORCE_EQ(input_dims.size(),
                    4,
                    common::errors::InvalidArgument(
                        "Input should be a 4-D tensor of format [N, C, H, W] "
                        "or [N, H, W, C], but got %u.",
                        input_dims.size()));
  PADDLE_ENFORCE_GE(downscale_factor,
                    1,
                    common::errors::InvalidArgument(
                        "downscale_factor should be larger than 0."));
  PADDLE_ENFORCE_EQ(data_format == "NCHW" || data_format == "NHWC",
                    true,
                    common::errors::InvalidArgument(
                        "data_format must be one of "
                        "NCHW and NHWC. But received data_format: %s",
                        data_format));

  const bool channel_last = (data_format == "NHWC");

  if (!channel_last) {
    PADDLE_ENFORCE_EQ((input_dims[2] % downscale_factor) == 0 &&
                          (input_dims[3] % downscale_factor) == 0,
                      true,
                      common::errors::InvalidArgument(
                          "Downscale factor[%u] should divide both "
                          "height[%u] and width[%u]",
                          downscale_factor,
                          input_dims[2],
                          input_dims[3]));
  } else {
    PADDLE_ENFORCE_EQ((input_dims[1] % downscale_factor) == 0 &&
                          (input_dims[2] % downscale_factor) == 0,
                      true,
                      common::errors::InvalidArgument(
                          "Downscale factor[%u] should divide both "
                          "height[%u] and width[%u]",
                          downscale_factor,
                          input_dims[1],
                          input_dims[2]));
  }
  auto output_dims = input_dims;
  output_dims[0] = input_dims[0];
  if (!channel_last) {
    output_dims[1] = input_dims[1] * (downscale_factor * downscale_factor);
    output_dims[2] = input_dims[2] / downscale_factor;
    output_dims[3] = input_dims[3] / downscale_factor;
  } else {
    output_dims[1] = input_dims[1] / downscale_factor;
    output_dims[2] = input_dims[2] / downscale_factor;
    output_dims[3] = input_dims[3] * (downscale_factor * downscale_factor);
  }
  out->set_dtype(x.dtype());
  out->set_dims(output_dims);
}

void PNormInferMeta(const MetaTensor& x,
                    float porder,
                    int axis,
                    float epsilon,
                    bool keepdim,
                    bool asvector,
                    MetaTensor* out) {
  auto x_dim = x.dims();
  auto x_rank = x_dim.size();

  PADDLE_ENFORCE_EQ((axis >= -x_rank && axis < x_rank) || x_rank == 0,
                    true,
                    errors::InvalidArgument(
                        "Attr(axis) value should be in range [-R, R-1], R is "
                        "the rank of Input(X). But received axis: %d, R: %d. "
                        "Current Input(X)'s shape is=[%s].",
                        axis,
                        x_rank,
                        x_dim));

  std::vector<int> out_dim_vector;
  if (asvector) {
    if (keepdim) {
      for (int i = 0; i < x_rank; ++i) {
        out_dim_vector.emplace_back(1);
      }
    } else {
      out_dim_vector = {};
    }
  } else {
    if (axis < 0) axis = axis + x_rank;
    if (keepdim) {
      for (int i = 0; i < x_dim.size(); ++i) {
        if (i != axis) {
          out_dim_vector.emplace_back(x_dim[i]);
        } else {
          out_dim_vector.emplace_back(1);
        }
      }
    } else {
      for (int i = 0; i < x_dim.size(); ++i) {
        if (i != axis) out_dim_vector.emplace_back(x_dim[i]);
      }
    }
  }

  out->set_dims(common::make_ddim(out_dim_vector));
  out->set_dtype(x.dtype());
}

void Pool2DInferMeta(const MetaTensor& x,
                     const IntArray& kernel_size,
                     const std::vector<int>& strides,
                     const std::vector<int>& paddings,
                     bool ceil_mode,
                     bool exclusive,
                     const std::string& data_format,
                     const std::string& pooling_type,
                     bool global_pooling,
                     bool adaptive,
                     const std::string& padding_algorithm,
                     MetaTensor* out,
                     MetaConfig config) {
  const bool channel_last = (config.is_run_mkldnn_kernel == false) &&
                            (data_format == "NHWC" || data_format == "NDHWC");
  if (!config.is_runtime && kernel_size.FromTensor()) {
    auto x_dims = x.dims();
    std::vector<int64_t> output_shape = common::vectorize(x_dims);
    // set dims of HW -1
    output_shape[x_dims.size() - 2] = -1;
    if (channel_last) {  // for NHWC, NDHWC
      output_shape[x_dims.size() - 3] = -1;
    } else {  // for NCHW
      output_shape[x_dims.size() - 1] = -1;
    }
    out->set_dims(common::make_ddim(output_shape));
    out->share_lod(x);
    out->set_dtype(x.dtype());
  } else {
    std::vector<int> kernel_size_val(kernel_size.GetData().begin(),
                                     kernel_size.GetData().end());
    PoolInferMeta(x,
                  kernel_size_val,
                  strides,
                  paddings,
                  ceil_mode,
                  exclusive,
                  data_format,
                  pooling_type,
                  global_pooling,
                  adaptive,
                  padding_algorithm,
                  out,
                  config);
  }
}

void PSendInferMeta(const MetaTensor& x, int peer) {
  LOG(INFO) << "SendBaseInferMeta begin";
  PADDLE_ENFORCE_GE(
      peer,
      0,
      errors::InvalidArgument(
          "The peer (%d) for p_send op must be non-negative.", peer));
}

void PSendArrayInferMeta(const MetaTensor& x, int peer) {
  PADDLE_ENFORCE_GE(
      peer,
      0,
      errors::InvalidArgument(
          "The peer (%d) for p_send op must be non-negative.", peer));
}

void SetInferMeta(const MetaTensor& x,
                  const std::vector<int64_t>& shape,
                  const std::vector<int64_t>& stride,
                  MetaTensor* out) {
  out->set_dtype(x.dtype());
  out->set_dims(common::make_ddim(shape));
  out->set_strides(common::make_ddim(stride));
}

void SendV2InferMeta(const int peer, const int ring_id) {
  PADDLE_ENFORCE_GE(
      peer,
      0,
      errors::InvalidArgument(
          "The peer (%d) for send_v2 op must be non-negative.", peer));
  PADDLE_ENFORCE_GE(
      ring_id,
      0,
      errors::InvalidArgument(
          "The ring_id (%d) for send_v2 op must be non-negative.", ring_id));
}

void PoolInferMeta(const MetaTensor& x,
                   const std::vector<int>& kernel_size,
                   const std::vector<int>& strides,
                   const std::vector<int>& paddings,
                   bool ceil_mode,
                   bool exclusive,
                   const std::string& data_format,
                   const std::string& pooling_type,
                   bool global_pooling,
                   bool adaptive,
                   const std::string& padding_algorithm,
                   MetaTensor* out,
                   MetaConfig config) {
  std::vector<int> paddings_ = paddings;
  std::vector<int> kernel_size_ = kernel_size;

  auto x_dims = x.dims();
  PADDLE_ENFORCE_EQ(
      x_dims.size() == 4 || x_dims.size() == 5,
      true,
      errors::InvalidArgument(
          "the input of Op(pool) should be 4-D or 5-D Tensor. But "
          "received: %u-D Tensor and it's shape is [%s].",
          x_dims.size(),
          x_dims));

  PADDLE_ENFORCE_EQ(x_dims.size() - kernel_size_.size(),
                    2U,
                    errors::InvalidArgument(
                        "the dimension of input minus the size of "
                        "Attr(kernel_size_) must be equal to 2 in Op(pool). "
                        "But received: the dimension of input minus the size "
                        "of Attr(kernel_size_) is %d, the "
                        "input's dimension is %d, the shape of input "
                        "is [%s], the Attr(kernel_size_)'s size is %d, the "
                        "Attr(kernel_size_) is [%s].",
                        x_dims.size() - kernel_size_.size(),
                        x_dims.size(),
                        x_dims,
                        kernel_size_.size(),
                        common::make_ddim(kernel_size_)));

  PADDLE_ENFORCE_EQ(
      kernel_size_.size(),
      strides.size(),
      errors::InvalidArgument(
          "the size of Attr(kernel_size_) and Attr(strides) in "
          "Op(pool) must be equal. "
          "But received: Attr(kernel_size_)'s size is %d, Attr(strides)'s "
          "size is %d, Attr(kernel_size_) is [%s], Attr(strides)is [%s].",
          kernel_size_.size(),
          strides.size(),
          common::make_ddim(kernel_size_),
          common::make_ddim(strides)));

  // MKL-DNN Kernels are using NCHW order of dims description
  // so we ignore data_format consideration for MKL-DNN kernel
  const bool channel_last = (config.is_run_mkldnn_kernel == false) &&
                            (data_format == "NHWC" || data_format == "NDHWC");

  // update paddings if "SAME" or global_pooling
  DDim data_dims;
  if (channel_last) {
    data_dims = slice_ddim(x_dims, 1, x_dims.size() - 1);
  } else {
    data_dims = slice_ddim(x_dims, 2, x_dims.size());
  }
  funcs::UpdatePadding(&paddings_,
                       global_pooling,
                       adaptive,
                       padding_algorithm,
                       data_dims,
                       strides,
                       kernel_size_);

  if (global_pooling) {
    funcs::UpdateKernelSize(&kernel_size_, data_dims);
  }

  std::vector<int64_t> output_shape;
  if (adaptive) {
    output_shape.insert(
        output_shape.end(), kernel_size_.begin(), kernel_size_.end());
  } else {
    for (int i = 0; i < data_dims.size(); ++i) {
      if ((!config.is_runtime) && (data_dims[i] < 0)) {
        output_shape.push_back(data_dims[i]);
      } else {
        output_shape.push_back(
            funcs::PoolOutputSize(static_cast<int>(data_dims[i]),
                                  kernel_size_[i],
                                  paddings_[2 * i],
                                  paddings_[2 * i + 1],
                                  strides[i],
                                  ceil_mode));
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

  out->set_dims(common::make_ddim(output_shape));
  out->share_lod(x);
  out->set_dtype(x.dtype());
}

void PushDenseInferMeta(const std::vector<const MetaTensor*>& ids,
                        int table_id,
                        float scale_data_norm,
                        const std::vector<std::string>& input_names) {
  auto ids_num = ids.size();
  PADDLE_ENFORCE_GE(ids_num,
                    1UL,
                    common::errors::InvalidArgument(
                        "Input(Ids) of PushDenseOp can not be null."));
}

void RealAndImagInferMeta(const MetaTensor& x, MetaTensor* out) {
  out->set_dims(x.dims());
  out->set_dtype(dtype::ToReal(x.dtype()));
  out->set_layout(x.layout());
}

void QrInferMeta(const MetaTensor& x,
                 const std::string& mode,
                 MetaTensor* q,
                 MetaTensor* r) {
  auto x_dims = x.dims();
  int x_rank = x_dims.size();
  PADDLE_ENFORCE_GE(
      x_dims.size(),
      2,
      common::errors::InvalidArgument("the rank of input must greater than 2"));
  bool compute_q = false;
  bool reduced_mode = false;
  int m = static_cast<int>(x_dims[x_rank - 2]);
  int n = static_cast<int>(x_dims[x_rank - 1]);
  int min_mn = std::min(m, n);
  std::tie(compute_q, reduced_mode) = phi::funcs::ParseQrMode(mode);

  if (compute_q) {
    int k = reduced_mode ? min_mn : m;
    auto q_dims_vec = common::vectorize(x_dims);
    q_dims_vec[q_dims_vec.size() - 1] = k;
    q->set_dims(common::make_ddim(q_dims_vec));
  } else {
    q->set_dims(common::make_ddim({0}));
  }

  int k = reduced_mode ? min_mn : m;
  auto r_dims_vec = common::vectorize(x_dims);
  r_dims_vec[r_dims_vec.size() - 2] = k;
  r_dims_vec[r_dims_vec.size() - 1] = n;
  r->set_dims(common::make_ddim(r_dims_vec));

  q->share_lod(x);
  r->share_lod(x);
  q->set_dtype(x.dtype());
  r->set_dtype(x.dtype());
}

DDim ReduceInferDim(const MetaTensor& x,
                    const std::vector<int64_t>& axis,
                    bool keep_dim,
                    bool reduce_all) {
  int x_rank = x.dims().size();

  std::vector<int64_t> formatted_axis = axis;
  for (size_t i = 0; i < axis.size(); ++i) {
    if (x_rank == 0) {
      PADDLE_ENFORCE_EQ(
          axis[i] == 0 || axis[i] == -1,
          true,
          common::errors::InvalidArgument(
              "When input 0D Tensor, the axis can only be -1, 0, None or []"));
    } else {
      PADDLE_ENFORCE_LT(
          axis[i],
          x_rank,
          errors::InvalidArgument(
              "The reduce dim index %d should be in the "
              "range [ -dimension(X), dimension(X) ) "
              "which dimension = %d. But received dim index = %d.",
              i,
              x_rank,
              axis[i]));
      PADDLE_ENFORCE_GE(
          axis[i],
          -x_rank,
          errors::InvalidArgument(
              "The reduce dim index %d should be in the "
              "range [ -dimension(X), dimension(X) )  "
              "which dimension = %d. But received dim index = %d.",
              i,
              x_rank,
              axis[i]));
    }

    if (axis[i] < 0) {
      formatted_axis[i] = axis[i] + x_rank;
    }
  }

  bool full_dim = true;
  std::set<int64_t> dims_set(formatted_axis.begin(), formatted_axis.end());
  for (int64_t i = 0; i < x_rank; ++i) {
    if (dims_set.find(i) == dims_set.end()) {
      full_dim = false;
      break;
    }
  }
  bool empty_dim = axis.size() == 0;
  reduce_all = reduce_all || full_dim || empty_dim;

  std::vector<int64_t> out_dim_vector;
  for (int i = 0; i < x_rank; ++i) {
    if (reduce_all || dims_set.find(i) != dims_set.end()) {
      if (keep_dim) {
        out_dim_vector.push_back(1);
      } else {
        continue;
      }
    } else {
      out_dim_vector.push_back(x.dims().at(i));
    }
  }

  DDim out_dim = common::make_ddim(out_dim_vector);
  return out_dim;
}

void ReduceInferMetaBase(const MetaTensor& x,
                         const std::vector<int64_t>& axis,
                         bool keep_dim,
                         bool reduce_all,
                         MetaTensor* out) {
  DDim out_dim = ReduceInferDim(x, axis, keep_dim, reduce_all);
  out->set_dims(out_dim);
  out->set_dtype(x.dtype());
  out->set_layout(x.layout());
}

void ReduceSumInferMeta(const MetaTensor& x,
                        const std::vector<int64_t>& axis,
                        bool keep_dim,
                        DataType dtype,
                        MetaTensor* out) {
  bool reduce_all = false;
  if (axis.empty()) {
    reduce_all = true;
  }
  SumRawInferMeta(x, axis, keep_dim, reduce_all, dtype, out);
}

void ReduceInferMeta(const MetaTensor& x,
                     const std::vector<int64_t>& axis,
                     bool keep_dim,
                     MetaTensor* out) {
  bool reduce_all = false;
  if (axis.empty()) {
    reduce_all = true;
  }

  ReduceInferMetaBase(x, axis, keep_dim, reduce_all, out);
}

DDim ReduceInferDimForIntArrayAxis(const MetaTensor& x,
                                   const IntArray& axis,
                                   bool keep_dim,
                                   bool reduce_all) {
  std::vector<int64_t> vec_axis = axis.GetData();
  std::vector<int64_t> vec_dim;
  if (reduce_all) {
    if (keep_dim) {
      vec_dim = std::vector<int64_t>(x.dims().size(), 1);
    } else {
      vec_dim = {};
    }
  } else {
    if (keep_dim) {
      vec_dim = std::vector<int64_t>(x.dims().size(), -1);
    } else {
      auto x_rank = static_cast<size_t>(x.dims().size());
      if (vec_axis.size() > x_rank) {
        vec_dim = {-1};
      } else {
        vec_dim = std::vector<int64_t>(x.dims().size() - vec_axis.size(), -1);
      }
    }
  }
  return common::make_ddim(vec_dim);
}

void ReduceIntArrayAxisInferMetaBase(const MetaTensor& x,
                                     const IntArray& axis,
                                     bool keep_dim,
                                     bool reduce_all,
                                     MetaTensor* out,
                                     MetaConfig config) {
  DDim out_dim;
  if (config.is_runtime || !axis.FromTensor()) {
    out_dim = ReduceInferDim(x, axis.GetData(), keep_dim, reduce_all);
  } else {
    out_dim = ReduceInferDimForIntArrayAxis(x, axis, keep_dim, reduce_all);
  }
  out->set_dims(out_dim);
  out->set_dtype(x.dtype());
  out->set_layout(x.layout());
}

void ReduceIntArrayAxisInferMeta(const MetaTensor& x,
                                 const IntArray& axis,
                                 bool keep_dim,
                                 MetaTensor* out,
                                 MetaConfig config) {
  bool reduce_all = false;
  if (axis.size() == 0) {
    reduce_all = true;
  }
  ReduceIntArrayAxisInferMetaBase(x, axis, keep_dim, reduce_all, out, config);
}

void ReduceScatterInferMeta(const MetaTensor& x, int nranks, MetaTensor* out) {
  auto dim = x.dims();
  if (dim[0] > 0 || dim[0] < -1) {
    PADDLE_ENFORCE_EQ(
        dim[0] % nranks,
        0,
        errors::InvalidArgument(
            "dim[0] (%d) is not divisible by nranks(%d)", dim[0], nranks));
    dim[0] /= nranks;
  }
  out->set_dims(dim);
  out->set_dtype(x.dtype());
}

void RepeatInterleaveInferMeta(const MetaTensor& x,
                               int repeats,
                               int dim,
                               MetaTensor* out) {
  const auto& input_dim = x.dims();
  auto output_dim = common::vectorize(input_dim);
  auto n_dim = dim;

  if (n_dim < 0) n_dim += input_dim.size();

  PADDLE_ENFORCE_LT(
      dim,
      input_dim.size(),
      common::errors::OutOfRange(
          "Attr(dim) is out of range, It's expected "
          "to be in range of [%d, %d]. But received Attr(dim) = %d.",
          -input_dim.size(),
          input_dim.size() - 1,
          dim));
  PADDLE_ENFORCE_GE(
      dim,
      (0 - input_dim.size()),
      common::errors::OutOfRange(
          "Attr(dim) is out of range, It's expected "
          "to be in range of [%d, %d]. But received Attr(dim) = %d.",
          -input_dim.size(),
          input_dim.size() - 1,
          dim));

  PADDLE_ENFORCE_GT(
      repeats,
      0,
      common::errors::InvalidArgument("repeats should be larger than zero"));

  PADDLE_ENFORCE_NOT_NULL(
      out,
      common::errors::InvalidArgument(
          "repeat_interleave's output tensor can't be nullptr"));

  if (input_dim[n_dim] != -1) output_dim[n_dim] = input_dim[n_dim] * repeats;
  out->set_dims(common::make_ddim(output_dim));
  out->share_lod(x);
  out->set_dtype(x.dtype());
}

void ReshapeInferMeta(const MetaTensor& x,
                      const IntArray& shape,
                      MetaTensor* out,
                      MetaConfig config) {
  auto shape_data = shape.GetData();
  PADDLE_ENFORCE_NOT_NULL(out,
                          common::errors::InvalidArgument(
                              "Output(Out) of ReshapeOp should not be null."));
  if (!config.is_runtime && shape.FromTensor()) {
    const int64_t copy_dim_flag = 0;
    const auto& in_dims = x.dims();
    for (size_t i = 0; i < shape_data.size(); ++i) {
      if (shape_data[i] == copy_dim_flag) {
        PADDLE_ENFORCE_LT(
            static_cast<int>(i),
            in_dims.size(),
            common::errors::InvalidArgument(
                "The index of 0 in `shape` must be less than "
                "the input tensor X's dimensions. But received shape[%d] "
                "= 0, X's dimensions = %d, X's shape = [%s].",
                i,
                in_dims.size(),
                in_dims));
        shape_data[i] = static_cast<int>(in_dims[static_cast<int>(i)]);
      }
    }
    out->set_dims(common::make_ddim(shape_data));
    out->share_lod(x);
    out->set_dtype(x.dtype());
    return;
  }
  InferMetaFromVecValue(x, shape_data, out);
}

void ViewShapeInferMeta(const MetaTensor& input,
                        const std::vector<int64_t>& shape,
                        MetaTensor* out) {
  InferMetaFromVecValue(input, shape, out);
}

void ReshapeWithXShapeInferMeta(const MetaTensor& x,
                                const IntArray& shape,
                                MetaTensor* out,
                                MetaTensor* xshape,
                                MetaConfig config) {
  PADDLE_ENFORCE_NOT_NULL(
      xshape,
      common::errors::InvalidArgument(
          "Output(XShape) of ReshapeOp should not be null."));
  const auto& x_dims = x.dims();
  std::vector<int64_t> xshape_dims(x_dims.size() + 1);
  xshape_dims[0] = 0;
  for (int i = 0; i < x_dims.size(); ++i) {
    xshape_dims[i + 1] = x_dims[i];
  }
  xshape->set_dims(common::make_ddim(xshape_dims));
  xshape->share_lod(x);
  xshape->set_strides(x.strides());
  xshape->set_dtype(DataType::INT64);
  ReshapeInferMeta(x, shape, out, config);
}

void ReverseInferMeta(const MetaTensor& x,
                      const IntArray& axis,
                      MetaTensor* out,
                      MetaConfig config) {
  // NOTE(Aurelius84): In Reverse Op, output TensorMeta is always same
  // as input, so we only verify axis when it is not from Tensor or in
  // runtime.
  if (!config.is_runtime && axis.FromTensor()) {
    out->share_meta(x);
    return;
  }
  auto& axis_data = axis.GetData();
  PADDLE_ENFORCE_NE(
      axis_data.empty(),
      true,
      common::errors::InvalidArgument("'axis' can not be empty."));
  const auto& x_dims = x.dims();
  for (int64_t a : axis_data) {
    PADDLE_ENFORCE_LT(a,
                      x_dims.size(),
                      common::errors::OutOfRange(
                          "The axis must be less than input tensor's rank. "
                          "but got %d >= %d",
                          a,
                          x_dims.size()));
    PADDLE_ENFORCE_GE(
        a,
        -x_dims.size(),
        common::errors::OutOfRange(
            "The axis must be greater than the negative number of "
            "input tensor's rank, but got %d < %d",
            a,
            -x_dims.size()));
  }
  out->share_meta(x);
}

void ReverseArrayInferMeta(const std::vector<const phi::MetaTensor*>& x,
                           const IntArray& axis,
                           std::vector<phi::MetaTensor*> out,
                           MetaConfig config) {
  if (!config.is_runtime && axis.FromTensor()) {
    return;
  }
  auto& axis_data = axis.GetData();
  PADDLE_ENFORCE_EQ(
      axis_data.size(),
      1,
      common::errors::InvalidArgument(
          "The size of axis must be 1 when the Input(X) is phi::TensorArray, "
          "but received %d.",
          axis_data.size()));
  PADDLE_ENFORCE_EQ(
      axis_data[0],
      0,
      common::errors::InvalidArgument("The value of axis should be 1 when "
                                      "the Input(X) is phi::TensorArray, "
                                      "but received %d.",
                                      axis_data[0]));
}

void RollInferMeta(const MetaTensor& x,
                   const IntArray& shifts,
                   const std::vector<int64_t>& axis,
                   MetaTensor* out) {
  auto shifts_data = shifts.GetData();

  if (!axis.empty()) {
    PADDLE_ENFORCE_EQ(
        axis.size(),
        shifts_data.size(),
        common::errors::InvalidArgument("When dims.size() != 0, dims.size() "
                                        "should be equal to "
                                        "shifts.size(). But received "
                                        "dims.size() = %d, shifts.size() = %d",
                                        axis.size(),
                                        shifts_data.size()));
  } else {
    PADDLE_ENFORCE_EQ(
        shifts_data.size(),
        1,
        common::errors::InvalidArgument("When dims.size() == 0, shifts.size() "
                                        "should be equal to 1, But received "
                                        "shifts.size() = %d",
                                        shifts_data.size()));
  }

  out->set_dims(x.dims());
  out->share_lod(x);
  out->set_dtype(x.dtype());
}

void RReluInferMeta(const MetaTensor& x,
                    float lower,
                    float upper,
                    bool is_test,
                    MetaTensor* out,
                    MetaTensor* noise) {
  auto x_dims = x.dims();
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
          "The upper value should be greater than or equal to lower value "
          "But received upper value = %f, lower value = %f.",
          upper,
          lower));

  out->set_dims(x_dims);
  out->set_dtype(x.dtype());
  out->set_layout(x.layout());
  out->share_lod(x);

  if (noise != nullptr) {
    noise->set_dims(x_dims);
    noise->set_dtype(x.dtype());
    noise->set_layout(x.layout());
  }
}

void RReluGradInferMeta(const MetaTensor& out_grad,
                        const MetaTensor& noise,
                        MetaTensor* x_grad) {
  auto do_dims = out_grad.dims();
  x_grad->set_dims(do_dims);
  x_grad->set_dtype(out_grad.dtype());
  x_grad->share_lod(out_grad);
}

void SetValueInferMeta(const MetaTensor& x, MetaTensor* out) {
  auto in_dims = x.dims();
  PADDLE_ENFORCE_LT(
      in_dims.size(),
      7,
      common::errors::InvalidArgument(
          "The rank of input should be less than 7, but received %d.",
          in_dims.size()));
  out->set_dims(in_dims);
  out->set_dtype(x.dtype());
}

void ShapeInferMeta(const MetaTensor& input, MetaTensor* out) {
  auto in_dim = input.dims();
  out->set_dims(common::make_ddim({in_dim.size()}));
  out->set_dtype(DataType::INT32);
}

void Shape64InferMeta(const MetaTensor& input,
                      MetaTensor* out,
                      MetaConfig config) {
  auto in_dim = input.dims();
  out->set_dims(common::make_ddim({in_dim.size()}));
  if (config.is_run_mkldnn_kernel) {
    out->set_dtype(DataType::INT32);
  } else {
    out->set_dtype(DataType::INT64);
  }
}

void ShareDataInferMeta(const MetaTensor& x, MetaTensor* out) {
  out->set_dims(x.dims());
  out->set_dtype(x.dtype());
}

void ShardIndexInferMeta(const MetaTensor& in,
                         int index_num,
                         int nshards,
                         int shard_id,
                         int ignore_value,
                         MetaTensor* out,
                         MetaConfig config) {
  auto x_dims = in.dims();
  PADDLE_ENFORCE_GE(
      x_dims.size(),
      2,
      common::errors::InvalidArgument("Rank of Input(X) should be at least 2, "
                                      "but the value given is %d.",
                                      x_dims.size()));
  if (config.is_runtime || x_dims[x_dims.size() - 1] > 0) {
    PADDLE_ENFORCE_EQ(x_dims[x_dims.size() - 1],
                      1U,
                      common::errors::InvalidArgument(
                          "The last dimension of Input(X) should be 1, "
                          "but the value given is %d.",
                          x_dims[x_dims.size() - 1]));
  }

  out->set_dims(x_dims);
  out->share_lod(in);
  out->set_dtype(in.dtype());
}

void NumelInferMeta(const MetaTensor& input, MetaTensor* out) {
  out->set_dtype(DataType::INT64);
  out->set_dims(common::make_ddim({}));
}

void ShuffleChannelInferMeta(const MetaTensor& x, int group, MetaTensor* out) {
  auto input_dims = x.dims();
  PADDLE_ENFORCE_EQ(
      input_dims.size(),
      4,
      common::errors::InvalidArgument("The layout of input is NCHW."));

  out->set_dtype(x.dtype());
  out->set_dims(input_dims);
}

// This logic is copied from
// `paddle/fluid/operators/slice_op.cc/SliceOp::InferShape`
void SliceArrayInferMeta(const MetaTensor& input,
                         const IntArray& starts,
                         const IntArray& ends,
                         MetaTensor* out,
                         MetaConfig config) {
  SliceArrayDenseInferMeta(input, starts, out, config);
}

void SliceArrayDenseInferMeta(const MetaTensor& input,
                              const IntArray& starts,
                              MetaTensor* out,
                              MetaConfig config) {
  if (config.is_runtime) {
    return;
  }
  out->set_dtype(input.dtype());
  out->set_dims(input.dims());
}

void SliceRawInferMeta(const MetaTensor& input,
                       const std::vector<int64_t>& axes,
                       const IntArray& starts_arr,
                       const IntArray& ends_arr,
                       const std::vector<int64_t>& infer_flags_t,
                       const std::vector<int64_t>& decrease_axis,
                       MetaTensor* out,
                       MetaConfig config) {
  const auto& in_dims = input.dims();

  PADDLE_ENFORCE_LT(in_dims.size(),
                    7,
                    common::errors::InvalidArgument(
                        "The rank of input should be less than 7."));

  std::vector<int64_t> infer_flags = infer_flags_t;
  if (infer_flags.empty()) {
    // Initialize infer_flags with 1.
    // To be compatible with other op tests in which infer_flags is not set.
    infer_flags = std::vector<int64_t>(axes.size(), 1);
  }
  auto new_axes = axes;
  for (auto& axis : new_axes) {
    if (axis < 0) {
      axis = std::max(int64_t(0), axis + int64_t(in_dims.size()));
    }
  }

  PADDLE_ENFORCE_EQ(
      axes.size(),
      starts_arr.size(),
      common::errors::InvalidArgument(
          "The length of axes (%d) and length of starts (%d) should be same.",
          axes.size(),
          starts_arr.size()));
  PADDLE_ENFORCE_EQ(
      axes.size(),
      ends_arr.size(),
      common::errors::InvalidArgument(
          "The length of axes (%d) and length of ends (%d) should be same.",
          axes.size(),
          ends_arr.size()));

  // 2.1 Check attrs.
  std::vector<int64_t> starts = starts_arr.GetData();
  std::vector<int64_t> ends = ends_arr.GetData();

  phi::funcs::CheckAndUpdateSliceAttrs<int64_t>(
      in_dims, new_axes, &starts, &ends, nullptr, &infer_flags);

  auto slice_dims = phi::funcs::GetSliceDims<int64_t>(
      in_dims, new_axes, starts, ends, nullptr, &infer_flags);

  DDim out_dims(in_dims);
  out_dims = config.is_runtime ? phi::funcs::GetDecreasedDims<int64_t>(
                                     slice_dims, decrease_axis, &infer_flags)
                               : phi::funcs::GetDecreasedDims<int64_t>(
                                     slice_dims, decrease_axis, nullptr);

  out->set_dims(out_dims);
  if (!new_axes.empty() && new_axes[0] != 0) {
    out->share_lod(input);
  }
  out->set_dtype(input.dtype());
}

void ViewSliceInferMeta(const MetaTensor& input,
                        int64_t begin_idx,
                        int64_t end_idx,
                        MetaTensor* out) {
  const auto& in_dims = input.dims();
  PADDLE_ENFORCE_GE(
      begin_idx,
      0,
      common::errors::OutOfRange("The start row index must be greater than 0."
                                 "But received the start index is d%.",
                                 begin_idx));
  PADDLE_ENFORCE_LE(
      end_idx,
      in_dims[0],
      common::errors::OutOfRange("The end row index is out of bound."));

  PADDLE_ENFORCE_LT(
      begin_idx,
      end_idx,
      common::errors::InvalidArgument(
          "The start row index must be less than the end row index."
          "But received the start index = %d, the end index = %d.",
          begin_idx,
          end_idx));

  DDim out_dims(in_dims);
  out_dims[0] = end_idx - begin_idx;
  out->set_dims(out_dims);
  out->set_dtype(input.dtype());
}

void SoftmaxInferMeta(const MetaTensor& x, int axis, MetaTensor* out) {
  auto dim_x = x.dims();
  auto rank_x = dim_x.size();
  if (rank_x > 0) {
    PADDLE_ENFORCE_GE(axis,
                      -rank_x,
                      common::errors::InvalidArgument(
                          "Attr(axis) value should be in range [-R, R-1], "
                          "R is the rank of Input(X)."));
    PADDLE_ENFORCE_LT(axis,
                      rank_x,
                      common::errors::InvalidArgument(
                          "Attr(axis) value should be in range [-R, R-1], "
                          "R is the rank of Input(X)."));
  } else {
    PADDLE_ENFORCE_GE(axis,
                      -1,
                      common::errors::InvalidArgument(
                          "Attr(axis) value should be in range [-1, "
                          "0] when input is 0D Tensor "));
    PADDLE_ENFORCE_LE(axis,
                      0,
                      common::errors::InvalidArgument(
                          "Attr(axis) value should be in range [-1, "
                          "0] when input is 0D Tensor "));
  }

  out->set_dims(x.dims());
  out->set_dtype(x.dtype());
  out->share_lod(x);
}

int GetSplitAxisValue(const MetaTensor& x,
                      const Scalar& axis,
                      MetaConfig config) {
  // Tensor has no value in static graph compile time
  if (axis.FromTensor() && !config.is_runtime) {
    return -1;
  } else {
    if (axis.dtype() == DataType::FLOAT32 ||
        axis.dtype() == DataType::FLOAT64) {
      PADDLE_THROW(
          common::errors::InvalidArgument("%s(): argument (position 3) must be "
                                          "int, but got %s",
                                          "split",
                                          "float"));  // NOLINT
    }
    int axis_value = axis.to<int>();
    int rank = x.dims().size();
    PADDLE_ENFORCE_EQ(
        axis_value >= -rank && axis_value < rank,
        true,
        common::errors::InvalidArgument(
            "The axis is expected to be in range of [%d, %d), but got %d",
            -rank,
            rank,
            axis_value));
    if (axis_value < 0) {
      axis_value = axis_value + rank;
    }
    return axis_value;
  }
}

void FillSplitOutDims(const MetaTensor& x,
                      const int axis_value,
                      const std::vector<int64_t>& sections_vec,
                      std::vector<MetaTensor*>* out) {
  std::vector<phi::DDim> out_dims(sections_vec.size(), x.dims());
  if (x.dims().at(axis_value) > 0) {
    for (size_t i = 0; i < sections_vec.size(); ++i) {
      out_dims[i][axis_value] = sections_vec[i];
    }
  } else {
    for (size_t i = 0; i < sections_vec.size(); ++i) {
      out_dims[i][axis_value] = -1;
    }
  }
  for (size_t i = 0; i < sections_vec.size(); ++i) {
    if (axis_value != 0) {
      // Only pass LoD when not splitting along the first dim.
      (*out)[i]->set_dtype(x.dtype());
      (*out)[i]->set_dims(out_dims[i]);
      (*out)[i]->set_layout(x.layout());
      (*out)[i]->share_lod(x);
    } else {
      (*out)[i]->set_dtype(x.dtype());
      (*out)[i]->set_dims(out_dims[i]);
      (*out)[i]->set_layout(x.layout());
    }
  }
}

void QuantizeXPUInferMeta(const MetaTensor& x,
                          DataType out_dtype,
                          float scale,
                          MetaTensor* y) {
  auto x_dims = x.dims();
  y->set_dims(x_dims);
  y->set_dtype(out_dtype);
}

void SequenceSoftmaxInferMeta(const MetaTensor& x, MetaTensor* out) {
  out->set_dims(x.dims());
  out->share_lod(x);
}

void SplitInferMeta(const MetaTensor& x,
                    const IntArray& sections,
                    const Scalar& axis,
                    std::vector<MetaTensor*> out,
                    MetaConfig config) {
  // get axis value
  int axis_value = GetSplitAxisValue(x, axis, config);

  auto sections_data = sections.GetData();
  // fill out dims with -1
  if ((sections.FromTensor() && !config.is_runtime) || axis_value == -1 ||
      (axis_value >= 0 && x.dims().at(axis_value) <= 0)) {
    std::vector<phi::DDim> out_dims;
    if ((sections.FromTensor() && !config.is_runtime) ||
        axis_value == -1) {  // NOLINT
      out_dims = std::vector<phi::DDim>(
          sections_data.size(),
          common::make_ddim(std::vector<int>(x.dims().size(), -1)));
    } else {
      out_dims = std::vector<phi::DDim>(sections_data.size(), x.dims());
    }
    for (size_t i = 0; i < sections_data.size(); ++i) {
      if (axis_value != 0) {
        // Only pass LoD when not splitting along the first dim.
        out[i]->set_dtype(x.dtype());
        out[i]->set_dims(out_dims[i]);
        out[i]->set_layout(x.layout());
        out[i]->share_lod(x);
      } else {
        out[i]->set_dtype(x.dtype());
        out[i]->set_dims(out_dims[i]);
        out[i]->set_layout(x.layout());
      }
    }
  } else {
    auto input_axis_dim = x.dims().at(axis_value);
    std::vector<int64_t> sections_vec;
    const int unknow_dim_val = -1;
    int unknow_dim_idx = -1;
    int num_of_unknow = 0;
    int sum_of_section = 0;

    for (int i = 0; i < static_cast<int>(sections_data.size()); ++i) {
      sections_vec.push_back(sections_data[i]);

      if (sections_data[i] == unknow_dim_val) {
        num_of_unknow++;
        unknow_dim_idx = i;
      } else {
        sum_of_section += static_cast<int>(sections_data[i]);
      }
    }

    PADDLE_ENFORCE_LE(num_of_unknow,
                      1,
                      common::errors::InvalidArgument(
                          "Only one dimension value of Attr(num_or_sections) "
                          "in SplitOp can be -1. "
                          "But received Attr(num_or_sections) = [%s].",
                          common::make_ddim(sections_data)));

    if (unknow_dim_idx != -1) {
      // for example, input shape = [4 ,5], axis = 1, sections = [2, 3, -1].
      // input_axis_dim = 5, sum_of_sections = 5.
      // the following check will fail.
      PADDLE_ENFORCE_LT(
          sum_of_section,
          input_axis_dim,
          common::errors::InvalidArgument(
              "Sum of Attr(num_or_sections) other than unknown section "
              "must be less than the input's "
              "size "
              "along the split dimension. But received Attr(num_or_sections) "
              "= [%s], input(X)'s shape = [%s], Attr(dim) = %d.",
              common::make_ddim(sections_data),
              x.dims(),
              axis_value));

      sections_vec[unknow_dim_idx] = input_axis_dim - sum_of_section;
    } else {
      PADDLE_ENFORCE_EQ(
          sum_of_section,
          input_axis_dim,
          common::errors::InvalidArgument(
              "Sum of Attr(num_or_sections) must be equal to the input's "
              "size "
              "along the split dimension. But received Attr(num_or_sections)"
              " = [%s], input(X)'s shape = [%s], Attr(dim) = %d.",
              common::make_ddim(sections_data),
              x.dims(),
              axis_value));
    }
    // fill out dims
    FillSplitOutDims(x, axis_value, sections_vec, &out);
  }
}

void SplitWithNumInferMeta(const MetaTensor& x,
                           int num,
                           const Scalar& axis,
                           std::vector<MetaTensor*> out,
                           MetaConfig config) {
  int axis_value = GetSplitAxisValue(x, axis, config);
  // fill out dims with -1
  if (axis_value == -1 || (axis_value >= 0 && x.dims().at(axis_value) <= 0)) {
    std::vector<phi::DDim> out_dims;
    if (axis_value == -1) {  // NOLINT
      out_dims = std::vector<phi::DDim>(
          num, common::make_ddim(std::vector<int>(x.dims().size(), -1)));
    } else {
      out_dims = std::vector<phi::DDim>(num, x.dims());
    }
    for (int i = 0; i < num; ++i) {
      if (axis_value != 0) {
        // Only pass LoD when not splitting along the first dim.
        out[i]->set_dtype(x.dtype());
        out[i]->set_dims(out_dims[i]);
        out[i]->set_layout(x.layout());
        out[i]->share_lod(x);
      } else {
        out[i]->set_dtype(x.dtype());
        out[i]->set_dims(out_dims[i]);
        out[i]->set_layout(x.layout());
      }
    }
  } else {
    auto input_axis_dim = x.dims().at(axis_value);
    // step1: get formatted sections
    std::vector<int64_t> sections_vec;
    PADDLE_ENFORCE_NE(num,
                      0,
                      common::errors::InvalidArgument(
                          "Attr(num_or_sections) should not be 0."));
    PADDLE_ENFORCE_EQ(input_axis_dim % num,
                      0,
                      common::errors::InvalidArgument(
                          "The input's size along the split dimension "
                          "must be evenly divisible by Attr(num_or_sections). "
                          "But received Attr(num_or_sections) "
                          "= %d, input(X)'s shape = [%s], Attr(dim) = %d.",
                          num,
                          x.dims(),
                          axis_value));

    for (int i = 0; i < num; ++i) {
      sections_vec.push_back(input_axis_dim / num);
    }
    // setp2: fill out dims
    FillSplitOutDims(x, axis_value, sections_vec, &out);
  }
}

void SequenceMaskScalarInferMeta(const MetaTensor& x,
                                 const Scalar& max_len,
                                 DataType out_dtype,
                                 MetaTensor* y) {
  auto dim = phi::vectorize<int>(x.dims());
  int maxlen = max_len.to<int>();
  dim.push_back(maxlen > 0 ? maxlen : -1);
  y->set_dims(phi::make_ddim(dim));
  y->set_dtype(out_dtype);
}

void SequencePoolInferMeta(const MetaTensor& x,
                           bool is_test,
                           const std::string& pooltype,
                           float pad_value,
                           MetaTensor* out,
                           MetaTensor* max_index,
                           MetaConfig config) {
  out->set_dims(x.dims());
  if (pooltype == "MAX") {
    max_index->set_dims(x.dims());
  }
}

void SquaredL2NormInferMeta(const MetaTensor& x, MetaTensor* out) {
  out->set_dims({1});
  out->set_dtype(x.dtype());
}

void L1NormInferMeta(const MetaTensor& x, MetaTensor* out) {
  out->set_dims(common::make_ddim({}));
  out->set_dtype(x.dtype());
}

void SqueezeInferMeta(const MetaTensor& x,
                      const IntArray& axes,
                      MetaTensor* out,
                      MetaConfig config) {
  const auto& x_dims = x.dims();
  if (!config.is_runtime && axes.FromTensor()) {
    // compile time infershape, set all elements to -1.
    int output_size = static_cast<int>(x.dims().size() - axes.GetData().size());
    if (x.dims().size() == 0 && output_size == -1) {
      output_size = 0;
    }
    std::vector<int64_t> vec_out_dims(output_size, -1);

    out->set_dims(common::make_ddim(vec_out_dims));
  } else {
    std::vector<int32_t> tmp;
    tmp.reserve(axes.GetData().size());
    std::for_each(axes.GetData().begin(),
                  axes.GetData().end(),
                  [&tmp](const int64_t& t) { tmp.push_back(t); });  // NOLINT
    auto out_dims = funcs::GetOutputSqueezeShape(tmp, x_dims, false);
    out->set_dims(out_dims);
    if (x_dims[0] == out_dims[0]) {
      // Only pass LoD when the first dimension of output and Input(X)
      // are the same.
      out->share_lod(x);
    }
  }
  out->set_dtype(x.dtype());
}

void SqueezeWithXShapeInferMeta(const MetaTensor& x,
                                const IntArray& axes,
                                MetaTensor* out,
                                MetaTensor* xshape,
                                MetaConfig config) {
  const auto& x_dims = x.dims();
  SqueezeInferMeta(x, axes, out, config);
  std::vector<int64_t> xshape_dims(x_dims.size() + 1);
  xshape_dims[0] = 0;
  for (int i = 0; i < x_dims.size(); ++i) {
    xshape_dims[i + 1] = x_dims[i];
  }
  if (xshape) {
    xshape->set_dims(common::make_ddim(xshape_dims));
    xshape->share_lod(x);
    xshape->set_dtype(x.dtype());
  }
}

void StridedSliceRawInferMeta(const MetaTensor& x,
                              const std::vector<int>& axes,
                              const IntArray& starts,
                              const IntArray& ends,
                              const IntArray& strides,
                              const std::vector<int>& infer_flags,
                              const std::vector<int>& decrease_axis,
                              MetaTensor* out,
                              MetaConfig config) {
  auto in_dims = x.dims();
  PADDLE_ENFORCE_LT(
      in_dims.size(),
      7,
      errors::InvalidArgument(
          "The dimension of StridedSlice operator's input should be less "
          "than 7, but received dimension is %d.",
          in_dims.size()));

  auto starts_ = starts.GetData();
  auto ends_ = ends.GetData();
  auto strides_ = strides.GetData();

  auto starts_size = starts_.size();
  auto ends_size = ends_.size();
  auto strides_size = strides_.size();

  for (size_t i = 0; i < axes.size(); ++i) {
    PADDLE_ENFORCE_GE(
        axes[i],
        0,
        errors::InvalidArgument("The axis should be greater than or equal to 0."
                                "But received %d of axes[%d]",
                                axes[i],
                                i));
    PADDLE_ENFORCE_LT(
        axes[i],
        in_dims.size(),
        errors::InvalidArgument(
            "The axes should be less than or equal to input tensor's rank."
            "But received %d of axes[%d], input tensor shape [%d]",
            axes[i],
            i,
            in_dims.size()));
  }

  auto tensor_input = false;
  auto HasInput = [](const IntArray& arr) { return arr.FromTensor(); };
  if (HasInput(starts) || HasInput(ends) || HasInput(strides)) {
    tensor_input = true;
  }
  if (!HasInput(ends)) {
    PADDLE_ENFORCE_EQ(
        ends_size,
        axes.size(),
        errors::InvalidArgument(
            "The size of ends attribute in StridedSlice operator is not "
            "equal to the size of axes attribute. The ends attribute's size "
            "is %d, axes attribute's size is %d.",
            ends_size,
            axes.size()));
  }
  if (!HasInput(starts)) {
    PADDLE_ENFORCE_EQ(
        starts_size,
        axes.size(),
        errors::InvalidArgument(
            "The size of starts attribute in StridedSlice operator is not "
            "equal to the size of axes attribute. The starts attribute's "
            "size is %d, axes attribute's size is %d.",
            starts_size,
            axes.size()));
  }
  if (!HasInput(strides)) {
    PADDLE_ENFORCE_EQ(
        strides_size,
        axes.size(),
        errors::InvalidArgument(
            "The size of strides attribute in StridedSlice operator is not "
            "equal to the size of axes attribute. The strides attribute's "
            "size is %d, axes attribute's size is %d.",
            strides_size,
            axes.size()));
  }
  // we need to analysis strided slice op is valid for
  // the parameter that we get from python front
  std::vector<int64_t> out_dims_vector(in_dims.size(), -1);
  if (!tensor_input || config.is_runtime) {
    phi::funcs::StridedSliceOutDims(starts_,
                                    ends_,
                                    strides_,
                                    axes,
                                    infer_flags,
                                    in_dims,
                                    decrease_axis,
                                    out_dims_vector.data(),
                                    axes.size(),
                                    true);
  }
  DDim out_dims(common::make_ddim(out_dims_vector));
  // generate new shape
  if (!decrease_axis.empty()) {
    std::vector<int64_t> new_out_shape;
    for (size_t i = 0; i < decrease_axis.size(); ++i) {
      if (config.is_runtime && infer_flags[i] != -1) {
        PADDLE_ENFORCE_EQ(out_dims[decrease_axis[i]],
                          1,
                          errors::InvalidArgument(
                              "the size of decrease dimension should be 1, "
                              "but received %d.",
                              out_dims[decrease_axis[i]]));
      }
      out_dims[decrease_axis[i]] = 0;
    }

    for (int i = 0; i < out_dims.size(); ++i) {
      if (out_dims[i] != 0) {
        new_out_shape.push_back(out_dims[i]);
      }
    }
    out_dims = common::make_ddim(new_out_shape);
  }
  VLOG(4) << "out_dims: " << out_dims;
  out->set_dims(out_dims);
  out->share_lod(x);
  out->set_dtype(x.dtype());
}

void StridedSliceInferMeta(const MetaTensor& x,
                           const std::vector<int>& axes,
                           const IntArray& starts,
                           const IntArray& ends,
                           const IntArray& strides,
                           MetaTensor* out,
                           MetaConfig config) {
  std::vector<int> infer_flags(axes.size(), 1);
  std::vector<int> decrease_axis;
  StridedSliceRawInferMeta(
      x, axes, starts, ends, strides, infer_flags, decrease_axis, out, config);
}

void SumRawInferMeta(const MetaTensor& x,
                     const IntArray& axis,
                     bool keep_dim,
                     bool reduce_all,
                     DataType dtype,
                     MetaTensor* out,
                     MetaConfig config) {
  DDim out_dim;
  if (config.is_runtime || !axis.FromTensor()) {
    out_dim = ReduceInferDim(x, axis.GetData(), keep_dim, reduce_all);
  } else {
    out_dim = ReduceInferDimForIntArrayAxis(x, axis, keep_dim, reduce_all);
  }

  DataType out_dtype;
  if (dtype != DataType::UNDEFINED) {
    out_dtype = dtype;
  } else {
    if (x.dtype() == DataType::BOOL || x.dtype() == DataType::INT32) {
      out_dtype = DataType::INT64;
    } else {
      out_dtype = x.dtype();
    }
  }

  out->set_dims(out_dim);
  out->set_dtype(out_dtype);
  out->set_layout(x.layout());
}

/*  Why not use SumRawInferMeta directly?
    Because we need make InferMetaFunction's args follow the design of
   ops.yaml
*/
void SumInferMeta(const MetaTensor& x,
                  const IntArray& axis,
                  DataType dtype,
                  bool keep_dim,
                  MetaTensor* out,
                  MetaConfig config) {
  bool reduce_all = false;
  if (axis.size() == 0) {
    reduce_all = true;
  }
  SumRawInferMeta(x, axis, keep_dim, reduce_all, dtype, out, config);
}

void DetInferMeta(const MetaTensor& x, MetaTensor* out, MetaConfig config) {
  // remove the last two dimension
  auto out_dim = common::vectorize<int>(x.dims());
  out_dim.pop_back();
  out_dim.pop_back();

  out->set_dims(common::make_ddim(out_dim));
  out->set_dtype(x.dtype());
  out->set_layout(x.layout());
}

void PartialSumInferMeta(const std::vector<const MetaTensor*>& xs,
                         int start_index,
                         int length,
                         MetaTensor* out,
                         MetaConfig config) {
  int64_t batch_size = -1;
  int64_t input_len = -1;

  auto inputs_num = xs.size();
  PADDLE_ENFORCE_GT(inputs_num,
                    0,
                    common::errors::InvalidArgument(
                        "ShapeError: Input tensors count should > 0. But "
                        "received inputs' length is 0."));

  if (inputs_num == 1) {
    VLOG(3) << "Warning: partial_sum op have only one input, may be useless";
  }

  // Only support two dimensions now, should be extended later
  // when length is -1, need make sure all dimensions to be added are the same
  for (size_t i = 0; i < inputs_num; i++) {
    auto x_dim = xs[i]->dims();

    PADDLE_ENFORCE_EQ(x_dim.size(),
                      2,
                      common::errors::InvalidArgument(
                          "Only support two dimensions input now."));

    if (i == 0) {
      batch_size = x_dim[0];
      input_len = x_dim[1];
    } else {
      // each tensor's dim must eq
      PADDLE_ENFORCE_EQ(x_dim[0],
                        batch_size,
                        common::errors::InvalidArgument(
                            "The batch size of all inputs must be same"));
      PADDLE_ENFORCE_EQ(x_dim[1],
                        input_len,
                        common::errors::InvalidArgument(
                            "The input len of all inputs must be same"));
    }
  }
  PADDLE_ENFORCE_GT(
      input_len,
      start_index,
      common::errors::OutOfRange("start_index must be less than input len"));
  if (length > 0) {
    PADDLE_ENFORCE_GE(input_len,
                      start_index + length,
                      common::errors::OutOfRange(
                          "start_index + length is larger than input length"));
  }

  std::vector<int64_t> out_dims(2);
  out_dims[0] = batch_size;
  out_dims[1] = (length == -1) ? input_len - start_index : length;
  DDim out_dim = common::make_ddim(out_dims);
  out->set_dims(out_dim);
  out->set_dtype(xs[0]->dtype());
}

void PartialConcatInferMeta(const std::vector<const MetaTensor*>& xs,
                            int start_index,
                            int length,
                            MetaTensor* out,
                            MetaConfig config) {
  int64_t batch_size = -1;
  int64_t input_len = -1;

  auto inputs_num = xs.size();
  PADDLE_ENFORCE_GT(inputs_num,
                    0,
                    common::errors::InvalidArgument(
                        "ShapeError: Input tensors count should > 0. But "
                        "received inputs' length is 0."));

  // Only support two dimensions now, should be extended later
  // when length is -1, need make sure all dimensions to be added are the same
  for (size_t i = 0; i < inputs_num; i++) {
    auto x_dim = xs[i]->dims();

    PADDLE_ENFORCE_EQ(x_dim.size(),
                      2,
                      common::errors::InvalidArgument(
                          "Only support two dimensions input now."));

    if (i == 0) {
      batch_size = x_dim[0];
      input_len = x_dim[1];
    } else {
      // each tensor's dim must eq
      PADDLE_ENFORCE_EQ(x_dim[0],
                        batch_size,
                        common::errors::InvalidArgument(
                            "The batch size of all inputs must be same"));
      PADDLE_ENFORCE_EQ(x_dim[1],
                        input_len,
                        common::errors::InvalidArgument(
                            "The input len of all inputs must be same"));
    }
  }

  PADDLE_ENFORCE_EQ(
      start_index >= -input_len && start_index < input_len,
      true,
      common::errors::InvalidArgument(
          "The start_index is expected to be in range of [%d, %d), but got %d",
          -input_len,
          input_len,
          start_index));

  if (start_index < 0) {
    start_index += input_len;
  }

  if (length > 0) {
    PADDLE_ENFORCE_GE(input_len,
                      start_index + length,
                      common::errors::OutOfRange(
                          "start_index + length is larger than input length"));
  }

  std::vector<int64_t> out_dims(2);
  out_dims[0] = batch_size;
  // colnum = input_num * length
  out_dims[1] = (length < 0) ? input_len - start_index : length;
  out_dims[1] *= inputs_num;
  DDim out_dim = common::make_ddim(out_dims);
  out->set_dims(out_dim);
  out->set_dtype(xs[0]->dtype());
}

void SvdvalsInferMeta(const MetaTensor& x, MetaTensor* s) {
  auto SDDim = [](const DDim& x_dim, int k) {
    auto x_vec = common::vectorize(x_dim);
    x_vec.erase(x_vec.end() - 2, x_vec.end());
    x_vec.push_back(k);
    return common::make_ddim(x_vec);
  };

  auto in_dims = x.dims();
  int64_t x_rank = in_dims.size();

  PADDLE_ENFORCE_GE(
      x_rank,
      2,
      common::errors::InvalidArgument("The rank of input tensor must be >= 2"));

  int64_t m = in_dims[x_rank - 2];
  int64_t n = in_dims[x_rank - 1];

  int64_t k = std::min(m, n);
  s->set_dims(SDDim(in_dims, k));
  s->share_lod(x);
  s->set_dtype(x.dtype());
}

void SvdInferMeta(const MetaTensor& x,
                  bool full_matrices,
                  MetaTensor* u,
                  MetaTensor* s,
                  MetaTensor* vh) {
  auto UDDim = [](const DDim& x_dim, int k) {
    // get x_dim and return the ddim of U
    auto x_vec = common::vectorize(x_dim);
    x_vec[x_vec.size() - 1] = k;
    return common::make_ddim(x_vec);
  };

  auto VHDDim = [](const DDim& x_dim, int k) {
    // get x_dim and return the ddim of U
    auto x_vec = common::vectorize(x_dim);
    x_vec[x_vec.size() - 2] = k;
    return common::make_ddim(x_vec);
  };

  auto SDDim = [](const DDim& x_dim, int k) {
    // get x_dim and return the ddim of U
    auto x_vec = common::vectorize(x_dim);
    x_vec[x_vec.size() - 2] = k;
    x_vec.erase(x_vec.end() - 1);  // rank - 1
    return common::make_ddim(x_vec);
  };

  auto in_dims = x.dims();
  int x_rank = in_dims.size();
  PADDLE_ENFORCE_GE(
      in_dims.size(),
      2,
      common::errors::InvalidArgument("the rank of input must greater than 2"));
  int m = static_cast<int>(in_dims[x_rank - 2]);
  int n = static_cast<int>(in_dims[x_rank - 1]);
  int k = std::min(m, n);
  u->set_dims(!full_matrices ? UDDim(in_dims, k) : UDDim(in_dims, m));
  vh->set_dims(!full_matrices ? VHDDim(in_dims, k) : VHDDim(in_dims, n));
  s->set_dims(SDDim(in_dims, k));
  u->share_lod(x);
  vh->share_lod(x);
  s->share_lod(x);
  u->set_dtype(x.dtype());
  vh->set_dtype(x.dtype());
  s->set_dtype(x.dtype());
}

void TemporalShiftInferMeta(const MetaTensor& x,
                            int seg_num,
                            float shift_ratio,
                            const std::string& data_format,
                            MetaTensor* out,
                            MetaConfig config) {
  auto dim_x = x.dims();
  PADDLE_ENFORCE_EQ(dim_x.size(),
                    4,
                    common::errors::InvalidArgument(
                        "Input(X) rank should be 4 in shape of [N*T, C, H, "
                        "W], but received X rank(%d)",
                        dim_x.size()));

  PADDLE_ENFORCE_GT(
      seg_num,
      0,
      common::errors::InvalidArgument(
          "Attr(seg_num) should be greater than 0, but received %d", seg_num));
  PADDLE_ENFORCE_GT(
      shift_ratio,
      0.,
      common::errors::InvalidArgument(
          "Attr(shift_ratio) should be greater than 0, but received %d",
          shift_ratio));
  PADDLE_ENFORCE_LT(
      shift_ratio,
      0.5,
      common::errors::InvalidArgument(
          "Attr(shift_ratio) should be less than 0.5, but received %d",
          shift_ratio));

  if (config.is_runtime) {
    PADDLE_ENFORCE_EQ(dim_x[0] % seg_num,
                      0,
                      common::errors::InvalidArgument(
                          "Input(X) dimension[0] should be divided exactly "
                          "by Attr(seg_num), but received X dimension[0](%d) "
                          "mod seg_num(%d) != 0",
                          dim_x[0],
                          seg_num));
  }

  out->share_meta(x);
}

void TileInferMeta(const MetaTensor& x,
                   const IntArray& repeat_times,
                   MetaTensor* out,
                   MetaConfig config) {
#define TILE_MAX_RANK_SUPPORTED 6

  auto repeat_times_data = repeat_times.GetData();
  auto x_dims = x.dims();
  if (repeat_times_data.empty()) {
    repeat_times_data = std::vector<int64_t>(x_dims.size(), 1);
  }

  PADDLE_ENFORCE_LE(
      x_dims.size(),
      TILE_MAX_RANK_SUPPORTED,
      errors::InvalidArgument(
          "The rank of the input 'x' for tile op "
          "must not be greater than %d, but the value received is %d.",
          TILE_MAX_RANK_SUPPORTED,
          x_dims.size()));
  PADDLE_ENFORCE_LE(
      repeat_times_data.size(),
      TILE_MAX_RANK_SUPPORTED,
      errors::InvalidArgument(
          "The size of the shape of input 'repeat_times' for tile op "
          "must not be greater than %d, but the value received is %d.",
          TILE_MAX_RANK_SUPPORTED,
          repeat_times_data.size()));
  PADDLE_ENFORCE_GE(
      repeat_times_data.size(),
      0,
      errors::InvalidArgument(
          "The size of the shape of input 'repeat_times' for tile op "
          "must be positive integers, but the value received is %d.",
          repeat_times_data.size()));

  auto out_rank =
      std::max(static_cast<size_t>(x_dims.size()), repeat_times_data.size());
  std::vector<int64_t> out_shape(out_rank);
  auto x_dim_vec = common::vectorize<int>(x_dims);
  if (x_dim_vec.size() > repeat_times_data.size()) {
    auto diff = x_dim_vec.size() - repeat_times_data.size();
    repeat_times_data.insert(repeat_times_data.begin(), diff, 1);
  } else {
    auto diff = repeat_times_data.size() - x_dim_vec.size();
    x_dim_vec.insert(x_dim_vec.begin(), diff, 1);
  }
  for (size_t i = 0; i < repeat_times_data.size(); ++i) {
    if (x_dim_vec[i] == -1 || repeat_times_data[i] == -1) {
      out_shape[i] = -1;
    } else {
      PADDLE_ENFORCE_GT(
          repeat_times_data[i],
          0,
          errors::InvalidArgument(
              "Every element of the input 'repeat_times' for tile op must be "
              "greater than 0, but the value given is %d.",
              repeat_times_data[i]));
      out_shape[i] = x_dim_vec[i] * repeat_times_data[i];
    }
  }

  out->set_dims(common::make_ddim(out_shape));
  if (out_rank > 0 && (out_shape[0] == x_dims[0])) {
    out->share_lod(x);
  }
  out->set_dtype(x.dtype());
#undef TILE_MAX_RANK_SUPPORTED
}

void TopKInferMeta(const MetaTensor& x,
                   const Scalar& k_scalar,
                   int axis,
                   bool largest,
                   bool sorted,
                   MetaTensor* out,
                   MetaTensor* indices,
                   MetaConfig config) {
  auto input_dims = x.dims();
  const int& dim_size = input_dims.size();
  if (dim_size != 0) {
    PADDLE_ENFORCE_EQ(
        (axis < dim_size) && (axis >= (-1 * dim_size)),
        true,
        common::errors::InvalidArgument(
            "the axis of topk must be [-%d, %d), but you set axis is %d",
            dim_size,
            dim_size,
            axis));
  } else {
    PADDLE_ENFORCE_EQ(
        (axis == dim_size) || (axis == -1),
        true,
        common::errors::InvalidArgument("the axis of topk must be 0 or -1 when "
                                        "x.dims() = 0, but you set axis is %d",
                                        axis));
  }

  if (axis < 0) axis += dim_size;

  int k = k_scalar.to<int>();
  if (k_scalar.FromTensor()) {
    k = -1;
  } else {
    PADDLE_ENFORCE_EQ(k >= 1,
                      true,
                      common::errors::InvalidArgument(
                          "the attribute of k in the topk must >= 1 or be a "
                          "Tensor, but received %d .",
                          k));
  }

  PADDLE_ENFORCE_GE(
      input_dims.size(),
      0,
      common::errors::InvalidArgument("input of topk must have >= 0d shape"));

  phi::DDim dims = input_dims;
  if (input_dims.size() > 0) {
    dims[axis] = k;
  }
  out->set_dims(dims);
  out->share_lod(x);
  out->set_dtype(x.dtype());
  indices->set_dims(dims);
  indices->share_lod(x);
  indices->set_dtype(DataType::INT64);
}

void TopkV1InferMeta(const MetaTensor& x,
                     const Scalar& k_scalar,
                     MetaTensor* out,
                     MetaTensor* indices,
                     MetaConfig config) {
  const auto& input_dims = x.dims();

  int k = k_scalar.to<int>();
  if (k_scalar.FromTensor()) {
    k = -1;
  } else {
    PADDLE_ENFORCE_EQ(k >= 1,
                      true,
                      common::errors::InvalidArgument(
                          "the attribute of k in the topk must >= 1 or be a "
                          "Tensor, but received %d .",
                          k));
  }

  PADDLE_ENFORCE_GE(
      input_dims.size(),
      0,
      common::errors::InvalidArgument("input of topk must have >= 0d shape"));

  phi::DDim dims = input_dims;
  dims[dims.size() - 1] = k;
  out->set_dims(dims);
  out->share_lod(x);
  out->set_dtype(x.dtype());
  indices->set_dims(dims);
  indices->share_lod(x);
  indices->set_dtype(DataType::INT64);
}

void TraceInferMeta(
    const MetaTensor& x, int offset, int axis1, int axis2, MetaTensor* out) {
  int dim1 = axis1;
  int dim2 = axis2;

  auto x_dims = x.dims();

  int dim1_ = dim1 < 0 ? x_dims.size() + dim1 : dim1;
  int dim2_ = dim2 < 0 ? x_dims.size() + dim2 : dim2;

  PADDLE_ENFORCE_GE(
      x_dims.size(),
      2,
      common::errors::OutOfRange(
          "Input(x)'s dim is out of range (expected at least 2, but got %ld).",
          x_dims.size()));
  PADDLE_ENFORCE_LT(
      dim1_,
      x_dims.size(),
      common::errors::OutOfRange(
          "axis1 is out of range (expected to be in range of [%ld, "
          "%ld], but got %ld).",
          -(x_dims.size()),
          (x_dims.size() - 1),
          dim1));
  PADDLE_ENFORCE_GE(
      dim1_,
      0,
      common::errors::OutOfRange(
          "axis1 is out of range (expected to be in range of [%ld, "
          "%ld], but got %ld).",
          -(x_dims.size()),
          (x_dims.size() - 1),
          dim1));
  PADDLE_ENFORCE_LT(
      dim2_,
      x_dims.size(),
      common::errors::OutOfRange(
          "axis2 is out of range (expected to be in range of [%ld, "
          "%ld], but got %ld).",
          -(x_dims.size()),
          (x_dims.size() - 1),
          dim2));
  PADDLE_ENFORCE_GE(
      dim2_,
      0,
      common::errors::OutOfRange(
          "axis2 is out of range (expected to be in range of [%ld, "
          "%ld], but got %ld).",
          -(x_dims.size()),
          (x_dims.size() - 1),
          dim2));
  PADDLE_ENFORCE_NE(
      dim1_,
      dim2_,
      common::errors::InvalidArgument("The dimensions should not be identical "
                                      "%ld vs %ld.",
                                      dim1,
                                      dim2));

  auto sizes = common::vectorize(x_dims);
  if (x_dims.size() == 2) {
    sizes.clear();
  } else {
    sizes.erase(sizes.begin() + std::max(dim1_, dim2_));
    sizes.erase(sizes.begin() + std::min(dim1_, dim2_));
  }
  out->set_dims(common::make_ddim(sizes));
  out->set_dtype(x.dtype());
}

void TransferLayoutInferMeta(const MetaTensor& x,
                             int src_layout,
                             int dst_layout,
                             MetaTensor* out) {
  out->set_dims(x.dims());
  out->set_dtype(x.dtype());
  out->set_layout(static_cast<DataLayout>(dst_layout));
  out->share_lod(x);
}

void TransposeInferMeta(const MetaTensor& x,
                        const std::vector<int>& axis,
                        MetaTensor* out) {
  auto x_dims = x.dims();
  int x_rank = x_dims.size();
  int axis_size = static_cast<int>(axis.size());

  // Note: x_rank > axis_size when fuse squeeze2 + transpose2, else x_rank ==
  // axis_size
  PADDLE_ENFORCE_GE(x_rank,
                    axis_size,
                    errors::InvalidArgument(
                        "The input tensor's dimension "
                        "should be equal to or greater than the axis's size. "
                        "But received input tensor's dimension is %d, "
                        "axis's size is %d",
                        x_rank,
                        axis_size));

  std::vector<int> formatted_axis = axis;
  std::vector<int> count(axis_size, 0);
  for (int i = 0; i < axis_size; i++) {
    PADDLE_ENFORCE_LT(axis[i],
                      x_rank,
                      errors::InvalidArgument(
                          "The reduce dim index %d should be in the "
                          "range [ -dimension(X), dimension(X) ) "
                          "which dimension = %d. But received dim index = %d.",
                          i,
                          x_rank,
                          axis[i]));
    PADDLE_ENFORCE_GE(axis[i],
                      -x_rank,
                      errors::InvalidArgument(
                          "The reduce dim index %d should be in the "
                          "range [ -dimension(X), dimension(X) )  "
                          "which dimension = %d. But received dim index = %d.",
                          i,
                          x_rank,
                          axis[i]));

    if (axis[i] < 0) {
      formatted_axis[i] = axis[i] + x_rank;
    }
    PADDLE_ENFORCE_EQ(
        ++count[formatted_axis[i]],
        1,
        errors::InvalidArgument("Each element of axis should be unique. but "
                                "axis[%d] is %d appear not only once",
                                i,
                                axis[i]));
  }

  phi::DDim out_dims(x_dims);
  for (int i = 0; i < axis_size; ++i) {
    out_dims[i] = x_dims[formatted_axis[i]];
  }

  out->set_dims(out_dims);
  out->set_dtype(x.dtype());
}

void UnbindInferMeta(const MetaTensor& x,
                     int axis,
                     std::vector<MetaTensor*> outs) {
  auto in_dims = x.dims();
  std::vector<int> out_dim;

  PADDLE_ENFORCE_GE(
      axis,
      -in_dims.size(),
      common::errors::InvalidArgument(
          "axis must be in range(%d, %d).", -in_dims.size(), in_dims.size()));
  PADDLE_ENFORCE_LT(
      axis,
      in_dims.size(),
      common::errors::InvalidArgument(
          "axis must be in range(%d, %d).", -in_dims.size(), in_dims.size()));

  axis = axis < 0 ? in_dims.size() + axis : axis;

  for (int i = 0; i < in_dims.size(); ++i) {
    if (i != axis) out_dim.push_back(in_dims[i]);  // NOLINT
  }
  auto out_dims = common::make_ddim(out_dim);

  for (auto& out : outs) {
    out->set_dtype(x.dtype());
    out->set_dims(out_dims);
    out->set_layout(x.layout());
    out->share_lod(x);
  }
}

void TrilTriuInferMeta(const MetaTensor& x,
                       int diagonal,
                       bool lower,
                       MetaTensor* out) {
  const auto& x_dims = x.dims();
  PADDLE_ENFORCE_GE(x_dims.size(),
                    2,
                    common::errors::InvalidArgument(
                        "Input(X)'s rank must be at least 2 in TrilTriuOp."));
  out->set_dims(x.dims());
  out->share_lod(x);
  out->set_dtype(x.dtype());
}

void TrilInferMeta(const MetaTensor& x, int diagonal, MetaTensor* out) {
  TrilTriuInferMeta(x, diagonal, true, out);
}

void TriuInferMeta(const MetaTensor& x, int diagonal, MetaTensor* out) {
  TrilTriuInferMeta(x, diagonal, false, out);
}

// Some operator having oneDnn kernel will be set layout in kernel.
void UnchangedExceptLayoutInferMeta(const MetaTensor& x, MetaTensor* out) {
  out->set_dims(x.dims());
  out->set_dtype(x.dtype());
  out->share_lod(x);
}

void UnchangedExceptDtypeInferMeta(const MetaTensor& x, MetaTensor* out) {
  out->set_dims(x.dims());
  out->set_layout(x.layout());
  out->share_lod(x);
}

void UnchangedInferMetaIncludingTensorArray(const MetaTensor& x,
                                            MetaTensor* out) {
  if (x.is_tensor_array()) {
    UnchangedArrayInferMeta(x, out);
  } else {
    UnchangedInferMeta(x, out);
  }
}

void UnchangedInferMeta(const MetaTensor& x, MetaTensor* out) {
  out->share_meta(x);
}

void UnchangedArrayInferMeta(const MetaTensor& x, MetaTensor* out) {
  out->set_dtype(x.dtype());
  out->set_layout(x.layout());
}

void UnchangedVectorInferMeta(const std::vector<const MetaTensor*>& xs,
                              std::vector<MetaTensor*> outs) {
  for (size_t i = 0; i < xs.size(); ++i) {
    outs[i]->share_meta(*xs[i]);
  }
}

// meta x -> out without change, check if axis in range [-Rank(x), Rank(x)-1]
void UnchangedInferMetaCheckAxis(const MetaTensor& x,
                                 int axis,
                                 MetaTensor* out) {
  auto rank = x.dims().size();
  if (rank > 0) {
    PADDLE_ENFORCE_GE(axis,
                      -rank,
                      common::errors::InvalidArgument(
                          "Attr(axis) value should be in range [-R, R-1], "
                          "R is the rank of Input(X)."));
    PADDLE_ENFORCE_LT(axis,
                      rank,
                      common::errors::InvalidArgument(
                          "Attr(axis) value should be in range [-R, R-1], "
                          "R is the rank of Input(X)."));
  } else if (rank == 0) {
    PADDLE_ENFORCE_GE(axis,
                      -1,
                      common::errors::InvalidArgument(
                          "Attr(axis) value should be in range [-1, "
                          "0] when input is 0D Tensor "));
    PADDLE_ENFORCE_LE(axis,
                      0,
                      common::errors::InvalidArgument(
                          "Attr(axis) value should be in range [-1, "
                          "0] when input is 0D Tensor "));
  }
  out->share_meta(x);
}

void UnfoldInferMeta(const MetaTensor& x,
                     const std::vector<int>& kernel_sizes,
                     const std::vector<int>& strides,
                     const std::vector<int>& paddings,
                     const std::vector<int>& dilations,
                     MetaTensor* out,
                     MetaConfig config) {
  auto in_dims = x.dims();
  // Only [N, C, H, W] input supported now
  PADDLE_ENFORCE_EQ(
      in_dims.size(),
      4,
      common::errors::InvalidArgument(
          "Input should be 4-D tensor of format [N, C, H, W], but get %u",
          in_dims.size()));
  PADDLE_ENFORCE_EQ(
      in_dims.size() - kernel_sizes.size(),
      2U,
      common::errors::InvalidArgument(
          "The dims of X should be larger than that of kernel_sizes "
          "by a number of 2, due to the batch size and input channel dim. "
          "But received dims(X:%u) - dims(kernel_sizes:%u) != 2",
          in_dims.size(),
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

  // check kernel_sizes
  PADDLE_ENFORCE_GT(kernel_sizes[0],
                    0,
                    common::errors::InvalidArgument(
                        "The `kernel_sizes` should be greater than zero, "
                        "but received kernel_height: %d kernel_width: %d.",
                        kernel_sizes[0],
                        kernel_sizes[1]));
  PADDLE_ENFORCE_GT(kernel_sizes[1],
                    0,
                    common::errors::InvalidArgument(
                        "The `kernel_sizes` should be greater than zero, "
                        "but received kernel_height: %d kernel_width: %d.",
                        kernel_sizes[0],
                        kernel_sizes[1]));
  // check strides
  PADDLE_ENFORCE_GT(strides[0],
                    0,
                    common::errors::InvalidArgument(
                        "The `strides` should be greater than zero, "
                        "but received strides_height: %d strides_width: %d.",
                        strides[0],
                        strides[1]));
  PADDLE_ENFORCE_GT(strides[1],
                    0,
                    common::errors::InvalidArgument(
                        "The `strides` should be greater than zero, "
                        "but received strides_height: %d strides_width: %d.",
                        strides[0],
                        strides[1]));
  // check dilations
  PADDLE_ENFORCE_GT(
      dilations[0],
      0,
      common::errors::InvalidArgument(
          "The `dilations` should be greater than zero, "
          "but received dilations_height: %d dilations_width: %d.",
          dilations[0],
          dilations[1]));
  PADDLE_ENFORCE_GT(
      dilations[1],
      0,
      common::errors::InvalidArgument(
          "The `dilations` should be greater than zero, "
          "but received dilations_height: %d dilations_width: %d.",
          dilations[0],
          dilations[1]));

  std::vector<int> out_dims;
  out_dims.push_back(in_dims[0]);  // NOLINT
  int output_channels =
      static_cast<int>(in_dims[1] * kernel_sizes[0] * kernel_sizes[1]);
  out_dims.push_back(output_channels);

  int output_height = phi::funcs::CalcOutputSize(static_cast<int>(in_dims[2]),
                                                 kernel_sizes[0],
                                                 dilations[0],
                                                 paddings[0],
                                                 paddings[2],
                                                 strides[0]);
  int output_width = phi::funcs::CalcOutputSize(static_cast<int>(in_dims[3]),
                                                kernel_sizes[1],
                                                dilations[1],
                                                paddings[1],
                                                paddings[3],
                                                strides[1]);
  int output_col_length = output_height * output_width;
  if (config.is_runtime) {
    // only check output height and width in runtime
    PADDLE_ENFORCE_GT(
        output_height,
        0,
        common::errors::InvalidArgument(
            "The sliding blocks calculated from input spatial size "
            "(%d, %d), kernel_sizes (%d, %d), strides (%d, %d), "
            "dilations (%d, %d), is (%d, %d), which should be a "
            "positive integer.",
            in_dims[2],
            in_dims[3],
            kernel_sizes[0],
            kernel_sizes[1],
            strides[0],
            strides[1],
            dilations[0],
            dilations[1],
            output_height,
            output_width));
    PADDLE_ENFORCE_GT(
        output_width,
        0,
        common::errors::InvalidArgument(
            "The sliding blocks calculated from input spatial size "
            "(%d, %d), kernel_sizes (%d, %d), strides (%d, %d), "
            "dilations (%d, %d), is (%d, %d), which should be a "
            "positive integer.",
            in_dims[2],
            in_dims[3],
            kernel_sizes[0],
            kernel_sizes[1],
            strides[0],
            strides[1],
            dilations[0],
            dilations[1],
            output_height,
            output_width));
  } else {
    output_col_length =
        output_height == -1 || output_width == -1 ? -1 : output_col_length;
  }
  out_dims.push_back(output_col_length);
  out->set_dims(common::make_ddim(out_dims));
  out->set_dtype(x.dtype());
}

void UniformRandomInplaceInferMeta(const MetaTensor& x,
                                   float min,
                                   float max,
                                   int seed,
                                   int diag_num,
                                   int diag_step,
                                   float diag_val,
                                   MetaTensor* out) {
  PADDLE_ENFORCE_LT(
      min,
      max,
      errors::InvalidArgument(
          "The uniform_random's min must less then max. But received min = "
          "%f great than or equal max = %f.",
          min,
          max));
  PADDLE_ENFORCE_GE(diag_num,
                    0,
                    errors::InvalidArgument(
                        "The uniform_random's diag_num must greater than or "
                        "equal 0. But received diag_num (%d) < 0.",
                        diag_num));
  PADDLE_ENFORCE_GE(diag_step,
                    0,
                    errors::InvalidArgument(
                        "The uniform_random's diag_step must greater than or "
                        "equal 0. But received diag_step (%d) < 0.",
                        diag_step));
  PADDLE_ENFORCE_NE(out,
                    nullptr,
                    common::errors::InvalidArgument(
                        "uniform_random should have output tensor out."));
  auto xdim = x.dims();
  out->set_dims(xdim);
  out->set_dtype(x.dtype());
}

void UniformRandomBatchSizeLikeInferMeta(const MetaTensor& input,
                                         const std::vector<int>& shape,
                                         int input_dim_idx,
                                         int output_dim_idx,
                                         float min,
                                         float max,
                                         int seed,
                                         int diag_num,
                                         int diag_step,
                                         float diag_val,
                                         DataType dtype,
                                         MetaTensor* out,
                                         MetaConfig config) {
  phi::BatchSizeLikeInferMeta(input, shape, input_dim_idx, output_dim_idx, out);
  out->set_dtype(dtype);
}

void UniqueConsecutiveInferMeta(const MetaTensor& x,
                                bool return_inverse,
                                bool return_counts,
                                const std::vector<int>& axis,
                                DataType dtype,
                                MetaTensor* out,
                                MetaTensor* index,
                                MetaTensor* counts) {
  PADDLE_ENFORCE_NE(out,
                    nullptr,
                    common::errors::InvalidArgument(
                        "unique_consecutive should have output tensor out."));

  auto in_dims = x.dims();
  if (x.dims().size() == 0) {
    PADDLE_ENFORCE_EQ(axis.empty(),
                      true,
                      common::errors::InvalidArgument(
                          "The Input(X) with 0-D Tensor, axis must be None"
                          "But now the axis is %d.",
                          axis[0]));
  }
  if (return_inverse) {
    PADDLE_ENFORCE_NE(
        index,
        nullptr,
        common::errors::InvalidArgument("Tensor index should not be null if "
                                        "return_inverse is set to True."));
  }
  if (return_counts) {
    PADDLE_ENFORCE_NE(
        counts,
        nullptr,
        common::errors::InvalidArgument("Tensor counts should not be null if "
                                        "return_counts is set to True."));
  }

  if (axis.empty()) {
    out->set_dims({-1});
    out->set_dtype(x.dtype());
    if (return_inverse) {
      index->set_dims({common::product(in_dims)});
    }
  } else {
    int axis_value = axis[0];
    if (axis_value < 0) {
      axis_value += in_dims.size();
    }
    PADDLE_ENFORCE_LT(
        axis_value,
        in_dims.size(),
        common::errors::InvalidArgument("The axis(%d) should be less than "
                                        "the dimension size(%d) of x.",
                                        axis_value,
                                        in_dims.size()));
    auto out_dims = in_dims;
    out_dims[axis_value] = -1;
    out->set_dims(out_dims);
    out->set_dtype(x.dtype());
    if (return_inverse) {
      index->set_dims({in_dims[axis_value]});
    }
  }
  if (return_counts) {
    counts->set_dims({-1});
  }
}

void UniqueInferMeta(const MetaTensor& x,
                     bool return_index,
                     bool return_inverse,
                     bool return_counts,
                     const std::vector<int>& axis,
                     DataType dtype,
                     MetaTensor* out,
                     MetaTensor* indices,
                     MetaTensor* index,
                     MetaTensor* counts) {
  bool is_sorted = true;
  UniqueRawInferMeta(x,
                     return_index,
                     return_inverse,
                     return_counts,
                     axis,
                     dtype,
                     is_sorted,
                     out,
                     indices,
                     index,
                     counts);
}

void UniqueRawInferMeta(const MetaTensor& x,
                        bool return_index,
                        bool return_inverse,
                        bool return_counts,
                        const std::vector<int>& axis,
                        DataType dtype,
                        bool is_sorted,
                        MetaTensor* out,
                        MetaTensor* indices,
                        MetaTensor* index,
                        MetaTensor* counts) {
  if (!is_sorted) {
    PADDLE_ENFORCE_EQ(x.dims().size() == 1 || x.dims().size() == 0,
                      true,
                      common::errors::InvalidArgument(
                          "The Input(X) should be 0-D or 1-D Tensor, "
                          "But now the dims of Input(X) is %d.",
                          x.dims().size()));
    out->set_dims(common::make_ddim({-1}));
    index->set_dims(x.dims());
    return;
  }

  if (x.dims().size() == 0) {
    PADDLE_ENFORCE_EQ(axis.empty(),
                      true,
                      common::errors::InvalidArgument(
                          "The Input(X) with 0-D Tensor, axis must be None"
                          "But now the axis is %d.",
                          axis[0]));
  }

  if (axis.empty()) {
    out->set_dims(common::make_ddim({-1}));
    if (return_inverse) {
      index->set_dims(common::make_ddim({common::product(x.dims())}));
    }
  } else {
    int axis_value = axis[0];
    if (axis_value < 0) {
      axis_value += x.dims().size();
    }

    PADDLE_ENFORCE_LT(
        axis_value,
        x.dims().size(),
        common::errors::InvalidArgument("The axis(%d) should be less than "
                                        "the dimension size(%d) of x.",
                                        axis_value,
                                        x.dims().size()));
    PADDLE_ENFORCE_GE(
        axis_value,
        0,
        common::errors::InvalidArgument(
            "The axis(%d) + rank(x) (%d) should be greater than or equal to 0.",
            axis_value,
            -x.dims().size()));

    auto out_dims = x.dims();
    out_dims[axis_value] = -1;
    out->set_dims(out_dims);
    if (return_inverse) {
      index->set_dims(common::make_ddim({x.dims()[axis_value]}));
      index->set_dtype(dtype);
    }
  }
  if (return_index) {
    indices->set_dims(common::make_ddim({-1}));
    indices->set_dtype(dtype);
  }
  if (return_counts) {
    counts->set_dims(common::make_ddim({-1}));
    counts->set_dtype(dtype);
  }
}

void UnsqueezeInferMeta(const MetaTensor& x,
                        const IntArray& axes,
                        MetaTensor* out,
                        MetaConfig config) {
#define UNSQUEEZE_MAX_RANK_SUPPORTED 8
  const auto& x_dims = x.dims();
  // Validity Check: input tensor dims (<=8).
  PADDLE_ENFORCE_LE(x_dims.size(),
                    UNSQUEEZE_MAX_RANK_SUPPORTED,
                    common::errors::InvalidArgument(
                        "Invalid "
                        "dimensions, the rank of Input(X) "
                        "should be in the range of [1, %d] (Eigen limit)",
                        UNSQUEEZE_MAX_RANK_SUPPORTED));
  if (!config.is_runtime && axes.FromTensor()) {
    // compile time infershape.  set all elements to -1.
    int output_size = static_cast<int>(x.dims().size() + axes.GetData().size());
    std::vector<int64_t> vec_out_dims(output_size, -1);
    out->set_dtype(x.dtype());
    out->set_dims(common::make_ddim(vec_out_dims));
  } else {
    auto out_dims = funcs::GetUnsqueezeShape(axes.GetData(), x_dims);
    out->set_dims(out_dims);
    if (x_dims.size() > 0 && x_dims[0] == out_dims[0]) {
      out->share_lod(x);
    }
    out->set_dtype(x.dtype());
  }
#undef UNSQUEEZE_MAX_RANK_SUPPORTED
}

void UnsqueezeWithXShapeInferMeta(const MetaTensor& x,
                                  const IntArray& axes,
                                  MetaTensor* out,
                                  MetaTensor* xshape,
                                  MetaConfig config) {
  const auto& x_dims = x.dims();
  UnsqueezeInferMeta(x, axes, out, config);
  // set xshape dims.
  std::vector<int64_t> xshape_dims(x_dims.size() + 1);
  xshape_dims[0] = 0;
  for (int i = 0; i < x_dims.size(); ++i) {
    xshape_dims[i + 1] = x_dims[i];
  }
  if (xshape) {
    xshape->set_dims(common::make_ddim(xshape_dims));
    xshape->share_lod(x);
    xshape->set_dtype(x.dtype());
  }
}

void UnStackInferMeta(const MetaTensor& x,
                      int axis,
                      int num,
                      std::vector<MetaTensor*> outs) {
  auto x_dim = x.dims();
  int rank = x_dim.size();
  PADDLE_ENFORCE_GE(axis,
                    -rank,
                    common::errors::InvalidArgument(
                        "The attribute axis is out of range, it must be "
                        "inside [-rank, rank), where rank = %d",
                        rank));
  PADDLE_ENFORCE_LT(axis,
                    rank,
                    common::errors::InvalidArgument(
                        "The attribute axis is out of range, it must be "
                        "inside [-rank, rank), where rank = %d",
                        rank));
  if (axis < 0) axis += rank;

  size_t output_count = outs.size();
  PADDLE_ENFORCE_EQ(output_count,
                    static_cast<size_t>(num),
                    common::errors::InvalidArgument(
                        "Number of Outputs(Y) is wrong. Got %d , but it must "
                        "equal to attribute num which is %d.",
                        output_count,
                        static_cast<size_t>(num)));
  if (x_dim[axis] > 0) {
    PADDLE_ENFORCE_EQ(
        num,
        x_dim[axis],
        common::errors::InvalidArgument(
            "The number of attribute num is not equal to the length of the "
            "%d axis of Input(X). Expect %d but got %d.",
            axis,
            x_dim[axis],
            num));
  }
  auto vec = common::vectorize<int>(x_dim);
  vec.erase(vec.begin() + axis);
  for (size_t i = 0; i < output_count; i++) {
    outs[i]->set_dims(common::make_ddim(vec));
    outs[i]->set_dtype(x.dtype());
  }
}

void WeightQuantizeInferMeta(const MetaTensor& x,
                             const std::string& algo,
                             const int32_t arch,
                             const int32_t group_size,
                             MetaTensor* out,
                             MetaTensor* scale) {
#ifdef PADDLE_WITH_CUDA
  PADDLE_ENFORCE_EQ(
      ((arch == 70) || (arch == 75) || (arch == 80) || (arch == 86) ||
       (arch == 89) || (arch == 90)),
      true,
      common::errors::InvalidArgument(
          "Currently, arch only support 70, 75, 80, 86, 89, 90."));
#endif

  auto x_dims = x.dims();
  PADDLE_ENFORCE_EQ(
      x_dims.size(),
      2UL,
      common::errors::InvalidArgument(
          "The x tensor of quant op must be 2D, but got[%d]", x_dims.size()));
  PADDLE_ENFORCE_EQ(
      x_dims[0] % 64,
      0,
      common::errors::InvalidArgument(
          "The first dimension of input must be divisible by 64, but got[%d]",
          x_dims[0]));
  PADDLE_ENFORCE_EQ(
      x_dims[1] % 16,
      0,
      common::errors::InvalidArgument(
          "The second dimension of input must be divisible by 16, but got[%d]",
          x_dims[1]));
  PADDLE_ENFORCE_EQ(
      ((group_size == -1) || (group_size == 64) || (group_size == 128)),
      true,
      common::errors::InvalidArgument(
          "Currently, group_size only support -1, 64 or 128."));

  std::vector<int64_t> dim_scale;
  if (group_size != -1) {
    int64_t scale_dim0 = (x_dims[0] + (group_size - 1)) / group_size;
    int64_t scale_dim1 = x_dims[1];
    dim_scale = std::vector<int64_t>({scale_dim0, scale_dim1});
  } else {
    dim_scale = std::vector<int64_t>({x_dims[1]});
  }

  std::vector<int64_t> dim_out;
  if (algo == "weight_only_int8" || algo == "llm.int8") {  // NOLINT
    dim_out = std::vector<int64_t>({x_dims[1], x_dims[0]});
  } else if (algo == "weight_only_int4") {
    dim_out = std::vector<int64_t>({x_dims[1] / 2, x_dims[0]});
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "The algo must be in ['weight_only_int8', 'weight_only_int4', "
        "'llm.int8'], but got[%s]",
        algo));
  }
  out->set_dims(common::make_ddim(dim_out));

  out->set_dtype(DataType::INT8);

  scale->set_dims(phi::make_ddim(dim_scale));
  scale->set_dtype(x.dtype());
}

void ChannelShuffleInferMeta(const MetaTensor& x,
                             int groups,
                             const std::string& data_format,
                             MetaTensor* out) {
  auto input_dims = x.dims();
  PADDLE_ENFORCE_EQ(input_dims.size(),
                    4,
                    common::errors::InvalidArgument(
                        "Input should be a 4-D tensor of format [N, C, H, W] "
                        "or [N, H, W, C], but got %u.",
                        input_dims.size()));
  PADDLE_ENFORCE_GE(
      groups,
      1,
      common::errors::InvalidArgument("groups should be larger than 0."));
  PADDLE_ENFORCE_EQ(data_format == "NCHW" || data_format == "NHWC",
                    true,
                    common::errors::InvalidArgument(
                        "data_format must be one of "
                        "NCHW and NHWC. But received data_format: %s",
                        data_format));

  const bool channel_last = (data_format == "NHWC");

  if (!channel_last) {
    PADDLE_ENFORCE_EQ(input_dims[1] % groups,
                      0,
                      common::errors::InvalidArgument(
                          "The number of groups to divide channels in [%u] "
                          "should divide the number of channel [%u]",
                          groups,
                          input_dims[1]));
  } else {
    PADDLE_ENFORCE_EQ(input_dims[3] % groups,
                      0,
                      common::errors::InvalidArgument(
                          "The number of groups to divide channels in [%u] "
                          "should divide the number of channel [%u]",
                          groups,
                          input_dims[3]));
  }
  auto output_dims = input_dims;
  out->set_dtype(x.dtype());
  out->set_dims(output_dims);
}

void CheckNumericsInferMeta(const MetaTensor& tensor,
                            const std::string& op_type,
                            const std::string& var_name,
                            const int check_nan_inf_level,
                            const int stack_height_limit,
                            const std::string& output_dir,
                            MetaTensor* stats,
                            MetaTensor* values) {
  stats->set_dtype(DataType::INT64);
  stats->set_dims(common::make_ddim({3}));
  values->set_dtype(DataType::FLOAT32);
  values->set_dims(common::make_ddim({3}));
}

void StridedUnChangedInferMeta(const MetaTensor& x, MetaTensor* out) {
  out->share_meta(x);
  out->set_strides(x.strides());
}

void StraightThroughEstimatorInferMeta(const MetaTensor& out_grad,
                                       MetaTensor* x_grad) {
  x_grad->set_dims(out_grad.dims());
  x_grad->set_dtype(out_grad.dtype());
}

void NumberCountInferMeta(const MetaTensor& x,
                          int upper_range,
                          MetaTensor* out) {
  out->share_meta(x);
}

void LrnInferMeta(const MetaTensor& x,
                  int n,
                  MetaTensor* out,
                  MetaTensor* mid_out) {
  auto x_dim = x.dims();
  PADDLE_ENFORCE_EQ(
      x_dim.size(),
      4,
      common::errors::InvalidArgument("Input(input) rank should be 4, "
                                      "but received input rank (%d) != 4",
                                      x_dim.size()));

  PADDLE_ENFORCE_GT(
      n,
      0UL,
      common::errors::InvalidArgument("Argument(n) should be positive, "
                                      "but received n(%d) not greater than 0",
                                      n));
  PADDLE_ENFORCE_EQ(
      n % 2,
      1UL,
      common::errors::InvalidArgument("Argument(n) should be odd value, "
                                      "but received n(%d) is not an odd value",
                                      n));

  out->set_dtype(x.dtype());
  out->set_dims(x.dims());
  out->set_layout(x.layout());
  out->share_lod(x);
  mid_out->set_dtype(x.dtype());
  mid_out->set_dims(x.dims());
  mid_out->set_layout(x.layout());
  mid_out->share_lod(x);
}

void ArrayPopInferMeta(const MetaTensor& array,
                       int index,
                       MetaTensor* array_out,
                       MetaTensor* out,
                       MetaConfig config) {
  array_out->set_dtype(array.dtype());
  array_out->set_layout(array.layout());
  if (config.is_runtime) {
    return;
  }
  // TODO(dev): Support get a tensor dims from TensorArray.
  // out->set_dims(x.dims());
  out->set_dtype(array.dtype());
}

}  // namespace phi

PD_REGISTER_INFER_META_FN(flatten, phi::FlattenInferMeta);
