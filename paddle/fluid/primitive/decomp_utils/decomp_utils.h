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

#pragma once
#include <vector>

#include "paddle/common/ddim.h"
#include "paddle/fluid/framework/details/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/primitive/base/lazy_tensor.h"
#include "paddle/fluid/primitive/primitive/primitive.h"
#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/kernels/funcs/common_infer_shape_functions.h"

namespace paddle {
class Tensor;
namespace primitive {
template <typename T>
static Tensor get_slice(const Tensor& x, int64_t idx) {
  return slice<T>(x, {0}, {idx}, {idx + 1}, {1}, {});
}

template <typename T>
static Tensor get_slice_vec(const Tensor& x,
                            int64_t start_idx,
                            int64_t end_idx) {
  return slice<T>(x, {0}, {start_idx}, {end_idx}, {1}, {});
}

template <typename T>
void set_output(const Tensor& x_tmp, Tensor* x);

template <typename T>
void by_pass(const Tensor& x_tmp, Tensor* x);

// This function determine whether dtype is in [float16, bfloat16, uint16]
static bool is_half_dtype(const DataType& dtype) {
  if (dtype == phi::DataType::FLOAT16 || dtype == phi::DataType::BFLOAT16 ||
      dtype == phi::DataType::UINT16) {
    return true;
  } else {
    return false;
  }
}

// This function expands the dimension of origin Tensor based on the value of
// axis
static std::vector<int64_t> get_expand_dims(const Tensor& origin,
                                            const std::vector<int64_t>& axis) {
  std::vector<int64_t> result(origin.shape());
  for (size_t i = 0; i < axis.size(); ++i) {
    int64_t offset = axis[i];
    if (offset < 0) {
      offset += result.size() + 1;
    }

    PADDLE_ENFORCE_LE(
        offset,
        result.size(),
        common::errors::OutOfRange("Your index [%lu] exceeds the number of "
                                   "elements in origin_dims[%lu].",
                                   offset,
                                   result.size()));
    result.insert(result.begin() + offset, 1);
  }
  return result;
}

// This function compute unsqueeze dims for reshape to replace unsqueeze.
static std::vector<int64_t> get_unsqueeze_dims(
    const Tensor& origin, const std::vector<int64_t>& axis) {
  auto sort_axis = axis;
  std::sort(sort_axis.begin(), sort_axis.end());
  auto origin_dims = origin.shape();
  auto total_shape_size = origin_dims.size() + sort_axis.size();
  std::vector<int64_t> result;
  size_t j = 0, k = 0;
  for (size_t i = 0; i < total_shape_size; ++i) {
    if (j < sort_axis.size() && sort_axis[j] == int64_t(i)) {
      result.push_back(1);
      j++;
    } else {
      PADDLE_ENFORCE_LT(
          k,
          origin_dims.size(),
          common::errors::OutOfRange("Your index [%lu] exceeds the number of "
                                     "elements in origin_dims[%lu].",
                                     k,
                                     origin_dims.size()));
      result.push_back(origin_dims[k]);
      k++;
    }
  }
  return result;
}

// This function compute `dynamic` unsqueeze dims for reshape to replace
// unsqueeze. And should used only on `dynamic`.
template <typename T>
Tensor get_unsqueeze_dims(const Tensor& origin_shape,
                          const std::vector<int64_t>& axis) {
  auto total_shape_size = origin_shape.numel() + axis.size();
  const Tensor one = full<T>({1}, 1, origin_shape.dtype());

  std::vector<Tensor> result(total_shape_size, one);
  // to support axis not in increasing order.
  std::vector<bool> is_set(total_shape_size, false);

  for (size_t i = 0; i < axis.size(); ++i) {
    PADDLE_ENFORCE_LT(
        axis[i],
        total_shape_size,
        common::errors::OutOfRange("Your index [%lu] exceeds the number of "
                                   "elements in origin_dims[%lu].",
                                   axis[i],
                                   total_shape_size));
    is_set[axis[i]] = true;
  }

  size_t j = 0;
  for (size_t i = 0; i < total_shape_size; ++i) {
    if (is_set[i]) {
      continue;
    }
    result[i] = get_slice<T>(origin_shape, int64_t(j));
    is_set[i] = true;
    ++j;
  }
  return concat<T>(result);
}

// This function compute unsqueeze dims for reshape to replace unsqueeze.
static std::vector<int64_t> get_squeeze_dims(const Tensor& origin,
                                             const std::vector<int64_t>& axis) {
  auto origin_dims = origin.shape();
  auto total_shape_size = origin_dims.size();
  std::vector<int64_t> result;
  for (size_t i = 0; i < total_shape_size; ++i) {
    if (origin_dims[i] != 1) {
      result.push_back(origin_dims[i]);
    } else if (origin_dims[i] == 1 &&
               std::find(axis.begin(), axis.end(), int64_t(i)) == axis.end()) {
      result.push_back(1);
    } else {
      continue;
    }
  }
  return result;
}

static std::vector<int64_t> process_dims(const Tensor& origin,
                                         const std::vector<int64_t>& axis) {
  auto origin_dims = origin.shape();
  auto total_shape_size = origin_dims.size();
  std::vector<int64_t> result;
  auto axis_size = axis.size();
  if (axis_size == 0) {
    for (size_t i = 0; i < total_shape_size; ++i) {
      result.push_back(i);
    }
  } else {
    for (size_t i = 0; i < axis_size; ++i) {
      if (axis[i] < 0) {
        result.push_back(axis[i] + total_shape_size);
      } else {
        result.push_back(axis[i]);
      }
    }
  }
  return result;
}

// These method don't need to be specified
// These method only handle the static shape case
static phi::DDim get_reduce_dims_from_out(const phi::DDim& dout_dims,
                                          const phi::DDim& in_dims) {
  bool has_dynamic_shape = false;
  for (int i = 0; i < dout_dims.size(); i++) {
    if (dout_dims[i] == -1) {
      has_dynamic_shape = true;
      break;
    }
  }
  PADDLE_ENFORCE_EQ(
      has_dynamic_shape,
      false,
      common::errors::InvalidArgument(
          "Function get_reduce_dims_from_out() only use in static shape case, "
          "but the input [dout_dims] have the dynamic shape."));

  for (int i = 0; i < in_dims.size(); i++) {
    if (in_dims[i] == -1) {
      has_dynamic_shape = true;
      break;
    }
  }
  PADDLE_ENFORCE_EQ(
      has_dynamic_shape,
      false,
      common::errors::InvalidArgument(
          "Function get_reduce_dims_from_out() only use in static shape case, "
          "but the input [in_dims] have the dynamic shape."));

  int bat = dout_dims.size() - in_dims.size();
  std::vector<int64_t> result;
  for (int i = 0; i < bat; ++i) {
    result.push_back(i);
  }
  for (int i = 0; i < in_dims.size(); ++i) {
    if (in_dims[i] == 1 && dout_dims[i + bat] != 1) {
      result.push_back(i + bat);
    } else {
      PADDLE_ENFORCE_EQ(
          in_dims[i],
          dout_dims[i + bat],
          common::errors::InvalidArgument(
              "ReduceDims dimension mismatch. Operands could "
              "not be broadcast together with the shape of dout = [%s] and "
              "the shape of in_dims = [%s]. Received [%d] in X is not equal to "
              "[%d] in Y at i:%d.",
              dout_dims,
              in_dims,
              dout_dims[i + bat],
              in_dims[i],
              i));
    }
  }
  return common::make_ddim(result);
}

static phi::DDim get_reduce_dims(const phi::DDim& x_dims,
                                 const phi::DDim& y_dims) {
  auto out_dims = phi::funcs::BroadcastTwoDims(x_dims, y_dims);
  return get_reduce_dims_from_out(out_dims, x_dims);
}

void SetEmptyGrad(const std::vector<std::vector<Tensor>>& outputs,
                  const std::vector<std::vector<bool>>& stop_gradients);

std::vector<std::vector<Tensor>> ConstructVjpResultByStopGradients(
    const std::vector<std::vector<Tensor>>& outputs,
    const std::vector<std::vector<bool>>& stop_gradients);

static bool find_value(const std::vector<int64_t>& vec, int64_t value) {
  if (std::find(vec.begin(), vec.end(), value) != vec.end()) {
    return true;
  } else {
    return false;
  }
}

static bool has_dynamic_shape(const std::vector<int64_t>& shape) {
  return std::find(shape.begin(), shape.end(), -1) != shape.end();
}

static bool has_dynamic_shape(const std::vector<int64_t>& shape,
                              const std::vector<int64_t>& axis) {
  bool flag = false;
  const int64_t rank = shape.size();
  for (int64_t idx : axis) {
    if (idx < 0) idx += rank;
    PADDLE_ENFORCE_LT(
        idx,
        rank,
        ::common::errors::PreconditionNotMet(
            "Required idx < shape.size(), but received %d.", idx));
    if (shape[idx] == -1) {
      flag = true;
      break;
    }
  }
  return flag;
}

template <typename T>
Tensor ConvertToMT(const Tensor& x) {
  bool need_cast = x.dtype() == phi::DataType::FLOAT16 ||
                   x.dtype() == phi::DataType::BFLOAT16 ||
                   x.dtype() == phi::DataType::UINT16;
  if (need_cast) {
    return cast<T>(x, phi::DataType::FLOAT32);
  }
  return x;
}

template <typename T>
Tensor ConvertToOrig(const Tensor& out, phi::DataType input_dtype) {
  bool need_cast = out.dtype() != input_dtype;
  if (need_cast) {
    return cast<T>(out, input_dtype);
  }
  return out;
}

class LayerNormDecompHelper {
 public:
  LayerNormDecompHelper(const Tensor& x,
                        const paddle::optional<Tensor>& scale,
                        const paddle::optional<Tensor>& bias,
                        int begin_norm_axis) {
    auto x_dims = x.dims();
    x_rank_ = x_dims.size();
    begin_norm_axis_ = begin_norm_axis;
    if (begin_norm_axis_ < 0) {
      begin_norm_axis_ += x_rank_;
    }

    scale_need_reshape_ = (begin_norm_axis + 1 != x_rank_);
    static_norm_shape_ = true;

    for (int i = begin_norm_axis; i < x_rank_; ++i) {
      if (x_dims[i] < 0) {
        static_norm_shape_ = false;
        normalized_numel_ = -1;
        break;
      }

      normalized_shape_.push_back(x_dims[i]);

      normalized_numel_ *= x_dims[i];
    }

    if (!static_norm_shape_) {
      // try get static norm numel from scale for bias
      normalized_numel_ = -1;
      if (scale.get_ptr()) {
        normalized_numel_ = scale->dims()[0];
      } else if (bias.get_ptr()) {
        normalized_numel_ = bias->dims()[0];
      }
    }
  }

  template <typename T>
  Tensor Process(const Tensor& s, const Tensor& x) {
    if (!scale_need_reshape_) {
      return s;
    }

    if (static_norm_shape_) {
      return reshape<T>(s, normalized_shape_);
    } else {
      return backend::reshape<T>(
          s, get_slice_vec<T>(shape64<T>(x), begin_norm_axis_, x_rank_));
    }
  }

  template <typename T>
  Tensor GetNormalizedNumel(const Tensor& x) {
    if (normalized_numel_ != -1) {
      return full_scalar<T>(normalized_numel_, x.dtype());
    } else {
      auto x_shape = shape64<T>(x);
      auto numel = get_slice<T>(x_shape, begin_norm_axis_);
      for (int64_t i = begin_norm_axis_ + 1; i < x_rank_; ++i) {
        numel = numel * get_slice<T>(x_shape, i);
      }

      return cast<T>(numel, x.type());
    }
  }

 private:
  std::vector<int64_t> normalized_shape_;
  bool scale_need_reshape_;
  bool static_norm_shape_;
  int64_t x_rank_;
  int64_t normalized_numel_{1};
  int begin_norm_axis_;
};

template <typename T>
class BatchNormDecompHelper {
 public:
  BatchNormDecompHelper(const Tensor& x,
                        const paddle::optional<Tensor>& scale,
                        const paddle::optional<Tensor>& bias,
                        const std::string& data_format) {
    auto x_dims = phi::vectorize(x.dims());
    x_rank_ = x_dims.size();

    if (data_format == "NCHW") {
      channel_axis_ = 1;
      reduce_axis_.push_back(0);
      for (int64_t i = channel_axis_ + 1; i < x_rank_; ++i) {
        reduce_axis_.push_back(i);
      }
    } else if (data_format == "NHWC") {
      channel_axis_ = x_rank_ - 1;
      for (int64_t i = 0; i < channel_axis_; ++i) {
        reduce_axis_.push_back(i);
      }
    } else {
      PADDLE_THROW(
          common::errors::Unimplemented("Only support NCHW and NHWC format."));
    }

    scale_bias_new_shape_.push_back(0);
    for (int64_t i = channel_axis_ + 1; i < x_rank_; ++i) {
      scale_bias_new_shape_.push_back(1);
    }

    // int64_t channel_dim = x_dims[channel_axis_];
    // if ((channel_dim < 0) && scale) {
    //   channel_dim = scale->dims()[0];
    // }
    // if ((channel_dim < 0) && bias) {
    //   channel_dim = bias->dims()[0];
    // }
  }

  const std::vector<int64_t>& GetReduceAxis() const { return reduce_axis_; }

  const std::vector<int64_t>& GetScaleBiasNewShape() const {
    return scale_bias_new_shape_;
  }

  Tensor GetNHW(const Tensor& x) {
    auto x_dims = x.dims();

    bool static_nhw = true;
    int64_t nhw_numel = 1;
    for (int64_t i = 0; i < x_rank_; ++i) {
      if (i == channel_axis_) {
        continue;
      }
      if (x_dims[i] < 0) {
        static_nhw = false;
        break;
      }
      nhw_numel *= x_dims[i];
    }

    if (static_nhw) {
      return full_scalar<T>(nhw_numel, x.dtype());
    } else {
      auto x_shape = shape64<T>(x);
      auto nhw = get_slice<T>(x_shape, 0);
      for (int64_t i = 1; i < x_rank_; ++i) {
        if (i == channel_axis_) {
          continue;
        }

        nhw = nhw * get_slice<T>(x_shape, i);
      }

      return cast<T>(nhw, x.dtype());
    }
  }

 private:
  std::vector<int64_t> reduce_axis_;
  std::vector<int64_t> scale_bias_new_shape_;
  int64_t channel_axis_;
  int64_t x_rank_;
};

template <typename T>
class InstanceNormDecompHelper {
 public:
  explicit InstanceNormDecompHelper(const Tensor& x) {
    x_dims_ = phi::vectorize(x.dims());
    x_rank_ = x_dims_.size();

    for (int64_t i = 2; i < x_rank_; ++i) {
      reduce_axis_.push_back(i);
      n_plus_reduce_axis_.push_back(i);
    }
    n_plus_reduce_axis_.push_back(0);
  }

  Tensor GetHW(const Tensor& x) {
    auto dims = phi::vectorize(x.dims());
    int64_t rank = dims.size();
    if (has_dynamic_shape(x.shape())) {
      Tensor x_shape = shape64<T>(x);
      auto hw = full_scalar<T>(1.0, x.dtype());
      for (int64_t i = 2; i < rank; ++i) {
        hw = hw * get_slice<T>(x_shape, i);
      }
      return cast<T>(hw, x.dtype());
    } else {
      int64_t hw = 1;
      for (int64_t i = 2; i < rank; ++i) {
        hw *= dims[i];
      }
      return full_scalar<T>(hw, x.dtype());
    }
  }

  const std::vector<int64_t> GetReduceAxis() const { return reduce_axis_; }
  const std::vector<int64_t> GetNPlusReduceAxis() const {
    return n_plus_reduce_axis_;
  }
  const std::vector<int64_t>& GetDims() const { return x_dims_; }

 private:
  std::vector<int64_t> reduce_axis_;
  std::vector<int64_t> n_plus_reduce_axis_;
  std::vector<int64_t> x_dims_;
  int64_t x_rank_;
};

}  // namespace primitive
}  // namespace paddle
