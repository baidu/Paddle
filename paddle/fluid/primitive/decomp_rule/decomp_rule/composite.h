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

#include <numeric>
#include "paddle/fluid/primitive/base/lazy_tensor.h"
#include "paddle/fluid/primitive/decomp_utils/decomp_utils.h"

namespace paddle {
namespace primitive {
namespace details {

template <typename T>
Tensor any_decomp(const Tensor& x, const IntArray& axis, bool keepdim) {
  auto org_dtype = x.dtype();

  auto res = cast<T>(sum<T>(x, axis, org_dtype, keepdim), DataType::BOOL);
  if (org_dtype != DataType::BOOL) {
    return cast<T>(res, org_dtype);
  } else {
    return res;
  }
}

template <typename T>
Tensor mean_decomp(const Tensor& x, const IntArray& axis, bool keepdim) {
  auto x_tmp = ConvertToMT<T>(x);

  std::vector<int64_t> x_dim = x_tmp.shape();
  int64_t axis_size = axis.size();
  int64_t x_dim_size = x_dim.size();
  auto axis_ = std::vector<int64_t>();
  if (axis_size == 0) {
    for (int64_t i = 0; i < x_dim_size; i++) {
      axis_.push_back(i);
    }
  } else {
    axis_ = axis.GetData();
    for (int64_t i = 0; i < axis_size; i++) {
      if (axis[i] < 0) {
        axis_[i] = axis[i] + x_dim_size;
      }
    }
  }
  auto sum_x = sum<T>(x_tmp, axis_, x_tmp.dtype(), keepdim);

  Tensor value;
  bool switch_dynamic = false;
  for (const int64_t& idx : axis_) {
    if (x_dim[idx] == -1) {
      switch_dynamic = true;
    }
  }
  if (switch_dynamic) {
    auto x_shape = shape64<T>(x);
    value = slice<T>(x_shape, {0}, {axis_[0]}, {axis_[0] + 1}, {1}, {0});
    for (size_t i = 1; i < axis_.size(); ++i) {
      value =
          value * slice<T>(x_shape, {0}, {axis_[i]}, {axis_[i] + 1}, {1}, {0});
    }

    value = cast<T>(value, x_tmp.dtype());
  } else {
    int64_t value_ = 1;
    for (size_t i = 0; i < axis_.size(); i++) {
      value_ *= x_dim[axis_[i]];
    }
    value = full_scalar<T>(value_, sum_x.dtype(), sum_x.place());
  }

  Tensor res = sum_x / value;

  return ConvertToOrig<T>(res, x.dtype());
}

static void check_valid_type(const DataType& dtype) {
  switch (dtype) {
    case DataType::INT8:
    case DataType::INT16:
    case DataType::INT32:
    case DataType::INT64:
    case DataType::UINT8:
    case DataType::UINT16:
    case DataType::UINT32:
    case DataType::UINT64:
    case DataType::FLOAT16:
    case DataType::FLOAT32:
    case DataType::FLOAT64:
      break;
    default:
      PADDLE_THROW(common::errors::InvalidArgument(
          "Unsupported data type: %s", phi::DataTypeToString(dtype)));
  }
}

template <typename T>
Tensor p_norm_decomp(const Tensor& x,
                     const float& porder = 2.0,
                     const int& axis = -1,
                     const float epsilon = 1.0e-12f,
                     const bool& keepdim = false,
                     const bool& asvector = false) {
  // NOTE: if asvector is True, then axis will be ignored
  // and will reduce all elements in x

  auto x_tmp = ConvertToMT<T>(x);

  Tensor res;
  std::vector<int> reduce_axis = {};
  if (!asvector) {
    reduce_axis.push_back(axis);
  }
  if (porder == 0.0) {
    // 0-norm
    auto zero = full_scalar<T>(0, x_tmp.dtype(), x_tmp.place());
    auto none_zero = not_equal<T>(x_tmp, zero);
    res = cast<T>(none_zero, x_tmp.dtype());
    res = sum<T>(res, reduce_axis, x_tmp.dtype(), keepdim);
  } else if (porder == 1.0) {
    // 1-norm
    res = abs<T>(x_tmp);
    res = sum<T>(res, reduce_axis, x_tmp.dtype(), keepdim);
  } else if (porder == 2.0) {
    // 2-norm
    res = sqrt<T>(sum<T>(x_tmp * x_tmp, reduce_axis, x_tmp.dtype(), keepdim));
  } else if (porder == INFINITY) {
    // +INF-norm
    res = abs<T>(x_tmp);
    res = max<T>(x_tmp, reduce_axis, keepdim);
  } else if (porder == -INFINITY) {
    // -INF-norm
    res = abs<T>(x_tmp);
    res = min<T>(x_tmp, reduce_axis, keepdim);
  } else {
    // vanilla p-norm
    auto porder_tensor = full_scalar<T>(porder, x_tmp.dtype(), x_tmp.place());
    auto inv_porder_tensor =
        full_scalar<T>(1 / porder, x_tmp.dtype(), x_tmp.place());
    res = elementwise_pow<T>(x_tmp, porder_tensor);
    res = sum<T>(res, reduce_axis, x_tmp.dtype(), keepdim);
    res = elementwise_pow<T>(res, inv_porder_tensor);
  }

  return ConvertToOrig<T>(res, x.dtype());
}

template <typename T>
std::tuple<Tensor, Tensor> huber_loss_decomp(const Tensor& input,
                                             const Tensor& label,
                                             float delta) {
  Tensor delta_full;
  if (has_dynamic_shape(input.shape())) {
    delta_full =
        backend::full_with_tensor<T>(shape64<T>(input), delta, input.dtype());
  } else {
    delta_full = full<T>(input.shape(), delta, input.dtype(), input.place());
  }
  auto val = label - input;
  auto abs_val = abs<T>(val);
  auto factor = full_scalar<T>(0.5, input.dtype(), input.place());
  auto ans = where<T>(abs_val <= delta_full,
                      factor * val * val,
                      delta_full * (abs_val - factor * delta_full));
  return std::make_tuple(ans, val);
}

template <typename T>
Tensor one_hot_decomp(const Tensor& x, const Tensor& num_classes) {
  auto start = full<T>({1}, 0, x.dtype(), x.place());
  auto step = full<T>({1}, 1, x.dtype(), x.place());
  auto arange_class =
      backend::arange<T>(start, num_classes, step, x.dtype(), x.place());
  auto reshape_x = backend::unsqueeze<T>(x, {-1});
  auto equal_res = backend::equal<T>(reshape_x, arange_class);
  return cast<T>(equal_res, phi::DataType::FLOAT32);
}

template <typename T>
Tensor squared_l2_norm_decomp(const Tensor& x) {
  auto res = sum<T>(x * x, {}, x.dtype(), false);
  return backend::reshape<T>(res, {1});
}

template <typename T>
Tensor reciprocal_decomp(const Tensor& x) {
  return full_scalar<T>(1.0, x.dtype(), x.place()) / x;
}

template <typename T>
Tensor bce_loss_decomp(const Tensor& x, const Tensor& label) {
  auto org_dtype = x.dtype();
  auto x_mt = ConvertToMT<T>(x);

  auto neg_100 = full_scalar<T>(-100, x_mt.dtype(), x.place());
  auto one = full_scalar<T>(1, x_mt.dtype(), x.place());

  auto log_x = maximum<T>(log<T>(x_mt), neg_100);
  auto log_1_x = maximum<T>(log<T>(one - x_mt), neg_100);

  auto ans = full_scalar<T>(-1, x_mt.dtype(), x.place()) *
             (label * log_x + (one - label) * log_1_x);
  ans = ConvertToOrig<T>(ans, org_dtype);

  return ans;
}

template <typename T>
Tensor bmm_decomp(const Tensor& x, const Tensor& y) {
  std::size_t x_ndims = x.dims().size();
  std::size_t y_ndims = y.dims().size();
  if (x_ndims != 3) {
    PADDLE_THROW(common::errors::InvalidArgument(
        "Input(X) of BmmOp must be 3-dimensional in BmmOp, "
        "but received X's shape: [%s].",
        x_ndims));
  }
  if (y_ndims != 3) {
    PADDLE_THROW(common::errors::InvalidArgument(
        "Input(Y) of BmmOp must be 3-dimensional in BmmOp, "
        "but received Y's shape: [%s].",
        y_ndims));
  }

  auto x_shape = phi::vectorize(x.dims());
  auto y_shape = phi::vectorize(y.dims());

  if (x_shape[0] != y_shape[0] && x_shape[0] != -1 && y_shape[0] != -1) {
    PADDLE_THROW(common::errors::InvalidArgument(
        "Input(X) and Input(Y) must have the same batch size in BmmOp, "
        "but received X's batch size: [%s],"
        "Y's batch size [%s].",
        x_shape[0],
        y_shape[0]));
  }

  if (x_shape[2] != y_shape[1] && x_shape[2] != -1 && y_shape[1] != -1) {
    PADDLE_THROW(common::errors::InvalidArgument(
        "Input(X)'s width must be equal with Input(Y)'s height in BmmOp,"
        "but receive X's width: [%s],"
        "Y's height: [%s].",
        x_shape[2],
        y_shape[1]));
  }
  return matmul<T>(x, y, false, false);
}

template <typename T>
std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor> batch_norm_decomp(
    const Tensor& x,
    const Tensor& run_mean,
    const Tensor& run_var,
    const paddle::optional<Tensor>& scale,
    const paddle::optional<Tensor>& bias,
    bool is_test,
    float momentum,
    float epsilon,
    const std::string& data_layout,
    bool use_global_stats,
    bool trainable_statistics) {
  auto org_dtype = x.dtype();
  Tensor x_cast = ConvertToMT<T>(x);

  BatchNormDecompHelper<T> decomp_help(x, scale, bias, data_layout);

  auto reduce_axes = decomp_help.GetReduceAxis();
  auto scale_bias_new_shape = decomp_help.GetScaleBiasNewShape();

  bool use_run_stat = (is_test && (!trainable_statistics)) || use_global_stats;

  Tensor y, run_mean_, run_var_, batch_mean_, inv_std_, reserve_space;

  std::vector<int64_t> x_dim = x_cast.shape();
  std::vector<int64_t> stats_shape;
  Tensor eps = full_scalar<T>(epsilon, x_cast.dtype(), x_cast.place());

  Tensor x_hat;
  Tensor batch_mean;
  Tensor inv_std;
  if (!use_run_stat) {
    batch_mean = mean_decomp<T>(x_cast, reduce_axes, true);
    auto temp = mean_decomp<T>(x_cast * x_cast, reduce_axes, true);
    auto batch_var = temp - batch_mean * batch_mean;
    inv_std = rsqrt<T>(batch_var + eps);

    x_hat = (x_cast - batch_mean) * inv_std;

    run_mean_ = reshape<T>(run_mean, scale_bias_new_shape) * momentum +
                batch_mean * (1. - momentum);
    run_var_ = reshape<T>(run_var, scale_bias_new_shape) * momentum +
               batch_var * (1. - momentum);

    run_mean_ = squeeze<T>(run_mean_, reduce_axes);
    run_var_ = squeeze<T>(run_var_, reduce_axes);
    assign_out_<T>(run_mean_, run_mean);
    assign_out_<T>(run_var_, run_var);
  } else {
    x_hat = (x_cast - reshape<T>(run_mean, scale_bias_new_shape)) *
            rsqrt<T>(reshape<T>(run_var, scale_bias_new_shape) + eps);

    run_mean_ = run_mean;
    run_var_ = run_var;
  }

  y = x_hat;
  if (scale) {
    y = y * (scale_bias_new_shape.size() > 1
                 ? reshape<T>(scale.get(), scale_bias_new_shape)
                 : scale.get());
  }

  if (bias) {
    y = y + (scale_bias_new_shape.size() > 1
                 ? reshape<T>(bias.get(), scale_bias_new_shape)
                 : bias.get());
  }

  y = ConvertToOrig<T>(y, org_dtype);

  if (!use_run_stat) {
    batch_mean_ = squeeze<T>(batch_mean, reduce_axes);
    inv_std_ = squeeze<T>(inv_std, reduce_axes);
    return std::make_tuple(
        y, run_mean_, run_var_, batch_mean_, inv_std_, reserve_space);
  } else {
    Tensor batch_mean_none;
    Tensor inv_std_none;
    return std::make_tuple(
        y, run_mean_, run_var_, batch_mean_none, inv_std_none, reserve_space);
  }
}

template <typename T>
Tensor softmax_decomp(const Tensor& x, const int& axis) {
  auto x_tmp = ConvertToMT<T>(x);

  auto max_tmp = max<T>(x_tmp, {axis}, true);
  auto molecular = exp<T>(x_tmp - max_tmp);
  auto res = molecular / sum<T>(molecular, {axis}, molecular.dtype(), true);

  return ConvertToOrig<T>(res, x.dtype());
}

template <typename T>
Tensor log_softmax_decomp(const Tensor& x, const int& axis) {
  auto x_tmp = ConvertToMT<T>(x);

  auto max_tmp = max<T>(x_tmp, {axis}, true);
  auto sub = x_tmp - max_tmp;
  auto molecular = exp<T>(sub);
  auto res = sub - log<T>(sum<T>(molecular, {axis}, molecular.dtype(), true));

  return ConvertToOrig<T>(res, x.dtype());
}

template <typename T>
Tensor stack_decomp(const std::vector<Tensor>& x, const int& axis) {
  std::vector<Tensor> concat_x;
  bool is_dynamic = false;
  size_t rank = x[0].shape().size();

  std::vector<int64_t> combined_shape(rank, -1);
  for (auto& item : x) {
    auto item_shape = item.shape();
    for (size_t i = 0; i < item_shape.size(); i++) {
      if (item_shape[i] == -1) {
        is_dynamic = true;
      } else {
        combined_shape[i] = std::max(combined_shape[i], item_shape[i]);
      }
    }
  }

  if (is_dynamic && has_dynamic_shape(combined_shape)) {
    std::vector<Tensor> shapes;
    Tensor temp_shape = shape64<T>(x[0]);
    for (size_t j = 0; j < rank; j++) {
      if (combined_shape[j] == -1) {
        shapes.push_back(get_slice<T>(temp_shape, j));
      } else {
        shapes.push_back(
            full<T>({1}, combined_shape[j], temp_shape.type(), x[0].place()));
      }
    }
    if (axis < 0) {
      shapes.insert(shapes.begin() + (axis + rank + 1),
                    full<T>({1}, 1, temp_shape.type(), x[0].place()));
    } else {
      shapes.insert(shapes.begin() + axis,
                    full<T>({1}, 1, temp_shape.type(), x[0].place()));
    }

    Tensor out_shape = concat<T>(shapes);
    for (size_t i = 0; i < x.size(); ++i) {
      concat_x.push_back(backend::reshape<T>(x[i], out_shape));
    }
  } else {
    std::vector<int64_t> axis_tmp = {axis};
    std::vector<int64_t> out_shape = get_expand_dims(x[0], axis_tmp);
    for (size_t i = 0; i < x.size(); ++i) {
      concat_x.push_back(reshape<T>(x[i], out_shape));
    }
  }

  return concat<T>(concat_x, axis);
}

template <typename T>
Tensor silu_decomp(const Tensor& x) {
  auto x_tmp = ConvertToMT<T>(x);
  auto res = x_tmp * sigmoid<T>(x_tmp);
  return ConvertToOrig<T>(res, x.dtype());
}

template <typename T>
Tensor swiglu_decomp(const Tensor& x, const paddle::optional<Tensor>& y) {
  if (y) {
    return silu_decomp<T>(x) * y.get();
  } else {
    int axis = x.shape().size() - 1;
    int num = 2;
    std::vector<Tensor> xs = backend::split_with_num<T>(x, num, axis);
    return silu_decomp<T>(xs[0]) * xs[1];
  }
}

template <typename T>
Tensor relu_decomp(const Tensor& x) {
  return maximum<T>(x, full_scalar<T>(0.0, x.dtype(), x.place()));
}

template <typename T>
Tensor relu6_decomp(const Tensor& x) {
  auto tmp = maximum<T>(x, full_scalar<T>(0.0, x.dtype(), x.place()));
  auto res = minimum<T>(tmp, full_scalar<T>(6.0, x.dtype(), x.place()));
  return res;
}

template <typename T>
Tensor squeeze_decomp(const Tensor& x, const IntArray& axis) {
  auto axis_ = process_dims(x, axis.GetData());
  auto out_shape = get_squeeze_dims(x, axis_);
  Tensor out = reshape<T>(x, out_shape);
  return out;
}

template <typename T>
Tensor unsqueeze_decomp(const Tensor& x, const IntArray& axis) {
  auto out_shape = get_expand_dims(x, axis.GetData());
  Tensor out = reshape<T>(x, out_shape);
  return out;
}

template <typename T>
Tensor add_n_decomp(const std::vector<Tensor>& x) {
  Tensor res = x[0];
  for (size_t i = 1; i < x.size(); i++) {
    res = res + x[i];
  }
  return res;
}

template <typename T>
std::vector<Tensor> meshgrid_decomp(const std::vector<Tensor>& x) {
  int64_t rank = x.size();
  std::vector<Tensor> res;
  std::vector<int64_t> tar_shape(rank, 1);
  for (int64_t i = 0; i < rank; i++) {
    if (x[i].shape().size() == 1) {
      tar_shape[i] = x[i].shape()[0];
    }
  }
  if (has_dynamic_shape(tar_shape)) {
    std::vector<Tensor> tmp_shape;
    for (int64_t i = 0; i < rank; i++) {
      if (tar_shape[i] == 1) {
        tmp_shape.push_back(
            full<T>({1}, tar_shape[i], DataType::INT64, x[0].place()));
      } else {
        tmp_shape.push_back(shape64<T>(x[i]));
      }
    }
    auto tar_tensor_shape = concat<T>(tmp_shape);

    for (int64_t i = 0; i < rank; i++) {
      if (tar_shape[i] == 1) {
        res.push_back(backend::expand<T>(x[i], tar_tensor_shape));
      } else {
        std::vector<int64_t> unsqueeze_dim;
        for (int64_t k = 0; k < rank; k++) {
          if (i != k) {
            unsqueeze_dim.push_back(k);
          }
        }
        res.push_back(backend::expand<T>(unsqueeze<T>(x[i], unsqueeze_dim),
                                         tar_tensor_shape));
      }
    }

  } else {
    for (int64_t i = 0; i < rank; i++) {
      std::vector<int64_t> view_shape(rank, 1);
      if (x[i].shape().size() == 1) {
        view_shape[i] = x[i].shape()[0];
      }
      res.push_back(expand<T>(reshape<T>(x[i], view_shape), tar_shape));
    }
  }

  return res;
}

template <typename T>
std::vector<Tensor> unbind_decomp(const Tensor x, int axis) {
  std::vector<Tensor> res;
  if (axis < 0) {
    axis = x.shape().size() + axis;
  }
  if (x.shape()[axis] == -1) {
    PADDLE_THROW(
        common::errors::Unimplemented("unbind axis must not be dynamic"));
  }
  size_t num = x.shape()[axis];
  std::vector<Tensor> tmp = backend::split_with_num<T>(x, num, axis);
  for (size_t i = 0; i < tmp.size(); i++) {
    res.push_back(squeeze<T>(tmp[i], {axis}));
  }
  return res;
}

template <typename T>
std::tuple<Tensor, Tensor, Tensor> layer_norm_decomp(
    const Tensor& x,
    const paddle::optional<Tensor>& scale,
    const paddle::optional<Tensor>& bias,
    float epsilon,
    int begin_norm_axis) {
  std::vector<int64_t> reduce_axis;
  auto org_dtype = x.dtype();
  Tensor x_cast = ConvertToMT<T>(x);

  auto x_dims = x.dims();

  LayerNormDecompHelper decomp_helper(x, scale, bias, begin_norm_axis);

  for (int i = begin_norm_axis; i < x_dims.size(); i++) {
    reduce_axis.push_back(static_cast<int64_t>(i));
  }
  auto mean_ = mean_decomp<T>(x_cast, reduce_axis, true);
  auto difference = x_cast - mean_;
  auto var_tmp1 = difference * difference;
  auto variance = mean_decomp<T>(var_tmp1, reduce_axis, true);
  auto var_tmp3 =
      variance + full_scalar<T>(epsilon, variance.dtype(), variance.place());
  auto rsqrt_var = rsqrt<T>(var_tmp3);
  auto out = difference * rsqrt_var;

  Tensor scale_cast;
  if (scale) {
    scale_cast = decomp_helper.Process<T>(scale.get(), x_cast);
    scale_cast = ConvertToMT<T>(scale_cast);
    out = out * scale_cast;
  }
  Tensor bias_cast;
  if (bias) {
    bias_cast = decomp_helper.Process<T>(bias.get(), x_cast);
    bias_cast = ConvertToMT<T>(bias_cast);
    out = out + bias_cast;
  }
  mean_ = squeeze<T>(mean_, reduce_axis);
  variance = squeeze<T>(variance, reduce_axis);

  // same as LayerNormInferMeta
  // x: float32 --> out: float32, mean: float32, variance: float32
  // x: float16 --> out: float16, mean: float32, variance: float32
  out = ConvertToOrig<T>(out, org_dtype);
  return std::make_tuple(out, mean_, variance);
}

template <typename T>
Tensor full_like_decomp(const Tensor& x,
                        const paddle::Scalar& value,
                        const DataType& dtype,
                        const Place& place) {
  std::vector<int64_t> x_shape = x.shape();
  if (has_dynamic_shape(x_shape)) {
    return backend::full_with_tensor<T>(shape64<T>(x), value, dtype);
  } else {
    return full<T>(x_shape, value, dtype, place);
  }
}

template <typename T>
std::tuple<Tensor, Tensor> dropout_decomp(
    const Tensor& x,
    const paddle::optional<Tensor>& seed_tensor,
    const paddle::Scalar& p,
    bool is_test,
    const std::string& mode,
    const int seed,
    bool fix_seed) {
  auto org_dtype = x.dtype();
  bool upscale_in_train = false;
  if (mode == std::string("upscale_in_train")) {
    upscale_in_train = true;
  }

  int seed_tmp = 0;
  if (fix_seed) {
    seed_tmp = seed;
  }

  auto dtype_tmp = org_dtype;
  if (is_half_dtype(org_dtype)) {
    dtype_tmp = DataType::FLOAT32;
  }
  Tensor uniform_tensor;
  if (has_dynamic_shape(x.shape())) {
    auto shape_tensor = shape64<T>(x);
    auto zero = full_scalar<T>(0.0, dtype_tmp, x.place());
    auto one = full_scalar<T>(1.0, dtype_tmp, x.place());
    uniform_tensor = backend::uniform<T>(
        shape_tensor, zero, one, org_dtype, seed_tmp, x.place());
  } else {
    uniform_tensor = uniform<T>(
        phi::vectorize(x.dims()), org_dtype, 0.0, 1.0, seed_tmp, x.place());
  }
  auto mask = cast<T>(
      greater_equal<T>(uniform_tensor,
                       full_scalar<T>(p, org_dtype, uniform_tensor.place())),
      org_dtype);
  auto ones_p = full_scalar<T>(1.0 - p.to<float>(), org_dtype, x.place());
  if (upscale_in_train) {
    if (is_test) {
      // inference: out = input
      return std::make_tuple(x, cast<T>(mask, DataType::UINT8));
    } else {
      // train: out = input * mask / ( 1.0 - p )
      if (p.to<float>() == 1.0) {
        // Process p=1. for avoid divide zero error (x*mask/(1.0-p))
        auto zero = full_like_decomp<T>(x, 0.0, org_dtype, x.place());
        return std::make_tuple(x * zero, cast<T>(zero, DataType::UINT8));
      } else {
        auto ans = (x * mask) / ones_p;
        return std::make_tuple(ans, cast<T>(mask, DataType::UINT8));
      }
    }
  } else {
    if (is_test) {
      // inference: out = input * (1.0 - p)
      return std::make_tuple(x * ones_p, cast<T>(mask, DataType::UINT8));
    } else {
      // train: out = input * mask
      return std::make_tuple(x * mask, cast<T>(mask, DataType::UINT8));
    }
  }
}

template <typename T>
Tensor gelu_decomp(const Tensor& x, bool approximate) {
  const double PM_2_SQRTPI = 1.12837916709551257390; /* 2/sqrt(pi) */
  const double PM_SQRT1_2 = 0.70710678118654752440;  /* 1/sqrt(2) */

  auto org_dtype = x.dtype();
  auto half = full_scalar<T>(0.5, org_dtype, x.place());
  auto one = full_scalar<T>(1.0, org_dtype, x.place());
  if (approximate) {
    // gelu(x) = 0.5 * x * (1 + tanh(sqrt(2 / \pi) * (x + 0.044715 * x^{3})))
    auto kAlpha =
        full_scalar<T>(PM_2_SQRTPI * PM_SQRT1_2, org_dtype, x.place());
    auto GELU_CONSTANT = full_scalar<T>(0.044715, org_dtype, x.place());
    auto x_pow3 =
        elementwise_pow<T>(x, full_scalar<T>(3, org_dtype, x.place()));
    auto tanh_out = tanh<T>(kAlpha * (x + x_pow3 * GELU_CONSTANT));

    auto res = x * half * (one + tanh_out);
    return res;
  } else {
    // gelu(x) = 0.5 * x *  (1 + erf(x / sqrt(2)))
    auto M_SQRT1_2T = full_scalar<T>(PM_SQRT1_2, org_dtype, x.place());
    auto erf_out = one + erf<T>(x * M_SQRT1_2T);

    auto res = x * half * erf_out;
    return res;
  }
}

template <typename T>
Tensor hardsigmoid_decomp(const Tensor& x, float slope, float offset) {
  const double MAX_VALUE = 1.0;
  const double MIN_VALUE = 0.0;
  return maximum<T>(minimum<T>(x * full_scalar<T>(slope, x.dtype(), x.place()) +
                                   full_scalar<T>(offset, x.dtype(), x.place()),
                               full_scalar<T>(MAX_VALUE, x.dtype(), x.place())),
                    full_scalar<T>(MIN_VALUE, x.dtype(), x.place()));
}

template <typename T>
Tensor hardswish_decomp(const Tensor& x) {
  const double OFFSET = 3.0;
  const double THRESHOLD = 6.0;
  const double SCALE = 6.0;

  // out = minimum(maximum(x + offset, 0), threshold) * x / scale
  auto minimum_out =
      minimum<T>(maximum<T>(x + full_scalar<T>(OFFSET, x.dtype(), x.place()),
                            full_scalar<T>(0.0, x.dtype(), x.place())),
                 full_scalar<T>(THRESHOLD, x.dtype()));
  return (minimum_out * x) / full_scalar<T>(SCALE, x.dtype(), x.place());
}

template <typename T>
Tensor heaviside_decomp(const Tensor& x, const Tensor& y) {
  Tensor zero, one;
  if (has_dynamic_shape(x.shape())) {
    Tensor zero_x = backend::full_with_tensor<T>(shape64<T>(x), 0.0, x.dtype());
    Tensor zero_y = backend::full_with_tensor<T>(shape64<T>(y), 0.0, x.dtype());
    zero = zero_x + zero_y;
    one = backend::full_with_tensor<T>(shape64<T>(zero), 1.0, x.dtype());
  } else {
    auto out_dims = phi::funcs::BroadcastTwoDims(x.dims(), y.dims());
    zero = full<T>(phi::vectorize(out_dims), 0.0, x.dtype(), x.place());
    one = full<T>(phi::vectorize(out_dims), 1.0, x.dtype(), x.place());
  }
  Tensor broadcast_x = x + zero;
  Tensor broadcast_y = y + zero;
  Tensor res = where<T>(broadcast_x > zero, one, broadcast_x);
  res = where<T>(broadcast_x == zero, broadcast_y, res);
  res = where<T>(broadcast_x < zero, zero, res);
  return res;
}

template <typename T>
Tensor leaky_relu_decomp(const Tensor& x, float negative_slope) {
  auto multiply_tmp = full_scalar<T>(negative_slope, x.dtype(), x.place()) * x;
  if (negative_slope < 1.0) {
    return maximum<T>(x, multiply_tmp);
  } else {
    return minimum<T>(x, multiply_tmp);
  }
}

template <typename T>
std::tuple<Tensor, Tensor, Tensor> instance_norm_decomp(
    const Tensor& x,
    const paddle::optional<Tensor>& scale,
    const paddle::optional<Tensor>& bias,
    float epsilon) {
  auto org_dtype = x.dtype();
  Tensor x_cast = ConvertToMT<T>(x);
  const std::vector<int64_t> x_dims = x.shape();

  if (has_dynamic_shape(x_dims)) {
    std::vector<int64_t> axis;
    for (size_t i = 2; i < x_dims.size(); i++) {
      axis.push_back(static_cast<int64_t>(i));
    }
    bool reduce_axes_empty = axis.empty();

    // out = (x - mean(x)) / sqrt(var + epsilon))
    // var = mean((x-mean(x))^2)
    auto mean_ =
        reduce_axes_empty ? x_cast : mean_decomp<T>(x_cast, axis, true);
    auto difference = x_cast - mean_;
    auto var_tmp1 = difference * difference;
    auto variance =
        reduce_axes_empty ? var_tmp1 : mean_decomp<T>(var_tmp1, axis, true);
    auto var_shape = shape64<T>(variance);
    auto var_tmp3 =
        variance + full_scalar<T>(epsilon, variance.dtype(), variance.place());
    auto rsqrt_var = rsqrt<T>(var_tmp3);
    auto out = difference * rsqrt_var;

    int dim_size = x_dims.size();
    auto x_shape_tensor = shape64<T>(x);
    std::vector<Tensor> slice_shape_concat;

    slice_shape_concat.push_back(full<T>({1}, 1, x_shape_tensor.dtype()));
    slice_shape_concat.push_back(
        cast<T>(get_slice<T>(x_shape_tensor, 1), x_shape_tensor.dtype()));
    if (dim_size > 2) {
      slice_shape_concat.push_back(
          full<T>({dim_size - 2}, 1, x_shape_tensor.dtype()));
    }
    auto slice_shape_tensor = concat<T>(slice_shape_concat, 0);

    if (scale) {
      auto scale_cast = backend::reshape<T>(scale.get(), slice_shape_tensor);
      scale_cast = ConvertToMT<T>(scale_cast);
      out = out * scale_cast;
    }

    if (bias) {
      auto bias_cast = backend::reshape<T>(bias.get(), slice_shape_tensor);
      bias_cast = ConvertToMT<T>(bias_cast);
      out = out + bias_cast;
    }

    std::vector<int64_t> res_shape(1, -1);
    auto mean_out = reshape<T>(mean_, res_shape);
    auto variance_out = reshape<T>(rsqrt_var, res_shape);
    auto res = ConvertToOrig<T>(out, org_dtype);

    return std::make_tuple(res, mean_out, variance_out);
  }

  // static shape
  const int64_t N = x_dims[0];
  const int64_t C = x_dims[1];
  const int64_t NxC = N * C;
  const int64_t sample_size = x.numel() / N / C;
  const std::vector<int64_t> shape{NxC, sample_size};
  const std::vector<int64_t> rdims{1};

  auto x_arr = reshape<T>(x_cast, shape);
  auto mean_ = mean_decomp<T>(x_arr, rdims, true);
  auto difference = x_arr - mean_;
  auto variance = mean_decomp<T>(difference * difference, rdims, true);
  auto var_tmp3 = variance + epsilon;
  auto rsqrt_var = rsqrt<T>(var_tmp3);
  auto out = difference * rsqrt_var;

  std::vector<int64_t> slice_shape(x_dims.size(), 1);
  slice_shape[1] = x_dims[1];

  out = reshape<T>(out, x_dims);
  if (scale) {
    auto scale_cast = reshape<T>(scale.get(), slice_shape);
    scale_cast = ConvertToMT<T>(scale_cast);
    out = out * scale_cast;
  }

  if (bias) {
    auto bias_cast = reshape<T>(bias.get(), slice_shape);
    bias_cast = ConvertToMT<T>(bias_cast);
    out = out + bias_cast;
  }

  std::vector<int64_t> res_shape(1, -1);
  auto mean_out = reshape<T>(mean_, res_shape);
  auto variance_out = reshape<T>(rsqrt_var, res_shape);
  auto res = ConvertToOrig<T>(out, org_dtype);

  return std::make_tuple(res, mean_out, variance_out);
}

template <typename T>
Tensor flatten_decomp(const Tensor& x, int start_axis, int end_axis) {
  auto x_dim = x.shape();
  if (x_dim.size() == 0) {
    start_axis = 0;
    end_axis = 0;
  }
  if (start_axis < 0) {
    start_axis += x_dim.size();
  }

  if (end_axis < 0) {
    end_axis += x_dim.size();
  }

  if (end_axis < start_axis) {
    PADDLE_THROW(common::errors::Unimplemented(
        "end_axis must be greater than or equal to start_axis."));
  }

  if (has_dynamic_shape(x.shape())) {
    auto x_shape = shape64<T>(x);
    if (end_axis == start_axis) {
      return backend::reshape<T>(x, x_shape);
    }
    std::vector<Tensor> out_shape;

    for (size_t i = 0; i < x_dim.size();) {
      if (i == static_cast<size_t>(start_axis)) {
        Tensor flat = get_slice<T>(x_shape, i);

        for (auto t = start_axis + 1; t <= end_axis; ++t) {
          flat = flat * get_slice<T>(x_shape, t);
        }
        out_shape.push_back(flat);
        i = end_axis + 1;
      } else {
        out_shape.push_back(get_slice<T>(x_shape, i));
        i++;
      }
    }

    Tensor out_shape_tensor = concat<T>(out_shape);
    return backend::reshape<T>(x, out_shape_tensor);
  } else {
    std::vector<int64_t> tmp_shape(x_dim);
    tmp_shape.insert(tmp_shape.begin(), 0);
    if (x_dim.size() == 0) {
      std::vector<int64_t> res_shape(1, 1);
      return reshape<T>(x, res_shape);
    }
    if (end_axis == start_axis) {
      return reshape<T>(x, x_dim);
    }

    int slice_numel = 1;
    for (int i = start_axis; i <= end_axis; ++i) {
      slice_numel *= x_dim[i];
    }
    std::vector<int64_t> out_shape;
    for (int i = 0; i < start_axis; ++i) {
      out_shape.push_back(x_dim[i]);
    }
    out_shape.push_back(slice_numel);
    for (size_t i = end_axis + 1; i < x_dim.size(); ++i) {
      out_shape.push_back(x_dim[i]);
    }

    return reshape<T>(x, out_shape);
  }
}

template <typename T>
Tensor clip_decomp(const Tensor& x, const Tensor& min, const Tensor& max) {
  auto min_reshape = min;
  auto max_reshape = max;

  if (x.shape().size() == 0) {
    min_reshape = reshape<T>(min_reshape, {});
    max_reshape = reshape<T>(max_reshape, {});
  }

  if (has_dynamic_shape(x.shape())) {
    min_reshape = backend::expand<T>(min_reshape, shape64<T>(x));
    max_reshape = backend::expand<T>(max_reshape, shape64<T>(x));
  } else {
    min_reshape = expand<T>(min_reshape, x.shape());
    max_reshape = expand<T>(max_reshape, x.shape());
  }
  if (min_reshape.dtype() != x.dtype()) {
    min_reshape = cast<T>(min_reshape, x.dtype());
  }

  if (max_reshape.dtype() != x.dtype()) {
    max_reshape = cast<T>(max_reshape, x.dtype());
  }

  auto ans = maximum<T>(minimum<T>(x, max_reshape), min_reshape);
  return ans;
}

template <typename T>
Tensor index_select_decomp(const Tensor& x, const Tensor& index, int axis) {
  int axis_tmp = axis;
  if (axis < 0) {
    axis_tmp += x.dims().size();
  }

  return gather<T>(x, index, axis_tmp);
}

template <typename T>
std::tuple<Tensor, Tensor, Tensor> group_norm_decomp(
    const Tensor& x,
    const paddle::optional<Tensor>& scale,
    const paddle::optional<Tensor>& bias,
    const float epsilon,
    const int groups,
    const std::string& data_format) {
  size_t rank = x.shape().size();
  GroupNormDecompHelper<T> decomp_helper(x, scale, bias, groups, data_format);
  const std::vector<int64_t>& c_axis = decomp_helper.GetReduceAxis();
  const std::vector<int64_t>& scale_bias_new_shape =
      decomp_helper.GetScaleBiasNewShape();

  if (rank < 3) {
    PADDLE_THROW(common::errors::Unimplemented(
        "Only support NCHW and NHWC format in rank higher or equal to 3. "
        "Current rank: %zu",
        rank));
  }

  auto org_dtype = x.dtype();
  Tensor x_cast = ConvertToMT<T>(x);

  auto x_dim = x_cast.shape();
  x_cast = decomp_helper.Split(x_cast);

  auto mean_ = mean_decomp<T>(x_cast, c_axis, true);
  auto var_tmp_ = mean_decomp<T>(x_cast * x_cast, c_axis, true) - mean_ * mean_;
  auto var_ = maximum<T>(var_tmp_, full<T>({}, 0, var_tmp_.dtype()));
  auto var_inv = rsqrt<T>(var_ + full_scalar<T>(epsilon, var_.dtype()));
  auto out = (x_cast - mean_) * var_inv;

  Tensor scale_cast;
  if (scale) {
    scale_cast = reshape<T>(scale.get(), scale_bias_new_shape);
    scale_cast = ConvertToMT<T>(scale_cast);
    out = out * scale_cast;
  }
  Tensor bias_cast;
  if (bias) {
    bias_cast = reshape<T>(bias.get(), scale_bias_new_shape);
    bias_cast = ConvertToMT<T>(bias_cast);
    out = out + bias_cast;
  }

  Tensor mean_out = squeeze<T>(mean_, c_axis);
  Tensor var_out = squeeze<T>(var_, c_axis);
  out = decomp_helper.Merge(out);
  out = ConvertToMT<T>(out, org_dtype);

  return std::make_tuple(out, mean_out, var_out);
}

template <typename T>
Tensor square_decomp(const Tensor& x) {
  auto x_cast = ConvertToMT<T>(x);

  Tensor two;
  two = full_scalar<T>(2, x_cast.dtype(), x_cast.place());

  auto ans = elementwise_pow<T>(x_cast, two);
  return ConvertToOrig<T>(ans, x.dtype());
}

template <typename T>
Tensor sigmoid_cross_entropy_with_logits_decomp(
    const Tensor& x,
    const Tensor& label,
    const paddle::optional<Tensor>& pos_weight,
    bool normalize,
    int ignore_index) {
  const Tensor zero = full_like_decomp<T>(x, 0, x.type(), x.place());
  const Tensor one = full_like_decomp<T>(x, 1, x.type(), x.place());
  Tensor pos_weight_tensor;
  Tensor tmp_out;
  if (pos_weight) {
    pos_weight_tensor = pos_weight.get();
    auto max_val = where<T>(x < zero, -x, zero);
    auto term1 = (one - label) * x;
    auto term2 = log<T>(exp<T>(-max_val) + exp<T>(-x - max_val));
    tmp_out = term1 + pos_weight_tensor * (term2 + max_val);
  } else {
    auto term1 = where<T>(x > zero, x, zero);
    auto term2 = x * label;
    auto term3 = log<T>(one + exp<T>(-abs<T>(x)));
    tmp_out = term1 - term2 + term3;
  }
  const Tensor ignore_index_tensor =
      full_like_decomp<T>(x, ignore_index, label.type(), label.place());
  auto out = where<T>(label == ignore_index_tensor, zero, tmp_out);
  if (normalize) {
    // Follow the implementation in
    // paddle/phi/kernels/cpu/sigmoid_cross_entropy_with_logits_kernel.cc
    const Tensor eps1 = full_like_decomp<T>(x, 1e-6, x.type(), x.place());
    auto diff = label - ignore_index_tensor;
    const Tensor tmp_norm = sum<T>(where<T>(abs<T>(diff) > eps1, one, zero));
    // Follow the implementation in
    // paddle/phi/kernels/cpu/sigmoid_cross_entropy_with_logits_kernel.cc
    const Tensor eps2 = full_scalar<T>(1e-5, x.type(), x.place());
    auto norm = where<T>(tmp_norm > eps2, tmp_norm, eps2);
    out = out / norm;
  }
  return out;
}

template <typename T>
Tensor mean_all_decomp(const Tensor& x) {
  auto x_cast = ConvertToMT<T>(x);
  auto x_shape = x.shape();

  Tensor ans;
  if (has_dynamic_shape(x_shape)) {
    Tensor x_shape_tensor = shape64<T>(x_cast);
    Tensor value = get_slice<T>(x_shape_tensor, 0);
    for (size_t i = 1; i < x_shape.size(); i++) {
      value = value * get_slice<T>(x_shape_tensor, i);
    }
    value = reshape<T>(value, {});
    ans = sum<T>(x_cast) / cast<T>(value, x_cast.dtype());
  } else {
    ans = sum<T>(x_cast) / x_cast.numel();
  }

  return ConvertToOrig<T>(ans, x.dtype());
}

template <typename T>
Tensor embedding_decomp(const Tensor& x,
                        const Tensor& weight,
                        const int64_t padding_idx,
                        const bool sparse) {
  if (weight.dims().size() != 2) {
    PADDLE_THROW(
        common::errors::Unimplemented("Only support weight with 2-D."));
  }

  const int64_t NoPadding = -1;
  Tensor weight_tmp = weight;
  Tensor res;
  if (has_dynamic_shape(x.shape())) {
    if (padding_idx != NoPadding) {
      Tensor put_shape = shape64<T>(sum<T>(weight, {0}, weight.dtype(), true));
      Tensor padding_idx_tensor =
          backend::full_with_tensor<T>(put_shape, padding_idx, DataType::INT64);
      Tensor zeros =
          backend::full_with_tensor<T>(put_shape, 0.0, weight.dtype());
      weight_tmp = put_along_axis<T>(weight, padding_idx_tensor, zeros, 0);
    }

    if (x.dims().size() <= 1) {
      res = gather<T>(weight_tmp, x);
      if (x.dims().size() == 0) {
        res = squeeze<T>(res, {0});
      }
    } else {
      std::vector<int64_t> tar_shape{-1};
      auto x_reshape = reshape<T>(x, tar_shape);
      auto out = gather<T>(weight_tmp, x_reshape);
      auto x_t_shape = shape64<T>(x);
      auto token_dim = get_slice<T>(shape64<T>(out), 1);
      auto res_t_shape = concat<T>({x_t_shape, token_dim}, 0);
      res = backend::reshape<T>(out, res_t_shape);
    }
  } else {
    if (padding_idx != NoPadding) {
      std::vector<int64_t> put_shape{1, weight.dims()[1]};
      Tensor padding_idx_tensor =
          full<T>(put_shape, padding_idx, DataType::INT64, x.place());
      Tensor zeros = full<T>(put_shape, 0.0, weight.dtype(), weight.place());
      weight_tmp = put_along_axis<T>(weight, padding_idx_tensor, zeros, 0);
    }

    if (x.dims().size() <= 1) {
      res = gather<T>(weight_tmp, x);
      if (x.dims().size() == 0) {
        res = squeeze_decomp<T>(res, {0});
      }
    } else {
      std::vector<int64_t> tar_shape{-1};
      auto x_reshape = reshape<T>(x, tar_shape);
      auto out = gather<T>(weight_tmp, x_reshape);

      auto res_dims = x.shape();
      res_dims.push_back(-1);
      res = reshape<T>(out, res_dims);
    }
  }
  if (res.dtype() != weight.dtype()) {
    res = cast<T>(res, weight.dtype());
  }
  return res;
}

template <typename T>
Tensor index_sample_decomp(const Tensor& x, const Tensor& index) {
  std::vector<int64_t> tmp_shape{-1, 1};
  auto index_dim = get_slice<T>(shape64<T>(index), 0);
  auto start = full<T>({1}, 0, index_dim.dtype());
  auto step = full<T>({1}, 1, index_dim.dtype());
  auto arange_tmp = reshape<T>(
      backend::arange<T>(start, index_dim, step, index.dtype(), index.place()),
      tmp_shape);

  auto index_res =
      reshape<T>(backend::expand<T>(arange_tmp, shape64<T>(index)), tmp_shape);
  auto index_ = reshape<T>(index, tmp_shape);
  auto concat_res = concat<T>({index_res, index_}, 1);
  auto res =
      backend::reshape<T>(gather_nd<T>(x, concat_res), shape64<T>(index));

  if (res.dtype() != x.dtype()) {
    return cast<T>(res, x.dtype());
  } else {
    return res;
  }
}

template <typename T>
Tensor elu_decomp(const Tensor& x, const float alpha) {
  auto x_cast = ConvertToMT<T>(x);

  Tensor zero;
  Tensor tmp_res;

  if (has_dynamic_shape(x_cast.shape())) {
    zero = backend::full_with_tensor<T>(shape64<T>(x_cast), 0, x_cast.dtype());
    tmp_res =
        full_scalar<T>(alpha, x_cast.dtype(), x_cast.place()) *
        (exp<T>(x_cast) - full_scalar<T>(1, x_cast.dtype(), x_cast.place()));
  } else {
    zero = full<T>(x_cast.shape(), 0, x_cast.type(), x_cast.place());
    tmp_res = alpha * (exp<T>(x_cast) - 1);
  }
  auto ans = where<T>(x_cast > zero, x_cast, tmp_res);
  return ConvertToOrig<T>(ans, x.dtype());
}

template <typename T>
Tensor lerp_decomp(const Tensor& x, const Tensor& y, const Tensor& weight) {
  Tensor x_cast = ConvertToMT<T>(x);
  Tensor y_cast = ConvertToMT<T>(y);
  Tensor weight_cast = ConvertToMT<T>(weight);
  Tensor res = x_cast + weight_cast * (y_cast - x_cast);
  return ConvertToOrig<T>(res, x.dtype());
}

template <typename T>
Tensor log_loss_decomp(const Tensor& input,
                       const Tensor& label,
                       float epsilon) {
  Tensor ones = full_scalar<T>(1.0, input.dtype(), input.place());
  Tensor eps = full_scalar<T>(epsilon, input.dtype(), input.place());
  Tensor term1 = -label * log<T>(input + eps);
  Tensor term2 = (ones - label) * log<T>(ones - input + eps);
  return term1 - term2;
}

template <typename T>
Tensor kldiv_loss_decomp(const Tensor& x,
                         const Tensor& label,
                         const std::string& reduction,
                         bool log_target) {
  bool dynamic_shape = has_dynamic_shape(x.shape());
  Tensor loss;
  if (log_target) {
    loss = exp<T>(label) * (label - x);
  } else {
    Tensor output = label * (log<T>(label) - x);
    Tensor zero = full_scalar<T>(0.0, label.dtype(), label.place());
    Tensor zeros;
    if (dynamic_shape) {
      zeros = backend::full_with_tensor<T>(shape64<T>(x), 0, x.dtype());
    } else {
      zeros = full<T>(x.shape(), 0, x.dtype(), x.place());
    }
    loss = where<T>(label > zero, output, zeros);
  }

  if (reduction == "batchmean") {
    if (x.shape().size() > 0) {
      if (dynamic_shape) {
        return sum<T>(loss) / get_slice<T>(shape64<T>(x), 0);
      } else {
        return sum<T>(loss) / x.shape()[0];
      }
    } else {
      return sum<T>(loss);
    }
  }
  if (reduction == "mean") {
    return mean_decomp<T>(loss, {}, false);
  }
  if (reduction == "sum") {
    return sum<T>(loss);
  }
  return loss;
}

template <typename T>
Tensor softsign_decomp(const Tensor& x) {
  // softsign = x / (1 + abs(x))

  Tensor x_abs = abs<T>(x);
  Tensor one = full_scalar<T>(1.0, x.dtype(), x.place());
  return x / (one + x_abs);
}

template <typename T>
std::vector<Tensor> unstack_decomp(const Tensor& x, int axis, const int num) {
  if (axis < 0) {
    axis += x.dims().size();
  }
  std::vector<int64_t> x_shape = x.shape();
  if (x_shape[axis] < 0) {
    PADDLE_THROW(
        common::errors::Unimplemented("unstack axis must not be dynamic."));
  }
  PADDLE_ENFORCE_EQ(
      num,
      x_shape[axis],
      common::errors::InvalidArgument(
          "The number of unstacks should be equal to the value of "
          "x.shape[axis], but received num is %d and x.shape[axis] is %d.",
          num,
          x_shape[axis]));

  std::vector<int> sections(num, 1);
  std::vector<Tensor> res = backend::split<T>(x, sections, axis);
  if (has_dynamic_shape(x_shape)) {
    const Tensor x_shape_tensor = shape64<T>(x);

    // find new shape of each tensor.
    std::vector<Tensor> new_shape_vec;
    for (size_t i = 0; i < x_shape.size(); ++i) {
      if (static_cast<int>(i) != axis) {
        new_shape_vec.push_back(get_slice<T>(x_shape_tensor, i));
      }
    }
    const Tensor new_shape = concat<T>(new_shape_vec);
    std::transform(res.begin(), res.end(), res.begin(), [&](Tensor& x) {
      return backend::reshape<T>(x, new_shape);
    });
  } else {
    std::vector<int64_t> new_shape;
    // find new shape of each tensor.
    for (size_t i = 0; i < x_shape.size(); ++i) {
      if (static_cast<int>(i) != axis) {
        new_shape.push_back(x_shape[i]);
      }
    }
    std::transform(res.begin(), res.end(), res.begin(), [&](Tensor& x) {
      return reshape<T>(x, new_shape);
    });
  }
  return res;
}

template <typename T>
Tensor numel_decomp(const Tensor& x) {
  auto x_shape = x.shape();
  if (has_dynamic_shape(x_shape)) {
    const Tensor x_shape_tensor = shape64<T>(x);
    Tensor value = full<T>({1}, 1, x_shape_tensor.dtype());
    for (size_t i = 0; i < x_shape.size(); ++i) {
      value = value * get_slice<T>(x_shape_tensor, i);
    }
    return cast<T>(reshape<T>(value, {}), DataType::INT64);
  } else {
    return full_scalar<T>(x.numel(), DataType::INT64, x.place());
  }
}

template <typename T>
Tensor swish_decomp(const Tensor& x) {
  return x * sigmoid<T>(x);
}

template <typename T>
Tensor addmm_decomp(const Tensor& input,
                    const Tensor& x,
                    const Tensor& y,
                    const float beta,
                    const float alpha) {
  Tensor x_y_mat = matmul<T>(x, y);
  return full_scalar<T>(alpha, x_y_mat.dtype()) * x_y_mat +
         full_scalar<T>(beta, input.dtype()) * input;
}

template <typename T>
Tensor eye_decomp(const paddle::Scalar& num_rows,
                  const paddle::Scalar& num_columns,
                  const DataType dtype,
                  const Place& place) {
  int32_t min_num = std::min(num_rows.to<int>(), num_columns.to<int>());
  Tensor zero_tensor =
      full<T>({num_rows.to<int>(), num_columns.to<int>()}, 0, dtype, place);
  auto zero_tensor_cast = ConvertToMT<T>(zero_tensor);
  Tensor diag_one = unsqueeze<T>(full<T>({min_num}, 1, dtype, place), {1});
  auto diag_one_cast = ConvertToMT<T>(diag_one);

  auto start = full<T>({1}, 0, dtype, place);
  auto stop = full<T>({1}, min_num, dtype, place);
  auto step = full<T>({1}, 1, dtype, place);
  Tensor index = unsqueeze<T>(
      backend::arange<T>(start, stop, step, DataType::INT32, place), {1});

  auto index_cast = ConvertToMT<T>(index);
  Tensor res = put_along_axis<T>(zero_tensor_cast, index, diag_one_cast, 1);

  return ConvertToOrig<T>(res, dtype);
}

template <typename T>
Tensor diag_decomp(const Tensor& x,
                   const int& offset = 0,
                   const float& padding_value = 0.0) {
  Tensor cast_x = ConvertToMT<T>(x);
  int64_t rank = cast_x.dims().size();
  Tensor res;
  if (rank == 1) {
    std::vector<int64_t> x_dims = cast_x.shape();
    int64_t n = x_dims[0];
    int64_t abs_offset = std::abs(offset);
    int64_t m = n + abs_offset;

    Tensor result =
        full<T>({m, m}, padding_value, cast_x.dtype(), cast_x.place());
    Tensor insert_value = cast_x;
    Tensor indices = backend::arange<T>(
        abs_offset, abs_offset + n, 1, DataType::INT64, cast_x.place());
    if (offset >= 0) {
      insert_value = reshape<T>(insert_value, {n, 1});
      indices = reshape<T>(indices, {n, 1});
      res = put_along_axis<T>(result, indices, insert_value, 1);
    } else {
      insert_value = reshape<T>(insert_value, {1, n});
      indices = reshape<T>(indices, {1, n});
      res = put_along_axis<T>(result, indices, insert_value, 0);
    }
  } else {
    // This is the case for 2D tensor.
    std::vector<int64_t> x_dims = cast_x.shape();
    int64_t n = x_dims[0];
    int64_t m = x_dims[1];
    if (offset <= -n || offset >= m) {
      return res;
    }
    Tensor x_flat = reshape<T>(cast_x, {n * m});
    int64_t start = offset >= 0 ? offset : -offset * m;
    int64_t num =
        offset >= 0 ? std::min(n, m - offset) : std::min(n + offset, m);
    int64_t stride = m + 1;
    int64_t end = start + num * stride;

    Tensor indices =
        backend::arange<T>(start, end, stride, DataType::INT64, cast_x.place());
    res = take_along_axis<T>(x_flat, indices, 0);
  }
  return ConvertToOrig<T>(res, x.dtype());
}

}  // namespace details

}  // namespace primitive
}  // namespace paddle
