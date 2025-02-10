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

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

#include <math.h>
#include <vector>
#include "paddle/common/ddim.h"
#include "paddle/fluid/prim/api/generated_prim/prim_generated_api.h"
#include "paddle/fluid/primitive/base/lazy_tensor.h"
#include "paddle/fluid/primitive/decomp_utils/decomp_utils.h"
#include "paddle/phi/common/amp_type_traits.h"

namespace paddle {
namespace primitive {
namespace details {

template <typename T>
void abs_grad(const Tensor& x, const Tensor& out_grad, Tensor* x_grad) {
  if (x_grad) {
    auto sign_tmp = sign<T>(x);
    set_output<T>(out_grad * sign_tmp, x_grad);
  }
}

template <typename T>
void assign_grad(const Tensor& out_grad, Tensor* x_grad) {
  if (x_grad) {
    by_pass<T>(out_grad, x_grad);
  }
}

template <typename T>
void bce_loss_grad(const Tensor& input,
                   const Tensor& label,
                   const Tensor& out_grad,
                   Tensor* input_grad) {
  using MT = typename phi::dtype::MPTypeTrait<T>::Type;
  if (input_grad) {
    auto input_mt = ConvertToMT<MT>(input);
    auto term = maximum<MT>((1 - input_mt) * input_mt,
                            full_scalar<MT>(1e-12, input_mt.dtype()));
    auto out_base =
        ConvertToMT<MT>(out_grad) * (input_mt - ConvertToMT<MT>(label)) / term;
    set_output<T>(ConvertToOrig<T>(out_base, input.dtype()), input_grad);
  }
}

template <typename T>
void cumsum_grad(const Tensor& x,
                 const Tensor& out_grad,
                 const Scalar& axis,
                 bool flatten,
                 bool exclusive,
                 bool reverse,
                 Tensor* x_grad) {
  if (x_grad) {
    auto grad = cumsum<T>(out_grad, axis, flatten, exclusive, !reverse);
    if (has_dynamic_shape(x.shape())) {
      grad = backend::reshape<T>(grad, shape64<T>(x));
    } else {
      grad = reshape<T>(grad, x.shape());
    }
    set_output<T>(grad, x_grad);
  }
}

template <typename T>
void cumprod_grad(const Tensor& x,
                  const Tensor& out,
                  const Tensor& out_grad,
                  int dim,
                  bool exclusive,
                  bool reverse,
                  Tensor* x_grad) {
  if (x_grad) {
    // dx = cumsum(out * out_grad, dim, false, exclusive, !reverse) / x
    Tensor zero_tensor, ones_tensor;
    if (has_dynamic_shape(x.shape())) {
      zero_tensor = backend::full_with_tensor<T>(
          shape64<T>(x), 0.0, x.dtype(), x.place());
      ones_tensor = backend::full_with_tensor<T>(
          shape64<T>(x), 1.0, x.dtype(), x.place());
    } else {
      zero_tensor = full<T>(x.shape(), 0.0, x.dtype(), x.place());
      ones_tensor = full<T>(x.shape(), 1.0, x.dtype(), x.place());
    }
    auto zero_mask = cast<T>(equal<T>(x, zero_tensor), x.dtype());
    // determine the index of first zero
    auto zero_mask_cumsum_exclusive =
        cumsum<T>(zero_mask, dim, false, true, reverse);
    auto zero_mask_cumsum = scale<T>(zero_mask_cumsum_exclusive, 2) + zero_mask;

    auto first_zero_mask =
        cast<T>(equal<T>(zero_mask_cumsum, ones_tensor), x.dtype());
    // compute the grad for position with value not equal to 0
    auto common_dx = cumsum<T>(out * out_grad, dim, false, exclusive, !reverse);
    // fill the positions of 0 with 1.
    auto replace_one = (ones_tensor - zero_mask) * x + zero_mask;
    // fill the first positions of 0 with 1.
    auto replace_first_one =
        (ones_tensor - first_zero_mask) * x + first_zero_mask;
    // recompute the grad of the first position with 0
    auto cumprod_recompute =
        cumprod<T>(replace_first_one, dim, exclusive, reverse);
    auto zeros_dx = cumsum<T>(
        cumprod_recompute * out_grad, dim, false, exclusive, !reverse);
    auto x_grad_res = ((ones_tensor - first_zero_mask) * common_dx +
                       first_zero_mask * zeros_dx) /
                      replace_one;
    set_output<T>(x_grad_res, x_grad);
  }
}

template <typename T>
void divide_grad(const Tensor& x,
                 const Tensor& y,
                 const Tensor& out,
                 const Tensor& out_grad,
                 int axis,
                 Tensor* dx,
                 Tensor* dy) {
  if (dy) {
    // dy = -(x/y^2) * dout
    auto dy_res = -out_grad * (x / y / y);
    if (has_dynamic_shape(y.shape()) || has_dynamic_shape(out_grad.shape()) ||
        out_grad.dims() != y.dims()) {
      auto dy_tmp = reduce_as<T>(dy_res, y);
      set_output<T>(dy_tmp, dy);
    } else {
      set_output<T>(dy_res, dy);
    }
  }  // indicate we will compute dy
  if (dx) {
    // dx = (1/y) * dout
    auto dx_res = out_grad / y;
    if (has_dynamic_shape(x.shape()) || has_dynamic_shape(out_grad.shape()) ||
        out_grad.dims() != x.dims()) {
      auto dx_tmp = reduce_as<T>(dx_res, x);
      set_output<T>(dx_tmp, dx);
    } else {
      set_output<T>(dx_res, dx);
    }
  }  // indicate we will compute dx
}

template <typename T>
void floor_grad(const Tensor& out_grad, Tensor* x_grad) {
  if (x_grad) {
    Tensor zero_tensor;
    if (has_dynamic_shape(out_grad.shape())) {
      zero_tensor = backend::full_with_tensor<T>(
          shape64<T>(out_grad), 0.0, out_grad.dtype(), out_grad.place());
    } else {
      zero_tensor = full<T>(common::vectorize(out_grad.dims()),
                            0.0,
                            out_grad.dtype(),
                            out_grad.place());
    }
    set_output<T>(zero_tensor, x_grad);
  }
}

template <typename T>
void sum_grad(const Tensor& x,
              const Tensor& out_grad,
              const IntArray& axis,
              bool keepdim,
              bool reduce_all,
              Tensor* x_grad) {
  if (!x_grad) {
    return;
  }

  int64_t axis_size = axis.size();
  int64_t x_dim_size = x.dims().size();
  auto x_grad_tmp = Tensor();
  reduce_all = false;
  if (reduce_all || axis_size == 0 || axis_size == x_dim_size) {
    reduce_all = true;
  } else {
    reduce_all = false;
  }
  if (has_dynamic_shape(x.shape())) {
    Tensor x_shape = shape64<T>(x);
    if (x_dim_size == 1) {
      x_grad_tmp = backend::expand<T>(out_grad, x_shape);
    } else {
      if (!keepdim) {
        auto axis_ = std::vector<int64_t>();
        if (reduce_all) {
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
        Tensor out_grad_shape = shape64<T>(out_grad);
        size_t total_shape_size = out_grad.shape().size() + axis_.size();
        std::vector<Tensor> result_shape;
        size_t j = 0, k = 0;
        Tensor ones = full<T>({1}, 1, x_shape.dtype(), x.place());
        for (size_t i = 0; i < total_shape_size; i++) {
          if (j < axis_.size() && axis_[j] == int64_t(i)) {
            result_shape.push_back(ones);
            j++;
          } else {
            result_shape.push_back(get_slice<T>(out_grad_shape, int64_t(k)));
            k++;
          }
        }
        auto out_grad_ = backend::reshape<T>(out_grad, concat<T>(result_shape));
        x_grad_tmp = backend::expand<T>(out_grad_, x_shape);
      } else {
        x_grad_tmp = backend::expand<T>(out_grad, x_shape);
      }
    }
  } else {
    std::vector<int64_t> x_dim = common::vectorize<int64_t>(x.dims());
    if (x_dim_size == 1) {
      x_grad_tmp = expand<T>(out_grad, IntArray(x_dim));
    } else {
      if (!keepdim) {
        auto axis_ = std::vector<int64_t>();
        if (reduce_all) {
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
        auto out_grad_shape = get_unsqueeze_dims(out_grad, axis_);
        auto out_grad_ = reshape<T>(out_grad, out_grad_shape);
        x_grad_tmp = expand<T>(out_grad_, IntArray(x_dim));
      } else {
        x_grad_tmp = expand<T>(out_grad, IntArray(x_dim));
      }
    }
  }
  set_output<T>(x_grad_tmp, x_grad);
}

template <typename T>
void mean_grad(const Tensor& x,
               const Tensor& out_grad,
               const IntArray& axis,
               bool keepdim,
               bool reduce_all,
               Tensor* x_grad) {
  if (!x_grad) {
    return;
  }
  Tensor x_grad_tmp;
  sum_grad<T>(x, out_grad, axis, keepdim, reduce_all, &x_grad_tmp);

  Tensor div_factor = [&] {
    Tensor factor_tensor;
    auto axis_data = axis.GetData();
    const std::vector<int64_t> x_dim = x.shape();
    if (axis.size() == 0) {
      for (size_t i = 0; i < x_dim.size(); ++i) {
        axis_data.push_back(i);
      }
    }

    for (int64_t& idx : axis_data) {
      if (idx < 0) {
        idx += x_dim.size();
      }
    }

    if (has_dynamic_shape(x_dim, axis_data)) {
      auto x_shape = shape64<T>(x);
      factor_tensor = full<T>({1}, 1.0, x_shape.dtype(), x_shape.place());
      for (int64_t idx : axis_data) {
        factor_tensor = factor_tensor * get_slice<T>(x_shape, idx);
      }
      factor_tensor = cast<T>(factor_tensor, x.dtype());
    } else {
      int64_t factor = 1;
      for (int64_t idx : axis_data) {
        factor *= x_dim[idx];
      }
      factor_tensor =
          full<T>(std::vector<int64_t>{}, factor, x.dtype(), x.place());
    }
    return factor_tensor;
  }();

  set_output<T>(x_grad_tmp / div_factor, x_grad);
}

template <typename T>
void gelu_grad(const Tensor& x,
               const Tensor& out_grad,
               bool approximate,
               Tensor* x_grad) {
  if (!x_grad) return;
  // Automatically promote to fp32 when the input type is fp16 for keeping
  // consistent with phi kernel

  auto promoted_x = ConvertToMT<T>(x);
  auto promoted_out_grad = ConvertToMT<T>(out_grad);
  if (approximate) {
    float kbeta = M_SQRT2 * M_2_SQRTPI * 0.5;
    float kkappa = 0.044715;
    Tensor kbeta_ = full_scalar<T>(kbeta, promoted_x.dtype());
    Tensor kkappa_ = full_scalar<T>(kkappa, promoted_x.dtype());

    auto x_sq = promoted_x * promoted_x;
    auto x_cube = x_sq * promoted_x;
    auto inner = kbeta_ * (promoted_x + kkappa_ * x_cube);
    auto tanh_inner = tanh<T>(inner);

    auto left = scale<T>(promoted_x, 0.5);
    auto right = scale<T>(tanh_inner, 1., 1.);

    auto left_derivative = scale<T>(right, 0.5);

    auto tanh_derivative = scale<T>(tanh_inner * tanh_inner, -1., 1.);
    auto inner_derivative = kbeta_ * (scale<T>(3 * kkappa_ * x_sq, 1., 1.));
    auto right_derivative = left * tanh_derivative * inner_derivative;

    set_output<T>(
        ConvertToOrig<T>(
            promoted_out_grad * (left_derivative + right_derivative), x.type()),
        x_grad);
  } else {
    float kalpha = M_SQRT1_2;
    float kbeta = M_2_SQRTPI * M_SQRT1_2 * 0.5;
    Tensor kalpha_ = full_scalar<T>(kalpha, promoted_x.dtype());
    Tensor kbeta_ = full_scalar<T>(kbeta, promoted_x.dtype());

    auto cdf = scale<T>(scale<T>(erf<T>(kalpha_ * promoted_x), 1., 1.), 0.5);
    auto pdf = kbeta_ * exp<T>(scale<T>(promoted_x * promoted_x, -0.5));
    set_output<T>(ConvertToOrig<T>(promoted_out_grad * (cdf + promoted_x * pdf),
                                   x.type()),
                  x_grad);
  }
}

template <typename T>
void reduce_as_grad(const Tensor& x,
                    const Tensor& target,
                    const Tensor& out_grad,
                    Tensor* x_grad) {
  if (!x_grad) {
    return;
  }
  if (has_dynamic_shape(x.shape()) || has_dynamic_shape(out_grad.shape())) {
    auto x_grad_tmp = backend::expand<T>(out_grad, shape64<T>(x));
    set_output<T>(x_grad_tmp, x_grad);
  } else {
    std::vector<int64_t> x_dim = common::vectorize<int64_t>(x.dims());
    std::vector<int64_t> axis = common::vectorize<int64_t>(
        get_reduce_dims_from_out(x.dims(), target.dims()));
    int64_t axis_size = axis.size();
    if (axis_size == 0) {
      by_pass<T>(out_grad, x_grad);
      return;
    }
    int64_t x_dim_size = x_dim.size();

    auto x_grad_tmp = Tensor();
    if (x_dim_size == 1) {
      x_grad_tmp = expand<T>(out_grad, IntArray(x_dim));
    } else {
      auto axis_ = std::vector<int64_t>();
      for (int64_t i = 0; i < axis_size; i++) {
        axis_.push_back(axis[i]);
        if (axis[i] < 0) {
          axis_[i] += x_dim_size;
        }
      }
      Tensor out_grad_ = out_grad;
      if (out_grad.shape().size() != x.shape().size()) {
        auto out_grad_shape = get_unsqueeze_dims(out_grad, axis_);
        out_grad_ = reshape<T>(out_grad, out_grad_shape);
      }
      x_grad_tmp = expand<T>(out_grad_, IntArray(x_dim));
    }

    set_output<T>(x_grad_tmp, x_grad);
  }
}

template <typename T>
void reshape_grad(const Tensor& x, const Tensor& grad_out, Tensor* grad_x) {
  if (grad_x) {
    Tensor grad_x_tmp;
    if (has_dynamic_shape(x.shape())) {
      grad_x_tmp = backend::reshape<T>(grad_out, shape64<T>(x));
    } else {
      const auto& x_dims = x.dims();
      grad_x_tmp = reshape<T>(grad_out, common::vectorize(x_dims));
    }
    set_output<T>(grad_x_tmp, grad_x);
  }
}

template <typename T>
void roll_grad(const Tensor& x,
               const Tensor& out_grad,
               const IntArray& shifts,
               const std::vector<int64_t>& axis,
               Tensor* x_grad) {
  if (x_grad) {
    auto shifts_ = shifts.GetData();
    int64_t nums = shifts_.size();
    for (int64_t i = 0; i < nums; i++) {
      shifts_[i] = 0 - shifts_[i];
    }
    auto x_grad_output = roll<T>(out_grad, shifts_, axis);
    set_output<T>(x_grad_output, x_grad);
  }
}

template <typename T>
void transpose_grad(const Tensor& grad_out,
                    const std::vector<int>& perm,
                    Tensor* grad_x) {
  if (grad_x) {
    std::vector<int> reverse_perm(perm);
    // make origin ranks
    for (int i = 0; i < static_cast<int>(perm.size()); ++i) {
      if (perm[i] >= 0) {
        reverse_perm[perm[i]] = i;
      } else {
        reverse_perm[perm[i] + perm.size()] = i;
      }
    }
    auto grad_x_tmp = transpose<T>(grad_out, reverse_perm);
    set_output<T>(grad_x_tmp, grad_x);
  }
}

template <typename T>
void scatter_grad(const Tensor& index,
                  const Tensor& updates,
                  const Tensor& out_grad,
                  bool overwrite,
                  Tensor* x_grad,
                  Tensor* updates_grad) {
  if (x_grad) {
    Tensor zero_tensor;
    if (has_dynamic_shape(updates.shape())) {
      zero_tensor = backend::full_with_tensor<T>(
          shape64<T>(updates), 0.0, updates.dtype(), updates.place());
    } else {
      zero_tensor = full<T>(common::vectorize(updates.dims()),
                            0.0,
                            updates.dtype(),
                            updates.place());
    }
    auto tmp_grad = scatter<T>(out_grad, index, zero_tensor, false);
    set_output<T>(tmp_grad, x_grad);
  }

  if (updates_grad) {
    Scalar tmp_zero = 0;
    auto tmp_updates_grad = gather<T>(out_grad, index, tmp_zero);

    // NOTE: len(index) can be smaller than len(updates) when updates is not a
    // scalar
    auto updates_dims = common::vectorize(updates.dims());
    auto index_dims = common::vectorize(index.dims());
    if (updates_dims.size() > 0 && updates_dims[0] > index_dims[0]) {
      // Pad zeros to the end of tmp_updates_grad to make its shape the same as
      // updates.
      decltype(updates_dims) padding_dims = updates_dims;
      padding_dims[0] = updates_dims[0] - index_dims[0];
      auto padding_zeros =
          full<T>(padding_dims, 0, updates.dtype(), updates.place());
      tmp_updates_grad =
          concat<T>({tmp_updates_grad, std::move(padding_zeros)}, 0);
    }
    set_output<T>(tmp_updates_grad, updates_grad);
  }
}

template <typename T>
void scatter_nd_add_grad(const Tensor& index,
                         const Tensor& updates,
                         const Tensor& out_grad,
                         Tensor* x_grad,
                         Tensor* updates_grad) {
  if (x_grad) {
    by_pass<T>(out_grad, x_grad);
  }
  if (updates_grad) {
    // Gradient by Gather: dUpdates = dO[Ids]
    auto tmp_updates_grad = gather_nd<T>(out_grad, index);
    set_output<T>(tmp_updates_grad, updates_grad);
  }
}

template <typename T>
void sin_grad(const Tensor& x, const Tensor& out_grad, Tensor* x_grad) {
  auto x_grad_tmp = cos<T>(x) * out_grad;
  set_output<T>(x_grad_tmp, x_grad);
}

template <typename T>
void cos_grad(const Tensor& x, const Tensor& out_grad, Tensor* x_grad) {
  auto x_grad_tmp = -sin<T>(x) * out_grad;
  set_output<T>(x_grad_tmp, x_grad);
}

template <typename T>
void tanh_grad(const Tensor& out, const Tensor& grad_out, Tensor* grad_x) {
  if (!grad_x) return;
  auto grad_x_tmp = grad_out * (full_scalar<T>(1.0, out.dtype()) - out * out);
  set_output<T>(grad_x_tmp, grad_x);
}

template <typename T>
void concat_grad(const std::vector<Tensor>& x,
                 const Tensor& out_grad,
                 const Scalar& axis,
                 std::vector<Tensor*> x_grad) {
  int axis_value = axis.to<int>();
  int rank = x[0].dims().size();
  if (axis_value < 0) {
    axis_value = axis_value + rank;
  }
  axis_value = axis_value > 0 ? axis_value : 0;

  int x_num = x.size();
  std::vector<Tensor> x_grad_tmp;

  int neg_num = 0;
  for (size_t i = 0; i < x.size(); i++) {
    if (x[i].dims()[axis_value] < 0) {
      neg_num++;
    }
  }

  if (neg_num > 1) {
    std::vector<Tensor> sections;
    for (int i = 0; i < x_num; i++) {
      sections.push_back(get_slice<T>(shape64<T>(x[i]), int64_t(axis_value)));
    }
    Tensor sections_tensor = concat<T>(sections);
    x_grad_tmp = backend::split<T>(
        out_grad,
        sections_tensor,
        full<T>(
            {1}, axis_value, sections_tensor.dtype(), sections_tensor.place()));
  } else {
    std::vector<int> sections;
    for (int i = 0; i < x_num; ++i) {
      sections.push_back(x[i].dims()[axis_value]);
    }
    x_grad_tmp = split<T>(out_grad, IntArray(sections), axis_value);
  }
  for (int i = 0; i < x_num; ++i) {
    if (x_grad[i]) {
      set_output<T>(x_grad_tmp.at(i), x_grad.at(i));
    }
  }
}

template <typename T>
void split_grad(const std::vector<Tensor>& out_grad,
                const Scalar& axis,
                Tensor* x_grad) {
  if (x_grad) {
    auto grad = concat<T>(out_grad, axis);
    set_output<T>(grad, x_grad);
  }
}

template <typename T>
void cast_grad(const Tensor& x, const Tensor& out_grad, Tensor* x_grad) {
  if (x_grad) {
    auto res = cast<T>(out_grad, x.dtype());
    set_output<T>(res, x_grad);
  }
}

template <typename T>
void add_grad(const Tensor& x,
              const Tensor& y,
              const Tensor& out_grad,
              int axis,
              Tensor* dx,
              Tensor* dy) {
  if (dy) {
    if (has_dynamic_shape(y.shape()) || has_dynamic_shape(out_grad.shape()) ||
        out_grad.dims() != y.dims()) {
      auto dy_tmp = reduce_as<T>(out_grad, y);
      set_output<T>(dy_tmp, dy);
    } else {
      by_pass<T>(out_grad, dy);
    }
  }

  if (dx) {
    if (has_dynamic_shape(x.shape()) || has_dynamic_shape(out_grad.shape()) ||
        out_grad.dims() != x.dims()) {
      auto dx_tmp = reduce_as<T>(out_grad, x);
      set_output<T>(dx_tmp, dx);
    } else {
      by_pass<T>(out_grad, dx);
    }
  }
}

template <typename T>
void subtract_grad(const Tensor& x,
                   const Tensor& y,
                   const Tensor& out_grad,
                   int axis,
                   Tensor* dx,
                   Tensor* dy) {
  if (dy) {
    auto scale_out_grad = scale<T>(out_grad, -1.0, 0.0, true);
    if (has_dynamic_shape(y.shape()) || has_dynamic_shape(out_grad.shape()) ||
        out_grad.dims() != y.dims()) {
      auto dy_tmp = reduce_as<T>(scale_out_grad, y);
      set_output<T>(dy_tmp, dy);
    } else {
      set_output<T>(scale_out_grad, dy);
    }
  }
  if (dx) {
    if (has_dynamic_shape(x.shape()) || has_dynamic_shape(out_grad.shape()) ||
        out_grad.dims() != x.dims()) {
      auto dx_tmp = reduce_as<T>(out_grad, x);
      set_output<T>(dx_tmp, dx);
    } else {
      by_pass<T>(out_grad, dx);
    }
  }
}

template <typename T>
void multiply_grad(const Tensor& x,
                   const Tensor& y,
                   const Tensor& out_grad,
                   int axis,
                   Tensor* x_grad,
                   Tensor* y_grad) {
  if (x_grad) {
    auto x_grad_unreduce = out_grad * y;
    if (has_dynamic_shape(x.shape()) ||
        has_dynamic_shape(x_grad_unreduce.shape()) ||
        x_grad_unreduce.dims() != x.dims()) {
      auto x_grad_reduced = reduce_as<T>(x_grad_unreduce, x);
      set_output<T>(x_grad_reduced, x_grad);
    } else {
      set_output<T>(x_grad_unreduce, x_grad);
    }
  }
  if (y_grad) {
    auto y_grad_unreduce = out_grad * x;
    if (has_dynamic_shape(y.shape()) ||
        has_dynamic_shape(y_grad_unreduce.shape()) ||
        y_grad_unreduce.dims() != y.dims()) {
      auto y_grad_reduced = reduce_as<T>(y_grad_unreduce, y);
      set_output<T>(y_grad_reduced, y_grad);
    } else {
      set_output<T>(y_grad_unreduce, y_grad);
    }
  }
}

template <typename T>
void elementwise_pow_grad(const Tensor& x,
                          const Tensor& y,
                          const Tensor& out_grad,
                          Tensor* dx,
                          Tensor* dy) {
  if (dy) {
    // dy = lnx * x^y
    auto lnx = log<T>(x);
    auto x_pow_y = elementwise_pow<T>(x, y);
    auto dy_res = lnx * x_pow_y * out_grad;
    if (has_dynamic_shape(out_grad.shape()) || has_dynamic_shape(y.shape()) ||
        out_grad.dims() != y.dims()) {
      auto dy_reduce_res = reduce_as<T>(dy_res, y);
      set_output<T>(dy_reduce_res, dy);
    } else {
      set_output<T>(dy_res, dy);
    }
  }  // indicate we will compute dy
  if (dx) {
    // dx = y * x^(y-1)
    if (has_dynamic_shape(out_grad.shape()) || has_dynamic_shape(x.shape())) {
      Tensor one_tensor = full_scalar<T>(1.0, y.dtype());
      Tensor x_pow_z = elementwise_pow<T>(x, y - one_tensor);
      Tensor dx_res = y * x_pow_z * out_grad;
      auto dx_reduce_res = reduce_as<T>(dx_res, x);
      set_output<T>(dx_reduce_res, dx);
    } else {
      auto tmp_z = y - 1.0;
      auto x_pow_z = elementwise_pow<T>(x, tmp_z);
      auto dx_res = y * x_pow_z * out_grad;
      if (out_grad.dims() != x.dims()) {
        auto dx_reduce_res = reduce_as<T>(dx_res, x);
        set_output<T>(dx_reduce_res, dx);
      } else {
        set_output<T>(dx_res, dx);
      }
    }
  }  // indicate we will compute dx
}

template <typename T>
void pow_grad(const Tensor& x,
              const Tensor& out_grad,
              const Scalar& y,
              Tensor* x_grad) {
  if (x_grad) {
    if (!y.FromTensor()) {
      float pow_val = y.to<float>();

      if (pow_val == 1.0f) {
        set_output<T>(out_grad, x_grad);
      } else {
        auto dx_res =
            x.pow(pow_val - 1) * full_scalar<T>(pow_val, x.dtype()) * out_grad;
        set_output<T>(dx_res, x_grad);
      }
    } else {
      Tensor one_tensor = full_scalar<T>(1.0, x.dtype());
      auto dx_res = y * elementwise_pow<T>(x, y - one_tensor) * out_grad;
      set_output<T>(dx_res, x_grad);
    }
  }
}

template <typename T>
void scale_grad(const Tensor& out_grad, const Scalar& scale, Tensor* x_grad) {
  if (x_grad) {
    auto dx_res = primitive::scale<T>(
        out_grad, scale, /*bias=*/0.0f, /*bias_after_scale=*/true);
    set_output<T>(dx_res, x_grad);
  }
}

template <typename T>
void stack_grad(const std::vector<Tensor>& x,
                const Tensor& out_grad,
                int axis,
                std::vector<Tensor*> x_grad) {
  // use rank of **stacked** tensor as len of axes
  int out_rank = out_grad.dims().size();  // len(x[0].shape)

  // ensure axis >= 0
  if (axis < 0) {
    axis = ((axis % out_rank) + out_rank) % out_rank;
  }

  // split out_grad to grads for each input tensor
  int x_num = x.size();
  std::vector<int> sections(x_num, 1);
  std::vector<Tensor> x_grad_tmp =
      split<T>(out_grad, phi::IntArray(sections), axis);
  auto out_dim = out_grad.dims().size();
  if (has_dynamic_shape(out_grad.shape())) {
    Tensor out_grad_shape = shape64<T>(out_grad);
    std::vector<Tensor> grad_shape;
    for (int i = 0; i < out_dim; i++) {
      if (i != axis) {
        grad_shape.push_back(get_slice<T>(out_grad_shape, i));
      }
    }
    Tensor grad_shape_tensor = concat<T>(grad_shape);

    for (int i = 0; i < x_num; i++) {
      if (x_grad[i]) {
        set_output<T>(backend::reshape<T>(x_grad_tmp[i], grad_shape_tensor),
                      x_grad[i]);
      }
    }
  } else {
    // compose shape for each input tensor
    std::vector<int64_t> grad_shape;
    for (int i = 0; i < out_dim; ++i) {
      if (i != axis) {
        grad_shape.push_back(out_grad.dims()[i]);
      }
    }

    // assign to each input tensor if need grad(stop_gradient=False)
    for (int i = 0; i < x_num; ++i) {
      if (x_grad[i]) {
        set_output<T>(reshape<T>(x_grad_tmp[i], grad_shape), x_grad[i]);
      }
    }
  }
}

template <typename T>
void layer_norm_grad(const Tensor& x,
                     const paddle::optional<Tensor>& scale,
                     const paddle::optional<Tensor>& bias,
                     const Tensor& mean,
                     const Tensor& variance,
                     const Tensor& out_grad,
                     float epsilon,
                     int begin_norm_axis,
                     Tensor* x_grad,
                     Tensor* scale_grad,
                     Tensor* bias_grad) {
  auto x_dims = x.dims();

  auto scale_ptr = scale.get_ptr();
  auto bias_ptr = bias.get_ptr();
  LayerNormDecompHelper decomp_help(x, scale, bias, begin_norm_axis);

  std::vector<int64_t> normalized_axis;
  std::vector<int64_t> mean_var_new_shape(mean.dims().size(), 0);
  for (int i = begin_norm_axis; i < x_dims.size(); ++i) {
    mean_var_new_shape.push_back(1);
    normalized_axis.push_back(i);
  }

  std::vector<int64_t> un_normalized_axis;
  for (int i = 0; i < begin_norm_axis; ++i) {
    un_normalized_axis.push_back(i);
  }

  auto mean_ = reshape<T>(mean, mean_var_new_shape);
  auto variance_ = reshape<T>(variance, mean_var_new_shape);

  auto x_cast = ConvertToMT<T>(x);
  Tensor scale_cast;
  if (scale_ptr) {
    scale_cast = decomp_help.Process<T>(*scale_ptr, x_cast);
  }

  // cast dtype to float32 if dtype =float16 or bfloat16

  auto out_grad_cast = ConvertToMT<T>(out_grad);
  if (scale_ptr) {
    scale_cast = ConvertToMT<T>(scale_cast);
  }

  auto x_sub_mean = x_cast - mean_;  // M,N
  auto tmp = (full_scalar<T>(1.0, variance_.dtype()) /
              (variance_ + full_scalar<T>(epsilon, variance_.dtype())));  // M,1
  auto sqrt_var_1 = sqrt<T>(tmp);                                         // M,1
  auto x_sub_mean_mul_sqrt_var_1 = x_sub_mean * sqrt_var_1;

  if (x_grad) {
    auto out_grad_scale = out_grad_cast;  // M,N
    if (scale_ptr) {
      out_grad_scale = out_grad_cast * scale_cast;  // M,N * 1,N = M,N
    }

    auto dx_end = sqrt_var_1 * out_grad_scale;
    auto d_mean = dx_end.sum(normalized_axis, x_cast.dtype(), true);  // M,1

    auto d_std_1 = (tmp * x_sub_mean * out_grad_scale)
                       .sum(normalized_axis, x_cast.dtype(), true);  // M,1
    auto d_std = d_std_1 * x_sub_mean_mul_sqrt_var_1;  // M,1 * M,N = M,N

    auto d_mean_d_std =
        (d_mean + d_std) / decomp_help.GetNormalizedNumel<T>(d_std);

    auto x_grad_tmp = dx_end - d_mean_d_std;
    x_grad_tmp = ConvertToOrig<T>(x_grad_tmp, x.dtype());

    set_output<T>(x_grad_tmp, x_grad);
  }

  if (scale_grad) {
    if (scale_ptr) {
      auto scale_grad_tmp = (x_sub_mean_mul_sqrt_var_1 * out_grad_cast)
                                .sum(un_normalized_axis, x_cast.dtype(), true);
      scale_grad_tmp = reshape<T>(scale_grad_tmp, {-1});
      scale_grad_tmp = ConvertToOrig<T>(scale_grad_tmp, scale_ptr->dtype());

      set_output<T>(scale_grad_tmp, scale_grad);
    } else {
      scale_grad = nullptr;
    }
  }

  if (bias_grad) {
    if (bias_ptr) {
      auto bias_grad_tmp =
          out_grad_cast.sum(un_normalized_axis, x_cast.dtype(), true);
      bias_grad_tmp = reshape<T>(bias_grad_tmp, {-1});
      bias_grad_tmp = ConvertToOrig<T>(bias_grad_tmp, bias_ptr->dtype());

      set_output<T>(bias_grad_tmp, bias_grad);
    } else {
      bias_grad = nullptr;
    }
  }
}

template <typename T>
void dropout_grad(const Tensor& mask,
                  const Tensor& out_grad,
                  const Scalar& p,
                  bool is_test,
                  const std::string& mode,
                  Tensor* x_grad) {
  if (!x_grad) return;
  if (is_test) {
    if (mode == "upscale_in_train") {
      by_pass<T>(out_grad, x_grad);
    } else {
      Tensor scalar = full_scalar<T>(1.0 - p.to<float>(), out_grad.dtype());
      set_output<T>(out_grad * scalar, x_grad);
    }
  } else {
    if (mode == "upscale_in_train") {
      if (has_dynamic_shape(out_grad.shape())) {
        if (p.to<float>() == 1.0f) {
          Tensor zero = full_scalar<T>(0.0, out_grad.dtype());
          set_output<T>(backend::scale<T>(out_grad, zero), x_grad);
        } else {
          Tensor scalar =
              full_scalar<T>(1.0 / (1.0 - p.to<float>()), out_grad.dtype());
          set_output<T>(backend::scale<T>(
                            out_grad * cast<T>(mask, out_grad.dtype()), scalar),
                        x_grad);
        }
      } else {
        if (p.to<float>() == 1.0f) {
          set_output<T>(scale<T>(out_grad, 0.0), x_grad);
        } else {
          set_output<T>(scale<T>(out_grad * cast<T>(mask, out_grad.dtype()),
                                 1.0 / (1.0 - p.to<float>())),
                        x_grad);
        }
      }
    } else {
      set_output<T>(out_grad * cast<T>(mask, out_grad.dtype()), x_grad);
    }
  }
}

template <typename T>
void erf_grad(const Tensor& x, const Tensor& out_grad, Tensor* x_grad) {
  if (x_grad) {
    auto m_2_sqrt_pi = full_scalar<T>(M_2_SQRTPI, x.dtype());
    auto neg_one = full_scalar<T>(-1.0, x.dtype());
    auto neg_tmp = neg_one * x * x;
    auto mul_tmp = m_2_sqrt_pi * exp<T>(neg_tmp);
    set_output<T>(out_grad * mul_tmp, x_grad);
  }
}

template <typename T>
void expand_grad(const Tensor& x,
                 const Tensor& out_grad,
                 const IntArray& shape,
                 Tensor* x_grad) {
  if (x_grad) {
    if (has_dynamic_shape(x.shape()) || has_dynamic_shape(out_grad.shape()) ||
        out_grad.dims() != x.dims()) {
      auto reduced = reduce_as<T>(out_grad, x);
      set_output<T>(reduced, x_grad);
    } else {
      by_pass<T>(out_grad, x_grad);
    }
  }
}

template <typename T>
void log_grad(const Tensor& x, const Tensor& out_grad, Tensor* x_grad) {
  if (x_grad) {
    // dx = dout / x
    set_output<T>(out_grad / x, x_grad);
  }
}

template <typename T>
void square_grad(const Tensor& x, const Tensor& out_grad, Tensor* x_grad) {
  if (x_grad) {
    auto two = full_scalar<T>(2.0, x.dtype());
    Tensor x_grad_tmp = two * x * out_grad;
    set_output<T>(x_grad_tmp, x_grad);
  }
}

template <typename T>
void exp_grad(const Tensor& out, const Tensor& out_grad, Tensor* x_grad) {
  if (x_grad) {
    Tensor out_promote = ConvertToMT<T>(out);
    Tensor out_grad_promote = ConvertToMT<T>(out_grad);

    auto x_grad_tmp = out_promote * out_grad_promote;
    set_output<T>(ConvertToOrig<T>(x_grad_tmp, out.dtype()), x_grad);
  }
}

template <typename T>
void sqrt_grad(const Tensor& out, const Tensor& out_grad, Tensor* x_grad) {
  if (x_grad) {
    // This calculation is important for resnet.
    auto factor = full_scalar<T>(0.5, out.dtype());
    auto x_grad_tmp = (factor / out) * out_grad;
    set_output<T>(x_grad_tmp, x_grad);
  }
}

template <typename T>
void rsqrt_grad(const Tensor& out, const Tensor& out_grad, Tensor* x_grad) {
  if (x_grad) {
    // This calculation is important for resnet.
    auto factor = full_scalar<T>(-0.5, out.dtype());
    auto x_grad_tmp = factor * out * out * out * out_grad;
    set_output<T>(x_grad_tmp, x_grad);
  }
}

template <typename T>
void silu_grad(const Tensor& x,
               const Tensor& out,
               const Tensor& out_grad,
               Tensor* x_grad) {
  if (x_grad) {
    auto one = full_scalar<T>(1.0, x.dtype());

    auto x_cast = ConvertToMT<T>(x);
    auto out_cast = ConvertToMT<T>(out);
    auto out_grad_cast = ConvertToMT<T>(out_grad);
    auto res = out_grad_cast * sigmoid<T>(x_cast) * (one + x_cast - out_cast);
    set_output<T>(ConvertToOrig<T>(res, x.dtype()), x_grad);
  }
}

template <typename T>
void softmax_grad(const Tensor& out,
                  const Tensor& out_grad,
                  int axis,
                  Tensor* x_grad) {
  if (x_grad) {
    if (axis < 0) {
      axis += out.dims().size();
    }

    if (out_grad.dims().size() > 0) {
      auto new_out_grad = out_grad * out;
      auto tmp_x_grad =
          new_out_grad - out * sum<T>(new_out_grad, {axis}, out.dtype(), true);
      set_output<T>(tmp_x_grad, x_grad);
    } else {
      auto zeros =
          full<T>(common::vectorize(out.dims()), 0, out.dtype(), out.place());
      set_output<T>(zeros, x_grad);
    }
  }
}

template <typename T>
void squeeze_grad(const Tensor& x,
                  const Tensor& out_grad,
                  const IntArray& axis,
                  Tensor* x_grad) {
  if (x_grad) {
    auto x_grad_out = out_grad.dims().size() == x.dims().size()
                          ? out_grad
                          : unsqueeze<T>(out_grad, axis);
    set_output<T>(x_grad_out, x_grad);
  }
}

template <typename T>
void unsqueeze_grad(const Tensor& x,
                    const Tensor& out_grad,
                    const IntArray& axis,
                    Tensor* x_grad) {
  // for xshape = [10, 2, 5], axis = [3, 1, 1], out_grad.shape = [10, 1, 1, 2,
  // 5, 1], it outputs squeeze axis = [5, 2, 1]
  const auto& IncreaseAxis = [](std::vector<int64_t>* axis_data,
                                int64_t pivot) {
    for (size_t i = 0; i < axis_data->size(); ++i) {
      if ((*axis_data)[i] >= pivot) (*axis_data)[i] += 1;
    }
  };
  const auto& GetRealAxis = [&](const IntArray& axis) -> decltype(auto) {
    // for axis = [0, 3, 3], it outputs [0, 3, 3+1], because unsqueeze support
    // duplicated axis.
    std::vector<int64_t> output_axis;
    const int64_t x_rank = x.dims().size();
    const std::vector<int64_t> axis_data = axis.GetData();
    for (size_t i = 0; i < axis_data.size(); ++i) {
      int64_t value = axis_data[i];
      if (value < 0) value += (x_rank + i + 1);
      IncreaseAxis(&output_axis, value);
      output_axis.push_back(value);
    }
    return output_axis;
  };

  if (x_grad) {
    auto x_grad_out = squeeze<T>(out_grad, GetRealAxis(axis));
    set_output<T>(x_grad_out, x_grad);
  }
}

template <typename T>
void matmul_grad(const Tensor& x,
                 const Tensor& y,
                 const Tensor& out_grad,
                 bool transpose_x,
                 bool transpose_y,
                 Tensor* x_grad,
                 Tensor* y_grad) {
  auto unsqueeze_out_grad = out_grad;
  size_t out_grad_rank = out_grad.shape().size();
  size_t x_rank = x.shape().size();
  size_t y_rank = y.shape().size();
  int temp_rank_y = out_grad_rank - 1;
  int temp_rank_x = out_grad_rank;
  if (out_grad_rank < y_rank) {
    unsqueeze_out_grad = unsqueeze<T>(out_grad, {temp_rank_y});
  }
  if (out_grad_rank < x_rank) {
    unsqueeze_out_grad = unsqueeze<T>(out_grad, {temp_rank_x});
  }

  auto temp_x_unsqueeze = x;
  if (x_rank == 1) {
    temp_x_unsqueeze = unsqueeze<T>(x, {0});
  }

  auto temp_y_unsqueeze = y;
  if (y_rank == 1) {
    temp_y_unsqueeze = unsqueeze<T>(y, {1});
  }

  if (x_grad) {
    auto x_grad_mm =
        matmul<T>(unsqueeze_out_grad, temp_y_unsqueeze, false, !transpose_y);
    auto x_grad_trans = x_grad_mm;

    if (transpose_x) {
      std::vector<int> reverse_perm;
      for (size_t i = 0; i < x_grad_trans.shape().size(); i++) {
        reverse_perm.push_back(i);
      }
      std::swap(reverse_perm[reverse_perm.size() - 1],
                reverse_perm[reverse_perm.size() - 2]);
      x_grad_trans = transpose<T>(x_grad_mm, reverse_perm);
    }
    if (has_dynamic_shape(x.shape()) ||
        has_dynamic_shape(x_grad_trans.shape()) ||
        x_grad_trans.dims() != x.dims()) {
      auto x_grad_out = reduce_as<T>(x_grad_trans, temp_x_unsqueeze);
      set_output<T>(x_grad_out, x_grad);
    } else {
      auto x_grad_out = x_grad_trans;
      set_output<T>(x_grad_out, x_grad);
    }
  }

  if (y_grad) {
    auto y_grad_mm =
        matmul<T>(temp_x_unsqueeze, unsqueeze_out_grad, !transpose_x, false);
    auto y_grad_trans = y_grad_mm;

    if (transpose_y) {
      std::vector<int> reverse_perm;
      for (size_t i = 0; i < y_grad_mm.shape().size(); i++) {
        reverse_perm.push_back(i);
      }
      std::swap(reverse_perm[reverse_perm.size() - 1],
                reverse_perm[reverse_perm.size() - 2]);
      y_grad_trans = transpose<T>(y_grad_mm, reverse_perm);
    }
    if (has_dynamic_shape(y.shape()) ||
        has_dynamic_shape(y_grad_trans.shape()) ||
        y_grad_trans.dims() != y.dims()) {
      auto y_grad_out = reduce_as<T>(y_grad_trans, temp_y_unsqueeze);
      set_output<T>(y_grad_out, y_grad);
    } else {
      auto y_grad_out = y_grad_trans;
      set_output<T>(y_grad_out, y_grad);
    }
  }
}

template <typename T>
void maximum_grad(const Tensor& x,
                  const Tensor& y,
                  const Tensor& out_grad,
                  Tensor* x_grad,
                  Tensor* y_grad) {
  if (x_grad) {
    auto x_tmp = cast<T>(greater_than<T>(x, y), out_grad.dtype());
    auto dx_res = out_grad * x_tmp;
    if (has_dynamic_shape(x.shape()) || has_dynamic_shape(out_grad.shape()) ||
        out_grad.dims() != x.dims()) {
      auto dx_reduce_res = reduce_as<T>(dx_res, x);
      set_output<T>(dx_reduce_res, x_grad);
    } else {
      set_output<T>(dx_res, x_grad);
    }
  }

  if (y_grad) {
    auto y_tmp = cast<T>(less_equal<T>(x, y), out_grad.dtype());
    auto dy_res = out_grad * y_tmp;
    if (has_dynamic_shape(y.shape()) || has_dynamic_shape(out_grad.shape()) ||
        out_grad.dims() != y.dims()) {
      auto dy_reduce_res = reduce_as<T>(dy_res, y);
      set_output<T>(dy_reduce_res, y_grad);
    } else {
      set_output<T>(dy_res, y_grad);
    }
  }
}

template <typename T>
void masked_select_grad(const Tensor& x,
                        const Tensor& mask,
                        const Tensor& out_grad,
                        Tensor* x_grad) {
  if (x_grad) {
    auto promoted_x = ConvertToMT<T>(x);
    auto promoted_out_grad = ConvertToMT<T>(out_grad);

    auto x_num = 1;
    for (size_t i = 0; i < promoted_x.shape().size(); i++) {
      x_num *= promoted_x.shape()[i];
    }

    auto grad_num = 1;
    for (size_t i = 0; i < promoted_out_grad.shape().size(); i++) {
      grad_num *= promoted_out_grad.shape()[i];
    }

    auto end = full<T>({1}, x_num, promoted_x.dtype(), x.place());
    auto start = full<T>({1}, 0, promoted_x.dtype(), x.place());
    auto step = full<T>({1}, 1, promoted_x.dtype(), x.place());
    auto x_arange = backend::arange<T>(
        start, end, step, promoted_x.dtype(), promoted_x.place());

    auto x_arange_reshape = reshape<T>(x_arange, promoted_x.shape());

    auto x_index = masked_select<T>(x_arange_reshape, mask);

    auto index_num = x_index.shape()[0];

    auto grad_reshape =
        cast<T>(reshape<T>(promoted_out_grad, {grad_num}), promoted_x.dtype());

    auto grad_trans = grad_reshape;
    if (grad_num > index_num) {
      grad_trans = slice<T>(grad_reshape, {0}, {0}, {index_num}, {1}, {});
    } else if (grad_num < index_num) {
      auto pad_zeros = full<T>(
          {index_num - grad_num}, 0, promoted_x.dtype(), promoted_x.place());
      grad_trans = concat<T>({grad_reshape, pad_zeros}, 0);
    }

    auto input_tensor =
        full<T>({x_num}, 0, promoted_x.dtype(), promoted_x.place());
    auto index_tensor = cast<T>(x_index, DataType::INT64);
    auto update_tensor = grad_trans;
    auto x_output =
        scatter<T>(input_tensor, index_tensor, update_tensor, false);
    auto res = cast<T>(reshape<T>(x_output, promoted_x.shape()), x.dtype());
    set_output<T>(res, x_grad);
  }
}

template <typename T>
void relu_grad(const Tensor& out, const Tensor& out_grad, Tensor* x_grad) {
  if (x_grad) {
    Tensor zeros = full_scalar<T>(0.0, out.dtype());
    auto mask = greater_than<T>(out, zeros);
    auto res = cast<T>(mask, out.dtype()) * out_grad;
    set_output<T>(res, x_grad);
  }
}

template <typename T>
void relu6_grad(const Tensor& out, const Tensor& out_grad, Tensor* x_grad) {
  if (x_grad) {
    Tensor zeros = full_scalar<T>(0.0, out.dtype());
    Tensor six = full_scalar<T>(6.0, out.dtype());
    auto mask_gt = greater_than<T>(out, zeros);
    auto mask_lt = less_than<T>(out, six);
    auto mask = backend::logical_and<T>(mask_gt, mask_lt);
    auto res = cast<T>(mask, out.dtype()) * out_grad;
    set_output<T>(res, x_grad);
  }
}

template <typename T>
void gather_grad(const Tensor& x,
                 const Tensor& index,
                 const Tensor& out_grad,
                 const Scalar& axis,
                 Tensor* grad_x) {
  Tensor zero_tensor;
  if (has_dynamic_shape(x.shape())) {
    zero_tensor =
        backend::full_with_tensor<T>(shape64<T>(x), 0.0, x.dtype(), x.place());
  } else {
    zero_tensor =
        full<T>(common::vectorize(x.dims()), 0.0, x.dtype(), x.place());
  }
  std::vector<int> tmp_perm;

  // change axis to rank 0
  int axis_value = axis.to<int>();
  int rank = x.dims().size();
  if (axis_value < 0) {
    axis_value += rank;
  }

  tmp_perm.push_back(axis_value);
  // make other ranks
  for (int i = 0; i < rank; ++i) {
    if (i != axis_value) {
      tmp_perm.push_back(i);
    }
  }
  std::vector<int> reverse_perm(tmp_perm);
  // make origin ranks
  for (int i = 0; i < static_cast<int>(tmp_perm.size()); ++i) {
    if (tmp_perm[i] >= 0) {
      reverse_perm[tmp_perm[i]] = i;
    } else {
      reverse_perm[tmp_perm[i] + tmp_perm.size()] = i;
    }
  }

  // transpose out_grad and zero grad to target rank.
  auto tmp_zero_x_grad = zero_tensor;
  auto tmp_out_grad = out_grad;
  if (zero_tensor.dims().size() > 0) {
    tmp_zero_x_grad = transpose<T>(zero_tensor, tmp_perm);
  }
  if (out_grad.dims().size() > 0) {
    tmp_out_grad = transpose<T>(out_grad, tmp_perm);
  }
  // scatter grad to grad_x
  auto tmp_grad_x = scatter<T>(tmp_zero_x_grad, index, tmp_out_grad, false);
  auto tmp_grad_x_transposed = tmp_grad_x;
  if (tmp_grad_x.dims().size() > 0) {
    tmp_grad_x_transposed = transpose<T>(tmp_grad_x, reverse_perm);
  }
  set_output<T>(tmp_grad_x_transposed, grad_x);
}

template <typename T>
void gather_nd_grad(const Tensor& x,
                    const Tensor& index,
                    const Tensor& out_grad,
                    Tensor* x_grad) {
  if (x_grad) {
    Tensor zero_tensor;
    if (has_dynamic_shape(x.shape())) {
      zero_tensor = backend::full_with_tensor<T>(
          shape64<T>(x), 0.0, x.dtype(), x.place());
    } else {
      zero_tensor =
          full<T>(common::vectorize(x.dims()), 0.0, x.dtype(), x.place());
    }
    auto x_grad_tmp = scatter_nd_add<T>(zero_tensor, index, out_grad);
    set_output<T>(x_grad_tmp, x_grad);
  }
}

template <typename T>
void instance_norm_grad(const Tensor& x,
                        const paddle::optional<Tensor>& scale,
                        const Tensor& saved_mean,
                        const Tensor& saved_variance,
                        const Tensor& y_grad,
                        float epsilon,
                        Tensor* x_grad,
                        Tensor* scale_grad,
                        Tensor* bias_grad) {
  const int n = x.dims()[0];
  const int c = x.dims()[1];
  InstanceNormDecompHelper<T> decomp_helper(x);

  std::vector<int64_t> reduce_axes = decomp_helper.GetReduceAxis();
  std::vector<int64_t> n_reduce_axes = decomp_helper.GetNPlusReduceAxis();
  Tensor hw = decomp_helper.GetHW(x);

  auto promoted_y_grad = ConvertToMT<T>(y_grad);

  Tensor x_hat;
  Tensor std_inv;
  if (scale_grad || x_grad) {
    auto promoted_x = ConvertToMT<T>(x);
    auto promoted_saved_mean = ConvertToMT<T>(saved_mean);
    auto promoted_saved_var = ConvertToMT<T>(saved_variance);

    std::vector<int64_t> mean_new_shape{n, c};
    for (size_t i = 0; i < reduce_axes.size(); ++i) {
      mean_new_shape.push_back(1);
    }
    auto mean = reshape<T>(promoted_saved_mean, mean_new_shape);
    std_inv = reshape<T>(promoted_saved_var, mean_new_shape);
    x_hat = (promoted_x - mean) * std_inv;
  }

  // x_grad = scale * inv_var * (y_grad - y_grad.mean(2,3) - x_hat * (y_grad *
  // x_hat).mean((h,w)))
  if (x_grad) {
    bool is_reduce_empty = reduce_axes.empty();
    Tensor scale_data_tensor =
        scale.get_ptr() ? scale.get()
                        : full<T>(IntArray({c}), 1., x.dtype(), x.place());
    auto unsqueeze_shape = get_unsqueeze_dims(scale_data_tensor, n_reduce_axes);
    auto scale_data = reshape<T>(scale_data_tensor, unsqueeze_shape);
    auto promoted_scale = ConvertToMT<T>(scale_data);
    auto tmp1 =
        is_reduce_empty
            ? promoted_y_grad
            : promoted_y_grad.sum(reduce_axes, promoted_y_grad.dtype(), true);
    auto tmp2 = is_reduce_empty
                    ? (promoted_y_grad * x_hat)
                    : (promoted_y_grad * x_hat)
                          .sum(reduce_axes, promoted_y_grad.dtype(), true);
    auto result = (promoted_scale * std_inv) *
                  (promoted_y_grad - tmp1 / hw - (x_hat * tmp2 / hw));
    set_output<T>(ConvertToOrig<T>(result, x.dtype()), x_grad);
  }
  // scale_grad = x_hat * y_grad.sum(n, h, w)
  if (scale_grad) {
    auto result = (promoted_y_grad * x_hat).sum(n_reduce_axes);
    auto scale_dtype = scale.get_ptr() ? scale.get().dtype() : x.dtype();
    set_output<T>(ConvertToOrig<T>(result, scale_dtype), scale_grad);
  }
  // d_bias = y_grad.sum(n, h, w)
  if (bias_grad) {
    auto result = promoted_y_grad.sum(n_reduce_axes);
    auto scale_dtype = scale.get_ptr() ? scale.get().dtype() : x.dtype();
    set_output<T>(ConvertToOrig<T>(result, scale_dtype), bias_grad);
  }
}

template <typename T>
void pad_grad(const Tensor& input,
              const Tensor& out_grad,
              const std::vector<int>& paddings,
              const Scalar& pad_value,
              Tensor* input_grad) {
  if (input_grad) {
    Tensor out_tmp;
    size_t rank = input.dims().size();
    std::vector<int64_t> axes(rank, 0);
    std::vector<int64_t> infer_flags(rank, 1);
    std::vector<int64_t> decrease_axis({});
    if (has_dynamic_shape(out_grad.shape())) {
      auto out_shape = shape64<T>(out_grad);
      std::vector<Tensor> starts, ends;
      for (size_t i = 0; i < rank; ++i) {
        starts.push_back(
            full<T>({1}, paddings[2 * i], out_shape.dtype(), out_grad.place()));
        ends.push_back(get_slice<T>(out_shape, i) - full<T>({1},
                                                            paddings[2 * i + 1],
                                                            out_shape.dtype(),
                                                            out_grad.place()));
        axes[i] = i;
      }
      out_tmp = backend::slice<T>(out_grad,
                                  concat<T>(starts),
                                  concat<T>(ends),
                                  axes,
                                  infer_flags,
                                  decrease_axis);
    } else {
      auto out_dims = out_grad.dims();

      std::vector<int64_t> starts(rank, 0);
      std::vector<int64_t> ends(rank, 0);

      for (size_t i = 0; i < rank; ++i) {
        starts[i] = static_cast<int64_t>(paddings[2 * i]);
        ends[i] = static_cast<int64_t>(out_dims[i] - paddings[2 * i + 1]);
        axes[i] = i;
      }
      out_tmp =
          slice<T>(out_grad, axes, starts, ends, infer_flags, decrease_axis);
    }
    set_output<T>(out_tmp, input_grad);
  }
}

template <typename T>
void max_grad(const Tensor& x,
              const Tensor& out,
              const Tensor& out_grad,
              const IntArray& axis,
              bool keepdim,
              bool reduce_all,
              Tensor* x_grad) {
  if (!x_grad) {
    return;
  }

  Tensor x_grad_tmp;
  if (has_dynamic_shape(x.shape())) {
    const Tensor x_shape = shape64<T>(x);
    const Tensor zero_tensor =
        backend::full_with_tensor<T>(x_shape, 0.0, x.dtype(), x.place());
    const int64_t axis_size = axis.size();
    const int64_t x_dim_size = x.dims().size();

    reduce_all = false;
    if (reduce_all || axis_size == 0 || axis_size == x_dim_size) {
      reduce_all = true;
    } else {
      reduce_all = false;
    }

    if (x_dim_size == 0 || x_dim_size == 1 || keepdim) {
      auto out_grad_tmp = backend::expand<T>(out_grad, x_shape);
      auto out_tmp = backend::expand<T>(out, x_shape);
      auto mask = equal<T>(x, out_tmp);
      x_grad_tmp = where<T>(mask, out_grad_tmp, zero_tensor);
    } else {
      const Tensor out_grad_shape = shape64<T>(out_grad);
      auto axis_ = std::vector<int64_t>();

      if (reduce_all) {
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
      const Tensor out_grad_shape_extend =
          get_unsqueeze_dims<T>(out_grad_shape, axis_);
      auto out_grad_ = backend::reshape<T>(out_grad, out_grad_shape_extend);
      auto out_ = backend::reshape<T>(out, out_grad_shape_extend);
      auto out_grad_tmp = backend::expand<T>(out_grad_, x_shape);
      auto out_tmp = backend::expand<T>(out_, x_shape);
      auto mask = equal<T>(x, out_tmp);
      x_grad_tmp = where<T>(mask, out_grad_tmp, zero_tensor);
    }
  } else {
    auto zero_tensor =
        full<T>(common::vectorize(x.dims()), 0.0, x.dtype(), x.place());
    std::vector<int64_t> x_dim = common::vectorize<int64_t>(x.dims());
    int64_t axis_size = axis.size();
    int64_t x_dim_size = x_dim.size();
    reduce_all = false;
    if (reduce_all || axis_size == 0 || axis_size == x_dim_size) {
      reduce_all = true;
    } else {
      reduce_all = false;
    }

    if (x_dim_size == 0 || x_dim_size == 1 || keepdim) {
      auto out_grad_tmp = out_grad.expand(IntArray(x_dim));
      auto out_tmp = out.expand(IntArray(x_dim));
      auto mask = equal<T>(x, out_tmp);
      x_grad_tmp = where<T>(mask, out_grad_tmp, zero_tensor);
    } else {
      auto axis_ = std::vector<int64_t>();
      if (reduce_all) {
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
      auto out_grad_shape = get_unsqueeze_dims(out_grad, axis_);
      auto out_grad_ = reshape<T>(out_grad, out_grad_shape);
      auto out_ = reshape<T>(out, out_grad_shape);
      auto out_grad_tmp = out_grad_.expand(IntArray(x_dim));
      auto out_tmp = out_.expand(IntArray(x_dim));
      auto mask = equal<T>(x, out_tmp);
      x_grad_tmp = where<T>(mask, out_grad_tmp, zero_tensor);
    }
  }
  set_output<T>(x_grad_tmp, x_grad);
}

template <typename T>
void slice_grad(const Tensor& input,
                const Tensor& out_grad,
                const std::vector<int64_t>& axes,
                const IntArray& starts,
                const IntArray& ends,
                const std::vector<int64_t>& infer_flags,
                const std::vector<int64_t>& decrease_axis,
                Tensor* input_grad) {
  if (input_grad) {
    size_t rank = input.dims().size();
    auto out_dims = out_grad.dims();
    std::vector<int64_t> origin_out_shape;
    auto in_dims = input.dims();

    auto decrease_size = decrease_axis.size();
    if (decrease_size > 0) {
      if (decrease_size == static_cast<size_t>(in_dims.size())) {
        // all dims decrease
        out_dims = common::make_ddim(std::vector<int>(decrease_size, 1));
      } else {
        origin_out_shape.resize(out_dims.size() + decrease_size, -1);
        for (size_t i = 0; i < decrease_size; ++i) {
          origin_out_shape[decrease_axis[i]] = 1;
        }

        int index = 0;
        for (size_t i = 0; i < origin_out_shape.size(); ++i) {
          if (origin_out_shape[i] == -1) {
            origin_out_shape[i] = out_dims[index];
            ++index;
          }
        }
        out_dims = common::make_ddim(origin_out_shape);
      }
    }

    std::vector<int> offsets(rank, 0);
    std::vector<int> extents(rank, 0);
    for (size_t i = 0; i < rank; ++i) {
      offsets[i] = 0;
      extents[i] = out_dims[i];
    }
    for (size_t i = 0; i < axes.size(); ++i) {
      int axis = axes[i];
      int64_t start = starts[i] < 0 ? (starts[i] + in_dims[axis]) : starts[i];
      start = std::max(start, static_cast<int64_t>(0));
      offsets[axis] = start;
    }

    std::vector<int> paddings;
    for (size_t i = 0; i < rank; ++i) {
      paddings.push_back(offsets[i]);
      paddings.push_back((in_dims[i] - out_dims[i]) - offsets[i]);
    }
    Tensor reshape_out_grad;
    if (out_grad.shape().size() == 0) {
      reshape_out_grad = full<T>({1}, 1, input.dtype(), input.place());
    } else {
      reshape_out_grad = out_grad;
    }

    // If axes.size() is 1, we can attempt to use concatenation instead of
    // padding.
    if (axes.size() == 1) {
      const int64_t axis = axes[0];
      const std::vector<int64_t> input_shape = input.shape();
      if (decrease_size > 0 &&
          (decrease_size != static_cast<size_t>(in_dims.size()))) {
        reshape_out_grad = reshape<T>(reshape_out_grad, origin_out_shape);
      }

      std::vector<Tensor> concat_tensors;
      // if concat axis has a shape of 0, concatenation may lead to errors.
      if (paddings[2 * axis] != 0) {
        std::vector<int64_t> left_shape(input_shape);
        left_shape[axis] = paddings[2 * axis];
        concat_tensors.push_back(
            full<T>(left_shape, 0.0, out_grad.dtype(), out_grad.place()));
      }
      concat_tensors.push_back(reshape_out_grad);

      if (paddings[2 * axis + 1] != 0) {
        std::vector<int64_t> right_shape(input_shape);
        right_shape[axis] = paddings[2 * axis + 1];
        concat_tensors.push_back(
            full<T>(right_shape, 0.0, out_grad.dtype(), out_grad.place()));
      }

      set_output<T>(concat<T>(concat_tensors, axis), input_grad);
    } else {
      if (decrease_size > 0 &&
          (decrease_size != static_cast<size_t>(in_dims.size()))) {
        auto out_tmp = pad<T>(
            reshape<T>(reshape_out_grad, origin_out_shape), paddings, 0.0);
        set_output<T>(out_tmp, input_grad);
      } else {
        auto out_tmp = pad<T>(reshape_out_grad, paddings, 0.0);
        set_output<T>(out_tmp, input_grad);
      }
    }
  }
}

template <typename T>
void tile_grad(const Tensor& x,
               const Tensor& out_grad,
               const IntArray& repeat_times,
               Tensor* x_grad) {
  if (x_grad) {
    std::vector<int64_t> repeat_times_data = repeat_times.GetData();
    Tensor out_grad_tmp = out_grad;
    Tensor x_grad_tmp;

    if (has_dynamic_shape(x.shape()) || has_dynamic_shape(out_grad.shape())) {
      std::vector<Tensor> out_grad_shape_vec;
      for (int64_t i = 0; i < out_grad.dims().size(); ++i) {
        auto out_grad_shape_slice = get_slice<T>(shape64<T>(out_grad_tmp), i);
        out_grad_shape_vec.push_back(out_grad_shape_slice);
      }
      if (repeat_times_data.size() != 0) {
        while (true) {
          std::vector<Tensor> expand_shape_vec;
          for (int64_t i = 0; i < out_grad_tmp.dims().size(); ++i) {
            auto expand_shape = get_slice<T>(shape64<T>(out_grad_tmp), i);
            expand_shape_vec.push_back(expand_shape);
          }
          int num_reduce = 0;
          while (repeat_times_data.size() != 0 &&
                 expand_shape_vec.size() <= 8) {
            auto repeat = repeat_times_data.back();
            auto orig_size =
                cast<T>(out_grad_shape_vec.back() / repeat, DataType::INT64);
            size_t out_grad_last_index = out_grad_shape_vec.size() - 1;
            expand_shape_vec[out_grad_last_index] =
                full<T>({1}, repeat, DataType::INT64);
            expand_shape_vec.insert(
                expand_shape_vec.begin() + out_grad_shape_vec.size(),
                orig_size);

            repeat_times_data.pop_back();
            out_grad_shape_vec.pop_back();
            ++num_reduce;
          }
          int axis = static_cast<int>(out_grad_shape_vec.size());
          std::vector<Tensor> reduce_axes_vec;
          for (int i = 0; i < num_reduce; ++i) {
            reduce_axes_vec.push_back(full<T>({1}, axis, DataType::INT32));
            axis += 2;
          }
          out_grad_tmp =
              backend::reshape<T>(out_grad_tmp, concat<T>(expand_shape_vec));
          out_grad_tmp =
              backend::sum<T>(out_grad_tmp, concat<T>(reduce_axes_vec));

          if (repeat_times_data.size() == 0) {
            break;
          }
        }
      }
      x_grad_tmp = backend::reshape<T>(out_grad_tmp, shape64<T>(x));
    } else {
      std::vector<int64_t> out_grad_shape(out_grad.shape());

      if (repeat_times_data.size() != 0) {
        while (true) {
          std::vector<int64_t> expand_shape(out_grad_tmp.shape());

          int num_reduce = 0;
          // By definition, out_grad_shape.size() is guaranteed to be greater
          // than or equal to repeat_times.size(). Paddle only supports up to 9
          // dimensions.
          while (repeat_times_data.size() != 0 && expand_shape.size() <= 8) {
            // We construct the reduction from the backward direction, as the
            // repeats are aligned with the output from right to left.
            int64_t repeat = repeat_times_data.back();
            int64_t orig_size = out_grad_shape.back() / repeat;
            size_t out_grad_last_index = out_grad_shape.size() - 1;

            // Reshape the corresponding dimension to be `repeat` multiplied by
            // `orig_size`.
            expand_shape[out_grad_last_index] = repeat;
            expand_shape.insert(
                expand_shape.begin() + out_grad_shape.size(), 1, orig_size);

            repeat_times_data.pop_back();
            out_grad_shape.pop_back();
            ++num_reduce;
          }

          // Find the reduce_axes, which are determined from the forward
          // direction. Since there can be some axes that haven't been reduced,
          // we simply skip them this round.
          int64_t axis = static_cast<int64_t>(out_grad_shape.size());
          std::vector<int64_t> reduce_axes;
          for (int i = 0; i < num_reduce; ++i) {
            reduce_axes.push_back(axis);
            axis += 2;
          }
          out_grad_tmp = reshape<T>(out_grad_tmp, expand_shape);
          out_grad_tmp = sum<T>(out_grad_tmp, reduce_axes);

          if (repeat_times_data.size() == 0) {
            break;
          }
        }
      }
      x_grad_tmp = reshape<T>(out_grad_tmp, x.shape());
    }

    set_output<T>(x_grad_tmp, x_grad);
  }
}

template <typename T>
void hardsigmoid_grad(const Tensor& out,
                      const Tensor& out_grad,
                      float slope,
                      float offset,
                      Tensor* x_grad) {
  if (x_grad) {
    Tensor zeros = full_scalar<T>(0.0, out.dtype());
    Tensor one = full_scalar<T>(1.0, out.dtype());
    auto mask_gt = greater_than<T>(out, zeros);
    auto mask_lt = less_than<T>(out, one);
    auto mask = backend::logical_and<T>(mask_gt, mask_lt);
    Tensor slope_t = full_scalar<T>(slope, out.dtype());
    auto res = cast<T>(mask, out.dtype()) * slope_t * out_grad;
    set_output<T>(res, x_grad);
  }
}

template <typename T>
void hardswish_grad(const Tensor& x, const Tensor& out_grad, Tensor* x_grad) {
  if (x_grad) {
    const Tensor offset = full_scalar<T>(3.0, x.dtype());
    const Tensor neg_offset = full_scalar<T>(-3.0, x.dtype());
    const Tensor threshold = full_scalar<T>(6.0, x.dtype());

    auto factor = full_scalar<T>(0.5, x.dtype());

    auto one = full_scalar<T>(1.0, x.dtype());
    auto t1 = greater_than<T>(x, neg_offset);
    auto t2 = less_than<T>(x, threshold - offset);
    t1 = cast<T>(t1, x.dtype());
    t2 = cast<T>(t2, x.dtype());

    auto res = out_grad * (t1 * t2 * (x / offset + factor) + one - t2);
    // auto res = out_grad * (t1 * t2 * (x / offset + factor) );
    set_output<T>(res, x_grad);
  }
}

template <typename T>
void leaky_relu_grad(const Tensor& out,
                     const Tensor& out_grad,
                     float negative_slope,
                     Tensor* x_grad) {
  if (x_grad) {
    auto zero = full_scalar<T>(0.0, out.dtype());
    auto condition = greater_than<T>(out, zero);
    auto res = where<T>(condition, out_grad, out_grad * negative_slope);
    set_output<T>(res, x_grad);
  }
}

template <typename T>
void sigmoid_grad(const Tensor& out, const Tensor& out_grad, Tensor* x_grad) {
  if (x_grad) {
    auto one_tensor = full_scalar<T>(1.0, out.dtype());
    set_output<T>(out_grad * (out * (one_tensor - out)), x_grad);
  }
}

template <typename T>
void topk_grad(const Tensor& x,
               const Tensor& indices,
               const Tensor& out_grad,
               const Scalar& k,
               int axis,
               const bool& largest,
               const bool& sorted,
               Tensor* x_grad) {
  if (x_grad) {
    // put_along_axis doesn't support zero dim
    if (x.dims().size() == 0) {
      by_pass<T>(out_grad, x_grad);
      return;
    }

    // function `put_along_axis` requires a non-negative axis
    if (axis < 0) {
      axis += x.dims().size();
    }

    Tensor zero_tensor;
    if (has_dynamic_shape(x.shape())) {
      zero_tensor =
          backend::full_with_tensor<T>(shape64<T>(x), 0, x.dtype(), x.place());
    } else {
      zero_tensor =
          full<T>(common::vectorize(x.dims()), 0, x.dtype(), x.place());
    }
    auto x_grad_tmp = put_along_axis<T>(zero_tensor, indices, out_grad, axis);
    set_output<T>(x_grad_tmp, x_grad);
  }
}

template <typename T>
void batch_norm_grad(const Tensor& x,
                     const paddle::optional<Tensor>& scale,
                     const paddle::optional<Tensor>& bias,
                     const paddle::optional<Tensor>& mean_out,
                     const paddle::optional<Tensor>& variance_out,
                     const Tensor& saved_mean,
                     const Tensor& saved_variance,
                     const paddle::optional<Tensor>& reserve_space,
                     const Tensor& out_grad,
                     float momentum,
                     float epsilon,
                     const std::string& data_layout,
                     bool is_test,
                     bool use_global_stats,
                     bool trainable_statistics,
                     Tensor* x_grad,
                     Tensor* scale_grad,
                     Tensor* bias_grad) {
  use_global_stats = is_test || use_global_stats;

  Tensor x_data = ConvertToMT<T>(x);
  Tensor out_grad_data = ConvertToMT<T>(out_grad);

  Tensor mean_data;
  Tensor rsqrt_var;
  auto dtype = x_data.dtype();

  BatchNormDecompHelper<T> decomp_help(x, scale, bias, data_layout);

  auto reduce_axes = decomp_help.GetReduceAxis();
  auto scale_bias_new_shape = decomp_help.GetScaleBiasNewShape();

  if (use_global_stats) {
    auto run_var = variance_out.get();
    auto run_mean = mean_out.get();

    run_var = reshape<T>(run_var, scale_bias_new_shape);
    run_mean = reshape<T>(run_mean, scale_bias_new_shape);
    auto eps = full_scalar<T>(epsilon, run_var.dtype());
    mean_data = run_mean;
    rsqrt_var = rsqrt<T>(run_var + eps);
  } else {
    mean_data = reshape<T>(saved_mean, scale_bias_new_shape);
    rsqrt_var = reshape<T>(saved_variance, scale_bias_new_shape);
  }

  if (x_grad) {
    auto out_grad_data_sum = sum<T>(out_grad_data, reduce_axes, dtype, true);
    auto sum_dout_mul_diff =
        sum<T>(out_grad_data * (x_data - mean_data), reduce_axes, dtype, true);

    if (use_global_stats) {
      auto x_grad_data = rsqrt_var * out_grad_data;
      if (scale) {
        x_grad_data =
            reshape<T>(scale.get(), scale_bias_new_shape) * x_grad_data;
      }
      x_grad_data = ConvertToOrig<T>(x_grad_data, x.dtype());
      set_output<T>(x_grad_data, x_grad);
    } else {
      auto part1 = rsqrt_var;
      if (scale) {
        part1 = reshape<T>(scale.get(), scale_bias_new_shape) * part1;
      }
      auto mean_temp1 = out_grad_data_sum / decomp_help.GetNHW(x_data);
      auto mean_temp2 = sum_dout_mul_diff / decomp_help.GetNHW(x_data) *
                        rsqrt_var * rsqrt_var;

      auto part2 =
          out_grad_data - mean_temp1 - (x_data - mean_data) * mean_temp2;

      auto x_grad_data = part1 * part2;
      x_grad_data = ConvertToOrig<T>(x_grad_data, x.dtype());
      set_output<T>(x_grad_data, x_grad);
    }
    if (scale_grad) {
      auto scale_grad_data = sum_dout_mul_diff * rsqrt_var;
      scale_grad_data = reshape<T>(scale_grad_data, {-1});
      set_output<T>(scale_grad_data, scale_grad);
    }
    if (bias_grad) {
      set_output<T>(reshape<T>(out_grad_data_sum, {-1}), bias_grad);
    }
  }
}

template <typename T>
void prod_grad(const Tensor& x,
               const Tensor& out,
               const Tensor& out_grad,
               const IntArray& axis,
               bool keep_dim,
               bool reduce_all,
               Tensor* x_grad) {
  if (x_grad) {
    int64_t axis_size = axis.size();
    int64_t x_dim_size = x.dims().size();
    reduce_all = false;
    if (reduce_all || axis_size == 0 || axis_size == x_dim_size) {
      reduce_all = true;
    } else {
      reduce_all = false;
    }
    auto out_grad_tmp = Tensor();
    auto x_reshape = Tensor();
    if (has_dynamic_shape(x.shape())) {
      Tensor x_dim = shape64<T>(x);
      std::vector<int64_t> unchange_axis, change_axis;
      std::vector<int> transpose_dim, origin_position;
      std::vector<Tensor> transpose_shape, cumprod_shape;
      if (x_dim_size == 1) {
        out_grad_tmp = backend::expand<T>(out_grad, x_dim);
      } else {
        if (!keep_dim) {
          auto axis_ = std::vector<int64_t>();
          if (reduce_all) {
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
          Tensor out_grad_shape =
              get_unsqueeze_dims<T>(shape64<T>(out_grad), axis_);
          Tensor out_grad_ = backend::reshape<T>(out_grad, out_grad_shape);
          out_grad_tmp = backend::expand<T>(out_grad_, x_dim);
        } else {
          out_grad_tmp = backend::expand<T>(out_grad, x_dim);
        }
      }
      if (reduce_all) {
        Tensor numel = full<T>({1}, 1.0, x_dim.dtype(), x.place());
        for (int64_t i = 0; i < x_dim_size; i++) {
          numel = numel * get_slice<T>(x_dim, i);
        }
        cumprod_shape.push_back(numel);
        x_reshape = backend::reshape<T>(x, concat<T>(cumprod_shape));
        Tensor left_cumprod = cumprod<T>(x_reshape, -1, true, false);
        Tensor right_cumprod = cumprod<T>(x_reshape, -1, true, true);
        Tensor x_grad_tmp = left_cumprod * right_cumprod;
        Tensor x_grad_tmp2 = backend::reshape<T>(x_grad_tmp, x_dim);
        Tensor x_grad_res = x_grad_tmp2 * out_grad_tmp;
        set_output<T>(x_grad_res, x_grad);
      } else {
        auto axis_ = std::vector<int64_t>();
        int64_t unchange_size = x_dim_size - axis_size;
        int64_t unchange_index = 0;
        for (int64_t i = 0; i < axis_size; i++) {
          if (axis[i] < 0) {
            axis_.push_back(axis[i] + x_dim_size);
          } else {
            axis_.push_back(axis[i]);
          }
        }
        for (int64_t i = 0; i < x_dim_size; i++) {
          auto it = find(axis_.begin(), axis_.end(), i);
          if (it != axis_.end()) {
            int64_t index = it - axis_.begin();
            origin_position.push_back(static_cast<int>(unchange_size + index));
          } else {
            unchange_axis.push_back(i);
            origin_position.push_back(static_cast<int>(unchange_index));
            unchange_index += 1;
          }
        }
        Tensor numel = full<T>({1}, 1.0, x_dim.dtype(), x.place());
        for (int64_t i = 0; i < unchange_size; i++) {
          transpose_shape.push_back(get_slice<T>(x_dim, unchange_axis[i]));
          cumprod_shape.push_back(get_slice<T>(x_dim, unchange_axis[i]));
          transpose_dim.push_back(static_cast<int>(unchange_axis[i]));
        }
        for (int64_t i = 0; i < axis_size; i++) {
          transpose_shape.push_back(get_slice<T>(x_dim, axis_[i]));
          transpose_dim.push_back(static_cast<int>(axis_[i]));
          numel = numel * get_slice<T>(x_dim, axis_[i]);
        }
        cumprod_shape.push_back(numel);
        Tensor x_transpose = transpose<T>(x, transpose_dim);
        x_reshape = backend::reshape<T>(x_transpose, concat<T>(cumprod_shape));
        Tensor left_cumprod = cumprod<T>(x_reshape, -1, true, false);
        Tensor right_cumprod = cumprod<T>(x_reshape, -1, true, true);
        Tensor x_grad_tmp = left_cumprod * right_cumprod;
        Tensor x_grad_reshape =
            backend::reshape<T>(x_grad_tmp, concat<T>(transpose_shape));
        Tensor x_grad_tmp2 = transpose<T>(x_grad_reshape, origin_position);
        Tensor x_grad_res = x_grad_tmp2 * out_grad_tmp;
        set_output<T>(x_grad_res, x_grad);
      }
    } else {
      std::vector<int64_t> x_dim = common::vectorize<int64_t>(x.dims());
      std::vector<int64_t> unchange_axis, change_axis, transpose_shape,
          cumprod_shape;
      std::vector<int> transpose_dim, origin_position;
      if (x_dim_size == 1) {
        out_grad_tmp = out_grad.expand(IntArray(x_dim));
      } else {
        if (!keep_dim) {
          auto axis_ = std::vector<int64_t>();
          if (reduce_all) {
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
          auto out_grad_shape = get_unsqueeze_dims(out_grad, axis_);
          auto out_grad_ = reshape<T>(out_grad, out_grad_shape);
          out_grad_tmp = out_grad_.expand(IntArray(x_dim));
        } else {
          out_grad_tmp = out_grad.expand(IntArray(x_dim));
        }
      }
      if (reduce_all) {
        int64_t numel = 1;
        for (int64_t i = 0; i < x_dim_size; i++) {
          numel *= x_dim[i];
        }
        cumprod_shape.push_back(numel);
        x_reshape = reshape<T>(x, cumprod_shape);
        auto left_cumprod = cumprod<T>(x_reshape, -1, true, false);
        auto right_cumprod = cumprod<T>(x_reshape, -1, true, true);
        auto x_grad_tmp = left_cumprod * right_cumprod;
        auto x_grad_tmp2 = reshape<T>(x_grad_tmp, x.shape());
        auto x_grad_res = x_grad_tmp2 * out_grad_tmp;
        set_output<T>(x_grad_res, x_grad);
      } else {
        auto axis_ = std::vector<int64_t>();
        int64_t unchange_size = x_dim_size - axis_size;
        int64_t unchange_index = 0;
        for (int64_t i = 0; i < axis_size; i++) {
          if (axis[i] < 0) {
            axis_.push_back(axis[i] + x_dim_size);
          } else {
            axis_.push_back(axis[i]);
          }
        }
        for (int64_t i = 0; i < x_dim_size; i++) {
          auto it = find(axis_.begin(), axis_.end(), i);
          if (it != axis_.end()) {
            int64_t index = it - axis_.begin();
            origin_position.push_back(static_cast<int>(unchange_size + index));
          } else {
            unchange_axis.push_back(i);
            origin_position.push_back(static_cast<int>(unchange_index));
            unchange_index += 1;
          }
        }
        int64_t numel = 1;
        for (int64_t i = 0; i < unchange_size; i++) {
          transpose_shape.push_back(x_dim[unchange_axis[i]]);
          cumprod_shape.push_back(x_dim[unchange_axis[i]]);
          transpose_dim.push_back(static_cast<int>(unchange_axis[i]));
        }
        for (int64_t i = 0; i < axis_size; i++) {
          transpose_shape.push_back(x_dim[axis_[i]]);
          transpose_dim.push_back(static_cast<int>(axis_[i]));
          numel *= x_dim[axis_[i]];
        }
        cumprod_shape.push_back(numel);
        auto x_transpose = transpose<T>(x, transpose_dim);
        x_reshape = reshape<T>(x_transpose, cumprod_shape);
        auto left_cumprod = cumprod<T>(x_reshape, -1, true, false);
        auto right_cumprod = cumprod<T>(x_reshape, -1, true, true);
        auto x_grad_tmp = left_cumprod * right_cumprod;
        auto x_grad_reshape = reshape<T>(x_grad_tmp, transpose_shape);
        auto x_grad_tmp2 = transpose<T>(x_grad_reshape, origin_position);
        auto x_grad_res = x_grad_tmp2 * out_grad_tmp;
        set_output<T>(x_grad_res, x_grad);
      }
    }
  }
}

template <typename T>
void minimum_grad(const Tensor& x,
                  const Tensor& y,
                  const Tensor& out_grad,
                  Tensor* x_grad,
                  Tensor* y_grad) {
  if (x_grad) {
    auto x_tmp = cast<T>(less_than<T>(x, y), out_grad.dtype());
    auto dx_res = out_grad * x_tmp;
    if (has_dynamic_shape(x.shape()) || has_dynamic_shape(out_grad.shape()) ||
        out_grad.dims() != x.dims()) {
      auto dx_reduce_res = reduce_as<T>(dx_res, x);
      set_output<T>(dx_reduce_res, x_grad);
    } else {
      set_output<T>(dx_res, x_grad);
    }
  }

  if (y_grad) {
    auto y_tmp = cast<T>(greater_equal<T>(x, y), out_grad.dtype());
    auto dy_res = out_grad * y_tmp;
    if (has_dynamic_shape(y.shape()) || has_dynamic_shape(out_grad.shape()) ||
        out_grad.dims() != y.dims()) {
      auto dy_reduce_res = reduce_as<T>(dy_res, y);
      set_output<T>(dy_reduce_res, y_grad);
    } else {
      set_output<T>(dy_res, y_grad);
    }
  }
}

template <typename T>
void group_norm_grad(const Tensor& x,
                     const paddle::optional<Tensor>& scale,
                     const paddle::optional<Tensor>& bias,
                     const Tensor& y,
                     const Tensor& mean,
                     const Tensor& variance,
                     const Tensor& out_grad,
                     float epsilon,
                     int groups,
                     const std::string& data_layout,
                     Tensor* x_grad,
                     Tensor* scale_grad,
                     Tensor* bias_grad) {
  GroupNormDecompHelper<T> decomp_helper(x, scale, bias, groups, data_layout);
  const std::vector<int64_t>& scale_bias_new_shape =
      decomp_helper.GetScaleBiasNewShape();

  std::vector<int64_t> x_dims = x.shape();
  int rank = x_dims.size();
  if (rank < 3) {
    PADDLE_THROW(common::errors::Unimplemented(
        "Only support NCHW and NHWC format in rank higher or equal to 3. "
        "Current rank: %zu",
        rank));
  }

  Tensor x_data = ConvertToMT<T>(x);
  Tensor out_grad_data = ConvertToMT<T>(out_grad);

  std::cerr << "x data " << x_data.dims() << std::endl;
  x_data = decomp_helper.Split(x_data);
  std::cerr << "new data " << x_data.dims() << std::endl;
  out_grad_data = decomp_helper.Split(out_grad_data);

  const auto& reduce_axis = decomp_helper.GetReduceAxis();
  std::cerr << "var " << variance.dims() << std::endl;

  const auto& squeeze_axis = decomp_helper.GetMeanVarSqueezeAxis();
  for (auto d : squeeze_axis) {
    std::cerr << "dd  " << d << std::endl;
  }
  auto variance_new = unsqueeze<T>(variance, squeeze_axis);
  auto mean_new = unsqueeze<T>(mean, squeeze_axis);

  std::cerr << "var new " << variance_new.dims() << std::endl;
  for (auto& d : reduce_axis) {
    std::cerr << "reduc axis " << d << std::endl;
  }

  Tensor scale_data;
  if (scale) {
    scale_data = reshape<T>(scale.get(), scale_bias_new_shape);
  }
  Tensor bias_data;
  if (bias) {
    bias_data = reshape<T>(bias.get(), scale_bias_new_shape);
  }

  auto x_sub_mean = x_data - mean_new;
  auto tmp =
      (full_scalar<T>(1.0, variance_new.dtype()) /
       (variance_new + full_scalar<T>(epsilon, variance_new.dtype())));  // M,1
  auto sqrt_var_1 = sqrt<T>(tmp);                                        // M,1
  auto x_sub_mean_mul_sqrt_var_1 = x_sub_mean * sqrt_var_1;

  if (x_grad) {
    auto out_grad_scale = out_grad_data;  // M,N
    if (scale) {
      out_grad_scale = out_grad_data * scale_data;  // M,N * 1,N = M,N
    }

    auto dx_end = sqrt_var_1 * out_grad_scale;
    auto d_mean = dx_end.sum(reduce_axis, x_data.dtype(), true);  // M,1

    auto d_std_1 = (tmp * x_sub_mean * out_grad_scale)
                       .sum(reduce_axis, x_data.dtype(), true);  // M,1
    auto d_std = d_std_1 * x_sub_mean_mul_sqrt_var_1;  // M,1 * M,N = M,N

    auto d_mean_d_std = (d_mean + d_std) / decomp_helper.GetHW(x_data);

    auto x_grad_tmp = dx_end - d_mean_d_std;
    x_grad_tmp = ConvertToMT<T>(x_grad_tmp, x.dtype());
    x_grad_tmp = decomp_helper.Merge(x_grad_tmp);
    set_output<T>(x_grad_tmp, x_grad);
  }

  auto reduce_axis_except_channel = decomp_helper.GetReduceAxisExceptChannel();
  if (scale_grad) {
    if (scale) {
      auto third_shape = get_unsqueeze_dims(mean, {2});
      auto tmp1 = out_grad_data * (x_data - mean_new) * sqrt_var_1;

      auto scale_grad_tmp = reshape<T>(
          tmp1.sum(reduce_axis_except_channel, scale->dtype(), false), {-1});
      set_output<T>(scale_grad_tmp, scale_grad);
    }
  }

  if (bias_grad) {
    if (bias) {
      auto bias_grad_tmp =
          out_grad_data.sum(reduce_axis_except_channel, bias->dtype(), false);

      set_output<T>(reshape<T>(bias_grad_tmp, {-1}), bias_grad);
    }
  }
}

template <typename T>
void swiglu_grad(const Tensor& x,
                 const paddle::optional<Tensor>& y,
                 const Tensor& dz,
                 Tensor* dx,
                 Tensor* dy) {
  const auto& x_shape = x.shape();
  auto one_tensor = full_scalar<T>(1.0, x.dtype());
  Tensor x_grad;
  if (y) {
    const auto& y_tensor = y.get();
    Tensor sig = sigmoid<T>(x);
    Tensor tmp = sig * x;
    x_grad = dz * y_tensor * sig * (one_tensor + x - tmp);
    Tensor y_grad = dz * tmp;
    set_output<T>(y_grad, dy);
  } else {
    int axis = x.shape().size() - 1;
    int num = 2;
    std::vector<Tensor> xs = backend::split_with_num<T>(x, num, axis);
    Tensor sig = sigmoid<T>(xs[0]);
    Tensor tmp = sig * xs[0];
    Tensor x0_grad = dz * xs[1] * sig * (one_tensor + xs[0] - tmp);
    Tensor x1_grad = dz * tmp;
    int64_t c_axis = x_shape.size() - 1;
    x_grad = concat<T>({x0_grad, x1_grad}, c_axis);
  }
  set_output<T>(x_grad, dx);
}

template <typename T>
void softsign_grad(const Tensor& x, const Tensor& out_grad, Tensor* x_grad) {
  // x_grad = out_grad / ((1 + abs(x))^2)

  if (x_grad) {
    Tensor x_abs = abs<T>(x);
    Tensor x_abs_plusone = x_abs + full_scalar<T>(1.0, x.dtype());
    Tensor x_grad_tmp = out_grad / (x_abs_plusone * x_abs_plusone);
    set_output<T>(x_grad_tmp, x_grad);
  }
}

template <typename T>
void where_grad(const Tensor& condition,
                const Tensor& x,
                const Tensor& y,
                const Tensor& out_grad,
                Tensor* x_grad,
                Tensor* y_grad) {
  Tensor zero;
  if (has_dynamic_shape(out_grad.shape())) {
    zero = backend::full_with_tensor<T>(
        shape64<T>(out_grad), 0.0, out_grad.dtype(), out_grad.place());
  } else {
    zero = full<T>(common::vectorize(out_grad.dims()),
                   0.0,
                   out_grad.dtype(),
                   out_grad.place());
  }

  if (x_grad) {
    Tensor x_grad_tmp = where<T>(condition, out_grad, zero);
    set_output<T>(x_grad_tmp, x_grad);
  }
  if (y_grad) {
    Tensor y_grad_tmp = where<T>(condition, zero, out_grad);
    set_output<T>(y_grad_tmp, y_grad);
  }
}

template <typename T>
void expm1_grad(const Tensor& out, const Tensor& out_grad, Tensor* x_grad) {
  if (x_grad) {
    Tensor x_grad_tmp = out_grad * out + out_grad;
    set_output<T>(x_grad_tmp, x_grad);
  }
}

template <typename T>
void atan2_grad(const Tensor& x,
                const Tensor& y,
                const Tensor& out_grad,
                Tensor* x_grad,
                Tensor* y_grad) {
  Tensor tmp = x * x + y * y;
  if (x_grad) {
    Tensor x_grad_tmp = (out_grad * y) / tmp;
    set_output<T>(x_grad_tmp, x_grad);
  }

  if (y_grad) {
    Tensor y_grad_tmp = (-out_grad * x) / tmp;
    set_output<T>(y_grad_tmp, y_grad);
  }
}

template <typename T>
void put_along_axis_grad(const Tensor& x,
                         const Tensor& index,
                         const Tensor& value,
                         const Tensor& out,
                         const Tensor& out_grad,
                         int axis,
                         const std::string& reduce,
                         bool include_self,
                         Tensor* x_grad,
                         Tensor* value_grad) {
  if (x_grad) {
    Tensor x_grad_tmp = out_grad;
    if (include_self == false || reduce == "assign") {
      Tensor zero_tensor =
          full<T>(index.shape(), 0, out_grad.dtype(), out_grad.place());
      x_grad_tmp = put_along_axis<T>(out_grad, index, zero_tensor, axis);
      set_output<T>(x_grad_tmp, x_grad);
    } else if (reduce == "multiply" || reduce == "mul") {
      Tensor zero_tensor_x = full<T>(x.shape(), 0, x.dtype(), x.place());
      Tensor one_tensor_idx = full<T>(index.shape(), 1, x.dtype(), x.place());
      Tensor mask =
          put_along_axis<T>(zero_tensor_x, index, one_tensor_idx, axis);
      x_grad_tmp = where<T>(mask > zero_tensor_x, out_grad * out / x, out_grad);
      set_output<T>(x_grad_tmp, x_grad);
    } else if (reduce == "amin" || reduce == "amax") {
      Tensor zero_tensor = full<T>(x.shape(), 0, x.dtype(), x.place());
      Tensor one_tensor = full<T>(x.shape(), 1, x.dtype(), x.place());

      auto zero_result = cast<T>(equal<T>(out, x), x.dtype());

      Tensor num = zero_tensor;
      int64_t select_num = static_cast<int64_t>(index.shape()[axis]);
      for (int64_t i = 0; i < select_num; i++) {
        Tensor sub_index = slice<T>(index, {axis}, {i}, {i + 1}, {1}, {});
        Tensor sub_value = slice<T>(value, {axis}, {i}, {i + 1}, {1}, {});
        Tensor sub_out = take_along_axis<T>(out, sub_index, axis);
        Tensor sub_count = cast<T>(equal<T>(sub_out, sub_value), x.dtype());
        num = num + put_along_axis<T>(zero_tensor, sub_index, sub_count, axis);
      }
      x_grad_tmp = zero_result * out_grad / (num + 1);
      set_output<T>(x_grad_tmp, x_grad);
    } else if (reduce == "mean") {
      Tensor zero_tensor_x = full<T>(x.shape(), 0, x.dtype(), x.place());

      Tensor num = zero_tensor_x;
      int64_t select_num = static_cast<int64_t>(index.shape()[axis]);
      for (int64_t i = 0; i < select_num; i++) {
        Tensor sub_index = slice<T>(index, {axis}, {i}, {i + 1}, {1}, {});
        Tensor sub_one_tensor =
            full<T>(sub_index.shape(), 1, x.dtype(), x.place());
        num = num +
              put_along_axis<T>(zero_tensor_x, sub_index, sub_one_tensor, axis);
      }
      x_grad_tmp =
          where<T>(num > zero_tensor_x, out_grad / (num + 1), out_grad);
      set_output<T>(x_grad_tmp, x_grad);
    } else if (reduce == "add") {
      by_pass<T>(out_grad, x_grad);
    }
  }

  if (value_grad) {
    Tensor value_grad_tmp = full<T>(index.shape(), 0, x.dtype(), x.place());
    if (reduce == "assign") {
      int64_t select_num = static_cast<int64_t>(index.shape()[axis]);
      Tensor mask =
          full<T>(out_grad.shape(), 1, out_grad.dtype(), out_grad.place());
      Tensor zero =
          full<T>(out_grad.shape(), 0, out_grad.dtype(), out_grad.place());
      std::vector<Tensor> res(select_num);
      for (int64_t i = select_num - 1; i >= 0; i--) {
        Tensor sub_index = slice<T>(index, {axis}, {i}, {i + 1}, {1}, {});
        Tensor tmp_grad = out_grad * mask;
        res[i] = take_along_axis<T>(tmp_grad, sub_index, axis);
        mask = put_along_axis<T>(out_grad, sub_index, zero, axis);
      }
      value_grad_tmp = concat<T>(res, axis);
    } else if (reduce == "add") {
      value_grad_tmp = take_along_axis<T>(out_grad, index, axis);
    } else if (reduce == "mean") {
      Tensor one_tensor =
          full<T>(out_grad.shape(), 1, out_grad.dtype(), out_grad.place());
      Tensor zero_tensor =
          full<T>(out_grad.shape(), 0, out_grad.dtype(), out_grad.place());
      Tensor num = include_self ? one_tensor : zero_tensor;
      int64_t select_num = static_cast<int64_t>(index.shape()[axis]);
      for (int64_t i = 0; i < select_num; i++) {
        Tensor sub_index = slice<T>(index, {axis}, {i}, {i + 1}, {1}, {});
        num = num + put_along_axis<T>(zero_tensor, sub_index, one_tensor, axis);
      }
      Tensor grad_result = out_grad / num;
      value_grad_tmp = take_along_axis<T>(grad_result, index, axis);
    } else if (reduce == "mul" || reduce == "multiply") {
      Tensor out_grad_select = take_along_axis<T>(out_grad, index, axis);
      Tensor out_select = take_along_axis<T>(out, index, axis);
      value_grad_tmp = out_grad_select * (out_select / value);
    } else if (reduce == "amin" || reduce == "amax") {
      Tensor one_tensor_out =
          full<T>(out_grad.shape(), 1, out_grad.dtype(), out_grad.place());
      Tensor zero_tensor_out =
          full<T>(out_grad.shape(), 0, out_grad.dtype(), out_grad.place());
      Tensor num = zero_tensor_out;
      int64_t select_num = static_cast<int64_t>(index.shape()[axis]);
      for (int64_t i = 0; i < select_num; i++) {
        Tensor sub_index = slice<T>(index, {axis}, {i}, {i + 1}, {1}, {});
        Tensor sub_value = slice<T>(value, {axis}, {i}, {i + 1}, {1}, {});
        Tensor one_tensor_idx =
            full<T>(sub_index.shape(), 1, out_grad.dtype(), out_grad.place());
        Tensor sub_mask =
            put_along_axis<T>(zero_tensor_out, sub_index, one_tensor_idx, axis);
        Tensor sub_put_res =
            put_along_axis<T>(zero_tensor_out, sub_index, sub_value, axis);
        num = num +
              cast<T>(equal<T>(out, sub_put_res), out_grad.dtype()) * sub_mask;
      }
      Tensor select_out = take_along_axis<T>(out, index, axis);
      Tensor mask = cast<T>(equal<T>(select_out, value), out_grad.dtype());
      Tensor select_out_grad = take_along_axis<T>(out_grad, index, axis);
      Tensor select_cnt = take_along_axis<T>(num, index, axis);
      Tensor select_x = take_along_axis<T>(x, index, axis);
      Tensor res = where<T>(select_out == select_x,
                            select_out_grad / (select_cnt + 1),
                            select_out_grad / select_cnt);
      value_grad_tmp = res * mask;
    }
    set_output<T>(value_grad_tmp, value_grad);
  }
}

template <typename T>
void atan_grad(const Tensor& x, const Tensor& out_grad, Tensor* x_grad) {
  if (x_grad) {
    const Tensor one = full_scalar<T>(1.0, x.dtype());
    Tensor x_grad_tmp = out_grad / (one + x * x);
    set_output<T>(x_grad_tmp, x_grad);
  }
}

template <typename T>
void swish_grad(const Tensor& x, const Tensor& out_grad, Tensor* x_grad) {
  if (x_grad) {
    const Tensor one = full_scalar<T>(1.0, x.dtype());
    const Tensor sig = sigmoid<T>(x);
    Tensor res = out_grad * sig * (one + x * (one - sig));
    set_output<T>(res, x_grad);
  }
}

template <typename T>
void fmax_grad(const Tensor& x,
               const Tensor& y,
               const Tensor& out_grad,
               Tensor* x_grad,
               Tensor* y_grad) {
  const Tensor nan_x = isnan<T>(x);
  const Tensor nan_y = isnan<T>(y);
  Tensor mask_x = backend::logical_or<T>(nan_y, greater_equal<T>(x, y));
  Tensor mask_y = backend::logical_not<T>(mask_x);

  if (x_grad) {
    Tensor dx = cast<T>(mask_x, out_grad.dtype()) * out_grad;
    if (has_dynamic_shape(x.shape()) || has_dynamic_shape(out_grad.shape())) {
      dx = reduce_as<T>(dx, x);
    } else {
      if (out_grad.dims() != x.dims()) {
        auto reduce_dim = get_reduce_dims_from_out(out_grad.dims(), x.dims());
        Tensor dx_reduce_res =
            dx.sum(common::vectorize(reduce_dim), x.dtype(), false);
        dx = reshape<T>(dx_reduce_res, common::vectorize(x.dims()));
      }
    }
    set_output<T>(dx, x_grad);
  }

  if (y_grad) {
    Tensor dy = cast<T>(mask_y, out_grad.dtype()) * out_grad;
    if (has_dynamic_shape(y.shape()) || has_dynamic_shape(out_grad.shape())) {
      dy = reduce_as<T>(dy, x);
    } else {
      if (out_grad.dims() != y.dims()) {
        auto reduce_dim = get_reduce_dims_from_out(out_grad.dims(), y.dims());
        Tensor dy_reduce_res =
            dy.sum(common::vectorize(reduce_dim), y.dtype(), false);
        dy = reshape<T>(dy_reduce_res, common::vectorize(y.dims()));
      }
    }
    set_output<T>(dy, y_grad);
  }
}

template <typename T>
void fmin_grad(const Tensor& x,
               const Tensor& y,
               const Tensor& out_grad,
               Tensor* x_grad,
               Tensor* y_grad) {
  const Tensor nan_x = isnan<T>(x);
  const Tensor nan_y = isnan<T>(y);
  Tensor mask_x = backend::logical_or<T>(nan_y, less_equal<T>(x, y));
  Tensor mask_y = backend::logical_not<T>(mask_x);

  if (x_grad) {
    Tensor dx = cast<T>(mask_x, out_grad.dtype()) * out_grad;
    if (has_dynamic_shape(x.shape()) || has_dynamic_shape(out_grad.shape())) {
      dx = reduce_as<T>(dx, x);
    } else {
      if (out_grad.dims() != x.dims()) {
        auto reduce_dim = get_reduce_dims_from_out(out_grad.dims(), x.dims());
        Tensor dx_reduce_res =
            dx.sum(common::vectorize(reduce_dim), x.dtype(), false);
        dx = reshape<T>(dx_reduce_res, common::vectorize(x.dims()));
      }
    }
    set_output<T>(dx, x_grad);
  }

  if (y_grad) {
    Tensor dy = cast<T>(mask_y, out_grad.dtype()) * out_grad;
    if (has_dynamic_shape(y.shape()) || has_dynamic_shape(out_grad.shape())) {
      dy = reduce_as<T>(dy, x);
    } else {
      if (out_grad.dims() != y.dims()) {
        auto reduce_dim = get_reduce_dims_from_out(out_grad.dims(), y.dims());
        Tensor dy_reduce_res =
            dy.sum(common::vectorize(reduce_dim), y.dtype(), false);
        dy = reshape<T>(dy_reduce_res, common::vectorize(y.dims()));
      }
    }
    set_output<T>(dy, y_grad);
  }
}

template <typename T>
void dot_grad(const Tensor& x,
              const Tensor& y,
              const Tensor& out_grad,
              Tensor* x_grad,
              Tensor* y_grad) {
  const int64_t out_grad_dim_size = out_grad.dims().size();
  Tensor out_grad_ = out_grad;

  if (has_dynamic_shape(x.shape()) || has_dynamic_shape(y.shape())) {
    auto out_grad_shape =
        get_unsqueeze_dims<T>(shape64<T>(out_grad_), {out_grad_dim_size});
    out_grad_ = backend::reshape<T>(out_grad_, out_grad_shape);
    out_grad_ = backend::expand<T>(out_grad_, shape64<T>(x));
  } else {
    std::vector<int64_t> x_dim = common::vectorize<int64_t>(x.dims());
    auto out_grad_shape = get_unsqueeze_dims(out_grad, {out_grad_dim_size});
    out_grad_ =
        expand<T>(reshape<T>(out_grad_, out_grad_shape), IntArray(x_dim));
  }

  if (x_grad) {
    Tensor x_grad_tmp = out_grad_ * y;
    set_output<T>(x_grad_tmp, x_grad);
  }

  if (y_grad) {
    Tensor y_grad_tmp = out_grad_ * x;
    set_output<T>(y_grad_tmp, y_grad);
  }
}

template <typename T>
void logcumsumexp_grad(const Tensor& x,
                       const Tensor& out,
                       const Tensor& out_grad,
                       int axis,
                       bool flatten,
                       bool exclusive,
                       bool reverse,
                       Tensor* x_grad) {
  if (x_grad) {
    reverse = !reverse;
    Tensor tmp, lowest, x_grad_tmp;
    Tensor x_cast = ConvertToMT<T>(x);
    Tensor out_cast = ConvertToMT<T>(out);
    Tensor out_grad_cast = ConvertToMT<T>(out_grad);

    const Tensor out_grad_log = log<T>(abs<T>(out_grad_cast));
    auto out_grad_dtype = out_grad_cast.dtype();

    if (has_dynamic_shape(x_cast.shape()) ||
        has_dynamic_shape(out_grad_cast.shape())) {
      const Tensor x_shape = shape64<T>(x_cast);
      const Tensor out_grad_shape = shape64<T>(out_grad_cast);
      const Tensor reshape_x = backend::reshape<T>(x_cast, out_grad_shape);

      if (out_grad_dtype == DataType::FLOAT32) {
        lowest =
            backend::full_with_tensor<T>(out_grad_shape,
                                         std::numeric_limits<float>::lowest(),
                                         out_grad_dtype,
                                         out_grad.place());
      } else if (out_grad_dtype == DataType::FLOAT64) {
        lowest =
            backend::full_with_tensor<T>(out_grad_shape,
                                         std::numeric_limits<double>::lowest(),
                                         out_grad_dtype,
                                         out_grad.place());
      }
      const Tensor zero = backend::full_with_tensor<T>(
          out_grad_shape, 0.0, out_grad_dtype, out_grad.place());

      // compute positive
      Tensor out_grad_pos =
          where<T>(out_grad_cast > zero, out_grad_log, lowest);
      tmp = out_grad_pos - out_cast;
      out_grad_pos = logcumsumexp<T>(tmp, axis, flatten, exclusive, reverse);
      out_grad_pos = exp<T>(out_grad_pos + reshape_x);

      // compute negative
      Tensor out_grad_neg =
          where<T>(out_grad_cast < zero, out_grad_log, lowest);
      tmp = out_grad_neg - out_cast;
      out_grad_neg = logcumsumexp<T>(tmp, axis, flatten, exclusive, reverse);
      out_grad_neg = exp<T>(out_grad_neg + reshape_x);

      x_grad_tmp = backend::reshape<T>(out_grad_pos - out_grad_neg, x_shape);
    } else {
      const Tensor reshape_x = reshape<T>(x_cast, out_grad_cast.shape());
      if (out_grad_dtype == DataType::FLOAT32) {
        lowest = full<T>(out_grad_cast.shape(),
                         std::numeric_limits<float>::lowest(),
                         out_grad_dtype,
                         out_grad_cast.place());
      } else if (out_grad_dtype == DataType::FLOAT64) {
        lowest = full<T>(out_grad_cast.shape(),
                         std::numeric_limits<double>::lowest(),
                         out_grad_dtype,
                         out_grad_cast.place());
      }
      const Tensor zero = full<T>(
          out_grad_cast.shape(), 0.0, out_grad_dtype, out_grad_cast.place());

      // compute positive
      Tensor out_grad_pos =
          where<T>(out_grad_cast > zero, out_grad_log, lowest);
      tmp = out_grad_pos - out_cast;
      out_grad_pos = logcumsumexp<T>(tmp, axis, flatten, exclusive, reverse);
      out_grad_pos = exp<T>(out_grad_pos + reshape_x);

      // compute negative
      Tensor out_grad_neg =
          where<T>(out_grad_cast < zero, out_grad_log, lowest);
      tmp = out_grad_neg - out_cast;
      out_grad_neg = logcumsumexp<T>(tmp, axis, flatten, exclusive, reverse);
      out_grad_neg = exp<T>(out_grad_neg + reshape_x);

      x_grad_tmp = reshape<T>(out_grad_pos - out_grad_neg, x_cast.shape());
    }

    set_output<T>(ConvertToOrig<T>(x_grad_tmp, x.dtype()), x_grad);
  }
}

template <typename T>
void logsumexp_grad(const Tensor& x,
                    const Tensor& out,
                    const Tensor& out_grad,
                    const IntArray& axis,
                    bool keepdim,
                    bool reduce_all,
                    Tensor* x_grad) {
  if (x_grad) {
    int64_t axis_size = axis.size();
    int64_t x_dim_size = x.dims().size();
    reduce_all = false;

    if (reduce_all || axis_size == 0 || axis_size == x_dim_size) {
      reduce_all = true;
    } else {
      reduce_all = false;
    }

    auto x_grad_tmp = Tensor();

    if (has_dynamic_shape(x.shape())) {
      Tensor x_shape = shape64<T>(x);
      if (x_dim_size == 1) {
        x_grad_tmp = backend::expand<T>(out_grad, x_shape) * exp<T>(x - out);
      } else {
        if (!keepdim) {
          auto axis_ = std::vector<int64_t>();
          if (reduce_all) {
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

          auto result_shape =
              get_unsqueeze_dims<T>(shape64<T>(out_grad), axis_);
          auto out_ = backend::reshape<T>(out, result_shape);
          auto softmax = exp<T>(x - backend::expand<T>(out_, x_shape));

          auto out_grad_ = backend::reshape<T>(out_grad, result_shape);
          x_grad_tmp = backend::expand<T>(out_grad_, x_shape) * softmax;
        } else {
          x_grad_tmp = backend::expand<T>(out_grad, x_shape) * exp<T>(x - out);
        }
      }
    } else {
      std::vector<int64_t> x_dim = common::vectorize<int64_t>(x.dims());
      if (x_dim_size == 1) {
        x_grad_tmp = expand<T>(out_grad, IntArray(x_dim)) * exp<T>(x - out);
      } else {
        if (!keepdim) {
          auto axis_ = std::vector<int64_t>();
          if (reduce_all) {
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
          auto out_shape = get_unsqueeze_dims(out, axis_);
          auto out_ = reshape<T>(out, out_shape);
          auto softmax = exp<T>(x - expand<T>(out_, IntArray(x_dim)));

          auto out_grad_shape = get_unsqueeze_dims(out_grad, axis_);
          auto out_grad_ = reshape<T>(out_grad, out_grad_shape);
          x_grad_tmp = expand<T>(out_grad_, IntArray(x_dim)) * softmax;
        } else {
          x_grad_tmp = expand<T>(out_grad, IntArray(x_dim)) * exp<T>(x - out);
        }
      }
    }
    set_output<T>(x_grad_tmp, x_grad);
  }
}

template <typename T>
void trunc_grad(const Tensor& out_grad, Tensor* x_grad) {
  Tensor zero;
  if (x_grad) {
    if (has_dynamic_shape(out_grad.shape())) {
      zero = backend::full_with_tensor<T>(
          shape64<T>(out_grad), 0.0, out_grad.dtype(), out_grad.place());
    } else {
      zero = full<T>(out_grad.shape(), 0.0, out_grad.dtype(), out_grad.place());
    }
    set_output<T>(zero, x_grad);
  }
}

template <typename T>
void kthvalue_grad(const Tensor& x,
                   const Tensor& indices,
                   const Tensor& out_grad,
                   int k,
                   int axis,
                   bool keepdim,
                   Tensor* x_grad) {
  if (x_grad) {
    auto x_cast = ConvertToMT<T>(x);
    auto out_grad_cast = ConvertToMT<T>(out_grad);
    // put_along_axis doesn't support zero dim
    if (x.dims().size() == 0) {
      by_pass<T>(out_grad, x_grad);
      return;
    }

    // function `put_along_axis` requires a non-negative axis
    if (axis < 0) {
      axis += x.dims().size();
    }

    Tensor zero_tensor;
    Tensor x_grad_tmp;
    if (has_dynamic_shape(x_cast.shape())) {
      zero_tensor = backend::full_with_tensor<T>(
          shape64<T>(x_cast), 0, x_cast.dtype(), x_cast.place());

      if (keepdim) {
        x_grad_tmp = backend::put_along_axis<T>(
            zero_tensor, indices, out_grad_cast, axis);
      } else {
        auto axis_ = std::vector<int64_t>(1, axis);
        auto out_grad_shape =
            get_unsqueeze_dims<T>(shape64<T>(out_grad_cast), axis_);
        auto out_grad_ = backend::reshape<T>(out_grad_cast, out_grad_shape);
        auto indices_shape = get_unsqueeze_dims<T>(shape64<T>(indices), axis_);
        auto indices_ = backend::reshape<T>(indices, indices_shape);
        x_grad_tmp =
            backend::put_along_axis<T>(zero_tensor, indices_, out_grad_, axis);
      }
    } else {
      zero_tensor = full<T>(
          common::vectorize(x_cast.dims()), 0, x_cast.dtype(), x_cast.place());
      if (keepdim) {
        x_grad_tmp =
            put_along_axis<T>(zero_tensor, indices, out_grad_cast, axis);
      } else {
        auto axis_ = std::vector<int64_t>(1, axis);
        auto out_grad_shape = get_unsqueeze_dims(out_grad_cast, axis_);
        auto out_grad_ = reshape<T>(out_grad_cast, out_grad_shape);
        auto indices_shape = get_unsqueeze_dims(indices, axis_);
        auto indices_ = reshape<T>(indices, indices_shape);
        x_grad_tmp = put_along_axis<T>(zero_tensor, indices_, out_grad_, axis);
      }
    }
    set_output<T>(ConvertToOrig<T>(x_grad_tmp, x.dtype()), x_grad);
  }
}

template <typename T>
void argsort_grad(const Tensor& indices,
                  const Tensor& x,
                  const Tensor& out_grad,
                  int axis,
                  bool descending,
                  bool stable,
                  Tensor* x_grad) {
  if (x_grad) {
    auto indices_cast = ConvertToMT<T>(indices);
    auto x_cast = ConvertToMT<T>(x);
    auto out_grad_cast = ConvertToMT<T>(out_grad);

    if (axis < 0) {
      axis += x_cast.dims().size();
    }
    Tensor zero_tensor;
    auto x_grad_tmp = Tensor();
    if (has_dynamic_shape(x_cast.shape())) {
      zero_tensor = backend::full_with_tensor<T>(
          shape64<T>(x_cast), 0, x_cast.dtype(), x_cast.place());
    } else {
      zero_tensor = full<T>(
          common::vectorize(x_cast.dims()), 0, x_cast.dtype(), x_cast.place());
    }
    x_grad_tmp =
        put_along_axis<T>(zero_tensor, indices_cast, out_grad_cast, axis);

    set_output<T>(ConvertToOrig<T>(x_grad_tmp, x.dtype()), x_grad);
  }
}

template <typename T>
void kron_grad(const Tensor& x,
               const Tensor& y,
               const Tensor& out_grad,
               Tensor* x_grad,
               Tensor* y_grad) {
  if (x_grad) {
    Tensor zero = full<T>({1}, 0, DataType::INT32, x.place());
    Tensor x_grad_tmp;
    if (has_dynamic_shape(x.shape()) || has_dynamic_shape(y.shape())) {
      Tensor x_ = x;
      Tensor y_ = y;
      auto diff = std::abs(static_cast<int>(x.dims().size()) -
                           static_cast<int>(y.dims().size()));
      while (diff--) {
        if (x_.dims().size() > y_.dims().size()) {
          y_ = backend::unsqueeze<T>(y_, zero);
        } else {
          x_ = backend::unsqueeze<T>(x_, zero);
        }
      }

      // tile
      std::vector<Tensor> x_shape_vec;
      for (int64_t i = 0; i < x_.dims().size(); ++i) {
        auto x_shape_slice = get_slice<T>(shape64<T>(x_), i);
        x_shape_vec.push_back(x_shape_slice);
      }

      auto y_tile = backend::tile<T>(y_, shape64<T>(x_));

      auto out_grad_tmp = y_tile * out_grad;

      std::vector<Tensor> out_grad_shape_vec;
      for (int64_t i = 0; i < out_grad.dims().size(); ++i) {
        auto out_grad_shape_slice = get_slice<T>(shape64<T>(out_grad), i);
        out_grad_shape_vec.push_back(out_grad_shape_slice);
      }
      if (x_shape_vec.size() != 0) {
        while (true) {
          std::vector<Tensor> expand_shape_vec;
          for (int64_t i = 0; i < out_grad_tmp.dims().size(); ++i) {
            auto expand_shape = get_slice<T>(shape64<T>(out_grad_tmp), i);
            expand_shape_vec.push_back(expand_shape);
          }
          int num_reduce = 0;
          while (x_shape_vec.size() != 0 && expand_shape_vec.size() <= 8) {
            Tensor repeat = x_shape_vec.back();
            auto orig_size =
                cast<T>(out_grad_shape_vec.back() / repeat, DataType::INT64);
            size_t out_grad_last_index = out_grad_shape_vec.size() - 1;
            expand_shape_vec[out_grad_last_index] = repeat;
            expand_shape_vec.insert(
                expand_shape_vec.begin() + out_grad_shape_vec.size(),
                orig_size);

            x_shape_vec.pop_back();
            out_grad_shape_vec.pop_back();
            ++num_reduce;
          }

          int axis = static_cast<int>(out_grad_shape_vec.size()) + 1;
          std::vector<Tensor> reduce_axes_vec;
          for (int i = 0; i < num_reduce; ++i) {
            reduce_axes_vec.push_back(
                full<T>({1}, axis, DataType::INT32, x.place()));
            axis += 2;
          }

          out_grad_tmp =
              backend::reshape<T>(out_grad_tmp, concat<T>(expand_shape_vec));
          out_grad_tmp =
              backend::sum<T>(out_grad_tmp, concat<T>(reduce_axes_vec));
          if (x_shape_vec.size() == 0) {
            break;
          }
        }
      }
      x_grad_tmp = backend::reshape<T>(out_grad_tmp, shape64<T>(x));
    } else {
      auto x_shape = x.shape();
      auto y_shape = y.shape();

      auto diff = std::abs(static_cast<int>(x_shape.size()) -
                           static_cast<int>(y_shape.size()));
      for (int i = 0; i < diff; i++) {
        if (x_shape.size() > y_shape.size()) {
          y_shape.insert(y_shape.begin(), 1);
        } else {
          x_shape.insert(x_shape.begin(), 1);
        }
      }

      auto x_ = reshape<T>(x, x_shape);
      auto y_ = reshape<T>(y, y_shape);

      // tile
      std::vector<int64_t> x_dim = common::vectorize<int64_t>(x_.dims());
      auto y_tile = tile<T>(y_, x_dim);

      auto out_grad_tmp = y_tile * out_grad;

      std::vector<int64_t> out_grad_shape(out_grad_tmp.shape());

      if (x_dim.size() != 0) {
        while (true) {
          std::vector<int64_t> expand_shape(out_grad_tmp.shape());

          int num_reduce = 0;
          while (x_dim.size() != 0 && expand_shape.size() <= 8) {
            int64_t repeat = x_dim.back();
            int64_t orig_size = out_grad_shape.back() / repeat;
            size_t out_grad_last_index = out_grad_shape.size() - 1;

            expand_shape[out_grad_last_index] = repeat;
            expand_shape.insert(
                expand_shape.begin() + out_grad_shape.size(), 1, orig_size);

            x_dim.pop_back();
            out_grad_shape.pop_back();
            ++num_reduce;
          }

          int64_t axis = static_cast<int64_t>(out_grad_shape.size()) + 1;
          std::vector<int64_t> reduce_axes;
          for (int i = 0; i < num_reduce; ++i) {
            reduce_axes.push_back(axis);
            axis += 2;
          }

          out_grad_tmp = reshape<T>(out_grad_tmp, expand_shape);
          out_grad_tmp = sum<T>(out_grad_tmp, reduce_axes);

          if (x_dim.size() == 0) {
            break;
          }
        }
      }
      x_grad_tmp = reshape<T>(out_grad_tmp, x.shape());
    }
    set_output<T>(x_grad_tmp, x_grad);
  }
  if (y_grad) {
    Tensor zero = full<T>({1}, 0, DataType::INT32, y.place());
    auto x_cast = ConvertToMT<T>(x);
    auto out_grad_cast = ConvertToMT<T>(out_grad);
    Tensor out_grad_tmp;
    Tensor y_grad_tmp;

    if (has_dynamic_shape(x_cast.shape()) || has_dynamic_shape(y.shape())) {
      Tensor x_ = x_cast;
      Tensor y_ = y;
      auto diff = std::abs(static_cast<int>(x_cast.dims().size()) -
                           static_cast<int>(y.dims().size()));
      while (diff--) {
        if (x_.dims().size() > y_.dims().size()) {
          y_ = backend::unsqueeze<T>(y_, zero);
        } else {
          x_ = backend::unsqueeze<T>(x_, zero);
        }
      }

      std::vector<Tensor> x_shape_vec;
      for (int64_t i = 0; i < x_.dims().size(); ++i) {
        auto x_shape_slice = get_slice<T>(shape64<T>(x_), i);
        x_shape_vec.push_back(x_shape_slice);
      }

      for (int64_t i = 0; i < x_.dims().size(); ++i) {
        auto y_shape_slice = get_slice<T>(shape64<T>(y_), i);
        auto x_shape_slice = get_slice<T>(shape64<T>(x_), i);
        auto y_shape_tile = backend::tile<T>(y_shape_slice, x_shape_slice);
        x_ = backend::repeat_interleave_with_tensor_index<T>(
            x_, y_shape_tile, i);
      }
      out_grad_tmp = out_grad_cast * x_;

      std::vector<Tensor> out_grad_shape_vec;
      for (int64_t i = 0; i < out_grad.dims().size(); ++i) {
        auto out_grad_shape_slice = get_slice<T>(shape64<T>(out_grad_cast), i);
        out_grad_shape_vec.push_back(out_grad_shape_slice);
      }

      if (x_shape_vec.size() != 0) {
        while (true) {
          std::vector<Tensor> expand_shape_vec;
          for (int64_t i = 0; i < out_grad_tmp.dims().size(); ++i) {
            auto expand_shape = get_slice<T>(shape64<T>(out_grad_tmp), i);
            expand_shape_vec.push_back(expand_shape);
          }
          int num_reduce = 0;
          while (x_shape_vec.size() != 0 && expand_shape_vec.size() <= 8) {
            auto repeat = x_shape_vec.back();
            auto orig_size =
                cast<T>(out_grad_shape_vec.back() / repeat, DataType::INT64);
            size_t out_grad_last_index = out_grad_shape_vec.size() - 1;
            expand_shape_vec[out_grad_last_index] = repeat;
            expand_shape_vec.insert(
                expand_shape_vec.begin() + out_grad_shape_vec.size(),
                orig_size);

            x_shape_vec.pop_back();
            out_grad_shape_vec.pop_back();
            ++num_reduce;
          }
          int axis = static_cast<int>(out_grad_shape_vec.size());
          std::vector<Tensor> reduce_axes_vec;
          for (int i = 0; i < num_reduce; ++i) {
            reduce_axes_vec.push_back(
                full<T>({1}, axis, DataType::INT32, y.place()));
            axis += 2;
          }
          out_grad_tmp =
              backend::reshape<T>(out_grad_tmp, concat<T>(expand_shape_vec));
          out_grad_tmp =
              backend::sum<T>(out_grad_tmp, concat<T>(reduce_axes_vec));

          if (x_shape_vec.size() == 0) {
            break;
          }
        }
      }
      y_grad_tmp = backend::reshape<T>(
          ConvertToOrig<T>(out_grad_tmp, out_grad.dtype()), shape64<T>(y));
    } else {
      auto x_shape = x_cast.shape();
      auto y_shape = y.shape();

      auto diff = std::abs(static_cast<int>(x_shape.size()) -
                           static_cast<int>(y_shape.size()));
      for (int i = 0; i < diff; i++) {
        if (x_shape.size() > y_shape.size()) {
          y_shape.insert(y_shape.begin(), 1);
        } else {
          x_shape.insert(x_shape.begin(), 1);
        }
      }

      auto x_ = reshape<T>(x_cast, x_shape);
      auto y_ = reshape<T>(y, y_shape);

      std::vector<int64_t> x_dim = common::vectorize<int64_t>(x_.dims());
      for (size_t i = 0; i < x_dim.size(); ++i) {
        x_ = repeat_interleave<T>(x_, y_shape[i], i);
      }
      out_grad_tmp = out_grad_cast * x_;

      tile_grad<T>(y_, out_grad_tmp, IntArray(x_dim), &y_grad_tmp);
      y_grad_tmp =
          reshape<T>(ConvertToOrig<T>(y_grad_tmp, y.dtype()), y.shape());
    }
    set_output<T>(y_grad_tmp, y_grad);
  }
}

template <typename T>
void take_along_axis_grad(const Tensor& arr,
                          const Tensor& indices,
                          const Tensor& out_grad,
                          int axis,
                          Tensor* arr_grad) {
  if (arr_grad) {
    auto arr_cast = ConvertToMT<T>(arr);
    auto out_grad_cast = ConvertToMT<T>(out_grad);
    // put_along_axis doesn't support zero dim
    if (arr_cast.dims().size() == 0) {
      by_pass<T>(ConvertToOrig<T>(out_grad_cast, out_grad.dtype()), arr_grad);
      return;
    }

    // function `put_along_axis` requires a non-negative axis
    if (axis < 0) {
      axis += arr_cast.dims().size();
    }

    Tensor zero_tensor;
    if (has_dynamic_shape(arr_cast.shape())) {
      zero_tensor = backend::full_with_tensor<T>(
          shape64<T>(arr_cast), 0, arr_cast.dtype(), arr_cast.place());
    } else {
      zero_tensor = full<T>(common::vectorize(arr_cast.dims()),
                            0,
                            arr_cast.dtype(),
                            arr_cast.place());
    }
    auto arr_grad_tmp = put_along_axis<T>(zero_tensor,
                                          indices,
                                          out_grad_cast,
                                          axis,
                                          /*reduce*/ "add",
                                          /*include_self*/ true);
    set_output<T>(ConvertToOrig<T>(arr_grad_tmp, arr.dtype()), arr_grad);
  }
}

template <typename T>
void ceil_grad(const Tensor& out_grad, Tensor* x_grad) {
  if (x_grad) {
    Tensor zero_tensor;
    if (has_dynamic_shape(out_grad.shape())) {
      zero_tensor = backend::full_with_tensor<T>(
          shape64<T>(out_grad), 0.0, out_grad.dtype());
    } else {
      zero_tensor =
          full<T>(common::vectorize(out_grad.dims()), 0.0, out_grad.dtype());
    }
    set_output<T>(zero_tensor, x_grad);
  }
}

template <typename T>
void amax_grad(const Tensor& x,
               const Tensor& out,
               const Tensor& out_grad,
               const IntArray& axis,
               bool keepdim,
               bool reduce_all,
               Tensor* x_grad) {
  if (x_grad) {
    Tensor x_grad_tmp;
    if (has_dynamic_shape(x.shape())) {
      const Tensor x_shape = shape64<T>(x);
      const Tensor zero_tensor =
          backend::full_with_tensor<T>(x_shape, 0.0, x.dtype());
      const int64_t axis_size = axis.size();
      const int64_t x_dim_size = x.dims().size();

      reduce_all = false;
      if (reduce_all || axis_size == 0 || axis_size == x_dim_size) {
        reduce_all = true;
      }

      if (x_dim_size == 0 || x_dim_size == 1 || keepdim) {
        auto out_grad_tmp = backend::expand<T>(out_grad, x_shape);
        auto out_tmp = backend::expand<T>(out, x_shape);
        auto mask = equal<T>(x, out_tmp);
        auto mask_sum = backend::sum<T>(mask, axis, x.dtype(), keepdim = true);
        auto grad_tmp = out_grad_tmp / mask_sum;
        x_grad_tmp = where<T>(mask, grad_tmp, zero_tensor);
      } else {
        const Tensor out_grad_shape = shape64<T>(out_grad);
        auto axis_ = std::vector<int64_t>();

        if (reduce_all) {
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
        const Tensor out_grad_shape_extend =
            get_unsqueeze_dims<T>(out_grad_shape, axis_);
        auto out_grad_ = backend::reshape<T>(out_grad, out_grad_shape_extend);
        auto out_ = backend::reshape<T>(out, out_grad_shape_extend);
        auto out_grad_tmp = backend::expand<T>(out_grad_, x_shape);
        auto out_tmp = backend::expand<T>(out_, x_shape);
        auto mask = equal<T>(x, out_tmp);
        auto mask_sum = backend::sum<T>(mask, axis_, x.dtype(), keepdim = true);
        auto grad_tmp = out_grad_tmp / mask_sum;
        x_grad_tmp = where<T>(mask, grad_tmp, zero_tensor);
      }
    } else {
      auto zero_tensor = full<T>(common::vectorize(x.dims()), 0.0, x.dtype());
      std::vector<int64_t> x_dim = common::vectorize<int64_t>(x.dims());
      int64_t axis_size = axis.size();
      int64_t x_dim_size = x_dim.size();
      reduce_all = false;
      if (reduce_all || axis_size == 0 || axis_size == x_dim_size) {
        reduce_all = true;
      }

      if (x_dim_size == 0 || x_dim_size == 1 || keepdim) {
        auto out_grad_tmp = out_grad.expand(IntArray(x_dim));
        auto out_tmp = out.expand(IntArray(x_dim));
        auto mask = equal<T>(x, out_tmp);
        auto mask_sum = sum<T>(mask, axis, x.dtype(), keepdim = true);
        auto grad_tmp = out_grad_tmp / mask_sum;
        x_grad_tmp = where<T>(mask, grad_tmp, zero_tensor);
      } else {
        auto axis_ = std::vector<int64_t>();
        if (reduce_all) {
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
        auto out_grad_shape = get_unsqueeze_dims(out_grad, axis_);
        auto out_grad_ = reshape<T>(out_grad, out_grad_shape);
        auto out_ = reshape<T>(out, out_grad_shape);
        auto out_grad_tmp = out_grad_.expand(IntArray(x_dim));
        auto out_tmp = out_.expand(IntArray(x_dim));
        auto mask = equal<T>(x, out_tmp);
        auto mask_sum = sum<T>(mask, axis_, x.dtype(), keepdim = true);
        auto grad_tmp = out_grad_tmp / mask_sum;
        x_grad_tmp = where<T>(mask, grad_tmp, zero_tensor);
      }
    }
    set_output<T>(x_grad_tmp, x_grad);
  }
}

template <typename T>
void amin_grad(const Tensor& x,
               const Tensor& out,
               const Tensor& out_grad,
               const IntArray& axis,
               bool keepdim,
               bool reduce_all,
               Tensor* x_grad) {
  if (x_grad) {
    Tensor x_grad_tmp;
    amax_grad<T>(x, out, out_grad, axis, keepdim, reduce_all, &x_grad_tmp);

    set_output<T>(x_grad_tmp, x_grad);
  }
}

template <typename T>
void p_norm_grad(const Tensor& x,
                 /*output of forward was reserved for efficient backward*/
                 const Tensor& out,
                 const Tensor& out_grad,
                 float porder,
                 int axis,
                 float epsilon,
                 bool keepdim,
                 bool asvector,
                 Tensor* x_grad) {
  if (x_grad) {
    if (axis < 0) {
      axis += x.dims().size();
    }

    Tensor x_grad_tmp;
    if (porder == 0.0) {
      // dx = 0
      if (has_dynamic_shape(x.shape())) {
        x_grad_tmp = backend::full_with_tensor<T>(
            shape64<T>(x), 0, x.dtype(), x.place());
      } else {
        x_grad_tmp =
            full<T>(common::vectorize(x.dims()), 0, x.dtype(), x.place());
      }
    } else {
      /* generic case formula:
        dx = {
          dy * y^(1-p) * |x|^(p-1) * sgn(x), if p != +-inf,
          dy * sgn(x) * (x==y), if p == +-inf.
        }
      */
      Tensor expand_out = out;
      Tensor expand_out_grad = out_grad;
      // firstly expand output_grad to same ndim with x for convenience
      if (!keepdim) {
        if (has_dynamic_shape(x.shape())) {
          Tensor expand_shape;
          if (asvector) {
            // reduce all dimensions in forward
            expand_shape = full<T>(std::vector<int64_t>(x.dims().size(), 1),
                                   1,
                                   DataType::INT64,
                                   out_grad.place());
          } else {
            // only reduce one dimension in forward
            expand_shape = shape64<T>(out_grad);
            std::vector<Tensor> expand_shape_vec;
            for (int64_t i = 0; i < expand_shape.size(); ++i) {
              expand_shape_vec.push_back(get_slice<T>(expand_shape, i));
            }
            expand_shape_vec.insert(
                expand_shape_vec.begin() + axis,
                full<T>({1}, 1, expand_shape.dtype(), expand_shape.place()));
            expand_shape = concat<T>(expand_shape_vec);
          }
          expand_out_grad = backend::reshape<T>(out_grad, expand_shape);
          expand_out = backend::reshape<T>(out, expand_shape);
        } else {
          std::vector<int64_t> expand_shape =
              common::vectorize(out_grad.dims());
          if (asvector) {
            // reduce all dimensions in forward
            expand_shape = std::vector<int64_t>(x.dims().size(), 1);
          } else {
            // only reduce one dimension in forward
            expand_shape.insert(expand_shape.begin() + axis, 1);
          }
          expand_out_grad = reshape<T>(out_grad, expand_shape);
          expand_out = reshape<T>(out, expand_shape);
        }
      }

      if (porder == 1.0) {
        // dx = dy * sign(x)
        auto x_sign = sign<T>(x);
        x_grad_tmp = x_sign * expand_out_grad;
      } else if (porder == 2.0) {
        // dx = dy * (x / y)
        x_grad_tmp = x / expand_out;
        // fill zero to avoid division by zero
        Tensor _zero_tensor;
        if (has_dynamic_shape(x.shape())) {
          _zero_tensor = backend::full_with_tensor<T>(
              shape64<T>(x), 0, x.dtype(), x.place());
        } else {
          _zero_tensor =
              full<T>(common::vectorize(x.dims()), 0, x.dtype(), x.place());
        }

        auto finite_mask = isfinite<T>(x_grad_tmp);
        x_grad_tmp = where<T>(finite_mask, x_grad_tmp, _zero_tensor);
        x_grad_tmp = expand_out_grad * (x_grad_tmp);

      } else if (porder == INFINITY || porder == -INFINITY) {
        // dy * sgn(x) * (x==y), if p == +-inf.
        auto x_abs = abs<T>(x);
        auto mask =
            cast<T>(bitwise_or<T>(equal<T>(x_abs, expand_out), isnan<T>(x_abs)),
                    expand_out.dtype());
        auto x_sign = sign<T>(x);
        x_grad_tmp =
            x_sign * ((expand_out_grad /
                       sum<T>(mask, {axis}, expand_out_grad.dtype(), true)) *
                      mask);

      } else if (porder < 1.0) {
        // dx = dy * y^(1-p) * |x|^(p-1) * sgn(x)
        auto x_sign = sign<T>(x);
        auto x_abs_pow = abs<T>(x);
        x_abs_pow = x_abs_pow.pow(porder - 1);

        auto x_scaled = x_sign * x_abs_pow;
        x_grad_tmp = x_scaled * expand_out_grad * expand_out.pow(1 - porder);

      } else if (porder < 2.0) {
        // dx = dy * y^(1-p) * |x|^(p-1) * sgn(x)
        auto x_sign = sign<T>(x);
        auto x_abs_pow = abs<T>(x);
        x_abs_pow = x_abs_pow.pow(porder - 1);

        // auto scale_v = expand_out_grad / expand_out.pow(porder - 1);
        // auto _zero_tensor =
        //     full<T>(common::vectorize(x.dims()), 0.0, x.dtype());
        // auto out_non_zero_mask = not_equal<T>(expand_out, _zero_tensor);
        // scale_v = scale_v * cast<T>(out_non_zero_mask, scale_v.dtype());
        // x_grad_tmp = x_sign * x_abs_pow * scale_v;

        auto scale_v = expand_out_grad * expand_out.pow(1 - porder);
        x_grad_tmp = x_sign * x_abs_pow * scale_v;

      } else {
        // dx = dy * y^(1-p) * |x|^(p-1) * sgn(x)
        auto x_sign = sign<T>(x);
        auto x_abs_pow = abs<T>(x);
        x_abs_pow = x_abs_pow.pow(porder - 1);

        auto x_scaled = x_sign * x_abs_pow;
        x_grad_tmp = x_scaled * expand_out_grad * expand_out.pow(1 - porder);
      }
    }
    set_output<T>(x_grad_tmp, x_grad);
  }
}

template <typename T>
void angle_grad(const Tensor& x, const Tensor& out_grad, Tensor* x_grad) {
  if (x_grad) {
    Tensor cast_x = ConvertToMT<T>(x);
    Tensor zero_tensor;
    if (has_dynamic_shape(cast_x.shape())) {
      const Tensor x_shape = shape64<T>(cast_x);
      zero_tensor = backend::full_with_tensor<T>(
          x_shape, 0.0, cast_x.dtype(), cast_x.place());
    } else {
      zero_tensor = full<T>(cast_x.shape(), 0, cast_x.dtype(), cast_x.place());
    }

    set_output<T>(ConvertToOrig<T>(zero_tensor, x.dtype()), x_grad);
  }
}

}  // namespace details
}  // namespace primitive
}  // namespace paddle
