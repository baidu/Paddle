// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/common/hostdevice.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/funcs/for_range.h"
#if defined(__NVCC__) || defined(__HIPCC__)
#include "thrust/device_vector.h"
#endif

namespace phi {
namespace funcs {

// Giving a linear destination index and strides of tensor, get_idx return the
// corresponding linear position of source tensor.
// The linear index is the position of flatten tensor.
// Giving a linear destination index and strides of tensor, get_idx return the
// corresponding linear position of source tensor.
// The linear index is the position of flatten tensor.
HOSTDEVICE inline int64_t get_src_idx(const int64_t dst_idx,
                                      const int64_t* dst_strides,
                                      const int64_t* dst_shape,
                                      const int64_t* src_strides,
                                      const bool* is_fft_axis,
                                      const bool conj,
                                      const int64_t rank) {
  int64_t src_idx = 0;
  int64_t quotient = dst_idx;
  int64_t remainder = 0;

  for (int64_t i = 0; i < rank; i++) {
    remainder = quotient % dst_strides[i];
    quotient = quotient / dst_strides[i];
    if (conj && is_fft_axis[i]) {
      src_idx += ((dst_shape[i] - quotient) % dst_shape[i]) * src_strides[i];
    } else {
      src_idx += src_strides[i] * quotient;
    }
    quotient = remainder;
  }

  return src_idx;
}

HOSTDEVICE inline bool is_conj_part(const int64_t dst_idx,
                                    const int64_t* dst_strides,
                                    const int64_t last_axis,
                                    const int64_t last_axis_size) {
  int64_t quotient = dst_idx;
  int64_t remainder = 0;

  for (int64_t i = 0; i < last_axis + 1; i++) {
    remainder = quotient % dst_strides[i];
    quotient = quotient / dst_strides[i];

    if ((i == last_axis) && (quotient > last_axis_size - 1)) {
      return true;
    }

    quotient = remainder;
  }

  return false;
}

// FFTFillConjFunctor fill the destination tensor with source tensor and
// conjugate symmetry element of source tensor .
// Use framework::ForRange to iterate destination element with
// supporting different device
template <typename C>
struct FFTFillConjFunctor {
  FFTFillConjFunctor(const C* src_data,
                     C* dst_data,
                     const int64_t* src_strides,
                     const int64_t* dst_strides,
                     const int64_t* dst_shape,
                     const bool* is_fft_axis,
                     const int64_t last_axis,
                     const int64_t last_axis_size,
                     const int64_t rank)
      : src_data_(src_data),
        dst_data_(dst_data),
        src_strides_(src_strides),
        dst_strides_(dst_strides),
        dst_shape_(dst_shape),
        is_fft_axis_(is_fft_axis),
        last_axis_(last_axis),
        last_axis_size_(last_axis_size),
        rank_(rank) {}
  HOSTDEVICE void operator()(int64_t dst_idx) {
    if (is_conj_part(dst_idx, dst_strides_, last_axis_, last_axis_size_)) {
      const auto conj_idx = get_src_idx(dst_idx,
                                        dst_strides_,
                                        dst_shape_,
                                        src_strides_,
                                        is_fft_axis_,
                                        true,
                                        rank_);
      auto src_value = src_data_[conj_idx];
      auto conj_value = C(src_value.real, -src_value.imag);
      dst_data_[dst_idx] = conj_value;
    } else {
      const auto copy_idx = get_src_idx(dst_idx,
                                        dst_strides_,
                                        dst_shape_,
                                        src_strides_,
                                        is_fft_axis_,
                                        false,
                                        rank_);
      dst_data_[dst_idx] = src_data_[copy_idx];
    }
  }

  const C* src_data_;
  C* dst_data_;
  const int64_t* src_strides_;
  const int64_t* dst_strides_;
  const int64_t* dst_shape_;
  const bool* is_fft_axis_;
  const int64_t last_axis_;
  const int64_t last_axis_size_;
  const int64_t rank_;
};

template <typename DeviceContext, typename C>
void FFTFillConj(const DeviceContext& ctx,
                 const DenseTensor* src,
                 DenseTensor* dst,
                 const std::vector<int64_t>& axes) {
  std::vector<int64_t> src_strides_v =
      common::vectorize<int64_t>(common::stride(src->dims()));
  std::vector<int64_t> dst_strides_v =
      common::vectorize<int64_t>(common::stride(dst->dims()));
  std::vector<int64_t> dst_shape_v = common::vectorize<int64_t>(dst->dims());
  const auto src_data = src->data<C>();
  auto dst_data = dst->data<C>();
  const auto last_axis = axes.back();
  const auto last_axis_size = dst->dims().at(last_axis) / 2 + 1;
  const int64_t rank = dst->dims().size();
  auto _is_fft_axis = std::make_unique<bool[]>(rank);
  for (const auto i : axes) {
    _is_fft_axis[i] = true;
  }

#if defined(__NVCC__) || defined(__HIPCC__)
  DenseTensor src_strides_g;
  src_strides_g.Resize({(int64_t)src_strides_v.size()});
  int64_t* src_strides = ctx.template Alloc<int64_t>(&src_strides_g);
  DenseTensor dst_strides_g;
  dst_strides_g.Resize({(int64_t)dst_strides_v.size()});
  int64_t* dst_strides = ctx.template Alloc<int64_t>(&dst_strides_g);
  DenseTensor dst_shape_g;
  dst_shape_g.Resize({(int64_t)dst_shape_v.size()});
  int64_t* dst_shape = ctx.template Alloc<int64_t>(&dst_shape_g);
  DenseTensor is_fft_axis_g;
  is_fft_axis_g.Resize({rank});
  bool* p_is_fft_axis = ctx.template Alloc<bool>(&is_fft_axis_g);
  auto cplace = phi::CPUPlace();
  const auto gplace = ctx.GetPlace();
  memory_utils::Copy(gplace,
                     src_strides,
                     cplace,
                     src_strides_v.data(),
                     sizeof(int64_t) * src_strides_v.size(),
                     ctx.stream());
  memory_utils::Copy(gplace,
                     dst_strides,
                     cplace,
                     dst_strides_v.data(),
                     sizeof(int64_t) * dst_strides_v.size(),
                     ctx.stream());
  memory_utils::Copy(gplace,
                     dst_shape,
                     cplace,
                     dst_shape_v.data(),
                     sizeof(int64_t) * dst_shape_v.size(),
                     ctx.stream());
  memory_utils::Copy(gplace,
                     p_is_fft_axis,
                     cplace,
                     _is_fft_axis.get(),
                     sizeof(bool) * rank,
                     ctx.stream());

#else
  const auto src_strides = src_strides_v.data();
  const auto dst_strides = dst_strides_v.data();
  const auto dst_shape = dst_shape_v.data();
  const auto p_is_fft_axis = _is_fft_axis.get();
#endif
  ForRange<DeviceContext> for_range(ctx, dst->numel());
  FFTFillConjFunctor<C> fill_conj_functor(src_data,
                                          dst_data,
                                          src_strides,
                                          dst_strides,
                                          dst_shape,
                                          p_is_fft_axis,
                                          last_axis,
                                          last_axis_size,
                                          rank);
  for_range(fill_conj_functor);
}

template <typename T>
struct FFTFillConjGradFunctor {
  T* input_;
  const size_t axis_;
  const int64_t stride_to_last_axis;
  const int64_t stride_second_to_last_axis;
  const size_t double_length_;

  FFTFillConjGradFunctor(T* input,
                         size_t axis,
                         int64_t stride_second_to_last_axis,
                         int64_t stride_to_last_axis,
                         size_t double_length)
      : input_(input),
        axis_(axis),
        stride_to_last_axis(stride_to_last_axis),
        stride_second_to_last_axis(stride_second_to_last_axis),
        double_length_(double_length) {}

  HOSTDEVICE void operator()(size_t index) {
    size_t index_i = (index % stride_second_to_last_axis) / stride_to_last_axis;
    if ((0 < index_i) && (index_i < double_length_ + 1)) {
      input_[index] *= static_cast<T>(2);
    }
  }
};

}  // namespace funcs
}  // namespace phi
