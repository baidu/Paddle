/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/impl/bessel_kernel_cuda_impl.h"

namespace phi {

template <typename T>
struct CudaI0GradFunctor {
  __device__ __forceinline__ T operator()(const T _x, const T _out_grad) const {
    using MT = typename phi::dtype::MPTypeTrait<T>::Type;
    const MT mp_x = static_cast<MT>(_x);
    const MT mp_out_grad = static_cast<MT>(_out_grad);
    // get output of i1
    MT x = std::abs(mp_x);
    if (x <= MT{8.0}) {
      auto coeff_pair_A = ChebyshevCoefficientsI1e_A<MT>();
      auto A = std::get<0>(coeff_pair_A);
      auto len = std::get<1>(coeff_pair_A);
      MT y = (x / MT{2.0}) - MT{2.0};

      const MT i1_out = std::exp(x) * x * Chbevl<MT>(y, A, len);
      const MT i1_data = (mp_x < MT{0.0}) ? -i1_out : i1_out;
      // calculate i0 gradient
      return static_cast<T>(i1_data * mp_out_grad);
    }
    auto coeff_pair_B = ChebyshevCoefficientsI1e_B<MT>();
    auto B = std::get<0>(coeff_pair_B);
    auto len = std::get<1>(coeff_pair_B);
    MT y = (MT{32.0} / x) - MT{2.0};

    const MT i1_out = (std::exp(x) * Chbevl<MT>(y, B, len)) / std::sqrt(x);
    const MT i1_data = (mp_x < MT{0.0}) ? -i1_out : i1_out;

    return static_cast<T>(i1_data * mp_out_grad);
  }
};

template <typename T>
struct CudaI0eGradFunctor {
  __device__ __forceinline__ T operator()(const T _x,
                                          const T _out,
                                          const T _out_grad) const {
    using MT = typename phi::dtype::MPTypeTrait<T>::Type;
    const MT mp_x = static_cast<MT>(_x);
    const MT mp_out = static_cast<MT>(_out);
    const MT mp_out_grad = static_cast<MT>(_out_grad);
    // get output of i1e
    MT x = std::abs(mp_x);
    if (x <= MT{8.0}) {
      auto coeff_pair_A = ChebyshevCoefficientsI1e_A<MT>();
      auto A = std::get<0>(coeff_pair_A);
      auto len = std::get<1>(coeff_pair_A);
      MT y = (x / MT{2.0}) - MT{2.0};

      const MT i1e_out = Chbevl<MT>(y, A, len) * x;
      const MT i1e_data = (mp_x < MT{0.0}) ? -i1e_out : i1e_out;
      // calculate i0e gradient
      return static_cast<T>((i1e_data - std::copysign(MT{1.0}, mp_x) * mp_out) *
                            mp_out_grad);
    }
    auto coeff_pair_B = ChebyshevCoefficientsI1e_B<MT>();
    auto B = std::get<0>(coeff_pair_B);
    auto len = std::get<1>(coeff_pair_B);
    MT y = (MT{32.0} / x) - MT{2.0};

    const MT i1e_out = Chbevl<MT>(y, B, len) / std::sqrt(x);
    const MT i1e_data = (mp_x < MT{0.0}) ? -i1e_out : i1e_out;

    return static_cast<T>((i1e_data - std::copysign(MT{1.0}, mp_x) * mp_out) *
                          mp_out_grad);
  }
};

template <typename T>
struct CudaI1GradFunctor {
  __device__ __forceinline__ T operator()(const T _x,
                                          const T _out,
                                          const T _out_grad) const {
    using MT = typename phi::dtype::MPTypeTrait<T>::Type;
    const MT mp_x = static_cast<MT>(_x);
    const MT mp_out = static_cast<MT>(_out);
    const MT mp_out_grad = static_cast<MT>(_out_grad);
    MT x = std::abs(mp_x);
    if (x <= MT{8.0}) {
      auto coeff_pair_A = ChebyshevCoefficientsI0e_A<MT>();
      auto A = std::get<0>(coeff_pair_A);
      auto len = std::get<1>(coeff_pair_A);
      MT y = (x / MT{2.0}) - MT{2.0};
      MT eps = static_cast<MT>(std::numeric_limits<T>::epsilon());

      if (x <= eps) {
        MT out = (MT{0.5}) * mp_out_grad;
        return static_cast<T>(out);
      } else {
        return static_cast<T>(
            (std::exp(x) * Chbevl<MT>(y, A, len) - mp_out / mp_x) *
            mp_out_grad);
      }
    }
    auto coeff_pair_B = ChebyshevCoefficientsI0e_B<MT>();
    auto B = std::get<0>(coeff_pair_B);
    auto len = std::get<1>(coeff_pair_B);
    MT y = (MT{32.0} / x) - MT{2.0};

    return static_cast<T>(
        (std::exp(x) * Chbevl<MT>(y, B, len) / std::sqrt(x) - mp_out / mp_x) *
        mp_out_grad);
  }
};

template <typename T>
struct CudaI1eGradFunctor {
  __device__ __forceinline__ T operator()(const T _x,
                                          const T _out,
                                          const T _out_grad) const {
    using MT = typename phi::dtype::MPTypeTrait<T>::Type;
    const MT mp_x = static_cast<MT>(_x);
    const MT mp_out = static_cast<MT>(_out);
    const MT mp_out_grad = static_cast<MT>(_out_grad);
    MT x = std::abs(mp_x);
    if (x <= MT{8.0}) {
      auto coeff_pair_A = ChebyshevCoefficientsI0e_A<MT>();
      auto A = std::get<0>(coeff_pair_A);
      auto len = std::get<1>(coeff_pair_A);
      MT y = (x / MT{2.0}) - MT{2.0};
      MT eps = static_cast<MT>(std::numeric_limits<T>::epsilon());

      if (x <= eps) {
        MT out = (MT{0.5}) * mp_out_grad;
        return static_cast<T>(out);
      } else {
        MT out = (Chbevl<MT>(y, A, len) -
                  mp_out * (std::copysign(MT{1.0}, mp_x) + (MT{1.0}) / mp_x)) *
                 mp_out_grad;
        return static_cast<T>(out);
      }
    }
    auto coeff_pair_B = ChebyshevCoefficientsI0e_B<MT>();
    auto B = std::get<0>(coeff_pair_B);
    auto len = std::get<1>(coeff_pair_B);
    MT y = (MT{32.0} / x) - MT{2.0};

    return static_cast<T>(
        (Chbevl<T>(y, B, len) / std::sqrt(x) -
         mp_out * (std::copysign(MT{1.0}, mp_x) + (MT{1.0}) / mp_x)) *
        mp_out_grad);
  }
};

}  // namespace phi
