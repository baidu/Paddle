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

#pragma once
#include <cmath>
#include <string>
#include "glog/logging.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/common/complex.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/dense_tensor.h"

#include "paddle/phi/common/memory_utils.h"

namespace phi {

template <typename DeviceContext, typename T>
struct AccuracyCheckFunctor {
  void operator()(const DeviceContext& ctx,
                  const DenseTensor& in,
                  const DenseTensor& other,
                  const std::string& fn_name,
                  const float rtol,
                  const float atol,
                  bool equal_nan,
                  DenseTensor* output);
};

template <typename T>
struct AccuracyCheckFunctor<phi::CPUContext, T> {
  void operator()(const phi::CPUContext& ctx,
                  const DenseTensor& in,
                  const DenseTensor& other,
                  const std::string& fn_name,
                  const double rtol,
                  const double atol,
                  bool equal_nan,
                  DenseTensor* output) {
    auto* in_a = in.data<T>();
    auto* in_b = other.data<T>();
    auto* out_data = ctx.template Alloc<bool>(output);
    auto num = in.numel();
    // *out_data = true;
    for (int i = 0; i < num; i++) {
      out_data[i] = true;
    }
    bool val;
    int res_index = -1;
    for (int i = 0; i < num; i++) {
      const double a = in_a[i], b = in_b[i];

      if (std::isnan(a) || std::isnan(b)) {
        val = equal_nan && std::isnan(a) == std::isnan(b);
      } else {
        double left = (a > b ? a - b : b - a);
        double right = atol + (b > 0 ? rtol * b : (-rtol) * b);
        double diff = (left > right ? left - right : right - left);
        val = a == b || left <= right || diff <= 1e-10;
      }
      // *out_data &= val;
      out_data[i] = val;
      if (!val) {
        VLOG(2) << "Accuracy check failed between" << a << " and " << b
                << " at index= " << i;
        res_index = i;
        break;
      }
    }
    PADDLE_ENFORCE_EQ(val,
                      true,
                      common::errors::PreconditionNotMet(
                          "Accuracy check failed, kernel name %s, res index %d",
                          fn_name,
                          res_index));
  }
};

template <typename T>
struct AccuracyCheckFunctor<phi::CPUContext, phi::dtype::complex<T>> {
  void operator()(const phi::CPUContext& ctx,
                  const DenseTensor& in,
                  const DenseTensor& other,
                  const std::string& fn_name,
                  const double rtol,
                  const double atol,
                  bool equal_nan,
                  DenseTensor* output) {
    auto* in_a = in.data<phi::dtype::complex<T>>();
    auto* in_b = other.data<phi::dtype::complex<T>>();
    auto* out_data = ctx.template Alloc<bool>(output);
    auto num = in.numel();
    // *out_data = true;
    for (int i = 0; i < num; i++) {
      out_data[i] = true;
    }
    bool val;
    int res_index = -1;
    for (int i = 0; i < num; i++) {
      const phi::dtype::complex<T> a = in_a[i], b = in_b[i];
      if (std::isnan(a) || std::isnan(b)) {
        val = equal_nan && std::isnan(a) == std::isnan(b);
      } else {
        T left = abs(a - b);
        T right = atol + rtol * abs(b);
        T diff = abs(left - right);
        val = a == b || left <= right || diff <= 1e-10;
        // *out_data &= val;
        out_data[i] = val;
        if (!val) {
          res_index = i;
          break;
        }
      }
    }
    PADDLE_ENFORCE_EQ(val,
                      true,
                      common::errors::PreconditionNotMet(
                          "Accuracy check failed, kernel name %s, res index %d",
                          fn_name,
                          res_index));
  }
};

#if defined(__NVCC__) || defined(__HIPCC__)
template <typename T>
__global__ void AccuracyCheckCUDAKernel(const T* in_data,
                                        const T* other_data,
                                        const double rtol,
                                        const double atol,
                                        bool equal_nan,
                                        int num,
                                        bool* out_data) {
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
  bool val;
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
  for (int i = idx; i < num; i += blockDim.x * gridDim.x) {
    const double a = static_cast<MPType>(in_data[i]);
    const double b = static_cast<MPType>(other_data[i]);
    if (isnan(a) || isnan(b)) {
      val = equal_nan && isnan(a) == isnan(b);
    } else {
      double left = (a > b ? a - b : b - a);
      double right = atol + (b > 0 ? rtol * b : (-rtol) * b);
      double diff = (left > right ? left - right : right - left);
      val = a == b || left <= right || diff <= 1e-10;
    }
    out_data[i] = val;
    if (!val) {
      *out_data = false;
      break;
    }
  }
}
template <>
__global__ void AccuracyCheckCUDAKernel<phi::dtype::complex<float>>(
    const phi::dtype::complex<float>* in_data,
    const phi::dtype::complex<float>* other_data,
    const double rtol,
    const double atol,
    bool equal_nan,
    int num,
    bool* out_data) {
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
  bool val;
  for (int i = idx; i < num; i += blockDim.x * gridDim.x) {
    const phi::dtype::complex<float> a = in_data[i];
    const phi::dtype::complex<float> b = other_data[i];
    if (isnan(a) || isnan(b)) {
      val = equal_nan && isnan(a) == isnan(b);
    } else {
      float left = abs(a - b);
      float right = atol + rtol * abs(b);
      float diff = abs(left - right);
      val = a == b || left <= right || diff <= 1e-10;
    }
    out_data[i] = val;
    // if (!val) {
    //   *out_data = false;
    //   break;
    // }
  }
}

template <>
__global__ void AccuracyCheckCUDAKernel<phi::dtype::complex<double>>(
    const phi::dtype::complex<double>* in_data,
    const phi::dtype::complex<double>* other_data,
    const double rtol,
    const double atol,
    bool equal_nan,
    int num,
    bool* out_data) {
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
  bool val;
  for (int i = idx; i < num; i += blockDim.x * gridDim.x) {
    const phi::dtype::complex<double> a = in_data[i];
    const phi::dtype::complex<double> b = other_data[i];
    if (isnan(a) || isnan(b)) {
      val = equal_nan && isnan(a) == isnan(b);
    } else {
      double left = abs(a - b);
      double right = atol + rtol * abs(b);
      double diff = abs(left - right);
      val = a == b || left <= right || diff <= 1e-10;
    }
    out_data[i] = val;
    if (!val) {
      *out_data = false;
      break;
    }
  }
}

template <typename T>
struct AccuracyCheckFunctor<phi::GPUContext, T> {
  void operator()(const phi::GPUContext& dev_ctx,
                  const DenseTensor& in,
                  const DenseTensor& other,
                  const std::string& fn_name,
                  const double rtol,
                  const double atol,
                  bool equal_nan,
                  DenseTensor* output) {
    int num = in.numel();
    const T* in_data = in.data<T>();
    const T* other_data = other.data<T>();
    bool* out_data = dev_ctx.template Alloc<bool>(output);
    int block = 1024;
    int grid = (block - 1 + num) / block;
    grid = (grid > block) ? block : grid;
#ifdef PADDLE_WITH_HIP
    hipMemset(out_data, true, num * sizeof(bool));
#else
    cudaMemset(out_data, true, num * sizeof(bool));
#endif
    AccuracyCheckCUDAKernel<T><<<grid, block, 0, dev_ctx.stream()>>>(
        in_data, other_data, rtol, atol, equal_nan, num, out_data);

    DenseTensor out_cpu;
    phi::Copy(dev_ctx, *output, phi::CPUPlace(), true, &out_cpu);
    auto data_ptr = out_cpu.data<bool>();

    // check all data here
    bool check = false;

    std::cerr << "ato l rtol " << rtol << "\t" << atol << std::endl;
    for (int i = 0; i < num; ++i) {
      if (!data_ptr[i]) {
        check = true;
      }
      break;
      std::cerr << data_ptr[i] << " ";
    }
    std::cerr << "\n!!!@@@@@@@@@@\n";

    PADDLE_ENFORCE_EQ(check,
                      false,
                      common::errors::PreconditionNotMet(
                          "Accuracy check failed, kernel name %s", fn_name));
  }
};
#endif

template <typename T, typename Context>
void AccuracyCheckKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         const DenseTensor& y,
                         const std::string& fn_name,
                         const double rtol,
                         const double atol,
                         bool equal_nan,
                         DenseTensor* out) {
  if (fn_name.find("reduce_") != std::string::npos) {
    return;
  }
  AccuracyCheckFunctor<Context, T>()(
      dev_ctx, x, y, fn_name, rtol, atol, equal_nan, out);
}
}  // namespace phi
