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

#include "paddle/phi/kernels/histogram_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

template <typename T, typename Context>
void HistogramKernel(const Context& dev_ctx,
                     const DenseTensor& input,
                     const paddle::optional<DenseTensor>& weight,
                     int64_t bins,
                     int min,
                     int max,
                     bool density,
                     DenseTensor* output) {
  auto& nbins = bins;
  auto& minval = min;
  auto& maxval = max;

  const T* input_data = input.data<T>();
  auto input_numel = input.numel();

  if (input_data == nullptr) {
    dev_ctx.template Alloc<T>(output);
    phi::funcs::SetConstant<Context, T>()(dev_ctx, output, static_cast<T>(0));
    return;
  }

  T output_min = static_cast<T>(minval);
  T output_max = static_cast<T>(maxval);
  if (output_min == output_max) {
    output_min = *std::min_element(input_data, input_data + input_numel);
    output_max = *std::max_element(input_data, input_data + input_numel);
  }
  if (output_min == output_max) {
    output_min = output_min - 1;
    output_max = output_max + 1;
  }

  PADDLE_ENFORCE_EQ((std::isinf(static_cast<float>(output_min)) ||
                     std::isnan(static_cast<float>(output_max)) ||
                     std::isinf(static_cast<float>(output_min)) ||
                     std::isnan(static_cast<float>(output_max))),
                    false,
                    phi::errors::OutOfRange("range of min, max is not finite"));
  PADDLE_ENFORCE_GE(
      output_max,
      output_min,
      phi::errors::InvalidArgument(
          "max must be larger or equal to min. If min and max are both zero, "
          "the minimum and maximum values of the data are used. "
          "But received max is %d, min is %d",
          maxval,
          minval));

  bool has_weight = weight.is_initialized();
  auto weight_data = (weight.get_ptr() == nullptr ? nullptr : weight.get_ptr()->data<T>());

  // compute output
  if (density) {
    T total = static_cast<T>(0);
    for(int64_t i = 0; i < input_numel; i++) {
      if (input_data[i] >= output_min && input_data[i] <= output_max) {
        total += has_weight ? static_cast<T>(weight_data[i]) : static_cast<T>(1);
      }
    }
    float* out_data = dev_ctx.template Alloc<float>(output);
    phi::funcs::SetConstant<Context, float>()(dev_ctx, output, static_cast<float>(0));

    const float interval_len = static_cast<float>(output_max - output_min) / nbins;
    for (int64_t i = 0; i < input_numel; i++) {
      if (input_data[i] >= output_min && input_data[i] <= output_max) {
        const int64_t bin = (int64_t)((input_data[i] - output_min) * nbins /
                                      (output_max - output_min));
        T weight_idx = weight_data == nullptr ? 1 : weight_data[i];
        out_data[std::min(bin, nbins - 1)] += (static_cast<float>(weight_idx)
                                              / total) / interval_len;
      }
    }
  } else {
    T* out_data = dev_ctx.template Alloc<T>(output);
    phi::funcs::SetConstant<Context, T>()(dev_ctx, output, static_cast<T>(0));
    for (int64_t i = 0; i < input_numel; i++) {
      if (input_data[i] >= output_min && input_data[i] <= output_max) {
        const int64_t bin = (int64_t)((input_data[i] - output_min) * nbins /
                                      (output_max - output_min));
        T weight_idx = weight_data == nullptr ? 1 : weight_data[i];
        out_data[std::min(bin, nbins - 1)] += weight_idx;
      } 
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(histogram,
                   CPU,
                   ALL_LAYOUT,
                   phi::HistogramKernel,
                   float,
                   double,
                   int,
                   int64_t) {
  kernel->OutputAt(0).SetDataType(phi::DataType::UNDEFINED);
}
