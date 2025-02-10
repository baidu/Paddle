/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/kernels/strings/strings_lower_upper_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/common/pstring.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/strings/unicode.h"

using pstring = ::phi::dtype::pstring;
namespace phi {
namespace strings {

template <typename CharConverter>
__global__ void StringCaseConvertCUDAKernel(
    pstring* out,
    const pstring* in,
    size_t num,
    const stream_attachment_type& exec_policy) {
  CUDA_KERNEL_LOOP(i, num) {
    out[i] = pstring(in[i]);
    thrust::transform(exec_policy,
                      in[i].begin(),
                      in[i].end(),
                      out[i].mdata(),
                      CharConverter());
  }
}

template <typename CharConverter>
struct AsciiCaseConverter<phi::GPUContext, CharConverter> {
  void operator()(const phi::GPUContext& dev_ctx,
                  const pstring* in,
                  pstring* out,
                  size_t num) const {
#ifdef PADDLE_WITH_HIP
    dim3 block_size = dim3(256, 1);
    const auto& exec_policy = thrust::hip::par.on(dev_ctx.stream());
#else
    dim3 block_size = dim3(PREDEFINED_BLOCK_SIZE, 1);
    const auto& exec_policy = thrust::cuda::par.on(dev_ctx.stream());
#endif
    dim3 grid_size =
        dim3((num + PREDEFINED_BLOCK_SIZE - 1) / PREDEFINED_BLOCK_SIZE, 1);
    StringCaseConvertCUDAKernel<CharConverter>
        <<<grid_size, block_size, 0, dev_ctx.stream()>>>(
            out, in, num, exec_policy);
  }
};

template <template <typename DeviceContextT> typename CharConverter>
struct UTF8CaseConverter<phi::GPUContext, CharConverter> {
  void operator()(const phi::GPUContext& dev_ctx,
                  const pstring* in,
                  pstring* out,
                  size_t num) const {
#ifdef PADDLE_WITH_CUDA
    const auto& exec_policy = thrust::cuda::par.on(dev_ctx.stream());
#else
    const auto& exec_policy = thrust::hip::par.on(dev_ctx.stream());
#endif
    auto unicode_flag_map = GetGPUUniflagMap();
    auto cases_map = GetGPUCharCasesMap();
    thrust::device_vector<uint32_t> unicode_offsets(num + 1, 0);
    uint32_t* unicode_offsets_ptr =
        thrust::raw_pointer_cast(unicode_offsets.data());

    thrust::for_each_n(exec_policy,
                       thrust::make_counting_iterator<unsigned int>(0),
                       num,
                       [unicode_offsets_ptr, in] __device__(uint32_t idx) {
                         unicode_offsets_ptr[idx + 1] =
                             GetUnicodeStrLen(in[idx].data(), in[idx].size());
                       });
    uint32_t total_lengths = thrust::reduce(
        exec_policy, unicode_offsets_ptr, unicode_offsets_ptr + num + 1, 0);
    if (total_lengths == 0) {
      return;
    }

    thrust::device_vector<uint32_t> unicode_output(total_lengths, 0);
    uint32_t* unicode_output_ptr =
        thrust::raw_pointer_cast(unicode_output.data());

    CharConverter<GPUContext> converter(unicode_flag_map, cases_map);
    thrust::for_each_n(
        exec_policy,
        thrust::make_counting_iterator<unsigned int>(0),
        num,
        [in,
         out,
         unicode_output_ptr,
         unicode_offsets_ptr,
         converter] __device__(uint32_t idx) {
          uint32_t unicode_len =
              unicode_offsets_ptr[idx + 1] - unicode_offsets_ptr[idx];
          GetUnicodeStr(in[idx].data(),
                        unicode_output_ptr + unicode_offsets_ptr[idx],
                        unicode_len);
          uint32_t* curr_unicode_output_ptr =
              unicode_output_ptr + unicode_offsets_ptr[idx];
          for (uint32_t i = 0; i < unicode_len; ++i) {
            curr_unicode_output_ptr[i] = converter(curr_unicode_output_ptr[i]);
          }
          thrust::transform(exec_policy,
                            unicode_output_ptr + unicode_offsets_ptr[idx],
                            unicode_output_ptr + unicode_offsets_ptr[idx + 1],
                            unicode_output_ptr + unicode_offsets_ptr[idx],
                            converter);
        });

    thrust::device_vector<uint32_t> utf8_offsets(num + 1, 0);
    uint32_t* utf8_offsets_ptr = thrust::raw_pointer_cast(utf8_offsets.data());

    thrust::for_each_n(
        exec_policy,
        thrust::make_counting_iterator<unsigned int>(0),
        num,
        [utf8_offsets_ptr, unicode_output_ptr, unicode_offsets_ptr] __device__(
            uint32_t idx) {
          uint32_t unicode_len =
              unicode_offsets_ptr[idx + 1] - unicode_offsets_ptr[idx];
          utf8_offsets_ptr[idx + 1] = GetUTF8StrLen(
              unicode_output_ptr + unicode_offsets_ptr[idx], unicode_len);
        });
    uint32_t total_utf8_lengths = thrust::reduce(
        exec_policy, utf8_offsets_ptr, utf8_offsets_ptr + num + 1, 0);

    thrust::device_vector<char> utf8_output(total_utf8_lengths, 0);
    char* utf8_output_ptr = thrust::raw_pointer_cast(utf8_output.data());
    thrust::for_each_n(exec_policy,
                       thrust::make_counting_iterator<unsigned int>(0),
                       num,
                       [utf8_output_ptr,
                        utf8_offsets_ptr,
                        unicode_output_ptr,
                        unicode_offsets_ptr,
                        out] __device__(uint32_t idx) {
                         uint32_t unicode_len = unicode_offsets_ptr[idx + 1] -
                                                unicode_offsets_ptr[idx];
                         const uint32_t* input_ptr =
                             unicode_output_ptr + unicode_offsets_ptr[idx];
                         char* result_ptr =
                             utf8_output_ptr + utf8_offsets_ptr[idx];
                         GetUTF8Str(input_ptr, result_ptr, unicode_len);
                         out[idx] = result_ptr;
                       });
  }
};

template <typename ContextT>
void StringLowerKernel(const ContextT& dev_ctx,
                       const StringTensor& x,
                       bool use_utf8_encoding,
                       StringTensor* out) {
  StringCaseConvertKernel<AsciiCaseConverter<ContextT, AsciiToLower>,
                          UTF8CaseConverter<ContextT, UTF8ToLower>,
                          ContextT>()(dev_ctx, x, use_utf8_encoding, out);
}

template <typename ContextT>
void StringUpperKernel(const ContextT& dev_ctx,
                       const StringTensor& x,
                       bool use_utf8_encoding,
                       StringTensor* out) {
  StringCaseConvertKernel<AsciiCaseConverter<ContextT, AsciiToUpper>,
                          UTF8CaseConverter<ContextT, UTF8ToUpper>,
                          ContextT>()(dev_ctx, x, use_utf8_encoding, out);
}

}  // namespace strings
}  // namespace phi

PD_REGISTER_KERNEL_FOR_ALL_DTYPE(
    strings_lower,
    GPU,
    ALL_LAYOUT,
    phi::strings::StringLowerKernel<phi::GPUContext>) {}

PD_REGISTER_KERNEL_FOR_ALL_DTYPE(
    strings_upper,
    GPU,
    ALL_LAYOUT,
    phi::strings::StringUpperKernel<phi::GPUContext>) {}
