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

#include "paddle/phi/core/tensor_utils.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/compat/convert_utils.h"
#include "paddle/phi/core/kernel_registry.h"

// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/memory/malloc.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/platform/device_context.h"

namespace phi {

template <typename Context>
void Copy(const Context& dev_ctx,
          const DenseTensor& src,
          Place dst_place,
          bool blocking,
          DenseTensor* dst) {
  auto* src_ptr = src.data();
  const auto& src_place = src.place();

  if (&src == dst) {
    if (paddle::platform::is_same_place(src_place, dst_place)) {
      VLOG(6) << "Skip copy the same data(" << src_ptr << ") from " << src_place
              << " to " << dst_place;
    } else {
      VLOG(6) << "Src and dst are the same Tensor, in-place copy data("
              << src_ptr << ") from " << src_place << " to " << dst_place;
      const DenseTensor src_copy = src;
      Copy(dev_ctx, src_copy, dst_place, blocking, dst);
    }
    return;
  }

  VLOG(3) << "TensorCopy " << src.dims() << " from " << src.place() << " to "
          << dst_place;

  dst->Resize(src.dims());

  void* dst_ptr = nullptr;
  if (paddle::platform::is_cpu_place(dst_place)) {
    dst_ptr = dev_ctx.HostAlloc(dst, src.dtype());
#ifdef PADDLE_WITH_MKLDNN
    dst->set_layout(src.layout());
#endif
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  } else if (paddle::platform::is_gpu_place(dst_place) ||
             paddle::platform::is_cuda_pinned_place(dst_place)) {
    dst_ptr = dev_ctx.Alloc(
        dst, src.dtype(), 0, paddle::platform::is_cuda_pinned_place(dst_place));
#endif

#ifdef PADDLE_WITH_XPU
  } else if (paddle::platform::is_xpu_place(dst_place)) {
    dst_ptr = dev_ctx.Alloc(dst, src.dtype());
#endif
#ifdef PADDLE_WITH_CUSTOM_DEVICE
  } else if (paddle::platform::is_custom_place(dst_place)) {
    dst_ptr = dev_ctx.Alloc(dst, src.dtype());
#endif
  }

  auto size = src.numel() * paddle::experimental::SizeOf(src.dtype());
  if (UNLIKELY(size) == 0) {
    return;
  }

  PADDLE_ENFORCE_EQ(
      dst->place(),
      dst_place,
      errors::Unavailable(
          "The Dst Tensor's place and dst_place do not match, Tensor's place "
          "place is %s, dst_place is %s.",
          dst->place(),
          dst_place));

  if (src_ptr == dst_ptr && src_place == dst_place) {
    VLOG(3) << "Skip copy the same data async from " << src_place << " to "
            << dst_place;
    return;
  }
  VLOG(4) << "src:" << src_ptr << ", dst:" << dst_ptr;
  CHECK(dst->layout() == src.layout());

  if (paddle::platform::is_cpu_place(src_place) &&
      paddle::platform::is_cpu_place(dst_place)) {
    paddle::memory::Copy(src_place, dst_ptr, src_place, src_ptr, size);
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  } else if ((paddle::platform::is_cpu_place(src_place) ||
              paddle::platform::is_cuda_pinned_place(src_place)) &&  // NOLINT
             (paddle::platform::is_cpu_place(dst_place) ||
              paddle::platform::is_cuda_pinned_place(dst_place))) {
    paddle::memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size, nullptr);
  } else if (paddle::platform::is_gpu_place(src_place) &&  // NOLINT
             paddle::platform::is_cpu_place(dst_place)) {
    auto src_gpu_place = src_place;
    auto dst_cpu_place = dst_place;
    auto ctx_place = dev_ctx.GetPlace();
    PADDLE_ENFORCE_EQ(
        paddle::platform::is_gpu_place(ctx_place),
        true,
        errors::PreconditionNotMet(
            "Context place error, excepted GPUPlace, but actually %s.",
            ctx_place));
    auto ctx_gpu_place = ctx_place;
    PADDLE_ENFORCE_EQ(src_gpu_place,
                      ctx_gpu_place,
                      errors::Unavailable(
                          "Source place and context place do not match, source "
                          "place is %s, context place is %s.",
                          src_gpu_place,
                          ctx_gpu_place));
    auto stream =
        blocking ? nullptr
                 : reinterpret_cast<const phi::GPUContext&>(dev_ctx).stream();
    paddle::memory::Copy(
        dst_cpu_place, dst_ptr, src_gpu_place, src_ptr, size, stream);
  } else if ((paddle::platform::is_cpu_place(src_place) ||
              paddle::platform::is_cuda_pinned_place(src_place)) &&  // NOLINT
             paddle::platform::is_gpu_place(dst_place)) {
    auto src_cpu_place = src_place;
    auto dst_gpu_place = dst_place;
    auto ctx_place = dev_ctx.GetPlace();
    PADDLE_ENFORCE_EQ(
        paddle::platform::is_gpu_place(ctx_place),
        true,
        errors::PreconditionNotMet(
            "Context place error, excepted GPUPlace, but actually %s.",
            ctx_place));
    auto ctx_gpu_place = ctx_place;
    PADDLE_ENFORCE_EQ(
        dst_gpu_place,
        ctx_gpu_place,
        errors::Unavailable("Destination place and context place do not match, "
                            "destination place is %s, context place is %s.",
                            dst_gpu_place,
                            ctx_gpu_place));
    auto stream =
        blocking ? nullptr
                 : reinterpret_cast<const phi::GPUContext&>(dev_ctx).stream();
    paddle::memory::Copy(
        dst_gpu_place, dst_ptr, src_cpu_place, src_ptr, size, stream);
  } else if (paddle::platform::is_gpu_place(src_place) &&  // NOLINT
             paddle::platform::is_gpu_place(dst_place)) {
    auto src_gpu_place = src_place;
    auto dst_gpu_place = dst_place;
    auto ctx_place = dev_ctx.GetPlace();
    PADDLE_ENFORCE_EQ(
        paddle::platform::is_gpu_place(ctx_place),
        true,
        errors::PreconditionNotMet(
            "Context place error, excepted GPUPlace, but actually %s.",
            ctx_place));
    auto stream =
        blocking ? nullptr
                 : reinterpret_cast<const phi::GPUContext&>(dev_ctx).stream();
    if (paddle::platform::is_same_place(src_place, dst_place)) {
      paddle::memory::Copy(
          dst_gpu_place, dst_ptr, src_gpu_place, src_ptr, size, stream);
    } else {
      if (paddle::platform::is_same_place(ctx_place, src_place)) {
        paddle::memory::Copy(
            dst_gpu_place, dst_ptr, src_gpu_place, src_ptr, size, stream);
        paddle::platform::DeviceContextPool::Instance()
            .Get(src.place())
            ->Wait();
      } else if (paddle::platform::is_same_place(ctx_place, dst_place)) {
        paddle::platform::DeviceContextPool::Instance()
            .Get(src.place())
            ->Wait();
        paddle::memory::Copy(
            dst_gpu_place, dst_ptr, src_gpu_place, src_ptr, size, stream);
      } else {
        PADDLE_THROW(errors::Unavailable(
            "Context place dose not match the source and destination place."));
      }
    }
  } else if (paddle::platform::is_gpu_place(src_place) &&  // NOLINT
             paddle::platform::is_cuda_pinned_place(dst_place)) {
    auto src_gpu_place = src_place;
    auto dst_cuda_pinned_place = dst_place;
    auto ctx_place = dev_ctx.GetPlace();
    PADDLE_ENFORCE_EQ(
        paddle::platform::is_gpu_place(ctx_place),
        true,
        errors::PreconditionNotMet(
            "Context place error, excepted GPUPlace, but actually %s.",
            ctx_place));
    auto ctx_gpu_place = ctx_place;
    PADDLE_ENFORCE_EQ(src_gpu_place,
                      ctx_gpu_place,
                      errors::Unavailable(
                          "Source place and context place do not match, source "
                          "place is %s, context place is %s.",
                          src_gpu_place,
                          ctx_gpu_place));
    auto stream =
        blocking ? nullptr
                 : reinterpret_cast<const phi::GPUContext&>(dev_ctx).stream();
    paddle::memory::Copy(
        dst_cuda_pinned_place, dst_ptr, src_gpu_place, src_ptr, size, stream);
#endif
#ifdef PADDLE_WITH_XPU
  } else if (paddle::platform::is_xpu_place(src_place) &&  // NOLINT
             paddle::platform::is_cpu_place(dst_place)) {
    paddle::memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size);
  } else if (paddle::platform::is_cpu_place(src_place) &&
             paddle::platform::is_xpu_place(dst_place)) {
    paddle::memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size);
  } else if (paddle::platform::is_xpu_place(src_place) &&
             paddle::platform::is_xpu_place(dst_place)) {
    if (src_ptr == dst_ptr) {
      VLOG(3) << "Skip copy the same data async from " << src_place << " to "
              << dst_place;
      return;
    }
    paddle::memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size);
#endif
#ifdef PADDLE_WITH_CUSTOM_DEVICE
  } else if (paddle::platform::is_custom_place(src_place) &&  // NOLINT
             paddle::platform::is_cpu_place(dst_place)) {
    auto stream =
        blocking
            ? nullptr
            : reinterpret_cast<const paddle::platform::CustomDeviceContext&>(
                  dev_ctx)
                  .stream();
    paddle::memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size, stream);
  } else if (paddle::platform::is_cpu_place(src_place) &&  // NOLINT
             paddle::platform::is_custom_place(dst_place)) {
    auto stream =
        blocking
            ? nullptr
            : reinterpret_cast<const paddle::platform::CustomDeviceContext&>(
                  dev_ctx)
                  .stream();
    paddle::memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size, stream);
  } else if (paddle::platform::is_custom_place(src_place) &&  // NOLINT
             paddle::platform::is_custom_place(dst_place)) {
    auto stream =
        blocking
            ? nullptr
            : reinterpret_cast<const paddle::platform::CustomDeviceContext&>(
                  dev_ctx)
                  .stream();
    paddle::memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size, stream);
#endif
  } else {
    PADDLE_THROW(errors::Unimplemented(
        "Copy from %s to %s is not supported.", src_place, dst_place));
  }
}

template <typename Context>
void Copy(const Context& dev_ctx,
          const SelectedRows& src,
          Place dst_place,
          bool blocking,
          SelectedRows* dst) {
  if (src.value().Holder() != dst->value().Holder() ||
      src.value().data() != dst->value().data()) {
    dst->set_rows(src.rows());
    dst->set_height(src.height());
  }
  Copy<Context>(
      dev_ctx, src.value(), dst_place, blocking, dst->mutable_value());
}

template <typename Context>
void Copy(const Context& dev_ctx,
          const SparseCooTensor& src,
          Place dst_place,
          bool blocking,
          SparseCooTensor* dst) {
  phi::Copy<Context>(dev_ctx,
                     src.non_zero_indices(),
                     dst_place,
                     blocking,
                     dst->mutable_non_zero_indices());

  phi::Copy<Context>(dev_ctx,
                     src.non_zero_elements(),
                     dst_place,
                     blocking,
                     dst->mutable_non_zero_elements());
  dst->set_meta(src.meta());
  dst->SetCoalesced(src.coalesced());
}

template <typename Context>
void Copy(const Context& dev_ctx,
          const SparseCsrTensor& src,
          Place dst_place,
          bool blocking,
          SparseCsrTensor* dst) {
  phi::Copy<Context>(dev_ctx,
                     src.non_zero_crows(),
                     dst_place,
                     blocking,
                     dst->mutable_non_zero_crows());

  phi::Copy<Context>(dev_ctx,
                     src.non_zero_cols(),
                     dst_place,
                     blocking,
                     dst->mutable_non_zero_cols());

  phi::Copy<Context>(dev_ctx,
                     src.non_zero_elements(),
                     dst_place,
                     blocking,
                     dst->mutable_non_zero_elements());
  dst->set_dims(src.dims());
}

template void Copy(const CPUContext& dev_ctx,
                   const DenseTensor& src,
                   Place dst_place,
                   bool blocking,
                   DenseTensor* dst);

template void Copy(const DeviceContext& dev_ctx,
                   const DenseTensor& src,
                   Place dst_place,
                   bool blocking,
                   DenseTensor* dst);

template void Copy(const CPUContext& dev_ctx,
                   const SelectedRows& src,
                   Place dst_place,
                   bool blocking,
                   SelectedRows* dst);
template void Copy(const DeviceContext& dev_ctx,
                   const SelectedRows& src,
                   Place dst_place,
                   bool blocking,
                   SelectedRows* dst);

template void Copy(const CPUContext& dev_ctx,
                   const SparseCooTensor& src,
                   Place dst_place,
                   bool blocking,
                   SparseCooTensor* dst);

template void Copy(const DeviceContext& dev_ctx,
                   const SparseCooTensor& src,
                   Place dst_place,
                   bool blocking,
                   SparseCooTensor* dst);

template void Copy(const CPUContext& dev_ctx,
                   const SparseCsrTensor& src,
                   Place dst_place,
                   bool blocking,
                   SparseCsrTensor* dst);

template void Copy(const DeviceContext& dev_ctx,
                   const SparseCsrTensor& src,
                   Place dst_place,
                   bool blocking,
                   SparseCsrTensor* dst);

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
template void Copy(const GPUContext& dev_ctx,
                   const DenseTensor& src,
                   Place dst_place,
                   bool blocking,
                   DenseTensor* dst);
template void Copy(const GPUContext& dev_ctx,
                   const SelectedRows& src,
                   Place dst_place,
                   bool blocking,
                   SelectedRows* dst);
template void Copy(const GPUContext& dev_ctx,
                   const SparseCooTensor& src,
                   Place dst_place,
                   bool blocking,
                   SparseCooTensor* dst);
template void Copy(const GPUContext& dev_ctx,
                   const SparseCsrTensor& src,
                   Place dst_place,
                   bool blocking,
                   SparseCsrTensor* dst);
#endif

#ifdef PADDLE_WITH_XPU
template void Copy(const XPUContext& dev_ctx,
                   const DenseTensor& src,
                   Place dst_place,
                   bool blocking,
                   DenseTensor* dst);
#endif

#ifdef PADDLE_WITH_CUSTOM_DEVICE
template void Copy(const CustomContext& dev_ctx,
                   const DenseTensor& src,
                   Place dst_place,
                   bool blocking,
                   DenseTensor* dst);
#endif

#ifdef PADDLE_WITH_MKLDNN
template void Copy(const OneDNNContext& dev_ctx,
                   const DenseTensor& src,
                   Place dst_place,
                   bool blocking,
                   DenseTensor* dst);
#endif

template <typename T>
void TensorFromVector(const std::vector<T>& src,
                      const phi::DeviceContext& ctx,
                      phi::DenseTensor* dst) {
  auto dst_place = ctx.GetPlace();
  auto src_ptr = static_cast<const void*>(src.data());
  phi::CPUPlace src_place;
  dst->Resize({static_cast<int64_t>(src.size())});
  ctx.template Alloc<T>(dst);
  auto dst_ptr = static_cast<void*>(dst->data<T>());
  auto size = src.size() * sizeof(T);

  if (paddle::platform::is_cpu_place(dst_place)) {
    paddle::memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size);
  }
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  else if (paddle::platform::is_gpu_place(dst_place)) {  // NOLINT
    paddle::memory::Copy(
        dst_place,
        dst_ptr,
        src_place,
        src_ptr,
        size,
        reinterpret_cast<const phi::GPUContext&>(ctx).stream());
  }
#endif
#ifdef PADDLE_WITH_CUSTOM_DEVICE
  else if (paddle::platform::is_custom_place(dst_place)) {  // NOLINT
    paddle::memory::Copy(
        dst_place,
        dst_ptr,
        src_place,
        src_ptr,
        size,
        reinterpret_cast<const phi::CustomContext&>(ctx).stream());
  }
#endif
#ifdef PADDLE_WITH_XPU
  else if (paddle::platform::is_xpu_place(dst_place)) {  // NOLINT
    paddle::memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size);
  }
#endif
  else {  // NOLINT
    PADDLE_THROW(phi::errors::Unimplemented(
        "TensorFromVector on %s is not supported.", dst_place));
  }
}

template <>
void TensorFromVector(const std::vector<bool>& src,
                      const phi::DeviceContext& ctx,
                      phi::DenseTensor* dst) {
  // vector<bool> has no data() member, use array instead.
  // See details:
  // https://stackoverflow.com/questions/46115669/why-does-stdvectorbool-have-no-data/46115714
  bool* array = new bool[src.size()];
  for (unsigned int i = 0; i < src.size(); i++) {
    array[i] = static_cast<bool>(src[i]);
  }

  auto dst_place = ctx.GetPlace();
  auto src_ptr = static_cast<const void*>(array);
  phi::CPUPlace src_place{};
  dst->Resize({static_cast<int64_t>(src.size())});
  auto dst_ptr = ctx.template Alloc<bool>(dst);
  auto size = src.size() * sizeof(bool);

  if (paddle::platform::is_cpu_place(dst_place)) {
    paddle::memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size);
  }
#ifdef PADDLE_WITH_CUDA
  else if (paddle::platform::is_gpu_place(dst_place)) {  // NOLINT
    paddle::memory::Copy(
        dst_place,
        dst_ptr,
        src_place,
        src_ptr,
        size,
        reinterpret_cast<const phi::GPUContext&>(ctx).stream());
  }
#endif
#ifdef PADDLE_WITH_CUSTOM_DEVICE
  else if (paddle::platform::is_custom_place(dst_place)) {  // NOLINT
    auto stream = reinterpret_cast<const phi::CustomContext&>(ctx).stream();
    paddle::memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size, stream);
  }
#endif
#ifdef PADDLE_WITH_XPU
  else if (paddle::platform::is_xpu_place(dst_place)) {  // NOLINT
    paddle::memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size);
  }
#endif
  else {  // NOLINT
    PADDLE_THROW(phi::errors::Unimplemented(
        "TensorFromVector on %s is not supported.", dst_place));
  }
  delete[] array;
}

template void TensorFromVector<int8_t>(const std::vector<int8_t>& src,
                                       const phi::DeviceContext& ctx,
                                       phi::DenseTensor* dst);

template void TensorFromVector<uint8_t>(const std::vector<uint8_t>& src,
                                        const phi::DeviceContext& ctx,
                                        phi::DenseTensor* dst);

template void TensorFromVector<int16_t>(const std::vector<int16_t>& src,
                                        const phi::DeviceContext& ctx,
                                        phi::DenseTensor* dst);

template void TensorFromVector<int>(const std::vector<int>& src,
                                    const phi::DeviceContext& ctx,
                                    phi::DenseTensor* dst);

template void TensorFromVector<int64_t>(const std::vector<int64_t>& src,
                                        const phi::DeviceContext& ctx,
                                        phi::DenseTensor* dst);

template void TensorFromVector<float>(const std::vector<float>& src,
                                      const phi::DeviceContext& ctx,
                                      phi::DenseTensor* dst);

template void TensorFromVector<double>(const std::vector<double>& src,
                                       const phi::DeviceContext& ctx,
                                       phi::DenseTensor* dst);

template void TensorFromVector<phi::dtype::bfloat16>(
    const std::vector<phi::dtype::bfloat16>& src,
    const phi::DeviceContext& ctx,
    phi::DenseTensor* dst);

template void TensorFromVector<phi::dtype::float16>(
    const std::vector<phi::dtype::float16>& src,
    const phi::DeviceContext& ctx,
    phi::DenseTensor* dst);

template void TensorFromVector<phi::dtype::complex<float>>(
    const std::vector<phi::dtype::complex<float>>& src,
    const phi::DeviceContext& ctx,
    phi::DenseTensor* dst);

template void TensorFromVector<phi::dtype::complex<double>>(
    const std::vector<phi::dtype::complex<double>>& src,
    const phi::DeviceContext& ctx,
    phi::DenseTensor* dst);

template <typename T>
void TensorFromArray(const T* src,
                     const size_t& array_size,
                     const phi::DeviceContext& ctx,
                     phi::DenseTensor* dst) {
  auto dst_place = ctx.GetPlace();
  auto src_ptr = static_cast<const void*>(src);
  phi::CPUPlace src_place;
  dst->Resize({static_cast<int64_t>(array_size)});
  ctx.template Alloc<T>(dst);
  auto dst_ptr = static_cast<void*>(dst->data<T>());
  auto size = array_size * sizeof(T);

  if (paddle::platform::is_cpu_place(dst_place)) {
    paddle::memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size);
  }
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  else if (paddle::platform::is_gpu_place(dst_place)) {  // NOLINT
    paddle::memory::Copy(
        dst_place,
        dst_ptr,
        src_place,
        src_ptr,
        size,
        reinterpret_cast<const phi::GPUContext&>(ctx).stream());
  }
#endif
#ifdef PADDLE_WITH_CUSTOM_DEVICE
  else if (paddle::platform::is_custom_place(dst_place)) {  // NOLINT
    paddle::memory::Copy(
        dst_place,
        dst_ptr,
        src_place,
        src_ptr,
        size,
        reinterpret_cast<const phi::CustomContext&>(ctx).stream());
  }
#endif
#ifdef PADDLE_WITH_XPU
  else if (paddle::platform::is_xpu_place(dst_place)) {  // NOLINT
    paddle::memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size);
  }
#endif
  else {  // NOLINT
    PADDLE_THROW(phi::errors::Unimplemented(
        "TensorFromArray on %s is not supported.", dst_place));
  }
}

template void TensorFromArray<bool>(const bool* src,
                                    const size_t& array_size,
                                    const phi::DeviceContext& ctx,
                                    phi::DenseTensor* dst);

template void TensorFromArray<int16_t>(const int16_t* src,
                                       const size_t& array_size,
                                       const phi::DeviceContext& ctx,
                                       phi::DenseTensor* dst);

template void TensorFromArray<int>(const int* src,
                                   const size_t& array_size,
                                   const phi::DeviceContext& ctx,
                                   phi::DenseTensor* dst);

template void TensorFromArray<int64_t>(const int64_t* src,
                                       const size_t& array_size,
                                       const phi::DeviceContext& ctx,
                                       phi::DenseTensor* dst);

template void TensorFromArray<float>(const float* src,
                                     const size_t& array_size,
                                     const phi::DeviceContext& ctx,
                                     phi::DenseTensor* dst);

template void TensorFromArray<double>(const double* src,
                                      const size_t& array_size,
                                      const phi::DeviceContext& ctx,
                                      phi::DenseTensor* dst);

template void TensorFromArray<phi::dtype::bfloat16>(
    const phi::dtype::bfloat16* src,
    const size_t& array_size,
    const phi::DeviceContext& ctx,
    phi::DenseTensor* dst);

template void TensorFromArray<phi::dtype::float16>(
    const phi::dtype::float16* src,
    const size_t& array_size,
    const phi::DeviceContext& ctx,
    phi::DenseTensor* dst);

template void TensorFromArray<phi::dtype::complex<float>>(
    const phi::dtype::complex<float>* src,
    const size_t& array_size,
    const phi::DeviceContext& ctx,
    phi::DenseTensor* dst);

template void TensorFromArray<phi::dtype::complex<double>>(
    const phi::dtype::complex<double>* src,
    const size_t& array_size,
    const phi::DeviceContext& ctx,
    phi::DenseTensor* dst);

template <typename T>
void TensorToVector(const phi::DenseTensor& src,
                    const phi::DeviceContext& ctx,
                    std::vector<T>* dst) {
  auto src_ptr = static_cast<const void*>(src.data<T>());
  auto size = src.numel() * sizeof(T);

  phi::CPUPlace dst_place{};
  dst->resize(src.numel());
  auto dst_ptr = static_cast<void*>(dst->data());

  if (paddle::platform::is_cpu_place(src.place())) {
    paddle::memory::Copy(dst_place, dst_ptr, src.place(), src_ptr, size);
  }
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  else if (paddle::platform::is_gpu_place(src.place())) {  // NOLINT
    paddle::memory::Copy(
        dst_place,
        dst_ptr,
        src.place(),
        src_ptr,
        size,
        reinterpret_cast<const phi::GPUContext&>(ctx).stream());
  }
#endif
#if defined(PADDLE_WITH_XPU)
  else if (paddle::platform::is_xpu_place(src.place())) {  // NOLINT
    paddle::memory::Copy(dst_place, dst_ptr, src.place(), src_ptr, size);
  }
#endif
#ifdef PADDLE_WITH_CUSTOM_DEVICE
  else if (paddle::platform::is_custom_place(src.place())) {  // NOLINT
    paddle::memory::Copy(
        dst_place, dst_ptr, src.place(), src_ptr, size, nullptr);
  }
#endif
  else {  // NOLINT
    PADDLE_THROW(phi::errors::Unimplemented(
        "TensorToVector on %s is not supported.", src.place()));
  }
}

template <>
void TensorToVector(const phi::DenseTensor& src,
                    const phi::DeviceContext& ctx,
                    std::vector<bool>* dst) {
  auto src_ptr = static_cast<const void*>(src.data<bool>());
  auto size = src.numel() * sizeof(bool);

  bool* array = new bool[src.numel()];

  phi::CPUPlace dst_place{};
  dst->resize(src.numel());
  auto dst_ptr = static_cast<void*>(array);

  if (paddle::platform::is_cpu_place(src.place())) {
    paddle::memory::Copy(dst_place, dst_ptr, src.place(), src_ptr, size);
  }
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  else if (paddle::platform::is_gpu_place(src.place())) {  // NOLINT
    paddle::memory::Copy(
        dst_place,
        dst_ptr,
        src.place(),
        src_ptr,
        size,
        reinterpret_cast<const phi::GPUContext&>(ctx).stream());
  }
#endif
#if defined(PADDLE_WITH_XPU)
  else if (paddle::platform::is_xpu_place(src.place())) {  // NOLINT
    paddle::memory::Copy(dst_place, dst_ptr, src.place(), src_ptr, size);
  }
#endif
#ifdef PADDLE_WITH_CUSTOM_DEVICE
  else if (paddle::platform::is_custom_place(src.place())) {  // NOLINT
    paddle::memory::Copy(
        dst_place, dst_ptr, src.place(), src_ptr, size, nullptr);
  }
#endif
  for (unsigned int i = 0; i < src.numel(); i++) {
    (*dst)[i] = static_cast<bool>(array[i]);
  }
  delete[] array;
}

template void TensorToVector(const phi::DenseTensor& src,
                             const phi::DeviceContext& ctx,
                             std::vector<int16_t>* dst);

template void TensorToVector(const phi::DenseTensor& src,
                             const phi::DeviceContext& ctx,
                             std::vector<int>* dst);

template void TensorToVector(const phi::DenseTensor& src,
                             const phi::DeviceContext& ctx,
                             std::vector<int64_t>* dst);

template void TensorToVector(const phi::DenseTensor& src,
                             const phi::DeviceContext& ctx,
                             std::vector<float>* dst);

template void TensorToVector(const phi::DenseTensor& src,
                             const phi::DeviceContext& ctx,
                             std::vector<double>* dst);

template void TensorToVector(const phi::DenseTensor& src,
                             const phi::DeviceContext& ctx,
                             std::vector<phi::dtype::bfloat16>* dst);

template void TensorToVector(const phi::DenseTensor& src,
                             const phi::DeviceContext& ctx,
                             std::vector<phi::dtype::float16>* dst);

template void TensorToVector(const phi::DenseTensor& src,
                             const phi::DeviceContext& ctx,
                             std::vector<phi::dtype::complex<float>>* dst);

template void TensorToVector(const phi::DenseTensor& src,
                             const phi::DeviceContext& ctx,
                             std::vector<phi::dtype::complex<double>>* dst);

template <typename T>
void TensorToVector(const phi::DenseTensor& src, std::vector<T>* dst) {
  auto src_ptr = static_cast<const void*>(src.data<T>());
  auto size = src.numel() * sizeof(T);

  phi::CPUPlace dst_place{};
  dst->resize(src.numel());
  auto dst_ptr = static_cast<void*>(dst->data());

  PADDLE_ENFORCE_EQ(
      paddle::platform::is_cpu_place(src.place()),
      true,
      phi::errors::InvalidArgument(
          "The input tensor should be CPU device, but actually it is in %s.",
          src.place()));

  paddle::memory::Copy(dst_place, dst_ptr, src.place(), src_ptr, size);
}

template <>
void TensorToVector(const phi::DenseTensor& src, std::vector<bool>* dst) {
  auto src_ptr = static_cast<const void*>(src.data<bool>());
  auto size = src.numel() * sizeof(bool);

  bool* array = new bool[src.numel()];

  paddle::platform::CPUPlace dst_place{};
  dst->resize(src.numel());
  auto dst_ptr = static_cast<void*>(array);

  PADDLE_ENFORCE_EQ(
      paddle::platform::is_cpu_place(src.place()),
      true,
      phi::errors::InvalidArgument(
          "The input tensor should be CPU device, but actually it is in %s.",
          src.place()));

  paddle::memory::Copy(dst_place, dst_ptr, src.place(), src_ptr, size);

  for (unsigned int i = 0; i < src.numel(); i++) {
    (*dst)[i] = static_cast<bool>(array[i]);
  }
  delete[] array;
}

template void TensorToVector(const phi::DenseTensor& src,
                             std::vector<int16_t>* dst);

template void TensorToVector(const phi::DenseTensor& src,
                             std::vector<int>* dst);

template void TensorToVector(const phi::DenseTensor& src,
                             std::vector<int64_t>* dst);

template void TensorToVector(const phi::DenseTensor& src,
                             std::vector<float>* dst);

template void TensorToVector(const phi::DenseTensor& src,
                             std::vector<double>* dst);

template void TensorToVector(const phi::DenseTensor& src,
                             std::vector<phi::dtype::bfloat16>* dst);

template void TensorToVector(const phi::DenseTensor& src,
                             std::vector<phi::dtype::float16>* dst);

template void TensorToVector(const phi::DenseTensor& src,
                             std::vector<phi::dtype::complex<float>>* dst);

template void TensorToVector(const phi::DenseTensor& src,
                             std::vector<phi::dtype::complex<double>>* dst);

phi::DenseTensor ReshapeToMatrix(const phi::DenseTensor& src,
                                 int num_col_dims) {
  int rank = src.dims().size();
  PADDLE_ENFORCE_GE(
      rank,
      2,
      phi::errors::InvalidArgument(
          "'ReshapeToMatrix()' is only used for flatten high rank "
          "tensors to matrixs. The dimensions of phi::DenseTensor must be "
          "greater or equal than 2. "
          "But received dimensions of phi::DenseTensor is %d",
          rank));
  if (rank == 2) {
    return src;
  }
  phi::DenseTensor res;
  res.ShareDataWith(src);
  res.Resize(phi::flatten_to_2d(src.dims(), num_col_dims));
  return res;
}

// get tensor data point by DLDataType
void* GetDstPtrByDLDataType(DLDataType type,
                            phi::DenseTensor* dst,
                            const phi::Place& dst_place) {
  // vector types not currently supported
  PADDLE_ENFORCE_LE(
      type.lanes,
      1,
      phi::errors::Unimplemented("Vector type is not supported currently."));

  phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
  phi::DeviceContext* dev_ctx = pool.Get(dst->place());

  switch (type.bits) {
    case 8:
      if (type.code == kDLInt)
        return static_cast<void*>(dev_ctx->Alloc<int8_t>(dst));
      if (type.code == kDLUInt)
        return static_cast<void*>(dev_ctx->Alloc<uint8_t>(dst));
      PADDLE_THROW(phi::errors::Unimplemented(
          "DLDataType code <%d> is illegal when DLDataType.bits is <%d>.",
          type.code,
          type.bits));
    case 16:
      if (type.code == kDLInt)
        return static_cast<void*>(dev_ctx->Alloc<int16_t>(dst));
      if (type.code == kDLFloat)
        return static_cast<void*>(dev_ctx->Alloc<phi::dtype::float16>(dst));
      if (type.code == kDLBfloat)
        return static_cast<void*>(dev_ctx->Alloc<phi::dtype::bfloat16>(dst));
      PADDLE_THROW(phi::errors::Unimplemented(
          "DLDataType code <%d> is illegal when DLDataType.bits is <%d>.",
          type.code,
          type.bits));
    case 32:
      if (type.code == kDLInt)
        return static_cast<void*>(dev_ctx->Alloc<int32_t>(dst));
      if (type.code == kDLFloat)
        return static_cast<void*>(dev_ctx->Alloc<float>(dst));
      PADDLE_THROW(phi::errors::Unimplemented(
          "DLDataType code <%d> is illegal when DLDataType.bits is <%d>.",
          type.code,
          type.bits));
    case 64:
      if (type.code == kDLInt)
        return static_cast<void*>(dev_ctx->Alloc<int64_t>(dst));
      if (type.code == kDLFloat)
        return static_cast<void*>(dev_ctx->Alloc<double>(dst));
      if (type.code == kDLComplex)
        return static_cast<void*>(
            dev_ctx->Alloc<phi::dtype::complex<float>>(dst));
      PADDLE_THROW(phi::errors::Unimplemented(
          "DLDataType code <%d> is illegal when DLDataType.bits is <%d>.",
          type.code,
          type.bits));
    case 128:
      if (type.code == kDLComplex)
        return static_cast<void*>(
            dev_ctx->Alloc<phi::dtype::complex<double>>(dst));
      PADDLE_THROW(phi::errors::Unimplemented(
          "DLDataType code <%d> is illegal when DLDataType.bits is <%d>.",
          type.code,
          type.bits));
    default:
      PADDLE_THROW(phi::errors::Unimplemented("Unsupported DLDataType.bits %d.",
                                              type.bits));
  }
}

void TensorFromDLPack(const ::DLTensor& dl_tensor, phi::DenseTensor* dst) {
  phi::CPUPlace dst_place = phi::CPUPlace();
  phi::CPUPlace src_place = phi::CPUPlace();

  std::vector<int64_t> vec;
  std::copy(dl_tensor.shape,
            dl_tensor.shape + dl_tensor.ndim,
            std::back_inserter(vec));

  phi::DDim vddim = phi::make_ddim(vec);

  dst->Resize(vddim);
  ::DLDataType type = dl_tensor.dtype;
  void* dst_ptr = GetDstPtrByDLDataType(type, dst, dst_place);

  auto src_ptr = static_cast<const void*>(dl_tensor.data);
  auto size = phi::product(vddim) * type.bits / 8;

  if (dl_tensor.device.device_type == kDLCPU) {
    paddle::memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size);
  }
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  if (dl_tensor.device.device_type == kDLGPU) {
    phi::CUDAPlace dst_place = phi::CUDAPlace(dl_tensor.device.device_id);
    phi::CUDAPlace src_place = phi::CUDAPlace(dl_tensor.device.device_id);
    dst_ptr = GetDstPtrByDLDataType(type, dst, dst_place);
    auto* ctx = phi::DeviceContextPool::Instance().GetByPlace(dst_place);
    paddle::memory::Copy(
        dst_place,
        dst_ptr,
        src_place,
        src_ptr,
        size,
        reinterpret_cast<const phi::GPUContext&>(*ctx).stream());
  }
#endif
#ifdef PADDLE_WITH_XPU
  PADDLE_THROW(platform::errors::Unimplemented("XPUPlace is not supported"));
#endif
}

void TensorFromDLPack(const DLManagedTensor* src, phi::DenseTensor* dst) {
  std::vector<int64_t> vec;
  std::copy(src->dl_tensor.shape,
            src->dl_tensor.shape + src->dl_tensor.ndim,
            std::back_inserter(vec));

  phi::DDim vddim = phi::make_ddim(vec);
  dst->Resize(vddim);
  ::DLDataType type = src->dl_tensor.dtype;

  auto src_ptr = static_cast<const void*>(src->dl_tensor.data);
  auto size = phi::product(vddim) * type.bits / 8;

  if (src->dl_tensor.device.device_type == kDLCPU) {
    phi::CPUPlace dst_place = phi::CPUPlace();
    phi::CPUPlace src_place = phi::CPUPlace();
    void* dst_ptr = GetDstPtrByDLDataType(type, dst, dst_place);
    paddle::memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size);
  }
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  if (src->dl_tensor.device.device_type == kDLGPU) {
    phi::CUDAPlace dst_place = phi::CUDAPlace(src->dl_tensor.device.device_id);
    phi::CUDAPlace src_place = phi::CUDAPlace(src->dl_tensor.device.device_id);
    void* dst_ptr = GetDstPtrByDLDataType(type, dst, dst_place);
    auto* ctx = phi::DeviceContextPool::Instance().GetByPlace(dst_place);
    // Fix copy by share allocation.
    paddle::memory::Copy(
        dst_place,
        dst_ptr,
        src_place,
        src_ptr,
        size,
        reinterpret_cast<const phi::GPUContext&>(*ctx).stream());
  }
#endif
  src->deleter(const_cast<DLManagedTensor*>(src));
#ifdef PADDLE_WITH_XPU
  PADDLE_THROW(phi::errors::Unimplemented("XPUPlace is not supported"));
#endif
}

template <typename T>
T GetValue(const phi::DenseTensor* x) {
  T value = static_cast<T>(0);
  if (!paddle::platform::is_cpu_place(x->place())) {
    phi::DenseTensor cpu_x{};
    phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
    phi::DeviceContext* dev_ctx = pool.Get(x->place());
    phi::Copy(*dev_ctx, *x, phi::CPUPlace(), true, &cpu_x);
    value = cpu_x.data<T>()[0];
  } else {
    value = x->data<T>()[0];
  }
  return value;
}

template bool GetValue(const phi::DenseTensor* x);

template int16_t GetValue(const phi::DenseTensor* x);

template int GetValue(const phi::DenseTensor* x);

template int64_t GetValue(const phi::DenseTensor* x);

template float GetValue(const phi::DenseTensor* x);

template double GetValue(const phi::DenseTensor* x);

template phi::dtype::bfloat16 GetValue(const phi::DenseTensor* x);

template phi::dtype::float16 GetValue(const phi::DenseTensor* x);

template phi::dtype::complex<float> GetValue(const phi::DenseTensor* x);

template phi::dtype::complex<double> GetValue(const phi::DenseTensor* x);

template <typename T>
std::string format_tensor(const phi::DenseTensor& tensor) {
  // TODO(zhiqiu): use the print option to format tensor.
  return "NOT IMPLEMENTED";
}

template <typename T>
std::ostream& print_tensor(std::ostream& os, const phi::DenseTensor& tensor) {
  auto inspect = tensor.data<T>();
  auto element_num = tensor.numel();

  os << "  - data: [";
  // Note: int8_t && uint8_t is typedf of char, ostream unable to print properly
  if (typeid(int8_t) == typeid(T) || typeid(uint8_t) == typeid(T)) {
    if (element_num > 0) {
      os << signed(inspect[0]);
      for (int j = 1; j < element_num; ++j) {
        os << " " << signed(inspect[j]);
      }
    }
  } else {
    if (element_num > 0) {
      os << inspect[0];
      for (int j = 1; j < element_num; ++j) {
        os << " " << inspect[j];
      }
    }
  }
  os << "]";
  return os;
}

template <>
std::ostream& print_tensor<phi::dtype::complex<float>>(
    std::ostream& os, const phi::DenseTensor& tensor) {
  auto inspect = tensor.data<phi::dtype::complex<float>>();
  auto element_num = tensor.numel();

  os << "  - data: [";
  if (element_num > 0) {
    os << signed(inspect[0].real) << "+" << signed(inspect[0].imag) << "j";
    for (int j = 1; j < element_num; ++j) {
      os << " " << signed(inspect[j].real) << "+" << signed(inspect[j].imag)
         << "j";
    }
  }
  os << "]";
  return os;
}

template <>
std::ostream& print_tensor<phi::dtype::complex<double>>(
    std::ostream& os, const phi::DenseTensor& tensor) {
  auto inspect = tensor.data<phi::dtype::complex<double>>();
  auto element_num = tensor.numel();

  os << "  - data: [";
  if (element_num > 0) {
    os << signed(inspect[0].real) << "+" << signed(inspect[0].imag) << "j";
    for (int j = 1; j < element_num; ++j) {
      os << " " << signed(inspect[j].real) << "+" << signed(inspect[j].imag)
         << "j";
    }
  }
  os << "]";
  return os;
}

std::ostream& operator<<(std::ostream& os, const LoD& lod) {
  // NOTE(xiongkun):
  // https://stackoverflow.com/questions/5195512/namespaces-and-operator-resolution
  // if we don't redefine, the operator << of phi / framework LoD is not found.
  paddle::string::operator<<(os, lod);
  return os;
}

std::ostream& operator<<(std::ostream& os, const phi::DenseTensor& t) {
  if (t.lod().size() > 0) {
    os << "  - lod: " << t.lod() << "\n";
  }

  os << "  - place: " << t.place() << "\n";
  os << "  - shape: [" << t.dims() << "]\n";
  os << "  - layout: " << phi::DataLayoutToString(t.layout()) << "\n";

  DenseTensor tensor;
  tensor.Resize(t.dims());
  if (paddle::platform::is_cpu_place(t.place())) {
    tensor.ShareDataWith(t);
  } else {
    phi::CPUPlace place;
    phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
    auto& dev_ctx = *pool.Get(t.place());
    phi::Copy(dev_ctx, t, t.place(), false, &tensor);
    dev_ctx.Wait();
  }

#define PrintTensorCallback(cpp_type, data_type) \
  do {                                           \
    if (tensor.dtype() == data_type) {           \
      os << "  - dtype: " << data_type << "\n";  \
      phi::print_tensor<cpp_type>(os, tensor);   \
      return os;                                 \
    }                                            \
  } while (0);

  using namespace paddle::experimental;
  PrintTensorCallback(bool, DataType::BOOL);
  PrintTensorCallback(int8_t, DataType::INT8);
  PrintTensorCallback(uint8_t, DataType::UINT8);
  PrintTensorCallback(int16_t, DataType::INT16);
  PrintTensorCallback(uint16_t, DataType::UINT16);
  PrintTensorCallback(int32_t, DataType::INT32);
  PrintTensorCallback(uint32_t, DataType::UINT32);
  PrintTensorCallback(int64_t, DataType::INT64);
  PrintTensorCallback(uint64_t, DataType::UINT64);
  PrintTensorCallback(bfloat16, DataType::BFLOAT16);
  PrintTensorCallback(float16, DataType::FLOAT16);
  PrintTensorCallback(float, DataType::FLOAT32);
  PrintTensorCallback(double, DataType::FLOAT64);
  PrintTensorCallback(complex64, DataType::COMPLEX64);
  PrintTensorCallback(complex128, DataType::COMPLEX128);

  // pstring no support for print_tensor

  // PrintTensorCallback(pstring, DataType::PSTRING);
  // PD_FOR_EACH_DATA_TYPE(PrintTensorCallback);
  VLOG(1) << "PrintVar: unrecognized data type:" << t.type();
  return os;
}

}  // namespace phi
