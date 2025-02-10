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

#include "glog/logging.h"

#include "paddle/phi/api/lib/data_transform.h"
#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/compat/convert_utils.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename Context>
void Copy(const Context& dev_ctx,
          const DenseTensor& src,
          Place dst_place,
          bool blocking,
          DenseTensor* dst) {
  if (!src.meta().is_contiguous()) {
    DenseTensor src_copy = paddle::experimental::Trans2Contiguous(src);
    Copy(dev_ctx, src_copy, dst_place, blocking, dst);
    return;
  }

  auto* src_ptr = src.data();
  const auto& src_place = src.place();

  if (&src == dst) {
    if (src_place.GetType() == dst_place.GetType()) {
      VLOG(6) << "Skip copy the same data(" << src_ptr << ") from " << src_place
              << " to " << dst_place;
    } else {
      VLOG(6) << "Src and dst are the same Tensor, in-place copy data("
              << src_ptr << ") from " << src_place << " to " << dst_place;
      DenseTensor src_copy = src;
      Copy(dev_ctx, src_copy, dst_place, blocking, dst);
    }
    return;
  }

  VLOG(3) << "TensorCopy " << src.dims() << " from " << src.place() << " to "
          << dst_place;

  dst->Resize(src.dims());

  void* dst_ptr = nullptr;
  if (dst_place.GetType() == AllocationType::CPU) {
    dst_ptr = dev_ctx.HostAlloc(dst, src.dtype());
#ifdef PADDLE_WITH_DNNL
    dst->set_layout(src.layout());
#endif
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  } else if (dst_place.GetType() == AllocationType::GPU ||
             dst_place.GetType() == AllocationType::GPUPINNED) {
    dst_ptr = dev_ctx.Alloc(
        dst, src.dtype(), 0, dst_place.GetType() == AllocationType::GPUPINNED);
#endif
#ifdef PADDLE_WITH_XPU
  } else if (dst_place.GetType() == AllocationType::XPU) {
    dst_ptr = dev_ctx.Alloc(dst, src.dtype());
#endif
#ifdef PADDLE_WITH_CUSTOM_DEVICE
  } else if (dst_place.GetType() == AllocationType::CUSTOM) {
    dst_ptr = dev_ctx.Alloc(dst, src.dtype());
#endif
  }

  auto size = src.numel() * phi::SizeOf(src.dtype());
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
  PADDLE_ENFORCE_EQ(dst->layout(),
                    src.layout(),
                    common::errors::PreconditionNotMet(
                        "dst's layout differs from src's layout"));

  if (src_place.GetType() == AllocationType::CPU &&
      dst_place.GetType() == AllocationType::CPU) {
    memory_utils::Copy(src_place, dst_ptr, src_place, src_ptr, size);
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  } else if ((src_place.GetType() == AllocationType::CPU ||
              src_place.GetType() == AllocationType::GPUPINNED) &&  // NOLINT
             (dst_place.GetType() == AllocationType::CPU ||
              dst_place.GetType() == AllocationType::GPUPINNED)) {
    memory_utils::Copy(dst_place, dst_ptr, src_place, src_ptr, size, nullptr);
  } else if (src_place.GetType() == AllocationType::GPU &&  // NOLINT
             dst_place.GetType() == AllocationType::CPU) {
    auto src_gpu_place = src_place;
    auto dst_cpu_place = dst_place;
    auto ctx_place = dev_ctx.GetPlace();
    PADDLE_ENFORCE_EQ(
        ctx_place.GetType() == AllocationType::GPU,
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
    memory_utils::Copy(
        dst_cpu_place, dst_ptr, src_gpu_place, src_ptr, size, stream);
  } else if ((src_place.GetType() == AllocationType::CPU ||
              src_place.GetType() == AllocationType::GPUPINNED) &&  // NOLINT
             dst_place.GetType() == AllocationType::GPU) {
    auto src_cpu_place = src_place;
    auto dst_gpu_place = dst_place;
    auto ctx_place = dev_ctx.GetPlace();
    PADDLE_ENFORCE_EQ(
        ctx_place.GetType() == AllocationType::GPU,
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
    memory_utils::Copy(
        dst_gpu_place, dst_ptr, src_cpu_place, src_ptr, size, stream);
  } else if (src_place.GetType() == AllocationType::GPU &&  // NOLINT
             dst_place.GetType() == AllocationType::GPU) {
    auto src_gpu_place = src_place;
    auto dst_gpu_place = dst_place;
    auto ctx_place = dev_ctx.GetPlace();
    PADDLE_ENFORCE_EQ(
        ctx_place.GetType() == AllocationType::GPU,
        true,
        errors::PreconditionNotMet(
            "Context place error, excepted GPUPlace, but actually %s.",
            ctx_place));
    auto stream =
        blocking ? nullptr
                 : reinterpret_cast<const phi::GPUContext&>(dev_ctx).stream();
    if (src_place.GetType() == dst_place.GetType()) {
      memory_utils::Copy(
          dst_gpu_place, dst_ptr, src_gpu_place, src_ptr, size, stream);
    } else {
      if (ctx_place.GetType() == src_place.GetType()) {
        memory_utils::Copy(
            dst_gpu_place, dst_ptr, src_gpu_place, src_ptr, size, stream);
        phi::DeviceContextPool::Instance().Get(src.place())->Wait();
      } else if (ctx_place.GetType() == dst_place.GetType()) {
        phi::DeviceContextPool::Instance().Get(src.place())->Wait();
        memory_utils::Copy(
            dst_gpu_place, dst_ptr, src_gpu_place, src_ptr, size, stream);
      } else {
        PADDLE_THROW(errors::Unavailable(
            "Context place dose not match the source and destination place."));
      }
    }
  } else if (src_place.GetType() == AllocationType::GPU &&  // NOLINT
             dst_place.GetType() == AllocationType::GPUPINNED) {
    auto src_gpu_place = src_place;
    auto dst_cuda_pinned_place = dst_place;
    auto ctx_place = dev_ctx.GetPlace();
    PADDLE_ENFORCE_EQ(
        ctx_place.GetType() == AllocationType::GPU,
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
    memory_utils::Copy(
        dst_cuda_pinned_place, dst_ptr, src_gpu_place, src_ptr, size, stream);
#endif
#ifdef PADDLE_WITH_XPU
  } else if (src_place.GetType() == AllocationType::XPU &&  // NOLINT
             dst_place.GetType() == AllocationType::CPU) {
    memory_utils::Copy(dst_place, dst_ptr, src_place, src_ptr, size);
  } else if (src_place.GetType() == AllocationType::CPU &&
             dst_place.GetType() == AllocationType::XPU) {
    memory_utils::Copy(dst_place, dst_ptr, src_place, src_ptr, size);
  } else if (src_place.GetType() == AllocationType::XPU &&
             dst_place.GetType() == AllocationType::XPU) {
    if (src_ptr == dst_ptr) {
      VLOG(3) << "Skip copy the same data async from " << src_place << " to "
              << dst_place;
      return;
    }
    memory_utils::Copy(dst_place, dst_ptr, src_place, src_ptr, size);
#endif
#ifdef PADDLE_WITH_CUSTOM_DEVICE
  } else if (src_place.GetType() == AllocationType::CUSTOM &&  // NOLINT
             dst_place.GetType() == AllocationType::CPU) {
    auto stream =
        blocking
            ? nullptr
            : reinterpret_cast<const phi::CustomContext&>(dev_ctx).stream();
    memory_utils::Copy(dst_place, dst_ptr, src_place, src_ptr, size, stream);
  } else if (src_place.GetType() == AllocationType::CPU &&  // NOLINT
             dst_place.GetType() == AllocationType::CUSTOM) {
    auto stream =
        blocking
            ? nullptr
            : reinterpret_cast<const phi::CustomContext&>(dev_ctx).stream();
    memory_utils::Copy(dst_place, dst_ptr, src_place, src_ptr, size, stream);
  } else if (src_place.GetType() == AllocationType::CUSTOM &&  // NOLINT
             dst_place.GetType() == AllocationType::CUSTOM) {
    auto stream =
        blocking
            ? nullptr
            : reinterpret_cast<const phi::CustomContext&>(dev_ctx).stream();
    memory_utils::Copy(dst_place, dst_ptr, src_place, src_ptr, size, stream);
#endif
  } else {
    PADDLE_THROW(errors::Unimplemented(
        "Copy from %s to %s is not supported.", src_place, dst_place));
  }
  dst->set_strides(src.strides());
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

template <typename Context>
void Copy(const Context& dev_ctx UNUSED,
          const TensorArray& src UNUSED,
          Place dst_place UNUSED,
          bool blocking UNUSED,
          TensorArray* dst UNUSED) {
  // NOTE(Ruibiao): implements Copy() for TensorArray when needed.
  PADDLE_THROW(errors::Unimplemented("Copy for TensorArray is unimplemented."));
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

template void Copy(const CPUContext& dev_ctx,
                   const TensorArray& src,
                   Place dst_place,
                   bool blocking,
                   TensorArray* dst);

template void Copy(const DeviceContext& dev_ctx,
                   const TensorArray& src,
                   Place dst_place,
                   bool blocking,
                   TensorArray* dst);

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
template void Copy(const GPUContext& dev_ctx,
                   const TensorArray& src,
                   Place dst_place,
                   bool blocking,
                   TensorArray* dst);
#endif

#ifdef PADDLE_WITH_XPU
template void Copy(const XPUContext& dev_ctx,
                   const DenseTensor& src,
                   Place dst_place,
                   bool blocking,
                   DenseTensor* dst);
template void Copy(const XPUContext& dev_ctx,
                   const TensorArray& src,
                   Place dst_place,
                   bool blocking,
                   TensorArray* dst);
#endif

#ifdef PADDLE_WITH_CUSTOM_DEVICE
template void Copy(const CustomContext& dev_ctx,
                   const DenseTensor& src,
                   Place dst_place,
                   bool blocking,
                   DenseTensor* dst);
template void Copy(const CustomContext& dev_ctx,
                   const TensorArray& src,
                   Place dst_place,
                   bool blocking,
                   TensorArray* dst);
#endif

#ifdef PADDLE_WITH_DNNL
template void Copy(const OneDNNContext& dev_ctx,
                   const DenseTensor& src,
                   Place dst_place,
                   bool blocking,
                   DenseTensor* dst);
template void Copy(const OneDNNContext& dev_ctx,
                   const TensorArray& src,
                   Place dst_place,
                   bool blocking,
                   TensorArray* dst);
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

  if (dst_place.GetType() == AllocationType::CPU) {
    memory_utils::Copy(dst_place, dst_ptr, src_place, src_ptr, size);
  }
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  else if (dst_place.GetType() == AllocationType::GPU) {  // NOLINT
    memory_utils::Copy(dst_place,
                       dst_ptr,
                       src_place,
                       src_ptr,
                       size,
                       reinterpret_cast<const phi::GPUContext&>(ctx).stream());
  }
#endif
#ifdef PADDLE_WITH_CUSTOM_DEVICE
  else if (dst_place.GetType() == AllocationType::CUSTOM) {  // NOLINT
    memory_utils::Copy(
        dst_place,
        dst_ptr,
        src_place,
        src_ptr,
        size,
        reinterpret_cast<const phi::CustomContext&>(ctx).stream());
  }
#endif
#ifdef PADDLE_WITH_XPU
  else if (dst_place.GetType() == AllocationType::XPU) {  // NOLINT
    memory_utils::Copy(dst_place, dst_ptr, src_place, src_ptr, size);
  }
#endif
  else {  // NOLINT
    PADDLE_THROW(common::errors::Unimplemented(
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

  if (dst_place.GetType() == AllocationType::CPU) {
    memory_utils::Copy(dst_place, dst_ptr, src_place, src_ptr, size);
  }
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  else if (dst_place.GetType() == AllocationType::GPU) {  // NOLINT
    memory_utils::Copy(dst_place,
                       dst_ptr,
                       src_place,
                       src_ptr,
                       size,
                       reinterpret_cast<const phi::GPUContext&>(ctx).stream());
  }
#endif
#ifdef PADDLE_WITH_CUSTOM_DEVICE
  else if (dst_place.GetType() == AllocationType::CUSTOM) {  // NOLINT
    auto stream = reinterpret_cast<const phi::CustomContext&>(ctx).stream();
    memory_utils::Copy(dst_place, dst_ptr, src_place, src_ptr, size, stream);
  }
#endif
#ifdef PADDLE_WITH_XPU
  else if (dst_place.GetType() == AllocationType::XPU) {  // NOLINT
    memory_utils::Copy(dst_place, dst_ptr, src_place, src_ptr, size);
  }
#endif
  else {  // NOLINT
    PADDLE_THROW(common::errors::Unimplemented(
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

  if (dst_place.GetType() == AllocationType::CPU) {
    memory_utils::Copy(dst_place, dst_ptr, src_place, src_ptr, size);
  }
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  else if (dst_place.GetType() == AllocationType::GPU) {  // NOLINT
    memory_utils::Copy(dst_place,
                       dst_ptr,
                       src_place,
                       src_ptr,
                       size,
                       reinterpret_cast<const phi::GPUContext&>(ctx).stream());
  }
#endif
#ifdef PADDLE_WITH_CUSTOM_DEVICE
  else if (dst_place.GetType() == AllocationType::CUSTOM) {  // NOLINT
    memory_utils::Copy(
        dst_place,
        dst_ptr,
        src_place,
        src_ptr,
        size,
        reinterpret_cast<const phi::CustomContext&>(ctx).stream());
  }
#endif
#ifdef PADDLE_WITH_XPU
  else if (dst_place.GetType() == AllocationType::XPU) {  // NOLINT
    memory_utils::Copy(dst_place, dst_ptr, src_place, src_ptr, size);
  }
#endif
  else {  // NOLINT
    PADDLE_THROW(common::errors::Unimplemented(
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

  if (src.place().GetType() == AllocationType::CPU) {
    memory_utils::Copy(dst_place, dst_ptr, src.place(), src_ptr, size);
  }
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  else if (src.place().GetType() == AllocationType::GPU) {  // NOLINT
    memory_utils::Copy(dst_place,
                       dst_ptr,
                       src.place(),
                       src_ptr,
                       size,
                       reinterpret_cast<const phi::GPUContext&>(ctx).stream());
  }
#endif
#if defined(PADDLE_WITH_XPU)
  else if (src.place().GetType() == AllocationType::XPU) {  // NOLINT
    memory_utils::Copy(dst_place, dst_ptr, src.place(), src_ptr, size);
  }
#endif
#ifdef PADDLE_WITH_CUSTOM_DEVICE
  else if (src.place().GetType() == AllocationType::CUSTOM) {  // NOLINT
    memory_utils::Copy(dst_place, dst_ptr, src.place(), src_ptr, size, nullptr);
  }
#endif
  else {  // NOLINT
    PADDLE_THROW(common::errors::Unimplemented(
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

  if (src.place().GetType() == AllocationType::CPU) {
    memory_utils::Copy(dst_place, dst_ptr, src.place(), src_ptr, size);
  }
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  else if (src.place().GetType() == AllocationType::GPU) {  // NOLINT
    memory_utils::Copy(dst_place,
                       dst_ptr,
                       src.place(),
                       src_ptr,
                       size,
                       reinterpret_cast<const phi::GPUContext&>(ctx).stream());
  }
#endif
#if defined(PADDLE_WITH_XPU)
  else if (src.place().GetType() == AllocationType::XPU) {  // NOLINT
    memory_utils::Copy(dst_place, dst_ptr, src.place(), src_ptr, size);
  }
#endif
#ifdef PADDLE_WITH_CUSTOM_DEVICE
  else if (src.place().GetType() == AllocationType::CUSTOM) {  // NOLINT
    memory_utils::Copy(dst_place, dst_ptr, src.place(), src_ptr, size, nullptr);
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
      src.place().GetType() == AllocationType::CPU,
      true,
      common::errors::InvalidArgument(
          "The input tensor should be CPU device, but actually it is in %s.",
          src.place()));

  memory_utils::Copy(dst_place, dst_ptr, src.place(), src_ptr, size);
}

template <>
void TensorToVector(const phi::DenseTensor& src, std::vector<bool>* dst) {
  auto src_ptr = static_cast<const void*>(src.data<bool>());
  auto size = src.numel() * sizeof(bool);

  bool* array = new bool[src.numel()];

  phi::CPUPlace dst_place{};
  dst->resize(src.numel());
  auto dst_ptr = static_cast<void*>(array);

  PADDLE_ENFORCE_EQ(
      src.place().GetType() == AllocationType::CPU,
      true,
      common::errors::InvalidArgument(
          "The input tensor should be CPU device, but actually it is in %s.",
          src.place()));

  memory_utils::Copy(dst_place, dst_ptr, src.place(), src_ptr, size);

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
      common::errors::InvalidArgument(
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
  res.Resize(common::flatten_to_2d(src.dims(), num_col_dims));
  return res;
}

template <typename T>
T GetValue(const phi::DenseTensor* x) {
  T value = static_cast<T>(0);
  if (x->place().GetType() != AllocationType::CPU) {
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
std::vector<T> GetVectorFromTensor(const phi::DenseTensor* x) {
  std::vector<T> vec_new_data;
  if (phi::TransToProtoVarType(x->dtype()) == ProtoDataType::INT32) {
    auto* data = x->data<int>();
    phi::DenseTensor cpu_attr_tensor;
    if (x->place().GetType() != phi::AllocationType::CPU) {
      phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
      auto dev_ctx = pool.Get(x->place());
      phi::Copy(*dev_ctx, *x, CPUPlace(), true, &cpu_attr_tensor);
      data = cpu_attr_tensor.data<int>();
    }
    vec_new_data = std::vector<T>(data, data + x->numel());
  } else if (phi::TransToProtoVarType(x->dtype()) == ProtoDataType::INT64) {
    auto* data = x->data<int64_t>();
    phi::DenseTensor cpu_attr_tensor;
    if (x->place().GetType() != phi::AllocationType::CPU) {
      phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
      auto dev_ctx = pool.Get(x->place());
      phi::Copy(*dev_ctx, *x, CPUPlace(), true, &cpu_attr_tensor);
      data = cpu_attr_tensor.data<int64_t>();
    }
    // NOTE: Converting int64 to int32 may cause data overflow.
    vec_new_data = std::vector<T>(data, data + x->numel());
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "The dtype of Tensor must be int32 or int64, but received: %s",
        phi::TransToProtoVarType(x->dtype())));
  }
  return vec_new_data;
}

template std::vector<int32_t> GetVectorFromTensor(const phi::DenseTensor* x);

template std::vector<int64_t> GetVectorFromTensor(const phi::DenseTensor* x);

namespace {

template <typename T>
std::vector<T> _GetVectorFromTensor(const phi::DenseTensor* x) {
  auto* data = x->data<T>();
  phi::DenseTensor cpu_attr_tensor;
  if (x->place().GetType() != phi::AllocationType::CPU) {
    phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
    auto dev_ctx = pool.Get(x->place());
    phi::Copy(*dev_ctx, *x, CPUPlace(), true, &cpu_attr_tensor);
    data = cpu_attr_tensor.data<T>();
  }
  return std::vector<T>(data, data + x->numel());
}

}  // namespace

template <>
std::vector<float> GetVectorFromTensor<float>(const phi::DenseTensor* x) {
  if (phi::TransToProtoVarType(x->dtype()) != ProtoDataType::FP32) {
    PADDLE_THROW(common::errors::InvalidArgument(
        "The dtype of Tensor must be float32, but received: %s",
        phi::TransToProtoVarType(x->dtype())));
  }
  return _GetVectorFromTensor<float>(x);
}

template <>
std::vector<double> GetVectorFromTensor<double>(const phi::DenseTensor* x) {
  if (phi::TransToProtoVarType(x->dtype()) != ProtoDataType::FP64) {
    PADDLE_THROW(common::errors::InvalidArgument(
        "The dtype of Tensor must be float64, but received: %s",
        phi::TransToProtoVarType(x->dtype())));
  }
  return _GetVectorFromTensor<double>(x);
}

}  // namespace phi
