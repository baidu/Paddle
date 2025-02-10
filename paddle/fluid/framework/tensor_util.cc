/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/tensor_util.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/data_type.h"
#include "paddle/phi/api/lib/data_transform.h"
#include "paddle/phi/common/complex.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/platform/profiler/event_tracing.h"

#ifdef PADDLE_WITH_DNNL
#include "dnnl_debug.h"  // NOLINT
#endif

namespace paddle::framework {

template <typename TENSOR>
void TensorCopyImpl(const TENSOR& src,
                    const phi::Place& dst_place,
                    const phi::DeviceContext& ctx,
                    TENSOR* dst) {
  if (&src == dst) {
    auto src_copy = src;
    TensorCopyImpl(src_copy, dst_place, ctx, dst);
    return;
  }
  VLOG(3) << "TensorCopy " << src.dims() << " from " << src.place() << " to "
          << dst_place;
  src.check_memory_size();
  dst->Resize(src.dims());
  dst->set_layout(src.layout());
  auto src_place = src.place();
  auto src_ptr = src.data();
#ifdef PADDLE_WITH_DNNL
  dst->set_mem_desc(src.mem_desc());
  // oneDNN tensors due to padding may be of bigger size
  // than numel()*size(type())
  auto dst_ptr =
      src.layout() == DataLayout::ONEDNN
          ? dst->mutable_data(dst_place, src.dtype(), src.memory_size())
          : dst->mutable_data(dst_place, src.dtype());
#else
  auto dst_ptr = dst->mutable_data(dst_place, src.dtype());
#endif
  dst->set_layout(src.layout());
  if (src_ptr == dst_ptr && src_place == dst_place) {
    VLOG(3) << "Skip copy the same data async from " << src_place << " to "
            << dst_place;
    return;
  }
  VLOG(4) << "src:" << src_ptr << ", dst:" << dst_ptr;

#ifdef PADDLE_WITH_DNNL
  auto size = src.layout() == DataLayout::ONEDNN
                  ? src.memory_size()
                  : src.numel() * phi::SizeOf(src.dtype());
#else
  auto size = src.numel() * phi::SizeOf(src.dtype());
#endif

  if (phi::is_cpu_place(src_place) && phi::is_cpu_place(dst_place)) {  // NOLINT
    memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size);
  }
#ifdef PADDLE_WITH_CUSTOM_DEVICE
  else if (phi::is_custom_place(src_place) &&  // NOLINT
           phi::is_cpu_place(dst_place)) {
    auto stream = reinterpret_cast<const phi::CustomContext&>(ctx).stream();
    memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size, stream);
  } else if (phi::is_cpu_place(src_place) &&  // NOLINT
             phi::is_custom_place(dst_place)) {
    auto stream = reinterpret_cast<const phi::CustomContext&>(ctx).stream();
    memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size, stream);
  } else if (phi::is_custom_place(src_place) &&  // NOLINT
             phi::is_custom_place(dst_place)) {
    if (src_ptr == dst_ptr) {
      VLOG(3) << "Skip copy the same data async from " << src_place << " to "
              << dst_place;
      return;
    }
    auto stream = reinterpret_cast<const phi::CustomContext&>(ctx).stream();
    memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size, stream);
  }
#endif
#ifdef PADDLE_WITH_XPU
  else if (phi::is_xpu_place(src_place) &&  // NOLINT
           phi::is_cpu_place(dst_place)) {
    memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size);
  } else if (phi::is_cpu_place(src_place) && phi::is_xpu_place(dst_place)) {
    memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size);
  } else if (phi::is_xpu_place(src_place) && phi::is_xpu_place(dst_place)) {
    if (src_ptr == dst_ptr) {
      VLOG(3) << "Skip copy the same data async from " << src_place << " to "
              << dst_place;
      return;
    }
    memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size);
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "Copy from %s to %s is not supported.", src_place, dst_place));
  }
#endif
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  else if (phi::is_cuda_pinned_place(src_place) &&  // NOLINT
           phi::is_cuda_pinned_place(dst_place)) {
    memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size);
  }
  else if (phi::is_cuda_pinned_place(src_place) &&  // NOLINT
           phi::is_cpu_place(dst_place)) {
    memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size);
  }
  else if (phi::is_cpu_place(src_place) &&  // NOLINT
           phi::is_cuda_pinned_place(dst_place)) {
    memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size);
  }
  else if (phi::is_gpu_place(src_place) &&  // NOLINT
           phi::is_cpu_place(dst_place)) {
    auto src_gpu_place = src_place;
    auto dst_cpu_place = dst_place;
    auto ctx_place = ctx.GetPlace();
    PADDLE_ENFORCE_EQ(
        phi::is_gpu_place(ctx_place),
        true,
        common::errors::PreconditionNotMet(
            "Context place error, excepted GPUPlace, but actually %s.",
            ctx_place));
    auto ctx_gpu_place = ctx_place;
    PADDLE_ENFORCE_EQ(src_gpu_place,
                      ctx_gpu_place,
                      common::errors::Unavailable(
                          "Source place and context place do not match, source "
                          "place is %s, context place is %s.",
                          src_gpu_place,
                          ctx_gpu_place));
    auto stream = reinterpret_cast<const phi::GPUContext&>(ctx).stream();
    memory::Copy(dst_cpu_place, dst_ptr, src_gpu_place, src_ptr, size, stream);
  }
  else if (phi::is_cpu_place(src_place) &&  // NOLINT
           phi::is_gpu_place(dst_place)) {
    auto src_cpu_place = src_place;
    auto dst_gpu_place = dst_place;
    auto ctx_place = ctx.GetPlace();
    PADDLE_ENFORCE_EQ(
        phi::is_gpu_place(ctx_place),
        true,
        common::errors::PreconditionNotMet(
            "Context place error, excepted GPUPlace, but actually %s.",
            ctx_place));
    auto ctx_gpu_place = ctx_place;
    PADDLE_ENFORCE_EQ(dst_gpu_place,
                      ctx_gpu_place,
                      common::errors::Unavailable(
                          "Destination place and context place do not match, "
                          "destination place is %s, context place is %s.",
                          dst_gpu_place,
                          ctx_gpu_place));
    auto stream = reinterpret_cast<const phi::GPUContext&>(ctx).stream();
    memory::Copy(dst_gpu_place, dst_ptr, src_cpu_place, src_ptr, size, stream);
  }
  else if (phi::is_gpu_place(src_place) &&  // NOLINT
           phi::is_cuda_pinned_place(dst_place)) {
    auto src_gpu_place = src_place;
    auto dst_cuda_pinned_place = dst_place;
    auto ctx_place = ctx.GetPlace();
    PADDLE_ENFORCE_EQ(
        phi::is_gpu_place(ctx_place),
        true,
        common::errors::PreconditionNotMet(
            "Device context place mismatch. When copying phi::DenseTensor "
            "data from GPU memory to CUDA Pinned memory, current "
            "device context place should be GPU."));
    auto ctx_gpu_place = ctx_place;
    PADDLE_ENFORCE_EQ(src_gpu_place,
                      ctx_gpu_place,
                      common::errors::PreconditionNotMet(
                          "The source GPU device and current device context do "
                          "not match. The source GPU device number is %d, but "
                          "device context GPU number is %d.",
                          src_gpu_place.device,
                          ctx_gpu_place.device));
    auto stream = reinterpret_cast<const phi::GPUContext&>(ctx).stream();
    memory::Copy(
        dst_cuda_pinned_place, dst_ptr, src_gpu_place, src_ptr, size, stream);
  }
  else if (phi::is_cuda_pinned_place(src_place) &&  // NOLINT
           phi::is_gpu_place(dst_place)) {
    auto src_cuda_pinned_place = src_place;
    auto dst_gpu_place = dst_place;
    auto ctx_place = ctx.GetPlace();
    PADDLE_ENFORCE_EQ(
        phi::is_gpu_place(ctx_place),
        true,
        common::errors::PreconditionNotMet(
            "Device context place mismatch. When copying phi::DenseTensor "
            "data from CUDA Pinned memory to GPU memory, current "
            "device context place should be GPU."));
    auto ctx_gpu_place = ctx_place;
    PADDLE_ENFORCE_EQ(dst_gpu_place,
                      ctx_gpu_place,
                      common::errors::PreconditionNotMet(
                          "The target GPU device and current device context do "
                          "not match. The target GPU device number is %d, but "
                          "device context GPU number is %d.",
                          dst_gpu_place.device,
                          ctx_gpu_place.device));
    auto stream = reinterpret_cast<const phi::GPUContext&>(ctx).stream();
    memory::Copy(
        dst_gpu_place, dst_ptr, src_cuda_pinned_place, src_ptr, size, stream);
  }
  else if (phi::is_gpu_place(src_place) &&  // NOLINT
           phi::is_gpu_place(dst_place)) {
    auto src_gpu_place = src_place;
    auto dst_gpu_place = dst_place;
    auto ctx_place = ctx.GetPlace();
    PADDLE_ENFORCE_EQ(
        phi::is_gpu_place(ctx_place),
        true,
        common::errors::PreconditionNotMet(
            "Context place error, excepted GPUPlace, but actually %s.",
            ctx_place));
    auto stream = reinterpret_cast<const phi::GPUContext&>(ctx).stream();
    if (phi::is_same_place(src_place, dst_place)) {
      memory::Copy(
          dst_gpu_place, dst_ptr, src_gpu_place, src_ptr, size, stream);
    } else {
      if (phi::is_same_place(ctx_place, src_place)) {
        memory::Copy(
            dst_gpu_place, dst_ptr, src_gpu_place, src_ptr, size, stream);
        phi::DeviceContextPool::Instance().Get(src.place())->Wait();
      } else if (phi::is_same_place(ctx_place, dst_place)) {
        phi::DeviceContextPool::Instance().Get(src.place())->Wait();
        memory::Copy(
            dst_gpu_place, dst_ptr, src_gpu_place, src_ptr, size, stream);
      } else {
        PADDLE_THROW(common::errors::Unavailable(
            "Context place dose not match the source and destination place."));
      }
    }
  }
  else {  // NOLINT
    PADDLE_THROW(common::errors::Unimplemented(
        "Copying from %s to %s is not supported.", src_place, dst_place));
  }
#endif
}

template <typename TENSOR>
void TensorCopyImpl(const TENSOR& src,
                    const phi::Place& dst_place,
                    TENSOR* dst) {
  phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
  const phi::DeviceContext* dev_ctx = nullptr;
  if (phi::is_gpu_place(dst_place) || phi::is_custom_place(dst_place)) {
    dev_ctx = pool.Get(dst_place);
  } else {
    dev_ctx = pool.Get(src.place());
  }
  TensorCopyImpl(src, dst_place, *dev_ctx, dst);
}

void TensorCopy(const phi::DenseTensor& src,
                const phi::Place& dst_place,
                phi::DenseTensor* dst) {
  TensorCopyImpl<phi::DenseTensor>(src, dst_place, dst);
  dst->set_strides(src.strides());
}
void TensorCopy(const phi::DenseTensor& src,
                const phi::Place& dst_place,
                const phi::DeviceContext& ctx,
                phi::DenseTensor* dst) {
  TensorCopyImpl<phi::DenseTensor>(src, dst_place, ctx, dst);
  dst->set_strides(src.strides());
}

void TensorCopySync(const phi::DenseTensor& src,
                    const phi::Place& dst_place,
                    phi::DenseTensor* dst) {
  if (&src == dst) {
    auto src_copy = src;
    TensorCopySync(src_copy, dst_place, dst);
    return;
  }

  src.check_memory_size();
  dst->Resize(src.dims());
  dst->set_layout(src.layout());
#ifdef PADDLE_WITH_DNNL
  if (src.layout() == DataLayout::ONEDNN) {
    dst->set_mem_desc(src.mem_desc());
  }
#endif
  auto src_place = src.place();
  auto src_ptr = src.data();
  auto dst_ptr = dst->mutable_data(dst_place, src.dtype());
  VLOG(4) << "src:" << src_ptr << ", dst:" << dst_ptr;

  if (src_ptr == dst_ptr && src_place == dst_place) {
    VLOG(3) << "Skip copy the same data from " << src_place << " to "
            << dst_place;
    return;
  }
  auto size = src.numel() * phi::SizeOf(src.dtype());
  if (phi::is_cpu_place(src_place) && phi::is_cpu_place(dst_place)) {  // NOLINT
    memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size);
  }
#ifdef PADDLE_WITH_CUSTOM_DEVICE
  else if (phi::is_custom_place(src_place) &&  // NOLINT
           phi::is_cpu_place(dst_place)) {     /* custom_device -> cpu*/
    memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size, nullptr);
  }                                           // NOLINT
  else if (phi::is_cpu_place(src_place) &&    // NOLINT
           phi::is_custom_place(dst_place)) { /* cpu -> custom_device*/
    memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size, nullptr);
  }                                            // NOLINT
  else if (phi::is_custom_place(src_place) &&  // NOLINT
           phi::is_custom_place(
               dst_place)) { /* custom_device -> custom_device*/
    if (src_ptr == dst_ptr) {
      VLOG(3) << "Skip copy the same data sync from " << src_place << " to "
              << dst_place;
      return;
    }
    memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size, nullptr);
  }
#endif
#ifdef PADDLE_WITH_XPU
  else if (phi::is_xpu_place(src_place) &&  // NOLINT
           phi::is_cpu_place(dst_place)) {
    memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size);
  }                                         // NOLINT
  else if (phi::is_cpu_place(src_place) &&  // NOLINT
           phi::is_xpu_place(dst_place)) {
    memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size);
  }                                         // NOLINT
  else if (phi::is_xpu_place(src_place) &&  // NOLINT
           phi::is_xpu_place(dst_place)) {
    if (src_ptr == dst_ptr) {
      VLOG(3) << "Skip copy the same data async from " << src_place << " to "
              << dst_place;
      return;
    }
    memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size);
    phi::XPUPlace xpu_dst_place = dst_place;
    phi::XPUPlace xpu_src_place = src_place;
    if (xpu_dst_place.device == xpu_src_place.device) {
      auto xpu_ctx = phi::DeviceContextPool::Instance().Get(xpu_dst_place);
      xpu_ctx->Wait();
    }
  }       // NOLINT
  else {  // NOLINT
    PADDLE_THROW(common::errors::Unimplemented(
        "Copy from %s to %s is not supported.", src_place, dst_place));
  }
#endif
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  else if (phi::is_cuda_pinned_place(src_place) &&  // NOLINT
           phi::is_cuda_pinned_place(dst_place)) {
    memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size);
  }
  else if (phi::is_cuda_pinned_place(src_place) &&  // NOLINT
           phi::is_cpu_place(dst_place)) {
    memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size);
  }
  else if (phi::is_cpu_place(src_place) &&  // NOLINT
           phi::is_cuda_pinned_place(dst_place)) {
    memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size);
  }
  else if (phi::is_gpu_place(src_place) &&  // NOLINT
           phi::is_cuda_pinned_place(dst_place)) {
    memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size, nullptr);
  }
  else if (phi::is_gpu_place(src_place) &&  // NOLINT
           phi::is_cpu_place(dst_place)) {
    auto src_gpu_place = src_place;
    auto dst_cpu_place = dst_place;
    memory::Copy(dst_cpu_place, dst_ptr, src_gpu_place, src_ptr, size, nullptr);
  }
  else if (phi::is_cpu_place(src_place) &&  // NOLINT
           phi::is_gpu_place(dst_place)) {
    auto src_cpu_place = src_place;
    auto dst_gpu_place = dst_place;
    memory::Copy(dst_gpu_place, dst_ptr, src_cpu_place, src_ptr, size, nullptr);
  }
  else if (phi::is_gpu_place(src_place) &&  // NOLINT
           phi::is_gpu_place(dst_place)) {
    auto src_gpu_place = src_place;
    auto dst_gpu_place = dst_place;
    memory::Copy(dst_gpu_place, dst_ptr, src_gpu_place, src_ptr, size, nullptr);
  }
  else if (phi::is_cuda_pinned_place(src_place) &&  // NOLINT
           phi::is_gpu_place(dst_place)) {
    auto src_pinned_place = src_place;
    auto dst_gpu_place = dst_place;
    memory::Copy(
        dst_gpu_place, dst_ptr, src_pinned_place, src_ptr, size, nullptr);
  }
  else {  // NOLINT
    PADDLE_THROW(common::errors::Unimplemented(
        "Copy from %s to %s is not supported.", src_place, dst_place));
  }
#endif
#ifdef PADDLE_WITH_IPU
  else if (phi::is_ipu_place(src_place) &&  // NOLINT
           phi::is_cpu_place(dst_place)) {
    memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size);
  }
  else if (phi::is_cpu_place(src_place) &&  // NOLINT
           phi::is_ipu_place(dst_place)) {
    memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size);
  }
  else if (phi::is_ipu_place(src_place) &&  // NOLINT
           phi::is_ipu_place(dst_place)) {
    if (src_ptr == dst_ptr) {
      VLOG(3) << "Skip copy the same data sync from " << src_place << " to "
              << dst_place;
      return;
    }
    memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size);
  }
  else {  // NOLINT
    PADDLE_THROW(common::errors::Unimplemented(
        "Copy from %s to %s is not supported.", src_place, dst_place));
  }
#endif
  dst->set_strides(src.strides());
}

void TensorToStream(std::ostream& os,
                    const phi::DenseTensor& tensor,
                    const phi::DeviceContext& dev_ctx) {
  const auto ensure_contiguous = [](const phi::DenseTensor& tensor) {
    if (tensor.meta().is_contiguous()) {
      return tensor;
    }
    return paddle::experimental::Trans2Contiguous(tensor);
  };
  const phi::DenseTensor& contiguous_tensor = ensure_contiguous(tensor);
  {  // the 1st field, uint32_t version
    constexpr uint32_t version = 0;
    os.write(reinterpret_cast<const char*>(&version), sizeof(version));
  }
  {  // the 2nd field, tensor description
     // int32_t  size
     // void*    protobuf message
    proto::VarType::TensorDesc desc;
    desc.set_data_type(
        framework::TransToProtoVarType(contiguous_tensor.dtype()));
    auto dims = common::vectorize(contiguous_tensor.dims());
    auto* pb_dims = desc.mutable_dims();
    pb_dims->Resize(static_cast<int>(dims.size()), 0);
    std::copy(dims.begin(), dims.end(), pb_dims->begin());
    int32_t size = desc.ByteSize();
    os.write(reinterpret_cast<const char*>(&size), sizeof(size));
    auto out = desc.SerializeAsString();
    os.write(out.data(), size);
  }
  {  // the 3rd field, tensor data
    uint64_t size =
        contiguous_tensor.numel() * phi::SizeOf(contiguous_tensor.dtype());

    auto* data_ptr = contiguous_tensor.data();
    PADDLE_ENFORCE_LT(size,
                      (std::numeric_limits<std::streamsize>::max)(),
                      common::errors::ResourceExhausted(
                          "tensor size %d overflow when writing tensor", size));
    if (phi::is_gpu_place(contiguous_tensor.place())) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      constexpr size_t kBufSize = 1024 * 1024 * 64;  // 64MB
      std::unique_ptr<char[]> buf(new char[kBufSize]);
      auto& gpu_dev_ctx = static_cast<const phi::GPUContext&>(dev_ctx);
      phi::CPUPlace cpu;
      uintptr_t data = reinterpret_cast<uintptr_t>(data_ptr);
      while (size != 0) {
        size_t size_to_write = std::min(kBufSize, static_cast<size_t>(size));
        memory::Copy(cpu,
                     buf.get(),
                     contiguous_tensor.place(),
                     reinterpret_cast<const void*>(data),  // NOLINT
                     size_to_write,
                     gpu_dev_ctx.stream());
        gpu_dev_ctx.Wait();
        os.write(buf.get(), size_to_write);
        data += size_to_write;
        size -= size_to_write;
      }
#else
      PADDLE_THROW(common::errors::Unimplemented(
          "CUDAPlace is not supported when not compiled with CUDA"));
#endif
    } else if (phi::is_xpu_place(contiguous_tensor.place())) {
#ifdef PADDLE_WITH_XPU
      constexpr size_t kBufSize = 1024 * 1024 * 64;  // 64MB
      std::unique_ptr<char[]> buf(new char[kBufSize]);
      auto& xpu_dev_ctx = static_cast<const phi::XPUContext&>(dev_ctx);
      phi::CPUPlace cpu;
      uintptr_t data = reinterpret_cast<uintptr_t>(data_ptr);
      while (size != 0) {
        size_t size_to_write = std::min(kBufSize, static_cast<size_t>(size));
        memory::Copy(cpu,
                     buf.get(),
                     contiguous_tensor.place(),
                     reinterpret_cast<const void*>(data),
                     size_to_write);
        xpu_dev_ctx.Wait();
        os.write(buf.get(), size_to_write);
        data += size_to_write;
        size -= size_to_write;
      }
#else
      PADDLE_THROW(common::errors::Unimplemented(
          "XPUPlace is not supported when not compiled with XPU"));
#endif
    } else if (phi::is_custom_place(contiguous_tensor.place())) {
#ifdef PADDLE_WITH_CUSTOM_DEVICE
      constexpr size_t kBufSize = 1024 * 1024 * 64;     // 64MB
      std::unique_ptr<char[]> buf(new char[kBufSize]);  // NOLINT
      auto& custom_device_context =
          static_cast<const phi::CustomContext&>(dev_ctx);
      phi::CPUPlace cpu;
      uintptr_t data = reinterpret_cast<uintptr_t>(data_ptr);
      while (size != 0) {
        size_t size_to_write = std::min(kBufSize, static_cast<size_t>(size));
        memory::Copy(cpu,
                     buf.get(),
                     contiguous_tensor.place(),
                     reinterpret_cast<const void*>(data),
                     size_to_write,
                     custom_device_context.stream());
        custom_device_context.Wait();
        os.write(buf.get(), size_to_write);
        data += size_to_write;
        size -= size_to_write;
      }
#else
      PADDLE_THROW(common::errors::Unimplemented(
          "CustomPlace is not supported when not compiled with "
          "CustomDevice"));
#endif
    } else {
      os.write(static_cast<const char*>(data_ptr),
               static_cast<std::streamsize>(size));
    }
  }
}

struct DeserializedDataFunctor {
  DeserializedDataFunctor(void** buf,
                          phi::DenseTensor* tensor,
                          const phi::Place& place)
      : buf_(buf), tensor_(tensor), place_(place) {}

  template <typename T>
  void apply() {
    *buf_ = tensor_->mutable_data<T>(place_);
  }

  void** buf_;
  phi::DenseTensor* tensor_;
  phi::Place place_;
};

void TensorFromStream(std::istream& is,
                      phi::DenseTensor* tensor,
                      const phi::DeviceContext& dev_ctx,
                      const size_t& seek,
                      const std::vector<int64_t>& shape) {
  uint32_t version = 0;
  is.read(reinterpret_cast<char*>(&version), sizeof(version));

  PADDLE_ENFORCE_EQ(
      version,
      0U,
      common::errors::InvalidArgument(
          "tensor version %u is not supported, Only version 0 is supported",
          version));

  proto::VarType::TensorDesc desc;
  {  // int32_t size
    // proto buffer
    int32_t size = 0;
    is.read(reinterpret_cast<char*>(&size), sizeof(size));
    std::unique_ptr<char[]> buf(new char[size]);  // NOLINT
    is.read(reinterpret_cast<char*>(buf.get()), size);
    PADDLE_ENFORCE_EQ(
        desc.ParseFromArray(buf.get(), size),
        true,
        common::errors::InvalidArgument("Cannot parse tensor desc"));
  }
  {  // read tensor
    tensor->Resize(common::make_ddim(shape));
    size_t seekg = seek * framework::SizeOfType(desc.data_type());
    is.seekg(seekg, is.cur);  // NOLINT

    void* buf = nullptr;
    phi::CPUContext ctx;
    size_t size = tensor->numel() * framework::SizeOfType(desc.data_type());
    if (phi::is_gpu_place(dev_ctx.GetPlace()) ||
        phi::is_xpu_place(dev_ctx.GetPlace()) ||
        phi::is_custom_place(dev_ctx.GetPlace())) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP) || \
    defined(PADDLE_WITH_XPU) || defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DenseTensor cpu_tensor;
      cpu_tensor.Resize(common::make_ddim(shape));
      framework::VisitDataType(
          desc.data_type(),
          DeserializedDataFunctor(&buf, &cpu_tensor, ctx.GetPlace()));
      is.read(static_cast<char*>(buf), size);  // NOLINT
      auto dst_place = dev_ctx.GetPlace();
      framework::TensorCopy(cpu_tensor, dst_place, dev_ctx, tensor);
      if (phi::is_custom_place(dev_ctx.GetPlace())) {
        dev_ctx.Wait();
      }
#else
      if (phi::is_gpu_place(dev_ctx.GetPlace())) {
        PADDLE_THROW(common::errors::Unimplemented(
            "CUDAPlace is not supported when not compiled with CUDA"));
      } else if (phi::is_xpu_place(dev_ctx.GetPlace())) {
        PADDLE_THROW(common::errors::Unimplemented(
            "XPUPlace is not supported when not compiled with XPU"));
      }
#endif
    } else {
      framework::VisitDataType(
          desc.data_type(),
          DeserializedDataFunctor(&buf, tensor, ctx.GetPlace()));
      is.read(static_cast<char*>(buf), size);  // NOLINT
    }
  }
}

void TensorFromStream(std::istream& is,
                      phi::DenseTensor* tensor,
                      const phi::DeviceContext& dev_ctx) {
  uint32_t version = 0;
  is.read(reinterpret_cast<char*>(&version), sizeof(version));
  PADDLE_ENFORCE_EQ(
      version,
      0U,
      common::errors::InvalidArgument(
          "tensor version %u is not supported, Only version 0 is supported",
          version));
  proto::VarType::TensorDesc desc;
  {  // int32_t size
     // proto buffer
    int32_t size = -1;
    is.read(reinterpret_cast<char*>(&size), sizeof(size));
    PADDLE_ENFORCE_EQ(
        is.good(),
        true,
        common::errors::Unavailable("Cannot read tensor desc size"));
    PADDLE_ENFORCE_GE(size,
                      0,
                      common::errors::InvalidArgument(
                          "phi::DenseTensor desc size should >= 0"));
    std::unique_ptr<char[]> buf(new char[size]);  // NOLINT
    is.read(reinterpret_cast<char*>(buf.get()), size);
    PADDLE_ENFORCE_EQ(
        desc.ParseFromArray(buf.get(), size),
        true,
        common::errors::InvalidArgument("Cannot parse tensor desc"));
  }
  {  // read tensor
    std::vector<int64_t> dims;
    dims.reserve(static_cast<size_t>(desc.dims().size()));
    std::copy(desc.dims().begin(), desc.dims().end(), std::back_inserter(dims));
    tensor->Resize(common::make_ddim(dims));
    void* buf = nullptr;
    phi::CPUContext ctx;
    size_t size = tensor->numel() * framework::SizeOfType(desc.data_type());
    if (phi::is_gpu_place(dev_ctx.GetPlace()) ||
        phi::is_xpu_place(dev_ctx.GetPlace()) ||
        phi::is_custom_place(dev_ctx.GetPlace())) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP) || \
    defined(PADDLE_WITH_XPU) || defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DenseTensor cpu_tensor;
      cpu_tensor.Resize(common::make_ddim(dims));
      framework::VisitDataType(
          desc.data_type(),
          DeserializedDataFunctor(&buf, &cpu_tensor, ctx.GetPlace()));
      is.read(static_cast<char*>(buf), size);  // NOLINT
      auto dst_place = dev_ctx.GetPlace();
      framework::TensorCopy(cpu_tensor, dst_place, dev_ctx, tensor);
      if (phi::is_custom_place(dev_ctx.GetPlace())) {
        dev_ctx.Wait();
      }
#else
      if (phi::is_gpu_place(dev_ctx.GetPlace())) {
        PADDLE_THROW(common::errors::Unimplemented(
            "CUDAPlace is not supported when not compiled with CUDA"));
      } else if (phi::is_xpu_place(dev_ctx.GetPlace())) {
        PADDLE_THROW(common::errors::Unimplemented(
            "XPUPlace is not supported when not compiled with XPU"));
      } else {
        PADDLE_THROW(
            common::errors::Unimplemented("CustomPlace is not supported when "
                                          "not compiled with CustomDevice"));
      }
#endif
    } else {
      framework::VisitDataType(
          desc.data_type(),
          DeserializedDataFunctor(&buf, tensor, ctx.GetPlace()));
      is.read(static_cast<char*>(buf), size);  // NOLINT
    }
  }
}

// get tensor data point by DLDataType
void* GetDstPtrByDLDataType(DLDataType type,
                            phi::DenseTensor* dst,
                            const phi::Place& dst_place) {
  // vector types not currently supported
  PADDLE_ENFORCE_LE(
      type.lanes,
      1,
      common::errors::Unimplemented("Vector type is not supported currently."));

  switch (type.bits) {
    case 8:
      if (type.code == kDLInt)
        return static_cast<void*>(dst->mutable_data<int8_t>(dst_place));
      if (type.code == kDLUInt)
        return static_cast<void*>(dst->mutable_data<uint8_t>(dst_place));
      PADDLE_THROW(common::errors::Unimplemented(
          "DLDataType code <%d> is illegal when DLDataType.bits is <%d>.",
          type.code,
          type.bits));
    case 16:
      if (type.code == kDLInt)
        return static_cast<void*>(dst->mutable_data<int16_t>(dst_place));
      if (type.code == kDLFloat)
        return static_cast<void*>(
            dst->mutable_data<phi::dtype::float16>(dst_place));
      if (type.code == kDLBfloat)
        return static_cast<void*>(
            dst->mutable_data<phi::dtype::bfloat16>(dst_place));
      PADDLE_THROW(common::errors::Unimplemented(
          "DLDataType code <%d> is illegal when DLDataType.bits is <%d>.",
          type.code,
          type.bits));
    case 32:
      if (type.code == kDLInt)
        return static_cast<void*>(dst->mutable_data<int32_t>(dst_place));
      if (type.code == kDLFloat)
        return static_cast<void*>(dst->mutable_data<float>(dst_place));
      PADDLE_THROW(common::errors::Unimplemented(
          "DLDataType code <%d> is illegal when DLDataType.bits is <%d>.",
          type.code,
          type.bits));
    case 64:
      if (type.code == kDLInt)
        return static_cast<void*>(dst->mutable_data<int64_t>(dst_place));
      if (type.code == kDLFloat)
        return static_cast<void*>(dst->mutable_data<double>(dst_place));
      if (type.code == kDLComplex)
        return static_cast<void*>(
            dst->mutable_data<phi::dtype::complex<float>>(dst_place));
      PADDLE_THROW(common::errors::Unimplemented(
          "DLDataType code <%d> is illegal when DLDataType.bits is <%d>.",
          type.code,
          type.bits));
    case 128:
      if (type.code == kDLComplex)
        return static_cast<void*>(
            dst->mutable_data<phi::dtype::complex<double>>(dst_place));
      PADDLE_THROW(common::errors::Unimplemented(
          "DLDataType code <%d> is illegal when DLDataType.bits is <%d>.",
          type.code,
          type.bits));
    default:
      PADDLE_THROW(common::errors::Unimplemented(
          "Unsupported DLDataType.bits %d.", type.bits));
  }
}

// get Tensor data dtype from given DLDataType
phi::DataType GetDstPtrByDLDataType(DLDataType type) {
  // vector types not currently supported
  PADDLE_ENFORCE_LE(
      type.lanes,
      1,
      common::errors::Unimplemented("Vector type is not supported currently."));

  switch (type.bits) {
    case 8:
      if (type.code == kDLBool) return phi::DataType::BOOL;
      if (type.code == kDLInt) return phi::DataType::INT8;
      if (type.code == kDLUInt) return phi::DataType::UINT8;
      PADDLE_THROW(common::errors::Unimplemented(
          "DLDataType code <%d> is illegal when DLDataType.bits is <%d>.",
          type.code,
          type.bits));
    case 16:
      if (type.code == kDLInt) return phi::DataType::INT16;
      if (type.code == kDLFloat) return phi::DataType::FLOAT16;
      if (type.code == kDLBfloat) return phi::DataType::BFLOAT16;
      PADDLE_THROW(common::errors::Unimplemented(
          "DLDataType code <%d> is illegal when DLDataType.bits is <%d>.",
          type.code,
          type.bits));
    case 32:
      if (type.code == kDLInt) return phi::DataType::INT32;
      if (type.code == kDLFloat) return phi::DataType::FLOAT32;
      PADDLE_THROW(common::errors::Unimplemented(
          "DLDataType code <%d> is illegal when DLDataType.bits is <%d>.",
          type.code,
          type.bits));
    case 64:
      if (type.code == kDLInt) return phi::DataType::INT64;
      if (type.code == kDLFloat) return phi::DataType::FLOAT64;
      if (type.code == kDLComplex) return phi::DataType::COMPLEX64;
      PADDLE_THROW(common::errors::Unimplemented(
          "DLDataType code <%d> is illegal when DLDataType.bits is <%d>.",
          type.code,
          type.bits));
    case 128:
      if (type.code == kDLComplex) return phi::DataType::COMPLEX128;
      PADDLE_THROW(common::errors::Unimplemented(
          "DLDataType code <%d> is illegal when DLDataType.bits is <%d>.",
          type.code,
          type.bits));
    default:
      PADDLE_THROW(common::errors::Unimplemented(
          "Unsupported DLDataType.bits %d.", type.bits));
  }
}

/*
dlpack related code ref:
https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/DLConvertor.cpp
and paddle/phi/api/lib/tensor_utils.cc
*/
using Deleter = std::function<void(void*)>;

std::unordered_map<void*, std::function<void(phi::Allocation*)>> ptr_to_deleter;
std::mutex ptr_to_deleter_mutex;  // use mutex to keep thread safe

void DeleterBridge(phi::Allocation* alloc) {
  std::lock_guard<std::mutex> lock(ptr_to_deleter_mutex);
  auto it = ptr_to_deleter.find(static_cast<void*>(alloc->ptr()));
  if (it != ptr_to_deleter.end()) {
    it->second(alloc);         // call the deleter
    ptr_to_deleter.erase(it);  // remove the entry from the map safely
  }
}

phi::DataType ConvertToPDDataType(const std::string& typestr) {
  static const std::unordered_map<std::string, phi::DataType> type_map = {
      {"<c8", phi::DataType::COMPLEX64},
      {"<c16", phi::DataType::COMPLEX128},
      {"<f2", phi::DataType::BFLOAT16},
      {"<f4", phi::DataType::FLOAT32},
      {"<f8", phi::DataType::FLOAT64},
      {"|u1", phi::DataType::UINT8},
      {"|i1", phi::DataType::INT8},
      {"<i2", phi::DataType::INT16},
      {"<i4", phi::DataType::INT32},
      {"<i8", phi::DataType::INT64},
      {"|b1", phi::DataType::BOOL},
      // NOTE: Paddle not support uint32, uint64, uint16 yet.
      // {"<u2", phi::DataType::UINT16},
      // {"<u4", phi::DataType::UINT32},
      // {"<u8", phi::DataType::UINT64},
  };
  auto it = type_map.find(typestr);
  PADDLE_ENFORCE_NE(
      it,
      type_map.end(),
      common::errors::InvalidArgument("Unsupported typestr: " + typestr));
  return it->second;
}

phi::DenseTensor from_blob(void* data,
                           DLManagedTensor* src,
                           const phi::DDim& shape,
                           const phi::DDim& strides,
                           phi::DataType dtype,
                           const phi::Place& place,
                           const Deleter& deleter) {
  auto meta = phi::DenseTensorMeta(dtype, shape, strides);

  phi::Allocation::DeleterFnPtr f = nullptr;
  if (deleter) {
    auto g = [deleter, src](phi::Allocation* p) {
      if (src->manager_ctx) {
        deleter(src);
      }
    };

    {
      std::lock_guard<std::mutex> lock(ptr_to_deleter_mutex);
      ptr_to_deleter[data] = g;
    }

    f = DeleterBridge;
  }

  // Calculate the number of elements of underlying storage
  size_t size = 1;
  for (auto i = 0; i < shape.size(); ++i) {
    if (shape[i] == 0) {
      size = 0;
      break;
    }
    size += strides[i] * (shape[i] - 1);
  }

  auto alloc =
      std::make_shared<phi::Allocation>(data, size * SizeOf(dtype), f, place);
  return phi::DenseTensor(alloc, meta);
}

phi::DenseTensor TensorFromDLPack(DLManagedTensor* src, Deleter deleter) {
  std::vector<int64_t> shape_vec;
  std::copy(src->dl_tensor.shape,
            src->dl_tensor.shape + src->dl_tensor.ndim,
            std::back_inserter(shape_vec));

  phi::Place place;
  if (src->dl_tensor.device.device_type == kDLCPU) {
    place = phi::CPUPlace();
  } else if (src->dl_tensor.device.device_type == kDLCUDA) {
    place = phi::GPUPlace(src->dl_tensor.device.device_id);
  } else if (src->dl_tensor.device.device_type == kDLCUDAHost) {
    place = phi::GPUPinnedPlace();
  } else {
    PADDLE_THROW(common::errors::Unimplemented("Given Place is not supported"));
  }

  ::DLDataType type = src->dl_tensor.dtype;
  auto dtype = GetDstPtrByDLDataType(type);
  if (!src->dl_tensor.strides) {
    return from_blob(
        src->dl_tensor.data,
        src,
        common::make_ddim(shape_vec),
        phi::DenseTensorMeta::calc_strides(common::make_ddim(shape_vec)),
        dtype,
        place,
        std::move(deleter));
  } else {
    std::vector<int64_t> strides_vec;
    std::copy(src->dl_tensor.strides,
              src->dl_tensor.strides + src->dl_tensor.ndim,
              std::back_inserter(strides_vec));
    return from_blob(src->dl_tensor.data,
                     src,
                     common::make_ddim(shape_vec),
                     common::make_ddim(strides_vec),
                     dtype,
                     place,
                     deleter);
  }
}

phi::DenseTensor TensorFromDLPack(DLManagedTensor* src) {
  auto deleter = [src](void* self [[maybe_unused]]) {
    if (src->deleter) {
      src->deleter(src);
    }
  };
  return TensorFromDLPack(src, std::move(deleter));
}

// Keep the this overloaded version of the interface unchanged.
void TensorFromDLPack(const ::DLTensor& dl_tensor, phi::DenseTensor* dst) {
  phi::CPUPlace dst_place = phi::CPUPlace();
  phi::CPUPlace src_place = phi::CPUPlace();

  std::vector<int64_t> vec;
  std::copy(dl_tensor.shape,
            dl_tensor.shape + dl_tensor.ndim,
            std::back_inserter(vec));

  phi::DDim vddim = common::make_ddim(vec);

  dst->Resize(vddim);
  ::DLDataType type = dl_tensor.dtype;
  void* dst_ptr = GetDstPtrByDLDataType(type, dst, dst_place);

  auto src_ptr = static_cast<const void*>(dl_tensor.data);
  auto size = common::product(vddim) * type.bits / 8;

  if (dl_tensor.device.device_type == kDLCPU) {
    memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size);
  }
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  if (dl_tensor.device.device_type == kDLCUDA) {
    phi::GPUPlace dst_place = phi::GPUPlace(dl_tensor.device.device_id);
    phi::GPUPlace src_place = phi::GPUPlace(dl_tensor.device.device_id);
    dst_ptr = GetDstPtrByDLDataType(type, dst, dst_place);
    auto* ctx = phi::DeviceContextPool::Instance().GetByPlace(dst_place);
    memory::Copy(dst_place,
                 dst_ptr,
                 src_place,
                 src_ptr,
                 size,
                 reinterpret_cast<const phi::GPUContext&>(*ctx).stream());
  }
#endif
#ifdef PADDLE_WITH_XPU
  PADDLE_THROW(common::errors::Unimplemented("XPUPlace is not supported"));
#endif
}

template <typename T>
std::string format_tensor(const phi::DenseTensor& tensor) {
  // TODO(zhiqiu): use the print option to format tensor.
  return "NOT IMPLEMENTED";
}

template <typename T>
std::ostream& print_tensor(std::ostream& os, const phi::DenseTensor& tensor) {
  auto inspect = tensor.data<T>();
  auto element_num = tensor.numel();

  if (element_num > 100) {
    element_num = 100;
  }
  os << "  - data: [";
  // Note: int8_t && uint8_t is typedef of char, ostream unable to print
  // properly
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

std::ostream& operator<<(std::ostream& os, const LegacyLoD& lod) {
  // NOTE(xiongkun):
  // https://stackoverflow.com/questions/5195512/namespaces-and-operator-resolution
  // if we don't redefine, the operator << of phi / framework LoD is not found.
  paddle::string::operator<<(os, lod);
  return os;
}

}  // namespace paddle::framework

namespace phi {

std::ostream& operator<<(std::ostream& os, const LegacyLoD& lod) {
  paddle::string::operator<<(os, lod);
  return os;
}

TEST_API std::ostream& operator<<(std::ostream& os, const phi::DenseTensor& t) {
  if (!t.valid()) {
    os << "invalid\n";
    return os;
  }

  if (!t.lod().empty()) {
    os << "  - lod: " << t.lod() << "\n";
  }
  os << "  - shape: [" << t.dims() << "]\n";
  os << "  - layout: " << common::DataLayoutToString(t.layout()) << "\n";

  if (!t.has_allocation()) {
    os << "uninited\n";
    return os;
  }

  os << "  - place: " << t.place() << "\n";

  DenseTensor tensor;
  tensor.Resize(t.dims());
  if (phi::is_cpu_place(t.place())) {
    tensor.ShareDataWith(t);
  } else {
    phi::CPUPlace place;
    paddle::framework::TensorCopy(t, place, &tensor);
    phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
    auto& dev_ctx = *pool.Get(t.place());
    dev_ctx.Wait();
  }

#define PrintTensorCallback(cpp_type, proto_type)                 \
  do {                                                            \
    if (paddle::framework::TransToProtoVarType(tensor.dtype()) == \
        proto_type) {                                             \
      os << "  - dtype: " << tensor.dtype() << "\n";              \
      paddle::framework::print_tensor<cpp_type>(os, tensor);      \
      return os;                                                  \
    }                                                             \
  } while (0)

  _ForEachDataType_(PrintTensorCallback);
  VLOG(1) << "PrintVar: unrecognized data type:" << t.type();
  return os;
}
}  // namespace phi
