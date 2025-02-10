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

#include "paddle/phi/api/include/tensor_utils.h"
#include "glog/logging.h"

#include "paddle/phi/api/lib/api_registry.h"
#include "paddle/phi/api/lib/data_transform.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/reshard_utils.h"
#include "paddle/phi/core/enforce.h"

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#ifdef PADDLE_WITH_CUDA
#include <cuda_runtime.h>
#else
#include <hip/hip_runtime.h>
#endif
#endif

namespace paddle {

PD_REGISTER_API(from_blob)

phi::Place GetPlaceFromPtr(void* data) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#ifdef PADDLE_WITH_CUDA
#if CUDA_VERSION >= 10000
  cudaPointerAttributes attr = {};
  cudaError_t status = cudaPointerGetAttributes(&attr, data);
  if (status == cudaSuccess && attr.type == cudaMemoryTypeDevice) {
    return phi::GPUPlace(attr.device);
  }
#else
  PADDLE_THROW(
      common::errors::Unimplemented("The GetPlaceFromPtr() method is only "
                                    "supported when CUDA version >= 10.0."));
#endif
#else
  hipPointerAttribute_t attr = {};
  hipError_t status = hipPointerGetAttributes(&attr, data);
#if HIP_VERSION >= 60000000
  if (status == hipSuccess && attr.type == hipMemoryTypeDevice) {
#else
  if (status == hipSuccess && attr.memoryType == hipMemoryTypeDevice) {
#endif
    return phi::GPUPlace(attr.device);
  }
#endif
#endif
  return phi::CPUPlace();
}

struct DeleterManeger {
  static DeleterManeger* Instance() {
    static DeleterManeger instance;
    return &instance;
  }
  DeleterManeger() = default;

  void DeletePtr(void* ptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = ptr2deleter_.find(ptr);
    if (it != ptr2deleter_.end()) {
      it->second(ptr);
      ptr2deleter_.erase(it);
    } else {
      PADDLE_THROW(common::errors::InvalidArgument(
          "The deleter of the pointer is not found."));
    }
  }

  void RegisterPtr(void* ptr, const Deleter& deleter) {
    std::lock_guard<std::mutex> lock(mutex_);
    ptr2deleter_[ptr] = deleter;
  }

 private:
  std::unordered_map<void*, Deleter> ptr2deleter_;
  std::mutex mutex_;
};

using AllocationDeleter = void (*)(phi::Allocation*);

Tensor FromBlobImpl(void* data,
                    const phi::DenseTensorMeta& meta,
                    const phi::Place& place,
                    const Deleter& deleter) {
  PADDLE_ENFORCE_NOT_NULL(
      data, common::errors::InvalidArgument("data can not be nullptr."));

  phi::Place data_place;
  if (place.GetType() == phi::AllocationType::UNDEFINED ||
      place.GetType() == phi::AllocationType::CPU ||
      place.GetType() == phi::AllocationType::GPU) {
    data_place = GetPlaceFromPtr(data);
    if (place.GetType() != phi::AllocationType::UNDEFINED) {
      PADDLE_ENFORCE_EQ(data_place,
                        place,
                        common::errors::InvalidArgument(
                            "Specified place does not match place of data. ",
                            "Specified: %s, Expected: %s.",
                            data_place.DebugString(),
                            place.DebugString()));
    }
  } else {
    data_place = place;
  }

  // Calculate the number of elements of underlying storage
  size_t size = 1;
  for (auto i = 0; i < meta.dims.size(); ++i) {
    if (meta.dims[i] == 0) {
      size = 0;
      break;
    }
    size += meta.strides[i] * (meta.dims[i] - 1);
  }

  AllocationDeleter alloc_deleter = nullptr;
  if (deleter) {
    DeleterManeger::Instance()->RegisterPtr(data, deleter);
    alloc_deleter = [](phi::Allocation* p) {
      DeleterManeger::Instance()->DeletePtr(p->ptr());
    };
  }

  auto alloc = std::make_shared<phi::Allocation>(
      data, size * SizeOf(meta.dtype), alloc_deleter, data_place);

  return Tensor(std::make_shared<phi::DenseTensor>(alloc, meta));
}

PADDLE_API Tensor from_blob(void* data,
                            const phi::IntArray& shape,
                            phi::DataType dtype,
                            phi::DataLayout layout,
                            const phi::Place& place,
                            const Deleter& deleter) {
  auto meta =
      phi::DenseTensorMeta(dtype, common::make_ddim(shape.GetData()), layout);
  return FromBlobImpl(data, meta, place, deleter);
}

PADDLE_API Tensor from_blob(void* data,
                            const phi::IntArray& shape,
                            const phi::IntArray& strides,
                            phi::DataType dtype,
                            phi::DataLayout layout,
                            const phi::Place& place,
                            const Deleter& deleter) {
  auto meta = phi::DenseTensorMeta(dtype,
                                   common::make_ddim(shape.GetData()),
                                   common::make_ddim(strides.GetData()));
  return FromBlobImpl(data, meta, place, deleter);
}

#ifdef PADDLE_WITH_DISTRIBUTE
PD_REGISTER_API(reshard)

PADDLE_API std::shared_ptr<phi::distributed::DistTensor> reshard(
    const paddle::Tensor& input,
    const phi::distributed::TensorDistAttr& dist_attr) {
  PADDLE_ENFORCE_EQ(input.is_dist_tensor(),
                    true,
                    common::errors::InvalidArgument(
                        "The input tensor of ReshardFunction should be "
                        "``phi::distributed::DistTensor``. "
                        "However it's %s",
                        typeid(input.impl().get()).name()));
  auto dev_ctx = phi::distributed::GetDistTensorDeviceContext(
      static_cast<phi::distributed::DistTensor*>(input.impl().get()));
  const auto& input_tensor_impl = input.impl();
  std::shared_ptr<phi::distributed::DistTensor> dist_out_ptr = nullptr;
  if (input_tensor_impl) {
    phi::distributed::DistTensor* dist_tensor =
        static_cast<phi::distributed::DistTensor*>(input_tensor_impl.get());

    if (!IsCurRankInMesh(dist_attr.process_mesh()) &&
        !IsCurRankInMesh(dist_tensor->dist_attr().process_mesh())) {
      PADDLE_ENFORCE_EQ(
          dist_tensor->initialized(),
          false,
          common::errors::InvalidArgument(
              "Only "
              "uninitialized ``phi::distributed::DistTensor`` is allowed. "));
      VLOG(4) << "reshard tensor which is not in current mesh, just set its "
                 "dist_attr "
              << "from " << dist_tensor->dist_attr() << " to " << dist_attr;

      phi::distributed::DistTensor* dist_tensor =
          static_cast<phi::distributed::DistTensor*>(input_tensor_impl.get());
      dist_out_ptr = std::make_shared<phi::distributed::DistTensor>(
          dist_tensor->dims(), dist_attr);
      phi::DenseTensor* dense_out = dist_out_ptr->unsafe_mutable_value();
      *dense_out = dist_tensor->value();
      return dist_out_ptr;
    }

    if (dist_tensor->dist_attr() != dist_attr) {
      auto tensor_name = (input.name().empty() ? "None" : input.name());
      VLOG(4) << "Reshard func: tensor(" << tensor_name << ") "
              << paddle::experimental::ReshardDebugInfo(*dist_tensor,
                                                        dist_attr);
      auto* func = phi::distributed::ChooseProperReshardFunction(*dist_tensor,
                                                                 dist_attr);
      dist_out_ptr = func->Eval(dev_ctx, *dist_tensor, dist_attr);
    } else {
      dist_out_ptr = std::static_pointer_cast<phi::distributed::DistTensor>(
          input_tensor_impl);
    }
  }
  return dist_out_ptr;
}
#endif
}  // namespace paddle
