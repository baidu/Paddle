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

#include "paddle/phi/api/include/manual_api.h"

#include <memory>

#include "glog/logging.h"

#include "paddle/phi/api/lib/api_registry.h"
#include "paddle/phi/api/lib/api_utils.h"
#include "paddle/phi/api/lib/data_transform.h"
#include "paddle/phi/api/lib/kernel_dispatch.h"
#include "paddle/phi/api/lib/utils/storage.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/meta_tensor.h"
#include "paddle/phi/infermeta/unary.h"

PT_DECLARE_KERNEL(copy, CPU, ALL_LAYOUT);
PT_DECLARE_KERNEL(split, CPU, ALL_LAYOUT);

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PT_DECLARE_KERNEL(copy, GPU, ALL_LAYOUT);
#endif

#ifdef PADDLE_WITH_XPU
PT_DECLARE_KERNEL(copy, XPU, ALL_LAYOUT);
#endif

namespace paddle {
namespace experimental {

PADDLE_API Tensor copy_to(const Tensor& x, Backend backend, bool blocking) {
  auto kernel_key_set = ParseKernelKeyByInputArgs(x);
  kernel_key_set.backend_set = kernel_key_set.backend_set | BackendSet(backend);
  auto kernel_key = kernel_key_set.GetHigestPriorityKernelKey();
  auto kernel = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "copy", kernel_key);

  VLOG(6) << "copy API kernel key: " << kernel_key;
  VLOG(6) << "copy API kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_key.backend());

  auto dense_x = TensorToDenseTensor(x);

  Tensor out;
  auto kernel_out = SetKernelOutput(kernel_key.backend(), &out);
  phi::MetaTensor meta_out(kernel_out);
  phi::UnchangedInferMeta(*dense_x, &meta_out);

  using kernel_signature = void (*)(const platform::DeviceContext&,
                                    const phi::DenseTensor&,
                                    phi::Place,
                                    bool,
                                    phi::DenseTensor*);

  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  (*kernel_fn)(
      *dev_ctx, *dense_x, phi::TransToPtenPlace(backend), blocking, kernel_out);

  return out;
}

PADDLE_API std::vector<Tensor> split(const Tensor& x,
                                     const ScalarArray& num_or_sections,
                                     const Scalar& axis) {
  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED ||
      kernel_layout == DataLayout::UNDEFINED ||
      kernel_data_type == DataType::UNDEFINED) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHigestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  auto kernel = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "split", {kernel_backend, kernel_layout, kernel_data_type});
  VLOG(6) << "split API kernel key: [" << kernel_backend << ", "
          << kernel_layout << ", " << kernel_data_type << "]";
  VLOG(6) << "split API kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_backend);

  auto dense_x = PrepareData(x, kernel.InputAt(0), {});

  // Calculate the number of out tensors
  size_t out_number;
  if (num_or_sections.GetData().size() == 1) {
    out_number = num_or_sections.GetData()[0];
  } else {
    out_number = num_or_sections.GetData().size();
  }

  std::vector<Tensor> out;
  auto dense_outs = SetKernelOutput(out_number, kernel_backend, &out);
  std::vector<phi::MetaTensor> meta_outs;
  for (size_t i = 0; i < out_number; ++i) {
    meta_outs.push_back(dense_outs[i]);
  }

  phi::SplitInferMeta(
      MakeMetaTensor(*dense_x), num_or_sections, axis, &meta_outs);

  using kernel_signature = void (*)(const platform::DeviceContext&,
                                    const phi::DenseTensor&,
                                    const phi::ScalarArray&,
                                    const phi::Scalar&,
                                    std::vector<phi::DenseTensor*>&);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  (*kernel_fn)(*dev_ctx,
               *dense_x,
               phi::ScalarArray(num_or_sections),
               phi::Scalar(axis),
               dense_outs);

  return out;
}
}  // namespace experimental
}  // namespace paddle

PT_REGISTER_API(Utils);
