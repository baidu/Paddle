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

#include "paddle/phi/kernels/accuracy_kernel.h"

#include <thrust/execution_policy.h>
#include <thrust/reduce.h>

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {
using phi::PADDLE_CUDA_NUM_THREADS;

template <int BlockSize, typename T>
__global__ void AccuracyCudaKernel(const int N,
                                   const int D,
                                   const int64_t* Xdata,
                                   const int64_t* labeldata,
                                   int* correct_data,
                                   T* accuracy,
                                   int* total_data,
                                   gpuStream_t stream) {
  using MT = typename phi::dtype::MPTypeTrait<T>::Type;
  int count = 0;
  __shared__ int total[BlockSize];

  // support only 1 block
  for (int i = threadIdx.x; i < (N); i += BlockSize) {
    for (int j = 0; j < D; ++j) {
      if (Xdata[i * D + j] == labeldata[i]) {
        ++count;
        break;
      }
    }
  }
  total[threadIdx.x] = count;
  __syncthreads();

// reduce the count with init value 0, and output accuracy.
#ifdef PADDLE_WITH_CUDA
  int result =
      thrust::reduce(thrust::cuda::par.on(stream), total, total + BlockSize, 0);
#else
  // HIP thrust::reduce not support __device__
  for (int s = BlockSize / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      total[threadIdx.x] += total[threadIdx.x + s];
    }
    __syncthreads();
  }
  int result = total[0];
#endif
  if (threadIdx.x == 0) {
    *correct_data = result;
    *accuracy = static_cast<T>(static_cast<MT>(result) / static_cast<MT>(N));
    *total_data = N;
  }
}

template <typename T, typename Context>
void AccuracyKernel(const Context& dev_ctx,
                    const DenseTensor& inference,
                    const DenseTensor& indices,
                    const DenseTensor& label,
                    DenseTensor* accuracy,
                    DenseTensor* correct,
                    DenseTensor* total) {
  // FIXME(typhoonzero): only support indices currently
  // if add support for output values, how to detect the data type?
  const int64_t* indices_data = indices.data<int64_t>();
  const int64_t* label_data = label.data<int64_t>();

  PADDLE_ENFORCE_EQ(
      inference.dims().size(),
      2,
      common::errors::InvalidArgument(
          "Rank(Input) of AccuracyOp must be 2, with shape "
          "[sample_number, class_dim], But received rank(Input) is %d",
          inference.dims().size()));

  int* correct_data = dev_ctx.template Alloc<int>(correct);
  int* total_data = dev_ctx.template Alloc<int>(total);
  T* accuracy_data = dev_ctx.template Alloc<T>(accuracy);

  int num_samples = static_cast<int>(inference.dims()[0]);
  size_t infer_width = inference.dims()[1];
  auto stream = dev_ctx.stream();
  phi::backends::gpu::GpuMemsetAsync(accuracy_data, 0, sizeof(T), stream);

  PADDLE_ENFORCE_GT(label.dims().size(),
                    0,
                    common::errors::InvalidArgument(
                        "Rank(Label) of AccuracyOp must greater than 0, "
                        "But received rank(Label) is %d",
                        label.dims().size()));

  PADDLE_ENFORCE_GE(label.dims()[0],
                    inference.dims()[0],
                    common::errors::InvalidArgument(
                        "num_samples(%d) of Label should less than "
                        "or equal to num_samples(%d) of Input",
                        label.dims()[0],
                        num_samples));

  if (num_samples == 0) {
    return;
  }

  AccuracyCudaKernel<PADDLE_CUDA_NUM_THREADS, T>
      <<<1, PADDLE_CUDA_NUM_THREADS, 0, stream>>>(num_samples,
                                                  infer_width,
                                                  indices_data,
                                                  label_data,
                                                  correct_data,
                                                  accuracy_data,
                                                  total_data,
                                                  stream);
}
}  // namespace phi

// FIXME(typhoonzero): types of T is for inference data.
// label data is always int64
PD_REGISTER_KERNEL(accuracy,
                   GPU,
                   ALL_LAYOUT,
                   phi::AccuracyKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   float,
                   double) {
  kernel->InputAt(1).SetDataType(phi::DataType::INT64);
  kernel->InputAt(2).SetDataType(phi::DataType::INT64);
  kernel->OutputAt(1).SetDataType(phi::DataType::INT32);
  kernel->OutputAt(2).SetDataType(phi::DataType::INT32);
}
