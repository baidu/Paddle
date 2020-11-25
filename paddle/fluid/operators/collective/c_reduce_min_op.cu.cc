/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/collective/c_reduce_op.h"

namespace paddle {
namespace platform {
struct CUDAPlace;
struct float16;
}  // namespace platform
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_CUDA_KERNEL(c_reduce_min,
                        ops::CReduceOpCUDAKernel<ops::kRedMin, float>,
                        ops::CReduceOpCUDAKernel<ops::kRedMin, double>,
                        ops::CReduceOpCUDAKernel<ops::kRedMin, int>,
                        ops::CReduceOpCUDAKernel<ops::kRedMin, int64_t>,
                        ops::CReduceOpCUDAKernel<ops::kRedMin, plat::float16>)
