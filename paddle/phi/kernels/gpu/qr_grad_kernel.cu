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

#include "paddle/phi/kernels/qr_grad_kernel.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/impl/qr_grad_kernel_impl.h"

#ifdef PADDLE_WITH_HIP
PD_REGISTER_KERNEL(qr_grad, GPU, ALL_LAYOUT, phi::QrGradKernel, float, double) {
}
#else
PD_REGISTER_KERNEL(qr_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::QrGradKernel,
                   float,
                   double,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
#endif
