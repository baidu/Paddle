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

#include "paddle/phi/kernels/scale_kernel.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/onednn/onednn_reuse.h"

namespace phi {

template <typename T, typename Context>
void ScaleKernel(const Context& dev_ctx,
                 const DenseTensor& x,
                 const Scalar& scale,
                 float bias,
                 bool bias_after_scale,
                 DenseTensor* out) {
  float alpha = scale.to<float>();
  float beta = bias_after_scale ? bias : bias * alpha;

  funcs::ActivationMKLDNNHandler<T> handler(dnnl::algorithm::eltwise_linear,
                                            alpha,
                                            beta,
                                            dev_ctx.GetEngine(),
                                            dev_ctx.GetPlace(),
                                            &x);

  auto src_memory_p = handler.AcquireSrcMemory(&x);
  auto dst_memory_p = handler.AcquireDstMemory(out);
  auto activation_p = handler.AcquireForwardPrimitive();

  auto& astream = OneDNNContext::tls().get_stream();
  activation_p->execute(
      astream, {{DNNL_ARG_FROM, *src_memory_p}, {DNNL_ARG_TO, *dst_memory_p}});
  astream.wait();

  out->set_mem_desc(dst_memory_p->get_desc());
}

}  // namespace phi

PD_REGISTER_KERNEL(
    scale, OneDNN, ALL_LAYOUT, phi::ScaleKernel, float, phi::dtype::bfloat16) {}
