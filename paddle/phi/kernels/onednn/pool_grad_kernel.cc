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

#include "paddle/phi/kernels/pool_grad_kernel.h"

#include "paddle/phi/backends/onednn/onednn_reuse.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {
bool Pool2dGradCheckIfOneDNNSupport(const KernelContext* ctx) {
  if (ctx->AttrAt<bool>(8) == false) {
    // adaptive
    return true;
  }
  // oneDNN is supporting only unchangeable in size pool window
  auto src_tz = common::vectorize(ctx->InputAt<phi::DenseTensor>(0).dims());
  const TensorRef& kernel_size_tmp = ctx->AttrAt<TensorRef>(0);
  IntArray kernel_size_array = IntArray(*kernel_size_tmp.Get());
  std::vector<int64_t> kernel_size = kernel_size_array.GetData();
  // Fast but not exhaustive check
  return ((src_tz[src_tz.size() - 1] % kernel_size[1] == 0) &&
          (src_tz[src_tz.size() - 2] % kernel_size[0] == 0));
}

template <typename T, typename Context>
void Pool2dGradKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& out UNUSED,
                      const DenseTensor& dout,
                      const IntArray& kernel_size,
                      const std::vector<int>& strides,
                      const std::vector<int>& paddings,
                      bool ceil_mode,
                      bool exclusive,
                      const std::string& data_format UNUSED,
                      const std::string& pooling_type,
                      bool global_pooling,
                      bool adaptive,
                      const std::string& padding_algorithm,
                      DenseTensor* dx) {
  funcs::PoolingOneDNNHandler<T> handler(dev_ctx,
                                         pooling_type,
                                         kernel_size,
                                         strides,
                                         paddings,
                                         global_pooling,
                                         padding_algorithm,
                                         ceil_mode,
                                         exclusive,
                                         adaptive,
                                         &x,
                                         &dout,
                                         dx);

  auto diff_dst_memory = handler.AcquireDiffDstMemory(&dout);
  auto diff_src_memory = handler.AcquireDiffSrcMemory(dx);

  auto pool_bwd_p = handler.AcquireBackwardPrimitive();

  auto& astream = OneDNNContext::tls().get_stream();
  if (pooling_type == "max") {
    // Max - pooling needs Workspace
    auto workspace_memory = handler.AcquireWorkspaceMemory(dev_ctx, "Out");
    pool_bwd_p->execute(astream,
                        {{DNNL_ARG_DIFF_SRC, *diff_src_memory},
                         {DNNL_ARG_DIFF_DST, *diff_dst_memory},
                         {DNNL_ARG_WORKSPACE, *workspace_memory}});
  } else {
    // Average Pooling
    pool_bwd_p->execute(astream,
                        {{DNNL_ARG_DIFF_SRC, *diff_src_memory},
                         {DNNL_ARG_DIFF_DST, *diff_dst_memory}});
  }
  astream.wait();

  dx->set_mem_desc(diff_src_memory->get_desc());
}

phi::KernelKey PoolOpGradGetKernelTypeForVar(
    const GetKernelTypeForVarContext* ctx) {
  const DenseTensor& tensor = ctx->GetTensor();
  const KernelKey& expected_kernel_type = ctx->GetKernelKey();
#ifdef PADDLE_WITH_DNNL
  if ((expected_kernel_type.layout() == phi::DataLayout::ONEDNN) &&
      (tensor.layout() != phi::DataLayout::ONEDNN)) {
    const AttributeMap& attrs = ctx->GetAttrs();
    auto it = attrs.find("data_format");
    const std::string data_format = PADDLE_GET_CONST(std::string, it->second);
    return phi::KernelKey(tensor.place(),
                          common::StringToDataLayout(data_format),
                          expected_kernel_type.dtype());
  }
#endif
  return phi::KernelKey(
      tensor.place(), tensor.layout(), expected_kernel_type.dtype());
}

}  // namespace phi

PD_REGISTER_KERNEL(pool2d_grad,
                   OneDNN,
                   ONEDNN,
                   phi::Pool2dGradKernel,
                   float,
                   phi::dtype::bfloat16) {
  kernel->get_kerneltype_forvar_fn_ = phi::PoolOpGradGetKernelTypeForVar;
  kernel->check_if_onednn_kernel_support_ = phi::Pool2dGradCheckIfOneDNNSupport;
}
