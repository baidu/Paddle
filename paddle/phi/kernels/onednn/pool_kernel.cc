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

#include "paddle/phi/kernels/pool_kernel.h"

#include "paddle/phi/backends/onednn/onednn_reuse.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {
bool Pool2dCheckIfOneDNNSupport(const KernelContext* ctx) {
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
void Pool2dKernel(const Context& dev_ctx,
                  const DenseTensor& x,
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
                  DenseTensor* out) {
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
                                         out);
  bool is_test = dev_ctx.HasDnnAttr("is_test")
                     ? PADDLE_GET_CONST(bool, dev_ctx.GetDnnAttr("is_test"))
                     : false;

  auto src_memory = handler.AcquireSrcMemory(&x);
  auto dst_memory = handler.AcquireDstMemory(out);

  auto pool_p = handler.AcquireForwardPrimitive();

  auto& astream = OneDNNContext::tls().get_stream();
  if (is_test == false && pooling_type == "max") {
    // Training
    auto workspace_memory = handler.AcquireWorkspaceMemory(dev_ctx, "Out");
    pool_p->execute(astream,
                    {{DNNL_ARG_SRC, *src_memory},
                     {DNNL_ARG_DST, *dst_memory},
                     {DNNL_ARG_WORKSPACE, *workspace_memory}});
  } else {
    // Inference
    pool_p->execute(astream,
                    {{DNNL_ARG_SRC, *src_memory}, {DNNL_ARG_DST, *dst_memory}});
  }
  astream.wait();

  out->set_mem_desc(dst_memory->get_desc());
}

phi::KernelKey PoolOpGetKernelTypeForVar(
    const GetKernelTypeForVarContext* ctx) {
  const phi::DenseTensor& tensor = ctx->GetTensor();
  const phi::KernelKey& expected_kernel_type = ctx->GetKernelKey();
#ifdef PADDLE_WITH_DNNL
  if ((expected_kernel_type.layout() == phi::DataLayout::ONEDNN) &&
      (tensor.layout() != phi::DataLayout::ONEDNN)) {
    const AttributeMap& attrs = ctx->GetAttrs();
    auto it = attrs.find("data_format");
    const std::string data_format = PADDLE_GET_CONST(std::string, it->second);
    auto dl = common::StringToDataLayout(data_format);
    // Some models may have intentionally set "AnyLayout" for pool
    // op. Treat this as NCHW (default data_format value)
    if (dl != phi::DataLayout::kAnyLayout) {
      return phi::KernelKey(tensor.place(), dl, expected_kernel_type.dtype());
    }
  }
#endif
  return phi::KernelKey(
      tensor.place(), tensor.layout(), expected_kernel_type.dtype());
}

}  // namespace phi

PD_REGISTER_KERNEL(pool2d,
                   OneDNN,
                   ONEDNN,
                   phi::Pool2dKernel,
                   float,
                   int8_t,
                   uint8_t,
                   phi::dtype::bfloat16) {
  kernel->get_kerneltype_forvar_fn_ = phi::PoolOpGetKernelTypeForVar;
  kernel->check_if_onednn_kernel_support_ = phi::Pool2dCheckIfOneDNNSupport;
}
