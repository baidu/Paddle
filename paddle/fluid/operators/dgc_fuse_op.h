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

#pragma once
#include <vector>

#include "dgc/dgc.h"
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/memory/malloc.h"
#include "paddle/fluid/operators/dgc_op.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_function.h"
#include "paddle/phi/kernels/funcs/elementwise_functor.h"
#include "paddle/fluid/platform/device_context.h"
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/fluid/distributed/collective/ProcessGroup.h"
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/device/gpu/nccl_helper.h"
#endif

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class DGCFuseOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto g = ctx.Input<phi::DenseTensor>("Grad");
    auto grad_out = ctx.Output<framework::Tensor>("Grad_out");
    auto place = ctx.GetPlace();
    grad_out->mutable_data<T>(g->dims(), place);
    auto& dev_ctx = ctx.template device_context<DeviceContext>();

    // nranks
    auto nranks_tensor = ctx.Input<framework::Tensor>("nranks");
    const int nranks = static_cast<const int>(*nranks_tensor->data<float>());
    PADDLE_ENFORCE_GT(nranks, 1,
                      platform::errors::PreconditionNotMet(
                          "DGC is not useful when num_trainers <= 1. Please "
                          "use multi card or multi machine GPU"));
    // stream
    const int rid = ctx.Attr<int>("ring_id");
    auto map = distributed::ProcessGroupMapFromGid::getInstance();
    PADDLE_ENFORCE_EQ(
    map->has(rid), true,
    platform::errors::InvalidArgument("dgc only nomally work after PaddlePaddle==2.3.1"));
    bool is_use_dgc = ctx.Attr<bool>("is_use_dgc");
    if (!is_use_dgc) {
        distributed::ProcessGroup* pg = map->get(rid);
        std::vector<phi::DenseTensor> in_tensor = {*g};
        std::vector<phi::DenseTensor> out_tensor = {*grad_out};
        pg->AllReduce(in_tensor, out_tensor);
        return;
    }
    
    LOG(INFO) << "========33333333333=========";
    // reuse dgc op
    if (!DGCOpFunction<DeviceContext, T>(ctx)){
	    return;
    }
    LOG(INFO) << "========44444444444=========";

    auto encode_grad_out = ctx.Output<framework::Tensor>("EncodeGrad");
    auto gather_buff = ctx.Output<framework::Tensor>("GatherBuff");


    LOG(INFO) << "========5555555555=========";
    auto k_out = ctx.Output<framework::Tensor>("k");
    int64_t k = static_cast<int64_t>(*k_out->data<T>());

    LOG(INFO) << "======nnnnnnnnn========";
    // do dgc comm
    distributed::ProcessGroup* pg = map->get(rid);
    std::vector<phi::DenseTensor> in_tensor;
    std::vector<phi::DenseTensor> out_tensor;
    in_tensor.push_back(*encode_grad_out);
    out_tensor.push_back(*gather_buff);

    LOG(INFO) << "======6666666666========";
    pg->AllGather(in_tensor, out_tensor);
    
    LOG(INFO) << "======+AAAAAAAAA=========";
    std::vector<std::unique_ptr<phi::GPUContext>> ctxs = pg->GetDeviceContext(in_tensor);

    PADDLE_ENFORCE_EQ(
        paddle::communication::dgc::sparseReduce(
            static_cast<void*>(gather_buff->data()), k, grad_out->data<T>(),
            grad_out->numel(), nranks, ctxs[0]->stream()),
        true, platform::errors::Unavailable("Calling sparseReduce() failed."));

    LOG(INFO) << "======BBBBBBBBBBB=========";
  }

};
}  // namespace operators
}  // namespace paddle
