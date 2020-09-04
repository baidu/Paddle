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

#include "paddle/fluid/operators/collective/c_scatter_op.h"

#if defined(PADDLE_WITH_NCCL)
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/nccl_helper.h"
#endif

namespace paddle {
namespace operators {

template <typename T>
class CScatterOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
#if defined(PADDLE_WITH_NCCL)
    auto x = ctx.Input<framework::LoDTensor>("X");
    auto out = ctx.Output<framework::LoDTensor>("Out");
    int numel = x->numel();
    ncclDataType_t dtype = platform::ToNCCLDataType(x->type());

    int nranks = ctx.Attr<int>("nranks");
    int root_id = ctx.Attr<int>("root");
    int ring_id = ctx.Attr<int>("ring_id");
    auto place = ctx.GetPlace();
    auto comm = platform::NCCLCommContext::Instance().Get(ring_id, place);
    PADDLE_ENFORCE_EQ(nranks, comm->nranks(),
                      platform::errors::InvalidArgument(
                          "The number of ranks (%d) you set must "
                          "be equal to comm->nranks (%d).",
                          nranks, comm->nranks()));
    PADDLE_ENFORCE_GE(
        root_id, 0,
        platform::errors::InvalidArgument(
            "The root_id (%d) for c_scatter_op must be non-negative.",
            root_id));
    PADDLE_ENFORCE_GE(
        ring_id, 0,
        platform::errors::InvalidArgument(
            "The ring_id (%d) for c_scatter_op must be non-negative.",
            ring_id));

    cudaStream_t stream = nullptr;
    if (ctx.Attr<bool>("use_calc_stream")) {
      auto dev_ctx = platform::DeviceContextPool::Instance().Get(place);
      stream = static_cast<platform::CUDADeviceContext*>(dev_ctx)->stream();
    } else {
      stream = comm->stream();
    }

    framework::DDim x_dims = x->dims();
    framework::DDim out_dims(x_dims);
    out_dims[0] /= nranks;
    auto send_buf = x->data<T>();
    auto send_numel = numel / nranks;
    auto recv_buf = out->mutable_data<T>(out_dims, place);
    auto offset = 0;
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::ncclGroupStart());
    if (root_id == comm->rank()) {
      for (auto i = 0; i < nranks; ++i) {
        PADDLE_ENFORCE_CUDA_SUCCESS(
            platform::dynload::ncclSend(send_buf + offset, send_numel, dtype,
                                        root_id, comm->comm(), stream));
        offset += send_numel;
      }
    }
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::ncclRecv(
        recv_buf, send_numel, dtype, root_id, comm->comm(), stream));
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::ncclGroupEnd());
#else
    PADDLE_ENFORCE_EQ(
        true, false,
        platform::errors::Unavailable("PaddlePaddle should compile with GPU."));
#endif
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_CUDA_KERNEL(c_scatter, ops::CScatterOpCUDAKernel<float>,
                        ops::CScatterOpCUDAKernel<double>,
                        ops::CScatterOpCUDAKernel<int>,
                        ops::CScatterOpCUDAKernel<int64_t>,
                        ops::CScatterOpCUDAKernel<plat::float16>);
