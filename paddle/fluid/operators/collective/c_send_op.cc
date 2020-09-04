/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/collective/c_send_op.h"

namespace paddle {
namespace operators {

class CSendOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "CSend");
    int peer = ctx->Attrs().Get<int>("peer");
    int ring_id = ctx->Attrs().Get<int>("ring_id");
    PADDLE_ENFORCE_GE(
        peer, 0,
        platform::errors::InvalidArgument(
            "The peer (%d) for c_send_op must be non-negative.", peer));
    PADDLE_ENFORCE_GE(
        ring_id, 0,
        platform::errors::InvalidArgument(
            "The ring_id (%d) for c_send_op must be non-negative.", ring_id));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace());
  }
};

class CSendOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("X", "(Tensor) tensor to be sent.");
    AddAttr<int>("ring_id", "(int default 0) nccl communication ring id.")
        .SetDefault(0);
    AddAttr<int>("peer", "(int default 0) rank id for receiver.").SetDefault(0);
    AddAttr<bool>(
        "use_calc_stream",
        "(bool default false) eject CUDA operations to calculation stream.")
        .SetDefault(false);
    AddComment(R"DOC(
CSend Operator

Reference: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/p2p.html#sendrecv
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_WITHOUT_GRADIENT(c_send, ops::CSendOp, ops::CSendOpMaker);

REGISTER_OP_CPU_KERNEL(c_send, ops::CSendOpCPUKernel<float>,
                       ops::CSendOpCPUKernel<double>,
                       ops::CSendOpCPUKernel<int>,
                       ops::CSendOpCPUKernel<int64_t>,
                       ops::CSendOpCPUKernel<plat::float16>);
