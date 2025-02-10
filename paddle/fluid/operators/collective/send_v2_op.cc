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

#include "paddle/fluid/operators/collective/send_v2_op.h"

namespace paddle::operators {

class SendOpV2 : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "SendV2");
    int peer = ctx->Attrs().Get<int>("peer");
    int ring_id = ctx->Attrs().Get<int>("ring_id");
    PADDLE_ENFORCE_GE(
        peer,
        0,
        common::errors::InvalidArgument(
            "The peer (%d) for send_v2 op must be non-negative.", peer));
    PADDLE_ENFORCE_GE(
        ring_id,
        0,
        common::errors::InvalidArgument(
            "The ring_id (%d) for send_v2 op must be non-negative.", ring_id));
  }

 protected:
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    const framework::Variable* var = ctx.InputVar("X");
    if (var->IsType<phi::TensorArray>()) {
      auto t_arr = var->Get<phi::TensorArray>();
      // NOTE(sandyhouse): Support an empty tensor array as Input.
      // And set the kernel type is float.
      if (t_arr.empty()) {
        return phi::KernelKey(framework::proto::VarType::FP32, ctx.GetPlace());
      }
    }
    return phi::KernelKey(OperatorWithKernel::IndicateVarDataType(ctx, "X"),
                          ctx.GetPlace());
  }
};

class SendOpV2Maker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor) tensor to be sent.");
    AddAttr<int>("ring_id", "(int default 0) nccl communication ring id.")
        .SetDefault(0);
    AddAttr<int>("peer", "(int default 0) rank id for receiver.").SetDefault(0);

    AddAttr<bool>(
        "use_calc_stream",
        "(bool default false) eject CUDA operations to calculation stream.")
        .SetDefault(false);
    AddAttr<bool>(
        "dynamic_shape",
        "(bool default false) the send/recv will be done with dynamic shape.")
        .SetDefault(false);
    AddComment(R"DOC(
Send Operator

Reference: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/p2p.html#sendrecv
)DOC");
  }
};

}  // namespace paddle::operators

namespace ops = paddle::operators;

REGISTER_OP_WITHOUT_GRADIENT(send_v2, ops::SendOpV2, ops::SendOpV2Maker);
