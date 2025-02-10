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

#include "paddle/fluid/operators/collective/recv_v2_op.h"

#include <string>

namespace paddle::operators {

class RecvOpV2 : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "Recv_V2");
    int peer = ctx->Attrs().Get<int>("peer");
    int ring_id = ctx->Attrs().Get<int>("ring_id");
    PADDLE_ENFORCE_GE(
        peer,
        0,
        common::errors::InvalidArgument(
            "The peer (%d) for recv_v2 op must be non-negative.", peer));
    PADDLE_ENFORCE_GE(
        ring_id,
        0,
        common::errors::InvalidArgument(
            "The ring_id (%d) for recv_v2 op must be non-negative.", ring_id));

    if (ctx->GetOutputsVarType("Out").front() ==
        framework::proto::VarType::DENSE_TENSOR) {
      auto out_shape = ctx->Attrs().Get<std::vector<int>>("out_shape");
      PADDLE_ENFORCE_GE(
          out_shape.size(),
          1,
          common::errors::InvalidArgument(
              "The size of the output shape must be greater than 0 "
              "but the value given is %d.",
              out_shape.size()));
      bool dynamic_shape = ctx->Attrs().Get<bool>("dynamic_shape");
      if (!dynamic_shape) {
        // No need to check out shape if with dynamic_shape,
        // since the shape will be recv from send_v2
        for (size_t i = 0; i < out_shape.size(); ++i) {
          PADDLE_ENFORCE_GE(out_shape[i],
                            1,
                            common::errors::InvalidArgument(
                                "The shape attribute for recv_v2 must be set "
                                "explicitly, but the %dth element is %d which "
                                "is less than 1. Or dynamic_shape should be "
                                "set to True for both send_v2 and recv_v2.",
                                i,
                                out_shape[i]));
        }
        ctx->SetOutputDim("Out", common::make_ddim(out_shape));
      }
    }
  }

 protected:
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    int dtype = ctx.Attr<int>("dtype");
    framework::proto::VarType::Type type =
        framework::proto::VarType::Type(dtype);
    return phi::KernelKey(type, ctx.GetPlace());
  }
};

class RecvOpV2Maker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddOutput("Out", "(Tensor) tensor to receive.");
    AddAttr<int>("ring_id", "(int default 0) nccl communication ring id.")
        .SetDefault(0);
    AddAttr<int>("peer", "(int default 0) rank id for sender.").SetDefault(0);
    AddAttr<int>("dtype", "(int default 5('float32')) data type of tensor.")
        .SetDefault(5);

    AddAttr<std::vector<int>>("out_shape", "shape of the output tensor.")
        .SetDefault(std::vector<int>());
    AddAttr<bool>(
        "use_calc_stream",
        "(bool default false) eject CUDA operations to calculation stream.")
        .SetDefault(false);
    AddAttr<bool>(
        "dynamic_shape",
        "(bool default false) the send/recv will be done with dynamic shape.")
        .SetDefault(false);
    AddComment(R"DOC(
Recv Operator

Reference: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/p2p.html#sendrecv
)DOC");
  }
};

}  // namespace paddle::operators

namespace ops = paddle::operators;

REGISTER_OP_WITHOUT_GRADIENT(recv_v2, ops::RecvOpV2, ops::RecvOpV2Maker);
