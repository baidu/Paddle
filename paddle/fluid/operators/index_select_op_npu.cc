/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/index_select_op.h"
#include "paddle/fluid/operators/npu_op_runner.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class IndexSelectNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* index = ctx.Input<Tensor>("Index");
    auto dim = ctx.Attr<int>("dim");

    auto* out = ctx.Output<Tensor>("Out");
    out->mutable_data<T>(ctx.GetPlace());

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    NpuOpRunner runner;
    runner.SetType("GatherV2")
        .AddInput(*x)
        .AddInput(*index)
        .AddInput(std::vector<int32_t>{dim})
        .AddOutput(*out);
    runner.Run(stream);
  }
};

template <typename DeviceContext, typename T>
class IndexSelectGradNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x_grad = ctx.Output<framework::Tensor>(framework::GradVarName("X"));
    auto* index = ctx.Input<Tensor>("Index");
    auto* out_grad =
        ctx.Input<framework::Tensor>(framework::GradVarName("Out"));

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    auto x_dims = x_grad->dims();
    auto out_dims = out_grad->dims();

    int dim = ctx.Attr<int>("dim");
    if (dim < 0) {
      dim += out_dims.size();
    }

    std::vector<int> in_trans_perm;
    in_trans_perm.push_back(dim);
    for (int i = 0; i < out_dims.size(); ++i) {
      if (i == dim) continue;
      in_trans_perm.push_back(i);
    }
    framework::DDim transed_out_dims(out_dims);
    for (size_t i = 0; i < in_trans_perm.size(); ++i) {
      transed_out_dims[i] = out_dims[in_trans_perm[i]];
    }
    Tensor transed_out_grad;
    transed_out_grad.mutable_data<T>(transed_out_dims, ctx.GetPlace());
    framework::NPUAttributeMap in_trans_attr = {{"perm", in_trans_perm}};

    const auto& in_trans_runner = NpuOpRunner(
        "TransposeD", {*out_grad}, {transed_out_grad}, in_trans_attr);
    in_trans_runner.Run(stream);

    framework::DDim sum_dims(x_dims);
    sum_dims[0] = x_dims[dim];
    auto idx = 1;
    for (int i = 0; i < x_dims.size(); ++i) {
      if (i == dim) continue;
      sum_dims[idx++] = x_dims[i];
    }
    Tensor sum_out;
    sum_out.mutable_data<T>(sum_dims, ctx.GetPlace());

    const auto& zeros_runner = NpuOpRunner("ZerosLike", {sum_out}, {sum_out});
    zeros_runner.Run(stream);

    if (index->type() == framework::proto::VarType::INT32) {
      NpuOpRunner runner;
      runner.SetType("UnsortedSegmentSum")
          .AddInput(transed_out_grad)
          .AddInput(*index)
          .AddInput(std::vector<int64_t>{x_dims[dim]})
          .AddOutput(sum_out);
      runner.Run(stream);
    } else {
      Tensor casted_index;
      casted_index.mutable_data<int32_t>(index->dims(), ctx.GetPlace());
      const auto& cast_runner = NpuOpRunner("Cast", {*index}, {casted_index},
                                            {{"dst_type", ACL_INT32}});
      cast_runner.Run(stream);
      NpuOpRunner runner;
      runner.SetType("UnsortedSegmentSum")
          .AddInput(transed_out_grad)
          .AddInput(casted_index)
          .AddInput(std::vector<int64_t>{x_dims[dim]})
          .AddOutput(sum_out);
      runner.Run(stream);
    }

    std::vector<int> out_trans_perm;
    for (int i = 1; i < 1 + dim; ++i) {
      out_trans_perm.push_back(i);
    }
    out_trans_perm.push_back(0);
    for (int i = 1 + dim; i < x_dims.size(); ++i) {
      out_trans_perm.push_back(i);
    }
    framework::NPUAttributeMap out_trans_attr = {{"perm", out_trans_perm}};
    x_grad->mutable_data<T>(ctx.GetPlace());
    const auto& out_trans_runner =
        NpuOpRunner("TransposeD", {sum_out}, {*x_grad}, out_trans_attr);
    out_trans_runner.Run(stream);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_NPU_KERNEL(
    index_select,
    ops::IndexSelectNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::IndexSelectNPUKernel<paddle::platform::NPUDeviceContext, int>,
    ops::IndexSelectNPUKernel<paddle::platform::NPUDeviceContext, int64_t>);
REGISTER_OP_NPU_KERNEL(
    index_select_grad,
    ops::IndexSelectGradNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::IndexSelectGradNPUKernel<paddle::platform::NPUDeviceContext, int>,
    ops::IndexSelectGradNPUKernel<paddle::platform::NPUDeviceContext, int64_t>);
