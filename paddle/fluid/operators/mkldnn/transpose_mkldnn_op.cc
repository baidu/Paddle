/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include "paddle/fluid/framework/data_layout_transform.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/memory/malloc.h"
#include "paddle/fluid/platform/mkldnn_reuse.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using framework::DataLayout;

template <typename T>
class TransposeMKLDNNOpKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(paddle::platform::is_cpu_place(ctx.GetPlace()),
                   "It must use CPUPlace.");
    auto& dev_ctx =
        ctx.template device_context<paddle::platform::MKLDNNDeviceContext>();
    const auto& mkldnn_engine = dev_ctx.GetEngine();
    std::vector<int> axis = ctx.Attr<std::vector<int>>("axis");
    int ndims = axis.size();
    auto* input = ctx.Input<Tensor>("X");
    auto* output = ctx.Output<Tensor>("Out");
    const T* input_data = input->data<T>();

    if (ndims == 1) {
      output->ShareDataWith(*input);
      return;
    }

    auto nchw_tz = paddle::framework::vectorize<int>(input->dims());

    const std::string key =
        platform::CreateKey(nchw_tz, axis, ctx.op().Output("Out") +
                                               std::to_string(input->format()));

    platform::TransposeMKLDNNHandler handler(nchw_tz, axis, dev_ctx,
                                             mkldnn_engine, key);

    auto transpose_src_memory_p = handler.AcquireSrcMemory(
        input->format(), platform::to_void_cast<T>(input_data));
    auto transpose_dst_memory_p =
        handler.AcquireDstMemory(output, ctx.GetPlace());
    auto transpose_p = handler.AcquireTranspose(transpose_dst_memory_p,
                                                transpose_src_memory_p);

    std::vector<mkldnn::primitive> pipeline;
    pipeline.push_back(*transpose_p);
    mkldnn::stream(mkldnn::stream::kind::eager).submit(pipeline).wait();

    output->set_layout(DataLayout::kNCHW);
    output->set_format(MKLDNNMemoryFormat::format_undef);
  }
};

template <typename T>
class TransposeMKLDNNGradOpKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(paddle::platform::is_cpu_place(ctx.GetPlace()),
                   "It must use CPUPlace.");
    auto* out_grad =
        ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto* x_grad = ctx.Output<framework::Tensor>(framework::GradVarName("X"));
    if (!x_grad) return;

    auto& dev_ctx =
        ctx.template device_context<paddle::platform::MKLDNNDeviceContext>();
    const auto& mkldnn_engine = dev_ctx.GetEngine();
    std::vector<int> axis = ctx.Attr<std::vector<int>>("axis");
    std::vector<int> reversed_axis(axis);
    int ndims = axis.size();
    if (ndims == 1) {
      x_grad->ShareDataWith(*out_grad);
      return;
    }

    for (size_t i = 0; i < axis.size(); i++) {
      reversed_axis[axis[i]] = i;
    }

    const T* out_grad_data = out_grad->data<T>();
    x_grad->mutable_data<T>(ctx.GetPlace());

    auto nchw_tz = paddle::framework::vectorize<int>(out_grad->dims());

    const std::string key = platform::CreateKey(
        nchw_tz, axis, ctx.op().Output(framework::GradVarName("X")));

    platform::TransposeMKLDNNHandler handler(nchw_tz, reversed_axis, dev_ctx,
                                             mkldnn_engine, key);

    auto transpose_src_memory_p = handler.AcquireSrcMemory(
        out_grad->format(), platform::to_void_cast<T>(out_grad_data));
    auto transpose_dst_memory_p =
        handler.AcquireDstMemory(x_grad, ctx.GetPlace());
    auto transpose_p = handler.AcquireTranspose(transpose_dst_memory_p,
                                                transpose_src_memory_p);

    std::vector<mkldnn::primitive> pipeline;
    pipeline.push_back(*transpose_p);
    mkldnn::stream(mkldnn::stream::kind::eager).submit(pipeline).wait();
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_KERNEL(transpose2, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::TransposeMKLDNNOpKernel<float>);

REGISTER_OP_KERNEL(transpose, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::TransposeMKLDNNOpKernel<float>);

REGISTER_OP_KERNEL(transpose_grad, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::TransposeMKLDNNGradOpKernel<float>);
REGISTER_OP_KERNEL(transpose2_grad, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::TransposeMKLDNNGradOpKernel<float>);
