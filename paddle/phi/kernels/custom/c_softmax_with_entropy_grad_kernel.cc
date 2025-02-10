// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/api/backward/backward_api.h"
#include "paddle/phi/api/include/api.h"
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/backends/device_manager.h"
#include "paddle/phi/core/distributed/collective/process_group.h"
#include "paddle/phi/core/distributed/comm_context_manager.h"
#include "paddle/phi/core/distributed/xccl_comm_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/axis_utils.h"
#ifdef PADDLE_WITH_CUSTOM_DEVICE
namespace phi {

template <typename T, typename Context>
void CSoftmaxWithEntropyGradKernel(const Context& dev_ctx,
                                   const DenseTensor& softmax_in,
                                   const DenseTensor& label_in,
                                   const DenseTensor& loss_grad_in,
                                   int64_t ignore_index,
                                   int ring_id,
                                   int rank,
                                   int nranks,
                                   DenseTensor* logits_grad) {
  const phi::DenseTensor* labels = &label_in;
  const phi::DenseTensor* loss_grad = &loss_grad_in;
  const phi::DenseTensor* softmax = &softmax_in;
  phi::DenseTensor* logit_grad = logits_grad;

  if (logit_grad != softmax) {
    phi::Copy(dev_ctx, *softmax, dev_ctx.GetPlace(), false, logit_grad);
  }
  const auto softmax_dims = softmax->dims();
  const int axis = softmax_dims.size() - 1;
  const int N = phi::funcs::SizeToAxis(axis, softmax_dims);
  const int D = phi::funcs::SizeFromAxis(axis, softmax_dims);
  const auto& label_type = labels->dtype();

  if (label_type == phi::DataType::INT32 ||
      label_type == phi::DataType::INT64) {
    auto logit_grad_t = std::make_shared<phi::DenseTensor>();
    logit_grad_t->ShareDataWith(*logit_grad).Resize({N, D});
    auto loss_grad_t = std::make_shared<phi::DenseTensor>();
    loss_grad_t->ShareDataWith(*loss_grad).Resize({N});
    auto labels_1d = std::make_shared<phi::DenseTensor>();
    labels_1d->ShareDataWith(*labels).Resize({N});
    paddle::Tensor logits_grad_tensor(logit_grad_t),
        loss_grad_tensor(loss_grad_t), labels_1d_tensor(labels_1d);

    auto labels_1d_not_equal_ignore = paddle::experimental::reshape(
        paddle::experimental::not_equal(
            labels_1d_tensor,
            paddle::experimental::full_like(labels_1d_tensor,
                                            ignore_index,
                                            labels_1d_tensor.dtype(),
                                            labels_1d_tensor.place())),
        {N, 1});
    auto start_index_tensor =
        paddle::experimental::full_like(labels_1d_tensor,
                                        rank * D,
                                        labels_1d_tensor.dtype(),
                                        labels_1d_tensor.place());

    auto logits_grad_out_tensor1 = paddle::experimental::subtract(
        paddle::experimental::multiply(
            logits_grad_tensor,
            paddle::experimental::cast(labels_1d_not_equal_ignore,
                                       logits_grad_tensor.dtype())),
        paddle::experimental::cast(
            paddle::experimental::one_hot(
                paddle::experimental::subtract(labels_1d_tensor,
                                               start_index_tensor),
                D),
            logits_grad_tensor.dtype()));

    auto logits_grad_out_tensor2 = paddle::experimental::multiply(
        logits_grad_out_tensor1,
        paddle::experimental::reshape(loss_grad_tensor, {N, 1}));
    logit_grad
        ->ShareDataWith(*reinterpret_cast<phi::DenseTensor*>(
            logits_grad_out_tensor2.impl().get()))
        .Resize(softmax_dims);
  } else {
    PADDLE_THROW(common::errors::Unavailable(
        "CustomDevice c_softmax_with_cross_entropy_grad "
        "label_type only support int32/int64"));
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(c_softmax_with_cross_entropy_grad,
                   Custom,
                   ALL_LAYOUT,
                   phi::CSoftmaxWithEntropyGradKernel,
                   float,
                   double,
                   phi::dtype::float16) {}
#endif
