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

#include "paddle/phi/kernels/dgc_kernel.h"

#include <glog/logging.h>

#include "dgc/dgc.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/broadcast_function.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"
#include "paddle/phi/kernels/funcs/elementwise_functor.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

inline float get_period_sparcity(const std::vector<float>& sparsity,
                                 float cur_step,
                                 float rampup_steps) {
  PADDLE_ENFORCE_GE(
      static_cast<int>(cur_step),
      0,
      common::errors::InvalidArgument("DGC current step=%d, but it must >= 0, "
                                      "please submit issue in github",
                                      static_cast<int>(cur_step)));

  size_t idx = static_cast<int>(cur_step * sparsity.size() / rampup_steps);
  if (idx >= sparsity.size()) {
    idx = sparsity.size() - 1;
  }

  PADDLE_ENFORCE_LT(
      idx,
      sparsity.size(),
      common::errors::OutOfRange(
          "sparsity index out of bounds. idx=%d >= sparsity.size=%d",
          idx,
          sparsity.size()));
  return sparsity[idx];
}

template <typename T, typename Context>
void DGCKernel(const Context& dev_ctx,
               const DenseTensor& u,
               const DenseTensor& v,
               const DenseTensor& grad,
               const DenseTensor& param,
               const DenseTensor& current_step_tensor,
               const DenseTensor& nranks_tensor,
               float m,
               bool use_nesterov,
               const std::vector<float>& sparsity,
               float rampup_begin_step,
               float rampup_step,
               float regular_coeff,
               int regular_type,
               DenseTensor* u_out,
               DenseTensor* v_out,
               DenseTensor* encode_grad_out,
               DenseTensor* grad_out,
               DenseTensor* k_out,
               DenseTensor* gather_buff) {
  // nranks
  const int nranks = static_cast<int>(*nranks_tensor.data<float>());
  PADDLE_ENFORCE_GT(nranks,
                    1,
                    common::errors::PreconditionNotMet(
                        "DGC is not useful when num_trainers <= 1. Please "
                        "use multi card or multi machine GPU"));

  auto param_e = phi::EigenVector<T>::Flatten(param);
  auto grad_e = phi::EigenVector<T>::Flatten(grad);
  auto grad_out_e = phi::EigenVector<T>::Flatten(*grad_out);

  auto& eigen_ctx = *dev_ctx.eigen_device();

  // NOTE. In paddle, loss has divided by nranks. Because dgc_op is before
  // allreduce, so local regular_coeff need div nranks too. But now we
  // multi grad with nranks in dgc_op, in that case regular_coeff don't
  // need to /nranks, can prevent precision loss. For coeff often equal
  // with 1e-4, if nranks=32, coeff/nranks will be 3.125e-6, the numerical
  // accuracy of coeff/nranks will be too low.
  PADDLE_ENFORCE_EQ(regular_type >= 0 && regular_type <= 2,
                    true,
                    common::errors::InvalidArgument(
                        "DGC only support one of None|L1Decay|L2Decay "
                        "Regularization for now."));
  if (regular_type == 0) {
    grad_out_e.device(eigen_ctx) = (1.0 * nranks) * grad_e;
  } else if (regular_type == 1) {
    // L1Decay. grad = grad + coeff * sign(param)
    grad_out_e.device(eigen_ctx) =
        (1.0 * nranks) * grad_e + regular_coeff * param_e.sign();
  } else if (regular_type == 2) {
    // L2Decay. grad = grad + coeff * param
    grad_out_e.device(eigen_ctx) =
        (1.0 * nranks) * grad_e + regular_coeff * param_e;
  }

  // current step
  const float* current_step = current_step_tensor.data<float>();

  if (static_cast<int>(*current_step) < static_cast<int>(rampup_begin_step)) {
    VLOG(10) << "current_step:" << *current_step
             << " < rampup_begin_step:" << rampup_begin_step
             << " so doesn't use dgc";
    return;
  }

  float ratio = 1 - get_period_sparcity(
                        sparsity,
                        static_cast<float>(*current_step - rampup_begin_step),
                        rampup_step);
  PADDLE_ENFORCE_GE(
      ratio,
      0.0,
      common::errors::InvalidArgument("DGC sparsity ratio must >= 0"));
  PADDLE_ENFORCE_LT(
      ratio,
      1.0,
      common::errors::InvalidArgument("DGC sparsity ratio must < 1"));
  int k = static_cast<int>(grad.numel() * ratio);

  VLOG(10) << "m:" << m << ", use_nesterov:" << use_nesterov
           << ", rampup_begin_step:" << rampup_begin_step
           << ", rampup_step:" << rampup_step
           << ",  current_step:" << *current_step << ", ratio:" << ratio
           << ", k:" << k << ", nranks:" << nranks;

  T* k_out_data = k_out->data<T>();
  *k_out_data = k;

  // FIXME(gongwb): use cublas.
  auto u_out_e = phi::EigenVector<T>::Flatten(*u_out);
  auto u_e = phi::EigenVector<T>::Flatten(u);

  // calc local momentum from global momentum
  // NOTE. If grad not multi nranks, need add below code.
  // if (static_cast<int>(*current_step) ==
  //     static_cast<int>(rampup_begin_step)) {
  //   u_out_e.device(eigen_ctx) = (1.0 / nranks) * u_e;
  // }

  if (use_nesterov) {
    // u = m * (u + grad)
    u_out_e.device(eigen_ctx) = m * (u_e + grad_out_e);

    // v = u + v + grad
    dev_ctx.template Alloc<T>(v_out);
    phi::funcs::ElementwiseCompute<phi::funcs::AddFunctor<T>, T>(
        dev_ctx, u, v, phi::funcs::AddFunctor<T>(), v_out, 0);

    phi::funcs::ElementwiseCompute<phi::funcs::AddFunctor<T>, T>(
        dev_ctx, grad, v, phi::funcs::AddFunctor<T>(), v_out, 0);
  } else {
    // u = m * u + grad
    u_out_e.device(eigen_ctx) = m * u_e + grad_out_e;

    // v = u + v
    dev_ctx.template Alloc<T>(v_out);
    phi::funcs::ElementwiseCompute<phi::funcs::AddFunctor<T>, T>(
        dev_ctx, u, v, phi::funcs::AddFunctor<T>(), v_out, 0);
  }

  T* v_out_data = dev_ctx.template Alloc<T>(v_out);
  T* u_out_data = dev_ctx.template Alloc<T>(u_out);

  encode_grad_out->Resize(phi::DDim{2 * k});
  T* encode_grad_out_data = dev_ctx.template Alloc<T>(encode_grad_out);

  gather_buff->Resize(phi::DDim{2 * k * nranks});
  dev_ctx.template Alloc<T>(gather_buff);

  int buf_size = paddle::communication::dgc::get_buffer_size(k);
  phi::Allocator::AllocationPtr tmp_ious_data;
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  if (dev_ctx.GetPlace().GetType() == phi::AllocationType::GPU) {
    tmp_ious_data = phi::memory_utils::Alloc(
        dev_ctx.GetPlace(),
        buf_size,
        phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));
  }
#endif
  if (dev_ctx.GetPlace().GetType() == phi::AllocationType::CPU) {
    tmp_ious_data = phi::memory_utils::Alloc(dev_ctx.GetPlace(), buf_size);
  }

  void* buf = reinterpret_cast<void*>(tmp_ious_data->ptr());

  if (!paddle::communication::dgc::k_select(
          static_cast<void*>(encode_grad_out_data),
          k,
          v_out_data,
          static_cast<int>(v_out->numel()),
          buf,
          dev_ctx.stream(),
          u_out_data)) {
    // TODO(weihang): owner should polish this error message
    PADDLE_THROW(common::errors::InvalidArgument(
        "V_out numel error, V_out numel is %d.", v_out->numel()));
  }

  phi::funcs::SetConstant<Context, T> tset;
  tset(dev_ctx, grad_out, static_cast<T>(0));
}

}  // namespace phi

PD_REGISTER_KERNEL(dgc, GPU, ALL_LAYOUT, phi::DGCKernel, float) {}
