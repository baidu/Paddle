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

#include "paddle/phi/kernels/adam_kernel.h"

#include "glog/logging.h"

#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/adam_functors.h"

namespace phi {

template <typename T, typename Context>
void AdamDenseKernel(
    const Context& dev_ctx,
    const DenseTensor& param,
    const DenseTensor& grad,
    const DenseTensor& learning_rate,
    const DenseTensor& moment1,
    const DenseTensor& moment2,
    const paddle::optional<DenseTensor>& moment2_max,  // UNUSED
    const DenseTensor& beta1_pow,
    const DenseTensor& beta2_pow,
    const paddle::optional<DenseTensor>& master_param,
    const paddle::optional<DenseTensor>& skip_update,
    const Scalar& beta1,
    const Scalar& beta2,
    const Scalar& epsilon,
    bool lazy_mode,
    int64_t min_row_size_to_use_multithread,
    bool multi_precision,
    bool use_global_beta_pow,
    bool amsgrad,  // UNUSED
    DenseTensor* param_out,
    DenseTensor* moment1_out,
    DenseTensor* moment2_out,
    DenseTensor* moment2_max_out,  // UNUSED
    DenseTensor* beta1_pow_out,
    DenseTensor* beta2_pow_out,
    DenseTensor* master_param_outs) {
  PADDLE_ENFORCE_NE(
      amsgrad,
      true,
      phi::errors::Unimplemented("Operation amsgrad is not supported yet."));

  xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
  float* param_ptr = nullptr;
  funcs::GetDataPointer<Context, float>(
      param, &param_ptr, dev_ctx, &RAII_GUARD);

  float* mom1_ptr = nullptr;
  funcs::GetDataPointer<Context, float>(
      moment1, &mom1_ptr, dev_ctx, &RAII_GUARD);

  float* mom2_ptr = nullptr;
  funcs::GetDataPointer<Context, float>(
      moment2, &mom2_ptr, dev_ctx, &RAII_GUARD);

  float* lr_ptr = nullptr;
  funcs::GetDataPointer<Context, float>(
      learning_rate, &lr_ptr, dev_ctx, &RAII_GUARD);

  float* beta1_pow_ptr = nullptr;
  const float* beta1_const_pow_ptr = nullptr;
  DenseTensor xpu_beta1_pow;
  if (beta1_pow.place() == CPUPlace()) {
    phi::Copy(dev_ctx, beta1_pow, dev_ctx.GetPlace(), false, &xpu_beta1_pow);
    if (xpu_beta1_pow.dtype() == DataType::FLOAT16)
      funcs::GetDataPointer<Context, float>(
          xpu_beta1_pow, &beta1_pow_ptr, dev_ctx, &RAII_GUARD);
    else
      beta1_const_pow_ptr = xpu_beta1_pow.template data<float>();
  } else {
    if (beta1_pow.dtype() == DataType::FLOAT16)
      funcs::GetDataPointer<Context, float>(
          beta1_pow, &beta1_pow_ptr, dev_ctx, &RAII_GUARD);
    else
      beta1_const_pow_ptr = beta1_pow.template data<float>();
  }

  float* beta2_pow_ptr = nullptr;
  const float* beta2_const_pow_ptr = nullptr;
  DenseTensor xpu_beta2_pow;
  if (beta2_pow.place() == CPUPlace()) {
    phi::Copy(dev_ctx, beta2_pow, dev_ctx.GetPlace(), false, &xpu_beta2_pow);
    if (xpu_beta2_pow.dtype() == DataType::FLOAT16)
      funcs::GetDataPointer<Context, float>(
          xpu_beta2_pow, &beta2_pow_ptr, dev_ctx, &RAII_GUARD);
    else
      beta2_const_pow_ptr = xpu_beta2_pow.template data<float>();
  } else {
    if (beta2_pow.dtype() == DataType::FLOAT16)
      funcs::GetDataPointer<Context, float>(
          beta2_pow, &beta2_pow_ptr, dev_ctx, &RAII_GUARD);
    else
      beta2_const_pow_ptr = beta2_pow.template data<float>();
  }

  DenseTensor xpu_param_out;
  float* param_out_ptr = nullptr;
  const phi::DenseTensorMeta meta_param(DataType::FLOAT32, param_out->dims());
  xpu_param_out.set_meta(meta_param);
  funcs::GetOutDataPointer<Context, float>(
      param_out, &xpu_param_out, &param_out_ptr, dev_ctx);

  DenseTensor xpu_mom1_out;
  float* mom1_out_ptr = nullptr;
  const phi::DenseTensorMeta meta_mom1(DataType::FLOAT32, moment1_out->dims());
  xpu_mom1_out.set_meta(meta_mom1);
  funcs::GetOutDataPointer<Context, float>(
      moment1_out, &xpu_mom1_out, &mom1_out_ptr, dev_ctx);

  DenseTensor xpu_mom2_out;
  float* mom2_out_ptr = nullptr;
  const phi::DenseTensorMeta meta_mom2(DataType::FLOAT32, moment2_out->dims());
  xpu_mom2_out.set_meta(meta_mom2);
  funcs::GetOutDataPointer<Context, float>(
      moment2_out, &xpu_mom2_out, &mom2_out_ptr, dev_ctx);

  bool skip_update_ = false;
  if (skip_update.is_initialized()) {
    PADDLE_ENFORCE_EQ(
        skip_update->numel(),
        1,
        errors::InvalidArgument("Input(SkipUpdate) size must be 1, but get %d",
                                skip_update->numel()));
    std::vector<bool> skip_update_vec;
    phi::TensorToVector(*skip_update, dev_ctx, &skip_update_vec);
    skip_update_ = skip_update_vec[0];
  }

  if (skip_update_) {
    VLOG(4) << "Adam skip update";
    phi::Copy(dev_ctx, param, dev_ctx.GetPlace(), false, param_out);
    phi::Copy(dev_ctx, moment1, dev_ctx.GetPlace(), false, moment1_out);
    phi::Copy(dev_ctx, moment2, dev_ctx.GetPlace(), false, moment2_out);
    if (!use_global_beta_pow) {
      phi::Copy(dev_ctx, beta1_pow, beta1_pow.place(), false, beta1_pow_out);
      phi::Copy(dev_ctx, beta2_pow, beta2_pow.place(), false, beta2_pow_out);
    }
    return;
  }

  PADDLE_ENFORCE_EQ(
      beta1_pow_out->numel(),
      1,
      errors::InvalidArgument("Tensor holds the wrong size, Expected beta1 pow "
                              "output size is 1, but received "
                              "value is:%d.",
                              beta1_pow_out->numel()));

  PADDLE_ENFORCE_EQ(
      beta2_pow_out->numel(),
      1,
      errors::InvalidArgument("Tensor holds the wrong size, Expected beta2 pow "
                              "output size is 1, but received "
                              "value is:%d.",
                              beta2_pow_out->numel()));

  VLOG(4) << "use_global_beta_pow:" << use_global_beta_pow;

  auto beta1_ = beta1.to<float>();
  auto beta2_ = beta2.to<float>();
  auto epsilon_ = epsilon.to<float>();

  float* grad_c = nullptr;
  funcs::GetDataPointer<Context, float>(grad, &grad_c, dev_ctx, &RAII_GUARD);

  int r = xpu::adam(
      dev_ctx.x_context(),
      grad_c != nullptr ? grad_c : grad.template data<float>(),
      mom1_ptr != nullptr ? mom1_ptr : moment1.template data<float>(),
      mom2_ptr != nullptr ? mom2_ptr : moment2.template data<float>(),
      param_ptr != nullptr ? param_ptr : param.template data<float>(),
      beta1_pow_ptr != nullptr ? beta1_pow_ptr : beta1_const_pow_ptr,
      beta2_pow_ptr != nullptr ? beta2_pow_ptr : beta2_const_pow_ptr,
      lr_ptr != nullptr ? lr_ptr : learning_rate.template data<float>(),
      mom1_out_ptr,
      mom2_out_ptr,
      param_out_ptr,
      beta1_,
      beta2_,
      epsilon_,
      param.numel());

  PADDLE_ENFORCE_XDNN_SUCCESS(r, "adam");

  funcs::CopyOutData<Context, float>(
      xpu_mom1_out, moment1_out, dev_ctx, &RAII_GUARD);
  funcs::CopyOutData<Context, float>(
      xpu_mom2_out, moment2_out, dev_ctx, &RAII_GUARD);
  funcs::CopyOutData<Context, float>(
      xpu_param_out, param_out, dev_ctx, &RAII_GUARD);

  if (!use_global_beta_pow) {
    // update in cpu and then copy to xpu
    if (beta1_pow.place() == CPUPlace() && beta2_pow.place() == CPUPlace()) {
      funcs::SetBetaData<Context, float>(
          beta1_pow, beta1_pow_out, beta1_, dev_ctx);

      funcs::SetBetaData<Context, float>(
          beta2_pow, beta2_pow_out, beta2_, dev_ctx);
    } else {
      float* beta1_pow_out_p1 = nullptr;

      if (beta1_pow_out->dtype() == DataType::FLOAT16) {
        funcs::Scale<Context, float>(beta1_pow_out,
                                     beta1_pow,
                                     beta1_pow_ptr,
                                     beta1_,
                                     dev_ctx,
                                     &RAII_GUARD);
      } else {
        const float* beta1_pow_data = beta1_pow.template data<float>();
        beta1_pow_out_p1 = dev_ctx.template Alloc<float>(beta1_pow_out);
        r = xpu::scale(dev_ctx.x_context(),
                       beta1_pow_data,
                       beta1_pow_out_p1,
                       beta1_pow.numel(),
                       false,
                       beta1_,
                       0.0f);
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "adam");
      }

      float* beta2_pow_out_p1 = nullptr;
      if (beta2_pow_out->dtype() == DataType::FLOAT16) {
        funcs::Scale<Context, float>(beta2_pow_out,
                                     beta2_pow,
                                     beta2_pow_ptr,
                                     beta2_,
                                     dev_ctx,
                                     &RAII_GUARD);
      } else {
        const float* beta2_pow_data = beta2_pow.template data<float>();
        beta2_pow_out_p1 = dev_ctx.template Alloc<float>(beta2_pow_out);
        r = xpu::scale(dev_ctx.x_context(),
                       beta2_pow_data,
                       beta2_pow_out_p1,
                       beta2_pow.numel(),
                       false,
                       beta2_,
                       0.0f);
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "adam");
      }
    }
  }
}

template <typename T, typename Context>
void MergedAdamKernel(
    const Context& dev_ctx,
    const std::vector<const DenseTensor*>& param,
    const std::vector<const DenseTensor*>& grad,
    const std::vector<const DenseTensor*>& learning_rate,
    const std::vector<const DenseTensor*>& moment1,
    const std::vector<const DenseTensor*>& moment2,
    const paddle::optional<std::vector<const DenseTensor*>>&
        moment2_max,  // UNUSED
    const std::vector<const DenseTensor*>& beta1_pow,
    const std::vector<const DenseTensor*>& beta2_pow,
    const paddle::optional<std::vector<const DenseTensor*>>& master_param,
    const Scalar& beta1,
    const Scalar& beta2,
    const Scalar& epsilon,
    bool multi_precision,
    bool use_global_beta_pow,
    bool amsgrad,  // UNUSED
    std::vector<DenseTensor*> param_out,
    std::vector<DenseTensor*> moment1_out,
    std::vector<DenseTensor*> moment2_out,
    std::vector<DenseTensor*> moment2_max_out,  // UNUSED
    std::vector<DenseTensor*> beta1_pow_out,
    std::vector<DenseTensor*> beta2_pow_out,
    std::vector<DenseTensor*> master_param_out) {
  PADDLE_ENFORCE_NE(
      amsgrad,
      true,
      phi::errors::Unimplemented("Operation amsgrad is not supported yet."));

  VLOG(4) << "use_global_beta_pow:" << use_global_beta_pow;

  auto beta1_ = beta1.to<float>();
  auto beta2_ = beta2.to<float>();
  auto epsilon_ = epsilon.to<float>();
  int64_t step_ = 0;
  int64_t mode_ = 2;
  int64_t bias_correction_ = 1;
  float weight_decay_ = 0.0;

  DenseTensor lr_host;
  lr_host.Resize(learning_rate[0]->dims());
  dev_ctx.template HostAlloc<float>(&lr_host);
  phi::Copy(dev_ctx, *learning_rate[0], CPUPlace(), false, &lr_host);
  float lr_ = *(lr_host.template data<float>());

  float beta1_pow_data;
  if (beta1_pow[0]->place() == CPUPlace()) {
    beta1_pow_data = *(beta1_pow[0]->data<float>());
  } else {
    DenseTensor beta1_pow_host;
    beta1_pow_host.Resize(beta1_pow[0]->dims());
    dev_ctx.template HostAlloc<float>(&beta1_pow_host);
    phi::Copy(dev_ctx, *beta1_pow[0], CPUPlace(), false, &beta1_pow_host);
    beta1_pow_data = *(beta1_pow_host.template data<float>());
  }

  float beta2_pow_data;
  if (beta2_pow[0]->place() == CPUPlace()) {
    beta2_pow_data = *(beta2_pow[0]->data<float>());
  } else {
    DenseTensor beta2_pow_host;
    beta2_pow_host.Resize(beta2_pow[0]->dims());
    dev_ctx.template HostAlloc<float>(&beta2_pow_host);
    phi::Copy(dev_ctx, *beta2_pow[0], CPUPlace(), false, &beta2_pow_host);
    beta2_pow_data = *(beta2_pow_host.template data<float>());
  }

  int param_num = param.size();
  PADDLE_ENFORCE_EQ(param_num,
                    param_out.size(),
                    errors::InvalidArgument(
                        "The size of Output(ParamOut) must be equal to "
                        "Input(Param), but got the size of Output(ParamOut) "
                        "is %d, the size of Input(Param) is %d.",
                        param_out.size(),
                        param_num));
  PADDLE_ENFORCE_EQ(
      param_num,
      moment1_out.size(),
      errors::InvalidArgument(
          "The size of Input(Moment1) must be equal to Input(Param), but got "
          "the size of Input(Moment1) is %d, the size of Input(Param) is %d.",
          moment1.size(),
          param_num));
  PADDLE_ENFORCE_EQ(
      param_num,
      moment2_out.size(),
      errors::InvalidArgument(
          "The size of Input(Moment1) must be equal to Input(Param), but got "
          "the size of Input(Moment1) is %d, the size of Input(Param) is %d.",
          moment2.size(),
          param_num));
  PADDLE_ENFORCE_EQ(param_num,
                    beta1_pow_out.size(),
                    errors::InvalidArgument(
                        "The size of Output(Beta1PowOut) must be equal to "
                        "Input(Param), but got the size of Output(Beta1PowOut) "
                        "is %d, the size of Input(Param) is %d.",
                        beta1_pow_out.size(),
                        param_num));
  PADDLE_ENFORCE_EQ(param_num,
                    beta2_pow_out.size(),
                    errors::InvalidArgument(
                        "The size of Output(Beta2PowOut) must be equal to "
                        "Input(Param), but got the size of Output(Beta2PowOut) "
                        "is %d, the size of Input(Param) is %d.",
                        beta2_pow_out.size(),
                        param_num));
  PADDLE_ENFORCE_EQ(
      param_num,
      grad.size(),
      errors::InvalidArgument(
          "The size of Input(Grad) must be equal to Input(Param), but got "
          "the size of Input(Grad) is %d, the size of Input(Param) is %d.",
          grad.size(),
          param_num));
  PADDLE_ENFORCE_EQ(
      param_num,
      moment1.size(),
      errors::InvalidArgument(
          "The size of Input(Moment1) must be equal to Input(Param), but got "
          "the size of Input(Moment1) is %d, the size of Input(Param) is %d.",
          moment1.size(),
          param_num));
  PADDLE_ENFORCE_EQ(
      param_num,
      moment2.size(),
      errors::InvalidArgument(
          "The size of Input(Moment1) must be equal to Input(Param), but got "
          "the size of Input(Moment1) is %d, the size of Input(Param) is %d.",
          moment2.size(),
          param_num));

  std::vector<float*> param_list(param_num);
  std::vector<float*> grad_list(param_num);
  std::vector<float*> moment1_list(param_num);
  std::vector<float*> moment2_list(param_num);
  std::vector<int64_t> shape_list(param_num);

  for (int j = 0; j < param_num; j++) {
    param_list[j] = const_cast<float*>(param[j]->data<float>());
    grad_list[j] = const_cast<float*>(grad[j]->data<float>());
    moment1_list[j] = const_cast<float*>(moment1[j]->data<float>());
    moment2_list[j] = const_cast<float*>(moment2[j]->data<float>());
    shape_list[j] = param[j]->numel();

    PADDLE_ENFORCE_EQ(
        param[j],
        param_out[j],
        errors::InvalidArgument("The size of Input(Param) and Output(ParamOut) "
                                "must be the same Tensors."));
    PADDLE_ENFORCE_EQ(
        moment1[j],
        moment1_out[j],
        errors::InvalidArgument("The size of Input(Param) and Output(ParamOut) "
                                "must be the same Tensors."));
    PADDLE_ENFORCE_EQ(
        moment2[j],
        moment2_out[j],
        errors::InvalidArgument("The size of Input(Param) and Output(ParamOut) "
                                "must be the same Tensors."));

    dev_ctx.template Alloc<float>(param_out[j]);
    dev_ctx.template Alloc<float>(moment1_out[j]);
    dev_ctx.template Alloc<float>(moment2_out[j]);
  }

  int r = xpu::multi_tensor_adam(dev_ctx.x_context(),
                                 grad_list,
                                 param_list,
                                 moment1_list,
                                 moment2_list,
                                 shape_list,
                                 lr_,
                                 beta1_,
                                 beta2_,
                                 epsilon_,
                                 step_,
                                 mode_,
                                 bias_correction_,
                                 weight_decay_,
                                 beta1_pow_data,
                                 beta2_pow_data);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "merged_adam");

  // update param, moment1, moment2
  for (int i = 0; i < param_num; i++) {
    phi::Copy(dev_ctx, *param[i], dev_ctx.GetPlace(), false, param_out[i]);
    phi::Copy(dev_ctx, *moment1[i], dev_ctx.GetPlace(), false, moment1_out[i]);
    phi::Copy(dev_ctx, *moment2[i], dev_ctx.GetPlace(), false, moment2_out[i]);
  }

  if (!use_global_beta_pow) {
    for (int i = 0; i < param_num; i++) {
      if (beta1_pow[i]->place() == CPUPlace() &&
          beta2_pow[i]->place() == CPUPlace()) {
        funcs::SetBetaData<Context, float>(
            *beta1_pow[i], beta1_pow_out[i], beta1_, dev_ctx);

        funcs::SetBetaData<Context, float>(
            *beta2_pow[i], beta2_pow_out[i], beta2_, dev_ctx);
      } else {
        float* beta1_pow_out_ptr = nullptr;
        const float* beta1_pow_data = beta1_pow[i]->data<float>();
        beta1_pow_out_ptr = dev_ctx.template Alloc<float>(beta1_pow_out[i]);
        r = xpu::scale(dev_ctx.x_context(),
                       beta1_pow_data,
                       beta1_pow_out_ptr,
                       beta1_pow[i]->numel(),
                       false,
                       beta1_,
                       0.0f);
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "merged_adam");

        float* beta2_pow_out_ptr = nullptr;
        const float* beta2_pow_data = beta2_pow[i]->data<float>();
        beta2_pow_out_ptr = dev_ctx.template Alloc<float>(beta2_pow_out[i]);
        r = xpu::scale(dev_ctx.x_context(),
                       beta2_pow_data,
                       beta2_pow_out_ptr,
                       beta2_pow[i]->numel(),
                       false,
                       beta2_,
                       0.0f);
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "merged_adam");
      }
    }
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(
    adam, XPU, ALL_LAYOUT, phi::AdamDenseKernel, float, phi::dtype::float16) {
  // Skip beta1_pow, beta2_pow, skip_update data transform
  kernel->InputAt(6).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(7).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(9).SetBackend(phi::Backend::ALL_BACKEND);

  kernel->OutputAt(4).SetBackend(phi::Backend::UNDEFINED);
  kernel->OutputAt(5).SetBackend(phi::Backend::UNDEFINED);
}

PD_REGISTER_KERNEL(merged_adam, XPU, ALL_LAYOUT, phi::MergedAdamKernel, float) {
  // Skip beta1_pow, beta2_pow data transform
  kernel->InputAt(6).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(7).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->OutputAt(4).SetBackend(phi::Backend::UNDEFINED);
  kernel->OutputAt(5).SetBackend(phi::Backend::UNDEFINED);
}
