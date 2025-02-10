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

#include "paddle/phi/kernels/adamw_kernel.h"

#include <vector>

#include "glog/logging.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"

#include "paddle/phi/common/amp_type_traits.h"

namespace phi {

template <typename Context>
float GetAbsMax(const Context& dev_ctx,
                const float* input,
                float* buffer_xpu,
                int64_t numel) {
  int max_ptr_size = phi::backends::xpu::get_xpu_max_ptr_size(-1);
  std::vector<float> buffer_cpu(max_ptr_size);
  // int findmax(Context* ctx, const T* x, float* maxptr, int64_t len);
  int r = xpu::findmax<float>(dev_ctx.x_context(), input, buffer_xpu, numel);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "findmax");
  memory_utils::Copy(CPUPlace(),
                     static_cast<void*>(buffer_cpu.data()),
                     dev_ctx.GetPlace(),
                     static_cast<void*>(buffer_xpu),
                     sizeof(float) * max_ptr_size);
  return *std::max_element(buffer_cpu.begin(), buffer_cpu.end());
}

template <typename T, typename Context>
void AdamwDenseKernelKL3(const Context& dev_ctx,
                         const DenseTensor& param,
                         const DenseTensor& grad,
                         const DenseTensor& learning_rate,
                         const DenseTensor& moment1,
                         const DenseTensor& moment2,
                         const DenseTensor& beta1_pow,
                         const DenseTensor& beta2_pow,
                         const paddle::optional<DenseTensor>& master_param,
                         const paddle::optional<DenseTensor>& skip_update,
                         const Scalar& beta1,
                         const Scalar& beta2,
                         const Scalar& epsilon,
                         float lr_ratio,
                         float coeff,
                         bool with_decay,
                         bool lazy_mode,
                         int64_t min_row_size_to_use_multithread,
                         bool multi_precision,
                         bool use_global_beta_pow,
                         DenseTensor* param_out,
                         DenseTensor* moment1_out,
                         DenseTensor* moment2_out,
                         DenseTensor* beta1_pow_out,
                         DenseTensor* beta2_pow_out,
                         DenseTensor* master_param_outs) {
  // TODO(houj04):
  // 当KL3稳定以后，并且不需要支持KL1和KL2的时候，拿这里的AdamwDenseKernelKL3替换掉AdamwDenseKernel
  using MPDType = typename phi::dtype::MPTypeTrait<T>::Type;
  using XPUType = typename XPUTypeTrait<T>::Type;

  const auto grad_type = grad.dtype();

  VLOG(4) << "use_global_beta_pow:" << use_global_beta_pow;

  MPDType coeff_ = static_cast<MPDType>(coeff);
  MPDType lr_ratio_ = static_cast<MPDType>(lr_ratio);

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

  // skip_update=true, just copy input to output
  if (skip_update_) {
    VLOG(4) << "Adamw skip update";
    phi::Copy(dev_ctx, param, dev_ctx.GetPlace(), false, param_out);
    phi::Copy(dev_ctx, moment1, dev_ctx.GetPlace(), false, moment1_out);
    phi::Copy(dev_ctx, moment2, dev_ctx.GetPlace(), false, moment2_out);
    if (!use_global_beta_pow) {
      phi::Copy(dev_ctx, beta1_pow, beta1_pow.place(), false, beta1_pow_out);
      phi::Copy(dev_ctx, beta2_pow, beta2_pow.place(), false, beta2_pow_out);
    }
    return;
  }

  // if with_decay = false, coeff = 0
  if (!with_decay) {
    coeff_ = static_cast<MPDType>(0.0);
  }

  MPDType beta1_ = beta1.to<MPDType>();
  MPDType beta2_ = beta2.to<MPDType>();
  MPDType epsilon_ = epsilon.to<MPDType>();
  VLOG(3) << "beta1_pow.numel() : " << beta1_pow.numel()
          << "beta2_pow.numel() : " << beta2_pow.numel();
  VLOG(3) << "param.numel(): " << param.numel();
  PADDLE_ENFORCE_EQ(
      beta1_pow_out->numel(),
      1,
      errors::InvalidArgument("beta1 pow output size should be 1, but received "
                              "value is:%d.",
                              beta1_pow_out->numel()));

  PADDLE_ENFORCE_EQ(
      beta2_pow_out->numel(),
      1,
      errors::InvalidArgument("beta2 pow output size should be 1, but received "
                              "value is:%d.",
                              beta2_pow_out->numel()));

  const MPDType* master_in_data =
      multi_precision ? master_param->data<MPDType>() : nullptr;
  MPDType* master_out_data =
      multi_precision ? dev_ctx.template Alloc<MPDType>(master_param_outs)
                      : nullptr;

  // check moment_dtype
  auto moment1_dtype = moment1.dtype();
  auto moment2_dtype = moment2.dtype();
  PADDLE_ENFORCE_EQ(moment1_dtype,
                    moment1_out->dtype(),
                    errors::InvalidArgument(
                        "moment1.dtype does not match moment1_out->dtype"));
  PADDLE_ENFORCE_EQ(moment2_dtype,
                    moment2_out->dtype(),
                    errors::InvalidArgument(
                        "moment2.dtype does not match moment2_out->dtype"));
  PADDLE_ENFORCE_EQ(
      moment1_dtype,
      moment2_dtype,
      errors::InvalidArgument("moment1.dtype does not match moment2.dtype"));

  bool moment_in_fp16 = false;
  if (moment1_dtype == phi::DataType::FLOAT16) {
    moment_in_fp16 = true;
  } else {
    PADDLE_ENFORCE_EQ(
        moment1_dtype,
        phi::DataType::FLOAT32,
        errors::InvalidArgument("moment1.dtype is neither fp32 nor fp16"));
  }

  float* moment1_input_for_xdnn = nullptr;
  float* moment2_input_for_xdnn = nullptr;
  float* moment1_output_for_xdnn = nullptr;
  float* moment2_output_for_xdnn = nullptr;

  xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
  if (moment_in_fp16) {
    // allocate temp buffer on XPU
    moment1_input_for_xdnn = RAII_GUARD.alloc_l3_or_gm<float>(moment1.numel());
    PADDLE_ENFORCE_XDNN_NOT_NULL(moment1_input_for_xdnn);
    moment2_input_for_xdnn = RAII_GUARD.alloc_l3_or_gm<float>(moment2.numel());
    PADDLE_ENFORCE_XDNN_NOT_NULL(moment2_input_for_xdnn);
    moment1_output_for_xdnn =
        RAII_GUARD.alloc_l3_or_gm<float>(moment1_out->numel());
    PADDLE_ENFORCE_XDNN_NOT_NULL(moment1_output_for_xdnn);
    moment2_output_for_xdnn =
        RAII_GUARD.alloc_l3_or_gm<float>(moment2_out->numel());
    PADDLE_ENFORCE_XDNN_NOT_NULL(moment2_output_for_xdnn);

    int r = 0;
    using XPUType16 = typename XPUTypeTrait<phi::dtype::float16>::Type;

    // cast moment1 and moment2, from fp16 to fp32
    // int cast(Context* ctx, const TX* x, TY* y, int64_t len);
    r = xpu::cast<XPUType16, float>(
        dev_ctx.x_context(),
        reinterpret_cast<const XPUType16*>(
            moment1.template data<phi::dtype::float16>()),
        moment1_input_for_xdnn,
        moment1.numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast moment1 from fp16 to float");
    r = xpu::cast<XPUType16, float>(
        dev_ctx.x_context(),
        reinterpret_cast<const XPUType16*>(
            moment2.template data<phi::dtype::float16>()),
        moment2_input_for_xdnn,
        moment2.numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast moment2 from fp16 to float");

    // acquire xpu_scale_value
    float moment1_scale_value = XPUStorageProperties::default_xpu_scale_value;
    if (moment1.storage_properties_initialized()) {
      moment1_scale_value =
          moment1.storage_properties<XPUStorageProperties>().xpu_scale_value;
    }
    float moment2_scale_value = XPUStorageProperties::default_xpu_scale_value;
    if (moment2.storage_properties_initialized()) {
      moment2_scale_value =
          moment2.storage_properties<XPUStorageProperties>().xpu_scale_value;
    }

    // de-scale using scale_value
    // int scale(Context* ctx, const T* x, T* y, int64_t len, bool
    // bias_after_scale, float _scale, float _bias);
    if (moment1_scale_value > 0) {
      r = xpu::scale<float>(dev_ctx.x_context(),
                            moment1_input_for_xdnn,
                            moment1_input_for_xdnn,
                            moment1.numel(),
                            false,
                            1.0f / moment1_scale_value,
                            0.0f);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "de-scale for moment1");
    }
    if (moment2_scale_value > 0) {
      r = xpu::scale<float>(dev_ctx.x_context(),
                            moment2_input_for_xdnn,
                            moment2_input_for_xdnn,
                            moment2.numel(),
                            false,
                            1.0f / moment2_scale_value,
                            0.0f);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "de-scale for moment2");
    }
  }

  // template <typename T, typename TG, typename MT> DLL_EXPORT int
  // adamw(Context* ctx, MT beta1, MT beta2, MT epsilon, MT coeff, MT lr_ratio,
  // const MT* beta1_pow, MT beta1_pow_scalar, const MT* beta2_pow, MT
  // beta2_pow_scalar, const MT* moment1, MT* moment1_out, const MT* moment2,
  // MT* moment2_out, const MT* lr, const TG* grad, const T* param, T*
  // param_out, const MT* master_param, MT* master_param_out, int64_t n);
  if (beta1_pow.place() == CPUPlace() && beta2_pow.place() == CPUPlace()) {
    // Compute with betapow in REG
    if (grad_type == phi::DataType::FLOAT32) {
      int r = xpu::adamw<XPUType, float, MPDType>(
          dev_ctx.x_context(),
          beta1_,
          beta2_,
          epsilon_,
          coeff_,
          lr_ratio_,
          nullptr,                     // beta1_pow
          *beta1_pow.data<MPDType>(),  // beta1_pow_scalar
          nullptr,                     // beta2_pow
          *beta2_pow.data<MPDType>(),  // beta2_pow_scalar
          moment_in_fp16 ? moment1_input_for_xdnn
                         : moment1.template data<MPDType>(),
          moment_in_fp16 ? moment1_output_for_xdnn
                         : dev_ctx.template Alloc<MPDType>(moment1_out),
          moment_in_fp16 ? moment2_input_for_xdnn
                         : moment2.template data<MPDType>(),
          moment_in_fp16 ? moment2_output_for_xdnn
                         : dev_ctx.template Alloc<MPDType>(moment2_out),
          learning_rate.data<MPDType>(),
          grad.data<float>(),
          reinterpret_cast<const XPUType*>(param.data<T>()),
          reinterpret_cast<XPUType*>(dev_ctx.template Alloc<T>(param_out)),
          master_in_data,
          master_out_data,
          param.numel());
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "adamw");
    } else {
      int r = xpu::adamw<XPUType, XPUType, MPDType>(
          dev_ctx.x_context(),
          beta1_,
          beta2_,
          epsilon_,
          coeff_,
          lr_ratio_,
          nullptr,                     // beta1_pow
          *beta1_pow.data<MPDType>(),  // beta1_pow_scalar
          nullptr,                     // beta2_pow
          *beta2_pow.data<MPDType>(),  // beta2_pow_scalar
          moment_in_fp16 ? moment1_input_for_xdnn
                         : moment1.template data<MPDType>(),
          moment_in_fp16 ? moment1_output_for_xdnn
                         : dev_ctx.template Alloc<MPDType>(moment1_out),
          moment_in_fp16 ? moment2_input_for_xdnn
                         : moment2.template data<MPDType>(),
          moment_in_fp16 ? moment2_output_for_xdnn
                         : dev_ctx.template Alloc<MPDType>(moment2_out),
          learning_rate.data<MPDType>(),
          reinterpret_cast<const XPUType*>(grad.data<T>()),
          reinterpret_cast<const XPUType*>(param.data<T>()),
          reinterpret_cast<XPUType*>(dev_ctx.template Alloc<T>(param_out)),
          master_in_data,
          master_out_data,
          param.numel());
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "adamw");
    }
    if (!use_global_beta_pow) {
      // Cpu update
      dev_ctx.template HostAlloc<MPDType>(beta1_pow_out)[0] =
          beta1_ * beta1_pow.data<MPDType>()[0];
      dev_ctx.template HostAlloc<MPDType>(beta2_pow_out)[0] =
          beta2_ * beta2_pow.data<MPDType>()[0];
    }
  } else {
    if (grad_type == phi::DataType::FLOAT32) {
      int r = xpu::adamw<XPUType, float, MPDType>(
          dev_ctx.x_context(),
          beta1_,
          beta2_,
          epsilon_,
          coeff_,
          lr_ratio_,
          beta1_pow.data<MPDType>(),  // beta1_pow
          0.0f,                       // beta1_pow_scalar
          beta2_pow.data<MPDType>(),  // beta2_pow
          0.0f,                       // beta2_pow_scalar
          moment_in_fp16 ? moment1_input_for_xdnn
                         : moment1.template data<MPDType>(),
          moment_in_fp16 ? moment1_output_for_xdnn
                         : dev_ctx.template Alloc<MPDType>(moment1_out),
          moment_in_fp16 ? moment2_input_for_xdnn
                         : moment2.template data<MPDType>(),
          moment_in_fp16 ? moment2_output_for_xdnn
                         : dev_ctx.template Alloc<MPDType>(moment2_out),
          learning_rate.data<MPDType>(),
          grad.data<float>(),
          reinterpret_cast<const XPUType*>(param.data<T>()),
          reinterpret_cast<XPUType*>(dev_ctx.template Alloc<T>(param_out)),
          master_in_data,
          master_out_data,
          param.numel());
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "adamw");
    } else {
      int r = xpu::adamw<XPUType, XPUType, MPDType>(
          dev_ctx.x_context(),
          beta1_,
          beta2_,
          epsilon_,
          coeff_,
          lr_ratio_,
          beta1_pow.data<MPDType>(),  // beta1_pow
          0.0f,                       // beta1_pow_scalar
          beta2_pow.data<MPDType>(),  // beta2_pow
          0.0f,                       // beta2_pow_scalar
          moment_in_fp16 ? moment1_input_for_xdnn
                         : moment1.template data<MPDType>(),
          moment_in_fp16 ? moment1_output_for_xdnn
                         : dev_ctx.template Alloc<MPDType>(moment1_out),
          moment_in_fp16 ? moment2_input_for_xdnn
                         : moment2.template data<MPDType>(),
          moment_in_fp16 ? moment2_output_for_xdnn
                         : dev_ctx.template Alloc<MPDType>(moment2_out),
          learning_rate.data<MPDType>(),
          reinterpret_cast<const XPUType*>(grad.data<T>()),
          reinterpret_cast<const XPUType*>(param.data<T>()),
          reinterpret_cast<XPUType*>(dev_ctx.template Alloc<T>(param_out)),
          master_in_data,
          master_out_data,
          param.numel());
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "adamw");
    }
    if (!use_global_beta_pow) {
      // Update with xpu
      int r = xpu::scale(dev_ctx.x_context(),
                         beta1_pow.data<MPDType>(),
                         dev_ctx.template Alloc<MPDType>(beta1_pow_out),
                         beta1_pow.numel(),
                         false,
                         beta1_,
                         0.0f);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "scale");
      r = xpu::scale(dev_ctx.x_context(),
                     beta2_pow.data<MPDType>(),
                     dev_ctx.template Alloc<MPDType>(beta2_pow_out),
                     beta2_pow.numel(),
                     false,
                     beta2_,
                     0.0f);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "scale");
    }
  }

  if (moment_in_fp16) {
    int r = 0;
    using XPUType16 = typename XPUTypeTrait<phi::dtype::float16>::Type;

    // findmax and calculate scale_value for moment1 and moment2
    int max_ptr_size = phi::backends::xpu::get_xpu_max_ptr_size(-1);
    float* buffer_for_findmax = RAII_GUARD.alloc_l3_or_gm<float>(max_ptr_size);

    // for moment1
    float moment1_max = GetAbsMax<Context>(dev_ctx,
                                           moment1_output_for_xdnn,
                                           buffer_for_findmax,
                                           moment1_out->numel());
    float moment1_scale_value = 65504.0f / moment1_max / 2.0f;
    // int scale(Context* ctx, const T* x, T* y, int64_t len, bool
    // bias_after_scale, float _scale, float _bias);
    r = xpu::scale<float>(dev_ctx.x_context(),
                          moment1_output_for_xdnn,
                          moment1_output_for_xdnn,
                          moment1_out->numel(),
                          false,
                          moment1_scale_value,
                          0.0f);
    PADDLE_ENFORCE_XDNN_SUCCESS(
        r, "scale before convert to fp16, for moment1_output_for_xdnn");
    // write to moment1_out
    std::unique_ptr<phi::StorageProperties> moment1_out_sp =
        std::make_unique<phi::XPUStorageProperties>(moment1_scale_value);
    moment1_out->set_storage_properties(std::move(moment1_out_sp));

    // for moment2
    float moment2_max_ = GetAbsMax<Context>(dev_ctx,
                                            moment2_output_for_xdnn,
                                            buffer_for_findmax,
                                            moment2_out->numel());
    float moment2_scale_value = 65504.0f / moment2_max_ / 2.0f;
    // int scale(Context* ctx, const T* x, T* y, int64_t len, bool
    // bias_after_scale, float _scale, float _bias);
    r = xpu::scale<float>(dev_ctx.x_context(),
                          moment2_output_for_xdnn,
                          moment2_output_for_xdnn,
                          moment2_out->numel(),
                          false,
                          moment2_scale_value,
                          0.0f);
    PADDLE_ENFORCE_XDNN_SUCCESS(
        r, "scale before convert to fp16, for moment2_output_for_xdnn");
    // write to moment2_out
    std::unique_ptr<phi::StorageProperties> moment2_out_sp =
        std::make_unique<phi::XPUStorageProperties>(moment2_scale_value);
    moment2_out->set_storage_properties(std::move(moment2_out_sp));

    // cast moment1 and moment2 output, from fp32 to fp16
    // int cast(Context* ctx, const TX* x, TY* y, int64_t len);
    r = xpu::cast<float, XPUType16>(
        dev_ctx.x_context(),
        moment1_output_for_xdnn,
        reinterpret_cast<XPUType16*>(
            dev_ctx.template Alloc<phi::dtype::float16>(moment1_out)),
        moment1.numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast moment1_out from float to fp16");
    r = xpu::cast<float, XPUType16>(
        dev_ctx.x_context(),
        moment2_output_for_xdnn,
        reinterpret_cast<XPUType16*>(
            dev_ctx.template Alloc<phi::dtype::float16>(moment2_out)),
        moment2.numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast moment2_out from float to fp16");
  }
  return;
}

template <typename T, typename Context>
void AdamwDenseKernel(
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
    float lr_ratio,
    float coeff,
    bool with_decay,
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

  auto dev_version =
      phi::backends::xpu::get_xpu_version(dev_ctx.GetPlace().GetDeviceId());
  if (dev_version == phi::backends::xpu::XPUVersion::XPU3) {
    AdamwDenseKernelKL3<T, Context>(dev_ctx,
                                    param,
                                    grad,
                                    learning_rate,
                                    moment1,
                                    moment2,
                                    beta1_pow,
                                    beta2_pow,
                                    master_param,
                                    skip_update,
                                    beta1,
                                    beta2,
                                    epsilon,
                                    lr_ratio,
                                    coeff,
                                    with_decay,
                                    lazy_mode,
                                    min_row_size_to_use_multithread,
                                    multi_precision,
                                    use_global_beta_pow,
                                    param_out,
                                    moment1_out,
                                    moment2_out,
                                    beta1_pow_out,
                                    beta2_pow_out,
                                    master_param_outs);
    return;
  }

  // check moment_dtype
  auto moment1_dtype = moment1.dtype();
  auto moment2_dtype = moment2.dtype();
  PADDLE_ENFORCE_EQ(moment1_dtype,
                    moment1_out->dtype(),
                    errors::InvalidArgument(
                        "moment1.dtype does not match moment1_out->dtype"));
  PADDLE_ENFORCE_EQ(moment2_dtype,
                    moment2_out->dtype(),
                    errors::InvalidArgument(
                        "moment2.dtype does not match moment2_out->dtype"));
  PADDLE_ENFORCE_EQ(
      moment1_dtype,
      moment2_dtype,
      errors::InvalidArgument("moment1.dtype does not match moment2.dtype"));

  bool moment_in_fp16 = false;
  if (moment1_dtype == phi::DataType::FLOAT16) {
    moment_in_fp16 = true;
  } else {
    PADDLE_ENFORCE_EQ(
        moment1_dtype,
        phi::DataType::FLOAT32,
        errors::InvalidArgument("moment1.dtype is neither fp32 nor fp16"));
  }

  float* moment1_input_for_xdnn = nullptr;
  float* moment2_input_for_xdnn = nullptr;
  float* moment1_output_for_xdnn = nullptr;
  float* moment2_output_for_xdnn = nullptr;

  xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
  if (moment_in_fp16) {
    // allocate temp buffer on XPU
    moment1_input_for_xdnn = RAII_GUARD.alloc_l3_or_gm<float>(moment1.numel());
    PADDLE_ENFORCE_XDNN_NOT_NULL(moment1_input_for_xdnn);
    moment2_input_for_xdnn = RAII_GUARD.alloc_l3_or_gm<float>(moment2.numel());
    PADDLE_ENFORCE_XDNN_NOT_NULL(moment2_input_for_xdnn);
    moment1_output_for_xdnn =
        RAII_GUARD.alloc_l3_or_gm<float>(moment1_out->numel());
    PADDLE_ENFORCE_XDNN_NOT_NULL(moment1_output_for_xdnn);
    moment2_output_for_xdnn =
        RAII_GUARD.alloc_l3_or_gm<float>(moment2_out->numel());
    PADDLE_ENFORCE_XDNN_NOT_NULL(moment2_output_for_xdnn);

    int r = 0;
    using XPUType16 = typename XPUTypeTrait<phi::dtype::float16>::Type;

    // cast moment1 and moment2, from fp16 to fp32
    // int cast(Context* ctx, const TX* x, TY* y, int64_t len);
    r = xpu::cast<XPUType16, float>(
        dev_ctx.x_context(),
        reinterpret_cast<const XPUType16*>(
            moment1.template data<phi::dtype::float16>()),
        moment1_input_for_xdnn,
        moment1.numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast moment1 from fp16 to float");
    r = xpu::cast<XPUType16, float>(
        dev_ctx.x_context(),
        reinterpret_cast<const XPUType16*>(
            moment2.template data<phi::dtype::float16>()),
        moment2_input_for_xdnn,
        moment2.numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast moment2 from fp16 to float");

    // acquire xpu_scale_value
    float moment1_scale_value = XPUStorageProperties::default_xpu_scale_value;
    if (moment1.storage_properties_initialized()) {
      moment1_scale_value =
          moment1.storage_properties<XPUStorageProperties>().xpu_scale_value;
    }
    float moment2_scale_value = XPUStorageProperties::default_xpu_scale_value;
    if (moment2.storage_properties_initialized()) {
      moment2_scale_value =
          moment2.storage_properties<XPUStorageProperties>().xpu_scale_value;
    }

    // de-scale using scale_value
    // int scale(Context* ctx, const T* x, T* y, int64_t len, bool
    // bias_after_scale, float _scale, float _bias);
    if (moment1_scale_value > 0) {
      r = xpu::scale<float>(dev_ctx.x_context(),
                            moment1_input_for_xdnn,
                            moment1_input_for_xdnn,
                            moment1.numel(),
                            false,
                            1.0f / moment1_scale_value,
                            0.0f);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "de-scale for moment1");
    }
    if (moment2_scale_value > 0) {
      r = xpu::scale<float>(dev_ctx.x_context(),
                            moment2_input_for_xdnn,
                            moment2_input_for_xdnn,
                            moment2.numel(),
                            false,
                            1.0f / moment2_scale_value,
                            0.0f);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "de-scale for moment2");
    }
  }

  using XPUType = typename XPUTypeTrait<T>::Type;
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
    VLOG(4) << "Adamw skip update";
    phi::Copy(dev_ctx, param, dev_ctx.GetPlace(), false, param_out);
    phi::Copy(dev_ctx, moment1, dev_ctx.GetPlace(), false, moment1_out);
    phi::Copy(dev_ctx, moment2, dev_ctx.GetPlace(), false, moment2_out);
    if (!use_global_beta_pow) {
      phi::Copy(dev_ctx, beta1_pow, beta1_pow.place(), false, beta1_pow_out);
      phi::Copy(dev_ctx, beta2_pow, beta2_pow.place(), false, beta2_pow_out);
    }
    return;
  }

  auto beta1_ = beta1.to<float>();
  auto beta2_ = beta2.to<float>();
  auto epsilon_ = epsilon.to<float>();

  const float* beta1_pow_ptr = beta1_pow.template data<float>();
  const float* beta2_pow_ptr = beta2_pow.template data<float>();
  DenseTensor xpu_beta1_pow;
  DenseTensor xpu_beta2_pow;
  if (beta1_pow.place() == CPUPlace() && beta2_pow.place() == CPUPlace()) {
    phi::Copy(dev_ctx, beta1_pow, dev_ctx.GetPlace(), false, &xpu_beta1_pow);
    phi::Copy(dev_ctx, beta2_pow, dev_ctx.GetPlace(), false, &xpu_beta2_pow);
    dev_ctx.Wait();
    beta1_pow_ptr = xpu_beta1_pow.template data<float>();
    beta2_pow_ptr = xpu_beta2_pow.template data<float>();
  }
  if (!with_decay) {
    coeff = static_cast<float>(0.0);
  }

  float* new_lr = RAII_GUARD.alloc_l3_or_gm<float>(learning_rate.numel());
  PADDLE_ENFORCE_XDNN_NOT_NULL(new_lr);
  int r = 0;
  r = xpu::scale(dev_ctx.x_context(),
                 learning_rate.template data<float>(),
                 new_lr,
                 learning_rate.numel(),
                 false,
                 lr_ratio,
                 0.0f);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "scale");

  if (multi_precision) {
    const float* master_param_in_data = master_param->data<float>();
    float* master_param_out_data =
        dev_ctx.template Alloc<float>(master_param_outs);
    // convert grad to float if necessary
    float* grad_fp32 = nullptr;
    const auto grad_type = grad.dtype();
    if (grad_type != phi::DataType::FLOAT32) {
      grad_fp32 = RAII_GUARD.alloc_l3_or_gm<float>(grad.numel());
      PADDLE_ENFORCE_XDNN_NOT_NULL(grad_fp32);
      // int cast(Context* ctx, const TX* x, TY* y, int64_t len);
      int r = xpu::cast<XPUType, float>(
          dev_ctx.x_context(),
          reinterpret_cast<const XPUType*>(grad.template data<T>()),
          grad_fp32,
          grad.numel());
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast");
    }
    // int adamw(Context* ctx, const T* g, const float* mom1, const float* mom2,
    // const T* param, const float* beta1_pow, const float* beta2_pow, const
    // float* lr, float* moment1_out, float* moment2_out, T* param_out, float
    // beta1, float beta2, float epsilon, float coeff, int64_t n);
    r = xpu::adamw<float>(
        dev_ctx.x_context(),
        (grad_type == phi::DataType::FLOAT32) ? grad.data<float>() : grad_fp32,
        moment_in_fp16 ? moment1_input_for_xdnn
                       : moment1.template data<float>(),
        moment_in_fp16 ? moment2_input_for_xdnn
                       : moment2.template data<float>(),
        master_param_in_data,
        beta1_pow_ptr,
        beta2_pow_ptr,
        new_lr,
        moment_in_fp16 ? moment1_output_for_xdnn
                       : dev_ctx.template Alloc<float>(moment1_out),
        moment_in_fp16 ? moment2_output_for_xdnn
                       : dev_ctx.template Alloc<float>(moment2_out),
        master_param_out_data,
        beta1_,
        beta2_,
        epsilon_,
        coeff,
        param.numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "adamw");
    // convert master_param_out(fp32) to param_out(T)
    r = xpu::cast<float, XPUType>(
        dev_ctx.x_context(),
        master_param_out_data,
        reinterpret_cast<XPUType*>(dev_ctx.template Alloc<T>(param_out)),
        param_out->numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast");
  } else {
    // int adamw(Context* ctx, const T* g, const float* mom1, const float* mom2,
    // const T* param, const float* beta1_pow, const float* beta2_pow, const
    // float* lr, float* moment1_out, float* moment2_out, T* param_out, float
    // beta1, float beta2, float epsilon, float coeff, int64_t n);
    r = xpu::adamw(
        dev_ctx.x_context(),
        reinterpret_cast<const XPUType*>(grad.template data<T>()),
        moment_in_fp16 ? moment1_input_for_xdnn
                       : moment1.template data<float>(),
        moment_in_fp16 ? moment2_input_for_xdnn
                       : moment2.template data<float>(),
        reinterpret_cast<const XPUType*>(param.template data<T>()),
        beta1_pow_ptr,
        beta2_pow_ptr,
        new_lr,
        moment_in_fp16 ? moment1_output_for_xdnn
                       : dev_ctx.template Alloc<float>(moment1_out),
        moment_in_fp16 ? moment2_output_for_xdnn
                       : dev_ctx.template Alloc<float>(moment2_out),
        reinterpret_cast<XPUType*>(dev_ctx.template Alloc<T>(param_out)),
        beta1_,
        beta2_,
        epsilon_,
        coeff,
        param.numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "adamw");
  }

  if (moment_in_fp16) {
    int r = 0;
    using XPUType16 = typename XPUTypeTrait<phi::dtype::float16>::Type;

    // findmax and calculate scale_value for moment1 and moment2
    int max_ptr_size = phi::backends::xpu::get_xpu_max_ptr_size(-1);
    float* buffer_for_findmax = RAII_GUARD.alloc_l3_or_gm<float>(max_ptr_size);

    // for moment1
    float moment1_max = GetAbsMax<Context>(dev_ctx,
                                           moment1_output_for_xdnn,
                                           buffer_for_findmax,
                                           moment1_out->numel());
    float moment1_scale_value = 65504.0f / moment1_max / 2.0f;
    // int scale(Context* ctx, const T* x, T* y, int64_t len, bool
    // bias_after_scale, float _scale, float _bias);
    r = xpu::scale<float>(dev_ctx.x_context(),
                          moment1_output_for_xdnn,
                          moment1_output_for_xdnn,
                          moment1_out->numel(),
                          false,
                          moment1_scale_value,
                          0.0f);
    PADDLE_ENFORCE_XDNN_SUCCESS(
        r, "scale before convert to fp16, for moment1_output_for_xdnn");
    // write to moment1_out
    std::unique_ptr<phi::StorageProperties> moment1_out_sp =
        std::make_unique<phi::XPUStorageProperties>(moment1_scale_value);
    moment1_out->set_storage_properties(std::move(moment1_out_sp));

    // for moment2
    float moment2_max_ = GetAbsMax<Context>(dev_ctx,
                                            moment2_output_for_xdnn,
                                            buffer_for_findmax,
                                            moment2_out->numel());
    float moment2_scale_value = 65504.0f / moment2_max_ / 2.0f;
    // int scale(Context* ctx, const T* x, T* y, int64_t len, bool
    // bias_after_scale, float _scale, float _bias);
    r = xpu::scale<float>(dev_ctx.x_context(),
                          moment2_output_for_xdnn,
                          moment2_output_for_xdnn,
                          moment2_out->numel(),
                          false,
                          moment2_scale_value,
                          0.0f);
    PADDLE_ENFORCE_XDNN_SUCCESS(
        r, "scale before convert to fp16, for moment2_output_for_xdnn");
    // write to moment2_out
    std::unique_ptr<phi::StorageProperties> moment2_out_sp =
        std::make_unique<phi::XPUStorageProperties>(moment2_scale_value);
    moment2_out->set_storage_properties(std::move(moment2_out_sp));

    // cast moment1 and moment2 output, from fp32 to fp16
    // int cast(Context* ctx, const TX* x, TY* y, int64_t len);
    r = xpu::cast<float, XPUType16>(
        dev_ctx.x_context(),
        moment1_output_for_xdnn,
        reinterpret_cast<XPUType16*>(
            dev_ctx.template Alloc<phi::dtype::float16>(moment1_out)),
        moment1.numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast moment1_out from float to fp16");
    r = xpu::cast<float, XPUType16>(
        dev_ctx.x_context(),
        moment2_output_for_xdnn,
        reinterpret_cast<XPUType16*>(
            dev_ctx.template Alloc<phi::dtype::float16>(moment2_out)),
        moment2.numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast moment2_out from float to fp16");
  }

  if (!use_global_beta_pow) {
    // update in cpu
    if (beta1_pow.place() == CPUPlace() && beta2_pow.place() == CPUPlace()) {
      const float* beta1_pow_p = beta1_pow.template data<float>();
      dev_ctx.template HostAlloc<float>(beta1_pow_out)[0] =
          beta1_ * beta1_pow_p[0];
      const float* beta2_pow_p = beta2_pow.template data<float>();
      dev_ctx.template HostAlloc<float>(beta2_pow_out)[0] =
          beta2_ * beta2_pow_p[0];
      xpu_wait(dev_ctx.x_context()->xpu_stream);
    } else {  // update in  xpu
      float* beta1_pow_out_p = dev_ctx.template Alloc<float>(beta1_pow_out);
      float* beta2_pow_out_p = dev_ctx.template Alloc<float>(beta2_pow_out);
      int r = xpu::scale(dev_ctx.x_context(),
                         beta1_pow_ptr,
                         beta1_pow_out_p,
                         beta1_pow.numel(),
                         false,
                         beta1_,
                         0.0f);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "scale");
      r = xpu::scale(dev_ctx.x_context(),
                     beta2_pow_ptr,
                     beta2_pow_out_p,
                     beta2_pow.numel(),
                     false,
                     beta2_,
                     0.0f);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "scale");
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(adamw,
                   XPU,
                   ALL_LAYOUT,
                   phi::AdamwDenseKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {
  // Skip beta1_pow, beta2_pow, skip_update data transform
  kernel->InputAt(6).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(7).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(9).SetBackend(phi::Backend::ALL_BACKEND);

  if (kernel_key.dtype() == phi::DataType::FLOAT16 ||
      kernel_key.dtype() == phi::DataType::BFLOAT16) {
    kernel->OutputAt(1).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(2).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(3).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(4).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(5).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(6).SetDataType(phi::DataType::FLOAT32);
  }
  kernel->OutputAt(4).SetBackend(phi::Backend::UNDEFINED);
  kernel->OutputAt(5).SetBackend(phi::Backend::UNDEFINED);
}
