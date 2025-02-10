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

#pragma once

#ifdef PADDLE_WITH_XPU

#include <vector>
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_header.h"
#include "paddle/phi/backends/xpu/xpu_info.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"

#ifdef PADDLE_WITH_XPU_XRE5
#include "xblas/cublasLt.h"
namespace xblas = baidu::xpu::xblas;
#endif

namespace phi {

using XPUTypeFP16 = typename XPUTypeTrait<phi::dtype::float16>::Type;
using XPUTypeBF16 = typename XPUTypeTrait<phi::dtype::bfloat16>::Type;

enum XPUFCCalcType {
  FC_INT16 = 0,
  FC_INT32,
  FC_FLOAT,
  FC_INT32_WITH_LL,
  FC_TF32,
  FC_FLOAT16,
};

template <typename T>
XPUFCCalcType FCCalcType() {
  const char* xpu_paddle_fc_float16 = std::getenv("XPU_PADDLE_FC_FLOAT16");
  if (xpu_paddle_fc_float16 != nullptr &&
      (std::is_same<phi::dtype::float16, T>::value ||
       std::is_same<XPUTypeFP16, T>::value || std::is_same<float, T>::value)) {
    return XPUFCCalcType::FC_FLOAT16;
  } else if (std::is_same<phi::dtype::float16, T>::value ||
             std::is_same<XPUTypeFP16, T>::value) {
    return XPUFCCalcType::FC_INT16;
  } else if (std::getenv("XPU_PADDLE_FC_INT32") != nullptr) {
    return XPUFCCalcType::FC_INT32;
  } else if (std::getenv("XPU_PADDLE_FC_LOCAL_INT16") != nullptr) {
    return XPUFCCalcType::FC_FLOAT;
  } else if (std::getenv("XPU_PADDLE_FC_INT32_WITH_LL") != nullptr) {
    return XPUFCCalcType::FC_INT32_WITH_LL;
  } else if ((std::is_same<float, T>::value &&
              std::getenv("XPU_PADDLE_FC_INT16") != nullptr)) {
    return XPUFCCalcType::FC_INT16;
  }
  return XPUFCCalcType::FC_TF32;
}

struct XpuFcInfo {
  int bs;
  int m;
  int n;
  int k;
  bool trans_x;
  bool trans_y;
  int stride_x;
  int stride_y;
  int stride_out;
  float* max_x;
  float* max_y;
  float* max_out;
  const float* bias;
  bool is_x_need_broadcast;
  bool is_y_need_broadcast;
  const float* scale_x;
  const float* scale_y;
  int scale_x_mode;
  int scale_y_mode;

  XpuFcInfo()
      : bs(0),
        m(0),
        n(0),
        k(0),
        trans_x(false),
        trans_y(false),
        stride_x(0),
        stride_y(0),
        stride_out(0),
        max_x(nullptr),
        max_y(nullptr),
        max_out(nullptr),
        bias(nullptr),
        is_x_need_broadcast(false),
        is_y_need_broadcast(false),
        scale_x(nullptr),
        scale_y(nullptr),
        scale_x_mode(0),
        scale_y_mode(0) {}
  void InitFcInfo(int bs,
                  int m,
                  int n,
                  int k,
                  bool trans_x,
                  bool trans_y,
                  float* max_x,
                  float* max_y,
                  float* max_out) {
    this->bs = bs;
    this->m = m;
    this->n = n;
    this->k = k;
    this->trans_x = trans_x;
    this->trans_y = trans_y;
    this->max_x = max_x;
    this->max_y = max_y;
    this->max_out = max_out;

    if (this->bs <= 1) {
      this->stride_x = trans_x ? m : k;
      this->stride_y = trans_y ? k : n;
      this->stride_out = n;
    } else {
      this->stride_x = m * k;
      this->stride_y = k * n;
      this->stride_out = m * n;
    }
  }
};

static std::ostream& operator<<(std::ostream& os, const XpuFcInfo& fc_inf) {
  os << "fc_inf[ bs, m, n, k, trans_x, trans_y, stride_x, stride_y, "
        "stride_out] = "
     << "[" << fc_inf.bs << ", " << fc_inf.m << ", " << fc_inf.n << ", "
     << fc_inf.k << ", " << fc_inf.trans_x << ", " << fc_inf.trans_y << ", "
     << fc_inf.stride_x << ", " << fc_inf.stride_y << ", " << fc_inf.stride_out;
  return os;
}

static void GetFCInfo(const phi::DDim& x_dims,
                      const phi::DDim& y_dims,
                      bool trans_x,
                      bool trans_y,
                      XpuFcInfo* info) {
  DDim new_x_dims =
      (x_dims.size() > 1) ? x_dims : common::make_ddim({1, x_dims[0]});
  DDim new_y_dims =
      (y_dims.size() > 1) ? y_dims : common::make_ddim({y_dims[0], 1});

  auto mat_dim_a = phi::funcs::CreateMatrixDescriptor(new_x_dims, 0, trans_x);
  auto mat_dim_b = phi::funcs::CreateMatrixDescriptor(new_y_dims, 0, trans_y);

  if (x_dims.size() >= 3 && y_dims.size() <= 2) {
    if (!trans_x || mat_dim_a.batch_size_ == 1) {
      mat_dim_a.height_ *= mat_dim_a.batch_size_;
      mat_dim_a.batch_size_ = 0;
    } else {
      info->is_y_need_broadcast = true;
    }
  }

  if (y_dims.size() >= 3 && x_dims.size() <= 2) {
    info->is_x_need_broadcast = (mat_dim_b.batch_size_ > 1);
  }
  PADDLE_ENFORCE_EQ(mat_dim_a.width_,
                    mat_dim_b.height_,
                    common::errors::InvalidArgument(
                        "Shape mistake in matmul_op xdims = %s ydims = %s "
                        "x_trans = %d y_trans = %d",
                        x_dims.to_str(),
                        y_dims.to_str(),
                        mat_dim_a.trans_,
                        mat_dim_b.trans_));

  if (mat_dim_a.batch_size_ == 0 && mat_dim_b.batch_size_ == 1) {
    mat_dim_a.batch_size_ = mat_dim_b.batch_size_ = 0;
  }
  if (mat_dim_a.batch_size_ == 1 && mat_dim_b.batch_size_ == 0) {
    mat_dim_a.batch_size_ = mat_dim_b.batch_size_ = 0;
  }

  info->m = mat_dim_a.height_;
  info->n = mat_dim_b.width_;
  info->k = mat_dim_a.width_;
  info->bs = std::max(mat_dim_a.batch_size_, mat_dim_b.batch_size_);
  info->trans_x = trans_x;
  info->trans_y = trans_y;

  if (info->bs <= 1) {
    info->stride_x = trans_x ? info->m : info->k;
    info->stride_y = trans_y ? info->k : info->n;
    info->stride_out = info->n;
  } else {
    info->stride_x = info->m * info->k;
    info->stride_y = info->k * info->n;
    info->stride_out = info->m * info->n;
  }
}

template <typename XPUType, typename FCT>
static void xblas_fc_wrapper(xpu::Context* ctx,
                             const XPUType* x,
                             const XPUType* w,
                             XPUType* y,
                             int m,
                             int n,
                             int k,
                             bool x_trans,
                             bool w_trans,
                             const float* x_maxptr,
                             const float* w_maxptr,
                             float* y_maxptr,
                             int ldx,
                             int ldw,
                             int ldy,
                             float alpha,
                             float beta,
                             const float* bias,
                             const xpu::Activation_t& act,
                             const float* scale_x,
                             const float* scale_w,
                             int scale_x_mode,
                             int scale_w_mode) {
  int r = 0;
  xpu::ctx_guard RAII_GUARD(ctx);
  if (x_trans && std::getenv("XPU_PADDLE_FC_TRANS_A") != nullptr &&
      std::is_same<float, XPUType>::value) {
    XPUType* l3_addr = nullptr;
    l3_addr = RAII_GUARD.alloc_l3_or_gm<XPUType>(m * k);
    PADDLE_ENFORCE_XDNN_NOT_NULL(l3_addr);

    std::vector<int> shape = {k, m};
    std::vector<int> axis = {1, 0};
    r = xpu::transpose<XPUType>(ctx, x, l3_addr, shape, axis);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "transpose");
#ifdef PADDLE_WITH_XPU_XRE5
    r = xblas::fc_fusion<XPUType, XPUType, XPUType, FCT>(ctx,
                                                         l3_addr,
                                                         w,
                                                         y,
                                                         m,
                                                         n,
                                                         k,
                                                         false,
                                                         w_trans,
                                                         x_maxptr,
                                                         w_maxptr,
                                                         y_maxptr,
                                                         k,
                                                         ldw,
                                                         ldy,
                                                         alpha,
                                                         beta,
                                                         bias,
                                                         act,
                                                         scale_x,
                                                         scale_w,
                                                         scale_x_mode,
                                                         scale_w_mode);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "xblas_fc_fusion");
#else
    r = xpu::fc_fusion<XPUType, XPUType, XPUType, FCT>(ctx,
                                                       l3_addr,
                                                       w,
                                                       y,
                                                       m,
                                                       n,
                                                       k,
                                                       false,
                                                       w_trans,
                                                       x_maxptr,
                                                       w_maxptr,
                                                       y_maxptr,
                                                       k,
                                                       ldw,
                                                       ldy,
                                                       alpha,
                                                       beta,
                                                       bias,
                                                       act);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "xpu_fc_fusion");
#endif
  } else {
#ifdef PADDLE_WITH_XPU_XRE5
    r = xblas::fc_fusion<XPUType, XPUType, XPUType, FCT>(ctx,
                                                         x,
                                                         w,
                                                         y,
                                                         m,
                                                         n,
                                                         k,
                                                         x_trans,
                                                         w_trans,
                                                         x_maxptr,
                                                         w_maxptr,
                                                         y_maxptr,
                                                         ldx,
                                                         ldw,
                                                         ldy,
                                                         alpha,
                                                         beta,
                                                         bias,
                                                         act,
                                                         scale_x,
                                                         scale_w,
                                                         scale_x_mode,
                                                         scale_w_mode);

    PADDLE_ENFORCE_XDNN_SUCCESS(r, "xblas_fc_fusion");
#else
    r = xpu::fc_fusion<XPUType, XPUType, XPUType, FCT>(ctx,
                                                       x,
                                                       w,
                                                       y,
                                                       m,
                                                       n,
                                                       k,
                                                       x_trans,
                                                       w_trans,
                                                       x_maxptr,
                                                       w_maxptr,
                                                       y_maxptr,
                                                       ldx,
                                                       ldw,
                                                       ldy,
                                                       alpha,
                                                       beta,
                                                       bias,
                                                       act);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "xpu_fc_fusion");
#endif
  }
}

#define DECLARE_UNSUPPORTED_XBLAS_FC_WRAPPER(XPUType, FCT)          \
  template <>                                                       \
  void xblas_fc_wrapper<XPUType, FCT>(xpu::Context * ctx,           \
                                      const XPUType* x,             \
                                      const XPUType* w,             \
                                      XPUType* y,                   \
                                      int m,                        \
                                      int n,                        \
                                      int k,                        \
                                      bool x_trans,                 \
                                      bool w_trans,                 \
                                      const float* x_maxptr,        \
                                      const float* w_maxptr,        \
                                      float* y_maxptr,              \
                                      int ldx,                      \
                                      int ldw,                      \
                                      int ldy,                      \
                                      float alpha,                  \
                                      float beta,                   \
                                      const float* bias,            \
                                      const xpu::Activation_t& act, \
                                      const float* scale_x,         \
                                      const float* scale_w,         \
                                      int scale_x_mode,             \
                                      int scale_w_mode) {           \
    int r = xpu::Error_t::INVALID_PARAM;                            \
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "xblas_fc_wrapper");             \
  }

DECLARE_UNSUPPORTED_XBLAS_FC_WRAPPER(XPUTypeBF16, int_with_ll_t)
DECLARE_UNSUPPORTED_XBLAS_FC_WRAPPER(XPUTypeBF16, int16_t)
DECLARE_UNSUPPORTED_XBLAS_FC_WRAPPER(XPUTypeBF16, int32_t)
DECLARE_UNSUPPORTED_XBLAS_FC_WRAPPER(XPUTypeBF16, XPUTypeFP16)
DECLARE_UNSUPPORTED_XBLAS_FC_WRAPPER(XPUTypeFP16, int32_t)
DECLARE_UNSUPPORTED_XBLAS_FC_WRAPPER(XPUTypeFP16, tfloat32)

#ifndef PADDLE_WITH_XPU_XRE5
// disable in kl2
DECLARE_UNSUPPORTED_XBLAS_FC_WRAPPER(float, XPUTypeFP16)
#endif

template <typename XPUType, typename FCT, typename TGEMM_OUT>
static void xblas_fc_batch_wrapper(xpu::Context* xpu_ctx,
                                   int bs,
                                   bool trans_x,
                                   bool trans_w,
                                   int m,
                                   int n,
                                   int k,
                                   float alpha,
                                   const XPUType* x,
                                   int stride_x,
                                   const XPUType* w,
                                   int stride_w,
                                   float beta,
                                   XPUType* y,
                                   int stride_y,
                                   const float* x_maxptr,
                                   const float* w_maxptr) {
#ifdef PADDLE_WITH_XPU_XRE5
  int r = xblas::fc_batched<XPUType, XPUType, XPUType, FCT, TGEMM_OUT, 0>(
      xpu_ctx,
      bs,
      trans_x,
      trans_w,
      m,
      n,
      k,
      alpha,
      reinterpret_cast<const XPUType*>(x),
      stride_x,
      reinterpret_cast<const XPUType*>(w),
      stride_w,
      beta,
      reinterpret_cast<XPUType*>(y),
      stride_y,
      x_maxptr,
      w_maxptr);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "xblas_fc_batched");
#else
  int r = xpu::fc_batched<XPUType, XPUType, XPUType, FCT>(
      xpu_ctx,
      bs,
      trans_x,
      trans_w,
      m,
      n,
      k,
      alpha,
      reinterpret_cast<const XPUType*>(x),
      stride_x,
      reinterpret_cast<const XPUType*>(w),
      stride_w,
      beta,
      reinterpret_cast<XPUType*>(y),
      stride_y,
      x_maxptr,
      w_maxptr);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "xdnn_fc_batched");
#endif
}

#define DECLARE_UNSUPPORTED_XBLAS_FC_BATCH_WRAPPER(XPUType, FCT, TGEMM_OUT) \
  template <>                                                               \
  void xblas_fc_batch_wrapper<XPUType, FCT, TGEMM_OUT>(                     \
      xpu::Context * xpu_ctx,                                               \
      int bs,                                                               \
      bool trans_x,                                                         \
      bool trans_w,                                                         \
      int m,                                                                \
      int n,                                                                \
      int k,                                                                \
      float alpha,                                                          \
      const XPUType* x,                                                     \
      int stride_x,                                                         \
      const XPUType* w,                                                     \
      int stride_w,                                                         \
      float beta,                                                           \
      XPUType* y,                                                           \
      int stride_y,                                                         \
      const float* x_maxptr,                                                \
      const float* w_maxptr) {                                              \
    int r = xpu::Error_t::INVALID_PARAM;                                    \
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "xblas_fc_batched");                     \
  }

DECLARE_UNSUPPORTED_XBLAS_FC_BATCH_WRAPPER(XPUTypeBF16,
                                           int_with_ll_t,
                                           XPUTypeFP16)
DECLARE_UNSUPPORTED_XBLAS_FC_BATCH_WRAPPER(XPUTypeFP16, tfloat32, XPUTypeFP16)
DECLARE_UNSUPPORTED_XBLAS_FC_BATCH_WRAPPER(XPUTypeBF16, float, XPUTypeFP16)
DECLARE_UNSUPPORTED_XBLAS_FC_BATCH_WRAPPER(XPUTypeBF16,
                                           XPUTypeFP16,
                                           XPUTypeFP16)
DECLARE_UNSUPPORTED_XBLAS_FC_BATCH_WRAPPER(XPUTypeBF16, int32_t, XPUTypeFP16)
DECLARE_UNSUPPORTED_XBLAS_FC_BATCH_WRAPPER(XPUTypeBF16, int16_t, XPUTypeFP16)
DECLARE_UNSUPPORTED_XBLAS_FC_BATCH_WRAPPER(XPUTypeFP16, int32_t, XPUTypeFP16)
DECLARE_UNSUPPORTED_XBLAS_FC_BATCH_WRAPPER(XPUTypeBF16, int_with_ll_t, float)
DECLARE_UNSUPPORTED_XBLAS_FC_BATCH_WRAPPER(XPUTypeBF16, XPUTypeFP16, float)
DECLARE_UNSUPPORTED_XBLAS_FC_BATCH_WRAPPER(XPUTypeFP16, tfloat32, float)
DECLARE_UNSUPPORTED_XBLAS_FC_BATCH_WRAPPER(XPUTypeBF16, int32_t, float)
DECLARE_UNSUPPORTED_XBLAS_FC_BATCH_WRAPPER(XPUTypeBF16, int16_t, float)
DECLARE_UNSUPPORTED_XBLAS_FC_BATCH_WRAPPER(XPUTypeFP16, int32_t, float)

#ifndef PADDLE_WITH_XPU_XRE5
// disable in kl2
DECLARE_UNSUPPORTED_XBLAS_FC_BATCH_WRAPPER(XPUTypeBF16, tfloat32, float)
DECLARE_UNSUPPORTED_XBLAS_FC_BATCH_WRAPPER(XPUTypeBF16, float, float)
DECLARE_UNSUPPORTED_XBLAS_FC_BATCH_WRAPPER(float, XPUTypeFP16, float)
DECLARE_UNSUPPORTED_XBLAS_FC_BATCH_WRAPPER(float, XPUTypeFP16, XPUTypeFP16)
DECLARE_UNSUPPORTED_XBLAS_FC_BATCH_WRAPPER(XPUTypeFP16, XPUTypeFP16, float)
DECLARE_UNSUPPORTED_XBLAS_FC_BATCH_WRAPPER(XPUTypeFP16,
                                           XPUTypeFP16,
                                           XPUTypeFP16)
#endif

template <typename T>
static void MatMulXPUFunction(
    xpu::Context* xpu_ctx,
    const T* x,
    const T* y,
    T* out,
    const XpuFcInfo& fcinfo,
    float alpha,
    float beta = 0.f,
    bool is_grad = false,
    xpu::Activation_t act = xpu::Activation_t::LINEAR) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  int fc_calc_type = FCCalcType<XPUType>();

  decltype(&xblas_fc_wrapper<XPUType, int16_t>) xblas_fc_api_list[6] = {
      &xblas_fc_wrapper<XPUType, int16_t>,
      &xblas_fc_wrapper<XPUType, int32_t>,
      &xblas_fc_wrapper<XPUType, float>,
      &xblas_fc_wrapper<XPUType, int_with_ll_t>,
      &xblas_fc_wrapper<XPUType, tfloat32>,
      &xblas_fc_wrapper<XPUType, XPUTypeFP16>,
  };

  decltype(&xblas_fc_batch_wrapper<XPUType, int16_t, float>)
      xblas_fc_batch_api_list[6] = {
          &xblas_fc_batch_wrapper<XPUType, int16_t, float>,
          &xblas_fc_batch_wrapper<XPUType, int32_t, float>,
          &xblas_fc_batch_wrapper<XPUType, float, float>,
          &xblas_fc_batch_wrapper<XPUType, int_with_ll_t, float>,
          &xblas_fc_batch_wrapper<XPUType, tfloat32, float>,
          &xblas_fc_batch_wrapper<XPUType, XPUTypeFP16, float>,
      };

  auto xblas_fc_api = xblas_fc_api_list[fc_calc_type];

  if (std::getenv("XPU_PADDLE_FC_GRAD_LOCAL") != nullptr) {
    if (is_grad) {
      xblas_fc_api = xblas_fc_api_list[2];
    }
  }
  auto xblas_fc_batch_api = xblas_fc_batch_api_list[fc_calc_type];

  if (fc_calc_type == XPUFCCalcType::FC_FLOAT16 &&
      std::getenv("XPU_PADDLE_FC_FLOAT16") != nullptr) {
    xblas_fc_batch_api =
        &xblas_fc_batch_wrapper<XPUType, XPUTypeFP16, XPUTypeFP16>;
  }
  int m = fcinfo.m;
  int n = fcinfo.n;
  int k = fcinfo.k;
  int batch_size = fcinfo.bs;
  int ldx = fcinfo.stride_x;
  int ldy = fcinfo.stride_y;
  int ldout = fcinfo.stride_out;
  bool trans_x = fcinfo.trans_x;
  bool trans_y = fcinfo.trans_y;
  float* max_x = fcinfo.max_x;
  float* max_y = fcinfo.max_y;
  float* max_out = fcinfo.max_out;
  bool is_x_need_broadcast = fcinfo.is_x_need_broadcast;
  bool is_y_need_broadcast = fcinfo.is_y_need_broadcast;
  const float* bias = fcinfo.bias;
  const float* scale_x = fcinfo.scale_x;
  const float* scale_y = fcinfo.scale_y;
  int scale_x_mode = fcinfo.scale_x_mode;
  int scale_y_mode = fcinfo.scale_y_mode;

  xpu::ctx_guard RAII_GUARD(xpu_ctx);
  if (batch_size <= 1) {
    xblas_fc_api(xpu_ctx,
                 reinterpret_cast<const XPUType*>(x),
                 reinterpret_cast<const XPUType*>(y),
                 reinterpret_cast<XPUType*>(out),
                 m,
                 n,
                 k,
                 trans_x,
                 trans_y,
                 max_x,
                 max_y,
                 max_out,
                 ldx,
                 ldy,
                 ldout,
                 alpha,
                 beta,
                 bias,
                 act,
                 scale_x,
                 scale_y,
                 scale_x_mode,
                 scale_y_mode);
  } else {
    const XPUType* x_data = reinterpret_cast<const XPUType*>(x);
    if (is_x_need_broadcast) {
      XPUType* x_broadcast_data = nullptr;
      x_broadcast_data = RAII_GUARD.alloc_l3_or_gm<XPUType>(batch_size * m * k);
      PADDLE_ENFORCE_XDNN_NOT_NULL(x_broadcast_data);
      std::vector<int> x_shape = {1, m, k};
      std::vector<int> new_x_shape = {batch_size, m, k};
      int r = xpu::broadcast<XPUType>(
          xpu_ctx, x_data, x_broadcast_data, x_shape, new_x_shape);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast");
      x_data = x_broadcast_data;
    }
    const XPUType* y_data = reinterpret_cast<const XPUType*>(y);
    if (is_y_need_broadcast) {
      XPUType* y_broadcast_data = nullptr;
      y_broadcast_data = RAII_GUARD.alloc_l3_or_gm<XPUType>(batch_size * k * n);
      PADDLE_ENFORCE_XDNN_NOT_NULL(y_broadcast_data);
      std::vector<int> y_shape = {1, k, n};
      std::vector<int> new_y_shape = {batch_size, k, n};
      int r = xpu::broadcast<XPUType>(
          xpu_ctx, y_data, y_broadcast_data, y_shape, new_y_shape);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast");
      y_data = y_broadcast_data;
    }
    // batch matmul
    xblas_fc_batch_api(xpu_ctx,                          // Context* ctx,
                       batch_size,                       // int batch_size,
                       trans_x,                          // bool x_trans,
                       trans_y,                          // bool w_trans,
                       m,                                // int m,
                       n,                                // int n,
                       k,                                // int k,
                       alpha,                            // float alpha,
                       x_data,                           // const TX* x,
                       ldx,                              // int stride_a,
                       y_data,                           // const TW* w,
                       ldy,                              // int stride_b,
                       beta,                             // float beta,
                       reinterpret_cast<XPUType*>(out),  // TY* y,
                       ldout,                            // int stride_c,
                       max_x,   // const float* x_maxptr,
                       max_y);  // const float* w_maxptr
  }
}

template <typename T>
static std::tuple<XpuFcInfo, XpuFcInfo, const T*, const T*, const T*, const T*>
MatmulGradFcInfo(xpu::Context* xpu_ctx,
                 xpu::ctx_guard* RAII_GUARD,
                 const XpuFcInfo& dout_shape,
                 bool trans_x,
                 bool trans_y,
                 const T* x,
                 const T* y,
                 const T* dout) {
  XpuFcInfo dx_shape, dy_shape;
  const T* dx_a = NULL;
  const T* dx_b = NULL;
  const T* dy_a = NULL;
  const T* dy_b = NULL;
  bool copy_to_l3 = false;
  float* max_dout = NULL;
  int maxptr_size = xpu_ctx->max_ptr_size();
  uint64_t l3_size = uint64_t(xpu_ctx->_l3_mgr.get_size());
  int bs = (dout_shape.bs <= 1) ? (1) : (dout_shape.bs);
  int dx_size = bs * dout_shape.m * dout_shape.k;
  int dy_size = bs * dout_shape.k * dout_shape.n;
  int dout_size = bs * dout_shape.m * dout_shape.n;
  if (trans_x && trans_y) {
    copy_to_l3 = l3_size >= (dout_size * 2 + dy_size) * sizeof(T);
  } else if (trans_x) {
    copy_to_l3 = l3_size >= dout_size * sizeof(T);
  } else if (trans_y) {
    copy_to_l3 = l3_size >= dout_size * 2 * sizeof(T);
  } else {
    copy_to_l3 = l3_size >= (dout_size + dx_size) * sizeof(T);
  }

  const T* dout_new = dout;
  int r = 0;
  if (copy_to_l3) {
    T* dout_l3 = RAII_GUARD->alloc_l3<T>(dout_size);
    PADDLE_ENFORCE_XDNN_NOT_NULL(dout_l3);
    if ((dout_shape.bs > 1) || ((dout_shape.bs <= 1) &&
                                (FCCalcType<T>() == XPUFCCalcType::FC_FLOAT))) {
      r = xpu::copy(xpu_ctx, dout, dout_l3, dout_size);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "copy");
      dout_new = dout_l3;
    } else {
      max_dout = RAII_GUARD->alloc_l3_or_gm<float>(maxptr_size);
      PADDLE_ENFORCE_XDNN_NOT_NULL(max_dout);

      r = xpu::findmax_copy_fusion(xpu_ctx, dout, max_dout, dout_l3, dout_size);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "findmax_copy_fusion");
      dout_new = dout_l3;
    }
  } else if (((dout_shape.bs <= 1) &&
              (FCCalcType<T>() != XPUFCCalcType::FC_FLOAT))) {
    max_dout = RAII_GUARD->alloc_l3_or_gm<float>(maxptr_size);
    PADDLE_ENFORCE_XDNN_NOT_NULL(max_dout);
    r = xpu::findmax_copy_fusion(
        xpu_ctx, dout, max_dout, reinterpret_cast<T*>(NULL), dout_size);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "findmax_copy_fusion");
  }

  if (trans_x && trans_y) {
    // dx = T(y) * T(dout)
    dx_shape.InitFcInfo(dout_shape.bs,
                        dout_shape.k,
                        dout_shape.m,
                        dout_shape.n,
                        true,
                        true,
                        nullptr,
                        max_dout,
                        nullptr);
    dx_a = y, dx_b = dout_new;
    dx_shape.is_x_need_broadcast = dout_shape.is_y_need_broadcast;
    // dy = T(dout) * T(x)
    dy_shape.InitFcInfo(dout_shape.bs,
                        dout_shape.n,
                        dout_shape.k,
                        dout_shape.m,
                        true,
                        true,
                        max_dout,
                        nullptr,
                        nullptr);
    dy_a = dout_new, dy_b = x;
    dy_shape.is_y_need_broadcast = dout_shape.is_x_need_broadcast;
  } else if (trans_x) {
    // dx = y * T(dout)
    dx_shape.InitFcInfo(dout_shape.bs,
                        dout_shape.k,
                        dout_shape.m,
                        dout_shape.n,
                        false,
                        true,
                        nullptr,
                        max_dout,
                        nullptr);
    dx_a = y, dx_b = dout_new;
    dx_shape.is_x_need_broadcast = dout_shape.is_y_need_broadcast;
    // dy = x * dout
    dy_shape.InitFcInfo(dout_shape.bs,
                        dout_shape.k,
                        dout_shape.n,
                        dout_shape.m,
                        false,
                        false,
                        nullptr,
                        max_dout,
                        nullptr);
    dy_shape.is_x_need_broadcast = dout_shape.is_x_need_broadcast;
    dy_a = x, dy_b = dout_new;
  } else if (trans_y) {
    // dx = dout * y
    dx_shape.InitFcInfo(dout_shape.bs,
                        dout_shape.m,
                        dout_shape.k,
                        dout_shape.n,
                        false,
                        false,
                        max_dout,
                        nullptr,
                        nullptr);
    dx_a = dout_new, dx_b = y;
    dx_shape.is_y_need_broadcast = dout_shape.is_y_need_broadcast;
    // dy =  T(dout) * x
    dy_shape.InitFcInfo(dout_shape.bs,
                        dout_shape.n,
                        dout_shape.k,
                        dout_shape.m,
                        true,
                        false,
                        max_dout,
                        nullptr,
                        nullptr);
    dy_a = dout_new, dy_b = x;
    dy_shape.is_y_need_broadcast = dout_shape.is_x_need_broadcast;
  } else {
    // dx = dout * T(y)
    dx_shape.InitFcInfo(dout_shape.bs,
                        dout_shape.m,
                        dout_shape.k,
                        dout_shape.n,
                        false,
                        true,
                        max_dout,
                        nullptr,
                        nullptr);
    dx_a = dout_new, dx_b = y;
    dx_shape.is_y_need_broadcast = dout_shape.is_y_need_broadcast;
    // dy = T(x) * dout
    dy_shape.InitFcInfo(dout_shape.bs,
                        dout_shape.k,
                        dout_shape.n,
                        dout_shape.m,
                        true,
                        false,
                        nullptr,
                        max_dout,
                        nullptr);
    dy_shape.is_x_need_broadcast = dout_shape.is_x_need_broadcast;
    dy_a = x, dy_b = dout_new;
  }
  std::tuple<XpuFcInfo, XpuFcInfo, const T*, const T*, const T*, const T*>
      result = std::make_tuple(dx_shape, dy_shape, dx_a, dx_b, dy_a, dy_b);

  return result;
}

}  // namespace phi
#endif
