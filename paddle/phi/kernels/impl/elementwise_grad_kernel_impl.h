/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include "glog/logging.h"

#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/common/complex.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/expand_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/broadcast_function.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/elementwise_functor.h"
#include "paddle/phi/kernels/funcs/elementwise_utils.h"

namespace phi {

template <typename T, typename Context, typename GradFunc>
void AddGradImpl(const Context& dev_ctx,
                 const DenseTensor& x,
                 const DenseTensor& y,
                 const DenseTensor& out_grad,
                 int axis,
                 DenseTensor* x_grad,
                 DenseTensor* y_grad,
                 GradFunc grad_func) {
  phi::funcs::ElementwiseGradPreProcess(out_grad, x_grad);
  auto* out = &out_grad;
  // Special case when y_grad is not needed and x_grad doesn't reduce
  if (x_grad != nullptr && y_grad == nullptr &&
      x_grad->dims() == out_grad.dims()) {
    VLOG(4) << "Special case when y_grad is not needed and x_grad doesn't "
               "reduce";
    phi::Copy(dev_ctx, out_grad, dev_ctx.GetPlace(), false, x_grad);
  } else if (x_grad == nullptr && y_grad != nullptr &&
             y_grad->dims() == out_grad.dims()) {
    VLOG(4) << "Special case when x_grad is not needed and y_grad doesn't "
               "reduce";
    phi::Copy(dev_ctx, out_grad, dev_ctx.GetPlace(), false, y_grad);
  } else {
    grad_func(dev_ctx, x, y, *out, out_grad, x_grad, y_grad, axis);
  }
}

template <typename T, typename Context>
void AddDoubleGradImpl(const Context& dev_ctx,
                       const DenseTensor& y,
                       const paddle::optional<DenseTensor>& ddx,
                       const paddle::optional<DenseTensor>& ddy,
                       const DenseTensor& dout,
                       int axis,
                       DenseTensor* ddout) {
  // ddOut = ddx + ddy
  if (ddout) {
    auto* ddx_tensor = ddx.get_ptr();
    auto* ddy_tensor = ddy.get_ptr();
    auto out_shape = dout.dims();
    dev_ctx.template Alloc<T>(ddout);
    if (ddx_tensor == nullptr && ddy_tensor == nullptr) {
      VLOG(4) << "Special case when ddx and ddy are not needed \n";
      ddout = nullptr;
    } else if (ddx_tensor == nullptr && ddy_tensor != nullptr) {
      if (ddy_tensor->dims() != out_shape) {
        VLOG(4) << "Special case when ddx is not needed and ddy needs to "
                   "broadcast\n";
        std::vector<const DenseTensor*> ins = {ddy_tensor};
        std::vector<DenseTensor*> outs = {ddout};
        ExpandKernel<T, Context>(dev_ctx,
                                 *ddy_tensor,
                                 IntArray{phi::vectorize<int64_t>(out_shape)},
                                 ddout);
      } else {
        VLOG(4) << "Special case when ddx is not needed and ddy doesn't need "
                   "to broadcast\n";
        phi::Copy(dev_ctx, *ddy_tensor, dev_ctx.GetPlace(), false, ddout);
      }
    } else if (ddx_tensor != nullptr && ddy_tensor == nullptr) {
      if (ddx_tensor->dims() != out_shape) {
        VLOG(4) << "Special case when ddy is not needed and ddx need to "
                   "broadcast\n";
        std::vector<const DenseTensor*> ins = {ddx_tensor};
        std::vector<DenseTensor*> outs = {ddout};
        ExpandKernel<T, Context>(dev_ctx,
                                 *ddx_tensor,
                                 IntArray{phi::vectorize<int64_t>(out_shape)},
                                 ddout);
      } else {
        VLOG(4) << "Special case when ddx is not needed and ddy doesn't need "
                   "to broadcast\n";
        phi::Copy(dev_ctx, *ddx_tensor, dev_ctx.GetPlace(), false, ddout);
      }
    } else {
      auto ddx_dims = ddx_tensor->dims();
      auto ddy_dims = ddy_tensor->dims();
      if (ddx_dims.size() >= ddy_dims.size()) {
        funcs::ElementwiseCompute<funcs::AddFunctor<T>, T>(
            dev_ctx,
            *ddx_tensor,
            *ddy_tensor,
            funcs::AddFunctor<T>(),
            ddout,
            axis);
      } else {
        funcs::ElementwiseCompute<funcs::InverseAddFunctor<T>, T>(
            dev_ctx,
            *ddx_tensor,
            *ddy_tensor,
            funcs::InverseAddFunctor<T>(),
            ddout,
            axis);
      }
    }
  }
}

template <typename T, typename Context>
void SubtractDoubleGradImpl(const Context& dev_ctx,
                            const DenseTensor& y,
                            const paddle::optional<DenseTensor>& ddx,
                            const paddle::optional<DenseTensor>& ddy,
                            const DenseTensor& dout,
                            int axis,
                            DenseTensor* ddout) {
  // DDOut = ddx - ddy
  if (ddout) {
    DenseTensor ddx_safe, ddy_safe;
    funcs::GetDoubleGradSafeTensor<Context, T>(
        dev_ctx, dout, ddx.get_ptr(), &ddx_safe);
    funcs::GetDoubleGradSafeTensor<Context, T>(
        dev_ctx, y, ddy.get_ptr(), &ddy_safe);

    dev_ctx.template Alloc<T>(ddout);
    funcs::ElementwiseCompute<funcs::SubtractFunctor<T>, T>(
        dev_ctx, ddx_safe, ddy_safe, funcs::SubtractFunctor<T>(), ddout, axis);
  }
}

/*
******************************
    Divide Grad
******************************
*/

template <typename T>
struct DivGradDX {
  HOSTDEVICE T operator()(T x UNUSED, T y, T out UNUSED, T dout) const {
    return dout / y;
  }
};

template <typename T>
struct DivGradDX<phi::dtype::complex<T>> {
  HOSTDEVICE phi::dtype::complex<T> operator()(
      phi::dtype::complex<T> x UNUSED,
      phi::dtype::complex<T> y,
      phi::dtype::complex<T> out UNUSED,
      phi::dtype::complex<T> dout) const {
    phi::dtype::complex<T> y_conj(y.real, -y.imag);
    return dout / y_conj;
  }
};

template <typename T>
struct DivGradDY {
  HOSTDEVICE T operator()(T x UNUSED, T y, T out, T dout) const {
    return -dout * out / y;
  }
};

template <typename T>
struct DivGradDY<phi::dtype::complex<T>> {
  HOSTDEVICE phi::dtype::complex<T> operator()(
      phi::dtype::complex<T> x UNUSED,
      phi::dtype::complex<T> y,
      phi::dtype::complex<T> out,
      phi::dtype::complex<T> dout) const {
    phi::dtype::complex<T> out_div_y_conj((out / y).real, -(out / y).imag);
    return -dout * out_div_y_conj;
  }
};

template <typename T>
struct DivDoubleDY {
  HOSTDEVICE T operator()(const T& x,
                          const T& y,
                          const T& out,
                          const T& dout) const {
    return (y * out - x) * dout;
  }
};

template <typename T>
struct DivDoubleDY_Only_DDY {
  HOSTDEVICE T operator()(const T& x,
                          const T& y,
                          const T& out,
                          const T& dout) const {
    return y * out * dout;
  }
};

template <typename T>
struct DivDoubleDY_Only_DDX {
  HOSTDEVICE T operator()(const T& x,
                          const T& y,
                          const T& out,
                          const T& dout) const {
    return -x * dout;
  }
};

// ddOut = ddX / Y - Out * ddY / Y = (ddX - Out * ddY) / Y
template <typename T>
struct DivDoubleDDOut {
  HOSTDEVICE T operator()(const T& ddx,
                          const T& ddy,
                          const T& y,
                          const T& out) const {
    return (ddx - out * ddy) / y;
  }
};

template <typename T>
struct DivDoubleDDOut_Only_DDY {
  HOSTDEVICE T operator()(const T& ddx UNUSED,
                          const T& ddy,
                          const T& y,
                          const T& out) const {
    return -out * ddy / y;
  }
};

template <typename T, typename DDout_OP, typename OutType = T>
void ComputeDDoutWithoutBroadcast(const CPUContext& dev_ctx UNUSED,
                                  const phi::DenseTensor& ddx,
                                  const phi::DenseTensor& ddy,
                                  const phi::DenseTensor& y,
                                  const phi::DenseTensor& out,
                                  phi::DenseTensor* ddout,
                                  DDout_OP dout_op) {
  auto out_numel = out.numel();
  auto* ddx_data = ddx.data<T>();
  auto* ddy_data = ddy.data<T>();
  auto* y_data = y.data<T>();
  auto* out_data = out.data<T>();
  auto* ddout_data = ddout->data<T>();
  for (int i = 0; i < out_numel; i++) {
    ddout_data[i] = dout_op(ddx_data[i], ddy_data[i], y_data[i], out_data[i]);
  }
}

template <typename T, typename DDout_OP, typename OutType = T>
void ComputeDDoutWithBroadcast(const CPUContext& dev_ctx UNUSED,
                               const phi::DenseTensor& ddx,
                               const phi::DenseTensor& ddy,
                               const phi::DenseTensor& y,
                               const phi::DenseTensor& out,
                               phi::DenseTensor* ddout,
                               const int* x_dims_array,
                               const int* y_dims_array,
                               const int* out_dims_array,
                               const int max_dim,
                               DDout_OP dout_op) {
  auto out_numel = out.numel();
  auto* ddx_data = ddx.data<T>();
  auto* ddy_data = ddy.data<T>();
  auto* y_data = y.data<T>();
  auto* out_data = out.data<T>();
  auto* ddout_data = ddout->data<T>();
  std::vector<int> index_array(max_dim, 0);
  for (int i = 0; i < out_numel; i++) {
    int x_index = phi::funcs::GetElementwiseIndex(
        x_dims_array, max_dim, index_array.data());
    int y_index = phi::funcs::GetElementwiseIndex(
        y_dims_array, max_dim, index_array.data());
    ddout_data[i] = dout_op(
        ddx_data[x_index], ddy_data[y_index], y_data[y_index], out_data[i]);
    phi::funcs::UpdateElementwiseIndexArray(
        out_dims_array, max_dim, index_array.data());
  }
}

#if defined(__NVCC__) || defined(__HIPCC__)

/*
Since __global__ does not allow std::vector as a type parameter,
a custom CudaIntArray is used to pass an array containing a small number(<=8) of
integers, e.g. pass an shape array(rank<=8) to a kernel function.
*/
#define MAX_SIZE 8
#define STR(x) #x
#define XSTR(x) STR(x)

struct CudaIntArray {
  int a0, a1, a2, a3, a4, a5, a6, a7;

  CudaIntArray(const int& a0_,
               const int& a1_,
               const int& a2_,
               const int& a3_,
               const int& a4_,
               const int& a5_,
               const int& a6_,
               const int& a7_)
      : a0(a0_),
        a1(a1_),
        a2(a2_),
        a3(a3_),
        a4(a4_),
        a5(a5_),
        a6(a6_),
        a7(a7_) {}

  __device__ __host__ int operator[](const int64_t& idx) const {
#if defined(__CUDA_ARCH__) || defined(__HIPCC__)
    assert(0 <= idx && idx < MAX_SIZE);
#endif

    switch (idx) {
      case 0:
        return a0;
      case 1:
        return a1;
      case 2:
        return a2;
      case 3:
        return a3;
      case 4:
        return a4;
      case 5:
        return a5;
      case 6:
        return a6;
      case 7:
        return a7;
      default:
        return 0;
    }
  }
};

CudaIntArray initCudaIntArray(const int* vec, const int& size) {
  PADDLE_ENFORCE_LE(
      size,
      MAX_SIZE,
      common::errors::OutOfRange(
          "Given size to init CudaIntArray must be less than" XSTR(MAX_SIZE)));
  PADDLE_ENFORCE_GT(
      size,
      0,
      common::errors::OutOfRange(
          "Given size to init CudaIntArray must be greater than 0"));
  return CudaIntArray(size > 0 ? vec[0] : 0,
                      size > 1 ? vec[1] : 0,
                      size > 2 ? vec[2] : 0,
                      size > 3 ? vec[3] : 0,
                      size > 4 ? vec[4] : 0,
                      size > 5 ? vec[5] : 0,
                      size > 6 ? vec[6] : 0,
                      size > 7 ? vec[7] : 0);
}

template <typename T, typename DDout_OP, typename OutType = T>
__global__ void ComputeDDoutWithoutBroadcastGPUKernel(const T* ddx_data,
                                                      const T* ddy_data,
                                                      const T* y_data,
                                                      const T* out_data,
                                                      T* ddout_data,
                                                      int numel,
                                                      DDout_OP dout_op) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= numel) return;
  ddout_data[tid] =
      dout_op(ddx_data[tid], ddy_data[tid], y_data[tid], out_data[tid]);
}

template <typename T, typename DDout_OP, typename OutType = T>
void ComputeDDoutWithoutBroadcast(const GPUContext& dev_ctx UNUSED,
                                  const phi::DenseTensor& ddx,
                                  const phi::DenseTensor& ddy,
                                  const phi::DenseTensor& y,
                                  const phi::DenseTensor& out,
                                  phi::DenseTensor* ddout,
                                  DDout_OP dout_op) {
  auto out_numel = out.numel();
  auto* ddx_data = ddx.data<T>();
  auto* ddy_data = ddy.data<T>();
  auto* y_data = y.data<T>();
  auto* out_data = out.data<T>();
  auto* ddout_data = ddout->data<T>();
  int block = 512;
  int64_t grid = (out_numel + block - 1) / block;
  auto stream = reinterpret_cast<const phi::GPUContext&>(dev_ctx).stream();
  ComputeDDoutWithoutBroadcastGPUKernel<T, DDout_OP, T>
      <<<grid, block, 0, stream>>>(
          ddx_data, ddy_data, y_data, out_data, ddout_data, out_numel, dout_op);
}

template <typename T, typename DDout_OP, typename OutType = T>
__global__ void ComputeDDoutWithBroadcastGPUKernel(
    const T* ddx_data,
    const T* ddy_data,
    const T* y_data,
    const T* out_data,
    T* ddout_data,
    int numel,
    const CudaIntArray x_dims_array,
    const CudaIntArray y_dims_array,
    const CudaIntArray out_dims_array,
    const int max_dim,
    DDout_OP dout_op) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= numel) return;
  int x_index = 0, y_index = 0, x_index_prod = 1, y_index_prod = 1,
      out_index = tid, dim_index;
  for (int64_t i = max_dim - 1; i >= 0; i--) {
    if (out_index == 0) break;
    dim_index = out_index % out_dims_array[i];
    out_index = out_index / out_dims_array[i];
    if (x_dims_array[i] > 1) {
      x_index += dim_index * x_index_prod;
      x_index_prod *= x_dims_array[i];
    }
    if (y_dims_array[i] > 1) {
      y_index += dim_index * y_index_prod;
      y_index_prod *= y_dims_array[i];
    }
  }
  ddout_data[tid] = dout_op(
      ddx_data[x_index], ddy_data[y_index], y_data[y_index], out_data[tid]);
}

template <typename T, typename DDout_OP, typename OutType = T>
void ComputeDDoutWithBroadcast(const GPUContext& dev_ctx UNUSED,
                               const phi::DenseTensor& ddx,
                               const phi::DenseTensor& ddy,
                               const phi::DenseTensor& y,
                               const phi::DenseTensor& out,
                               phi::DenseTensor* ddout,
                               const int* x_dims_array,
                               const int* y_dims_array,
                               const int* out_dims_array,
                               const int max_dim,
                               DDout_OP dout_op) {
  auto out_numel = out.numel();
  auto* ddx_data = ddx.data<T>();
  auto* ddy_data = ddy.data<T>();
  auto* y_data = y.data<T>();
  auto* out_data = out.data<T>();
  auto* ddout_data = ddout->data<T>();

  // Use the lightweight `CudaIntArray` structure to avoid unnecessary copy time
  // caused by `cudaMemcpy` or `cudaMemcpyAsync`.
  CudaIntArray x_dims_array_gpu_data = initCudaIntArray(x_dims_array, max_dim);
  CudaIntArray y_dims_array_gpu_data = initCudaIntArray(y_dims_array, max_dim);
  CudaIntArray out_dims_array_gpu_data =
      initCudaIntArray(out_dims_array, max_dim);

  int block = 512;
  int64_t grid = (out_numel + block - 1) / block;
  auto stream = reinterpret_cast<const phi::GPUContext&>(dev_ctx).stream();
  ComputeDDoutWithBroadcastGPUKernel<T, DDout_OP, T>
      <<<grid, block, 0, stream>>>(ddx_data,
                                   ddy_data,
                                   y_data,
                                   out_data,
                                   ddout_data,
                                   out_numel,
                                   x_dims_array_gpu_data,
                                   y_dims_array_gpu_data,
                                   out_dims_array_gpu_data,
                                   max_dim,
                                   dout_op);
}

#endif

template <typename DeviceContext,
          typename T,
          typename DDout_OP,
          typename Tout = T>
void DivDoubleDDoutCompute(const DeviceContext& dev_ctx,
                           const phi::DenseTensor& ddx,
                           const phi::DenseTensor& ddy,
                           const phi::DenseTensor& y,
                           const phi::DenseTensor& out,
                           int axis,
                           phi::DenseTensor* ddout,
                           DDout_OP dout_op) {
  auto x_dims = ddx.dims();
  auto y_dims = ddy.dims();
  if (x_dims == y_dims) {
    ComputeDDoutWithoutBroadcast<T, DDout_OP, T>(
        dev_ctx, ddx, ddy, y, out, ddout, dout_op);
  } else {
    int max_dim = std::max(x_dims.size(), y_dims.size());
    axis = (axis == -1 ? std::abs(x_dims.size() - y_dims.size()) : axis);
    std::vector<int> x_dims_array(max_dim, 0);
    std::vector<int> y_dims_array(max_dim, 0);
    std::vector<int> out_dims_array(max_dim, 0);
    phi::funcs::GetBroadcastDimsArrays(x_dims,
                                       y_dims,
                                       x_dims_array.data(),
                                       y_dims_array.data(),
                                       out_dims_array.data(),
                                       max_dim,
                                       axis);
    ComputeDDoutWithBroadcast<T, DDout_OP, T>(dev_ctx,
                                              ddx,
                                              ddy,
                                              y,
                                              out,
                                              ddout,
                                              x_dims_array.data(),
                                              y_dims_array.data(),
                                              out_dims_array.data(),
                                              max_dim,
                                              dout_op);
  }
}

template <typename T, typename Context>
void DivideDoubleGradKernel(const Context& dev_ctx,
                            const DenseTensor& y,
                            const DenseTensor& out,
                            const DenseTensor& grad_out,
                            const paddle::optional<DenseTensor>& dx,
                            const paddle::optional<DenseTensor>& ddx,
                            const paddle::optional<DenseTensor>& ddy,
                            int axis,
                            DenseTensor* dy,
                            DenseTensor* dout,
                            DenseTensor* ddout) {
  auto* ddx_tensor = ddx.get_ptr();
  auto* ddy_tensor = ddy.get_ptr();
  auto* dx_tensor = dx.get_ptr();
  DenseTensor dz_div_y;
  if ((dy || dout) && (!dx_tensor || dx_tensor->dims() != out.dims())) {
    dz_div_y.Resize(out.dims());
    dev_ctx.template Alloc<T>(&dz_div_y);
    funcs::DefaultElementwiseOperator<Context,
                                      T,
                                      funcs::DivideFunctor<T>,
                                      funcs::InverseDivideFunctor<T>>(
        dev_ctx, grad_out, y, &dz_div_y, axis);
    dx_tensor = &dz_div_y;
  }
  // ddOut = ddX / Y - Out * ddY / Y = (ddX - Out * ddY) / Y
  // dY = Out * dX * ddY / Y - dX * ddX / Y
  // dOut = - dX * ddY
  // To save memory, (1) dout can be used as 'tmp' tensor, (2) ddout can
  // inplace ddx
  DenseTensor tmp;
  if (dout) {
    dout->Resize(out.dims());
    dev_ctx.template Alloc<T>(dout);
    tmp = *dout;
  } else {
    tmp.Resize(out.dims());
    dev_ctx.template Alloc<T>(&tmp);
  }
  if (dy) {
    dy->Resize(y.dims());
    dev_ctx.template Alloc<T>(dy);
    if (!ddx_tensor && !ddy_tensor) {
      FullLikeKernel<T, Context>(
          dev_ctx, y, Scalar(static_cast<T>(0.0)), y.dtype(), dy);
    } else {
      // pre-compute 'dX / Y' into 'tmp' for 'ddout' and/or 'dy'
      funcs::DefaultElementwiseOperator<Context,
                                        T,
                                        funcs::DivideFunctor<T>,
                                        funcs::InverseDivideFunctor<T>>(
          dev_ctx, *dx_tensor, y, &tmp, axis);
      if (ddx_tensor && !ddy_tensor) {
        // dy = -dX * ddX / Y
        phi::funcs::ElemwiseGradCompute<Context,
                                        T,
                                        DivGradDX<T>,
                                        DivDoubleDY_Only_DDX<T>>(
            dev_ctx,
            *ddx_tensor,  // ddx
            y,
            out,  // out
            tmp,  // dX /Y
            axis,
            nullptr,
            dy,
            DivGradDX<T>(),
            DivDoubleDY_Only_DDX<T>());
      } else if (!ddx_tensor && ddy_tensor) {
        // dY = Out * dX * ddY / Y
        phi::funcs::ElemwiseGradCompute<Context,
                                        T,
                                        DivGradDX<T>,
                                        DivDoubleDY_Only_DDY<T>>(
            dev_ctx,
            *dx_tensor,
            *ddy_tensor,  // ddy
            out,          // out
            tmp,          // dX / Y
            axis,
            nullptr,
            dy,
            DivGradDX<T>(),
            DivDoubleDY_Only_DDY<T>());
      } else {
        // dY = Out * dX * ddY / Y - dX * ddX / Y

        // NOTE(dengkaipeng): in the following ElemwiseGradCompute, for the
        // first output tensor is nullptr, the branch to calculate first
        // output tensor will not be activated, DivGradDx function will not
        // be called and can be ignored, the first branch has little effect
        // on running speed.
        phi::funcs::
            ElemwiseGradCompute<Context, T, DivGradDX<T>, DivDoubleDY<T>>(
                dev_ctx,
                *ddx_tensor,  // ddx
                *ddy_tensor,  // ddy
                out,          // out
                tmp,          // dX / Y
                axis,
                nullptr,
                dy,
                DivGradDX<T>(),
                DivDoubleDY<T>());
      }
    }
  }

  if (ddout) {
    ddout->Resize(out.dims());
    dev_ctx.template Alloc<T>(ddout);
    // ddOut = ddX / Y - Out * ddY / Y = (ddX - Out * ddY) / Y
    if (!ddx_tensor && !ddy_tensor) {
      FullLikeKernel<T, Context>(
          dev_ctx, out, Scalar(static_cast<T>(0.0)), out.dtype(), ddout);
    } else if (ddx_tensor != nullptr && ddy_tensor == nullptr) {
      // ddOut = ddX / Y
      funcs::DefaultElementwiseOperator<Context,
                                        T,
                                        funcs::DivideFunctor<T>,
                                        funcs::InverseDivideFunctor<T>>(
          dev_ctx, *ddx_tensor, y, ddout, axis);
    } else if (!ddx_tensor && ddy_tensor) {
// ddOut = - Out * ddY / Y
#if defined(__xpu__)
      funcs::DefaultElementwiseOperator<Context,
                                        T,
                                        funcs::MultiplyFunctor<T>,
                                        funcs::InverseMultiplyFunctor<T>>(
          dev_ctx, out, *ddy_tensor, &tmp, axis);
      funcs::DefaultElementwiseOperator<Context,
                                        T,
                                        funcs::DivideFunctor<T>,
                                        funcs::InverseDivideFunctor<T>>(
          dev_ctx, tmp, y, ddout, axis);
      auto& place = *dev_ctx.eigen_device();
      auto ddout_result = phi::EigenVector<T>::Flatten(*ddout);
      ddout_result.device(place) = static_cast<T>(-1) * ddout_result;
#else
      DivDoubleDDoutCompute<Context, T, DivDoubleDDOut_Only_DDY<T>, T>(
          dev_ctx,
          *dx_tensor,
          *ddy_tensor,
          y,
          out,
          axis,
          ddout,
          DivDoubleDDOut_Only_DDY<T>());
#endif
    } else {
#if defined(__xpu__)
      funcs::DefaultElementwiseOperator<Context,
                                        T,
                                        funcs::MultiplyFunctor<T>,
                                        funcs::InverseMultiplyFunctor<T>>(
          dev_ctx, out, *ddy_tensor, &tmp, axis);
      funcs::DefaultElementwiseOperator<Context,
                                        T,
                                        funcs::SubtractFunctor<T>,
                                        funcs::InverseSubtractFunctor<T>>(
          dev_ctx, *ddx_tensor, tmp, &tmp, axis);
      funcs::DefaultElementwiseOperator<Context,
                                        T,
                                        funcs::DivideFunctor<T>,
                                        funcs::InverseDivideFunctor<T>>(
          dev_ctx, tmp, y, ddout, axis);
#else
      DivDoubleDDoutCompute<Context, T, DivDoubleDDOut<T>, T>(
          dev_ctx,
          *ddx_tensor,
          *ddy_tensor,
          y,
          out,
          axis,
          ddout,
          DivDoubleDDOut<T>());
#endif
    }
  }

  if (dout) {
    if (!ddy_tensor) {
      FullLikeKernel<T, Context>(
          dev_ctx, out, Scalar(static_cast<T>(0.0)), out.dtype(), dout);
    } else {
      // dOut = - dX * ddY
      funcs::DefaultElementwiseOperator<Context,
                                        T,
                                        funcs::MultiplyFunctor<T>,
                                        funcs::InverseMultiplyFunctor<T>>(
          dev_ctx, *dx_tensor, *ddy_tensor, dout, axis);
      auto& place = *dev_ctx.eigen_device();
      auto dout_result = phi::EigenVector<T>::Flatten(*dout);
      dout_result.device(place) = static_cast<T>(-1) * dout_result;
    }
  }
}

template <typename T, typename Context>
void ElementwiseFMaxGradKernel(const Context& dev_ctx,
                               const DenseTensor& x,
                               const DenseTensor& y,
                               const DenseTensor& out_grad,
                               DenseTensor* x_grad,
                               DenseTensor* y_grad) {
  funcs::ElementwiseGradPreProcess(out_grad, x_grad);

  auto out = out_grad;  // Fake out, not used
  auto x_dim = x.dims();
  auto y_dim = y.dims();
  int axis = -1;
  if (x.dims() == y.dims()) {
    funcs::ElemwiseGradComputeNoBroadcast<Context,
                                          T,
                                          funcs::FMaxGradDx<T>,
                                          funcs::FMaxGradDy<T>>(
        dev_ctx,
        x_dim,
        y_dim,
        x,
        y,
        out,
        out_grad,
        axis,
        x_grad,
        y_grad,
        funcs::FMaxGradDx<T>(),
        funcs::FMaxGradDy<T>());
  } else {
    funcs::ElemwiseGradComputeWithBroadcast<T,
                                            funcs::FMaxGradDx<T>,
                                            funcs::FMaxGradDy<T>>(
        dev_ctx,
        x_dim,
        y_dim,
        x,
        y,
        out,
        out_grad,
        axis,
        x_grad,
        y_grad,
        funcs::FMaxGradDx<T>(),
        funcs::FMaxGradDy<T>());
  }
}

template <typename T, typename Context>
void ElementwiseFMinGradKernel(const Context& dev_ctx,
                               const DenseTensor& x,
                               const DenseTensor& y,
                               const DenseTensor& out_grad,
                               DenseTensor* x_grad,
                               DenseTensor* y_grad) {
  funcs::ElementwiseGradPreProcess(out_grad, x_grad);
  auto out = out_grad;  // Fake out, not used
  auto x_dim = x.dims();
  auto y_dim = y.dims();
  int axis = -1;
  if (x.dims() == y.dims()) {
    funcs::ElemwiseGradComputeNoBroadcast<Context,
                                          T,
                                          funcs::FMinGradDx<T>,
                                          funcs::FMinGradDy<T>>(
        dev_ctx,
        x_dim,
        y_dim,
        x,
        y,
        out,
        out_grad,
        axis,
        x_grad,
        y_grad,
        funcs::FMinGradDx<T>(),
        funcs::FMinGradDy<T>());
  } else {
    funcs::ElemwiseGradComputeWithBroadcast<T,
                                            funcs::FMinGradDx<T>,
                                            funcs::FMinGradDy<T>>(
        dev_ctx,
        x_dim,
        y_dim,
        x,
        y,
        out,
        out_grad,
        axis,
        x_grad,
        y_grad,
        funcs::FMinGradDx<T>(),
        funcs::FMinGradDy<T>());
  }
}

template <typename T>
struct MulGradDX {
  HOSTDEVICE T operator()(T x UNUSED, T y, T out UNUSED, T dout) const {
    return dout * y;
  }
};

// avoid [-Wint-in-bool-context] warning
template <>
struct MulGradDX<bool> {
  HOSTDEVICE bool operator()(bool x UNUSED,
                             bool y,
                             bool out UNUSED,
                             bool dout) const {
    return dout && y;
  }
};

template <typename T>
struct MulGradDX<phi::dtype::complex<T>> {
  HOSTDEVICE phi::dtype::complex<T> operator()(
      phi::dtype::complex<T> x UNUSED,
      phi::dtype::complex<T> y,
      phi::dtype::complex<T> out UNUSED,
      phi::dtype::complex<T> dout) const {
    phi::dtype::complex<T> y_conj(y.real, -y.imag);
    return dout * y_conj;
  }
};

/*
******************************
    Multiply Grad
******************************
*/

template <typename T>
struct MulGradDY {
  HOSTDEVICE T operator()(T x, T y UNUSED, T out UNUSED, T dout) const {
    return dout * x;
  }
};

// avoid [-Wint-in-bool-context] warning
template <>
struct MulGradDY<bool> {
  HOSTDEVICE bool operator()(bool x,
                             bool y UNUSED,
                             bool out UNUSED,
                             bool dout) const {
    return dout && x;
  }
};

template <typename T>
struct MulGradDY<phi::dtype::complex<T>> {
  HOSTDEVICE phi::dtype::complex<T> operator()(
      phi::dtype::complex<T> x,
      phi::dtype::complex<T> y UNUSED,
      phi::dtype::complex<T> out UNUSED,
      phi::dtype::complex<T> dout) const {
    phi::dtype::complex<T> x_conj(x.real, -x.imag);
    return dout * x_conj;
  }
};

template <typename T, typename Context>
void MultiplyDoubleGradKernel(const Context& dev_ctx,
                              const DenseTensor& x,
                              const DenseTensor& y,
                              const DenseTensor& dout,
                              const paddle::optional<DenseTensor>& ddx,
                              const paddle::optional<DenseTensor>& ddy,
                              int axis,
                              DenseTensor* dx,
                              DenseTensor* dy,
                              DenseTensor* ddout) {
  if (ddout) dev_ctx.template Alloc<T>(ddout);

  DenseTensor ddx_safe, ddy_safe;
  funcs::GetDoubleGradSafeTensor<Context, T>(
      dev_ctx, x, ddx.get_ptr(), &ddx_safe);
  funcs::GetDoubleGradSafeTensor<Context, T>(
      dev_ctx, y, ddy.get_ptr(), &ddy_safe);

  // dx = dout * ddy
  // dy = dout * ddx
  // ddout = ddx * y + x * ddy
  // change computation sequence to save memory, so ddout can inplace ddx and
  // dx can be used as 'tmp' tensor
  // (1) dx = x * ddy
  // (2) dy = dout * ddx
  // (3) ddout = ddx * y
  // (4) ddout = ddout + dx
  // (5) dx = dout * ddy
  if (ddout) {
    auto& place = *dev_ctx.eigen_device();
    // size(ddout) > size(ddx) or we don't have ddx, ddout can't use memory of
    // ddx using inplace

    bool without_ddx = (ddx.get_ptr() == nullptr);
    if (!without_ddx) {
      without_ddx = (ddout->numel() > ddx.get_ptr()->numel());
    }
    if (without_ddx) {
      phi::funcs::ElemwiseGradCompute<Context, T, MulGradDX<T>, MulGradDY<T>>(
          dev_ctx,
          ddx_safe,
          ddy_safe,
          dout,
          dout,
          axis,
          dx,
          dy,
          MulGradDX<T>(),
          MulGradDY<T>());

      DenseTensor ddout_tmp;
      ddout_tmp.Resize(ddout->dims());
      dev_ctx.template Alloc<T>(&ddout_tmp);

      funcs::DefaultElementwiseOperator<Context,
                                        T,
                                        funcs::MultiplyFunctor<T>,
                                        funcs::InverseMultiplyFunctor<T>>(
          dev_ctx, y, ddx_safe, ddout, axis);

      funcs::DefaultElementwiseOperator<Context,
                                        T,
                                        funcs::MultiplyFunctor<T>,
                                        funcs::InverseMultiplyFunctor<T>>(
          dev_ctx, ddy_safe, x, &ddout_tmp, axis);

      auto ddout_t = phi::EigenVector<T>::Flatten(*ddout);
      auto ddout_tmp_t = phi::EigenVector<T>::Flatten(ddout_tmp);
      ddout_t.device(place) = ddout_t + ddout_tmp_t;
    } else {
      // use dx to save memory, other than alloc tmp tensor
      if (dx) {
        DenseTensor* ddout_tmp = dx;
        funcs::DefaultElementwiseOperator<Context,
                                          T,
                                          funcs::MultiplyFunctor<T>,
                                          funcs::InverseMultiplyFunctor<T>>(
            dev_ctx, x, ddy_safe, ddout_tmp, axis);

        // NOTE: in the following ElemwiseGradCompute, for the
        // first output tensor is nullptr, the branch to calculate first
        // output tensor will not be activated, DivGradDx function will not
        // be called and can be ignored, the first branch has little effect
        // on running speed.
        phi::funcs::ElemwiseGradCompute<Context, T, MulGradDX<T>, MulGradDY<T>>(
            dev_ctx,
            ddx_safe,
            ddy_safe,
            dout,
            dout,
            axis,
            nullptr,
            dy,
            MulGradDX<T>(),
            MulGradDY<T>());

        funcs::DefaultElementwiseOperator<Context,
                                          T,
                                          funcs::MultiplyFunctor<T>,
                                          funcs::InverseMultiplyFunctor<T>>(
            dev_ctx, ddx_safe, y, ddout, axis);

        auto ddout_t = phi::EigenVector<T>::Flatten(*ddout);
        auto ddout_tmp_t = phi::EigenVector<T>::Flatten(*ddout_tmp);
        ddout_t.device(place) = ddout_t + ddout_tmp_t;

        funcs::DefaultElementwiseOperator<Context,
                                          T,
                                          funcs::MultiplyFunctor<T>,
                                          funcs::InverseMultiplyFunctor<T>>(
            dev_ctx, dout, ddy_safe, dx, axis);

      } else if ((!dx) && dy) {
        DenseTensor tmp_a(ddout->dtype());
        tmp_a.Resize(ddout->dims());

        dev_ctx.template Alloc<T>(&tmp_a);
        funcs::DefaultElementwiseOperator<Context,
                                          T,
                                          funcs::MultiplyFunctor<T>,
                                          funcs::InverseMultiplyFunctor<T>>(
            dev_ctx, x, ddy_safe, &tmp_a, axis);

        auto ddout_t1 = phi::EigenVector<T>::Flatten(tmp_a);

        funcs::DefaultElementwiseOperator<Context,
                                          T,
                                          funcs::MultiplyFunctor<T>,
                                          funcs::InverseMultiplyFunctor<T>>(
            dev_ctx, ddx_safe, y, ddout, axis);

        auto ddout_t2 = phi::EigenVector<T>::Flatten(*ddout);
        ddout_t2.device(place) = ddout_t2 + ddout_t1;

        // NOTE: in the following ElemwiseGradCompute, for the
        // first output tensor is nullptr, the branch to calculate first
        // output tensor will not be activated, DivGradDx function will not
        // be called and can be ignored, the first branch has little effect
        // on running speed.
        phi::funcs::ElemwiseGradCompute<Context, T, MulGradDX<T>, MulGradDY<T>>(
            dev_ctx,
            ddx_safe,
            ddy_safe,
            dout,
            dout,
            axis,
            nullptr,
            dy,
            MulGradDX<T>(),
            MulGradDY<T>());
      } else {
        DenseTensor tmp_a(ddout->dtype());
        tmp_a.Resize(ddout->dims());

        dev_ctx.template Alloc<T>(&tmp_a);

        funcs::DefaultElementwiseOperator<Context,
                                          T,
                                          funcs::MultiplyFunctor<T>,
                                          funcs::InverseMultiplyFunctor<T>>(
            dev_ctx, x, ddy_safe, &tmp_a, axis);

        auto ddout_t1 = phi::EigenVector<T>::Flatten(tmp_a);

        funcs::DefaultElementwiseOperator<Context,
                                          T,
                                          funcs::MultiplyFunctor<T>,
                                          funcs::InverseMultiplyFunctor<T>>(
            dev_ctx, ddx_safe, y, ddout, axis);

        auto ddout_t2 = phi::EigenVector<T>::Flatten(*ddout);
        ddout_t2.device(place) = ddout_t2 + ddout_t1;
      }
    }
  } else {
    VLOG(3) << "Calculating here with dx: " << dx << ", dy: " << dy;
    phi::funcs::ElemwiseGradCompute<Context, T, MulGradDX<T>, MulGradDY<T>>(
        dev_ctx,
        ddx_safe,
        ddy_safe,
        dout,
        dout,
        axis,
        dx,
        dy,
        MulGradDX<T>(),
        MulGradDY<T>());
  }
}

template <typename T, typename Context>
void MultiplyTripleGradKernel(const Context& dev_ctx,
                              const DenseTensor& x,
                              const DenseTensor& y,
                              const DenseTensor& dout,
                              const paddle::optional<DenseTensor>& ddx,
                              const paddle::optional<DenseTensor>& ddy,
                              const paddle::optional<DenseTensor>& d_dx,
                              const paddle::optional<DenseTensor>& d_dy,
                              const paddle::optional<DenseTensor>& d_ddout,
                              int axis,
                              DenseTensor* d_x,
                              DenseTensor* d_y,
                              DenseTensor* d_dout,
                              DenseTensor* d_ddx,
                              DenseTensor* d_ddy) {
  if (d_x) {
    d_x->Resize(x.dims());
    dev_ctx.template Alloc<T>(d_x);
  }
  if (d_y) {
    d_y->Resize(y.dims());
    dev_ctx.template Alloc<T>(d_y);
  }
  if (d_dout) {
    d_dout->Resize(dout.dims());
    dev_ctx.template Alloc<T>(d_dout);
  }
  if (d_ddx) {
    d_ddx->Resize(x.dims());
    dev_ctx.template Alloc<T>(d_ddx);
  }
  if (d_ddy) {
    d_ddy->Resize(y.dims());
    dev_ctx.template Alloc<T>(d_ddy);
  }

  auto& place = *dev_ctx.eigen_device();

  DenseTensor ddx_safe, ddy_safe;
  funcs::GetDoubleGradSafeTensor<Context, T>(
      dev_ctx, x, ddx.get_ptr(), &ddx_safe);
  funcs::GetDoubleGradSafeTensor<Context, T>(
      dev_ctx, y, ddy.get_ptr(), &ddy_safe);

  if (d_ddout.get_ptr()) {
    if (d_x) {
      // d_x = ddy * d_ddout
      funcs::DefaultElementwiseOperator<Context,
                                        T,
                                        funcs::MultiplyFunctor<T>,
                                        funcs::InverseMultiplyFunctor<T>>(
          dev_ctx, ddy_safe, *(d_ddout.get_ptr()), d_x, axis);
    }
    if (d_y) {
      // d_y = ddx * d_ddout
      funcs::DefaultElementwiseOperator<Context,
                                        T,
                                        funcs::MultiplyFunctor<T>,
                                        funcs::InverseMultiplyFunctor<T>>(
          dev_ctx, ddx_safe, *(d_ddout.get_ptr()), d_y, axis);
    }
  } else {
    if (d_x) {
      FullLikeKernel<T, Context>(dev_ctx, x, Scalar(0.0), x.dtype(), d_x);
    }
    if (d_y) {
      FullLikeKernel<T, Context>(dev_ctx, y, Scalar(0.0), y.dtype(), d_y);
    }
  }

  if (d_dout) {
    // get d_dout
    // d_dout = ddy * d_dx + d_dy * ddx
    DenseTensor d_dout_tmp;
    d_dout_tmp.Resize(dout.dims());
    dev_ctx.template Alloc<T>(&d_dout_tmp);

    if (d_dy && d_dx) {
      funcs::DefaultElementwiseOperator<Context,
                                        T,
                                        funcs::MultiplyFunctor<T>,
                                        funcs::InverseMultiplyFunctor<T>>(
          dev_ctx, d_dy.get(), ddx_safe, d_dout, axis);

      funcs::DefaultElementwiseOperator<Context,
                                        T,
                                        funcs::MultiplyFunctor<T>,
                                        funcs::InverseMultiplyFunctor<T>>(
          dev_ctx, ddy_safe, d_dx.get(), &d_dout_tmp, axis);

      auto d_dout_t = phi::EigenVector<T>::Flatten(*d_dout);
      auto d_dout_tmp_t = phi::EigenVector<T>::Flatten(d_dout_tmp);
      d_dout_t.device(place) = d_dout_t + d_dout_tmp_t;
    } else if (d_dy && !d_dx) {
      funcs::DefaultElementwiseOperator<Context,
                                        T,
                                        funcs::MultiplyFunctor<T>,
                                        funcs::InverseMultiplyFunctor<T>>(
          dev_ctx, d_dy.get(), ddx_safe, d_dout, axis);
      auto d_dout_t = phi::EigenVector<T>::Flatten(*d_dout);
      d_dout_t.device(place) = d_dout_t;
    } else if (!d_dy && d_dx) {
      funcs::DefaultElementwiseOperator<Context,
                                        T,
                                        funcs::MultiplyFunctor<T>,
                                        funcs::InverseMultiplyFunctor<T>>(
          dev_ctx, ddy_safe, d_dx.get(), d_dout, axis);

      auto d_dout_t = phi::EigenVector<T>::Flatten(*d_dout);
      d_dout_t.device(place) = d_dout_t;
    } else {
      FullLikeKernel<T, Context>(
          dev_ctx, dout, Scalar(0.0), dout.dtype(), d_dout);
    }
  }

  if (d_ddx && ddx) {
    // get d_ddx
    // d_ddx = dout * d_dy + y * d_ddout
    DenseTensor d_ddx_tmp;
    d_ddx_tmp.Resize(ddx->dims());
    dev_ctx.template Alloc<T>(&d_ddx_tmp);
    if (d_dy && d_ddout) {
      funcs::DefaultElementwiseOperator<Context,
                                        T,
                                        funcs::MultiplyFunctor<T>,
                                        funcs::InverseMultiplyFunctor<T>>(
          dev_ctx, dout, d_dy.get(), d_ddx, axis);

      funcs::DefaultElementwiseOperator<Context,
                                        T,
                                        funcs::MultiplyFunctor<T>,
                                        funcs::InverseMultiplyFunctor<T>>(
          dev_ctx, y, *(d_ddout.get_ptr()), &d_ddx_tmp, axis);

      auto d_ddx_t = phi::EigenVector<T>::Flatten(*d_ddx);
      auto d_ddx_tmp_t = phi::EigenVector<T>::Flatten(d_ddx_tmp);
      d_ddx_t.device(place) = d_ddx_t + d_ddx_tmp_t;
    } else if (d_dy && !d_ddout) {
      funcs::DefaultElementwiseOperator<Context,
                                        T,
                                        funcs::MultiplyFunctor<T>,
                                        funcs::InverseMultiplyFunctor<T>>(
          dev_ctx, dout, d_dy.get(), d_ddx, axis);

      auto d_ddx_t = phi::EigenVector<T>::Flatten(*d_ddx);
      d_ddx_t.device(place) = d_ddx_t;
    } else if (!d_dy && d_ddout) {
      funcs::DefaultElementwiseOperator<Context,
                                        T,
                                        funcs::MultiplyFunctor<T>,
                                        funcs::InverseMultiplyFunctor<T>>(
          dev_ctx, y, *(d_ddout.get_ptr()), d_ddx, axis);

      auto d_ddx_t = phi::EigenVector<T>::Flatten(*d_ddx);
      d_ddx_t.device(place) = d_ddx_t;
    } else {
      FullLikeKernel<T, Context>(dev_ctx, x, Scalar(0.0), x.dtype(), d_ddx);
    }
  }

  if (d_ddy && ddy) {
    // get d_ddy
    // d_ddy = dout * d_dx + x * d_ddout
    DenseTensor d_ddy_tmp;
    d_ddy_tmp.Resize(ddy->dims());
    dev_ctx.template Alloc<T>(&d_ddy_tmp);

    if (d_dx && d_ddout) {
      funcs::DefaultElementwiseOperator<Context,
                                        T,
                                        funcs::MultiplyFunctor<T>,
                                        funcs::InverseMultiplyFunctor<T>>(
          dev_ctx, dout, d_dx.get(), d_ddy, axis);

      funcs::DefaultElementwiseOperator<Context,
                                        T,
                                        funcs::MultiplyFunctor<T>,
                                        funcs::InverseMultiplyFunctor<T>>(
          dev_ctx, x, *(d_ddout.get_ptr()), &d_ddy_tmp, axis);

      auto d_ddy_t = phi::EigenVector<T>::Flatten(*d_ddy);
      auto d_ddy_tmp_t = phi::EigenVector<T>::Flatten(d_ddy_tmp);
      d_ddy_t.device(place) = d_ddy_t + d_ddy_tmp_t;
    } else if (d_dx && !d_ddout) {
      funcs::DefaultElementwiseOperator<Context,
                                        T,
                                        funcs::MultiplyFunctor<T>,
                                        funcs::InverseMultiplyFunctor<T>>(
          dev_ctx, dout, d_dx.get(), d_ddy, axis);

      auto d_ddy_t = phi::EigenVector<T>::Flatten(*d_ddy);
      d_ddy_t.device(place) = d_ddy_t;
    } else if (!d_dx && d_ddout) {
      funcs::DefaultElementwiseOperator<Context,
                                        T,
                                        funcs::MultiplyFunctor<T>,
                                        funcs::InverseMultiplyFunctor<T>>(
          dev_ctx, x, *(d_ddout.get_ptr()), d_ddy, axis);

      auto d_ddy_t = phi::EigenVector<T>::Flatten(*d_ddy);
      d_ddy_t.device(place) = d_ddy_t;
    } else {
      FullLikeKernel<T, Context>(dev_ctx, y, Scalar(0.0), y.dtype(), d_ddy);
    }
  }
}

/*
******************************
    Maximum Grad
******************************
*/

template <typename T>
struct MaxGradDx {
  HOSTDEVICE T operator()(T x, T y, T out UNUSED, T dout) const {
    return dout * static_cast<T>(x > y);
  }
};

template <typename T>
struct MaxGradDy {
  HOSTDEVICE T operator()(T x, T y, T out UNUSED, T dout) const {
    return dout * static_cast<T>(x <= y);
  }
};

/*
******************************
    Minimum Grad
******************************
*/
template <typename T>
struct MinGradDx {
  HOSTDEVICE T operator()(T x, T y, T out UNUSED, T dout) const {
    return dout * static_cast<T>(x < y);
  }
};

template <typename T>
struct MinGradDy {
  HOSTDEVICE T operator()(T x, T y, T out UNUSED, T dout) const {
    return dout * static_cast<T>(x >= y);
  }
};

template <typename T>
struct HeavisideGradDx {
  HOSTDEVICE T operator()(T x UNUSED, T y UNUSED, T out UNUSED, T dout) const {
    return dout * static_cast<T>(0);
  }
};

template <typename T>
struct HeavisideGradDy {
  HOSTDEVICE T operator()(T x, T y UNUSED, T out UNUSED, T dout) const {
    return dout * static_cast<T>(x == static_cast<T>(0));
  }
};

template <typename T, typename Context>
void HeavisideGradKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         const DenseTensor& y,
                         const DenseTensor& dout,
                         DenseTensor* dx,
                         DenseTensor* dy) {
  funcs::ElementwiseGradPreProcess(dout, dx);
  phi::funcs::
      ElemwiseGradCompute<Context, T, HeavisideGradDx<T>, HeavisideGradDy<T>>(
          dev_ctx,
          x,
          y,
          dout,
          dout,
          -1,
          dx,
          dy,
          HeavisideGradDx<T>(),
          HeavisideGradDy<T>());
}

template <typename T>
using ComplexType = phi::dtype::complex<T>;

#if defined(__CUDA_ARCH__) || defined(__HIPCC__)
template <typename T, typename MPType>
HOSTDEVICE typename std::enable_if<std::is_integral<T>::value, T>::type
compute_pow_grad_dx(T x, T y, T out, T dout) {
  return dout * y *
         std::pow(static_cast<double>(x), static_cast<double>(y - 1));
}
template <typename T, typename MPType>
HOSTDEVICE typename std::enable_if<!std::is_integral<T>::value, T>::type
compute_pow_grad_dx(T x, T y, T out, T dout) {
  MPType x_val = static_cast<MPType>(x);
  MPType y_val = static_cast<MPType>(y);
  return static_cast<T>(static_cast<MPType>(dout) * y_val *
                        std::pow(x_val, y_val - 1));
}
template <typename T, typename MPType>
HOSTDEVICE typename std::enable_if<std::is_integral<T>::value, T>::type
compute_pow_grad_dy(T x, T y, T out, T dout) {
  return dout * std::log(static_cast<double>(x)) *
         std::pow(static_cast<double>(x), static_cast<double>(y));
}
template <typename T, typename MPType>
HOSTDEVICE typename std::enable_if<!std::is_integral<T>::value, T>::type
compute_pow_grad_dy(T x, T y, T out, T dout) {
  MPType x_val = static_cast<MPType>(x);
  MPType y_val = static_cast<MPType>(y);
  return static_cast<T>(static_cast<MPType>(dout) * std::log(x_val) *
                        std::pow(x_val, y_val));
}
#else
template <typename T, typename MPType>
HOSTDEVICE T compute_pow_grad_dx(T x, T y, T out UNUSED, T dout) {
  MPType x_val = static_cast<MPType>(x);
  MPType y_val = static_cast<MPType>(y);
  return static_cast<T>(static_cast<MPType>(dout) * y_val *
                        std::pow(x_val, y_val - 1));
}
template <typename T, typename MPType>
HOSTDEVICE T compute_pow_grad_dy(T x, T y, T out UNUSED, T dout) {
  MPType x_val = static_cast<MPType>(x);
  MPType y_val = static_cast<MPType>(y);
  return static_cast<T>(static_cast<MPType>(dout) * std::log(x_val) *
                        std::pow(x_val, y_val));
}
#endif

template <typename T>
struct PowGradDX {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
  HOSTDEVICE T operator()(T x, T y, T out, T dout) const {
    return compute_pow_grad_dx<T, MPType>(x, y, out, dout);
  }
};

template <typename T>
struct PowGradDX<ComplexType<T>> {
  HOSTDEVICE ComplexType<T> operator()(ComplexType<T> x,
                                       ComplexType<T> y,
                                       ComplexType<T> out,
                                       ComplexType<T> dout) const {
    auto a_ = x.real;
    auto b_ = x.imag;
    auto c_ = y.real;
    auto d_ = y.imag;
    auto arctan_ = atan(b_ / a_);
    auto square_ = a_ * a_ + b_ * b_;
    auto e_ = exp(c_ / 2 * log(square_) - d_ * arctan_);
    auto v_ = d_ / 2 * log(square_) + c_ * arctan_;

    auto ux = e_ / square_ *
              ((a_ * c_ + b_ * d_) * cos(v_) + (b_ * c_ - a_ * d_) * sin(v_));
    auto uy = e_ / square_ *
              ((b_ * c_ - a_ * d_) * cos(v_) - (b_ * d_ + a_ * c_) * sin(v_));

    return dout * ComplexType<T>(ux, uy);
  }
};

template <typename T, typename Enable = void>
struct PowGradDY {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
  HOSTDEVICE T operator()(T x, T y, T out, T dout) const {
    return compute_pow_grad_dy<T, MPType>(x, y, out, dout);
  }
};

template <typename T>
struct PowGradDY<ComplexType<T>> {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
  HOSTDEVICE ComplexType<T> operator()(ComplexType<T> x,
                                       ComplexType<T> y,
                                       ComplexType<T> out,
                                       ComplexType<T> dout) const {
    return dout * conj(pow(x, y) * log(x));
  }
};

template <typename T, typename Context>
void ElementwisePowGradKernel(const Context& dev_ctx,
                              const DenseTensor& x,
                              const DenseTensor& y,
                              const DenseTensor& dout,
                              DenseTensor* dx,
                              DenseTensor* dy) {
  funcs::ElementwiseGradPreProcess(dout, dx);
  int axis = -1;
  phi::funcs::ElemwiseGradCompute<Context, T, PowGradDX<T>, PowGradDY<T>>(
      dev_ctx, x, y, dout, dout, axis, dx, dy, PowGradDX<T>(), PowGradDY<T>());
}

/*
******************************
    Remainder Grad
******************************
*/
// RemainderGradDx
template <typename T>
struct RemainderGradDx {
  HOSTDEVICE T operator()(T x, T y, T out UNUSED, T dout) const {
    // dx = dout
    return dout;
  }
};

// RemainderGradDy
template <typename T, typename Enable = void>
struct RemainderGradDy {
  HOSTDEVICE T operator()(T x, T y, T out UNUSED, T dout) const {
    return -dout * (std::floor(static_cast<double>(x / y)));
  }
};
template <typename T>
struct RemainderGradDy<
    T,
    typename std::enable_if<std::is_floating_point<T>::value>::type> {
  HOSTDEVICE T operator()(T x, T y, T out UNUSED, T dout) const {
    using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
    auto x_ = static_cast<MPType>(x);
    auto y_ = static_cast<MPType>(y);
    return static_cast<T>(-static_cast<MPType>(dout) * (std::floor((x_ / y_))));
  }
};
template <typename T>
struct RemainderGradDy<
    T,
    typename std::enable_if<std::is_integral<T>::value>::type> {
  HOSTDEVICE T operator()(T x, T y, T out UNUSED, T dout) const {
    // dy = -dout * (x / y)
    if (phi::is_negative(x) != phi::is_negative(y)) {
      // Subtracts one from the results of truncation division if the
      // divisor and dividend have different sign(bit)s and the remainder of
      // the division is nonzero
      const auto quot = x / y;
      const auto rem = x % y;
      auto ret = rem ? quot - 1 : quot;
      return -dout * ret;
    }
    return -dout * (x / y);
  }
};
/*
******************************
    Copysign Grad
******************************
*/
template <typename T>
HOSTDEVICE T compute_copysign_grad_dx(T x, T y, T out, T dout) {
  if (x == static_cast<T>(0))
    return x;
  else
    return static_cast<T>(dout * (funcs::copysign_func(x, y) / x));
}

template <typename T>
struct CopySignGradDX {
  HOSTDEVICE T operator()(T x, T y, T out, T dout) const {
    return compute_copysign_grad_dx<T>(x, y, out, dout);
  }
};

template <typename T>
struct CopySignGradDY {
  HOSTDEVICE T operator()(T x, T y, T out, T dout) const {
    return static_cast<T>(0);
  }
};

}  // namespace phi
