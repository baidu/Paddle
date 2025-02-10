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

#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/matrix_solve.h"
#include "paddle/phi/kernels/impl/lu_kernel_impl.h"
#include "paddle/phi/kernels/lu_solve_grad_kernel.h"
#include "paddle/phi/kernels/lu_solve_kernel.h"
#include "paddle/phi/kernels/lu_unpack_kernel.h"
#include "paddle/phi/kernels/triangular_solve_kernel.h"

namespace phi {

template <typename T, typename Context>
void LuSolveGradKernel(const Context& dev_ctx,
                       const DenseTensor& x,
                       const DenseTensor& lu,
                       const DenseTensor& pivots,
                       const DenseTensor& out,
                       const DenseTensor& out_grad,
                       const std::string& trans,
                       DenseTensor* x_grad,
                       DenseTensor* lu_grad) {
  
  if (x_grad != nullptr) {
    std::string trans_t = (trans == "N") ? "T" : "N";
    phi::LuSolveKernel<T, Context>(dev_ctx, out_grad, lu, pivots, trans_t, x_grad);
  }

  if (lu_grad!= nullptr) {
    DenseTensor p, l, u, l_mH, u_mH;
    bool unpack_pivots = (trans == "N") ? false : true;
    phi::LUUnpackKernel<T, Context>(dev_ctx, lu, pivots, true, unpack_pivots, &p, &l, &u);
    
    phi::Tensor_Conj<Context, T>(dev_ctx, u, &l_mH);
    l_mH = phi::Transpose2DTo6D<Context, T>(dev_ctx, l_mH);
    
    phi::Tensor_Conj<Context, T>(dev_ctx, u, &u_mH);
    u_mH = phi::Transpose2DTo6D<Context, T>(dev_ctx, u_mH);
    if (trans == "N") {
      // gR = U^{-H}op_2(-gX)op_2(X)^Ha
      DenseTensor gR, psi_tmp, out_mH;
      phi::Tensor_Conj<Context, T>(dev_ctx, out, &out_mH);
      out_mH = phi::Transpose2DTo6D<Context, T>(dev_ctx, out_mH);

      auto blas = phi::funcs::GetBlas<Context, T>(dev_ctx);
      auto out_grad_dims = out_grad.dims();
      auto mat_dim_l = phi::funcs::CreateMatrixDescriptor(out_grad_dims, 0, false);
      auto out_mH_dims = out_mH.dims();
      auto mat_dim_g = phi::funcs::CreateMatrixDescriptor(out_mH_dims, 0, true);
      psi_tmp.Resize(out_grad_dims);
      dev_ctx.template Alloc<T>(&psi_tmp);
      blas.MatMul(out_grad,
                  mat_dim_l,
                  out_mH,
                  mat_dim_g,
                  static_cast<T>(-1),
                  &psi_tmp,
                  static_cast<T>(0));
      phi::TriangularSolveKernel<T, Context>(
        dev_ctx, u_mH, psi_tmp, false, false, false, &gR);


      // gL = (L^{-H} gR U^H).tril(-1)
      DenseTensor mul_tmp, gL;
      auto gr_dims = gR.dims();
      auto mat_dim_r = phi::funcs::CreateMatrixDescriptor(gr_dims, 0, false);
      auto gu_dims = u_mH.dims();
      auto mat_dim_u = phi::funcs::CreateMatrixDescriptor(gu_dims, 0, true);
      mul_tmp.Resize(gr_dims);
      dev_ctx.template Alloc<T>(&mul_tmp);
      blas.MatMul(gR,
                  mat_dim_r,
                  u_mH,
                  mat_dim_u,
                  static_cast<T>(1),
                  &mul_tmp,
                  static_cast<T>(0));
      phi::TriangularSolveKernel<T, Context>(
        dev_ctx, l_mH, mul_tmp, true, true, true, &gL);
      
      auto phil_rank = gL.dims().size();
      auto phir_rank = gR.dims().size();
      phi::funcs::ForRange<Context> l_for_range(dev_ctx, gL.numel());
      phi::funcs::TrilTriuCompute<T> tril_computer(gL.data<T>(),
                                               -1,
                                               true,
                                               gL.dims()[phil_rank - 2],
                                               gL.dims()[phil_rank - 1],
                                               gL.data<T>());
      l_for_range(tril_computer);

      phi::funcs::ForRange<Context> r_for_range(dev_ctx, gR.numel());
      phi::funcs::TrilTriuCompute<T> triu_computer(gR.data<T>(),
                                               0,
                                               false,
                                               gR.dims()[phir_rank - 2],
                                               gR.dims()[phir_rank - 1],
                                               gR.data<T>());
      r_for_range(triu_computer);
      Tensor_Add<Context, T>(dev_ctx, gL, gR, lu_grad);
    } else {
      DenseTensor gR, p_mT, tem_out, out_grad_mH, tem_out1, tem_out2, tem_out3, gU;
      p_mT = Transpose2DTo6D<Context, T>(dev_ctx, p);
      auto PmTdims = p_mT.dims();
      auto Outdims = out.dims();
      auto mat_dim_p = phi::funcs::CreateMatrixDescriptor(PmTdims, 0, false);
      auto mat_dim_o = phi::funcs::CreateMatrixDescriptor(Outdims, 0, true);
      tem_out.Resize(PmTdims);
      auto blas = phi::funcs::GetBlas<Context, T>(dev_ctx);
      // gR = -P^T op_3(X)op_1(op_2(gX))P
      blas.MatMul(p_mT,
                  mat_dim_p,
                  out,
                  mat_dim_o,
                  static_cast<T>(-1),
                  &tem_out,
                  static_cast<T>(0));
      
      Tensor_Conj<Context, T>(dev_ctx, out_grad, &out_grad_mH);
      out_grad_mH = Transpose2DTo6D<Context, T>(dev_ctx, out_grad_mH);
      auto TemOutdims = tem_out.dims();
      auto OutGradmHdims = out_grad_mH.dims();
      auto mat_dim_tem_out = phi::funcs::CreateMatrixDescriptor(TemOutdims, 0, false);
      auto mat_dim_out_grad_mH = phi::funcs::CreateMatrixDescriptor(OutGradmHdims, 0, true);
      tem_out1.Resize(TemOutdims);
      blas.MatMul(tem_out,
                  mat_dim_tem_out,
                  out_grad_mH,
                  mat_dim_out_grad_mH,
                  static_cast<T>(1),
                  &tem_out1,
                  static_cast<T>(0));
      auto TemOutdims1 = tem_out1.dims();
      auto pdims = p.dims();
      auto mat_dim_tem_out1 = phi::funcs::CreateMatrixDescriptor(TemOutdims1, 0, false);
      auto mat_dim_p1 = phi::funcs::CreateMatrixDescriptor(pdims, 0, true);
      tem_out2.Resize(TemOutdims1);
      blas.MatMul(tem_out1,
                  mat_dim_tem_out1,
                  p,
                  mat_dim_p1,
                  static_cast<T>(1),
                  &tem_out2,
                  static_cast<T>(0));
      // gR = gR L^{-H}
      phi::TriangularSolveKernel<T, Context>(
        dev_ctx, l_mH, tem_out2, true, false, true, &gR);
      // gU = (L^H gR U^{-H}).triu()
      auto LmHdims = l_mH.dims();
      auto gRdims = gR.dims();
      auto mat_dim_l_mh = phi::funcs::CreateMatrixDescriptor(LmHdims, 0, false);
      auto mat_dim_gr = phi::funcs::CreateMatrixDescriptor(gRdims, 0, true);
      tem_out3.Resize(LmHdims);
      blas.MatMul(l_mH,
                  mat_dim_l_mh,
                  gR,
                  mat_dim_gr,
                  static_cast<T>(1),
                  &tem_out3,
                  static_cast<T>(0));
      phi::TriangularSolveKernel<T, Context>(
        dev_ctx, u_mH, tem_out3, false, false, false, &gU);
      
      auto phiu_rank = gU.dims().size();
      auto phir_rank = gR.dims().size();
      phi::funcs::ForRange<Context> l_for_range(dev_ctx, gR.numel());
      phi::funcs::TrilTriuCompute<T> tril_computer(gR.data<T>(),
                                               -1,
                                               true,
                                               gR.dims()[phir_rank - 2],
                                               gR.dims()[phir_rank - 1],
                                               gR.data<T>());
      l_for_range(tril_computer);

      phi::funcs::ForRange<Context> r_for_range(dev_ctx, gU.numel());
      phi::funcs::TrilTriuCompute<T> triu_computer(gU.data<T>(),
                                               0,
                                               false,
                                               gU.dims()[phiu_rank - 2],
                                               gU.dims()[phiu_rank - 1],
                                               gU.data<T>());
      r_for_range(triu_computer);
      Tensor_Add<Context, T>(dev_ctx, gR, gU, lu_grad);
    }
  }
}

}  // namespace phi
