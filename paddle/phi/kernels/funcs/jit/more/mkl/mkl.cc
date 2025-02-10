/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. */

#include "paddle/phi/kernels/funcs/jit/more/mkl/mkl.h"

#include "paddle/phi/backends/cpu/cpu_info.h"
#include "paddle/phi/backends/dynload/mklml.h"
#include "paddle/phi/kernels/funcs/jit/refer/refer.h"
#include "paddle/phi/kernels/funcs/jit/registry.h"

namespace phi::jit::more::mkl {

template <>
void MatMul<float>(const float* a,
                   const float* b,
                   float* c,
                   const matmul_attr_t* attr) {
  phi::dynload::cblas_sgemm(CblasRowMajor,
                            CblasNoTrans,
                            CblasNoTrans,
                            attr->m,
                            attr->n,
                            attr->k,
                            1.f,
                            a,
                            attr->k,
                            b,
                            attr->n,
                            0.f,
                            c,
                            attr->n);
}

template <>
void MatMul<double>(const double* a,
                    const double* b,
                    double* c,
                    const matmul_attr_t* attr) {
  phi::dynload::cblas_dgemm(CblasRowMajor,
                            CblasNoTrans,
                            CblasNoTrans,
                            attr->m,
                            attr->n,
                            attr->k,
                            1.0,
                            a,
                            attr->k,
                            b,
                            attr->n,
                            0.0,
                            c,
                            attr->n);
}

template <>
void VMul<float>(const float* x, const float* y, float* z, int n) {
  phi::dynload::vsMul(n, x, y, z);
}

template <>
void VMul<double>(const double* x, const double* y, double* z, int n) {
  phi::dynload::vdMul(n, x, y, z);
}

template <>
void VAdd<float>(const float* x, const float* y, float* z, int n) {
  phi::dynload::vsAdd(n, x, y, z);
}

template <>
void VAdd<double>(const double* x, const double* y, double* z, int n) {
  phi::dynload::vdAdd(n, x, y, z);
}

template <>
void VScal<float>(const float* a, const float* x, float* y, int n) {
  if (x == y) {
    phi::dynload::cblas_sscal(n, *a, y, 1);
  } else {
    refer::VScal<float>(a, x, y, n);
  }
}

template <>
void VScal<double>(const double* a, const double* x, double* y, int n) {
  if (x == y) {
    phi::dynload::cblas_dscal(n, *a, y, 1);
  } else {
    refer::VScal<double>(a, x, y, n);
  }
}

template <>
void VExp<float>(const float* x, float* y, int n) {
  phi::dynload::vsExp(n, x, y);
}

template <>
void VExp<double>(const double* x, double* y, int n) {
  phi::dynload::vdExp(n, x, y);
}

template <>
void VSquare<float>(const float* x, float* y, int n) {
  phi::dynload::vsSqr(n, x, y);
}

template <>
void VSquare<double>(const double* x, double* y, int n) {
  phi::dynload::vdSqr(n, x, y);
}

template <>
void VCopy<float>(const float* x, float* y, int n) {
  phi::dynload::cblas_scopy(n, x, 1, y, 1);
}

template <>
void VCopy<double>(const double* x, double* y, int n) {
  phi::dynload::cblas_dcopy(n, x, 1, y, 1);
}

template <>
void VAXPY<float>(float a, const float* x, float* y, int n) {
  phi::dynload::cblas_saxpy(n, a, x, 1, y, 1);
}

template <>
void VAXPY<double>(double a, const double* x, double* y, int n) {
  phi::dynload::cblas_daxpy(n, a, x, 1, y, 1);
}

template <>
void ASum<float>(const float* x, float* res, int n) {
  res[0] = phi::dynload::cblas_sasum(n, x, 1);
}

template <>
void ASum<double>(const double* x, double* res, int n) {
  res[0] = phi::dynload::cblas_dasum(n, x, 1);
}

// TODO(TJ): tuning me carefully on AVX, AVX2 and AVX512
template <>
bool VMulKernel<float>::CanBeUsed(const int& d) const {
  return phi::backends::cpu::MayIUse(phi::backends::cpu::avx512f) && d > 512;
}

template <>
bool VAddKernel<float>::CanBeUsed(const int& d) const {
  return phi::backends::cpu::MayIUse(phi::backends::cpu::avx) && d > 512;
}

template <>
bool VScalKernel<float>::CanBeUsed(const int& d) const {
  return phi::backends::cpu::MayIUse(phi::backends::cpu::avx512f) && d > 512;
}

template <>
bool VExpKernel<float>::CanBeUsed(const int& d) const {
  return d > 7;
}

template <>
bool VSquareKernel<float>::CanBeUsed(const int& d) const {
  return d > 7;
}

template <>
bool VCopyKernel<float>::CanBeUsed(const int& d) const {
  return d > 15;
}

template <>
bool VBroadcastKernel<float>::CanBeUsed(const int64_t& d) const {
  return d > 127;
}

template <>
bool VBroadcastKernel<double>::CanBeUsed(const int64_t& attr) const {
  return true;
}

template <>
bool VSigmoidKernel<float>::CanBeUsed(const int& d) const {
  return d > 7;
}

template <>
bool VTanhKernel<float>::CanBeUsed(const int& d) const {
  return d > 7;
}

template <>
bool SeqPoolKernel<float>::CanBeUsed(const seq_pool_attr_t& attr) const {
  return true;
}

template <>
bool SeqPoolKernel<double>::CanBeUsed(const seq_pool_attr_t& attr) const {
  return true;
}

template <>
bool EmbSeqPoolKernel<float>::CanBeUsed(const emb_seq_pool_attr_t& attr) const {
  return true;
}

template <>
bool EmbSeqPoolKernel<double>::CanBeUsed(
    const emb_seq_pool_attr_t& attr) const {
  return true;
}

template <>
bool SgdKernel<float>::CanBeUsed(const sgd_attr_t& attr) const {
  return true;
}

template <>
bool SgdKernel<double>::CanBeUsed(const sgd_attr_t& attr) const {
  return true;
}

template <>
bool MatMulKernel<float>::CanBeUsed(const matmul_attr_t& attr) const {
  return phi::backends::cpu::MayIUse(phi::backends::cpu::avx);
}

template <>
bool MatMulKernel<double>::CanBeUsed(const matmul_attr_t& attr) const {
  return true;
}

#define AWALYS_USE_ME_WITH_DOUBLE(func)                             \
  template <>                                                       \
  bool func##Kernel<double>::CanBeUsed(const int& d UNUSED) const { \
    return true;                                                    \
  }

AWALYS_USE_ME_WITH_DOUBLE(VMul);
AWALYS_USE_ME_WITH_DOUBLE(VAdd);
AWALYS_USE_ME_WITH_DOUBLE(VScal);
AWALYS_USE_ME_WITH_DOUBLE(VExp);
AWALYS_USE_ME_WITH_DOUBLE(VSigmoid);
AWALYS_USE_ME_WITH_DOUBLE(VTanh);
AWALYS_USE_ME_WITH_DOUBLE(VSquare);
AWALYS_USE_ME_WITH_DOUBLE(VCopy);

#undef AWALYS_USE_ME_WITH_DOUBLE
}  // namespace phi::jit::more::mkl

namespace mkl = phi::jit::more::mkl;

#define REGISTER_MKL_KERNEL(func) \
  REGISTER_JITKERNEL_MORE(        \
      k##func, mkl, mkl::func##Kernel<float>, mkl::func##Kernel<double>)

REGISTER_MKL_KERNEL(MatMul);
REGISTER_MKL_KERNEL(VMul);
REGISTER_MKL_KERNEL(VAdd);
REGISTER_MKL_KERNEL(VScal);
REGISTER_MKL_KERNEL(VExp);
REGISTER_MKL_KERNEL(VSquare);
REGISTER_MKL_KERNEL(VCopy);
REGISTER_MKL_KERNEL(VBroadcast);
REGISTER_MKL_KERNEL(VSigmoid);
REGISTER_MKL_KERNEL(VTanh);
REGISTER_MKL_KERNEL(SeqPool);
REGISTER_MKL_KERNEL(EmbSeqPool);
REGISTER_MKL_KERNEL(Sgd);

#undef REGISTER_MKL_KERNEL
