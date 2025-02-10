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

#include "paddle/phi/kernels/sparse/elementwise_kernel.h"
#include "paddle/phi/common/complex.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_meta.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/core/visit_type.h"
#include "paddle/phi/kernels/elementwise_add_kernel.h"
#include "paddle/phi/kernels/elementwise_kernel.h"
#include "paddle/phi/kernels/funcs/elementwise_functor.h"
#include "paddle/phi/kernels/funcs/sparse/flatten_indices.h"
#include "paddle/phi/kernels/sparse/empty_kernel.h"
#include "paddle/phi/kernels/sparse/sparse_utils_kernel.h"

namespace phi::sparse {

template <typename T, typename Functor>
struct BinaryOPWithZeroCompareFunctor {
  explicit BinaryOPWithZeroCompareFunctor(Functor functor)
      : functor_(functor) {}
  inline HOSTDEVICE void operator()(const T* a,
                                    const T* b,
                                    T* result,
                                    const int64_t len) const {
    for (int64_t i = 0; i < len; ++i) {
      result[i] = functor_(a[i], b[i]);
    }
  }
  Functor functor_;
};

template <typename T, typename IntT, typename Functor>
void Merge(const IntT el_len,
           const IntT* a_index,
           const T* a_values,
           const IntT len_a,
           const IntT* b_index_org,
           const T* b_values_org,
           const IntT len_b,
           const IntT len_b_max,
           IntT* c_index,
           T* c_values,
           IntT* out_nnz,
           const Functor& functor_org,
           const bool is_divide) {
  IntT a = 0;
  IntT b = 0;
  IntT& nnz = (*out_nnz);
  nnz = 0;
  const IntT* b_index = nullptr;
  std::vector<IntT> b_full_index;
  const std::vector<T> zero(el_len, 0);
  auto functor = BinaryOPWithZeroCompareFunctor<T, Functor>(functor_org);

  std::vector<const T*> b_values(len_b_max, zero.data());
  for (auto i = 0; i < len_b; ++i) {
    b_values[b_index_org[i]] = b_values_org + i * el_len;
  }
  //  if is divide expend b_index_org to b_full_index
  if (is_divide) {
    b_full_index = std::vector<IntT>(len_b_max);
    for (int64_t j = 0; j < static_cast<int64_t>(b_full_index.size()); ++j) {
      b_full_index[j] = j;
    }
    b_index = b_full_index.data();
  } else {
    b_index = b_index_org;
  }
  // merge
  while (a < len_a && b < (is_divide ? len_b_max : len_b)) {
    if (a_index[a] == b_index[b]) {
      functor(a_values + a * el_len,
              b_values[b_index[b]],
              c_values + nnz * el_len,
              el_len);
      c_index[nnz] = a_index[a];
      ++nnz;
      ++a;
      ++b;
    } else if (a_index[a] < b_index[b]) {  // coordinate x[a] < coordinate y[b]
      functor(
          a_values + a * el_len, zero.data(), c_values + nnz * el_len, el_len);
      c_index[nnz] = a_index[a];
      ++nnz;
      ++a;
    } else if (a_index[a] > b_index[b]) {  // coordinate x[a] > coordinate y[b]
      functor(
          zero.data(), b_values[b_index[b]], c_values + nnz * el_len, el_len);
      c_index[nnz] = b_index[b];
      ++nnz;
      ++b;
    }
  }
  // a tail
  while (a < len_a) {
    functor(
        a_values + a * el_len, zero.data(), c_values + nnz * el_len, el_len);
    c_index[nnz] = a_index[a];
    ++nnz;
    ++a;
  }
  //  b tail
  while (b < (is_divide ? len_b_max : len_b)) {
    functor(zero.data(), b_values[b_index[b]], c_values + nnz * el_len, el_len);
    c_index[nnz] = b_index[b];
    ++nnz;
    ++b;
  }
}

// SparseCooTensor elementwise op, only support same shape tensor now
template <typename T, typename IntT, typename Context, typename Functor>
void ElementWiseCooKernelImpl(const Context& dev_ctx,
                              const SparseCooTensor& x,
                              const SparseCooTensor& y,
                              SparseCooTensor* out,
                              const Functor& functor) {
  PADDLE_ENFORCE_EQ(x.dims(),
                    y.dims(),
                    common::errors::InvalidArgument(
                        "Currently only support same shape elementwise "
                        "compute. The input tensor X's shape "
                        "should be identical with Y's shape. But received X's "
                        "shape = [%s], Y's shape = [%s].",
                        x.dims(),
                        y.dims()));

  // temporary policy: for broadcast add
  // TODO(zhangkaihuo): implement a correct function
  const bool is_add = std::is_same<Functor, funcs::AddFunctor<T>>::value;
  if (is_add && x.indices().numel() == y.indices().numel()) {
    int compare_indices = memcmp(x.indices().data<IntT>(),
                                 y.indices().data<IntT>(),
                                 sizeof(IntT) * x.indices().numel());
    if (compare_indices == 0) {
      EmptyLikeCooKernel<T, Context>(dev_ctx, x, out);
      phi::AddKernel<T, Context>(
          dev_ctx, x.values(), y.values(), out->mutable_values());
      return;
    }
  }
  int64_t element_size = 1;
  for (auto j = 1; j < x.values().dims().size(); ++j) {
    element_size *= x.values().dims()[j];
  }
  IntT nnz = 0;
  const auto x_values = x.values().data<T>();
  const auto y_values = y.values().data<T>();
  const auto sparse_dim = x.indices().dims()[0];
  const bool is_divide = std::is_same<Functor, funcs::DivideFunctor<T>>::value;

  int64_t max_len = 1;
  for (auto j = 0; j < sparse_dim; ++j) {
    max_len *= x.dims()[j];
  }

  std::vector<IntT> sparse_offsets(sparse_dim), x_indices(x.nnz()),
      y_indices(y.nnz());

  phi::funcs::sparse::CalcOffsetsPerDim<IntT>(
      x.dims(), sparse_dim, sparse_offsets.data());

  phi::funcs::sparse::FlattenIndices(x.indices().data<IntT>(),
                                     sparse_offsets.data(),
                                     x.nnz(),
                                     sparse_dim,
                                     0,
                                     1,
                                     x_indices.data());

  phi::funcs::sparse::FlattenIndices(y.indices().data<IntT>(),
                                     sparse_offsets.data(),
                                     y.nnz(),
                                     sparse_dim,
                                     0,
                                     1,
                                     y_indices.data());

  std::vector<IntT> out_indices;
  std::vector<T> out_values_vec;
  if (is_divide) {
    out_indices.reserve(max_len);
  } else {
    out_indices.reserve(x.nnz() + y.nnz());
  }
  out_values_vec.reserve(max_len * element_size);

  //  merge x and y
  Merge<T, IntT, Functor>(element_size,
                          x_indices.data(),
                          x_values,
                          x_indices.size(),
                          y_indices.data(),
                          y_values,
                          y_indices.size(),
                          max_len,
                          out_indices.data(),
                          out_values_vec.data(),
                          &nnz,
                          functor,
                          is_divide);

  std::vector<IntT> out_indices_vec;
  out_indices_vec.resize(nnz * sparse_dim);

  Dim<DDim::kMaxRank> const_dims;
  for (auto i = 0; i < x.dims().size(); i++) {
    const_dims[i] = x.dims()[i];
  }

  funcs::sparse::IndexToCoordinate<IntT>(out_indices.data(),
                                         const_dims,
                                         nnz,
                                         sparse_dim,
                                         0,
                                         1,
                                         out_indices_vec.data());

  if (nnz == 0) {
    phi::DenseTensor out_indices = phi::EmptyLike<IntT>(dev_ctx, x.indices());
    phi::DenseTensor out_values = phi::EmptyLike<T>(dev_ctx, x.values());
    out->SetMember(out_indices, out_values, x.dims());
  } else {
    DenseTensorMeta indices_meta(
        phi::CppTypeToDataType<IntT>::Type(),
        common::make_ddim(
            {static_cast<int64_t>(sparse_dim), static_cast<int64_t>(nnz)}),
        DataLayout::NCHW);
    auto indices_dim = common::vectorize(
        slice_ddim(x.values().dims(), 1, x.values().dims().size()));
    indices_dim.insert(indices_dim.begin(), nnz);
    DenseTensorMeta values_meta(
        x.dtype(), common::make_ddim(indices_dim), DataLayout::NCHW);
    phi::DenseTensor out_indices = phi::Empty(dev_ctx, std::move(indices_meta));
    phi::DenseTensor out_values = phi::Empty(dev_ctx, std::move(values_meta));

    std::memcpy(out_indices.data<IntT>(),
                out_indices_vec.data(),
                sizeof(IntT) * sparse_dim * nnz);
    std::memcpy(out_values.data<T>(),
                out_values_vec.data(),
                sizeof(T) * nnz * element_size);

    out->SetMember(out_indices, out_values, x.dims());
  }
}

#define DEFINE_CSR_ELEMENTWISE_CPU_KERNEL(name)                               \
  template <typename T, typename IntT, typename Context>                      \
  void ElementWise##name##CsrCPUKernel(const Context& dev_ctx,                \
                                       const SparseCsrTensor& x,              \
                                       const SparseCsrTensor& y,              \
                                       SparseCsrTensor* out) {                \
    auto coo_x = CsrToCoo<T>(dev_ctx, x);                                     \
    auto coo_y = CsrToCoo<T>(dev_ctx, y);                                     \
    auto coo_out = ElementWise##name##Coo<T, Context>(dev_ctx, coo_x, coo_y); \
    CooToCsrKernel<T>(dev_ctx, coo_out, out);                                 \
  }

#define DEFINE_CSR_ELEMENTWISE_KERNEL(name)                               \
  template <typename T, typename Context>                                 \
  void ElementWise##name##CsrKernel(const Context& dev_ctx,               \
                                    const SparseCsrTensor& x,             \
                                    const SparseCsrTensor& y,             \
                                    SparseCsrTensor* out) {               \
    PD_VISIT_BASE_INTEGRAL_TYPES(                                         \
        x.crows().dtype(), "ElementWise##name##CsrCPUKernel", ([&] {      \
          ElementWise##name##CsrCPUKernel<T, data_t>(dev_ctx, x, y, out); \
        }));                                                              \
  }

#define DEFINE_COO_ELEMENTWISE_CPU_KERNEL(name)                          \
  template <typename T, typename IntT, typename Context>                 \
  void ElementWise##name##CooCPUKernel(const Context& dev_ctx,           \
                                       const SparseCooTensor& x,         \
                                       const SparseCooTensor& y,         \
                                       SparseCooTensor* out) {           \
    funcs::name##Functor<T> functor;                                     \
    ElementWiseCooKernelImpl<T, IntT, Context, funcs::name##Functor<T>>( \
        dev_ctx, x, y, out, functor);                                    \
  }

#define DEFINE_COO_ELEMENTWISE_KERNEL(name)                               \
  template <typename T, typename Context>                                 \
  void ElementWise##name##CooKernel(const Context& dev_ctx,               \
                                    const SparseCooTensor& x,             \
                                    const SparseCooTensor& y,             \
                                    SparseCooTensor* out) {               \
    PD_VISIT_BASE_INTEGRAL_TYPES(                                         \
        x.indices().dtype(), "ElementWise##name##CooCPUKernel", ([&] {    \
          ElementWise##name##CooCPUKernel<T, data_t>(dev_ctx, x, y, out); \
        }));                                                              \
  }

DEFINE_CSR_ELEMENTWISE_CPU_KERNEL(Add)
DEFINE_CSR_ELEMENTWISE_CPU_KERNEL(Subtract)
DEFINE_CSR_ELEMENTWISE_CPU_KERNEL(Multiply)
DEFINE_CSR_ELEMENTWISE_CPU_KERNEL(Divide)

DEFINE_CSR_ELEMENTWISE_KERNEL(Add)
DEFINE_CSR_ELEMENTWISE_KERNEL(Subtract)
DEFINE_CSR_ELEMENTWISE_KERNEL(Multiply)
DEFINE_CSR_ELEMENTWISE_KERNEL(Divide)

DEFINE_COO_ELEMENTWISE_CPU_KERNEL(Add)
DEFINE_COO_ELEMENTWISE_CPU_KERNEL(Subtract)
DEFINE_COO_ELEMENTWISE_CPU_KERNEL(Multiply)
DEFINE_COO_ELEMENTWISE_CPU_KERNEL(Divide)

DEFINE_COO_ELEMENTWISE_KERNEL(Add)
DEFINE_COO_ELEMENTWISE_KERNEL(Subtract)
DEFINE_COO_ELEMENTWISE_KERNEL(Multiply)
DEFINE_COO_ELEMENTWISE_KERNEL(Divide)

}  // namespace phi::sparse

using complex64 = ::phi::dtype::complex<float>;
using complex128 = ::phi::dtype::complex<double>;

PD_REGISTER_KERNEL(add_csr_csr,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::ElementWiseAddCsrKernel,
                   float,
                   double,
                   int16_t,
                   int,
                   int64_t,
                   complex64,
                   complex128) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_CSR);
  kernel->InputAt(1).SetDataLayout(phi::DataLayout::SPARSE_CSR);
}

PD_REGISTER_KERNEL(add_coo_coo,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::ElementWiseAddCooKernel,
                   float,
                   double,
                   int16_t,
                   int,
                   int64_t,
                   complex64,
                   complex128) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
  kernel->InputAt(1).SetDataLayout(phi::DataLayout::SPARSE_COO);
}

PD_REGISTER_KERNEL(subtract_csr_csr,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::ElementWiseSubtractCsrKernel,
                   float,
                   double,
                   int16_t,
                   int,
                   int64_t,
                   complex64,
                   complex128) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_CSR);
  kernel->InputAt(1).SetDataLayout(phi::DataLayout::SPARSE_CSR);
}

PD_REGISTER_KERNEL(subtract_coo_coo,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::ElementWiseSubtractCooKernel,
                   float,
                   double,
                   int16_t,
                   int,
                   int64_t,
                   complex64,
                   complex128) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
  kernel->InputAt(1).SetDataLayout(phi::DataLayout::SPARSE_COO);
}

PD_REGISTER_KERNEL(multiply_csr_csr,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::ElementWiseMultiplyCsrKernel,
                   float,
                   double,
                   int16_t,
                   int,
                   int64_t,
                   complex64,
                   complex128) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_CSR);
  kernel->InputAt(1).SetDataLayout(phi::DataLayout::SPARSE_CSR);
}

PD_REGISTER_KERNEL(multiply_coo_coo,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::ElementWiseMultiplyCooKernel,
                   float,
                   double,
                   int16_t,
                   int,
                   int64_t,
                   complex64,
                   complex128) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
  kernel->InputAt(1).SetDataLayout(phi::DataLayout::SPARSE_COO);
}

PD_REGISTER_KERNEL(divide_csr_csr,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::ElementWiseDivideCsrKernel,
                   float,
                   double,
                   int16_t,
                   int,
                   int64_t,
                   complex64,
                   complex128) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_CSR);
  kernel->InputAt(1).SetDataLayout(phi::DataLayout::SPARSE_CSR);
}

PD_REGISTER_KERNEL(divide_coo_coo,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::ElementWiseDivideCooKernel,
                   float,
                   double,
                   int16_t,
                   int,
                   int64_t,
                   complex64,
                   complex128) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
  kernel->InputAt(1).SetDataLayout(phi::DataLayout::SPARSE_COO);
}

PD_REGISTER_KERNEL(add_coo_dense,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::ElementWiseAddDenseKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   complex64,
                   complex128) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
}
