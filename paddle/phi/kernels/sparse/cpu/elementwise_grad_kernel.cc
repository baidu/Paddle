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

#include "paddle/phi/kernels/sparse/elementwise_grad_kernel.h"
#include "paddle/phi/kernels/sparse/elementwise_kernel.h"

#include "glog/logging.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_meta.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/core/visit_type.h"
#include "paddle/phi/kernels/activation_kernel.h"
#include "paddle/phi/kernels/complex_kernel.h"
#include "paddle/phi/kernels/elementwise_kernel.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/sparse/flatten_indices.h"
#include "paddle/phi/kernels/sparse/empty_kernel.h"

namespace phi::sparse {

template <typename T, typename IntT, typename Context>
void AllocCsrPtr(const Context& dev_ctx,
                 const SparseCsrTensor& x,
                 SparseCsrTensor* dx) {
  DenseTensor dx_crows = phi::EmptyLike<IntT>(dev_ctx, x.crows());
  DenseTensor dx_cols = phi::EmptyLike<IntT>(dev_ctx, x.cols());
  DenseTensor dx_values = phi::EmptyLike<T>(dev_ctx, x.values());
  dx->set_meta(x.meta());  // NOLINT
  dx->SetMember(dx_crows, dx_cols, dx_values, x.dims());
}

template <typename T, typename IntT, typename Context>
void AllocCooPtr(const Context& dev_ctx,
                 const SparseCooTensor& x,
                 SparseCooTensor* dx) {
  DenseTensor dx_indices = phi::EmptyLike<IntT>(dev_ctx, x.indices());
  DenseTensor dx_values = phi::EmptyLike<T>(dev_ctx, x.values());
  dx->set_meta(x.meta());  // NOLINT
  dx->SetMember(dx_indices, dx_values, x.dims(), x.coalesced());
}

template <typename T, typename IntT, typename Context>
void CopyCooValues(const Context& dev_ctx,
                   const SparseCooTensor& dout,
                   const SparseCooTensor& x,
                   SparseCooTensor* dx) {
  Copy(dev_ctx, x.indices(), dev_ctx.GetPlace(), false, dx->mutable_indices());

  const int sparse_dim = x.sparse_dim();
  std::vector<IntT> sparse_offsets(sparse_dim), dout_indices(dout.nnz()),
      x_indices(x.nnz());

  phi::funcs::sparse::CalcOffsetsPerDim<IntT>(
      dout.dims(), sparse_dim, sparse_offsets.data());

  phi::funcs::sparse::FlattenIndices(dout.indices().data<IntT>(),
                                     sparse_offsets.data(),
                                     dout.nnz(),
                                     sparse_dim,
                                     0,
                                     1,
                                     dout_indices.data());

  phi::funcs::sparse::FlattenIndices(x.indices().data<IntT>(),
                                     sparse_offsets.data(),
                                     x.nnz(),
                                     sparse_dim,
                                     0,
                                     1,
                                     x_indices.data());

  size_t i = 0, j = 0;
  T* dx_values_ptr = dx->mutable_values()->data<T>();
  const T* dout_values_ptr = dout.values().data<T>();

  int64_t element_size = 1;
  for (auto j = 1; j < x.values().dims().size(); ++j) {
    element_size *= x.values().dims()[j];
  }

  while (i < dout_indices.size() && j < x_indices.size()) {
    if (dout_indices[i] == x_indices[j]) {
      memcpy(dx_values_ptr + j * element_size,
             dout_values_ptr + i * element_size,
             element_size * sizeof(T));
      ++i;
      ++j;
    } else if (dout_indices[i] > x_indices[j]) {
      memset(dx_values_ptr + j * element_size, 0, element_size * sizeof(T));
      ++j;
    } else {
      ++i;
    }
  }
  while (j < x_indices.size()) {
    memset(dx_values_ptr + j * element_size, 0, element_size * sizeof(T));
    ++j;
  }
}

template <typename T, typename IntT, typename Context>
void CopyCsrValues(const Context& dev_ctx,
                   const SparseCsrTensor& dout,
                   const SparseCsrTensor& x,
                   SparseCsrTensor* dx) {
  Copy(dev_ctx, x.crows(), dev_ctx.GetPlace(), false, dx->mutable_crows());
  Copy(dev_ctx, x.cols(), dev_ctx.GetPlace(), false, dx->mutable_cols());

  const auto& x_dims = x.dims();
  int batch = static_cast<int>(x_dims.size() == 2 ? 1 : x_dims[0]);
  int rows = static_cast<int>(x_dims.size() == 2 ? x_dims[0] : x_dims[1]);

  const IntT* x_crows_ptr = x.crows().data<IntT>();
  const IntT* x_cols_ptr = x.cols().data<IntT>();

  const IntT* dout_crows_ptr = dout.crows().data<IntT>();
  const IntT* dout_cols_ptr = dout.cols().data<IntT>();
  const T* dout_values_ptr = dout.values().data<T>();

  T* dx_values_ptr = dx->mutable_values()->data<T>();

  for (int b = 0; b < batch; b++) {
    for (int r = 0; r < rows; r++) {
      int x_start = x_crows_ptr[b * (rows + 1) + r];
      int dout_start = dout_crows_ptr[b * (rows + 1) + r];
      int x_row_nnz = x_crows_ptr[b * (rows + 1) + r + 1] - x_start;
      int dout_row_nnz = dout_crows_ptr[b * (rows + 1) + r + 1] - dout_start;
      int i = 0, j = 0;
      while (i < x_row_nnz && j < dout_row_nnz) {
        if (x_cols_ptr[x_start + i] == dout_cols_ptr[dout_start + j]) {
          dx_values_ptr[x_start + i] = dout_values_ptr[dout_start + j];
          ++i;
          ++j;
        } else if (x_cols_ptr[x_start + i] < dout_cols_ptr[dout_start + j]) {
          dx_values_ptr[x_start + i] = static_cast<T>(0);
          ++i;
        } else {
          ++j;
        }
      }
      while (i < x_row_nnz) {
        dx_values_ptr[x_start + i] = static_cast<T>(0);
        ++i;
      }
    }
  }
}

template <typename T, typename IntT, typename Context>
void ConjugateCsrValues(const Context& dev_ctx,
                        const SparseCsrTensor& x,
                        SparseCsrTensor* x_conj) {
  AllocCsrPtr<T, IntT>(dev_ctx, x, x_conj);
  CopyCsrValues<T, IntT, Context>(dev_ctx, x, x, x_conj);
  DenseTensor x_conj_values = x_conj->values();
  x_conj_values = phi::Conj<T, Context>(dev_ctx, x_conj_values);
  DenseTensor x_conj_crows = x_conj->crows();
  DenseTensor x_conj_cols = x_conj->cols();
  x_conj->SetMember(x_conj_crows, x_conj_cols, x_conj_values, x_conj->dims());
}

template <typename T, typename IntT, typename Context>
void ConjugateCooValues(const Context& dev_ctx,
                        const SparseCooTensor& x,
                        SparseCooTensor* x_conj) {
  AllocCooPtr<T, IntT>(dev_ctx, x, x_conj);
  CopyCooValues<T, IntT, Context>(dev_ctx, x, x, x_conj);
  DenseTensor x_conj_values = x_conj->values();
  x_conj_values = phi::Conj<T, Context>(dev_ctx, x_conj_values);
  DenseTensor x_conj_indices = x_conj->indices();
  x_conj->SetMember(
      x_conj_indices, x_conj_values, x_conj->dims(), x_conj->coalesced());
}

template <typename T, typename IntT, typename Context>
void ElementWiseAddCsrGradCPUKernel(const Context& dev_ctx,
                                    const SparseCsrTensor& x,
                                    const SparseCsrTensor& y,
                                    const SparseCsrTensor& dout,
                                    SparseCsrTensor* dx,
                                    SparseCsrTensor* dy) {
  // Special case when y_grad is not needed
  if (dx != nullptr && dy == nullptr) {
    VLOG(4) << "Special case when dy is not needed";
    AllocCsrPtr<T, IntT>(dev_ctx, x, dx);
    CopyCsrValues<T, IntT, Context>(dev_ctx, dout, x, dx);
  } else if (dx == nullptr && dy != nullptr) {
    VLOG(4) << "Special case when dx is not needed";
    AllocCsrPtr<T, IntT>(dev_ctx, y, dy);
    CopyCsrValues<T, IntT, Context>(dev_ctx, dout, y, dy);
  } else {
    AllocCsrPtr<T, IntT>(dev_ctx, x, dx);
    AllocCsrPtr<T, IntT>(dev_ctx, y, dy);
    CopyCsrValues<T, IntT, Context>(dev_ctx, dout, x, dx);
    CopyCsrValues<T, IntT, Context>(dev_ctx, dout, y, dy);
  }
}

template <typename T, typename IntT, typename Context>
void ElementWiseSubtractCsrGradCPUKernel(const Context& dev_ctx,
                                         const SparseCsrTensor& x,
                                         const SparseCsrTensor& y,
                                         const SparseCsrTensor& dout,
                                         SparseCsrTensor* dx,
                                         SparseCsrTensor* dy) {
  if (dx) {
    AllocCsrPtr<T, IntT>(dev_ctx, x, dx);
    CopyCsrValues<T, IntT, Context>(dev_ctx, dout, x, dx);
  }

  if (dy) {
    AllocCsrPtr<T, IntT>(dev_ctx, y, dy);
    CopyCsrValues<T, IntT, Context>(dev_ctx, dout, y, dy);
    phi::NegativeKernel<T, Context>(
        dev_ctx, dout.values(), dy->mutable_values());
  }
}

template <typename T, typename IntT, typename Context>
void ElementWiseMultiplyCsrGradCPUKernel(const Context& dev_ctx,
                                         const SparseCsrTensor& x,
                                         const SparseCsrTensor& y,
                                         const SparseCsrTensor& dout,
                                         SparseCsrTensor* dx,
                                         SparseCsrTensor* dy) {
  if (dx) {
    AllocCsrPtr<T, IntT>(dev_ctx, x, dx);
    SparseCsrTensor tmp_dx;
    AllocCsrPtr<T, IntT>(dev_ctx, x, &tmp_dx);
    if (std::is_same<T, phi::dtype::complex<float>>::value ||
        std::is_same<T, phi::dtype::complex<double>>::value) {
      //    dout*y_conj
      SparseCsrTensor y_conj;
      ConjugateCsrValues<T, IntT, Context>(dev_ctx, y, &y_conj);
      sparse::ElementWiseMultiplyCsrKernel<T, Context>(
          dev_ctx, dout, y_conj, &tmp_dx);
    } else {
      //    dout*y
      sparse::ElementWiseMultiplyCsrKernel<T, Context>(
          dev_ctx, dout, y, &tmp_dx);
    }
    CopyCsrValues<T, IntT, Context>(dev_ctx, tmp_dx, x, dx);
  }

  if (dy) {
    AllocCsrPtr<T, IntT>(dev_ctx, y, dy);
    SparseCsrTensor tmp_dy;
    AllocCsrPtr<T, IntT>(dev_ctx, y, &tmp_dy);
    if (std::is_same<T, phi::dtype::complex<float>>::value ||
        std::is_same<T, phi::dtype::complex<double>>::value) {
      //    dout*x_conj
      SparseCsrTensor x_conj;
      ConjugateCsrValues<T, IntT, Context>(dev_ctx, x, &x_conj);
      sparse::ElementWiseMultiplyCsrKernel<T, Context>(
          dev_ctx, dout, x_conj, &tmp_dy);
    } else {
      //    dout*x
      sparse::ElementWiseMultiplyCsrKernel<T, Context>(
          dev_ctx, dout, x, &tmp_dy);
    }
    CopyCsrValues<T, IntT, Context>(dev_ctx, tmp_dy, y, dy);
  }
}

template <typename T, typename IntT, typename Context>
void ElementWiseDivideCsrGradCPUKernel(const Context& dev_ctx,
                                       const SparseCsrTensor& x,
                                       const SparseCsrTensor& y,
                                       const SparseCsrTensor& out,
                                       const SparseCsrTensor& dout,
                                       SparseCsrTensor* dx,
                                       SparseCsrTensor* dy) {
  if (dx) {
    AllocCsrPtr<T, IntT>(dev_ctx, x, dx);
    SparseCsrTensor tmp_dx;
    AllocCsrPtr<T, IntT>(dev_ctx, x, &tmp_dx);
    if (std::is_same<T, phi::dtype::complex<float>>::value ||
        std::is_same<T, phi::dtype::complex<double>>::value) {
      //    dout/y_conj
      SparseCsrTensor y_conj;
      ConjugateCsrValues<T, IntT, Context>(dev_ctx, y, &y_conj);
      sparse::ElementWiseDivideCsrKernel<T, Context>(
          dev_ctx, dout, y_conj, &tmp_dx);
    } else {
      //    dout/y
      sparse::ElementWiseDivideCsrKernel<T, Context>(dev_ctx, dout, y, &tmp_dx);
    }
    CopyCsrValues<T, IntT, Context>(dev_ctx, tmp_dx, x, dx);
  }

  if (dy) {
    //    -dout * out / y
    AllocCsrPtr<T, IntT>(dev_ctx, y, dy);
    SparseCsrTensor tmp_dy;
    AllocCsrPtr<T, IntT>(dev_ctx, y, &tmp_dy);

    Copy(dev_ctx, dout, dev_ctx.GetPlace(), false, &tmp_dy);
    phi::NegativeKernel<T, Context>(
        dev_ctx, dout.values(), tmp_dy.mutable_values());
    if (std::is_same<T, phi::dtype::complex<float>>::value ||
        std::is_same<T, phi::dtype::complex<double>>::value) {
      //    -dout * (out / y)_conj = -dout * out_conj / y_conj
      SparseCsrTensor out_conj;
      ConjugateCsrValues<T, IntT, Context>(dev_ctx, out, &out_conj);
      SparseCsrTensor y_conj;
      ConjugateCsrValues<T, IntT, Context>(dev_ctx, y, &y_conj);
      auto tmp =
          sparse::ElementWiseMultiplyCsr<T, Context>(dev_ctx, tmp_dy, out_conj);
      sparse::ElementWiseDivideCsrKernel<T, Context>(
          dev_ctx, tmp, y_conj, &tmp_dy);
    } else {
      auto tmp =
          sparse::ElementWiseMultiplyCsr<T, Context>(dev_ctx, tmp_dy, out);
      sparse::ElementWiseDivideCsrKernel<T, Context>(dev_ctx, tmp, y, &tmp_dy);
    }
    CopyCsrValues<T, IntT, Context>(dev_ctx, tmp_dy, y, dy);
  }
}

template <typename T, typename IntT, typename Context>
void ElementWiseAddCooGradCPUKernel(const Context& dev_ctx,
                                    const SparseCooTensor& x,
                                    const SparseCooTensor& y,
                                    const SparseCooTensor& dout,
                                    SparseCooTensor* dx,
                                    SparseCooTensor* dy) {
  //     Special case when y_grad is not needed*/
  if (dx != nullptr && dy == nullptr) {
    VLOG(4) << "Special case when dy is not needed";
    AllocCooPtr<T, IntT>(dev_ctx, x, dx);
    CopyCooValues<T, IntT, Context>(dev_ctx, dout, x, dx);
  } else if (dx == nullptr && dy != nullptr) {
    VLOG(4) << "Special case when dx is not needed";
    AllocCooPtr<T, IntT>(dev_ctx, y, dy);
    CopyCooValues<T, IntT, Context>(dev_ctx, dout, y, dy);
  } else {
    AllocCooPtr<T, IntT>(dev_ctx, x, dx);
    AllocCooPtr<T, IntT>(dev_ctx, y, dy);
    CopyCooValues<T, IntT, Context>(dev_ctx, dout, x, dx);
    CopyCooValues<T, IntT, Context>(dev_ctx, dout, y, dy);
  }
}

template <typename T, typename IntT, typename Context>
void ElementWiseSubtractCooGradCPUKernel(const Context& dev_ctx,
                                         const SparseCooTensor& x,
                                         const SparseCooTensor& y,
                                         const SparseCooTensor& dout,
                                         SparseCooTensor* dx,
                                         SparseCooTensor* dy) {
  if (dx) {
    AllocCooPtr<T, IntT>(dev_ctx, x, dx);
    CopyCooValues<T, IntT, Context>(dev_ctx, dout, x, dx);
  }

  if (dy) {
    AllocCooPtr<T, IntT>(dev_ctx, y, dy);
    CopyCooValues<T, IntT, Context>(dev_ctx, dout, y, dy);
    phi::NegativeKernel<T, Context>(
        dev_ctx, dout.values(), dy->mutable_values());
  }
}

template <typename T, typename IntT, typename Context>
void ElementWiseMultiplyCooGradCPUKernel(const Context& dev_ctx,
                                         const SparseCooTensor& x,
                                         const SparseCooTensor& y,
                                         const SparseCooTensor& dout,
                                         SparseCooTensor* dx,
                                         SparseCooTensor* dy) {
  if (dx) {
    AllocCooPtr<T, IntT>(dev_ctx, x, dx);
    SparseCooTensor tmp_dx;
    AllocCooPtr<T, IntT>(dev_ctx, x, &tmp_dx);
    if (std::is_same<T, phi::dtype::complex<float>>::value ||
        std::is_same<T, phi::dtype::complex<double>>::value) {
      //    dout*y_conj
      SparseCooTensor y_conj;
      ConjugateCooValues<T, IntT, Context>(dev_ctx, y, &y_conj);
      sparse::ElementWiseMultiplyCooKernel<T, Context>(
          dev_ctx, dout, y_conj, &tmp_dx);
    } else {
      //    dout*y
      sparse::ElementWiseMultiplyCooKernel<T, Context>(
          dev_ctx, dout, y, &tmp_dx);
    }
    CopyCooValues<T, IntT, Context>(dev_ctx, tmp_dx, x, dx);
  }

  if (dy) {
    AllocCooPtr<T, IntT>(dev_ctx, y, dy);
    SparseCooTensor tmp_dy;
    AllocCooPtr<T, IntT>(dev_ctx, y, &tmp_dy);
    if (std::is_same<T, phi::dtype::complex<float>>::value ||
        std::is_same<T, phi::dtype::complex<double>>::value) {
      //    dout*x_conj
      SparseCooTensor x_conj;
      ConjugateCooValues<T, IntT, Context>(dev_ctx, x, &x_conj);
      sparse::ElementWiseMultiplyCooKernel<T, Context>(
          dev_ctx, dout, x_conj, &tmp_dy);
    } else {
      //    dout*x
      sparse::ElementWiseMultiplyCooKernel<T, Context>(
          dev_ctx, dout, x, &tmp_dy);
    }
    CopyCooValues<T, IntT, Context>(dev_ctx, tmp_dy, y, dy);
  }
}

template <typename T, typename IntT, typename Context>
void ElementWiseDivideCooGradCPUKernel(const Context& dev_ctx,
                                       const SparseCooTensor& x,
                                       const SparseCooTensor& y,
                                       const SparseCooTensor& out,
                                       const SparseCooTensor& dout,
                                       SparseCooTensor* dx,
                                       SparseCooTensor* dy) {
  if (dx) {
    AllocCooPtr<T, IntT>(dev_ctx, x, dx);
    SparseCooTensor tmp_dx;
    AllocCooPtr<T, IntT>(dev_ctx, x, &tmp_dx);
    if (std::is_same<T, phi::dtype::complex<float>>::value ||
        std::is_same<T, phi::dtype::complex<double>>::value) {
      //    dout/y_conj
      SparseCooTensor y_conj;
      ConjugateCooValues<T, IntT, Context>(dev_ctx, y, &y_conj);
      sparse::ElementWiseDivideCooKernel<T, Context>(
          dev_ctx, dout, y_conj, &tmp_dx);
    } else {
      //    dout/y
      sparse::ElementWiseDivideCooKernel<T, Context>(dev_ctx, dout, y, &tmp_dx);
    }
    CopyCooValues<T, IntT, Context>(dev_ctx, tmp_dx, x, dx);
  }

  if (dy) {
    //    -dout * out / y
    AllocCooPtr<T, IntT>(dev_ctx, y, dy);
    SparseCooTensor tmp_dy;
    AllocCooPtr<T, IntT>(dev_ctx, y, &tmp_dy);
    Copy(dev_ctx, dout, dev_ctx.GetPlace(), false, &tmp_dy);
    phi::NegativeKernel<T, Context>(
        dev_ctx, dout.values(), tmp_dy.mutable_values());
    if (std::is_same<T, phi::dtype::complex<float>>::value ||
        std::is_same<T, phi::dtype::complex<double>>::value) {
      //    -dout * (out / y)_conj = -dout * out_conj / y_conj
      SparseCooTensor out_conj;
      ConjugateCooValues<T, IntT, Context>(dev_ctx, out, &out_conj);
      SparseCooTensor y_conj;
      ConjugateCooValues<T, IntT, Context>(dev_ctx, y, &y_conj);
      auto tmp =
          sparse::ElementWiseMultiplyCoo<T, Context>(dev_ctx, tmp_dy, out_conj);
      sparse::ElementWiseDivideCooKernel<T, Context>(
          dev_ctx, tmp, y_conj, &tmp_dy);
    } else {
      auto tmp =
          sparse::ElementWiseMultiplyCoo<T, Context>(dev_ctx, tmp_dy, out);
      sparse::ElementWiseDivideCooKernel<T, Context>(dev_ctx, tmp, y, &tmp_dy);
    }
    CopyCooValues<T, IntT, Context>(dev_ctx, tmp_dy, y, dy);
  }
}

template <typename T, typename Context>
void ElementWiseDivideCsrGradKernel(const Context& dev_ctx,
                                    const SparseCsrTensor& x,
                                    const SparseCsrTensor& y,
                                    const SparseCsrTensor& out,
                                    const SparseCsrTensor& dout,
                                    SparseCsrTensor* dx,
                                    SparseCsrTensor* dy) {
  PD_VISIT_BASE_INTEGRAL_TYPES(
      x.crows().dtype(), "ElementWiseDivideCsrGradCPUKernel", ([&] {
        ElementWiseDivideCsrGradCPUKernel<T, data_t>(
            dev_ctx, x, y, out, dout, dx, dy);
      }));
}
template <typename T, typename Context>
void ElementWiseDivideCooGradKernel(const Context& dev_ctx,
                                    const SparseCooTensor& x,
                                    const SparseCooTensor& y,
                                    const SparseCooTensor& out,
                                    const SparseCooTensor& dout,
                                    SparseCooTensor* dx,
                                    SparseCooTensor* dy) {
  PD_VISIT_BASE_INTEGRAL_TYPES(
      x.indices().dtype(), "ElementWiseDivideCooGradCPUKernel", ([&] {
        ElementWiseDivideCooGradCPUKernel<T, data_t>(
            dev_ctx, x, y, out, dout, dx, dy);
      }));
}

#define DEFINE_ELEMENTWISE_GRAD_KERNEL(name) \
  DEFINE_ELEMENTWISE_GRAD_KERNEL_CSR(name)   \
                                             \
  DEFINE_ELEMENTWISE_GRAD_KERNEL_COO(name)

#define DEFINE_ELEMENTWISE_GRAD_KERNEL_CSR(name)                         \
  template <typename T, typename Context>                                \
  void ElementWise##name##CsrGradKernel(const Context& dev_ctx,          \
                                        const SparseCsrTensor& x,        \
                                        const SparseCsrTensor& y,        \
                                        const SparseCsrTensor& dout,     \
                                        SparseCsrTensor* dx,             \
                                        SparseCsrTensor* dy) {           \
    PD_VISIT_BASE_INTEGRAL_TYPES(                                        \
        x.crows().dtype(), "ElementWise##name##CsrGradCPUKernel", ([&] { \
          ElementWise##name##CsrGradCPUKernel<T, data_t>(                \
              dev_ctx, x, y, dout, dx, dy);                              \
        }));                                                             \
  }

#define DEFINE_ELEMENTWISE_GRAD_KERNEL_COO(name)                           \
  template <typename T, typename Context>                                  \
  void ElementWise##name##CooGradKernel(const Context& dev_ctx,            \
                                        const SparseCooTensor& x,          \
                                        const SparseCooTensor& y,          \
                                        const SparseCooTensor& dout,       \
                                        SparseCooTensor* dx,               \
                                        SparseCooTensor* dy) {             \
    PD_VISIT_BASE_INTEGRAL_TYPES(                                          \
        x.indices().dtype(), "ElementWise##name##CooGradCPUKernel", ([&] { \
          ElementWise##name##CooGradCPUKernel<T, data_t>(                  \
              dev_ctx, x, y, dout, dx, dy);                                \
        }));                                                               \
  }

DEFINE_ELEMENTWISE_GRAD_KERNEL(Add)
DEFINE_ELEMENTWISE_GRAD_KERNEL(Subtract)
DEFINE_ELEMENTWISE_GRAD_KERNEL(Multiply)

}  // namespace phi::sparse

PD_REGISTER_KERNEL(add_csr_csr_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::ElementWiseAddCsrGradKernel,
                   float,
                   double,
                   int16_t,
                   int,
                   int64_t,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_CSR);
  kernel->InputAt(1).SetDataLayout(phi::DataLayout::SPARSE_CSR);
  kernel->InputAt(2).SetDataLayout(phi::DataLayout::SPARSE_CSR);
}

PD_REGISTER_KERNEL(subtract_csr_csr_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::ElementWiseSubtractCsrGradKernel,
                   float,
                   double,
                   int16_t,
                   int,
                   int64_t,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_CSR);
  kernel->InputAt(1).SetDataLayout(phi::DataLayout::SPARSE_CSR);
  kernel->InputAt(2).SetDataLayout(phi::DataLayout::SPARSE_CSR);
}

PD_REGISTER_KERNEL(multiply_csr_csr_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::ElementWiseMultiplyCsrGradKernel,
                   float,
                   double,
                   int16_t,
                   int,
                   int64_t,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_CSR);
  kernel->InputAt(1).SetDataLayout(phi::DataLayout::SPARSE_CSR);
  kernel->InputAt(2).SetDataLayout(phi::DataLayout::SPARSE_CSR);
}

PD_REGISTER_KERNEL(divide_csr_csr_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::ElementWiseDivideCsrGradKernel,
                   float,
                   double,
                   int16_t,
                   int,
                   int64_t,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_CSR);
  kernel->InputAt(1).SetDataLayout(phi::DataLayout::SPARSE_CSR);
  kernel->InputAt(2).SetDataLayout(phi::DataLayout::SPARSE_CSR);
  kernel->InputAt(3).SetDataLayout(phi::DataLayout::SPARSE_CSR);
}

PD_REGISTER_KERNEL(add_coo_coo_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::ElementWiseAddCooGradKernel,
                   float,
                   double,
                   int16_t,
                   int,
                   int64_t,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
  kernel->InputAt(1).SetDataLayout(phi::DataLayout::SPARSE_COO);
  kernel->InputAt(2).SetDataLayout(phi::DataLayout::SPARSE_COO);
}

PD_REGISTER_KERNEL(subtract_coo_coo_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::ElementWiseSubtractCooGradKernel,
                   float,
                   double,
                   int16_t,
                   int,
                   int64_t,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
  kernel->InputAt(1).SetDataLayout(phi::DataLayout::SPARSE_COO);
  kernel->InputAt(2).SetDataLayout(phi::DataLayout::SPARSE_COO);
}

PD_REGISTER_KERNEL(multiply_coo_coo_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::ElementWiseMultiplyCooGradKernel,
                   float,
                   double,
                   int16_t,
                   int,
                   int64_t,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
  kernel->InputAt(1).SetDataLayout(phi::DataLayout::SPARSE_COO);
  kernel->InputAt(2).SetDataLayout(phi::DataLayout::SPARSE_COO);
}

PD_REGISTER_KERNEL(divide_coo_coo_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::ElementWiseDivideCooGradKernel,
                   float,
                   double,
                   int16_t,
                   int,
                   int64_t,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
  kernel->InputAt(1).SetDataLayout(phi::DataLayout::SPARSE_COO);
  kernel->InputAt(2).SetDataLayout(phi::DataLayout::SPARSE_COO);
  kernel->InputAt(3).SetDataLayout(phi::DataLayout::SPARSE_COO);
}

PD_REGISTER_KERNEL(add_coo_dense_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::ElementWiseAddDenseGradKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
}
