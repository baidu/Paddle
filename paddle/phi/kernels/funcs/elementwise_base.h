/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/common/transform.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/funcs/common_shape.h"
#include "paddle/phi/kernels/funcs/elementwise_utils.h"
#include "paddle/phi/kernels/funcs/math_function.h"

#if defined(__NVCC__) || defined(__HIPCC__) || defined(__xpu__)
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/kernels/funcs/aligned_vector.h"
#include "paddle/phi/kernels/funcs/function_traits.h"
#include "paddle/phi/kernels/primitive/kernel_primitives.h"

#define HOSTDEVICE __host__ __device__
namespace kps = phi::kps;

#endif

namespace phi {

/* Packing scalar type T(float, int etc.) into Array<T, NumOuts> type
   for supporting multiple-output feature in elementwise system.*/
template <class T, int Num>
using ConditionalT = typename std::conditional_t<Num == 1, T, Array<T, Num>>;

namespace funcs {
using DDim = phi::DDim;

template <typename T, typename DeviceContext>
class RowwiseTransformIterator;

template <typename T, typename DeviceContext>
class MidWiseTransformIterator;

// NOTE(dzhwinter): ptrdiff_t in iterator is deprecated in c++17
template <typename T>
class RowwiseTransformIterator<T, CPUContext>
    : public std::iterator<std::random_access_iterator_tag,
                           T,
                           std::ptrdiff_t,
                           T *,
                           T &> {
 public:
  RowwiseTransformIterator(const T *ptr, int n) : ptr_(ptr), i_(0), n_(n) {}

  RowwiseTransformIterator<T, CPUContext> &operator++() {
    ++i_;
    if (UNLIKELY(i_ == n_)) {
      i_ = 0;
    }
    return *this;
  }

  RowwiseTransformIterator<T, CPUContext> &operator+(int n) {
    while (n-- > 0) {
      ++i_;
      if (UNLIKELY(i_ == n_)) {
        i_ = 0;
      }
    }

    return *this;
  }

  bool operator==(const RowwiseTransformIterator<T, CPUContext> &rhs) const {
    return (ptr_ + i_) == &(*rhs);
  }

  bool operator!=(const RowwiseTransformIterator<T, CPUContext> &rhs) const {
    return (ptr_ + i_) != &(*rhs);
  }

  const T &operator*() { return ptr_[i_]; }

 private:
  const T *ptr_;
  int i_;
  int64_t n_;
};

template <typename T>
class MidWiseTransformIterator<T, CPUContext>
    : public std::iterator<std::random_access_iterator_tag,
                           T,
                           std::ptrdiff_t,
                           T *,
                           T &> {
 public:
  MidWiseTransformIterator(const T *ptr, int n, int post)
      : ptr_(ptr), i_(0), j_(0), n_(n), post_(post) {}

  MidWiseTransformIterator<T, CPUContext> &operator++() {
    ++j_;
    if (UNLIKELY(j_ == post_)) {
      ++i_;
      j_ = 0;
      if (UNLIKELY(i_ == n_)) {
        i_ = 0;
      }
    }
    return *this;
  }

  MidWiseTransformIterator<T, CPUContext> &operator+(int n) {
    while (n-- > 0) {
      ++j_;
      if (UNLIKELY(j_ == post_)) {
        ++i_;
        j_ = 0;
        if (UNLIKELY(i_ == n_)) {
          i_ = 0;
        }
      }
    }
    return *this;
  }

  bool operator==(const MidWiseTransformIterator<T, CPUContext> &rhs) const {
    return (ptr_ + i_) == &(*rhs);
  }

  bool operator!=(const MidWiseTransformIterator<T, CPUContext> &rhs) const {
    return (ptr_ + i_) != &(*rhs);
  }

  const T &operator*() { return ptr_[i_]; }

 private:
  const T *ptr_;
  int64_t i_;
  int64_t j_;
  int64_t n_;
  int64_t post_;
};

#if defined(__NVCC__) || defined(__HIPCC__)
template <typename T>
class RowwiseTransformIterator<T, GPUContext>
    : public thrust::iterator_adaptor<RowwiseTransformIterator<T, GPUContext>,
                                      const T *> {
 public:
  typedef thrust::iterator_adaptor<RowwiseTransformIterator<T, GPUContext>,
                                   const T *>
      super_t;
  HOSTDEVICE RowwiseTransformIterator(const T *x, int n)
      : super_t(x), begin_(x), n_(n) {}
  friend class thrust::iterator_core_access;

 private:
  unsigned int n_;
  const T *begin_;
  HOSTDEVICE typename super_t::reference dereference() const {
    return *(begin_ + (this->base() - begin_) % n_);
  }
};

template <typename T>
class MidWiseTransformIterator<T, GPUContext>
    : public thrust::iterator_adaptor<MidWiseTransformIterator<T, GPUContext>,
                                      const T *> {
 public:
  typedef thrust::iterator_adaptor<MidWiseTransformIterator<T, GPUContext>,
                                   const T *>
      super_t;
  HOSTDEVICE MidWiseTransformIterator(const T *x, int n, int post)
      : super_t(x), begin_(x), n_(n), post_(post) {}
  friend class thrust::iterator_core_access;

 private:
  unsigned int post_;
  unsigned int n_;
  const T *begin_;
  HOSTDEVICE typename super_t::reference dereference() const {
    return *(begin_ + (((this->base() - begin_) / post_) % n_));
  }
};
#endif

template <typename Functor,
          typename T,
          typename DeviceContext,
          typename OutType = T>
class TransformFunctor {
 public:
  TransformFunctor(const DenseTensor &x,
                   const DenseTensor &y,
                   DenseTensor *z,
                   const DeviceContext &ctx,
                   Functor func,
                   const bool is_xsize_larger = true)
      : x_(x.data<T>()),
        y_(y.data<T>()),
        z_(ctx.template Alloc<OutType>(z)),
        nx_(x.numel()),
        ctx_(ctx),
        func_(func),
        is_xsize_larger_(is_xsize_larger) {
    if (is_xsize_larger_ == false) {
      nx_ = y.numel();
    }
  }

  inline void Run() const {
    phi::Transform<DeviceContext> trans;
    trans(ctx_, x_, x_ + nx_, y_, z_, func_);
  }

  inline void RunRowWise(int n) const {
    phi::Transform<DeviceContext> trans;
    if (is_xsize_larger_) {
      trans(ctx_,
            x_,
            x_ + nx_,
            RowwiseTransformIterator<T, DeviceContext>(y_, n),
            z_,
            func_);
    } else {
      trans(ctx_,
            y_,
            y_ + nx_,
            RowwiseTransformIterator<T, DeviceContext>(x_, n),
            z_,
            func_);
    }
  }

  inline void RunMidWise(int n, int post) const {
    phi::Transform<DeviceContext> trans;
    if (is_xsize_larger_) {
      trans(ctx_,
            x_,
            x_ + nx_,
            MidWiseTransformIterator<T, DeviceContext>(y_, n, post),
            z_,
            func_);
    } else {
      trans(ctx_,
            y_,
            y_ + nx_,
            MidWiseTransformIterator<T, DeviceContext>(x_, n, post),
            z_,
            func_);
    }
  }

 private:
  const T *x_;
  const T *y_;
  OutType *z_;
  int64_t nx_;
  const DeviceContext &ctx_;
  Functor func_;
  bool is_xsize_larger_;
};

template <typename Functor, typename T, typename OutType = T>
void CommonForwardBroadcastCPU(const DenseTensor &x,
                               const DenseTensor &y,
                               DenseTensor *z,
                               int *x_dims_array,
                               int *y_dims_array,
                               int *out_dims_array,
                               int max_dim,
                               const CPUContext &ctx,
                               Functor func,
                               const bool is_xsize_larger = true) {
  std::vector<int> index_array(max_dim, 0);
  const T *x_data = x.data<T>();
  const T *y_data = y.data<T>();
  PADDLE_ENFORCE_NOT_NULL(
      x_data, errors::InvalidArgument("The input X should not be empty."));
  PADDLE_ENFORCE_NOT_NULL(
      y_data, errors::InvalidArgument("The input Y should not be empty."));
  OutType *out_data = ctx.Alloc<OutType>(z);

  const int out_size = std::accumulate(
      out_dims_array, out_dims_array + max_dim, 1, std::multiplies<int>());
  int x_index, y_index;
  for (int out_index = 0; out_index < out_size; ++out_index) {
    x_index = GetElementwiseIndex(x_dims_array, max_dim, index_array.data());
    y_index = GetElementwiseIndex(y_dims_array, max_dim, index_array.data());
    if (is_xsize_larger) {
      out_data[out_index] = func(x_data[x_index], y_data[y_index]);
    } else {
      out_data[out_index] = func(y_data[y_index], x_data[x_index]);
    }

    UpdateElementwiseIndexArray(out_dims_array, max_dim, index_array.data());
  }
}

template <typename Functor, typename T, typename OutType = T>
void CommonElementwiseBroadcastForward(const CPUContext &dev_ctx,
                                       const DenseTensor &x,
                                       const DenseTensor &y,
                                       DenseTensor *z,
                                       const DDim &x_dims,
                                       const DDim &y_dims,
                                       Functor func,
                                       int axis,
                                       const bool is_xsize_larger = true) {
  int max_dim = (std::max)(x_dims.size(), y_dims.size());
  axis = (axis == -1 ? std::abs(x_dims.size() - y_dims.size()) : axis);
  PADDLE_ENFORCE_GE(
      axis,
      0,
      common::errors::InvalidArgument(
          "Axis should be great than or equal to 0, but received axis is %d.",
          axis));
  PADDLE_ENFORCE_LE(
      axis,
      max_dim,
      common::errors::InvalidArgument(
          "Axis should be less than or equal to %d, but received axis is %d.",
          max_dim,
          axis));
  std::vector<int> x_dims_array(max_dim);
  std::vector<int> y_dims_array(max_dim);
  std::vector<int> out_dims_array(max_dim);
  GetBroadcastDimsArrays(x_dims,
                         y_dims,
                         x_dims_array.data(),
                         y_dims_array.data(),
                         out_dims_array.data(),
                         max_dim,
                         axis);

  CommonForwardBroadcastCPU<Functor, T, OutType>(x,
                                                 y,
                                                 z,
                                                 x_dims_array.data(),
                                                 y_dims_array.data(),
                                                 out_dims_array.data(),
                                                 max_dim,
                                                 dev_ctx,
                                                 func,
                                                 is_xsize_larger);
}

// It is a common CPU implementation to compute binary calculation with the
// support of broadcast. Note:
// 1. CPU implementation cannot support the case when x needs broadcast, thus
//    this function need to be called with XxxFunctor and XxxInverseFunctor,
//    like AddFunctor and InverseAddFunctor.
// 2. The corresponding GPU implementation supports all the broadcast cases,
//    thus there is no need to define and call with XxxInverseFunctor.
// TODO(liuyiqun): optimize the CPU implementation to support all broadcast
// cases and avoid the need of XxxInverseFunctor.
template <typename Functor, typename T, typename OutType = T>
void ElementwiseCompute(const CPUContext &dev_ctx,
                        const DenseTensor &x,
                        const DenseTensor &y,
                        Functor func,
                        DenseTensor *z,
                        int axis = -1) {
  dev_ctx.Alloc<OutType>(z);
  auto x_dims = x.dims();
  auto y_dims = y.dims();
  bool is_xsize_larger = true;
  int max_dim = x_dims.size();
  if (x_dims.size() < y_dims.size()) {
    is_xsize_larger = false;
    max_dim = y_dims.size();
  }
  TransformFunctor<Functor, T, CPUContext, OutType> functor(
      x, y, z, dev_ctx, func, is_xsize_larger);
  if (x_dims == y_dims) {
    functor.Run();
    return;
  }

  axis = (axis == -1 ? std::abs(x_dims.size() - y_dims.size()) : axis);
  PADDLE_ENFORCE_GE(
      axis,
      0,
      errors::InvalidArgument(
          "Axis should be great than or equal to 0, but received axis is %d.",
          axis));
  PADDLE_ENFORCE_LE(
      axis,
      max_dim,
      errors::InvalidArgument(
          "Axis should be less than or equal to %d, but received axis is %d.",
          max_dim,
          axis));

  int pre, n, post, is_run_common_broadcast, axis_trim = 0;
  if (is_xsize_larger) {
    auto y_dims_trimmed = TrimTrailingSingularDims(y_dims);
    axis_trim = (y_dims_trimmed.size() == 0) ? x_dims.size() : axis;
    GetMidDims(x_dims,
               y_dims_trimmed,
               axis_trim,
               &pre,
               &n,
               &post,
               &is_run_common_broadcast);
  } else {
    auto x_dims_trimmed = TrimTrailingSingularDims(x_dims);
    axis_trim = (x_dims_trimmed.size() == 0) ? y_dims.size() : axis;
    GetMidDims(y_dims,
               x_dims_trimmed,
               axis_trim,
               &pre,
               &n,
               &post,
               &is_run_common_broadcast);
  }
  // special case for common implementation.
  // case 1: x=[2,3,1,5], y=[2,1,4,1]
  // case 2: x=[2,3,4], y=[1,1,4]
  if (is_run_common_broadcast == 1) {
    CommonElementwiseBroadcastForward<Functor, T, OutType>(
        dev_ctx, x, y, z, x_dims, y_dims, func, axis, is_xsize_larger);
    return;
  }

  if (post == 1) {
    functor.RunRowWise(n);
    return;
  } else {
    functor.RunMidWise(n, post);
    return;
  }
}

// for broadcast backwards
static inline std::vector<int> GetReduceDim(const DDim &in,
                                            const DDim &out,
                                            int axis) {
  axis =
      (axis == -1 ? std::abs(static_cast<int>(out.size() - in.size())) : axis);
  std::vector<int> dims;
  for (int i = 0; i < axis; ++i) {
    dims.push_back(i);
  }
  for (int i = 0; i < in.size(); ++i) {
    if (out[i + axis] != in[i]) {
      dims.push_back(i + axis);
    }
  }
  for (int i = axis + in.size(); i < out.size(); ++i) {
    dims.push_back(i);
  }
  return dims;
}

template <typename DeviceContext, typename T>
static inline void GetDoubleGradSafeTensor(const DeviceContext &dev_ctx,
                                           const DenseTensor &x,
                                           const DenseTensor *ddx,
                                           DenseTensor *ddx_safe) {
  if (ddx) {
    *ddx_safe = *ddx;
  } else {
    auto meta = phi::DenseTensorMeta(x.dtype(), x.dims(), x.layout());
    *ddx_safe = phi::Empty(dev_ctx, std::move(meta));
    dev_ctx.template Alloc<T>(ddx_safe);
    SetConstant<DeviceContext, T> set_zero;
    set_zero(dev_ctx, ddx_safe, static_cast<T>(0));
  }
}

inline void ElementwiseGradPreProcess(const DenseTensor &dout,
                                      DenseTensor *dx) {
  if (dx != nullptr) {
    dx->set_lod(dout.lod());
  }
}

#if defined(__NVCC__) || defined(__HIPCC__) || defined(__xpu__)

// static unroller
template <template <int Index, int VecSize> typename Func,
          int VecSize,
          int End,
          int Begin = 0>
struct Unroller {
  template <typename... Args>
  static HOSTDEVICE inline void step(Args &&...args) {
    Func<Begin, VecSize>::Apply(std::forward<Args>(args)...);
    Unroller<Func, VecSize, End, Begin + 1>::step(args...);
  }
};

template <template <int Index, int VecSize> typename Func, int VecSize, int End>
struct Unroller<Func, VecSize, End, End> {
  template <typename... Args>
  static HOSTDEVICE inline void step(Args &&...args) {}
};

// static unroller without VecSize for broadcast
template <template <int Index> typename Func, int End, int Begin = 0>
struct UnrollerWithoutVecSize {
  template <typename... Args>
  static HOSTDEVICE inline void step(Args &&...args) {
    Func<Begin>::Apply(std::forward<Args>(args)...);
    UnrollerWithoutVecSize<Func, End, Begin + 1>::step(args...);
  }
};

template <template <int Index> typename Func, int End>
struct UnrollerWithoutVecSize<Func, End, End> {
  template <typename... Args>
  static HOSTDEVICE inline void step(Args &&...args) {}
};

template <int Index, int VecSize>
struct Loader {
  template <typename Array, typename ArgsT>
  static __device__ __forceinline__ void Apply(const Array &in,
                                               ArgsT *args,
                                               kps::IndexType offset,
                                               int num,
                                               int read_lens,
                                               bool is_boundary) {
    using Type = std::tuple_element_t<Index, ArgsT>;
    kps::Init<Type, ArgsT, Index, VecSize>(
        args, static_cast<Type>(1.0f), read_lens);
    if (is_boundary) {
      kps::ReadData<Type, VecSize, 1, ArgsT, Index, true>(
          args,
          reinterpret_cast<const _ptr_ Type *>(in[Index]) + offset,
          num,
          read_lens);
    } else {
      kps::ReadData<Type, VecSize, 1, ArgsT, Index, false>(
          args,
          reinterpret_cast<const _ptr_ Type *>(in[Index]) + offset,
          num,
          read_lens);
    }
  }
};

template <int Index>
struct InputSetter {
  template <typename Array, typename ArgsT>
  static void Apply(const std::vector<const DenseTensor *> &ins_tensor,
                    const ArgsT &args,
                    Array *ins_data) {
    using Type = std::tuple_element_t<Index, ArgsT>;
    (*ins_data)[Index] = (const _ptr_ char *)(ins_tensor[Index]->data<Type>());
  }
};

static int GetVectorizedSizeForTensors(
    const std::vector<const DenseTensor *> &ins,
    const std::vector<DenseTensor *> &outs) {
#ifdef PADDLE_WITH_XPU_KP
  int vec_size = 256;
#else
  int vec_size = 4;
  for (size_t i = 0; i < ins.size(); ++i) {
    vec_size = std::min(vec_size, phi::GetVectorizedSize(ins[i]));
  }
  for (size_t i = 0; i < outs.size(); ++i) {
    vec_size = std::min(vec_size, phi::GetVectorizedSize(outs[i]));
  }
#endif
  return vec_size;
}

namespace detail {
template <class F, class Tuple, std::size_t... Index>
// GCC/Clang need the decltype() return type
HOSTDEVICE constexpr decltype(auto) ApplyImpl(F &&f,
                                              Tuple &&t,
                                              std::index_sequence<Index...>) {
  return std::forward<F>(f)(std::get<Index>(std::forward<Tuple>(t))...);
}
}  // namespace detail

template <class F, class Tuple>
HOSTDEVICE constexpr decltype(auto) Apply(F &&f, Tuple &&t) {
  return detail::ApplyImpl(
      std::forward<F>(f),
      std::forward<Tuple>(t),
      std::make_index_sequence<
          std::tuple_size<std::remove_reference_t<Tuple>>::value>{});
}

template <typename OutT,
          int VecSize,
          typename Functor,
          typename ArgsT,
          int Arity>
struct SameDimsElementwisePrimitiveCaller {
  __device__ inline void operator()(Functor func,
                                    ArgsT *args,
                                    OutT *result,
                                    int read_lens) {
#ifdef PADDLE_WITH_XPU_KP
    for (int idx = 0; idx < read_lens; ++idx) {
      result[idx] = static_cast<OutT>(Apply(func, args[idx]));
    }
#else
#pragma unroll
    for (int idx = 0; idx < VecSize; ++idx) {
      result[idx] = static_cast<OutT>(Apply(func, args[idx]));
    }
#endif
  }
};

template <typename OutT, int VecSize, bool IsBoundary, int NumOuts>
struct ElementwiseWriteDataCallerBc {
  __device__ __forceinline__ void operator()(
      Array<_ptr_ OutT *, NumOuts> outs,
      ConditionalT<OutT, NumOuts> src[VecSize],
      kps::IndexType block_offset,
      int num,
      int read_lens) {
    OutT dst[NumOuts][VecSize];
#pragma unroll
    for (int i = 0; i < read_lens; ++i) {
#pragma unroll
      for (int j = 0; j < NumOuts; ++j) {
        dst[j][i] = (src[i])[j];
      }
    }
#pragma unroll
    for (int i = 0; i < NumOuts; ++i) {
      kps::WriteData<OutT, VecSize, 1, IsBoundary>(
          outs[i] + block_offset, dst[i], num, read_lens);
    }
  }
};

template <typename OutT, int VecSize, bool IsBoundary>
struct ElementwiseWriteDataCallerBc<OutT, VecSize, IsBoundary, 1> {
  __device__ __forceinline__ void operator()(Array<_ptr_ OutT *, 1> outs,
                                             OutT src[VecSize],
                                             kps::IndexType block_offset,
                                             int num,
                                             int read_lens) {
    kps::WriteData<OutT, VecSize, 1, IsBoundary>(
        outs[0] + block_offset, src, num, read_lens);
  }
};

template <typename OutT,
          typename Functor,
          int Arity,
          int NumOuts,
          int VecSize,
          bool IsBoundary>
__device__ void VectorizedElementwiseKernelImpl(
    const Array<const _ptr_ char *__restrict__, Arity> &in,
    Array<_ptr_ OutT *, NumOuts> outs,
    kps::IndexType offset,
    int num,
    int read_lens,
    Functor func) {
  using Traits = phi::funcs::FunctionTraits<Functor>;
  using ArgsT = typename Traits::ArgsTuple;
  ArgsT args[VecSize];
  ConditionalT<OutT, NumOuts> result[VecSize];

  Unroller<Loader, VecSize, Arity>::step(
      in, args, offset, num, read_lens, IsBoundary);

  SameDimsElementwisePrimitiveCaller<ConditionalT<OutT, NumOuts>,
                                     VecSize,
                                     Functor,
                                     ArgsT,
                                     Arity>()(func, args, result, read_lens);

  ElementwiseWriteDataCallerBc<OutT, VecSize, IsBoundary, NumOuts>()(
      outs, result, offset, num, read_lens);
}

template <typename OutT, typename Functor, int Arity, int NumOuts, int VecSize>
__global__ void VectorizedElementwiseKernel(
    Array<const _ptr_ char *__restrict__, Arity> ins,
    Array<_ptr_ OutT *, NumOuts> outs,
    kps::IndexType numel,
    kps::IndexType main_offset,
    int read_lens,
    Functor func) {
  kps::IndexType data_offset =
      static_cast<kps::IndexType>(BLOCK_ID_X) * BLOCK_NUM_X * read_lens;
  kps::IndexType stride =
      static_cast<kps::IndexType>(BLOCK_NUM_X) * GRID_NUM_X * read_lens;
  for (; data_offset < main_offset; data_offset += stride) {
    VectorizedElementwiseKernelImpl<OutT,
                                    Functor,
                                    Arity,
                                    NumOuts,
                                    VecSize,
                                    false>(
        ins, outs, data_offset, read_lens * BLOCK_NUM_X, read_lens, func);
  }

  kps::IndexType remain = numel - data_offset;
  if (remain > 0) {
    VectorizedElementwiseKernelImpl<OutT,
                                    Functor,
                                    Arity,
                                    NumOuts,
                                    VecSize,
                                    true>(
        ins, outs, data_offset, static_cast<int>(remain), read_lens, func);
  }
}

template <typename OutT, typename Functor, int Arity, int NumOuts, int VecSize>
void LaunchElementwiseKernel(const KPDevice &ctx,
                             const std::vector<const DenseTensor *> &ins,
                             std::vector<DenseTensor *> *outs,
                             Functor func) {
  // There are at least 1 output, but maybe 0 input (ins.size() == 0).
  // For large tensor numel * sizeof(T) > 2^31, we must use int64_t as index
  // type.
  int64_t numel = (*outs)[0]->numel();
  Array<const _ptr_ char *__restrict__, Arity> ins_data;
  Array<_ptr_ OutT *, NumOuts> outs_data;

  using Traits = phi::funcs::FunctionTraits<Functor>;
  using ArgsT = typename Traits::ArgsTuple;
  ArgsT arg;
  UnrollerWithoutVecSize<InputSetter, Arity>::step(ins, arg, &ins_data);
  for (int i = 0; i < outs->size(); ++i) {
    outs_data[i] = (*outs)[i]->data<OutT>();
  }

#ifdef PADDLE_WITH_XPU_KP
  int block_size = 64;
  int grid_size = 8;
  int read_lens = kps::details::GetXpuReadLens(numel, block_size, grid_size);
  auto stream = ctx.x_context()->xpu_stream;
  int64_t main_offset =
      (numel / (read_lens * block_size)) * read_lens * block_size;
  VectorizedElementwiseKernel<OutT, Functor, Arity, NumOuts, VecSize>
      <<<grid_size, block_size, 0, stream>>>(
          ins_data, outs_data, numel, main_offset, read_lens, func);
#else
  auto gpu_config =
      phi::backends::gpu::GetGpuLaunchConfig1D(ctx, numel, VecSize);
  int64_t main_offset = (numel / (VecSize * gpu_config.GetBlockSize())) *
                        VecSize * gpu_config.GetBlockSize();
  auto stream = ctx.stream();
  VectorizedElementwiseKernel<OutT, Functor, Arity, NumOuts, VecSize>
      <<<gpu_config.block_per_grid, gpu_config.thread_per_block, 0, stream>>>(
          ins_data, outs_data, numel, main_offset, VecSize, func);
#endif
}

template <typename OutT, typename Functor, int Arity, int NumOuts = 1>
typename std::enable_if<!NeedVectorized<OutT>::value, void>::type
ElementwiseKernelForDifferentVecSize(
    const KPDevice &ctx,
    const std::vector<const DenseTensor *> &ins,
    std::vector<DenseTensor *> *outs,
    Functor func) {
  LaunchElementwiseKernel<OutT, Functor, Arity, NumOuts, VecSizeS>(
      ctx, ins, outs, func);
}

template <typename OutT, typename Functor, int Arity, int NumOuts = 1>
typename std::enable_if<NeedVectorized<OutT>::value, void>::type
ElementwiseKernelForDifferentVecSize(
    const KPDevice &ctx,
    const std::vector<const DenseTensor *> &ins,
    std::vector<DenseTensor *> *outs,
    Functor func) {
  // calculate the max vec_size for all ins and outs
  int vec_size = GetVectorizedSizeForTensors(ins, *outs);
  switch (vec_size) {
    case VecSizeL:
      LaunchElementwiseKernel<OutT, Functor, Arity, NumOuts, VecSizeL>(
          ctx, ins, outs, func);
      break;
    case VecSizeM:
      LaunchElementwiseKernel<OutT, Functor, Arity, NumOuts, VecSizeM>(
          ctx, ins, outs, func);
      break;
    case VecSizeS:
      LaunchElementwiseKernel<OutT, Functor, Arity, NumOuts, VecSizeS>(
          ctx, ins, outs, func);
      break;
    default: {
      PADDLE_THROW(common::errors::Unimplemented(
          "Unsupported vectorized size: %d !", vec_size));
      break;
    }
  }
}

template <typename OutT, typename Functor, int NumOuts = 1>
void ElementwiseKernel(const KPDevice &ctx,
                       const std::vector<const DenseTensor *> &ins,
                       std::vector<DenseTensor *> *outs,
                       Functor func) {
  using Traits = phi::funcs::FunctionTraits<Functor>;
  const int kArity = Traits::arity;
  PADDLE_ENFORCE_EQ(ins.size(),
                    kArity,
                    common::errors::InvalidArgument(
                        "The number of inputs is expected to be equal to the "
                        "arity of functor. But received: the number of inputs "
                        "is %d, the arity of functor is %d.",
                        ins.size(),
                        kArity));
  PADDLE_ENFORCE_EQ(outs->size(),
                    NumOuts,
                    common::errors::InvalidArgument(
                        "Number of outputs shall equal to number of functions, "
                        "but number of outputs is %d, of functions is %d.",
                        outs->size(),
                        NumOuts));

  for (int i = 0; i < outs->size(); ++i) {
    if (i > 0) {
      PADDLE_ENFORCE_EQ(
          (*outs)[i]->dims(),
          (*outs)[0]->dims(),
          common::errors::InvalidArgument(
              "The shape of each output tensor shall be identical yet, "
              "but %dth output tensor`s shape is not.",
              i));
    }
    ctx.template Alloc<OutT>((*outs)[i]);
  }

  ElementwiseKernelForDifferentVecSize<OutT, Functor, kArity, NumOuts>(
      ctx, ins, outs, func);
}

#endif

}  // namespace funcs
}  // namespace phi
