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

#include "paddle/phi/kernels/funcs/gather_scatter_functor.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {
namespace funcs {

class TensorAssign {
 public:
  // Dummy function, unused in actual execution(because specialized Assign
  // Kernel existed)
  template <typename tensor_t>
  __device__ static constexpr tensor_t I() noexcept {
    return tensor_t(0);
  }
  template <typename tensor_t>
  constexpr void operator()(tensor_t* self_data, tensor_t* src_data) const {
    *self_data = *src_data;
  }
};
static TensorAssign tensor_assign;

class ReduceAdd {
 public:
  template <typename tensor_t>
  __device__ static constexpr tensor_t I() noexcept {
    return tensor_t(0);
  }
  template <
      typename tensor_t,
      std::enable_if_t<!std::is_same<tensor_t, uint8_t>::value>* = nullptr>
  __device__ void operator()(tensor_t* self_data, tensor_t* src_data) const {
    phi::CudaAtomicAdd(self_data, *src_data);
  }
  template <typename tensor_t,
            std::enable_if_t<std::is_same<tensor_t, uint8_t>::value>* = nullptr>
  __device__ void operator()(tensor_t* self_data, tensor_t* src_data) const {
    *self_data += *src_data;
  }
};
static ReduceAdd reduce_add;

class ReduceMul {
 public:
  template <typename tensor_t>
  __device__ static constexpr tensor_t I() noexcept {
    return tensor_t(1);
  }
  template <
      typename tensor_t,
      std::enable_if_t<!std::is_same<tensor_t, uint8_t>::value>* = nullptr>
  __device__ void operator()(tensor_t* self_data, tensor_t* src_data) const {
    phi::CudaAtomicMul(self_data, *src_data);
  }
  template <typename tensor_t,
            std::enable_if_t<std::is_same<tensor_t, uint8_t>::value>* = nullptr>
  __device__ void operator()(tensor_t* self_data, tensor_t* src_data) const {
    *self_data *= *src_data;
  }
};
static ReduceMul reduce_mul;

class ReduceMax {
 public:
  template <typename tensor_t>
  __device__ static constexpr tensor_t I() noexcept {
    return tensor_t(-std::numeric_limits<tensor_t>::max());
  }
  template <
      typename tensor_t,
      std::enable_if_t<!std::is_same<tensor_t, uint8_t>::value>* = nullptr>
  __device__ void operator()(tensor_t* self_data, tensor_t* src_data) const {
    phi::CudaAtomicMax(self_data, *src_data);
  }
  template <typename tensor_t,
            std::enable_if_t<std::is_same<tensor_t, uint8_t>::value>* = nullptr>
  __device__ void operator()(tensor_t* self_data, tensor_t* src_data) const {
    *self_data = *src_data > *self_data ? *src_data : *self_data;
  }
};
static ReduceMax reduce_max;

class ReduceMin {
 public:
  template <typename tensor_t>
  __device__ static constexpr tensor_t I() noexcept {
    return tensor_t(std::numeric_limits<tensor_t>::max());
  }
  template <
      typename tensor_t,
      std::enable_if_t<!std::is_same<tensor_t, uint8_t>::value>* = nullptr>
  __device__ void operator()(tensor_t* self_data, tensor_t* src_data) const {
    phi::CudaAtomicMin(self_data, *src_data);
  }
  template <typename tensor_t,
            std::enable_if_t<std::is_same<tensor_t, uint8_t>::value>* = nullptr>
  __device__ void operator()(tensor_t* self_data, tensor_t* src_data) const {
    *self_data = *src_data < *self_data ? *src_data : *self_data;
  }
};
static ReduceMin reduce_min;

__global__ void CudaMemsetAsync(int* dest, int value, size_t size) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid * sizeof(int) >= size) return;
  dest[tid] = value;
}

template <typename tensor_t,
          typename index_t,
          typename func_t,
          bool is_scatter_like = true>
__global__ void ScatterAssignGPUKernel(tensor_t* self_data,
                                       int dim,
                                       const index_t* index_data,
                                       tensor_t* src_data,
                                       int select_dim_size,
                                       int self_select_dim_size,
                                       int src_select_dim_size,
                                       int64_t outer_dim_size,
                                       int64_t outer_dim_size_self,
                                       int64_t outer_dim_size_src,
                                       int64_t numel,
                                       int64_t numel_data,
                                       const func_t& reduce_op,
                                       int* thread_ids) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= numel) return;
  int64_t i, j, k;  // The i, j, k here is the index of the 3 layers loop
                    // squeezed from the N layers loop.
  /* tid = i * select_dim_size * outer_dim_size + j * outer_dim_size + k */
  i = tid / (select_dim_size * outer_dim_size);
  int64_t remind = tid % (select_dim_size * outer_dim_size);
  j = remind / outer_dim_size;
  k = remind % outer_dim_size;
  index_t index = index_data[tid];
  /*
    gather computation formula:

    self[i][j][k] = src[index[i][j][k]][j][k]  # if dim == 0
    self[i][j][k] = src[i][index[i][j][k]][k]  # if dim == 1
    self[i][j][k] = src[i][j][index[i][j][k]]  # if dim == 2

    scatter computation formula:

    self[index[i][j][k]][j][k] = src[i][j][k]  # if dim == 0
    self[i][index[i][j][k]][k] = src[i][j][k]  # if dim == 1
    self[i][j][index[i][j][k]] = src[i][j][k]  # if dim == 2

  */
  // index matrix has different shape with self matrix or src matrix.
  int64_t replace_index_self, replace_index_src;
  if (is_scatter_like) {
    // scatter
    PADDLE_ENFORCE(
        index >= -self_select_dim_size && index < self_select_dim_size,
        "The index is out of bounds, "
        "please check whether the index and "
        "input's shape meet the requirements. It should "
        "be greater or equal to [%d] and less than [%d], but received [%ld]",
        -self_select_dim_size,
        self_select_dim_size,
        (int64_t)index);
    if (index < 0) {
      index += self_select_dim_size;
    }
    replace_index_self = k + index * outer_dim_size_self +
                         i * outer_dim_size_self * self_select_dim_size;

    replace_index_src = k + j * outer_dim_size_src +
                        i * outer_dim_size_src * src_select_dim_size;
  } else {
    // gather
    PADDLE_ENFORCE(
        index >= -src_select_dim_size && index < src_select_dim_size,
        "The index is out of bounds, "
        "please check whether the index and "
        "input's shape meet the requirements. It should "
        "be greater or equal to [%d] and less than [%d], but received [%d]",
        -src_select_dim_size,
        src_select_dim_size,
        (int32_t)index);
    if (index < 0) {
      index += src_select_dim_size;
    }
    replace_index_self = tid;

    replace_index_src = k + index * outer_dim_size_src +
                        i * outer_dim_size_src * src_select_dim_size;
  }

  atomicMax(thread_ids + replace_index_self, tid);
  __syncthreads();

  if (tid == thread_ids[replace_index_self]) {
    reduce_op(static_cast<tensor_t*>(self_data + replace_index_self),
              static_cast<tensor_t*>(src_data + replace_index_src));
  }
}

template <typename tensor_t,
          typename index_t,
          typename func_t,
          bool is_scatter_like = true>
__global__ void GatherScatterGPUKernel(tensor_t* self_data,
                                       int dim,
                                       const index_t* index_data,
                                       tensor_t* src_data,
                                       int select_dim_size,
                                       int self_select_dim_size,
                                       int src_select_dim_size,
                                       int64_t outer_dim_size,
                                       int64_t outer_dim_size_self,
                                       int64_t outer_dim_size_src,
                                       int64_t numel,
                                       int64_t numel_data,
                                       bool include_self,
                                       const func_t& reduce_op,
                                       int* shared_mem) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= numel) return;
  int64_t i, j, k;  // The i, j, k here is the index of the 3 layers loop
                    // squeezed from the N layers loop.
  /* tid = i * select_dim_size * outer_dim_size + j * outer_dim_size + k */
  i = tid / (select_dim_size * outer_dim_size);
  int64_t remind = tid % (select_dim_size * outer_dim_size);
  j = remind / outer_dim_size;
  k = remind % outer_dim_size;
  index_t index = index_data[tid];
  /*
    gather computation formula:

    self[i][j][k] = src[index[i][j][k]][j][k]  # if dim == 0
    self[i][j][k] = src[i][index[i][j][k]][k]  # if dim == 1
    self[i][j][k] = src[i][j][index[i][j][k]]  # if dim == 2

    scatter computation formula:

    self[index[i][j][k]][j][k] = src[i][j][k]  # if dim == 0
    self[i][index[i][j][k]][k] = src[i][j][k]  # if dim == 1
    self[i][j][index[i][j][k]] = src[i][j][k]  # if dim == 2

  */
  // index matrix has different shape with self matrix or src matrix.
  int64_t replace_index_self, replace_index_src;
  if (is_scatter_like) {
    // scatter
    PADDLE_ENFORCE(
        index >= -self_select_dim_size && index < self_select_dim_size,
        "The index is out of bounds, "
        "please check whether the index and "
        "input's shape meet the requirements. It should "
        "be greater or equal to [%d] and less than [%d], but received [%ld]",
        -self_select_dim_size,
        self_select_dim_size,
        (int64_t)index);
    if (index < 0) {
      index += self_select_dim_size;
    }
    replace_index_self = k + index * outer_dim_size_self +
                         i * outer_dim_size_self * self_select_dim_size;

    replace_index_src = k + j * outer_dim_size_src +
                        i * outer_dim_size_src * src_select_dim_size;
  } else {
    // gather
    PADDLE_ENFORCE(
        index >= -src_select_dim_size && index < src_select_dim_size,
        "The index is out of bounds, "
        "please check whether the index and "
        "input's shape meet the requirements. It should "
        "be greater or equal to [%d] and less than [%d], but received [%d]",
        -src_select_dim_size,
        src_select_dim_size,
        (int32_t)index);
    if (index < 0) {
      index += src_select_dim_size;
    }
    replace_index_self = tid;

    replace_index_src = k + index * outer_dim_size_src +
                        i * outer_dim_size_src * src_select_dim_size;
  }
  /*
  bool is_op_done = false;
  if (include_self == false) {
    self_data[replace_index_self] = reduce_op.template I<tensor_t>();
    __syncthreads();
  }
  */
  reduce_op(static_cast<tensor_t*>(self_data + replace_index_self),
            static_cast<tensor_t*>(src_data + replace_index_src));
}

template <typename tensor_t,
          typename index_t,
          typename func_t,
          bool is_scatter_like = true>
__global__ void ScatterMeanGPUKernel(tensor_t* self_data,
                                     int dim,
                                     const index_t* index_data,
                                     tensor_t* src_data,
                                     int select_dim_size,
                                     int self_select_dim_size,
                                     int src_select_dim_size,
                                     int64_t outer_dim_size,
                                     int64_t outer_dim_size_self,
                                     int64_t outer_dim_size_src,
                                     int64_t numel,
                                     int64_t numel_data,
                                     bool include_self,
                                     const func_t& reduce_op,
                                     int* shared_mem) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= numel) return;

  int64_t i, j, k;  // The i, j, k here is the index of the 3 layers loop
                    // squeezed from the N layers loop.
  /* tid = i * select_dim_size * outer_dim_size + j * outer_dim_size + k */
  i = tid / (select_dim_size * outer_dim_size);
  int64_t remind = tid % (select_dim_size * outer_dim_size);
  j = remind / outer_dim_size;
  k = remind % outer_dim_size;
  index_t index = index_data[tid];
  /*
    gather computation formula:

    self[i][j][k] = src[index[i][j][k]][j][k]  # if dim == 0
    self[i][j][k] = src[i][index[i][j][k]][k]  # if dim == 1
    self[i][j][k] = src[i][j][index[i][j][k]]  # if dim == 2

    scatter computation formula:

    self[index[i][j][k]][j][k] = src[i][j][k]  # if dim == 0
    self[i][index[i][j][k]][k] = src[i][j][k]  # if dim == 1
    self[i][j][index[i][j][k]] = src[i][j][k]  # if dim == 2

  */
  // index matrix has different shape with self matrix or src matrix.
  int64_t replace_index_self, replace_index_src;
  if (is_scatter_like) {
    // scatter
    PADDLE_ENFORCE(
        index >= -self_select_dim_size && index < self_select_dim_size,
        "The index is out of bounds, "
        "please check whether the index and "
        "input's shape meet the requirements. It should "
        "be greater or equal to [%d] and less than [%d], but received [%ld]",
        -self_select_dim_size,
        self_select_dim_size,
        (int64_t)index);
    if (index < 0) {
      index += self_select_dim_size;
    }
    replace_index_self = k + index * outer_dim_size_self +
                         i * outer_dim_size_self * self_select_dim_size;

    replace_index_src = k + j * outer_dim_size_src +
                        i * outer_dim_size_src * src_select_dim_size;
  } else {
    // gather
    PADDLE_ENFORCE(
        index >= -src_select_dim_size && index < src_select_dim_size,
        "The index is out of bounds, "
        "please check whether the index and "
        "input's shape meet the requirements. It should "
        "be greater or equal to [%d] and less than [%d], but received [%d]",
        -src_select_dim_size,
        src_select_dim_size,
        (int32_t)index);
    if (index < 0) {
      index += src_select_dim_size;
    }
    replace_index_self = tid;

    replace_index_src = k + index * outer_dim_size_src +
                        i * outer_dim_size_src * src_select_dim_size;
  }
  if (include_self == false) {
    self_data[replace_index_self] = 0;
    __syncthreads();
  }
  reduce_op(static_cast<tensor_t*>(self_data + replace_index_self),
            static_cast<tensor_t*>(src_data + replace_index_src));

  phi::CudaAtomicMax(shared_mem + replace_index_self, tid);
  phi::CudaAtomicAdd(shared_mem + numel_data + replace_index_self, 1);
  __syncthreads();

  if (tid == shared_mem[replace_index_self]) {
    self_data[replace_index_self] =
        self_data[replace_index_self] /
        static_cast<tensor_t>(shared_mem[replace_index_self + numel_data]);
  }
}

template <typename tensor_t,
          typename index_t = int64_t,
          bool is_scatter_like = true>
struct gpu_gather_scatter_functor {
  template <typename func_t>
  void operator()(phi::DenseTensor self,
                  int dim,
                  const phi::DenseTensor& index,
                  phi::DenseTensor src,
                  const std::string& method_name,
                  const func_t& reduce_op,
                  bool include_self,
                  const phi::DeviceContext& ctx) {
    if (index.numel() == 0) {
      return;
    }
    auto* self_data = self.data<tensor_t>();
    auto* index_data = index.data<index_t>();
    auto* src_data = src.data<tensor_t>();
    int64_t self_size = self.numel();
    int64_t index_size = index.numel();
    int64_t src_size = src.numel();
    auto self_dims = self.dims();
    auto index_dims = index.dims();
    auto src_dims = src.dims();
    if (self_size == 0 || src_size == 0 || index_size == 0) return;
    int select_dim_size = index_dims[dim];
    // index matrix has different shape with self matrix or src matrix.
    int self_select_dim_size = self_dims[dim];
    int src_select_dim_size = src_dims[dim];
    int64_t outer_dim_size_self = 1;
    int64_t outer_dim_size_src = 1;
    int64_t inner_dim_size = 1;
    int64_t outer_dim_size = 1;
    for (int64_t i = 0; i < dim; ++i) {
      inner_dim_size *= index_dims[i];
    }

    for (int i = dim + 1; i < index_dims.size(); i++) {
      outer_dim_size *= index_dims[i];
      outer_dim_size_self *= self_dims[i];
      outer_dim_size_src *= src_dims[i];
    }

    int block = 512;
    int64_t n = inner_dim_size * select_dim_size * outer_dim_size;
    int64_t grid = (n + block - 1) / block;
    auto stream = reinterpret_cast<const phi::GPUContext&>(ctx).stream();
    DenseTensor shared_mem_tensor;
    if (method_name == "scatter_assign_gpu") {
      shared_mem_tensor.Resize({self_size});
      ctx.Alloc<int>(&shared_mem_tensor);
      phi::funcs::set_constant(ctx, &shared_mem_tensor, 0);

      int* shared_mem = shared_mem_tensor.data<int>();
      ScatterAssignGPUKernel<tensor_t, index_t, func_t, is_scatter_like>
          <<<grid, block, 0, stream>>>(self_data,
                                       dim,
                                       index_data,
                                       src_data,
                                       select_dim_size,
                                       self_select_dim_size,
                                       src_select_dim_size,
                                       outer_dim_size,
                                       outer_dim_size_self,
                                       outer_dim_size_src,
                                       index_size,
                                       self_size,
                                       reduce_op,
                                       shared_mem);
    } else if (method_name == "scatter_mean_gpu") {
      shared_mem_tensor.Resize({self_size * 2});
      ctx.Alloc<int>(&shared_mem_tensor);
      if (include_self) {
        int64_t grid_memset = (self_size * 2 + block - 1) / block;
        phi::funcs::set_constant(ctx, &shared_mem_tensor, 1);
      } else {
        phi::funcs::set_constant(ctx, &shared_mem_tensor, 0);
      }

      int* shared_mem = shared_mem_tensor.data<int>();
      ScatterMeanGPUKernel<tensor_t, index_t, func_t, is_scatter_like>
          <<<grid, block, 0, stream>>>(self_data,
                                       dim,
                                       index_data,
                                       src_data,
                                       select_dim_size,
                                       self_select_dim_size,
                                       src_select_dim_size,
                                       outer_dim_size,
                                       outer_dim_size_self,
                                       outer_dim_size_src,
                                       index_size,
                                       self_size,
                                       include_self,
                                       reduce_op,
                                       shared_mem);
    } else {
      int* shared_mem = nullptr;
      if (include_self == false) {
        DenseTensor Itensor;
        Itensor.Resize({src_size});
        ctx.Alloc<tensor_t>(&Itensor);
        phi::funcs::set_constant(ctx, &Itensor, reduce_op.template I<tensor_t>() );
        shared_mem_tensor.Resize({self_size});
        ctx.Alloc<int>(&shared_mem_tensor);
        phi::funcs::set_constant(ctx, &shared_mem_tensor, 0);

        int* shared_mem = shared_mem_tensor.data<int>();
        ScatterAssignGPUKernel<tensor_t, index_t, func_t, is_scatter_like>
            <<<grid, block, 0, stream>>>(self_data,
                                        dim,
                                        index_data,
                                        Itensor.data<tensor_t>(),
                                        select_dim_size,
                                        self_select_dim_size,
                                        src_select_dim_size,
                                        outer_dim_size,
                                        outer_dim_size_self,
                                        outer_dim_size_src,
                                        index_size,
                                        self_size,
                                        reduce_op,
                                        shared_mem);
      }
      GatherScatterGPUKernel<tensor_t, index_t, func_t, is_scatter_like>
          <<<grid, block, 0, stream>>>(self_data,
                                       dim,
                                       index_data,
                                       src_data,
                                       select_dim_size,
                                       self_select_dim_size,
                                       src_select_dim_size,
                                       outer_dim_size,
                                       outer_dim_size_self,
                                       outer_dim_size_src,
                                       index_size,
                                       self_size,
                                       include_self,
                                       reduce_op,
                                       shared_mem);
    }
  }
};  // struct gpu_gather_scatter_functor

template <typename tensor_t, typename index_t>
void gpu_gather_kernel(phi::DenseTensor self,
                       int dim,
                       const phi::DenseTensor& index,
                       phi::DenseTensor result,
                       bool include_self,
                       const phi::DeviceContext& ctx) {
  gpu_gather_scatter_functor<tensor_t,
                             index_t,
                             /*is_scatter_like=*/false>()(result,
                                                          dim,
                                                          index,
                                                          self,
                                                          "gather_out_gpu",
                                                          tensor_assign,
                                                          include_self,
                                                          ctx);
  return;
}

template <typename tensor_t, typename index_t>
void gpu_scatter_assign_kernel(phi::DenseTensor self,
                               int dim,
                               const phi::DenseTensor& index,
                               phi::DenseTensor src,
                               bool include_self,
                               const phi::DeviceContext& ctx) {
  gpu_gather_scatter_functor<tensor_t,
                             index_t,
                             /*is_scatter_like=*/true>()(self,
                                                         dim,
                                                         index,
                                                         src,
                                                         "scatter_assign_gpu",
                                                         tensor_assign,
                                                         include_self,
                                                         ctx);
}

template <typename tensor_t, typename index_t>
void gpu_scatter_add_kernel(phi::DenseTensor self,
                            int dim,
                            const phi::DenseTensor& index,
                            phi::DenseTensor src,
                            bool include_self,
                            const phi::DeviceContext& ctx) {
  gpu_gather_scatter_functor<tensor_t,
                             index_t,
                             /*is_scatter_like=*/true>()(
      self, dim, index, src, "scatter_add_gpu", reduce_add, include_self, ctx);
}

template <typename tensor_t, typename index_t>
void gpu_scatter_mul_kernel(phi::DenseTensor self,
                            int dim,
                            const phi::DenseTensor& index,
                            phi::DenseTensor src,
                            bool include_self,
                            const phi::DeviceContext& ctx) {
  gpu_gather_scatter_functor<tensor_t,
                             index_t,
                             /*is_scatter_like=*/true>()(
      self, dim, index, src, "scatter_mul_gpu", reduce_mul, include_self, ctx);
}

template <typename tensor_t, typename index_t>
void gpu_scatter_mean_kernel(phi::DenseTensor self,
                             int dim,
                             const phi::DenseTensor& index,
                             phi::DenseTensor src,
                             bool include_self,
                             const phi::DeviceContext& ctx) {
  gpu_gather_scatter_functor<tensor_t,
                             index_t,
                             /*is_scatter_like=*/true>()(
      self, dim, index, src, "scatter_mean_gpu", reduce_add, include_self, ctx);
}

template <typename tensor_t, typename index_t>
void gpu_scatter_max_kernel(phi::DenseTensor self,
                            int dim,
                            const phi::DenseTensor& index,
                            phi::DenseTensor src,
                            bool include_self,
                            const phi::DeviceContext& ctx) {
  gpu_gather_scatter_functor<tensor_t,
                             index_t,
                             /*is_scatter_like=*/true>()(
      self, dim, index, src, "scatter_max_gpu", reduce_max, include_self, ctx);
}

template <typename tensor_t, typename index_t>
void gpu_scatter_min_kernel(phi::DenseTensor self,
                            int dim,
                            const phi::DenseTensor& index,
                            phi::DenseTensor src,
                            bool include_self,
                            const phi::DeviceContext& ctx) {
  gpu_gather_scatter_functor<tensor_t,
                             index_t,
                             /*is_scatter_like=*/true>()(
      self, dim, index, src, "scatter_min_gpu", reduce_min, include_self, ctx);
}

template <typename tensor_t, typename index_t>
__global__ void ScatterInputGradGPUKernel(tensor_t* grad_data,
                                          int dim,
                                          const index_t* index_data,
                                          int select_dim_size,
                                          int grad_select_dim_size,
                                          int64_t outer_dim_size,
                                          int64_t outer_dim_size_data,
                                          int64_t numel,
                                          int64_t numel_data) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= numel) return;
  int64_t i, j, k;
  i = tid / (select_dim_size * outer_dim_size);
  int64_t remind = tid % (select_dim_size * outer_dim_size);
  j = remind / outer_dim_size;
  k = remind % outer_dim_size;
  index_t index = index_data[tid];
  int64_t replace_index = k + index * outer_dim_size_data +
                          i * outer_dim_size_data * grad_select_dim_size;

  grad_data[replace_index] = 0;
}
template <typename tensor_t, typename index_t>
void gpu_scatter_input_grad_kernel(phi::DenseTensor self,
                                   int dim,
                                   const phi::DenseTensor& index,
                                   phi::DenseTensor grad,
                                   bool include_self UNUSED,
                                   const phi::DeviceContext& ctx) {
  auto* index_data = index.data<index_t>();
  auto* grad_data = grad.data<tensor_t>();

  auto index_dims = index.dims();
  auto grad_dims = grad.dims();
  int64_t index_size = index.numel();
  int64_t grad_size = grad.numel();

  int64_t inner_dim_size = 1;
  int64_t outer_dim_size = 1;
  int64_t outer_dim_size_data = 1;
  int select_dim_size = index_dims[dim];
  int grad_select_dim_size = grad_dims[dim];
  for (int64_t i = 0; i < dim; ++i) {
    inner_dim_size *= index_dims[i];
  }

  for (int i = dim + 1; i < index_dims.size(); i++) {
    outer_dim_size *= index_dims[i];
    outer_dim_size_data *= grad_dims[i];
  }

  int block = 512;
  int64_t n = inner_dim_size * select_dim_size * outer_dim_size;
  int64_t grid = (n + block - 1) / block;
  auto stream = reinterpret_cast<const phi::GPUContext&>(ctx).stream();
  ScatterInputGradGPUKernel<tensor_t, index_t>
      <<<grid, block, 0, stream>>>(grad_data,
                                   dim,
                                   index_data,
                                   select_dim_size,
                                   grad_select_dim_size,
                                   outer_dim_size,
                                   outer_dim_size_data,
                                   index_size,
                                   grad_size);
}

template <typename tensor_t, typename index_t>
__global__ void ScatterMulInputGradGPUKernel(tensor_t* grad_data,
                                             int dim,
                                             const index_t* index_data,
                                             const tensor_t* out_data,
                                             const tensor_t* x_data,
                                             int select_dim_size,
                                             int grad_select_dim_size,
                                             int64_t outer_dim_size,
                                             int64_t outer_dim_size_grad,
                                             int64_t numel,
                                             int64_t numel_grad,
                                             int* thread_ids) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= numel) return;
  int64_t i, j, k;
  i = tid / (select_dim_size * outer_dim_size);
  int64_t remind = tid % (select_dim_size * outer_dim_size);
  j = remind / outer_dim_size;
  k = remind % outer_dim_size;
  index_t index = index_data[tid];
  int64_t replace_index = k + index * outer_dim_size_grad +
                          i * outer_dim_size_grad * grad_select_dim_size;
  atomicMax(thread_ids + replace_index, tid);
  __syncthreads();
  if (tid == thread_ids[replace_index]) {
    grad_data[replace_index] = grad_data[replace_index] *
                               out_data[replace_index] / x_data[replace_index];
  }
}

template <typename tensor_t, typename index_t>
__global__ void ScatterMinMaxInputGradGPUKernel(tensor_t* grad_data,
                                                int dim,
                                                const index_t* index_data,
                                                const tensor_t* out_data,
                                                const tensor_t* x_data,
                                                const tensor_t* value_data,
                                                const tensor_t* self_data,
                                                int select_dim_size,
                                                int grad_select_dim_size,
                                                int value_select_dim_size,
                                                int64_t outer_dim_size,
                                                int64_t outer_dim_size_grad,
                                                int64_t outer_dim_size_value,
                                                int64_t numel,
                                                int64_t numel_grad,
                                                const std::string& reduce,
                                                int* shared_mem) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= numel) return;
  int64_t i, j, k;
  i = tid / (select_dim_size * outer_dim_size);
  int64_t remind = tid % (select_dim_size * outer_dim_size);
  j = remind / outer_dim_size;
  k = remind % outer_dim_size;
  index_t index = index_data[tid];
  int64_t replace_index = k + index * outer_dim_size_grad +
                          i * outer_dim_size_grad * grad_select_dim_size;
  int64_t replace_index_value =
      k + j * outer_dim_size_value +
      i * outer_dim_size_value * value_select_dim_size;
  if (value_data[replace_index_value] == out_data[replace_index])
    phi::CudaAtomicAdd(shared_mem + replace_index, 1);
  __syncthreads();
  if (out_data[replace_index] != x_data[replace_index]) {
    grad_data[replace_index] = 0;
  } else {
    grad_data[replace_index] = self_data[replace_index] /
                               static_cast<tensor_t>(shared_mem[replace_index]);
  }
}

template <typename tensor_t, typename index_t>
void gpu_scatter_mul_min_max_input_grad_kernel(phi::DenseTensor self,
                                               int dim,
                                               const phi::DenseTensor& index,
                                               const phi::DenseTensor& out,
                                               const phi::DenseTensor& x,
                                               const phi::DenseTensor& value
                                                   UNUSED,
                                               phi::DenseTensor grad,
                                               const std::string& reduce,
                                               bool include_self UNUSED,
                                               const phi::DeviceContext& ctx) {
  auto* index_data = index.data<index_t>();
  auto* grad_data = grad.data<tensor_t>();
  auto* out_data = out.data<tensor_t>();
  auto* x_data = x.data<tensor_t>();
  auto* value_data = value.data<tensor_t>();
  auto* self_data = self.data<tensor_t>();

  int64_t grad_size = grad.numel();
  int64_t index_size = index.numel();
  auto index_dims = index.dims();
  auto grad_dims = grad.dims();
  auto x_dims = x.dims();
  auto value_dims = value.dims();

  int64_t inner_dim_size = 1;
  int64_t outer_dim_size = 1;
  int64_t outer_dim_size_grad = 1;
  int64_t outer_dim_size_value = 1;
  int64_t select_dim_size = index_dims[dim];
  int64_t grad_select_dim_size = grad_dims[dim];
  int64_t value_select_dim_size = grad_dims[dim];
  for (int i = 0; i < dim; ++i) {
    inner_dim_size *= index_dims[i];
  }

  for (int i = dim + 1; i < index_dims.size(); i++) {
    outer_dim_size *= index_dims[i];
    outer_dim_size_grad *= grad_dims[i];
    outer_dim_size_value *= value_dims[i];
  }
  int block = 512;
  int64_t n = inner_dim_size * select_dim_size * outer_dim_size;
  int64_t grid = (n + block - 1) / block;
  auto stream = reinterpret_cast<const phi::GPUContext&>(ctx).stream();
  DenseTensor shared_mem_tensor;
  shared_mem_tensor.Resize({grad_size});
  ctx.Alloc<int>(&shared_mem_tensor);
  int* shared_mem = shared_mem_tensor.data<int>();
  if (reduce == "mul" || reduce == "multiply") {
    phi::funcs::set_constant(ctx, &shared_mem_tensor, 0);
    ScatterMulInputGradGPUKernel<tensor_t, index_t>
        <<<grid, block, 0, stream>>>(grad_data,
                                     dim,
                                     index_data,
                                     out_data,
                                     x_data,
                                     select_dim_size,
                                     grad_select_dim_size,
                                     outer_dim_size,
                                     outer_dim_size_grad,
                                     index_size,
                                     grad_size,
                                     shared_mem);
  } else if (reduce == "amin" || reduce == "amax") {
    phi::funcs::set_constant(ctx, &shared_mem_tensor, 1);
    ScatterMinMaxInputGradGPUKernel<tensor_t, index_t>
        <<<grid, block, 0, stream>>>(grad_data,
                                     dim,
                                     index_data,
                                     out_data,
                                     x_data,
                                     value_data,
                                     self_data,
                                     select_dim_size,
                                     grad_select_dim_size,
                                     value_select_dim_size,
                                     outer_dim_size,
                                     outer_dim_size_grad,
                                     outer_dim_size_value,
                                     index_size,
                                     grad_size,
                                     reduce,
                                     shared_mem);
  }
}

template <typename tensor_t, typename index_t>
__global__ void ScatterMeanInputGradGPUKernel(tensor_t* grad_data,
                                              int dim,
                                              const index_t* index_data,
                                              int select_dim_size,
                                              int grad_select_dim_size,
                                              int64_t outer_dim_size,
                                              int64_t outer_dim_size_grad,
                                              int64_t numel,
                                              int64_t numel_grad,
                                              int* shared_mem) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= numel) return;
  int64_t i, j, k;
  i = tid / (select_dim_size * outer_dim_size);
  int64_t remind = tid % (select_dim_size * outer_dim_size);
  j = remind / outer_dim_size;
  k = remind % outer_dim_size;
  index_t index = index_data[tid];
  int64_t replace_index = k + index * outer_dim_size_grad +
                          i * outer_dim_size_grad * grad_select_dim_size;
  atomicMax(shared_mem + replace_index, tid);
  phi::CudaAtomicAdd(shared_mem + numel_grad + replace_index, 1);
  __syncthreads();
  if (tid == shared_mem[replace_index]) {
    grad_data[replace_index] =
        grad_data[replace_index] /
        static_cast<tensor_t>(shared_mem[numel_grad + replace_index]);
  }
}

template <typename tensor_t, typename index_t>
void gpu_scatter_mean_input_grad_kernel(phi::DenseTensor self,
                                        int dim,
                                        const phi::DenseTensor& index,
                                        phi::DenseTensor grad,
                                        bool include_self UNUSED,
                                        const phi::DeviceContext& ctx) {
  auto* index_data = index.data<index_t>();
  auto* grad_data = grad.data<tensor_t>();

  auto index_dims = index.dims();
  auto grad_dims = grad.dims();

  int64_t grad_size = grad.numel();
  int64_t index_size = index.numel();

  int64_t inner_dim_size = 1;
  int64_t outer_dim_size = 1;
  int64_t outer_dim_size_grad = 1;
  int64_t select_dim_size = index_dims[dim];
  int64_t grad_select_dim_size = grad_dims[dim];
  for (int i = 0; i < dim; ++i) {
    inner_dim_size *= index_dims[i];
  }

  for (int i = dim + 1; i < index_dims.size(); i++) {
    outer_dim_size *= index_dims[i];
    outer_dim_size_grad *= grad_dims[i];
  }

  DenseTensor shared_mem_tensor;
  shared_mem_tensor.Resize({grad_size * 2});
  ctx.Alloc<int>(&shared_mem_tensor);
  phi::funcs::set_constant(ctx, &shared_mem_tensor, 0);
  int* shared_mem = shared_mem_tensor.data<int>();

  int block = 512;
  int64_t grid_memset = (grad_size + block - 1) / block;
  auto stream = reinterpret_cast<const phi::GPUContext&>(ctx).stream();
  CudaMemsetAsync<<<grid_memset, block, 0, stream>>>(
      shared_mem + grad_size, 1, sizeof(int) * grad_size);

  int64_t n = inner_dim_size * select_dim_size * outer_dim_size;
  int64_t grid = (n + block - 1) / block;
  ScatterMeanInputGradGPUKernel<tensor_t, index_t>
      <<<grid, block, 0, stream>>>(grad_data,
                                   dim,
                                   index_data,
                                   select_dim_size,
                                   grad_select_dim_size,
                                   outer_dim_size,
                                   outer_dim_size_grad,
                                   index_size,
                                   grad_size,
                                   shared_mem);
}

template <typename tensor_t, typename index_t>
__global__ void ScatterValueGradGPUKernel(tensor_t* grad_data,
                                          int dim,
                                          const tensor_t* self_data,
                                          const index_t* index_data,
                                          int select_dim_size,
                                          int self_select_dim_size,
                                          int grad_select_dim_size,
                                          int64_t outer_dim_size,
                                          int64_t outer_dim_size_self,
                                          int64_t outer_dim_size_grad,
                                          int64_t numel,
                                          int64_t numel_data,
                                          int* thread_ids) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= numel) return;

  int64_t i, j, k;
  i = tid / (select_dim_size * outer_dim_size);
  int64_t remind = tid % (select_dim_size * outer_dim_size);
  j = remind / outer_dim_size;
  k = remind % outer_dim_size;
  index_t index = index_data[tid];
  int64_t replace_index_self = k + index * outer_dim_size_self +
                               i * outer_dim_size_self * self_select_dim_size;
  atomicMax(thread_ids + replace_index_self, tid);
  __syncthreads();

  if (tid == thread_ids[replace_index_self]) {
    int64_t replace_index_grad = k + j * outer_dim_size_grad +
                                 i * outer_dim_size_grad * grad_select_dim_size;
    grad_data[replace_index_grad] = self_data[replace_index_self];
  }
}
template <typename tensor_t, typename index_t>
void gpu_scatter_value_grad_kernel(phi::DenseTensor self,
                                   int dim,
                                   const phi::DenseTensor& index,
                                   phi::DenseTensor grad,
                                   bool include_self UNUSED,
                                   const phi::DeviceContext& ctx) {
  auto* self_data = self.data<tensor_t>();
  auto* index_data = index.data<index_t>();
  auto* grad_data = grad.data<tensor_t>();

  auto index_dims = index.dims();
  auto self_dims = self.dims();
  auto grad_dims = grad.dims();
  int64_t index_size = index.numel();
  int64_t self_size = self.numel();

  int64_t inner_dim_size = 1;
  int64_t outer_dim_size = 1;
  int64_t outer_dim_size_self = 1;
  int64_t outer_dim_size_grad = 1;
  int select_dim_size = index_dims[dim];
  int self_select_dim_size = self_dims[dim];
  int grad_select_dim_size = grad_dims[dim];
  for (int64_t i = 0; i < dim; ++i) {
    inner_dim_size *= index_dims[i];
  }

  for (int i = dim + 1; i < index_dims.size(); i++) {
    outer_dim_size *= index_dims[i];
    outer_dim_size_self *= self_dims[i];
    outer_dim_size_grad *= grad_dims[i];
  }

  DenseTensor shared_mem_tensor;
  shared_mem_tensor.Resize({self_size});
  ctx.Alloc<int>(&shared_mem_tensor);
  phi::funcs::set_constant(ctx, &shared_mem_tensor, 0);
  int* shared_mem = shared_mem_tensor.data<int>();

  int block = 512;
  int64_t n = inner_dim_size * select_dim_size * outer_dim_size;
  int64_t grid = (n + block - 1) / block;
  auto stream = reinterpret_cast<const phi::GPUContext&>(ctx).stream();
  ScatterValueGradGPUKernel<tensor_t, index_t>
      <<<grid, block, 0, stream>>>(grad_data,
                                   dim,
                                   self_data,
                                   index_data,
                                   select_dim_size,
                                   self_select_dim_size,
                                   grad_select_dim_size,
                                   outer_dim_size,
                                   outer_dim_size_self,
                                   outer_dim_size_grad,
                                   index_size,
                                   self_size,
                                   shared_mem);
}

template <typename tensor_t, typename index_t>
__global__ void ScatterMeanValueGradGPUKernel(tensor_t* grad_data,
                                              int dim,
                                              const tensor_t* self_data,
                                              const index_t* index_data,
                                              int select_dim_size,
                                              int self_select_dim_size,
                                              int grad_select_dim_size,
                                              int64_t outer_dim_size,
                                              int64_t outer_dim_size_self,
                                              int64_t outer_dim_size_grad,
                                              int64_t numel,
                                              int64_t numel_self,
                                              int* shared_mem) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= numel) return;

  int64_t i, j, k;
  i = tid / (select_dim_size * outer_dim_size);
  int64_t remind = tid % (select_dim_size * outer_dim_size);
  j = remind / outer_dim_size;
  k = remind % outer_dim_size;
  index_t index = index_data[tid];
  int64_t replace_index_self = k + index * outer_dim_size_self +
                               i * outer_dim_size_self * self_select_dim_size;

  phi::CudaAtomicAdd(shared_mem + replace_index_self, 1);
  __syncthreads();

  int64_t replace_index_grad = k + j * outer_dim_size_grad +
                               i * outer_dim_size_grad * grad_select_dim_size;
  grad_data[replace_index_grad] =
      self_data[replace_index_self] /
      static_cast<tensor_t>(shared_mem[replace_index_self]);
}

template <typename tensor_t, typename index_t>
__global__ void ScatterAddValueGradGPUKernel(tensor_t* grad_data,
                                             int dim,
                                             const tensor_t* self_data,
                                             const index_t* index_data,
                                             int select_dim_size,
                                             int self_select_dim_size,
                                             int grad_select_dim_size,
                                             int64_t outer_dim_size,
                                             int64_t outer_dim_size_self,
                                             int64_t outer_dim_size_grad,
                                             int64_t numel) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= numel) return;
  int64_t i, j, k;
  i = tid / (select_dim_size * outer_dim_size);
  int64_t remind = tid % (select_dim_size * outer_dim_size);
  j = remind / outer_dim_size;
  k = remind % outer_dim_size;
  index_t index = index_data[tid];
  int64_t replace_index_self = k + index * outer_dim_size_self +
                               i * outer_dim_size_self * self_select_dim_size;
  int64_t replace_index_grad = k + j * outer_dim_size_grad +
                               i * outer_dim_size_grad * grad_select_dim_size;
  grad_data[replace_index_grad] = self_data[replace_index_self];
}

template <typename tensor_t, typename index_t>
void gpu_scatter_add_mean_value_grad_kernel(
    phi::DenseTensor self,
    int dim,
    const phi::DenseTensor& index,
    const phi::DenseTensor& out UNUSED,
    const phi::DenseTensor& x UNUSED,
    const phi::DenseTensor& value UNUSED,
    phi::DenseTensor grad,
    const std::string& reduce,
    bool include_self,
    const phi::DeviceContext& ctx UNUSED) {
  auto* self_data = self.data<tensor_t>();
  auto* index_data = index.data<index_t>();
  auto* grad_data = grad.data<tensor_t>();

  auto index_dims = index.dims();
  auto self_dims = self.dims();
  auto grad_dims = grad.dims();

  int64_t self_size = self.numel();
  int64_t grad_size = grad.numel();
  int64_t index_size = index.numel();

  int64_t inner_dim_size = 1;
  int64_t outer_dim_size = 1;
  int64_t outer_dim_size_self = 1;
  int64_t outer_dim_size_grad = 1;
  int64_t select_dim_size = index_dims[dim];
  int64_t self_select_dim_size = self_dims[dim];
  int64_t grad_select_dim_size = grad_dims[dim];
  for (int i = 0; i < dim; ++i) {
    inner_dim_size *= index_dims[i];
  }

  for (int i = dim + 1; i < index_dims.size(); i++) {
    outer_dim_size *= index_dims[i];
    outer_dim_size_self *= self_dims[i];
    outer_dim_size_grad *= grad_dims[i];
  }
  int block = 512;
  int64_t n = inner_dim_size * select_dim_size * outer_dim_size;
  int64_t grid = (n + block - 1) / block;
  auto stream = reinterpret_cast<const phi::GPUContext&>(ctx).stream();
  if (reduce == "mean") {
    DenseTensor shared_mem_tensor;
    shared_mem_tensor.Resize({self_size});
    ctx.Alloc<int>(&shared_mem_tensor);
    if (include_self) {
      phi::funcs::set_constant(ctx, &shared_mem_tensor, 1);
    } else {
      phi::funcs::set_constant(ctx, &shared_mem_tensor, 0);
    }
    int* shared_mem = shared_mem_tensor.data<int>();
    ScatterMeanValueGradGPUKernel<tensor_t, index_t>
        <<<grid, block, 0, stream>>>(grad_data,
                                     dim,
                                     self_data,
                                     index_data,
                                     select_dim_size,
                                     self_select_dim_size,
                                     grad_select_dim_size,
                                     outer_dim_size,
                                     outer_dim_size_self,
                                     outer_dim_size_grad,
                                     index_size,
                                     self_size,
                                     shared_mem);
  } else if (reduce == "add") {
    ScatterAddValueGradGPUKernel<tensor_t, index_t>
        <<<grid, block, 0, stream>>>(grad_data,
                                     dim,
                                     self_data,
                                     index_data,
                                     select_dim_size,
                                     self_select_dim_size,
                                     grad_select_dim_size,
                                     outer_dim_size,
                                     outer_dim_size_self,
                                     outer_dim_size_grad,
                                     index_size);
  }
}

template <typename tensor_t, typename index_t>
__global__ void ScatterMulValueGradGPUKernel(tensor_t* grad_data,
                                             int dim,
                                             const index_t* index_data,
                                             const tensor_t* self_data,
                                             const tensor_t* value_data,
                                             const tensor_t* out_data,
                                             int select_dim_size,
                                             int self_select_dim_size,
                                             int grad_select_dim_size,
                                             int64_t outer_dim_size,
                                             int64_t outer_dim_size_self,
                                             int64_t outer_dim_size_grad,
                                             int64_t numel) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= numel) return;
  int64_t i, j, k;
  i = tid / (select_dim_size * outer_dim_size);
  int64_t remind = tid % (select_dim_size * outer_dim_size);
  j = remind / outer_dim_size;
  k = remind % outer_dim_size;
  index_t index = index_data[tid];
  int64_t replace_index_self = k + index * outer_dim_size_self +
                               i * outer_dim_size_self * self_select_dim_size;
  int64_t replace_index_grad = k + j * outer_dim_size_grad +
                               i * outer_dim_size_grad * grad_select_dim_size;
  grad_data[replace_index_grad] =
      self_data[replace_index_self] *
      (out_data[replace_index_self] / value_data[replace_index_grad]);
}

template <typename tensor_t, typename index_t>
__global__ void ScatterMinMaxValueGradGPUKernel(tensor_t* grad_data,
                                                int dim,
                                                const index_t* index_data,
                                                const tensor_t* self_data,
                                                const tensor_t* value_data,
                                                const tensor_t* out_data,
                                                const tensor_t* x_data,
                                                int select_dim_size,
                                                int self_select_dim_size,
                                                int grad_select_dim_size,
                                                int64_t outer_dim_size,
                                                int64_t outer_dim_size_self,
                                                int64_t outer_dim_size_grad,
                                                int64_t numel,
                                                int64_t numel_self,
                                                bool include_self,
                                                int* shared_mem) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= numel) return;
  int64_t i, j, k;
  i = tid / (select_dim_size * outer_dim_size);
  int64_t remind = tid % (select_dim_size * outer_dim_size);
  j = remind / outer_dim_size;
  k = remind % outer_dim_size;
  index_t index = index_data[tid];
  int64_t replace_index_self = k + index * outer_dim_size_self +
                               i * outer_dim_size_self * self_select_dim_size;
  int64_t replace_index_grad = k + j * outer_dim_size_grad +
                               i * outer_dim_size_grad * grad_select_dim_size;

  if (include_self &&
      x_data[replace_index_self] == out_data[replace_index_self])
    phi::CudaAtomicAdd(shared_mem + replace_index_self, 1);
  __syncthreads();
  grad_data[replace_index_grad] = 0;
  if (value_data[replace_index_grad] == out_data[replace_index_self])
    phi::CudaAtomicAdd(shared_mem + replace_index_self, 1);
  __syncthreads();
  if (value_data[replace_index_grad] == out_data[replace_index_self])
    grad_data[replace_index_grad] =
        self_data[replace_index_self] /
        static_cast<tensor_t>(shared_mem[replace_index_self]);
}

template <typename tensor_t, typename index_t>
void gpu_scatter_mul_min_max_value_grad_kernel(phi::DenseTensor self,
                                               int dim,
                                               const phi::DenseTensor& index,
                                               const phi::DenseTensor& out,
                                               const phi::DenseTensor& x,
                                               const phi::DenseTensor& value,
                                               phi::DenseTensor grad,
                                               const std::string& reduce,
                                               bool include_self,
                                               const phi::DeviceContext& ctx) {
  auto* self_data = self.data<tensor_t>();
  auto* index_data = index.data<index_t>();
  auto* grad_data = grad.data<tensor_t>();
  auto* out_data = out.data<tensor_t>();
  auto* x_data = x.data<tensor_t>();
  auto* value_data = value.data<tensor_t>();

  auto index_dims = index.dims();
  auto self_dims = self.dims();
  auto grad_dims = grad.dims();

  int64_t self_size = self.numel();
  int64_t index_size = index.numel();

  int64_t inner_dim_size = 1;
  int64_t outer_dim_size = 1;
  int64_t outer_dim_size_self = 1;
  int64_t outer_dim_size_grad = 1;
  int64_t select_dim_size = index_dims[dim];
  int64_t self_select_dim_size = self_dims[dim];
  int64_t grad_select_dim_size = grad_dims[dim];
  for (int i = 0; i < dim; ++i) {
    inner_dim_size *= index_dims[i];
  }

  for (int i = dim + 1; i < index_dims.size(); i++) {
    outer_dim_size *= index_dims[i];
    outer_dim_size_self *= self_dims[i];
    outer_dim_size_grad *= grad_dims[i];
  }
  int block = 512;
  int64_t n = inner_dim_size * select_dim_size * outer_dim_size;
  int64_t grid = (n + block - 1) / block;
  auto stream = reinterpret_cast<const phi::GPUContext&>(ctx).stream();
  if (reduce == "mul" || reduce == "multiply") {
    ScatterMulValueGradGPUKernel<tensor_t, index_t>
        <<<grid, block, 0, stream>>>(grad_data,
                                     dim,
                                     index_data,
                                     self_data,
                                     value_data,
                                     out_data,
                                     select_dim_size,
                                     self_select_dim_size,
                                     grad_select_dim_size,
                                     outer_dim_size,
                                     outer_dim_size_self,
                                     outer_dim_size_grad,
                                     index_size);
  } else if (reduce == "amin" || reduce == "amax") {
    DenseTensor shared_mem_tensor;
    shared_mem_tensor.Resize({self_size});
    ctx.Alloc<int>(&shared_mem_tensor);
    phi::funcs::set_constant(ctx, &shared_mem_tensor, 0);

    int* shared_mem = shared_mem_tensor.data<int>();
    ScatterMinMaxValueGradGPUKernel<tensor_t, index_t>
        <<<grid, block, 0, stream>>>(grad_data,
                                     dim,
                                     index_data,
                                     self_data,
                                     value_data,
                                     out_data,
                                     x_data,
                                     select_dim_size,
                                     self_select_dim_size,
                                     grad_select_dim_size,
                                     outer_dim_size,
                                     outer_dim_size_self,
                                     outer_dim_size_grad,
                                     index_size,
                                     self_size,
                                     include_self,
                                     shared_mem);
  }
}

Instantiate_Template_Function(gpu_gather_kernel)                  // NOLINT
    Instantiate_Template_Function(gpu_scatter_assign_kernel)      // NOLINT
    Instantiate_Template_Function(gpu_scatter_add_kernel)         // NOLINT
    Instantiate_Template_Function(gpu_scatter_mul_kernel)         // NOLINT
    Instantiate_Template_Function(gpu_scatter_min_kernel)         // NOLINT
    Instantiate_Template_Function(gpu_scatter_max_kernel)         // NOLINT
    Instantiate_Template_Function(gpu_scatter_mean_kernel)        // NOLINT
    Instantiate_Template_Function(gpu_scatter_input_grad_kernel)  // NOLINT
    Instantiate_Template_Function(gpu_scatter_value_grad_kernel)  // NOLINT
    Instantiate_Template_Function_With_Out(
        gpu_scatter_mul_min_max_input_grad_kernel)                     // NOLINT
    Instantiate_Template_Function(gpu_scatter_mean_input_grad_kernel)  // NOLINT
    Instantiate_Template_Function_With_Out(
        gpu_scatter_add_mean_value_grad_kernel)  // NOLINT
    Instantiate_Template_Function_With_Out(
        gpu_scatter_mul_min_max_value_grad_kernel)  // NOLINT
}  // namespace funcs
}  // namespace phi
