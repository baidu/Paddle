// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/lars_momentum_kernel.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/aligned_vector.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/math_cuda_utils.h"
#include "paddle/utils/optional.h"

#if CUDA_VERSION >= 11000
#include <cooperative_groups.h>
#endif

#define LARS_BLOCK_SIZE 512

#define LARS_MAX_MERGED_OPS 60

namespace phi {

template <typename T>
using MultiPrecisionType = typename phi::dtype::MPTypeTrait<T>::Type;

__device__ __forceinline__ float Sqrt(float x) { return sqrtf(x); }
__device__ __forceinline__ double Sqrt(double x) { return sqrt(x); }
__device__ __forceinline__ float Fma(float x, float y, float z) {
  return fmaf(x, y, z);
}
__device__ __forceinline__ double Fma(double x, double y, double z) {
  return fma(x, y, z);
}

template <typename T>
class LarsThreadConfig {
 public:
  int grid_for_norm;
  int grid_for_lars;
#if CUDA_VERSION >= 11000

 private:
  int grid_stride;

 public:
  explicit LarsThreadConfig(int64_t numel, int sm_num, int num_blocks_per_sm) {
    int grid = (numel + LARS_BLOCK_SIZE - 1) / LARS_BLOCK_SIZE;
    grid_for_lars =
        std::min(std::min(sm_num * num_blocks_per_sm, grid), LARS_BLOCK_SIZE);
    grid_stride = LARS_BLOCK_SIZE * grid_for_lars;
  }

  int GetRepeatTimes(int64_t numel) {
    return (numel + grid_stride - 1) / grid_stride - 1;
  }
#else
  int repeat_times;
  explicit LarsThreadConfig(const int64_t numel) {
    int grid = (numel + LARS_BLOCK_SIZE - 1) / LARS_BLOCK_SIZE;
    grid_for_norm = std::min(grid, LARS_BLOCK_SIZE);
    const int grid_stride = grid_for_norm * LARS_BLOCK_SIZE;
    repeat_times = (numel + grid_stride - 1) / grid_stride - 1;
    // Determine to read 4 fp16 or float data once, but 2 double data once.
    grid_for_lars =
        std::is_same<double, T>::value
            ? (numel + (LARS_BLOCK_SIZE << 1) - 1) / (LARS_BLOCK_SIZE << 1)
            : (numel + (LARS_BLOCK_SIZE << 2) - 1) / (LARS_BLOCK_SIZE << 2);
  }
#endif
};

template <typename T, typename MT, int VecSize, bool IsAmp = false>
__device__ inline void VectorizeLarsUpdate(const T* __restrict__ grad,
                                           const MT* param,
                                           const MT* velocity,
                                           T* param_out,
                                           MT* velocity_out,
                                           const MT mu,
                                           MT local_lr,
                                           const MT lars_weight_decay,
                                           const MT rescale_grad,
                                           const int tid,
                                           const int grid_stride,
                                           const int numel,
                                           MT* master_param_out = nullptr) {
  using VecType = phi::AlignedVector<T, VecSize>;
  using VecMType = phi::AlignedVector<MT, VecSize>;
  int main = numel >> (VecSize >> 1);
  int tail_offset = main * VecSize;

  const VecType* grad_vec = reinterpret_cast<const VecType*>(grad);
  const VecMType* param_vec = reinterpret_cast<const VecMType*>(param);
  const VecMType* velocity_vec = reinterpret_cast<const VecMType*>(velocity);
  VecType* param_out_vec = reinterpret_cast<VecType*>(param_out);
  VecMType* velocity_out_vec = reinterpret_cast<VecMType*>(velocity_out);

  VecMType* master_param_out_vec;
  if (IsAmp) {
    master_param_out_vec = reinterpret_cast<VecMType*>(master_param_out);
  }

  for (int i = tid; i < main; i += grid_stride) {
    VecType param_out_tmp;
    VecMType velocity_tmp, param_tmp;
    VecType grad_data = grad_vec[i];
    VecMType param_data = param_vec[i];
    VecMType velocity_data = velocity_vec[i];
#pragma unroll
    for (int j = 0; j < VecSize; ++j) {
      MT grad_val = static_cast<MT>(grad_data[j]) * rescale_grad;
      velocity_tmp[j] =
          Fma(velocity_data[j],
              mu,
              local_lr * Fma(lars_weight_decay, param_data[j], grad_val));
      param_tmp[j] = param_data[j] - velocity_tmp[j];
      param_out_tmp[j] = static_cast<T>(param_tmp[j]);
    }
    param_out_vec[i] = param_out_tmp;
    velocity_out_vec[i] = velocity_tmp;
    if (IsAmp) {
      master_param_out_vec[i] = param_tmp;
    }
  }

  for (int i = tid + tail_offset; i < numel; i += grid_stride) {
    MT grad_val = static_cast<MT>(grad[i]) * rescale_grad;
    MT param_val = param[i];
    MT velocity_tmp =
        Fma(velocity[i],
            mu,
            local_lr * Fma(lars_weight_decay, param_val, grad_val));
    MT param_tmp = param_val - velocity_tmp;
    param_out[i] = static_cast<T>(param_tmp);
    velocity_out[i] = velocity_tmp;
    if (IsAmp) {
      master_param_out[i] = param_tmp;
    }
  }
}

#if CUDA_VERSION >= 11000
/* Once CUDA_VERSION is beyond 11, cooperative_groups can be involved in without
  --rdc=true compile flag, then L2_norm kernel can be set with __device__ and
  cooperative_groups::grid_group also can be involved. Otherwise, adding this
  flag may affect much, L2_norm kernel shall be set with __global__.*/
// TODO(limingshu): declaration of cooperative_groups wapper is invalid in host.
template <typename T, typename MT>
__forceinline__ __device__ void L2NormKernel(
    const cooperative_groups::grid_group* cg,
#else
template <typename T, typename MT>
__global__ void L2NormKernel(
#endif
    const T* p_data,
    const T* __restrict__ g_data,
    MT* __restrict__ p_buffer,
    MT* __restrict__ g_buffer,
    const int64_t numel,
    const int repeat_times,
    const MT rescale_grad,
    const int thresh = 0,
    MT* __restrict__ p_n = nullptr,
    MT* __restrict__ g_n = nullptr) {
  __shared__ MT s_buffer[2];
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  int grid_stride = LARS_BLOCK_SIZE * gridDim.x;

  MT p_tmp = static_cast<MT>(0);
  MT g_tmp = static_cast<MT>(0);
  while (tid < numel) {
    MT tmp0 = static_cast<MT>(p_data[tid]);
    MT tmp1 = static_cast<MT>(g_data[tid]);
    p_tmp += (tmp0 * tmp0);
    g_tmp += (tmp1 * tmp1);
    tid += grid_stride;
  }
  p_tmp = phi::funcs::BlockReduceSum<MT>(p_tmp, FINAL_MASK);
  g_tmp = phi::funcs::BlockReduceSum<MT>(g_tmp, FINAL_MASK);

  if (threadIdx.x == 0) {
    p_buffer[blockIdx.x] = p_tmp;
    g_buffer[blockIdx.x] = g_tmp;
  }
#if CUDA_VERSION >= 11000
  cg->sync();  // Grid sync for writring partial result to global memory
  MT p_part_sum = threadIdx.x < gridDim.x ? p_buffer[threadIdx.x] : 0;
  MT g_part_sum = threadIdx.x < gridDim.x ? g_buffer[threadIdx.x] : 0;
  MT tmp0 = phi::funcs::BlockReduceSum<MT>(p_part_sum, FINAL_MASK);
  MT tmp1 = phi::funcs::BlockReduceSum<MT>(g_part_sum, FINAL_MASK);
  if (threadIdx.x == 0) {
    s_buffer[0] = tmp0;
    s_buffer[1] = tmp1;
  }
  __syncthreads();
  *p_n = Sqrt(s_buffer[0]);
  *g_n = rescale_grad * Sqrt(s_buffer[1]);
#endif
}

template <typename T, typename MT>
__forceinline__ __device__ void MomentumUpdate(
    const T* param,
    const T* __restrict__ grad,
    const MT* velocity,
    T* param_out,
    MT* velocity_out,
    const MT* master_param,
    MT* master_param_out,
    const MT* __restrict__ learning_rate,
    const MT mu,
    const MT lars_weight_decay,
    const MT lars_coeff,
    const MT epsilon,
    const MT rescale_grad,
    const MT param_norm,
    const MT grad_norm,
    const int tid,
    const int grid_stride,
    const int64_t numel,
    const bool is_amp) {
  const MT lr = learning_rate[0];
  MT local_lr = lr;
  if (param_norm > static_cast<MT>(0) && grad_norm > static_cast<MT>(0)) {
    local_lr = lr * lars_coeff * param_norm /
               (fma(lars_weight_decay, param_norm, grad_norm) + epsilon);
  }
  if (is_amp) {
    VectorizeLarsUpdate<T, MT, /*VecSize=*/4, /*IsAmp=*/true>(grad,
                                                              master_param,
                                                              velocity,
                                                              param_out,
                                                              velocity_out,
                                                              mu,
                                                              local_lr,
                                                              lars_weight_decay,
                                                              rescale_grad,
                                                              tid,
                                                              grid_stride,
                                                              numel,
                                                              master_param_out);
  } else {
    if (std::is_same<T, float>::value ||
        std::is_same<T, dtype::float16>::value) {
      /* TODO(limingshu): pointer cast may damage memory accessing for fp16 */
      VectorizeLarsUpdate<T, MT, /*VecSize=*/4, /*IsAmp=*/false>(
          grad,
          reinterpret_cast<const MT*>(param),
          velocity,
          param_out,
          velocity_out,
          mu,
          local_lr,
          lars_weight_decay,
          rescale_grad,
          tid,
          grid_stride,
          numel);
    } else {
      VectorizeLarsUpdate<T, MT, /*VecSize=*/2, /*IsAmp=*/false>(
          grad,
          reinterpret_cast<const MT*>(param),
          velocity,
          param_out,
          velocity_out,
          mu,
          local_lr,
          lars_weight_decay,
          rescale_grad,
          tid,
          grid_stride,
          numel);
    }
  }
}

#if CUDA_VERSION >= 11000
template <typename T, typename MT>
struct LarsParamWrapper {
  int64_t numel_arr[LARS_MAX_MERGED_OPS];
  int repeat_arr[LARS_MAX_MERGED_OPS];
  const T* __restrict__ g_arr[LARS_MAX_MERGED_OPS];
  const MT* __restrict__ lr_arr[LARS_MAX_MERGED_OPS];
  T* __restrict__ p_out_arr[LARS_MAX_MERGED_OPS];
  MT* __restrict__ v_out_arr[LARS_MAX_MERGED_OPS];
  MT* __restrict__ master_p_out_arr[LARS_MAX_MERGED_OPS];
  MT weight_decay_arr[LARS_MAX_MERGED_OPS];
};

template <typename T, typename MT>
__global__ void MergedMomentumLarsKernel(LarsParamWrapper<T, MT> lars_wrapper,
                                         MT* __restrict__ p_buffer,
                                         MT* __restrict__ g_buffer,
                                         const int op_num,
                                         const MT mu,
                                         const MT lars_coeff,
                                         const MT epsilon,
                                         const MT rescale_grad,
                                         const bool is_amp) {
  int grid_stride = gridDim.x * LARS_BLOCK_SIZE;
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  const cooperative_groups::grid_group cg = cooperative_groups::this_grid();
  for (int i = 0; i < op_num; ++i) {
    int numel = lars_wrapper.numel_arr[i];
    MT param_norm = static_cast<MT>(0);
    MT grad_norm = static_cast<MT>(0);
    L2NormKernel<T, MT>(&cg,
                        lars_wrapper.p_out_arr[i],
                        lars_wrapper.g_arr[i],
                        p_buffer,
                        g_buffer,
                        numel,
                        lars_wrapper.repeat_arr[i],
                        rescale_grad,
                        0,
                        &param_norm,
                        &grad_norm);
    MomentumUpdate<T, MT>(lars_wrapper.p_out_arr[i],
                          lars_wrapper.g_arr[i],
                          lars_wrapper.v_out_arr[i],
                          lars_wrapper.p_out_arr[i],
                          lars_wrapper.v_out_arr[i],
                          lars_wrapper.master_p_out_arr[i],
                          lars_wrapper.master_p_out_arr[i],
                          lars_wrapper.lr_arr[i],
                          mu,
                          lars_wrapper.weight_decay_arr[i],
                          lars_coeff,
                          epsilon,
                          rescale_grad,
                          param_norm,
                          grad_norm,
                          tid,
                          grid_stride,
                          numel,
                          is_amp);
  }
}
#endif

template <typename T, typename MT>
__global__ void MomentumLarsKernel(const T* param,
                                   const T* __restrict__ grad,
                                   const MT* velocity,
                                   T* param_out,
                                   MT* velocity_out,
                                   const MT* master_param,
                                   MT* master_param_out,
                                   const MT* __restrict__ learning_rate,
                                   MT* __restrict__ p_buffer,
                                   MT* __restrict__ g_buffer,
                                   const MT mu,
                                   const MT lars_coeff,
                                   const MT lars_weight_decay,
                                   const MT epsilon,
                                   const MT rescale_grad,
                                   const int repeat_times,
                                   const int thresh,
                                   const int64_t numel,
                                   const bool is_amp) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int grid_stride = gridDim.x * LARS_BLOCK_SIZE;
#if CUDA_VERSION >= 11000
  const cooperative_groups::grid_group cg = cooperative_groups::this_grid();
  MT param_norm = static_cast<MT>(0);
  MT grad_norm = static_cast<MT>(0);
  L2NormKernel<T, MT>(&cg,
                      param,
                      grad,
                      p_buffer,
                      g_buffer,
                      numel,
                      repeat_times,
                      rescale_grad,
                      gridDim.x,
                      &param_norm,
                      &grad_norm);
#else
  const MT rescale_grad_pow = rescale_grad * rescale_grad;
  MT param_part_norm = threadIdx.x < thresh ? p_buffer[threadIdx.x] : 0;
  MT grad_part_norm = threadIdx.x < thresh ? g_buffer[threadIdx.x] : 0;
  __syncthreads();
  MT param_norm =
      Sqrt(phi::funcs::BlockReduceSum<MT>(param_part_norm, FINAL_MASK));
  MT grad_norm = Sqrt(rescale_grad_pow * phi::funcs::BlockReduceSum<MT>(
                                             grad_part_norm, FINAL_MASK));
#endif
  MomentumUpdate<T, MT>(param,
                        grad,
                        velocity,
                        param_out,
                        velocity_out,
                        master_param,
                        master_param_out,
                        learning_rate,
                        mu,
                        lars_weight_decay,
                        lars_coeff,
                        epsilon,
                        rescale_grad,
                        param_norm,
                        grad_norm,
                        tid,
                        grid_stride,
                        numel,
                        is_amp);
}

template <typename T, typename MT>
inline void SeparatedLarsMomentumOpCUDAKernel(const GPUContext& cuda_ctx,
                                              const T* param_data,
                                              T* param_out_data,
                                              const MT* velocity_data,
                                              MT* velocity_out_data,
                                              const T* grad_data,
                                              const MT* lr,
                                              MT* p_buffer,
                                              MT* g_buffer,
                                              const MT mu,
                                              const MT lars_coeff,
                                              const MT weight_decay,
                                              const MT epsilon,
                                              const MT rescale_grad,
                                              const int64_t numel,
                                              const MT* master_param_data,
                                              MT* master_out_data,
                                              const bool is_amp) {
  LarsThreadConfig<T> lars_thread_config(numel);
  L2NormKernel<T, MT><<<lars_thread_config.grid_for_norm,
                        LARS_BLOCK_SIZE,
                        0,
                        cuda_ctx.stream()>>>(param_data,
                                             grad_data,
                                             p_buffer,
                                             g_buffer,
                                             numel,
                                             lars_thread_config.repeat_times,
                                             rescale_grad);

  MomentumLarsKernel<T, MT>
      <<<lars_thread_config.grid_for_lars,
         LARS_BLOCK_SIZE,
         0,
         cuda_ctx.stream()>>>(param_data,
                              grad_data,
                              velocity_data,
                              param_out_data,
                              velocity_out_data,
                              master_param_data,
                              master_out_data,
                              lr,
                              p_buffer,
                              g_buffer,
                              mu,
                              lars_coeff,
                              weight_decay,
                              epsilon,
                              rescale_grad,
                              0,
                              lars_thread_config.grid_for_norm,
                              numel,
                              is_amp);
}

template <typename T, typename Context>
void LarsMomentumKernel(
    const Context& dev_ctx,
    const std::vector<const DenseTensor*>& param,
    const std::vector<const DenseTensor*>& velocity,
    const std::vector<const DenseTensor*>& learning_rate,
    const std::vector<const DenseTensor*>& grad,
    const paddle::optional<std::vector<const DenseTensor*>>& master_param,
    const std::vector<float>& weight_decay_arr,
    float mu,
    float lars_coeff,
    float epsilon,
    bool multi_precision,
    float rescale_grad,
    std::vector<DenseTensor*> param_out,
    std::vector<DenseTensor*> velocity_out,
    std::vector<DenseTensor*> master_param_out) {
  using MT = MultiPrecisionType<T>;
  int num_blocks_per_sm = 0;
  int sm_num = dev_ctx.GetSMCount();
  // phi::DenseTensor tmp_buffer_t = ctx.AllocateTmpTensor<MT, phi::GPUContext>(
  //     {LARS_BLOCK_SIZE << 1}, cuda_ctx);
  phi::DenseTensor tmp_buffer_t;
  tmp_buffer_t.Resize({LARS_BLOCK_SIZE << 1});
  MT* p_buffer = dev_ctx.template Alloc<MT>(&tmp_buffer_t);
  MT* g_buffer = p_buffer + LARS_BLOCK_SIZE;

  MT mu_ = static_cast<MT>(mu);
  MT lars_coeff_ = static_cast<MT>(lars_coeff);
  MT epsilon_ = static_cast<MT>(epsilon);
  MT rescale_grad_ = static_cast<MT>(rescale_grad);

  int op_num = grad.size();
#if CUDA_VERSION >= 11000
  if (op_num > 1) {
    LarsParamWrapper<T, MT> lars_wrapper;
    PADDLE_ENFORCE_LT(
        op_num,
        LARS_MAX_MERGED_OPS,
        errors::InvalidArgument(
            "The maximum number of merged-ops supported is (%d), but"
            "lars op required for training this model is (%d)\n",
            LARS_MAX_MERGED_OPS,
            op_num));

    /* Implementation of lars optimizer consists of following two steps:
      1. Figure out the L2 norm statistic result of grad data and param data.
      2. Update param and velocity with usage of L2 norm statistic result.
    Step1 and step2 can be merged with api provided by nvidia
      cudaLaunchCooperativeKernel:
      - The thread quantity shall less than physical SM limited threads
      - Launches as thread-block can synchronizlly execute. */
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &num_blocks_per_sm,
        MergedMomentumLarsKernel<T, MT>,
        LARS_BLOCK_SIZE,
        sizeof(MT) << 1);

    size_t total_numel = 0;
    for (int i = 0; i < op_num; ++i) {
      size_t temp_numel = param[i]->numel();
      total_numel += temp_numel;
      lars_wrapper.numel_arr[i] = temp_numel;
      lars_wrapper.g_arr[i] = grad[i]->data<T>();
      lars_wrapper.lr_arr[i] = learning_rate[i]->data<MT>();
      lars_wrapper.p_out_arr[i] = dev_ctx.template Alloc<T>(param_out[i]);
      lars_wrapper.v_out_arr[i] = dev_ctx.template Alloc<MT>(velocity_out[i]);
      lars_wrapper.weight_decay_arr[i] = static_cast<MT>(weight_decay_arr[i]);
      PADDLE_ENFORCE_EQ(
          param[i]->data<T>(),
          lars_wrapper.p_out_arr[i],
          errors::InvalidArgument(
              "Input(Param) and Output(ParamOut) must be the same Tensors."));
      PADDLE_ENFORCE_EQ(velocity[i]->data<MT>(),
                        lars_wrapper.v_out_arr[i],
                        errors::InvalidArgument(
                            "Input(Velocity) and Output(VelocityOut) must be "
                            "the same Tensors."));
    }
    int64_t avg_numel = total_numel / op_num;
    LarsThreadConfig<float> lars_thread_config(
        avg_numel, sm_num, num_blocks_per_sm);
    for (int i = 0; i < op_num; ++i) {
      lars_wrapper.repeat_arr[i] =
          lars_thread_config.GetRepeatTimes(lars_wrapper.numel_arr[i]);
    }
    if (multi_precision) {
      for (int i = 0; i < op_num; ++i) {
        lars_wrapper.master_p_out_arr[i] =
            dev_ctx.template Alloc<MT>(master_param_out[i]);
        PADDLE_ENFORCE_EQ(master_param.get()[i]->data<MT>(),
                          lars_wrapper.master_p_out_arr[i],
                          errors::InvalidArgument(
                              "Input(MasterParam) and Output(MasterParamOut) "
                              "must be the same Tensors."));
      }
    }
    void* cuda_param[] = {reinterpret_cast<void*>(&lars_wrapper),
                          reinterpret_cast<void*>(&p_buffer),
                          reinterpret_cast<void*>(&g_buffer),
                          reinterpret_cast<void*>(&op_num),
                          reinterpret_cast<void*>(&mu_),
                          reinterpret_cast<void*>(&lars_coeff_),
                          reinterpret_cast<void*>(&epsilon_),
                          reinterpret_cast<void*>(&rescale_grad_),
                          reinterpret_cast<void*>(&multi_precision)};
    // Launch all sm threads, and thead of each block synchronized cooperate.
    cudaLaunchCooperativeKernel(
        reinterpret_cast<void*>(MergedMomentumLarsKernel<T, MT>),
        lars_thread_config.grid_for_lars,
        LARS_BLOCK_SIZE,
        cuda_param,
        0,
        dev_ctx.stream());
  } else {
    auto* param_data = param[0]->data<T>();
    auto* grad_data = grad[0]->data<T>();
    auto* velocity_data = velocity[0]->data<MT>();
    auto* lr = learning_rate[0]->data<MT>();
    auto* param_out_data = dev_ctx.template Alloc<T>(param_out[0]);
    auto* velocity_out_data = dev_ctx.template Alloc<MT>(velocity_out[0]);
    const MT* master_param_data =
        multi_precision ? master_param.get()[0]->data<MT>() : nullptr;
    MT* master_param_out_data =
        multi_precision ? dev_ctx.template Alloc<MT>(master_param_out[0])
                        : nullptr;
    int64_t numel = param[0]->numel();
    MT lars_weight_decay = weight_decay_arr[0];

    // Figure out how many blocks can be active in each sm.
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm,
                                                  MomentumLarsKernel<T, MT>,
                                                  LARS_BLOCK_SIZE,
                                                  sizeof(MT) << 1);
    LarsThreadConfig<float> lars_thread_config(
        numel, sm_num, num_blocks_per_sm);
    int repeat_times = lars_thread_config.GetRepeatTimes(numel);
    int thresh = 0;
    void* cuda_param[] = {
        reinterpret_cast<void*>(&param_data),
        reinterpret_cast<void*>(&grad_data),
        reinterpret_cast<void*>(&velocity_data),
        reinterpret_cast<void*>(&param_out_data),
        reinterpret_cast<void*>(&velocity_out_data),
        reinterpret_cast<void*>(&master_param_data),
        reinterpret_cast<void*>(&master_param_out_data),
        reinterpret_cast<void*>(&lr),
        reinterpret_cast<void*>(&p_buffer),
        reinterpret_cast<void*>(&g_buffer),
        reinterpret_cast<void*>(&mu_),
        reinterpret_cast<void*>(&lars_coeff_),
        reinterpret_cast<void*>(&lars_weight_decay),
        reinterpret_cast<void*>(&epsilon_),
        reinterpret_cast<void*>(&rescale_grad_),
        reinterpret_cast<void*>(&repeat_times),
        reinterpret_cast<void*>(&thresh),  // Just a placeholder
        reinterpret_cast<void*>(&numel),
        reinterpret_cast<void*>(&multi_precision)};
    // Launch all sm threads.
    cudaLaunchCooperativeKernel(
        reinterpret_cast<void*>(MomentumLarsKernel<T, MT>),
        lars_thread_config.grid_for_lars,
        LARS_BLOCK_SIZE,
        cuda_param,
        0,
        dev_ctx.stream());
  }
#else
  for (int i = 0; i < op_num; ++i) {
    const MT* master_param_data =
        multi_precision ? master_param.get()[i]->data<MT>() : nullptr;
    MT* master_param_out_data =
        multi_precision ? dev_ctx.template Alloc<MT>(master_param_out[i])
                        : nullptr;
    SeparatedLarsMomentumOpCUDAKernel<T, MT>(
        dev_ctx,
        param[i]->data<T>(),
        dev_ctx.template Alloc<T>(param_out[i]),
        velocity[i]->data<MT>(),
        dev_ctx.template Alloc<MT>(velocity_out[i]),
        grad[i]->data<T>(),
        learning_rate[i]->data<MT>(),
        p_buffer,
        g_buffer,
        mu_,
        lars_coeff_,
        weight_decay_arr[i],
        epsilon_,
        rescale_grad_,
        param[i]->numel(),
        master_param_data,
        master_param_out_data,
        multi_precision);
  }
#endif
}
}  // namespace phi

PD_REGISTER_KERNEL(lars_momentum,
                   GPU,
                   ALL_LAYOUT,
                   phi::LarsMomentumKernel,
                   float,
                   double,
                   phi::dtype::float16) {
  if (kernel_key.dtype() == phi::DataType::FLOAT16) {
    kernel->OutputAt(1).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(2).SetDataType(phi::DataType::FLOAT32);
  }
}
