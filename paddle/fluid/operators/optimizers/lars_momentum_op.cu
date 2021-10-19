/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/amp/fp16_type_traits.h"
#include "paddle/fluid/operators/math/math_cuda_utils.h"
#include "paddle/fluid/operators/optimizers/lars_momentum_op.h"
#include "paddle/fluid/platform/fast_divmod.h"

#if CUDA_VERSION >= 11000
#include <cooperative_groups.h>
#endif

#ifdef __HIPCC__
#define LARS_BLOCK_SIZE 256
#else
#define LARS_BLOCK_SIZE 512
#endif

namespace paddle {
namespace operators {

template <typename T>
using MultiPrecisionType = typename details::MPTypeTrait<T>::Type;

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
  int grid_for_lars;
#if CUDA_VERSION >= 11000
 public:
  explicit LarsThreadConfig(int64_t numel, int sm_num, int num_blocks_per_sm) {
    int grid = (numel + LARS_BLOCK_SIZE - 1) / LARS_BLOCK_SIZE;
    grid_for_lars =
        std::min(std::min(sm_num * num_blocks_per_sm, grid), LARS_BLOCK_SIZE);
  }

#else
  int grid_for_norm;
  explicit LarsThreadConfig(const int64_t numel) {
    int grid = (numel + LARS_BLOCK_SIZE - 1) / LARS_BLOCK_SIZE;
    grid_for_norm = std::min(grid, LARS_BLOCK_SIZE);
    // Determine to read 4 fp16 or float data once, but 2 double data once.
    grid_for_lars =
        std::is_same<double, T>::value
            ? (numel + (LARS_BLOCK_SIZE << 1) - 1) / (LARS_BLOCK_SIZE << 1)
            : (numel + (LARS_BLOCK_SIZE << 2) - 1) / (LARS_BLOCK_SIZE << 2);
  }
#endif
};

template <typename T, typename MT, int VecSize, bool IsAmp = false>
__device__ inline void VectorizeLarsUpdate(
    const T* __restrict__ grad, const MT* param, const MT* velocity,
    T* param_out, MT* velocity_out, const MT mu, MT local_lr,
    const MT lars_weight_decay, const MT rescale_grad, const int tid,
    const int grid_stride, const int numel, MT* master_param_out = nullptr) {
  using VecType = paddle::platform::AlignedVector<T, VecSize>;
  using VecMType = paddle::platform::AlignedVector<MT, VecSize>;
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
          Fma(velocity_data[j], mu,
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
    MT velocity_tmp = Fma(velocity[i], mu, local_lr * Fma(lars_weight_decay,
                                                          param_val, grad_val));
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
    const T* p_data, const T* __restrict__ g_data,
    MT* __restrict__ buffer_for_param_norm,
    MT* __restrict__ buffer_for_grad_norm, const int64_t numel,
    const MT rescale_grad, MT* __restrict__ p_n = nullptr,
    MT* __restrict__ g_n = nullptr) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  int grid_stride = blockDim.x * gridDim.x;

  MT p_tmp = static_cast<MT>(0);
  MT g_tmp = static_cast<MT>(0);
  while (tid < numel) {
    MT tmp0 = static_cast<MT>(p_data[tid]);
    MT tmp1 = static_cast<MT>(g_data[tid]);
    p_tmp += (tmp0 * tmp0);
    g_tmp += (tmp1 * tmp1);
    tid += grid_stride;
  }
  p_tmp = math::blockReduceSum<MT>(p_tmp, FINAL_MASK);
  g_tmp = math::blockReduceSum<MT>(g_tmp, FINAL_MASK);

  if (threadIdx.x == 0) {
    buffer_for_param_norm[blockIdx.x] = p_tmp;
    buffer_for_grad_norm[blockIdx.x] = g_tmp;
  }
#if CUDA_VERSION >= 11000
  __shared__ MT s_buffer[2];
  cg->sync();  // Grid sync for writring partial result to gloabl memory
  MT p_part_sum =
      threadIdx.x < gridDim.x ? buffer_for_param_norm[threadIdx.x] : 0;
  MT g_part_sum =
      threadIdx.x < gridDim.x ? buffer_for_grad_norm[threadIdx.x] : 0;
  MT tmp0 = math::blockReduceSum<MT>(p_part_sum, FINAL_MASK);
  MT tmp1 = math::blockReduceSum<MT>(g_part_sum, FINAL_MASK);
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
    const T* param, const T* __restrict__ grad, const MT* velocity,
    T* param_out, MT* velocity_out, const MT* master_param,
    MT* master_param_out, const MT* __restrict__ learning_rate, const MT mu,
    const MT lars_weight_decay, const MT lars_coeff, const MT epsilon,
    const MT rescale_grad, const MT param_norm, const MT grad_norm,
    const int tid, const int grid_stride, const int64_t numel,
    const bool is_amp) {
  const MT lr = learning_rate[0];
  MT local_lr = lr;
  if (lars_weight_decay > static_cast<MT>(0)) {
    local_lr = lr * lars_coeff * param_norm /
               (fma(lars_weight_decay, param_norm, grad_norm) + epsilon);
  }
  if (is_amp) {
    VectorizeLarsUpdate<T, MT, /*VecSize=*/4, /*IsAmp=*/true>(
        grad, master_param, velocity, param_out, velocity_out, mu, local_lr,
        lars_weight_decay, rescale_grad, tid, grid_stride, numel,
        master_param_out);
  } else {
    if (std::is_same<T, float>::value ||
        std::is_same<T, paddle::platform::float16>::value) {
      /* TODO(limingshu): pointer cast may damage memory accessing for fp16 */
      VectorizeLarsUpdate<T, MT, /*VecSize=*/4, /*IsAmp=*/false>(
          grad, reinterpret_cast<const MT*>(param), velocity, param_out,
          velocity_out, mu, local_lr, lars_weight_decay, rescale_grad, tid,
          grid_stride, numel);
    } else {
      VectorizeLarsUpdate<T, MT, /*VecSize=*/2, /*IsAmp=*/false>(
          grad, reinterpret_cast<const MT*>(param), velocity, param_out,
          velocity_out, mu, local_lr, lars_weight_decay, rescale_grad, tid,
          grid_stride, numel);
    }
  }
}

#if CUDA_VERSION >= 11000
template <typename MT, int OpNum, typename T>
struct MasterParamHelper {
  DEVICE inline MT* GetMasterParam(size_t) const { return nullptr; }
  constexpr void SetMasterParam(size_t, MT*) {}
};

template <typename MT, int OpNum>
struct MasterParamHelper<MT, OpNum, paddle::platform::float16> {
  MT* master_params[OpNum];

  DEVICE inline MT* GetMasterParam(size_t idx) const {
    return master_params[idx];
  }
  void SetMasterParam(size_t idx, MT* p) { master_params[idx] = p; }
};

template <typename T, typename MT,
          int OpNum =
              std::is_same<T, paddle::platform::float16>::value ? 80 : 90>
struct LarsParamWarpper : public MasterParamHelper<MT, OpNum, T> {
  static constexpr int kNum = OpNum;

  int numel_arr[OpNum];
  const MT* __restrict__ lr_arr[OpNum];
  const T* __restrict__ g_arr[OpNum];
  T* p_arr[OpNum];
  MT* v_arr[OpNum];
  MT weight_decay[OpNum];
};

template <typename T, typename MT>
__global__ void MergedMomentumLarsKernel(LarsParamWarpper<T, MT> lars_warpper,
                                         MT* __restrict__ buffer_for_param_norm,
                                         MT* __restrict__ buffer_for_grad_norm,
                                         const int op_num, const MT mu,
                                         const MT lars_coeff, const MT epsilon,
                                         const MT rescale_grad,
                                         const bool is_amp) {
  int grid_stride = gridDim.x * blockDim.x;
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  const cooperative_groups::grid_group cg = cooperative_groups::this_grid();
  for (int i = 0; i < op_num; ++i) {
    int numel = lars_warpper.numel_arr[i];
    MT param_norm = static_cast<MT>(0);
    MT grad_norm = static_cast<MT>(0);
    L2NormKernel<T, MT>(&cg, lars_warpper.p_arr[i], lars_warpper.g_arr[i],
                        buffer_for_param_norm, buffer_for_grad_norm, numel,
                        rescale_grad, &param_norm, &grad_norm);
    MomentumUpdate<T, MT>(
        lars_warpper.p_arr[i], lars_warpper.g_arr[i], lars_warpper.v_arr[i],
        lars_warpper.p_arr[i], lars_warpper.v_arr[i],
        lars_warpper.GetMasterParam(i), lars_warpper.GetMasterParam(i),
        lars_warpper.lr_arr[i], mu, lars_warpper.weight_decay[i], lars_coeff,
        epsilon, rescale_grad, param_norm, grad_norm, tid, grid_stride, numel,
        is_amp);
  }
}
#endif

template <typename T, typename MT>
__global__ void MomentumLarsKernel(
    const T* param, const T* __restrict__ grad, const MT* velocity,
    T* param_out, MT* velocity_out, const MT* master_param,
    MT* master_param_out, const MT* __restrict__ learning_rate,
    MT* __restrict__ buffer_for_param_norm,
    MT* __restrict__ buffer_for_grad_norm, const MT mu, const MT lars_coeff,
    const MT lars_weight_decay, const MT epsilon, const MT rescale_grad,
    const int thresh, const int64_t numel, const bool is_amp) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int grid_stride = gridDim.x * blockDim.x;
#if CUDA_VERSION >= 11000
  const cooperative_groups::grid_group cg = cooperative_groups::this_grid();
  MT param_norm = static_cast<MT>(0);
  MT grad_norm = static_cast<MT>(0);
  L2NormKernel<T, MT>(&cg, param, grad, buffer_for_param_norm,
                      buffer_for_grad_norm, numel, rescale_grad, &param_norm,
                      &grad_norm);
#else
  __shared__ MT s_buffer[2];
  MT p_part_sum = threadIdx.x < thresh ? buffer_for_param_norm[threadIdx.x] : 0;
  MT g_part_sum = threadIdx.x < thresh ? buffer_for_grad_norm[threadIdx.x] : 0;
  MT tmp0 = math::blockReduceSum<MT>(p_part_sum, FINAL_MASK);
  MT tmp1 = math::blockReduceSum<MT>(g_part_sum, FINAL_MASK);
  if (threadIdx.x == 0) {
    s_buffer[0] = tmp0;
    s_buffer[1] = tmp1;
  }
  __syncthreads();
  MT param_norm = Sqrt(s_buffer[0]);
  MT grad_norm = rescale_grad * Sqrt(s_buffer[1]);
#endif
  MomentumUpdate<T, MT>(param, grad, velocity, param_out, velocity_out,
                        master_param, master_param_out, learning_rate, mu,
                        lars_weight_decay, lars_coeff, epsilon, rescale_grad,
                        param_norm, grad_norm, tid, grid_stride, numel, is_amp);
}

template <typename T, typename MT>
inline void SeparatedLarsMomentumOpCUDAKernel(
    const platform::CUDADeviceContext& cuda_ctx, const T* param_data,
    T* param_out_data, const MT* velocity_data, MT* velocity_out_data,
    const T* grad_data, const MT* lr, MT* buffer_for_param_norm,
    MT* buffer_for_grad_norm, const MT mu, const MT lars_coeff,
    const MT weight_decay, const MT epsilon, const MT rescale_grad,
    const int64_t numel, const MT* master_param_data, MT* master_out_data,
    const bool is_amp) {
  LarsThreadConfig<T> lars_thread_config(numel);
  L2NormKernel<T, MT><<<lars_thread_config.grid_for_norm, LARS_BLOCK_SIZE, 0,
                        cuda_ctx.stream()>>>(
      param_data, grad_data, buffer_for_param_norm, buffer_for_grad_norm, numel,
      rescale_grad);

  MomentumLarsKernel<T, MT><<<lars_thread_config.grid_for_lars, LARS_BLOCK_SIZE,
                              0, cuda_ctx.stream()>>>(
      param_data, grad_data, velocity_data, param_out_data, velocity_out_data,
      master_param_data, master_out_data, lr, buffer_for_param_norm,
      buffer_for_grad_norm, mu, lars_coeff, weight_decay, epsilon, rescale_grad,
      lars_thread_config.grid_for_norm, numel, is_amp);
}

template <typename DeviceContext, typename T>
class LarsMomentumOpCUDAKernel : public framework::OpKernel<T> {
  using MT = MultiPrecisionType<T>;

 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    int num_blocks_per_sm = 0;
    bool multi_precision = ctx.Attr<bool>("multi_precision");
    auto& cuda_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    int sm_num = cuda_ctx.GetSMCount();
    framework::Tensor tmbuffer_for_param_norm_t =
        ctx.AllocateTmpTensor<MT, platform::CUDADeviceContext>(
            {LARS_BLOCK_SIZE << 1}, cuda_ctx);
    auto* buffer_for_param_norm =
        tmbuffer_for_param_norm_t.mutable_data<MT>(ctx.GetPlace());
    auto* buffer_for_grad_norm = buffer_for_param_norm + LARS_BLOCK_SIZE;

    MT mu = static_cast<MT>(ctx.Attr<float>("mu"));
    MT lars_coeff = static_cast<MT>(ctx.Attr<float>("lars_coeff"));
    MT epsilon = static_cast<MT>(ctx.Attr<float>("epsilon"));
    MT rescale_grad = static_cast<MT>(ctx.Attr<float>("rescale_grad"));
    auto weight_decay_arr = ctx.Attr<std::vector<float>>("lars_weight_decay");

    auto grad = ctx.MultiInput<framework::LoDTensor>("Grad");
    auto param = ctx.MultiInput<framework::LoDTensor>("Param");
    auto velocity = ctx.MultiInput<framework::LoDTensor>("Velocity");
    auto param_out = ctx.MultiOutput<framework::LoDTensor>("ParamOut");
    auto velocity_out = ctx.MultiOutput<framework::LoDTensor>("VelocityOut");
    auto learning_rate = ctx.MultiInput<framework::LoDTensor>("LearningRate");
    auto master_param = ctx.MultiInput<framework::LoDTensor>("MasterParam");
    auto master_param_out =
        ctx.MultiOutput<framework::LoDTensor>("MasterParamOut");

    int op_num = grad.size();
#if CUDA_VERSION >= 11000
    if (op_num > 1) {
      LarsParamWarpper<T, MT> lars_warpper;
      VLOG(10) << "Num of ops merged in lars_warpper is " << lars_warpper.kNum;
      /* Implementation of lars optimizer consists of following two steps:
        1. Figure out the L2 norm statistic result of grad data and param data.
        2. Update param and velocity with usage of L2 norm statistic result.
      Step1 and step2 can be merged with api provided by nvida
        cudaLaunchCooperativeKernel:
        - The thread quantity shall less than pyhsical SM limited threads
        - Launche as thread-block can synchronizlly execute. */
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &num_blocks_per_sm, MergedMomentumLarsKernel<T, MT>, LARS_BLOCK_SIZE,
          sizeof(MT) << 1);

      int loop = (op_num + lars_warpper.kNum - 1) / lars_warpper.kNum;
      for (int j = 0; j < loop; ++j) {
        size_t total_numel = 0;
        int start_idx = j * lars_warpper.kNum;
        int warpper_num = std::min(lars_warpper.kNum, op_num - start_idx);

        for (int i = 0; i < warpper_num; ++i) {
          size_t temp_numel = param[start_idx + i]->numel();
          total_numel += temp_numel;
          lars_warpper.numel_arr[i] = temp_numel;
          lars_warpper.weight_decay[i] = static_cast<MT>(weight_decay_arr[i]);
          lars_warpper.g_arr[i] = grad[start_idx + i]->data<T>();
          lars_warpper.p_arr[i] =
              param_out[start_idx + i]->mutable_data<T>(ctx.GetPlace());
          lars_warpper.v_arr[i] =
              velocity_out[start_idx + i]->mutable_data<MT>(ctx.GetPlace());
          lars_warpper.lr_arr[i] = learning_rate[start_idx + i]->data<MT>();
          if (multi_precision) {
            auto master_param_data =
                master_param_out[start_idx + i]->mutable_data<MT>(
                    ctx.GetPlace());
            lars_warpper.SetMasterParam(i, master_param_data);
            PADDLE_ENFORCE_EQ(
                master_param[start_idx + i]->data<MT>(), master_param_data,
                platform::errors::InvalidArgument(
                    "Since Input(MasterParam) and Output(MasterParamOut) of "
                    "lars optimizer must be the same Tensors."));
          }
          PADDLE_ENFORCE_EQ(
              param[start_idx + i]->data<T>(), lars_warpper.p_arr[i],
              platform::errors::InvalidArgument(
                  "Since Input(Param) and Output(ParamOut) of lars optimizer "
                  "must be the same Tensors."));
          PADDLE_ENFORCE_EQ(
              velocity[start_idx + i]->data<MT>(), lars_warpper.v_arr[i],
              platform::errors::InvalidArgument(
                  "Since Input(Velocity) and Output(VelocityOut) of "
                  "lars optimizer must be "
                  "the same Tensors."));
        }
        VLOG(10) << "Ops warpped in this loop " << j << " is : " << warpper_num;
        int64_t avg_numel = total_numel / warpper_num;
        LarsThreadConfig<float> lars_thread_config(avg_numel, sm_num,
                                                   num_blocks_per_sm);
        void* cuda_param[] = {reinterpret_cast<void*>(&lars_warpper),
                              reinterpret_cast<void*>(&buffer_for_param_norm),
                              reinterpret_cast<void*>(&buffer_for_grad_norm),
                              reinterpret_cast<void*>(&warpper_num),
                              reinterpret_cast<void*>(&mu),
                              reinterpret_cast<void*>(&lars_coeff),
                              reinterpret_cast<void*>(&epsilon),
                              reinterpret_cast<void*>(&rescale_grad),
                              reinterpret_cast<void*>(&multi_precision)};
        // Lanuch all sm theads,thead of each block synchronizedly cooperate.
        cudaLaunchCooperativeKernel(
            reinterpret_cast<void*>(MergedMomentumLarsKernel<T, MT>),
            lars_thread_config.grid_for_lars, LARS_BLOCK_SIZE, cuda_param, 0,
            cuda_ctx.stream());
      }
    } else {
      auto* param_data = param[0]->data<T>();
      auto* grad_data = grad[0]->data<T>();
      auto* velocity_data = velocity[0]->data<MT>();
      auto* lr = learning_rate[0]->data<MT>();
      auto* param_out_data = param_out[0]->mutable_data<T>(ctx.GetPlace());
      auto* velocity_out_data =
          velocity_out[0]->mutable_data<MT>(ctx.GetPlace());
      const MT* master_param_data =
          multi_precision ? master_param[0]->data<MT>() : nullptr;
      MT* master_param_out_data =
          multi_precision
              ? master_param_out[0]->mutable_data<MT>(ctx.GetPlace())
              : nullptr;
      int64_t numel = param[0]->numel();
      MT lars_weight_decay = static_cast<MT>(weight_decay_arr[0]);

      // Figure out how many blocks can be active in each sm.
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &num_blocks_per_sm, MomentumLarsKernel<T, MT>, LARS_BLOCK_SIZE,
          sizeof(MT) << 1);
      LarsThreadConfig<float> lars_thread_config(numel, sm_num,
                                                 num_blocks_per_sm);
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
          reinterpret_cast<void*>(&buffer_for_param_norm),
          reinterpret_cast<void*>(&buffer_for_grad_norm),
          reinterpret_cast<void*>(&mu),
          reinterpret_cast<void*>(&lars_coeff),
          reinterpret_cast<void*>(&lars_weight_decay),
          reinterpret_cast<void*>(&epsilon),
          reinterpret_cast<void*>(&rescale_grad),
          reinterpret_cast<void*>(&thresh),  // Just a placeholder
          reinterpret_cast<void*>(&numel),
          reinterpret_cast<void*>(&multi_precision)};
      // Lanuch all sm theads.
      cudaLaunchCooperativeKernel(
          reinterpret_cast<void*>(MomentumLarsKernel<T, MT>),
          lars_thread_config.grid_for_lars, LARS_BLOCK_SIZE, cuda_param, 0,
          cuda_ctx.stream());
    }
#else
    for (int i = 0; i < op_num; ++i) {
      const MT* master_param_data =
          multi_precision ? master_param[i]->data<MT>() : nullptr;
      MT* master_param_out_data =
          multi_precision
              ? master_param_out[i]->mutable_data<MT>(ctx.GetPlace())
              : nullptr;
      SeparatedLarsMomentumOpCUDAKernel<T, MT>(
          cuda_ctx, param[i]->data<T>(),
          param_out[i]->mutable_data<T>(ctx.GetPlace()),
          velocity[i]->data<MT>(),
          velocity_out[i]->mutable_data<MT>(ctx.GetPlace()), grad[i]->data<T>(),
          learning_rate[i]->data<MT>(), buffer_for_param_norm,
          buffer_for_grad_norm, mu, lars_coeff,
          static_cast<MT>(weight_decay_arr[i]), epsilon, rescale_grad,
          param[i]->numel(), master_param_data, master_param_out_data,
          multi_precision);
    }
#endif
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    lars_momentum,
    ops::LarsMomentumOpCUDAKernel<paddle::platform::CUDADeviceContext, float>,
    ops::LarsMomentumOpCUDAKernel<paddle::platform::CUDADeviceContext, double>,
    ops::LarsMomentumOpCUDAKernel<paddle::platform::CUDADeviceContext,
                                  paddle::platform::float16>);
