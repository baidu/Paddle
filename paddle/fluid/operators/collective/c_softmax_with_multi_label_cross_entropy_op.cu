/* Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/collective/c_softmax_with_multi_label_cross_entropy_op.h"
#include "paddle/phi/core/distributed/comm_context_manager.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"

#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/device/gpu/nccl_helper.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/kernels/funcs/axis_utils.h"
#include "paddle/phi/kernels/funcs/cross_entropy.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/math.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/softmax_impl.h"
#include "paddle/utils/string/string_helper.h"

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/common/flags.h"
#include "paddle/phi/core/distributed/nccl_comm_context.h"
COMMON_DECLARE_bool(dynamic_static_unified_comm);
#endif

namespace paddle {
namespace operators {

static constexpr int kNumCUDAThreads = 512;
static constexpr int64_t kNumMaxinumNumBlocks = 4096;

static inline int64_t NumBlocks(const int64_t N) {
  return std::min((N + kNumCUDAThreads - 1) / kNumCUDAThreads,
                  kNumMaxinumNumBlocks);
}

template <typename T, typename IndexT>
__global__ void SoftMaskLabelByIndex(T* predicted_logits,
                                     const T* logit,
                                     const IndexT* label,
                                     const IndexT ignore_index,
                                     const int64_t start_index,
                                     const int64_t end_index,
                                     const int64_t N,
                                     const int64_t D,
                                     const int64_t C,
                                     const int nranks) {
  CUDA_KERNEL_LOOP_TYPE(i, N, int64_t) {
    for (int j = 0; j < C; ++j) {
      auto real_label = label[i * C + j];
      PADDLE_ENFORCE(((real_label < D * nranks) && (real_label >= 0)) ||
                         (real_label == ignore_index),
                     "The index is out of bounds, "
                     "please check whether the value of label and "
                     "input meet the class number. It should "
                     "be less than [%ld] or equal to [%ld], but received [%ld]",
                     static_cast<int64_t>(D * nranks),
                     static_cast<int64_t>(ignore_index),
                     static_cast<int64_t>(real_label));

      if (real_label >= start_index && real_label < end_index) {
        predicted_logits[i * C + j] = logit[i * D + real_label - start_index];
      }
    }
  }
}

template <typename T, typename IndexT>
__global__ void CaculateSoftLoss(T* loss,
                                 const T* predict_logits,
                                 const T* sum_exp_logits,
                                 const IndexT* label,
                                 const T* smooth_weight,
                                 const int64_t ignore_index,
                                 const int64_t N,
                                 const int64_t C,
                                 const bool sum_multi_label_loss) {
  CUDA_KERNEL_LOOP_TYPE(i, N, int64_t) {
    T tmp_loss = static_cast<T>(0);
    loss[i] = static_cast<T>(0);
    T log_sum_exp_logits = phi::funcs::TolerableValue<T>()(
        phi::funcs::real_log(sum_exp_logits[i]));
    for (int j = 0; j < C; ++j) {
      int64_t label_idx = i * C + j;
      auto real_label = static_cast<int64_t>(label[label_idx]);
      auto prob = static_cast<T>(smooth_weight[label_idx]);
      tmp_loss =
          ignore_index == real_label
              ? static_cast<T>(0)
              : phi::funcs::TolerableValue<T>()(
                    (log_sum_exp_logits - predict_logits[label_idx]) * prob);
      if (sum_multi_label_loss) {
        loss[i] += tmp_loss;
      } else {
        loss[label_idx] = tmp_loss;
      }
    }
  }
}

template <typename T, typename IndexT>
__global__ void CaculateSoftLogitsGrad(T* logits_grad,
                                       const IndexT* labels,
                                       const T* smooth_weight,
                                       const IndexT ignore_index,
                                       const int64_t start_index,
                                       const int64_t end_index,
                                       const int64_t N,
                                       const int64_t D,
                                       const int64_t C) {
  CUDA_KERNEL_LOOP_TYPE(i, N, int64_t) {
    for (int j = 0; j < C; ++j) {
      auto real_label = labels[i * C + j];
      auto prob = smooth_weight[i * C + j];
      if (real_label >= start_index && real_label < end_index) {
        int64_t idx = i * D + real_label - start_index;
        logits_grad[idx] = real_label == ignore_index
                               ? static_cast<T>(0)
                               : (logits_grad[idx] - prob);
      }
    }
  }
}

template <typename T, typename IndexT>
__global__ void SoftMaskLabelByIndexGrad(T* logits_grad,
                                         const T* loss_grad,
                                         const int64_t N,
                                         const int64_t D,
                                         const int64_t C,
                                         const bool sum_multi_label_loss) {
  CUDA_KERNEL_LOOP_TYPE(i, N * D, int64_t) {
    auto loss_grad_idx = sum_multi_label_loss ? (i / D) : (i / D * C);
    logits_grad[i] *= loss_grad[loss_grad_idx];
  }
}

template <typename T, typename DeviceContext>
class CSoftmaxWithMultiLabelCrossEntropyOpCUDAKernel
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const int rid = ctx.Attr<int>("ring_id");
    auto map = distributed::ProcessGroupMapFromGid::getInstance();
    if (map->has(rid)) {
      CSoftmaxWithMultiLableCrossEntropyProcessGroupFunctor<phi::GPUContext, T>
          functor_;
      functor_(ctx);
    } else {
      CSoftmaxWithMultiLabelCrossEntropyFunctor<phi::GPUContext, T> functor_;
      functor_(ctx);
    }
  }
};

template <typename T>
struct CSoftmaxWithMultiLabelCrossEntropyFunctor<phi::GPUContext, T> {
  void operator()(const framework::ExecutionContext& ctx) {
    const phi::DenseTensor* logits = ctx.Input<phi::DenseTensor>("Logits");
    const phi::DenseTensor* labels = ctx.Input<phi::DenseTensor>("Label");
    const phi::DenseTensor* smooth_weight =
        ctx.Input<phi::DenseTensor>("SmoothWeight");
    phi::DenseTensor* softmax = ctx.Output<phi::DenseTensor>("Softmax");
    phi::DenseTensor* loss = ctx.Output<phi::DenseTensor>("Loss");

    const int64_t ignore_index = ctx.Attr<int64_t>("ignore_index");
    const bool sum_multi_label_loss = ctx.Attr<bool>("sum_multi_label_loss");
    const int rid = ctx.Attr<int>("ring_id");
    const int nranks = ctx.Attr<int>("nranks");
    const int rank = ctx.Attr<int>("rank");

    const auto& place = ctx.GetPlace();
    auto& dev_ctx = ctx.template device_context<phi::GPUContext>();

    gpuStream_t stream = nullptr;
    platform::NCCLComm* comm = nullptr;
    phi::distributed::NCCLCommContext* comm_ctx = nullptr;

    const auto& comm_context_manager =
        phi::distributed::CommContextManager::GetInstance();

    if (FLAGS_dynamic_static_unified_comm) {
      PADDLE_ENFORCE_EQ(comm_context_manager.Has(std::to_string(rid)),
                        true,
                        common::errors::InvalidArgument(
                            "You choose to use new communication library by "
                            "setting environment "
                            "variable FLAGS_dynamic_static_unified_comm True. "
                            "But ring_id(%d) is "
                            "not found in comm_context_manager.",
                            std::to_string(rid)));
      comm_ctx = static_cast<phi::distributed::NCCLCommContext*>(
          comm_context_manager.Get(std::to_string(rid)));
      PADDLE_ENFORCE_NE(comm_ctx,
                        nullptr,
                        common::errors::Unavailable(
                            "NCCLCommContext is nullptr, collective op should "
                            "has ring_id attr."));

      stream = dev_ctx.stream();
      VLOG(3) << "new comm_context_manager has ring_id " << rid;
    } else {  // old comm_context
      comm = platform::NCCLCommContext::Instance().Get(rid, place);

      stream = dev_ctx.stream();
      VLOG(3) << "old NCCLCommContext has ring_id " << rid;
    }

    // allocate memory on device.
    softmax->mutable_data<T>(place);
    loss->mutable_data<T>(place);

    const auto& logits_dims = logits->dims();
    const auto& labels_dims = labels->dims();

    const int axis = logits_dims.size() - 1;
    const int64_t N = phi::funcs::SizeToAxis<int64_t>(axis, logits_dims);
    const int64_t D = phi::funcs::SizeFromAxis<int64_t>(axis, logits_dims);
    const int64_t C = phi::funcs::SizeFromAxis<int64_t>(axis, labels_dims);

    phi::DenseTensor logits_2d, softmax_2d, loss_2d;
    logits_2d.ShareDataWith(*logits).Resize({N, D});
    softmax_2d.ShareDataWith(*softmax).Resize({N, D});
    int64_t loss_last_dim = sum_multi_label_loss ? 1 : C;
    loss_2d.ShareDataWith(*loss).Resize({N, loss_last_dim});

    auto eigen_logits = phi::funcs::EigenMatrix<T>::From(logits_2d);
    auto eigen_softmax = phi::funcs::EigenMatrix<T>::From(softmax_2d);

    // step 1, obtain logit_max
    phi::DenseTensor logits_max;
    logits_max = ctx.AllocateTmpTensor<T, phi::GPUContext>({N, 1}, dev_ctx);

    auto eigen_logits_max = phi::funcs::EigenMatrix<T>::From(logits_max);
    Eigen::DSizes<int, 1> along_axis(1);
    eigen_logits_max.device(*dev_ctx.eigen_device()) =
        eigen_logits.maximum(along_axis);

    if (comm_ctx) {
      comm_ctx->AllReduce(&logits_max, logits_max, ncclMax, stream);
    } else {
      void* logits_max_buff = logits_max.mutable_data<T>(place);

      PADDLE_ENFORCE_GPU_SUCCESS(
          phi::dynload::ncclAllReduce(logits_max_buff,
                                      logits_max_buff,
                                      logits_max.numel(),
                                      phi::ToNCCLDataType(logits_max.dtype()),
                                      ncclMax,
                                      comm->comm(),
                                      stream));
    }

    // step 2, obtain logit - logit_max
    Eigen::DSizes<int, 2> batch_by_one(N, 1);
    Eigen::DSizes<int, 2> one_by_class(1, D);

    eigen_softmax.device(*dev_ctx.eigen_device()) =
        (eigen_logits -
         eigen_logits_max.reshape(batch_by_one).broadcast(one_by_class));

    // step 3, obtain predict target
    phi::DenseTensor predicted_logits;
    predicted_logits =
        ctx.AllocateTmpTensor<T, phi::GPUContext>({N, C}, dev_ctx);
    predicted_logits.mutable_data<T>(place);

    auto t = phi::EigenVector<T>::Flatten(predicted_logits);
    t.device(*dev_ctx.eigen_device()) = t.constant(static_cast<T>(0));

    const int64_t start_index = rank * D;
    const int64_t end_index = start_index + D;

    int64_t blocks = NumBlocks(N);
    int threads = kNumCUDAThreads;
    const auto& label_type = framework::TransToProtoVarType(labels->dtype());

    if (label_type == framework::proto::VarType::INT32) {
      SoftMaskLabelByIndex<T, int32_t>
          <<<blocks, threads, 0, dev_ctx.stream()>>>(
              predicted_logits.data<T>(),
              softmax_2d.data<T>(),
              labels->data<int32_t>(),
              static_cast<int32_t>(ignore_index),
              start_index,
              end_index,
              N,
              D,
              C,
              nranks);
    } else if (label_type == framework::proto::VarType::INT64) {
      SoftMaskLabelByIndex<T, int64_t>
          <<<blocks, threads, 0, dev_ctx.stream()>>>(predicted_logits.data<T>(),
                                                     softmax_2d.data<T>(),
                                                     labels->data<int64_t>(),
                                                     ignore_index,
                                                     start_index,
                                                     end_index,
                                                     N,
                                                     D,
                                                     C,
                                                     nranks);
    }

    predicted_logits.mutable_data<T>(place);
    if (comm_ctx) {
      comm_ctx->AllReduce(&predicted_logits, predicted_logits, ncclSum, stream);
    } else {
      void* predict_logits_buff = predicted_logits.data();
      PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::ncclAllReduce(
          predict_logits_buff,
          predict_logits_buff,
          predicted_logits.numel(),
          phi::ToNCCLDataType(predicted_logits.dtype()),
          ncclSum,
          comm->comm(),
          stream));
    }

    // step 4, obtain exp(logit)
    eigen_softmax.device(*dev_ctx.eigen_device()) = eigen_softmax.exp();

    // step 5, obtain sum_exp_logits
    phi::DenseTensor sum_exp_logits;
    sum_exp_logits = ctx.AllocateTmpTensor<T, phi::GPUContext>({N, 1}, dev_ctx);
    sum_exp_logits.mutable_data<T>(place);

    phi::SumKernel<T, phi::GPUContext>(
        dev_ctx, softmax_2d, {-1}, softmax_2d.dtype(), true, &sum_exp_logits);

    if (comm_ctx) {
      comm_ctx->AllReduce(&sum_exp_logits, sum_exp_logits, ncclSum, stream);
    } else {
      void* sum_exp_logits_buff = sum_exp_logits.data();
      PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::ncclAllReduce(
          sum_exp_logits_buff,
          sum_exp_logits_buff,
          sum_exp_logits.numel(),
          phi::ToNCCLDataType(sum_exp_logits.dtype()),
          ncclSum,
          comm->comm(),
          stream));
    }

    if (label_type == framework::proto::VarType::INT32) {
      CaculateSoftLoss<T, int32_t>
          <<<blocks, threads, 0, dev_ctx.stream()>>>(loss_2d.data<T>(),
                                                     predicted_logits.data<T>(),
                                                     sum_exp_logits.data<T>(),
                                                     labels->data<int32_t>(),
                                                     smooth_weight->data<T>(),
                                                     ignore_index,
                                                     N,
                                                     C,
                                                     sum_multi_label_loss);

    } else {
      CaculateSoftLoss<T, int64_t>
          <<<blocks, threads, 0, dev_ctx.stream()>>>(loss_2d.data<T>(),
                                                     predicted_logits.data<T>(),
                                                     sum_exp_logits.data<T>(),
                                                     labels->data<int64_t>(),
                                                     smooth_weight->data<T>(),
                                                     ignore_index,
                                                     N,
                                                     C,
                                                     sum_multi_label_loss);
    }

    auto eigen_sum_exp_logits =
        phi::funcs::EigenMatrix<T>::From(sum_exp_logits);
    eigen_softmax.device(*dev_ctx.eigen_device()) =
        (eigen_softmax *
         eigen_sum_exp_logits.inverse().broadcast(one_by_class));
  }
};

template <typename T>
struct CSoftmaxWithMultiLableCrossEntropyProcessGroupFunctor<phi::GPUContext,
                                                             T> {
  void operator()(const framework::ExecutionContext& ctx) {
    const phi::DenseTensor* logits = ctx.Input<phi::DenseTensor>("Logits");
    const phi::DenseTensor* labels = ctx.Input<phi::DenseTensor>("Label");
    const phi::DenseTensor* smooth_weight =
        ctx.Input<phi::DenseTensor>("SmoothWeight");
    phi::DenseTensor* softmax = ctx.Output<phi::DenseTensor>("Softmax");
    phi::DenseTensor* loss = ctx.Output<phi::DenseTensor>("Loss");

    const int64_t ignore_index = ctx.Attr<int64_t>("ignore_index");
    const bool sum_multi_label_loss = ctx.Attr<bool>("sum_multi_label_loss");
    const int rid = ctx.Attr<int>("ring_id");
    const int nranks = ctx.Attr<int>("nranks");
    const int rank = ctx.Attr<int>("rank");

    const auto& place = ctx.GetPlace();
    auto& dev_ctx = ctx.template device_context<phi::GPUContext>();

    auto map = distributed::ProcessGroupMapFromGid::getInstance();
    distributed::ProcessGroup* pg = map->get(rid);
    distributed::AllreduceOptions opts;
    opts.reduce_op = distributed::ReduceOp::MAX;

    // allocate memory on device.
    softmax->mutable_data<T>(place);
    loss->mutable_data<T>(place);

    const auto& logits_dims = logits->dims();
    const auto& labels_dims = labels->dims();

    const int axis = logits_dims.size() - 1;
    const int64_t N = phi::funcs::SizeToAxis<int64_t>(axis, logits_dims);
    const int64_t D = phi::funcs::SizeFromAxis<int64_t>(axis, logits_dims);
    const int64_t C = phi::funcs::SizeFromAxis<int64_t>(axis, labels_dims);

    phi::DenseTensor logits_2d, softmax_2d, loss_2d;
    logits_2d.ShareDataWith(*logits).Resize({N, D});
    softmax_2d.ShareDataWith(*softmax).Resize({N, D});
    int64_t loss_last_dim = sum_multi_label_loss ? 1 : C;
    loss_2d.ShareDataWith(*loss).Resize({N, loss_last_dim});

    auto eigen_logits = phi::funcs::EigenMatrix<T>::From(logits_2d);
    auto eigen_softmax = phi::funcs::EigenMatrix<T>::From(softmax_2d);

    // step 1, obtain logit_max
    phi::DenseTensor logits_max;
    logits_max = ctx.AllocateTmpTensor<T, phi::GPUContext>({N, 1}, dev_ctx);

    auto eigen_logits_max = phi::funcs::EigenMatrix<T>::From(logits_max);
    Eigen::DSizes<int, 1> along_axis(1);
    eigen_logits_max.device(*dev_ctx.eigen_device()) =
        eigen_logits.maximum(along_axis);

    pg->AllReduce(&logits_max, logits_max, opts, true, true);

    // step 2, obtain logit - logit_max
    Eigen::DSizes<int, 2> batch_by_one(N, 1);
    Eigen::DSizes<int, 2> one_by_class(1, D);

    eigen_softmax.device(*dev_ctx.eigen_device()) =
        (eigen_logits -
         eigen_logits_max.reshape(batch_by_one).broadcast(one_by_class));

    // step 3, obtain predict target
    phi::DenseTensor predicted_logits;
    predicted_logits =
        ctx.AllocateTmpTensor<T, phi::GPUContext>({N, C}, dev_ctx);
    predicted_logits.mutable_data<T>(place);

    auto t = phi::EigenVector<T>::Flatten(predicted_logits);
    t.device(*dev_ctx.eigen_device()) = t.constant(static_cast<T>(0));

    const int64_t start_index = rank * D;
    const int64_t end_index = start_index + D;

    int64_t blocks = NumBlocks(N);
    int threads = kNumCUDAThreads;
    const auto& label_type = framework::TransToProtoVarType(labels->dtype());

    if (label_type == framework::proto::VarType::INT32) {
      SoftMaskLabelByIndex<T, int32_t>
          <<<blocks, threads, 0, dev_ctx.stream()>>>(
              predicted_logits.data<T>(),
              softmax_2d.data<T>(),
              labels->data<int32_t>(),
              static_cast<int32_t>(ignore_index),
              start_index,
              end_index,
              N,
              D,
              C,
              nranks);
    } else if (label_type == framework::proto::VarType::INT64) {
      SoftMaskLabelByIndex<T, int64_t>
          <<<blocks, threads, 0, dev_ctx.stream()>>>(
              predicted_logits.data<T>(),
              softmax_2d.data<T>(),
              labels->data<int64_t>(),
              static_cast<int32_t>(ignore_index),
              start_index,
              end_index,
              N,
              D,
              C,
              nranks);
    }

    opts.reduce_op = distributed::ReduceOp::SUM;
    pg->AllReduce(&predicted_logits, predicted_logits, opts, true, true);

    // step 4, obtain exp(logit)
    eigen_softmax.device(*dev_ctx.eigen_device()) = eigen_softmax.exp();

    // step 5, obtain sum_exp_logits
    phi::DenseTensor sum_exp_logits;
    sum_exp_logits = ctx.AllocateTmpTensor<T, phi::GPUContext>({N, 1}, dev_ctx);
    void* sum_exp_logits_buff = sum_exp_logits.mutable_data<T>(place);

    phi::SumKernel<T, phi::GPUContext>(
        dev_ctx, softmax_2d, {-1}, softmax_2d.dtype(), true, &sum_exp_logits);

    opts.reduce_op = distributed::ReduceOp::SUM;
    pg->AllReduce(&sum_exp_logits, sum_exp_logits, opts, true, true);

    if (label_type == framework::proto::VarType::INT32) {
      CaculateSoftLoss<T, int32_t>
          <<<blocks, threads, 0, dev_ctx.stream()>>>(loss_2d.data<T>(),
                                                     predicted_logits.data<T>(),
                                                     sum_exp_logits.data<T>(),
                                                     labels->data<int32_t>(),
                                                     smooth_weight->data<T>(),
                                                     ignore_index,
                                                     N,
                                                     C,
                                                     sum_multi_label_loss);

    } else {
      CaculateSoftLoss<T, int64_t>
          <<<blocks, threads, 0, dev_ctx.stream()>>>(loss_2d.data<T>(),
                                                     predicted_logits.data<T>(),
                                                     sum_exp_logits.data<T>(),
                                                     labels->data<int64_t>(),
                                                     smooth_weight->data<T>(),
                                                     ignore_index,
                                                     N,
                                                     C,
                                                     sum_multi_label_loss);
    }

    auto eigen_sum_exp_logits =
        phi::funcs::EigenMatrix<T>::From(sum_exp_logits);
    eigen_softmax.device(*dev_ctx.eigen_device()) =
        (eigen_softmax *
         eigen_sum_exp_logits.inverse().broadcast(one_by_class));
  }
};

template <typename T, typename DeviceContext>
class CSoftmaxWithMultiLabelCrossEntropyGradCUDAKernel
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const phi::DenseTensor* labels = context.Input<phi::DenseTensor>("Label");
    const phi::DenseTensor* smooth_weight =
        context.Input<phi::DenseTensor>("SmoothWeight");
    const phi::DenseTensor* loss_grad =
        context.Input<phi::DenseTensor>(framework::GradVarName("Loss"));
    phi::DenseTensor* logit_grad =
        context.Output<phi::DenseTensor>(framework::GradVarName("Logits"));
    const phi::DenseTensor* softmax =
        context.Input<phi::DenseTensor>("Softmax");

    const int64_t ignore_index = context.Attr<int64_t>("ignore_index");
    const bool sum_multi_label_loss =
        context.Attr<bool>("sum_multi_label_loss");
    const int rank = context.Attr<int>("rank");
    auto& dev_ctx = context.template device_context<phi::GPUContext>();

    if (logit_grad != softmax) {
      framework::TensorCopy(
          *softmax, context.GetPlace(), context.device_context(), logit_grad);
    }
    const auto sofrmax_dims = softmax->dims();
    const int axis = sofrmax_dims.size() - 1;
    const int64_t N = phi::funcs::SizeToAxis<int64_t>(axis, sofrmax_dims);
    const int64_t D = phi::funcs::SizeFromAxis<int64_t>(axis, sofrmax_dims);

    const auto label_dims = labels->dims();
    const int64_t C = label_dims[axis];

    phi::DenseTensor logit_grad_2d;
    logit_grad_2d.ShareDataWith(*logit_grad).Resize({N, D});

    int64_t blocks = NumBlocks(N * D);
    int64_t blocks_cal = NumBlocks(N);
    int threads = kNumCUDAThreads;
    const auto& label_type = framework::TransToProtoVarType(labels->dtype());
    const int64_t start_index = rank * D;
    const int64_t end_index = start_index + D;

    if (label_type == framework::proto::VarType::INT32) {
      CaculateSoftLogitsGrad<T, int32_t>
          <<<blocks_cal, threads, 0, dev_ctx.stream()>>>(
              logit_grad_2d.data<T>(),
              labels->data<int32_t>(),
              smooth_weight->data<T>(),
              ignore_index,
              start_index,
              end_index,
              N,
              D,
              C);

      SoftMaskLabelByIndexGrad<T, int32_t>
          <<<blocks, threads, 0, dev_ctx.stream()>>>(logit_grad_2d.data<T>(),
                                                     loss_grad->data<T>(),
                                                     N,
                                                     D,
                                                     C,
                                                     sum_multi_label_loss);

    } else if (label_type == framework::proto::VarType::INT64) {
      CaculateSoftLogitsGrad<T, int64_t>
          <<<blocks_cal, threads, 0, dev_ctx.stream()>>>(
              logit_grad_2d.data<T>(),
              labels->data<int64_t>(),
              smooth_weight->data<T>(),
              ignore_index,
              start_index,
              end_index,
              N,
              D,
              C);

      SoftMaskLabelByIndexGrad<T, int64_t>
          <<<blocks, threads, 0, dev_ctx.stream()>>>(logit_grad_2d.data<T>(),
                                                     loss_grad->data<T>(),
                                                     N,
                                                     D,
                                                     C,
                                                     sum_multi_label_loss);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

PD_REGISTER_STRUCT_KERNEL(c_softmax_with_multi_label_cross_entropy,
                          GPU,
                          ALL_LAYOUT,
                          ops::CSoftmaxWithMultiLabelCrossEntropyOpCUDAKernel,
                          float,
                          double,
                          phi::dtype::float16) {}
PD_REGISTER_STRUCT_KERNEL(c_softmax_with_multi_label_cross_entropy_grad,
                          GPU,
                          ALL_LAYOUT,
                          ops::CSoftmaxWithMultiLabelCrossEntropyGradCUDAKernel,
                          float,
                          double,
                          phi::dtype::float16) {}
