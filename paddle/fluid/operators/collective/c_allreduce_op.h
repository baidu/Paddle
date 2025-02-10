/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

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

#include <string>

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/core/distributed/collective/process_group.h"
#include "paddle/phi/core/distributed/comm_context_manager.h"
#include "paddle/phi/core/memory/memcpy.h"
#include "paddle/phi/core/memory/memory.h"

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL) || \
    defined(PADDLE_WITH_XPU_BKCL)
#include "paddle/common/flags.h"
#include "paddle/phi/core/platform/collective_helper.h"
COMMON_DECLARE_bool(dynamic_static_unified_comm);
#endif

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/fluid/platform/device/gpu/nccl_helper.h"
#include "paddle/phi/core/distributed/nccl_comm_context.h"
#elif defined(PADDLE_WITH_XPU_BKCL)
#include "paddle/fluid/platform/device/xpu/bkcl_helper.h"
#include "paddle/phi/core/distributed/bkcl_comm_context.h"
#endif

#if defined(PADDLE_WITH_GLOO)
#include "paddle/phi/core/distributed/gloo_comm_context.h"
#endif

namespace paddle {
namespace operators {

enum ReduceType { kRedSum, kRedMax, kRedMin, kRedProd, kRedAvg };

class CAllReduceOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    ctx->SetOutputDim("Out", ctx->GetInputDim("X"));
  }

 protected:
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return phi::KernelKey(OperatorWithKernel::IndicateVarDataType(ctx, "X"),
                          ctx.GetPlace());
  }

  phi::KernelKey GetKernelTypeForVar(
      const std::string& var_name,
      const phi::DenseTensor& tensor,
      const phi::KernelKey& expected_kernel_type) const {
    if (var_name == "Cond") {
      return phi::KernelKey(phi::Backend::ALL_BACKEND,
                            expected_kernel_type.layout(),
                            expected_kernel_type.dtype());
    } else {
      return phi::KernelKey(
          tensor.place(), tensor.layout(), expected_kernel_type.dtype());
    }
  }
};

template <ReduceType red_type, typename T>
class CAllReduceOpCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
#if defined(PADDLE_WITH_GLOO)
    auto& dev_ctx = ctx.device_context<phi::CPUContext>();
    auto x = *(ctx.Input<phi::DenseTensor>("X"));
    auto out = ctx.Output<phi::DenseTensor>("Out");
    out->Resize(x.dims());
    dev_ctx.Alloc<T>(out);

    auto comm_ctx = static_cast<phi::distributed::GlooCommContext*>(
        dev_ctx.GetCommContext());
    PADDLE_ENFORCE_NE(comm_ctx,
                      nullptr,
                      ::common::errors::Unavailable(
                          "NCCLCommContext is nullptr, collective op should "
                          "has ring_id attr."));
    comm_ctx->AllReduce(out, x, static_cast<int>(red_type));
#else
    PADDLE_THROW(common::errors::Unavailable(
        "PaddlePaddle should compile with GLOO by setting WITH_GLOO=ON"));
#endif
  }
};

#define DEFINE_C_ALLREDUCE_CPU_KERNEL(op_name, red_type) \
  template <typename T, typename DeviceContext>          \
  class op_name##CPUKernel : public CAllReduceOpCPUKernel<red_type, T> {};

template <ReduceType red_type, typename T>
class CAllReduceOpXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
#if defined(PADDLE_WITH_XPU_BKCL)
    if (ctx.HasInput("Cond")) {
      auto cond = ctx.Input<phi::DenseTensor>("Cond");
      auto place = cond->place();
      PADDLE_ENFORCE_EQ(place.GetType() == phi::AllocationType::CPU,
                        true,
                        common::errors::PreconditionNotMet(
                            "The input `cond` tensor should be on cpu place"));
      PADDLE_ENFORCE_EQ(cond->numel(),
                        1,
                        common::errors::PreconditionNotMet(
                            "The input `cond` should be shape [1]"));
      if (!cond->data<bool>()[0]) {
        VLOG(4) << "Skip all reduce Op since cond is 0";
        return;
      }
    }

    auto in = ctx.Input<phi::DenseTensor>("X");
    auto out = ctx.Output<phi::DenseTensor>("Out");
    int rid = ctx.Attr<int>("ring_id");

    auto place = ctx.GetPlace();
    BKCLDataType dtype = phi::ToBKCLDataType(in->dtype());
    int64_t numel = in->numel();
    const void* sendbuff = in->data<T>();
    out->Resize(in->dims());
    void* recvbuff = out->mutable_data<T>(place);

    auto map = phi::distributed::ProcessGroupMapFromGid::getInstance();
    if (map->has(rid)) {
      // Use ProcessGroup
      phi::distributed::ProcessGroup* pg = map->get(rid);
      phi::distributed::AllreduceOptions opts;
      switch (red_type) {
        case kRedSum:
          opts.reduce_op = phi::distributed::ReduceOp::SUM;
          break;

        case kRedMax:
          opts.reduce_op = phi::distributed::ReduceOp::MAX;
          break;

        case kRedMin:
          opts.reduce_op = phi::distributed::ReduceOp::MIN;
          break;

        case kRedProd:
          opts.reduce_op = phi::distributed::ReduceOp::PRODUCT;
          break;

        default:
          PADDLE_THROW(common::errors::InvalidArgument(
              "Invalid reduce type: %d", red_type));
      }

      auto task = pg->AllReduce(out, *in, opts, false, true);
      task->Wait();
      return;
    }

    XPUStream stream = nullptr;
    platform::BKCLComm* comm = nullptr;
    phi::distributed::BKCLCommContext* comm_ctx = nullptr;

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
      comm_ctx = static_cast<phi::distributed::BKCLCommContext*>(
          comm_context_manager.Get(std::to_string(rid)));
      PADDLE_ENFORCE_NE(comm_ctx,
                        nullptr,
                        common::errors::Unavailable(
                            "BKCLCommContext is nullptr, collective op should "
                            "has ring_id attr."));
      stream = comm_ctx->GetStream();
      VLOG(3) << "new comm_context_manager has rid " << rid;
    } else {
      comm = platform::BKCLCommContext::Instance().Get(rid, place);
      stream = comm->stream();
      VLOG(3) << "old BKCLCommContext has rid " << rid;
    }
    if (ctx.Attr<bool>("use_calc_stream")) {
      auto dev_ctx = phi::DeviceContextPool::Instance().Get(place);
      stream = static_cast<phi::XPUContext*>(dev_ctx)->x_context()->xpu_stream;
    }

    BKCLOp bkcl_red_type = BKCL_ADD;
    switch (red_type) {
      case kRedSum:
        bkcl_red_type = BKCL_ADD;
        break;

      case kRedMax:
        bkcl_red_type = BKCL_MAX;
        break;

      case kRedMin:
        bkcl_red_type = BKCL_MIN;
        break;

      case kRedProd:
        bkcl_red_type = BKCL_PRODUCT;
        break;

      default:
        PADDLE_THROW(common::errors::InvalidArgument("Invalid reduce type: %d",
                                                     red_type));
    }

    if (comm_ctx) {
      comm_ctx->AllReduce(out, *in, bkcl_red_type, stream);
    } else {
      PADDLE_ENFORCE_XPU_SUCCESS(bkcl_all_reduce(comm->comm(),
                                                 sendbuff,
                                                 recvbuff,
                                                 numel,
                                                 dtype,
                                                 bkcl_red_type,
                                                 stream));
    }
#else
    PADDLE_THROW(common::errors::PreconditionNotMet(
        "PaddlePaddle should be compiled with XPU."));
#endif
  }
};

#define DEFINE_C_ALLREDUCE_XPU_KERNEL(op_name, red_type) \
  template <typename T, typename DeviceContext>          \
  class op_name##XPUKernel : public CAllReduceOpXPUKernel<red_type, T> {};

template <ReduceType red_type, typename T>
class CAllReduceOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    if (ctx.HasInput("Cond")) {
      auto cond = ctx.Input<phi::DenseTensor>("Cond");
      auto place = cond->place();
      PADDLE_ENFORCE_EQ(place.GetType() == phi::AllocationType::CPU,
                        true,
                        common::errors::PreconditionNotMet(
                            "The input `cond` tensor should be on cpu place"));
      PADDLE_ENFORCE_EQ(cond->numel(),
                        1,
                        common::errors::PreconditionNotMet(
                            "The input `cond` should be shape [1]"));
      if (!cond->data<bool>()[0]) {
        VLOG(4) << "Skip all reduce Op since cond is 0";
        return;
      }
    }

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
    auto in = ctx.Input<phi::DenseTensor>("X");
    auto out = ctx.Output<phi::DenseTensor>("Out");
    int rid = ctx.Attr<int>("ring_id");

    auto place = ctx.GetPlace();
    ncclDataType_t dtype = phi::ToNCCLDataType(in->dtype());
    int64_t numel = in->numel();
    const void* sendbuff = in->data<T>();
    out->Resize(in->dims());
    void* recvbuff = out->mutable_data<T>(place);

    auto map = phi::distributed::ProcessGroupMapFromGid::getInstance();
    if (map->has(rid)) {
      // Use ProcessGroup
      phi::distributed::ProcessGroup* pg = map->get(rid);
      phi::distributed::AllreduceOptions opts;
      switch (red_type) {
        case kRedSum:
          opts.reduce_op = phi::distributed::ReduceOp::SUM;
          break;

        case kRedMax:
          opts.reduce_op = phi::distributed::ReduceOp::MAX;
          break;

        case kRedMin:
          opts.reduce_op = phi::distributed::ReduceOp::MIN;
          break;

        case kRedProd:
          opts.reduce_op = phi::distributed::ReduceOp::PRODUCT;
          break;

        default:
          PADDLE_THROW(common::errors::InvalidArgument(
              "Invalid reduce type: %d", red_type));
      }

      auto task = pg->AllReduce(out, *in, opts, false, true);
      task->Wait();
      return;
    }

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
      stream = comm_ctx->GetStream();
      VLOG(3) << "new comm_context_manager has rid " << rid;
    } else {
      comm = platform::NCCLCommContext::Instance().Get(rid, place);
      stream = comm->stream();
      VLOG(3) << "old NCCLCommContext has rid " << rid;
    }
    if (ctx.Attr<bool>("use_calc_stream")) {
      // should not use global ctx for calc stream.
      // auto dev_ctx = phi::DeviceContextPool::Instance().Get(place);
      // stream = static_cast<phi::GPUContext*>(dev_ctx)->stream();
      stream = ctx.cuda_device_context().stream();
    }
    VLOG(10) << "all reduce buffer:" << sendbuff << ", numel:" << numel
             << ", reduce type:" << static_cast<int>(red_type)
             << ", dtype:" << dtype << ", comm:" << comm
             << ", stream:" << stream;

    ncclRedOp_t nccl_red_type = ncclSum;
    switch (red_type) {
      case kRedSum:
        nccl_red_type = ncclSum;
        break;

      case kRedMax:
        nccl_red_type = ncclMax;
        break;

      case kRedMin:
        nccl_red_type = ncclMin;
        break;

      case kRedProd:
        nccl_red_type = ncclProd;
        break;

#if (NCCL_VERSION_CODE >= 21000 && CUDA_VERSION >= 11000) || \
    defined(PADDLE_WITH_HIP)
      case kRedAvg:
        nccl_red_type = ncclAvg;
        break;
#endif

      default:
        PADDLE_THROW(common::errors::InvalidArgument("Invalid reduce type: %d",
                                                     red_type));
    }

    if (comm_ctx) {
      comm_ctx->AllReduce(out, *in, nccl_red_type, stream);
    } else {
      PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::ncclAllReduce(sendbuff,
                                                             recvbuff,
                                                             numel,
                                                             dtype,
                                                             nccl_red_type,
                                                             comm->comm(),
                                                             stream));
    }
#else
    PADDLE_THROW(common::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU."));
#endif
  }
};

#define DEFINE_C_ALLREDUCE_CUDA_KERNEL(op_name, red_type) \
  template <typename T, typename DeviceContext>           \
  class op_name##CUDAKernel : public CAllReduceOpCUDAKernel<red_type, T> {};

class CAllReduceOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("X", "(Tensor), tensor to be allreduced.");
    AddOutput("Out", "(Tensor) the allreduced result.");
    AddAttr<int>("ring_id", "(int default 0) communication ring id.")
        .SetDefault(0);
    AddAttr<bool>(
        "use_calc_stream",
        "(bool default false) eject CUDA operations to calculation stream.")
        .SetDefault(false);
    AddAttr<bool>(
        "use_model_parallel",
        "(bool default false) use this op with model parallel mode. In model "
        "parallel mode, the backward is c_identity which returns itself for "
        "c_allreduce_sum.")
        .SetDefault(false);
    AddComment(string::Sprintf(R"DOC(
CAllReduce %s Operator

Call collective AllReduce with reduce type %s. If input and output are
the same variable, in-place allreduce will be used.
Reference: https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/docs/usage/operations.html#allreduce
)DOC",
                               GetName(),
                               GetName()));
    ExtraMake();
  }

 protected:
  virtual std::string GetName() const = 0;
  virtual void ExtraMake() {}
};

}  // namespace operators
}  // namespace paddle
