/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/collective/recv_v2_op.h"

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/common/flags.h"
#include "paddle/fluid/platform/device/gpu/nccl_helper.h"
#include "paddle/phi/core/distributed/comm_context_manager.h"
#include "paddle/phi/core/distributed/nccl_comm_context.h"
#include "paddle/phi/core/platform/collective_helper.h"
COMMON_DECLARE_bool(dynamic_static_unified_comm);
#endif

#include "paddle/fluid/distributed/collective/process_group.h"
#include "paddle/phi/api/include/tensor.h"

namespace paddle {
namespace operators {

#if (defined(PADDLE_WITH_RCCL) || defined(PADDLE_WITH_NCCL)) && \
    NCCL_VERSION_CODE >= 2703
phi::DDim recv_shape_info(const phi::Place &place,
                          const gpuStream_t &stream,
                          platform::NCCLComm *comm,
                          phi::distributed::NCCLCommContext *comm_ctx,
                          const int &peer,
                          distributed::ProcessGroup *group) {
  if (!group) {
    PADDLE_ENFORCE_EQ(
        ((stream != nullptr && comm != nullptr) || comm_ctx != nullptr),
        true,
        common::errors::InvalidArgument(
            "NCCLComm and Stream should be provided if use NCCL "
            "to send the shape info."));
  }

  phi::DataType shape_dtype = phi::DataType::INT32;
  ncclDataType_t nccl_dtype = phi::ToNCCLDataType(shape_dtype);

  // step1: recv the shape size
  phi::DenseTensor gpu_shape_size_tensor(shape_dtype);
  if (!group) {
    gpu_shape_size_tensor.Resize({1});
    gpu_shape_size_tensor.mutable_data(place, shape_dtype);
    auto *gpu_data = gpu_shape_size_tensor.data<int>();

    if (comm_ctx) {
      comm_ctx->Recv(&gpu_shape_size_tensor, 1, peer, stream);
    } else {
      PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::ncclRecv(
          gpu_data, 1, nccl_dtype, peer, comm->comm(), stream));
    }
  }

  // copy the shape size tensor to cpu
  phi::DenseTensor *cpu_shape_size_tensor = new phi::DenseTensor(shape_dtype);
  cpu_shape_size_tensor->Resize({1});
  cpu_shape_size_tensor->mutable_data(phi::CPUPlace(), shape_dtype);
  if (group) {
    std::vector<phi::DenseTensor> shape_size_tensor;
    shape_size_tensor.emplace_back(*cpu_shape_size_tensor);
    auto shape_size_task = group->Recv(shape_size_tensor, peer);
  } else {
    framework::TensorCopySync(
        gpu_shape_size_tensor, phi::CPUPlace(), cpu_shape_size_tensor);
  }
  auto *cpu_data = cpu_shape_size_tensor->data<int>();
  int shape_size = cpu_data[0];
  VLOG(3) << "recv the shape size: " << shape_size << " from peer";

  // step2: recv the shape
  phi::DenseTensor gpu_shape_tensor(shape_dtype);
  if (!group) {
    gpu_shape_tensor.Resize({shape_size});
    gpu_shape_tensor.mutable_data(place, shape_dtype);
    auto *gpu_shape_data = gpu_shape_tensor.data<int>();
    if (comm_ctx) {
      comm_ctx->Recv(&gpu_shape_tensor, shape_size, peer, stream);
    } else {
      PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::ncclRecv(
          gpu_shape_data, shape_size, nccl_dtype, peer, comm->comm(), stream));
    }
  }

  // copy the shape tensor to cpu
  phi::DenseTensor *cpu_shape_tensor = new phi::DenseTensor(shape_dtype);
  cpu_shape_tensor->Resize({shape_size});
  cpu_shape_tensor->mutable_data(phi::CPUPlace(), shape_dtype);
  if (group) {
    std::vector<phi::DenseTensor> shape_tensor;
    shape_tensor.emplace_back(*cpu_shape_tensor);
    auto shape_task = group->Recv(shape_tensor, peer);
  } else {
    framework::TensorCopySync(
        gpu_shape_tensor, phi::CPUPlace(), cpu_shape_tensor);
  }
  auto *cpu_shape_data = cpu_shape_tensor->data<int>();
  std::vector<int> all_shape;
  for (int i = 0; i < shape_size; ++i) {
    all_shape.emplace_back(cpu_shape_data[i]);
  }
  phi::DDim new_dim;
  new_dim = new_dim.reshape(all_shape);
  VLOG(3) << "recv the shape: (" << new_dim << ") from peer";

  return new_dim;
}
#endif

template <typename T, typename DeviceContext>
class RecvOpV2CUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
#if (defined(PADDLE_WITH_RCCL) || defined(PADDLE_WITH_NCCL)) && \
    NCCL_VERSION_CODE >= 2703
    int rid = ctx.Attr<int>("ring_id");
    bool dynamic_shape = ctx.Attr<bool>("dynamic_shape");
    PADDLE_ENFORCE_GE(
        rid,
        0,
        common::errors::InvalidArgument(
            "The ring_id (%d) for recv_v2 op must be non-negative.", rid));

    int peer = ctx.Attr<int>("peer");
    PADDLE_ENFORCE_GE(
        peer,
        0,
        common::errors::InvalidArgument(
            "The peer (%d) for recv_v2 op must be non-negative.", peer));

    gpuStream_t stream = nullptr;
    auto place = ctx.GetPlace();
    auto map = distributed::ProcessGroupMapFromGid::getInstance();
    if (map->has(rid)) {
      // Use ProcessGroup
      distributed::ProcessGroup *pg = map->get(rid);
      std::vector<phi::DenseTensor> out_tensor;
      auto out_shape = ctx.Attr<std::vector<int>>("out_shape");
      auto out = ctx.Output<phi::DenseTensor>("Out");
      // auto out_dims = out->dims();

      if (dynamic_shape) {
        VLOG(3) << "recv_v2 will use dynamic shape with send_v2 for switch";
        phi::DDim new_dim = recv_shape_info(ctx.GetPlace(),
                                            /* gpuStream_t */ nullptr,
                                            /* NCCLComm* */ nullptr,
                                            /* NCCLCommContext* */ nullptr,
                                            peer,
                                            pg);
        out->Resize(new_dim);
        ctx.cuda_device_context().Alloc<T>(out);
      } else {
        ctx.cuda_device_context().Alloc<T>(out);
      }

      out_tensor.emplace_back(*out);
      auto task = pg->Recv(out_tensor, peer);
      return;
    }

    platform::NCCLComm *comm = nullptr;
    phi::distributed::NCCLCommContext *comm_ctx = nullptr;

    const auto &comm_context_manager =
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
      comm_ctx = static_cast<phi::distributed::NCCLCommContext *>(
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
      PADDLE_ENFORCE_LT(
          peer,
          comm->nranks(),
          common::errors::InvalidArgument("The value of peer (%d) you set must "
                                          "be less than comm->nranks (%d).",
                                          peer,
                                          comm->nranks()));
      stream = comm->stream();
      VLOG(3) << "old NCCLCommContext has rid " << rid;
    }

    if (ctx.Attr<bool>("use_calc_stream")) {
      // should ExecutionContext for calc stream.
      stream = ctx.cuda_device_context().stream();
    }
    int data_type = ctx.Attr<int>("dtype");
    framework::proto::VarType::Type type =
        framework::proto::VarType::Type(data_type);
    ncclDataType_t dtype = platform::ToNCCLDataType(type);

    auto *out_var = ctx.OutputVar("Out");
    if (out_var->IsType<phi::TensorArray>()) {
      PADDLE_ENFORCE_EQ(
          dynamic_shape,
          false,
          common::errors::InvalidArgument("Dynamic shape for send/recv not "
                                          "support DenseTensorArray for now."));
      auto out_array = out_var->GetMutable<phi::TensorArray>();
      for (size_t idx = 0; idx < out_array->size(); ++idx) {
        VLOG(3) << "DenseTensorArray: idx(" << idx << ")";
        auto out = &out_array->at(idx);
        auto out_dims = out->dims();
        ctx.cuda_device_context().Alloc<T>(out);
        auto numel = out->numel();
        if (comm_ctx) {
          comm_ctx->Recv(out, numel, peer, stream);
        } else {
          PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::ncclRecv(
              out->data<T>(), numel, dtype, peer, comm->comm(), stream));
          VLOG(3) << "rank " << comm->rank() << " recv "
                  << common::product(out_dims) << " from " << peer;
        }
      }
      return;
    }

    auto out_shape = ctx.Attr<std::vector<int>>("out_shape");
    auto out = ctx.Output<phi::DenseTensor>("Out");
    // auto out_dims = out->dims();
    auto numel = out->numel();

    if (dynamic_shape) {
      VLOG(3) << "recv_v2 will use dynamic shape with send_v2";
      phi::DDim new_dim = recv_shape_info(place,
                                          stream,
                                          comm,
                                          comm_ctx,
                                          peer,
                                          /* ProcessGroup* */ nullptr);
      out->Resize(new_dim);
      numel = out->numel();
      ctx.cuda_device_context().Alloc<T>(out);
    } else {
      ctx.cuda_device_context().Alloc<T>(out);
    }
    if (comm_ctx) {
      comm_ctx->Recv(out, numel, peer, stream);
    } else {
      comm = platform::NCCLCommContext::Instance().Get(rid, place);
      PADDLE_ENFORCE_LT(
          peer,
          comm->nranks(),
          common::errors::InvalidArgument("The value of peer (%d) you set must "
                                          "be less than comm->nranks (%d).",
                                          peer,
                                          comm->nranks()));
      PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::ncclRecv(
          out->data<T>(), numel, dtype, peer, comm->comm(), stream));
      VLOG(3) << "rank " << comm->rank() << " recv "
              << common::product(out->dims()) << " from " << peer;
    }
#else
    PADDLE_THROW(common::errors::Unavailable(
        "PaddlePaddle should be compiled with NCCL and "
        "NCCL version >= 2.7.3 is needed."));
#endif
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

PD_REGISTER_STRUCT_KERNEL(recv_v2,
                          GPU,
                          ALL_LAYOUT,
                          ops::RecvOpV2CUDAKernel,
                          float,
                          double,
#if (NCCL_VERSION_CODE >= 21000 && CUDA_VERSION >= 11000) || \
    defined(PADDLE_WITH_HIP)
                          phi::dtype::bfloat16,
#endif
                          int,
                          int64_t,
                          int8_t,
                          phi::dtype::float16) {
}
