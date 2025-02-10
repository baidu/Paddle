//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/details/op_handle_base.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/scope.h"
#ifdef PADDLE_WITH_CUDA
#include "paddle/phi/backends/dynload/nccl.h"
#endif
#ifdef PADDLE_WITH_HIP
#include "paddle/phi/backends/dynload/rccl.h"
#endif
#include "paddle/common/flags.h"
#include "paddle/fluid/platform/device/gpu/nccl_helper.h"

COMMON_DECLARE_bool(sync_nccl_allreduce);

namespace paddle {
namespace framework {
namespace details {

class NCCLOpHandleBase : public OpHandleBase {
 public:
  NCCLOpHandleBase(ir::Node* node,
                   const std::vector<phi::Place>& places,
                   const platform::NCCLCommunicator* nccl_ctxs)
      : OpHandleBase(node), places_(places), nccl_ctxs_(nccl_ctxs) {
    if (nccl_ctxs == nullptr) {
      return;
    }
    // init device context
    auto default_nccl_ctxs = nccl_ctxs_->DefaultFlatCtx();
    for (auto& p : places_) {
      this->SetDeviceContext(p, default_nccl_ctxs->DevCtx(p));
    }
  }
  virtual ~NCCLOpHandleBase() {
    for (auto& ev : inter_events_) {
#ifdef PADDLE_WITH_HIP
      PADDLE_ENFORCE_GPU_SUCCESS(hipEventDestroy(ev.second));
#else
      PADDLE_ENFORCE_GPU_SUCCESS(cudaEventDestroy(ev.second));
#endif
    }
    for (auto& ev : exter_events_) {
#ifdef PADDLE_WITH_HIP
      PADDLE_ENFORCE_GPU_SUCCESS(hipEventDestroy(ev.second));
#else
      PADDLE_ENFORCE_GPU_SUCCESS(cudaEventDestroy(ev.second));
#endif
    }
  }

  const platform::NCCLCommunicator* GetNcclContext() const {
    return nccl_ctxs_;
  }

  ncclComm_t GetComm() const {
    PADDLE_ENFORCE_EQ(
        places_.size(),
        1,
        common::errors::Unimplemented(
            "Only supported for single place now, but got %d", places_.size()));
    PADDLE_ENFORCE_EQ(use_hierarchical_allreduce_,
                      0,
                      common::errors::Unimplemented(
                          "Not supported use_hierarchical_allreduce_ now"));
    PADDLE_ENFORCE_NOT_NULL(
        nccl_ctxs_,
        common::errors::NotFound("Can't get flat %d nccl contexts.",
                                 run_order_));
    auto flat_nccl_ctxs = nccl_ctxs_->GetFlatCtx(run_order_);
    int dev_id = places_[0].device;
    auto& nccl_ctx = flat_nccl_ctxs->at(dev_id);
    auto comm = nccl_ctx.comm_;
    return comm;
  }

  void SetRunEnv(int run_order, bool use_hierarchical_allreduce) {
    PADDLE_ENFORCE_GE(
        run_order,
        0,
        common::errors::InvalidArgument(
            "The argument run_order must be >= 0, but got %d.", run_order));
    run_order_ = run_order;
    use_hierarchical_allreduce_ = use_hierarchical_allreduce;

    VLOG(10) << "SetRunEnv "
             << " run_order:" << run_order
             << ", use_hierarchical_allreduce:" << use_hierarchical_allreduce
             << ", nccl_ctx_:" << nccl_ctxs_;

    if (nccl_ctxs_ == nullptr) {
      return;
    }

    if (!use_hierarchical_allreduce_) {
      auto ctxs = nccl_ctxs_->GetFlatCtx(run_order);
      for (auto& p : places_) {
        this->SetDeviceContext(p, ctxs->DevCtx(p));
      }
      return;
    }

    PADDLE_ENFORCE_EQ(places_.size(),
                      1,
                      common::errors::InvalidArgument(
                          "HierarchicalAllReduce can only run "
                          "one process with one card mode, but got %d cards.",
                          places_.size()));

    for (auto& p : places_) {
      auto ctxs = nccl_ctxs_->GetHierarchicalInterCtx(run_order);
      this->SetDeviceContext(p, ctxs->DevCtx(p));
    }

    for (auto& p : dev_ctxes_) {
      int dev_id = p.first.device;
      if (inter_events_.find(dev_id) != inter_events_.end()) {
        continue;
      }

      platform::SetDeviceId(dev_id);
#ifdef PADDLE_WITH_HIP
      PADDLE_ENFORCE_GPU_SUCCESS(hipEventCreateWithFlags(
          &inter_events_[dev_id], hipEventDisableTiming));
      PADDLE_ENFORCE_GPU_SUCCESS(hipEventCreateWithFlags(
          &exter_events_[dev_id], hipEventDisableTiming));
#else
      PADDLE_ENFORCE_GPU_SUCCESS(cudaEventCreateWithFlags(
          &inter_events_[dev_id], cudaEventDisableTiming));
      PADDLE_ENFORCE_GPU_SUCCESS(cudaEventCreateWithFlags(
          &exter_events_[dev_id], cudaEventDisableTiming));
#endif
      VLOG(10) << "Create events on dev_id:" << dev_id
               << ", inter_event:" << &inter_events_[dev_id]
               << ", exter_event:" << &exter_events_[dev_id];
    }
  }

  void FlatNCCLAllReduce(phi::Place place,
                         const void* sendbuff,
                         void* recvbuff,
                         size_t count,
                         ncclDataType_t datatype,
                         ncclRedOp_t op) {
    PADDLE_ENFORCE_GE(
        run_order_,
        0,
        common::errors::InvalidArgument(
            "The argument run_order_ must be >= 0, but got %d.", run_order_));
    auto flat_nccl_ctxs = nccl_ctxs_->GetFlatCtx(run_order_);
    int dev_id = place.device;
    auto& nccl_ctx = flat_nccl_ctxs->at(dev_id);
    auto stream = nccl_ctx.stream();
    auto comm = nccl_ctx.comm_;

    VLOG(10) << "before all reduce buffer:" << sendbuff << ", numel:" << count
             << ", dev_id:" << dev_id << ", dtype:" << datatype
             << ", place:" << place;

    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::ncclAllReduce(
        sendbuff, recvbuff, count, datatype, op, comm, stream));
  }

  void NCCLAllReduce(phi::Place place,
                     const void* sendbuff,
                     void* recvbuff,
                     size_t count,
                     ncclDataType_t datatype,
                     ncclRedOp_t op) {
    PADDLE_ENFORCE_GE(
        run_order_,
        0,
        common::errors::InvalidArgument(
            "The argument run_order_ must be >= 0, but got %d.", run_order_));
    if (!use_hierarchical_allreduce_) {
      FlatNCCLAllReduce(place, sendbuff, recvbuff, count, datatype, op);
      return;
    }

    HierarchicalAllReduce(place, sendbuff, recvbuff, count, datatype, op);
  }

  void HierarchicalAllReduce(phi::Place place,
                             const void* sendbuff,
                             void* recvbuff,
                             size_t count,
                             ncclDataType_t datatype,
                             ncclRedOp_t op) {
    PADDLE_ENFORCE_GE(
        run_order_,
        0,
        common::errors::InvalidArgument(
            "The argument run_order_ must be >= 0, but got %d.", run_order_));
    InterReduce(place, sendbuff, recvbuff, count, datatype, op);
    // When a trainer is not in exter allreduce ring
    // they need not to call this.
    if (nccl_ctxs_->NeedExterAllReduce()) {
      ExterAllReduce(place, recvbuff, recvbuff, count, datatype, op);
    }
    InterBroadCast(place, recvbuff, count, datatype, op);
  }

 protected:
  void InterReduce(phi::Place place,
                   const void* sendbuff,
                   void* recvbuff,
                   size_t count,
                   ncclDataType_t datatype,
                   ncclRedOp_t op UNUSED) {
    auto nccl_ctxs = nccl_ctxs_->GetHierarchicalInterCtx(run_order_);
    int dev_id = place.device;
    auto& nccl_ctx = nccl_ctxs->at(dev_id);
    auto stream = nccl_ctx.stream();
    auto comm = nccl_ctx.comm_;

    VLOG(10) << "before all reduce"
             << " run_order:" << run_order_ << ", buffer:" << sendbuff
             << ", numel:" << count << ", dev_id:" << dev_id
             << ", dtype:" << datatype << ", place:" << place
             << ", stream:" << stream;

    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::ncclReduce(
        sendbuff, recvbuff, count, datatype, ncclSum, 0, comm, stream));

#ifdef PADDLE_WITH_HIP
    hipEventRecord(inter_events_.at(dev_id), stream);
#else
    cudaEventRecord(inter_events_.at(dev_id), stream);
#endif

    if (FLAGS_sync_nccl_allreduce) {
      platform::GpuStreamSync(stream);
    }
  }

  void ExterAllReduce(phi::Place place,
                      const void* sendbuff,
                      void* recvbuff,
                      size_t count,
                      ncclDataType_t datatype,
                      ncclRedOp_t op) {
    auto nccl_ctxs = nccl_ctxs_->GetHierarchicalExterCtx(run_order_);
    PADDLE_ENFORCE_NOT_NULL(
        nccl_ctxs_,
        common::errors::NotFound("Can't get exter %d nccl contexts.",
                                 run_order_));
    int dev_id = place.device;
    auto& nccl_ctx = nccl_ctxs->at(dev_id);
    auto stream = nccl_ctx.stream();
    auto comm = nccl_ctx.comm_;

    VLOG(10) << "before all reduce run_order:" << run_order_
             << "buffer:" << sendbuff << ", numel:" << count
             << ", dev_id:" << dev_id << ", dtype:" << datatype
             << ", place:" << place << ", stream:" << stream;

#ifdef PADDLE_WITH_HIP
    hipStreamWaitEvent(stream, inter_events_.at(dev_id), 0);

    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::ncclAllReduce(
        sendbuff, recvbuff, count, datatype, op, comm, stream));

    hipEventRecord(exter_events_.at(dev_id), stream);
#else
    cudaStreamWaitEvent(stream, inter_events_.at(dev_id), 0);

    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::ncclAllReduce(
        sendbuff, recvbuff, count, datatype, op, comm, stream));

    cudaEventRecord(exter_events_.at(dev_id), stream);
#endif
    if (FLAGS_sync_nccl_allreduce) {
      platform::GpuStreamSync(stream);
    }
  }

  void InterBroadCast(phi::Place place,
                      void* sendbuff,
                      size_t count,
                      ncclDataType_t datatype,
                      ncclRedOp_t op UNUSED) {
    auto nccl_ctxs = nccl_ctxs_->GetHierarchicalInterCtx(run_order_);
    int dev_id = place.device;
    auto& nccl_ctx = nccl_ctxs->at(dev_id);
    auto stream = nccl_ctx.stream();
    auto comm = nccl_ctx.comm_;

    VLOG(10) << "before InterBroadCast buffer:" << sendbuff
             << ", numel:" << count << ", dev_id:" << dev_id
             << ", dtype:" << datatype << ", place:" << place
             << ", stream:" << stream;
#ifdef PADDLE_WITH_HIP
    hipStreamWaitEvent(stream, exter_events_.at(dev_id), 0);
#else
    cudaStreamWaitEvent(stream, exter_events_.at(dev_id), 0);
#endif
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::ncclBcast(sendbuff, count, datatype, 0, comm, stream));
  }

 protected:
  std::vector<phi::Place> places_;
  const platform::NCCLCommunicator* nccl_ctxs_{nullptr};
  // When multi trainer call collective function, they need run the same order.
  // Or the program will hang.So we use allreduce_deps_pass to set this
  // run_order_.
  int run_order_{0};
  // Use 2d allreduce or not.
  bool use_hierarchical_allreduce_{false};

 private:
  // hierarchical needed events
  std::unordered_map<int, gpuEvent_t> inter_events_;
  std::unordered_map<int, gpuEvent_t> exter_events_;
};

}  // namespace details
}  // namespace framework
}  // namespace paddle
