//   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/imperative/gloo_context.h"

#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/fleet/gloo_wrapper.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/platform/device_context.h"
#include "paddle/utils/string/split.h"
#include "paddle/utils/string/string_helper.h"

namespace paddle::framework {
class Variable;
}  // namespace paddle::framework

namespace paddle::imperative {

void GLOOParallelContext::Init() {
  // PADDLE_THROW(common::errors::OutOfRange(
  //  "Still not implement Init"));
  VLOG(4) << "Start GLOOParallelContext initialization";
  auto gloo_wrapper = framework::GlooWrapper::GetInstance();
  gloo_wrapper->SetSize(strategy_.nranks_);
  gloo_wrapper->SetRank(strategy_.local_rank_);
  gloo_wrapper->SetPrefix("");
  gloo_wrapper->SetIface("");
  auto addr = paddle::string::Split(strategy_.trainer_endpoints_[0], ':');
  VLOG(4) << "Server is" << strategy_.trainer_endpoints_[0];
  std::string host = addr[0];
  int port = std::stoi(addr[1]);
  gloo_wrapper->SetHttpStore(host, port, "worker");
  gloo_wrapper->Init();
  device_ = std::make_unique<phi::CPUContext>(phi::CPUPlace());
  device_->SetAllocator(paddle::memory::allocation::AllocatorFacade::Instance()
                            .GetAllocator(phi::CPUPlace())
                            .get());
  device_->SetHostAllocator(
      paddle::memory::allocation::AllocatorFacade::Instance()
          .GetAllocator(phi::CPUPlace())
          .get());
  device_->SetZeroAllocator(
      paddle::memory::allocation::AllocatorFacade::Instance()
          .GetZeroAllocator(phi::CPUPlace())
          .get());
}

void GLOOParallelContext::InitWithRingID(int ring_id) {
  PADDLE_THROW(
      common::errors::OutOfRange("Still not implement InitWithRingID"));
}

#define GLOO_CASE(type, T, gw)                                  \
  case type: {                                                  \
    std::vector<T> send_vector##T;                              \
    framework::TensorToVector<T>(src_tensor, &send_vector##T);  \
    auto recv_vector##T = gw->AllReduce<T>(send_vector##T);     \
    framework::TensorFromVector<T>(recv_vector##T, dst_tensor); \
    break;                                                      \
  }

void GLOOParallelContext::AllReduceByStream(const framework::Variable &src,
                                            framework::Variable *dst,
                                            int ring_id,
                                            bool use_calc_stream) {
  // AllReduce(src, dst, strategy_, ring_id, use_calc_stream);
  if (src.IsType<phi::DenseTensor>()) {
    if (!dst->IsType<phi::DenseTensor>()) {
      dst->Clear();
    }
    AllReduce(src.Get<phi::DenseTensor>(), dst->GetMutable<phi::DenseTensor>());
  } else if (src.IsType<phi::SelectedRows>()) {
    if (&src != dst) {
      if (!dst->IsType<phi::SelectedRows>()) {
        dst->Clear();
      }
      AllReduce(src.Get<phi::SelectedRows>(),
                dst->GetMutable<phi::SelectedRows>());
    } else {
      // SelectedRows cannot be allreduce in-place
      framework::Variable tmp_dst;
      AllReduce(src.Get<phi::SelectedRows>(),
                tmp_dst.GetMutable<phi::SelectedRows>());
      *dst = std::move(tmp_dst);
    }
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "Unsupported variable type %s for imperative allreduce, only "
        "DenseTensor and SelectedRows are supported.",
        common::demangle(framework::ToTypeName(src.Type()))));
  }
}

void GLOOParallelContext::AllReduce(const phi::DenseTensor &src_tensor,
                                    phi::DenseTensor *dst_tensor) {
  auto gloo_wrapper = framework::GlooWrapper::GetInstance();
  dst_tensor->Resize(src_tensor.dims());
  switch (framework::TransToProtoVarType(src_tensor.dtype())) {
    GLOO_CASE(framework::proto::VarType::FP32, float, gloo_wrapper);
    GLOO_CASE(framework::proto::VarType::FP64, double, gloo_wrapper);
    GLOO_CASE(framework::proto::VarType::INT32, int, gloo_wrapper);
    GLOO_CASE(framework::proto::VarType::INT64, int64_t, gloo_wrapper);
    default: {
      PADDLE_THROW(
          common::errors::InvalidArgument("Invalid datatype for allreduce"));
    }
  }
  gloo_wrapper->Barrier();
}

#define GLOO_ALL_GATHER_CASE(type, T, gw)                         \
  case type: {                                                    \
    const auto *src_tensor_ptr = src_tensor.data<T>();            \
    gw->AllGatherVector<T>(const_cast<T *>(src_tensor_ptr),       \
                           reinterpret_cast<T *>(dst_tensor_ptr), \
                           element_nums);                         \
    break;                                                        \
  }

void GLOOParallelContext::AllReduce(const phi::SelectedRows &src,
                                    phi::SelectedRows *dst) {
  // auto ;
  // int local_rank = strategy_.local_rank_;
  int nranks = strategy_.nranks_;
  VLOG(3) << "SelectedRows AllReduce start";
  const auto &src_tensor = src.value();
  const auto &place = src_tensor.place();
  auto dtype = framework::TransToProtoVarType(src_tensor.dtype());
  // 1. Gather rows number from all workers. Here use ncclAllGather to do this,
  // but we can use other ways to implement is in the future
  auto &src_rows = src.rows();
  auto gloo_wrapper = framework::GlooWrapper::GetInstance();
  size_t local_row_num = src_rows.size();
  std::vector<size_t> rows_num_vector =
      gloo_wrapper->AllGather<size_t>(local_row_num);
  const auto *cpu_rows_num_ptr = rows_num_vector.data();
  auto rows_num = std::accumulate(
      cpu_rows_num_ptr, cpu_rows_num_ptr + nranks, static_cast<int64_t>(0));
  dst->set_height(src.height());
  VLOG(3) << "Gather rows: " << string::join_strings(rows_num_vector, ',')
          << ", total rows number: " << rows_num
          << ", height: " << src.height();
  auto *dst_rows = dst->mutable_rows();
  dst_rows->resize(rows_num);
  phi::MixVector<int64_t> mixv_dst_rows(dst_rows);
  auto *dst_rows_ptr = mixv_dst_rows.MutableData(place);
  phi::MixVector<int64_t> mixv_src_rows(&src_rows);
  const int64_t *src_rows_ptr = mixv_src_rows.Data(place);

  auto *dst_tensor = dst->mutable_value();
  auto dims = src_tensor.dims();
  dims[0] = rows_num;
  auto feature_size = common::product(dims) / dims[0];
  dst_tensor->Resize(dims);

  std::vector<size_t> element_nums = rows_num_vector;
  std::for_each(element_nums.begin(),
                element_nums.end(),
                [feature_size](size_t &x) { x = x * feature_size; });

  auto *dst_tensor_ptr = dst_tensor->mutable_data(place, src_tensor.dtype());
  gloo_wrapper->AllGatherVector<int64_t>(const_cast<int64_t *>(src_rows_ptr),
                                         static_cast<int64_t *>(dst_rows_ptr),
                                         rows_num_vector);

  switch (dtype) {
    GLOO_ALL_GATHER_CASE(framework::proto::VarType::FP32, float, gloo_wrapper);
    GLOO_ALL_GATHER_CASE(framework::proto::VarType::FP64, double, gloo_wrapper);
    GLOO_ALL_GATHER_CASE(framework::proto::VarType::INT32, int, gloo_wrapper);
    GLOO_ALL_GATHER_CASE(
        framework::proto::VarType::INT64, int64_t, gloo_wrapper);
    default: {
      PADDLE_THROW(
          common::errors::InvalidArgument("Invalid datatype for allreduce"));
    }
  }
}

void GLOOParallelContext::Broadcast(framework::Variable *src, int ring_id) {
  PADDLE_THROW(common::errors::Unimplemented(
      "Unimplemented inter-broadcast for CPU now."));
}

phi::DeviceContext *GLOOParallelContext::GetDeviceContext(int ring_id) {
  // return the CPUContext
  return device_.get();
}

void GLOOParallelContext::WaitCompute(int ring_id) {
  // do nothing because cpu don't need sync
  return;
}

void GLOOParallelContext::WaitComm(int ring_id) {
  // do nothing because cpu don't need sync
  return;
}

void GLOOParallelContext::SynchronizeCompute() {
  // do nothing because cpu don't need sync
  return;
}

}  // namespace paddle::imperative
