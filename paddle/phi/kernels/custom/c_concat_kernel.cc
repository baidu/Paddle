// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/api/backward/backward_api.h"
#include "paddle/phi/api/include/api.h"
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/backends/device_manager.h"
#include "paddle/phi/core/distributed/collective/process_group.h"
#include "paddle/phi/core/distributed/comm_context_manager.h"
#include "paddle/phi/core/distributed/xccl_comm_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#ifdef PADDLE_WITH_CUSTOM_DEVICE
namespace phi {

template <typename T, typename Context>
void CConcatKernel(const Context& dev_ctx,
                   const DenseTensor& x_in,
                   int rank,
                   int nranks,
                   int ring_id UNUSED,
                   bool use_calc_stream UNUSED,
                   bool use_model_parallel UNUSED,
                   DenseTensor* out) {
  auto x = &x_in;
  int rid = ring_id;
  auto place = dev_ctx.GetPlace();

  PADDLE_ENFORCE_GE(rank,
                    0,
                    common::errors::PreconditionNotMet(
                        "The value of rank (%d) for c_concat must be "
                        "greater than or equal to 0.",
                        rank));
  PADDLE_ENFORCE_GE(nranks,
                    2,
                    common::errors::PreconditionNotMet(
                        "The value of nranks (%d) for c_concat must be "
                        "greater than or equal to 2.",
                        nranks));
  PADDLE_ENFORCE_LT(rank,
                    nranks,
                    common::errors::PreconditionNotMet(
                        "The value of rank (%d) for c_concat must be "
                        "less than that of nranks (%d).",
                        rank,
                        nranks));

  phi::DenseTensor temp_out;
  phi::DDim temp_out_dims = x->dims();
  temp_out_dims[0] *= nranks;
  temp_out.Resize(temp_out_dims);
  dev_ctx.template Alloc<T>(&temp_out);

  auto map = distributed::ProcessGroupMapFromGid::getInstance();
  if (map->has(rid)) {
    // Use ProcessGroup
    distributed::ProcessGroup* pg = map->get(rid);
    std::vector<phi::DenseTensor> in_tensor;
    std::vector<phi::DenseTensor> out_tensor;
    in_tensor.push_back(*x);
    out_tensor.push_back(temp_out);
    auto task = pg->AllGather(in_tensor, out_tensor);
    task->Wait();
  } else {
    auto comm = reinterpret_cast<phi::distributed::XCCLCommContext*>(
        phi::distributed::CommContextManager::GetInstance().Get(
            std::to_string(rid)));
    PADDLE_ENFORCE_EQ(
        nranks,
        comm->GetSize(),
        common::errors::InvalidArgument(
            "nranks: %s should equal to %s", nranks, comm->GetSize()));

    int64_t send_numel = x->numel();
    const T* send_buff = x->data<T>();
    T* recv_buff = temp_out.data<T>();
    // should ExecutionContext for calc stream.
    auto& stream = *dev_ctx.GetStream();
    phi::DeviceManager::CCLAllGather(
        place.GetDeviceType(),
        reinterpret_cast<void*>(const_cast<T*>(send_buff)),
        recv_buff,
        send_numel,
        x->dtype(),
        comm->GetXcclComm(),
        stream);
  }
  std::vector<phi::DenseTensor> inputs;
  int axis = x->dims().size() - 1;
  auto out_dims = x->dims();
  out_dims[out_dims.size() - 1] *= nranks;
  int rows_per_tensor = x->dims()[0];
  int offset = 0;
  for (int i = 0; i < nranks; i++) {
    phi::DenseTensor temp = temp_out.Slice(offset, offset + rows_per_tensor);
    inputs.emplace_back(temp);
    offset += rows_per_tensor;
  }

  out->Resize(out_dims);
  std::vector<paddle::Tensor> inputs_t(inputs.size());
  for (size_t i = 0; i < inputs.size(); i++) {
    auto t = std::make_shared<phi::DenseTensor>();
    t->ShareDataWith(inputs[i]);
    inputs_t[i].set_impl(t);
  }
  auto output = paddle::experimental::concat(inputs_t, axis);
  out->ShareDataWith(*reinterpret_cast<phi::DenseTensor*>(output.impl().get()));
}
}  // namespace phi

PD_REGISTER_KERNEL(c_concat,
                   Custom,
                   ALL_LAYOUT,
                   phi::CConcatKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
#endif
