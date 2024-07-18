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

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/var_type.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/phi/core/stream.h"

namespace phi {
class DenseTensor;
}  // namespace phi

namespace paddle {
namespace framework {
class Variable;
class SelectedRows;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace operators {
class MemcpyH2DFunctor {
 public:
  MemcpyH2DFunctor(framework::Variable *out,
                   const phi::DeviceContext &dev_ctx,
                   const int dst_place_type)
      : out_(out), dev_ctx_(dev_ctx), dst_place_type_(dst_place_type) {}

  void operator()(const phi::DenseTensor &lod_tensor) const {
    auto &out_tensor = *out_->GetMutable<phi::DenseTensor>();
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    auto stream = static_cast<const phi::GPUContext *>(&dev_ctx_)->stream();
#else
    auto stream = nullptr;
#endif
    out_tensor.mutable_data(
        dev_ctx_.GetPlace(),
        lod_tensor.dtype(),
        phi::Stream(reinterpret_cast<phi::StreamId>(stream)));

    if (dst_place_type_ >= 0 && dst_place_type_ <= 3) {
      framework::TensorCopy(
          lod_tensor, dev_ctx_.GetPlace(), dev_ctx_, &out_tensor);
    } else {
      PADDLE_THROW(phi::errors::Unimplemented(
          "memcpy dst_place_type: %d is not supported yet.", dst_place_type_));
    }
    out_tensor.set_lod(lod_tensor.lod());
  }

  void operator()(const phi::SelectedRows &rows) const {
    // (JZ-LIANG) to support SelectedRows
    PADDLE_THROW(phi::errors::Unimplemented(
        "Memcpy for SelectedRows is NOT support yet."));
  }

  template <typename T>
  void operator()(const T &v UNUSED) const {
    PADDLE_ENFORCE_EQ(
        true,
        false,
        phi::errors::PermissionDenied(
            "Not support type for Memcpy  op with type %s", typeid(T).name()));
  }

 private:
  framework::Variable *out_;
  const phi::DeviceContext &dev_ctx_;
  const int dst_place_type_;
};

}  // namespace operators
}  // namespace paddle
