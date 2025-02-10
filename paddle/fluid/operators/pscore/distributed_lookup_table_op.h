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

#pragma once
#include <algorithm>
#include <string>
#include <vector>

#include "paddle/fluid/distributed/ps/wrapper/fleet.h"
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {

template <typename T, typename DeviceContext>
class DistributedLookupTableKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto padding_idx = context.Attr<int64_t>("padding_idx");
    auto table_id = context.Attr<int>("table_id");
    bool is_test = context.Attr<bool>("is_test");

    auto *var = context.InputVar("W");
    int64_t emb_dim = 0;

    if (var->IsType<phi::DenseTensor>()) {
      emb_dim = var->Get<phi::DenseTensor>().dims()[1];
    } else if (var->IsType<phi::SelectedRows>()) {
      emb_dim = var->Get<phi::SelectedRows>().value().dims()[1];
    } else {
      PADDLE_THROW(common::errors::InvalidArgument(
          "Expected type of `W` must be Tensor, SelectedRows.But got "
          "unsupported type: %s.",
          framework::ToTypeName(var->Type())));
    }

    auto inputs = context.MultiInput<phi::DenseTensor>("Ids");
    auto outputs = context.MultiOutput<phi::DenseTensor>("Outputs");

    auto fleet = distributed::FleetWrapper::GetInstance();

    if (context.GetPlace().GetType() == phi::AllocationType::CPU) {
      fleet->PullSparseToTensorSync(static_cast<uint64_t>(table_id),
                                    emb_dim,
                                    static_cast<uint64_t>(padding_idx),
                                    context.GetPlace(),
                                    !is_test,
                                    &inputs,
                                    &outputs);
    } else {
      auto inputs_variable = context.MultiInputVar("Ids");
      auto outputs_variable = context.MultiOutputVar("Outputs");

      auto cpu_place = phi::CPUPlace();

      std::vector<const phi::DenseTensor *> tmp_input_vec;
      auto input_var_size = inputs_variable.size();
      std::vector<phi::DenseTensor *> tmp_output_vec;
      auto output_var_size = outputs_variable.size();

      std::vector<std::shared_ptr<phi::DenseTensor>> tmp_tensors;

      // create temp input
      for (size_t idx = 0; idx < input_var_size; ++idx) {
        tmp_tensors.emplace_back(std::make_shared<phi::DenseTensor>());
        auto *p = tmp_tensors.back().get();
        framework::TensorCopy(inputs_variable[idx]->Get<phi::DenseTensor>(),
                              cpu_place,
                              context.device_context(),
                              p);
        tmp_input_vec.push_back(p);
      }

      // create temp output
      for (size_t idx = 0; idx < output_var_size; ++idx) {
        tmp_tensors.emplace_back(std::make_shared<phi::DenseTensor>());
        auto *p = tmp_tensors.back().get();
        p->Resize(outputs[idx]->dims());
        tmp_output_vec.push_back(p);
      }

      // use fleet->PullSparse
      fleet->PullSparseToTensorSync(static_cast<uint64_t>(table_id),
                                    emb_dim,
                                    static_cast<uint64_t>(padding_idx),
                                    cpu_place,
                                    !is_test,
                                    &tmp_input_vec,
                                    &tmp_output_vec);

      // cp temp to origin
      for (size_t idx = 0; idx < output_var_size; ++idx) {
        framework::TensorCopy(
            *tmp_output_vec[idx],
            context.GetPlace(),
            context.device_context(),
            outputs_variable[idx]->GetMutable<phi::DenseTensor>());
      }
    }

    auto lookup_table_version =
        context.Attr<std::string>("lookup_table_version");
    auto id_vars = context.MultiInputVar("Ids");
    auto out_vars = context.MultiOutputVar("Outputs");

    if (lookup_table_version == "lookup_table_v2") {
      for (size_t i = 0; i < id_vars.size(); ++i) {
        auto *id_tensor = id_vars[i]->GetMutable<phi::DenseTensor>();
        auto *out_tensor = out_vars[i]->GetMutable<phi::DenseTensor>();

        auto id_dims = common::vectorize<int64_t>(id_tensor->dims());
        out_tensor->Resize(common::make_ddim({static_cast<int64_t>(id_dims[0]),
                                              static_cast<int64_t>(id_dims[1]),
                                              static_cast<int64_t>(emb_dim)}));
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
