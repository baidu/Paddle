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

#include <string>

#include "paddle/fluid/framework/data_layout_transform.h"
#include "paddle/fluid/framework/feed_fetch_type.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/core/platform/device_context.h"

namespace paddle {
namespace framework {
class OpDesc;
class InferShapeContext;
template <typename T>
class EmptyGradOpMaker;
}  // namespace framework
namespace imperative {
class OpBase;
}  // namespace imperative
}  // namespace paddle

namespace paddle {
namespace operators {

static void DeepCopy(const phi::DenseTensor &src_item,
                     const std::string &fetch_var_name,
                     phi::DenseTensor *dst_item) {
  if (src_item.IsInitialized()) {
#ifdef PADDLE_WITH_DNNL
    // Conversion from MKL-DNN to Paddle
    if (src_item.layout() == phi::DataLayout::ONEDNN) {
      phi::DenseTensor out;
      // Convert to desired Paddle layout, apart from grads of filter
      // as params are not a subject to paddle's data_format
      phi::funcs::TransDataLayoutFromOneDNN(
          src_item.layout(),
          fetch_var_name == framework::GradVarName("Filter")
              ? phi::DataLayout::kNCHW
              : phi::OneDNNContext::tls().get_cur_paddle_data_layout(),
          src_item,
          &out,
          phi::CPUPlace());
      paddle::framework::TensorCopySync(out, phi::CPUPlace(), dst_item);
    } else {
      paddle::framework::TensorCopySync(src_item, phi::CPUPlace(), dst_item);
    }
#else
    paddle::framework::TensorCopySync(src_item, phi::CPUPlace(), dst_item);
#endif
  } else {
    VLOG(4) << "No copy";
  }
  dst_item->set_lod(src_item.lod());
}

class FetchV2Op : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {}

 protected:
  phi::KernelKey GetKernelTypeForVar(
      const std::string &var_name,
      const phi::DenseTensor &tensor,
      const phi::KernelKey &expected_kernel_type) const override {
    if (!tensor.IsInitialized()) {
      return phi::KernelKey(phi::Backend::ALL_BACKEND,
                            expected_kernel_type.layout(),
                            expected_kernel_type.dtype());
    }
    return phi::KernelKey(
        tensor.place(), tensor.layout(), expected_kernel_type.dtype());
  }

  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto *fetch_var = ctx.InputVar("X");
    if (fetch_var == nullptr) {
      return phi::KernelKey(framework::proto::VarType::FP32, phi::CPUPlace());
    }

    if (fetch_var->IsType<phi::DenseTensor>()) {
      auto &src_item = fetch_var->Get<phi::DenseTensor>();
      if (!src_item.IsInitialized()) {
        return phi::KernelKey(framework::proto::VarType::FP32, phi::CPUPlace());
      }
    } else if (fetch_var->IsType<phi::SparseCooTensor>()) {
      auto &src_item = fetch_var->Get<phi::SparseCooTensor>();
      if (!src_item.initialized()) {
        return phi::KernelKey(framework::proto::VarType::FP32, phi::CPUPlace());
      }
    } else {
      auto &src_item = fetch_var->Get<phi::TensorArray>();
      if (src_item.empty() || !src_item[0].IsInitialized()) {
        return phi::KernelKey(framework::proto::VarType::FP32, phi::CPUPlace());
      }
    }

    return phi::KernelKey(OperatorWithKernel::IndicateVarDataType(ctx, "X"),
                          phi::CPUPlace());
  }
};

template <typename T, typename DeviceContext>
class FetchV2Kernel {
 public:
  void operator()(const framework::ExecutionContext &ctx) const {
    auto fetch_var_name = ctx.InputName("X");
    auto *fetch_var = ctx.InputVar("X");
    if (fetch_var == nullptr) {
      return;
    }
    PADDLE_ENFORCE_EQ(
        ctx.HasOutput("Out"),
        true,
        common::errors::NotFound("Output(Out) of fetch_v2_op is not found."));
    auto *out_var = ctx.OutputVar("Out");

    int col = ctx.Attr<int>("col");
    PADDLE_ENFORCE_GE(
        col,
        0,
        common::errors::InvalidArgument(
            "Expected the column index (the attribute 'col' of "
            "operator 'Fetch') of current fetching variable to be "
            "no less than 0. But received column index = %d.",
            col));
    VLOG(3) << "Fetch variable " << fetch_var_name << "'s " << col
            << " column.";

    auto *fetch_list = out_var->GetMutable<framework::FetchList>();

    if (static_cast<size_t>(col) >= fetch_list->size()) {
      fetch_list->resize(col + 1);
    }

    bool deepcopy = ctx.Attr<bool>("deepcopy");

    if (fetch_var->IsType<phi::DenseTensor>()) {
      auto &src_item = fetch_var->Get<phi::DenseTensor>();
      if (!src_item.IsInitialized()) {
        return;
      }
      auto *dst_item = &(PADDLE_GET(phi::DenseTensor, fetch_list->at(col)));
      bool check_place =
          src_item.place().GetType() == phi::AllocationType::CPU ||
          src_item.place().GetType() == phi::AllocationType::GPUPINNED ||
          src_item.place().GetType() == phi::AllocationType::CUSTOM;
      PADDLE_ENFORCE_EQ(
          check_place,
          true,
          common::errors::InvalidArgument("Tensor's place of input(X) must "
                                          "be CPUPlace or CUDAPinnedPlace."));
      if (deepcopy) {
        DeepCopy(src_item, fetch_var_name, dst_item);
      } else {
        dst_item->ShareDataWith(src_item);
        dst_item->set_lod(src_item.lod());
      }
    } else if (fetch_var->IsType<phi::SparseCooTensor>()) {
      auto &src_item = fetch_var->Get<phi::SparseCooTensor>();
      if (!src_item.initialized()) {
        return;
      }
      fetch_list->at(col) = src_item;
    } else {
      auto &src_item = fetch_var->Get<phi::TensorArray>();
      phi::TensorArray tmp(src_item.size());
      fetch_list->at(col) = tmp;
      auto &dst_item = PADDLE_GET(phi::TensorArray, fetch_list->at(col));
      for (size_t i = 0; i < src_item.size(); ++i) {
        PADDLE_ENFORCE_EQ(
            src_item[i].place().GetType() == phi::AllocationType::CPU,
            true,
            common::errors::InvalidArgument(
                "Tensor's place of input(X) must be CPUPlace."));
        if (deepcopy) {
          DeepCopy(src_item[i], fetch_var_name, &dst_item[i]);
        } else {
          dst_item[i].ShareDataWith(src_item[i]);
          dst_item[i].set_lod(src_item[i].lod());
        }
      }
    }
  }
};

class FetchV2OpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(phi::DenseTensor) The resulted phi::DenseTensor which is "
             "expected to return "
             "to users.");
    AddOutput("Out",
              "(vector<phi::DenseTensor>) A fetching list of phi::DenseTensor "
              "which may have "
              "different dimension, shape and data type.");
    AddAttr<int>("col", "(int) The column index of fetching object.");
    AddAttr<bool>("deepcopy", "(bool) Whether deep copy is required.")
        .SetDefault(true);
    AddComment(R"DOC(
FetchV2 Operator.
It should not be configured by users directly.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    fetch_v2,
    ops::FetchV2Op,
    ops::FetchV2OpProtoMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
