/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <thread>
#include "paddle/framework/op_registry.h"
#include "paddle/operators/detail/safe_ref.h"
#include "paddle/platform/place.h"
#ifdef PADDLE_WITH_CUDA
#include "paddle/platform/gpu_info.h"
#endif

namespace paddle {
namespace operators {

static size_t CUDADevCount() {
#ifdef PADDLE_WITH_CUDA
  return platform::GetCUDADeviceCount();
#else
  return 0UL;
#endif
}

class GetPlacesOp : public framework::OperatorBase {
 public:
  GetPlacesOp(const std::string &type, const framework::VariableNameMap &inputs,
              const framework::VariableNameMap &outputs,
              const framework::AttributeMap &attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}
  void Run(const framework::Scope &scope,
           const platform::Place &place) const override {
    std::string device_type = Attr<std::string>("device_type");
    auto device_count = static_cast<size_t>(Attr<int>("device_count"));
    if (device_count == 0) {
      if (device_type == "CUDA") {
        device_count = CUDADevCount();
      } else if (device_type == "CPU") {
        device_count = std::thread::hardware_concurrency();
      }
    }
    PADDLE_ENFORCE_NE(device_count, 0, "Cannot indicate %s device count",
                      device_type);

    auto out_var_name = Output("Out");
    auto &places =
        *(detail::Ref(scope.FindVar(out_var_name),
                      "Output variable %s cannot be found", out_var_name)
              .GetMutable<platform::PlaceList>());
    places.reserve(device_count);
    if (device_type == "CUDA") {
      PADDLE_ENFORCE_LE(device_count, CUDADevCount(),
                        "Only %d CUDA devices found, cannot set to %d",
                        CUDADevCount(), device_count);
      for (size_t i = 0; i < device_count; ++i) {
        places.emplace_back(platform::CUDAPlace(i));
      }
    } else if (device_type == "CPU") {
      for (size_t i = 0; i < device_count; ++i) {
        places.emplace_back(platform::CPUPlace());
      }
    }
  }
};

class GetPlacesOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  GetPlacesOpProtoMaker(OpProto *proto, OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddOutput("Out", "vector of Place");
    AddAttr<int>("device_count", "device count").SetDefault(1);
    AddAttr<std::string>("device_type",
                         R"(device type must be in ["CPU", "CUDA"])")
        .InEnum({"CPU", "CUDA"});
    AddComment(R"DOC(
Returns a list of places based on flags. The list will be used for parallel
execution.
)DOC");
  }
};

class GetPlacesInferVarType : public framework::VarTypeInference {
 public:
  void operator()(const framework::OpDesc &op_desc,
                  framework::BlockDesc *block) const override {
    for (auto &o_name : op_desc.Output("Out")) {
      block->FindRecursiveOrCreateVar(o_name).SetType(
          framework::proto::VarDesc::PLACE_LIST);
    }
  }
};

class GetPlacesInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *context) const override {
    // Do nothing
  }
};

}  // namespace operators
}  // namespace paddle
namespace ops = paddle::operators;

REGISTER_OPERATOR(get_places, ops::GetPlacesOp, ops::GetPlacesOpProtoMaker,
                  ops::GetPlacesInferVarType, ops::GetPlacesInferShape);
