/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include "paddle/fluid/operators/data/map_op.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/imperative/type_defs.h"

namespace paddle {
namespace operators {

using framework::Tensor;

class MapOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasOutputs("X"), "Input", "X", "MapOp");
    OP_INOUT_CHECK(ctx->HasOutputs("Out"), "Output", "Out", "MapOp");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(framework::proto::VarType::FP32,
                                   ctx.GetPlace());
  }

  framework::OpKernelType GetKernelTypeForVar(
      const std::string& var_name, const framework::Tensor& tensor,
      const framework::OpKernelType& expected_kernel_type) const override {
    return expected_kernel_type;
  }
};

class MapOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("In",
             "(LoDTensorBlockingQueueHolder)"
              "The output tensors of Map operator")
        .AsDuplicable();
    AddOutput("Out",
              "(LoDTensorBlockingQueueHolder)"
              "The output tensors of Map operator")
        .AsDuplicable();
    AddAttr<BlockDesc*>("global_block",
                        "(BlockDesc *)"
                        "The global block of executed map program "
                        "desc.");
    AddAttr<int64_t>("start_op_index",
                     "(int64_t)"
                     "The index of the op to start execution");
    AddAttr<int64_t>("end_op_index",
                     "(int64_t)"
                     "The index of the op to stop execution");
    AddAttr<int64_t>("program_id",
                     "(int64_t)"
                     "The unique hash id used as cache key for "
                     "ExecutorInfoCache");
    AddAttr<std::vector<std::string>>("input_var_names",
                     "(list of string)"
                     "input variable names for map program");
    AddAttr<std::vector<std::string>>("output_var_names",
                     "(list of string)"
                     "output variable names for map program");
    AddComment(R"DOC(
        Map Op
         )DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(map, ops::MapOp, ops::MapOpMaker);
REGISTER_OP_CPU_KERNEL(map, ops::MapOpKernel<paddle::platform::CPUDeviceContext, float>);
