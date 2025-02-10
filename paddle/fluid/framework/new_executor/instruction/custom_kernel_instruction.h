// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/new_executor/instruction/instruction_base.h"
#include "paddle/fluid/pir/dialect/operator/utils/op_yaml_info_parser.h"
#include "paddle/phi/api/ext/op_meta_info.h"

namespace pir {
class Operation;
}  // namespace pir

namespace paddle {
namespace framework {
class Scope;
class CustomKernelInstruction : public InstructionBase {
 public:
  CustomKernelInstruction(size_t id,
                          const phi::Place& place,
                          ::pir::Operation* op,
                          const ValueExecutionInfo& value_exec_info);

  ::pir::Operation* Operation() const override { return op_; }

  void Run() override;

  const std::string& Name() const override { return custom_op_name_; }

  void clear();

 private:
  void BuildCustomContext(
      const paddle::dialect::OpYamlInfoParser& op_yaml_info);

  void BuildShapeDtype();

  void UpdateOutputMeta(const std::vector<std::vector<int64_t>>& output_shapes,
                        const std::vector<DataType>& output_dtypes);

  paddle::CustomOpKernelContext custom_kernel_ctx_;

  paddle::InferShapeFunc infershape_func_ = nullptr;
  paddle::InferDtypeFunc inferdtype_func_ = nullptr;
  paddle::KernelFunc kernel_func_ = nullptr;

  // key is input name, value is a index in input_shapes_ or vec_input_shapes_
  std::unordered_map<std::string, int> input_name2id_map_;
  std::unordered_map<std::string, int> vec_input_name2id_map_;

  // use for running infershape
  std::vector<std::vector<int64_t>> input_shapes_;
  std::vector<std::vector<std::vector<int64_t>>> vec_input_shapes_;
  std::vector<paddle::any> custom_attrs_;

  // use for running inferdtype
  std::vector<DataType> input_dtypes_;
  std::vector<std::vector<DataType>> vec_input_dtypes_;

  // use for calculate input shapes and dtypes in runtime
  std::vector<phi::DenseTensor*> input_ptrs_;
  std::vector<std::vector<phi::DenseTensor*>> vec_input_ptrs_;

  // use for update output
  std::vector<phi::DenseTensor*> cache_out_ptrs_;

  std::string custom_op_name_;

  ::pir::Operation* op_{nullptr};  // not owned

  const paddle::OpMetaInfo* custom_op_meta_;   // not owned
  const ValueExecutionInfo& value_exec_info_;  // not owned
};

}  // namespace framework
}  // namespace paddle
