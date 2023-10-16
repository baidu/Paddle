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

#include "paddle/fluid/framework/new_executor/instruction/phi_kernel_instruction.h"

#include "paddle/fluid/framework/new_executor/interpreter/interpreter_util.h"
#include "paddle/fluid/framework/new_executor/interpreter/stream_analyzer.h"
#include "paddle/fluid/framework/new_executor/pir_adaptor/pir_adaptor_util.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/pir/dialect/operator/interface/infermeta.h"
#include "paddle/fluid/pir/dialect/operator/interface/op_yaml_info.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/utils/op_yaml_info_parser.h"
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/core/meta_tensor.h"
#include "paddle/phi/core/type_defs.h"

#include "paddle/pir/core/builtin_attribute.h"
#include "paddle/pir/core/operation.h"
#include "paddle/pir/core/value.h"

#include "paddle/fluid/framework/new_executor/instruction/instruction_util.h"
namespace paddle {
namespace framework {

PhiKernelInstruction::PhiKernelInstruction(
    size_t id,
    const platform::Place& place,
    pir::Operation* op,
    const ValueExecutionInfo& value_exec_info)
    : InstructionBase(id, place), value_exec_info_(value_exec_info) {
  auto op_attributes = op->attributes();
  auto op_name =
      op_attributes.at("op_name").dyn_cast<pir::StrAttribute>().AsString();
  pir::OpInfo op_info =
      pir::IrContext::Instance()->GetRegisteredOpInfo(op_name);
  op_ = op;
  phi_op_name_ = op_name;
  VLOG(6) << "construct phi kernel instruction for: " << phi_op_name_;

  // Todo: support paddle::dialect::DistAttribute
  //   if (op_attributes.count("dist_attr") != 0) {
  //     if (op_attributes.count("execution_stream") != 0) {
  //         SetExecutionStream(op_attributes.at("execution_stream")
  //                             .dyn_cast<pir::StrAttribute>()
  //                             .data());
  //     }
  //     if (op_attributes.count("stream_priority") != 0) {
  //         SetStreamPriority(op_attributes.at("stream_priority")
  //                             .dyn_cast<pir::Int32Attribute>()
  //                             .data());
  //     }
  //     if (op_attributes.count("scheduling_priority") != 0) {
  //         SetSchedulingPriority(op_attributes.at("scheduling_priority")
  //                                 .dyn_cast<pir::Int64Attribute>()
  //                                 .data());
  //     }
  //   } else {
  //     if (interpreter::IsCommunicationOp(op)) {
  //       // NOTE(Ruibiao): Dispatching computation before communication
  //       improves
  //       // multi-stream overlap when the time cost of communication less than
  //       // that of the calculation (e.g., ResNet50_bs128_pure_fp16 N4C32
  //       // training).
  //       op_func_node.scheduling_priority_ = 1;
  //     }
  //   }
  VLOG(6) << "finish process dist attributes";

  SetKernelType(AnalyseOpFuncType(op, place));
  VLOG(6) << "finish process analyse kernel type";

  infer_meta_interface_ =
      op_info.GetInterfaceImpl<paddle::dialect::InferMetaInterface>();
  VLOG(6) << "finish process infer_meta_interface_";

  auto yaml_interface =
      op_info.GetInterfaceImpl<paddle::dialect::OpYamlInfoInterface>();
  PADDLE_ENFORCE_NOT_NULL(
      yaml_interface,
      phi::errors::PreconditionNotMet(
          "can not find OpYamlInfoInterface from [%s]", phi_op_name_));
  paddle::dialect::OpYamlInfoParser yaml_info_parser(
      yaml_interface->get_op_info_());
  VLOG(6) << "finish process yaml_info_parser";

  if (infer_meta_interface_) {
    BuildPhiContext<
        phi::InferMetaContext,
        phi::MetaTensor,
        phi::MetaTensor,
        paddle::small_vector<phi::MetaTensor, phi::kInputSmallVectorSize>,
        paddle::small_vector<phi::MetaTensor, phi::kInputSmallVectorSize>,
        false>(op, value_exec_info_, yaml_info_parser, &infer_meta_context_);
  }
  VLOG(6) << "finish process infer meta context";

  auto kernel_name =
      op_attributes.at("kernel_name").dyn_cast<pir::StrAttribute>().AsString();
  auto kernel_key = op_attributes.at("kernel_key")
                        .dyn_cast<paddle::dialect::KernelAttribute>()
                        .data();
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      kernel_name, kernel_key);
  phi_kernel_ = new phi::Kernel(kernel_result.kernel);
  PADDLE_ENFORCE_EQ(
      phi_kernel_->IsValid(), true, "not found kernel for [%s]", kernel_name);
  VLOG(6) << "finish process select kernel";

  BuildPhiContext<phi::KernelContext,
                  const phi::TensorBase*,
                  phi::TensorBase*,
                  paddle::small_vector<const phi::TensorBase*>,
                  paddle::small_vector<phi::TensorBase*>,
                  true>(
      op, value_exec_info_, yaml_info_parser, &kernel_context_);

  for (size_t i = 0; i < op->num_results(); ++i) {
    std::cerr << "var name  " << value_exec_info_.GetVarName(op->result(i))
              << std::endl;
  }

  kernel_context_.SetDeviceContext(phi::DeviceContextPool::Instance().Get(
      phi::TransToPhiPlace(kernel_key.backend())));
  VLOG(6) << "finish process kernel context";

  SetDeviceContext(
      ParseDeviceContext(op,
                         phi::DeviceContextPool::Instance().Get(
                             phi::TransToPhiPlace(kernel_key.backend())),
                         place,
                         GetExecutionStream(),
                         GetStreamPriority()));
  VLOG(6) << "finish process device context";

  InitInputsOutputsIds(op, value_exec_info);
  VLOG(6) << "finish process inputs outputs index";

  auto& no_need_buffer_ids = yaml_info_parser.NoNeedBufferIds();
  std::unordered_set<pir::Value> no_need_buffer_values;
  for (size_t id = 0; id < no_need_buffer_ids.size(); id++) {
    no_need_buffer_values.insert(op->operand_source(no_need_buffer_ids[id]));
  }
  SetNoNeedBuffer(no_need_buffer_values);
  VLOG(6) << "finish process no need buffer";
}

PhiKernelInstruction::~PhiKernelInstruction() {
  if (phi_kernel_ != nullptr) {
    delete phi_kernel_;
  }
}

void PhiKernelInstruction::Run() {
  std::cerr << "before phi kernel run" << std::endl;
  if (infer_meta_interface_) {
    infer_meta_interface_->infer_meta_(&(infer_meta_context_));
  }
  std::cerr << "fin infer meta " << std::endl;
  VLOG(6) << "Run op " << phi_op_name_ << " infer meta.";
  (*(phi_kernel_))(&(kernel_context_));
  VLOG(6) << "Run op " << phi_op_name_ << " kernel.";
  std::cerr << "fin kernel run " << std::endl;
}

}  // namespace framework
}  // namespace paddle
