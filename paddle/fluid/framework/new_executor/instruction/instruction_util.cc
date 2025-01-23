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

#include "paddle/fluid/framework/new_executor/instruction/instruction_util.h"

#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/new_executor/new_executor_defs.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_dialect.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/phi/api/profiler/event.h"
#include "paddle/phi/core/platform/device_context.h"
#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/core/operation.h"
#include "paddle/pir/include/core/value.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_op.h"

#include "paddle/fluid/framework/new_executor/interpreter/interpreter_util.h"
#include "paddle/fluid/framework/new_executor/interpreter/stream_analyzer.h"
#include "paddle/fluid/framework/new_executor/pir_adaptor/pir_adaptor_util.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/pir/include/core/block_argument.h"
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/common/flags.h"
#include "paddle/fluid/distributed/collective/process_group.h"
#include "paddle/fluid/distributed/collective/process_group_nccl.h"
#include "paddle/phi/core/distributed/comm_context_manager.h"
#include "paddle/phi/core/distributed/nccl_comm_context.h"
#include "paddle/phi/core/platform/collective_helper.h"
COMMON_DECLARE_bool(dynamic_static_unified_comm);
#endif

namespace paddle::framework {

std::vector<int> GetValueIds(pir::Value value,
                             const ValueExecutionInfo& value_exec_info) {
  std::vector<int> ids;
  ids.push_back(value_exec_info.GetVarId(value));
  // NOTE(zhangbo): Value maybe a VariableRefArray
  auto var =
      value_exec_info.GetScope()->FindVar(value_exec_info.GetVarName(value));
  if (var->IsType<paddle::framework::VariableRefArray>()) {
    auto& var_array = var->Get<paddle::framework::VariableRefArray>();
    for (auto item : var_array) {
      ids.push_back(value_exec_info.GetVarId(item));
    }
  }
  return ids;
}

phi::DeviceContext* ParseDeviceContext(pir::Operation* op,
                                       phi::DeviceContext* origin_dev_ctx,
                                       const phi::Place& place,
                                       const std::string& execution_stream,
                                       const int stream_priority) {
  auto& op_attributes = op->attributes();
  auto op_name =
      op_attributes.at("op_name").dyn_cast<pir::StrAttribute>().AsString();
  interpreter::ContextManager& ctx_manager =
      interpreter::ContextManager::Instance();

  phi::DeviceContext* dev_ctx = nullptr;

  // only gpu need update. xpu not need, because xpu memcpy op kernel is
  // synchronous.
  if (phi::is_gpu_place(place) || phi::is_custom_place(place)) {
    VLOG(6) << "Parse DeviceContext for " << op_name
            << ", execution stream = " << execution_stream;
    if (execution_stream != kDefaultStream) {
      dev_ctx = ctx_manager
                    .Get(std::string(kCustomStream) + "-" + execution_stream,
                         place,
                         stream_priority)
                    .get()
                    .get();
      interpreter::SetDeviceCommContext(op, dev_ctx);
      return dev_ctx;
    }

    if (op_name.compare(paddle::dialect::MemcpyD2hOp::name()) == 0) {
      dev_ctx = ctx_manager.Get(std::string(kD2HStream), place, stream_priority)
                    .get()
                    .get();
      interpreter::SetDeviceCommContext(op, dev_ctx);
      return dev_ctx;
    } else if (op_name.compare(paddle::dialect::MemcpyH2dOp::name()) == 0) {
      dev_ctx = ctx_manager.Get(std::string(kH2DStream), place, stream_priority)
                    .get()
                    .get();
      interpreter::SetDeviceCommContext(op, dev_ctx);
      return dev_ctx;
    }

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
    // NOTE(Ruibiao): Here supports multi-stream overlap for c_allreduce_sum
    // with use_cal_stream==false by returning a device context getting from the
    // global NCCLCommContext instance. Because when use_calc_stream==false, in
    // OP kernel, the NCCL communication will be launched to the stream directly
    // getting from the global NCCLCommContext instance rather than the
    // DeviceContext passed from executor (see CAllReduceOpCUDAKernel in
    // c_allreduce_op.h). Now it is just a temporary solution for ONLY
    // c_allreduce_sum which is used in ResNet50 distributed training.
    if ((op_name.compare(paddle::dialect::CAllreduceSumOp::name()) == 0 ||
         op_name.compare(paddle::dialect::CAllreduceSum_Op::name()) == 0) &&
        op_attributes.at("use_calc_stream")
                .dyn_cast<pir::BoolAttribute>()
                .data() == false) {
      int ring_id =
          op_attributes.at("ring_id").dyn_cast<pir::Int32Attribute>().data();
      if (FLAGS_dynamic_static_unified_comm) {
        const auto& comm_context_manager =
            phi::distributed::CommContextManager::GetInstance();
        dev_ctx = static_cast<phi::DeviceContext*>(
            static_cast<phi::distributed::NCCLCommContext*>(
                comm_context_manager.Get(std::to_string(ring_id)))
                ->GetDevContext());
      } else {
        dev_ctx = platform::NCCLCommContext::Instance()
                      .Get(ring_id, place)
                      ->dev_context();
      }
      return dev_ctx;
    }

    // handle comm op
    if (op_attributes.count("ring_id") != 0 &&
        FLAGS_dynamic_static_unified_comm) {
      int ring_id =
          op_attributes.at("ring_id").dyn_cast<pir::Int32Attribute>().data();
      const auto& comm_context_manager =
          phi::distributed::CommContextManager::GetInstance();

      phi::distributed::CommContext* comm_context = nullptr;
      if (comm_context_manager.Has(std::to_string(ring_id))) {
        comm_context = comm_context_manager.Get(std::to_string(ring_id));
      } else if (op_name.compare(paddle::dialect::MpAllreduceSum_Op::name()) ==
                     0 ||
                 op_name.compare(paddle::dialect::AllReduce_Op::name()) == 0 ||
                 op_name.compare(paddle::dialect::CIdentity_Op::name()) == 0 ||
                 op_name.compare(paddle::dialect::CConcatOp::name()) == 0 ||
                 op_name.compare(paddle::dialect::Broadcast_Op::name()) == 0) {
        auto map = distributed::ProcessGroupMapFromGid::getInstance();
        distributed::ProcessGroup* pg = map->get(ring_id);
        comm_context = static_cast<paddle::distributed::ProcessGroupNCCL*>(pg)
                           ->GetOrCreateCommContext(place);
      }

      if (comm_context) {
        dev_ctx = static_cast<platform::DeviceContext*>(
            static_cast<phi::distributed::NCCLCommContext*>(comm_context)
                ->GetDevContext());
        dev_ctx->SetCommContext(comm_context);
        if (op_name.compare(paddle::dialect::ReduceScatterOp::name()) == 0 ||
            op_name.compare(paddle::dialect::AllReduceOp::name()) == 0 ||
            op_name.compare(paddle::dialect::AllReduce_Op::name()) == 0 ||
            op_name.compare(paddle::dialect::Broadcast_Op::name()) == 0 ||
            op_name.compare(paddle::dialect::BroadcastOp::name()) == 0 ||
            op_name.compare(paddle::dialect::AllGatherOp::name()) == 0 ||
            op_name.compare(paddle::dialect::MpAllreduceSum_Op::name()) == 0 ||
            op_name.compare(paddle::dialect::CIdentity_Op::name()) == 0 ||
            op_name.compare(paddle::dialect::CConcatOp::name()) == 0 ||
            op_name.compare(paddle::dialect::AllGatherOp::name()) == 0 ||
            op_name.compare(paddle::dialect::AllToAllOp::name()) == 0 ||
            op_name.compare(
                paddle::dialect::CSoftmaxWithCrossEntropyOp::name()) == 0) {
          if (phi::is_gpu_place(place) && execution_stream == kDefaultStream) {
            if (origin_dev_ctx != nullptr) {
              // set stream
              auto default_stream =
                  static_cast<phi::GPUContext*>(origin_dev_ctx)->cuda_stream();
              static_cast<phi::GPUContext*>(dev_ctx)->SetCUDAStream(
                  default_stream, false);
              // set allocator
              auto& instance =
                  paddle::memory::allocation::AllocatorFacade::Instance();
              dev_ctx->SetAllocator(
                  instance
                      .GetAllocator(
                          place,
                          static_cast<phi::GPUContext*>(dev_ctx)->stream())
                      .get());
            } else {
              VLOG(3) << "op " << op_name << " ring_id " << ring_id
                      << " origin_dev_ctx is nullptr";
            }
          }
        }
        return dev_ctx;
      } else {
        VLOG(3) << "ring_id " << ring_id
                << " not found in comm_context_manager for op " << op_name;
      }
    }
#endif
  }

  if (origin_dev_ctx != nullptr) {
    interpreter::SetDeviceCommContext(op, origin_dev_ctx);
  }
  return origin_dev_ctx;
}
OpFuncType AnalyseOpFuncType(pir::Operation* op, const phi::Place& place) {
  if (phi::is_cpu_place(place)) {
    return OpFuncType::kCpuSync;
  }

  PADDLE_ENFORCE_EQ(
      interpreter::IsSupportedHeterPlace(place),
      true,
      common::errors::Fatal("Unsupported current place %s", place));

  auto& op_attributes = op->attributes();

  if ((op->dialect()->name() == paddle::dialect::KernelDialect::name()) &&
      (op_attributes.count("kernel_key") > 0)) {
    auto kernel_key = op_attributes.at("kernel_key")
                          .dyn_cast<dialect::KernelAttribute>()
                          .data();
    if (phi::TransToPhiPlace(kernel_key.backend()).GetType() ==
        phi::AllocationType::CPU) {
      return OpFuncType::kCpuSync;
    }
  }

  // Some GPU OPs do not launch CUDA Kernel, but spend a lot of time on CPU
  // computing. They execute serially in device thread and block CUDA kernel
  // launching in other GPU OPs. To improve performance, set them as kGpuSync
  // and so that they would be dispatched to host thread.
  if ((op->dialect()->name() == "pd_kernel") &&
      (op_attributes.count("op_name") > 0)) {
    auto op_name =
        op_attributes.at("op_name").dyn_cast<pir::StrAttribute>().AsString();
    if (op_name == "pd_op.coalesce_tensor" &&
        (!phi::is_xpu_place(place) ||
         op->attribute<pir::BoolAttribute>("persist_output").data() == false) &&
        op->attribute<pir::BoolAttribute>("set_constant").data() == false &&
        op->attribute<pir::BoolAttribute>("copy_data").data() == false) {
      return OpFuncType::kGpuSync;
    }

    if (phi::is_gpu_place(place) && (op_name == "pd_op.memcpy_d2h" ||
                                     op_name == "pd_op.memcpy_d2h_multi_io")) {
      return OpFuncType::kGpuSync;
    }

    if (op_name.compare(paddle::dialect::ShapeOp::name()) == 0 ||
        op_name.compare(paddle::dialect::Shape64Op::name()) == 0) {
      return OpFuncType::kGpuSync;
    }
  }

  if (auto combine_op = op->dyn_cast<pir::CombineOp>()) {
    for (size_t i = 0; i < combine_op.num_operands(); ++i) {
      if (auto combine_operand_type =
              combine_op.operand_source(i)
                  .type()
                  .dyn_cast<paddle::dialect::AllocatedDenseTensorType>()) {
        if (phi::is_cpu_place(combine_operand_type.place())) {
          return OpFuncType::kCpuSync;
        }
      }
    }
  }

  return OpFuncType::kGpuAsync;
}

void GetInputIds(pir::Operation* op,
                 const ValueExecutionInfo& value_exec_info,
                 std::unordered_map<pir::Value, std::vector<int>>* input_ids) {
  for (size_t i = 0; i < op->num_operands(); i++) {
    pir::Value value = op->operand_source(i);
    if (value && value.type()) {
      PADDLE_ENFORCE_EQ(
          value_exec_info.HasValue(value),
          true,
          common::errors::PreconditionNotMet(
              "input should in name map, [%d] 'th input of [%s] op",
              i,
              "if op"));
      input_ids->emplace(value, GetValueIds(value, value_exec_info));
    }
  }
}

std::unordered_set<pir::Value> GetInternalOutputs(pir::Block* block) {
  std::unordered_set<pir::Value> inner_outputs;
  for (size_t arg_id = 0; arg_id < block->args_size(); ++arg_id) {
    inner_outputs.insert(block->arg(arg_id));
  }
  for (auto& op : *block) {
    std::string op_name = op.name();
    if (op.attributes().count("op_name")) {
      op_name = op.attributes()
                    .at("op_name")
                    .dyn_cast<pir::StrAttribute>()
                    .AsString();
    }
    VLOG(8) << "GetInternalOutputs of " << op_name;
    if (op.num_regions()) {
      for (size_t i = 0; i < op.num_regions(); ++i) {
        for (auto& sub_block : op.region(i)) {
          std::unordered_set<pir::Value> sub_set =
              GetInternalOutputs(&sub_block);
          inner_outputs.insert(sub_set.begin(), sub_set.end());
        }
      }
    }

    for (size_t i = 0; i < op.num_results(); ++i) {
      inner_outputs.insert(op.result(i));
      VLOG(10) << op_name << "'s inner_output: " << op.result(i).impl();
    }
  }
  return inner_outputs;
}

std::unordered_set<pir::Value> GetInternalInputs(pir::Block* block) {
  std::unordered_set<pir::Value> inner_inputs;
  for (auto& op : *block) {
    std::string op_name = op.name();
    if (op.attributes().count("op_name")) {
      op_name = op.attributes()
                    .at("op_name")
                    .dyn_cast<pir::StrAttribute>()
                    .AsString();
    }
    VLOG(8) << "GetInternalInputs of " << op_name;
    if (op.num_regions()) {
      for (size_t i = 0; i < op.num_regions(); ++i) {
        for (auto& sub_block : op.region(i)) {
          std::unordered_set<pir::Value> sub_set =
              GetInternalInputs(&sub_block);
          inner_inputs.insert(sub_set.begin(), sub_set.end());
        }
      }
    }
    if (op.isa<pir::TuplePopOp>()) {
      auto tuple_pop_op = op.dyn_cast<pir::TuplePopOp>();
      if (tuple_pop_op.has_container()) {
        inner_inputs.insert(tuple_pop_op.container());
      }
    }
    for (size_t i = 0; i < op.num_operands(); ++i) {
      inner_inputs.insert(op.operand_source(i));
      VLOG(10) << op_name << "'s inner_input: " << op.operand_source(i).impl();
    }
  }
  return inner_inputs;
}

std::vector<pir::Value> GetExternalInputs(
    pir::Block* block,
    const ValueExecutionInfo& value_exec_info,
    std::unordered_map<pir::Value, std::vector<int>>* input_ids) {
  std::unordered_set<pir::Value> inner_outputs;
  inner_outputs = GetInternalOutputs(block);

  std::unordered_set<pir::Value> inner_inputs;
  inner_inputs = GetInternalInputs(block);

  std::vector<pir::Value> outside_op_inputs;
  for (pir::Value value : inner_inputs) {
    if (value && (!inner_outputs.count(value))) {
      PADDLE_ENFORCE_EQ(value_exec_info.HasValue(value),
                        true,
                        common::errors::PreconditionNotMet(
                            "input %s should be in name map", value.impl()));
      input_ids->emplace(value, GetValueIds(value, value_exec_info));
      outside_op_inputs.push_back(value);
      VLOG(6) << "GetExternalInputs of " << value.impl();
    }
  }
  return outside_op_inputs;
}

std::unordered_set<pir::Value> GetTuplePushContainer(pir::Block* block) {
  std::unordered_set<pir::Value> inner_outputs;
  for (auto& op : *block) {
    VLOG(8) << "GetTuplePushContainer of " << op.name();
    if (op.num_regions()) {
      for (size_t i = 0; i < op.num_regions(); ++i) {
        for (auto& sub_block : op.region(i)) {
          std::unordered_set<pir::Value> sub_set =
              GetTuplePushContainer(&sub_block);
          inner_outputs.insert(sub_set.begin(), sub_set.end());
        }
      }
    }
    if (op.isa<pir::TuplePushOp>()) {
      auto tuple_push_op = op.dyn_cast<pir::TuplePushOp>();
      inner_outputs.insert(tuple_push_op.container());
    }
  }
  return inner_outputs;
}

void InsertTuplePushContainerToOuts(
    pir::Block* block,
    const ValueExecutionInfo& value_exec_info,
    std::unordered_map<pir::Value, std::vector<int>>* outputs) {
  std::unordered_set<pir::Value> inner_stack_outputs;
  inner_stack_outputs = GetTuplePushContainer(block);

  for (pir::Value value : inner_stack_outputs) {
    outputs->emplace(value, GetValueIds(value, value_exec_info));
    VLOG(6) << "InsertTuplePushContainerToOuts of " << value.impl();
  }
}

void InsertInplacedExternalInputsToOuts(
    pir::Block* block,
    const std::vector<pir::Value>& external_inputs,
    const ValueExecutionInfo& value_exec_info,
    std::unordered_map<pir::Value, std::vector<int>>* outputs) {
  for (auto& op : *block) {
    if (op.attributes().count("is_inplace") != 0 &&
        op.attributes()
            .at("is_inplace")
            .dyn_cast<pir::BoolAttribute>()
            .data()) {
      std::string op_name = op.name();
      if (op.attributes().count("op_name")) {
        op_name = op.attributes()
                      .at("op_name")
                      .dyn_cast<pir::StrAttribute>()
                      .AsString();
      }
      pir::OpInfo op_info =
          pir::IrContext::Instance()->GetRegisteredOpInfo(op_name);
      paddle::dialect::OpYamlInfoParser yaml_parser(
          op_info.GetInterfaceImpl<paddle::dialect::OpYamlInfoInterface>()
              ->get_op_info_(op_name),
          paddle::dialect::IsLegacyOp(op_name));

      for (size_t i = 0; i < op.num_results(); ++i) {
        pir::Value value = op.result(i);
        if (!IsInvalid(value)) {
          VLOG(8) << "Number " << i << " result of " << op_name
                  << " is not invalid, so skip build a variable.";
          continue;
        }
        std::string value_name = yaml_parser.OutputNames()[i];
        if (yaml_parser.HasInplace(value_name)) {
          const std::string& inplace_name = yaml_parser.InplaceName(value_name);
          pir::Value inplace_value =
              op.operand_source(yaml_parser.InputName2Id().at(inplace_name));
          if (std::find(external_inputs.begin(),
                        external_inputs.end(),
                        inplace_value) != external_inputs.end()) {
            outputs->emplace(value,
                             GetValueIds(inplace_value, value_exec_info));
          }
        }
      }
    }
  }
}

bool GetCondData(const phi::DenseTensor& cond) {
  if (phi::is_cpu_place(cond.place())) {
    return cond.data<bool>()[0];
  }
  // when phi::is_gpu_place(cond.place()) or
  // phi::is_xpu_place(cond.place()) is true
  std::unique_ptr<phi::DenseTensor> cpu_cond{new phi::DenseTensor()};
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP) || \
    defined(PADDLE_WITH_XPU) || defined(PADDLE_WITH_CUSTOM_DEVICE)
  paddle::framework::TensorCopySync(cond, phi::CPUPlace(), cpu_cond.get());
#else
  PADDLE_THROW(common::errors::PreconditionNotMet(
      "This version of PaddlePaddle does NOT support GPU/XPU but got "
      "GPU/XPU tensor Cond in WhileOp. Please compile WITH_GPU or "
      "WITH_XPU option."));
#endif
  return cpu_cond->data<bool>()[0];
}

// NOTE(chenxi67): Here, we only perform inplace processing for variables whose
// type is NOT TensorArray. It has already been processed in the previous
// step(HandleForInplaceVarOp).
void HandleForInplaceOp(pir::Operation* op,
                        const ValueExecutionInfo* value_exe_info,
                        InstructionBase* instr) {
  if (op->num_results() < 1) return;
  pir::IrContext* ctx = pir::IrContext::Instance();
  std::string op_name = op->name();
  if (op->attributes().count("op_name")) {
    op_name =
        op->attributes().at("op_name").dyn_cast<pir::StrAttribute>().AsString();
  }

  pir::OpInfo op_info = ctx->GetRegisteredOpInfo(op_name);
  paddle::dialect::OpYamlInfoParser yaml_parser(
      op_info.GetInterfaceImpl<paddle::dialect::OpYamlInfoInterface>()
          ->get_op_info_(op_name),
      paddle::dialect::IsLegacyOp(op_name));

  for (size_t i = 0; i < op->num_results(); ++i) {
    pir::Value value = op->result(i);
    if (!IsInvalid(value)) {
      VLOG(8) << "Number " << i << " result of " << op_name
              << " is not invalid, so skip build a variable.";
      continue;
    }
    if (IsNeedVarInplace(op, value, op_name)) {
      continue;
    }
    std::string value_name = yaml_parser.OutputNames()[i];
    if (yaml_parser.HasInplace(value_name)) {
      const std::string& inplace_name = yaml_parser.InplaceName(value_name);
      pir::Value inplace_value =
          op->operand_source(yaml_parser.InputName2Id().at(inplace_name));
      std::string input_var_name = value_exe_info->GetVarName(inplace_value);
      std::string output_var_name = value_exe_info->GetVarName(value);
      PADDLE_ENFORCE_NE(input_var_name,
                        "",
                        common::errors::InvalidArgument(
                            "The input var name of inplace op is empty."));
      PADDLE_ENFORCE_NE(output_var_name,
                        "",
                        common::errors::InvalidArgument(
                            "The output var name of inplace op is empty."));
      VLOG(4) << "inplace: " << value_name << " -> " << inplace_name
              << " (var: " << input_var_name << ")";
      instr->AddInplace(value_exe_info->GetVarByValue(inplace_value),
                        value_exe_info->GetVarByValue(value));
    } else if (yaml_parser.HasView(value_name)) {
      const std::string& view_name = yaml_parser.ViewName(value_name);
      pir::Value view_value =
          op->operand_source(yaml_parser.InputName2Id().at(view_name));
      // const std::string& var_name = value_2_var_name->at(view_value);
      std::string input_var_name = value_exe_info->GetVarName(view_value);
      std::string output_var_name = value_exe_info->GetVarName(value);

      PADDLE_ENFORCE_NE(input_var_name,
                        "",
                        common::errors::InvalidArgument(
                            "The input var name of view op is empty."));
      PADDLE_ENFORCE_NE(output_var_name,
                        "",
                        common::errors::InvalidArgument(
                            "The output var name of view op is empty."));
      VLOG(4) << "view: " << value_name << " -> " << view_name
              << " (var: " << input_var_name << ")";
      instr->AddInplace(value_exe_info->GetVarByValue(view_value),
                        value_exe_info->GetVarByValue(value));
    }
  }
}

void ShareVarBuffer(const Variable* src_var, Variable* dst_var) {
  if (src_var->IsType<phi::DenseTensor>()) {
    auto& src_tensor = src_var->Get<phi::DenseTensor>();
    auto* tmp_dst_tensor = dst_var->GetMutable<phi::DenseTensor>();
    tmp_dst_tensor->ShareBufferWith(src_tensor);
    return;
  } else if (src_var->IsType<phi::SelectedRows>()) {
    auto* tmp_dst_slr = dst_var->GetMutable<phi::SelectedRows>();
    auto* dst_t = tmp_dst_slr->mutable_value();
    auto& src_slr = src_var->Get<phi::SelectedRows>();
    auto& src_t = src_slr.value();
    dst_t->ShareBufferWith(src_t);
    return;
  } else if (src_var->IsType<VariableRefArray>()) {
    auto src_var_array = src_var->Get<VariableRefArray>();
    auto* dst_var_array = dst_var->GetMutable<VariableRefArray>();
    for (size_t i = 0; i < src_var_array.size(); ++i) {
      Variable* copy_var = const_cast<Variable*>(dst_var_array->at(i));
      ShareVarBuffer(src_var_array.at(i), copy_var);
    }
    return;
  } else {
    PADDLE_THROW(common::errors::PreconditionNotMet(
        "Output only support DenseTensorType "
        "or SelectedRowsType or VariableRefArray"));
  }
  return;
}

}  // namespace paddle::framework
