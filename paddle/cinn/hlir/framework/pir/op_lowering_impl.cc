// Copyright (c) 2022 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/hlir/framework/pir/op_lowering_impl.h"

#include <string>

#include "paddle/cinn/adt/map_expr_ctx.h"
#include "paddle/cinn/ast_gen_ius/tensor_group.h"
#include "paddle/cinn/backends/codegen_device_util.h"
#include "paddle/cinn/common/dim_expr_converter.h"
#include "paddle/cinn/common/target.h"
#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/group_merge/op_with_group_merge_util.h"
#include "paddle/cinn/hlir/framework/compile_error.h"
#include "paddle/cinn/hlir/framework/pir/op_lowering_util.h"
#include "paddle/cinn/hlir/framework/pir/trivial_op_impl.h"
#include "paddle/cinn/hlir/framework/pir/utils.h"
#include "paddle/cinn/hlir/pe/map_expr_to_ir.h"
#include "paddle/cinn/ir/dim.h"
#include "paddle/cinn/ir/group_schedule/base_group_scheduler.h"
#include "paddle/cinn/ir/group_schedule/config/group_tile_config.h"
#include "paddle/cinn/ir/ir_analyzer/ir_analyzer.h"
#include "paddle/cinn/ir/schedule/ir_schedule.h"
#include "paddle/cinn/ir/utils/stmt_converter.h"
#include "paddle/cinn/lang/placeholder.h"
#include "paddle/cinn/operator_fusion/fusion_interface.h"
#include "paddle/cinn/optim/check_tensor_buffer_map.h"
#include "paddle/cinn/optim/eliminate_common_global_memory_read.h"
#include "paddle/cinn/optim/longlong2int_pass.h"
#include "paddle/cinn/optim/schedule_block_dce.h"
#include "paddle/cinn/optim/transform_gpu_forloop.h"
#include "paddle/common/ddim.h"
#include "paddle/common/enforce.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_op.h"
#include "paddle/pir/include/dialect/shape/utils/shape_analysis.h"

PD_DECLARE_bool(cinn_use_cuda_vectorize);
PD_DECLARE_bool(cinn_check_tensor_buffer_map);
PD_DECLARE_bool(cinn_longlong2int);
const int default_priority = 100;

namespace cinn {
namespace hlir {
namespace framework {
namespace pir {

using cinn::common::Type;
using framework::OpPatternKind;
using framework::StrategyFunction;

namespace details {

NodeAttr CollectAttrs(const ::pir::Operation& op) {
  NodeAttr node_attrs;
  VLOG(4) << "op.attributes():" << op.attributes().size();
  auto attrs = CompatibleInfo::ConvertAttributes(op);
  node_attrs.node_name = CompatibleInfo::OpName(op);
  node_attrs.attr_store = std::move(attrs);

  return node_attrs;
}

std::optional<std::vector<ir::Expr>> GetTensorValueFromShapeOrData(
    const symbol::ShapeOrDataDimExprs& shape_or_data) {
  if (!shape_or_data.data()) return std::nullopt;
  std::vector<ir::Expr> result;
  result.reserve(shape_or_data.data()->size());
  for (const auto& data : *shape_or_data.data()) {
    result.push_back(common::DimExprConverter().ConvertToIrExpr(data));
  }
  return result;
}

}  // namespace details

OpLowererImpl::OpLowererImpl(const Target& target) : target_(target) {
  name_gene_ = new PrettyNamer();
}

BucketLoweredFuncsWrapper OpLowererImpl::BucketLower(
    const OpLoweringGroupPtr& group) {
  VLOG(4) << "BucketLower Group : \n" << *group;
  // 1.Do compute, lower and schedule for each op.
  const auto& ops = group->ops();
  auto X86Expr = LowerX86(group, ops, false);
  VLOG(3) << "After x86 lower, ir is: \n" << X86Expr;

  std::vector<ir::Tensor> group_func_arg_tensors;
  std::unordered_map<::pir::Value, ir::Tensor> tensor_map;
  // for some op, it will output more tmp value and regard as
  // XX_0, XX_1, so we log them in tmp_tensor_info;
  std::vector<ir::stmt::BlockRef> func_body_blocks =
      LowerOps(group, ops, &group_func_arg_tensors, &tensor_map);
  // TODO(Hongqing-work): delete this convert after using new IR update
  // schedule.
  std::vector<ir::Expr> func_bodies;
  for (const auto& body : func_body_blocks) {
    func_bodies.emplace_back(ir::ConvertStmtBlockToExprBlock(body));
  }
  if (FLAGS_cinn_check_tensor_buffer_map) {
    optim::CheckTensorBufferMap(func_bodies, "BucketLower LowerOps");
    VLOG(3) << "LowerOps tensor-buffer map check succeed";
  }

  // =========== OpFusion ============

  // VLOG(4) << "Bucket Lower output values is : " << group->output_values();
  func_bodies = OperationFusion(ops, func_bodies, group->fusion_tracker_ptr);

  std::unordered_set<std::string> fusion_group_args;
  for (auto value : group->GetInputOpValues()) {
    fusion_group_args.insert(ValueName(value));
  }

  for (auto value : group->GetGroupOutputValues()) {
    fusion_group_args.insert(ValueName(value));
  }

  std::shared_ptr<FusionGroupInfo> fusion_group_info =
      GetFusionGroupInfo(func_bodies, fusion_group_args);
  // TODO(liangshuhao): grid reduce is disabled for broadcast-leaf group now,
  // because grid reduce introduces extra func args that currently cannot be
  // unified with other broadcast-leaf groups.
  if (group->IsBroadcastLeaf()) {
    fusion_group_info->can_apply_grid_reduce = false;
  }

  if (FLAGS_cinn_check_tensor_buffer_map) {
    optim::CheckTensorBufferMap(func_bodies, "BucketLower OpFusion");
    VLOG(3) << "OpFusion tensor-buffer map check succeed";
  }

  // =========== CodeGen And Optimizer ================

  // 2.Do group schedule.
  ir::ModuleExpr mod_expr(func_bodies);
  ir::IRSchedule ir_sch(
      mod_expr, -1, false, cinn::utils::ErrorMessageLevel::kGeneral, true);
  ir_sch.MergeExprs();
  std::vector<std::pair<ir::SymbolicPredicate, ir::Expr>> cond2func_bodies;
  std::vector<int> priorities;
  VLOG(3) << "After lower, ir is: \n" << ir_sch.GetModule().GetExprs().at(0);

  if (FLAGS_cinn_check_tensor_buffer_map) {
    optim::CheckTensorBufferMap(ir_sch.GetModule().GetExprs(),
                                "BucketLower MergeExprs");
    VLOG(3) << "MergeExprs tensor-buffer map check succeed";
  }

  std::unordered_set<std::string> output_tensor_names;
  for (auto value : group->GetGroupOutputValues()) {
    output_tensor_names.insert(ValueName(value));
  }

  std::unique_ptr<ir::GroupScheduler> group_scheduler =
      ir::GroupScheduler::Make(&ir_sch,
                               output_tensor_names,
                               target_,
                               /* is_dy_shape = */ true,
                               fusion_group_info);

  VLOG(4) << "Start apply group_scheduler->Schedule()";
  group_scheduler->Schedule();
  VLOG(4) << "End   apply group_scheduler->Schedule()";

  cond2func_bodies = group_scheduler->GetIRs();
  VLOG(4) << "End   group_scheduler->GetIRs";

  priorities = group_scheduler->GetPriorities();
  VLOG(4) << "End group_scheduler->GetPriorities";

  // The last func is stored as a kernel on x86
  cond2func_bodies.emplace_back(ir::Expr(true), X86Expr);

  if (FLAGS_cinn_check_tensor_buffer_map) {
    for (std::pair<ir::SymbolicPredicate, ir::Expr>& cond2body :
         cond2func_bodies) {
      optim::CheckTensorBufferMap(cond2body.second, "BucketLower schedule");
    }
    VLOG(3) << "Schedule tensor-buffer map check succeed";
  }

  // 3.Do post-processing,
  // including preparing function args and temporary variables,
  // applying low-level optimization passes, etc.
  std::vector<ir::Expr> scheduled_func_bodies;
  std::vector<ir::SymbolicPredicate> predicates;
  for (std::pair<ir::SymbolicPredicate, ir::Expr>& cond2body :
       cond2func_bodies) {
    predicates.push_back(cond2body.first);
    scheduled_func_bodies.push_back(cond2body.second);
  }
  std::vector<ir::Tensor> group_func_arg_tensors_copy = group_func_arg_tensors;
  std::vector<ir::Argument> group_func_args;
  std::vector<ir::Tensor> infer_shape_tensor_args;

  std::vector<CondFuncPriorWrapper> warps_processed =
      PostProcess(group,
                  tensor_map,
                  fusion_group_info,
                  {scheduled_func_bodies},
                  {predicates},
                  {priorities},
                  &group_func_arg_tensors_copy,
                  &group_func_args,
                  &infer_shape_tensor_args);
  if (FLAGS_cinn_check_tensor_buffer_map) {
    for (auto& warp : warps_processed) {
      optim::CheckTensorBufferMap(std::get<1>(warp)->body,
                                  "BucketLower PostProcess");
    }
    VLOG(3) << "PostProcess tensor-buffer map check succeed";
  }

  BucketLoweredFuncsWrapper funcs_wrapper;
  for (int i = 0; i < warps_processed.size() - 1; ++i) {
    funcs_wrapper.predicate2funcs.emplace_back(warps_processed[i]);
  }

  // The last func is x86 kernel.
  auto [predicate_postprocessed, func_postprocessed, _] =
      warps_processed[warps_processed.size() - 1];
  if (func_postprocessed->body != ir::Expr(-1)) {
    func_postprocessed->name = func_postprocessed->name + "_CX86";
    funcs_wrapper.predicate2funcsCX86.emplace_back(predicate_postprocessed,
                                                   func_postprocessed);
  }

  funcs_wrapper.infer_shape_func =
      GenerateInferShapeFunc(group, infer_shape_tensor_args, group_func_args);

  VLOG(4) << "End This function.";
  return funcs_wrapper;
}

std::unordered_set<std::string> CollectStoreBufferNames(
    const std::vector<ir::Expr>& func_bodies) {
  std::unordered_set<std::string> buffer_names;
  std::vector<ir::Expr> blocks = ir::analyzer::GetAllBlocks(func_bodies);
  for (const ir::Expr& block : blocks) {
    ir::Tensor tensor = ir::analyzer::GetStoreTensorOfSBlock(block);
    if (tensor->buffer.defined()) {
      buffer_names.insert(tensor->buffer->name);
    }
  }
  return buffer_names;
}

std::vector<CondFuncPriorWrapper> OpLowererImpl::PostProcess(
    const OpLoweringGroupPtr& group,
    const std::unordered_map<::pir::Value, ir::Tensor>& tensor_map,
    const std::shared_ptr<FusionGroupInfo>& fusion_group_info,
    std::vector<ir::Expr> func_bodies,
    std::vector<ir::SymbolicPredicate> predicates,
    std::vector<int> priorities,
    std::vector<ir::Tensor>* group_func_arg_tensors,
    std::vector<ir::Argument>* group_func_args,
    std::vector<ir::Tensor>* infer_shape_arg_tensor) {
  // 1.Prepare function args
  group->mut_input_names().clear();
  std::unordered_set<std::string> store_buffer_names =
      CollectStoreBufferNames(func_bodies);
  std::unordered_set<std::string> arg_name_set;
  const int& input_tensor_size = group_func_arg_tensors->size();
  for (auto& arg_tensor : *group_func_arg_tensors) {
    // input data name.
    group->mut_input_names().push_back(arg_tensor->name);
    // args
    ir::Argument::IO io_type =
        store_buffer_names.count(arg_tensor->buffer->name) > 0
            ? ir::Argument::IO::kOutput
            : ir::Argument::IO::kInput;
    (*group_func_args).emplace_back(arg_tensor->buffer, io_type);
    arg_name_set.insert(arg_tensor->buffer->name);
  }

  group->mut_output_names().clear();

  // collect all output tensor.
  for (auto op_result : group->output_values()) {
    if (tensor_map.count(op_result) == 0) {
      continue;
    }
    auto tensor = tensor_map.at(op_result);
    bool contain_unknown_dim = [&]() {
      bool check = op_result && op_result.type() &&
                   op_result.type().isa<paddle::dialect::DenseTensorType>();
      PADDLE_ENFORCE_EQ(
          check,
          true,
          phi::errors::PreconditionNotMet("cinn only support DenseTensorType"));
      const auto dims =
          op_result.type().dyn_cast<paddle::dialect::DenseTensorType>().dims();
      return ::common::contain_unknown_dim(dims);
    }();

    if (contain_unknown_dim && group->HasShapeOrDataExprs(op_result)) {
      tensor->shape.clear();
      for (size_t i = 0;
           i < group->GetShapeOrDataExprs(op_result).shape().size();
           ++i) {
        ir::Dim t(tensor->name,
                  group->GetShapeOrDataExprs(op_result).shape()[i]);
        tensor->shape.push_back(t->dim_expr);
      }
    }
    infer_shape_arg_tensor->push_back(tensor);

    if (arg_name_set.count(tensor->buffer->name) != 0) {
      continue;
    }

    // output arg tensors
    group_func_arg_tensors->push_back(tensor);
    // output args
    group->mut_output_names().push_back(tensor->name);
    (*group_func_args).emplace_back(tensor->buffer, ir::Argument::IO::kOutput);
    arg_name_set.insert(tensor->buffer->name);
  }

  std::map<int, CINNKernelInfo::SymbolArgBindInfo> mps;
  // update args for dynamic dim
  int non_tensor_arg_idx = group_func_args->size();

  std::vector<ir::Argument> group_func_args_int32;
  for (const auto& arg : *group_func_args) {
    group_func_args_int32.emplace_back(arg);
  }

  std::unordered_set<std::string> symbol_args_set;
  for (int tensor_arg_idx = 0; tensor_arg_idx < input_tensor_size;
       tensor_arg_idx++) {
    const auto& AddDimSymbolArgs = [&]() {
      const auto& tensor_dim =
          (*group_func_arg_tensors).at(tensor_arg_idx)->sym_shape;
      int tensor_dim_size = tensor_dim.size();
      for (int tensor_arg_dim_idx = 0; tensor_arg_dim_idx < tensor_dim_size;
           tensor_arg_dim_idx++) {
        if (tensor_dim[tensor_arg_dim_idx]->IsUniSymbolic()) {
          const std::string& symbol_name =
              tensor_dim[tensor_arg_dim_idx]->ToString();
          if (symbol_args_set.count(symbol_name) != 0) {
            continue;
          }
          symbol_args_set.insert(symbol_name);
          group_func_args->emplace_back(
              ir::_Var_::Make(symbol_name, cinn::common::Int(64)));
          group_func_args_int32.emplace_back(
              ir::_Var_::Make(symbol_name, cinn::common::Int(32)));
          group->mut_symbol_args_map()[non_tensor_arg_idx++] =
              CINNKernelInfo::ArgDimIdx{tensor_arg_idx, tensor_arg_dim_idx};
          VLOG(4) << "device kernel func's " << symbol_name << " is from "
                  << tensor_arg_idx << ".shape(" << tensor_arg_dim_idx << ")";
        }
      }
    };
    const auto& AddValueSymbolArgs = [&]() {
      const auto& opt_tensor_value =
          (*group_func_arg_tensors).at(tensor_arg_idx)->value();
      if (opt_tensor_value) {
        const int& tensor_value_size = opt_tensor_value.value().size();
        for (int value_idx = 0; value_idx < tensor_value_size; ++value_idx) {
          const auto& value_expr = opt_tensor_value.value()[value_idx];
          if (value_expr.is_var()) {
            const std::string& symbol_name = value_expr.as_var()->name;
            if (symbol_args_set.count(symbol_name) != 0) {
              continue;
            }
            symbol_args_set.insert(symbol_name);
            group_func_args->emplace_back(
                ir::_Var_::Make(symbol_name, cinn::common::Int(64)));
            group_func_args_int32.emplace_back(
                ir::_Var_::Make(symbol_name, cinn::common::Int(32)));
            group->mut_symbol_args_map()[non_tensor_arg_idx++] =
                CINNKernelInfo::ArgValueIdx{tensor_arg_idx, value_idx};
            VLOG(4) << "device kernel func's " << symbol_name << " is from "
                    << tensor_arg_idx << ".data(" << value_idx << ")";
          }
        }
      }
    };
    AddDimSymbolArgs();
    AddValueSymbolArgs();
  }

  std::vector<ir::LoweredFunc> ret_lowered_funcs;
  std::vector<ir::SymbolicPredicate> ret_predicates;
  std::vector<int> ret_priorities;

  for (int i = 0; i < func_bodies.size(); ++i) {
    ir::Expr func_body = func_bodies[i];
    optim::EliminateDeadScheduleBlock(&(func_body), group->output_names());
    if (i != func_bodies.size() - 1) {
      cinn::common::DefaultDeviceTarget().arch.Match(
          [&](std::variant<common::UnknownArch,
                           common::X86Arch,
                           common::ARMArch>) {},
          [&](common::NVGPUArch) {
#ifdef CINN_WITH_CUDA
            // optim::EliminateCommonGlobalMemoryRead(&(func_body));
            optim::OptimizeExprGPU(&(func_body));
#endif
          },
          [&](std::variant<common::HygonDCUArchHIP, common::HygonDCUArchSYCL>) {
            // optim::EliminateCommonGlobalMemoryRead(&(func_body));
            optim::OptimizeExprGPU(&(func_body));
          });
    }

    // 2.Prepare temp buffers
    auto temp_buffers =
        lang::GetTempBuffers(*group_func_arg_tensors, func_body);
    // 3.Building LoweredFunc
    auto func = ir::_LoweredFunc_::Make(
        group->FuncName(), *group_func_args, func_body, temp_buffers);

    // 4.Apply low level pass
    if (i != func_bodies.size() - 1) {
      func = optim::Optimize(func, target_, false);
    } else {
      func = optim::Optimize(func, common::DefaultHostTarget(), false);
    }
    func->num_output_tensors = infer_shape_arg_tensor->size();

    if (FLAGS_cinn_longlong2int && i != func_bodies.size() - 1) {
      if (fusion_group_info->is_dynamic) {
        ir::LoweredFunc func_copied = ir::ir_utils::IRCopy(func);
        // set lowered_func's args to int32 type
        func_copied->args = group_func_args_int32;
        ir::Expr predicates_copied = ir::ir_utils::IRCopy(predicates[i]);

        ir::Expr elems_num(1);
        for (const auto& loop : fusion_group_info->loop_ranges_expr) {
          elems_num = ir::Mul::Make(elems_num, loop);
        }
        VLOG(10) << "elems_num in dynamic branch: " << elems_num;

        // Deal with predicates, set original predicates to range int64
        ir::Expr predicate_int32 = ir::And::Make(
            predicates_copied, ir::LT::Make(elems_num, ir::Expr(INT32_MAX)));
        predicates[i] = ir::And::Make(
            predicates[i], ir::GE::Make(elems_num, ir::Expr(INT32_MAX)));

        ir::stmt::BlockRef block =
            ir::ConvertExprBlockToStmtBlock(func_copied->body);
        VLOG(10) << "Before CastLonglong2Int: \n" << block;

        optim::TryCastLonglong2Int(block, /*enforce_cast*/ true);
        VLOG(10) << "After CastLonglong2Int: \n" << block;
        func_copied->body = ir::ConvertStmtBlockToExprBlock(block);

        ret_predicates.push_back(std::move(predicate_int32));
        ret_lowered_funcs.push_back(std::move(func_copied));
        ret_priorities.push_back(priorities[i]);
      } else {
        // static branch
        ir::stmt::BlockRef block = ir::ConvertExprBlockToStmtBlock(func->body);

        int64_t elems_num = 1;
        for (const auto& loop : fusion_group_info->loop_ranges) {
          elems_num *= loop;
        }
        VLOG(10) << "elems_num in static branch: " << elems_num;
        VLOG(10) << "Before CastLonglong2Int: \n" << block;
        optim::TryCastLonglong2Int(block,
                                   /*enforce_cast*/ elems_num < INT32_MAX);
        VLOG(10) << "After CastLonglong2Int: \n" << block;
        func->body = ir::ConvertStmtBlockToExprBlock(block);
      }
    }

    ret_predicates.push_back(std::move(predicates[i]));
    ret_lowered_funcs.push_back(std::move(func));
    // host func has no priority, since tuples require alignment, set -1 here.
    if (i != func_bodies.size() - 1) {
      ret_priorities.push_back(std::move(priorities[i]));
    } else {
      ret_priorities.push_back(-1);
    }
  }

  // 5. Unify temp_space args and set temp_space sizes
  UnifyTempSpaceArgs(&ret_lowered_funcs);
  group->mut_temp_space_sizes() = CollectTempSpaceSizes(ret_lowered_funcs);

  PADDLE_ENFORCE_EQ(
      ret_lowered_funcs.size(),
      ret_predicates.size(),
      ::common::errors::InvalidArgument(
          "The size of ret_lowered_funcs and ret_predicates should be "
          "the same."));
  PADDLE_ENFORCE_EQ(
      ret_lowered_funcs.size(),
      ret_priorities.size(),
      ::common::errors::InvalidArgument(
          "The size of ret_lowered_funcs and ret_priorities should be "
          "the same."));

  std::vector<CondFuncPriorWrapper> ret;
  for (size_t i = 0; i < ret_lowered_funcs.size(); ++i) {
    ret.emplace_back(std::move(ret_predicates[i]),
                     std::move(ret_lowered_funcs[i]),
                     std::move(ret_priorities[i]));
  }
  return ret;
}

std::vector<ir::stmt::BlockRef> OpLowererImpl::LowerOps(
    const OpLoweringGroupPtr& group,
    const std::vector<::pir::Operation*>& ops,
    std::vector<ir::Tensor>* group_func_arg_tensors,
    std::unordered_map<::pir::Value, ir::Tensor>* tensor_map) {
  auto& strategy = Operator::GetAttrs<StrategyFunction>("CINNStrategy");
  std::vector<ir::stmt::BlockRef> func_bodies;

  for (auto* op : ops) {
    VLOG(4) << "start lowering op:" << op->name() << " id: " << op->id();
    std::string cinn_op_name = CompatibleInfo::OpName(*op);

    VLOG(4) << "cinn op name " << cinn_op_name << std::endl;

    // 1.Select Op impl
    std::vector<ir::Tensor> op_func_arg_tensors =
        CollectInputTensor(group, op, group_func_arg_tensors, tensor_map);
    VLOG(4) << "input size:" << op_func_arg_tensors.size();

    const hlir::framework::Operator* cinn_op = Operator::Get(cinn_op_name);
    std::shared_ptr<OpImpl> op_impl = nullptr;

    std::vector<Type> out_types;
    std::vector<std::vector<ir::Dim>> out_shapes;
    CollectOutputInfo(op, &out_types, &out_shapes, group);

    PADDLE_ENFORCE_EQ(out_types.size(),
                      out_shapes.size(),
                      ::common::errors::InvalidArgument(
                          "The size of out_types and out_shapes should be "
                          "the same."));
    VLOG(4) << "out_types.size(): " << out_types.size();
    NodeAttr node_attrs = details::CollectAttrs(*op);
    auto& strategy_map =
        Operator::GetAttrs<StrategyFunctionSymbolic>("CINNStrategySymbolic");
    StrategyFunctionSymbolic strategy = strategy_map[cinn_op];
    PADDLE_ENFORCE_EQ(
        static_cast<bool>(strategy),
        true,
        ::common::errors::PreconditionNotMet(
            "cinn_op_name: %s has no CINNStrategySymbolic registered.",
            cinn_op_name));
    op_impl = OpStrategy::SelectImpl(strategy(
        node_attrs, op_func_arg_tensors, out_types, out_shapes, this->target_));

    // 2.Perform the lower process of Op
    std::vector<ir::LoweredFunc> funcs =
        DoOpLower(op_impl, op, tensor_map, &op_func_arg_tensors);

    for (const ir::LoweredFunc& func : funcs) {
      func_bodies.push_back(func->body_block);
    }
  }

  VLOG(4) << "group_func_arg_tensors.size(): "
          << group_func_arg_tensors->size();

  return func_bodies;
}

std::vector<ir::LoweredFunc> OpLowererImpl::DoOpLower(
    std::shared_ptr<hlir::framework::OpImpl> op_impl,
    ::pir::Operation* op,
    std::unordered_map<::pir::Value, ir::Tensor>* tensor_map,
    std::vector<ir::Tensor>* op_func_arg_tensors) {
  VLOG(4) << "Do lower with Compute, op: " << op->name();
  std::vector<cinn::common::CINNValue> cinn_inputs;
  for (const ir::Tensor& tensor : *op_func_arg_tensors) {
    cinn_inputs.push_back(cinn::common::CINNValue(ir::Expr(tensor)));
  }

  // set tensor name = operand hash name
  auto op_results = op->results();
  for (const auto& result : op_results) {
    std::string output_id = ValueName(result);
    cinn_inputs.push_back(cinn::common::CINNValue(output_id));
  }

  // 1.Do compute
  cinn::common::CINNValuePack pack =
      op_impl->fcompute(cinn::common::CINNValuePack{cinn_inputs});

  std::string post = "";
  std::vector<ir::Tensor> stage_tensors;
  for (int idx = 0; idx < pack.size(); ++idx) {
    Expr expr = pack[idx];
    // Insert the output tensor defined by Compute into the tensor_map
    if (pack.size() - 1 > op_results.size()) {
      // Some op may output multiple temp tensors in their Compute
      // definition, but only one output  in the graph, and we use id +
      // "_0"/"_1" as key.
      if (idx < op_results.size()) {
        (*tensor_map)[op_results[idx]] = expr.as_tensor_ref();
      }
      std::string tensor_name = ValueName(op_results[0]) + post;
      VLOG(3) << "Add tmp tensor name for reducer op: " << tensor_name;
      post = "_" + std::to_string(idx);
    } else {
      // If the number of output tensors defined by Compute is less equal than
      // the output node_data on the graph, then there is a one-to-one
      // correspondence, and the redundant output node_data contact empty.
      (*tensor_map)[op_results[idx]] = expr.as_tensor_ref();

      stage_tensors.push_back(expr.as_tensor_ref());
    }

    // Insert output tensors into function arg
    target_.arch.Match(
        [&](common::NVGPUArch) {
          if (!expr.as_tensor_ref()->buffer.defined()) {
            op_func_arg_tensors->push_back(expr.as_tensor_ref());
            expr.as_tensor_ref()->WithBuffer();
          } else {
            op_func_arg_tensors->push_back(expr.as_tensor_ref());
          }
        },
        [&](std::variant<common::UnknownArch,
                         common::X86Arch,
                         common::ARMArch>) {
          op_func_arg_tensors->push_back(expr.as_tensor_ref());
          expr.as_tensor_ref()->WithBuffer();
        },
        [&](std::variant<common::HygonDCUArchHIP, common::HygonDCUArchSYCL>) {
          if (!expr.as_tensor_ref()->buffer.defined()) {
            op_func_arg_tensors->push_back(expr.as_tensor_ref());
            expr.as_tensor_ref()->WithBuffer();
          } else {
            op_func_arg_tensors->push_back(expr.as_tensor_ref());
          }
        });
  }

  VLOG(4) << "op_func_arg_tensors.size(): " << op_func_arg_tensors->size();

  // 2.Do lower
  std::string lower_fn_name = CompatibleInfo::OpFuncName(*op);

  // using output value build tensor group
  ast_gen_ius::TensorGroup tensor_group(stage_tensors);
  std::vector<ir::LoweredFunc> funcs = lang::LowerToAstVec(
      lower_fn_name, *op_func_arg_tensors, {&tensor_group}, this->target_);
  VLOG(4) << "Lower op: " << lower_fn_name << ", get " << funcs.size()
          << " LoweredFunc:\n";
  if (VLOG_IS_ON(4)) {
    for (auto fun : funcs) {
      VLOG(4) << fun;
    }
  }

  return funcs;
}

ir::Tensor OpLowererImpl::GetTensor(const OpLoweringGroupPtr& group,
                                    const ::pir::Value& value) {
  auto type_info = value.type().dyn_cast<paddle::dialect::DenseTensorType>();
  auto dtype = type_info.dtype();
  std::string input_id = ValueName(value);

  auto ForEachDimExpr = [&](const auto& DoEach) {
    const auto& dims = type_info.dims();
    if (::common::contain_unknown_dim(dims)) {  // dynamic shape
      const auto& sym_vec = group->GetShapeOrDataExprs(value).shape();
      for (const auto& dim_expr : sym_vec) {
        DoEach(dim_expr);
      }
    } else {  // static shape
      for (int i = 0; i < dims.size(); ++i) {
        DoEach(::symbol::DimExpr{dims[i]});
      }
    }
  };

  std::vector<ir::Dim> sym_shape;
  ForEachDimExpr(
      [&](const auto& sym) { sym_shape.emplace_back(input_id, sym); });
  if (sym_shape.empty()) {
    sym_shape.emplace_back(input_id, symbol::DimExpr{1});
  }
  auto tensor = lang::CreatePlaceHolder(
      sym_shape, CompatibleInfo::ConvertIRType(dtype), input_id);
  auto IsIntType = [](const ::pir::Type& t) {
    return t.isa<::pir::Int32Type>() || t.isa<::pir::Int64Type>();
  };
  if (IsIntType(dtype) && group->HasShapeOrDataExprs(value)) {
    const auto& tensor_value = details::GetTensorValueFromShapeOrData(
        group->GetShapeOrDataExprs(value));
    if (tensor_value.has_value()) {
      tensor->set_value(*tensor_value);
    }
  }
  return tensor;
}

std::vector<ir::Tensor> OpLowererImpl::CollectInputTensor(
    const OpLoweringGroupPtr& group,
    const ::pir::Operation* op,
    std::vector<ir::Tensor>* func_args,
    std::unordered_map<::pir::Value, ir::Tensor>* tensor_map) {
  std::vector<ir::Tensor> tensors;
  for (auto in_value : CompatibleInfo::RealOperandSources(*op)) {
    VLOG(4) << "input tensor name: " << ValueName(in_value);
    ir::Tensor tensor = GetTensor(group, in_value);
    VLOG(4) << "shape: " << tensor->shape;
    VLOG(4) << "sym_shape: " << tensor->sym_shape;

    if (!tensor_map->count(in_value)) {
      // record tensor.
      (*tensor_map)[in_value] = tensor;
      // record func input args
      if (func_args != nullptr) {
        func_args->push_back(tensor);
      }
    } else {
      // TODO(6clc): After supporting symbolic calculation,
      // 1. Check that the shape of the tensor with the same name is the same
      // size
      // 2. Or make the symbol expression in compute output tensor consistent
      //    with the one inferred in shape_analysis
      (*tensor_map)[in_value]->sym_shape = tensor->sym_shape;
      (*tensor_map)[in_value]->shape = tensor->shape;
      (*tensor_map)[in_value]->sym_domain = tensor->sym_domain;
      (*tensor_map)[in_value]->domain = tensor->domain;
    }
    tensors.push_back(tensor);
  }
  return tensors;
}

void OpLowererImpl::CollectOutputInfo(::pir::Operation* op,
                                      std::vector<Type>* out_types,
                                      std::vector<std::vector<int>>* out_shapes,
                                      const OpLoweringGroupPtr& group) {
  auto op_results = op->results();
  for (auto& out_value : op_results) {
    std::string output_id = ValueName(out_value);

    auto type_info =
        out_value.type().dyn_cast<paddle::dialect::DenseTensorType>();

    out_types->push_back(CompatibleInfo::ConvertIRType(type_info.dtype()));
    auto out_shape = ::common::vectorize<int>(type_info.dims());
    if (out_shape.empty()) {
      out_shape.push_back(1);
    }
    out_shapes->push_back(std::move(out_shape));
  }
}

void OpLowererImpl::CollectOutputInfo(
    ::pir::Operation* op,
    std::vector<Type>* out_types,
    std::vector<std::vector<ir::Dim>>* out_shapes,
    const OpLoweringGroupPtr& group) {
  auto op_results = op->results();
  for (auto& out_value : op_results) {
    std::string output_id = ValueName(out_value);

    auto type_info =
        out_value.type().dyn_cast<paddle::dialect::DenseTensorType>();

    out_types->push_back(CompatibleInfo::ConvertIRType(type_info.dtype()));

    auto ForEachDimExpr = [&](const auto& DoEach) {
      const auto& dims = type_info.dims();
      if (::common::contain_unknown_dim(dims)) {  // dynamic shape
        const auto& sym_vec = group->GetShapeOrDataExprs(out_value).shape();
        std::vector<ir::Dim> sym_shape;
        for (const auto& sym : sym_vec) {
          DoEach(sym);
        }
      } else {  // static shape
        auto out_shape = ::common::vectorize<int64_t>(dims);
        for (int64_t dim : out_shape) {
          DoEach(symbol::DimExpr{dim});
        }
      }
    };
    std::vector<ir::Dim> sym_shape;
    ForEachDimExpr(
        [&](const auto& sym) { sym_shape.emplace_back(output_id, sym); });
    if (sym_shape.empty()) {
      sym_shape.emplace_back(output_id, symbol::DimExpr{1});
    }
    out_shapes->emplace_back(std::move(sym_shape));
  }
}

std::string OpLowererImpl::ValueName(::pir::Value value) {
  auto name = name_gene_->GetOrNew(value, CompatibleInfo::kNamePrefix);

  return name;
}

ir::LoweredFunc OpLowererImpl::GenerateInferShapeFunc(
    const OpLoweringGroupPtr& group,
    const std::vector<ir::Tensor> group_func_arg_tensors,
    const std::vector<ir::Argument> group_func_args) {
  std::vector<ir::Expr> ir_bodys;
  int output_tensor_idx = 0;
  for (int tensor_arg_idx = 0; tensor_arg_idx < group_func_arg_tensors.size();
       ++tensor_arg_idx) {
    auto tensor_dim = group_func_arg_tensors[tensor_arg_idx]->sym_shape;
    int tensor_dim_size = tensor_dim.size();
    auto tensor_shape = group_func_arg_tensors[tensor_arg_idx]->shape;

    ir::Var tensor_shape_args(TENSOR_SHAPE_ARGS, type_of<int64_t**>());
    for (int i = 0; i < tensor_shape.size(); i++) {
      ir::Expr call_set_infer_shape_value =
          ir::Call::Make(type_of<void>(),
                         runtime::intrinsic::infer_shape_set_value,
                         {ir::Expr(output_tensor_idx),
                          ir::Expr(i),
                          tensor_shape[i],
                          tensor_shape_args},
                         {},
                         ir::CallType::Extern,
                         ir::FunctionRef(),
                         0);
      ir_bodys.push_back(call_set_infer_shape_value);
    }
    ++output_tensor_idx;
  }
  ir::LoweredFunc infer_shape_func =
      ir::_LoweredFunc_::Make(group->FuncName() + "_infer_shape",
                              group_func_args,
                              ir::Block::Make(ir_bodys),
                              {});
  infer_shape_func->body_block =
      ir::ConvertExprBlockToStmtBlock(infer_shape_func->body);
  return infer_shape_func;
}
ir::Expr OpLowererImpl::LowerX86(const OpLoweringGroupPtr& group,
                                 const std::vector<::pir::Operation*>& ops,
                                 bool apply_op_schedule) {
  std::vector<ir::Tensor> group_func_arg_tensors;
  std::unordered_map<::pir::Value, ir::Tensor> tensor_map;
  // for some op, it will output more tmp value and regard as
  // XX_0, XX_1, so we log them in tmp_tensor_info;

  std::vector<::pir::Value> vec_inputs;
  std::vector<::pir::Value> vec_outputs;
  for (auto* op : ops) {
    for (size_t i = 0; i < op->num_operands(); ++i) {
      auto in = op->operand_source(i);
      if (!in || !in.type()) {
        continue;
      }

      vec_inputs.push_back(in);
    }

    for (size_t i = 0; i < op->num_results(); ++i) {
      auto out = op->result(i);
      if (!out || !out.type()) {
        continue;
      }

      vec_outputs.push_back(out);
    }
  }

  if (!paddle::dialect::CanGroupOpRunCpuKernel(vec_inputs, vec_outputs)) {
    return ir::Expr(-1);
  }

  this->target_ = common::DefaultHostTarget();
  cinn::runtime::CurrentTarget::SetCurrentTarget(this->target_);

  std::vector<ir::stmt::BlockRef> func_bodies =
      LowerOps(group, ops, &group_func_arg_tensors, &tensor_map);
  this->target_ = common::DefaultDeviceTarget();
  cinn::runtime::CurrentTarget::SetCurrentTarget(this->target_);
  // TODO(Hongqing-work): delete this convert after using new IR update
  // schedule.
  std::vector<ir::Expr> expr_func_bodies;
  for (const auto& body : func_bodies) {
    expr_func_bodies.emplace_back(ir::ConvertStmtBlockToExprBlock(body));
  }
  ir::ModuleExpr mod_expr(expr_func_bodies);
  ir::IRSchedule ir_sch(
      mod_expr, -1, false, cinn::utils::ErrorMessageLevel::kGeneral, true);
  ir_sch.MergeExprs();
  auto X86Expr = ir::ir_utils::IRCopy(ir_sch.GetModule().GetExprs().at(0));
  return X86Expr;
}

}  // namespace pir
}  // namespace framework
}  // namespace hlir
}  // namespace cinn
