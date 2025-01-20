// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/lang/lower_impl.h"

#include <algorithm>
#include <queue>
#include <string>
#include <unordered_set>

#include "paddle/cinn/common/common.h"
#include "paddle/cinn/common/context.h"
#include "paddle/cinn/common/ir_util.h"
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/tensor.h"
#include "paddle/cinn/optim/ir_simplify_pass.h"
#include "paddle/cinn/optim/replace_var_with_expr.h"
#include "paddle/cinn/optim/transform_polyfor_to_for.h"
#include "paddle/cinn/poly/stage.h"
#include "paddle/common/enforce.h"

PD_DECLARE_bool(cinn_runtime_display_debug_info);

namespace cinn {
namespace lang {
namespace detail {

void CheckNoIslCallRemains(Expr* expr) {
  auto isl_calls = ir::ir_utils::CollectIRNodes(*expr, [](const Expr* expr) {
    return expr->As<ir::Call>() && expr->As<ir::Call>()->is_isl_call();
  });
#ifdef CINN_DEBUG
  for (auto& item : isl_calls) {
    LOG(ERROR) << "ISL call: " << item;
  }
#endif
  if (!isl_calls.empty()) {
    LOG(WARNING) << "Some ISL call nodes remained, get " << isl_calls.size()
                 << " isl_calls, the first one is " << *isl_calls.begin();
  }
}

bool TensorContainsGPUInfo(ir::Tensor t, poly::Stage* stage) {
  if (stage->inlined()) return false;
  if (stage) {
    for (auto& info : stage->forloop_infos()) {
      if (info.second.device == ir::DeviceAPI::GPU) {
        return true;
      }
    }
  }
  return false;
}

const char* CompuGraphNode::__type_info__ = "ComputeGraphNode";
const char* CompuGraphNode::type_info() const { return __type_info__; }
std::string CompuGraphNode::id() const {
  PADDLE_ENFORCE_EQ(
      tensor.defined(),
      true,
      ::common::errors::InvalidArgument("Tensor is not defined. Please ensure "
                                        "tensor is properly initialized."));
  return tensor->name;
}

void LowerImpl::CheckArgsUnique() {
  for (auto& tensor : tensor_args_) {
    if (!tensor->buffer.defined()) {
      LOG(ERROR) << "tensor [" << tensor->name << "] buffer is null";
      continue;
    }
  }
}

std::vector<ir::Argument> LowerImpl::GenerateFunctionArgumentList(
    Expr fn_body) {
  CheckArgsUnique();

  std::vector<ir::Argument> args;
  auto teller = ir::ir_utils::CollectTensorNeedsWrite(&fn_body);

  std::set<std::string> arg_names;

  for (auto& scalar : scalar_args_) {
    PADDLE_ENFORCE_EQ(
        arg_names.count(scalar->name),
        0,
        ::common::errors::InvalidArgument(
            "Argument name '%s' already exists in the argument names set.",
            scalar->name));
    auto* scalar_node = scalar.As<ir::_Var_>();
    PADDLE_ENFORCE_EQ(
        scalar_node->type().valid(),
        true,
        ::common::errors::InvalidArgument(
            "The type of scalar node '%s' is not valid.", scalar->name));
    arg_names.insert(scalar->name);

    args.emplace_back(scalar, ir::Argument::IO::kInput);
  }

  for (auto& tensor : tensor_args_) {
    auto* tensor_node = tensor.As<ir::_Tensor_>();
    bool is_output = teller.count(tensor->name);
    VLOG(1) << "tensor argument " << tensor->name << " buffer "
            << tensor->buffer->name;

    // avoid duplicate
    if (!tensor_node->buffer.defined()) continue;
    // if a argument is already marked as kInput, mark it as kOutput and move it
    // to the back.
    if (arg_names.count(tensor_node->buffer->name)) {
      auto it =
          std::find_if(args.begin(), args.end(), [&](const ir::Argument& x) {
            return x.name() == tensor_node->buffer->name;
          });
      PADDLE_ENFORCE_EQ(
          it != args.end(),
          true,
          ::common::errors::InvalidArgument(
              "Argument with name '%s' not found in the argument list.",
              tensor_node->buffer->name));
      if (it->is_input()) {
        args.erase(it);
      } else if (it->is_output()) {
        continue;
      }
    }

    arg_names.insert(tensor_node->buffer->name);

    auto io = is_output ? ir::Argument::IO::kOutput : ir::Argument::IO::kInput;
    VLOG(3) << "Collect " << (is_output ? "W" : "R") << " argument "
            << tensor->buffer->name;
    args.emplace_back(tensor_node->buffer, io);
  }

  return args;
}
// Generate Function Arguments for splitted kernel.
std::vector<ir::Argument> LowerImpl::GenFuncArgForSplitKernel(
    Expr func_iterator, std::vector<ir::Tensor> temp_tensors) {
  CheckArgsUnique();

  std::vector<ir::Argument> in_args;
  std::vector<ir::Argument> out_args;
  auto teller = ir::ir_utils::CollectTensorNeedsWrite(&func_iterator);
  std::set<std::string> arg_names;
  std::set<std::string> all_tensor_names;

  for (auto& scalar : scalar_args_) {
    PADDLE_ENFORCE_EQ(
        arg_names.count(scalar->name),
        0,
        ::common::errors::InvalidArgument(
            "Argument name '%s' already exists in the argument names set.",
            scalar->name));
    auto* scalar_node = scalar.As<ir::_Var_>();
    PADDLE_ENFORCE_EQ(
        scalar_node->type().valid(),
        true,
        ::common::errors::InvalidArgument(
            "The type of scalar node '%s' is not valid.", scalar->name));
    arg_names.insert(scalar->name);

    in_args.emplace_back(scalar, ir::Argument::IO::kInput);
  }

  auto all_tensors = ir::ir_utils::CollectIRNodes(
      func_iterator, [&](const Expr* x) { return x->as_tensor(); });

  auto all_vars = ir::ir_utils::CollectIRNodes(
      func_iterator, [&](const Expr* x) { return x->as_var(); });

  for (auto& i : all_tensors) {
    auto* tensor = i.as_tensor();
    all_tensor_names.insert(tensor->name);
    VLOG(3) << "In all_tensors, it has : " << tensor->name;
  }
  for (auto& i : all_vars) {
    auto* var = i.as_var();
    VLOG(3) << "In all_vars, it has : " << var->name;
  }

  for (auto& i : scalar_args_) {
    VLOG(3) << "In scalar_args_, var has : " << i->name;
  }

  std::set<std::string> temp_tensor_names;

  for (auto& i : temp_tensors) {
    VLOG(3) << "In temp_tensors, it has : " << i->name;
    temp_tensor_names.insert(i->name);
  }

  for (auto& tensor : tensor_args_) {
    VLOG(3) << "In tensor_args_, it has : " << tensor->name;
    if (temp_tensor_names.count(tensor->name) > 0) continue;
    if (all_tensor_names.count(tensor->name) == 0) continue;
    bool is_output = teller.count(tensor->name);
    VLOG(3) << "tensor argument " << tensor->name << " buffer "
            << tensor->buffer->name;

    // avoid duplicate
    if (!tensor->buffer.defined()) {
      VLOG(3) << "tensor->buffer is not defined";
      continue;
    }
    // if a argument is already marked as kInput, mark it as kOutput and move it
    // to the back.
    if (arg_names.count(tensor->buffer->name)) {
      auto it = std::find_if(
          in_args.begin(), in_args.end(), [&](const ir::Argument& x) {
            return x.name() == tensor->buffer->name;
          });
      if (it != in_args.end()) {
        in_args.erase(it);
      } else {
        continue;
      }
    }

    arg_names.insert(tensor->buffer->name);

    auto io = is_output ? ir::Argument::IO::kOutput : ir::Argument::IO::kInput;
    if (io == ir::Argument::IO::kInput)
      in_args.emplace_back(tensor->buffer, io);
    else
      out_args.emplace_back(tensor->buffer, io);
  }
  if (out_args.empty()) {
    for (auto& i : all_tensors) {
      auto* tensor = i.as_tensor();
      VLOG(3) << "Tensor " << tensor->name;
      if (tensor->buffer.defined() && !arg_names.count(tensor->buffer->name)) {
        bool is_output =
            teller.count(tensor->name) && teller.count(tensor->name);
        if (is_output)
          out_args.emplace_back(tensor->buffer, ir::Argument::IO::kOutput);
      }
    }
  }

  std::vector<ir::Argument> args(in_args.begin(), in_args.end());
  args.insert(std::end(args), out_args.begin(), out_args.end());
  return args;
}

std::vector<Tensor> LowerImpl::CollectTemporaryTensors() {
  // a temporary should be in the comp_graph but not contained in the
  // tensor_args.
  absl::flat_hash_map<std::string, Tensor> tensor_arg_map = GenTensorArgMap();
  absl::flat_hash_map<std::string, Tensor> temp_tensor_map;

  for (auto* node : compu_graph_->nodes()) {
    auto* cnode = node->safe_as<CompuGraphNode>();
    PADDLE_ENFORCE_NOT_NULL(
        cnode,
        ::common::errors::InvalidArgument(
            "Node could not be safely cast to CompuGraphNode."));
    if (!tensor_arg_map.count(cnode->tensor->name)) {
      temp_tensor_map[cnode->tensor->name] = cnode->tensor;
    }
  }

  std::vector<Tensor> temp_tensors;
  std::transform(
      temp_tensor_map.begin(),
      temp_tensor_map.end(),
      std::back_inserter(temp_tensors),
      [&](const decltype(temp_tensor_map)::value_type& x) { return x.second; });
  return temp_tensors;
}

absl::flat_hash_map<std::string, Tensor> LowerImpl::GenTensorArgMap() {
  absl::flat_hash_map<std::string, Tensor> map;
  for (auto& t : tensor_args_) {
    map[t->name] = t;
  }
  return map;
}

absl::flat_hash_map<std::string, Tensor> LowerImpl::GenAllTensorMap() {
  absl::flat_hash_map<std::string, Tensor> map;
  for (auto& t : CollectAllTensors()) {
    map[t->name] = t;
  }
  return map;
}

std::vector<ir::LoweredFunc> LowerImpl::operator()() {
  std::vector<poly::Stage*> stages;
  std::map<std::string, ir::Tensor> all_tensor_map;
  for (auto& t : CollectAllTensors()) {
    all_tensor_map[t->name] = t;
  }

  auto deps = CollectExtraDependencies();
  auto schedule =
      poly::CreateSchedule(stages,
                           poly::ScheduleKind::Poly,
                           std::vector<std::pair<std::string, std::string>>(
                               deps.begin(), deps.end()));
  auto func_body = GenerateFunctionBody(schedule.get());

  std::vector<ir::LoweredFunc> result;
  int num_func = 0;
  for (auto& func_iterator : func_body) {
    if (support_ir_schedule_) {
      // add ScheduleBlockRealize
      func_iterator = ir::ScheduleBlockRealize::Make(
          {},
          ir::ScheduleBlock::Make(
              {}, {}, {}, cinn::common::UniqName("root"), func_iterator));
    }
    std::set<std::string> temp_tensor_names;
    for (auto& t : temp_tensor_args_) temp_tensor_names.insert(t->name);

    auto store_exprs = ir::ir_utils::CollectIRNodes(
        func_iterator, [](const Expr* x) { return x->As<ir::Store>(); });
    std::vector<ir::Tensor> new_temp_tensors;
    for (auto& expr : store_exprs) {
      auto* store_node = expr.As<ir::Store>();
      PADDLE_ENFORCE_NOT_NULL(
          store_node,
          ::common::errors::InvalidArgument(
              "Expression could not be cast to ir::Store."));
      auto* tensor = store_node->tensor.As<ir::_Tensor_>();
      PADDLE_ENFORCE_NOT_NULL(
          tensor,
          ::common::errors::InvalidArgument(
              "Store node's tensor could not be cast to ir::_Tensor_."));
      VLOG(3) << "In store_exprs, its name is : " << tensor->name;
      PADDLE_ENFORCE_EQ(
          tensor->buffer.defined(),
          true,
          ::common::errors::InvalidArgument(
              "Tensor buffer is not defined for tensor with name '%s'.",
              tensor->name));
      if (tensor->buffer->memory_type != ir::MemoryType::Heap) {
        new_temp_tensors.push_back(store_node->tensor.as_tensor_ref());
      }
    }

    auto func_temp_tensors = CollectTemporaryTensors();
    std::vector<ir::Buffer> temp_buffers;
    std::unordered_set<std::string> buffer_name_set;
    // TODO(Superjomn) write buffer latter.

    ir::LoweredFunc func;
    if (target_ == cinn::common::DefaultNVGPUTarget()) {
      auto func_args2 =
          GenFuncArgForSplitKernel(func_iterator, new_temp_tensors);
      std::string new_fn_name = fn_name_;
      if (num_func > 0) {
        new_fn_name += "_" + std::to_string(num_func);
      }
      VLOG(3) << "Making func :" << new_fn_name;
      for (auto& i : func_args2) {
        VLOG(3) << "func_args2 is : " << i.name();
      }
      for (auto& i : temp_buffers) {
        VLOG(3) << "temp_buffers is : " << i->name;
      }
      func = ir::_LoweredFunc_::Make(
          new_fn_name, func_args2, func_iterator, temp_buffers);
    } else {
      auto func_args = GenerateFunctionArgumentList(func_iterator);
      func = ir::_LoweredFunc_::Make(
          fn_name_, func_args, func_iterator, temp_buffers);
    }

    if (support_ir_schedule_) {
      optim::TransformPolyForToFor(&func->body);
      func->body = ir::Block::Make({func->body});
      result.push_back(func);
      num_func++;
    } else {
      auto res = optim::Optimize(func,
                                 target_,
                                 FLAGS_cinn_runtime_display_debug_info,
                                 /* remove_gpu_for_loops = */ false);

      if (cuda_axis_info_.size() > num_func &&
          cuda_axis_info_[num_func].valid()) {
        res->cuda_axis_info = cuda_axis_info_[num_func];
      }
      result.push_back(res);
      num_func++;
    }
  }
  return result;
}

std::vector<Tensor> LowerImpl::CollectAllTensors() {
  std::vector<Tensor> tensors;
  auto topo_order = compu_graph_->topological_order();  // NOLINT
  auto& nodes = std::get<0>(topo_order);
  auto& edges = std::get<1>(topo_order);
  for (auto* node : nodes) {
    auto* cnode = node->safe_as<CompuGraphNode>();
    PADDLE_ENFORCE_NOT_NULL(
        cnode,
        ::common::errors::InvalidArgument(
            "Node could not be safely cast to CompuGraphNode."));
    tensors.push_back(cnode->tensor);
  }
  return tensors;
}

std::set<std::pair<std::string, std::string>>
LowerImpl::CollectExtraDependencies() const {
  std::set<std::pair<std::string, std::string>> deps;
  for (auto* node : compu_graph_->nodes()) {
    auto* cnode = node->safe_as<CompuGraphNode>();
    PADDLE_ENFORCE_NOT_NULL(
        cnode,
        ::common::errors::InvalidArgument(
            "Node could not be safely cast to CompuGraphNode."));
  }
  return deps;
}

std::vector<Expr> LowerImpl::GenerateFunctionBody(
    const poly::Schedule* schedule) {
  // generate the expressions for each group.
  std::vector<Expr> exprs;
  std::vector<Expr> result;
  auto tensor_map = GenAllTensorMap();
  std::map<std::string, Expr> tuple_to_expr;
  PADDLE_ENFORCE_NE(
      schedule->groups.empty(),
      true,
      ::common::errors::NotFound("No group is generated in the schedule."));

  std::map<std::string, ir::Tensor> global_tensor_map;
  std::unordered_map<std::string, std::vector<Expr>> resized_buffer_cache;

  for (auto& group : schedule->groups) {
    PADDLE_ENFORCE_GT(
        group.nodes.size(),
        0,
        ::common::errors::InvalidArgument(
            "Group is empty"
            "Expected size of group is larger than 0, but receive %d. ",
            group.nodes.size()));
    bool all_temp_tensor = true;
    for (auto& node : group.nodes) {
      if (!tensor_map.count(node->id())) {
        VLOG(2) << "tensor_map doesn't count " << node->id();
        continue;
      }
      auto& tensor = tensor_map[node->id()];
      if (!tensor->has_expression()) continue;
      all_temp_tensor =
          all_temp_tensor &&
          ((tensor->buffer.defined() &&
            (tensor->buffer->memory_type == ir::MemoryType::GPUShared ||
             tensor->buffer->memory_type == ir::MemoryType::GPULocal)));
      auto store_body = tensor->tensor_store_expanded_body();
      if (support_ir_schedule_) {
        // add schedule block of tensor computation for schedule IR
        int var_counts = tensor->shape.size() + tensor->reduce_axis.size();
        std::vector<int> int_shape;
        VLOG(3) << "Tensor " << tensor->name
                << "'s shape is : " << utils::Join(tensor->shape, ",");
        for (auto& expr : tensor->shape) {
          PADDLE_ENFORCE_EQ(expr.is_constant(),
                            true,
                            ::common::errors::InvalidArgument(
                                "Expected constant expression for tensor "
                                "shape, but got non-constant expression."));
          int_shape.push_back(static_cast<int>(expr.get_constant()));
        }
        for (auto& var : tensor->reduce_axis) {
          PADDLE_ENFORCE_EQ(
              var->lower_bound.defined(),
              true,
              ::common::errors::InvalidArgument(
                  "Lower bound of reduce axis variable is not defined."));
          PADDLE_ENFORCE_EQ(
              var->upper_bound.defined(),
              true,
              ::common::errors::InvalidArgument(
                  "Upper bound of reduce axis variable is not defined."));
          PADDLE_ENFORCE_EQ(cinn::common::is_zero(var->lower_bound),
                            true,
                            ::common::errors::InvalidArgument(
                                "Expected lower bound of reduce axis variable "
                                "to be zero, but got non-zero."));
          PADDLE_ENFORCE_EQ(
              var->upper_bound.is_constant(),
              true,
              ::common::errors::InvalidArgument(
                  "Expected constant expression for upper bound of reduce axis "
                  "variable, but got non-constant expression."));
          int_shape.push_back(
              static_cast<int>(var->upper_bound.get_constant()));
        }
        // create block itervars, i0,i1...
        std::vector<Var> block_vars;
        std::vector<Expr> iter_values;
        std::vector<Var> axis_vars =
            cinn::common::GenDefaultAxis(tensor->shape.size());
        // bind var_values
        axis_vars.insert(axis_vars.end(),
                         tensor->reduce_axis.begin(),
                         tensor->reduce_axis.end());
        for (int i = 0; i < var_counts; i++) {
          block_vars.push_back(Var(Expr(0),
                                   Expr(int_shape[i]),
                                   cinn::UniqName("i" + std::to_string(i)),
                                   false));
          if (i >= tensor->shape.size()) {
            block_vars[i]->is_reduce_axis = true;
            axis_vars[i]->is_reduce_axis = true;
          }
          iter_values.push_back(axis_vars[i]);
          // replace store's indice
          VLOG(3) << "replace axis_var " << axis_vars[i]->name
                  << " to block_var " << block_vars[i];
          optim::ReplaceVarWithExpr(&store_body, axis_vars[i], block_vars[i]);
        }
        store_body = ir::ScheduleBlockRealize::Make(
            iter_values,
            ir::ScheduleBlock::Make(
                block_vars, {}, {}, tensor->name, store_body));
        // iter_values, ir::ScheduleBlock::Make(block_vars, {}, {},
        // cinn::common::UniqName(tensor->name), store_body));
        VLOG(3) << "store body\n" << store_body;
      }
      tuple_to_expr[tensor->name] = store_body;
    }

    ir::CudaAxisInfo temp_cuda_axis_info;
    Expr group_expr;

    if (group_expr.defined()) {
      cuda_axis_info_.emplace_back(std::move(temp_cuda_axis_info));
      if (target_ == cinn::common::DefaultNVGPUTarget() && !all_temp_tensor) {
        exprs.push_back(group_expr);
        Expr body = ir::Block::Make(exprs);
        result.push_back(body);
        exprs.clear();
      } else {
        exprs.push_back(group_expr);
      }
    }
  }
  if (target_ == cinn::common::DefaultHostTarget()) {
    Expr body = ir::Block::Make(exprs);
    result.push_back(body);
    exprs.clear();
  } else if (!exprs.empty()) {
    Expr body = ir::Block::Make(exprs);
    result.push_back(body);
    exprs.clear();
  }

  return result;
}

}  // namespace detail
}  // namespace lang
}  // namespace cinn
