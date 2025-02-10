// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/lang/lower_tensor_group.h"

#include <algorithm>
#include <queue>
#include <string>
#include <unordered_set>

#include "paddle/cinn/ast_gen_ius/ast_gen.h"
#include "paddle/cinn/ast_gen_ius/tensor_group.h"
#include "paddle/cinn/common/common.h"
#include "paddle/cinn/common/context.h"
#include "paddle/cinn/common/ir_util.h"
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/ir/ir_mutator.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/tensor.h"
#include "paddle/cinn/optim/ir_simplify.h"
#include "paddle/cinn/optim/replace_var_with_expr.h"
#include "paddle/cinn/optim/transform_polyfor_to_for.h"
#include "paddle/cinn/poly/stage.h"

using cinn::ir::stmt::BlockRef;
using cinn::ir::stmt::Schedule;
using cinn::ir::stmt::StmtRef;
using cinn::ir::stmt::Store;

namespace cinn {
namespace lang {
namespace detail {

LowerTensorGroup::LowerTensorGroup(
    const std::string& fn_name,
    const std::vector<ir::Tensor>& tensor_args,
    const std::vector<ir::Var>& scalar_args,
    ast_gen_ius::TensorGroup* tensor_group,
    const std::vector<ir::Tensor>& temp_tensor_args,
    const Target& target)
    : fn_name_(fn_name),
      tensor_args_(tensor_args),
      scalar_args_(scalar_args),
      tensor_group_(tensor_group),
      temp_tensor_args_(temp_tensor_args),
      target_(target) {}

std::vector<ir::LoweredFunc> LowerTensorGroup::operator()() {
  std::vector<ir::LoweredFunc> result;
  int num_func = 0;

  // 1. Generate function body
  std::vector<BlockRef> func_bodies = GenerateFunctionBody(tensor_group_);
  for (const BlockRef& func_body : func_bodies) {
    // 2. Assign buffer to tensors
    auto tensor_map = tensor_group_->AllocateBuffers();
    // copy the tensor(with buffer assigned) back to func's args.
    for (auto& arg : tensor_args_) {
      if (arg->is_placeholder_node() || arg->buffer.defined()) {
        continue;
      }
      if (arg->body().As<ir::Call>() && arg->body().type().is_void()) {
        continue;  // extern call
      }

      if (tensor_map.find(arg->name) == tensor_map.end()) {
        LOG(INFO) << "Didn't find arg tensor " << arg->name
                  << "in tensor_map.\n"
                  << "The function is " << fn_name_
                  << "\nAnd all the arg tensors are:\n";
        for (auto& i : tensor_args_) {
          LOG(INFO) << i->name;
        }
        PADDLE_THROW(::common::errors::InvalidArgument("Fatal Error!"));
      }
      Reference(&arg)->buffer = tensor_map.at(arg->name)->buffer;
    }

    // 3. Collect temp tensor buffers
    std::set<std::string> temp_tensor_names;
    for (auto& t : temp_tensor_args_) {
      temp_tensor_names.insert(t->name);
    }

    // Some store tensors are also temp tensors;
    const auto& CollectTempTensorsInStore = [&](const StmtRef& stmt) {
      if (stmt.isa<Store>()) {
        const auto& store_stmt = stmt.as<Store>();
        PADDLE_ENFORCE_EQ(store_stmt.defined(),
                          true,
                          ::common::errors::InvalidArgument(
                              "store stmt should not be nullptr"));
        auto* tensor = store_stmt->tensor().As<ir::_Tensor_>();
        PADDLE_ENFORCE_NOT_NULL(
            tensor,
            ::common::errors::InvalidArgument(
                "tensor of store stmt should not be nullptr"));
        VLOG(3) << "In store stmt, its name is : " << tensor->name;
        PADDLE_ENFORCE_EQ(
            tensor->buffer.defined(),
            true,
            ::common::errors::InvalidArgument("tensor->buffer is nullptr"));
        if (tensor->buffer->memory_type != ir::MemoryType::Heap) {
          temp_tensor_names.insert(store_stmt->tensor().as_tensor_ref()->name);
        }
      }
    };
    ir::stmt::Visit(
        func_body, CollectTempTensorsInStore, [](const StmtRef& stmt) {});

    std::vector<ir::Buffer> temp_buffers;
    std::unordered_set<std::string> buffer_name_set;
    for (const std::string& name : temp_tensor_names) {
      if (!tensor_map.count(name)) {
        continue;
      }
      ir::Tensor& t = tensor_map[name];
      if (t->buffer.defined() && !buffer_name_set.count(t->buffer->name)) {
        temp_buffers.push_back(t->buffer);
        buffer_name_set.insert(t->buffer->name);
      }
    }

    // 4. Handle function args
    std::vector<ir::Argument> func_args =
        GenerateFunctionArgumentList(func_body);

    // 5. Actual function make
    std::string actual_fn_name = fn_name_;
    if (num_func > 0) {
      actual_fn_name += "_" + std::to_string(num_func);
      VLOG(3) << "Making func :" << actual_fn_name;
    }
    for (auto& i : func_args) {
      VLOG(3) << "func_args is : " << i.name();
    }
    for (auto& i : temp_buffers) {
      VLOG(3) << "temp_buffers is : " << i->name;
    }

    // 6. Final wrap with schedule root
    ir::LoweredFunc func = ir::_LoweredFunc_::Make(
        actual_fn_name,
        func_args,
        BlockRef(std::vector<StmtRef>{Schedule(
            {}, {}, {}, {}, cinn::common::UniqName("root"), func_body)}),
        temp_buffers);
    result.push_back(func);
    num_func++;
  }
  return result;
}

std::vector<ir::Argument> LowerTensorGroup::GenerateFunctionArgumentList(
    const BlockRef& fn_body) {
  std::vector<ir::Argument> args;
  auto teller = ir::ir_utils::CollectTensorNeedsWrite(fn_body);

  std::set<std::string> arg_names;

  for (auto& scalar : scalar_args_) {
    PADDLE_ENFORCE_EQ(!arg_names.count(scalar->name),
                      true,
                      ::common::errors::InvalidArgument(
                          "arg_names.count(scalar->name) is true"));
    auto* scalar_node = scalar.As<ir::_Var_>();
    PADDLE_ENFORCE_EQ(scalar_node->type().valid(),
                      true,
                      ::common::errors::InvalidArgument(
                          "scalar_node->type().valid() is false"));
    arg_names.insert(scalar->name);

    args.emplace_back(scalar, ir::Argument::IO::kInput);
  }

  for (auto& tensor : tensor_args_) {
    auto* tensor_node = tensor.As<ir::_Tensor_>();
    bool is_output = teller.count(tensor->name);
    VLOG(6) << "tensor argument " << tensor->name << ", buffer "
            << tensor->buffer->name << ", is output: " << is_output;

    // avoid duplicate
    if (!tensor_node->buffer.defined()) {
      continue;
    }
    // if a argument is already marked as kInput, mark it as kOutput and move
    // it to the back.
    if (arg_names.count(tensor_node->buffer->name)) {
      auto it =
          std::find_if(args.begin(), args.end(), [&](const ir::Argument& x) {
            return x.name() == tensor_node->buffer->name;
          });
      PADDLE_ENFORCE_EQ(it != args.end(),
                        true,
                        ::common::errors::InvalidArgument(
                            "it which refers to first element should be end"));
      if (it->is_input()) {
        args.erase(it);
      } else if (it->is_output()) {
        continue;
      }
    }

    arg_names.insert(tensor_node->buffer->name);

    auto io = is_output ? ir::Argument::IO::kOutput : ir::Argument::IO::kInput;
    VLOG(6) << "Collect " << (is_output ? "W" : "R") << " argument "
            << tensor->buffer->name;
    args.emplace_back(tensor_node->buffer, io);
  }

  return args;
}

std::vector<BlockRef> LowerTensorGroup::GenerateFunctionBody(
    ast_gen_ius::TensorGroup* tensor_group) {
  // TODO(zhhsplendid): GetGenFuncTopoOrder() may remove args
  std::vector<ir::Tensor> ordered_tensors = tensor_group->GetGenFuncTopoOrder();

  std::vector<BlockRef> result;
  std::vector<StmtRef> bodies;
  for (const ir::Tensor& tensor : ordered_tensors) {
    VLOG(6) << "tensor_name = " << tensor->name;
    if (!tensor->is_placeholder_node() && tensor->has_expression()) {
      VLOG(6) << "ast_gen_ius::AstGen::Build for Tensor " << tensor;
      bodies.emplace_back(ast_gen_ius::AstGen::Build(tensor, tensor_group));

      bool gpu_local =
          tensor->buffer.defined() &&
          (tensor->buffer->memory_type == ir::MemoryType::GPUShared ||
           tensor->buffer->memory_type == ir::MemoryType::GPULocal);
      target_.arch.Match(
          [&](common::NVGPUArch) {
            if (!gpu_local) {
              result.push_back(BlockRef(bodies));
              bodies.clear();
            }
          },
          [&](std::variant<common::HygonDCUArchHIP, common::HygonDCUArchSYCL>) {
            if (!gpu_local) {
              result.push_back(BlockRef(bodies));
              bodies.clear();
            }
          },
          [&](std::variant<common::UnknownArch,
                           common::X86Arch,
                           common::ARMArch>) {});
    }
  }

  if (!bodies.empty()) {
    result.push_back(BlockRef(bodies));
    bodies.clear();
  }
  return result;
}

}  // namespace detail
}  // namespace lang
}  // namespace cinn
