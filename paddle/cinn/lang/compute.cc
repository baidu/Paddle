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

#include "paddle/cinn/lang/compute.h"

#include "paddle/cinn/backends/extern_func_protos.h"
#include "paddle/cinn/common/common.h"
#include "paddle/cinn/ir/operation.h"
#include "paddle/cinn/optim/ir_simplify_pass.h"
#include "paddle/cinn/poly/dim.h"
#include "paddle/cinn/poly/domain.h"
#include "paddle/cinn/poly/stage.h"
#include "paddle/cinn/runtime/use_extern_funcs.h"

namespace cinn {
namespace lang {

ir::Tensor Compute(const std::vector<Expr> &domain,
                   std::function<Expr()> fn,
                   const std::string &name,
                   const std::vector<Expr> &shape) {
  return Compute(
      domain,
      [fn](const std::vector<Expr> &axis) -> Expr { return fn(); },
      name,
      shape);
}

ir::Tensor Compute(const std::vector<Expr> &domain,
                   std::function<Expr(Expr)> fn,
                   const std::string &name,
                   const std::vector<Expr> &shape) {
  return Compute(
      domain,
      [fn](const std::vector<Expr> &axis) -> Expr {
        PADDLE_ENFORCE_EQ(axis.size(),
                          1,
                          ::common::errors::InvalidArgument(
                              "The size of axis vector is incorrect"
                              "Expected value is 1, but receive %d. ",
                              axis.size()));
        return fn(axis[0]);
      },
      name,
      shape);
}

ir::Tensor Compute(const std::vector<Expr> &domain,
                   std::function<Expr(Expr, Expr)> fn,
                   const std::string &name,
                   const std::vector<Expr> &shape) {
  return Compute(
      domain,
      [fn](const std::vector<Expr> &axis) -> Expr {
        PADDLE_ENFORCE_EQ(axis.size(),
                          2,
                          ::common::errors::InvalidArgument(
                              "The size of axis vector is incorrect"
                              "Expected value is 2, but receive %d. ",
                              axis.size()));
        return fn(axis[0], axis[1]);
      },
      name,
      shape);
}

ir::Tensor Compute(const std::vector<Expr> &domain,
                   std::function<Expr(Expr, Expr, Expr)> fn,
                   const std::string &name,
                   const std::vector<Expr> &shape) {
  return Compute(
      domain,
      [fn](const std::vector<Expr> &axis) -> Expr {
        PADDLE_ENFORCE_EQ(axis.size(),
                          3,
                          ::common::errors::InvalidArgument(
                              "The size of axis vector is incorrect"
                              "Expected value is 3, but receive %d. ",
                              axis.size()));
        return fn(axis[0], axis[1], axis[2]);
      },
      name,
      shape);
}

ir::Tensor Compute(const std::vector<Expr> &domain,
                   std::function<Expr(Expr, Expr, Expr, Expr)> fn,
                   const std::string &name,
                   const std::vector<Expr> &shape) {
  return Compute(
      domain,
      [fn](const std::vector<Expr> &axis) -> Expr {
        PADDLE_ENFORCE_EQ(axis.size(),
                          4,
                          ::common::errors::InvalidArgument(
                              "The size of axis vector is incorrect"
                              "Expected value is 4, but receive %d. ",
                              axis.size()));
        return fn(axis[0], axis[1], axis[2], axis[3]);
      },
      name,
      shape);
}

ir::Tensor Compute(const std::vector<Expr> &domain,
                   std::function<Expr(Expr, Expr, Expr, Expr, Expr)> fn,
                   const std::string &name,
                   const std::vector<Expr> &shape) {
  return Compute(
      domain,
      [fn](const std::vector<Expr> &axis) -> Expr {
        PADDLE_ENFORCE_EQ(axis.size(),
                          5,
                          ::common::errors::InvalidArgument(
                              "The size of axis vector is incorrect"
                              "Expected value is 5, but receive %d. ",
                              axis.size()));
        return fn(axis[0], axis[1], axis[2], axis[3], axis[4]);
      },
      name,
      shape);
}

ir::Tensor Compute(const std::vector<Expr> &domain,
                   std::function<Expr(Expr, Expr, Expr, Expr, Expr, Expr)> fn,
                   const std::string &name,
                   const std::vector<Expr> &shape) {
  return Compute(
      domain,
      [fn](const std::vector<Expr> &axis) -> Expr {
        PADDLE_ENFORCE_EQ(axis.size(),
                          6,
                          ::common::errors::InvalidArgument(
                              "The size of axis vector is incorrect"
                              "Expected value is 6, but receive %d. ",
                              axis.size()));
        return fn(axis[0], axis[1], axis[2], axis[3], axis[4], axis[5]);
      },
      name,
      shape);
}

ir::Tensor Compute(const std::vector<Expr> &domain,
                   std::function<Expr(const std::vector<Expr> &)> fn,
                   const std::string &name,
                   const std::vector<Expr> &shape) {
  auto axes = cinn::common::GenDefaultAxis(domain.size());
  std::vector<Expr> _axis;
  for (auto &x : axes) _axis.push_back(x);
  Expr fn_body = fn(_axis);

  std::vector<Var> reduce_axis;
  if (fn_body.defined() && fn_body.As<ir::Reduce>()) {
    auto &fn_reduce_axis = fn_body.As<ir::Reduce>()->reduce_axis;
    reduce_axis.insert(
        std::begin(reduce_axis), fn_reduce_axis.begin(), fn_reduce_axis.end());
  }

  // When the fn_body is a CallExtern, a tensor will return directly.
  if (fn_body.as_tensor()) {
    return fn_body.as_tensor_ref();
  }

  // shape is the buffer's shape.
  std::vector<Expr> domain_without_reduce_axis;
  std::vector<Expr> shape_simplified;

  // construct the shape.
  for (auto dim : domain) {
    auto copied = dim;
    optim::Simplify(&copied, optim::SimplifyType::kExpr);
    domain_without_reduce_axis.push_back(copied);
  }

  for (auto dim : shape) {
    auto copied = dim;
    optim::Simplify(&copied, optim::SimplifyType::kExpr);
    shape_simplified.push_back(copied);
  }

  auto real_shape =
      shape_simplified.empty() ? domain_without_reduce_axis : shape_simplified;

  // The body returns void, that means no buffer is needed.
  if (fn_body.type() == Void()) real_shape.clear();

  auto unique_name = name.empty() ? Context::Global().NewName("tensor") : name;

  // check reduce_axis not include the reserved axis name
  for (auto &ra : reduce_axis) {
    PADDLE_ENFORCE_EQ(
        !cinn::common::IsAxisNameReserved(ra->name),
        true,
        ::common::errors::InvalidArgument(
            "Reduce axis [%s]'s name is reserved.", ra->name.c_str()));
  }

  VLOG(3) << "tensor " << name
          << "'s domain is : " << domain_without_reduce_axis;

  auto op = ir::ComputeOp::Make(
      unique_name, fn, real_shape, domain_without_reduce_axis, reduce_axis);

  auto tensor = ir::Tensor(unique_name,
                           fn_body.type(),
                           real_shape,
                           domain_without_reduce_axis,
                           op,
                           reduce_axis);
  const auto set_keep_dim_for_tensor = [&]() {
    for (int i = 0; i < _axis.size(); ++i) {
      const auto &axis_var = _axis.at(i);
      tensor->axis_[i]->is_keepdim = axis_var.as_var_ref()->is_keepdim;
    }
  };
  set_keep_dim_for_tensor();
  return tensor;
}

std::vector<ir::Tensor> CallLowered(
    const std::string &func_name,
    const std::vector<Expr> &args,
    const std::vector<ReturnType> &return_types) {
  auto call = ir::Call::Make(
      Void(), func_name, args, {}, ir::CallType::CINN, ir::FunctionRef(), 0);
  std::vector<ir::Tensor> new_tensors;
  for (int i = 0; i < return_types.size(); i++) {
    auto &return_type = return_types[i];
    auto call_op = ir::CallOp::Make(func_name, call);
    auto new_tensor = ir::Tensor(return_type.name,
                                 return_type.type,
                                 return_type.dims,
                                 {Expr(1)},
                                 call_op);
    // Append write tensors in the tail.
    call.As<ir::Call>()->write_args.push_back(new_tensor);
    new_tensor->set_type(return_type.type);
    new_tensor->WithBuffer();
    new_tensors.push_back(new_tensor);
  }

  return new_tensors;
}

Expr CallExtern(const std::string &func_name,
                const std::vector<Expr> &args,
                const std::map<std::string, attr_t> &attrs) {
  auto *proto =
      backends::ExternFunctionProtoRegistry::Global().Lookup(func_name);
  PADDLE_ENFORCE_NOT_NULL(
      proto,
      ::common::errors::InvalidArgument(
          "No extern function prototype %s found\nExisting records are:\n%s",
          func_name,
          backends::ExternFunctionProtoRegistry::Global().debug_string()));

  auto call = ir::Call::Make(proto->ret_type,
                             func_name,
                             args,
                             {},
                             ir::CallType::Extern,
                             ir::FunctionRef(),
                             0,
                             attrs);
  std::vector<Expr> mutable_args;
  // Call a function with multiple outputs.
  if (proto->ret_type.is_void()) {
    for (int i = 0; i < proto->mutable_arg_types.size(); i++) {
      auto shape = proto->shape_inference(args, i);
      auto op = ir::CallOp::Make(func_name, call);
      op->as<ir::CallOp>()->value_slot = i;
      op->as<ir::CallOp>()->is_tuple_get = true;
      auto name = cinn::UniqName("tuple_" + func_name + "_out" +
                                 std::to_string(i) + "_");
      auto ret =
          ir::Tensor(name, proto->mutable_arg_types[i], shape, shape, op, {});
      mutable_args.push_back(ret);
    }
    call.As<ir::Call>()->write_args = mutable_args;
  }
  return call;
}

}  // namespace lang
}  // namespace cinn
