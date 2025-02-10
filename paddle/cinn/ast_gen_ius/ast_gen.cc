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

#include "paddle/cinn/ast_gen_ius/ast_gen.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/operation.h"
#include "paddle/cinn/ir/tensor.h"
#include "paddle/cinn/lang/compute.h"
#include "paddle/cinn/optim/replace_var_with_expr.h"

using cinn::ir::stmt::BlockRef;
using cinn::ir::stmt::For;
using cinn::ir::stmt::Schedule;
using cinn::ir::stmt::StmtRef;
using cinn::ir::stmt::Store;

namespace cinn {
namespace ast_gen_ius {

bool IsReduceBool(const ir::Expr& lhs, const ir::Expr& rhs) {
  return lhs.type().is_bool() || rhs.type().is_bool();
}

StmtRef ConvertReduceBody(ir::Expr body,
                          ir::Tensor tensor,
                          const std::vector<Expr>& axis_exprs) {
  ir::Reduce* reduce_node = body.As<ir::Reduce>();
  if (!reduce_node) {
    return Store(tensor, body, axis_exprs);
  }

  switch (reduce_node->reduce_type) {
    case ir::Reduce::kSum:
      if (IsReduceBool(tensor(axis_exprs), reduce_node->body)) {
        return Store(
            tensor, tensor(axis_exprs) || reduce_node->body, axis_exprs);
      }
      return Store(tensor, tensor(axis_exprs) + reduce_node->body, axis_exprs);
    case ir::Reduce::kMul:
      if (IsReduceBool(tensor(axis_exprs), reduce_node->body)) {
        return Store(
            tensor, tensor(axis_exprs) && reduce_node->body, axis_exprs);
      }
      return Store(tensor, tensor(axis_exprs) * reduce_node->body, axis_exprs);
    case ir::Reduce::kMax:
      return Store(tensor,
                   ir::Max::Make(tensor(axis_exprs), reduce_node->body),
                   axis_exprs);
    case ir::Reduce::kMin:
      return Store(tensor,
                   ir::Min::Make(tensor(axis_exprs), reduce_node->body),
                   axis_exprs);
    case ir::Reduce::kAll:
      return Store(tensor, tensor(axis_exprs) && reduce_node->body, axis_exprs);
    case ir::Reduce::kAny:
      return Store(tensor, tensor(axis_exprs) || reduce_node->body, axis_exprs);
    default:
      CINN_NOT_IMPLEMENTED
  }
}

StmtRef AstGen::Build(const ir::Tensor& tensor, TensorGroup* tensor_group) {
  const std::vector<ir::Var>& axis = tensor->axis();
  const std::vector<ir::Expr>& shape = tensor->shape;
  size_t axis_len = axis.size();
  PADDLE_ENFORCE_EQ(
      shape.size(),
      axis_len,
      ::common::errors::InvalidArgument("Internal Error: Tensor has different "
                                        "shape and axis length in AstGen"));
  std::vector<ir::Expr> axis_exprs;
  for (const auto& a : axis) {
    axis_exprs.push_back(a);
  }

  if (tensor->is_reduce_tensor()) {
    // Make an init Tensor for domain without reduce axis
    Expr init_value = tensor->GetReduceInitVal();
    // TODO(zhhsplendid): Clean the hardcoded "__reduce_init" string
    std::string reduce_init_name = tensor->name + "__reduce_init";
    const std::vector<Expr>& domain = tensor->domain_without_reduce_axis();
    ir::Tensor init_tensor = lang::Compute(
        domain,
        [=](const std::vector<Expr>& axis) { return init_value; },
        reduce_init_name);
    tensor_group->Insert(init_tensor);
    tensor_group->MarkShareMemBuffer(tensor, init_tensor);
    tensor_group->CtrlDepend(tensor, init_tensor);
    StmtRef init_body = Store(init_tensor, init_value, axis_exprs);
    // create schedule block itervars, i0,i1...
    std::vector<ir::Var> block_vars;
    std::vector<ir::Expr> iter_values;
    // reduce body and reduce init schedule block should have different objects
    // for same axis so we re-create objects
    std::vector<Var> axis_vars = cinn::common::GenDefaultAxis(axis_len);
    const std::vector<ir::Var>& reduce_axis = tensor->reduce_axis;
    VLOG(4) << "ast gen: tensor init_body is " << init_body;
    for (int i = 0; i < shape.size(); ++i) {
      block_vars.push_back(Var(Expr(0),
                               shape[i],
                               cinn::UniqName("i" + std::to_string(i)),
                               /*is_reduce = */ false));
      optim::ReplaceVarWithExpr(init_body, axis[i], block_vars.back());
      axis_vars[i]->is_reduce_axis = false;
      iter_values.push_back(axis_vars[i]);
      ir::TryElevateInt32ToInt64({ir::Expr(axis_vars[i]), shape[i]});
    }
    VLOG(4) << "iter_value.size() and block_vars.size() is "
            << iter_values.size() << " " << block_vars.size();
    init_body = Schedule(block_vars,
                         iter_values,
                         {},
                         {},
                         reduce_init_name,
                         BlockRef({init_body}));

    // For the remaining reduce axis, make reduce body
    StmtRef reduce_body = ConvertReduceBody(tensor->body(), tensor, axis_exprs);

    VLOG(4) << "ast gen: reduce body is " << reduce_body;

    // create schedule block itervars, i0,i1...
    std::vector<ir::Var> reduce_block_vars;
    std::vector<ir::Expr> reduce_iter_values;
    // reduce body and reduce init schedule block should have different objects
    // for same axis so we re-create objects
    std::vector<Var> reduce_axis_vars = cinn::common::GenDefaultAxis(axis_len);
    for (int i = 0; i < shape.size(); ++i) {
      reduce_block_vars.push_back(Var(Expr(0),
                                      shape[i],
                                      cinn::UniqName("i" + std::to_string(i)),
                                      /*is_reduce = */ false));
      reduce_axis_vars[i]->is_reduce_axis = false;
      reduce_iter_values.push_back(axis_vars[i]);
      ir::TryElevateInt32ToInt64({ir::Expr(axis_vars[i]), shape[i]});
    }
    VLOG(4) << "ast gen: reduce body is after replace 0" << reduce_body;
    for (int i = 0; i < reduce_axis.size(); ++i) {
      int count = shape.size() + i;
      reduce_block_vars.push_back(
          Var(reduce_axis[i]->lower_bound,
              reduce_axis[i]->upper_bound,
              cinn::UniqName("i" + std::to_string(count)),
              /*is_reduce = */ true));
      ir::Var reduce_axis_var = reduce_axis[i];
      reduce_axis_var->is_reduce_axis = true;
      reduce_iter_values.push_back(reduce_axis_var);
    }

    int non_zero_axis_size = 0;
    std::vector<ir::Var> non_reduce_axis_vars = [&]() {
      std::vector<ir::Var> res;
      for (int i = 0; i < shape.size(); ++i) {
        res.push_back(axis[i]);
      }
      return res;
    }();
    for (int i = 0; i < non_reduce_axis_vars.size(); ++i) {
      optim::ReplaceVarWithExpr(
          reduce_body, non_reduce_axis_vars[i], reduce_block_vars[i]);
      ++non_zero_axis_size;
    }

    VLOG(4) << "to replace : " << non_zero_axis_size << " "
            << reduce_block_vars.size();
    for (auto i = 0; i < reduce_block_vars.size(); i++) {
      VLOG(4) << "reduce_block_vars[" << i << "] = " << reduce_block_vars[i];
    }
    for (auto i = 0; i < reduce_axis.size(); i++) {
      VLOG(4) << "reduce_axis[" << i << "] = " << reduce_axis[i];
    }
    VLOG(4) << "before replace body: " << reduce_body;
    for (int i = non_zero_axis_size; i < reduce_block_vars.size(); ++i) {
      optim::ReplaceVarWithExpr(reduce_body,
                                reduce_axis[i - non_zero_axis_size],
                                reduce_block_vars[i]);
    }

    reduce_body = Schedule(reduce_block_vars,
                           reduce_iter_values,
                           {},
                           {},
                           tensor->name,
                           BlockRef({reduce_body}));
    for (int i = static_cast<int>(reduce_axis.size()) - 1; i >= 0; --i) {
      ir::TryElevateInt32ToInt64({reduce_axis[i],
                                  reduce_axis[i]->lower_bound,
                                  reduce_axis[i]->upper_bound});
      reduce_body = For(reduce_axis[i],
                        reduce_axis[i]->lower_bound,
                        reduce_axis[i]->upper_bound,
                        ir::ForType::Serial,
                        ir::DeviceAPI::Host,
                        BlockRef({reduce_body}));
    }

    // Put the two parts together
    std::vector<StmtRef> block_body{init_body, reduce_body};
    for (int i = static_cast<int>(axis_len) - 1; i >= 0; --i) {
      ir::Var loop_var = axis[i];
      ir::Expr loop_extent = shape[i];
      block_body = std::vector<StmtRef>{For(loop_var,
                                            Expr(0),
                                            loop_extent,
                                            ir::ForType::Serial,
                                            ir::DeviceAPI::Host,
                                            BlockRef(block_body))};
    }
    return block_body[0];
  } else {
    StmtRef body = Store(tensor, tensor->body(), axis_exprs);
    // create schedule block itervars, i0,i1...
    std::vector<ir::Var> block_vars;
    std::vector<ir::Expr> iter_values;
    std::vector<Var> axis_vars = cinn::common::GenDefaultAxis(axis_len);
    for (int i = 0; i < shape.size(); ++i) {
      block_vars.push_back(Var(
          Expr(0), shape[i], cinn::UniqName("i" + std::to_string(i)), false));
      optim::ReplaceVarWithExpr(body, axis[i], block_vars[i]);
      axis_vars[i]->is_reduce_axis = false;
      iter_values.push_back(axis_vars[i]);
    }
    body = Schedule(
        block_vars, iter_values, {}, {}, tensor->name, BlockRef({body}));
    for (int i = static_cast<int>(axis_len) - 1; i >= 0; --i) {
      ir::Var loop_var = axis[i];
      ir::Expr loop_extent = shape[i];
      body = For(loop_var,
                 Expr(0),
                 loop_extent,
                 ir::ForType::Serial,
                 ir::DeviceAPI::Host,
                 BlockRef({body}));
    }
    return body;
  }
}

}  // namespace ast_gen_ius
}  // namespace cinn
