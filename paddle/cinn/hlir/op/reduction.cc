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

#include "paddle/cinn/hlir/pe/reduction.h"

#include <algorithm>
#include <iostream>
#include <vector>

#include "paddle/cinn/adt/op_equation_context.h"
#include "paddle/cinn/hlir/framework/op.h"
#include "paddle/cinn/hlir/framework/op_strategy.h"
#include "paddle/cinn/hlir/pe/broadcast.h"
#include "paddle/cinn/hlir/pe/ir_schedule_pe.h"
#include "paddle/cinn/hlir/pe/schedule.h"
#include "paddle/cinn/hlir/pe/transform.h"
#include "paddle/cinn/ir/op/ir_operators.h"
#include "paddle/cinn/ir/schedule/ir_schedule.h"
#include "paddle/cinn/optim/ir_simplify_pass.h"
#include "paddle/cinn/runtime/flags.h"

PD_DECLARE_bool(cinn_enable_map_expr);

namespace cinn {
namespace hlir {
namespace op {
using cinn::common::_CINNValuePack_;
using cinn::common::CINNValue;
using cinn::common::CINNValuePack;
using framework::OpStrategy;
using framework::shape_t;
using framework::StrategyFunction;

using BlockReduceFunc =
    std::function<std::vector<ir::Tensor>(const ir::Tensor &,
                                          const std::vector<int> &,
                                          const bool,
                                          const std::string &)>;
using ReduceFunc = std::function<ir::Tensor(const ir::Tensor &,
                                            const std::vector<int> &,
                                            const bool,
                                            const std::string &)>;

std::shared_ptr<OpStrategy> StrategyForReduce(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<int>> &output_shapes,
    const Target &target,
    const std::string &op_name,
    BlockReduceFunc gpu_reduce_with_last_axis_func,
    BlockReduceFunc gpu_reduce_without_last_axis_func,
    ReduceFunc common_reduce_func) {
  std::vector<int> reduce_axes;
  auto ndim = inputs[0]->shape.size();
  if (attrs.attr_store.count("axis")) {
    reduce_axes = [&] {
      if (absl::holds_alternative<std::vector<int64_t>>(
              attrs.attr_store.at("axis"))) {
        const auto &dim_attr =
            absl::get<std::vector<int64_t>>(attrs.attr_store.at("axis"));
        return std::vector<int>(dim_attr.begin(), dim_attr.end());
      } else if (absl::holds_alternative<std::vector<int>>(
                     attrs.attr_store.at("axis"))) {
        return absl::get<std::vector<int>>(attrs.attr_store.at("axis"));
      } else if (absl::holds_alternative<bool>(attrs.attr_store.at("axis"))) {
        return std::vector<int>{};
      } else {
        PADDLE_THROW(::common::errors::InvalidArgument(
            "reduce dimension's type is invalid!"));
      }
    }();
    if (reduce_axes.empty()) {
      for (int i = 0; i < ndim; ++i) {
        reduce_axes.push_back(i);
      }
    } else {
      std::for_each(reduce_axes.begin(), reduce_axes.end(), [&ndim](int &x) {
        if (x < 0) x += ndim;
      });
    }
    std::sort(reduce_axes.begin(), reduce_axes.end());
    // check reduce_axes
    PADDLE_ENFORCE_LE(
        reduce_axes.size(),
        ndim,
        ::common::errors::InvalidArgument(
            "The reduce axes size %d should be less than or equal "
            "to the input tensor's dimension %d.",
            reduce_axes.size(),
            ndim));
    PADDLE_ENFORCE_LE(
        reduce_axes.back(),
        ndim,
        ::common::errors::InvalidArgument(
            "The reduce axes size %d should be less than or equal "
            "to the input tensor's dimension %d.",
            reduce_axes.back(),
            ndim));
    for (int idx = 1; idx < reduce_axes.size(); ++idx) {
      PADDLE_ENFORCE_NE(reduce_axes[idx - 1],
                        reduce_axes[idx],
                        ::common::errors::InvalidArgument(
                            "The reduce axes should be unique!"));
    }
  } else {
    PADDLE_THROW(
        ::common::errors::InvalidArgument("reduce dimension is not set!"));
  }

  bool keepdim = false;
  if (attrs.attr_store.count("keepdim")) {
    keepdim = absl::get<bool>(attrs.attr_store.at("keepdim"));
  }

  framework::CINNCompute reduction_compute([=](lang::Args args,
                                               lang::RetValue *ret) {
    PADDLE_ENFORCE_EQ(
        !args.empty(),
        true,
        ::common::errors::InvalidArgument(
            "The input argument of %s compute is empty! Please check.",
            op_name));
    CINNValuePack arg_packs = args[0];
    PADDLE_ENFORCE_EQ(
        arg_packs.size(),
        2U,
        ::common::errors::InvalidArgument(
            "There should be 2 input args for %s compute", op_name));
    PADDLE_ENFORCE_EQ(arg_packs[1].is_string(),
                      true,
                      ::common::errors::InvalidArgument(
                          "The arg_packs[1] is not empty! Please check."));
    std::string tensor_name = arg_packs[1].operator std::string();
    Expr x_expr = arg_packs[0];
    PADDLE_ENFORCE_NOT_NULL(x_expr.as_tensor(),
                            ::common::errors::InvalidArgument(
                                "The x_expr can not as tensor! Please check."));
    ir::Tensor x = x_expr.as_tensor_ref();

    std::unordered_set<std::string> bool_reduce_op = {"reduce_all",
                                                      "reduce_any"};
    PADDLE_ENFORCE_EQ(!bool_reduce_op.count(op_name) || x->type().is_bool(),
                      true,
                      ::common::errors::InvalidArgument(
                          "The type of input argument %s of %s should be bool, "
                          "but get %s! Please check.",
                          x->name,
                          op_name,
                          x->type().to_string()));

    const auto &NaiveCompute = [&]() {
      VLOG(3) << "Do Reduce Compute!";
      auto out = common_reduce_func(x, reduce_axes, keepdim, tensor_name);

      std::vector<CINNValue> cinn_values{CINNValue(out)};
      *ret = CINNValuePack{cinn_values};
    };
    target.arch.Match(
        [&](common::NVGPUArch) { NaiveCompute(); },
        [&](std::variant<common::UnknownArch,
                         common::X86Arch,
                         common::ARMArch>) { NaiveCompute(); },
        [&](std::variant<common::HygonDCUArchHIP, common::HygonDCUArchSYCL>) {
          NaiveCompute();
        });
  });

  framework::CINNSchedule reduction_schedule([=](lang::Args args,
                                                 lang::RetValue *ret) {
    PADDLE_ENFORCE_EQ(!args.empty(),
                      true,
                      ::common::errors::InvalidArgument(
                          "The input argument of schedule is empty! Please "
                          "check."));

    CINNValuePack arg_pack = args[0];
    PADDLE_ENFORCE_GE(arg_pack.size(),
                      2UL,
                      ::common::errors::InvalidArgument(
                          "The input tensor size should be greater than 2!"));
    PADDLE_ENFORCE_LE(arg_pack.size(),
                      8UL,
                      ::common::errors::InvalidArgument(
                          "The input tensor size should be less than 8!"));
    std::vector<Expr> vec_ast;
    std::vector<Expr> vec_tensor;
    for (int i = 0; i < arg_pack.size(); i++) {
      if (arg_pack[i].is_expr()) {
        Expr temp = arg_pack[i];
        // TODO(zhhsplendid): old reduction schedule assumes all length-1
        // for loops are simplified, but it is not after we add length-1
        // back. Reduction schedule is complex and we haven't changed it to
        // support the length-1 for loop yet. So we simplify here. The todo
        // is that remove SimplifyForLoops below and change reduction schedule
        optim::SimplifyForLoops(&temp);
        vec_ast.emplace_back(temp);
      } else if (arg_pack[i].is_tensor()) {
        Expr temp = arg_pack[i];
        vec_tensor.emplace_back(temp);
      }
    }
    PADDLE_ENFORCE_EQ(!vec_ast.empty(),
                      true,
                      ::common::errors::InvalidArgument(
                          "The input argument of schedule is empty! Please "
                          "check."));
    ir::ModuleExpr mod_expr(vec_ast);
    ir::IRSchedule ir_sch(mod_expr);
    ir_sch.MergeExprs();
    target.arch.Match(
        [&](common::UnknownArch) { CINN_NOT_IMPLEMENTED; },
        [&](common::X86Arch) {
          std::vector<CINNValue> res{
              CINNValue(ir_sch.GetModule().GetExprs().at(0))};
          *ret = CINNValuePack{res};
        },
        [&](common::ARMArch) {
          std::vector<CINNValue> res{
              CINNValue(ir_sch.GetModule().GetExprs().at(0))};
          *ret = CINNValuePack{res};
        },
        [&](common::NVGPUArch) {
          std::vector<CINNValue> res{
              CINNValue(ir_sch.GetModule().GetExprs().at(0))};
          *ret = CINNValuePack{res};
        },
        [&](std::variant<common::HygonDCUArchHIP, common::HygonDCUArchSYCL>) {
          std::vector<CINNValue> res{
              CINNValue(ir_sch.GetModule().GetExprs().at(0))};
          *ret = CINNValuePack{res};
        });
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(
      reduction_compute, reduction_schedule, "strategy." + op_name + ".x86", 1);

  return strategy;
}

std::shared_ptr<OpStrategy> StrategyForReduceSymbolic(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<ir::Dim>> &output_shapes,
    const Target &target,
    const std::string &op_name,
    BlockReduceFunc gpu_reduce_with_last_axis_func,
    BlockReduceFunc gpu_reduce_without_last_axis_func,
    ReduceFunc common_reduce_func) {
  std::vector<int> reduce_axes;
  auto ndim = inputs[0]->shape.size();
  if (attrs.attr_store.count("axis")) {
    reduce_axes = [&] {
      if (absl::holds_alternative<std::vector<int64_t>>(
              attrs.attr_store.at("axis"))) {
        const auto &dim_attr =
            absl::get<std::vector<int64_t>>(attrs.attr_store.at("axis"));
        return std::vector<int>(dim_attr.begin(), dim_attr.end());
      } else if (absl::holds_alternative<std::vector<int>>(
                     attrs.attr_store.at("axis"))) {
        return absl::get<std::vector<int>>(attrs.attr_store.at("axis"));
      } else if (absl::holds_alternative<bool>(attrs.attr_store.at("axis"))) {
        return std::vector<int>{};
      } else {
        PADDLE_THROW(::common::errors::InvalidArgument(
            "reduce dimension's type is invalid!"));
      }
    }();
    if (reduce_axes.empty()) {
      for (int i = 0; i < ndim; ++i) {
        reduce_axes.push_back(i);
      }
    } else {
      std::for_each(reduce_axes.begin(), reduce_axes.end(), [&ndim](int &x) {
        if (x < 0) x += ndim;
      });
    }
    std::sort(reduce_axes.begin(), reduce_axes.end());
    PADDLE_ENFORCE_LE(
        reduce_axes.size(),
        ndim,
        ::common::errors::InvalidArgument(
            "The reduce axes size %d should be less than or equal "
            "to the input tensor's dimension %d.",
            reduce_axes.size(),
            ndim));
    PADDLE_ENFORCE_LT(reduce_axes.back(),
                      ndim,
                      ::common::errors::InvalidArgument(
                          "The reduce axes back %d should be less than "
                          "to the input tensor's dimension %d.",
                          reduce_axes.back(),
                          ndim));
    for (int idx = 1; idx < reduce_axes.size(); ++idx) {
      PADDLE_ENFORCE_NE(reduce_axes[idx - 1],
                        reduce_axes[idx],
                        ::common::errors::InvalidArgument(
                            "The reduce axes should be unique!"));
    }
  } else {
    PADDLE_THROW(
        ::common::errors::InvalidArgument("reduce dimension is not set!"));
  }

  bool keepdim = false;
  if (attrs.attr_store.count("keepdim")) {
    keepdim = absl::get<bool>(attrs.attr_store.at("keepdim"));
  }

  framework::CINNCompute reduction_compute([=](lang::Args args,
                                               lang::RetValue *ret) {
    PADDLE_ENFORCE_EQ(
        !args.empty(),
        true,
        ::common::errors::InvalidArgument(
            "The input argument of compute is empty! Please check."));
    CINNValuePack arg_packs = args[0];
    PADDLE_ENFORCE_EQ(arg_packs.size(),
                      2U,
                      ::common::errors::InvalidArgument(
                          "There should be 2 input args for compute"));
    PADDLE_ENFORCE_EQ(arg_packs[1].is_string(),
                      true,
                      ::common::errors::InvalidArgument(
                          "The arg_packs[1] is not empty! Please check."));
    std::string tensor_name = arg_packs[1].operator std::string();
    Expr x_expr = arg_packs[0];
    PADDLE_ENFORCE_NOT_NULL(x_expr.as_tensor(),
                            ::common::errors::InvalidArgument(
                                "The x_expr can not as tensor! Please check."));

    ir::Tensor x = x_expr.as_tensor_ref();

    std::unordered_set<std::string> bool_reduce_op = {"reduce_all",
                                                      "reduce_any"};
    PADDLE_ENFORCE_EQ(!bool_reduce_op.count(op_name) || x->type().is_bool(),
                      true,
                      ::common::errors::InvalidArgument(
                          "The type of input argument should be bool, "
                          "Please check."));

    VLOG(3) << "Do Reduce Compute!";
    auto out = common_reduce_func(x, reduce_axes, keepdim, tensor_name);

    std::vector<CINNValue> cinn_values{CINNValue(out)};
    *ret = CINNValuePack{cinn_values};
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(
      reduction_compute, lang::PackedFunc(), "strategy." + op_name + ".x86", 1);

  return strategy;
}

#define STRATEGY_FOR_REDUCE(op_name_,                           \
                            reduce_op_,                         \
                            gpu_reduce_with_last_axis_func,     \
                            gpu_reduce_without_last_axis_func,  \
                            common_reduce_func)                 \
  std::shared_ptr<OpStrategy> StrategyFor##reduce_op_(          \
      const framework::NodeAttr &attrs,                         \
      const std::vector<ir::Tensor> &inputs,                    \
      const std::vector<Type> &out_type,                        \
      const std::vector<std::vector<int>> &output_shapes,       \
      const Target &target) {                                   \
    return StrategyForReduce(attrs,                             \
                             inputs,                            \
                             out_type,                          \
                             output_shapes,                     \
                             target,                            \
                             #op_name_,                         \
                             gpu_reduce_with_last_axis_func,    \
                             gpu_reduce_without_last_axis_func, \
                             common_reduce_func);               \
  }

#define STRATEGY_FOR_REDUCE_SYMBOLIC(op_name_,                          \
                                     reduce_op_,                        \
                                     gpu_reduce_with_last_axis_func,    \
                                     gpu_reduce_without_last_axis_func, \
                                     common_reduce_func)                \
  std::shared_ptr<OpStrategy> StrategyFor##reduce_op_##Symbolic(        \
      const framework::NodeAttr &attrs,                                 \
      const std::vector<ir::Tensor> &inputs,                            \
      const std::vector<Type> &out_type,                                \
      const std::vector<std::vector<ir::Dim>> &output_shapes,           \
      const Target &target) {                                           \
    return StrategyForReduceSymbolic(attrs,                             \
                                     inputs,                            \
                                     out_type,                          \
                                     output_shapes,                     \
                                     target,                            \
                                     #op_name_,                         \
                                     gpu_reduce_with_last_axis_func,    \
                                     gpu_reduce_without_last_axis_func, \
                                     common_reduce_func);               \
  }

STRATEGY_FOR_REDUCE(reduce_sum,
                    ReduceSum,
                    pe::TwoStepBlockReduceSum,
                    pe::BlockShuffleReduceSum,
                    pe::ReduceSum);
STRATEGY_FOR_REDUCE(reduce_prod,
                    ReduceProd,
                    pe::TwoStepBlockReduceProd,
                    pe::BlockShuffleReduceProd,
                    pe::ReduceProd);
STRATEGY_FOR_REDUCE(reduce_max,
                    ReduceMax,
                    pe::TwoStepBlockReduceMax,
                    pe::BlockShuffleReduceMax,
                    pe::ReduceMax);
STRATEGY_FOR_REDUCE(reduce_min,
                    ReduceMin,
                    pe::TwoStepBlockReduceMin,
                    pe::BlockShuffleReduceMin,
                    pe::ReduceMin);
STRATEGY_FOR_REDUCE(reduce_all,
                    ReduceAll,
                    pe::TwoStepBlockReduceAll,
                    pe::BlockShuffleReduceAll,
                    pe::ReduceAll);
STRATEGY_FOR_REDUCE(reduce_any,
                    ReduceAny,
                    pe::TwoStepBlockReduceAny,
                    pe::BlockShuffleReduceAny,
                    pe::ReduceAny);

STRATEGY_FOR_REDUCE_SYMBOLIC(reduce_sum,
                             ReduceSum,
                             pe::TwoStepBlockReduceSum,
                             pe::BlockShuffleReduceSum,
                             pe::ReduceSum);
STRATEGY_FOR_REDUCE_SYMBOLIC(reduce_prod,
                             ReduceProd,
                             pe::TwoStepBlockReduceProd,
                             pe::BlockShuffleReduceProd,
                             pe::ReduceProd);
STRATEGY_FOR_REDUCE_SYMBOLIC(reduce_max,
                             ReduceMax,
                             pe::TwoStepBlockReduceMax,
                             pe::BlockShuffleReduceMax,
                             pe::ReduceMax);
STRATEGY_FOR_REDUCE_SYMBOLIC(reduce_min,
                             ReduceMin,
                             pe::TwoStepBlockReduceMin,
                             pe::BlockShuffleReduceMin,
                             pe::ReduceMin);
STRATEGY_FOR_REDUCE_SYMBOLIC(reduce_all,
                             ReduceAll,
                             pe::TwoStepBlockReduceAll,
                             pe::BlockShuffleReduceAll,
                             pe::ReduceAll);
STRATEGY_FOR_REDUCE_SYMBOLIC(reduce_any,
                             ReduceAny,
                             pe::TwoStepBlockReduceAny,
                             pe::BlockShuffleReduceAny,
                             pe::ReduceAny);

#undef STRATEGY_FOR_REDUCE
#undef STRATEGY_FOR_REDUCE_SYMBOLIC

}  // namespace op
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(reduce_ops) {
#define CINN_REGISTER_REDUCTION_WITH_DTYPE(op__, op_strategy__, dtype__) \
  CINN_REGISTER_OP(op__)                                                 \
      .describe(#op__ " function")                                       \
      .set_num_inputs(1)                                                 \
      .set_num_outputs(1)                                                \
      .set_attr<cinn::hlir::framework::StrategyFunction>(                \
          "CINNStrategy", cinn::hlir::op::StrategyFor##op_strategy__)    \
      .set_attr<cinn::hlir::framework::StrategyFunctionSymbolic>(        \
          "CINNStrategySymbolic",                                        \
          cinn::hlir::op::StrategyFor##op_strategy__##Symbolic)          \
      .set_attr<cinn::hlir::framework::OpPatternKind>(                   \
          "OpPattern", cinn::hlir::framework::OpPatternKind::kReduction) \
      .set_support_level(4);

#define CINN_REGISTER_REDUCTION(op__, op_strategy__) \
  CINN_REGISTER_REDUCTION_WITH_DTYPE(op__, op_strategy__, )

  CINN_REGISTER_REDUCTION(reduce_sum, ReduceSum);
  CINN_REGISTER_REDUCTION(reduce_prod, ReduceProd);
  CINN_REGISTER_REDUCTION(reduce_max, ReduceMax);
  CINN_REGISTER_REDUCTION(reduce_min, ReduceMin);

#undef CINN_REGISTER_REDUCTION

  CINN_REGISTER_REDUCTION_WITH_DTYPE(reduce_all, ReduceAll, Bool);
  CINN_REGISTER_REDUCTION_WITH_DTYPE(reduce_any, ReduceAny, Bool);

#undef CINN_REGISTER_REDUCTION_WITH_DTYPE

  return true;
}
