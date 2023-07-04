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

#include "paddle/cinn/hlir/op/contrib/one_hot.h"

#include <gflags/gflags.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "paddle/cinn/common/cas.h"
#include "paddle/cinn/common/common.h"
#include "paddle/cinn/common/context.h"
#include "paddle/cinn/common/macros.h"
#include "paddle/cinn/hlir/framework/node.h"
#include "paddle/cinn/hlir/framework/op.h"
#include "paddle/cinn/hlir/framework/op_strategy.h"
#include "paddle/cinn/hlir/op/op_util.h"
#include "paddle/cinn/hlir/pe/ir_schedule_pe.h"
#include "paddle/cinn/hlir/pe/nn.h"
#include "paddle/cinn/hlir/pe/transform.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/ir/ir_schedule.h"
#include "paddle/cinn/ir/tensor.h"
#include "paddle/cinn/lang/builtin.h"
#include "paddle/cinn/lang/compute.h"

DECLARE_bool(cinn_ir_schedule);

namespace cinn {
namespace hlir {
namespace op {

using common::CINNValuePack;

ir::Tensor OneHot(const ir::Tensor& indices,
                  const ir::Tensor& on_value,
                  const ir::Tensor& off_value,
                  const int depth,
                  const int axis,
                  const Type& dtype,
                  const std::string& output_name) {
  int ndim = static_cast<int>(indices->shape.size());
  CHECK(axis == -1 || (0 <= axis && axis <= ndim))
      << "one_hot only accepts `axis` in [-1, data.ndim]"
      << ", but got axis = " << axis << ", and data.ndim = " << ndim;
  CHECK(depth > 0) << "one_hot only accepts `depth > 0`"
                   << ", but got depth = " << depth;

  CHECK(on_value->shape.size() == 1U && on_value->shape[0].as_int32() == 1U)
      << "The shape of on_value must be [1]";
  CHECK(off_value->shape.size() == 1U && off_value->shape[0].as_int32() == 1U)
      << "The shape of off_value must be [1]";

  int true_axis = (axis == -1) ? ndim : axis;
  std::vector<Expr> new_shape;
  int indices_index = 0;

  for (int i = 0; i < ndim + 1; ++i) {
    if (i == true_axis) {
      new_shape.push_back(Expr(depth));
    } else {
      new_shape.push_back(indices->shape[indices_index++]);
    }
  }

  Expr on_value_cast = ir::Cast::Make(dtype, on_value(Expr(0)));
  Expr off_value_cast = ir::Cast::Make(dtype, off_value(Expr(0)));

  ir::Tensor res = lang::Compute(
      new_shape,
      [=](const std::vector<Expr>& iter) {
        std::vector<Expr> indices_indices;

        for (size_t i = 0; i < iter.size(); i++) {
          if (static_cast<int>(i) == true_axis) {
            continue;
          }
          indices_indices.push_back(iter[i]);
        }

        Expr idx = iter[true_axis];
        Expr elem = ir::Cast::Make(idx.type(), indices(indices_indices));
        return ir::Select::Make(
            ir::EQ::Make(elem, idx), on_value_cast, off_value_cast);
      },
      common::UniqName(output_name));

  return res;
}

std::vector<framework::shape_t> InferShapeForOneHot(
    const std::vector<framework::shape_t>& inputs_shape,
    const framework::AttrMapType& attrs) {
  CHECK_EQ(inputs_shape.size(), 3UL)
      << "The number of one_hot's input should be 3";

  int depth;
  int axis;

  for (auto& iter : attrs) {
    if (iter.first == "depth") {
      depth = absl::get<int>(iter.second);
    } else if (iter.first == "axis") {
      axis = absl::get<int>(iter.second);
    }
  }

  const std::vector<int>& in_shape = inputs_shape[0];
  int ndim = static_cast<int>(in_shape.size());
  int true_axis = (axis == -1) ? in_shape.size() : axis;
  int indices_index = 0;
  std::vector<int> new_shape;

  for (int i = 0; i < ndim + 1; ++i) {
    if (i == true_axis) {
      new_shape.push_back(depth);
    } else {
      new_shape.push_back(in_shape[indices_index++]);
    }
  }

  std::vector<std::vector<int>> res{new_shape};
  return res;
}

std::vector<Type> InferDtypeForOneHot(const std::vector<Type>& inputs_type,
                                      const framework::AttrMapType& attrs) {
  CHECK(!inputs_type.empty())
      << "The input's type size is 0! Please check again.";

  std::string dtype = "float32";
  if (attrs.find("dtype") != attrs.end()) {
    dtype = absl::get<std::string>(attrs.at("dtype"));
  }

  std::vector<Type> res{common::Str2Type(dtype)};
  return res;
}

std::shared_ptr<framework::OpStrategy> StrategyForOneHot(
    const framework::NodeAttr& attrs,
    const std::vector<ir::Tensor>& inputs,
    const std::vector<Type>& out_type,
    const std::vector<std::vector<int>>& output_shapes,
    const Target& target) {
  int depth;
  int axis;
  std::string dtype = "float32";

  for (auto& iter : attrs.attr_store) {
    if (iter.first == "depth") {
      depth = absl::get<int>(iter.second);
    } else if (iter.first == "axis") {
      axis = absl::get<int>(iter.second);
    } else if (iter.first == "dtype") {
      dtype = absl::get<std::string>(iter.second);
    }
  }

  CHECK(depth > 0) << "one_hot only accepts `depth > 0`"
                   << ", but got depth = " << depth;

  framework::CINNCompute one_hot_compute([=](lang::Args args,
                                             lang::RetValue* ret) {
    CHECK(!args.empty())
        << "The input argument of one_hot compute is empty! Please check.\n";
    common::CINNValuePack pack_args = args[0];
    CHECK(!pack_args.empty())
        << "at least one input tensor for transpose compute\n";
    CHECK_GE(pack_args.size(), 3U);
    Expr indices_expr = pack_args[0];
    Expr on_value_expr = pack_args[1];
    Expr off_value_expr = pack_args[2];
    CHECK(indices_expr.as_tensor());
    CHECK(on_value_expr.as_tensor());
    CHECK(off_value_expr.as_tensor());

    ir::Tensor indices = indices_expr.as_tensor_ref();
    ir::Tensor on_value = on_value_expr.as_tensor_ref();
    ir::Tensor off_value = off_value_expr.as_tensor_ref();

    CHECK_EQ(pack_args.size(), 4U);
    std::string tensor_name = pack_args[3].operator std::string();

    ir::Tensor out = OneHot(indices,
                            on_value,
                            off_value,
                            depth,
                            axis,
                            common::Str2Type(dtype),
                            tensor_name);

    std::vector<common::CINNValue> res;
    auto stages = CreateStages({indices, on_value, off_value});
    stages->InsertLazily(out);
    res.push_back(common::CINNValue(out));
    res.push_back(common::CINNValue(stages));
    *ret = common::CINNValuePack{res};
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(one_hot_compute,
                    GetInjectiveScheduleFunc(output_shapes, target),
                    "strategy.one_hot.x86",
                    1);

  return strategy;
}
}  // namespace op
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(one_hot_ops) {
  CINN_REGISTER_OP(one_hot)
      .describe(
          "Returns a one-hot tensor where the locations repsented by indices "
          "take value `on_value`, "
          "other locations take value `off_value`.")
      .set_num_inputs(3)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>(
          "CINNStrategy", cinn::hlir::op::StrategyForOneHot)
      .set_attr("infershape",
                MakeOpFunction(cinn::hlir::op::InferShapeForOneHot))
      .set_attr("inferdtype",
                MakeOpFunction(cinn::hlir::op::InferDtypeForOneHot))
      .set_attr<cinn::hlir::framework::OpPatternKind>(
          "OpPattern", cinn::hlir::framework::OpPatternKind::kInjective)
      .set_support_level(4);

  return true;
}
