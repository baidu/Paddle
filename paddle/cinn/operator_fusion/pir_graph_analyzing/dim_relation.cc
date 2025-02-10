// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/cinn/operator_fusion/pir_graph_analyzing/dim_relation.h"
#include "paddle/cinn/hlir/dialect/operator/ir/cinn_op.h"

#include "paddle/common/enforce.h"

namespace cinn::fusion {

ValueUsage GetValueUsage(const pir::Value& v, const size_t usage_idx) {
  ValueUsage value_dim;
  size_t rank = GetRank(v);
  for (size_t i = 0; i < rank; ++i) {
    value_dim.emplace_back(v, i, usage_idx);
  }
  return value_dim;
}

static std::vector<ValueUsage> GetInputValueUsage(pir::Operation* op) {
  std::vector<ValueUsage> value_dims;
  for (const auto& v : op->operands()) {
    value_dims.emplace_back(
        GetValueUsage(v.source(), GetUsageIdx(v.source(), op)));
  }
  return value_dims;
}

static std::vector<std::vector<ValueUsage>> GetOutputValueUsage(
    pir::Operation* op) {
  std::vector<std::vector<ValueUsage>> result;
  for (const auto& v : op->results()) {
    std::vector<ValueUsage> single_output_value_dim;
    for (size_t i = 0; i < v.use_count(); i++) {
      single_output_value_dim.emplace_back(GetValueUsage(v, i));
    }
    result.emplace_back(single_output_value_dim);
  }
  return result;
}

static std::vector<DimUsage> GetAllInputDimUsage(pir::Operation* op) {
  return ConcatAll(GetInputValueUsage(op));
}

static std::vector<DimUsage> GetAllOutputDimUsage(pir::Operation* op) {
  return ConcatAll(ConcatAll(GetOutputValueUsage(op)));
}

static DimUsageRelation CreateOpRelativenessForDefault(pir::Operation* op) {
  DimUsageRelation res;
  const std::vector<DimUsage>& input_single_dims = GetAllInputDimUsage(op);
  const std::vector<DimUsage>& output_single_dims = GetAllOutputDimUsage(op);

  for (const auto& in_dim : input_single_dims) {
    for (const auto& out_dim : output_single_dims) {
      res[in_dim][out_dim] = true;
    }
  }
  return res;
}

static DimUsageRelation CreateOpRelativenessForElementWise(pir::Operation* op) {
  DimUsageRelation res;
  const std::vector<ValueUsage>& input_value_dims = GetInputValueUsage(op);
  const std::vector<ValueUsage>& output_value_dims =
      ConcatAll(GetOutputValueUsage(op));

  for (auto in_value_dim : input_value_dims) {
    for (auto out_value_dim : output_value_dims) {
      PADDLE_ENFORCE_EQ(
          in_value_dim.size(),
          out_value_dim.size(),
          ::common::errors::PreconditionNotMet(
              "Required in_value_dim and out_value_dim have same size."));
      for (int i = 0; i < in_value_dim.size(); ++i) {
        res[in_value_dim[i]][out_value_dim[i]] = true;
      }
    }
  }
  return res;
}

static DimUsageRelation CreateOpRelativenessForBroadcast(pir::Operation* op) {
  DimUsageRelation res;
  const auto& in_value = op->operand(0).source();
  const auto& out_value = op->result(0);
  size_t usage_idx = GetUsageIdx(in_value, op);
  for (const auto& t : GetNonBroadCastDims(op)) {
    for (size_t i = 0; i < out_value.use_count(); i++) {
      res[DimUsage(in_value, t.first, usage_idx)]
         [DimUsage(out_value, t.second, i)] = true;
    }
  }
  return res;
}

static DimUsageRelation CreateOpRelativenessForReduce(pir::Operation* op) {
  const auto& reduce_axis_idx = GetReduceAxisIdx(op);
  DimUsageRelation res;
  const size_t input_rank = GetCompatibleRank(op->operand_source(0));
  int out_idx = 0;
  bool keep_dim = GetReduceOpKeepDims(op);
  for (size_t i = 0; i < input_rank; i++) {
    if (std::find(reduce_axis_idx.begin(), reduce_axis_idx.end(), i) ==
        reduce_axis_idx.end()) {
      auto input_dim = DimUsage(
          op->operand_source(0), i, GetUsageIdx(op->operand_source(0), op));
      for (size_t j = 0; j < op->result(0).use_count(); ++j) {
        res[input_dim][DimUsage(op->result(0), out_idx, j)] = true;
      }
      out_idx += 1;
    } else {
      out_idx += keep_dim;
    }
  }
  return res;
}

static std::optional<DimUsageRelation> CreateOpRelativenessForSpecialOps(
    pir::Operation* op) {
  if (op->num_results() != 1) {
    VLOG(4) << "Now we do not support op with multi outputs, use default: "
            << op->name();
    return CreateOpRelativenessForDefault(op);
  }
  if (op->name() == "cinn_op.reshape") {
    // Special Elementwise.
    return CreateOpRelativenessForDefault(op);
  }
  if (op->name() == "pd_op.reshape") {
    // Special Elementwise.
    return CreateOpRelativenessForDefault(op);
  }
  if (op->name() == "cinn_op.generate_shape") {
    return CreateOpRelativenessForDefault(op);
  }
  return {};
}

static DimUsageRelation GetSingleOpRelation(pir::Operation* op) {
  const auto& special_result = CreateOpRelativenessForSpecialOps(op);
  if (special_result != std::nullopt) {
    VLOG(5) << "[DimUsageRelation] GetSingleOpRelation for special op: \n"
            << op->name() << " : " << RelationDebugStr(special_result.value());
    return special_result.value();
  }

  const hlir::framework::OpPatternKind kind = GetOpPatternKind(op);
  DimUsageRelation result;
  if (kind == hlir::framework::kReduction) {
    result = CreateOpRelativenessForReduce(op);
  } else if (kind == hlir::framework::kElementWise) {
    result = CreateOpRelativenessForElementWise(op);
  } else if (kind == hlir::framework::kBroadcast) {
    result = CreateOpRelativenessForBroadcast(op);
  } else {
    result = CreateOpRelativenessForDefault(op);
  }
  VLOG(5) << "[DimUsageRelation] GetSingleOpRelation: \n"
          << op->name() << " : " << RelationDebugStr(result);
  return result;
}

static std::vector<std::pair<DimUsage, DimUsage>> FlattenRelation(
    const DimUsageRelation& axes_relation) {
  std::vector<std::pair<DimUsage, DimUsage>> res;
  for (const auto& in_dim_pair : axes_relation) {
    for (const auto& out_dim_pair : in_dim_pair.second) {
      res.emplace_back(in_dim_pair.first, out_dim_pair.first);
    }
  }
  return res;
}

DimUsageRelation AnalysisIndexExprRelation(
    const std::vector<pir::Operation*>& ops) {
  DimUsageRelation res;

  for (size_t i = ops.size(); i >= 1; --i) {
    pir::Operation* op = ops[i - 1];
    if (op->name() == "cf.yield") continue;

    const auto& value_dim_relation = GetSingleOpRelation(op);
    for (const auto& in_out_pair : FlattenRelation(value_dim_relation)) {
      for (const auto& out_relation : res[in_out_pair.second]) {
        res[in_out_pair.first][out_relation.first] = true;
      }
      res[in_out_pair.first][in_out_pair.second] = true;
    }
  }

  VLOG(4) << "[AnalysisIndexExprRelation] Result " << RelationDebugStr(res);
  return res;
}

std::string RelationDebugStr(const DimUsageRelation& relation) {
  std::stringstream ss;
  ss << "DimUsage Relation:\n";
  for (const auto& [src, dsts] : relation) {
    ss << src.DebugStr() << " \n Related To: -> \n";
    for (const auto& [dst, _boolean] : dsts) {
      ss << dst.DebugStr() << "\n";
    }
    ss << "\n";
  }
  return ss.str();
}

}  // namespace cinn::fusion
