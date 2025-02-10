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

#include <typeinfo>
#include <unordered_map>
#include <variant>

#include "glog/logging.h"
#include "paddle/cinn/adt/equation.h"
#include "paddle/cinn/adt/equation_solver.h"
#include "paddle/cinn/adt/equation_value.h"
#include "paddle/cinn/adt/index_expr_infer_context.h"
#include "paddle/cinn/adt/print.h"
#include "paddle/cinn/adt/simplify_value.h"
#include "paddle/cinn/adt/tags.h"
#include "paddle/cinn/common/equation_graph_topo_walker.h"
#include "paddle/common/enforce.h"

namespace cinn::adt {

std::unordered_map<Variable, Value> InferValuesImpl(
    const Identity<tOut<Iterator>, tIn<Iterator>>& id,
    IndexExprInferContext* ctx) {
  const auto& [out_iter, in_iter] = id.tuple();
  Variable in_variable{in_iter.value()};
  PADDLE_ENFORCE_EQ(
      ctx->HasValue(in_variable),
      true,
      ::common::errors::NotFound("The param id's out_iter must contain "
                                 "its in_iter's value"));
  return {{out_iter.value(), ctx->GetValue(in_variable)}};
}

std::unordered_map<Variable, Value> InferValuesImpl(
    const Identity<tOut<Index>, tIn<Index>>& id, IndexExprInferContext* ctx) {
  const auto& [out_index, in_index] = id.tuple();
  Variable in_variable{in_index.value()};
  PADDLE_ENFORCE_EQ(
      ctx->HasValue(in_variable),
      true,
      ::common::errors::NotFound("The param id's out_iter must contain "
                                 "its in_iter's value"));
  return {{out_index.value(), ctx->GetValue(in_variable)}};
}

namespace {
template <bool IsSameType>
struct IsReplicatedSymbolicValuesStruct;

template <>
struct IsReplicatedSymbolicValuesStruct<false> {
  template <typename T0, typename T1>
  static bool Call(const T0& lhs, const T1& rhs) {
    return false;
  }
};

template <>
struct IsReplicatedSymbolicValuesStruct<true> {
  static bool Call(const DimExpr& lhs, const DimExpr& rhs) { return false; }

  static bool Call(const Undefined& lhs, const Undefined& rhs) { return false; }
  static bool Call(const Ok& lhs, const Ok& rhs) { return false; }

  static bool Call(const Iterator& lhs, const Iterator& rhs) {
    return lhs == rhs;
  }

  static bool Call(const List<Value>& lhs, const List<Value>& rhs) {
    return lhs == rhs;
  }

  static bool Call(const IndexDotValue<Value, List<DimExpr>>& lhs,
                   const IndexDotValue<Value, List<DimExpr>>& rhs) {
    return lhs == rhs;
  }

  static bool Call(const IndexUnDotValue<Value, List<DimExpr>>& lhs,
                   const IndexUnDotValue<Value, List<DimExpr>>& rhs) {
    return lhs == rhs;
  }

  static bool Call(const ListGetItem<Value, DimExpr>& lhs,
                   const ListGetItem<Value, DimExpr>& rhs) {
    return lhs == rhs;
  }

  static bool Call(const BroadcastedIterator<Value, DimExpr>& lhs,
                   const BroadcastedIterator<Value, DimExpr>& rhs) {
    return lhs == rhs;
  }

  static bool Call(const PtrGetItem<Value>& lhs, const PtrGetItem<Value>& rhs) {
    return lhs == rhs;
  }
};

}  // namespace

bool IsReplicatedSymbolicValues(const Value& lhs, const Value& rhs) {
  return std::visit(
      [&](const auto& lhs, const auto& rhs) {
        return IsReplicatedSymbolicValuesStruct<
            std::is_same_v<std::decay_t<decltype(lhs)>,
                           std::decay_t<decltype(rhs)>>>::Call(lhs, rhs);
      },
      lhs.variant(),
      rhs.variant());
}

bool HasReplicatedSymbolicValues(const List<Value>& values) {
  for (std::size_t i = 0; i < values->size(); ++i) {
    for (std::size_t j = i + 1; j < values->size(); ++j) {
      if (IsReplicatedSymbolicValues(values->at(i), values->at(j))) {
        return true;
      }
    }
  }
  return false;
}

std::unordered_map<Variable, Value> InferValuesImpl(
    const IndexDot<List<DimExpr>, tOut<Index>, tIn<List<Iterator>>>& dot,
    IndexExprInferContext* ctx) {
  const auto& [dims, out_index, in_iters] = dot.tuple();
  List<Value> in_values;
  for (const auto& iter : *in_iters.value()) {
    in_values->emplace_back(ctx->GetValue(iter));
  }
  if (HasReplicatedSymbolicValues(in_values)) {
    return {{out_index.value(), Undefined{}}};
  }
  List<DimExpr> dim_constants{};
  for (const auto& dim : *dims) {
    dim_constants->emplace_back(dim);
  }
  IndexDotValue<Value, List<DimExpr>> index_dot{in_values, dim_constants};
  return {{out_index.value(), index_dot}};
}

std::unordered_map<Variable, Value> InferValuesImpl(
    const GetBroadcastedIterator<DimExpr, tOut<Iterator>, tIn<Iterator>>&
        broadcast,
    IndexExprInferContext* ctx) {
  const auto& [dim, out_iterator, in_iterator] = broadcast.tuple();
  BroadcastedIterator<Value, DimExpr> broadcast_iterator{
      ctx->GetValue(in_iterator.value()), dim};
  return {{out_iterator.value(), broadcast_iterator}};
}

std::unordered_map<Variable, Value> InferValuesImpl(
    const IndexUnDot<List<DimExpr>, tOut<List<Iterator>>, tIn<Index>>& undot,
    IndexExprInferContext* ctx) {
  const auto& [dims, out_iters, in_index] = undot.tuple();

  List<DimExpr> dim_constants{};
  for (const auto& dim : *dims) {
    dim_constants->emplace_back(dim);
  }
  IndexUnDotValue<Value, List<DimExpr>> index_undot{
      ctx->GetValue(in_index.value()), dim_constants};

  std::unordered_map<Variable, Value> ret{};
  for (std::size_t idx = 0; idx < out_iters.value()->size(); ++idx) {
    ListGetItem<Value, DimExpr> list_get_item{
        Value{index_undot}, DimExpr(static_cast<std::int64_t>(idx))};
    ret.emplace(out_iters.value()->at(idx), list_get_item);
  }
  return ret;
}

std::unordered_map<Variable, Value> InferValuesImpl(
    const InMsg2OutMsg<tOut<FakeOpPlaceHolder>,
                       tOut<OpArgIndexes<std::optional<Index>>>,
                       tIn<OpArgIndexes<Index>>>& in_msg2out_msg,
    IndexExprInferContext* ctx) {
  const auto& [op_placeholder, out_msg_indexes, in_msg_indexes] =
      in_msg2out_msg.tuple();
  const auto& [out_msg_in_indexes, out_msg_out_indexes] =
      out_msg_indexes.value().tuple();
  const auto& [in_msg_in_indexes, in_msg_out_indexes] =
      in_msg_indexes.value().tuple();
  std::unordered_map<Variable, Value> ret{{op_placeholder.value(), Ok{}}};
  PADDLE_ENFORCE_EQ(
      out_msg_in_indexes.value()->size() == in_msg_in_indexes.value()->size(),
      true,
      ::common::errors::InvalidArgument(
          "The size of out_msg_in_indexes should be equal to the size of "
          "in_msg_in_indexes, but got out_msg_in_indexes size = %d, "
          "in_msg_in_indexes size = %d.",
          out_msg_in_indexes.value()->size(),
          in_msg_in_indexes.value()->size()));
  PADDLE_ENFORCE_EQ(
      out_msg_out_indexes.value()->size() == in_msg_out_indexes.value()->size(),
      true,
      ::common::errors::InvalidArgument(
          "The size of out_msg_out_indexes should be equal to the size of "
          "in_msg_out_indexes, but got out_msg_out_indexes size = %d, "
          "in_msg_out_indexes size = %d.",
          out_msg_out_indexes.value()->size(),
          in_msg_out_indexes.value()->size()));
  for (std::size_t i = 0; i < out_msg_in_indexes.value()->size(); ++i) {
    const auto& value = ctx->GetValue(in_msg_in_indexes.value()->at(i));
    PADDLE_ENFORCE_EQ(
        ret.emplace(out_msg_in_indexes.value()->at(i), value).second,
        true,
        ::common::errors::AlreadyExists([&]() {
          std::ostringstream oss;
          oss << "Failed to insert the variable '"
              << "out_msg_in_indexes.value()->at(" << i
              << ")' into the map: key already exists.";
          return oss.str();
        }()));
  }
  for (std::size_t i = 0; i < out_msg_out_indexes.value()->size(); ++i) {
    const auto& value = ctx->GetValue(in_msg_out_indexes.value()->at(i));
    const auto& out_index = out_msg_out_indexes.value()->at(i);
    if (out_index.has_value()) {
      PADDLE_ENFORCE_EQ(ret.emplace(out_index.value(), value).second,
                        true,
                        ::common::errors::AlreadyExists([&]() {
                          std::ostringstream oss;
                          oss << "Failed to insert the variable '"
                              << "out_index.value()"
                              << "' into the map: key already exists.";
                          return oss.str();
                        }()));
    }
  }
  return ret;
}

std::unordered_map<Variable, Value> InferValuesImpl(
    const ConstantFunction<tOut<Iterator>, tIn<Index>>& constant_function,
    IndexExprInferContext* ctx) {
  const auto& [out_iter, in_index, constant] = constant_function.tuple();
  return std::unordered_map<Variable, Value>{{out_iter.value(), constant}};
}

std::unordered_map<Variable, Value> InferValues(const Function* function,
                                                IndexExprInferContext* ctx) {
  return std::visit(
      [&](auto&& function) { return InferValuesImpl(function, ctx); },
      function->variant());
}

DEFINE_ADT_TAG(tValueInferSuccess);

template <typename OnFailT>
tValueInferSuccess<bool> MergeInferredValuesIntoCtx(const Function* function,
                                                    IndexExprInferContext* ctx,
                                                    const OnFailT& OnFail) {
  auto output_variable2value = InferValues(function, ctx);
  for (const auto& [variable, unsimplified_value] : output_variable2value) {
    Value simplified_value({SimplifyValue(unsimplified_value, *ctx)});
    if (simplified_value.Has<Undefined>()) {
      return OnFail(std::optional<Value>{std::nullopt}, simplified_value);
    }
    if (!ctx->HasValue(variable)) {
      ctx->SetValue(variable, simplified_value);
    } else {
      std::optional<Value> opt_old_value = ctx->GetValue(variable);
      if (simplified_value != opt_old_value.value()) {
        return OnFail(opt_old_value, simplified_value);
      }
    }
  }
  return tValueInferSuccess<bool>{true};
}

tValueInferSuccess<bool> MergeInferredValuesIntoCtx(
    const Function* function, IndexExprInferContext* ctx) {
  return MergeInferredValuesIntoCtx(
      function, ctx, [&](const std::optional<Value>& lhs, const Value& rhs) {
        if (lhs.has_value()) {
          VLOG(1) << "opt_old_value = " << ToTxtString(lhs.value());
        }
        VLOG(1) << "simplified = " << ToTxtString(rhs);
        return tValueInferSuccess<bool>{false};
      });
}

std::string GetFunctionName(const Function* function) {
  return std::visit(
      [](auto&& arg) -> std::string { return typeid(arg).name(); },
      function->variant());
}

void SolveEquations(
    const EquationGraphTopoWalker<Variable, const Function*>& walker,
    const std::vector<Variable>& starts,
    IndexExprInferContext* ctx) {
  walker.WalkFunction(
      starts.begin(), starts.end(), [&](const Function* function) {
        tValueInferSuccess<bool> has_unique_value =
            MergeInferredValuesIntoCtx(function, ctx);
        PADDLE_ENFORCE_EQ(
            has_unique_value.value(),
            true,
            ::common::errors::InvalidArgument([&]() {
              std::ostringstream oss;
              oss << "Failed to merge inferred values into the context for "
                     "function '"
                  << GetFunctionName(function) << "'.";
              return oss.str();
            }()));
      });
}

void CheckEquationsSolvable(
    const EquationGraphTopoWalker<Variable, const Function*>& walker,
    const Variable& start,
    IndexExprInferContext* ctx) {
  const auto& CheckNoConflictInferredValue = [&](const Function* function) {
    MergeInferredValuesIntoCtx(
        function,
        ctx,
        [&](const auto& opt_old_value, const auto& simplified_value) {
          LOG(ERROR) << "old_value: " << ToTxtString(opt_old_value);
          LOG(ERROR) << "simplified_value: " << ToTxtString(simplified_value);
          PADDLE_THROW(::common::errors::InvalidArgument(
              "CheckEquationsSolvable Failed"));
          return tValueInferSuccess<bool>{false};
        });
  };

  walker.WalkFunction(start, CheckNoConflictInferredValue);
}

tHasNoConflictValue<bool> TrySolveEquations(
    const EquationGraphTopoWalker<Variable, const Function*>& walker,
    const Variable& start,
    IndexExprInferContext* ctx) {
  bool has_no_conflict_value = true;

  const auto& HasConflictInferredValue = [&](const Function* function) {
    tValueInferSuccess<bool> has_unique_value =
        MergeInferredValuesIntoCtx(function, ctx);
    return !has_unique_value.value();
  };

  walker.WalkFunction(start, [&](const Function* function) {
    if (has_no_conflict_value && HasConflictInferredValue(function)) {
      has_no_conflict_value = false;
    }
  });
  return tHasNoConflictValue<bool>{has_no_conflict_value};
}

}  // namespace cinn::adt
