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

#include "paddle/cinn/adt/write_broadcast_disabled_bidirection_equation_generator.h"

#include "paddle/cinn/adt/equation_graph.h"
#include "paddle/cinn/adt/equation_solver.h"
#include "paddle/common/enforce.h"
namespace cinn::adt {

namespace {

using EquationCtx4OpStmtT =
    std::function<std::shared_ptr<config::NaiveOpEquationContext>(
        const OpStmt&)>;

template <
    typename DoEachT /*: void(&)(std::size_t, OpStmt, OpEquationContext)*/>
void VisitEachOpStmtAndEquationCtx(
    const List<OpStmt>& op_stmts,
    const EquationCtx4OpStmtT& EquationCtx4OpStmt,
    const DoEachT& DoEach) {
  for (std::size_t i = 0; i < op_stmts->size(); ++i) {
    const auto& ctx = EquationCtx4OpStmt(op_stmts->at(i));
    DoEach(i, op_stmts->at(i), ctx);
  }
}

using InMsg2OutMsgT = InMsg2OutMsg<tOut<FakeOpPlaceHolder>,
                                   tOut<OpArgIndexes<std::optional<Index>>>,
                                   tIn<OpArgIndexes<Index>>>;

std::unordered_map<Variable, const Value> MakeAnchorIndex2Ok(
    const Index& anchor_index) {
  return {{anchor_index, Ok{}}};
}

bool LocalEquationsSolvable(const GraphView& graph_view,
                            const Index& anchor_index,
                            const FakeOpPlaceHolder& fake_op_placeholder) {
  const auto& init_var2value = MakeAnchorIndex2Ok(anchor_index);
  IndexExprInferContext ctx{init_var2value};
  bool has_no_conflict_value =
      TrySolveEquations(graph_view, anchor_index, &ctx).value();
  return has_no_conflict_value && ctx.HasValue(fake_op_placeholder);
}

List<std::optional<Index>> GetMaskedOutIndexes(
    const List<Index>& in_msg_out_indexes,
    const List<std::optional<Index>>& out_msg_out_indexes,
    const std::vector<Index>& erased_in_msg_out_tensor_indexes) {
  List<std::optional<Index>> ret{};
  const auto& erased = erased_in_msg_out_tensor_indexes;
  PADDLE_ENFORCE_EQ(
      in_msg_out_indexes->size(),
      out_msg_out_indexes->size(),
      ::common::errors::InvalidArgument(
          "The size of in_msg_out_indexes and out_msg_out_indexes "
          "should be equal, but got in_msg_out_indexes size = %d, "
          "out_msg_out_indexes size = %d.",
          in_msg_out_indexes->size(),
          out_msg_out_indexes->size()));
  for (std::size_t i = 0; i < in_msg_out_indexes->size(); ++i) {
    const auto& in_msg_index = in_msg_out_indexes->at(i);
    if (std::find(erased.begin(), erased.end(), in_msg_index) == erased.end()) {
      ret->emplace_back(out_msg_out_indexes->at(i));
    } else {
      ret->emplace_back(std::nullopt);
    }
  }
  return ret;
}

Equation EraseIndexes(
    const Equation& equation,
    const std::vector<Index>& erased_in_msg_out_tensor_indexes) {
  const auto& in_msg2out_msg = equation.Get<InMsg2OutMsgT>();
  const auto& [op_placeholder, out_msg_indexes, in_msg_indexes] =
      in_msg2out_msg.tuple();

  const auto& [_, in_msg_out_indexes] = in_msg_indexes.value().tuple();
  const auto& [out_msg_in_indexes, out_msg_out_indexes] =
      out_msg_indexes.value().tuple();
  const auto& masked_out_indexes =
      GetMaskedOutIndexes(in_msg_out_indexes.value(),
                          out_msg_out_indexes.value(),
                          erased_in_msg_out_tensor_indexes);

  OpArgIndexes<std::optional<Index>> new_out_msg_indexes{out_msg_in_indexes,
                                                         masked_out_indexes};

  Equation ret_equation =
      InMsg2OutMsgT{op_placeholder, new_out_msg_indexes, in_msg_indexes};

  return ret_equation;
}

std::vector<Index> GenerateWriteBroadcastTensorIndices(
    const std::shared_ptr<config::NaiveOpEquationContext>& ctx,
    const Equations& in_msg2out_msg_equations) {
  const auto& equation_graph_view =
      Graph<Variable, Equation>::New(ctx->equations())->GetGraphView();
  GraphView graph_view = equation_graph_view.Merge(
      Graph<Variable, Equation>::New(in_msg2out_msg_equations)->GetGraphView());
  std::vector<Index> ret{};
  const auto& fake_op_placeholder = ctx->fake_op_placeholder();
  ctx->VisitEachOutputTensorIndex([&](const auto& out_index) {
    if (!LocalEquationsSolvable(graph_view, out_index, fake_op_placeholder)) {
      ret.emplace_back(out_index);
    }
  });
  return ret;
}

}  // namespace

Equations
WriteBroadcastDisabledBidirectionEquationGenerator::GetDirectionEquations()
    const {
  Equations ret{};
  VisitEachOpStmtAndEquationCtx(
      naive_bidirection_equation_generator_.op_stmts(),
      naive_bidirection_equation_generator_.EquationCtx4OpStmt(),
      [&](std::size_t idx,
          const OpStmt& op_stmt,
          const std::shared_ptr<config::NaiveOpEquationContext>& ctx) {
        const auto& in_msg2out_msg_equations =
            naive_bidirection_equation_generator_.equations();
        const auto& truncated_output_tensor_indices =
            GenerateWriteBroadcastTensorIndices(ctx, in_msg2out_msg_equations);
        ret->emplace_back(EraseIndexes(in_msg2out_msg_equations->at(idx),
                                       truncated_output_tensor_indices));
      });
  return ret;
}

}  // namespace cinn::adt
