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

#include "paddle/cinn/adt/schedule_dim.h"

#include "paddle/cinn/adt/equation_graph.h"
#include "paddle/cinn/adt/equation_solver.h"
#include "paddle/cinn/adt/igroup.h"
#include "paddle/cinn/adt/index_expr_infer_context.h"
#include "paddle/cinn/adt/m_ir.h"
#include "paddle/cinn/adt/naive_op_equation_context.h"
#include "paddle/cinn/adt/print.h"

namespace cinn::adt {

namespace {

template <typename DoEachT>
void VisitEachOpEquationContext(const IGroup& igroup, const DoEachT& DoEach) {
  for (const auto& op_stmt : *igroup.op_stmts()) {
    const auto& EquationCtx4OpStmt = igroup.EquationCtx4OpStmt();
    const auto& ctx = EquationCtx4OpStmt(op_stmt);
    DoEach(ctx);
  }
}

List<Iterator> GetOpEquationCtxInputIterators(
    const std::shared_ptr<config::NaiveOpEquationContext>& ctx) {
  List<Iterator> ret{};
  std::size_t input_size = ctx->in_indexes()->size();
  for (std::size_t i = 0; i < input_size; ++i) {
    for (const auto& iterator : *ctx->GetInIteratorTuple(i)) {
      ret->emplace_back(iterator);
    }
  }
  return ret;
}

std::shared_ptr<IndexExprInferContext> InitIndexExprInferContext(
    const std::shared_ptr<config::NaiveOpEquationContext>& ctx,
    const List<Iterator>& input_iterators) {
  std::unordered_map<Variable, const Value> init_var2value;
  for (const auto& iterator : *input_iterators) {
    PADDLE_ENFORCE_EQ(
        init_var2value.emplace(iterator, iterator).second,
        true,
        ::common::errors::InvalidArgument(
            "Insertion failed in init_var2value map. The key already exists."));
  }

  return std::make_shared<IndexExprInferContext>(init_var2value);
}

template <typename DoEachT>
void VisitEachInputIteratorTuple(
    const std::shared_ptr<config::NaiveOpEquationContext>& op_ctx,
    const DoEachT& DoEach) {
  std::size_t input_size = op_ctx->in_indexes()->size();
  for (std::size_t i = 0; i < input_size; ++i) {
    DoEach(op_ctx->GetInIteratorTuple(i));
  }
}

template <typename DoEachT>
void VisitEachOutputIterator(
    const std::shared_ptr<config::NaiveOpEquationContext>& op_ctx,
    const DoEachT& DoEach) {
  std::size_t output_size = op_ctx->out_indexes()->size();
  for (std::size_t i = 0; i < output_size; ++i) {
    for (const auto& output_iterator : *op_ctx->GetOutIteratorTuple(i)) {
      DoEach(output_iterator);
    }
  }
}

void FilterReducedIterator(
    const std::shared_ptr<IndexExprInferContext>& infer_ctx,
    const std::shared_ptr<config::NaiveOpEquationContext>& op_ctx,
    const List<Iterator>& input_iterators,
    std::unordered_set<Iterator>* unused_input_iterators) {
  std::unordered_set<Iterator> used{};
  bool is_output_inferred = true;
  VisitEachOutputIterator(op_ctx, [&](const Iterator& output_iterator) {
    if (infer_ctx->HasValue(output_iterator)) {
      const auto& iterator_expr = infer_ctx->GetValue(output_iterator);
      CollectTensorIndexIterators(iterator_expr, &used);
    } else {
      is_output_inferred = false;
    }
  });
  if (!is_output_inferred) {
    return;
  }
  for (const auto& input_iterator : *input_iterators) {
    if (used.find(input_iterator) == used.end()) {
      unused_input_iterators->emplace(input_iterator);
    }
  }
}

std::unordered_set<Iterator> GenerateReducedIterator(
    const std::shared_ptr<config::NaiveOpEquationContext>& ctx) {
  const auto& graph_view =
      Graph<Variable, Equation>::New(ctx->equations())->GetGraphView();

  std::unordered_set<Iterator> ret{};
  VisitEachInputIteratorTuple(ctx, [&](const List<Iterator>& input_iterators) {
    const auto& infer_ctx = InitIndexExprInferContext(ctx, input_iterators);

    std::vector<Variable> starts{};
    for (const auto& iterator : *input_iterators) {
      starts.emplace_back(iterator);
    }
    SolveEquations(graph_view, starts, infer_ctx.get());

    /*
      y = Reduce(x)

      y_i = f(x_i, x_j, ...)
      used_set = {x_i, x_j, ...}
      reduce_iterator = input_all_iterator_set - used_set
    */
    FilterReducedIterator(infer_ctx, ctx, input_iterators, &ret);
  });

  return ret;
}

std::unordered_set<Iterator> FilterTemporalIterators(
    const IGroup& igroup,
    const std::function<Value(const Iterator&)>& Value4Iterator) {
  std::unordered_set<Iterator> ret{};

  VisitEachOpEquationContext(
      igroup, [&](const std::shared_ptr<config::NaiveOpEquationContext>& ctx) {
        std::unordered_set<Iterator> reduced_iterators =
            GenerateReducedIterator(ctx);
        for (const auto& input_reduced_iterator : reduced_iterators) {
          const auto& sd_iterator_expr = Value4Iterator(input_reduced_iterator);
          CollectTensorIndexIterators(sd_iterator_expr, &ret);
        }
      });

  return ret;
}

}  // namespace

List<ScheduleDim> MakeAnchorScheduleDims(
    const IGroup& igroup,
    const std::function<Value(const Iterator&)>& Value4Iterator,
    const List<LoopSize>& loop_sizes,
    const List<Iterator>& anchor_iterators) {
  std::unordered_set<Iterator> temporal_sd_iterators =
      FilterTemporalIterators(igroup, Value4Iterator);

  List<ScheduleDim> ret{};
  for (std::size_t i = 0; i < loop_sizes->size(); ++i) {
    const auto& loop_iterator = anchor_iterators->at(i);
    if (temporal_sd_iterators.count(loop_iterator) > 0) {
      ret->emplace_back(tReduced<LoopSize>{loop_sizes->at(i)});
    } else {
      ret->emplace_back(tInjective<LoopSize>{loop_sizes->at(i)});
    }
  }

  return ret;
}

LoopSize GetLoopSize(const ScheduleDim& sched_dim) {
  return std::visit([&](const auto& impl) { return impl.value(); },
                    sched_dim.variant());
}

List<int> GetReduceAxis(const List<ScheduleDim>& loop_sizes) {
  List<int> reduce_axis{};
  for (std::size_t i = 0; i < loop_sizes->size(); ++i) {
    const auto& sched_dim = loop_sizes->at(i);
    if (sched_dim.Has<tReduced<LoopSize>>()) {
      reduce_axis->emplace_back(i);
    } else if (sched_dim.Has<tInjective<LoopSize>>()) {
      // do nothing
    } else {
      PADDLE_THROW(::common::errors::Fatal("Dead code"));
    }
  }
  return reduce_axis;
}

List<int> GetInjectiveAxis(const List<ScheduleDim>& loop_sizes) {
  List<int> injective_axis{};
  for (std::size_t i = 0; i < loop_sizes->size(); ++i) {
    const auto& sched_dim = loop_sizes->at(i);
    if (sched_dim.Has<tReduced<LoopSize>>()) {
      // do nothing
    } else if (sched_dim.Has<tInjective<LoopSize>>()) {
      injective_axis->emplace_back(i);
    } else {
      PADDLE_THROW(::common::errors::Fatal("Dead code"));
    }
  }
  return injective_axis;
}

}  // namespace cinn::adt
