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

#pragma once

#include <algorithm>
#include <atomic>
#include <memory>
#include <optional>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

#include "glog/logging.h"

#include "paddle/cinn/adt/adt.h"
#include "paddle/cinn/hlir/framework/op.h"
#include "paddle/cinn/operator_fusion/pattern.h"
#include "paddle/cinn/operator_fusion/utils.h"

// This file is the protocol of the pattern fuser. Please implement
// ConvertToStmtPattern and MergePatternImpl in the specializations.

namespace cinn::fusion {

static StmtPattern ConvertToStmtPattern(const PatternContent& content) {
  const auto& kind = GetOpPatternKind(content.op);
  if (kind == hlir::framework::kReduction) {
    auto result =
        ReducePattern({content.op}, std::make_shared<FusionTracker>());
    result.tracker_->append(
        std::make_shared<InitPatternInstr>(content.op, result.id()));
    return result;
  } else if (kind == hlir::framework::kElementWise ||
             kind == hlir::framework::kBroadcast ||
             kind == hlir::framework::kInjective) {
    auto result = TrivialPattern(
        {content.op}, content.op, std::make_shared<FusionTracker>());
    result.tracker_->append(
        std::make_shared<InitPatternInstr>(content.op, result.id()));
    return result;
  } else {
    PADDLE_THROW(::common::errors::InvalidArgument(
        "Unsupported op for fusion: %s", OpsDebugStr({content.op})));
    auto result =
        UnsupportedPattern({content.op}, std::make_shared<FusionTracker>());
    result.tracker_->append(
        std::make_shared<InitPatternInstr>(content.op, result.id()));
    return result;
  }
}

// Trivial x other

static StmtPattern MergePatternImpl(const TrivialPattern& first,
                                    const TrivialPattern& second) {
  const auto& contents =
      UniqueConcatVector(GetOpsInPattern(first), GetOpsInPattern(second));
  return TrivialPattern(
      contents,
      second.sink_op(),
      std::make_shared<FusionTracker>(first.tracker_, second.tracker_));
}

static StmtPattern MergePatternImpl(const TrivialPattern& first,
                                    const ReducePattern& second) {
  const auto& contents =
      UniqueConcatVector(GetOpsInPattern(first), GetOpsInPattern(second));
  return ReducePattern(
      contents,
      std::make_shared<FusionTracker>(first.tracker_, second.tracker_));
}

template <typename A, typename B>
B FusePatternIfConnected(A up_pattern,
                         B down_pattern,
                         std::vector<pir::Operation*> connect_ops) {
  if (AnyFirstInSecond(connect_ops, down_pattern.ops())) {
    return std::get<B>(MergePatternImpl(up_pattern, down_pattern));
  } else {
    return down_pattern;
  }
}

static StmtPattern MergePatternImpl(const TrivialPattern& first,
                                    const ReduceTreePattern& second) {
  auto connect_ops = FindDownstreamOps(first.sink_op());

  auto old_children = second.children();
  std::vector<ReduceTreePattern> new_children;
  for (const auto& old_child : old_children) {
    new_children.emplace_back(
        FusePatternIfConnected(first, old_child, connect_ops));
  }

  return ReduceTreePattern(
      new_children,
      FusePatternIfConnected(first, second.GetRootPattern(), connect_ops),
      std::make_shared<FusionTracker>(first.tracker_, second.tracker_));
}

static StmtPattern MergePatternImpl(
    const TrivialPattern& first, const ReduceTreePlusTrivialPattern& second) {
  auto connect_ops = FindDownstreamOps(first.sink_op());
  auto result = ReduceTreePlusTrivialPattern(
      FusePatternIfConnected(first, second.tree, connect_ops),
      FusePatternIfConnected(first, second.sink_trivial, connect_ops),
      std::make_shared<FusionTracker>(first.tracker_, second.tracker_));
  result.fake_reduce_iter_idx = second.fake_reduce_iter_idx;
  return result;
}

static StmtPattern MergePatternImpl(const TrivialPattern& first,
                                    const ItersPermutationPattern& second) {
  return ItersPermutationPattern(
      UniqueConcatVector(GetOpsInPattern(first), GetOpsInPattern(second)),
      std::make_shared<FusionTracker>(first.tracker_, second.tracker_),
      second.loop_dims());
}

// RR & RT

static int InsertUpstreamIntoTree(const ReduceTreePattern& upstream,
                                  ReduceTreePattern& downstream) {  // NOLINT
  auto is_direct_upstream = [&](const ReducePattern& upstream,
                                const ReducePattern& downstream) -> bool {
    auto upstream_result = upstream.GetReduceOp()->result(0);
    auto user_ops = FindUserOp(downstream.ops(), upstream_result);
    return !user_ops.empty();
  };

  if (is_direct_upstream(upstream.GetRootPattern(),
                         downstream.GetRootPattern())) {
    downstream.InsertChild(upstream);
    return 1;
  }
  int insert_num = 0;
  for (auto& child : downstream.children()) {
    insert_num += InsertUpstreamIntoTree(upstream, child);
  }
  return insert_num;
}

static StmtPattern MergePatternImpl(const ReduceTreePattern& upstream,
                                    const ReduceTreePattern& downstream) {
  ReduceTreePattern result = ReduceTreePattern(
      downstream.children(),
      downstream.GetRootPattern(),
      std::make_shared<FusionTracker>(upstream.tracker_,
                                      downstream.tracker_));  // copy first.
  int insert_num = InsertUpstreamIntoTree(upstream, result);
  PADDLE_ENFORCE_EQ(insert_num,
                    1,
                    ::common::errors::PreconditionNotMet(
                        "Must insert only once, but insert %d", insert_num));
  return result;
}

static StmtPattern MergePatternImpl(const ReduceTreePattern& first,
                                    const TrivialPattern& second) {
  return ReduceTreePlusTrivialPattern(
      first,
      second,
      std::make_shared<FusionTracker>(first.tracker_, second.tracker_));
}

static std::vector<pir::Operation*> GetOutputOpsInPattern(
    const StmtPattern& pattern) {
  struct Visitor {
    std::vector<pir::Operation*> operator()(const ReducePattern& pattern) {
      return {pattern.GetReduceOp()};
    }
    std::vector<pir::Operation*> operator()(const TrivialPattern& pattern) {
      return {pattern.sink_op()};
    }
    std::vector<pir::Operation*> operator()(const UnsupportedPattern& pattern) {
      PADDLE_THROW(::common::errors::Unimplemented(
          "Get output ops in UnsupportedPattern is not implement!"));
    }
    std::vector<pir::Operation*> operator()(const ReduceTreePattern& pattern) {
      return this->operator()(pattern.GetRootPattern());
    }
    std::vector<pir::Operation*> operator()(
        const ReduceTreePlusTrivialPattern& pattern) {
      return {this->operator()(pattern.sink_trivial)};
    }
    std::vector<pir::Operation*> operator()(
        const HorizontalFusionPattern& horizontal) {
      using PaddingStmtPattern =
          typename HorizontalFusionPattern::PaddingStmtPattern;
      return VectorFlatMap(horizontal.padding_patterns_,
                           std::function<std::vector<pir::Operation*>(
                               const PaddingStmtPattern& pattern)>(
                               [](const PaddingStmtPattern& pattern) {
                                 return std::visit(Visitor(), pattern.pattern);
                               }));
    }
    std::vector<pir::Operation*> operator()(
        const ItersPermutationPattern& pattern) {
      PADDLE_THROW(::common::errors::Unimplemented(
          "Can't get output ops for ItersPermutationPattern Currently."));
    }
  };
  return std::visit(Visitor(), pattern);
}

using LoopValueDims = std::vector<ValueDim>;

static std::vector<LoopValueDims> GetLoopValueDims(const StmtPattern& pattern);

struct LoopValueDimsVisitor {
  std::vector<LoopValueDims> operator()(const ReducePattern& pattern) {
    pir::Operation* reduce_op = pattern.GetReduceOp();
    const auto& flatten_loops = GetAllValueDimFromValue(reduce_op->result(0));
    const auto& reduce_axes = GetReduceAxisIdx(reduce_op);
    std::function<ValueDim(int64_t)> f = [&reduce_op](int64_t i) {
      return ValueDim(reduce_op->operand(0).source(), i);
    };
    std::vector<LoopValueDims> res;
    res.emplace_back(ConcatVector(flatten_loops, MapVector(reduce_axes, f)));
    return res;
  }

  std::vector<LoopValueDims> operator()(const ReduceTreePattern& pattern) {
    return GetLoopValueDims(StmtPattern(pattern.GetRootPattern()));
  }

  std::vector<LoopValueDims> operator()(const TrivialPattern& pattern) {
    pir::Operation* t_op = pattern.sink_op();
    const auto& value_dims = GetAllValueDimFromValue(t_op->result(0));
    std::vector<LoopValueDims> res;
    res.emplace_back(value_dims);
    return res;
  }

  std::vector<LoopValueDims> operator()(
      const HorizontalFusionPattern& pattern) {
    // Horizontal Fusion must have the same loop framework.
    using PaddingStmt = typename HorizontalFusionPattern::PaddingStmtPattern;
    return VectorFlatMap(
        pattern.padding_patterns_,
        std::function<std::vector<LoopValueDims>(const PaddingStmt&)>(
            [](const PaddingStmt& padding_stmt) {
              const auto& base_vdims_vec =
                  GetLoopValueDims(StmtPattern(padding_stmt.pattern));
              const auto& padding_vector = padding_stmt.padding_pos;
              std::vector<LoopValueDims> res;
              for (int i = 0; i < base_vdims_vec.size(); ++i) {
                const auto& base_value_dims = base_vdims_vec[i];
                LoopValueDims exprs(base_value_dims.size() +
                                    padding_vector.size());
                int pointer = 0;
                for (int i = 0; i < exprs.size(); i++) {
                  if (std::find(padding_vector.begin(),
                                padding_vector.end(),
                                i) == padding_vector.end()) {
                    exprs[i] = base_value_dims[pointer++];
                  }
                }
                res.push_back(exprs);
              }
              return res;
            }));
  }

  std::vector<LoopValueDims> operator()(
      const ReduceTreePlusTrivialPattern& pattern) {
    const auto& sink_trivial = pattern.sink_trivial;
    const auto& trivial_loop =
        GetLoopValueDims(StmtPattern(pattern.sink_trivial));
    std::vector<LoopValueDims> res;
    if (pattern.fake_reduce_iter_idx.empty()) {
      // we add reduce loop to the end;
      int reduce_axes_len =
          GetReduceAxisIdx(pattern.tree.GetRootPattern().GetReduceOp()).size();
      const auto& reduce_loop =
          GetLoopValueDims(StmtPattern(pattern.tree.GetRootPattern()));
      res.emplace_back(ConcatVector(
          trivial_loop[0],
          SliceVector(
              reduce_loop[0], -reduce_axes_len, reduce_loop[0].size())));
    } else {
      // we always put fake into the end to make the loop framework consistent.
      const auto& non_fake = GatherVector(
          trivial_loop[0],
          ExcludeIndex(trivial_loop[0].size(), pattern.fake_reduce_iter_idx));
      const auto& fake =
          GatherVector(trivial_loop[0], pattern.fake_reduce_iter_idx);
      res.emplace_back(ConcatVector(non_fake, fake));
    }
    return res;
  }

  std::vector<LoopValueDims> operator()(const UnsupportedPattern& pattern) {
    PADDLE_ENFORCE(false, "Not support GetLoopRange.");
  }

  std::vector<LoopValueDims> operator()(
      const ItersPermutationPattern& pattern) {
    PADDLE_THROW(::common::errors::Unimplemented(
        "Can't get loop value dims for ItersPermutationPattern Currently."));
  }
};

static std::vector<LoopValueDims> GetLoopValueDims(const StmtPattern& pattern) {
  return std::visit(LoopValueDimsVisitor(), pattern);
}

using LoopExprs = std::vector<symbol::DimExpr>;

struct MaybeLoopFramework {
  std::string DebugStr() const {
    return "loop: " + utils::Join(loop, ",") +
           ", is_reduce: " + utils::Join(is_reduce, ",");
  }
  LoopExprs loop;
  std::vector<bool> is_reduce;
};

static MaybeLoopFramework GetLoopFramework(const StmtPattern& pattern);

static MaybeLoopFramework SqueezeLoopFramework(
    const MaybeLoopFramework& input) {
  MaybeLoopFramework result;
  auto loop = input.loop;
  for (int i = 0; i < loop.size(); i++) {
    if (loop[i] == 1) {
      continue;  // skip 1
    } else {
      result.loop.push_back(loop[i]);
      result.is_reduce.push_back(input.is_reduce[i]);
    }
  }
  return result;
}

static std::pair<LoopExprs, LoopExprs> SplitReduceLoop(
    const MaybeLoopFramework& loops) {
  LoopExprs non_reduce_loops;
  LoopExprs reduce_loops;
  for (int i = 0; i < loops.is_reduce.size(); ++i) {
    if (loops.is_reduce[i]) {
      reduce_loops.push_back(loops.loop[i]);
    } else {
      non_reduce_loops.push_back(loops.loop[i]);
    }
  }
  return {non_reduce_loops, reduce_loops};
}

static std::vector<bool> CreateIsReduceVector(const size_t& nums_flatten,
                                              const size_t& nums_reduce) {
  return ConcatVector(std::vector<bool>(nums_flatten, false),
                      std::vector<bool>(nums_reduce, true));
}

static bool IsLoopFrameworkEqual(const StmtPattern& lhs,
                                 const StmtPattern& rhs) {
  const auto& lhs_loops = GetLoopFramework(lhs);
  const auto& rhs_loops = GetLoopFramework(rhs);
  VLOG(4) << "lhs " << lhs_loops.DebugStr();
  VLOG(4) << "rhs " << rhs_loops.DebugStr();

  // TODO(huangjiyi): support horizontal fusion without reduce dims euqal.
  const auto get_reduce_loop = [](const MaybeLoopFramework& loop) {
    LoopExprs reduce_loop;
    for (int i = 0; i < loop.is_reduce.size(); ++i) {
      if (loop.is_reduce[i]) {
        reduce_loop.push_back(loop.loop[i]);
      }
    }
    return reduce_loop;
  };
  const auto lhs_reduce_loop = get_reduce_loop(lhs_loops);
  const auto rhs_reduce_loop = get_reduce_loop(rhs_loops);

  bool reduce_euqal = lhs_reduce_loop.empty() || rhs_reduce_loop.empty()
                          ? true
                          : lhs_reduce_loop == rhs_reduce_loop;

  const auto& squeezed_lhs_loops = SqueezeLoopFramework(lhs_loops);
  const auto& squeezed_rhs_loops = SqueezeLoopFramework(rhs_loops);
  bool loop_equal = squeezed_lhs_loops.loop == squeezed_rhs_loops.loop;

  return loop_equal && reduce_euqal;
}

struct LoopFrameworkVisitor {
  MaybeLoopFramework operator()(const ReducePattern& pattern) {
    pir::Operation* reduce_op = pattern.GetReduceOp();
    const auto& flatten_loops = GetDimExprsFromValue(reduce_op->result(0));
    const auto& reduce_axes = GetReduceAxisIdx(reduce_op);
    const auto& reduce_loops = GatherVector(
        GetDimExprsFromValue(reduce_op->operand(0).source()), reduce_axes);
    const auto& loop = ConcatVector(flatten_loops, reduce_loops);
    const auto& is_reduce =
        CreateIsReduceVector(flatten_loops.size(), reduce_loops.size());
    return {loop, is_reduce};
  }

  MaybeLoopFramework operator()(const ReduceTreePattern& pattern) {
    return GetLoopFramework(pattern.GetRootPattern());
  }

  MaybeLoopFramework operator()(const TrivialPattern& pattern) {
    pir::Operation* t_op = pattern.sink_op();
    const auto& loop = GetDimExprsFromValue(t_op->result(0));
    return {loop, std::vector<bool>(loop.size(), false)};
  }

  MaybeLoopFramework operator()(const HorizontalFusionPattern& pattern) {
    // Horizontal Fusion must have the same loop framework.
    VLOG(4) << "Get loop framework for HorizontalFusionPattern.";
    auto base_pattern = pattern.padding_patterns_.back();
    auto [base_loop, base_is_reduce] = GetLoopFramework(base_pattern.pattern);
    for (const auto& padding_pattern : pattern.padding_patterns_) {
      const auto& [loop, is_reduce] = GetLoopFramework(padding_pattern.pattern);
      if (std::any_of(
              is_reduce.begin(), is_reduce.end(), [](bool x) { return x; })) {
        base_pattern = padding_pattern;
        base_loop = loop;
        base_is_reduce = is_reduce;
        break;
      }
    }
    const auto& padding_vector = base_pattern.padding_pos;
    const auto& padded_size = base_loop.size() + padding_vector.size();
    LoopExprs loop(padded_size, 1);
    std::vector<bool> is_reduce(padded_size, false);
    int pointer = 0;
    for (int i = 0; i < loop.size(); i++) {
      if (std::find(padding_vector.begin(), padding_vector.end(), i) ==
          padding_vector.end()) {
        loop[i] = base_loop[pointer];
        is_reduce[i] = base_is_reduce[pointer++];
      }
    }
    return {loop, is_reduce};
  }

  MaybeLoopFramework operator()(const ReduceTreePlusTrivialPattern& pattern) {
    const auto& sink_trivial = pattern.sink_trivial;
    auto trivial_loop = GetLoopFramework(pattern.sink_trivial).loop;
    if (!pattern.fake_reduce_iter_idx.empty()) {
      trivial_loop = GatherVector(
          trivial_loop,
          ExcludeIndex(trivial_loop.size(), pattern.fake_reduce_iter_idx));
    }
    const auto& [_UNUSED, reduce_loop] =
        SplitReduceLoop(GetLoopFramework(pattern.tree.GetRootPattern()));
    return {ConcatVector(trivial_loop, reduce_loop),
            CreateIsReduceVector(trivial_loop.size(), reduce_loop.size())};
  }

  MaybeLoopFramework operator()(const UnsupportedPattern& pattern) {
    PADDLE_THROW(
        ::common::errors::Unimplemented("Unsupported for GetLoopRange."));
  }

  MaybeLoopFramework operator()(const ItersPermutationPattern& pattern) {
    const auto loop_dims = pattern.loop_dims();
    return {loop_dims.first, loop_dims.second};
  }
};

static MaybeLoopFramework GetLoopFramework(const StmtPattern& pattern) {
  return std::visit(LoopFrameworkVisitor(), pattern);
}

static inline auto GetPaddingVector(const LoopExprs& first,
                                    const LoopExprs& second) {
  // two pointer to get the padding body.
  std::vector<int> padding_f;
  std::vector<int> padding_s;
  VLOG(4) << "GetPaddingVector for: " << utils::Join(first, ",") << " vs "
          << utils::Join(second, ",");

  std::function<void(int, int, int)> RecursivePadding =
      [&first, &second, &padding_f, &padding_s, &RecursivePadding](
          int pf, int ps, int padding_size) {
        VLOG(4) << "Padding Process: " << pf << " " << ps << " "
                << padding_size;
        if (pf == first.size() && ps == second.size()) {
          return;
        } else if (pf == first.size()) {
          PADDLE_ENFORCE(second[ps] == 1, "second[ps] must be '1' to padding.");
          padding_f.push_back(padding_size);
          RecursivePadding(pf, ps + 1, padding_size + 1);
        } else if (ps == second.size()) {
          PADDLE_ENFORCE(first[pf] == 1, "second[ps] must be '1' to padding.");
          padding_s.push_back(padding_size);
          RecursivePadding(pf + 1, ps, padding_size + 1);
        } else if (second[ps] == first[pf]) {
          RecursivePadding(pf + 1, ps + 1, padding_size + 1);
        } else if (second[ps] == 1) {
          padding_f.push_back(padding_size);
          RecursivePadding(pf, ps + 1, padding_size + 1);
        } else if (first[pf] == 1) {
          padding_s.push_back(padding_size);
          RecursivePadding(pf + 1, ps, padding_size + 1);
        } else {
          PADDLE_THROW("Padding Error.");
        }
      };
  RecursivePadding(0, 0, 0);
  VLOG(4) << "GetPaddingVector result: " << utils::Join(padding_f, ",")
          << " vs " << utils::Join(padding_s, ",");
  return std::tuple(padding_f, padding_s);
}

static StmtPattern MergePatternImpl(const HorizontalFusionPattern& first,
                                    const HorizontalFusionPattern& second) {
  const auto& [f, s] = GetPaddingVector(GetLoopFramework(first).loop,
                                        GetLoopFramework(second).loop);
  typename HorizontalFusionPattern::PaddingStmtPattern pad_first = {first, f};
  typename HorizontalFusionPattern::PaddingStmtPattern pad_second = {second, s};
  return HorizontalFusionPattern(
      {pad_first, pad_second},
      std::make_shared<FusionTracker>(first.tracker_, second.tracker_));
}

static StmtPattern MergePattern(const StmtPattern& first,
                                const StmtPattern& second) {
  VLOG(4) << "MergePattern: " << GetPatternId(first) << " x "
          << GetPatternId(second);
  const auto PatternMatch = adt::match{
      [&](const ReduceTreePattern& lhs, const ReduceTreePattern& rhs) {
        return MergePatternImpl(lhs, rhs);
      },
      [&](const ReduceTreePattern& lhs, const TrivialPattern& rhs) {
        return MergePatternImpl(lhs, rhs);
      },
      [&](const TrivialPattern& lhs, const ReducePattern& rhs) {
        return MergePatternImpl(lhs, rhs);
      },
      [&](const TrivialPattern& lhs, const TrivialPattern& rhs) {
        return MergePatternImpl(lhs, rhs);
      },
      [&](const TrivialPattern& lhs, const ReduceTreePattern& rhs) {
        return MergePatternImpl(lhs, rhs);
      },
      [&](const TrivialPattern& lhs, const ReduceTreePlusTrivialPattern& rhs) {
        return MergePatternImpl(lhs, rhs);
      },
      [&](const TrivialPattern& lhs, const ItersPermutationPattern& rhs) {
        return MergePatternImpl(lhs, rhs);
      },
      [&](const HorizontalFusionPattern& lhs,
          const HorizontalFusionPattern& rhs) {
        return MergePatternImpl(lhs, rhs);
      },
      [&](const auto& lhs, const auto& rhs) -> StmtPattern {
        PADDLE_THROW(::common::errors::Unimplemented(
            "Not support for MergePatternImpl"));
      },
  };
  return std::visit(PatternMatch, first, second);
}

static void SetReturnInstr(const StmtPattern& s) {
  std::visit(
      [](const auto& impl) {
        impl.tracker_->append(std::make_shared<ReturnInstr>(impl.id()));
      },
      s);
}

}  // namespace cinn::fusion
