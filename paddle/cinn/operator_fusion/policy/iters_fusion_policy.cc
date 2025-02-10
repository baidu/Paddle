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

#include "paddle/cinn/operator_fusion/policy/iters_fusion_policy.h"
#include "paddle/cinn/operator_fusion/pattern.h"

COMMON_DECLARE_bool(enable_append_iters_in_fusion);
COMMON_DECLARE_bool(enable_reuse_iters_in_fusion);
COMMON_DECLARE_bool(enable_transpose_iters_in_fusion);

namespace cinn::fusion {

bool ItersFusionPolicy::CheckItersRelation(const PatternNodePtr& source,
                                           const PatternNodePtr& target) {
  const auto source_iters = source->fusion_iters().loop_iters;
  const auto target_iters = target->fusion_iters().loop_iters;
  auto related_iters = MapKeyToVector(iters_manager_->related_iters_map());
  const auto [source_related_iters, _UNUESD] =
      SplitFirstWhetherInSecond(source_iters, related_iters);
  const auto target_unique_iters =
      GatherFirstNotInSecond(target_iters, source_iters);
  if (source_related_iters.empty() || target_unique_iters.empty()) {
    return true;
  } else {
    for (const auto& related_iter : source_related_iters) {
      if (iters_manager_->CanFindRelatedIters(related_iter,
                                              target_unique_iters)) {
        VLOG(4) << "Can not fuse from " << source->fusion_iters().DebugStr()
                << " to " << target->fusion_iters().DebugStr();
        return false;
      }
    }
    return true;
  }
}

bool ItersFusionPolicy::CanFuseSource2Target(const PatternNodePtr& source,
                                             const PatternNodePtr& target) {
  VLOG(4) << "Search iters transform route from "
          << source->fusion_iters().DebugStr() << " to "
          << target->fusion_iters().DebugStr();
  if (source->fusion_iters().loop_iters.empty() ||
      target->fusion_iters().loop_iters.empty()) {
    VLOG(4) << "Pattern with empty loop iters can't be fused.";
    return false;
  }
  auto iters_transforms = SearchItersTransformRoute(
      source->fusion_iters(), target->fusion_iters(), false);
  if (iters_transforms == std::nullopt) {
    iters_transforms = SearchItersTransformRoute(
        source->fusion_iters(), target->fusion_iters(), true);
  }
  if (iters_transforms != std::nullopt) {
    VLOG(4) << "Find iters transforms: "
            << DebugStrItersTransformRoute(iters_transforms.value());
    routes_[source][target] = iters_transforms.value();
    return true;
  } else {
    VLOG(4) << "Can't find iters transform route.";
    return false;
  }
}

std::optional<ItersTransformRoute> ItersFusionPolicy::GetItersTransformRoute(
    const PatternNodePtr& source, const PatternNodePtr& target) {
  if (routes_.count(source) && routes_[source].count(target)) {
    PADDLE_ENFORCE_GT(routes_[source][target].size(),
                      0,
                      ::common::errors::InvalidArgument(
                          "Iters transform route should not be empty."));
    return routes_[source][target];
  } else {
    return std::nullopt;
  }
}

FusionItersSignature ItersFusionPolicy::SingleDownstreamItersFusion(
    const PatternNodePtr& upstream, const PatternNodePtr& downstream) {
  return iters_manager_->SingleDownstreamItersFusion(
      upstream->fusion_iters(), downstream->fusion_iters());
}

FusionItersSignature ItersFusionPolicy::MultiDownstreamItersFusion(
    const PatternNodePtr& upstream,
    const PatternNodePtr& downstream,
    const FusionItersManager::FusionDirection& direction) {
  return iters_manager_->MultiDownstreamItersFusion(
      upstream->fusion_iters(), downstream->fusion_iters(), direction);
}

std::pair<std::vector<symbol::DimExpr>, std::vector<bool>>
ItersFusionPolicy::GetLoopDims(const FusionItersSignature& sig) {
  std::vector<symbol::DimExpr> dims;
  for (const auto& iter : sig.loop_iters) {
    dims.push_back(iters_manager_->GetIterSymbol(iter));
  }
  const auto is_reduce =
      ConcatVector(std::vector<bool>(dims.size() - sig.reduce_iter_nums, false),
                   std::vector<bool>(sig.reduce_iter_nums, true));
  return {dims, is_reduce};
}

std::optional<ItersTransform> ItersFusionPolicy::GetReuseItersTransform(
    FusionIters* source_iters, const FusionIters& target_iters) {
  const auto [shared_iters, source_unique_iters] =
      SplitFirstWhetherInSecond(*source_iters, target_iters);
  const auto target_unique_iters =
      GatherFirstNotInSecond(target_iters, shared_iters);

  if (!source_unique_iters.empty() && !target_unique_iters.empty()) {
    if (!transform_strategy_[ItersTransformType::ReuseIters] ||
        !FLAGS_enable_reuse_iters_in_fusion) {
      VLOG(4) << "Can not reuse iters in fusion, because of ReuseIters "
                 "transform is disabled.";
      return std::nullopt;
    }
    std::unordered_map<std::string, std::string> reuse_target_to_source;
    for (const auto& source_iter : source_unique_iters) {
      for (const auto& target_iter : target_unique_iters) {
        if (iters_manager_->IterSymbolEqual(source_iter, target_iter) &&
            !reuse_target_to_source.count(target_iter)) {
          reuse_target_to_source[target_iter] = source_iter;
          break;
        }
      }
    }
    if (reuse_target_to_source.size() != source_unique_iters.size()) {
      // Exist iters in source can not reuse target iter
      return std::nullopt;
    }
    // Replace source iters with reused target iters
    for (const auto& [target_iter, source_iter] : reuse_target_to_source) {
      const auto it =
          std::find(source_iters->begin(), source_iters->end(), source_iter);
      if (it != source_iters->end()) {
        *it = target_iter;
      }
    }
    return ReuseItersTransform(reuse_target_to_source);
  } else {
    // No need to reuse iters
    return IdentityItersTransform();
  }
}

std::optional<ItersTransformRoute>
ItersFusionPolicy::SearchTransformRouteFromReduce2Reduce(
    const FusionItersSignature& source, const FusionItersSignature& target) {
  VLOG(4) << "Start search transform Route from reduce to reduce.";
  if (source.loop_iters.size() == target.loop_iters.size() &&
      source.reduce_iter_nums == target.reduce_iter_nums) {
    // Currently only support fusion with same iter_nums and same reduce axis
    // TODO(huangjiyi): Analysis fusion with different non reduce axis
    auto [source_flatten_iters, source_reduce_iters] = SplitReduceIters(source);
    auto [target_flatten_iters, target_reduce_iters] = SplitReduceIters(target);

    if (AnyFirstInSecond(source_reduce_iters, target_flatten_iters) ||
        AnyFirstInSecond(target_reduce_iters, source_flatten_iters)) {
      VLOG(4) << "Exist reduce iters in one ReduceOp found in the flatten "
                 "iters in another ReduceOp.";
      return std::nullopt;
    }

    ItersTransformRoute route;
    // 1. Apply ReuseItersTransform
    const auto flatten_reuse_iters_transform =
        GetReuseItersTransform(&source_flatten_iters, target_flatten_iters);
    const auto reduce_reuse_iters_transform =
        GetReuseItersTransform(&source_reduce_iters, target_reduce_iters);
    if (flatten_reuse_iters_transform == std::nullopt ||
        reduce_reuse_iters_transform == std::nullopt) {
      VLOG(4) << "Exist iters in source can not reuse target iter.";
      return std::nullopt;
    }
    route.push_back(flatten_reuse_iters_transform.value());
    route.push_back(reduce_reuse_iters_transform.value());

    // 2. Apply TransposeItersTransform
    if (source_flatten_iters == target_flatten_iters &&
        source_reduce_iters == target_reduce_iters) {
      return route;
    } else if (!transform_strategy_[ItersTransformType::TransposeIters] ||
               !FLAGS_enable_transpose_iters_in_fusion) {
      VLOG(4) << "Can not transpose iters in fusion, because of TransposeIters "
                 "transform is disabled.";
      return std::nullopt;
    } else if (source_flatten_iters != target_flatten_iters &&
               source_reduce_iters == target_reduce_iters) {
      const auto flatten_perm =
          GetTransposePerm<int32_t>(source_flatten_iters, target_flatten_iters);
      const auto perm = ConcatVector(
          flatten_perm,
          ArangeVector<int32_t>(flatten_perm.size(), target.loop_iters.size()));
      route.push_back(TransposeItersTransform(perm));
      return route;
    } else {
      // TODO(huangjiyi): Support transpose reduce axis
      VLOG(4) << "Can't not transpose reduce axis currently.";
      return std::nullopt;
    }
  } else {
    VLOG(4) << "Can't fuse because loop iter number or reduce iter number is "
               "not equal";
    return std::nullopt;
  }
}

std::optional<ItersTransformRoute> ItersFusionPolicy::SearchItersTransformRoute(
    const FusionItersSignature& source,
    const FusionItersSignature& target,
    bool squeeze_source) {
  ItersTransformRoute iters_transforms;

  auto squeezed_source = source;
  if (squeeze_source) {
    // Remove iters equal to one in source
    auto source_ones = MapVectorIfTrue<std::pair<std::string, int>, int>(
        Enumerate(source.loop_iters),
        [this](std::pair<std::string, int> p) { return p.second; },
        [this](std::pair<std::string, int> p) {
          return this->iters_manager_->IterSymbolEqualOne(p.first);
        });
    if (!source_ones.empty() &&
        source_ones.size() != source.loop_iters.size()) {
      iters_transforms.emplace_back(RemoveOnesTransform(source_ones));
      squeezed_source.loop_iters =
          GatherVectorExcept(source.loop_iters, source_ones);
    } else {
      return std::nullopt;
    }
  }

  if (squeezed_source.loop_iters.size() > target.loop_iters.size()) {
    VLOG(4) << "Can not decrease iters in multi downstream fusion.";
    return std::nullopt;
  }

  // Search ItersTransformRoute based on different cases
  if (squeezed_source.reduce_iter_nums && target.reduce_iter_nums) {
    // Reduce -> Reduce ItersTransform
    const auto result =
        SearchTransformRouteFromReduce2Reduce(squeezed_source, target);
    if (result != std::nullopt) {
      return ConcatVector(iters_transforms, result.value());
    } else {
      return std::nullopt;
    }
  } else if (!squeezed_source.reduce_iter_nums && target.reduce_iter_nums) {
    // Trivial -> Reduce ItersTransform
    // Can fuse with non fake reduce dims or small inner reduce loop
    auto [target_flatten_iters, _UNUSED] = SplitReduceIters(target);
    if (!AllFirstInSecond(squeezed_source.loop_iters, target_flatten_iters)) {
      const auto reduce_dims_product =
          iters_manager_->GetReduceDimsProduct(target);
      if (reduce_dims_product.isa<std::int64_t>() &&
          reduce_dims_product.dyn_cast<std::int64_t>() > 1024 * 8) {
        VLOG(4) << "Can not fuse trivial to reduce with large reduce dims: "
                << reduce_dims_product.dyn_cast<std::int64_t>();
        return std::nullopt;
      }
    }
  } else if (squeezed_source.reduce_iter_nums && !target.reduce_iter_nums) {
    VLOG(4) << "Can not transform iters from Reduce to Trivial.";
    return std::nullopt;
  }
  // else: Search Trivial -> Trivial ItersTransform

  auto source_iters = squeezed_source.loop_iters;
  const auto target_iters = target.loop_iters;

  // TODO(huangjiyi): Append iters when target contains duplicate elements
  if (ToSet(source_iters).size() != source_iters.size() ||
      ToSet(target_iters).size() != target_iters.size()) {
    VLOG(4) << "The source or target iters contain duplicate elements.";
    return std::nullopt;
  }

  // 1. Apply IdentityItersTransform if source iters are equal to target
  if (source_iters == target_iters) {
    iters_transforms.push_back(IdentityItersTransform());
    return iters_transforms;
  }

  // 2. Apply ReuseItersTransform
  // if all source unique iters can reuse target iters
  FusionIters reused_source_iters = source_iters;
  const auto reuse_iters_transform =
      GetReuseItersTransform(&reused_source_iters, target_iters);
  if (reuse_iters_transform == std::nullopt) {
    VLOG(4) << "Exist iters in source can not reuse target iter.";
    return std::nullopt;
  } else {
    iters_transforms.push_back(reuse_iters_transform.value());
  }
  VLOG(4) << "Source iters after reuse: "
          << PrintFusionIters(reused_source_iters);

  PADDLE_ENFORCE_EQ(
      AllFirstInSecond(reused_source_iters, target_iters),
      true,
      ::common::errors::PreconditionNotMet("The reused source iters should not "
                                           "contains element not in target."));
  const auto reused_target_unique_iters =
      GatherFirstNotInSecond(target_iters, reused_source_iters);

  // 3. Apply AppendItersTransform
  // if exist iters in target can not find in source
  FusionIters appended_source_iters = reused_source_iters;
  if (!reused_target_unique_iters.empty()) {
    if (!transform_strategy_[ItersTransformType::AppendIters] ||
        !FLAGS_enable_append_iters_in_fusion) {
      VLOG(4) << "Can not append iters in fusion, because of AppendIters "
                 "transform is disabled.";
      return std::nullopt;
    }
    std::vector<int32_t> append_axis;
    std::vector<symbol::DimExpr> append_symbols;
    for (const auto& iter : reused_target_unique_iters) {
      const size_t pos =
          std::find(target_iters.begin(), target_iters.end(), iter) -
          target_iters.begin();
      append_axis.push_back(pos);
      append_symbols.push_back(iters_manager_->GetIterSymbol(iter));
      appended_source_iters.insert(appended_source_iters.begin() + pos, iter);
    }
    iters_transforms.push_back(
        AppendItersTransform(append_axis, append_symbols));
    if (appended_source_iters == target_iters) {
      return iters_transforms;
    }
  }
  VLOG(4) << "source iters after reuse and append: "
          << PrintFusionIters(appended_source_iters);

  PADDLE_ENFORCE_EQ(ToSet(appended_source_iters),
                    ToSet(target_iters),
                    ::common::errors::PreconditionNotMet(
                        "The source iters after reuse and append "
                        "should have same elements as target."));

  // 4. Apply TransposeItersTransform
  // if source iters after reuse and append are not equal to target
  if (appended_source_iters != target_iters) {
    if (!transform_strategy_[ItersTransformType::TransposeIters] ||
        !FLAGS_enable_transpose_iters_in_fusion) {
      VLOG(4) << "Can not transpose iters in fusion, because of "
                 "TransposeIters transform is disabled";
      return std::nullopt;
    }
    const auto perm =
        GetTransposePerm<int32_t>(appended_source_iters, target_iters);
    iters_transforms.push_back(TransposeItersTransform(perm));
    return iters_transforms;
  }

  return iters_transforms;
}

std::ostream& operator<<(std::ostream& os, const ItersTransform& trans) {
  os << std::visit([](auto&& arg) { return arg.DebugStr(); }, trans);
  return os;
}

std::string DebugStrItersTransformRoute(const ItersTransformRoute& route) {
  return cinn::utils::Join(route, " -> ");
}

}  // namespace cinn::fusion
