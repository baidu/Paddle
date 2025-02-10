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
#include <functional>
#include <memory>
#include <optional>
#include <set>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

#include "glog/logging.h"
#include "paddle/cinn/hlir/framework/op.h"
#include "paddle/cinn/hlir/framework/pir/utils.h"
#include "paddle/cinn/utils/string.h"
#include "paddle/common/enforce.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_op.h"
#include "paddle/pir/include/dialect/shape/utils/shape_analysis.h"

namespace cinn::fusion {

using OpPatternKind = cinn::hlir::framework::OpPatternKind;
static OpPatternKind GetOpPatternKind(const ::pir::Operation* op) {
  return hlir::framework::pir::CompatibleInfo::OpKind(*op);
}

static std::string GetNewTmpId(std::string origin_id) {
  if (origin_id.find('_tmp') == std::string::npos) {
    return origin_id + "_tmp_0";
  } else {
    int ith = std::stoi(origin_id.substr(origin_id.size() - 1));
    return origin_id.substr(0, origin_id.size() - 1) + std::to_string(ith + 1);
  }
}

static size_t GetRank(pir::Value value) {
  PADDLE_ENFORCE_EQ(value.type().isa<pir::DenseTensorType>(),
                    true,
                    ::common::errors::InvalidArgument(
                        "The type of value should be a DenseTensorType."));
  return value.type().dyn_cast<pir::DenseTensorType>().dims().size();
}

// FIXME(Aurelius84): 0D Tensor is not compatible with other rank.
// So we need to add a special case for 0D Tensor.
static size_t GetCompatibleRank(pir::Value value) {
  size_t rank = GetRank(value);
  return rank == 0 ? 1 : rank;
}

std::vector<int64_t> GetInt64ArrayAttributeData(
    const ::pir::Attribute& attr_val);

std::vector<int32_t> GetInt32ArrayAttributeData(
    const ::pir::Attribute& attr_val);

std::vector<int64_t> GetReduceAxisIdx(pir::Operation* reduce_op);

std::pair<std::vector<int64_t>, bool> GetSliceAxis(pir::Operation* slice_op);

bool GetReduceOpKeepDims(pir::Operation* reduce_op);

std::optional<std::pair<pir::Value, pir::Value>> GetBroadcastOpInputOutputValue(
    pir::Operation* op);

std::vector<std::pair<size_t, size_t>> GetNonBroadCastDims(pir::Operation* op);

static std::string OpsDebugStr(std::vector<pir::Operation*> ops) {
  std::stringstream ss;
  pir::IrPrinter printer(ss);
  for (const auto* op : ops) {
    printer.PrintOperation(*op);
    ss << "(" << op << ")"
       << "\n";
  }
  return ss.str();
}

template <typename T>
void RemoveFromVector(std::vector<T>* vec, T item) {
  auto iter = std::find(vec->begin(), vec->end(), item);
  if (iter != vec->end()) {
    vec->erase(iter);
  }
}

template <typename T>
std::vector<T> ConcatVector(const std::vector<T>& first,
                            const std::vector<T>& second) {
  std::vector<T> result = first;
  result.insert(result.end(), second.begin(), second.end());
  return result;
}

template <typename T>
std::vector<T> ReverseVector(const std::vector<T>& as) {
  std::vector<T> result = as;
  std::reverse(result.begin(), result.end());
  return result;
}

template <typename T>
std::vector<T> ConcatAll(const std::vector<std::vector<T>>& all) {
  std::vector<T> result;
  for (const auto& vec : all) {
    result = ConcatVector(result, vec);
  }
  return result;
}

template <typename T, typename F>
std::vector<T> FilterVector(const std::vector<T>& first, const F& func) {
  std::vector<T> result;
  for (const auto& i : first) {
    if (func(i)) {
      result.push_back(i);
    }
  }
  return result;
}

template <typename T, typename F = std::function<bool(T, T)>>
bool VectorEqual(const std::vector<T>& first,
                 const std::vector<T>& second,
                 const F& func = nullptr) {
  if (first.size() != second.size()) {
    return false;
  }
  for (size_t i = 0; i < first.size(); ++i) {
    if (func) {
      if (!func(first[i], second[i])) {
        return false;
      }
    } else {
      if (first[i] != second[i]) {
        return false;
      }
    }
  }
  return true;
}

template <class A, class B>
std::vector<B> MapVector(const std::vector<A>& as,
                         const std::function<B(A)>& func) {
  std::vector<B> res;
  for (const auto& a : as) {
    res.push_back(func(a));
  }
  return res;
}

template <class A, class B>
std::vector<B> MapVectorIfTrue(const std::vector<A>& as,
                               const std::function<B(A)>& func,
                               const std::function<bool(A)>& pred) {
  std::vector<B> res;
  for (const auto& a : as) {
    if (pred(a)) {
      res.push_back(func(a));
    }
  }
  return res;
}

template <class A>
std::vector<std::pair<A, int>> Enumerate(const std::vector<A>& inputs) {
  std::vector<std::pair<A, int>> res;
  int idx = 0;
  for (const auto& a : inputs) {
    res.push_back(std::make_pair(a, idx));
    idx++;
  }
  return res;
}

template <typename T>
std::set<T> ToSet(const std::vector<T>& input) {
  std::set<T> result(input.begin(), input.end());
  return result;
}

template <typename T>
std::unordered_set<T> ToUnorderedSet(const std::vector<T>& input) {
  std::unordered_set<T> result(input.begin(), input.end());
  return result;
}

template <typename T>
std::vector<T> SetToVector(const std::set<T>& input) {
  std::vector<T> result(input.begin(), input.end());
  return result;
}
template <typename T>
std::vector<T> SetToVector(const std::unordered_set<T>& input) {
  std::vector<T> result(input.begin(), input.end());
  return result;
}

template <typename T1, typename T2>
std::vector<T1> MapKeyToVector(const std::map<T1, T2>& input) {
  std::vector<T1> result;
  for (const auto& pair : input) {
    result.push_back(pair.first);
  }
  return result;
}
template <typename T1, typename T2>
std::vector<T1> MapKeyToVector(const std::unordered_map<T1, T2>& input) {
  std::vector<T1> result;
  for (const auto& pair : input) {
    result.push_back(pair.first);
  }
  return result;
}

template <typename Set>
Set SetUnion(const Set& A, const Set& B) {
  Set result;
  std::set_union(A.begin(),
                 A.end(),
                 B.begin(),
                 B.end(),
                 std::inserter(result, result.begin()));
  return result;
}
template <typename Set>
Set SetIntersection(const Set& A, const Set& B) {
  Set result;
  std::set_intersection(A.begin(),
                        A.end(),
                        B.begin(),
                        B.end(),
                        std::inserter(result, result.begin()));
  return result;
}
template <typename Set>
Set SetDifference(const Set& A, const Set& B) {
  Set result;
  std::set_difference(A.begin(),
                      A.end(),
                      B.begin(),
                      B.end(),
                      std::inserter(result, result.begin()));
  return result;
}

template <typename T>
bool IsAnyFirstInSecond(const std::vector<T>& first,
                        const std::vector<T>& second) {
  const auto& second_set = ToSet(second);
  for (const auto& ele : first) {
    if (second_set.count(ele)) {
      return true;
    }
  }
  return false;
}

template <typename T>
std::pair<std::vector<T>, std::vector<T>> SplitFirstWhetherInSecond(
    const std::vector<T>& first, const std::vector<T>& second) {
  std::vector<T> used;
  std::vector<T> unused;
  for (size_t i = 0; i < first.size(); ++i) {
    if (std::find(second.begin(), second.end(), first[i]) != second.end()) {
      used.emplace_back(first[i]);
    } else {
      unused.emplace_back(first[i]);
    }
  }
  return {used, unused};
}

template <typename T>
std::vector<T> GatherFirstNotInSecond(const std::vector<T>& first,
                                      const std::vector<T>& second) {
  std::vector<T> result;
  for (size_t i = 0; i < first.size(); ++i) {
    if (std::find(second.begin(), second.end(), first[i]) == second.end()) {
      result.emplace_back(first[i]);
    }
  }
  return result;
}

template <typename T>
std::vector<T> UniqueVectorBySet(const std::vector<T>& v) {
  std::unordered_set<T> unique(v.begin(), v.end());
  return std::vector<T>(unique.begin(), unique.end());
}

template <typename T>
void ExtendVector(std::vector<T>* first, const std::vector<T>& second) {
  std::unordered_set<T> visited =
      std::unordered_set<T>(first->begin(), first->end());
  for (auto iter = second.begin(); iter != second.end(); iter++) {
    if (visited.find(*iter) == visited.end()) {
      visited.emplace(*iter);
      first->emplace_back(*iter);
    }
  }
}

template <typename T>
std::vector<T> UniqueConcatVector(const std::vector<T>& first,
                                  const std::vector<T>& second) {
  std::vector<T> result = std::vector<T>(first);
  ExtendVector(&result, second);
  return result;
}

template <typename Int, typename T>
std::vector<Int> GetTransposePerm(const std::vector<T>& source,
                                  const std::vector<T>& target) {
  PADDLE_ENFORCE_EQ(source.size(),
                    target.size(),
                    ::common::errors::InvalidArgument(
                        "The size of source and target should be equal."));
  std::vector<Int> perm;
  for (size_t i = 0; i < source.size(); ++i) {
    auto iter = std::find(source.begin(), source.end(), target[i]);
    PADDLE_ENFORCE_NE(iter,
                      source.end(),
                      ::common::errors::InvalidArgument(
                          "The target should contain all elements in source."));
    perm.emplace_back(iter - source.begin());
  }
  return perm;
}

template <typename T, typename Int>
std::vector<T> TransposeVector(const std::vector<T>& v,
                               const std::vector<Int>& perm) {
  PADDLE_ENFORCE_EQ(
      v.size(),
      perm.size(),
      ::common::errors::InvalidArgument(
          "The size of transpose vector and perm should be equal."));
  std::vector<T> result;
  for (size_t i = 0; i < perm.size(); ++i) {
    result.emplace_back(v[perm[i]]);
  }
  return result;
}

struct ValueDim {
  pir::Value v_;
  size_t idx_ = -1;
  std::weak_ptr<pir::ShapeConstraintIRAnalysis> shape_analysis_;
  ValueDim(pir::Value v, size_t idx) : v_(v), idx_(idx) {
    // Just get a related op to get the shape analysis. It can be value's
    // upstream op (defining op) or downstream op (user op).
    const auto GetRelatedOpFromValue =
        [](const pir::Value& v) -> pir::Operation* {
      if (v.defining_op() != nullptr) {
        return v.defining_op();
      }
      // For inputs of the program, the defining_op is nullptr, we use it's
      // user as the related op.
      PADDLE_ENFORCE_EQ(v.use_empty(),
                        false,
                        ::common::errors::PreconditionNotMet(
                            "Value is an input value, it should have a use."));
      return v.first_use().owner();
    };
    shape_analysis_ = pir::ShapeAnalysisManager::Instance()
                          .Get(GetRelatedOpFromValue(v)->GetParentProgram())
                          .shared_from_this();
  }
  ValueDim() = default;
  ValueDim(const ValueDim& v) = default;
  bool operator==(const ValueDim& v) const {
    return (idx_ == v.idx_) && (v_ == v.v_);
  }

  symbol::DimExpr GetSymbolicDim() const {
    PADDLE_ENFORCE_NOT_NULL(v_.impl(), "Empty value is not expected.");
    return shape_analysis().GetProductDimExpr(v_, {static_cast<int>(idx_)});
  }

  bool empty() const { return idx_ == -1; }

  bool SymbolicEqualTo(const ValueDim& other) const {
    return shape_analysis().IsEqual(GetSymbolicDim(), other.GetSymbolicDim());
  }

  std::string DebugStr() const {
    std::ostringstream oss;
    oss << "ValueDim: ";
    oss << "Index: " << idx_;
    oss << ", ";
    v_.defining_op()->Print(oss);
    return oss.str();
  }

  pir::ShapeConstraintIRAnalysis& shape_analysis() const {
    auto shape_analysis_ptr = shape_analysis_.lock();
    PADDLE_ENFORCE_NOT_NULL(
        shape_analysis_ptr,
        ::common::errors::PreconditionNotMet("shape_analysis_ptr is nullptr."));
    return *shape_analysis_ptr;
  }
};

static std::vector<ValueDim> GetAllValueDimFromValue(const pir::Value& v) {
  std::vector<ValueDim> value_dims;
  size_t rank = GetCompatibleRank(v);
  for (size_t i = 0; i < rank; ++i) {
    value_dims.emplace_back(v, i);
  }
  return value_dims;
}

struct ValueDimHash {
  std::size_t operator()(const ValueDim& p) const {
    auto h1 = std::hash<size_t>{}(p.idx_);
    auto h2 = std::hash<pir::Value>{}(p.v_);
    // Mainly for demonstration purposes, i.e. works but is overly simple
    // In the real world, use sth. like boost.hash_combine
    return h1 ^ (h2 << 1);
  }
};

static std::vector<symbol::DimExpr> GetDimExprsFromValue(pir::Value value) {
  const auto& value_dims = GetAllValueDimFromValue(value);

  std::function<symbol::DimExpr(ValueDim)> func =
      [](const ValueDim& value_dim) {
        const auto& symbolic_dim = value_dim.GetSymbolicDim();
        return symbolic_dim;
      };
  return MapVector(value_dims, func);
}

template <typename T, typename Int>
std::vector<T> GatherVector(const std::vector<T>& inp,
                            std::vector<Int> gathers) {
  std::vector<T> result;
  for (auto i : gathers) {
    result.push_back(inp[i]);
  }
  return result;
}

template <typename Int>
std::vector<Int> ExcludeIndex(int n, std::vector<Int> excludes) {
  std::vector<Int> result;
  for (int i = 0; i < n; ++i) {
    if (std::find(excludes.begin(), excludes.end(), i) == excludes.end()) {
      result.push_back(i);
    }
  }
  return result;
}

template <typename T, typename U>
std::vector<T> GatherVectorExcept(const std::vector<T>& source,
                                  const std::vector<U>& idx) {
  std::vector<T> result;
  for (U i = 0; i < source.size(); i++) {
    if (std::find(idx.begin(), idx.end(), i) == idx.end()) {
      result.emplace_back(source[i]);
    }
  }
  return result;
}

template <typename T>
std::vector<T> SliceVector(const std::vector<T>& inp, int start, int end) {
  if (start < 0) {
    start = inp.size() + start;
  }
  if (end < 0) {
    end = inp.size() + end;
  }
  std::vector<T> result;
  for (int i = start; i < end; ++i) {
    result.push_back(inp[i]);
  }
  return result;
}

template <typename T, typename U>
std::vector<U> VectorFlatMap(
    const std::vector<T>& inp,
    const std::function<std::vector<U>(const T&)>& func) {
  std::vector<U> result;
  for (const auto& i : inp) {
    result = ConcatVector(result, func(i));
  }
  return result;
}

template <typename T>
bool AnyFirstInSecond(const std::vector<T>& first,
                      const std::vector<T>& second) {
  std::unordered_set<T> pool = ToUnorderedSet(second);
  for (const auto& item : first) {
    if (pool.find(item) != pool.end()) {
      return true;
    }
  }
  return false;
}

template <typename T>
bool AllFirstInSecond(const std::vector<T>& first,
                      const std::vector<T>& second) {
  std::unordered_set<T> pool = ToUnorderedSet(second);
  for (const auto& item : first) {
    if (pool.find(item) == pool.end()) {
      return false;
    }
  }
  return true;
}

static std::vector<pir::Operation*> FindDownstreamOps(pir::Operation* op) {
  std::vector<pir::Operation*> result;
  for (int i = 0; i < op->num_results(); i++) {
    auto v = op->result(i);
    for (auto consumer_it = v.use_begin(); consumer_it != v.use_end();
         ++consumer_it) {
      result.emplace_back(consumer_it->owner());
    }
  }
  return result;
}

static const size_t GetUsageIdx(const pir::Value& v, pir::Operation* op) {
  size_t i = 0;
  for (auto consumer_it = v.use_begin(); consumer_it != v.use_end();
       ++consumer_it, ++i) {
    if (consumer_it->owner() == op) {
      return i;
    }
  }
  PADDLE_THROW(::common::errors::NotFound(
      "Can not find the usage of value %s in op %s", v.impl(), op->name()));
}

static const size_t GetOperandIdx(const pir::Value& v, pir::Operation* op) {
  for (size_t i = 0; i < op->num_operands(); i++) {
    if (op->operand(i).source() == v) {
      return i;
    }
  }
  PADDLE_THROW(::common::errors::NotFound(
      "Can not find the value %s as operand of op %s", v.impl(), op->name()));
}

static const size_t GetResultIdx(const pir::Value& v, pir::Operation* op) {
  size_t i = 0;
  for (size_t i = 0; i < op->num_results(); i++) {
    if (op->result(i) == v) {
      return i;
    }
  }
  PADDLE_THROW(::common::errors::NotFound(
      "Can not find the value %s as result of op %s", v.impl(), op->name()));
}

static std::vector<pir::Operation*> FindUserOp(
    const std::vector<pir::Operation*>& candidates, const pir::Value& value) {
  std::vector<pir::Operation*> results;
  for (auto consumer_it = value.use_begin(); consumer_it != value.use_end();
       ++consumer_it) {
    pir::Operation* user_op = consumer_it.owner();
    auto iter = std::find(candidates.begin(), candidates.end(), user_op);
    if (iter != candidates.end()) {
      results.emplace_back(*iter);
    }
  }
  return results;
}

static bool IsDirectUpstream(const pir::Operation* upstream,
                             const pir::Operation* downstream) {
  for (const auto& value : upstream->results()) {
    for (const auto& operand : downstream->operands()) {
      if (value == operand.source()) {
        return true;
      }
    }
  }
  return false;
}

inline std::vector<pir::Value> GetInputsValue(
    const std::vector<pir::Operation*>& ops) {
  // include middle value.
  std::function<std::vector<pir::Value>(pir::Operation* const&)> get_inputs =
      [](const pir::Operation* const& in) { return in->operands_source(); };
  const auto& all_inputs =
      VectorFlatMap<pir::Operation*, pir::Value>(ops, get_inputs);
  return UniqueVectorBySet(all_inputs);
}

inline std::vector<pir::Value> GetOutputsValue(
    const std::vector<pir::Operation*>& ops) {
  // include middle value.
  std::function<std::vector<pir::Value>(pir::Operation* const&)> get_outputs =
      [](const pir::Operation* const& in) { return in->results(); };
  const auto& all_outputs =
      VectorFlatMap<pir::Operation*, pir::Value>(ops, get_outputs);
  return UniqueVectorBySet(all_outputs);
}

template <typename T>
std::vector<T> VectorDiff(const std::vector<T>& left,
                          const std::vector<T>& right) {
  const auto& set = ToSet(right);
  std::vector<T> res;
  for (const auto& v : left) {
    if (!set.count(v)) res.push_back(v);
  }
  return res;
}

inline bool All(const std::vector<bool> a) {
  bool res = true;
  for (bool i : a) {
    res &= i;
  }
  return res;
}

inline bool Any(const std::vector<bool> a) {
  bool res = false;
  for (bool i : a) {
    res |= i;
  }
  return res;
}

template <typename Int>
std::vector<Int> ArangeVector(Int start, Int end, Int step = 1) {
  std::vector<Int> res;
  for (Int i = start; i < end; i += step) {
    res.push_back(i);
  }
  return res;
}

symbol::DimExpr GetShapeProduct(const std::vector<symbol::DimExpr>& shape,
                                int start,
                                int end);

bool ShapeProductEqual(const std::vector<symbol::DimExpr>& in_shape,
                       const std::vector<symbol::DimExpr>& out_shape,
                       int in_start,
                       int in_end,
                       int out_start,
                       int out_end);

std::vector<std::pair<int, int>> PartitionReshapeAxes(
    const std::vector<symbol::DimExpr>& in_shape,
    const std::vector<symbol::DimExpr>& out_shape);

}  // namespace cinn::fusion
