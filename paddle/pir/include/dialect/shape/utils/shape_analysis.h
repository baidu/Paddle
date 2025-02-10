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

#pragma once

#include <memory>
#include <optional>
#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/core/builtin_type_interfaces.h"
#include "paddle/pir/include/core/dll_decl.h"
#include "paddle/pir/include/core/utils.h"
#include "paddle/pir/include/dialect/shape/ir/shape_op.h"
#include "paddle/pir/include/dialect/shape/utils/constraints_manager.h"
#include "paddle/pir/include/dialect/shape/utils/dim_expr_builder.h"
#include "paddle/pir/include/dialect/shape/utils/shape_or_data_expr.h"

namespace pir {
using InferSymbolicShapeCacheValue = std::vector<symbol::ShapeOrDataDimExprs>;

enum TransLayoutType { NCHW2NHWC, NHWC2NCHW, INVALID };

/**
 * This class represents information needed to determine the output
 * shape of an operator, which includes the operator's name, input shapes, and
 * attributes.
 */
class IR_API InferSymbolicShapeCacheKey {
 public:
  InferSymbolicShapeCacheKey(
      const std::string& op_name,
      const std::vector<symbol::ShapeOrDataDimExprs>& input_shape_or_datas,
      const std::map<std::string, Attribute>& attributes)
      : op_name_(op_name),
        input_shape_or_datas_(input_shape_or_datas),
        attributes_(attributes) {}
  bool operator==(const InferSymbolicShapeCacheKey& other) const;
  std::size_t GetHashValue() const;
  friend std::ostream& operator<<(std::ostream& os,
                                  const InferSymbolicShapeCacheKey& info);
  friend class InferSymbolicShapeContext;

 private:
  std::string op_name_;
  std::vector<symbol::ShapeOrDataDimExprs> input_shape_or_datas_;
  std::map<std::string, Attribute> attributes_;
  const std::vector<symbol::ShapeOrDataDimExprs>& GetInputShapeOrDatas() const;
  void SetInputShapeOrDatas(
      const std::vector<symbol::ShapeOrDataDimExprs>& input_shape_or_datas);
};

struct InputDynamicDimSpec {
  std::string dim_name;
  // input_bind = [(input_name, dim_index)]
  std::vector<std::pair<std::string, int>> input_bind;
  symbol::ConstraintsManager::Range range;
};
}  // namespace pir

namespace std {
template <>
struct hash<pir::InferSymbolicShapeCacheKey> {
  std::size_t operator()(const pir::InferSymbolicShapeCacheKey& obj) const {
    return obj.GetHashValue();
  }
};
}  // namespace std

namespace pir {
void InferSymExprForAllValues(ModuleOp module_op);

class IR_API InferSymbolicShapeContext {
 public:
  InferSymbolicShapeContext() = default;
  InferSymbolicShapeContext(const InferSymbolicShapeContext&) = delete;
  InferSymbolicShapeContext(InferSymbolicShapeContext&&) = delete;
  void Init(const std::vector<InputDynamicDimSpec>& input_dynamic_dim_spec);

  // Note: Only initialize the symbol info, the value info is not update.
  void RegisterSymbolConstraintFromContext(
      const InferSymbolicShapeContext& other);

  const std::string GetNextSymName();

  bool HasShapeOrDataForValue(Value val) const;

  const symbol::ShapeOrDataDimExprs& GetShapeOrDataForValue(Value val) const;

  void SetSymbolForValueByStaticShape(Value val);

  void SetShapeOrDataForValue(Value val,
                              const symbol::ShapeOrDataDimExprs& shape_or_data);

  void AddEqualCstr(const symbol::DimExpr& lhs, const symbol::DimExpr& rhs);

  // Add equal constraints for each dim in lhs and rhs.
  void AddEqualCstr(const std::vector<symbol::DimExpr>& lhs,
                    const std::vector<symbol::DimExpr>& rhs);

  bool IsEqual(const symbol::DimExpr& lhs, const symbol::DimExpr& rhs) const;

  // Returns true if:
  //    lhs[i] == rhs[i] for all i
  bool IsEqual(const std::vector<symbol::DimExpr>& lhs,
               const std::vector<symbol::DimExpr>& rhs) const;

  void AddGreatThanOneCstr(const symbol::DimExpr& dim_expr);

  bool IsGreatThanOne(const symbol::DimExpr& dim_expr) const;

  void AddBroadcastableCstr(const symbol::DimExpr& lhs,
                            const symbol::DimExpr& rhs);

  bool IsBroadcastable(const symbol::DimExpr& lhs,
                       const symbol::DimExpr& rhs) const;

  bool HasPredefinedRange(const symbol::DimExpr& dim_expr) const;

  void PrintShapeOrDatas() const;

  void SetOpInferSymbolicShapeCache(
      const InferSymbolicShapeCacheKey& op_infer_cache_key,
      InferSymbolicShapeCacheValue result_shape);

  std::optional<InferSymbolicShapeCacheValue> GetOpInferSymbolicShapeCache(
      const InferSymbolicShapeCacheKey& op_infer_cache_key) const;

  const symbol::ConstraintsManager& constraints_manager() const {
    return constraints_manager_;
  }

  struct DimIndexAndExpr {
    int index;
    symbol::DimExpr dim_expr;
    DimIndexAndExpr(int index_val, const symbol::DimExpr& dim_expr_val)
        : index(index_val), dim_expr(dim_expr_val) {}
  };

  bool HasPredefinedDimExprForInputName(const std::string& input_name) const;

  const std::vector<DimIndexAndExpr> GetPredefinedDimExprForInputName(
      const std::string& input_name) const;

 private:
  symbol::ShapeOrDataDimExprs SimplifyBroadcastForShapeOrData(
      const symbol::ShapeOrDataDimExprs& shape_or_data);

  void SubstituteDimExpr(const symbol::DimExpr& origin,
                         const symbol::DimExpr& substituted);

  int64_t sym_idx_begin_ = 0;
  int64_t next_sym_idx_ = 0;

  std::unordered_map<uint64_t, symbol::ShapeOrDataDimExprs>
      value_id_to_shape_or_data_;

  symbol::ConstraintsManager constraints_manager_;

  using DimExprSubstitutionPattern =
      std::unordered_map<symbol::DimExpr, symbol::DimExpr>;
  DimExprSubstitutionPattern substitution_pattern_;

  std::unordered_map<InferSymbolicShapeCacheKey, InferSymbolicShapeCacheValue>
      infer_symbolic_shape_cache_;

  std::unordered_map<std::string, std::vector<DimIndexAndExpr>>
      predefined_dimexpr_map_for_inputs_;

  std::unordered_map<std::string, symbol::DimExpr>
      input_dynamic_dim_name_spec_to_dimexpr_map_;
};

class IR_API ShapeConstraintIRAnalysis final
    : public std::enable_shared_from_this<ShapeConstraintIRAnalysis> {
 public:
  ShapeConstraintIRAnalysis() = default;
  ShapeConstraintIRAnalysis(const ShapeConstraintIRAnalysis&) = delete;
  ShapeConstraintIRAnalysis(ShapeConstraintIRAnalysis&&) = delete;
  void InitInferContext();

  void RegisterSymbolConstraintFromShapeAnalysis(
      const ShapeConstraintIRAnalysis& other);

  const std::string GetNextSymName();

  bool HasShapeOrDataForValue(Value val) const;

  const symbol::ShapeOrDataDimExprs& GetShapeOrDataForValue(Value val);

  void SetShapeOrDataForValue(Value val,
                              const symbol::ShapeOrDataDimExprs& shape_or_data);

  // Set ShapeOrData of `to` value by ShapeOrData of `from` value.
  void ShareShapeOrData(Value from, Value to);

  // Update Symbol Shape for value by layout transformation.
  void UpdateShapeOrDataByTransLayout(Value val,
                                      TransLayoutType trans_layout_type);

  void AddEqualCstr(const symbol::DimExpr& lhs, const symbol::DimExpr& rhs);

  bool IsEqual(const symbol::DimExpr& lhs, const symbol::DimExpr& rhs) const;

  bool IsGreatThanOne(const symbol::DimExpr& dim_expr) const;

  bool IsBroadcastable(const symbol::DimExpr& lhs,
                       const symbol::DimExpr& rhs) const;

  // Used to debug
  void PrintShapeOrDatas() const;

  // Returns true if the two value have the same symbolic shape.
  bool IsShapeEqual(Value lhs, Value rhs);

  // Suppose:
  //    lhs_dim_idxs = {ld0, ld1, ...}
  //    rhs_dim_idxs = {rd0, rd1, ...}
  // Returns true if:
  //    lhs.shape[ld0] * lhs.shape[ld1] * ... ==
  //    rhs.shape[rd0] * rhs.shape[rd1] * ...
  bool IsProductEqual(Value lhs,
                      const std::vector<int>& lhs_dim_idxs,
                      Value rhs,
                      const std::vector<int>& rhs_dim_idxs);

  // Returns true if:
  //    lhs.shape[lhs_from] * ... lhs.shape[lhs_to-1] ==
  //    rhs.shape[rhs_from] * ... rhs.shape[rhs_to-1]
  bool IsProductEqual(
      Value lhs, int lhs_from, int lhs_to, Value rhs, int rhs_from, int rhs_to);

  // Returns true if the two value have the same number elements.
  bool IsSameNumel(Value lhs, Value rhs);

  pir::PrintHooks PrintHook();

  symbol::DimExpr GetProductDimExpr(Value lhs,
                                    const std::vector<int>& lhs_dim_idxs);

  const symbol::ConstraintsManager& constraints_manager() const {
    return context_.constraints_manager();
  }

  void SetInputDynamicDimSpec(
      const std::vector<InputDynamicDimSpec>& input_dynamic_dim_spec);

 private:
  InferSymbolicShapeContext* MutInferSymbolicShapeContext() {
    return &context_;
  }

  friend void InferSymExprForAllValues(ModuleOp module_op);

  void SetSymbolForValueByStaticShape(Value val);

  void InferShapeOrDataForValue(Value val);

 private:
  InferSymbolicShapeContext context_;
  std::vector<InputDynamicDimSpec> input_dynamic_dim_spec_;
};

class IR_API ShapeAnalysisManager {
 public:
  static ShapeAnalysisManager& Instance();
  ShapeConstraintIRAnalysis& Get(const pir::Program* program);

  ShapeAnalysisManager(const ShapeAnalysisManager&) = delete;
  ShapeAnalysisManager(ShapeAnalysisManager&&) = delete;
  ShapeAnalysisManager& operator=(const ShapeAnalysisManager&) = delete;

 private:
  ShapeAnalysisManager() {}
  std::unordered_map<uint64_t, std::shared_ptr<ShapeConstraintIRAnalysis>>
      tables_;
};

#define OP_DECLARE_INFER_SYMBOLIC_SHAPE(name) \
  bool name##OpInferSymbolicShape(            \
      pir::Operation* op, pir::InferSymbolicShapeContext* infer_context);

bool IsStaticShape(const Value& value);

}  // namespace pir
