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

#include "paddle/fluid/pir/transforms/general/common_subexpression_elimination_pass.h"
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "paddle/common/flags.h"
#include "paddle/fluid/pir/dialect/operator/ir/control_flow_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/dialect/operator/utils/op_yaml_info_parser.h"
#include "paddle/pir/include/core/block.h"
#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/core/op_trait.h"
#include "paddle/pir/include/core/utils.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_op.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

COMMON_DECLARE_int32(cse_max_count);
#ifdef PADDLE_WITH_CINN
COMMON_DECLARE_bool(use_cinn);
#endif

namespace {

// TODO(SigureMo): Consider use interface to check whether the operation is
// commutative.
static std::unordered_map<std::string, std::vector<std::vector<size_t>>>
    kCommutativeOps = {
        {paddle::dialect::MultiplyOp::name(), {{0, 1}}},
        {paddle::dialect::AddOp::name(), {{0, 1}}},
        {paddle::dialect::MaximumOp::name(), {{0, 1}}},
        {paddle::dialect::MinimumOp::name(), {{0, 1}}},
        {paddle::dialect::LogicalAndOp::name(), {{0, 1}}},
        {paddle::dialect::LogicalOrOp::name(), {{0, 1}}},
        {paddle::dialect::LogicalXorOp::name(), {{0, 1}}},
        {paddle::dialect::EqualOp::name(), {{0, 1}}},
        {paddle::dialect::NotEqualOp::name(), {{0, 1}}},
        {paddle::dialect::BitwiseOrOp::name(), {{0, 1}}},
        {paddle::dialect::BitwiseXorOp::name(), {{0, 1}}},
        {paddle::dialect::BitwiseAndOp::name(), {{0, 1}}},
};

// If an op results used by these ops, we should not replace it.
static std::unordered_set<std::string> kBlockedUserOps = {
    pir::YieldOp::name(),
    paddle::dialect::WhileOp::name(),  // while op will trigger a early gc bug
};

bool IsDenseTensorOrVectorOfDenseTensorType(const pir::Value& value) {
  if (value.type().isa<paddle::dialect::DenseTensorType>()) {
    return true;
  }
  if (!value.type().isa<pir::VectorType>()) {
    return false;
  }
  auto type_vec = value.type().dyn_cast<pir::VectorType>().data();
  return std::all_of(type_vec.begin(), type_vec.end(), [](pir::Type type) {
    return type.isa<paddle::dialect::DenseTensorType>();
  });
}

template <typename T>
std::vector<T> SortElementsAtIndices(
    const std::vector<T>& vec,
    const std::vector<size_t>& indices,
    std::function<bool(const T&, const T&)> cmp_fn =
        [](const T& lhs, const T& rhs) { return lhs < rhs; }) {
  std::vector<T> selected_elements;
  for (auto& idx : indices) {
    PADDLE_ENFORCE_LT(
        idx,
        vec.size(),
        common::errors::OutOfRange(
            "The index %d is out of vector size %d.", idx, vec.size()));
    selected_elements.push_back(vec[idx]);
  }
  std::sort(selected_elements.begin(), selected_elements.end(), cmp_fn);
  std::vector<T> sorted_vec;
  for (size_t i = 0; i < vec.size(); ++i) {
    if (std::find(indices.begin(), indices.end(), i) != indices.end()) {
      sorted_vec.push_back(selected_elements.front());
      selected_elements.erase(selected_elements.begin());
    } else {
      sorted_vec.push_back(vec[i]);
    }
  }
  return sorted_vec;
}

std::map<int, int> GetOpInplaceInfo(const pir::Operation* op) {
  std::map<int, int> inplace_info;
  if (!op->HasTrait<paddle::dialect::InplaceTrait>()) {
    return inplace_info;
  }
  pir::IrContext* ctx = pir::IrContext::Instance();
  std::string op_name = op->name();
  if (op->attributes().count("op_name")) {
    op_name =
        op->attributes().at("op_name").dyn_cast<pir::StrAttribute>().AsString();
  }

  pir::OpInfo op_info = ctx->GetRegisteredOpInfo(op_name);
  auto yaml_info_interface =
      op_info.GetInterfaceImpl<paddle::dialect::OpYamlInfoInterface>();
  if (!yaml_info_interface) {
    return inplace_info;
  }
  paddle::dialect::OpYamlInfoParser yaml_parser(
      yaml_info_interface->get_op_info_(op_name),
      paddle::dialect::IsLegacyOp(op_name));

  for (size_t i = 0; i < op->num_results(); ++i) {
    std::string value_name = yaml_parser.OutputNames()[i];
    if (yaml_parser.HasInplace(value_name)) {
      const std::string& inplace_name = yaml_parser.InplaceName(value_name);
      inplace_info[i] = yaml_parser.InputName2Id().at(inplace_name);
    }
    if (yaml_parser.HasView(value_name)) {
      const std::string& view_name = yaml_parser.ViewName(value_name);
      inplace_info[i] = yaml_parser.InputName2Id().at(view_name);
    }
  }

  return inplace_info;
}

bool IsTerminateOp(pir::Operation* op) {
  bool res = op->isa<paddle::dialect::DataOp>() ||
             op->isa<pir::ParameterOp>() || op->isa<pir::ConstantTensorOp>();
#ifdef PADDLE_WITH_CINN
  // In CINN mode, if an OP has no inputs, it can usually be fused with
  // neighboring OPs, such as the FullOp.
  // Eliminating it would make fusion impossible, so we make it a terminate
  // node to avoid eliminating it.
  res = res || (FLAGS_use_cinn && op->num_operands() == 0);
#endif
  return res;
}
bool IsTerminateValue(const pir::Value& value) {
  return !value.defining_op() || IsTerminateOp(value.defining_op());
}

struct Expression {
 public:
  Expression(
      pir::Operation* op,
      std::unordered_map<void*, std::pair<size_t, bool>>* op_info_registry)
      : op_(op), op_info_registry_(op_info_registry) {}

  pir::Operation* op() const { return op_; }

  size_t hash() const { return GetOperationHash(op_); }

  bool equal_to(const Expression& other) const {
    if (hash() != other.hash()) {
      // NOTE(SigureMo): This is a default behavior of std::unordered_set on
      // Linux. But it is not guaranteed by the standard. On Windows, it always
      // calls equal_to even if hash is different. So we need to check hash
      // first to avoid expensive equal_to call.
      return false;
    }
    bool is_equal = CheckOperationEqual(op_, other.op());
    if (!is_equal) {
      VLOG(7) << "Hash collision detected. lhs: " << op_->name() << " [" << op_
              << "] hash: " << GetOperationHash(op_)
              << " vs rhs: " << other.op()->name() << " [" << other.op()
              << "] hash: " << GetOperationHash(other.op());
    }
    return is_equal;
  }

  bool CanBeSafeToReplace() const {
    return GetOperationCanBeSafeToReplace(op_);
  }

  std::pair<size_t, bool> CalcOpInfo() {
    return {CalcOperationHash(op_), CalcOperationCanBeSafeToReplace(op_)};
  }

 private:
  size_t GetOperationHash(pir::Operation* op) const {
    PADDLE_ENFORCE_EQ(
        op_info_registry_->count(reinterpret_cast<void*>(op)),
        1,
        common::errors::PreconditionNotMet(
            "The operation %s is not registered in the table.", op->name()));
    return op_info_registry_->at(reinterpret_cast<void*>(op)).first;
  }

  bool GetOperationCanBeSafeToReplace(pir::Operation* op) const {
    PADDLE_ENFORCE_EQ(
        op_info_registry_->count(reinterpret_cast<void*>(op)),
        1,
        common::errors::PreconditionNotMet(
            "The operation %s is not registered in the table.", op->name()));
    return op_info_registry_->at(reinterpret_cast<void*>(op)).second;
  }

  size_t CalcOperationHash(pir::Operation* op) {
    PADDLE_ENFORCE_EQ(
        op_info_registry_->count(reinterpret_cast<void*>(op)),
        0,
        common::errors::PreconditionNotMet(
            "The operation %s is already registered in the table, don't call "
            "CalcOperationHash twice.",
            op->name()));
    // hash(op) = hash(operands) ^ hash(name) ^ hash(attributes)
    size_t hash = 0;
    VLOG(7) << "[CalcOperationHash] op [" << op << "] " << op->name()
            << " start";

    std::vector<size_t> values_hash;
    for (auto& value : op->operands_source()) {
      values_hash.push_back(CalcValueHash(value));
    }
    if (kCommutativeOps.count(op->name())) {
      for (auto& commutative_indices : kCommutativeOps[op->name()]) {
        values_hash = SortElementsAtIndices(values_hash, commutative_indices);
      }
    }
    for (auto& value_hash : values_hash) {
      hash = pir::detail::hash_combine(hash, value_hash);
    }
    hash =
        pir::detail::hash_combine(hash, std::hash<std::string>{}(op->name()));
    for (auto& attr_name : op->info().GetAttributesName()) {
      hash =
          pir::detail::hash_combine(hash, std::hash<std::string>{}(attr_name));
      auto attr = op->attribute(attr_name);
      hash = pir::detail::hash_combine(hash, attr.hash());
    }
    VLOG(7) << "[CalcOperationHash] op [" << op << "] " << op->name()
            << " hash: " << hash;
    return hash;
  }

  size_t CalcValueHash(const pir::Value& value) const {
    // hash(value) = hash(defining_op) ^ value_result_idx
    if (!IsTerminateValue(value)) {
      return pir::detail::hash_combine(GetOperationHash(value.defining_op()),
                                       GetOpResultId(value));
    }
    // hash(terminate_value) = terminate_value_id
    return reinterpret_cast<size_t>(value.impl());
  }

  size_t GetOpResultId(const pir::Value& value) const {
    size_t value_id = 0;
    for (auto& result : value.defining_op()->results()) {
      if (value == result) {
        break;
      }
      value_id++;
    }
    return value_id;
  }

  bool CalcOperationCanBeSafeToReplace(pir::Operation* op) {
    PADDLE_ENFORCE_EQ(
        op_info_registry_->count(reinterpret_cast<void*>(op)),
        0,
        common::errors::PreconditionNotMet(
            "The operation %s is already registered in the table, don't call "
            "CalcOperationCanBeSafeToReplace twice.",
            op->name()));
    if (op->HasTrait<pir::SideEffectTrait>()) {
      VLOG(7) << "[CalcOperationCanBeSafeToReplace] " << op->name()
              << " has side effect";
      return false;
    }
    for (auto& value : op->results()) {
      if (!IsDenseTensorOrVectorOfDenseTensorType(value)) {
        VLOG(7)
            << "[CalcOperationCanBeSafeToReplace] " << op->name()
            << " has result " << value.defining_op()->name() << " ["
            << value.defining_op()
            << "] which is not DenseTensorType or Vector of DenseTensorType";
        return false;
      }
      for (auto it = value.use_begin(); it != value.use_end(); ++it) {
        if (kBlockedUserOps.count(it->owner()->name())) {
          VLOG(7) << "[CalcOperationCanBeSafeToReplace] " << op->name()
                  << " has result " << value.defining_op()->name() << " ["
                  << value.defining_op() << "] which is used by "
                  << it->owner()->name();
          return false;
        }
        auto inplace_info = GetOpInplaceInfo(it->owner());
        for (auto& [out_idx, in_idx] : inplace_info) {
          if (it->owner()->operands()[in_idx].source() == value) {
            VLOG(7) << "[CalcOperationCanBeSafeToReplace] " << op->name()
                    << " has operand " << value.defining_op()->name() << " ["
                    << value.defining_op() << "] which is inplace to "
                    << it->owner()->name();
            return false;
          }
        }
      }
    }
    for (auto& value : op->operands_source()) {
      if (!IsTerminateValue(value) &&
          !GetOperationCanBeSafeToReplace(value.defining_op())) {
        VLOG(7) << "[CalcOperationCanBeSafeToReplace] " << op->name()
                << " has operand with defining op "
                << value.defining_op()->name() << " [" << value.defining_op()
                << "] which can not be safe to replace";
        return false;
      }
    }
    VLOG(7) << "[CalcOperationCanBeSafeToReplace] " << op->name() << " [" << op
            << "] can be safe to replace";
    return true;
  }

  bool CheckOperationEqual(pir::Operation* lhs, pir::Operation* rhs) const {
    VLOG(7) << "[CheckOperationEqual] lhs [" << lhs << "] " << lhs->name()
            << " vs rhs [" << rhs << "] " << rhs->name();
    if (lhs == rhs) {
      VLOG(7) << "[CheckOperationEqual] lhs [" << lhs << "] " << lhs->name()
              << " vs rhs [" << rhs << "] " << rhs->name() << " equal";
      return true;
    }
    if (lhs->num_regions() > 0 || rhs->num_regions() > 0) {
      VLOG(7) << "[CheckOperationEqual] lhs [" << lhs << "] " << lhs->name()
              << " vs rhs [" << rhs << "] " << rhs->name()
              << " has region, which is not supported";
      return false;
    }
    if (lhs->name() != rhs->name()) {
      VLOG(7) << "[CheckOperationEqual] lhs [" << lhs << "] " << lhs->name()
              << " vs rhs [" << rhs << "] " << rhs->name() << " name not equal";
      return false;
    }
    for (auto attr_name : lhs->info().GetAttributesName()) {
      if (lhs->attribute(attr_name) != rhs->attribute(attr_name)) {
        VLOG(7) << "[CheckOperationEqual] lhs [" << lhs << "] " << lhs->name()
                << " vs rhs [" << rhs << "] " << rhs->name()
                << " attribute not equal: " << attr_name;
        return false;
      }
    }
    if (lhs->num_operands() != rhs->num_operands()) {
      VLOG(7) << "[CheckOperationEqual] lhs [" << lhs << "] " << lhs->name()
              << " vs rhs [" << rhs << "] " << rhs->name()
              << " num_operands not equal";
      return false;
    }

    auto lhs_operands = lhs->operands_source();
    auto rhs_operands = rhs->operands_source();
    if (kCommutativeOps.count(lhs->name())) {
      for (auto& commutative_indices : kCommutativeOps[lhs->name()]) {
        const auto ValueCompare = [&](const pir::Value& lhs,
                                      const pir::Value& rhs) -> bool {
          return CalcValueHash(lhs) < CalcValueHash(rhs);
        };
        lhs_operands = SortElementsAtIndices<pir::Value>(
            lhs_operands, commutative_indices, ValueCompare);
        rhs_operands = SortElementsAtIndices<pir::Value>(
            rhs_operands, commutative_indices, ValueCompare);
      }
    }
    for (size_t i = 0; i < lhs_operands.size(); ++i) {
      if (!CheckValueEqual(lhs_operands[i], rhs_operands[i])) {
        VLOG(7) << "[CheckOperationEqual] lhs [" << lhs << "] " << lhs->name()
                << " vs rhs [" << rhs << "] " << rhs->name() << " operand " << i
                << " not equal";
        return false;
      }
    }

    if (lhs->num_results() != rhs->num_results()) {
      VLOG(7) << "[CheckOperationEqual] lhs [" << lhs << "] " << lhs->name()
              << " vs rhs [" << rhs << "] " << rhs->name()
              << " num_results not equal";
      return false;
    }
    for (size_t i = 0; i < lhs->num_results(); ++i) {
      if (lhs->result_type(i) != rhs->result_type(i)) {
        VLOG(7) << "[CheckOperationEqual] lhs [" << lhs << "] " << lhs->name()
                << " vs rhs [" << rhs << "] " << rhs->name() << " result type "
                << i << " not equal";
        return false;
      }
    }
    VLOG(7) << "[CheckOperationEqual] lhs [" << lhs << "] " << lhs->name()
            << " vs rhs [" << rhs << "] " << rhs->name() << " equal";
    return true;
  }

  bool CheckValueEqual(const pir::Value& lhs, const pir::Value& rhs) const {
    if (IsTerminateValue(lhs) != IsTerminateValue(rhs)) {
      VLOG(7) << "[CheckValueEqual] lhs and rhs has different terminate type";
      return false;
    }
    // Compare two terminate values
    if (IsTerminateValue(lhs) && IsTerminateValue(rhs)) {
      if (lhs != rhs) {
        VLOG(7) << "[CheckValueEqual] lhs and rhs has different terminate "
                   "value";
        return false;
      }
      return true;
    }
    // Compare two non-terminate values
    if (!CheckOperationEqual(lhs.defining_op(), rhs.defining_op())) {
      VLOG(7) << "[CheckValueEqual] lhs and rhs has different defining op";
      return false;
    }
    if (lhs.type() != rhs.type()) {
      VLOG(7) << "[CheckValueEqual] lhs and rhs has different type";
      return false;
    }
    if (GetOpResultId(lhs) != GetOpResultId(rhs)) {
      VLOG(7) << "[CheckValueEqual] lhs and rhs has different result id";
      return false;
    }
    return true;
  }

  pir::Operation* op_;
  std::unordered_map<void*, std::pair<size_t, bool>>*
      op_info_registry_;  // owned by ExpressionTable
};

struct ExpressionHash {
  size_t operator()(const Expression& expr) const { return expr.hash(); }
};

struct ExpressionEqual {
  bool operator()(const Expression& lhs, const Expression& rhs) const {
    return lhs.equal_to(rhs);
  }
};

struct ExpressionTable {
 public:
  ExpressionTable() = default;
  void RegisterExpression(Expression expr) {
    auto op_info = expr.CalcOpInfo();
    VLOG(7) << "[RegisterExpression] op " << expr.op()->name() << " ["
            << expr.op() << "]"
            << "\n  hash: " << op_info.first
            << "\n  can_be_safe_to_replace: " << std::boolalpha
            << op_info.second;
    op_info_registry_[reinterpret_cast<void*>(expr.op())] = op_info;
  }

  Expression CreateExpression(pir::Operation* op) {
    return Expression(op, &op_info_registry_);
  }

  void Insert(Expression expr) { common_exprs_.insert(expr); }

  std::optional<Expression> Lookup(Expression expr) {
    VLOG(7) << "[Lookup] op [" << expr.op() << "] " << expr.op()->name()
            << " start";
    auto found_expr_iter = common_exprs_.find(expr);
    if (found_expr_iter == common_exprs_.end()) {
      return std::nullopt;
    }
    VLOG(7) << "[Lookup] op [" << expr.op() << "] " << expr.op()->name()
            << " found common subexpression: " << found_expr_iter->op()->name();
    return *found_expr_iter;
  }

  void Rehash(size_t size) { common_exprs_.rehash(size); }

 private:
  std::unordered_set<Expression, ExpressionHash, ExpressionEqual> common_exprs_;
  std::unordered_map<void*, std::pair<size_t, bool>> op_info_registry_;
};

struct CSEAnalyzer {
 public:
  CSEAnalyzer() = default;
  void SimplifyOperation(pir::Operation* op,
                         ExpressionTable* expression_table) {
    VLOG(7) << "[SimplifyOperation] op [" << op << "]";
    if (IsTerminateOp(op)) {
      return;
    }

    // Handle sub blocks
    for (auto& region : *op) {
      for (auto& block : region) {
        SimplifyBlock(&block, expression_table);
      }
    }

    // Handle the operation
    auto expr = expression_table->CreateExpression(op);
    expression_table->RegisterExpression(expr);
    auto maybe_same_expression = expression_table->Lookup(expr);
    if (expr.CanBeSafeToReplace()) {
      if (!maybe_same_expression.has_value()) {
        expression_table->Insert(expr);
      } else {
        VLOG(7) << "Found common subexpression: " << op->name();
        to_erase_ops_.push_back(
            std::make_pair(expr.op(), maybe_same_expression.value().op()));
      }
    }
  }

  void SimplifyBlock(pir::Block* block,
                     ExpressionTable* parent_expression_table) {
    // Make a clone to inherit the expressions from parent block
    ExpressionTable expression_table = *parent_expression_table;
    expression_table.Rehash(block->num_ops());

    for (auto& op : *block) {
      SimplifyOperation(&op, &expression_table);
    }
  }

  const std::vector<std::pair<pir::Operation*, pir::Operation*>>& to_erase_ops()
      const {
    return to_erase_ops_;
  };

 private:
  std::vector<std::pair<pir::Operation*, pir::Operation*>> to_erase_ops_;
};

pir::Value CreateAssignOp(const pir::Value& value,
                          pir::Operation* op,
                          pir::Block* block) {
  pir::IrContext* ctx = pir::IrContext::Instance();
  pir::Builder builder(ctx, block);
  builder.SetInsertionPointAfter(op);
  auto assign_op = builder.Build<paddle::dialect::AssignOp>(value);
  return assign_op.result(0);
}

void ReplaceOpWith(pir::Operation* op, pir::Operation* new_op) {
  VLOG(7) << "Replacing op " << op->name() << " [" << op << "] with new op "
          << new_op->name() << " [" << new_op << "]";
  PADDLE_ENFORCE_EQ(
      op->num_results(),
      new_op->num_results(),
      common::errors::InvalidArgument("the num of result should be the same."));
  for (uint32_t i = 0; i < op->num_results(); ++i) {
    auto value = op->result(i);
    auto new_value = new_op->result(i);
    // NOTE(SigureMo): If the value has a shadow output, we could not replace
    // it directly. It will cause a value has two shadow outputs. It is
    // invalid for executor, so we make a copy by inserting a assign op.
    const bool used_by_shadow_output = [](const pir::Value& value) {
      bool used_by_shadow_output = false;
      for (auto it = value.use_begin(); it != value.use_end(); ++it) {
        if (it->owner()->isa<pir::ShadowOutputOp>()) {
          used_by_shadow_output = true;
          break;
        }
      }
      return used_by_shadow_output;
    }(value);
    value.ReplaceUsesWithIf(new_value, [](pir::OpOperand operand) {
      return !operand.owner()->isa<pir::ShadowOutputOp>();
    });
    if (used_by_shadow_output) {
      auto copied_value = CreateAssignOp(new_value, new_op, op->GetParent());
      value.ReplaceUsesWithIf(copied_value, [](pir::OpOperand operand) {
        return operand.owner()->isa<pir::ShadowOutputOp>();
      });
    }
  }
  op->Erase();
}

class CommonSubexpressionEliminationPass : public pir::Pass {
 public:
  CommonSubexpressionEliminationPass()
      : pir::Pass("common_subexpression_elimination_pass", 1) {}

  void Run(pir::Operation* op) override {
    VLOG(6) << "apply common_subexpression_elimination_pass";
    int64_t num_erasers{0};
    CSEAnalyzer cse_analyzer;
    ExpressionTable root_expression_table;
    cse_analyzer.SimplifyBlock(op->GetParentProgram()->block(),
                               &root_expression_table);
    size_t op_count_before_cse = op->GetParentProgram()->block()->num_ops();
    std::unordered_map<std::string, size_t> op_stats;
    VLOG(7) << "Found " << cse_analyzer.to_erase_ops().size()
            << " common subexpression";

    int32_t cse_count = 0;
    for (auto [op, existing_op] : cse_analyzer.to_erase_ops()) {
      if (FLAGS_cse_max_count != -1 && cse_count >= FLAGS_cse_max_count) {
        break;
      }
      ReplaceOpWith(op, existing_op);
      num_erasers++;
      cse_count++;
      op_stats[existing_op->name()]++;
    }
    size_t op_count_after_cse = op->GetParentProgram()->block()->num_ops();
    VLOG(7) << "op count before cse: " << op_count_before_cse
            << ", op count after cse: " << op_count_after_cse << ", erased "
            << num_erasers << " (" << num_erasers * 100.0 / op_count_before_cse
            << "%)";
    VLOG(7) << "Erased op statistics:";
    for (auto& [op_name, count] : op_stats) {
      VLOG(7) << "    " << op_name << ": " << count;
    }
    AddStatistics(num_erasers);
  }

  bool CanApplyOn(pir::Operation* op) const override {
    return op->isa<pir::ModuleOp>() && op->num_regions() > 0;
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateCommonSubexpressionEliminationPass() {
  return std::make_unique<CommonSubexpressionEliminationPass>();
}

}  // namespace pir

REGISTER_IR_PASS(common_subexpression_elimination_pass,
                 CommonSubexpressionEliminationPass);
