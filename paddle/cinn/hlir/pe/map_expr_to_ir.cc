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

#include "paddle/cinn/hlir/pe/map_expr_to_ir.h"

#include <unordered_map>
#include <vector>

#include "paddle/cinn/adt/dim_expr.h"
#include "paddle/cinn/adt/equation_value_match_trait.h"
#include "paddle/cinn/adt/inline_translator.h"
#include "paddle/cinn/adt/map_expr.h"
#include "paddle/cinn/adt/map_expr_ctx.h"
#include "paddle/cinn/adt/match.h"
#include "paddle/cinn/adt/no_inline_translator.h"
#include "paddle/cinn/adt/print.h"
#include "paddle/cinn/common/target.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/runtime/flags.h"
#include "paddle/common/enforce.h"
#include "paddle/pir/include/dialect/shape/ir/shape_op.h"
PD_DECLARE_bool(cinn_enable_map_expr_inline);

namespace cinn::adt {

namespace {

using IteratorInt = std::int32_t;
using Node2LoweredFuncs =
    std::unordered_map<::pir::Operation*, std::vector<ir::LoweredFunc>>;

using TensorIteratorExpr4TensorT =
    std::function<adt::List<adt::TensorIteratorExpr>(const adt::Tensor&)>;
using IterExprs4TensorT =
    std::function<std::vector<ir::Expr>(const adt::Tensor&)>;
using LoopDescriptor4LoopIteratorT =
    std::function<adt::LoopDescriptor(const adt::Iterator&)>;

class MapExprToIrTranslator {
 public:
  explicit MapExprToIrTranslator(const MapExpr& map_expr,
                                 const Node2LoweredFuncs& node2lowered_funcs,
                                 const cinn::common::Target& target)
      : map_expr_(map_expr),
        node2lowered_funcs_(&node2lowered_funcs),
        target_(target) {
    const auto& [anchored_map_stmts, _0, _1] = map_expr.tuple();
    PADDLE_ENFORCE_EQ(anchored_map_stmts->size(),
                      1,
                      ::common::errors::InvalidArgument(
                          "Only support one AnchoredMapStmt now!"));
    TensorIteratorExpr4Tensor = std::get<4>(anchored_map_stmts->at(0).tuple());
    LoopDescriptor4LoopIterator =
        std::get<5>(anchored_map_stmts->at(0).tuple());
  }

  ir::Expr Translate() const {
    VLOG(1) << "Translate MapExpr: ";
    VLOG(1) << ToTxtString(map_expr_, "");
    return ir::Block::Make({Translate(map_expr_)});
  }

 private:
  ir::Expr GetStoreExprForOp(const ::pir::Operation* op) const {
    const auto& iter =
        node2lowered_funcs_->find(const_cast<::pir::Operation*>(op));
    PADDLE_ENFORCE_NE(iter,
                      node2lowered_funcs_->end(),
                      ::common::errors::InvalidArgument(
                          "Iterator reached the end, which indicates the node "
                          "was not found in node2lowered_funcs_."));

    const auto& lowered_funcs = iter->second;
    PADDLE_ENFORCE_EQ(
        lowered_funcs.size(),
        1,
        ::common::errors::InvalidArgument("Only support one LoweredFunc now!"));
    std::optional<ir::Expr> ret{std::nullopt};
    VisitEachStoreExpr(lowered_funcs.at(0)->body, [&](const ir::Expr& expr) {
      PADDLE_ENFORCE_EQ(
          ret.has_value(),
          false,
          ::common::errors::InvalidArgument(
              "ret already has a value. Expected it to be uninitialized."));
      ret = expr;
    });
    PADDLE_ENFORCE_EQ(
        ret.has_value(),
        true,
        ::common::errors::InvalidArgument(
            "ret does not have a value after VisitEachStoreExpr execution."));

    return ret.value();
  }

  ir::Expr GetStoreExprForOp(
      const tReduceInit<const ::pir::Operation*>& op) const {
    const auto& iter =
        node2lowered_funcs_->find(const_cast<::pir::Operation*>(op.value()));
    PADDLE_ENFORCE_NE(iter,
                      node2lowered_funcs_->end(),
                      ::common::errors::NotFound(
                          "The node was not found in node2lowered_funcs_."));

    const auto& lowered_funcs = iter->second;
    PADDLE_ENFORCE_EQ(
        lowered_funcs.size(),
        1,
        ::common::errors::InvalidArgument("Only support one LoweredFunc now!"));
    std::vector<ir::Expr> stores{};
    VisitEachStoreExpr(lowered_funcs.at(0)->body, [&](const ir::Expr& expr) {
      stores.emplace_back(expr);
    });
    PADDLE_ENFORCE_EQ(stores.size(),
                      2,
                      ::common::errors::InvalidArgument(
                          "Only support two stores for tReduceInit now!"));
    return stores.at(0);
  }

  ir::Expr GetStoreExprForOp(
      const tReduceAcc<const ::pir::Operation*>& op) const {
    const auto& iter =
        node2lowered_funcs_->find(const_cast<::pir::Operation*>(op.value()));
    PADDLE_ENFORCE_NE(iter,
                      node2lowered_funcs_->end(),
                      ::common::errors::NotFound(
                          "The node was not found in node2lowered_funcs_."));

    const auto& lowered_funcs = iter->second;
    PADDLE_ENFORCE_EQ(
        lowered_funcs.size(),
        1,
        ::common::errors::InvalidArgument("Only support one LoweredFunc now!"));
    std::vector<ir::Expr> stores{};
    VisitEachStoreExpr(lowered_funcs.at(0)->body, [&](const ir::Expr& expr) {
      stores.emplace_back(expr);
    });
    PADDLE_ENFORCE_EQ(stores.size(),
                      2,
                      ::common::errors::InvalidArgument(
                          "Only support two stores for tReduceAcc now!"));
    return stores.at(1);
  }

  std::optional<ir::Expr> GetStoreExprImpl(
      const OpCall<OpExpr>& op_expr) const {
    const auto& [op, _] = op_expr.tuple();
    return std::visit([&](const auto& impl) { return GetStoreExprForOp(impl); },
                      op.variant());
  }

  std::optional<ir::Expr> GetStoreExprImpl(const Load<Tensor>& load) const {
    return std::nullopt;
  }

  // using OpExpr = Tree<OpCall, Load<Tensor>>;
  std::optional<ir::Expr> GetStoreExpr(const OpExpr& op_expr) const {
    return std::visit([&](const auto& impl) { return GetStoreExprImpl(impl); },
                      op_expr.variant());
  }

  template <typename DoEachT>
  void VisitEachStoreExpr(const ir::Expr& expr, const DoEachT& DoEach) const {
    switch (expr.node_type()) {
      case ir::IrNodeTy::Block:
        for (const auto& stmt : expr.As<ir::Block>()->stmts) {
          VisitEachStoreExpr(stmt, DoEach);
        }
        break;
      case ir::IrNodeTy::ScheduleBlockRealize:
        VisitEachStoreExpr(expr.As<ir::ScheduleBlockRealize>()->schedule_block,
                           DoEach);
        break;
      case ir::IrNodeTy::ScheduleBlock:
        VisitEachStoreExpr(expr.As<ir::ScheduleBlock>()->body, DoEach);
        break;
      case ir::IrNodeTy::For:
        VisitEachStoreExpr(expr.As<ir::For>()->body, DoEach);
        break;
      case ir::IrNodeTy::Store:
        DoEach(expr);
        break;
      default:
        std::stringstream ss;
        ss << "Visit node_type = " << expr.node_type() << ", not supported!";
        PADDLE_THROW(::common::errors::InvalidArgument(ss.str()));
        break;
    }
  }

  template <typename DoEachT>
  void VisitEachStmt(const List<Stmt>& stmts, const DoEachT& DoEach) const {
    for (const auto& stmt : *stmts) {
      DoEach(stmt);
    }
  }

  ir::Expr Translate(const MapExpr& map_expr) const {
    const auto& [anchored_map_stmts, _0, _1] = map_expr.tuple();
    PADDLE_ENFORCE_EQ(anchored_map_stmts->size(),
                      1,
                      ::common::errors::InvalidArgument(
                          "Only support one AnchoredMapStmt now!"));
    return Translate(anchored_map_stmts->at(0));
  }

  ir::Expr Translate(const AnchoredMapStmt& anchored_map_stmt) const {
    const MapStmt<Stmt>& map_stmt = std::get<0>(anchored_map_stmt.tuple());
    ir::Expr ret = Translate(map_stmt);
    ret = ir::ScheduleBlock::Make({}, {}, {}, "root", ret);
    ret = ir::ScheduleBlockRealize::Make({}, ret);
    return ret;
  }

  using InternalLeafStmt = Store<Tensor, OpCall<Load<Tensor>>>;
  using InternalStmt = Tree<MapStmt, InternalLeafStmt>;

  InternalStmt ConvertToInternalStmtImpl(const OpStmt& op_stmt) const {
    const auto& [op, inputs, outputs] = op_stmt.tuple();
    PADDLE_ENFORCE_EQ(
        outputs.value()->size(),
        1,
        ::common::errors::InvalidArgument("Only support one output now!"));
    List<Load<Tensor>> loads{};
    for (const auto& in : *inputs.value()) {
      loads->emplace_back(Load<Tensor>{in});
    }
    OpCall<Load<Tensor>> op_call{op, loads};
    return InternalLeafStmt{outputs.value()->at(0), op_call};
  }

  InternalStmt ConvertToInternalStmtImpl(const MapStmt<Stmt>& map_stmt) const {
    const auto& [iterators, stmts] = map_stmt.tuple();
    List<InternalStmt> children{};
    for (const auto& stmt : *stmts) {
      children->emplace_back(ConvertToInternalStmt(stmt));
    }
    return MapStmt<InternalStmt>{iterators, children};
  }

  InternalStmt ConvertToInternalStmt(const Stmt& stmt) const {
    return std::visit(
        [&](const auto& impl) { return ConvertToInternalStmtImpl(impl); },
        stmt.variant());
  }

  InlineStmt ConvertToInlineStmt(const InternalStmt& internal_stmt) const {
    if (FLAGS_cinn_enable_map_expr_inline) {
      return InlineTranslator<MapStmt, OpCall, Tensor>::Call(internal_stmt);
    } else {
      return NoInlineTranslator<MapStmt, OpCall, Tensor>::Call(internal_stmt);
    }
    PADDLE_THROW(::common::errors::Fatal("Dead code"));
  }

  std::optional<ir::Expr> TranslateOpExprImpl(
      const Load<Tensor>& load,
      const std::optional<Tensor>& opt_output_tensor,
      const IterExprs4TensorT& IterExprs4Tensor) const {
    return std::nullopt;
  }

  std::vector<ir::Expr> TranslateTensorIndexImpl(
      const OpCall<OpExpr>& op_call,
      const IterExprs4TensorT& IterExprs4Tensor) const {
    PADDLE_THROW(::common::errors::InvalidArgument(
        "Dead code, no TensorIndexExpr for OpCall"));
  }

  std::vector<ir::Expr> TranslateTensorIndexImpl(
      const Load<Tensor>& op_call,
      const IterExprs4TensorT& IterExprs4Tensor) const {
    const auto& [tensor] = op_call.tuple();
    return IterExprs4Tensor(tensor);
  }

  // using OpExpr = Tree<OpCall, Load<Tensor>>;
  std::vector<ir::Expr> TranslateTensorIndex(
      const OpExpr& op_expr, const IterExprs4TensorT& IterExprs4Tensor) const {
    return std::visit(
        [&](const auto& impl) {
          return TranslateTensorIndexImpl(impl, IterExprs4Tensor);
        },
        op_expr.variant());
  }

  std::optional<ir::Expr> MakeLoadExpr(
      const ir::Expr& input_expr,
      const List<OpExpr>& op_expr_children,
      const IterExprs4TensorT& IterExprs4Tensor) const {
    ir::Expr store_rvalue = ir::ir_utils::IRCopy(input_expr);
    PADDLE_ENFORCE_EQ(
        store_rvalue->operands.size(),
        0,
        ::common::errors::InvalidArgument("Only support one operand!"));
    PADDLE_ENFORCE_EQ(
        op_expr_children->size(),
        1,
        ::common::errors::InvalidArgument("Only support one child!"));
    store_rvalue.As<ir::Load>()->indices =
        TranslateTensorIndex(op_expr_children->at(0), IterExprs4Tensor);
    return store_rvalue;
  }

  std::optional<ir::Expr> MakeCallExpr(
      const ir::Expr& input_expr,
      const List<OpExpr>& op_expr_children,
      const IterExprs4TensorT& IterExprs4Tensor) const {
    ir::Expr store_rvalue = ir::ir_utils::IRCopy(input_expr);
    PADDLE_ENFORCE_EQ(
        store_rvalue->operands.size(),
        0,
        ::common::errors::InvalidArgument("Only support one operand!"));
    PADDLE_ENFORCE_EQ(
        op_expr_children->empty(),
        false,
        ::common::errors::InvalidArgument("The op_expr_children list is empty. "
                                          "Expected it to contain elements."));

    PADDLE_ENFORCE_EQ(
        (store_rvalue.As<ir::Call>()->read_args.size()),
        (op_expr_children->size()),
        ::common::errors::InvalidArgument("Mismatched read_args size"));
    for (int i = 0; i < op_expr_children->size(); ++i) {
      const auto& opt_operant = TranslateOpExpr(
          op_expr_children->at(i), std::nullopt, IterExprs4Tensor);
      // Note: Only handles read_args here, consider handles other variables in
      // ir::Call
      if (opt_operant.has_value()) {
        store_rvalue.As<ir::Call>()->read_args.at(i) = opt_operant.value();
      } else {
        store_rvalue.As<ir::Call>()->read_args.at(i).As<ir::Load>()->indices =
            TranslateTensorIndex(op_expr_children->at(i), IterExprs4Tensor);
      }
    }
    return store_rvalue;
  }

  std::optional<ir::Expr> MakeGeneralExpr(
      const ir::Expr& input_expr,
      const List<OpExpr>& op_expr_children,
      const IterExprs4TensorT& IterExprs4Tensor) const {
    ir::Expr store_rvalue = ir::ir_utils::IRCopy(input_expr);
    PADDLE_ENFORCE_EQ(
        store_rvalue->operands.size(),
        op_expr_children->size(),
        ::common::errors::InvalidArgument("Mismatched operands size"));
    for (int i = 0; i < op_expr_children->size(); ++i) {
      const auto& opt_operant = TranslateOpExpr(
          op_expr_children->at(i), std::nullopt, IterExprs4Tensor);
      if (opt_operant.has_value()) {
        store_rvalue->operands.at(i) = opt_operant.value();
      } else {
        store_rvalue->operands.at(i).As<ir::Load>()->indices =
            TranslateTensorIndex(op_expr_children->at(i), IterExprs4Tensor);
      }
    }
    return store_rvalue;
  }

  std::optional<ir::Expr> MakeScaleRvalueExpr(
      const ir::Expr& input_expr,
      const List<OpExpr>& op_expr_children,
      const IterExprs4TensorT& IterExprs4Tensor) const {
    // Scale Expr example: ((float32(0.00130208337f) * var_4[i0_4, i1_4, i2_2])
    // + float32(0.00000000f))
    ir::Expr store_rvalue = ir::ir_utils::IRCopy(input_expr);
    PADDLE_ENFORCE_EQ(
        op_expr_children->size(),
        1,
        ::common::errors::InvalidArgument("Only support one child!"));
    PADDLE_ENFORCE_EQ(
        store_rvalue->operands.size(),
        2,
        ::common::errors::InvalidArgument("Mismatched operands size"));
    const auto& opt_operant = TranslateOpExpr(
        op_expr_children->at(0), std::nullopt, IterExprs4Tensor);

    ir::Mul* mul_operant = store_rvalue->operands.at(0).As<ir::Mul>();
    if (opt_operant.has_value()) {
      mul_operant->operands().at(1) = opt_operant.value();
    } else {
      mul_operant->operands().at(1).As<ir::Load>()->indices =
          TranslateTensorIndex(op_expr_children->at(0), IterExprs4Tensor);
    }
    return store_rvalue;
  }

  typedef std::optional<ir::Expr> (
      MapExprToIrTranslator::*MakeStoreRvalueExprT)(
      const ir::Expr&, const List<OpExpr>&, const IterExprs4TensorT&) const;

  static std::unordered_map<std::string, MakeStoreRvalueExprT>
  MakeGetterMakeStoreRvalueExpr4Op() {
    std::unordered_map<std::string, MakeStoreRvalueExprT> ret{
        {"elementwise_mul", &MapExprToIrTranslator::MakeGeneralExpr},
        {"subtract", &MapExprToIrTranslator::MakeGeneralExpr},
        {"broadcast_to", &MapExprToIrTranslator::MakeGeneralExpr},

        {"exp", &MapExprToIrTranslator::MakeCallExpr},
        {"rsqrt", &MapExprToIrTranslator::MakeCallExpr},

        {"scale", &MapExprToIrTranslator::MakeScaleRvalueExpr},
    };
    return ret;
  }

  MakeStoreRvalueExprT GetMakeStoreRvalueExpr4Op(
      const std::string& op_name, const ir::Expr& store_expr) const {
    static std::unordered_map<std::string, MakeStoreRvalueExprT>
        MakeStoreRvalueExpr4Op(MakeGetterMakeStoreRvalueExpr4Op());
    const auto& iter = MakeStoreRvalueExpr4Op.find(op_name);
    PADDLE_ENFORCE_NE(iter,
                      MakeStoreRvalueExpr4Op.end(),
                      ::common::errors::Unimplemented(
                          "Operation %s is not supported yet! store_expr: %s",
                          op_name,
                          store_expr));

    return iter->second;
  }

  std::optional<ir::Expr> TranslateOpCallImpl(
      const ::pir::Operation* op,
      const OpCall<OpExpr>& op_expr,
      const std::optional<Tensor>& opt_output_tensor,
      const IterExprs4TensorT& IterExprs4Tensor) const {
    const auto& [_, op_expr_children] = op_expr.tuple();
    if (op_expr_children->empty()) {
      return GetStoreExpr(op_expr).value().As<ir::Store>()->value;
    }
    std::optional<ir::Expr> store_expr = GetStoreExpr(op_expr);
    PADDLE_ENFORCE_EQ(store_expr.has_value(),
                      true,
                      ::common::errors::InvalidArgument(
                          "Expected store_expr to have a value, but it is "
                          "currently uninitialized or empty. Ensure that the "
                          "store_expr is correctly set before this check."));

    ir::Expr store_rvalue = store_expr.value().As<ir::Store>()->value;
    if (store_rvalue.As<ir::Load>()) {
      return MakeLoadExpr(store_rvalue, op_expr_children, IterExprs4Tensor);
    } else {
      const auto& op_name = hlir::framework::pir::CompatibleInfo::OpName(*op);
      const auto& make_store_rvalue_expr =
          GetMakeStoreRvalueExpr4Op(op_name, store_expr.value());
      return (this->*make_store_rvalue_expr)(
          store_rvalue, op_expr_children, IterExprs4Tensor);
    }
    PADDLE_THROW(::common::errors::Fatal("Dead code"));
  }

  std::optional<ir::Expr> TranslateOpCallImpl(
      const tReduceInit<const ::pir::Operation*>&,
      const OpCall<OpExpr>& op_expr,
      const std::optional<Tensor>& opt_output_tensor,
      const IterExprs4TensorT& IterExprs4Tensor) const {
    const auto& [_, op_expr_children] = op_expr.tuple();
    std::optional<ir::Expr> store_expr = GetStoreExpr(op_expr);
    PADDLE_ENFORCE_EQ(store_expr.has_value(),
                      true,
                      ::common::errors::InvalidArgument(
                          "Expected store_expr to have a value, but it is "
                          "currently uninitialized or empty. Ensure that the "
                          "store_expr is correctly set before this check."));

    ir::Expr store_rvalue = store_expr.value().As<ir::Store>()->value;
    VLOG(1) << "tReduceInit store_rvalue:\n" << store_rvalue;

    PADDLE_ENFORCE_EQ(
        store_rvalue->operands.size(),
        0,
        ::common::errors::InvalidArgument("Mismatched operands size"));
    PADDLE_ENFORCE_EQ(
        op_expr_children->size(),
        0,
        ::common::errors::InvalidArgument("Mismatched children size"));
    return store_rvalue;
  }

  std::optional<ir::Expr> TranslateOpCallImpl(
      const tReduceAcc<const ::pir::Operation*>&,
      const OpCall<OpExpr>& op_expr,
      const std::optional<Tensor>& opt_output_tensor,
      const IterExprs4TensorT& IterExprs4Tensor) const {
    const auto& [_, op_expr_children] = op_expr.tuple();
    std::optional<ir::Expr> store_expr = GetStoreExpr(op_expr);
    PADDLE_ENFORCE_EQ(store_expr.has_value(),
                      true,
                      ::common::errors::InvalidArgument(
                          "Expected store_expr to have a value, but it is "
                          "currently uninitialized or empty. Ensure that the "
                          "store_expr is correctly set before this check."));

    ir::Expr store_rvalue = store_expr.value().As<ir::Store>()->value;
    VLOG(1) << "tReduceAcc store_rvalue:\n" << store_rvalue;

    PADDLE_ENFORCE_EQ(
        store_rvalue->operands.size(),
        2,
        ::common::errors::InvalidArgument("Mismatched operands size"));
    PADDLE_ENFORCE_EQ(
        op_expr_children->size(),
        1,
        ::common::errors::InvalidArgument("Mismatched children size"));
    PADDLE_ENFORCE_EQ(
        opt_output_tensor.has_value(),
        true,
        ::common::errors::InvalidArgument(
            "Expected opt_output_tensor to have a value, but it is currently "
            "uninitialized or empty. Ensure that the opt_output_tensor is "
            "correctly set before this check."));

    store_rvalue->operands.at(0).As<ir::Load>()->indices =
        IterExprs4Tensor(opt_output_tensor.value());
    {
      const auto& opt_operant = TranslateOpExpr(
          op_expr_children->at(0), std::nullopt, IterExprs4Tensor);
      if (opt_operant.has_value()) {
        store_rvalue->operands.at(1) = opt_operant.value();
      } else {
        store_rvalue->operands.at(1).As<ir::Load>()->indices =
            TranslateTensorIndex(op_expr_children->at(0), IterExprs4Tensor);
      }
    }
    return store_rvalue;
  }

  std::optional<ir::Expr> TranslateOpExprImpl(
      const OpCall<OpExpr>& op_expr,
      const std::optional<Tensor>& opt_output_tensor,
      const IterExprs4TensorT& IterExprs4Tensor) const {
    const auto& [op, op_expr_children] = op_expr.tuple();
    return std::visit(
        [&](const auto& impl) {
          return TranslateOpCallImpl(
              impl, op_expr, opt_output_tensor, IterExprs4Tensor);
        },
        op.variant());
  }

  // using OpExpr = Tree<OpCall, Load<Tensor>>;
  std::optional<ir::Expr> TranslateOpExpr(
      const OpExpr& op_expr,
      const std::optional<Tensor>& opt_output_tensor,
      const IterExprs4TensorT& IterExprs4Tensor) const {
    return std::visit(
        [&](const auto& impl) {
          return TranslateOpExprImpl(impl, opt_output_tensor, IterExprs4Tensor);
        },
        op_expr.variant());
  }

  ir::Expr TranslateImpl(const OpExprStmt& op_expr_stmt) const {
    return Translate(op_expr_stmt);
  }

  template <typename DoEachT /*void(&)(const Value&)*/>
  void VisitEachIteratorValue(const Tensor& tensor,
                              const DoEachT& DoEach) const {
    const List<Value>& iterator_values = TensorIteratorExpr4Tensor(tensor);
    for (const auto& iterator_value : *iterator_values) {
      DoEach(iterator_value);
    }
  }

  template <typename DoEachT /*void(&)(const Value&)*/>
  void VisitEachIteratorValueImpl(const OpCall<OpExpr>& op_call,
                                  const DoEachT& DoEach) const {
    const auto& [_, children] = op_call.tuple();
    for (const auto& child : *children) {
      VisitEachIteratorValue(child, DoEach);
    }
  }

  template <typename DoEachT /*void(&)(const Value&)*/>
  void VisitEachIteratorValueImpl(const Load<Tensor>& load,
                                  const DoEachT& DoEach) const {
    const auto& [tensor] = load.tuple();
    VisitEachIteratorValue(tensor, DoEach);
  }

  template <typename DoEachT /*void(&)(const Value&)*/>
  void VisitEachIteratorValue(const OpExpr& op_expr,
                              const DoEachT& DoEach) const {
    return std::visit(
        [&](const auto& impl) {
          return VisitEachIteratorValueImpl(impl, DoEach);
        },
        op_expr.variant());
  }

  template <typename DoEachT /*void(&)(const Value&)*/>
  void VisitEachIteratorValue(const OpExprStmt& op_expr_stmt,
                              const DoEachT& DoEach) const {
    const auto& [tensor, op_expr] = op_expr_stmt.tuple();
    VisitEachIteratorValue(tensor, DoEach);
    VisitEachIteratorValue(op_expr, DoEach);
  }

  IterExprs4TensorT MakeGetterIterExprs4Tensor(
      const OpExprStmt& op_expr_stmt,
      std::vector<std::pair<ir::Var, ir::Expr>>* binding_var2value) const {
    std::unordered_map<Value, std::pair<ir::Var, ir::Expr>> value2var_expr{};
    VisitEachIteratorValue(op_expr_stmt, [&](const Value& value) {
      if (value2var_expr.count(value) == 0) {
        ir::Var var{std::string("m_expr_i_") +
                    std::to_string(UniqueId::New().unique_id())};
        ir::Expr expr = TranslateTensorIterator(value);
        PADDLE_ENFORCE_EQ(
            value2var_expr.emplace(value, std::make_pair(var, expr)).second,
            true,
            ::common::errors::InvalidArgument(
                "Failed to insert the value-var-expr pair into value2var_expr. "
                "This indicates that the value already exists in the map. "
                "Ensure that each value is unique before inserting."));

      } else {
        // Do nothing
      }
    });
    for (const auto& [_, pair] : value2var_expr) {
      binding_var2value->push_back(pair);
    }
    return [value2var_expr, this](const Tensor& tensor) {
      const List<Value>& iterator_values = TensorIteratorExpr4Tensor(tensor);
      std::vector<ir::Expr> ret{};
      ret.reserve(iterator_values->size());
      for (const auto& iterator_value : *iterator_values) {
        const auto& it = value2var_expr.find(iterator_value);
        PADDLE_ENFORCE_NE(
            it,
            value2var_expr.end(),
            ::common::errors::NotFound(
                "The iterator reached the end of value2var_expr, indicating "
                "that the value was not found in the map. Ensure that the "
                "value exists in the map before attempting to access it."));

        ret.emplace_back(it->second.first);
      }
      return ret;
    };
  }

  std::vector<ir::Var> GetVectorOfPairFirst(
      const std::vector<std::pair<ir::Var, ir::Expr>>& pairs) const {
    std::vector<ir::Var> ret{};
    ret.reserve(pairs.size());
    for (const auto& pair : pairs) {
      ret.emplace_back(pair.first);
    }
    return ret;
  }

  std::vector<ir::Expr> GetVectorOfPairSecond(
      const std::vector<std::pair<ir::Var, ir::Expr>>& pairs) const {
    std::vector<ir::Expr> ret{};
    ret.reserve(pairs.size());
    for (const auto& pair : pairs) {
      ret.emplace_back(pair.second);
    }
    return ret;
  }

  // using OpExprStmt = Store<Tensor, OpExpr>;
  ir::Expr Translate(const OpExprStmt& op_expr_stmt) const {
    const auto& [output_tensor, op_expr] = op_expr_stmt.tuple();
    std::optional<ir::Expr> store_expr = GetStoreExpr(op_expr);
    PADDLE_ENFORCE_EQ(store_expr.has_value(),
                      true,
                      ::common::errors::InvalidArgument(
                          "Expected store_expr to have a value, but it is "
                          "currently uninitialized or empty. Ensure that the "
                          "store_expr is correctly set before this check."));

    std::optional<Tensor> opt_output_tensor = output_tensor;

    std::vector<std::pair<ir::Var, ir::Expr>> binding_var2value{};
    const auto& IterExprs4Tensor =
        MakeGetterIterExprs4Tensor(op_expr_stmt, &binding_var2value);

    const auto& opt_rvalue =
        TranslateOpExpr(op_expr, opt_output_tensor, IterExprs4Tensor);
    PADDLE_ENFORCE_EQ(opt_rvalue.has_value(),
                      true,
                      ::common::errors::InvalidArgument(
                          "Expected opt_rvalue to have a value, but it is "
                          "currently uninitialized or empty. Ensure that "
                          "opt_rvalue is correctly set before this check."));

    const auto& output_expr =
        ir::Store::Make(store_expr.value().As<ir::Store>()->tensor,
                        opt_rvalue.value(),
                        IterExprs4Tensor(output_tensor));

    ir::Expr ret = ir::ScheduleBlock::Make(
        GetVectorOfPairFirst(binding_var2value),
        {},
        {},
        output_expr.As<ir::Store>()->tensor.as_tensor()->name,
        output_expr);
    ret = ir::ScheduleBlockRealize::Make(
        GetVectorOfPairSecond(binding_var2value), ret);
    return ret;
  }

  ir::Expr Translate(const List<InlineStmt>& stmts) const {
    std::vector<ir::Expr> exprs;
    for (const auto& stmt : *stmts) {
      exprs.emplace_back(Translate(stmt));
    }
    return ir::Block::Make(exprs);
  }

  ir::Expr TranslateImpl(const MapStmt<InlineStmt>& map_stmt) const {
    const auto& [iterators, stmts] = map_stmt.tuple();
    ir::Expr ret = Translate(stmts);
    PADDLE_ENFORCE_GT(iterators->size(),
                      0,
                      ::common::errors::InvalidArgument(
                          "iterators->size() should be greater than 0"));
    for (int i = iterators->size() - 1; i >= 0; --i) {
      const auto& iterator = iterators->at(i);
      const auto& ld = LoopDescriptor4LoopIterator(iterator);
      ir::Var var{"v_" + std::to_string(iterator.value().unique_id())};
      ir::Expr min{IteratorInt(0)};
      ir::Expr extent = GetLoopSize(ld);
      const auto& [for_type, vectorize_info, bind_info] = GetForTypeAndInfo(ld);
      ir::DeviceAPI device_api = GetDeviceApi();
      ret = ir::For::Make(var,
                          min,
                          extent,
                          for_type,
                          device_api,
                          ret,
                          vectorize_info,
                          bind_info);
    }
    return ret;
  }

  ir::Expr Translate(const InlineStmt& inline_stmt) const {
    return std::visit([&](const auto& impl) { return TranslateImpl(impl); },
                      inline_stmt.variant());
  }

  ir::Expr Translate(const MapStmt<Stmt>& map_stmt) const {
    Stmt stmt = map_stmt;
    InternalStmt internal_stmt = ConvertToInternalStmt(stmt);
    InlineStmt inline_stmt = ConvertToInlineStmt(internal_stmt);
    return Translate(inline_stmt);
  }

  ir::DeviceAPI GetDeviceApi() const { return ir::DeviceAPI::Host; }

  ir::Expr GetLoopSize(const LoopDescriptor& ld) const {
    const auto& [_, loop_size] = ld.tuple();
    return TranslateDimExpr(loop_size);
  }

  std::tuple<ir::ForType, ir::VectorizeInfo, ir::BindInfo>
  GetForTypeAndInfoImpl(const S0x& loop_type, const LoopDescriptor& ld) const {
    ir::ForType for_type = ir::ForType::GPUBlock;
    ir::BindInfo bind_info{for_type, 0, ir::DeviceAPI::GPU};
    return std::make_tuple(for_type, ir::VectorizeInfo(), bind_info);
  }

  std::tuple<ir::ForType, ir::VectorizeInfo, ir::BindInfo>
  GetForTypeAndInfoImpl(const S0y& loop_type, const LoopDescriptor& ld) const {
    ir::ForType for_type = ir::ForType::GPUBlock;
    ir::BindInfo bind_info{for_type, 1, ir::DeviceAPI::GPU};
    return std::make_tuple(for_type, ir::VectorizeInfo(), bind_info);
  }

  std::tuple<ir::ForType, ir::VectorizeInfo, ir::BindInfo>
  GetForTypeAndInfoImpl(const S0z& loop_type, const LoopDescriptor& ld) const {
    ir::ForType for_type = ir::ForType::GPUBlock;
    ir::BindInfo bind_info{for_type, 2, ir::DeviceAPI::GPU};
    return std::make_tuple(for_type, ir::VectorizeInfo(), bind_info);
  }

  std::tuple<ir::ForType, ir::VectorizeInfo, ir::BindInfo>
  GetForTypeAndInfoImpl(const S1x& loop_type, const LoopDescriptor& ld) const {
    ir::ForType for_type = ir::ForType::GPUThread;
    ir::BindInfo bind_info{for_type, 0, ir::DeviceAPI::GPU};
    return std::make_tuple(for_type, ir::VectorizeInfo(), bind_info);
  }

  std::tuple<ir::ForType, ir::VectorizeInfo, ir::BindInfo>
  GetForTypeAndInfoImpl(const S1y& loop_type, const LoopDescriptor& ld) const {
    ir::ForType for_type = ir::ForType::GPUThread;
    ir::BindInfo bind_info{for_type, 1, ir::DeviceAPI::GPU};
    return std::make_tuple(for_type, ir::VectorizeInfo(), bind_info);
  }

  std::tuple<ir::ForType, ir::VectorizeInfo, ir::BindInfo>
  GetForTypeAndInfoImpl(const S1z& loop_type, const LoopDescriptor& ld) const {
    ir::ForType for_type = ir::ForType::GPUThread;
    ir::BindInfo bind_info{for_type, 2, ir::DeviceAPI::GPU};
    return std::make_tuple(for_type, ir::VectorizeInfo(), bind_info);
  }

  std::tuple<ir::ForType, ir::VectorizeInfo, ir::BindInfo>
  GetForTypeAndInfoImpl(const Temporal& loop_type,
                        const LoopDescriptor& ld) const {
    return std::make_tuple(
        ir::ForType::Serial, ir::VectorizeInfo(), ir::BindInfo());
  }

  std::tuple<ir::ForType, ir::VectorizeInfo, ir::BindInfo>
  GetForTypeAndInfoImpl(const Vectorize& loop_type,
                        const LoopDescriptor& ld) const {
    PADDLE_THROW(
        ::common::errors::InvalidArgument("Vectorize not supported yet"));
  }

  std::tuple<ir::ForType, ir::VectorizeInfo, ir::BindInfo>
  GetForTypeAndInfoImpl(const Unroll& loop_type,
                        const LoopDescriptor& ld) const {
    PADDLE_THROW(::common::errors::InvalidArgument("Unroll not supported yet"));
  }

  std::tuple<ir::ForType, ir::VectorizeInfo, ir::BindInfo> GetForTypeAndInfo(
      const LoopDescriptor& ld) const {
    const auto& [loop_type, _] = ld.tuple();
    return std::visit(
        [&](const auto& impl) { return GetForTypeAndInfoImpl(impl, ld); },
        loop_type.variant());
  }

  ir::Expr Accumulate(const std::vector<ir::Expr>& ir_exprs) const {
    if (ir_exprs.size() == 0) {
      PADDLE_THROW(::common::errors::Fatal("Dead code"));
    } else if (ir_exprs.size() == 1) {
      return ir_exprs.at(0);
    } else {
      ir::Expr ret = ir_exprs.at(0);
      for (int i = 1; i < ir_exprs.size(); ++i) {
        ret = ir::Add::Make(ret, ir_exprs.at(i));
      }
      return ret;
    }
    PADDLE_THROW(::common::errors::Fatal("Dead code"));
  }

  ir::Expr Multiply(const std::vector<ir::Expr>& ir_exprs) const {
    if (ir_exprs.size() == 0) {
      PADDLE_THROW(::common::errors::Fatal("Dead code"));
    } else if (ir_exprs.size() == 1) {
      return ir_exprs.at(0);
    } else {
      ir::Expr ret = ir_exprs.at(0);
      for (int i = 1; i < ir_exprs.size(); ++i) {
        ret = ir::Mul::Make(ret, ir_exprs.at(i));
      }
      return ret;
    }
    PADDLE_THROW(::common::errors::Fatal("Dead code"));
  }

  ir::Expr Divide(const std::vector<ir::Expr>& ir_exprs) const {
    if (ir_exprs.size() < 2) {
      PADDLE_THROW(::common::errors::Fatal("Dead code"));
    } else {
      return ir::Div::Make(ir_exprs.at(0), ir_exprs.at(1));
    }
    PADDLE_THROW(::common::errors::Fatal("Dead code"));
  }

  ir::Expr GetStride(const List<DimExpr>& dims, int start) const {
    PADDLE_ENFORCE_GE(start,
                      -1,
                      ::common::errors::InvalidArgument(
                          "start should be greater than or equal to -1"));
    PADDLE_ENFORCE_LT(start + 1,
                      dims->size(),
                      ::common::errors::InvalidArgument(
                          "start + 1 should be less than dims->size()"));
    ir::Expr ret = TranslateDimExpr(dims->at(start + 1));
    for (int idx = start + 2; idx < dims->size(); ++idx) {
      ret = ir::Mul::Make(ret, TranslateDimExpr(dims->at(idx)));
    }
    return ret;
  }

  using IndexDotValueOfList = IndexDotValue<List<Value>, List<std::int64_t>>;
  ir::Expr TranslateIndexDotValueOfList(const Value& value) const {
    const auto& [list_value, dim_values] =
        value.Get<IndexDotValue<Value, List<DimExpr>>>().tuple();
    const auto& values = list_value.Get<List<Value>>();
    PADDLE_ENFORCE_EQ(
        values->size(),
        dim_values->size(),
        ::common::errors::InvalidArgument(
            "values->size() should be equal to dim_values->size()"));

    std::vector<ir::Expr> strided_exprs{};
    for (std::size_t i = 0; i < values->size(); ++i) {
      const auto& value_expr = TranslateTensorIterator(values->at(i));
      const auto& stride_value = GetStride(dim_values, i);
      strided_exprs.emplace_back(ir::Mul::Make(value_expr, stride_value));
    }
    return Accumulate(strided_exprs);
  }

  using ListGetItemOfUnDot =
      ListGetItem<IndexUnDotValue<Value, List<std::int64_t>>, std::int64_t>;
  ir::Expr TranslateListGetItemOfUnDot(const Value& value) const {
    const auto& [undot_value, idx_value] =
        value.Get<ListGetItem<Value, DimExpr>>().tuple();
    const auto& [tensor_index_value, dims] =
        undot_value.Get<IndexUnDotValue<Value, List<DimExpr>>>().tuple();
    ir::Expr tensor_index_expr = TranslateTensorIterator(tensor_index_value);
    std::int64_t idx = idx_value.Get<std::int64_t>();

    ir::Expr mod_operand = GetStride(dims, idx - 1);
    ir::Expr div_operant = GetStride(dims, idx);
    return ir::Div::Make(ir::Mod::Make(tensor_index_expr, mod_operand),
                         div_operant);
  }

  ir::Expr TranslateIterator(const Value& value) const {
    const auto& iterator = value.Get<Iterator>();
    return ir::Var("v_" + std::to_string(iterator.value().unique_id()));
  }

  ir::Expr TranslateDimExprImpl(std::int64_t dim_expr) const {
    return ir::Expr(IteratorInt(dim_expr));
  }

  ir::Expr TranslateDimExprImpl(const SymbolicDim& dim_expr) const {
    return ir::Var{dim_expr};
  }

  ir::Expr TranslateDimExprImpl(
      const ::symbol::Negative<DimExpr>& dim_expr) const {
    const auto& [inner_dim_expr] = *dim_expr;
    ir::Expr inner_expr = TranslateDimExpr(inner_dim_expr);
    return ir::Sub::Make(ir::Expr(0), inner_expr);
  }

  ir::Expr TranslateDimExprImpl(const ::symbol::Add<DimExpr>& dim_expr) const {
    std::vector<ir::Expr> ir_exprs{};
    const auto& [exprs] = dim_expr;
    for (const auto& expr : *exprs) {
      ir_exprs.emplace_back(TranslateDimExpr(expr));
    }
    return Accumulate(ir_exprs);
  }

  ir::Expr TranslateDimExprImpl(const ::symbol::Mul<DimExpr>& dim_expr) const {
    std::vector<ir::Expr> ir_exprs{};
    const auto& [exprs] = dim_expr;
    for (const auto& expr : *exprs) {
      ir_exprs.emplace_back(TranslateDimExpr(expr));
    }
    return Multiply(ir_exprs);
  }

  ir::Expr TranslateDimExprImpl(const ::symbol::Div<DimExpr>& dim_expr) const {
    std::vector<ir::Expr> ir_exprs{};
    const auto& lhs = dim_expr->lhs;
    const auto& rhs = dim_expr->rhs;
    ir_exprs.emplace_back(TranslateDimExpr(lhs));
    ir_exprs.emplace_back(TranslateDimExpr(rhs));
    return Divide(ir_exprs);
  }

  ir::Expr TranslateDimExprImpl(const ::symbol::Max<DimExpr>& dim_expr) const {
    PADDLE_THROW(::common::errors::Unimplemented("Not supported yet"));
  }

  ir::Expr TranslateDimExprImpl(const ::symbol::Min<DimExpr>& dim_expr) const {
    PADDLE_THROW(::common::errors::Unimplemented("Not supported yet"));
  }

  ir::Expr TranslateDimExprImpl(
      const ::symbol::Broadcast<DimExpr>& dim_expr) const {
    PADDLE_THROW(::common::errors::Unimplemented("Not supported yet"));
  }

  ir::Expr TranslateDimExpr(const Value& value) const {
    const auto& dim_expr = value.Get<DimExpr>();
    return std::visit(
        [&](const auto& impl) { return TranslateDimExprImpl(impl); },
        dim_expr.variant());
  }

  // ADT_TODO(Hongyu Jia) : Directly return BI iterator maybe a little tricky
  using BroadcastedSymbolicIterator = BroadcastedIterator<Value, SymbolicDim>;
  ir::Expr TranslateBI(const Value& value) const {
    const auto& [iterator, symbolic_dim] =
        value.Get<BroadcastedIterator<Value, DimExpr>>().tuple();
    return TranslateIterator(iterator);
  }

  ir::Expr TranslateTensorIterator(const Value& value) const {
    if (Match<IndexDotValueOfList>(value)) {
      return TranslateIndexDotValueOfList(value);
    } else if (Match<ListGetItemOfUnDot>(value)) {
      return TranslateListGetItemOfUnDot(value);
    } else if (Match<Iterator>(value)) {
      return TranslateIterator(value);
    } else if (Match<DimExpr>(value)) {
      return TranslateDimExpr(value);
    } else if (Match<BroadcastedSymbolicIterator>(value)) {
      return TranslateBI(value);
    } else {
      std::stringstream ss;
      ss << "Not supported yet! " << ToTxtString(value);
      PADDLE_THROW(::common::errors::InvalidArgument(ss.str()));
    }
  }

  std::vector<ir::Expr> Translate(
      const List<TensorIteratorExpr>& iterator_exprs) const {
    std::vector<ir::Expr> ret{};
    for (const auto& iterator_expr : *iterator_exprs) {
      ret.emplace_back(TranslateTensorIterator(iterator_expr));
    }
    return ret;
  }

  MapExpr map_expr_;
  const Node2LoweredFuncs* node2lowered_funcs_;
  const cinn::common::Target target_;
  TensorIteratorExpr4TensorT TensorIteratorExpr4Tensor;
  LoopDescriptor4LoopIteratorT LoopDescriptor4LoopIterator;
};

}  // namespace

ir::Expr MapExprToIr(const MapExprCtx& map_expr_ctx,
                     const cinn::common::Target& target) {
  const auto& expr =
      MapExprToIrTranslator(
          map_expr_ctx.map_expr(), map_expr_ctx.node2lowered_funcs(), target)
          .Translate();
  VLOG(1) << "Finish MapExprToIr\n" << expr;
  return expr;
}

}  // namespace cinn::adt
