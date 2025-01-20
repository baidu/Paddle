// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/optim/vectorize_loops.h"

#include <absl/container/flat_hash_map.h>
#include <glog/logging.h>

#include <algorithm>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "paddle/cinn/common/cas.h"
#include "paddle/cinn/common/ir_util.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/op/ir_operators.h"
#include "paddle/cinn/ir/utils/ir_copy.h"
#include "paddle/cinn/ir/utils/ir_nodes_collector.h"
#include "paddle/cinn/ir/utils/ir_replace.h"
#include "paddle/cinn/optim/ir_simplify_pass.h"
#include "paddle/cinn/optim/unroll_loops.h"
#include "paddle/cinn/utils/functional.h"

namespace cinn {
namespace optim {
using namespace ir;  // NOLINT
using cinn::common::make_const;
using cinn::common::make_one;
using cinn::common::make_zero;

//! Widen an expression to the given number of lanes.
Expr Widen(Expr e, int lanes) {
  if (e.type().lanes() == lanes) return e;
  if (const ir::Broadcast *op = e.As<ir::Broadcast>()) {
    if (lanes % op->lanes == 0) {
      return ir::Broadcast::Make(op->value, lanes);
    }
  }

  PADDLE_ENFORCE_EQ(
      e.type().lanes(),
      1,
      ::common::errors::InvalidArgument(
          "Cannot broadcast lanes from %d to %d.", e.type().lanes(), lanes));
  return ir::Broadcast::Make(e, lanes);
}

// tell whether a tensor can be vectorized or not on CUDA by collecting names
// of tensors which meet all check predicates of vectoring
class TensorVectorizeTeller : public ir::IRMutator<const Expr *> {
 public:
  TensorVectorizeTeller(
      const Var &iter_var,
      const int factor,
      const absl::flat_hash_map<std::string, cinn::common::CasInterval>
          *var_intervals)
      : iter_var_(iter_var), factor_(factor), var_intervals_(var_intervals) {}

  void Collect(const Expr *op) { IRMutator::Visit(op, op); }

  // return true if input tensor can be vectorized
  bool CanBeVectorized(const std::string &tensor_name) const {
    auto it = tensor2flag_.find(tensor_name);
    return it != tensor2flag_.end() && it->second;
  }

 private:
  const Var
      iter_var_;  // loop var of new for-loop split from the vectorized loop
  const int factor_;
  const absl::flat_hash_map<std::string, cinn::common::CasInterval>
      *var_intervals_;
  // save (tensor name) -> (bool flag) to identify whether tensors can be
  // vectorized or not
  std::unordered_map<std::string, bool> tensor2flag_;

  void Visit(const ir::Store *expr, const Expr *op) override {
    auto *node = op->As<ir::Store>();
    PADDLE_ENFORCE_NOT_NULL(node,
                            ::common::errors::InvalidArgument(
                                "Expected Store node, but received nullptr."));
    IRMutator::Visit(&node->value, &node->value);
    auto *tensor = node->tensor.As<ir::_Tensor_>();
    PADDLE_ENFORCE_NOT_NULL(
        tensor,
        ::common::errors::InvalidArgument(
            "Expected _Tensor_ node in Store, but received nullptr."));

    // a tensor should pass all check of pre-conditions in every time it appears
    if (!tensor2flag_.count(tensor->name) || tensor2flag_.at(tensor->name)) {
      bool flag = MeetConditions(node->tensor, node->indices);
      tensor2flag_[tensor->name] = flag;
    }
  }

  void Visit(const ir::Load *expr, const Expr *op) override {
    auto *node = op->As<ir::Load>();
    PADDLE_ENFORCE_NOT_NULL(node,
                            ::common::errors::InvalidArgument(
                                "Expected Load node, but received nullptr."));
    auto *tensor = node->tensor.As<ir::_Tensor_>();
    PADDLE_ENFORCE_NOT_NULL(
        tensor,
        ::common::errors::InvalidArgument(
            "Expected _Tensor_ node in Load, but received nullptr."));

    // a tensor should pass all check of pre-conditions in every time it appears
    if (!tensor2flag_.count(tensor->name) || tensor2flag_.at(tensor->name)) {
      bool flag = MeetConditions(node->tensor, node->indices);
      tensor2flag_[tensor->name] = flag;
    }
  }

  // return true if the tensor meets all conditions of vectorizing
  bool MeetConditions(const Expr &expr, const std::vector<Expr> &indices) {
    const ir::_Tensor_ *tensor = expr.As<ir::_Tensor_>();
    auto find_matched_var_fn = [&](const Expr *x) {
      return x->As<_Var_>() && x->As<_Var_>()->name == iter_var_->name;
    };

    // the size of the last dim should be divisible by factor
    Expr last_size = tensor->shape.back();
    if (tensor->shape.empty() || !tensor->shape.back().As<IntImm>() ||
        tensor->shape.back().as_int32() % factor_ != 0) {
      VLOG(5) << "Size of the last dim of tensor:" << tensor->name
              << " can't be divisible by factor:" << factor_
              << ", shape:" << utils::Join(tensor->shape, ",");
      return false;
    }

    // the iter val must appear in the last index
    if (indices.empty() ||
        ir::ir_utils::CollectIRNodes(indices.back(), find_matched_var_fn)
            .empty()) {
      VLOG(5) << "Loop var:" << iter_var_->name
              << " is not used in the last index";
      return false;
    }

    // the iter val can't appear in multiple indices
    for (int i = 0; i < indices.size() - 1; ++i) {
      auto repeat_found =
          ir::ir_utils::CollectIRNodes(indices[i], find_matched_var_fn);
      if (!repeat_found.empty()) {
        VLOG(5) << "Loop var:" << iter_var_->name
                << " is used at more than last index, current:" << i;
        return false;
      }
    }

    // check tensor accessed sequentially by comparing index one by one
    Expr first_idx = ir::ir_utils::IRCopy(indices.back());
    cinn::ir::ir_utils::IrReplaceVarBroadcast(
        &first_idx, Expr(iter_var_), Expr(0));
    const auto &interval = var_intervals_->at(iter_var_->name);
    for (int i = 1; i < interval.r; ++i) {
      Expr next_idx = ir::ir_utils::IRCopy(indices.back());
      cinn::ir::ir_utils::IrReplaceVarBroadcast(
          &next_idx, Expr(iter_var_), Expr(i));
      auto gap = cinn::optim::ArithSimplify(Expr(next_idx - first_idx));
      if (!gap.As<IntImm>() || gap.as_int32() != i) {
        VLOG(5) << "Tensor:" << tensor->name
                << " is not accessed sequentially, next:" << next_idx
                << ", first:" << first_idx << ", gap:" << gap;
        return false;
      }
      VLOG(5) << "Tensor:" << tensor->name
              << " is accessed sequentially, next:" << next_idx
              << ", first:" << first_idx << ", gap:" << gap;
    }

    auto dtype = expr->type().ElementOf();
    bool type_supported = dtype.is_float(32) || dtype.is_int(32) ||
                          dtype.is_uint(32) || dtype.is_float16() ||
                          dtype.is_bfloat16();
    if (!type_supported) {
      VLOG(5)
          << "Only support vectorizing int,uint,float,float16,bloat16, but got "
          << dtype;
      return false;
    }
    return true;
  }
};

// find tensors accessed sequentially in a for-loop to be vectorized,
// and substitute the corresponding cuda built-in vector for them
class CudaVectorizer : public IRMutator<Expr *> {
  const Var iter_var_;  // the loop var of the vectorized loop
  const int factor_;    // the factor for vectorize

  std::set<std::string> write_teller_;
  TensorVectorizeTeller vectorized_teller_;

  absl::flat_hash_map<std::string, Var> tensor2vectorized_vars_;
  std::vector<Expr> vectorized_cast_exprs_;
  std::vector<Expr> vectorized_store_exprs_;

 public:
  static constexpr int CudaVectorTypeMaxLanes = 8;
  CudaVectorizer(
      const Var &iter_var,
      const int factor,
      const absl::flat_hash_map<std::string, cinn::common::CasInterval>
          *var_intervals)
      : iter_var_(iter_var),
        factor_(factor),
        vectorized_teller_(iter_var, factor, var_intervals) {
    PADDLE_ENFORCE_EQ(
        factor <= CudaVectorTypeMaxLanes,
        true,
        ::common::errors::InvalidArgument(
            "The maximum lanes of valid CUDA vector types: %d, but factor: %d.",
            CudaVectorTypeMaxLanes,
            factor));
  }

  // return all cast statements collected through vectorizing
  std::vector<Expr> VectorizedTypeCastExprs() { return vectorized_cast_exprs_; }

  // return all store statements collected through vectorizing
  std::vector<Expr> VectorizedTypeStoreExprs() {
    return vectorized_store_exprs_;
  }

  void Visit(Expr *expr) {
    write_teller_ = ir::ir_utils::CollectTensorNeedsWrite(expr);
    vectorized_teller_.Collect(expr);
    IRMutator<Expr *>::Visit(expr, expr);
  }

  void Visit(const Load *op, Expr *expr) override {
    auto *node = expr->As<Load>();
    auto *tensor = node->tensor.As<ir::_Tensor_>();
    if (node->is_addr_tensor() &&
        vectorized_teller_.CanBeVectorized(tensor->name)) {
      TensorVectorized(node, &node->indices, false);
    }
  }

  void Visit(const Store *op, Expr *expr) override {
    auto *node = expr->As<Store>();
    auto *tensor = node->tensor.As<ir::_Tensor_>();
    PADDLE_ENFORCE_NOT_NULL(
        tensor,
        ::common::errors::InvalidArgument(
            "Expected _Tensor_ node in Store, but received nullptr."));
    if (vectorized_teller_.CanBeVectorized(tensor->name)) {
      TensorVectorized(node, &node->indices, true);
    }

    IRMutator::Visit(&node->value, &node->value);
  }

 private:
  void TensorVectorized(ir::LoadStoreAddrMnger *node,
                        std::vector<Expr> *indices,
                        bool is_store) {
    auto *tensor = node->tensor.As<ir::_Tensor_>();
    VLOG(5) << "Vectorizing tensor:" << tensor->name;

    // save the tensor and its corresponding vector name when it first appear
    if (!tensor2vectorized_vars_.count(tensor->name)) {
      AppendCast(node->tensor, *indices, is_store);
    }

    auto vectorized_var = tensor2vectorized_vars_.at(tensor->name);
    // substitute a new tensor with the vector name and dtype
    auto t = vectorized_var->type().is_cpp_handle()
                 ? node->tensor->type().PointerOf()
                 : node->tensor->type();
    node->tensor = ir::Tensor(vectorized_var->name,
                              t,
                              {Expr(factor_)},
                              {Expr(factor_)},
                              tensor->operation);
    // remain the last iterative indice
    indices->assign({iter_var_});
  }

  std::string GetVectorTypeName(Type type) {
    std::string name_prefix =
        cinn::common::customized_type::kcuda_builtin_vector_t;
#define GET_CUDA_VECTOR_TYPE_NAME(pred_expr, scalar_name)       \
  if (pred_expr) {                                              \
    return name_prefix + scalar_name + std::to_string(factor_); \
  }

    GET_CUDA_VECTOR_TYPE_NAME(type.is_int(32), "int");
    GET_CUDA_VECTOR_TYPE_NAME(type.is_uint(32), "uint");
    GET_CUDA_VECTOR_TYPE_NAME(type.is_float(32), "float");
    GET_CUDA_VECTOR_TYPE_NAME(type.is_float16(), "half");
    GET_CUDA_VECTOR_TYPE_NAME(type.is_bfloat16(), "bfloat16");
#undef GET_CUDA_VECTOR_TYPE_NAME

    // others are not implemented yet
    CINN_NOT_IMPLEMENTED
    return "";
  }

  void AppendCast(Expr tensor,
                  const std::vector<Expr> &indices,
                  bool is_store) {
    auto *node = tensor.As<ir::_Tensor_>();
    bool is_const = !write_teller_.count(node->name);

    // generate the corresponding vector type
    Type scalar_type = tensor->type().ElementOf();
    Type vector_type_ptr(Type::type_t::Customized, scalar_type.bits(), factor_);
    Type vector_type(Type::type_t::Customized, scalar_type.bits(), factor_);
    vector_type_ptr.set_customized_type(GetVectorTypeName(scalar_type));
    vector_type_ptr.set_cpp_handle();
    vector_type_ptr.set_cpp_const(is_const);

    vector_type.set_customized_type(GetVectorTypeName(scalar_type));
    vector_type.set_cpp_const(is_const);

    // generate a local vector variable to be used in subsequent statements
    std::string vectorized_name = "vectorized_" + node->name;
    Var vectorized_var = _Var_::Make(vectorized_name, vector_type);
    tensor2vectorized_vars_.emplace(node->name, vectorized_var);

    // generate a get_addr expr to get the address of the tensor
    Expr converted_tensor = Load::Make(tensor, indices);
    cinn::ir::ir_utils::IrReplaceVarBroadcast(
        &converted_tensor, iter_var_, Expr(int32_t(0)));
    auto get_addr = ir::intrinsics::GetAddr::Make(converted_tensor);

    // generate a let expression to cast the tensor into the local vector
    auto cast = ir::Cast::Make(vector_type_ptr, get_addr);
    if (!is_store) {
      auto load = Load::Make(cast, {make_const(0)});
      auto let = Let::Make(vectorized_var, load);
      vectorized_cast_exprs_.emplace_back(let);
      VLOG(5) << "Append a vectorized expr:" << let;
    } else {
      Var vectorized_ptr =
          _Var_::Make(vectorized_name + "_ptr", vector_type_ptr);

      auto let1 = Let::Make(vectorized_ptr, cast);
      auto let2 = Let::Make(vectorized_var, Expr(0));
      vectorized_cast_exprs_.emplace_back(let1);
      vectorized_cast_exprs_.emplace_back(let2);

      VLOG(5) << "Append a vectorized expr:" << let1;
      VLOG(5) << "Append a vectorized expr:" << let2;

      auto t = ir::Tensor(vectorized_ptr->name,
                          node->type().PointerOf(),
                          {Expr(factor_)},
                          {Expr(factor_)},
                          node->operation);
      auto store = Store::Make(t, vectorized_var, {make_const(0)});

      vectorized_store_exprs_.emplace_back(store);
      VLOG(5) << "Append a vectorized expr:" << store;
    }
  }
};

//! Substitutes a vector for a scalar var in a Stmt.
class Vectorizer : public IRMutator<Expr *> {
  //! The name of the variable to be vectorized.
  Var var;

  int lanes_{-1};

  bool need_scalarize_{false};

  bool to_vectorize_{false};

  Expr ramp_;

  absl::flat_hash_map<std::string, cinn::common::CasInterval> var_intervals_;

  //! A suffix to attach to widened variables.
  std::string widen_suffix;

 public:
  Vectorizer(const Var &var,
             int lanes,
             const absl::flat_hash_map<std::string, cinn::common::CasInterval>
                 &var_intervals = {})
      : var(var), lanes_(lanes), var_intervals_(var_intervals) {
    // the identity ramp.
    ramp_ = Ramp::Make(make_zero(), make_one(), lanes_);
  }

  void Visit(Expr *expr) {
    PADDLE_ENFORCE_EQ(
        !need_scalarize_,
        true,
        ::common::errors::InvalidArgument("Expression needs scalarization."));
    IRMutator<Expr *>::Visit(expr, expr);

    if (need_scalarize_) {
      need_scalarize_ = false;
      Scalarize(expr);
    }
  }

  void Visit(const Cast *op, Expr *expr) override {
    auto *node = expr->As<Cast>();
    auto v0 = node->v();
    Visit(&node->v());
    if (v0.same_as(node->v())) return;

    Type t = op->type().with_lanes(node->v().type().lanes());
    node->set_type(t);
  }

  void Visit(const _Var_ *op, Expr *expr) override {
    if (op->name == var->name) {
      *expr = Expr(ramp_);
    }
  }

  void Visit(const Add *op, Expr *expr) override {
    MutateAddSubOperator(op, expr);
  }
  void Visit(const Sub *op, Expr *expr) override {
    MutateAddSubOperator(op, expr);
  }
  void Visit(const Mul *op, Expr *expr) override {
    MutateMulDivOperator(op, expr);
  }
  void Visit(const Div *op, Expr *expr) override {
    MutateMulDivOperator(op, expr);
  }
  void Visit(const Mod *op, Expr *expr) override {
    MutateMulDivOperator(op, expr);
  }
  void Visit(const Min *op, Expr *expr) override {
    BinaryOperatorVec(op, expr);
  }
  void Visit(const Max *op, Expr *expr) override {
    BinaryOperatorVec(op, expr);
  }
  void Visit(const EQ *op, Expr *expr) override { BinaryOperatorVec(op, expr); }
  void Visit(const NE *op, Expr *expr) override { BinaryOperatorVec(op, expr); }
  void Visit(const LT *op, Expr *expr) override { BinaryOperatorVec(op, expr); }
  void Visit(const LE *op, Expr *expr) override { BinaryOperatorVec(op, expr); }
  void Visit(const GT *op, Expr *expr) override { BinaryOperatorVec(op, expr); }
  void Visit(const GE *op, Expr *expr) override { BinaryOperatorVec(op, expr); }
  void Visit(const And *op, Expr *expr) override {
    BinaryOperatorVec(op, expr);
  }
  void Visit(const Or *op, Expr *expr) override { BinaryOperatorVec(op, expr); }

  void Visit(const Ramp *op, Expr *expr) override {}

  void Visit(const Select *op, Expr *expr) override {
    auto *node = expr->As<Select>();
    auto condition0 = node->condition;
    auto true_value0 = node->true_value;
    auto false_value0 = node->false_value;

    Visit(&node->condition);
    Visit(&node->true_value);
    Visit(&node->false_value);

    if (condition0.same_as(node->condition) &&
        true_value0.same_as(node->true_value) &&
        false_value0.same_as(node->false_value))
      return;

    int lanes = utils::Max(node->condition.type().lanes(),
                           node->true_value.type().lanes(),
                           node->false_value.type().lanes());
    node->true_value = Widen(node->true_value, lanes);
    node->false_value = Widen(node->false_value, lanes);
  }

  void Visit(const Load *op, Expr *expr) override {
    auto *node = expr->As<Load>();
    std::vector<Expr> indices = node->indices;
    // We ignore the predicate here.
    bool need_visit = false;
    for (int i = 0; i < indices.size(); i++) {
      Visit(&node->indices[i]);
      if (!node->indices[i].same_as(indices[i])) {
        need_visit = true;
      }
    }
    if (!need_visit) return;
    int lanes = 0;
    for (auto &idx : node->indices) {
      lanes = std::max(idx.type().lanes(), lanes);
    }
    std::vector<Expr> new_indices;
    for (auto &idx : node->indices) {
      new_indices.push_back(Widen(idx, lanes));
    }
    *expr = Load::Make(node->tensor, new_indices);
  }

  void Visit(const Store *op, Expr *expr) override {
    auto *node = expr->As<Store>();
    auto value0 = node->value;
    Visit(&node->value);

    std::vector<Expr> indices = node->indices;
    // We ignore the predicate here.
    for (auto &idx : node->indices) {
      Visit(&idx);
    }

    bool need_visit = false;
    for (int i = 0; i < indices.size(); i++) {
      if (!node->indices[i].same_as(indices[i])) {
        need_visit = true;
      }
    }
    if (!need_visit) return;

    int lanes = 0;
    for (auto &idx : node->indices) lanes = std::max(idx.type().lanes(), lanes);
    lanes = std::max(lanes, node->value.type().lanes());

    node->value = Widen(node->value, lanes);

    std::vector<Expr> new_indices;
    for (auto &idx : node->indices) {
      new_indices.push_back(Widen(idx, lanes));
    }
    *expr = Store::Make(node->tensor, node->value, new_indices);
  }

  void Visit(const Call *op, Expr *expr) override {
    std::vector<Expr> read_args = op->read_args;
    std::vector<Expr> write_args = op->write_args;
    auto *node = expr->As<Call>();
    ir::IRMutator<>::Visit(op, expr);
    bool is_changed = false;
    int lanes = 0;
    for (int i = 0; i < node->read_args.size(); i++) {
      lanes = std::max(node->read_args[i].type().lanes(), lanes);
      if (!node->read_args[i].same_as(read_args[i])) {
        is_changed = true;
      }
    }
    for (int i = 0; i < node->write_args.size(); i++) {
      lanes = std::max(node->write_args[i].type().lanes(), lanes);
      if (!node->write_args[i].same_as(write_args[i])) {
        is_changed = true;
      }
    }
    if (!is_changed) return;

    for (int i = 0; i < read_args.size(); i++) {
      node->read_args[i] = Widen(node->read_args[i], lanes);
    }
    for (int i = 0; i < write_args.size(); i++) {
      node->write_args[i] = Widen(node->write_args[i], lanes);
    }

    PADDLE_ENFORCE_EQ(!read_args.empty(),
                      true,
                      ::common::errors::InvalidArgument(
                          "The argument is empty, please check."));
    Type type = op->type().with_lanes(lanes);
    *expr = Call::Make(type,
                       node->name,
                       node->read_args,
                       node->write_args,
                       node->call_type,
                       node->func,
                       node->value_index,
                       node->attrs);
  }

  void Visit(const Let *op, Expr *expr) override {
    auto *node = expr->As<Let>();
    Visit(&node->symbol);
    LOG(ERROR) << "Let not supported";
  }

  void Visit(const IfThenElse *op, Expr *expr) override {
    auto *node = expr->As<IfThenElse>();
    Visit(&node->condition);
    int lanes = node->condition.type().lanes();
    Visit(&node->true_case);
    Visit(&node->false_case);
    LOG(ERROR) << "Ignore Width IfThenElse";
  }

  void Visit(const For *op, Expr *expr) override {
    ir::IRMutator<>::Visit(op, expr);
  }

  void Scalarize(Expr *expr) {
    Var idx(var->name + "_s", Int(32));
    std::map<const ir::_Var_ *, Expr> var_map;
    var_map[var.As<ir::_Var_>()] = idx;

    cinn::common::Substitute(expr, var_map);
    *expr = ir::For::Make(idx,
                          cinn::common::make_const(0),
                          cinn::common::make_const(lanes_),
                          ForType::Serial,
                          DeviceAPI::Host,
                          *expr);
  }

  template <typename T>
  void MutateAddSubOperator(const T *op, Expr *expr) {
    auto *node = expr->As<T>();
    Expr a0 = node->a();
    Expr b0 = node->b();

    Visit(&node->a());
    Visit(&node->b());

    // if (a0.same_as(node->a()) && b0.same_as(node->b())) return;

    int lanes = std::max(node->a().type().lanes(), node->b().type().lanes());
    if (lanes != 1) {
      const Ramp *a_ramp_n = node->a().template As<Ramp>();
      const Ramp *b_ramp_n = node->b().template As<Ramp>();
      if (node->a().type().lanes() == 1 && b_ramp_n) {
        // a + Ramp(base,stride,lanes) = Ramp(base+a, stride,lanes)
        *expr = Ramp::Make(T::Make(node->a(), b_ramp_n->base),  // base
                           b_ramp_n->stride,                    // stride
                           b_ramp_n->lanes);
        return;
      }
      if (node->b().type().lanes() == 1 && a_ramp_n) {
        *expr = Ramp::Make(T::Make(node->b(), a_ramp_n->base),  // base
                           a_ramp_n->stride,                    // stride
                           a_ramp_n->lanes);
        return;
      }
    }

    *expr = T::Make(Widen(node->a(), lanes), Widen(node->b(), lanes));
  }

  template <typename T>
  void MutateMulDivOperator(const T *op, Expr *expr) {
    Expr a0 = op->a();
    Expr b0 = op->b();
    auto *node = expr->As<T>();
    Visit(&node->a());
    Visit(&node->b());

    // if (a0.same_as(node->a()) && b0.same_as(node->b())) return;
    int lanes = std::max(node->a().type().lanes(), node->b().type().lanes());
    if (lanes != 1) {
      const Ramp *a_ramp_n = node->a().template As<Ramp>();
      const Ramp *b_ramp_n = node->b().template As<Ramp>();
      if (node->a().type().lanes() == 1 && b_ramp_n) {
        // a * Ramp(base,stride,lanes) = Ramp(base*a, stride*a,lanes)
        *expr = Ramp::Make(T::Make(node->a(), b_ramp_n->base),    // base
                           T::Make(node->a(), b_ramp_n->stride),  // stride
                           b_ramp_n->lanes);

        return;
      }
      // Ramp(base,stride,lanes) * b  = Ramp(base*b, stride*b,lanes)
      if (node->b().type().lanes() == 1 && a_ramp_n) {
        *expr = Ramp::Make(T::Make(a_ramp_n->base, node->b()),    // base
                           T::Make(a_ramp_n->stride, node->b()),  // stride
                           a_ramp_n->lanes);
        return;
      }
    }

    *expr = T::Make(Widen(node->a(), lanes), Widen(node->b(), lanes));
  }

  template <typename T>
  void BinaryOperatorVec(const T *op, Expr *expr) {
    auto *node = expr->As<T>();
    Expr a0 = node->a();
    Expr b0 = node->b();
    Visit(&node->a());
    Visit(&node->b());
    // if (a0.same_as(node->a()) && b0.same_as(node->b())) return *expr;

    int lanes = std::max(node->a().type().lanes(), node->b().type().lanes());
    *expr = T::Make(Widen(node->a(), lanes), Widen(node->b(), lanes));
  }
};

struct VectorizeLoops_ : public IRMutator<Expr *> {
  const Target &target;
  absl::flat_hash_map<std::string, cinn::common::CasInterval> var_intervals;
  bool vectorizable_ = true;

  explicit VectorizeLoops_(const Target &t) : target(t) {}

  void operator()(Expr *expr) { IRMutator::Visit(expr, expr); }

  void Visit(const Load *op, Expr *expr) override {
    auto *node = expr->As<Load>();
    std::vector<Expr> indices = node->indices;

    bool is_changed = false;
    // simplify the complicated index from poly in the format of div/mod
    for (int i = 0; i < indices.size(); i++) {
      node->indices[i] =
          cinn::common::AutoSimplify(node->indices[i], var_intervals);
      Simplify(&node->indices[i], optim::SimplifyType::kExpr);
      if (!node->indices[i].same_as(indices[i])) {
        is_changed = true;
      }
    }
    if (!is_changed) return;

    *expr = Load::Make(node->tensor, node->indices);
  }

  void Visit(const Store *op, Expr *expr) override {
    auto *node = expr->As<Store>();
    auto value = node->value;
    IRMutator::Visit(&node->value, &node->value);

    std::vector<Expr> indices = node->indices;
    bool is_changed = false;
    // simplify the complicated index from poly in the format of div/mod
    for (int i = 0; i < indices.size(); i++) {
      node->indices[i] =
          cinn::common::AutoSimplify(node->indices[i], var_intervals);
      Simplify(&node->indices[i], optim::SimplifyType::kExpr);
      if (!node->indices[i].same_as(indices[i])) {
        is_changed = true;
      }
    }
    if (!is_changed) return;

    *expr = Store::Make(node->tensor, node->value, node->indices);
  }

  void Visit(const Call *op, Expr *expr) override {
    auto it = op->attrs.find("vectorizable");
    if (it != op->attrs.end()) {
      vectorizable_ = absl::get<bool>(it->second);
    }
  }

  void Visit(const For *forloop, Expr *expr) {
    auto *node = expr->As<For>();
    auto loop_var_name = forloop->loop_var->name;
    auto *extern_i = forloop->extent.As<IntImm>();
    if (extern_i && extern_i->value > 0) {
      var_intervals.emplace(loop_var_name,
                            cinn::common::CasInterval{static_cast<int64_t>(0),
                                                      extern_i->value - 1});
    } else if (!extern_i) {
      var_intervals.emplace(
          loop_var_name,
          cinn::common::CasInterval{Expr(0), forloop->extent - 1});
    }
    // the extent the forloops marked as Vectorized should be int constant
    if (forloop->is_vectorized()) {
      Context::info_rgt().Get<int>("vectorized_forloop_count")++;

      PADDLE_ENFORCE_GT(
          forloop->vectorize_info().factor,
          0,
          ::common::errors::InvalidArgument(
              "The value of factor in forloop's vectorize_info is incorrect."
              "Expected value is larger than 0, but receive %d. ",
              forloop->vectorize_info().factor));

      PADDLE_ENFORCE_EQ(
          is_zero(forloop->min),
          true,
          ::common::errors::InvalidArgument(
              "The minimum of forloop should be zero, please check."));
      Expr for_extent = cinn::optim::ArithSimplify(forloop->extent);
      Simplify(&for_extent, optim::SimplifyType::kExpr);
      node->extent = for_extent;
      auto *extent_min = for_extent.As<Min>();
      auto *extent_max = for_extent.As<Max>();

      vectorizable_ = true;
      IRMutator<>::Visit(&node->body, &node->body);

      target.arch.Match(
          [&](common::NVGPUArch) {
            if (!forloop->extent.As<IntImm>() ||
                forloop->extent.as_int32() % forloop->vectorize_info().factor !=
                    0) {
              vectorizable_ = false;
              VLOG(5) << "GPU vectorize only support extent is a multiple of "
                         "factor";
            }
          },
          [&](std::variant<common::HygonDCUArchHIP, common::HygonDCUArchSYCL>) {
            if (!forloop->extent.As<IntImm>() ||
                forloop->extent.as_int32() % forloop->vectorize_info().factor !=
                    0) {
              vectorizable_ = false;
              VLOG(5) << "DCU vectorize only support extent is a multiple of "
                         "factor";
            }
          },
          [&](std::variant<common::UnknownArch,
                           common::X86Arch,
                           common::ARMArch>) {});

      if (extent_min || extent_max || !vectorizable_) {
        // not vectorize if has tail blocks, for llvm to optimize
        node->reset_vectorize_info();
        var_intervals.erase(forloop->loop_var->name);
        return;
      }

      const int factor = forloop->vectorize_info().factor;
      auto _new_forloop = SplitForLoop(node, factor);
      if (!_new_forloop.defined()) {
        IRMutator<>::Visit(&node->body, &node->body);
        var_intervals.erase(forloop->loop_var->name);
        return;
      }

      node->reset_vectorize_info();

      auto *new_forloop = _new_forloop.As<ir::For>();

      // The forloop generated from polyhedral analysis might have a complex
      // condition that is not something like "i<20" or "i<=20", those cases is
      // not possible to extract the extent.
      auto *extent_int = new_forloop->extent.As<IntImm>();

      if (!extent_int) {
        IRMutator<>::Visit(&node->body, &node->body);
        var_intervals.erase(forloop->loop_var->name);
        return;
      }

      int extent = extent_int->value;
      PADDLE_ENFORCE_GT(
          extent,
          0,
          ::common::errors::InvalidArgument(
              "Loop over %s has extent %d"
              ". Can only vectorize loops over a constant extent > 1",
              Expr(new_forloop->loop_var),
              new_forloop->extent));

      VLOG(2) << "Vectorizing " << new_forloop->loop_var << " extent "
              << extent;
      VLOG(2) << "before vectorize body:\n" << node->body;
      auto setNvHygon = [&] {
        CudaVectorizer cuda_vectorizer(
            new_forloop->loop_var, factor, &var_intervals);
        cuda_vectorizer.Visit(&new_forloop->body);
        // unroll the new forloop to compute each element of the vector
        // iteratively
        auto copied_loop =
            ir::ir_utils::IRCopy(_new_forloop, /* copy_buffer_node = */ false);
        copied_loop.As<ir::For>()->set_unrolled();
        optim::UnrollLoop(&copied_loop);
        // add cast exprs of vector type in the front of vectorized forloop,
        // and replace original compute statements with the correspond
        // unrolled ones
        auto unroll_body = copied_loop.As<ir::Block>()->stmts;
        auto cast_exprs = cuda_vectorizer.VectorizedTypeCastExprs();
        auto store_exprs = cuda_vectorizer.VectorizedTypeStoreExprs();
        auto &body_stmts = new_forloop->body.As<ir::Block>()->stmts;
        body_stmts.assign(cast_exprs.begin(), cast_exprs.end());
        body_stmts.insert(
            body_stmts.end(), unroll_body.begin(), unroll_body.end());
        body_stmts.insert(
            body_stmts.end(), store_exprs.begin(), store_exprs.end());
      };
      target.arch.Match(
          [&](common::NVGPUArch) { setNvHygon(); },
          [&](std::variant<common::UnknownArch,
                           common::X86Arch,
                           common::ARMArch>) {
            Vectorizer(new_forloop->loop_var, extent, var_intervals)
                .Visit(&new_forloop->body);
          },
          [&](std::variant<common::HygonDCUArchHIP, common::HygonDCUArchSYCL>) {
            setNvHygon();
          });

      VLOG(2) << "after vectorize body:\n" << node->body;

      // Remove the forloop, the new_forloop's body is vectorized to Ramp, so no
      // forloop is needed.
      if (is_zero(forloop->extent - 1)) {
        *expr = new_forloop->body;
      } else {
        node->body = new_forloop->body;
      }
    } else {
      IRMutator::Visit(forloop, expr);
    }
    var_intervals.erase(loop_var_name);
  }

  //! unroll the forloop if its' extent is min type by solving the condition
  //! extent
  //! @return The new forloop.
  bool UnrollCmpFor(For *outer_for, For *inner_for, Expr *expr) {
    PADDLE_ENFORCE_NOT_NULL(
        outer_for,
        ::common::errors::InvalidArgument(
            "Outer_for is nullptr in UnrollCmpFor function."));
    PADDLE_ENFORCE_NOT_NULL(
        inner_for,
        ::common::errors::InvalidArgument(
            "Inner_for is nullptr in UnrollCmpFor function."));
    Expr inner_for_extent = cinn::common::AutoSimplify(inner_for->extent);
    Simplify(&inner_for_extent, optim::SimplifyType::kExpr);
    auto *extent_min = inner_for_extent.As<Min>();
    if (extent_min) {
      PADDLE_ENFORCE_EQ(
          is_zero(inner_for->min),
          true,
          ::common::errors::InvalidArgument(
              "The minimum of inner_for should be zero, please check."));
      // simplify the complicated indices of load/store from poly
      IRMutator::Visit(&inner_for->body, &inner_for->body);
      Expr a, b, condition;
      a = extent_min->a();
      b = extent_min->b();
      auto a_int = a.As<IntImm>();
      auto b_int = a.As<IntImm>();
      if (a_int || b_int) {
        condition =
            cinn::common::SolveInequality(LE::Make(a, b), outer_for->loop_var);
        Simplify(&condition, optim::SimplifyType::kExpr);
      }
      if (condition.defined()) {
        auto le_n = condition.As<ir::LE>();
        bool can_split = le_n && le_n->b().is_constant();
        if (le_n && le_n->b().is_constant()) {
          Expr inner_for_a =
              Block::Make({For::Make(inner_for->loop_var,
                                     inner_for->min,
                                     a,
                                     ForType::Vectorized,
                                     DeviceAPI::UNK,
                                     inner_for->body,
                                     inner_for->vectorize_info())});
          Expr new_extent_a = cinn::optim::ArithSimplify(le_n->b() + 1);
          Expr out_for_a = For::Make(outer_for->loop_var,
                                     outer_for->min,
                                     new_extent_a,
                                     outer_for->for_type(),
                                     outer_for->device_api,
                                     inner_for_a,
                                     outer_for->vectorize_info());
          Var new_iterator_inner(
              cinn::common::UniqName(inner_for->loop_var->name + "_s"));
          Var new_iterator_outer(
              cinn::common::UniqName(outer_for->loop_var->name + "_s"));

          Expr inner_for_b = Block::Make({For::Make(
              new_iterator_inner,
              inner_for->min,
              b,
              ForType::Serial,
              DeviceAPI::UNK,
              ir::ir_utils::IRCopy(inner_for->body,
                                   /* copy_buffer_node = */ false))});
          cinn::ir::ir_utils::IrReplaceVarBroadcast(
              &inner_for_b, inner_for->loop_var, Expr(new_iterator_inner));

          Expr out_for_b = For::Make(new_iterator_outer,
                                     new_extent_a,
                                     outer_for->extent,
                                     outer_for->for_type(),
                                     outer_for->device_api,
                                     inner_for_b,
                                     outer_for->vectorize_info());
          cinn::ir::ir_utils::IrReplaceVarBroadcast(
              &out_for_b, outer_for->loop_var, Expr(new_iterator_outer));
          *expr = Block::Make({out_for_a, out_for_b});
          VLOG(2) << *expr;
          IRMutator::Visit(expr, expr);
          return true;
        }
      }
    }
    return false;
  }

  //! Split the forloop with size \p factor.
  //! @return The new forloop.
  Expr SplitForLoop(For *forloop, int factor) {
    PADDLE_ENFORCE_GT(factor,
                      1,
                      ::common::errors::InvalidArgument(
                          "The value of factor in SplitForLoop is incorrect."
                          "Expected value is larger than 1, but receive %d. ",
                          factor));
    auto *for_min_i = forloop->min.As<IntImm>();
    PADDLE_ENFORCE_NOT_NULL(
        forloop,
        ::common::errors::InvalidArgument(
            "The forloop is nullptr in SplitForLoop function."));
    if (!for_min_i) return Expr();
    if (for_min_i->value != 0) return Expr();

    auto *extent_ptr = forloop->extent.As<IntImm>();
    Expr times;
    if (extent_ptr) {
      if (extent_ptr->value == 0) return Expr();
      int extent_int = forloop->extent.as_int32();
      int extent_trunc = extent_int / factor;
      int extent_times =
          extent_int % factor == 0 ? extent_trunc : extent_trunc + 1;
      times = cinn::common::make_const(forloop->extent->type(), extent_times);
    } else {
      times = cinn::optim::ArithSimplify(
          Div::Make(forloop->extent, make_const(factor)));
      Simplify(&times, optim::SimplifyType::kExpr);
    }

    // update the current forloop
    auto times_int = times.As<IntImm>();
    forloop->set_vectorized(false);

    forloop->extent = times;
    if (times_int) {
      var_intervals.emplace(forloop->loop_var->name,
                            cinn::common::CasInterval{static_cast<int64_t>(0),
                                                      times_int->value - 1});
    } else {
      var_intervals.erase(forloop->loop_var->name);
      var_intervals.emplace(
          forloop->loop_var->name,
          cinn::common::CasInterval{Expr(0), forloop->extent - 1});
    }

    // create the new forloop
    {
      Var new_iterator(Context::Global().NewName("vi"));
      var_intervals.emplace(new_iterator->name,
                            cinn::common::CasInterval{0, factor - 1});
      // eliminate for 1
      Expr new_index;
      if (cinn::common::is_zero(times - 1)) {
        new_index = Expr(new_iterator);
      } else {
        new_index = Expr(forloop->loop_var) * factor + Expr(new_iterator);
      }
      cinn::ir::ir_utils::IrReplaceVarBroadcast(
          &forloop->body, forloop->loop_var, new_index);
      auto new_forloop = For::Make(new_iterator,
                                   forloop->min,
                                   make_const(factor),
                                   ForType::Vectorized,
                                   DeviceAPI::UNK,
                                   forloop->body,
                                   forloop->vectorize_info());
      forloop->body = Block::Make({new_forloop});
      return new_forloop;
    }
  }
};

void VectorizeLoops(Expr *expr, const Target &target) {
  return VectorizeLoops_(target)(expr);
}

namespace detail {

void Vectorize(Var var, int lanes, Expr *expr) {
  Vectorizer vectorizer(var, lanes);
  vectorizer.Visit(expr);
}

}  // namespace detail

}  // namespace optim
}  // namespace cinn
