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

#include "paddle/cinn/optim/vectorize_for_trans.h"

#include <unordered_map>
#include <unordered_set>
#include "paddle/cinn/common/ir_util.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/ir/ir_mutator.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/utils/ir_copy.h"
#include "paddle/cinn/ir/utils/ir_nodes_collector.h"
#include "paddle/cinn/ir/utils/ir_replace.h"
#include "paddle/cinn/optim/ir_simplify.h"
#include "paddle/cinn/optim/unroll_loops.h"

namespace cinn {
namespace optim {

namespace {

std::unordered_map<std::string, ir::Var> CollectExprSymbols(Expr *x) {
  struct Mutator : public ir::IRMutator<Expr *> {
    void operator()(ir::Expr *expr) { ir::IRMutator<>::Visit(expr, expr); }
    void Visit(const ir::_Var_ *op, Expr *expr) override {
      auto *node = expr->As<ir::_Var_>();
      PADDLE_ENFORCE_NOT_NULL(node,
                              ::common::errors::InvalidArgument(
                                  "Sorry, but the node expr is nullptr"));
      if (!symbols_.count(op->name)) {
        symbols_.insert({op->name, ir::Var(node)});
      }
    }

    std::unordered_map<std::string, ir::Var> GetSymbols() { return symbols_; }

   private:
    std::unordered_map<std::string, ir::Var> symbols_;
  };

  Mutator mutator;
  mutator(x);
  return std::move(mutator.GetSymbols());
}

Expr CalculateTensorOffsetWithIndexes(Expr *tensor,
                                      const std::vector<ir::Expr> &indices) {
  auto *tensor_ptr = tensor->As<ir::_Tensor_>();
  PADDLE_ENFORCE_NOT_NULL(
      tensor_ptr,
      ::common::errors::InvalidArgument(
          "Expected _Tensor_ node in Store, but received nullptr."));

  Expr offset = indices[0];
  for (int i = 1; i < tensor_ptr->shape.size(); ++i) {
    Expr size = tensor_ptr->shape[i];
    Expr index = indices[i];
    offset = ir::Add::Make(ir::Mul::Make(offset, size), index);
  }
  return offset;
}

Expr UpdateOffsetOnlyContainsVectorizeAxis(Expr offset, Var vectorize_axis) {
  PADDLE_ENFORCE_NOT_NULL(
      &offset,
      ::common::errors::InvalidArgument(
          "Expected offset expr ptr, but received nullptr."));
  auto var_symbols = CollectExprSymbols(&offset);
  auto update_offset = ir::ir_utils::IRCopy(offset);
  for (const auto &[key, value] : var_symbols) {
    if (key == vectorize_axis->name) continue;
    cinn::ir::ir_utils::IrReplaceVarBroadcast(
        &update_offset, Expr(value), Expr(int32_t(0)));
  }
  cinn::optim::Simplify(&update_offset);
  return update_offset;
}

bool IsSelectOpWithSpecialOffset(Expr offset) {
  PADDLE_ENFORCE_NOT_NULL(
      &offset,
      ::common::errors::InvalidArgument(
          "Expected offset expr ptr, but received nullptr."));
  auto var_symbols = CollectExprSymbols(&offset);
  auto selectOp_offset = cinn::ir::ir_utils::IRCopy(offset);
  for (const auto &[key, value] : var_symbols) {
    cinn::ir::ir_utils::IrReplaceVarBroadcast(
        &selectOp_offset, Expr(value), Expr(int32_t(0)));
  }
  cinn::optim::Simplify(&selectOp_offset);
  auto const_val = selectOp_offset.As<ir::IntImm>();
  if (const_val && const_val->value < 0) {
    return true;
  }
  return false;
}

Expr CalculateOffsetWithVectorizeAxis(Expr offset,
                                      Expr origin_offset,
                                      Var var_iter,
                                      const int value) {
  PADDLE_ENFORCE_NOT_NULL(
      &offset,
      ::common::errors::InvalidArgument(
          "Expected offset expr ptr, but received nullptr."));
  PADDLE_ENFORCE_NOT_NULL(
      &origin_offset,
      ::common::errors::InvalidArgument(
          "Expected offset expr ptr, but received nullptr."));
  Expr next = cinn::ir::ir_utils::IRCopy(offset);
  cinn::ir::ir_utils::IrReplaceVarBroadcast(
      &next, Expr(var_iter), Expr(int32_t(value)));
  cinn::optim::Simplify(&next);
  auto compare = ir::Sub::Make(next, origin_offset);
  cinn::optim::Simplify(&compare);
  return compare;
}

Expr GetOriginOffsetWithVectorizeAxis(Expr offset, Var var_iter) {
  PADDLE_ENFORCE_NOT_NULL(
      &offset,
      ::common::errors::InvalidArgument(
          "Expected offset expr ptr, but received nullptr."));
  Expr origin_offset = cinn::ir::ir_utils::IRCopy(offset);
  cinn::ir::ir_utils::IrReplaceVarBroadcast(
      &origin_offset, Expr(var_iter), Expr(int32_t(0)));
  cinn::optim::Simplify(&origin_offset);
  return origin_offset;
}

class ForOpWithMultiScheduleBlockSupportVectorize
    : public ir::IRMutator<ir::Expr *> {
 public:
  void operator()(ir::Expr *expr) { ir::IRMutator<>::Visit(expr, expr); }

 private:
  void Visit(const ir::ScheduleBlockRealize *op, Expr *expr) override {
    auto *node = expr->As<ir::ScheduleBlockRealize>();
    PADDLE_ENFORCE_NOT_NULL(
        node,
        ::common::errors::InvalidArgument("The input expr should be a Block"));
    IRMutator<>::Visit(op, expr);
    if (in_vectorize_scope) {
      for_op_blocks_.push_back(expr);
    }
  }

  void Visit(const ir::For *op, ir::Expr *expr) override {
    auto *forloop = expr->As<ir::For>();
    if (forloop->is_vectorized()) in_vectorize_scope = true;

    IRMutator<>::Visit(op, expr);

    if (for_op_blocks_.size() > 1 && in_vectorize_scope) {
      std::vector<Expr> stmts;
      for (auto block : for_op_blocks_) {
        Var new_iterator(
            cinn::common::UniqName(forloop->loop_var->name + "_s"));

        cinn::ir::ir_utils::IrReplaceVarBroadcast(
            block, forloop->loop_var, Expr(new_iterator));

        ir::Expr f_expr = ir::For::Make(new_iterator,
                                        forloop->min,
                                        forloop->extent,
                                        forloop->for_type(),
                                        forloop->device_api,
                                        ir::Block::Make({*block}),
                                        forloop->vectorize_info(),
                                        forloop->bind_info());
        stmts.push_back(f_expr);
      }
      Expr block_expr = ir::Block::Make(stmts);
      *expr = block_expr;
    }
    in_vectorize_scope = false;
    for_op_blocks_.clear();
  }

  bool in_vectorize_scope{false};
  std::vector<ir::Expr *> for_op_blocks_;
};

class ScheduleBlockTensorVectorizeTeller : public ir::IRMutator<Expr *> {
 public:
  ScheduleBlockTensorVectorizeTeller(Var iter_var, const int factor)
      : iter_var_(iter_var), factor_(factor) {}

  void Collect(ir::Expr *expr) { ir::IRMutator<>::Visit(expr, expr); }

  bool EnableVectorize() {
    return vectorize_tensors_.size() != 0 && schedule_block_can_vectorize_;
  }

  std::unordered_set<std::string> GetVectorizeTensors() const {
    return vectorize_tensors_;
  }

 private:
  void Visit(const ir::Store *expr, Expr *op) override {
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

    if (!schedule_block_can_vectorize_) {
      vectorize_tensors_.clear();
      return;
    }

    bool tensor_can_vetorize = TensorCanVectorize(node, node->indices);
    if (node->is_addr_tensor() && tensor_can_vetorize) {
      vectorize_tensors_.insert(tensor->name);
      return;
    }

    if (!tensor_can_vetorize && vectorize_tensors_.count(tensor->name)) {
      vectorize_tensors_.erase(tensor->name);
      return;
    }

    return;
  }

  void Visit(const ir::Load *expr, Expr *op) override {
    auto *node = op->As<ir::Load>();
    PADDLE_ENFORCE_NOT_NULL(node,
                            ::common::errors::InvalidArgument(
                                "Expected Load node, but received nullptr."));
    auto *tensor = node->tensor.As<ir::_Tensor_>();
    PADDLE_ENFORCE_NOT_NULL(
        tensor,
        ::common::errors::InvalidArgument(
            "Expected _Tensor_ node in Load, but received nullptr."));

    if (!schedule_block_can_vectorize_) {
      vectorize_tensors_.clear();
      return;
    }

    bool tensor_can_vetorize = TensorCanVectorize(node, node->indices);
    if (node->is_addr_tensor() && tensor_can_vetorize) {
      vectorize_tensors_.insert(tensor->name);
      return;
    }

    if (!tensor_can_vetorize && vectorize_tensors_.count(tensor->name)) {
      vectorize_tensors_.erase(tensor->name);
      return;
    }

    return;
  }

  bool IsScalarTensorWithoutVectorizeAxis(
      ir::LoadStoreAddrMnger *node, const std::vector<ir::Expr> &indices) {
    bool without_vectorize_axis = true;
    for (auto var : indices) {
      auto index_symbols = CollectExprSymbols(&var);
      if (index_symbols.count(iter_var_->name)) {
        without_vectorize_axis = false;
        break;
      }
    }
    if (without_vectorize_axis) return true;
    return false;
  }

  /**
   * Situation 1. Check if tensor can vectorize.
   * eg 1 : Address access of tensor without vectorize axis.
   *  serial for (i, 0, 4)
   *    serial for (j, 0, 4)
   *      vectorize[4] for (v1, 0, 4)
   *        float a[i, j, v1] = float b[i, j, v1] + float c[i, j]
   *
   *  c[i, j] is a scalar tensor.
   *
   * eg 2: Address access of tensor contains vectorize axis.
   * but tensor is a scalar tensor in the vectorize loop.
   *  serial for (i, 0, 4)
   *  {
   *    serial for (j, 0, 16)
   *    {
   *      vectorize[4] for (v1, 0, 4)
   *      {
   *        float a[i, j, v1] =  float b[(i * 64 + j * 4 + v1) / 4]
   *      }
   *    }
   *  }
   *
   *  b[(i * 64 + j * 4 + v1) / 4] is a scalar tensor.
   *
   * Situation 2. don't deal with select situation with offset < 0.
   *  serial for (i, 0, 4)
   *  {
   *    serial for (j, 0, 16)
   *    {
   *      vectorize[4] for (v1, 0, 4)
   *      {
   *        float a[i, j, v1] =  select(i < 2, float b[i, j, v1], float c[i - 2,
   * j, v1])
   *      }
   *    }
   *  }
   * c[i - 2, j, v1] when i = 0, j = 0, v1 = 0, offset = -128
   *
   * Situation 3 . Tensor offset with vectorize axis is continuous
   */
  bool TensorCanVectorize(ir::LoadStoreAddrMnger *node,
                          const std::vector<ir::Expr> &indices) {
    // not support bool type tensor
    auto *tensor = node->tensor.As<ir::_Tensor_>();
    if (tensor->type().ElementOf().is_bool()) {
      return false;
    }

    // situation 1 : Tensor is scalar in vectorize var_loop
    // eg 1 : Address access of tensor without vectorize axis.
    if (IsScalarTensorWithoutVectorizeAxis(node, indices)) {
      return false;
    }

    // eg 2 : Address access of tensor contains vectorize axis.
    Expr offset = CalculateTensorOffsetWithIndexes(&node->tensor, indices);
    // situation 2. don't deal with select situation
    if (IsSelectOpWithSpecialOffset(offset)) {
      vectorize_tensors_.clear();
      schedule_block_can_vectorize_ = false;
      return false;
    }

    Expr only_vectorize_axis_offset =
        UpdateOffsetOnlyContainsVectorizeAxis(offset, iter_var_);

    Expr origin_offset =
        GetOriginOffsetWithVectorizeAxis(only_vectorize_axis_offset, iter_var_);
    bool offset_is_zero = true;
    bool tensor_is_continuous = true;
    for (int i = 1; i < factor_; i++) {
      Expr compare = CalculateOffsetWithVectorizeAxis(
          only_vectorize_axis_offset, origin_offset, iter_var_, i);
      auto const_val = compare.As<ir::IntImm>();
      if (!const_val) return false;
      if (const_val->value != 0) {
        offset_is_zero = false;
      }

      if (const_val->value != i) {
        tensor_is_continuous = false;
        break;
      }
    }

    // Tensor is scalar tensor
    if (offset_is_zero) {
      return false;
    }

    if (!tensor_is_continuous) {
      vectorize_tensors_.clear();
      schedule_block_can_vectorize_ = false;
      return false;
    }

    return true;
  }

  Var iter_var_;
  const int factor_;
  bool schedule_block_can_vectorize_ = true;
  std::unordered_set<std::string> vectorize_tensors_;
};

class VectorizeForTransMutator : public ir::IRMutator<ir::Expr *> {
 public:
  void operator()(ir::Expr *expr) { ir::IRMutator<>::Visit(expr, expr); }

  void Visit(const ir::Load *op, ir::Expr *expr) override {
    auto *node = expr->As<ir::Load>();
    auto *tensor = node->tensor.As<ir::_Tensor_>();
    if (in_vectorize_ && node->is_addr_tensor() &&
        tensor_can_vectorized_.count(tensor->name)) {
      TensorVectorized(node, &node->indices, false);
    }
  }

  void Visit(const ir::Store *op, ir::Expr *expr) override {
    auto *node = expr->As<ir::Store>();
    auto *tensor = node->tensor.As<ir::_Tensor_>();
    PADDLE_ENFORCE_NOT_NULL(
        tensor,
        ::common::errors::InvalidArgument(
            "Expected _Tensor_ node in Store, but received nullptr."));
    if (in_vectorize_ && node->is_addr_tensor() &&
        tensor_can_vectorized_.count(tensor->name)) {
      is_assignment_ = IsAssignment(node->value, node->type());
      TensorVectorized(node, &node->indices, true);
    }

    IRMutator::Visit(&node->value, &node->value);
  }

  // forOp don't support vectorize in adjaccnt if-block.
  void Visit(const ir::IfThenElse *op, Expr *expr) override {
    in_vectorize_ = false;
    ir::IRMutator<>::Visit(op, expr);
  }

  void Visit(const ir::For *op, ir::Expr *expr) override {
    auto *forloop = expr->As<ir::For>();
    if (op->is_vectorized()) {
      vectorize_factor_ = forloop->vectorize_info().factor;
      loop_var_ = op->loop_var;
      ScheduleBlockTensorVectorizeTeller teller(loop_var_, vectorize_factor_);
      teller.Collect(&forloop->body);
      tensor_can_vectorized_.insert(teller.GetVectorizeTensors().begin(),
                                    teller.GetVectorizeTensors().end());
      in_vectorize_ = teller.EnableVectorize();
    }

    // deal with vectorize Tensor load and store
    IRMutator::Visit(forloop, expr);

    if (in_vectorize_) {
      const int factor = forloop->vectorize_info().factor;
      PADDLE_ENFORCE_GT(factor,
                        1,
                        ::common::errors::InvalidArgument(
                            "The value of factor in SplitForLoop is incorrect."
                            "Expected value is larger than 1, but receive %d. ",
                            factor));

      auto copied_loop =
          ir::ir_utils::IRCopy(forloop, /* copy_buffer_node = */ false);
      copied_loop.As<ir::For>()->set_unrolled();
      optim::UnrollLoop(&copied_loop);
      auto unroll_body = copied_loop.As<ir::Block>()->stmts;
      auto &body_stmts = forloop->body.As<ir::Block>()->stmts;
      if (!update_cast_stmts_.empty()) {
        body_stmts.assign(update_cast_stmts_.begin(), update_cast_stmts_.end());
        update_cast_stmts_.clear();
      }

      if (!is_assignment_) {
        body_stmts.insert(
            body_stmts.end(), unroll_body.begin(), unroll_body.end());
      }

      if (!update_store_stmts_.empty()) {
        body_stmts.insert(body_stmts.end(),
                          update_store_stmts_.begin(),
                          update_store_stmts_.end());
        update_store_stmts_.clear();
      }
      *expr = forloop->body;
    }

    tensor2vectorized_vars_.clear();
    tensor_can_vectorized_.clear();
    in_vectorize_ = false;
    is_assignment_ = false;
  }

 private:
  std::string GetVectorTypeName(ir::Type type) {
    std::string name_prefix =
        cinn::common::customized_type::kcuda_builtin_vector_t;

#define GET_CUDA_VECTOR_TYPE_NAME(pred_expr, scalar_name)                 \
  if (pred_expr) {                                                        \
    return name_prefix + scalar_name + std::to_string(vectorize_factor_); \
  }
    GET_CUDA_VECTOR_TYPE_NAME(type.is_int(8), "char");
    GET_CUDA_VECTOR_TYPE_NAME(type.is_int(16), "short");
    GET_CUDA_VECTOR_TYPE_NAME(type.is_int(32), "int");
    GET_CUDA_VECTOR_TYPE_NAME(type.is_int(64), "longlong");
    GET_CUDA_VECTOR_TYPE_NAME(type.is_uint(8), "uchar");
    GET_CUDA_VECTOR_TYPE_NAME(type.is_uint(16), "ushort");
    GET_CUDA_VECTOR_TYPE_NAME(type.is_uint(32), "uint");
    GET_CUDA_VECTOR_TYPE_NAME(type.is_uint(64), "ulonglong");
    GET_CUDA_VECTOR_TYPE_NAME(type.is_float(32), "float");
    GET_CUDA_VECTOR_TYPE_NAME(type.is_float16(), "float16");
    GET_CUDA_VECTOR_TYPE_NAME(type.is_float(64), "double");
    GET_CUDA_VECTOR_TYPE_NAME(type.is_bfloat16(), "bfloat16");
#undef GET_CUDA_VECTOR_TYPE_NAME

    // others are not implemented yet
    CINN_NOT_IMPLEMENTED
    return "";
  }
  void TensorVectorized(ir::LoadStoreAddrMnger *node,
                        std::vector<ir::Expr> *indices,
                        bool is_store) {
    auto *tensor = node->tensor.As<ir::_Tensor_>();

    if (!tensor2vectorized_vars_.count(tensor->name)) {
      AppendCast(node->tensor, *indices, is_store);
    }

    if (!is_assignment_) {
      auto vectorized_var = tensor2vectorized_vars_.at(tensor->name);
      // substitute a new tensor with the vector name and dtype
      auto t = vectorized_var->type().is_cpp_handle()
                   ? node->tensor->type().PointerOf()
                   : node->tensor->type();
      node->tensor = ir::Tensor(vectorized_var->name,
                                t,
                                {ir::Expr(vectorize_factor_)},
                                {ir::Expr(vectorize_factor_)},
                                tensor->operation);
    }
    // remain the last iterative indice
    indices->assign({loop_var_});
  }

  void AppendCast(ir::Expr tensor,
                  const std::vector<ir::Expr> &indices,
                  bool is_store) {
    auto *node = tensor.As<ir::_Tensor_>();
    // generate the corresponding vector type
    Type scalar_type = tensor->type().ElementOf();
    Type vector_type_ptr(
        ir::Type::type_t::Customized, scalar_type.bits(), vectorize_factor_);
    Type vector_type(
        ir::Type::type_t::Customized, scalar_type.bits(), vectorize_factor_);
    vector_type_ptr.set_customized_type(GetVectorTypeName(scalar_type));
    vector_type_ptr.set_cpp_handle();
    vector_type_ptr.set_cpp_const(false);

    vector_type.set_customized_type(GetVectorTypeName(scalar_type));
    vector_type.set_cpp_const(false);

    // generate a local vector variable to be used in subsequent statements
    std::string vectorized_name =
        "vectorized_" + node->name + "_" + std::to_string(var_index_++);
    Var vectorized_var = ir::_Var_::Make(vectorized_name, vector_type);
    if (!is_assignment_) {
      tensor2vectorized_vars_.emplace(node->name, vectorized_var);
    }

    // generate a get_addr expr to get the address of the tensor
    Expr converted_tensor = ir::Load::Make(tensor, indices);
    cinn::ir::ir_utils::IrReplaceVarBroadcast(
        &converted_tensor, loop_var_, Expr(int32_t(0)));
    auto get_addr = ir::intrinsics::GetAddr::Make(converted_tensor);

    // generate a let expression to cast the tensor into the local vector
    auto cast = ir::Cast::Make(vector_type_ptr, get_addr);
    if (!is_store) {
      auto load = ir::Load::Make(cast, {cinn::common::make_const(0)});
      auto let = ir::Let::Make(vectorized_var, load);
      update_cast_stmts_.emplace_back(let);
    } else {
      Var vectorized_ptr =
          ir::_Var_::Make(vectorized_name + "_ptr", vector_type_ptr);
      auto let1 = ir::Let::Make(vectorized_ptr, cast);
      update_cast_stmts_.emplace_back(let1);

      auto t = ir::Tensor(vectorized_ptr->name,
                          node->type().PointerOf(),
                          {ir::Expr(vectorize_factor_)},
                          {ir::Expr(vectorize_factor_)},
                          node->operation);

      if (is_assignment_) {
        std::string load_vectorized_name = "vectorized_" +
                                           assignment_tensor_name_ + "_" +
                                           std::to_string(var_index_);
        Var load_vectorized_var =
            ir::_Var_::Make(load_vectorized_name, vector_type);
        auto store = ir::Store::Make(
            t, load_vectorized_var, {cinn::common::make_const(0)});
        update_store_stmts_.emplace_back(store);
        VLOG(5) << "Append a assignment vectorized expr:" << store;
      } else {
        auto let2 = ir::Let::Make(vectorized_var, ir::Expr(0));
        update_cast_stmts_.emplace_back(let2);

        auto t = ir::Tensor(vectorized_ptr->name,
                            node->type().PointerOf(),
                            {ir::Expr(vectorize_factor_)},
                            {ir::Expr(vectorize_factor_)},
                            node->operation);
        auto store =
            ir::Store::Make(t, vectorized_var, {cinn::common::make_const(0)});
        update_store_stmts_.emplace_back(store);
        VLOG(5) << "Append a vectorized expr:" << store;
      }
    }
  }

  // A store is considered to be a pure assignment statement only if the store
  // value is load or cast(load).
  bool IsAssignment(ir::Expr &value, const Type &store_type) {  // NOLINT
    if (auto *cast_op = value.As<ir::Cast>()) {
      return IsAssignment(cast_op->v(), store_type);
    }

    auto *load_op = value.As<ir::Load>();
    if (!load_op) {
      return false;
    }

    auto tensor_load = load_op->tensor.As<ir::_Tensor_>();
    PADDLE_ENFORCE_NOT_NULL(
        tensor_load,
        ::common::errors::InvalidArgument(
            "Expected _Tensor_ node in Store, but received nullptr."));
    Type load_type = tensor_load->type();
    if (store_type != load_type) return false;
    if (tensor_can_vectorized_.count(tensor_load->name) == 0) return false;

    is_assignment_ = true;
    assignment_tensor_name_ = tensor_load->name;
    return true;
  }

  std::vector<ir::Expr> update_cast_stmts_;
  std::vector<ir::Expr> update_store_stmts_;
  absl::flat_hash_map<std::string, ir::Var> tensor2vectorized_vars_;
  std::unordered_set<std::string> tensor_can_vectorized_;

  int vectorize_factor_{0};
  ir::Var loop_var_;
  bool in_vectorize_{false};
  int var_index_{0};
  bool is_assignment_{false};
  std::string assignment_tensor_name_;
};

}  // namespace

void VectorizeForTrans(Expr *expr) {
  ForOpWithMultiScheduleBlockSupportVectorize update;
  VLOG(5) << "before multi schedule block deal with vectorize " << *expr;
  update(expr);
  VLOG(5) << "after multi schedule block deal with vectorize " << *expr;
  VectorizeForTransMutator collector;
  VLOG(5) << "before vectorize for trans " << *expr;
  collector(expr);
  VLOG(5) << "after vectorize for trans " << *expr;
}

}  // namespace optim
}  // namespace cinn
