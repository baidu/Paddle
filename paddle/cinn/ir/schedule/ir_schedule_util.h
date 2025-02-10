// Copyright (c) 2022 CINN Authors. All Rights Reserved.
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
#include <map>
#include <random>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "paddle/cinn/common/cas.h"
#include "paddle/cinn/common/macros.h"
#include "paddle/cinn/common/shared.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/ir/ir_mutator.h"
#include "paddle/cinn/ir/op/ir_operators.h"
#include "paddle/cinn/ir/schedule/ir_schedule.h"
#include "paddle/cinn/ir/tensor.h"
#include "paddle/cinn/ir/utils/ir_copy.h"
#include "paddle/cinn/optim/replace_var_with_expr.h"
#include "paddle/cinn/utils/random_engine.h"
#include "paddle/cinn/utils/string.h"
#include "paddle/common/enforce.h"
namespace cinn {
namespace ir {
struct IterRange;
struct CacheBlockInfo;
struct CompVar;
struct CompExpr;

/**
 * \brief Given a ScheduleBlockRealize node, return the Store tensor in its
 * body.
 * @param block The given ScheduleBlockRealize node
 * @return The Store tensor in block
 */
Tensor GetTensor(const Expr& block);

/**
 * \brief Given a ScheduleBlockRealize node, return the index-th Load tensor in
 * its body.
 * @param block The given ScheduleBlockRealize node
 * @param index The index of Load tensor
 * @return The index-th Load tensor in block
 */
Tensor GetReadTensor(const Expr& block, int index);

/**
 * \brief Given a For node, return its extent as int.
 * @param loop The given For node
 * @return The extent of For node
 */
int GetLoopExtent(const Expr& loop);

/**
 * \brief Given a For node, return its extent as int.
 * @param loop The given For node
 * @return The extent of For node
 */
int GetLoopExtent(const ir::stmt::For loop);

/**
 * \brief Given a vector of Exprs, return whether they contain a var with
 * specific name.
 * @param exprs The given vector of Exprs
 * @param var_name The name of specific var
 * @return Whether there is a Var with the same name as var_name
 */
bool ContainVar(const std::vector<Expr>& exprs, const std::string& var_name);

/**
 * \brief Given a _LoweredFunc_, set its cuda_axis_info based on its func_body.
 * @param lowered_func A pointer to the given _LoweredFunc_
 */
void SetCudaAxisInfo(ir::LoweredFunc lowered_func);

/*!
 * \brief Check if a Expr node contains a ScheduleBlockRealize node.
 * \param container The container Expr node.
 * \param expr The node we want to find.
 * \return If the container contains the expr.
 */
bool Contains(const Expr& container, const Expr& expr);

/**
 * \brief Given a For loop, return the next For loop in its body.
 * @param for_loop The given For loop.
 * @return The next For loop.
 */
Expr GetNextForLoop(const Expr& for_loop);

/**
 * \brief Given two For loops, return all ir::IfThenElse nodes between them.
 * @param top The given top For loop.
 * @param bottom The given bottom For loop.
 * @return All ir::IfThenElse nodes between them.
 */
std::vector<Expr> GetIfThenElseInRange(const Expr& top, const Expr& bottom);

/**
 * Replace Vars in replaced to Exprs in candidates in source. Vars -> Exprs is
 * one-to-one correspondence.
 * @param source The Expr we will implement the change.
 * @param replaced The Vars to be replaced.
 * @param candidates The Exprs to replace Vars in replaced.
 */
void ReplaceExpr(Expr* source,
                 const std::vector<Var>& replaced,
                 const std::vector<Expr>& candidates);

/**
 * Replace Vars in replaced to Exprs in candidates in source.
 * @param source The Expr we will implement the change.
 * @param replacing_map The one-to-one corresponded Vars -> Exprs to be
 * replaced.
 */
void ReplaceExpr(Expr* source,
                 const std::map<Var, Expr, CompVar>& replacing_map);

/**
 * Validate the factors param of Split. We will check if factors are validate
 * and change -1 to positive integer.
 * @param factors The original factors.
 * @param total_extent The extent of the loop to be splitted.
 * @return return The validated factors.
 */
std::vector<int> ValidateFactors(const std::vector<int>& factors,
                                 int total_extent,
                                 const ModuleExpr& module_expr);

void CHECKRfactorValidation(const Expr& rf_loop, int rf_axis);

/**
 * Return loops that contain the expr.
 * @param expr The expr.
 * @param root The root of the whole AST.
 * @return return Loops in AST that contain the expr.
 */
std::vector<Expr> GetLoopsOfExpr(const Expr& expr, const Expr& root);

/**
 * Given an index Expr and all vars' range, return the accessed range in this
 * indice.
 * @param index The Expr of a specified indice.
 * @param iter_vars The vars in expr.
 * @param iter_range Each var's range.
 * @return return an IterRange represents the accessed range of this indice, If
 * it is not constant, return corresponding tensor's shape.
 */
IterRange GetAccessedRange(const Expr& index,
                           const std::vector<Var>& iter_vars,
                           const std::vector<IterRange>& iter_ranges);

/**
 * Given a ScheduleBlockRealize, an AST root, a tensor and its tensor_indices,
 * return the accessed buffer region of the tensor in block.
 * @param block The ScheduleBlockRealize.
 * @param tensor_indices The tensor's indices.
 * @param tensor The tensor.
 * @param root The root of whole AST.
 * @return return The accessed buffer region of the tensor in block.
 */

std::vector<IterRange> CalculateTensorRegions(
    const Expr& block,
    const std::vector<Expr>& tensor_indices,
    const Tensor& tensor,
    const Expr& root);

/**
 * Return n-th access tensor in block
 * @param block The ScheduleBlockRealize.
 * @param index The index indicating which tensor we want to get.
 * @param is_write We want to get write tensor or read tensor.
 * @return return The n-th access tensor in block. Should be ir::Store(is_write)
 * or ir::Load(!is_write).
 */
Expr GetNthAccessExpr(const Expr& block, int index, bool is_write);

/**
 * Make a tensor's cache tensor.
 * @param tensor The original tensor.
 * @param memory_type The memory type of the cache tensor.
 * @return return The tensor's cache tensor.
 */
Tensor MakeCacheTensor(const Tensor& tensor, const std::string& memory_type);

/**
 * Make a the cache tensor's block.
 * @param buffer_region The accessed region of cache tensor.
 * @param info The information of cache block.
 * @param memory_type The memory type of cache tensor.
 * @param device_api The device api of this Expr.
 * @return return ScheduleBlockRealize of the cache tensor.
 */
Expr MakeCacheBlock(const std::vector<IterRange>& buffer_ranges,
                    CacheBlockInfo* info,
                    const std::string& memory_type,
                    DeviceAPI device_api);

/**
 * Find cache tensor block's insertion point in the whole AST(root).
 * @param root The whole AST.
 * @param info The information of cache block.
 * @param is_write Are we inserting a write cache tensor or a read cache tensor.
 */
void FindInsertionPoint(const Expr& root, CacheBlockInfo* info, bool is_write);

/**
 * \brief Given a vector of For loops, return a set of them.
 * @param loops The given vector of For loops.
 * @return A set containing all the For loops in loops.
 */
const std::set<Expr, CompExpr> CollectLoopsToSet(
    const std::vector<Expr>& loops);

/**
 * \brief Given a set of For loops, return the boundary among them.
 * @param loop_set The given set of For loops.
 * @return A pair of the boundary among For loops.(The top For and bottom For)
 */
std::pair<Expr, Expr> GetBoundaryOfReorderRange(
    const std::set<Expr, CompExpr>& loop_set);

/**
 * \brief Given two For loops, return all loops between them.
 * @param top The top For loop.
 * @param bottom The bottom For loop.
 * @return A vector containing all For loops between the boundary, stored in
 * ascending order.
 */
std::vector<Expr> GetLoopsInRange(const Expr& top, const Expr& bottom);

/**
 * \brief Given params, construct a new loop.
 */
Expr ConstructNewLoopChain(const std::vector<Expr>& chain,
                           const std::vector<Expr>& ordered_loops,
                           const std::set<Expr, CompExpr>& loop_set,
                           std::vector<Expr>& if_nodes);  // NOLINT

/*!
 * \brief Find producers of block in root.
 * \param block The ScheduleBlockRealize node we want to find its producers.
 * \param root The root ScheduleBlockRealize node.
 * \return block's producers(ScheduleBlockRealize nodes) in root.
 */
std::vector<Expr> GetProducers(const Expr& block, const Expr& root);

/*!
 * \brief Find consumers of block in root.
 * \param block The ScheduleBlockRealize node we want to find its consumers.
 * \param root The root ScheduleBlockRealize node.
 * \return block's consumers(ScheduleBlockRealize nodes) in root.
 */
std::vector<Expr> GetConsumers(const Expr& block, const Expr& root);

/*!
 * \brief Check if the params of ComputeAt is validate.
 * \param block The block node we want to move in ComputeAt.
 * \param loop The for node we want to put the block under in ComputeAt.
 * \param root The root ScheduleBlockRealize node of block and loop.
 */
void CheckComputeAtValidation(const Expr& block,
                              const Expr& loop,
                              const Expr& root);

/*!
 * \brief Insert a new ScheduleBlockRealize in a loop's body(under its
 * IfThenElse Node, if any) \param for_loop The for loop whose body we want to
 * modify \param insertion The ScheduleBlockRealize we want to insert \param
 * index The position index of the for_loop body `stmts` to be inserted:
 *        - `index = -1` means inserted into the tail
 *        - otherwise, it should be a index between [0, stmts size)
 */
void InsertBlock(Expr& for_loop,  // NOLINT
                 const Expr& insertion,
                 int index = 0);  // NOLINT

/*!
 * \brief Make a union of two range. The detailed function is :
 * new_range.min = min(range1.min, range2.min)
 * new_range.extent = max(range1.min + range1.extent, range2.min +
 * range2.extent) - new_range.min Notice that the pair<Expr, Expr> indicates a
 * range's min and extent. \param range1 The first range \param range2 The
 * second range \return The union of these two ranges
 */
IterRange RangeUnion(const IterRange& range1, const IterRange& range2);

/*!
 * \brief Calculate the required buffer region given a block and its required
 * blocks. For example, if block is : B[i0, j0] = A[i0, j0] loop is : for (i, 0,
 * 64) { for (j, 0, 64) { C[i, j] = B[i, j]
 *   }
 * }
 * And required_blocks is :
 * C[i, j] = B[i, j]
 * Then we get the required B's region:
 * B[i, j], where:
 * i : [i, i]
 * j : [0, 64]
 * \param block The ScheduleBlockRealize node begin required
 * \param loop The loop where we will insert the block under it
 * @param root The root of the whole AST.
 * \param required_blocks vector of ScheduleBlockRealize nodes that require the
 * block
 * \param is_store_provided Whether Store nodes of the block provide the
 * tensor, true means it is in compute_at case, otherwise false means in
 * reverse_compute_at case
 * \return Each index's range and can_keep_loop flag of block's tensor.
 * Indicating the buffer region being required.
 */
std::vector<IterRange> CalculateRequiredRegions(
    const Expr& block,
    const Expr& loop,
    const Expr& root,
    const std::vector<Expr>& required_blocks,
    bool is_store_provided = true);

Expr CheckComputeInlineValidationAndGetStore(const Expr& schedule_block,
                                             const Expr& root);

/*!
 * \brief Check if the reverse compute inline validation passes for a given
 * schedule block and root expression, and retrieve the store expression if so.
 * Reverse compute inline validation ensures that the outputs of a loop nest are
 * properly computed in reverse order. \param schedule_block The schedule block
 * to check. \param root The root expression of the loop nest. \return A tuple
 * containing the load that will be inlined, the store that will be inlined and
 * the target store.
 */
std::tuple<Expr, Expr, Expr> CheckReverseComputeInlineValidationAndGetExprs(
    const Expr& schedule_block, const Expr& root);

/*!
 * \brief Get the prime factors of a number.
 * For example, 12 = 2^2 * 3^1, then the return value is {2: 2, 3: 1}.
 * \param n The number to be factorized.
 * \return A map of prime factors and their corresponding exponents.
 */
std::unordered_map<int, int> PrimeFactorize(int n);

/*!
 * \brief Given a number returns the form of the product of its n factors
 * For example:
 *  n = 2, dividend = 12, return one of {2, 6}, {6, 2}, {3, 4}, {4, 3}
 * \param seed The random number generator to use.
 * \param n The number to be factorized.
 * \param dividend The dividend of the number.
 */
std::vector<int> SampleTile(utils::LinearRandomEngine::StateType* rand_seed,
                            int n,
                            int dividend);

// a struct to present the min value and the extent of a iterable range,
// where it is represented as a semi-closed interval, i.e [min, min + extent)
struct IterRange {
  IterRange(Expr begin, Expr length) : min(begin), extent(length) {}

  Expr min;
  Expr extent;
};

struct CacheBlockInfo {
  /*! \brief The tensor to be read. */
  Tensor read_tensor;
  /*! \brief The tensor to be written. */
  Tensor write_tensor;
  /*! \brief The tensor allocation to be inserted into the block signature. */
  Tensor alloc;
  /*! \brief The AST node whose body is where the cache stage should be
   * inserted. */
  Expr loc_block;
  /*! \brief The index to insert the cache_read/cache_write stage. */
  int loc_pos;
  /*! \brief The cache_read/cache_write stage to be inserted. */
  Expr cache_block;
};

// The struct used to mutate new rfactor forloop and its' schedule block.
struct RfMutator : public ir::IRMutator<> {
 public:
  RfMutator(const Expr& rf_loop, const int& rf_axis)
      : rf_loop_(rf_loop), rf_axis_(rf_axis) {}
  void operator()(Expr* expr) {
    auto* rf_for = rf_loop_.As<For>();
    PADDLE_ENFORCE_NOT_NULL(
        rf_for,
        ::common::errors::InvalidArgument(
            "Failed to cast rf_loop_ to For type. The rf_loop_ "
            "object is not of type For or is null."));
    old_rf_loop_var_ = rf_for->loop_var;
    new_rf_loop_var_ = Var("rf_" + old_rf_loop_var_->name);
    IRMutator::Visit(expr, expr);
  }

  Tensor GetNewRfTensor() { return new_rf_tensor_; }

  void Visit(const ScheduleBlockRealize* op, Expr* expr) override {
    // modify iter_vars and iter_values
    auto* node = expr->As<ScheduleBlockRealize>();
    PADDLE_ENFORCE_NOT_NULL(
        node,
        ::common::errors::InvalidArgument(
            "Failed to cast expr to ScheduleBlockRealize. The expr object is "
            "not of type ScheduleBlockRealize or is null."));
    auto* schedule_block = node->schedule_block.As<ScheduleBlock>();
    PADDLE_ENFORCE_NOT_NULL(
        schedule_block,
        ::common::errors::InvalidArgument(
            "Failed to cast node->schedule_block to ScheduleBlock. The "
            "schedule_block object is not of type ScheduleBlock or is null."));
    old_output_name_ = schedule_block->name;
    find_tensor_ = false;
    auto& block_vars = schedule_block->iter_vars;
    auto& iter_values = node->iter_values;
    PADDLE_ENFORCE_EQ(old_rf_loop_var_.defined(),
                      true,
                      ::common::errors::InvalidArgument(
                          "Old RF loop variable is not defined. Please ensure "
                          "old_rf_loop_var_ is properly initialized."));

    PADDLE_ENFORCE_EQ(new_rf_loop_var_.defined(),
                      true,
                      ::common::errors::InvalidArgument(
                          "New RF loop variable is not defined. Please ensure "
                          "new_rf_loop_var_ is properly initialized."));
    PADDLE_ENFORCE_EQ(
        iter_values.size(),
        block_vars.size(),
        ::common::errors::InvalidArgument(
            "The size of iter_values and block_vars should be the same."));
    int rf_index = -1;
    for (int i = 0; i < iter_values.size(); ++i) {
      // substitute the old rfactor loop var to new rfactor loop var
      if (ContainVar({iter_values[i]}, old_rf_loop_var_->name)) {
        PADDLE_ENFORCE_EQ(
            rf_index,
            -1,
            ::common::errors::InvalidArgument(
                "only one block var can bind the rfactor loop var"));
        PADDLE_ENFORCE_NOT_NULL(
            iter_values[i].As<_Var_>(),
            ::common::errors::NotFound(
                "Rfactor loop var not support composite bindings. The variable "
                "at index %d cannot be cast to the expected type.",
                i));
        rf_index = i;
        optim::ReplaceVarWithExpr(
            &iter_values[i], old_rf_loop_var_, new_rf_loop_var_);
        new_rf_itervar_ = block_vars[i];
      }
    }
    // create new rfactor block var if not exist
    if (rf_index == -1) {
      new_rf_itervar_ =
          Var(cinn::UniqName("i" + std::to_string(block_vars.size())));
      iter_values.push_back(new_rf_loop_var_);
      block_vars.push_back(new_rf_itervar_);
    }
    IRMutator::Visit(&node->schedule_block, &node->schedule_block);
    PADDLE_ENFORCE_EQ(
        find_tensor_,
        true,
        ::common::errors::InvalidArgument(
            "Can not find the store tensor with the schedule block name %s.",
            old_output_name_.c_str()));
    schedule_block->name = "rf_" + old_output_name_;
  }

  void Visit(const Load* op, Expr* expr) override {
    // insert the new rfactor indice if not exist
    auto* node = expr->As<Load>();
    PADDLE_ENFORCE_NOT_NULL(
        node,
        ::common::errors::NotFound(
            "The expression could not be cast to a Load node."));
    auto* tensor = node->tensor.As<_Tensor_>();
    PADDLE_ENFORCE_NOT_NULL(tensor,
                            ::common::errors::NotFound(
                                "The tensor in the Load node could not be cast "
                                "to the expected _Tensor_ type."));
    if (tensor->name == "rf_" + old_output_name_) {
      int size = node->indices.size();
      PADDLE_ENFORCE_LE(rf_axis_,
                        size,
                        ::common::errors::InvalidArgument(
                            "rf_axis should not be greater than indice size"));
      PADDLE_ENFORCE_EQ(
          new_rf_itervar_.defined(),
          true,
          ::common::errors::InvalidArgument(
              "The new rfactor iteration variable must be defined."));
      PADDLE_ENFORCE_EQ(!ContainVar(node->indices, new_rf_itervar_->name),
                        true,
                        ::common::errors::InvalidArgument(
                            "Original output tensor %s should not "
                            "have the new rfactor index %s.",
                            old_output_name_.c_str(),
                            new_rf_itervar_->name.c_str()));
      node->indices.insert(node->indices.begin() + rf_axis_, new_rf_itervar_);
    }
  }

  void Visit(const Store* op, Expr* expr) override {
    // insert the new rfactor indice if not exist
    auto* node = expr->As<Store>();
    PADDLE_ENFORCE_NOT_NULL(
        node,
        ::common::errors::NotFound(
            "The expression could not be cast to a Store node."));
    auto* tensor = node->tensor.As<_Tensor_>();
    PADDLE_ENFORCE_NOT_NULL(
        tensor,
        ::common::errors::NotFound(
            "The tensor in the Store node could not be cast "
            "to the expected _Tensor_ type."));
    if (tensor->name == old_output_name_) {
      find_tensor_ = true;
      tensor->name = "rf_" + tensor->name;
      int size = node->indices.size();
      PADDLE_ENFORCE_LE(rf_axis_,
                        size,
                        ::common::errors::InvalidArgument(
                            "rf_axis should not be greater than indice size"));
      PADDLE_ENFORCE_EQ(!ContainVar(node->indices, new_rf_itervar_->name),
                        true,
                        ::common::errors::InvalidArgument(
                            "Original output tensor %s should not "
                            "have the new rfactor index %s.",
                            old_output_name_.c_str(),
                            new_rf_itervar_->name.c_str()));
      node->indices.insert(node->indices.begin() + rf_axis_, new_rf_itervar_);
      auto* rf_for = rf_loop_.As<For>();
      PADDLE_ENFORCE_NOT_NULL(
          rf_for,
          ::common::errors::InvalidArgument(
              "Failed to cast rf_loop_ to For type. The rf_loop_ "
              "object is not of type For or is null."));
      PADDLE_ENFORCE_EQ(
          is_zero(rf_for->min),
          true,
          ::common::errors::InvalidArgument(
              "The rfactor loop's minimum value should be zero."));
      auto extent = optim::ArithSimplify(rf_for->extent);
      auto& shape = tensor->shape;
      auto& domain = tensor->domain;
      PADDLE_ENFORCE_LE(
          rf_axis_,
          shape.size(),
          ::common::errors::InvalidArgument(
              "rf_axis should not be greater than tensor shape size"));
      PADDLE_ENFORCE_LE(
          rf_axis_,
          domain.size(),
          ::common::errors::InvalidArgument(
              "rf_axis should not be greater than tensor domain size"));
      shape.insert(shape.begin() + rf_axis_, extent);
      domain.insert(domain.begin() + rf_axis_, extent);
      if (tensor->buffer.defined()) {
        if (tensor->buffer->name.find_first_of("rf") == std::string::npos) {
          tensor->buffer->name = "rf_" + tensor->buffer->name;
          tensor->buffer->shape = shape;
        }
      }
      new_rf_tensor_ = Tensor(tensor);
    }
    IRMutator::Visit(&node->value, &node->value);
  }

  void Visit(const For* op, Expr* expr) override {
    auto* node = expr->As<For>();
    PADDLE_ENFORCE_NOT_NULL(
        node,
        ::common::errors::NotFound(
            "The expression could not be cast to a For node."));
    depth++;
    auto* rf_for = rf_loop_.As<For>();
    PADDLE_ENFORCE_NOT_NULL(
        rf_for,
        ::common::errors::InvalidArgument(
            "Failed to cast rf_loop_ to For type. The rf_loop_ "
            "object is not of type For or is null."));
    // erase the original rfactor forloop
    if (node->loop_var->name == old_rf_loop_var_->name) {
      auto body = node->body.As<Block>();
      if (body && body->stmts.size() == 1) {
        *expr = body->stmts[0];
      } else {
        *expr = node->body;
      }
      IRMutator::Visit(expr, expr);
    } else {
      IRMutator::Visit(&node->body, &node->body);
    }
    if (rf_axis_ == 0 && depth == rf_axis_) {
      // insert new rfactor forloop in the rf_axis as serial loop
      *expr = For::Make(new_rf_loop_var_,
                        rf_for->min,
                        rf_for->extent,
                        ForType::Serial,
                        rf_for->device_api,
                        Block::Make({*expr}));
    } else if (depth == rf_axis_ - 1) {
      // insert new rfactor forloop in the rf_axis as serial loop
      node->body = Block::Make({For::Make(new_rf_loop_var_,
                                          rf_for->min,
                                          rf_for->extent,
                                          ForType::Serial,
                                          rf_for->device_api,
                                          node->body)});
    }
    depth--;
  }

 private:
  Expr rf_loop_;
  Var old_rf_loop_var_;
  Var new_rf_loop_var_;
  int rf_axis_;
  int depth = -1;
  bool find_tensor_ = false;
  std::string old_output_name_;
  Var new_rf_itervar_;
  Tensor new_rf_tensor_;
};

// The struct used to reconstruct the new For node to replace the old For node.
struct LoopReconstructor : public ir::IRMutator<> {
 public:
  explicit LoopReconstructor(const Expr& root,
                             const Expr& block,
                             const Expr& loop)
      : root_(root), block_(block), loop_(loop) {}

  void operator()(Expr* expr) { IRMutator::Visit(expr, expr); }

  /* \param inserted_pos The position index of the new_loop_ body `stmts` to be
   * inserted:
   *        - `index = -1` means inserted into the tail
   *        - otherwise, it should be a index between [0, stmts size)
   */
  std::string MakeNewLoop(const std::vector<IterRange>& iter_ranges,
                          bool keep_unit_loops,
                          int inserted_pos = -1) {
    int n_iters = iter_ranges.size();
    std::vector<Var> loop_vars;
    std::vector<Expr> loop_extents;
    std::vector<Expr> iter_values;
    loop_vars.reserve(n_iters);
    loop_extents.reserve(n_iters);
    iter_values.reserve(n_iters);
    std::vector<std::string> new_var_names;
    for (int i = 0; i < n_iters; ++i) {
      const auto& range = iter_ranges[i];
      if (keep_unit_loops || range.extent != Expr(1)) {
        std::string var_name =
            cinn::common::UniqName("ax" + std::to_string(loop_vars.size()));
        new_var_names.push_back(var_name);
        Var var(var_name, Int(32));
        loop_vars.push_back(var);
        loop_extents.push_back(range.extent);
        iter_values.push_back(optim::ArithSimplify(range.min) + var);
      } else {
        iter_values.push_back(optim::ArithSimplify(range.min));
      }
    }
    auto schedule_block_node =
        block_.As<ir::ScheduleBlockRealize>()->schedule_block;
    new_block_ = ScheduleBlockRealize::Make(std::move(iter_values),
                                            std::move(schedule_block_node));
    Expr loop_body = new_block_;
    for (int i = static_cast<int>(loop_vars.size()) - 1; i >= 0; --i) {
      auto loop_var = loop_vars[i];
      auto loop_extent = loop_extents[i];
      if (!loop_body.As<ir::Block>()) loop_body = Block::Make({loop_body});
      loop_body = For::Make(loop_var,
                            Expr(0),
                            loop_extent,
                            ForType::Serial,
                            loop_.As<ir::For>()->device_api,
                            std::move(loop_body));
    }
    new_loop_ = ir::ir_utils::IRCopy(loop_);

    // Replace the copied Tensor object with the original Tensor object,
    // to ensure that the same Tensor in a AST is the same object.
    std::unordered_map<std::string, ir::Expr> tensors_map;
    ir::ir_utils::CollectIRNodesWithoutTensor(
        loop_, [&tensors_map](const Expr* x) {
          if (x->as_tensor()) {
            tensors_map.insert({x->as_tensor()->name, *x});
            return true;
          }
          return false;
        });
    auto find_store = ir::ir_utils::CollectIRNodesWithoutTensor(
        new_loop_, [](const Expr* x) { return x->As<ir::Store>(); });
    for (auto store : find_store) {
      store.As<ir::Store>()->tensor =
          tensors_map.at(store.As<ir::Store>()->tensor.as_tensor()->name);
    }
    auto find_load = ir::ir_utils::CollectIRNodesWithoutTensor(
        new_loop_, [](const Expr* x) { return x->As<ir::Load>(); });
    for (auto load : find_load) {
      load.As<ir::Load>()->tensor =
          tensors_map.at(load.As<ir::Load>()->tensor.as_tensor()->name);
    }

    InsertBlock(new_loop_, loop_body, inserted_pos);
    return utils::Join(new_var_names, ",");
  }

 public:
  /*! \brief The root block */
  Expr root_;
  /*! \brief The given block to be moved */
  Expr block_;
  /*! \brief The given loop the block and its loop nest to be put under */
  Expr loop_;
  /*! \brief The new loop to replace the original loop */
  Expr new_loop_{nullptr};
  /*! \brief The new block realize to the moved block */
  Expr new_block_{nullptr};
  /*! \brief The plan to remove the given block by replacing this loop/block in
   * the AST */
  Expr source_expr{nullptr};
  /*! \brief The plan to remove the given block by replacing to this loop/block
   * in the AST */
  Expr target_expr{nullptr};
};

// The struct used to mutate final write-back forloop and schedule block.
struct FinalMutator : public ir::IRMutator<> {
 public:
  FinalMutator(const Expr& rf_loop,
               const int& rf_axis,
               const Tensor& new_rf_tensor)
      : rf_loop_(rf_loop), rf_axis_(rf_axis), new_rf_tensor_(new_rf_tensor) {}
  void operator()(Expr* expr) {
    auto* rf_for = rf_loop_.As<For>();
    PADDLE_ENFORCE_NOT_NULL(
        rf_for,
        ::common::errors::InvalidArgument(
            "Failed to cast rf_loop_ to For type. The rf_loop_ "
            "object is not of type For or is null."));
    old_rf_loop_var_ = rf_for->loop_var;
    IRMutator::Visit(expr, expr);
  }

  void Visit(const ScheduleBlockRealize* op, Expr* expr) override {
    auto* node = expr->As<ScheduleBlockRealize>();
    PADDLE_ENFORCE_NOT_NULL(
        node,
        ::common::errors::NotFound(
            "The ScheduleBlockRealize node cannot be null."));
    auto* schedule_block = node->schedule_block.As<ScheduleBlock>();
    PADDLE_ENFORCE_NOT_NULL(schedule_block,
                            ::common::errors::NotFound(
                                "The ScheduleBlock pointer cannot be null."));
    auto& iter_vars = schedule_block->iter_vars;
    auto& iter_values = node->iter_values;
    output_name_ = schedule_block->name;
    visit_init_block_ = output_name_.rfind("_init") != std::string::npos;
    if (!visit_init_block_) {
      for (int i = 0; i < iter_values.size(); ++i) {
        if (ContainVar({iter_values[i]}, old_rf_loop_var_->name)) {
          // record the rfactor loop var's block var
          PADDLE_ENFORCE_NOT_NULL(iter_values[i].As<_Var_>(),
                                  ::common::errors::NotFound(
                                      "Can not support complex reduce bindings "
                                      "of iteration values at position %d.",
                                      i));
          old_rf_iter_var_ = iter_vars[i];
          break;
        }
      }
    }
    IRMutator::Visit(&node->schedule_block, &node->schedule_block);
    // modify iter_vars and iter_values, erase other reduce block vars and
    // values
    for (auto it = iter_values.begin(); it != iter_values.end(); ++it) {
      for (auto erase_var : erase_reduce_loopvars_) {
        if (ContainVar({*it}, erase_var)) {
          PADDLE_ENFORCE_NOT_NULL(
              (*it).As<_Var_>(),
              ::common::errors::NotFound(
                  "Not support complex reduce bindings: %s", *it));
          iter_vars.erase(it - iter_values.begin() + iter_vars.begin());
          iter_values.erase(it);
          --it;
          break;
        }
      }
    }
  }

  // currently only support reduce_sum, reduce_mul, reduce_min and reduce_max
  void Visit(const Add* op, Expr* expr) override {
    auto* node = expr->As<Add>();
    PADDLE_ENFORCE_NOT_NULL(
        node,
        ::common::errors::NotFound(
            "The expression could not be cast to an Add node."));
    auto& oper_b = node->b();
    oper_b = Load::Make(new_rf_tensor_, new_rf_indice_);
  }

  void Visit(const Mul* op, Expr* expr) override {
    auto* node = expr->As<Mul>();
    PADDLE_ENFORCE_NOT_NULL(
        node,
        ::common::errors::NotFound(
            "The expression could not be cast to a Mul node."));
    auto& oper_b = node->b();
    oper_b = Load::Make(new_rf_tensor_, new_rf_indice_);
  }

  void Visit(const Min* op, Expr* expr) override {
    auto* node = expr->As<Min>();
    PADDLE_ENFORCE_NOT_NULL(
        node,
        ::common::errors::NotFound(
            "The expression could not be cast to a Min node."));
    auto& oper_b = node->b();
    oper_b = Load::Make(new_rf_tensor_, new_rf_indice_);
  }

  void Visit(const Max* op, Expr* expr) override {
    auto* node = expr->As<Max>();
    PADDLE_ENFORCE_NOT_NULL(
        node,
        ::common::errors::NotFound(
            "The expression could not be cast to a Max node."));
    auto& oper_b = node->b();
    oper_b = Load::Make(new_rf_tensor_, new_rf_indice_);
  }

  void Visit(const Store* op, Expr* expr) override {
    // insert the new rfactor indice if not exist
    auto* node = expr->As<Store>();
    PADDLE_ENFORCE_NOT_NULL(
        node,
        ::common::errors::NotFound(
            "The expression could not be cast to a Store node."));
    auto* tensor = node->tensor.As<_Tensor_>();
    PADDLE_ENFORCE_NOT_NULL(
        tensor,
        ::common::errors::NotFound(
            "The tensor in the Store node could not be cast "
            "to the expected _Tensor_ type."));
    PADDLE_ENFORCE_EQ(
        tensor->name,
        output_name_,
        ::common::errors::InvalidArgument(
            "store name should be same with the schedule block name"));
    if (!visit_init_block_) {
      new_rf_indice_ = node->indices;
      PADDLE_ENFORCE_LE(
          rf_axis_,
          new_rf_indice_.size(),
          ::common::errors::InvalidArgument(
              "rf_axis_ should not be greater than tensor indice size"));
      PADDLE_ENFORCE_EQ(
          old_rf_iter_var_.defined(),
          true,
          ::common::errors::InvalidArgument(
              "Old RF iter variable is not defined. Please ensure "
              "old_rf_iter_var_ is properly initialized."));
      new_rf_indice_.insert(new_rf_indice_.begin() + rf_axis_,
                            old_rf_iter_var_);
      IRMutator::Visit(&node->value, &node->value);
    }
  }

  void Visit(const For* op, Expr* expr) override {
    auto* node = expr->As<For>();
    PADDLE_ENFORCE_NOT_NULL(
        node,
        ::common::errors::NotFound(
            "The expression could not be cast to a For node."));
    auto* rf_for = rf_loop_.As<For>();
    // erase the reduce forloops after the init block except the rfactor loop
    if (visit_init_block_ && node->loop_var->name != old_rf_loop_var_->name) {
      erase_reduce_loopvars_.insert(node->loop_var->name);
      auto body = node->body.As<Block>();
      if (body && body->stmts.size() == 1) {
        *expr = body->stmts[0];
      } else {
        *expr = node->body;
      }
      IRMutator::Visit(expr, expr);
    } else {
      IRMutator::Visit(&node->body, &node->body);
    }
  }

 private:
  Expr rf_loop_;
  int rf_axis_;
  Var old_rf_loop_var_;
  Var old_rf_iter_var_;
  std::string output_name_;
  // collect reduce loop vars except rfactor loop var
  std::set<std::string> erase_reduce_loopvars_;
  bool visit_init_block_ = false;
  Tensor new_rf_tensor_;
  std::vector<Expr> new_rf_indice_;
};

//! Visit all ScheduleBlock and change its body to ir::Block if it is not.
struct ChangeBodyToBlock : public ir::IRMutator<> {
 public:
  static void Change(Expr* expr) {
    ChangeBodyToBlock mutator;
    mutator(expr);
  }

  void operator()(Expr* expr) { IRMutator::Visit(expr, expr); }

 private:
  void Visit(const ir::ScheduleBlock* expr, Expr* op) override {
    if (!op->As<ScheduleBlock>()->body.As<Block>()) {
      op->As<ScheduleBlock>()->body =
          Block::Make({op->As<ScheduleBlock>()->body});
    }
    IRMutator::Visit(expr, op);
  }
};

struct CacheWriteRewriter : public ir::IRMutator<> {
 public:
  static Expr Rewrite(const Expr& root, CacheBlockInfo* info) {
    CacheWriteRewriter rewriter(root, info);
    Expr new_root = ir::ir_utils::IRCopy(root);
    rewriter.mutate_cache_block = true;
    rewriter(&info->cache_block);
    rewriter.mutate_cache_block = false;
    rewriter(&new_root);
    auto find_tensor = ir::ir_utils::CollectIRNodesWithoutTensor(
        new_root,
        [&](const Expr* x) {
          return x->As<Store>() &&
                 (x->As<Store>()->tensor == Expr(info->read_tensor));
        },
        true);
    if (!find_tensor.empty()) {
      auto find_store = ir::ir_utils::CollectIRNodesWithoutTensor(
          (*find_tensor.begin()), [&](const Expr* x) {
            return x->As<Load>() &&
                   (x->As<Load>()->tensor == Expr(info->write_tensor));
          });
      for (auto load_ir : find_store) {
        load_ir.As<Load>()->tensor = Expr(info->read_tensor);
      }
    }
    return new_root;
  }

  void operator()(Expr* expr) { IRMutator::Visit(expr, expr); }

 private:
  explicit CacheWriteRewriter(const Expr& root, CacheBlockInfo* info)
      : root_(root), info_(info) {}

  void Visit(const ir::Block* expr, Expr* op) override {
    if (*op == info_->loc_block) {
      IRMutator::Visit(expr, op);
      op->As<Block>()->stmts.insert(
          op->As<Block>()->stmts.begin() + info_->loc_pos, info_->cache_block);
    } else {
      IRMutator::Visit(expr, op);
    }
  }

  void Visit(const ir::ScheduleBlock* expr, Expr* op) override {
    if (op->As<ScheduleBlock>()->name == info_->write_tensor->name) {
      op->As<ScheduleBlock>()->name = info_->read_tensor->name;
    } else if (op->As<ScheduleBlock>()->name == info_->read_tensor->name) {
      op->As<ScheduleBlock>()->name = info_->write_tensor->name;
    }
    IRMutator::Visit(expr, op);
  }

  void Visit(const ir::Load* expr, Expr* op) override {
    IRMutator::Visit(expr, op);
    if (op->As<Load>()->tensor == Expr(info_->write_tensor) &&
        mutate_cache_block) {
      op->As<Load>()->tensor = Expr(info_->read_tensor);
    } else if (op->As<Load>()->tensor == Expr(info_->read_tensor) &&
               mutate_cache_block) {
      op->As<Load>()->tensor = Expr(info_->write_tensor);
    }
  }

  void Visit(const ir::Store* expr, Expr* op) override {
    IRMutator::Visit(expr, op);
    if (op->As<Store>()->tensor == Expr(info_->write_tensor)) {
      op->As<Store>()->tensor = Expr(info_->read_tensor);
    } else if (op->As<Store>()->tensor == Expr(info_->read_tensor) &&
               mutate_cache_block) {
      op->As<Store>()->tensor = Expr(info_->write_tensor);
    }
  }

 private:
  /*! \brief The parent scope of the insertion */
  const Expr& root_;
  /*! \brief The info for inserting cache stage */
  CacheBlockInfo* info_;
  /*! \brief Are we mutating the cache tensor's block */
  bool mutate_cache_block{true};
};

struct FixLocalBufferSize : public ir::IRMutator<> {
 public:
  explicit FixLocalBufferSize(const std::string& tensor_name)
      : tensor_name_(tensor_name) {}

  void operator()(Expr* expr) { IRMutator::Visit(expr, expr); }

 private:
  void Visit(const ir::Store* expr, Expr* op) override {
    if (op->As<Store>()->tensor.As<_Tensor_>()->name == tensor_name_) {
      op->As<Store>()->tensor.As<_Tensor_>()->shape = {Expr(1)};
      op->As<Store>()->tensor.As<_Tensor_>()->domain = {Expr(1)};
      op->As<Store>()->tensor.As<_Tensor_>()->buffer->shape = {Expr(1)};
      op->As<Store>()->indices = {Expr(0)};
    }
    IRMutator::Visit(expr, op);
  }

  void Visit(const ir::Load* expr, Expr* op) override {
    if (op->As<Load>()->tensor.As<_Tensor_>()->name == tensor_name_) {
      op->As<Load>()->tensor.As<_Tensor_>()->shape = {Expr(1)};
      op->As<Load>()->tensor.As<_Tensor_>()->domain = {Expr(1)};
      op->As<Load>()->tensor.As<_Tensor_>()->buffer->shape = {Expr(1)};
      op->As<Load>()->indices = {Expr(0)};
    }
    IRMutator::Visit(expr, op);
  }
  std::string tensor_name_;
};

struct InsertExpr : public ir::IRMutator<> {
 public:
  static void Insert(const Expr& ir_node,
                     const Expr& insert_node,
                     bool after_node,
                     Expr* expr) {
    InsertExpr mutator(ir_node, insert_node, after_node);
    mutator(expr);
  }

  void operator()(Expr* expr) { IRMutator::Visit(expr, expr); }

 private:
  explicit InsertExpr(const Expr& ir_node,
                      const Expr& insert_node,
                      bool after_node)
      : ir_node_(ir_node), insert_node_(insert_node), after_node_(after_node) {}

  void Visit(const ir::Block* expr, Expr* op) override {
    for (int i = 0; i < expr->stmts.size(); i++) {
      if (expr->stmts[i] == ir_node_) {
        if (after_node_) {
          op->As<ir::Block>()->stmts.insert(
              op->As<ir::Block>()->stmts.begin() + i + 1, insert_node_);
        } else {
          op->As<ir::Block>()->stmts.insert(
              op->As<ir::Block>()->stmts.begin() + i, insert_node_);
        }
        return;
      }
    }
    IRMutator::Visit(expr, op);
  }

  void Visit(const ir::For* expr, Expr* op) override {
    if (expr->body == ir_node_) {
      if (after_node_)
        op->As<ir::For>()->body =
            ir::Block::Make({op->As<ir::For>()->body, insert_node_});
      else
        op->As<ir::For>()->body =
            ir::Block::Make({insert_node_, op->As<ir::For>()->body});
      return;
    }
    IRMutator::Visit(expr, op);
  }

 private:
  const Expr& ir_node_;
  const Expr& insert_node_;
  bool after_node_;
};

// Self-defined operator to support std::set<Expr>
struct CompExpr {
  bool operator()(const Expr& left, const Expr& right) const {
    return utils::GetStreamCnt(left) < utils::GetStreamCnt(right);
  }
};

// Self-defined operator to support std::set<Var>
struct CompVar {
  bool operator()(const Var& left, const Var& right) const {
    return left->name < right->name;
  }
};

struct MappingVarToExprMutator : public ir::IRMutator<> {
  explicit MappingVarToExprMutator(
      const std::map<Var, Expr, CompVar>& replacing_map)
      : replacing_map_(replacing_map) {}

  void operator()(Expr* expr) { IRMutator::Visit(expr, expr); }

 private:
  void Visit(const ir::_Var_* expr, Expr* op) override {
    if (replacing_map_.count(op->as_var_ref())) {
      *op = replacing_map_.at(op->as_var_ref());
    }
  }

 private:
  const std::map<Var, Expr, CompVar>& replacing_map_;
};

struct FindBlocksVisitor {
  explicit FindBlocksVisitor(const std::string& block_name = "")
      : block_name_(block_name) {}

  std::vector<Expr> operator()(const Expr* expr) {
    Visit(expr);
    return result;
  }

 private:
  void Visit(const Expr* expr) {
    if (!expr->defined()) return;
    if (!block_name_.empty() && !result.empty()) return;
    if (expr->As<ir::For>()) {
      Visit(&(expr->As<ir::For>()->body));
    } else if (expr->As<ir::ScheduleBlockRealize>()) {
      if (expr->As<ir::ScheduleBlockRealize>()
              ->schedule_block.As<ScheduleBlock>()
              ->name.substr(0, 4) != "root") {
        auto* schedule_block = expr->As<ir::ScheduleBlockRealize>()
                                   ->schedule_block.As<ir::ScheduleBlock>();
        if (block_name_.empty() || schedule_block->name == block_name_) {
          result.emplace_back(*expr);
        }
      } else {
        Visit(&(expr->As<ir::ScheduleBlockRealize>()->schedule_block));
      }
    } else if (expr->As<ir::ScheduleBlock>()) {
      Visit(&(expr->As<ir::ScheduleBlock>()->body));
    } else if (expr->As<ir::Block>()) {
      for (auto& n : expr->As<ir::Block>()->stmts) Visit(&n);
    } else if (expr->As<ir::IfThenElse>()) {
      Visit(&(expr->As<ir::IfThenElse>()->true_case));
      Visit(&(expr->As<ir::IfThenElse>()->false_case));
    }
  }
  std::string block_name_;
  std::vector<Expr> result{};
};

struct FindLoopsVisitor {
  explicit FindLoopsVisitor(const Expr& block) : block_(block) {}

  std::vector<Expr> operator()(const Expr* expr) {
    PADDLE_ENFORCE_NOT_NULL(
        block_.As<ir::ScheduleBlockRealize>(),
        ::common::errors::NotFound("The ScheduleBlockRealize cannot be null."));
    visit_end = false;
    Visit(expr);
    return result;
  }

 private:
  void Visit(const Expr* expr) {
    if (visit_end || !expr->defined()) return;
    if (expr->As<ir::For>()) {
      father_loops.emplace_back(*expr);
      Visit(&(expr->As<ir::For>()->body));
      father_loops.pop_back();
    } else if (expr->As<ir::ScheduleBlockRealize>()) {
      if (*expr == block_) {
        result = father_loops;
        visit_end = true;
        return;
      } else {
        Visit(&(expr->As<ir::ScheduleBlockRealize>()->schedule_block));
      }
    } else if (expr->As<ir::ScheduleBlock>()) {
      Visit(&(expr->As<ir::ScheduleBlock>()->body));
    } else if (expr->As<ir::Block>()) {
      for (auto& n : expr->As<ir::Block>()->stmts) Visit(&n);
    } else if (expr->As<ir::IfThenElse>()) {
      Visit(&(expr->As<ir::IfThenElse>()->true_case));
      Visit(&(expr->As<ir::IfThenElse>()->false_case));
    }
  }

  std::vector<Expr> father_loops{};
  std::vector<Expr> result{};
  bool visit_end{false};
  const Expr& block_;
};

struct FindBlockParent : public ir::IRMutator<> {
 public:
  explicit FindBlockParent(const std::string& block_name)
      : block_name_(block_name) {}

  void operator()(Expr* expr) { IRMutator::Visit(expr, expr); }

 private:
  void Visit(const ir::Block* expr, Expr* op) override {
    if (target_) return;
    for (auto& stmt : expr->stmts) {
      if (stmt.As<ir::ScheduleBlockRealize>()) {
        if (stmt.As<ir::ScheduleBlockRealize>()
                ->schedule_block.As<ir::ScheduleBlock>()
                ->name == block_name_) {
          target_ = op;
          return;
        }
      }
    }
    IRMutator::Visit(expr, op);
  }

  void Visit(const ir::For* expr, Expr* op) override {
    if (target_) return;
    if (expr->body.As<ir::ScheduleBlockRealize>()) {
      if (expr->body.As<ir::ScheduleBlockRealize>()
              ->schedule_block.As<ir::ScheduleBlock>()
              ->name == block_name_) {
        target_ = op;
        return;
      }
    }
    IRMutator::Visit(expr, op);
  }

  void Visit(const ir::ScheduleBlock* expr, Expr* op) override {
    if (target_) return;
    if (expr->body.As<ir::ScheduleBlockRealize>()) {
      if (expr->body.As<ir::ScheduleBlockRealize>()
              ->schedule_block.As<ir::ScheduleBlock>()
              ->name == block_name_) {
        target_ = op;
        return;
      }
    }
    IRMutator::Visit(expr, op);
  }

  std::string block_name_;

 public:
  ir::Expr* target_{nullptr};
};

// The struct used to create all stmts after rfactor transformation.
struct RfCreator : public ir::IRMutator<> {
 public:
  RfCreator(const Expr& root, const Expr& rf_loop, const int& rf_axis)
      : root_(root), rf_loop_(rf_loop), rf_axis_(rf_axis) {}
  void operator()(Expr* expr) { IRMutator::Visit(expr, expr); }

  Expr CreateRfAllStmts() {
    auto root_realize = root_.As<ScheduleBlockRealize>();
    PADDLE_ENFORCE_NOT_NULL(
        root_realize,
        ::common::errors::NotFound(
            "The ScheduleBlockRealize node cannot be null."));
    auto root_block = root_realize->schedule_block.As<ScheduleBlock>();
    PADDLE_ENFORCE_NOT_NULL(
        root_block,
        ::common::errors::NotFound(
            "The ScheduleBlock of ScheduleBlockRealize cannot be null."));
    Expr root_loop = ir::ir_utils::IRCopy(root_block->body);
    if (auto block = root_loop.As<Block>()) {
      PADDLE_ENFORCE_EQ(block->stmts.size(),
                        1U,
                        ::common::errors::InvalidArgument(
                            "rfactor root should only have one block stmt"));
      root_loop = block->stmts[0];
    }
    auto* root_for = root_loop.As<For>();
    PADDLE_ENFORCE_NOT_NULL(
        root_for,
        ::common::errors::InvalidArgument(
            "Failed to cast rf_loop_ to For type. The rf_loop_ "
            "object is not of type For or is null."));
    auto rf_for = rf_loop_.As<For>();
    PADDLE_ENFORCE_NOT_NULL(
        rf_for,
        ::common::errors::InvalidArgument(
            "Failed to cast rf_loop_ to For type. The rf_loop_ "
            "object is not of type For or is null."));
    // create new rfactor forloops
    Expr new_rf_forloop = ir::ir_utils::IRCopy(root_loop);
    RfMutator rf_mutator(rf_loop_, rf_axis_);
    rf_mutator(&new_rf_forloop);
    VLOG(3) << "After RfMutator, new rf_forloop is\n" << new_rf_forloop;
    auto new_rf_tensor = rf_mutator.GetNewRfTensor();
    // create final write-back forloops
    Expr final_forloop = ir::ir_utils::IRCopy(root_loop);
    FinalMutator final_mutator(rf_loop_, rf_axis_, new_rf_tensor);
    final_mutator(&final_forloop);
    VLOG(3) << "After FinalMuator, final write-back forloop is\n"
            << final_forloop;
    // combine the new created rfactor forloops with the final write-back
    // forloops and replace
    root_block->body = Block::Make({new_rf_forloop, final_forloop});
    return new_rf_tensor;
  }

  Expr root_;
  Expr rf_loop_;
  int rf_axis_;
};

// Judge the `For` node of input `expr` hold dynamic `extent` or not.
bool ContainDynamicShape(const Expr& expr);

}  // namespace ir
}  // namespace cinn
