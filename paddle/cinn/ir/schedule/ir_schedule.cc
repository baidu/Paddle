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

#include "paddle/cinn/ir/schedule/ir_schedule.h"

#include <math.h>

#include <algorithm>
#include <iostream>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/cinn/common/cas.h"
#include "paddle/cinn/common/common.h"
#include "paddle/cinn/common/dev_info_manager.h"
#include "paddle/cinn/common/ir_util.h"
#include "paddle/cinn/common/target.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_analyzer/ir_analyzer.h"
#include "paddle/cinn/ir/ir_mutator.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/ir_visitor.h"
#include "paddle/cinn/ir/op/ir_operators.h"
#include "paddle/cinn/ir/schedule/impl/ir_schedule.h"
#include "paddle/cinn/ir/schedule/ir_schedule_util.h"
#include "paddle/cinn/ir/utils/ir_copy.h"
#include "paddle/cinn/lang/compute.h"
#include "paddle/cinn/optim/replace_var_with_expr.h"
#include "paddle/cinn/utils/string.h"
#include "paddle/common/enforce.h"
PD_DECLARE_int32(cinn_error_message_level);

namespace cinn {
namespace ir {

std::unique_ptr<ScheduleBase> ScheduleBase::Make(
    const ModuleExpr& module_expr,
    bool debug_flag,
    utils::ErrorMessageLevel err_msg_level,
    bool is_dynamic) {
  return std::make_unique<DyScheduleImpl>(
      module_expr, debug_flag, err_msg_level);
}

std::unique_ptr<ScheduleBase> ScheduleBase::Make(ModuleExpr&& module_expr,
                                                 bool is_dynamic) {
  return std::make_unique<DyScheduleImpl>(std::move(module_expr));
}

/** \brief A macro that guards the beginning of each implementation of schedule
 */
#define CINN_IR_SCHEDULE_BEGIN() try {
/**
 * \brief A macro that pairs with `CINN_IR_SCHEDULE_BEGIN`, handling potential
 * errors and error message printing.
 * @param primitive A string representing the kind of schedule primitive.
 * @param err_msg_level A ScheduleErrorMessageLevel enum, level of error message
 * printing
 */
#define CINN_IR_SCHEDULE_END(err_msg_level)              \
  }                                                      \
  catch (const utils::ErrorHandler& err_handler) {       \
    PADDLE_THROW(::common::errors::Fatal(                \
        err_handler.FormatErrorMessage(err_msg_level))); \
  }

void BaseInliner::operator()(Expr* expr) {
  IRMutator::Visit(&tgt_stmt, &tgt_stmt);
  IRMutator::Visit(expr, expr);
}

void BaseInliner::Visit(const ir::Block* expr, Expr* op) {
  if (*op == src_stmt) {
    *op = tgt_stmt;
    return;
  }
  IRMutator::Visit(expr, op);
}

bool BaseInliner::UpdateAndCheckIndexVars(const std::vector<Expr>& indices) {
  if (idx_expr_.empty()) {
    idx_expr_ = std::move(indices);
  } else {
    if (idx_expr_.size() != indices.size()) return false;
    for (int i = 0; i < indices.size(); ++i) {
      if (Expr(idx_expr_[i]) != Expr(indices[i])) return false;
    }
  }
  return true;
}

void BaseInliner::SetIndexSubstitution(const std::vector<Expr>& indices) {
  PADDLE_ENFORCE_EQ(indices.size(),
                    idx_expr_.size(),
                    ::common::errors::InvalidArgument(
                        "The size of indices should be equal to idx_expr_"));
  int n = indices.size();
  idx_sub_var_.clear();
  idx_sub_expr_.clear();
  for (int i = 0; i < n; ++i) {
    if (idx_expr_[i].is_var()) {
      idx_sub_var_.push_back(idx_expr_[i].as_var_ref());
      idx_sub_expr_.push_back(indices[i]);
    }
  }
}

bool ComputeInliner::BodyPatternAllowInline() {
  if (!inlined_store_.defined()) {
    return false;
  }
  PADDLE_ENFORCE_NOT_NULL(
      inlined_store_.As<Store>(),
      ::common::errors::NotFound(
          "Param inlined store should be Store node! Please check."));
  if (!UpdateAndCheckIndexVars(inlined_store_.As<Store>()->indices)) {
    return false;
  }
  return true;
}

void ComputeInliner::Visit(const ir::Load* expr, Expr* op) {
  if ((expr->tensor).as_tensor_ref()->name == inlined_tensor_->name) {
    *op = ReplaceInlinedTensor(op);
    return;
  }
  IRMutator::Visit(expr, op);
}

//! Replace the 'Load' node on the tensor to 'Load' node of its producers.
Expr ComputeInliner::ReplaceInlinedTensor(Expr* load) {
  PADDLE_ENFORCE_NOT_NULL(
      load->As<ir::Load>(),
      ::common::errors::NotFound(
          "Param load should be ir::Load node! Please check."));
  SetIndexSubstitution(load->As<ir::Load>()->indices);
  Expr value_copy = ir::ir_utils::IRCopy(inlined_store_.As<Store>()->value);
  ReplaceExpr(&value_copy, idx_sub_var_, idx_sub_expr_);
  return value_copy;
}

bool ComputeInlineChecker::Check() {
  Expr root = ir_schedule_.GetRootBlock(block_);
  store_ = CheckComputeInlineValidationAndGetStore(block_, root);
  IRMutator::Visit(&root, &root);
  return !should_skip_;
}

void ComputeInlineChecker::BuildDataDependency() {
  ir_schedule_.SetBuffer(block_, "shared", true);
  auto loops = ir_schedule_.GetLoops(block_);
  ir_schedule_.SyncThreads(loops.back(), true);
}

bool ReverseComputeInliner::BodyPatternAllowInline() {
  if (!inlined_store_.defined()) {
    return false;
  }
  if (!inlined_load_.defined()) {
    return false;
  }
  if (!target_store_.defined()) {
    return false;
  }
  PADDLE_ENFORCE_NOT_NULL(
      inlined_store_.As<Store>(),
      ::common::errors::NotFound(
          "Param inlined store should be Store node! Please check."));
  PADDLE_ENFORCE_NOT_NULL(
      inlined_load_.As<Load>(),
      ::common::errors::NotFound(
          "Param inlined load should be Load node! Please check."));
  PADDLE_ENFORCE_NOT_NULL(
      target_store_.As<Store>(),
      ::common::errors::NotFound(
          "Param target store should be Store node! Please check."));
  auto find_vars = ir::ir_utils::CollectIRNodesWithoutTensor(
      inlined_store_, [&](const Expr* x) { return x->as_var(); });
  std::set<Var, CompVar> vars_set;
  for (auto& i : find_vars) vars_set.insert(i.as_var_ref());
  int n_vars = vars_set.size();
  if (!UpdateAndCheckIndexVars(inlined_store_.As<Store>()->indices)) {
    return false;
  }
  return true;
}

void ReverseComputeInliner::Visit(const ir::Load* expr, Expr* op) {
  if ((expr->tensor).as_tensor_ref()->name == inlined_tensor_->name) {
    *op = inlined_store_.As<Store>()->value;
    return;
  }
  IRMutator::Visit(expr, op);
}

void ReverseComputeInliner::Visit(const ir::Store* expr, Expr* op) {
  if ((expr->tensor).as_tensor_ref()->name == inlined_tensor_->name) {
    *op = ReplaceTargetTensor(op);
    return;
  }
  IRMutator::Visit(expr, op);
}

//! Replace the 'Load' node on the tensor to 'Load' node of its producers.
Expr ReverseComputeInliner::ReplaceInlinedTensor(Expr* load) {
  PADDLE_ENFORCE_NOT_NULL(
      load->As<ir::Load>(),
      ::common::errors::NotFound(
          "Param load should be ir::Load node! Please check."));
  SetIndexSubstitution(load->As<ir::Load>()->indices);
  Expr value_copy = ir::ir_utils::IRCopy(inlined_store_.As<Store>()->value);
  return value_copy;
}

Expr ReverseComputeInliner::ReplaceTargetTensor(Expr* store) {
  auto indices = inlined_load_.As<ir::Load>()->indices;
  PADDLE_ENFORCE_EQ(indices.size(),
                    idx_expr_.size(),
                    ::common::errors::InvalidArgument(
                        "The size of indices should be equal to idx_expr_"));
  size_t n = idx_expr_.size();
  idx_sub_var_.clear();
  idx_sub_expr_.clear();
  for (int i = 0; i < n; ++i) {
    if (indices[i].is_var()) {
      idx_sub_var_.emplace_back(indices[i].as_var_ref());
      idx_sub_expr_.emplace_back(idx_expr_[i]);
    }
  }

  Expr value_copy = ir::ir_utils::IRCopy(target_store_);
  ReplaceExpr(&value_copy, idx_sub_var_, idx_sub_expr_);
  return value_copy;
}

IRSchedule::IRSchedule() {}

IRSchedule::IRSchedule(const ModuleExpr& module_expr,
                       utils::LinearRandomEngine::StateType rand_seed,
                       bool debug_flag,
                       utils::ErrorMessageLevel err_msg_level,
                       bool is_dynamic_shape)
    : impl_(ScheduleBase::Make(
          module_expr, debug_flag, err_msg_level, is_dynamic_shape)),
      is_dynamic_shape_(is_dynamic_shape) {
  this->InitSeed(rand_seed);
}

IRSchedule::IRSchedule(ir::ModuleExpr&& mod_expr,
                       ScheduleDesc&& trace,
                       utils::LinearRandomEngine::StateType rand_seed,
                       bool is_dynamic_shape)
    : impl_(ScheduleBase::Make(std::move(mod_expr), is_dynamic_shape)),
      trace_(std::move(trace)),
      is_dynamic_shape_(is_dynamic_shape) {
  this->InitSeed(rand_seed);
}

IRSchedule::IRSchedule(const IRSchedule& other)
    : impl_(ScheduleBase::Make(ir::ir_utils::IRCopy(other.GetModule()),
                               other.IsDynamicShape())),
      trace_(other.trace_),
      is_dynamic_shape_(other.IsDynamicShape()) {
  this->InitSeed(other.ForkSeed());
}

IRSchedule& IRSchedule::operator=(const IRSchedule& src) {
  impl_ = ScheduleBase::Make(ir::ir_utils::IRCopy(src.GetModule()),
                             src.IsDynamicShape());
  trace_ = src.trace_;
  is_dynamic_shape_ = src.IsDynamicShape();
  this->InitSeed(src.ForkSeed());
  return *this;
}

IRSchedule::IRSchedule(IRSchedule&& other)
    : impl_(std::move(other.impl_)),
      trace_(std::move(other.trace_)),
      is_dynamic_shape_(other.IsDynamicShape()) {
  this->InitSeed(other.ForkSeed());
}

IRSchedule& IRSchedule::operator=(IRSchedule&& src) {
  impl_ = std::move(src.impl_);
  trace_ = std::move(src.trace_);
  is_dynamic_shape_ = src.IsDynamicShape();
  this->InitSeed(src.ForkSeed());
  return *this;
}

IRSchedule::~IRSchedule() {}

void IRSchedule::InitSeed(utils::LinearRandomEngine::StateType rand_seed) {
  this->rand_seed_ = utils::LinearRandomEngine::NormalizeState(rand_seed);
}

utils::LinearRandomEngine::StateType IRSchedule::ForkSeed() const {
  return utils::ForkRandomState(&rand_seed_);
}

void IRSchedule::SetExprs(const std::vector<Expr>& exprs) {
  return impl_->SetExprs(exprs);
  // no need to trace
}

const ModuleExpr& IRSchedule::GetModule() const {
  return impl_->GetModule();
  // no need to trace
}

bool IRSchedule::HasBlock(const std::string& block_name) const {
  return impl_->HasBlock(block_name);
  // no need to trace
}

void IRSchedule::MergeExprs() {
  impl_->MergeExprs();
  trace_.Append(ScheduleDesc::Step("MergeExprs", {}, {}, {}));
}

std::vector<Expr> IRSchedule::GetLoops(const Expr& block) const {
  auto results = impl_->GetLoops(block);
  trace_.Append(ScheduleDesc::Step(
      "GetLoops", {{"block", std::vector<Expr>({block})}}, {}, results));
  return results;
}

std::vector<Expr> IRSchedule::GetLoops(const std::string& block_name) const {
  auto results = impl_->GetLoops(block_name);
  trace_.Append(ScheduleDesc::Step(
      "GetLoopsWithName", {}, {{"block_name", block_name}}, results));
  return results;
}

std::vector<Expr> IRSchedule::GetAllBlocks() const {
  auto results = impl_->GetAllBlocks();
  trace_.Append(ScheduleDesc::Step("GetAllBlocks", {}, {}, results));
  return results;
}

std::vector<Expr> IRSchedule::GetChildBlocks(const Expr& expr) const {
  auto results = impl_->GetChildBlocks(expr);
  trace_.Append(ScheduleDesc::Step(
      "GetChildBlocks", {{"expr", std::vector<Expr>({expr})}}, {}, results));
  return results;
}

Expr IRSchedule::GetBlock(const std::string& block_name) const {
  auto result = impl_->GetBlock(block_name);
  trace_.Append(ScheduleDesc::Step(
      "GetBlock", {}, {{"block_name", block_name}}, {result}));
  return result;
}

std::vector<Expr> IRSchedule::Split(const Expr& loop,
                                    const std::vector<int>& factors) {
  if (IsDynamicShape()) return impl_->Split(loop, factors);
  std::vector<Expr> decision = SamplePerfectTile(
      loop, factors.size(), loop.As<ir::For>()->extent.as_int64(), factors);
  auto results = Split(loop, decision);
  return results;
}

std::vector<Expr> IRSchedule::Split(const std::string& block_name,
                                    int loop_index,
                                    const std::vector<int>& factors) {
  std::vector<Expr> all_loops = this->GetLoops(block_name);
  Expr loop_expr;
  PADDLE_ENFORCE_LT(loop_index,
                    (int)all_loops.size(),
                    ::common::errors::InvalidArgument(
                        "The loop index in Split should be less than total "
                        "loop's number."));
  PADDLE_ENFORCE_GE(loop_index,
                    0,
                    ::common::errors::InvalidArgument(
                        "The loop index in Split should be >= 0."));
  loop_expr = all_loops[loop_index];

  return this->Split(loop_expr, factors);
}

std::vector<Expr> IRSchedule::Split(const Expr& loop,
                                    const std::vector<Expr>& factors) {
  std::vector<int> int_factors;
  std::vector<Expr> results;
  std::for_each(factors.begin(), factors.end(), [&int_factors](const Expr& e) {
    if (e.is_constant()) int_factors.push_back(e.as_int64());
  });
  if (int_factors.size() == factors.size()) {
    results = impl_->Split(loop, int_factors);
  } else {
    results = impl_->Split(loop, factors);
  }

  trace_.Append(ScheduleDesc::Step(
      "Split",
      {{"loop", std::vector<Expr>({loop})}, {"factors", factors}},
      {},
      results));
  return results;
}

Expr IRSchedule::Fuse(const std::vector<Expr>& loops) {
  auto result = impl_->Fuse(loops);
  trace_.Append(ScheduleDesc::Step("Fuse", {{"loops", loops}}, {}, {result}));
  return result;
}

Expr IRSchedule::Fuse(const std::string& block_name,
                      const std::vector<int>& loops_index) {
  auto result = impl_->Fuse(block_name, loops_index);
  trace_.Append(ScheduleDesc::Step(
      "FuseWithName",
      {},
      {{"block_name", block_name}, {"loops_index", loops_index}},
      {result}));
  return result;
}

Expr IRSchedule::Fuse(const Expr& block, const std::vector<int>& loops_index) {
  auto result = impl_->Fuse(block, loops_index);
  trace_.Append(ScheduleDesc::Step("FuseWithBlock",
                                   {{"block", std::vector<Expr>({block})}},
                                   {{"loops_index", loops_index}},
                                   {result}));
  return result;
}

void IRSchedule::ComputeAt(const Expr& block,
                           const Expr& loop,
                           bool keep_unit_loops) {
  impl_->ComputeAt(block, loop, keep_unit_loops);
  trace_.Append(ScheduleDesc::Step("ComputeAt",
                                   {{"block", std::vector<Expr>({block})},
                                    {"loop", std::vector<Expr>({loop})}},
                                   {{"keep_unit_loops", keep_unit_loops}},
                                   {}));
}

void IRSchedule::SimpleComputeAt(const Expr& block, const Expr& loop) {
  impl_->SimpleComputeAt(block, loop);
  trace_.Append(ScheduleDesc::Step("SimpleComputeAt",
                                   {{"block", std::vector<Expr>({block})},
                                    {"loop", std::vector<Expr>({loop})}},
                                   {},
                                   {}));
}

void IRSchedule::ReverseComputeAt(const Expr& block,
                                  const Expr& loop,
                                  bool keep_unit_loops) {
  impl_->ReverseComputeAt(block, loop, keep_unit_loops);
  trace_.Append(ScheduleDesc::Step("ReverseComputeAt",
                                   {{"block", std::vector<Expr>({block})},
                                    {"loop", std::vector<Expr>({loop})}},
                                   {{"keep_unit_loops", keep_unit_loops}},
                                   {}));
}

Expr IRSchedule::GetRootBlock(const Expr& expr) const {
  auto result = impl_->GetRootBlock(expr);
  trace_.Append(ScheduleDesc::Step(
      "GetRootBlock", {{"expr", std::vector<Expr>({expr})}}, {}, {result}));
  return result;
}

Expr IRSchedule::CacheRead(const Expr& block,
                           int read_buffer_index,
                           const std::string& memory_type) {
  auto result = impl_->CacheRead(block, read_buffer_index, memory_type);
  trace_.Append(ScheduleDesc::Step(
      "CacheRead",
      {{"block", std::vector<Expr>({block})}},
      {{"read_buffer_index", read_buffer_index}, {"memory_type", memory_type}},
      {result}));
  return result;
}

Expr IRSchedule::CacheWrite(const Expr& block,
                            int write_buffer_index,
                            const std::string& memory_type) {
  auto result = impl_->CacheWrite(block, write_buffer_index, memory_type);
  trace_.Append(ScheduleDesc::Step("CacheWrite",
                                   {{"block", std::vector<Expr>({block})}},
                                   {{"write_buffer_index", write_buffer_index},
                                    {"memory_type", memory_type}},
                                   {result}));
  return result;
}

void IRSchedule::SyncThreads(const Expr& ir_node, bool after_node) {
  impl_->SyncThreads(ir_node, after_node);
  trace_.Append(ScheduleDesc::Step("SyncThreads",
                                   {{"ir_node", std::vector<Expr>({ir_node})}},
                                   {{"after_node", after_node}},
                                   {}));
}

void IRSchedule::SetBuffer(Expr& block,
                           const std::string& memory_type,
                           bool fixed) {
  impl_->SetBuffer(block, memory_type, fixed);
  trace_.Append(
      ScheduleDesc::Step("SetBuffer",
                         {{"block", std::vector<Expr>({block})}},
                         {{"memory_type", memory_type}, {"fixed", fixed}},
                         {}));
}

Expr IRSchedule::AddUnitLoop(const Expr& block) {
  Expr ret = impl_->AddUnitLoop(block);
  trace_.Append(ScheduleDesc::Step(
      "AddUnitLoop", {{"block", std::vector<Expr>({block})}}, {}, {ret}));
  return ret;
}

Expr IRSchedule::Reorder(const std::vector<Expr>& loops) {
  Expr ret = impl_->Reorder(loops);
  trace_.Append(ScheduleDesc::Step("Reorder", {{"loops", loops}}, {}, {ret}));
  return ret;
}

Expr IRSchedule::Reorder(const std::string& block_name,
                         const std::vector<int>& loops_index) {
  Expr ret = impl_->Reorder(block_name, loops_index);
  trace_.Append(ScheduleDesc::Step(
      "ReorderWithName",
      {},
      {{"block_name", block_name}, {"loops_index", loops_index}},
      {ret}));
  return ret;
}

Expr IRSchedule::Reorder(const Expr& block,
                         const std::vector<int>& loops_index) {
  Expr ret = impl_->Reorder(block, loops_index);
  trace_.Append(ScheduleDesc::Step("ReorderWithBlock",
                                   {{"block", std::vector<Expr>({block})}},
                                   {{"loops_index", loops_index}},
                                   {ret}));
  return ret;
}

void IRSchedule::Parallel(const Expr& loop) {
  impl_->Parallel(loop);
  trace_.Append(ScheduleDesc::Step(
      "Parallel", {{"loop", std::vector<Expr>({loop})}}, {}, {}));
}

void IRSchedule::Vectorize(const Expr& loop, int factor) {
  impl_->Vectorize(loop, factor);
  trace_.Append(ScheduleDesc::Step("Vectorize",
                                   {{"loop", std::vector<Expr>({loop})}},
                                   {{"factor", factor}},
                                   {}));
}

void IRSchedule::Unroll(const Expr& loop) {
  impl_->Unroll(loop);
  trace_.Append(ScheduleDesc::Step(
      "Unroll", {{"loop", std::vector<Expr>({loop})}}, {}, {}));
}

void IRSchedule::ComputeInline(const Expr& schedule_block) {
  impl_->ComputeInline(schedule_block);
  trace_.Append(ScheduleDesc::Step(
      "ComputeInline",
      {{"schedule_block", std::vector<Expr>({schedule_block})}},
      {},
      {}));
}

void IRSchedule::ReverseComputeInline(const Expr& schedule_block) {
  impl_->ReverseComputeInline(schedule_block);
  trace_.Append(ScheduleDesc::Step(
      "ReverseComputeInline",
      {{"schedule_block", std::vector<Expr>({schedule_block})}},
      {},
      {}));
}

void IRSchedule::Bind(const Expr& loop, const std::string& thread_axis) {
  impl_->Bind(loop, thread_axis);
  trace_.Append(ScheduleDesc::Step("Bind",
                                   {{"loop", std::vector<Expr>({loop})}},
                                   {{"thread_axis", thread_axis}},
                                   {}));
}

Expr IRSchedule::Rfactor(const Expr& rf_loop, int rf_axis) {
  auto result = impl_->Rfactor(rf_loop, rf_axis);
  trace_.Append(ScheduleDesc::Step("Rfactor",
                                   {{"rf_loop", std::vector<Expr>({rf_loop})}},
                                   {{"rf_axis", rf_axis}},
                                   {result}));
  return result;
}

Expr IRSchedule::FactorizeReduction(const Expr& rf_loop,
                                    int rf_axis,
                                    bool with_write_back_block_init) {
  auto result =
      impl_->FactorizeReduction(rf_loop, rf_axis, with_write_back_block_init);
  trace_.Append(ScheduleDesc::Step(
      "FactorizeReduction",
      {{"rf_loop", std::vector<Expr>({rf_loop})}},
      {{"rf_axis", rf_axis},
       {"with_write_back_block_init", with_write_back_block_init}},
      {result}));
  return result;
}

void IRSchedule::Annotate(const Expr& block,
                          const std::string& key,
                          const attr_t& value) {
  impl_->Annotate(block, key, value);

#define TRACE_ANNOTATE_ITEM(data_type, step_name)               \
  if (absl::holds_alternative<data_type>(value)) {              \
    trace_.Append(ScheduleDesc::Step(                           \
        #step_name,                                             \
        {{"block", std::vector<Expr>({block})}},                \
        {{"key", key}, {"value", absl::get<data_type>(value)}}, \
        {}));                                                   \
    return;                                                     \
  }
  TRACE_ANNOTATE_ITEM(int, AnnotateIntAttr)
  TRACE_ANNOTATE_ITEM(bool, AnnotateBoolAttr)
  TRACE_ANNOTATE_ITEM(float, AnnotateFloatAttr)
  TRACE_ANNOTATE_ITEM(std::string, AnnotateStringAttr)
#undef TRACE_ANNOTATE_ITEM

  std::stringstream ss;
  ss << "Value of attribute:" << key << " input unsupported data type";
  PADDLE_THROW(::common::errors::InvalidArgument(ss.str()));
}

void IRSchedule::Unannotate(Expr& block, const std::string& key) {
  impl_->Unannotate(block, key);
  trace_.Append(ScheduleDesc::Step("Unannotate",
                                   {{"block", std::vector<Expr>({block})}},
                                   {{"key", key}},
                                   {}));
}

void IRSchedule::FlattenLoops(const std::vector<Expr>& loops,
                              const bool force_flat) {
  impl_->FlattenLoops(loops, force_flat);
  trace_.Append(ScheduleDesc::Step("FlattenLoops",
                                   {{"loop", std::vector<Expr>({loops})}},
                                   {{"force_flat", force_flat}},
                                   {}));
}

void IRSchedule::CopyTransformAndLoopInfo(const Expr& block,
                                          const Expr& block_target) {
  impl_->CopyTransformAndLoopInfo(block, block_target);
  // don't support to trace, because we can't ensure both blocks are from the
  // same ModuleExpr
}

void IRSchedule::CopyTransformAndLoopInfo(
    const std::string& block_name, const std::string& block_target_name) {
  impl_->CopyTransformAndLoopInfo(block_name, block_target_name);
  // don't support to trace, because we can't ensure both blocks are from the
  // same ModuleExpr
}

std::vector<Expr> IRSchedule::SamplePerfectTile(
    const Expr& loop,
    int n,
    int max_innermost_factor,
    const std::vector<int>& decision) {
  std::vector<Expr> factors;
  std::vector<int> new_decision;
  if (decision.empty()) {
    factors =
        impl_->SamplePerfectTile(&rand_seed_, loop, n, max_innermost_factor);
    std::transform(factors.begin(),
                   factors.end(),
                   std::back_inserter(new_decision),
                   [](Expr x) { return x.as_int32(); });
  } else {
    new_decision = decision;
    std::transform(decision.begin(),
                   decision.end(),
                   std::back_inserter(factors),
                   [](int x) { return Expr(x); });
  }
  trace_.Append(
      ScheduleDesc::Step("SamplePerfectTile",
                         {{"loop", std::vector<Expr>({loop})}},
                         {{"n", n},
                          {"max_innermost_factor", max_innermost_factor},
                          {"decision", new_decision}},
                         factors));
  return factors;
}

void IRSchedule::TagPostSchedule() {
  trace_.Append(ScheduleDesc::Step("TagPostSchedule", {}, {}, {}));
}

Expr IRSchedule::SampleCategorical(const std::vector<int>& candidates,
                                   const std::vector<float>& probs,
                                   const std::vector<int>& decision) {
  Expr result;
  std::vector<int> new_decision;
  if (decision.empty()) {
    result = impl_->SampleCategorical(&rand_seed_, candidates, probs);
    new_decision.push_back(result.as_int32());
  } else {
    new_decision = decision;
    for (auto ndco : new_decision) {
      result = Expr(ndco);
    }
  }
  trace_.Append(ScheduleDesc::Step("SampleCategorical",
                                   {},
                                   {{"candidates", candidates},
                                    {"probs", probs},
                                    {"decision", new_decision}},
                                   {result}));
  return result;
}

}  // namespace ir
}  // namespace cinn
