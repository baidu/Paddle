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

#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/cinn/hlir/framework/compile_error.h"
#include "paddle/cinn/hlir/framework/pir/op_lowering_util.h"
#include "paddle/cinn/hlir/framework/pir/utils.h"
#include "paddle/cinn/hlir/pe/map_expr_to_ir.h"
#include "paddle/cinn/ir/dim.h"
#include "paddle/cinn/ir/group_schedule/base_group_scheduler.h"
#include "paddle/cinn/ir/schedule/ir_schedule.h"
#include "paddle/cinn/lang/placeholder.h"
#include "paddle/cinn/optim/schedule_block_dce.h"
#include "paddle/cinn/optim/transform_gpu_forloop.h"
#include "paddle/common/ddim.h"
#include "paddle/common/enforce.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_op.h"

namespace cinn {
namespace hlir {
namespace framework {
namespace pir {
namespace trivial_fusion_detail {

namespace ComposeUtils {

template <typename T>
std::vector<T> ConcatVector(const std::vector<T>& first,
                            const std::vector<T>& second) {
  std::vector<T> result = first;
  result.insert(result.end(), second.begin(), second.end());
  return result;
}

template <typename T, typename U>
std::unordered_map<T, U> MakeMap(const std::vector<T>& keys,
                                 const std::vector<U>& values) {
  std::unordered_map<T, U> result = std::unordered_map<T, U>();

  PADDLE_ENFORCE_EQ(keys.size(),
                    values.size(),
                    ::common::errors::InvalidArgument(
                        "Required keys shall have same size with values."));
  for (int i = 0; i < keys.size(); i++) {
    result[keys[i]] = values[i];
  }
  return result;
}

std::vector<ir::Var> ExprVec2VarVec(const std::vector<ir::Expr>& in);
std::vector<ir::Expr> VarVec2ExprVec(const std::vector<ir::Var>& in);

std::vector<ir::Expr> GetEachTensorLoadExpr(const ir::Expr& body,
                                            const ir::Tensor& tensor);

struct MappingTargetExprToDestExprMutator : public ir::IRMutator<> {
  explicit MappingTargetExprToDestExprMutator(const ir::Expr& source,
                                              const ir::Expr& dest);

  void operator()(Expr* expr);

 private:
  void Visit(const ir::Load* load, Expr* op) override;
  void Visit(const ir::Store* store, Expr* op) override;
  void Visit(const ir::Reduce* reduce, Expr* op) override;
  void Visit(const ir::For* for_node, Expr* op) override;
  void Visit(const ir::Block* block_node, Expr* op) override;
  void Visit(const ir::ScheduleBlockRealize* realize, Expr* op) override;

 private:
  ir::Expr source_;
  ir::Expr dest_;
};

bool CheckIterEq(const std::vector<ir::Var>& up_iter,
                 const std::vector<ir::Var>& down_iter);

ir::Expr CopiedReplaceExpr(const Expr& source,
                           const std::vector<Var>& replaced,
                           const std::vector<Expr>& candidates);
void SubstituteTargetExprWithDestExpr(const ir::Expr& source,
                                      const ir::Expr& dest,
                                      ir::Expr* body);

ir::Expr SubstituteIndexVector(const Expr& source,
                               const std::vector<Var>& load_vars,
                               const std::vector<ir::Expr>& indices);

template <typename FusionOp>
void ReplaceDownstreamLoadExprWithUpstreamComputeBody(
    const FusionOp& upstream,
    const ir::Expr& downstream_load_expr,
    ir::Expr* downstream_body) {
  ComposeUtils::SubstituteTargetExprWithDestExpr(
      downstream_load_expr,
      ComposeUtils::SubstituteIndexVector(
          GetComputeBody(upstream),
          GetOutputIters(upstream),
          downstream_load_expr.As<ir::Load>()->indices),
      downstream_body);
}
}  // namespace ComposeUtils

namespace ExprSetFinderUtils {

using ExprSet = std::vector<ir::Expr>;
using Expr2ExprSet = std::function<ExprSet(const ir::Expr& x)>;
struct ExprSetFinder {
  Expr2ExprSet f_;
  std::string name;
  explicit ExprSetFinder(Expr2ExprSet f, std::string s = "");

  ExprSet operator()(const ir::Expr& x) const;
  ir::Expr GetSingle(const ir::Expr& x) const;
  ExprSetFinder operator*(ExprSetFinder x) const;
  static ExprSetFinder GetIdentity();
};

template <typename Teller>
ExprSetFinder Collector(Teller t, std::string name = "") {
  return ExprSetFinder(
      [=](const ir::Expr& x) -> ExprSet {
        const auto& rs = cinn::ir::ir_utils::CollectIRNodesInOrder(x, t);
        return std::vector(rs.begin(), rs.end());
      },
      name);
}

template <typename FilterFunc>
ExprSetFinder FilterMaker(FilterFunc t, std::string name) {
  return ExprSetFinder(
      [=](const ir::Expr& x) -> ExprSet {
        if (t(x)) {
          return {x};
        }
        return {};
      },
      name);
}

extern ExprSetFinder Identity;

extern ExprSetFinder Store2Value;

extern ExprSetFinder Realizer2ScheduleBlock;

extern ExprSetFinder Realizer2IterValues;

extern ExprSetFinder ScheduleBlock2Body;

extern ExprSetFinder ScheduleBlockRealizeNotRoot;

extern ExprSetFinder ScheduleBlockRealizeIsRoot;

extern ExprSetFinder ScheduleBlockRealizeIsNotInit;

extern ExprSetFinder ScheduleBlockRealizeIsInit;

extern ExprSetFinder IsFor;

extern ExprSetFinder ChildScheduleBlocks;

extern ExprSetFinder ChildScheduleBlockRealizes;

extern ExprSetFinder ChildRootScheduleBlockRealizes;

extern ExprSetFinder For2Min;

extern ExprSetFinder For2Max;

extern ExprSetFinder ChildStores;

extern ExprSetFinder ChildTensorLoads;

extern ExprSetFinder ChildTensorStores;

extern ExprSetFinder ChildFors;

extern ExprSetFinder ChildIfThenElses;

ExprSetFinder IsForIterVar(const ir::Var& var);

ExprSetFinder FilterLoadByTensor(const ir::Tensor& tensor);

ExprSetFinder FindFather(const ir::Expr& root);

template <class T, class M>
std::vector<T> MapVector(const std::vector<T>& as, M func) {
  std::vector<T> res;
  for (const auto& a : as) {
    res.push_back(func(a));
  }
  return res;
}
}  // namespace ExprSetFinderUtils

namespace ExprTransformerUtils {
using ExprTransformFunc = std::function<ir::Expr(ir::Expr)>;
struct ExprTransformer {
  ExprTransformFunc f_;
  explicit ExprTransformer(ExprTransformFunc f);
  ir::Expr operator()(const ir::Expr& x) const;
  ExprTransformer operator*(const ExprTransformer& x) const;
};

extern ExprTransformer Identity;

ExprTransformer WrapForTransformer(const ir::Var& v);

ExprTransformer WrapForsTransformer(const std::vector<ir::Var>& vs);
ExprTransformer ChangeTensorLoadTransformer(const ir::Tensor& tensor,
                                            const ir::Expr& dst_load);

void ReplaceTarget(ir::Expr* e, const ir::Expr& t, const ir::Expr dst);

bool IsReduceBool(const ir::Expr& lhs, const ir::Expr& rhs);

ExprTransformer WrapStoreTransformer(const ir::Tensor& tensor,
                                     const std::vector<ir::Expr>& indices);

ExprTransformer WrapReduceOperation(const ir::Reduce::ReduceType& reduce_type,
                                    const ir::Tensor& tensor,
                                    const std::vector<ir::Expr>& axis_exprs);

std::vector<ir::Var> CreateInnerBlockVars(
    const std::vector<ir::Var>& block_vars);

ExprTransformer ChangeVarTransformer(const std::vector<ir::Var>& target_vars,
                                     const std::vector<ir::Var>& dest_vars);

ExprTransformer ReplaceVarTransformer(const std::vector<ir::Var>& target_vars,
                                      const std::vector<ir::Expr>& dest_exprs);

// insert after followed_finder. only support For and ScheduleBlockRealizer
ExprTransformer UnsqueezeForTransformer(
    const ExprSetFinderUtils::ExprSetFinder& followed_finder,
    const ir::Var& to_append_var);

ExprTransformer SubstituteByScheduleBlockRealize(const ir::Expr& realize);

ExprTransformer WrapScheduleRealizer(const std::vector<ir::Var>& block_vars,
                                     const std::string& tensor_name);

ExprTransformer TransposeForsTransformer(const std::vector<int32_t>& perm);
ExprTransformer RemoveOnesTransformer(const std::vector<int32_t>& ones);
ExprTransformer InsertForsTransformer(const std::vector<int32_t>& axis,
                                      const std::vector<ir::Var>& vars);
ExprTransformer InsertIfForAppendVarsTransformer();
int InplaceMutateSingleExpr(ir::Expr* root,
                            const ExprSetFinderUtils::ExprSetFinder& finder,
                            const ExprTransformer& transformer);
}  // namespace ExprTransformerUtils

std::vector<OpPatternKind> GetOpPatternKindVector(
    const std::vector<::pir::Operation*>& ops);

template <class A, class C, class Func>
void SequenceMutator(const std::vector<A>& as, C* acc, const Func& mutator) {
  VLOG(4) << "SequenceTransform Init: " << acc;
  for (int i = 0; i < as.size(); ++i) {
    mutator(as[i], acc);
    VLOG(4) << "SequenceTransform Iter: " << acc;
  }
}

bool IsTrivialKind(OpPatternKind kind);

void CheckFusionInputValid(const std::vector<ir::Expr>& op_compute_bodies,
                           const std::vector<OpPatternKind>& op_patterns);

static bool IsReduceBody(const ir::Expr& expr_body) {
  return !(ExprSetFinderUtils::ChildScheduleBlockRealizes *
           ExprSetFinderUtils::ScheduleBlockRealizeIsInit)(expr_body)
              .empty();
}

std::vector<ir::Var> AppendBound(const std::vector<ir::Var> vars,
                                 const ir::Expr& root);

std::vector<ir::Var> GetNonReduceLoopVars(const ir::Expr& root);
std::vector<ir::Var> GetAllLoopVars(const ir::Expr& root);

ir::Expr GetBodyBlock(const ir::Expr& root);

ir::Expr ReshapeLoop(const ir::Expr& root,
                     const std::vector<symbol::DimExpr>& in_shape,
                     const std::vector<symbol::DimExpr>& out_shape);

void CheckLoopAlignment(const std::vector<ir::Expr>& roots);

}  // namespace trivial_fusion_detail
}  // namespace pir
}  // namespace framework
}  // namespace hlir
}  // namespace cinn
