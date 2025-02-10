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

#include "paddle/cinn/poly/stage.h"

#include <math.h>

#include <algorithm>
#include <set>
#include <unordered_set>
#include <utility>

#include "paddle/cinn/common/axis.h"
#include "paddle/cinn/ir/ir_mutator.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/ir_visitor.h"
#include "paddle/cinn/ir/operation.h"
#include "paddle/cinn/ir/tensor.h"
#include "paddle/cinn/ir/utils/ir_copy.h"
#include "paddle/cinn/ir/utils/ir_nodes_collector.h"
#include "paddle/cinn/ir/utils/ir_replace.h"
#include "paddle/cinn/lang/compute.h"
#include "paddle/cinn/optim/ir_simplify.h"
#include "paddle/cinn/optim/replace_var_with_expr.h"
#include "paddle/cinn/poly/compute_at_transform.h"
#include "paddle/cinn/poly/isl_utils.h"
#include "paddle/cinn/utils/functional.h"
#include "paddle/cinn/utils/string.h"
#include "paddle/common/enforce.h"
namespace cinn {
namespace poly {
void RemoveDuplicate(std::vector<std::vector<Expr>> &indices) {  // NOLINT
  std::set<std::string> temp;
  for (int i = 0; i < indices.size(); i++) {
    std::string index_str = "";
    for (auto &j : indices[i]) {
      index_str += utils::GetStreamCnt(j) + ",";
    }
    if (temp.count(index_str) == 0) {
      temp.insert(index_str);
    } else {
      indices.erase(indices.begin() + i);
      i--;
    }
  }
}

std::vector<Iterator> NamesToIterators(const std::vector<std::string> &names) {
  std::vector<Iterator> res;
  for (auto &name : names) {
    res.emplace_back(name);
  }
  return res;
}

void Stage::InitTransform() {
  std::string id = isl_set_get_tuple_name(domain_.get());

  auto dims = isl_get_dim_names(domain_);
  auto dims_repr = utils::Join(dims, ", ");

  auto repr = utils::StringFormat("{ %s[%s] -> %s[%s] }",
                                  id.c_str(),
                                  dims_repr.c_str(),
                                  id.c_str(),
                                  dims_repr.c_str());
  transform_ = isl::map(domain_.ctx(), repr);

  // set dimension names
  for (int i = 0; i < dims.size(); i++) {
    transform_ = isl::manage(isl_map_set_dim_name(
        transform_.release(), isl_dim_in, i, dims[i].c_str()));
    transform_ = isl::manage(isl_map_set_dim_name(
        transform_.release(), isl_dim_out, i, dims[i].c_str()));
  }
}

Stage::Stage(const isl::set &domain, Expr expr, ir::_Tensor_ *tensor)
    : domain_(domain), expr_(expr), tensor_(tensor) {
  PADDLE_ENFORCE_EQ(!domain_.is_null(),
                    true,
                    ::common::errors::InvalidArgument(
                        "Domain should not be null, please check!"));
  PADDLE_ENFORCE_EQ(!domain_.is_empty(),
                    true,
                    ::common::errors::InvalidArgument(
                        "Domain should not be empty, please check!"));
  InitTransform();
}

std::tuple<Iterator, Iterator> Stage::SplitOuter(const std::string &level,
                                                 int nparts) {
  return std::move(SplitOuter(Iterator(level), nparts));
}

std::tuple<Iterator, Iterator> Stage::SplitOuter(int level, int nparts) {
  AssertAxisIsNotLocked(level);
  auto dim_names = isl_get_dim_names(transform_, isl_dim_out);
  auto axis_name = dim_names.at(level);
  return SplitOuter(axis_name, nparts);
}

int Stage::GetDimRange(int level) {
  auto _minv_maxv_ = isl_set_get_axis_range(transformed_domain().get(), level);
  auto &minv = std::get<0>(_minv_maxv_);
  auto &maxv = std::get<1>(_minv_maxv_);
  int max_iv = maxv.get_num_si();
  int min_iv = minv.get_num_si();
  PADDLE_ENFORCE_EQ(
      0,
      min_iv,
      ::common::errors::InvalidArgument(
          "The min value of the axis should be 0, but got %d", min_iv));
  return max_iv + 1;
}

std::tuple<Iterator, Iterator> Stage::SplitOuter(const Iterator &level,
                                                 int nparts) {
  int offset = isl_set_find_dim_by_name(
      transformed_domain().get(), isl_dim_set, level.id.c_str());
  PADDLE_ENFORCE_GE(
      offset,
      0,
      ::common::errors::InvalidArgument("Offset must greater or equal to 0."));
  AssertAxisIsNotLocked(offset);
  auto _minv_maxv_ = isl_set_get_axis_range(transformed_domain().get(), offset);
  auto &minv = std::get<0>(_minv_maxv_);
  auto &maxv = std::get<1>(_minv_maxv_);
  int max_iv = maxv.get_num_si();
  auto dim_names = isl_get_dim_names(transform_, isl_dim_out);
  double temp = static_cast<double>(max_iv + 1.0) / static_cast<double>(nparts);
  int factor_inner = ceil(temp);
  return Split(level, factor_inner);
}

std::tuple<Iterator, Iterator> Stage::Split(int level, int factor) {
  AssertAxisIsNotLocked(level);
  auto dim_names = isl_get_dim_names(transform_, isl_dim_out);
  auto axis_name = dim_names.at(level);
  return Split(axis_name, factor);
}

std::tuple<Iterator, Iterator> Stage::Split(const Iterator &level, int factor) {
  int offset = isl_set_find_dim_by_name(
      transformed_domain().get(), isl_dim_set, level.id.c_str());
  PADDLE_ENFORCE_GE(
      offset,
      0,
      ::common::errors::InvalidArgument("Offset must greater or equal to 0."));
  AssertAxisIsNotLocked(offset);

  auto dim_names = isl_get_dim_names(transform_, isl_dim_out);

  VLOG(2) << "domain: " << domain_;
  VLOG(2) << "schedule: " << transform_;

  auto from_iters = NamesToIterators(dim_names);
  std::vector<Iterator> to_iters;
  std::vector<Condition> conds;
  Iterator inner_iter(InnerName(level.id));
  Iterator outer_iter(OuterName(level.id));
  for (auto &dim : dim_names) {
    if (dim == level.id) {
      to_iters.push_back(outer_iter);
      to_iters.push_back(inner_iter);

      conds.emplace_back(utils::StringFormat(
          "%s=floor(%s/%d)", outer_iter.id.c_str(), level.id.c_str(), factor));
      VLOG(3) << "outer cond: " << conds.back();
      conds.emplace_back(utils::StringFormat(
          "%s=%s %s %d", inner_iter.id.c_str(), level.id.c_str(), "%", factor));

      VLOG(3) << "inner cond: " << conds.back();
    } else {
      to_iters.emplace_back(dim);
    }
  }

  Map transform(domain_.ctx(), id(), from_iters, to_iters, conds, id());
  VLOG(3) << "transform: " << transform.__str__();
  transform_ = transform_.apply_range(transform.to_isl());
  auto range_dims = utils::Map<std::vector<Iterator>, std::string>(
      to_iters, [](const Iterator &x) { return x.id; });
  isl_set_dim_names(&transform_, isl_dim_out, range_dims);

  VLOG(3) << "transform " << transform.to_isl();
  VLOG(3) << "schedule after transform: " << transform_;
  VLOG(3) << "iterators: " << outer_iter << " " << inner_iter;

  return std::make_tuple(outer_iter, inner_iter);
}

void Stage::Reorder(const std::vector<Iterator> &order) {
  auto in_names = isl_get_dim_names(transform_, isl_dim_out);
  // assert all the iterators in the isl::set.
  std::unordered_set<std::string> in_name_set(in_names.begin(), in_names.end());
  std::set<Iterator> order_set(order.begin(), order.end());

  std::vector<Iterator> range_iters, domain_iters;
  for (auto &o : order) {
    PADDLE_ENFORCE_GT(in_name_set.count(o.id),
                      0UL,
                      ::common::errors::InvalidArgument(
                          "Iterator %s not in the existing axis.", o.id));
  }

  int order_offset = 0;
  for (auto &iter_name : in_names) {
    Iterator iter(iter_name);

    domain_iters.push_back(iter);

    if (order_set.count(iter)) {
      range_iters.push_back(order[order_offset++]);
    } else {
      range_iters.push_back(iter);
    }
  }

  PADDLE_ENFORCE_EQ(
      range_iters.size(),
      in_names.size(),
      ::common::errors::InvalidArgument(
          "The size of range_iters should be equal to in_names.size."));

  Map transform(domain().ctx(), id(), domain_iters, range_iters, {}, id());
  transform_ = transform_.apply_range(transform.to_isl());
}

void Stage::Reorder(const std::vector<int> &order) {
  std::vector<Iterator> iters;
  for (int id : order) iters.push_back(ith_iterator(id));
  Reorder(iters);
}

std::tuple<Iterator, Iterator, Iterator, Iterator>  //
Stage::Tile(int level0, int level1, int factor0, int factor1) {
  AssertAxisIsNotLocked(level0);
  AssertAxisIsNotLocked(level1);
  Iterator i0(cinn::common::axis_name(level0));
  Iterator i1(cinn::common::axis_name(level1));
  return Tile(i0, i1, factor0, factor1);
}

std::tuple<Iterator, Iterator, Iterator, Iterator> Stage::Tile(
    const Iterator &level0, const Iterator &level1, int factor0, int factor1) {
  auto _level0_outer_level0_inner_ = Split(level0, factor0);  // NOLINT
  auto &level0_outer = std::get<0>(_level0_outer_level0_inner_);
  auto &level0_inner = std::get<1>(_level0_outer_level0_inner_);
  auto _level1_outer_level1_inner_ = Split(level1, factor1);  // NOLINT
  auto &level1_outer = std::get<0>(_level1_outer_level1_inner_);
  auto &level1_inner = std::get<1>(_level1_outer_level1_inner_);
  return std::make_tuple(
      level0_outer, level0_inner, level1_outer, level1_inner);
}

void Stage::ComputeAtSchedule(Stage *other, int level, ComputeAtKind kind) {
  // TODO(Superjomn) Check there are data dependency between `self` and `other`,
  // or the `ComputeAt` is meaningless.
  PADDLE_ENFORCE_NOT_NULL(
      other->tensor(),
      ::common::errors::InvalidArgument(
          "Stage tensor should not be null, please check!"));
  PADDLE_ENFORCE_NOT_NULL(tensor(),
                          ::common::errors::InvalidArgument(
                              "Tensor should not be null, please check!"));

  ComputeAtRelation relation;
  relation.stage = other;
  relation.level = level;

  PADDLE_ENFORCE_EQ(
      relation.IsCompatible(this),
      true,
      ::common::errors::InvalidArgument(
          "Cannot apply ComputeAtSchedule with level: %d from \n%s\n to \n%s.",
          level,
          isl_set_to_str(this->transformed_domain().get()),
          isl_set_to_str(other->transformed_domain().get())));
  compute_ats_[other->id()] = relation;

  // Consider the order if provide.
  switch (kind) {
    case kComputeAtBefore:
      other->CtrlDepend(ir::Tensor(tensor()));
      break;
    case kComputeAtAfter:
      CtrlDepend(ir::Tensor(other->tensor()));
      break;
    case kComputeAtAuto:
      // Do nothing.
      break;
  }

  // Lock all the axis.
  for (int i = 0; i < isl_map_dim(transform_.get(), isl_dim_out); i++) {
    LockAxis(i);
  }
}

void Stage::ChangeIndex(Stage *other) {
  auto indices =
      optim::CollectTensorIndex(&(other->expr_), this->tensor()->name);
  RemoveDuplicate(indices);
  if (indices.empty()) {
    return;
  }
  if (indices.size() >= 2) {
    AddForLoopInTransform(indices);
  }
  this->tensor()->new_indices = indices[0];

  std::vector<Var> axis_var = cinn::common::GenDefaultAxis(indices[0].size());
  for (int i = 0; i < axis_var.size(); i++) {
    optim::ReplaceVarWithExpr(&(this->expr_), axis_var[i], indices[0][i]);
  }
}

// Return a - b as integer.
int Minus(const Expr &a, const Expr &b) {
  Expr diff = ir::Sub::Make(a, b);
  optim::Simplify(&diff);
  if (!diff.is_constant()) {
    LOG(ERROR) << "Range is not constant";
  }
  int int_range = diff.as_int32();
  return int_range;
}

// Return the range = max - min among all indices[i][axis](i = 0,1,2,...)
int GetRange(std::vector<std::vector<Expr>> &indices, int axis) {  // NOLINT
  Expr max_expr = indices[0][axis];
  Expr min_expr = indices[0][axis];
  for (auto i = 1; i < indices.size(); i++) {
    if (Minus(indices[i][axis], min_expr) < 0) min_expr = indices[i][axis];
    if (Minus(max_expr, indices[i][axis]) < 0) max_expr = indices[i][axis];
  }
  indices[0][axis] = min_expr;
  return Minus(max_expr, min_expr);
}

void Stage::AddForLoopInTransform(std::vector<std::vector<Expr>> &indices) {
  for (int i = 0; i < indices[0].size(); i++) {
    int int_range = GetRange(indices, i);
    if (int_range == 0) continue;

    std::string dim_name = cinn::common::axis_name(i) + "_at";
    Var dim_var(dim_name);
    indices[0][i] = ir::Add::Make(indices[0][i], Expr(dim_var));
    std::string this_domain = isl_set_to_str(domain_.get());
    std::string this_transform = isl_map_to_str(transform_.get());
    isl::ctx this_ctx = domain_.ctx();
    isl::set domain2(this_ctx, this_domain);
    std::string tuple_name = isl_set_get_tuple_name(domain_.get());
    domain2 = isl::manage(isl_set_add_dims(domain2.release(), isl_dim_out, 1));
    int dim_size = isl_set_dim(domain2.get(), isl_dim_out);

    domain2 = isl::manage(isl_set_set_dim_name(
        domain2.release(), isl_dim_out, dim_size - 1, dim_name.c_str()));
    domain2 = isl::manage(
        isl_set_set_tuple_name(domain2.release(), tuple_name.c_str()));
    std::string domain2_str = isl_set_to_str(domain2.get());
    domain2_str = domain2_str.substr(0, domain2_str.size() - 1) +
                  "and 0 <= " + dim_name + " <= " + std::to_string(int_range) +
                  " }";
    VLOG(2) << "Edited domain is: " << domain2_str;
    isl::set domain_res(this_ctx, domain2_str);
    domain_ = domain_res;

    isl::map transform2(this_ctx, this_transform);
    transform2 =
        isl::manage(isl_map_add_dims(transform2.release(), isl_dim_in, 1));
    dim_size = isl_map_dim(transform2.get(), isl_dim_in);
    transform2 = isl::manage(isl_map_set_dim_name(
        transform2.release(), isl_dim_in, dim_size - 1, dim_name.c_str()));
    transform2 = isl::manage(isl_map_set_tuple_name(
        transform2.release(), isl_dim_in, tuple_name.c_str()));
    std::string transform2_str = isl_map_to_str(transform2.get());
    int found_index = transform2_str.find_last_of("]");
    transform2_str =
        transform2_str.substr(0, found_index) + ", " + dim_name +
        "' = " + dim_name +
        transform2_str.substr(found_index, transform2_str.size() - found_index);
    VLOG(2) << "Edited transform is: " << transform2_str;
    isl::map trans_res(this_ctx, transform2_str);
    transform_ = trans_res;
  }
}
/**
 * Change this stage's domain to be consistent with other's domain.
 * @param level Change the domain lower than level to be consistent with other's
 * domain. For example, when this->domain_ is "{ [i0, i1] : 0 <= i0 <= 9 and 0
 * <= i1 <= 9 }", other->domain_ is "{ [i0, i1] : 0 <= i0 <= 4 and 0 <= i1 <= 4
 * }" and level = 0. Then this->domain_ will be changed to "{ [i0, i1] : 0 <=
 * i0 <= 4 and 0 <= i1 <= 9 }".
 */
void Stage::ChangeDomain(Stage *other, int level) {
  auto indices =
      optim::CollectTensorIndex(&(other->expr_), this->tensor()->name);
  if (indices.empty()) {
    return;
  }
  std::string this_domain = isl_set_to_str(this->domain().get());
  isl::ctx this_ctx = domain_.ctx();
  auto dim_names = isl_get_dim_names(domain_.get());
  auto map_names = isl_get_dim_names(other->transform().get(), isl_dim_out);
  std::set<std::string> uniq_names;
  for (int i = 0; i <= level; i++) {
    uniq_names.insert(map_names[i].substr(0, 1));
  }
  // The new level is the compute level of original domain axis(i, j, k, ...)
  // instead of transformed axis(i_outer, i_inner, j, k, ...)
  level = uniq_names.size() - 1;
  for (int i = 0; i <= level; i++) {
    auto _minv_maxv_ = isl_set_get_axis_range(domain_.get(), i);
    auto &minv = std::get<0>(_minv_maxv_);
    auto &maxv = std::get<1>(_minv_maxv_);
    int min_iv = minv.get_num_si();
    int max_iv = maxv.get_num_si();
    auto _minv2_maxv2_ = isl_set_get_axis_range(other->domain().get(), i);
    auto &minv2 = std::get<0>(_minv2_maxv2_);
    auto &maxv2 = std::get<1>(_minv2_maxv2_);
    int min_tar = minv2.get_num_si();
    int max_tar = maxv2.get_num_si();
    // Change each dim's range.
    // e.g., from "0 <= i0 <= 9" to "0 <= i0 <= 4"
    utils::Replace(&this_domain,
                   std::to_string(min_iv) + " <= " + dim_names[i] +
                       " <= " + std::to_string(max_iv),
                   std::to_string(min_tar) + " <= " + dim_names[i] +
                       " <= " + std::to_string(max_tar));
  }
  VLOG(3) << "Final changed domain is: " << this_domain;
  isl::set res_set(this_ctx, this_domain);
  domain_ = res_set;
}

/**
 * Edit temp tensor's shape, its buffer's shape and index when doing ComputeAt2.
 * @param level The level of dims to be changed.
 * For example, when this->domain_ is "{ [i0, i1] : 0 <= i0 <= 9 and 0 <= i1 <=
 * 9 }", and 1st loop is binded to threadIdx.x, then i0 will be erased in this
 * temp tensor's axes.
 */
void Stage::EditTempTensor(Stage *other, int level) {
  auto bind_info = other->forloop_infos();
  auto transform_domain_names = axis_names();
  std::set<std::string> erase_var;
  std::string tensor_name = this->tensor()->name;
  for (int i = 0; i <= level; i++) {
    if (isl_is_removed_axis(this->transformed_domain().get(), i)) {
      continue;
    }
    int new_i = i - isl_get_preceding_removed_axes_counts(
                        this->transformed_domain().get(), i);
    if (bind_info.count(new_i) != 0) {
      if (bind_info[new_i].for_type == ir::ForType::GPUThread &&
          (this->scope() == ScopeKind::kShared)) {
        continue;
      }
    }
    // Iterators of loop within level will be erased.
    auto related_dim_in = GetRelatedInputAxes(
        this->transform(), this->domain(), {transform_domain_names[i]});
    for (auto &j : related_dim_in) {
      erase_var.insert(j);
    }
  }
  std::set<std::string> undo_erase_var;
  // Beyond level, if the loop is binded to certain thread/block, it will also
  // be earsed.
  for (int i = level + 1; i < transform_domain_names.size(); i++) {
    if (isl_is_removed_axis(this->transformed_domain().get(), i)) {
      continue;
    }
    int new_i = i - isl_get_preceding_removed_axes_counts(
                        this->transformed_domain().get(), i);
    if (bind_info.count(new_i) != 0) {
      if (bind_info[new_i].for_type == ir::ForType::GPUBlock &&
          (this->scope() == ScopeKind::kShared ||
           this->scope() == ScopeKind::kLocal)) {
        auto related_dim_in = GetRelatedInputAxes(
            this->transform(), this->domain(), {transform_domain_names[i]});
        for (auto &j : related_dim_in) {
          erase_var.insert(j);
        }
      } else if (bind_info[new_i].for_type == ir::ForType::GPUThread &&
                 (this->scope() == ScopeKind::kLocal)) {
        auto related_dim_in = GetRelatedInputAxes(
            this->transform(), this->domain(), {transform_domain_names[i]});
        for (auto &j : related_dim_in) {
          erase_var.insert(j);
        }
      } else {
        auto related_dim_in = GetRelatedInputAxes(
            this->transform(), this->domain(), {transform_domain_names[i]});
        for (auto &j : related_dim_in) {
          undo_erase_var.insert(j);
        }
      }
    } else {
      auto related_dim_in = GetRelatedInputAxes(
          this->transform(), this->domain(), {transform_domain_names[i]});
      for (auto &j : related_dim_in) {
        undo_erase_var.insert(j);
      }
    }
  }
  std::vector<std::string> erase_var_vec;
  for (auto &i : erase_var) {
    if (undo_erase_var.count(i) == 0) {
      erase_var_vec.push_back(i);
    }
  }
  // Erase loop iterators.
  for (auto &j : erase_var_vec) {
    Var dim_var(j);
    for (auto &i : this->tensor()->new_indices) {
      optim::ReplaceVarWithExpr(&i, dim_var, Expr(0));
    }
    optim::ReplaceVarWithExpr(&(other->expr_), dim_var, Expr(0), tensor_name);
  }
  // Store each loop's range.
  std::map<std::string, int> dim_to_range;
  std::vector<std::string> this_dim_names = isl_get_dim_names(domain_);
  for (int i = 0; i < this_dim_names.size(); i++) {
    auto _minv_maxv_ = isl_set_get_axis_range(domain_.get(), i);
    auto &minv = std::get<0>(_minv_maxv_);
    auto &maxv = std::get<1>(_minv_maxv_);
    int min_iv = minv.get_num_si();
    int max_iv = maxv.get_num_si();
    dim_to_range[this_dim_names[i]] = max_iv;
  }

  std::vector<Expr> new_shape;
  for (auto &i : this->tensor()->new_indices) {
    new_shape.push_back(ir::ir_utils::IRCopy(i));
  }
  for (auto &i : new_shape) {
    for (auto &j : dim_to_range) {
      Var dim_var(j.first);
      optim::ReplaceVarWithExpr(&i, dim_var, Expr(j.second));
    }
    i = ir::Add::Make(i, Expr(1));
    optim::Simplify(&i);
  }
  // Set new shape.
  VLOG(3) << "Tensor is : " << this->tensor()->name;
  for (auto &i : new_shape) {
    VLOG(3) << "In Temp Buffer, shape is: " << utils::GetStreamCnt(i);
  }
  this->tensor()->shape = new_shape;
  PADDLE_ENFORCE_EQ(this->tensor()->buffer.defined(),
                    true,
                    ::common::errors::InvalidArgument(
                        "Tensor buffer is not defined, please check."));
  this->tensor()->buffer->shape = new_shape;
}

void Stage::ComputeAt(Stage *other, int level) {
  isl::set this_domain(domain().ctx(), isl_set_to_str(domain().get()));
  isl::set target_domain(other->domain().ctx(),
                         isl_set_to_str(other->domain().get()));

  auto reduce_axes = origin_reduce_axis_names();
  for (auto &i : reduce_axes) {
    this_domain =
        isl::manage(isl_remove_axis_by_name(this_domain.release(), i.c_str()));
  }
  isl::map write_access = isl::manage(isl_set_identity(this_domain.release()));
  isl::map read_access = isl::manage(isl_set_identity(target_domain.release()));
  read_access = isl::manage(
      isl_map_set_tuple_name(read_access.release(),
                             isl_dim_out,
                             isl_set_get_tuple_name(domain().get())));
  int num_out_dim = isl_map_dim(read_access.get(), isl_dim_out);
  read_access = isl::manage(
      isl_map_remove_dims(read_access.release(), isl_dim_out, 0, num_out_dim));
  auto indices =
      optim::CollectTensorIndex(&(other->expr_), this->tensor()->name);
  RemoveDuplicate(indices);
  if (indices.empty()) {
    LOG(ERROR) << "No Access Relation between [" << other->id() << "] and ["
               << this->id() << "]! Please check.";
  }
  PADDLE_ENFORCE_EQ(
      indices.size(),
      1,
      ::common::errors::InvalidArgument(
          "The size of indices should be 1, but got %d", indices.size()));
  std::vector<std::string> target_dims = isl_get_dim_names(other->domain());
  std::set<std::string> target_dims_set;
  for (auto &i : target_dims) {
    target_dims_set.insert(i);
  }
  std::vector<std::string> index_names;
  for (auto &i : indices[0]) {
    std::string str_name = utils::GetStreamCnt(i);
    if (target_dims_set.count(str_name) > 0) {
      target_dims_set.erase(str_name);
      str_name = str_name + "' = " + str_name;
    }
    index_names.push_back(str_name);
  }

  // New Transform = W.(R^-1).S
  // W is the write access relation
  // R is the read access relation
  // S is the original schedule of Stage *other
  read_access = isl::manage(
      isl_map_add_dims(read_access.release(), isl_dim_out, index_names.size()));
  isl_set_dim_names(&read_access, isl_dim_out, index_names);
  read_access = isl::manage(
      isl_map_set_tuple_name(read_access.release(),
                             isl_dim_out,
                             isl_set_get_tuple_name(domain().get())));
  std::string read_access_str = isl_map_to_str(read_access.get());
  isl::map read_access2(read_access.ctx(), read_access_str);
  read_access2 = isl::manage(isl_map_reverse(read_access2.release()));

  auto new_map = isl::manage(
      isl_map_apply_range(write_access.release(), read_access2.release()));
  isl::map new_target_transform(other->transform().ctx(),
                                isl_map_to_str(other->transform().get()));
  auto target_map_dims =
      isl_get_dim_names(new_target_transform.get(), isl_dim_out);
  auto target_map_dims_in =
      isl_get_dim_names(new_target_transform.get(), isl_dim_in);
  // For axis out of the level, we don't copy their transform except for they
  // are related to axis within the level.
  std::vector<std::string> level_out_dims;
  std::set<std::string> related_output_dims_set;
  for (int i = 0; i <= level; i++) {
    level_out_dims.push_back(target_map_dims[i]);
    related_output_dims_set.insert(target_map_dims[i]);
  }
  auto related_input_dims = GetRelatedInputAxes(
      new_target_transform, other->domain(), level_out_dims);
  auto related_output_dims = GetRelatedOutputAxes(
      new_target_transform, other->domain(), related_input_dims);
  for (auto &i : related_output_dims) {
    related_output_dims_set.insert(i);
  }
  std::set<std::string> related_input_dims_set;
  for (auto &i : related_input_dims) {
    related_input_dims_set.insert(i);
  }
  for (auto &i : target_map_dims) {
    if (related_output_dims_set.count(i) == 0) {
      new_target_transform = isl::manage(isl_remove_axis_by_name(
          new_target_transform.release(), isl_dim_out, i.c_str()));
    }
  }

  for (auto &i : target_map_dims_in) {
    if (related_input_dims_set.count(i) == 0) {
      new_target_transform = isl::manage(
          isl_map_add_dims(new_target_transform.release(), isl_dim_out, 1));
      int level = isl_map_dim(new_target_transform.get(), isl_dim_out);
      std::string dim_name_add = i + "' = " + i;
      new_target_transform =
          isl::manage(isl_map_set_dim_name(new_target_transform.release(),
                                           isl_dim_out,
                                           level - 1,
                                           dim_name_add.c_str()));
    }
  }
  new_target_transform = isl::manage(isl_map_set_tuple_name(
      new_target_transform.release(), isl_dim_out, other->id()));

  isl::map f_target_transform(other->transform().ctx(),
                              isl_map_to_str(new_target_transform.get()));
  auto trans_res = isl::manage(
      isl_map_apply_range(new_map.release(), f_target_transform.release()));
  trans_res = isl::manage(
      isl_map_set_tuple_name(trans_res.release(), isl_dim_out, this->id()));

  // When there are reduce axes, we need to add these axes manually
  if (!reduce_axes.empty()) {
    std::vector<std::string> reduce_axes_out;
    for (auto &i : reduce_axes) {
      reduce_axes_out.push_back(i + "' = " + i);
    }
    int map_dim_in = isl_map_dim(trans_res.get(), isl_dim_in);
    int map_dim_out = isl_map_dim(trans_res.get(), isl_dim_out);

    trans_res = isl::manage(
        isl_map_add_dims(trans_res.release(), isl_dim_in, reduce_axes.size()));
    for (int i = 0; i < reduce_axes.size(); i++) {
      trans_res = isl::manage(isl_map_set_dim_name(trans_res.release(),
                                                   isl_dim_in,
                                                   map_dim_in + i,
                                                   reduce_axes[i].c_str()));
    }
    trans_res = isl::manage(isl_map_add_dims(
        trans_res.release(), isl_dim_out, reduce_axes_out.size()));
    for (int i = 0; i < reduce_axes_out.size(); i++) {
      trans_res = isl::manage(isl_map_set_dim_name(trans_res.release(),
                                                   isl_dim_out,
                                                   map_dim_out + i,
                                                   reduce_axes_out[i].c_str()));
    }
    trans_res = isl::manage(
        isl_map_set_tuple_name(trans_res.release(), isl_dim_in, this->id()));
    trans_res = isl::manage(
        isl_map_set_tuple_name(trans_res.release(), isl_dim_out, this->id()));

    std::string trans_res_str = isl_map_to_str(trans_res.get());
    for (int i = 0; i < reduce_axes.size(); i++) {
      auto _minv_maxv_ = isl_set_get_axis_range(domain_.get(), i + map_dim_in);
      auto &minv = std::get<0>(_minv_maxv_);
      auto &maxv = std::get<1>(_minv_maxv_);
      int min_iv = minv.get_num_si();
      int max_iv = maxv.get_num_si();

      trans_res_str = trans_res_str.substr(0, trans_res_str.size() - 1) +
                      "and " + std::to_string(min_iv) +
                      " <= " + reduce_axes[i] +
                      " <= " + std::to_string(max_iv) + " }";
    }
    isl::map temp_trans(trans_res.ctx(), trans_res_str);
    trans_res = temp_trans;
  }

  VLOG(3) << "trans_res is : " << trans_res;

  {
    auto trans_dim_out = isl_get_dim_names(trans_res.get(), isl_dim_out);
    auto transformed_res = domain_.apply(trans_res);
    for (int i = level + 1; i < trans_dim_out.size(); i++) {
      auto _minv_maxv_ = isl_set_get_axis_range(transformed_res.get(), i);
      auto &minv = std::get<0>(_minv_maxv_);
      auto &maxv = std::get<1>(_minv_maxv_);
      int max_iv = maxv.get_num_si();
      int min_iv = minv.get_num_si();
      auto related_input_dims =
          GetRelatedInputAxes(trans_res, domain_, {trans_dim_out[i]}, true);
      if (max_iv != min_iv && related_input_dims.empty()) {
        trans_res = isl::manage(isl_remove_axis_by_name(
            trans_res.release(), isl_dim_out, trans_dim_out[i].c_str()));
      }
      VLOG(3) << "Input axis related to output axis [" << trans_dim_out[i]
              << "] (from " << min_iv << " to " << max_iv << ") is : ";
      for (auto &j : related_input_dims) {
        VLOG(3) << j << ", ";
      }
    }
  }
  VLOG(3) << "After removing redundant output axis, trans_res is : "
          << trans_res;
  transform_ = trans_res;
  PADDLE_ENFORCE_NOT_NULL(tensor_,
                          ::common::errors::InvalidArgument(
                              "Tensor should not be null, please check!"));

  ComputeAtRelation relation;
  relation.stage = other;
  relation.level = level;
  other->CtrlDepend(ir::Tensor(tensor()));

  PADDLE_ENFORCE_EQ(
      relation.IsCompatible(this),
      true,
      ::common::errors::InvalidArgument(
          "Cannot apply ComputeAt with level: %d from \n%s\n to \n%s.",
          level,
          isl_set_to_str(this->transformed_domain().get()),
          isl_set_to_str(other->transformed_domain().get())));
  compute_ats_[other->id()] = relation;
  for (int i = 0; i <= level; i++)
    AddForloopInfo(i,
                   StageForloopInfo{ir::ForType::Default, DeviceAPI::UNK, 0});
}

void Stage::ComputeAt2(Stage *other, int level) {
  // TODO(Superjomn) Check there are data dependency between `self` and `other`,
  // or the `ComputeAt` is meaningless.
  PADDLE_ENFORCE_GE(
      level,
      0,
      ::common::errors::InvalidArgument(
          "level param of ComputeAt2 must be >= 0. Please check!"));
  this->ChangeDomain(other, level);
  this->CopyTransform(other, level);
  this->ChangeIndex(other);
  PADDLE_ENFORCE_NOT_NULL(tensor_,
                          ::common::errors::InvalidArgument(
                              "Tensor should not be null, please check!"));
  other->CtrlDepend(ir::Tensor(tensor()));
  if (this->tensor()->buffer.defined()) {
    std::string t_name = this->tensor()->buffer->name;
    if (utils::EndsWith(t_name, "_read_cache") ||
        utils::EndsWith(t_name, "_write_cache")) {
      EditTempTensor(other, level);
    }
  }
  ComputeAtRelation relation;
  relation.stage = other;
  relation.level = level;
  other->CtrlDepend(ir::Tensor(tensor()));

  PADDLE_ENFORCE_EQ(
      relation.IsCompatible(this),
      true,
      ::common::errors::InvalidArgument(
          "Cannot apply ComputeAt2 with level: %d from \n%s\n to \n%s.",
          level,
          isl_set_to_str(this->transformed_domain().get()),
          isl_set_to_str(other->transformed_domain().get())));
  compute_ats_[other->id()] = relation;
}

void Stage::ComputeAt3(Stage *other, int level) {
  this->ChangeDomain(other, level);
  this->CopyTransform(other, level);
  this->ChangeIndex(other);
  PADDLE_ENFORCE_NOT_NULL(tensor_,
                          ::common::errors::InvalidArgument(
                              "Tensor should not be null, please check!"));
  other->CtrlDepend(ir::Tensor(tensor()));
  if (this->tensor()->buffer.defined()) {
    std::string t_name = this->tensor()->buffer->name;
    if (utils::EndsWith(t_name, "_read_cache") ||
        utils::EndsWith(t_name, "_write_cache")) {
      EditTempTensor(other, level);
    }
  }
}

void Stage::SimpleComputeAt(Stage *other, int level) {
  PADDLE_ENFORCE_NOT_NULL(tensor_,
                          ::common::errors::InvalidArgument(
                              "Tensor should not be null, please check!"));
  other->CtrlDepend(ir::Tensor(tensor()));
  if (this->tensor()->buffer.defined()) {
    std::string t_name = this->tensor()->buffer->name;
    if (utils::EndsWith(t_name, "_read_cache") ||
        utils::EndsWith(t_name, "_write_cache")) {
      EditTempTensor(other, level);
    }
  }
  ComputeAtRelation relation;
  relation.stage = other;
  relation.level = level;
  other->CtrlDepend(ir::Tensor(tensor()));

  PADDLE_ENFORCE_EQ(
      relation.IsCompatible(this),
      true,
      ::common::errors::InvalidArgument(
          "Cannot apply SimpleComputeAt with level: %d from \n%s\n to \n%s.",
          level,
          isl_set_to_str(this->transformed_domain().get()),
          isl_set_to_str(other->transformed_domain().get())));
  compute_ats_[other->id()] = relation;
  auto other_expr = other->expr();
  auto find_tensors =
      ir::ir_utils::CollectIRNodesWithoutTensor(other_expr, [&](const Expr *x) {
        return x->as_tensor() && x->as_tensor_ref()->name == tensor()->name;
      });
  if (!find_tensors.empty()) {
    for (int i = 0; i <= level; i++)
      AddForloopInfo(i,
                     StageForloopInfo{ir::ForType::Default, DeviceAPI::UNK, 0});
  }
}

std::tuple<Iterator, Iterator> Stage::Skew(const Iterator &i,
                                           const Iterator &j,
                                           int factor) {
  CINN_NOT_IMPLEMENTED
  Iterator i_new(i.id + "_skew");
  Iterator j_new(j.id + "_skew");

  return std::make_tuple(i_new, j_new);
}

Iterator Stage::Fuse(int level0, int level1) {
  AssertAxisIsNotLocked(level0);
  AssertAxisIsNotLocked(level1);
  auto dims = isl_get_dim_names(transformed_domain());
  PADDLE_ENFORCE_LT(level0,
                    dims.size(),
                    ::common::errors::InvalidArgument(
                        "level0 should be less than the size of dims"));
  PADDLE_ENFORCE_LT(level1,
                    dims.size(),
                    ::common::errors::InvalidArgument(
                        "level1 should be less than the size of dims"));

  Iterator iter0(dims[level0]);
  Iterator iter1(dims[level1]);

  return Fuse(iter0, iter1);
}

Iterator Stage::Fuse(const std::vector<int> &levels) {
  auto dims = isl_get_dim_names(transformed_domain());
  for (auto i : levels) {
    AssertAxisIsNotLocked(i);
    PADDLE_ENFORCE_LT(i,
                      dims.size(),
                      ::common::errors::InvalidArgument(
                          "level should be less than the size of dims"));
  }
  Iterator fused_axis(dims[levels[0]]);
  for (size_t i = 1; i < levels.size(); i++) {
    fused_axis = Fuse(fused_axis, Iterator(dims[levels[i]]));
  }
  return fused_axis;
}

Iterator Stage::Fuse(const std::string &level0, const std::string &level1) {
  return Fuse(Iterator(level0), Iterator(level1));
}

Iterator Stage::FuseDirect(const std::vector<int> &levels) {
  auto dims = isl_get_dim_names(transformed_domain());
  for (auto i : levels) {
    AssertAxisIsNotLocked(i);
    PADDLE_ENFORCE_LT(i,
                      dims.size(),
                      ::common::errors::InvalidArgument(
                          "level should be less than the size of dims"));
  }
  std::vector<Iterator> iterators;
  for (size_t i = 0; i < levels.size(); i++) {
    iterators.push_back(Iterator(dims[levels[i]]));
  }
  return Fuse(iterators);
}

Iterator Stage::Fuse(const std::vector<Iterator> &levels) {
  PADDLE_ENFORCE_GT(levels.size(),
                    1,
                    ::common::errors::InvalidArgument(
                        "The size of levels should be greater than 1"));
  std::vector<int> offsets;
  std::string new_iter_name;
  for (auto &level : levels) {
    int offset = isl_set_find_dim_by_name(
        transformed_domain().get(), isl_dim_set, level.id.c_str());
    if (!offsets.empty())
      PADDLE_ENFORCE_EQ(
          offsets.back() + 1,
          offset,
          ::common::errors::InvalidArgument(
              "level offsets.back and level offset should be adjacent"));
    AssertAxisIsNotLocked(offset);
    offsets.push_back(offset);
    new_iter_name += utils::StringFormat("%s_", level.id.c_str());
  }
  new_iter_name += "fused";

  // Aff { s[i,j,k] -> [j] } and get the j's max value
  // to apply something like { S[i,j] -> S[k]: k = i+j }
  auto from_dim_names = isl_get_dim_names(transform_, isl_dim_out);
  auto from_iters = NamesToIterators(from_dim_names);

  std::vector<int> iterator_max_val;
  for (auto &level : levels) {
    Aff aff(domain_.ctx(),
            id(),
            from_iters,
            std::vector<Iterator>({Iterator(level.id)}),
            {});
    int level_max_val =
        transformed_domain().max_val(aff.to_isl()).get_num_si() + 1;
    iterator_max_val.push_back(level_max_val);
  }

  // Map { s[i,j,k] -> s[n,k] : n = i * max_val + j }
  std::vector<Iterator> to_iters;
  {
    Iterator new_iter(new_iter_name);
    for (int i = 0; i < from_iters.size(); i++) {
      int offset = isl_set_find_dim_by_name(
          transformed_domain().get(), isl_dim_set, from_iters[i].id.c_str());
      if (i == offsets.back()) {
        to_iters.push_back(new_iter);
      } else if (i >= offsets.front() && i < offsets.back()) {
      } else {
        to_iters.push_back(from_iters[i]);
      }
    }
  }
  auto my_prod = [=](int a, int b) { return a * b; };
  std::vector<Condition> conds;
  conds.emplace_back(utils::StringFormat(
      "%s = floor(%s / %d)",
      levels.front().id.c_str(),
      new_iter_name.c_str(),
      static_cast<int>(std::accumulate(
          iterator_max_val.begin() + 1, iterator_max_val.end(), 1, my_prod))));
  conds.emplace_back(utils::StringFormat("%s = %s mod %d",
                                         levels.back().id.c_str(),
                                         new_iter_name.c_str(),
                                         iterator_max_val.back()));

  for (int i = 1; i < levels.size() - 1; i++) {
    conds.emplace_back(utils::StringFormat(
        "%s = floor(%s / %d) mod %d",
        levels[i].id.c_str(),
        new_iter_name.c_str(),
        static_cast<int>(std::accumulate(iterator_max_val.begin() + i + 1,
                                         iterator_max_val.end(),
                                         1,
                                         my_prod)),
        iterator_max_val[i]));
  }

  Map trans(domain_.ctx(), id(), from_iters, to_iters, conds, id());

  VLOG(2) << "In Fuse, trans is : " << trans.to_isl();
  VLOG(2) << "Before Fuse, transform_ is : " << transform_;
  transform_ = transform_.apply_range(trans.to_isl());
  {
    std::vector<std::string> iter_names;
    for (auto &iter : to_iters) iter_names.push_back(iter.id);

    isl_set_dim_names(&transform_, isl_dim_out, iter_names);
  }
  VLOG(2) << "After Fuse, transform_ is : " << transform_;

  return Iterator(new_iter_name);
}

/*
 * Fuse use a polyhedral transform.
 */
Iterator Stage::Fuse(const Iterator &level0, const Iterator &level1) {
  int offset0 = isl_set_find_dim_by_name(
      transformed_domain().get(), isl_dim_set, level0.id.c_str());
  int offset1 = isl_set_find_dim_by_name(
      transformed_domain().get(), isl_dim_set, level1.id.c_str());
  PADDLE_ENFORCE_EQ(offset1,
                    offset0 + 1,
                    ::common::errors::InvalidArgument(
                        "level offset1 and level offset0 + 1 should be same"));
  AssertAxisIsNotLocked(offset0);
  AssertAxisIsNotLocked(offset1);

  auto new_iter_name =
      utils::StringFormat("%s_%s_fused", level0.id.c_str(), level1.id.c_str());

  // Aff { s[i,j,k] -> [j] } and get the j's max value
  // to apply something like { S[i,j] -> S[k]: k = i+j }
  auto from_dim_names = isl_get_dim_names(transform_, isl_dim_out);
  auto from_iters = NamesToIterators(from_dim_names);

  Aff aff(domain_.ctx(),
          id(),
          from_iters,
          std::vector<Iterator>({Iterator(level1.id)}),
          {});

  int level1_max_val =
      transformed_domain().max_val(aff.to_isl()).get_num_si() + 1;

  // Map { s[i,j,k] -> s[n,k] : n = i * max_val + j }
  std::vector<Iterator> to_iters;
  {
    Iterator new_iter(new_iter_name);
    for (auto &iter : from_iters) {
      if (iter == level0) {
      } else if (iter == level1) {
        to_iters.push_back(new_iter);
      } else {
        to_iters.push_back(iter);
      }
    }
  }

  std::vector<Condition> conds;
  conds.emplace_back(utils::StringFormat("%s = %s * %d + %s",
                                         new_iter_name.c_str(),
                                         level0.id.c_str(),
                                         level1_max_val,
                                         level1.id.c_str()));

  Map trans(domain_.ctx(), id(), from_iters, to_iters, conds, id());

  transform_ = transform_.apply_range(trans.to_isl());
  {
    std::vector<std::string> iter_names;
    for (auto &iter : to_iters) iter_names.push_back(iter.id);

    isl_set_dim_names(&transform_, isl_dim_out, iter_names);
  }

  return Iterator(new_iter_name);
}

std::vector<std::string> Stage::input_statements() const {
  if (!expr_.defined()) return {};
  VLOG(3) << "stage " << id() << " expr: " << expr_;
  auto load_exprs = ir::ir_utils::CollectIRNodes(
      expr_, [](const Expr *x) { return x->As<ir::Load>(); });
  std::set<std::string> statements;
  for (auto &expr : load_exprs) {
    auto *load_node = expr.As<ir::Load>();
    PADDLE_ENFORCE_NOT_NULL(load_node,
                            ::common::errors::InvalidArgument(
                                "Load node should not be null, please check!"));
    auto *tensor = load_node->tensor.As<ir::_Tensor_>();
    PADDLE_ENFORCE_NOT_NULL(tensor,
                            ::common::errors::InvalidArgument(
                                "Tensor should not be null, please check!"));
    auto tensor_name = tensor->name;
    if (tensor_name != id()) statements.insert(tensor_name);
  }
  return std::vector<std::string>(statements.begin(), statements.end());
}

std::string InnerName(const std::string &name) { return name + "_inner"; }
std::string OuterName(const std::string &name) { return name + "_outer"; }
std::string InnerName(const Iterator &iterator) {
  return InnerName(iterator.id);
}
std::string OuterName(const Iterator &iterator) {
  return OuterName(iterator.id);
}

const char *Stage::id() const { return isl_set_get_tuple_name(domain_.get()); }

std::tuple<Iterator, Iterator> Stage::Split(const std::string &level,
                                            int factor) {
  return std::move(Split(Iterator(level), factor));
}

Shared<Stage> Stage::New(const isl::set &domain,
                         Expr expr,
                         ir::_Tensor_ *tensor) {
  return new Stage(domain, expr, tensor);
}

std::vector<ComputeAtRelation> Stage::compute_ats() const {
  std::vector<ComputeAtRelation> xs;
  for (auto &item : compute_ats_) xs.push_back(item.second);
  return xs;
}

void Stage::ShowISL() const {
  LOG(INFO) << "Tensor " << id()
            << " domain is: " << isl_set_to_str(domain().get());
  LOG(INFO) << "transformed_domain is: "
            << isl_set_to_str(transformed_domain().get());
  LOG(INFO) << "transform is: " << isl_map_to_str(transform().get());
}

bool ComputeAtRelation::IsCompatible(Stage *self) {
  PADDLE_ENFORCE_GE(level,
                    0,
                    ::common::errors::InvalidArgument(
                        "level should be greater than or equal to 0"));
  PADDLE_ENFORCE_EQ(!self->domain().is_null(),
                    true,
                    ::common::errors::InvalidArgument(
                        "Domain should not be null, please check!"));
  PADDLE_ENFORCE_EQ(!stage->domain().is_null(),
                    true,
                    ::common::errors::InvalidArgument(
                        "Domain should not be null, please check!"));

  PADDLE_ENFORCE_LE(level,
                    isl_set_dim(self->transformed_domain().get(), isl_dim_set),
                    ::common::errors::InvalidArgument(
                        "level should be less than the dim of set"));
  PADDLE_ENFORCE_LE(level,
                    isl_set_dim(stage->transformed_domain().get(), isl_dim_set),
                    ::common::errors::InvalidArgument(
                        "level should be less than the dim of set"));

  int level_without_reduce_axis = level;
  if (self->tensor()) {
    level_without_reduce_axis = self->tensor()->domain.size() - 1;
  }
  std::vector<int> selected_dims;
  for (int i = 0; i <= std::min(level, level_without_reduce_axis); i++) {
    selected_dims.push_back(i);
  }

  auto stage_partial_set =
      SetGetDims(stage->transformed_domain(), selected_dims);
  auto self_partial_set = SetGetDims(self->transformed_domain(), selected_dims);

  stage_partial_set =
      isl::manage(isl_set_set_tuple_name(stage_partial_set.release(), ""));
  self_partial_set =
      isl::manage(isl_set_set_tuple_name(self_partial_set.release(), ""));

  // remove parameters, we don't consider them yet
  auto remove_params = [](isl::set &set) {
    int nparams = isl_set_dim(set.get(), isl_dim_param);
    if (nparams > 0) {
      set = isl::manage(
          isl_set_remove_dims(set.release(), isl_dim_param, 0, nparams));
    }
  };

  remove_params(stage_partial_set);
  remove_params(self_partial_set);

  VLOG(3) << "stage0.partial_set " << stage_partial_set;
  VLOG(3) << "stage1.partial_set " << self_partial_set;
  return isl_set_is_equal(stage_partial_set.get(), self_partial_set.get());
}

void Stage::Vectorize(int level, int factor) {
  AssertAxisIsNotLocked(level);
  PADDLE_ENFORCE_GE(level,
                    0,
                    ::common::errors::InvalidArgument(
                        "level should be greater than or equal to 0"));
  PADDLE_ENFORCE_LT(level,
                    n_out_dims(),
                    ::common::errors::InvalidArgument(
                        "level should be less than the dim of out dims"));
  PADDLE_ENFORCE_GT(
      factor,
      0,
      ::common::errors::InvalidArgument("factor should be greater than 0"));
  if (factor == 1) {
    VLOG(3) << "Vectorize-factor 1 has no sense, skip it";
    return;
  }
  auto transformed_domain = this->transformed_domain();
  if (isl_is_removed_axis(transformed_domain.get(), level)) {
    VLOG(3) << "Vectorizing for-1 has no sense, skip it";
    return;
  }
  int removed_axes_counts =
      isl_get_preceding_removed_axes_counts(transformed_domain.get(), level);
  VLOG(3) << "removed_axes_counts are " << removed_axes_counts
          << " before axis " << ith_dim_name(level);
  VLOG(3) << "vectorize level: " << level - removed_axes_counts
          << ", factor: " << factor;
  vectorize_info_.set(level - removed_axes_counts /*inner*/, factor);
}

void Stage::Vectorize(const std::string &axis, int factor) {
  auto dims = isl_get_dim_names(transformed_domain());
  auto it = std::find(dims.begin(), dims.end(), axis);
  PADDLE_ENFORCE_EQ(
      it != dims.end(),
      true,
      ::common::errors::InvalidArgument("No dimension called %s.", axis));
  Vectorize(std::distance(dims.begin(), it), factor);
}

void Stage::Vectorize(const Iterator &axis, int factor) {
  return Vectorize(axis.id, factor);
}

void Stage::Parallel(const std::string &axis) {
  auto dims = isl_get_dim_names(transformed_domain());
  auto it = std::find(dims.begin(), dims.end(), axis);
  PADDLE_ENFORCE_EQ(
      it != dims.end(),
      true,
      ::common::errors::InvalidArgument("No dimension called %s.", axis));
  Parallel(std::distance(dims.begin(), it));
}

void Stage::Parallel(const Iterator &axis) { return Parallel(axis.id); }

void Stage::Parallel(int level) {
  PADDLE_ENFORCE_GE(level,
                    0,
                    ::common::errors::InvalidArgument(
                        "level should be greater than or equal to 0"));
  AssertAxisIsNotLocked(level);
  auto transformed_domain = this->transformed_domain();
  VLOG(3) << "transformed_domain" << transformed_domain;
  if (isl_is_removed_axis(transformed_domain.get(), level)) {
    VLOG(3) << "Paralleling for-1 has no sense, skip it";
    return;
  }
  int removed_axes_counts =
      isl_get_preceding_removed_axes_counts(transformed_domain.get(), level);
  VLOG(3) << "removed_axes_counts are " << removed_axes_counts
          << " before axis " << ith_dim_name(level);
  parallel_info_.insert(level - removed_axes_counts);
}

void Stage::Unroll(int level) {
  PADDLE_ENFORCE_GE(level,
                    0,
                    ::common::errors::InvalidArgument(
                        "level should be greater than or equal to 0"));
  AssertAxisIsNotLocked(level);
  auto transformed_domain = this->transformed_domain();
  if (isl_is_removed_axis(transformed_domain.get(), level)) {
    VLOG(1) << "Unrolling for-1 has no sense, skip it";
    return;
  }
  int removed_axes_counts =
      isl_get_preceding_removed_axes_counts(transformed_domain.get(), level);
  VLOG(3) << "removed_axes_counts are " << removed_axes_counts
          << " before axis " << ith_dim_name(level);
  unroll_info_.insert(level - removed_axes_counts);
}

std::string Stage::ith_dim_name(int level) {
  auto dims = isl_get_dim_names(transformed_domain());
  PADDLE_ENFORCE_LT(level,
                    dims.size(),
                    ::common::errors::InvalidArgument(
                        "level should be less than the size of dims"));
  return dims[level];
}

Iterator Stage::ith_iterator(int level) {
  return Iterator(ith_dim_name(level));
}

isl::set Stage::transformed_domain() const {
  PADDLE_ENFORCE_EQ(!domain_.is_null(),
                    true,
                    ::common::errors::InvalidArgument(
                        "Domain should not be null, please check!"));
  PADDLE_ENFORCE_EQ(!transform_.is_null(),
                    true,
                    ::common::errors::InvalidArgument(
                        "Transform should not be null, please check!"));
  return domain_.apply(transform_);
}

std::vector<std::pair<std::string, std::string>> ExtractExtraDepLinksFromStages(
    const std::vector<Stage *> &stages) {
  std::vector<std::pair<std::string, std::string>> extra_links;
  for (auto &stage : stages) {
    for (auto &tensor : stage->ctrl_depends()) {
      VLOG(1) << "get extra stage: " << tensor->name << " -> " << stage->id();
      extra_links.emplace_back(tensor->name, stage->id());
    }
  }

  return extra_links;
}

void Stage::Unroll(const std::string &level) {
  auto dim_names = axis_names();
  auto it = std::find(dim_names.begin(), dim_names.end(), level);
  int l = std::distance(dim_names.begin(), it);
  AssertAxisIsNotLocked(l);
  Unroll(l);
}

void Stage::Unroll(const Iterator &level) {
  auto dim_names = axis_names();
  auto it = std::find(dim_names.begin(), dim_names.end(), level.id);
  int l = std::distance(dim_names.begin(), it);
  AssertAxisIsNotLocked(l);
  Unroll(l);
}

std::vector<std::string> Stage::axis_names() const {
  return isl_get_dim_names(transformed_domain());
}

std::vector<std::string> Stage::origin_reduce_axis_names() {
  auto reduce_axis_var = this->tensor()->reduce_axis;
  std::vector<std::string> reduce_axis_names;
  for (auto &i : reduce_axis_var) {
    reduce_axis_names.push_back(i->name);
  }
  return reduce_axis_names;
}

void Stage::Bind(int level, const std::string &axis) {
  PADDLE_ENFORCE_LT(level,
                    n_out_dims(),
                    ::common::errors::InvalidArgument(
                        "level should be less than the dim of out dims"));
  LockAxis(level);

  if (axis == "threadIdx.x" || axis == "threadIdx.y" || axis == "threadIdx.z") {
    uint8_t offset = axis.back() - 'x';
    AddForloopInfo(
        level,
        StageForloopInfo{ir::ForType::GPUThread, DeviceAPI::GPU, offset});
  } else if (axis == "blockIdx.x" || axis == "blockIdx.y" ||
             axis == "blockIdx.z") {
    uint8_t offset = axis.back() - 'x';
    AddForloopInfo(
        level, StageForloopInfo{ir::ForType::GPUBlock, DeviceAPI::GPU, offset});
  } else {
    CINN_NOT_IMPLEMENTED
  }
}

Iterator Stage::axis(int i) const {
  auto names = axis_names();
  PADDLE_ENFORCE_LT(i,
                    names.size(),
                    ::common::errors::InvalidArgument(
                        "i should be less than the size of names"));
  return Iterator(names[i]);
}
Iterator Stage::axis(const std::string &i) const {
  auto names = axis_names();
  auto it = std::find(names.begin(), names.end(), i);
  PADDLE_ENFORCE_EQ(
      it != names.end(),
      true,
      ::common::errors::InvalidArgument("No dimension called %s.", i));
  return Iterator(*it);
}

bool Stage::has_expression() const {
  PADDLE_ENFORCE_NOT_NULL(tensor_,
                          ::common::errors::InvalidArgument(
                              "Tensor should not be null, please check!"));
  return tensor_->has_expression();
}

namespace {

/**
 * Replace the reader of a cache tensor to tensor.
 */
struct CacheReplaceMutator : public ir::IRMutator<> {
  std::string tensor_name;
  ir::Tensor cache;
  bool read_or_write{};

  /**
   * construct
   * @param tensor_name name of the tensor to cache
   * @param cache the cache
   * @param read_or_write read or write cache
   */
  CacheReplaceMutator(const std::string &tensor_name,
                      ir::Tensor cache,
                      bool read_or_write)
      : tensor_name(tensor_name), cache(cache), read_or_write(read_or_write) {}

  void operator()(Expr *expr) { ir::IRMutator<>::Visit(expr, expr); }

 private:
  void Visit(const ir::_Tensor_ *op, Expr *expr) override {
    if (to_mutate_ && tensor_name == op->name) {
      *expr = cache;
    }
  }

  void Visit(const ir::Load *op, Expr *expr) override {
    auto *node = expr->As<ir::Load>();
    PADDLE_ENFORCE_NOT_NULL(
        node->tensor.as_tensor(),
        ::common::errors::InvalidArgument(
            "Node tensor should not be null, please check!"));
    auto *tensor = node->tensor.as_tensor();
    for (auto &index : node->indices) {
      ir::IRMutator<>::Visit(&index, &index);
    }
    ir::IRMutator<>::Visit(&node->tensor, &node->tensor);
  }

  bool to_mutate_{true};
};
}  // namespace

void CacheReadWriteReplace(const std::vector<ir::Tensor> &readers,
                           ir::Tensor cache_tensor,
                           std::string origin_tensor_name) {
  for (auto k : readers) {
    auto op = k->operation->as<ir::ComputeOp>()->body;
    for (auto j : op) {
      CacheReplaceMutator(origin_tensor_name, cache_tensor, true /*read*/)(&j);
    }
  }
}

void Stage::SetBuffer(const std::string &memory_type) {
  tensor_->WithBuffer(memory_type, "_" + this->tensor_->name + "_temp_buffer");
  if (memory_type == "shared") {
    this->SetScope(ScopeKind::kShared);
  } else if (memory_type == "local") {
    this->SetScope(ScopeKind::kLocal);
  } else if (memory_type == "global") {
    this->SetScope(ScopeKind::kGlobal);
  } else {
    CINN_NOT_IMPLEMENTED
  }
  if (tensor_->buffer.defined()) {
    VLOG(2) << "Has tensor_->buffer: " << tensor_->buffer->name;
  } else {
    VLOG(2) << "No tensor_->buffer";
  }
}

void Stage::ComputeInline() {
  PADDLE_ENFORCE_NOT_NULL(tensor_,
                          ::common::errors::InvalidArgument(
                              "Tensor should not be null, please check!"));
  meta.compute_inline = true;
}

void Stage::DisableComputeInline() {
  PADDLE_ENFORCE_NOT_NULL(tensor_,
                          ::common::errors::InvalidArgument(
                              "Tensor should not be null, please check!"));
  meta.compute_inline = false;
}

void Stage::ShareBufferWith(Stage *other) {
  PADDLE_ENFORCE_NOT_NULL(tensor_,
                          ::common::errors::InvalidArgument(
                              "Tensor should not be null, please check!"));
  PADDLE_ENFORCE_EQ(
      !other->meta.compute_inline,
      true,
      ::common::errors::InvalidArgument(
          "Other meta compute inline should be null, please check!"));
  PADDLE_ENFORCE_EQ(!meta.compute_inline,
                    true,
                    ::common::errors::InvalidArgument(
                        "Meta compute inline should be null, please check!"));

  meta.tensors_to_share_buffer_with.insert(other->id());
  other->meta.tensors_to_share_buffer_with.insert(tensor_->name);
}

isl_map *__isl_give GatherAccesses(Stage *stage,
                                   const std::string &tensor_name) {
  PADDLE_ENFORCE_NOT_NULL(
      stage->tensor_,
      ::common::errors::InvalidArgument(
          "Stage tensor should not be null, please check!"));
  auto loads =
      ir::ir_utils::CollectIRNodes(stage->tensor_->body(), [&](const Expr *x) {
        return x->As<ir::Load>() &&
               x->As<ir::Load>()->tensor.as_tensor()->name == tensor_name;
      });

  auto vars = stage->tensor_->axis_with_reduce();

  std::string in_tuple_name = stage->tensor_->name;
  std::string out_tuple_name = tensor_name;
  std::vector<std::string> in_dim_names, out_loads;
  std::transform(vars.begin(),
                 vars.end(),
                 std::back_inserter(in_dim_names),
                 [](const Var &x) { return x->name; });
  std::transform(loads.begin(),
                 loads.end(),
                 std::back_inserter(out_loads),
                 [](const Expr &x) { return utils::GetStreamCnt(x); });

  isl_map *res = nullptr;
  for (auto &load : out_loads) {
    std::string repr =
        utils::StringFormat("{ %s[%s] -> %s }",
                            in_tuple_name.c_str(),
                            utils::Join(in_dim_names, ",").c_str(),
                            load.c_str());
    isl_map *access =
        isl_map_read_from_str(stage->domain().ctx().get(), repr.c_str());
    if (res) {
      res = isl_map_union(res, access);
    } else {
      res = access;
    }
  }

  return res;
}

void Stage::AddForloopInfo(int level, const StageForloopInfo &info) {
  cuda_bind_info_ = true;
  int num_levels = isl_map_dim(transform_.get(), isl_dim_out);
  PADDLE_ENFORCE_GE(level,
                    0,
                    ::common::errors::InvalidArgument(
                        "level should be greater than or equal to 0"));
  PADDLE_ENFORCE_LT(level,
                    num_levels,
                    ::common::errors::InvalidArgument(
                        "level should be less than the num of levels"));
  auto transformed_domain = this->transformed_domain();
  int removed_axes_counts =
      isl_get_preceding_removed_axes_counts(transformed_domain.get(), level);

  if (isl_is_removed_axis(transformed_domain.get(), level)) {
    // For scalar case, forloop info will be lost after for-1 and reduce-axis
    // elimination. We record the forloop info in the -1th level for backup.
    if (level == 0) {
      VLOG(3) << "add forloop_infos in the -1 level for backup";
      forloop_infos_[-1] = info;
    }
    VLOG(3) << "for-1 has no sense, skip it";
    return;
  }
  VLOG(3) << "removed_axes_counts are " << removed_axes_counts
          << " before axis " << ith_dim_name(level);
  forloop_infos_[level - removed_axes_counts] = info;
}

void Stage::CopyTransform(Stage *other, int level) {
  auto target_transform = RemoveAxesByInputNames(
      other->transform(), other->domain(), other->origin_reduce_axis_names());
  isl::set target_origin_domain(other->domain().ctx(),
                                isl_set_to_str(other->domain().get()));
  for (auto &i : other->origin_reduce_axis_names()) {
    target_origin_domain = isl::manage(
        isl_remove_axis_by_name(target_origin_domain.release(), i.c_str()));
  }
  std::string str_target_trans = isl_map_to_str(target_transform.get());
  std::string this_tensor_name = isl_set_get_tuple_name(domain_.get());
  isl::ctx this_ctx = domain_.ctx();
  isl::map temp_transform_(this_ctx, str_target_trans);

  auto this_map_dims = isl_get_dim_names(transform_.get(), isl_dim_in);
  auto target_map_dims = isl_get_dim_names(target_transform.get(), isl_dim_in);

  // Edit level. e.g. if A->Split(0,10) and B->CopyTransform(A,0), level should
  // increase to 1.
  isl::map temp_target_trans(this_ctx, str_target_trans);
  if (level + 1 < isl_map_dim(temp_target_trans.get(), isl_dim_out)) {
    std::string pivot_dim_out =
        isl_map_get_dim_name(temp_target_trans.get(), isl_dim_out, level + 1);
    std::vector<std::string> dim_out_level;
    for (int i = 0; i <= level; i++) {
      dim_out_level.push_back(
          isl_map_get_dim_name(temp_target_trans.get(), isl_dim_out, i));
    }
    auto related_dim_in = GetRelatedInputAxes(
        temp_target_trans, target_origin_domain, dim_out_level);
    auto related_dim_out = GetRelatedOutputAxes(
        temp_target_trans, target_origin_domain, related_dim_in);
    for (auto &i : related_dim_out) {
      if (i == pivot_dim_out) {
        this->CopyTransform(other, level + 1);
        return;
      }
    }
  } else if (level >= isl_map_dim(temp_target_trans.get(), isl_dim_out)) {
    LOG(ERROR) << "ComputeAt level: " << level
               << " is not less than the axis number : "
               << isl_map_dim(temp_target_trans.get(), isl_dim_out)
               << ", please check.";
  }

  //! When this->tensor's dim is more than other->tensor, we need to supplment
  //! dims.
  std::vector<std::string> sup_dims;
  for (int i = target_map_dims.size(); i < this_map_dims.size(); i++) {
    sup_dims.push_back(this_map_dims[i]);
  }
  //! Check the dim range in this domain and target domain. Corresponding dim's
  //! range must be equal.

  auto dim_names = isl_get_dim_names(domain_.get());
  std::set<std::string> this_dim_names;
  std::vector<std::string> erase_dim_names;
  for (int i = 0; i < isl_set_dim(domain_.get(), isl_dim_set); i++) {
    this_dim_names.insert(isl_set_get_dim_name(domain_.get(), isl_dim_set, i));
  }
  //! Delete redundant input dims in transform_ (e,g. B[i,j] ->
  //! CopyTransform(C[i,j,k]) , Redundant dim k will be deleted.)
  for (int i = 0; i < isl_map_dim(temp_transform_.get(), isl_dim_in); i++) {
    if (this_dim_names.count(
            isl_map_get_dim_name(temp_transform_.get(), isl_dim_in, i)) == 0) {
      temp_transform_ = isl::manage(
          isl_map_remove_dims(temp_transform_.release(), isl_dim_in, i, 1));
      i--;
    }
  }
  //! Check related output dims in transform_ and delete them (e,g. C[i,j,k] ->
  //! C[i,j,k1,k2] , Redundant output dim k1 nad k2 will be deleted.)
  std::string new_target_trans = isl_map_to_str(temp_transform_.get());
  for (int i = 0; i < isl_map_dim(temp_transform_.get(), isl_dim_out); i++) {
    std::string temp_dim =
        isl_map_get_dim_name(temp_transform_.get(), isl_dim_out, i);
    if (utils::Count(&new_target_trans, temp_dim) !=
        utils::Count(&str_target_trans, temp_dim)) {
      temp_transform_ = isl::manage(
          isl_map_remove_dims(temp_transform_.release(), isl_dim_out, i, 1));
      i--;
    }
  }
  //! Add dims
  if (level >= 0) {
    std::set<std::string> keep_names;
    int dim_size = isl_map_dim(temp_transform_.get(), isl_dim_out);
    for (int i = level + 1; i < dim_size; i++) {
      std::string temp =
          isl_map_get_dim_name(temp_transform_.get(), isl_dim_out, i);
      temp = temp.substr(0, 1);
      temp = temp + "' = " + temp;
      keep_names.insert(temp);
    }
    temp_transform_ = isl::manage(isl_map_remove_dims(temp_transform_.release(),
                                                      isl_dim_out,
                                                      level + 1,
                                                      dim_size - level - 1));
    for (auto i : keep_names) {
      VLOG(3) << "i in keep_names is: " << i;
      temp_transform_ = isl::manage(
          isl_map_add_dims(temp_transform_.release(), isl_dim_out, 1));
      temp_transform_ = isl::manage(isl_map_set_dim_name(
          temp_transform_.release(), isl_dim_out, level + 1, i.c_str()));
      level++;
    }
  }
  if (sup_dims.size() > 0) {
    int level_in = isl_map_dim(temp_transform_.get(), isl_dim_in);
    int level_out = isl_map_dim(temp_transform_.get(), isl_dim_out);
    for (auto i : sup_dims) {
      VLOG(3) << "i in sup_dims is: " << i;
      temp_transform_ = isl::manage(
          isl_map_add_dims(temp_transform_.release(), isl_dim_in, 1));
      temp_transform_ = isl::manage(isl_map_set_dim_name(
          temp_transform_.release(), isl_dim_in, level_in, i.c_str()));
      level_in++;
      std::string i_dim_out = i + "' = " + i;
      temp_transform_ = isl::manage(
          isl_map_add_dims(temp_transform_.release(), isl_dim_out, 1));
      temp_transform_ =
          isl::manage(isl_map_set_dim_name(temp_transform_.release(),
                                           isl_dim_out,
                                           level_out,
                                           i_dim_out.c_str()));
      level_out++;
    }
  }
  isl_map_set_tuple_name(
      temp_transform_.get(), isl_dim_in, this_tensor_name.c_str());
  isl_map_set_tuple_name(
      temp_transform_.get(), isl_dim_out, this_tensor_name.c_str());
  std::string res_trans = isl_map_to_str(temp_transform_.get());
  isl::map res_map(this_ctx, res_trans);
  VLOG(2) << "This domain is: " << isl_set_to_str(domain_.get());
  VLOG(2) << "After Copytransform this trans is : "
          << isl_map_to_str(res_map.get());
  VLOG(2) << "Target transform is : "
          << isl_map_to_str(other->transform().get());
  VLOG(2) << "CopyTransform Level is : " << level;
  transform_ = res_map;
}

void Stage::CopyLoopInfo(Stage *other) {
  // copy other stage's forloop_infos
  auto &target_forloop_infos = other->forloop_infos();
  auto target_transformed_domain = other->transformed_domain();
  std::vector<std::string> this_dim_names =
      isl_get_dim_names(transformed_domain());
  std::vector<std::string> target_dim_names =
      isl_get_dim_names(target_transformed_domain);

  for (auto &i : target_forloop_infos) {
    for (int j = 0; j < this_dim_names.size(); j++) {
      int new_i = poly::isl_get_original_axes_from_optimized_level(
          target_transformed_domain.get(), i.first);
      if (target_dim_names[new_i] == this_dim_names[j]) {
        this->AddForloopInfo(j, i.second);
      }
    }
  }
  // copy other stage's vectorize/unroll/parallel info
  auto &target_vectorize_info = other->vectorize_info();
  auto &target_unroll_info = other->unroll_info();
  auto &target_parallel_info = other->parallel_info();
  vectorize_info_ = target_vectorize_info;
  unroll_info_ = target_unroll_info;
  parallel_info_ = target_parallel_info;
}

void Stage::LockAxis(uint32_t level) {
  PADDLE_ENFORCE_LT(level,
                    n_out_dims(),
                    ::common::errors::InvalidArgument(
                        "level should be less than the dim of out dims"));
  locked_axis_.insert(level);
}

void Stage::UnlockAxis(uint32_t level) {
  PADDLE_ENFORCE_LT(level,
                    n_out_dims(),
                    ::common::errors::InvalidArgument(
                        "level should be less than the dim of out dims"));
  locked_axis_.erase(level);
}

bool Stage::is_axis_locked(uint32_t level) const {
  PADDLE_ENFORCE_LT(level,
                    n_out_dims(),
                    ::common::errors::InvalidArgument(
                        "level should be less than the dim of out dims"));
  return locked_axis_.count(level);
}

void Stage::AssertAxisIsNotLocked(uint32_t level) {
  PADDLE_ENFORCE_EQ(
      !is_axis_locked(level),
      true,
      ::common::errors::InvalidArgument(
          "The %d-th axis is locked, cannot perform schedule.", level));
}

int Stage::GetTransformedLevel(int level) {
  if (!compute_ats().empty()) {
    // The ComputeAt schedule will insert some consumer axis in the preceding of
    // this, so the raw before ComputeAt should add the numbers of axis
    // inserted.
    PADDLE_ENFORCE_EQ(
        compute_ats().size(),
        1UL,
        ::common::errors::InvalidArgument("compute_ats size should be 1"));
    auto &compute_at = compute_ats().front();
    return compute_at.level + level + 1;
  }

  // or just return the original.
  return level;
}

void Stage::CtrlDepend(const ir::Tensor &t) { ctrl_depends_.insert(t); }

const std::set<ir::Tensor> &Stage::ctrl_depends() const {
  return ctrl_depends_;
}

ir::Tensor Stage::LookupCtrlDepend(const std::string &tensor_name) const {
  auto it = std::find_if(
      ctrl_depends_.begin(), ctrl_depends_.end(), [&](const ir::Tensor &x) {
        return x->name == tensor_name;
      });
  if (it == ctrl_depends_.end()) return ir::Tensor();
  return *it;
}

Stage *_StageMap_::operator[](const ir::Tensor &tensor) {
  PADDLE_ENFORCE_GT(
      data_.count(tensor->name),
      0UL,
      ::common::errors::InvalidArgument(
          "StageMap has no stage for tensor [%s].", tensor->name));
  return data_[tensor->name].get();
}
const Stage *_StageMap_::operator[](const ir::Tensor &tensor) const {
  PADDLE_ENFORCE_GT(
      data_.count(tensor->name),
      0UL,
      ::common::errors::InvalidArgument(
          "StageMap has no stage for tensor [%s].", tensor->name));
  return data_.at(tensor->name).get();
}
Stage *_StageMap_::operator[](const ir::_Tensor_ *tensor) {
  PADDLE_ENFORCE_GT(
      data_.count(tensor->name),
      0UL,
      ::common::errors::InvalidArgument(
          "StageMap has no stage for tensor [%s].", tensor->name));
  return data_[tensor->name].get();
}
const Stage *_StageMap_::operator[](const ir::_Tensor_ *tensor) const {
  PADDLE_ENFORCE_GT(
      data_.count(tensor->name),
      0UL,
      ::common::errors::InvalidArgument(
          "StageMap has no stage for tensor [%s].", tensor->name));
  return data_.at(tensor->name).get();
}

Stage *_StageMap_::Insert(const ir::Tensor &key, Stage *stage) {
  PADDLE_ENFORCE_NOT_NULL(stage,
                          ::common::errors::InvalidArgument(
                              "Stage should not be null, please check!"));
  data_[key->name].Reset(stage);
  return stage;
}

Stage *_StageMap_::InsertLazily(const ir::Tensor &key) {
  if (data_.count(key->name)) return operator[](key);
  return Insert(key, ir::CreateStage(key).get());
}

Stage *_StageMap_::InsertLazily(const ir::Tensor &key, Stage *stage) {
  if (data_.count(key->name)) return operator[](key);
  PADDLE_ENFORCE_NOT_NULL(stage,
                          ::common::errors::InvalidArgument(
                              "Stage should not be null, please check!"));
  data_[key->name].Reset(stage);
  return stage;
}

Stage *_StageMap_::Lookup(const std::string &name) const {
  auto it = data_.find(name);
  if (it == data_.end()) return nullptr;
  return it->second.get();
}

}  // namespace poly
}  // namespace cinn
