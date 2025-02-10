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

#include "paddle/fluid/primitive/base/decomp_trans.h"
#include <regex>
#include "paddle/fluid/eager/api/utils/global_utils.h"
#include "paddle/fluid/imperative/amp_auto_cast.h"
#include "paddle/fluid/pir/dialect/operator/ir/api_builder.h"
#include "paddle/fluid/pir/dialect/operator/ir/control_flow_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/fluid/prim/utils/utils.h"
#include "paddle/fluid/primitive/base/primitive_ops.h"
#include "paddle/pir/include/core/builtin_dialect.h"
#include "paddle/pir/include/core/program.h"

COMMON_DECLARE_bool(prim_check_ops);
COMMON_DECLARE_bool(prim_enable_dynamic);
COMMON_DECLARE_string(prim_forward_blacklist);

using paddle::dialect::DenseTensorType;
using paddle::dialect::SelectedRowsType;

namespace paddle {

using Program = pir::Program;

// some outputs like xshape will no longer used after decomp, and those outputs
// will skip checking.
std::unordered_set<std::string> decomp_op_contain_none = {
    "pd_op.squeeze",
    "pd_op.unsqueeze",
    "pd_op.flatten",
    "pd_op.batch_norm",
    "pd_op.batch_norm_",
    "pd_op.dropout",
    "pd_op.instance_norm",
};
//

std::unordered_set<std::string> dynamic_shape_blacklist = {"pd_op.squeeze",
                                                           "pd_op.unsqueeze",
                                                           "pd_op.flatten",
                                                           "pd_op.eye",
                                                           "pd_op.diag"};

namespace {
std::set<std::string> StringSplit(const std::string& str) {
  std::istringstream iss(str);
  std::set<std::string> tokens;
  std::string token;

  while (std::getline(iss, token, ';')) {
    size_t startpos = token.find_first_not_of(' ');
    size_t endpos = token.find_last_not_of(' ');
    if ((startpos != std::string::npos) && (endpos != std::string::npos)) {
      token = token.substr(startpos, endpos - startpos + 1);
    } else if (startpos != std::string::npos) {
      token = token.substr(startpos);
    }
    tokens.insert(token);
  }
  return tokens;
}

void RemoveOp(pir::Block* block, pir::Operation* op) {
  bool remove_op = true;
  for (auto& item : op->results()) {
    if (item.HasOneUse()) {
      remove_op = false;
      break;
    }
  }
  if (remove_op) {
    auto op_iter = std::find(block->begin(), block->end(), *op);
    block->erase(op_iter);
  }
}

}  // namespace

static bool has_dynamic_shape(const phi::DDim& dims) {
  std::vector<int64_t> vec = common::vectorize<int64_t>(dims);
  if (std::find(vec.begin(), vec.end(), -1) != vec.end()) {
    return true;
  } else {
    return false;
  }
}

static const phi::DDim GetValueDims(pir::Value value) {
  pir::Type origin_type = value.type();
  if (!origin_type) {
    PADDLE_THROW(
        common::errors::InvalidArgument("The type of value is nullptr."));
  }
  auto getdims = [](pir::Type value_type) -> phi::DDim {
    if (value_type.isa<DenseTensorType>()) {
      return value_type.dyn_cast<DenseTensorType>().dims();
    } else if (value_type.isa<SelectedRowsType>()) {
      return value_type.dyn_cast<SelectedRowsType>().dims();
    } else {
      PADDLE_THROW(common::errors::InvalidArgument(
          "[Prim] Currently, we can only get shape for dense "
          "tensor."));
    }
  };
  phi::DDim value_dim;
  if (origin_type.isa<pir::VectorType>()) {
    pir::VectorType types = origin_type.dyn_cast<pir::VectorType>();
    // all tensor dim in VectorType must be the same, expect dynamic shape.
    for (size_t idx = 0; idx < types.size(); idx++) {
      value_dim = getdims(types[idx]);
      if (has_dynamic_shape(value_dim)) {
        return value_dim;
      }
    }
  } else {
    value_dim = getdims(origin_type);
  }
  return value_dim;
}

static phi::DataType GetValueDtype(pir::Value value) {
  if (value.type().isa<DenseTensorType>()) {
    return paddle::dialect::TransToPhiDataType(
        value.type().dyn_cast<DenseTensorType>().dtype());
  } else if (value.type().isa<SelectedRowsType>()) {
    return paddle::dialect::TransToPhiDataType(
        value.type().dyn_cast<SelectedRowsType>().dtype());
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "Currently, we can only get phi::DataType from DenseTensorType and "
        "SelectedRowsType."));
  }
}

static bool check_dynamic_shape(const pir::OpOperand& item,
                                const pir::Operation& op) {
  auto dims = GetValueDims(item.source());
  if (has_dynamic_shape(dims)) {
    VLOG(6) << "[Prim] Decomp op receives dynamic shape [" << dims
            << "] in inputs of op " << op.name();
    return true;
  } else {
    return false;
  }
}

bool has_decomp_rule(const pir::Operation& op) {
  pir::IrContext* ctx = pir::IrContext::Instance();
  pir::OpInfo op_info = ctx->GetRegisteredOpInfo(op.name());
  auto decomp_interface_impl =
      op_info.GetInterfaceImpl<paddle::dialect::DecompInterface>();
  return decomp_interface_impl != nullptr;
}

bool has_decomp_vjp(const pir::Operation& vjp_op) {
  pir::IrContext* ctx = pir::IrContext::Instance();
  pir::OpInfo vjp_op_info = ctx->GetRegisteredOpInfo(vjp_op.name());
  auto decomp_vjp_interface_impl =
      vjp_op_info.GetInterfaceImpl<paddle::dialect::DecompVjpInterface>();
  return decomp_vjp_interface_impl != nullptr;
}
void DecompProgram::check_ops() {
  auto primitives_set = GetPrimitiveOpNames();
  std::set<std::string> undecomposed_set;
  for (const auto& element : decomposed_prog_ops_set_) {
    if (primitives_set.find(element) == primitives_set.end() &&
        blacklist_.find(element) == blacklist_.end()) {
      undecomposed_set.insert(element);
    }
  }
  if (!undecomposed_set.empty()) {
    std::string decomposed_ops_stream;
    for (const auto& item : undecomposed_set) {
      decomposed_ops_stream.append(" ");
      decomposed_ops_stream.append(item);
    }
    PADDLE_THROW(common::errors::InvalidArgument(
        "[Prim] Currently, decomposed program "
        "should not contain none primitive ops: %s .",
        decomposed_ops_stream));
  }
  return;
}

bool DecompProgram::check_decomp_dynamic_shape(pir::Operation* op) {
  for (auto item : op->operands()) {
    auto value = item.source();
    // check if initialized in case of optional input.
    if (!paddle::dialect::IsEmptyValue(value)) {
      pir::Operation* prev_op = value.defining_op();
      if (prev_op && prev_op->name() == "builtin.combine") {
        for (pir::OpOperand& sub_item : prev_op->operands()) {
          if (check_dynamic_shape(sub_item, *op)) {
            return true;
          }
        }
      } else {
        if (check_dynamic_shape(item, *op)) {
          return true;
        }
      }
    }
  }
  return false;
}

void DecompProgram::check_decomp_outputs(
    const std::string& op_name,
    const std::vector<pir::Value>& orig_outs,
    const std::vector<pir::Value>& decomp_outs) {
  bool skip_invalid_op_check =
      decomp_op_contain_none.find(op_name) != decomp_op_contain_none.end();
  for (size_t i = 0; i < orig_outs.size(); i++) {
    if (skip_invalid_op_check &&
        (paddle::dialect::IsEmptyValue(orig_outs[i]) ||
         paddle::dialect::IsEmptyValue(decomp_outs[i]))) {
      VLOG(4) << "[Prim] Decomp op skip check of " << i
              << "-index output of op " << op_name;
    } else {
      PADDLE_ENFORCE(
          !paddle::dialect::IsEmptyValue(orig_outs[i]),
          common::errors::PreconditionNotMet(
              "[Prim] For op %s, its origin %d-index output is invalid",
              op_name,
              i));
      PADDLE_ENFORCE(
          !paddle::dialect::IsEmptyValue(decomp_outs[i]),
          common::errors::PreconditionNotMet(
              "[Prim] For op %s, its decomp %d-index output is invalid",
              op_name,
              i));
      auto orig_dtype = GetValueDtype(orig_outs[i]);
      auto decomp_dtype = GetValueDtype(decomp_outs[i]);

      PADDLE_ENFORCE(orig_dtype == decomp_dtype,
                     common::errors::PreconditionNotMet(
                         "[Prim] For op %s, its origin %d-index output dtype "
                         "%s is not equal to "
                         "decomp output dtype %s ",
                         op_name,
                         i,
                         orig_dtype,
                         decomp_dtype));

      auto orig_dim = GetValueDims(orig_outs[i]);
      auto decomp_dim = GetValueDims(decomp_outs[i]);

      PADDLE_ENFORCE(
          orig_dim.size() == decomp_dim.size(),
          common::errors::PreconditionNotMet(
              "[Prim] For op %s, its origin %d-index output rank of shape"
              "[%s] is not equal to "
              "decomp output rank of shape[%s] ",
              op_name,
              i,
              orig_dim,
              decomp_dim));

      if (has_dynamic_shape(orig_dim)) {
        VLOG(6) << "[Prim] Decomp op receives dynamic shape [" << orig_dim
                << "] in " << i << "-index output of origin op " << op_name;
      }
      if (has_dynamic_shape(decomp_dim)) {
        VLOG(6) << "[Prim] Decomp op receives dynamic shape [" << decomp_dim
                << "] in " << i << "-index output of decomp op " << op_name;
      }

      for (int j = 0; j < orig_dim.size(); j++) {
        if (orig_dim[j] != -1 && decomp_dim[j] != -1) {
          PADDLE_ENFORCE(
              orig_dim[j] == decomp_dim[j],
              common::errors::PreconditionNotMet(
                  "[Prim] For op %s, its origin %d-index output shape "
                  "[%s] is not equal to "
                  "decomp output shape [%s] ",
                  op_name,
                  i,
                  orig_dim,
                  decomp_dim));
        }
      }
    }
  }
  return;
}

std::vector<pir::Value> DecompProgram::format_decomp_res(
    const std::string& op_name,
    const std::vector<pir::Value>& orig_outs,
    const std::vector<std::vector<pir::Value>>& decomp_outs) {
  PADDLE_ENFORCE_EQ(
      orig_outs.size(),
      decomp_outs.size(),
      common::errors::PreconditionNotMet(
          "[Prim] For op %s, its origin output num %d is not equal to "
          "decomp output num %d ",
          op_name,
          orig_outs.size(),
          decomp_outs.size()));
  std::vector<pir::Value> new_decomp_outs(orig_outs.size());
  for (size_t i = 0; i < orig_outs.size(); i++) {
    if (orig_outs[i]) {
      PADDLE_ENFORCE_EQ(
          decomp_outs[i].size(),
          1,
          common::errors::PreconditionNotMet(
              "[Prim] For op %s, each element of decomp output num must "
              "be 1, but num of index %d is %d ",
              op_name,
              i,
              decomp_outs[i].size()));
      new_decomp_outs[i] = decomp_outs[i][0];
    }
  }
  return new_decomp_outs;
}

void DecompProgram::construct_dst_vars(
    const std::string& op_name,
    const std::vector<pir::Value>& orig_outs,
    const std::vector<pir::Value>& decomp_outs,
    std::unordered_map<pir::Value, int> orig_vars_dict,
    std::vector<pir::Value>* tar_vars) {
  PADDLE_ENFORCE_EQ(
      orig_outs.size(),
      decomp_outs.size(),
      common::errors::PreconditionNotMet(
          "[Prim] For op %s, its origin output num %d is not equal to "
          "decomp output num %d ",
          op_name,
          orig_outs.size(),
          decomp_outs.size()));
  for (size_t i = 0; i < orig_outs.size(); i++) {
    if (orig_vars_dict.find(orig_outs[i]) != orig_vars_dict.end()) {
      (*tar_vars)[orig_vars_dict[orig_outs[i]]] = decomp_outs[i];
    }
  }
}

std::vector<pir::Value> DecompProgram::get_dst_vars() {
  if (!paddle::prim::PrimCommonUtils::IsFwdPrimEnabled()) {
    return src_vars_;
  } else {
    return dst_vars_;
  }
}

bool DecompProgram::enable_decomp_by_filter(const std::string& op_name) {
  bool flag = true;

  if (!whitelist_.empty()) {
    if (whitelist_.find(op_name) == whitelist_.end()) {
      flag = false;
    }
  }
  auto from_flag_blacklist = StringSplit(FLAGS_prim_forward_blacklist);
  if (!from_flag_blacklist.empty())
    blacklist_.insert(from_flag_blacklist.begin(), from_flag_blacklist.end());
  if (!blacklist_.empty() && blacklist_.find(op_name) != blacklist_.end())
    flag = false;
  return flag;
}

std::vector<std::vector<pir::Value>> call_decomp_rule(pir::Operation* op) {
  paddle::dialect::DecompInterface decomp_interface =
      op->dyn_cast<paddle::dialect::DecompInterface>();
  PADDLE_ENFORCE(decomp_interface,
                 common::errors::InvalidArgument(
                     "[Prim] The decomp function is not registered in %s op ",
                     op->name()));
  std::vector<std::vector<pir::Value>> decomp_res = decomp_interface.Decomp(op);
  return decomp_res;
}

std::vector<std::vector<pir::Value>> call_decomp_vjp(pir::Operation* vjp_op) {
  paddle::dialect::DecompVjpInterface decomp_vjp_interface =
      vjp_op->dyn_cast<paddle::dialect::DecompVjpInterface>();
  PADDLE_ENFORCE(
      decomp_vjp_interface,
      common::errors::InvalidArgument(
          "[Prim] The decomp_vjp function is not registered in %s vjp_op ",
          vjp_op->name()));
  std::vector<std::vector<pir::Value>> decomp_res =
      decomp_vjp_interface.DecompVjp(vjp_op);
  return decomp_res;
}
std::vector<pir::Operation*> DecompProgram::parse_block_ops(pir::Block* block) {
  std::vector<pir::Operation*> ops_list;
  for (auto& op : *block) {
    ops_list.push_back(&op);
  }
  if (program_->block() != block || (start_index_ == 0 && end_index_ == -1)) {
    return ops_list;
  }

  VLOG(4) << "start_index_:  " << start_index_ << ", end_index_: " << end_index_
          << ", ops_list.size(): " << ops_list.size();
  int start_idx = std::max(start_index_, 0);
  int end_idx = (end_index_ == -1) ? ops_list.size() : end_index_;
  if (start_idx == end_idx) {
    return std::vector<pir::Operation*>();
  }
  PADDLE_ENFORCE_LT(start_idx,
                    end_idx,
                    common::errors::PreconditionNotMet(
                        "Required start_idx < end_idx in DecompProgram."));
  PADDLE_ENFORCE_LE(
      end_idx,
      ops_list.size(),
      common::errors::PreconditionNotMet(
          "Required end_idx <= block.ops().size() in DecompProgram."));
  return std::vector<pir::Operation*>(ops_list.begin() + start_idx,
                                      ops_list.begin() + end_idx);
}

void DecompProgram::decomp_program() {
  std::unordered_map<pir::Value, int> orig_vars_dict;
  for (size_t i = 0; i < src_vars_.size(); i++) {  // NOLINT
    orig_vars_dict[src_vars_[i]] = static_cast<int>(i);
  }
  std::ostringstream orig_prog_stream;
  program_->Print(orig_prog_stream);
  if (VLOG_IS_ON(4)) {
    std::cout << "[Prim] Origin program before decomp :\n"
              << orig_prog_stream.str() << std::endl;
  }

  if (!paddle::prim::PrimCommonUtils::IsFwdPrimEnabled()) {
    return;
  }
  std::vector<pir::Value> tar_vars(src_vars_.size());
  pir::Block* block = program_->block();
  {
    // NOTE(dev): Prim decomposed rules will call paddle::dialect::xx
    // api, which has amp strategy. But Prim already process cast operation
    // and we need to disable amp strategy here.
    paddle::imperative::AutoCastGuard guard(
        egr::Controller::Instance().GetCurrentAmpAttrs(),
        paddle::imperative::AmpLevel::O0);
    decomp_block(block, orig_vars_dict, tar_vars);
  }
  std::ostringstream decomp_prog_stream;
  program_->Print(decomp_prog_stream);
  if (VLOG_IS_ON(4)) {
    std::cout << "[Prim] New program after decomp :\n"
              << decomp_prog_stream.str() << std::endl;
  }
  if (FLAGS_prim_check_ops) {
    check_ops();
  }
  dst_vars_ = tar_vars;
  return;
}

void DecompProgram::decomp_block(
    pir::Block* block,
    const std::unordered_map<pir::Value, int>& orig_vars_dict,
    std::vector<pir::Value>& tar_vars) {  // NOLINT
  std::vector<pir::Operation*> ops_list = parse_block_ops(block);
  for (size_t i = 0; i < ops_list.size(); i++) {
    auto op = ops_list[i];
    if (op->name() == "pd_op.if") {
      auto& sub_true_block = op->dyn_cast<dialect::IfOp>().true_block();
      auto& sub_false_block = op->dyn_cast<dialect::IfOp>().false_block();
      decomp_block(&sub_true_block, orig_vars_dict, tar_vars);
      decomp_block(&sub_false_block, orig_vars_dict, tar_vars);
    } else if (op->name() == "pd_op.while") {
      auto& sub_body = op->dyn_cast<dialect::WhileOp>().body();
      decomp_block(&sub_body, orig_vars_dict, tar_vars);
    }
    bool enable_prim =
        has_decomp_rule(*op) && enable_decomp_by_filter(op->name());
    if (enable_prim && check_decomp_dynamic_shape(op) &&
        (!FLAGS_prim_enable_dynamic ||
         dynamic_shape_blacklist.find(op->name()) !=
             dynamic_shape_blacklist.end())) {
      enable_prim = false;
    }
    if (enable_prim) {
      VLOG(4) << "[Prim] decomp op name " << op->name();
      check_decomp_dynamic_shape(op);
      std::shared_ptr<pir::Builder> builder =
          paddle::dialect::ApiBuilder::Instance().GetBuilder();
      builder->set_insertion_point(op);

      int op_role = (op->attribute<pir::Int32Attribute>("op_role"))
                        ? op->attribute<pir::Int32Attribute>("op_role").data()
                        : -1;
      int chunk_id = (op->attribute<pir::Int32Attribute>("chunk_id"))
                         ? op->attribute<pir::Int32Attribute>("chunk_id").data()
                         : -1;
      pir::BuilderAttrGuard guard(builder, op_role, chunk_id);

      std::vector<std::vector<pir::Value>> decomp_res = call_decomp_rule(op);
      if (decomp_res.size() == 0) {
        // if we don't decomp this op, then leave it intact.
        continue;
      }
      std::vector<pir::Value> orig_outs = op->results();
      bool is_next_builtin_split_slice = false;

      for (size_t i = 0; i < orig_outs.size(); i++) {
        auto item = orig_outs[i];
        if (item.use_count() >= 1) {
          auto next_op = item.first_use().owner();

          if (next_op->name() == "builtin.slice") {
            is_next_builtin_split_slice = true;
            std::vector<pir::Operation*> slice_ops;
            for (auto it = item.use_begin(); it != item.use_end(); ++it) {
              slice_ops.push_back(it->owner());
            }
            for (size_t j = 0; j < slice_ops.size(); j++) {
              int attr_idx = slice_ops[j]
                                 ->attribute("index")
                                 .dyn_cast<pir::Int32Attribute>()
                                 .data();
              slice_ops[j]->ReplaceAllUsesWith(decomp_res[i][attr_idx]);
              RemoveOp(block, slice_ops[j]);
            }
          }

          if (next_op->name() == "builtin.split") {
            is_next_builtin_split_slice = true;

            check_decomp_outputs(
                next_op->name(), next_op->results(), decomp_res[i]);
            construct_dst_vars(next_op->name(),
                               next_op->results(),
                               decomp_res[i],
                               orig_vars_dict,
                               &tar_vars);

            next_op->ReplaceAllUsesWith(decomp_res[i]);
            RemoveOp(block, next_op);
          }
        }
      }
      if (!is_next_builtin_split_slice) {
        std::vector<pir::Value> standard_decomp_res =
            format_decomp_res(op->name(), orig_outs, decomp_res);
        check_decomp_outputs(op->name(), orig_outs, standard_decomp_res);
        construct_dst_vars(op->name(),
                           orig_outs,
                           standard_decomp_res,
                           orig_vars_dict,
                           &tar_vars);

        op->ReplaceAllUsesWith(standard_decomp_res);
      }
      RemoveOp(block, op);
    }
  }
  if (FLAGS_prim_check_ops) {
    for (auto& op : *block) {
      decomposed_prog_ops_set_.insert(op.name());
    }
  }
  for (size_t i = 0; i < tar_vars.size(); i++) {
    if (!tar_vars[i]) {
      tar_vars[i] = src_vars_[i];
    }
  }
  std::shared_ptr<pir::Builder> builder =
      paddle::dialect::ApiBuilder::Instance().GetBuilder();
  builder->SetInsertionPointToBlockEnd(block);
}

}  // namespace paddle
