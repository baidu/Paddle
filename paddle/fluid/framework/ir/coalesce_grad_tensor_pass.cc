//   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/coalesce_grad_tensor_pass.h"

#include <algorithm>
#include <string>

#include "paddle/common/flags.h"
#include "paddle/fluid/framework/details/multi_devices_helper.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
namespace paddle::framework {
class ProgramDesc;
class VarDesc;
}  // namespace paddle::framework

PHI_DEFINE_EXPORTED_double(
    fuse_parameter_memory_size,
    -1.0,  // MBytes
    "fuse_parameter_memory_size is up limited memory size(MB)"
    "of one group parameters' gradient which is the input "
    "of communication calling(e.g NCCLAllReduce). "
    "The default value is 0, it means that "
    "not set group according to memory_size.");
PHI_DEFINE_EXPORTED_int32(
    fuse_parameter_groups_size,
    1,
    "fuse_parameter_groups_size is the up limited size of one group "
    "parameters' gradient. "
    "The default value is a experimental result. If the "
    "fuse_parameter_groups_size is 1, it means that the groups size is "
    "the number of parameters' gradient. If the fuse_parameter_groups_size is "
    "-1, it means that there are only one group. The default value is 3, it is "
    "an experimental value.");

namespace paddle::framework::ir {
// unit of the FLAGS_fuse_parameter_memory_size.
static constexpr double kMB = 1048576.0;

// SetFuseParameterGroupsSize and SetFuseParameterMemorySize are used in unit
// test, because it is invalid that setting 'FLAGS_fuse_parameter_memory_size'
// and 'FLAGS_fuse_parameter_groups_size' in unit test.
void SetFuseParameterGroupsSize(int group_size) {
  FLAGS_fuse_parameter_groups_size = group_size;
}

int GetFuseParameterGroupsSize() { return FLAGS_fuse_parameter_groups_size; }

void SetFuseParameterMemorySize(double memory_size) {
  FLAGS_fuse_parameter_memory_size = memory_size;
}

double GetFuseParameterMemorySize() { return FLAGS_fuse_parameter_memory_size; }

class CoalesceGradTensorPass : public ir::Pass {
 protected:
  void ApplyImpl(ir::Graph *graph) const override {
    if (Get<size_t>(details::kNRanks) <= 1) {
      VLOG(6) << "The number of place is" << Get<size_t>(details::kNRanks)
              << ", there doesn't need apply FuseAllReduceOpPass.";
      return;
    }
    ir::Graph &result = *graph;
    details::ParamsAndGrads params_grads;
    RecordParamsAndGrads(result, &params_grads);

    ResetAttribute<details::ParamsAndGrads>(details::kParamsAndDenseGrads,
                                            &result);
    ResetAttribute<details::ParamsAndGrads>(details::kParamsAndSparseGrads,
                                            &result);
    ResetAttribute<details::GroupParamsAndGrads>(
        details::kGroupParamsAndDenseGrads, &result);

    VLOG(10) << "The number of params and grads is:" << params_grads.size();
    if (params_grads.empty()) {
      return;
    }

    auto &p_g_dense_grad =
        result.Get<details::ParamsAndGrads>(details::kParamsAndDenseGrads);
    auto &p_g_sparse_grad =
        result.Get<details::ParamsAndGrads>(details::kParamsAndSparseGrads);

    auto vars_info = GetVarInfo(result);
    for (auto &param_grad : params_grads) {
      if (IsDenseTensorType(GetTypeOfVar(vars_info, param_grad.second))) {
        p_g_dense_grad.emplace_back(param_grad);
      } else {
        p_g_sparse_grad.emplace_back(param_grad);
      }
    }

    VLOG(10) << "Dense grads: " << p_g_dense_grad.size()
             << ", Sparse grads: " << p_g_sparse_grad.size();
    if (p_g_dense_grad.empty()) {
      return;
    }

    auto num_of_p_g_dense_grad = p_g_dense_grad.size();
    auto &group_params_grads = result.Get<details::GroupParamsAndGrads>(
        details::kGroupParamsAndDenseGrads);
    // Note: the order of p_g_dense_grad may be changed by
    // SetGroupParamsAndGrads.
    SetGroupParamsAndGrads(vars_info, p_g_dense_grad, &group_params_grads);

    p_g_dense_grad.clear();
    p_g_dense_grad.reserve(num_of_p_g_dense_grad);
    for (auto &group_p_g : group_params_grads) {
      p_g_dense_grad.insert(
          p_g_dense_grad.end(), group_p_g.begin(), group_p_g.end());
    }
    PADDLE_ENFORCE_EQ(p_g_dense_grad.size(),
                      num_of_p_g_dense_grad,
                      common::errors::InvalidArgument(
                          "The number of dense grads is not consistent with "
                          "previous. Previous(%d), now(%d).",
                          p_g_dense_grad.size(),
                          num_of_p_g_dense_grad));

    auto &pinned_var_set =
        graph->GetOrInit<details::PinnedVars>(details::kPinnedVars);
    if (IsUnifiedDtype(p_g_dense_grad, vars_info)) {  // NOLINT
      RecordGradients(p_g_dense_grad, vars_info, &pinned_var_set);
      CoalesceTensors(vars_info, p_g_dense_grad, &result);
    } else {
      for (auto &sub_param_grad : group_params_grads) {
        RecordGradients(p_g_dense_grad, vars_info, &pinned_var_set);
        PADDLE_ENFORCE_EQ(
            IsUnifiedDtype(sub_param_grad, vars_info),
            true,
            common::errors::InvalidArgument("All gradient variable in "
                                            "kGroupParamsAndDenseGrads, must "
                                            "have same type."));
        CoalesceTensors(vars_info, sub_param_grad, &result);
      }
    }
  }

  void RecordGradients(
      const std::vector<std::pair<std::string, std::string>> &sub_param_grad,
      const std::unordered_map<std::string, std::vector<ir::Node *>> &vars_info,
      std::unordered_set<std::string> *pinned_var_set) const {
    // The Gradients should not be reused during memory optimization.
    for (auto &p_g : sub_param_grad) {
      auto iter = vars_info.find(p_g.second);
      PADDLE_ENFORCE_EQ(iter != vars_info.end(),
                        true,
                        common::errors::NotFound(
                            "Parameter@Grad %s is not found.", p_g.second));
      PADDLE_ENFORCE_EQ(
          !iter->second.empty(),
          true,
          common::errors::InvalidArgument(
              "Parameter@Grad %s's var node is empty.", p_g.second));
      for (auto it : iter->second) {
        PADDLE_ENFORCE_NOT_NULL(
            it->Var(),
            common::errors::InvalidArgument(
                "A node of Parameter@Grad %s does not hold variable.",
                p_g.second));
        pinned_var_set->insert(it->Var()->Name());
      }
      PADDLE_ENFORCE_EQ(
          IsDenseTensorType(GetTypeOfVar(vars_info, p_g.second)),
          true,
          common::errors::InvalidArgument(
              "Parameter@Grad %s is not phi::DenseTensor.", p_g.second));
    }
  }

  bool IsUnifiedDtype(
      const details::ParamsAndGrads &params_grads,
      const std::unordered_map<std::string, std::vector<ir::Node *>> &vars_info)
      const {
    if (params_grads.empty()) return true;
    auto dtype = GetDtypeOfVar(vars_info, params_grads.front().second);
    for (auto const &p_g : params_grads) {
      auto next_dtype = GetDtypeOfVar(vars_info, p_g.second);
      if (next_dtype != dtype) {
        return false;
      }
    }
    return true;
  }

  void CoalesceTensors(
      const std::unordered_map<std::string, std::vector<ir::Node *>> &vars_info,
      const details::ParamsAndGrads &params_grads,
      Graph *result) const {
    // Create a FusedVarsSet to avoid duplicating names for fused_var in other
    // pass.
    if (!result->Has(details::kFusedVars)) {
      result->Set(details::kFusedVars, new details::FusedVars);
    }
    // the kFusedGrads is used be fuse_optimizer_op_pass.
    if (!result->Has(details::kFusedGrads)) {
      result->Set(details::kFusedGrads, new details::FusedGrads);
    }
    if (!result->Has(details::kStartupProgramDescs)) {
      result->Set(details::kStartupProgramDescs, new details::ProgramDescs);
    }
    if (!result->Has(details::kProgramDescs)) {
      result->Set(details::kProgramDescs, new details::ProgramDescs);
    }

    auto type = GetTypeOfVar(vars_info, params_grads.front().second);

    bool persistable = false;
    for (auto &p_g : params_grads) {
      if (IsPersistableVar(vars_info, p_g.second)) {
        // NOTE. If one of the grads is persistable, then the fused_grad_var
        // should be set to persistable.
        persistable = true;
        break;
      }
    }

    // the fused_var_name should be unique, so it appends
    // params_grads.begin()->second.
    auto fused_grad_var_name = std::string(details::kFusedVarNamePrefix) +
                               "@GRAD@" + params_grads.begin()->second;
    // what a pity, visual c++ unsupported {.type_ = type}
    details::VariableInfo var_info;
    var_info.name_ = fused_grad_var_name;
    var_info.type_ = type;
    var_info.persistable_ = persistable;

    auto &fused_var_set = result->Get<details::FusedVars>(details::kFusedVars);
    PADDLE_ENFORCE_EQ(
        fused_var_set.count(fused_grad_var_name),
        0,
        common::errors::AlreadyExists("Var(%s) is duplicate in FusedVars.",
                                      fused_grad_var_name));
    fused_var_set.insert({fused_grad_var_name, var_info});

    result->Get<details::FusedGrads>(details::kFusedGrads)
        .emplace_back(fused_grad_var_name);

    InitFusedVarsAndAllocSpaceForVars(
        vars_info, fused_grad_var_name, params_grads, result);
  }

  template <typename AttrType>
  void ResetAttribute(const std::string &attr_name, ir::Graph *graph) const {
    if (graph->Has(attr_name)) {
      VLOG(10) << attr_name << " is reset.";
      graph->Erase(attr_name);
    }
    graph->Set(attr_name, new AttrType);
  }

  void SetGroupParamsAndGrads(
      const std::unordered_map<std::string, std::vector<ir::Node *>> &vars_info,
      const details::ParamsAndGrads &params_grads,
      details::GroupParamsAndGrads *group_params_grads) const {
    if (GetFuseParameterMemorySize() == 0) {
      group_params_grads->resize(1);
      auto &result_param_grads = (*group_params_grads)[0];
      result_param_grads = params_grads;
      std::sort(result_param_grads.begin(), result_param_grads.end());
    } else {
      SetGroupAccordingToLayers(vars_info, params_grads, group_params_grads);
      SetGroupAccordingToMemorySize(vars_info, group_params_grads);
    }
    if (!IsUnifiedDtype(params_grads, vars_info)) {
      ReGroupByDtype(vars_info, group_params_grads);
    }
  }

  void SetGroupAccordingToLayers(
      const std::unordered_map<std::string, std::vector<ir::Node *>> &vars_info,
      const details::ParamsAndGrads &params_grads,
      details::GroupParamsAndGrads *group_params_grads) const {
    std::map<std::string, size_t> var_idx;

    for (const auto &params_grad : params_grads) {
      auto pos = params_grad.first.find_first_of(".");

      std::string var_key;
      if (pos == std::string::npos) {
        var_key = params_grad.first;
      } else {
        var_key = params_grad.first.substr(0, pos);
      }

      size_t idx = 0;
      auto var_idx_iter = var_idx.find(var_key);
      if (var_idx_iter != var_idx.end()) {
        idx = var_idx_iter->second;
      } else {
        group_params_grads->emplace_back();
        idx = group_params_grads->size() - 1;
        var_idx[var_key] = idx;
      }
      auto &local_group_params_grads = group_params_grads->at(idx);
      local_group_params_grads.emplace_back(
          std::make_pair(params_grad.first, params_grad.second));
    }

    if (VLOG_IS_ON(10)) {
      VLOG(10) << "SetGroupAccordingToLayers: ";
      PrintGroupInfo(vars_info, group_params_grads);
    }
  }

  void PrintGroupInfo(
      const std::unordered_map<std::string, std::vector<ir::Node *>> &vars_info,
      details::GroupParamsAndGrads *group_params_grads) const {
    for (size_t i = 0; i < group_params_grads->size(); ++i) {
      VLOG(10) << "group " << i;
      std::stringstream out;
      size_t gps_size = 0;
      for (auto &p_g : group_params_grads->at(i)) {
        auto var_desc = GetVarDescFromVarsInfo(vars_info, p_g.first);
        auto shape = var_desc->GetShape();
        size_t size = framework::SizeOfType(var_desc->GetDataType());
        std::for_each(shape.begin(), shape.end(), [&size](const int64_t &n) {
          size *= n;
        });
        gps_size += size;
        out << string::Sprintf("(%s(%d), %s)", p_g.first, size, p_g.second);
      }

      auto dtype =
          GetDtypeOfVar(vars_info, group_params_grads->at(i).front().second);
      VLOG(10) << out.str()
               << ", group size:" << group_params_grads->at(i).size()
               << ", group memory size:" << static_cast<double>(gps_size) / kMB
               << "(MB), dtype:" << dtype;
    }
  }

  void SetGroupAccordingToMemorySize(
      const std::unordered_map<std::string, std::vector<ir::Node *>> &vars_info,
      details::GroupParamsAndGrads *group_params_grads) const {
    const double group_memory_size = GetFuseParameterMemorySize();
    if (group_memory_size <= 0.0) {
      return;
    }
    details::GroupParamsAndGrads local_group_params_grads;

    size_t j = 0;
    while (j < group_params_grads->size()) {
      local_group_params_grads.emplace_back();
      auto &group_p_g = local_group_params_grads.back();

      size_t local_group_memory_size = 0;
      while (j < group_params_grads->size()) {
        for (auto &p_g_iter : group_params_grads->at(j)) {
          auto var_desc = GetVarDescFromVarsInfo(vars_info, p_g_iter.second);
          size_t size = framework::SizeOfType(var_desc->GetDataType());
          auto shape = var_desc->GetShape();
          std::for_each(shape.begin(), shape.end(), [&size](const int64_t &n) {
            size *= n;
          });
          local_group_memory_size += size;
        }

        group_p_g.insert(group_p_g.end(),
                         group_params_grads->at(j).begin(),
                         group_params_grads->at(j).end());

        ++j;
        if (j >= group_params_grads->size()) {
          break;
        }

        if (GetFuseParameterGroupsSize() > 1 &&
            group_p_g.size() >
                static_cast<size_t>(GetFuseParameterGroupsSize())) {
          break;
        }

        if (static_cast<double>(local_group_memory_size) / kMB >=
            group_memory_size) {
          break;
        }
      }
    }

    std::swap(*group_params_grads, local_group_params_grads);

    if (VLOG_IS_ON(10)) {
      VLOG(10) << string::Sprintf(
          "SetGroupAccordingToMemorySize(memory_size: %f MB):",
          GetFuseParameterMemorySize());
      PrintGroupInfo(vars_info, group_params_grads);
    }
  }

  void ReGroupByDtype(
      const std::unordered_map<std::string, std::vector<ir::Node *>> &vars_info,
      details::GroupParamsAndGrads *group_params_grads) const {
    details::GroupParamsAndGrads new_group_params_grads;

    for (auto &group_p_g : *group_params_grads) {
      std::map<proto::VarType::Type, size_t> type_idx;
      details::GroupParamsAndGrads local_group_params_grads;

      for (auto &p_g : group_p_g) {
        auto dtype = GetDtypeOfVar(vars_info, p_g.second);

        size_t idx = 0;
        auto var_idx_iter = type_idx.find(dtype);
        if (var_idx_iter != type_idx.end()) {
          idx = var_idx_iter->second;
        } else {
          local_group_params_grads.emplace_back();
          idx = local_group_params_grads.size() - 1;
          type_idx[dtype] = idx;
        }

        auto &local = local_group_params_grads.at(idx);
        local.emplace_back(p_g);
      }

      VLOG(10) << "local_group_params_grads size:"
               << local_group_params_grads.size();
      new_group_params_grads.insert(new_group_params_grads.end(),
                                    local_group_params_grads.begin(),
                                    local_group_params_grads.end());
    }

    std::swap(*group_params_grads, new_group_params_grads);

    if (VLOG_IS_ON(10)) {
      VLOG(10) << string::Sprintf("ReGroupByDtype(memory_size: %f MB, %u):",
                                  GetFuseParameterMemorySize(),
                                  GetFuseParameterGroupsSize());
      PrintGroupInfo(vars_info, group_params_grads);
    }
  }

  proto::VarType::Type GetDtypeOfVar(
      const std::unordered_map<std::string, std::vector<ir::Node *>> &vars_info,
      const std::string &name) const {
    auto var_desc = GetVarDescFromVarsInfo(vars_info, name);
    return var_desc->GetDataType();
  }

  proto::VarType::Type GetTypeOfVar(
      const std::unordered_map<std::string, std::vector<ir::Node *>> &vars_info,
      const std::string &name) const {
    auto var_desc = GetVarDescFromVarsInfo(vars_info, name);
    return var_desc->GetType();
  }

  bool IsPersistableVar(
      const std::unordered_map<std::string, std::vector<ir::Node *>> &vars_info,
      const std::string &name) const {
    auto var_desc = GetVarDescFromVarsInfo(vars_info, name);
    return var_desc->Persistable();
  }

 private:
  bool IsDenseTensorType(const proto::VarType::Type &type) const {
    // Current only support DENSE_TENSOR.
    return type == proto::VarType::DENSE_TENSOR;
  }

  std::unordered_map<std::string, std::vector<Node *>> GetVarInfo(
      const Graph &result) const {
    std::unordered_map<std::string, std::vector<Node *>> vars;
    for (Node *node : result.Nodes()) {
      if (node->IsVar() && node->Var()) {
        // Note: The graph may have the same name node. For example, parameter
        // is the input of operator and it also is the output of optimizer;
        vars[node->Var()->Name()].emplace_back(node);
      }
    }
    return vars;
  }

  const VarDesc *GetVarDescFromVarsInfo(
      const std::unordered_map<std::string, std::vector<Node *>> &vars_info,
      const std::string &var_name) const {
    auto grad_iter = vars_info.find(var_name);
    PADDLE_ENFORCE_EQ(
        grad_iter != vars_info.end(),
        true,
        common::errors::NotFound("Variable %s is not found.", var_name));
    PADDLE_ENFORCE_EQ(!grad_iter->second.empty(),
                      true,
                      common::errors::InvalidArgument(
                          "Variable %s's node is empty.", var_name));
    PADDLE_ENFORCE_NOT_NULL(
        grad_iter->second.front()->Var(),
        common::errors::InvalidArgument("A node of %s does not hold variable.",
                                        var_name));
    return grad_iter->second.front()->Var();
  }

  void RecordParamsAndGrads(const ir::Graph &graph,
                            details::ParamsAndGrads *params_grads) const {
    std::vector<ir::Node *> topo_nodes = ir::TopologySortOperations(graph);
    for (auto &node : topo_nodes) {
      auto &op_desc = *(node->Op());

      bool is_bk_op = details::IsOpRole(op_desc, OpRole::kBackward);
      if (!is_bk_op) continue;
      // Currently, we assume that once gradient is generated, it can be
      // broadcast, and each gradient is only broadcast once.
      auto backward_vars = details::GetOpRoleVarsOrEmpty(op_desc);
      for (size_t i = 0; i < backward_vars.size(); i += 2) {
        VLOG(10) << "Trainable parameter: " << backward_vars[i]
                 << ", gradient: " << backward_vars[i + 1];

        params_grads->emplace_back(std::make_pair(
            backward_vars[i] /*param*/, backward_vars[i + 1] /*grad*/));
      }
    }
  }

  void InitFusedVarsAndAllocSpaceForVars(
      const std::unordered_map<std::string, std::vector<ir::Node *>> &vars_info,
      const std::string &fused_var_name,
      const details::ParamsAndGrads &params_grads,
      ir::Graph *result) const {
    // Alloc continuous space for vars.
    std::vector<std::string> grads_name;
    std::vector<std::string> params_name;
    grads_name.reserve(params_grads.size());
    params_name.reserve(params_grads.size());

    auto dtype = GetDtypeOfVar(vars_info, params_grads.front().second);
    for (auto &p_g : params_grads) {
      params_name.emplace_back(p_g.first);
      grads_name.emplace_back(p_g.second);
      auto next_dtype = GetDtypeOfVar(vars_info, p_g.second);
      PADDLE_ENFORCE_EQ(next_dtype,
                        dtype,
                        common::errors::InvalidArgument(
                            "All Parameter@Grad should have same dtype, but "
                            "there are two different type: %s, %s.",
                            DataTypeToString(next_dtype),
                            DataTypeToString(dtype)));
    }

    bool any_persistable = false;
    bool all_persistable = true;
    for (auto &p_g : params_grads) {
      if (IsPersistableVar(vars_info, p_g.second)) {
        any_persistable = true;
      } else {
        all_persistable = false;
      }
    }
    VLOG(4) << "all_persistable:" << all_persistable;
    VLOG(4) << "any_persistable:" << any_persistable;
    // NOTE. In scope_buffered_ssa_graph_executor, after each execution of
    // DropScope(), non persistable vars will be Erase or Clear. So
    // coalesce_tensor op needs to be executed again after the execution
    // of DropScope().

    // we can make fused_output persistable, so the memory is not cleared
    // and coalesce_tensor op do nothing if the inputs are already continue.

    result->Get<details::ProgramDescs>(details::kProgramDescs).emplace_back();
    ProgramDesc &program_desc =
        result->Get<details::ProgramDescs>(details::kProgramDescs).back();
    auto *global_block = program_desc.MutableBlock(0);
    AppendAllocSpaceForVarsOp(params_name,
                              grads_name,
                              fused_var_name,
                              dtype,
                              any_persistable,
                              global_block);
  }

  void AppendAllocSpaceForVarsOp(const std::vector<std::string> &params_name,
                                 const std::vector<std::string> &grads_name,
                                 const std::string &fused_var_name,
                                 const proto::VarType::Type &dtype,
                                 bool persistable,
                                 BlockDesc *global_block) const {
    auto fused_out_var = global_block->Var(fused_var_name);
    fused_out_var->SetPersistable(persistable);

    auto op_desc = global_block->AppendOp();
    op_desc->SetType("coalesce_tensor");
    op_desc->SetInput("Input", params_name);
    op_desc->SetOutput("Output", grads_name);
    op_desc->SetOutput("FusedOutput", {fused_var_name});
    op_desc->SetAttr("dtype", static_cast<int>(dtype));
    op_desc->SetAttr("persist_output", persistable);
  }
};
}  // namespace paddle::framework::ir

REGISTER_PASS(coalesce_grad_tensor_pass,
              paddle::framework::ir::CoalesceGradTensorPass)
    .RequirePassAttr(paddle::framework::details::kNRanks);
