// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/new_executor/interpreter/dependency_builder.h"

#include <queue>
#include <sstream>
#include <stack>
#include "paddle/common/flags.h"
#include "paddle/fluid/framework/new_executor/instruction/instruction_base.h"
#include "paddle/fluid/framework/new_executor/instruction/phi_kernel_instruction.h"
#include "paddle/fluid/framework/new_executor/interpreter/interpreter_util.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/phi/common/reduce_type.h"

PHI_DEFINE_EXPORTED_bool(
    add_dependency_for_communication_op,
    true,
    "Whether to add dependency for communication Ops. It is just a temporary "
    "FLAGS especially for auto parallel to avoid the concurrency damage by the "
    "communication dependency added in standalone executor.");

// The difference between "sequential_run" and "serial_run":
// "sequential_run" dispatches OPs one by one according to the sequence in the
// Program, while "serial_run" ensures that all Ops are scheduled in a signal
// thread. In standalone executor, "sequential_run" is also "serial_run", while
// "serial_run" is not necessarily "sequential_run".
PHI_DEFINE_EXPORTED_bool(new_executor_sequential_run,
                         false,
                         "Enable sequential execution for standalone "
                         "executor, only applied to GPU OPs.");
COMMON_DECLARE_int32(enable_adjust_op_order);
// add debug info
PHI_DEFINE_EXPORTED_bool(enable_dependency_builder_debug_info,
                         false,
                         "Enable dependency builder debug info");

namespace paddle::framework::interpreter {

size_t CountDownstreamMap(
    const std::map<size_t, std::set<size_t>>& downstream_map) {
  size_t count = 0;
  for (auto const& pair : downstream_map) {
    count += pair.second.size();
  }
  return count;
}
const std::string StringizeDownstreamMap(
    const std::map<size_t, std::set<size_t>>& downstream_map) {
  std::ostringstream oss;
  oss << "\n"
      << std::left << std::setw(7) << "id" << std::setw(40) << "down_stream_id"
      << "\n";
  for (auto const& pair : downstream_map) {
    oss << std::setw(7) << pair.first << std::setw(40) << " -> ";
    std::copy(pair.second.begin(),
              pair.second.end(),
              std::ostream_iterator<size_t>(oss, " "));
    oss << std::endl;
  }
  return oss.str();
}

DependencyBuilder::DependencyBuilder()
    : is_build_(false),
      op_num_(0),
      ops_before_(),
      ops_behind_(),
      op_downstream_map_(nullptr),
      op_happens_before_(nullptr),
      instructions_(nullptr) {
  op_downstream_map_ = std::make_shared<std::map<size_t, std::set<size_t>>>();
  op_happens_before_ = std::make_shared<std::vector<std::vector<bool>>>();
}

const std::map<size_t, std::set<size_t>>& DependencyBuilder::Build(
    const std::vector<Instruction>& instructions) {
  if (is_build_) {
    return *op_downstream_map_;
  }

  std::tie(op_downstream_map_, op_happens_before_) = GetDependency();

  instructions_ = &instructions;
  op_num_ = instructions_->size();

  ops_before_.assign(op_num_, {});
  ops_behind_.assign(op_num_, {});
  op_happens_before_->assign(op_num_, std::vector<bool>(op_num_, false));

  BuildDownstreamMap();
  VLOG(6) << "Finish BuildDownstreamMap";

  ShrinkDownstreamMap();
  VLOG(6) << "Finish ShrinkDownstreamMap";

  if (FLAGS_new_executor_sequential_run) {
    AddDependencyForSequentialRun();
  }

  AddDependencyForCoalesceTensorOp();

  if (FLAGS_add_dependency_for_communication_op) {
    AddDependencyForCommunicationOp();
    VLOG(6) << "Finish AddDependencyForSequentialRun";
  }

  AddDependencyForRandomOp();
  VLOG(6) << "Finish AddDependencyForRandomOp";

  AddDependencyForReadOp();
  VLOG(6) << "Finish AddDependencyForReadOp";

  VLOG(6) << "Finish build dependency";
  VLOG(8) << "downstream count: " << CountDownstreamMap(*op_downstream_map_);
  VLOG(8) << "downstream_map: " << std::endl
          << StringizeDownstreamMap(*op_downstream_map_);

  is_build_ = true;

  return *op_downstream_map_;
}

std::tuple<std::shared_ptr<std::map<size_t, std::set<size_t>>>,
           std::shared_ptr<std::vector<std::vector<bool>>>>
DependencyBuilder::GetDependency() const {
  return std::make_tuple(op_downstream_map_, op_happens_before_);
}

void DependencyBuilder::ShareDependencyFrom(const DependencyBuilder& src) {
  std::tie(op_downstream_map_, op_happens_before_) = src.GetDependency();
  is_build_ = true;
}

const std::string& DependencyBuilder::GetInstructionName(size_t op_idx) const {
  return (*instructions_)[op_idx].OpBase()->Type();
}

const std::map<size_t, std::set<size_t>>& DependencyBuilder::OpDownstreamMap()
    const {
  PADDLE_ENFORCE_EQ(
      is_build_,
      true,
      common::errors::Unavailable(
          "DependencyBuilder is not yet built, call Build() firstly."));
  return *op_downstream_map_;
}

void DependencyBuilder::AddDependencyForCoalesceTensorOp() {
  for (size_t op_idx = 0; op_idx < op_num_; ++op_idx) {
    if (instructions_->at(op_idx).OpBaseValid() &&
        instructions_->at(op_idx).OpBase()->Type() == kCoalesceTensor) {
      VLOG(4) << "Add depend for " << kCoalesceTensor << " " << op_idx;
      auto fused_out = instructions_->at(op_idx).Outputs().at("FusedOutput")[0];
      auto outputs = instructions_->at(op_idx).Outputs().at("Output");

      auto is_read = [](const Instruction& inst, size_t var_id) -> bool {
        for (auto const& pair : inst.Inputs()) {
          for (size_t item : pair.second) {
            if (item == var_id) {
              return true;
            }
          }
        }
        return false;
      };

      auto is_write = [](const Instruction& inst, size_t var_id) -> bool {
        for (auto const& pair : inst.Outputs()) {
          for (size_t item : pair.second) {
            if (item == var_id) {
              return true;
            }
          }
        }
        return false;
      };

      // find first op that reads fused_out
      auto first_read_fused_out_op = ULLONG_MAX;
      for (auto j = op_idx + 1; j < op_num_; ++j) {
        if (is_read(instructions_->at(j), fused_out)) {
          first_read_fused_out_op = j;
          break;
        }
      }

      if (UNLIKELY(first_read_fused_out_op == ULLONG_MAX)) {
        VLOG(4) << "No op read FusedOutput";
        continue;
      }

      // find ops that write 'outputs' between (op_index,
      // first_read_fused_out_op)
      // add depend: them->first_read_fused_out_op
      for (auto j = op_idx + 1;
           j < static_cast<size_t>(first_read_fused_out_op);
           ++j) {
        for (auto var_id : outputs) {
          if (is_write(instructions_->at(j), var_id)) {
            AddDownstreamOp(j, first_read_fused_out_op);
          }
        }
      }

      // find first op read 'outputs' between (first_read_fused_out_op, end)
      // add depend:  first_read_fused_out_op -> first op that reads 'outputs'

      // special case for consecutive communication ops, for example,
      // FusedOutput = c_sync_calc_stream(FusedOutput)
      // FusedOutput= c_allreduce_sum(FusedOutput)
      // FusedOutput = c_sync_comm_stream(FusedOutput)
      // we should take the last one to add depend instead of
      // 'first_read_fused_out_op'
      size_t target = first_read_fused_out_op;
      for (size_t j = first_read_fused_out_op + 1; j < op_num_; ++j) {
        if (j == target + 1 && IsCommunicationOp(instructions_->at(target)) &&
            IsCommunicationOp(instructions_->at(j))) {
          VLOG(4) << "Found consecutive communication ops, "
                  << instructions_->at(target).OpBase()->Type() << " -> "
                  << instructions_->at(j).OpBase()->Type();
          target = j;
          continue;
        }

        for (auto var_id : outputs) {
          if (is_read(instructions_->at(j), var_id)) {
            AddDownstreamOp(target, j);
          }
        }
      }
    }
  }
}

void DependencyBuilder::AddDependencyForCommunicationOp() {
  size_t dependence_op_idx = ULLONG_MAX;
  for (size_t op_idx = 0; op_idx < op_num_; ++op_idx) {
    if (IsCommunicationOp(instructions_->at(op_idx))) {
      if (dependence_op_idx != ULLONG_MAX) {
        AddDownstreamOp(dependence_op_idx, op_idx);
      }
      dependence_op_idx = op_idx;
    }
  }

  // TODO(zhiqiu): there still some cases not handled
  // add dependency for c_sync_comm_stream

  // in program, we can add only one c_sync_comm_stream to sync all
  // communication ops.
  // c_allreduce_sum(a)
  // c_allreduce_sum(b)
  // c_allreduce_sum(c)
  // c_sync_comm_stream(a)
  const std::string kCSyncComm = "c_sync_comm_stream";
  const std::string kSyncComm = "sync_comm_stream";
  dependence_op_idx = ULLONG_MAX;
  for (size_t op_idx = 0; op_idx < op_num_; ++op_idx) {
    if (instructions_->at(op_idx).OpBaseValid() &&
        (instructions_->at(op_idx).OpBase()->Type() == kCSyncComm ||
         instructions_->at(op_idx).OpBase()->Type() == kSyncComm)) {
      dependence_op_idx = op_idx;
    } else {
      if (dependence_op_idx != ULLONG_MAX) {
        AddDownstreamOp(dependence_op_idx, op_idx);
      }
    }
  }
}

// make sure that the random op is scheduled sequentially
void DependencyBuilder::AddDependencyForRandomOp() {
  const std::set<std::string> random_op_set = {"bernoulli",
                                               "poisson",
                                               "multinomial",
                                               "gaussian_random",
                                               "truncated_gaussian_random",
                                               "uniform_random",
                                               "randint",
                                               "randperm",
                                               "exponential",
                                               "sampling_id",
                                               "dropout",
                                               "class_center_sample"};

  size_t dependence_op_idx = ULLONG_MAX;
  for (size_t op_idx = 0; op_idx < op_num_; ++op_idx) {
    if (instructions_->at(op_idx).OpBaseValid() &&
        random_op_set.count(instructions_->at(op_idx).OpBase()->Type())) {
      if (dependence_op_idx != ULLONG_MAX) {
        AddDownstreamOp(dependence_op_idx, op_idx);
      }
      dependence_op_idx = op_idx;
    }
  }
}

// equivalent to add_reader_dependency_pass
void DependencyBuilder::AddDependencyForReadOp() {
  std::vector<bool> is_startup_ops(op_num_, true);
  for (size_t op_idx = 0; op_idx < op_num_; ++op_idx) {
    auto it = op_downstream_map_->find(op_idx);
    if (it != op_downstream_map_->end()) {
      for (size_t downstream_op_idx : it->second) {
        is_startup_ops[downstream_op_idx] = false;
      }
    }
  }

  std::vector<size_t> read_ops;
  std::vector<size_t> startup_ops;
  for (size_t op_idx = 0; op_idx < op_num_; ++op_idx) {
    if (instructions_->at(op_idx).OpBaseValid() &&
        instructions_->at(op_idx).OpBase()->Type() == "read") {
      read_ops.push_back(op_idx);
    }

    if (is_startup_ops[op_idx]) {
      startup_ops.push_back(op_idx);
    }
  }

  for (size_t read_op_idx : read_ops) {
    for (size_t downstream_op_idx : startup_ops) {
      if (read_op_idx != downstream_op_idx &&
          !OpHappensBefore(downstream_op_idx, read_op_idx)) {
        AddDownstreamOp(read_op_idx, downstream_op_idx);
      }
    }
  }
}

void DependencyBuilder::AddDependencyForSequentialRun() {
  size_t dependence_op_idx = ULLONG_MAX;
  for (size_t op_idx = 0; op_idx < op_num_; ++op_idx) {
    if (dependence_op_idx != ULLONG_MAX) {
      if (this->GetInstructionName(op_idx) == "pd_op.full_int_array") {
        VLOG(8) << "Skip adding dependency for sequential run: "
                << dependence_op_idx << "->" << op_idx << " "
                << this->GetInstructionName(dependence_op_idx) << "->"
                << this->GetInstructionName(op_idx);
        continue;
      }

      AddDownstreamOp(dependence_op_idx, op_idx);
    }
    dependence_op_idx = op_idx;
  }
}

void DependencyBuilder::AddDownstreamOp(size_t prior_op_idx,
                                        size_t posterior_op_idx) {
  PADDLE_ENFORCE_EQ(
      OpHappensBefore(posterior_op_idx, prior_op_idx),
      false,
      common::errors::Unavailable(
          "Can not add dependency %d->%d because %d is run before %d",
          prior_op_idx,
          posterior_op_idx,
          posterior_op_idx,
          prior_op_idx));
  std::set<size_t>& downstream_ops = (*op_downstream_map_)[prior_op_idx];
  // NOTE(Ruibiao): Here the downstream map shrinking is best-effort, therefore
  // ShrinkDownstreamMap after BuildDownstreamMap is still helpful. For example,
  // a->c will not be shrunk in the following case: AddDownstreamOp(a, b) ->
  // AddDownstreamOp(a, c) -> AddDownstreamOp(b, c), it should be shrunk by
  // ShrinkDownstreamMap.
  for (size_t op_idx : downstream_ops) {
    if (OpHappensBefore(op_idx, posterior_op_idx)) {
      VLOG(7) << "Find dependencies " << prior_op_idx << "->" << op_idx << "->"
              << posterior_op_idx << ", skip adding " << prior_op_idx << "->"
              << posterior_op_idx;
      return;
    }
  }
  downstream_ops.insert(posterior_op_idx);

  std::vector<size_t> prior_of_prior = ops_before_[prior_op_idx];
  std::vector<size_t> posterior_of_posterior = ops_behind_[posterior_op_idx];

  auto update_op_happen_before = [this](size_t prior_op_idx,
                                        size_t posterior_op_idx) {
    if (!(*op_happens_before_)[prior_op_idx][posterior_op_idx]) {
      (*op_happens_before_)[prior_op_idx][posterior_op_idx] = true;
      ops_before_[posterior_op_idx].push_back(prior_op_idx);
      ops_behind_[prior_op_idx].push_back(posterior_op_idx);
    }
  };

  update_op_happen_before(prior_op_idx, posterior_op_idx);

  // All ops before prior-op are also before posterior-op
  for (size_t op_idx : prior_of_prior) {
    update_op_happen_before(op_idx, posterior_op_idx);
  }
  // All ops after posterior-op are also after prior-op
  for (size_t op_idx : posterior_of_posterior) {
    update_op_happen_before(prior_op_idx, op_idx);
  }

  VLOG(8) << prior_op_idx << "->" << posterior_op_idx;
  VLOG(8) << "Add dependency from "
          << "prior_op_idx(" << prior_op_idx << ") to "
          << "posterior_op_idx(" << posterior_op_idx << ")";
}

void DependencyBuilder::BuildDownstreamMap() {
  auto var2min_rw_op =
      std::map<size_t, std::list<size_t>>();  // # map from variable id to read
                                              //  write op id.
  auto var2recent_write_op =
      std::map<size_t, size_t>();  // # map from variable to recent write op.
  auto op2dependences =
      std::map<size_t,
               std::set<size_t>>();  // # map from op to the dependence list,
                                     //  op must run after the dependence.
  std::set<size_t>
      remove_duplicate;  // remove the duplicate between inputs and outputs

  // reserve
  for (size_t op_idx = 0; op_idx < op_num_; ++op_idx) {
    op2dependences[op_idx] = std::set<size_t>();
  }

  for (size_t op_idx = 0; op_idx < op_num_; ++op_idx) {
    remove_duplicate.clear();
    // step1: update the op2dependences structure
    for (auto& item :
         instructions_->at(op_idx).Inputs()) {  // for all inputs(read only)
      for (auto var : item.second) {
        if (var2recent_write_op.count(var))
          op2dependences[op_idx].insert(var2recent_write_op[var]);
      }
    }

    for (auto& item :
         instructions_->at(op_idx).Outputs()) {  // for all write vars
      for (auto var : item.second) {
        if (var2min_rw_op.count(var)) {
          for (auto dep_op : var2min_rw_op[var]) {
            op2dependences[op_idx].insert(dep_op);
          }
        }
      }
    }
    // the original output of inplace op is also change.
    if (!instructions_->at(op_idx).InplaceBackMap().empty()) {
      auto& m = instructions_->at(op_idx).InplaceBackMap();
      for (auto& p : m) {
        auto& var = p.second;
        if (var2min_rw_op.count(var)) {
          for (auto dep_op : var2min_rw_op[var]) {
            op2dependences[op_idx].insert(dep_op);
          }
        }
      }
    }

    // step2: update 2 var2xxxx data structure
    for (auto& item :
         instructions_->at(op_idx).Outputs()) {  // for all write vars
      for (auto var : item.second) {
        var2recent_write_op[var] = op_idx;
        var2min_rw_op[var] = {static_cast<size_t>(op_idx)};
        remove_duplicate.insert(var);
      }
    }

    // NOTE(zhiqiu): The inplace op with `transfer` also changes
    // original output after that so add original output as well
    // original: a->op->a
    // after: a->data_transfer->a'->op->a'->transfer_back->a
    // which means op writes a and a'
    if (!instructions_->at(op_idx).InplaceBackMap().empty()) {
      auto& m = instructions_->at(op_idx).InplaceBackMap();
      for (auto& p : m) {
        auto var = p.second;
        var2recent_write_op[var] = op_idx;
        var2min_rw_op[var] = {static_cast<size_t>(op_idx)};
        remove_duplicate.insert(var);
      }
    }

    for (auto& item :
         instructions_->at(op_idx).Inputs()) {  // for all inputs(read only)
      for (auto var : item.second) {
        if (remove_duplicate.count(var) ==
            0) {  // var in input list and in output list, so remove it.
          UpdateVarMinRwOp(op2dependences, &var2min_rw_op, op_idx, var);
        }
      }
    }
  }

  // convert op2dependences to downstream_map directly. op2dependences is op ->
  // it's dependences, we want to get op -> [next ops] map, where ops is the
  // next instruction of op. The size of downstream != size of op2dependences
  // since there are some ops that have no downstream-op.
  for (auto& item : op2dependences) {
    size_t op = item.first;
    for (auto dep_op : item.second) {
      AddDownstreamOp(dep_op, op);
    }
  }
}

void DependencyBuilder::ShrinkDownstreamMap() {
  // remove unnecessary downstream ops
  // for example, a->b->c
  // a: b, c
  // b: c
  // =>
  // a: b
  // b: c

  // shrink, find the downstream op that has no other op in the
  // downstream list happens before it
  for (size_t i = 0; i < op_num_; ++i) {
    if (op_downstream_map_->find(i) == op_downstream_map_->end()) {
      continue;
    }

    std::set<size_t> minimum_nexts;
    for (size_t item : op_downstream_map_->at(i)) {
      bool not_after_any = true;
      // find the op that is not executed after any
      for (size_t other_item : op_downstream_map_->at(i)) {
        if (OpHappensBefore(other_item, item)) {
          VLOG(8) << "happens_before: " << other_item << "->" << item
                  << ", so skip " << item;
          not_after_any = false;
          break;
        }
      }
      if (not_after_any) {
        VLOG(8) << "downstream op of " << i << ": " << item;
        minimum_nexts.insert(item);
      }
    }
    // NOTE(Ruibiao): op_happens_before will not be changed when shrink
    // downstream map
    (*op_downstream_map_)[i] = minimum_nexts;
  }
  VLOG(8) << "Finish shrink downstream map";
  VLOG(8) << "downstream count: " << CountDownstreamMap(*op_downstream_map_);
  VLOG(8) << "downstream_map: " << std::endl
          << StringizeDownstreamMap(*op_downstream_map_);
}

void DependencyBuilder::UpdateVarMinRwOp(
    const std::map<size_t, std::set<size_t>>& op2dependences,
    std::map<size_t, std::list<size_t>>* var2min_rw_op,
    size_t cur_op,
    size_t rw_var) {
  // rw_var is inputs or outputs of cur_op
  // this function update the var2min_rw_op set .
  if (var2min_rw_op->find(rw_var) == var2min_rw_op->end()) {
    (*var2min_rw_op)[rw_var] = std::list<size_t>();
  }
  for (auto dep_op : op2dependences.at(cur_op)) {
    var2min_rw_op->at(rw_var).remove(dep_op);
  }
  var2min_rw_op->at(rw_var).push_back(cur_op);
}

/// ======================== ///
///        For new ir        ///
/// ======================== ///
PirDependencyBuilder::PirDependencyBuilder() : instructions_() {
  is_build_ = false;
  op_downstream_map_ = std::make_shared<std::map<size_t, std::set<size_t>>>();
  op_happens_before_ = std::make_shared<std::vector<std::vector<bool>>>();
}

const std::string& PirDependencyBuilder::GetInstructionName(
    size_t op_idx) const {
  return (instructions_)[op_idx]->Name();
}

void PirDependencyBuilder::AddDependencyForCommunicationOp() {
  size_t dependence_op_idx = ULLONG_MAX;
  for (size_t op_idx = 0; op_idx < op_num_; ++op_idx) {
    if (instructions_.at(op_idx)->Operation() &&
        IsCommunicationOp(instructions_.at(op_idx)->Operation())) {
      if (dependence_op_idx != ULLONG_MAX) {
        AddDownstreamOp(dependence_op_idx, op_idx);
      }
      dependence_op_idx = op_idx;
    }
  }

  // TODO(zhiqiu): there still some cases not handled
  // add dependency for c_sync_comm_stream

  // in program, we can add only one c_sync_comm_stream to sync all
  // communication ops.
  // c_allreduce_sum(a)
  // c_allreduce_sum(b)
  // c_allreduce_sum(c)
  // c_sync_comm_stream(a)
  const std::string kSyncComm = dialect::SyncCommStreamOp::name();
  dependence_op_idx = ULLONG_MAX;
  for (size_t op_idx = 0; op_idx < op_num_; ++op_idx) {
    if (instructions_.at(op_idx)->Name() == kSyncComm) {
      dependence_op_idx = op_idx;
    } else {
      if (dependence_op_idx != ULLONG_MAX) {
        AddDownstreamOp(dependence_op_idx, op_idx);
      }
    }
  }
}

void PirDependencyBuilder::AddDependencyForRandomOp() {
  const std::set<std::string> random_op_set = {
      dialect::BernoulliOp::name(),
      dialect::PoissonOp::name(),
      dialect::MultinomialOp::name(),
      dialect::GaussianOp::name(),
      dialect::TruncatedGaussianRandomOp::name(),
      dialect::UniformOp::name(),
      dialect::RandintOp::name(),
      dialect::RandpermOp::name(),
      dialect::Exponential_Op::name(),
      dialect::DropoutOp::name(),
      dialect::ClassCenterSampleOp::name()};

  size_t dependence_op_idx = ULLONG_MAX;
  for (size_t op_idx = 0; op_idx < op_num_; ++op_idx) {
    if (dynamic_cast<PhiKernelInstruction*>(instructions_.at(op_idx)) &&
        random_op_set.count(instructions_.at(op_idx)->Name())) {
      if (dependence_op_idx != ULLONG_MAX) {
        AddDownstreamOp(dependence_op_idx, op_idx);
      }
      dependence_op_idx = op_idx;
    }
  }
}

const std::map<size_t, std::set<size_t>>& PirDependencyBuilder::Build(
    std::vector<paddle::framework::InstructionBase*> instructions) {
  if (is_build_) {
    return *op_downstream_map_;
  }

  std::tie(op_downstream_map_, op_happens_before_) = GetDependency();

  instructions_ = instructions;
  op_num_ = instructions_.size();

  ops_before_.assign(op_num_, {});
  ops_behind_.assign(op_num_, {});
  op_happens_before_->assign(op_num_, std::vector<bool>(op_num_, false));

  BuildDownstreamMap();
  VLOG(6) << "Finish BuildDownstreamMap";

  ShrinkDownstreamMap();
  VLOG(6) << "Finish ShrinkDownstreamMap";

  if (FLAGS_new_executor_sequential_run) {
    AddDependencyForSequentialRun();
  }

  if (FLAGS_add_dependency_for_communication_op) {
    AddDependencyForCommunicationOp();
    VLOG(6) << "Finish AddDependencyForSequentialRun";
  }

  // TODO(zhangbo): Add dependency for special op ？
  // Note(lvyongkang): necessary for reproducibility
  AddDependencyForRandomOp();

  VLOG(6) << "Finish build dependency";
  VLOG(8) << "downstream count: " << CountDownstreamMap(*op_downstream_map_);
  VLOG(8) << "downstream_map: " << std::endl
          << StringizeDownstreamMap(*op_downstream_map_);

  is_build_ = true;

  return *op_downstream_map_;
}

void PirDependencyBuilder::BuildDownstreamMap() {
  auto var2min_rw_op =
      std::map<size_t, std::list<size_t>>();  // # map from variable id to read
                                              //  write op id.
  auto var2recent_write_op =
      std::map<size_t, size_t>();  // # map from variable to recent write op.

  auto op2dependences =
      std::map<size_t,
               std::set<size_t>>();  //# map from op to the dependence list,
                                     // op must run after the dependence.
  std::set<size_t>
      remove_duplicate;  // remove the duplicate between inputs and outputs

  // reserve
  for (size_t op_idx = 0; op_idx < op_num_; ++op_idx) {
    op2dependences[op_idx] = std::set<size_t>();
  }

  for (size_t op_idx = 0; op_idx < op_num_; ++op_idx) {
    remove_duplicate.clear();
    // step1: update the op2dependences structure
    for (auto& item :
         instructions_.at(op_idx)->Inputs()) {  // for all inputs(read only)
      for (auto var : item.second) {
        if (var2recent_write_op.count(var))
          op2dependences[op_idx].insert(var2recent_write_op[var]);
      }
    }

    for (auto& item :
         instructions_.at(op_idx)->Outputs()) {  // for all write vars
      for (auto var : item.second) {
        if (var2min_rw_op.count(var)) {
          for (auto dep_op : var2min_rw_op[var]) {
            op2dependences[op_idx].insert(dep_op);
          }
        }
      }
    }

    // step2: update 2 var2xxxx data structure
    for (auto& item :
         instructions_.at(op_idx)->Outputs()) {  // for all write vars
      for (auto var : item.second) {
        var2recent_write_op[var] = op_idx;
        var2min_rw_op[var] = {static_cast<size_t>(op_idx)};
        remove_duplicate.insert(var);
      }
    }

    for (auto& item :
         instructions_.at(op_idx)->Inputs()) {  // for all inputs(read only)
      for (auto var : item.second) {
        if (remove_duplicate.count(var) ==
            0) {  // var in input list and in output list, so remove it.
          UpdateVarMinRwOp(op2dependences, &var2min_rw_op, op_idx, var);
        }
      }
    }
  }

  // convert op2dependences to downstream_map directly. op2dependences is op ->
  // it's dependences, we want to get op -> [next ops] map, where ops is the
  // next instruction of op. The size of downstream != size of op2dependences
  // since there are some ops that have no downstream-op.
  for (auto& item : op2dependences) {
    size_t op = item.first;
    for (auto dep_op : item.second) {
      AddDownstreamOp(dep_op, op);
    }
  }
}

void PirDependencyBuilder::ShareDependencyFrom(
    const PirDependencyBuilder& src) {
  std::tie(op_downstream_map_, op_happens_before_) = src.GetDependency();
  is_build_ = true;
}

void DependencyBuilderSimplify::GetAllbehind() {
  auto update_op_happen_before = [this](size_t prior_op_idx,
                                        size_t posterior_op_idx) {
    if (!op_happens_before_[prior_op_idx][posterior_op_idx]) {
      op_happens_before_[prior_op_idx][posterior_op_idx] = true;
      ops_before_[posterior_op_idx].push_back(prior_op_idx);
      ops_behind_[prior_op_idx].push_back(posterior_op_idx);
    }
  };
  for (size_t i = start_index_; i < op_num_; i++) {
    auto& behind_ops = ops_behind_[i];
    auto& before_ops = ops_before_[i];
    for (auto before_op : before_ops) {
      for (auto behind_op : behind_ops) {
        update_op_happen_before(before_op, behind_op);
      }
    }
  }
}

const std::map<size_t, std::set<size_t>>& DependencyBuilderSimplify::Build(
    const std::vector<std::unique_ptr<OperatorBase>>& ops,
    size_t start_index,
    bool is_sharding_mode) {
  PADDLE_ENFORCE_EQ(
      is_build_,
      false,
      common::errors::AlreadyExists("The op dependency has been built"));
  start_index_ = start_index;
  is_sharding_mode_ = is_sharding_mode;
  _ops_ptr = &ops;
  op_num_ = _ops_ptr->size();
  SetSameStream();
  for (size_t op_idx = start_index_; op_idx < op_num_; ++op_idx) {
    if (del_c_sync_comm_list.count(op_idx)) {
      continue;
    }
    ops_list.push_back(op_idx);
  }

  ops_before_.assign(op_num_, {});
  ops_behind_.assign(op_num_, {});
  for (size_t i = 0; i < op_num_; i++) {
    ops_before_[i].reserve(op_num_);
    ops_behind_[i].reserve(op_num_);
  }
  op_happens_before_.assign(op_num_, std::vector<bool>(op_num_, false));
  auto print_log = [this, &ops](std::string msg) {
    for (auto it : op_downstream_map_) {
      if (it.second.size() > 0) {
        std::stringstream ss;
        ss << msg.c_str() << " op ";
        auto& input_op = ops[it.first];
        ss << input_op->Type() << "_" << it.first << " inputs [  ";
        for (auto& name_pair : input_op->Inputs()) {
          for (auto& name : name_pair.second) {
            ss << name << " ";
          }
        }
        ss << " ]  outputs [ ";
        for (auto& name_pair : input_op->Outputs()) {
          for (auto& name : name_pair.second) {
            ss << name << " ";
          }
        }
        ss << " ] before { ";
        for (auto index : it.second) {
          auto& op = ops[index];
          ss << "(" << op->Type() << "_" << index << " ) "
             << " inputs [  ";
          for (auto& name_pair : op->Inputs()) {
            for (auto& name : name_pair.second) {
              ss << name << " ";
            }
          }
          ss << " ]  outputs [ ";
          for (auto& name_pair : op->Outputs()) {
            for (auto& name : name_pair.second) {
              ss << name << " ";
            }
          }
          ss << " ]   ";
        }
        ss << " } ";
        VLOG(2) << ss.str();
      }
    }
  };

  BuildDownstreamMap();
  VLOG(6) << "Finish BuildDownstreamMap";

  ShrinkDownstreamMap();
  VLOG(6) << "Finish ShrinkDownstreamMap";
  if (FLAGS_enable_dependency_builder_debug_info) {
    print_log(std::string("after shrinkDownstreamMap"));
  }
  AddDependencyForCoalesceTensorOp();

  // when is_sharding_mode_ is true hbm not safe should run in debug model need
  // to fix
  if (!is_sharding_mode_) {
    AddDependencyForCommunicationOp();
  }
  VLOG(6) << "Finish AddDependencyForSequentialRun";

  AddDependencyForRandomOp();
  VLOG(6) << "Finish AddDependencyForRandomOp";

  AddDependencyForReadOp();
  // when FLAGS_enable_adjust_op_order >1 , it will reduce more hbm, but also
  // reduce performance
  if (FLAGS_enable_adjust_op_order == 1) {
    AddDependencyForBroadcastOp();
  }
  GetAllbehind();
  GetOpBehindNum();
  VLOG(6) << "Finish AddDependencyForReadOp";
  VLOG(6) << "Finish build dependency";
  VLOG(8) << "downstream count: " << CountDownstreamMap(op_downstream_map_);
  VLOG(8) << "downstream_map: " << std::endl
          << StringizeDownstreamMap(op_downstream_map_);
  is_build_ = true;
  if (FLAGS_enable_dependency_builder_debug_info) {
    print_log(std::string(" all depend info "));
  }

  return op_downstream_map_;
}

void DependencyBuilderSimplify::BuildDownstreamMap() {
  auto var2min_rw_op =
      std::map<std::string,
               std::list<size_t>>();  // # map from variable name to read
                                      //  write op id.
  auto var2recent_write_op =
      std::map<std::string,
               size_t>();  // # map from variable to recent write op.
  auto op2dependences =
      std::map<size_t,
               std::set<size_t>>();  //# map from op to the dependence list,
                                     // op must run after the dependence.
  std::set<std::string>
      remove_duplicate;  // remove the duplicate between inputs and outputs

  // reserve
  for (size_t op_idx = 0; op_idx < op_num_; ++op_idx) {
    op2dependences[op_idx] = std::set<size_t>();
  }

  auto update_var_min_rw_op =
      [](const std::map<size_t, std::set<size_t>>& op2dependences,
         std::map<std::string, std::list<size_t>>* var2min_rw_op,
         size_t cur_op,
         std::string rw_var) {
        // rw_var is inputs or outputs of cur_op
        // this function update the var2min_rw_op set .
        if (var2min_rw_op->find(rw_var) == var2min_rw_op->end()) {
          (*var2min_rw_op)[rw_var] = std::list<size_t>();
        }
        for (auto dep_op : op2dependences.at(cur_op)) {
          var2min_rw_op->at(rw_var).remove(dep_op);
        }
        var2min_rw_op->at(rw_var).push_back(cur_op);
      };

  for (auto op_idx : ops_list) {
    remove_duplicate.clear();
    // step1: update the op2dependences structure
    for (auto& item :
         _ops_ptr->at(op_idx)->Inputs()) {  // for all inputs(read only)
      for (auto var : item.second) {
        if (var2recent_write_op.count(var))
          op2dependences[op_idx].insert(var2recent_write_op[var]);
      }
    }

    for (auto& item : _ops_ptr->at(op_idx)->Outputs()) {  // for all write vars
      for (auto var : item.second) {
        if (var2min_rw_op.count(var)) {
          for (auto dep_op : var2min_rw_op[var]) {
            op2dependences[op_idx].insert(dep_op);
          }
        }
      }
    }

    // step2: update 2 var2xxxx data structure
    for (auto& item : _ops_ptr->at(op_idx)->Outputs()) {  // for all write vars
      for (auto var : item.second) {
        var2recent_write_op[var] = op_idx;
        var2min_rw_op[var] = {static_cast<size_t>(op_idx)};
        remove_duplicate.insert(var);
      }
    }

    for (auto& item :
         _ops_ptr->at(op_idx)->Inputs()) {  // for all inputs(read only)
      for (auto var : item.second) {
        if (remove_duplicate.count(var) ==
            0) {  // var in input list and in output list, so remove it.
          update_var_min_rw_op(op2dependences, &var2min_rw_op, op_idx, var);
        }
      }
    }
  }

  // convert op2dependences to downstream_map directly. op2dependences is op ->
  // it's dependences, we want to get op -> [next ops] map, where ops is the
  // next instruction of op. The size of downstream != size of op2dependences
  // since there are some ops that have no downstream-op.
  for (auto& item : op2dependences) {
    size_t op = item.first;
    for (auto dep_op : item.second) {
      AddDownstreamOp(dep_op, op);
    }
  }
}

void DependencyBuilderSimplify::ShrinkDownstreamMap() {
  // remove unnecessary downstream ops
  // for example, a->b->c
  // a: b, c
  // b: c
  // =>
  // a: b
  // b: c

  // shrink, find the downstream op that has no other op in the
  // downstream list happens before it
  for (size_t i = 0; i < op_num_; ++i) {
    if (op_downstream_map_.find(i) == op_downstream_map_.end()) {
      continue;
    }

    std::set<size_t> minimum_nexts;
    for (size_t item : op_downstream_map_.at(i)) {
      bool not_after_any = true;
      // find the op that is not executed  any
      for (size_t other_item : op_downstream_map_.at(i)) {
        if (OpHappensBefore(other_item, item)) {
          VLOG(8) << "happens_before: " << other_item << "->" << item
                  << ", so skip " << item;
          not_after_any = false;
          break;
        }
      }
      if (not_after_any) {
        VLOG(8) << "downstream op of " << i << ": " << item;
        minimum_nexts.insert(item);
      }
    }
    // NOTE(Ruibiao): op_happens_before will not be changed when shrink
    // downstream map
    op_downstream_map_.at(i) = minimum_nexts;
  }
  VLOG(8) << "Finish shrink downstream map";
  VLOG(8) << "downstream count: " << CountDownstreamMap(op_downstream_map_);
  VLOG(8) << "downstream_map: " << std::endl
          << StringizeDownstreamMap(op_downstream_map_);
}

void DependencyBuilderSimplify::AddDependencyForCoalesceTensorOp() {
  for (auto op_idx : ops_list) {
    if (_ops_ptr->at(op_idx)->Type() == kCoalesceTensor) {
      VLOG(4) << "Add depend for " << kCoalesceTensor << " " << op_idx;
      auto fused_out = _ops_ptr->at(op_idx)->Outputs().at("FusedOutput")[0];
      auto outputs = _ops_ptr->at(op_idx)->Outputs().at("Output");

      auto is_read = [](const std::unique_ptr<OperatorBase>& inst,
                        std::string var_name) -> bool {
        for (auto pair : inst->Inputs()) {
          for (auto item : pair.second) {
            if (item == var_name) {
              return true;
            }
          }
        }
        return false;
      };

      auto is_write = [](const std::unique_ptr<OperatorBase>& inst,
                         std::string var_name) -> bool {
        for (auto pair : inst->Outputs()) {
          for (auto item : pair.second) {
            if (item == var_name) {
              return true;
            }
          }
        }
        return false;
      };

      // find first op that reads fused_out
      auto first_read_fused_out_op = ULLONG_MAX;
      for (auto j = op_idx + 1; j < op_num_; ++j) {
        if (is_read(_ops_ptr->at(j), fused_out)) {
          first_read_fused_out_op = j;
          break;
        }
      }

      if (UNLIKELY(first_read_fused_out_op == ULLONG_MAX)) {
        VLOG(4) << "No op read FusedOutput";
        continue;
      }

      // find ops that write 'outputs' between (op_index,
      // first_read_fused_out_op)
      // add depend: them->first_read_fused_out_op
      for (auto j = op_idx + 1;
           j < static_cast<size_t>(first_read_fused_out_op);
           ++j) {
        for (auto var_name : outputs) {
          if (is_write(_ops_ptr->at(j), var_name)) {
            AddDownstreamOp(j, first_read_fused_out_op);
          }
        }
      }

      // find first op read 'outputs' between (first_read_fused_out_op, end)
      // add depend:  first_read_fused_out_op -> first op that reads 'outputs'

      // special case for consecutive communication ops, for example,
      // FusedOutput = c_sync_calc_stream(FusedOutput)
      // FusedOutput= c_allreduce_sum(FusedOutput)
      // FusedOutput = c_sync_comm_stream(FusedOutput)
      // we should take the last one to add depend instead of
      // 'first_read_fused_out_op'
      size_t target = first_read_fused_out_op;
      for (size_t j = first_read_fused_out_op + 1; j < op_num_; ++j) {
        if (del_c_sync_comm_list.count(j)) {
          continue;
        }
        if (j == target + 1 && IsCommunicationOp(_ops_ptr->at(target).get()) &&
            IsCommunicationOp(_ops_ptr->at(j).get())) {
          VLOG(4) << "Found consecutive communication ops, "
                  << _ops_ptr->at(target)->Type() << " -> "
                  << _ops_ptr->at(j)->Type();
          target = j;
          continue;
        }
        for (auto var_name : outputs) {
          if (is_read(_ops_ptr->at(j), var_name)) {
            AddDownstreamOp(target, j);
          }
        }
      }
    }
  }
}

void DependencyBuilderSimplify::AddDependencyForCommunicationOp() {
  /*
  size_t dependence_op_idx = ULLONG_MAX;
  for (size_t op_idx = start_index_; op_idx < op_num_; ++op_idx) {
    if (IsCommunicationOp(_ops_ptr->at(op_idx)->Type())) {
      if (dependence_op_idx != ULLONG_MAX) {
        AddDownstreamOp(dependence_op_idx, op_idx);
      }
      dependence_op_idx = op_idx;
   }
  }
 */
  const std::string kCSyncComm = "c_sync_comm_stream";
  const std::string kSyncComm = "sync_comm_stream";
  std::vector<size_t> com_op_vector;
  std::vector<size_t> sync_com_op_vector;
  for (auto op_idx : ops_list) {
    if (IsCommunicationOp(_ops_ptr->at(op_idx).get())) {
      com_op_vector.push_back(op_idx);
      for (auto sync_op : sync_com_op_vector) {
        AddDownstreamOp(sync_op, op_idx);
      }
    }

    if (_ops_ptr->at(op_idx)->Type() == kCSyncComm ||
        _ops_ptr->at(op_idx)->Type() == kSyncComm) {
      for (auto com_op_id : com_op_vector) {
        if (com_op_id < op_idx) {
          AddDownstreamOp(com_op_id, op_idx);
        }
      }
      sync_com_op_vector.push_back(op_idx);
      com_op_vector.clear();
    }
  }
}

// make sure that the random op is scheduled sequentially
void DependencyBuilderSimplify::AddDependencyForRandomOp() {
  const std::set<std::string> random_op_set = {"bernoulli",
                                               "poisson",
                                               "multinomial",
                                               "gaussian_random",
                                               "truncated_gaussian_random",
                                               "uniform_random",
                                               "randint",
                                               "randperm",
                                               "exponential",
                                               "sampling_id",
                                               "dropout",
                                               "class_center_sample"};
  size_t dependence_op_idx = ULLONG_MAX;
  for (auto op_idx : ops_list) {
    if (del_c_sync_comm_list.count(op_idx)) {
      continue;
    }
    if (random_op_set.count(_ops_ptr->at(op_idx)->Type())) {
      if (dependence_op_idx != ULLONG_MAX) {
        AddDownstreamOp(dependence_op_idx, op_idx);
      }
      dependence_op_idx = op_idx;
    }
  }
}

// equivalent to add_reader_dependency_pass
void DependencyBuilderSimplify::AddDependencyForReadOp() {
  std::vector<bool> is_startup_ops(op_num_, true);
  for (auto op_idx : ops_list) {
    auto it = op_downstream_map_.find(op_idx);
    if (it != op_downstream_map_.end()) {
      for (size_t downstream_op_idx : it->second) {
        is_startup_ops[downstream_op_idx] = false;
      }
    }
  }

  std::vector<size_t> read_ops;
  std::vector<size_t> startup_ops;
  for (auto op_idx : ops_list) {
    if (_ops_ptr->at(op_idx)->Type() == "read") {
      read_ops.push_back(op_idx);
    }

    if (is_startup_ops[op_idx]) {
      startup_ops.push_back(op_idx);
    }
  }

  for (size_t read_op_idx : read_ops) {
    for (size_t downstream_op_idx : startup_ops) {
      if (read_op_idx != downstream_op_idx &&
          !OpHappensBefore(downstream_op_idx, read_op_idx)) {
        AddDownstreamOp(read_op_idx, downstream_op_idx);
      }
    }
  }
}

// for speed up com and calc parallel
void DependencyBuilderSimplify::AddDependencyForBroadcastOp() {
  const std::string broadcast = "c_broadcast";
  const std::string kCSyncComm = "c_sync_comm_stream";
  const std::string kSyncComm = "sync_comm_stream";
  std::vector<size_t> op_between_broadcast_and_sync;
  std::vector<size_t> op_broadcast;
  size_t index = 0;
  for (auto op_idx : ops_list) {
    if (_ops_ptr->at(op_idx)->Type() == broadcast) {
      op_broadcast.push_back(op_idx);
      op_between_broadcast_and_sync.clear();
    } else if (_ops_ptr->at(op_idx)->Type() == kCSyncComm ||
               _ops_ptr->at(op_idx)->Type() == kSyncComm) {
      op_broadcast.clear();
      for (auto op : op_between_broadcast_and_sync) {
        AddDownstreamOp(op, op_idx);
      }
      op_between_broadcast_and_sync.clear();
      index = 0;
    } else if (op_broadcast.size() > 0) {
      op_between_broadcast_and_sync.push_back(op_idx);
      AddDownstreamOp(op_broadcast[index++ % op_broadcast.size()], op_idx);
    }
  }
}

void DependencyBuilderSimplify::SetSameStream() {
  std::string use_calc_stream("use_calc_stream");
  // for unsharing
  for (size_t i = start_index_; i < op_num_; i++) {
    std::string op_name = _ops_ptr->at(i)->Type();
    if (op_name == "c_allreduce_sum") {
      _ops_ptr->at(i)->SetAttr(use_calc_stream, true);
    }
  }

  size_t last_pos = -1;
  std::set<std::string> inputs;
  // for sharing
  for (size_t i = start_index_; i < op_num_; i++) {
    std::string op_name = _ops_ptr->at(i)->Type();
    if (op_name == "reduce" && _ops_ptr->at(i)->Attr<int>("reduce_type") ==
                                   static_cast<int>(phi::ReduceType::kRedSum)) {
      _ops_ptr->at(i)->SetAttr(use_calc_stream, true);
      for (auto it : _ops_ptr->at(i)->Inputs()) {
        for (auto var : it.second) {
          inputs.insert(var);
        }
      }
      last_pos = i;
    }
  }

  bool is_sync_for_reduce = true;
  if (last_pos > 0) {
    for (size_t i = last_pos + 1; i < op_num_; i++) {
      std::string op_name = _ops_ptr->at(i)->Type();
      if (op_name == "c_sync_comm_stream" || op_name == "sync_comm_stream") {
        for (auto it : _ops_ptr->at(i)->Inputs()) {
          for (auto var : it.second) {
            if (inputs.count(var) == 0) {
              is_sync_for_reduce = false;
              VLOG(1) << var << " not in c_sync_comm_stream";
              break;
            }
          }
        }
        if (is_sync_for_reduce) {
          del_c_sync_comm_list.insert(i);
          if (FLAGS_enable_dependency_builder_debug_info) {
            VLOG(0) << " del op c_sync_comm_stream index is " << i;
          }
        }
        break;
      }
    }
  }
}

// get_new_executor_order  by dfs
std::vector<size_t> DependencyBuilderSimplify::get_new_executor_order() {
  PADDLE_ENFORCE_EQ(
      is_build_,
      true,
      common::errors::AlreadyExists("The op dependency has not been built"));
  std::vector<size_t> new_order;
  std::vector<bool> is_visit(op_num_, false);
  std::vector<size_t> adam_vector;
  std::priority_queue<std::pair<size_t, size_t>> adam_pq;
  const std::string push_op = "push";
  std::vector<bool> useful_op(op_num_, false);
  for (size_t i = start_index_; i < op_num_; i++) {
    int op_role = _ops_ptr->at(i)->Attr<int>("op_role");
    std::string op_name = _ops_ptr->at(i)->Type();
    if (op_role == static_cast<int>(OpRole::kOptimize) ||
        op_name.find(push_op) != std::string::npos) {
      adam_vector.push_back(i);
      useful_op[i] = true;
      adam_pq.push(std::make_pair(-op_before_num[i], i));
    }
  }
  for (size_t i = start_index_; i < op_num_; i++) {
    for (auto j : adam_vector)
      if (op_happens_before_[i][j]) {
        useful_op[i] = true;
        break;
      }
  }
  std::set<size_t> not_useful_op;
  for (size_t i = start_index_; i < op_num_; i++) {
    if (useful_op[i] == false) {
      not_useful_op.insert(i);
      if (FLAGS_enable_dependency_builder_debug_info) {
        VLOG(0) << "not useful op " << _ops_ptr->at(i)->Type() << "_" << i;
      }
    }
  }
  for (auto del_op : del_c_sync_comm_list) {
    if (not_useful_op.count(del_op) == 0) {
      if (FLAGS_enable_dependency_builder_debug_info) {
        VLOG(0) << " error " << del_op << " not in useful_op";
      }
    }
  }

  // std::vector<bool> is_calc_ed(op_num_, false);
  for (size_t op_idx = 0; op_idx < start_index_; ++op_idx) {
    new_order.push_back(op_idx);
    is_visit[op_idx] = true;
  }

  std::vector<size_t> dependency_count(op_num_, 0);
  for (auto it : op_downstream_map_) {
    for (auto op_idx : it.second) {
      dependency_count[op_idx]++;
    }
  }
  std::stack<size_t> s;
  std::priority_queue<std::pair<size_t, size_t>> pq;

  for (size_t op_idx = op_num_ - 1; op_idx >= start_index_; op_idx--) {
    if (dependency_count[op_idx] == 0) {
      pq.push(std::make_pair(op_behind_num[op_idx], op_idx));
    }
  }
  while (!pq.empty()) {
    auto op_idx = pq.top().second;
    s.push(op_idx);
    pq.pop();
  }
  while (!s.empty()) {
    auto current = s.top();
    s.pop();
    if (is_visit[current] == false) {
      if (!not_useful_op.count(current)) {
        new_order.push_back(current);
      }
      is_visit[current] = true;
      for (auto it = op_downstream_map_[current].rbegin();
           it != op_downstream_map_[current].rend();
           it++) {
        if (--dependency_count[*it] == 0 && !not_useful_op.count(current)) {
          pq.push(std::make_pair(op_behind_num[*it], *it));
          // s.push(*it);
        }
      }
      while (!pq.empty()) {
        auto op_idx = pq.top().second;
        s.push(op_idx);
        pq.pop();
      }
    }
  }

  PADDLE_ENFORCE_EQ(
      new_order.size(),
      op_num_ - not_useful_op.size(),
      common::errors::AlreadyExists("new_order size not equal op num"));
  if (FLAGS_enable_dependency_builder_debug_info) {
    std::stringstream ss;
    ss << " new order [ ";
    for (auto index : new_order) {
      ss << index << " ";
    }
    ss << " ] ";
    VLOG(0) << ss.str();
  }
  return new_order;
}

void DependencyBuilderSimplify::GetOpBehindNum() {
  for (size_t i = 0; i < op_num_; i++) {
    if (i < start_index_) {
      op_behind_num.push_back(0);
      op_before_num.push_back(0);
    } else {
      size_t behind_num = 0;
      size_t before_num = 0;
      for (size_t j = start_index_; j < i; j++) {
        before_num += static_cast<int>(op_happens_before_[j][i]);
      }

      for (size_t j = i + 1; j < op_num_; j++) {
        behind_num += static_cast<int>(op_happens_before_[i][j]);
      }
      op_behind_num.push_back(behind_num);
      op_before_num.push_back(before_num);
    }
  }
}

void DependencyBuilderSimplify::AddDownstreamOp(size_t prior_op_idx,
                                                size_t posterior_op_idx) {
  PADDLE_ENFORCE_EQ(
      OpHappensBefore(posterior_op_idx, prior_op_idx),
      false,
      common::errors::Unavailable(
          "Can not add dependency %d->%d because %d is run before %d",
          prior_op_idx,
          posterior_op_idx,
          posterior_op_idx,
          prior_op_idx));

  std::set<size_t>& downstream_ops = op_downstream_map_[prior_op_idx];
  // NOTE(Ruibiao): Here the downstream map shrinking is best-effort, therefore
  // ShrinkDownstreamMap after BuildDownstreamMap is still helpful. For example,
  // a->c will not be shrunk in the following case: AddDownstreamOp(a, b) ->
  // AddDownstreamOp(a, c) -> AddDownstreamOp(b, c), it should be shrunk by
  // ShrinkDownstreamMap.
  for (size_t op_idx : downstream_ops) {
    if (OpHappensBefore(op_idx, posterior_op_idx)) {
      VLOG(7) << "Find dependencies " << prior_op_idx << "->" << op_idx << "->"
              << posterior_op_idx << ", skip adding " << prior_op_idx << "->"
              << posterior_op_idx;
      return;
    }
  }
  downstream_ops.insert(posterior_op_idx);

  std::vector<size_t>& prior_of_prior = ops_before_[prior_op_idx];
  std::vector<size_t>& posterior_of_posterior = ops_behind_[posterior_op_idx];

  auto update_op_happen_before = [this](size_t prior_op_idx,
                                        size_t posterior_op_idx) {
    if (!op_happens_before_[prior_op_idx][posterior_op_idx]) {
      op_happens_before_[prior_op_idx][posterior_op_idx] = true;
      ops_before_[posterior_op_idx].push_back(prior_op_idx);
      ops_behind_[prior_op_idx].push_back(posterior_op_idx);
    }
  };

  update_op_happen_before(prior_op_idx, posterior_op_idx);

  // All ops before prior-op are also before posterior-op
  for (size_t op_idx : prior_of_prior) {
    update_op_happen_before(op_idx, posterior_op_idx);
  }

  // All ops after posterior-op are also after prior-op
  for (size_t op_idx : posterior_of_posterior) {
    update_op_happen_before(prior_op_idx, op_idx);
  }
}

}  // namespace paddle::framework::interpreter
