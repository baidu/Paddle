// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/executor_cache.h"

#include "paddle/common/flags.h"
#include "paddle/common/macros.h"
#include "paddle/fluid/framework/new_executor/interpretercore.h"
#include "paddle/fluid/framework/op_info.h"
#include "paddle/fluid/ir_adaptor/translator/translate.h"
#include "paddle/fluid/pir/transforms/general/inplace_pass.h"
#include "paddle/fluid/pir/transforms/general/remove_shadow_feed_pass.h"
#include "paddle/fluid/pir/transforms/pd_op_to_kernel_pass.h"
#include "paddle/pir/include/core/program.h"
#include "paddle/pir/include/core/value.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_manager.h"
#include "paddle/pir/include/pass/pass_registry.h"

DECLARE_FILE_SYMBOLS(print_statistics);

COMMON_DECLARE_bool(pir_apply_inplace_pass);
COMMON_DECLARE_bool(print_ir);
COMMON_DECLARE_string(enable_custom_engine);

namespace paddle::framework {
class ProgramDesc;
}  // namespace paddle::framework

namespace paddle::framework::details {

void AppendSkipDeletionVars(const std::vector<std::string> &append_vars,
                            std::set<std::string> *all_vars) {
  for (auto &var : append_vars) {
    all_vars->insert(var);
  }
}

std::set<std::string> ParseSafeEagerDeletionSkipVarsSet(
    const ProgramDesc &backward_program, bool skip_no_need_buffer) {
  std::set<std::string> skip_eager_delete_vars;
  auto backward_ops = backward_program.Block(0).AllOps();
  auto &op_info_map = OpInfoMap::Instance();
  std::unordered_set<std::string> op_outputs;
  std::unordered_set<std::string> op_inputs;
  std::unordered_set<std::string> no_need_buffer_ins;
  for (auto op : backward_ops) {
    VLOG(4) << "parse op type: " << op->Type();
    if (op->Type() == "share_buffer") {
      VLOG(1) << "skip share_buffer op";
      continue;
    }
    // NOTE: skip NoNeedBufferVars of grad_op and GC its memory in advance.
    auto &op_info = op_info_map.Get(op->Type());
    auto &inferer = op_info.NoNeedBufferVarsInferer();
    no_need_buffer_ins.clear();
    // TODO(Aurelius84): Need remove skip_no_need_buffer after cinn fix this
    // problem.
    if (inferer != nullptr && !skip_no_need_buffer) {
      no_need_buffer_ins =
          inferer(op->Inputs(), op->Outputs(), op->GetAttrMap());
    }
    for (auto &in_names : op->Inputs()) {
      if (no_need_buffer_ins.count(in_names.first) == 0) {
        for (auto &in_name : in_names.second) {
          op_inputs.emplace(in_name);
        }
      } else {
        VLOG(2) << op->Type() << " has no_need_buffer_in: " << in_names.first
                << " , skip it.";
      }
    }
    for (const std::string &out_arg_name : op->OutputArgumentNames()) {
      op_outputs.emplace(out_arg_name);
    }
  }
  for (const std::string &var_name : op_inputs) {
    VLOG(4) << "parse op.input: " << var_name;
    if (op_outputs.find(var_name) == op_outputs.end()) {
      VLOG(1) << "skip eager var: " << var_name;
      skip_eager_delete_vars.insert(var_name);
    }
  }
  VLOG(1) << "Found skip_eager_delete_vars: " << skip_eager_delete_vars.size();
  return skip_eager_delete_vars;
}
}  // namespace paddle::framework::details
namespace paddle::framework {

// C++11 removes the need for manual locking. Concurrent execution shall wait if
// a static local variable is already being initialized.
// https://stackoverflow.com/questions/11711920/how-to-implement-multithread-safe-singleton-in-c11-without-using-mutex

int64_t hash_with_seed(int64_t value, int64_t seed) {
  return value + 0x9e3779b9 + (value << 6) + (seed >> 2);
}

InterpreterCoreInfoCache &InterpreterCoreInfoCache::Instance() {
  static InterpreterCoreInfoCache g_info_cache;
  return g_info_cache;
}

std::shared_ptr<InterpreterCore> CreateProgramInterpreterCoreInfoToCache(
    const ProgramDesc &program_desc,
    const phi::Place &place,
    bool is_grad,
    int64_t program_id,
    framework::Scope *scope,
    const int64_t &place_hash_key) {
  auto &cache = framework::InterpreterCoreInfoCache::Instance();
  if (cache.Size() > 256000u /* max_cached_size*/) {
    PADDLE_THROW(common::errors::Fatal(
        "The cached info size has exceeded max_cached_size: 256000, "
        "which will cause error. "));
  }
  interpreter::ExecutionConfig execution_config;
  execution_config.create_local_scope = false;
  execution_config.used_for_jit = true;

  std::shared_ptr<InterpreterCore> core = nullptr;

  core.reset(new InterpreterCore(
      place, program_desc.Block(0), scope, execution_config));

  auto &cached_value = cache.GetMutable(
      program_id, scope, place_hash_key, is_grad, /*in_pir_mode=*/false);
  cached_value.core_ = core;
  return core;
}

std::shared_ptr<InterpreterCore> CreatePirInterpreterCoreInfoToCache(
    std::unique_ptr<::pir::Program> ir_program,
    const phi::Place &place,
    bool is_grad,
    int64_t program_id,
    framework::Scope *scope,
    const int64_t &place_hash_key,
    bool used_for_sot) {
  auto &cache = framework::InterpreterCoreInfoCache::Instance();
  if (cache.Size() > 256000u /* max_cached_size*/) {
    PADDLE_THROW(common::errors::Fatal(
        "The cached info size has exceeded max_cached_size: 256000, "
        "which will cause error. "));
  }
  interpreter::ExecutionConfig execution_config;
  execution_config.create_local_scope = false;
  execution_config.used_for_jit = true;
  execution_config.used_for_sot = used_for_sot;

  std::shared_ptr<InterpreterCore> core = nullptr;

  core.reset(new InterpreterCore(
      place, {}, ir_program->block(), scope, execution_config));

  auto &cached_value = cache.GetMutable(
      program_id, scope, place_hash_key, is_grad, /*in_pir_mode=*/true);
  cached_value.core_ = core;
  cached_value.ir_prog_ = std::move(ir_program);
  return core;
}

bool TensorSortHelper(const paddle::Tensor &t1, const paddle::Tensor &t2) {
  return t1.name() < t2.name();
}

std::unique_ptr<::pir::Program> ApplyIrPass(
    ::pir::Program *program,
    phi::Place place,
    const std::set<std::string> &no_need_buffer_names) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
  if (!FLAGS_enable_custom_engine.empty()) {
    std::string custom_engine_translate_pass = FLAGS_enable_custom_engine;
    std::istringstream ss(custom_engine_translate_pass);
    std::string pass;
    std::vector<std::string> passes;

    while (std::getline(ss, pass, ',')) {
      passes.push_back(pass);
      VLOG(4) << "Add CustomEngine pass : " << pass;
    }

    ::pir::PassManager pass_pm(::pir::IrContext::Instance(), 3);
    for (std::string custom_pass : passes) {
      pass_pm.AddPass(pir::PassRegistry::Instance().Get(custom_pass));
      pass_pm.Run(program);
    }
  }
#endif
  auto ir_res = paddle::dialect::PdOpLowerToKernelPass(program, place);

  if (FLAGS_pir_apply_inplace_pass) {
    ::pir::PassManager pm(::pir::IrContext::Instance(), 3);
    pm.AddPass(::pir::CreateInplacePass(no_need_buffer_names));
    pm.Run(ir_res.get());

    if (FLAGS_print_ir) {
      std::cout << "IR After inplace -------------------" << std::endl;
      std::cout << *ir_res << std::endl;
    }
  }

  return ir_res;
}

std::unique_ptr<::pir::Program> ApplyRemoveShadowFeedPass(
    std::unique_ptr<::pir::Program> program,
    const pir::Block *block,
    const phi::Place &place,
    const paddle::framework::Scope *scope) {
  ::pir::PassManager pm(::pir::IrContext::Instance(), 3);
  auto pass = ::pir::CreateRemoveShadowFeedPass();
  pass->SetNotOwned("top_block", block);
  pass->SetNotOwned(pir::Pass::kPlaceAttr, &place);
  pass->SetNotOwned(pir::Pass::kParamScopeAttr, scope);
  pm.AddPass(std::move(pass));
  pm.Run(program.get());

  if (FLAGS_print_ir) {
    std::cout << "IR After RemoveShadowFeedPass -------------------"
              << std::endl;
    std::cout << *program << std::endl;
  }

  return program;
}

std::unique_ptr<::pir::Program> ConstructForwardIrProgram(
    const paddle::framework::BlockDesc *forward_global_block,
    const paddle::framework::BlockDesc *backward_global_block,
    const std::vector<std::string> &output_names,
    const std::vector<paddle::Tensor> &x,
    const std::vector<std::string> &x_names,
    const std::vector<paddle::Tensor> &params,
    const phi::Place &place) {
  std::set<std::string> set_output_names;
  auto local_program =
      paddle::framework::ProgramDesc(*(forward_global_block->Program()));

  for (auto op_desc : local_program.Block(0).AllOps()) {
    for (const auto &n : op_desc->Outputs()) {
      const auto &input_var_names = n.second;
      for (const auto &var_name : input_var_names) {
        set_output_names.insert(var_name);
      }
    }
  }

  // add data op to program
  auto *block = local_program.MutableBlock(0);
  for (size_t i = 0; i < x.size(); ++i) {
    auto &name = x_names[i];
    auto &in_t = x[i];
    if (block->FindVarRecursive(name) == nullptr) {
      continue;
    }
    auto p = in_t.place().GetType();

    auto op_desc = block->PrependOp();
    op_desc->SetType("data");
    op_desc->SetAttr("shape", std::vector<int64_t>());
    // TODO(phlrain) : using tensor dtype
    op_desc->SetAttr("dtype", 0);
    op_desc->SetAttr("place", static_cast<int>(p));
    if (p == phi::AllocationType::CUSTOM) {
      op_desc->SetAttr("place_device_id", in_t.place().GetDeviceId());
      op_desc->SetAttr("place_device_type", in_t.place().GetDeviceType());
    }

    op_desc->SetAttr("name", name);
    op_desc->SetOutput("out", {name});
  }

  std::set<std::string> input_param_names;
  auto sorted_params = params;
  std::sort(sorted_params.begin(), sorted_params.end(), TensorSortHelper);
  for (auto &param : sorted_params) {
    auto &name = param.name();
    auto p = param.place().GetType();

    auto op_desc = local_program.MutableBlock(0)->PrependOp();
    op_desc->SetType("data");
    op_desc->SetAttr("shape", std::vector<int64_t>());
    // TODO(phlrain) : using tensor dtype
    op_desc->SetAttr("dtype", 0);
    op_desc->SetAttr("place", static_cast<int>(p));
    if (p == phi::AllocationType::CUSTOM) {
      op_desc->SetAttr("place_device_id", param.place().GetDeviceId());
      op_desc->SetAttr("place_device_type", param.place().GetDeviceType());
    }
    op_desc->SetAttr("name", name);
    op_desc->SetOutput("out", {name});

    input_param_names.insert(name);
  }

  std::set<std::string> set_parameter_names;
  for (auto &t : output_names) {
    set_parameter_names.insert(t);
  }

  if (backward_global_block != nullptr) {
    for (auto op_desc : backward_global_block->Program()->Block(0).AllOps()) {
      for (const auto &n : op_desc->Inputs()) {
        const auto &input_var_names = n.second;
        for (const auto &var_name : input_var_names) {
          set_parameter_names.insert(var_name);
        }
      }
    }
  }

  for (auto &name : set_parameter_names) {
    if (!set_output_names.count(name)) {
      continue;
    }

    if (input_param_names.count(name)) {
      continue;
    }

    auto op_desc = local_program.MutableBlock(0)->AppendOp();
    op_desc->SetType("shadow_output");
    op_desc->SetAttr("name", name);
    op_desc->SetInput("x", {name});
    op_desc->SetOutput("out", {"@EMPTY@"});
  }
  auto program = TranslateLegacyProgramToProgram(local_program);

  return ApplyIrPass(program.get(), place, {});
}

std::unique_ptr<::pir::Program> ConstructBackwardIrProgram(
    const paddle::framework::BlockDesc *backward_global_block,
    const std::vector<paddle::Tensor> &out_grad,
    const std::vector<paddle::Tensor *> &x_grad,
    const std::vector<paddle::Tensor *> &params_grad,
    const paddle::framework::Scope *scope,
    const phi::Place &place) {
  auto local_program =
      paddle::framework::ProgramDesc(*(backward_global_block->Program()));

  // get feed with data
  std::set<std::string> set_parameter_names;
  for (auto op_desc : backward_global_block->Program()->Block(0).AllOps()) {
    for (const auto &n : op_desc->Inputs()) {
      const auto &input_var_names = n.second;
      for (const auto &var_name : input_var_names) {
        set_parameter_names.insert(var_name);
      }
    }
  }

  for (auto &var_name : set_parameter_names) {
    if (scope->FindVar(var_name)) {
      auto tensor = scope->FindVar(var_name)->Get<phi::DenseTensor>();
      phi::AllocationType p = place.GetType();
      if (tensor.initialized()) {
        p = tensor.place().GetType();
      }

      if (var_name == "@EMPTY@") {
        continue;
      }
      auto op_desc = local_program.MutableBlock(0)->PrependOp();
      op_desc->SetType("data");
      op_desc->SetAttr("shape", std::vector<int64_t>());
      // TODO(phlrain) : using tensor dtype
      op_desc->SetAttr("dtype", 0);
      op_desc->SetAttr("place", static_cast<int>(p));
      if (p == phi::AllocationType::CUSTOM) {
        op_desc->SetAttr("place_device_id", tensor.place().GetDeviceId());
        op_desc->SetAttr("place_device_type", tensor.place().GetDeviceType());
      }

      op_desc->SetAttr("name", var_name);
      op_desc->SetOutput("out", {var_name});
    }
  }

  std::vector<std::string> param_grad_names;
  for (auto &p_g : params_grad) {
    param_grad_names.push_back(p_g->name());
  }

  for (auto &t : x_grad) {
    param_grad_names.push_back(t->name());
  }

  std::sort(param_grad_names.begin(), param_grad_names.end());
  for (auto &name : param_grad_names) {
    if (name == "@EMPTY@") {
      continue;
    }
    auto op_desc = local_program.MutableBlock(0)->AppendOp();
    op_desc->SetType("shadow_output");
    op_desc->SetAttr("name", name);
    op_desc->SetInput("x", {name});
    op_desc->SetOutput("out", {"@EMPTY@"});
  }

  auto program = TranslateLegacyProgramToProgram(local_program);

  auto res = paddle::dialect::PdOpLowerToKernelPass(program.get(), place);

  if (FLAGS_pir_apply_inplace_pass) {
    ::pir::PassManager pm(::pir::IrContext::Instance(), 3);
    pm.AddPass(::pir::CreateInplacePass());
    if (VLOG_IS_ON(6)) {
      pm.EnableIRPrinting();
      pm.EnablePrintStatistics();
    }
    pm.Run(res.get());
    if (FLAGS_print_ir) {
      std::cout << "IR After inplace -------------------" << std::endl;
      std::cout << *res << std::endl;
    }
  }

  return res;
}

}  // namespace paddle::framework
