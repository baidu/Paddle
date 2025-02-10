// Copyright (c) 2024 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/ir/group_schedule/search/measurer.h"

#include "paddle/cinn/hlir/dialect/operator/ir/op_dialect.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/add_cinn_pass.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/transforms/build_cinn_pass.h"
#include "paddle/fluid/pir/transforms/pd_op_to_kernel_pass.h"
#include "paddle/pir/include/core/builtin_dialect.h"
#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/core/program.h"
#include "paddle/pir/include/dialect/shape/ir/shape_dialect.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_manager.h"

COMMON_DECLARE_bool(print_ir);
PD_DECLARE_string(cinn_kernel_execution_label);

namespace cinn {
namespace ir {
namespace search {

std::shared_ptr<pir::PassManager> CreatePassManager() {
  pir::IrContext* ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<cinn::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<pir::shape::ShapeDialect>();
  auto pass_manager = std::make_shared<pir::PassManager>(ctx);
  if (FLAGS_print_ir) {
    pass_manager->EnableIRPrinting();
  }
  return pass_manager;
}

Measurer::Measurer(::pir::Program* program) : program_(program) {
  std::stringstream ss;
  ss << *program_;
  compile_label_ = "Compile Program\n" + ss.str();
  execute_label_ = "Execute Program\n" + ss.str();
}

void Measurer::Compile() {
  common::PerformanceStatisticsStart(compile_label_);
  ::pir::IrMapping ir_mapping;
  std::shared_ptr<::pir::Program> program_cloned = program_->Clone(ir_mapping);
  cinn::dialect::ir::ApplyCinnPass(program_cloned.get(), CreatePassManager);
  kernel_program_ = std::move(
      paddle::dialect::PdOpLowerToKernelPass(program_cloned.get(), place_));
  executor_.reset(new paddle::framework::InterpreterCore(
      place_, {"out@fetch"}, kernel_program_->block(), exe_scope_.get()));
  common::PerformanceStatisticsEnd(compile_label_);
}

std::string ConcatShapeAsLabel(
    const std::unordered_map<std::string, std::vector<int64_t>>&
        input_name_and_shape) {
  std::stringstream ss;
  ss << "Shape  ";
  for (const auto item : input_name_and_shape) {
    ss << item.first << "=";
    for (int n : item.second) {
      ss << n << "x";
    }
  }
  std::string label = ss.str();
  label.pop_back();
  return label;
}

void Measurer::Run(const std::unordered_map<std::string, std::vector<int64_t>>&
                       input_name_and_shape,
                   int repeat) {
  std::vector<std::string> input_names;
  std::vector<phi::DenseTensor> input_tensors;
  for (const auto item : input_name_and_shape) {
    input_names.push_back(item.first);
    auto tensor =
        executor_->local_scope()->FindVar(item.first)->Get<phi::DenseTensor>();
    phi::DDim ddim(item.second.data(), item.second.size());
    tensor.ResizeAndAllocate(ddim);
    float* data = tensor.mutable_data<float>(ddim, place_);
    input_tensors.push_back(tensor);
  }
  std::string input_shape_label = ConcatShapeAsLabel(input_name_and_shape);

  common::PerformanceStatistician& ps =
      common::PerformanceStatistician::Instance();
  for (int i = 0; i < repeat; ++i) {
    ps.Start(execute_label_ + "\n" + input_shape_label);
    executor_->Run(input_names, input_tensors, true);
    ps.End(execute_label_ + "\n" + input_shape_label);
  }
}

MeasureResult Measurer::Result() const {
  MeasureResult result;
  common::PerformanceStatistician& ps =
      common::PerformanceStatistician::Instance();

  auto compile_durations =
      ::common::PerformanceReporter::ExtractDuration(ps.Record(compile_label_));
  auto total_execute_durations = ::common::PerformanceReporter::ExtractDuration(
      ps.RecordWithSubLabel(execute_label_));
  auto kernel_record = ps.Record(FLAGS_cinn_kernel_execution_label);
  auto kernel_execute_durations =
      ::common::PerformanceReporter::ExtractDuration(kernel_record);

  auto compile_time = ::common::PerformanceReporter::Mean(compile_durations);
  auto avg_total_execute_time =
      ::common::PerformanceReporter::Mean(total_execute_durations);
  VLOG(6) << " report: "
          << ::common::PerformanceReporter::Report(kernel_record);
  auto avg_kernel_execute_time =
      ::common::PerformanceReporter::TrimMean(kernel_execute_durations);

  result.compile_time = compile_time;
  result.avg_total_execute_time = avg_total_execute_time;
  result.avg_kernel_execute_time = avg_kernel_execute_time;

  ps.Reset();
  return result;
}

}  // namespace search
}  // namespace ir
}  // namespace cinn
