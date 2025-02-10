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

#include "paddle/cinn/hlir/dialect/operator/transforms/add_cinn_pass.h"

#include <chrono>
#include "paddle/common/errors.h"
#include "paddle/common/flags.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/utils/shape_analysis_utils.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/pir/include/core/ir_context.h"
#include "paddle/pir/include/core/program.h"
#include "paddle/pir/include/dialect/shape/ir/shape_dialect.h"
#include "paddle/pir/include/dialect/shape/transforms/shape_optimization_pass.h"
#include "paddle/pir/include/dialect/shape/utils/shape_analysis.h"
#include "paddle/pir/include/pass/pass_manager.h"

#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/op_dialect.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/accuracy_check_pass.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/add_broadcast_to_elementwise_pass.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/add_store_in_fusion_op_pass.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/add_store_in_group_op_pass.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/cinn_group_cluster_pass.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/conv2d_transpose_filter_pass.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/convert_fa_to_qkvmha_pass.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/convert_memory_effec_attn_to_flash_attn_pass.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/dynamic_reshape_pass.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/fold_manipulation_ops_pass.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/fuse_parallel_matmul_pass.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/fuse_shape_ops_into_generate_shape_op_pass.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/fusion_fallback_pass.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/group_merge/convert_dynamic_to_static_dim_pass.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/group_merge/convert_static_dim_to_dynamic_pass.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/group_merge/move_generate_shape_ops_to_prologue_pass.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/group_merge/simplify_dim_expr_pass.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/group_merge/single_op_fallback_to_phi.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/insert_broadcast_pass.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/lowering_pass/lower_cinn_fusion_op_pass.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/pd_to_cinn_pass.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/pir_to_py_code_converter.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/reduce_as_to_sum_pass.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/remove_assign_out_pass.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/replace_dynamic_expand_pass.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/replace_zero_scale_to_full_pass.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/shape_ops_fallback_to_phi_pass.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/specify_input_dynamic_dim_util.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/split_generate_shape_into_shape_ops_pass.h"
#include "paddle/fluid/pir/transforms/build_cinn_pass.h"
#include "paddle/fluid/pir/transforms/general/common_subexpression_elimination_pass.h"
#include "paddle/fluid/pir/transforms/general/dead_code_elimination_pass.h"
#include "paddle/fluid/pir/transforms/gpu/fused_gemm_epilogue_pass.h"

COMMON_DECLARE_bool(cinn_specify_input_dynamic_dim);
COMMON_DECLARE_string(cinn_input_dynamic_dim_spec_file);
COMMON_DECLARE_bool(print_ir);
COMMON_DECLARE_bool(disable_dyshape_in_train);
COMMON_DECLARE_bool(enable_cinn_accuracy_check);
COMMON_DECLARE_bool(enable_fuse_parallel_matmul_pass);
COMMON_DECLARE_bool(enable_fusion_fallback);
COMMON_DECLARE_bool(logging_pir_py_code_dump_symbolic_dims);

namespace cinn::dialect::ir {

namespace {
bool HasDynamicShape(const pir::Program& program) {
  if (FLAGS_disable_dyshape_in_train) {
    return false;
  }
  for (const auto& op : *program.block()) {
    if (op.isa<pir::CombineOp>()) {
      continue;
    }
    for (uint32_t i = 0; i < op.num_results(); ++i) {
      if (op.result(i) && op.result(i).type()) {
        auto shape_type =
            op.result(i).type().dyn_cast<pir::ShapedTypeInterface>();
        if (shape_type && shape_type.IsDynamicShape()) {
          return true;
        }
      }
    }
  }
  return false;
}
}  // namespace

void ApplyShapeOptimizationPass(
    ::pir::Program* program,
    const std::function<std::shared_ptr<::pir::PassManager>()>&
        CreatePassManager) {
  std::shared_ptr<pir::PassManager> pass_manager = CreatePassManager();
  pir::OriginalAttributesFilter::Instance().SetOriginalAttributesMap(
      paddle::dialect::GetAllOpOriginalAttributes());

  if (FLAGS_cinn_specify_input_dynamic_dim) {
    PADDLE_ENFORCE_NE(
        FLAGS_cinn_input_dynamic_dim_spec_file,
        "",
        ::common::errors::InvalidArgument(
            "'FLAGS_cinn_input_dynamic_dim_spec_file' should not be empty "
            "when using FLAGS_cinn_specify_input_dynamic_dim."));
    SpecifyInputDynamicDimFromFile(program,
                                   FLAGS_cinn_input_dynamic_dim_spec_file);
  }
  pass_manager->AddPass(pir::CreateShapeOptimizationPass());
  pass_manager->Run(program);
}

void ApplyPdToCinnPass(
    ::pir::Program* program,
    const std::function<std::shared_ptr<::pir::PassManager>()>&
        CreatePassManager) {
  std::shared_ptr<pir::PassManager> pass_manager = CreatePassManager();
  pass_manager->AddPass(cinn::dialect::ir::CreateReduceAsToSumPass());
  pass_manager->AddPass(cinn::dialect::ir::CreateReplaceZeroScaleToFullPass());
  pass_manager->AddPass(pir::CreateFusedGemmEpiloguePass());
  if (FLAGS_enable_fuse_parallel_matmul_pass) {
    pass_manager->AddPass(cinn::dialect::ir::CreateFuseParallelMatmulPass());
  }
  pass_manager->AddPass(cinn::dialect::ir::CreateRemoveAssignOutPass());
  pass_manager->AddPass(cinn::dialect::ir::CreateConv2dTransposeFilterPass());
  pass_manager->AddPass(cinn::dialect::ir::CreateConvertMEA2FAPass());
  pass_manager->AddPass(cinn::dialect::ir::CreateConvertFA2QKVMHAPass());
  pass_manager->AddPass(cinn::dialect::ir::CreatePdOpToCinnOpPass());

  pass_manager->AddPass(pir::CreateDeadCodeEliminationPass());

  pass_manager->Run(program);
}

void ApplyCinnPreprocessPass(
    ::pir::Program* program,
    const std::function<std::shared_ptr<::pir::PassManager>()>&
        CreatePassManager) {
  std::shared_ptr<pir::PassManager> pass_manager = CreatePassManager();
  bool has_dynamic_shape = HasDynamicShape(*program);

  if (has_dynamic_shape) {
    pass_manager->AddPass(
        cinn::dialect::ir::CreateFuseShapeOpsIntoGenerateShapeOpPass());
    pass_manager->AddPass(pir::CreateDeadCodeEliminationPass());
  }
  pass_manager->Run(program);
}

void ApplyBuildGroupOpPass(
    ::pir::Program* program,
    const std::function<std::shared_ptr<pir::PassManager>()>&
        CreatePassManager) {
  std::shared_ptr<pir::PassManager> pass_manager = CreatePassManager();
  pass_manager->AddPass(cinn::dialect::ir::CreateFoldManipulationOpsPass());

  pass_manager->AddPass(pir::CreateBuildCinnPass());

  pass_manager->Run(program);
}

void ApplyGroupOpPass(::pir::Program* program,
                      const std::function<std::shared_ptr<pir::PassManager>()>&
                          CreatePassManager) {
  std::shared_ptr<pir::PassManager> pass_manager = CreatePassManager();
  pass_manager->AddPass(
      cinn::dialect::ir::CreateAddBroadcastToElementwisePass());
  pass_manager->AddPass(cinn::dialect::ir::CreateInsertBroadcastPass());
  pass_manager->AddPass(
      cinn::dialect::ir::CreateFuseShapeOpsIntoGenerateShapeOpPass());
  pass_manager->AddPass(
      cinn::dialect::ir::CreateMoveGenerateShapeOpsToProloguePass());

  pass_manager->AddPass(cinn::dialect::ir::CreateDynamicReshapeOpPass());
  pass_manager->AddPass(cinn::dialect::ir::CreateFoldManipulationOpsPass());
  pass_manager->AddPass(pir::CreateDeadCodeEliminationPass());

  pass_manager->Run(program);
}

void ApplyDivideGroupOpToFusionOpPass(
    ::pir::Program* program,
    const std::function<std::shared_ptr<pir::PassManager>()>&
        CreatePassManager) {
  std::shared_ptr<pir::PassManager> pass_manager = CreatePassManager();
  pass_manager->AddPass(cinn::dialect::ir::CreateAddStoreInGroupOpPass());
  pass_manager->AddPass(cinn::dialect::ir::CreateCinnGroupClusterPass());

  pass_manager->AddPass(cinn::dialect::ir::CreateSingleOpFallbackToPhiPass());
  pass_manager->AddPass(cinn::dialect::ir::CreateShapeOpsFallbackToPhiPass());

  pass_manager->Run(program);
}

void ApplyCinnLowerPass(
    ::pir::Program* program,
    const std::function<std::shared_ptr<pir::PassManager>()>&
        CreatePassManager) {
  std::shared_ptr<pir::PassManager> pass_manager = CreatePassManager();

  bool has_dynamic_shape = HasDynamicShape(*program);

  bool force_static_shape = false;
  if (auto pass = cinn::dialect::ir::CreateConvertDynamicToStaticDimPass()) {
    pass_manager->AddPass(std::move(pass.value()));
    force_static_shape = true;
  }
  if (auto pass = cinn::dialect::ir::CreateConvertStaticDimToDynamicPass()) {
    pass_manager->AddPass(std::move(pass.value()));
  }

  if (FLAGS_enable_cinn_accuracy_check) {
    VLOG(0) << "Enable CINN Accuracy Check Pass";
    pass_manager->AddPass(cinn::dialect::ir::CreateAccuracyCheckPass());
  }
  if (FLAGS_enable_fusion_fallback) {
    VLOG(0) << "Enable Fusion Fallback Pass";
    pass_manager->AddPass(cinn::dialect::ir::CreateFusionFallbackPass());
  }
  if (has_dynamic_shape && !force_static_shape) {
    pass_manager->AddPass(
        cinn::dialect::ir::CreateLowerCinnDyShapeFusionOpPass());
  } else {
    pass_manager->AddPass(cinn::dialect::ir::CreateLowerCinnFusionOpPass());
  }
  pass_manager->AddPass(
      cinn::dialect::ir::CreateSplitGenerateShapeIntoShapeOpsPass());

  pass_manager->Run(program);
}

template <typename OP_TYPE>
int64_t GetOpCount(const ::pir::Operation* op) {
  int64_t count = 0;
  for (auto& region : *op) {
    for (auto& block : region) {
      for (auto& sub_op : block) {
        if (sub_op.isa<OP_TYPE>()) {
          count++;
          continue;
        }
        if (sub_op.num_regions() > 0) {
          count += GetOpCount<OP_TYPE>(&sub_op);
        }
      }
    }
  }
  return count;
}

void ApplyCinnPass(
    ::pir::Program* program,
    const std::function<std::shared_ptr<pir::PassManager>()>& CreatePassManager,
    bool is_train_mode,
    bool need_infer_symbolic_shape) {
  const uint32_t origin_num_ops = program->num_ops();
  PirToPyCodeConverter(program)
      .file_name("original_programs.py")
      .dump_symbolic_shape(FLAGS_logging_pir_py_code_dump_symbolic_dims)
      .SaveIfFlagEnabled();
  if (is_train_mode && need_infer_symbolic_shape) {
    // Skip infer symbol shape in inference, because we have run this pass in
    // the previous process
    ApplyShapeOptimizationPass(program, CreatePassManager);
  }
  ApplyPdToCinnPass(program, CreatePassManager);
  ApplyCinnPreprocessPass(program, CreatePassManager);
  ApplyBuildGroupOpPass(program, CreatePassManager);
  PirToPyCodeConverter(program)
      .file_name("group_op_programs.py")
      .SaveIfFlagEnabled();
  ApplyGroupOpPass(program, CreatePassManager);
  ApplyDivideGroupOpToFusionOpPass(program, CreatePassManager);
  PirToPyCodeConverter(program)
      .file_name("fusion_op_programs.py")
      .SaveIfFlagEnabled();
  LOG(INFO) << "FusionOp count before lowering : *****[ "
            << GetOpCount<cinn::dialect::FusionOp>(program->module_op())
            << " ]*****";
  if (VLOG_IS_ON(1)) {
    auto& shape_analysis = pir::ShapeAnalysisManager::Instance().Get(program);
    std::cout << "Program before lowering: \n"
              << pir::CustomPrintHelper(*program, shape_analysis.PrintHook())
              << std::endl;
  }

  auto start = std::chrono::high_resolution_clock::now();
  ApplyCinnLowerPass(program, CreatePassManager);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
  LOG(INFO) << "Time of lowering and compiling program: ***** [ "
            << duration.count() << " ] ***** seconds.";

  const uint32_t new_num_ops = program->num_ops();
  LOG(INFO) << "Number of ops in the original program is: " << origin_num_ops
            << ", after lowering it becomes: " << new_num_ops
            << ". (compression ratio: " << new_num_ops << "/" << origin_num_ops
            << " = " << static_cast<float>(new_num_ops) / origin_num_ops << ")";
}

}  // namespace cinn::dialect::ir
