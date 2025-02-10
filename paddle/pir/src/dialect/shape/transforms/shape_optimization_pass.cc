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

#include "paddle/pir/include/dialect/shape/transforms/shape_optimization_pass.h"

#include "paddle/common/errors.h"
#include "paddle/common/flags.h"
#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/core/dialect.h"
#include "paddle/pir/include/core/ir_printer.h"
#include "paddle/pir/include/dialect/shape/interface/infer_symbolic_shape/cache_grad_op_symbolic_shape.h"
#include "paddle/pir/include/dialect/shape/interface/infer_symbolic_shape/infer_symbolic_shape.h"
#include "paddle/pir/include/dialect/shape/ir/shape_attribute.h"
#include "paddle/pir/include/dialect/shape/ir/shape_dialect.h"
#include "paddle/pir/include/dialect/shape/utils/original_attributes_filter.h"
#include "paddle/pir/include/dialect/shape/utils/shape_analysis.h"
#include "paddle/pir/include/pass/pass_manager.h"
#include "paddle/pir/include/pass/pass_registry.h"

COMMON_DECLARE_bool(pir_apply_shape_optimization_pass);

constexpr int vlog_level = 3;

// TODO(zhangbopd): Some op results inferred by InferSymbolicShape is NOT
// consist with the result inferred by InferMeta and should be fixed.
namespace {
bool NeedCheckInferSymbolicWithInferMeta(const std::string& op_name,
                                         size_t result_idx) {
  static std::unordered_map<std::string, std::unordered_set<int>> blacklist{
      {"pd_op.reshape", {1}},
      {"pd_op.empty", {0}},
  };
  const auto& iter = blacklist.find(op_name);
  if (iter == blacklist.end()) return true;
  return iter->second.count(result_idx) == 0;
}
}  // namespace

namespace pir {
namespace {

using PassPipelineRunner =
    std::function<bool(pir::PassManager&, pir::ModuleOp)>;

void PrintProgram(pir::ModuleOp m, std::string msg) {
  ShapeConstraintIRAnalysis& shape_analysis =
      ShapeAnalysisManager::Instance().Get(m.program());
  if (VLOG_IS_ON(vlog_level)) {
    std::cerr << "===================== [ShapeDialect]" << msg
              << " =====================\n"
              << pir::CustomPrintHelper(*m.program(),
                                        shape_analysis.PrintHook())
              << std::endl;
  }
}

std::string PrintOperationWithNoRegion(Operation* op) {
  std::ostringstream os;
  pir::IrPrinter printer(os);

  // print OpResults
  os << "(";
  auto num_op_result = op->num_results();
  for (size_t idx = 0; idx < num_op_result; idx++) {
    os << "%op_" << op->id() << "_" << idx;
    if (idx < num_op_result - 1) os << ", ";
  }
  os << ")";

  os << " =";

  // print OpName & OpId
  os << " \"" << op->name() << "(op_" << op->id() << ")"
     << "\"";

  // print OpOperands
  os << " (";
  auto num_op_operands = op->num_operands();
  for (size_t idx = 0; idx < num_op_operands; idx++) {
    const pir::Value& input = op->operand_source(idx);
    if (input.defining_op()) {
      os << "op_" << input.defining_op()->id() << "_"
         << input.dyn_cast<pir::OpResult>().index();
    } else {
      os << "op_NULL";
    }
    if (idx < num_op_operands - 1) os << ", ";
  }
  os << ")";

  printer.PrintAttributeMap(*op);
  os << " :";

  // PrintOpSignature
  printer.PrintOperandsType(*op);
  os << " -> ";

  printer.PrintOpReturnType(*op);

  return os.str();
}

void PrintOpInfo(pir::Operation* op) {
  if (VLOG_IS_ON(vlog_level)) {
    VLOG(vlog_level) << op->name() << "(op_id: op_" << op->id()
                     << ", num_results=" << op->num_results() << ")"
                     << " has InferSymbolicShapeInterface.\n\t"
                     << PrintOperationWithNoRegion(op);
    if (op->name() == "cinn_op.group") {
      std::cerr << "<<<<<<<<<<<<<<<<<<<< " << op->name() << "(op_id: op_"
                << op->id() << ") START..." << std::endl;
    }
  }
}

void DebugPrintOpInfo(pir::Operation* op,
                      pir::InferSymbolicShapeContext* infer_context = nullptr) {
  std::ostringstream print_stream;
  for (uint32_t i = 0; i < op->num_results(); ++i) {
    const auto& res = op->result(i);
    if (!res || !res.type()) {
      continue;
    }

    print_stream << "\tresult(" << res.dyn_cast<pir::OpResult>().index() << ") "
                 << "ShapeOrData: {";

    if (infer_context != nullptr) {
      print_stream << infer_context->GetShapeOrDataForValue(res);
    }
    print_stream << " }\n";
  }
  if (VLOG_IS_ON(vlog_level)) {
    std::cerr << print_stream.str();
  }
}

void CheckInferSymWithInferMeta(
    pir::Operation* op,
    pir::InferSymbolicShapeContext* infer_context = nullptr) {
  for (uint32_t i = 0; i < op->num_results(); ++i) {
    const auto& res = op->result(i);
    if (!res || !res.type() || res.use_empty()) {
      continue;
    }

    std::ostringstream print_stream;

    // InferMeta funcs of some Ops are not correct now, we don't check them.
    if (!NeedCheckInferSymbolicWithInferMeta(op->name(), i)) continue;

    if (res.type().isa<pir::DenseTensorType>()) {
      const std::vector<int64_t>& infer_meta_shape =
          common::vectorize(res.type().dyn_cast<pir::DenseTensorType>().dims());
      const std::vector<symbol::DimExpr>& infer_sym_shape =
          infer_context->GetShapeOrDataForValue(res).shape();

      if (res.type().dyn_cast<pir::DenseTensorType>().dims().size() == -1) {
        LOG(WARNING) << "Warning: For" << op->name() << " [id:" << op->id()
                     << "] 's result(" << i << ")."
                     << " Rank of infer_meta_shape is dynamic. "
                     << "Received infer_sym_shape is " << infer_sym_shape;
        continue;
      }
      // Check rank.
      if (infer_meta_shape.size() != infer_sym_shape.size()) {
        std::ostringstream print_stream;
        print_stream << "Warning : Check InferSymbolicShape for " << op->name()
                     << " [id:" << op->id() << "] "
                     << " carefully! rank of infer_meta_shape is ["
                     << infer_meta_shape.size()
                     << "], but rank of infer_sym_shape is ["
                     << infer_sym_shape.size() << "].";
        LOG(ERROR) << print_stream.str();
        continue;
      }

      // Check each dim.
      for (size_t i = 0; i < infer_meta_shape.size(); ++i) {
        // Check Static shape should NOT be a symbol.
        if (infer_meta_shape[i] != -1) {
          if (!infer_sym_shape[i].isa<int64_t>()) {
            std::ostringstream print_stream;
            print_stream
                << "Warning : Check InferSymbolicShape for " << op->name()
                << " [id:" << op->id() << "] "
                << " carefully! "
                << "shape[" << i
                << "] of infer_sym_shape should be int64_t NOT a symbol!";
            LOG(ERROR) << print_stream.str();
            continue;
          }

          // Check Static shape should be consist.
          if (infer_meta_shape[i] != infer_sym_shape[i].dyn_cast<int64_t>()) {
            std::ostringstream print_stream;
            print_stream << "Warning : Check InferSymbolicShape for "
                         << op->name() << " [id:" << op->id() << "] "
                         << " carefully! "
                         << "infer_sym_shape is [" << infer_meta_shape[i]
                         << "], but infer_meta_shape is ["
                         << infer_sym_shape[i].dyn_cast<int64_t>() << "].";
            LOG(ERROR) << print_stream.str();
          }
        }
      }
    }
  }
}

class ShapeOptimizationPass : public pir::Pass {
 public:
  ShapeOptimizationPass() : pir::Pass("shape_optimization_pass", 0) {}

  void Run(pir::Operation* op) override {
    VLOG(vlog_level)
        << "===================== ShapeOptimizationPass Run start... "
           "=====================";
    auto module_op = op->dyn_cast<pir::ModuleOp>();
    PADDLE_ENFORCE_NOT_NULL(
        module_op,
        common::errors::InvalidArgument(
            "ShapeOptimizationPass should run on module op."));
    PrintProgram(module_op, "Origin Program");

    ::pir::InferSymExprForAllValues(module_op);
    // Runner is for Canonicalizer.
    PassPipelineRunner runner = [](pir::PassManager& pm, pir::ModuleOp m) {
      pm.EnableIRPrinting();
      return pm.Run(m.program());
    };

    PrintProgram(module_op, "ShapeOptimizationPass Program");
    VLOG(vlog_level) << "===================== ShapeOptimizationPass Run End. "
                        "=====================";
  }

  bool CanApplyOn(pir::Operation* op) const override {
    return op->isa<pir::ModuleOp>() && op->num_regions() > 0;
  }
};

}  // namespace

void InferSymExprForOp(Operation* op,
                       InferSymbolicShapeContext* infer_context,
                       const InferSymbolicShapeCacheKey& op_infer_cache_key) {
  auto infer_symbolic_shape_interface =
      op->dyn_cast<pir::InferSymbolicShapeInterface>();
  if (infer_symbolic_shape_interface) {
    PrintOpInfo(op);
    PADDLE_ENFORCE_EQ(
        infer_symbolic_shape_interface.InferSymbolicShape(infer_context),
        true,
        common::errors::Fatal("InferSymbolicShape for %s failed.", op->name()));

    if (op->num_results() > 0) {
      // TODO(lanxianghit): deal with the ops which have more than 1
      // ACTUAL results
      pir::shape::SetShapeAttrForOp(
          op, infer_context->GetShapeOrDataForValue(op->result(0)));
    }
  } else {
    bool is_grad_op = [&]() {
      std::string suffix = "_grad";
      const auto& op_name = op->name();
      if (op_name.size() < suffix.size()) return false;
      return op_name.compare(
                 op_name.size() - suffix.size(), suffix.size(), suffix) == 0;
    }();
    if (!is_grad_op)
      LOG(WARNING) << op->name()
                   << " DOES NOT have InferSymbolicShapeInterface!";
    const bool all_outs_static_dims = [&] {
      bool all_static_dims = true;
      for (uint32_t i = 0; i < op->num_results(); ++i) {
        if (IsStaticShape(op->result(i))) {
          continue;
        } else {
          all_static_dims = false;
          break;
        }
      }
      return all_static_dims;
    }();

    if (all_outs_static_dims) {
      for (uint32_t i = 0; i < op->num_results(); ++i) {
        infer_context->SetSymbolForValueByStaticShape(op->result(i));
      }
    } else {
      if (infer_context->GetOpInferSymbolicShapeCache(op_infer_cache_key)
              .has_value()) {
        std::vector<symbol::ShapeOrDataDimExprs> cached_result_shape_or_data =
            infer_context->GetOpInferSymbolicShapeCache(op_infer_cache_key)
                .value();
        PADDLE_ENFORCE_EQ(cached_result_shape_or_data.size(),
                          op->num_results(),
                          common::errors::Fatal(
                              "Cached number of result %u is not equal to the "
                              "given number of output %u",
                              cached_result_shape_or_data.size(),
                              op->num_results()));
        for (uint32_t i = 0; i < op->num_results(); ++i) {
          infer_context->SetShapeOrDataForValue(op->result(i),
                                                cached_result_shape_or_data[i]);
        }
      } else {
        // risk set
        LOG(WARNING) << "Not found symbolic shape cache for " << op->name()
                     << "[id:" << op->id()
                     << "], op_infer_cache_key is :" << op_infer_cache_key;
        for (uint32_t i = 0; i < op->num_results(); ++i) {
          if (!op->result(i) || !op->result(i).type()) {
            continue;
          }
          infer_context->SetSymbolForValueByStaticShape(op->result(i));
        }
      }
    }
  }
}

void CacheForwardOpSymbolicShape(
    Operation* op,
    InferSymbolicShapeContext* infer_context,
    const InferSymbolicShapeCacheKey& op_infer_cache_key) {
  std::vector<symbol::ShapeOrDataDimExprs> result_shape_or_data;
  const auto& CheckInferSymbolicShapeCacheConsistency =
      [&](const InferSymbolicShapeCacheValue& infer_result,
          const InferSymbolicShapeCacheValue& cache_result) {
        if (infer_result.size() != cache_result.size()) {
          LOG(WARNING) << "cached shape is not consistent with real shape";
        } else {
          for (uint32_t i = 0; i < cache_result.size(); ++i) {
            if (infer_result[i] != cache_result[i]) {
              LOG(WARNING) << "cached shape is not consistent with real shape";
              VLOG(3) << "InferSymbolicShapeCacheKey is: "
                      << op_infer_cache_key;
              VLOG(3) << "cached shape is: " << cache_result[i];
              VLOG(3) << "real shape is: " << infer_result[i];
            }
          }
        }
      };
  for (const auto& result : op->results()) {
    result_shape_or_data.emplace_back(
        infer_context->GetShapeOrDataForValue(result));
  }
  if (infer_context->GetOpInferSymbolicShapeCache(op_infer_cache_key)
          .has_value()) {
    std::vector<symbol::ShapeOrDataDimExprs> cached_result_shape_or_data =
        infer_context->GetOpInferSymbolicShapeCache(op_infer_cache_key).value();
    // TODO(Hongqing-work): delete check and only set cache for op without
    // InferSymbolicShapeInterface after fixing all warnings.
    CheckInferSymbolicShapeCacheConsistency(result_shape_or_data,
                                            cached_result_shape_or_data);
  } else {
    infer_context->SetOpInferSymbolicShapeCache(op_infer_cache_key,
                                                result_shape_or_data);
  }
}

void CacheBackwardOpSymbolicShape(Operation* op,
                                  InferSymbolicShapeContext* infer_context) {
  auto cache_grad_op_symbolic_shape_interface =
      op->dyn_cast<pir::CacheGradOpSymbolicShapeInterface>();
  if (cache_grad_op_symbolic_shape_interface) {
    VLOG(3) << "CacheBackwardOpSymbolicShape for: " << op->name();
    cache_grad_op_symbolic_shape_interface.CacheGradOpSymbolicShape(
        infer_context);
  }
}

void InferSymExprForBlock(const Block& block,
                          InferSymbolicShapeContext* infer_context) {
  for (auto& op : block) {
    std::vector<symbol::ShapeOrDataDimExprs> input_shape_or_data;
    for (auto& input : op.operands_source()) {
      input_shape_or_data.emplace_back(
          infer_context->GetShapeOrDataForValue(input));
    }
    InferSymbolicShapeCacheKey op_infer_cache_key(
        op.name(),
        input_shape_or_data,
        GetOrderedOriginalAttributes(op.name(), op.attributes()));
    InferSymExprForOp(&op, infer_context, op_infer_cache_key);
    CacheForwardOpSymbolicShape(&op, infer_context, op_infer_cache_key);
    CacheBackwardOpSymbolicShape(&op, infer_context);
    DebugPrintOpInfo(&op, infer_context);
    CheckInferSymWithInferMeta(&op, infer_context);
  }
}

void InferSymExprForAllValues(ModuleOp module_op) {
  ShapeConstraintIRAnalysis& shape_analysis =
      ShapeAnalysisManager::Instance().Get(module_op.program());
  auto* infer_context = shape_analysis.MutInferSymbolicShapeContext();

  // hold the kwargs symbol shape info to avoid be cleared when call init.
  const std::unordered_map<pir::Value, symbol::ShapeOrDataDimExprs>
      symbol_shape_map = [&] {
        std::unordered_map<pir::Value, symbol::ShapeOrDataDimExprs>
            symbol_shape_map;
        for (const auto& [_, value] : module_op.block().kwargs()) {
          if (!infer_context->HasShapeOrDataForValue(value)) {
            infer_context->SetSymbolForValueByStaticShape(value);
          }
          symbol_shape_map.emplace(
              value, infer_context->GetShapeOrDataForValue(value));
        }
        return symbol_shape_map;
      }();

  shape_analysis.InitInferContext();
  // init the kwarg symbol shape info
  for (const auto& kv : symbol_shape_map) {
    infer_context->SetShapeOrDataForValue(kv.first, kv.second);
  }

  InferSymExprForBlock(module_op.block(), infer_context);
}

std::unique_ptr<Pass> CreateShapeOptimizationPass() {
  return std::make_unique<ShapeOptimizationPass>();
}

}  // namespace pir

namespace pir::shape {

bool HasDynamicShape(const pir::Program& program) {
  for (const auto& op : *program.block()) {
    if (op.isa<pir::CombineOp>()) {
      continue;
    }
    for (uint32_t i = 0; i < op.num_results(); ++i) {
      if (op.result(i) && op.result(i).type()) {
        auto shape_type =
            op.result(i).type().dyn_cast<pir::ShapedTypeInterface>();
        if (shape_type && shape_type.IsDynamicShape()) {
          VLOG(vlog_level) << "###### HasDynamicShape == true";
          return true;
        }
      }
    }
  }
  VLOG(vlog_level) << "###### HasDynamicShape == false";
  return false;
}

void AddShapeOptimizationPass(
    std::shared_ptr<pir::PassManager>& pass_manager,  // NOLINT
    pir::Program& program) {                          // NOLINT
  pir::IrContext* ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<pir::shape::ShapeDialect>();
  if (FLAGS_pir_apply_shape_optimization_pass) {
    pass_manager->AddPass(pir::CreateShapeOptimizationPass());
  }
}

}  // namespace pir::shape

// REGISTER_IR_PASS(shape_optimization_pass, pir::ShapeOptimizationPass);
