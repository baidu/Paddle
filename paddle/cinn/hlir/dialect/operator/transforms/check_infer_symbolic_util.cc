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

#include "paddle/cinn/hlir/dialect/operator/transforms/check_infer_symbolic_util.h"

#include <functional>
#include <memory>
#include <optional>
#include <random>
#include "paddle/cinn/hlir/dialect/operator/transforms/check_infer_symbolic_pass.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/local_infer_symbolic_util.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/split_generate_shape_into_shape_ops_pass.h"
#include "paddle/common/flags.h"
#include "paddle/fluid/pir/dialect/operator/interface/infermeta.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/core/ir_context.h"
#include "paddle/pir/include/dialect/shape/transforms/shape_optimization_pass.h"
#include "paddle/pir/include/dialect/shape/utils/shape_analysis.h"

COMMON_DECLARE_bool(check_infer_symbolic);
PD_DECLARE_bool(prim_all);

namespace cinn {
namespace dialect {
namespace ir {

namespace {

std::ostream& operator<<(std::ostream& stream,
                         const std::vector<std::vector<int64_t>>& shapes) {
  stream << "[ ";
  for (auto shape : shapes) {
    stream << "[";
    for (auto dim : shape) {
      stream << dim << " ";
    }
    stream << "] ";
  }
  stream << "]";
  return stream;
}

DimExprs4ValueT MakeDimExprs4Value(
    pir::Program* program, const PassManagerCreator& CreatePassManager) {
  std::shared_ptr<pir::PassManager> pass_manager = CreatePassManager();
  pass_manager->AddPass(pir::CreateShapeOptimizationPass());
  pass_manager->Run(program);
  auto* shape_analysis = &pir::ShapeAnalysisManager::Instance().Get(program);
  return
      [shape_analysis](pir::Value value) -> const symbol::ShapeOrDataDimExprs& {
        return shape_analysis->GetShapeOrDataForValue(value);
      };
}

template <typename DoEachT>
void VisitEachOp(const pir::ModuleOp& op, const DoEachT& DoEach) {
  for (uint32_t i = 0; i < op->num_regions(); i++) {
    for (pir::Block& block : op->region(i)) {
      for (pir::Operation& sub_op : block) {
        DoEach(&sub_op);
      }
    }
  }
}

template <typename DoEachT>
void WalkLeafOp(pir::Program* program, const DoEachT& DoEach) {
  auto module_op = program->module_op();
  VisitEachOp(module_op, [&](pir::Operation* op) {
    if (op->num_regions() != 0) return;
    DoEach(op);
  });
}

struct ShapeSignatureGenerator {
  pir::Operation* op;
  DimExprs4ValueT GraphDimExprs4Value;

  ShapeSignatureGenerator(pir::Operation* op_,
                          DimExprs4ValueT GraphDimExprs4Value_)
      : op(op_), GraphDimExprs4Value(GraphDimExprs4Value_) {}

  using SymbolBindings = std::map<std::string, int64_t>;
  using ShapeAnalysisPtr = std::shared_ptr<pir::ShapeConstraintIRAnalysis>;
  using Shape = std::vector<int64_t>;
  using Data = std::vector<int64_t>;
  using ShapeList = std::vector<Shape>;
  using DataList = std::vector<Data>;
  using DoEachShapeSignatureT =
      std::function<void(const ShapeList& input_shapes,
                         const DataList& input_datas,
                         const ShapeList& output_shapes)>;

  void Generate(const DoEachShapeSignatureT& DoEachShapeSignature) {
    auto op_shape_analysis = MakeOpShapeAnalysis(op, GraphDimExprs4Value);
    const std::unordered_set<symbol::DimExpr>& symbols =
        GetAllSymbols(*op, op_shape_analysis);
    VisitInputSymbolBinding(
        op_shape_analysis, [&](const SymbolBindings& bindings) {
          VLOG(4) << "SymbolBindings";
          for (auto pair : bindings) {
            VLOG(4) << pair.first << " : " << pair.second;
          }
          const auto& substitute_pattern =
              GetSubstitutePattern(bindings, symbols, op_shape_analysis);
          VLOG(4) << "substitute_pattern: size=" << substitute_pattern.size();
          for (auto pair : substitute_pattern) {
            VLOG(4) << pair.first << " : " << pair.second;
          }
          const auto& [input_shapes, input_datas] =
              GetInputs(*op, op_shape_analysis, substitute_pattern);
          const auto& output_shapes =
              GetOutputShapes(*op, op_shape_analysis, substitute_pattern);
          if (input_shapes.size() == 0 || output_shapes.size() == 0) return;
          DoEachShapeSignature(input_shapes, input_datas, output_shapes);
        });
  }

  void GetSymbolsForOneDimExpr(const symbol::DimExpr& dim_expr,
                               std::unordered_set<symbol::DimExpr>* symbols) {
    auto DimExprVisitor = common::Overloaded{
        [&](std::int64_t) {},
        [&](const std::string& str) { symbols->insert(symbol::DimExpr(str)); },
        [&](const symbol::Negative<symbol::DimExpr>& negative) {
          GetSymbolsForOneDimExpr(negative->data, symbols);
        },
        [&](const symbol::Reciprocal<symbol::DimExpr>& reciprocal) {
          GetSymbolsForOneDimExpr(reciprocal->data, symbols);
        },
        [&](const symbol::Add<symbol::DimExpr>& add) {
          for (const auto& dim_expr : *add.operands) {
            GetSymbolsForOneDimExpr(dim_expr, symbols);
          }
        },
        [&](const symbol::Mul<symbol::DimExpr>& mul) {
          for (const auto& dim_expr : *mul.operands) {
            GetSymbolsForOneDimExpr(dim_expr, symbols);
          }
        },
        [&](const symbol::Max<symbol::DimExpr>& max) {
          for (const auto& dim_expr : *max.operands) {
            GetSymbolsForOneDimExpr(dim_expr, symbols);
          }
        },
        [&](const symbol::Min<symbol::DimExpr>& min) {
          for (const auto& dim_expr : *min.operands) {
            GetSymbolsForOneDimExpr(dim_expr, symbols);
          }
        },
        [&](const symbol::Broadcast<symbol::DimExpr>& broadcast) {
          for (const auto& dim_expr : *broadcast.operands) {
            GetSymbolsForOneDimExpr(dim_expr, symbols);
          }
        }};
    std::visit(DimExprVisitor, dim_expr.variant());
  }

  std::unordered_set<symbol::DimExpr> GetAllSymbols(
      const pir::Operation& op,
      const std::shared_ptr<pir::ShapeConstraintIRAnalysis>&
          op_shape_analysis) {
    std::unordered_set<symbol::DimExpr> symbols;
    auto GetSymbolsForOneValue = [&](pir::Value value) {
      const auto& shape_or_data =
          op_shape_analysis->GetShapeOrDataForValue(value);
      shape_or_data.Match(
          [&](const symbol::TensorShapeOrDataDimExprs& impl) {
            for (const auto& dim_expr : impl.shape()) {
              GetSymbolsForOneDimExpr(dim_expr, &symbols);
            }
            if (impl.data().has_value()) {
              for (const auto& dim_expr : *impl.data()) {
                GetSymbolsForOneDimExpr(dim_expr, &symbols);
              }
            }
          },
          [&](const symbol::TensorListShapeOrDataDimExprs& impl) { return; },
          [&](const symbol::RankedTensorArrayShapeOrDataDimExprs& impl) {
            return;
          },
          [&](const symbol::NullShapeOrDataDimExpr& impl) { return; });
    };

    for (std::size_t i = 0; i < op.num_operands(); ++i) {
      GetSymbolsForOneValue(op.operand_source(i));
    }
    for (std::size_t i = 0; i < op.num_results(); ++i) {
      GetSymbolsForOneValue(op.result(i));
    }
    return symbols;
  }

  std::unordered_map<symbol::DimExpr, symbol::DimExpr> GetSubstitutePattern(
      const SymbolBindings& bindings,
      const std::unordered_set<symbol::DimExpr>& symbols,
      std::shared_ptr<pir::ShapeConstraintIRAnalysis> op_shape_analysis) {
    std::random_device rd;
    std::default_random_engine eng(rd());
    std::uniform_int_distribution<int> distr(0, 1000);
    std::unordered_map<symbol::DimExpr, symbol::DimExpr> substitute_pattern(
        bindings.begin(), bindings.end());
    for (const auto& dim_expr : symbols) {
      if (substitute_pattern.count(dim_expr) <= 0) {
        substitute_pattern[dim_expr] = symbol::DimExpr(distr(eng));
      }
    }
    return substitute_pattern;
  }

  std::pair<std::optional<Shape>, std::optional<Data>> ConvertSymbolToConst(
      symbol::ShapeOrDataDimExprs shape_or_data,
      const std::unordered_map<symbol::DimExpr, symbol::DimExpr>&
          substitute_pattern) {
    using ResType = std::pair<std::optional<Shape>, std::optional<Data>>;
    auto res = shape_or_data.Match(
        [&](const symbol::TensorShapeOrDataDimExprs& impl) -> ResType {
          Shape shape;
          Data data;
          const auto& const_shape_or_data =
              symbol::SubstituteShapeOrData(impl, substitute_pattern);
          for (const auto& symbolic_shape : const_shape_or_data.shape()) {
            const auto& const_symbolic_shape =
                symbol::SimplifyDimExpr(symbolic_shape);
            if (!const_symbolic_shape.isa<std::int64_t>())
              return std::make_pair(std::nullopt, std::nullopt);
            shape.push_back(symbolic_shape.Get<std::int64_t>());
          }
          if (!const_shape_or_data.data().has_value())
            return std::make_pair(shape, data);
          for (const auto& symbolic_data : *const_shape_or_data.data()) {
            const auto& const_symbolic_data =
                symbol::SimplifyDimExpr(symbolic_data);
            if (!const_symbolic_data.isa<std::int64_t>())
              return std::make_pair(std::nullopt, std::nullopt);
            data.push_back(const_symbolic_data.Get<std::int64_t>());
          }
          return std::make_pair(shape, data);
        },
        [&](const symbol::TensorListShapeOrDataDimExprs& impl) -> ResType {
          return std::make_pair(std::nullopt, std::nullopt);
        },
        [&](const symbol::RankedTensorArrayShapeOrDataDimExprs& impl)
            -> ResType { return std::make_pair(std::nullopt, std::nullopt); },
        [&](const symbol::NullShapeOrDataDimExpr& impl) -> ResType {
          return std::make_pair(std::nullopt, std::nullopt);
        });
    return res;
  }

  std::pair<ShapeList, DataList> GetInputs(
      const pir::Operation& op,
      const ShapeAnalysisPtr& op_shape_analysis,
      const std::unordered_map<symbol::DimExpr, symbol::DimExpr>&
          substitute_pattern) {
    ShapeList shape_list;
    DataList data_list;
    VLOG(4) << "GetInputShapes: ";
    for (std::size_t i = 0; i < op.num_operands(); ++i) {
      VLOG(4) << "  ShapeOrData:"
              << op_shape_analysis->GetShapeOrDataForValue(
                     op.operand_source(i));
      const symbol::ShapeOrDataDimExprs& shape_or_data =
          op_shape_analysis->GetShapeOrDataForValue(op.operand_source(i));
      const auto& [shape, data] =
          ConvertSymbolToConst(shape_or_data, substitute_pattern);
      if (!shape.has_value() || shape->empty())
        return std::make_pair(ShapeList(), DataList());
      shape_list.emplace_back(*shape);
      data_list.emplace_back(*data);
    }
    VLOG(4) << "  shape_list: " << shape_list;
    VLOG(4) << "  data_list: " << data_list;
    return std::make_pair(shape_list, data_list);
  }

  ShapeList GetOutputShapes(
      const pir::Operation& op,
      const ShapeAnalysisPtr& op_shape_analysis,
      const std::unordered_map<symbol::DimExpr, symbol::DimExpr>&
          substitute_pattern) {
    ShapeList shape_list;
    VLOG(4) << "GetOutputShapes: ";
    for (std::size_t i = 0; i < op.num_results(); ++i) {
      VLOG(4) << "  ShapeOrData:"
              << op_shape_analysis->GetShapeOrDataForValue(op.result(i));
      const symbol::ShapeOrDataDimExprs& shape_or_data =
          op_shape_analysis->GetShapeOrDataForValue(op.result(i));
      const auto& [shape, data] =
          ConvertSymbolToConst(shape_or_data, substitute_pattern);
      if (!shape.has_value() || shape->empty()) return ShapeList();
      shape_list.emplace_back(*shape);
    }
    VLOG(4) << "  shape_list: " << shape_list;
    return shape_list;
  }

  struct CstrEqSymbolNames {
    std::vector<std::string> symbol_names;
  };

  struct CstrBroadcastableSymbolNames {
    std::vector<std::string> symbol_names;
  };

  using ConstrainedSymbolNames =
      std::variant<CstrEqSymbolNames, CstrBroadcastableSymbolNames>;

  using ConstrainedSymbolNamesList = std::vector<ConstrainedSymbolNames>;

  using ConstrainedSymbolNamesAndDimBindingList =
      std::vector<std::pair<ConstrainedSymbolNames, int64_t>>;

  using DoEachSymbolBindingT = std::function<void(const SymbolBindings&)>;
  void VisitInputSymbolBinding(
      const ShapeAnalysisPtr& op_shape_analysis,
      const DoEachSymbolBindingT& DoEachSymbolBinding) {
    std::optional<ConstrainedSymbolNamesList> constrained_sym_list =
        GetConstrainedSymbolNamesList(op_shape_analysis);
    if (!constrained_sym_list.has_value()) return;
    VisitSymbolsBindings(*constrained_sym_list, [&](const auto& syms_and_dims) {
      VisitSymbolBindings(syms_and_dims, {}, DoEachSymbolBinding);
    });
  }

  void VisitSymbolBindings(
      const ConstrainedSymbolNamesAndDimBindingList& syms_and_dims,
      const SymbolBindings& collected,
      const DoEachSymbolBindingT& DoEachSymbolBinding) {
    if (syms_and_dims.empty()) return DoEachSymbolBinding(collected);
    ConstrainedSymbolNamesAndDimBindingList remainder{
        std::next(syms_and_dims.begin()), syms_and_dims.end()};
    const auto& first = syms_and_dims.at(0);
    VisitConstrainedSymbolBindings(first, [&](auto& cur_bindings) {
      if (HasConflict(collected, cur_bindings)) return;
      cur_bindings.insert(collected.begin(), collected.end());
      VisitSymbolBindings(remainder, cur_bindings, DoEachSymbolBinding);
    });
  }

  bool HasConflict(const SymbolBindings& lhs, const SymbolBindings& rhs) {
    for (const auto& [sym_name, dim] : lhs) {
      const auto& iter = rhs.find(sym_name);
      if (iter == rhs.end()) continue;
      if (iter->second != dim) return true;
    }
    return false;
  }

  template <typename DoEachT>
  void VisitConstrainedSymbolBindings(
      const std::pair<ConstrainedSymbolNames, int64_t>& syms_and_dim,
      const DoEachT& DoEach) {
    const auto& [syms, dim] = syms_and_dim;
    return std::visit(
        [&](const auto& impl) {
          return VisitConstrainedSymbolBindingsImpl(impl, dim, DoEach);
        },
        syms);
  }

  template <typename DoEachT>
  void VisitConstrainedSymbolBindingsImpl(const CstrEqSymbolNames& syms,
                                          int64_t dim,
                                          const DoEachT& DoEach) {
    SymbolBindings bindings;
    for (const auto& sym_name : syms.symbol_names) {
      bindings[sym_name] = dim;
    }
    DoEach(bindings);
  }

  template <typename DoEachT>
  void VisitConstrainedSymbolBindingsImpl(
      const CstrBroadcastableSymbolNames& syms,
      int64_t dim,
      const DoEachT& DoEach) {
    VisitEachSubSet(syms.symbol_names.size(), {}, [&](const auto& flags) {
      SymbolBindings bindings;
      for (int i = 0; i < syms.symbol_names.size(); ++i) {
        bindings[syms.symbol_names.at(i)] = (flags.at(i) ? dim : 1);
      }
      DoEach(bindings);
    });
  }

  using IsSubset = int;

  template <typename DoEachT>
  void VisitEachSubSet(int set_size,
                       const std::vector<IsSubset>& is_subset_flags,
                       const DoEachT& DoEach) {
    if (set_size <= 0) return DoEach(is_subset_flags);

    const auto& RecursiveVisit = [&](bool is_subset) {
      std::vector<IsSubset> current_is_subset_flags(is_subset_flags);
      current_is_subset_flags.push_back(static_cast<int>(is_subset));
      VisitEachSubSet(set_size - 1, current_is_subset_flags, DoEach);
    };
    RecursiveVisit(true);
    RecursiveVisit(false);
  }

  std::optional<ConstrainedSymbolNamesList> GetConstrainedSymbolNamesList(
      const ShapeAnalysisPtr& op_shape_analysis) {
    ConstrainedSymbolNamesList cstr_list;
    bool all_string = true;
    const auto& cstr_manager = op_shape_analysis->constraints_manager();
    VLOG(4) << "constraints:" << cstr_manager;
    std::function<void(const std::vector<symbol::DimExpr>&)> GetEquals =
        [&](const auto& clusters) {
          CstrEqSymbolNames equals;
          for (const symbol::DimExpr& dim_expr : clusters) {
            if (dim_expr.isa<std::string>())
              equals.symbol_names.push_back(dim_expr.Get<std::string>());
            else if (dim_expr.isa<std::int64_t>())
              continue;
            else
              all_string = false;
          }
          if (!equals.symbol_names.empty()) cstr_list.emplace_back(equals);
        };
    cstr_manager.VisitEqualClusters(GetEquals);

    std::function<void(std::unordered_set<
                       symbol::Broadcastable<symbol::DimExpr>>::const_iterator)>
        GetBroadcastables = [&](const auto& it) {
          if (it->data->lhs.template isa<std::string>() &&
              it->data->rhs.template isa<std::string>()) {
            CstrBroadcastableSymbolNames bcables;
            bcables.symbol_names.push_back(
                it->data->lhs.template Get<std::string>());
            bcables.symbol_names.push_back(
                it->data->rhs.template Get<std::string>());
            cstr_list.emplace_back(bcables);
          }
        };
    cstr_manager.BroadcastableConstraintsVisitor(GetBroadcastables);

    if (!all_string) return std::nullopt;
    return cstr_list;
  }

  using ConstrainedSymbolNamesAndDimBindingsList =
      std::vector<std::pair<ConstrainedSymbolNames, std::vector<int64_t>>>;

  template <typename DoEachT>
  void VisitSymbolsBindings(
      const ConstrainedSymbolNamesList& constrained_sym_list,
      const DoEachT& DoEach) {
    ConstrainedSymbolNamesAndDimBindingsList names_and_dims =
        GetConstrainedSymbolNamesAndDimBindingsList(constrained_sym_list);
    VisitCombinedSymbolsBindings(names_and_dims, {}, DoEach);
  }

  ConstrainedSymbolNamesAndDimBindingsList
  GetConstrainedSymbolNamesAndDimBindingsList(
      const ConstrainedSymbolNamesList& constrained_sym_list) {
    static const std::vector<int64_t> table{1, 2, 3, 4, 5, 6, 7};

    ConstrainedSymbolNamesAndDimBindingsList list;
    for (const auto& cstr : constrained_sym_list) {
      list.push_back(std::pair(cstr, table));
    }
    return list;
  }

  template <typename DoEachT>
  void VisitCombinedSymbolsBindings(
      const ConstrainedSymbolNamesAndDimBindingsList& names_and_dims,
      const ConstrainedSymbolNamesAndDimBindingList& names_and_dim,
      const DoEachT& DoEach) {
    if (names_and_dims.empty()) return DoEach(names_and_dim);
    ConstrainedSymbolNamesAndDimBindingsList remainder{
        std::next(names_and_dims.begin()), names_and_dims.end()};
    const auto* first = &names_and_dims.at(0);
    auto cur_names_and_dim =
        ConstrainedSymbolNamesAndDimBindingList(names_and_dim);
    cur_names_and_dim.push_back(std::pair(first->first, 0));
    for (int64_t dim_binding : first->second) {
      cur_names_and_dim.back().second = dim_binding;
      VisitCombinedSymbolsBindings(remainder, cur_names_and_dim, DoEach);
    }
  }
};

void DoInferMeta(const std::vector<std::vector<int64_t>>& input_shapes,
                 const std::vector<std::vector<int64_t>>& input_datas,
                 pir::Builder* builder,
                 pir::Operation* op,
                 std::vector<pir::Operation*>* op_list,
                 std::vector<std::vector<int64_t>>* infer_meta_result) {
  std::vector<pir::Value> input_values;
  for (int i = 0; i < input_shapes.size(); i++) {
    if (input_datas[i].size() == 0) {
      paddle::dialect::EmptyOp empty_op =
          builder->Build<paddle::dialect::EmptyOp>(input_shapes[i]);
      op_list->push_back(empty_op.operation());
      input_values.push_back(empty_op.out());
    } else {
      paddle::dialect::FullIntArrayOp full_int_array_op =
          builder->Build<paddle::dialect::FullIntArrayOp>(
              input_datas[i], phi::DataType::INT64, phi::CPUPlace());
      op_list->push_back(full_int_array_op.operation());
      input_values.push_back(full_int_array_op.out());
    }
  }

  pir::AttributeMap attribute_map = op->attributes();
  paddle::dialect::InferMetaInterface interface =
      op->dyn_cast<paddle::dialect::InferMetaInterface>();
  const auto& types = interface.InferMeta(input_values, &attribute_map);
  for (const auto& type : types) {
    infer_meta_result->push_back(
        common::vectorize(type.dyn_cast<pir::DenseTensorType>().dims()));
  }
}

void EraseTempOp(const std::vector<pir::Operation*>& op_list) {
  for (pir::Operation* op : op_list) {
    if (op->isa<paddle::dialect::EmptyOp>()) {
      std::vector<pir::Operation*> full_int_array_op_list;
      for (auto& value : op->operands_source()) {
        full_int_array_op_list.push_back(value.defining_op());
      }
      op->Erase();
      for (pir::Operation* op : full_int_array_op_list) {
        op->Erase();
      }
    } else {
      op->Erase();
    }
  }
}

void CheckByInferMeta(pir::Operation* op,
                      pir::Builder* builder,
                      const std::vector<std::vector<int64_t>>& input_shapes,
                      const std::vector<std::vector<int64_t>>& input_datas,
                      const std::vector<std::vector<int64_t>>& output_shapes) {
  std::vector<pir::Operation*> op_list;
  std::vector<std::vector<int64_t>> infer_meta_result;
  VLOG(4) << "input_shapes     : " << input_shapes;
  VLOG(4) << "input_datas      : " << input_datas;
  VLOG(4) << "output_shapes    : " << output_shapes;

  try {
    DoInferMeta(
        input_shapes, input_datas, builder, op, &op_list, &infer_meta_result);
    VLOG(4) << "infer_meta_result: " << infer_meta_result;
    PADDLE_ENFORCE_EQ(
        infer_meta_result.size(),
        output_shapes.size(),
        ::common::errors::InvalidArgument(
            "infer_meta_result.size() not equal output_shapes.size() for %s",
            op->name()));
    for (int i = 0; i < infer_meta_result.size(); i++) {
      PADDLE_ENFORCE_EQ(infer_meta_result[i].size(),
                        output_shapes[i].size(),
                        ::common::errors::InvalidArgument(
                            "infer_meta_result[%d].size() not equal "
                            "output_shapes[%d].size() for %s",
                            i,
                            i,
                            op->name()));
      for (int j = 0; j < infer_meta_result[i].size(); j++) {
        if (infer_meta_result[i][j] != -1)
          PADDLE_ENFORCE_EQ(infer_meta_result[i][j],
                            output_shapes[i][j],
                            ::common::errors::InvalidArgument(
                                "infer_meta_result[%d][%d] not "
                                "equal output_shapes[%d][%d] for %s",
                                i,
                                j,
                                i,
                                j,
                                op->name()));
      }
    }
    VLOG(4) << "check constraints success for " << op->name();
  } catch (common::enforce::EnforceNotMet error) {
    LOG(ERROR) << "check constraints error for " << op->name() << ", "
               << error.error_str();
  }
  EraseTempOp(op_list);
}

void CheckOpDimExprConstraints(pir::Operation* op,
                               const DimExprs4ValueT& GraphDimExprs4Value) {
  ShapeSignatureGenerator generator(op, GraphDimExprs4Value);
  pir::IrContext* ctx = pir::IrContext::Instance();
  pir::Builder builder = pir::Builder(ctx, op->GetParent());
  generator.Generate([&](const auto& input_shapes,
                         const auto& input_datas,
                         const auto& output_shapes) {
    CheckByInferMeta(op, &builder, input_shapes, input_datas, output_shapes);
  });
}

void CheckProgramDimExprConstraints(
    pir::Program* program, const DimExprs4ValueT& GraphDimExprs4Value) {
  WalkLeafOp(program, [&](pir::Operation* op) {
    auto infer_symbolic_shape_interface =
        op->dyn_cast<pir::InferSymbolicShapeInterface>();
    if (!infer_symbolic_shape_interface) return;
    VLOG(4) << "CheckProgramDimExprConstraints for op: " << op->name();
    CheckOpDimExprConstraints(op, GraphDimExprs4Value);
  });
}

}  // namespace

void CheckInferSymbolicIfNeed(pir::Program* program,
                              const PassManagerCreator& CreatePassManager) {
  if (!FLAGS_prim_all || !FLAGS_check_infer_symbolic) return;
  const auto& GraphDimExprs4Value =
      MakeDimExprs4Value(program, CreatePassManager);
  // CheckProgramDimExprConstraints has some bug, so we comment it.
  // CheckProgramDimExprConstraints(program, GraphDimExprs4Value);
  std::shared_ptr<pir::PassManager> pass_manager = CreatePassManager();
  pass_manager->AddPass(CreateCheckInferSymbolicPass(GraphDimExprs4Value));
  pass_manager->AddPass(CreateSplitGenerateShapeIntoShapeOpsPass());
  pass_manager->Run(program);
}

}  // namespace ir
}  // namespace dialect
}  // namespace cinn
