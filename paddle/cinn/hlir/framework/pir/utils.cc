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

#include "paddle/cinn/hlir/framework/pir/utils.h"

#include <regex>
#include <string>
#include <unordered_map>
#include "glog/logging.h"
#include "paddle/cinn/common/bfs_walker.h"
#include "paddle/cinn/hlir/dialect/operator/ir/generate_shape_util.h"
#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/op_dialect.h"
#include "paddle/cinn/hlir/framework/op.h"
#include "paddle/cinn/hlir/framework/pir/op_mapper.h"
#include "paddle/common/enforce.h"
#include "paddle/common/flags.h"
#include "paddle/fluid/pir/dialect/operator/interface/infer_symbolic_shape/infer_sym_utils.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_dialect.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_op.h"
#include "paddle/pir/include/dialect/shape/ir/shape_attribute.h"
#include "paddle/pir/include/dialect/shape/utils/shape_analysis.h"

PD_DECLARE_string(allow_cinn_ops);
PD_DECLARE_string(deny_cinn_ops);
COMMON_DECLARE_bool(disable_dyshape_in_train);

namespace cinn {
namespace hlir {
namespace framework {
namespace pir {

// Mapping PaddleDialect Op into CINN AST Compute register Op.
// All key names are also supported in CINN. For ops not in this
// list, we judge them by search it in CINN global Operator table.
const std::unordered_map<std::string, std::string> CompatibleInfo::OP_NAMES = {
    {"pd_op.full", "fill_constant"},
    {"pd_op.sum", "reduce_sum"},
    {"pd_op.max", "reduce_max"},
    {"pd_op.min", "reduce_min"},
    {"pd_op.prod", "reduce_prod"},
    {"pd_op.add", "elementwise_add"},
    {"pd_op.elementwise_pow", "pow"},
    {"pd_op.multiply", "elementwise_mul"},
    {"pd_op.maximum", "max"},
    {"pd_op.minimum", "min"},
    {"pd_op.reshape", "reshape"},
    {"pd_op.squeeze", "reshape"},
    {"pd_op.unsqueeze", "reshape"},
    {"pd_op.split_with_num", "split"},
    {"pd_op.expand", "broadcast_to"},
    {"pd_op.where", "select"},
    {"cinn_op.generate_shape", "generate_shape"},
    {"cinn_op.broadcast", "broadcast_to"}};

namespace {
using GroupOpsVec = std::vector<::pir::Operation*>;
// The delim(`;`) that is used to split the FLAGS_allow_cinn_ops
// & FLAGS_deny_cinn_ops.
constexpr char kDelim[] = ";";

std::unordered_set<std::string> StringSplit(const std::string& str,
                                            const std::string& delim) {
  std::regex reg(delim);
  std::unordered_set<std::string> elems{
      std::sregex_token_iterator(str.begin(), str.end(), reg, -1),
      std::sregex_token_iterator()};
  elems.erase("");
  return elems;
}

std::string GetDebugInfo(const std::unordered_set<std::string>& names) {
  std::string debug_info = "[";
  for (auto& name : names) {
    debug_info.append(name);
    debug_info.append(", ");
  }
  debug_info.append("]");
  return debug_info;
}

// OpTransInfo contains information used to detect subgraphs
// supported by the CINN compiler.
class OpTransInfo {
  using DeParamCondT =
      std::unordered_map<std::string, std::unordered_set<std::string>>;

 public:
  OpTransInfo() {}

  const DeParamCondT& deny_param_cond() const { return deny_param_cond_; }
  bool IsDeniedByDefault(const std::string& op_name) const {
    return default_deny_ops_.count(op_name) || IsDeniedInFLAGS(op_name);
  }

  bool IsDeniedInFLAGS(const std::string& op_name) const {
    auto allow_ops = StringSplit(FLAGS_allow_cinn_ops, kDelim);
    auto deny_ops = StringSplit(FLAGS_deny_cinn_ops, kDelim);
    if (VLOG_IS_ON(4)) {
      LOG_FIRST_N(INFO, 1) << "The allowed Cinn Ops: "
                           << GetDebugInfo(allow_ops);
      LOG_FIRST_N(INFO, 1) << "The denied Cinn Ops: " << GetDebugInfo(deny_ops);
    }
    if (!allow_ops.empty()) {
      return allow_ops.count(op_name) == 0U;
    } else if (!deny_ops.empty()) {
      return deny_ops.count(op_name);
    }
    return false;
  }

 private:
  DeParamCondT deny_param_cond_{{"batch_norm", {"ReserveSpace"}},
                                {"batch_norm_grad", {"ReserveSpace"}}};

  std::unordered_set<std::string> default_deny_ops_{"feed",
                                                    "fetch",
                                                    "conv2d",
                                                    "conv2d_grad",
                                                    "depthwise_conv2d",
                                                    "depthwise_conv2d_grad",
                                                    "dropout",
                                                    "pool2d",
                                                    "pool2d_grad",
                                                    "pool3d",
                                                    "pool3d_grad"
                                                    "split",
                                                    "matmul",
                                                    "matmul_grad",
                                                    "embedding_grad",
                                                    "embedding",
                                                    "arange",
                                                    "argmax",
                                                    "argmin",
                                                    "argsort",
                                                    "assign_value",
                                                    "one_hot",
                                                    "softmax",
                                                    "randint"};
};

std::string OpNameAfterStripDialect(const ::pir::Operation& op) {
  std::string name = op.name();
  auto pos = name.find(".");
  if (pos == std::string::npos) {
    return name;
  }
  auto op_name = name.substr(pos + 1);
  VLOG(7) << "GetOpName: " << name << " -> " << op_name;
  PADDLE_ENFORCE_NE(
      op_name,
      "",
      ::common::errors::InvalidArgument("Not Allow op name is empty"));
  return op_name;
}

bool IsSupportInCinn(const ::pir::Operation& op);

// In case of op has some attributes generated by FullOp, it need
// implement OpPattern in pd_to_cinn_pass. Otherwise, we mark them
// as unimplement ops.
bool UnimplementOps(const ::pir::Operation& op) {
  // cinn not support uniform, the FullOp of max and min support
  // NOT generate by CINN
  if (op.isa<paddle::dialect::FullOp>()) {
    auto out = op.result(0);
    if (out.use_count() > 0) {
      return !IsSupportInCinn(*(out.first_use().owner()));
    }
  }
  return false;
}

bool HaveUnkDim(const ::pir::Operation& op) {
  auto HasNegDim = [](const ::pir::Type& type) {
    auto tensor_type = type.dyn_cast<::pir::DenseTensorType>();

    if (tensor_type) {
      for (size_t i = 0; i < tensor_type.dims().size(); ++i) {
        if (tensor_type.dims()[i] < 0) {
          return true;
        }
      }
    }

    return false;
  };

  // Judge for vector<Type>
  auto HasUnkDimInVT = [&](const std::vector<::pir::Type>& types) {
    for (auto& type : types) {
      if (HasNegDim(type)) return true;
    }
    return false;
  };

  for (size_t i = 0; i < op.num_operands(); ++i) {
    auto value = op.operand_source(i);
    if (!value || !value.type()) continue;
    // TODO(Hongqing-work): check if tensor array is needed
    if (auto vector_type = value.type().dyn_cast<::pir::VectorType>()) {
      if (HasUnkDimInVT(vector_type.data())) return true;
    } else if (HasNegDim(value.type())) {
      return true;
    }
  }

  for (size_t i = 0; i < op.num_results(); ++i) {
    auto value = op.result(i);
    if (!value || !value.type()) continue;
    if (auto vector_type = value.type().dyn_cast<::pir::VectorType>()) {
      if (HasUnkDimInVT(vector_type.data())) return true;
    } else if (HasNegDim(value.type())) {
      return true;
    }
  }
  return false;
}

bool HasDynamicRank(const ::pir::Operation& op) {
  for (size_t i = 0; i < op.num_operands(); i++) {
    ::pir::Value value = op.operand_source(i);
    if (value.type().isa<::pir::DenseTensorType>()) {
      if (value.type().dyn_cast<::pir::DenseTensorType>().dims().size() == -1) {
        return true;
      }
    }
  }
  for (size_t i = 0; i < op.num_results(); i++) {
    ::pir::Value value = op.result(i);
    if (value.type().isa<::pir::DenseTensorType>()) {
      if (value.type().dyn_cast<::pir::DenseTensorType>().dims().size() == -1) {
        return true;
      }
    }
  }
  return false;
}

bool AllInputDenseTensor(const ::pir::Operation& op) {
  for (size_t i = 0; i < op.num_operands(); ++i) {
    auto value = op.operand_source(i);
    if (!value || !value.type()) continue;
    if (!value.type().isa<::pir::DenseTensorType>()) {
      return false;
    }
  }

  return true;
}

bool IsSmallNumelOp(const ::pir::Operation& op) {
  const auto& GetNumElementsFromDim = [](const ::pir::DDim& dim) -> int64_t {
    if (::common::contain_unknown_dim(dim)) {
      return std::numeric_limits<int32_t>::max();
    } else {
      return ::common::product(dim);
    }
  };

  const auto& GetNumElementsFromValue =
      [&](const ::pir::Value& value) -> int64_t {
    int64_t numel = -1;
    if (value && value.type()) {
      auto type = value.type().dyn_cast<::pir::DenseTensorType>();
      if (type) {
        numel = GetNumElementsFromDim(type.dims());
      }
    }
    return numel;
  };
  const int64_t max_value_numel = [&] {
    int64_t max_value_numel = -1;
    if (op.num_operands() == 0) {  // no input
      return max_value_numel;
    }

    for (uint32_t i = 0; i < op.num_operands(); ++i) {
      max_value_numel = std::max(GetNumElementsFromValue(op.operand_source(i)),
                                 max_value_numel);
    }
    for (uint32_t i = 0; i < op.num_results(); ++i) {
      max_value_numel =
          std::max(GetNumElementsFromValue(op.result(i)), max_value_numel);
    }
    return max_value_numel;
  }();

  // max value check
  return (0 <= max_value_numel && max_value_numel < 32);
}

// Mainly used for pd_to_cinn_pass and reused in IsSupportInCinn function.
bool IsDeniedInCinn(const ::pir::Operation& op) {
  if (FLAGS_disable_dyshape_in_train && HaveUnkDim(op)) {
    return true;
  }
  if (!AllInputDenseTensor(op) || UnimplementOps(op)) {
    VLOG(5) << "Found " << op.name()
            << " UnimplementOps or NotAllInputDenseTensor. "
            << "So mark IsDeniedForCinn: " << true;
    return true;
  }
  if (HasDynamicRank(op)) {
    VLOG(5) << "Found " << op.name()
            << " has dynamic rank in operand or result value. "
            << "So mark IsDeniedForCinn: " << true;
    return true;
  }

  // Strip the dialect, like pd_op.abs -> abs
  const auto op_name = OpNameAfterStripDialect(op);
  const bool is_denied = OpTransInfo().IsDeniedByDefault(op_name);
  VLOG(5) << op_name << " is denied in FLAGS or defaultly: " << is_denied;
  return is_denied;
}

bool IsRegisteredInCINN(const ::pir::Operation& op) {
  return OpRegistry::Global()->Find(CompatibleInfo::OpName(op)) != nullptr;
}

namespace {
std::unordered_set<std::string> CollectSymbols(
    const symbol::ShapeOrDataDimExprs& shape_or_data,
    std::function<std::vector<symbol::DimExpr>(
        const symbol::TensorShapeOrDataDimExprs& tensor_shape_or_data)>
        get_dim_exprs_vec_func) {
  std::unordered_set<std::string> res;
  const auto& CollectVectorDimExprSymbols =
      [&](const std::vector<symbol::DimExpr>& dim_exprs) {
        for (const auto& dim_expr : dim_exprs) {
          const auto& single_dim_expr_symbols =
              symbol::CollectDimExprSymbols(dim_expr);
          res.insert(single_dim_expr_symbols.begin(),
                     single_dim_expr_symbols.end());
        }
      };

  const auto& CollectTensorDimExprSymbols =
      [&](const symbol::TensorShapeOrDataDimExprs& tensor_shape_or_data) {
        CollectVectorDimExprSymbols(
            get_dim_exprs_vec_func(tensor_shape_or_data));
      };

  shape_or_data.Match(
      [&](const symbol::TensorShapeOrDataDimExprs& impl) {
        CollectTensorDimExprSymbols(impl);
      },
      [&](const symbol::TensorListShapeOrDataDimExprs& impl) {
        for (const auto& tensor_shape_or_data : impl) {
          CollectTensorDimExprSymbols(tensor_shape_or_data);
        }
      },
      [&](const symbol::RankedTensorArrayShapeOrDataDimExprs& impl) {
        // Tensor array no need to collect symbols.
        return;
      },
      [&](const symbol::NullShapeOrDataDimExpr& impl) { return; });

  return res;
}

std::unordered_set<std::string> CollectSymbolsFromShape(
    const symbol::ShapeOrDataDimExprs& shape_or_data) {
  return CollectSymbols(
      shape_or_data,
      [](const symbol::TensorShapeOrDataDimExprs& tensor_shape_or_data)
          -> std::vector<symbol::DimExpr> {
        return tensor_shape_or_data.shape();
      });
}

std::unordered_set<std::string> CollectSymbolsFromData(
    const symbol::ShapeOrDataDimExprs& shape_or_data) {
  return CollectSymbols(
      shape_or_data,
      [](const symbol::TensorShapeOrDataDimExprs& tensor_shape_or_data) {
        std::vector<symbol::DimExpr> res;
        if (tensor_shape_or_data.data()) {
          res = tensor_shape_or_data.data().value();
        }
        return res;
      });
}

class SymbolGetter {
 public:
  using GetSymbolFuncT =
      std::function<symbol::ShapeOrDataDimExprs(const ::pir::Value&)>;
  explicit SymbolGetter(const GetSymbolFuncT& get_shape_or_data_func)
      : get_shape_or_data_func_(get_shape_or_data_func) {}
  symbol::ShapeOrDataDimExprs operator()(const ::pir::Value& value) const {
    return get_shape_or_data_func_(value);
  }

 private:
  GetSymbolFuncT get_shape_or_data_func_;
};

template <typename T>
bool HaveIntersection(const std::unordered_set<T>& lhs,
                      const std::unordered_set<T>& rhs) {
  return std::any_of(lhs.begin(), lhs.end(), [&rhs](T elem) {
    return rhs.find(elem) != rhs.end();
  });
}

template <typename T>
std::unordered_set<T> GetDifference(const std::unordered_set<T>& lhs,
                                    const std::unordered_set<T>& rhs) {
  std::unordered_set<T> result;
  for (const auto& elem : lhs) {
    if (rhs.find(elem) == rhs.end()) {
      result.insert(elem);
    }
  }
  return result;
}

bool HasNewDataSymbolUsedByDownstream(
    const ::pir::Value& output_value,
    const std::unordered_set<std::string>& new_data_symbol,
    const SymbolGetter& symbol_getter) {
  bool res = false;
  const auto& VisitNextNewDataSymbolValue =
      [&](::pir::Value value, const std::function<void(::pir::Value)>& Visit) {
        if (res) return;

        bool has_item_in_new_data_symbol_set = [&]() {
          return HaveIntersection(CollectSymbolsFromData(symbol_getter(value)),
                                  new_data_symbol);
        }();

        if (has_item_in_new_data_symbol_set) {
          for (auto iter = value.use_begin(); iter != value.use_end(); ++iter) {
            const auto& downstream_op = iter->owner();
            for (const auto& downstream_value : downstream_op->results()) {
              Visit(downstream_value);
            }
          }
        }
      };

  ::common::BfsWalker<::pir::Value> value_bfs_walker(
      VisitNextNewDataSymbolValue);
  value_bfs_walker(output_value, [&](::pir::Value value) {
    if (HaveIntersection(CollectSymbolsFromShape(symbol_getter(value)),
                         new_data_symbol)) {
      res = true;
      return;
    }
    for (auto iter = value.use_begin(); iter != value.use_end(); ++iter) {
      const auto& downstream_op = iter->owner();
      if (downstream_op->isa<paddle::dialect::SliceOp>()) {
        if (downstream_op->operand_source(1) == value ||
            downstream_op->operand_source(2) == value) {
          res = true;
          return;
        }
      }
    }
  });
  return res;
}
}  // namespace

bool CauseNewSymbolicShape(const ::pir::Operation& op) {
  auto& shape_analysis = ::pir::ShapeAnalysisManager::Instance().Get(
      const_cast<::pir::Operation&>(op).GetParentProgram());
  SymbolGetter symbol_getter([&](const ::pir::Value& value) {
    return shape_analysis.GetShapeOrDataForValue(value);
  });

  const auto& IsProcessableSlice = [&]() -> bool {
    using paddle::dialect::details::HasCompleteData;
    const auto& starts_shape_data = symbol_getter(op.operand_source(1));
    const auto& ends_shape_data = symbol_getter(op.operand_source(2));
    return HasCompleteData(starts_shape_data) &&
           HasCompleteData(ends_shape_data);
  };

  if (op.isa<paddle::dialect::SliceOp>() && !IsProcessableSlice()) {
    return true;
  }

  std::unordered_set<std::string> input_symbols = [&]() {
    std::unordered_set<std::string> res;
    for (const auto& input_value : op.operands_source()) {
      const auto& shape_symbol =
          CollectSymbolsFromShape(symbol_getter(input_value));
      const auto& data_symbol =
          CollectSymbolsFromData(symbol_getter(input_value));
      res.insert(shape_symbol.begin(), shape_symbol.end());
      res.insert(data_symbol.begin(), data_symbol.end());
    }
    return res;
  }();

  bool outputs_shape_have_new_symbol = [&]() {
    for (const auto& output_value : op.results()) {
      if (!GetDifference(CollectSymbolsFromShape(symbol_getter(output_value)),
                         input_symbols)
               .empty())
        return true;
    }
    return false;
  }();

  bool outputs_data_have_new_used_symbol = [&]() {
    for (const auto& output_value : op.results()) {
      const auto& new_data_symbol = [&]() -> std::unordered_set<std::string> {
        return GetDifference(
            CollectSymbolsFromData(symbol_getter(output_value)), input_symbols);
      }();
      if (new_data_symbol.empty()) {
        return false;
      }
      if (HasNewDataSymbolUsedByDownstream(
              output_value, new_data_symbol, symbol_getter)) {
        return true;
      }
    }
    return false;
  }();

  return outputs_shape_have_new_symbol || outputs_data_have_new_used_symbol;
}

#define PD_OP_NAME(op) paddle::dialect::op::name()
// For op supports AttributeTensor but has handled in
// pd_to_cinn_pass. Such as cinn_op.reshape, except pd_op.reshape;
const std::unordered_set<std::string> TOCINN_OPS = {
    PD_OP_NAME(SumOp),
    PD_OP_NAME(MaxOp),
    PD_OP_NAME(MinOp),
    PD_OP_NAME(ProdOp),
    PD_OP_NAME(PowOp),
    PD_OP_NAME(ScaleOp),
    PD_OP_NAME(Pool2dOp),
    PD_OP_NAME(IscloseOp),
    // PD_OP_NAME(SliceOp),
    PD_OP_NAME(ConcatOp),
    PD_OP_NAME(SplitOp),
    PD_OP_NAME(SplitWithNumOp),
    PD_OP_NAME(AddNOp),
    PD_OP_NAME(UniformOp),
    PD_OP_NAME(GatherOp),
};
#undef PD_OP_NAME

bool HasHandledInPass(const ::pir::Operation& op) {
  return TOCINN_OPS.count(op.name()) == 0U;
}

// In following cases, the op is marked SupportCinn:
// 1. it is NOT denied in IsDeniedInCinn(op)
// 2. it should be registered in OpRegistry;
// 3. it should be handled in pd_to_cinn_pass;
bool IsSupportInCinn(const ::pir::Operation& op) {
  const bool is_denied = IsDeniedInCinn(op);
  if (IsDeniedInCinn(op)) {
    VLOG(5) << op.name() << "[id:" << op.id() << "] is denied in CINN";
    return false;
  }
  if (!IsRegisteredInCINN(op)) {
    VLOG(5) << op.name() << "[id:" << op.id() << "] isn't registered in CINN";
    return false;
  }
  if (!HasHandledInPass(op)) {
    VLOG(5) << op.name() << "[id:" << op.id() << "] isn't handled in CINN";
    return false;
  }
  if (CauseNewSymbolicShape(op)) {
    VLOG(5) << op.name() << "[id:" << op.id()
            << "] caused new symbolic shape in CINN";
    return false;
  }
  return true;
}
}  // namespace

bool IsComplex(const ::pir::Operation& op) {
  const auto& IsComplexType = [&](const ::pir::Value& value) -> bool {
    if (!value) {
      return false;
    }
    auto type = value.type();
    if (!type) {
      return false;
    }
    if (type.isa<paddle::dialect::DenseTensorType>()) {
      auto dtype = type.dyn_cast<paddle::dialect::DenseTensorType>().dtype();
      if (dtype && (dtype.isa<::pir::Complex64Type>() ||
                    dtype.isa<::pir::Complex128Type>())) {
        return true;
      }
    }
    return false;
  };

  for (size_t i = 0; i < op.num_operands(); ++i) {
    if (IsComplexType(op.operand_source(i))) {
      return true;
    }
  }

  for (size_t i = 0; i < op.num_results(); ++i) {
    if (IsComplexType(op.result(i))) {
      return true;
    }
  }

  return false;
}

bool CompatibleInfo::IsDeniedForCinn(const ::pir::Operation& op) {
  bool flag = IsDeniedInCinn(op) || CauseNewSymbolicShape(op) || IsComplex(op);
  VLOG(4) << "CompatibleInfo::IsDeniedForCinn of " << op.name()
          << " is: " << flag;
  return flag;
}

bool CompatibleInfo::IsSupportForCinn(const ::pir::Operation& op) {
  // check input or output
  if (IsComplex(op)) {
    return false;
  }
  const bool not_builtin_op = op.dialect()->name() != "builtin";
  const bool flag = IsSupportInCinn(op) && not_builtin_op;

  VLOG(4) << "CompatibleInfo::IsSupportForCinn of " << op.name()
          << " is: " << flag;
  return flag;
}

std::string CompatibleInfo::OpName(const ::pir::Operation& op) {
  std::string name = op.name();
  if (OP_NAMES.count(name)) {
    return OP_NAMES.at(name);
  }
  return OpNameAfterStripDialect(op);
}

std::string ShortenOpName(const std::string& name) {
  static const std::unordered_map<std::string, std::string> OP_SHORT_NAMES = {
      {"fill_constant", "full"},
      {"reduce_sum", "sum"},
      {"reduce_max", "r_max"},
      {"reduce_min", "r_min"},
      {"reduce_prod", "prod"},
      {"elementwise_add", "add"},
      {"elementwise_mul", "mul"},
      {"subtract", "sub"},
      {"divide", "div"},
      {"broadcast_to", "bc"},
      {"generate_shape", "gs"},
      {"yield_store", "yield"},
  };
  if (OP_SHORT_NAMES.count(name)) {
    return OP_SHORT_NAMES.at(name);
  }
  return name;
}

std::string CompatibleInfo::OpFuncName(const ::pir::Operation& op) {
  std::string op_name = ShortenOpName(OpName(op));
  std::string func_name =
      cinn::common::Context::Global().NewName("fn_" + op_name);
  return func_name;
}

std::string CompatibleInfo::GroupOpsName(
    const std::vector<::pir::Operation*>& ops) {
  std::string name = "fn_";
  for (auto* op : ops) {
    name += ShortenOpName(OpName(*op));
    name += "_";
  }
  return cinn::common::Context::Global().NewName(name);
}

std::string CompatibleInfo::ValueName(const ::pir::Value& value) {
  size_t hash_key = std::hash<::pir::Value>()(value);
  return cinn::common::Context::Global().PrettyUniqName(
      hash_key, CompatibleInfo::kNamePrefix);
}

std::vector<::pir::Value> CompatibleInfo::RealOperandSources(
    const ::pir::Operation& op) {
  if (OpMapper::Instance().has(op, MapperType::OPERAND)) {
    return OpMapper::Instance().RealOperandSources(op);
  } else {
    return op.operands_source();
  }
}

#define CASE_ATTRIBUTE(val_type, attr_type)                     \
  std::vector<val_type> res;                                    \
  for (auto element : attr_vec) {                               \
    res.push_back(element.dyn_cast<::pir::attr_type>().data()); \
  }                                                             \
  dst_attr = res;

static utils::Attribute ConvertArrayAttribute(
    const ::pir::Attribute& src_attr) {
  utils::Attribute dst_attr;
  if (src_attr.isa<paddle::dialect::IntArrayAttribute>()) {
    auto& arr = src_attr.dyn_cast<paddle::dialect::IntArrayAttribute>()
                    .data()
                    .GetData();
    std::vector<int> val(arr.begin(), arr.end());
    dst_attr = val;
  } else if (src_attr.isa<paddle::dialect::DataTypeAttribute>()) {
    auto dtype = src_attr.dyn_cast<paddle::dialect::DataTypeAttribute>().data();
    dst_attr = phi::DataTypeToString(dtype);
  } else if (src_attr.isa<::pir::ArrayAttribute>()) {
    auto attr_vec = src_attr.dyn_cast<::pir::ArrayAttribute>().AsVector();
    if (attr_vec.size() > 0) {
      if (attr_vec[0].isa<::pir::Int32Attribute>()) {
        CASE_ATTRIBUTE(int32_t, Int32Attribute)
      } else if (attr_vec[0].isa<::pir::Int64Attribute>()) {
        CASE_ATTRIBUTE(int64_t, Int64Attribute)
      } else if (attr_vec[0].isa<::pir::BoolAttribute>()) {
        CASE_ATTRIBUTE(bool, BoolAttribute)
      } else if (attr_vec[0].isa<::pir::FloatAttribute>()) {
        CASE_ATTRIBUTE(float, FloatAttribute)
      } else if (attr_vec[0].isa<::pir::DoubleAttribute>()) {
        CASE_ATTRIBUTE(double, DoubleAttribute)
      } else if (attr_vec[0].isa<::pir::StrAttribute>()) {
        std::vector<std::string> dst_attr;
        for (auto element : attr_vec) {
          dst_attr.push_back(
              element.dyn_cast<::pir::StrAttribute>().AsString());
        }
      } else {
        PADDLE_THROW(::common::errors::InvalidArgument(
            "only support bool/int32/int64/float/double/string attribute in "
            "ArrayAttribute"));
      }
    }
    // TODO(xiazichao): ADD branch logic for 0-size ArrayAttribute.
  } else if (src_attr.isa<::pir::shape::SymbolAttribute>()) {
    // do nothing for now
  } else {
    std::stringstream ss;
    ss << "unknown Attribute: " << src_attr;
    PADDLE_THROW(::common::errors::InvalidArgument(ss.str()));
  }
  return dst_attr;
}
#undef CASE_ATTRIBUTE

#define CASE_SINGLE_ATTR(attr_type, func)               \
  else if (src_attr.isa<::pir::attr_type>()) dst_attr = \
      src_attr.dyn_cast<::pir::attr_type>().func();

utils::Attribute CompatibleInfo::ConvertAttribute(
    const ::pir::Attribute& src_attr) {
  utils::Attribute dst_attr;
  if (src_attr.isa<::pir::BoolAttribute>())
    dst_attr = src_attr.dyn_cast<::pir::BoolAttribute>().data();
  CASE_SINGLE_ATTR(FloatAttribute, data)
  CASE_SINGLE_ATTR(DoubleAttribute, data)
  CASE_SINGLE_ATTR(Int32Attribute, data)
  CASE_SINGLE_ATTR(Int64Attribute, data)
  CASE_SINGLE_ATTR(StrAttribute, AsString)
  else if (src_attr.isa<::pir::shape::SymbolAttribute>()) return dst_attr;
  else dst_attr = ConvertArrayAttribute(src_attr);  // NOLINT
  return dst_attr;
}
#undef CASE_SINGLE_ATTR

utils::AttributeMap CompatibleInfo::ConvertAttributes(
    const ::pir::Operation& op) {
  auto& src_attrs = op.attributes();
  utils::AttributeMap dst_attrs;
  for (auto& item : src_attrs) {
    VLOG(4) << "deal with " << item.first;
    if (item.first == ::pir::kStopGradientAttrName) {
      continue;
    } else if (item.first == ::pir::kSymbolBindings) {
      auto symbol_bindings =
          cinn::dialect::GenerateShapeOp::ConvertAttributeToSymbolBindings(
              item.second);
      PADDLE_ENFORCE(symbol_bindings.has_value(),
                     ::common::errors::PreconditionNotMet(
                         "Required success to execute convert attribute to "
                         "symbol bindings."));
      dst_attrs[::pir::kSymbolBindings] = symbol_bindings.value();
    } else if (item.first == ::pir::kOutputDimExprs) {
      auto dim_exprs = cinn::dialect::ConvertAttributeToDimExprs(item.second);
      PADDLE_ENFORCE(
          dim_exprs.has_value(),
          ::common::errors::PreconditionNotMet(
              "Required success to execute convert attribute to dim exprs."));
      dst_attrs[::pir::kOutputDimExprs] = dim_exprs.value();
    } else if (item.second.isa<paddle::dialect::PlaceAttribute>()) {
      auto is_cpu =
          item.second.dyn_cast<paddle::dialect::PlaceAttribute>().data() ==
          phi::CPUPlace();
      dst_attrs["force_cpu"] = is_cpu;
    } else {
      dst_attrs[item.first] = std::move(ConvertAttribute(item.second));
    }
  }

  if (OpMapper::Instance().has(op, MapperType::ATTRIBUTE)) {
    OpMapper::Instance().AppendVariantAttrs(op, dst_attrs);
  }
  VLOG(4) << "dst_attrs.size(): " << dst_attrs.size();
  return dst_attrs;
}

#define CASE_TYPE(src, dst) \
  else if (type.isa<::pir::src>()) return cinn::common::dst();

cinn::common::Type CompatibleInfo::ConvertIRType(::pir::Type type) {
  if (type.isa<::pir::BFloat16Type>()) return cinn::common::BF16();
  CASE_TYPE(Float16Type, F16)
  CASE_TYPE(Float32Type, F32)
  CASE_TYPE(Float64Type, F64)
  CASE_TYPE(Int8Type, I8)
  CASE_TYPE(UInt8Type, UI8)
  CASE_TYPE(Int16Type, I16)
  CASE_TYPE(Int32Type, I32)
  CASE_TYPE(Int64Type, I64)
  CASE_TYPE(IndexType, I32)
  CASE_TYPE(BoolType, UI1)

  std::stringstream ss;
  ss << "unknown ir::Type " << type;
  PADDLE_THROW(::common::errors::InvalidArgument(ss.str()));
}
#undef CASE_TYPE

int CompatibleInfo::ShapeProduct(const std::vector<int>& shape) {
  return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
}

OpPatternKind CompatibleInfo::OpKind(const ::pir::Operation& op) {
  auto& op_pattern_dict = Operator::GetAttrs<OpPatternKind>("OpPattern");
  auto op_name = CompatibleInfo::OpName(op);
  if (op_name == "generate_shape") {
    return hlir::framework::kElementWise;
  }
  const hlir::framework::Operator* cinn_op = Operator::Get(op_name);
  PADDLE_ENFORCE_EQ(
      op_pattern_dict.Find(cinn_op),
      true,
      ::common::errors::PreconditionNotMet(
          "Failed to find the op pattern kind for the operator in "
          "Operator::GetAttrs<OpPatternKind>. "
          "Ensure that the operator is registered "
          "and its pattern is available."));
  auto kind = op_pattern_dict[cinn_op];
  if (kind == hlir::framework::kBroadcast) {
    // As binary op was defined as broadcast, actually it should be
    // element-wise. See fusion_helper_base.h for detail.
    if (op_name != "broadcast_to") {
      kind = hlir::framework::kElementWise;
    }
  }
  VLOG(4) << op_name << " OpPatternKind: " << kind;
  return kind;
}

std::vector<int> CompatibleInfo::ValueShape(const ::pir::Value& value) {
  auto& dim = value.type().dyn_cast<::pir::DenseTensorType>().dims();
  return ::common::vectorize<int>(dim);
}

std::vector<int64_t> GetBroadcastAxis(const phi::DDim& in_shape,
                                      const std::vector<int64_t>& out_shape) {
  std::vector<int64_t> broadcast_axes(in_shape.size(), 0);
  auto in_shape_size = in_shape.size();
  if (in_shape_size >= 1) {
    for (int i = 1; i <= in_shape_size; ++i) {
      broadcast_axes[in_shape_size - i] = out_shape.size() - i;
    }
  }

  return broadcast_axes;
}

std::vector<::pir::Value> GetBlockOutsideInput(
    const std::vector<::pir::Operation*>& op_list) {
  std::vector<::pir::Value> vec_res;
  std::unordered_set<::pir::Value> block_inner_output;
  for (size_t k = 0; k < op_list.size(); ++k) {
    for (size_t i = 0; i < op_list[k]->num_results(); ++i) {
      block_inner_output.insert(op_list[k]->result(i));
    }
  }

  std::unordered_set<::pir::Value> insert_value;
  for (size_t k = 0; k < op_list.size(); ++k) {
    for (size_t i = 0; i < op_list[k]->num_operands(); ++i) {
      if (!block_inner_output.count(op_list[k]->operand_source(i)) &&
          !insert_value.count(op_list[k]->operand_source(i))) {
        vec_res.push_back(op_list[k]->operand_source(i));
        insert_value.insert(op_list[k]->operand_source(i));
      }
    }
  }
  return vec_res;
}

}  // namespace pir
}  // namespace framework
}  // namespace hlir
}  // namespace cinn
