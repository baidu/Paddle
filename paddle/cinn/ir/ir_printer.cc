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

#include "paddle/cinn/ir/ir_printer.h"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <vector>
#include "paddle/cinn/ir/lowered_func.h"
#include "paddle/cinn/ir/module.h"
#include "paddle/cinn/ir/tensor.h"
#include "paddle/cinn/optim/ir_simplify.h"
#include "paddle/cinn/runtime/intrinsic.h"
#include "paddle/cinn/utils/string.h"
#include "paddle/common/enforce.h"

namespace cinn {
namespace ir {

using cinn::common::bfloat16;
using cinn::common::float16;

void IrPrinter::Print(const Expr &e) {
  Visit(e);
  os_ << str_;
  str_ = "";
}
void IrPrinter::Print(const stmt::StmtRef &stmt) {
  VisitStmt(stmt);
  os_ << str_;
  str_ = "";
}
void IrPrinter::Print(const stmt::BlockRef &block) {
  VisitBlock(block);
  os_ << str_;
  str_ = "";
}
void IrPrinter::Print(const std::vector<Expr> &exprs,
                      const std::string &splitter) {
  for (std::size_t i = 0; !exprs.empty() && i + 1 < exprs.size(); i++) {
    Visit(exprs[i]);
    str_ += splitter;
  }
  if (!exprs.empty()) Visit(exprs.back());
  os_ << str_;
  str_ = "";
}

void IrPrinter::Print(const ir::LoweredFunc &fn) {
  Visit(fn.As<ir::_LoweredFunc_>());
  os_ << str_;
  str_ = "";
}

void IrPrinter::Visit(const IntImm *x) {
  if (x->type().is_int(64)) {
    str_ += std::to_string(x->value);
    str_ += "ll";
  } else if (x->type().is_int(32)) {
    // The min int32_t constant(-2147483648) will be recognized as long
    // and max(long, int32_t) is illegal, so we need to add cast here.
    if (x->value == std::numeric_limits<std::int32_t>::min()) {
      str_ += "(int32_t)";
    }
    str_ += std::to_string(x->value);
  } else if (x->type().is_int(16)) {
    str_ += "(int16_t)";
    str_ += std::to_string(x->value);
  } else if (x->type().is_int(8)) {
    str_ += "(int8_t)";
    str_ += std::to_string(x->value);
  } else {
    std::stringstream ss;
    ss << "Not support int type: " << x->type();
    PADDLE_THROW(::common::errors::InvalidArgument(ss.str()));
  }
}
void IrPrinter::Visit(const UIntImm *x) {
  if (x->type().is_uint(64)) {
    str_ += std::to_string(x->value);
    str_ += "ull";
  } else if (x->type().is_uint(32)) {
    str_ += std::to_string(x->value);
  } else if (x->type().is_uint(16)) {
    str_ += "(uint16_t)";
    str_ += std::to_string(x->value);
  } else if (x->type().is_uint(8)) {
    str_ += "(uint8_t)";
    str_ += std::to_string(x->value);
  } else if (x->type().is_uint(1)) {
    if (x->value) {
      str_ += "true";
    } else {
      str_ += "false";
    }
  } else {
    std::stringstream ss;
    ss << "Not support uint type: " << x->type();
    PADDLE_THROW(::common::errors::InvalidArgument(ss.str()));
  }
}

namespace {
template <typename T>
bool IsCloseEqualBoundValue(T value) {
  T max_value = std::numeric_limits<T>::max();
  T min_value = std::numeric_limits<T>::lowest();
  T tol = std::numeric_limits<T>::denorm_min();
  return (max_value - value) < tol || (value - min_value) < tol;
}

template <typename T>
T TruncateInfinity(T value) {
  T max_value = std::numeric_limits<T>::max();
  T min_value = std::numeric_limits<T>::lowest();
  if (value > max_value) {
    return max_value;
  }
  if (value < min_value) {
    return min_value;
  }
  return value;
}

}  // namespace

void IrPrinter::Visit(const FloatImm *x) {
  std::ostringstream ss;
  if (x->type().is_float16()) {
    if (std::isinf(x->value)) {
      if (x->value == std::numeric_limits<double>::infinity()) {
        ss << "cinn::common::raw_uint16_to_float16(0x7c00)";
      } else {
        ss << "cinn::common::raw_uint16_to_float16(0xfc00)";
      }
    } else if (std::isnan(x->value)) {
      ss << "cinn::common::raw_uint16_to_float16(0x7e00)";
    } else {
      ss << "(float16)";
      ss << std::setprecision(std::numeric_limits<float16>::max_digits10);
      ss << static_cast<float16>(x->value) << "f";
    }
  } else if (x->type().is_bfloat16()) {
    if (std::isinf(x->value)) {
      if (x->value == std::numeric_limits<double>::infinity()) {
        ss << "cinn::common::raw_uint16_to_bfloat16(0x7F80)";
      } else {
        ss << "cinn::common::raw_uint16_to_bfloat16(0xFF80)";
      }
    } else if (std::isnan(x->value)) {
      ss << "cinn::common::raw_uint16_to_bfloat16(0x7FC0)";
    } else {
      ss << "(bfloat16)";
      ss << std::setprecision(std::numeric_limits<bfloat16>::max_digits10);
      ss << static_cast<bfloat16>(x->value) << "f";
    }
  } else if (x->type().is_float(32)) {
    if (std::isinf(x->value)) {
      if (x->value == std::numeric_limits<double>::infinity()) {
        ss << "__int_as_float(0x7f800000)";
      } else {
        ss << "__int_as_float(0xff800000)";
      }
    } else {
      float v = TruncateInfinity<float>(x->value);
      if (IsCloseEqualBoundValue<float>(v)) v = std::trunc(v);
      ss << std::setprecision(std::numeric_limits<float>::max_digits10);
      ss << std::showpoint;
      ss << v;
      if (std::isfinite(v)) {
        ss << "f";
      }
    }
  } else if (x->type().is_float(64)) {
    if (std::isinf(x->value)) {
      if (x->value == std::numeric_limits<double>::infinity()) {
        ss << "__int_as_float(0x7f800000)";
      } else {
        ss << "__int_as_float(0xff800000)";
      }
    } else {
      ss << std::setprecision(std::numeric_limits<double>::max_digits10);
      ss << std::showpoint;
      ss << x->value;
    }
  } else {
    std::stringstream ss;
    ss << "Not support float type: " << x->type();
    PADDLE_THROW(::common::errors::InvalidArgument(ss.str()));
  }
  str_ += ss.str();
}
void IrPrinter::Visit(const StringImm *x) {
  str_ += "\"";
  str_ += x->value;
  str_ += "\"";
}

void IrPrinter::Visit(const IterMark *x) {
  str_ += "IterMark(";
  Visit(x->source);
  str_ += ",";
  Visit(x->extent);
  str_ += ")";
}
void IrPrinter::Visit(const IterSum *x) {
  str_ += "IterSum(";
  for (const auto &arg : x->args) {
    Visit(arg);
    str_ += "+";
  }
  Visit(x->base);
  str_ += ")";
}
void IrPrinter::Visit(const IterSplit *x) {
  str_ += "IterSplit(";
  Visit(x->source);
  str_ += "/";
  Visit(x->lower_factor);
  str_ += "%";
  Visit(x->extent);
  str_ += "*";
  Visit(x->scale);
  str_ += ")";
}

void IrPrinter::Visit(const Add *x) { PrintBinaryOp("+", x); }
void IrPrinter::Visit(const Sub *x) { PrintBinaryOp("-", x); }
void IrPrinter::Visit(const Mul *x) { PrintBinaryOp("*", x); }
void IrPrinter::Visit(const Div *x) { PrintBinaryOp("/", x); }
void IrPrinter::Visit(const Mod *x) { PrintBinaryOp("%", x); }
void IrPrinter::Visit(const EQ *x) { PrintBinaryOp("==", x); }
void IrPrinter::Visit(const NE *x) { PrintBinaryOp("!=", x); }
void IrPrinter::Visit(const LT *x) { PrintBinaryOp("<", x); }
void IrPrinter::Visit(const LE *x) { PrintBinaryOp("<=", x); }
void IrPrinter::Visit(const GT *x) { PrintBinaryOp(">", x); }
void IrPrinter::Visit(const GE *x) { PrintBinaryOp(">=", x); }
void IrPrinter::Visit(const And *x) { PrintBinaryOp("and", x); }
void IrPrinter::Visit(const Or *x) { PrintBinaryOp("or", x); }
void IrPrinter::Visit(const Not *x) {
  str_ += "!";
  Visit(x->v());
}
void IrPrinter::Visit(const Min *x) {
  str_ += "cinn_min(";
  Visit(x->a());
  str_ += ", ";
  Visit(x->b());
  str_ += ")";
}
void IrPrinter::Visit(const Max *x) {
  str_ += "cinn_max(";
  Visit(x->a());
  str_ += ", ";
  Visit(x->b());
  str_ += ")";
}
void IrPrinter::Visit(const Minus *x) {
  str_ += "-(";
  Visit(x->v());
  str_ += ")";
}
void IrPrinter::Visit(const For *x) {
  if (x->is_parallel()) {
    str_ += "parallel for (";
  } else if (x->is_unrolled()) {
    str_ += "unroll for (";
  } else if (x->is_vectorized()) {
    int factor = x->vectorize_info().factor;
    str_ += "vectorize[";
    str_ += std::to_string(factor);
    str_ += "] for (";
  } else if (x->is_binded()) {
    auto &bind_info = x->bind_info();
    if (bind_info.valid()) {
      char axis_name = 'x' + bind_info.offset;
      auto for_type = bind_info.for_type;
      std::string prefix =
          for_type == ForType::GPUBlock ? "blockIdx." : "threadIdx.";
      str_ += "thread_bind[";
      str_ += prefix;
      str_ += axis_name;
      str_ += "] for (";
    } else {
      str_ += "thread_bind[invalid info] for (";
    }
  } else if (x->is_serial()) {
    str_ += "serial for (";
  } else if (x->is_default()) {
    str_ += "default for (";
  } else {
    str_ += "for (";
  }
  Visit(x->loop_var);
  str_ += ", ";
  Visit(x->min);
  str_ += ", ";
  Visit(x->extent);
  str_ += ")\n";

  DoIndent();
  Visit(x->body);
}

void IrPrinter::Visit(const PolyFor *x) {
  if (x->is_parallel()) {
    str_ += "parallel poly_for (";
  } else {
    str_ += "poly_for (";
  }
  Visit(x->iterator);
  str_ += ", ";
  Visit(x->init);
  str_ += ", ";
  Visit(x->condition);
  str_ += ", ";
  Visit(x->inc);
  str_ += ")\n";

  DoIndent();
  Visit(x->body);
}
void IrPrinter::Visit(const IfThenElse *x) {
  str_ += "if (";
  Visit(x->condition);
  str_ += ") ";
  Visit(x->true_case);

  if (x->false_case.defined()) {
    str_ += " else ";
    Visit(x->false_case);
  }
}
void IrPrinter::Visit(const Block *x) {
  str_ += "{\n";

  IncIndent();
  for (std::size_t i = 0; !x->stmts.empty() && i + 1 < x->stmts.size(); i++) {
    DoIndent();
    Visit(x->stmts[i]);
    str_ += "\n";
  }
  if (!x->stmts.empty()) {
    DoIndent();
    Visit(x->stmts.back());
  }
  DecIndent();
  str_ += "\n";
  DoIndent();
  str_ += "}";
}
void IrPrinter::Visit(const Call *x) {
  str_ += x->name;
  str_ += "(";
  if (!x->read_args.empty()) {
    for (std::size_t i = 0; i + 1 < x->read_args.size(); i++) {
      Visit(x->read_args[i]);
      str_ += ", ";
    }
    Visit(x->read_args.back());
  }

  if (!x->write_args.empty()) {
    if (!x->read_args.empty()) str_ += ", ";

    for (std::size_t i = 0; i + 1 < x->write_args.size(); i++) {
      Visit(x->write_args[i]);
      str_ += ", ";
    }
    Visit(x->write_args.back());
  }

  str_ += ")";
}
void IrPrinter::Visit(const Cast *x) {
  str_ += x->type().to_string();
  str_ += "(";
  Visit(x->v());
  str_ += ")";
}
void IrPrinter::Visit(const _Module_ *x) {}
void IrPrinter::Visit(const _Var_ *x) { str_ += x->name; }
void IrPrinter::Visit(const Alloc *x) {
  auto *buffer = x->destination.As<ir::_Buffer_>();
  PADDLE_ENFORCE_NOT_NULL(buffer,
                          ::common::errors::InvalidArgument(
                              "The destination is not a valid buffer. "
                              "Please ensure that `x->destination` is "
                              "properly assigned to a buffer."));
  str_ += "alloc(";
  str_ += buffer->name;
  str_ += ", ";
  Visit(x->extents);
  str_ += ")";
}
void IrPrinter::Visit(const Select *x) {
  str_ += "select(";
  Visit(x->condition);
  str_ += ", ";
  Visit(x->true_value);
  str_ += ", ";
  Visit(x->false_value);
  str_ += ")";
}
void IrPrinter::Visit(const Load *x) {
  if (x->is_addr_tensor()) {
    auto *tensor = x->tensor.As<ir::_Tensor_>();
    PADDLE_ENFORCE_NOT_NULL(
        tensor,
        ::common::errors::InvalidArgument("The tensor is not valid. "
                                          "Please ensure that `x->tensor` is "
                                          "properly assigned to a tensor."));
    str_ += tensor->name;
  } else if (x->is_addr_scalar()) {
    Visit(x->tensor);
  } else {
    CINN_NOT_IMPLEMENTED
  }

  str_ += "[";
  for (std::size_t i = 0; i + 1 < x->indices.size(); i++) {
    Visit(x->indices[i]);
    str_ += ", ";
  }
  if (!x->indices.empty()) Visit(x->indices.back());
  str_ += "]";
}
void IrPrinter::Visit(const Store *x) {
  if (x->is_addr_tensor()) {
    auto *tensor_node = x->tensor.As<ir::_Tensor_>();
    PADDLE_ENFORCE_NOT_NULL(tensor_node,
                            ::common::errors::InvalidArgument(
                                "The tensor node is not valid. "
                                "Please ensure that `x->tensor` is "
                                "properly assigned to a tensor node."));

    str_ += tensor_node->name;
  } else if (x->is_addr_scalar()) {
    Visit(x->tensor);
  } else {
    CINN_NOT_IMPLEMENTED
  }

  str_ += "[";
  for (std::size_t i = 0; i + 1 < x->indices.size(); i++) {
    Visit(x->indices[i]);
    str_ += ", ";
  }
  if (!x->indices.empty()) Visit(x->indices.back());
  str_ += "] = ";
  Visit(x->value);
}
void IrPrinter::Visit(const Free *x) {
  auto *buffer = x->destination.As<ir::_Buffer_>();
  PADDLE_ENFORCE_NOT_NULL(buffer,
                          ::common::errors::InvalidArgument(
                              "The destination is not a valid buffer. "
                              "Please ensure that `x->destination` is "
                              "properly assigned to a buffer."));

  str_ += "free(";
  str_ += buffer->name;
  str_ += ")";
}

void IrPrinter::DoIndent() { str_ += std::string(indent_, ' '); }
void IrPrinter::IncIndent() { indent_ += indent_unit; }
void IrPrinter::DecIndent() { indent_ -= indent_unit; }

void IrPrinter::Visit(const _Buffer_ *x) {
  std::vector<std::string> dim_names;
  std::transform(x->shape.begin(),
                 x->shape.end(),
                 std::back_inserter(dim_names),
                 [&](const Expr &x) { return utils::GetStreamCnt(x); });

  str_ += "_Buffer_<";

  str_ += x->type().to_string();
  str_ += ": ";
  str_ += utils::Join(dim_names, ",");
  str_ += ">(";
  str_ += x->name;
  str_ += ")";
}
void IrPrinter::Visit(const _Tensor_ *x) {
  str_ += "Tensor(";
  str_ += x->name;
  str_ += ", ";
  str_ += "[";
  if (!x->shape.empty()) {
    for (std::size_t i = 0; i + 1 < x->shape.size(); i++) {
      Visit(x->shape[i]);
      str_ += ",";
    }
    Visit(x->shape.back());
  }
  str_ += "])";
}
void IrPrinter::Visit(const _LoweredFunc_ *f) {
  str_ += "function ";
  str_ += f->name;
  str_ += " ";

  std::vector<std::string> arg_names;
  for (auto &arg : f->args) {
    arg_names.push_back(arg.name());
  }
  str_ += "(";
  str_ += utils::Join(arg_names, ", ");
  str_ += ")\n";

  if (f->body.defined()) {
    Visit(f->body);
  } else {
    PADDLE_ENFORCE_EQ(
        f->body_block.defined(),
        true,
        ::common::errors::InvalidArgument(
            "Please ensure that `f->body` or `f->body_block` is defined."));
    VisitBlock(f->body_block);
  }
}
void IrPrinter::Visit(const Let *f) {
  PADDLE_ENFORCE_EQ(
      f->type().valid(),
      true,
      ::common::errors::InvalidArgument(
          "The type of `f` is not valid. "
          "Please ensure that `f->type()` returns a valid type."));

  str_ += f->type().to_string();
  str_ += " ";
  Visit(f->symbol);
  if (f->body.defined()) {
    str_ += " = ";
    Visit(f->body);
  }
}

void IrPrinter::Visit(const Reduce *f) {
  str_ += "Reduce(";
  switch (f->reduce_type) {
    case Reduce::ReduceType::kSum:
      str_ += "sum";
      break;
    case Reduce::ReduceType::kSub:
      str_ += "sub";
      break;
    case Reduce::ReduceType::kDiv:
      str_ += "Div";
      break;
    case Reduce::ReduceType::kMul:
      str_ += "Mul";
      break;
    case Reduce::ReduceType::kMax:
      str_ += "Max";
      break;
    case Reduce::ReduceType::kMin:
      str_ += "Min";
      break;
    case Reduce::ReduceType::kAll:
      str_ += "&&";
      break;
    case Reduce::ReduceType::kAny:
      str_ += "||";
      break;
  }
  str_ += ", ";
  Visit(f->body);
  str_ += ",";
  Visit(f->init);
  str_ += ")";
}

void IrPrinter::Visit(const Ramp *x) {
  str_ += "Ramp(";
  Visit(x->base);
  str_ += ",";
  Visit(x->stride);
  str_ += ",";
  str_ += std::to_string(x->lanes);
  str_ += ")";
}

void IrPrinter::Visit(const Broadcast *x) {
  str_ += "Broadcast(";
  Visit(x->value);
  str_ += ",";
  str_ += std::to_string(x->lanes);
  str_ += ")";
}

void IrPrinter::Visit(const FracOp *x) {
  str_ += "(";
  Visit(x->a());
  str_ += " / ";
  Visit(x->b());
  str_ += ")";
}

void IrPrinter::Visit(const Product *x) {
  str_ += "(";
  for (std::size_t i = 0; i + 1 < x->operands().size(); i++) {
    Visit(x->operand(i));
    str_ += " * ";
  }
  if (!x->operands().empty()) Visit(x->operands().back());
  str_ += ")";
}

void IrPrinter::Visit(const Sum *x) {
  str_ += "(";
  for (std::size_t i = 0; i + 1 < x->operands().size(); i++) {
    Visit(x->operand(i));
    str_ += " + ";
  }
  if (!x->operands().empty()) Visit(x->operands().back());
  str_ += ")";
}

void IrPrinter::Visit(const PrimitiveNode *x) {
  str_ += x->name;
  str_ += "(";
  std::vector<std::string> args_repr;
  for (auto &args : x->arguments) {
    std::vector<std::string> arg_repr;
    for (auto &arg : args) {
      arg_repr.push_back(utils::GetStreamCnt(arg));
    }
    args_repr.push_back(utils::Join(arg_repr, ","));
  }

  str_ += utils::Join(args_repr, ",");
  str_ += ")";
}

void IrPrinter::Visit(const _BufferRange_ *x) {
  auto *buffer = x->buffer.As<ir::_Buffer_>();
  PADDLE_ENFORCE_NOT_NULL(
      buffer,
      ::common::errors::InvalidArgument(
          "The buffer is not valid. "
          "Please ensure that `x->buffer` is properly assigned to a buffer."));
  str_ += buffer->name;
  str_ += "[";
  for (std::size_t i = 0; i < x->ranges.size(); i++) {
    if (i) str_ += ", ";
    auto &range = x->ranges[i];
    str_ += range->name;
    str_ += "(";
    if (range->lower_bound.defined()) {
      Visit(range->lower_bound);
      str_ += ":";
    } else {
      str_ += "undefined:";
    }

    if (range->upper_bound.defined()) {
      Visit(range->upper_bound);
    } else {
      str_ += "undefined";
    }
    str_ += ")";
  }
  str_ += "]";
}

void IrPrinter::Visit(const ScheduleBlock *x) {}

void IrPrinter::Visit(const ScheduleBlockRealize *x) {
  auto *schedule_block = x->schedule_block.As<ScheduleBlock>();
  str_ += "ScheduleBlock(";
  str_ += schedule_block->name;
  str_ += ")\n";
  DoIndent();
  str_ += "{\n";
  // print block vars and bindings
  auto iter_vars = schedule_block->iter_vars;
  auto iter_values = x->iter_values;
  PADDLE_ENFORCE_EQ(iter_vars.size(),
                    iter_values.size(),
                    ::common::errors::InvalidArgument(
                        "iter_vars.size() != iter_values.size()"));
  IncIndent();
  if (!iter_vars.empty()) DoIndent();
  for (std::size_t i = 0; i < iter_vars.size(); i++) {
    if (i) str_ += ", ";
    str_ += iter_vars[i]->name;
  }
  if (!iter_vars.empty()) str_ += " = axis.bind(";
  for (std::size_t i = 0; i < iter_values.size(); i++) {
    if (i) str_ += ", ";
    Visit(iter_values[i]);
  }
  if (!iter_vars.empty()) str_ += ")\n";
  // print block body
  if (!schedule_block->read_buffers.empty()) {
    DoIndent();
    str_ += "read_buffers(";
    auto &read_buffers = schedule_block->read_buffers;
    for (std::size_t i = 0; i < read_buffers.size(); i++) {
      if (i) str_ += ", ";
      Visit(read_buffers[i]);
    }
    str_ += ")\n";
  }
  if (!schedule_block->write_buffers.empty()) {
    DoIndent();
    str_ += "write_buffers(";
    auto &write_buffers = schedule_block->write_buffers;
    for (std::size_t i = 0; i < write_buffers.size(); i++) {
      if (i) str_ += ", ";
      Visit(write_buffers[i]);
    }
    str_ += ")\n";
  }
  if (!schedule_block->attrs.empty()) {
    DoIndent();
    str_ += "attrs(";
    bool comma = false;
    for (auto &&kv : schedule_block->attrs) {
      if (comma) str_ += ", ";
      str_ += kv.first;
      str_ += ":";
      absl::visit(
          [this](auto &&arg) {
            std::ostringstream ss;
            ss << arg;
            this->str_ += ss.str();
          },
          kv.second);
      comma = true;
    }
    str_ += ")\n";
  }
  DoIndent();
  Visit(schedule_block->body);
  str_ += "\n";
  DecIndent();
  DoIndent();
  str_ += "}";
}

void IrPrinter::Visit(const _Dim_ *x) {
  str_ += "Dim(name: ";
  str_ += x->name;
  str_ += ", symbol: ";
  str_ += x->ToString();
  str_ += ")";
}

void IrPrinter::Visit(const IntrinsicOp *x) {
  switch (x->getKind()) {
#define __(op__)                                \
  case IntrinsicKind::k##op__:                  \
    Visit(llvm::dyn_cast<intrinsics::op__>(x)); \
    break;

    INTRINSIC_KIND_FOR_EACH(__)
#undef __
  }
}
void IrPrinter::Visit(const intrinsics::BufferGetDataHandle *x) {
  str_ += runtime::intrinsic::buffer_get_data_handle;
  Visit(x->buffer);
  str_ += ")";
}
void IrPrinter::Visit(const intrinsics::BufferGetDataConstHandle *x) {
  str_ += runtime::intrinsic::buffer_get_data_const_handle;
  Visit(x->buffer);
  str_ += ")";
}
void IrPrinter::Visit(const intrinsics::PodValueToX *x) {
  str_ += "pod_value_to_";
  str_ += x->GetOutputType(0).to_string();
  str_ += "(";
  Visit(x->pod_value_ptr);
  str_ += ")";
}
void IrPrinter::Visit(const intrinsics::BufferCreate *x) {
  str_ += runtime::intrinsic::buffer_create;
  str_ += "()";
}
void IrPrinter::Visit(const intrinsics::GetAddr *x) {
  str_ += "get_addr(";
  Visit(x->data);
  str_ += ")";
}
void IrPrinter::Visit(const intrinsics::ArgsConstruct *x) {
  str_ += runtime::intrinsic::args_construct_repr;
  str_ += "(";
  Visit(std::vector<Expr>(x->args.begin(), x->args.end()));
  str_ += ")";
}

void IrPrinter::Visit(const intrinsics::BuiltinIntrin *x) {
  str_ += runtime::intrinsic::builtin_intrin_repr;
  str_ += "_";
  str_ += x->name;
  str_ += "(";
  if (!x->args.empty()) {
    for (std::size_t i = 0; i + 1 < x->args.size(); i++) {
      Visit(x->args[i]);
      str_ += ", ";
    }
    Visit(x->args.back());
  }

  str_ += ")";
}

void IrPrinter::VisitStmt(const stmt::Let &stmt) {
  PADDLE_ENFORCE_EQ(
      stmt->type().valid(),
      true,
      ::common::errors::InvalidArgument(
          "The type of `f` is not valid. "
          "Please ensure that `f->type()` returns a valid type."));

  str_ += stmt->type().to_string();
  str_ += " ";
  Visit(stmt->symbol());
  if (stmt->body().defined()) {
    str_ += " = ";
    Visit(stmt->body());
  }
}

void IrPrinter::VisitStmt(const stmt::Store &stmt) {
  if (stmt->is_addr_tensor()) {
    auto *tensor_node = stmt->tensor().As<ir::_Tensor_>();
    PADDLE_ENFORCE_NOT_NULL(tensor_node,
                            ::common::errors::InvalidArgument(
                                "The tensor node is not valid. "
                                "Please ensure that `stmt->tensor` is "
                                "properly assigned to a tensor node."));

    str_ += tensor_node->name;
  } else if (stmt->is_addr_scalar()) {
    Visit(stmt->tensor());
  } else {
    CINN_NOT_IMPLEMENTED
  }

  str_ += "[";
  for (std::size_t i = 0; i + 1 < stmt->indices().size(); i++) {
    Visit(stmt->indices()[i]);
    str_ += ", ";
  }
  if (!stmt->indices().empty()) Visit(stmt->indices().back());
  str_ += "] = ";
  Visit(stmt->value());
}

void IrPrinter::VisitStmt(const stmt::Alloc &stmt) {
  auto *buffer = stmt->destination().As<ir::_Buffer_>();
  PADDLE_ENFORCE_NOT_NULL(buffer,
                          ::common::errors::InvalidArgument(
                              "The destination is not a valid buffer. "
                              "Please ensure that `stmt->destination` is "
                              "properly assigned to a buffer."));
  str_ += "alloc(";
  str_ += buffer->name;
  str_ += ", ";
  Visit(stmt->extents());
  str_ += ")";
}

void IrPrinter::VisitStmt(const stmt::Free &stmt) {
  auto *buffer = stmt->destination().As<ir::_Buffer_>();
  PADDLE_ENFORCE_NOT_NULL(buffer,
                          ::common::errors::InvalidArgument(
                              "The destination is not a valid buffer. "
                              "Please ensure that `stmt->destination` is "
                              "properly assigned to a buffer."));

  str_ += "free(";
  str_ += buffer->name;
  str_ += ")";
}

void IrPrinter::VisitStmt(const stmt::IfThenElse &stmt) {
  str_ += "if (";
  Visit(stmt->condition());
  str_ += ") ";
  VisitBlock(stmt->true_case());

  if (!stmt->false_case()->stmts().empty()) {
    str_ += " else ";
    VisitBlock(stmt->false_case());
  }
}

void IrPrinter::VisitStmt(const stmt::For &stmt) {
  if (stmt->is_parallel()) {
    str_ += "parallel for (";
  } else if (stmt->is_unrolled()) {
    str_ += "unroll for (";
  } else if (stmt->is_vectorized()) {
    int factor = stmt->vectorize_info().factor;
    str_ += "vectorize[";
    str_ += std::to_string(factor);
    str_ += "] for (";
  } else if (stmt->is_binded()) {
    auto &bind_info = stmt->bind_info();
    if (bind_info.valid()) {
      char axis_name = 'x' + bind_info.offset;
      auto for_type = bind_info.for_type;
      std::string prefix =
          for_type == ForType::GPUBlock ? "blockIdx." : "threadIdx.";
      str_ += "thread_bind[";
      str_ += prefix;
      str_ += axis_name;
      str_ += "] for (";
    } else {
      str_ += "thread_bind[invalid info] for (";
    }
  } else if (stmt->is_serial()) {
    str_ += "serial for (";
  } else if (stmt->is_default()) {
    str_ += "default for (";
  } else {
    str_ += "for (";
  }
  Visit(stmt->loop_var());
  str_ += ", ";
  Visit(stmt->min());
  str_ += ", ";
  Visit(stmt->extent());
  str_ += ") ";
  VisitBlock(stmt->body());
}

void IrPrinter::VisitStmt(const stmt::Schedule &stmt) {
  str_ += "Schedule (";
  str_ += stmt->name();
  str_ += ") {\n";
  // print block vars and bindings
  auto iter_vars = stmt->iter_vars();
  auto iter_values = stmt->iter_values();
  PADDLE_ENFORCE_EQ(iter_vars.size(),
                    iter_values.size(),
                    ::common::errors::InvalidArgument(
                        "iter_vars.size() != iter_values.size()"));
  IncIndent();
  if (!iter_vars.empty()) DoIndent();
  for (std::size_t i = 0; i < iter_vars.size(); i++) {
    if (i) str_ += ", ";
    str_ += iter_vars[i]->name;
  }
  if (!iter_vars.empty()) str_ += " = axis.bind(";
  for (std::size_t i = 0; i < iter_values.size(); i++) {
    if (i) str_ += ", ";
    Visit(iter_values[i]);
  }
  if (!iter_vars.empty()) str_ += ")\n";

  // print block body
  if (!stmt->read_buffers().empty()) {
    DoIndent();
    str_ += "read_buffers(";
    auto &read_buffers = stmt->read_buffers();
    for (std::size_t i = 0; i < read_buffers.size(); i++) {
      if (i) str_ += ", ";
      Visit(read_buffers[i]);
    }
    str_ += ")\n";
  }
  if (!stmt->write_buffers().empty()) {
    DoIndent();
    str_ += "write_buffers(";
    auto &write_buffers = stmt->write_buffers();
    for (std::size_t i = 0; i < write_buffers.size(); i++) {
      if (i) str_ += ", ";
      Visit(write_buffers[i]);
    }
    str_ += ")\n";
  }
  if (!stmt->attrs().empty()) {
    DoIndent();
    str_ += "attrs(";
    bool comma = false;
    for (auto &&kv : stmt->attrs()) {
      if (comma) str_ += ", ";
      str_ += kv.first;
      str_ += ":";
      absl::visit(
          [this](auto &&arg) {
            std::ostringstream ss;
            ss << arg;
            this->str_ += ss.str();
          },
          kv.second);
      comma = true;
    }
    str_ += ")\n";
  }

  for (const auto &inner_stmt : stmt->body()->stmts()) {
    DoIndent();
    VisitStmt(inner_stmt);
    str_ += "\n";
  }
  DecIndent();
  DoIndent();
  str_ += "}";
}

void IrPrinter::VisitStmt(const stmt::Evaluate &stmt) { Visit(stmt->value()); }

void IrPrinter::VisitBlock(const stmt::BlockRef &block) {
  str_ += "{\n";

  IncIndent();
  for (const auto &stmt : block->stmts()) {
    DoIndent();
    VisitStmt(stmt);
    str_ += "\n";
  }
  DecIndent();
  DoIndent();
  str_ += "}";
}

std::ostream &operator<<(std::ostream &os, Expr a) {
  std::stringstream ss;
  IrPrinter printer(ss);
  printer.Print(a);
  os << ss.str();
  return os;
}

std::ostream &operator<<(std::ostream &os, const ir::LoweredFunc &func) {
  std::stringstream ss;
  IrPrinter printer(ss);
  printer.Print(func);
  os << ss.str();
  return os;
}

std::ostream &operator<<(std::ostream &os, const ir::Module &m) {
  os << "Module " << m->name << " {\n\n";
  for (auto &fn : m->functions) {
    os << fn << '\n';
  }
  os << "\n\n}";
  return os;
}

std::ostream &operator<<(std::ostream &os, const std::vector<Expr> &a) {
  std::stringstream ss;
  IrPrinter printer(ss);
  printer.Print(a);
  os << ss.str();
  return os;
}

std::ostream &operator<<(std::ostream &os, const std::vector<Dim> &a) {
  std::stringstream ss;
  IrPrinter printer(ss);
  printer.Print(std::vector<Expr>(a.begin(), a.end()));
  os << ss.str();
  return os;
}

namespace stmt {
std::ostream &operator<<(std::ostream &os, const stmt::BlockRef &block) {
  std::stringstream ss;
  IrPrinter printer(ss);
  printer.Print(block);
  os << ss.str();
  return os;
}

std::ostream &operator<<(std::ostream &os, const stmt::StmtRef &stmt) {
  std::stringstream ss;
  IrPrinter printer(ss);
  printer.Print(stmt);
  os << ss.str();
  return os;
}
}  // namespace stmt

}  // namespace ir
}  // namespace cinn
