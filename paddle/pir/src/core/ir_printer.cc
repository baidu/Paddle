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

#include <list>
#include <ostream>
#include <string>
#include <unordered_map>

#include "glog/logging.h"
#include "paddle/common/flags.h"
#include "paddle/pir/include/core/block.h"
#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/core/dialect.h"
#include "paddle/pir/include/core/ir_printer.h"
#include "paddle/pir/include/core/operation.h"
#include "paddle/pir/include/core/program.h"
#include "paddle/pir/include/core/utils.h"
#include "paddle/pir/include/core/value.h"

COMMON_DECLARE_bool(print_ir);
COMMON_DECLARE_string(disable_logging_op_attr_list);
namespace pir {

namespace {
constexpr char newline[] = "\n";  // NOLINT
constexpr size_t indent_size = 4;
}  // namespace

void BasicIrPrinter::PrintType(Type type) {
  if (!type) {
    os << "<<NULL TYPE>>";
    return;
  }

  if (type.isa<UndefinedType>()) {
    os << "undefined";
  } else if (type.isa<BFloat16Type>()) {
    os << "bf16";
  } else if (type.isa<Float16Type>()) {
    os << "f16";
  } else if (type.isa<Float32Type>()) {
    os << "f32";
  } else if (type.isa<Float64Type>()) {
    os << "f64";
  } else if (type.isa<BoolType>()) {
    os << "b";
  } else if (type.isa<Int8Type>()) {
    os << "i8";
  } else if (type.isa<UInt8Type>()) {
    os << "u8";
  } else if (type.isa<Int16Type>()) {
    os << "i16";
  } else if (type.isa<Int32Type>()) {
    os << "i32";
  } else if (type.isa<Int64Type>()) {
    os << "i64";
  } else if (type.isa<IndexType>()) {
    os << "index";
  } else if (type.isa<Complex64Type>()) {
    os << "c64";
  } else if (type.isa<Complex128Type>()) {
    os << "c128";
  } else if (type.isa<Float8E4M3FNType>()) {
    os << "f8e4m3fn";
  } else if (type.isa<Float8E5M2Type>()) {
    os << "f8e5m2";
  } else if (type.isa<VectorType>()) {
    os << "vec[";
    auto inner_types = type.dyn_cast<VectorType>().data();
    pir::detail::PrintInterleave(
        inner_types.begin(),
        inner_types.end(),
        [this](Type v) { this->PrintType(v); },
        [this]() { this->os << ","; });
    os << "]";
  } else {
    auto& dialect = type.dialect();
    dialect.PrintType(type, os);
  }
}

void BasicIrPrinter::PrintAttribute(Attribute attr) {
  if (!attr) {
    os << "<#AttrNull>";
    return;
  }

  if (auto t = attr.dyn_cast<TensorNameAttribute>()) {
    std::string t_val = t.data();
    std::string replacement = "\\\"";
    std::string search = "\"";
    size_t found = t_val.find(search);
    while (found != std::string::npos) {
      t_val.replace(found, search.length(), replacement);
      found = t_val.find(search, found + replacement.length());
    }
    os << "\"" << t_val << "\"";
  } else if (auto s = attr.dyn_cast<StrAttribute>()) {
    std::string s_val = s.AsString();
    std::string replacement = "\\\"";
    std::string search = "\"";
    size_t found = s_val.find(search);
    while (found != std::string::npos) {
      s_val.replace(found, search.length(), replacement);
      found = s_val.find(search, found + replacement.length());
    }
    os << "\"" << s_val << "\"";
  } else if (auto b = attr.dyn_cast<BoolAttribute>()) {
    if (b.data()) {
      os << "true";
    } else {
      os << "false";
    }
  } else if (auto f = attr.dyn_cast<FloatAttribute>()) {
    os << f.data();
  } else if (auto d = attr.dyn_cast<DoubleAttribute>()) {
    os << d.data();
  } else if (auto i = attr.dyn_cast<Int32Attribute>()) {
    os << i.data();
  } else if (auto i = attr.dyn_cast<Int64Attribute>()) {
    os << i.data();
  } else if (auto i = attr.dyn_cast<IndexAttribute>()) {
    os << i.data();
  } else if (auto p = attr.dyn_cast<PointerAttribute>()) {
    os << p.data();
  } else if (auto p = attr.dyn_cast<Complex64Attribute>()) {
    os << p.data();
  } else if (auto p = attr.dyn_cast<Complex128Attribute>()) {
    os << p.data();
  } else if (auto arr = attr.dyn_cast<ArrayAttribute>()) {
    const auto& vec = arr.AsVector();
    os << "[";
    pir::detail::PrintInterleave(
        vec.begin(),
        vec.end(),
        [this](Attribute v) { this->PrintAttribute(v); },
        [this]() { this->os << ","; });
    os << "]";
  } else if (auto type = attr.dyn_cast<TypeAttribute>()) {
    os << type.data();
  } else {
    auto& dialect = attr.dialect();
    dialect.PrintAttribute(attr, os);
  }
}

void IrPrinter::AddIndentation() {
  cur_indentation_ += std::string(indent_size, ' ');
}

void IrPrinter::DecreaseIndentation() {
  cur_indentation_.resize(cur_indentation_.size() - indent_size);
}

void IrPrinter::PrintProgram(const Program* program) {
  auto top_level_op = program->module_op();
  for (size_t i = 0; i < top_level_op->num_regions(); ++i) {
    auto& region = top_level_op->region(i);
    PrintRegion(region);
  }
}

void IrPrinter::PrintOperation(const Operation& op) {
  os << indentation();

  if (auto* dialect = op.dialect()) {
    if (auto print_fn = dialect->PrintOperation(op)) {
      print_fn(op, *this);
      return;
    }
  }

  PrintGeneralOperation(op);
}

void IrPrinter::PrintOperationWithNoRegion(const Operation& op) {
  // TODO(lyk): add API to get opresults directly
  PrintOpResult(op);
  os << " = ";
  PrintOpName(op);

  PrintOpId(op);

  // TODO(lyk): add API to get operands directly
  PrintOpOperands(op);

  PrintAttributeMap(op);
  os << " :";

  // PrintOpSignature
  PrintOperandsType(op);
  os << " -> ";

  // TODO(lyk): add API to get opresults directly
  PrintOpReturnType(op);

  os << " # {op_id: " << op->id() << "}";
}

void IrPrinter::PrintGeneralOperation(const Operation& op) {
  PrintOperationWithNoRegion(op);
  if (op.num_regions() > 0) {
    os << newline;
  }
  for (size_t i = 0; i < op.num_regions(); ++i) {
    auto& region = op.region(i);
    PrintRegion(region);
  }
}

void IrPrinter::PrintRegion(const Region& region) {
  for (auto& block : region) {
    PrintBlock(block);
  }
}

void IrPrinter::PrintBlock(const Block& block) {
  os << indentation() << "{\n";
  AddIndentation();
  if (!block.kwargs_empty()) {
    os << indentation() << "^kw:\n";
    AddIndentation();
    for (auto iter = block.kwargs_begin(); iter != block.kwargs_end(); ++iter) {
      os << indentation();
      PrintValue(iter->second);
      os << "\n";
    }
    DecreaseIndentation();
  }
  for (auto& item : block) {
    PrintOperation(item);
    os << "\n";
  }
  DecreaseIndentation();
  os << indentation() << "}\n";
}

void IrPrinter::PrintValue(Value v) {
  if (!v) {
    os << "<<NULL VALUE>>";
    return;
  }
  const void* key = v.impl();
  auto ret = aliases_.find(key);
  if (ret != aliases_.end()) {
    os << ret->second;
    return;
  }
  if (v.isa<OpResult>()) {
    std::string new_name = "%" + std::to_string(cur_result_number_);
    cur_result_number_++;
    aliases_[key] = new_name;
    os << new_name;
  } else {
    auto arg = v.dyn_cast<BlockArgument>();
    os << (aliases_[key] =
               arg.is_kwarg()
                   ? "%kwarg_" + arg.keyword()
                   : "%arg_" + std::to_string(cur_block_argument_number_++));
    auto& arg_attributes = arg.attributes();
    os << " {";
    pir::detail::PrintInterleave(
        arg_attributes.begin(),
        arg_attributes.end(),
        [this](std::pair<std::string, Attribute> it) {
          this->os << it.first;
          this->os << ":";
          this->PrintAttribute(it.second);
        },
        [this]() { this->os << ","; });

    os << "}";
  }
}

void IrPrinter::PrintOpResult(const Operation& op) {
  os << "(";
  auto num_op_result = op.num_results();
  std::vector<Value> op_results;
  op_results.reserve(num_op_result);
  for (size_t idx = 0; idx < num_op_result; idx++) {
    op_results.push_back(op.result(idx));
  }
  pir::detail::PrintInterleave(
      op_results.begin(),
      op_results.end(),
      [this](Value v) { this->PrintValue(v); },
      [this]() { this->os << ", "; });
  os << ")";
}

std::vector<std::string> ParseFlagsToDisable(std::string Flag) {
  std::vector<std::string> attrs_to_disable;
  std::istringstream iss(Flag);
  std::string attr;
  while (std::getline(iss, attr, ';')) {
    attrs_to_disable.push_back(attr);
  }
  return attrs_to_disable;
}

void IrPrinter::PrintAttributeMap(const Operation& op) {
  AttributeMap attributes = op.attributes();
  std::map<std::string, Attribute, std::less<>> order_attributes(
      attributes.begin(), attributes.end());

  // Filter out the callstack attribute
  order_attributes.erase("op_callstack");
  order_attributes.erase("sym_shape_str");

  if (!FLAGS_disable_logging_op_attr_list.empty()) {
    std::vector<std::string> attrs_to_disable =
        ParseFlagsToDisable(FLAGS_disable_logging_op_attr_list);
    // Remove the specified attributes if they exist in order_attributes
    for (const auto& attr : attrs_to_disable) {
      if (order_attributes.count(attr) > 0) {
        order_attributes.erase(attr);
      }
    }
  }

  os << " {";

  pir::detail::PrintInterleave(
      order_attributes.begin(),
      order_attributes.end(),
      [this](std::pair<std::string, Attribute> it) {
        this->os << it.first;
        this->os << ":";
        this->PrintAttribute(it.second);
      },
      [this]() { this->os << ","; });

  os << "}";
}

void IrPrinter::PrintOpName(const Operation& op) {
  os << "\"" << op.name() << "\"";
}
void IrPrinter::PrintOpId(const Operation& op) {
  if (VLOG_IS_ON(1) || FLAGS_print_ir) {
    os << " [id:" << op.id() << "]";
  }
}

void IrPrinter::PrintOpOperands(const Operation& op) {
  os << " (";
  auto num_op_operands = op.num_operands();
  std::vector<Value> op_operands;
  op_operands.reserve(num_op_operands);
  for (size_t idx = 0; idx < num_op_operands; idx++) {
    op_operands.push_back(op.operand_source(idx));
  }
  pir::detail::PrintInterleave(
      op_operands.begin(),
      op_operands.end(),
      [this](Value v) { this->PrintValue(v); },
      [this]() { this->os << ", "; });
  os << ")";
}

void IrPrinter::PrintOperandsType(const Operation& op) {
  auto num_op_operands = op.num_operands();
  std::vector<Type> op_operand_types;
  op_operand_types.reserve(num_op_operands);
  for (size_t idx = 0; idx < num_op_operands; idx++) {
    auto op_operand = op.operand(idx);
    if (op_operand) {
      op_operand_types.push_back(op_operand.type());
    } else {
      op_operand_types.emplace_back();
    }
  }
  os << " (";
  pir::detail::PrintInterleave(
      op_operand_types.begin(),
      op_operand_types.end(),
      [this](Type t) { this->PrintType(t); },
      [this]() { this->os << ", "; });
  os << ")";
}

void IrPrinter::PrintOpReturnType(const Operation& op) {
  auto num_op_result = op.num_results();
  std::vector<Type> op_result_types;
  op_result_types.reserve(num_op_result);
  for (size_t idx = 0; idx < num_op_result; idx++) {
    auto op_result = op.result(idx);
    if (op_result) {
      op_result_types.push_back(op_result.type());
    } else {
      op_result_types.emplace_back(nullptr);
    }
  }
  pir::detail::PrintInterleave(
      op_result_types.begin(),
      op_result_types.end(),
      [this](Type t) { this->PrintType(t); },
      [this]() { this->os << ", "; });
}

void IrPrinter::AddValueAlias(Value v, const std::string& alias) {
  const void* key = v.impl();
  if (aliases_.find(key) == aliases_.end()) {
    aliases_[key] = alias;
  }
}

class CustomPrinter : public IrPrinter {
 public:
  explicit CustomPrinter(std::ostream& os, const PrintHooks& hooks)
      : IrPrinter(os), hooks_(hooks) {}
  ~CustomPrinter() {}
  void PrintType(Type type) override {
    if (hooks_.type_print_hook) {
      hooks_.type_print_hook(type, *this);
    } else {
      IrPrinter::PrintType(type);
    }
  }

  void PrintAttribute(Attribute attr) override {
    if (hooks_.attribute_print_hook) {
      hooks_.attribute_print_hook(attr, *this);
    } else {
      IrPrinter::PrintAttribute(attr);
    }
  }

  void PrintOperation(const Operation& op) override {
    if (hooks_.op_print_hook) {
      hooks_.op_print_hook(op, *this);
    } else {
      IrPrinter::PrintOperation(op);
    }
  }

  void PrintValue(Value v) override {
    if (hooks_.value_print_hook) {
      hooks_.value_print_hook(v, *this);
    } else {
      IrPrinter::PrintValue(v);
    }
  }

 private:
  const PrintHooks hooks_;
};

std::ostream& operator<<(std::ostream& os, const CustomPrintHelper& p) {
  CustomPrinter printer(os, p.hooks_);
  printer.PrintProgram(&p.prog_);
  return os;
}

void Program::Print(std::ostream& os) const {
  IrPrinter printer(os);
  printer.PrintProgram(this);
}

void Operation::Print(std::ostream& os) const {
  IrPrinter printer(os);
  printer.PrintOperation(*this);
}

void Value::Print(std::ostream& os) const {
  IrPrinter printer(os);
  printer.PrintValue(*this);
}

void Type::Print(std::ostream& os) const {
  BasicIrPrinter printer(os);
  printer.PrintType(*this);
}

void Attribute::Print(std::ostream& os) const {
  BasicIrPrinter printer(os);
  printer.PrintAttribute(*this);
}

std::ostream& operator<<(std::ostream& os, Type type) {
  type.Print(os);
  return os;
}

std::ostream& operator<<(std::ostream& os, Attribute attr) {
  attr.Print(os);
  return os;
}

std::ostream& operator<<(std::ostream& os, const Block& block) {
  IrPrinter printer(os);
  printer.PrintBlock(block);
  return os;
}

std::ostream& operator<<(std::ostream& os, const Program& prog) {
  prog.Print(os);
  return os;
}

}  // namespace pir
