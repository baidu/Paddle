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

#pragma once

#include <ostream>
#include <string>
#include <unordered_map>

#include "paddle/pir/include/core/attribute.h"
#include "paddle/pir/include/core/block.h"
#include "paddle/pir/include/core/operation.h"
#include "paddle/pir/include/core/program.h"
#include "paddle/pir/include/core/region.h"
#include "paddle/pir/include/core/type.h"
#include "paddle/pir/include/core/value.h"

namespace pir {

class BasicIrPrinter {
 public:
  explicit BasicIrPrinter(std::ostream& os) : os(os), id_(GenerateId()) {}

  virtual void PrintType(Type type);

  virtual void PrintAttribute(Attribute attr);
  uint64_t id() const { return id_; }

 public:
  std::ostream& os;

 private:
  static uint64_t GenerateId() {
    static std::atomic<std::uint64_t> uid{0};
    return ++uid;
  }
  const uint64_t id_ = -1;
};

class IR_API IrPrinter : public BasicIrPrinter {
 public:
  explicit IrPrinter(std::ostream& os) : BasicIrPrinter(os) {}

  /// @brief print program
  /// @param program
  void PrintProgram(const Program* program);

  /// @brief dispatch to custom printer function or PrintGeneralOperation
  virtual void PrintOperation(const Operation& op);
  /// @brief print operation itself without its regions
  void PrintOperationWithNoRegion(const Operation& op);
  /// @brief print operation and its regions
  void PrintGeneralOperation(const Operation& op);

  void PrintRegion(const Region& Region);
  void PrintBlock(const Block& block);

  virtual void PrintValue(Value v);

  void PrintOpResult(const Operation& op);

  void PrintAttributeMap(const Operation& op);

  void PrintOpName(const Operation& op);

  void PrintOpId(const Operation& op);

  void PrintOpOperands(const Operation& op);

  void PrintOperandsType(const Operation& op);

  void PrintOpReturnType(const Operation& op);

  void AddValueAlias(Value value, const std::string& alias);

  void AddIndentation();
  void DecreaseIndentation();
  const std::string& indentation() const { return cur_indentation_; }

 private:
  size_t cur_result_number_{0};
  size_t cur_block_argument_number_{0};
  std::string cur_indentation_;
  std::unordered_map<const void*, std::string> aliases_;
};

using ValuePrintHook =
    std::function<void(Value value, IrPrinter& printer)>;  // NOLINT
using TypePrintHook =
    std::function<void(Type type, IrPrinter& printer)>;  // NOLINT
using AttributePrintHook =
    std::function<void(Attribute attr, IrPrinter& printer)>;  // NOLINT
using OpPrintHook =
    std::function<void(const Operation& op, IrPrinter& printer)>;  // NOLINT

struct IR_API PrintHooks {
  ValuePrintHook value_print_hook{nullptr};
  TypePrintHook type_print_hook{nullptr};
  AttributePrintHook attribute_print_hook{nullptr};
  OpPrintHook op_print_hook{nullptr};
};

class IR_API CustomPrintHelper {
 public:
  explicit CustomPrintHelper(const Program& program, const PrintHooks& hooks)
      : hooks_(hooks), prog_(program) {}
  friend IR_API std::ostream& operator<<(std::ostream& os,
                                         const CustomPrintHelper& p);

 private:
  const PrintHooks& hooks_;
  const Program& prog_;
};

IR_API std::ostream& operator<<(std::ostream& os, const CustomPrintHelper& p);

}  // namespace pir
