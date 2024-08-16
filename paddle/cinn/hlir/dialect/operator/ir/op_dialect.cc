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

#include "paddle/cinn/hlir/dialect/operator/ir/op_dialect.h"
// NOTE(chenxi67): File cinn_op.h is generated by op_gen.py, see details in
// paddle/cinn/hlir/dialect/CMakeLists.txt.
#include "paddle/cinn/hlir/dialect/operator/ir/cinn_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/op_attribute.h"

namespace cinn {
namespace dialect {

OperatorDialect::OperatorDialect(::pir::IrContext *context)
    : ::pir::Dialect(name(),
                     context,
                     ::pir::TypeId::get<cinn::dialect::OperatorDialect>()) {
  this->initialize();
}

void OperatorDialect::initialize() {
  // NOTE(chenxi67): GET_OP_LIST is defined in cinn_op.h which is
  // generated by op_gen.py, see details in
  // paddle/cinn/hlir/dialect/CMakeLists.txt.

  // NOTE(cocoshe): VS2017 has a limit on the length of template
  // parameters, which causes "fatal error C1202".
  // Split GET_OP_LIST into two part on WIN32 here.
#ifdef WIN32
  RegisterOps<
#define GET_OP_LIST1
#include "paddle/cinn/hlir/dialect/operator/ir/cinn_op_info.cc"  // NOLINT
      >();
  RegisterOps<
#define GET_OP_LIST2
#include "paddle/cinn/hlir/dialect/operator/ir/cinn_op_info.cc"  // NOLINT
      >();
#else
  RegisterOps<
#define GET_OP_LIST
#include "paddle/cinn/hlir/dialect/operator/ir/cinn_op_info.cc"  // NOLINT
      >();
#endif
  RegisterOp<GroupOp>();
  RegisterOp<FusionOp>();
  RegisterOp<ConcatOp>();
  RegisterOp<SplitOp>();
  RegisterOp<YieldStoreOp>();
  RegisterOp<GenerateShapeOp>();
  RegisterAttribute<GroupInfoAttribute>();
  RegisterAttribute<CINNKernelInfoAttribute>();
  RegisterAttribute<FusionTrackerPtrAttribute>();
}

void OperatorDialect::PrintType(pir::Type type, std::ostream &os) const {}

void OperatorDialect::PrintAttribute(pir::Attribute attr,
                                     std::ostream &os) const {
  if (attr.isa<GroupInfoAttribute>()) {
    os << "(" << attr.dialect().name();
    os << '.';
    if (auto group_info_attr = attr.dyn_cast<GroupInfoAttribute>()) {
      const GroupInfo &data = group_info_attr.data();
      os << "GroupInfo)"
         << "[" << data.fn_name << "]";
    }
    { os << "<#AttrNotImplemented>"; }
  } else if (attr.isa<CINNKernelInfoAttribute>()) {
    auto cinn_kernel_info = attr.dyn_cast<CINNKernelInfoAttribute>();

    os << "(" << cinn_kernel_info.data().fn_ptr;
    os << ')';
  } else if (attr.isa<FusionTrackerPtrAttribute>()) {
    auto tracker = attr.dyn_cast<FusionTrackerPtrAttribute>();
    os << "(" << tracker;
    os << ')';
  } else {
    PADDLE_THROW(::common::errors::Unimplemented(
        "cinn dialect only support GroupInfo and CINNKernelInfo"));
  }
}

pir::OpPrintFn OperatorDialect::PrintOperation(pir::Operation *op) const {
  if (auto group_op = op->dyn_cast<GroupOp>()) {
    return [](pir::Operation *op, pir::IrPrinter &printer) {
      auto group_op = op->dyn_cast<GroupOp>();
      group_op.Print(printer);
    };
  } else if (auto fusion_op = op->dyn_cast<FusionOp>()) {
    return [](pir::Operation *op, pir::IrPrinter &printer) {
      auto fusion_op = op->dyn_cast<FusionOp>();
      fusion_op.Print(printer);
    };
  }
  return nullptr;
}

}  // namespace dialect
}  // namespace cinn

IR_DEFINE_EXPLICIT_TYPE_ID(cinn::dialect::OperatorDialect)
