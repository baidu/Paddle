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

#include "paddle/cinn/backends/codegen_c.h"

#include <fstream>
#include <string>

#include "paddle/cinn/backends/extern_func_emitter.h"
#include "paddle/cinn/backends/extern_func_emitter_builtin.h"
#include "paddle/cinn/ir/lowered_func.h"
#include "paddle/cinn/ir/op/ir_operators.h"
#include "paddle/cinn/ir/utils/ir_verify.h"
#include "paddle/cinn/optim/ir_simplify_pass.h"
#include "paddle/cinn/runtime/cpu/thread_backend.h"
#include "paddle/cinn/runtime/intrinsic.h"
#include "paddle/cinn/utils/string.h"
#include "paddle/common/enforce.h"
//! Root of the builtin code.
PD_DECLARE_string(cinn_x86_builtin_code_root);

namespace cinn {
namespace backends {
using namespace utils;  // NOLINT
using cinn::common::float16;

const char *kCKeywordRestrict = "__restrict__";

void CodeGenC::Compile(const ir::Module &module, const Outputs &outputs) {
  ir::ir_utils::IrVerify(module.As<ir::_Module_>());

  if (!outputs.c_header_name.empty()) {
    auto source = Compile(module, OutputKind::CHeader);
    str_ = "";
    std::ofstream file(outputs.c_header_name);
    PADDLE_ENFORCE_EQ(file.is_open(),
                      true,
                      ::common::errors::InvalidArgument(
                          "failed to open file %s", outputs.c_header_name));
    file << source;
    file.close();
    LOG(WARNING) << "Output C header to file " << outputs.c_header_name;
  }

  if (!outputs.c_source_name.empty()) {
    auto source = Compile(module, OutputKind::CImpl);
    str_ = "";
    std::ofstream file(outputs.c_source_name);
    PADDLE_ENFORCE_EQ(file.is_open(),
                      true,
                      ::common::errors::InvalidArgument(
                          "failed to open file %s", outputs.c_source_name));
    file << source;
    file.close();
    LOG(WARNING) << "Output C source to file " << outputs.c_source_name;
  }
}

CodeGenC::CodeGenC(Target target) : ir::IrPrinter(ss_) {}

std::string CodeGenC::Compile(const ir::Module &module,
                              OutputKind output_kind) {
  if (output_kind == OutputKind::CHeader) {
    GenerateHeaderFile(module);
  } else if (output_kind == OutputKind::CImpl) {
    PrintIncludes();

    if (inline_builtin_codes_) PrintBuiltinCodes();

    for (auto &func : module.functions()) {
      Compile(func);
    }
  } else {
    PADDLE_THROW(::common::errors::Unimplemented("Not supported OutputKind"));
  }
  return str_;
}

// TODO(LiuYang): Here the Ret type seems unuseful
void CodeGenC::Compile(const ir::LoweredFunc &function) {
  PADDLE_ENFORCE_EQ(
      function.defined(),
      true,
      ::common::errors::InvalidArgument("The function is not defined."));
  IrPrinter::Visit(function);
  str_ += "\n\n";
}

std::string CodeGenC::GetTypeName(Type type) {
  // common scalar type
#define GET_SCALAR_TYPE(pred_expr, scalar_name) \
  if (pred_expr) {                              \
    return scalar_name;                         \
  }

  GET_SCALAR_TYPE(type.is_void(), "void");
  GET_SCALAR_TYPE(type.is_bool(), "bool");

  GET_SCALAR_TYPE(type.is_int(8), "int8_t");
  GET_SCALAR_TYPE(type.is_int(16), "int16_t");
  GET_SCALAR_TYPE(type.is_int(32), "int32_t");
  GET_SCALAR_TYPE(type.is_int(64), "int64_t");

  GET_SCALAR_TYPE(type.is_uint(8), "uint8_t");
  GET_SCALAR_TYPE(type.is_uint(16), "uint16_t");
  GET_SCALAR_TYPE(type.is_uint(32), "uint32_t");
  GET_SCALAR_TYPE(type.is_uint(64), "uint64_t");

  GET_SCALAR_TYPE(type.is_bfloat16(), "bfloat16");
  GET_SCALAR_TYPE(type.is_float16(), "float16");
  GET_SCALAR_TYPE(type.is_float(32), "float")
  GET_SCALAR_TYPE(type.is_float(64), "double")
#undef GET_SCALAR_TYPE

  // customized_type
  if (type.is_customized_type()) {
    PADDLE_ENFORCE_EQ(type.customized_type().empty(),
                      false,
                      ::common::errors::InvalidArgument(
                          "customized_type is empty. It can't be empty."));
    auto customized_name = type.customized_type();
    // get name of a cuda built-in vector type, it is started with a
    // 'CudaVectorType::' prefix
    if (utils::StartsWith(
            customized_name,
            cinn::common::customized_type::kcuda_builtin_vector_t)) {
      customized_name.erase(
          0, strlen(cinn::common::customized_type::kcuda_builtin_vector_t));
    }
    return customized_name;
  }

  // other types are not implemented yet
  CINN_NOT_IMPLEMENTED
  return "";
}

std::string CodeGenC::GetTypeRepr(Type type) {
  std::string str;
  if (type.is_cpp_const()) {
    str = "const ";
  }

  str += GetTypeName(type);
  if (type.is_cpp_handle()) {
    str += "*";
  } else if (type.is_cpp_handle2()) {
    str += "**";
  }
  return str;
}
void CodeGenC::Visit(const ir::IntImm *op) { IrPrinter::Visit(op); }
void CodeGenC::Visit(const ir::UIntImm *op) { IrPrinter::Visit(op); }
void CodeGenC::Visit(const ir::FloatImm *op) { IrPrinter::Visit(op); }
void CodeGenC::Visit(const ir::StringImm *op) { IrPrinter::Visit(op); }
void CodeGenC::Visit(const ir::Add *op) { IrPrinter::Visit(op); }
void CodeGenC::Visit(const ir::Sub *op) { IrPrinter::Visit(op); }
void CodeGenC::Visit(const ir::Mul *op) { IrPrinter::Visit(op); }
void CodeGenC::Visit(const ir::Div *op) { IrPrinter::Visit(op); }
void CodeGenC::Visit(const ir::Mod *op) {
  auto copied = op->b();
  optim::Simplify(&copied, optim::SimplifyType::kExpr);
  if (copied.is_constant()) {
    int temp = static_cast<int>(copied.get_constant());
    if ((temp & (temp - 1)) == 0) {
      str_ += "(";
      IrPrinter::Visit(op->a());
      str_ += " & ";
      str_ += std::to_string(temp - 1);
      str_ += ")";
      return;
    }
  }
  PrintBinaryOp("%", op);
}
void CodeGenC::Visit(const ir::EQ *op) { IrPrinter::Visit(op); }
void CodeGenC::Visit(const ir::NE *op) { IrPrinter::Visit(op); }
void CodeGenC::Visit(const ir::LT *op) { IrPrinter::Visit(op); }
void CodeGenC::Visit(const ir::LE *op) { IrPrinter::Visit(op); }
void CodeGenC::Visit(const ir::GT *op) { IrPrinter::Visit(op); }
void CodeGenC::Visit(const ir::GE *op) { IrPrinter::Visit(op); }
void CodeGenC::Visit(const ir::And *op) { PrintBinaryOp("&&", op); }
void CodeGenC::Visit(const ir::Or *op) { PrintBinaryOp("||", op); }
void CodeGenC::Visit(const ir::Min *op) { IrPrinter::Visit(op); }
void CodeGenC::Visit(const ir::Max *op) { IrPrinter::Visit(op); }
void CodeGenC::Visit(const ir::Minus *op) { IrPrinter::Visit(op); }
void CodeGenC::Visit(const ir::Not *op) {
  str_ += "(!";
  IrPrinter::Visit(op->v());
  str_ += ")";
}
void CodeGenC::Visit(const ir::Cast *op) { PrintCastExpr(op->type(), op->v()); }
void CodeGenC::Visit(const ir::For *op) {
  Expr extent = op->extent;
  Expr min = op->min;
  int num_task = 1;
  if (op->is_parallel()) {
    str_ += "int num_task = max_concurrency();\n";
    DoIndent();
    str_ += "omp_set_num_threads(num_task);\n";
    DoIndent();
    str_ += "auto flambda = [=](int task_id, int num_task) -> int {\n";
    IncIndent();
    DoIndent();
    str_ += "int n_per_task = ";
    Expr num_task_var = Var("num_task");
    IrPrinter::Visit((op->extent + num_task_var - 1) / num_task_var);
    str_ += ";\n";
    PADDLE_ENFORCE_EQ(min.as_int32(),
                      0,
                      ::common::errors::InvalidArgument(
                          "The min of the for loop should be 0"));
    auto task_id = Var("task_id");
    auto n_per_task = Var("n_per_task");
    min = task_id * n_per_task;
    extent = (task_id + 1) * n_per_task;
    DoIndent();
  }
  str_ += "for (";
  str_ += GetTypeRepr(Int(32));
  str_ += " ";
  str_ += op->loop_var->name;
  str_ += " = ";
  IrPrinter::Visit(min);
  str_ += "; ";
  str_ += op->loop_var->name;
  str_ += " < ";
  IrPrinter::Visit(op->extent);
  if (op->is_parallel()) {
    str_ += " && ";
    str_ += op->loop_var->name;
    str_ += " < ";
    IrPrinter::Visit(extent);
  }
  str_ += "; ";

  str_ += op->loop_var->name;
  str_ += " += 1";
  str_ += ") ";

  IrPrinter::Visit(op->body);
  if (op->is_parallel()) {
    str_ += "\n";
    DoIndent();
    str_ += "return 0;\n";
    DecIndent();
    DoIndent();
    str_ += "};\n";
    str_ += "#pragma omp parallel num_threads(num_task)\n";
    DoIndent();
    str_ += "{\n";
    IncIndent();
    DoIndent();
    str_ += "int task_id = omp_get_thread_num();\n";
    DoIndent();
    str_ += "flambda(task_id, num_task);\n";
    DecIndent();
    DoIndent();
    str_ += "}";
  }
}
void CodeGenC::Visit(const ir::PolyFor *op) {
  str_ += "for (";
  str_ += GetTypeRepr(Int(32));
  str_ += " ";
  str_ += op->iterator->name;
  str_ += " = ";
  IrPrinter::Visit(op->init);
  str_ += "; ";
  IrPrinter::Visit(op->condition);
  str_ += "; ";

  str_ += op->iterator->name;
  str_ += " += ";
  IrPrinter::Visit(op->inc);
  str_ += ") ";

  IrPrinter::Visit(op->body);
}
void CodeGenC::Visit(const ir::Select *op) {
  str_ += "(";
  str_ += "(";
  IrPrinter::Visit(op->condition);
  str_ += ") ? ";
  IrPrinter::Visit(op->true_value);
  str_ += " : ";
  IrPrinter::Visit(op->false_value);
  str_ += ")";
}
void CodeGenC::Visit(const ir::IfThenElse *op) {
  str_ += "if (";
  IrPrinter::Visit(op->condition);
  str_ += ") ";

  IrPrinter::Visit(op->true_case);

  if (op->false_case.defined()) {
    str_ += " else ";
    IrPrinter::Visit(op->false_case);
  }
}
void CodeGenC::Visit(const ir::Block *op) {
  str_ += "{\n";

  IncIndent();

  // Note: size_t (0 - 1) = 18446744073709551615
  if (op->stmts.size() >= 1) {
    for (int i = 0; i < op->stmts.size() - 1; i++) {
      DoIndent();
      IrPrinter::Visit(op->stmts[i]);
      str_ += ";\n";
    }
    DoIndent();
    IrPrinter::Visit(op->stmts.back());
    str_ += ";";
  }

  DecIndent();
  str_ += "\n";
  DoIndent();
  str_ += "}";
}
void CodeGenC::Visit(const ir::Call *op) {
  if (op->name == runtime::intrinsic::buffer_malloc) {
    PrintCall_buffer_malloc(op);
  } else if (op->name == runtime::intrinsic::pod_values_to_array_repr) {
    PrintCall_pod_values_to_array(op);
  } else if (op->is_intrinsic_call()) {
    str_ += op->name;
    str_ += "(";
    PrintCallArgs(op);
    str_ += ")";
  } else if (op->is_cinn_call()) {  // call CINN LoweredFunc
    str_ += op->name;
    str_ += "(";
    PrintCallArgs(op);
    str_ += ")";
  } else if (op->is_extern_call()) {
    const auto &fn_name = ExternFunctionEmitterRegistry::Global().Lookup(
        ExternFuncID{backend_C, op->name.c_str()});
    if (!fn_name.empty()) {
      ExternFunctionLLVMEmitter emitter(fn_name);
      emitter.BindCodeGen(this);
      emitter.Emit(op);
    } else {
      if (op->read_args.empty())
        PADDLE_ENFORCE_EQ(
            op->read_args.empty(),
            false,
            ::common::errors::InvalidArgument(
                "Either read_args or write_args must not be empty."));
      str_ += op->name;
      str_ += "(";
      PrintCallArgs(op);
      str_ += ")";
    }
  } else {
    CINN_NOT_IMPLEMENTED
  }
}
void CodeGenC::PrintCallArgs(const ir::Call *op) {
  if (!op->read_args.empty()) {
    for (int i = 0; i < op->read_args.size() - 1; i++) {
      IrPrinter::Visit(op->read_args[i]);
      str_ += ", ";
    }
    IrPrinter::Visit(op->read_args.back());
  }
  if (!op->write_args.empty()) {
    if (!op->read_args.empty()) str_ += ", ";

    for (int i = 0; i < op->write_args.size() - 1; i++) {
      IrPrinter::Visit(op->write_args[i]);
      str_ += ", ";
    }
    IrPrinter::Visit(op->write_args.back());
  }
}

void CodeGenC::PrintCall_buffer_malloc(const ir::Call *op) {
  PADDLE_ENFORCE_EQ(
      op->read_args.size(),
      2UL,
      ::common::errors::InvalidArgument("The number of read_args should be 2"));
  str_ += op->name;
  str_ += "(";
  PrintCastExpr("void*", op->read_args[0]);
  str_ += ", ";
  IrPrinter::Visit(op->read_args[1]);
  str_ += ")";
}

void CodeGenC::PrintCall_cinn_pod_value_to_(const ir::Call *op) {
  PADDLE_ENFORCE_EQ(
      op->read_args.size(),
      1UL,
      ::common::errors::InvalidArgument("The number of read_args should be 1"));
  str_ += op->name;
  str_ += "(";
  str_ += "&(";
  IrPrinter::Visit(op->read_args[0]);
  str_ += ")";
  str_ += ")";
}

void CodeGenC::PrintCall_get_address(const ir::Call *op) {
  PADDLE_ENFORCE_EQ(
      op->read_args.size(),
      1UL,
      ::common::errors::InvalidArgument("The number of read_args should be 1"));
  PADDLE_ENFORCE_EQ(op->write_args.empty(),
                    true,
                    ::common::errors::InvalidArgument(
                        "write_args is not empty. It must be empty."));
  auto *read_var = op->read_args.front().as_var();
  auto *read_buf = op->read_args.front().as_buffer();
  if (!read_var)
    PADDLE_ENFORCE_NOT_NULL(read_buf,
                            ::common::errors::InvalidArgument(
                                "Only Var or Buffer can get address"));

  if (read_var) {
    if (read_var->type().lanes() <= 1) str_ += "&";
    str_ += read_var->name;
  } else if (read_buf) {
    if (read_buf->type().lanes() <= 1) str_ += "&";
    str_ += read_buf->name;
  } else {
    CINN_NOT_IMPLEMENTED
  }
}

void CodeGenC::PrintCall_pod_values_to_array(const ir::Call *op) {
  PADDLE_ENFORCE_EQ(op->read_args.empty(),
                    false,
                    ::common::errors::InvalidArgument(
                        "The read_args is empty. It should not be empty."));
  PADDLE_ENFORCE_EQ(op->write_args.size(),
                    1UL,
                    ::common::errors::InvalidArgument(
                        "The number of write_args should be 1"));
  auto output_var = op->write_args.front().as_var_ref();
  PADDLE_ENFORCE_EQ(
      output_var.defined(),
      true,
      ::common::errors::InvalidArgument(
          "The variable 'output_var' is not be defined. It must be defined."));

  std::vector<std::string> arg_names;
  for (auto &arg : op->read_args) {
    auto arg_var = arg.as_var();
    PADDLE_ENFORCE_NOT_NULL(
        arg_var,
        ::common::errors::InvalidArgument(
            "The 'arg_var' is an invalid argument. It must be true."));
    arg_names.push_back(arg_var->name);
  }

  str_ += "cinn_pod_value_t ";
  str_ += output_var->name;
  str_ += "[] = ";
  str_ += "{ ";

  str_ += utils::Join(arg_names, ", ");

  str_ += " }";
}

void CodeGenC::Visit(const ir::_Module_ *op) { CINN_NOT_IMPLEMENTED }
void CodeGenC::Visit(const ir::_Var_ *op) { str_ += op->name; }

void CodeGenC::Visit(const ir::Load *op) {
  ir::Expr offset = [&] {
    if (load_to_offset_.count(op) == 0) {
      load_to_offset_[op] = op->index();
    }
    return load_to_offset_.at(op);
  }();

  Expr dense_strided_ramp = detail::StridedRampBase(offset, 1);
  if (dense_strided_ramp.defined()) {  // Loading a continuous Ramp address.
    PADDLE_ENFORCE_EQ(
        op->type().is_vector(),
        true,
        ::common::errors::InvalidArgument(
            "The operation type is not a vector. It must be a vector."));
    PrintStackVecType(op->type().ElementOf(), offset.type().lanes());
    str_ += "::";
    str_ += "Load(";
    str_ += op->tensor.As<ir::_Tensor_>()->name;
    str_ += ",";
    IrPrinter::Visit(dense_strided_ramp);
    str_ += ")";
  } else if (offset.type().is_vector()) {
    // gather
    PADDLE_ENFORCE_EQ(
        op->type().is_vector(),
        true,
        ::common::errors::InvalidArgument(
            "The operation type is not a vector. It must be a vector."));
    PrintStackVecType(op->type().ElementOf(), offset.type().lanes());
    str_ += "::Load(";
    str_ += op->tensor.As<ir::_Tensor_>()->name;
    str_ += ",";
    IrPrinter::Visit(offset);
    str_ += ")";
  } else if (op->is_addr_tensor()) {
    auto *tensor = op->tensor.As<ir::_Tensor_>();
    str_ += tensor->name;
    str_ += "[";
    IrPrinter::Visit(offset);
    str_ += "]";
  } else {
    IrPrinter::Visit(op);
  }
}

void CodeGenC::Visit(const ir::Store *op) {
  PADDLE_ENFORCE_EQ(
      op->is_addr_tensor(),
      true,
      ::common::errors::InvalidArgument(
          "The operation type is invalid. It must be an address tensor."));
  ir::Expr offset = [&] {
    if (store_to_offset_.count(op) == 0) {
      store_to_offset_[op] = op->index();
    }
    return store_to_offset_.at(op);
  }();
  auto *tensor = op->tensor.As<ir::_Tensor_>();
  PADDLE_ENFORCE_NOT_NULL(tensor,
                          ::common::errors::InvalidArgument(
                              "The tensor is null. It must not be null."));
  str_ += tensor->name;
  str_ += "[";
  IrPrinter::Visit(offset);
  str_ += "]";
  str_ += " = ";
  IrPrinter::Visit(op->value);
}
void CodeGenC::Visit(const ir::Alloc *op) {
  str_ += runtime::intrinsic::buffer_malloc;
  str_ += "(";
  str_ += "(void*)(0), ";

  auto *buffer = op->destination.As<ir::_Buffer_>();
  str_ += buffer->name;
  str_ += ")";
}

void CodeGenC::Visit(const ir::Free *op) {
  str_ += runtime::intrinsic::buffer_free;
  str_ += "(";
  str_ += "(void*)(0), ";

  auto *buffer = op->destination.As<ir::_Buffer_>();
  str_ += buffer->name;
  str_ += ")";
}

void CodeGenC::Visit(const ir::_Buffer_ *op) { str_ += op->name; }
void CodeGenC::Visit(const ir::_Tensor_ *op) { str_ += op->buffer->name; }
void CodeGenC::Visit(const ir::Let *op) {
  bool is_vec = false;
  PADDLE_ENFORCE_EQ(op->type().valid(),
                    true,
                    ::common::errors::InvalidArgument(
                        "The operation type is invalid. It must be valid."));
  if (op->body.defined() && op->body.As<ir::Broadcast>()) {
    // broadcast's type is hard to print, so use c++11 auto instead.
    str_ += "auto";
    is_vec = true;
  } else {
    str_ += GetTypeRepr(op->type());
  }

  str_ += " ";
  IrPrinter::Visit(op->symbol);

  // native C array.
  if (op->type().lanes() > 1 && !is_vec) {
    str_ += "[";
    str_ += std::to_string(op->type().lanes());
    str_ += "]";
  }

  if (op->body.defined()) {
    str_ += " = ";
    IrPrinter::Visit(op->body);
  }
}

void CodeGenC::Visit(const ir::Reduce *op) {
  PADDLE_THROW(::common::errors::InvalidArgument(
      "Reduce IR is just for internal representation, should not be "
      "used for CodeGen."));
}

void CodeGenC::Visit(const ir::Ramp *op) {
  str_ += "StackVec<";
  str_ += std::to_string(op->lanes);
  str_ += ",";
  str_ += GetTypeRepr(op->type().ElementOf());
  str_ += ">::Ramp(";
  IrPrinter::Visit(op->base);
  str_ += ", ";
  IrPrinter::Visit(op->stride);
  str_ += ", ";
  str_ += std::to_string(op->lanes);
  str_ += ")";
}

void CodeGenC::Visit(const ir::Broadcast *op) {
  str_ += "StackVec<";
  str_ += std::to_string(op->lanes);
  str_ += ",";
  str_ += GetTypeRepr(op->type().ElementOf());
  str_ += ">::Broadcast(";
  IrPrinter::Visit(op->value);
  str_ += ", ";
  str_ += std::to_string(op->lanes);
  str_ += ")";
}

void CodeGenC::Visit(const ir::FracOp *op) { ir::IrPrinter::Visit(op); }
void CodeGenC::Visit(const ir::Sum *op) { ir::IrPrinter::Visit(op); }
void CodeGenC::Visit(const ir::Product *op) { ir::IrPrinter::Visit(op); }

void CodeGenC::PrintCastExpr(const Type &type, Expr e) {
  str_ += "((";
  str_ += GetTypeRepr(type);
  str_ += ")";
  str_ += "(";
  IrPrinter::Visit(e);
  str_ += "))";
}
void CodeGenC::PrintCastExpr(const std::string &type, Expr e) {
  str_ += "(";
  str_ += type;
  str_ += ")";
  str_ += "(";
  IrPrinter::Visit(e);
  str_ += ")";
}

void CodeGenC::PrintShape(const std::vector<Expr> &shape,
                          char leftb,
                          char rightb) {
  str_ += leftb;
  str_ += " ";

  for (int i = 0; i < shape.size() - 1; i++) {
    IrPrinter::Visit(shape[i]);
    str_ += ", ";
  }
  if (shape.size() > 1) IrPrinter::Visit(shape.back());

  str_ += " ";
  str_ += rightb;
}

void CodeGenC::Visit(const ir::_LoweredFunc_ *op) {
  PrintFunctionDeclaration(op);
  str_ += "\n";

  DoIndent();

  PADDLE_ENFORCE_EQ(
      op->alloc_output_buffer_exprs.size(),
      op->dealloc_output_buffer_exprs.size(),
      ::common::errors::InvalidArgument(
          "The count of allocation and deallocation expressions is not "
          "match"));

  std::vector<Expr> new_body;

  std::vector<Expr> create_temp_buffers = op->PrepareCreateTempBufferExprs();
  std::vector<Expr> alloca_temp_buffers = op->PrepareAllocTempBufferExprs();
  std::vector<Expr> dealloca_temp_buffers = op->PrepareDeallocTempBufferExprs();
#define APPEND_TO_NEW_BODY(field__) \
  new_body.insert(                  \
      std::end(new_body), std::begin(op->field__), std::end(op->field__));
  APPEND_TO_NEW_BODY(argument_prepare_exprs)
  new_body.insert(std::end(new_body),
                  std::begin(create_temp_buffers),
                  std::end(create_temp_buffers));
  APPEND_TO_NEW_BODY(alloc_output_buffer_exprs)
  new_body.insert(std::end(new_body),
                  std::begin(alloca_temp_buffers),
                  std::end(alloca_temp_buffers));
  APPEND_TO_NEW_BODY(buffer_data_cast_exprs)
  new_body.push_back(op->body);
  new_body.insert(std::end(new_body),
                  std::begin(dealloca_temp_buffers),
                  std::end(dealloca_temp_buffers));
  APPEND_TO_NEW_BODY(dealloc_output_buffer_exprs)

  Expr func_body = ir::Block::Make(new_body);

  IrPrinter::Visit(func_body);
}
void CodeGenC::PrintIncludes() {
  str_ += "#include <cinn_runtime.h>\n";
  str_ += "#include <stdio.h>\n";
  str_ += "\n";
}

void CodeGenC::PrintFileGuardOpen(const std::string &name) {
  str_ += utils::StringFormat("#ifndef _%s_CINN_H_\n", Uppercase(name).c_str());
  str_ += utils::StringFormat("#define _%s_CINN_H_\n", Uppercase(name).c_str());
  str_ += "\n";
}
void CodeGenC::PrintFileGuardClose(const std::string &module_name) {
  str_ += utils::StringFormat("#endif  // _%s_CINN_H_\n",
                              Uppercase(module_name).c_str());
}

void CodeGenC::PrintBufferCreation(const std::vector<ir::Buffer> &buffers) {
  for (auto &buffer : buffers) {
    // Ignore the buffer in other devices.
    if (!buffer->is_on_host()) continue;
    DoIndent();
    auto buffer_ptr_type =
        Type()
            .set_customized_type(cinn::common::customized_type::kbuffer_t)
            .set_cpp_handle();
    Var variable = ir::_Var_::Make(buffer->name, buffer_ptr_type);
    auto expr = ir::intrinsics::BufferCreate::Make(buffer);
    expr = ir::Let::Make(variable, expr);
    IrPrinter::Visit(expr);
    str_ += ";\n";
  }
}

void CodeGenC::PrintBufferDestroy(const std::vector<ir::Buffer> &buffers) {
  for (auto &buffer : buffers) {
    DoIndent();
    IrPrinter::Visit(buffer.DestroyExpr());
    str_ += ";\n";
  }
}

void CodeGenC::GenerateHeaderFile(const ir::Module &module) {
  PrintFileGuardOpen(module.name());
  PrintIncludes();

  for (auto &func : module.functions()) {
    PrintFunctionDeclaration(func.As<ir::_LoweredFunc_>());
    str_ += ";\n";
    str_ += "\n\n";
  }

  PrintFileGuardClose(module.name());
}

void CodeGenC::PrintFuncArg(const ir::Argument &arg) {
  if (arg.is_buffer()) {
    if (arg.is_input()) {
      str_ += "const struct cinn_buffer_t *";
    } else {
      str_ += "struct cinn_buffer_t *";
    }
  } else if (arg.is_var()) {
    str_ += GetTypeRepr(arg.type());
    str_ += " ";
    str_ += arg.name();
  } else {
    CINN_NOT_IMPLEMENTED
  }
  str_ += arg.name();
}

void CodeGenC::PrintRuntimeType(const cinn_type_t &type) {
  if (type == cinn_bool_t()) {
    str_ += "cinn_bool_t()";
  } else if (type == cinn_int8_t()) {
    str_ += "cinn_int8_t()";
  } else if (type == cinn_int16_t()) {
    str_ += "cinn_int16_t()";
  } else if (type == cinn_int32_t()) {
    str_ += "cinn_int32_t()";
  } else if (type == cinn_int64_t()) {
    str_ += "cinn_int64_t()";
  } else if (type == cinn_uint8_t()) {
    str_ += "cinn_uint8_t()";
  } else if (type == cinn_uint16_t()) {
    str_ += "cinn_uint16_t()";
  } else if (type == cinn_uint32_t()) {
    str_ += "cinn_uint32_t()";
  } else if (type == cinn_uint64_t()) {
    str_ += "cinn_uint64_t()";
  } else if (type == cinn_bfloat16_t()) {
    str_ += "cinn_bfloat16_t()";
  } else if (type == cinn_float16_t()) {
    str_ += "cinn_float16_t()";
  } else if (type == cinn_float32_t()) {
    str_ += "cinn_float32_t()";
  } else if (type == cinn_float64_t()) {
    str_ += "cinn_float64_t()";
  } else {
    PADDLE_THROW(::common::errors::InvalidArgument(
        "Unknown type is not supported to print"));
  }
}

void CodeGenC::PrintStackVecType(Type type, int lanes) {
  str_ += "StackedVec<";
  str_ += GetTypeRepr(type);
  str_ += ",";
  str_ += std::to_string(lanes);
  str_ += ">";
}

void CodeGenC::Visit(const ir::PrimitiveNode *op) { CINN_NOT_IMPLEMENTED }
void CodeGenC::Visit(const ir::_BufferRange_ *op) { CINN_NOT_IMPLEMENTED }
void CodeGenC::Visit(const ir::ScheduleBlock *op) { CINN_NOT_IMPLEMENTED }
void CodeGenC::Visit(const ir::ScheduleBlockRealize *op) {
  CINN_NOT_IMPLEMENTED
}
void CodeGenC::Visit(const ir::_Dim_ *op) { CINN_NOT_IMPLEMENTED }

void CodeGenC::Visit(const ir::IntrinsicOp *op) {
  switch (op->getKind()) {
#define __(x)                                     \
  case ir::IntrinsicKind::k##x:                   \
    Visit(llvm::dyn_cast<ir::intrinsics::x>(op)); \
    break;

    INTRINSIC_KIND_FOR_EACH(__)
#undef __
  }
}

void CodeGenC::Visit(const ir::intrinsics::BufferGetDataHandle *op) {
  str_ += op->buffer.as_buffer()->name;
  str_ += "->";
  str_ += "memory";
}

void CodeGenC::Visit(const ir::intrinsics::BufferGetDataConstHandle *op) {
  str_ += op->buffer.as_buffer()->name;
  str_ += "->";
  str_ += "memory";
}

void CodeGenC::Visit(const ir::intrinsics::PodValueToX *op) {
  auto to_type = op->GetOutputType(0);
  if (to_type == type_of<float>()) {
    str_ += runtime::intrinsic::pod_value_to_float;
  } else if (to_type == type_of<double>()) {
    str_ += runtime::intrinsic::pod_value_to_double;
  } else if (to_type == type_of<float16>()) {
    str_ += runtime::intrinsic::pod_value_to_float16;
  } else if (to_type == type_of<bool>()) {
    str_ += runtime::intrinsic::pod_value_to_bool;
  } else if (to_type == type_of<int8_t>()) {
    str_ += runtime::intrinsic::pod_value_to_int8;
  } else if (to_type == type_of<int16_t>()) {
    str_ += runtime::intrinsic::pod_value_to_int16;
  } else if (to_type == type_of<int32_t>()) {
    str_ += runtime::intrinsic::pod_value_to_int32;
  } else if (to_type == type_of<int64_t>()) {
    str_ += runtime::intrinsic::pod_value_to_int64;
  } else if (to_type == type_of<uint8_t>()) {
    str_ += runtime::intrinsic::pod_value_to_uint8;
  } else if (to_type == type_of<uint16_t>()) {
    str_ += runtime::intrinsic::pod_value_to_uint16;
  } else if (to_type == type_of<uint32_t>()) {
    str_ += runtime::intrinsic::pod_value_to_uint32;
  } else if (to_type == type_of<uint64_t>()) {
    str_ += runtime::intrinsic::pod_value_to_uint64;
  } else if (to_type == type_of<void *>()) {
    str_ += runtime::intrinsic::pod_value_to_void_p;
  } else if (to_type == type_of<cinn_buffer_t *>()) {
    str_ += runtime::intrinsic::pod_value_to_buffer_p;
  } else {
    std::stringstream ss;
    ss << "Not supported type: " << to_type;
    PADDLE_THROW(::common::errors::InvalidArgument(ss.str()));
  }

  str_ += "(";
  IrPrinter::Visit(op->pod_value_ptr);
  str_ += ")";
}

void CodeGenC::Visit(const ir::intrinsics::BufferCreate *op) {
  const ir::_Buffer_ *buffer_arg = op->buffer.as_buffer();
  PADDLE_ENFORCE_NOT_NULL(
      buffer_arg,
      ::common::errors::InvalidArgument(
          "The buffer argument is invalid. It must be true."));

  str_ += runtime::intrinsic::buffer_create;
  str_ += "(";
  PrintCastExpr("cinn_device_kind_t", Expr(buffer_arg->target.runtime_arch()));
  str_ += "/*target*/, ";
  PrintRuntimeType(runtime::ToRuntimeType(buffer_arg->dtype.ElementOf()));
  str_ += ", ";
  PrintShape(buffer_arg->shape);
  if (buffer_arg->data_alignment > 0) {
    str_ += ", ";
    str_ += std::to_string(buffer_arg->data_alignment);
    str_ += "/*align*/";
  }
  str_ += ")";
}

void CodeGenC::Visit(const ir::intrinsics::GetAddr *op) {
  if (op->data.as_buffer()) {
    str_ += "&";
    str_ += op->data.as_buffer()->name;
  } else if (op->data.as_var()) {
    str_ += "&";
    str_ += op->data.as_var()->name;
  } else {
    str_ += "&(";
    IrPrinter::Visit(op->data);
    str_ += ")";
  }
}

void CodeGenC::Visit(const ir::intrinsics::ArgsConstruct *op) {
  str_ += runtime::intrinsic::args_construct_repr;
  str_ += "(";
  str_ += op->var->name;
  str_ += ", ";
  str_ += std::to_string(op->args.size());
  str_ += ", ";
  for (int i = 0; i < op->args.size() - 1; i++) {
    IrPrinter::Visit(op->args[i]);
    str_ += ", ";
  }
  if (!op->args.empty()) {
    IrPrinter::Visit(op->args.back());
  }
  str_ += ")";
}

void CodeGenC::Visit(const ir::intrinsics::BuiltinIntrin *op) {
  str_ += op->name;
  str_ += "(";
  if (!op->args.empty()) {
    for (int i = 0; i < op->args.size() - 1; i++) {
      IrPrinter::Visit(op->args[i]);
      str_ += ", ";
    }
    IrPrinter::Visit(op->args.back());
  }
  str_ += ")";
}

std::string ReadWholeFile(const std::string &path) {
  PADDLE_ENFORCE_EQ(path.empty(),
                    false,
                    ::common::errors::InvalidArgument(
                        "The path is empty. It must not be empty."));
  std::ifstream file(path);
  PADDLE_ENFORCE_EQ(file.is_open(),
                    true,
                    ::common::errors::InvalidArgument("Failed to open file: %s",
                                                      path.c_str()));
  std::stringstream ss;
  ss << file.rdbuf();
  return ss.str();
}

void CodeGenC::PrintBuiltinCodes() {
  PADDLE_ENFORCE_EQ(
      FLAGS_cinn_x86_builtin_code_root.empty(),
      false,
      ::common::errors::InvalidArgument(
          "The flag cinn_x86_builtin_code_root should be set first"));

  const std::string x86_code_file = "_x86_builtin_source.cc";

  auto source =
      ReadWholeFile(FLAGS_cinn_x86_builtin_code_root + "/" + x86_code_file);

  str_ += source;
  str_ += "\n";
}

namespace detail {

Expr StridedRampBase(Expr e, int stride) {
  auto *ramp_n = e.As<ir::Ramp>();
  if (ramp_n) {
    auto *iv = ramp_n->stride.As<ir::IntImm>();
    if (iv && iv->value == stride) return ramp_n->base;
  }
  return Expr();
}

}  // namespace detail

}  // namespace backends

}  // namespace cinn
