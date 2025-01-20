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

#include "paddle/cinn/backends/codegen_gpu_dev.h"

#include <glog/logging.h>
#include <paddle/cinn/utils/string.h>

#include <fstream>
#include <set>
#include <unordered_set>

#include "paddle/cinn/common/cas.h"
#include "paddle/cinn/common/ir_util.h"
#include "paddle/cinn/ir/op/ir_operators.h"
#include "paddle/cinn/ir/utils/ir_verify.h"
#include "paddle/cinn/optim/ir_simplify_pass.h"
#include "paddle/common/enforce.h"
#include "paddle/common/errors.h"
namespace cinn {
namespace backends {

CodeGenGpuDev::CodeGenGpuDev(Target target) : CodeGenC(target) {}

std::string CodeGenGpuDev::Compile(const ir::Module &module, bool use_rtc) {
  use_rtc_ = use_rtc;
  auto source = Compile(module, OutputKind::CImpl);

  return source;
}

void CodeGenGpuDev::Compile(const ir::Module &module, const Outputs &outputs) {
  ir::ir_utils::IrVerify(module.As<ir::_Module_>());

  CodeGenC::inline_builtin_codes_ = false;
  if (!outputs.c_header_name.empty()) {
    auto source = Compile(module, OutputKind::CHeader);
    str_ = "";
    std::ofstream file(outputs.c_header_name);
    PADDLE_ENFORCE_EQ(file.is_open(),
                      true,
                      ::common::errors::PreconditionNotMet(
                          "failed to open file %s", outputs.c_header_name));
    file << source;
    file.close();
    VLOG(5) << "Output C header to file " << outputs.c_header_name;
  }

  if (!outputs.cuda_source_name.empty()) {
    auto source = Compile(module, OutputKind::CImpl);
    str_ = "";
    std::ofstream file(outputs.cuda_source_name);
    PADDLE_ENFORCE_EQ(file.is_open(),
                      true,
                      ::common::errors::PreconditionNotMet(
                          "failed to open file %s", outputs.cuda_source_name));
    file << source;
    file.close();
    VLOG(5) << "Output C source to file " << outputs.cuda_source_name;
  }
}

void CodeGenGpuDev::Compile(const ir::LoweredFunc &func) {
  dyn_shared_mem_offset_ = Expr(-1);
  Visit(func.As<ir::_LoweredFunc_>());
}

std::vector<Expr> CodeGenGpuDev::GenerateBufferAliasExprs(
    const ir::_LoweredFunc_ *op, const std::vector<ir::Buffer> &temp_buffers) {
  std::set<ir::Buffer> temp_buffer_set(temp_buffers.begin(),
                                       temp_buffers.end());
  // prepare temp buffer alias
  std::vector<Expr> buffer_alias;
  auto tensors = ir::ir_utils::CollectIRNodes(op->body, [&](const Expr *x) {
    return x->as_tensor() && x->as_tensor()->buffer.defined() &&
           temp_buffer_set.count(x->as_tensor()->buffer);
  });

  // unique tensors
  std::set<ir::Tensor> unique_tensors;
  for (auto &e : tensors) {
    unique_tensors.insert(e.as_tensor_ref());
  }

  for (auto &t : unique_tensors) {
    auto data_type = t->type();
    auto data_ptr_type = data_type;
    data_ptr_type.set_cpp_handle();

    Var t_var(t->name, data_ptr_type);
    Var buf_var(t->buffer->name, data_ptr_type);
    buffer_alias.push_back(ir::Let::Make(t_var, buf_var));
  }

  return buffer_alias;
}

std::vector<Expr> FilterDeallocTempBuffers(const std::vector<Expr> &frees) {
  std::vector<Expr> filtered;
  for (const Expr &free : frees) {
    const ir::Free *op = free.As<ir::Free>();
    PADDLE_ENFORCE_NOT_NULL(
        op, ::common::errors::InvalidArgument("Free is not a free node"));
    bool has_symbolic_constant = false;
    const ir::_Buffer_ *buffer = op->destination.As<ir::_Buffer_>();
    for (Expr shape : buffer->shape) {
      shape = optim::ArithSimplify(shape);
      ir::ir_utils::CollectIRNodes(shape, [&](const Expr *x) {
        if (x->as_var()) {
          PADDLE_ENFORCE_EQ(
              x->as_var()->is_symbolic_constant,
              true,
              ::common::errors::PreconditionNotMet(
                  "var in buffer shape must be symbolic constant."));
          has_symbolic_constant = true;
        }
        return false;
      });
    }
    if (has_symbolic_constant &&
        buffer->memory_type == ir::MemoryType::GPULocal) {
      filtered.emplace_back(free);
    }
  }
  return filtered;
}

void CodeGenGpuDev::Visit(const ir::_LoweredFunc_ *op) {
  // clear names valid within scope when enter a new function
  vectorized_tensor_names_.clear();
  str_ += "__global__\n";

  PrintFunctionDeclaration(op);
  str_ += "\n";

  DoIndent();

  std::vector<Expr> new_body;

  auto axis_range_assumptions = op->PrepareAxisRangeAssumptions();
  auto alloca_temp_buffers = op->PrepareAllocTempBufferExprs();
  auto temp_buffer_alias = GenerateBufferAliasExprs(op, op->temp_bufs);
  auto alias_var_exprs = op->CudaAliasVarExprs();
  auto dealloc_temp_buffers =
      FilterDeallocTempBuffers(op->PrepareDeallocTempBufferExprs());

#define APPEND_TO_NEW_BODY(field__) \
  new_body.insert(std::end(new_body), std::begin(field__), std::end(field__));
  APPEND_TO_NEW_BODY(axis_range_assumptions)
  APPEND_TO_NEW_BODY(alloca_temp_buffers)
  APPEND_TO_NEW_BODY(temp_buffer_alias)
  APPEND_TO_NEW_BODY(alias_var_exprs)

  new_body.push_back(op->body);
  APPEND_TO_NEW_BODY(dealloc_temp_buffers);

  Expr func_body = ir::Block::Make(new_body);

  // Make sure that the function's body is wrapped by a block
  if (!func_body.As<ir::Block>()) {
    func_body = ir::Block::Make({func_body});
  }
  IrPrinter::Visit(func_body);
}

void CodeGenGpuDev::Visit(const ir::Free *op) {
  str_ += "delete [] ";
  str_ += op->destination.As<ir::_Buffer_>()->name;
  str_ += ";\n";
}

void CodeGenGpuDev::Visit(const ir::_Var_ *op) {
  if (utils::StartsWith(op->name, "threadIdx") ||
      utils::StartsWith(op->name, "blockIdx")) {
    str_ += "(int)";
    str_ += op->name;
  } else {
    str_ += op->name;
  }
}

void CodeGenGpuDev::Visit(const ir::Alloc *op) {
  PADDLE_ENFORCE_NE(op->destination.as_buffer(),
                    nullptr,
                    ::common::errors::PreconditionNotMet(
                        "Buffer shouldn't be null in Alloc instruction."));
  PrintTempBufferCreation(op->destination.as_buffer_ref());
}

void CodeGenGpuDev::Visit(const ir::Min *op) {
  str_ += "min(";
  IrPrinter::Visit(op->a());
  str_ += ", ";
  IrPrinter::Visit(op->b());
  str_ += ")";
}

void CodeGenGpuDev::Visit(const ir::Max *op) {
  str_ += "max(";
  IrPrinter::Visit(op->a());
  str_ += ", ";
  IrPrinter::Visit(op->b());
  str_ += ")";
}

void CodeGenGpuDev::PrintFunctionDeclaration(const ir::_LoweredFunc_ *op) {
  str_ += "void ";
  if (op->cuda_axis_info.valid()) {
    bool has_symbol_in_thread_num = false;
    int thread_num = 1;
    for (int i = 0; i < 3; i++) {
      ir::Expr block_dim = op->cuda_axis_info.block_dim(i);
      if (block_dim.is_constant()) {
        thread_num *= block_dim.get_constant();
      } else {
        has_symbol_in_thread_num = true;
        break;
      }
    }
    if (!has_symbol_in_thread_num) {
      str_ += "__launch_bounds__(";
      str_ += std::to_string(thread_num);
      str_ += ") ";
    }
  }

  str_ += op->name;
  str_ += "(";
  for (int i = 0; i < op->args.size() - 1; i++) {
    auto &arg = op->args[i];
    PrintFuncArg(arg);
    str_ += ", ";
  }
  if (!op->args.empty()) {
    PrintFuncArg(op->args.back());
  }
  str_ += ")";
}

void CodeGenGpuDev::PrintFuncArg(const ir::Argument &arg) {
  if (arg.is_buffer()) {
    // In CUDA/HIP kernel, only primitive type is supported, so we replace the
    // buffer with T*j
    if (arg.is_input()) str_ += "const ";
    str_ += GetTypeRepr(arg.buffer_arg()->dtype);
    str_ += "* ";
    str_ += kCKeywordRestrict;
    str_ += " ";
    str_ += ir::BufferGetTensorName(arg.buffer_arg().As<ir::_Buffer_>());
  } else if (arg.is_var()) {
    if (arg.var_arg()->type().is_cpp_handle()) {
      str_ += kCKeywordRestrict;
    }
    str_ += GetTypeRepr(arg.type());
    str_ += " ";
    str_ += arg.name();
  } else {
    CINN_NOT_IMPLEMENTED
  }
}

void CodeGenGpuDev::PrintBuiltinCodes() {}

std::string CodeGenGpuDev::Compile(const ir::Module &module,
                                   CodeGenC::OutputKind output_kind) {
  if (output_kind == OutputKind::CHeader) {
    GenerateHeaderFile(module);
  } else if (output_kind == OutputKind::CImpl) {
    if (use_rtc_) {
      str_ += "\nextern \"C\" {\n\n";
    }

    PrintBuiltinCodes();

    for (auto &func : module.functions()) {
      Compile(func);
    }
  } else {
    PADDLE_THROW(::common::errors::InvalidArgument("Not supported OutputKind"));
  }

  if (use_rtc_) {
    str_ += "\n\n}";
  }
  return str_;
}

void CodeGenGpuDev::PrintTempBufferCreation(const ir::Buffer &buffer) {
  PADDLE_ENFORCE_NE(
      buffer->type(),
      Void(),
      ::common::errors::InvalidArgument("buffer type should not be void"));
  // Calculate buffer size and determine if it contains a symbolic constant
  Expr buffer_size(1);
  for (int i = 0; i < buffer->shape.size(); i++) {
    buffer_size = buffer_size * buffer->shape[i];
  }
  optim::Simplify(&buffer_size, optim::SimplifyType::kExpr);
  bool has_symbolic_constant = false;
  ir::ir_utils::CollectIRNodes(buffer_size, [&](const Expr *x) {
    if (x->as_var()) {
      PADDLE_ENFORCE_EQ(x->as_var()->is_symbolic_constant,
                        true,
                        ::common::errors::PreconditionNotMet(
                            "var in buffer shape must be symbolic constant."));
      has_symbolic_constant = true;
    }
    return false;
  });

  if (buffer->memory_type == ir::MemoryType::GPUShared) {
    if (MathEqual(dyn_shared_mem_offset_, Expr(-1))) {
      // The first shared memory buffer, uint8_t as a byte
      str_ += "extern __shared__ uint8_t dyn_shared_buffer[];\n  ";
      dyn_shared_mem_offset_ = Expr(0);
    }
    std::string type_name = GetTypeRepr(buffer->dtype);
    str_ += type_name;
    str_ += " *";
    str_ += buffer->name;
    str_ += " = (";
    str_ += type_name;
    str_ += "*)&dyn_shared_buffer[ ";
    IrPrinter::Visit(dyn_shared_mem_offset_);
    str_ += " ]";

    int type_bytes = buffer->dtype.bytes();
    dyn_shared_mem_offset_ =
        dyn_shared_mem_offset_ + buffer_size * Expr(type_bytes);
    optim::Simplify(&dyn_shared_mem_offset_, optim::SimplifyType::kExpr);
    VLOG(6) << "dyn_shared_mem_offset_ = " << dyn_shared_mem_offset_;
  } else if (buffer->memory_type == ir::MemoryType::GPULocal) {
    // print func of static allocation
    auto print_gpu_memory = [&](const std::string &mark) {
      str_ += mark;
      str_ += GetTypeRepr(buffer->dtype);
      str_ += " ";
      str_ += buffer->name;
      str_ += " ";

      str_ += "[ ";
      IrPrinter::Visit(buffer_size);
      str_ += " ]";
    };
    // print func of dynamic allocation
    auto print_gpu_local_memory_dynamic_allocation = [&]() {
      str_ += GetTypeRepr(buffer->dtype);
      str_ += " *";
      str_ += buffer->name;
      str_ += " = new ";
      str_ += GetTypeRepr(buffer->dtype);
      str_ += "[ ";
      IrPrinter::Visit(buffer_size);
      str_ += " ]";
    };
    if (has_symbolic_constant) {
      print_gpu_local_memory_dynamic_allocation();
    } else {
      print_gpu_memory("");
    }
  } else {
    std::stringstream ss;
    ss << "CUDA/HIP device codegen not support memory " << buffer->name
       << ", type " << buffer->memory_type;
    PADDLE_THROW(::common::errors::InvalidArgument(
        "CUDA/HIP codegen error in CINN: %s", ss.str()));
  }
}

void CodeGenGpuDev::Visit(const ir::Call *op) {
  str_ += op->name;
  str_ += "(";

  if (!op->read_args.empty()) {
    for (int i = 0; i < op->read_args.size() - 1; i++) {
      auto &arg = op->read_args[i];
      if (arg.as_tensor()) {
        str_ += arg.as_tensor()->name;
        str_ += ", ";
      } else {
        IrPrinter::Visit(arg);
        str_ += ", ";
      }
    }
    if (op->read_args.back().as_tensor()) {
      str_ += op->read_args.back().as_tensor()->name;
    } else {
      IrPrinter::Visit(op->read_args.back());
    }
  }

  if (!op->write_args.empty()) {
    str_ += ", ";
    for (int i = 0; i < op->write_args.size() - 1; i++) {
      auto &arg = op->write_args[i];
      if (arg.as_tensor()) {
        str_ += arg.as_tensor()->name;
        str_ += ", ";
      } else {
        IrPrinter::Visit(arg);
        str_ += ", ";
      }
    }
    if (op->write_args.back().as_tensor()) {
      str_ += op->write_args.back().as_tensor()->name;
    } else {
      IrPrinter::Visit(op->write_args.back());
    }
  }

  str_ += ")";
}

void CodeGenGpuDev::Visit(const ir::Let *op) {
  PADDLE_ENFORCE_EQ(
      op->type().valid(),
      true,
      ::common::errors::PreconditionNotMet("Let op type must be valid."));
  // identify vectorized tensors by checking their dtypes are customized_type
  // with customized_type::kcuda_builtin_vector_t prefix, and save their names
  if (op->type().is_customized() &&
      utils::StartsWith(
          op->type().customized_type(),
          cinn::common::customized_type::kcuda_builtin_vector_t)) {
    str_ += GetTypeRepr(op->type());
    if (op->type().is_cpp_handle()) {
      str_ += " ";
      str_ += kCKeywordRestrict;
    }
    str_ += " ";
    IrPrinter::Visit(op->symbol);
    vectorized_tensor_names_.insert(utils::GetStreamCnt(op->symbol));
    // skip "=0" in "half8 temp = 0;" since the operator= of half8 may not
    // overloaded.
    if (op->body.As<ir::IntImm>() && op->body.As<ir::IntImm>()->value == 0) {
      return;
    }
    str_ += " = ";
    IrPrinter::Visit(op->body);
  } else {
    CodeGenC::Visit(op);
  }
}

bool CodeGenGpuDev::PrintBuiltinVectorAccess(const ir::LoadStoreAddrMnger *op,
                                             ir::Expr index_expr,
                                             bool is_store) {
  static constexpr char index2suffix[8] = {
      'x', 'y', 'z', 'w', 'v', 'u', 't', 's'};

  // addr of op should be a place of tensor and the index is simple int number
  if (!op->is_addr_tensor() || !index_expr.As<ir::IntImm>()) {
    return false;
  }
  auto *tensor = op->tensor.As<ir::_Tensor_>();
  PADDLE_ENFORCE_NOT_NULL(
      tensor,
      ::common::errors::InvalidArgument(
          "LoadStoreAddrMnger contains NULL tensor, which is an "
          "illegal argument"));

  // identify vectorized tensors by their names
  if (!vectorized_tensor_names_.count(tensor->name)) {
    return false;
  }

  // the index can't exceed the range of cuda/hip built-in vector type
  int index = index_expr.As<ir::IntImm>()->value;
  if (index < 0 || index >= 8) {
    return false;
  }
  if (is_store && tensor->type().is_cpp_handle()) {
    str_ += tensor->name;
    str_ += "[";
    str_ += std::to_string(index);
    str_ += "]";
  } else {
    str_ += tensor->name;
    str_ += (tensor->type().is_cpp_handle() ? "->" : ".");
    str_ += index2suffix[index];
  }
  return true;
}

void CodeGenGpuDev::Visit(const ir::Load *op) {
  // overload this visit function to especially deal with the case when it
  // accesses element at a cuda/hip built-in vector, others still resolve to
  // CodeGenC
  if (!PrintBuiltinVectorAccess(op, op->index(), false)) {
    CodeGenC::Visit(op);
  }
}

void CodeGenGpuDev::Visit(const ir::Store *op) {
  // overload this visit function to especially deal with the case when it
  // accesses element at a cuda/hip built-in vector, others still resolve to
  // CodeGenC
  if (PrintBuiltinVectorAccess(op, op->index(), true)) {
    str_ += " = ";
    IrPrinter::Visit(op->value);
  } else {
    CodeGenC::Visit(op);
  }
}

ir::Expr CalculateSharedMemory(const ir::Buffer &buffer) {
  Expr buffer_size(1);
  for (int i = 0; i < buffer->shape.size(); i++) {
    buffer_size = buffer_size * buffer->shape[i];
  }
  int type_bytes = buffer->dtype.bytes();
  return buffer_size * Expr(type_bytes);
}

ir::Expr CalculateSharedMemory(const ir::LoweredFunc &func) {
  auto alloc_temp_buffers = func->PrepareAllocTempBufferExprs();
  ir::Expr shm_size{0};
  for (const auto &alloc : alloc_temp_buffers) {
    PADDLE_ENFORCE_NOT_NULL(
        alloc.As<ir::Alloc>(),
        ::common::errors::InvalidType("expr is not a Alloc node"));
    PADDLE_ENFORCE_NOT_NULL(
        alloc.As<ir::Alloc>()->destination.as_buffer(),
        ::common::errors::InvalidType("expr is not a Buffer node"));

    auto buffer = alloc.As<ir::Alloc>()->destination.as_buffer_ref();
    if (buffer->memory_type == ir::MemoryType::GPUShared) {
      shm_size = shm_size + CalculateSharedMemory(buffer);
    }
  }
  return optim::ArithSimplify(shm_size);
}

}  // namespace backends
}  // namespace cinn
