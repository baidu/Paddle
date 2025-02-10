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

#include "paddle/cinn/backends/sycl/codegen_sycl_dev.h"
#include <glog/logging.h>
#include <paddle/cinn/utils/string.h>

#include <fstream>
#include <set>
#include <unordered_set>

#include "paddle/cinn/ir/op/ir_operators.h"
#include "paddle/cinn/ir/utils/ir_verify.h"
#include "paddle/cinn/optim/ir_simplify.h"

#include "paddle/cinn/backends/sycl/compiler_sycl.h"
using cinn::backends::syclrtc::NUM;

namespace cinn {
namespace backends {
namespace sycl {

const std::string CodeGenSyclDevice::source_header_ =  // NOLINT
    R"(#include <sycl/sycl.hpp>
    #include "cinn_sycl_runtime_source.h"
    typedef sycl::half float16;
)";

std::string CodeGenSyclDevice::Compile(const ir::Module &module,
                                       bool for_syclrtc) {
  for_syclrtc_ = for_syclrtc;
  auto source = Compile(module, OutputKind::CImpl);
  return source;
}

std::string CodeGenSyclDevice::Compile(const ir::Module &module,
                                       CodeGenC::OutputKind output_kind) {
  if (output_kind == OutputKind::CHeader) {
    GenerateHeaderFile(module);
  } else if (output_kind == OutputKind::CImpl) {
    PrintIncludes();

    if (for_syclrtc_) {
      str_ += "#ifdef __cplusplus\n";
      str_ += "extern \"C\" {\n";
      str_ += "#endif\n";
    }

    PrintBuiltinCodes();

    for (auto &func : module.functions()) {
      Compile(func);
    }
  } else {
    PADDLE_THROW(::common::errors::Fatal("SYCL Not supported OutputKind !"));
  }

  if (for_syclrtc_) {
    str_ += "\n#ifdef __cplusplus\n";
    str_ += "}\n";
    str_ += "#endif\n";
  }
  return str_;
}

void CodeGenSyclDevice::Compile(const ir::Module &module,
                                const Outputs &outputs) {
  CINN_NOT_IMPLEMENTED
}

void CodeGenSyclDevice::Compile(const ir::LoweredFunc &func) {
  IrPrinter::Visit(Expr(func));
}

void CodeGenSyclDevice::Visit(const ir::_LoweredFunc_ *op) {
  // clear names valid within scope when enter a new function
  vectorized_tensor_names_.clear();

  // Print the packed function
  str_ += "// CodeGenSyclDevice: NOTE: Auto-generated packed function\n";
  str_ += "void ";
  str_ += op->name;
  str_ +=
      "(sycl::queue &Q, sycl::range<3> dimGrid, sycl::range<3> dimBlock, "
      "void** void_args) {\n";
  IncIndent();
  // read void_args
  PrintFunctionDeclaration(op);
  DoIndent();
  str_ += "Q.submit([&](sycl::handler &h) {\n";
  IncIndent();
  DoIndent();
  str_ += "h.parallel_for<class " + GenerateKernelName(op) +
          ">(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock), "
          "[=](sycl::nd_item<3> item) "
          "[[intel::kernel_args_restrict]]";
  if (op->cuda_axis_info.valid()) {
    bool has_symbol_in_thread_num = false;
    std::string launch_bounds_max_work_group_size =
        "[[intel::max_work_group_size(";
    for (int i = 0; i < 3; i++) {
      ir::Expr block_dim = op->cuda_axis_info.block_dim(i);
      if (block_dim.is_constant()) {
        launch_bounds_max_work_group_size +=
            std::to_string(block_dim.as_int64());
        if (i < 2) {
          launch_bounds_max_work_group_size += ", ";
        }
      } else {
        has_symbol_in_thread_num = true;
        break;
      }
    }
    launch_bounds_max_work_group_size += ")]]";
    if (!has_symbol_in_thread_num) {
      str_ += launch_bounds_max_work_group_size;
    }
  }
  str_ += "\n";
  // function body
  PrintFunctionBody(op);

  str_ += ");\n";
  DecIndent();
  DoIndent();
  str_ += "});\n";
  DecIndent();
  str_ += "}\n";
}

void CodeGenSyclDevice::Visit(const ir::_Var_ *op) {
  if (utils::StartsWith(op->name, "threadIdx") ||
      utils::StartsWith(op->name, "blockIdx")) {
    if (utils::StartsWith(op->name, "threadIdx")) {
      str_ += "(int)item.get_local_id(";
    } else {
      str_ += "(int)item.get_group(";
    }
    if (utils::EndsWith(op->name, "x")) {
      str_ += std::to_string(2);
    } else if (utils::EndsWith(op->name, "y")) {
      str_ += std::to_string(1);
    } else if (utils::EndsWith(op->name, "z")) {
      str_ += std::to_string(0);
    }
    str_ += ")";
  } else {
    str_ += op->name;
  }
}

void CodeGenSyclDevice::Visit(const ir::Min *op) {
  str_ += "sycl::min(";
  IrPrinter::Visit(op->a());
  str_ += ", ";
  IrPrinter::Visit(op->b());
  str_ += ")";
}

void CodeGenSyclDevice::Visit(const ir::Max *op) {
  str_ += "sycl::max(";
  IrPrinter::Visit(op->a());
  str_ += ", ";
  IrPrinter::Visit(op->b());
  str_ += ")";
}

void CodeGenSyclDevice::PrintFunctionBody(const ir::_LoweredFunc_ *op) {
  DoIndent();

  std::vector<Expr> new_body;

  auto alloca_temp_buffers = op->PrepareAllocTempBufferExprs();
  auto temp_buffer_alias = GenerateBufferAliasExprs(op, op->temp_bufs);
  auto alias_var_exprs = op->CudaAliasVarExprs();

#define APPEND_TO_NEW_BODY(field__) \
  new_body.insert(std::end(new_body), std::begin(field__), std::end(field__));
  APPEND_TO_NEW_BODY(alloca_temp_buffers)
  APPEND_TO_NEW_BODY(temp_buffer_alias)
  APPEND_TO_NEW_BODY(alias_var_exprs)

  new_body.push_back(op->body);

  Expr func_body = ir::Block::Make(new_body);

  optim::SimplifyBlocks(&func_body);
  // Make sure that the function's body is wrapped by a block
  if (!func_body.As<ir::Block>()) {
    func_body = ir::Block::Make({func_body});
  }
  IrPrinter::Visit(func_body);
}

void CodeGenSyclDevice::PrintFunctionDeclaration(const ir::_LoweredFunc_ *op) {
  for (int i = 0; i < op->args.size(); i++) {
    DoIndent();
    auto &arg = op->args[i];
    if (arg.is_buffer()) {
      // In CUDA kernel, only primitive type is supported, so we replace the
      // buffer with T*j
      if (arg.is_input()) str_ += "const ";
      str_ += GetTypeRepr(arg.buffer_arg()->dtype);
      str_ += "* ";
      // str_ += kCKeywordRestrict;
      str_ += " ";
      str_ += ir::BufferGetTensorName(arg.buffer_arg().As<ir::_Buffer_>());
      str_ += " = (";
      str_ += GetTypeRepr(arg.buffer_arg()->dtype);
      str_ += "* ";
    } else if (arg.is_var()) {
      if (arg.var_arg()->type().is_cpp_handle()) {
        // str_ += kCKeywordRestrict;
      }
      str_ += GetTypeRepr(arg.type());
      str_ += " ";
      str_ += arg.name();
      str_ += " = (";
      str_ += GetTypeRepr(arg.type());
    } else {
      CINN_NOT_IMPLEMENTED
    }
    str_ += ")(*(void **)(void_args[";
    str_ += std::to_string(i);
    str_ += "]));\n";
  }
}

void CodeGenSyclDevice::PrintTempBufferCreation(const ir::Buffer &buffer) {
  PADDLE_ENFORCE_NE(
      buffer->type(),
      Void(),
      ::common::errors::InvalidArgument("this buffer is invalid!"));
  auto print_gpu_memory = [&](const std::string &mark) {
    str_ += mark;
    str_ += GetTypeRepr(buffer->dtype);
    str_ += " ";
    str_ += buffer->name;
    str_ += " ";

    str_ += "[ ";
    Expr buffer_size(1);
    for (int i = 0; i < buffer->shape.size(); i++) {
      buffer_size = buffer_size * buffer->shape[i];
    }
    optim::Simplify(&buffer_size);
    IrPrinter::Visit(buffer_size);
    str_ += " ]";
  };
  switch (buffer->memory_type) {
    case ir::MemoryType::GPUShared: {
      str_ += "auto ";
      str_ += buffer->name;
      str_ += " = *sycl::ext::oneapi::group_local_memory<";
      str_ += GetTypeRepr(buffer->dtype);
      str_ += "[ ";
      Expr buffer_size(1);
      for (int i = 0; i < buffer->shape.size(); i++) {
        buffer_size = buffer_size * buffer->shape[i];
      }
      optim::Simplify(&buffer_size);
      IrPrinter::Visit(buffer_size);
      str_ += " ]>(item.get_group())";
      break;
    }

    case ir::MemoryType::GPULocal:
      print_gpu_memory("");
      break;

    default:
      PADDLE_THROW(::common::errors::InvalidArgument(
                       "SYCL device codegen not support memory %s, type %s"),
                   buffer->name,
                   ,
                   buffer->memory_type);
  }
}

void CodeGenSyclDevice::Visit(const ir::Call *op) {
  if (op->name == "__syncthreads") {
    str_ += "sycl::group_barrier(item.get_group())";
    return;
  }
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
  // sycl need parameter nd_item
  if ((op->name.find("cinn_block_reduce") != std::string::npos) ||
      (op->name.find("cinn_warp_reduce") != std::string::npos)) {
    str_ += ", item";
  }

  str_ += ")";
}

std::string CodeGenSyclDevice::GenerateKernelName(const ir::_LoweredFunc_ *op) {
  std::string kernel_name = "space" + std::to_string(NUM::getNum());
  kernel_name += "_";
  kernel_name += op->name;
  return kernel_name;
}

const std::string &CodeGenSyclDevice::GetSourceHeader() {
  return source_header_;
}

CodeGenSyclDevice::CodeGenSyclDevice(Target target) : CodeGenGpuDev(target) {}

void CodeGenSyclDevice::PrintIncludes() { str_ += GetSourceHeader(); }

}  // namespace sycl
}  // namespace backends
}  // namespace cinn
