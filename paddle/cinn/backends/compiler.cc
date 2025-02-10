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

#include "paddle/cinn/backends/compiler.h"

#include <sys/stat.h>
#include <fstream>
#include "paddle/cinn/backends/codegen_cuda_host.h"
#include "paddle/cinn/backends/codegen_device_util.h"
#include "paddle/cinn/backends/llvm/runtime_symbol_registry.h"
#include "paddle/cinn/common/context.h"
#include "paddle/cinn/hlir/framework/graph_compiler_util.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/runtime/backend_api.h"
#ifdef CINN_WITH_CUDA
#include "paddle/cinn/backends/codegen_cuda_dev.h"
#include "paddle/cinn/backends/nvrtc/nvrtc_util.h"
#include "paddle/cinn/runtime/cuda/cuda_module.h"
#include "paddle/cinn/runtime/cuda/cuda_util.h"
#include "paddle/cinn/runtime/flags.h"
#endif
#ifdef CINN_WITH_HIP
#include "paddle/cinn/backends/hip/codegen_hip_dev.h"
#include "paddle/cinn/backends/hip/compiler_hip.h"
#include "paddle/cinn/runtime/hip/hip_module.h"
#endif
#ifdef CINN_WITH_SYCL
#include "paddle/cinn/backends/sycl/codegen_sycl_dev.h"
#include "paddle/cinn/backends/sycl/compiler_sycl.h"
#include "paddle/cinn/runtime/sycl/sycl_module.h"
#endif
#include "paddle/cinn/adt/adt.h"

PD_DECLARE_string(cinn_source_code_save_path);
PD_DECLARE_string(cinn_dump_group_lowered_func);
PD_DECLARE_string(cinn_dump_group_source_code);
PD_DECLARE_string(cinn_dump_group_ptx);
PD_DECLARE_string(cinn_dump_group_instruction);
PD_DECLARE_string(cinn_debug_custom_code_path);

namespace {

bool MakeDirectory(const std::string& dirname, mode_t mode) {
  struct stat st;
  std::string path;
  for (int i = 0; i < dirname.size(); ++i) {
    path.push_back(dirname[i]);
    if (!(dirname[i] == '/' || i + 1 == dirname.size())) {
      continue;
    }
    if (stat(path.c_str(), &st) == 0) {
      if (S_ISDIR(st.st_mode)) {
        continue;
      } else {
        LOG(WARNING) << path << " is not a directory, please check your path.";
        return false;
      }
    } else {
      if (mkdir(path.c_str(), mode) == 0) {
        continue;
      } else {
        LOG(WARNING) << "Make directory fail: " << path;
        return false;
      }
    }
  }
  return true;
}
}  // namespace

namespace cinn {
namespace backends {
using ir::Module;
using CompilationStatus = hlir::framework::CompilationStatus;

static constexpr int DebugLogMaxLen = 30000;

void CompilationInfoDumper::DumpLoweredFuncByGroupIndex(
    const ir::LoweredFunc& lowered_func, const int gidx, const int device_id) {
  if (FLAGS_cinn_dump_group_lowered_func.empty() ||
      lowered_func.get() == nullptr) {
    return;
  }
  std::stringstream content;
  content << lowered_func;
  Dump(FLAGS_cinn_dump_group_lowered_func,
       gidx,
       device_id,
       "lowered_function.txt",
       content.str());
}

void CompilationInfoDumper::DumpSourceCodeByGroupIndex(
    const std::string& source_code, const int gidx, const int device_id) {
  if (FLAGS_cinn_dump_group_source_code.empty()) {
    return;
  }
  Dump(FLAGS_cinn_dump_group_source_code,
       gidx,
       device_id,
       "source_code.cu",
       source_code);
}

void CompilationInfoDumper::DumpPtxCodeByGroupIndex(
    const std::string& source_ptx, const int gidx, const int device_id) {
  if (FLAGS_cinn_dump_group_ptx.empty()) {
    return;
  }
  Dump(
      FLAGS_cinn_dump_group_ptx, gidx, device_id, "source_ptx.ptx", source_ptx);
}

void CompilationInfoDumper::DumpLoweredFunc() {
  if (FLAGS_cinn_dump_group_lowered_func.empty()) {
    return;
  }
  for (int idx = 0; idx < info_.Size(); ++idx) {
    std::stringstream content;
    if (info_.Status(idx) > CompilationStatus::LOWERING_FAIL) {
      content << info_.LoweredFuncs(idx).front();
    } else {
      content << "[No lowered func generated]\n\n" << info_.Message(idx);
    }
    Dump(FLAGS_cinn_dump_group_lowered_func,
         idx,
         device_id_,
         "lowered_function.txt",
         content.str());
  }
}

void CompilationInfoDumper::DumpSourceCode() {
  if (FLAGS_cinn_dump_group_source_code.empty()) {
    return;
  }
  for (int idx = 0; idx < info_.Size(); ++idx) {
    std::string dump_str;
    if (info_.Status(idx) > CompilationStatus::CODEGEN_JIT_FAIL) {
      dump_str = info_.SourceCode(idx);
    } else {
      dump_str = "[No source code generated]\n\n" + info_.Message(idx);
    }
    Dump(FLAGS_cinn_dump_group_source_code,
         idx,
         device_id_,
         "source_code.cu",
         dump_str);
  }
}

void CompilationInfoDumper::DumpPtxCode() {
  if (FLAGS_cinn_dump_group_ptx.empty()) {
    return;
  }
  for (int idx = 0; idx < info_.Size(); ++idx) {
    std::string dump_str;
    if (info_.Status(idx) > CompilationStatus::CODEGEN_JIT_FAIL) {
      dump_str = info_.SourcePtx(idx);
    } else {
      dump_str = "[No source ptxs generated]\n\n" + info_.Message(idx);
    }
    Dump(
        FLAGS_cinn_dump_group_ptx, idx, device_id_, "source_ptx.ptx", dump_str);
  }
}

void CompilationInfoDumper::Dump(const std::string& base_path,
                                 const int idx,
                                 const int device_id,
                                 const std::string& file_name,
                                 const std::string& content) {
  auto dump_path = utils::StringFormat(
      "%s/device_%d/fusion_group_%d", base_path.c_str(), device_id, idx);
  if (!MakeDirectory(dump_path,
                     S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH)) {
    LOG(WARNING) << "Failed to make directory: \"" << dump_path
                 << "\", the instruction for this group will not dump.";
  } else {
    auto dump_file =
        utils::StringFormat("%s/%s", dump_path.c_str(), file_name.c_str());
    VLOG(7) << "Dump instruction to: " << dump_file;
    std::ofstream of(dump_file, std::ios_base::out);
    if (of.is_open()) {
      of << content;
      of.close();
    } else {
      LOG(WARNING) << "Failed to open file: " << dump_file
                   << ", please check your path.";
    }
  }
}

SourceCodePrint::SourceCodePrint() {
  if (!FLAGS_cinn_source_code_save_path.empty()) {
    LOG(INFO)
        << "The CINN auto generated source code will writing into file: \""
        << FLAGS_cinn_source_code_save_path << "\"";
    of.open(FLAGS_cinn_source_code_save_path, std::ios_base::out);
  }
}

SourceCodePrint::~SourceCodePrint() {
  if (of.is_open()) {
    of.close();
  }
}

void SourceCodePrint::write(const std::string& source_code) {
  std::lock_guard<std::mutex> guard(mtx_);
  if (of.is_open()) {
    of << source_code << std::endl;
  } else if (!FLAGS_cinn_source_code_save_path.empty()) {
    LOG(WARNING) << "Failed to open \"" << FLAGS_cinn_source_code_save_path
                 << "\", source code will print.";
    if (source_code.size() > DebugLogMaxLen) {
      LOG(INFO) << "[CUDA] source code-0:\n"
                << source_code.substr(0, DebugLogMaxLen);
      for (int i = 1; i * DebugLogMaxLen < source_code.size(); ++i) {
        LOG(INFO) << "[CUDA] source code-" << i << ":\n"
                  << source_code.substr(DebugLogMaxLen * i, DebugLogMaxLen);
      }
    } else {
      LOG(INFO) << "[CUDA] source code:\n" << source_code;
    }
  }
}

void Compiler::Build(const Module& module, const std::string& code) {
  target_.arch.Match(
      [&](common::UnknownArch) { CINN_NOT_IMPLEMENTED; },
      [&](common::X86Arch) { CompileX86Module(module); },
      [&](common::ARMArch) { CINN_NOT_IMPLEMENTED; },
      [&](common::NVGPUArch) { CompileCudaModule(module, code); },
      [&](common::HygonDCUArchHIP) { CompileHipModule(module, code); },
      [&](common::HygonDCUArchSYCL) { CompileSyclModule(module, code); });
}

void Compiler::AppendCX86(const Module& module) {
  VLOG(3) << "Start Compiler::BuildCX86" << module;
  CompileX86Module(module);
  VLOG(3) << "Over Compiler::BuildCX86";
}

void Compiler::AppendBroadcastSwitchModule(const ir::Module& module) {
  engine_->Link<CodeGenSwitchHost>(module);
}

void Compiler::EndCompile() {
  RegisterDeviceModuleSymbol();
  engine_->AddSelfModule();
}

std::string Compiler::GetSourceCode(const ir::Module& module) {
  return target_.arch.Match(
      [&](common::UnknownArch) -> std::string { CINN_NOT_IMPLEMENTED; },
      [&](common::X86Arch) -> std::string { CINN_NOT_IMPLEMENTED; },
      [&](common::ARMArch) -> std::string { CINN_NOT_IMPLEMENTED; },
      [&](common::NVGPUArch) -> std::string {
#ifdef CINN_WITH_CUDA
        auto _host_module_device_module_ =
            SplitDeviceAndHostModule(module);  // NOLINT
        auto& host_module = std::get<0>(_host_module_device_module_);
        auto& device_module = std::get<1>(_host_module_device_module_);
        CodeGenCudaDev codegen(target_);
        auto source_code = codegen.Compile(device_module);
        return source_code;
#else
        CINN_NOT_IMPLEMENTED
#endif
      },
      [&](common::HygonDCUArchHIP) -> std::string {
#ifdef CINN_WITH_HIP
        auto _host_module_device_module_ =
            SplitDeviceAndHostModule(module);  // NOLINT
        auto& host_module = std::get<0>(_host_module_device_module_);
        auto& device_module = std::get<1>(_host_module_device_module_);
        hip::CodeGenHipDevice codegen(target_);
        auto source_code = codegen.Compile(device_module);
        return source_code;
#else
        CINN_NOT_IMPLEMENTED
#endif
      },
      [&](common::HygonDCUArchSYCL) -> std::string {
#ifdef CINN_WITH_SYCL
        auto _host_module_device_module_ =
            SplitDeviceAndHostModule(module);  // NOLINT
        auto& host_module = std::get<0>(_host_module_device_module_);
        auto& device_module = std::get<1>(_host_module_device_module_);
        sycl::CodeGenSyclDevice codegen(target_);
        auto source_code = codegen.Compile(device_module);
        return source_code;
#else
        CINN_NOT_IMPLEMENTED
#endif
      });
}

void Compiler::BuildDefault(const Module& module) {
  target_.arch.Match(
      [&](common::UnknownArch) { CINN_NOT_IMPLEMENTED; },
      [&](common::X86Arch) { CompileX86Module(module); },
      [&](common::ARMArch) { CINN_NOT_IMPLEMENTED; },
      [&](common::NVGPUArch) { CompileCudaModule(module); },
      [&](common::HygonDCUArchHIP) { CompileHipModule(module); },
      [&](common::HygonDCUArchSYCL) { CompileSyclModule(module); });
}

namespace {
std::string GetFileContent(const std::string& path) {
  std::ifstream file(path);

  if (!file.is_open()) {
    std::cerr << "Unable to open file: " << path << std::endl;
    return "";
  }

  std::ostringstream ss;
  ss << file.rdbuf();
  std::string content = ss.str();

  file.close();
  return content;
}
}  // namespace

void Compiler::RegisterDeviceModuleSymbol() {
  return target_.arch.Match(
      [&](common::UnknownArch) { CINN_NOT_IMPLEMENTED; },
      [&](common::X86Arch) { return; },
      [&](common::ARMArch) { return; },
      [&](common::NVGPUArch) { RegisterCudaModuleSymbol(); },
      [&](common::HygonDCUArchHIP) { RegisterHipModuleSymbol(); },
      [&](common::HygonDCUArchSYCL) { RegisterSyclModuleSymbol(); });
}

void Compiler::RegisterCudaModuleSymbol() {
#ifdef CINN_WITH_CUDA
  nvrtc::Compiler compiler;
  std::string source_code = CodeGenCudaDev::GetSourceHeader() + device_fn_code_;
  auto ptx = compiler(source_code);
  PADDLE_ENFORCE_EQ(!ptx.empty(),
                    true,
                    ::common::errors::InvalidArgument(
                        "Compile PTX failed from source code\n"));
  using runtime::cuda::CUDAModule;
  cuda_module_.reset(new CUDAModule(ptx,
                                    compiler.compile_to_cubin()
                                        ? CUDAModule::Kind::CUBIN
                                        : CUDAModule::Kind::PTX));

  RuntimeSymbols symbols;
  for (const auto& kernel_fn_name : device_fn_name_) {
    auto fn_kernel = cuda_module_->GetFunction(kernel_fn_name);
    PADDLE_ENFORCE_NOT_NULL(fn_kernel,
                            ::common::errors::InvalidArgument(
                                "Fail to get CUfunction kernel_fn_name"));
    fn_ptr_.push_back(reinterpret_cast<void*>(fn_kernel));
    symbols.RegisterVar(kernel_fn_name + "_ptr_",
                        reinterpret_cast<void*>(fn_kernel));
  }
  engine_->RegisterModuleRuntimeSymbols(std::move(symbols));
#else
  CINN_NOT_IMPLEMENTED
#endif
}

void Compiler::RegisterHipModuleSymbol() {
#ifdef CINN_WITH_HIP
  hiprtc::Compiler compiler;
  std::string source_code =
      hip::CodeGenHipDevice::GetSourceHeader() + device_fn_code_;
  std::string hsaco = compiler(source_code);
  PADDLE_ENFORCE_EQ(
      !hsaco.empty(),
      true,
      ::common::errors::Fatal("Compile hsaco failed from source code:\n%s",
                              source_code));
  using runtime::hip::HIPModule;
  hip_module_.reset(new HIPModule(hsaco));
  // get device id
  using cinn::runtime::BackendAPI;
  int device_id = BackendAPI::get_backend(target_)->get_device();
  // register kernel
  RuntimeSymbols symbols;
  for (const auto& kernel_fn_name : device_fn_name_) {
    auto fn_kernel = hip_module_->GetFunction(device_id, kernel_fn_name);
    PADDLE_ENFORCE_NOT_NULL(
        fn_kernel,
        ::common::errors::Fatal("HIP GetFunction Error: get valid kernel."));
    fn_ptr_.push_back(reinterpret_cast<void*>(fn_kernel));
    symbols.RegisterVar(kernel_fn_name + "_ptr_",
                        reinterpret_cast<void*>(fn_kernel));
  }
  engine_->RegisterModuleRuntimeSymbols(std::move(symbols));
#else
  CINN_NOT_IMPLEMENTED
#endif
}

void Compiler::RegisterSyclModuleSymbol() {
#ifdef CINN_WITH_SYCL
  syclrtc::Compiler compiler;
  std::string source_code =
      sycl::CodeGenSyclDevice::GetSourceHeader() + device_fn_code_;
  std::string hsaco = compiler(source_code);
  PADDLE_ENFORCE_EQ(
      !hsaco.empty(),
      true,
      ::common::errors::Fatal("Compile hsaco failed from source code:\n%s",
                              source_code));
  using runtime::sycl::SYCLModule;
  sycl_module_.reset(new SYCLModule(source_code, hsaco, SYCLModule::Kind::so));
  // get device id
  using cinn::runtime::BackendAPI;
  int device_id = BackendAPI::get_backend(target_)->get_device();
  // register kernel
  RuntimeSymbols symbols;
  for (const auto& kernel_fn_name : device_fn_name_) {
    auto fn_kernel = sycl_module_->GetFunction(kernel_fn_name);
    PADDLE_ENFORCE_NOT_NULL(
        fn_kernel,
        ::common::errors::Fatal("HIP GetFunction Error: get valid kernel."));
    fn_ptr_.push_back(reinterpret_cast<void*>(fn_kernel));
    symbols.RegisterVar(kernel_fn_name + "_ptr_",
                        reinterpret_cast<void*>(fn_kernel));
  }
  engine_->RegisterModuleRuntimeSymbols(std::move(symbols));
#else
  CINN_NOT_IMPLEMENTED
#endif
}

void Compiler::CompileCudaModule(const Module& module,
                                 const std::string& code) {
#ifdef CINN_WITH_CUDA
  auto _host_module_device_module_ =
      SplitDeviceAndHostModule(module);  // NOLINT
  auto& host_module = std::get<0>(_host_module_device_module_);
  auto& device_module = std::get<1>(_host_module_device_module_);
  VLOG(3) << "[CUDA] host module:\n" << host_module;

  VLOG(3) << "[CUDA] device module:\n" << device_module;
  std::string source_code;

  if (!FLAGS_cinn_debug_custom_code_path.empty()) {
    std::string file_path = FLAGS_cinn_debug_custom_code_path;
    source_code = GetFileContent(file_path);
  } else if (code.empty()) {
    CodeGenCudaDev codegen(target_);
    source_code = codegen.Compile(device_module);
  } else {
    source_code = code;
  }

  PADDLE_ENFORCE_EQ(!source_code.empty(),
                    true,
                    ::common::errors::InvalidArgument(
                        "Compile CUDA C code failed from device module"));
  VLOG(3) << "[CUDA] C:\n" << source_code;
  SourceCodePrint::GetInstance()->write(source_code);
  device_fn_code_ += source_code;

  for (auto& fn : device_module.functions()) {
    std::string kernel_fn_name = fn->name;
    device_fn_name_.emplace_back(kernel_fn_name);
  }
  engine_->Link<CodeGenGpuHost>(host_module);
#else
  CINN_NOT_IMPLEMENTED
#endif
}

void Compiler::CompileHipModule(const Module& module, const std::string& code) {
#ifdef CINN_WITH_HIP
  auto _host_module_device_module_ =
      SplitDeviceAndHostModule(module);  // NOLINT
  auto& host_module = std::get<0>(_host_module_device_module_);
  auto& device_module = std::get<1>(_host_module_device_module_);
  VLOG(3) << "[HIP] host module:\n" << host_module;
  VLOG(3) << "[HIP] device module:\n" << device_module;
  std::string source_code;
  if (!FLAGS_cinn_debug_custom_code_path.empty()) {
    std::string file_path = FLAGS_cinn_debug_custom_code_path;
    source_code = GetFileContent(file_path);
  } else if (code.empty()) {
    hip::CodeGenHipDevice codegen(target_);
    source_code = codegen.Compile(device_module);
  } else {
    source_code = code;
  }
  PADDLE_ENFORCE_EQ(
      !source_code.empty(),
      true,
      ::common::errors::Fatal("Compile HIP code failed from device module:\n%s",
                              device_module));
  VLOG(3) << "[HIP]:\n" << source_code;
  SourceCodePrint::GetInstance()->write(source_code);
  device_fn_code_ += source_code;
  for (auto& fn : device_module.functions()) {
    std::string kernel_fn_name = fn->name;
    device_fn_name_.emplace_back(kernel_fn_name);
  }
  engine_->Link<CodeGenGpuHost>(host_module);
#else
  CINN_NOT_IMPLEMENTED
#endif
}

void Compiler::CompileSyclModule(const Module& module,
                                 const std::string& code) {
#ifdef CINN_WITH_SYCL
  auto _host_module_device_module_ =
      SplitDeviceAndHostModule(module);  // NOLINT
  auto& host_module = std::get<0>(_host_module_device_module_);
  auto& device_module = std::get<1>(_host_module_device_module_);
  VLOG(3) << "[SYCL] host module:\n" << host_module;
  VLOG(3) << "[SYCL] device module:\n" << device_module;
  std::string source_code;
  if (!FLAGS_cinn_debug_custom_code_path.empty()) {
    std::string file_path = FLAGS_cinn_debug_custom_code_path;
    source_code = GetFileContent(file_path);
  } else if (code.empty()) {
    sycl::CodeGenSyclDevice codegen(target_);
    source_code = codegen.Compile(device_module);
  } else {
    source_code = code;
  }
  PADDLE_ENFORCE_EQ(
      !source_code.empty(),
      true,
      ::common::errors::Fatal(
          "Compile SYCL code failed from device module:\n%s", device_module));
  VLOG(3) << "[SYCL]:\n" << source_code;
  SourceCodePrint::GetInstance()->write(source_code);
  device_fn_code_ += source_code;
  for (auto& fn : device_module.functions()) {
    std::string kernel_fn_name = fn->name;
    device_fn_name_.emplace_back(kernel_fn_name);
  }
  engine_->Link<CodeGenGpuHost>(host_module);
#else
  CINN_NOT_IMPLEMENTED
#endif
}

void Compiler::CompileX86Module(const Module& module) {
  engine_->Link<CodeGenX86>(module);
}

void Compiler::ExportObject(const std::string& path) {
  engine_->ExportObject(path);
}

void* Compiler::Lookup(absl::string_view fn_name) {
  PADDLE_ENFORCE_NOT_NULL(
      engine_, ::common::errors::InvalidArgument("Sorry, engine_ is nullptr"));
  if (engine_->Lookup(fn_name) != nullptr) {
    return engine_->Lookup(fn_name);
  }
  return nullptr;
}

}  // namespace backends
}  // namespace cinn
