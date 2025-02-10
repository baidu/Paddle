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

#pragma once

#include <absl/strings/string_view.h>

#include <fstream>
#include <memory>
#include <mutex>
#include <string>

#include "paddle/cinn/backends/llvm/codegen_llvm.h"
#include "paddle/cinn/backends/llvm/execution_engine.h"
#include "paddle/cinn/backends/llvm/simple_jit.h"
#include "paddle/cinn/hlir/framework/graph_compiler_util.h"
#include "paddle/cinn/lang/packed_func.h"
#ifdef CINN_WITH_CUDA
#include "paddle/cinn/runtime/cuda/cuda_module.h"
#endif
#ifdef CINN_WITH_HIP
#include "paddle/cinn/runtime/hip/hip_module.h"
#endif
#ifdef CINN_WITH_SYCL
#include "paddle/cinn/runtime/sycl/sycl_module.h"
#endif

namespace cinn {
namespace backends {

/**
 * A class for dumping the code after compilation.
 * Use FLAGS_cinn_dump_group_lowered_func to specify the directory to dump
 * lowered function. Use FLAGS_cinn_dump_group_source_code to specify the
 * directory to dump the source code. Use FLAGS_cinn_dump_group_ptx to specify
 * the directory to dump ptx. Use FLAGS_cinn_dump_group_instruction to specify
 * the directory to dump instruction.
 */
class CompilationInfoDumper {
 public:
  explicit CompilationInfoDumper(const hlir::framework::CompilationResult& info,
                                 const int device_id)
      : info_(info), device_id_(device_id) {
    DumpLoweredFunc();
    DumpSourceCode();
    DumpPtxCode();
    DumpInstruction();
  }

  static void DumpLoweredFuncByGroupIndex(const ir::LoweredFunc& lowered_func,
                                          const int gidx,
                                          const int device_id);
  static void DumpSourceCodeByGroupIndex(const std::string& source_code,
                                         const int gidx,
                                         const int device_id);
  static void DumpPtxCodeByGroupIndex(const std::string& source_ptx,
                                      const int gidx,
                                      const int device_id);

 private:
  void DumpLoweredFunc();
  void DumpSourceCode();
  void DumpPtxCode();
  void DumpInstruction();
  static void Dump(const std::string& base_path,
                   const int idx,
                   const int device_id,
                   const std::string& file_name,
                   const std::string& content);

  const hlir::framework::CompilationResult& info_;
  const int device_id_;
};

class SourceCodePrint {
 public:
  static SourceCodePrint* GetInstance() {
    static SourceCodePrint print;
    return &print;
  }

  void write(const std::string& source_code);

 private:
  SourceCodePrint();
  ~SourceCodePrint();

  std::ofstream of;
  std::mutex mtx_;
};

class Compiler final {
 public:
  static std::unique_ptr<Compiler> Create(const Target& target) {
    return std::unique_ptr<Compiler>(new Compiler(target));
  }

  /**
   * Compile and link to a CINN module.
   */
  void Build(const ir::Module& module, const std::string& code = "");

  void AppendCX86(const ir::Module& module);

  void AppendBroadcastSwitchModule(const ir::Module& module);

  void EndCompile();

  void ExportObject(const std::string& path);

  std::string GetSourceCode(const ir::Module& module);

  void BuildDefault(const ir::Module& module);

  /**
   * Retrieve a function by \p fn_name.
   * @return function address or null if not exists.
   */
  void* Lookup(absl::string_view fn_name);

  std::vector<void*> GetFnPtr() const { return fn_ptr_; }

 private:
  // do not register device symbol until end=true for build function
  void RegisterDeviceModuleSymbol();

  void RegisterCudaModuleSymbol();

  void RegisterHipModuleSymbol();

  void RegisterSyclModuleSymbol();

  void CompileCudaModule(const ir::Module& module,
                         const std::string& code = "");

  void CompileHipModule(const ir::Module& module, const std::string& code = "");

  void CompileSyclModule(const ir::Module& module,
                         const std::string& code = "");

  void CompileX86Module(const ir::Module& module);

  explicit Compiler(const Target& target)
      : target_(target), engine_(ExecutionEngine::Create(ExecutionOptions())) {}

  CINN_DISALLOW_COPY_AND_ASSIGN(Compiler);

 private:
  Target target_;
  std::unique_ptr<ExecutionEngine> engine_;

  std::vector<void*> fn_ptr_;
  // only heterogeneous systems need to record device func and module
  std::vector<std::string> device_fn_name_;
  std::string device_fn_code_;
#ifdef CINN_WITH_CUDA
  std::unique_ptr<runtime::cuda::CUDAModule> cuda_module_;
#endif
#ifdef CINN_WITH_HIP
  std::unique_ptr<runtime::hip::HIPModule> hip_module_;
#endif
#ifdef CINN_WITH_SYCL
  std::unique_ptr<runtime::sycl::SYCLModule> sycl_module_;
#endif
};

}  // namespace backends
}  // namespace cinn
