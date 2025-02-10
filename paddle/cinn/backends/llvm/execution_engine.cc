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

#include "paddle/cinn/backends/llvm/execution_engine.h"

#include <absl/strings/string_view.h>
#include <llvm/ADT/Triple.h>
#include <llvm/AsmParser/Parser.h>
#include <llvm/Config/llvm-config.h>
#include <llvm/ExecutionEngine/JITSymbol.h>
#include <llvm/ExecutionEngine/Orc/Core.h>
#include <llvm/ExecutionEngine/Orc/ExecutionUtils.h>
#include <llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/InitializePasses.h>
#include <llvm/PassRegistry.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/Host.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/TargetRegistry.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Transforms/InstCombine/InstCombine.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Scalar/GVN.h>
#include <llvm/Transforms/Scalar/NewGVN.h>
#include <llvm/Transforms/Scalar/Reassociate.h>
#include <llvm/Transforms/Scalar/SimplifyCFG.h>

#include <cmath>
#include <memory>
#include <mutex>  // NOLINT
#include <string>
#include <utility>

#include "paddle/cinn/backends/codegen_cuda_host.h"
#include "paddle/cinn/backends/llvm/cinn_runtime_llvm_ir.h"
#include "paddle/cinn/backends/llvm/codegen_llvm.h"
#include "paddle/cinn/backends/llvm/codegen_x86.h"
#include "paddle/cinn/backends/llvm/llvm_optimizer.h"
#include "paddle/cinn/backends/llvm/llvm_util.h"
#include "paddle/cinn/backends/llvm/runtime_symbol_registry.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/runtime/intrinsic.h"
#include "paddle/cinn/utils/profiler.h"

namespace cinn::backends {
namespace {
void InitializeLLVMPasses() {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  auto &registry = *llvm::PassRegistry::getPassRegistry();
  llvm::initializeCore(registry);
  llvm::initializeTransformUtils(registry);
  llvm::initializeScalarOpts(registry);
  llvm::initializeIPO(registry);
  llvm::initializeInstCombine(registry);
  llvm::initializeAggressiveInstCombine(registry);
  llvm::initializeAnalysis(registry);
  llvm::initializeVectorization(registry);
  llvm::initializeSROALegacyPassPass(registry);

  // llvm::initializeCodeGen(registry);
  // llvm::initializeTarget(registry);
  // llvm::initializeCodeGenPreparePass(registry);
}
}  // namespace
void NaiveObjectCache::notifyObjectCompiled(const llvm::Module *m,
                                            llvm::MemoryBufferRef obj_buffer) {
  cached_objects_[m->getModuleIdentifier()] =
      llvm::MemoryBuffer::getMemBufferCopy(obj_buffer.getBuffer(),
                                           obj_buffer.getBufferIdentifier());
}

std::unique_ptr<llvm::MemoryBuffer> NaiveObjectCache::getObject(
    const llvm::Module *m) {
  auto it = cached_objects_.find(m->getModuleIdentifier());
  if (it == cached_objects_.end()) {
    VLOG(1) << "No object for " << m->getModuleIdentifier()
            << " in cache. Compiling.";
    return nullptr;
  }

  VLOG(3) << "Object for " << m->getModuleIdentifier() << " loaded from cache.";
  return llvm::MemoryBuffer::getMemBuffer(it->second->getMemBufferRef());
}

/*static*/ std::unique_ptr<ExecutionEngine> ExecutionEngine::Create(
    const ExecutionOptions &config) {
  VLOG(6) << "===================== Create CINN ExecutionEngine begin "
             "====================";
  VLOG(6) << "initialize llvm config";
  VLOG(6) << "llvm version: " << LLVM_VERSION_STRING;
  VLOG(6) << "llvm default target triple: " << LLVM_DEFAULT_TARGET_TRIPLE;

  static std::once_flag flag;
  std::call_once(flag, InitializeLLVMPasses);

  auto engine = std::make_unique<ExecutionEngine>(/*enable_object_cache=*/true);

  auto compile_layer_creator =
      [&engine](llvm::orc::JITTargetMachineBuilder jtmb)
      -> llvm::Expected<
          std::unique_ptr<llvm::orc::IRCompileLayer::IRCompiler>> {
    auto machine = llvm::cantFail(jtmb.createTargetMachine());
    VLOG(6) << "create llvm compile layer";
    VLOG(6) << "Target Name: " << machine->getTarget().getName();
    VLOG(6) << "Target CPU: " << machine->getTargetCPU().str() << std::endl;
    return std::make_unique<llvm::orc::TMOwningSimpleCompiler>(
        std::move(machine), engine->cache_.get());
  };

  auto object_layer_creator = [&](llvm::orc::ExecutionSession &session,
                                  const llvm::Triple &triple) {
    auto object_layer = std::make_unique<llvm::orc::RTDyldObjectLinkingLayer>(
        session,
        []() { return std::make_unique<llvm::SectionMemoryManager>(); });
    llvm::orc::JITDylib *main_jd = session.getJITDylibByName("<main>");
    if (!main_jd) {
      main_jd = &llvm::cantFail(session.createJITDylib("<main>"));
    }
    return object_layer;
  };

  VLOG(6) << "create jit execution engine";
  engine->jit_ =
      llvm::cantFail(llvm::orc::LLJITBuilder()
                         .setCompileFunctionCreator(compile_layer_creator)
                         .setObjectLinkingLayerCreator(object_layer_creator)
                         .create());
  engine->jit_->getMainJITDylib().addGenerator(llvm::cantFail(
      llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
          engine->jit_->getDataLayout().getGlobalPrefix())));

  VLOG(6) << "register global runtime call symbols";

  engine->RegisterGlobalRuntimeSymbols();

  VLOG(6) << "===================== Create CINN ExecutionEngine end "
             "====================";
  engine->ctx = std::make_unique<llvm::LLVMContext>();
  engine->b = std::make_unique<llvm::IRBuilder<>>(*engine->ctx);
  llvm::SMDiagnostic error;
  engine->m = llvm::parseAssemblyString(
      AsStringRef(backends::kRuntimeLlvmIr), error, *engine->ctx);

  return engine;
}

template <typename CodeGenT>
void ExecutionEngine::Link(const ir::Module &module) {
  if (module.functions().size() == 0) {
    return;
  }
  utils::RecordEvent("ExecutionEngine Link", utils::EventType::kOrdinary);
  auto ir_emitter = std::make_unique<CodeGenT>(m.get(), b.get());
  VLOG(3) << "ir_emitter->Compile(module) Begin";
  ir_emitter->Compile(module);
  VLOG(3) << "ir_emitter->Compile(module) Succeed!";
  PADDLE_ENFORCE_EQ(
      !llvm::verifyModule(*m, &llvm::errs()),
      true,
      ::common::errors::InvalidArgument("Sorry,Invalid module found"));
  auto machine = std::move(llvm::cantFail(
      llvm::cantFail(llvm::orc::JITTargetMachineBuilder::detectHost())
          .createTargetMachine()));
  LLVMModuleOptimizer optimize(machine.get(), 3, {}, true);
  // optimize(m.get());
  PADDLE_ENFORCE_EQ(
      !llvm::verifyModule(*m, &llvm::errs()),
      true,
      ::common::errors::InvalidArgument("Invalid optimized module detected"));
  for (auto &f : *m) {
    VLOG(5) << "function: " << DumpToString(f);
  }

  // llvm::raw_svector_ostream rawstream(buffer_);
  // llvm::legacy::PassManager pass_manager;
  // machine->addPassesToEmitFile(
  //     pass_manager, rawstream, nullptr, llvm::CGFT_ObjectFile);
  // pass_manager.run(*m);

  if (VLOG_IS_ON(5)) {
    VLOG(5) << "======= dump jit execution session ======";
    std::string buffer;
    llvm::raw_string_ostream os(buffer);
    decltype(auto) es = jit_->getExecutionSession();
    es.dump(os);
    os.flush();
    VLOG(5) << buffer;
  }
}

template <>
void ExecutionEngine::Link<CodeGenGpuHost>(const ir::Module &module) {
  if (module.functions().size() == 0) {
    return;
  }
  utils::RecordEvent("ExecutionEngine Link", utils::EventType::kOrdinary);
  auto ir_emitter = std::make_unique<CodeGenGpuHost>(m.get(), b.get());
  ir_emitter->Compile(module);
}

bool ExecutionEngine::AddModule(std::unique_ptr<llvm::Module> module,
                                std::unique_ptr<llvm::LLVMContext> context) {
  utils::RecordEvent("ExecutionEngine AddModule", utils::EventType::kOrdinary);
  module->setDataLayout(jit_->getDataLayout());
  if (VLOG_IS_ON(5)) {
    VLOG(5) << "======= dump jit lib ==========";
    std::string buffer;
    llvm::raw_string_ostream os(buffer);
    module->print(os, {});
    // main_jd_->dump(os);
    os.flush();
    VLOG(5) << buffer;
  }
  llvm::orc::ThreadSafeContext tsc(std::move(context));
  llvm::orc::ThreadSafeModule tsm(std::move(module), std::move(tsc));
  llvm::cantFail(jit_->addIRModule(std::move(tsm)));
  return true;
}

void ExecutionEngine::RegisterModuleRuntimeSymbols(
    RuntimeSymbols &&module_symbols) {
  module_symbols_ = std::forward<RuntimeSymbols>(module_symbols);
  auto *session = &jit_->getExecutionSession();
  for (const auto &sym : module_symbols_.All()) {
    VLOG(3) << "Add symbol: {" << sym.first << ":" << sym.second << "}";
    llvm::cantFail(jit_->define(llvm::orc::absoluteSymbols(
        {{session->intern(sym.first),
          {llvm::pointerToJITTargetAddress(sym.second),
           llvm::JITSymbolFlags::Exported}}})));
  }
}

bool ExecutionEngine::AddSelfModule() {
  return AddModule(std::move(m), std::move(ctx));
}

void ExecutionEngine::ExportObject(const std::string &path) {
  FILE *of = fopen(path.c_str(), "w");
  fwrite(buffer_.data(), 1, buffer_.size(), of);
  fclose(of);
}

void *ExecutionEngine::Lookup(absl::string_view name) {
  utils::RecordEvent("ExecutionEngine Lookup", utils::EventType::kOrdinary);
  std::lock_guard<std::mutex> lock(mu_);
  if (auto symbol = jit_->lookup(AsStringRef(name))) {
    return reinterpret_cast<void *>(symbol->getAddress());
  }

  LOG(ERROR) << "Unknown symbol name[" << name << "]";
  return nullptr;
}

void ExecutionEngine::RegisterGlobalRuntimeSymbols() {
  utils::RecordEvent("ExecutionEngine RegisterGlobalRuntimeSymbols",
                     utils::EventType::kOrdinary);
  const auto &registry = GlobalSymbolRegistry::Global();
  auto *session = &jit_->getExecutionSession();
  for (const auto &sym : registry.All()) {
    llvm::cantFail(jit_->define(llvm::orc::absoluteSymbols(
        {{session->intern(sym.first),
          {llvm::pointerToJITTargetAddress(sym.second),
           llvm::JITSymbolFlags::None}}})));
  }
}

template void ExecutionEngine::Link<CodeGenLLVM>(const ir::Module &module);
template void ExecutionEngine::Link<CodeGenX86>(const ir::Module &module);
template void ExecutionEngine::Link<CodeGenGpuHost>(const ir::Module &module);
template void ExecutionEngine::Link<CodeGenSwitchHost>(
    const ir::Module &module);

}  // namespace cinn::backends
