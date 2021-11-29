// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/infrt/kernel/basic_kernels.h"

#include <iostream>
#include <string>

#include "llvm/Support/raw_ostream.h"
#include "paddle/infrt/host_context/kernel_registry.h"
#include "paddle/infrt/host_context/kernel_utils.h"

using infrt::host_context::Attribute;

namespace infrt::kernel {

template <typename T>
T add(T a, T b) {
  return a + b;
}

template <typename T>
T sub(T a, T b) {
  return a - b;
}

template <typename T>
T mul(T a, T b) {
  return a * b;
}

template <typename T>
T div(T a, T b) {
  return a / b;
}

template <typename T>
void print(T a) {
  std::cout << a << std::endl;
}

static std::string GetString(Attribute<std::string> value) {
  return value.get();
}

static void PrintString(const std::string &str) {
  llvm::outs() << "string = " << str << '\n';
  llvm::outs().flush();
}

void RegisterBasicKernels(host_context::KernelRegistry *registry) {
  RegisterIntBasicKernels(registry);
  RegisterFloatBasicKernels(registry);
  registry->AddKernel("cinn.get_string", CINN_KERNEL(GetString));
  registry->AddKernel("cinn.print_string", CINN_KERNEL(PrintString));
}

void RegisterIntBasicKernels(host_context::KernelRegistry *registry) {
  registry->AddKernel("cinn.add.i32", CINN_KERNEL(add<int32_t>));
  registry->AddKernel("cinn.sub.i32", CINN_KERNEL(sub<int32_t>));
  registry->AddKernel("cinn.mul.i32", CINN_KERNEL(mul<int32_t>));
  registry->AddKernel("cinn.div.i32", CINN_KERNEL(div<int32_t>));
  registry->AddKernel("cinn.print.i32", CINN_KERNEL(print<int32_t>));
}

void RegisterFloatBasicKernels(host_context::KernelRegistry *registry) {
  registry->AddKernel("cinn.add.f32", CINN_KERNEL(add<float>));
  registry->AddKernel("cinn.sub.f32", CINN_KERNEL(sub<float>));
  registry->AddKernel("cinn.mul.f32", CINN_KERNEL(mul<float>));
  registry->AddKernel("cinn.div.f32", CINN_KERNEL(div<float>));
  registry->AddKernel("cinn.print.f32", CINN_KERNEL(print<float>));
}

}  // namespace infrt::kernel
