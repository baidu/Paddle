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

#include <gtest/gtest.h>
#include <stdlib.h>

#include <fstream>
#include <tuple>
#include <vector>

#include "paddle/cinn/backends/codegen_cuda_dev.h"
#include "paddle/cinn/backends/codegen_cuda_host.h"
#include "paddle/cinn/backends/codegen_device_util.h"
#include "paddle/cinn/backends/extern_func_jit_register.h"
#include "paddle/cinn/backends/llvm/execution_engine.h"
#include "paddle/cinn/backends/llvm/simple_jit.h"
#include "paddle/cinn/cinn.h"
#include "paddle/cinn/common/ir_util.h"
#include "paddle/cinn/common/test_helper.h"
#include "paddle/cinn/hlir/pe/nn.h"
#include "paddle/cinn/hlir/pe/schedule.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/schedule/ir_schedule.h"
#include "paddle/cinn/lang/lower.h"
#include "paddle/cinn/utils/timer.h"
#include "paddle/common/enforce.h"

namespace cinn {
namespace backends {

TEST(CUDAFile, Module_output) {
  std::string cuda_source_name = "_generated1.cu";
  std::string cuda_source_code = R"ROC(
extern "C" {

__global__
void __launch_bounds__(200) elementwise_mul(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C)
{
  if (((int)blockIdx.x < 100)) {
    if (((int)threadIdx.x < 200)) {
      C[((200 * (int)blockIdx.x) + (int)threadIdx.x)] = (A[((200 * (int)blockIdx.x) + (int)threadIdx.x)] * B[((200 * (int)blockIdx.x) + (int)threadIdx.x)]);
    };
  };
}

}
  )ROC";
  std::ofstream file(cuda_source_name);
  PADDLE_ENFORCE_EQ(file.is_open(),
                    true,
                    ::common::errors::Unavailable(
                        "Failed to open file: %s. Please check if the file "
                        "path is correct and the file is accessible.",
                        cuda_source_name));
  file << CodeGenCudaDev::GetSourceHeader();
  file << cuda_source_code;
  file.close();
  LOG(WARNING) << "Output C source to file " << cuda_source_name;
}

}  // namespace backends
}  // namespace cinn
