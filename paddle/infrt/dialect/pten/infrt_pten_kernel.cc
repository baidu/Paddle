// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/infrt/dialect/pten/infrt_pten_kernel.h"

#include <mlir/IR/BuiltinTypes.h>

#include "paddle/infrt/dialect/dense_tensor.h"
#define GET_OP_CLASSES
#include "paddle/infrt/dialect/pten/infrt_pten_kernel.h.inc"
#include "paddle/infrt/dialect/pten/infrt_pten_kernelDialect.cpp.inc"

namespace infrt {
namespace pten {

void PTENKernelDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "paddle/infrt/dialect/pten/infrt_pten_kernel.cpp.inc"
      >();
}

}  // namespace pten
}  // namespace infrt

#define GET_OP_CLASSES
#include "paddle/infrt/dialect/pten/infrt_pten_kernel.cpp.inc"  // NOLINT
