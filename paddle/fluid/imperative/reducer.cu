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

#include "paddle/fluid/imperative/reducer.h"

namespace paddle {
namespace imperative {

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
void Group::DivNRanks(phi::DenseTensor *tensor,
                      int64_t nranks,
                      const phi::DeviceContext &context) {
#ifdef PADDLE_WITH_HIP
  if (dtype_ == paddle::framework::proto::VarType_Type_BF16) {
    PADDLE_THROW(
        common::errors::Fatal("Unsupported BF16 in DataParallel for now"));
  }
  framework::VisitDataTypeForHIP(
      dtype_, DivNRanksForAllReduce<phi::GPUContext>(tensor, nranks, context));
#else
  framework::VisitDataType(
      dtype_, DivNRanksForAllReduce<phi::GPUContext>(tensor, nranks, context));
#endif
}
#endif

}  // namespace imperative
}  // namespace paddle
