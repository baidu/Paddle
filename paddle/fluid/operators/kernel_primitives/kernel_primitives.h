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

#pragma once
#ifdef PADDLE_WITH_XPU
#include "paddle/fluid/operators/kernel_primitives/compute_primitives_xpu2.h"
#include "paddle/fluid/operators/kernel_primitives/datamover_primitives_xpu2.h"
#define BLOCK_ID_X cluster_id()
#define BLOCK_NUM_X core_num()
#define GRID_NUN_X cluster_num()
#else
#include "paddle/fluid/operators/kernel_primitives/compute_primitives.h"
#include "paddle/fluid/operators/kernel_primitives/datamover_primitives.h"
#define BLOCK_ID_X blockIdx.x
#define GRID_NUN_X gridDim.x
#define BLOCK_NUM_X blockDim.x
#endif
#include "paddle/fluid/operators/kernel_primitives/functor_primitives.h"
#include "paddle/fluid/operators/kernel_primitives/helper_primitives.h"

namespace paddle {
namespace operators {
namespace kernel_primitives {}
}
}
