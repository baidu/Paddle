// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/pir/include/dialect/shape/utils/shape_analysis.h"

namespace paddle::dialect {
OP_DECLARE_INFER_SYMBOLIC_SHAPE(All)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Amax)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Amin)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Any)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Argmax)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Argmin)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(AsComplex)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(AsReal)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Cummax)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Cummin)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Cumprod)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Cumprod_)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Cumsum)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Cumsum_)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(DiagEmbed)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Diagonal)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(DistributeFpnProposals)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(FftC2c)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(FftC2r)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(FftR2c)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(FillDiagonal)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(FillDiagonal_)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Flatten)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Flatten_)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Kthvalue)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Logcumsumexp)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Logsumexp)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Max)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Min)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Nonzero)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Pad)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Pad3d)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Prod)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(RepeatInterleave)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Reshape)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Reshape_)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Shape)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(ShapeSr)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Slice)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Split)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(SplitWithNum)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Squeeze)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Squeeze_)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Sum)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Tile)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Topk)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(TopkV1)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Transpose)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Transpose_)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Unbind)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Unique)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(UniqueConsecutive)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Unsqueeze)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Unsqueeze_)

}  // namespace paddle::dialect
