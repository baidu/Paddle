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
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Arange)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(AssignValue)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(AssignValue_)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(CudnnLstm)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Data)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Empty)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Eye)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Feed)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Full)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Full_)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(FullIntArray)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Gaussian)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Randint)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Randperm)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(ReadFile)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Seed)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(PRecv)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(TrilIndices)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(TriuIndices)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(TruncatedGaussianRandom)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Uniform)
}  // namespace paddle::dialect
