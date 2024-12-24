/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include "paddle/phi/common/int_array.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/meta_tensor.h"

namespace phi {

// Common InferMeta Functions for 0-nary operators(no input tensor), The format
// like:
//
//   1. void [FunctionDesc|OpName]InferMeta(..., MetaTensor* out)
//
// NOTE: The name "InferShape" may be not appropriate. "InferMeta" may be good.
//   Because functions in this file not only can infer shape, but also need
//   infer lod or other useful data.
//
// The InferMeta Functions in this file are arranged in alphabetic order.

void ArangeInferMeta(const Scalar& start,
                     const Scalar& end,
                     const Scalar& step,
                     DataType dtype,
                     MetaTensor* out);

void AssignValueInferMeta(const std::vector<int>& shape,
                          DataType dtype,
                          MetaTensor* out);

void CommInitAllInferMeta(const std::vector<int>& devices, int ring_id);

void CreateVecShapeInferMeta(const std::vector<int64_t>& shape,
                             DataType dtype,
                             MetaTensor* out);

void CreateArrayInferMeta(DataType dtype, MetaTensor* out);

TEST_API void CreateInferMeta(const IntArray& shape,
                              DataType dtype,
                              MetaTensor* out,
                              MetaConfig config = MetaConfig());

void CreateInferMetaBase(const std::vector<int64_t>& shape,
                         DataType dtype,
                         DataLayout layout,
                         MetaTensor* out);

void DataInferMeta(const std::string& name,
                   const phi::IntArray& shape,
                   phi::DataType data_type,
                   MetaTensor* out);

void EyeInferMeta(const Scalar& num_rows,
                  const Scalar& num_columns,
                  DataType dtype,
                  MetaTensor* out,
                  MetaConfig config = MetaConfig());

void GaussianInferMeta(const IntArray& shape,
                       float mean,
                       float std,
                       int seed,
                       DataType dtype,
                       MetaTensor* out);

void LoadInferMeta(MetaTensor* out, MetaConfig config = MetaConfig());

void RandpermInferMeta(int n, DataType dtype, MetaTensor* out);

void RandintInferMeta(
    int low, int high, const IntArray& shape, DataType dtype, MetaTensor* out);

void PartialRecvInferMeta(int peer,
                          DataType dtype,
                          const std::vector<int>& out_shape,
                          int num,
                          int id,
                          MetaTensor* out);

void PRecvInferMeta(const int peer,
                    DataType dtype,
                    const std::vector<int>& out_shape,
                    const bool dynamic_shape,
                    MetaTensor* out);

void PRecvArrayInferMeta(int peer,
                         DataType dtype,
                         const std::vector<int>& out_shape,
                         MetaTensor* out);

void SeedInferMeta(int seed, MetaTensor* out);

void TruncatedGaussianRandomInferMeta(const std::vector<int>& shape,
                                      float mean,
                                      float std,
                                      int seed,
                                      float a,
                                      float b,
                                      DataType dtype,
                                      MetaTensor* out);

void UniformRandomInferMeta(const IntArray& shape,
                            DataType dtype,
                            MetaTensor* out);

void TrilIndicesInferMeta(
    int rows, int cols, int offset, DataType dtype, MetaTensor* out);

void TriuIndicesInferMeta(
    int row, int col, int offset, DataType dtype, MetaTensor* out);

void ReadFileInferMeta(const std::string& filename, MetaTensor* out);

}  // namespace phi
