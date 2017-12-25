/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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
#include "paddle/framework/data_type.h"
#include "paddle/operators/math/math_function.h"

namespace paddle {
namespace operators {
namespace math {

template <typename DeviceContext, typename T>
void SetConstant<DeviceContext, T>::operator()(const DeviceContext& context,
                                               framework::Tensor* tensor,
                                               T num) {
  auto t = framework::EigenVector<T>::Flatten(*tensor);
  t.device(*context.eigen_device()) = t.constant(static_cast<T>(num));
}

template <typename DeviceContext, typename T, int Rank>
void Transpose<DeviceContext, T, Rank>::operator()(
    const DeviceContext& context, const framework::Tensor& in,
    framework::Tensor* out, const std::vector<int>& axis) {
  Eigen::array<int, Rank> permute;
  for (int i = 0; i < Rank; i++) {
    permute[i] = axis[i];
  }
  auto in_dim = in.dims();
  auto out_dim = out->dims();

  auto eigen_in = framework::EigenTensor<T, Rank>::From(in);
  auto eigen_out = framework::EigenTensor<T, Rank>::From(*out);
  auto* dev = context.eigen_device();
  eigen_out.device(*dev) = eigen_in.shuffle(permute);
}

template <typename DeviceContext, typename T>
void RowwiseAdd<DeviceContext, T>::operator()(const DeviceContext& context,
                                              const framework::Tensor& input,
                                              const framework::Tensor& vector,
                                              framework::Tensor* output) {
  auto in_dims = input.dims();
  auto size = input.numel() / in_dims[0];
  PADDLE_ENFORCE_EQ(vector.numel(), size);
  PADDLE_ENFORCE_EQ(output->dims(), in_dims);

  auto in = framework::EigenMatrix<T>::From(input);
  auto vec = framework::EigenMatrix<T>::From(vector);
  auto out = framework::EigenMatrix<T>::From(*output);
  Eigen::array<int, 2> shape({{1, static_cast<int>(size)}});
  Eigen::array<int, 2> bcast({{static_cast<int>(in_dims[0]), 1}});
  out.device(*context.eigen_device()) =
      in + vec.reshape(shape).broadcast(bcast);
}

template <typename DeviceContext, typename T>
void ColwiseSum<DeviceContext, T>::operator()(const DeviceContext& context,
                                              const framework::Tensor& input,
                                              framework::Tensor* vector) {
  auto in_dims = input.dims();
  auto size = input.numel() / in_dims[0];
  PADDLE_ENFORCE_EQ(vector->numel(), size);

  auto vec = framework::EigenMatrix<T>::From(*vector);
  auto in = framework::EigenMatrix<T>::From(input);
  Eigen::array<int, 2> shape({{1, static_cast<int>(size)}});
  vec.reshape(shape).device(*context.eigen_device()) =
      in.sum(Eigen::array<int, 1>({{0}})).reshape(shape);
}

template <typename DeviceContext, typename T>
void RowwiseSum<DeviceContext, T>::operator()(const DeviceContext& context,
                                              const framework::Tensor& input,
                                              framework::Tensor* vector) {
  auto in_dims = input.dims();
  auto size = input.numel() / in_dims[1];
  PADDLE_ENFORCE_EQ(vector->numel(), size);

  auto in = framework::EigenMatrix<T, Eigen::ColMajor>::From(input);
  auto vec = framework::EigenMatrix<T, Eigen::ColMajor>::From(*vector);
  Eigen::array<int, 2> shape({{static_cast<int>(size), 1}});
  vec.reshape(shape).device(*context.eigen_device()) =
      in.sum(Eigen::array<int, 1>({{1}})).reshape(shape);
}
}  // namespace math
}  // namespace operators
}  // namespace paddle
