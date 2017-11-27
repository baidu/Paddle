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

#include "paddle/operators/math/unpooling.h"

namespace paddle {
namespace operators {
namespace math {

// All tensors are in NCHW format
template <typename T, typename T2>
class Unpool2dMaxFunctor<platform::CPUPlace, T, T2> {
 public:
  void operator()(const platform::DeviceContext& context,
                  const framework::Tensor& input,
                  const framework::Tensor& indices,
                  framework::Tensor * output) {
    const int batch_size = input.dims()[0];
    const int input_height = input.dims()[2];
    const int input_width = input.dims()[3];
    const int output_channels = output->dims()[1];
    const int output_height = output->dims()[2];
    const int output_width = output->dims()[3];
    int input_feasize = input_height * input_width;
    int output_feasize = output_height * output_width;
    const T* input_data = input.data<T>();
    const T2 * indices_data = indices.data<T2>();
    T* output_data = output->mutable_data<T>(context.GetPlace());
    for (int b = 0; b < batch_size; ++b) {
      for (int c = 0; c < output_channels; ++c) {
        for (int i = 0; i < input_feasize; ++i) {
          int index =  indices_data[i];
          PADDLE_ENFORCE(index < output_feasize, "err index in unpooling!");
          output_data[index] = input_data[i];
        }
        input_data += input_feasize;
        indices_data += input_feasize;
        output_data += output_feasize;
      }
    }
  }
};



template <class T, typename T2>
class Unpool2dMaxGradFunctor<platform::CPUPlace, T, T2> {
public:
  void operator()(const platform::DeviceContext& context,
                  const framework::Tensor& input,
                  const framework::Tensor& indices,
                  const framework::Tensor& output,
                  const framework::Tensor& output_grad,
                  framework::Tensor * input_grad) {
    const int batch_size = input.dims()[0];
    const int input_height = input.dims()[2];
    const int input_width = input.dims()[3];
    const int output_channels = output.dims()[1];
    const int output_height = output.dims()[2];
    const int output_width = output.dims()[3];
    int input_feasize = input_height * input_width;
    int output_feasize = output_height * output_width;
    const T2 * indices_data = indices.data<T2>();
    const T* output_grad_data = output_grad.data<T>();
    T* input_grad_data = input_grad->mutable_data<T>(context.GetPlace());

    for (int b = 0; b < batch_size; ++b) {
      for (int c = 0; c < output_channels; ++c) {
        for (int i = 0; i < input_feasize; ++i) {
          int index = indices_data[i];
          PADDLE_ENFORCE(index < output_feasize, "err index in unpooling!");
          input_grad_data[i] = output_grad_data[index];
        }
        input_grad_data += input_feasize;
        indices_data += input_feasize;
        output_grad_data += output_feasize;
      }
    }
  }
};

template class Unpool2dMaxGradFunctor<platform::CPUPlace, float, int>;
template class Unpool2dMaxGradFunctor<platform::CPUPlace, double, int>;
template class Unpool2dMaxFunctor<platform::CPUPlace, float, int>;
template class Unpool2dMaxFunctor<platform::CPUPlace, double, int>;

}  // namespace math
}  // namespace operators
}  // namespace paddle
