/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/math/fc.h"
#include "paddle/fluid/operators/jit/kernels.h"
#include "paddle/fluid/operators/math/blas.h"

namespace paddle {
namespace operators {
namespace math {

template <typename T>
class FCFunctor<platform::CPUDeviceContext, T> {
 public:
  void operator()(const platform::CPUDeviceContext& context, const int M,
                  const int N, const int K, const T* X, const T* W, T* Y,
                  const T* B = nullptr, bool relu = false) {
    auto blas = math::GetBlas<platform::CPUDeviceContext, T>(context);
    framework::Tensor Y1;
    Y1.Resize({M * (N + 4)});
    T* Y1_data = Y1.mutable_data<T>(platform::CPUPlace());
    framework::Tensor X1;
    X1.Resize({M * (K + 4)});
    T* X1_data = X1.mutable_data<T>(platform::CPUPlace());
    if (N % 128 == 0 && K % 128 == 0) {
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
      for (int i = 0; i < M; i++) {
        memcpy(X1_data + i * (K + 4), X + i * K, K * sizeof(X[0]));
      }

      blas.MatMul(M, N, K, X1_data, W, Y1_data);
    } else {
      blas.MatMul(M, N, K, X, W, Y);
    }

    if (B == NULL) {
      return;
    }
    if (relu) {
      auto compute =
          jit::KernelFuncs<jit::VAddReluTuple<T>, platform::CPUPlace>::Cache()
              .At(N);
      for (int i = 0; i < M; i++) {
        T* dst = Y + i * N;
        compute(B, dst, dst, N);
      }
    } else {
      auto compute =
          jit::KernelFuncs<jit::VAddTuple<T>, platform::CPUPlace>::Cache().At(
              N);

#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
      for (int i = 0; i < M; i++) {
        T* dst = Y + i * N;
        T* src = nullptr;
        if (M % 128 == 0 && N % 128 == 0 && K % 128 == 0)
          src = Y1_data + i * (N + 4);
        else
          src = dst;
        compute(B, src, dst, N);
      }
    }
  }
};

template class FCFunctor<platform::CPUDeviceContext, float>;
template class FCFunctor<platform::CPUDeviceContext, double>;

}  // namespace math
}  // namespace operators
}  // namespace paddle
