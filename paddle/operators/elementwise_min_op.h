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

#include "paddle/operators/elementwise_op_function.h"

namespace paddle {
namespace operators {

template <typename T>
struct MinFunctor {
  inline HOSTDEVICE T operator()(T a, T b) const { return a < b ? a : b; }
};

template <typename DeviceContext, typename T>
class ElementwiseMinKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    using Tensor = framework::Tensor;

    auto* x = ctx.Input<Tensor>("X");
    auto* y = ctx.Input<Tensor>("Y");
    auto* z = ctx.Output<Tensor>("Out");
    z->mutable_data<T>(ctx.GetPlace());
    TransformFunctor<MinFunctor<T>, T, DeviceContext> functor(
        x, y, z, ctx.template device_context<DeviceContext>(), MinFunctor<T>());

    auto x_dims = x->dims();
    auto y_dims = y->dims();
    PADDLE_ENFORCE_GE(x_dims.size(), y_dims.size(),
                      "Rank of first input must >= rank of second input.");

    if (x_dims == y_dims) {
      functor.Run();
      return;
    }

    int axis = ctx.Attr<int>("axis");
    axis = (axis == -1 ? x_dims.size() - y_dims.size() : axis);
    PADDLE_ENFORCE(axis >= 0 && axis < x_dims.size(),
                   "Axis should be in range [0, x_dims)");

    int pre, n, post;
    get_mid_dims(x_dims, y_dims, axis, pre, n, post);
    if (post == 1) {
      functor.RunRowWise(n, pre);
      return;
    } else {
      functor.RunMidWise(n, pre, post);
      return;
    }
  }
};

template <typename T>
struct ElementwiseMinGradFunctor {
  template <typename Device, typename X, typename Y, typename Z, typename dX,
            typename dY, typename dZ>
  void operator()(Device d, X x, Y y, Z z, dX dx, dY dy, dZ dz) {
    auto x_e = framework::EigenVector<T>::Flatten(*x);
    auto y_e = framework::EigenVector<T>::Flatten(*y);
    auto dz_e = framework::EigenVector<T>::Flatten(*dz);

    if (dx) {
      auto dx_e = framework::EigenVector<T>::Flatten(*dx);
      dx_e.device(d) = (x_e < y_e).template cast<T>() * dz_e;
    }
    if (dy) {
      auto dy_e = framework::EigenVector<T>::Flatten(*dy);
      dy_e.device(d) = (x_e >= y_e).template cast<T>() * dz_e;
    }
  }
};

template <typename T>
struct ElementwiseMinBroadCastGradFunctor {
  template <typename Device, typename X, typename Y, typename Z, typename dX,
            typename dY, typename dZ, typename Pre, typename N>
  void operator()(Device d, X x, Y y, Z z, dX dx, dY dy, dZ dz, Pre pre, N n) {
    auto x_e = framework::EigenVector<T>::Flatten(*x);
    auto y_e = framework::EigenVector<T>::Flatten(*y);
    auto dz_e = framework::EigenVector<T>::Flatten(*dz);

    auto y_e_bcast = y_e.reshape(Eigen::DSizes<int, 2>(1, n))
                         .broadcast(Eigen::DSizes<int, 2>(pre, 1))
                         .reshape(Eigen::DSizes<int, 1>(x_e.size()));

    if (dx) {
      auto dx_e = framework::EigenVector<T>::Flatten(*dx);
      dx_e.device(d) = (x_e < y_e_bcast).template cast<T>() * dz_e;
    }

    if (dy) {
      auto dy_e = framework::EigenVector<T>::Flatten(*dy);
      dy_e.device(d) = ((x_e >= y_e_bcast).template cast<T>() * dz_e)
                           .reshape(Eigen::DSizes<int, 2>(pre, n))
                           .sum(Eigen::array<int, 1>{{0}});
    }
  }
};

template <typename T>
struct ElementwiseMinBroadCast2GradFunctor {
  template <typename Device, typename X, typename Y, typename Z, typename dX,
            typename dY, typename dZ, typename Pre, typename N, typename Post>
  void operator()(Device d, X x, Y y, Z z, dX dx, dY dy, dZ dz, Pre pre, N n,
                  Post post) {
    auto x_e = framework::EigenVector<T>::Flatten(*x);
    auto y_e = framework::EigenVector<T>::Flatten(*y);
    auto dz_e = framework::EigenVector<T>::Flatten(*dz);

    auto y_e_bcast = y_e.reshape(Eigen::DSizes<int, 3>(1, n, 1))
                         .broadcast(Eigen::DSizes<int, 3>(pre, 1, post))
                         .reshape(Eigen::DSizes<int, 1>(x_e.size()));
    if (dx) {
      auto dx_e = framework::EigenVector<T>::Flatten(*dx);
      dx_e.device(d) = (x_e < y_e_bcast).template cast<T>() * dz_e;
    }

    if (dy) {
      auto dy_e = framework::EigenVector<T>::Flatten(*dy);
      dy_e.device(d) = ((x_e >= y_e_bcast).template cast<T>() * dz_e)
                           .reshape(Eigen::DSizes<int, 3>(pre, n, post))
                           .sum(Eigen::array<int, 2>{{0, 2}});
    }
  }
};

template <typename DeviceContext, typename T>
class ElementwiseMinGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    ElementwiseGradCompute<DeviceContext, T, ElementwiseMinGradFunctor<T>,
                           ElementwiseMinBroadCastGradFunctor<T>,
                           ElementwiseMinBroadCast2GradFunctor<T>>(ctx);
  }
};

}  // namespace operators
}  // namespace paddle
