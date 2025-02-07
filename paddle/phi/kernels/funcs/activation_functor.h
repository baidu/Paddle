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

#pragma once

#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

#include <type_traits>

#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/extensions.h"

#ifdef PADDLE_WITH_XPU_KP
#define __forceinline__ __inline__
#endif

namespace phi {
namespace funcs {
enum ActBwdOpFwdDeps {
  kNoDeps = 0x00,  // Do not need any forward input/output
  kDepX = 0x01,    // Only need forward input X
  kDepOut = 0x02,  // Only need forward output Out
};

template <typename T>
struct BaseActivationFunctor {
  using ELEMENT_TYPE = T;

  using AttrPair = std::vector<std::pair<const char*, float*>>;

  AttrPair GetAttrs() { return AttrPair(); }
};

template <typename T>
struct Sine {
  HOSTDEVICE T operator()(const T& val) const { return sin(val); }
};

template <>
struct Sine<dtype::float16> {
  HOSTDEVICE dtype::float16 operator()(const dtype::float16& val) const {
    return dtype::float16(sin(static_cast<float>(val)));
  }
};

template <>
struct Sine<dtype::bfloat16> {
  HOSTDEVICE dtype::bfloat16 operator()(const dtype::bfloat16& val) const {
    return dtype::bfloat16(sin(static_cast<float>(val)));
  }
};

template <typename T>
struct Cosine {
  HOSTDEVICE T operator()(const T& val) const { return cos(val); }
};

template <>
struct Cosine<dtype::float16> {
  HOSTDEVICE dtype::float16 operator()(const dtype::float16& val) const {
    return dtype::float16(cos(static_cast<float>(val)));
  }
};

template <>
struct Cosine<dtype::bfloat16> {
  HOSTDEVICE dtype::bfloat16 operator()(const dtype::bfloat16& val) const {
    return dtype::bfloat16(cos(static_cast<float>(val)));
  }
};

template <typename T>
using ComplexType = phi::dtype::complex<T>;

// T is phi::dtype::complex<float> or phi::dtype::complex<double>
template <typename T>
struct Conj {
  HOSTDEVICE ComplexType<T> operator()(const ComplexType<T>& val) const {
    return ComplexType<T>(val.real, -val.imag);
  }
};

// T is phi::dtype::complex<float> or phi::dtype::complex<double>
template <typename T>
struct Real {
  HOSTDEVICE ComplexType<T> operator()(const ComplexType<T>& val) const {
    return ComplexType<T>(val.real);
  }
};

template <typename T>
struct RealCp {
  HOSTDEVICE const T operator()(const ComplexType<T>& val) const {
    return val.real;
  }
};

template <typename T>
struct ImagCp {
  HOSTDEVICE const T operator()(const ComplexType<T>& val) const {
    return val.imag;
  }
};

template <typename T>
struct ComplexAssemble {
  HOSTDEVICE ComplexAssemble() {}

  HOSTDEVICE ComplexType<T> operator()(const T& r, const T& i) const {
    return ComplexType<T>(r, i);
  }
};

// sine'(x) = cos(x)
template <typename T>
struct SinGradFunctor : public BaseActivationFunctor<T> {
  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out UNUSED, dOut dout, dX dx) const {
    dx.device(d) = dout * x.unaryExpr(Cosine<T>());
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};

template <typename T>
struct SinGradFunctor<ComplexType<T>>
    : public BaseActivationFunctor<ComplexType<T>> {
  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out UNUSED, dOut dout, dX dx) const {
    dx.device(d) =
        dout * x.unaryExpr(Cosine<ComplexType<T>>()).unaryExpr(Conj<T>());
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};
// sine(x) = sin(x)
template <typename T>
struct SinFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    // Note(GGBond8488): Since Eigen3.3, Behavior like {A = (B * A).cwiseAbs()}
    // will give wrong result, details see
    // http://eigen.tuxfamily.org/dox/group__TopicAliasing.html
    out.device(d) = x.unaryExpr(Sine<T>()).eval();
  }
};

// sine''(x) = -sin(x)
template <typename T>
struct SinDoubleGradFunctor : public BaseActivationFunctor<T> {
  template <typename Device>
  void operator()(const Device& dev,
                  const DenseTensor* X,
                  const DenseTensor* dOut,
                  const DenseTensor* ddX,
                  DenseTensor* dX,
                  DenseTensor* ddOut) const {
    auto* d = dev.eigen_device();
    auto d2d1x = EigenVector<T>::Flatten(
        GET_DATA_SAFELY(ddX, "Input", "d2d1x", "SinDoubleGrad"));
    auto x = EigenVector<T>::Flatten(
        GET_DATA_SAFELY(X, "Input", "x", "SinDoubleGrad"));

    // calculate d2x first, so d2d1y can inplace d2d1x
    auto d2x = EigenVector<T>::Flatten(
        GET_DATA_SAFELY(dX, "Output", "d2x", "SinDoubleGrad"));

    if (dX) {
      if (dOut) {
        auto d1y = EigenVector<T>::Flatten(
            GET_DATA_SAFELY(dOut, "Output", "d1y", "SinDoubleGrad"));
        d2x.device(*d) = -d2d1x * x.unaryExpr(Sine<T>()) * d1y;
      } else {
        d2x.device(*d) = -d2d1x * x.unaryExpr(Sine<T>()) * static_cast<T>(0);
      }
    }

    // calculate d2d1y
    if (ddOut) {
      auto d2d1y = EigenVector<T>::Flatten(
          GET_DATA_SAFELY(ddOut, "Output", "d2d1y", "SinDoubleGrad"));
      d2d1y.device(*d) = d2d1x * x.unaryExpr(Cosine<T>());
    }
  }
  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};

// 1st reverse grad
// y = sin(x)
// x --> y
// d1x = d1y * cos(x)
//
// 2nd reverse grad
// x, d1y --> d1x
// d2x = -sin(x) * d1y * d2d1x
// d2d1y = cos(x) * d2d1x
//
// 3rd reverse grad
// x, d1y, d2d1x --> d2x, d2d1y
// d3x = -cos(x) * d1y * d2d1x * d3d2x - sin(x) * d2d1x * d3d2d1y
// d3d1y = -sin(x) * d2d1x * d3d2x
// d3d2d1x = -sin(x) * d1y * d3d2x + cos(x) * d3d2d1y
template <typename T>
struct SinTripleGradFunctor : public BaseActivationFunctor<T> {
  template <typename Device>
  void operator()(const Device& dev,
                  const DenseTensor* X,
                  const DenseTensor* ddX,
                  const DenseTensor* dOut,
                  const DenseTensor* d_DDOut,
                  const DenseTensor* d_dx_New,
                  DenseTensor* d_d_Out,
                  DenseTensor* d_x_New,
                  DenseTensor* d_DDx) const {
    auto* d = dev.eigen_device();
    auto x = EigenVector<T>::Flatten(
        GET_DATA_SAFELY(X, "Input", "x", "SinTripleGrad"));
    auto d3d2x = EigenVector<T>::Flatten(
        GET_DATA_SAFELY(d_dx_New, "Input", "d3d2x", "SinTripleGrad"));
    if (d_x_New) {
      auto d3x = EigenVector<T>::Flatten(
          GET_DATA_SAFELY(d_x_New, "Output", "d3x", "SinTripleGrad"));
      if (dOut && ddX && d_DDOut) {
        auto d2d1x = EigenVector<T>::Flatten(
            GET_DATA_SAFELY(ddX, "Input", "d2d1x", "SinTripleGrad"));
        auto d1y = EigenVector<T>::Flatten(
            GET_DATA_SAFELY(dOut, "Input", "d1y", "SinTripleGrad"));
        auto d3d2d1y = EigenVector<T>::Flatten(
            GET_DATA_SAFELY(d_DDOut, "Input", "d3d2d1y", "SinTripleGrad"));
        d3x.device(*d) = -x.unaryExpr(Cosine<T>()) * d1y * d2d1x * d3d2x -
                         x.unaryExpr(Sine<T>()) * d2d1x * d3d2d1y;
      } else if (!dOut && ddX && d_DDOut) {
        auto d2d1x = EigenVector<T>::Flatten(
            GET_DATA_SAFELY(ddX, "Input", "d2d1x", "SinTripleGrad"));
        auto d3d2d1y = EigenVector<T>::Flatten(
            GET_DATA_SAFELY(d_DDOut, "Input", "d3d2d1y", "SinTripleGrad"));
        d3x.device(*d) = -x.unaryExpr(Sine<T>()) * d2d1x * d3d2d1y;
      } else if (dOut && ddX && !d_DDOut) {
        auto d2d1x = EigenVector<T>::Flatten(
            GET_DATA_SAFELY(ddX, "Input", "d2d1x", "SinTripleGrad"));
        auto d1y = EigenVector<T>::Flatten(
            GET_DATA_SAFELY(dOut, "Input", "d1y", "SinTripleGrad"));
        d3x.device(*d) = -x.unaryExpr(Cosine<T>()) * d1y * d2d1x * d3d2x;
      } else {
        d3x.device(*d) = x * static_cast<T>(0);
      }
    }

    if (d_d_Out) {
      auto d3d1y = EigenVector<T>::Flatten(
          GET_DATA_SAFELY(d_d_Out, "Output", "d3d1y", "SinTripleGrad"));
      if (ddX) {
        auto d2d1x = EigenVector<T>::Flatten(
            GET_DATA_SAFELY(ddX, "Input", "d2d1x", "SinTripleGrad"));
        d3d1y.device(*d) = -x.unaryExpr(Sine<T>()) * d2d1x * d3d2x;
      } else {
        d3d1y.device(*d) = static_cast<T>(0) * x;
      }
    }

    if (d_DDx) {
      auto d3d2d1x = EigenVector<T>::Flatten(
          GET_DATA_SAFELY(d_DDx, "Output", "d3d2d1x", "SinTripleGrad"));
      if (dOut && d_DDOut) {
        auto d1y = EigenVector<T>::Flatten(
            GET_DATA_SAFELY(dOut, "Input", "d1y", "SinTripleGrad"));
        auto d3d2d1y = EigenVector<T>::Flatten(
            GET_DATA_SAFELY(d_DDOut, "Input", "d3d2d1y", "SinTripleGrad"));
        d3d2d1x.device(*d) = -x.unaryExpr(Sine<T>()) * d1y * d3d2x +
                             x.unaryExpr(Cosine<T>()) * d3d2d1y;
      } else if (dOut && !d_DDOut) {
        auto d1y = EigenVector<T>::Flatten(
            GET_DATA_SAFELY(dOut, "Input", "d1y", "SinTripleGrad"));
        d3d2d1x.device(*d) = -x.unaryExpr(Sine<T>()) * d1y * d3d2x;
      } else if (!dOut && d_DDOut) {
        auto d3d2d1y = EigenVector<T>::Flatten(
            GET_DATA_SAFELY(d_DDOut, "Input", "d3d2d1y", "SinTripleGrad"));
        d3d2d1x.device(*d) = x.unaryExpr(Cosine<T>()) * d3d2d1y;
      } else {
        d3d2d1x.device(*d) = x * static_cast<T>(0);
      }
    }
  }
  static constexpr ActBwdOpFwdDeps FwdDeps() {
    return ActBwdOpFwdDeps::kDepOut;
  }
};

// reciprocal(x) = 1 / x
template <typename T>
struct ReciprocalFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    out.device(d) = static_cast<T>(1) / x;
  }
};

template <typename T>
struct ReciprocalGradFunctor : public BaseActivationFunctor<T> {
  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(Device d, X x UNUSED, Out out, dOut dout, dX dx) const {
    dx.device(d) = dout * static_cast<T>(-1) * out * out;
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() {
    return ActBwdOpFwdDeps::kDepOut;
  }
};

template <typename T>
struct ReciprocalGradFunctor<ComplexType<T>>
    : public BaseActivationFunctor<ComplexType<T>> {
  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(Device d, X x UNUSED, Out out, dOut dout, dX dx) const {
    dx.device(d) = dout * static_cast<ComplexType<T>>(-1) *
                   (out * out).unaryExpr(Conj<T>());
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() {
    return ActBwdOpFwdDeps::kDepOut;
  }
};

// 1st reverse grad
// y = cos(x)
// x --> y
// d1x = d1y * -sin(x)
//
// 2nd reverse grad
// x, d1y --> d1x
// d2x = -cos(x) * d1y * d2d1x
// d2d1y = -sin(x) * d2d1x
//
// 3rd reverse grad
// x, d1y, d2d1x --> d2x, d2d1y
// d3x = sin(x) * d1y * d2d1x * d3d2x - cos(x) * d2d1x * d3d2d1y
// d3d1y = -cos(x) * d2d1x * d3d2x
// d3d2d1x = -cos(x) * d1y * d3d2x - sin(x) * d3d2d1y

// cosine'(x) = -sin(x)
template <typename T>
struct CosGradFunctor : public BaseActivationFunctor<T> {
  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out UNUSED, dOut dout, dX dx) const {
    dx.device(d) = -dout * x.unaryExpr(Sine<T>());
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};

template <typename T>
struct CosGradFunctor<ComplexType<T>>
    : public BaseActivationFunctor<ComplexType<T>> {
  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out UNUSED, dOut dout, dX dx) const {
    dx.device(d) =
        -dout * x.unaryExpr(Sine<ComplexType<T>>()).unaryExpr(Conj<T>());
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};

// cos''(x) = -cos(x)
template <typename T>
struct CosDoubleGradFunctor : public BaseActivationFunctor<T> {
  template <typename Device>
  void operator()(const Device& dev,
                  const DenseTensor* X,
                  const DenseTensor* dOut,
                  const DenseTensor* ddX,
                  DenseTensor* dX,
                  DenseTensor* ddOut) const {
    auto* device = dev.eigen_device();
    auto d2d1x = EigenVector<T>::Flatten(
        GET_DATA_SAFELY(ddX, "Input", "d2d1x", "CosDoubleGrad"));
    auto x = EigenVector<T>::Flatten(
        GET_DATA_SAFELY(X, "Input", "x", "CosDoubleGrad"));

    // calculate d2x first, so d2d1y can inplace d2d1x
    auto d2x = EigenVector<T>::Flatten(
        GET_DATA_SAFELY(dX, "Output", "d2x", "CosDoubleGrad"));
    if (dX) {
      auto d1y = EigenVector<T>::Flatten(
          GET_DATA_SAFELY(dOut, "Output", "d1y", "CosDoubleGrad"));
      d2x.device(*device) = -d2d1x * x.unaryExpr(Cosine<T>()) * d1y;
    }

    if (ddOut) {
      // calculate d2d1y
      auto d2d1y = EigenVector<T>::Flatten(
          GET_DATA_SAFELY(ddOut, "Output", "d2d1y", "CosDoubleGrad"));
      d2d1y.device(*device) = -d2d1x * x.unaryExpr(Sine<T>());
    }
  }
  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};

template <typename T>
struct CosTripleGradFunctor : public BaseActivationFunctor<T> {
  template <typename Device>
  void operator()(const Device& dev,
                  const DenseTensor* X,
                  const DenseTensor* ddX,
                  const DenseTensor* dOut,
                  const DenseTensor* d_DDOut,
                  const DenseTensor* d_dx_New,
                  DenseTensor* d_d_Out,
                  DenseTensor* d_x_New,
                  DenseTensor* d_DDx) const {
    auto* d = dev.eigen_device();
    auto x = EigenVector<T>::Flatten(
        GET_DATA_SAFELY(X, "Input", "x", "CosTripleGrad"));
    auto d3d2x = EigenVector<T>::Flatten(
        GET_DATA_SAFELY(d_dx_New, "Input", "d3d2x", "CosTripleGrad"));

    if (d_x_New) {
      auto d3x = EigenVector<T>::Flatten(
          GET_DATA_SAFELY(d_x_New, "Output", "d3x", "CosTripleGrad"));
      if (dOut && ddX && d_DDOut) {
        auto d2d1x = EigenVector<T>::Flatten(
            GET_DATA_SAFELY(ddX, "Input", "d2d1x", "CosTripleGrad"));
        auto d1y = EigenVector<T>::Flatten(
            GET_DATA_SAFELY(dOut, "Input", "d1y", "CosTripleGrad"));
        auto d3d2d1y = EigenVector<T>::Flatten(
            GET_DATA_SAFELY(d_DDOut, "Input", "d3d2d1y", "CosTripleGrad"));
        d3x.device(*d) = x.unaryExpr(Sine<T>()) * d1y * d2d1x * d3d2x -
                         x.unaryExpr(Cosine<T>()) * d2d1x * d3d2d1y;
      } else if (dOut && ddX && !d_DDOut) {
        auto d2d1x = EigenVector<T>::Flatten(
            GET_DATA_SAFELY(ddX, "Input", "d2d1x", "CosTripleGrad"));
        auto d1y = EigenVector<T>::Flatten(
            GET_DATA_SAFELY(dOut, "Input", "d1y", "CosTripleGrad"));
        d3x.device(*d) = x.unaryExpr(Sine<T>()) * d1y * d2d1x * d3d2x;
      } else if (!dOut && ddX && d_DDOut) {
        auto d2d1x = EigenVector<T>::Flatten(
            GET_DATA_SAFELY(ddX, "Input", "d2d1x", "CosTripleGrad"));
        auto d3d2d1y = EigenVector<T>::Flatten(
            GET_DATA_SAFELY(d_DDOut, "Input", "d3d2d1y", "CosTripleGrad"));
        d3x.device(*d) = -x.unaryExpr(Cosine<T>()) * d2d1x * d3d2d1y;
      } else {
        d3x.device(*d) = static_cast<T>(0) * x;
      }
    }

    if (d_d_Out) {
      auto d3d1y = EigenVector<T>::Flatten(
          GET_DATA_SAFELY(d_d_Out, "Output", "d3d1y", "CosTripleGrad"));
      if (ddX) {
        auto d2d1x = EigenVector<T>::Flatten(
            GET_DATA_SAFELY(ddX, "Input", "d2d1x", "CosTripleGrad"));
        d3d1y.device(*d) = -x.unaryExpr(Cosine<T>()) * d2d1x * d3d2x;
      } else {
        d3d1y.device(*d) = static_cast<T>(0) * x;
      }
    }

    if (d_DDx) {
      auto d3d2d1x = EigenVector<T>::Flatten(
          GET_DATA_SAFELY(d_DDx, "Output", "d3d2d1x", "CosTripleGrad"));
      if (dOut && d_DDOut) {
        auto d1y = EigenVector<T>::Flatten(
            GET_DATA_SAFELY(dOut, "Input", "d1y", "CosTripleGrad"));
        auto d3d2d1y = EigenVector<T>::Flatten(
            GET_DATA_SAFELY(d_DDOut, "Input", "d3d2d1y", "CosTripleGrad"));
        d3d2d1x.device(*d) = -x.unaryExpr(Cosine<T>()) * d1y * d3d2x -
                             x.unaryExpr(Sine<T>()) * d3d2d1y;
      } else if (!dOut && d_DDOut) {
        auto d3d2d1y = EigenVector<T>::Flatten(
            GET_DATA_SAFELY(d_DDOut, "Input", "d3d2d1y", "CosTripleGrad"));
        d3d2d1x.device(*d) = -x.unaryExpr(Sine<T>()) * d3d2d1y;
      } else if (dOut && !d_DDOut) {
        auto d1y = EigenVector<T>::Flatten(
            GET_DATA_SAFELY(dOut, "Input", "d1y", "CosTripleGrad"));
        d3d2d1x.device(*d) = -x.unaryExpr(Cosine<T>()) * d1y * d3d2x;
      } else {
        d3d2d1x.device(*d) = static_cast<T>(0) * x;
      }
    }
  }
  static constexpr ActBwdOpFwdDeps FwdDeps() {
    return ActBwdOpFwdDeps::kDepOut;
  }
};

// cosine(x) = cos(x)
template <typename T>
struct CosFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    out.device(d) = x.unaryExpr(Cosine<T>()).eval();
  }
};

template <typename T>
struct LogitFunctor {
  template <typename Device, typename X, typename Out, typename P>
  void operator()(Device d, X x, Out out, P p, float eps) const {
    // logit(x) = ln(x/(1-x))
    auto tmp_x =
        (x.cwiseMin(static_cast<T>(1.0 - eps))).cwiseMax(static_cast<T>(eps));

    if (!eps) {
      out.device(d) = (x < static_cast<T>(0.0) || x > static_cast<T>(1.0))
                          .select(p.constant(static_cast<T>(NAN)),
                                  (tmp_x / (static_cast<T>(1) - tmp_x)).log());
    } else {
      out.device(d) = (tmp_x / (static_cast<T>(1) - tmp_x)).log();
    }
  }
};

// mish(x) = x * tanh(softplus(x))
// softplus(x) = x, if x > threshold
//             = ln(1 + exp(x)), otherwise

template <typename T>
struct MishFunctor : public BaseActivationFunctor<T> {
  float threshold;
  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"threshold", &threshold}};
  }

  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    auto sp = (x > static_cast<T>(threshold))  // NOLINT
                  .select(x, (static_cast<T>(1) + x.exp()).log());
    out.device(d) = x * sp.tanh();
  }
};

// dx = dout * (tanh(sp) + x * (1 - tanh(sp) ** 2) * (1 - exp(-sp)))
// sp = softplus(x)

template <typename T>
struct MishGradFunctor : public BaseActivationFunctor<T> {
  float threshold;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"threshold", &threshold}};
  }

  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out UNUSED, dOut dout, dX dx) const {
    auto sp = (x > static_cast<T>(threshold))  // NOLINT
                  .select(x, (static_cast<T>(1) + x.exp()).log());
    auto gsp = static_cast<T>(1) - (-sp).exp();
    auto tsp = sp.tanh();
    dx.device(d) = dout * (tsp + x * (static_cast<T>(1) - tsp * tsp) * gsp);
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct STanhFunctor : public BaseActivationFunctor<T> {
  float scale_a;
  float scale_b;
  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"scale_a", &scale_a}, {"scale_b", &scale_b}};
  }

  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    out.device(d) = static_cast<T>(scale_b) *
                    (static_cast<T>(scale_a) * x).tanh();  // NOLINT
  }
};

template <typename T>
struct STanhGradFunctor : public BaseActivationFunctor<T> {
  float scale_a;
  float scale_b;
  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"scale_a", &scale_a}, {"scale_b", &scale_b}};
  }

  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out UNUSED, dOut dout, dX dx) const {
    auto a = static_cast<T>(scale_a);  // NOLINT
    auto b = static_cast<T>(scale_b);
    auto temp = (a * x).tanh() * (a * x).tanh();
    dx.device(d) = dout * a * b * (static_cast<T>(1) - temp);
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct STanhGradFunctor<ComplexType<T>>
    : public BaseActivationFunctor<ComplexType<T>> {
  float scale_a;
  float scale_b;
  typename BaseActivationFunctor<ComplexType<T>>::AttrPair GetAttrs() {
    return {{"scale_a", &scale_a}, {"scale_b", &scale_b}};
  }

  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out UNUSED, dOut dout, dX dx) const {
    auto a = static_cast<ComplexType<T>>(scale_a);  // NOLINT
    auto b = static_cast<ComplexType<T>>(scale_b);
    auto temp = (a * x).tanh() * (a * x).tanh();
    dx.device(d) =
        dout *
        (a * b * (static_cast<ComplexType<T>>(1) - temp)).unaryExpr(Conj<T>());
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct Tangent {
  HOSTDEVICE T operator()(const T& val) const { return tan(val); }
};

template <>
struct Tangent<dtype::float16> {
  HOSTDEVICE dtype::float16 operator()(const dtype::float16& val) const {
    return dtype::float16(tan(static_cast<float>(val)));
  }
};

// Tangent'(x) = -Tangent(x)
template <typename T>
struct TanGradFunctor : public BaseActivationFunctor<T> {
  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out UNUSED, dOut dout, dX dx) const {
    dx.device(d) = dout / x.unaryExpr(Cosine<T>()).square();
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};

template <typename T>
struct TanGradFunctor<ComplexType<T>>
    : public BaseActivationFunctor<ComplexType<T>> {
  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out UNUSED, dOut dout, dX dx) const {
    // auto dx_ =
    // static_cast<ComplexType<T>>(x.unaryExpr(Cosine<T>()).square());
    // ComplexType<T> dx_conj_(dx_.real, -dx_.imag);
    // dx.device(d) = dout / dx_conj_;
    dx.device(d) =
        dout /
        x.unaryExpr(Cosine<ComplexType<T>>()).square().unaryExpr(Conj<T>());
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};

// square(x) = x^2
template <typename T>
struct SquareFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    out.device(d) = x.square();
  }
};

template <typename T>
struct SquareGradFunctor : public BaseActivationFunctor<T> {
  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out UNUSED, dOut dout, dX dx) const {
    dx.device(d) = dout * static_cast<T>(2) * x;
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct SquareGradFunctor<ComplexType<T>>
    : public BaseActivationFunctor<ComplexType<T>> {
  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out UNUSED, dOut dout, dX dx) const {
    dx.device(d) =
        dout * static_cast<ComplexType<T>>(2) * x.unaryExpr(Conj<T>());
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

// sqrt(x) = x^(1/2)
template <typename T>
struct SqrtFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    out.device(d) = x.sqrt();
  }
};

template <typename T>
struct SqrtGradFunctor : public BaseActivationFunctor<T> {
  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(Device d, X x UNUSED, Out out, dOut dout, dX dx) const {
    dx.device(d) = static_cast<T>(0.5) * dout / out;
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() {
    return ActBwdOpFwdDeps::kDepOut;
  }
};

template <typename T>
struct SqrtGradFunctor<ComplexType<T>>
    : public BaseActivationFunctor<ComplexType<T>> {
  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(Device d, X x UNUSED, Out out, dOut dout, dX dx) const {
    dx.device(d) =
        dout * (static_cast<ComplexType<T>>(0.5) / out).unaryExpr(Conj<T>());
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() {
    return ActBwdOpFwdDeps::kDepOut;
  }
};

// rsqrt(x) = x^(-1/2)
template <typename T>
struct RsqrtFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    out.device(d) = x.rsqrt();
  }
};

template <typename T>
struct RsqrtGradFunctor : public BaseActivationFunctor<T> {
  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(Device d, X x UNUSED, Out out, dOut dout, dX dx) const {
    dx.device(d) = static_cast<T>(-0.5) * dout * out * out * out;
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() {
    return ActBwdOpFwdDeps::kDepOut;
  }
};

// // For numerical stability, using the following formula instead of
// softplus(x) =
// // log(1 + exp(x))
// // softplus(x) = log(1 + exp(beta * x)) / beta when beta * x <=
// threshold(beta =
// // 1, threshold = 20 by default), otherwise x

template <typename T>
struct SoftplusFunctor : public BaseActivationFunctor<T> {
  float beta;
  float threshold;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"beta", &beta}, {"threshold", &threshold}};
  }

  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    auto x_beta = static_cast<T>(beta) * x;  // NOLINT
    out.device(d) = (x_beta > static_cast<T>(threshold))
                        .select(x,
                                (static_cast<T>(1) + x_beta.exp()).log() /
                                    static_cast<T>(beta));
  }
};

// For numerical stability, using the following formula instead of
// d(softplus(x))/dx = 1 / (1 + exp(-x))
// d(softplus(x))/dx = 1 / (1 + exp(-beta * x)) when beta * x <= threshold(beta
// = 1, threshold = 20 by default), otherwise x

template <typename T>
struct SoftplusGradFunctor : public BaseActivationFunctor<T> {
  float beta;
  float threshold;
  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"beta", &beta}, {"threshold", &threshold}};
  }
  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out UNUSED, dOut dout, dX dx) const {
    auto x_beta = static_cast<T>(beta) * x;  // NOLINT
    dx.device(d) =
        (x_beta > static_cast<T>(threshold))
            .select(dout, dout / (static_cast<T>(1) + (-x_beta).exp()));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct SoftplusGradFunctor<ComplexType<T>>
    : public BaseActivationFunctor<ComplexType<T>> {
  float beta;
  float threshold;
  typename BaseActivationFunctor<ComplexType<T>>::AttrPair GetAttrs() {
    return {{"beta", &beta}, {"threshold", &threshold}};
  }
  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out UNUSED, dOut dout, dX dx) const {
    auto x_beta = static_cast<ComplexType<T>>(beta) * x;  // NOLINT
    dx.device(d) =
        (x_beta > static_cast<ComplexType<T>>(threshold))
            .select(dout,
                    dout / (static_cast<ComplexType<T>>(1) + (-x_beta).exp())
                               .unaryExpr(Conj<T>()));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct SoftplusDoubleGradFunctor : public BaseActivationFunctor<T> {
  float beta;
  float threshold;
  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"beta", &beta}, {"threshold", &threshold}};
  }
  template <typename Device>
  void operator()(const Device& dev,
                  const DenseTensor* X,
                  const DenseTensor* dOut,
                  const DenseTensor* ddX,
                  DenseTensor* dX,
                  DenseTensor* ddOut) const {
    auto* d = dev.eigen_device();
    auto x = EigenVector<T>::Flatten(
        GET_DATA_SAFELY(X, "Input", "X", "SoftplusDoubleGrad"));
    auto x_beta = static_cast<T>(beta) * x;  // NOLINT
    auto ddx = EigenVector<T>::Flatten(
        GET_DATA_SAFELY(ddX, "Input", "DDX", "SoftplusDoubleGrad"));

    if (dX) {
      auto dx = EigenVector<T>::Flatten(
          GET_DATA_SAFELY(dX, "Output", "DX", "SoftplusDoubleGrad"));
      auto dout = EigenVector<T>::Flatten(
          GET_DATA_SAFELY(dOut, "Output", "DOut", "SoftplusDoubleGrad"));
      // ddx * dout * beta * exp(x_beta) / (exp(x_beta) + 1) ^ 2, if x_beta
      // <= threshold
      // 0, if x_beta > threshold
      dx.device(*d) =
          (x_beta > static_cast<T>(threshold))
              .select(x.constant(static_cast<T>(0)),
                      ddx * dout * static_cast<T>(beta) * x_beta.exp() /
                          (x_beta.exp() + static_cast<T>(1))
                              .pow(static_cast<T>(2)));
    }

    if (ddOut) {
      auto ddout = EigenVector<T>::Flatten(
          GET_DATA_SAFELY(ddOut, "Output", "DDOut", "SoftplusDoubleGrad"));
      // ddx / (1 + exp(-x_beta)), if x_beta <= threshold
      // ddx, if x_beta > threshold
      ddout.device(*d) =
          (x_beta > static_cast<T>(threshold))
              .select(ddx, ddx / (static_cast<T>(1) + (-x_beta).exp()));
    }
  }
  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

// Tangent(x) = tan(x)
template <typename T>
struct TanFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    // Note(GGBond8488): Since Eigen3.3, Behavior like {A = (B * A).cwiseAbs()}
    // will give wrong result, details see
    // http://eigen.tuxfamily.org/dox/group__TopicAliasing.html
    out.device(d) = x.unaryExpr(Tangent<T>()).eval();
  }
};

template <typename T>
struct Sinh {
  HOSTDEVICE T operator()(const T& val) const { return sinh(val); }
};

template <>
struct Sinh<dtype::float16> {
  HOSTDEVICE dtype::float16 operator()(const dtype::float16& val) const {
    return dtype::float16(sinhf(static_cast<float>(val)));
  }
};

template <typename T>
struct Cosh {
  HOSTDEVICE T operator()(const T& val) const { return cosh(val); }
};

template <>
struct Cosh<dtype::float16> {
  HOSTDEVICE dtype::float16 operator()(const dtype::float16& val) const {
    return dtype::float16(coshf(static_cast<float>(val)));
  }
};

// sinh(x) = sinh(x)
template <typename T>
struct SinhFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    out.device(d) = x.unaryExpr(Sinh<T>()).eval();
  }
};

// cosh(x) = cosh(x)
template <typename T>
struct CoshFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    out.device(d) = x.unaryExpr(Cosh<T>()).eval();
  }
};

// sinh'(x) = cosh(x)
template <typename T>
struct SinhGradFunctor : public BaseActivationFunctor<T> {
  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out UNUSED, dOut dout, dX dx) const {
    dx.device(d) = dout * x.unaryExpr(Cosh<T>());
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};

template <typename T>
struct SinhGradFunctor<ComplexType<T>>
    : public BaseActivationFunctor<ComplexType<T>> {
  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out UNUSED, dOut dout, dX dx) const {
    dx.device(d) =
        dout * x.unaryExpr(Cosh<ComplexType<T>>()).unaryExpr(Conj<T>());
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};

// cosh'(x) = sinh(x)
template <typename T>
struct CoshGradFunctor : public BaseActivationFunctor<T> {
  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out UNUSED, dOut dout, dX dx) const {
    dx.device(d) = dout * x.unaryExpr(Sinh<T>());
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};

template <typename T>
struct CoshGradFunctor<ComplexType<T>>
    : public BaseActivationFunctor<ComplexType<T>> {
  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out UNUSED, dOut dout, dX dx) const {
    dx.device(d) =
        dout * x.unaryExpr(Sinh<ComplexType<T>>()).unaryExpr(Conj<T>());
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};

template <typename T>
struct Acos {
  HOSTDEVICE T operator()(const T& val) const { return acos(val); }
};

template <>
struct Acos<dtype::float16> {
  HOSTDEVICE dtype::float16 operator()(const dtype::float16& val) const {
    return dtype::float16(acos(static_cast<float>(val)));
  }
};

// Acos(x) = acos(x)
template <typename T>
struct AcosFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    out.device(d) = x.unaryExpr(Acos<T>()).eval();
  }
};

// acos'(x) = -1/sqrt(1-x^2)
template <typename T>
struct AcosGradFunctor : public BaseActivationFunctor<T> {
  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out UNUSED, dOut dout, dX dx) const {
    dx.device(d) =
        -dout * static_cast<T>(1) / (static_cast<T>(1) - x.square()).sqrt();
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};

template <typename T>
struct AcosGradFunctor<ComplexType<T>>
    : public BaseActivationFunctor<ComplexType<T>> {
  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out UNUSED, dOut dout, dX dx) const {
    dx.device(d) =
        -dout * (static_cast<ComplexType<T>>(1) /
                 (static_cast<ComplexType<T>>(1) - x.square()).sqrt())
                    .unaryExpr(Conj<T>());
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};

template <typename T>
struct Asin {
  HOSTDEVICE T operator()(const T& val) const { return asin(val); }
};

template <>
struct Asin<dtype::float16> {
  HOSTDEVICE dtype::float16 operator()(const dtype::float16& val) const {
    return dtype::float16(asin(static_cast<float>(val)));
  }
};

// Asin(x) = asin(x)
template <typename T>
struct AsinFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    out.device(d) = x.unaryExpr(Asin<T>()).eval();
  }
};

// asin'(x) = 1/sqrt(1-x^2)
template <typename T>
struct AsinGradFunctor : public BaseActivationFunctor<T> {
  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out UNUSED, dOut dout, dX dx) const {
    dx.device(d) =
        dout * static_cast<T>(1) / (static_cast<T>(1) - x.square()).sqrt();
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};

template <typename T>
struct AsinGradFunctor<ComplexType<T>>
    : public BaseActivationFunctor<ComplexType<T>> {
  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out UNUSED, dOut dout, dX dx) const {
    dx.device(d) = dout * (static_cast<ComplexType<T>>(1) /
                           (static_cast<ComplexType<T>>(1) - x.square()).sqrt())
                              .unaryExpr(Conj<T>());
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};

template <typename T>
struct Atan {
  HOSTDEVICE T operator()(const T& val) const { return atan(val); }
};

template <>
struct Atan<dtype::float16> {
  HOSTDEVICE dtype::float16 operator()(const dtype::float16& val) const {
    return dtype::float16(atan(static_cast<float>(val)));
  }
};

// Atan(x) = atan(x)
template <typename T>
struct AtanFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    out.device(d) = x.unaryExpr(Atan<T>()).eval();
  }
};

// atan'(x) =  1 / (1 + x^2)
template <typename T>
struct AtanGradFunctor : public BaseActivationFunctor<T> {
  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out UNUSED, dOut dout, dX dx) const {
    dx.device(d) = dout * static_cast<T>(1) / (static_cast<T>(1) + x.square());
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};

template <typename T>
struct AtanGradFunctor<ComplexType<T>>
    : public BaseActivationFunctor<ComplexType<T>> {
  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out UNUSED, dOut dout, dX dx) const {
    dx.device(d) = dout * (static_cast<ComplexType<T>>(1) /
                           (static_cast<ComplexType<T>>(1) + x.square()))
                              .unaryExpr(Conj<T>());
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};

template <typename T>
struct LogitGradFunctor {
  template <typename Device, typename X, typename dOut, typename dX, typename P>
  void operator()(Device d, X x, dOut dout, dX dx, P p, float eps) const {
    // logit(x)' = 1/(x*(1-x))
    dx.device(d) =
        (x < static_cast<T>(eps) || x > static_cast<T>(1.0 - eps))
            .select(p.constant(static_cast<T>(0)),
                    dout * (static_cast<T>(1) / ((static_cast<T>(1) - x) * x)));
  }
};

template <typename T>
struct Acosh {
  HOSTDEVICE T operator()(const T& val) const { return acosh(val); }
};

template <>
struct Acosh<dtype::float16> {
  HOSTDEVICE dtype::float16 operator()(const dtype::float16& val) const {
    return dtype::float16(acosh(static_cast<float>(val)));
  }
};

// Acosh(x) = acosh(x)
template <typename T>
struct AcoshFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    out.device(d) = x.unaryExpr(Acosh<T>()).eval();
  }
};

// acosh'(x) =  1/sqrt(x^2 - 1)
template <typename T>
struct AcoshGradFunctor : public BaseActivationFunctor<T> {
  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out UNUSED, dOut dout, dX dx) const {
    dx.device(d) =
        dout * static_cast<T>(1) / (x * x - static_cast<T>(1)).sqrt();
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};

template <typename T>
struct AcoshGradFunctor<ComplexType<T>>
    : public BaseActivationFunctor<ComplexType<T>> {
  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out UNUSED, dOut dout, dX dx) const {
    dx.device(d) =
        dout * (static_cast<ComplexType<T>>(1) /
                (-static_cast<ComplexType<T>>(1) + x.square()).sqrt())
                   .unaryExpr(Conj<T>());
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};
template <typename T>
struct Asinh {
  HOSTDEVICE T operator()(const T& val) const { return asinh(val); }
};

template <>
struct Asinh<dtype::float16> {
  HOSTDEVICE dtype::float16 operator()(const dtype::float16& val) const {
    return dtype::float16(asinh(static_cast<float>(val)));
  }
};

// Asinh(x) = asinh(x)
template <typename T>
struct AsinhFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    out.device(d) = x.unaryExpr(Asinh<T>()).eval();
  }
};

// asinh'(x) =  1/sqrt(x^2 + 1)
template <typename T>
struct AsinhGradFunctor : public BaseActivationFunctor<T> {
  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out UNUSED, dOut dout, dX dx) const {
    dx.device(d) =
        dout * static_cast<T>(1) / (x.square() + static_cast<T>(1)).sqrt();
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};

template <typename T>
struct AsinhGradFunctor<ComplexType<T>>
    : public BaseActivationFunctor<ComplexType<T>> {
  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out UNUSED, dOut dout, dX dx) const {
    dx.device(d) = dout * (static_cast<ComplexType<T>>(1) /
                           (x.square() + static_cast<ComplexType<T>>(1)).sqrt())
                              .unaryExpr(Conj<T>());
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};

template <typename T>
struct Atanh {
  HOSTDEVICE T operator()(const T& val) const { return atanh(val); }
};

template <>
struct Atanh<dtype::float16> {
  HOSTDEVICE dtype::float16 operator()(const dtype::float16& val) const {
    return dtype::float16(atanh(static_cast<float>(val)));
  }
};

// Atanh(x) = atanh(x)
template <typename T>
struct AtanhFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    out.device(d) = x.unaryExpr(Atanh<T>()).eval();
  }
};

// atanh'(x) =  1/(1 - x^2)
template <typename T>
struct AtanhGradFunctor : public BaseActivationFunctor<T> {
  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out UNUSED, dOut dout, dX dx) const {
    dx.device(d) = dout * static_cast<T>(1) / (static_cast<T>(1) - x.square());
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};

template <typename T>
struct AtanhGradFunctor<ComplexType<T>>
    : public BaseActivationFunctor<ComplexType<T>> {
  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out UNUSED, dOut dout, dX dx) const {
    dx.device(d) = dout * (static_cast<ComplexType<T>>(1) /
                           (static_cast<ComplexType<T>>(1) - x.square()))
                              .unaryExpr(Conj<T>());
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};
// exp functor
// exp(x) = e^x
template <typename T>
struct ExpFunctor : public BaseActivationFunctor<T> {
  using U = typename std::conditional_t<std::is_integral<T>::value, float, T>;

  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    out.device(d) = x.template cast<U>().exp();
  }
};

template <typename T>
struct ExpGradFunctor : public BaseActivationFunctor<T> {
  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(Device d, X x UNUSED, Out out, dOut dout, dX dx) const {
    dx.device(d) = dout * out;
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() {
    return ActBwdOpFwdDeps::kDepOut;
  }
};

template <typename T>
struct ExpGradFunctor<ComplexType<T>>
    : public BaseActivationFunctor<ComplexType<T>> {
  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(Device d, X x UNUSED, Out out, dOut dout, dX dx) const {
    dx.device(d) = dout * out.unaryExpr(Conj<T>());
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() {
    return ActBwdOpFwdDeps::kDepOut;
  }
};

template <typename T>
struct Expm1 {};

template <typename T>
struct Expm1<ComplexType<T>> {
  HOSTDEVICE ComplexType<T> operator()(const ComplexType<T>& val) const {
    return exp(val) - static_cast<ComplexType<T>>(1);
  }
};

// expm1(x) = e^x - 1
template <typename T>
struct Expm1Functor : public BaseActivationFunctor<T> {
  using U = typename std::conditional_t<std::is_integral<T>::value, float, T>;

  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    out.device(d) = x.template cast<U>().expm1();
  }
};

template <typename T>
struct Expm1Functor<ComplexType<T>>
    : public BaseActivationFunctor<ComplexType<T>> {
  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    out.device(d) = x.unaryExpr(Expm1<ComplexType<T>>()).eval();
  }
};

template <typename T>
struct Expm1GradFunctor : public BaseActivationFunctor<T> {
  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(Device d, X x UNUSED, Out out, dOut dout, dX dx) const {
    dx.device(d) = dout * out + dout;
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() {
    return ActBwdOpFwdDeps::kDepOut;
  }
};

template <typename T>
struct Expm1GradFunctor<ComplexType<T>>
    : public BaseActivationFunctor<ComplexType<T>> {
  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(Device d, X x UNUSED, Out out, dOut dout, dX dx) const {
    dx.device(d) = dout * out.unaryExpr(Conj<T>()) + dout;
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepOut; }
};

// relu(x) = max(x, 0)
template <typename T>
struct ReluCPUFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    out.device(d) = x.unaryExpr([] HOSTDEVICE(T v) {
      return v > static_cast<T>(0) ? v : static_cast<T>(0);
    });
  }
};

template <typename T>
struct ReluCUDAFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    out.device(d) = x.cwiseMax(static_cast<T>(0));
  }
};

template <typename T>
struct ReluGradFunctor : public BaseActivationFunctor<T> {
  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(Device d, X x UNUSED, Out out, dOut dout, dX dx) const {
    dx.device(d) = dout * (out > static_cast<T>(0)).template cast<T>();
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() {
    return ActBwdOpFwdDeps::kDepOut;
  }
};

template <typename T>
struct ReluGradGradFunctor : public BaseActivationFunctor<T> {
  template <typename Device>
  void operator()(const Device& dev,
                  const DenseTensor* X UNUSED,
                  const DenseTensor* Out,
                  const DenseTensor* ddX,
                  DenseTensor* ddOut,
                  DenseTensor* dOut UNUSED,
                  DenseTensor* dX UNUSED) const {
    auto* d = dev.eigen_device();
    auto ddx = EigenVector<T>::Flatten(
        GET_DATA_SAFELY(ddX, "Input", "DDX", "ReluGradGrad"));
    auto out = EigenVector<T>::Flatten(
        GET_DATA_SAFELY(Out, "Output", "Out", "ReluGradGrad"));
    if (ddOut) {
      auto ddout = EigenVector<T>::Flatten(
          GET_DATA_SAFELY(ddOut, "Output", "DDOut", "ReluGradGrad"));
      ddout.device(*d) = ddx * (out > static_cast<T>(0)).template cast<T>();
    }
  }
  static constexpr ActBwdOpFwdDeps FwdDeps() {
    return ActBwdOpFwdDeps::kDepOut;
  }
};

// tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
template <typename T>
struct TanhFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    out.device(d) = x.tanh();
  }
};

template <typename T>
struct TanhGradFunctor : public BaseActivationFunctor<T> {
  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(Device d, X x UNUSED, Out out, dOut dout, dX dx) const {
    dx.device(d) = dout * (static_cast<T>(1) - out * out);
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() {
    return ActBwdOpFwdDeps::kDepOut;
  }
};

template <typename T>
struct TanhGradFunctor<ComplexType<T>>
    : public BaseActivationFunctor<ComplexType<T>> {
  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(Device d, X x UNUSED, Out out, dOut dout, dX dx) const {
    // auto dx_ = static_cast<ComplexType<T>>(1) - out * out;
    // ComplexType<T> dx_conj_(dx_.real, -dx_.imag);
    // dx.device(d) = dout * dx_conj_;
    dx.device(d) =
        dout *
        (static_cast<ComplexType<T>>(1) - out * out).unaryExpr(Conj<T>());
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() {
    return ActBwdOpFwdDeps::kDepOut;
  }
};

template <typename T>
struct TanhGradGradFunctor : public BaseActivationFunctor<T> {
  template <typename Device>
  void operator()(const Device& dev,
                  const DenseTensor* Out,
                  const DenseTensor* ddX,
                  const DenseTensor* dOut,
                  DenseTensor* dOutNew,
                  DenseTensor* ddOut) const {
    auto* d = dev.eigen_device();
    auto ddx = EigenVector<T>::Flatten(
        GET_DATA_SAFELY(ddX, "Input", "DDX", "TanhGradGrad"));
    auto out = EigenVector<T>::Flatten(
        GET_DATA_SAFELY(Out, "Input", "Out", "TanhGradGrad"));
    // tanh grad grad : ddout = (1 - out^2) * ddx, dout = - (dout_old * 2 * out
    // * ddx)
    if (dOutNew) {
      auto dout = EigenVector<T>::Flatten(
          GET_DATA_SAFELY(dOut, "Input", "DOut", "TanhGradGrad"));
      auto dout_new = EigenVector<T>::Flatten(
          GET_DATA_SAFELY(dOutNew, "Output", "DOutNew", "TanhGradGrad"));
      dout_new.device(*d) =
          static_cast<T>(-1) * dout * static_cast<T>(2) * out * ddx;
    }
    if (ddOut) {
      auto ddout = EigenVector<T>::Flatten(
          GET_DATA_SAFELY(ddOut, "Output", "DDOut", "TanhGradGrad"));
      ddout.device(*d) = (static_cast<T>(1) - out * out) * ddx;
    }
  }
  static constexpr ActBwdOpFwdDeps FwdDeps() {
    return ActBwdOpFwdDeps::kDepOut;
  }
};
/*
    Out
    DOut                            D_Dout
    DDx     -> TanhTripleGrad ->    D_DDx
    D_DDout                         d_OutNew
    D_Dout_new

    D_Dout = (-2) * Out * DDx * D_Dout_new
    D_DDx = (1-Out^2)*D_DDout + (-2) * Out * DOut * D_Dout_new
    D_OutNew = (-2) * Out * DDx * D_DDout + (-2) * DOut * DDx * D_Dout_new

    Out, DDX, DOut, D_DDOut, D_DOut_New   // input
    D_OutNew, D_DOut, D_DDx               // output
*/
template <typename T>
struct TanhTripleGradFunctor : public BaseActivationFunctor<T> {
  template <typename Device>
  void operator()(const Device& dev,
                  const DenseTensor* Out,
                  const DenseTensor* ddX,
                  const DenseTensor* dOut,
                  const DenseTensor* d_DDOut,
                  const DenseTensor* d_dOut_New,
                  DenseTensor* d_d_Out,
                  DenseTensor* d_Out_New,
                  DenseTensor* d_DDx) const {
    auto* d = dev.eigen_device();
    auto ddx = EigenVector<T>::Flatten(
        GET_DATA_SAFELY(ddX, "Input", "DDX", "TanhTripleGrad"));
    auto out = EigenVector<T>::Flatten(
        GET_DATA_SAFELY(Out, "Input", "Out", "TanhTripleGrad"));
    auto dout = EigenVector<T>::Flatten(
        GET_DATA_SAFELY(dOut, "Input", "DOut", "TanhTripleGrad"));

    if (d_Out_New) {
      auto d_OutNew = EigenVector<T>::Flatten(
          GET_DATA_SAFELY(d_Out_New, "Output", "D_OutNew", "TanhTripleGrad"));

      if (d_DDOut && d_dOut_New) {
        auto d_ddOut = EigenVector<T>::Flatten(
            GET_DATA_SAFELY(d_DDOut, "Input", "D_DDOut", "TanhTripleGrad"));
        auto d_dOutNew = EigenVector<T>::Flatten(GET_DATA_SAFELY(
            d_dOut_New, "Input", "D_DOut_New", "TanhTripleGrad"));

        d_OutNew.device(*d) = (static_cast<T>(-2) * out * ddx * d_ddOut) -
                              (static_cast<T>(2) * dout * ddx * d_dOutNew);

      } else if (d_DDOut && !d_dOut_New) {
        auto d_ddOut = EigenVector<T>::Flatten(
            GET_DATA_SAFELY(d_DDOut, "Input", "D_DDOut", "TanhTripleGrad"));

        d_OutNew.device(*d) = (static_cast<T>(-2) * out * ddx * d_ddOut);

      } else if (!d_DDOut && d_dOut_New) {
        auto d_dOutNew = EigenVector<T>::Flatten(GET_DATA_SAFELY(
            d_dOut_New, "Input", "D_DOut_New", "TanhTripleGrad"));

        d_OutNew.device(*d) = -(static_cast<T>(2) * dout * ddx * d_dOutNew);
      } else {
        d_OutNew.device(*d) = static_cast<T>(0) * out;
      }
    }
    if (d_d_Out) {
      auto d_dOut = EigenVector<T>::Flatten(
          GET_DATA_SAFELY(d_d_Out, "Output", "D_DOut", "TanhTripleGrad"));

      if (d_dOut_New) {
        auto d_dOutNew = EigenVector<T>::Flatten(GET_DATA_SAFELY(
            d_dOut_New, "Input", "D_DOut_New", "TanhTripleGrad"));
        d_dOut.device(*d) = static_cast<T>(-2) * out * ddx * d_dOutNew;
      } else {
        d_dOut.device(*d) = static_cast<T>(0) * out;
      }
    }
    if (d_DDx) {
      auto d_ddx = EigenVector<T>::Flatten(
          GET_DATA_SAFELY(d_DDx, "Output", "D_DDx", "TanhTripleGrad"));

      if (d_DDOut && d_dOut_New) {
        auto d_ddOut = EigenVector<T>::Flatten(
            GET_DATA_SAFELY(d_DDOut, "Input", "D_DDOut", "TanhTripleGrad"));
        auto d_dOutNew = EigenVector<T>::Flatten(GET_DATA_SAFELY(
            d_dOut_New, "Input", "D_DOut_New", "TanhTripleGrad"));
        d_ddx.device(*d) = (static_cast<T>(1) - (out * out)) * d_ddOut -
                           static_cast<T>(2) * out * dout * d_dOutNew;

      } else if (d_DDOut && !d_dOut_New) {
        auto d_ddOut = EigenVector<T>::Flatten(
            GET_DATA_SAFELY(d_DDOut, "Input", "D_DDOut", "TanhTripleGrad"));
        d_ddx.device(*d) = (static_cast<T>(1) - (out * out)) * d_ddOut;
      } else if (!d_DDOut && d_dOut_New) {
        auto d_dOutNew = EigenVector<T>::Flatten(GET_DATA_SAFELY(
            d_dOut_New, "Input", "D_DOut_New", "TanhTripleGrad"));
        d_ddx.device(*d) = -static_cast<T>(2) * out * dout * d_dOutNew;
      } else {
        d_ddx.device(*d) = static_cast<T>(0) * ddx;
      }
    }
  }
  static constexpr ActBwdOpFwdDeps FwdDeps() {
    return ActBwdOpFwdDeps::kDepOut;
  }
};

template <typename T>
struct HardTanhFunctor : public BaseActivationFunctor<T> {
  float t_min;
  float t_max;

  // NOTE: Explicit hides the `BaseActivationFunctor<T>::GetAttrs`
  // not polymorphism for speed.
  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"t_min", &t_min}, {"t_max", &t_max}};
  }

  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    out.device(d) = x.cwiseMax(static_cast<T>(t_min))
                        .cwiseMin(static_cast<T>(t_max));  // NOLINT
  }
};

template <typename T>
struct HardTanhGradFunctor : public BaseActivationFunctor<T> {
  float t_min;
  float t_max;
  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"t_min", &t_min}, {"t_max", &t_max}};
  }
  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out UNUSED, dOut dout, dX dx) const {
    dx.device(d) = dout * ((x > static_cast<T>(t_min)) *
                           (x < static_cast<T>(t_max)))  // NOLINT
                              .template cast<T>();
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct LeakyReluFunctor : public BaseActivationFunctor<T> {
  float alpha;
  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"alpha", &alpha}};
  }

  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    if (alpha < 1.f) {  // NOLINT
      out.device(d) = x.cwiseMax(static_cast<T>(alpha) * x);
    } else {
      out.device(d) = x.cwiseMin(static_cast<T>(alpha) * x);
    }
  }
};

template <typename T>
struct LeakyReluGradFunctor : public BaseActivationFunctor<T> {
  float alpha;
  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"alpha", &alpha}};
  }
  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out UNUSED, dOut dout, dX dx) const {
    auto temp1 = static_cast<T>(alpha) *
                 (x < static_cast<T>(0)).template cast<T>();  // NOLINT
    auto temp2 = (x >= static_cast<T>(0)).template cast<T>();
    dx.device(d) = dout * (temp1 + temp2).template cast<T>();
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct LeakyReluGradGradFunctor : public BaseActivationFunctor<T> {
  float alpha;
  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"alpha", &alpha}};
  }
  template <typename Device>
  void operator()(const Device& dev,
                  const DenseTensor* X,
                  const DenseTensor* Out UNUSED,
                  const DenseTensor* ddX,
                  DenseTensor* ddOut,
                  DenseTensor* dOut UNUSED,
                  DenseTensor* dX UNUSED) const {
    if (ddOut) {
      auto* d = dev.eigen_device();
      auto ddx = EigenVector<T>::Flatten(
          GET_DATA_SAFELY(ddX, "Input", "DDX", "LeakyReluGradGrad"));
      auto x = EigenVector<T>::Flatten(
          GET_DATA_SAFELY(X, "Input", "X", "LeakyReluGradGrad"));
      auto ddout = EigenVector<T>::Flatten(
          GET_DATA_SAFELY(ddOut, "Output", "DOut", "LeakyReluGradGrad"));
      ddout.device(*d) = ddx * ((x > static_cast<T>(0)).template cast<T>() +
                                static_cast<T>(alpha) *
                                    (x <= static_cast<T>(0)).template cast<T>())
                                   .template cast<T>();
    }
  }
  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct ThresholdedReluFunctor : public BaseActivationFunctor<T> {
  float threshold;
  float value;
  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"threshold", &threshold}, {"value", &value}};
  }

  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    auto th = static_cast<T>(threshold);  // NOLINT
    out.device(d) = (x > th).template cast<T>() * x +
                    (x <= th).template cast<T>() * static_cast<T>(value);
  }
};

template <typename T>
struct ThresholdedReluGradFunctor : public BaseActivationFunctor<T> {
  float threshold;
  float value;
  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"threshold", &threshold}, {"value", &value}};
  }

  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out UNUSED, dOut dout, dX dx) const {
    auto th = static_cast<T>(threshold);  // NOLINT
    dx.device(d) = dout * (x > th).template cast<T>();
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

// relu6(x) = min(max(0, x), 6)
template <typename T>
struct Relu6Functor : public BaseActivationFunctor<T> {
  float threshold;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"threshold", &threshold}};
  }

  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    out.device(d) = x.cwiseMax(static_cast<T>(0))
                        .cwiseMin(static_cast<T>(threshold));  // NOLINT
  }
};

template <typename T>
struct Relu6GradFunctor : public BaseActivationFunctor<T> {
  typename BaseActivationFunctor<T>::AttrPair GetAttrs() { return {{}}; }
  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(Device d, X x UNUSED, Out out, dOut dout, dX dx) const {
    float threshold = 6;
    dx.device(d) =
        dout * ((out > static_cast<T>(0)) * (out < static_cast<T>(threshold)))
                   .template cast<T>();
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() {
    return ActBwdOpFwdDeps::kDepOut;
  }
};

// tanhshrink(x) = x - tanh(x)
// where tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
template <typename T>
struct TanhShrinkFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    out.device(d) = x - x.tanh();
  }
};

template <typename T>
struct TanhShrinkGradFunctor : public BaseActivationFunctor<T> {
  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out UNUSED, dOut dout, dX dx) const {
    dx.device(d) = dout * (x.tanh() * x.tanh());
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

// tanhshrink(x) = x - tanh(x)
// where tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
template <typename T>
struct HardShrinkFunctor : public BaseActivationFunctor<T> {
  float threshold;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"threshold", &threshold}};
  }
  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    auto temp1 = x < static_cast<T>(threshold * -1.f);  // NOLINT
    auto temp2 = x > static_cast<T>(threshold);
    out.device(d) = x * (temp1 || temp2).template cast<T>();
  }
};

template <typename T>
struct HardShrinkGradFunctor : public BaseActivationFunctor<T> {
  float threshold;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"threshold", &threshold}};
  }

  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out UNUSED, dOut dout, dX dx) const {
    auto temp1 = x < static_cast<T>(threshold * -1.f);  // NOLINT
    auto temp2 = x > static_cast<T>(threshold);
    dx.device(d) = dout * (temp1 || temp2).template cast<T>();
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

// softshrink(x) = x - lambda, if x > lambda; x + lambda, if x < -lambda; 0
// otherwise
template <typename T>
struct SoftShrinkFunctor : public BaseActivationFunctor<T> {
  float lambda;
  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"lambda", &lambda}};
  }

  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    auto lambdaT = static_cast<T>(lambda);  // NOLINT
    auto temp1 = (x > lambdaT).template cast<T>();
    auto temp2 = (x < -lambdaT).template cast<T>();
    out.device(d) = temp1 * (x - lambdaT) + temp2 * (x + lambdaT);
  }
};

template <typename T>
struct SoftShrinkGradFunctor : public BaseActivationFunctor<T> {
  float lambda;
  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"lambda", &lambda}};
  }
  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out UNUSED, dOut dout, dX dx) const {
    auto lambdaT = static_cast<T>(lambda);  // NOLINT
    auto temp1 = (x > lambdaT).template cast<T>();
    auto temp2 = (x < -lambdaT).template cast<T>();
    dx.device(d) = dout * (temp1 + temp2).template cast<T>();
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct ELUFunctor : public BaseActivationFunctor<T> {
  float alpha;
  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"alpha", &alpha}};
  }

  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    out.device(d) =
        (x < static_cast<T>(0))
            .select(static_cast<T>(alpha) * (x.exp() - static_cast<T>(1)),
                    x);  // NOLINT
  }
};

template <typename T>
struct ELUGradFunctor : public BaseActivationFunctor<T> {
  float alpha;
  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"alpha", &alpha}};
  }
  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(Device d, X x UNUSED, Out out, dOut dout, dX dx) const {
    // case 1: alpha >= 0
    // dx = dout, if out > 0
    // dx = dout * (out + alpha), if out <= 0
    dx.device(d) = (out > static_cast<T>(0))
                       .select(dout, dout * (out + static_cast<T>(alpha)));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct ELUGradNegativeAlphaFunctor : public BaseActivationFunctor<T> {
  float alpha;
  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"alpha", &alpha}};
  }
  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out UNUSED, dOut dout, dX dx) const {
    // case 2: alpha < 0
    // dx = dout, if x > 0
    // dx = dout * (out + alpha), if x <=0
    dx.device(d) = (x > static_cast<T>(0))
                       .select(dout, dout * static_cast<T>(alpha) * x.exp());
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct ELUGradGradFunctor : public BaseActivationFunctor<T> {
  float alpha;
  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"alpha", &alpha}};
  }
  template <typename Device>
  void operator()(const Device& dev,
                  const DenseTensor* X,
                  const DenseTensor* ddX,
                  DenseTensor* ddOut,
                  const DenseTensor* dOut,
                  DenseTensor* dX) const {
    auto* d = dev.eigen_device();
    auto ddx = EigenVector<T>::Flatten(
        GET_DATA_SAFELY(ddX, "Input", "DDX", "ELUGradGrad"));
    auto x = EigenVector<T>::Flatten(
        GET_DATA_SAFELY(X, "Input", "X", "ELUGradGrad"));

    if (dX) {
      auto dx = EigenVector<T>::Flatten(
          GET_DATA_SAFELY(dX, "Output", "DX", "ELUGradGrad"));
      auto dout = EigenVector<T>::Flatten(
          GET_DATA_SAFELY(dOut, "Output", "DOut", "ELUGradGrad"));
      dx.device(*d) = ddx * dout * static_cast<T>(alpha) * x.exp() *
                      (x <= static_cast<T>(0)).template cast<T>();
    }

    if (ddOut) {
      auto ddout = EigenVector<T>::Flatten(
          GET_DATA_SAFELY(ddOut, "Output", "DDOut", "ELUGradGrad"));
      ddout.device(*d) = ddx * ((x > static_cast<T>(0)).template cast<T>() +
                                static_cast<T>(alpha) * x.exp() *
                                    (x <= static_cast<T>(0)).template cast<T>())
                                   .template cast<T>();
    }
  }
  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

// silu(x) = x / (1 + exp(-x))
template <typename T>
struct SiluFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    auto temp = static_cast<T>(1) / (static_cast<T>(1) + (-x).exp());
    out.device(d) = x * temp;
  }
};

// silu'(x) = (1 / (1 + e^{-x}))  * (1 + out * e^{-x}))
template <typename T>
struct SiluGradFunctor : public BaseActivationFunctor<T> {
  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out UNUSED, dOut dout, dX dx) const {
    auto temp1 = static_cast<T>(1) + (-x).exp();  // 1+e^(-x)
    auto temp2 = x * (-x).exp();                  // x*e^(-x)
    dx.device(d) = dout * ((static_cast<T>(1) / temp1) *
                           (static_cast<T>(1) + (temp2 / temp1)));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct SiluGradFunctor<ComplexType<T>>
    : public BaseActivationFunctor<ComplexType<T>> {
  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out UNUSED, dOut dout, dX dx) const {
    auto temp1 = static_cast<ComplexType<T>>(1) + (-x).exp();  // 1+e^(-x)
    auto temp2 = x * (-x).exp();                               // x*e^(-x)
    dx.device(d) = dout * ((static_cast<ComplexType<T>>(1) / temp1) *
                           (static_cast<ComplexType<T>>(1) + (temp2 / temp1)))
                              .unaryExpr(Conj<T>());
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct SoftsignFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    out.device(d) = x / (static_cast<T>(1) + x.abs());
  }
};

// d(softsign(x))/dx = 1 / (1 + |x|)^2
// Taken from https://en.wikipedia.org/wiki/Activation_function

template <typename T>
struct SoftsignGradFunctor : public BaseActivationFunctor<T> {
  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out UNUSED, dOut dout, dX dx) const {
    dx.device(d) =
        dout * (static_cast<T>(1) / (static_cast<T>(1) + x.abs()).square());
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct SoftsignGradFunctor<ComplexType<T>>
    : public BaseActivationFunctor<ComplexType<T>> {
  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out UNUSED, dOut dout, dX dx) const {
    ComplexType<T> one = static_cast<ComplexType<T>>(1.0f);
    auto temp = (-x / (one + x.abs()).square()).unaryExpr(Real<T>());

    dx.device(d) = dout * (one / (one + x.abs()) + temp * x / x.abs());
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

// sigmoid(x) = 1 / (1 + exp(-x))
template <typename T>
struct SigmoidFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    out.device(d) = static_cast<T>(1) / (static_cast<T>(1) + (-x).exp());
  }
};

template <typename T>
struct SigmoidGradFunctor : public BaseActivationFunctor<T> {
  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(Device d, X x UNUSED, Out out, dOut dout, dX dx) const {
    dx.device(d) = dout * out * (static_cast<T>(1) - out);
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() {
    return ActBwdOpFwdDeps::kDepOut;
  }
};

template <typename T>
struct SigmoidGradFunctor<ComplexType<T>>
    : public BaseActivationFunctor<ComplexType<T>> {
  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(Device d, X x UNUSED, Out out, dOut dout, dX dx) const {
    ComplexType<T> one = static_cast<ComplexType<T>>(1);
    dx.device(d) = dout * (out * (one - out)).unaryExpr(Conj<T>());
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() {
    return ActBwdOpFwdDeps::kDepOut;
  }
};

/*
    Out
    DOut -> SigmoidGradGrad -> DOutNew
    DDX                        DDOut

    DDOut = (1-Out)*Out*DDX
    DOutNew = (1-2*Out)*DOut*DDX
*/
template <typename T>
struct SigmoidGradGradFunctor : public BaseActivationFunctor<T> {
  template <typename Device>
  void operator()(const Device& dev,
                  const DenseTensor* Out,
                  const DenseTensor* ddX,
                  const DenseTensor* dOut,
                  DenseTensor* dOutNew,
                  DenseTensor* ddOut) const {
    auto* d = dev.eigen_device();
    auto ddx = EigenVector<T>::Flatten(
        GET_DATA_SAFELY(ddX, "Input", "DDX", "SigmoidGradGrad"));
    auto out = EigenVector<T>::Flatten(
        GET_DATA_SAFELY(Out, "Input", "Out", "SigmoidGradGrad"));

    if (dOutNew) {
      auto dout = EigenVector<T>::Flatten(
          GET_DATA_SAFELY(dOut, "Input", "DOut", "SigmoidGradGrad"));
      auto dout_new = EigenVector<T>::Flatten(
          GET_DATA_SAFELY(dOutNew, "Output", "DOutNew", "SigmoidGradGrad"));
      dout_new.device(*d) =
          (static_cast<T>(1) - static_cast<T>(2) * out) * dout * ddx;
    }
    if (ddOut) {
      auto ddout = EigenVector<T>::Flatten(
          GET_DATA_SAFELY(ddOut, "Output", "DDOut", "SigmoidGradGrad"));
      ddout.device(*d) = (static_cast<T>(1) - out) * out * ddx;
    }
  }
  static constexpr ActBwdOpFwdDeps FwdDeps() {
    return ActBwdOpFwdDeps::kDepOut;
  }
};

/*
    Out
    DOut                            D_Dout
    DDx     -> SigmoidTripleGrad -> D_DDx
    D_DDout                         d_OutNew
    D_Dout_new

    D_Dout = (1-2*Out)*DDx*D_Dout_new
    D_DDx = (1-Out)*Out*D_DDout + (1-2*Out)*DOut*D_Dout_new
    D_OutNew = (DDx-2*Out*DDx)*D_DDout - 2*DOut*DDx*D_Dout_new

    Out, DDX, DOut, D_DDOut, D_DOut_New   // input
    D_OutNew, D_DOut, D_DDx               // output
*/
template <typename T>
struct SigmoidTripleGradFunctor : public BaseActivationFunctor<T> {
  template <typename Device>
  void operator()(const Device& dev,
                  const DenseTensor* Out,
                  const DenseTensor* ddX,
                  const DenseTensor* dOut,
                  const DenseTensor* d_DDOut,
                  const DenseTensor* d_dOut_New,
                  DenseTensor* d_d_Out,
                  DenseTensor* d_Out_New,
                  DenseTensor* d_DDx) const {
    auto* d = dev.eigen_device();
    auto ddx = EigenVector<T>::Flatten(
        GET_DATA_SAFELY(ddX, "Input", "DDX", "SigmoidTripleGrad"));
    auto out = EigenVector<T>::Flatten(
        GET_DATA_SAFELY(Out, "Input", "Out", "SigmoidTripleGrad"));
    auto dout = EigenVector<T>::Flatten(
        GET_DATA_SAFELY(dOut, "Input", "DOut", "SigmoidTripleGrad"));
    auto d_dOutNew = EigenVector<T>::Flatten(GET_DATA_SAFELY(
        d_dOut_New, "Input", "D_DOut_New", "SigmoidTripleGrad"));

    if (d_Out_New) {
      auto d_OutNew = EigenVector<T>::Flatten(GET_DATA_SAFELY(
          d_Out_New, "Output", "D_OutNew", "SigmoidTripleGrad"));
      d_OutNew.device(*d) = -static_cast<T>(2) * dout * ddx * d_dOutNew;
      if (d_DDOut) {
        auto d_ddOut = EigenVector<T>::Flatten(
            GET_DATA_SAFELY(d_DDOut, "Input", "D_DDOut", "SigmoidTripleGrad"));
        d_OutNew.device(*d) =
            (ddx - static_cast<T>(2) * out * ddx) * d_ddOut + d_OutNew;
      }
    }
    if (d_d_Out) {
      auto d_dOut = EigenVector<T>::Flatten(
          GET_DATA_SAFELY(d_d_Out, "Output", "D_DOut", "SigmoidTripleGrad"));
      d_dOut.device(*d) =
          (static_cast<T>(1) - static_cast<T>(2) * out) * ddx * d_dOutNew;
    }
    if (d_DDx) {
      auto d_ddx = EigenVector<T>::Flatten(
          GET_DATA_SAFELY(d_DDx, "Output", "D_DDx", "SigmoidTripleGrad"));
      d_ddx.device(*d) =
          (static_cast<T>(1) - static_cast<T>(2) * out) * dout * d_dOutNew;
      if (d_DDOut) {
        auto d_ddOut = EigenVector<T>::Flatten(
            GET_DATA_SAFELY(d_DDOut, "Input", "D_DDOut", "SigmoidTripleGrad"));
        d_ddx.device(*d) = d_ddx + (static_cast<T>(1) - out) * out * d_ddOut;
      }
    }
  }
  static constexpr ActBwdOpFwdDeps FwdDeps() {
    return ActBwdOpFwdDeps::kDepOut;
  }
};

// Originally: logsigmoid(x) = -log (1 + exp(-x))
// For numerical stability, we can use the log-sum-exp trick:
// https://hips.seas.harvard.edu/blog/2013/01/09/computing-log-sum-exp/
// We can rewrite the above equation as:
// out = -log( exp(0) + exp(-x)) [since exp(0) = 1]
//   = -log( exp(max(-x, 0) - max(-x, 0)) + exp(-x + max(-x, 0) - max(-x, 0)))
//   = -log( exp(max(-x, 0)) * exp(-max(-x, 0)) - exp(max(-x, 0)) * exp(-x -
//           max(-x, 0)))
//   = -log( exp(max(-x, 0)) * (exp(-max(-x, 0)) + exp(-x - max(-x, 0))))
//   = -log( exp(max(-x, 0)) - log(exp(-max(-x, 0)) + exp(-x - max(-x, 0)))
//
// Hence, logsigmoid(x) = - (max(-x, 0) + log(exp(-max(-x, 0))
// + exp(-x - max(-x, 0))))
template <typename T>
struct LogSigmoidFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    auto temp = (-x).cwiseMax(static_cast<T>(0));  // temp = max(-x, 0)
    out.device(d) = -temp - (((-temp).exp() + (-x - temp).exp()).log());
  }
};

// Originally: f' = exp(-x) / (1 + exp(-x))
// For numerical stability: f' = exp(-x - max(-x, 0)) / (exp(-max(-x, 0)) +
// exp(-x - max(-x, 0)))
template <typename T>
struct LogSigmoidGradFunctor : public BaseActivationFunctor<T> {
  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out UNUSED, dOut dout, dX dx) const {
    auto temp = (-x).cwiseMax(static_cast<T>(0));  // temp = max(-x, 0)
    dx.device(d) =
        dout * ((-x - temp).exp() / ((-temp).exp() + (-x - temp).exp()));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct LogSigmoidGradFunctor<ComplexType<T>>
    : public BaseActivationFunctor<ComplexType<T>> {
  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out UNUSED, dOut dout, dX dx) const {
    auto temp =
        (-x).cwiseMax(static_cast<ComplexType<T>>(0));  // temp = max(-x, 0)
    dx.device(d) =
        dout * ((-x - temp).exp() / ((-temp).exp() + (-x - temp).exp()))
                   .unaryExpr(Conj<T>());
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct HardSigmoidFunctor : public BaseActivationFunctor<T> {
  float slope;
  float offset;
  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"slope", &slope}, {"offset", &offset}};
  }

  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    auto temp = x * static_cast<T>(slope) + static_cast<T>(offset);  // NOLINT
    out.device(d) =
        temp.cwiseMax(static_cast<T>(0)).cwiseMin(static_cast<T>(1));
  }
};

template <typename T>
struct HardSigmoidGradFunctor : public BaseActivationFunctor<T> {
  float slope;
  float offset;
  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"slope", &slope}, {"offset", &offset}};
  }
  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(Device d, X x UNUSED, Out out, dOut dout, dX dx) const {
    dx.device(d) = dout *  // NOLINT
                   ((out > static_cast<T>(0)) * (out < static_cast<T>(1)))
                       .template cast<T>() *
                   static_cast<T>(slope);
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() {
    return ActBwdOpFwdDeps::kDepOut;
  }
};

template <typename T>
struct Log {
  HOSTDEVICE T operator()(const T& val) const { return std::log(val); }
};

template <typename T>
struct Log<ComplexType<T>> {
  HOSTDEVICE ComplexType<T> operator()(const ComplexType<T>& val) const {
    return ComplexType<T>(std::log(std::complex<T>(val)));
  }
};

template <>
struct Log<dtype::float16> {
  HOSTDEVICE dtype::float16 operator()(const dtype::float16& val) const {
    return dtype::float16(std::log(static_cast<float>(val)));
  }
};

template <>
struct Log<dtype::bfloat16> {
  HOSTDEVICE dtype::bfloat16 operator()(const dtype::bfloat16& val) const {
    return dtype::bfloat16(std::log(static_cast<float>(val)));
  }
};

// log(x) = natural logarithm of x
template <typename T>
struct LogFunctor : public BaseActivationFunctor<T> {
  using U = typename std::conditional_t<std::is_integral<T>::value, float, T>;

  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    out.device(d) = x.template cast<U>().unaryExpr(Log<U>()).eval();
  }
};

template <typename T>
struct LogGradFunctor : public BaseActivationFunctor<T> {
  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out UNUSED, dOut dout, dX dx) const {
    dx.device(d) = dout * (static_cast<T>(1) / x);
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct LogGradFunctor<ComplexType<T>>
    : public BaseActivationFunctor<ComplexType<T>> {
  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out UNUSED, dOut dout, dX dx) const {
    dx.device(d) =
        dout * (static_cast<ComplexType<T>>(1) / x).unaryExpr(Conj<T>());
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct Log2 {
  HOSTDEVICE T operator()(const T& val) const { return std::log2(val); }
};

template <typename T>
struct Log2<ComplexType<T>> {
  HOSTDEVICE ComplexType<T> operator()(const ComplexType<T>& val) const {
    return ComplexType<T>(std::log(std::complex<T>(val)) /
                          std::log(std::complex<T>(2)));
  }
};

template <>
struct Log2<dtype::float16> {
  HOSTDEVICE dtype::float16 operator()(const dtype::float16& val) const {
    return dtype::float16(std::log2(static_cast<float>(val)));
  }
};

template <>
struct Log2<dtype::bfloat16> {
  HOSTDEVICE dtype::bfloat16 operator()(const dtype::bfloat16& val) const {
    return dtype::bfloat16(std::log2(static_cast<float>(val)));
  }
};

// log2(x) = logarithm to the base 2 of the elements of x
template <typename T>
struct Log2Functor : public BaseActivationFunctor<T> {
  using U = typename std::conditional_t<std::is_integral<T>::value, float, T>;

  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    out.device(d) = x.template cast<U>().unaryExpr(Log2<U>()).eval();
  }
};

// the gradient of log2(x) is 1/(x*ln(2))
template <typename T>
struct Log2GradFunctor : public BaseActivationFunctor<T> {
  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out UNUSED, dOut dout, dX dx) const {
    dx.device(d) = dout * static_cast<T>(1) / (x * static_cast<T>(log(2)));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct Log2GradFunctor<ComplexType<T>>
    : public BaseActivationFunctor<ComplexType<T>> {
  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out UNUSED, dOut dout, dX dx) const {
    dx.device(d) = dout * (static_cast<ComplexType<T>>(1) /
                           (x * static_cast<ComplexType<T>>(log(2))))
                              .unaryExpr(Conj<T>());
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct Log10 {
  HOSTDEVICE T operator()(const T& val) const { return std::log10(val); }
};

template <typename T>
struct Log10<ComplexType<T>> {
  HOSTDEVICE ComplexType<T> operator()(const ComplexType<T>& val) const {
    return ComplexType<T>(std::log10(std::complex<T>(val)));
  }
};

template <>
struct Log10<dtype::float16> {
  HOSTDEVICE dtype::float16 operator()(const dtype::float16& val) const {
    return dtype::float16(std::log10(static_cast<float>(val)));
  }
};

template <>
struct Log10<dtype::bfloat16> {
  HOSTDEVICE dtype::bfloat16 operator()(const dtype::bfloat16& val) const {
    return dtype::bfloat16(std::log10(static_cast<float>(val)));
  }
};

// log10(x) = logarithm to the base 10 of the elements of x
template <typename T>
struct Log10Functor : public BaseActivationFunctor<T> {
  using U = typename std::conditional_t<std::is_integral<T>::value, float, T>;

  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    out.device(d) = x.template cast<U>().unaryExpr(Log10<U>()).eval();
  }
};

// the gradient of log10(x) is 1/(x*ln(10))
template <typename T>
struct Log10GradFunctor : public BaseActivationFunctor<T> {
  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out UNUSED, dOut dout, dX dx) const {
    dx.device(d) = dout * static_cast<T>(1) / (x * static_cast<T>(log(10)));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct Log10GradFunctor<ComplexType<T>>
    : public BaseActivationFunctor<ComplexType<T>> {
  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out UNUSED, dOut dout, dX dx) const {
    dx.device(d) = dout * (static_cast<ComplexType<T>>(1) /
                           (x * static_cast<ComplexType<T>>(log(10))))
                              .unaryExpr(Conj<T>());
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct Log1p {
  HOSTDEVICE T operator()(const T& val) const { return std::log1p(val); }
};

template <typename T>
struct Log1p<ComplexType<T>> {
  HOSTDEVICE ComplexType<T> operator()(const ComplexType<T>& val) const {
    return ComplexType<T>(std::log(std::complex<T>(1) + std::complex<T>(val)));
  }
};

template <>
struct Log1p<dtype::float16> {
  HOSTDEVICE dtype::float16 operator()(const dtype::float16& val) const {
    return dtype::float16(std::log1p(static_cast<float>(val)));
  }
};

template <>
struct Log1p<dtype::bfloat16> {
  HOSTDEVICE dtype::bfloat16 operator()(const dtype::bfloat16& val) const {
    return dtype::bfloat16(std::log1p(static_cast<float>(val)));
  }
};

// log1p(x) = natural logarithm of x+1
template <typename T>
struct Log1pFunctor : public BaseActivationFunctor<T> {
  using U = typename std::conditional_t<std::is_integral<T>::value, float, T>;

  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    out.device(d) = x.template cast<U>().unaryExpr(Log1p<U>()).eval();
  }
};

template <typename T>
struct Log1pGradFunctor : public BaseActivationFunctor<T> {
  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out UNUSED, dOut dout, dX dx) const {
    dx.device(d) = dout * (static_cast<T>(1) / (x + static_cast<T>(1)));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct Log1pGradFunctor<ComplexType<T>>
    : public BaseActivationFunctor<ComplexType<T>> {
  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out UNUSED, dOut dout, dX dx) const {
    dx.device(d) = dout * (static_cast<ComplexType<T>>(1) /
                           (x + static_cast<ComplexType<T>>(1)))
                              .unaryExpr(Conj<T>());
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct LogGradGradFunctor : public BaseActivationFunctor<T> {
  template <typename Device>
  void operator()(const Device& dev,
                  const DenseTensor* X,
                  const DenseTensor* ddX,
                  DenseTensor* ddOut,
                  const DenseTensor* dOut,
                  DenseTensor* dX) const {
    auto* d = dev.eigen_device();
    auto ddx = EigenVector<T>::Flatten(
        GET_DATA_SAFELY(ddX, "Input", "DDX", "LogGradGrad"));
    auto x = EigenVector<T>::Flatten(
        GET_DATA_SAFELY(X, "Input", "X", "LogGradGrad"));
    // ddout = ddx / x; dx = -(dout / x) * (ddx / x)
    // calculate dx first, so ddout can inplace ddx
    if (dX) {
      auto dout = EigenVector<T>::Flatten(
          GET_DATA_SAFELY(dOut, "Output", "DOut", "LogGradGrad"));
      auto dx = EigenVector<T>::Flatten(
          GET_DATA_SAFELY(dX, "Output", "DX", "LogGradGrad"));
      dx.device(*d) = dout * static_cast<T>(-1) * ddx / (x * x);
    }
    if (ddOut) {
      auto ddout = EigenVector<T>::Flatten(
          GET_DATA_SAFELY(ddOut, "Output", "DDOut", "LogGradGrad"));
      ddout.device(*d) = ddx * static_cast<T>(1) / x;
    }
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct LogGradGradFunctor<ComplexType<T>>
    : public BaseActivationFunctor<ComplexType<T>> {
  template <typename Device>
  void operator()(const Device& dev,
                  const DenseTensor* X,
                  const DenseTensor* ddX,
                  DenseTensor* ddOut,
                  const DenseTensor* dOut,
                  DenseTensor* dX) const {
    auto* d = dev.eigen_device();
    auto ddx = EigenVector<ComplexType<T>>::Flatten(
        GET_DATA_SAFELY(ddX, "Input", "DDX", "LogGradGrad"));
    auto x = EigenVector<ComplexType<T>>::Flatten(
        GET_DATA_SAFELY(X, "Input", "X", "LogGradGrad"));
    // ddout = ddx / x; dx = -(dout / x) * (ddx / x)
    // calculate dx first, so ddout can inplace ddx
    if (dX) {
      auto dout = EigenVector<ComplexType<T>>::Flatten(
          GET_DATA_SAFELY(dOut, "Output", "DOut", "LogGradGrad"));
      auto dx = EigenVector<ComplexType<T>>::Flatten(
          GET_DATA_SAFELY(dX, "Output", "DX", "LogGradGrad"));
      dx.device(*d) = dout * static_cast<ComplexType<T>>(-1) * ddx /
                      (x * x).unaryExpr(Conj<T>());
    }
    if (ddOut) {
      auto ddout = EigenVector<ComplexType<T>>::Flatten(
          GET_DATA_SAFELY(ddOut, "Output", "DDOut", "LogGradGrad"));
      ddout.device(*d) =
          ddx * static_cast<ComplexType<T>>(1) / x.unaryExpr(Conj<T>());
    }
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

// HardSwish = min(max(0, x+3), 6) * x / 6
template <typename T>
struct HardSwishFunctor : public BaseActivationFunctor<T> {
  float threshold;
  float scale;
  float offset;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"threshold", &threshold}, {"scale", &scale}, {"offset", &offset}};
  }

  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    out.device(d) = (x + static_cast<T>(offset))  // NOLINT
                        .cwiseMax(static_cast<T>(0))
                        .cwiseMin(static_cast<T>(threshold)) *
                    x / static_cast<T>(scale);
  }
};

template <typename T>
struct HardSwishGradFunctor : public BaseActivationFunctor<T> {
  float threshold;
  float scale;
  float offset;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"threshold", &threshold}, {"scale", &scale}, {"offset", &offset}};
  }
  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out UNUSED, dOut dout, dX dx) const {
    auto tmp =
        ((x + static_cast<T>(offset)) < static_cast<T>(threshold))  // NOLINT
            .template cast<T>();
    dx.device(d) =
        dout *
        (((x + static_cast<T>(offset)) > static_cast<T>(0)).template cast<T>() *
             (static_cast<T>(2) * x + static_cast<T>(offset)) /
             static_cast<T>(scale) * tmp +
         static_cast<T>(1) * (static_cast<T>(1) - tmp));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};
template <typename T>
struct HardSwishGradFunctor<ComplexType<T>>
    : public BaseActivationFunctor<ComplexType<T>> {
  float threshold;
  float scale;
  float offset;

  typename BaseActivationFunctor<ComplexType<T>>::AttrPair GetAttrs() {
    return {{"threshold", &threshold}, {"scale", &scale}, {"offset", &offset}};
  }
  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out UNUSED, dOut dout, dX dx) const {
    auto offset_t = static_cast<ComplexType<T>>(offset);
    auto threshold_t = static_cast<ComplexType<T>>(threshold);
    auto one = static_cast<ComplexType<T>>(1);
    auto zero = static_cast<ComplexType<T>>(0);
    auto two = static_cast<ComplexType<T>>(2);
    auto scale_t = static_cast<ComplexType<T>>(scale);
    auto tmp1 = ((x + offset_t) < threshold_t)  // NOLINT
                    .template cast<ComplexType<T>>();
    auto tmp2 = ((x + offset_t) > zero).template cast<ComplexType<T>>();
    // dx = 0, when x <= -offset
    // dout , when x >= threshold - offset
    // dout * (2 * x / scale + offset / scale), otherwise
    // threshold = scale = 6, offset = 3 by default
    dx.device(d) = dout * (tmp2 * (two * x + offset_t) / scale_t * tmp1 +
                           one * (one - tmp1))
                              .unaryExpr(Conj<T>());
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};
template <typename T>
struct SwishFunctor : public BaseActivationFunctor<T> {
  float beta;
  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"beta", &beta}};
  }

  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    out.device(d) =
        x / (static_cast<T>(1) + (static_cast<T>(-beta) * x).exp());  // NOLINT
  }
};

template <typename T>
struct SwishGradFunctor : public BaseActivationFunctor<T> {
  typename BaseActivationFunctor<T>::AttrPair GetAttrs() { return {{}}; }

  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(Device d, X x, Out fake_out UNUSED, dOut dout, dX dx) const {
    float beta = 1.0;
    auto temp1 = static_cast<T>(1) /
                 (static_cast<T>(1) + (static_cast<T>(-beta) * x).exp());
    auto out = x * temp1;
    auto temp2 = temp1 * (static_cast<T>(1) - (static_cast<T>(beta) * out));
    dx.device(d) = dout * ((static_cast<T>(beta) * out) + temp2);
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

// FIXME(qijun) https://github.com/PaddlePaddle/Paddle/issues/5198
template <typename T>
struct PowFunctor : public BaseActivationFunctor<T> {
  float factor;
  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"factor", &factor}};
  }

  T factor_cp;
  using ComplexAttrPair = std::vector<std::pair<const char*, T*>>;
  ComplexAttrPair GetComplexAttrs() { return {{"factor", &factor_cp}}; }

  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    out.device(d) = x.pow(static_cast<T>(factor));  // NOLINT
  }
};

template <typename T>
struct PowFunctor<ComplexType<T>>
    : public BaseActivationFunctor<ComplexType<T>> {
  phi::dtype::complex<T> factor;

  using ComplexAttrPair =
      std::vector<std::pair<const char*, phi::dtype::complex<T>*>>;
  ComplexAttrPair GetComplexAttrs() { return {{"factor", &factor}}; }

  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    out.device(d) = x.pow(factor);  // NOLINT
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct PowGradFunctor : public BaseActivationFunctor<T> {
  float factor;
  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"factor", &factor}};
  }
  T factor_cp;
  using ComplexAttrPair = std::vector<std::pair<const char*, T*>>;
  ComplexAttrPair GetComplexAttrs() { return {{"factor", &factor_cp}}; }
  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out UNUSED, dOut dout, dX dx) const {
    dx.device(d) = dout * static_cast<T>(factor) *
                   x.pow(static_cast<T>(factor) - static_cast<T>(1));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct PowGradFunctor<ComplexType<T>>
    : public BaseActivationFunctor<ComplexType<T>> {
  ComplexType<T> factor;
  using ComplexAttrPair =
      std::vector<std::pair<const char*, phi::dtype::complex<T>*>>;
  ComplexAttrPair GetComplexAttrs() { return {{"factor", &factor}}; }
  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out UNUSED, dOut dout, dX dx) const {
    auto a_ = x.unaryExpr(RealCp<T>());
    auto b_ = x.unaryExpr(ImagCp<T>());
    auto c_2 = factor.real / 2;
    auto d_2 = factor.imag / 2;
    auto c_ = factor.real;
    auto d_ = factor.imag;
    auto arctan_ = (b_ / a_).unaryExpr(Atan<T>());
    auto square_ = a_ * a_ + b_ * b_;
    auto e_ = (square_.log() * c_2 - d_ * arctan_).exp();
    auto v_ = square_.log() * d_2 + c_ * arctan_;

    auto ux = e_ / square_ *
              ((a_ * c_ + b_ * d_) * v_.unaryExpr(Cosine<T>()) +
               (b_ * c_ - a_ * d_) * v_.unaryExpr(Sine<T>()));
    auto uy = e_ / square_ *
              ((b_ * c_ - a_ * d_) * v_.unaryExpr(Cosine<T>()) -
               (b_ * d_ + a_ * c_) * v_.unaryExpr(Sine<T>()));

    dx.device(d) = ux.binaryExpr(uy, ComplexAssemble<T>());
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

// floor(x) = flooring(x)
template <typename T>
struct FloorFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    out.device(d) = x.floor();
  }
};

// round(x) = [x]
template <typename T>
struct RoundFunctor : public BaseActivationFunctor<T> {
  int decimals;

  std::vector<std::pair<const char*, int*>> GetAttrs() {
    return {{"decimals", &decimals}};
  }

  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    if (decimals == 0) {
      out.device(d) = x.round();
    } else if (decimals > 0) {
      auto ten_pow_decimals = static_cast<T>(std::pow(10, decimals));
      out.device(d) = (x * ten_pow_decimals).round() / ten_pow_decimals;
    } else {
      auto ten_pow_decimals = static_cast<T>(std::pow(10, -decimals));
      out.device(d) = (x / ten_pow_decimals).round() * ten_pow_decimals;
    }
  }
};

// ceil(x) = ceiling(x)
template <typename T>
struct CeilFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    out.device(d) = x.ceil();
  }
};

template <typename T>
struct NegativeFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    out.device(d) = -x;
  }
};

template <typename T>
struct ZeroGradFunctor : public BaseActivationFunctor<T> {
  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(
      Device d, X x UNUSED, Out out, dOut dout UNUSED, dX dx) const {
    dx.device(d) = static_cast<T>(0) * out;
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() {
    return ActBwdOpFwdDeps::kNoDeps;
  }
};

template <typename T>
struct SqrtGradGradFunctor : public BaseActivationFunctor<T> {
  template <typename Device>
  void operator()(const Device& dev,
                  const DenseTensor* Out,
                  const DenseTensor* dX,
                  const DenseTensor* ddX,
                  DenseTensor* dOut,
                  DenseTensor* ddOut) const {
    auto* d = dev.eigen_device();
    auto ddx = EigenVector<T>::Flatten(
        GET_DATA_SAFELY(ddX, "Input", "DDX", "SqrtGradGrad"));
    auto out = EigenVector<T>::Flatten(
        GET_DATA_SAFELY(Out, "Output", "Out", "SqrtGradGrad"));
    // sqrt GradGrad: ddy = 0.5 * ddx / y, dy = -1 * dx * ddx
    // calculate dy first, so ddy can inplace ddx
    if (dOut) {
      auto dx = EigenVector<T>::Flatten(
          GET_DATA_SAFELY(dX, "Output", "DX", "SqrtGradGrad"));
      auto dout = EigenVector<T>::Flatten(
          GET_DATA_SAFELY(dOut, "Output", "DOut", "SqrtGradGrad"));
      dout.device(*d) = dx * ddx * static_cast<T>(-1) / out;
    }
    if (ddOut) {
      auto ddout = EigenVector<T>::Flatten(
          GET_DATA_SAFELY(ddOut, "Output", "DDOut", "SqrtGradGrad"));
      ddout.device(*d) = ddx * static_cast<T>(0.5) / out;
    }
  }
  static constexpr ActBwdOpFwdDeps FwdDeps() {
    return ActBwdOpFwdDeps::kDepOut;
  }
};

template <typename T>
struct RsqrtGradGradFunctor : public BaseActivationFunctor<T> {
  template <typename Device>
  void operator()(const Device& dev,
                  const DenseTensor* Out,
                  const DenseTensor* dX,
                  const DenseTensor* ddX,
                  DenseTensor* dOut,
                  DenseTensor* ddOut) const {
    auto* d = dev.eigen_device();
    auto ddx = EigenVector<T>::Flatten(
        GET_DATA_SAFELY(ddX, "Input", "DDX", "RsqrtGradGrad"));
    auto out = EigenVector<T>::Flatten(
        GET_DATA_SAFELY(Out, "Output", "Out", "RsqrtGradGrad"));

    // rsqrt GradGrad: ddy = -0.5 * ddx * y * y * y, dy = (3/y) * dx * ddx
    if (dOut) {
      auto dx = EigenVector<T>::Flatten(
          GET_DATA_SAFELY(dX, "Output", "DX", "RsqrtGradGrad"));
      auto dout = EigenVector<T>::Flatten(
          GET_DATA_SAFELY(dOut, "Output", "DOut", "RsqrtGradGrad"));
      dout.device(*d) = (static_cast<T>(3.0) / out) * dx * ddx;
    }
    if (ddOut) {
      auto ddout = EigenVector<T>::Flatten(
          GET_DATA_SAFELY(ddOut, "Output", "DDOut", "RsqrtGradGrad"));
      ddout.device(*d) = ddx * static_cast<T>(-0.5) * out * out * out;
    }
  }
  static constexpr ActBwdOpFwdDeps FwdDeps() {
    return ActBwdOpFwdDeps::kDepOut;
  }
};

template <typename T>
struct CELUFunctor : public BaseActivationFunctor<T> {
  float alpha;
  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"alpha", &alpha}};
  }

  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    out.device(d) =
        (x < static_cast<T>(0))
            .select(static_cast<T>(alpha) * ((x / static_cast<T>(alpha)).exp() -
                                             static_cast<T>(1)),  // NOLINT
                    x);
  }
};

template <typename T>
struct CELUGradFunctor : public BaseActivationFunctor<T> {
  float alpha;
  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"alpha", &alpha}};
  }
  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out UNUSED, dOut dout, dX dx) const {
    auto temp_a_pos = static_cast<T>(alpha > 0);  // NOLINT
    auto temp_a_neg = static_cast<T>(alpha <= 0);
    auto temp_x_pos = (x > static_cast<T>(0)).template cast<T>();
    auto temp_x_neg = (x <= static_cast<T>(0)).template cast<T>();

    // dx = dout, if alpha > 0 and x > 0
    // dx = dout * (x/alpha).exp(), if alpha > 0 and x <= 0
    // dx = dout , if alpha < 0 and x > 0
    // dx = dout * (x/alpha).exp(), if alpha < 0 and x <=0
    dx.device(d) =
        dout * temp_a_pos * temp_x_pos +
        dout * (x / static_cast<T>(alpha)).exp() * temp_a_pos * temp_x_neg +
        dout * temp_a_neg * temp_x_pos +
        dout * (x / static_cast<T>(alpha)).exp() * temp_a_neg * temp_x_neg;
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct CELUGradGradFunctor : public BaseActivationFunctor<T> {
  float alpha;
  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"alpha", &alpha}};
  }
  template <typename Device>
  void operator()(const Device& dev,
                  const DenseTensor* X,
                  const DenseTensor* dOut,
                  const DenseTensor* ddX,
                  DenseTensor* dX,
                  DenseTensor* ddOut) const {
    auto* d = dev.eigen_device();
    auto ddx = EigenVector<T>::Flatten(
        GET_DATA_SAFELY(ddX, "Input", "DDX", "CELUGradGrad"));
    auto x = EigenVector<T>::Flatten(
        GET_DATA_SAFELY(X, "Input", "X", "CELUGradGrad"));

    if (dX) {
      auto dx = EigenVector<T>::Flatten(
          GET_DATA_SAFELY(dX, "Output", "DX", "CELUGradGrad"));
      auto dout = EigenVector<T>::Flatten(
          GET_DATA_SAFELY(dOut, "Output", "DOut", "CELUGradGrad"));
      dx.device(*d) = ddx * dout / static_cast<T>(alpha) *
                      (x / static_cast<T>(alpha)).exp() *
                      (x <= static_cast<T>(0)).template cast<T>();
    }

    if (ddOut) {
      auto ddout = EigenVector<T>::Flatten(
          GET_DATA_SAFELY(ddOut, "Output", "DDOut", "CELUGradGrad"));
      ddout.device(*d) = ddx * ((x > static_cast<T>(0)).template cast<T>() +
                                (x / static_cast<T>(alpha)).exp() *
                                    (x <= static_cast<T>(0)).template cast<T>())
                                   .template cast<T>();
    }
  }
  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct SquareGradGradFunctor : public BaseActivationFunctor<T> {
  template <typename Device>
  void operator()(const Device& dev,
                  const DenseTensor* X,
                  const DenseTensor* dOut,
                  const DenseTensor* ddX,
                  DenseTensor* dX,
                  DenseTensor* ddOut) const {
    auto* d = dev.eigen_device();
    auto ddx = EigenVector<T>::Flatten(
        GET_DATA_SAFELY(ddX, "Input", "DDX", "SquareGradGrad"));
    auto x = EigenVector<T>::Flatten(
        GET_DATA_SAFELY(X, "Input", "X", "SquareGradGrad"));
    // square GradGrad: ddy=2x*ddx, dx=2dy*ddx
    // calculate dx first, so ddy can inplace ddx
    if (dX) {
      auto dx = EigenVector<T>::Flatten(
          GET_DATA_SAFELY(dX, "Output", "DX", "SquareGradGrad"));
      auto dout = EigenVector<T>::Flatten(
          GET_DATA_SAFELY(dOut, "Output", "DOut", "SquareGradGrad"));
      dx.device(*d) = ddx * static_cast<T>(2) * dout;
    }
    if (ddOut) {
      auto ddout = EigenVector<T>::Flatten(
          GET_DATA_SAFELY(ddOut, "Output", "DDOut", "SquareGradGrad"));
      ddout.device(*d) = ddx * static_cast<T>(2) * x;
    }
  }
  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

#if defined(__NVCC__) || defined(__HIPCC__) || defined(__xpu__)

template <typename T>
struct CudaLogitFunctor : public BaseActivationFunctor<T> {
  using MT = typename phi::dtype::MPTypeTrait<T>::Type;

  MT zero = static_cast<MT>(0.0f);
  MT one = static_cast<MT>(1.0f);
  float eps;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"eps", &eps}};
  }

  // logit(x) = ln(x/(1-x))
  __device__ __forceinline__ T operator()(const T arg_x) const {
    MT x = static_cast<MT>(arg_x);
    MT y = min(x, (one - static_cast<MT>(eps)));
    y = max(y, static_cast<MT>(eps));

    if (!eps) {
      y = x < zero || x > one ? static_cast<T>(NAN) : log(y / (one - y));
    } else {
      y = log(y / (one - y));
    }
    return static_cast<T>(y);
  }
};

template <typename T>
struct CudaLogitGradFunctor : public BaseActivationFunctor<T> {
  using MT = typename phi::dtype::MPTypeTrait<T>::Type;

  float eps;
  MT zero = static_cast<MT>(0.0f);
  MT one = static_cast<MT>(1.0f);

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"eps", &eps}};
  }
  // logit(x)' = 1/(x*(1-x))
  __device__ __forceinline__ T operator()(const T dout, const T arg_x) const {
    MT x = static_cast<MT>(arg_x);
    MT dx = (x < static_cast<MT>(eps) || x > one - static_cast<MT>(eps))
                ? zero
                : (static_cast<MT>(dout) / (x * (one - x)));
    return static_cast<T>(dx);
  }
  static constexpr ActBwdOpFwdDeps FwdDeps() {
    return ActBwdOpFwdDeps::kDepOut;
  }
};

template <typename T>
struct CudaReluFunctor : public BaseActivationFunctor<T> {
  T zero = static_cast<T>(0.0f);

  // relu(x) = max(x, 0)
  __device__ __forceinline__ T operator()(const T x) const {
    return x < zero ? zero : x;
  }
};

template <typename T>
struct CudaReluGradFunctor : public BaseActivationFunctor<T> {
  T zero = static_cast<T>(0.0f);

  // dx = dout * (out > 0)
  __device__ __forceinline__ T operator()(const T dout, const T out) const {
    return out > zero ? dout : zero;
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() {
    return ActBwdOpFwdDeps::kDepOut;
  }
};

template <typename T>
struct CudaCosFunctor : public BaseActivationFunctor<T> {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;

  // cos(x) = cos(x)
  __device__ __forceinline__ T operator()(const T arg_x) const {
    MPType x = static_cast<MPType>(arg_x);
    return static_cast<T>(cos(x));
  }
};

template <typename T>
struct CudaCosGradFunctor : public BaseActivationFunctor<T> {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;

  // dx = dout * (-sin(x))
  __device__ __forceinline__ T operator()(const T arg_dout,
                                          const T arg_x) const {
    MPType dout = static_cast<MPType>(arg_dout);
    MPType x = static_cast<MPType>(arg_x);
    return static_cast<T>(-dout * sin(x));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct CudaCosGradFunctor<ComplexType<T>>
    : public BaseActivationFunctor<ComplexType<T>> {
  // dx = dout * (-sin(x))
  __device__ __forceinline__ ComplexType<T> operator()(
      const ComplexType<T> dout, const ComplexType<T> x) const {
    return static_cast<ComplexType<T>>(-dout * conj(sin(x)));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct CudaExpFunctor : public BaseActivationFunctor<T> {
  // exp(x) = expf(x)
  using U = typename std::conditional_t<std::is_integral<T>::value, float, T>;

  __device__ __forceinline__ U operator()(const T x) const {
    return static_cast<U>(expf(static_cast<float>(x)));
  }
};

template <>
struct CudaExpFunctor<double> : public BaseActivationFunctor<double> {
  // exp(x) = exp(x)
  __device__ __forceinline__ double operator()(const double x) const {
    return exp(x);
  }
};

template <typename T>
struct CudaExpFunctor<ComplexType<T>>
    : public BaseActivationFunctor<ComplexType<T>> {
  // exp(x) = exp(x)
  __device__ __forceinline__ ComplexType<T> operator()(
      const ComplexType<T> x) const {
    return static_cast<ComplexType<T>>(exp(x));
  }
};

template <typename T>
struct CudaSeluFunctor : public BaseActivationFunctor<T> {
  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"scale", &scale}, {"alpha", &alpha}};
  }

  __device__ __forceinline__ T operator()(const T x) const {
    using MT =
        typename std::conditional<(sizeof(T) > sizeof(float)), T, float>::type;
    MT res = static_cast<MT>(x);
    if (x <= zero) {
      res = alpha * expf(res) - alpha;
    }
    res *= scale;
    return static_cast<T>(res);
  }

 private:
  float scale;
  float alpha;
  T zero = static_cast<T>(0.0f);
};

template <>
struct CudaSeluFunctor<double> : public BaseActivationFunctor<double> {
  typename BaseActivationFunctor<double>::AttrPair GetAttrs() {
    return {{"scale", &scale}, {"alpha", &alpha}};
  }

  __device__ __forceinline__ double operator()(const double x) const {
    double res = x;
    double alpha_cast = static_cast<double>(alpha);
    double scale_cast = static_cast<double>(scale);
    if (res <= zero) {
      res = alpha_cast * exp(res) - alpha_cast;
    }
    res *= scale_cast;
    return res;
  }

 private:
  float scale;
  float alpha;
  double zero = static_cast<double>(0.0f);
};

template <typename T>
struct CudaSquareFunctor : public BaseActivationFunctor<T> {
  // square(x) = x * x
  __device__ __forceinline__ T operator()(const T x) const { return x * x; }
};

template <typename T>
struct CudaSquareGradFunctor : public BaseActivationFunctor<T> {
  T two = static_cast<T>(2.0f);

  // dx = dout * 2 * x
  __device__ __forceinline__ T operator()(const T dout, const T x) const {
    return dout * two * x;
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct CudaSquareGradFunctor<ComplexType<T>>
    : public BaseActivationFunctor<ComplexType<T>> {
  ComplexType<T> two = static_cast<ComplexType<T>>(2.0f);

  // dx = dout * 2 * x
  __device__ __forceinline__ ComplexType<T> operator()(
      const ComplexType<T> dout, const ComplexType<T> x) const {
    return static_cast<ComplexType<T>>(dout * two * conj(x));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct CudaExpGradFunctor : public BaseActivationFunctor<T> {
  // dx = dout * out
  __device__ __forceinline__ T operator()(const T dout, const T out) const {
    return dout * out;
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() {
    return ActBwdOpFwdDeps::kDepOut;
  }
};

template <typename T>
struct CudaExpGradFunctor<ComplexType<T>>
    : public BaseActivationFunctor<ComplexType<T>> {
  // dx = dout * exp(x)
  __device__ __forceinline__ ComplexType<T> operator()(
      const ComplexType<T> dout, const ComplexType<T> out) const {
    return static_cast<ComplexType<T>>(dout * conj(out));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() {
    return ActBwdOpFwdDeps::kDepOut;
  }
};

template <typename T>
struct CudaReciprocalFunctor : public BaseActivationFunctor<T> {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
  MPType one = static_cast<MPType>(1.0f);

  __device__ __forceinline__ T operator()(const T x) const {
    return static_cast<T>(one / static_cast<MPType>(x));
  }
};

template <typename T>
struct CudaReciprocalGradFunctor : public BaseActivationFunctor<T> {
  // dx = -dout * out^2
  __device__ __forceinline__ T operator()(const T dout, const T out) const {
    return -dout * out * out;
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() {
    return ActBwdOpFwdDeps::kDepOut;
  }
};

template <typename T>
struct CudaReciprocalGradFunctor<ComplexType<T>>
    : public BaseActivationFunctor<ComplexType<T>> {
  // dx = -dout * out^2
  __device__ __forceinline__ ComplexType<T> operator()(
      const ComplexType<T> dout, const ComplexType<T> out) const {
    return -dout * conj(out * out);
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() {
    return ActBwdOpFwdDeps::kDepOut;
  }
};

template <typename T>
struct CudaExpm1Functor : public BaseActivationFunctor<T> {
  using U = typename std::conditional_t<std::is_integral<T>::value, float, T>;

  // expm1(x) = expm1f(x)
  __device__ __forceinline__ U operator()(const T x) const {
    return static_cast<U>(::expm1f(static_cast<float>(x)));
  }
};

template <>
struct CudaExpm1Functor<double> : public BaseActivationFunctor<double> {
  // expm1(x) = expm1(x)
  __device__ __forceinline__ double operator()(const double x) const {
    return ::expm1(x);
  }
};

template <typename T>
struct CudaExpm1Functor<ComplexType<T>>
    : public BaseActivationFunctor<ComplexType<T>> {
  __device__ __forceinline__ ComplexType<T> operator()(
      const ComplexType<T> x) const {
    return static_cast<ComplexType<T>>(Expm1<ComplexType<T>>()(x));
  }
};

template <typename T>
struct CudaExpm1GradFunctor : public BaseActivationFunctor<T> {
  // dx = dout * out
  __device__ __forceinline__ T operator()(const T dout, const T out) const {
    return dout * out + dout;
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() {
    return ActBwdOpFwdDeps::kDepOut;
  }
};

template <typename T>
struct CudaExpm1GradFunctor<ComplexType<T>>
    : public BaseActivationFunctor<ComplexType<T>> {
  // dx = dout * exp(x)
  __device__ __forceinline__ ComplexType<T> operator()(
      const ComplexType<T> dout, const ComplexType<T> out) const {
    return static_cast<ComplexType<T>>(dout * conj(out) + dout);
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() {
    return ActBwdOpFwdDeps::kDepOut;
  }
};

template <typename T>
struct CudaSinFunctor : public BaseActivationFunctor<T> {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;

  // sin(x) = sin(x)
  __device__ __forceinline__ T operator()(const T arg_x) const {
    MPType x = static_cast<MPType>(arg_x);
    return static_cast<T>(sin(x));
  }
};

template <typename T>
struct CudaSinGradFunctor : public BaseActivationFunctor<T> {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;

  // dx = dout * cos(x)
  __device__ __forceinline__ T operator()(const T arg_dout,
                                          const T arg_x) const {
    MPType dout = static_cast<MPType>(arg_dout);
    MPType x = static_cast<MPType>(arg_x);
    return static_cast<T>(dout * cos(x));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct CudaSinGradFunctor<ComplexType<T>>
    : public BaseActivationFunctor<ComplexType<T>> {
  // dx = dout * cos(x)
  __device__ __forceinline__ ComplexType<T> operator()(
      const ComplexType<T> dout, const ComplexType<T> x) const {
    return static_cast<ComplexType<T>>(dout * conj(cos(x)));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct CudaTanFunctor : public BaseActivationFunctor<T> {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;

  // tan(x) = tan(x)
  __device__ __forceinline__ T operator()(const T arg_x) const {
    MPType x = static_cast<MPType>(arg_x);
    return static_cast<T>(tan(x));
  }
};

template <typename T>
struct CudaTanGradFunctor : public BaseActivationFunctor<T> {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;

  // dx = dout / cos(x)^2
  __device__ __forceinline__ T operator()(const T arg_dout,
                                          const T arg_x) const {
    MPType dout = static_cast<MPType>(arg_dout);
    MPType x = static_cast<MPType>(arg_x);
    return static_cast<T>(dout / (cos(x) * cos(x)));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct CudaTanGradFunctor<ComplexType<T>>
    : public BaseActivationFunctor<ComplexType<T>> {
  // dx = dout / cos(x)^2
  __device__ __forceinline__ ComplexType<T> operator()(
      const ComplexType<T> dout, const ComplexType<T> x) const {
    return static_cast<ComplexType<T>>(dout / conj(cos(x) * cos(x)));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct CudaAsinFunctor : public BaseActivationFunctor<T> {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;

  // asin(x) = asin(x)
  __device__ __forceinline__ T operator()(const T arg_x) const {
    MPType x = static_cast<MPType>(arg_x);
    return static_cast<T>(asin(x));
  }
};

template <typename T>
struct CudaAsinGradFunctor : public BaseActivationFunctor<T> {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
  MPType one = static_cast<MPType>(1.0f);

  // dx = dout / sqrt(1 - x^2)
  __device__ __forceinline__ T operator()(const T arg_dout,
                                          const T arg_x) const {
    MPType dout = static_cast<MPType>(arg_dout);
    MPType x = static_cast<MPType>(arg_x);
    return static_cast<T>(dout / sqrt(one - x * x));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct CudaAsinGradFunctor<ComplexType<T>>
    : public BaseActivationFunctor<ComplexType<T>> {
  ComplexType<T> one = static_cast<ComplexType<T>>(1.0f);

  // dx = dout / sqrt(1 - x^2)
  __device__ __forceinline__ ComplexType<T> operator()(
      const ComplexType<T> dout, const ComplexType<T> x) const {
    return static_cast<ComplexType<T>>(dout / conj(sqrt(one - x * x)));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct CudaAcosFunctor : public BaseActivationFunctor<T> {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;

  // acos(x) = acos(x)
  __device__ __forceinline__ T operator()(const T arg_x) const {
    MPType x = static_cast<MPType>(arg_x);
    return static_cast<T>(acos(x));
  }
};

template <typename T>
struct CudaAcosGradFunctor : public BaseActivationFunctor<T> {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
  MPType one = static_cast<MPType>(1.0f);

  // dx = -dout / sqrt(1 - x^2)
  __device__ __forceinline__ T operator()(const T arg_dout,
                                          const T arg_x) const {
    MPType dout = static_cast<MPType>(arg_dout);
    MPType x = static_cast<MPType>(arg_x);
    return static_cast<T>(-dout / sqrt(one - x * x));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct CudaAcosGradFunctor<ComplexType<T>>
    : public BaseActivationFunctor<ComplexType<T>> {
  ComplexType<T> one = static_cast<ComplexType<T>>(1.0f);

  // dx = -dout / sqrt(1 - x^2)
  __device__ __forceinline__ ComplexType<T> operator()(
      const ComplexType<T> dout, const ComplexType<T> x) const {
    return static_cast<ComplexType<T>>(-dout / conj(sqrt(one - x * x)));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct CudaCoshFunctor : public BaseActivationFunctor<T> {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;

  // cosh(x) = cosh(x)
  __device__ __forceinline__ T operator()(const T arg_x) const {
    MPType x = static_cast<MPType>(arg_x);
    return static_cast<T>(cosh(x));
  }
};

template <typename T>
struct CudaCoshGradFunctor : public BaseActivationFunctor<T> {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;

  // dx = dout * sinh(x)
  __device__ __forceinline__ T operator()(const T arg_dout,
                                          const T arg_x) const {
    MPType dout = static_cast<MPType>(arg_dout);
    MPType x = static_cast<MPType>(arg_x);
    return static_cast<T>(dout * sinh(x));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct CudaCoshGradFunctor<ComplexType<T>>
    : public BaseActivationFunctor<ComplexType<T>> {
  // dx = dout * sinh(x)
  __device__ __forceinline__ ComplexType<T> operator()(
      const ComplexType<T> dout, const ComplexType<T> x) const {
    return static_cast<ComplexType<T>>(dout * conj(sinh(x)));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct CudaSinhFunctor : public BaseActivationFunctor<T> {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;

  // sinh(x) = sinh(x)
  __device__ __forceinline__ T operator()(const T arg_x) const {
    MPType x = static_cast<MPType>(arg_x);
    return static_cast<T>(sinh(x));
  }
};

template <typename T>
struct CudaSinhGradFunctor : public BaseActivationFunctor<T> {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;

  // dx = dout * cosh(x)
  __device__ __forceinline__ T operator()(const T arg_dout,
                                          const T arg_x) const {
    MPType dout = static_cast<MPType>(arg_dout);
    MPType x = static_cast<MPType>(arg_x);
    return static_cast<T>(dout * cosh(x));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct CudaSinhGradFunctor<ComplexType<T>>
    : public BaseActivationFunctor<ComplexType<T>> {
  // dx = dout * cosh(x)
  __device__ __forceinline__ ComplexType<T> operator()(
      const ComplexType<T> dout, const ComplexType<T> x) const {
    return static_cast<ComplexType<T>>(dout * conj(cosh(x)));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct CudaAcoshFunctor : public BaseActivationFunctor<T> {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;

  // Acosh(x) = acosh(x)
  __device__ __forceinline__ T operator()(const T arg_x) const {
    MPType x = static_cast<MPType>(arg_x);
    return static_cast<T>(acosh(x));
  }
};

template <typename T>
struct CudaAcoshGradFunctor : public BaseActivationFunctor<T> {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
  MPType one = static_cast<MPType>(1.0f);
  // dx = dout * 1 / sqrt(x^2 - 1)
  __device__ __forceinline__ T operator()(const T arg_dout,
                                          const T arg_x) const {
    MPType dout = static_cast<MPType>(arg_dout);
    MPType x = static_cast<MPType>(arg_x);
    return static_cast<T>(dout * one / sqrt(x * x - one));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct CudaAcoshGradFunctor<ComplexType<T>>
    : public BaseActivationFunctor<ComplexType<T>> {
  ComplexType<T> one = static_cast<ComplexType<T>>(1.0f);
  // dx = dout * 1 / sqrt(x^2 - 1)
  __device__ __forceinline__ ComplexType<T> operator()(
      const ComplexType<T> dout, const ComplexType<T> x) const {
    return static_cast<ComplexType<T>>(dout * conj(one / sqrt(x * x - one)));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct CudaAsinhFunctor : public BaseActivationFunctor<T> {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;

  // Asinh(x) = asinh(x)
  __device__ __forceinline__ T operator()(const T arg_x) const {
    MPType x = static_cast<MPType>(arg_x);
    return static_cast<T>(asinh(x));
  }
};

template <typename T>
struct CudaAsinhGradFunctor : public BaseActivationFunctor<T> {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
  MPType one = static_cast<MPType>(1.0f);

  // dx = dout * 1/sqrt(x^2 + 1)
  __device__ __forceinline__ T operator()(const T arg_dout,
                                          const T arg_x) const {
    MPType dout = static_cast<MPType>(arg_dout);
    MPType x = static_cast<MPType>(arg_x);
    return static_cast<T>(dout * one / sqrt(x * x + one));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct CudaAsinhGradFunctor<ComplexType<T>>
    : public BaseActivationFunctor<ComplexType<T>> {
  ComplexType<T> one = static_cast<ComplexType<T>>(1.0f);

  // dx = dout * 1/sqrt(x^2 + 1)
  __device__ __forceinline__ ComplexType<T> operator()(
      const ComplexType<T> dout, const ComplexType<T> x) const {
    return static_cast<ComplexType<T>>(dout * conj(one / sqrt(x * x + one)));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct CudaAtanhFunctor : public BaseActivationFunctor<T> {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;

  // Atanh(x) = atanh(x)
  __device__ __forceinline__ T operator()(const T arg_x) const {
    MPType x = static_cast<MPType>(arg_x);
    return static_cast<T>(atanh(x));
  }
};

template <typename T>
struct CudaSTanhFunctor : public BaseActivationFunctor<T> {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
  float scale_a;
  float scale_b;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"scale_a", &scale_a}, {"scale_b", &scale_b}};
  }

  // stanh(x) = b * tanh(a * x)
  __device__ __forceinline__ T operator()(const T arg_x) const {
    MPType x = static_cast<MPType>(arg_x);
    MPType a = static_cast<MPType>(scale_a);
    MPType b = static_cast<MPType>(scale_b);
    return static_cast<T>(b * tanh(a * x));
  }
};

template <typename T>
struct CudaSTanhGradFunctor : public BaseActivationFunctor<T> {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
  MPType one = static_cast<MPType>(1.0f);
  float scale_a;
  float scale_b;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"scale_a", &scale_a}, {"scale_b", &scale_b}};
  }

  // dx = dout * a * b * (1 - tanh(a * x) * tanh(a * x))
  __device__ __forceinline__ T operator()(const T arg_dout,
                                          const T arg_x) const {
    MPType dout = static_cast<MPType>(arg_dout);
    MPType x = static_cast<MPType>(arg_x);
    MPType a = static_cast<MPType>(scale_a);
    MPType b = static_cast<MPType>(scale_b);
    MPType temp = tanh(a * x);
    return static_cast<T>(dout * a * b * (one - temp * temp));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct CudaSTanhGradFunctor<ComplexType<T>>
    : public BaseActivationFunctor<ComplexType<T>> {
  ComplexType<T> one = static_cast<ComplexType<T>>(1.0f);
  float scale_a;
  float scale_b;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"scale_a", &scale_a}, {"scale_b", &scale_b}};
  }

  // dx = dout * a * b * (1 - tanh(a * x) * tanh(a * x))
  __device__ __forceinline__ ComplexType<T> operator()(
      const ComplexType<T> arg_dout, const ComplexType<T> arg_x) const {
    ComplexType<T> dout = static_cast<ComplexType<T>>(arg_dout);
    ComplexType<T> x = static_cast<ComplexType<T>>(arg_x);
    ComplexType<T> a = static_cast<ComplexType<T>>(scale_a);
    ComplexType<T> b = static_cast<ComplexType<T>>(scale_b);
    ComplexType<T> temp = tanh(a * x);
    return static_cast<ComplexType<T>>(dout *
                                       conj(a * b * (one - temp * temp)));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct CudaSoftplusFunctor : public BaseActivationFunctor<T> {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
  MPType one = static_cast<MPType>(1.0f);
  float beta;
  float threshold;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"beta", &beta}, {"threshold", &threshold}};
  }

  // softplus(x) = beta * x > threshold ? x : log(1 + exp(beta * x)) / beta
  __device__ __forceinline__ T operator()(const T arg_x) const {
    MPType x = static_cast<MPType>(arg_x);
    MPType b = static_cast<MPType>(beta);
    MPType t = static_cast<MPType>(threshold);
    MPType x_beta = x * static_cast<MPType>(beta);
    return static_cast<T>(x_beta > t ? x : log(one + exp(x_beta)) / b);
  }
};

template <typename T>
struct CudaSoftplusGradFunctor : public BaseActivationFunctor<T> {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
  MPType one = static_cast<MPType>(1.0f);
  float beta;
  float threshold;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"beta", &beta}, {"threshold", &threshold}};
  }

  // dx = x * beta > threshold ? dout : dout / (1 + exp(-beta * x))
  __device__ __forceinline__ T operator()(const T arg_dout,
                                          const T arg_x) const {
    MPType dout = static_cast<MPType>(arg_dout);
    MPType x = static_cast<MPType>(arg_x);
    MPType b = static_cast<MPType>(beta);
    MPType t = static_cast<MPType>(threshold);
    MPType x_beta = x * beta;
    return x_beta > t ? arg_dout : static_cast<T>(dout / (one + exp(-x_beta)));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct CudaSoftplusGradFunctor<ComplexType<T>>
    : public BaseActivationFunctor<ComplexType<T>> {
  using MPType = typename phi::dtype::MPTypeTrait<ComplexType<T>>::Type;
  MPType one = static_cast<MPType>(1.0f);
  float beta;
  float threshold;

  typename BaseActivationFunctor<ComplexType<T>>::AttrPair GetAttrs() {
    return {{"beta", &beta}, {"threshold", &threshold}};
  }

  // dx = x * beta > threshold ? dout : dout / (1 + exp(-beta * x))
  __device__ __forceinline__ ComplexType<T> operator()(
      const ComplexType<T> arg_dout, const ComplexType<T> arg_x) const {
    MPType dout = static_cast<MPType>(arg_dout);
    MPType x = static_cast<MPType>(arg_x);
    MPType b = static_cast<MPType>(beta);
    MPType t = static_cast<MPType>(threshold);
    MPType x_beta = x * static_cast<MPType>(beta);
    return x_beta > t
               ? dout
               : static_cast<ComplexType<T>>(dout / conj(one + exp(-x_beta)));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct CudaAtanhGradFunctor : public BaseActivationFunctor<T> {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
  MPType one = static_cast<MPType>(1.0f);
  // dx = dout * 1/(1- x^2)
  __device__ __forceinline__ T operator()(const T arg_dout,
                                          const T arg_x) const {
    MPType dout = static_cast<MPType>(arg_dout);
    MPType x = static_cast<MPType>(arg_x);
    return static_cast<T>(dout * one / (one - x * x));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct CudaAtanhGradFunctor<ComplexType<T>>
    : public BaseActivationFunctor<ComplexType<T>> {
  ComplexType<T> one = static_cast<ComplexType<T>>(1.0f);
  // dx = dout * 1/(1- x^2)
  __device__ __forceinline__ ComplexType<T> operator()(
      const ComplexType<T> dout, const ComplexType<T> x) const {
    return static_cast<ComplexType<T>>(dout * conj(one / (one - x * x)));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct CudaSqrtFunctor : public BaseActivationFunctor<T> {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;

  // sqrt(x) = sqrt(x)
  __device__ __forceinline__ T operator()(const T arg_x) const {
    MPType x = static_cast<MPType>(arg_x);
    return static_cast<T>(sqrt(x));
  }
};

template <typename T>
struct CudaSqrtGradFunctor : public BaseActivationFunctor<T> {
  T one_half = static_cast<T>(0.5f);

  // dx = dout * 0.5 / out
  __device__ __forceinline__ T operator()(const T dout, const T out) const {
    return one_half * dout / out;
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() {
    return ActBwdOpFwdDeps::kDepOut;
  }
};

template <typename T>
struct CudaSqrtGradFunctor<ComplexType<T>>
    : public BaseActivationFunctor<ComplexType<T>> {
  ComplexType<T> one_half = static_cast<ComplexType<T>>(0.5f);

  // dx = dout * 0.5 / out
  __device__ __forceinline__ ComplexType<T> operator()(
      const ComplexType<T> dout, const ComplexType<T> out) const {
    return dout * conj(one_half / out);
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() {
    return ActBwdOpFwdDeps::kDepOut;
  }
};

template <typename T>
struct CudaRsqrtFunctor : public BaseActivationFunctor<T> {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;

  // rsqrt(x) = rsqrt(x)
  __device__ __forceinline__ T operator()(const T arg_x) const {
    MPType x = static_cast<MPType>(arg_x);
    return static_cast<T>(rsqrt(x));
  }
};

template <typename T>
struct CudaRsqrtGradFunctor : public BaseActivationFunctor<T> {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
  MPType minus_one_half = static_cast<MPType>(-0.5f);

  // dx = -0.5 * dout * out^3
  __device__ __forceinline__ T operator()(const T arg_dout,
                                          const T arg_out) const {
    MPType dout = static_cast<MPType>(arg_dout);
    MPType out = static_cast<MPType>(arg_out);
    return static_cast<T>(minus_one_half * dout * out * out * out);
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() {
    return ActBwdOpFwdDeps::kDepOut;
  }
};

template <typename T>
struct CudaAtanFunctor : public BaseActivationFunctor<T> {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;

  // atan(x) = atan(x)
  __device__ __forceinline__ T operator()(const T arg_x) const {
    MPType x = static_cast<MPType>(arg_x);
    return static_cast<T>(atan(x));
  }
};

template <typename T>
struct CudaAtanGradFunctor : public BaseActivationFunctor<T> {
  T one = static_cast<T>(1.0f);

  // dx = dout / (1 + x^2)
  __device__ __forceinline__ T operator()(const T dout, const T x) const {
    return dout / (one + x * x);
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct CudaAtanGradFunctor<ComplexType<T>>
    : public BaseActivationFunctor<ComplexType<T>> {
  ComplexType<T> one = static_cast<ComplexType<T>>(1.0f);

  // dx = dout / (1 + x^2)
  __device__ __forceinline__ ComplexType<T> operator()(
      const ComplexType<T> dout, const ComplexType<T> x) const {
    return dout / conj(one + x * x);
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct CudaTanhFunctor : public BaseActivationFunctor<T> {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;

  // tanh(x) = tanh(x)
  __device__ __forceinline__ T operator()(const T arg_x) const {
    MPType x = static_cast<MPType>(arg_x);
    return static_cast<T>(tanh(x));
  }
};

template <typename T>
struct CudaTanhGradFunctor : public BaseActivationFunctor<T> {
  T one = static_cast<T>(1.0f);

  // dx = dout * (1 - out^2)
  __device__ __forceinline__ T operator()(const T dout, const T out) const {
    return dout * (one - out * out);
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() {
    return ActBwdOpFwdDeps::kDepOut;
  }
};

template <typename T>
struct CudaTanhGradFunctor<ComplexType<T>>
    : public BaseActivationFunctor<ComplexType<T>> {
  ComplexType<T> one = static_cast<ComplexType<T>>(1.0f);

  // dx = dout * (1 - out^2)
  __device__ __forceinline__ ComplexType<T> operator()(
      const ComplexType<T> dout, const ComplexType<T> out) const {
    return dout * conj(one - out * out);
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() {
    return ActBwdOpFwdDeps::kDepOut;
  }
};

template <typename T>
struct CudaHardTanhFunctor : public BaseActivationFunctor<T> {
  float t_min;
  float t_max;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"t_min", &t_min}, {"t_max", &t_max}};
  }

  // brelu(x) = min(max(x, t_min), t_max)
  __device__ __forceinline__ T operator()(const T x) const {
    T t_min_cast = static_cast<T>(t_min);
    T t_max_cast = static_cast<T>(t_max);
    T temp_max = x > t_min_cast ? x : t_min_cast;
    T temp_min = temp_max < t_max_cast ? temp_max : t_max_cast;
    return temp_min;
  }
};

template <typename T>
struct CudaMishFunctor : public BaseActivationFunctor<T> {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
  MPType one = static_cast<MPType>(1.0f);
  float threshold;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"threshold", &threshold}};
  }

  // mish(x) = x * tanh(softplus(x))
  // softplus(x) = x, if x > threshold
  //             = ln(1 + exp(x)), otherwise
  // Inputs: args[0], the input x
  __device__ __forceinline__ T operator()(const T arg_x) const {
    MPType x = static_cast<MPType>(arg_x);
    MPType sp = (x > static_cast<MPType>(threshold)) ? x : log(one + exp(x));
    return static_cast<T>(x * tanh(sp));
  }
};

template <typename T>
struct CudaMishGradFunctor : public BaseActivationFunctor<T> {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
  MPType one = static_cast<MPType>(1.0f);
  float threshold;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"threshold", &threshold}};
  }

  // dx = dout * (tanh(sp) + x * (1 - tanh(sp) ** 2) * (1 - exp(-sp)))
  // sp = softplus(x)
  // Inputs: args[0], the input dout
  //         args[1], the input x
  __device__ __forceinline__ T operator()(const T arg_dout,
                                          const T arg_x) const {
    MPType dout = static_cast<MPType>(arg_dout);
    MPType x = static_cast<MPType>(arg_x);
    MPType sp = (x > static_cast<MPType>(threshold)) ? x : log(one + exp(x));
    MPType gsp =
        (x > static_cast<MPType>(threshold)) ? one : one / (one + exp(-x));
    MPType tsp = tanh(sp);
    return static_cast<T>(dout * (tsp + x * (one - tsp * tsp) * gsp));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct CudaHardTanhGradFunctor : public BaseActivationFunctor<T> {
  T zero = static_cast<T>(0.0f);
  float t_min;
  float t_max;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"t_min", &t_min}, {"t_max", &t_max}};
  }

  // dx = (x > t_min && x < t_max) ? dout : 0
  __device__ __forceinline__ T operator()(const T dout, const T x) const {
    T t_min_cast = static_cast<T>(t_min);
    T t_max_cast = static_cast<T>(t_max);
    return (x > t_min_cast && x < t_max_cast) ? dout : zero;
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct CudaThresholdedReluFunctor : public BaseActivationFunctor<T> {
  float threshold;
  float value;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"threshold", &threshold}, {"value", &value}};
  }

  // thresholded_relu(x, threshold, value) = x > threshold ? x : value
  __device__ __forceinline__ T operator()(const T x) const {
    return x > static_cast<T>(threshold) ? x : static_cast<T>(value);
  }
};

template <typename T>
struct CudaThresholdedReluGradFunctor : public BaseActivationFunctor<T> {
  T zero = static_cast<T>(0.0f);
  float threshold;
  float value;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"threshold", &threshold}, {"value", &value}};
  }

  // dx = x > threshold ? dout : 0
  __device__ __forceinline__ T operator()(const T dout, const T x) const {
    return x > static_cast<T>(threshold) ? dout : zero;
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct CudaRelu6Functor : public BaseActivationFunctor<T> {
  T zero = static_cast<T>(0.0f);
  float threshold;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"threshold", &threshold}};
  }

  // relu6(x) = min(max(0, x), 6)
  __device__ __forceinline__ T operator()(const T x) const {
    T t = static_cast<T>(threshold);
    return x <= zero ? zero : (x < t ? x : t);
  }
};

template <typename T>
struct CudaRelu6GradFunctor : public BaseActivationFunctor<T> {
  T zero = static_cast<T>(0.0f);

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() { return {{}}; }

  // dx = (out > 0 && out < t) ? dout : 0
  __device__ __forceinline__ T operator()(const T dout, const T out) const {
    float threshold = 6;
    T t = static_cast<T>(threshold);
    return (out > zero && out < t) ? dout : zero;
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() {
    return ActBwdOpFwdDeps::kDepOut;
  }
};
template <typename T>
struct CudaLeakyReluFunctor : public BaseActivationFunctor<T> {
  T zero = static_cast<T>(0.0f);
  float alpha;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"alpha", &alpha}};
  }

  // leakyrelu(x) = x > 0 ? x : alpha * x
  __device__ __forceinline__ T operator()(const T x) const {
    return x > zero ? x : static_cast<T>(alpha) * x;
  }
};

template <typename T>
struct CudaLeakyReluGradFunctor : public BaseActivationFunctor<T> {
  T zero = static_cast<T>(0.0f);
  float alpha;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"alpha", &alpha}};
  }

  // dx = dout * (x > 0 ? 1 : alpha)
  __device__ __forceinline__ T operator()(const T dout, const T x) const {
    return x > zero ? dout : static_cast<T>(alpha) * dout;
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct CudaSoftShrinkFunctor : public BaseActivationFunctor<T> {
  float lambda;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"lambda", &lambda}};
  }

  // softshrink(x) = x - lambda, if x > lambda;
  //                 x + lambda, if x < -lambda;
  //                 0, otherwise.
  __device__ __forceinline__ T operator()(const T x) const {
    T l = static_cast<T>(lambda);
    T temp1 = static_cast<T>(x > l);
    T temp2 = static_cast<T>(x < -l);
    return temp1 * (x - l) + temp2 * (x + l);
  }
};

template <typename T>
struct CudaSoftShrinkGradFunctor : public BaseActivationFunctor<T> {
  T zero = static_cast<T>(0.0f);
  float lambda;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"lambda", &lambda}};
  }

  // dx = dout, if x > lambda or x < -lambda else 0
  __device__ __forceinline__ T operator()(const T dout, const T x) const {
    T l = static_cast<T>(lambda);
    return (x >= -l && x <= l) ? zero : dout;
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct CudaTanhShrinkFunctor : public BaseActivationFunctor<T> {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;

  // tanhshrink(x) = x - tanh(x)
  __device__ __forceinline__ T operator()(const T arg_x) const {
    MPType x = static_cast<MPType>(arg_x);
    return static_cast<T>(x - tanh(x));
  }
};

template <typename T>
struct CudaTanhShrinkGradFunctor : public BaseActivationFunctor<T> {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;

  // dx = dout * tanh(x)^2
  __device__ __forceinline__ T operator()(const T arg_dout,
                                          const T arg_x) const {
    MPType dout = static_cast<MPType>(arg_dout);
    MPType x = static_cast<MPType>(arg_x);
    return static_cast<T>(dout * tanh(x) * tanh(x));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct CudaHardShrinkFunctor : public BaseActivationFunctor<T> {
  T zero = static_cast<T>(0.0f);
  float threshold;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"threshold", &threshold}};
  }

  // hadrshrink(x) = (x > -threshold && x < threshold) ? 0 : x
  __device__ __forceinline__ T operator()(const T x) const {
    T t = static_cast<T>(threshold);
    return (x > -t && x < t) ? zero : x;
  }
};

template <typename T>
struct CudaHardShrinkGradFunctor : public BaseActivationFunctor<T> {
  T zero = static_cast<T>(0.0f);
  float threshold;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"threshold", &threshold}};
  }

  // dx = (x > -threshold && x < threshold) ? 0 : dout
  __device__ __forceinline__ T operator()(const T dout, const T x) const {
    T t = static_cast<T>(threshold);
    return (x > -t && x < t) ? zero : dout;
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct CudaELUFunctor : public BaseActivationFunctor<T> {
  using CT = typename phi::dtype::MPTypeTrait<T>::Type;
  CT zero = static_cast<CT>(0.0f);
  CT one = static_cast<CT>(1.0f);
  float alpha;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"alpha", &alpha}};
  }

  // elu(x) = x, if x > 0
  // elu(x) = alpha * (e^x - 1), if x <= 0
  __device__ __forceinline__ T operator()(const T arg_x) const {
    CT x = static_cast<CT>(arg_x);
    CT temp = static_cast<CT>(alpha) * (exp(x) - one);
    CT res = x > zero ? x : temp;
    return static_cast<T>(res);
  }
};

template <typename T>
struct CudaELUGradFunctor : public BaseActivationFunctor<T> {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
  MPType zero = static_cast<MPType>(0.0f);
  float alpha;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"alpha", &alpha}};
  }

  // case 1: alpha >= 0
  // dx = dout, if out > 0
  // dx = dout * (out + alpha), if out <= 0
  __device__ __forceinline__ T operator()(T arg_dout, T arg_out) const {
    MPType dout = static_cast<MPType>(arg_dout);
    MPType out = static_cast<MPType>(arg_out);
    MPType a = static_cast<MPType>(alpha);
    MPType out_pos = static_cast<MPType>(out > zero);
    MPType out_neg = static_cast<MPType>(out <= zero);
    return static_cast<T>(dout * (out_pos + out_neg * (out + a)));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() {
    return ActBwdOpFwdDeps::kDepOut;
  }
};

template <typename T>
struct CudaELUGradNegativeAlphaFunctor : public BaseActivationFunctor<T> {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
  MPType zero = static_cast<MPType>(0.0f);
  float alpha;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"alpha", &alpha}};
  }

  // case 2: alpha < 0
  // dx = dout, if x > 0
  // dx = dout * (out + alpha), if x <=0
  __device__ __forceinline__ T operator()(const T arg_dout,
                                          const T arg_out,
                                          const T arg_x) const {
    MPType dout = static_cast<MPType>(arg_dout);
    MPType out = static_cast<MPType>(arg_out);
    MPType x = static_cast<MPType>(arg_x);
    MPType a = static_cast<MPType>(alpha);
    MPType x_pos = static_cast<MPType>(x > zero);
    MPType x_neg = static_cast<MPType>(x <= zero);
    return static_cast<T>(dout * (x_pos + x_neg * (out + a)));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct CudaSiluFunctor : public BaseActivationFunctor<T> {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
  MPType one = static_cast<MPType>(1.0f);

  // silu(x) = x / (1 + exp(-x))
  __device__ __forceinline__ T operator()(const T arg_x) const {
    MPType x = static_cast<MPType>(arg_x);
    return static_cast<T>(x / (one + exp(-x)));
  }
};

template <typename T>
struct CudaSiluGradFunctor : public BaseActivationFunctor<T> {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
  MPType one = static_cast<MPType>(1.0f);

  // dx = dout * (1 + exp(-x) + x * exp(-x) / (1 + exp(-x))^2)
  __device__ __forceinline__ T operator()(const T arg_dout,
                                          const T arg_x) const {
    MPType dout = static_cast<MPType>(arg_dout);
    MPType x = static_cast<MPType>(arg_x);
    MPType temp = one / (one + exp(-x));
    return static_cast<T>(dout * (temp * (one + x * (one - temp))));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct CudaSiluGradFunctor<ComplexType<T>>
    : public BaseActivationFunctor<ComplexType<T>> {
  ComplexType<T> one = static_cast<ComplexType<T>>(1.0f);

  // dx = dout * (1 + exp(-x) + x * exp(-x) / (1 + exp(-x))^2)
  __device__ __forceinline__ ComplexType<T> operator()(
      const ComplexType<T> arg_dout, const ComplexType<T> arg_x) const {
    ComplexType<T> dout = static_cast<ComplexType<T>>(arg_dout);
    ComplexType<T> x = static_cast<ComplexType<T>>(arg_x);
    ComplexType<T> temp = one / (one + exp(-x));
    return dout * conj(temp * (one + x * (one - temp)));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct CudaSoftsignFunctor : public BaseActivationFunctor<T> {
  T one = static_cast<T>(1.0f);

  // softsign(x) = x / (1 + abs(x))
  __device__ __forceinline__ T operator()(const T x) const {
    // Using abs directly will cause namespace conflict
    return x / (one + (x > -x ? x : -x));
  }
};

template <typename T>
struct CudaSoftsignFunctor<ComplexType<T>>
    : public BaseActivationFunctor<ComplexType<T>> {
  using Complex = ComplexType<T>;
  Complex one = static_cast<Complex>(1.0f);

  __device__ __forceinline__ Complex operator()(const Complex x) const {
    return x / (one + static_cast<Complex>(abs(x)));
  }
};

template <typename T>
struct CudaSoftsignGradFunctor : public BaseActivationFunctor<T> {
  T one = static_cast<T>(1.0f);

  // dx = dout / (1 + abs(x))^2
  __device__ __forceinline__ T operator()(const T dout, const T x) const {
    // Using abs directly will cause namespace conflict
    T temp = one + (x > -x ? x : -x);
    return dout / (temp * temp);
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct CudaSoftsignGradFunctor<ComplexType<T>>
    : public BaseActivationFunctor<ComplexType<T>> {
  using Complex = ComplexType<T>;
  Complex one = static_cast<Complex>(1.0f);

  __device__ __forceinline__ Complex operator()(const Complex dout,
                                                const Complex x) const {
    Complex abs_x = static_cast<Complex>(abs(x));
    Complex abs_x_plus = one + abs_x;
    Complex temp = static_cast<Complex>((-x / (abs_x_plus * abs_x_plus)).real);
    return dout * (one / abs_x_plus + temp * x / abs_x);
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct CudaSigmoidFunctor : public BaseActivationFunctor<T> {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
  MPType one = static_cast<MPType>(1.0f);

  // sigmoid(x) = 1 / (1 + exp(-x))
  __device__ __forceinline__ T operator()(const T arg_x) const {
    MPType x = static_cast<MPType>(arg_x);
    return static_cast<T>(one / (one + exp(-x)));
  }
};

template <typename T>
struct CudaSigmoidGradFunctor : public BaseActivationFunctor<T> {
  T one = static_cast<T>(1.0f);

  // dx = dout * out * (1 - out)
  __device__ __forceinline__ T operator()(const T dout, const T out) const {
    return dout * out * (one - out);
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() {
    return ActBwdOpFwdDeps::kDepOut;
  }
};

template <typename T>
struct CudaSigmoidGradFunctor<ComplexType<T>>
    : public BaseActivationFunctor<ComplexType<T>> {
  using Complex = ComplexType<T>;
  Complex one = Complex(1.0f);
  // dx = dout * out * (1 - out)
  __device__ __forceinline__ Complex operator()(const Complex dout,
                                                const Complex out) const {
    Complex y = out * (one - out);
    return dout * conj(y);
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() {
    return ActBwdOpFwdDeps::kDepOut;
  }
};

template <typename T>
struct CudaLogSigmoidFunctor : public BaseActivationFunctor<T> {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
  MPType zero = static_cast<MPType>(0.0f);

  // logsigmoid(x) = log(1 / (1 + exp(-x)))
  // For numerical stability,
  // logsigmoid(x) =
  //          - (max(-x, 0) + log(exp(-max(-x, 0)) + exp(-x - max(-x, 0))))
  __device__ __forceinline__ T operator()(const T arg_x) const {
    MPType x = static_cast<MPType>(arg_x);
    MPType temp = x > zero ? zero : -x;
    return static_cast<T>(-temp - log(exp(-temp) + exp(-x - temp)));
  }
};

template <typename T>
struct CudaLogSigmoidGradFunctor : public BaseActivationFunctor<T> {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
  MPType zero = static_cast<MPType>(0.0f);

  // dx = dout * exp(-x) / (1 + exp(-x))
  // For numerical stability:
  // dx = dout * exp(-x - max(-x, 0)) / (exp(-max(-x, 0)) + exp(-x - max(-x,
  // 0)))
  __device__ __forceinline__ T operator()(const T arg_dout,
                                          const T arg_x) const {
    MPType dout = static_cast<MPType>(arg_dout);
    MPType x = static_cast<MPType>(arg_x);
    MPType temp1 = x > zero ? zero : -x;
    MPType temp2 = exp(-x - temp1);
    return static_cast<T>(dout * (temp2 / (exp(-temp1) + temp2)));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct CudaLogSigmoidGradFunctor<ComplexType<T>>
    : public BaseActivationFunctor<ComplexType<T>> {
  ComplexType<T> zero = static_cast<ComplexType<T>>(0.0f);

  // dx = dout * exp(-x) / (1 + exp(-x))
  // For numerical stability:
  // dx = dout * exp(-x - max(-x, 0)) / (exp(-max(-x, 0)) + exp(-x - max(-x,
  // 0)))
  __device__ __forceinline__ ComplexType<T> operator()(
      const ComplexType<T> arg_dout, const ComplexType<T> arg_x) const {
    ComplexType<T> dout = static_cast<ComplexType<T>>(arg_dout);
    ComplexType<T> x = static_cast<ComplexType<T>>(arg_x);
    ComplexType<T> temp1 = x > zero ? zero : -x;
    ComplexType<T> temp2 = exp(-x - temp1);
    return static_cast<ComplexType<T>>(dout *
                                       conj(temp2 / (exp(-temp1) + temp2)));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct CudaHardSigmoidFunctor : public BaseActivationFunctor<T> {
  T zero = static_cast<T>(0.0f);
  T one = static_cast<T>(1.0f);
  float slope;
  float offset;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"slope", &slope}, {"offset", &offset}};
  }

  // hard_sigmoid(x) = 0, when x <= -3
  //                   1, when x >= 3
  //                   x * slope + offset, otherwise
  __device__ __forceinline__ T operator()(const T x) const {
    T temp = x * static_cast<T>(slope) + static_cast<T>(offset);
    T temp_max = temp > zero ? temp : zero;
    T temp_min = temp_max < one ? temp_max : one;
    return temp_min;
  }
};

template <typename T>
struct CudaHardSigmoidGradFunctor : public BaseActivationFunctor<T> {
  T zero = static_cast<T>(0.0f);
  T one = static_cast<T>(1.0f);
  float slope;
  float offset;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"slope", &slope}, {"offset", &offset}};
  }

  // dx = (out > 0 && out < 1) ? dout * slope : 0
  __device__ __forceinline__ T operator()(const T dout, const T out) const {
    return (out > zero && out < one) ? dout * static_cast<T>(slope) : zero;
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() {
    return ActBwdOpFwdDeps::kDepOut;
  }
};

template <typename T>
__device__ __forceinline__
    std::conditional_t<std::is_integral<T>::value, float, T>
    log_local(T x) {
  static_assert(!std::is_same<T, double>::value,
                "this template must be used with float or less precise type");

#if defined(__CUDA_ARCH__) || defined(__HIP_ARCH__)
  // use __logf fast approximation for peak bandwidth
  return __logf(x);
#else
  return ::log(x);
#endif
}

template <>
__device__ __forceinline__ double log_local<double>(double x) {
  return ::log(x);
}

template <typename T>
struct CudaLogFunctor : public BaseActivationFunctor<T> {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
  using U = typename std::conditional_t<std::is_integral<T>::value, float, T>;

  // log(x) = log(x)
  __device__ __forceinline__ U operator()(const T arg_x) const {
    MPType x = static_cast<MPType>(arg_x);
    return static_cast<U>(log_local(x));
  }
};

template <typename T>
struct CudaLogFunctor<ComplexType<T>>
    : public BaseActivationFunctor<ComplexType<T>> {
  // log(x) = log(x)
  __device__ __forceinline__ ComplexType<T> operator()(
      const ComplexType<T> arg_x) const {
    return static_cast<ComplexType<T>>(log(arg_x));
  }
};

template <typename T>
struct CudaLogGradFunctor : public BaseActivationFunctor<T> {
  // dx = dout / x
  __device__ __forceinline__ T operator()(const T dout, const T x) const {
    return dout / x;
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct CudaLogGradFunctor<ComplexType<T>>
    : public BaseActivationFunctor<ComplexType<T>> {
  // dx = dout / conj(x)
  __device__ __forceinline__ ComplexType<T> operator()(
      const ComplexType<T> dout, const ComplexType<T> x) const {
    return dout / conj(x);
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct CudaLog1pFunctor : public BaseActivationFunctor<T> {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
  MPType one = static_cast<MPType>(1.0f);
  using U = typename std::conditional_t<std::is_integral<T>::value, float, T>;

  // log1p(x) = log(1 + x)
  __device__ __forceinline__ U operator()(const T arg_x) const {
    MPType x = static_cast<MPType>(arg_x);
    return static_cast<U>(log_local(one + x));
  }
};

template <typename T>
struct CudaLog1pFunctor<ComplexType<T>>
    : public BaseActivationFunctor<ComplexType<T>> {
  // log1p(x) = log(1 + x)
  __device__ __forceinline__ ComplexType<T> operator()(
      const ComplexType<T> arg_x) const {
    return static_cast<ComplexType<T>>(
        log(static_cast<ComplexType<T>>(1) + arg_x));
  }
};

template <typename T>
struct CudaLog1pGradFunctor : public BaseActivationFunctor<T> {
  T one = static_cast<T>(1.0f);

  // dx = dout / (1 + x)
  __device__ __forceinline__ T operator()(const T dout, const T x) const {
    return dout / (one + x);
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct CudaLog1pGradFunctor<ComplexType<T>>
    : public BaseActivationFunctor<ComplexType<T>> {
  ComplexType<T> one = static_cast<ComplexType<T>>(1.0f);

  // dx = dout / conj(1 + x)
  __device__ __forceinline__ ComplexType<T> operator()(
      const ComplexType<T> dout, const ComplexType<T> x) const {
    return dout / conj(one + x);
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
__device__ __forceinline__
    std::conditional_t<std::is_integral<T>::value, float, T>
    log2_local(T x) {
  static_assert(!std::is_same<T, double>::value,
                "this template must be used with float or less precise type");

#if defined(__CUDA_ARCH__) || defined(__HIP_ARCH__)
  // use __logf fast approximation for peak bandwidth
  return __log2f(x);
#else
  return ::log2(x);
#endif
}

template <>
__device__ __forceinline__ double log2_local<double>(double x) {
  return ::log2(x);
}

template <typename T>
struct CudaLog2Functor : public BaseActivationFunctor<T> {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
  using U = typename std::conditional_t<std::is_integral<T>::value, float, T>;

  // log2(x) = log2(x)
  __device__ __forceinline__ U operator()(const T arg_x) const {
    MPType x = static_cast<MPType>(arg_x);
    return static_cast<U>(log2_local(x));
  }
};

template <typename T>
struct CudaLog2Functor<ComplexType<T>>
    : public BaseActivationFunctor<ComplexType<T>> {
  // log2(x) = log(x)/log(2)
  __device__ __forceinline__ ComplexType<T> operator()(
      const ComplexType<T> arg_x) const {
    return static_cast<ComplexType<T>>(log(arg_x) /
                                       static_cast<ComplexType<T>>(log(2.0f)));
  }
};

template <typename T>
struct CudaLog2GradFunctor : public BaseActivationFunctor<T> {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
  T log_two = static_cast<T>(log(static_cast<MPType>(2.0f)));

  // dx = dout / (x * log(2))
  __device__ __forceinline__ T operator()(const T dout, const T x) const {
    return dout / (x * log_two);
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct CudaLog2GradFunctor<ComplexType<T>>
    : public BaseActivationFunctor<ComplexType<T>> {
  // dx = dout / conj(x * log(2))
  __device__ __forceinline__ ComplexType<T> operator()(
      const ComplexType<T> dout, const ComplexType<T> x) const {
    return dout / conj(x * static_cast<ComplexType<T>>(log(2.0f)));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
__device__ __forceinline__
    std::conditional_t<std::is_integral<T>::value, float, T>
    log10_local(T x) {
  static_assert(!std::is_same<T, double>::value,
                "this template must be used with float or less precise type");

#if defined(__CUDA_ARCH__) || defined(__HIP_ARCH__)
  // use __logf fast approximation for peak bandwidth
  return __log10f(x);
#else
  return ::log10(x);
#endif
}

template <>
__device__ __forceinline__ double log10_local(double x) {
  return ::log10(x);
}

template <typename T>
struct CudaLog10Functor : public BaseActivationFunctor<T> {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
  using U = typename std::conditional_t<std::is_integral<T>::value, float, T>;

  // log10(x) = log10(x)
  __device__ __forceinline__ U operator()(const T arg_x) const {
    MPType x = static_cast<MPType>(arg_x);
    return static_cast<U>(log10_local(x));
  }
};

template <typename T>
struct CudaLog10Functor<ComplexType<T>>
    : public BaseActivationFunctor<ComplexType<T>> {
  // log10(x) = log(x)/log(10)
  __device__ __forceinline__ ComplexType<T> operator()(
      const ComplexType<T> arg_x) const {
    return static_cast<ComplexType<T>>(log(arg_x) /
                                       static_cast<ComplexType<T>>(log(10.0f)));
  }
};

template <typename T>
struct CudaLog10GradFunctor : public BaseActivationFunctor<T> {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
  T log_ten = static_cast<T>(log(static_cast<MPType>(10.0f)));

  // dx = dout / (x * log(10))
  __device__ __forceinline__ T operator()(const T dout, const T x) const {
    return dout / (x * log_ten);
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct CudaLog10GradFunctor<ComplexType<T>>
    : public BaseActivationFunctor<ComplexType<T>> {
  // dx = dout / conj(x * log(10))
  __device__ __forceinline__ ComplexType<T> operator()(
      const ComplexType<T> dout, const ComplexType<T> x) const {
    return dout / conj(x * static_cast<ComplexType<T>>(log(10.0f)));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct CudaSwishFunctor : public BaseActivationFunctor<T> {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
  MPType one = static_cast<MPType>(1.0f);
  float beta = 1.0;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"beta", &beta}};
  }

  // swish(x) = x / (1 + exp(-beta * x))
  __device__ __forceinline__ T operator()(const T arg_x) const {
    MPType x = static_cast<MPType>(arg_x);
    MPType b = static_cast<MPType>(beta);
    return static_cast<T>(x / (one + exp(-b * x)));
  }
};

template <typename T>
struct CudaSwishGradFunctor : public BaseActivationFunctor<T> {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
  MPType one = static_cast<MPType>(1.0f);

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() { return {{}}; }

  // dx = dout * (1 + exp(-b * x) + b * x * exp(-b * x) / (1 + exp(-b * x))^2)
  __device__ __forceinline__ T operator()(const T arg_dout,
                                          const T arg_x) const {
    float beta = 1.0;
    MPType dout = static_cast<MPType>(arg_dout);
    MPType x = static_cast<MPType>(arg_x);
    MPType b = static_cast<MPType>(beta);
    MPType temp1 = one / (one + exp(-b * x));
    MPType out = x * temp1;
    MPType temp2 = b * out;
    MPType temp3 = temp1 * (one - temp2);
    return static_cast<T>(dout * (temp2 + temp3));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct CudaHardSwishFunctor : public BaseActivationFunctor<T> {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
  const MPType zero = static_cast<MPType>(0.0f);
  float threshold;
  float scale;
  float offset;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"threshold", &threshold}, {"scale", &scale}, {"offset", &offset}};
  }

  // hard_swish(x) = 0, when x <= -offset
  //                 x , when x >= threshold - offset
  //                 x * (x + offset) / scale, otherwise
  // threshold = scale = 6, offset = 3 by default
  __device__ __forceinline__ T operator()(const T x) const {
    const MPType x_t = static_cast<MPType>(x);
    const MPType x_offset_t = x_t + static_cast<MPType>(offset);
    const MPType temp_max = (x_offset_t >= zero) ? x_offset_t : zero;
    const MPType threshold_t = static_cast<MPType>(threshold);
    const MPType temp_min = (temp_max < threshold_t) ? temp_max : threshold_t;
    return static_cast<T>(temp_min * x_t / static_cast<MPType>(scale));
  }
};

template <typename T>
struct CudaHardSwishGradFunctor : public BaseActivationFunctor<T> {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
  const MPType zero = static_cast<MPType>(0.0f);
  const MPType one = static_cast<MPType>(1.0f);
  const MPType two = static_cast<MPType>(2.0f);
  float threshold;
  float scale;
  float offset;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"threshold", &threshold}, {"scale", &scale}, {"offset", &offset}};
  }

  // dx = 0, when x <= -offset
  //      dout , when x >= threshold - offset
  //      dout * (2 * x / scale + offset / scale), otherwise
  // threshold = scale = 6, offset = 3 by default
  __device__ __forceinline__ T operator()(const T dout, const T x) const {
    const MPType dout_t = static_cast<MPType>(dout);
    const MPType x_t = static_cast<MPType>(x);
    const MPType offset_t = static_cast<MPType>(offset);
    const MPType scale_t = static_cast<MPType>(scale);
    const MPType temp1 = static_cast<MPType>(x_t + offset_t > zero);
    const MPType temp2 =
        static_cast<MPType>(x_t + offset_t < static_cast<MPType>(threshold));

    return static_cast<T>(
        dout_t *
        (temp1 * temp2 * (two * x_t + offset_t) / scale_t + one - temp2));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct CudaHardSwishGradFunctor<ComplexType<T>>
    : public BaseActivationFunctor<ComplexType<T>> {
  const ComplexType<T> zero = static_cast<ComplexType<T>>(0.0f);
  const ComplexType<T> one = static_cast<ComplexType<T>>(1.0f);
  const ComplexType<T> two = static_cast<ComplexType<T>>(2.0f);
  float threshold;
  float scale;
  float offset;

  typename BaseActivationFunctor<ComplexType<T>>::AttrPair GetAttrs() {
    return {{"threshold", &threshold}, {"scale", &scale}, {"offset", &offset}};
  }

  // dx = 0, when x <= -offset
  //      dout , when x >= threshold - offset
  //      dout * (2 * x / scale + offset / scale), otherwise
  // threshold = scale = 6, offset = 3 by default
  __device__ __forceinline__ ComplexType<T> operator()(
      const ComplexType<T> dout, const ComplexType<T> x) const {
    const ComplexType<T> dout_t = static_cast<ComplexType<T>>(dout);
    const ComplexType<T> x_t = static_cast<ComplexType<T>>(x);
    const ComplexType<T> offset_t = static_cast<ComplexType<T>>(offset);
    const ComplexType<T> scale_t = static_cast<ComplexType<T>>(scale);
    const ComplexType<T> temp1 =
        static_cast<ComplexType<T>>(x_t + offset_t > zero);
    const ComplexType<T> temp2 = static_cast<ComplexType<T>>(
        x_t + offset_t < static_cast<ComplexType<T>>(threshold));

    return static_cast<ComplexType<T>>(
        dout_t *
        conj(temp1 * temp2 * (two * x_t + offset_t) / scale_t + one - temp2));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct CudaCeilFunctor : public BaseActivationFunctor<T> {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;

  // ceil(x) = ceil(x)
  __device__ __forceinline__ T operator()(const T arg_x) const {
    MPType x = static_cast<MPType>(arg_x);
    return static_cast<T>(ceil(x));
  }
};

template <typename T>
struct CudaFloorFunctor : public BaseActivationFunctor<T> {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;

  // floor(x) = floor(x)
  __device__ __forceinline__ T operator()(const T arg_x) const {
    MPType x = static_cast<MPType>(arg_x);
    return static_cast<T>(floor(x));
  }
};

template <typename T>
struct CudaRoundFunctor : public BaseActivationFunctor<T> {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
  int decimals;

  std::vector<std::pair<const char*, int*>> GetAttrs() {
    return {{"decimals", &decimals}};
  }
  // round(x) = round(x)
  __device__ __forceinline__ T operator()(const T arg_x) const {
    MPType x = static_cast<MPType>(arg_x);

    if (decimals == 0) {
      return static_cast<T>(round(x));
    } else if (decimals > 0) {
      float ten_pow_decimals = powf(10., decimals);
      return static_cast<T>(round(x * static_cast<MPType>(ten_pow_decimals)) /
                            ten_pow_decimals);
    } else {
      float ten_pow_decimals = powf(10., -decimals);
      return static_cast<T>(round(x / static_cast<MPType>(ten_pow_decimals)) *
                            ten_pow_decimals);
    }
  }
};

// GradFunctor for ceil, floor and round
template <typename T>
struct CudaZeroGradFunctor : public BaseActivationFunctor<T> {
  __device__ __forceinline__ T operator()(const T x) const {
    return static_cast<T>(0.0f);
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() {
    return ActBwdOpFwdDeps::kNoDeps;
  }
};

template <typename T>
struct CudaCELUFunctor : public BaseActivationFunctor<T> {
  using CT = typename phi::dtype::MPTypeTrait<T>::Type;
  CT zero = static_cast<CT>(0.0f);
  CT one = static_cast<CT>(1.0f);
  float alpha;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"alpha", &alpha}};
  }

  // celu(x) = max(0, x) + min(0, alpha * (exp(x/alpha) - 1))
  __device__ __forceinline__ T operator()(const T arg_x) const {
    CT x = static_cast<CT>(arg_x);
    CT temp = static_cast<CT>(alpha) * (exp(x / static_cast<CT>(alpha)) - one);
    CT res = (x > zero ? x : zero) + (temp > zero ? zero : temp);
    return static_cast<T>(res);
  }
};

template <typename T>
struct CudaCELUGradFunctor : public BaseActivationFunctor<T> {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
  MPType zero = static_cast<MPType>(0.0f);
  MPType one = static_cast<MPType>(1.0f);
  float alpha;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"alpha", &alpha}};
  }

  // dx = dout, if alpha > 0 and x > 0
  // dx = dout * (x/alpha).exp(), if alpha > 0 and x <= 0
  // dx = dout , if alpha < 0 and x > 0
  // dx = dout * (x/alpha).exp(), if alpha < 0 and x <=0
  __device__ __forceinline__ T operator()(const T arg_dout,
                                          const T arg_x) const {
    MPType dout = static_cast<MPType>(arg_dout);
    MPType x = static_cast<MPType>(arg_x);
    MPType a = static_cast<MPType>(alpha);
    MPType temp_a_pos = static_cast<MPType>(alpha > 0.0f);
    MPType temp_a_neg = static_cast<MPType>(alpha <= 0.0f);
    MPType temp_x_pos = static_cast<MPType>(x > zero);
    MPType temp_x_neg = static_cast<MPType>(x <= zero);
    return static_cast<T>(
        dout *
        (temp_a_pos * temp_x_pos + temp_a_pos * temp_x_neg * exp(x / a) +
         temp_a_neg * temp_x_pos + exp(x / a) * temp_a_neg * temp_x_neg));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

#endif

template <typename T>
struct SwiGLUFunctor {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;

  HOSTDEVICE T operator()(T x, T y) const {
    MPType mp_x = static_cast<MPType>(x);
    MPType mp_y = static_cast<MPType>(y);
    MPType one = static_cast<MPType>(1);
    return static_cast<T>(mp_y * mp_x / (one + exp(-mp_x)));
  }
};

template <typename T, bool HasDX = true, bool HasDY = true>
struct SwiGLUGradFunctor {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;

  HOSTDEVICE void operator()(T x, T y, T dz, T* dx, T* dy) const {
    MPType one = static_cast<MPType>(1);

    MPType mp_x = static_cast<MPType>(x);
    MPType mp_dz = static_cast<MPType>(dz);

    MPType sigmoid = one / (one + exp(-mp_x));
    MPType tmp = mp_x * sigmoid;
    if (HasDX) {
      MPType mp_y = static_cast<MPType>(y);
      *dx = static_cast<T>(mp_dz * mp_y * sigmoid * (one + mp_x - tmp));
    }
    if (HasDY) {
      *dy = static_cast<T>(mp_dz * tmp);
    }
  }
};

}  // namespace funcs
}  // namespace phi
