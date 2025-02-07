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

#include "paddle/common/hostdevice.h"
#include "paddle/common/macros.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/complex.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/enforce.h"
#if defined(__xpu__)
#include <xpu/runtime.h>

#include "xpu/kernel/math_xpu2.h"  // pow()
#endif
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/common/type_safe_sign_math.h"

namespace phi {
namespace funcs {

// Define the binary functors used in elementwise ops.
// Note: InverseXxxFunctor is needed when calling ElementwiseComputeEx on CPU.

// Add
template <typename T>
struct AddFunctor {
  inline HOSTDEVICE T operator()(const T a, const T b) const { return a + b; }
};
template <typename T>
using InverseAddFunctor = AddFunctor<T>;

template <typename T, typename Ty = T>
struct MultiPrecisionAddFunctor {
  inline HOSTDEVICE T operator()(const T x, const Ty y) const {
    return x + static_cast<T>(y);
  }
};

// Float32Bfloat16Add
template <typename T>
struct Float32Bfloat16AddFunctor {
  inline HOSTDEVICE T operator()(const T x, const phi::bfloat16 y) {
    return x + static_cast<T>(y);
  }
};

// Float32Float16Add
template <typename T>
struct Float32Float16AddFunctor {
  inline HOSTDEVICE T operator()(const T x, const phi::float16 y) {
    return x + static_cast<T>(y);
  }
};

// Subtract
template <typename T>
struct SubtractFunctor {
  inline HOSTDEVICE T operator()(const T a, const T b) const { return a - b; }
};
template <typename T>
struct InverseSubtractFunctor {
  inline HOSTDEVICE T operator()(const T a, const T b) const { return b - a; }
};

// Multiply
template <typename T>
struct MultiplyFunctor {
  inline HOSTDEVICE T operator()(const T a, const T b) const { return a * b; }
};
template <>
struct MultiplyFunctor<bool> {
  inline HOSTDEVICE bool operator()(const bool a, const bool b) const {
    return a && b;
  }
};
template <typename T>
using InverseMultiplyFunctor = MultiplyFunctor<T>;

template <typename T>
struct IsZeroFunctor {
  HOSTDEVICE bool operator()(T x) const { return x == static_cast<T>(0); }
};

// Divide
#define DIV_ERROR_INFO                                             \
  "InvalidArgumentError: Integer division by zero encountered in " \
  "(floor) divide. Please check the input value."

template <typename T, typename Enable = void>
struct DivideFunctor {
  inline HOSTDEVICE T operator()(const T a, const T b) const { return a / b; }
};

template <typename T>
struct DivideFunctor<
    T,
    typename std::enable_if<std::is_integral<T>::value>::type> {
  inline HOSTDEVICE T operator()(const T a, const T b) const {
    // For int32/int64, need to check whether the division is zero.
    PADDLE_ENFORCE(b != 0, DIV_ERROR_INFO);
    return a / b;
  }
};

template <typename T, typename Enable = void>
struct InverseDivideFunctor {
  inline HOSTDEVICE T operator()(const T a, const T b) const { return b / a; }
};

template <typename T>
using ComplexType = phi::dtype::complex<T>;

template <typename InT, typename OutT>
struct DivGradXYFunctor {
  inline HOSTDEVICE phi::Array<OutT, 2> operator()(const InT a,
                                                   const InT b,
                                                   const InT c) {
    // dx = dout / y
    // dy = - dout * out / y
    phi::Array<OutT, 2> outs;
    outs[0] = a / c;
    outs[1] = -a * ((b / c) / c);
    return outs;
  }
};

template <typename InT, typename OutT>
struct DivGradXYFunctor<ComplexType<InT>, ComplexType<OutT>> {
  inline HOSTDEVICE phi::Array<ComplexType<OutT>, 2> operator()(
      const ComplexType<InT> a,
      const ComplexType<InT> b,
      const ComplexType<InT> c) {
    phi::Array<ComplexType<OutT>, 2> outs;
    ComplexType<InT> c_conj(c.real, -c.imag);
    ComplexType<InT> out_div_c_conj(((b / c) / c).real, -((b / c) / c).imag);
    outs[0] = a / c_conj;
    outs[1] = -a * out_div_c_conj;
    return outs;
  }
};

// Float div grad
template <typename T>
struct DivGradXFunctor {
  inline HOSTDEVICE T operator()(const T a, const T b) const { return a / b; }
};

// ComplexType div grad
template <typename T>
struct DivGradXFunctor<ComplexType<T>> {
  inline HOSTDEVICE ComplexType<T> operator()(const ComplexType<T> a,
                                              const ComplexType<T> b) const {
    ComplexType<T> b_conj(b.real, -b.imag);
    return a / b_conj;
  }
};

// Float mul and div
template <typename T>
struct DivGradYFunctor {
  inline HOSTDEVICE T operator()(const T a, const T b, const T c) const {
    return -a * ((b / c) / c);
  }
};

// ComplexType mul and div
template <typename T>
struct DivGradYFunctor<ComplexType<T>> {
  inline HOSTDEVICE ComplexType<T> operator()(const ComplexType<T> a,
                                              const ComplexType<T> b,
                                              const ComplexType<T> c) const {
    ComplexType<T> out_div_c_conj(((b / c) / c).real, -((b / c) / c).imag);
    return -a * out_div_c_conj;
  }
};
// Fmin
template <typename T>
struct FMinFunctor {
  inline HOSTDEVICE T operator()(const T a, const T b) const {
    return std::fmin(a, b);
  }
};

template <>
struct FMinFunctor<dtype::float16> {
  inline HOSTDEVICE dtype::float16 operator()(const dtype::float16 a,
                                              const dtype::float16 b) const {
    float float_a = static_cast<float>(a);
    float float_b = static_cast<float>(b);
    auto result = std::fmin(float_a, float_b);
    return static_cast<dtype::float16>(result);
  }
};

template <>
struct FMinFunctor<dtype::bfloat16> {
  inline HOSTDEVICE dtype::bfloat16 operator()(const dtype::bfloat16 a,
                                               const dtype::bfloat16 b) const {
    float float_a = static_cast<float>(a);
    float float_b = static_cast<float>(b);
    auto result = std::fmin(float_a, float_b);
    return static_cast<dtype::bfloat16>(result);
  }
};

template <>
struct FMinFunctor<int> {
  inline HOSTDEVICE int operator()(const int a, const int b) const {
    float float_a = static_cast<float>(a);
    float float_b = static_cast<float>(b);
    auto result = std::fmin(float_a, float_b);
    return std::lrint(result);
  }
};

template <>
struct FMinFunctor<int64_t> {
  inline HOSTDEVICE int64_t operator()(const int64_t a, const int64_t b) const {
    double double_a = static_cast<double>(a);
    double double_b = static_cast<double>(b);
    auto result = std::fmin(double_a, double_b);
    return std::llrint(result);
  }
};

// Fmax
template <typename T>
struct FMaxFunctor {
  inline HOSTDEVICE T operator()(const T a, const T b) const {
    return std::fmax(a, b);
  }
};

template <>
struct FMaxFunctor<dtype::float16> {
  inline HOSTDEVICE dtype::float16 operator()(const dtype::float16 a,
                                              const dtype::float16 b) const {
    float float_a = static_cast<float>(a);
    float float_b = static_cast<float>(b);
    auto result = std::fmax(float_a, float_b);
    return static_cast<dtype::float16>(result);
  }
};

template <>
struct FMaxFunctor<dtype::bfloat16> {
  inline HOSTDEVICE dtype::bfloat16 operator()(const dtype::bfloat16 a,
                                               const dtype::bfloat16 b) const {
    float float_a = static_cast<float>(a);
    float float_b = static_cast<float>(b);
    auto result = std::fmax(float_a, float_b);
    return static_cast<dtype::bfloat16>(result);
  }
};

template <>
struct FMaxFunctor<int> {
  inline HOSTDEVICE int operator()(const int a, const int b) const {
    float float_a = static_cast<float>(a);
    float float_b = static_cast<float>(b);
    auto result = std::fmax(float_a, float_b);
    return std::lrint(result);
  }
};

template <>
struct FMaxFunctor<int64_t> {
  inline HOSTDEVICE int64_t operator()(const int64_t a, const int64_t b) const {
    double double_a = static_cast<double>(a);
    double double_b = static_cast<double>(b);
    auto result = std::fmax(double_a, double_b);
    return std::llrint(result);
  }
};

template <typename T>
struct FMaxGradDx {
  HOSTDEVICE T operator()(T x, T y, T out UNUSED, T dout) const {
    return dout * static_cast<T>((x >= y) || isnan(y));
  }
};

template <>
struct FMaxGradDx<dtype::float16> {
  HOSTDEVICE dtype::float16 operator()(dtype::float16 x,
                                       dtype::float16 y,
                                       dtype::float16 out UNUSED,
                                       dtype::float16 dout) const {
    return dout * static_cast<dtype::float16>((x >= y) || dtype::isnan(y));
  }
};

template <>
struct FMaxGradDx<int> {
  HOSTDEVICE int operator()(int x, int y, int out UNUSED, int dout) const {
    return dout * static_cast<int>((x >= y));
  }
};

template <>
struct FMaxGradDx<int64_t> {
  HOSTDEVICE int64_t operator()(int64_t x,
                                int64_t y,
                                int64_t out UNUSED,
                                int64_t dout) const {
    return dout * static_cast<int64_t>((x >= y));
  }
};

template <typename T>
struct FMaxGradDy {
  HOSTDEVICE T operator()(T x, T y, T out UNUSED, T dout) const {
    return dout * static_cast<T>(!((x >= y) || isnan(y)));
  }
};

template <>
struct FMaxGradDy<dtype::float16> {
  HOSTDEVICE dtype::float16 operator()(dtype::float16 x,
                                       dtype::float16 y,
                                       dtype::float16 out UNUSED,
                                       dtype::float16 dout) const {
    return dout * static_cast<dtype::float16>(!((x >= y) || dtype::isnan(y)));
  }
};

template <>
struct FMaxGradDy<int64_t> {
  HOSTDEVICE int64_t operator()(int64_t x,
                                int64_t y,
                                int64_t out UNUSED,
                                int64_t dout) const {
    return dout * static_cast<int64_t>(!((x >= y)));
  }
};

template <>
struct FMaxGradDy<int> {
  HOSTDEVICE int operator()(int x, int y, int out UNUSED, int dout) const {
    return dout * static_cast<int>(!((x >= y)));
  }
};

template <typename T>
struct FMinGradDx {
  HOSTDEVICE T operator()(T x, T y, T out UNUSED, T dout) const {
    return dout * static_cast<T>((x <= y) || isnan(y));
  }
};

template <>
struct FMinGradDx<dtype::float16> {
  HOSTDEVICE dtype::float16 operator()(dtype::float16 x,
                                       dtype::float16 y,
                                       dtype::float16 out UNUSED,
                                       dtype::float16 dout) const {
    return dout * static_cast<dtype::float16>((x <= y) || dtype::isnan(y));
  }
};

template <>
struct FMinGradDx<int> {
  HOSTDEVICE int operator()(int x, int y, int out UNUSED, int dout) const {
    return dout * static_cast<int>((x <= y));
  }
};

template <>
struct FMinGradDx<int64_t> {
  HOSTDEVICE int64_t operator()(int64_t x,
                                int64_t y,
                                int64_t out UNUSED,
                                int64_t dout) const {
    return dout * static_cast<int64_t>((x <= y));
  }
};

template <typename T>
struct FMinGradDy {
  HOSTDEVICE T operator()(T x, T y, T out UNUSED, T dout) const {
    return dout * static_cast<T>(!((x <= y) || isnan(y)));
  }
};

template <>
struct FMinGradDy<dtype::float16> {
  HOSTDEVICE dtype::float16 operator()(dtype::float16 x,
                                       dtype::float16 y,
                                       dtype::float16 out UNUSED,
                                       dtype::float16 dout) const {
    return dout * static_cast<dtype::float16>(!((x <= y) || dtype::isnan(y)));
  }
};

template <>
struct FMinGradDy<int> {
  HOSTDEVICE int operator()(int x, int y, int out UNUSED, int dout) const {
    return dout * static_cast<int>(!((x <= y)));
  }
};

template <>
struct FMinGradDy<int64_t> {
  HOSTDEVICE int64_t operator()(int64_t x,
                                int64_t y,
                                int64_t out UNUSED,
                                int64_t dout) const {
    return dout * static_cast<int64_t>(!((x <= y)));
  }
};

template <typename T>
struct MultiplyGradFunctor {
  inline HOSTDEVICE T operator()(const T a, const T b) const { return a * b; }
};
template <typename T>
struct MultiplyGradFunctor<ComplexType<T>> {
  inline HOSTDEVICE ComplexType<T> operator()(const ComplexType<T> a,
                                              const ComplexType<T> b) const {
    ComplexType<T> b_conj(b.real, -b.imag);
    return a * b_conj;
  }
};

template <typename InT, typename OutT>
struct MultiplyGradXYFunctor {
  inline HOSTDEVICE phi::Array<OutT, 2> operator()(const InT a,
                                                   const InT b,
                                                   const InT c) {
    phi::Array<OutT, 2> outs;
    // dx = dout * y
    outs[0] = a * b;
    // dy = dout * x
    outs[1] = a * c;
    return outs;
  }
};

template <typename InT, typename OutT>
struct MultiplyGradXYFunctor<ComplexType<InT>, ComplexType<OutT>> {
  inline HOSTDEVICE phi::Array<ComplexType<OutT>, 2> operator()(
      const ComplexType<InT> a,
      const ComplexType<InT> b,
      const ComplexType<InT> c) {
    phi::Array<ComplexType<OutT>, 2> outs;
    // dx = dout * y
    ComplexType<InT> b_conj(b.real, -b.imag);
    outs[0] = a * b_conj;
    // dy = dout * x
    ComplexType<InT> c_conj(c.real, -c.imag);
    outs[1] = a * c_conj;
    return outs;
  }
};

// Maximum
template <typename T>
struct MaximumFunctor {
  inline HOSTDEVICE T operator()(const T a, const T b) const {
    return a > b ? a : b;
  }
};

template <typename T>
struct MaxGradXFunctor {
  inline HOSTDEVICE T operator()(const T x, const T y, const T dout) const {
    return dout * static_cast<T>(x > y);
  }
};

template <typename T>
struct MaxGradYFunctor {
  inline HOSTDEVICE T operator()(const T x, const T y, const T dout) const {
    return dout * static_cast<T>(x <= y);
  }
};

template <typename InT, typename OutT>
struct MaxGradXYFunctor {
  inline HOSTDEVICE phi::Array<OutT, 2> operator()(const InT x,
                                                   const InT y,
                                                   const InT dout) {
    phi::Array<OutT, 2> outs;
    // dx = dout * (x > y)
    outs[0] = static_cast<OutT>(dout * static_cast<InT>(x > y));
    // dy = dout * (x <= y)
    outs[1] = static_cast<OutT>(dout * static_cast<InT>(x <= y));
    return outs;
  }
};

// Minimum
template <typename T>
struct MinimumFunctor {
  inline HOSTDEVICE T operator()(const T a, const T b) const {
    return a < b ? a : b;
  }
};
template <typename T>
struct MinGradXFunctor {
  inline HOSTDEVICE T operator()(const T x, const T y, const T dout) const {
    return dout * static_cast<T>(x < y);
  }
};
template <typename T>
struct MinGradYFunctor {
  inline HOSTDEVICE T operator()(const T x, const T y, const T dout) const {
    return dout * static_cast<T>(x >= y);
  }
};

template <typename InT, typename OutT>
struct MinGradXYFunctor {
  inline HOSTDEVICE phi::Array<OutT, 2> operator()(const InT x,
                                                   const InT y,
                                                   const InT dout) {
    phi::Array<OutT, 2> outs;
    // dx = dout * (x < y)
    outs[0] = static_cast<OutT>(dout * static_cast<InT>(x < y));
    // dy = dout * (x >= y)
    outs[1] = static_cast<OutT>(dout * static_cast<InT>(x >= y));
    return outs;
  }
};

// Modulo
template <typename T, typename Enable = void>
struct RemainderFunctor {
  inline HOSTDEVICE T operator()(const T a, const T b) const {
    PADDLE_ENFORCE(b != 0, DIV_ERROR_INFO);
    T res = a % b;

    // According to #PR26732: in dividen % divsor
    // remainder shall have the same sign as divsor.
    if ((res != 0) && ((b ^ res) < 0)) res += b;
    return res;
  }
};

template <typename T>
struct RemainderFunctor<
    T,
    typename std::enable_if_t<std::is_floating_point<T>::value>> {
  inline HOSTDEVICE T operator()(const T a, const T b) const {
    T res = fmod(a, b);

    // According to #PR26732: in dividen % divsor
    // remainder shall have the same sign as divsor.
    if ((res != 0) && ((res < 0) != (b < 0))) res += b;
    return res;
  }
};

template <>
struct RemainderFunctor<dtype::float16> {
  inline HOSTDEVICE dtype::float16 operator()(const dtype::float16 a,
                                              const dtype::float16 b) const {
    float b_float = static_cast<float>(b);
    float res = fmod(static_cast<float>(a), b_float);
    // According to #PR26732: in dividen % divsor
    // remainder shall have the same sign as divsor.
    if ((res != 0.0f) && ((res < 0.0f) != (b_float < 0.0f))) res += b_float;
    return static_cast<dtype::float16>(res);
  }
};

template <>
struct RemainderFunctor<dtype::bfloat16> {
  inline HOSTDEVICE dtype::bfloat16 operator()(const dtype::bfloat16 a,
                                               const dtype::bfloat16 b) const {
    float b_float = static_cast<float>(b);
    float res = fmod(static_cast<float>(a), b_float);

    // According to #PR26732: in dividen % divsor
    // remainder shall have the same sign as divsor.
    if ((res != 0.0f) && ((res < 0.0f) != (b_float < 0.0f))) res += b_float;
    return static_cast<dtype::bfloat16>(res);
  }
};

// RemainderGradXFunctor
template <typename T>
struct RemainderGradXFunctor {
  inline HOSTDEVICE T operator()(const T x, const T y, const T dout) const {
    // dx = dout
    return dout;
  }
};

// RemainderGradYFunctor
template <typename T, typename Enable = void>
struct RemainderGradYFunctor {
  inline HOSTDEVICE T operator()(const T x, const T y, const T dout) const {
    // dy = -dout * (floor_div(x, y))
    return -dout * static_cast<T>((std::floor(x / y)));
  }
};
template <typename T>
struct RemainderGradYFunctor<
    T,
    typename std::enable_if<std::is_floating_point<T>::value>::type> {
  inline HOSTDEVICE T operator()(const T x, const T y, const T dout) const {
    using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
    // dy = -dout * (floor_div(x, y))
    auto x_ = static_cast<MPType>(x);
    auto y_ = static_cast<MPType>(y);
    return static_cast<T>(-static_cast<MPType>(dout) * (std::floor((x_ / y_))));
  }
};
template <typename T>
struct RemainderGradYFunctor<
    T,
    typename std::enable_if<std::is_integral<T>::value>::type> {
  inline HOSTDEVICE T operator()(const T x, const T y, const T dout) const {
    // dy = -dout * (floor_div(x, y))
    if (phi::is_negative(x) != phi::is_negative(y)) {
      // Subtracts one from the results of truncation division if the
      // divisor and dividend have different sign(bit)s and the remainder of
      // the division is nonzero
      const auto quot = x / y;
      const auto rem = x % y;
      auto ret = rem ? quot - 1 : quot;
      return -dout * static_cast<T>(ret);
    }
    return -dout * static_cast<T>(x / y);
  }
};

// RemainderGradXYFunctor
template <typename InT, typename OutT, typename Enable = void>
struct RemainderGradXYFunctor {
  inline HOSTDEVICE phi::Array<OutT, 2> operator()(const InT x,
                                                   const InT y,
                                                   const InT dout) {
    phi::Array<OutT, 2> outs;
    // dx = dout
    outs[0] = static_cast<OutT>(dout);
    // dy = -dout * (floor_div(x, y))
    outs[1] = static_cast<OutT>(dout * static_cast<InT>(std::floor(x / y)));
    return outs;
  }
};
template <typename InT, typename OutT>
struct RemainderGradXYFunctor<
    InT,
    OutT,
    typename std::enable_if<std::is_floating_point<InT>::value>::type> {
  inline HOSTDEVICE Array<OutT, 2> operator()(const InT x,
                                              const InT y,
                                              const InT dout) {
    Array<OutT, 2> outs;
    // dx = dout
    outs[0] = static_cast<OutT>(dout);
    // dy = -dout * (x / y)
    using MPType = typename phi::dtype::MPTypeTrait<InT>::Type;
    auto x_ = static_cast<MPType>(x);
    auto y_ = static_cast<MPType>(y);
    outs[1] =
        static_cast<OutT>(static_cast<MPType>(-dout) * std::floor(x_ / y_));
    return outs;
  }
};
template <typename InT, typename OutT>
struct RemainderGradXYFunctor<
    InT,
    OutT,
    typename std::enable_if<std::is_integral<InT>::value>::type> {
  inline HOSTDEVICE Array<OutT, 2> operator()(const InT x,
                                              const InT y,
                                              const InT dout) {
    Array<OutT, 2> outs;
    // dx = dout
    outs[0] = static_cast<OutT>(dout);
    // dy = -dout * (x / y)
    if (phi::is_negative(x) != phi::is_negative(y)) {
      // Subtracts one from the results of truncation division if the
      // divisor and dividend have different sign(bit)s and the remainder of
      // the division is nonzero
      const auto quot = x / y;
      const auto rem = x % y;
      auto ret = rem ? quot - 1 : quot;
      outs[1] = -static_cast<OutT>(dout) * static_cast<OutT>(ret);
    }
    outs[1] = -static_cast<OutT>(dout) * static_cast<OutT>(x / y);
    return outs;
  }
};

template <typename T, typename Enable = void>
struct InverseRemainderFunctor {
  inline HOSTDEVICE T operator()(const T a, const T b) const {
    T res = b % a;
    if ((res != 0) && ((res < 0) != (a < 0))) res += a;
    return res;
  }
};

template <typename T>
struct InverseRemainderFunctor<
    T,
    typename std::enable_if_t<std::is_floating_point<T>::value>> {
  inline HOSTDEVICE T operator()(const T a, const T b) const {
    T res = fmod(b, a);
    if ((res != 0) && ((a < 0) != (res < 0))) res += a;
    return res;
  }
};

template <typename T>
struct ElementwiseHeavisideFunctor {
  inline HOSTDEVICE T operator()(const T a, const T b) const {
    return a == static_cast<T>(0) ? b : static_cast<T>(a > static_cast<T>(0));
  }
};

template <typename T, typename Enable = void>
struct FloorDivideFunctor {
  inline HOSTDEVICE T operator()(const T a, const T b) const {
#ifndef PADDLE_WITH_XPU_KP
    PADDLE_ENFORCE(b != 0, DIV_ERROR_INFO);
#endif

    if (phi::is_negative(a) != phi::is_negative(b)) {
      // Subtracts one from the results of truncation division if the
      // divisor and dividend have different sign(bit)s and the remainder of
      // the division is nonzero
      const auto quot = a / b;
      const auto rem = a % b;
      auto ret = rem ? quot - 1 : quot;
      return static_cast<T>(ret);
    }

    return static_cast<T>(a / b);
  }
};

template <typename T>
struct FloorDivideFunctor<
    T,
    typename std::enable_if_t<std::is_floating_point<T>::value>> {
  inline HOSTDEVICE T operator()(const T a, const T b) const {
    if (UNLIKELY(b == 0)) {
      // Divide by zero: return standard IEEE result
      return static_cast<T>(a / b);
    }

    auto mod = std::fmod(a, b);
    auto div = (a - mod) / b;
    if ((mod != 0) && (b < 0) != (mod < 0)) {
      div -= T(1);
    }

    T floordiv;
    if (div != 0) {
      floordiv = std::floor(div);
      if (div - floordiv > T(0.5)) {
        floordiv += T(1.0);
      }
    } else {
      floordiv = phi::copysign(T(0), a / b);
    }
    return floordiv;
  }
};

template <>
struct FloorDivideFunctor<dtype::float16> {
  inline HOSTDEVICE dtype::float16 operator()(const dtype::float16 a,
                                              const dtype::float16 b) const {
    float b_float = static_cast<float>(b);
    float a_float = static_cast<float>(a);

    if (UNLIKELY(b_float == 0)) {
      // Divide by zero: return standard IEEE result
      return static_cast<dtype::float16>(a_float / b_float);
    }

    auto mod = std::fmod(a_float, b_float);
    auto div = (a_float - mod) / b_float;
    if ((mod != 0) && (b_float < 0) != (mod < 0)) {
      div -= static_cast<float>(1);
    }

    float floordiv;
    if (div != 0) {
      floordiv = std::floor(div);
      if (div - floordiv > static_cast<float>(0.5)) {
        floordiv += static_cast<float>(1.0);
      }
    } else {
      floordiv = phi::copysign(static_cast<float>(0), a_float / b_float);
    }

    return static_cast<dtype::float16>(floordiv);
  }
};

template <>
struct FloorDivideFunctor<dtype::bfloat16> {
  inline HOSTDEVICE dtype::bfloat16 operator()(const dtype::bfloat16 a,
                                               const dtype::bfloat16 b) const {
    float b_float = static_cast<float>(b);
    float a_float = static_cast<float>(a);

    if (UNLIKELY(b_float == 0)) {
      // Divide by zero: return standard IEEE result
      return static_cast<dtype::bfloat16>(a_float / b_float);
    }

    auto mod = std::fmod(a_float, b_float);
    auto div = (a_float - mod) / b_float;
    if ((mod != 0) && (b_float < 0) != (mod < 0)) {
      div -= static_cast<float>(1);
    }

    float floordiv;
    if (div != 0) {
      floordiv = std::floor(div);
      if (div - floordiv > static_cast<float>(0.5)) {
        floordiv += static_cast<float>(1.0);
      }
    } else {
      floordiv = phi::copysign(static_cast<float>(0), a_float / b_float);
    }

    return static_cast<dtype::bfloat16>(floordiv);
  }
};

template <typename T, typename Enable = void>
struct InverseFloorDivideFunctor {
  inline HOSTDEVICE T operator()(const T a, const T b) const {
#ifndef PADDLE_WITH_XPU_KP
    PADDLE_ENFORCE(a != 0, DIV_ERROR_INFO);
#endif
    if (phi::is_negative(a) != phi::is_negative(b)) {
      // Subtracts one from the results of truncation division if the
      // divisor and dividend have different sign(bit)s and the remainder of
      // the division is nonzero
      const auto quot = b / a;
      const auto rem = b % a;
      auto ret = rem ? quot - 1 : quot;
      return static_cast<T>(ret);
    }

    return static_cast<T>(b / a);
  }
};

template <typename T>
struct InverseFloorDivideFunctor<
    T,
    typename std::enable_if_t<std::is_floating_point<T>::value>> {
  inline HOSTDEVICE T operator()(const T a, const T b) const {
    if (UNLIKELY(a == 0)) {
      // Divide by zero: return standard IEEE result
      return static_cast<T>(b / a);
    }

    auto mod = std::fmod(b, a);
    auto div = (b - mod) / a;
    if ((mod != 0) && (a < 0) != (mod < 0)) {
      div -= T(1);
    }

    T floordiv;
    if (div != 0) {
      floordiv = std::floor(div);
      if (div - floordiv > T(0.5)) {
        floordiv += T(1.0);
      }
    } else {
      floordiv = phi::copysign(T(0), b / a);
    }
    return floordiv;
  }
};

template <>
struct InverseFloorDivideFunctor<dtype::float16> {
  inline HOSTDEVICE dtype::float16 operator()(const dtype::float16 a,
                                              const dtype::float16 b) const {
    float b_float = static_cast<float>(a);
    float a_float = static_cast<float>(b);

    if (UNLIKELY(b_float == 0)) {
      // Divide by zero: return standard IEEE result
      return static_cast<dtype::float16>(a_float / b_float);
    }

    auto mod = std::fmod(a_float, b_float);
    auto div = (a_float - mod) / b_float;
    if ((mod != 0) && (b_float < 0) != (mod < 0)) {
      div -= static_cast<float>(1);
    }

    float floordiv;
    if (div != 0) {
      floordiv = std::floor(div);
      if (div - floordiv > static_cast<float>(0.5)) {
        floordiv += static_cast<float>(1.0);
      }
    } else {
      floordiv = phi::copysign(static_cast<float>(0), a_float / b_float);
    }

    return static_cast<dtype::float16>(floordiv);
  }
};

template <>
struct InverseFloorDivideFunctor<dtype::bfloat16> {
  inline HOSTDEVICE dtype::bfloat16 operator()(const dtype::bfloat16 a,
                                               const dtype::bfloat16 b) const {
    float b_float = static_cast<float>(a);
    float a_float = static_cast<float>(b);

    if (UNLIKELY(b_float == 0)) {
      // Divide by zero: return standard IEEE result
      return static_cast<dtype::bfloat16>(a_float / b_float);
    }

    auto mod = std::fmod(a_float, b_float);
    auto div = (a_float - mod) / b_float;
    if ((mod != 0) && (b_float < 0) != (mod < 0)) {
      div -= static_cast<float>(1);
    }

    float floordiv;
    if (div != 0) {
      floordiv = std::floor(div);
      if (div - floordiv > static_cast<float>(0.5)) {
        floordiv += static_cast<float>(1.0);
      }
    } else {
      floordiv = phi::copysign(static_cast<float>(0), a_float / b_float);
    }

    return static_cast<dtype::bfloat16>(floordiv);
  }
};

#if defined(__CUDA_ARCH__) || defined(__HIPCC__)
template <typename T, typename MPType>
inline HOSTDEVICE typename std::enable_if<std::is_integral<T>::value, T>::type
compute_pow(const T a, const T b) {
  // TODO(wujionghao): A potential speed improvement is supporting different
  // types in C++.
  // On CUDAPlace, std::pow(3, 1) calls pow(float, float), and
  // it will return a float number like 2.99... , which floor to 2
  // when cast to int by default and it is wrong.
  // Use llrint to cast it to the nearest integer, which is 3.
  return std::llrint(std::pow(static_cast<double>(a), static_cast<double>(b)));
}
template <typename T, typename MPType>
inline HOSTDEVICE typename std::enable_if<!std::is_integral<T>::value, T>::type
compute_pow(const T a, const T b) {
  MPType a_val = static_cast<MPType>(a);
  MPType b_val = static_cast<MPType>(b);
#ifdef PADDLE_WITH_XPU_KP
  return static_cast<T>(pow(a_val, b_val));
#endif
  return static_cast<T>(std::pow(a_val, b_val));
}
#else
template <typename T, typename MPType>
inline HOSTDEVICE typename std::enable_if<
    !std::is_same<T, phi::dtype::complex<float>>::value &&
        !std::is_same<T, phi::dtype::complex<double>>::value,
    T>::type
compute_pow(const T a, const T b) {
  MPType a_val = static_cast<MPType>(a);
  MPType b_val = static_cast<MPType>(b);
#ifdef PADDLE_WITH_XPU_KP
  return static_cast<T>(pow(a_val, b_val));
#endif
  return static_cast<T>(std::pow(a_val, b_val));
}
template <typename T, typename MPType>
inline HOSTDEVICE typename std::enable_if<
    std::is_same<T, phi::dtype::complex<float>>::value ||
        std::is_same<T, phi::dtype::complex<double>>::value,
    T>::type
compute_pow(const T a, const T b) {
  MPType a_val = static_cast<MPType>(a);
  MPType b_val = static_cast<MPType>(b);
  return static_cast<T>(pow(a_val, b_val));
}
#endif

template <typename T>
struct ElementwisePowFunctor {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
  inline HOSTDEVICE T operator()(const T a, const T b) const {
    return compute_pow<T, MPType>(a, b);
  }
};

template <typename T>
struct ElementwiseInversePowFunctor {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
  inline HOSTDEVICE T operator()(const T a, const T b) const {
    return compute_pow<T, MPType>(b, a);
  }
};

// copysign forward and grad functors
template <typename T>
inline HOSTDEVICE auto copysign_func(const T& a, const T& b) {
#ifdef WIN32
  using U = typename std::conditional_t<std::is_integral<T>::value, float, T>;
  return static_cast<T>(std::copysign(static_cast<U>(a), static_cast<U>(b)));
#else
  return static_cast<T>(std::copysign(a, b));
#endif
}

inline HOSTDEVICE phi::dtype::float16 copysign_func(phi::dtype::float16 a,
                                                    phi::dtype::float16 b) {
  return phi::dtype::raw_uint16_to_float16((a.x & 0x7fff) | (b.x & 0x8000));
}

inline HOSTDEVICE phi::dtype::bfloat16 copysign_func(phi::dtype::bfloat16 a,
                                                     phi::dtype::bfloat16 b) {
  return phi::dtype::raw_uint16_to_bfloat16((a.x & 0x7fff) | (b.x & 0x8000));
}

template <typename T>
struct CopySignGradXFunctor {
  inline HOSTDEVICE T operator()(const T x, const T y, const T dout) const {
    if (x == static_cast<T>(0)) return x;
    return dout * (funcs::copysign_func(x, y) / x);
  }
};

template <typename T>
struct CopySignGradYFunctor {
  inline HOSTDEVICE T operator()(const T x, const T y, const T dout) const {
    return static_cast<T>(0);
  }
};

template <typename InT, typename OutT>
struct CopySignGradXYFunctor {
  inline HOSTDEVICE phi::Array<OutT, 2> operator()(const InT x,
                                                   const InT y,
                                                   const InT dout) {
    phi::Array<OutT, 2> outs;
    // dx
    if (x == static_cast<InT>(0))
      outs[0] = static_cast<OutT>(0);
    else
      outs[0] = static_cast<OutT>(dout * (funcs::copysign_func(x, y)) / x);
    // dy = 0
    outs[1] = static_cast<OutT>(0);
    return outs;
  }
};

template <typename T>
struct CopySignFunctor {
  inline HOSTDEVICE T operator()(const T a, const T b) const {
    return copysign_func(a, b);
  }
};
template <typename T>
struct InverseCopySignFunctor {
  inline HOSTDEVICE T operator()(const T a, const T b) const {
    return copysign_func(b, a);
  }
};

}  // namespace funcs
}  // namespace phi
