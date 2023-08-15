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

#include <stdint.h>

#include <complex>
#include <cstring>
#include <iostream>
#include <limits>
#include "paddle/phi/core/hostdevice.h"
#ifdef PADDLE_WITH_CUDA
#include <cuComplex.h>
#include <thrust/complex.h>
#endif  // PADDLE_WITH_CUDA

#ifdef PADDLE_WITH_HIP
#include <hip/hip_complex.h>
#include <thrust/complex.h>  // NOLINT
#endif

#ifndef PADDLE_WITH_HIP
#if !defined(_WIN32)
#define PADDLE_ALIGN(x) __attribute__((aligned(x)))
#else
#define PADDLE_ALIGN(x) __declspec(align(x))
#endif
#else
#define PADDLE_ALIGN(x)
#endif

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
// todo
#define PADDLE_WITH_CUDA_OR_HIP_COMPLEX
#endif

namespace phi {
namespace dtype {

template <typename T>
struct PADDLE_ALIGN(sizeof(T) * 2) complex {
 public:
  T real;
  T imag;

  using value_type = T;

  complex() = default;
  complex(const complex<T>& o) = default;
  complex& operator=(const complex<T>& o) = default;
  complex(complex<T>&& o) = default;
  complex& operator=(complex<T>&& o) = default;
  ~complex() = default;

  HOSTDEVICE complex(T real, T imag) : real(real), imag(imag) {}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)

  template <typename T1>
  HOSTDEVICE inline explicit complex(const thrust::complex<T1>& c) {
    real = c.real();
    imag = c.imag();
  }

  template <typename T1>
  HOSTDEVICE inline explicit operator thrust::complex<T1>() const {
    return thrust::complex<T1>(real, imag);
  }

#ifdef PADDLE_WITH_HIP
  HOSTDEVICE inline explicit operator hipFloatComplex() const {
    return make_hipFloatComplex(real, imag);
  }

  HOSTDEVICE inline explicit operator hipDoubleComplex() const {
    return make_hipDoubleComplex(real, imag);
  }
#else
  HOSTDEVICE inline explicit operator cuFloatComplex() const {
    return make_cuFloatComplex(real, imag);
  }

  HOSTDEVICE inline explicit operator cuDoubleComplex() const {
    return make_cuDoubleComplex(real, imag);
  }
#endif
#endif

  template <typename T1,
            typename std::enable_if<std::is_floating_point<T1>::value ||
                                        std::is_integral<T1>::value,
                                    int>::type = 0>
  HOSTDEVICE complex(const T1& val) {
    real = static_cast<T>(val);
    imag = static_cast<T>(0.0);
  }

  template <typename T1 = T>
  HOSTDEVICE explicit complex(
      const typename std::enable_if<std::is_same<T1, float>::value,
                                    complex<double>>::type& val) {
    real = val.real;
    imag = val.imag;
  }

  template <typename T1 = T>
  HOSTDEVICE explicit complex(
      const typename std::enable_if<std::is_same<T1, double>::value,
                                    complex<float>>::type& val) {
    real = val.real;
    imag = val.imag;
  }

  template <typename T1>
  HOSTDEVICE inline explicit operator std::complex<T1>() const {
    return static_cast<std::complex<T1>>(std::complex<T>(real, imag));
  }

  template <typename T1>
  HOSTDEVICE complex(const std::complex<T1>& val)
      : real(val.real()), imag(val.imag()) {}

  template <typename T1,
            typename std::enable_if<std::is_floating_point<T1>::value ||
                                        std::is_integral<T1>::value,
                                    int>::type = 0>
  HOSTDEVICE inline complex& operator=(const T1& val) {
    real = static_cast<T>(val);
    imag = static_cast<T>(0.0);
    return *this;
  }

  HOSTDEVICE inline explicit operator bool() const {
    return static_cast<bool>(this->real) || static_cast<bool>(this->imag);
  }

  HOSTDEVICE inline explicit operator int8_t() const {
    return static_cast<int8_t>(this->real);
  }

  HOSTDEVICE inline explicit operator uint8_t() const {
    return static_cast<uint8_t>(this->real);
  }

  HOSTDEVICE inline explicit operator int16_t() const {
    return static_cast<int16_t>(this->real);
  }

  HOSTDEVICE inline explicit operator uint16_t() const {
    return static_cast<uint16_t>(this->real);
  }

  HOSTDEVICE inline explicit operator int32_t() const {
    return static_cast<int32_t>(this->real);
  }

  HOSTDEVICE inline explicit operator uint32_t() const {
    return static_cast<uint32_t>(this->real);
  }

  HOSTDEVICE inline explicit operator int64_t() const {
    return static_cast<int64_t>(this->real);
  }

  HOSTDEVICE inline explicit operator uint64_t() const {
    return static_cast<uint64_t>(this->real);
  }

  HOSTDEVICE inline explicit operator float() const {
    return static_cast<float>(this->real);
  }

  HOSTDEVICE inline explicit operator double() const {
    return static_cast<double>(this->real);
  }
};

template <typename T>
HOSTDEVICE inline complex<T> operator+(const complex<T>& a,
                                       const complex<T>& b) {
#if defined(PADDLE_WITH_CUDA_OR_HIP_COMPLEX) && \
    (defined(__CUDA_ARCH__) || defined(__HIPCC__))
  return complex<T>(thrust::complex<T>(a) + thrust::complex<T>(b));
#else
  return complex<T>(a.real + b.real, a.imag + b.imag);
#endif
}

template <typename T>
HOSTDEVICE inline complex<T> operator-(const complex<T>& a,
                                       const complex<T>& b) {
#if defined(PADDLE_WITH_CUDA_OR_HIP_COMPLEX) && \
    (defined(__CUDA_ARCH__) || defined(__HIPCC__))
  return complex<T>(thrust::complex<T>(a) - thrust::complex<T>(b));
#else
  return complex<T>(a.real - b.real, a.imag - b.imag);
#endif
}

template <typename T>
HOSTDEVICE inline complex<T> operator*(const complex<T>& a,
                                       const complex<T>& b) {
#if defined(PADDLE_WITH_CUDA_OR_HIP_COMPLEX) && \
    (defined(__CUDA_ARCH__) || defined(__HIPCC__))
  return complex<T>(thrust::complex<T>(a) * thrust::complex<T>(b));
#else
  return complex<T>(a.real * b.real - a.imag * b.imag,
                    a.imag * b.real + b.imag * a.real);
#endif
}

template <typename T>
HOSTDEVICE inline complex<T> operator/(const complex<T>& a,
                                       const complex<T>& b) {
#if defined(PADDLE_WITH_CUDA_OR_HIP_COMPLEX) && \
    (defined(__CUDA_ARCH__) || defined(__HIPCC__))
  return complex<T>(thrust::complex<T>(a) / thrust::complex<T>(b));
#else
  T denominator = b.real * b.real + b.imag * b.imag;
  return complex<T>((a.real * b.real + a.imag * b.imag) / denominator,
                    (a.imag * b.real - a.real * b.imag) / denominator);
#endif
}

template <typename T>
HOSTDEVICE inline complex<T> operator-(const complex<T>& a) {
#if defined(PADDLE_WITH_CUDA_OR_HIP_COMPLEX) && \
    (defined(__CUDA_ARCH__) || defined(__HIPCC__))
  return complex<T>(-thrust::complex<T>(a.real, a.imag));
#else
  complex<T> res;
  res.real = -a.real;
  res.imag = -a.imag;
  return res;
#endif
}

template <typename T>
HOSTDEVICE inline complex<T>& operator+=(complex<T>& a,  // NOLINT
                                         const complex<T>& b) {
#if defined(PADDLE_WITH_CUDA_OR_HIP_COMPLEX) && \
    (defined(__CUDA_ARCH__) || defined(__HIPCC__))
  a = complex<T>(thrust::complex<T>(a.real, a.imag) +=
                 thrust::complex<T>(b.real, b.imag));
  return a;
#else
  a.real += b.real;
  a.imag += b.imag;
  return a;
#endif
}

template <typename T>
HOSTDEVICE inline complex<T>& operator-=(complex<T>& a,  // NOLINT
                                         const complex<T>& b) {
#if defined(PADDLE_WITH_CUDA_OR_HIP_COMPLEX) && \
    (defined(__CUDA_ARCH__) || defined(__HIPCC__))
  a = complex<T>(thrust::complex<T>(a.real, a.imag) -=
                 thrust::complex<T>(b.real, b.imag));
  return a;
#else
  a.real -= b.real;
  a.imag -= b.imag;
  return a;
#endif
}

template <typename T>
HOSTDEVICE inline complex<T>& operator*=(complex<T>& a,  // NOLINT
                                         const complex<T>& b) {
#if defined(PADDLE_WITH_CUDA_OR_HIP_COMPLEX) && \
    (defined(__CUDA_ARCH__) || defined(__HIPCC__))
  a = complex<T>(thrust::complex<T>(a.real, a.imag) *=
                 thrust::complex<T>(b.real, b.imag));
  return a;
#else
  a.real = a.real * b.real - a.imag * b.imag;
  a.imag = a.imag * b.real + b.imag * a.real;
  return a;
#endif
}

template <typename T>
HOSTDEVICE inline complex<T>& operator/=(complex<T>& a,  // NOLINT
                                         const complex<T>& b) {
#if defined(PADDLE_WITH_CUDA_OR_HIP_COMPLEX) && \
    (defined(__CUDA_ARCH__) || defined(__HIPCC__))
  a = complex<T>(thrust::complex<T>(a.real, a.imag) /=
                 thrust::complex<T>(b.real, b.imag));
  return a;
#else
  T denominator = b.real * b.real + b.imag * b.imag;
  a.real = (a.real * b.real + a.imag * b.imag) / denominator;
  a.imag = (a.imag * b.real - a.real * b.imag) / denominator;
  return a;
#endif
}

template <typename T>
HOSTDEVICE inline complex<T> raw_uint16_to_complex64(uint16_t a) {
  complex<T> res;
  res.real = a;
  res.imag = 0.0;
  return res;
}

template <typename T>
HOSTDEVICE inline bool operator==(const complex<T>& a, const complex<T>& b) {
  return a.real == b.real && a.imag == b.imag;
}

template <typename T>
HOSTDEVICE inline bool operator!=(const complex<T>& a, const complex<T>& b) {
  return a.real != b.real || a.imag != b.imag;
}

template <typename T>
HOSTDEVICE inline bool operator<(const complex<T>& a, const complex<T>& b) {
  return a.real < b.real;
}

template <typename T>
HOSTDEVICE inline bool operator<=(const complex<T>& a, const complex<T>& b) {
  return a.real <= b.real;
}

template <typename T>
HOSTDEVICE inline bool operator>(const complex<T>& a, const complex<T>& b) {
  return a.real > b.real;
}

template <typename T>
HOSTDEVICE inline bool operator>=(const complex<T>& a, const complex<T>& b) {
  return a.real >= b.real;
}

template <typename T>
HOSTDEVICE inline complex<T>(max)(const complex<T>& a, const complex<T>& b) {
  return (a.real >= b.real) ? a : b;
}

template <typename T>
HOSTDEVICE inline complex<T>(min)(const complex<T>& a, const complex<T>& b) {
  return (a.real < b.real) ? a : b;
}

template <typename T>
HOSTDEVICE inline bool(isnan)(const complex<T>& a) {
#if defined(PADDLE_WITH_CUDA_OR_HIP_COMPLEX) && \
    (defined(__CUDA_ARCH__) || defined(__HIPCC__))
  return ::isnan(a.real) || ::isnan(a.imag);
#else
  return std::isnan(a.real) || std::isnan(a.imag);
#endif
}

template <typename T>
HOSTDEVICE inline bool isinf(const complex<T>& a) {
#if defined(PADDLE_WITH_CUDA_OR_HIP_COMPLEX) && \
    (defined(__CUDA_ARCH__) || defined(__HIPCC__))
  return ::isinf(a.real) || ::isinf(a.imag);
#else
  return std::isinf(a.real) || std::isinf(a.imag);
#endif
}

template <typename T>
HOSTDEVICE inline bool isfinite(const complex<T>& a) {
#if defined(PADDLE_WITH_CUDA_OR_HIP_COMPLEX) && \
    (defined(__CUDA_ARCH__) || defined(__HIPCC__))
  return ::isfinite(a.real) || ::isfinite(a.imag);
#else
  return std::isfinite(a.real) || std::isfinite(a.imag);
#endif
}

template <typename T>
HOSTDEVICE inline T abs(const complex<T>& a) {
#if defined(PADDLE_WITH_CUDA_OR_HIP_COMPLEX) && \
    (defined(__CUDA_ARCH__) || defined(__HIPCC__))
  return thrust::abs(thrust::complex<T>(a));
#else
  return std::abs(std::complex<T>(a));
#endif
}

template <typename T>
HOSTDEVICE inline T arg(const complex<T>& a) {
#if defined(PADDLE_WITH_CUDA_OR_HIP_COMPLEX) && \
    (defined(__CUDA_ARCH__) || defined(__HIPCC__))
  return thrust::arg(thrust::complex<T>(a));
#else
  return std::arg(std::complex<T>(a));
#endif
}

template <typename T>
HOSTDEVICE inline complex<T> pow(const complex<T>& a, const complex<T>& b) {
#if defined(PADDLE_WITH_CUDA_OR_HIP_COMPLEX) && \
    (defined(__CUDA_ARCH__) || defined(__HIPCC__))
  return complex<T>(thrust::pow(thrust::complex<T>(a), thrust::complex<T>(b)));
#else
  return complex<T>(std::pow(std::complex<T>(a), std::complex<T>(b)));
#endif
}

template <typename T>
HOSTDEVICE inline complex<T> sqrt(const complex<T>& a) {
#if defined(PADDLE_WITH_CUDA_OR_HIP_COMPLEX) && \
    (defined(__CUDA_ARCH__) || defined(__HIPCC__))
  return complex<T>(thrust::sqrt(thrust::complex<T>(a)));
#else
  return complex<T>(std::sqrt(std::complex<T>(a)));
#endif
}

template <typename T>
HOSTDEVICE inline complex<T> sin(const complex<T>& a) {
#if defined(PADDLE_WITH_CUDA_OR_HIP_COMPLEX) && \
    (defined(__CUDA_ARCH__) || defined(__HIPCC__))
  return complex<T>(thrust::sin(thrust::complex<T>(a)));
#else
  return complex<T>(std::sin(std::complex<T>(a)));
#endif
}

template <typename T>
HOSTDEVICE inline complex<T> cos(const complex<T>& a) {
#if defined(PADDLE_WITH_CUDA_OR_HIP_COMPLEX) && \
    (defined(__CUDA_ARCH__) || defined(__HIPCC__))
  return complex<T>(thrust::cos(thrust::complex<T>(a)));
#else
  return complex<T>(std::cos(std::complex<T>(a)));
#endif
}

template <typename T>
HOSTDEVICE inline complex<T> tan(const complex<T>& a) {
#if defined(PADDLE_WITH_CUDA_OR_HIP_COMPLEX) && \
    (defined(__CUDA_ARCH__) || defined(__HIPCC__))
  return complex<T>(thrust::tan(thrust::complex<T>(a)));
#else
  return complex<T>(std::tan(std::complex<T>(a)));
#endif
}

template <typename T>
HOSTDEVICE inline complex<T> tanh(const complex<T>& a) {
#if defined(PADDLE_WITH_CUDA_OR_HIP_COMPLEX) && \
    (defined(__CUDA_ARCH__) || defined(__HIPCC__))
  return complex<T>(thrust::tanh(thrust::complex<T>(a)));
#else
  return complex<T>(std::tanh(std::complex<T>(a)));
#endif
}

template <typename T>
HOSTDEVICE inline complex<T> conj(const complex<T>& a) {
#if defined(PADDLE_WITH_CUDA_OR_HIP_COMPLEX) && \
    (defined(__CUDA_ARCH__) || defined(__HIPCC__))
  return complex<T>(thrust::conj(thrust::complex<T>(a)));
#else
  return complex<T>(std::conj(std::complex<T>(a)));
#endif
}

template <typename T>
HOSTDEVICE inline complex<T> log(const complex<T>& a) {
#if defined(PADDLE_WITH_CUDA_OR_HIP_COMPLEX) && \
    (defined(__CUDA_ARCH__) || defined(__HIPCC__))
  return complex<T>(thrust::log(thrust::complex<T>(a)));
#else
  return complex<T>(std::log(std::complex<T>(a)));
#endif
}

template <typename T>
HOSTDEVICE inline complex<T> floor(const complex<T>& a) {
#if defined(PADDLE_WITH_CUDA_OR_HIP_COMPLEX) && \
    (defined(__CUDA_ARCH__) || defined(__HIPCC__))
  return complex<T>({thrust::floor(a.real()), thrust::floor(a.imag())});
#else
  return complex<T>({std::floor(a.real()), std::floor(a.imag())});
#endif
}

template <typename T>
inline std::ostream& operator<<(std::ostream& os, const complex<T>& a) {
  os << "real:" << a.real << " imag:" << a.imag;
  return os;
}
}  // namespace dtype
}  // namespace phi

namespace std {

template <typename T>
struct is_pod<phi::dtype::complex<T>> {
  static const bool value = true;
};

template <typename T>
struct is_floating_point<phi::dtype::complex<T>>
    : std::integral_constant<bool, false> {};

template <typename T>
struct is_signed<phi::dtype::complex<T>> {
  static const bool value = false;
};

template <typename T>
struct is_unsigned<phi::dtype::complex<T>> {
  static const bool value = false;
};

template <typename T>
inline bool isnan(const phi::dtype::complex<T>& a) {
  return phi::dtype::isnan(a);
}

template <typename T>
inline bool isinf(const phi::dtype::complex<T>& a) {
  return phi::dtype::isinf(a);
}

template <typename T>
struct numeric_limits<phi::dtype::complex<T>> {
  static const bool is_specialized = false;
  static const bool is_signed = false;
  static const bool is_integer = false;
  static const bool is_exact = false;
  static const bool has_infinity = false;
  static const bool has_quiet_NaN = false;
  static const bool has_signaling_NaN = false;
  static const float_denorm_style has_denorm = denorm_absent;
  static const bool has_denorm_loss = false;
  static const std::float_round_style round_style = std::round_toward_zero;
  static const bool is_iec559 = false;
  static const bool is_bounded = false;
  static const bool is_modulo = false;
  static const int digits = 0;
  static const int digits10 = 0;
  static const int max_digits10 = 0;
  static const int radix = 0;
  static const int min_exponent = 0;
  static const int min_exponent10 = 0;
  static const int max_exponent = 0;
  static const int max_exponent10 = 0;
  static const bool traps = false;
  static const bool tinyness_before = false;

  static phi::dtype::complex<T>(min)() {
    return phi::dtype::complex<T>(0.0, 0.0);
  }
  static phi::dtype::complex<T> lowest() {
    return phi::dtype::complex<T>(0.0, 0.0);
  }
  static phi::dtype::complex<T>(max)() {
    return phi::dtype::complex<T>(0.0, 0.0);
  }
  static phi::dtype::complex<T> epsilon() {
    return phi::dtype::complex<T>(0.0, 0.0);
  }
  static phi::dtype::complex<T> round_error() {
    return phi::dtype::complex<T>(0.0, 0.0);
  }
  static phi::dtype::complex<T> infinity() {
    return phi::dtype::complex<T>(0.0, 0.0);
  }
  static phi::dtype::complex<T> quiet_NaN() {
    return phi::dtype::complex<T>(0.0, 0.0);
  }
  static phi::dtype::complex<T> signaling_NaN() {
    return phi::dtype::complex<T>(0.0, 0.0);
  }
  static phi::dtype::complex<T> denorm_min() {
    return phi::dtype::complex<T>(0.0, 0.0);
  }
};

}  // namespace std
