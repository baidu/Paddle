// Copyright (c) 2022 CINN Authors. All Rights Reserved.
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

#ifndef CINN_COMMON_FLOAT16_H
#define CINN_COMMON_FLOAT16_H

#ifdef __cplusplus
#pragma once
#endif  // __cplusplus

#if defined(_M_X64) || defined(__x86_64__) || defined(_M_IX86) || \
    defined(__i386__)
#define __CINN_x86__
#include <immintrin.h>
#endif

#include <cstdint>

#ifdef CINN_WITH_CUDA
#include <cuda.h>

#if (defined(__CUDACC__) || defined(__CUDACC_RTC__)) && CUDA_VERSION >= 7050
#define CINN_CUDA_FP16
#include <cuda_fp16.h>
#else
#include <cmath>
#include <cstring>

#define CUDA_ARCH_FP16_SUPPORTED(CUDA_ARCH) (CUDA_ARCH >= 600)
#endif  // __CUDACC__
#endif  // CINN_WITH_CUDA

#ifdef __cplusplus
#ifndef _WIN32
#define CINN_ALIGN(x) __attribute__((aligned(x)))
#else  // _WIN32
#define CINN_ALIGN(x) __declspec(align(x))
#endif  // _WIN32

#else  // __cplusplus
#define CINN_ALIGN(x)
#endif  // __cplusplus

// The `HOST` macro definition is not used here, it has a potential
// conflict with the enumeration `kHOST` representing the backend.
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif

#ifdef __cplusplus
namespace cinn {
namespace common {
#endif  // __cplusplus

// Use CINN_ALIGNED(2) to ensure that each float16 will be allocated
// and aligned at least on a 2-byte boundary, which leads to efficient
// memory access of float16 struct and also makes float16 compatible
// with CUDA half
struct CINN_ALIGN(2) float16 {
  uint16_t x;

#ifdef __cplusplus
  // The following defaulted special class member functions
  // are added to make float16 pass the std::is_trivial test
  float16() = default;
  float16(const float16& o) = default;
  float16& operator=(const float16& o) = default;
  float16(float16&& o) = default;
  float16& operator=(float16&& o) = default;
  ~float16() = default;

// Constructors
#ifdef CINN_CUDA_FP16
  __host__ __device__ inline explicit float16(const half& h) {
#if (CUDA_VERSION >= 9000)
    x = reinterpret_cast<__half_raw*>(const_cast<half*>(&h))->x;
#else
    x = h.x;
#endif  // CUDA_VERSION >= 9000
  }
#endif  // CINN_CUDA_FP16

  __host__ __device__ inline explicit float16(float val) {
#if defined(CINN_CUDA_FP16) && (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300)
    half tmp = __float2half(val);
    x = *reinterpret_cast<uint16_t*>(&tmp);

#elif defined(__F16C__) && defined(__CINN_x86__)
    x = _cvtss_sh(val, 0);

#else
    // Conversion routine adapted from
    // http://stackoverflow.com/questions/1659440/32-bit-to-16-bit-floating-point-conversion
    Bits v, s;
    v.f = val;
    uint32_t sign = v.si & sigN;
    v.si ^= sign;
    sign >>= shiftSign;  // logical shift
    s.si = mulN;
    s.si = s.f * v.f;  // correct subnormals
    v.si ^= (s.si ^ v.si) & -(minN > v.si);
    v.si ^= (infN ^ v.si) & -((infN > v.si) & (v.si > maxN));
    v.si ^= (nanN ^ v.si) & -((nanN > v.si) & (v.si > infN));
    v.ui >>= shift;  // logical shift
    v.si ^= ((v.si - maxD) ^ v.si) & -(v.si > maxC);
    v.si ^= ((v.si - minD) ^ v.si) & -(v.si > subC);
    x = v.ui | sign;

#endif
  }

  __host__ __device__ inline explicit float16(bool b) : x(b ? 0x3c00 : 0) {}

  template <class T>
  __host__ __device__ inline explicit float16(const T& val)
      : x(float16(static_cast<float>(val)).x) {}

// Assignment operators
#ifdef CINN_CUDA_FP16
  __host__ __device__ inline float16& operator=(const half& rhs) {
#if CUDA_VERSION >= 9000
    x = reinterpret_cast<__half_raw*>(const_cast<half*>(&rhs))->x;
#else
    x = rhs.x;
#endif
    return *this;
  }
#endif

  __host__ __device__ inline float16& operator=(bool b) {
    x = b ? 0x3c00 : 0;
    return *this;
  }

  __host__ __device__ inline float16& operator=(int8_t val) {
    x = float16(val).x;
    return *this;
  }

  __host__ __device__ inline float16& operator=(uint8_t val) {
    x = float16(val).x;
    return *this;
  }

  __host__ __device__ inline float16& operator=(int16_t val) {
    x = float16(val).x;
    return *this;
  }

  __host__ __device__ inline float16& operator=(uint16_t val) {
    x = float16(val).x;
    return *this;
  }

  __host__ __device__ inline float16& operator=(int32_t val) {
    x = float16(val).x;
    return *this;
  }

  __host__ __device__ inline float16& operator=(uint32_t val) {
    x = float16(val).x;
    return *this;
  }

  __host__ __device__ inline float16& operator=(int64_t val) {
    x = float16(val).x;
    return *this;
  }

  __host__ __device__ inline float16& operator=(uint64_t val) {
    x = float16(val).x;
    return *this;
  }

  __host__ __device__ inline float16& operator=(float val) {
    x = float16(val).x;
    return *this;
  }

  __host__ __device__ inline float16& operator=(double val) {
    x = float16(val).x;
    return *this;
  }

// Conversion operators
#ifdef CINN_CUDA_FP16
  __host__ __device__ inline half to_half() const {
#if CUDA_VERSION >= 9000
    __half_raw h;
    h.x = x;
    return half(h);
#else
    half h;
    h.x = x;
    return h;
#endif  // CUDA_VERSION >= 9000
  }
#endif  // CINN_CUDA_FP16

  __host__ __device__ inline operator float() const {
#if defined(CINN_CUDA_FP16) && (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300)
    half tmp = *reinterpret_cast<const half*>(this);
    return __half2float(tmp);

#elif defined(__F16C__)
    return _cvtsh_ss(this->x);

#else
    // Conversion routine adapted from
    // http://stackoverflow.com/questions/1659440/32-bit-to-16-bit-floating-point-conversion
    Bits v;
    v.ui = this->x;
    int32_t sign = v.si & sigC;
    v.si ^= sign;
    sign <<= shiftSign;
    v.si ^= ((v.si + minD) ^ v.si) & -(v.si > subC);
    v.si ^= ((v.si + maxD) ^ v.si) & -(v.si > maxC);
    Bits s;
    s.si = mulC;
    s.f *= v.si;
    int32_t mask = -(norC > v.si);
    v.si <<= shift;
    v.si ^= (s.si ^ v.si) & mask;
    v.si |= sign;
    return v.f;

#endif
  }

  __host__ __device__ inline explicit operator bool() const {
    return (x & 0x7fff) != 0;
  }

  __host__ __device__ inline explicit operator int8_t() const {
    return static_cast<int8_t>(static_cast<float>(*this));
  }

  __host__ __device__ inline explicit operator uint8_t() const {
    return static_cast<uint8_t>(static_cast<float>(*this));
  }

  __host__ __device__ inline explicit operator int16_t() const {
    return static_cast<int16_t>(static_cast<float>(*this));
  }

  __host__ __device__ inline explicit operator uint16_t() const {
    return static_cast<uint16_t>(static_cast<float>(*this));
  }

  __host__ __device__ inline explicit operator int32_t() const {
    return static_cast<int32_t>(static_cast<float>(*this));
  }

  __host__ __device__ inline explicit operator uint32_t() const {
    return static_cast<uint32_t>(static_cast<float>(*this));
  }

  __host__ __device__ inline explicit operator int64_t() const {
    return static_cast<int64_t>(static_cast<float>(*this));
  }

  __host__ __device__ inline explicit operator uint64_t() const {
    return static_cast<uint64_t>(static_cast<float>(*this));
  }

  __host__ __device__ inline operator double() const {
    return static_cast<double>(static_cast<float>(*this));
  }

 private:
  union Bits {
    float f;
    int32_t si;
    uint32_t ui;
  };

  static const int shift = 13;
  static const int shiftSign = 16;

  static const int32_t infN = 0x7F800000;
  static const int32_t maxN = 0x477FE000;  // max flt16 as flt32
  static const int32_t minN = 0x38800000;  // min flt16 normal as flt32
  static const int32_t sigN = 0x80000000;  // sign bit

  static constexpr int32_t infC = infN >> shift;
  static constexpr int32_t nanN = (infC + 1)
                                  << shift;  // minimum flt16 nan as float32
  static constexpr int32_t maxC = maxN >> shift;
  static constexpr int32_t minC = minN >> shift;
  static constexpr int32_t sigC = sigN >> shiftSign;

  static const int32_t mulN = 0x52000000;  // (1 << 23) / minN
  static const int32_t mulC = 0x33800000;  // minN / (1 << (23 - shift))
  static const int32_t subC = 0x003FF;     // max flt32 subnormal downshifted
  static const int32_t norC = 0x00400;     // min flt32 normal downshifted

  static constexpr int32_t maxD = infC - maxC - 1;
  static constexpr int32_t minD = minC - subC - 1;
#endif  // __cplusplus
};

struct CINN_ALIGN(32) float8 {
  float x, y, z, w, v, u, t, s;
};

struct CINN_ALIGN(16) half8 {
  float16 x, y, z, w, v, u, t, s;
};

struct CINN_ALIGN(8) half4 {
  float16 x, y, z, w;
};

struct CINN_ALIGN(16) float168 {
  float16 x, y, z, w, v, u, t, s;
};

struct CINN_ALIGN(8) float164 {
  float16 x, y, z, w;
};

struct CINN_ALIGN(4) float162 {
  float16 x, y;
};

#ifdef __cplusplus
// Arithmetic operators on GPU
// CUDA 9.0 provides built-in arithmetic operators for half while
// CUDA 7.5 and 8.0 do not. The arithmetic operators defined here are
// for users to write similar CUDA code in CUDA 7.5 and 8.0 as in
// CUDA 9.0 regarding the half data type.
// ROCM has built-in arithmetic operators as not defined
// __HIP_NO_HALF_OPERATORS__
#if defined(CINN_CUDA_FP16) && CUDA_VERSION < 9000
__device__ inline half operator+(const half& a, const half& b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  return __hadd(a, b);
#else
  float res = static_cast<float>(float16(a)) + static_cast<float>(float16(b));
  return float16(res).to_half();
#endif
}

__device__ inline half operator-(const half& a, const half& b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  return __hsub(a, b);
#else
  float res = static_cast<float>(float16(a)) - static_cast<float>(float16(b));
  return float16(res).to_half();
#endif
}

__device__ inline half operator*(const half& a, const half& b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  return __hmul(a, b);
#else
  float res = static_cast<float>(float16(a)) * static_cast<float>(float16(b));
  return float16(res).to_half();
#endif
}

__device__ inline half operator/(const half& a, const half& b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  float num = __half2float(a);
  float denom = __half2float(b);
  return __float2half(num / denom);
#else
  float res = static_cast<float>(float16(a)) / static_cast<float>(float16(b));
  return float16(res).to_half();
#endif
}

__device__ inline half operator-(const half& a) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  return __hneg(a);
#else
  float res = -static_cast<float>(float16(a));
  return float16(res).to_half();
#endif
}

__device__ inline half& operator+=(half& a, const half& b) {  // NOLINT
  a = a + b;
  return a;
}

__device__ inline half& operator-=(half& a, const half& b) {  // NOLINT
  a = a - b;
  return a;
}

__device__ inline half& operator*=(half& a, const half& b) {  // NOLINT
  a = a * b;
  return a;
}

__device__ inline half& operator/=(half& a, const half& b) {  // NOLINT
  a = a / b;
  return a;
}

__device__ inline bool operator==(const half& a, const half& b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  return __heq(a, b);
#else
  return static_cast<float>(float16(a)) == static_cast<float>(float16(b));
#endif
}

__device__ inline bool operator!=(const half& a, const half& b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  return __hne(a, b);
#else
  return static_cast<float>(float16(a)) != static_cast<float>(float16(b));
#endif
}

__device__ inline bool operator<(const half& a, const half& b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  return __hlt(a, b);
#else
  return static_cast<float>(float16(a)) < static_cast<float>(float16(b));
#endif
}

__device__ inline bool operator<=(const half& a, const half& b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  return __hle(a, b);
#else
  return static_cast<float>(float16(a)) <= static_cast<float>(float16(b));
#endif
}

__device__ inline bool operator>(const half& a, const half& b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  return __hgt(a, b);
#else
  return static_cast<float>(float16(a)) > static_cast<float>(float16(b));
#endif
}

__device__ inline bool operator>=(const half& a, const half& b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  return __hge(a, b);
#else
  return static_cast<float>(float16(a)) >= static_cast<float>(float16(b));
#endif
}

#endif  // CINN_CUDA_FP16

// Arithmetic operators for float16 on GPU
__host__ __device__ inline float16 operator+(const float16& a,
                                             const float16& b) {
#if defined(CINN_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  return float16(__hadd(a.to_half(), b.to_half()));
#else
  return float16(static_cast<float>(a) + static_cast<float>(b));
#endif
}

__host__ __device__ inline float16 operator-(const float16& a,
                                             const float16& b) {
#if defined(CINN_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  return float16(__hsub(a.to_half(), b.to_half()));
#else
  return float16(static_cast<float>(a) - static_cast<float>(b));
#endif
}

__host__ __device__ inline float16 operator*(const float16& a,
                                             const float16& b) {
#if defined(CINN_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  return float16(__hmul(a.to_half(), b.to_half()));
#else
  return float16(static_cast<float>(a) * static_cast<float>(b));
#endif
}

__host__ __device__ inline float16 operator/(const float16& a,
                                             const float16& b) {
#if defined(CINN_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  // TODO(kexinzhao): check which cuda version starts to support __hdiv
  float num = __half2float(a.to_half());
  float denom = __half2float(b.to_half());
  return float16(num / denom);
#else
  return float16(static_cast<float>(a) / static_cast<float>(b));
#endif
}

__host__ __device__ inline float16 operator-(const float16& a) {
#if defined(CINN_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  return float16(__hneg(a.to_half()));
#else
  float16 res;
  res.x = a.x ^ 0x8000;
  return res;
#endif
}

__host__ __device__ inline float16& operator+=(float16& a,          // NOLINT
                                               const float16& b) {  // NOLINT
  a = a + b;
  return a;
}

__host__ __device__ inline float16& operator-=(float16& a,          // NOLINT
                                               const float16& b) {  // NOLINT
  a = a - b;
  return a;
}

__host__ __device__ inline float16& operator*=(float16& a,          // NOLINT
                                               const float16& b) {  // NOLINT
  a = a * b;
  return a;
}

__host__ __device__ inline float16& operator/=(float16& a,          // NOLINT
                                               const float16& b) {  // NOLINT
  a = a / b;
  return a;
}

__host__ __device__ inline bool operator==(const float16& a, const float16& b) {
#if defined(CINN_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  return __heq(a.to_half(), b.to_half());
#else
  return static_cast<float>(a) == static_cast<float>(b);
#endif
}

__host__ __device__ inline bool operator!=(const float16& a, const float16& b) {
#if defined(CINN_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  return __hne(a.to_half(), b.to_half());
#else
  return static_cast<float>(a) != static_cast<float>(b);
#endif
}

__host__ __device__ inline bool operator<(const float16& a, const float16& b) {
#if defined(CINN_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  return __hlt(a.to_half(), b.to_half());
#else
  return static_cast<float>(a) < static_cast<float>(b);
#endif
}

__host__ __device__ inline bool operator<=(const float16& a, const float16& b) {
#if defined(CINN_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  return __hle(a.to_half(), b.to_half());
#else
  return static_cast<float>(a) <= static_cast<float>(b);
#endif
}

__host__ __device__ inline bool operator>(const float16& a, const float16& b) {
#if defined(CINN_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  return __hgt(a.to_half(), b.to_half());
#else
  return static_cast<float>(a) > static_cast<float>(b);
#endif
}

__host__ __device__ inline bool operator>=(const float16& a, const float16& b) {
#if defined(CINN_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  return __hge(a.to_half(), b.to_half());
#else
  return static_cast<float>(a) >= static_cast<float>(b);
#endif
}
#endif  // __cplusplus

__host__ __device__ inline float16 raw_uint16_to_float16(uint16_t a) {
  float16 res;
  res.x = a;
  return res;
}

__host__ __device__ inline bool(isnan)(const float16& a) {
#if defined(CINN_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  return __hisnan(a.to_half());
#else
  return (a.x & 0x7fff) > 0x7c00;
#endif
}

__host__ __device__ inline bool(isinf)(const float16& a) {
  return (a.x & 0x7fff) == 0x7c00;
}

__host__ __device__ inline bool(isfinite)(const float16& a) {
  return !((isnan)(a)) && !((isinf)(a));
}

__host__ __device__ inline float16(abs)(const float16& a) {
#if defined(CINN_CUDA_FP16) && (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530)
  return static_cast<float16>(__habs(a.to_half()));
#else
  return static_cast<float16>(fabsf(static_cast<float>(a)));
#endif
}

#ifdef __cplusplus
}  // namespace common
}  // namespace cinn
#endif  // __cplusplus

#if defined(__cplusplus) && defined(CINN_CUDA_FP16)
__device__ inline cinn::common::float16 __shfl_sync(unsigned mask,
                                                    cinn::common::float16 var,
                                                    int srcLane,
                                                    int width = warpSize) {
  return cinn::common::float16(
      __shfl_sync(mask, var.to_half(), srcLane, width));
}

__device__ inline cinn::common::float16 __shfl_up_sync(
    unsigned mask,
    cinn::common::float16 var,
    unsigned int delta,
    int width = warpSize) {
  return cinn::common::float16(
      __shfl_up_sync(mask, var.to_half(), delta, width));
}

__device__ inline cinn::common::float16 __shfl_down_sync(
    unsigned mask,
    cinn::common::float16 var,
    unsigned int delta,
    int width = warpSize) {
  return cinn::common::float16(
      __shfl_down_sync(mask, var.to_half(), delta, width));
}

__device__ inline cinn::common::float16 __shfl_xor_sync(
    unsigned mask,
    cinn::common::float16 var,
    int laneMask,
    int width = warpSize) {
  return cinn::common::float16(
      __shfl_xor_sync(mask, var.to_half(), laneMask, width));
}

__host__ __device__ inline cinn::common::float16 max(
    const cinn::common::float16& a, const cinn::common::float16& b) {
  return a > b ? a : b;
}
__host__ __device__ inline cinn::common::float16 min(
    const cinn::common::float16& a, const cinn::common::float16& b) {
  return a < b ? a : b;
}
#endif  // __cplusplus && CINN_CUDA_FP16

#endif  // CINN_COMMON_FLOAT16_H
