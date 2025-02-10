// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

#ifndef CINN_COMMON_BFLOAT16_H
#define CINN_COMMON_BFLOAT16_H

#ifdef __cplusplus
#pragma once
#endif  // __cplusplus

#include <stdint.h>

#ifdef CINN_WITH_CUDA
#include <cuda.h>

#if (defined(__CUDACC__) || defined(__CUDACC_RTC__)) && CUDA_VERSION >= 11000
#define CINN_CUDA_BF16
#include <cuda_bf16.h>
#else
#include <cstring>

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

// Use CINN_ALIGNED(2) to ensure that each bfloat16 will be allocated
// and aligned at least on a 2-byte boundary, which leads to efficient
// memory access of float16 struct and also makes bfloat16 compatible
// with CUDA half
struct CINN_ALIGN(2) bfloat16 {
  uint16_t x;

#ifdef __cplusplus
  // Constructors
  bfloat16() = default;
  bfloat16(const bfloat16& o) = default;
  bfloat16& operator=(const bfloat16& o) = default;
  bfloat16(bfloat16&& o) = default;
  bfloat16& operator=(bfloat16&& o) = default;
  ~bfloat16() = default;

  __host__ __device__ inline explicit bfloat16(float val) {
#if defined(CINN_CUDA_BF16)
    __nv_bfloat16 tmp = __float2bfloat16(val);
    x = *reinterpret_cast<uint16_t*>(&tmp);
#else
    std::memcpy(&x, reinterpret_cast<char*>(&val) + 2, 2);
#endif
  }

#if defined(CINN_CUDA_BF16)
  __host__ __device__ inline explicit bfloat16(const __nv_bfloat16& val) {
    x = *reinterpret_cast<const unsigned short*>(&val);  // NOLINT
  }
#endif

  template <class T>
  __host__ __device__ inline explicit bfloat16(const T& val)
      : x(bfloat16(static_cast<float>(val)).x) {}

// Assignment operators
#if defined(CINN_CUDA_BF16)
  __host__ __device__ inline bfloat16& operator=(const __nv_bfloat16& val) {
    x = *reinterpret_cast<const unsigned short*>(&val);  // NOLINT
    return *this;
  }
#endif

  __host__ __device__ inline bfloat16& operator=(bool b) {
    x = b ? 0x3f80 : 0;
    return *this;
  }

  __host__ __device__ inline bfloat16& operator=(int8_t val) {
    x = bfloat16(val).x;
    return *this;
  }

  __host__ __device__ inline bfloat16& operator=(uint8_t val) {
    x = bfloat16(val).x;
    return *this;
  }

  __host__ __device__ inline bfloat16& operator=(int16_t val) {
    x = bfloat16(val).x;
    return *this;
  }

  __host__ __device__ inline bfloat16& operator=(uint16_t val) {
    x = bfloat16(val).x;
    return *this;
  }

  __host__ __device__ inline bfloat16& operator=(int32_t val) {
    x = bfloat16(val).x;
    return *this;
  }

  __host__ __device__ inline bfloat16& operator=(uint32_t val) {
    x = bfloat16(val).x;
    return *this;
  }

  __host__ __device__ inline bfloat16& operator=(int64_t val) {
    x = bfloat16(val).x;
    return *this;
  }

  __host__ __device__ inline bfloat16& operator=(uint64_t val) {
    x = bfloat16(val).x;
    return *this;
  }

  __host__ __device__ inline bfloat16& operator=(float val) {
    x = bfloat16(val).x;
    return *this;
  }

  __host__ __device__ inline bfloat16& operator=(double val) {
    x = bfloat16(val).x;
    return *this;
  }

  // Conversion operators
  __host__ __device__ inline operator float() const {
#ifdef CINN_CUDA_BF16
    return __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&x));
#else
    float val = 0.f;
    uint16_t temp = x;
    std::memcpy(
        reinterpret_cast<char*>(&val) + 2, reinterpret_cast<char*>(&temp), 2);
    return val;
#endif
  }

#ifdef CINN_CUDA_BF16
  __host__ __device__ inline __nv_bfloat16 to_nv_bfloat16() const {
    return *reinterpret_cast<const __nv_bfloat16*>(&x);
  }
#endif

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
#endif  // __cplusplus
};

struct CINN_ALIGN(16) bfloat168 {
  bfloat16 x, y, z, w, v, u, t, s;
};

struct CINN_ALIGN(8) bfloat164 {
  bfloat16 x, y, z, w;
};

struct CINN_ALIGN(4) bfloat162 {
  bfloat16 x, y;
};

__host__ __device__ inline bfloat16 operator+(const bfloat16& a,
                                              const bfloat16& b) {
#if defined(CINN_CUDA_BF16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  return bfloat16(__hadd(a.to_nv_bfloat16(), b.to_nv_bfloat16()));
#else
  return bfloat16(static_cast<float>(a) + static_cast<float>(b));
#endif
}

__host__ __device__ inline bfloat16 operator-(const bfloat16& a,
                                              const bfloat16& b) {
#if defined(CINN_CUDA_BF16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  return bfloat16(__hsub(a.to_nv_bfloat16(), b.to_nv_bfloat16()));
#else
  return bfloat16(static_cast<float>(a) - static_cast<float>(b));
#endif
}

__host__ __device__ inline bfloat16 operator*(const bfloat16& a,
                                              const bfloat16& b) {
#if defined(CINN_CUDA_BF16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  return bfloat16(__hmul(a.to_nv_bfloat16(), b.to_nv_bfloat16()));
#else
  return bfloat16(static_cast<float>(a) * static_cast<float>(b));
#endif
}

__host__ __device__ inline bfloat16 operator/(const bfloat16& a,
                                              const bfloat16& b) {
#if defined(CINN_CUDA_BF16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  return bfloat16(__hdiv(a.to_nv_bfloat16(), b.to_nv_bfloat16()));
#else
  return bfloat16(static_cast<float>(a) / static_cast<float>(b));
#endif
}

__host__ __device__ inline bfloat16 operator-(const bfloat16& a) {
  bfloat16 res;
  res.x = a.x ^ 0x8000;
  return res;
}

__host__ __device__ inline bfloat16& operator+=(bfloat16& a,  // NOLINT
                                                const bfloat16& b) {
  a = a + b;
  return a;
}

__host__ __device__ inline bfloat16& operator-=(bfloat16& a,  // NOLINT
                                                const bfloat16& b) {
  a = a - b;
  return a;
}

__host__ __device__ inline bfloat16& operator*=(bfloat16& a,  // NOLINT
                                                const bfloat16& b) {
  a = a * b;
  return a;
}

__host__ __device__ inline bfloat16& operator/=(bfloat16& a,  // NOLINT
                                                const bfloat16& b) {
  a = a / b;
  return a;
}

__host__ __device__ inline bfloat16 raw_uint16_to_bfloat16(uint16_t a) {
  bfloat16 res;
  res.x = a;
  return res;
}

// Comparison operators
__host__ __device__ inline bool operator==(const bfloat16& a,
                                           const bfloat16& b) {
#if defined(CINN_CUDA_BF16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  return __heq(a.to_nv_bfloat16(), b.to_nv_bfloat16());
#else
  return static_cast<float>(a) == static_cast<float>(b);
#endif
}

__host__ __device__ inline bool operator!=(const bfloat16& a,
                                           const bfloat16& b) {
#if defined(CINN_CUDA_BF16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  return __hne(a.to_nv_bfloat16(), b.to_nv_bfloat16());
#else
  return static_cast<float>(a) != static_cast<float>(b);
#endif
}

__host__ __device__ inline bool operator<(const bfloat16& a,
                                          const bfloat16& b) {
#if defined(CINN_CUDA_BF16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  return __hlt(a.to_nv_bfloat16(), b.to_nv_bfloat16());
#else
  return static_cast<float>(a) < static_cast<float>(b);
#endif
}

__host__ __device__ inline bool operator<=(const bfloat16& a,
                                           const bfloat16& b) {
#if defined(CINN_CUDA_BF16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  return __hle(a.to_nv_bfloat16(), b.to_nv_bfloat16());
#else
  return static_cast<float>(a) <= static_cast<float>(b);
#endif
}

__host__ __device__ inline bool operator>(const bfloat16& a,
                                          const bfloat16& b) {
#if defined(CINN_CUDA_BF16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  return __hgt(a.to_nv_bfloat16(), b.to_nv_bfloat16());
#else
  return static_cast<float>(a) > static_cast<float>(b);
#endif
}

__host__ __device__ inline bool operator>=(const bfloat16& a,
                                           const bfloat16& b) {
#if defined(CINN_CUDA_BF16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  return __hge(a.to_nv_bfloat16(), b.to_nv_bfloat16());
#else
  return static_cast<float>(a) >= static_cast<float>(b);
#endif
}

__host__ __device__ inline bool(isnan)(const bfloat16& a) {
#if defined(CINN_CUDA_BF16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  return __hisnan(a.to_nv_bfloat16());
#else
  return (a.x & 0x7FFF) > 0x7F80;
#endif
}

__host__ __device__ inline bool(isinf)(const bfloat16& a) {
#if defined(CINN_CUDA_BF16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  return __hisinf(a.to_nv_bfloat16());
#else
  return (a.x & 0x7F80) == 0x7F80;
#endif
}

__host__ __device__ inline bool(isfinite)(const bfloat16& a) {
  return !((isnan)(a)) && !((isinf)(a));
}

__host__ __device__ inline bfloat16(abs)(const bfloat16& a) {
#if defined(CINN_CUDA_BF16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  return bfloat16(__habs(a.to_nv_bfloat16()));
#else
  return bfloat16(std::abs(static_cast<float>(a)));
#endif
}

#ifdef __cplusplus
}  // namespace common
}  // namespace cinn
#endif  // __cplusplus

// for runtime calls
#if defined(__cplusplus) && defined(CINN_CUDA_BF16)
__device__ inline cinn::common::bfloat16 __shfl_sync(unsigned mask,
                                                     cinn::common::bfloat16 var,
                                                     int srcLane,
                                                     int width = warpSize) {
  return cinn::common::bfloat16(
      __shfl_sync(mask, var.to_nv_bfloat16(), srcLane, width));
}

__device__ inline cinn::common::bfloat16 __shfl_up_sync(
    unsigned mask,
    cinn::common::bfloat16 var,
    unsigned int delta,
    int width = warpSize) {
  return cinn::common::bfloat16(
      __shfl_up_sync(mask, var.to_nv_bfloat16(), delta, width));
}

__device__ inline cinn::common::bfloat16 __shfl_down_sync(
    unsigned mask,
    cinn::common::bfloat16 var,
    unsigned int delta,
    int width = warpSize) {
  return cinn::common::bfloat16(
      __shfl_down_sync(mask, var.to_nv_bfloat16(), delta, width));
}

__device__ inline cinn::common::bfloat16 __shfl_xor_sync(
    unsigned mask,
    cinn::common::bfloat16 var,
    int laneMask,
    int width = warpSize) {
  return cinn::common::bfloat16(
      __shfl_xor_sync(mask, var.to_nv_bfloat16(), laneMask, width));
}

__host__ __device__ inline cinn::common::bfloat16 max(
    const cinn::common::bfloat16& a, const cinn::common::bfloat16& b) {
  return a > b ? a : b;
}
__host__ __device__ inline cinn::common::bfloat16 min(
    const cinn::common::bfloat16& a, const cinn::common::bfloat16& b) {
  return a < b ? a : b;
}
#endif  // __cplusplus && CINN_CUDA_FP16

#endif  // CINN_COMMON_BFLOAT16_H
