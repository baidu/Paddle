/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
   Copyright 2019 The TensorFlow Authors. All Rights Reserved.

This file is inspired by

    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/platform/ctstring_internal.h

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

#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "paddle/common/macros.h"

#if (defined(__NVCC__) || defined(__HIPCC__))
#define HOSTDEVICE __host__ __device__
#define DEVICE __device__
#define HOST __host__
#else
#define HOSTDEVICE
#define DEVICE
#define HOST
#endif

#if (defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__) && \
     __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__) ||                  \
    defined(_WIN32)
#define PD_PSTRING_LITTLE_ENDIAN 1
#elif defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__) && \
    __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#define PD_PSTRING_LITTLE_ENDIAN 0
#else
#error "Unable to detect endianness."
#endif

#if defined(__clang__) || \
    (defined(__GNUC__) && \
     ((__GNUC__ == 4 && __GNUC_MINOR__ >= 8) || __GNUC__ >= 5))
HOSTDEVICE static inline uint32_t swap32(uint32_t host_int) {
  return __builtin_bswap32(host_int);
}

#elif defined(_MSC_VER)
HOSTDEVICE static inline uint32_t swap32(uint32_t host_int) {
  return _byteswap_ulong(host_int);
}

#elif defined(__APPLE__)
HOSTDEVICE static inline uint32_t swap32(uint32_t host_int) {
  return OSSwapInt32(host_int);
}

#else
HOSTDEVICE static inline uint32_t swap32(uint32_t host_int) {
#if defined(__GLIBC__)
  return bswap_32(host_int);
#else   // defined(__GLIBC__)
  return (((host_int & uint32_t{0xFF}) << 24) |
          ((host_int & uint32_t{0xFF00}) << 8) |
          ((host_int & uint32_t{0xFF0000}) >> 8) |
          ((host_int & uint32_t{0xFF000000}) >> 24));
#endif  // defined(__GLIBC__)
}
#endif

#if PD_PSTRING_LITTLE_ENDIAN || (defined(__NVCC__) || defined(__HIPCC__))
#define PD_le32toh(x) x
#else  // PD_PSTRING_LITTLE_ENDIAN
#define PD_le32toh(x) swap32(x)
#endif  // PD_PSTRING_LARGE_ENDIAN

HOSTDEVICE static inline size_t PD_align16(size_t i) {
  return (i + 0xF) & ~0xF;
}

HOSTDEVICE static inline size_t PD_max(size_t a, size_t b) {
  return a > b ? a : b;
}
HOSTDEVICE static inline size_t PD_min(size_t a, size_t b) {
  return a < b ? a : b;
}

typedef enum PD_PString_Type {  // NOLINT
  PD_PSTR_SMALL = 0x00,
  PD_PSTR_LARGE = 0x01,
  PD_PSTR_OFFSET = 0x02,
  PD_PSTR_VIEW = 0x03,
  PD_PSTR_TYPE_MASK = 0x03
} PD_PString_Type;

typedef struct PD_PString_Large {  // NOLINT
  size_t size;
  size_t cap;
  char *ptr;
} PD_PString_Large;

typedef struct PD_PString_Offset {  // NOLINT
  uint32_t size;
  uint32_t offset;
  uint32_t count;
} PD_PString_Offset;

typedef struct PD_PString_View {  // NOLINT
  size_t size;
  const char *ptr;
} PD_PString_View;

typedef struct PD_PString_Raw {  // NOLINT
  uint8_t raw[24];
} PD_PString_Raw;

typedef union PD_PString_Union {  // NOLINT
  PD_PString_Large large;
  PD_PString_Offset offset;
  PD_PString_View view;
  PD_PString_Raw raw;
} PD_PString_Union;

enum {
  PD_PString_SmallCapacity =
      (sizeof(PD_PString_Union) - sizeof(/* null delim */ char) -
       sizeof(/* uint8_t size */ uint8_t)),
};

typedef struct PD_PString_Small {  // NOLINT
  uint8_t size;
  char str[PD_PString_SmallCapacity + sizeof(/* null delim */ char)];
} PD_PString_Small;

typedef struct PD_PString {  // NOLINT
  union {
    PD_PString_Small small;
    PD_PString_Large large;
    PD_PString_Offset offset;
    PD_PString_View view;
    PD_PString_Raw raw;
  } u;
} PD_PString;

HOSTDEVICE static inline PD_PString_Type PD_PString_GetType(
    const PD_PString *str) {
  return (PD_PString_Type)(str->u.raw.raw[0] & PD_PSTR_TYPE_MASK);  // NOLINT
}

HOSTDEVICE static inline size_t PD_PString_ToActualSizeT(size_t size) {
#if PD_PSTRING_LITTLE_ENDIAN
  return size >> 2;
#else   // PD_PSTRING_LITTLE_ENDIAN
  // 0xFF000000 or 0xFF00000000000000 depending on platform
  static const size_t mask = ~((~(size_t)0) >> 8);  // NOLINT

  return (((mask << 2) & size) >> 2) | (~mask & size);
#endif  // PD_PSTRING_LITTLE_ENDIAN
}

HOSTDEVICE static inline size_t PD_PString_ToInternalSizeT(
    size_t size, PD_PString_Type type) {
#if PD_PSTRING_LITTLE_ENDIAN
  return (size << 2) | type;
#else   // PD_PSTRING_LITTLE_ENDIAN
  // 0xFF000000 or 0xFF00000000000000 depending on platform
  static const size_t mask = ~((~(size_t)0) >> 8);  // NOLINT

  return (mask & (size << 2)) | (~mask & size) |
         ((size_t)type << ((sizeof(size_t) - 1) * 8));  // NOLINT
#endif  // PD_PSTRING_LITTLE_ENDIAN
}

/*
 * Need to implement in other source file.
 */
HOSTDEVICE static inline void PD_Free(void *ptr, size_t size UNUSED) {
  free(ptr);
}

HOSTDEVICE static inline void *PD_Memset(void *src, int ch, size_t size) {
  char *dst = (char *)src;  // NOLINT
  for (size_t i = 0; i < size; ++i) {
    dst[i] = ch;
  }
  return dst;
}

HOSTDEVICE static inline void *PD_Memcpy(void *dst,
                                         const void *src,
                                         size_t size) {
  for (size_t i = 0; i < size; ++i) {
    ((char *)dst)[i] = ((const char *)src)[i];  // NOLINT
  }
  return dst;
}

HOSTDEVICE static inline void *PD_Malloc(size_t size) { return malloc(size); }

HOSTDEVICE static inline void *PD_Realloc(void *ptr,
                                          size_t old_size UNUSED,
                                          size_t new_size) {
#if (defined(__NVCC__) || defined(__HIPCC__))
  if (old_size >= new_size) {
    return ptr;
  }
  void *new_ptr = malloc(new_size);
  PD_Memcpy(new_ptr, ptr, old_size);
  free(ptr);
  return new_ptr;
#else
  return realloc(ptr, new_size);
#endif
}

HOSTDEVICE static inline int PD_Memcmp(const void *s1,
                                       const void *s2,
                                       size_t size) {
  const uint8_t *lstr = (const uint8_t *)(s1);  // NOLINT
  const uint8_t *rstr = (const uint8_t *)(s2);  // NOLINT
  for (size_t i = 0; i < size; ++i) {
    if (lstr[i] != rstr[i]) {
      return (lstr[i] - rstr[i]);
    }
  }
  return 0;
}

HOSTDEVICE static inline void *PD_Memmove(void *dest,
                                          const void *src,
                                          size_t size) {
  const uint8_t *from = (const uint8_t *)(src);  // NOLINT
  uint8_t *to = (uint8_t *)(dest);               // NOLINT
  if (from == to || size == 0) {
    return dest;
  }

  if (to > from && (to - from < static_cast<int>(size))) {
    for (int i = size - 1; i >= 0; i--) {
      to[i] = from[i];
    }
    return dest;
  }
  if (from > to && (from - to < static_cast<int>(size))) {
    for (size_t i = 0; i < size; i++) {
      to[i] = from[i];
    }
    return dest;
  }
  dest = PD_Memcpy(dest, src, size);
  return dest;
}

HOSTDEVICE static inline void PD_PString_Init(PD_PString *str) {
  PD_Memset(str->u.raw.raw, 0, sizeof(PD_PString_Raw));
}

HOSTDEVICE static inline void PD_PString_Dealloc(PD_PString *str) {
  if (PD_PString_GetType(str) == PD_PSTR_LARGE &&
      str->u.large.ptr != NULL) {  // NOLINT
    PD_Free(str->u.large.ptr, str->u.large.cap + 1);
    PD_PString_Init(str);
  }
}

HOSTDEVICE static inline size_t PD_PString_GetSize(const PD_PString *str) {
  switch (PD_PString_GetType(str)) {
    case PD_PSTR_SMALL:
      return str->u.small.size >> 2;
    case PD_PSTR_LARGE:
      return PD_PString_ToActualSizeT(str->u.large.size);
    case PD_PSTR_OFFSET:
      return PD_le32toh(str->u.offset.size) >> 2;
    case PD_PSTR_VIEW:
      return PD_PString_ToActualSizeT(str->u.view.size);
    default:
      return 0;  // Unreachable.
  }
}

HOSTDEVICE static inline size_t PD_PString_GetCapacity(const PD_PString *str) {
  switch (PD_PString_GetType(str)) {
    case PD_PSTR_SMALL:
      return PD_PString_SmallCapacity;
    case PD_PSTR_LARGE:
      return str->u.large.cap;
    case PD_PSTR_OFFSET:
    case PD_PSTR_VIEW:
    default:
      return 0;
  }
}

HOSTDEVICE static inline const char *PD_PString_GetDataPointer(
    const PD_PString *str) {
  switch (PD_PString_GetType(str)) {
    case PD_PSTR_SMALL:
      return str->u.small.str;
    case PD_PSTR_LARGE:
      return str->u.large.ptr;
    case PD_PSTR_OFFSET:
      return (const char *)str + str->u.offset.offset;  // NOLINT
    case PD_PSTR_VIEW:
      return str->u.view.ptr;
    default:
      // Unreachable.
      return NULL;  // NOLINT
  }
}

HOSTDEVICE static inline char *PD_PString_ResizeUninitialized(PD_PString *str,
                                                              size_t new_size) {
  size_t curr_size = PD_PString_GetSize(str);
  size_t copy_size = PD_min(new_size, curr_size);

  PD_PString_Type curr_type = PD_PString_GetType(str);
  const char *curr_ptr = PD_PString_GetDataPointer(str);

  // Case: SMALL/LARGE/VIEW/OFFSET -> SMALL
  if (new_size <= PD_PString_SmallCapacity) {
    str->u.small.size = (uint8_t)((new_size << 2) | PD_PSTR_SMALL);  // NOLINT
    str->u.small.str[new_size] = '\0';

    if (curr_type != PD_PSTR_SMALL && copy_size) {
      PD_Memcpy(str->u.small.str, curr_ptr, copy_size);
    }

    if (curr_type == PD_PSTR_LARGE) {
      PD_Free((void *)curr_ptr, str->u.large.cap + 1);  // NOLINT
    }

    return str->u.small.str;
  }

  // Case: SMALL/LARGE/VIEW/OFFSET -> LARGE
  size_t new_cap;
  size_t curr_cap = PD_PString_GetCapacity(str);

  if (new_size < curr_size && new_size < curr_cap / 2) {
    new_cap = PD_align16(curr_cap / 2 + 1) - 1;
  } else if (new_size > curr_cap) {
    new_cap = PD_align16(new_size + 1) - 1;
  } else {
    new_cap = curr_cap;
  }

  char *new_ptr;
  if (new_cap == curr_cap) {
    new_ptr = str->u.large.ptr;
  } else if (curr_type == PD_PSTR_LARGE) {
    new_ptr = (char *)PD_Realloc(  // NOLINT
        str->u.large.ptr,
        curr_cap + 1,
        new_cap + 1);
  } else {
    new_ptr = (char *)PD_Malloc(new_cap + 1);  // NOLINT
    if (copy_size) {
      PD_Memcpy(new_ptr, curr_ptr, copy_size);
    }
  }

  str->u.large.size = PD_PString_ToInternalSizeT(new_size, PD_PSTR_LARGE);
  str->u.large.ptr = new_ptr;
  str->u.large.ptr[new_size] = '\0';
  str->u.large.cap = new_cap;

  return str->u.large.ptr;
}

HOSTDEVICE static inline char *PD_PString_GetMutableDataPointer(
    PD_PString *str) {
  switch (PD_PString_GetType(str)) {
    case PD_PSTR_SMALL:
      return str->u.small.str;
    case PD_PSTR_OFFSET:
    case PD_PSTR_VIEW:
      // Convert OFFSET/VIEW to SMALL/LARGE
      PD_PString_ResizeUninitialized(str, PD_PString_GetSize(str));
      return (PD_PString_GetType(str) == PD_PSTR_SMALL) ? str->u.small.str
                                                        : str->u.large.ptr;
    case PD_PSTR_LARGE:
      return str->u.large.ptr;
    default:
      // Unreachable.
      return NULL;  // NOLINT
  }
}

HOSTDEVICE static inline void PD_PString_Reserve(PD_PString *str,
                                                 size_t new_cap) {
  PD_PString_Type curr_type = PD_PString_GetType(str);

  if (new_cap <= PD_PString_SmallCapacity) {
    // We do nothing, we let Resize/GetMutableDataPointer handle the
    // conversion to SMALL from VIEW/OFFSET when the need arises.
    // In the degenerate case, where new_cap <= PD_PString_SmallCapacity,
    // curr_size > PD_PString_SmallCapacity, and the type is VIEW/OFFSET, we
    // defer the malloc to Resize/GetMutableDataPointer.
    return;
  }

  if (curr_type == PD_PSTR_LARGE && new_cap <= str->u.large.cap) {
    // We handle reduced cap in resize.
    return;
  }

  // Case: VIEW/OFFSET -> LARGE or grow an existing LARGE type
  size_t curr_size = PD_PString_GetSize(str);
  const char *curr_ptr = PD_PString_GetDataPointer(str);

  // Since VIEW and OFFSET types are read-only, their capacity is effectively 0.
  // So we make sure we have enough room in the VIEW and OFFSET cases.
  new_cap = PD_align16(PD_max(new_cap, curr_size) + 1) - 1;
  size_t curr_cap = PD_PString_GetCapacity(str);

  if (curr_type == PD_PSTR_LARGE) {
    str->u.large.ptr = (char *)PD_Realloc(  // NOLINT
        str->u.large.ptr,
        curr_cap + 1,
        new_cap + 1);
  } else {
    // Convert to Large
    char *new_ptr = (char *)PD_Malloc(new_cap + 1);  // NOLINT
    PD_Memcpy(new_ptr, curr_ptr, curr_size);

    str->u.large.size = PD_PString_ToInternalSizeT(curr_size, PD_PSTR_LARGE);
    str->u.large.ptr = new_ptr;
    str->u.large.ptr[curr_size] = '\0';
  }

  str->u.large.cap = new_cap;
}

HOSTDEVICE static inline void PD_PString_ReserveAmortized(PD_PString *str,
                                                          size_t new_cap) {
  const size_t curr_cap = PD_PString_GetCapacity(str);
  if (new_cap > curr_cap) {
    PD_PString_Reserve(str, new_cap > 2 * curr_cap ? new_cap : 2 * curr_cap);
  }
}

HOSTDEVICE static inline char *PD_PString_Resize(PD_PString *str,
                                                 size_t new_size,
                                                 char c) {
  size_t curr_size = PD_PString_GetSize(str);
  char *cstr = PD_PString_ResizeUninitialized(str, new_size);

  if (new_size > curr_size) {
    PD_Memset(cstr + curr_size, c, new_size - curr_size);
  }

  return cstr;
}

HOSTDEVICE static inline void PD_PString_AssignView(PD_PString *dst,
                                                    const char *src,
                                                    size_t size) {
  PD_PString_Dealloc(dst);

  dst->u.view.size = PD_PString_ToInternalSizeT(size, PD_PSTR_VIEW);
  dst->u.view.ptr = src;
}

HOSTDEVICE static inline void PD_PString_AppendN(PD_PString *dst,
                                                 const char *src,
                                                 size_t src_size) {
  if (!src_size) return;

  size_t dst_size = PD_PString_GetSize(dst);

  // For append use cases, we want to ensure amortized growth.
  PD_PString_ReserveAmortized(dst, dst_size + src_size);
  char *dst_c = PD_PString_ResizeUninitialized(dst, dst_size + src_size);

  PD_Memcpy(dst_c + dst_size, src, src_size);
}

HOSTDEVICE static inline void PD_PString_Append(PD_PString *dst,
                                                const PD_PString *src) {
  const char *src_c = PD_PString_GetDataPointer(src);
  size_t size = PD_PString_GetSize(src);

  PD_PString_AppendN(dst, src_c, size);
}

HOSTDEVICE static inline void PD_PString_Copy(PD_PString *dst,
                                              const char *src,
                                              size_t size) {
  char *dst_c = PD_PString_ResizeUninitialized(dst, size);

  if (size) PD_Memcpy(dst_c, src, size);
}

HOSTDEVICE static inline void PD_PString_Assign(PD_PString *dst,
                                                const PD_PString *src) {
  if (dst == src) return;

  PD_PString_Dealloc(dst);

  switch (PD_PString_GetType(src)) {
    case PD_PSTR_SMALL:
    case PD_PSTR_VIEW:
      *dst = *src;
      return;
    case PD_PSTR_LARGE: {
      const char *src_c = PD_PString_GetDataPointer(src);
      size_t size = PD_PString_GetSize(src);

      PD_PString_Copy(dst, src_c, size);
    }
      return;
    default:
      return;  // Unreachable.
  }
}

HOSTDEVICE static inline void PD_PString_Move(PD_PString *dst,
                                              PD_PString *src) {
  if (dst == src) return;

  PD_PString_Dealloc(dst);

  switch (PD_PString_GetType(src)) {
    case PD_PSTR_SMALL:
    case PD_PSTR_VIEW:
      *dst = *src;
      return;
    case PD_PSTR_LARGE:
      *dst = *src;
      PD_PString_Init(src);
      return;
    case PD_PSTR_OFFSET: {
      const char *src_c = PD_PString_GetDataPointer(src);
      size_t size = PD_PString_GetSize(src);

      PD_PString_AssignView(dst, src_c, size);
    }
      return;
    default:
      return;  // Unreachable.
  }
}
