// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

/**
 * This file contains some core runtime concepts, the basic definition is used
 * in C so that it can be deployed in some light-weight devices.
 */
#ifndef CINN_RUNTIME_CINN_RUNTIME_H_
#define CINN_RUNTIME_CINN_RUNTIME_H_
#ifdef __cplusplus
#pragma once
#endif

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
#include <functional>
#include <vector>
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum KernelArgsType {
  unk_args_type = -1,    //! Unknown type
  void_p_args_type = 0,  //! void*
  int64_args_type = 1,   //! int64_t
};

struct KernelArgsNode {
  void* data_ptr{nullptr};
  int64_t symbol_val{-1};

  KernelArgsType args_type{KernelArgsType::unk_args_type};
};

#define CINN_ALWAYS_INLINE __attribute__((always_inline)) inline

//! Code for the primitive types supported in CINN.
typedef enum cinn_type_code_t {
  cinn_type_unk = -1,    //! Unknown type
  cinn_type_int = 0,     //! signed int
  cinn_type_uint = 1,    //! unsigned int
  cinn_type_float = 2,   //! floating point
  cinn_type_handle = 3,  //! void*
  cinn_type_bfloat = 4   //! bfloat16
} cinn_type_code_t;

#ifndef CINN_ATTRIBUTE_ALIGN
#define CINN_ATTRIBUTE_ALIGN(n) __attribute__((aligned(n)))
#endif

/**
 * A runtime tag for type in CINN system.
 */
typedef struct cinn_type_t {
#if __cplusplus >= 201103L
  CINN_ATTRIBUTE_ALIGN(1) cinn_type_code_t code;
#else
  uint8_t code;
#endif

  //! Number of bits.
  uint8_t bits;

  //! Number of elements in a vector, 1 for scalar.
  uint16_t lanes;

  //! Number of '*', e.g. for `float*`, the num_asterisks is 1, `float**` it
  //! is 2.
  uint8_t num_asterisks{0};

#ifdef __cplusplus
  CINN_ALWAYS_INLINE cinn_type_t() : code(cinn_type_int), bits(0), lanes(0) {}
  CINN_ALWAYS_INLINE cinn_type_t(cinn_type_code_t code,
                                 uint8_t bits,
                                 uint16_t lanes = 1,
                                 uint8_t num_asterisks = 0)
      : code(code), bits(bits), lanes(lanes), num_asterisks(num_asterisks) {}
  CINN_ALWAYS_INLINE bool operator==(const cinn_type_t& other) const {
    return code == other.code && bits == other.bits && lanes == other.lanes;
  }
  CINN_ALWAYS_INLINE bool operator!=(const cinn_type_t& other) const {
    return !(*this == other);
  }
  CINN_ALWAYS_INLINE uint16_t bytes() const { return (bits + 7) / 8; }
#endif  // __cplusplus
} cinn_type_t;

//! Some primitive types.
// @{
extern cinn_type_t cinn_unk_t();
extern cinn_type_t cinn_bool_t(int num_asterisks = 0);

extern cinn_type_t cinn_int8_t(int num_asterisks = 0);
extern cinn_type_t cinn_int16_t(int num_asterisks = 0);
extern cinn_type_t cinn_int32_t(int num_asterisks = 0);
extern cinn_type_t cinn_int64_t(int num_asterisks = 0);

extern cinn_type_t cinn_uint8_t(int num_asterisks = 0);
extern cinn_type_t cinn_uint16_t(int num_asterisks = 0);
extern cinn_type_t cinn_uint32_t(int num_asterisks = 0);
extern cinn_type_t cinn_uint64_t(int num_asterisks = 0);

extern cinn_type_t cinn_float32_t(int num_asterisks = 0);
extern cinn_type_t cinn_float64_t(int num_asterisks = 0);
// @}

//! Help to define the size of a dimension, due to polyhedral representation, we
//! no need to record the extend or min(default to 0).
typedef int cinn_dimension_t;

//! Help to tell the kind of the device.
typedef enum cinn_device_kind_t {
  cinn_unk_device = -1,    // Undefined device.
  cinn_x86_device = 0,     // X86 device
  cinn_opencl_device = 1,  // OpenCL device
  cinn_arm_device = 2,     // ARM device
  cinn_nvgpu_device = 3    // NVIDIA GPU device
} cinn_device_kind_t;

//! Help to tell where the buffer locates.
typedef enum cinn_buffer_kind_t {
  cinn_buffer_on_host = 0,        //! buffer on host
  cinn_buffer_on_device = 1 << 1  // ! buffer on device e.g. GPU.
} cinn_buffer_kind_t;

struct cinn_buffer_t;

/**
 * All CINN backends implementation should provide an interface to be used.
 */
struct cinn_device_interface_impl_t;

struct cinn_device_interface_t {
  int (*malloc)(void* context, struct cinn_buffer_t* buf);
  int (*free)(void* context, struct cinn_buffer_t* buf);
  int (*sync)(void* context, struct cinn_buffer_t* buf);
  int (*release)(void* context,
                 const struct cinn_device_interface_t* device_interface);
  int (*copy_to_host)(void* context, struct cinn_buffer_t* buf);
  int (*copy_to_device)(void* context, struct cinn_buffer_t* buf);
  int (*buffer_copy)(void* context,
                     struct cinn_buffer_t* src,
                     struct cinn_buffer_t* dst);
  struct cinn_device_interface_impl_t* impl;
};

/**
 * Release all data associated with the given interface.
 */
extern int cinn_device_release(
    void* context, const struct cinn_device_interface_t* device_interface);

/*
 * Copy image data from device to host memory.
 */
extern int cinn_buffer_copy_to_host(void* context, struct cinn_buffer_t* buf);

//! Copy data from host to device memory.
extern int cinn_buffer_copy_to_device(void* context, struct cinn_buffer_t* buf);

//! Copy data from one buffer to another.
extern int cinn_buffer_copy(void* context,
                            struct cinn_buffer_t* src,
                            struct cinn_buffer_t* dst);

//! Wait for current device operations to complete.
extern int cinn_device_sync(void* context, struct cinn_buffer_t* buf);

//! Allocate device memory.
extern int cinn_buffer_malloc(void* context, struct cinn_buffer_t* buf);

//! Free device memory.
extern int cinn_buffer_free(void* context, struct cinn_buffer_t* buf);

//! Get the memory address in buffer.
extern void* cinn_buffer_get_data_handle(struct cinn_buffer_t* buf);
extern void* cinn_buffer_get_data_const_handle(const struct cinn_buffer_t* buf);

//! Create a new default cinn_buffer.
extern cinn_buffer_t* cinn_buffer_new_default(int target,
                                              uint64_t memory_size,
                                              int align = 32);

//! The raw representation of a buffer,used in the generated code/lib.
#define CINN_BUFFER_MAX_DIMS 8
typedef struct cinn_buffer_t {
  //! Tell which kind of device this buffer locates.
  cinn_device_kind_t device;

  //! The interface used to operate on device.
  const struct cinn_device_interface_t* device_interface;

  //! A pointer to the memory in host.
  uint8_t* memory;

  //! Extra flags.
  uint64_t flag;

  //! Data type.
  cinn_type_t type;

  //! Number of dimensions.
  int32_t dimensions;
  cinn_dimension_t dims[CINN_BUFFER_MAX_DIMS];

  //! Allocate and deallocate lazily, default true.
  char lazy;

  //! The actual memory size(in bytes).
  uint64_t memory_size;

  uint16_t align;

#ifdef __cplusplus
  cinn_buffer_t()
      : device(cinn_unk_device),
        device_interface(NULL),
        memory(NULL),
        flag(0UL),
        type(cinn_type_t()),
        dimensions(0),
        lazy(true),
        memory_size(0),
        align(0),
        external_malloc(NULL),
        external_free(NULL) {}

  static struct cinn_buffer_t* new_(cinn_device_kind_t device,
                                    cinn_type_t type,
                                    const std::vector<int>& shape,
                                    int align = 0);
  static void delete_(struct cinn_buffer_t* x) { delete x; }

  ~cinn_buffer_t() {
    if (external_malloc != NULL) {
      delete external_malloc;
    }
    if (external_free != NULL) {
      delete external_free;
    }
  }

  // NOTE the buffer should be resized first.
  static void alloc(struct cinn_buffer_t*);

  //! Set the shape of the buffer. NOTE this just record the shape, not allocate
  //! the memory.
  CINN_ALWAYS_INLINE void resize(const cinn_dimension_t* dims, int dimensions) {
    this->dimensions = dimensions;
    memcpy(this->dims, dims, dimensions * sizeof(cinn_dimension_t));
  }

  CINN_ALWAYS_INLINE uint64_t num_elements() const {
    uint64_t res = 1;
    for (int i = 0; i < dimensions; i++) {
      res *= dims[i];
    }
    return res;
  }

  CINN_ALWAYS_INLINE bool on_host() const {
    return get_flag(cinn_buffer_on_host);
  }
  CINN_ALWAYS_INLINE bool on_device() const {
    return get_flag(cinn_buffer_on_device);
  }
  CINN_ALWAYS_INLINE void set_on_host(bool x = true) {
    set_flag(cinn_buffer_on_host, x);
  }
  CINN_ALWAYS_INLINE void set_on_device(bool x = true) {
    set_flag(cinn_buffer_on_device, x);
  }

  CINN_ALWAYS_INLINE int device_sync(void* ctx = NULL) {
    if (device_interface && device_interface->sync) {
      return device_interface->sync(ctx, this);
    }
    return 0;
  }

  CINN_ALWAYS_INLINE uint8_t* begin() const { return 0; }
  CINN_ALWAYS_INLINE uint8_t* end() const {
    return memory + num_elements() * type.bytes();
  }

  CINN_ALWAYS_INLINE bool get_flag(cinn_buffer_kind_t flag) const {
    return (this->flag & flag) != 0;
  }
  CINN_ALWAYS_INLINE void set_flag(cinn_buffer_kind_t flag, bool value) {
    if (value)
      this->flag |= flag;
    else
      this->flag &= ~flag;
  }

  // The callback to control memory alloc. It is useful in Paddle-CINN
  // where the memory is managed out of CINN.
  std::function<int(void*, struct cinn_buffer_t*)>* external_malloc;

  // The callback to control memory free. It is useful in Paddle-CINN
  // where the memory is managed out of CINN.
  std::function<int(void*, struct cinn_buffer_t*)>* external_free;

#endif  // __cplusplus
} cinn_buffer_t;

#ifdef __cplusplus
//! Create a new cinn_buffer.
cinn_buffer_t* cinn_buffer_new(cinn_device_kind_t device,
                               cinn_type_t type,
                               const std::vector<int>& shape,
                               int align = 0);

struct cinn_device_interface_impl_t {
  int (*malloc)(void* context, struct cinn_buffer_t* buf);
  int (*free)(void* context, struct cinn_buffer_t* buf);
  int (*sync)(void* context, struct cinn_buffer_t* buf);
  int (*release)(void* context);
  int (*copy_to_host)(void* context, struct cinn_buffer_t* buf);
  int (*copy_to_device)(void* context, struct cinn_buffer_t* buf);
  int (*buffer_copy)(void* context,
                     struct cinn_buffer_t* src,
                     struct cinn_buffer_t* dst);
};

// The device implementations
extern struct cinn_device_interface_t* cinn_x86_device_interface();

inline float cinn_buffer_load_float32(struct cinn_buffer_t* buf,
                                      uint32_t index) {
  return ((float*)buf->memory)[index];  // NOLINT
}
inline double cinn_buffer_load_float64(struct cinn_buffer_t* buf,
                                       uint32_t index) {
  return ((double*)buf->memory)[index];  // NOLINT
}
#endif  // __cplusplus

#ifdef __cplusplus
extern "C" {
#endif

CINN_ALWAYS_INLINE void* cinn_buffer_slice(struct cinn_buffer_t* buf,
                                           uint32_t offset);

#ifdef __cplusplus
}
#endif

static inline int32_t cinn_min(int32_t a, int32_t b) { return a < b ? a : b; }
static inline int32_t cinn_max(int32_t a, int32_t b) { return a > b ? a : b; }

#ifdef __cplusplus
}  // extern "C"
#endif

#ifndef CINN_RUNTIME_NOT_IMPLEMENTED
#define CINN_RUNTIME_NOT_IMPLEMENTED     \
  do {                                   \
    fprintf(stderr, "Not Implemented!"); \
    abort();                             \
  } while (false);
#endif

#define ASSERT_NOT_NULL(v__)          \
  if (!v__) {                         \
    fprintf(stderr, #v__ " is null"); \
    return -1;                        \
  }
#define CINN_LOG(fmt, ...)      \
  do {                          \
    fprintf(stderr,             \
            "%s:%d:%s(): " fmt, \
            __FILE__,           \
            __LINE__,           \
            __func__,           \
            __VA_ARGS__);       \
  } while (0)

#define CINN_CHECK(cond)                \
  if (!(cond)) {                        \
    CINN_LOG("check %s failed", #cond); \
    abort();                            \
  }
#define CINN_CHECK_LT(a, b)                                \
  if (!(a < b)) {                                          \
    cinn_print_debug_string("check %d > %d failed", a, b); \
    abort();                                               \
  }
#define CINN_CHECKP(cond, ...) \
  if (!(cond)) {               \
    CINN_LOG(__VA_ARGS__);     \
    abort();                   \
  }
#define CINN_CHECK_EQ(a, b)                                        \
  {                                                                \
    if ((a) != (b)) {                                              \
      CINN_LOG("check %s == %s failed, %d != %d", #a, #b, (a), b); \
      abort();                                                     \
    }                                                              \
  }                                                                \
  while (false)                                                    \
    ;  // NOLINT

#endif  // CINN_RUNTIME_CINN_RUNTIME_H_

union cinn_value_t {
  int64_t v_int64;
  double v_float64;
  void* v_handle;
  char* v_str;
};

struct cinn_pod_value_t {
#ifdef __cplusplus
  // @{ PodValue

  cinn_pod_value_t() = default;

  cinn_pod_value_t(cinn_value_t value, int type_code);
  explicit cinn_pod_value_t(cinn_buffer_t* value);
  explicit cinn_pod_value_t(bool value);

  explicit cinn_pod_value_t(int8_t value);
  explicit cinn_pod_value_t(int16_t value);
  explicit cinn_pod_value_t(int32_t value);
  explicit cinn_pod_value_t(int64_t value);

  explicit cinn_pod_value_t(uint8_t value);
  explicit cinn_pod_value_t(uint16_t value);
  explicit cinn_pod_value_t(uint32_t value);
  explicit cinn_pod_value_t(uint64_t value);

  explicit cinn_pod_value_t(float value);
  explicit cinn_pod_value_t(double value);

  explicit cinn_pod_value_t(void* value);
  explicit cinn_pod_value_t(const char* value);

  //! The value getters for the supported types.
  //@{
  operator double() const;
  operator float() const;

  operator bool() const;

  operator int8_t() const;
  operator int16_t() const;
  operator int32_t() const;
  operator int64_t() const;

  operator uint8_t() const;
  operator uint16_t() const;
  operator uint32_t() const;
  operator uint64_t() const;

  operator void*() const;
  operator cinn_buffer_t*() const;
  operator char*() const;
  //@}

  int type_code() const { return type_code_; }

  void set_type_code(int x) { type_code_ = x; }
  void set_value(union cinn_value_t x) { value_ = x; }

 protected:
  // @}
#endif  // __cplusplus
  int type_code_;
  union cinn_value_t value_;
};

#ifdef __cplusplus
template <typename T>
constexpr int cinn_type_code();

//! Implement the type_code for all the supported types.
// @{
#define __m(T, code__)                \
  template <>                         \
  constexpr int cinn_type_code<T>() { \
    return code__;                    \
  }
__m(int32_t, 0);
__m(int64_t, 1);
__m(float, 2);
__m(double, 3);
__m(void*, 4);
__m(char*, 5);
__m(char const*, 6);
__m(cinn_buffer_t*, 7);
__m(int8_t, 8);
__m(bool, 9);
__m(int16_t, 11);
__m(uint8_t, 12);
__m(uint16_t, 13);
__m(uint32_t, 14);
__m(uint64_t, 15);
#undef __m
//@}
#endif  // __cplusplus

typedef struct cinn_pod_value_t cinn_pod_value_t;

// the LoweredFunc pointer type for JIT usage.
typedef void (*lower_func_ptr_t)(void*, int32_t);
typedef void (*lower_func_ptr_g)(void*, int32_t, void*);

#ifdef __cplusplus
extern "C" {
#endif
//! cinn_pod_value to specific types.
// @{
float cinn_pod_value_to_float(cinn_pod_value_t* value);
double cinn_pod_value_to_double(cinn_pod_value_t* value);

int64_t cinn_pod_value_to_int64(cinn_pod_value_t* value);
int32_t cinn_pod_value_to_int32(cinn_pod_value_t* value);
int16_t cinn_pod_value_to_int16(cinn_pod_value_t* value);
int8_t cinn_pod_value_to_int8(cinn_pod_value_t* value);

uint64_t cinn_pod_value_to_uint64(cinn_pod_value_t* value);
uint32_t cinn_pod_value_to_uint32(cinn_pod_value_t* value);
uint16_t cinn_pod_value_to_uint16(cinn_pod_value_t* value);
uint8_t cinn_pod_value_to_uint8(cinn_pod_value_t* value);

bool cinn_pod_value_to_bool(cinn_pod_value_t* value);

void* cinn_pod_value_to_void_p(cinn_pod_value_t* value);
int32_t* cinn_pod_value_to_int32_p(cinn_pod_value_t* value);
cinn_buffer_t* cinn_pod_value_to_buffer_p(cinn_pod_value_t* value);
// @}

//! other specific types to cinn_pod_value
// @{
void float_to_cinn_pod_value(float v, cinn_pod_value_t* out);

void double_to_cinn_pod_value(double v, cinn_pod_value_t* out);

void bool_to_cinn_pod_value(bool v, cinn_pod_value_t* out);

void int8_to_cinn_pod_value(int8_t v, cinn_pod_value_t* out);
void int16_to_cinn_pod_value(int16_t v, cinn_pod_value_t* out);
void int32_to_cinn_pod_value(int32_t v, cinn_pod_value_t* out);
void int64_to_cinn_pod_value(int64_t v, cinn_pod_value_t* out);

void uint8_to_cinn_pod_value(uint8_t v, cinn_pod_value_t* out);
void uint16_to_cinn_pod_value(uint16_t v, cinn_pod_value_t* out);
void uint32_to_cinn_pod_value(uint32_t v, cinn_pod_value_t* out);
void uint64_to_cinn_pod_value(uint64_t v, cinn_pod_value_t* out);

void handle_to_cinn_pod_value(void* v, cinn_pod_value_t* out);
void buffer_p_to_cinn_pod_value(const struct cinn_buffer_t* v,
                                cinn_pod_value_t* out);
// @}

void cinn_print_debug_string(const char* s, ...);

void cinn_print_debug_args(cinn_pod_value_t* args, int count);

/**
 * Construct a Args for LoweredFunc with a list of `cinn_pod_value_t*`
 * @param arr An array of `cinn_pod_value_t`
 * @param count Count of elements in the arg list.
 * @param ... variadic args of `cinn_pod_value_t*`
 */
void cinn_args_construct(cinn_pod_value_t* arr, int count, ...);

#ifdef __cplusplus
}  // extern "C"
#endif

#ifdef __cplusplus
template <typename T>
cinn_type_t cinn_type_of();

#endif  // __cplusplus
