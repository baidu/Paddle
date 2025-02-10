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

#include "paddle/cinn/runtime/cinn_runtime.h"

#include <stdarg.h>
#include <stdio.h>

#include <cmath>

extern "C" {

int cinn_buffer_malloc(void* context, struct cinn_buffer_t* buf) {
  // ASSERT_NOT_NULL(context)
  ASSERT_NOT_NULL(buf)
  ASSERT_NOT_NULL(buf->device_interface)
  return buf->device_interface->impl->malloc(context, buf);
}

int cinn_buffer_free(void* context, struct cinn_buffer_t* buf) {
  // ASSERT_NOT_NULL(context)
  ASSERT_NOT_NULL(buf)
  // If buffer is lazy, then we will not free this buffer, that will greatly
  // improve performance.
  if (buf->lazy) return 0;
  return buf->device_interface->impl->free(context, buf);
}

void* cinn_buffer_slice(struct cinn_buffer_t* buf, uint32_t offset) {
  CINN_CHECK(buf);
  uint64_t offset_byte = offset * buf->type.bytes();
  CINN_CHECK_LT(offset_byte, buf->memory_size);
  return buf->memory + offset_byte;
}

int cinn_device_sync(void* context, struct cinn_buffer_t* buf) {
  ASSERT_NOT_NULL(buf)
  ASSERT_NOT_NULL(buf->device_interface)
  // ASSERT_NOT_NULL(context)
  buf->device_interface->impl->sync(context, buf);
  return 0;
}

int cinn_device_release(
    void* context, const struct cinn_device_interface_t* device_interface) {
  // ASSERT_NOT_NULL(context)
  ASSERT_NOT_NULL(device_interface)
  CINN_RUNTIME_NOT_IMPLEMENTED
}

int cinn_buffer_copy_to_host(void* context, struct cinn_buffer_t* buf) {
  // ASSERT_NOT_NULL(context)
  ASSERT_NOT_NULL(buf)
  ASSERT_NOT_NULL(buf->device_interface)
  return buf->device_interface->impl->copy_to_host(context, buf);
}

int cinn_buffer_copy_to_device(void* context, struct cinn_buffer_t* buf) {
  // ASSERT_NOT_NULL(context)
  ASSERT_NOT_NULL(buf)
  ASSERT_NOT_NULL(buf->device_interface)
  return buf->device_interface->impl->copy_to_device(context, buf);
}
int cinn_buffer_copy(void* context,
                     struct cinn_buffer_t* src,
                     struct cinn_buffer_t* dst) {
  // ASSERT_NOT_NULL(context);
  ASSERT_NOT_NULL(src);
  ASSERT_NOT_NULL(dst);
  return dst->device_interface->buffer_copy(context, src, dst);
}

void* cinn_buffer_get_data_handle(struct cinn_buffer_t* buf) {
  CINN_CHECKP(buf, "%s", "buffer is null");
  return buf->memory;
}

void* cinn_buffer_get_data_const_handle(const struct cinn_buffer_t* buf) {
  CINN_CHECKP(buf, "%s", "buffer is null");
  return buf->memory;
}

cinn_buffer_t* cinn_buffer_new_default(int target,
                                       uint64_t memory_size,
                                       int align) {
  struct cinn_buffer_t* buf =
      (struct cinn_buffer_t*)malloc(sizeof(struct cinn_buffer_t));
  buf->type = cinn_float32_t();
  buf->device = (cinn_device_kind_t)target;
  buf->memory = nullptr;
  buf->memory_size = memory_size;
  buf->align = align;
  buf->lazy = true;
#ifdef __cplusplus
  buf->external_malloc = nullptr;
  buf->external_free = nullptr;
#endif  // __cplusplus
  // NOTE set device_interface for each buffer.
  switch (buf->device) {
    case cinn_x86_device:
      buf->device_interface = cinn_x86_device_interface();
      break;
    case cinn_unk_device:
      fprintf(stderr, "Device type of buffer should be set, found Unk");
      abort();
      break;
    default:
      fprintf(stderr, "Not supported device type");
      abort();
  }
  cinn_buffer_malloc(reinterpret_cast<void*>(0), buf);
  return buf;
}

cinn_type_t cinn_unk_t() { return cinn_type_t(cinn_type_unk, 0); }
cinn_type_t cinn_bool_t(int num_asterisks) {
  return cinn_type_t(cinn_type_int, 1, num_asterisks);
}

cinn_type_t cinn_int8_t(int num_asterisks) {
  return cinn_type_t(cinn_type_int, 8, num_asterisks);
}
cinn_type_t cinn_int16_t(int num_asterisks) {
  return cinn_type_t(cinn_type_int, 16, num_asterisks);
}
cinn_type_t cinn_int32_t(int num_asterisks) {
  return cinn_type_t(cinn_type_int, 32, num_asterisks);
}
cinn_type_t cinn_int64_t(int num_asterisks) {
  return cinn_type_t(cinn_type_int, 64, num_asterisks);
}

cinn_type_t cinn_uint8_t(int num_asterisks) {
  return cinn_type_t(cinn_type_uint, 8, num_asterisks);
}
cinn_type_t cinn_uint16_t(int num_asterisks) {
  return cinn_type_t(cinn_type_uint, 16, num_asterisks);
}
cinn_type_t cinn_uint32_t(int num_asterisks) {
  return cinn_type_t(cinn_type_uint, 32, num_asterisks);
}
cinn_type_t cinn_uint64_t(int num_asterisks) {
  return cinn_type_t(cinn_type_uint, 64, num_asterisks);
}

cinn_type_t cinn_float32_t(int num_asterisks) {
  return cinn_type_t(cinn_type_float, 32, num_asterisks);
}
cinn_type_t cinn_float64_t(int num_asterisks) {
  return cinn_type_t(cinn_type_float, 64, num_asterisks);
}

}  // extern "C"

struct cinn_buffer_t* cinn_buffer_t::new_(cinn_device_kind_t device,
                                          cinn_type_t type,
                                          const std::vector<int>& shape,
                                          int align) {
  int32_t dimensions = shape.size();
  CINN_CHECK(shape.size() < CINN_BUFFER_MAX_DIMS);

  struct cinn_buffer_t* buf =
      (struct cinn_buffer_t*)malloc(sizeof(struct cinn_buffer_t));
  memcpy(&(buf->dims[0]), shape.data(), shape.size() * sizeof(int));
  buf->type = type;
  buf->device = device;
  buf->memory = nullptr;
  buf->memory_size = 0;
  buf->lazy = true;
  // NOTE set device_interface for each buffer.
  switch (buf->device) {
    case cinn_x86_device:
      buf->device_interface = cinn_x86_device_interface();
      break;
    case cinn_unk_device:
      fprintf(stderr, "Device type of buffer should be set, found Unk");
      abort();
      break;
    default:
      fprintf(stderr, "Not supported device type");
      abort();
  }

  buf->dimensions = dimensions;
  buf->align = align;
#ifdef __cplusplus
  buf->external_malloc = nullptr;
  buf->external_free = nullptr;
#endif  // __cplusplus
  return buf;
}

cinn_buffer_t* cinn_buffer_new(cinn_device_kind_t device,
                               cinn_type_t type,
                               const std::vector<int>& shape,
                               int align) {
  return cinn_buffer_t::new_(device, type, shape, align);
}

cinn_pod_value_t::operator double() const {
  CINN_CHECK_EQ(type_code_, ::cinn_type_code<double>());
  return value_.v_float64;
}
cinn_pod_value_t::operator float() const {
  CINN_CHECK_EQ(type_code_, ::cinn_type_code<float>());
  return value_.v_float64;
}

cinn_pod_value_t::operator bool() const {
  CINN_CHECK_EQ(type_code_, ::cinn_type_code<bool>());
  return value_.v_int64;
}

cinn_pod_value_t::operator int8_t() const {
  CINN_CHECK_EQ(type_code_, ::cinn_type_code<int8_t>());
  return value_.v_int64;
}
cinn_pod_value_t::operator int16_t() const {
  CINN_CHECK_EQ(type_code_, ::cinn_type_code<int16_t>());
  return value_.v_int64;
}
cinn_pod_value_t::operator int32_t() const {
  CINN_CHECK_EQ(type_code_, ::cinn_type_code<int32_t>());
  return value_.v_int64;
}
cinn_pod_value_t::operator int64_t() const {
  CINN_CHECK_EQ(type_code_, ::cinn_type_code<int64_t>());
  return value_.v_int64;
}

cinn_pod_value_t::operator uint8_t() const {
  CINN_CHECK_EQ(type_code_, ::cinn_type_code<uint8_t>());
  return value_.v_int64;
}
cinn_pod_value_t::operator uint16_t() const {
  CINN_CHECK_EQ(type_code_, ::cinn_type_code<uint16_t>());
  return value_.v_int64;
}
cinn_pod_value_t::operator uint32_t() const {
  CINN_CHECK_EQ(type_code_, ::cinn_type_code<uint32_t>());
  return value_.v_int64;
}
cinn_pod_value_t::operator uint64_t() const {
  CINN_CHECK_EQ(type_code_, ::cinn_type_code<uint64_t>());
  return value_.v_int64;
}

cinn_pod_value_t::operator void*() const {
  CINN_CHECK_EQ(type_code_, ::cinn_type_code<void*>());
  return value_.v_handle;
}
cinn_pod_value_t::operator cinn_buffer_t*() const {
  CINN_CHECK_EQ(type_code_, ::cinn_type_code<cinn_buffer_t*>());
  return static_cast<cinn_buffer_t*>(value_.v_handle);
}
cinn_pod_value_t::operator char*() const {
  CINN_CHECK_EQ(type_code_, ::cinn_type_code<char*>());
  return static_cast<char*>(value_.v_handle);
}

cinn_pod_value_t::cinn_pod_value_t(cinn_value_t value, int type_code)
    : value_(value), type_code_(type_code) {}
cinn_pod_value_t::cinn_pod_value_t(cinn_buffer_t* value)
    : type_code_(::cinn_type_code<cinn_buffer_t*>()) {
  value_.v_handle = value;
}
cinn_pod_value_t::cinn_pod_value_t(bool value)
    : type_code_(::cinn_type_code<bool>()) {
  value_.v_int64 = value;
}

cinn_pod_value_t::cinn_pod_value_t(int8_t value)
    : type_code_(::cinn_type_code<int8_t>()) {
  value_.v_int64 = value;
}
cinn_pod_value_t::cinn_pod_value_t(int16_t value)
    : type_code_(::cinn_type_code<int16_t>()) {
  value_.v_int64 = value;
}
cinn_pod_value_t::cinn_pod_value_t(int32_t value)
    : type_code_(::cinn_type_code<int32_t>()) {
  value_.v_int64 = value;
}
cinn_pod_value_t::cinn_pod_value_t(int64_t value)
    : type_code_(::cinn_type_code<int64_t>()) {
  value_.v_int64 = value;
}

cinn_pod_value_t::cinn_pod_value_t(uint8_t value)
    : type_code_(::cinn_type_code<uint8_t>()) {
  value_.v_int64 = value;
}
cinn_pod_value_t::cinn_pod_value_t(uint16_t value)
    : type_code_(::cinn_type_code<uint16_t>()) {
  value_.v_int64 = value;
}
cinn_pod_value_t::cinn_pod_value_t(uint32_t value)
    : type_code_(::cinn_type_code<uint32_t>()) {
  value_.v_int64 = value;
}
cinn_pod_value_t::cinn_pod_value_t(uint64_t value)
    : type_code_(::cinn_type_code<uint64_t>()) {
  value_.v_int64 = value;
}

cinn_pod_value_t::cinn_pod_value_t(float value)
    : type_code_(::cinn_type_code<float>()) {
  value_.v_float64 = value;
}

cinn_pod_value_t::cinn_pod_value_t(double value)
    : type_code_(::cinn_type_code<double>()) {
  value_.v_float64 = value;
}
cinn_pod_value_t::cinn_pod_value_t(void* value)
    : type_code_(::cinn_type_code<void*>()) {
  value_.v_handle = value;
}
cinn_pod_value_t::cinn_pod_value_t(const char* value)
    : type_code_(::cinn_type_code<char*>()) {
  value_.v_handle = const_cast<char*>(value);
}

// @{
float cinn_pod_value_to_float(cinn_pod_value_t* value) { return *value; }
double cinn_pod_value_to_double(cinn_pod_value_t* value) { return *value; }

int64_t cinn_pod_value_to_int64(cinn_pod_value_t* value) { return *value; }
int32_t cinn_pod_value_to_int32(cinn_pod_value_t* value) { return *value; }
int16_t cinn_pod_value_to_int16(cinn_pod_value_t* value) { return *value; }
int8_t cinn_pod_value_to_int8(cinn_pod_value_t* value) { return *value; }

uint64_t cinn_pod_value_to_uint64(cinn_pod_value_t* value) { return *value; }
uint32_t cinn_pod_value_to_uint32(cinn_pod_value_t* value) { return *value; }
uint16_t cinn_pod_value_to_uint16(cinn_pod_value_t* value) { return *value; }
uint8_t cinn_pod_value_to_uint8(cinn_pod_value_t* value) { return *value; }

bool cinn_pod_value_to_bool(cinn_pod_value_t* value) { return *value; }

void* cinn_pod_value_to_void_p(cinn_pod_value_t* value) { return *value; }

cinn_buffer_t* cinn_pod_value_to_buffer_p(cinn_pod_value_t* value) {
  return *value;
}
// @}

// @{
void float_to_cinn_pod_value(float v, cinn_pod_value_t* out) {
  *out = cinn_pod_value_t(v);
}

void double_to_cinn_pod_value(double v, cinn_pod_value_t* out) {
  *out = cinn_pod_value_t(v);
}

void bool_to_cinn_pod_value(bool v, cinn_pod_value_t* out) {
  *out = cinn_pod_value_t(v);
}

void int8_to_cinn_pod_value(int8_t v, cinn_pod_value_t* out) {
  *out = cinn_pod_value_t(v);
}
void int16_to_cinn_pod_value(int16_t v, cinn_pod_value_t* out) {
  *out = cinn_pod_value_t(v);
}
void int32_to_cinn_pod_value(int32_t v, cinn_pod_value_t* out) {
  *out = cinn_pod_value_t(v);
}
void int64_to_cinn_pod_value(int64_t v, cinn_pod_value_t* out) {
  *out = cinn_pod_value_t(v);
}

void uint8_to_cinn_pod_value(uint8_t v, cinn_pod_value_t* out) {
  *out = cinn_pod_value_t(v);
}
void uint16_to_cinn_pod_value(uint16_t v, cinn_pod_value_t* out) {
  *out = cinn_pod_value_t(v);
}
void uint32_to_cinn_pod_value(uint32_t v, cinn_pod_value_t* out) {
  *out = cinn_pod_value_t(v);
}
void uint64_to_cinn_pod_value(uint64_t v, cinn_pod_value_t* out) {
  *out = cinn_pod_value_t(v);
}

void handle_to_cinn_pod_value(void* v, cinn_pod_value_t* out) {
  *out = cinn_pod_value_t(v);
}
void buffer_p_to_cinn_pod_value(const cinn_buffer_t* v, cinn_pod_value_t* out) {
  *out = cinn_pod_value_t(const_cast<cinn_buffer_t*>(v));
}
// @}

void cinn_print_debug_string(const char* s, ...) {
  va_list args;
  va_start(args, s);
  vfprintf(stderr, s, args);
  va_end(args);

  fprintf(stderr, "\n");
}

void cinn_print_debug_args(cinn_pod_value_t* args, int count) {
  cinn_print_debug_string("start debug ==");
  cinn_print_debug_string("args: %p\n", (void*)args);  // NOLINT
  cinn_print_debug_string("with %d arguments", count);
  if (!args) {
    cinn_print_debug_string("args is null!!");
    return;
  }

  for (int i = 0; i < count; i++) {
    cinn_print_debug_string("arg[%d]: %p\n", i, (void*)(&args[i]));  // NOLINT
  }
}

void cinn_args_construct(cinn_pod_value_t* arr, int count, ...) {
  CINN_CHECK(count < 1000);

  va_list args;
  va_start(args, count);
  for (int i = 0; i < count; i++) {
    cinn_pod_value_t* elem_addr = va_arg(args, cinn_pod_value_t*);
    arr[i] = *elem_addr;
    // debug_pod_value(*elem_addr, i);
  }
  va_end(args);
}

template <>
cinn_type_t cinn_type_of<bool>() {
  return cinn_bool_t();
}

template <>
cinn_type_t cinn_type_of<int8_t>() {
  return cinn_int8_t();
}
template <>
cinn_type_t cinn_type_of<int16_t>() {
  return cinn_int16_t();
}
template <>
cinn_type_t cinn_type_of<int32_t>() {
  return cinn_int32_t();
}
template <>
cinn_type_t cinn_type_of<int64_t>() {
  return cinn_int64_t();
}

template <>
cinn_type_t cinn_type_of<uint8_t>() {
  return cinn_uint8_t();
}
template <>
cinn_type_t cinn_type_of<uint16_t>() {
  return cinn_uint16_t();
}
template <>
cinn_type_t cinn_type_of<uint32_t>() {
  return cinn_uint32_t();
}
template <>
cinn_type_t cinn_type_of<uint64_t>() {
  return cinn_uint64_t();
}

template <>
cinn_type_t cinn_type_of<float>() {
  return cinn_float32_t();
}
template <>
cinn_type_t cinn_type_of<double>() {
  return cinn_float64_t();
}

template <>
cinn_type_t cinn_type_of<float*>() {
  return cinn_float64_t();
}
template <>
cinn_type_t cinn_type_of<double*>() {
  return cinn_float64_t();
}

#include "paddle/cinn/runtime/cinn_x86_device_impl.cc"
