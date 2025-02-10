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

#include "paddle/cinn/common/type.h"

#include <functional>
#include <string>
#include <unordered_map>
#include <utility>
#include "paddle/common/enforce.h"
namespace cinn {
namespace common {

struct Type::Storage {
  Storage() = default;
  Storage(type_t t, int b, int w, specific_type_t st)
      : type_(t), bits_(b), lanes_(w), specific_type_(st) {}

  type_t type_{type_t::Unk};
  // distinguish FP16/BF16, or E5M2/E4M3 (when FP8 is supported)
  specific_type_t specific_type_{specific_type_t::None};
  cpp_type_t cpp_type_{cpp_type_t::None};

  //! How many bits per element.
  int bits_{0};

  //! How many elements(if a vector type), for scalar types, it should be 1.
  int lanes_{1};

  //! Name of the customized type.
  std::string customized_type_;
};

Type::~Type() {}

std::string Type::to_string() const {
  std::string ret = "";
  if (is_cpp_const()) ret += "const ";
  ret += Type2Str(*this);

  if (lanes() > 1) {
    ret += "<";
    ret += std::to_string(lanes());
    ret += ">";
  }
  if (is_cpp_handle()) ret += "*";
  if (is_cpp_handle2()) ret += "**";

  return ret;
}

std::ostream &operator<<(std::ostream &os, const Type &t) {
  os << t.to_string();
  return os;
}

std::ostream &operator<<(std::ostream &os, Type::type_t t) {
  switch (t) {
    case Type::type_t::Void:
      os << "Void";
      break;
    case Type::type_t::UInt:
      os << "UInt";
      break;
    case Type::type_t::Int:
      os << "Int";
      break;
    case Type::type_t::Float:
      os << "Float";
      break;
    case Type::type_t::Unk:
      os << "Unk";
      break;
    case Type::type_t::Customized:
      os << "Customized";
  }
  return os;
}

Type &Type::set_cpp_handle(bool x) {
  // unset the other handle-related bits.
  set_cpp_handle2(false);

  auto &v = (*reinterpret_cast<uint8_t *>(&GetStorage().cpp_type_));
  // unset the other handle-related bits.
  v &= ~static_cast<uint8_t>(cpp_type_t::Handle);
  v &= ~static_cast<uint8_t>(cpp_type_t::HandleHandle);

  if (x)
    v |= static_cast<uint8_t>(cpp_type_t::Handle);
  else
    v &= ~static_cast<uint8_t>(cpp_type_t::Handle);

  return *this;
}

Type &Type::set_cpp_handle2(bool x) {
  auto &v = (*reinterpret_cast<uint8_t *>(&GetStorage().cpp_type_));

  // unset the other handle-related bits.
  v &= ~static_cast<uint8_t>(cpp_type_t::Handle);
  v &= ~static_cast<uint8_t>(cpp_type_t::HandleHandle);

  if (x)
    v |= static_cast<uint8_t>(cpp_type_t::HandleHandle);
  else
    v &= ~static_cast<uint8_t>(cpp_type_t::HandleHandle);

  return *this;
}

Type Type::VectorOf(int w) const {
  CheckTypeValid();
  return Type(type(), bits(), w, specific_type());
}

Type::Type(const Type &other) {
  if (other.storage_) storage_.reset(new Storage(*other.storage_));
}

Type Type::ElementOf() const {
  CheckTypeValid();
  auto type = *this;
  type.storage_->lanes_ = 1;
  return type;
}

void Type::CheckTypeValid() const {
  PADDLE_ENFORCE_NE(
      GetStorage().type_,
      type_t::Unk,
      ::common::errors::InvalidArgument("The type is not initialized."));

  if (GetStorage().type_ == type_t::Float && GetStorage().bits_ == 16) {
    PADDLE_ENFORCE_EQ((GetStorage().specific_type_ == specific_type_t::FP16 ||
                       GetStorage().specific_type_ == specific_type_t::BF16),
                      true,
                      ::common::errors::InvalidArgument(
                          "When creating a 16-bit Float, the 'specific_type_t' "
                          "must be FP16 or BF16. "
                          "Received: specific_type_t = %d.",
                          static_cast<int>(GetStorage().specific_type_)));
  }
}

Type Type::PointerOf() const {
  CheckTypeValid();
  auto x = *this;
  PADDLE_ENFORCE_EQ(x.is_cpp_handle2(),
                    false,
                    ::common::errors::InvalidArgument(
                        "Not support three levels of PointerOf."));
  if (x.is_cpp_handle())
    x.set_cpp_handle2();
  else
    x.set_cpp_handle();
  return x;
}

Type Type::ConstOf() const {
  CheckTypeValid();
  auto x = *this;
  x.set_cpp_const();
  return x;
}

bool Type::is_supported() const {
  return this->is_float(32) || this->is_float16() || this->is_bfloat16() ||
         this->is_float(64) || this->is_bool() || this->is_int(8) ||
         this->is_int(16) || this->is_int(32) || this->is_int(64) ||
         this->is_uint(8) || this->is_uint(16) || this->is_uint(32) ||
         this->is_uint(64);
}

Type Type::IgnoreConst() const {
  CheckTypeValid();
  auto x = *this;
  x.set_cpp_const(false);
  return x;
}

Type Type::with_bits(int x) const {
  PADDLE_ENFORCE_EQ(
      is_primitive(),
      true,
      ::common::errors::InvalidArgument(
          "The type must be primitive to set the number of bits."));
  Type type = *this;
  type.GetStorage().bits_ = x;
  return type;
}

Type Type::with_type(Type::type_t x) const {
  Type type = *this;
  type.GetStorage().type_ = x;
  return type;
}

Type Type::with_lanes(int x) const {
  PADDLE_ENFORCE_EQ(valid(),
                    true,
                    ::common::errors::InvalidArgument(
                        "The type must be valid to set the number of lanes."));
  Type type = *this;
  type.GetStorage().lanes_ = x;
  return type;
}

Type Type::with_cpp_const(bool x) const {
  Type type = *this;
  type.set_cpp_const(x);
  return type;
}

Type &Type::set_cpp_const(bool is_const) {
  uint8_t &data = *reinterpret_cast<uint8_t *>(&GetStorage().cpp_type_);
  if (is_const) {
    data |= static_cast<uint8_t>(cpp_type_t::Const);
  } else {
    data &= ~(static_cast<uint8_t>(cpp_type_t::Const));
  }

  return *this;
}
Type &Type::set_customized_type(const std::string &t) {
  GetStorage().type_ = type_t ::Customized;
  GetStorage().customized_type_ = t;

  return *this;
}

bool Type::valid() const {
  if (is_unk()) return false;
  if (is_customized()) {
    return !GetStorage().customized_type_.empty();
  }
  if (is_float() && GetStorage().bits_ == 16) {
    return (GetStorage().specific_type_ == specific_type_t::FP16 ||
            GetStorage().specific_type_ == specific_type_t::BF16);
  }
  if (is_primitive()) {
    return bits() != 0;
  }

  return true;
}

Type::Type(Type::type_t t, int b, int w, specific_type_t st)
    : storage_(new Storage(t, b, w, st)) {
  if (t == Type::type_t::Float && b == 16) {
    PADDLE_ENFORCE_EQ(
        (st == specific_type_t::FP16 || st == specific_type_t::BF16),
        true,
        ::common::errors::InvalidArgument(
            "When creating a 16-bit Float, the "
            "'specific_type_t' must be FP16 or BF16. "
            "Received: specific_type_t = %d.",
            static_cast<int>(st)));
  }
}
bool Type::is_primitive() const {
  return !is_unk() && type() != type_t::Customized;
}
bool Type::is_customized() const {
  return !is_unk() && type() == type_t::Customized;
}
bool Type::is_unk() const { return type() == type_t::Unk; }
bool Type::is_bool() const { return type() == type_t::UInt && bits() == 1; }
bool Type::is_void() const { return type() == type_t::Void; }
bool Type::is_vector() const { return lanes() > 1; }
bool Type::is_scalar() const { return lanes() == 1; }
// Note: when calling is_float(16), 'st' can't be specific_type_t::None to
// distinguish FP16/BF16, or use is_float16()/is_bfloat16() for short
bool Type::is_float(int bits, specific_type_t st) const {
  if (type() == type_t::Float && bits == 16) {
    PADDLE_ENFORCE_NE(
        st,
        specific_type_t::None,
        ::common::errors::InvalidArgument(
            "When calling is_float(16), 'st' can't be specific_type_t::None to "
            "distinguish FP16/BF16. Use is_float16() or is_bfloat16() for "
            "short."));
    return st == this->specific_type();
  } else {
    return type() == type_t::Float && (bits < 0 || bits == this->bits());
  }
}
bool Type::is_float16() const { return is_float(16, specific_type_t::FP16); }
bool Type::is_bfloat16() const { return is_float(16, specific_type_t::BF16); }
bool Type::is_uint(int bits) const {
  return type() == type_t::UInt && (bits < 0 || bits == this->bits());
}
bool Type::is_int(int bits) const {
  return type() == type_t::Int && (bits < 0 || bits == this->bits());
}
bool Type::is_integer(int bits) const {
  return (type() == type_t::Int || type() == type_t::UInt) &&
         (bits < 0 || bits == this->bits());
}
bool Type::is_index_type() {
  return is_int() && lanes() == 1 && (bits() == 32 || bits() == 64);
}
bool Type::is_cpp_handle() const {
  return static_cast<uint8_t>(GetStorage().cpp_type_) &
         static_cast<uint8_t>(cpp_type_t::Handle);
}
bool Type::is_cpp_handle2() const {
  return static_cast<uint8_t>(GetStorage().cpp_type_) &
         static_cast<uint8_t>(cpp_type_t::HandleHandle);
}
bool Type::is_cpp_const() const {
  return static_cast<uint8_t>(cpp_type_t::Const) &
         static_cast<uint8_t>(GetStorage().cpp_type_);
}
const std::string &Type::customized_type() const {
  return GetStorage().customized_type_;
}
bool Type::is_customized_type() const {
  return !GetStorage().customized_type_.empty();
}
Type::type_t Type::type() const { return GetStorage().type_; }
Type::specific_type_t Type::specific_type() const {
  return GetStorage().specific_type_;
}
int Type::bits() const { return GetStorage().bits_; }
int Type::lanes() const { return GetStorage().lanes_; }
Type::cpp_type_t Type::cpp_type() const { return GetStorage().cpp_type_; }
bool Type::operator==(const Type &other) const {
  return type() == other.type() && specific_type() == other.specific_type() &&
         bits() == other.bits() && lanes() == other.lanes() &&
         GetStorage().cpp_type_ == other.GetStorage().cpp_type_ &&
         customized_type() == other.customized_type();
}
bool Type::is_string() const { return type() == type_t::String; }

Type &Type::operator=(const Type &other) {
  if (other.storage_) {
    storage_.reset(new Storage(other.GetStorage().type_,
                               other.GetStorage().bits_,
                               other.GetStorage().lanes_,
                               other.GetStorage().specific_type_));
    storage_->cpp_type_ = other.GetStorage().cpp_type_;
    storage_->customized_type_ = other.GetStorage().customized_type_;
  }
  return *this;
}

Type::Storage &Type::GetStorage() {
  PADDLE_ENFORCE_NOT_NULL(storage_,
                          ::common::errors::InvalidArgument(
                              "The type is not initialized! Please check."));
  return *storage_;
}

const Type::Storage &Type::GetStorage() const {
  PADDLE_ENFORCE_NOT_NULL(storage_,
                          ::common::errors::InvalidArgument(
                              "The type is not initialized! Please check."));
  return *storage_;
}

Type::Type() : storage_(new Storage) {}
Type::Type(Type &&other) : storage_(std::move(other.storage_)) {}

const Type &BF16() {
  static auto t = Float(16, 1, Type::specific_type_t::BF16);
  return t;
}
const Type &F16() {
  static auto t = Float(16, 1, Type::specific_type_t::FP16);
  return t;
}
const Type &F32() {
  static auto t = Float(32);
  return t;
}
const Type &F64() {
  static auto t = Float(64);
  return t;
}
const Type &I8() {
  static auto t = Int(8);
  return t;
}
const Type &I16() {
  static auto t = Int(16);
  return t;
}
const Type &I32() {
  static auto t = Int(32);
  return t;
}
const Type &I64() {
  static auto t = Int(64);
  return t;
}
const Type &UI8() {
  static auto t = UInt(8);
  return t;
}
const Type &UI16() {
  static auto t = UInt(16);
  return t;
}
const Type &UI32() {
  static auto t = UInt(32);
  return t;
}
const Type &UI64() {
  static auto t = UInt(64);
  return t;
}
const Type &I1() {
  static auto t = Int(1);
  return t;
}
const Type &UI1() {
  static auto t = UInt(1);
  return t;
}

struct TypeHash {
  size_t operator()(const Type &type) const {
    std::string hash_str;
    hash_str += std::to_string(static_cast<int>(type.type()));
    hash_str += std::to_string(static_cast<int>(type.specific_type()));
    hash_str += std::to_string(type.bits());
    hash_str += std::to_string(type.lanes());
    hash_str += std::to_string(static_cast<int>(type.cpp_type()));
    if (type.is_customized_type()) {
      hash_str += type.customized_type();
    }

    return std::hash<std::string>()(hash_str);
  }
};

int Type::bytes() const {
  // if the type is a pointer
  auto cpp_type = this->cpp_type();
  if (cpp_type == Type::cpp_type_t::Handle ||
      cpp_type == Type::cpp_type_t::HandleHandle) {
    return sizeof(void *);
  }

// if the type is an known pod type
#define GET_TYPE_SIZE_PAIR(TYPE) \
  { type_of<TYPE>(), sizeof(TYPE) }
  static std::unordered_map<Type, int, TypeHash> type_bytes = {
      GET_TYPE_SIZE_PAIR(bfloat16),
      GET_TYPE_SIZE_PAIR(float16),
      GET_TYPE_SIZE_PAIR(float),
      GET_TYPE_SIZE_PAIR(double),

      GET_TYPE_SIZE_PAIR(char),
      GET_TYPE_SIZE_PAIR(signed char),
      GET_TYPE_SIZE_PAIR(unsigned char),

      GET_TYPE_SIZE_PAIR(int8_t),
      GET_TYPE_SIZE_PAIR(int16_t),
      GET_TYPE_SIZE_PAIR(int32_t),
      GET_TYPE_SIZE_PAIR(int64_t),

      GET_TYPE_SIZE_PAIR(uint8_t),
      GET_TYPE_SIZE_PAIR(uint16_t),
      GET_TYPE_SIZE_PAIR(uint32_t),
      GET_TYPE_SIZE_PAIR(uint64_t),

      GET_TYPE_SIZE_PAIR(bool),
  };
#undef GET_TYPE_SIZE_PAIR

  if (type_bytes.count(*this)) {
    return type_bytes.at(*this);
  }

  // else get size by bits size
  auto bit_size = this->bits();
  return (bit_size + 7) / 8;
}

Type Str2Type(const std::string &type) {
  static std::unordered_map<std::string, Type> str2type_map = {
      {"unk", Type()},
      {"void", Void()},
      {"bool", Bool()},
      {"unsigned char", UI8()},

      {"char", I8()},
      {"signed char", I8()},

      {"string", String()},

      {"bit", I1()},
      {"signed bit", I1()},
      {"int1", I1()},
      {"int1_t", I1()},

      {"ubit", UI1()},
      {"unsigned bit", UI1()},
      {"uint1", UI1()},
      {"uint1_t", UI1()},

      {"int8", I8()},
      {"int8_t", I8()},

      {"int16", I16()},
      {"int16_t", I16()},

      {"int", I32()},
      {"int32", I32()},
      {"int32_t", I32()},

      {"int64", I64()},
      {"int64_t", I64()},

      {"uint8", UI8()},
      {"uint8_t", UI8()},

      {"uint16", UI16()},
      {"uint16_t", UI16()},

      {"uint", UI32()},
      {"uint32", UI32()},
      {"uint32_t", UI32()},

      {"uint64", UI64()},
      {"uint64_t", UI64()},

      {"bfloat16", BF16()},
      {"float16", F16()},
      {"half", F16()},

      {"float", F32()},
      {"float32", F32()},

      {"float64", F64()},
      {"double", F64()},

      {"void*", type_of<void *>()},
      {"void_p", type_of<void *>()},
      {"void**", type_of<void **>()},
      {"void_p_p", type_of<void **>()},

      {"int8*", type_of<int8_t *>()},
      {"int8_p", type_of<int8_t *>()},
      {"int8_t*", type_of<int8_t *>()},

      {"uint8*", type_of<uint8_t *>()},
      {"uint8_p", type_of<uint8_t *>()},
      {"uint8_t*", type_of<uint8_t *>()},

      {"float*", type_of<float *>()},
      {"float32*", type_of<float *>()},
      {"float_p", type_of<float *>()},
      {"float32_p", type_of<float *>()},

      {"double*", type_of<double *>()},
      {"float64*", type_of<double *>()},
      {"double_p", type_of<double *>()},
      {"float64_p", type_of<double *>()},

      {"cinn_buffer", type_of<cinn_buffer_t>()},
      {"cinn_buffer*", type_of<cinn_buffer_t>()},
      {"cinn_buffer_p", type_of<cinn_buffer_t *>()},

      {"const cinn_buffer*", type_of<const cinn_buffer_t *>()},
      {"const_cinn_buffer_p", type_of<const cinn_buffer_t *>()},

      {"cinn_pod_value", type_of<cinn_pod_value_t>()},
      {"cinn_pod_value*", type_of<cinn_pod_value_t *>()},
      {"cinn_pod_value_p", type_of<cinn_pod_value_t *>()},
  };
  PADDLE_ENFORCE_NE(
      str2type_map.find(type),
      str2type_map.end(),
      ::common::errors::InvalidArgument(
          "Not supported type [%s]! Please check.", type.c_str()));
  return str2type_map.at(type);
}

std::string Type2Str(const Type &type) {
  switch (type.type()) {
    case Type::type_t::Int:
      return "int" + std::to_string(type.bits());

    case Type::type_t::UInt:
      if (type.bits() == 1) {
        return "bool";
      } else {
        return "uint" + std::to_string(type.bits());
      }

    case Type::type_t::Float:
      switch (type.specific_type()) {
        case Type::specific_type_t::None:
          return "float" + std::to_string(type.bits());
        case Type::specific_type_t::BF16:
          return "bfloat16";
        case Type::specific_type_t::FP16:
          return "float16";
        default:
          break;
      }

    case Type::type_t::Void:
      return "void";

    case Type::type_t::Customized:
      return type.customized_type();

    case Type::type_t::String:
      return "string";

    case Type::type_t::Unk:
      return "unk";

    default:
      std::stringstream ss;
      ss << "Not support type [" << type << "] ! Please Check.\n";
      PADDLE_THROW(::common::errors::InvalidArgument(ss.str()));
  }
  return "unk";
}

}  // namespace common
}  // namespace cinn
