// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
#include <functional>

namespace ir {

///
/// \brief TypeId is the unique identification of Type, each Type corresponds to
/// a unique TypeId, the same id indicates the same Type class. TypeId provides
/// an instantiation interface: TypeId::get.
///
/// Example:
/// \code{cpp}
///   class TypeA {};
///   TypeId type_a_id = TypeId::get<TypeA>();
/// \endcode
///
class TypeId {
  struct Storage {};

 public:
  ///
  /// \brief Returns the unique TypeId of Type T.
  ///
  /// \return The unique TypeId of Type T.
  ///
  template <typename T>
  static TypeId get();

  TypeId() = default;

  TypeId(const TypeId &other) = default;

  TypeId &operator=(const TypeId &other) = default;

  void *AsOpaquePointer() const { return storage_; }
  static TypeId RecoverFromOpaquePointer(void *pointer) {
    return TypeId(static_cast<Storage *>(pointer));
  }

  void *AsOpaquePointer() const { return storage_; }

  static TypeId RecoverFromOpaquePointer(const void *pointer) {
    return TypeId(static_cast<Storage *>(const_cast<void *>(pointer)));
  }

  ///
  /// \brief Comparison operations.
  ///
  inline bool operator==(const TypeId &other) const {
    return storage_ == other.storage_;
  }
  inline bool operator!=(const TypeId &other) const {
    return !(*this == other);
  }
  inline bool operator<(const TypeId &other) const {
    return storage_ < other.storage_;
  }

  ///
  /// \brief Enable hashing TypeId instances.
  ///
  friend struct std::hash<TypeId>;

 private:
  ///
  /// \brief Construct a TypeId and initialize storage.
  ///
  /// \param storage The storage of this TypeId.
  ///
  explicit TypeId(Storage *storage) : storage_(storage) {}

  Storage *storage_{nullptr};
};

namespace detail {
class alignas(8) SelfOwningTypeId {
 public:
  SelfOwningTypeId() = default;
  SelfOwningTypeId(const SelfOwningTypeId &) = delete;
  SelfOwningTypeId &operator=(const SelfOwningTypeId &) = delete;
  SelfOwningTypeId(SelfOwningTypeId &&) = delete;
  SelfOwningTypeId &operator=(SelfOwningTypeId &&) = delete;

  operator TypeId() const { return id(); }
  TypeId id() const { return TypeId::RecoverFromOpaquePointer(this); }
};
template <typename T>
class __declspec(dllexport) TypeIdResolver {
 public:
  static TypeId Resolve() { return id_; }
  static SelfOwningTypeId id_;
};

template <typename T>
void PrintDebug() {
  std::cout << &TypeIdResolver<T>::id_ << '\n';
}

}  // namespace detail

template <typename T>
TypeId TypeId::get() {
  return detail::TypeIdResolver<T>::Resolve();
}

#define IR_DECLARE_EXPLICIT_TYPE_ID(TYPE_CLASS)            \
  namespace ir {                                           \
  namespace detail {                                       \
  template <>                                              \
  class __declspec(dllexport) TypeIdResolver<TYPE_CLASS> { \
   public:                                                 \
    static TypeId Resolve() { return id_; }                \
    static SelfOwningTypeId id_;                           \
  };                                                       \
  }                                                        \
  }  // namespace ir

/*
#define IR_DECLARE_EXPLICIT_TYPE_ID(TYPE_CLASS)
*/

#define IR_DEFINE_EXPLICIT_TYPE_ID(TYPE_CLASS)           \
  namespace ir {                                         \
  namespace detail {                                     \
  SelfOwningTypeId TypeIdResolver<TYPE_CLASS>::id_ = {}; \
  }                                                      \
  }  // namespace ir

}  // namespace ir

namespace std {
///
/// \brief Enable hashing TypeId instances.
///
template <>
struct hash<ir::TypeId> {
  std::size_t operator()(const ir::TypeId &obj) const {
    return std::hash<const ir::TypeId::Storage *>()(obj.storage_);
  }
};
}  // namespace std
