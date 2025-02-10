// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
#include <ostream>
#include "paddle/cinn/hlir/framework/pir/op_lowering_group.h"
#include "paddle/pir/include/dialect/shape/utils/shape_or_data_expr.h"

namespace cinn::hlir::framework::pir {

class AttributeInfo {
 public:
  AttributeInfo(const std::string &name, const ::pir::Attribute &attr)
      : name_(name), attr_(attr) {}

  std::size_t hash() const;
  friend std::ostream &operator<<(std::ostream &os, const AttributeInfo &info);

 private:
  std::string name_;
  ::pir::Attribute attr_;
};

class ValueInfo {
 public:
  explicit ValueInfo(const ::pir::Value &value) : type_(value.type()) {}

  std::size_t hash() const;
  friend std::ostream &operator<<(std::ostream &os, const ValueInfo &info);

 private:
  // All value information is in TypeStorage.
  ::pir::Type type_;
};

class OperationInfo {
 public:
  explicit OperationInfo(const ::pir::Operation &op);

  std::size_t hash() const;
  friend std::ostream &operator<<(std::ostream &os, const OperationInfo &info);

 private:
  std::string name_;
  std::vector<ValueInfo> input_infos_;
  std::vector<ValueInfo> output_infos_;
  std::vector<AttributeInfo> attr_infos_;
};

class OpDepInfo {
 public:
  OpDepInfo(size_t upstream_index, size_t upstream_hash)
      : upstream_index_(upstream_index), upstream_hash_(upstream_hash) {}
  bool operator==(const OpDepInfo &other) {
    return this->upstream_index_ == other.upstream_index_ &&
           this->upstream_hash_ == other.upstream_hash_;
  }

  std::size_t hash() const;
  friend std::ostream &operator<<(std::ostream &os, const OpDepInfo &info);

 private:
  size_t upstream_index_;
  size_t upstream_hash_;
};

class FusionOpInfo {
 public:
  FusionOpInfo(const ::pir::Operation &op,
               const std::map<size_t, OpDepInfo> &deps)
      : op_info_(op), inner_deps_(deps) {}

  std::size_t hash() const;
  friend std::ostream &operator<<(std::ostream &os, const FusionOpInfo &info);

 private:
  OperationInfo op_info_;
  // operand_source id : OperationInfo hash
  std::map<size_t, OpDepInfo> inner_deps_;
};

class ProgramInfo {
 public:
  explicit ProgramInfo(const ::pir::Program &program);

  std::size_t hash() const;
  friend std::ostream &operator<<(std::ostream &os, const ProgramInfo &info);

 private:
  uint64_t id_;
};

class FusionInfo {
  using IntArgsMap = std::map<int, CINNKernelInfo::SymbolArgBindInfo>;

 public:
  explicit FusionInfo(const OpLoweringGroup &group);
  FusionInfo() = delete;
  FusionInfo(const FusionInfo &) = default;
  FusionInfo(FusionInfo &&) = default;

  std::size_t hash() const;

  bool operator==(const FusionInfo &other) const {
    return this->hash() == other.hash();
  }
  friend std::ostream &operator<<(std::ostream &os, const FusionInfo &info);

 private:
  void ParseOpInfos(const OpLoweringGroup &group);
  void ParseInputDimExprs(const OpLoweringGroup &group);
  void ParseProgramInfo(const OpLoweringGroup &group);

  std::vector<FusionOpInfo> op_infos_;
  std::vector<::symbol::ShapeOrDataDimExprs> input_dim_exprs_;
  std::shared_ptr<ProgramInfo> program_info_;
  std::size_t cached_hash_value_{0};

  // Used to make same subgraphs have unique FusionInfo while
  // FLAGS_enable_cinn_compile_cache = false, default empty;
  std::string unique_fn_name_{""};
};

std::ostream &operator<<(std::ostream &os, const AttributeInfo &info);
std::ostream &operator<<(std::ostream &os, const ValueInfo &info);
std::ostream &operator<<(std::ostream &os, const OperationInfo &info);
std::ostream &operator<<(std::ostream &os, const OpDepInfo &info);
std::ostream &operator<<(std::ostream &os, const FusionOpInfo &info);
std::ostream &operator<<(std::ostream &os, const FusionInfo &info);

// See boost.hash_combine for details
template <class T>
inline void hash_combine(std::size_t &seed,  // NOLINT
                         const T &v) {
  std::hash<T> hasher;
  seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

std::vector<const ::pir::Operation *> TopologySort(
    const OpLoweringGroup &group);

}  // namespace cinn::hlir::framework::pir

namespace std {
#define REGISTER_STD_HASH(class_name)                              \
  template <>                                                      \
  struct hash<cinn::hlir::framework::pir::class_name> {            \
    std::size_t operator()(                                        \
        const cinn::hlir::framework::pir::class_name &obj) const { \
      return obj.hash();                                           \
    }                                                              \
  };

REGISTER_STD_HASH(AttributeInfo)
REGISTER_STD_HASH(ValueInfo)
REGISTER_STD_HASH(OperationInfo)
REGISTER_STD_HASH(OpDepInfo)
REGISTER_STD_HASH(FusionOpInfo)
REGISTER_STD_HASH(ProgramInfo)
REGISTER_STD_HASH(FusionInfo)
}  // namespace std
