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

#include <map>
#include <vector>

#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/pass.h"

namespace paddle {
namespace framework {
namespace ir {

class OpCompat;

class AttrCompat {
 public:
  AttrCompat(const std::string& attr_name, OpCompat* op_compat)
      : optional_(false), attr_name_(attr_name), op_compat_(op_compat) {}

  //! Assert the attribute type is `T`.
  template <typename T>
  AttrCompat& IsType();

  // @{ String-related methods
  //! Assert the attribute is an string in the `candidates` domain.
  AttrCompat& IsStringEQ(const std::string& value);
  //! Assert the attribute is an string in the `candidates` domain.
  AttrCompat& IsStringIn(const std::set<std::string>& candidates);
  //! Assert the attribute is a string and match a custom judging function.
  AttrCompat& IsStringMatch(
      const std::function<bool(const std::string&)>& func);
  // @}

  //! Assert the attribute is an integer in the `candidates` domain.
  AttrCompat& IsIntIn(const std::set<int>& candidates);

  // @{ Number-related methods
  //! Assert the attribute is a number and > `v`.
  template <typename T>
  AttrCompat& IsNumGT(T v);
  //! Assert the attribute is a number and >= `v`.
  template <typename T>
  AttrCompat& IsNumGE(T v);
  //! Assert the attribute is a number and < `v`.
  template <typename T>
  AttrCompat& IsNumLT(T v);
  //! Assert the attribute is a number and <= `v`.
  template <typename T>
  AttrCompat& IsNumLE(T v);
  //! Assert the attribute is a number and == `v`.
  template <typename T>
  AttrCompat& IsNumEQ(T v);
  //! Assert the attribute is a number and matches a customized judging
  //! function.
  template <typename T>
  AttrCompat& IsNumMatch(bool (*func)(T));
  // @}

  //! Assert the attribute is a boolean value equals `v`.
  AttrCompat& IsBoolEQ(bool v);

  //! Tell whether this attribute is left as default value.
  AttrCompat& IsLeftDefault();

  AttrCompat& IsOptional();

  //! Jump back to retrieve OpCompat instance.
  OpCompat& End() { return *op_compat_; }

  bool operator()(const OpDesc& op_desc);

 private:
  bool optional_;
  std::string attr_name_;
  OpCompat* op_compat_;
  std::vector<std::function<bool(const Attribute&)>> conditions_;
};

class InputOrOutputCompat {
 public:
  InputOrOutputCompat(const std::string& name, OpCompat* op_compat)
      : optional_(false), name_(name), op_compat_(op_compat) {}

  InputOrOutputCompat& IsTensor();
  InputOrOutputCompat& IsOptional();
  bool Optional() const { return optional_; }
  bool operator()(const std::vector<std::string>& input) const;

  //! Jump back to retrieve OpCompat instance.
  OpCompat& End() { return *op_compat_; }

 private:
  bool optional_;
  std::string name_;
  OpCompat* op_compat_;
  std::vector<std::function<bool(const std::vector<std::string>&)>> conditions_;
};

/**
 * OpCompat is a helper class to help define the compatible Op definition.
 *
 * Usage:
 *   OpCompat compat("FC");
 *   compat.AddAttr("in_num_col_dims").IsNumLE(1).End()
 *         .AddAttr("activation_type").IsStringIn({"tanh", "sigmoid"}).End()
 *         .AddInput("Input").IsTensor().End()
 *         .AddInput("W").IsTensor().End()
 *         .AddInput("Bias").IsTensor().IsOptional().End()
 *         .AddOutput("Out").IsTensor().End()
 *
 * All the inference-aware Op definition is as above, all the other attributes
 * not contained in the definition should be set default value or it would be
 * judged incompatible.
 */
class OpCompat {
 public:
  explicit OpCompat(const std::string& op_name) : op_name_(op_name) {}
  explicit OpCompat(std::string&& op_name) : op_name_(std::move(op_name)) {}
  explicit OpCompat(const OpCompat&) = default;
  explicit OpCompat(OpCompat&&) = default;

  AttrCompat& AddAttr(const std::string& attr_name);
  InputOrOutputCompat& AddInput(const std::string& name);
  InputOrOutputCompat& AddOutput(const std::string& name);

  //! Judge whether an OpDesc match the defined Op compatibility.
  bool Judge(const OpDesc& op_desc, const std::string& pass_name);
  const std::string& Name() const { return op_name_; }

 private:
  std::string op_name_;
  std::unordered_map<std::string, AttrCompat> attr_compats_;
  std::unordered_map<std::string, InputOrOutputCompat> input_compats_;
  std::unordered_map<std::string, InputOrOutputCompat> output_compats_;
  std::unordered_set<std::string> attrs_set_;
  bool is_first_judge_ = true;
};

/**
 * OpCompatSensiblePass is a base class for all the passes those is sensitive
 * to Op update.
 * There are two methods to help tell the compatibility of an Op
 *   bool IsCompat(const GraphPatternDetector::subgraph_t& subgraph, Graph* g);
 *   bool IsCompat(const OpDesc& op_desc);
 *
 * One can register the related Op compatibilities using
 *   void AddOpCompat(OpCompat&& judger);
 *
 * Most of the Passes are used for fusing ops, so we define a method for such
 * scenarios.
 *   void AccessSubgraph(const GraphPatternDetector::subgraph_t& subgraph,
 Graph* g);
 * It will check the Op compatibility automatically.
 * For other scenarios, one should call `IsCompat` by himself.
 *
 * A FC fuse pass example:
 * class FcFusePass : public OpCompatSensiblePass {
 *  public:
 *   FcFusePass() {
 *     // define Mul op compatibility.
 *     AddOpCompat(OpCompat("Mul"))
 *        .AddInput("Input").IsTensor().End()
 *        .AddAttr("in_num_col_dims").IsNumGE(1);
 *     AddOpCompat(OpCompat("Add")). ...;
 *     // There are multiple activation implementation.
 *     AddOpCompat(OpCompat("Tanh")). ...;
 *     AddOpCompat(OpCompat("Sigmoid")). ...;
 *   }
 *
 *   // override the subgraph access method
 *   virtual bool AccessSubgraphImpl(
 *   const GraphPatternDetector::subgraph_t& subgraph,
 *         Graph* g) override { ... }
 *
 *   // Call the AccessSubgraph method in main procedure of this Pass.
 * };
 */
class OpCompatSensiblePass : public Pass {
 protected:
  /**
   * Developer should push the compatibility `teller` for each kind of Op in the
   * subgraph.
   * NOTE One should add all the related op compatibility in the construct so
   * that all the following methods are valid.
   */
  OpCompat& AddOpCompat(OpCompat&& op_compat);

  //! Tell the Op compatibility of a subgraph.
  bool IsCompat(const GraphPatternDetector::subgraph_t& subgraph,
                Graph* g) const;

  //! Tell the op compatibility of a single Op.
  bool IsCompat(const OpDesc& op_desc) const {
    if (!op_compat_judgers_.count(op_desc.Type())) return false;
    return op_compat_judgers_.at(op_desc.Type())->Judge(op_desc, Type());
  }

 private:
  std::map<std::string, std::unique_ptr<OpCompat>> op_compat_judgers_;
};

template <typename T>
AttrCompat& AttrCompat::IsType() {
  conditions_.emplace_back(
      [](const Attribute& attr) -> bool { return attr.type() == typeid(T); });
  return *this;
}

template <typename T>
AttrCompat& AttrCompat::IsNumGT(T v) {
  conditions_.emplace_back([v](const Attribute& attr) -> bool {
    T value = PADDLE_GET_CONST(T, attr);
    return value > v;
  });
  return *this;
}

template <typename T>
AttrCompat& AttrCompat::IsNumGE(T v) {
  conditions_.emplace_back([v](const Attribute& attr) -> bool {
    T value = PADDLE_GET_CONST(T, attr);
    return value >= v;
  });
  return *this;
}

template <typename T>
AttrCompat& AttrCompat::IsNumLT(T v) {
  conditions_.emplace_back([v](const Attribute& attr) -> bool {
    T value = PADDLE_GET_CONST(T, attr);
    return value < v;
  });
  return *this;
}

template <typename T>
AttrCompat& AttrCompat::IsNumLE(T v) {
  conditions_.emplace_back([v](const Attribute& attr) -> bool {
    T value = PADDLE_GET_CONST(T, attr);
    return value <= v;
  });
  return *this;
}

template <typename T>
AttrCompat& AttrCompat::IsNumEQ(T v) {
  conditions_.emplace_back([v](const Attribute& attr) -> bool {
    T value = PADDLE_GET_CONST(T, attr);
    return value == v;
  });
  return *this;
}

template <typename T>
AttrCompat& AttrCompat::IsNumMatch(bool (*func)(T)) {
  conditions_.emplace_back([func](const Attribute& attr) -> bool {
    T value = PADDLE_GET_CONST(T, attr);
    return func(value);
  });
  return *this;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle
