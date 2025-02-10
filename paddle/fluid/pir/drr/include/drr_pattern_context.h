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

#include <any>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

#include "paddle/fluid/pir/drr/include/drr_match_context.h"
#include "paddle/utils/test_macros.h"

namespace paddle {
namespace drr {

class Op;
class Tensor;
class OpCall;
class SourcePattern;
class ResultPattern;
class PatternGraph;
class SourcePatternGraph;
class ResultPatternGraph;

class NormalAttribute {
 public:
  NormalAttribute() = default;

  explicit NormalAttribute(const std::string& name) : attr_name_(name) {}

  const std::string& name() const { return attr_name_; }

 private:
  std::string attr_name_{"default_name"};
};

using AttrComputeFunc = std::function<std::any(const MatchContext&)>;

class ComputeAttribute {
 public:
  ComputeAttribute() = default;

  explicit ComputeAttribute(const AttrComputeFunc& attr_compute_func)
      : attr_compute_func_(attr_compute_func) {}

  const AttrComputeFunc& attr_compute_func() const {
    return attr_compute_func_;
  }

 private:
  AttrComputeFunc attr_compute_func_{nullptr};
};

using ConstraintFunction = std::function<bool(const MatchContext&)>;
class Constraint {
 public:
  explicit Constraint(const ConstraintFunction& constraint_fn)
      : is_meet_constraint_(constraint_fn) {}
  bool operator()(const MatchContext& match_context) const {
    return is_meet_constraint_(match_context);
  }

 private:
  ConstraintFunction is_meet_constraint_;
};

using PostProcessFunction = std::function<void(const MatchContext&)>;
class PostProcess {
 public:
  explicit PostProcess(const PostProcessFunction& post_process_fn)
      : post_process_after_match_(post_process_fn) {}
  void operator()(const MatchContext& match_context) const {
    return post_process_after_match_(match_context);
  }

 private:
  PostProcessFunction post_process_after_match_;
};

using Attribute = std::variant<NormalAttribute, ComputeAttribute>;
class TEST_API DrrPatternContext {
 public:
  DrrPatternContext();
  ~DrrPatternContext() = default;

  drr::SourcePattern SourcePattern();

  std::shared_ptr<SourcePatternGraph> source_pattern_graph() const {
    return source_pattern_graph_;
  }

  std::vector<Constraint> constraints() const;

  std::vector<PostProcess> post_processes() const;

  std::shared_ptr<ResultPatternGraph> result_pattern_graph() const {
    return result_pattern_graph_;
  }

 private:
  friend class drr::SourcePattern;
  friend class drr::ResultPattern;

  const Op& SourceOpPattern(
      const std::string& op_type,
      const std::unordered_map<std::string, Attribute>& attributes = {});
  drr::Tensor& SourceTensorPattern(const std::string& name);

  const Op& ResultOpPattern(
      const std::string& op_type,
      const std::unordered_map<std::string, Attribute>& attributes = {},
      const std::unordered_map<std::string, Attribute>& runtime_attributes =
          {});
  drr::Tensor& ResultTensorPattern(const std::string& name);

  void AddConstraint(const ConstraintFunction& constraint_fn);

  void AddPostProcess(const PostProcessFunction& post_process_fn);

  std::shared_ptr<SourcePatternGraph> source_pattern_graph_;
  std::vector<Constraint> constraints_;
  std::vector<PostProcess> post_processes_;
  std::shared_ptr<ResultPatternGraph> result_pattern_graph_;

  std::vector<std::shared_ptr<const drr::Op>> owned_ops_;
};

class Op {
 public:
  TEST_API const std::string& name() const { return op_type_name_; }

  TEST_API Tensor& operator()() const;
  TEST_API void operator()(const Tensor& arg, const Tensor* out) const;
  TEST_API Tensor& operator()(const Tensor& arg) const;
  TEST_API Tensor& operator()(const Tensor& arg0, const Tensor& arg1) const;
  TEST_API Tensor& operator()(const Tensor& arg0,
                              const Tensor& arg1,
                              const Tensor& arg2) const;
  TEST_API void operator()(const std::vector<const Tensor*>& args,
                           const std::vector<const Tensor*>& outputs) const;

  static const char* prefix;

 private:
  Op(const std::string& op_type_name,
     PatternGraph* pattern_graph,
     const std::unordered_map<std::string, Attribute>& attributes,
     const std::unordered_map<std::string, Attribute>& runtime_attributes = {})
      : op_type_name_(op_type_name),
        pattern_graph_(pattern_graph),
        attributes_(attributes),
        runtime_attributes_(runtime_attributes) {}

  std::string op_type_name_;
  PatternGraph* pattern_graph_{nullptr};
  std::unordered_map<std::string, Attribute> attributes_;
  std::unordered_map<std::string, Attribute> runtime_attributes_;

  thread_local static int64_t count;

  friend class DrrPatternContext;
  friend class OpCall;
};

class TEST_API Tensor {
 public:
  static constexpr const char* RESULT_INPUT_NONE_TENSOR_NAME =
      "__@result_input_none_tensor@__";
  static constexpr const char* RESULT_OUTPUT_NONE_TENSOR_NAME =
      "__@result_output_none_tensor@__";
  static constexpr const char* SOURCE_INPUT_NONE_TENSOR_NAME =
      "__@source_input_none_tensor@__";
  static constexpr const char* SOURCE_OUTPUT_NONE_TENSOR_NAME =
      "__@source_output_none_tensor@__";

  bool is_none() const {
    return name_ == RESULT_INPUT_NONE_TENSOR_NAME ||
           name_ == RESULT_OUTPUT_NONE_TENSOR_NAME ||
           name_ == SOURCE_INPUT_NONE_TENSOR_NAME ||
           name_ == SOURCE_OUTPUT_NONE_TENSOR_NAME;
  }

  void Assign(const Tensor& other);

  void operator=(const Tensor& other) const;  // NOLINT

  const std::string& name() const { return name_; }

  void set_name(const std::string& name) { name_ = name; }

  OpCall* producer() const { return producer_; }

  void set_producer(OpCall* producer) { producer_ = producer; }

  const std::unordered_set<const OpCall*>& consumers() const {
    return consumers_;
  }

  void AddConsumer(const OpCall* consumer) { consumers_.insert(consumer); }

 private:
  Tensor(const std::string& name, PatternGraph* pattern_graph)
      : name_(name), pattern_graph_(pattern_graph) {}

  friend class DrrPatternContext;
  friend class Op;

  std::string name_;
  OpCall* producer_{nullptr};
  std::unordered_set<const OpCall*> consumers_;
  PatternGraph* pattern_graph_{nullptr};
};

class TEST_API OpCall {
 public:
  OpCall(const Op* op,
         const std::vector<const Tensor*>& inputs,
         const std::vector<const Tensor*>& outputs)
      : op_name_(op->op_type_name_),
        inputs_(inputs),
        outputs_(outputs),
        attributes_(op->attributes_),
        runtime_attributes_(op->runtime_attributes_) {}

  const std::string& name() const { return op_name_; }

  const std::vector<const Tensor*>& inputs() const { return inputs_; }

  const std::vector<const Tensor*>& outputs() const { return outputs_; }

  const std::unordered_map<std::string, Attribute>& attributes() const {
    return attributes_;
  }

  const std::unordered_map<std::string, Attribute>& runtime_attributes() const {
    return runtime_attributes_;
  }

 private:
  std::string op_name_;
  std::vector<const Tensor*> inputs_;
  std::vector<const Tensor*> outputs_;
  std::unordered_map<std::string, Attribute> attributes_;
  std::unordered_map<std::string, Attribute> runtime_attributes_;
};

class TEST_API ResultPattern {
 public:
  const drr::Op&
  Op(const std::string& op_type,
     const std::unordered_map<std::string, Attribute>& attributes = {},
     const std::unordered_map<std::string, Attribute>& runtime_attributes = {});

  drr::Tensor& Tensor(const std::string& name);

  // Represent the input tensor which is none.
  // Example:
  // instance_norm has follow input tensor : (x, scale, bias), scale and
  // bias are optional(means it may be none).
  // When scale is none, we can write a instance_norm op in drr as follow:
  // res.Op("instance_norm")(res.Tensor("x"), res.InputNoneTensor(),
  // res.Tensor("bias"));
  drr::Tensor& InputNoneTensor();

  // Represent the output tensor which is none.
  // Example:
  // reshape has follow output tensor : (out, xshape), xshape is optional(means
  // it may be none). We can write a reshape op in drr as follow:
  // res.Op("reshape")({res.Tensor("x")}, {res.Tensor("out"),
  // res.OutputNoneTensor()});
  drr::Tensor& OutputNoneTensor();

  Attribute StrAttr(const std::string& value) const;

  Attribute BoolAttr(bool value) const;

  Attribute Int32Attr(int32_t value) const;

  Attribute Int64Attr(int64_t value) const;

  Attribute Float32Attr(float value) const;

  Attribute VectorInt64Attr(const std::vector<int64_t>& value) const;

  Attribute VectorInt32Attr(const std::vector<int32_t>& value) const;

  Attribute VectorFloatAttr(const std::vector<float>& value) const;

  // {"bool", phi::DataType::BOOL},
  // {"uint8", phi::DataType::UINT8},
  // {"int8", phi::DataType::INT8},
  // {"uint16", phi::DataType::UINT16},
  // {"int16", phi::DataType::INT16},
  // {"uint32", phi::DataType::UINT32},
  // {"int32", phi::DataType::INT32},
  // {"uint64", phi::DataType::UINT64},
  // {"int64", phi::DataType::INT64},
  // {"float32", phi::DataType::FLOAT32},
  // {"complex64", phi::DataType::COMPLEX64},
  // {"complex128", phi::DataType::COMPLEX128},
  // {"Undefined", phi::DataType::UNDEFINED},
  // {"pstring", phi::DataType::PSTRING},
  // {"float16", phi::DataType::FLOAT16},
  // {"bfloat16", phi::DataType::BFLOAT16},
  // {"float64", phi::DataType::FLOAT64}};
  Attribute DataTypeAttr(const std::string& value) const;

  // {"cpu", phi::CPUPlace{}},
  // {"gpu", phi::GPUPlace{}},
  // {"gpu_pinned", phi::GPUPinnedPlace{}},
  // {"xpu", phi::XPUPlace{}},
  // {"ipu", phi::IPUPlace{}},
  // {":", phi::CustomPlace{}},
  // {"undefined", phi::Place{}}};
  Attribute PlaceAttr(const std::string& value) const;

  // {"NHWC", phi::DataLayout::kNHWC},
  // {"NCHW", phi::DataLayout::kNCHW},
  // {"Undefined", phi::DataLayout::kAnyLayout},
  // {"ONEDNN", phi::DataLayout::ONEDNN},
  // {"SPARSE_COO", phi::DataLayout::SPARSE_COO},
  // {"SPARSE_CSR", phi::DataLayout::SPARSE_CSR},
  // {"NDHWC", phi::DataLayout::kNDHWC},
  // {"NCDHW", phi::DataLayout::kNCDHW},
  // {"PSTRING_UNION", phi::DataLayout::PSTRING_UNION},
  // {"STRIDED", phi::DataLayout::STRIDED}};
  Attribute DataLayoutAttr(const std::string& value) const;

  Attribute ComputeAttr(const AttrComputeFunc& attr_compute_func) const;

 private:
  friend class SourcePattern;

  explicit ResultPattern(DrrPatternContext* ctx) : ctx_(ctx) {}

  DrrPatternContext* ctx_{nullptr};
};

class TEST_API SourcePattern {
 public:
  drr::ResultPattern ResultPattern() const;

  const drr::Op& Op(
      const std::string& op_type,
      const std::unordered_map<std::string, Attribute>& attributes = {});

  const drr::Tensor& Tensor(const std::string& name);

  Attribute Attr(const std::string& attr_name) const;

  void AddConstraint(const ConstraintFunction& constraint_fn);

  void AddPostProcess(const PostProcessFunction& post_process_fn);

  // Same as a ResultPattern::InputNoneTensor
  drr::Tensor& InputNoneTensor();

  // Same as a ResultPattern::OutputNoneTensor
  drr::Tensor& OutputNoneTensor();

 private:
  friend class DrrPatternContext;
  explicit SourcePattern(DrrPatternContext* ctx) : ctx_(ctx) {}
  DrrPatternContext* ctx_{nullptr};
};

}  // namespace drr
}  // namespace paddle
