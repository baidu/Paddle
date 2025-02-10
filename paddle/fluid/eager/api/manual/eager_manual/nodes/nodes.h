// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/eager/api/utils/global_utils.h"
#include "paddle/fluid/eager/grad_node_info.h"
#include "paddle/fluid/eager/tensor_wrapper.h"
#include "paddle/fluid/imperative/tracer.h"

class Conv2dGradNodeFinal : public egr::GradNodeBase {
 public:
  Conv2dGradNodeFinal() : egr::GradNodeBase() {}
  Conv2dGradNodeFinal(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~Conv2dGradNodeFinal() override = default;

  virtual paddle::small_vector<std::vector<paddle::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::Tensor>,        // NOLINT
                                  egr::kSlotSmallVectorSize>& grads,  // NOLINT
             bool create_graph = false,                               // NOLINT
             bool is_new_grad = false) override;                      // NOLINT
  std::string name() override { return "Conv2dGradNodeFinal"; }

  void ClearTensorWrappers() override {
    input_.clear();
    filter_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<Conv2dGradNodeFinal>(new Conv2dGradNodeFinal(*this));
    VLOG(3) << "Copy Conv2dGradNodeFinal: " << this
            << " to: " << copied_node.get();
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapper_input(const paddle::Tensor& input) {
    input_ = egr::TensorWrapper(input, false);
  }
  void SetTensorWrapper_filter(const paddle::Tensor& filter) {
    filter_ = egr::TensorWrapper(filter, false);
  }

  // SetAttributes
  void SetAttribute_strides(const std::vector<int>& strides) {
    strides_ = strides;
  }
  void SetAttribute_paddings(const std::vector<int>& paddings) {
    paddings_ = paddings;
  }
  void SetAttribute_padding_algorithm(const std::string& padding_algorithm) {
    padding_algorithm_ = padding_algorithm;
  }
  void SetAttribute_groups(const int& groups) { groups_ = groups; }
  void SetAttribute_dilations(const std::vector<int>& dilations) {
    dilations_ = dilations;
  }
  void SetAttribute_data_format(const std::string& data_format) {
    data_format_ = data_format;
  }

 private:
  // TensorWrappers
  egr::TensorWrapper input_;
  egr::TensorWrapper filter_;

  // Attributes
  std::vector<int> strides_;
  std::vector<int> paddings_;
  std::string padding_algorithm_;
  int groups_;
  std::vector<int> dilations_;
  std::string data_format_;
};

class Conv2dDoubleGradNodeFinal : public egr::GradNodeBase {
 public:
  Conv2dDoubleGradNodeFinal() : egr::GradNodeBase() {}
  Conv2dDoubleGradNodeFinal(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~Conv2dDoubleGradNodeFinal() override = default;

  virtual paddle::small_vector<std::vector<paddle::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::Tensor>,        // NOLINT
                                  egr::kSlotSmallVectorSize>& grads,  // NOLINT
             bool create_graph = false,                               // NOLINT
             bool is_new_grad = false) override;                      // NOLINT
  std::string name() override { return "Conv2dDoubleGradNodeFinal"; }

  void ClearTensorWrappers() override {
    input_.clear();
    filter_.clear();
    grad_out_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<Conv2dDoubleGradNodeFinal>(
        new Conv2dDoubleGradNodeFinal(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapper_input(const paddle::Tensor& input) {
    input_ = egr::TensorWrapper(input, false);
  }
  void SetTensorWrapper_filter(const paddle::Tensor& filter) {
    filter_ = egr::TensorWrapper(filter, false);
  }
  void SetTensorWrapper_grad_out(const paddle::Tensor& grad_out) {
    grad_out_ = egr::TensorWrapper(grad_out, false);
  }

  // SetAttributes
  void SetAttribute_strides(const std::vector<int>& strides) {
    strides_ = strides;
  }
  void SetAttribute_paddings(const std::vector<int>& paddings) {
    paddings_ = paddings;
  }
  void SetAttribute_padding_algorithm(const std::string& padding_algorithm) {
    padding_algorithm_ = padding_algorithm;
  }
  void SetAttribute_groups(const int& groups) { groups_ = groups; }
  void SetAttribute_dilations(const std::vector<int>& dilations) {
    dilations_ = dilations;
  }
  void SetAttribute_data_format(const std::string& data_format) {
    data_format_ = data_format;
  }

 private:
  // TensorWrappers
  egr::TensorWrapper input_;
  egr::TensorWrapper filter_;
  egr::TensorWrapper grad_out_;

  // Attributes
  std::vector<int> strides_;
  std::vector<int> paddings_;
  std::string padding_algorithm_;
  int groups_;
  std::vector<int> dilations_;
  std::string data_format_;
};

class AddNGradNodeFinal : public egr::GradNodeBase {
 public:
  AddNGradNodeFinal() : egr::GradNodeBase() {}
  AddNGradNodeFinal(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~AddNGradNodeFinal() override = default;

  virtual paddle::small_vector<std::vector<paddle::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::Tensor>,        // NOLINT
                                  egr::kSlotSmallVectorSize>& grads,  // NOLINT
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "AddNGradNodeFinal"; }

  void ClearTensorWrappers() override {
    for (auto& tw : x_) {
      tw.clear();
    }

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<AddNGradNodeFinal>(new AddNGradNodeFinal(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapper_x(const std::vector<paddle::Tensor>& x) {
    for (const auto& eager_tensor : x) {
      x_.emplace_back(egr::TensorWrapper(eager_tensor, true));
    }
  }

  // SetAttributes

 private:
  // TensorWrappers
  std::vector<egr::TensorWrapper> x_;

  // Attributes
};
class MultiplyGradNode : public egr::GradNodeBase {
 public:
  MultiplyGradNode() : egr::GradNodeBase() {}
  MultiplyGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~MultiplyGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::Tensor>,        // NOLINT
                                  egr::kSlotSmallVectorSize>& grads,  // NOLINT
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "MultiplyGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    y_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<MultiplyGradNode>(new MultiplyGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapper_x(const paddle::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrapper_y(const paddle::Tensor& y) {
    y_ = egr::TensorWrapper(y, false);
  }

  void SetTensorWrapperNoNeedBuffer_x(const paddle::Tensor& x) {
    x_ = egr::TensorWrapper(x, true);
  }
  void SetTensorWrapperNoNeedBuffer_y(const paddle::Tensor& y) {
    y_ = egr::TensorWrapper(y, true);
  }

  // SetAttributes
  void SetAttribute_axis(const int& axis) { axis_ = axis; }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper y_;

  // Attributes
  int axis_ = -1;
};

class MultiplyDoubleGradNode : public egr::GradNodeBase {
 public:
  MultiplyDoubleGradNode() : egr::GradNodeBase() {}
  MultiplyDoubleGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~MultiplyDoubleGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::Tensor>,        // NOLINT
                                  egr::kSlotSmallVectorSize>& grads,  // NOLINT
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "MultiplyDoubleGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    y_.clear();
    grad_out_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<MultiplyDoubleGradNode>(
        new MultiplyDoubleGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapper_x(const paddle::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrapper_y(const paddle::Tensor& y) {
    y_ = egr::TensorWrapper(y, false);
  }
  void SetTensorWrapper_grad_out(const paddle::Tensor& grad_out) {
    grad_out_ = egr::TensorWrapper(grad_out, false);
  }

  // SetAttributes
  void SetAttribute_axis(const int& axis) { axis_ = axis; }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper y_;
  egr::TensorWrapper grad_out_;

  // Attributes
  int axis_ = -1;
};

class SyncBatchNormGradNode : public egr::GradNodeBase {
 public:
  SyncBatchNormGradNode() : egr::GradNodeBase() {}
  SyncBatchNormGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~SyncBatchNormGradNode() {
    egr::Controller::Instance().EraseForceSequentialNodes(this);
  }

  virtual paddle::small_vector<std::vector<paddle::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,  // NOLINT
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "SyncBatchNormGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    scale_.clear();
    bias_.clear();
    saved_mean_.clear();
    saved_variance_.clear();
    reserve_space_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<SyncBatchNormGradNode>(
        new SyncBatchNormGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapper_x(const paddle::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrapper_scale(const paddle::Tensor& scale) {
    scale_ = egr::TensorWrapper(scale, false);
  }
  void SetTensorWrapper_bias(const paddle::Tensor& bias) {
    bias_ = egr::TensorWrapper(bias, false);
  }
  void SetTensorWrapper_saved_mean(const paddle::Tensor& saved_mean) {
    saved_mean_ = egr::TensorWrapper(saved_mean, false);
  }
  void SetTensorWrapper_saved_variance(const paddle::Tensor& saved_variance) {
    saved_variance_ = egr::TensorWrapper(saved_variance, false);
  }
  void SetTensorWrapper_reserve_space(const paddle::Tensor& reserve_space) {
    reserve_space_ = egr::TensorWrapper(reserve_space, false);
  }

  // SetAttributes
  void SetAttribute_momentum(const float& momentum) { momentum_ = momentum; }
  void SetAttribute_epsilon(const float& epsilon) { epsilon_ = epsilon; }
  void SetAttribute_data_layout(const std::string& data_layout) {
    data_layout_ = data_layout;
  }
  void SetAttribute_is_test(const bool& is_test) { is_test_ = is_test; }
  void SetAttribute_use_global_stats(const bool& use_global_stats) {
    use_global_stats_ = use_global_stats;
  }
  void SetAttribute_trainable_statistics(const bool& trainable_statistics) {
    trainable_statistics_ = trainable_statistics;
  }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper scale_;
  egr::TensorWrapper bias_;
  egr::TensorWrapper saved_mean_;
  egr::TensorWrapper saved_variance_;
  egr::TensorWrapper reserve_space_;

  // Attributes
  float momentum_;
  float epsilon_;
  std::string data_layout_;
  bool is_test_;
  bool use_global_stats_;
  bool trainable_statistics_;
};

class ReshardGradNode : public egr::GradNodeBase {
 public:
  ReshardGradNode() : egr::GradNodeBase() {
    VLOG(3) << " Construct ReshardGrad Node.";
  }

  ReshardGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {
    VLOG(3) << " Construct ReshardGrad Node, bwd_in_slot_num: "
            << bwd_in_slot_num << ", bwd_out_slot_num: " << bwd_out_slot_num;
  }

  ~ReshardGradNode() override { VLOG(3) << " Destruct ReshardGrad Node."; }

  virtual paddle::small_vector<std::vector<paddle::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,  // NOLINT
             bool create_graph = false,
             bool is_new_grad = false) override;

  void ClearTensorWrappers() override {
    input_.clear();
    SetIsTensorWrappersCleared(true);
  }

  std::string name() override { return "ReshardGradNode"; }

  std::shared_ptr<GradNodeBase> Copy() const override {
    {
      auto copied_node =
          std::shared_ptr<ReshardGradNode>(new ReshardGradNode(*this));
      return copied_node;
    }
  }

  // SetTensorWrapperX
  // Only input's meta is needed.
  void SetTensorWrapperNoNeedBuffer_Input(const paddle::Tensor& input) {
    input_ = egr::TensorWrapper(input, true);
  }

 private:
  // TensorWrappers
  egr::TensorWrapper input_;
};

class DtensorToLocalGradNode : public egr::GradNodeBase {
 public:
  DtensorToLocalGradNode() : egr::GradNodeBase() {
    VLOG(3) << " Construct DtensorToLocalGradNode Node.";
  }

  DtensorToLocalGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {
    VLOG(3) << " Construct DtensorToLocalGradNode Node, bwd_in_slot_num: "
            << bwd_in_slot_num << ", bwd_out_slot_num: " << bwd_out_slot_num;
  }

  ~DtensorToLocalGradNode() override {
    VLOG(3) << " Destruct DtensorToLocalGradNode Node.";
  }

  virtual paddle::small_vector<std::vector<paddle::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,  // NOLINT
             bool create_graph = false,
             bool is_new_grad = false) override;

  void ClearTensorWrappers() override {
    input_.clear();
    SetIsTensorWrappersCleared(true);
  }

  std::string name() override { return "DtensorToLocalGradNode"; }

  std::shared_ptr<GradNodeBase> Copy() const override {
    {
      auto copied_node = std::shared_ptr<DtensorToLocalGradNode>(
          new DtensorToLocalGradNode(*this));
      return copied_node;
    }
  }

  // SetTensorWrapperX
  void SetTensorWrapperNoNeedBuffer_Input(const paddle::Tensor& input) {
    input_ = egr::TensorWrapper(input, true);
  }

 private:
  // TensorWrappers
  egr::TensorWrapper input_;
};

class DtensorFromLocalGradNode : public egr::GradNodeBase {
 public:
  DtensorFromLocalGradNode() : egr::GradNodeBase() {
    VLOG(3) << " Construct DtensorFromLocalGradNode Node.";
  }

  DtensorFromLocalGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {
    VLOG(3) << " Construct DtensorFromLocalGradNode Node, bwd_in_slot_num: "
            << bwd_in_slot_num << ", bwd_out_slot_num: " << bwd_out_slot_num;
  }

  ~DtensorFromLocalGradNode() override {
    VLOG(3) << " Destruct DtensorFromLocalGradNode Node.";
  }

  virtual paddle::small_vector<std::vector<paddle::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,  // NOLINT
             bool create_graph = false,
             bool is_new_grad = false) override;

  void ClearTensorWrappers() override {
    output_.clear();
    SetIsTensorWrappersCleared(true);
  }

  std::string name() override { return "DtensorFromLocalGradNode"; }

  std::shared_ptr<GradNodeBase> Copy() const override {
    {
      auto copied_node = std::shared_ptr<DtensorFromLocalGradNode>(
          new DtensorFromLocalGradNode(*this));
      return copied_node;
    }
  }

  // SetTensorWrapperX
  void SetTensorWrapperNoNeedBuffer_Output(const paddle::Tensor& output) {
    output_ = egr::TensorWrapper(output, true);
  }

 private:
  // TensorWrappers
  egr::TensorWrapper output_;
};

namespace sparse {
class SyncBatchNormGradNode : public egr::GradNodeBase {
 public:
  SyncBatchNormGradNode() : egr::GradNodeBase() {}
  SyncBatchNormGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~SyncBatchNormGradNode() {
    egr::Controller::Instance().EraseForceSequentialNodes(this);
  }

  virtual paddle::small_vector<std::vector<paddle::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,  // NOLINT
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "SyncBatchNormGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    scale_.clear();
    bias_.clear();
    saved_mean_.clear();
    saved_variance_.clear();
    reserve_space_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<SyncBatchNormGradNode>(
        new SyncBatchNormGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapper_x(const paddle::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrapper_scale(const paddle::Tensor& scale) {
    scale_ = egr::TensorWrapper(scale, false);
  }
  void SetTensorWrapper_bias(const paddle::Tensor& bias) {
    bias_ = egr::TensorWrapper(bias, false);
  }
  void SetTensorWrapper_saved_mean(const paddle::Tensor& saved_mean) {
    saved_mean_ = egr::TensorWrapper(saved_mean, false);
  }
  void SetTensorWrapper_saved_variance(const paddle::Tensor& saved_variance) {
    saved_variance_ = egr::TensorWrapper(saved_variance, false);
  }
  void SetTensorWrapper_reserve_space(const paddle::Tensor& reserve_space) {
    reserve_space_ = egr::TensorWrapper(reserve_space, false);
  }

  // SetAttributes
  void SetAttribute_momentum(const float& momentum) { momentum_ = momentum; }
  void SetAttribute_epsilon(const float& epsilon) { epsilon_ = epsilon; }
  void SetAttribute_data_layout(const std::string& data_layout) {
    data_layout_ = data_layout;
  }
  void SetAttribute_is_test(const bool& is_test) { is_test_ = is_test; }
  void SetAttribute_use_global_stats(const bool& use_global_stats) {
    use_global_stats_ = use_global_stats;
  }
  void SetAttribute_trainable_statistics(const bool& trainable_statistics) {
    trainable_statistics_ = trainable_statistics;
  }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper scale_;
  egr::TensorWrapper bias_;
  egr::TensorWrapper saved_mean_;
  egr::TensorWrapper saved_variance_;
  egr::TensorWrapper reserve_space_;

  // Attributes
  float momentum_;
  float epsilon_;
  std::string data_layout_;
  bool is_test_;
  bool use_global_stats_;
  bool trainable_statistics_;
};

class MultiplyGradNode : public egr::GradNodeBase {
 public:
  MultiplyGradNode() : egr::GradNodeBase() {}
  MultiplyGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~MultiplyGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::Tensor>,        // NOLINT
                                  egr::kSlotSmallVectorSize>& grads,  // NOLINT
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "MultiplyGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    y_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<MultiplyGradNode>(new MultiplyGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapper_x(const paddle::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrapper_y(const paddle::Tensor& y) {
    y_ = egr::TensorWrapper(y, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper y_;

  // Attributes
};

}  // namespace sparse
