// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/eager/autograd_meta.h"
#include "paddle/fluid/eager/eager_tensor.h"
#include "paddle/fluid/eager/fwd/forward_grad.h"
#include "paddle/fluid/eager/grad_node_info.h"
#include "paddle/phi/api/all.h"
#include "paddle/phi/api/lib/kernel_dispatch.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_tensor.h"
#include "paddle/utils/test_macros.h"

namespace egr {

class TensorWrapper;

/**
 * EagerUtils is utils used to do some static conversion or autograd
 * members access, this class is designed to be a full static functional
 * utils class
 **/

template <typename ElementType>
class IterHelper {
  virtual void visit(ElementType element) = 0;

  virtual void visit(std::vector<ElementType>* elements) {
    for (auto element : *elements) visit(element);
  }

  virtual void visit(const std::vector<ElementType>& elements) {
    for (auto element : elements) visit(element);
  }

  template <typename... Args>
  void apply() {}

 public:
  template <typename T, typename... Args>
  void apply(T&& arg, Args&&... args) {
    visit(std::forward<T>(arg));
    return apply(std::forward<Args>(args)...);
  }
  virtual ~IterHelper() = default;
};

class ComputeRequireGradIter : public IterHelper<AutogradMeta*> {
 public:
  bool RequireGrad() { return require_grad_; }

 private:
  void visit(AutogradMeta* element) override {
    // Dispensable Tensors feeds in nullptr autograd_meta
    if (!element) return;

    if (!require_grad_) {
      bool stop_gradient = element->StopGradient();
      if (!stop_gradient) {
        require_grad_ = true;
      }
    }
  }

  bool require_grad_ = false;
};

class ComputeRequireFwdGradIter : public IterHelper<AutogradMeta*> {
 public:
  bool RequireFwdGrad() { return require_fwd_grad_; }

 private:
  void visit(AutogradMeta* element) override {
    // Dispensable Tensors feeds in nullptr autograd_meta
    if (!element) return;

    if (!require_fwd_grad_) {
      bool require_fwd_grad = element->fw_grad(0).has_allocation();
      if (require_fwd_grad) {
        require_fwd_grad_ = true;
      }
    }
  }

  bool require_fwd_grad_ = false;
};

class PassStopGradientIter : public IterHelper<AutogradMeta*> {
 public:
  void SetStopGradient(bool stop_gradient) { stop_gradient_ = stop_gradient; }

 private:
  void visit(AutogradMeta* element) override {
    if (!element) {
      // TODO(jiabin): Add Tensor name here when we supported.
      VLOG(2) << "Tensor is NULL";
      return;
    }
    element->SetStopGradient(stop_gradient_);
  }

  bool stop_gradient_ = true;
};

class SetGradOutputDistAttrIter : public IterHelper<paddle::Tensor*> {
 public:
  explicit SetGradOutputDistAttrIter(
      const paddle::small_vector<std::vector<GradSlotMeta>,
                                 kSlotSmallVectorSize>& out_meta,
      const paddle::small_vector<size_t, kSlotSmallVectorSize>& out_indexes,
      const phi::distributed::ProcessMesh& mesh)
      : out_meta_(out_meta), out_indexes_{out_indexes}, mesh_(mesh) {}

 private:
  void visit_element(paddle::Tensor* element, const GradSlotMeta& meta);
  void visit(paddle::Tensor* element) override;
  void visit(const std::vector<paddle::Tensor*>& elements) override;

  const paddle::small_vector<std::vector<GradSlotMeta>, kSlotSmallVectorSize>&
      out_meta_;
  const paddle::small_vector<size_t, kSlotSmallVectorSize>& out_indexes_;
  const phi::distributed::ProcessMesh& mesh_;

  int cur_pos_{0};
};

class TEST_API EagerUtils {
 public:
  /**
   * We have to use autograd_meta and multi_autograd_meta to initialize
   * autograd_meta for tensor, since we can't init it in
   * egr::EagerVariable's
   * constructor (it's abstract class there)
   *
   * **/
  static AutogradMeta* autograd_meta(paddle::Tensor* target);

  static std::vector<AutogradMeta*> autograd_meta(
      std::vector<paddle::Tensor>* targets);

  static std::vector<AutogradMeta*> autograd_meta(
      std::vector<paddle::Tensor*>* targets);

  static std::pair<size_t, size_t> OutRankInfo(const paddle::Tensor& target);

  static std::shared_ptr<GradNodeBase> grad_node(const paddle::Tensor& target);
  static paddle::Tensor* mutable_grad(const paddle::Tensor& target);

  // Set history is used to set backward info during forward process, it will
  // set forward var's autograd meta's grad node as current backward node.
  static void SetHistory(std::vector<AutogradMeta*>* autograd_metas,
                         const std::shared_ptr<GradNodeBase>& grad_node);
  static void SetHistory(AutogradMeta* autograd_meta,
                         const std::shared_ptr<GradNodeBase>& grad_node);

  // This is used for Set vector of tensors' rank
  static void SetOutRankWithSlot(std::vector<AutogradMeta*>* targets,
                                 size_t slot_id);
  static void SetOutRankWithSlot(AutogradMeta* target, size_t slot_id);

  // This method will return an AutogradMeta pointer unsafely.
  static AutogradMeta* nullable_autograd_meta(const paddle::Tensor& target);
  static AutogradMeta* nullable_autograd_meta(
      const paddle::optional<paddle::Tensor>& target);
  static std::vector<AutogradMeta*> nullable_autograd_meta(
      const std::vector<paddle::Tensor>& targets);
  static std::vector<AutogradMeta*> nullable_autograd_meta(
      const paddle::optional<std::vector<paddle::Tensor>>& targets);
  static std::vector<AutogradMeta*> nullable_autograd_meta(
      const std::vector<paddle::Tensor*>& targets);
  static AutogradMeta* unsafe_autograd_meta(const paddle::Tensor& target);
  static std::vector<AutogradMeta*> unsafe_autograd_meta(
      const std::vector<paddle::Tensor>& targets);

  template <typename T, typename... Args>
  static bool ComputeRequireGrad(T trace_backward, Args&&... args) {
    if (!trace_backward) {
      VLOG(6) << "Do not require grad because trace_backward = false";
      return false;
    }

    auto iter = ComputeRequireGradIter();
    iter.apply(std::forward<Args>(args)...);

    return iter.RequireGrad();
  }

  template <typename... Args>
  static bool ComputeRequireFwdGrad(Args&&... args) {
    auto iter = ComputeRequireFwdGradIter();
    iter.apply(std::forward<Args>(args)...);

    return iter.RequireFwdGrad();
  }

  static const paddle::Tensor& toNonOptTensor(const paddle::Tensor& t) {
    return t;
  }

  static const paddle::Tensor& toNonOptFwGrad(const paddle::Tensor& t) {
    return t.has_allocation() ? t._fw_grad(/*level */ 0)
                              : ForwardGrad::undef_grad();
  }

  static const paddle::Tensor& toNonOptPrimal(const paddle::Tensor& t) {
    if (t.has_allocation()) {
      // if (t->unsafeGetTensorImpl()->is_wrapped_number()) {
      //   return *t;
      // } // 当t是一个单数字时，直接返回自己，因为数字本身的primal就是自己
      return t._fw_primal(/* level */ 0);  // 否则返回0级primal
    }
    return ForwardGrad::undef_grad();  // 返回一个空的primal
  }

  template <typename T, typename... Args>
  static void PassStopGradient(T stop_gradient, Args&&... args) {
    auto iter = PassStopGradientIter();
    iter.SetStopGradient(stop_gradient);
    iter.apply(std::forward<Args>(args)...);
  }

  // If and only if the tensor holds an AccumulationNode
  // Then it's treated as a leaf tensor
  static bool IsLeafTensor(const paddle::Tensor& target);

  static void CheckInplace(const paddle::Tensor& target,
                           const AutogradMeta* autograd_meta,
                           bool require_any_grad);

  // View Strategy
  static void HandleViewBetweenInputAndOutput(
      const std::shared_ptr<EagerVariable>& input_var,
      const std::shared_ptr<EagerVariable>& view_output_var);
  static void HandleViewBetweenInputAndOutput(
      const paddle::Tensor& input_tensor, paddle::Tensor* view_output_tensor);

  // TensorWrapper Utils
  static paddle::Tensor RecoverTensorWrapper(TensorWrapper* tw);
  static std::vector<paddle::Tensor> RecoverTensorWrapper(
      std::vector<TensorWrapper>* tw);

  // Intermediate needed remove this once we don't need legacy
  // Inner Method
  static std::shared_ptr<egr::EagerVariable> TrySyncToVar(
      const paddle::Tensor& tensor);
  // Basic Input
  static std::vector<std::shared_ptr<egr::EagerVariable>> TrySyncToVars(
      const paddle::Tensor& tensor);
  // Basic Output
  static std::vector<std::shared_ptr<egr::EagerVariable>> TrySyncToVars(
      paddle::Tensor* tensor);
  // Multi Output
  static std::vector<std::shared_ptr<egr::EagerVariable>> TrySyncToVars(
      const std::vector<paddle::Tensor*>& tensors);
  // Multi Input
  static std::vector<std::shared_ptr<egr::EagerVariable>> TrySyncToVars(
      const std::vector<paddle::Tensor>& tensors);
  // Construct empty output
  static std::vector<std::shared_ptr<EagerVariable>> CreateVars(
      const size_t num);
  // Construct Tensor From var
  static std::vector<paddle::Tensor> GetOutputs(
      const std::vector<std::shared_ptr<EagerVariable>>& outs);
  static paddle::Tensor GetOutput(const std::shared_ptr<EagerVariable>& out);
  static void GetOutput(const std::shared_ptr<EagerVariable>& out,
                        paddle::Tensor* out_var);
  static void GetOutputs(
      const std::vector<std::shared_ptr<EagerVariable>>& outs,
      std::vector<paddle::Tensor>* result);
  static void GetOutputs(
      const std::vector<std::shared_ptr<EagerVariable>>& outs,
      const std::vector<paddle::Tensor*>& out_var);
  static void GetOutputs(const std::shared_ptr<EagerVariable>& out,
                         std::vector<paddle::Tensor>* result);
  static void GetOutputs(const std::shared_ptr<EagerVariable>& out,
                         const std::vector<paddle::Tensor*>& out_var);

  static void Output2Result(const std::vector<paddle::Tensor*>& out_var,
                            std::vector<paddle::Tensor>* result);

  static std::shared_ptr<egr::GradNodeBase> GetGradAccumulationNode(
      const paddle::Tensor& tensor);

  /**
   * Fill Zero
   * **/
  static void FillZeroForEmptyOptionalGradInput(
      std::vector<paddle::Tensor>* in_grads,
      const std::vector<GradSlotMeta>& grad_in_metas);
  static void FillZeroForEmptyOptionalGradOutput(
      std::vector<paddle::Tensor>* out_grads,
      const std::vector<GradSlotMeta>& grad_out_metas);
  static void FillZeroForEmptyGradInput(paddle::Tensor* in_grad,
                                        const GradSlotMeta& grad_in_meta);
  static void FillZeroForEmptyOptionalGradInput(
      paddle::Tensor* in_grad, const GradSlotMeta& grad_in_meta);
  static void FillZeroForEmptyGradInput(
      std::vector<paddle::Tensor>* in_grads,
      const std::vector<GradSlotMeta>& grad_in_metas);

  /**
   * Set DistAttr
   */
  template <typename... Args>
  static void SetGradOutputDistAttr(
      const paddle::small_vector<std::vector<GradSlotMeta>,
                                 kSlotSmallVectorSize>& out_metas,
      const paddle::small_vector<size_t, kSlotSmallVectorSize>& out_indexes,
      const phi::distributed::ProcessMesh& mesh,
      Args&&... args) {
    SetGradOutputDistAttrIter(out_metas, out_indexes, mesh)
        .apply(std::forward<Args>(args)...);
  }

  /**
   * Print Input Output (level 0 means least info, level 2 means most info)
   * **/
  static std::string TensorStr(const paddle::Tensor& t);
  static std::string GradNodeStr(const egr::GradNodeBase& node);

  static std::string GradNodeStr(const paddle::Tensor& t);

  static std::string TensorStr(const std::vector<paddle::Tensor>& tensors);

  static std::string TensorStr(const paddle::optional<paddle::Tensor>& t);

  static std::string TensorStr(
      const paddle::optional<std::vector<paddle::Tensor>>& tensors);
};

using paddle::experimental::detail::ArgsIterator;

struct DistTensorTypeParser : ArgsIterator<DistTensorTypeParser> {
  bool result = false;
  const phi::distributed::ProcessMesh** mesh = nullptr;

  explicit DistTensorTypeParser(const phi::distributed::ProcessMesh** m)
      : mesh(m) {}

  bool short_circuit() { return result; }

  void operator()(const paddle::Tensor& x);
  void operator()(const paddle::optional<paddle::Tensor>& x);
  void operator()(const std::vector<paddle::Tensor>& x);
  void operator()(const paddle::optional<std::vector<paddle::Tensor>>& x);

  // skip other type args, these args don't used in kernel selection
  template <typename T>
  void operator()(const T& x) {
    // do nothing
  }
};

struct DistTensorConverter : ArgsIterator<DistTensorConverter> {
  const phi::distributed::ProcessMesh* mesh = nullptr;

  explicit DistTensorConverter(const phi::distributed::ProcessMesh* m) {
    PADDLE_ENFORCE_NE(
        m,
        nullptr,
        common::errors::InvalidArgument(
            "Input mesh of DistTensorConverter() shouldn't be nullptr."));
    mesh = m;
  }

  void convert(paddle::Tensor* x);
  void operator()(paddle::Tensor* x);
  void operator()(paddle::optional<paddle::Tensor>* x);
  void operator()(std::vector<paddle::Tensor>* x);
  void operator()(paddle::optional<std::vector<paddle::Tensor>>* x);

  // skip other type args, these args don't used in kernel selection
  template <typename T>
  void operator()(const T& x) {
    // do nothing
  }
};

template <typename... Args>
bool InputsContainDistTensor(const phi::distributed::ProcessMesh** mesh,
                             const Args&... args) {
  return DistTensorTypeParser(mesh).apply(args...).result;
}

template <typename... Args>
void ConvertAllInputsToDistTensor(const phi::distributed::ProcessMesh* mesh,
                                  Args&... args) {
  PADDLE_ENFORCE_NE(
      mesh,
      nullptr,
      common::errors::InvalidArgument("Input mesh should not be nullptr."));
  DistTensorConverter(mesh).apply(&args...);
}

void ConvertToDistTensor(paddle::Tensor* x,
                         const phi::distributed::ProcessMesh* mesh);

}  // namespace egr
