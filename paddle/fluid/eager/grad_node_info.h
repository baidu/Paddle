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

#include <memory>

#include "paddle/fluid/eager/api/utils/global_utils.h"
#include "paddle/fluid/eager/eager_tensor.h"
#include "paddle/fluid/eager/hooks.h"
#include "paddle/phi/api/all.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/reshard_utils.h"
#include "paddle/utils/test_macros.h"

namespace egr {
/**
 * GradNodeBase is base class of all grad node, which is what should be used by
 * eager execution, we define most of backward autograd members here, and for
 * each Operator, they should hold their own forward Inputs as TensorWrapper.
 *
 * The GradNodeBase will be held in autograd_meta, and it is also a member of
 * Edge, which indicates the edge of backward graph.
 *
 * TODO(yangzhanlue): GradNodeBase will also in charge of get the correct input
 * from GradOpDescMaker to GradNodeBase.
 *
 * NOTE: GradNodeBase has a method named run, this method should be overridden
 *by the specific derived class, it will prepare backward inputs and double
 * backward's depends. Then, it will call C++ API of backward kernel functions
 * to finish backward computation.
 *
 * NOTE: GradNodeBase holds its own inputs and Outputs
 *
 * Edge is defined to describe depend of backward, an Edge is what linked
 * between two node, it should contain a Node and rank of this Node (this is
 * used to indicate which input of grad this edge belong).
 **/
class AutogradMeta;
class GradNodeBase;

class Edge {
 public:
  // Default constructor for Edges in order to construct it for AutogradMeta
  Edge() : in_slot_id_(0), in_rank_(0), grad_node_(nullptr) {}

  // In real use cases we should create Edge from grad node and input rank which
  // indicate which edge it is.
  // Since we have slot design in operators we will have to locate an edge with
  // slot and rank.
  Edge(const std::shared_ptr<GradNodeBase>& grad_node,
       size_t in_slot_id,
       size_t in_rank)
      : in_slot_id_(in_slot_id), in_rank_(in_rank), grad_node_(grad_node) {}

  Edge(const std::shared_ptr<GradNodeBase>& grad_node,
       const std::pair</* slot_id */ size_t, /* rank */ size_t>& rank_info)
      : in_slot_id_(rank_info.first),
        in_rank_(rank_info.second),
        grad_node_(grad_node) {}

  GradNodeBase* GetGradNode() const { return grad_node_.get(); }

  std::shared_ptr<GradNodeBase> GetMutableGradNode() const {
    return grad_node_;
  }

  void SetGradNode(const std::shared_ptr<GradNodeBase>& node) {
    VLOG(7) << "Resetting Edge's Grad Node";
    grad_node_ = node;
  }

  std::pair<size_t, size_t> GetEdgeRankInfo() const {
    return std::make_pair(in_slot_id_, in_rank_);
  }

  void SetEdgeRankInfo(size_t slot_id, size_t in_rank) {
    in_slot_id_ = slot_id;
    in_rank_ = in_rank;
  }

  void SetEdgeRankInfo(
      const std::pair</* slot_id */ size_t, /* rank */ size_t>& edge_rank) {
    in_slot_id_ = edge_rank.first;
    in_rank_ = edge_rank.second;
  }

  // Currently we use grad_node_ to identify if a edge is initialized.
  bool IsInitialized() const {
    if (!grad_node_) {
      return false;
    } else {
      if (!(grad_node_.get())) {
        return false;
      } else {
        return true;
      }
    }
  }

  void Clear() {
    grad_node_.reset();
    in_slot_id_ = 0;
    in_rank_ = 0;
  }

 private:
  size_t in_slot_id_;
  size_t in_rank_;
  std::shared_ptr<GradNodeBase> grad_node_{nullptr};
};

/**
 * GradSlotMeta is used to Record Forward Tensor info to backward, since paddle
 * has lots of operators whose backward logic is depends on if it has some
 * specific inputs or outputs. So, we need a meta info to record it's needs.
 **/
class GradSlotMeta {
 public:
  GradSlotMeta() = default;
  bool IsStopGradient() const { return stop_gradient_; }
  void SetStopGradient(bool stop_gradient = true) {
    stop_gradient_ = stop_gradient;
  }

  void SetTensorMeta(const phi::DenseTensorMeta& meta) {
    meta_ = std::make_shared<phi::DenseTensorMeta>(meta);
  }
  bool HasTensorMeta() const { return meta_ && meta_.get(); }
  const phi::DenseTensorMeta& GetTensorMeta() const {
    if (!HasTensorMeta()) {
      PADDLE_THROW(common::errors::Fatal(
          "meta_ of GradSlotMeta has not been initialized yet."
          "You're expected to check Edge availability with HasTensorMeta()"
          "before calling GetTensorMeta() interface."));
    }
    return *meta_.get();
  }

  void SetPlace(const phi::Place& place) { place_ = place; }
  const phi::Place& GetPlace() const { return place_; }

  void SetEdge(const Edge& edge) { adj_edge_ = edge; }
  void SetEdge(
      const std::shared_ptr<GradNodeBase>& grad_node,
      const std::pair</* slot_id */ size_t, /* rank */ size_t>& rank_info) {
    adj_edge_.SetGradNode(grad_node);
    adj_edge_.SetEdgeRankInfo(rank_info);
  }
  Edge& GetMutableEdge() { return adj_edge_; }
  const Edge& GetEdge() const { return adj_edge_; }

  const phi::distributed::TensorDistAttr& DistAttr() const {
    return dist_attr_;
  }

  void SetDistAttr(const phi::distributed::TensorDistAttr& dist_attr) {
    dist_attr_ = dist_attr;
    is_dist_meta_ = true;
  }

  const phi::DDim& DistTensorGlobalDims() const {
    return dist_tensor_global_dims_;
  }

  void SetDistTensorGlobalDims(const phi::DDim& dims) {
    dist_tensor_global_dims_ = dims;
    is_dist_meta_ = true;
  }

  bool IsDistMeta() const { return is_dist_meta_; }

 private:
  bool stop_gradient_{false};
  phi::Place place_;
  std::shared_ptr<phi::DenseTensorMeta> meta_ = nullptr;
  Edge adj_edge_;
  // For dygraph semi-auto parallel
  // Save the dist attr of the forward input Tensor for proper resharding
  // operation when compute the input Tensor's gradient
  phi::distributed::TensorDistAttr dist_attr_;
  phi::DDim dist_tensor_global_dims_;
  bool is_dist_meta_{false};
};

class GradNodeBase {
 public:
  GradNodeBase() { VLOG(7) << "Construct GradNodeBase"; }
  TEST_API GradNodeBase(size_t bwd_in_slot_num, size_t bwd_out_slot_num);
  // TODO(jiabin): Should we have other constructor here?
  virtual ~GradNodeBase() { VLOG(7) << "Destruct GradNodeBase"; }

  /**
   * operator() designed to contain the real backward execution logic, it should
   * be overridden by derived class defined for each operator. It accepts a
   * vector of Tensor which contains grads input of current operator
   *
   * Note: why we need backward inputs and outputs construct as vector of vector
   * of paddle::Tensor?
   * Since all of paddle op composite in form of {"Slot name ", vector<Var>},
   * so, vector of vector is better choice to fit this format.
   * **/
  virtual paddle::small_vector<std::vector<paddle::Tensor>,
                               kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::Tensor>,
                                  kSlotSmallVectorSize>& grads,  // NOLINT
             bool create_graph = false,
             bool is_new_grad = false) = 0;

  virtual void ClearTensorWrappers() = 0;

  /**
   * Self-Copy interface designed for use in DoubleGrad
   * **/
  virtual std::shared_ptr<GradNodeBase> Copy() const = 0;

  // adj_edges were moved inside OutputMeta(), so no available direct access
  // from GradNodeBase.
  // To access Edges, get GradSlotMeta by calling OutputMeta(), then use
  // slot_meta.GetEdge()

  /**
   * Get Input Meta of current Grad node**/
  TEST_API const
      paddle::small_vector<std::vector<GradSlotMeta>, kSlotSmallVectorSize>&
      InputMeta() const;
  /**
   * Get Output Meta of current Grad node**/
  TEST_API const
      paddle::small_vector<std::vector<GradSlotMeta>, kSlotSmallVectorSize>&
      OutputMeta() const;

  paddle::small_vector<std::vector<GradSlotMeta>, kSlotSmallVectorSize>&
  MutableOutputMeta();
  /**
   * Set bwd ins and outs info with forward vars
   * **/

  void SetGradInMeta(const std::vector<paddle::Tensor>& fwd_out,
                     size_t slot_rank);
  void SetGradInMeta(const paddle::Tensor& fwd_out, size_t slot_rank);
  void SetGradInMeta(const std::vector<paddle::Tensor*>& fwd_out,
                     size_t slot_rank);
  void SetGradOutMeta(const std::vector<paddle::Tensor>& fwd_in,
                      size_t slot_rank);
  void SetGradOutMeta(const std::vector<const paddle::Tensor*>& fwd_in,
                      size_t slot_rank);
  TEST_API void SetGradOutMeta(const paddle::Tensor& fwd_in, size_t slot_rank);
  void SetGradOutMeta(const paddle::Tensor& fwd_in,
                      const AutogradMeta* fwd_in_other,
                      size_t slot_rank);
  /**
   * Default setters for Grad in/out meta this should be used for same special
   * Node which will not create by user
   * **/
  TEST_API void SetDefaultGradInOutMeta();
  /**
   * Register GradientHook
   * **/
  int64_t RegisterGradientHook(size_t slot_id,
                               size_t rank,
                               std::shared_ptr<egr::TensorHook>&& hook);

  /**
   * Remove GradientHook
   * **/
  bool RemoveGradientHook(const int64_t& hook_id) {
    auto remove_cnt = gradient_hooks_.erase(hook_id);
    if (remove_cnt == 0) {
      return false;
    }
    return true;
  }

  std::vector<std::shared_ptr<egr::GradNodeBase>> NextFunctions();

  uintptr_t GetPtr() const;

  /**
   * Apply GradientHook
   * **/
  inline bool GradientHooksRegistered() { return !gradient_hooks_.empty(); }

  std::map<int64_t, std::tuple<size_t, size_t, std::shared_ptr<TensorHook>>>
  GetGradientHookFunctions() {
    VLOG(7) << "GetGradientHookFunctions ";
    return gradient_hooks_;
  }

  void SetGradientHookFunctions(
      std::map<int64_t, std::tuple<size_t, size_t, std::shared_ptr<TensorHook>>>
          hooks) {
    VLOG(7) << "SetGradientHookFunctions ";
    gradient_hooks_ = hooks;
  }

  paddle::small_vector<std::vector<paddle::Tensor>, kSlotSmallVectorSize>
  ApplyGradientHooks(const paddle::small_vector<std::vector<paddle::Tensor>,
                                                kSlotSmallVectorSize>& tensors);

  /**
   * Handle Complex - Real Type Promotion
   * **/
  void HandleComplexGradToRealGrad(
      paddle::small_vector<std::vector<paddle::Tensor>, kSlotSmallVectorSize>*
          out_grads);
  bool NeedComplexToRealConversion() { return need_complex_to_real_; }

  virtual std::string name() { return "GradNodeBase"; }

  /**
   * The following interfaces are designed for no_need_buffer
   * **/
  bool IsTensorWrappersCleared() { return is_tensor_wrappers_cleared_; }

  void SetIsTensorWrappersCleared(bool is_tensor_wrappers_cleared) {
    is_tensor_wrappers_cleared_ = is_tensor_wrappers_cleared;
  }

  void SetForwardTrace(std::string trace) { forward_trace_ = trace; }

  std::string GetForwardTrace() { return forward_trace_; }

  /**
   * The following interfaces are designed for auto parallel
   * **/
  bool IsRunAutoParallel() const { return is_run_auto_parallel_; }
  void SetIsRunAutoParallel(bool is_run_auto_parallel) {
    is_run_auto_parallel_ = is_run_auto_parallel;
  }

 private:
  // bwd_out_meta_ is used to record Grad output info for backward
  paddle::small_vector<std::vector<GradSlotMeta>, kSlotSmallVectorSize>
      bwd_out_meta_;

  // bwd_in_meta_ used to record Grad input info for backward
  paddle::small_vector<std::vector<GradSlotMeta>, kSlotSmallVectorSize>
      bwd_in_meta_;
  // Gradient Hooks
  // Customer may register a list of hooks which will be called in order during
  // backward
  // Each entry consists of one pair of
  // <hook_id, <out_rank, std::shared_ptr<TensorHook>>>
  std::map<int64_t,
           std::tuple<
               /* slot id */ size_t,
               /* rank */ size_t,
               /* hook */ std::shared_ptr<TensorHook>>>
      gradient_hooks_;
  int64_t next_hook_id_{0};

  // We handle complex to real conversion only if any complex GradIn is involved
  bool need_complex_to_real_ = false;

  bool is_tensor_wrappers_cleared_ = false;
  // The trace of forward function
  std::string forward_trace_ = "";

  // With this flag, short-circuit the backward traversal of Tensor and
  // set the DistAttr to reduce the impact on scheduling performance
  bool is_run_auto_parallel_{false};
};

}  // namespace egr
