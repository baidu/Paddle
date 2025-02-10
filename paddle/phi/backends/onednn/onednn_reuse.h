/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

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

#include <algorithm>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "paddle/phi/backends/onednn/onednn_context.h"
#include "paddle/phi/backends/onednn/onednn_helper.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/funcs/axis_utils.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/data_layout_transform.h"
#include "paddle/phi/kernels/funcs/pooling.h"

namespace phi {
namespace funcs {

using memory = dnnl::memory;

using OneDNNMemoryFormat = dnnl::memory::format_tag;

template <typename T>
bool constexpr is_int8() {
  return std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value;
}

template <typename T>
constexpr bool is_bfloat16() {
  return std::is_same<T, dtype::bfloat16>::value;
}

// oneDNN's permute axis understand axes order in
// different way than PaddlePaddle's transpose
static std::vector<int> TransposeToPermuteAxes(const std::vector<int>& axis) {
  std::vector<int> permute_axis(axis.size());
  for (size_t i = 0; i < axis.size(); ++i) {
    permute_axis[axis[i]] = i;
  }
  return permute_axis;
}

// a trick is used here to fake transpose of out_md, so later it will be
// "untransposed", leaving output data in plain format tag
static std::vector<int64_t> FakeTransposeStrides(
    const std::vector<int64_t>& out_dims, const std::vector<int>& axis) {
  std::vector<int64_t> fake_strides(axis.size());
  int ndims = static_cast<int>(axis.size());

  int total_stride = 1;

  for (int i = ndims - 1; i >= 0; --i) {
    fake_strides[axis[i]] = total_stride;
    total_stride *= out_dims[axis[i]];
  }

  return fake_strides;
}

static std::unordered_map<std::string, dnnl::algorithm> OneDNNActivationMap() {
  return {{"abs", dnnl::algorithm::eltwise_abs},
          {"clip", dnnl::algorithm::eltwise_clip},
          {"gelu", dnnl::algorithm::eltwise_gelu_erf},
          {"gelu_erf", dnnl::algorithm::eltwise_gelu_erf},
          {"gelu_tanh", dnnl::algorithm::eltwise_gelu_tanh},
          {"hard_sigmoid", dnnl::algorithm::eltwise_hardsigmoid},
          {"hard_swish", dnnl::algorithm::eltwise_hardswish},
          {"leaky_relu", dnnl::algorithm::eltwise_relu},
          {"mish", dnnl::algorithm::eltwise_mish},
          {"relu", dnnl::algorithm::eltwise_relu},
          {"relu6", dnnl::algorithm::eltwise_clip},
          {"sigmoid", dnnl::algorithm::eltwise_logistic},
          {"sqrt", dnnl::algorithm::eltwise_sqrt},
          {"swish", dnnl::algorithm::eltwise_swish},
          {"tanh", dnnl::algorithm::eltwise_tanh}};
}

static void AppendActivation(const OneDNNContext& dev_ctx,
                             dnnl::post_ops& post_ops,  // NOLINT
                             std::string fuse_activation = "",
                             float fuse_alpha = 0.0f,
                             float fuse_beta = 0.0f) {
  if (fuse_activation == "") {
    const auto invalid_attribute =
        dev_ctx.HasDnnAttr("fuse_activation")
            ? PADDLE_GET_CONST(std::string,
                               dev_ctx.GetDnnAttr("fuse_activation"))
                  .empty()
            : true;
    if (invalid_attribute) return;

    fuse_activation =
        dev_ctx.HasDnnAttr("fuse_activation")
            ? PADDLE_GET_CONST(std::string,
                               dev_ctx.GetDnnAttr("fuse_activation"))
            : "";
    fuse_alpha = dev_ctx.HasDnnAttr("fuse_alpha")
                     ? PADDLE_GET_CONST(float, dev_ctx.GetDnnAttr("fuse_alpha"))
                     : 0.0f;
    fuse_beta = dev_ctx.HasDnnAttr("fuse_beta")
                    ? PADDLE_GET_CONST(float, dev_ctx.GetDnnAttr("fuse_beta"))
                    : 0.0f;
  }

  const auto activation_map = OneDNNActivationMap();

  const auto& activation_type = activation_map.find(fuse_activation);

  PADDLE_ENFORCE_NE(activation_type,
                    activation_map.end(),
                    errors::InvalidArgument(
                        "Activation '%s' not found in oneDNN algorithms mapper",
                        fuse_activation));

  post_ops.append_eltwise(activation_type->second, fuse_alpha, fuse_beta);
}

template <typename T,
          typename TForward,
          typename TBackward = onednn_dummy_primitive,
          typename TBackward_params = onednn_dummy_primitive>
class OneDNNHandlerT {
 public:
  OneDNNHandlerT(const OneDNNContext& dev_ctx,
                 dnnl::engine engine,
                 Place cpu_place,
                 const std::string& base_key)
      : dev_ctx_(dev_ctx),
        engine_(engine),
        place_(cpu_place),
        key_common_(base_key),
        key_(ExtendKeyWithThreadInfoIfNeeded(dev_ctx, base_key)),
        fwd_pd_(nullptr),
        bwd_pd_(nullptr) {
    OneDNNContext::tls().log_lib_version();
  }

  std::shared_ptr<TForward> AcquireForwardPrimitive() {
    const std::string key_p = key_ + "@fwd_p";
    auto forward_p =
        std::static_pointer_cast<TForward>(dev_ctx_.GetBlob(key_p));
    if (forward_p == nullptr) {
      forward_p = std::make_shared<TForward>(*fwd_pd_);
      dev_ctx_.SetBlob(key_p, forward_p);
    }
    return forward_p;
  }

  std::shared_ptr<TBackward> AcquireBackwardPrimitive() {
    const std::string key_p = key_ + "@bwd_p";
    auto backward_p =
        std::static_pointer_cast<TBackward>(dev_ctx_.GetBlob(key_p));
    if (backward_p == nullptr) {
      backward_p = std::make_shared<TBackward>(*bwd_pd_);
      dev_ctx_.SetBlob(key_p, backward_p);
    }
    return backward_p;
  }

  std::shared_ptr<TBackward_params> AcquireBackwardWeightsPrimitive() {
    const std::string key_p = key_ + "@bwd_w_p";
    auto backward_p =
        std::static_pointer_cast<TBackward_params>(dev_ctx_.GetBlob(key_p));
    if (backward_p == nullptr) {
      PADDLE_ENFORCE_NOT_NULL(
          bwd_w_pd_,
          errors::Unavailable("BWD_PD should be set when "
                              "getting BWD prim with key: %s .",
                              key_p));
      backward_p = std::make_shared<TBackward_params>(*bwd_w_pd_);
      dev_ctx_.SetBlob(key_p, backward_p);
    }
    return backward_p;
  }

  std::shared_ptr<dnnl::memory> AcquireSrcMemory(const DenseTensor* input) {
    const T* input_data = input->data<T>();
    return this->AcquireMemoryFromPrimitive(
        fwd_pd_->src_desc(), to_void_cast<T>(input_data), "@src_mem_p");
  }

  template <typename T_out = T>
  std::shared_ptr<dnnl::memory> AcquireDstMemory(DenseTensor* output) {
    T_out* ptr =
        output->mutable_data<T_out>(place_, fwd_pd_->dst_desc().get_size());
    return this->AcquireMemoryFromPrimitive(
        fwd_pd_->dst_desc(), ptr, "@dst_mem_p");
  }

  template <typename T_out = T>
  std::shared_ptr<dnnl::memory> AcquireDstMemory(void) {
    return this->AcquireMemoryFromPrimitive(fwd_pd_->dst_desc(), "@dstt_mem_p");
  }

  template <typename T_out = T>
  std::shared_ptr<dnnl::memory> AcquireDstMemory(const DenseTensor* output) {
    const T_out* output_data = output->data<T_out>();
    return this->AcquireMemoryFromPrimitive(bwd_pd_->dst_desc(),
                                            to_void_cast<T_out>(output_data),
                                            "@bwd-dst_mem_p");
  }

  std::shared_ptr<dnnl::memory> AcquireDiffDstMemory(
      const DenseTensor* diffdst) {
    const T* ptr = diffdst->data<T>();
    return this->AcquireMemoryFromPrimitive(
        bwd_pd_->diff_dst_desc(), to_void_cast<T>(ptr), "@diff_dst_mem_p");
  }

  std::shared_ptr<dnnl::memory> AcquireDiffSrcMemory(DenseTensor* diffsrc) {
    T* ptr =
        diffsrc->mutable_data<T>(place_, bwd_pd_->diff_src_desc().get_size());
    return this->AcquireMemoryFromPrimitive(
        bwd_pd_->diff_src_desc(), ptr, "@diff_src_mem_p");
  }

  // Buffer of given DenseTensor is used for oneDNN computation
  std::shared_ptr<dnnl::memory> AcquireDiffWeightsMemory(
      DenseTensor* diff_weights) {
    PADDLE_ENFORCE_NOT_NULL(
        bwd_w_pd_,
        errors::Unavailable(
            "BWD_W_PD should be set when getting BWD grad of weights."));
    T* ptr = diff_weights->mutable_data<T>(
        place_, bwd_w_pd_->diff_weights_desc().get_size());
    return this->AcquireMemoryFromPrimitive(
        bwd_w_pd_->diff_weights_desc(), ptr, "@diff_wei_mem_p");
  }

  // Buffer is allocated by oneDNN to store computation results
  std::shared_ptr<dnnl::memory> AcquireDiffWeightsMemory(void) {
    PADDLE_ENFORCE_NOT_NULL(
        bwd_w_pd_,
        errors::Unavailable(
            "BWD_W_PD should be set when getting BWD grad of weights."));
    return this->AcquireMemoryFromPrimitive(bwd_w_pd_->diff_weights_desc(),
                                            "@diff_wei_mem_p");
  }

 protected:
  bool isCached() {
    const std::string key_pd = key_ + "@fwd_pd";
    fwd_pd_ = std::static_pointer_cast<typename TForward::primitive_desc>(
        dev_ctx_.GetBlob(key_pd));

    return (fwd_pd_ != nullptr);
  }

  bool isBwdCached() {
    const std::string key_pd = key_ + "@bwd_pd";
    bwd_pd_ = std::static_pointer_cast<typename TBackward::primitive_desc>(
        dev_ctx_.GetBlob(key_pd));

    if (bwd_pd_ == nullptr) {
      return false;
    } else {
      if (std::is_same<TBackward_params, onednn_dummy_primitive>::value ==
          false) {
        const std::string key_bw_w_pd = key_ + "@bwd_w_pd";
        bwd_w_pd_ =
            std::static_pointer_cast<typename TBackward_params::primitive_desc>(
                dev_ctx_.GetBlob(key_bw_w_pd));
      }

      // When BWD is cached then still we need to Get FWD PD
      const std::string key_fpd = key_ + "@fwd_pd";
      fwd_pd_ = std::static_pointer_cast<typename TForward::primitive_desc>(
          dev_ctx_.GetBlob(key_fpd));
      PADDLE_ENFORCE_NOT_NULL(
          fwd_pd_,
          errors::Unavailable(
              "Error: FWD PD should be set when BWD PD is cached."));
      return true;
    }
  }

  // If your primitive descriptor requires attributes, pass them as a
  // first argument and parameters to descriptor constructor in the following
  // arguments. Otherwise, all arguments will be forwarded to descriptor
  // constructor, including the first one.
  template <typename Arg, typename... Args>
  void AcquireForwardPrimitiveDescriptor(Arg&& first_arg, Args&&... args) {
    // This is used when we can recreate FWD PD in BWD so
    // we do not need to pass FWD to BWD
    const std::string key_pd = key_ + "@fwd_pd";
    fwd_pd_ = std::static_pointer_cast<typename TForward::primitive_desc>(
        dev_ctx_.GetBlob(key_pd));
    if (fwd_pd_ == nullptr) {
      CreateForwardPrimitiveDescriptor(first_arg, std::forward<Args>(args)...);
      dev_ctx_.SetBlob(key_pd, fwd_pd_);
    }
  }

  // Using sfinae to specialise variadic function. Workaround for not having
  // if constexpr in C++ 11.
  template <class First, class... Args>
  typename std::enable_if<std::is_same<typename std::decay<First>::type,
                                       dnnl::primitive_attr>::value>::type
  CreateForwardPrimitiveDescriptor(First&& first, Args&&... args) {
    fwd_pd_ = std::make_shared<typename TForward::primitive_desc>(
        engine_, std::forward<Args>(args)..., first);
  }

  template <class First, class... Args>
  typename std::enable_if<!std::is_same<typename std::decay<First>::type,
                                        dnnl::primitive_attr>::value>::type
  CreateForwardPrimitiveDescriptor(First&& first, Args&&... args) {
    fwd_pd_ = std::make_shared<typename TForward::primitive_desc>(
        engine_, std::forward<First>(first), std::forward<Args>(args)...);
  }

  template <typename... Args>
  void AcquireBackwardPrimitiveDescriptor(Args&&... args) {
    // fwd_pd_ is set during grad by calling
    // AcquireForwardPrimitiveDescriptor
    PADDLE_ENFORCE_NOT_NULL(
        fwd_pd_,
        errors::Unavailable("Get OneDNN Forward primitive %s failed.",
                            key_ + "@fwd_pd"));
    const std::string key_pd = key_ + "@bwd_pd";
    bwd_pd_ = std::static_pointer_cast<typename TBackward::primitive_desc>(
        dev_ctx_.GetBlob(key_pd));
    if (bwd_pd_ == nullptr) {
      bwd_pd_ = std::make_shared<typename TBackward::primitive_desc>(
          engine_, std::forward<Args>(args)..., *fwd_pd_);
      dev_ctx_.SetBlob(key_pd, bwd_pd_);
    }
  }

  template <typename... Args>
  void AcquireBackwardWeightsPrimitiveDescriptor(Args&&... args) {
    // fwd_pd_ is set during grad by calling
    // AcquireForwardPrimitiveDescriptor
    PADDLE_ENFORCE_NOT_NULL(
        fwd_pd_,
        errors::Unavailable("Get OneDNN Forward primitive %s failed.",
                            key_ + "@fwd_pd"));
    const std::string key_pd = key_ + "@bwd_w_pd";
    bwd_w_pd_ =
        std::static_pointer_cast<typename TBackward_params::primitive_desc>(
            dev_ctx_.GetBlob(key_pd));
    if (bwd_w_pd_ == nullptr) {
      bwd_w_pd_ = std::make_shared<typename TBackward_params::primitive_desc>(
          engine_, std::forward<Args>(args)..., *fwd_pd_);
      dev_ctx_.SetBlob(key_pd, bwd_w_pd_);
    }
  }

  std::shared_ptr<dnnl::memory> AcquireMemoryFromPrimitive(
      const std::string& suffix) {
    return std::static_pointer_cast<dnnl::memory>(
        dev_ctx_.GetBlob(key_ + suffix));
  }

  std::shared_ptr<dnnl::memory> AcquireMemoryFromPrimitive(
      dnnl::memory::desc md, void* ptr, const std::string& suffix) {
    const auto local_key = key_ + suffix;
    auto mem_p =
        std::static_pointer_cast<dnnl::memory>(dev_ctx_.GetBlob(local_key));
    if (mem_p == nullptr) {
      mem_p = std::make_shared<dnnl::memory>(md, engine_, ptr);
      dev_ctx_.SetBlob(local_key, mem_p);
    } else {
      mem_p->set_data_handle(ptr);
    }
    return mem_p;
  }

  std::shared_ptr<dnnl::memory> AcquireMemoryFromPrimitive(
      dnnl::memory::desc md, const std::string& suffix) {
    const auto local_key = key_ + suffix;
    auto mem_p =
        std::static_pointer_cast<dnnl::memory>(dev_ctx_.GetBlob(local_key));
    if (mem_p == nullptr) {
      mem_p = std::make_shared<dnnl::memory>(md, engine_);
      dev_ctx_.SetBlob(local_key, mem_p);
    }
    return mem_p;
  }

  void AcquireReorder(const std::shared_ptr<dnnl::memory>& user_memory_p,
                      const std::shared_ptr<dnnl::memory>& target_memory_p) {
    auto reorder_p =
        std::make_shared<dnnl::reorder>(*user_memory_p, *target_memory_p);

    auto& astream = OneDNNContext::tls().get_stream();

    reorder_p->execute(
        astream,
        {{DNNL_ARG_FROM, *user_memory_p}, {DNNL_ARG_TO, *target_memory_p}});
    astream.wait();
  }

  template <typename F = T>
  std::shared_ptr<dnnl::memory> AcquireMemoryWithReorder(
      const dnnl::memory::desc& user_md,
      const dnnl::memory::desc& target_md,
      void* ptr,
      const std::string& suffix,
      bool is_persistent = false,
      std::function<std::shared_ptr<F>(const F*)> custom_reorder_func = {},
      const std::vector<float>& scale_data = {1.0f},
      int mask = 0) {
    const auto target_key = key_ + suffix + "_target";
    const auto key_reorder_p = key_ + suffix + "reorder_p";
    const auto user_key = key_ + suffix + "_user";

    auto target_memory_p =
        std::static_pointer_cast<dnnl::memory>(dev_ctx_.GetBlob(target_key));

    if (target_memory_p == nullptr) {
      if (custom_reorder_func) {
        auto reordered_data =
            custom_reorder_func(reinterpret_cast<const F*>(ptr));
        dev_ctx_.SetBlob(key_reorder_p + "-custom_reorder", reordered_data);
        ptr = reinterpret_cast<void*>(reordered_data.get());
      }
      auto user_memory_p =
          std::make_shared<dnnl::memory>(user_md, engine_, ptr);
      if (user_md != target_md) {
        target_memory_p = std::make_shared<dnnl::memory>(target_md, engine_);
        dnnl::reorder::primitive_desc reorder_pdesc;
        if (is_int8<T>()) {
          dnnl::primitive_attr attr;
          attr.set_scales_mask(DNNL_ARG_SRC, mask);
          reorder_pdesc = dnnl::reorder::primitive_desc(
              *user_memory_p, *target_memory_p, attr);
        } else {
          reorder_pdesc =
              dnnl::reorder::primitive_desc(*user_memory_p, *target_memory_p);
        }
        auto reorder_p = std::make_shared<dnnl::reorder>(reorder_pdesc);
        dev_ctx_.SetBlob(key_reorder_p, reorder_p);

        auto& astream = OneDNNContext::tls().get_stream();
        std::unordered_map<int, dnnl::memory> reorder_args;
        reorder_args.insert({DNNL_ARG_SRC, *user_memory_p});
        reorder_args.insert({DNNL_ARG_DST, *target_memory_p});
        if (is_int8<T>()) {
          auto scale_md =
              dnnl::memory::desc({static_cast<int64_t>(scale_data.size())},
                                 dnnl::memory::data_type::f32,
                                 dnnl::memory::format_tag::x);
          auto scale_data_mem = dnnl::memory(scale_md, engine_);
          scale_data_mem.set_data_handle(
              phi::funcs::to_void_cast(scale_data.data()));
          reorder_args.insert(
              {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, scale_data_mem});
        }
        reorder_p->execute(astream, reorder_args);

        astream.wait();
      } else {
        target_memory_p = user_memory_p;
      }
      dev_ctx_.SetBlob(user_key, user_memory_p);
      dev_ctx_.SetBlob(target_key, target_memory_p);
    } else if (!is_persistent) {
      auto& astream = OneDNNContext::tls().get_stream();

      auto user_memory_p =
          std::static_pointer_cast<dnnl::memory>(dev_ctx_.GetBlob(user_key));
      user_memory_p->set_data_handle(ptr);

      // TODO(jczaja): Here we detect if reorder is cached it means it is needed
      // need to change this to get rid of keys
      auto reorder_p = std::static_pointer_cast<dnnl::reorder>(
          dev_ctx_.GetBlob(key_reorder_p));
      if (reorder_p != nullptr) {
        reorder_p->execute(
            astream,
            {{DNNL_ARG_FROM, *user_memory_p}, {DNNL_ARG_TO, *target_memory_p}});
        astream.wait();
      }
    }
    return target_memory_p;
  }

  std::shared_ptr<dnnl::memory> AcquireMemory(const std::string& suffix) {
    const auto local_key = key_ + suffix;
    return std::static_pointer_cast<dnnl::memory>(dev_ctx_.GetBlob(local_key));
  }

  void CacheMemory(const std::string& suffix,
                   const std::shared_ptr<dnnl::memory>& mem_p) {
    const auto local_key = key_ + suffix;
    dev_ctx_.SetBlob(local_key, mem_p);
    return;
  }

  const OneDNNContext& dev_ctx_;
  dnnl::engine engine_;
  Place place_;
  std::string key_common_;
  std::string key_;
  std::shared_ptr<typename TForward::primitive_desc> fwd_pd_;
  std::shared_ptr<typename TBackward::primitive_desc> bwd_pd_;
  std::shared_ptr<typename TBackward_params::primitive_desc> bwd_w_pd_;
};

template <typename T,
          typename TForward,
          typename TBackward = onednn_dummy_primitive,
          typename TBackward_params = onednn_dummy_primitive>
class OneDNNHandlerNoCachingT {
 public:
  OneDNNHandlerNoCachingT(dnnl::engine engine, Place cpu_place)
      : engine_(engine), place_(cpu_place), fwd_pd_(nullptr), bwd_pd_(nullptr) {
    OneDNNContext::tls().log_lib_version();
  }

  std::shared_ptr<TForward> AcquireForwardPrimitive() {
    return std::make_shared<TForward>(*fwd_pd_);
  }

  std::shared_ptr<TBackward> AcquireBackwardPrimitive() {
    return std::make_shared<TBackward>(*bwd_pd_);
  }

  std::shared_ptr<TBackward_params> AcquireBackwardWeightsPrimitive() {
    PADDLE_ENFORCE_NOT_NULL(bwd_w_pd_,
                            errors::Unavailable("BWD_PD should be set when "
                                                "getting BWD prim ."));
    return std::make_shared<TBackward_params>(*bwd_w_pd_);
  }

  std::shared_ptr<dnnl::memory> AcquireSrcMemory(const DenseTensor* input) {
    const T* input_data = input->data<T>();
    return this->AcquireMemoryFromPrimitive(fwd_pd_->src_desc(),
                                            to_void_cast<T>(input_data));
  }

  template <typename T_out = T>
  std::shared_ptr<dnnl::memory> AcquireDstMemory(DenseTensor* output) {
    T_out* ptr =
        output->mutable_data<T_out>(place_, fwd_pd_->dst_desc().get_size());
    return this->AcquireMemoryFromPrimitive(fwd_pd_->dst_desc(), ptr);
  }

  template <typename T_out = T>
  std::shared_ptr<dnnl::memory> AcquireDstMemory(void) {
    return this->AcquireMemoryFromPrimitive(fwd_pd_->dst_desc());
  }

  template <typename T_out = T>
  std::shared_ptr<dnnl::memory> AcquireDstMemory(const DenseTensor* output) {
    const T_out* output_data = output->data<T_out>();
    return this->AcquireMemoryFromPrimitive(bwd_pd_->dst_desc(),
                                            to_void_cast<T_out>(output_data));
  }

  std::shared_ptr<dnnl::memory> AcquireDiffDstMemory(
      const DenseTensor* diffdst) {
    const T* ptr = diffdst->data<T>();
    return this->AcquireMemoryFromPrimitive(bwd_pd_->diff_dst_desc(),
                                            to_void_cast<T>(ptr));
  }

  std::shared_ptr<dnnl::memory> AcquireDiffSrcMemory(DenseTensor* diffsrc) {
    T* ptr =
        diffsrc->mutable_data<T>(place_, bwd_pd_->diff_src_desc().get_size());
    return this->AcquireMemoryFromPrimitive(bwd_pd_->diff_src_desc(), ptr);
  }

  // Buffer of given DenseTensor is used for oneDNN computation
  std::shared_ptr<dnnl::memory> AcquireDiffWeightsMemory(
      DenseTensor* diff_weights) {
    PADDLE_ENFORCE_NOT_NULL(
        bwd_w_pd_,
        errors::Unavailable(
            "BWD_W_PD should be set when getting BWD grad of weights."));
    T* ptr = diff_weights->mutable_data<T>(
        place_, bwd_w_pd_->diff_weights_desc().get_size());
    return this->AcquireMemoryFromPrimitive(bwd_w_pd_->diff_weights_desc(),
                                            ptr);
  }

  // Buffer is allocated by oneDNN to store computation results
  std::shared_ptr<dnnl::memory> AcquireDiffWeightsMemory(void) {
    PADDLE_ENFORCE_NOT_NULL(
        bwd_w_pd_,
        errors::Unavailable(
            "BWD_W_PD should be set when getting BWD grad of weights."));
    return this->AcquireMemoryFromPrimitive(bwd_w_pd_->diff_weights_desc());
  }

 protected:
  // If your primitive descriptor requires attributes, pass them as a
  // first argument and parameters to descriptor constructor in the following
  // arguments. Otherwise, all arguments will be forwarded to descriptor
  // constructor, including the first one.
  template <typename Arg, typename... Args>
  void AcquireForwardPrimitiveDescriptor(Arg&& first_arg, Args&&... args) {
    CreateForwardPrimitiveDescriptor(first_arg, std::forward<Args>(args)...);
  }

  // Using sfinae to specialise variadic function. Workaround for not having
  // if constexpr in C++ 11.
  template <class First, class... Args>
  typename std::enable_if<std::is_same<typename std::decay<First>::type,
                                       dnnl::primitive_attr>::value>::type
  CreateForwardPrimitiveDescriptor(First&& first, Args&&... args) {
    fwd_pd_ = std::make_shared<typename TForward::primitive_desc>(
        engine_, std::forward<Args>(args)..., first);
  }

  template <class First, class... Args>
  typename std::enable_if<!std::is_same<typename std::decay<First>::type,
                                        dnnl::primitive_attr>::value>::type
  CreateForwardPrimitiveDescriptor(First&& first, Args&&... args) {
    fwd_pd_ = std::make_shared<typename TForward::primitive_desc>(
        engine_, std::forward<First>(first), std::forward<Args>(args)...);
  }

  template <typename... Args>
  void AcquireBackwardPrimitiveDescriptor(Args&&... args) {
    // fwd_pd_ is set during grad by calling
    // AcquireForwardPrimitiveDescriptor
    PADDLE_ENFORCE_NOT_NULL(
        fwd_pd_,
        errors::Unavailable("Get oneDNN Forward primitive %s failed."));
    bwd_pd_ = std::make_shared<typename TBackward::primitive_desc>(
        engine_, std::forward<Args>(args)..., *fwd_pd_);
  }

  template <typename... Args>
  void AcquireBackwardWeightsPrimitiveDescriptor(Args&&... args) {
    // fwd_pd_ is set during grad by calling
    // AcquireForwardPrimitiveDescriptor
    PADDLE_ENFORCE_NOT_NULL(
        fwd_pd_,
        errors::Unavailable("Get oneDNN Forward primitive %s failed."));
    auto bwd_desc =
        typename TBackward_params::desc(std::forward<Args>(args)...);
    bwd_w_pd_ = std::make_shared<typename TBackward_params::primitive_desc>(
        bwd_desc, engine_, *fwd_pd_);
  }

  std::shared_ptr<dnnl::memory> AcquireMemoryFromPrimitive(
      dnnl::memory::desc md, void* ptr) {
    return std::make_shared<dnnl::memory>(md, engine_, ptr);
  }

  std::shared_ptr<dnnl::memory> AcquireMemoryFromPrimitive(
      dnnl::memory::desc md) {
    return std::make_shared<dnnl::memory>(md, engine_);
  }

  void AcquireReorder(const std::shared_ptr<dnnl::memory>& user_memory_p,
                      const std::shared_ptr<dnnl::memory>& target_memory_p) {
    auto reorder_p =
        std::make_shared<dnnl::reorder>(*user_memory_p, *target_memory_p);

    auto& astream = OneDNNContext::tls().get_stream();

    reorder_p->execute(
        astream,
        {{DNNL_ARG_FROM, *user_memory_p}, {DNNL_ARG_TO, *target_memory_p}});
    astream.wait();
  }

  template <typename F = T>
  std::shared_ptr<dnnl::memory> AcquireMemoryWithReorder(
      const dnnl::memory::desc& user_md,
      const dnnl::memory::desc& target_md,
      void* ptr,
      bool is_persistent UNUSED = false,
      std::function<std::shared_ptr<F>(const F*)> custom_reorder_func = {}) {
    std::shared_ptr<dnnl::memory> target_memory_p;
    if (custom_reorder_func) {
      auto reordered_data =
          custom_reorder_func(reinterpret_cast<const F*>(ptr));
      ptr = reinterpret_cast<void*>(reordered_data.get());
    }
    auto user_memory_p = std::make_shared<dnnl::memory>(user_md, engine_, ptr);
    if (user_md != target_md) {
      target_memory_p = std::make_shared<dnnl::memory>(target_md, engine_);
      auto reorder_p =
          std::make_shared<dnnl::reorder>(*user_memory_p, *target_memory_p);

      auto& astream = OneDNNContext::tls().get_stream();
      reorder_p->execute(
          astream,
          {{DNNL_ARG_FROM, *user_memory_p}, {DNNL_ARG_TO, *target_memory_p}});
      astream.wait();
    } else {
      target_memory_p = user_memory_p;
    }
    return target_memory_p;
  }

  dnnl::engine engine_;
  Place place_;
  std::shared_ptr<typename TForward::primitive_desc> fwd_pd_;
  std::shared_ptr<typename TBackward::primitive_desc> bwd_pd_;
  std::shared_ptr<typename TBackward_params::primitive_desc> bwd_w_pd_;
};

template <typename T>
class ActivationOneDNNHandler
    : public OneDNNHandlerNoCachingT<T,
                                     dnnl::eltwise_forward,
                                     dnnl::eltwise_backward> {
 public:
  ActivationOneDNNHandler(dnnl::algorithm algorithm,
                          float alpha,
                          float beta,
                          const dnnl::engine engine,
                          Place cpu_place,
                          const DenseTensor* x)
      : OneDNNHandlerNoCachingT<T,
                                dnnl::eltwise_forward,
                                dnnl::eltwise_backward>(engine, cpu_place) {
    this->AcquireForwardPrimitiveDescriptor(dnnl::prop_kind::forward_training,
                                            algorithm,
                                            x->mem_desc(),
                                            x->mem_desc(),
                                            alpha,
                                            beta);
  }

  ActivationOneDNNHandler(dnnl::algorithm algorithm,
                          float alpha,
                          float beta,
                          const dnnl::engine engine,
                          Place cpu_place,
                          const DenseTensor* x,
                          const DenseTensor* dout)
      : OneDNNHandlerNoCachingT<T,
                                dnnl::eltwise_forward,
                                dnnl::eltwise_backward>(engine, cpu_place) {
    this->AcquireForwardPrimitiveDescriptor(dnnl::prop_kind::forward_training,
                                            algorithm,
                                            x->mem_desc(),
                                            x->mem_desc(),
                                            alpha,
                                            beta);
    this->AcquireBackwardPrimitiveDescriptor(algorithm,
                                             dout->mem_desc(),
                                             dout->mem_desc(),
                                             x->mem_desc(),
                                             alpha,
                                             beta);
  }

  std::shared_ptr<dnnl::memory> AcquireBackwardSrcMemory(
      const DenseTensor* input) {
    const T* input_data = input->data<T>();
    return this->AcquireMemoryFromPrimitive(this->bwd_pd_->src_desc(),
                                            to_void_cast<T>(input_data));
  }
};

template <typename T>
class SoftmaxOneDNNHandler
    : public OneDNNHandlerNoCachingT<T,
                                     dnnl::softmax_forward,
                                     dnnl::softmax_backward> {
 public:
  SoftmaxOneDNNHandler(const dnnl::engine onednn_engine,
                       Place cpu_place,
                       int axis,
                       const DenseTensor* x,
                       DenseTensor* out)
      : OneDNNHandlerNoCachingT<T,
                                dnnl::softmax_forward,
                                dnnl::softmax_backward>(onednn_engine,
                                                        cpu_place) {
    PADDLE_ENFORCE_EQ(
        x->dims(),
        out->dims(),
        errors::InvalidArgument(
            "The shape of input and output tensor must be identical."));

    int rank = x->dims().size() != 0 ? x->dims().size() : 1;
    const int canonical_axis = funcs::CanonicalAxis(axis, rank);
    this->AcquireForwardPrimitiveDescriptor(dnnl::prop_kind::forward_inference,
                                            dnnl::algorithm::softmax_accurate,
                                            x->mem_desc(),
                                            x->mem_desc(),
                                            canonical_axis);
  }

  SoftmaxOneDNNHandler(const dnnl::engine onednn_engine,
                       Place cpu_place,
                       int axis,
                       const DenseTensor* out,
                       const DenseTensor* out_grad)
      : OneDNNHandlerNoCachingT<T,
                                dnnl::softmax_forward,
                                dnnl::softmax_backward>(onednn_engine,
                                                        cpu_place) {
    int rank = out_grad->dims().size() != 0 ? out_grad->dims().size() : 1;
    const int canonical_axis = funcs::CanonicalAxis(axis, rank);
    this->AcquireForwardPrimitiveDescriptor(dnnl::prop_kind::forward_inference,
                                            dnnl::algorithm::softmax_accurate,
                                            out->mem_desc(),
                                            out->mem_desc(),
                                            canonical_axis);
    this->AcquireBackwardPrimitiveDescriptor(dnnl::algorithm::softmax_accurate,
                                             out_grad->mem_desc(),
                                             out_grad->mem_desc(),
                                             out->mem_desc(),
                                             canonical_axis);
  }
};

class ReorderOneDNNHandler {
 public:
  ReorderOneDNNHandler(std::vector<int64_t>& dims,  // NOLINT
                       DataType ptype,
                       dnnl::memory::data_type dtype,
                       dnnl::engine engine)
      : dims_(dims),
        ptype_(ptype),
        ptype_dst_(ptype),
        dtype_(dtype),
        dtype_dst_(dtype),
        engine_(engine) {}

  ReorderOneDNNHandler(std::vector<int64_t>& dims,  // NOLINT
                       DataType ptype,
                       dnnl::memory::data_type dtype,
                       DataType ptype_dst,
                       dnnl::memory::data_type dtype_dst,
                       dnnl::engine engine)
      : dims_(dims),
        ptype_(ptype),
        ptype_dst_(ptype_dst),
        dtype_(dtype),
        dtype_dst_(dtype_dst),
        engine_(engine) {}

  std::shared_ptr<dnnl::memory> AcquireSrcMemory(const dnnl::memory::desc& md,
                                                 void* ptr) {
    return std::make_shared<dnnl::memory>(md, engine_, ptr);
  }

  std::shared_ptr<dnnl::memory> AcquireSrcMemory(const OneDNNMemoryFormat& fmt,
                                                 void* ptr) {
    auto md = dnnl::memory::desc(dims_, dtype_, fmt);
    return std::make_shared<dnnl::memory>(md, engine_, ptr);
  }

  std::shared_ptr<dnnl::memory> AcquireSubmemory(
      const std::vector<int64_t>& dims,
      const std::vector<int64_t>& offset,
      const std::shared_ptr<dnnl::memory>& mem_p) {
    auto sub_md = mem_p->get_desc().submemory_desc(dims, {offset});
    auto sub_mem_p = std::make_shared<dnnl::memory>(
        sub_md, engine_, mem_p->get_data_handle());
    return sub_mem_p;
  }

  std::shared_ptr<dnnl::memory> AcquireDstMemory(DenseTensor* output,
                                                 const OneDNNMemoryFormat& fmt,
                                                 Place place) {
    auto dst_md = OneDNNMemDesc(dims_, dtype_dst_, fmt);
    auto dst_data = output->mutable_data(place, ptype_dst_, dst_md.get_size());
    return std::make_shared<dnnl::memory>(dst_md, engine_, dst_data);
  }

  std::shared_ptr<dnnl::memory> AcquireDstMemory(
      DenseTensor* output, const dnnl::memory::desc& src_md, Place place) {
    if (ptype_dst_ == ptype_) {
      auto dst_data =
          output->mutable_data(place, ptype_dst_, src_md.get_size());
      return std::make_shared<dnnl::memory>(src_md, engine_, dst_data);
    } else {
      auto dst_md = dnnl::memory::desc(
          src_md.get_dims(), dtype_dst_, src_md.get_strides());
      auto dst_data =
          output->mutable_data(place, ptype_dst_, dst_md.get_size());
      return std::make_shared<dnnl::memory>(dst_md, engine_, dst_data);
    }
  }

  std::shared_ptr<dnnl::memory> AcquireDstMemory(
      DenseTensor* output,
      const std::vector<int64_t>& dims,
      const std::vector<int64_t>& strides,
      Place place) {
    auto dst_md = dnnl::memory::desc(dims, dtype_dst_, strides);
    auto dst_data = output->mutable_data(place, ptype_dst_, dst_md.get_size());
    return std::make_shared<dnnl::memory>(dst_md, engine_, dst_data);
  }

  std::shared_ptr<dnnl::memory> AcquireDstMemory(
      DenseTensor* output,
      const std::vector<int64_t>& dims,
      const OneDNNMemoryFormat& fmt,
      Place place) {
    auto dst_md = OneDNNMemDesc(dims, dtype_dst_, fmt);
    auto dst_data = output->mutable_data(place, ptype_dst_, dst_md.get_size());
    return std::make_shared<dnnl::memory>(dst_md, engine_, dst_data);
  }

  std::shared_ptr<dnnl::reorder> AcquireReorder(
      std::shared_ptr<dnnl::memory> dst_memory_p,
      std::shared_ptr<dnnl::memory> src_memory_p) {
    return std::make_shared<dnnl::reorder>(*(src_memory_p), *(dst_memory_p));
  }

  std::shared_ptr<dnnl::reorder> AcquireReorder(
      std::shared_ptr<dnnl::memory> dst_memory_p,
      std::shared_ptr<dnnl::memory> src_memory_p,
      const dnnl::primitive_attr& attrs) {
    return std::make_shared<dnnl::reorder>(
        *(src_memory_p), *(dst_memory_p), attrs);
  }

 private:
  std::vector<int64_t> dims_;
  DataType ptype_, ptype_dst_;
  dnnl::memory::data_type dtype_, dtype_dst_;
  dnnl::engine engine_;
};

template <typename T>
class BinaryOneDNNHandler : public OneDNNHandlerNoCachingT<T, dnnl::binary> {
 public:
  bool use_broadcasting_hack;
  bool swin_case;
  BinaryOneDNNHandler(const dnnl::algorithm algo,
                      const int axis,
                      const dnnl::engine engine,
                      Place cpu_place,
                      const DenseTensor* x,
                      const DenseTensor* y,
                      DenseTensor* out,
                      float scale_x,
                      float scale_y,
                      float scale_out,
                      bool allow_hack,
                      const dnnl::post_ops& post_ops = dnnl::post_ops{})
      : OneDNNHandlerNoCachingT<T, dnnl::binary>(engine, cpu_place) {
    use_broadcasting_hack = false;
    swin_case = false;
    const auto src_x_tz = common::vectorize(x->dims());
    const auto src_y_tz = common::vectorize(y->dims());
    // if output tensor(z) is nullptr then we are computing into oneDNN
    // managed buffer
    auto rankdiff = x->dims().size() - y->dims().size();
    auto dst_tz =
        (out == nullptr)
            ? (rankdiff > 0 ? src_x_tz
                            : (y->dims().size() == 0 ? std::vector<int64_t>{1}
                                                     : src_x_tz))
            : (out->dims().size() == 0 ? std::vector<int64_t>{1}
                                       : common::vectorize(out->dims()));

    auto src0_md = x->mem_desc();
    auto src1_md = y->mem_desc();
    if (rankdiff > 0) {  // Second input is of smaller rank than first
      std::vector<int64_t> dims1_ex(rankdiff, 1);
      dims1_ex.insert(next(dims1_ex.begin(), (axis == -1 ? rankdiff : axis)),
                      src_y_tz.begin(),
                      src_y_tz.end());
      // For broadcasting for NHWC we need rotate extended shape
      if (OneDNNContext::tls().get_cur_paddle_data_layout() ==
          DataLayout::kNHWC) {
        std::rotate(dims1_ex.begin() + 1, dims1_ex.end() - 1, dims1_ex.end());
      }
      src1_md = src1_md.reshape(dims1_ex);
    } else if (rankdiff < 0) {  // First input is of smaller than second
      std::vector<int64_t> dims0_ex(-rankdiff, 1);
      dims0_ex.insert(next(dims0_ex.begin(), (axis == -1 ? -rankdiff : axis)),
                      src_x_tz.begin(),
                      src_x_tz.end());
      // For broadcasting for NHWC we need rotate extended shape
      if (OneDNNContext::tls().get_cur_paddle_data_layout() ==
          DataLayout::kNHWC) {
        std::rotate(dims0_ex.begin() + 1, dims0_ex.end() - 1, dims0_ex.end());
      }
      src0_md = src0_md.reshape(dims0_ex);
    }
    dnnl::primitive_attr attributes;
    std::tie(attributes, scale_0_, scale_1_) =
        CreateAttributes(algo, scale_x, scale_y, scale_out, post_ops);

    // Workaround for U2++ model which deletes first tensor dimensions to enable
    // optimized oneDNNs broadcasting. Output tensor is reshaped back afterwards
    // at the end of the kernel, after the computation
    auto src0_dims = src0_md.get_dims();
    auto src1_dims = src1_md.get_dims();
    if (allow_hack && dst_tz.size() == 4 && src0_dims[2] != src1_dims[2]) {
      auto are_strides_plain = [](std::vector<int64_t>& strides, int ndims) {
        for (int i = 0; i < ndims - 1; ++i) {
          if (strides[i] < strides[i + 1]) {
            return false;
          }
        }
        return true;
      };

      auto src0_strides = src0_md.get_strides();
      auto src1_strides = src1_md.get_strides();
      auto src0_dims = src0_md.get_dims();
      auto src1_dims = src1_md.get_dims();

      bool can_squeeze = src0_dims[0] == src1_dims[0] &&
                         src0_dims[1] == src1_dims[1] &&
                         src0_dims[3] == src1_dims[3];

      if (can_squeeze && are_strides_plain(src0_strides, 4) &&
          are_strides_plain(src1_strides, 4)) {
        src0_dims[1] *= dst_tz[0];
        src1_dims[1] *= dst_tz[0];
        dst_tz[1] *= dst_tz[0];
        dst_tz.erase(dst_tz.begin());
        src0_md = src0_md.reshape({src0_dims.begin() + 1, src0_dims.end()});
        src1_md = src1_md.reshape({src1_dims.begin() + 1, src1_dims.end()});
        use_broadcasting_hack = true;
      }
    }

    // Special case for swin transformer, which includes elewise_add/sub with
    // rare shapes. By extending the shape to utilize oneDNN optimized binary
    // optimization
    if (allow_hack && (dst_tz.size() == 3 || dst_tz.size() == 5)) {
      if (dst_tz.size() == 3 && algo == dnnl::algorithm::binary_sub) {
        if ((x->numel() == y->numel()) && (src0_dims[1] != src1_dims[1]) &&
            (src0_dims[2] != src1_dims[2]) && (src0_dims[1] == 1) &&
            (src1_dims[2] == 1)) {
          if (x->numel() < out->numel()) {
            src0_md = dnnl::memory::desc(
                dst_tz, OneDNNGetDataType<T>(), OneDNNMemoryFormat::abc);
            swin_case = true;
          }
        }
      } else if (dst_tz.size() == 5 && algo == dnnl::algorithm::binary_add) {
        if (src0_dims[0] == 1 && src0_dims[0] == src1_dims[0] &&
            src0_dims[1] == src1_dims[1] && src0_dims[3] == src1_dims[3] &&
            src0_dims[4] == src1_dims[4]) {
          if (src0_dims[1] != 1 && src0_dims[2] != src1_dims[2] &&
              src1_dims[2] == 1) {
            src1_md = dnnl::memory::desc(
                dst_tz, OneDNNGetDataType<T>(), OneDNNMemoryFormat::any);
            swin_case = true;
          }
        }
      }
    }

    auto dst_md =
        memory::desc(dst_tz, OneDNNGetDataType<T>(), OneDNNMemoryFormat::any);

    if (x->numel() < y->numel()) {
      if (algo == dnnl::algorithm::binary_sub) {
        std::tie(attributes, scale_0_, scale_1_) = CreateAttributes(
            algo, -1.0 * scale_x, -1.0 * scale_y, scale_out, post_ops);
      }
      this->AcquireForwardPrimitiveDescriptor(
          attributes, algo, src1_md, src0_md, dst_md);
    } else {
      this->AcquireForwardPrimitiveDescriptor(
          attributes, algo, src0_md, src1_md, dst_md);
    }
  }

  std::shared_ptr<dnnl::memory> AcquireExtendMemoryFromPrimitive(
      dnnl::memory::desc md, void* ptr, int src_idx) {
    auto mem_p = std::make_shared<dnnl::memory>(md, this->engine_);
    size_t size = md.get_size() / sizeof(T);
    T* dst = static_cast<T*>(mem_p->get_data_handle());
    T* new_ptr = static_cast<T*>(ptr);
    if (src_idx == 0) {
      auto C = md.get_dims()[1];
      auto H = md.get_dims()[2];
#if defined(_OPENMP)
#pragma omp parallel for
#endif
      for (size_t i = 0; i < size; ++i) {
        auto n = i / (C * H);
        auto c = (i - n * (C * H)) / H;
        auto h = i - n * (C * H) - c * H;
        auto mod_i = n * H + h;
        dst[i] = new_ptr[mod_i];
      }
    } else if (src_idx == 1) {
      auto C = md.get_dims()[2];
      auto H = md.get_dims()[3];
      auto W = md.get_dims()[4];
#if defined(_OPENMP)
#pragma omp parallel for
#endif
      for (size_t i = 0; i < size; ++i) {
        auto n = i / (C * H * W);
        auto c = (i - n * (C * H * W)) / (H * W);
        auto h = (i - n * (C * H * W) - c * (H * W)) / W;
        auto w = i - n * (C * H * W) - c * (H * W) - h * W;
        auto mod_i = n * H * W + h * W + w;
        dst[i] = new_ptr[mod_i];
      }
    }
    return mem_p;
  }

  // Special case for swin_transformer
  std::shared_ptr<dnnl::memory> AcquireExtendSrcMemory(const DenseTensor* input,
                                                       int src_idx) {
    const T* input_data = input->data<T>();
    return src_idx == 0 ? this->AcquireExtendMemoryFromPrimitive(
                              this->fwd_pd_->src0_desc(),
                              to_void_cast<T>(input_data),
                              src_idx)
                        : this->AcquireExtendMemoryFromPrimitive(
                              this->fwd_pd_->src1_desc(),
                              to_void_cast<T>(input_data),
                              src_idx);
  }

  std::shared_ptr<dnnl::memory> AcquireSecondSrcMemory(
      const DenseTensor* input) {
    const T* input_data = input->data<T>();
    return this->AcquireMemoryFromPrimitive(this->fwd_pd_->src1_desc(),
                                            to_void_cast<T>(input_data));
  }

  dnnl::memory Get_SRC_0_Scale_Memory() {
    return Get_SRC_Scale_Memory(scale_0_);
  }

  dnnl::memory Get_SRC_1_Scale_Memory() {
    return Get_SRC_Scale_Memory(scale_1_);
  }

  bool Has_SRC_0_Scale() { return std::fabs(scale_0_ - 1.0f) > 1e-6f; }

  bool Has_SRC_1_Scale() { return std::fabs(scale_1_ - 1.0f) > 1e-6f; }

 private:
  dnnl::memory Get_SRC_Scale_Memory(float scale) {
    auto scale_md = dnnl::memory::desc(
        {1}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::x);
    auto scale_memory = dnnl::memory(scale_md, this->engine_);
    auto scale_memory_buf = static_cast<float*>(scale_memory.get_data_handle());
    *scale_memory_buf = scale;
    return scale_memory;
  }

  static inline std::tuple<dnnl::primitive_attr, float, float> CreateAttributes(
      dnnl::algorithm op,
      float scale_x,
      float scale_y,
      float scale_out,
      dnnl::post_ops post_ops = dnnl::post_ops{}) {
    // Scales set in attributes for inputs contribute to the output equation
    // in the following way (assuming no broadcasting takes place):
    // output_i = scale_0 * x_i <+ or *> scale_1 * y_i;
    // Hence we have to create scales that will:
    // 1. Dequantize both values, by multiplying with (1.0 / scale_x_or_y)
    // 2. Quantize their result to output scale range, by multiplying with
    // (scale_z)
    // If we combine these two, we end up with following equation
    // output = scale_out * (1/scale_x * x <* or +> 1/scale_y * y)
    // Hence, to mimic such behaviour using provided interface,
    // For add operation the equation is equal to:
    // output = (scale_out / scale_x) * x + (scale_out / scale_y) * y
    //                <scale_0>                  <scale_1>
    // For mul operation on the other hand
    // output = (scale_out / scale_x) * x * (1.0 / scale_y) * y
    //                <scale_0>                 <scale_1>
    float scale_0 = scale_out / scale_x;
    float scale_1 =
        op == dnnl::algorithm::binary_add ? scale_out / scale_y : 1.0 / scale_y;
    dnnl::primitive_attr attributes;
    if (std::fabs(scale_0 - 1.0f) > 1e-6f) {
      attributes.set_scales_mask(/* input_x_id = */ DNNL_ARG_SRC_0,
                                 /* mask = */ 0);
    }
    if (std::fabs(scale_1 - 1.0f) > 1e-6f) {
      attributes.set_scales_mask(/* input_y_id = */ DNNL_ARG_SRC_1,
                                 /* mask = */ 0);
    }
    if (post_ops.len() > 0) attributes.set_post_ops(post_ops);
    return std::make_tuple(attributes, scale_0, scale_1);
  }

 private:
  float scale_0_ = 1.0f;
  float scale_1_ = 1.0f;
};

template <typename T>
class BroadcastDataOneDNNHandler
    : public OneDNNHandlerNoCachingT<T, dnnl::binary> {
 public:
  BroadcastDataOneDNNHandler(const dnnl::algorithm algo,
                             const dnnl::engine engine,
                             Place cpu_place,
                             const DenseTensor* x,
                             DenseTensor* out,
                             float scale_x,
                             float scale_y,
                             const std::vector<int64_t>& extended_x_dims)
      : OneDNNHandlerNoCachingT<T, dnnl::binary>(engine, cpu_place) {
    const auto src0_tz = out->dims().size() == 0
                             ? std::vector<int64_t>{1}
                             : common::vectorize(out->dims());
    const auto src0_md = dnnl::memory::desc(
        src0_tz, OneDNNGetDataType<T>(), GetPlainOneDNNFormat(src0_tz.size()));
    const auto reshape_dims =
        extended_x_dims.size() != 0 ? extended_x_dims : std::vector<int64_t>{1};
    const auto src1_md = x->mem_desc().reshape(reshape_dims);

    dnnl::primitive_attr attributes;
    attributes.set_scales_mask(DNNL_ARG_SRC_0, 0);
    attributes.set_scales_mask(DNNL_ARG_SRC_1, 0);

    this->AcquireForwardPrimitiveDescriptor(
        attributes, algo, src0_md, src1_md, src0_md);
  }

  template <typename T_out = T>
  std::shared_ptr<dnnl::memory> AcquireZeroedDstMemory(DenseTensor* out) {
    T_out* ptr = out->mutable_data<T_out>(this->place_,
                                          this->fwd_pd_->dst_desc().get_size());
    memset(ptr, 0, this->fwd_pd_->dst_desc().get_size());
    return this->AcquireMemoryFromPrimitive(this->fwd_pd_->dst_desc(), ptr);
  }

  inline dnnl::memory Get_Scale_Memory(float scale_value) {
    auto scale_md = dnnl::memory::desc(
        {1}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::x);
    auto scale_mem = dnnl::memory(scale_md, this->engine_);
    static_cast<float*>(scale_mem.get_data_handle())[0] = scale_value;
    return scale_mem;
  }
};

template <typename T>
class PReluOneDNNHandler
    : public OneDNNHandlerNoCachingT<T,
                                     dnnl::prelu_forward,
                                     dnnl::prelu_backward> {
 public:
  PReluOneDNNHandler(const dnnl::engine engine,
                     Place cpu_place,
                     const DenseTensor& x,
                     const DenseTensor& weights,
                     const std::string& mode,
                     const std::string& data_format UNUSED,
                     const bool is_test)
      : OneDNNHandlerNoCachingT<T, dnnl::prelu_forward, dnnl::prelu_backward>(
            engine, cpu_place) {
    auto weights_dims = common::vectorize(weights.dims());
    // weights must have same size as X only for "element" case
    if (weights.dims().size() != x.dims().size()) {
      auto new_weights_dims = std::vector<int64_t>(x.dims().size(), 1);
      if (mode == "channel") {
        new_weights_dims[1] =
            *std::max_element(weights_dims.begin(), weights_dims.end());
      }
      weights_dims = std::move(new_weights_dims);
    }
    if (weights_dims.empty()) {
      weights_dims = std::vector<int64_t>{1};
    }
    auto weights_md = memory::desc(
        weights_dims, OneDNNGetDataType<T>(), memory::format_tag::any);

    this->AcquireForwardPrimitiveDescriptor(dnnl::prop_kind::forward_training,
                                            x.mem_desc(),
                                            weights_md,
                                            x.mem_desc());
    if (!is_test) {
      this->AcquireBackwardPrimitiveDescriptor(
          x.mem_desc(), weights_md, x.mem_desc(), weights_md, x.mem_desc());
    }
  }

  std::shared_ptr<memory> AcquireWeightsMemoryPossiblyWithReorder(
      const DenseTensor* weights, const bool is_test) {
    const float* weights_data = weights->data<float>();

    // if weights are 1D, every format tag is correct, so we accept
    // format_tag::any's output and no reorder is needed
    if (weights->dims().size() == 1) {
      return this->AcquireMemoryFromPrimitive(
          this->fwd_pd_->weights_desc(), to_void_cast<float>(weights_data));
    }

    return this->AcquireMemoryWithReorder(weights->mem_desc(),
                                          this->fwd_pd_->weights_desc(),
                                          to_void_cast<float>(weights_data),
                                          is_test);
  }

  std::shared_ptr<memory> AcquireDiffWeightsMemory(DenseTensor* output) {
    T* output_data = output->mutable_data<T>(
        this->place_, this->bwd_pd_->diff_weights_desc().get_size());
    return this->AcquireMemoryFromPrimitive(this->bwd_pd_->diff_weights_desc(),
                                            output_data);
  }
};

template <typename T>
class ReductionOneDNNHandler
    : public OneDNNHandlerNoCachingT<T, dnnl::reduction> {
 public:
  ReductionOneDNNHandler(const dnnl::algorithm algo,
                         const float p,
                         const float eps,
                         const dnnl::engine engine,
                         Place cpu_place,
                         const DenseTensor* x,
                         const DenseTensor* out UNUSED,
                         std::vector<int64_t> out_tz,
                         const dnnl::primitive_attr& attrs = NULL)
      : OneDNNHandlerNoCachingT<T, dnnl::reduction>(engine, cpu_place) {
    const auto out_md = memory::desc(
        out_tz, OneDNNGetDataType<T>(), dnnl::memory::format_tag::any);

    if (attrs)
      this->AcquireForwardPrimitiveDescriptor(
          attrs, algo, x->mem_desc(), out_md, p, eps);
    else
      this->AcquireForwardPrimitiveDescriptor(
          algo, x->mem_desc(), out_md, p, eps);
  }
};

template <typename T>
class ClipOneDNNHandler
    : public OneDNNHandlerNoCachingT<T,
                                     dnnl::eltwise_forward,
                                     dnnl::eltwise_backward> {
 public:
  ClipOneDNNHandler(const Scalar& min,
                    const Scalar& max,
                    const dnnl::engine engine,
                    Place cpu_place,
                    const DenseTensor* x)
      : OneDNNHandlerNoCachingT<T,
                                dnnl::eltwise_forward,
                                dnnl::eltwise_backward>(engine, cpu_place) {
    float alpha = min.to<float>();
    float beta = max.to<float>();

    this->AcquireForwardPrimitiveDescriptor(dnnl::prop_kind::forward_training,
                                            dnnl::algorithm::eltwise_clip_v2,
                                            x->mem_desc(),
                                            x->mem_desc(),
                                            alpha,
                                            beta);
  }

  ClipOneDNNHandler(const Scalar& min,
                    const Scalar& max,
                    const dnnl::engine engine,
                    Place cpu_place,
                    const DenseTensor* x,
                    const DenseTensor* dout)
      : OneDNNHandlerNoCachingT<T,
                                dnnl::eltwise_forward,
                                dnnl::eltwise_backward>(engine, cpu_place) {
    float alpha = min.to<float>();
    float beta = max.to<float>();

    this->AcquireForwardPrimitiveDescriptor(dnnl::prop_kind::forward_training,
                                            dnnl::algorithm::eltwise_clip_v2,
                                            x->mem_desc(),
                                            x->mem_desc(),
                                            alpha,
                                            beta);

    this->AcquireBackwardPrimitiveDescriptor(dnnl::algorithm::eltwise_clip_v2,
                                             dout->mem_desc(),
                                             dout->mem_desc(),
                                             x->mem_desc(),
                                             alpha,
                                             beta);
  }
  std::shared_ptr<dnnl::memory> AcquireBackwardSrcMemory(
      const DenseTensor* input) {
    const T* input_data = input->data<T>();
    return this->AcquireMemoryFromPrimitive(this->bwd_pd_->src_desc(),
                                            to_void_cast<T>(input_data));
  }
};

template <typename T>
class BatchNormOneDNNHandler
    : public OneDNNHandlerNoCachingT<T,
                                     dnnl::batch_normalization_forward,
                                     dnnl::batch_normalization_backward> {
 public:
  BatchNormOneDNNHandler(const dnnl::engine engine,
                         Place cpu_place,
                         const DenseTensor* x,
                         const float epsilon,
                         const bool use_scale,
                         const bool use_bias,
                         const bool fuse_with_relu,
                         const bool global_stats,
                         const bool test_mode)
      : OneDNNHandlerNoCachingT<T,
                                dnnl::batch_normalization_forward,
                                dnnl::batch_normalization_backward>(engine,
                                                                    cpu_place) {
    // Flags are added by bitwise OR operation
    auto flags = dnnl::normalization_flags::none;
    if (use_scale) flags |= dnnl::normalization_flags::use_scale;
    if (use_bias) flags |= dnnl::normalization_flags::use_shift;
    if (global_stats) flags |= dnnl::normalization_flags::use_global_stats;
    if (fuse_with_relu && test_mode)
      flags |= dnnl::normalization_flags::fuse_norm_relu;

    this->AcquireForwardPrimitiveDescriptor(
        global_stats ? dnnl::prop_kind::forward_inference
                     : dnnl::prop_kind::forward_training,
        x->mem_desc(),
        x->mem_desc(),
        epsilon,
        flags);
  }

  BatchNormOneDNNHandler(const dnnl::engine engine,
                         Place cpu_place,
                         const float epsilon,
                         const DenseTensor* in_x,
                         const bool use_scale,
                         const bool use_bias,
                         const DenseTensor* out_grad)
      : OneDNNHandlerNoCachingT<T,
                                dnnl::batch_normalization_forward,
                                dnnl::batch_normalization_backward>(engine,
                                                                    cpu_place) {
    auto flags = dnnl::normalization_flags::none;
    if (use_scale) flags |= dnnl::normalization_flags::use_scale;
    if (use_bias) flags |= dnnl::normalization_flags::use_shift;

    this->AcquireForwardPrimitiveDescriptor(dnnl::prop_kind::forward_training,
                                            in_x->mem_desc(),
                                            in_x->mem_desc(),
                                            epsilon,
                                            flags);
    this->AcquireBackwardPrimitiveDescriptor(dnnl::prop_kind::backward,
                                             out_grad->mem_desc(),
                                             out_grad->mem_desc(),
                                             in_x->mem_desc(),
                                             epsilon,
                                             flags);
  }

  std::shared_ptr<dnnl::memory> AcquireScaleMemory(const DenseTensor* scale) {
    auto scale_tz = common::vectorize(scale->dims());
    PADDLE_ENFORCE_EQ(
        scale_tz.size(),
        1,
        errors::InvalidArgument(
            "Dims of scale tensor must be 1, but received scale's size is %d",
            scale_tz.size()));

    auto scale_memory = this->AcquireMemoryFromPrimitive(
        this->fwd_pd_->weights_desc(), to_void_cast<T>(scale->data<T>()));

    return scale_memory;
  }

  std::shared_ptr<dnnl::memory> AcquireShiftMemory(const DenseTensor* shift) {
    auto shift_tz = common::vectorize(shift->dims());
    PADDLE_ENFORCE_EQ(
        shift_tz.size(),
        1,
        errors::InvalidArgument(
            "Dims of bias tensor must be 1, but received bias's size is %d",
            shift_tz.size()));

    auto shift_memory = this->AcquireMemoryFromPrimitive(
        this->fwd_pd_->weights_desc(), to_void_cast<T>(shift->data<T>()));

    return shift_memory;
  }

  std::shared_ptr<dnnl::memory> AcquireDiffScaleMemory(T* diff_scale_data) {
    auto diff_scale_memory = this->AcquireMemoryFromPrimitive(
        this->bwd_pd_->diff_weights_desc(), diff_scale_data);

    return diff_scale_memory;
  }

  std::shared_ptr<dnnl::memory> AcquireDiffShiftMemory(T* diff_shift_data) {
    auto diff_shift_memory = this->AcquireMemoryFromPrimitive(
        this->bwd_pd_->diff_weights_desc(), diff_shift_data);

    return diff_shift_memory;
  }

  std::shared_ptr<dnnl::memory> AcquireMeanMemory(const DenseTensor* mean) {
    const T* mean_data = mean->data<T>();
    return this->AcquireMemoryFromPrimitive(this->fwd_pd_->mean_desc(),
                                            to_void_cast<T>(mean_data));
  }

  std::shared_ptr<dnnl::memory> AcquireMeanMemory(DenseTensor* mean) {
    T* mean_data = mean->mutable_data<T>(this->place_,
                                         this->fwd_pd_->mean_desc().get_size());
    return this->AcquireMemoryFromPrimitive(this->fwd_pd_->mean_desc(),
                                            mean_data);
  }

  std::shared_ptr<dnnl::memory> AcquireVarianceMemory(
      const DenseTensor* variance) {
    const T* variance_data = variance->data<T>();
    return this->AcquireMemoryFromPrimitive(this->fwd_pd_->variance_desc(),
                                            to_void_cast<T>(variance_data));
  }

  std::shared_ptr<dnnl::memory> AcquireVarianceMemory(DenseTensor* variance) {
    T* variance_data = variance->mutable_data<T>(
        this->place_, this->fwd_pd_->variance_desc().get_size());
    return this->AcquireMemoryFromPrimitive(this->fwd_pd_->variance_desc(),
                                            variance_data);
  }
};

template <typename T>
class PoolingOneDNNHandler
    : public OneDNNHandlerNoCachingT<T,
                                     dnnl::pooling_forward,
                                     dnnl::pooling_backward> {
 public:
  PoolingOneDNNHandler(const OneDNNContext& dev_ctx,
                       const std::string& pooling_type,
                       const IntArray& kernel_size,
                       const std::vector<int>& strides,
                       const std::vector<int>& paddings,
                       bool global_pooling,
                       const std::string& padding_algorithm,
                       bool ceil_mode,
                       bool exclusive,
                       bool adaptive,
                       const DenseTensor* input,
                       DenseTensor* output)
      : OneDNNHandlerNoCachingT<T,
                                dnnl::pooling_forward,
                                dnnl::pooling_backward>(dev_ctx.GetEngine(),
                                                        dev_ctx.GetPlace()) {
    std::vector<int64_t> copied_kernel_size(kernel_size.GetData().begin(),
                                            kernel_size.GetData().end());
    std::vector<int64_t> copied_strides(strides.begin(), strides.end());
    std::vector<int64_t> copied_paddings(paddings.begin(), paddings.end());
    // Only 2D pooling is supported now
    PADDLE_ENFORCE_EQ(
        copied_kernel_size.size(),
        2,
        errors::InvalidArgument("The copied_kernel_size must be 2D, i.e. 2D "
                                "pooling, but received %dD.",
                                copied_kernel_size.size()));
    PADDLE_ENFORCE_EQ(
        pooling_type == "max" || pooling_type == "avg",
        true,
        errors::InvalidArgument(
            "The pooling_type must be 'max' or 'avg', but received %s.",
            pooling_type));
    PADDLE_ENFORCE_EQ(
        input->dims().size(),
        4,
        errors::InvalidArgument(
            "Input dim must be with 4, i.e. NCHW, but received %d.",
            input->dims().size()));

    const auto input_dims = input->dims();
    DDim data_dims = slice_ddim(input_dims, 2, input_dims.size());

    if (global_pooling) {
      UpdateKernelSize<int64_t>(&copied_kernel_size, data_dims);
    }

    UpdatePadding<int64_t>(&copied_paddings,
                           global_pooling,
                           0,
                           padding_algorithm,
                           data_dims,
                           copied_strides,
                           copied_kernel_size);

    auto onednn_paddings = ToOneDNNPadding(copied_paddings);

    const auto dt = ToOneDNNDataType(input->dtype());
    const auto src_tz = common::vectorize(input->dims());
    const auto dst_tz = common::vectorize(output->dims());
    const auto dst_md = OneDNNMemDesc(dst_tz, dt, OneDNNMemoryFormat::any);

    if (ceil_mode) {
      CorrectOutputSize(src_tz,
                        dst_tz,
                        copied_kernel_size,
                        copied_paddings,
                        copied_strides,
                        onednn_paddings[1]);
    }

    if (adaptive) {
      ComputeAdaptivePoolParameters(src_tz,
                                    onednn_paddings[0],
                                    onednn_paddings[1],
                                    &copied_kernel_size,
                                    &copied_strides);
    }

    bool is_test = dev_ctx.HasDnnAttr("is_test")
                       ? PADDLE_GET_CONST(bool, dev_ctx.GetDnnAttr("is_test"))
                       : false;

    memory::dims dilation = {0, 0};

    this->AcquireForwardPrimitiveDescriptor(
        is_test ? dnnl::prop_kind::forward_inference
                : dnnl::prop_kind::forward_training,
        pooling_type == "max"
            ? dnnl::algorithm::pooling_max
            : (exclusive ? dnnl::algorithm::pooling_avg_exclude_padding
                         : dnnl::algorithm::pooling_avg_include_padding),
        input->mem_desc(),
        dst_md,
        copied_strides,
        copied_kernel_size,
        dilation,
        onednn_paddings[0],
        onednn_paddings[1]);
  }

  PoolingOneDNNHandler(const OneDNNContext& dev_ctx,
                       const std::string& pooling_type,
                       const IntArray& kernel_size,
                       const std::vector<int>& strides,
                       const std::vector<int>& paddings,
                       bool global_pooling,
                       const std::string& padding_algorithm,
                       bool ceil_mode,
                       bool exclusive,
                       bool adaptive,
                       const DenseTensor* in_x,
                       const DenseTensor* out_grad,
                       DenseTensor* in_x_grad)

      : OneDNNHandlerNoCachingT<T,
                                dnnl::pooling_forward,
                                dnnl::pooling_backward>(dev_ctx.GetEngine(),
                                                        dev_ctx.GetPlace()) {
    bool is_test = dev_ctx.HasDnnAttr("is_test")
                       ? PADDLE_GET_CONST(bool, dev_ctx.GetDnnAttr("is_test"))
                       : false;

    PADDLE_ENFORCE_EQ(
        is_test,
        false,
        errors::InvalidArgument(
            "is_test attribute should be set to False in training phase."));

    std::vector<int64_t> copied_kernel_size(kernel_size.GetData().begin(),
                                            kernel_size.GetData().end());
    std::vector<int64_t> copied_strides(strides.begin(), strides.end());
    std::vector<int64_t> copied_paddings(paddings.begin(), paddings.end());
    auto in_x_dims = in_x->dims();
    DDim data_dims = slice_ddim(in_x_dims, 2, in_x_dims.size());
    if (global_pooling) {
      UpdateKernelSize<int64_t>(&copied_kernel_size, data_dims);
    }

    UpdatePadding<int64_t>(&copied_paddings,
                           global_pooling,
                           0,
                           padding_algorithm,
                           data_dims,
                           copied_strides,
                           copied_kernel_size);

    auto src_tz = common::vectorize<int64_t>(in_x->dims());
    auto diff_src_tz = common::vectorize<int64_t>(in_x_grad->dims());
    auto diff_dst_tz = common::vectorize<int64_t>(out_grad->dims());

    const auto dt = ToOneDNNDataType(in_x->dtype());
    auto dst_md = dnnl::memory::desc(diff_dst_tz, dt, OneDNNMemoryFormat::any);
    auto diff_src_md = dnnl::memory::desc(
        diff_src_tz, OneDNNGetDataType<T>(), OneDNNMemoryFormat::any);

    auto onednn_paddings = ToOneDNNPadding(copied_paddings);

    if (ceil_mode) {
      CorrectOutputSize(src_tz,
                        diff_dst_tz,
                        copied_kernel_size,
                        copied_paddings,
                        copied_strides,
                        onednn_paddings[1]);
    }

    if (adaptive) {
      ComputeAdaptivePoolParameters(src_tz,
                                    onednn_paddings[0],
                                    onednn_paddings[1],
                                    &copied_kernel_size,
                                    &copied_strides);
    }
    memory::dims dilation = {0, 0};

    this->AcquireForwardPrimitiveDescriptor(
        dnnl::prop_kind::forward_training,
        pooling_type == "max"
            ? dnnl::algorithm::pooling_max
            : (exclusive ? dnnl::algorithm::pooling_avg_exclude_padding
                         : dnnl::algorithm::pooling_avg_include_padding),
        in_x->mem_desc(),
        dst_md,
        copied_strides,
        copied_kernel_size,
        dilation,
        onednn_paddings[0],
        onednn_paddings[1]);

    this->AcquireBackwardPrimitiveDescriptor(
        pooling_type == "max"
            ? dnnl::algorithm::pooling_max
            : (exclusive ? dnnl::algorithm::pooling_avg_exclude_padding
                         : dnnl::algorithm::pooling_avg_include_padding),
        diff_src_md,
        out_grad->mem_desc(),
        copied_strides,
        copied_kernel_size,
        dilation,
        onednn_paddings[0],
        onednn_paddings[1]);
  }

  std::shared_ptr<dnnl::memory> AcquireWorkspaceMemory(
      const OneDNNContext& dev_ctx, const std::string& unique_name) {
    dnnl::memory::desc workspace_md = this->fwd_pd_->workspace_desc();
    // Pooling Workspace has to be passed to Grad op that
    // may be executed by different thread, hence
    // for that one we use key that does not contain TID
    std::string workspace_key = CreateKey(dev_ctx,
                                          workspace_md.get_dims(),
                                          workspace_md.get_data_type(),
                                          unique_name,
                                          "@wrk");
    auto mem_p =
        std::static_pointer_cast<dnnl::memory>(dev_ctx.GetBlob(workspace_key));
    if (mem_p == nullptr) {
      static std::mutex acquire_barrier;
      std::lock_guard<std::mutex> block_threads_until_finish_this_job(
          acquire_barrier);
      mem_p = std::static_pointer_cast<dnnl::memory>(
          dev_ctx.GetBlob(workspace_key));
      if (mem_p == nullptr) {
        mem_p = std::make_shared<dnnl::memory>(workspace_md, this->engine_);
        dev_ctx.SetBlob(workspace_key, mem_p);
      }
    }
    return mem_p;
  }

  static void ComputeAdaptivePoolParameters(
      const std::vector<int64_t>& src_tz,
      const std::vector<int64_t>& padding_l,
      const std::vector<int64_t>& padding_r,
      std::vector<int64_t>* kernel_size,
      std::vector<int64_t>* strides) {
    // https://github.com/oneapi-src/oneDNN/tree/bkocot/adaptive-pooling/rfcs/20200818-adaptive-pooling
    auto IH = static_cast<double>(src_tz[src_tz.size() - 2]);
    auto IW = static_cast<double>(src_tz[src_tz.size() - 1]);
    auto OH = static_cast<double>(kernel_size->at(0));
    auto OW = static_cast<double>(kernel_size->at(1));

    /*
    The previous calculation formula is given by OneDNN rfc, but in some odd
    cases(mod(I/O)>=O/2) there will be problems with the calculation results.
    Now change the formula to the general calculation formula of
    AdaptivePool when in mod(I/O)>=O/2 case:
    stride=floor(input_size/output_size)
    kernel_size=input_size-(output_size-1)*stride
    */
    int mod_H = IH - floor(IH / OH) * OH;
    int mod_W = IW - floor(IW / OW) * OW;
    if (2 * mod_H < OH && 2 * mod_W < OW) {
      strides->at(0) =
          static_cast<int64_t>(floor((IH * 2.0) / OH) - floor(IH / OH));
      strides->at(1) =
          static_cast<int64_t>(floor((IW * 2.0) / OW) - floor(IW / OW));
      kernel_size->at(0) =
          static_cast<int64_t>(ceil((IH * 2.0) / OH) - floor(IH / OH));
      kernel_size->at(1) =
          static_cast<int64_t>(ceil((IW * 2.0) / OW) - floor(IW / OW));
    } else {
      strides->at(0) = static_cast<int64_t>(floor(IH / OH));
      strides->at(1) = static_cast<int64_t>(floor(IW / OW));
      kernel_size->at(0) = static_cast<int64_t>(
          IH + padding_l[0] + padding_r[0] - floor((OH - 1) * strides->at(0)));
      kernel_size->at(1) = static_cast<int64_t>(
          IW + padding_l[1] + padding_r[1] - floor((OW - 1) * strides->at(1)));
    }
  }

 private:
  static inline int ComputeCeiledOutput(int input_size,
                                        int kernel_size,
                                        int padding,
                                        int stride) {
    return (input_size - kernel_size + 2 * padding) / stride + 1;
  }

  static inline void CorrectOutputSize(
      const std::vector<int64_t>& src_tz,
      const std::vector<int64_t>& dst_tz,
      const std::vector<int64_t>& kernel_size,
      const std::vector<int64_t>& paddings,
      const std::vector<int64_t>& strides,
      std::vector<int64_t>& right_bot_padding) {  // NOLINT
    for (size_t i = 0; i < right_bot_padding.size(); i++) {
      int desired_size = ComputeCeiledOutput(
          src_tz[i + 2], kernel_size[i], paddings[i], strides[i]);
      if (desired_size != dst_tz[i + 2]) {
        right_bot_padding[i] += strides[i] - 1;
      }
    }
  }
};

template <typename T>
class SoftplusOneDNNHandler : public OneDNNHandlerNoCachingT<T, dnnl::binary> {
 public:
  SoftplusOneDNNHandler(const OneDNNContext& dev_ctx,
                        const phi::DenseTensor* x,
                        const float beta,
                        const std::string& fuse_activation = "",
                        const float fuse_alpha = 0.0f,
                        const float fuse_beta = 0.0f)
      : OneDNNHandlerNoCachingT<T, dnnl::binary>(dev_ctx.GetEngine(),
                                                 dev_ctx.GetPlace()) {
    dnnl::post_ops post_ops;
    // According to oneDNN document
    // https://oneapi-src.github.io/oneDNN/dev_guide_eltwise.html, we set alpha
    // to 1.0f here to make the soft_relu formula equal to loge(1+e^x). So that
    // the whole handler formula will be (1.0f/beta)*loge(1+e^(x*beta)), equal
    // to Paddle softplus definition:
    // https://www.paddlepaddle.org.cn/documentation/docs/en/api/paddle/nn/Softplus_en.html
    post_ops.append_eltwise(dnnl::algorithm::eltwise_soft_relu, 1.0f, 0.0f);
    if (beta != 1.0f) {
      post_ops.append_eltwise(
          dnnl::algorithm::eltwise_linear, 1.0f / beta, 0.0f);
    }
    AppendActivation(dev_ctx, post_ops, fuse_activation, fuse_alpha, fuse_beta);
    dnnl::primitive_attr attrs;
    attrs.set_post_ops(post_ops);

    // if x is a 0-D tensor, then:
    //     x->dims() is [] and x->mem_desc().dims() is [1], we should use
    //     the later shape since oneDNN doesn't support 0-D shape.
    // else, then:
    //    x->dims() == x->mem_desc().dims()
    // so, we can directly use x->mem_desc().dims() here
    auto x_tz = x->mem_desc().get_dims();
    auto beta_tz = std::vector<int64_t>(x_tz.size(), 1);
    auto beta_md = dnnl::memory::desc(
        beta_tz, OneDNNGetDataType<T>(), GetPlainOneDNNFormat(x_tz.size()));

    this->AcquireForwardPrimitiveDescriptor(attrs,
                                            dnnl::algorithm::binary_mul,
                                            x->mem_desc(),
                                            beta_md,
                                            x->mem_desc());
  }

  std::shared_ptr<dnnl::memory> AcquireBetaMemory(const float* beta) {
    return this->AcquireMemoryFromPrimitive(this->fwd_pd_->src1_desc(),
                                            to_void_cast<float>(beta));
  }
};

static void SetOutMemDescWithUnsqueeze2FuseSupport(
    const std::vector<int> fused_unsqueeze2_axes,
    phi::DenseTensor* out,
    const dnnl::memory::desc& out_md) {
  const std::vector<int64_t>& op_tz = out_md.get_dims();
  std::vector<int64_t> unsqueezed_op_tz(
      op_tz.size() + fused_unsqueeze2_axes.size(), 0);

  for (const auto& axis : fused_unsqueeze2_axes) {
    int positive_axis = axis < 0 ? unsqueezed_op_tz.size() + axis : axis;
    unsqueezed_op_tz[positive_axis] = 1;
  }

  int j = 0;
  for (size_t i = 0; i < unsqueezed_op_tz.size(); ++i) {
    if (unsqueezed_op_tz[i] == 0) {
      unsqueezed_op_tz[i] = op_tz[j++];
    }
  }
  out->set_mem_desc(out_md.reshape(unsqueezed_op_tz));
  out->Resize(common::make_ddim(unsqueezed_op_tz));
}

static void SetOutMemDescWithReshape2FuseSupport(
    const std::vector<int> fused_reshape2_shape_,
    phi::DenseTensor* out,
    const dnnl::memory::desc& out_md) {
  std::vector<int64_t> fused_reshape2_shape(fused_reshape2_shape_.begin(),
                                            fused_reshape2_shape_.end());

  const int out_shape_numel = out->numel();
  const int new_shape_numel = std::accumulate(fused_reshape2_shape.begin(),
                                              fused_reshape2_shape.end(),
                                              1,
                                              std::multiplies<int64_t>());

  for (size_t i = 0; i < fused_reshape2_shape.size(); ++i) {
    if (fused_reshape2_shape[i] == -1) {
      fused_reshape2_shape[i] = -out_shape_numel / new_shape_numel;
      break;
    }
  }

  out->set_mem_desc(out_md.reshape(fused_reshape2_shape));
  out->Resize(common::make_ddim(fused_reshape2_shape));
}

}  // namespace funcs
}  // namespace phi
