/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
Copyright (c) 2022 NVIDIA Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/details/build_strategy.h"

#include <glog/logging.h>

#include "paddle/common/flags.h"
#include "paddle/fluid/framework/ir/graph_printer.h"
#include "paddle/fluid/framework/ir/multi_devices_graph_pass/multi_devices_graph_pass.h"

PD_DECLARE_bool(convert_all_blocks);
COMMON_DECLARE_bool(use_mkldnn);
#ifdef PADDLE_WITH_CINN
PD_DECLARE_bool(use_cinn);
#endif

namespace paddle::framework::details {

static inline void ConvertDefaultValue(paddle::optional<bool> *default_value) {
  if (*default_value == paddle::none) {
    *default_value = true;
  }
}

class ParallelExecutorPassBuilder : public ir::PassBuilder {
 public:
  explicit ParallelExecutorPassBuilder(const BuildStrategy &strategy)
      : ir::PassBuilder(), strategy_(strategy) {
    ResolveOptionConfliction();

    AppendPrintGraphPass("graph_viz_pass", "_original_graph");

#ifdef PADDLE_WITH_CINN
    if (FLAGS_use_cinn || strategy.build_cinn_pass_) {
      // Note: This is a trick to support 0D-Tensor for CINN. This pass will be
      // removed in the near future.
      AppendPrintGraphPass("graph_viz_pass", "_build_cinn_graph");
    }
#endif

    AppendPassWithCheck(strategy_.sync_batch_norm_, "sync_batch_norm_pass");

    AppendOpFusePasses();
    AppendPrintGraphPass("graph_viz_pass", "_fused_graph");

    AppendMultiDevPass();
    AppendPassToSetMkldnnAttr("onednn_placement_pass");
    // runtime_context_cache pass should be the last pass to enable the attr of
    // all original and fused operators. But no operators can be enabled this
    // attr if putting it after MultiDevPass.
    // AppendPassWithCheck(strategy_.cache_runtime_context_,
    //                     "runtime_context_cache_pass");
  }

  void ResolveOptionConfliction() {
    // Specifies the restrictions between different pass.
    if (strategy_.enable_parallel_graph_) {
      LOG_IF(WARNING, strategy_.fuse_all_optimizer_ops_ == true)
          << "Currently, fuse_all_optimizer_ops doesn't work under "
             "parallel_graph.";
      strategy_.fuse_all_optimizer_ops_ = false;
      LOG_IF(WARNING, strategy_.fuse_all_reduce_ops_ == true)
          << "fuse_all_reduce_ops doesn't work under "
             "parallel_graph.";
      strategy_.fuse_all_reduce_ops_ = false;
    }
    if (strategy_.reduce_ == BuildStrategy::ReduceStrategy::kReduce) {
      LOG_IF(WARNING, strategy_.fuse_all_optimizer_ops_ == true)
          << "Currently, fuse_all_optimizer_ops only works under AllReduce "
             "mode.";
      strategy_.fuse_all_optimizer_ops_ = false;
      LOG_IF(WARNING, strategy_.fuse_all_reduce_ops_ == true)
          << "fuse_all_optimizer_ops only works under AllReduce "
             "mode.";
      strategy_.fuse_all_reduce_ops_ = false;
    }
    if (strategy_.reduce_ == BuildStrategy::ReduceStrategy::kAllReduce) {
      LOG_IF(WARNING, strategy_.fuse_broadcast_ops_ == true)
          << "Currently, fuse_broadcast_ops only works under Reduce "
             "mode.";
      strategy_.fuse_broadcast_ops_ = false;
    }

    ConvertDefaultValue(&strategy_.fuse_all_optimizer_ops_);
    ConvertDefaultValue(&strategy_.fuse_all_reduce_ops_);
    ConvertDefaultValue(&strategy_.fuse_broadcast_ops_);

    if (strategy_.fuse_all_optimizer_ops_ == true) {
      LOG_IF(WARNING, strategy_.async_mode_)
          << "Currently, fuse_all_optimizer_ops doesn't work under "
             "async mode.";
      strategy_.fuse_all_optimizer_ops_ = !strategy_.async_mode_;
    }
    if (strategy_.fuse_all_reduce_ops_ == true) {
      LOG_IF(WARNING, strategy_.async_mode_)
          << "Currently, fuse_all_reduce_ops doesn't work under "
             "async mode.";
      strategy_.fuse_all_reduce_ops_ = !strategy_.async_mode_;
    }
  }

  void AppendOpFusePasses() {
    // 1. inference pass if enabled.
    AppendPassWithCheck(
        strategy_.enable_inference_pass_ && strategy_.delete_dropout_,
        "delete_dropout_op_x_pass");
    AppendPassWithCheck(
        strategy_.enable_inference_pass_ && strategy_.use_mkldnn_,
        "onednn_placement_pass");

    // 2. training pass
#ifdef PADDLE_WITH_CUDNN_FRONTEND
    AppendPassWithCheck(strategy_.fuse_dot_product_attention_,
                        "fuse_dot_product_attention_pass");
    AppendPassWithCheck(strategy_.fuse_resunit_, "fuse_resunit_pass");
#endif
    AppendPassWithCheck(strategy_.fuse_relu_depthwise_conv_,
                        "fuse_relu_depthwise_conv_pass");
    AppendPassWithCheck(strategy_.fuse_bn_act_ops_, "fuse_bn_act_pass");
    AppendPassWithCheck(strategy_.fuse_bn_add_act_ops_, "fuse_bn_add_act_pass");
#if (defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)) && \
    !defined(_WIN32) && !defined(__APPLE__)
    AppendPassWithCheck(strategy_.enable_auto_fusion_, "fusion_group_pass");
#endif

#ifdef PADDLE_WITH_CUDA
    AppendPassWithCheck(strategy_.fused_attention_, "fused_attention_pass");
    AppendPassWithCheck(strategy_.fuse_adamw_, "fuse_adamw_op_pass");
#endif

#if (defined(PADDLE_WITH_CUDA) && CUDA_VERSION >= 11060)
    AppendPassWithCheck(strategy_.fuse_gemm_epilogue_,
                        "fuse_gemm_epilogue_pass");
#endif
    AppendPassWithCheck(strategy_.fuse_elewise_add_act_ops_,
                        "fuse_elewise_add_act_pass");
    // for single card training, fuse_all_reduce_ops is unnecessary.
    // coalesce_grad_tensor_pass should be before of MultiDevPass.
    AppendPassWithCheck(strategy_.fuse_all_reduce_ops_,
                        "coalesce_grad_tensor_pass");
    // Fuse all the optimization operators.
    // NOTE: fuse_all_xx_ops will count the number of xx operator first,
    // if the number is zero, fuse_all_reduce_ops will do nothing.
    // Currently, only one type of optimization algorithm can be fused.
    if (strategy_.fuse_all_optimizer_ops_ == true) {
      AppendPass("fuse_adam_op_pass");
      AppendPass("fuse_sgd_op_pass");
      AppendPass("fuse_momentum_op_pass");
    }
#ifdef PADDLE_WITH_CUDA
    AppendPassWithCheck(strategy_.fused_feedforward_, "fused_feedforward_pass");
#endif
  }

  // Convert graph to run on multi-devices.
  void AppendMultiDevPass() {
    ir::Pass *multi_devices_pass = nullptr;
    switch (strategy_.reduce_) {
      case BuildStrategy::ReduceStrategy::kAllReduce:
        multi_devices_pass =
            AppendPass("all_reduce_mode_multi_devices_pass").get();
        break;
      default:
        PADDLE_THROW(common::errors::Unimplemented("Unknown reduce strategy."));
    }
    multi_devices_pass->SetNotOwned<const BuildStrategy>("strategy",
                                                         &strategy_);
  }
  void AppendPrintGraphPass(const std::string &pass_name,
                            const std::string &debug_file_suffix) {
    if (!strategy_.debug_graphviz_path_.empty()) {
      auto viz_pass = AppendPass(pass_name);
      const std::string graph_path = string::Sprintf(
          "%s%s", strategy_.debug_graphviz_path_.c_str(), debug_file_suffix);
      viz_pass->Set<std::string>(ir::kGraphvizPath,
                                 new std::string(graph_path));
    }
  }

  void AppendPassWithCheck(const paddle::optional<bool> &append_pass,
                           const std::string &pass_name) {
    AppendPassWithCheck(append_pass == true, pass_name);
  }

  void AppendPassWithCheck(bool append_pass, const std::string &pass_name) {
    if (append_pass) {
      AppendPass(pass_name);
    }
  }

  void AppendPassToSetMkldnnAttr(const std::string &pass_name) {
#ifdef PADDLE_WITH_DNNL
    if (FLAGS_use_mkldnn) {
      AppendPass(pass_name);
    } else if (!strategy_.mkldnn_enabled_op_types_.empty()) {
      VLOG(1) << "mkldnn_enabled_op_types specify the operator type list to "
                 "use MKLDNN acceleration. It is null in default, means "
                 "that all the operators supported by MKLDNN will be "
                 "accelerated. And it should not be set when "
                 "FLAGS_use_mkldnn=false.";
    }
#else
    PADDLE_ENFORCE_NE(FLAGS_use_mkldnn,
                      true,
                      common::errors::PreconditionNotMet(
                          "FLAGS_use_mkldnn has been set to True, but "
                          "PaddlePaddle is compiled without MKLDNN. "
                          "Please compile PaddlePaddle with MKLDNN first."));
#endif
  }

 private:
  BuildStrategy strategy_;
};

std::shared_ptr<ir::PassBuilder> BuildStrategy::CreatePassesFromStrategy(
    bool finalize_strategy) const {
  if (is_finalized_) {
    return pass_builder_;
  }
  pass_builder_.reset(new ParallelExecutorPassBuilder(*this));
  if (finalize_strategy) {
    is_finalized_ = true;
  }
  return pass_builder_;
}

bool BuildStrategy::IsMultiDevPass(const std::string &pass_name) const {
  return framework::ir::MultiDevSSAGraphBuilder().count(pass_name) > 0;
}

ir::Graph *BuildStrategy::Apply(ir::Graph *graph,
                                const std::vector<phi::Place> &places,
                                const std::string &loss_var_name,
                                const std::vector<Scope *> &local_scopes,
                                const size_t &nranks,
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
                                DeviceType use_device,
                                platform::NCCLCommunicator *nccl_ctxs) const {
#elif defined(PADDLE_WITH_XPU) && defined(PADDLE_WITH_XPU_BKCL)
                                DeviceType use_device,
                                platform::BKCLCommunicator *bkcl_ctxs) const {
#else
                                DeviceType use_device) const {
#endif
  VLOG(1) << "apply all passes";
  if (FLAGS_convert_all_blocks) {
    PADDLE_ENFORCE_EQ(
        graph->IsMainGraph(),
        true,
        common::errors::InvalidArgument("This graph is not main_graph"));
  }
  // Create a default one if not finalized by user.
  CreatePassesFromStrategy(false);

  for (std::shared_ptr<ir::Pass> &pass : pass_builder_->AllPasses()) {
    VLOG(1) << "BuildStrategy::Apply pass:" << pass->Type();
    if (IsMultiDevPass(pass->Type())) {
      pass->Erase(kPlaces);
      pass->SetNotOwned<const std::vector<phi::Place>>(kPlaces, &places);
      pass->Erase(ir::kLossVarName);
      pass->SetNotOwned<const std::string>(ir::kLossVarName, &loss_var_name);
      pass->Erase(kLocalScopes);
      pass->SetNotOwned<const std::vector<Scope *>>(kLocalScopes,
                                                    &local_scopes);
      pass->Erase(kNRanks);
      pass->Set<size_t>(kNRanks, new size_t(nranks));

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
      platform::NCCLCommunicator *nctx =
          (use_device == p::kCUDA) ? nccl_ctxs : nullptr;
      pass->Erase(kNCCLCtxs);
      pass->SetNotOwned<platform::NCCLCommunicator>(kNCCLCtxs, nctx);
#elif defined(PADDLE_WITH_XPU) && defined(PADDLE_WITH_XPU_BKCL)
      // ToDo: more check
      platform::BKCLCommunicator *bkcl_ctx =
          (use_device == p::kXPU) ? bkcl_ctxs : nullptr;
      pass->Erase(kBKCLCtxs);
      pass->SetNotOwned<platform::BKCLCommunicator>(kBKCLCtxs, bkcl_ctx);
#endif
    } else if (pass->Type() == "coalesce_grad_tensor_pass") {
      pass->Erase(kNRanks);
      pass->Set<size_t>(kNRanks, new size_t(nranks));
    } else if (pass->Type() == "fuse_relu_depthwise_conv_pass") {
      if (use_device != p::kCUDA) {
        VLOG(1) << "fuse_relu_depthwise_conv_pass is only supported on "
                   "GPU, skipped.";
        continue;
      }
    } else if (pass->Type() == "fusion_group_pass") {
      pass->Set<bool>("use_gpu", new bool((use_device == p::kCUDA)));
      if (use_device != p::kCUDA) {
        VLOG(1) << "fusion_group_pass is only supported on GPU, skipped.";
        continue;
      }
    } else if (pass->Type() == "fuse_bn_act_pass") {
      if (use_device != p::kCUDA) {
        VLOG(1) << "fuse_bn_act_pass is only supported on "
                   "GPU, skipped.";
        continue;
      }
    } else if (pass->Type() == "fuse_bn_add_act_pass") {
      if (use_device != p::kCUDA) {
        VLOG(1) << "fuse_bn_add_act_pass is only supported on "
                   "GPU, skipped.";
        continue;
      }
    } else if (pass->Type() == "onednn_placement_pass") {
      pass->Set("mkldnn_enabled_op_types",
                new std::unordered_set<std::string>(mkldnn_enabled_op_types_));
    }
    VLOG(1) << "Start Apply Pass " << pass->Type();
    if (FLAGS_convert_all_blocks) {
      for (size_t i = 0; i < graph->SubGraphsSize(); ++i) {
        VLOG(3) << "Apply Pass " << pass->Type() << "to SubGraph " << i;
        pass->Apply(graph->GetSubGraph(i));
      }
    } else {
      graph = pass->Apply(graph);
    }
    VLOG(1) << "Finish Apply Pass " << pass->Type();
  }
  VLOG(1) << "All Passes Applied";
  return graph;
}

}  // namespace paddle::framework::details

USE_PASS(sync_batch_norm_pass);
USE_PASS(fuse_relu_depthwise_conv_pass);
USE_PASS(fuse_elewise_add_act_pass);
USE_PASS(fuse_bn_act_pass);
USE_PASS(fuse_bn_add_act_pass);
USE_PASS(graph_viz_pass);
USE_PASS(multi_batch_merge_pass);
USE_PASS(all_reduce_mode_multi_devices_pass);
USE_PASS(coalesce_grad_tensor_pass);
USE_PASS(fuse_adam_op_pass);
USE_PASS(fuse_sgd_op_pass);
USE_PASS(fuse_momentum_op_pass);
USE_PASS(runtime_context_cache_pass);
USE_PASS(delete_dropout_op_x_pass);
#ifdef PADDLE_WITH_CUDA
USE_PASS(fused_attention_pass);
USE_PASS(fuse_adamw_op_pass);
#endif
#ifdef PADDLE_WITH_CUDA
USE_PASS(fused_feedforward_pass);
#endif
#ifdef PADDLE_WITH_DNNL
USE_PASS(onednn_placement_pass);
#endif
#if (defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)) && \
    !defined(_WIN32) && !defined(__APPLE__)
USE_PASS(fusion_group_pass);
#endif
#if (defined(PADDLE_WITH_CUDA) && CUDA_VERSION >= 11060)
USE_PASS(fuse_gemm_epilogue_pass);
#endif
#ifdef PADDLE_WITH_CUDNN_FRONTEND
USE_PASS(fuse_dot_product_attention_pass);
USE_PASS(fuse_resunit_pass);
#endif
