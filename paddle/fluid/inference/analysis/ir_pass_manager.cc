// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/inference/analysis/ir_pass_manager.h"

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "paddle/common/errors.h"
#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/inference/analysis/argument.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/utils/string/pretty_log.h"

namespace paddle::inference::analysis {
using string::PrettyLogEndl;
using string::Style;

IRPassManager::IRPassManager(Argument *argument) : passes_() {
  disable_logs_ = argument->disable_logs();

  ARGUMENT_CHECK_FIELD(argument, ir_analysis_passes);
  CreatePasses(argument, argument->ir_analysis_passes());
}

void IRPassManager::CreatePasses(Argument *argument,
                                 const std::vector<std::string> &passes) {
  // For graph_viz_pass
  std::string pre_pass;
  int pass_num = 0;

  for (const std::string &pass_name : passes) {
    auto pass = framework::ir::PassRegistry::Instance().Get(pass_name);
    pass->Set("use_varseqlen", new bool(argument->tensorrt_use_varseqlen()));
    pass->Set("use_cutlass", new bool(argument->use_cutlass()));
    pass->Set("with_interleaved",
              new bool(argument->tensorrt_with_interleaved()));
    pass->Set("tensorrt_transformer_posid",
              new std::string(argument->tensorrt_transformer_posid()));
    pass->Set("tensorrt_transformer_maskid",
              new std::string(argument->tensorrt_transformer_maskid()));
    pass->Set("disable_logs", new bool(argument->disable_logs()));
    auto trt_precision_mode = argument->tensorrt_precision_mode();
    bool enable_int8 =
        trt_precision_mode == static_cast<int>(phi::DataType::INT8);
    pass->Set("enable_int8", new bool(enable_int8));
    pass->Set("max_input_shape",
              new std::map<std::string, std::vector<int>>(
                  argument->max_input_shape()));
    pass->Set("min_input_shape",
              new std::map<std::string, std::vector<int>>(
                  argument->min_input_shape()));
    pass->Set("optim_input_shape",
              new std::map<std::string, std::vector<int>>(
                  argument->optim_input_shape()));
    // Now, shape tensor value is not explicit set by user,
    // it is collected through API CollectShapeRangeInfo.
    pass->Set("max_shape_tensor",
              new std::map<std::string, std::vector<int>>());
    pass->Set("min_shape_tensor",
              new std::map<std::string, std::vector<int>>());
    pass->Set("optim_shape_tensor",
              new std::map<std::string, std::vector<int>>());

    // This gpu_device_id is used by some fp16 precision passes, so move it
    // here.
    pass->Set("gpu_device_id", new int(argument->gpu_device_id()));

    // tuned trt dynamic_shape
    pass->Set("trt_tuned_dynamic_shape",
              new bool(argument->tensorrt_tuned_dynamic_shape()));
    bool with_dynamic_shape = (!argument->max_input_shape().empty() &&
                               !argument->min_input_shape().empty() &&
                               !argument->optim_input_shape().empty()) ||
                              argument->tensorrt_tuned_dynamic_shape();
    pass->Set("with_dynamic_shape", new bool(with_dynamic_shape));

    // Mixed precision related.
    pass->Set(
        "mixed_black_list",
        new std::unordered_set<std::string>(argument->mixed_black_list()));
    pass->Set(
        "mixed_white_list",
        new std::unordered_set<std::string>(argument->mixed_white_list()));
    pass->Set("enable_gpu_mixed", new bool(argument->enable_gpu_mixed()));
    pass->Set("use_custom_device", new bool(argument->use_custom_device()));
    pass->Set("enable_custom_device_mixed",
              new bool(argument->enable_custom_device_mixed()));
    pass->Set("mixed_precision_mode",
              new int(argument->mixed_precision_mode()));
    pass->Set("model_precision", new int(argument->model_precision()));
    pass->Set("enable_low_precision_io",
              new bool(argument->enable_low_precision_io()));

    // "use_xpu" is used for passes in subgraphs.
    pass->Set("use_xpu", new bool(argument->use_xpu()));

    // "use_tensorrt" is used for passes in subgraphs.
    pass->Set("use_tensorrt", new bool(argument->use_tensorrt()));

    if (pass_name == "graph_viz_pass") {
      std::string optim_cache_dir = argument->optim_cache_dir();
      std::string dot_file_path;
      if (optim_cache_dir.empty()) {
        dot_file_path = std::to_string(pass_num) + "_ir_" +
                        (pre_pass.empty() ? "origin" : pre_pass) + ".dot";
      } else {
        dot_file_path = optim_cache_dir + "/" + std::to_string(pass_num) +
                        "_ir_" + (pre_pass.empty() ? "origin" : pre_pass) +
                        ".dot";
      }
      pass->Set("graph_viz_path", new std::string(std::move(dot_file_path)));
      pass->Set("optim_cache_dir", new std::string(std::move(optim_cache_dir)));
      pass_num++;
    } else if (pass_name == "onednn_placement_pass") {
      pass->Set("mkldnn_enabled_op_types",
                new std::unordered_set<std::string>(
                    argument->mkldnn_enabled_op_types()));
    } else if (pass_name == "cudnn_placement_pass") {
      pass->Set("cudnn_enabled_op_types",
                new std::unordered_set<std::string>());
#ifdef PADDLE_WITH_DNNL
    } else if (pass_name == "cpu_quantize_placement_pass") {
      pass->Set("quantize_enabled_op_types",
                new std::unordered_set<std::string>(
                    argument->quantize_enabled_op_types()));
      pass->Set(
          "quantize_excluded_op_ids",
          new std::unordered_set<int>(argument->quantize_excluded_op_ids()));
    } else if (pass_name == "cpu_quantize_pass") {
      if (argument->quantize_enabled_op_types().count("conv2d") ||
          argument->quantize_enabled_op_types().count("fused_conv2d") ||
          argument->quantize_enabled_op_types().count("depthwise_conv2d")) {
        pass->Set("data_layout", new std::string("NHWC"));
      }
      pass->Set("quant_var_scales",
                new VarQuantScale(argument->quant_var_scales()));
    } else if (pass_name == "cpu_bfloat16_placement_pass") {
      pass->Set("bfloat16_enabled_op_types",
                new std::unordered_set<std::string>(
                    argument->bfloat16_enabled_op_types()));
#endif
#ifdef PADDLE_WITH_OPENVINO
    } else if (pass_name == "openvino_subgraph_pass") {
      std::string model_program_path = argument->model_program_path();
      std::string model_params_path = argument->model_params_path();
      std::string optim_cache_dir = argument->optim_cache_dir();
      auto inference_precision = argument->openvino_inference_precision();
      int cpu_math_library_num_threads =
          argument->cpu_math_library_num_threads();
      if (optim_cache_dir.empty()) {
        optim_cache_dir = "./__cache__";
        LOG(INFO) << "optim_cache_dir is empty, set to ./__cache__";
      }
      if (!PathExists(optim_cache_dir)) {
        PADDLE_ENFORCE_NE(
            MKDIR(optim_cache_dir.c_str()),
            -1,
            common::errors::PreconditionNotMet(
                "Can not create optimize cache directory: %s, Make sure you "
                "have permission to write",
                optim_cache_dir));
      }
      pass->Set("model_program_path", new std::string(model_program_path));
      pass->Set("model_params_path", new std::string(model_params_path));
      pass->Set("model_opt_cache_dir", new std::string(optim_cache_dir));
      pass->Set("cpu_math_library_num_threads",
                new int(cpu_math_library_num_threads));
      pass->Set("inference_precision", new int(inference_precision));
#endif
    } else if (pass_name == "tensorrt_subgraph_pass") {
      pass->Set("workspace_size",
                new int64_t(argument->tensorrt_workspace_size()));
      pass->Set("max_batch_size", new int(argument->tensorrt_max_batch_size()));
      pass->Set("min_subgraph_size",
                new int(argument->tensorrt_min_subgraph_size()));
      pass->Set("mark_output", new bool(argument->trt_mark_output()));
      pass->Set(
          "output_tensor_names",
          new std::vector<std::string>(argument->trt_output_tensor_names()));
      pass->Set(
          "trt_exclude_var_names",
          new std::vector<std::string>(argument->trt_exclude_var_names()));
      pass->Set(
          "trt_parameter_run_fp16",
          new std::vector<std::string>(argument->trt_parameter_run_fp16()));
      pass->Set(
          "trt_parameter_run_int8",
          new std::vector<std::string>(argument->trt_parameter_run_int8()));
      pass->Set(
          "trt_parameter_run_bfp16",
          new std::vector<std::string>(argument->trt_parameter_run_bfp16()));
      pass->Set("forbid_dynamic_op",
                new bool(argument->trt_forbid_dynamic_op()));

      pass->Set("program",
                new framework::ProgramDesc *(&argument->main_program()));
      pass->Set("predictor_id", new int(argument->predictor_id()));
      bool use_calib_mode = argument->tensorrt_use_calib_mode();
      pass->Set("use_calib_mode", new bool(use_calib_mode));
      pass->Set("trt_precision_mode", new int(trt_precision_mode));
      pass->Set("context_memory_sharing",
                new bool(argument->trt_engine_memory_sharing()));
      pass->Set("use_cuda_graph",
                new bool(argument->tensorrt_use_cuda_graph()));
      bool use_static_engine = argument->tensorrt_use_static_engine();
      bool inspector_serialize = argument->tensorrt_inspector_serialize();
      bool model_from_memory = argument->model_from_memory();
      std::string optim_cache_dir = argument->optim_cache_dir();
      bool int8_valid = !(model_from_memory && optim_cache_dir.empty() &&
                          enable_int8 && use_calib_mode);
      PADDLE_ENFORCE_EQ(
          int8_valid,
          true,
          common::errors::PreconditionNotMet(
              "When you are in TRT INT8 mode, and load model from "
              "memory, you should set optim_cache_dir using "
              "config.SetOptimCacheDir()"));
      if (model_from_memory && use_static_engine) {
        PADDLE_ENFORCE_EQ(
            optim_cache_dir.empty(),
            false,
            common::errors::PreconditionNotMet(
                "When you are using Paddle-TRT, and using load model "
                "from memory, and also set the use_static to true. "
                "you must set optim_cache_dir using "
                "config.SetOptimCacheDir()."));
      }

      if (!optim_cache_dir.empty()) {
        if (!PathExists(optim_cache_dir)) {
          PADDLE_ENFORCE_NE(
              MKDIR(optim_cache_dir.c_str()),
              -1,
              common::errors::PreconditionNotMet(
                  "Can not create optimize cache directory: %s, Make sure you "
                  "have permission to write",
                  optim_cache_dir));
        }
        pass->Set("model_opt_cache_dir", new std::string(optim_cache_dir));
      } else if (use_static_engine || enable_int8 || with_dynamic_shape ||
                 inspector_serialize) {
        std::string model_opt_cache_dir =
            argument->Has("model_dir")
                ? argument->model_dir()
                : GetDirRoot(argument->model_program_path());
        pass->Set(
            "model_opt_cache_dir",
            new std::string(GetOrCreateModelOptCacheDir(model_opt_cache_dir)));
      }
      pass->Set("use_static_engine", new bool(use_static_engine));
      pass->Set("model_from_memory", new bool(argument->model_from_memory()));
      pass->Set("use_inspector", new bool(argument->tensorrt_use_inspector()));
      pass->Set("inspector_serialize",
                new bool(argument->tensorrt_inspector_serialize()));
      pass->Set("trt_ops_run_float",
                new std::unordered_set<std::string>(
                    argument->tensorrt_ops_run_float()));
      pass->Set("use_explicit_quantization",
                new bool(argument->tensorrt_use_explicit_quantization()));

      // tuned trt dynamic_shape
      pass->Set("trt_shape_range_info_path",
                new std::string(argument->tensorrt_shape_range_info_path()));
      pass->Set("trt_allow_build_at_runtime",
                new bool(argument->tensorrt_allow_build_at_runtime()));
      pass->Set(
          "trt_disabled_ops",
          new std::vector<std::string>(argument->tensorrt_disabled_ops()));
      pass->Set("trt_use_dla", new bool(argument->tensorrt_use_dla()));
      pass->Set("refit_params_path",
                new std::string(argument->refit_params_path()));
      pass->Set("trt_dla_core", new int(argument->tensorrt_dla_core()));
      pass->Set("optimization_level",
                new int(argument->tensorrt_optimization_level()));

      // Setting the disable_trt_plugin_fp16 to true means that TRT plugin will
      // not run fp16.
      pass->Set("disable_trt_plugin_fp16",
                new bool(argument->disable_trt_plugin_fp16()));
    } else if (pass_name == "memory_optimize_pass") {
      pass->Set("root_predictor_id", new int(argument->root_predictor_id()));
    } else if (pass_name == "build_cinn_pass") {
      pass->Set("is_inference_stage", new bool(argument->use_cinn_compiler()));
    } else if (pass_name == "fc_fuse_pass") {
      pass->Set("use_gpu", new bool(argument->use_gpu()));
      bool fc_onednn_pass = false;
      for (const std::string &pass_n : passes) {
        if (pass_n == "fc_onednn_pass") {
          fc_onednn_pass = true;
        }
      }
      bool use_fc_padding = !fc_onednn_pass && argument->use_fc_padding();
      pass->Set("use_fc_padding", new bool(use_fc_padding));
    } else if (pass_name == "fused_multi_transformer_xpu_pass") {
      int quant_post_dynamic_weight_precision =
          argument->xpu_quant_post_dynamic_weight_precision();
      if (quant_post_dynamic_weight_precision == 0) {
        pass->Set("quant_post_dynamic_weight_precision ", new int(0));
      }
    } else if (pass_name == "fc_xpu_fuse_pass") {
      std::map<std::string, int> quant_post_type =
          argument->xpu_quant_post_dynamic_weight_methods();
      if (!quant_post_type.empty()) {
        pass->Set("quant_post_dynamic_weight_methods",
                  new std::map<std::string, int>(quant_post_type));
      }
    } else if (pass_name == "conv2d_xpu_fuse_pass") {
      std::map<std::string, int> quant_post_type =
          argument->xpu_quant_post_dynamic_weight_methods();
      if (!quant_post_type.empty()) {
        pass->Set("quant_post_dynamic_weight_methods",
                  new std::map<std::string, int>(quant_post_type));
      }
    }
    pre_pass = pass_name;

    passes_.emplace_back(std::move(pass));
  }
}

std::unique_ptr<Graph> IRPassManager::Apply(std::unique_ptr<Graph> graph) {
  PADDLE_ENFORCE_NOT_NULL(
      graph.get(), common::errors::InvalidArgument("Graph cannot be null."));
  // Apply all the passes
  for (const auto &pass : passes_) {
    if (pass->Type() != "graph_viz_pass" && !disable_logs_) {
      PrettyLogEndl(Style::H2(), "--- Running IR pass [%s]", pass->Type());
    }
    graph.reset(pass->Apply(graph.release()));
  }
  return graph;
}

}  // namespace paddle::inference::analysis
