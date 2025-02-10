/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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

#ifdef PADDLE_WITH_CUDA
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "paddle/common/errors.h"
#include "paddle/fluid/framework/data_device_transform.h"
#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/inference/analysis/helper.h"
#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"
#include "paddle/fluid/inference/tensorrt/engine.h"
#include "paddle/fluid/inference/tensorrt/helper.h"
#include "paddle/fluid/inference/tensorrt/trt_int8_calibrator.h"
#include "paddle/fluid/inference/utils/io_utils.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/memory/memcpy.h"
#include "paddle/phi/kernels/cast_kernel.h"
#include "paddle/phi/kernels/funcs/data_type_transform.h"
#include "paddle/utils/string/string_helper.h"
// #include "paddle/fliud/inference/api/paddle_analysis_config.h"

namespace paddle {
namespace inference {
namespace tensorrt {
class TRTCalibratorEngine;
class TRTCalibratorEngineManager;
class TRTInt8Calibrator;
}  // namespace tensorrt
template <typename T>
struct Singleton;
}  // namespace inference
}  // namespace paddle

namespace paddle {

namespace operators {

using inference::Singleton;
using inference::tensorrt::TensorRTEngine;
using inference::tensorrt::TRTCalibratorEngine;
using inference::tensorrt::TRTCalibratorEngineManager;
using inference::tensorrt::TRTInt8Calibrator;

static void RuntimeStaticShapeCheck(std::vector<int64_t> runtime_input_shape,
                                    std::vector<int64_t> model_input_shape) {
  std::string model_input_shape_str =
      string::join_strings(model_input_shape, ',');
  std::string runtime_input_shape_str =
      string::join_strings(runtime_input_shape, ',');
  PADDLE_ENFORCE_EQ(
      model_input_shape == runtime_input_shape,
      true,
      common::errors::InvalidArgument(
          "Input shapes are inconsistent with the model. Expect [%s] in "
          "model description, but got [%s] in runtime. TRT 5 "
          "or lower version "
          "does not support dynamic input shapes. Please check and "
          "modify "
          "your input shapes.",
          model_input_shape_str,
          runtime_input_shape_str));
}

static phi::DataType TRT2FluidDataType(nvinfer1::DataType type) {
  switch (type) {
    case nvinfer1::DataType::kFLOAT:
      return phi::DataType::FLOAT32;
    case nvinfer1::DataType::kINT32:
      return phi::DataType::INT32;
    case nvinfer1::DataType::kHALF:
      return phi::DataType::FLOAT16;
    case nvinfer1::DataType::kINT8:
      return phi::DataType::INT8;
#if IS_TRT_VERSION_GE(7000)
    case nvinfer1::DataType::kBOOL:
      return phi::DataType::BOOL;
#endif
    default:
      PADDLE_THROW(common::errors::InvalidArgument(
          "unknown fluid datatype in Fluid op converter"));
      return phi::DataType::FLOAT32;
  }
}

static void RuntimeDynamicShapeCheck(
    const std::string &x,
    const std::vector<int32_t> &runtime_input_shape,
    const std::vector<int32_t> &min_input_shape,
    const std::vector<int32_t> &max_input_shape) {
  // PADDLE_ENFORCE_EQ(
  //     runtime_input_shape.size(), min_input_shape.size(),
  //     common::errors::InvalidArgument(
  //         "TRT engine runtime input %s dims size(%d) inconsistent "
  //         "with the dynamic shape size(%d)",
  //         x, runtime_input_shape.size(), min_input_shape.size()));
  auto is_input_shape_valid =
      [&](const std::vector<int32_t> &runtime_input_shape,
          const std::vector<int32_t> &min_input_shape,
          const std::vector<int32_t> &max_input_shape) -> bool {
    for (size_t i = 0; i < runtime_input_shape.size(); i++) {
      if (runtime_input_shape[i] <= max_input_shape[i] &&
          runtime_input_shape[i] >= min_input_shape[i]) {
        continue;
      } else {
        return false;
      }
    }
    return true;
  };
  std::string runtime_input_shape_str =
      string::join_strings(runtime_input_shape, ',');
  std::string min_input_shape_str = string::join_strings(min_input_shape, ',');
  std::string max_input_shape_str = string::join_strings(max_input_shape, ',');
  PADDLE_ENFORCE_EQ(is_input_shape_valid(
                        runtime_input_shape, min_input_shape, max_input_shape),
                    true,
                    common::errors::InvalidArgument(
                        "TRT runtime input shape of %s is invalid. Expect "
                        "runtime input shape to be within min/max input shape "
                        "configured in SetTRTDynamicShapeInfo(),"
                        "but got runtime input shape = [%s], min input shape = "
                        "[%s], max input shape = [%s].",
                        x,
                        runtime_input_shape_str,
                        min_input_shape_str,
                        max_input_shape_str));
}

class TensorRTEngineOp : public framework::OperatorBase {
 private:
  std::vector<std::string> input_names_;
  std::unordered_set<std::string> param_names_;
  std::vector<std::string> runtime_input_names_;
  mutable TensorRTEngine *trt_engine_{nullptr};
  int max_batch_size_;
  int64_t workspace_size_;
  std::unique_ptr<TRTInt8Calibrator> calibrator_;
  bool enable_int8_;
  bool enable_fp16_;
  bool use_calib_mode_;
  std::string calibration_data_;
  std::string engine_key_;
  std::string calibration_engine_key_;
  bool calibration_mode_;
  int predictor_id_;
  int device_id_;
  bool allow_build_at_runtime_{false};
  bool with_dynamic_shape_{false};
  std::string shape_range_info_path_;
  std::string model_opt_cache_dir_;
  bool use_static_engine_;
  phi::DataType precision_mode_;

 public:
  TensorRTEngineOp(const std::string &type,
                   const framework::VariableNameMap &inputs,
                   const framework::VariableNameMap &outputs,
                   const framework::AttributeMap &attrs)
      : framework::OperatorBase(type, inputs, outputs, attrs) {
    input_names_ = Inputs("Xs");
    max_batch_size_ = Attr<int>("max_batch_size");
    workspace_size_ = Attr<int64_t>("workspace_size");
    device_id_ = Attr<int>("gpu_device_id");
    enable_int8_ = Attr<bool>("enable_int8");
    enable_fp16_ = Attr<bool>("enable_fp16");
    use_calib_mode_ = Attr<bool>("use_calib_mode");
    calibration_data_ = Attr<std::string>("calibration_data");
    engine_key_ = Attr<std::string>("engine_key");
    calibration_engine_key_ = Attr<std::string>("calibration_engine_key");
    predictor_id_ = Attr<int>("predictor_id");
    shape_range_info_path_ = Attr<std::string>("shape_range_info_path");
    allow_build_at_runtime_ = Attr<bool>("allow_build_at_runtime");
    with_dynamic_shape_ = Attr<bool>("with_dynamic_shape");
    use_static_engine_ = Attr<bool>("use_static_engine");
    if (use_static_engine_) {
      model_opt_cache_dir_ = Attr<std::string>("model_opt_cache_dir");
    }

    auto params = Attr<std::vector<std::string>>("parameters");
    for (const auto &param : params) {
      param_names_.insert(param);
    }
    for (auto &x : input_names_) {
      if (param_names_.count(x)) continue;
      runtime_input_names_.emplace_back(x);
    }
    // calibration_mode is true represents we need to
    // generate the calibration table data.
    calibration_mode_ =
        (enable_int8_ && calibration_data_.empty() && use_calib_mode_);

    VLOG(4) << "calibration_mode: " << calibration_mode_;
    if (enable_int8_ && !calibration_data_.empty()) {
      calibrator_ = std::make_unique<TRTInt8Calibrator>(calibration_data_);
    }
    bool has_engine =
        inference::Singleton<inference::tensorrt::TRTEngineManager>::Global()
            .Has(engine_key_ + std::to_string(predictor_id_));

    if (!calibration_mode_ && has_engine) {
      trt_engine_ =
          inference::Singleton<inference::tensorrt::TRTEngineManager>::Global()
              .Get(engine_key_ + std::to_string(predictor_id_));
    }
    precision_mode_ = phi::DataType::FLOAT32;
    if (enable_int8_) {
      precision_mode_ = phi::DataType::INT8;
    }
    if (enable_fp16_) {
      precision_mode_ = phi::DataType::FLOAT16;
    }
  }

  void PrepareTRTEngine(const framework::Scope &scope,
                        TensorRTEngine *engine) const {
    LOG(INFO) << "Prepare TRT engine (Optimize model structure, Select OP "
                 "kernel etc). This process may cost a lot of time.";
    framework::proto::BlockDesc block_proto;
    block_proto.ParseFromString(Attr<std::string>("subgraph"));
    framework::BlockDesc block_desc(nullptr, &block_proto);

    std::vector<std::string> inputs = Inputs("Xs");
    std::vector<std::string> outputs =
        Attr<std::vector<std::string>>("output_name_mapping");

    inference::Singleton<inference::tensorrt::OpConverter>::Global()
        .ConvertBlockToTRTEngine(
            &block_desc, scope, inputs, param_names_, outputs, engine);
  }

 protected:
  void RunNativeImpl(const framework::Scope &scope,
                     const phi::Place &dev_place) const {
    framework::Executor executor(dev_place);
    auto *block = Attr<framework::BlockDesc *>("sub_block");
    auto *program = block->Program();
    auto &current_scope = scope.NewScope();
    auto ctx = executor.Prepare(*program, block->ID());
    executor.RunPreparedContext(ctx.get(), &current_scope, false, true, true);
  }

  void RunImpl(const framework::Scope &scope,
               const phi::Place &dev_place) const override {
    if (calibration_mode_ == true) {
      RunCalibration(scope, dev_place);
      return;
    }
    auto *trt_engine = GetEngine(scope, dev_place);
    if (trt_engine->with_dynamic_shape()) {
      // get runtime input shapes and shape tensors.
      std::map<std::string, std::vector<int32_t>> runtime_input_shape;
      std::map<std::string, std::vector<int32_t>> runtime_shape_tensor;
      for (auto name : runtime_input_names_) {
        // NOTE(liuyuanle): It is a trick. If you need a [name], then you need
        // to use [name.substr(0, idx)].
        // Maybe we insert suffix of "_cast_auto_mixed.tmp_" in
        // auto_mixed_precision_pass.
        std::string name_real = name;
        auto idx = name.find("_cast_auto_mixed.tmp_");
        name = name.substr(0, idx);

        auto &t = inference::analysis::GetFromScope<phi::DenseTensor>(
            scope, name_real);
        VLOG(4) << "trt engine runtime input name(" << name << "), dims("
                << t.dims() << ")";
        auto t_shape = common::vectorize<int32_t>(t.dims());
        runtime_input_shape.insert(std::make_pair(name, t_shape));
        // We need collect value range for shape tensor for Paddle-TRT's use.
        // To be noticed, this method to identify all inputs/outputs is shape
        // tensors; After, TRT Engine gets whether it is a real shape tensor.
        auto is_shape_tensor = true;
        if (trt_engine->engine()) {
          auto *engine = trt_engine->engine();
#if IS_TRT_VERSION_GE(8600)
          is_shape_tensor = engine->isShapeInferenceIO(name.c_str());
#else
          is_shape_tensor =
              engine->isShapeBinding(engine->getBindingIndex(name.c_str()));
#endif
          if (!is_shape_tensor) {
            runtime_shape_tensor.erase(name);
            VLOG(4) << "trt engine runtime delete shape name(" << name
                    << "), dims(" << t.dims() << ")";
          }
        }
        if ((t.dtype() == phi::DataType::INT32 ||
             t.dtype() == phi::DataType::INT64) &&
            is_shape_tensor) {
          std::vector<int> int32_host(t.numel());
          phi::DeviceContextPool &pool = phi::DeviceContextPool::Instance();

          if (t.place().GetType() == phi::AllocationType::CPU) {
            auto &int32_tensor = t;
            if (t.dtype() == phi::DataType::INT64) {
              auto *cpu_ctx = pool.Get(phi::CPUPlace());
              int32_tensor = phi::funcs::TransDataType(
                  reinterpret_cast<const phi::CPUContext &>(*cpu_ctx),
                  t,
                  DataType::INT32);
            }
            phi::memory_utils::Copy(phi::CPUPlace(),
                                    int32_host.data(),
                                    phi::CPUPlace(),
                                    int32_tensor.data<int>(),
                                    int32_tensor.numel() * sizeof(int));
          } else if (t.place().GetType() == phi::AllocationType::GPU) {
#if defined(PADDLE_WITH_CUDA)
            auto *dev_ctx = pool.Get(t.place());
            auto &int32_tensor = t;
            if (t.dtype() == phi::DataType::INT64) {
              int32_tensor = phi::funcs::TransDataType(
                  reinterpret_cast<const phi::GPUContext &>(*dev_ctx),
                  t,
                  DataType::INT32);
            }
            phi::memory_utils::Copy(phi::CPUPlace(),
                                    int32_host.data(),
                                    int32_tensor.place(),
                                    int32_tensor.data<int>(),
                                    int32_tensor.numel() * sizeof(int),
                                    nullptr);
#endif
          }
          runtime_shape_tensor[name] = int32_host;
        }
      }
      if (!allow_build_at_runtime_) {
        std::map<std::string, std::vector<int>> min_input_shape =
            trt_engine->min_input_shape();
        std::map<std::string, std::vector<int>> max_input_shape =
            trt_engine->max_input_shape();
        for (auto x : runtime_input_names_) {
          // NOTE(liuyuanle): It is a trick. If you need a [x], then you need
          // to use [x.substr(0, idx)].
          // Maybe we insert suffix of "_cast_auto_mixed.tmp_" in
          // auto_mixed_precision_pass.
          auto idx = x.find("_cast_auto_mixed.tmp_");
          x = x.substr(0, idx);

          PADDLE_ENFORCE_EQ(
              min_input_shape.count(x),
              true,
              common::errors::InvalidArgument(
                  "Input %s not found in TRT engine min_input_shape.", x));
          PADDLE_ENFORCE_EQ(
              max_input_shape.count(x),
              true,
              common::errors::InvalidArgument(
                  "Input %s not found in TRT engine max_input_shape.", x));
          RuntimeDynamicShapeCheck(x,
                                   runtime_input_shape[x],
                                   min_input_shape[x],
                                   max_input_shape[x]);
        }
      } else {
        // compare runtime_input_shape and trt_engine dynamic shapes.
        std::vector<std::string> shape_changed_name;
        std::vector<std::string> tensor_changed_name;
        bool is_adjusted =
            trt_engine->AdjustDynamicShapeRange(runtime_input_shape,
                                                runtime_shape_tensor,
                                                &shape_changed_name,
                                                &tensor_changed_name);
        if (is_adjusted) {
          LOG(INFO) << "Adjust dynamic shape range, rebuild trt engine!";
          if (trt_engine->engine()) {
            trt_engine->ResetContext();
            trt_engine->ClearTensorMap();
          }
          auto *anc = scope.parent();
          while (anc && anc->parent()) {
            anc = anc->parent();
          }
          if (anc == nullptr) {
            anc = &scope;
          }
          PrepareTRTEngine(*anc, trt_engine);
          // update shape_range_info_pbtxt
          if (!shape_range_info_path_.empty()) {
            inference::UpdateShapeRangeInfo(shape_range_info_path_,
                                            trt_engine->min_input_shape(),
                                            trt_engine->max_input_shape(),
                                            trt_engine->optim_input_shape(),
                                            trt_engine->min_shape_tensor(),
                                            trt_engine->max_shape_tensor(),
                                            trt_engine->optim_shape_tensor(),
                                            shape_changed_name,
                                            tensor_changed_name);
          }

          if (use_static_engine_) {
            nvinfer1::IHostMemory *serialized_engine_data =
                trt_engine->Serialize();
            std::string trt_engine_serialized_data =
                std::string((const char *)serialized_engine_data->data(),
                            serialized_engine_data->size());
            inference::analysis::SaveTrtEngineSerializedDataToFile(
                inference::analysis::GetTrtEngineSerializedPath(
                    model_opt_cache_dir_, engine_key_),
                trt_engine_serialized_data);
            LOG(INFO) << "Save TRT Optimized Info to "
                      << inference::analysis::GetTrtEngineSerializedPath(
                             model_opt_cache_dir_, engine_key_);
          }
        }
      }
    }
    RunTrt(scope, dev_place, trt_engine);
  }

  void RunCalibration(const framework::Scope &scope,
                      const phi::Place &dev_place) const {
    // This process will builds a 32-bit trt engine, runs it on the calibration
    // set, and records a histogram for each
    // tensor of the distribution of activation values.
    LOG_FIRST_N(INFO, 1) << "This process is generating calibration table for "
                            "Paddle TRT int8...";

    int runtime_batch = 1;
    if (!Singleton<TRTCalibratorEngineManager>::Global().Has(
            calibration_engine_key_)) {
      TRTCalibratorEngine *calib_res =
          Singleton<TRTCalibratorEngineManager>::Global().Create(
              calibration_engine_key_);
      std::unordered_map<std::string, size_t> calib_buffers;
      for (auto &x : input_names_) {
        if (param_names_.count(x)) continue;
        auto &t = inference::analysis::GetFromScope<phi::DenseTensor>(scope, x);
        calib_buffers[x] = t.memory_size();
        auto t_shape = common::vectorize(t.dims());
        runtime_batch = t_shape[0];
      }
      calib_res->calib_ = std::make_unique<TRTInt8Calibrator>(
          calib_buffers, runtime_batch, calibration_engine_key_, dev_place);
      calib_res->thr_.reset(new std::thread([&]() {
        TensorRTEngine::ConstructionParams params;
        params.max_batch_size = max_batch_size_;
        params.max_workspace_size = workspace_size_;
        params.precision = precision_mode_;
        params.calibrator = calib_res->calib_.get();
        params.device_id = dev_place.device;
        params.with_dynamic_shape = with_dynamic_shape_;
        if (!shape_range_info_path_.empty()) {
          inference::DeserializeShapeRangeInfo(shape_range_info_path_,
                                               &params.min_input_shape,
                                               &params.max_input_shape,
                                               &params.optim_input_shape,
                                               &params.min_shape_tensor,
                                               &params.max_shape_tensor,
                                               &params.optim_shape_tensor);
        } else {
          if (HasAttr("dynamic_shape_names") &&
              HasAttr("min_input_shape_vector") &&
              HasAttr("max_input_shape_vector") &&
              HasAttr("opt_input_shape_vector")) {
            std::vector<std::string> dynamic_shape_names;
            std::vector<std::vector<int>> min_input_shapes;
            std::vector<std::vector<int>> max_input_shapes;
            std::vector<std::vector<int>> opt_input_shapes;
            std::vector<int> dynamic_shape_lens;
            dynamic_shape_names =
                Attr<std::vector<std::string>>("dynamic_shape_names");
            std::vector<int> min_shapes =
                Attr<std::vector<int>>("min_input_shape_vector");
            std::vector<int> max_shapes =
                Attr<std::vector<int>>("max_input_shape_vector");
            std::vector<int> opt_shapes =
                Attr<std::vector<int>>("opt_input_shape_vector");
            dynamic_shape_lens = Attr<std::vector<int>>("dynamic_shape_lens");
            int idx = 0;
            for (size_t i = 0; i < dynamic_shape_lens.size(); ++i) {
              std::vector<int> tmp1, tmp2, tmp3;
              for (int j = 0; j < dynamic_shape_lens[i]; ++j) {
                tmp1.push_back(min_shapes[idx]);
                tmp2.push_back(max_shapes[idx]);
                tmp3.push_back(opt_shapes[idx++]);
              }
              min_input_shapes.emplace_back(tmp1);
              max_input_shapes.emplace_back(tmp2);
              opt_input_shapes.emplace_back(tmp3);
            }

            for (size_t i = 0; i < dynamic_shape_names.size(); ++i) {
              params.min_input_shape.insert(
                  std::make_pair(dynamic_shape_names[i], min_input_shapes[i]));
              params.max_input_shape.insert(
                  std::make_pair(dynamic_shape_names[i], max_input_shapes[i]));
              params.optim_input_shape.insert(
                  std::make_pair(dynamic_shape_names[i], opt_input_shapes[i]));
            }
          }
        }
        params.context_memory_sharing = Attr<bool>("context_memory_sharing");
        params.enable_low_precision_io = Attr<bool>("enable_low_precision_io");
        calib_res->engine_ = std::make_unique<TensorRTEngine>(params);

        VLOG(3) << "start the calib trt engine thread";
        PrepareTRTEngine(scope, calib_res->engine_.get());
      }));
    }

    TRTInt8Calibrator *temp_calibrator =
        Singleton<TRTCalibratorEngineManager>::Global()
            .Get(calibration_engine_key_)
            ->calib_.get();
    std::unordered_map<std::string, void *> calib_data;

    for (auto &x : Inputs("Xs")) {
      if (param_names_.count(x)) continue;
      auto &t = inference::analysis::GetFromScope<phi::DenseTensor>(scope, x);
      calib_data.emplace(x, t.data());
    }
    temp_calibrator->setBatch(calib_data);
    RunNativeImpl(scope, dev_place);
  }

  void RunTrt(const framework::Scope &scope,
              const phi::Place &dev_place,
              TensorRTEngine *engine) const {
    int runtime_batch = -1;
    phi::DeviceContextPool &pool = phi::DeviceContextPool::Instance();
    auto &dev_ctx = *pool.Get(dev_place);
    auto stream = reinterpret_cast<const phi::GPUContext &>(dev_ctx).stream();
    std::vector<std::string> output_maps =
        Attr<std::vector<std::string>>("output_name_mapping");

    // Get the total over all profiles
    const int num_bindings = engine->GetNbBindings();
    std::vector<void *> buffers(num_bindings, nullptr);

    int binding_offset = 0;
    nvinfer1::IExecutionContext *trt_context = nullptr;
    if (engine->with_dynamic_shape()) {
      // Initialize context and get offset by profile index
      trt_context = engine->context();
      binding_offset = engine->GetBindingsOffset();
    }
    // Bind input tensor to TRT.
#if IS_TRT_VERSION_GE(8600)
    std::unordered_map<std::string, int> tensor_index;
    for (int i = 0; i < engine->engine()->getNbIOTensors(); ++i) {
      auto tensor_name = engine->engine()->getIOTensorName(i);
      tensor_index[std::string(tensor_name)] = i;
    }
#endif
    for (auto x : runtime_input_names_) {
      // NOTE(liuyuanle): It is a trick. If you need a [x], then you need
      // to use [x.substr(0, idx)].
      // Maybe we insert suffix of "_cast_auto_mixed.tmp_" in
      // auto_mixed_precision_pass.
      std::string x_real = x;
      auto idx = x.find("_cast_auto_mixed.tmp_");
      x = x.substr(0, idx);

#if IS_TRT_VERSION_LT(8000)
      // trt may remove input tensor if it's unused or used only at compile-time
      if (engine->engine()->getBindingIndex(x.c_str()) < 0) continue;
#endif

      // convert input and copy to TRT engine's buffer
      auto &t =
          inference::analysis::GetFromScope<phi::DenseTensor>(scope, x_real);
      PADDLE_ENFORCE_GT(
          t.numel(),
          0,
          common::errors::InvalidArgument(
              "The input tensor named %s of trt-subgraph must "
              "have >0 elements, but now have %d elements. "
              "It's likely that this tensor is connected to a Concat op inside "
              "a trt-subgraph, "
              "try to use API to forbid this op into trt-subgraph.",
              x,
              t.numel()));

      // check the input_tensor
      if (!(t.place().GetType() == phi::AllocationType::GPU)) {
        phi::DenseTensor out;
        phi::Copy(dev_ctx, t, dev_place, false, &out);
        t.ShareDataWith(out);
      }
      auto t_shape = common::vectorize<int64_t>(t.dims());

      // This must be a zero dimension tensor.
      // At present, we convert it to a 1D tensor to feed them into Trt.
      if (t_shape.empty()) {
        PADDLE_ENFORCE_EQ(
            t.numel(),
            1UL,
            common::errors::PreconditionNotMet(
                "This tensor must have one element, but got %ld.", t.numel()));
        t_shape.push_back(1);
      }

      // Get index of profile 0 first, then plus binding offset
#if IS_TRT_VERSION_GE(8600)
      const int bind_index =
          tensor_index[std::string(x.c_str())] + binding_offset;
#else
      const int bind_index =
          engine->engine()->getBindingIndex(x.c_str()) + binding_offset;
#endif
      PADDLE_ENFORCE_LT(
          bind_index,
          num_bindings,
          common::errors::InvalidArgument(
              "Wrong TRT engine input binding index. Expected The "
              "binding index of TRT engine input to be less than "
              "the number of inputs and outputs. Received binding "
              "index=%d >= total inputs and outputs=%d",
              bind_index,
              num_bindings));
      if (!engine->with_dynamic_shape()) {
        // check if the input shapes are consistent with model.
        if (HasAttr(x + "_shape")) {
          std::vector<int64_t> i_shape =
              Attr<std::vector<int64_t>>(x + "_shape");
          std::vector<int64_t> model_input_shape(i_shape.begin() + 1,
                                                 i_shape.end());
          std::vector<int64_t> runtime_input_shape(t_shape.begin() + 1,
                                                   t_shape.end());
          RuntimeStaticShapeCheck(runtime_input_shape, model_input_shape);
          if (runtime_batch != -1) {
            PADDLE_ENFORCE_EQ(
                runtime_batch,
                t_shape[0],
                common::errors::InvalidArgument(
                    "Inputs of trt subgraphs has different batchsize. "
                    "It's not allowed in static shape mode. "
                    "Check whether the model you are running has multiple trt "
                    "subgraphs: \n "
                    "\tIf there are multiple trt subgraphs, you need to ensure "
                    "that the first dimension of the input tensor of these "
                    "subgraphs is "
                    "consistent.\n"
                    "\tIf there are inconsistent subgraphs, you need to filter "
                    "them "
                    "by "
                    "setting min_subgraph_size using EnableTensorrtEngine "
                    "interface.\n"
                    "\tThe min_subgraph_size should to be greater than the "
                    "number "
                    "of "
                    "nodes in the inconsistent subgraph.\n"));
          }
        }
      } else {
#if IS_TRT_VERSION_GE(6000)
#if IS_TRT_VERSION_GE(8500)
        if (engine->engine()->isShapeInferenceIO(x.c_str()) &&
            engine->engine()->getTensorIOMode(x.c_str()) ==
                nvinfer1::TensorIOMode::kINPUT) {
          std::vector<int> shape_v(t.numel());
          if (t.dtype() == phi::DataType::INT32) {
            phi::memory_utils::Copy(phi::CPUPlace(),
                                    shape_v.data(),
                                    t.place(),
                                    t.data<int32_t>(),
                                    t.numel() * sizeof(int),
                                    nullptr);
          } else if (t.dtype() == phi::DataType::INT64) {
            std::string x_t = x + "_cast_to_INT32";
            if (scope.FindVar(x_t) == nullptr) {
              const_cast<framework::Scope *>(&scope)->Var(x_t);
            }
            auto int32_tensor =
                scope.FindVar(x_t)->GetMutable<phi::DenseTensor>();
            *int32_tensor = phi::Cast<int64_t>(
                reinterpret_cast<const phi::GPUContext &>(dev_ctx),
                t,
                phi::DataType::INT32);
            phi::memory_utils::Copy(phi::CPUPlace(),
                                    shape_v.data(),
                                    int32_tensor->place(),
                                    int32_tensor->data<int32_t>(),
                                    int32_tensor->numel() * sizeof(int),
                                    nullptr);
          }
          trt_context->setTensorAddress(x.c_str(), shape_v.data());
        } else {
          trt_context->setInputShape(
              x.c_str(), inference::tensorrt::Vec2TRT_Dims(t_shape, x, true));
        }
#else
        trt_context->setBindingDimensions(
            bind_index, inference::tensorrt::Vec2TRT_Dims(t_shape, x, true));
        // If this x is a shape tensor, we need call setInputShapeBinding
        if (engine->engine()->isShapeBinding(bind_index) &&
            engine->engine()->bindingIsInput(bind_index)) {
          std::vector<int> shape_v(t.numel());
          if (t.dtype() == phi::DataType::INT32) {
            phi::memory_utils::Copy(phi::CPUPlace(),
                                    shape_v.data(),
                                    t.place(),
                                    t.data<int32_t>(),
                                    t.numel() * sizeof(int),
                                    nullptr);
          } else if (t.dtype() == phi::DataType::INT64) {
            std::string x_t = x + "_cast_to_INT32";
            if (scope.FindVar(x_t) == nullptr) {
              const_cast<framework::Scope *>(&scope)->Var(x_t);
            }
            auto int32_tensor =
                scope.FindVar(x_t)->GetMutable<phi::DenseTensor>();
            *int32_tensor = phi::Cast<int64_t>(
                reinterpret_cast<const phi::GPUContext &>(dev_ctx),
                t,
                phi::DataType::INT32);
            phi::memory_utils::Copy(phi::CPUPlace(),
                                    shape_v.data(),
                                    int32_tensor->place(),
                                    int32_tensor->data<int32_t>(),
                                    int32_tensor->numel() * sizeof(int),
                                    nullptr);
          }
          trt_context->setInputShapeBinding(bind_index, shape_v.data());
        }
#endif
#endif
      }
      runtime_batch = t_shape[0];
      VLOG(1) << "trt input [" << x << "] dtype is " << t.dtype();

      auto indata_type = inference::tensorrt::PhiType2NvType(t.dtype());
#if IS_TRT_VERSION_GE(8600)
      auto intrt_type = engine->engine()->getTensorDataType(x.c_str());
#else
      auto intrt_index = engine->engine()->getBindingIndex(x.c_str());
      auto intrt_type = engine->engine()->getBindingDataType(intrt_index);
#endif
      PADDLE_ENFORCE_EQ(indata_type,
                        intrt_type,
                        common::errors::InvalidArgument(
                            "The TRT Engine OP's input type [%d] should equal "
                            "to the input data type [%d].",
                            static_cast<int>(intrt_type),
                            static_cast<int>(indata_type)));

      if (t.dtype() == phi::DataType::FLOAT32) {
        buffers[bind_index] = static_cast<void *>(t.data<float>());
      } else if (t.dtype() == phi::DataType::FLOAT64) {
        std::string x_t = x + "_cast_to_FP32";
        if (scope.FindVar(x_t) == nullptr) {
          const_cast<framework::Scope *>(&scope)->Var(x_t);
        }
        auto fp32_tensor = scope.FindVar(x_t)->GetMutable<phi::DenseTensor>();
        *fp32_tensor = phi::Cast<double>(
            reinterpret_cast<const phi::GPUContext &>(dev_ctx),
            t,
            phi::DataType::FLOAT32);
        buffers[bind_index] = static_cast<void *>(fp32_tensor->data<float>());
      } else if (t.dtype() == phi::DataType::INT64) {
        std::string x_t = x + "_cast_to_INT32";
        if (scope.FindVar(x_t) == nullptr) {
          const_cast<framework::Scope *>(&scope)->Var(x_t);
        }
        auto int32_tensor = scope.FindVar(x_t)->GetMutable<phi::DenseTensor>();
        *int32_tensor = phi::Cast<int64_t>(
            reinterpret_cast<const phi::GPUContext &>(dev_ctx),
            t,
            phi::DataType::INT32);
        buffers[bind_index] =
            static_cast<void *>(int32_tensor->data<int32_t>());
      } else if (t.dtype() == phi::DataType::INT32) {
        buffers[bind_index] = static_cast<void *>(t.data<int32_t>());
      } else if (t.dtype() == phi::DataType::FLOAT16) {
        buffers[bind_index] = static_cast<void *>(t.data<float16>());
#if IS_TRT_VERSION_GE(8400)
      } else if (t.dtype() == phi::DataType::BOOL) {
        buffers[bind_index] = static_cast<void *>(t.data<bool>());
#endif
      } else {
        PADDLE_THROW(common::errors::Fatal(
            "The TRT Engine OP only support "
            "float/double/int32_t/int64_t/float16/bool input."));
      }
    }

    // Bind output tensor to TRT.
    int output_index = 0;
    std::vector<int> origin_output_rank =
        Attr<std::vector<int>>("origin_output_rank");
    VLOG(4) << "TensorRT Engine Op Outputs:";
    for (const auto &y : Outputs("Ys")) {
#if IS_TRT_VERSION_GE(8600)
      const int bind_index =
          tensor_index[std::string(output_maps[output_index].c_str())] +
          binding_offset;
#else
      const int bind_index =
          engine->engine()->getBindingIndex(output_maps[output_index].c_str()) +
          binding_offset;
#endif
      std::vector<int> ddim;

      if (!engine->with_dynamic_shape()) {
#if IS_TRT_VERSION_GE(8600)
        auto dims =
            engine->engine()->getTensorShape(output_maps[output_index].c_str());
#else
        auto dims = engine->engine()->getBindingDimensions(bind_index);
#endif
        for (int i = 0; i < dims.nbDims; i++) {
          ddim.push_back(dims.d[i]);
        }
      } else {
#if IS_TRT_VERSION_GE(8500)
        auto x_name = engine->engine()->getIOTensorName(bind_index);
        auto dims = trt_context->getTensorShape(x_name);
        int nb_dims = dims.nbDims;
        for (; nb_dims > 0; nb_dims--) {
          // some 'x 1' of shape is normal, no need to remove it
          if (dims.d[nb_dims - 1] != 1 ||
              nb_dims == origin_output_rank[output_index])
            break;
        }
        for (int i = 0; i < nb_dims; i++) {
          ddim.push_back(dims.d[i]);
        }
#else
        auto dims = trt_context->getBindingDimensions(bind_index);
        int nb_dims = dims.nbDims;
        for (; nb_dims > 0; nb_dims--) {
          // some 'x 1' of shape is normal, no need to remove it
          if (dims.d[nb_dims - 1] != 1 ||
              nb_dims == origin_output_rank[output_index])
            break;
        }
        for (int i = 0; i < nb_dims; i++) {
          ddim.push_back(dims.d[i]);
        }
#endif
      }
      auto *fluid_v = scope.FindVar(y);
      PADDLE_ENFORCE_NOT_NULL(
          fluid_v,
          common::errors::NotFound(
              "Output variable %s is not found in TensorRT subgraph.", y));
      auto *fluid_t = fluid_v->GetMutable<phi::DenseTensor>();
      fluid_t->Resize(common::make_ddim(ddim));

      PADDLE_ENFORCE_LT(bind_index,
                        num_bindings,
                        common::errors::InvalidArgument(
                            "The binding index in TRT engine should be less "
                            "than the number of bindings, but got binding "
                            "index = %d, number of bindings = %d.",
                            bind_index,
                            num_bindings));
#if IS_TRT_VERSION_GE(8600)
      auto trt_tensor_name = engine->engine()->getIOTensorName(bind_index);
      auto trt_type = engine->engine()->getTensorDataType(trt_tensor_name);
#else
      auto trt_type = engine->engine()->getBindingDataType(bind_index);
#endif
      // get adr and set type
      VLOG(1) << "trt output [" << y << "] dtype is "
              << TRT2FluidDataType(trt_type);
      buffers[bind_index] = static_cast<void *>(
          fluid_t->mutable_data(dev_place, TRT2FluidDataType(trt_type)));
      output_index += 1;
    }

    if (!engine->with_dynamic_shape()) {
      PADDLE_ENFORCE_LE(
          runtime_batch,
          max_batch_size_,
          common::errors::InvalidArgument(
              "The runtime batch size (%d) is greater than the max batch "
              "size(%d).\n"
              "There are two possible causes for this problem: \n"
              "1. Check whether the runtime batch is larger than the max_batch "
              "set by EnableTensorrtEngine()\n"
              "2. Check whether the model you are running has multiple trt "
              "subgraphs: \n "
              "\tIf there are multiple trt subgraphs, you need to ensure that "
              "the first dimension of the input tensor of these subgraphs is "
              "consistent.\n"
              "\tIf there are inconsistent subgraphs, you need to filter them "
              "by "
              "setting min_subgraph_size using EnableTensorrtEngine "
              "interface.\n"
              "\tThe min_subgraph_size should to be greater than the number "
              "of "
              "nodes in the inconsistent subgraph.\n",
              runtime_batch,
              max_batch_size_));
    }
    // Execute the engine.
    engine->Execute(runtime_batch, &buffers, stream);

    std::vector<int> origin_outputs_dtype =
        Attr<std::vector<int>>("origin_outputs_dtype");
    for (size_t i = 0; i < Outputs("Ys").size(); i++) {
      auto type =
          static_cast<framework::proto::VarType_Type>(origin_outputs_dtype[i]);

      if (type == framework::proto::VarType::INT64) {
        auto y = Outputs("Ys")[i];
        auto *fluid_v = scope.FindVar(y);
        auto *fluid_t = fluid_v->GetMutable<phi::DenseTensor>();
        std::string y_t = y + "_cast_to_INT64";
        if (scope.FindVar(y_t) == nullptr) {
          const_cast<framework::Scope *>(&scope)->Var(y_t);
        }
        auto int32_tensor = scope.FindVar(y_t)->GetMutable<phi::DenseTensor>();
        int32_tensor->Resize(fluid_t->dims());
        dev_ctx.Alloc<int32_t>(int32_tensor);
        phi::Copy(dev_ctx, *fluid_t, dev_place, false, int32_tensor);
        *fluid_t = phi::Cast<int32_t>(
            reinterpret_cast<const phi::GPUContext &>(dev_ctx),
            *int32_tensor,
            phi::DataType::INT64);
      } else if (type == framework::proto::VarType::FP64) {
        auto y = Outputs("Ys")[i];
        auto *fluid_v = scope.FindVar(y);
        auto *fluid_t = fluid_v->GetMutable<phi::DenseTensor>();
        std::string y_t = y + "_cast_to_FP64";
        if (scope.FindVar(y_t) == nullptr) {
          const_cast<framework::Scope *>(&scope)->Var(y_t);
        }
        auto fp32_tensor = scope.FindVar(y_t)->GetMutable<phi::DenseTensor>();
        fp32_tensor->Resize(fluid_t->dims());
        dev_ctx.Alloc<float>(fp32_tensor);
        phi::Copy(dev_ctx, *fluid_t, dev_place, false, fp32_tensor);
        *fluid_t =
            phi::Cast<float>(reinterpret_cast<const phi::GPUContext &>(dev_ctx),
                             *fp32_tensor,
                             phi::DataType::FLOAT64);
      }
    }
  }

  bool IsPersistable(const framework::VarDesc *var) const {
    if (var->Persistable() &&
        var->GetType() != framework::proto::VarType::FEED_MINIBATCH &&
        var->GetType() != framework::proto::VarType::FETCH_LIST &&
        var->GetType() != framework::proto::VarType::RAW) {
      return true;
    }
    return false;
  }

  TensorRTEngine *GetEngine(const framework::Scope &scope,
                            const phi::Place &dev_place) const {
    if (!trt_engine_) {
      TensorRTEngine::ConstructionParams params;
      params.max_batch_size = max_batch_size_;
      params.max_workspace_size = workspace_size_;
      params.precision = precision_mode_;
      params.calibrator = calibrator_.get();
      params.device_id = dev_place.device;
      params.with_dynamic_shape = with_dynamic_shape_;
      if (HasAttr("context_memory_sharing")) {
        params.context_memory_sharing = Attr<bool>("context_memory_sharing");
      }
      if (HasAttr("use_dla")) {
        params.use_dla = Attr<bool>("use_dla");
      }
      if (HasAttr("dla_core")) {
        params.dla_core = Attr<int>("dla_core");
      }
      if (HasAttr("refit_params_path")) {
        params.refit_params_path = Attr<std::string>("refit_params_path");
      }
      LOG(INFO) << "params.refit_params_path=" << params.refit_params_path;
      if (HasAttr("disable_trt_plugin_fp16")) {
        params.disable_trt_plugin_fp16 = Attr<bool>("disable_trt_plugin_fp16");
      }
      if (HasAttr("enable_low_precision_io")) {
        params.enable_low_precision_io = Attr<bool>("enable_low_precision_io");
      }
      if (HasAttr("use_inspector")) {
        params.use_inspector = Attr<bool>("use_inspector");
      }
      if (HasAttr("engine_info_path")) {
        params.engine_info_path = Attr<std::string>("engine_info_path");
      }
      if (HasAttr("optimization_level")) {
        params.optimization_level = Attr<int>("optimization_level");
      }
      if (!shape_range_info_path_.empty()) {
        inference::DeserializeShapeRangeInfo(shape_range_info_path_,
                                             &params.min_input_shape,
                                             &params.max_input_shape,
                                             &params.optim_input_shape,
                                             &params.min_shape_tensor,
                                             &params.max_shape_tensor,
                                             &params.optim_shape_tensor);
      } else {
        if (HasAttr("dynamic_shape_names") &&
            HasAttr("min_input_shape_vector") &&
            HasAttr("max_input_shape_vector") &&
            HasAttr("opt_input_shape_vector")) {
          std::vector<std::string> dynamic_shape_names;
          std::vector<std::vector<int>> min_input_shapes;
          std::vector<std::vector<int>> max_input_shapes;
          std::vector<std::vector<int>> opt_input_shapes;
          std::vector<int> dynamic_shape_lens;
          dynamic_shape_names =
              Attr<std::vector<std::string>>("dynamic_shape_names");
          std::vector<int> min_shapes =
              Attr<std::vector<int>>("min_input_shape_vector");
          std::vector<int> max_shapes =
              Attr<std::vector<int>>("max_input_shape_vector");
          std::vector<int> opt_shapes =
              Attr<std::vector<int>>("opt_input_shape_vector");
          dynamic_shape_lens = Attr<std::vector<int>>("dynamic_shape_lens");
          int idx = 0;
          for (size_t i = 0; i < dynamic_shape_lens.size(); ++i) {
            std::vector<int> tmp1, tmp2, tmp3;
            for (int j = 0; j < dynamic_shape_lens[i]; ++j) {
              tmp1.push_back(min_shapes[idx]);
              tmp2.push_back(max_shapes[idx]);
              tmp3.push_back(opt_shapes[idx++]);
            }
            min_input_shapes.emplace_back(tmp1);
            max_input_shapes.emplace_back(tmp2);
            opt_input_shapes.emplace_back(tmp3);
          }

          for (size_t i = 0; i < dynamic_shape_names.size(); ++i) {
            params.min_input_shape.insert(
                std::make_pair(dynamic_shape_names[i], min_input_shapes[i]));
            params.max_input_shape.insert(
                std::make_pair(dynamic_shape_names[i], max_input_shapes[i]));
            params.optim_input_shape.insert(
                std::make_pair(dynamic_shape_names[i], opt_input_shapes[i]));
          }
        }
      }

      trt_engine_ =
          inference::Singleton<inference::tensorrt::TRTEngineManager>::Global()
              .Create(engine_key_ + std::to_string(predictor_id_), params);

      if (use_static_engine_) {
        LOG(INFO) << "Load TRT Optimized Info from "
                  << inference::analysis::GetTrtEngineSerializedPath(
                         model_opt_cache_dir_, engine_key_);
        std::vector<std::string> all_weight_names;
        std::vector<phi::DenseTensor> all_weight_tensors;
        std::string trt_engine_serialized_data =
            inference::analysis::GetTrtEngineSerializedData(
                model_opt_cache_dir_, engine_key_);
        trt_engine_->Deserialize(trt_engine_serialized_data);
        if (!params.refit_params_path.empty()) {
          LOG(INFO) << "Begin to refit TRT Weights from path:"
                    << params.refit_params_path;
          // // paddle::AnalysisConfig config;
          // // LOG(INFO)<<"prog_file()"<<config.prog_file();
          // std::ifstream fin(params.refit_params_path, std::ios::in |
          // std::ios::binary); PADDLE_ENFORCE_EQ(
          //     static_cast<bool>(fin.is_open()),
          //     true,
          //     common::errors::NotFound("Cannot open refit_params_path file:
          //     %s", params.refit_params_path));
          // LOG(INFO)<<"读取完了";
          // std::string pb_content;
          // fin.seekg(0,std::ios::end);
          // pb_content.resize(fin.tellg());
          // fin.seekg(0,std::ios::beg);
          // fin.read(&(pb_content.at(0)),pb_content.size());
          // fin.close();
          // framework::proto::ProgramDesc proto;
          // LOG(INFO)<<"读取的文件大小"<<pb_content.size();
          // PADDLE_ENFORCE(proto.ParseFromString(pb_content),
          //                      common::errors::InvalidArgument(
          //                          "Failed to parse ProgramDesc from
          //                          refit_params_path"));
          // std::shared_ptr<framework::ProgramDesc> inference_program_ =
          //           std::make_unique<framework::ProgramDesc>(proto);
          // LOG(INFO) << "成功解析 ProgramDesc";

          // const auto &global_block = inference_program_->MutableBlock(0);
          // LOG(INFO) << "开始遍历全局块中的变量";
          // std::vector<std::string> all_weight_names;
          // std::vector<phi::DenseTensor> all_weight_tensors;
          // for(auto *var:global_block->AllVars())
          // {
          //   if(IsPersistable(var))
          //   {
          //     LOG(INFO)<<"var.name"<<var->Name();
          //   }
          // }
        }

      } else {
        // This branch mainly used to ut.
        PrepareTRTEngine(scope, trt_engine_);
      }
    }
    PADDLE_ENFORCE_NOT_NULL(
        trt_engine_,
        common::errors::Fatal(
            "The pointer to tensorrt engine should not be null."));
    return trt_engine_;
  }
};

}  // namespace operators
}  // namespace paddle

#endif  // PADDLE_WITH_CUDA
