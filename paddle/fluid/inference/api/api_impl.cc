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

#include "paddle/fluid/inference/api/api_impl.h"

#include <glog/logging.h>

#include <memory>
#include <sstream>
#include <string>

#include "paddle/fluid/framework/feed_fetch_method.h"
#include "paddle/fluid/inference/api/helper.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/platform/cpu_helper.h"
#include "paddle/phi/core/platform/profiler.h"

PD_DEFINE_bool(profile, false, "Turn on profiler for fluid");  // NOLINT

namespace paddle {
namespace {
using paddle::inference::Timer;

template <class T>
std::string num2str(T a) {
  std::stringstream istr;
  istr << a;
  return istr.str();
}
}  // namespace

void NativePaddlePredictor::PrepareFeedFetch() {
  for (auto *op : inference_program_->Block(0).AllOps()) {
    if (op->Type() == "feed") {
      int idx = PADDLE_GET_CONST(int, op->GetAttr("col"));
      if (feeds_.size() <= static_cast<size_t>(idx)) {
        feeds_.resize(idx + 1);
      }
      feeds_[idx] = op;
      feed_names_[op->Output("Out")[0]] = idx;
    } else if (op->Type() == "fetch") {
      int idx = PADDLE_GET_CONST(int, op->GetAttr("col"));
      if (fetches_.size() <= static_cast<size_t>(idx)) {
        fetches_.resize(idx + 1);
      }
      fetches_[idx] = op;
    }
  }
}

bool NativePaddlePredictor::Init(
    std::shared_ptr<framework::Scope> parent_scope) {
  VLOG(3) << "Predictor::init()";
  if (FLAGS_profile) {
    LOG(WARNING) << "Profiler is activated, might affect the performance";
    LOG(INFO) << "You can turn off by set gflags '-profile false'";

    auto tracking_device = config_.use_gpu ? platform::ProfilerState::kAll
                                           : platform::ProfilerState::kCPU;
    platform::EnableProfiler(tracking_device);
  }

  // no matter with or without OneDNN
  paddle::platform::SetNumThreads(config_.cpu_math_library_num_threads());

  if (config_.use_gpu) {
    PADDLE_ENFORCE_EQ(config_.use_xpu,
                      false,
                      common::errors::InvalidArgument(
                          "Only one choice can be made between CPU and XPU."));
    place_ = phi::GPUPlace(config_.device);
  } else if (config_.use_xpu) {
    place_ = phi::XPUPlace(config_.device);
  } else {
    place_ = phi::CPUPlace();
  }
  if (parent_scope) {
    scope_ = parent_scope;
    sub_scope_ = &(parent_scope->NewScope());
    PADDLE_ENFORCE_NOT_NULL(sub_scope_,
                            common::errors::PreconditionNotMet(
                                "The sub_scope should not be nullptr."));
  } else {
    paddle::framework::InitMemoryMethod();
    paddle::framework::InitDevices();
    paddle::framework::InitDefaultKernelSignatureMap();
    scope_ = std::make_unique<paddle::framework::Scope>();
  }

  executor_ = std::make_unique<paddle::framework::Executor>(place_);

  // Initialize the inference program
  if (!config_.model_dir.empty()) {  // NOLINT
    // Parameters are saved in separate files sited in
    // the specified `dirname`.
    inference_program_ = paddle::inference::Load(
        executor_.get(), scope_.get(), config_.model_dir);
  } else if (!config_.prog_file.empty() && !config_.param_file.empty()) {
    // All parameters are saved in a single file.
    // The file names should be consistent with that used
    // in Python API `fluid.io.save_inference_model`.
    inference_program_ = paddle::inference::Load(
        executor_.get(), scope_.get(), config_.prog_file, config_.param_file);
  } else {
    LOG(ERROR) << "fail to load inference model from " << config_.model_dir;
    return false;
  }

  ctx_ = executor_->Prepare(*inference_program_, 0);
  executor_->CreateVariables(
      *inference_program_, sub_scope_ ? sub_scope_ : scope_.get(), 0);

  // Get the feed_target_names and fetch_target_names
  PrepareFeedFetch();
  return true;
}

NativePaddlePredictor::~NativePaddlePredictor() {
  if (FLAGS_profile) {
    platform::DisableProfiler(platform::EventSortingKey::kTotal,
                              "./profile.log");
  }
  if (sub_scope_) {
    scope_->DeleteScope(sub_scope_);
  }
}

bool NativePaddlePredictor::Run(const std::vector<PaddleTensor> &inputs,
                                std::vector<PaddleTensor> *output_data,
                                int batch_size) {
#ifndef PADDLE_ON_INFERENCE
  LOG_FIRST_N(WARNING, 5) << "The NaiveExecutor can not work properly if the "
                             "cmake flag ON_INFER is not set.";
  LOG_FIRST_N(WARNING, 5) << "Unlike the training phase, all the scopes and "
                             "variables will be reused to save the allocation "
                             "overhead.";
  LOG_FIRST_N(WARNING, 5) << "Please re-compile the inference library by "
                             "setting the cmake flag ON_INFER=ON if you are "
                             "running Paddle Inference";
#endif  // PADDLE_ON_INFERENCE
  if (UNLIKELY(config_.cpu_math_library_num_threads() > 1)) {
    paddle::platform::SetNumThreads(config_.cpu_math_library_num_threads());
  }
  VLOG(3) << "Predictor::predict";
  Timer timer;
  timer.tic();
  // set feed variable
  framework::Scope *scope = sub_scope_ != nullptr ? sub_scope_ : scope_.get();
  if (!SetFeed(inputs, scope)) {
    LOG(ERROR) << "fail to set feed";
    return false;
  }
  // Run the inference program
  // if share variables, we need not create variables
  VLOG(4) << "Run prepared context";
  executor_->RunPreparedContext(ctx_.get(),
                                scope,
                                false, /* don't create local scope each time*/
                                false /* don't create variable each time */);
  VLOG(4) << "Finish prepared context";
  // get fetch variable
  if (!GetFetch(output_data, scope)) {
    LOG(ERROR) << "fail to get fetches";
    return false;
  }
  VLOG(3) << "predict cost: " << timer.toc() << "ms";

  // For some other vector like containers not cleaned after each batch.
  tensor_array_batch_cleaner_.CollectNoTensorVars(scope_.get());
  tensor_array_batch_cleaner_.ResetNoTensorVars();
  return true;
}

std::unique_ptr<PaddlePredictor> NativePaddlePredictor::Clone(void *stream) {
  std::lock_guard<std::mutex> lk(clone_mutex_);
  VLOG(3) << "Predictor::clone";
  std::unique_ptr<PaddlePredictor> cls(new NativePaddlePredictor(config_));
  // Hot fix the bug that result diff in multi-thread.
  // TODO(Superjomn) re-implement a real clone here.
  PADDLE_ENFORCE_NOT_NULL(
      dynamic_cast<NativePaddlePredictor *>(cls.get()),
      common::errors::PreconditionNotMet(
          "Dynamic_cast from PaddlePredictor to NativePaddlePredictor failed"));
  if (!dynamic_cast<NativePaddlePredictor *>(cls.get())->Init(nullptr)) {
    LOG(ERROR) << "fail to call Init";
    return nullptr;
  }
  return cls;
}

bool NativePaddlePredictor::SetFeed(const std::vector<PaddleTensor> &inputs,
                                    framework::Scope *scope) {
  VLOG(3) << "Predictor::set_feed";
  if (inputs.size() != feeds_.size()) {
    LOG(ERROR) << "wrong feed input size, need " << feeds_.size() << " but get "
               << inputs.size();
    return false;
  }

  // Cache the inputs memory for better concurrency performance.
  feed_tensors_.resize(inputs.size());

  for (size_t i = 0; i < inputs.size(); ++i) {
    auto &input = feed_tensors_[i];
    phi::DDim ddim = common::make_ddim(inputs[i].shape);
    void *input_ptr = nullptr;
    if (inputs[i].dtype == PaddleDType::INT64) {
      input_ptr = input.mutable_data<int64_t>(ddim, place_);
    } else if (inputs[i].dtype == PaddleDType::FLOAT32) {
      input_ptr = input.mutable_data<float>(ddim, place_);
    } else if (inputs[i].dtype == PaddleDType::INT32) {
      input_ptr = input.mutable_data<int32_t>(ddim, place_);
    } else if (inputs[i].dtype == PaddleDType::BFLOAT16) {
      input_ptr = input.mutable_data<bfloat16>(ddim, place_);
    } else {
      LOG(ERROR) << "unsupported feed type " << inputs[i].dtype;
      return false;
    }

    PADDLE_ENFORCE_NOT_NULL(input_ptr,
                            common::errors::InvalidArgument(
                                "The input_ptr should not be nullptr."));
    PADDLE_ENFORCE_NOT_NULL(
        inputs[i].data.data(),
        common::errors::InvalidArgument(
            "The data of input tensor should not be null."));
    PADDLE_ENFORCE_EQ(
        inputs[i].data.length(),
        input.numel() * phi::SizeOf(input.dtype()),
        common::errors::InvalidArgument(
            "The data contained in the input PaddleTensor had wrong length."));

    if (phi::is_cpu_place(place_)) {
      // TODO(panyx0718): Init DenseTensor from existing memcpy to save a copy.
      std::memcpy(static_cast<void *>(input_ptr),
                  inputs[i].data.data(),
                  inputs[i].data.length());
    } else if (phi::is_gpu_place(place_)) {
      PADDLE_ENFORCE_EQ(
          phi::is_xpu_place(place_),
          false,
          common::errors::InvalidArgument(
              "Only one choice can be made between CPU and XPU."));
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::DeviceContextPool &pool = phi::DeviceContextPool::Instance();
      auto *dev_ctx = static_cast<const phi::GPUContext *>(pool.Get(place_));
      auto dst_gpu_place = place_;
      memory::Copy(dst_gpu_place,
                   static_cast<void *>(input_ptr),
                   phi::CPUPlace(),
                   inputs[i].data.data(),
                   inputs[i].data.length(),
                   dev_ctx->stream());
#else
      PADDLE_THROW(common::errors::Unavailable(
          "Not compile with CUDA, should not reach here."));
#endif
    } else if (phi::is_xpu_place(place_)) {
#ifdef PADDLE_WITH_XPU
      auto dst_xpu_place = place_;
      memory::Copy(dst_xpu_place,
                   static_cast<void *>(input_ptr),
                   phi::CPUPlace(),
                   inputs[i].data.data(),
                   inputs[i].data.length());
#else
      PADDLE_THROW(common::errors::Unavailable(
          "Not compile with XPU, should not reach here."));
#endif
    }

    // TODO(Superjomn) Low performance, need optimization for heavy LoD copy.
    phi::LegacyLoD lod;
    for (auto &level : inputs[i].lod) {
      lod.emplace_back(level);
    }
    input.set_lod(lod);
    int idx = -1;
    if (config_.specify_input_name) {  // NOLINT
      idx = static_cast<int>(feed_names_[inputs[i].name]);
    } else {
      idx = PADDLE_GET_CONST(int, feeds_[i]->GetAttr("col"));
    }
    framework::SetFeedVariable(scope, input, "feed", idx);
  }
  return true;
}
template <typename T>
void NativePaddlePredictor::GetFetchOne(const phi::DenseTensor &fetch,
                                        PaddleTensor *output) {
  // set shape.
  auto shape = common::vectorize(fetch.dims());
  output->shape.assign(shape.begin(), shape.end());
  // set data.
  const T *data = fetch.data<T>();
  int num_elems = inference::VecReduceToInt(shape);
  output->data.Resize(num_elems * sizeof(T));
  // The fetched tensor output by fetch op, should always in CPU memory, so just
  // copy.
  memcpy(output->data.data(), data, num_elems * sizeof(T));
  // set lod
  output->lod.clear();
  for (auto &level : fetch.lod()) {
    output->lod.emplace_back(level.begin(), level.end());
  }
}

bool NativePaddlePredictor::GetFetch(std::vector<PaddleTensor> *outputs,
                                     framework::Scope *scope) {
  VLOG(3) << "Predictor::get_fetch";
  outputs->resize(fetches_.size());
  for (size_t i = 0; i < fetches_.size(); ++i) {
    int idx = PADDLE_GET_CONST(int, fetches_[i]->GetAttr("col"));
    PADDLE_ENFORCE_EQ(
        static_cast<size_t>(idx),
        i,
        common::errors::InvalidArgument(
            "Fetch op's col attr(%d) should be equal to the index(%d)",
            idx,
            i));
    framework::FetchType &fetch_var =
        framework::GetFetchVariable(*scope, "fetch", idx);
    auto fetch = PADDLE_GET_CONST(phi::DenseTensor, fetch_var);
    auto type = framework::TransToProtoVarType(fetch.dtype());
    auto output = &(outputs->at(i));
    output->name = fetches_[idx]->Input("X")[0];
    if (type == framework::DataTypeTrait<float>::DataType()) {
      GetFetchOne<float>(fetch, output);
      output->dtype = PaddleDType::FLOAT32;
    } else if (type == framework::DataTypeTrait<int64_t>::DataType()) {
      GetFetchOne<int64_t>(fetch, output);
      output->dtype = PaddleDType::INT64;
    } else if (type == framework::DataTypeTrait<int32_t>::DataType()) {
      GetFetchOne<int32_t>(fetch, output);
      output->dtype = PaddleDType::INT32;
    } else {
      LOG(ERROR) << "unknown type, only support float32, int64 and int32 now.";
    }
  }
  return true;
}

template <>
std::unique_ptr<PaddlePredictor>
CreatePaddlePredictor<NativeConfig, PaddleEngineKind::kNative>(
    const NativeConfig &config) {
  // TODO(NHZlX): Should add the link to the doc of
  // paddle_infer::CreatePredictor<paddle_infer::Config>
  VLOG(3) << "create NativePaddlePredictor";
  if (config.use_gpu) {
    // 1. GPU memory
    PADDLE_ENFORCE_GE(config.fraction_of_gpu_memory,
                      0.f,
                      common::errors::InvalidArgument(
                          "fraction_of_gpu_memory in the config should be set "
                          "to range (0., 1.]"));
    PADDLE_ENFORCE_GE(config.device,
                      0,
                      common::errors::PreconditionNotMet(
                          "Invalid device id %d, the device id should be "
                          "greater than or equal to 0.",
                          config.device));
    std::vector<std::string> flags;
    if (config.fraction_of_gpu_memory >= 0.0f ||
        config.fraction_of_gpu_memory <= 0.95f) {
      std::string flag = "--fraction_of_gpu_memory_to_use=" +
                         num2str<float>(config.fraction_of_gpu_memory);
      flags.push_back(flag);
      VLOG(3) << "set flag: " << flag;
      framework::InitGflags(flags);
    }
  }

  std::unique_ptr<PaddlePredictor> predictor(new NativePaddlePredictor(config));
  PADDLE_ENFORCE_NOT_NULL(
      dynamic_cast<NativePaddlePredictor *>(predictor.get()),
      common::errors::PreconditionNotMet(
          "Dynamic_cast from PaddlePredictor to NativePaddlePredictor failed"));
  if (!dynamic_cast<NativePaddlePredictor *>(predictor.get())->Init(nullptr)) {
    return nullptr;
  }
  return predictor;
}

template <>
std::unique_ptr<PaddlePredictor> CreatePaddlePredictor<NativeConfig>(
    const NativeConfig &config) {
  LOG(WARNING) << "Deprecated. Please use CreatePredictor instead.";
  return CreatePaddlePredictor<NativeConfig, PaddleEngineKind::kNative>(config);
}

}  // namespace paddle
