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

#include "paddle/fluid/platform/device/ipu/ipu_executor.h"

#include <chrono>
#include <popart/devicemanager.hpp>
#include <popdist/popdist_poplar.hpp>

#include "paddle/fluid/framework/data_type_transform.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/device/ipu/ipu_compiler.h"
#include "paddle/fluid/platform/device/ipu/ipu_names.h"
#include "paddle/fluid/platform/device/ipu/ipu_strategy.h"

namespace paddle {
namespace platform {
namespace ipu {

namespace {

model_runtime::AnchorCallbackPredicate PredFilterMain(
    const model_runtime::Session *session) {
  // Create predicates for binding Anchors from Main programs only
  return model_runtime::predicate_factory::predProgramFlowMain(
      session->model()->metadata.programFlow());
}

// Get paddle prefix and popart postfix of weight states
// Format: {popart_postfix, paddle_prefix}
std::vector<std::pair<std::string, std::string>> GetOptPrePostfix(
    const std::string &opt_type) {
  std::vector<std::pair<std::string, std::string>> pre_post_fix;
  // Weight self
  pre_post_fix.push_back(std::make_pair("", ""));

  // Weight states
  // TODO(alleng) support pair("Accl1___", "_moment1_{id!=0}")
  if (opt_type == "adam" || opt_type == "lamb" || opt_type == "adamw") {
    pre_post_fix.push_back(std::make_pair("Accl1___", "_moment1_0"));
    pre_post_fix.push_back(std::make_pair("Accl2___", "_moment2_0"));
    pre_post_fix.push_back(std::make_pair("Step___", "_beta1_pow_acc_0"));
  } else if (opt_type == "momentum") {
    pre_post_fix.push_back(std::make_pair("Accl___", "_velocity_0"));
  } else if (opt_type == "adamax") {
    pre_post_fix.push_back(std::make_pair("Accl1___", "_moment_0"));
    pre_post_fix.push_back(std::make_pair("Accl2___", "_inf_norm__0"));
    pre_post_fix.push_back(std::make_pair("Step___", "_beta1_pow_acc_0"));
  } else if (opt_type == "adagrad") {
    pre_post_fix.push_back(std::make_pair("Accl1___", "_moment_0"));
  } else if (opt_type == "adadelta") {
    pre_post_fix.push_back(std::make_pair("Accl1___", "__avg_squared_grad_0"));
    pre_post_fix.push_back(
        std::make_pair("Accl2___", "__avg_squared_update_0"));
  } else if (opt_type == "rmsprop") {
    pre_post_fix.push_back(std::make_pair("Accl1___", "_mean_square_0"));
    pre_post_fix.push_back(std::make_pair("Accl2___", "_mean_grad_0"));
    pre_post_fix.push_back(std::make_pair("Accl3___", "_momentum__0"));
  }
  return pre_post_fix;
}

class PdIArray final : public popart::IArray {
 public:
  explicit PdIArray(const Tensor *tensor) {
    tensor_.ShareDataWith(*tensor);
    for (int i = 0; i < tensor->dims().size(); ++i) {
      shape_.push_back(tensor->dims().at(i));
    }
  }

 public:
  void *data() { return tensor_.data(); }
  popart::DataType dataType() const {
    return PhiDType2PopartDType(tensor_.dtype());
  }
  std::size_t rank() const { return tensor_.dims().size(); }
  int64_t dim(size_t index) const { return tensor_.dims().at(index); }
  std::size_t nelms() const {
    return std::accumulate(shape_.begin(),
                           shape_.end(),
                           static_cast<int64_t>(1),
                           std::multiplies<int64_t>());
  }
  const popart::Shape shape() const { return shape_; }

 private:
  Tensor tensor_;
  std::vector<int64_t> shape_;
};

}  // namespace

Executor::~Executor() { Reset(); }

void Executor::Prepare(const std::string &proto) {
  VLOG(10) << "enter Executor::Prepare";
  compile_only_ = GetBoolEnv("IPU_COMPILE_ONLY");

  AcquireDevice();
  executor_resources_ = std::make_unique<ExecutorResources>();

  auto art = popart::AnchorReturnType("All");
  std::map<popart::TensorId, popart::AnchorReturnType> anchor_ids;
  for (const auto &id : compiler_resources_->outputs) {
    anchor_ids.emplace(id, art);
  }
  auto dataFlow = popart::DataFlow(ipu_strategy_->batches_per_step, anchor_ids);

  if (ipu_strategy_->is_training) {
    VLOG(10) << "Creating TrainingSession from Onnx Model...";
    auto optimizer = compiler_resources_->NewOptimizer();
    session_ = popart::TrainingSession::createFromOnnxModel(
        proto,
        dataFlow,
        compiler_resources_->loss_var,
        *optimizer,
        device_,
        popart::InputShapeInfo(),
        ipu_strategy_->popart_options,
        ipu_strategy_->popart_patterns);
  } else {
    VLOG(10) << "Creating InferenceSession from Onnx Model...";
    session_ = popart::InferenceSession::createFromOnnxModel(
        proto,
        dataFlow,
        device_,
        popart::InputShapeInfo(),
        ipu_strategy_->popart_options,
        ipu_strategy_->popart_patterns);
  }
  VLOG(10) << "Creating session from Onnx Model...done";

  if (compile_only_) {
    LOG(INFO)
        << "Save the offline cache as offline_cache.popart in current path.";
    VLOG(10) << "Compile only...";
    session_->compileAndExport("./offline_cache.popart");
    VLOG(10) << "Compile only...done";
    return;
  } else {
    VLOG(10) << "Preparing session device...";
    session_->prepareDevice();
    VLOG(10) << "Preparing session device...done";
  }

  SetWeightsIO();

  VLOG(10) << "Copy weights from paddle to popart...";
  WeightsFromPaddle();
  VLOG(10) << "Copy weights from paddle to popart...done";

  if (ipu_strategy_->random_seed != std::numeric_limits<std::uint64_t>::max()) {
    VLOG(10) << "Setting random seed to: " << ipu_strategy_->random_seed;
    session_->setRandomSeed(ipu_strategy_->random_seed);
  }
  enable_model_runtime_executor_ = ipu_strategy_->enable_model_runtime_executor;
  if (enable_model_runtime_executor_) {
    PreparePopefSession();
  }
}

void Executor::Run(const std::vector<const Tensor *> &inputs,
                   const std::vector<Tensor *> &outputs,
                   const framework::ExecutionContext &ctx) {
  if (compile_only_) {
    LOG(INFO) << "If IPU_COMPILE_ONLY=True, skip exe.run";
    return;
  }

  VLOG(10) << "enter Executor::Run";
  // inputs
  std::map<popart::TensorId, popart::IArray &> popart_inputs;
  std::map<popart::TensorId, PdIArray> input_wrappers;
  for (size_t i = 0; i < inputs.size(); i++) {
    auto tensor_id = compiler_resources_->inputs[i];
    input_wrappers.emplace(tensor_id, PdIArray(inputs[i]));
    popart_inputs.emplace(tensor_id, input_wrappers.at(tensor_id));
  }
  // anchors
  std::map<popart::TensorId, popart::IArray &> popart_anchors;
  std::map<popart::TensorId, PdIArray> anchor_wrappers;
  for (size_t i = 0; i < outputs.size(); i++) {
    auto tensor_id = compiler_resources_->outputs[i];
    // get dims & dtype from session
    auto fetch_info = session_->getInfo(tensor_id);
    auto output_shape = fetch_info.shape();
    if (ipu_strategy_->batches_per_step > 1) {
      output_shape.insert(output_shape.begin(),
                          ipu_strategy_->batches_per_step);
    }
    if (ipu_strategy_->popart_options.enableGradientAccumulation) {
      output_shape.insert(output_shape.begin(),
                          ipu_strategy_->popart_options.accumulationFactor);
    }
    if (ipu_strategy_->popart_options.enableReplicatedGraphs) {
      output_shape.insert(output_shape.begin(),
                          ipu_strategy_->popart_options.replicatedGraphCount);
    }

    auto *tensor = outputs[i];
    tensor->Resize(common::make_ddim(output_shape));
    auto fetch_dtype = fetch_info.dataType();
    auto paddle_type = PopartDType2VarType(fetch_dtype);
    tensor->mutable_data(ctx.GetPlace(), phi::TransToPhiDataType(paddle_type));
    anchor_wrappers.emplace(tensor_id, PdIArray(tensor));
    popart_anchors.emplace(tensor_id, anchor_wrappers.at(tensor_id));
  }
  VLOG(10) << "Prepared inputs/anchors";

  if (ipu_strategy_->is_training && compiler_resources_->with_lr_sched) {
    popart::Optimizer *optimizer;
    if (ipu_strategy_->runtime_options.enable_eval) {
      VLOG(10) << "Switch optimizer to eval mode";
      optimizer = compiler_resources_->eval_optimizer.get();
    } else {
      VLOG(10) << "Update learning_rate";
      float new_lr;
      if (ipu_strategy_->is_dynamic) {
        new_lr = ipu_strategy_->lr;
      } else {
        new_lr =
            GetSingleVarFromScope<float>(scope_, compiler_resources_->lr_var);
      }
      VLOG(10) << "New Lr: " << new_lr;
      optimizer = compiler_resources_->UpdateOptimizer(new_lr);
    }
    auto *session = dynamic_cast<popart::TrainingSession *>(session_.get());
    session->updateOptimizerFromHost(optimizer);
  }

  popart::StepIO stepio(popart_inputs, popart_anchors);
  VLOG(10) << "Running...";
  session_->run(stepio);
  VLOG(10) << "Running...done";
}

void Executor::PreparePopefSession() {
  VLOG(10) << "enter Executor::PreparePopefSession";
  if (popef_session_) {
    VLOG(10) << "popef: previous popef model is not released, reset resources.";
    ResetPopef();
  }
  auto popef_model = PopartSessionToPopefModel(session_.get());

  auto num_buffers = ipu_strategy_->num_buffers;

  // convert timeout_ms to timeout_ns
  const std::chrono::nanoseconds timeout_ns(
      int64_t(ipu_strategy_->timeout_ms * 1000000));

  // prepare popef session
  model_runtime::SessionConfig config;
  config.policy = model_runtime::LaunchPolicy::Immediate;

  popef_session_ =
      std::make_unique<model_runtime::Session>(popef_model, config);

  // prepare queue_manager
  auto timeout_cb = [this](model_runtime::InputRingBuffer *buffer) {
    VLOG(10) << "ModelRuntmie timeout callback is called.";
    std::unique_lock lock(this->queue_mutex_);
    if (buffer->readAvailable()) {
      return;
    }
    this->queue_manager_->flushAll();
  };

  queue_manager_ =
      popef_session_->createQueueManager(num_buffers,
                                         timeout_cb,
                                         timeout_ns,
                                         PredFilterMain(popef_session_.get()),
                                         PredFilterMain(popef_session_.get()));

  // prepare program
  popef_session_->runLoadPrograms();

  main_program_ = std::thread([&]() {
    while (!stop_.load()) {
      VLOG(13) << "popef: Run main program";
      popef_session_->runMainPrograms();
    }
  });

  // Detach device from popart session
  Detach();
}

void Executor::RunPopef(const std::vector<const Tensor *> &inputs,
                        const std::vector<Tensor *> &outputs,
                        const framework::ExecutionContext &ctx) {
  VLOG(10) << "enter Executor::RunPopef";

  auto input_names = ctx.InputNames("FeedList");
  auto output_names = ctx.OutputNames("FetchList");

  int batch_size = 0;
  bool auto_batch = (ipu_strategy_->timeout_ms != 0);

  auto tensor_check = [&](const Tensor *tensor,
                          const popef::TensorInfo &info,
                          int *batch_size,
                          Tensor *cast_tensor) {
    // check dtype
    auto popef_phi_dtype = PopefDtype2PhiDtype(info.dataType());
    bool casted = false;

    if (popef_phi_dtype != tensor->dtype()) {
      // popart may do some implicit conversion, int64->int32 for example, cast
      // is needed in some case.
      VLOG(10) << "Cast paddle input type " << tensor->dtype() << " to "
               << popef_phi_dtype;
      framework::TransDataType(
          *tensor, PopefDType2VarType(info.dataType()), cast_tensor);
      casted = true;
    }

    // check size
    auto popef_input_shape = info.shape();
    if (popef_input_shape.size() != tensor->dims().size()) {
      PADDLE_THROW(
          errors::Fatal("Incompatible size between paddle and popef."));
    }

    for (int i = 1; i < popef_input_shape.size(); ++i) {
      PADDLE_ENFORCE_EQ(
          popef_input_shape[i],
          tensor->dims().at(i),
          errors::InvalidArgument("Invalid tensor size at dim %s. "
                                  "popef expecting %s but received %s ",
                                  i,
                                  popef_input_shape[i],
                                  tensor->dims().at(i)));
    }

    // check batch_size
    if (!auto_batch) {
      // disable auto batching
      PADDLE_ENFORCE_EQ(
          popef_input_shape[0],
          tensor->dims().at(0),
          errors::InvalidArgument(
              "Batch size doesn't equal between paddle and popef."));
    } else {
      // enable auto batching
      bool is_single_batch = ipu_strategy_->micro_batch_size == 1;
      if (*batch_size == 0) {
        // retrieve batch_size
        *batch_size = is_single_batch ? 1 : tensor->dims().at(0);
      } else if (!is_single_batch) {
        // input/output should have batch info when enable auto batch.
        PADDLE_ENFORCE_EQ(*batch_size,
                          tensor->dims().at(0),
                          errors::InvalidArgument(
                              "batch size should be equal for each tensor"));
      }
    }
    return casted;
  };

  const auto &session_inputs = popef_session_->getUserInputAnchors();
  std::vector<Tensor> cast_tensor(inputs.size());
  const auto &session_outputs = popef_session_->getUserOutputAnchors();

  // ModelRuntime::Queue is not thread safety.
  std::unique_lock lock(queue_mutex_);
  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto &popef_input_name =
        compiler_resources_->tensors.at(input_names[i]);
    auto &elem_queue = queue_manager_->inputQueue(popef_input_name);
    const auto &info = elem_queue.tensorInfo();
    VLOG(10) << "popef: handle popef input: " << popef_input_name
             << " mapped with paddle " << input_names[i];

    bool casted = tensor_check(inputs[i], info, &batch_size, &(cast_tensor[i]));

    const void *data = casted ? cast_tensor[i].data() : inputs[i]->data();
    const auto size =
        casted ? cast_tensor[i].memory_size() : inputs[i]->memory_size();

    elem_queue.enqueue(data, size, [popef_input_name]() {
      VLOG(10) << "popef: enqueued data for input: " << popef_input_name;
    });
  }

  std::vector<std::future<void>> finish_indicators;
  finish_indicators.reserve(session_outputs.size());

  for (size_t i = 0; i < session_outputs.size(); ++i) {
    const auto &popef_output_name =
        compiler_resources_->tensors.at(output_names[i]);
    auto &out_queue = queue_manager_->outputQueue(popef_output_name);
    const auto &info = out_queue.tensorInfo();
    VLOG(10) << "popef: handle popef output: " << popef_output_name
             << " mapped with paddle " << output_names[i];

    auto popef_dtype = info.dataType();
    auto paddle_dtype = PopefDType2VarType(popef_dtype);
    auto output_shape = info.shape();
    if (auto_batch) {
      if (output_shape[0] == ipu_strategy_->micro_batch_size) {
        output_shape[0] = batch_size;
      } else {
        // shape of output must have batch info when when auto batch enabled
        PADDLE_THROW(common::errors::Unimplemented(
            "Auto batch doesn't support the tensor with no batch info. "
            "Expected batch size in output tensor: %d should equal to "
            "micro batch size: %d. Please make sure batch size is set "
            "correctly in both IPU program compiling and IpuStrategy.",
            output_shape[0],
            ipu_strategy_->micro_batch_size));
      }
    }

    auto *tensor = outputs[i];
    // resize output size to make data_ptr valid.
    tensor->Resize(common::make_ddim(output_shape));
    tensor->mutable_data(ctx.GetPlace(), phi::TransToPhiDataType(paddle_dtype));

    const auto size = tensor->memory_size();

    auto promise = std::make_shared<std::promise<void>>();
    finish_indicators.emplace_back(promise->get_future());
    out_queue.enqueue(tensor->data(), size, [popef_output_name, promise]() {
      VLOG(10) << "popef: received output: " << popef_output_name;
      promise->set_value();
    });
  }
  lock.unlock();

  // Synchronous waiting outputs. Asynchronous execution is not supported since
  // python api calling is synchronous and output data is copied outside.
  for (const auto &indicator : finish_indicators) {
    indicator.wait();
  }
}

void Executor::WeightsToHost() {
  if (ipu_strategy_->is_training && session_) {
    WeightsToPaddle();
  } else {
    LOG(WARNING) << "For a non-training graph, cannot sync weights from IPU.";
  }
}

void Executor::AcquireDevice() {
  VLOG(10) << "enter Executor::AcquireDevice";
  if (device_) {
    Detach();
    device_.reset();
  }

  bool use_ipu_model = GetBoolEnv("POPLAR_IPUMODEL");
  bool enable_distribution = ipu_strategy_->enable_distribution;
  if (use_ipu_model) {
    VLOG(10) << "Create IPU model device...";
    std::map<std::string, std::string> deviceOpts{
        {
            "numIPUs",
            std::to_string(ipu_strategy_->num_ipus),
        },
        {"tilesPerIPU", std::to_string(ipu_strategy_->tiles_per_ipu)},
        {"ipuVersion", "ipu2"},
    };
    device_ = popart::DeviceManager::createDeviceManager().createIpuModelDevice(
        deviceOpts);
    VLOG(10) << "Create IPU model device...done";
  } else if (compile_only_) {
    VLOG(10) << "Create offline device...";
    std::map<std::string, std::string> deviceOpts{
        {
            "numIPUs",
            std::to_string(ipu_strategy_->num_ipus),
        },
        {"tilesPerIPU", std::to_string(ipu_strategy_->tiles_per_ipu)},
        {"ipuVersion", "ipu2"},
    };
    device_ =
        popart::DeviceManager::createDeviceManager().createOfflineIPUDevice(
            deviceOpts);
    VLOG(10) << "Create offline device...done";
  } else if (enable_distribution) {
    VLOG(10) << "Create distribution device...";
    auto ipus_per_replica = ipu_strategy_->num_ipus /
                            ipu_strategy_->popart_options.replicatedGraphCount;
    auto device_id = popdist::getDeviceId(ipus_per_replica);
    device_ = popart::DeviceManager::createDeviceManager().acquireDeviceById(
        device_id);
    PADDLE_ENFORCE_NOT_NULL(
        device_,
        errors::Unavailable("Can't attach IPU in distribution, ipu_num = %d.",
                            RequestIpus(ipu_strategy_->num_ipus)));
    VLOG(10) << "Create distribution device...done";
  } else {
    VLOG(10) << "Create IPU device...";
    device_ =
        popart::DeviceManager::createDeviceManager().acquireAvailableDevice(
            RequestIpus(ipu_strategy_->num_ipus));
    PADDLE_ENFORCE_NOT_NULL(
        device_,
        errors::Unavailable("Can't attach IPU, ipu_num = %d.",
                            RequestIpus(ipu_strategy_->num_ipus)));
    VLOG(10) << "Create IPU device...done";
  }
  VLOG(10) << "leave Executor::AcquireDevice";
}

void Executor::Detach() {
  if (device_ && device_->isAttached()) {
    VLOG(10) << "trying to detach IPU";
    device_->detach();
    VLOG(10) << " detached IPU";
  }
}

void Executor::Reset() {
  Detach();
  session_.reset();
  executor_resources_.reset();
  if (enable_model_runtime_executor_) {
    ResetPopef();
  }
}

void Executor::ResetPopef() {
  VLOG(10) << "Reset popef resources.";
  stop_.store(true);
  if (queue_manager_) {
    queue_manager_->disconnectAll();
  }
  if (main_program_.joinable()) {
    const auto future = std::async(std::launch::async,
                                   [this]() { this->main_program_.join(); });
    if (future.wait_for(std::chrono::seconds(10)) ==
        std::future_status::timeout) {
      popef_session_->stop();
      VLOG(10) << "popef: failed to wait for main program. Force stop popef "
                  "session.";
    }
  }
  popef_session_.reset();

  // reset stop back to false in case executor is reused.
  stop_.store(false);
  queue_manager_ = nullptr;
}

void Executor::SetWeightsIO() {
  auto opt_type = compiler_resources_->optimizer_type;
  VLOG(10) << "SetWeightsIO for " << opt_type;
  auto pre_post_fix = GetOptPrePostfix(opt_type);
  for (const auto &weight_pd : compiler_resources_->weights) {
    for (const auto &pair : pre_post_fix) {
      // pair.first : popart prefix, pair.second : paddle postfix
      auto weight_pop = compiler_resources_->tensors[weight_pd];
      auto popart_var = pair.first + weight_pop;
      auto paddle_var = weight_pd + pair.second;

      if (scope_->FindVar(paddle_var) == nullptr) {
        continue;
      }
      if (!session_->hasInfo(popart_var)) {
        continue;
      }

      VLOG(10) << "Connect paddle weight: " << paddle_var
               << " with popart weight: " << popart_var;
      auto var = scope_->GetVar(paddle_var);
      auto data_ptr = var->GetMutable<phi::DenseTensor>()->data();
      auto tensor_info = session_->getInfo(popart_var);
      executor_resources_->weights_io.insert(popart_var,
                                             {data_ptr, tensor_info});
      executor_resources_->weights_and_opt_state.emplace_back(
          std::make_pair(popart_var, paddle_var));
    }
  }
}

// align_to_popart: align dtype to popart if true, else to paddle
void Executor::ConvertWeights(bool align_to_popart) {
  for (auto weight_pair : executor_resources_->weights_and_opt_state) {
    auto paddle_var = scope_->GetVar(weight_pair.second);
    auto paddle_var_dtype = PhiDType2PopartDType(
        paddle_var->GetMutable<phi::DenseTensor>()->dtype());

    PADDLE_ENFORCE_EQ((paddle_var_dtype == popart::DataType::FLOAT ||
                       paddle_var_dtype == popart::DataType::FLOAT16),
                      true,
                      errors::InvalidArgument(
                          "Currently, we only support FLOAT16 and FLOAT with "
                          "Paddle, but received type is %s.",
                          paddle_var_dtype));

    popart::TensorInfo info = session_->getInfo(weight_pair.first);
    auto popart_var_dtype = info.dataType();
    PADDLE_ENFORCE_EQ((popart_var_dtype == popart::DataType::FLOAT ||
                       popart_var_dtype == popart::DataType::FLOAT16),
                      true,
                      errors::InvalidArgument(
                          "Currently, we only support FLOAT16 and FLOAT with "
                          "popart, but received type is %s.",
                          popart_var_dtype));

    if (paddle_var_dtype == popart_var_dtype) {
      VLOG(10) << weight_pair.first << " and " << weight_pair.second
               << " have the same dtype : " << popart_var_dtype;
      continue;
    } else if (paddle_var_dtype == popart::DataType::FLOAT) {
      VLOG(10) << weight_pair.first << " and " << weight_pair.second
               << " have different dtype : " << popart_var_dtype;
      auto *data_ptr =
          paddle_var->GetMutable<phi::DenseTensor>()->data<float>();

      auto num_elem = info.nelms();
      if (align_to_popart) {
        std::vector<uint16_t> fp16_data;
        std::transform(data_ptr,
                       data_ptr + num_elem,
                       std::back_inserter(fp16_data),
                       [&](float elem) { return popart::floatToHalf(elem); });
        memcpy(reinterpret_cast<void *>(data_ptr),
               fp16_data.data(),
               num_elem * sizeof(float16));
      } else {
        std::vector<float> fp32_data;
        auto fp16_data_ptr = reinterpret_cast<uint16_t *>(data_ptr);
        std::transform(
            fp16_data_ptr,
            fp16_data_ptr + num_elem,
            std::back_inserter(fp32_data),
            [&](uint16_t elem) { return popart::halfToFloat(elem); });
        memcpy(reinterpret_cast<void *>(data_ptr),
               fp32_data.data(),
               num_elem * sizeof(float));
      }
    } else {
      PADDLE_THROW(
          errors::Unimplemented("Convert Paddle FLOAT16 to popart FLOAT"));
    }
  }
}

// |-----------------------------------------------------|
// | Paddle  | Popart  |             Method              |
// |-----------------------------------------------------|
// |  FLOAT  |  FLOAT  |         Paddle -> Popart        |
// |  FLOAT  | FLOAT16 | floatToHalf -> Paddle -> Popart |
// | FLOAT16 |  FLOAT  |         Unimplemented           |
// | FLOAT16 | FLOAT16 |         Paddle -> Popart        |
// |-----------------------------------------------------|
// floatToHalf -> Paddle: cast then save to paddle
// Paddle -> Popart: copy from paddle to popart
void Executor::WeightsFromPaddle() {
  ConvertWeights(true);
  session_->writeWeights(executor_resources_->weights_io);
  session_->weightsFromHost();
}

// |-----------------------------------------------------|
// | Paddle  | Popart  |             Method              |
// |-----------------------------------------------------|
// |  FLOAT  |  FLOAT  |         Popart -> Paddle        |
// |  FLOAT  | FLOAT16 | Popart -> Paddle -> halfToFloat |
// | FLOAT16 |  FLOAT  |         Unimplemented           |
// | FLOAT16 | FLOAT16 |         Popart -> Paddle        |
// |-----------------------------------------------------|
// Paddle -> halfToFloat: cast then save to paddle
// Popart -> Paddle: copy from paddle to popart
void Executor::WeightsToPaddle() {
  session_->weightsToHost();
  session_->readWeights(executor_resources_->weights_io);
  ConvertWeights(false);
}

void Executor::SaveModelToHost(const std::string &path) {
  if (session_) {
    WeightsToPaddle();
    session_->modelToHost(path);
  } else {
    LOG(WARNING) << "Model is empty";
  }
}

}  // namespace ipu
}  // namespace platform
}  // namespace paddle
