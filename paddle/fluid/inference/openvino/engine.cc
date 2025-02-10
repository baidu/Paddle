/* Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/inference/openvino/engine.h"
#include <glog/logging.h>
#include <string>

namespace paddle::inference::openvino {
bool OpenVINOEngine::IsModelStatic() {
  bool isStatic = true;
  for (auto&& it : complied_model_.inputs()) {
    if (isStatic && it.get_partial_shape().is_dynamic()) {
      isStatic = !isStatic;
    }
  }
  return isStatic;
}

ov::Shape OpenVINOEngine::GetOutputShape(const std::string& output_name,
                                         int64_t index) {
  auto ov_output_shape =
      HaveOutputTensorName(output_name)
          ? infer_request_.get_tensor(output_name).get_shape()
          : infer_request_.get_output_tensor(index).get_shape();
  return ov_output_shape;
}

phi::DataType OpenVINOEngine::GetOutputType(const std::string& output_name,
                                            int64_t index,
                                            ov::element::Type ov_paddle_type) {
  auto output_ov_type =
      HaveOutputTensorName(output_name)
          ? infer_request_.get_tensor(output_name).get_element_type()
          : infer_request_.get_output_tensor(index).get_element_type();
  PADDLE_ENFORCE_EQ(
      output_ov_type == ov_paddle_type,
      true,
      common::errors::PreconditionNotMet(
          "Model output_name[%s]  index[%d] requires output type [%s],but "
          "receives type [%s]",
          output_name,
          index,
          OVType2PhiType(ov_paddle_type),
          OVType2PhiType(output_ov_type)));
  return OVType2PhiType(output_ov_type);
}

void OpenVINOEngine::CopyOutputDataByName(const std::string& output_name,
                                          int64_t index,
                                          void* pd_data) {
  auto ov_tensor = HaveOutputTensorName(output_name)
                       ? infer_request_.get_tensor(output_name)
                       : infer_request_.get_output_tensor(index);
  std::memcpy(pd_data, ov_tensor.data(), ov_tensor.get_byte_size());
}

void OpenVINOEngine::Execute() {
  try {
    infer_request_.infer();
  } catch (const std::exception& exp) {
    LOG(ERROR) << exp.what();
  }
}

void OpenVINOEngine::BuildEngine() {
  std::string model_path = params_.model_program_path;
  std::string param_path = params_.model_params_path;

  ov::frontend::FrontEndManager fem;
  ov::frontend::FrontEnd::Ptr frontEnd;
  ov::frontend::InputModel::Ptr inputModel;
  bool flag_success{true};

  frontEnd = fem.load_by_framework(paddle_frontend_name_);
  PADDLE_ENFORCE_NOT_NULL(frontEnd,
                          common::errors::Unavailable(
                              "Cannot find the front-end paddle in OpenVINO."));
  inputModel = frontEnd->load(model_path);
  std::shared_ptr<ov::Model> model;
  model = frontEnd->convert_partially(inputModel);
  for (auto& node : model->get_ordered_ops()) {
    if (node->description() == "FrameworkNode") {
      flag_success = false;
      VLOG(3) << "Can't convert op:" << node->get_friendly_name()
              << " to openvino op.";
    }
  }
  PADDLE_ENFORCE_EQ(flag_success,
                    true,
                    common::errors::Unavailable(
                        "Can't convert paddle model to openvino model. Please "
                        "check if the model is supported by openvino."));

  core_ = ov::Core();
  core_.set_property(
      ov::inference_num_threads(params_.cpu_math_library_num_threads));
  core_.set_property(ov::cache_dir(params_.model_opt_cache_dir));
  core_.set_property(ov::hint::inference_precision(
      PhiType2OVType(static_cast<phi::DataType>(params_.inference_precision))));
  model_ =
      core_.read_model(params_.model_program_path, params_.model_params_path);
  complied_model_ = core_.compile_model(model_, "CPU");
  infer_request_ = complied_model_.create_infer_request();
}

}  // namespace paddle::inference::openvino
