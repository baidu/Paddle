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
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/inference/tensorrt/engine.h"
#include "paddle/fluid/platform/enforce.h"

#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"
#include "paddle/fluid/inference/tensorrt/plugin/spmm_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {

class TensorRTEngineTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ctx_ = new platform::CUDADeviceContext(platform::CUDAPlace(0));
    ctx_->SetAllocator(paddle::memory::allocation::AllocatorFacade::Instance()
                           .GetAllocator(platform::CUDAPlace(0), ctx_->stream())
                           .get());
    ctx_->SetHostAllocator(
        paddle::memory::allocation::AllocatorFacade::Instance()
            .GetAllocator(paddle::platform::CPUPlace())
            .get());
    ctx_->SetZeroAllocator(
        paddle::memory::allocation::AllocatorFacade::Instance()
            .GetZeroAllocator(platform::CUDAPlace(0))
            .get());
    ctx_->SetPinnedAllocator(
        paddle::memory::allocation::AllocatorFacade::Instance()
            .GetAllocator(paddle::platform::CUDAPinnedPlace())
            .get());
    ctx_->PartialInitWithAllocator();

    std::map<std::string, std::vector<int>> min_input_shape = {
        {"input", {16, 32, 1, 1}}};
    std::map<std::string, std::vector<int>> max_input_shape = {
        {"input", {16, 32, 1, 1}}};
    std::map<std::string, std::vector<int>> optim_input_shape = {
        {"input", {16, 32, 1, 1}}};

    engine_ = new TensorRTEngine(16, 1<<10,
      AnalysisConfig::Precision::kFloat32, nullptr, 0,
      min_input_shape, max_input_shape, optim_input_shape, false, NaiveLogger::Global());
    engine_->InitNetwork();
  }

  void TearDown() override {
    if (engine_) {
      delete engine_;
      engine_ = nullptr;
    }
  }

  void PrepareInputOutput(const std::vector<float> &input,
                          std::vector<int> output_shape) {
    paddle::framework::TensorFromVector(input, *ctx_, &input_);
    output_.Resize(phi::make_ddim(output_shape));
  }

  void GetOutput(std::vector<float> *output) {
    paddle::framework::TensorToVector(output_, *ctx_, output);
  }

 protected:
  framework::Tensor input_;
  framework::Tensor output_;
  TensorRTEngine *engine_;
  platform::CUDADeviceContext *ctx_;
};

TEST_F(TensorRTEngineTest, test_sparse_fc) {
  // Weight in CPU memory.
  // float raw_weight[16] = {1.0, 0, 0, 4.4,
  //                        1.0, 0, 0, 4.4,
  //                        1.0, 0, 0, 4.4,
  //                        1.0, 0, 0, 4.4};
  float raw_weight[512];
  for (int i=0; i<128; i++) {
    if (i % 16 <= 7) {
    raw_weight[4*i] = 1.0;
    raw_weight[4*i+1] = 0.0;
    raw_weight[4*i+2] = 0.0;
    raw_weight[4*i+3] = 4.0;
    }
    else {
    raw_weight[4*i] = 0.0;
    raw_weight[4*i+1] = 2.0;
    raw_weight[4*i+2] = 3.0;
    raw_weight[4*i+3] = 0.0;
    }
  }
  /*  
  float raw_weight_transpose[512];
  for (int i=0; i<32; i++) {
    for (int j=0; j<16; j++) {
      raw_weight_transpose[j*32+i] = raw_weight[i*16+j];
    }
  }
  */
  float raw_bias[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<void *> buffers(2);  // TRT binded inputs
  TensorRTEngine::Weight weight(nvinfer1::DataType::kFLOAT, raw_weight, 512);
  TensorRTEngine::Weight bias(nvinfer1::DataType::kFLOAT, raw_bias, 16);
  std::cout << "with_dynamic_shape: " << engine_->with_dynamic_shape() << std::endl;
  auto *x = engine_->DeclareInput("x", nvinfer1::DataType::kFLOAT,
                                  nvinfer1::Dims4{-1, 32, 1, 1});

  plugin::SpmmPluginDynamic::Activation act = plugin::SpmmPluginDynamic::Activation::kNone;

  plugin::SpmmPluginDynamic* plugin = new plugin::SpmmPluginDynamic("CustomSpmmPluginDynamic", nvinfer1::DataType::kFLOAT, 16, weight.get(), bias.get(), act);
  std::vector<nvinfer1::ITensor*> plugin_inputs;
  plugin_inputs.emplace_back(x);
  auto fc_layer = engine_->network()->addPluginV2(
      plugin_inputs.data(), plugin_inputs.size(), *plugin);

  PADDLE_ENFORCE_NOT_NULL(fc_layer,
                          platform::errors::InvalidArgument(
                              "TRT SPARSE FC layer building failed."));

  engine_->DeclareOutput(fc_layer, 0, "y");
  engine_->FreezeNetwork();
  ASSERT_EQ(engine_->engine()->getNbBindings(), 2);

  // fill in real data
  // std::vector<float> x_v = {1.0, 2.0, 3.0, 4.0,
  //                           1.0, 2.0, 3.0, 4.0,
  //                           1.0, 2.0, 3.0, 4.0,
  //                           1.0, 2.0, 3.0, 4.0};
  std::vector<float> x_v(512);
  for (int i=0; i<128; i++) {
    x_v[4*i] = 1.0;
    x_v[4*i+1] = 2.0;
    x_v[4*i+2] = 3.0;
    x_v[4*i+3] = 4.0;
  }

  std::vector<float> y_cpu;
  PrepareInputOutput(x_v, {16});

  auto *x_v_gpu_data = input_.mutable_data<float>(ctx_->GetPlace());
  auto *y_gpu_data = output_.mutable_data<float>(ctx_->GetPlace());

  auto context = engine_->context();
  context->setBindingDimensions(0, nvinfer1::Dims4{16,32,1,1});
  context->setBindingDimensions(1, nvinfer1::Dims4{16,16,1,1});
  buffers[0] = reinterpret_cast<void *>(x_v_gpu_data);
  buffers[1] = reinterpret_cast<void *>(y_gpu_data);

  engine_->Execute(16, &buffers, context, ctx_->stream());

  LOG(INFO) << "to get output";
  GetOutput(&y_cpu);

  auto dims = engine_->GetITensor("y")->getDimensions();
  ASSERT_EQ(dims.nbDims, 4);
  ASSERT_EQ(dims.d[1], 16);

  std::cout << "output: ";
  for (int i=0; i<256; i++) {
    std::cout << " " << y_cpu[i];
  }

  ASSERT_EQ(y_cpu[0], 136);
  ASSERT_EQ(y_cpu[1], 104);
  ASSERT_EQ(y_cpu[32], 136);
  ASSERT_EQ(y_cpu[64], 136);
  ASSERT_EQ(y_cpu[96], 136);
}

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
