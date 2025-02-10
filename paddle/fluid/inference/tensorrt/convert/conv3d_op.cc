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

#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"

namespace paddle::inference::tensorrt {

template <typename RegisterFunc, typename SetDilationFunc>
void ConvertConv3d(TensorRTEngine* engine,
                   const framework::proto::OpDesc& op,
                   const framework::Scope& scope,
                   bool test_mode,
                   RegisterFunc fadd_layer,
                   SetDilationFunc fset_dilation,
                   const std::string& name) {
  VLOG(3) << "convert a " << name << " op to tensorrt layer without bias";

  framework::OpDesc op_desc(op, nullptr);

  auto* X = engine->GetITensor(op_desc.Input("Input").front());
  std::string filter_var_name = op_desc.Input("Filter").front();
  auto* Y_v = scope.FindVar(filter_var_name);
  PADDLE_ENFORCE_NOT_NULL(
      Y_v,
      common::errors::NotFound("Can not find %s persistable var in scope.",
                               filter_var_name));
  auto* Y_t = Y_v->GetMutable<phi::DenseTensor>();
  bool enable_int8 = op_desc.HasAttr("enable_int8");

  if (enable_int8) {
    float in_scale = PADDLE_GET_CONST(float, op_desc.GetAttr("Input_scale"));
    engine->SetTensorDynamicRange(X, in_scale);
  }

  PADDLE_ENFORCE_EQ(Y_t->dims().size(),
                    5UL,
                    common::errors::InvalidArgument(
                        "The conv3d filter's dims size should be 5, but got %d",
                        Y_t->dims().size()));

  const int n_output = Y_t->dims()[0];
  const int n_input = Y_t->dims()[1];
  const int filter_d = Y_t->dims()[2];
  const int filter_h = Y_t->dims()[3];
  const int filter_w = Y_t->dims()[4];
  const int groups = PADDLE_GET_CONST(int, op_desc.GetAttr("groups"));
  const std::vector<int> dilations =
      PADDLE_GET_CONST(std::vector<int>, op_desc.GetAttr("dilations"));
  const std::vector<int> strides =
      PADDLE_GET_CONST(std::vector<int>, op_desc.GetAttr("strides"));
  const std::vector<int> paddings =
      PADDLE_GET_CONST(std::vector<int>, op_desc.GetAttr("paddings"));
  std::string padding_algorithm = "EXPLICIT";
  if (op_desc.HasAttr("padding_algorithm"))
    padding_algorithm =
        PADDLE_GET_CONST(std::string, op_desc.GetAttr("padding_algorithm"));

  // for conv3d_transpose
  std::vector<int> output_padding;
  if (op_desc.HasAttr("output_padding")) {
    output_padding =
        PADDLE_GET_CONST(std::vector<int>, op_desc.GetAttr("output_padding"));
  }

  nvinfer1::Dims3 nv_ksize(filter_d, filter_h, filter_w);
  nvinfer1::Dims3 nv_dilations(dilations[0], dilations[1], dilations[2]);
  nvinfer1::Dims3 nv_strides(strides[0], strides[1], strides[2]);
  nvinfer1::Dims3 nv_pre_paddings(paddings[0], paddings[1], paddings[2]);

  auto weight = engine->GetTrtWeight(op_desc.Input("Filter").front(), *Y_t);
  float* bias_data = nullptr;
  size_t bias_size = 0;

  TensorRTEngine::Weight bias{
      weight.get().type, static_cast<void*>(bias_data), bias_size};
  // In conv3d_transpose output channels = filter_dims[1] * groups
  auto* layer = (op_desc.Type() == "conv3d_transpose")
                    ? fadd_layer(X, n_input * groups, nv_ksize, weight, bias)
                    : fadd_layer(X, n_output, nv_ksize, weight, bias);

  PADDLE_ENFORCE_NOT_NULL(
      layer,
      common::errors::Fatal("TensorRT create conv3d/conv3d_transpose"
                            " layer failed."));
  layer->setStrideNd(nv_strides);
  layer->setPrePadding(nv_pre_paddings);
  nvinfer1::Dims3 nv_post_paddings = nv_pre_paddings;
  if (!output_padding.empty()) {
// Here is consistent with op_teller.cc
#if IS_TRT_VERSION_GE(8400)
    nv_post_paddings.d[0] -= output_padding[0];
    nv_post_paddings.d[1] -= output_padding[1];
    nv_post_paddings.d[2] -= output_padding[2];

    if (nv_post_paddings.d[0] < 0 || nv_post_paddings.d[1] < 0 ||
        nv_post_paddings.d[2] < 0) {
      PADDLE_THROW(common::errors::Fatal(
          "The value in conv3d_transpose's PostPadding should be >= 0."));
    }

#endif
  }
  layer->setPostPadding(nv_post_paddings);

  layer->setNbGroups(groups);
  if (padding_algorithm == "SAME") {
    layer->setPaddingMode(nvinfer1::PaddingMode::kSAME_UPPER);
  }
  // set dilations
  fset_dilation(layer, nv_dilations);

  auto output_name = op_desc.Output("Output").front();
  layer->setName((name + " (Output: " + output_name + ")").c_str());
  layer->getOutput(0)->setName(output_name.c_str());
  engine->SetITensor(output_name, layer->getOutput(0));

  if (test_mode) {
    engine->DeclareOutput(output_name);
  }
}

class Conv3dOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    ConvertConv3d(
        engine_,
        op,
        scope,
        test_mode,
        [&](nvinfer1::ITensor* inputs,
            int n_output, /* Conv output maps */
            nvinfer1::Dims& ksize,
            TensorRTEngine::Weight& weight,
            TensorRTEngine::Weight& bias) -> nvinfer1::IConvolutionLayer* {
          auto* layer = TRT_ENGINE_ADD_LAYER(engine_,
                                             ConvolutionNd,
                                             *inputs,
                                             n_output,
                                             ksize,
                                             weight.get(),
                                             bias.get());
          return layer;
        },
        [](nvinfer1::IConvolutionLayer* layer, nvinfer1::Dims& dilations) {
          layer->setDilationNd(dilations);
        },
        "conv3d");
  }
};

class Deconv3dOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    ConvertConv3d(
        engine_,
        op,
        scope,
        test_mode,
        [&](nvinfer1::ITensor* inputs,
            int n_output, /* Deconv input maps */
            nvinfer1::Dims& ksize,
            TensorRTEngine::Weight& weight,
            TensorRTEngine::Weight& bias) -> nvinfer1::IDeconvolutionLayer* {
          auto* layer = TRT_ENGINE_ADD_LAYER(engine_,
                                             DeconvolutionNd,
                                             *inputs,
                                             n_output,
                                             ksize,
                                             weight.get(),
                                             bias.get());
          return layer;
        },
        [](nvinfer1::IDeconvolutionLayer* layer, nvinfer1::Dims& dilations) {},
        "conv3d_transpose");
  }
};

}  // namespace paddle::inference::tensorrt

REGISTER_TRT_OP_CONVERTER(conv3d, Conv3dOpConverter);
REGISTER_TRT_OP_CONVERTER(conv3d_transpose, Deconv3dOpConverter);
