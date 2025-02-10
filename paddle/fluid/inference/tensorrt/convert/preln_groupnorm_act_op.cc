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

#include "paddle/fluid/inference/tensorrt/plugin/preln_groupnorm_act_op_plugin.h"

#include <vector>

#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"
#include "paddle/fluid/inference/tensorrt/engine.h"

namespace paddle::inference::tensorrt {

class PrelnGroupnormActOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(4) << "convert a preln_groupnorm_act op to tensorrt "
               "preln_groupnorm_act plugin";

    framework::OpDesc op_desc(op, nullptr);

    auto* input_x = engine_->GetITensor(op_desc.Input("X").front());
    auto* input_y = engine_->GetITensor(op_desc.Input("Y").front());
    std::vector<nvinfer1::ITensor*> inputs{input_x, input_y};

    int groups = PADDLE_GET_CONST(int, op_desc.GetAttr("groups"));
    float epsilon = PADDLE_GET_CONST(float, op_desc.GetAttr("epsilon"));
    bool with_silu = PADDLE_GET_CONST(bool, op_desc.GetAttr("with_silu"));

    std::string scale_name = op_desc.Input("Scale").front();
    std::string bias_name = op_desc.Input("Bias").front();

    // get the persistable var's data
    auto GetWeight = [&](const std::string& var_name,
                         phi::DDim* dims) -> TensorRTEngine::Weight {
      auto* temp_var = scope.FindVar(var_name);
      auto* temp_tensor = temp_var->GetMutable<phi::DenseTensor>();
      (*dims) = temp_tensor->dims();

      auto weight = engine_->GetTrtWeight(var_name, *temp_tensor);
      return weight;
    };

    phi::DDim scale_dims;
    phi::DDim bias_dims;
    auto scale_weights = GetWeight(scale_name, &scale_dims);
    auto bias_weights = GetWeight(bias_name, &bias_dims);
    bool with_fp16 = engine_->WithFp16() && !engine_->disable_trt_plugin_fp16();

    plugin::PrelnGroupnormActPluginDynamic* plugin =
        new plugin::PrelnGroupnormActPluginDynamic(
            static_cast<const float*>(scale_weights.get().values),
            scale_weights.get().count,
            static_cast<const float*>(bias_weights.get().values),
            bias_weights.get().count,
            epsilon,
            groups,
            with_silu,
            with_fp16);
    nvinfer1::ILayer* groupnorm_layer =
        engine_->AddDynamicPlugin(inputs.data(), 2, plugin);
    std::vector<std::string> output_names;
    output_names.emplace_back(op_desc.Output("Out_0").front());
    output_names.emplace_back(op_desc.Output("Out_1").front());
    ReplenishLayerAndOutput(
        groupnorm_layer, "preln_groupnorm_act", output_names, test_mode);
  }
};

}  // namespace paddle::inference::tensorrt

REGISTER_TRT_OP_CONVERTER(preln_groupnorm_act, PrelnGroupnormActOpConverter);
