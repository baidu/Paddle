// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"
#include "paddle/fluid/inference/tensorrt/plugin/c_allreduce_op_plugin.h"
#include "paddle/phi/common/data_type.h"

namespace paddle::inference::tensorrt {
using ReduceType = paddle::inference::tensorrt::plugin::ReduceType;
std::map<std::string, ReduceType> op_to_reduce_type = {
    {"c_allreduce_sum", paddle::inference::tensorrt::plugin::kRedSum},
    {"c_allreduce_max", paddle::inference::tensorrt::plugin::kRedMax},
    {"c_allreduce_min", paddle::inference::tensorrt::plugin::kRedMin},
    {"c_allreduce_prod", paddle::inference::tensorrt::plugin::kRedProd}};

class CAllReduceOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(4) << "convert c_allreduce op to tensorrt layer";
    if (!engine_->with_dynamic_shape()) {
      PADDLE_THROW(
          common::errors::Fatal("Unsupported static graph mode. Please set "
                                "dynamic shape of inputs."));
    }
    ReduceType red_type = op_to_reduce_type[op.type()];
    std::string name = op.type();

    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    int input_num = op_desc.Input("X").size();
    PADDLE_ENFORCE_EQ(
        input_num,
        1,
        common::errors::InvalidArgument(
            "The input X's size must equal to 1 in TRT c_allreduce op."
            " But received X's size %d.",
            input_num));
    auto* input = engine_->GetITensor(op_desc.Input("X")[0]);
    // Get output
    size_t output_num = op_desc.Output("Out").size();
    PADDLE_ENFORCE_EQ(
        output_num,
        1UL,
        common::errors::InvalidArgument(
            "The output Out's size must equal to 1 in TRT c_allreduce op. "
            "But received Out's size %u.",
            output_num));
    // Get attrs
    int ring_id = PADDLE_GET_CONST(int, op_desc.GetAttr("ring_id"));
    bool use_calc_stream =
        PADDLE_GET_CONST(bool, op_desc.GetAttr("use_calc_stream"));

    nvinfer1::ILayer* layer = nullptr;
#if IS_TRT_VERSION_GE(6000)
    bool with_fp16 = engine_->WithFp16() && !engine_->disable_trt_plugin_fp16();

    if (engine_->precision() == phi::DataType::INT8) {
      with_fp16 = true;
    }

    plugin::CAllReducePluginDynamic* plugin =
        new plugin::CAllReducePluginDynamic(
            ring_id, use_calc_stream, red_type, with_fp16);
    layer = engine_->AddDynamicPlugin(&input, input_num, plugin);
#else
    PADDLE_THROW(common::errors::Fatal(
        "You are running the TRT Dynamic Shape mode, need to confirm that "
        "your TRT version is no less than 6.0"));
#endif
    auto output_name = op_desc.Output("Out")[0];

    ReplenishLayerAndOutput(layer, name, {output_name}, test_mode);
  }
};

}  // namespace paddle::inference::tensorrt

REGISTER_TRT_OP_CONVERTER(c_allreduce_sum, CAllReduceOpConverter);
REGISTER_TRT_OP_CONVERTER(c_allreduce_max, CAllReduceOpConverter);
REGISTER_TRT_OP_CONVERTER(c_allreduce_min, CAllReduceOpConverter);
REGISTER_TRT_OP_CONVERTER(c_allreduce_prod, CAllReduceOpConverter);
