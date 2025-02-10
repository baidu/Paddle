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

#include "paddle/common/ddim.h"
#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"
#include "paddle/fluid/inference/tensorrt/convert/utils.h"
#include "paddle/fluid/inference/tensorrt/engine.h"
#include "paddle/fluid/inference/tensorrt/helper.h"
#include "paddle/fluid/inference/tensorrt/plugin/many_emb_layernorm_plugin.h"
#include "paddle/fluid/inference/tensorrt/plugin/many_emb_layernorm_varseqlen_plugin.h"

namespace paddle::inference::tensorrt {

class EmbEltwiseLayerNormOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(4) << "convert EmbEltwiseLayerNorm op to tensorrt layer";
    // get the persistable var's data
    auto GetWeight = [&](const std::string& var_name,
                         phi::DDim* dim) -> TensorRTEngine::Weight {
      auto* temp_var = scope.FindVar(var_name);
      auto* temp_tensor = temp_var->GetMutable<phi::DenseTensor>();
      *dim = temp_tensor->dims();
      auto weight = engine_->GetTrtWeight(var_name, *temp_tensor);
      return weight;
    };

    framework::OpDesc op_desc(op, nullptr);
    auto pos_id_name = engine_->tensorrt_transformer_posid();
    auto mask_id_name = engine_->tensorrt_transformer_maskid();
    bool flag_varseqlen = engine_->use_varseqlen() && !pos_id_name.empty() &&
                          !mask_id_name.empty();
    // bool with_fp16 = engine_->WithFp16() &&
    // !engine_->disable_trt_plugin_fp16(); int hidden = 0; Declare inputs
    std::vector<nvinfer1::ITensor*> input_ids;

    // Declare inputs_weight
    std::vector<nvinfer1::Weights> input_embs;
    std::vector<int> emb_sizes;
    TensorRTEngine::Weight weight;
    phi::DDim emb_dims;
    phi::DDim bias_dims, scale_dims;
    TensorRTEngine::Weight bias_weight, scale_weight;

    int64_t bias_size = common::product(bias_dims);
    int64_t scale_size = common::product(scale_dims);
    nvinfer1::ILayer* layer = nullptr;
    bool enable_int8 = op_desc.HasAttr("enable_int8");

    std::vector<std::string> id_names = op_desc.Input("Ids");
    std::vector<std::string> emb_names = op_desc.Input("Embs");
    int input_num = id_names.size();

    if (flag_varseqlen) {
      engine_->SetITensor("pos_id", engine_->GetITensor(pos_id_name));
      engine_->SetITensor("mask_id", engine_->GetITensor(mask_id_name));
      for (int i = 0; i < input_num; i++) {
        auto input_tensor = engine_->GetITensor(id_names[i]);
        weight = GetWeight(emb_names[i], &emb_dims);
        if (id_names[i] == pos_id_name) {
          input_ids.insert(input_ids.begin(), input_tensor);
          input_embs.insert(input_embs.begin(), weight.get());
          emb_sizes.insert(emb_sizes.begin(), weight.get().count);
        } else {
          input_ids.push_back(input_tensor);
          input_embs.push_back(weight.get());
          emb_sizes.push_back(weight.get().count);
        }
      }
      bias_weight = GetWeight(op_desc.Input("Bias").front(), &bias_dims);
      scale_weight = GetWeight(op_desc.Input("Scale").front(), &scale_dims);
      bias_size = common::product(bias_dims);
      scale_size = common::product(scale_dims);
      // other_id(except pos_id)
      engine_->SetITensor("word_id", input_ids[1]);

      int output_fp16 = static_cast<int>((engine_->WithFp16() == 1) ? 1 : 0);
      if (enable_int8) {
        output_fp16 = 1;
      }
      PADDLE_ENFORCE_EQ(
          output_fp16,
          1,
          common::errors::InvalidArgument(
              "Only Precision::KHalf(fp16) is supported when inferring "
              "ernie(bert) model with config.EnableVarseqlen(). "
              "But Precision::KFloat32 is setted."));

      std::vector<nvinfer1::PluginField> fields;
      std::vector<std::string> temp_fields_keys;
      fields.emplace_back("bert_embeddings_layernorm_beta",
                          bias_weight.get().values,
                          GetPluginFieldType(bias_weight.get().type),
                          static_cast<int32_t>(bias_size));
      fields.emplace_back("bert_embeddings_layernorm_gamma",
                          scale_weight.get().values,
                          GetPluginFieldType(scale_weight.get().type),
                          static_cast<int32_t>(scale_size));
      fields.emplace_back(
          "output_fp16", &output_fp16, nvinfer1::PluginFieldType::kINT32, 1);
      for (int i = 0; i < input_num; ++i) {
        temp_fields_keys.push_back("bert_embeddings_word_embeddings_" +
                                   std::to_string(i));
        fields.emplace_back(temp_fields_keys.rbegin()->c_str(),
                            input_embs[i].values,
                            GetPluginFieldType(input_embs[i].type),
                            static_cast<int32_t>(emb_sizes[i]));
      }

      std::unique_ptr<nvinfer1::PluginFieldCollection> plugin_ptr(
          new nvinfer1::PluginFieldCollection);
      plugin_ptr->nbFields = static_cast<int>(fields.size());
      plugin_ptr->fields = fields.data();

      std::vector<nvinfer1::ITensor*> plugin_inputs = input_ids;
      plugin_inputs.emplace_back(
          engine_->GetITensor("mask_id"));  // input mask_id
      auto creator = GetPluginRegistry()->getPluginCreator(
          "ManyEmbLayerNormVarlenPluginDynamic", "1");
      auto plugin_obj = creator->createPlugin(
          "ManyEmbLayerNormVarlenPluginDynamic", plugin_ptr.get());

      auto plugin_layer = engine_->network()->addPluginV2(
          plugin_inputs.data(), plugin_inputs.size(), *plugin_obj);

      plugin_layer->setName(("ManyEmbLayerNormVarlenPluginDynamicV1(Output: " +
                             op_desc.Output("Out")[0] + ")")
                                .c_str());
      plugin_ptr.reset();
      if (enable_int8) {
        float out_scale =
            PADDLE_GET_CONST(float, op_desc.GetAttr("out_threshold"));
        engine_->SetTensorDynamicRange(plugin_layer->getOutput(0),
                                       out_scale);  // output
        engine_->SetTensorDynamicRange(plugin_layer->getOutput(1),
                                       out_scale);  // mask
        engine_->SetTensorDynamicRange(plugin_layer->getOutput(2),
                                       out_scale);  // max seqlen
      }
      if (engine_->with_interleaved()) {
        VLOG(4) << "fused emb_eltwise_layernorm op: use_varseqlen and "
                   "with_interleaved";
        if (!enable_int8) {
          PADDLE_THROW(
              common::errors::Fatal("use with_interleaved must be int8."));
        }
        auto* shuffler_embed = TRT_ENGINE_ADD_LAYER(
            engine_, Shuffle, *(plugin_layer->getOutput(0)));
        nvinfer1::Permutation transpose_embed{2, 1, 0, 3};
        shuffler_embed->setSecondTranspose(transpose_embed);
        engine_->SetITensor(op_desc.Output("Out")[0],
                            shuffler_embed->getOutput(0));
        shuffler_embed->setName(
            ("Emb_eltwise_out_shuffler_transpose (Output: " +
             op_desc.Output("Out")[0] + ")")
                .c_str());
      } else {
        layer = plugin_layer;
        auto output_name = op_desc.Output("Out")[0];
        ReplenishLayerAndOutput(layer,
                                "ManyEmbLayerNormVarlenPluginDynamicV1",
                                {output_name,
                                 std::string("qkv_plugin_mask"),
                                 std::string("max_seqlen_tensor")},
                                test_mode);
      }
    } else {
      for (int i = 0; i < input_num; i++) {
        auto input_tensor = engine_->GetITensor(id_names[i]);
        weight = GetWeight(emb_names[i], &emb_dims);
        input_ids.push_back(input_tensor);
        input_embs.push_back(weight.get());
        emb_sizes.push_back(weight.get().count);
        //  hidden = emb_dims[1];
      }
      bias_weight = GetWeight(op_desc.Input("Bias").front(), &bias_dims);
      scale_weight = GetWeight(op_desc.Input("Scale").front(), &scale_dims);
      bias_size = common::product(bias_dims);
      scale_size = common::product(scale_dims);

      int output_fp16 = static_cast<int>((engine_->WithFp16() == 1) ? 1 : 0);
      if (enable_int8) {
        output_fp16 = 1;
      }

      std::vector<nvinfer1::PluginField> fields;
      std::vector<std::string> temp_fields_keys;
      fields.emplace_back("bert_embeddings_layernorm_beta",
                          bias_weight.get().values,
                          GetPluginFieldType(bias_weight.get().type),
                          static_cast<int32_t>(bias_size));
      fields.emplace_back("bert_embeddings_layernorm_gamma",
                          scale_weight.get().values,
                          GetPluginFieldType(scale_weight.get().type),
                          static_cast<int32_t>(scale_size));
      fields.emplace_back(
          "output_fp16", &output_fp16, nvinfer1::PluginFieldType::kINT32, 1);
      for (int i = 0; i < input_num; ++i) {
        temp_fields_keys.push_back("bert_embeddings_word_embeddings_" +
                                   std::to_string(i));
        fields.emplace_back(temp_fields_keys.rbegin()->c_str(),
                            input_embs[i].values,
                            GetPluginFieldType(input_embs[i].type),
                            static_cast<int32_t>(emb_sizes[i]));
      }

      std::unique_ptr<nvinfer1::PluginFieldCollection> plugin_ptr(
          new nvinfer1::PluginFieldCollection);
      plugin_ptr->nbFields = static_cast<int>(fields.size());
      plugin_ptr->fields = fields.data();

      std::vector<nvinfer1::ITensor*> plugin_inputs = input_ids;

      auto creator = GetPluginRegistry()->getPluginCreator(
          "ManyEmbLayerNormPluginDynamic", "1");
      auto plugin_obj = creator->createPlugin("ManyEmbLayerNormPluginDynamic",
                                              plugin_ptr.get());

      auto plugin_layer = engine_->network()->addPluginV2(
          plugin_inputs.data(), plugin_inputs.size(), *plugin_obj);

      plugin_layer->setName(("ManyEmbLayerNormPluginDynamicV1(Output: " +
                             op_desc.Output("Out")[0] + ")")
                                .c_str());
      plugin_ptr.reset();
      if (enable_int8) {
        float out_scale =
            PADDLE_GET_CONST(float, op_desc.GetAttr("out_threshold"));
        engine_->SetTensorDynamicRange(plugin_layer->getOutput(0),
                                       out_scale);  // output
      }
      layer = plugin_layer;
      auto output_name = op_desc.Output("Out")[0];
      ReplenishLayerAndOutput(
          layer, "ManyEmbLayerNormPluginDynamicV1", {output_name}, test_mode);
    }
  }
};

}  // namespace paddle::inference::tensorrt

REGISTER_TRT_OP_CONVERTER(fused_embedding_eltwise_layernorm,
                          EmbEltwiseLayerNormOpConverter);
