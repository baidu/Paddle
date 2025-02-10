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

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/inference/analysis/helper.h"
#include "paddle/fluid/inference/tensorrt/engine.h"
#include "paddle/fluid/inference/tensorrt/helper.h"
#include "paddle/fluid/inference/tensorrt/op_teller.h"
#include "paddle/fluid/inference/utils/singleton.h"
#include "paddle/phi/common/data_type.h"

namespace paddle {
namespace inference {
namespace tensorrt {

/*
 * Convert Op from Fluid to TensorRT Engine.
 */
class OpConverter {
 public:
  OpConverter() {}

  // Converter logic for an op.
  virtual void operator()(const framework::proto::OpDesc& op,
                          const framework::Scope& scope,
                          bool test_mode = false) {}

  // Convert a single fluid operator and add the corresponding layer to TRT.
  // test_mode: whether the instance executes in an unit test.
  void ConvertOp(const framework::proto::OpDesc& op,
                 const std::unordered_set<std::string>& parameters,
                 const framework::Scope& scope,
                 TensorRTEngine* engine,
                 bool test_mode = false,
                 const framework::proto::BlockDesc* block = nullptr) {
    framework::OpDesc op_desc(op, nullptr);

    OpConverter* it{nullptr};

    auto converter_type = static_cast<OpConverterType>(
        PADDLE_GET_CONST(int, op_desc.GetAttr("converter_type")));
    switch (converter_type) {
      case OpConverterType::Default:
        if (op_desc.Type().find("elementwise") != std::string::npos) {
          static std::unordered_set<std::string> add_tensor_op_set{
              "add", "mul", "sub", "div", "max", "min", "pow", "mod"};
          static std::unordered_set<std::string> add_weight_op_set{
              "add", "mul", "sub", "div", "max", "min", "pow", "mod"};
          PADDLE_ENFORCE_EQ(op_desc.Input("Y").size(),
                            1UL,
                            common::errors::InvalidArgument(
                                "The input op's Input(\"Y\")."
                                "size() should equal to 1, but received "
                                "Input(\"Y\").size() = %u.",
                                op_desc.Input("Y").size()));
          int op_type_len = op_desc.Type().size();
          std::string op_type =
              op_desc.Type().substr(op_type_len - 3, op_type_len);
          std::string Y = op_desc.Input("Y")[0];
          if (parameters.count(Y)) {
            PADDLE_ENFORCE_GT(
                add_weight_op_set.count(op_type),
                0,
                common::errors::Unimplemented("Unsupported elementwise type %s",
                                              op_type.c_str()));
            it = Registry<OpConverter>::Global().Lookup("elementwise_" +
                                                        op_type + "_weight");
            PADDLE_ENFORCE_NOT_NULL(
                it,
                common::errors::Unimplemented("no OpConverter for optype [%s]",
                                              op_desc.Type()));
          } else {
            PADDLE_ENFORCE_GT(
                add_tensor_op_set.count(op_type),
                0,
                common::errors::Unimplemented("Unsupported elementwise type %s",
                                              op_type.c_str()));
            it = Registry<OpConverter>::Global().Lookup("elementwise_" +
                                                        op_type + "_tensor");
          }
          PADDLE_ENFORCE_NOT_NULL(
              it,
              common::errors::Unimplemented("no OpConverter for optype [%s]",
                                            op_desc.Type()));
        }

        if (op_desc.Type() == "depthwise_conv2d") {
          it = Registry<OpConverter>::Global().Lookup("conv2d");
          PADDLE_ENFORCE_NOT_NULL(
              it,
              common::errors::Unimplemented("no OpConverter for optype [%s]",
                                            op_desc.Type()));
        }
        if (op_desc.Type() == "depthwise_conv2d_transpose") {
          it = Registry<OpConverter>::Global().Lookup("conv2d_transpose");
          PADDLE_ENFORCE_NOT_NULL(
              it,
              common::errors::Unimplemented("no OpConverter for optype [%s]",
                                            op_desc.Type()));
        }
        if (op_desc.Type() == "transpose2") {
          it = Registry<OpConverter>::Global().Lookup("transpose");
          PADDLE_ENFORCE_NOT_NULL(
              it,
              common::errors::Unimplemented("no OpConverter for optype [%s]",
                                            op_desc.Type()));
        }
        if (op_desc.Type() == "flatten2") {
          it = Registry<OpConverter>::Global().Lookup("flatten");
          PADDLE_ENFORCE_NOT_NULL(
              it,
              common::errors::Unimplemented("no OpConverter for optype [%s]",
                                            op_desc.Type()));
        }
        // reshape2 == reshape
        if (op_desc.Type() == "reshape2") {
          it = Registry<OpConverter>::Global().Lookup("reshape");
          PADDLE_ENFORCE_NOT_NULL(
              it,
              common::errors::Unimplemented("no OpConverter for optype [%s]",
                                            op_desc.Type()));
        }
        if (!it) {
          it = Registry<OpConverter>::Global().Lookup(op_desc.Type());
        }
        break;

      case OpConverterType::GenericPluginCreator:
        LOG(INFO) << "There is no OpConverter for type " << op_desc.Type()
                  << ", now use generic_plugin_creator!";
        it = Registry<OpConverter>::Global().Lookup("generic_plugin_creator");
        break;

      case OpConverterType::CustomPluginCreater:  // typos: disable-line
        LOG(INFO) << "There is no OpConverter for type " << op_desc.Type()
                  << ", now use custom_plugin_creater!";  // typos: disable-line
        it = Registry<OpConverter>::Global().Lookup(
            "custom_plugin_creater");  // typos: disable-line
        break;

      case OpConverterType::CustomGenericPluginCreator:
        LOG(INFO) << "There is no OpConverter for type " << op_desc.Type()
                  << ", now use custom_generic_plugin_creator!";
        it = Registry<OpConverter>::Global().Lookup(
            "custom_generic_plugin_creator");
        break;

      default:
        PADDLE_THROW(common::errors::Unimplemented(
            "No OpConverter for optype %s", op_desc.Type()));
    }

    PADDLE_ENFORCE_NOT_NULL(
        it,
        common::errors::Unimplemented("no OpConverter for optype [%s]",
                                      op_desc.Type()));

    std::string all_outputs_name = "(Outputs:";
    std::string all_inputs_name = "(Inputs:";
    for (auto it1 : op_desc.OutputNames()) {
      for (auto it2 : op_desc.Output(it1)) {
        all_outputs_name += it2;
        all_outputs_name += ",";
      }
    }
    all_outputs_name += ")";
    for (auto it1 : op_desc.InputNames()) {
      for (auto it2 : op_desc.Input(it1)) {
        all_inputs_name += it2;
        all_inputs_name += ",";
      }
    }

    all_inputs_name += ")";
    VLOG(1) << op_desc.Type() << all_inputs_name << all_outputs_name
            << "are to be converted to TensorRT layer";

    it->SetEngine(engine);
    engine->SetScope(&scope);
    it->SetBlockDesc(block);
    (*it)(op, scope, test_mode);

    size_t output_num = op_desc.OutputNames().size();
    // only one out SetTensorDynamicRange
    if (op_desc.HasAttr("out_threshold")) {
      float out_scale =
          PADDLE_GET_CONST(float, op_desc.GetAttr("out_threshold"));
      std::string output_name = "";
      if (op_desc.HasOutput("Output")) {
        output_name = op_desc.Output("Output").front();
      } else if (op_desc.HasOutput("Out")) {
        output_name = op_desc.Output("Out").front();
      } else if (op_desc.HasOutput("Y")) {
        output_name = op_desc.Output("Y").front();
      } else {
        PADDLE_THROW(
            common::errors::NotFound("Op %s has out threshold but doesn't "
                                     "have an output named \"Output\", "
                                     "\"Out\" or \"Y\".",
                                     op_desc.Type()));
      }

      auto* output_tensor = engine->GetITensor(output_name);
      engine->SetTensorDynamicRange(output_tensor, out_scale);
      VLOG(1) << "Set out scale = " << out_scale << " for tensor "
              << output_name << ".";
    }
    // outs SetTensorDynamicRange
    for (size_t i = 0; i < output_num; ++i) {
      if (op_desc.HasAttr("out_" + std::to_string(i) + "_threshold")) {
        float out_scale = PADDLE_GET_CONST(
            float, op_desc.GetAttr("out_" + std::to_string(i) + "_threshold"));
        std::string output_name =
            op_desc.Output(op_desc.OutputNames()[i]).front();
        auto* output_tensor = engine->GetITensor(output_name);
        engine->SetTensorDynamicRange(output_tensor, out_scale);
        VLOG(1) << "Set out scale = " << out_scale << " for tensor "
                << output_name << ".";
      }
    }

    // quant_dequant_linear support for paddle trt

    std::vector<std::string> inputs_name = op_desc.InputNames();
    std::vector<std::string> outputs_name = op_desc.OutputNames();

    for (size_t i = 0; i < inputs_name.size(); i++) {
      if (op_desc.HasAttr(inputs_name[i])) {
        std::string input_tensor_name = op_desc.Input(inputs_name[i])[0];
        auto* input_tensor = engine->GetITensor(input_tensor_name);
        float input_scale =
            PADDLE_GET_CONST(float, op_desc.GetAttr(inputs_name[i]));
        engine->SetTensorDynamicRange(input_tensor, input_scale);
        VLOG(1) << "Set input tensor scale = " << input_scale
                << " for tensor: " << input_tensor_name << ".";
      }
    }
    for (size_t i = 0; i < outputs_name.size(); i++) {
      if (op_desc.HasAttr(outputs_name[i])) {
        std::string output_tensor_name = op_desc.Output(outputs_name[i])[0];
        auto* output_tensor = engine->GetITensor(output_tensor_name);
        float output_scale =
            PADDLE_GET_CONST(float, op_desc.GetAttr(outputs_name[i]));
        engine->SetTensorDynamicRange(output_tensor, output_scale);
        VLOG(1) << "Set output tensor scale = " << output_scale
                << " for tensor: " << output_tensor_name << ".";
      }
    }
  }

  // Convert a fluid block to tensorrt network, NOTE it just convert
  // operators, the INetwork's inputs and outputs should specified in some
  // other modules.
  void ConvertBlock(const framework::proto::BlockDesc& block,
                    const std::unordered_set<std::string>& parameters,
                    const framework::Scope& scope,
                    TensorRTEngine* engine) {
    VLOG(1) << "Convert a fluid block to tensorrt network";
    std::unique_lock<std::mutex> lk(mut_);
    for (int i = 0; i < block.ops_size(); i++) {
      const auto& op = block.ops(i);
      ConvertOp(op, parameters, scope, engine, false, &block);
    }
    for (int i = 0; i < engine->network()->getNbLayers(); i++) {
      auto layer = engine->network()->getLayer(i);
      if (layer->getType() == nvinfer1::LayerType::kSHUFFLE) {
        auto* input_tensor = layer->getInput(0);
        auto* output_tensor = layer->getOutput(0);
        auto output_tensor_name = output_tensor->getName();
        auto input_tensor_name = input_tensor->getName();
        if (engine->DynamicRangeIsSet(input_tensor) &&
            !engine->DynamicRangeIsSet(output_tensor)) {
          float output_scale = engine->GetTensorDynamicRange(input_tensor);
          VLOG(1) << "Set output tensor scale = " << output_scale
                  << " for tensor in TensorRT: " << output_tensor_name << ".";
          engine->SetTensorDynamicRange(output_tensor, output_scale);
        } else {
          VLOG(1) << "Failed to get input tensor scale for tensor in TensorRT: "
                  << input_tensor_name << ".";
        }
      }
    }
  }

  // The scope here should be inited with the parameter vars.
  void ConvertBlockToTRTEngine(
      framework::BlockDesc* block_desc,
      const framework::Scope& scope,
      const std::vector<std::string>& inputs,
      const std::unordered_set<std::string>& parameters,
      const std::vector<std::string>& outputs,
      TensorRTEngine* engine) {
    engine->InitNetwork();
    for (auto input : inputs) {
      if (parameters.count(input)) continue;
      // NOTE(liuyuanle): It is a trick. If you need a name [input], then you
      // need to use [input.substr(0, idx)].
      // Maybe we insert suffix of "_cast_auto_mixed.tmp_" in
      // auto_mixed_precision_pass.
      auto idx = input.find("_cast_auto_mixed.tmp_");
      input = input.substr(0, idx);

      auto* var = block_desc->FindVar(input);
      PADDLE_ENFORCE_NOT_NULL(
          var,
          common::errors::NotFound("no variable called %s in block.",
                                   input.c_str()));
      PADDLE_ENFORCE_EQ(
          var->GetType(),
          FluidDT::VarType_Type_DENSE_TENSOR,
          common::errors::InvalidArgument("TensorRT engine only takes "
                                          "DenseTensor as input"));
      nvinfer1::DataType in_dtype = FluidDataType2TRT(var->GetDataType());
      if (engine->precision() == phi::DataType::FLOAT16 &&
          in_dtype == nvinfer1::DataType::kFLOAT &&
          engine->LowPrecisionIOEnabled()) {
        in_dtype = nvinfer1::DataType::kHALF;
      }

      auto var_shape = var->GetShape();
      if (engine->with_dynamic_shape()) {
#if IS_TRT_VERSION_GE(6000)
        if (!(engine->min_input_shape().count(input) &&
              engine->max_input_shape().count(input) &&
              engine->optim_input_shape().count(input))) {
          PADDLE_THROW(common::errors::InvalidArgument(
              "Cannot get %s min/max/opt shape", input));
        }
        auto min_input_shape = engine->min_input_shape().at(input);
        auto max_input_shape = engine->max_input_shape().at(input);
        auto optim_input_shape = engine->optim_input_shape().at(input);
        size_t ranks = min_input_shape.size();

        std::vector<int64_t> input_shape;
        // input_shape.push_back(-1);
        for (size_t i = 0; i < ranks; i++) {
          if (min_input_shape[i] != max_input_shape[i]) {
            input_shape.push_back(-1);
          } else {
            input_shape.push_back(min_input_shape[i]);
            // the i dimension should be same.
            PADDLE_ENFORCE_EQ(min_input_shape[i],
                              optim_input_shape[i],
                              common::errors::InvalidArgument(
                                  "The dim (%d) of the min_input_shape and "
                                  "optim_input_shape should be same."));
          }
        }
        engine->DeclareInput(
            input, in_dtype, Vec2TRT_Dims(input_shape, input, true));
#endif
      } else {
        auto input_dims = Vec2TRT_Dims(var_shape, input);
        if (input_dims.d[0] == -1) {
          input_dims.d[0] = engine->get_max_batch_size();
        }
        engine->DeclareInput(input, in_dtype, input_dims);
      }
      VLOG(1) << "set trt engine input dtype " << static_cast<int>(in_dtype);
    }

    framework::proto::BlockDesc* block_proto = block_desc->Proto();
    ConvertBlock(*block_proto, parameters, scope, engine);

    for (auto& output : outputs) {
      auto* var = block_desc->FindVar(output);
      PADDLE_ENFORCE_NOT_NULL(
          var,
          common::errors::NotFound("no variable called %s in block.",
                                   output.c_str()));
      PADDLE_ENFORCE_EQ(
          var->GetType(),
          FluidDT::VarType_Type_DENSE_TENSOR,
          common::errors::InvalidArgument(
              "The output tensor in TensorRT subgraph should be DenseTensor"));
      nvinfer1::DataType out_dtype = FluidDataType2TRT(var->GetDataType());
      if (engine->precision() == phi::DataType::FLOAT16 &&
          out_dtype == nvinfer1::DataType::kFLOAT &&
          engine->LowPrecisionIOEnabled()) {
        out_dtype = nvinfer1::DataType::kHALF;
      }
      engine->DeclareOutput(output, out_dtype);
      VLOG(1) << "set trt engine output dtype " << static_cast<int>(out_dtype);
    }

    engine->FreezeNetwork();
    engine->ClearWeights();
  }

  void SupportFP32MixPrecision(const std::string& output_name,
                               const std::string& op_type,
                               nvinfer1::ILayer* layer) {
    if (engine_->OpIsRunFloat(output_name) || engine_->OpIsRunFloat(op_type)) {
#if IS_TRT_VERSION_GE(8210)
      VLOG(3) << op_type << "(output: " << output_name << ")"
              << " is forced to run in FP32 precision.";
      layer->resetPrecision();
      layer->setPrecision(nvinfer1::DataType::kFLOAT);
#else
      VLOG(3)
          << op_type << "(output: " << output_name << ")"
          << ": Set layer precision needs TensorRT version 8.2.1 and after.";
#endif
    }
  }

  nvinfer1::ITensor* Cast(nvinfer1::ITensor* input, nvinfer1::DataType dtype) {
    auto* layer = TRT_ENGINE_ADD_LAYER(engine_, Identity, *input);
    layer->setOutputType(0, dtype);
    layer->getOutput(0)->setType(dtype);
    return layer->getOutput(0);
  }

  // rank(result) = rank(input)
  nvinfer1::ITensor* Gather(nvinfer1::ITensor* input,
                            const std::vector<int32_t> indices,
                            int axis = 0) {
    auto* indices_tensor = Add1DConstantLayer(indices, " ");
    auto* result =
        TRT_ENGINE_ADD_LAYER(engine_, Gather, *input, *indices_tensor, axis)
            ->getOutput(0);
    return result;
  }

  nvinfer1::ITensor* Unsqueeze(nvinfer1::ITensor* input,
                               const std::vector<int32_t> axis) {
    const auto dims = input->getDimensions();
    const std::unordered_set<int32_t> axis_data(axis.begin(), axis.end());
    std::vector<int32_t> subscripts(dims.nbDims);
    std::iota(subscripts.begin(), subscripts.end(), 0);
    for (const auto& axis_value : axis_data) {
      subscripts.insert(subscripts.begin() + axis_value, dims.nbDims);
    }
    nvinfer1::ITensor* input_shape{nullptr};
    input_shape = Shape(input);
    auto* new_dim =
        TRT_ENGINE_ADD_LAYER(engine_,
                             Gather,
                             *Concat(std::vector<nvinfer1::ITensor*>{
                                 input_shape, Add1DConstantLayer(1)}),
                             *Add1DConstantLayer(subscripts),
                             0)
            ->getOutput(0);
    auto result = Reshape(input, new_dim);
    return result;
  }

  nvinfer1::ITensor* Squeeze(nvinfer1::ITensor* input,
                             const std::vector<int32_t> axis) {
    const auto dims = input->getDimensions();
    std::vector<int32_t> subscripts(dims.nbDims);
    std::iota(subscripts.begin(), subscripts.end(), 0);
    auto p =
        std::remove_if(subscripts.begin(), subscripts.end(), [axis](int x) {
          return std::find(axis.begin(), axis.end(), x) != axis.end();
        });
    subscripts.resize(p - subscripts.begin());

    nvinfer1::ITensor* input_shape{nullptr};
    input_shape = Shape(input);

    auto* new_dim =
        TRT_ENGINE_ADD_LAYER(
            engine_, Gather, *input_shape, *Add1DConstantLayer(subscripts), 0)
            ->getOutput(0);
    auto result = Reshape(input, new_dim);
    return result;
  }

  // paddle allows negative index
  // for axis length = 5, paddle allows [-5, 4]
  nvinfer1::ITensor* FixNegIndices(nvinfer1::ITensor* input_shape,
                                   nvinfer1::ITensor* indices) {
    int rank = input_shape->getDimensions().nbDims;
    std::vector<int32_t> zero = std::vector<int32_t>(rank, 0);
    std::vector<int32_t> minus_one = std::vector<int32_t>(rank, -1);
    nvinfer1::ITensor* zero_tensor = Add1DConstantLayer(zero);
    nvinfer1::ITensor* minus_one_tensor = Add1DConstantLayer(minus_one);
    // -1, 0
    auto* sign = Max(Min(indices, zero_tensor), minus_one_tensor);
    return Sub(indices, Prod(sign, input_shape));
  }

  nvinfer1::ITensor* Shape(nvinfer1::ITensor* input) {
    auto* shape_tensor =
        TRT_ENGINE_ADD_LAYER(engine_, Shape, *input)->getOutput(0);
#if IS_TRT_VERSION_GE(10000)
    return Cast(shape_tensor, nvinfer1::DataType::kINT32);
#else
    return shape_tensor;
#endif
  }

  nvinfer1::ITensor* Reshape(nvinfer1::ITensor* input,
                             nvinfer1::ITensor* newShape,
                             const std::string& name = "") {
    auto* shuffle = TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *input);
    shuffle->setInput(1, *newShape);
    if (!name.empty()) {
      shuffle->setName(name.c_str());
    }
    return shuffle->getOutput(0);
  }

  nvinfer1::ITensor* Reshape(nvinfer1::ITensor* input,
                             nvinfer1::Dims shape,
                             const std::string& name = "") {
    auto* shuffle = TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *input);
    shuffle->setReshapeDimensions(shape);
    if (!name.empty()) {
      shuffle->setName(name.c_str());
    }
    return shuffle->getOutput(0);
  }

  nvinfer1::ITensor* BroadcastTensor(nvinfer1::ITensor* input,
                                     const int nbDims,
                                     const std::string& name = "") {
    auto oldShape = Shape(input);
    auto oldShapeDims = oldShape->getDimensions();
    const int rank = oldShapeDims.nbDims;
    if (rank > nbDims) {
      PADDLE_THROW(common::errors::InvalidArgument(
          "Cannot broadcast a higher rank tensor to a lower rank tensor."));
    }
    if (rank < nbDims) {
      nvinfer1::ITensor* concat_shape_tensor;
      auto* one_rank_tensor =
          Add1DConstantLayer(std::vector<int32_t>(nbDims - rank, 1));
      std::vector<nvinfer1::ITensor*> itensors;
      itensors.push_back(one_rank_tensor);
      itensors.push_back(oldShape);
      concat_shape_tensor = Concat(itensors);
      input = Reshape(input, concat_shape_tensor, name);
    }
    return input;
  }

  nvinfer1::ITensor* BroadcastTensors(nvinfer1::ITensor* a,
                                      nvinfer1::ITensor* b,
                                      const std::string& name = "") {
    const int aDims = a->getDimensions().nbDims;
    const int bDims = b->getDimensions().nbDims;
    if (aDims == bDims) {
      VLOG(3) << "Broadcast two equal rank tensors";
    }
    if (aDims > bDims) {
      return BroadcastTensor(b, aDims, name);
    }
    return BroadcastTensor(a, bDims, name);
  }

  // Concat not make rank changed
  nvinfer1::ITensor* Concat(const std::vector<nvinfer1::ITensor*>& inputs,
                            int axis = 0) {
    auto* layer = TRT_ENGINE_ADD_LAYER(
        engine_, Concatenation, inputs.data(), inputs.size());
    if (axis != 0) layer->setAxis(axis);
    nvinfer1::ITensor* c = layer->getOutput(0);
    return c;
  }

  nvinfer1::ITensor* Sum(nvinfer1::ITensor* a, nvinfer1::ITensor* b) {
    nvinfer1::ITensor* c =
        TRT_ENGINE_ADD_LAYER(
            engine_, ElementWise, *a, *b, nvinfer1::ElementWiseOperation::kSUM)
            ->getOutput(0);
    return c;
  }

  nvinfer1::ITensor* Prod(nvinfer1::ITensor* a, nvinfer1::ITensor* b) {
    nvinfer1::ITensor* c =
        TRT_ENGINE_ADD_LAYER(
            engine_, ElementWise, *a, *b, nvinfer1::ElementWiseOperation::kPROD)
            ->getOutput(0);
    return c;
  }

  nvinfer1::ITensor* Min(nvinfer1::ITensor* a, nvinfer1::ITensor* b) {
    nvinfer1::ITensor* c =
        TRT_ENGINE_ADD_LAYER(
            engine_, ElementWise, *a, *b, nvinfer1::ElementWiseOperation::kMIN)
            ->getOutput(0);
    return c;
  }

  nvinfer1::ITensor* Max(nvinfer1::ITensor* a, nvinfer1::ITensor* b) {
    nvinfer1::ITensor* c =
        TRT_ENGINE_ADD_LAYER(
            engine_, ElementWise, *a, *b, nvinfer1::ElementWiseOperation::kMAX)
            ->getOutput(0);
    return c;
  }

  nvinfer1::ITensor* Sub(nvinfer1::ITensor* a, nvinfer1::ITensor* b) {
    nvinfer1::ITensor* c =
        TRT_ENGINE_ADD_LAYER(
            engine_, ElementWise, *a, *b, nvinfer1::ElementWiseOperation::kSUB)
            ->getOutput(0);
    return c;
  }

  nvinfer1::ITensor* Div(nvinfer1::ITensor* a, nvinfer1::ITensor* b) {
    nvinfer1::ITensor* c =
        TRT_ENGINE_ADD_LAYER(
            engine_, ElementWise, *a, *b, nvinfer1::ElementWiseOperation::kDIV)
            ->getOutput(0);
    return c;
  }

  nvinfer1::ITensor* FloorDiv(nvinfer1::ITensor* a, nvinfer1::ITensor* b) {
    nvinfer1::ITensor* c =
        TRT_ENGINE_ADD_LAYER(engine_,
                             ElementWise,
                             *a,
                             *b,
                             nvinfer1::ElementWiseOperation::kFLOOR_DIV)
            ->getOutput(0);
    return c;
  }

  nvinfer1::ITensor* Pow(nvinfer1::ITensor* a, nvinfer1::ITensor* b) {
    nvinfer1::ITensor* c =
        TRT_ENGINE_ADD_LAYER(
            engine_, ElementWise, *a, *b, nvinfer1::ElementWiseOperation::kPOW)
            ->getOutput(0);
    return c;
  }

  nvinfer1::ITensor* Act(nvinfer1::ITensor* a,
                         nvinfer1::ActivationType act_type) {
    nvinfer1::ITensor* c =
        TRT_ENGINE_ADD_LAYER(engine_, Activation, *a, act_type)->getOutput(0);
    return c;
  }

  // Get element tensor of 1D shape tensor
  nvinfer1::ITensor* GetEleTensorOfShape(nvinfer1::ITensor* shape_tensor,
                                         int index,
                                         bool is_scalar = false) {
    PADDLE_ENFORCE_GE(
        index,
        0,
        common::errors::PreconditionNotMet(
            "The index should be greater or equal than 0, but got %d", index));

    auto* tensor =
        TRT_ENGINE_ADD_LAYER(engine_,
                             Gather,
                             *shape_tensor,
                             *Add1DConstantLayer(index, " ", is_scalar),
                             0)
            ->getOutput(0);
    return tensor;
  }

  // Create an constant layer with shape_tensor and value
  template <typename T>
  nvinfer1::ITensor* FillConstantLayer(nvinfer1::ITensor* shape_tensor,
                                       int tensor_rank,
                                       T value) {
    auto fill_layer = TRT_ENGINE_ADD_LAYER(
        engine_, Fill, nvinfer1::Dims{}, nvinfer1::FillOperation::kLINSPACE);
    fill_layer->setInput(0, *shape_tensor);
    std::vector<T> beta_vec(tensor_rank);
    std::vector<T> value_vec(1, value);
    fill_layer->setInput(1, *Add1DConstantLayer(value_vec, "value_vec", true));
    fill_layer->setInput(2, *Add1DConstantLayer(beta_vec, "beta_vec", false));
    auto tensor = fill_layer->getOutput(0);
    return tensor;
  }

  template <typename T>
  // Create and add Multi-D constant float/int32 layer
  nvinfer1::ITensor* AddConstantLayer(const T* data,
                                      nvinfer1::Dims shape,
                                      const std::string& weight_name = "") {
    if (!(std::is_same<T, float>::value ||
          std::is_same<T, phi::dtype::float16>::value ||
          std::is_same<T, int32_t>::value)) {
      PADDLE_THROW(common::errors::InvalidArgument(
          "Unsupported data type (%s) for TensorRT AddConstantLayer, only "
          "supports float, half or int32_t."));
    }

    int data_size = std::accumulate(
        shape.d, shape.d + shape.nbDims, 1, std::multiplies<int>());
    std::unique_ptr<phi::DenseTensor> tmp_tensor(new phi::DenseTensor());
    tmp_tensor->Resize({data_size});
    auto* tmp_data = tmp_tensor->mutable_data<T>(phi::CPUPlace());
    for (int i = 0; i < data_size; i++) {
      tmp_data[i] = data[i];
    }
    engine_->SetWeights(weight_name, std::move(tmp_tensor));

    nvinfer1::DataType trt_dtype = nvinfer1::DataType::kFLOAT;
    if (std::is_integral<T>::value) {
      trt_dtype = nvinfer1::DataType::kINT32;
    }

    TensorRTEngine::Weight weight{trt_dtype,
                                  static_cast<void*>(tmp_data),
                                  static_cast<size_t>(data_size)};

    auto const_layer =
        TRT_ENGINE_ADD_LAYER(engine_, Constant, shape, weight.get());
    return const_layer->getOutput(0);
  }

  // Create and add 1D constant float/int32 layer
  template <typename T>
  nvinfer1::ITensor* Add1DConstantLayer(const std::vector<T>& data,
                                        const std::string& weight_name = "",
                                        bool scalar = false) {
    if (!(std::is_same<T, float>::value ||
          std::is_same<T, phi::dtype::float16>::value ||
          std::is_same<T, int32_t>::value)) {
      PADDLE_THROW(common::errors::InvalidArgument(
          "Unsupported data type (%s) for TensorRT AddConstantLayer, only "
          "supports float, half or int32_t."));
    }

    std::unique_ptr<phi::DenseTensor> tmp_tensor(new phi::DenseTensor());
    int data_size = data.size();
    tmp_tensor->Resize({data_size});
    auto* tmp_data = tmp_tensor->mutable_data<T>(phi::CPUPlace());
    for (int i = 0; i < data_size; i++) {
      tmp_data[i] = data[i];
    }
    engine_->SetWeights(weight_name, std::move(tmp_tensor));

    nvinfer1::DataType trt_dtype = nvinfer1::DataType::kFLOAT;
    if (std::is_integral<T>::value) {
      trt_dtype = nvinfer1::DataType::kINT32;
    }

    TensorRTEngine::Weight weight{trt_dtype,
                                  static_cast<void*>(tmp_data),
                                  static_cast<size_t>(data_size)};
    nvinfer1::Dims input_shape;
    input_shape.nbDims = scalar ? 0 : 1;
    input_shape.d[0] = data_size;
    auto const_layer =
        TRT_ENGINE_ADD_LAYER(engine_, Constant, input_shape, weight.get());
    return const_layer->getOutput(0);
  }

  nvinfer1::ITensor* Add1DConstantLayer(nvinfer1::Dims data,
                                        const std::string& weight_name = "",
                                        bool scalar = false) {
    std::vector<int> tmp_data;
    for (int i = 0; i < data.nbDims; i++) tmp_data.push_back(data.d[i]);
    return Add1DConstantLayer(tmp_data, weight_name, scalar);
  }

  template <typename T>
  nvinfer1::ITensor* Add1DConstantLayer(T data,
                                        const std::string& weight_name = "",
                                        bool scalar = false) {
    std::vector<T> input_data;
    input_data.push_back(data);
    return Add1DConstantLayer(input_data, weight_name, scalar);
  }

  void ReplenishLayerAndOutput(
      nvinfer1::ILayer* layer,
      const std::string& layer_type,
      const std::vector<std::string>& output_tensor_names,
      bool test_mode = false) {
    if (layer == nullptr) {
      return;
    }
    size_t num_out = output_tensor_names.size();
    std::string layer_name = layer_type + " (Output: ";
    for (size_t i = 0; i < num_out; i++) {
      layer->getOutput(i)->setName(output_tensor_names[i].c_str());
      engine_->SetITensor(output_tensor_names[i], layer->getOutput(i));
      if (test_mode) {
        engine_->DeclareOutput(output_tensor_names[i]);
      }
      layer_name += output_tensor_names[i];
      if (i != num_out - 1) layer_name += ", ";
    }
    for (size_t i = 0; i < num_out; i++) {
      nvinfer1::Dims tmp_dims = layer->getOutput(i)->getDimensions();
      std::vector<int> tmp_vec;
      for (int i = 0; i < tmp_dims.nbDims; i++)
        tmp_vec.push_back(tmp_dims.d[i]);

      VLOG(3) << output_tensor_names[i] << "'s dimension :["
              << string::join_strings(tmp_vec, ',') << "]";
      VLOG(1) << "Paddle-TRT inferred " << output_tensor_names[i]
              << "'s dimension is :[" << string::join_strings(tmp_vec, ',')
              << "]";
      // The following check may cause errors in CI, but is necessary in the
      // latest version.
      // PADDLE_ENFORCE_GE(
      //     layer->getOutput(i)->getDimensions().nbDims,
      //     0,
      //     common::errors::InvalidArgument(
      //         "Error occurred in Paddle-TRT layer with output name: %s",
      //         output_tensor_names[i].c_str()));
    }
    layer->setName((layer_name + ")").c_str());
  }
  void SetEngine(TensorRTEngine* engine) { engine_ = engine; }

  void SetBlockDesc(const framework::proto::BlockDesc* block) {
    block_ = block;
  }

  virtual ~OpConverter() {}

  // TensorRT engine
  TensorRTEngine* engine_{nullptr};
  // BlockDesc
  const framework::proto::BlockDesc* block_{nullptr};

 protected:
  bool test_mode_;

 private:
  std::mutex mut_;
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

#define REGISTER_TRT_OP_CONVERTER(op_type__, Converter__)                      \
  struct trt_##op_type__##_converter : public ::paddle::framework::Registrar { \
    trt_##op_type__##_converter() {                                            \
      ::paddle::inference::Registry<                                           \
          paddle::inference::tensorrt::OpConverter>::Global()                  \
          .Register<::paddle::inference::tensorrt::Converter__>(#op_type__);   \
    }                                                                          \
  };                                                                           \
  trt_##op_type__##_converter trt_##op_type__##_converter__;                   \
  int TouchConverterRegister_##op_type__() {                                   \
    trt_##op_type__##_converter__.Touch();                                     \
    return 0;                                                                  \
  }

#define USE_TRT_CONVERTER(op_type__)                   \
  extern int TouchConverterRegister_##op_type__();     \
  static int use_op_converter_trt_##op_type__ UNUSED = \
      TouchConverterRegister_##op_type__();
