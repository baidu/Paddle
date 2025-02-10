/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

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

/*
 * Cumsum Op
 */
class CumsumOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
#if IS_TRT_VERSION_GE(7220)
    VLOG(3) << "convert a cumsum op to tensorrt layer";
    framework::OpDesc op_desc(op, nullptr);
    std::string input_x_name = op_desc.Input("X").front();
    std::string output_name = op_desc.Output("Out").front();
    auto* input_x_tensor = engine_->GetITensor(input_x_name);
    auto dims = input_x_tensor->getDimensions();
    auto rank = dims.nbDims;
    if (rank == 0) {
      nvinfer1::IShuffleLayer* layer =
          TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *input_x_tensor);
      nvinfer1::Dims cumsum_dim;
      cumsum_dim.nbDims = 0;
      cumsum_dim.d[0] = 0;
      if (op_desc.HasAttr("axis")) {
        cumsum_dim.nbDims = 1;
        cumsum_dim.d[0] = 1;
      }
      layer->setReshapeDimensions(cumsum_dim);
      ReplenishLayerAndOutput(layer, "cumsum", {output_name}, test_mode);
    } else {
      int axis = 0;
      if (op_desc.HasAttr("axis")) {
        axis = PADDLE_GET_CONST(int, op_desc.GetAttr("axis"));
        if (axis < 0) {
          axis += rank;
        }
      }

      // getAxisLength default is a scalar
      auto getAxisLength =
          [&](nvinfer1::ITensor* inpTensor, int axis, bool scalar = true) {
            auto dims = inpTensor->getDimensions();
            int d = dims.d[axis];
            if (d >= 0) {
              return Add1DConstantLayer(d, "", scalar);
            } else {
              nvinfer1::ITensor* inpShape = Shape(inpTensor);
              return GetEleTensorOfShape(inpShape, axis, scalar);
            }
          };

      // Create "inputSliced" tensor that is sliced on dimension[axis] to length
      // 1
      nvinfer1::Dims start;
      start.nbDims = rank;
      std::vector<int32_t> start_vec(rank, 0);
      std::fill(start.d, start.d + rank, 0);

      nvinfer1::Dims size;
      size.nbDims = rank;
      nvinfer1::Dims stride;
      stride.nbDims = rank;
      auto axisLength = getAxisLength(input_x_tensor, axis, false);

      auto starts_tensor =
          Add1DConstantLayer(start_vec, output_name + "_start_tensor_");
      auto sizes_tensor = axis == 0 ? Add1DConstantLayer(1)
                                    : getAxisLength(input_x_tensor, 0, false);
      auto strides_tensor = axis == 0 ? axisLength : Add1DConstantLayer(1);

      for (int i = 1; i < rank; i++) {
        if (i == axis) {
          std::vector<nvinfer1::ITensor*> strides_itensors = {strides_tensor,
                                                              axisLength};
          strides_tensor = Concat(strides_itensors);
          std::vector<nvinfer1::ITensor*> sizes_itensors = {
              sizes_tensor, Add1DConstantLayer(1)};
          sizes_tensor = Concat(sizes_itensors);
        } else {
          auto currLength = getAxisLength(input_x_tensor, i, false);
          std::vector<nvinfer1::ITensor*> strides_itensors = {
              strides_tensor, Add1DConstantLayer(1)};
          strides_tensor = Concat(strides_itensors);
          std::vector<nvinfer1::ITensor*> sizes_itensors = {sizes_tensor,
                                                            currLength};
          sizes_tensor = Concat(sizes_itensors);
        }
      }
      auto inputSliced = TRT_ENGINE_ADD_LAYER(
          engine_, Slice, *input_x_tensor, start, size, stride);
      inputSliced->setInput(1, *starts_tensor);
      inputSliced->setInput(2, *sizes_tensor);
      inputSliced->setInput(3, *strides_tensor);
      auto inputSliced_output = inputSliced->getOutput(0);

      // Scan through each slice across axis and add it to the running sum
      auto loop = TRT_ENGINE_ADD_LAYER(engine_, Loop);
      nvinfer1::ITensor* tripLimit = getAxisLength(input_x_tensor, axis);
      loop->addTripLimit(*tripLimit, nvinfer1::TripLimit::kCOUNT);
      auto iterator = loop->addIterator(*input_x_tensor, axis);
      auto data = iterator->getOutput(0);
      // Squeeze inputSliced down to same shape as `data`
      auto sliced_dims = inputSliced_output->getDimensions();
      std::vector<int32_t> subscripts(sliced_dims.nbDims);
      std::iota(subscripts.begin(), subscripts.end(), 0);
      auto p = std::remove_if(subscripts.begin(),
                              subscripts.end(),
                              [axis](int x) { return x == axis; });
      subscripts.resize(p - subscripts.begin());
      auto newDims = Gather(Shape(inputSliced_output), subscripts);
      inputSliced_output =
          Reshape(inputSliced_output,
                  newDims,
                  ("cumsum: reshape: (Output(" + output_name + ")").c_str());

      // create ZeroTensor
      std::vector<float> zero_vec{0.f};
      auto zero = Add1DConstantLayer(zero_vec);
      auto cast = TRT_ENGINE_ADD_LAYER(engine_, Identity, *zero);
      cast->setOutputType(0, inputSliced_output->getType());

      zero = TRT_ENGINE_ADD_LAYER(
                 engine_,
                 ElementWise,
                 *inputSliced_output,
                 *BroadcastTensors(cast->getOutput(0),
                                   inputSliced_output,
                                   ("cumsum: reshape_for_broadcast: (Output(" +
                                    output_name + ")")
                                       .c_str()),
                 nvinfer1::ElementWiseOperation::kPROD)
                 ->getOutput(0);
      auto runningSum = loop->addRecurrence(*zero);
      auto runningSumTensor = runningSum->getOutput(0);
      auto curSum = TRT_ENGINE_ADD_LAYER(engine_,
                                         ElementWise,
                                         *data,
                                         *runningSumTensor,
                                         nvinfer1::ElementWiseOperation::kSUM);
      runningSum->setInput(1, *curSum->getOutput(0));
      auto reverseFlag = nvinfer1::LoopOutput::kCONCATENATE;
      nvinfer1::ILoopOutputLayer* loopOut =
          loop->addLoopOutput(*curSum->getOutput(0), reverseFlag, axis);
      loopOut->setInput(1, *tripLimit);
      ReplenishLayerAndOutput(loopOut, "cumsum", {output_name}, test_mode);
    }
#else
    VLOG(3) << "Cumsum is not supported when TensorRT < 7.2.2";
#endif
  }
};

}  // namespace paddle::inference::tensorrt

REGISTER_TRT_OP_CONVERTER(cumsum, CumsumOpConverter);
