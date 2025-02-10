// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, softwarepool
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/inference/tensorrt/plugin/pool3d_op_plugin.h"
#include "paddle/phi/kernels/funcs/pooling.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

size_t Pool3DPlugin::getSerializationSize() const TRT_NOEXCEPT {
  return getBaseSerializationSize() + SerializedSize(ceil_mode_) +
         SerializedSize(pool3d_type_) + SerializedSize(adaptive_) +
         SerializedSize(ksize_) + SerializedSize(strides_) +
         SerializedSize(paddings_) + SerializedSize(input_shape_) +
         SerializedSize(output_shape_);
}

// TRT will call this func when we need to serialize the configuration of
// tensorrt.
void Pool3DPlugin::serialize(void *buffer) const TRT_NOEXCEPT {
  serializeBase(buffer);
  SerializeValue(&buffer, ceil_mode_);
  SerializeValue(&buffer, pool3d_type_);
  SerializeValue(&buffer, adaptive_);
  SerializeValue(&buffer, ksize_);
  SerializeValue(&buffer, strides_);
  SerializeValue(&buffer, paddings_);
  SerializeValue(&buffer, input_shape_);
  SerializeValue(&buffer, output_shape_);
}

Pool3DPlugin *Pool3DPlugin::clone() const TRT_NOEXCEPT {
  return new Pool3DPlugin(ceil_mode_,
                          pool3d_type_,
                          adaptive_,
                          ksize_,
                          strides_,
                          paddings_,
                          input_shape_);
}

const char *Pool3DPlugin::getPluginType() const TRT_NOEXCEPT {
  return "pool3d_plugin";
}

int Pool3DPlugin::getNbOutputs() const TRT_NOEXCEPT { return 1; }

int Pool3DPlugin::initialize() TRT_NOEXCEPT { return 0; }

nvinfer1::DataType Pool3DPlugin::getOutputDataType(
    int index,
    const nvinfer1::DataType *input_types,
    int nb_inputs) const TRT_NOEXCEPT {
  return input_types[0];
}

void Pool3DPlugin::destroy() TRT_NOEXCEPT { delete this; }

nvinfer1::Dims Pool3DPlugin::getOutputDimensions(
    int index, const nvinfer1::Dims *inputDims, int nbInputs) TRT_NOEXCEPT {
  PADDLE_ENFORCE_EQ(nbInputs,
                    1,
                    common::errors::InvalidArgument(
                        "The Pool3D Plugin only has one input, so the nbInputs "
                        "value should be 1, but get %d.",
                        nbInputs));
  PADDLE_ENFORCE_EQ(index,
                    0,
                    common::errors::InvalidArgument(
                        "The Pool3D Plugin only has one input, so "
                        "the index value should be 0, but get %d.",
                        index));
  PADDLE_ENFORCE_EQ(inputDims[0].nbDims,
                    4,
                    common::errors::InvalidArgument(
                        "The Pool3D Plugin only has four Dimensions, so the "
                        "nbDims value should be 4, but get %d.",
                        inputDims[0].nbDims));

  nvinfer1::Dims const &input_dims = inputDims[0];

  nvinfer1::Dims output_dims = input_dims;

  output_dims.d[1] = output_shape_[1];
  output_dims.d[2] = output_shape_[2];
  output_dims.d[3] = output_shape_[3];
  return output_dims;
}

int Pool3DPlugin::enqueue(int batchSize,
                          const void *const *inputs,
#if IS_TRT_VERSION_LT(8000)
                          void **outputs,
                          void *workspace,
                          cudaStream_t stream) TRT_NOEXCEPT {
#else
                          void *const *outputs,
                          void *workspace,
                          cudaStream_t stream) TRT_NOEXCEPT {
#endif
  int input_size = 0;
  float const *idata = reinterpret_cast<float const *>(inputs[0]);
  float *const *odatas = reinterpret_cast<float *const *>(outputs);

  std::vector<int> input_shape = input_shape_;
  std::vector<int> output_shape = output_shape_;
  input_shape.insert(input_shape.begin(), batchSize);
  output_shape.insert(output_shape.begin(), batchSize);

  if (pool3d_type_ == Pool3DType::max) {
    phi::funcs::MaxPool<float> pool_process;
    phi::funcs::Pool3dDirectCUDAFunctor<phi::funcs::MaxPool<float>, float>
        pool3d_forward;
    pool3d_forward(idata,
                   input_shape,
                   output_shape,
                   ksize_,
                   strides_,
                   paddings_,
                   true,
                   adaptive_,
                   odatas[0],
                   stream,
                   pool_process);
  } else if (pool3d_type_ == Pool3DType::avg) {
    phi::funcs::AvgPool<float> pool_process;
    phi::funcs::Pool3dDirectCUDAFunctor<phi::funcs::AvgPool<float>, float>
        pool3d_forward;
    pool3d_forward(idata,
                   input_shape,
                   output_shape,
                   ksize_,
                   strides_,
                   paddings_,
                   true,
                   adaptive_,
                   odatas[0],
                   stream,
                   pool_process);
  }

  return cudaGetLastError() != cudaSuccess;
}

// Dynamic Plugin below.

Pool3DPluginDynamic::Pool3DPluginDynamic(void const *serialData,
                                         size_t serialLength) {
  DeserializeValue(&serialData, &serialLength, &ceil_mode_);
  const char *pool3d_type;
  DeserializeValue(&serialData, &serialLength, &pool3d_type);
  pool3d_type_ = std::string(pool3d_type);
  DeserializeValue(&serialData, &serialLength, &adaptive_);
  DeserializeValue(&serialData, &serialLength, &ksize_);
  DeserializeValue(&serialData, &serialLength, &strides_);
  DeserializeValue(&serialData, &serialLength, &paddings_);
  DeserializeValue(&serialData, &serialLength, &is_global_);
}

nvinfer1::IPluginV2DynamicExt *Pool3DPluginDynamic::clone() const TRT_NOEXCEPT {
  return new Pool3DPluginDynamic(ceil_mode_,
                                 pool3d_type_,
                                 adaptive_,
                                 ksize_,
                                 strides_,
                                 paddings_,
                                 is_global_);
}

const char *Pool3DPluginDynamic::getPluginType() const TRT_NOEXCEPT {
  return "pool3d_plugin_dynamic";
}
int Pool3DPluginDynamic::getNbOutputs() const TRT_NOEXCEPT { return 1; }

int Pool3DPluginDynamic::initialize() TRT_NOEXCEPT { return 0; }

void Pool3DPluginDynamic::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc *in,
    int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc *out,
    int nbOutputs) TRT_NOEXCEPT {}

size_t Pool3DPluginDynamic::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc *inputs,
    int nbInputs,
    const nvinfer1::PluginTensorDesc *outputs,
    int nbOutputs) const TRT_NOEXCEPT {
  return 0;
}

size_t Pool3DPluginDynamic::getSerializationSize() const TRT_NOEXCEPT {
  return SerializedSize(ceil_mode_) + SerializedSize(pool3d_type_.c_str()) +
         SerializedSize(adaptive_) + SerializedSize(ksize_) +
         SerializedSize(strides_) + SerializedSize(paddings_) +
         SerializedSize(is_global_);
}

void Pool3DPluginDynamic::serialize(void *buffer) const TRT_NOEXCEPT {
  SerializeValue(&buffer, ceil_mode_);
  SerializeValue(&buffer, pool3d_type_.c_str());
  SerializeValue(&buffer, adaptive_);
  SerializeValue(&buffer, ksize_);
  SerializeValue(&buffer, strides_);
  SerializeValue(&buffer, paddings_);
  SerializeValue(&buffer, is_global_);
}

nvinfer1::DimsExprs Pool3DPluginDynamic::getOutputDimensions(
    int output_index,
    const nvinfer1::DimsExprs *inputs,
    int nb_inputs,
    nvinfer1::IExprBuilder &expr_builder) TRT_NOEXCEPT {
  PADDLE_ENFORCE_EQ(nb_inputs,
                    1,
                    common::errors::InvalidArgument(
                        "The Split plugin should be only one input."));

  PADDLE_ENFORCE_EQ(
      inputs[0].d[1]->isConstant(),
      true,
      common::errors::InvalidArgument("The channel dimension should be "
                                      "static, but we found it's dynamic."));
  nvinfer1::DimsExprs output(inputs[0]);
  if (is_global_) {
    output.d[2] = expr_builder.constant(1);
    output.d[3] = expr_builder.constant(1);
    output.d[4] = expr_builder.constant(1);
    return output;
  }
  if (adaptive_) {
    output.d[2] = expr_builder.constant(ksize_[0]);
    output.d[3] = expr_builder.constant(ksize_[1]);
    output.d[4] = expr_builder.constant(ksize_[2]);
    return output;
  }

  auto stri_0 = expr_builder.constant(strides_[0]);
  auto stri_1 = expr_builder.constant(strides_[1]);
  auto stri_2 = expr_builder.constant(strides_[2]);
  auto one_value = expr_builder.constant(1);

  auto v0_tmp = expr_builder.constant(-ksize_[0] + 2 * paddings_[0]);
  auto v1_tmp = expr_builder.constant(-ksize_[1] + 2 * paddings_[1]);
  auto v2_tmp = expr_builder.constant(-ksize_[2] + 2 * paddings_[2]);

  auto ceil_tmp =
      expr_builder.constant(-ksize_[0] + 2 * paddings_[0] + strides_[0] - 1);
  auto ceil1_tmp =
      expr_builder.constant(-ksize_[1] + 2 * paddings_[1] + strides_[1] - 1);
  auto ceil2_tmp =
      expr_builder.constant(-ksize_[2] + 2 * paddings_[2] + strides_[2] - 1);

  if (!ceil_mode_) {
    output.d[2] = expr_builder.operation(
        nvinfer1::DimensionOperation::kSUM,
        *expr_builder.operation(
            nvinfer1::DimensionOperation::kFLOOR_DIV,
            *expr_builder.operation(
                nvinfer1::DimensionOperation::kSUM, *inputs[0].d[2], *v0_tmp),
            *stri_0),
        *one_value);
    output.d[3] = expr_builder.operation(
        nvinfer1::DimensionOperation::kSUM,
        *expr_builder.operation(
            nvinfer1::DimensionOperation::kFLOOR_DIV,
            *expr_builder.operation(
                nvinfer1::DimensionOperation::kSUM, *inputs[0].d[3], *v1_tmp),
            *stri_1),
        *one_value);
    output.d[4] = expr_builder.operation(
        nvinfer1::DimensionOperation::kSUM,
        *expr_builder.operation(
            nvinfer1::DimensionOperation::kFLOOR_DIV,
            *expr_builder.operation(
                nvinfer1::DimensionOperation::kSUM, *inputs[0].d[4], *v2_tmp),
            *stri_2),
        *one_value);

  } else {
    output.d[2] = expr_builder.operation(
        nvinfer1::DimensionOperation::kSUM,
        *expr_builder.operation(
            nvinfer1::DimensionOperation::kFLOOR_DIV,
            *expr_builder.operation(
                nvinfer1::DimensionOperation::kSUM, *inputs[0].d[2], *ceil_tmp),
            *stri_0),
        *one_value);
    output.d[3] = expr_builder.operation(
        nvinfer1::DimensionOperation::kSUM,
        *expr_builder.operation(
            nvinfer1::DimensionOperation::kFLOOR_DIV,
            *expr_builder.operation(nvinfer1::DimensionOperation::kSUM,
                                    *inputs[0].d[3],
                                    *ceil1_tmp),
            *stri_1),
        *one_value);
    output.d[4] = expr_builder.operation(
        nvinfer1::DimensionOperation::kSUM,
        *expr_builder.operation(
            nvinfer1::DimensionOperation::kFLOOR_DIV,
            *expr_builder.operation(nvinfer1::DimensionOperation::kSUM,
                                    *inputs[0].d[4],
                                    *ceil2_tmp),
            *stri_2),
        *one_value);
  }

  return output;
}

bool Pool3DPluginDynamic::supportsFormatCombination(
    int pos,
    const nvinfer1::PluginTensorDesc *in_out,
    int nb_inputs,
    int nb_outputs) TRT_NOEXCEPT {
  PADDLE_ENFORCE_NOT_NULL(
      in_out,
      common::errors::InvalidArgument(
          "The input of swish plugin should not be nullptr."));

  PADDLE_ENFORCE_LT(
      pos,
      nb_inputs + nb_outputs,
      common::errors::InvalidArgument("The pos(%d) should be less than the "
                                      "num(%d) of the input and the output.",
                                      pos,
                                      nb_inputs + nb_outputs));
  (in_out && pos < (nb_inputs + nb_outputs));

  return ((in_out[pos].type == nvinfer1::DataType::kFLOAT) &&
          in_out[pos].format == nvinfer1::PluginFormat::kLINEAR);
}

nvinfer1::DataType Pool3DPluginDynamic::getOutputDataType(
    int index,
    const nvinfer1::DataType *input_types,
    int nb_inputs) const TRT_NOEXCEPT {
  PADDLE_ENFORCE_EQ(index,
                    0,
                    common::errors::InvalidArgument(
                        "The Pool3D Plugin only has one input, so the "
                        "index value should be 0, but get %d.",
                        index));
  PADDLE_ENFORCE_EQ(
      (input_types[0] == nvinfer1::DataType::kFLOAT),
      true,
      common::errors::InvalidArgument("The input type should be float"));
  return input_types[0];
}

int Pool3DPluginDynamic::enqueue(const nvinfer1::PluginTensorDesc *input_desc,
                                 const nvinfer1::PluginTensorDesc *output_desc,
                                 const void *const *inputs,
                                 void *const *outputs,
                                 void *workspace,
                                 cudaStream_t stream) TRT_NOEXCEPT {
  auto input_dims = input_desc[0].dims;
  int n = input_dims.d[0];
  int c = input_dims.d[1];
  int d = input_dims.d[2];
  int h = input_dims.d[3];
  int w = input_dims.d[4];

  const float *input = static_cast<const float *>(inputs[0]);
  float *output = static_cast<float *>(outputs[0]);

  std::vector<int> input_shape, output_shape;
  for (int i = 0; i < input_dims.nbDims; i++)
    input_shape.push_back(input_dims.d[i]);
  output_shape = input_shape;

  std::vector<int> ksize = ksize_;
  std::vector<int> paddings = paddings_;
  if (is_global_) {
    ksize[0] = d;
    ksize[1] = h;
    ksize[2] = w;
    paddings[0] = 0;
    paddings[1] = 0;
    paddings[2] = 0;
    output_shape[2] = 1;
    output_shape[3] = 1;
    output_shape[4] = 1;
  } else {
    auto data_dim = CalcOutputSize(
        {d, h, w}, ceil_mode_, adaptive_, ksize_, strides_, paddings_);
    output_shape[2] = data_dim[0];
    output_shape[3] = data_dim[1];
    output_shape[4] = data_dim[2];
  }

  if (pool3d_type_ == "max") {
    phi::funcs::MaxPool<float> pool_process;
    phi::funcs::Pool3dDirectCUDAFunctor<phi::funcs::MaxPool<float>, float>
        pool3d_forward;
    pool3d_forward(input,
                   input_shape,
                   output_shape,
                   ksize,
                   strides_,
                   paddings,
                   true,
                   adaptive_,
                   output,
                   stream,
                   pool_process);
  } else if (pool3d_type_ == "avg") {
    phi::funcs::AvgPool<float> pool_process;
    phi::funcs::Pool3dDirectCUDAFunctor<phi::funcs::AvgPool<float>, float>
        pool3d_forward;
    pool3d_forward(input,
                   input_shape,
                   output_shape,
                   ksize,
                   strides_,
                   paddings,
                   true,
                   adaptive_,
                   output,
                   stream,
                   pool_process);
  }

  return cudaGetLastError() != cudaSuccess;
}

PIRPool3DPluginDynamic::PIRPool3DPluginDynamic(void const *serialData,
                                               size_t serialLength) {
  DeserializeValue(&serialData, &serialLength, &ceil_mode_);
  const char *pool3d_type;
  DeserializeValue(&serialData, &serialLength, &pool3d_type);
  pool3d_type_ = std::string(pool3d_type);
  DeserializeValue(&serialData, &serialLength, &adaptive_);
  DeserializeValue(&serialData, &serialLength, &ksize_);
  DeserializeValue(&serialData, &serialLength, &strides_);
  DeserializeValue(&serialData, &serialLength, &paddings_);
  DeserializeValue(&serialData, &serialLength, &is_global_);
}

nvinfer1::IPluginV2DynamicExt *PIRPool3DPluginDynamic::clone() const
    TRT_NOEXCEPT {
  return new PIRPool3DPluginDynamic(ceil_mode_,
                                    pool3d_type_,
                                    adaptive_,
                                    ksize_,
                                    strides_,
                                    paddings_,
                                    is_global_);
}

const char *PIRPool3DPluginDynamic::getPluginType() const TRT_NOEXCEPT {
  return "pir_pool3d_plugin_dynamic";
}
int PIRPool3DPluginDynamic::getNbOutputs() const TRT_NOEXCEPT { return 1; }

int PIRPool3DPluginDynamic::initialize() TRT_NOEXCEPT { return 0; }

void PIRPool3DPluginDynamic::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc *in,
    int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc *out,
    int nbOutputs) TRT_NOEXCEPT {}

size_t PIRPool3DPluginDynamic::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc *inputs,
    int nbInputs,
    const nvinfer1::PluginTensorDesc *outputs,
    int nbOutputs) const TRT_NOEXCEPT {
  return 0;
}

size_t PIRPool3DPluginDynamic::getSerializationSize() const TRT_NOEXCEPT {
  return SerializedSize(ceil_mode_) + SerializedSize(pool3d_type_.c_str()) +
         SerializedSize(adaptive_) + SerializedSize(ksize_) +
         SerializedSize(strides_) + SerializedSize(paddings_) +
         SerializedSize(is_global_);
}

void PIRPool3DPluginDynamic::serialize(void *buffer) const TRT_NOEXCEPT {
  SerializeValue(&buffer, ceil_mode_);
  SerializeValue(&buffer, pool3d_type_.c_str());
  SerializeValue(&buffer, adaptive_);
  SerializeValue(&buffer, ksize_);
  SerializeValue(&buffer, strides_);
  SerializeValue(&buffer, paddings_);
  SerializeValue(&buffer, is_global_);
}

nvinfer1::DimsExprs PIRPool3DPluginDynamic::getOutputDimensions(
    int output_index,
    const nvinfer1::DimsExprs *inputs,
    int nb_inputs,
    nvinfer1::IExprBuilder &expr_builder) TRT_NOEXCEPT {
  PADDLE_ENFORCE_EQ(nb_inputs,
                    1,
                    common::errors::InvalidArgument(
                        "The Split plugin should be only one input."));

  PADDLE_ENFORCE_EQ(
      inputs[0].d[1]->isConstant(),
      true,
      common::errors::InvalidArgument("The channel dimension should be "
                                      "static, but we found it's dynamic."));
  nvinfer1::DimsExprs output(inputs[0]);
  if (is_global_) {
    output.d[2] = expr_builder.constant(1);
    output.d[3] = expr_builder.constant(1);
    output.d[4] = expr_builder.constant(1);
    return output;
  }
  if (adaptive_) {
    output.d[2] = expr_builder.constant(ksize_[0]);
    output.d[3] = expr_builder.constant(ksize_[1]);
    output.d[4] = expr_builder.constant(ksize_[2]);
    return output;
  }

  auto stri_0 = expr_builder.constant(strides_[0]);
  auto stri_1 = expr_builder.constant(strides_[1]);
  auto stri_2 = expr_builder.constant(strides_[2]);
  auto one_value = expr_builder.constant(1);

  auto v0_tmp = expr_builder.constant(-ksize_[0] + 2 * paddings_[0]);
  auto v1_tmp = expr_builder.constant(-ksize_[1] + 2 * paddings_[1]);
  auto v2_tmp = expr_builder.constant(-ksize_[2] + 2 * paddings_[2]);

  auto ceil_tmp =
      expr_builder.constant(-ksize_[0] + 2 * paddings_[0] + strides_[0] - 1);
  auto ceil1_tmp =
      expr_builder.constant(-ksize_[1] + 2 * paddings_[1] + strides_[1] - 1);
  auto ceil2_tmp =
      expr_builder.constant(-ksize_[2] + 2 * paddings_[2] + strides_[2] - 1);

  if (!ceil_mode_) {
    output.d[2] = expr_builder.operation(
        nvinfer1::DimensionOperation::kSUM,
        *expr_builder.operation(
            nvinfer1::DimensionOperation::kFLOOR_DIV,
            *expr_builder.operation(
                nvinfer1::DimensionOperation::kSUM, *inputs[0].d[2], *v0_tmp),
            *stri_0),
        *one_value);
    output.d[3] = expr_builder.operation(
        nvinfer1::DimensionOperation::kSUM,
        *expr_builder.operation(
            nvinfer1::DimensionOperation::kFLOOR_DIV,
            *expr_builder.operation(
                nvinfer1::DimensionOperation::kSUM, *inputs[0].d[3], *v1_tmp),
            *stri_1),
        *one_value);
    output.d[4] = expr_builder.operation(
        nvinfer1::DimensionOperation::kSUM,
        *expr_builder.operation(
            nvinfer1::DimensionOperation::kFLOOR_DIV,
            *expr_builder.operation(
                nvinfer1::DimensionOperation::kSUM, *inputs[0].d[4], *v2_tmp),
            *stri_2),
        *one_value);

  } else {
    output.d[2] = expr_builder.operation(
        nvinfer1::DimensionOperation::kSUM,
        *expr_builder.operation(
            nvinfer1::DimensionOperation::kFLOOR_DIV,
            *expr_builder.operation(
                nvinfer1::DimensionOperation::kSUM, *inputs[0].d[2], *ceil_tmp),
            *stri_0),
        *one_value);
    output.d[3] = expr_builder.operation(
        nvinfer1::DimensionOperation::kSUM,
        *expr_builder.operation(
            nvinfer1::DimensionOperation::kFLOOR_DIV,
            *expr_builder.operation(nvinfer1::DimensionOperation::kSUM,
                                    *inputs[0].d[3],
                                    *ceil1_tmp),
            *stri_1),
        *one_value);
    output.d[4] = expr_builder.operation(
        nvinfer1::DimensionOperation::kSUM,
        *expr_builder.operation(
            nvinfer1::DimensionOperation::kFLOOR_DIV,
            *expr_builder.operation(nvinfer1::DimensionOperation::kSUM,
                                    *inputs[0].d[4],
                                    *ceil2_tmp),
            *stri_2),
        *one_value);
  }

  return output;
}

bool PIRPool3DPluginDynamic::supportsFormatCombination(
    int pos,
    const nvinfer1::PluginTensorDesc *in_out,
    int nb_inputs,
    int nb_outputs) TRT_NOEXCEPT {
  PADDLE_ENFORCE_NOT_NULL(
      in_out,
      common::errors::InvalidArgument(
          "The input of swish plugin should not be nullptr."));

  PADDLE_ENFORCE_LT(
      pos,
      nb_inputs + nb_outputs,
      common::errors::InvalidArgument("The pos(%d) should be less than the "
                                      "num(%d) of the input and the output.",
                                      pos,
                                      nb_inputs + nb_outputs));
  (in_out && pos < (nb_inputs + nb_outputs));

  return ((in_out[pos].type == nvinfer1::DataType::kFLOAT) &&
          in_out[pos].format == nvinfer1::PluginFormat::kLINEAR);
}

nvinfer1::DataType PIRPool3DPluginDynamic::getOutputDataType(
    int index,
    const nvinfer1::DataType *input_types,
    int nb_inputs) const TRT_NOEXCEPT {
  PADDLE_ENFORCE_EQ(index,
                    0,
                    common::errors::InvalidArgument(
                        "The Pool3D Plugin only has one input, so the "
                        "index value should be 0, but get %d.",
                        index));
  PADDLE_ENFORCE_EQ(
      (input_types[0] == nvinfer1::DataType::kFLOAT),
      true,
      common::errors::InvalidArgument("The input type should be float"));
  return input_types[0];
}

int PIRPool3DPluginDynamic::enqueue(
    const nvinfer1::PluginTensorDesc *input_desc,
    const nvinfer1::PluginTensorDesc *output_desc,
    const void *const *inputs,
    void *const *outputs,
    void *workspace,
    cudaStream_t stream) TRT_NOEXCEPT {
  auto input_dims = input_desc[0].dims;
  int n = input_dims.d[0];
  int c = input_dims.d[1];
  int d = input_dims.d[2];
  int h = input_dims.d[3];
  int w = input_dims.d[4];

  const float *input = static_cast<const float *>(inputs[0]);
  float *output = static_cast<float *>(outputs[0]);

  std::vector<int> input_shape, output_shape;
  for (int i = 0; i < input_dims.nbDims; i++)
    input_shape.push_back(input_dims.d[i]);
  output_shape = input_shape;

  std::vector<int> ksize = ksize_;
  std::vector<int> paddings = paddings_;
  if (is_global_) {
    ksize[0] = d;
    ksize[1] = h;
    ksize[2] = w;
    paddings[0] = 0;
    paddings[1] = 0;
    paddings[2] = 0;
    output_shape[2] = 1;
    output_shape[3] = 1;
    output_shape[4] = 1;
  } else {
    auto data_dim = CalcOutputSize(
        {d, h, w}, ceil_mode_, adaptive_, ksize_, strides_, paddings_);
    output_shape[2] = data_dim[0];
    output_shape[3] = data_dim[1];
    output_shape[4] = data_dim[2];
  }

  if (pool3d_type_ == "max") {
    phi::funcs::MaxPool<float> pool_process;
    phi::funcs::Pool3dDirectCUDAFunctor<phi::funcs::MaxPool<float>, float>
        pool3d_forward;
    pool3d_forward(input,
                   input_shape,
                   output_shape,
                   ksize,
                   strides_,
                   paddings,
                   true,
                   adaptive_,
                   output,
                   stream,
                   pool_process);
  } else if (pool3d_type_ == "avg") {
    phi::funcs::AvgPool<float> pool_process;
    phi::funcs::Pool3dDirectCUDAFunctor<phi::funcs::AvgPool<float>, float>
        pool3d_forward;
    pool3d_forward(input,
                   input_shape,
                   output_shape,
                   ksize,
                   strides_,
                   paddings,
                   true,
                   adaptive_,
                   output,
                   stream,
                   pool_process);
  }

  return cudaGetLastError() != cudaSuccess;
}

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
