/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/platform/device/gpu/gpu_info.h"
#include "paddle/phi/core/platform/device_context.h"

#include "cub/cub.cuh"
#include "paddle/fluid/inference/tensorrt/plugin/transformer_input_output_convert_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

__global__ void remove_padding_kernel(const half* input0,
                                      const int32_t* input1,
                                      half* output) {
  int word_id = blockIdx.x * gridDim.y + blockIdx.y;
  int32_t sequence_length = input1[blockIdx.x + 1] - input1[blockIdx.x];
  if (blockIdx.y < sequence_length) {
    output[(input1[blockIdx.x] + blockIdx.y) * gridDim.z * blockDim.x +
           blockIdx.z * blockDim.x + threadIdx.x] =
        input0[word_id * gridDim.z * blockDim.x + blockIdx.z * blockDim.x +
               threadIdx.x];
  }
}

__global__ void recover_padding_kernel(const half* input0,
                                       const int32_t* input1,
                                       half* output) {
  int word_id = blockIdx.x * gridDim.y + blockIdx.y;
  int32_t sequence_length = input1[blockIdx.x + 1] - input1[blockIdx.x];
  if (blockIdx.y < sequence_length) {
    output[word_id * gridDim.z * blockDim.x + blockIdx.z * blockDim.x +
           threadIdx.x] =
        input0[(input1[blockIdx.x] + blockIdx.y) * gridDim.z * blockDim.x +
               blockIdx.z * blockDim.x + threadIdx.x];
  } else {
    output[word_id * gridDim.z * blockDim.x + blockIdx.z * blockDim.x +
           threadIdx.x] = 0;
  }
}

nvinfer1::DataType TransformerInputConvertPlugin::getOutputDataType(
    int index,
    const nvinfer1::DataType* input_types,
    int nb_inputs) const TRT_NOEXCEPT {
  if (index == 0) {  // new input
    return nvinfer1::DataType::kHALF;
  } else if (index == 1) {  // mask
    return nvinfer1::DataType::kHALF;
  } else if (index == 2) {  // pos id
    return nvinfer1::DataType::kINT32;
  } else if (index == 3) {  // max_seqlen_tensor
    return nvinfer1::DataType::kHALF;
  }
}

nvinfer1::DimsExprs TransformerInputConvertPlugin::getOutputDimensions(
    int outputIndex,
    const nvinfer1::DimsExprs* inputs,
    int nbInputs,
    nvinfer1::IExprBuilder& exprBuilder) TRT_NOEXCEPT {
  constexpr size_t threadsPerCta384 = 1 * 8 * 32;
  constexpr size_t xmmasM384 = 24;
  constexpr size_t packedMaskSize384 = xmmasM384 * threadsPerCta384;
  int32_t maskSize_ = packedMaskSize384;
  auto maskSize = exprBuilder.constant(maskSize_);
  auto fp16maskSize = exprBuilder.operation(
      nvinfer1::DimensionOperation::kPROD, *maskSize, *exprBuilder.constant(2));

  auto one = exprBuilder.constant(1);
  auto B = inputs[0].d[0];
  auto MaxLength = inputs[0].d[1];
  auto Hidden = inputs[0].d[2];

  nvinfer1::DimsExprs output_dims;
  if (outputIndex == 0) {  // new input
    output_dims.nbDims = 4;
    output_dims.d[0] = exprBuilder.operation(
        nvinfer1::DimensionOperation::kPROD, *B, *MaxLength);
    output_dims.d[1] = Hidden;
    output_dims.d[2] = exprBuilder.constant(1);
    output_dims.d[3] = exprBuilder.constant(1);
  } else if (outputIndex == 1) {  // mask
    output_dims.nbDims = 2;
    output_dims.d[0] = B;
    output_dims.d[1] = fp16maskSize;
  } else if (outputIndex == 2) {  // pos id
    output_dims.nbDims = 1;
    output_dims.d[0] =
        exprBuilder.operation(nvinfer1::DimensionOperation::kSUM, *B, *one);
  } else if (outputIndex == 3) {  // max_seqlen_tensor
    output_dims.nbDims = 1;
    output_dims.d[0] = MaxLength;
  }
  return output_dims;
}

bool TransformerInputConvertPlugin::supportsFormatCombination(
    int pos,
    const nvinfer1::PluginTensorDesc* inOut,
    int nbInputs,
    int nbOutputs) TRT_NOEXCEPT {
  PADDLE_ENFORCE_EQ(nbInputs,
                    2,
                    common::errors::InvalidArgument(
                        "TransformerInputConvertPlugin must have 2 inputs, "
                        "but got %d input(s). ",
                        nbInputs));
  PADDLE_ENFORCE_EQ(nbOutputs,
                    4,
                    common::errors::InvalidArgument(
                        "TransformerInputConvertPlugin must have 4 outputs, "
                        "but got %d output(s). ",
                        nbOutputs));
  if (pos == 0) {  //  input
    return inOut[pos].format == nvinfer1::TensorFormat::kLINEAR &&
           inOut[pos].type == nvinfer1::DataType::kHALF;
  } else if (pos == 1) {  //  reducesum_qk_bias
    return inOut[pos].format == nvinfer1::TensorFormat::kLINEAR &&
           inOut[pos].type == nvinfer1::DataType::kINT32;
  } else if (pos == 2) {  // new input
    return inOut[pos].format == nvinfer1::TensorFormat::kLINEAR &&
           inOut[pos].type == nvinfer1::DataType::kHALF;
  } else if (pos == 3) {  // mask
    return inOut[pos].format == nvinfer1::TensorFormat::kLINEAR &&
           inOut[pos].type == nvinfer1::DataType::kHALF;
  } else if (pos == 4) {  // pos id
    return inOut[pos].format == nvinfer1::TensorFormat::kLINEAR &&
           inOut[pos].type == nvinfer1::DataType::kINT32;
  } else if (pos == 5) {  // max_seqlen_tensor
    return inOut[pos].format == nvinfer1::TensorFormat::kLINEAR &&
           inOut[pos].type == nvinfer1::DataType::kHALF;
  }
}

void TransformerInputConvertPlugin::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc* inputs,
    int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* outputs,
    int nbOutputs) TRT_NOEXCEPT {}

void TransformerInputConvertPlugin::attachToContext(
    cudnnContext* cudnnContext,
    cublasContext* cublasContext,
    nvinfer1::IGpuAllocator* gpuAllocator) TRT_NOEXCEPT {}

void TransformerInputConvertPlugin::detachFromContext() TRT_NOEXCEPT {}

void TransformerInputConvertPlugin::terminate() TRT_NOEXCEPT {}

int TransformerInputConvertPlugin::enqueue(
    const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc,
    const void* const* inputs,
    void* const* outputs,
    void* workspace,
    cudaStream_t stream) TRT_NOEXCEPT {
  // input(no_varlen), reducesum_qk_bias, input(varlen), mask, pos_id,
  // max_seqlen_tensor
  const half* input0 = static_cast<const half*>(inputs[0]);  // input(no_varlen)
  const int32_t* input1 =
      static_cast<const int32_t*>(inputs[1]);            // reducesum_qk_bias
  half* output0 = static_cast<half*>(outputs[0]);        // input(varlen)
  int32_t* output2 = static_cast<int32_t*>(outputs[2]);  // pos_id
  const auto input0_desc = inputDesc[0];
  const int32_t B = input0_desc.dims.d[0];           // batches
  const int32_t MaxLength = input0_desc.dims.d[1];   // max token length
  const int32_t HiddenSize = input0_desc.dims.d[2];  // hidden size

  // Determine temporary device storage requirements
  size_t temp_storage_bytes = 0;
  cub::DeviceScan::ExclusiveSum(
      NULL, temp_storage_bytes, input1, output2, B + 1);
  // Allocate temporary storage
  phi::GPUPlace place(platform::GetCurrentDeviceId());
  auto d_temp_storage = phi::memory_utils::Alloc(place, temp_storage_bytes);
  // Run exclusive prefix sum
  cub::DeviceScan::ExclusiveSum(
      d_temp_storage->ptr(), temp_storage_bytes, input1, output2, B + 1);
  const int32_t vector_length = HiddenSize;
  int32_t num_threads;
  if (vector_length < 1024) {
    num_threads = vector_length;
  } else {
    if (vector_length % 512 == 0) {
      num_threads = 512;
    } else if (vector_length % 256 == 0) {
      num_threads = 256;
    } else if (vector_length % 128 == 0) {
      num_threads = 128;
    } else if (vector_length % 64 == 0) {
      num_threads = 64;
    } else if (vector_length % 32 == 0) {
      num_threads = 32;
    } else if (vector_length % 16 == 0) {
      num_threads = 16;
    } else if (vector_length % 8 == 0) {
      num_threads = 8;
    } else if (vector_length % 4 == 0) {
      num_threads = 4;
    } else if (vector_length % 2 == 0) {
      num_threads = 2;
    } else {
      num_threads = 1;
    }
  }
  const dim3 num_blocks(
      B,
      MaxLength,
      vector_length /
          num_threads);  //  batches, max sequence length, input0.dims.d[2]/*
  remove_padding_kernel<<<num_blocks, num_threads, 0, stream>>>(
      input0, output2, output0);  // input(no_varlen), pos_id, input(varlen)
  return cudaGetLastError() != cudaSuccess;
}

nvinfer1::DataType TransformerOutputConvertPlugin::getOutputDataType(
    int index,
    const nvinfer1::DataType* input_types,
    int nb_inputs) const TRT_NOEXCEPT {
  if (index == 0) {
    return nvinfer1::DataType::kHALF;
  }
}

nvinfer1::DimsExprs TransformerOutputConvertPlugin::getOutputDimensions(
    int outputIndex,
    const nvinfer1::DimsExprs* inputs,
    int nbInputs,
    nvinfer1::IExprBuilder& exprBuilder) TRT_NOEXCEPT {
  nvinfer1::DimsExprs output_dims;
  if (outputIndex == 0) {
    output_dims = inputs[1];
  }
  return output_dims;
}

bool TransformerOutputConvertPlugin::supportsFormatCombination(
    int pos,
    const nvinfer1::PluginTensorDesc* inOut,
    int nbInputs,
    int nbOutputs) TRT_NOEXCEPT {
  PADDLE_ENFORCE_EQ(nbInputs,
                    3,
                    common::errors::InvalidArgument(
                        "TransformerOutputConvertPlugin must have 3 inputs, "
                        "but got %d input(s). ",
                        nbInputs));
  PADDLE_ENFORCE_EQ(nbOutputs,
                    1,
                    common::errors::InvalidArgument(
                        "TransformerOutputConvertPlugin must have 1 output, "
                        "but got %d output(s). ",
                        nbOutputs));
  if (pos == 0) {  // qkv plugin output(varlen)
    return inOut[pos].format == nvinfer1::TensorFormat::kLINEAR &&
           inOut[pos].type == nvinfer1::DataType::kHALF;
  } else if (pos == 1) {  // qkv plugin input(no_varlen)
    return inOut[pos].format == nvinfer1::TensorFormat::kLINEAR &&
           inOut[pos].type == nvinfer1::DataType::kHALF;
  } else if (pos == 2) {  // pos id
    return inOut[pos].format == nvinfer1::TensorFormat::kLINEAR &&
           inOut[pos].type == nvinfer1::DataType::kINT32;
  } else if (pos == 3) {  // qkv plugin output(no_varlen)
    return inOut[pos].format == nvinfer1::TensorFormat::kLINEAR &&
           inOut[pos].type == nvinfer1::DataType::kHALF;
  }
}

void TransformerOutputConvertPlugin::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc* inputs,
    int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* outputs,
    int nbOutputs) TRT_NOEXCEPT {}

void TransformerOutputConvertPlugin::attachToContext(
    cudnnContext* cudnnContext,
    cublasContext* cublasContext,
    nvinfer1::IGpuAllocator* gpuAllocator) TRT_NOEXCEPT {}

void TransformerOutputConvertPlugin::detachFromContext() TRT_NOEXCEPT {}

void TransformerOutputConvertPlugin::terminate() TRT_NOEXCEPT {}

int TransformerOutputConvertPlugin::enqueue(
    const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc,
    const void* const* inputs,
    void* const* outputs,
    void* workspace,
    cudaStream_t stream) TRT_NOEXCEPT {
  const half* input0 =
      static_cast<const half*>(inputs[0]);  // qkv plugin output(varlen)
  const half* input1 =
      static_cast<const half*>(inputs[1]);  // qkv plugin input(no_varlen)
  const int32_t* input2 = static_cast<const int32_t*>(inputs[2]);  // pos id
  half* output =
      static_cast<half*>(outputs[0]);  // qkv plugin output(no_varlen)
  const auto input1_desc = inputDesc[1];
  const int32_t B = input1_desc.dims.d[0];           // batches
  const int32_t MaxLength = input1_desc.dims.d[1];   // max token length
  const int32_t HiddenSize = input1_desc.dims.d[2];  // hidden size

  const int32_t vector_length = HiddenSize;
  int32_t num_threads;
  if (vector_length < 1024) {
    num_threads = vector_length;
  } else {
    if (vector_length % 512 == 0) {
      num_threads = 512;
    } else if (vector_length % 256 == 0) {
      num_threads = 256;
    } else if (vector_length % 128 == 0) {
      num_threads = 128;
    } else if (vector_length % 64 == 0) {
      num_threads = 64;
    } else if (vector_length % 32 == 0) {
      num_threads = 32;
    } else if (vector_length % 16 == 0) {
      num_threads = 16;
    } else if (vector_length % 8 == 0) {
      num_threads = 8;
    } else if (vector_length % 4 == 0) {
      num_threads = 4;
    } else if (vector_length % 2 == 0) {
      num_threads = 2;
    } else {
      num_threads = 1;
    }
  }
  const dim3 num_blocks(
      B,
      MaxLength,
      vector_length / num_threads);  //  batches, max sequence length
                                     //  (mask_id.dims.d[1]),
                                     //  input.dims.d[1]/*
  recover_padding_kernel<<<num_blocks, num_threads, 0, stream>>>(
      input0, input2, output);
  return cudaGetLastError() != cudaSuccess;
}

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
