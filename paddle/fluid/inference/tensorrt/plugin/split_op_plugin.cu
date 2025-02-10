// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <cuda_fp16.h>
#include <thrust/device_vector.h>

#include <algorithm>

#include "paddle/fluid/inference/tensorrt/plugin/split_op_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

template <typename T>
__device__ int upper_bound(T const* vals, int n, T const& key) {
  int i = 0;
  while (n > 0) {
    int m = n / 2;
    int j = i + m;
    if (!(key < vals[j])) {
      i = j + 1;
      n -= m + 1;
    } else {
      n = m;
    }
  }
  return i;
}

nvinfer1::Dims SplitPlugin::getOutputDimensions(
    int index, const nvinfer1::Dims* input_dims, int num_inputs) TRT_NOEXCEPT {
  PADDLE_ENFORCE_EQ(num_inputs,
                    1,
                    common::errors::InvalidArgument(
                        "Invalid number of inputs of split TRT plugin. "
                        "Expected 1, received %d.",
                        num_inputs));
  PADDLE_ENFORCE_LT(
      index,
      this->getNbOutputs(),
      common::errors::InvalidArgument(
          "Index of output should be less than the total number of outputs in "
          "split TensorRT plugin. Received index = %d >= total outputs = %d",
          index,
          this->getNbOutputs()));

  nvinfer1::Dims output_dims = input_dims[0];
  output_dims.d[axis_] = output_length_.at(index);
  return output_dims;
}

void SplitPlugin::shareData(const SplitPlugin* another) {
  outer_rows_ = another->outer_rows_;
  inner_cols_ = another->inner_cols_;
  same_shape_ = another->same_shape_;
  axis_shape_ = another->axis_shape_;
  segment_offsets_ = another->segment_offsets_;
}

int SplitPlugin::initialize() TRT_NOEXCEPT {
  PADDLE_ENFORCE_LE(axis_,
                    nvinfer1::Dims::MAX_DIMS,
                    common::errors::InvalidArgument(
                        "Axis dimension exceeds max dimension in TensorRT. "
                        "Received axis = %d > MAX_DIMS = %d",
                        axis_,
                        nvinfer1::Dims::MAX_DIMS));
  // notice input dims is [C, H, W]
  nvinfer1::Dims dims = this->getInputDims(0);
  outer_rows_ = 1;
  inner_cols_ = 1;
  for (int i = 0; i < axis_; ++i) {
    outer_rows_ *= dims.d[i];
  }
  for (int i = axis_ + 1; i < dims.nbDims; ++i) {
    inner_cols_ *= dims.d[i];
  }
  same_shape_ = true;
  std::vector<int> segment_offsets(1, 0);
  for (int i = 0; i < this->getNbOutputs(); ++i) {
    if (output_length_[i] != output_length_[0]) {
      same_shape_ = false;
    }
    segment_offsets.push_back(segment_offsets.back() + output_length_[i]);
  }
  axis_shape_ = dims.d[axis_];
  segment_offsets_ = std::move(segment_offsets);
  return 0;
}

// nothing to release according to initialize
void SplitPlugin::terminate() TRT_NOEXCEPT {}

// The following part of the code refers to onnx-tensorrt
// https://github.com/onnx/onnx-tensorrt/blob/master/Split.cu
template <typename T>
__global__ void split_kernel(int nsegment,
                             int const* __restrict__ segment_offsets,
                             T const* __restrict__ idata,
                             T* const* odatas,
                             int inner_cols,
                             int axis_shape,
                             int outer_rows) {
  int x0 = threadIdx.x + blockIdx.x * blockDim.x;
  int src_y0 = threadIdx.y + blockIdx.y * blockDim.y;
  int z0 = threadIdx.z + blockIdx.z * blockDim.z;
  for (int z = z0; z < outer_rows; z += blockDim.z * gridDim.z) {
    for (int src_y = src_y0; src_y < axis_shape;
         src_y += blockDim.y * gridDim.y) {
      for (int x = x0; x < inner_cols; x += blockDim.x * gridDim.x) {
        int segment = upper_bound(segment_offsets, nsegment, src_y) - 1;
        int dst_y = src_y - segment_offsets[segment];
        int dst_ny = segment_offsets[segment + 1] - segment_offsets[segment];
        odatas[segment][x + inner_cols * (dst_y + dst_ny * z)] =
            idata[x + inner_cols * (src_y + axis_shape * z)];
      }
    }
  }
}

int SplitPlugin::enqueue(int batchSize,
                         const void* const* inputs,
#if IS_TRT_VERSION_LT(8000)
                         void** outputs,
                         void* workspace,
                         cudaStream_t stream) {
#else
                         void* const* outputs,
                         void* workspace,
                         cudaStream_t stream) TRT_NOEXCEPT {
#endif
  // this two thrust variables declared here , not with in .h
  // to avoid compiling error in cuda 11.6
  thrust::device_vector<int> d_segment_offsets = segment_offsets_;
  thrust::device_vector<float*> d_output_ptrs;
  d_output_ptrs.resize(segment_offsets_.size(), nullptr);
  const int* d_segment_offsets_ptr =
      thrust::raw_pointer_cast(&d_segment_offsets[0]);
  float const* input_ptr = reinterpret_cast<float const*>(inputs[0]);
  float* const* h_odatas = reinterpret_cast<float* const*>(outputs);
  float** output_ptrs = thrust::raw_pointer_cast(&d_output_ptrs[0]);
  PADDLE_ENFORCE_GPU_SUCCESS(
      cudaMemcpyAsync(output_ptrs,
                      h_odatas,
                      d_output_ptrs.size() * sizeof(float*),
                      cudaMemcpyHostToDevice,
                      stream));

  int outer_rows = outer_rows_ * batchSize;

  dim3 block(32, 16);
  dim3 grid(std::min((inner_cols_ - 1) / block.x + 1, 65535u),
            std::min((axis_shape_ - 1) / block.y + 1, 65535u),
            std::min((outer_rows_ - 1) / block.z + 1, 65535u));

  split_kernel<<<grid, block, 0, stream>>>(segment_offsets_.size(),
                                           d_segment_offsets_ptr,
                                           input_ptr,
                                           output_ptrs,
                                           inner_cols_,
                                           axis_shape_,
                                           outer_rows);
  return cudaGetLastError() != cudaSuccess;
}

// Dynamic Plugin below.
#if IS_TRT_VERSION_GE(6000)
int SplitPluginDynamic::initialize() TRT_NOEXCEPT { return 0; }

size_t SplitPluginDynamic::getSerializationSize() const TRT_NOEXCEPT {
  return SerializedSize(axis_) + SerializedSize(output_length_) +
         SerializedSize(with_fp16_);
}

void SplitPluginDynamic::serialize(void* buffer) const TRT_NOEXCEPT {
  SerializeValue(&buffer, axis_);
  SerializeValue(&buffer, output_length_);
  SerializeValue(&buffer, with_fp16_);
}

nvinfer1::DimsExprs SplitPluginDynamic::getOutputDimensions(
    int output_index,
    const nvinfer1::DimsExprs* inputs,
    int nb_inputs,
    nvinfer1::IExprBuilder& expr_builder) TRT_NOEXCEPT {
  PADDLE_ENFORCE_EQ(nb_inputs,
                    1,
                    common::errors::InvalidArgument(
                        "The Split plugin should be only one input."));
  PADDLE_ENFORCE_LT(output_index,
                    output_length_.size(),
                    common::errors::InvalidArgument(
                        "When GetOutputDimensions, the index(%d) should not "
                        "greater the num(%d) of the outputs.",
                        output_index,
                        output_length_.size()));

  nvinfer1::DimsExprs output_dims = inputs[0];
  output_dims.d[axis_] = expr_builder.constant(output_length_.at(output_index));

  return output_dims;
}

bool SplitPluginDynamic::supportsFormatCombination(
    int pos,
    const nvinfer1::PluginTensorDesc* in_out,
    int nb_inputs,
    int nb_outputs) TRT_NOEXCEPT {
  PADDLE_ENFORCE_NOT_NULL(
      in_out,
      common::errors::InvalidArgument(
          "The input of split plugin should not be nullptr."));

  PADDLE_ENFORCE_LT(
      pos,
      nb_inputs + nb_outputs,
      common::errors::InvalidArgument("The pos(%d) should be less than the "
                                      "num(%d) of the input and the output.",
                                      pos,
                                      nb_inputs + nb_outputs));
  (in_out && pos < (nb_inputs + nb_outputs));

  const nvinfer1::PluginTensorDesc& in = in_out[pos];
  if (pos == 0) {
    if (with_fp16_) {
      return (in.type == nvinfer1::DataType::kFLOAT ||
              in.type == nvinfer1::DataType::kHALF) &&
             (in.format == nvinfer1::TensorFormat::kLINEAR);
    } else {
      return (in.type == nvinfer1::DataType::kFLOAT) &&
             (in.format == nvinfer1::TensorFormat::kLINEAR);
    }
  }
  const nvinfer1::PluginTensorDesc& prev = in_out[pos - 1];
  // output
  return in.type == prev.type && in.format == prev.format;
}

nvinfer1::DataType SplitPluginDynamic::getOutputDataType(
    int index,
    const nvinfer1::DataType* input_types,
    int nb_inputs) const TRT_NOEXCEPT {
  return input_types[0];
}

int SplitPluginDynamic::enqueue(const nvinfer1::PluginTensorDesc* input_desc,
                                const nvinfer1::PluginTensorDesc* output_desc,
                                const void* const* inputs,
                                void* const* outputs,
                                void* workspace,
                                cudaStream_t stream) TRT_NOEXCEPT {
  auto input_dims = input_desc[0].dims;
  int outer_rows = 1;
  int inner_cols = 1;
  // with batch
  for (int i = 0; i < axis_; i++) {
    outer_rows *= input_dims.d[i];
  }

  for (int i = axis_ + 1; i < input_dims.nbDims; i++) {
    inner_cols *= input_dims.d[i];
  }

  std::vector<int> segment_offsets(1, 0);
  for (int i = 0; i < this->getNbOutputs(); i++) {
    segment_offsets.push_back(segment_offsets.back() + output_length_[i]);
  }
  int axis_shape = input_dims.d[axis_];
  thrust::device_vector<int> d_segment_offsets = segment_offsets;
  const int* d_segment_offsets_ptr =
      thrust::raw_pointer_cast(&d_segment_offsets[0]);

  dim3 block(32, 16);
  dim3 grid(std::min((inner_cols - 1) / block.x + 1, 65535u),
            std::min((axis_shape - 1) / block.y + 1, 65535u),
            std::min((outer_rows - 1) / block.z + 1, 65535u));

  auto input_type = input_desc[0].type;
  if (input_type == nvinfer1::DataType::kFLOAT) {
    VLOG(1) << "TRT Plugin DataType selected. Split-->fp32";
    thrust::device_vector<float*> d_output_ptrs;
    d_output_ptrs.resize(this->getNbOutputs(), nullptr);

    const float* input_ptr = static_cast<const float*>(inputs[0]);
    float* const* h_odatas = reinterpret_cast<float* const*>(outputs);
    float** output_ptrs = thrust::raw_pointer_cast(&d_output_ptrs[0]);

    PADDLE_ENFORCE_GPU_SUCCESS(
        cudaMemcpyAsync(output_ptrs,
                        h_odatas,
                        d_output_ptrs.size() * sizeof(float*),
                        cudaMemcpyHostToDevice,
                        stream));

    split_kernel<<<grid, block, 0, stream>>>(d_segment_offsets.size(),
                                             d_segment_offsets_ptr,
                                             input_ptr,
                                             output_ptrs,
                                             inner_cols,
                                             axis_shape,
                                             outer_rows);
  } else if (input_type == nvinfer1::DataType::kHALF) {
    VLOG(1) << "TRT Plugin DataType selected. Split-->fp16";
    thrust::device_vector<half*> d_output_ptrs;
    d_output_ptrs.resize(this->getNbOutputs(), nullptr);

    const half* input_ptr = static_cast<const half*>(inputs[0]);
    half* const* h_odatas = reinterpret_cast<half* const*>(outputs);
    half** output_ptrs = thrust::raw_pointer_cast(&d_output_ptrs[0]);

    PADDLE_ENFORCE_GPU_SUCCESS(
        cudaMemcpyAsync(output_ptrs,
                        h_odatas,
                        d_output_ptrs.size() * sizeof(half*),
                        cudaMemcpyHostToDevice,
                        stream));

    split_kernel<<<grid, block, 0, stream>>>(d_segment_offsets.size(),
                                             d_segment_offsets_ptr,
                                             input_ptr,
                                             output_ptrs,
                                             inner_cols,
                                             axis_shape,
                                             outer_rows);
  }
  return cudaGetLastError() != cudaSuccess;
}
#endif

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
