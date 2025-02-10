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

#pragma once

#include <string>
#include <vector>

#include "paddle/common/flags.h"

#include "paddle/common/errors.h"
#include "paddle/common/macros.h"
#include "paddle/phi/backends/dynload/miopen.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/enforce.h"

// MIOPEN do not have epsilon definition
#define CUDNN_BN_MIN_EPSILON 1e-05

COMMON_DECLARE_bool(cudnn_deterministic);

namespace phi {
namespace backends {
namespace gpu {

inline const char* miopenGetErrorString(miopenStatus_t status) {
  switch (status) {
    case miopenStatusSuccess:
      return "miopenStatusSuccess";
    case miopenStatusNotInitialized:
      return "miopenStatusNotInitialized";
    case miopenStatusAllocFailed:
      return "miopenStatusAllocFailed";
    case miopenStatusBadParm:        // typos: disable-line
      return "miopenStatusBadParm";  // typos: disable-line
    case miopenStatusInternalError:
      return "miopenStatusInternalError";
    case miopenStatusInvalidValue:
      return "miopenStatusInvalidValue";
    case miopenStatusUnknownError:
      return "miopenStatusUnknownError";
    case miopenStatusNotImplemented:
      return "miopenStatusNotImplemented";
    default:
      return "Unknown miopen error number";
  }
}

// no use, but will have compiling error if not defined
#define CUDNN_VERSION_COMPUTE(major, minor, patch)     \
  ((major) <= 8 ? (major)*1000 + (minor)*100 + (patch) \
                : (major)*10000 + (minor)*100 + (patch))

#define CUDNN_VERSION_MIN(major, minor, patch) \
  (CUDNN_VERSION >= CUDNN_VERSION_COMPUTE(major, minor, patch))

enum class DataLayout {  // Not use
  kNHWC,
  kNCHW,
  kNCDHW,
  kNDHWC,  // add, liyamei
  kNCHW_VECT_C,
};

enum class PoolingMode {
  kMaximum,
  kMaximumDeterministic,
  kAverageExclusive,
  kAverageInclusive,
};

enum class ActivationMode {
  kNone,  // activation identity
  kSigmoid,
  kRelu,
  kRelu6,
  kReluX,
  kTanh,
  kBandPass,
};

inline miopenPoolingMode_t GetPoolingMode(const PoolingMode& mode) {
  switch (mode) {
    case PoolingMode::kMaximumDeterministic:
      return miopenPoolingMax;
    case PoolingMode::kAverageExclusive:
      return miopenPoolingAverage;
    case PoolingMode::kAverageInclusive:
      return miopenPoolingAverageInclusive;
    case PoolingMode::kMaximum:
      return miopenPoolingMax;
    default:
      PADDLE_THROW(
          common::errors::Unimplemented("Unexpected MIOPEN pooling mode."));
  }
}

inline ActivationMode StringToActivationMode(const std::string& str) {
  if (str == "identity") {
    return ActivationMode::kNone;
  } else if (str == "sigmoid") {
    return ActivationMode::kSigmoid;
  } else if (str == "relu") {
    return ActivationMode::kRelu;
  } else if (str == "relu6") {
    return ActivationMode::kRelu6;
  } else if (str == "relux") {
    return ActivationMode::kReluX;
  } else if (str == "tanh") {
    return ActivationMode::kTanh;
  } else if (str == "bandpass") {
    return ActivationMode::kBandPass;
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "Unknown MIOPEN activation string: %s.", str));
  }
}

template <typename T>
class CudnnDataType;

template <>
class CudnnDataType<phi::dtype::float16> {
 public:
  static const miopenDataType_t type = miopenHalf;
  // The scaling param type is float for HALF and FLOAT tensors
  using ScalingParamType = const float;
  using BatchNormParamType = float;
  static ScalingParamType* kOne() {
    static ScalingParamType v = 1.0;
    return &v;
  }
  static ScalingParamType* kZero() {
    static ScalingParamType v = 0.0;
    return &v;
  }
};

template <>
class CudnnDataType<phi::dtype::bfloat16> {
 public:
  static const miopenDataType_t type = miopenBFloat16;
  // The scaling param type is float for HALF and FLOAT tensors
  using ScalingParamType = const float;
  using BatchNormParamType = float;
  static ScalingParamType* kOne() {
    static ScalingParamType v = 1.0;
    return &v;
  }
  static ScalingParamType* kZero() {
    static ScalingParamType v = 0.0;
    return &v;
  }
};

template <>
class CudnnDataType<float> {
 public:
  static const miopenDataType_t type = miopenFloat;
  using ScalingParamType = const float;
  using BatchNormParamType = float;
  static ScalingParamType* kOne() {
    static ScalingParamType v = 1.0;
    return &v;
  }
  static ScalingParamType* kZero() {
    static ScalingParamType v = 0.0;
    return &v;
  }
};

inline miopenTensorFormat_t GetCudnnTensorFormat(const DataLayout& order) {
  switch (order) {
    case DataLayout::kNHWC:
      return MIOPEN_TENSOR_NHWC;
    case DataLayout::kNCHW:
      return MIOPEN_TENSOR_NCHW;
    case DataLayout::kNCDHW:
      return MIOPEN_TENSOR_NCHW;
    case DataLayout::kNDHWC:
      return MIOPEN_TENSOR_NHWC;
    default:
      PADDLE_THROW(common::errors::Unimplemented(
          "MIOPEN has no equivalent dataLayout for input order."));
  }
  return MIOPEN_TENSOR_NCHW;
}

class ScopedTensorDescriptor {
 public:
  ScopedTensorDescriptor() {
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::miopenCreateTensorDescriptor(&desc_));
  }
  ~ScopedTensorDescriptor() PADDLE_MAY_THROW {
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::miopenDestroyTensorDescriptor(desc_));
  }

  inline miopenTensorDescriptor_t descriptor(const miopenTensorFormat_t format,
                                             const miopenDataType_t type,
                                             const std::vector<int>& dims,
                                             const int groups = 1) {
    // the format is not used now, will add later
    std::vector<int> strides(dims.size());
    strides[dims.size() - 1] = 1;
    for (int i = dims.size() - 2; i >= 0; i--) {
      strides[i] = dims[i + 1] * strides[i + 1];
    }
    // Update tensor descriptor dims setting if groups > 1
    // NOTE: Here, Assume using NCHW or NCDHW order
    std::vector<int> dims_with_group(dims.begin(), dims.end());
    if (groups > 1) {
      dims_with_group[1] = dims_with_group[1] / groups;
    }

    // MIOPEN ONLY support data layout of NCHW
    PADDLE_ENFORCE_EQ(format,
                      MIOPEN_TENSOR_NCHW,
                      common::errors::InvalidArgument(
                          "format should ONLY be NCHW in MIOPEN."));
    if (dims.size() == 4) {
      PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::miopenSetTensorDescriptor(
          desc_,
          type,
          dims_with_group.size(),
          const_cast<int*>(dims_with_group.data()),
          const_cast<int*>(strides.data())));
    } else if (dims.size() == 5) {
      PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::miopenSetTensorDescriptor(
          desc_,
          type,
          dims_with_group.size(),
          const_cast<int*>(dims_with_group.data()),
          const_cast<int*>(strides.data())));
    }
    return desc_;
  }

  template <typename T>
  inline miopenTensorDescriptor_t descriptor(const DataLayout& order,
                                             const std::vector<int>& dims,
                                             const int groups = 1) {
    return descriptor(
        GetCudnnTensorFormat(order), CudnnDataType<T>::type, dims, groups);
  }

  inline miopenTensorDescriptor_t descriptor(const miopenDataType_t miopen_type,
                                             const std::vector<int>& dim,
                                             const std::vector<int>& stride) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::miopenSetTensorDescriptor(
        desc_,
        miopen_type,
        dim.size(),
        const_cast<int*>(dim.data()),
        const_cast<int*>(stride.data())));
    return desc_;
  }

  template <typename T>
  inline miopenTensorDescriptor_t descriptor(const std::vector<int>& dim,
                                             const std::vector<int>& stride) {
    return descriptor(CudnnDataType<T>::type, dim, stride);
  }

  inline miopenTensorDescriptor_t desc() { return desc_; }

 private:
  miopenTensorDescriptor_t desc_;
  DISABLE_COPY_AND_ASSIGN(ScopedTensorDescriptor);
};

class ScopedDropoutDescriptor {
 public:
  ScopedDropoutDescriptor() {
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::miopenCreateDropoutDescriptor(&desc_));
  }
  ~ScopedDropoutDescriptor() PADDLE_MAY_THROW {
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::miopenDestroyDropoutDescriptor(desc_));
  }

  inline miopenDropoutDescriptor_t descriptor(const miopenHandle_t& handle,
                                              const phi::Place& place,
                                              bool initialized,
                                              float dropout_prob_,
                                              phi::DenseTensor* dropout_state_,
                                              int seed,
                                              size_t state_size) {
    if (dropout_state_ == nullptr) {  // for no dropout or test
      PADDLE_ENFORCE_GPU_SUCCESS(
          phi::dynload::miopenSetDropoutDescriptor(desc_,
                                                   handle,
                                                   0 /* dropout */,
                                                   nullptr,
                                                   0 /* state_size */,
                                                   0 /* seed */,
                                                   false,
                                                   false,
                                                   MIOPEN_RNG_PSEUDO_XORWOW));
      return desc_;
    }
    auto* dropout_state_data = dropout_state_->data<uint8_t>();
    if (!initialized) {
      PADDLE_ENFORCE_GPU_SUCCESS(
          phi::dynload::miopenSetDropoutDescriptor(desc_,
                                                   handle,
                                                   dropout_prob_,
                                                   dropout_state_data,
                                                   state_size,
                                                   seed,
                                                   false,
                                                   false,
                                                   MIOPEN_RNG_PSEUDO_XORWOW));
    } else {
      auto dropout_state_dims = dropout_state_->dims();
      state_size = dropout_state_dims[0];
      PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::miopenRestoreDropoutDescriptor(
          desc_,
          handle,
          dropout_prob_,
          dropout_state_data,
          state_size,
          0,
          false,
          false,
          MIOPEN_RNG_PSEUDO_XORWOW));
    }
    return desc_;
  }
  inline miopenDropoutDescriptor_t desc() { return desc_; }

 private:
  miopenDropoutDescriptor_t desc_;
  DISABLE_COPY_AND_ASSIGN(ScopedDropoutDescriptor);
};

class ScopedRNNDescriptor {
 public:
  ScopedRNNDescriptor() {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::miopenCreateRNNDescriptor(&desc_));
  }
  ~ScopedRNNDescriptor() PADDLE_MAY_THROW {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::miopenDestroyRNNDescriptor(desc_));
  }

  inline miopenRNNDescriptor_t desc() { return desc_; }

 private:
  miopenRNNDescriptor_t desc_;
  DISABLE_COPY_AND_ASSIGN(ScopedRNNDescriptor);
};

class ScopedFilterDescriptor {
 public:
  ScopedFilterDescriptor() {
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::miopenCreateTensorDescriptor(&desc_));
  }
  ~ScopedFilterDescriptor() PADDLE_MAY_THROW {
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::miopenDestroyTensorDescriptor(desc_));
  }

  inline miopenTensorDescriptor_t descriptor(const miopenTensorFormat_t format,
                                             const miopenDataType_t type,
                                             const std::vector<int>& kernel,
                                             const int groups = 1) {
    // filter layout: MCHW(MCDHW), where M is the number of
    // output image channels, C is the number of input image channels,
    // D is the depth of the filter, H is the height of the filter, and W is the
    // width of the filter.
    std::vector<int> kernel_with_group(kernel.begin(), kernel.end());
    if (groups > 1) {
      kernel_with_group[0] /= groups;
      // NOTE: input filter(C) of the filter is already asserted to be C/groups.
    }
    std::vector<int> stride_dim(kernel_with_group.size());
    stride_dim.push_back(1);
    for (int k = kernel_with_group.size() - 2; k >= 0; k--) {
      stride_dim[k] = stride_dim[k + 1] * kernel_with_group[k + 1];
    }
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::miopenSetTensorDescriptor(
        desc_,
        type,
        kernel_with_group.size(),
        const_cast<int*>(kernel_with_group.data()),
        const_cast<int*>(stride_dim.data())));
    return desc_;
  }

  template <typename T>
  inline miopenTensorDescriptor_t descriptor(const DataLayout& order,
                                             const std::vector<int>& kernel,
                                             const int groups = 1) {
    return descriptor(
        GetCudnnTensorFormat(order), CudnnDataType<T>::type, kernel, groups);
  }

  inline miopenTensorDescriptor_t desc() { return desc_; }

 private:
  miopenTensorDescriptor_t desc_;
  DISABLE_COPY_AND_ASSIGN(ScopedFilterDescriptor);
};

class ScopedConvolutionDescriptor {
 public:
  ScopedConvolutionDescriptor() {
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::miopenCreateConvolutionDescriptor(&desc_));
  }
  ~ScopedConvolutionDescriptor() PADDLE_MAY_THROW {
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::miopenDestroyConvolutionDescriptor(desc_));
  }

  inline miopenConvolutionDescriptor_t descriptor(
      miopenDataType_t type,
      const std::vector<int>& pads,
      const std::vector<int>& strides,
      const std::vector<int>& dilations) {
    PADDLE_ENFORCE_EQ(pads.size(),
                      strides.size(),
                      common::errors::InvalidArgument(
                          "The size of pads and strides should be equal. But "
                          "received size of pads is %d, size of strides is %d.",
                          pads.size(),
                          strides.size()));
    PADDLE_ENFORCE_EQ(
        pads.size(),
        dilations.size(),
        common::errors::InvalidArgument(
            "The size of pads and dilations should be equal. But received size "
            "of pads is %d, size of dilations is %d.",
            pads.size(),
            dilations.size()));
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::miopenInitConvolutionNdDescriptor(
        desc_,
        pads.size(),
        const_cast<int*>(pads.data()),
        const_cast<int*>(strides.data()),
        const_cast<int*>(dilations.data()),
        miopenConvolution));
    return desc_;
  }

  template <typename T>
  inline miopenConvolutionDescriptor_t descriptor(
      const std::vector<int>& pads,
      const std::vector<int>& strides,
      const std::vector<int>& dilations) {
    return descriptor(CudnnDataType<T>::type, pads, strides, dilations);
  }

 private:
  miopenConvolutionDescriptor_t desc_;
  DISABLE_COPY_AND_ASSIGN(ScopedConvolutionDescriptor);
};

class ScopedPoolingDescriptor {
 public:
  ScopedPoolingDescriptor() {
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::miopenCreatePoolingDescriptor(&desc_));
  }
  ~ScopedPoolingDescriptor() PADDLE_MAY_THROW {
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::miopenDestroyPoolingDescriptor(desc_));
  }

  inline miopenPoolingDescriptor_t descriptor(const PoolingMode& mode,
                                              const std::vector<int>& kernel,
                                              const std::vector<int>& pads,
                                              const std::vector<int>& strides) {
    PADDLE_ENFORCE_EQ(kernel.size(),
                      pads.size(),
                      common::errors::InvalidArgument(
                          "The size of kernel and pads should be equal. But "
                          "received size of kernel is %d, size of pads is %d.",
                          kernel.size(),
                          pads.size()));
    PADDLE_ENFORCE_EQ(
        kernel.size(),
        strides.size(),
        common::errors::InvalidArgument(
            "The size of kernel and strides should be equal. But "
            "received size of kernel is %d, size of strides is %d.",
            kernel.size(),
            strides.size()));
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::miopenSetNdPoolingDescriptor(
        desc_,
        GetPoolingMode(mode),
        kernel.size(),
        const_cast<int*>(kernel.data()),
        const_cast<int*>(pads.data()),
        const_cast<int*>(strides.data())));
    return desc_;
  }

 private:
  miopenPoolingDescriptor_t desc_;
  DISABLE_COPY_AND_ASSIGN(ScopedPoolingDescriptor);
};

class ScopedActivationDescriptor {
 public:
  ScopedActivationDescriptor() {
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::miopenCreateActivationDescriptor(&desc_));
  }
  ~ScopedActivationDescriptor() PADDLE_MAY_THROW {
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::miopenDestroyActivationDescriptor(desc_));
  }

  template <typename T>
  inline miopenActivationDescriptor_t descriptor(
      const std::string& act, double value_max = static_cast<double>(0.)) {
    double relu_ceiling = 0.0;
    ActivationMode activation_mode = StringToActivationMode(act);
    miopenActivationMode_t mode;
    switch (activation_mode) {
      case ActivationMode::kNone:
        mode = miopenActivationPASTHRU;
        break;
      case ActivationMode::kRelu6:
        relu_ceiling = 6.0;
        mode = miopenActivationCLIPPEDRELU;
        break;
      case ActivationMode::kReluX:
        relu_ceiling = value_max;
        mode = miopenActivationCLIPPEDRELU;
        break;
      case ActivationMode::kRelu:
        mode = miopenActivationRELU;
        break;
      case ActivationMode::kSigmoid:
        mode = miopenActivationLOGISTIC;
        break;
      case ActivationMode::kTanh:
        mode = miopenActivationTANH;
        break;
      default:
        PADDLE_THROW(common::errors::Unimplemented(
            "Unrecognized MIOPEN activation mode: %d.",
            static_cast<int>(activation_mode)));
    }
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::miopenSetActivationDescriptor(
        desc_, mode, relu_ceiling, 0.0, 0.0));
    return desc_;
  }

 private:
  miopenActivationDescriptor_t desc_;
  DISABLE_COPY_AND_ASSIGN(ScopedActivationDescriptor);
};

class ScopedCTCLossDescriptor {
 public:
  ScopedCTCLossDescriptor() {
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::miopenCreateCTCLossDescriptor(&desc_));
  }
  ~ScopedCTCLossDescriptor() PADDLE_MAY_THROW {
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::miopenDestroyCTCLossDescriptor(desc_));
  }

  template <typename T>
  inline miopenCTCLossDescriptor_t descriptor() {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::miopenSetCTCLossDescriptor(
        desc_, CudnnDataType<T>::type, 0, false));
    return desc_;
  }

 private:
  miopenCTCLossDescriptor_t desc_;
  DISABLE_COPY_AND_ASSIGN(ScopedCTCLossDescriptor);
};

}  // namespace gpu
}  // namespace backends
}  // namespace phi
