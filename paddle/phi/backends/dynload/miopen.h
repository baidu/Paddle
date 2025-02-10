/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

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
#include <glog/logging.h>
#include <miopen/miopen.h>
#include <miopen/version.h>

#include <mutex>  // NOLINT

#include "paddle/phi/backends/dynload/dynamic_loader.h"
#include "paddle/phi/common/port.h"

#define MIOPEN_VERSION                                       \
  (MIOPEN_VERSION_MAJOR * 1000 + MIOPEN_VERSION_MINOR * 10 + \
   MIOPEN_VERSION_PATCH)  // NOLINT

// MIOPEN only support NCHW, just for compatibility with CUDNN API
typedef enum {
  MIOPEN_TENSOR_NCHW = 0,
  MIOPEN_TENSOR_NHWC = 1,
} miopenTensorFormat_t;

namespace phi {
namespace dynload {

extern std::once_flag miopen_dso_flag;
extern void* miopen_dso_handle;
extern bool HasCUDNN();

inline const char* miopenGetErrorString(miopenStatus_t status) {
  switch (status) {
    case miopenStatusSuccess:
      return "MIOPEN_STATUS_SUCCESS";
    case miopenStatusNotInitialized:
      return "MIOPEN_STATUS_NOT_INITIALIZED";
    case miopenStatusInvalidValue:
      return "MIOPEN_STATUS_INVALID_VALUE";
    case miopenStatusBadParm:            // typos: disable-line
      return "MIOPEN_STATUS_BAD_PARAM";  // typos: disable-line
    case miopenStatusAllocFailed:
      return "MIOPEN_STATUS_ALLOC_FAILED";
    case miopenStatusInternalError:
      return "MIOPEN_STATUS_INTERNAL_ERROR";
    case miopenStatusNotImplemented:
      return "MIOPEN_STATUS_NOT_IMPLEMENTED";
    case miopenStatusUnsupportedOp:
      return "MIOPEN_STATUS_UNSUPPORTED_OP";
    case miopenStatusUnknownError:
    default:
      return "MIOPEN_STATUS_UNKNOWN_ERROR";
  }
}

extern void EnforceCUDNNLoaded(const char* fn_name);
#define DECLARE_DYNAMIC_LOAD_MIOPEN_WRAP(__name)                     \
  struct DynLoad__##__name {                                         \
    template <typename... Args>                                      \
    auto operator()(Args... args) -> DECLARE_TYPE(__name, args...) { \
      using miopen_func = decltype(&::__name);                       \
      std::call_once(miopen_dso_flag, []() {                         \
        miopen_dso_handle = phi::dynload::GetCUDNNDsoHandle();       \
      });                                                            \
      EnforceCUDNNLoaded(#__name);                                   \
      static void* p_##__name = dlsym(miopen_dso_handle, #__name);   \
      return reinterpret_cast<miopen_func>(p_##__name)(args...);     \
    }                                                                \
  };                                                                 \
  extern struct DynLoad__##__name __name

/**
 * include all needed miopen functions in HPPL
 **/
#define MIOPEN_DNN_ROUTINE_EACH(__macro)                  \
  __macro(miopenGetVersion);                              \
  __macro(miopenOpTensor);                                \
  __macro(miopenSet4dTensorDescriptor);                   \
  __macro(miopenSetTensorDescriptor);                     \
  __macro(miopenInitConvolutionNdDescriptor);             \
  __macro(miopenFindConvolutionForwardAlgorithm);         \
  __macro(miopenGetConvolutionNdForwardOutputDim);        \
  __macro(miopenFindConvolutionBackwardDataAlgorithm);    \
  __macro(miopenFindConvolutionBackwardWeightsAlgorithm); \
  __macro(miopenGetTensorDescriptor);                     \
  __macro(miopenCreateTensorDescriptor);                  \
  __macro(miopenDestroyTensorDescriptor);                 \
  __macro(miopenGetTensorDescriptorSize);                 \
  __macro(miopenSet2dPoolingDescriptor);                  \
  __macro(miopenGet2dPoolingDescriptor);                  \
  __macro(miopenGetPoolingNdForwardOutputDim);            \
  __macro(miopenCreateConvolutionDescriptor);             \
  __macro(miopenCreatePoolingDescriptor);                 \
  __macro(miopenDestroyPoolingDescriptor);                \
  __macro(miopenPoolingGetWorkSpaceSize);                 \
  __macro(miopenPoolingGetWorkSpaceSizeV2);               \
  __macro(miopenSetNdPoolingDescriptor);                  \
  __macro(miopenInitConvolutionDescriptor);               \
  __macro(miopenDestroyConvolutionDescriptor);            \
  __macro(miopenGetConvolutionNdDescriptor);              \
  __macro(miopenDeriveBNTensorDescriptor);                \
  __macro(miopenCreate);                                  \
  __macro(miopenDestroy);                                 \
  __macro(miopenSetStream);                               \
  __macro(miopenActivationForward);                       \
  __macro(miopenActivationBackward);                      \
  __macro(miopenConvolutionBackwardWeights);              \
  __macro(miopenConvolutionForward);                      \
  __macro(miopenConvolutionForwardBias);                  \
  __macro(miopenConvolutionBackwardBias);                 \
  __macro(miopenConvolutionForwardGetWorkSpaceSize);      \
  __macro(miopenConvolutionBackwardDataGetWorkSpaceSize); \
  __macro(miopenTransformTensor);                         \
  __macro(miopenPoolingForward);                          \
  __macro(miopenPoolingBackward);                         \
  __macro(miopenSoftmaxBackward);                         \
  __macro(miopenSoftmaxBackward_V2);                      \
  __macro(miopenSoftmaxForward);                          \
  __macro(miopenSoftmaxForward_V2);                       \
  __macro(miopenCreateDropoutDescriptor);                 \
  __macro(miopenDestroyDropoutDescriptor);                \
  __macro(miopenRestoreDropoutDescriptor);                \
  __macro(miopenDropoutGetStatesSize);                    \
  __macro(miopenSetDropoutDescriptor);                    \
  __macro(miopenCreateRNNDescriptor);                     \
  __macro(miopenDestroyRNNDescriptor);                    \
  __macro(miopenSetRNNDescriptor);                        \
  __macro(miopenSetRNNDescriptor_V2);                     \
  __macro(miopenGetRNNParamsSize);                        \
  __macro(miopenGetRNNWorkspaceSize);                     \
  __macro(miopenGetRNNTrainingReserveSize);               \
  __macro(miopenRNNForwardTraining);                      \
  __macro(miopenRNNBackwardData);                         \
  __macro(miopenRNNBackwardWeights);                      \
  __macro(miopenRNNForwardInference);                     \
  __macro(miopenGetTensorNumBytes);

MIOPEN_DNN_ROUTINE_EACH(DECLARE_DYNAMIC_LOAD_MIOPEN_WRAP)

#define MIOPEN_DNN_ROUTINE_EACH_R2(__macro) \
  __macro(miopenConvolutionBackwardData);
MIOPEN_DNN_ROUTINE_EACH_R2(DECLARE_DYNAMIC_LOAD_MIOPEN_WRAP)

// APIs available after R3:
#define MIOPEN_DNN_ROUTINE_EACH_AFTER_R3(__macro) \
  __macro(miopenConvolutionBackwardWeightsGetWorkSpaceSize);
MIOPEN_DNN_ROUTINE_EACH_AFTER_R3(DECLARE_DYNAMIC_LOAD_MIOPEN_WRAP)

// APIs available after R4:
#define MIOPEN_DNN_ROUTINE_EACH_AFTER_R4(__macro)    \
  __macro(miopenBatchNormalizationForwardTraining);  \
  __macro(miopenBatchNormalizationForwardInference); \
  __macro(miopenBatchNormalizationBackward);
MIOPEN_DNN_ROUTINE_EACH_AFTER_R4(DECLARE_DYNAMIC_LOAD_MIOPEN_WRAP)

// APIs in R5
#define MIOPEN_DNN_ROUTINE_EACH_R5(__macro)  \
  __macro(miopenCreateActivationDescriptor); \
  __macro(miopenSetActivationDescriptor);    \
  __macro(miopenGetActivationDescriptor);    \
  __macro(miopenDestroyActivationDescriptor);
MIOPEN_DNN_ROUTINE_EACH_R5(DECLARE_DYNAMIC_LOAD_MIOPEN_WRAP)

// APIs in R6
#define MIOPEN_DNN_ROUTINE_EACH_R6(__macro) \
/*__macro(miopenSetRNNDescriptor_v6);*/
MIOPEN_DNN_ROUTINE_EACH_R6(DECLARE_DYNAMIC_LOAD_MIOPEN_WRAP)

#define MIOPEN_DNN_ROUTINE_EACH_R7(__macro) \
  __macro(miopenSetConvolutionGroupCount);  \
  __macro(miopenCreateCTCLossDescriptor);   \
  __macro(miopenDestroyCTCLossDescriptor);  \
  __macro(miopenGetCTCLossDescriptor);      \
  __macro(miopenSetCTCLossDescriptor);      \
  __macro(miopenGetCTCLossWorkspaceSize);   \
  __macro(miopenCTCLoss);
MIOPEN_DNN_ROUTINE_EACH_R7(DECLARE_DYNAMIC_LOAD_MIOPEN_WRAP)

#define MIOPEN_DNN_ROUTINE_EACH_AFTER_R7(__macro)                    \
/*__macro(cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize); \
__macro(cudnnBatchNormalizationForwardTrainingEx);                   \
__macro(cudnnGetBatchNormalizationBackwardExWorkspaceSize);          \
__macro(cudnnBatchNormalizationBackwardEx);                          \
__macro(cudnnGetBatchNormalizationTrainingExReserveSpaceSize);*/
MIOPEN_DNN_ROUTINE_EACH_AFTER_R7(DECLARE_DYNAMIC_LOAD_MIOPEN_WRAP)
}  // namespace dynload
}  // namespace phi
