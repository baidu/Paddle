// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/hlir/pe/schedule.h"

#include <absl/container/flat_hash_map.h>
#include <isl/cpp.h>
#include <math.h>

#include <algorithm>
#include <fstream>
#include <functional>
#include <iostream>
#include <numeric>
#include <utility>

#include "paddle/cinn/common/cas.h"
#include "paddle/cinn/hlir/pe/load_x86_params.h"
#include "paddle/cinn/optim/ir_simplify.h"
#include "paddle/cinn/poly/isl_utils.h"
#include "paddle/cinn/utils/string.h"
#include "paddle/common/enforce.h"
PD_DECLARE_bool(cinn_use_cuda_vectorize);
namespace cinn {
namespace hlir {
namespace pe {

using ParamsT =
    absl::flat_hash_map<std::string,
                        absl::flat_hash_map<std::string, std::vector<int>>>;

ParamsT CreateParamsImpl(common::UnknownArch) {
  PADDLE_THROW(::common::errors::InvalidArgument(
      "Schedule params must be initialized with target x86 or nvgpu."));
}

ParamsT CreateParamsImpl(common::X86Arch) { return CreateX86Params(); }

ParamsT CreateParamsImpl(common::ARMArch) {
  PADDLE_THROW(::common::errors::InvalidArgument(
      "Schedule params must be initialized with target x86 or nvgpu."));
}

ParamsT CreateParamsImpl(common::NVGPUArch) { return CreateCudaParams(); }

ParamsT CreateParamsImpl(common::HygonDCUArchHIP) { return CreateCudaParams(); }

ParamsT CreateParamsImpl(common::HygonDCUArchSYCL) {
  return CreateCudaParams();
}

ParamsT CreateParams(common::Arch arch) {
  return std::visit([](const auto &impl) { return CreateParamsImpl(impl); },
                    arch.variant());
}

ScheduleParam::ScheduleParam(cinn::common::Arch arch) {
  param_data = CreateParams(arch);
}

ScheduleParam::~ScheduleParam() {}

int GetInnerSplitter(int origin, int other_axis) {
  if (origin <= 1) return 1;
  int two_exp = 1;
  while (origin % two_exp == 0) {
    two_exp *= 2;
  }
  two_exp = two_exp / 2;
  int a = SplitEven(two_exp);
  int b = two_exp / a;
  while (a * other_axis >= 1024 || b * other_axis >= 1024) {
    two_exp = two_exp / 2;
    a = SplitEven(two_exp);
    b = two_exp / a;
  }
  if (origin == two_exp) {
    return 2;
  }
  return origin / two_exp;
}

int SplitEven(int origin) {
  if (origin <= 1) return 1;
  int res = 1;
  while (origin % res == 0 && res * res < origin) {
    res *= 2;
  }
  res = res / 2;
  return res;
}

int GetBasicFactor(const Type &type, const cinn::common::Target &target) {
  int target_native_vector_bits = target.get_target_bits() * 8;
  int type_bits = type.bits();
  return target_native_vector_bits / type_bits;
}

int GetBetterSplitFactor(int shape, int split_factor) {
  int better_factor = split_factor;
  while (better_factor > shape) {
    better_factor /= 2;
  }
  if (better_factor < shape && better_factor != split_factor)
    return better_factor * 2;
  return better_factor;
}

int GetVectorizeFactor(int shape, int split_factor) {
  int better_factor = 1;
  for (int i = split_factor; i > 1; i--) {
    if (shape % i == 0) {
      better_factor = i;
      break;
    }
  }
  return better_factor;
}

void ScheduleInjectiveCPU(poly::Stage *stage,
                          const std::vector<int> &output_shape,
                          const cinn::common::Target &target,
                          bool vectorizable) {
  int dims = stage->n_out_dims();
  int factor = GetBasicFactor(stage->tensor()->type(), target);
  poly::Iterator fused = stage->axis(0);
  if (dims >= 5) {
    fused = stage->Fuse({0, 1, 2});
  } else if (dims >= 3) {
    fused = stage->Fuse({0, 1});
  }
  stage->Parallel(fused);
  dims = stage->n_out_dims();

  if (vectorizable) {
    poly::Iterator lo;
    poly::Iterator li;
    int last_shape = stage->GetDimRange(dims - 1);
    factor = GetVectorizeFactor(last_shape, factor);
    std::tie(lo, li) = stage->Split(stage->axis(dims - 1), factor);
    stage->Vectorize(li, factor);
    if (dims == 1) {
      stage->Parallel(0);
    }
  }
}

void ScheduleInjectiveCPU1(poly::Stage *stage,
                           const std::vector<int> &output_shape,
                           const cinn::common::Target &target,
                           bool vectorizable) {
  int dims = stage->n_out_dims();
  if (dims > 1) {
    PADDLE_ENFORCE_EQ(stage->n_out_dims(),
                      stage->n_in_dims(),
                      ::common::errors::InvalidArgument(
                          "The dims of stage in and out are not equal"));
    PADDLE_ENFORCE_EQ(
        stage->n_out_dims(),
        output_shape.size(),
        ::common::errors::InvalidArgument(
            "The dims of stage out and output_shape's size are not equal"));
    poly::Iterator fused = stage->axis(dims - 1);
    int target_native_vector_bits = target.get_target_bits() * 8;
    int type_bits = stage->tensor()->type().bits();
    int prod_size = output_shape.back();
    // fuse conservatively for the complex index from poly and may not benefit a
    // lot compared with llvm optimization, only fuse the last two dims when the
    // last dimension is too small and can split and vectorize Todo: try reorder
    if (output_shape.back() * type_bits < target_native_vector_bits) {
      int last_two_dim_bits =
          output_shape[dims - 2] * output_shape[dims - 1] * type_bits;
      if (last_two_dim_bits % target_native_vector_bits == 0) {
        fused = stage->Fuse(dims - 2, dims - 1);
        prod_size *= output_shape[dims - 2];
      }
    }
    int split_factor = target_native_vector_bits / type_bits;
    if (vectorizable) {
      if (prod_size <= split_factor) {
        split_factor = GetBetterSplitFactor(prod_size, split_factor);
        if (split_factor >= 8) {
          stage->Vectorize(fused, split_factor);
        }
      } else {
        auto ssplit = stage->Split(fused, split_factor);
        auto &j_outer = std::get<0>(ssplit);
        auto &j_inner = std::get<1>(ssplit);
        stage->Vectorize(j_inner, split_factor);
      }
    }
  }
  if (stage->n_out_dims() > 1) {
    stage->Parallel(0);
  }
}

int GetArrayPackingFactor(int shape,
                          const Type &type,
                          const cinn::common::Target &target) {
  int split_base = GetBasicFactor(type, target);
  int split_factor = 1;
  // temporarily use shape-1 instead of shape for isl wrong for1 elimination
  int i = split_base * split_base < shape ? split_base * split_base : shape;
  for (; i > 1; i--) {
    if (shape % i == 0) {
      split_factor = i;
      break;
    }
  }
  return split_factor;
}

int GetThreadBindAxis(const std::vector<ir::Expr> &shape) {
  int thread_axis = shape.size() - 1;
  for (int idx = thread_axis; idx >= 0; --idx) {
    if (shape[idx].as_int32() > 1) {
      thread_axis = idx;
      break;
    }
  }
  return thread_axis;
}

int GetBlockBindAxis(const std::vector<ir::Expr> &shape,
                     const int thread_axis) {
  int block_axis = 0, max_dim_size = shape[0].as_int32();
  for (int idx = 0; idx <= thread_axis; ++idx) {
    if (max_dim_size < shape[idx].as_int32()) {
      if (idx < thread_axis) {
        max_dim_size = shape[idx].as_int32();
        block_axis = idx;
      } else {
        if (max_dim_size == 1) {
          block_axis = thread_axis;
        }
      }
    }
  }
  return block_axis;
}

void GetConv2dFactors(absl::flat_hash_map<std::string, int> *factors,
                      int oc,
                      int ic,
                      int fc,
                      int oh,
                      int ow,
                      const Type &type,
                      const cinn::common::Target &target,
                      const std::string &key,
                      bool import_params) {
  if (import_params) {
    auto &params = ScheduleParam::get_x86_instance().GetParam();
    if (params.count(key)) {
      VLOG(3) << "find saved param, key is: " << key;
      PADDLE_ENFORCE_EQ(
          !params[key]["oc_bn"].empty(),
          true,
          ::common::errors::InvalidArgument(
              "The parameter 'oc_bn' for key '%s' must not be empty.", key));
      PADDLE_ENFORCE_EQ(
          !params[key]["ic_bn"].empty(),
          true,
          ::common::errors::InvalidArgument(
              "The parameter 'ic_bn' for key '%s' must not be empty.", key));
      PADDLE_ENFORCE_EQ(
          !params[key]["ow_bn"].empty(),
          true,
          ::common::errors::InvalidArgument(
              "The parameter 'ow_bn' for key '%s' must not be empty.", key));
      (*factors)["oc_bn"] = params[key]["oc_bn"].back();
      (*factors)["ic_bn"] = params[key]["ic_bn"].back();
      (*factors)["ow_bn"] = params[key]["ow_bn"].back();
      if (!params[key]["oh_bn"].empty()) {
        (*factors)["oh_bn"] = params[key]["oh_bn"].back();
      }
      if (!params[key]["unroll_kw"].empty()) {
        (*factors)["unroll_kw"] = params[key]["unroll_kw"].back();
      }
      if (ic == fc) {
        (*factors)["fc_bn"] = (*factors)["ic_bn"];
      } else {
        int fc_bn = 1;
        for (int i = (*factors)["oc_bn"]; i > 1; i--) {
          if (fc < 1) break;
          if (fc % i == 0) {
            fc_bn = i;
            break;
          }
        }
        (*factors)["fc_bn"] = fc_bn;
      }
      return;
    } else {
      VLOG(3) << "Can not find saved param, key is: " << key;
    }
  }
  int bn_base = GetBasicFactor(type, target);
  int oc_bn = 1;
  for (int i = bn_base; i > 1; i--) {
    if (oc < 1) break;
    if (oc % i == 0) {
      oc_bn = i;
      break;
    }
  }
  int ic_bn = 1;
  for (int i = oc_bn; i > 1; i--) {
    if (ic < 1) break;
    if (ic % i == 0) {
      ic_bn = i;
      break;
    }
  }
  int fc_bn = 1;
  for (int i = oc_bn; i > 1; i--) {
    if (fc < 1) break;
    if (fc % i == 0) {
      fc_bn = i;
      break;
    }
  }
  (*factors)["oc_bn"] = oc_bn;
  (*factors)["ic_bn"] = ic_bn;
  (*factors)["fc_bn"] = fc_bn;
  int ow_bn = 1;

  if (oh < 1) {
    for (int i = bn_base; i > 1; i--) {
      if (ow < 1) break;
      if (ow % i == 0) {
        ow_bn = i;
        break;
      }
    }
    (*factors)["ow_bn"] = ow_bn;
  } else {
    int oh_bn = 1;
    int begin = std::min(ow, bn_base);
    for (int i = begin; i >= 1; i--) {
      if (ow < 1) break;
      if (ow % i == 0) {
        ow_bn = i;
        for (int j = oh; j >= 1; j--) {
          if (oh % j == 0 && j * ow_bn <= 16) {
            oh_bn = j;
            (*factors)["oh_bn"] = oh_bn;
            (*factors)["ow_bn"] = ow_bn;
            return;
          }
        }
      }
    }
  }
}

void GetConv2d1x1Factors(absl::flat_hash_map<std::string, int> *factors,
                         int oc,
                         int ic,
                         int oh,
                         int ow,
                         const Type &type,
                         const cinn::common::Target &target) {
  int bn_base = GetBasicFactor(type, target);
  int oc_bn = 1;
  for (int i = bn_base; i > 1; i--) {
    if (oc < 1) break;
    if (oc % i == 0) {
      oc_bn = i;
      break;
    }
  }
  int ic_bn = 1;
  for (int i = oc_bn; i > 1; i--) {
    if (ic < 1) break;
    if (ic % i == 0) {
      ic_bn = i;
      break;
    }
  }
  (*factors)["oc_bn"] = oc_bn;
  (*factors)["ic_bn"] = ic_bn;
  int ow_bn = 1;
  int oh_bn = 1;
  int begin = std::min(ow, bn_base);
  for (int i = begin; i >= 1; i--) {
    if (ow < 1) break;
    if (ow % i == 0) {
      ow_bn = i;
      for (int j = oh; j >= 1; j--) {
        if (oh % j == 0 && j * ow_bn <= 16) {
          oh_bn = j;
          (*factors)["oh_bn"] = oh_bn;
          (*factors)["ow_bn"] = ow_bn;
          return;
        }
      }
    }
  }
}

std::string GenerateX86ConvKey(const std::vector<Expr> &input_shape,
                               const std::vector<Expr> &weight_shape,
                               const std::vector<int> &strides,
                               const std::vector<int> &paddings,
                               const std::vector<int> &dilations,
                               const int &index,
                               const std::string &model_name) {
  // format: (model_name + index +)schedule_name + input_shape + weight_shape +
  // strides + paddings + dilations e.g. resnet18 0 X86ScheduleConv input 1 3
  // 224 224 weight 64 3 7 7 stride 2 2 padding 3 3 dilation 1 1
  std::string key;
  if (model_name != "") {
    key = model_name + " index " + std::to_string(index) + " ";
  }
  key += "X86ScheduleConv input";
  for (auto &shape : input_shape) {
    key += " " + std::to_string(shape.as_int32());
  }
  key += " weight";
  for (auto &shape : weight_shape) {
    key += " " + std::to_string(shape.as_int32());
  }
  key += " stride";
  for (auto &stride : strides) {
    key += " " + std::to_string(stride);
  }
  key += " padding";
  for (auto &padding : paddings) {
    key += " " + std::to_string(padding);
  }
  key += " dilation";
  for (auto &dilation : dilations) {
    key += " " + std::to_string(dilation);
  }
  VLOG(3) << "key: " << key;
  return key;
}

std::string GenerateX86ConvKey(const std::vector<int> &input_shape,
                               const std::vector<int> &weight_shape,
                               const std::vector<int> &strides,
                               const std::vector<int> &paddings,
                               const std::vector<int> &dilations,
                               const int &index,
                               const std::string &model_name) {
  // format: (model_name + index +)schedule_name + input_shape + weight_shape +
  // strides + paddings + dilations
  std::string key;
  if (model_name != "") {
    key = model_name + " index " + std::to_string(index) + " ";
  }
  key += "X86ScheduleConv input";
  for (auto &shape : input_shape) {
    key += " " + std::to_string(shape);
  }
  key += " weight";
  for (auto &shape : weight_shape) {
    key += " " + std::to_string(shape);
  }
  key += " stride";
  for (auto &stride : strides) {
    key += " " + std::to_string(stride);
  }
  key += " padding";
  for (auto &padding : paddings) {
    key += " " + std::to_string(padding);
  }
  key += " dilation";
  for (auto &dilation : dilations) {
    key += " " + std::to_string(dilation);
  }
  VLOG(3) << "key: " << key;
  return key;
}

void CreateX86SerialData(const std::string &file_name) {
  /** The format of serial data is:
   * hash_key: schedule_name + shape of input + shape of weights + stride +
   * padding + dilation value: vector of params
   */
  SaveSerialData(CreateX86Params(), file_name);
}

inline void InputDirectConvCudaParam(
    absl::flat_hash_map<std::string,
                        absl::flat_hash_map<std::string, std::vector<int>>>
        &model_data,
    const std::string &key,
    const std::vector<std::vector<int>> &int_data) {
  PADDLE_ENFORCE_EQ(
      int_data.size(),
      6UL,
      ::common::errors::InvalidArgument("int_data size should be 6"));
  absl::flat_hash_map<std::string, std::vector<int>> schedule_data;
  schedule_data["rc"] = int_data[0];
  schedule_data["ry"] = int_data[1];
  schedule_data["rx"] = int_data[2];
  schedule_data["f"] = int_data[3];
  schedule_data["y"] = int_data[4];
  schedule_data["x"] = int_data[5];
  PADDLE_ENFORCE_EQ(model_data.count(key),
                    0,
                    ::common::errors::InvalidArgument(
                        "Key '%s' in conv CUDA param already exists.", key));
  model_data[key] = schedule_data;
}

inline void InputWinogradConvCudaParam(
    absl::flat_hash_map<std::string,
                        absl::flat_hash_map<std::string, std::vector<int>>>
        &model_data,
    const std::string &key,
    const std::vector<std::vector<int>> &int_data) {
  PADDLE_ENFORCE_EQ(
      int_data.size(),
      4UL,
      ::common::errors::InvalidArgument("int_data size should be 4"));
  absl::flat_hash_map<std::string, std::vector<int>> schedule_data;
  schedule_data["rc"] = int_data[0];
  schedule_data["x"] = int_data[1];
  schedule_data["y"] = int_data[2];
  schedule_data["b"] = int_data[3];
  model_data[key] = schedule_data;
}

absl::flat_hash_map<std::string,
                    absl::flat_hash_map<std::string, std::vector<int>>>
CreateCudaParams() {
  absl::flat_hash_map<std::string,
                      absl::flat_hash_map<std::string, std::vector<int>>>
      model_data;
  // The format of serial data is:
  // hash_key: string = name of schedule + shape of input_pad + shape of weights
  // + shape of output value: vector of params
  InputDirectConvCudaParam(
      model_data,
      "CudaDirectConvSchedule 1 3 230 230 64 3 7 7 1 64 112 112",
      {{3, 1}, {7, 1}, {1, 7}, {1, 4, 8, 2}, {112, 1, 1, 1}, {1, 7, 16, 1}});
  InputDirectConvCudaParam(
      model_data,
      "CudaDirectConvSchedule 1 64 56 56 64 64 1 1 1 64 56 56",
      {{4, 16}, {1, 1}, {1, 1}, {1, 8, 8, 1}, {56, 1, 1, 1}, {1, 2, 28, 1}});
  InputDirectConvCudaParam(
      model_data,
      "CudaDirectConvSchedule 1 64 58 58 128 64 3 3 1 128 28 28",
      {{32, 2}, {1, 3}, {1, 3}, {4, 2, 16, 1}, {28, 1, 1, 1}, {1, 2, 14, 1}});
  InputDirectConvCudaParam(
      model_data,
      "CudaDirectConvSchedule 1 64 56 56 128 64 1 1 1 128 28 28",
      {{4, 16}, {1, 1}, {1, 1}, {2, 2, 32, 1}, {28, 1, 1, 1}, {1, 2, 14, 1}});
  InputDirectConvCudaParam(
      model_data,
      "CudaDirectConvSchedule 1 128 30 30 256 128 3 3 1 256 14 14",
      {{32, 4}, {1, 3}, {1, 3}, {8, 1, 16, 2}, {7, 1, 2, 1}, {1, 1, 7, 2}});
  InputDirectConvCudaParam(
      model_data,
      "CudaDirectConvSchedule 1 128 28 28 256 128 1 1 1 256 14 14",
      {{16, 8}, {1, 1}, {1, 1}, {8, 1, 16, 2}, {14, 1, 1, 1}, {1, 1, 14, 1}});
  InputDirectConvCudaParam(
      model_data,
      "CudaDirectConvSchedule 1 256 16 16 512 256 3 3 1 512 7 7",
      {{64, 4}, {1, 3}, {1, 3}, {32, 1, 16, 1}, {7, 1, 1, 1}, {1, 1, 7, 1}});
  InputDirectConvCudaParam(
      model_data,
      "CudaDirectConvSchedule 1 256 14 14 512 256 1 1 1 512 7 7",
      {{16, 16}, {1, 1}, {1, 1}, {16, 1, 32, 1}, {7, 1, 1, 1}, {1, 1, 7, 1}});

  // winograd
  InputDirectConvCudaParam(
      model_data,
      "CudaDirectConvSchedule 1 64 58 58 64 64 3 3 1 64 56 56",
      {{32, 2}, {1, 3}, {1, 3}, {4, 1, 8, 2}, {28, 1, 2, 1}, {1, 2, 7, 4}});
  // winograd
  InputDirectConvCudaParam(
      model_data,
      "CudaDirectConvSchedule 1 512 9 9 512 512 3 3 1 512 7 7",
      {{64, 8}, {1, 3}, {1, 3}, {32, 1, 16, 1}, {7, 1, 1, 1}, {1, 1, 7, 1}});
  // winograd
  InputDirectConvCudaParam(
      model_data,
      "CudaDirectConvSchedule 1 256 16 16 256 256 3 3 1 256 14 14",
      {{64, 4}, {1, 3}, {1, 3}, {16, 1, 16, 1}, {14, 1, 1, 1}, {1, 1, 14, 1}});
  // winograd
  InputDirectConvCudaParam(
      model_data,
      "CudaDirectConvSchedule 1 128 30 30 128 128 3 3 1 128 28 28",
      {{32, 4}, {1, 3}, {1, 3}, {8, 1, 16, 1}, {14, 1, 2, 1}, {1, 1, 7, 4}});

  // MobileNetV2 schedule params
  /*   InputDirectConvCudaParam(model_data,
                             "CudaDirectConvSchedule 1 3 226 226 32 3 3 3 1 32
     112 112",
                             {{3, 1}, {1, 3}, {1, 3}, {-1, 2, 8, 2}, {-1, 1, 1,
     7}, {-1, 1, 16, 1}}); */
  InputDirectConvCudaParam(
      model_data,
      "CudaDirectConvSchedule 1 32 112 112 16 32 1 1 1 16 112 112",
      {{-1, 4},
       {-1, 1},
       {-1, 1},
       {-1, 2, 2, 4},
       {-1, 1, 2, 1},
       {-1, 1, 56, 2}});
  InputDirectConvCudaParam(
      model_data,
      "CudaDirectConvSchedule 1 32 112 112 32 32 1 1 1 32 112 112",
      {{-1, 4},
       {-1, 1},
       {-1, 1},
       {-1, 1, 4, 8},
       {-1, 1, 2, 1},
       {-1, 7, 16, 1}});
  InputDirectConvCudaParam(
      model_data,
      "CudaDirectConvSchedule 1 16 112 112 96 16 1 1 1 96 112 112",
      {{-1, 4},
       {-1, 1},
       {-1, 1},
       {-1, 4, 4, 2},
       {-1, 2, 2, 1},
       {-1, 1, 16, 1}});
  InputDirectConvCudaParam(
      model_data,
      "CudaDirectConvSchedule 1 96 56 56 24 96 1 1 1 24 56 56",
      {{-1, 4},
       {-1, 1},
       {-1, 1},
       {-1, 3, 4, 2},
       {-1, 1, 1, 1},
       {-1, 1, 28, 2}});
  InputDirectConvCudaParam(
      model_data,
      "CudaDirectConvSchedule 1 24 56 56 144 24 1 1 1 144 56 56",
      {{-1, 6},
       {-1, 1},
       {-1, 1},
       {-1, 9, 4, 2},
       {-1, 2, 1, 1},
       {-1, 1, 56, 1}});
  InputDirectConvCudaParam(
      model_data,
      "CudaDirectConvSchedule 1 144 56 56 24 144 1 1 1 24 56 56",
      {{-1, 12},
       {-1, 1},
       {-1, 1},
       {-1, 1, 8, 3},
       {-1, 1, 1, 1},
       {-1, 2, 14, 2}});
  InputDirectConvCudaParam(
      model_data,
      "CudaDirectConvSchedule 1 144 28 28 32 144 1 1 1 32 28 28",
      {{-1, 8},
       {-1, 1},
       {-1, 1},
       {-1, 4, 8, 1},
       {-1, 1, 1, 1},
       {-1, 1, 14, 2}});
  InputDirectConvCudaParam(
      model_data,
      "CudaDirectConvSchedule 1 32 28 28 192 32 1 1 1 192 28 28",
      {{-1, 4},
       {-1, 1},
       {-1, 1},
       {-1, 6, 4, 1},
       {-1, 2, 1, 2},
       {-1, 1, 28, 1}});
  InputDirectConvCudaParam(
      model_data,
      "CudaDirectConvSchedule 1 192 28 28 32 192 1 1 1 32 28 28",
      {{-1, 48},
       {-1, 1},
       {-1, 1},
       {-1, 4, 8, 1},
       {-1, 1, 1, 1},
       {-1, 1, 28, 1}});
  InputDirectConvCudaParam(
      model_data,
      "CudaDirectConvSchedule 1 192 14 14 64 192 1 1 1 64 14 14",
      {{-1, 12},
       {-1, 1},
       {-1, 1},
       {-1, 1, 8, 2},
       {-1, 2, 1, 1},
       {-1, 1, 14, 1}});
  InputDirectConvCudaParam(
      model_data,
      "CudaDirectConvSchedule 1 64 14 14 384 64 1 1 1 384 14 14",
      {{-1, 4}, {-1, 1}, {-1, 1}, {-1, 2, 4, 3}, {-1, 1, 7, 1}, {-1, 1, 7, 2}});
  InputDirectConvCudaParam(
      model_data,
      "CudaDirectConvSchedule 1 384 14 14 64 384 1 1 1 64 14 14",
      {{-1, 48},
       {-1, 1},
       {-1, 1},
       {-1, 2, 16, 1},
       {-1, 1, 2, 1},
       {-1, 1, 7, 2}});
  InputDirectConvCudaParam(
      model_data,
      "CudaDirectConvSchedule 1 384 14 14 96 384 1 1 1 96 14 14",
      {{-1, 12},
       {-1, 1},
       {-1, 1},
       {-1, 2, 6, 1},
       {-1, 1, 2, 1},
       {-1, 1, 7, 2}});
  InputDirectConvCudaParam(
      model_data,
      "CudaDirectConvSchedule 1 96 14 14 576 96 1 1 1 576 14 14",
      {{-1, 6}, {-1, 1}, {-1, 1}, {-1, 1, 6, 6}, {-1, 1, 7, 1}, {-1, 1, 7, 2}});
  InputDirectConvCudaParam(
      model_data,
      "CudaDirectConvSchedule 1 576 14 14 96 576 1 1 1 96 14 14",
      {{-1, 24},
       {-1, 1},
       {-1, 1},
       {-1, 1, 8, 3},
       {-1, 1, 2, 1},
       {-1, 1, 7, 2}});
  InputDirectConvCudaParam(
      model_data,
      "CudaDirectConvSchedule 1 576 7 7 160 576 1 1 1 160 7 7",
      {{-1, 36},
       {-1, 1},
       {-1, 1},
       {-1, 2, 2, 2},
       {-1, 1, 7, 1},
       {-1, 1, 7, 1}});
  InputDirectConvCudaParam(
      model_data,
      "CudaDirectConvSchedule 1 160 7 7 960 160 1 1 1 960 7 7",
      {{-1, 16},
       {-1, 1},
       {-1, 1},
       {-1, 6, 4, 1},
       {-1, 1, 7, 1},
       {-1, 1, 7, 1}});
  InputDirectConvCudaParam(
      model_data,
      "CudaDirectConvSchedule 1 960 7 7 160 960 1 1 1 160 7 7",
      {{-1, 60},
       {-1, 1},
       {-1, 1},
       {-1, 2, 4, 1},
       {-1, 1, 7, 1},
       {-1, 1, 7, 1}});
  InputDirectConvCudaParam(
      model_data,
      "CudaDirectConvSchedule 1 960 7 7 320 960 1 1 1 320 7 7",
      {{-1, 20},
       {-1, 1},
       {-1, 1},
       {-1, 2, 2, 2},
       {-1, 1, 7, 1},
       {-1, 1, 7, 1}});
  InputDirectConvCudaParam(
      model_data,
      "CudaDirectConvSchedule 1 320 7 7 1280 320 1 1 1 1280 7 7",
      {{-1, 16},
       {-1, 1},
       {-1, 1},
       {-1, 2, 16, 1},
       {-1, 7, 1, 1},
       {-1, 1, 7, 1}});

  // EfficientNet schedule params
  InputDirectConvCudaParam(
      model_data,
      "CudaDirectConvSchedule 1 3 228 228 32 3 3 3 1 32 113 113",
      {{-1, 1},
       {-1, 1},
       {-1, 3},
       {-1, 32, 1, 1},
       {-1, 1, 1, 1},
       {-1, 1, 113, 1}});
  InputDirectConvCudaParam(model_data,
                           "CudaDirectConvSchedule 1 32 1 1 8 32 1 1 1 8 1 1",
                           {{-1, 16},
                            {-1, 1},
                            {-1, 1},
                            {-1, 1, 4, 1},
                            {-1, 1, 1, 1},
                            {-1, 1, 1, 1}});
  InputDirectConvCudaParam(
      model_data,
      "CudaDirectConvSchedule 1 8 1 1 32 8 1 1 1 32 1 1",
      {{-1, 8}, {-1, 1}, {-1, 1}, {-1, 1, 8, 4}, {-1, 1, 1, 1}, {-1, 1, 1, 1}});
  InputDirectConvCudaParam(model_data,
                           "CudaDirectConvSchedule 1 96 1 1 4 96 1 1 1 4 1 1",
                           {{-1, 48},
                            {-1, 1},
                            {-1, 1},
                            {-1, 1, 4, 1},
                            {-1, 1, 1, 1},
                            {-1, 1, 1, 1}});
  InputDirectConvCudaParam(model_data,
                           "CudaDirectConvSchedule 1 4 1 1 96 4 1 1 1 96 1 1",
                           {{-1, 2},
                            {-1, 1},
                            {-1, 1},
                            {-1, 12, 1, 1},
                            {-1, 1, 1, 1},
                            {-1, 1, 1, 1}});
  InputDirectConvCudaParam(model_data,
                           "CudaDirectConvSchedule 1 144 1 1 6 144 1 1 1 6 1 1",
                           {{-1, 48},
                            {-1, 1},
                            {-1, 1},
                            {-1, 1, 6, 1},
                            {-1, 1, 1, 1},
                            {-1, 1, 1, 1}});
  InputDirectConvCudaParam(
      model_data,
      "CudaDirectConvSchedule 1 6 1 1 144 6 1 1 1 144 1 1",
      {{-1, 2}, {-1, 1}, {-1, 1}, {-1, 2, 8, 1}, {-1, 1, 1, 1}, {-1, 1, 1, 1}});
  InputDirectConvCudaParam(
      model_data,
      "CudaDirectConvSchedule 1 144 28 28 40 144 1 1 1 40 28 28",
      {{-1, 16},
       {-1, 1},
       {-1, 1},
       {-1, 5, 8, 1},
       {-1, 1, 1, 1},
       {-1, 1, 28, 1}});
  InputDirectConvCudaParam(
      model_data,
      "CudaDirectConvSchedule 1 40 28 28 240 40 1 1 1 240 28 28",
      {{-1, 8},
       {-1, 1},
       {-1, 1},
       {-1, 2, 8, 3},
       {-1, 4, 1, 1},
       {-1, 1, 28, 1}});
  InputDirectConvCudaParam(
      model_data,
      "CudaDirectConvSchedule 1 240 1 1 10 240 1 1 1 10 1 1",
      {{-1, 60},
       {-1, 1},
       {-1, 1},
       {-1, 1, 5, 1},
       {-1, 1, 1, 1},
       {-1, 1, 1, 1}});
  InputDirectConvCudaParam(
      model_data,
      "CudaDirectConvSchedule 1 10 1 1 240 10 1 1 1 240 1 1",
      {{-1, 10},
       {-1, 1},
       {-1, 1},
       {-1, 1, 40, 3},
       {-1, 1, 1, 1},
       {-1, 1, 1, 1}});
  InputDirectConvCudaParam(
      model_data,
      "CudaDirectConvSchedule 1 240 28 28 40 240 1 1 1 40 28 28",
      {{-1, 16},
       {-1, 1},
       {-1, 1},
       {-1, 1, 8, 5},
       {-1, 1, 1, 1},
       {-1, 1, 28, 1}});
  InputDirectConvCudaParam(
      model_data,
      "CudaDirectConvSchedule 1 240 14 14 80 240 1 1 1 80 14 14",
      {{-1, 20},
       {-1, 1},
       {-1, 1},
       {-1, 2, 8, 1},
       {-1, 1, 2, 1},
       {-1, 1, 7, 2}});
  InputDirectConvCudaParam(
      model_data,
      "CudaDirectConvSchedule 1 80 14 14 480 80 1 1 1 480 14 14",
      {{-1, 16},
       {-1, 1},
       {-1, 1},
       {-1, 2, 8, 3},
       {-1, 1, 7, 1},
       {-1, 1, 7, 2}});
  InputDirectConvCudaParam(
      model_data,
      "CudaDirectConvSchedule 1 480 1 1 20 480 1 1 1 20 1 1",
      {{-1, 60},
       {-1, 1},
       {-1, 1},
       {-1, 1, 4, 1},
       {-1, 1, 1, 1},
       {-1, 1, 1, 1}});
  InputDirectConvCudaParam(
      model_data,
      "CudaDirectConvSchedule 1 20 1 1 480 20 1 1 1 480 1 1",
      {{-1, 5},
       {-1, 1},
       {-1, 1},
       {-1, 1, 32, 1},
       {-1, 1, 1, 1},
       {-1, 1, 1, 1}});
  InputDirectConvCudaParam(
      model_data,
      "CudaDirectConvSchedule 1 480 14 14 80 480 1 1 1 80 14 14",
      {{-1, 40},
       {-1, 1},
       {-1, 1},
       {-1, 2, 8, 1},
       {-1, 1, 2, 1},
       {-1, 1, 14, 1}});
  InputDirectConvCudaParam(
      model_data,
      "CudaDirectConvSchedule 1 480 14 14 112 480 1 1 1 112 14 14",
      {{-1, 20},
       {-1, 1},
       {-1, 1},
       {-1, 1, 8, 2},
       {-1, 1, 2, 1},
       {-1, 1, 7, 2}});
  InputDirectConvCudaParam(
      model_data,
      "CudaDirectConvSchedule 1 112 14 14 672 112 1 1 1 672 14 14",
      {{-1, 14},
       {-1, 1},
       {-1, 1},
       {-1, 1, 7, 6},
       {-1, 1, 7, 1},
       {-1, 1, 7, 2}});
  InputDirectConvCudaParam(
      model_data,
      "CudaDirectConvSchedule 1 672 1 1 28 672 1 1 1 28 1 1",
      {{-1, 28},
       {-1, 1},
       {-1, 1},
       {-1, 1, 7, 1},
       {-1, 1, 1, 1},
       {-1, 1, 1, 1}});
  InputDirectConvCudaParam(
      model_data,
      "CudaDirectConvSchedule 1 28 1 1 672 28 1 1 1 672 1 1",
      {{-1, 28},
       {-1, 1},
       {-1, 1},
       {-1, 1, 16, 1},
       {-1, 1, 1, 1},
       {-1, 1, 1, 1}});
  InputDirectConvCudaParam(
      model_data,
      "CudaDirectConvSchedule 1 672 14 14 112 672 1 1 1 112 14 14",
      {{-1, 14},
       {-1, 1},
       {-1, 1},
       {-1, 2, 4, 2},
       {-1, 1, 2, 1},
       {-1, 1, 7, 2}});
  InputDirectConvCudaParam(
      model_data,
      "CudaDirectConvSchedule 1 672 7 7 192 672 1 1 1 192 7 7",
      {{-1, 28},
       {-1, 1},
       {-1, 1},
       {-1, 1, 2, 3},
       {-1, 1, 7, 1},
       {-1, 1, 7, 1}});
  InputDirectConvCudaParam(
      model_data,
      "CudaDirectConvSchedule 1 192 7 7 1152 192 1 1 1 1152 7 7",
      {{-1, 24},
       {-1, 1},
       {-1, 1},
       {-1, 1, 12, 3},
       {-1, 7, 1, 1},
       {-1, 1, 7, 1}});
  InputDirectConvCudaParam(
      model_data,
      "CudaDirectConvSchedule 1 1152 1 1 48 1152 1 1 1 48 1 1",
      {{-1, 576},
       {-1, 1},
       {-1, 1},
       {-1, 1, 3, 1},
       {-1, 1, 1, 1},
       {-1, 1, 1, 1}});
  InputDirectConvCudaParam(
      model_data,
      "CudaDirectConvSchedule 1 48 1 1 1152 48 1 1 1 1152 1 1",
      {{-1, 12},
       {-1, 1},
       {-1, 1},
       {-1, 1, 32, 1},
       {-1, 1, 1, 1},
       {-1, 1, 1, 1}});
  InputDirectConvCudaParam(
      model_data,
      "CudaDirectConvSchedule 1 1152 7 7 192 1152 1 1 1 192 7 7",
      {{-1, 36},
       {-1, 1},
       {-1, 1},
       {-1, 1, 2, 6},
       {-1, 1, 7, 1},
       {-1, 1, 7, 1}});
  InputDirectConvCudaParam(
      model_data,
      "CudaDirectConvSchedule 1 1152 7 7 320 1152 1 1 1 320 7 7",
      {{-1, 12},
       {-1, 1},
       {-1, 1},
       {-1, 1, 2, 4},
       {-1, 1, 7, 1},
       {-1, 1, 7, 1}});

  // FaceDet schedule params
  /*   InputDirectConvCudaParam(model_data,
                                 "CudaDirectConvSchedule 1 3 242 322 16 3 3 3 1
     16 120 160",
                                 {{-1, 1}, {-1, 3}, {-1, 3}, {-1, 2, 4, 2}, {-1,
     1, 1, 5}, {-1, 1, 32, 1}}); */
  InputDirectConvCudaParam(
      model_data,
      "CudaDirectConvSchedule 1 16 120 160 32 16 1 1 1 32 120 160",
      {{-1, 4},
       {-1, 1},
       {-1, 1},
       {-1, 8, 4, 1},
       {-1, 1, 1, 1},
       {-1, 5, 32, 1}});
  InputDirectConvCudaParam(
      model_data,
      "CudaDirectConvSchedule 1 32 60 80 32 32 1 1 1 32 60 80",
      {{-1, 4},
       {-1, 1},
       {-1, 1},
       {-1, 8, 4, 1},
       {-1, 3, 1, 1},
       {-1, 1, 40, 1}});
  /*   InputDirectConvCudaParam(model_data,
                                 "CudaDirectConvSchedule 1 32 30 40 64 32 1 1 1
     64 30 40",
                                 {{-1, 8}, {-1, 1}, {-1, 1}, {-1, 2, 8, 2}, {-1,
     1, 1, 3}, {-1, 1, 20, 1}}); */
  InputDirectConvCudaParam(
      model_data,
      "CudaDirectConvSchedule 1 64 30 40 64 64 1 1 1 64 30 40",
      {{-1, 8}, {-1, 1}, {-1, 1}, {-1, 2, 8, 2}, {-1, 1, 2, 1}, {-1, 5, 8, 1}});
  InputDirectConvCudaParam(
      model_data,
      "CudaDirectConvSchedule 1 64 30 40 8 64 1 1 1 8 30 40",
      {{-1, 8}, {-1, 1}, {-1, 1}, {-1, 2, 4, 1}, {-1, 1, 2, 1}, {-1, 1, 8, 1}});
  /*   InputDirectConvCudaParam(model_data,
                                 "CudaDirectConvSchedule 1 8 32 42 12 8 3 3 1 12
    30 40",
                                 {{-1, 4}, {-1, 3}, {-1, 3}, {-1, 1, 12, 1},
    {-1, 1, 1, 3}, {-1, 1, 10, 1}}); InputDirectConvCudaParam(model_data,
                                 "CudaDirectConvSchedule 1 8 32 42 16 8 3 3 1 16
    30 40",
                                 {{-1, 8}, {-1, 3}, {-1, 3}, {-1, 1, 16, 1},
    {-1, 3, 1, 2}, {-1, 1, 4, 2}}); */
  InputDirectConvCudaParam(
      model_data,
      "CudaDirectConvSchedule 1 16 36 46 16 16 3 3 1 16 30 40",
      {{-1, 16},
       {-1, 1},
       {-1, 1},
       {-1, 2, 8, 1},
       {-1, 1, 2, 1},
       {-1, 1, 8, 1}});
  /*   InputDirectConvCudaParam(model_data,
                             "CudaDirectConvSchedule 1 16 34 44 16 16 3 3 1 16
     30 40",
                             {{-1, 4}, {-1, 3}, {-1, 3}, {-1, 1, 4, 2}, {-1, 3,
     2, 1}, {-1, 1, 20, 1}}); */
  /*   InputDirectConvCudaParam(model_data,
                                 "CudaDirectConvSchedule 1 12 32 42 16 12 3 3 1
     16 30 40",
                                 {{-1, 4}, {-1, 3}, {-1, 3}, {-1, 1, 16, 1},
     {-1, 1, 2, 3}, {-1, 1, 2, 2}}); */
  /*   InputDirectConvCudaParam(model_data,
                             "CudaDirectConvSchedule 1 16 40 50 16 16 3 3 1 16
     30 40",
                             {{-1, 4}, {-1, 1}, {-1, 3}, {-1, 1, 1, 8}, {-1, 1,
     3, 1}, {-1, 1, 40, 1}}); */
  /*   InputDirectConvCudaParam(model_data,
                               "CudaDirectConvSchedule 1 48 30 40 64 48 1 1 1 64
     30 40",
                               {{-1, 8}, {-1, 1}, {-1, 1}, {-1, 2, 8, 2}, {-1,
     1, 1, 3}, {-1, 1, 20, 1}}); */
  InputDirectConvCudaParam(
      model_data,
      "CudaDirectConvSchedule 1 64 30 40 12 64 1 1 1 12 30 40",
      {{-1, 8},
       {-1, 1},
       {-1, 1},
       {-1, 1, 4, 3},
       {-1, 1, 3, 1},
       {-1, 1, 10, 1}});
  InputDirectConvCudaParam(
      model_data,
      "CudaDirectConvSchedule 1 64 30 40 6 64 1 1 1 6 30 40",
      {{-1, 8},
       {-1, 1},
       {-1, 1},
       {-1, 3, 2, 1},
       {-1, 1, 3, 1},
       {-1, 1, 10, 1}});
  InputDirectConvCudaParam(
      model_data,
      "CudaDirectConvSchedule 1 64 15 20 128 64 1 1 1 128 15 20",
      {{-1, 8},
       {-1, 1},
       {-1, 1},
       {-1, 2, 8, 2},
       {-1, 1, 3, 1},
       {-1, 1, 10, 2}});
  InputDirectConvCudaParam(
      model_data,
      "CudaDirectConvSchedule 1 128 15 20 128 128 1 1 1 128 15 20",
      {{-1, 8},
       {-1, 1},
       {-1, 1},
       {-1, 4, 8, 1},
       {-1, 1, 3, 1},
       {-1, 1, 10, 2}});
  InputDirectConvCudaParam(
      model_data,
      "CudaDirectConvSchedule 1 128 15 20 8 128 1 1 1 8 15 20",
      {{-1, 8},
       {-1, 1},
       {-1, 1},
       {-1, 1, 8, 1},
       {-1, 1, 1, 1},
       {-1, 1, 10, 2}});
  InputDirectConvCudaParam(
      model_data,
      "CudaDirectConvSchedule 1 128 15 20 4 128 1 1 1 4 15 20",
      {{-1, 16},
       {-1, 1},
       {-1, 1},
       {-1, 1, 4, 1},
       {-1, 1, 1, 1},
       {-1, 1, 20, 1}});
  InputDirectConvCudaParam(
      model_data,
      "CudaDirectConvSchedule 1 128 8 10 256 128 1 1 1 256 8 10",
      {{-1, 16},
       {-1, 1},
       {-1, 1},
       {-1, 1, 16, 2},
       {-1, 1, 8, 1},
       {-1, 1, 2, 1}});
  InputDirectConvCudaParam(
      model_data,
      "CudaDirectConvSchedule 1 256 8 10 256 256 1 1 1 256 8 10",
      {{-1, 8}, {-1, 1}, {-1, 1}, {-1, 4, 8, 1}, {-1, 1, 8, 1}, {-1, 1, 2, 1}});
  InputDirectConvCudaParam(
      model_data,
      "CudaDirectConvSchedule 1 256 8 10 64 256 1 1 1 64 8 10",
      {{-1, 16},
       {-1, 1},
       {-1, 1},
       {-1, 1, 16, 1},
       {-1, 1, 8, 1},
       {-1, 2, 1, 1}});
  InputDirectConvCudaParam(
      model_data,
      "CudaDirectConvSchedule 1 256 8 10 8 256 1 1 1 8 8 10",
      {{-1, 32},
       {-1, 1},
       {-1, 1},
       {-1, 1, 8, 1},
       {-1, 1, 2, 1},
       {-1, 1, 2, 1}});
  InputDirectConvCudaParam(
      model_data,
      "CudaDirectConvSchedule 1 256 8 10 4 256 1 1 1 4 8 10",
      {{-1, 32},
       {-1, 1},
       {-1, 1},
       {-1, 1, 4, 1},
       {-1, 1, 4, 1},
       {-1, 1, 2, 1}});
  InputDirectConvCudaParam(
      model_data,
      "CudaDirectConvSchedule 1 64 4 5 256 64 1 1 1 256 4 5",
      {{-1, 16},
       {-1, 1},
       {-1, 1},
       {-1, 1, 8, 1},
       {-1, 1, 4, 1},
       {-1, 1, 5, 1}});
  InputDirectConvCudaParam(
      model_data,
      "CudaDirectConvSchedule 1 256 6 7 12 256 3 3 1 12 4 5",
      {{-1, 32},
       {-1, 3},
       {-1, 3},
       {-1, 1, 4, 1},
       {-1, 1, 4, 1},
       {-1, 1, 1, 1}});
  InputDirectConvCudaParam(model_data,
                           "CudaDirectConvSchedule 1 256 6 7 6 256 3 3 1 6 4 5",
                           {{-1, 32},
                            {-1, 3},
                            {-1, 3},
                            {-1, 1, 2, 1},
                            {-1, 1, 4, 1},
                            {-1, 1, 1, 1}});

#ifndef CINN_WITH_CUDNN
  InputWinogradConvCudaParam(
      model_data,
      "CudaWinogradConvSchedule 1 512 9 9 512 512 3 3 1 512 7 7",
      {{32, 16}, {1, 1, 8, 2}, {8, 1, 16, 4}, {16, 1, 1, 1}});
  InputWinogradConvCudaParam(
      model_data,
      "CudaWinogradConvSchedule 1 256 6 7 12 256 3 3 1 12 4 5",
      {{-1, 256}, {-1, 1, 6, 1}, {-1, 1, 6, 1}, {-1, 1, 1, 1}});
  InputWinogradConvCudaParam(
      model_data,
      "CudaWinogradConvSchedule 1 256 6 7 6 256 3 3 1 12 4 5",
      {{-1, 256}, {-1, 1, 6, 1}, {-1, 1, 6, 1}, {-1, 1, 1, 1}});
  InputWinogradConvCudaParam(
      model_data,
      "CudaWinogradConvSchedule 1 12 32 42 16 12 3 3 1 16 30 40",
      {{-1, 12}, {-1, 2, 30, 1}, {-1, 4, 2, 2}, {-1, 1, 1, 1}});
  InputWinogradConvCudaParam(
      model_data,
      "CudaWinogradConvSchedule 1 8 32 42 12 8 3 3 1 12 30 40",
      {{-1, 8}, {-1, 2, 30, 1}, {-1, 1, 2, 6}, {-1, 1, 1, 1}});
  InputWinogradConvCudaParam(
      model_data,
      "CudaWinogradConvSchedule 1 8 32 42 16 8 3 3 1 16 30 40",
      {{-1, 4}, {-1, 2, 30, 1}, {-1, 1, 4, 4}, {-1, 1, 1, 1}});
#endif
  return model_data;
}

void CreateCudaSerialData(const std::string &file_name) {
  SaveSerialData(CreateCudaParams(), file_name);
}

int GetMaxSplitter(int a, int b) {
  while (a % b > 0) {
    b--;
  }
  return b;
}

void LoadSerialData(
    absl::flat_hash_map<std::string,
                        absl::flat_hash_map<std::string, std::vector<int>>>
        *params,
    const std::string &file_name) {
  proto::ModelData read_model_data;
  std::fstream input(file_name, std::ios::in | std::ios::binary);
  if (!read_model_data.ParseFromIstream(&input)) {
    std::cerr << "Failed to parse address book." << std::endl;
    exit(-1);
  }
  input.close();
  std::string test_write3;
  read_model_data.SerializeToString(&test_write3);
  auto read_model_map = read_model_data.data();
  for (auto &i : read_model_map) {
    auto read_schedule_map = i.second.data();
    absl::flat_hash_map<std::string, std::vector<int>> param_data;
    for (auto &j : read_schedule_map) {
      std::vector<int> temp_data;
      for (int k = 0; k < j.second.data_size(); k++) {
        temp_data.push_back(std::stoi(j.second.data(k)));
      }
      param_data[j.first] = temp_data;
    }
    (*params)[i.first] = param_data;
  }
}

void SaveSerialData(
    const absl::flat_hash_map<
        std::string,
        absl::flat_hash_map<std::string, std::vector<int>>> &model_data,
    const std::string &file_name) {
  proto::ModelData write_model_data;
  for (auto &i : model_data) {
    proto::ScheduleData write_schedule_data;
    for (auto &j : i.second) {
      proto::StringData write_vector_data;
      for (auto &k : j.second) {
        write_vector_data.add_data(std::to_string(k));
      }
      auto data_map = write_schedule_data.mutable_data();
      (*data_map)[j.first] = write_vector_data;
    }
    auto model_map = write_model_data.mutable_data();
    (*model_map)[i.first] = write_schedule_data;
    std::string test_write1;
    write_schedule_data.SerializeToString(&test_write1);
  }
  std::fstream output(file_name,
                      std::ios::out | std::ios::trunc | std::ios::binary);
  std::string test_write;
  write_model_data.SerializeToString(&test_write);
  if (!write_model_data.SerializeToOstream(&output)) {
    std::cerr << "Failed to write test_serial.log" << std::endl;
    exit(-1);
  }
  output.close();
}

int gcd(int a, int b) {
  int r;
  while (b > 0) {
    r = a % b;
    a = b;
    b = r;
  }
  return a;
}

int MaxFactorLessThan(int a, int b) {
  PADDLE_ENFORCE_GT(
      a,
      b,
      ::common::errors::InvalidArgument(
          "The first argument should be greater than the second argument"));
  int res = 1;
  for (int i = 2; i <= static_cast<int>(sqrt(static_cast<double>(a))); i++) {
    if (a % i == 0) {
      if (i <= b) res = std::max(res, i);
      if (a / i <= b) res = std::max(res, a / i);
    }
  }
  return res;
}

void CudaScheduleInjectiveWithVectorize(poly::Stage *stage,
                                        const std::vector<int> &output_shape,
                                        const cinn::common::Target &target) {
  int dims = stage->n_out_dims();
  int prod_size = std::accumulate(
      output_shape.begin(), output_shape.end(), 1, std::multiplies<int>());
  int num_thread = target.max_num_threads();
  int last_shape = stage->GetDimRange(stage->n_out_dims() - 1);
  // determine the factor of vectorize
  int vector_width = 1;
  if (last_shape % 4 == 0) {
    vector_width = 4;
  } else if (last_shape % 2 == 0) {
    vector_width = 2;
  }

  // print range of stage for debug
  auto range_str_fn = [stage]() {
    std::vector<int> dim_ranges;
    for (int i = 0; i < stage->n_out_dims(); ++i) {
      dim_ranges.push_back(stage->GetDimRange(i));
    }
    std::string res = "[" + utils::Join(dim_ranges, ",") + "]";
    return res;
  };

  // the first bind position from tail
  int bind_idx = stage->n_out_dims() - 1;
  // it will add a new dim by split before vectorize, but the new dim will
  // be eliminated when vectorizing, so the bind_idx doesn't need to increase
  if (vector_width > 1) {
    stage->Split(bind_idx, vector_width);
  }
  VLOG(5) << "vectorize result:" << range_str_fn();

  // revise dim for binding threadIdx.x, here only use the x of threadIdx
  if (stage->GetDimRange(bind_idx) > num_thread) {
    stage->Split(bind_idx, gcd(stage->GetDimRange(bind_idx), num_thread));
    ++bind_idx;
  }
  while (bind_idx > 0 &&
         stage->GetDimRange(bind_idx - 1) * stage->GetDimRange(bind_idx) <
             num_thread) {
    stage->Fuse(bind_idx - 1, bind_idx);
    --bind_idx;
  }
  // call vectorize on the last dim
  if (vector_width > 1) {
    stage->Vectorize(stage->n_out_dims() - 1, vector_width);
  }
  stage->Bind(bind_idx, "threadIdx.x");
  --bind_idx;
  VLOG(5) << "bind threadIdx.x result:" << range_str_fn();

  // revise dim for binding blockIdx, at most 3 indexes can be used
  while (bind_idx > 2) {
    stage->Fuse(bind_idx - 1, bind_idx);
    --bind_idx;
  }
  std::string block_idx = "blockIdx.x";
  for (int j = 0; bind_idx >= 0; ++j) {
    block_idx.back() = 'x' + j;
    stage->Bind(bind_idx, block_idx);
    --bind_idx;
  }
  VLOG(5) << "CudaScheduleInjectiveWithVectorize tensor:"
          << stage->tensor()->name << ", vector_width:" << vector_width
          << ", prod_size:" << prod_size << ", shape:["
          << utils::Join(output_shape, ",") << "]"
          << ", range:" << range_str_fn();
}

void CudaScheduleInjective(poly::Stage *stage,
                           const std::vector<int> &output_shape,
                           const cinn::common::Target &target) {
  PADDLE_ENFORCE_EQ(
      stage->n_out_dims(),
      stage->n_in_dims(),
      ::common::errors::InvalidArgument("The dims of op are not equal"));
  if (FLAGS_cinn_use_cuda_vectorize) {
    CudaScheduleInjectiveWithVectorize(stage, output_shape, target);
    return;
  }
  int dims = stage->n_out_dims();
  for (int i = 1; i < dims; i++) {
    stage->Fuse(0, 1);
  }

  int num_thread = target.max_num_threads();
  int prod_size = std::accumulate(
      output_shape.begin(), output_shape.end(), 1, std::multiplies<int>());
  if (prod_size <= num_thread) {
    stage->Bind(0, "threadIdx.x");
    return;
  }
  int new_num_thread = gcd(prod_size, num_thread);
  if (new_num_thread % 32 != 0) {
    new_num_thread = MaxFactorLessThan(prod_size, num_thread);
  }
  if (new_num_thread == 1) {
    std::stringstream ss;
    ss << "prod_size out of range: " << prod_size;
    PADDLE_THROW(::common::errors::InvalidArgument(ss.str()));
  }

  PADDLE_ENFORCE_GT(prod_size,
                    new_num_thread,
                    ::common::errors::InvalidArgument(
                        "The prod_size should be greater than new_num_thread"));
  stage->Split(0, new_num_thread);
  stage->Bind(0, "blockIdx.x");
  stage->Bind(1, "threadIdx.x");
}

}  // namespace pe
}  // namespace hlir
}  // namespace cinn
