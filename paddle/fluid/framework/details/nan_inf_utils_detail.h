// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include <string>
#include "paddle/common/flags.h"
#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/phi/common/complex.h"
#include "paddle/phi/core/platform/device_context.h"
#include "paddle/phi/kernels/check_numerics_kernel.h"
#include "paddle/phi/kernels/funcs/eigen/extensions.h"

COMMON_DECLARE_int32(check_nan_inf_level);

namespace paddle {
namespace framework {
namespace details {

void SetNanInfDebugPath(const std::string& nan_inf_path);

std::string GetNanPath();

void SetNanInfStackLimit(const int& stack_limit);

int GetNanInfStackLimit();

template <typename Context>
struct TensorCheckerVisitor {
  TensorCheckerVisitor(const std::string& o,
                       const std::string& v,
                       const phi::DenseTensor& t,
                       const phi::Place& p)
      : op_type(o), var_name(v), tensor(t), place(p) {}

  template <typename T>
  void apply(
      typename std::enable_if<std::is_integral<T>::value>::type* = 0) const {
    VLOG(10) << var_name << " need not to check, it's type is not float point";
  }

  template <typename T>
  void apply(typename std::enable_if<
                 std::is_floating_point<T>::value ||
                 std::is_same<T, ::phi::dtype::complex<float>>::value ||
                 std::is_same<T, ::phi::dtype::complex<double>>::value>::type* =
                 0) const {
    auto* dev_ctx = reinterpret_cast<Context*>(
        phi::DeviceContextPool::Instance().Get(tensor.place()));

    phi::DenseTensor stats;
    phi::DenseTensor values;
    auto file_path = GetNanPath();
    phi::CheckNumericsKernel<T, Context>(*dev_ctx,
                                         tensor,
                                         op_type,
                                         var_name,
                                         FLAGS_check_nan_inf_level,
                                         GetNanInfStackLimit(),
                                         file_path,
                                         &stats,
                                         &values);
  }

  std::string op_type;
  std::string var_name;
  const phi::DenseTensor& tensor;
  const phi::Place& place;
};

template <typename Context>
void tensor_check(const std::string& op_type,
                  const std::string& var_name,
                  const phi::DenseTensor& tensor,
                  const phi::Place& place) {
  TensorCheckerVisitor<Context> visitor(op_type, var_name, tensor, place);
  framework::VisitDataType(framework::TransToProtoVarType(tensor.dtype()),
                           visitor);
}

void InitWhiteListFormEnv();
std::unordered_set<std::string>& op_type_nan_inf_white_list();
std::unordered_map<std::string, std::vector<std::string>>&
op_var_nan_inf_white_list();
}  // namespace details
}  // namespace framework
}  // namespace paddle
