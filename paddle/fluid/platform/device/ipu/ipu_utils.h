/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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

#include <popart/ndarraywrapper.hpp>
#include <popart/session.hpp>
#include <popart/tensordata.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/vendored/any.hpp>
#include <popart/vendored/optional.hpp>
#include <popef/Model.hpp>
#include <popef/Reader.hpp>
#include <popef/Types.hpp>

#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/phi/common/float16.h"

using float16 = phi::dtype::float16;
using Scope = paddle::framework::Scope;
using OpDesc = paddle::framework::OpDesc;
using Graph = paddle::framework::ir::Graph;
using Node = paddle::framework::ir::Node;
using BlockDesc = paddle::framework::BlockDesc;
using VarType = paddle::framework::proto::VarType;

namespace paddle {
namespace platform {
namespace ipu {

template <typename T>
T GetSingleVarFromScope(const Scope* scope, const std::string& var_name) {
  auto var = scope->GetVar(var_name);
  auto tensor = var->Get<phi::DenseTensor>();
  return tensor.data<T>()[0];
}

struct IpuCustomOpIdentifier {
  IpuCustomOpIdentifier(const std::string& _paddle_op,
                        const std::string& _popart_op,
                        const std::string& _domain,
                        unsigned int _version)
      : paddle_op(_paddle_op), popart_op(_domain, _popart_op, _version) {}

  std::string repr() {
    std::ostringstream os;
    os << "paddle_op: " << paddle_op << ", domain: " << popart_op.domain
       << ", type: " << popart_op.type << ", version: " << popart_op.version;
    return os.str();
  }

  std::string paddle_op;
  popart::OperatorIdentifier popart_op;
};

// Onnx dtype
// https://github.com/onnx/onnx/blob/master/onnx/onnx-ml.proto3
enum ONNXDataType : int {
  UNDEFINED = 0,
  FLOAT = 1,
  UINT8 = 2,
  INT8 = 3,
  UINT16 = 4,
  INT16 = 5,
  INT32 = 6,
  INT64 = 7,
  STRING = 8,
  BOOL = 9,
  FLOAT16 = 10,
  DOUBLE = 11,
  UINT32 = 12,
  UINT64 = 13,
  COMPLEX64 = 14,
  COMPLEX128 = 15,
  BFLOAT16 = 16
};

// TODO(czr): remove const qualifier on return value.
const VarType::Type PopefDType2VarType(const popef::DataType type);
const phi::DataType PopefDtype2PhiDtype(const popef::DataType type);

// VarType::Type to popart::DataType
const popart::DataType VarType2PopartDType(const VarType::Type type);
// phi::DataType to popart::DataType
const popart::DataType PhiDType2PopartDType(const phi::DataType type);
// popart::DataType to VarType::Type
const VarType::Type PopartDType2VarType(const popart::DataType type);
// ONNXDataType to popart::DataType
const popart::DataType OnnxDType2PopartType(const ONNXDataType type);
// VarType::Type to ONNXDataType
const ONNXDataType VarType2OnnxDType(const VarType::Type type);
// VarType::Type to String in Popart
const std::string VarType2PopartStr(const VarType::Type type);
// Get bool from environment variable
const bool GetBoolEnv(const std::string& str);
// Request number of ipus must be pow(2, n)
const int RequestIpus(const int num_ipus);

// Convert popart session to popef
std::shared_ptr<popef::Model> PopartSessionToPopefModel(
    popart::Session* session);

}  // namespace ipu
}  // namespace platform
}  // namespace paddle
