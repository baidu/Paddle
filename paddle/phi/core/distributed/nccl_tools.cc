// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/distributed/nccl_tools.h"

#include <unordered_map>

#include "paddle/common/errors.h"
#include "paddle/phi/core/enforce.h"

#if NCCL_VERSION_CODE >= 21300
#define ENABLE_NCCL_GET_LAST_ERROR
#define NCCL_REMOTE_ERROR
#endif

namespace phi::distributed {

ncclRedOp_t ToNCCLRedType(ReduceOp reduction) {
  static const std::unordered_map<ReduceOp, ncclRedOp_t> red_type = {
    {ReduceOp::MIN, ncclMin},
    {ReduceOp::MAX, ncclMax},
    {ReduceOp::SUM, ncclSum},
    {ReduceOp::PRODUCT, ncclProd},
#if NCCL_VERSION_CODE >= 21000
    {ReduceOp::AVG, ncclAvg},
#endif
  };
  auto it = red_type.find(reduction);
  PADDLE_ENFORCE_EQ(it != red_type.end(),
                    true,
                    common::errors::InvalidArgument(
                        "Invalid nccl reduction. Must be ncclMin | ncclMax | "
                        "ncclProd | ncclSum | ncclAvg."));
  return it->second;
}

std::string SerializeNCCLUniqueId(const ncclUniqueId& ncclID) {
  const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&ncclID);
  std::ostringstream oss;
  for (auto i = 0; i < NCCL_UNIQUE_ID_BYTES; ++i) {
    oss << std::hex << static_cast<int>(bytes[i]);
  }
  return oss.str();
}

std::string NCCLDTypeToString(ncclDataType_t dtype) {
#define PD_NCCL_DTYPE_TO_STR(__nccl_dtype, __str_dtype) \
  if (dtype == __nccl_dtype) return __str_dtype;
  PD_NCCL_DTYPE_TO_STR(ncclFloat, "float32");
  PD_NCCL_DTYPE_TO_STR(ncclFloat32, "float32");
  PD_NCCL_DTYPE_TO_STR(ncclHalf, "float16");
  PD_NCCL_DTYPE_TO_STR(ncclFloat16, "float16");
#if (NCCL_VERSION_CODE >= 21000 && CUDA_VERSION >= 11000) || \
    defined(PADDLE_WITH_HIP)
  PD_NCCL_DTYPE_TO_STR(ncclBfloat16, "bfloat16");
#endif
  PD_NCCL_DTYPE_TO_STR(ncclDouble, "float64");
  PD_NCCL_DTYPE_TO_STR(ncclFloat64, "float64");

  PD_NCCL_DTYPE_TO_STR(ncclInt8, "int8");
  PD_NCCL_DTYPE_TO_STR(ncclChar, "int8");
  PD_NCCL_DTYPE_TO_STR(ncclUint8, "uint8");
  PD_NCCL_DTYPE_TO_STR(ncclInt32, "int32");
  PD_NCCL_DTYPE_TO_STR(ncclInt, "int32");
  PD_NCCL_DTYPE_TO_STR(ncclUint32, "uint32");
  PD_NCCL_DTYPE_TO_STR(ncclInt64, "int64");
  PD_NCCL_DTYPE_TO_STR(ncclUint64, "uint64");

#undef PD_NCCL_DTYPE_TO_STR
  PADDLE_THROW(common::errors::InvalidArgument(
      "This datatype %d in nccl is not supported.", static_cast<int>(dtype)));
}

std::string NCCLRedTypeToString(ncclRedOp_t op) {
  if (op == ncclSum) return "SUM";
  if (op == ncclProd) return "PROD";
  if (op == ncclMin) return "MIN";
  if (op == ncclMax) return "MAX";
#if NCCL_VERSION_CODE >= 21000
  if (op == ncclAvg) return "AVG";
#endif
  return "UDF_" + std::to_string(op);
}

}  // namespace phi::distributed
