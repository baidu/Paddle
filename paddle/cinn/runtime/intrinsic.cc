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

#include "paddle/cinn/runtime/intrinsic.h"

#include "paddle/cinn/common/common.h"
#include "paddle/cinn/ir/ir.h"

namespace cinn {
namespace runtime {

using cinn::common::bfloat16;
using cinn::common::float16;

cinn_type_t ToRuntimeType(Type type) {
#define SET_TYPE_CASE_ITEM(compiled_type, runtime_type) \
  if (type == cinn::common::compiled_type()) {          \
    return runtime_type();                              \
  }

  SET_TYPE_CASE_ITEM(Bool, cinn_bool_t)

  SET_TYPE_CASE_ITEM(I8, cinn_int8_t)
  SET_TYPE_CASE_ITEM(I16, cinn_int16_t)
  SET_TYPE_CASE_ITEM(I32, cinn_int32_t)
  SET_TYPE_CASE_ITEM(I64, cinn_int64_t)

  SET_TYPE_CASE_ITEM(UI8, cinn_uint8_t)
  SET_TYPE_CASE_ITEM(UI16, cinn_uint16_t)
  SET_TYPE_CASE_ITEM(UI32, cinn_uint32_t)
  SET_TYPE_CASE_ITEM(UI64, cinn_uint64_t)

  SET_TYPE_CASE_ITEM(F32, cinn_float32_t)
  SET_TYPE_CASE_ITEM(F64, cinn_float64_t)

  SET_TYPE_CASE_ITEM(Float(32).PointerOf, cinn_type_of<float*>);
  SET_TYPE_CASE_ITEM(Float(64).PointerOf, cinn_type_of<double*>);

  std::stringstream ss;
  ss << "Not supported type " << type;
  PADDLE_THROW(::common::errors::InvalidArgument(ss.str()));
  return cinn_unk_t();
#undef SET_TYPE_CASE_ITEM
}

Expr IntrinsicCall(Type type,
                   const std::string& fn_name,
                   const std::vector<Expr>& args,
                   const std::vector<Expr>& write_args) {
  return ir::Call::Make(type,
                        fn_name,
                        args,
                        write_args,
                        ir::CallType::Intrinsic,
                        ir::FunctionRef(),
                        0);
}

}  // namespace runtime
}  // namespace cinn
