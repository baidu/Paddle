/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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

#include <sys/time.h>
#include <iostream>
#include <string>
#include <vector>

#include "brpc/channel.h"
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/framework/var_type.h"
#include "paddle/fluid/operators/detail/send_recv.pb.h"
#include "paddle/fluid/operators/detail/sendrecvop_utils.h"

namespace paddle {
namespace operators {
namespace detail {

void SerializeToIOBuf(const std::string& name, framework::Variable* var,
                      const platform::DeviceContext& ctx, VarMsg* request,
                      butil::IOBuf* iobuf,
                      const std::string& out_varname = std::string());

void DeserializeFromIOBuf(const butil::IOBuf& iobuf,
                          const platform::DeviceContext& ctx,
                          const framework::Scope* scope,
                          framework::Variable** var);

}  // namespace detail
}  // namespace operators
}  // namespace paddle
