// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/ir/dialect/pd_type_storage.h"
#include "paddle/ir/core/builtin_attribute.h"
#include "paddle/ir/core/builtin_type.h"

namespace paddle {
namespace dialect {

struct OpInputInfo {
  std::string name;
  std::string type_name;
  bool optional = false;
  bool no_need_buffer = false;
  bool is_mutable_attribute = false;
  OpInputInfo() {}
  OpInputInfo(const OpInputInfo& input_info)
      : name(input_info.name),
        type_name(input_info.type_name),
        optional(input_info.optional),
        no_need_buffer(input_info.no_need_buffer),
        is_mutable_attribute(input_info.is_mutable_attribute) {}

  OpInputInfo(std::string name,
              std::string type_name,
              bool optional,
              bool no_need_buffer,
              bool is_mutable_attribute)
      : name(name),
        type_name(type_name),
        optional(optional),
        no_need_buffer(no_need_buffer),
        is_mutable_attribute(is_mutable_attribute) {}
};

struct OpOutputInfo {
  std::string name;
  std::string type_name;
  bool optional = false;
  bool intermediate = false;
  OpOutputInfo() {}
  OpOutputInfo(const OpOutputInfo& output_info)
      : name(output_info.name),
        type_name(output_info.type_name),
        optional(output_info.optional),
        intermediate(output_info.intermediate) {}
  OpOutputInfo(std::string name,
               std::string type_name,
               bool optional,
               bool intermediate)
      : name(name),
        type_name(type_name),
        optional(optional),
        intermediate(intermediate) {}
};

struct OpAttributeInfo {
  std::string name;
  std::string type_name;
  std::string data_type;
  OpAttributeInfo() {}
  OpAttributeInfo(const OpAttributeInfo& attr_info)
      : name(attr_info.name),
        type_name(attr_info.type_name),
        data_type(attr_info.data_type) {}
  OpAttributeInfo(std::string name,
                  std::string type_name,
                  std::string data_type)
      : name(name), type_name(type_name), data_type(data_type) {}
};

struct OpRunTimeInfo {
  std::string infer_meta_func;
  std::vector<std::string> infer_meta_param;
  std::vector<std::string> kernel_func;
  std::vector<std::string> kernel_param;
  std::vector<std::string> kernel_key_dtype;
  std::vector<std::pair<std::string, std::string>> inplace;
  std::vector<std::pair<std::string, std::string>> view;
  OpRunTimeInfo(std::string infer_meta_func,
                std::vector<std::string> infer_meta_param,
                std::vector<std::string> kernel_func,
                std::vector<std::string> kernel_param,
                std::vector<std::string> dtype,
                std::vector<std::pair<std::string, std::string>> inplace,
                std::vector<std::pair<std::string, std::string>> view)
      : infer_meta_func(infer_meta_func),
        infer_meta_param(infer_meta_param),
        kernel_func(kernel_func),
        kernel_param(kernel_param),
        kernel_key_dtype(dtype),
        inplace(inplace),
        view(view) {}
};

}  // namespace dialect
}  // namespace paddle
