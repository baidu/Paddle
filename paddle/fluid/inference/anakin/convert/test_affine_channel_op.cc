/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <gtest/gtest.h>
#include "paddle/fluid/inference/anakin/convert/affine_channel.h"
#include "paddle/fluid/inference/anakin/convert/op_converter.h"
#include "paddle/fluid/inference/anakin/convert/ut_helper.h"

namespace paddle {
namespace inference {
namespace anakin {

TEST(affine_channel, native) {
  // Declare the difference between the inputs.
  std::unordered_set<std::string> parameters({"scale", "bias"});

  framework::Scope scope;
  AnakinConvertValidation validator(parameters, &scope);
  validator.DeclInputVar("x", {1, 3, 5, 2});
  validator.DeclOutputVar("out", {1, 3, 5, 2});
  validator.DeclParamVar("scale", {1, 3, 1, 1});
  validator.DeclParamVar("bias", {1, 3, 1, 1});

  // Prepare Op descriptions.
  framework::OpDesc desc;
  desc.SetType("affine_channel");
  desc.SetInput("X", {"x"});
  desc.SetInput("Bias", {"bias"});
  desc.SetInput("Scale", {"scale"});
  desc.SetOutput("Out", {"out"});

  // Layout must be explicitly specified here as NCHW.
  desc.SetAttr("data_layout", std::string("NCHW"));

  validator.SetOp(*desc.Proto());
  validator.Execute(1);
}

}  // namespace anakin
}  // namespace inference
}  // namespace paddle

USE_OP(affine_channel);
USE_ANAKIN_CONVERTER(affine_channel);
