/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve.

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

#include <paddle/parameter/Argument.h>

#include <caffe/layer.hpp>
#include <caffe/blob.hpp>

namespace paddle {

std::vector<int> ArgShape2Vector(const Argument& arg);
void Vector2ArgShape(const Argument& arg, const std::vector<int>& vec);

void SetDataToBlob(Argument& arg, ::caffe::Blob<real>* blob);
void SetGradToBlob(Argument& arg, ::caffe::Blob<real>* blob);

}  // namespace paddle
