// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/impl/load_combine_kernel_impl.h"

PD_REGISTER_KERNEL(load_combine,
                   XPU,
                   ALL_LAYOUT,
                   phi::LoadCombineKernel,
                   float,
                   double,
                   int,
                   int8_t,
                   int64_t) {}

PD_REGISTER_KERNEL(load_combine_vocab,
                   XPU,
                   ALL_LAYOUT,
                   phi::LoadCombineVocabKernel,
                   float,
                   double,
                   int,
                   int8_t,
                   int64_t) {}

PD_REGISTER_KERNEL(load_combine_extended,
                   XPU,
                   ALL_LAYOUT,
                   phi::LoadCombineExtendedKernel,
                   float,
                   double,
                   int,
                   int8_t,
                   int64_t) {}
