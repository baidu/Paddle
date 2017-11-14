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

#include "paddle/operators/compare_op.h"

REGISTER_LOGICAL_KERNEL(less_than, GPU, paddle::operators::LessThanFunctor);
REGISTER_LOGICAL_KERNEL(less_equal, GPU,
                        paddle::operators::LessThanEqualFunctor);
REGISTER_LOGICAL_KERNEL(greater_than, GPU,
                        paddle::operators::GreaterThanFunctor);
REGISTER_LOGICAL_KERNEL(greater_equal, GPU,
                        paddle::operators::GreaterThanEqualFunctor);
REGISTER_LOGICAL_KERNEL(equal, GPU, paddle::operators::EqualFunctor);
