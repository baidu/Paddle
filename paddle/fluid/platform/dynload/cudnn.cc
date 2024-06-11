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

#include "paddle/fluid/platform/dynload/cudnn.h"

#include "paddle/phi/backends/dynload/cudnn.h"

namespace paddle::platform::dynload {

#define DEFINE_WRAP(__name) DynLoad__##__name __name

CUDNN_DNN_ROUTINE_EACH(DEFINE_WRAP);

#ifdef CUDNN_DNN_ROUTINE_EACH_AFTER_R7_LESS_R8
CUDNN_DNN_ROUTINE_EACH_AFTER_R7_LESS_R8(DEFINE_WRAP);
#endif

#ifdef CUDNN_DNN_ROUTINE_EACH_R7
CUDNN_DNN_ROUTINE_EACH_R7(DEFINE_WRAP);
#endif

#ifdef CUDNN_DNN_ROUTINE_EACH_AFTER_TWO_R7
CUDNN_DNN_ROUTINE_EACH_AFTER_TWO_R7(DEFINE_WRAP);
#endif

#ifdef CUDNN_DNN_ROUTINE_EACH_AFTER_R7
CUDNN_DNN_ROUTINE_EACH_AFTER_R7(DEFINE_WRAP);
#endif

#ifdef CUDNN_DNN_ROUTINE_EACH_R8
CUDNN_DNN_ROUTINE_EACH_R8(DEFINE_WRAP);
#endif

#ifdef CUDNN_DNN_ROUTINE_EACH_REMOVED_IN_E9
CUDNN_DNN_ROUTINE_EACH_REMOVED_IN_E9(DEFINE_WRAP);
#endif

#ifdef CUDNN_DNN_ROUTINE_EACH_AFTER_TWO_R7_REMOVED_IN_E9
CUDNN_DNN_ROUTINE_EACH_AFTER_TWO_R7_REMOVED_IN_E9(DEFINE_WRAP);
#endif

#ifdef CUDNN_DNN_ROUTINE_EACH_R9
CUDNN_DNN_ROUTINE_EACH_R9(DEFINE_WRAP);
#endif

bool HasCUDNN() { return phi::dynload::HasCUDNN(); }

}  // namespace paddle::platform::dynload
