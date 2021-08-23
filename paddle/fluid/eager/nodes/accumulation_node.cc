// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/eager/nodes/accumulation_node.h"
#include "paddle/fluid/eager/function_api.h"

#include "paddle/top/api/all.h"

#include "paddle/fluid/platform/device_context.h"

#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/errors.h"

#include "glog/logging.h"

static void CopyOrAddTensor(pt::Tensor& tensor, const pt::Tensor& t) {
    if(!tensor.defined() || !tensor.initialized()) {
        // Simply copy tensor->impl
        tensor = t;
    } else {
        // Accumulation
        egr::AccumulateTensorsAPI(tensor, t);
    }
}

namespace egr {
  
void GradNodeAccumulation::SetTensorWrappers(const std::vector<pt::Tensor>& tensors) {
    // Does nothing
}

void GradNodeAccumulation::RetainGrad(std::function<pt::Tensor(const pt::Tensor&)>& hook) {
    retain_grad_hook = hook;
}

std::vector<pt::Tensor> GradNodeAccumulation::operator()(const std::vector<pt::Tensor>& grads) {
    PADDLE_ENFORCE(grads.size() == 1,
                paddle::platform::errors::Fatal("GradNodeAccumulation should take exactly 1 grad tensor"
                                                "However received: %d", grads.size()));
    // Apply Gradient Hooks
    if(GradientHooksRegistered()) {
        std::vector<pt::Tensor> hooked_grads = ApplyGradientHooks(grads);
        CopyOrAddTensor(accumulated_grad, hooked_grads[0]);
    } else {
        CopyOrAddTensor(accumulated_grad, grads[0]);
    }
    
    // Apply Retain Grad Hook
    if(retain_grad_hook != nullptr) {
        retain_grad_hook(accumulated_grad);
    }
    
    // Apply Reduce Hooks
    if(ReduceHooksRegistered()) {
        ApplyReduceHooks();
    }
        
    return { accumulated_grad };
}

} // namespace egr
