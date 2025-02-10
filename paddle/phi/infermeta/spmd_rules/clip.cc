// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/infermeta/spmd_rules/clip.h"
#include "glog/logging.h"

namespace phi {
namespace distributed {

SpmdInfo ClipInferSpmd(const DistMetaTensor& x,
                       const Scalar& min,
                       const Scalar& max) {
  VLOG(4) << "ClipInferSpmd Call ElementwiseUnaryInferSpmd";
  return ElementwiseUnaryInferSpmd(x);
}

SpmdInfo ClipInferSpmdReverse(const DistMetaTensor& x,
                              const DistMetaTensor& out,
                              const Scalar& min,
                              const Scalar& max) {
  VLOG(4) << "ClipInferSpmdReverse Call ElementwiseUnaryInferSpmdReverse";
  return ElementwiseUnaryInferSpmdReverse(x, out);
}

SpmdInfo ClipGradInferSpmd(const DistMetaTensor& x,
                           const DistMetaTensor& out_grad,
                           const Scalar& min,
                           const Scalar& max) {
  VLOG(4) << "ClipGradInferSpmd Call ElementwiseUnaryGradInferSpmd";
  return ElementwiseUnaryGradInferSpmd(x, out_grad);
}

}  // namespace distributed
}  // namespace phi
