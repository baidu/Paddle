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

#include <string>
#include <vector>

namespace paddle {
namespace dialect {
struct PdOpSig {
  std::string name;
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
  PdOpSig() = default;
  PdOpSig(const PdOpSig& input_info) = default;

  PdOpSig(const std::string& name,
          const std::vector<std::string>& inputs,
          const std::vector<std::string>& outputs)
      : name(name), inputs(inputs), outputs(outputs) {}
};

bool HaveOpToMultiKernelsMap(std::string op_name);

const std::vector<PdOpSig>& LegacyOpToPdOpsMapping(std::string op_name);
const std::vector<PdOpSig>& SparseOpToPdOpsMapping(std::string op_name);

#ifdef PADDLE_WITH_DNNL
bool IsOneDNNOnlyOp(std::string op_name);
#endif

}  // namespace dialect
}  // namespace paddle
