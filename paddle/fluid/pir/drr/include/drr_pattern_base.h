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

#include <memory>
#include <string>

#include "paddle/fluid/pir/drr/include/drr_match_context.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_context.h"
#include "paddle/fluid/pir/drr/include/drr_rewrite_pattern.h"

namespace pir {
class IrContext;
}

namespace paddle {
namespace drr {

class DrrPatternBase : public std::enable_shared_from_this<DrrPatternBase> {
 public:
  virtual ~DrrPatternBase() = default;

  // Define the drr pattern.
  virtual void operator()(drr::DrrPatternContext* ctx) const = 0;

  // Give the drr pattern name.
  virtual std::string name() const = 0;

  // Give the drr pattern benefit.
  virtual uint32_t benefit() const { return 1; }
};

template <typename T, typename... Args>
static std::unique_ptr<DrrRewritePattern> Create(pir::IrContext* ir_context,
                                                 Args&&... args) {
  auto drr_pattern = std::make_shared<T>(std::forward<Args>(args)...);
  DrrPatternContext drr_context;
  drr_pattern->operator()(&drr_context);
  return std::make_unique<DrrRewritePattern>(drr_pattern->name(),
                                             drr_context,
                                             ir_context,
                                             drr_pattern->benefit(),
                                             drr_pattern->shared_from_this());
}

}  // namespace drr
}  // namespace paddle
