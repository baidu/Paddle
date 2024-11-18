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

#pragma once

#include "paddle/cinn/operator_fusion/utils.h"

namespace cinn::fusion {

struct ShardableAxes {
  ShardableAxes() : axis_names({}) {}
  explicit ShardableAxes(const std::vector<std::string>& names)
      : axis_names(names) {}
  std::vector<std::string> axis_names;
  std::string DebugStr() const;
};

struct ShardableAxesSignature {
  ShardableAxes loop;
  size_t reduce_size;
  std::vector<ShardableAxes> inputs;
  std::vector<ShardableAxes> outputs;
  std::string DebugStr() const;
};

struct ShardableAxesInfoManager {
  ShardableAxesInfoManager(const std::vector<pir::Operation*>& ops,
                           pir::ShapeConstraintIRAnalysis* shape_analysis);
  ShardableAxesSignature GetSignature(pir::Operation* op);
  ShardableAxesSignature GetModifiedSignature(pir::Operation* op);
  ShardableAxes GetAxes(pir::Value value);
  static std::string GetUniqueName();
  std::string NameUnionDebugStr() const;

  std::string GetNormalizedAxisName(const std::string& axis_name) {
    return normalized_root_name_map_[FindAxisRoot(axis_name)];
  }
  auto& related_axes_map() { return related_axes_; }

 private:
  ShardableAxesSignature CreateShardableSignature(pir::Operation* op);
  ShardableAxes ReplaceShardableAxesWithRootName(const ShardableAxes& axes,
                                                 bool normalize = false);
  std::string FindAxisRoot(const std::string& name);

  const std::vector<pir::Operation*>& ops_;
  pir::ShapeConstraintIRAnalysis* shape_analysis_;

  std::unordered_map<pir::Operation*, ShardableAxesSignature> op_signature_map_;
  std::unordered_map<pir::Value, ShardableAxes> value_axes_map_;
  std::unordered_map<std::string, std::string> name_union_;

  std::unordered_map<std::string, std::vector<std::string>> root_to_sons_;
  std::unordered_map<std::string, std::string> normalized_root_name_map_;
  std::unordered_map<std::string, std::unordered_set<std::string>>
      related_axes_;
};

}  // namespace cinn::fusion
