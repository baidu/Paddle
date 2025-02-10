/* Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHoutput WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/infermeta/spmd_rules/nonzero.h"

#include "glog/logging.h"

#include "paddle/phi/core/enforce.h"
#include "paddle/phi/infermeta/spmd_rules/spmd_rule_macro_define.h"
#include "paddle/phi/infermeta/spmd_rules/utils.h"

namespace phi::distributed {
using phi::distributed::auto_parallel::str_join;

SpmdInfo NonZeroInferSpmd(const DistMetaTensor& x) {
  // Step0: Verify input args based on nonzero logic
  EXTRACT_SHAPE_AND_DIST_ATTR_WITH_DIM_CK(x);

  // Step1: Build einsum notation
  std::string x_axes(x_ndim, '1');
  std::string output_axes(2, '1');

  // Step2: Sharding Propagation
  // Step2.1: Merge input shardings
  std::unordered_map<std::string, int64_t> axis_to_dim_map =
      ShardingMergeForTensors({{x_axes, x_dims_mapping_src}});

  // Step2.2: Infer input and output dims mapping
  auto x_dims_mapping_dst = GetDimsMappingForAxes(x_axes, axis_to_dim_map);
  auto x_dist_attr_dst = CopyTensorDistAttrForOutput(x_dist_attr_src);
  x_dist_attr_dst.set_dims_mapping(x_dims_mapping_dst);
  auto output_dims_mapping_dst =
      GetDimsMappingForAxes(output_axes, axis_to_dim_map);
  auto output_dist_attr_dst = CopyTensorDistAttrForOutput(x_dist_attr_src);
  output_dist_attr_dst.set_dims_mapping(output_dims_mapping_dst);

  // Step 3: Log messages
  VLOG(4) << "NonZeroInferSpmd:";
  VLOG(4) << "Einsum Notation: " << x_axes << "-->" << output_axes;

  LOG_SPMD_INPUT(x);
  LOG_SPMD_OUTPUT(output_dist_attr_dst);

  return SpmdInfo({x_dist_attr_dst}, {output_dist_attr_dst});
}

SpmdInfo NonZeroInferSpmdReverse(const DistMetaTensor& x,
                                 const DistMetaTensor& output) {
  // Step0: Verify input args based on nonzero logic
  EXTRACT_SHAPE_AND_DIST_ATTR_WITH_DIM_CK(x);
  EXTRACT_SHAPE_AND_DIST_ATTR_WITH_DIM_CK(output);

  // Step1: Build einsum notation
  std::string x_axes(x_ndim, '1');
  std::string output_axes(2, '1');

  // Step2: Sharding Propagation
  // Step2.1: Merge input shardings
  std::unordered_map<std::string, int64_t> axis_to_dim_map =
      ShardingMergeForTensors({{output_axes, output_dims_mapping_src}});

  // Step2.2: Infer input and output dims mapping
  auto x_dims_mapping_dst = GetDimsMappingForAxes(x_axes, axis_to_dim_map);
  auto x_dist_attr_dst = CopyTensorDistAttrForOutput(x_dist_attr_src);
  x_dist_attr_dst.set_dims_mapping(x_dims_mapping_dst);
  auto output_dims_mapping_dst =
      GetDimsMappingForAxes(output_axes, axis_to_dim_map);
  auto output_dist_attr_dst = CopyTensorDistAttrForOutput(output_dist_attr_src);
  output_dist_attr_dst.set_dims_mapping(output_dims_mapping_dst);

  // Step 3: Log messages
  VLOG(4) << "NonZeroInferSpmdReverse:";
  VLOG(4) << "Einsum Notation: " << x_axes << "-->" << output_axes;

  LOG_SPMD_INPUT(x);
  LOG_SPMD_INPUT(output);
  LOG_SPMD_OUTPUT(output_dist_attr_dst);

  return SpmdInfo({x_dist_attr_dst}, {output_dist_attr_dst});
}  // namespace distributed
}  // namespace phi::distributed
