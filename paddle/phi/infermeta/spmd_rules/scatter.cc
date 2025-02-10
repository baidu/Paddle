/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/infermeta/spmd_rules/scatter.h"

#include "glog/logging.h"

#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/inferspmd_utils.h"
#include "paddle/phi/core/distributed/auto_parallel/utils.h"
#include "paddle/phi/infermeta/spmd_rules/gather.h"
#include "paddle/phi/infermeta/spmd_rules/spmd_rule_macro_define.h"
#include "paddle/phi/infermeta/spmd_rules/utils.h"

namespace phi::distributed {

using phi::distributed::auto_parallel::str_join;

////////////////// Utils Functions //////////////////
SpmdInfo ScatterBaseInferSpmd(const DistMetaTensor& x,
                              const DistMetaTensor& index,
                              const DistMetaTensor& updates) {
  // Step0: Verify Input Args Based on Scatter Logic
  // extract and check x_ndim, x_shape, x_dist_attr_src and
  // x_dims_mapping_src with the macro
  EXTRACT_SHAPE_AND_DIST_ATTR(x);
  EXTRACT_SHAPE_AND_DIST_ATTR(index);
  EXTRACT_SHAPE_AND_DIST_ATTR(updates);
  PADDLE_ENFORCE_LE(
      index_ndim,
      updates_ndim,
      common::errors::InvalidArgument(
          "%s (%d): The Index's rank [%d] should be less or equal "
          "to Updates' rank [%d].",
          __FILE__,
          __LINE__,
          index_ndim,
          updates_ndim));

  // Step1: Build Einsum Notation
  std::string alphabet = "abcdefghijklmnopqrstuvwxyz";
  // x should be replicated on 0th axis
  std::string index_axes = GetBroadcastAxes(index_ndim, index_ndim, alphabet);
  std::string updates_axes =
      GetBroadcastAxes(updates_ndim, updates_ndim, alphabet);
  std::string out_axes = GetBroadcastAxes(x_ndim, x_ndim, alphabet);
  out_axes[0] = '1';

  // Step2: Sharding Propagation
  // Step2.1: Merge input shardings
  std::unordered_map<std::string, int64_t> axis_to_dim_map =
      ShardingMergeForTensors({{index_axes, index_dims_mapping_src},
                               {updates_axes, updates_dims_mapping_src}});

  std::vector<int64_t> index_dims_mapping =
      GetDimsMappingForAxes(index_axes, axis_to_dim_map);
  TensorDistAttr index_dist_attr_dst =
      CopyTensorDistAttrForOutput(index_dist_attr_src);
  index_dist_attr_dst.set_dims_mapping(index_dims_mapping);

  std::vector<int64_t> updates_dims_mapping =
      GetDimsMappingForAxes(updates_axes, axis_to_dim_map);
  TensorDistAttr updates_dist_attr_dst =
      CopyTensorDistAttrForOutput(updates_dist_attr_src);
  updates_dist_attr_dst.set_dims_mapping(updates_dims_mapping);

  // Step2.2: Infer output dims mapping
  std::vector<int64_t> out_dims_mapping =
      GetDimsMappingForAxes(out_axes, axis_to_dim_map);
  // the batch axis of output must be replicated
  out_dims_mapping[0] = -1;
  TensorDistAttr out_dist_attr = CopyTensorDistAttrForOutput(x_dist_attr_src);
  out_dist_attr.set_dims_mapping(out_dims_mapping);

  // the dims mapping of x should be the same as output
  TensorDistAttr x_dist_attr_dst = CopyTensorDistAttrForOutput(x_dist_attr_src);
  x_dist_attr_dst.set_dims_mapping(out_dims_mapping);

  // Step3: Handle partial
  // output partial status
  // output is partialed if the batch axis of index and updates are sharded
  if (updates_dims_mapping[0] != -1) {
    std::vector<int64_t> partial_dims(1, updates_dims_mapping[0]);
    out_dist_attr.set_partial_status(partial_dims);
  }

  VLOG(4) << "index_axes: " << index_axes << " updates_axes: " << updates_axes
          << " out_axes: " << out_axes;
  LOG_SPMD_INPUT(x);
  LOG_SPMD_INPUT(index);
  LOG_SPMD_INPUT(updates);
  VLOG(4) << "Out dist_attr: [" << out_dist_attr.to_string() << "]\n\n";
  return {{x_dist_attr_dst, index_dist_attr_dst, updates_dist_attr_dst},
          {out_dist_attr}};
}

SpmdInfo ScatterInferSpmd(const DistMetaTensor& x,
                          const DistMetaTensor& index,
                          const DistMetaTensor& updates,
                          bool overwrite) {
  return ScatterBaseInferSpmd(x, index, updates);
}

SpmdInfo ScatterNdAddInferSpmd(const DistMetaTensor& x,
                               const DistMetaTensor& index,
                               const DistMetaTensor& updates) {
  return ScatterBaseInferSpmd(x, index, updates);
}

SpmdInfo ScatterBaseInferSpmdReverse(const DistMetaTensor& x,
                                     const DistMetaTensor& index,
                                     const DistMetaTensor& updates,
                                     const DistMetaTensor& out) {
  // Step0: Verify Input Args Based on Scatter Logic
  // extract and check out_ndim, out_shape, out_dist_attr_src and
  // out_dims_mapping_src with the macro
  EXTRACT_SHAPE_AND_DIST_ATTR(x);
  EXTRACT_SHAPE_AND_DIST_ATTR(index);
  EXTRACT_SHAPE_AND_DIST_ATTR(updates);
  EXTRACT_SHAPE_AND_DIST_ATTR(out);

  // Step1: Build Einsum Notation
  std::string alphabet = "abcdefghijklmnopqrstuvwxyz";
  // x should be replicated on 0th axis
  std::string x_axes = GetBroadcastAxes(x_ndim, x_ndim, alphabet);
  std::string index_axes = GetBroadcastAxes(index_ndim, index_ndim, alphabet);
  std::string updates_axes =
      GetBroadcastAxes(updates_ndim, updates_ndim, alphabet);
  std::string out_axes = GetBroadcastAxes(out_ndim, out_ndim, alphabet);

  // Step2: Sharding Propagation
  // Step2.1: Merge output shardings
  // the batch axis of output must be replicated
  // TODO(zhangyichen): consider the case when the output is partial
  std::vector<int64_t> out_dims_mapping(out_dims_mapping_src);
  out_dims_mapping[0] = -1;
  TensorDistAttr out_dist_attr_dst =
      CopyTensorDistAttrForOutput(out_dist_attr_src);
  out_dist_attr_dst.set_dims_mapping(out_dims_mapping);
  std::unordered_map<std::string, int64_t> axis_to_dim_map =
      ShardingMergeForTensors({{out_axes, out_dims_mapping}});

  // Step2.2: Infer input dims mapping
  std::vector<int64_t> x_dims_mapping =
      GetDimsMappingForAxes(x_axes, axis_to_dim_map);
  std::vector<int64_t> index_dims_mapping =
      GetDimsMappingForAxes(index_axes, axis_to_dim_map);
  std::vector<int64_t> updates_dims_mapping =
      GetDimsMappingForAxes(updates_axes, axis_to_dim_map);
  TensorDistAttr x_dist_attr_dst = CopyTensorDistAttrForOutput(x_dist_attr_src);
  x_dist_attr_dst.set_dims_mapping(x_dims_mapping);
  TensorDistAttr index_dist_attr_dst =
      CopyTensorDistAttrForOutput(index_dist_attr_src);
  index_dist_attr_dst.set_dims_mapping(index_dims_mapping);
  TensorDistAttr updates_dist_attr_dst =
      CopyTensorDistAttrForOutput(updates_dist_attr_src);
  updates_dist_attr_dst.set_dims_mapping(updates_dims_mapping);

  LOG_SPMD_INPUT(out);
  LOG_SPMD_INPUT(x);
  LOG_SPMD_INPUT(index);
  LOG_SPMD_INPUT(updates);
  VLOG(4) << std::endl;
  return {{x_dist_attr_dst, index_dist_attr_dst, updates_dist_attr_dst},
          {out_dist_attr_dst}};
}

SpmdInfo ScatterInferSpmdReverse(const DistMetaTensor& x,
                                 const DistMetaTensor& index,
                                 const DistMetaTensor& updates,
                                 const DistMetaTensor& out,
                                 bool overwrite) {
  return ScatterBaseInferSpmdReverse(x, index, updates, out);
}

SpmdInfo ScatterNdAddInferSpmdReverse(const DistMetaTensor& x,
                                      const DistMetaTensor& index,
                                      const DistMetaTensor& updates,
                                      const DistMetaTensor& out) {
  return ScatterBaseInferSpmdReverse(x, index, updates, out);
}

SpmdInfo ScatterGradInferSpmd(const DistMetaTensor& index,
                              const DistMetaTensor& updates,
                              const DistMetaTensor& out_grad,
                              bool overwrite) {
  EXTRACT_SHAPE_AND_DIST_ATTR(index);
  EXTRACT_SHAPE_AND_DIST_ATTR(updates);
  EXTRACT_SHAPE_AND_DIST_ATTR(out_grad);

  // the batch axis of index, updates, out_grad must be replicated
  std::vector<int64_t> index_dims_mapping(index_dims_mapping_src);
  index_dims_mapping[0] = -1;
  std::vector<int64_t> updates_dims_mapping(updates_dims_mapping_src);
  updates_dims_mapping[0] = -1;
  std::vector<int64_t> out_grad_dims_mapping(out_grad_dims_mapping_src);
  out_grad_dims_mapping[0] = -1;

  TensorDistAttr index_dist_attr_dst =
      CopyTensorDistAttrForOutput(index_dist_attr_src);
  index_dist_attr_dst.set_dims_mapping(index_dims_mapping);
  TensorDistAttr updates_dist_attr_dst =
      CopyTensorDistAttrForOutput(updates_dist_attr_src);
  updates_dist_attr_dst.set_dims_mapping(updates_dims_mapping);
  TensorDistAttr out_grad_dist_attr_dst =
      CopyTensorDistAttrForOutput(out_grad_dist_attr_src);
  out_grad_dist_attr_dst.set_dims_mapping(out_grad_dims_mapping);

  TensorDistAttr x_grad_dist_attr(out_grad_dist_attr_src);
  std::vector<int64_t> x_dims_mapping(out_grad_dims_mapping);
  x_grad_dist_attr.set_dims_mapping(x_dims_mapping);

  DistMetaTensor out_grad_dst(out_grad.dims(), out_grad_dist_attr_dst);
  DistMetaTensor index_dst(index.dims(), index_dist_attr_dst);

  SpmdInfo spmd_info = GatherInferSpmdBase(out_grad_dst, index_dst, 0);
  TensorDistAttr updates_grad_dist_attr =
      PADDLE_GET_CONST(TensorDistAttr, spmd_info.second[0]);

  return {{index_dist_attr_dst, updates_dist_attr_dst, out_grad_dist_attr_dst},
          {x_grad_dist_attr, updates_grad_dist_attr}};
}

}  // namespace phi::distributed
