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

#include "paddle/phi/infermeta/spmd_rules/gather.h"

#include "glog/logging.h"

#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/inferspmd_utils.h"
#include "paddle/phi/core/distributed/auto_parallel/utils.h"
#include "paddle/phi/infermeta/spmd_rules/spmd_rule_macro_define.h"
#include "paddle/phi/infermeta/spmd_rules/utils.h"

namespace phi::distributed {

using phi::distributed::auto_parallel::str_join;

SpmdInfo GatherInferSpmdBase(const DistMetaTensor& x,
                             const DistMetaTensor& index,
                             int axis) {
  // Step0: Verify Input Args Based on Gather Logic
  // extract and check x_ndim, x_shape, x_dist_attr_src and
  // x_dims_mapping_src with the macro
  EXTRACT_SHAPE_AND_DIST_ATTR(x);
  // index may be 0-d tensor, verify it specifically
  auto index_shape = common::vectorize(index.dims());
  int index_ndim = index_shape.size();
  const TensorDistAttr& index_dist_attr_src = index.dist_attr();
  const std::vector<int64_t>& index_dims_mapping_src =
      index_dist_attr_src.dims_mapping();
  if (index_ndim == 0) {
    PADDLE_ENFORCE_EQ(index_dims_mapping_src.size(),
                      1,
                      common::errors::InvalidArgument(
                          "index is 0-d tensor, it's dims_mapping size "
                          "must be 1, but received [%d]",
                          index_dims_mapping_src.size()));
  } else {
    PADDLE_ENFORCE_EQ(index_ndim,
                      index_dims_mapping_src.size(),
                      common::errors::InvalidArgument(
                          "Tensor index's rank [%d] and "
                          "dims_mapping size [%d] are not matched.",
                          index_ndim,
                          index_dims_mapping_src.size()));
  }

  // Step1: Build Einsum Notation
  std::string alphabet = "abcdefghijlmnopqrstuvwxyz";
  std::string x_axes = GetBroadcastAxes(x_ndim, x_ndim, alphabet);
  std::string index_axes = "k";
  std::string out_axes = x_axes;
  if (index_ndim == 0) {
    if (axis < x_ndim) {
      out_axes.erase(axis, 1);
    }
    index_axes = "";
  } else {
    out_axes[axis] = 'k';
  }

  // Step2: Sharding Propagation
  // Step2.1: Merge input shardings
  std::vector<int64_t> x_dims_mapping(x_dims_mapping_src);
  if (axis < x_ndim) {
    x_dims_mapping[axis] = -1;
  }
  std::vector<int64_t> index_dims_mapping(index_dims_mapping_src);
  if (index_ndim == 0) {
    index_dims_mapping[0] = -1;
  }
  std::unordered_map<std::string, int64_t> axis_to_dim_map =
      ShardingMergeForTensors(
          {{x_axes, x_dims_mapping}, {index_axes, index_dims_mapping}});

  TensorDistAttr x_dist_attr_dst = CopyTensorDistAttrForOutput(x_dist_attr_src);
  x_dist_attr_dst.set_dims_mapping(x_dims_mapping);

  TensorDistAttr index_dist_attr_dst =
      CopyTensorDistAttrForOutput(index_dist_attr_src);
  index_dist_attr_dst.set_dims_mapping(index_dims_mapping);

  // Step2.2: Infer output dims mapping
  std::vector<int64_t> out_dims_mapping =
      GetDimsMappingForAxes(out_axes, axis_to_dim_map);
  TensorDistAttr out_dist_attr = CopyTensorDistAttrForOutput(x_dist_attr_src);
  out_dist_attr.set_dims_mapping(out_dims_mapping);

  VLOG(4) << "x_axes: " << x_axes << " index_axes: " << index_axes
          << " out_axes: " << out_axes;
  LOG_SPMD_INPUT(x);
  LOG_SPMD_INPUT(index);
  VLOG(4) << "out";
  VLOG(4) << "dist_attr: [" << out_dist_attr.to_string() << "]";
  return {{x_dist_attr_dst, index_dist_attr_dst}, {out_dist_attr}};
}

SpmdInfo GatherInferSpmdReverseBase(const DistMetaTensor& x,
                                    const DistMetaTensor& index,
                                    const DistMetaTensor& out,
                                    int axis) {
  // Step0: Verify Input Args Based on Gather Logic
  // extract and check out_ndim, out_shape, out_dist_attr_src and
  // out_dims_mapping_src with the macro
  EXTRACT_SHAPE_AND_DIST_ATTR(x);
  EXTRACT_SHAPE_AND_DIST_ATTR(index);
  EXTRACT_SHAPE_AND_DIST_ATTR(out);

  // Step1: Build Einsum Notation
  std::string alphabet = "abcdefghijlmnopqrstuvwxyz";
  // x should be replicated on 0th axis
  std::string x_axes = GetBroadcastAxes(x_ndim, x_ndim, alphabet);
  std::string index_axes = "k";
  std::string out_axes = x_axes;
  if (index_ndim == 0) {
    index_axes = "";
    if (axis < x_ndim) {
      out_axes.erase(axis, 1);
    }
  } else {
    out_axes[axis] = 'k';
  }

  // Step2: Sharding Propagation
  // Step2.1: Merge output shardings
  std::unordered_map<std::string, int64_t> axis_to_dim_map =
      ShardingMergeForTensors({{out_axes, out_dims_mapping_src}});

  // Step2.2: Infer input dims mapping
  std::vector<int64_t> x_dims_mapping =
      GetDimsMappingForAxes(x_axes, axis_to_dim_map, true);
  if (axis < x_ndim) {
    x_dims_mapping[axis] = -1;
  }
  std::vector<int64_t> index_dims_mapping =
      GetDimsMappingForAxes(index_axes, axis_to_dim_map, true);
  TensorDistAttr x_dist_attr_dst = CopyTensorDistAttrForOutput(x_dist_attr_src);
  x_dist_attr_dst.set_dims_mapping(x_dims_mapping);
  TensorDistAttr index_dist_attr_dst =
      CopyTensorDistAttrForOutput(index_dist_attr_src);
  index_dist_attr_dst.set_dims_mapping(index_dims_mapping);

  VLOG(4) << "out_axes: " << out_axes << " x_axes: " << x_axes
          << " index_axes: " << index_axes;
  VLOG(4) << "out dist_attr: [" << out_dist_attr_src.to_string() << "]";
  LOG_SPMD_INPUT(x);
  LOG_SPMD_INPUT(index);
  VLOG(4) << std::endl;
  return {{x_dist_attr_dst, index_dist_attr_dst}, {out_dist_attr_src}};
}

SpmdInfo GatherInferSpmdDynamic(const DistMetaTensor& x,
                                const DistMetaTensor& index,
                                const Scalar& axis) {
  return GatherInferSpmdBase(x, index, axis.to<int32_t>());
}

SpmdInfo GatherInferSpmdReverseDynamic(const DistMetaTensor& x,
                                       const DistMetaTensor& index,
                                       const DistMetaTensor& out,
                                       const Scalar& axis) {
  return GatherInferSpmdReverseBase(x, index, out, axis.to<int32_t>());
}

SpmdInfo GatherGradInferSpmd(const DistMetaTensor& x,
                             const DistMetaTensor& index,
                             const DistMetaTensor& out_grad,
                             const Scalar& axis) {
  EXTRACT_SHAPE_AND_DIST_ATTR(x);
  EXTRACT_SHAPE_AND_DIST_ATTR(out_grad);
  auto index_shape = common::vectorize(index.dims());
  int index_ndim = index_shape.size();
  const TensorDistAttr& index_dist_attr_src = index.dist_attr();
  const std::vector<int64_t>& index_dims_mapping_src =
      index_dist_attr_src.dims_mapping();
  int axis_ = axis.to<int32_t>();

  // TODO(zhangyichen): support shard on index and out_grad[axis]
  std::vector<int64_t> out_grad_dims_mapping_dst(out_grad_dims_mapping_src);
  TensorDistAttr out_grad_dist_attr_dst(out_grad_dist_attr_src);
  if (index_ndim == 0) {
    out_grad_dims_mapping_dst.insert(out_grad_dims_mapping_dst.begin() + axis_,
                                     -1);
  } else {
    out_grad_dims_mapping_dst[axis_] = -1;
    out_grad_dist_attr_dst.set_dims_mapping(out_grad_dims_mapping_dst);
  }

  std::vector<int64_t> index_dims_mapping_dst(index_dims_mapping_src);
  TensorDistAttr index_dist_attr_dst(index_dims_mapping_src);
  index_dims_mapping_dst[axis_] = -1;
  index_dist_attr_dst.set_dims_mapping(index_dims_mapping_dst);

  std::vector<int64_t> x_grad_dims_mapping(x_dims_mapping_src);
  for (int i = 0; i < x_ndim; ++i) {
    x_grad_dims_mapping[i] = out_grad_dims_mapping_dst[i];
  }

  TensorDistAttr x_grad_dist_attr(x_dist_attr_src);
  x_grad_dist_attr.set_dims_mapping(x_grad_dims_mapping);

  return {{x_dist_attr_src, index_dist_attr_dst, out_grad_dist_attr_dst},
          {x_grad_dist_attr}};
}

}  // namespace phi::distributed
