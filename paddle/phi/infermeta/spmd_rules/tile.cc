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

#include "paddle/phi/infermeta/spmd_rules/tile.h"

#include "glog/logging.h"

#include "paddle/phi/core/enforce.h"
#include "paddle/phi/infermeta/spmd_rules/utils.h"

namespace phi::distributed {
using phi::distributed::auto_parallel::str_join;

SpmdInfo TileInferSpmd(const DistMetaTensor& x,
                       const std::vector<int64_t>& repeat_times) {
  auto x_shape = common::vectorize(x.dims());
  int x_ndim = x_shape.size();
  const auto& x_dist_attr_src = x.dist_attr();
  const std::vector<int64_t>& x_dims_mapping = x_dist_attr_src.dims_mapping();
  PADDLE_ENFORCE_EQ(
      x_ndim,
      x_dims_mapping.size(),
      common::errors::InvalidArgument("The Tensor x's rank [%d] and Input's "
                                      "dims_mapping size [%d] are not matched.",
                                      x_ndim,
                                      x_dims_mapping.size()));

  PADDLE_ENFORCE_LE(x_ndim,
                    repeat_times.size(),
                    common::errors::InvalidArgument(
                        "The Tensor x's rank [%d] and repeat_times's "
                        "size [%d] are not matched.",
                        x_ndim,
                        repeat_times.size()));

  int64_t broadcast_dims = repeat_times.size() - x_ndim;

  std::vector<int64_t> dims_to_unshard;
  for (int64_t i = broadcast_dims;
       i < static_cast<int64_t>(repeat_times.size());
       ++i) {
    if (repeat_times[i] == 1) {
      continue;
    }
    dims_to_unshard.push_back(i - broadcast_dims);
  }
  auto x_dist_attr_dst = UnShardTensorDims(x_dist_attr_src, dims_to_unshard);
  std::vector<int64_t> out_dims_mapping(repeat_times.size(), -1);
  const auto& x_dims_mapping_dst = x_dist_attr_dst.dims_mapping();
  for (int64_t i = broadcast_dims;
       i < static_cast<int64_t>(repeat_times.size());
       i++) {
    out_dims_mapping[i] = x_dims_mapping_dst[i - broadcast_dims];
  }
  auto out_dist_attr = CopyTensorDistAttrForOutput(x_dist_attr_dst);
  out_dist_attr.set_dims_mapping(out_dims_mapping);
  VLOG(4) << "TileInferSpmd:";
  VLOG(4) << "x shape: [" << str_join(x_shape) << "]"
          << "src_dims_mapping: [" << str_join(x_dist_attr_src.dims_mapping())
          << "] "
          << "dst_dims_mapping: [" << str_join(x_dist_attr_dst.dims_mapping())
          << "]";

  VLOG(4) << "Output"
          << " dims_mapping: [" << str_join(out_dist_attr.dims_mapping())
          << "]";
  VLOG(4) << std::endl;

  return {{x_dist_attr_dst}, {out_dist_attr}};
}

SpmdInfo TileInferSpmdDynamic(const DistMetaTensor& x,
                              const IntArray& repeat_times) {
  return TileInferSpmd(x, repeat_times.GetData());
}

SpmdInfo TileInferSpmdReverse(const DistMetaTensor& x,
                              const DistMetaTensor& out,
                              const std::vector<int64_t>& repeat_times) {
  auto x_shape = common::vectorize(x.dims());
  int x_ndim = x_shape.size();
  const auto& x_dist_attr_src = x.dist_attr();
  const std::vector<int64_t>& x_dims_mapping = x_dist_attr_src.dims_mapping();
  PADDLE_ENFORCE_EQ(
      x_ndim,
      x_dims_mapping.size(),
      common::errors::InvalidArgument("The Tensor x's rank [%d] and Input's "
                                      "dims_mapping size [%d] are not matched.",
                                      x_ndim,
                                      x_dims_mapping.size()));

  PADDLE_ENFORCE_LE(x_ndim,
                    repeat_times.size(),
                    common::errors::InvalidArgument(
                        "The Tensor x's rank [%d] and repeat_times's "
                        "size [%d] are not matched.",
                        x_ndim,
                        repeat_times.size()));

  auto out_shape = common::vectorize(out.dims());
  int out_ndim = out_shape.size();
  const auto& out_dist_attr_src = out.dist_attr();
  const std::vector<int64_t>& out_dims_mapping =
      out_dist_attr_src.dims_mapping();
  PADDLE_ENFORCE_EQ(
      out_ndim,
      out_dims_mapping.size(),
      common::errors::InvalidArgument("The Tensor out's rank [%d] and Input's "
                                      "dims_mapping size [%d] are not matched.",
                                      out_ndim,
                                      out_dims_mapping.size()));

  PADDLE_ENFORCE_EQ(out_ndim,
                    repeat_times.size(),
                    common::errors::InvalidArgument(
                        "The Tensor out's rank [%d] and repeat_times's "
                        "size [%d] are not matched.",
                        out_ndim,
                        repeat_times.size()));

  int64_t broadcast_dims = repeat_times.size() - x_ndim;

  std::vector<int64_t> dims_to_unshard;
  for (int64_t i = broadcast_dims;
       i < static_cast<int64_t>(repeat_times.size());
       ++i) {
    if (repeat_times[i] == 1) {
      continue;
    }
    dims_to_unshard.push_back(i);
  }
  auto out_dist_attr_dst =
      UnShardTensorDims(out_dist_attr_src, dims_to_unshard);

  const auto& out_dims_mapping_dst = out_dist_attr_dst.dims_mapping();
  std::vector<int64_t> x_dims_mapping_dst(x_ndim, -1);
  for (int64_t i = 0; i < static_cast<int64_t>(x_ndim); i++) {
    x_dims_mapping_dst[i] = out_dims_mapping_dst[i + broadcast_dims];
  }
  auto x_dist_attr_dst = CopyTensorDistAttrForOutput(x_dist_attr_src);
  x_dist_attr_dst.set_dims_mapping(x_dims_mapping_dst);

  VLOG(4) << "TileInferSpmdReverse:";

  VLOG(4) << "out shape: [" << str_join(out_shape) << "]"
          << "src_dims_mapping: [" << str_join(out_dist_attr_src.dims_mapping())
          << "] "
          << "dst_dims_mapping: [" << str_join(out_dist_attr_dst.dims_mapping())
          << "]";

  VLOG(4) << "x: "
          << "dst_dims_mapping: [" << str_join(x_dist_attr_dst.dims_mapping())
          << "]";

  return {{x_dist_attr_dst}, {out_dist_attr_dst}};
}

SpmdInfo TileGradInferSpmd(const DistMetaTensor& x,
                           const DistMetaTensor& out_grad,
                           IntArray repeat_times) {
  auto x_shape = common::vectorize(x.dims());
  int x_ndim = x_shape.size();
  const auto& x_dist_attr_src = x.dist_attr();
  const std::vector<int64_t>& x_dims_mapping = x_dist_attr_src.dims_mapping();
  PADDLE_ENFORCE_EQ(
      x_ndim,
      x_dims_mapping.size(),
      common::errors::InvalidArgument("The Tensor x's rank [%d] and Input's "
                                      "dims_mapping size [%d] are not matched.",
                                      x_ndim,
                                      x_dims_mapping.size()));

  PADDLE_ENFORCE_LE(x_ndim,
                    repeat_times.size(),
                    common::errors::InvalidArgument(
                        "The Tensor x's rank [%d] and repeat_times's "
                        "size [%d] are not matched.",
                        x_ndim,
                        repeat_times.size()));

  auto out_grad_shape = common::vectorize(out_grad.dims());
  int out_grad_ndim = out_grad_shape.size();
  const auto& out_grad_dist_attr_src = out_grad.dist_attr();
  const std::vector<int64_t>& out_grad_dims_mapping =
      out_grad_dist_attr_src.dims_mapping();
  PADDLE_ENFORCE_EQ(out_grad_ndim,
                    out_grad_dims_mapping.size(),
                    common::errors::InvalidArgument(
                        "The Tensor out_grad's rank [%d] and Input's "
                        "dims_mapping size [%d] are not matched.",
                        out_grad_ndim,
                        out_grad_dims_mapping.size()));

  PADDLE_ENFORCE_EQ(out_grad_ndim,
                    repeat_times.size(),
                    common::errors::InvalidArgument(
                        "The Tensor out_grad's rank [%d] and repeat_times's "
                        "size [%d] are not matched.",
                        out_grad_ndim,
                        repeat_times.size()));

  int64_t broadcast_dims = repeat_times.size() - x_ndim;

  std::vector<int64_t> dims_to_unshard_for_x;
  std::vector<int64_t> dims_to_unshard_for_out;
  for (int64_t i = broadcast_dims;
       i < static_cast<int64_t>(repeat_times.size());
       ++i) {
    if (repeat_times[i] == 1) {
      continue;
    }
    dims_to_unshard_for_x.push_back(i - broadcast_dims);
    dims_to_unshard_for_out.push_back(i);
  }
  auto x_dist_attr_dst =
      UnShardTensorDims(x_dist_attr_src, dims_to_unshard_for_x);
  auto out_grad_dist_attr_dst =
      UnShardTensorDims(out_grad_dist_attr_src, dims_to_unshard_for_out);

  std::string alphabet = "abcdefghijklmnopqrstuvwxyz";
  std::string x_axes = alphabet.substr(broadcast_dims, x_ndim);
  std::string out_grad_axes = alphabet.substr(0, out_grad_ndim);
  std::vector<std::pair<std::string, std::vector<int64_t>>> axes_sharding_info;
  axes_sharding_info.emplace_back(x_axes, x_dist_attr_dst.dims_mapping());
  axes_sharding_info.emplace_back(out_grad_axes,
                                  out_grad_dist_attr_dst.dims_mapping());
  auto axis_to_dim_map = ShardingMergeForTensors(axes_sharding_info);

  auto x_dim_mapping_dst = GetDimsMappingForAxes(x_axes, axis_to_dim_map, true);
  auto out_grad_dim_mapping_dst =
      GetDimsMappingForAxes(out_grad_axes, axis_to_dim_map, true);
  x_dist_attr_dst.set_dims_mapping(x_dim_mapping_dst);
  out_grad_dist_attr_dst.set_dims_mapping(out_grad_dim_mapping_dst);
  auto x_grad_dist_attr = CopyTensorDistAttrForOutput(x_dist_attr_dst);
  x_grad_dist_attr.set_dims_mapping(x_dim_mapping_dst);
  // partial grad dim
  std::vector<int64_t> partial_on_dims;
  const auto& dim_mapping = out_grad_dist_attr_dst.dims_mapping();
  for (int i = 0; i < broadcast_dims; ++i) {
    auto mapping = dim_mapping[i];
    if (mapping != -1) {
      partial_on_dims.push_back(mapping);
    }
  }
  x_grad_dist_attr.set_partial_status(partial_on_dims);

  VLOG(4) << "TileGradInferSpmd:";

  VLOG(4) << "x: " << str_join(x_shape) << "]"
          << "src_dims_mapping: [" << str_join(x_dist_attr_src.dims_mapping())
          << "] "
          << "dst_dims_mapping: [" << str_join(x_dist_attr_dst.dims_mapping())
          << "]";

  VLOG(4) << "out_grad: " << str_join(out_grad_shape) << "]"
          << "src_dims_mapping: ["
          << str_join(out_grad_dist_attr_src.dims_mapping()) << "] "
          << "dst_dims_mapping: ["
          << str_join(out_grad_dist_attr_dst.dims_mapping()) << "]";

  VLOG(4) << "x grad"
          << "dst_dims_mapping: [" << str_join(x_grad_dist_attr.dims_mapping())
          << "]";

  return {{x_dist_attr_dst, out_grad_dist_attr_dst}, {x_grad_dist_attr}};
}
}  // namespace phi::distributed
