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

#include "paddle/phi/core/distributed/auto_parallel/reshard/s_to_s_reshard_function.h"

#include "glog/logging.h"

#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_tensor.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/reshard_utils.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/same_status_reshard_function.h"
#include "paddle/phi/core/distributed/store/store_utils.h"
#include "paddle/phi/kernels/all_to_all_kernel.h"
#include "paddle/phi/kernels/reshape_kernel.h"
#include "paddle/phi/kernels/transpose_kernel.h"

namespace phi::distributed {

bool SToSReshardFunction::IsSuitable(const DistTensor& in,
                                     const TensorDistAttr& out_dist_attr) {
  const auto& in_dist_attr = in.dist_attr();

  RESHARD_SHORTCUT_IF_FALSE(in_dist_attr.dims_mapping() !=
                            out_dist_attr.dims_mapping());

  RESHARD_SHORTCUT_IF_FALSE(in_dist_attr.is_shard());
  RESHARD_SHORTCUT_IF_FALSE(out_dist_attr.is_shard());

  const auto& in_process_mesh = in_dist_attr.process_mesh();
  const auto& out_process_mesh = out_dist_attr.process_mesh();

  RESHARD_SHORTCUT_IF_FALSE(in_process_mesh.ndim() == 1);
  RESHARD_SHORTCUT_IF_FALSE(out_process_mesh.ndim() == 1);
  RESHARD_SHORTCUT_IF_FALSE(in_process_mesh == out_process_mesh);

  return true;
}

void SToSReshardFunction::Eval(phi::DeviceContext* dev_ctx,
                               const DistTensor& in,
                               const TensorDistAttr& out_dist_attr,
                               DistTensor* out) {
  VLOG(3) << "Call " << Name();
  const auto& in_process_mesh = in.dist_attr().process_mesh();
  const auto& in_process_ids = in_process_mesh.process_ids();
  auto dtype = in.dtype();
  const auto& logical_ddim = in.dims();
  int64_t nranks = static_cast<int64_t>(in_process_ids.size());
  int in_split_axis =
      GetSplitAxisWithDimsMapping(in.dist_attr().dims_mapping()).begin()->first;
  int out_split_axis =
      GetSplitAxisWithDimsMapping(out_dist_attr.dims_mapping()).begin()->first;

  DenseTensor in_all_to_all;
  // 1. preprocess, reshape and transpose the input tensor
  if (out_split_axis != 0) {
    // 1.1 calc the shape and reshape
    std::vector<int64_t> pre_shape_vec = common::vectorize(logical_ddim);
    pre_shape_vec[in_split_axis] /= nranks;
    pre_shape_vec[out_split_axis] /= nranks;
    pre_shape_vec.insert(pre_shape_vec.begin() + out_split_axis, nranks);

    DenseTensor out_reshape1;
    RESHARD_FUNCTOR(
        dev_ctx, Reshape, dtype, in.value(), pre_shape_vec, &out_reshape1);

    // 1.2 calc the the desire axis and transpose
    std::vector<int> axis;
    axis.emplace_back(out_split_axis);
    for (size_t i = 0; i < pre_shape_vec.size(); ++i) {
      if (static_cast<int>(i) != out_split_axis) {
        axis.emplace_back(i);
      }
    }
    DenseTensor out_transpose;
    RESHARD_FUNCTOR(
        dev_ctx, Transpose, dtype, out_reshape1, axis, &out_transpose);

    // 1.3 calc the final shape and reshape
    pre_shape_vec.erase(pre_shape_vec.begin() + out_split_axis);
    pre_shape_vec[in_split_axis] *= nranks;
    RESHARD_FUNCTOR(
        dev_ctx, Reshape, dtype, out_transpose, pre_shape_vec, &in_all_to_all);
  } else {
    in_all_to_all.ShareDataNoCheckWith(in.value());
  }

  // 2. use all to all to switch data to other ranks
#if defined(PADDLE_WITH_XPU)
  PADDLE_THROW(
      ::common::errors::Unimplemented("Not supported AllToAll on xpu yet."));
#else
  RESHARD_FUNCTOR_WITH_COMM(dev_ctx,
                            AllToAll,
                            dtype,
                            in_process_ids,
                            in_all_to_all,
                            GetMutableTensor(out));
#endif

  // 3. postprocess, reshape and transpose the output tensor
  if (in_split_axis != 0) {
    // 3.1 calc the shape and reshape
    std::vector<int64_t> post_shape_vec = common::vectorize(logical_ddim);
    post_shape_vec[in_split_axis] /= nranks;
    post_shape_vec[out_split_axis] /= nranks;
    post_shape_vec.insert(post_shape_vec.begin(), nranks);

    DenseTensor out_reshape1;
    RESHARD_FUNCTOR(
        dev_ctx, Reshape, dtype, out->value(), post_shape_vec, &out_reshape1);

    // 3.2 calc the the desire axis and transpose
    std::vector<int> axis;
    for (size_t i = 1; i < post_shape_vec.size(); ++i) {
      axis.emplace_back(i);
    }
    axis.insert(axis.begin() + in_split_axis, 0);
    DenseTensor out_transpose;
    RESHARD_FUNCTOR(
        dev_ctx, Transpose, dtype, out_reshape1, axis, &out_transpose);

    // 3.3 calc the final shape and reshape
    post_shape_vec.erase(post_shape_vec.begin());
    post_shape_vec[in_split_axis] *= nranks;
    RESHARD_FUNCTOR(dev_ctx,
                    Reshape,
                    dtype,
                    out_transpose,
                    post_shape_vec,
                    GetMutableTensor(out));
  }

  SetDistProps(out, in.dims(), out_dist_attr);
}

bool SToSReshardFunctionCrossMesh::IsSuitable(
    const DistTensor& in, const TensorDistAttr& out_dist_attr) {
  const auto& in_dist_attr = in.dist_attr();

  RESHARD_SHORTCUT_IF_FALSE(in_dist_attr.dims_mapping() !=
                            out_dist_attr.dims_mapping());

  RESHARD_SHORTCUT_IF_FALSE(in_dist_attr.is_shard());
  RESHARD_SHORTCUT_IF_FALSE(out_dist_attr.is_shard());

  const auto& in_process_mesh = in_dist_attr.process_mesh();
  const auto& out_process_mesh = out_dist_attr.process_mesh();

  RESHARD_SHORTCUT_IF_FALSE(in_process_mesh.ndim() == 1);
  RESHARD_SHORTCUT_IF_FALSE(out_process_mesh.ndim() == 1);
  RESHARD_SHORTCUT_IF_FALSE(in_process_mesh != out_process_mesh);

  return true;
}

void SToSReshardFunctionCrossMesh::Eval(DeviceContext* dev_ctx,
                                        const DistTensor& in,
                                        const TensorDistAttr& out_dist_attr,
                                        DistTensor* out) {
  VLOG(3) << "Call " << Name();
  const auto& out_process_mesh = out_dist_attr.process_mesh();

  SameStatusReshardFunction same_status_func;
  DistTensor tmp_result;

  TensorDistAttr tmp_dist_attr = in.dist_attr();
  tmp_dist_attr.set_process_mesh(out_process_mesh);
  same_status_func.Eval(dev_ctx, in, tmp_dist_attr, &tmp_result);

  int64_t cur_global_rank = GetCurGlobalRank();
  if (out_process_mesh.contains(cur_global_rank)) {
    SToSReshardFunction s_to_s_func;
    PADDLE_ENFORCE(
        s_to_s_func.IsSuitable(tmp_result, out_dist_attr),
        common::errors::InvalidArgument(
            "Invoke the s to s reshard function is not valid from %s to %s.",
            tmp_result.dist_attr(),
            out_dist_attr));
    s_to_s_func.Eval(dev_ctx, tmp_result, out_dist_attr, out);
  } else {
    SetDistProps(out, in.dims(), out_dist_attr);
    SetValue(out, tmp_result.value());
  }
}

}  // namespace phi::distributed
