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

#include "paddle/phi/core/distributed/auto_parallel/reshard/s_to_r_reshard_function.h"

#include "glog/logging.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_tensor.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/reshard_utils.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/same_status_reshard_function.h"
#include "paddle/phi/core/distributed/store/store_utils.h"
#include "paddle/phi/kernels/all_gather_kernel.h"
#include "paddle/phi/kernels/concat_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/split_kernel.h"

namespace phi::distributed {

namespace {

void ReshardSToRWithPadding(DeviceContext* dev_ctx,
                            int64_t split_axis,
                            const std::vector<int64_t>& process_ids,
                            const DenseTensor& in,
                            int64_t padding_nums,
                            DenseTensor* out) {
  int64_t num_of_process = process_ids.size();
  auto dtype = in.dtype();

  // For balanced split to replicate, we need to do all gather first.
  // If the input value doesn't split on axis 0, we need to split
  // and concat on specific axis.
#if defined(PADDLE_WITH_XPU)
  PADDLE_THROW(
      ::common::errors::Unimplemented("Not supported AllGather on xpu yet."));
#else
  RESHARD_FUNCTOR_WITH_COMM(
      dev_ctx, AllGather, dtype, process_ids, in, num_of_process, out);
#endif

  if (split_axis != 0 || padding_nums != 0) {
    IntArray sections(std::vector<int64_t>(num_of_process, in.dims()[0]));

    std::vector<DenseTensor> split_out_vec;
    RESHARD_FUNCTOR(dev_ctx,
                    Split,
                    dtype,
                    *out,
                    sections,
                    /*split_axis*/ 0,
                    &split_out_vec);

    if (padding_nums != 0) {
      std::vector<DenseTensor> tmp_out_vec;
      IntArray tmp_sections(std::vector<int64_t>{
          in.dims()[split_axis] - padding_nums, padding_nums});
      RESHARD_FUNCTOR(dev_ctx,
                      Split,
                      dtype,
                      split_out_vec[num_of_process - 1],
                      tmp_sections,
                      split_axis,
                      &tmp_out_vec);
      split_out_vec[num_of_process - 1] = tmp_out_vec[0];
    }

    // Concat the result after split on correct axis.
    std::vector<const DenseTensor*> concat_input_vec;
    concat_input_vec.reserve(split_out_vec.size());
    for (const auto& tensor : split_out_vec) {
      concat_input_vec.emplace_back(&tensor);
    }
    RESHARD_FUNCTOR(dev_ctx, Concat, dtype, concat_input_vec, split_axis, out);
  }
}

}  // namespace

bool SToRReshardFunction::IsSuitable(const DistTensor& in,
                                     const TensorDistAttr& out_dist_attr) {
  const auto& in_dist_attr = in.dist_attr();

  RESHARD_SHORTCUT_IF_FALSE(in_dist_attr.is_shard());
  RESHARD_SHORTCUT_IF_FALSE(out_dist_attr.is_replicated());

  const auto& in_process_mesh = in_dist_attr.process_mesh();
  const auto& out_process_mesh = out_dist_attr.process_mesh();

  RESHARD_SHORTCUT_IF_FALSE(in_process_mesh.ndim() == 1);
  RESHARD_SHORTCUT_IF_FALSE(out_process_mesh.ndim() == 1);
  RESHARD_SHORTCUT_IF_FALSE(in_process_mesh == out_process_mesh);

  return true;
}

void SToRReshardFunction::Eval(DeviceContext* dev_ctx,
                               const DistTensor& in,
                               const TensorDistAttr& out_dist_attr,
                               DistTensor* out) {
  VLOG(3) << "Call " << Name();
  const auto& in_dist_attr = in.dist_attr();
  const auto& in_dims_mapping = in_dist_attr.dims_mapping();
  const auto& in_process_mesh = in_dist_attr.process_mesh();
  const auto& in_process_ids = in_process_mesh.process_ids();

  int split_axis = GetSplitAxisWithDimsMapping(in_dims_mapping).begin()->first;
  int64_t num_of_process = in_process_mesh.size();
  int64_t num_of_padding = in.dims()[split_axis] % num_of_process;
  bool is_balanced_split = (num_of_padding == 0);

  if (is_balanced_split) {
    VLOG(3) << "Balanced reshard from shard to replicated";
    ReshardSToRWithPadding(dev_ctx,
                           split_axis,
                           in_process_ids,
                           in.value(),
                           /*padding_nums*/ 0,
                           GetMutableTensor(out));
  } else {
    VLOG(3) << "Unbalanced reshard from shard to replicated";
    int64_t avg_size_on_split_axis =
        (in.dims()[split_axis] + num_of_process - 1) / num_of_process;
    int64_t padding_nums =
        avg_size_on_split_axis * num_of_process - in.dims()[split_axis];
    bool need_padding = (in.local_dims()[split_axis] != avg_size_on_split_axis);

    if (need_padding) {
      DDim concat_local_shape = in.local_dims();
      concat_local_shape[split_axis] = padding_nums;
      IntArray concat_local_shape_int_array(concat_local_shape.Get(),
                                            concat_local_shape.size());
      auto dtype = in.dtype();

      DenseTensor concat_local_tensor;
      RESHARD_FUNCTOR(dev_ctx,
                      Full,
                      dtype,
                      concat_local_shape_int_array,
                      0,
                      &concat_local_tensor);

      DenseTensor in_local_tensor = in.value();
      std::vector<const DenseTensor*> concat_input_vec = {&in_local_tensor,
                                                          &concat_local_tensor};

      DenseTensor concat_result;
      RESHARD_FUNCTOR(
          dev_ctx, Concat, dtype, concat_input_vec, split_axis, &concat_result);
      ReshardSToRWithPadding(dev_ctx,
                             split_axis,
                             in_process_ids,
                             concat_result,
                             padding_nums,
                             GetMutableTensor(out));
    } else {
      ReshardSToRWithPadding(dev_ctx,
                             split_axis,
                             in_process_ids,
                             in.value(),
                             padding_nums,
                             GetMutableTensor(out));
    }
  }
  SetDistProps(out, in.dims(), out_dist_attr);
}

bool SToRReshardFunctionCrossMesh::IsSuitable(
    const DistTensor& in, const TensorDistAttr& out_dist_attr) {
  const auto& in_dist_attr = in.dist_attr();

  RESHARD_SHORTCUT_IF_FALSE(in_dist_attr.is_shard());
  RESHARD_SHORTCUT_IF_FALSE(out_dist_attr.is_replicated());

  const auto& in_process_mesh = in_dist_attr.process_mesh();
  const auto& out_process_mesh = out_dist_attr.process_mesh();

  RESHARD_SHORTCUT_IF_FALSE(in_process_mesh.ndim() == 1);
  RESHARD_SHORTCUT_IF_FALSE(out_process_mesh.ndim() == 1);
  RESHARD_SHORTCUT_IF_FALSE(in_process_mesh.shape() ==
                            out_process_mesh.shape());
  RESHARD_SHORTCUT_IF_FALSE(in_process_mesh != out_process_mesh);

  return true;
}

void SToRReshardFunctionCrossMesh::Eval(DeviceContext* dev_ctx,
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
    SToRReshardFunction s_to_r_func;
    PADDLE_ENFORCE(
        s_to_r_func.IsSuitable(tmp_result, out_dist_attr),
        common::errors::InvalidArgument(
            "Invoke the s to r reshard function is not valid from %s to %s.",
            tmp_result.dist_attr(),
            out_dist_attr));
    s_to_r_func.Eval(dev_ctx, tmp_result, out_dist_attr, out);
  } else {
    SetDistProps(out, in.dims(), out_dist_attr);
    SetValue(out, tmp_result.value());
  }
}

}  // namespace phi::distributed
