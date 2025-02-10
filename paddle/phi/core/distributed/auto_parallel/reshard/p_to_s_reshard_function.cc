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

#include "paddle/phi/core/distributed/auto_parallel/reshard/p_to_s_reshard_function.h"

#include "glog/logging.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_tensor.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/reshard_utils.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/same_status_reshard_function.h"
#include "paddle/phi/core/distributed/store/store_utils.h"
#include "paddle/phi/kernels/concat_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/reduce_scatter_kernel.h"
#include "paddle/phi/kernels/split_kernel.h"
#include "paddle/phi/kernels/transpose_kernel.h"

namespace phi::distributed {

bool PToSReshardFunction::IsSuitable(const DistTensor& in,
                                     const TensorDistAttr& out_dist_attr) {
  const auto& in_dist_attr = in.dist_attr();

  RESHARD_SHORTCUT_IF_FALSE(in_dist_attr.is_partial());
  RESHARD_SHORTCUT_IF_FALSE(out_dist_attr.is_shard());

  const auto& in_process_mesh = in_dist_attr.process_mesh();
  const auto& out_process_mesh = out_dist_attr.process_mesh();

  RESHARD_SHORTCUT_IF_FALSE(in_process_mesh.ndim() == 1);
  RESHARD_SHORTCUT_IF_FALSE(out_process_mesh.ndim() == 1);
  RESHARD_SHORTCUT_IF_FALSE(in_process_mesh == out_process_mesh);

  return true;
}

void ReshardPToSWithPadding(DeviceContext* dev_ctx,
                            int64_t split_axis,
                            const std::vector<int64_t>& process_ids,
                            const DenseTensor& in,
                            int64_t padding_nums,
                            DenseTensor* out) {
  DenseTensor in_reduce_scatter;
  std::vector<int> axis;
  const auto& logical_ddim = in.dims();
  auto dtype = in.dtype();

  if (split_axis != 0) {
    for (size_t i = 0; i < common::vectorize(logical_ddim).size(); ++i) {
      axis.emplace_back(i);
    }
    std::swap(axis[0], axis[split_axis]);
    RESHARD_FUNCTOR(dev_ctx, Transpose, dtype, in, axis, &in_reduce_scatter);
  } else {
    in_reduce_scatter.ShareDataNoCheckWith(in);
  }

  DenseTensor out_reduce_scatter;
#if defined(PADDLE_WITH_XPU)
  PADDLE_THROW(::common::errors::Unimplemented(
      "Not supported Reducescatter on xpu yet."));
#else
  RESHARD_FUNCTOR_WITH_COMM(dev_ctx,
                            ReduceScatter,
                            dtype,
                            process_ids,
                            in_reduce_scatter,
                            static_cast<int64_t>(process_ids.size()),
                            &out_reduce_scatter);
#endif
  DenseTensor out_result;
  if (split_axis != 0) {
    RESHARD_FUNCTOR(
        dev_ctx, Transpose, dtype, out_reduce_scatter, axis, &out_result);
  } else {
    out_result.ShareDataNoCheckWith(out_reduce_scatter);
  }

  int64_t cur_global_rank = GetCurGlobalRank();
  if (cur_global_rank == process_ids.back() && padding_nums != 0) {
    std::vector<DenseTensor> tmp_out_vec;
    IntArray tmp_sections(std::vector<int64_t>{
        out_result.dims()[split_axis] - padding_nums, padding_nums});
    RESHARD_FUNCTOR(dev_ctx,
                    Split,
                    dtype,
                    out_result,
                    tmp_sections,
                    split_axis,
                    &tmp_out_vec);
    // TODO(liyurui): Since we can not separate local tensor with [0, 10] shape
    // and uninitialized tensor, here we use a tricky solution.
    // Give local tensor which has, for example [0, 10] shape, a little
    // allocation, to make it difference from uninitialized tensor in pipeline
    // strategy.
    if (tmp_out_vec[0].dims()[split_axis] == 0) {
      tmp_out_vec[0].mutable_data(tmp_out_vec[0].place(), 4);
    }
    out->ShareDataNoCheckWith(tmp_out_vec[0]);
  } else {
    out->ShareDataNoCheckWith(out_result);
  }
}

void PToSReshardFunction::Eval(DeviceContext* dev_ctx,
                               const DistTensor& in,
                               const TensorDistAttr& out_dist_attr,
                               DistTensor* out) {
  VLOG(3) << "Call " << Name();
  const auto& in_dist_attr = in.dist_attr();
  const auto& in_process_mesh = in_dist_attr.process_mesh();
  const auto& in_process_ids = in_process_mesh.process_ids();

  int out_split_axis =
      GetSplitAxisWithDimsMapping(out_dist_attr.dims_mapping()).begin()->first;
  int64_t num_of_process = in_process_mesh.size();
  int64_t num_of_padding = in.dims()[out_split_axis] % num_of_process;
  bool is_balanced_split = (num_of_padding == 0);

  if (is_balanced_split) {
    VLOG(3) << "Balanced reshard from partial to shard";
    ReshardPToSWithPadding(dev_ctx,
                           out_split_axis,
                           in_process_ids,
                           in.value(),
                           /*padding_nums*/ 0,
                           GetMutableTensor(out));
  } else {
    VLOG(3) << "Unbalanced reshard from partial to shard";
    int64_t avg_size_on_split_axis =
        (in.dims()[out_split_axis] + num_of_process - 1) / num_of_process;
    int64_t padding_nums =
        avg_size_on_split_axis * num_of_process - in.dims()[out_split_axis];

    DDim concat_local_shape = in.local_dims();
    concat_local_shape[out_split_axis] = padding_nums;
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
    RESHARD_FUNCTOR(dev_ctx,
                    Concat,
                    dtype,
                    concat_input_vec,
                    out_split_axis,
                    &concat_result);

    ReshardPToSWithPadding(dev_ctx,
                           out_split_axis,
                           in_process_ids,
                           concat_result,
                           padding_nums,
                           GetMutableTensor(out));
  }

  SetDistProps(out, in.dims(), out_dist_attr);
}

bool PToSReshardFunctionCrossMesh::IsSuitable(
    const DistTensor& in, const TensorDistAttr& out_dist_attr) {
  const auto& in_dist_attr = in.dist_attr();

  RESHARD_SHORTCUT_IF_FALSE(in_dist_attr.is_partial());
  RESHARD_SHORTCUT_IF_FALSE(out_dist_attr.is_shard());

  const auto& in_process_mesh = in_dist_attr.process_mesh();
  const auto& out_process_mesh = out_dist_attr.process_mesh();

  RESHARD_SHORTCUT_IF_FALSE(in_process_mesh.ndim() == 1);
  RESHARD_SHORTCUT_IF_FALSE(out_process_mesh.ndim() == 1);
  RESHARD_SHORTCUT_IF_FALSE(in_process_mesh != out_process_mesh);

  return true;
}

void PToSReshardFunctionCrossMesh::Eval(DeviceContext* dev_ctx,
                                        const DistTensor& in,
                                        const TensorDistAttr& out_dist_attr,
                                        DistTensor* out) {
  VLOG(3) << "Call " << Name();
  const auto& in_dist_attr = in.dist_attr();

  DistTensor tmp_result;
  TensorDistAttr in_dist_attr_shard = in_dist_attr;
  in_dist_attr_shard.clean_partial_status();
  in_dist_attr_shard.set_dims_mapping(out_dist_attr.dims_mapping());

  int64_t cur_global_rank = GetCurGlobalRank();
  if (in_dist_attr.process_mesh().contains(cur_global_rank)) {
    PToSReshardFunction p_to_s_func;
    PADDLE_ENFORCE(
        p_to_s_func.IsSuitable(in, in_dist_attr_shard),
        common::errors::InvalidArgument(
            "Invoke the p to s reshard function is not valid from %s to %s.",
            in_dist_attr,
            in_dist_attr_shard));
    p_to_s_func.Eval(dev_ctx, in, in_dist_attr_shard, &tmp_result);
  } else {
    SetDistProps(&tmp_result, in.dims(), in_dist_attr_shard);
    SetValue(&tmp_result, in.value());
  }

  SameStatusReshardFunction same_status_func;
  PADDLE_ENFORCE(
      same_status_func.IsSuitable(tmp_result, out_dist_attr),
      common::errors::InvalidArgument("Invoke the same status reshard function "
                                      "is not valid from %s to %s.",
                                      tmp_result.dist_attr(),
                                      out_dist_attr));
  same_status_func.Eval(dev_ctx, tmp_result, out_dist_attr, out);
}

}  // namespace phi::distributed
