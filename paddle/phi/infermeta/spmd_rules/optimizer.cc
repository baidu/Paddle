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

#include "paddle/phi/infermeta/spmd_rules/optimizer.h"

#include "glog/logging.h"

#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/reshard_utils.h"
#include "paddle/phi/core/distributed/auto_parallel/utils.h"
#include "paddle/phi/infermeta/spmd_rules/elementwise.h"
#include "paddle/phi/infermeta/spmd_rules/utils.h"

namespace phi::distributed {

SpmdInfo AdamInferSpmdDynamic(
    const DistMetaTensor& param,
    const DistMetaTensor& grad,
    const DistMetaTensor& learning_rate,
    const DistMetaTensor& moment1,
    const DistMetaTensor& moment2,
    const paddle::optional<DistMetaTensor>& moment2_max,
    const DistMetaTensor& beta1_pow,
    const DistMetaTensor& beta2_pow,
    const DistMetaTensor& master_param,
    const DistMetaTensor& skip_update,
    const Scalar& beta1,
    const Scalar& beta2,
    const Scalar& epsilon,
    bool lazy_mode,
    int64_t min_row_size_to_use_multithread,
    bool multi_precision,
    bool use_global_beta_pow,
    bool amsgrad) {
  // shape check
  PADDLE_ENFORCE(
      param.dims().size() == grad.dims().size() &&
          moment1.dims().size() == moment2.dims().size() &&
          param.dims().size() == moment1.dims().size(),
      errors::InvalidArgument(
          "param, grad, momentum1 and momentum2 have different ndim."));

  // Do spmd infer on param and grad in case of the param and grad
  // has different dist attr. This difference may be caused by other spmd.
  // No need do the spmd infer on the two momentum, since they are
  // separated from the forward backward computation.
  SpmdInfo param_grad_spmd = ElementwiseBinaryInferSpmd(param, grad);
  TensorDistAttr param_dist_attr_spmd =
      PADDLE_GET(TensorDistAttr, param_grad_spmd.first[0]);
  TensorDistAttr grad_dist_attr_spmd =
      PADDLE_GET(TensorDistAttr, param_grad_spmd.first[1]);

  VLOG(3) << "The source dims mapping for param is: "
          << auto_parallel::str_join(param.dist_attr().dims_mapping());
  VLOG(3) << "The source dims mapping for grad is: "
          << auto_parallel::str_join(grad.dist_attr().dims_mapping());
  VLOG(3) << "The inter dims mapping for param after elementwise spmd is: "
          << auto_parallel::str_join(param.dist_attr().dims_mapping());
  VLOG(3) << "The inter dims mapping for grad after elementwise spmd is: "
          << auto_parallel::str_join(grad.dist_attr().dims_mapping());

  // create all output dist attrs
  TensorDistAttr param_dist_attr =
      CopyTensorDistAttrForOutput(param_dist_attr_spmd);
  TensorDistAttr grad_dist_attr =
      CopyTensorDistAttrForOutput(grad_dist_attr_spmd);
  TensorDistAttr lr_dist_attr =
      CopyTensorDistAttrForOutput(learning_rate.dist_attr());
  TensorDistAttr moment1_dist_attr =
      CopyTensorDistAttrForOutput(moment1.dist_attr());
  TensorDistAttr moment2_dist_attr =
      CopyTensorDistAttrForOutput(moment2.dist_attr());
  TensorDistAttr moment2_max_dist_attr =
      amsgrad ? CopyTensorDistAttrForOutput(moment2_max.get().dist_attr())
              : TensorDistAttr();
  TensorDistAttr beta1_pow_dist_attr =
      CopyTensorDistAttrForOutput(beta1_pow.dist_attr());
  TensorDistAttr beta2_pow_dist_attr =
      CopyTensorDistAttrForOutput(beta2_pow.dist_attr());
  TensorDistAttr master_param_dist_attr =
      master_param.initialized()
          ? CopyTensorDistAttrForOutput(master_param.dist_attr())
          : TensorDistAttr();
  // If skip_update is on global_mesh, it should be reshard into
  // local mesh. (currently occurs in static mode pipeline parallel)
  auto skip_update_dist_attr = TensorDistAttr();
  if (skip_update.initialized()) {
    skip_update_dist_attr = skip_update.dist_attr();
    PADDLE_ENFORCE_EQ(
        skip_update_dist_attr.dims_mapping()[0],
        -1,
        errors::InvalidArgument(
            "skip_update should be replicated, but got shard on mesh %d.",
            skip_update_dist_attr.dims_mapping()[0]));
    skip_update_dist_attr.clean_partial_status();
    if (skip_update_dist_attr.process_mesh().ndim() > 1 &&
        phi::distributed::IsSubMesh(skip_update_dist_attr.process_mesh(),
                                    param_dist_attr.process_mesh())) {
      skip_update_dist_attr.set_process_mesh(param_dist_attr.process_mesh());
    }
  }
  // set the unchanged dims mapping
  lr_dist_attr.set_dims_mapping(learning_rate.dist_attr().dims_mapping());
  beta1_pow_dist_attr.set_dims_mapping(beta1_pow.dist_attr().dims_mapping());
  beta2_pow_dist_attr.set_dims_mapping(beta2_pow.dist_attr().dims_mapping());

  // set the changeable dims mapping
  auto param_spmd_dims_mapping = param_dist_attr_spmd.dims_mapping();
  auto grad_spmd_dims_mapping = grad_dist_attr_spmd.dims_mapping();
  auto momentum1_src_dims_mapping = moment1.dist_attr().dims_mapping();
  auto momentum2_src_dims_mapping = moment2.dist_attr().dims_mapping();

  std::vector<int64_t> momentum2_max_src_dims_mapping;
  if (amsgrad) {
    momentum2_max_src_dims_mapping =
        moment2_max.get().dist_attr().dims_mapping();
  }

  // Get the final dist attr for param, master_param, grad and momentum.
  // Whatever the input dist attrs are, the output dist attr should be same.
  // For a specific dim of the tensor:
  // If the dim has been sharded on one or more tensors
  // and these tensors use a same mesh to shard this dim,
  // then this shard status should be kept on the shard tensors
  // and should be brought to those unshard tensors.
  // Otherwise, if the dim hasn't been sharded an any tensor,
  // or different tensors use different meshes to shard the dim,
  // then the shard status should be removed on the shard tensors
  // and the unshard tensors should keep unshard status.
  std::vector<int64_t> dst_dims_mapping;
  for (int64_t i = 0; i < param.dims().size(); ++i) {
    std::vector<int64_t> shard_status;
    if (amsgrad) {
      shard_status.assign({param_spmd_dims_mapping[i],
                           grad_spmd_dims_mapping[i],
                           momentum1_src_dims_mapping[i],
                           momentum2_src_dims_mapping[i],
                           momentum2_max_src_dims_mapping[i]});

    } else {
      shard_status.assign({param_spmd_dims_mapping[i],
                           grad_spmd_dims_mapping[i],
                           momentum1_src_dims_mapping[i],
                           momentum2_src_dims_mapping[i]});
    }
    int64_t dst_shard_status = -1;
    for (auto status : shard_status) {
      if (status == -1) {
        // The dim i hasn't been sharded on current tensor.
        continue;
      } else {
        // The dim i has been sharded on current tensor.
        if (dst_shard_status == -1) {
          dst_shard_status = status;
        } else if (dst_shard_status != status) {
          // Tensors use different meshes to shard dim i.
          // The shard info should be removed.
          dst_shard_status = -1;
          break;
        }
      }
    }
    dst_dims_mapping.emplace_back(dst_shard_status);
  }

  VLOG(3) << "The source dims mapping for momentum1 is: "
          << auto_parallel::str_join(momentum1_src_dims_mapping);
  VLOG(3) << "The source dims mapping for momentum2 is: "
          << auto_parallel::str_join(momentum2_src_dims_mapping);
  if (master_param.initialized()) {
    VLOG(3) << "The source dims mapping for master param is: "
            << auto_parallel::str_join(master_param.dist_attr().dims_mapping());
  }
  VLOG(3) << "The final dims mapping for param, master param (if available), "
             "grad and momentum1, momentum 2 is: "
          << auto_parallel::str_join(dst_dims_mapping);

  param_dist_attr.set_dims_mapping(dst_dims_mapping);
  grad_dist_attr.set_dims_mapping(dst_dims_mapping);
  if (master_param.initialized()) {
    master_param_dist_attr.set_dims_mapping(dst_dims_mapping);
  }
  moment1_dist_attr.set_dims_mapping(dst_dims_mapping);
  moment2_dist_attr.set_dims_mapping(dst_dims_mapping);
  if (amsgrad) {
    moment2_max_dist_attr.set_dims_mapping(dst_dims_mapping);
  }

  return {{param_dist_attr,
           grad_dist_attr,
           lr_dist_attr,
           moment1_dist_attr,
           moment2_dist_attr,
           moment2_max_dist_attr,
           beta1_pow_dist_attr,
           beta2_pow_dist_attr,
           master_param_dist_attr,
           skip_update_dist_attr},
          {param_dist_attr,
           moment1_dist_attr,
           moment2_dist_attr,
           moment2_max_dist_attr,
           beta1_pow_dist_attr,
           beta2_pow_dist_attr,
           master_param_dist_attr}};
}

SpmdInfo AdamwInferSpmdDynamic(
    const DistMetaTensor& param,
    const DistMetaTensor& grad,
    const DistMetaTensor& learning_rate,
    const DistMetaTensor& moment1,
    const DistMetaTensor& moment2,
    const paddle::optional<DistMetaTensor>& moment2_max,
    const DistMetaTensor& beta1_pow,
    const DistMetaTensor& beta2_pow,
    const DistMetaTensor& master_param,
    const DistMetaTensor& skip_update,
    const Scalar& beta1,
    const Scalar& beta2,
    const Scalar& epsilon,
    float lr_ratio,
    float coeff,
    bool with_decay,
    bool lazy_mode,
    int64_t min_row_size_to_use_multithread,
    bool multi_precision,
    bool use_global_beta_pow,
    bool amsgrad) {
  return AdamInferSpmdDynamic(param,
                              grad,
                              learning_rate,
                              moment1,
                              moment2,
                              moment2_max,
                              beta1_pow,
                              beta2_pow,
                              master_param,
                              skip_update,
                              beta1,
                              beta2,
                              epsilon,
                              lazy_mode,
                              min_row_size_to_use_multithread,
                              multi_precision,
                              use_global_beta_pow,
                              amsgrad);
}

SpmdInfo SgdInferSpmd(const DistMetaTensor& param,
                      const DistMetaTensor& learning_rate,
                      const DistMetaTensor& grad,
                      const DistMetaTensor& master_param,
                      bool multi_precision) {
  SpmdInfo param_grad_spmd = ElementwiseBinaryInferSpmd(param, grad);
  TensorDistAttr param_dist_attr_spmd =
      PADDLE_GET(TensorDistAttr, param_grad_spmd.first[0]);
  TensorDistAttr grad_dist_attr_spmd =
      PADDLE_GET(TensorDistAttr, param_grad_spmd.first[1]);

  VLOG(3) << "The source dims mapping for param is: "
          << auto_parallel::str_join(param.dist_attr().dims_mapping());
  VLOG(3) << "The source dims mapping for grad is: "
          << auto_parallel::str_join(grad.dist_attr().dims_mapping());
  VLOG(3) << "The inter dims mapping for param (master param if available) "
          << "after elementwise spmd is: "
          << auto_parallel::str_join(param.dist_attr().dims_mapping());
  VLOG(3) << "The inter dims mapping for grad after elementwise spmd is: "
          << auto_parallel::str_join(grad.dist_attr().dims_mapping());

  TensorDistAttr param_dist_attr =
      CopyTensorDistAttrForOutput(param_dist_attr_spmd);
  TensorDistAttr grad_dist_attr =
      CopyTensorDistAttrForOutput(grad_dist_attr_spmd);
  TensorDistAttr lr_dist_attr =
      CopyTensorDistAttrForOutput(learning_rate.dist_attr());
  TensorDistAttr master_param_dist_attr =
      master_param.initialized()
          ? CopyTensorDistAttrForOutput(master_param.dist_attr())
          : TensorDistAttr();
  param_dist_attr.set_dims_mapping(param_dist_attr_spmd.dims_mapping());
  grad_dist_attr.set_dims_mapping(grad_dist_attr_spmd.dims_mapping());
  if (master_param.initialized()) {
    master_param_dist_attr.set_dims_mapping(
        param_dist_attr_spmd.dims_mapping());
  }
  lr_dist_attr.set_dims_mapping(learning_rate.dist_attr().dims_mapping());

  return {
      {param_dist_attr, lr_dist_attr, grad_dist_attr, master_param_dist_attr},
      {param_dist_attr, master_param_dist_attr}};
}

}  // namespace phi::distributed
