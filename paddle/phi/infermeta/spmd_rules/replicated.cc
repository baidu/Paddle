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

#include "paddle/phi/infermeta/spmd_rules/replicated.h"

#include "glog/logging.h"

#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/inferspmd_utils.h"
#include "paddle/phi/core/distributed/auto_parallel/utils.h"

namespace phi::distributed {

using phi::distributed::auto_parallel::str_join;

////////////////// Utils Functions //////////////////
std::vector<int64_t> GetReplicatedDimsMapping(const int ndim) {
  std::vector<int64_t> dims_mapping(ndim, -1);
  return dims_mapping;
}

////////////////// InferMeta(Contains SPMD) Functions //////////////////
SpmdInfo ReplicatedInferSpmd(const std::vector<const DistMetaTensor*>& ins,
                             const std::vector<const DistMetaTensor*>& outs) {
  // step1: Build Einsum Notation for input tensor's batch axis
  int64_t ninputs = static_cast<int64_t>(ins.size());
  int64_t noutputs = static_cast<int64_t>(outs.size());

  // Step2: Unshard Output's Dims Mapping.
  std::vector<TensorDistAttr> output_dist_attrs;
  for (int64_t i = 0; i < noutputs; i++) {
    VLOG(4) << outs[i]->dist_attr().to_string();
    VLOG(4) << outs[i]->dims().to_str();
    int ndim = outs[i]->dims().size();
    TensorDistAttr dist_attr_dst =
        CopyTensorDistAttrForOutput(ins[0]->dist_attr());
    std::vector<int64_t> dst_dims_mapping = GetReplicatedDimsMapping(ndim);
    dist_attr_dst.set_dims_mapping(dst_dims_mapping);
    output_dist_attrs.emplace_back(dist_attr_dst);
  }

  // Step3: Merge and get Inputs' Batch Axis New Dims Mapping.
  std::vector<TensorDistAttr> dst_input_dist_attrs;
  for (int64_t i = 0; i < ninputs; i++) {
    // `ndim == -1` means input is nullptr
    int ndim = ins[i]->dims().size();
    if (ndim == -1) {
      continue;
    }
    TensorDistAttr dist_attr_dst =
        CopyTensorDistAttrForOutput(ins[i]->dist_attr());
    std::vector<int64_t> dst_dims_mapping = GetReplicatedDimsMapping(ndim);
    dist_attr_dst.set_dims_mapping(dst_dims_mapping);
    dst_input_dist_attrs.emplace_back(dist_attr_dst);
  }

  VLOG(4) << "ReplicatedSpmd InferForward:";
  for (int64_t i = 0; i < ninputs; i++) {
    if (ins[i]->dims().size() == -1) {
      continue;
    }
    VLOG(4) << "Input" << std::to_string(i) << " shape: ["
            << str_join(common::vectorize(ins[i]->dims())) << "] "
            << "src_dims_mapping: ["
            << str_join(ins[i]->dist_attr().dims_mapping()) << "] "
            << "dst_dims_mapping: ["
            << str_join(dst_input_dist_attrs[i].dims_mapping()) << "]";
  }

  for (int64_t i = 0; i < noutputs; i++) {
    VLOG(4) << "Output" << std::to_string(i) << " shape: ["
            << str_join(common::vectorize(outs[i]->dims())) << "] "
            << "dst_dims_mapping: ["
            << str_join(output_dist_attrs[i].dims_mapping()) << "]";
  }

  return {ToArgDistAttr(dst_input_dist_attrs),
          ToArgDistAttr(output_dist_attrs)};
}

SpmdInfo ReplicatedInferSpmdReverse(
    const std::vector<const DistMetaTensor*>& ins,
    const std::vector<const DistMetaTensor*>& outs) {
  // step1: Build Einsum Notation for input tensor's batch axis
  int64_t ninputs = static_cast<int64_t>(ins.size());
  int64_t noutputs = static_cast<int64_t>(outs.size());

  // Step2: Unshard Output's Dims Mapping.
  std::vector<TensorDistAttr> output_dist_attrs;
  for (int64_t i = 0; i < noutputs; i++) {
    int ndim = outs[i]->dims().size();
    TensorDistAttr dist_attr_dst =
        CopyTensorDistAttrForOutput(outs[i]->dist_attr());
    std::vector<int64_t> dst_dims_mapping = GetReplicatedDimsMapping(ndim);
    dist_attr_dst.set_dims_mapping(dst_dims_mapping);
    output_dist_attrs.emplace_back(dist_attr_dst);
  }

  // Step3: Merge and get Inputs' Batch Axis New Dims Mapping.
  std::vector<TensorDistAttr> dst_input_dist_attrs;
  for (int64_t i = 0; i < ninputs; i++) {
    int ndim = ins[i]->dims().size();
    TensorDistAttr dist_attr_dst =
        CopyTensorDistAttrForOutput(ins[i]->dist_attr());
    std::vector<int64_t> dst_dims_mapping = GetReplicatedDimsMapping(ndim);
    dist_attr_dst.set_dims_mapping(dst_dims_mapping);
    dst_input_dist_attrs.emplace_back(dist_attr_dst);
  }

  VLOG(4) << "ReplicatedSpmd InferBackward:";
  for (int64_t i = 0; i < noutputs; i++) {
    VLOG(4) << "Output" << std::to_string(i) << " shape: ["
            << str_join(common::vectorize(outs[i]->dims())) << "] "
            << "src_dims_mapping: ["
            << str_join(outs[i]->dist_attr().dims_mapping()) << "] "
            << "dst_dims_mapping: ["
            << str_join(output_dist_attrs[i].dims_mapping()) << "]";
  }

  for (int64_t i = 0; i < ninputs; i++) {
    VLOG(4) << "Input" << std::to_string(i) << " shape: ["
            << str_join(common::vectorize(ins[i]->dims())) << "] "
            << "dst_dims_mapping: ["
            << str_join(dst_input_dist_attrs[i].dims_mapping()) << "]";
  }

  return {ToArgDistAttr(dst_input_dist_attrs),
          ToArgDistAttr(output_dist_attrs)};
}

SpmdInfo ReplicatedInferDynamic(
    const std::vector<paddle::variant<const DistMetaTensor*,
                                      const std::vector<DistMetaTensor>*>>&
        inputs) {
  std::vector<const DistMetaTensor*> nonnull_inputs;
  int64_t ninputs = static_cast<int64_t>(inputs.size());
  SpmdInfo spmd_info;

  auto build_tensor_dist_attr = [&nonnull_inputs](
                                    const DistMetaTensor& dist_meta_tensor) {
    int ndim = dist_meta_tensor.dims().size();
    TensorDistAttr dist_attr_dst =
        CopyTensorDistAttrForOutput(dist_meta_tensor.dist_attr());
    // `ndim == -1` means input is nullptr
    if (ndim >= 0) {
      std::vector<int64_t> dst_dims_mapping = GetReplicatedDimsMapping(ndim);
      dist_attr_dst.set_dims_mapping(dst_dims_mapping);
      nonnull_inputs.push_back(&dist_meta_tensor);
    }
    return dist_attr_dst;
  };

  for (int64_t i = 0; i < ninputs; i++) {
    if (paddle::holds_alternative<const DistMetaTensor*>(inputs[i])) {
      auto dist_meta_tensor_ptr = paddle::get<0>(inputs[i]);
      auto& dist_meta_tensor = *dist_meta_tensor_ptr;
      auto dist_attr_dst = build_tensor_dist_attr(dist_meta_tensor);
      VLOG(4) << "input " << i << ": dist attr: " << dist_attr_dst.to_string();
      spmd_info.first.emplace_back(dist_attr_dst);
    } else {
      std::vector<phi::distributed::TensorDistAttr> list_dist_attr;
      auto dist_meta_tensors_ptr = paddle::get<1>(inputs[i]);
      auto& dist_meta_tensors = *dist_meta_tensors_ptr;
      for (const auto& dist_meta_tensor : dist_meta_tensors) {
        auto dist_attr_dst = build_tensor_dist_attr(dist_meta_tensor);
        VLOG(4) << "input " << i
                << ": dist attr: " << dist_attr_dst.to_string();
        list_dist_attr.emplace_back(std::move(dist_attr_dst));
      }
      spmd_info.first.emplace_back(std::move(list_dist_attr));
    }
  }
  return spmd_info;
}

}  // namespace phi::distributed
