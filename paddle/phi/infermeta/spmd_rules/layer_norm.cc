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

#include "paddle/phi/infermeta/spmd_rules/layer_norm.h"

#include "glog/logging.h"

#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/inferspmd_utils.h"
#include "paddle/phi/core/distributed/auto_parallel/utils.h"
#include "paddle/phi/infermeta/spmd_rules/utils.h"

namespace phi::distributed {

using phi::distributed::auto_parallel::str_join;

SpmdInfo LayerNormInferSpmd(const DistMetaTensor& x,
                            const DistMetaTensor& scale,
                            const DistMetaTensor& bias,
                            float epsilon,
                            int begin_norm_axis) {
  // Step0: verify input args based on layer_norm logic
  auto x_shape = common::vectorize(x.dims());
  auto scale_shape = common::vectorize(scale.dims());
  auto bias_shape = common::vectorize(bias.dims());
  int x_ndim = static_cast<int>(x_shape.size());
  int scale_ndim = static_cast<int>(scale_shape.size());
  int bias_ndim = static_cast<int>(bias_shape.size());
  const TensorDistAttr& x_dist_attr_src = x.dist_attr();
  std::vector<int64_t> x_dims_mapping = x_dist_attr_src.dims_mapping();
  std::vector<int64_t> scale_dims_mapping = scale.dist_attr().dims_mapping();
  std::vector<int64_t> bias_dims_mapping = bias.dist_attr().dims_mapping();

  PADDLE_ENFORCE_EQ(
      scale_ndim,
      1,
      common::errors::InvalidArgument(
          "The ndim of scale in layer_norm should be 1, but got [%d].",
          scale_ndim));

  PADDLE_ENFORCE_EQ(
      bias_ndim,
      1,
      common::errors::InvalidArgument(
          "The ndim of bias in layer_norm should be 1, but got [%d].",
          bias_ndim));

  // Step1: Build Einsum Notation
  // ijk,k,k->ijk,z,z (x,scale,bias->out,mean,variance, begin_norm_axis=2, z=ij)
  // ijkl,y(kl),y(kl)->ijkl,z(ij),z(ij) (x,scale,bias->out,mean,variance,
  // begin_norm_axis=2, z=ij, y=kl)
  std::string alphabet = "ijklmnopqrstuvwxyz";
  // get input notation
  // Because the mean and variance is 'flattened' from
  // x[0:begin_norm_axis], only the first axis of x can
  // be sharded
  std::string x_axes(x_ndim, '1');
  std::string mean_axes(begin_norm_axis, '1');
  std::string variance_axes(begin_norm_axis, '1');
  // allow axis before begin_norm_axis be sharded
  for (int i = 0; i < begin_norm_axis; ++i) {
    x_axes[i] = alphabet[i];
    mean_axes[i] = alphabet[i];
    variance_axes[i] = alphabet[i];
  }
  // x_axes[0] = alphabet[0];
  std::string scale_axes(1, x_axes[x_ndim - 1]);
  std::string bias_axes(1, x_axes[x_ndim - 1]);

  // get output notation
  std::string out_axes = x_axes;

  // Step2: Sharding Propagation
  // Step2.1: merge input sharding
  // As the mean and variance in outputs are `flattened` from
  // x[0:begin_norm_axis], only the first axis can be sharded,
  // the axes 1 to begin_norm_axis-1 are set to be replicated.
  std::fill(x_dims_mapping.begin() + begin_norm_axis, x_dims_mapping.end(), -1);
  std::unordered_map<std::string, int64_t> axis_to_dim_map =
      ShardingMergeForTensors({{x_axes, x_dims_mapping}});

  // Step2.2: infer output dims mapping
  TensorDistAttr out_dist_attr = CopyTensorDistAttrForOutput(x_dist_attr_src);
  TensorDistAttr mean_dist_attr = CopyTensorDistAttrForOutput(x_dist_attr_src);
  TensorDistAttr variance_dist_attr =
      CopyTensorDistAttrForOutput(x_dist_attr_src);
  out_dist_attr.set_dims_mapping(
      GetDimsMappingForAxes(out_axes, axis_to_dim_map));
  mean_dist_attr.set_dims_mapping(
      GetDimsMappingForAxes(mean_axes, axis_to_dim_map));
  variance_dist_attr.set_dims_mapping(
      GetDimsMappingForAxes(variance_axes, axis_to_dim_map));

  // Step2.3: update input dims mapping
  TensorDistAttr x_dist_attr_dst = CopyTensorDistAttrForOutput(x_dist_attr_src);
  TensorDistAttr scale_dist_attr_dst =
      CopyTensorDistAttrForOutput(scale.dist_attr());
  TensorDistAttr bias_dist_attr_dst =
      CopyTensorDistAttrForOutput(bias.dist_attr());
  x_dist_attr_dst.set_dims_mapping(x_dims_mapping);
  // TODO(zhiqiu): support sharding on scale and bias
  // Now, apply replicating.
  scale_dist_attr_dst.set_dims_mapping({-1});
  bias_dist_attr_dst.set_dims_mapping({-1});

  // Step2.4.  handle input and out tensor partial
  // LayerNorm not support
  VLOG(4) << "LayerNormInferSpmd:";
  VLOG(4) << "begin_norm_axis: " << begin_norm_axis;
  VLOG(4) << "Einsum Notation: " << x_axes << "," << scale_axes << ","
          << bias_axes << "-->" << out_axes << "," << mean_axes << ","
          << variance_axes;
  VLOG(4) << "X"
          << " shape: [" << str_join(x_shape) << "] "
          << "src_dims_mapping: [" << str_join(x_dist_attr_src.dims_mapping())
          << "] "
          << "dst_dims_mapping: [" << str_join(x_dims_mapping) << "]";
  VLOG(4) << "Scale"
          << " shape: [" << str_join(scale_shape) << "] "
          << "src_dims_mapping: [" << str_join(scale_dims_mapping) << "] "
          << "dst_dims_mapping: ["
          << str_join(scale_dist_attr_dst.dims_mapping()) << "]";
  VLOG(4) << "Bias"
          << " shape: [" << str_join(bias_shape) << "] "
          << "src_dims_mapping: [" << str_join(bias_dims_mapping) << "] "
          << "dst_dims_mapping: ["
          << str_join(bias_dist_attr_dst.dims_mapping()) << "]";
  VLOG(4) << "Out dims mapping: [" << str_join(out_dist_attr.dims_mapping())
          << "]";
  VLOG(4) << "Mean dims mapping: [" << str_join(mean_dist_attr.dims_mapping())
          << "]";
  VLOG(4) << "Variance dims mapping: ["
          << str_join(variance_dist_attr.dims_mapping()) << "]";
  VLOG(4) << std::endl;

  return {{x_dist_attr_dst, scale_dist_attr_dst, bias_dist_attr_dst},
          {out_dist_attr, mean_dist_attr, variance_dist_attr}};
}

SpmdInfo LayerNormInferSpmdReverse(const DistMetaTensor& x,
                                   const DistMetaTensor& scale,
                                   const DistMetaTensor& bias,
                                   const DistMetaTensor& out,
                                   const DistMetaTensor& mean,
                                   const DistMetaTensor& variance,
                                   float epsilon,
                                   int begin_norm_axis) {
  // Step0: Verify input args based on layer_norm logic
  auto x_shape = common::vectorize(x.dims());
  auto out_shape = common::vectorize(out.dims());
  auto mean_shape = common::vectorize(mean.dims());
  auto variance_shape = common::vectorize(variance.dims());
  int x_ndim = static_cast<int>(x_shape.size());
  int out_ndim = static_cast<int>(out_shape.size());
  int mean_ndim = static_cast<int>(mean_shape.size());
  int variance_ndim = static_cast<int>(variance_shape.size());
  auto out_dist_attr_src = out.dist_attr();
  auto mean_dist_attr_src = mean.dist_attr();
  auto variance_dist_attr_src = variance.dist_attr();
  std::vector<int64_t> out_dims_mapping = out_dist_attr_src.dims_mapping();
  std::vector<int64_t> mean_dims_mapping = mean_dist_attr_src.dims_mapping();
  std::vector<int64_t> variance_dims_mapping =
      variance_dist_attr_src.dims_mapping();
  PADDLE_ENFORCE_EQ(
      out_ndim,
      out_dims_mapping.size(),
      common::errors::InvalidArgument("The Tensor Out's rank [%d] and Out's "
                                      "dims_mapping size [%d] are not matched.",
                                      out_ndim,
                                      out_dims_mapping.size()));
  PADDLE_ENFORCE_EQ(
      mean_ndim,
      mean_dims_mapping.size(),
      common::errors::InvalidArgument("The Tensor Mean's rank [%d] and Mean's "
                                      "dims_mapping size [%d] are not matched.",
                                      mean_ndim,
                                      mean_dims_mapping.size()));
  PADDLE_ENFORCE_EQ(variance_ndim,
                    variance_dims_mapping.size(),
                    common::errors::InvalidArgument(
                        "The Tensor Variance's rank [%d] and Variance's "
                        "dims_mapping size [%d] are not matched.",
                        variance_ndim,
                        variance_dims_mapping.size()));
  // Step1: Build Einsum Notation
  // ijk,k,k->ijk,z,z (x,scale,bias->out,mean,variance, begin_norm_axis=2, z=ij)
  // ijkl,y(kl),y(kl)->ijkl,z(ij),z(ij) (x,scale,bias->out,mean,variance,
  // begin_norm_axis=2, z=ij, y=kl)
  std::string alphabet = "ijklmnopqrstuvwxyz";
  // the axes after norm_axis should be replicated,
  // so set their notation to '1'.
  std::string x_axes(x_ndim, '1');
  std::string mean_axes(begin_norm_axis, '1');
  std::string variance_axes(begin_norm_axis, '1');
  // allow axis before begin_norm_axis be sharded
  for (int i = 0; i < begin_norm_axis; ++i) {
    x_axes[i] = alphabet[i];
    mean_axes[i] = alphabet[i];
    variance_axes[i] = alphabet[i];
  }

  std::string scale_axes(1, x_axes[x_ndim - 1]);
  std::string bias_axes(1, x_axes[x_ndim - 1]);
  std::string out_axes = x_axes;

  // Step2: Sharding Propagation
  // For the axes after norm_axis in both input and output tensors,
  // set their dims mappings to -1. For the other axes, set input
  // tensor's dims mapping the same as output tensor's dims mapping.
  // step2.1 merge dims mappings of output, mean, variance.
  std::vector<std::pair<std::string, std::vector<int64_t>>> axes_sharding_info;
  axes_sharding_info.emplace_back(out_axes, out_dims_mapping);
  axes_sharding_info.emplace_back(mean_axes, mean_dims_mapping);
  axes_sharding_info.emplace_back(variance_axes, variance_dims_mapping);
  std::unordered_map<std::string, int64_t> axis_to_dim_map =
      ShardingMergeForTensors(axes_sharding_info);

  // Step2.2 infer input dims mapping
  std::vector<int64_t> x_dims_mapping =
      GetDimsMappingForAxes(x_axes, axis_to_dim_map);
  std::vector<TensorDistAttr> input_dist_attrs;
  input_dist_attrs.emplace_back(x.dist_attr());
  input_dist_attrs.emplace_back(scale.dist_attr());
  input_dist_attrs.emplace_back(bias.dist_attr());

  input_dist_attrs[0].set_dims_mapping(x_dims_mapping);
  // set bias and scale to be replicated
  input_dist_attrs[1].set_dims_mapping({-1});
  input_dist_attrs[2].set_dims_mapping({-1});

  // Step2.3 Update output dims mappings with merged one
  std::vector<TensorDistAttr> output_dist_attrs;
  output_dist_attrs.emplace_back(out_dist_attr_src);
  output_dist_attrs.emplace_back(mean_dist_attr_src);
  output_dist_attrs.emplace_back(variance_dist_attr_src);
  output_dist_attrs[0].set_dims_mapping(
      GetDimsMappingForAxes(out_axes, axis_to_dim_map));
  output_dist_attrs[1].set_dims_mapping(
      GetDimsMappingForAxes(mean_axes, axis_to_dim_map));
  output_dist_attrs[2].set_dims_mapping(
      GetDimsMappingForAxes(variance_axes, axis_to_dim_map));

  VLOG(4) << "LayerNormInferSpmdReverse:";
  VLOG(4) << "begin_norm_axis: " << begin_norm_axis;
  VLOG(4) << "Einsum Notation: " << x_axes << "," << scale_axes << ","
          << bias_axes << "-->" << out_axes << "," << mean_axes << ","
          << variance_axes;
  VLOG(4) << "Out"
          << " shape: [" << str_join(out_shape) << "] "
          << " src_dims_mapping: [" << str_join(out_dims_mapping) << "] "
          << "dst_dims_mapping: ["
          << str_join(output_dist_attrs[0].dims_mapping()) << "]";
  VLOG(4) << "Mean"
          << " shape: [" << str_join(mean_shape) << "] "
          << " src_dims_mapping: [" << str_join(mean_dims_mapping) << "] "
          << "dst_dims_mapping: ["
          << str_join(output_dist_attrs[1].dims_mapping()) << "]";
  VLOG(4) << "Variance"
          << " shape: [" << str_join(variance_shape) << "] "
          << " src_dims_mapping: [" << str_join(variance_dims_mapping) << "] "
          << "dst_dims_mapping: ["
          << str_join(output_dist_attrs[2].dims_mapping()) << "]";

  for (int i = 0, n = static_cast<int>(input_dist_attrs.size()); i < n; i++) {
    VLOG(4) << "Input" << std::to_string(i) << " dims_mapping: ["
            << str_join(input_dist_attrs[i].dims_mapping()) << "]";
  }
  VLOG(4) << std::endl;

  return {ToArgDistAttr(input_dist_attrs), ToArgDistAttr(output_dist_attrs)};
}

std::tuple<std::vector<std::string>, std::string> BuildLayerNormGradEinsum(
    int64_t input_rank, int64_t begin_norm_axis) {
  std::string alphabet = "ijklmnopqrstuvwxyz";
  std::string x_notation = alphabet.substr(0, input_rank);
  std::string mean_variance_notation = x_notation.substr(0, begin_norm_axis);
  std::string align_notation = x_notation.substr(0, begin_norm_axis);
  return {
      {x_notation, mean_variance_notation, mean_variance_notation, x_notation},
      align_notation};
}

SpmdInfo LayerNormGradInferSpmd(const DistMetaTensor& x,
                                const DistMetaTensor& scale,
                                const DistMetaTensor& bias,
                                const DistMetaTensor& mean,
                                const DistMetaTensor& variance,
                                const DistMetaTensor out_grad,
                                float epsilon,
                                int begin_norm_axis) {
  auto get_shape = [](const auto& meta) {
    return common::vectorize<int64_t>(meta.dims());
  };
  // 1、check tensors shapes
  auto x_shape = get_shape(x);
  auto scale_shape = get_shape(scale);
  auto bias_shape = get_shape(bias);
  auto mean_shape = get_shape(mean);
  auto variance_shape = get_shape(variance);
  auto out_grad_shape = get_shape(out_grad);
  PADDLE_ENFORCE_GE(
      x_shape.size(),
      begin_norm_axis,
      common::errors::InvalidArgument(
          "The Tensor x's rank [%d] and begin_norm_axis [%d] are not matched.",
          x_shape.size(),
          begin_norm_axis));
  PADDLE_ENFORCE_EQ(
      x_shape.size(),
      out_grad_shape.size(),
      common::errors::InvalidArgument("The Tensor x's rank [%d] and Tensor "
                                      "out_grad's rank [%d] are not matched.",
                                      x_shape.size(),
                                      out_grad_shape.size()));

  PADDLE_ENFORCE_EQ(
      scale_shape.size(),
      bias_shape.size(),
      common::errors::InvalidArgument("The Tensor scale's rank [%d] and Tensor "
                                      "bias's rank [%d] are not matched.",
                                      scale_shape.size(),
                                      bias_shape.size()));

  PADDLE_ENFORCE_EQ(
      mean_shape.size(),
      variance_shape.size(),
      common::errors::InvalidArgument("The Tensor mean's rank [%d] and Tensor "
                                      "variance's rank [%d] are not matched.",
                                      mean_shape.size(),
                                      variance_shape.size()));

  // 2、align sharding
  TensorDistAttr x_dist_attr;
  TensorDistAttr mean_dist_attr;
  TensorDistAttr variance_dist_attr;
  TensorDistAttr out_grad_dist_attr;

  std::vector<TensorDistAttr> dist_attrs;
  dist_attrs.push_back(x.dist_attr());
  dist_attrs.push_back(mean.dist_attr());
  dist_attrs.push_back(variance.dist_attr());
  out_grad_dist_attr = out_grad.dist_attr();
  out_grad_dist_attr.clean_partial_status();
  dist_attrs.push_back(out_grad_dist_attr);

  if (begin_norm_axis > 0) {
    std::vector<std::vector<int64_t>> shapes = {
        x_shape, mean_shape, variance_shape, x_shape};
    std::vector<std::string> annotations;
    std::string align_annotation;
    std::tie(annotations, align_annotation) =
        BuildLayerNormGradEinsum(x_shape.size(), begin_norm_axis);

    // Sharding Propagation
    std::vector<std::pair<std::string, std::vector<int64_t>>>
        axes_sharding_info;
    auto x_dims_mapping = dist_attrs[0].dims_mapping();
    auto out_grad_dims_mapping = dist_attrs[3].dims_mapping();
    std::fill(
        x_dims_mapping.begin() + begin_norm_axis, x_dims_mapping.end(), -1);
    std::fill(out_grad_dims_mapping.begin() + begin_norm_axis,
              out_grad_dims_mapping.end(),
              -1);
    axes_sharding_info.emplace_back(annotations[0], x_dims_mapping);
    axes_sharding_info.emplace_back(annotations[1],
                                    dist_attrs[1].dims_mapping());
    axes_sharding_info.emplace_back(annotations[2],
                                    dist_attrs[2].dims_mapping());
    axes_sharding_info.emplace_back(annotations[3], out_grad_dims_mapping);
    std::unordered_map<std::string, int64_t> axis_to_dim_map =
        ShardingMergeForTensors(axes_sharding_info);

    x_dist_attr = std::move(dist_attrs[0]);
    x_dist_attr.set_dims_mapping(
        GetDimsMappingForAxes(annotations[0], axis_to_dim_map));
    mean_dist_attr = std::move(dist_attrs[1]);
    mean_dist_attr.set_dims_mapping(
        GetDimsMappingForAxes(annotations[1], axis_to_dim_map));
    variance_dist_attr = std::move(dist_attrs[2]);
    variance_dist_attr.set_dims_mapping(
        GetDimsMappingForAxes(annotations[2], axis_to_dim_map));
    out_grad_dist_attr = std::move(dist_attrs[3]);
    out_grad_dist_attr.set_dims_mapping(
        GetDimsMappingForAxes(annotations[3], axis_to_dim_map));
  } else {
    x_dist_attr = GetReplicatedDistAttr(dist_attrs[0]);
    mean_dist_attr = GetReplicatedDistAttr(dist_attrs[1]);
    variance_dist_attr = GetReplicatedDistAttr(dist_attrs[2]);
    out_grad_dist_attr = GetReplicatedDistAttr(dist_attrs[3]);
  }
  // TODO(liuzhenhai): support sharded scale and bias
  TensorDistAttr scale_dist_attr = GetReplicatedDistAttr(scale.dist_attr());
  TensorDistAttr bias_dist_attr = GetReplicatedDistAttr(bias.dist_attr());
  TensorDistAttr x_grad_dist_attr = out_grad_dist_attr;
  TensorDistAttr scale_grad_dist_attr =
      GetReplicatedDistAttr(scale.dist_attr());
  TensorDistAttr bias_grad_dist_attr = GetReplicatedDistAttr(bias.dist_attr());
  // partial grad dim
  std::vector<int64_t> partial_on_dims;
  const auto& dim_mapping = x_dist_attr.dims_mapping();
  for (int i = 0; i < begin_norm_axis; ++i) {
    auto mapping = dim_mapping[i];
    if (mapping != -1) {
      partial_on_dims.push_back(mapping);
    }
  }
  scale_grad_dist_attr.set_partial_status(partial_on_dims);
  bias_grad_dist_attr.set_partial_status(partial_on_dims);

  VLOG(4) << "LayerNormGradInferSpmd:";
  VLOG(4) << "begin_norm_axis: " << begin_norm_axis;
  LogInputDistAttr("X", x_shape, x.dist_attr(), x_dist_attr);
  LogInputDistAttr("Scale", scale_shape, scale.dist_attr(), scale_dist_attr);
  LogInputDistAttr("Bias", bias_shape, bias.dist_attr(), bias_dist_attr);
  LogInputDistAttr("Mean", mean_shape, mean.dist_attr(), mean_dist_attr);
  LogInputDistAttr(
      "Variance", variance_shape, variance.dist_attr(), variance_dist_attr);
  LogInputDistAttr(
      "OutGrad", out_grad_shape, out_grad.dist_attr(), out_grad_dist_attr);
  LogOutputDistAttr("XGrad", x_grad_dist_attr);
  LogOutputDistAttr("ScaleGrad", scale_grad_dist_attr);
  LogOutputDistAttr("BiasGrad", bias_grad_dist_attr);
  VLOG(4) << std::endl;

  return SpmdInfo(
      {x_dist_attr,
       scale_dist_attr,
       bias_dist_attr,
       mean_dist_attr,
       variance_dist_attr,
       out_grad_dist_attr},
      {x_grad_dist_attr, scale_grad_dist_attr, bias_grad_dist_attr});
}

SpmdInfo FastLnInferSpmd(const DistMetaTensor& x,
                         const DistMetaTensor& scale,
                         const DistMetaTensor& bias,
                         float epsilon) {
  int begin_norm_axis = x.dims().size() - 1;
  VLOG(4) << "FastLnInferSpmd call LayerNormInferSpmd with begin_norm_axis="
          << begin_norm_axis;
  return LayerNormInferSpmd(x, scale, bias, epsilon, begin_norm_axis);
}

SpmdInfo FastLnGradInferSpmd(const DistMetaTensor& x,
                             const DistMetaTensor& scale,
                             const DistMetaTensor& mean,
                             const DistMetaTensor& invvar,
                             const DistMetaTensor& y_grad,
                             float epsilon) {
  int begin_norm_axis = x.dims().size() - 1;
  const DistMetaTensor& bias(scale);  // bias is not used in FastLnGrad
  VLOG(4)
      << "FastLnGradInferSpmd call LayerNormGradInferSpmd with begin_norm_axis="
      << begin_norm_axis << ", the input 'bias' will be ignored.";
  SpmdInfo spmd_info = LayerNormGradInferSpmd(
      x, scale, bias, mean, invvar, y_grad, epsilon, begin_norm_axis);
  spmd_info.first.erase(spmd_info.first.begin() + 2);  // remove bias_dist_attr
  return spmd_info;
}

}  // namespace phi::distributed
