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

#include "paddle/phi/infermeta/spmd_rules/concat.h"

#include <limits>
#include <set>

#include "glog/logging.h"

#include "paddle/phi/infermeta/spmd_rules/elementwise.h"
#include "paddle/phi/infermeta/spmd_rules/utils.h"

namespace phi::distributed {

std::tuple<std::string, std::string> FillConcatNotation(int64_t n_axis,
                                                        int64_t concat_axis) {
  PADDLE_ENFORCE_GT(
      n_axis,
      concat_axis,
      common::errors::InvalidArgument(
          "n_axis [%d] and concat_axis[%d] not match", n_axis, concat_axis));
  static const std::string alphabet = "abcdefghijlopqrstuvwxyz";
  PADDLE_ENFORCE_GT(alphabet.size(),
                    static_cast<size_t>(n_axis),
                    common::errors::InvalidArgument(
                        "alphabet.size() [%d]; n_axis [%d] is too large",
                        alphabet.size(),
                        n_axis));
  std::string all_axis = alphabet.substr(0, n_axis);
  std::string align_axis =
      std::string(all_axis.begin(), all_axis.begin() + concat_axis) +
      std::string(all_axis.begin() + concat_axis + 1, all_axis.end());
  return {all_axis, align_axis};
}

SpmdInfo ConcatInferSpmd(const std::vector<DistMetaTensor>& x, int axis) {
  /*
# paddle.concat requires all tensors must either have the same shape (except
# in the concatenating dimension) or be "empty". "Empty" here strictly means
# tensor.ndim == 0. When tensor.ndim > 0, it will be treated
# as a non-empty tensor and the shape must match on non-cat dimensions.
 */

  // 1、check tensors shapes
  std::vector<std::vector<int64_t>> tensor_shapes;
  std::transform(x.begin(),
                 x.end(),
                 std::back_inserter(tensor_shapes),
                 [](const DistMetaTensor& meta) {
                   return common::vectorize<int64_t>(meta.dims());
                 });
  bool all_empty =
      std::all_of(tensor_shapes.begin(), tensor_shapes.end(), IsEmpty);
  if (all_empty) {
    return SpmdInfo();
  }

  auto non_empty_iter =
      std::find_if(tensor_shapes.begin(), tensor_shapes.end(), [](auto& shape) {
        return !IsEmpty(shape);
      });
  auto non_empty_index = non_empty_iter - tensor_shapes.begin();
  int64_t ndim = static_cast<int64_t>(tensor_shapes[non_empty_index].size());
  // normalize dim
  auto dim = axis < 0 ? ndim + axis : axis;
  std::vector<TensorDistAttr> input_attrs;
  std::transform(
      x.begin(), x.end(), std::back_inserter(input_attrs), [](auto& meta) {
        return meta.dist_attr();
      });

  std::string all_axis;
  std::string align_axis;
  std::tie(all_axis, align_axis) = FillConcatNotation(ndim, dim);
  std::vector<std::string> axis_names(input_attrs.size(), all_axis);
  AlignDimsSharding(
      &input_attrs, tensor_shapes, axis_names, {}, align_axis, true);

  auto out_dist_attr =
      CopyTensorDistAttrForOutput(input_attrs[non_empty_index]);
  out_dist_attr.set_dims_mapping(input_attrs[non_empty_index].dims_mapping());
  VLOG(4) << "concat out " << out_dist_attr.to_string();
  return {{input_attrs}, {out_dist_attr}};
}

SpmdInfo ConcatInferSpmdReverse(const std::vector<DistMetaTensor>& x,
                                const DistMetaTensor& output,
                                int axis) {
  auto out_dist_attr = output.dist_attr();
  out_dist_attr = UnShardTensorDims(out_dist_attr, {axis});
  auto n_inputs = x.size();
  TensorDistAttr input_attr = CopyTensorDistAttrForOutput(out_dist_attr);
  const auto& input_dim_mapping = out_dist_attr.dims_mapping();
  input_attr.set_dims_mapping(input_dim_mapping);
  std::vector<TensorDistAttr> input_attrs(n_inputs, input_attr);
  return {{input_attrs}, {output.dist_attr()}};
}

SpmdInfo ConcatInferSpmdDynamic(const std::vector<DistMetaTensor>& x,
                                const Scalar& axis) {
  return ConcatInferSpmd(x, axis.to<int32_t>());
}

SpmdInfo ConcatGradInferSpmdDynamic(const std::vector<DistMetaTensor>& x,
                                    const DistMetaTensor& output_grad,
                                    const Scalar& axis) {
  // 1、check tensors shapes
  std::vector<std::vector<int64_t>> tensor_shapes;
  std::transform(x.begin(),
                 x.end(),
                 std::back_inserter(tensor_shapes),
                 [](const DistMetaTensor& meta) {
                   return common::vectorize<int64_t>(meta.dims());
                 });
  bool all_empty =
      std::all_of(tensor_shapes.begin(), tensor_shapes.end(), IsEmpty);
  if (all_empty) {
    return SpmdInfo();
  }

  auto non_empty_iter =
      std::find_if(tensor_shapes.begin(), tensor_shapes.end(), [](auto& shape) {
        return !IsEmpty(shape);
      });
  auto non_empty_index = non_empty_iter - tensor_shapes.begin();
  int64_t ndim = static_cast<int64_t>(tensor_shapes[non_empty_index].size());
  auto dim = axis.to<int64_t>();
  // normalize dim
  dim = dim < 0 ? ndim + dim : dim;
  std::vector<TensorDistAttr> input_attrs;
  std::transform(
      x.begin(), x.end(), std::back_inserter(input_attrs), [](auto& meta) {
        return meta.dist_attr();
      });
  input_attrs.push_back(output_grad.dist_attr());
  tensor_shapes.push_back(common::vectorize<int64_t>(output_grad.dims()));
  std::string all_axis;
  std::string align_axis;
  std::tie(all_axis, align_axis) = FillConcatNotation(ndim, dim);
  std::vector<std::string> axis_names(input_attrs.size(), all_axis);
  AlignDimsSharding(
      &input_attrs, tensor_shapes, axis_names, {}, align_axis, true);
  auto output_grad_attr = input_attrs.back();
  input_attrs.pop_back();
  std::vector<TensorDistAttr> inputs_grad = input_attrs;
  return {{input_attrs, output_grad_attr}, {inputs_grad}};
}

}  // namespace phi::distributed
