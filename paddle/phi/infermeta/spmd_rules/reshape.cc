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

#include "paddle/phi/infermeta/spmd_rules/reshape.h"
#include <numeric>

#include "glog/logging.h"

#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/inferspmd_utils.h"
#include "paddle/phi/core/distributed/auto_parallel/utils.h"
#include "paddle/phi/infermeta/spmd_rules/dim_trans.h"
#include "paddle/phi/infermeta/spmd_rules/utils.h"

namespace phi::distributed {

using phi::distributed::auto_parallel::str_join;

// The target shape in reshape may contains a -1 dimension,
// this function is used to infer what the "-1" dimension is.
std::vector<int64_t> InferTargetShape(const std::vector<int64_t>& shape,
                                      int64_t len) {
  int64_t infer_idx = -1;
  for (int64_t i = 0, n = static_cast<int64_t>(shape.size()); i < n; i++) {
    if (shape[i] == -1) {
      PADDLE_ENFORCE_EQ(
          infer_idx,
          -1,
          common::errors::InvalidArgument(
              "There can't be more than one -1 dimension in target shape."));
      infer_idx = i;
    }
  }

  int64_t product =
      std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<>());
  if (product > 0) {
    PADDLE_ENFORCE_EQ(
        product,
        len,
        common::errors::InvalidArgument("The total size are not matched."));
    return std::vector<int64_t>(shape);
  } else {
    std::vector<int64_t> new_shape(shape);
    product = -product;
    int64_t infer_size = len / product;
    PADDLE_ENFORCE_EQ(
        len % infer_size,
        0,
        common::errors::InvalidArgument(
            "The total element number of the src tensor (%lld) is not "
            "divisible by the inferred size (%lld) of the -1 dimension.",
            len,
            infer_size));
    new_shape[infer_idx] = infer_size;
    return new_shape;
  }
}

// Compute how each dimension in target shape
// is obtained from the input dimensions
std::vector<std::shared_ptr<DimTrans>> MakeReshapeDimTrans(
    const std::vector<int64_t>& src_shape,
    const std::vector<int64_t>& tgt_shape) {
  std::vector<std::shared_ptr<DimTrans>> ret;
  int64_t total_elem_num_src = std::accumulate(
      src_shape.begin(), src_shape.end(), 1LL, std::multiplies<>());
  std::vector<int64_t> inferred_tgt_shape =
      InferTargetShape(tgt_shape, total_elem_num_src);

  int src_idx = 0, tgt_idx = 0;
  int s, t;
  int src_len, tgt_len;
  src_len = static_cast<int>(src_shape.size());
  tgt_len = static_cast<int>(inferred_tgt_shape.size());
  while (src_idx < src_len || tgt_idx < tgt_len) {
    std::vector<int64_t> src_dims, tgt_splitted_shape;
    if (src_idx >= src_len) {
      s = 1;
    } else {
      s = static_cast<int>(src_shape[src_idx]);
      src_dims.emplace_back(src_idx);
      src_idx++;
    }
    if (tgt_idx >= tgt_len) {
      t = 1;
    } else {
      t = static_cast<int>(inferred_tgt_shape[tgt_idx]);
      tgt_splitted_shape.emplace_back(t);
      tgt_idx++;
    }

    // deal with the singleton case
    if (s == 1 && t != 1) {
      // case [1] [a]
      tgt_idx--;
      tgt_splitted_shape.clear();
    } else if (s != 1 && t == 1) {
      src_idx--;
      src_dims.clear();
    } else {
      while (s != t) {
        if (s < t) {
          src_dims.emplace_back(src_idx);
          s *= static_cast<int>(src_shape[src_idx]);
          src_idx++;
        } else {
          tgt_splitted_shape.emplace_back(inferred_tgt_shape[tgt_idx]);
          t *= static_cast<int>(inferred_tgt_shape[tgt_idx]);
          tgt_idx++;
        }
      }
    }

    if (!tgt_splitted_shape.empty()) {
      std::vector<std::shared_ptr<DimTrans>> input_dims;
      for (auto in_dim : src_dims) {
        if (src_shape[in_dim] > 1) {
          input_dims.emplace_back(std::make_shared<InputDim>(in_dim));
        } else if (src_shape[in_dim] == 1 && s == 1 && t == 1) {
          // NOTE: for the case like:
          //    shape: [1, 512, 4096] --> [1, 2, 256, 4096],
          //    input dims_mapping: [0, 1, -1]
          //    expected output dims_mapping: [0, 1, -1, -1] (not [-1, 1, -1,
          //    -1])
          // In this case, the dim0 in target shape is 1 and it is from
          // dim0 in source shape. make the dim0's transformation be InputDim
          // rather than Singleton so that the sharding status can be
          // propagated.
          input_dims.emplace_back(std::make_shared<InputDim>(in_dim));
        }
      }
      std::shared_ptr<DimTrans> flatten = make_flatten(input_dims);

      for (int64_t i = 0, n = static_cast<int64_t>(tgt_splitted_shape.size());
           i < n;
           i++) {
        ret.emplace_back(make_split(flatten, tgt_splitted_shape, i));
      }
    }
  }
  return ret;
}

SpmdInfo ReshapeInferSpmd(const DistMetaTensor& x,
                          const std::vector<int64_t>& shape) {
  // Step0: Verify input args based on reshape logic
  auto x_shape = phi::vectorize(x.dims());

  int x_ndim = static_cast<int>(x_shape.size());
  int out_ndim = static_cast<int>(shape.size());
  auto x_dist_attr_src = x.dist_attr();
  std::vector<int64_t> x_dims_mapping = x_dist_attr_src.dims_mapping();
  PADDLE_ENFORCE_EQ(
      x_ndim,
      x_dims_mapping.size(),
      common::errors::InvalidArgument("The Tensor X's rank [%d] and X's "
                                      "dims_mapping size [%d] are not matched.",
                                      x_ndim,
                                      x_dims_mapping.size()));
  VLOG(4) << "ReshapeInferSpmd: X shape: [" << str_join(x_shape) << "]";
  VLOG(4) << "Out shape: [" << str_join(shape) << "]";

  // Step1: Build the transformation from
  // the original shape to the target shape

  // handle the case of dynamic shape, like [-1, -1, ...] --> [0, 0, ...].
  // This is used in inference but reshape allows only one '-1' in the
  // target shape, so set the shape to a special value '256'
  for (int i = 0; i < x_ndim; i++) {
    if (x_shape[i] == -1) {
      x_shape[i] = 256;
    }
  }

  // handle the '0' values in target shape, '0' indicates
  // that the target shape is equal to the source shape
  std::vector<int64_t> tgt_shape(shape);
  for (int64_t i = 0; i < out_ndim; i++) {
    if (tgt_shape[i] == 0) {
      tgt_shape[i] = x_shape[i];
    }
  }

  std::vector<std::shared_ptr<DimTrans>> trans =
      MakeReshapeDimTrans(x_shape, tgt_shape);

  // Step2: Infer the dims mapping of input (if reshard is
  // needed) and output from the dimension transformation.
  std::vector<std::vector<int64_t>> dims_mapping_vec =
      InferFromDimTrans(x, trans);

  // Step3: Update the dist attributes of input
  // and output with the inferred dims mapping.
  TensorDistAttr x_dist_attr_dst(x_dist_attr_src);
  x_dist_attr_dst.set_dims_mapping(dims_mapping_vec[0]);
  if (x_dist_attr_dst.dynamic_dims().size() !=
      x_dist_attr_dst.dims_mapping().size()) {
    VLOG(3) << "Reshape InferSPMD change input dist attr dynamic dims";
    x_dist_attr_dst.set_default_dynamic_dims(x_dist_attr_dst.dims_mapping());
  }
  TensorDistAttr out_dist_attr(x_dist_attr_src);
  out_dist_attr.set_dims_mapping(dims_mapping_vec[1]);
  if (out_dist_attr.dynamic_dims().size() !=
      out_dist_attr.dims_mapping().size()) {
    VLOG(3) << "Reshape InferSPMD change output dist attr dynamic dims";
    out_dist_attr.set_default_dynamic_dims(out_dist_attr.dims_mapping());
  }

  VLOG(4) << "Transformation from input to output:";
  for (int64_t i = 0, n = static_cast<int64_t>(trans.size()); i < n; i++) {
    std::shared_ptr<DimTrans> t = trans[i];
    VLOG(4) << "\tOut axis[" << i << "]: " << t->to_string();
  }
  VLOG(4) << "X dims_mapping_src: [" << str_join(x_dims_mapping)
          << "] dims_mapping_dst: [" << str_join(dims_mapping_vec[0]) << "]";
  VLOG(4) << "Out dims_mapping: [" << str_join(dims_mapping_vec[1]) << "]\n\n";

  return {{x_dist_attr_dst}, {out_dist_attr}};
}

SpmdInfo ReshapeInferSpmdReverse(const DistMetaTensor& x,
                                 const DistMetaTensor& out,
                                 const std::vector<int64_t>& shape) {
  // Step0: Verify input args based on reshape logic
  auto x_shape = common::vectorize(x.dims());
  auto out_shape = common::vectorize(out.dims());
  int x_ndim = static_cast<int>(x_shape.size());
  int out_ndim = static_cast<int>(out_shape.size());
  auto out_dist_attr_src = out.dist_attr();
  std::vector<int64_t> out_dims_mapping = out_dist_attr_src.dims_mapping();
  PADDLE_ENFORCE_EQ(
      out_ndim,
      out_dims_mapping.size(),
      common::errors::InvalidArgument("The Tensor Out's rank [%d] and Out's "
                                      "dims_mapping size [%d] are not matched.",
                                      out_ndim,
                                      out_dims_mapping.size()));
  VLOG(4) << "ReshapeInferSpmdReverse: Out shape: [" << str_join(out_shape)
          << "], X shape: [" << str_join(x_shape) << "]";

  // Step1: Build the transformation from the output shape
  // to original shape. This function infers the dims mapping
  // from output to input, we first get the transformation
  // from output to input so that we can infer the dims mapping
  // with the map from output axes to input axes.

  // handle the case of dynamic shape, like [-1, -1, ...] --> [0, 0, ...].
  // This is used in inference but reshape allows only one '-1' in the
  // target shape, so set the shape to a special value '256'
  for (int i = 0; i < x_ndim; i++) {
    if (x_shape[i] == -1) {
      x_shape[i] = 256;
    }
  }

  // handle the '0' values in target shape, '0' indicates
  // that the target shape is equal to the source shape
  std::vector<int64_t> tgt_shape(shape);
  for (int64_t i = 0; i < out_ndim; i++) {
    if (shape[i] == 0) {
      out_shape[i] = x_shape[i];
    }
  }

  // The out_shape may contain '-1', which will cause error
  // when inferring the transformation from out_shape to
  // x_shape, so infer the '-1' value before inferring DimTrans
  int64_t nelm =
      std::accumulate(x_shape.begin(), x_shape.end(), 1LL, std::multiplies<>());
  out_shape = InferTargetShape(out_shape, nelm);
  std::vector<std::shared_ptr<DimTrans>> trans =
      MakeReshapeDimTrans(out_shape, x_shape);

  // Step2: Infer the dims mapping of input with
  // output's dims_mapping and the transformation.
  std::vector<std::vector<int64_t>> dims_mapping_vec =
      InferFromDimTrans(out, trans);

  // Step3: Update the dist attributes of input
  // and output with the inferred dims mapping
  TensorDistAttr out_dist_attr_dst(out_dist_attr_src);
  out_dist_attr_dst.set_dims_mapping(dims_mapping_vec[0]);
  if (out_dist_attr_dst.dynamic_dims().size() !=
      out_dist_attr_dst.dims_mapping().size()) {
    VLOG(3) << "Reshape InferSPMD change output dist attr dynamic dims";
    out_dist_attr_dst.set_default_dynamic_dims(
        out_dist_attr_dst.dims_mapping());
  }
  TensorDistAttr x_dist_attr(x.dist_attr());
  x_dist_attr.set_dims_mapping(dims_mapping_vec[1]);
  if (x_dist_attr.dynamic_dims().size() != x_dist_attr.dims_mapping().size()) {
    VLOG(3) << "Reshape InferSPMD change input dist attr dynamic dims";
    x_dist_attr.set_default_dynamic_dims(x_dist_attr.dims_mapping());
  }

  VLOG(4) << "Transformation from output to input:";
  for (int64_t i = 0, n = static_cast<int64_t>(trans.size()); i < n; i++) {
    std::shared_ptr<DimTrans> t = trans[i];
    VLOG(4) << "\tX axis[" << i << "]: " << t->to_string();
  }
  VLOG(4) << "Out dims_mapping_src: [" << str_join(out_dims_mapping) << "] "
          << "dims_mapping_dst: [" << str_join(dims_mapping_vec[0]) << "]";
  VLOG(4) << "X dims_mapping: [" << str_join(dims_mapping_vec[1]) << "]\n\n";

  return {{x_dist_attr}, {out_dist_attr_dst}};
}

// FIXME(dev): XShape will be deprecated in the future, so we
// need unify inferSpmd into ReshapeInferSpmd function.
SpmdInfo ReshapeInferSpmdDynamic(const DistMetaTensor& x,
                                 const std::vector<int64_t>& shape) {
  auto spmd_info = ReshapeInferSpmd(x, shape);
  auto xshape_dist_dst = PADDLE_GET_CONST(TensorDistAttr, spmd_info.first[0]);
  auto xshape_dims_mapping = xshape_dist_dst.dims_mapping();
  xshape_dims_mapping.insert(xshape_dims_mapping.begin(), -1);
  xshape_dist_dst.set_dims_mapping(xshape_dims_mapping);
  spmd_info.second.emplace_back(xshape_dist_dst);
  return spmd_info;
}

SpmdInfo ReshapeGradInferSpmd(const DistMetaTensor& x,
                              const DistMetaTensor& out_grad) {
  std::vector<int64_t> out_grad_shape = common::vectorize(out_grad.dims());
  auto x_dist_tmp = x.dist_attr();
  auto tmp =
      ReshapeInferSpmd(DistMetaTensor(x.dims(), x_dist_tmp), out_grad_shape);
  // check no shard is needed
  const auto& x_dist_dst = PADDLE_GET_CONST(TensorDistAttr, tmp.first[0]);
  const auto& out_grad_dist_dst =
      PADDLE_GET_CONST(TensorDistAttr, tmp.second[0]);
  if (x_dist_dst.dims_mapping() != x_dist_tmp.dims_mapping()) {
    x_dist_tmp.set_dims_mapping(x_dist_dst.dims_mapping());
  }
  return {{x_dist_tmp, out_grad_dist_dst}, {x_dist_dst}};
}

}  // namespace phi::distributed
