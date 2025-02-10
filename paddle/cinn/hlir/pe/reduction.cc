// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/hlir/pe/reduction.h"

#include <paddle/cinn/ir/ir_base.h>

#include <algorithm>

#include "paddle/cinn/common/common.h"
#include "paddle/cinn/common/ir_util.h"
#include "paddle/cinn/hlir/pe/broadcast.h"
#include "paddle/cinn/hlir/pe/elementwise.h"
#include "paddle/cinn/hlir/pe/nn_util.h"
#include "paddle/cinn/ir/op/ir_operators.h"
#include "paddle/cinn/ir/tensor.h"
#include "paddle/cinn/lang/compute.h"
#include "paddle/cinn/utils/string.h"
#include "paddle/common/enforce.h"
namespace cinn {
namespace hlir {
namespace pe {

using ir::Tensor;
using lang::Compute;

/**
 * @brief transform reduction axes which could be empty or have negative
 * elements into real axes with valid dimension indices.
 *
 * @param ndim Number of dimensions of the output tensor.
 * @param axes The axes parameter.
 * @param real_axes A non-empty sorted array of valid dimension indices, with no
 * duplicates.
 *
 * @notes If the input axes are empty, the result will be axes including all
 * dimensions. If any input element is negative, it will be treated as an offset
 * from the last dimension (same as python indexing rules).
 */
void GetRealAxes(int ndim,
                 const std::vector<int>& axes,
                 std::vector<int>* real_axes) {
  PADDLE_ENFORCE_NOT_NULL(real_axes,
                          ::common::errors::InvalidArgument(
                              "The 'real_axes' pointer must not be null."));
  if (axes.empty()) {
    for (int i = 0; i < ndim; ++i) {
      real_axes->push_back(i);
    }
  } else {
    for (auto axis : axes) {
      if (axis < 0) {
        axis += ndim;
      }
      PADDLE_ENFORCE_LE(
          axis,
          ndim,
          ::common::errors::InvalidArgument("The axis(%d) exceeds the "
                                            "maximum dimension(%d).",
                                            axis,
                                            ndim));
      PADDLE_ENFORCE_GE(
          axis,
          0,
          ::common::errors::InvalidArgument("The axis(%d) is less than "
                                            "the minimum dimension(0).",
                                            axis));
      real_axes->push_back(axis);
    }
    real_axes->resize(std::unique(real_axes->begin(), real_axes->end()) -
                      real_axes->begin());
    std::sort(real_axes->begin(), real_axes->end());
  }
}

std::string Type2StrForReduce(cinn::common::Type type) {
  std::string suffix;
  if (type.is_int(32)) {
    return "_int32";
  } else if (type.is_int(64)) {
    return "_int64";
  } else if (type.is_bfloat16()) {
    return "_bf16";
  } else if (type.is_float16()) {
    return "_fp16";
  } else if (type.is_float(32)) {
    return "_fp32";
  } else if (type.is_float(64)) {
    return "_fp64";
  } else if (type.is_bool()) {
    return "";
  } else if (type.is_customized_type()) {
    return "_" + type.customized_type();
  }
  PADDLE_THROW(
      ::common::errors::InvalidArgument("Reduce type not supported: %s", type));
}

/**
 * @brief Calculate the target reduced shape.
 *
 * @param real_axes A non-empty sorted array of valid dimension indices, with no
 * duplicates.
 * @param output_shape The output Tensor shape.
 * @param tensor The input tensor.
 * @param keep_dims If this is set to true, the reduced axes are kept as
 * dimensions with size one. This enables the result to broadcast correctly
 * against the input array.
 */
void GetOutputShape(const std::vector<int>& real_axes,
                    std::vector<Expr>* output_shape,
                    const Tensor& tensor,
                    bool keep_dims) {
  PADDLE_ENFORCE_NOT_NULL(output_shape,
                          ::common::errors::InvalidArgument(
                              "The 'output_shape' pointer must not be null."));
  auto ndim = tensor->shape.size();
  if (keep_dims) {
    for (size_t i = 0; i < ndim; ++i) {
      if (std::find(real_axes.begin(), real_axes.end(), i) != real_axes.end()) {
        output_shape->push_back(cinn::common::make_one());
      } else {
        output_shape->push_back(tensor->shape[i]);
      }
    }
  } else {
    for (size_t i = 0; i < ndim; ++i) {
      if (std::find(real_axes.begin(), real_axes.end(), i) == real_axes.end()) {
        output_shape->push_back(tensor->shape[i]);
      }
    }
  }
  if (output_shape->empty()) {
    output_shape->push_back(cinn::common::make_one());
  }

  PADDLE_ENFORCE_EQ(!tensor->shape.empty(),
                    true,
                    ::common::errors::InvalidArgument(
                        "The 'tensor' shape must not be empty."));
  if (tensor->shape[0]->type() == Int(64)) {
    for (auto& shape_item : *output_shape) {
      shape_item->convert_int32_to_int64();
    }
  }
}

/*!
 * @brief Create a reduction PE.
 *
 * @param tensor The input tensor.
 * @param fn The reduction function eg. ReduceSum
 * @param output_shape The output Tensor shape.
 * @param real_axes The real axes where the reduction is performed.
 * @param squeeze_axes The real axes to squeeze. If unsqueezed, reduced axes
 * will have shape 1 in the output tensor.
 * @param initial Starting value for the sum.
 * @param output_name The name of the output Tensor.
 *
 * @return The result tensor.
 */
template <typename FuncOp>
Tensor DoReduce(const Tensor& tensor,
                const FuncOp& fn,
                const std::vector<Expr>& output_shape,
                const std::vector<int>& real_axes,
                const std::vector<int>& squeeze_axes,
                Expr initial,
                const std::string& output_name) {
  std::vector<Var> reduce_axes;
  int reduce_k_id = 0;
  for (auto& axis : real_axes) {
    std::string name =
        cinn::UniqName(std::string("reduce_k_") + std::to_string(reduce_k_id));
    reduce_axes.push_back(Var(tensor->shape[axis], name));
    reduce_k_id++;
  }
  auto compute = [&](const std::vector<Expr>& indices) -> Expr {
    std::vector<Expr> eval_indice;
    int indice_cnt = 0;
    int reduce_cnt = 0;

    // Set keepdim flags of indices.
    if (tensor->shape.size() == indices.size()) {
      for (const auto& i : real_axes) {
        VLOG(4) << "Set is_keepdim = true for var(" << i << ")";
        indices[i].as_var_ref()->is_keepdim = true;
      }
    }

    for (size_t i = 0; i < tensor->shape.size(); ++i) {
      bool squeeze_i = std::find(squeeze_axes.begin(), squeeze_axes.end(), i) !=
                       squeeze_axes.end();
      if (std::find(real_axes.begin(), real_axes.end(), i) != real_axes.end()) {
        eval_indice.push_back(reduce_axes[reduce_cnt]);
        reduce_cnt++;
        indice_cnt += !squeeze_i;
        continue;
      }
      eval_indice.push_back(indices[indice_cnt]);
      indice_cnt++;
    }
    return fn(tensor(eval_indice), reduce_axes, initial);
  };

  Tensor C = Compute(output_shape, compute, output_name);
  return C;
}

/**
 * @brief reduction PE
 *
 * @param tensor The input tensor.
 * @param axes The axes along which the reduction are performed.
 * @param fn The reduction function eg. ReduceSum
 * @param keep_dims If it is set true, the axes which are reduced are left in
 * the result as dimensions with size one.
 * @param initial Starting value for the sum.
 *
 * @return The result tensor.
 */
template <typename FuncOp>
Tensor Reduce(const Tensor& tensor,
              const std::vector<int>& axes,
              const FuncOp& fn,
              bool keep_dims,
              ir::Expr initial,
              const std::string& output_name) {
  auto ndim = tensor->shape.size();
  PADDLE_ENFORCE_GT(
      ndim,
      0,
      ::common::errors::InvalidArgument("Reduce tensor's dim must be "
                                        "more than 0"));
  std::vector<int> real_axes;
  GetRealAxes(static_cast<int>(ndim), axes, &real_axes);
  std::vector<Expr> output_shapes;
  GetOutputShape(real_axes, &output_shapes, tensor, keep_dims);
  return DoReduce(tensor,
                  fn,
                  output_shapes,
                  real_axes,
                  keep_dims ? std::vector<int>() : real_axes,
                  initial,
                  output_name);
}

Tensor ReduceSum(const Tensor& A,
                 const std::vector<int>& axes,
                 const bool keep_dims,
                 const std::string& output_name) {
  return Reduce(
      A, axes, lang::ReduceSum, keep_dims, ir::Zero(A->type()), output_name);
}

Tensor ReduceProd(const Tensor& A,
                  const std::vector<int>& axes,
                  const bool keep_dims,
                  const std::string& output_name) {
  return Reduce(
      A, axes, lang::ReduceMul, keep_dims, lang::One(A->type()), output_name);
}

Tensor ReduceMax(const Tensor& A,
                 const std::vector<int>& axes,
                 const bool keep_dims,
                 const std::string& output_name) {
  return Reduce(A,
                axes,
                lang::ReduceMax,
                keep_dims,
                lang::min_value(A->type()),
                output_name);
}

Tensor ReduceMin(const Tensor& A,
                 const std::vector<int>& axes,
                 const bool keep_dims,
                 const std::string& output_name) {
  return Reduce(A,
                axes,
                lang::ReduceMin,
                keep_dims,
                lang::max_value(A->type()),
                output_name);
}

Tensor ReduceAll(const Tensor& A,
                 const std::vector<int>& axes,
                 const bool keep_dims,
                 const std::string& output_name) {
  return Reduce(A, axes, lang::ReduceAll, keep_dims, Expr(true), output_name);
}

Tensor ReduceAny(const Tensor& A,
                 const std::vector<int>& axes,
                 const bool keep_dims,
                 const std::string& output_name) {
  return Reduce(A, axes, lang::ReduceAny, keep_dims, Expr(false), output_name);
}

std::vector<Tensor> WarpReduce(const ir::Tensor& A,
                               const int last_reduce_dim_num,
                               const bool keep_dim,
                               const std::string& reduce_type,
                               const std::string& output_name) {
  // compute shape size without last reduce dimension.
  int shape_size_without_reduce_dim = A->shape.size() - last_reduce_dim_num;

  // compute reduce dimension size.
  Expr reduce_width(1);
  for (int idx = shape_size_without_reduce_dim; idx < A->shape.size(); ++idx) {
    reduce_width = reduce_width * A->shape[idx].as_int32();
  }

  // compute tmp output shape.
  std::vector<Expr> tmp_shape(A->shape.begin(),
                              A->shape.begin() + shape_size_without_reduce_dim);
  tmp_shape.push_back(Expr(32));
  auto tmp_out = Compute(
      tmp_shape,
      [=](const std::vector<Expr>& indices) -> Expr {
        std::vector<Expr> tmp_indices(indices.begin(),
                                      indices.begin() + indices.size() - 1);
        for (int idx = 0; idx < last_reduce_dim_num; ++idx) {
          tmp_indices.push_back(Expr(0));
        }
        PADDLE_ENFORCE_EQ(A->shape.size(),
                          tmp_indices.size(),
                          ::common::errors::InvalidArgument(
                              "indices size is not equal to Input shape!"));
        Expr offset = cinn::common::IndiceToAbsOffset(A->shape, tmp_indices);
        return lang::CallExtern(reduce_type, {A, offset, reduce_width});
      },
      UniqName(output_name + "_" + reduce_type));

  // compute output shape.
  std::vector<Expr> out_shape(A->shape.begin(),
                              A->shape.begin() + shape_size_without_reduce_dim);
  for (int idx = 0; idx < last_reduce_dim_num && keep_dim; ++idx) {
    out_shape.push_back(Expr(1));
  }
  // if reduce on all dimension, the out_shape = {1}.
  if (out_shape.size() == 0) {
    out_shape.push_back(Expr(1));
  }
  auto out = Compute(
      out_shape,
      [=](const std::vector<Expr>& indices) -> Expr {
        std::vector<Expr> tmp_indices(
            indices.begin(), indices.begin() + shape_size_without_reduce_dim);
        tmp_indices.push_back(Expr(0));
        return tmp_out(tmp_indices);
      },
      output_name);

  return {out, tmp_out};
}

std::vector<ir::Tensor> WarpReduceMax(const ir::Tensor& A,
                                      const int last_reduce_dim_num,
                                      const bool keep_dim,
                                      const std::string& output_name) {
  return WarpReduce(A,
                    last_reduce_dim_num,
                    keep_dim,
                    "cinn_warp_reduce_max" + Type2StrForReduce(A->type()),
                    output_name);
}

std::vector<ir::Tensor> WarpReduceSum(const ir::Tensor& A,
                                      const int last_reduce_dim_num,
                                      const bool keep_dim,
                                      const std::string& output_name) {
  return WarpReduce(A,
                    last_reduce_dim_num,
                    keep_dim,
                    "cinn_warp_reduce_sum" + Type2StrForReduce(A->type()),
                    output_name);
}

std::vector<ir::Tensor> WarpReduceAvg(const ir::Tensor& A,
                                      const int last_reduce_dim_num,
                                      const bool keep_dim,
                                      const std::string& output_name) {
  return WarpReduce(A,
                    last_reduce_dim_num,
                    keep_dim,
                    "cinn_warp_reduce_avg" + Type2StrForReduce(A->type()),
                    output_name);
}

std::vector<ir::Tensor> BlockReduceInternal(const ir::Tensor& A,
                                            const std::vector<int>& axes,
                                            const bool keep_dim,
                                            const std::string& reduce_type,
                                            const std::string& output_name) {
  PADDLE_ENFORCE_GE(A->shape.size(),
                    axes.back() + 1,
                    ::common::errors::InvalidArgument("Axes is over size!"));
  // compute reduce dimension size.
  Expr reduce_width(1);
  for (int idx = axes.front(); idx < A->shape.size(); ++idx) {
    reduce_width = reduce_width * A->shape[idx].as_int32();
  }

  // compute tmp output shape.
  std::vector<Expr> tmp_shape(A->shape.begin(),
                              A->shape.begin() + axes.front());
  tmp_shape.push_back(reduce_width);

  // compute the reduce dimension stride.
  std::vector<Expr> last_reduce_stride(A->shape.size() - axes.front(), Expr(1));
  for (int idx = A->shape.size(),
           index = static_cast<int>(last_reduce_stride.size()) - 2;
       index >= 0;
       --index) {
    last_reduce_stride[index] = last_reduce_stride[index + 1] * A->shape[--idx];
  }

  auto tmp_out = Compute(
      tmp_shape,
      [=](const std::vector<Expr>& indices) -> Expr {
        // compute index map from output to input.
        auto last_index = indices.back();
        std::vector<Expr> input_indices(indices.begin(),
                                        indices.begin() + indices.size() - 1);
        for (int idx = 0; idx < A->shape.size() - axes.front(); ++idx) {
          input_indices.push_back(last_index / last_reduce_stride[idx]);
          last_index = last_index % last_reduce_stride[idx];
        }

        // checkout input_indices size equals input shape
        PADDLE_ENFORCE_EQ(input_indices.size(),
                          A->shape.size(),
                          ::common::errors::InvalidArgument(
                              "indices size is not equal to Input shape!"));
        return lang::CallExtern(reduce_type, {A(input_indices)});
      },
      UniqName(output_name + "_tmp"));

  // compute output shape.
  std::vector<Expr> out_shape(A->shape.begin(),
                              A->shape.begin() + axes.front());
  int tailf = keep_dim ? (static_cast<int>(A->shape.size()) - axes.front())
                       : (static_cast<int>(A->shape.size()) - axes.back() - 1);
  for (int idx = 0; idx < tailf; ++idx) {
    out_shape.push_back(Expr(1));
  }
  // if reduce on all dimension, the out_shape = {1}.
  if (out_shape.size() == 0) {
    out_shape.push_back(Expr(1));
  }
  auto out = Compute(
      out_shape,
      [=](const std::vector<Expr>& indices) -> Expr {
        std::vector<Expr> tmp_indices(indices.begin(),
                                      indices.begin() + axes.front());
        tmp_indices.push_back(Expr(0));
        return tmp_out(tmp_indices);
      },
      output_name);
  return {out, tmp_out};
}

std::vector<ir::Tensor> BlockReduceSumInternal(const ir::Tensor& A,
                                               const std::vector<int>& axes,
                                               const bool keep_dim,
                                               const std::string& output_name) {
  return BlockReduceInternal(
      A,
      axes,
      keep_dim,
      "cinn_block_reduce_sum" + Type2StrForReduce(A->type()) + "_internal",
      output_name);
}

std::vector<ir::Tensor> BlockReduceProdInternal(
    const ir::Tensor& A,
    const std::vector<int>& axes,
    const bool keep_dim,
    const std::string& output_name) {
  return BlockReduceInternal(
      A,
      axes,
      keep_dim,
      "cinn_block_reduce_prod" + Type2StrForReduce(A->type()) + "_internal",
      output_name);
}

std::vector<ir::Tensor> BlockReduceMaxInternal(const ir::Tensor& A,
                                               const std::vector<int>& axes,
                                               const bool keep_dim,
                                               const std::string& output_name) {
  return BlockReduceInternal(
      A,
      axes,
      keep_dim,
      "cinn_block_reduce_max" + Type2StrForReduce(A->type()) + "_internal",
      output_name);
}

std::vector<ir::Tensor> BlockReduceMinInternal(const ir::Tensor& A,
                                               const std::vector<int>& axes,
                                               const bool keep_dim,
                                               const std::string& output_name) {
  return BlockReduceInternal(
      A,
      axes,
      keep_dim,
      "cinn_block_reduce_min" + Type2StrForReduce(A->type()) + "_internal",
      output_name);
}

std::vector<ir::Tensor> BlockReduceAllInternal(const ir::Tensor& A,
                                               const std::vector<int>& axes,
                                               const bool keep_dim,
                                               const std::string& output_name) {
  return BlockReduceInternal(
      A, axes, keep_dim, "cinn_block_reduce_all_internal", output_name);
}

std::vector<ir::Tensor> BlockReduceAnyInternal(const ir::Tensor& A,
                                               const std::vector<int>& axes,
                                               const bool keep_dim,
                                               const std::string& output_name) {
  return BlockReduceInternal(
      A, axes, keep_dim, "cinn_block_reduce_any_internal", output_name);
}

/**
 * @brief compute the sum of array elements over the last dimension with block
 * reduce
 *
 * @param A The input Tensor.
 * @param last_reduce_dim_num the number of last reduce dimension.
 * @param keep_dim keep the output tensor shape size as input.
 * @param output_name The name of the output Tensor.
 */
std::vector<ir::Tensor> BlockReduce(const ir::Tensor& A,
                                    const std::vector<int>& axes,
                                    const int block_size,
                                    const bool keep_dim,
                                    const std::string& reduce_type,
                                    const std::string& output_name) {
  // compute reduce dimension size.
  Expr reduce_width(1);
  for (int idx = axes.front(); idx < A->shape.size(); ++idx) {
    reduce_width = reduce_width * A->shape[idx].as_int32();
  }

  // compute tmp output tensor shape
  std::vector<Expr> tmp_shape(A->shape.begin(),
                              A->shape.begin() + axes.front());
  tmp_shape.push_back(Expr(block_size));
  auto tmp_out = Compute(
      tmp_shape,
      [=](const std::vector<Expr>& indices) -> Expr {
        std::vector<Expr> tmp_indices(indices.begin(),
                                      indices.begin() + axes.front());
        for (int idx = 0; idx < A->shape.size() - axes.front(); ++idx) {
          tmp_indices.push_back(Expr(0));
        }
        // checkout input shape size equals tmp indices size.
        PADDLE_ENFORCE_EQ(A->shape.size(),
                          tmp_indices.size(),
                          ::common::errors::InvalidArgument(
                              "indices size is not equal to Input shape!"));
        // compute offset.
        Expr offset = cinn::common::IndiceToAbsOffset(A->shape, tmp_indices);
        // call block reduce sum
        return lang::CallExtern(reduce_type, {A, offset, reduce_width});
      },
      UniqName(output_name + "_tmp"));

  // compute output tensor shape.
  std::vector<Expr> out_shape(A->shape.begin(),
                              A->shape.begin() + axes.front());
  int tailf = keep_dim ? (static_cast<int>(A->shape.size()) - axes.front())
                       : (static_cast<int>(A->shape.size()) - axes.back() - 1);
  for (int idx = 0; idx < tailf; ++idx) {
    out_shape.push_back(Expr(1));
  }
  // if reduce on all dimension, the out_shape = {1}.
  if (out_shape.size() == 0) {
    out_shape.push_back(Expr(1));
  }
  auto out = Compute(
      out_shape,
      [=](const std::vector<Expr>& indices) -> Expr {
        // compute input index
        std::vector<Expr> tmp_indices(indices.begin(),
                                      indices.begin() + axes.front());
        tmp_indices.push_back(Expr(0));
        return tmp_out(tmp_indices);
      },
      output_name);

  return {out, tmp_out};
}

std::vector<ir::Tensor> BlockReduceSum(const ir::Tensor& A,
                                       const std::vector<int>& axes,
                                       const int block_size,
                                       const bool keep_dim,
                                       const std::string& output_name) {
  return BlockReduce(A,
                     axes,
                     block_size,
                     keep_dim,
                     "cinn_block_reduce_sum" + Type2StrForReduce(A->type()),
                     output_name);
}

std::vector<ir::Tensor> BlockReduceProd(const ir::Tensor& A,
                                        const std::vector<int>& axes,
                                        const int block_size,
                                        const bool keep_dim,
                                        const std::string& output_name) {
  return BlockReduce(A,
                     axes,
                     block_size,
                     keep_dim,
                     "cinn_block_reduce_prod" + Type2StrForReduce(A->type()),
                     output_name);
}

std::vector<ir::Tensor> BlockReduceMax(const ir::Tensor& A,
                                       const std::vector<int>& axes,
                                       const int block_size,
                                       const bool keep_dim,
                                       const std::string& output_name) {
  return BlockReduce(A,
                     axes,
                     block_size,
                     keep_dim,
                     "cinn_block_reduce_max" + Type2StrForReduce(A->type()),
                     output_name);
}

std::vector<ir::Tensor> BlockReduceMin(const ir::Tensor& A,
                                       const std::vector<int>& axes,
                                       const int block_size,
                                       const bool keep_dim,
                                       const std::string& output_name) {
  return BlockReduce(A,
                     axes,
                     block_size,
                     keep_dim,
                     "cinn_block_reduce_min" + Type2StrForReduce(A->type()),
                     output_name);
}

std::vector<ir::Tensor> BlockReduceAll(const ir::Tensor& A,
                                       const std::vector<int>& axes,
                                       const int block_size,
                                       const bool keep_dim,
                                       const std::string& output_name) {
  return BlockReduce(
      A, axes, block_size, keep_dim, "cinn_block_reduce_all", output_name);
}

std::vector<ir::Tensor> BlockReduceAny(const ir::Tensor& A,
                                       const std::vector<int>& axes,
                                       const int block_size,
                                       const bool keep_dim,
                                       const std::string& output_name) {
  return BlockReduce(
      A, axes, block_size, keep_dim, "cinn_block_reduce_any", output_name);
}

int GetPostParallelSize(const ir::Tensor& A, const std::vector<int>& axes) {
  int parallel_size = 1;
  for (int idx = axes.back() + 1; idx < A->shape.size(); ++idx) {
    parallel_size *= A->shape[idx].as_int32();
  }
  return parallel_size;
}

int GetParallelSize(const ir::Tensor& A, const std::vector<int>& axes) {
  int parallel_size = 1;
  for (int idx = 0; idx < A->shape.size(); ++idx) {
    if (std::find(axes.begin(), axes.end(), idx) != axes.end()) {
      continue;
    }
    parallel_size *= A->shape[idx].as_int32();
  }
  return parallel_size;
}

using ReduceFunc = std::function<ir::Tensor(const ir::Tensor&,
                                            const std::vector<int>&,
                                            const bool,
                                            const std::string&)>;

std::vector<ir::Tensor> ReduceInternal(const ir::Tensor& A,
                                       const std::vector<int>& axes,
                                       const bool keep_dim,
                                       const std::string& output_name,
                                       ReduceFunc reduce_func,
                                       ir::Expr initial,
                                       std::string reduce_type) {
  int tail = 0;
  bool inbound = true;
  std::vector<int> inshape;
  std::transform(A->shape.begin(),
                 A->shape.end(),
                 std::back_inserter(inshape),
                 [](ir::Expr expr) { return expr.as_int32(); });
  auto reduce_shape = GetFirstStepReduceShape(inshape, axes, inbound, tail);
  PADDLE_ENFORCE_GT(
      reduce_shape.size(),
      0,
      ::common::errors::InvalidArgument("Reduce shape size is 0!"));

  VLOG(4) << "Reduce " << output_name << " on " << reduce_type
          << " with input shape=[" << cinn::utils::Join(inshape, ", ")
          << "], and first step reduce_shape=["
          << cinn::utils::Join(reduce_shape, ", ") << "] at axes=["
          << cinn::utils::Join(axes, ", ") << "]";

  // reshape input
  auto do_reshape_inbound = [&]() {
    int axis = axes.back();
    std::vector<ir::Expr> reshape_output_shape;
    // last successive axis in reduce axes.
    int axis_index = axes.size() - 1;
    for (; axis_index >= 1; --axis_index) {
      if (axes[axis_index] - 1 != axes[axis_index - 1]) {
        break;
      }
    }
    // compute reduce stride.
    std::vector<ir::Expr> strides(1, ir::Expr(1));
    for (int idx = axes.back(); idx > axes[axis_index]; --idx) {
      strides.insert(strides.begin(), strides.front() * ir::Expr(inshape[idx]));
    }
    PADDLE_ENFORCE_EQ(strides.size(),
                      axes.size() - axis_index,
                      ::common::errors::InvalidArgument(
                          "Strides size is not equal to axes size!"));
    std::transform(reduce_shape.begin(),
                   reduce_shape.end(),
                   std::back_inserter(reshape_output_shape),
                   [](int val) { return ir::Expr(val); });
    return Compute(
        reshape_output_shape,
        [=](const std::vector<Expr>& indices) -> Expr {
          // index is last axis in axes and index is last axis >= tail.
          auto selected = ir::And::Make(
              ir::EQ::Make(indices[axis], ir::Expr(reduce_shape[axis] - 1)),
              ir::GE::Make(indices[axis + 1], ir::Expr(tail)));
          auto index =
              indices[axis] * ir::Expr(reshape_output_shape[axis + 1]) +
              indices[axis + 1];

          // first part index
          std::vector<ir::Expr> tmp_indices(indices.begin(),
                                            indices.begin() + axes[axis_index]);
          // second part index
          for (int idx = 0; idx < strides.size(); ++idx) {
            tmp_indices.push_back(index / strides[idx]);
            index = index % strides[idx];
          }
          // third part index
          for (int idx = axis + 2; idx < indices.size(); ++idx) {
            tmp_indices.push_back(indices[idx]);
          }

          PADDLE_ENFORCE_EQ(tmp_indices.size(),
                            A->shape.size(),
                            ::common::errors::InvalidArgument(
                                "indices size is not equal to Input shape!"));
          return ir::Select::Make(selected, A(tmp_indices), initial);
        },
        UniqName(output_name + "_reshape"));
  };
  auto reshape = inbound
                     ? pe::Reshape(A, reduce_shape, output_name + "_reshape")
                     : do_reshape_inbound();
  // do first step reduce
  auto internal =
      reduce_func(reshape, axes, keep_dim, output_name + "_internal");
  // do second step reduce
  std::vector<int> s_axes = {};
  if (keep_dim) {
    s_axes = {axes.back() + 1};
  } else {
    s_axes = {axes.back() + 1 - static_cast<int>(axes.size())};
  }
  auto reduce_out = reduce_func(internal, s_axes, false, output_name);

  return {reduce_out, internal, reshape};
}

#define BLOCK_SHUFFLE_REDUCE(name, reduce_type, initial)                       \
  std::vector<ir::Tensor> BlockShuffleReduce##name(                            \
      const ir::Tensor& A,                                                     \
      const std::vector<int>& axes,                                            \
      const bool keep_dim,                                                     \
      const std::string& output_name) {                                        \
    if (cinn::common::GetMaxThreads() / GetParallelSize(A, axes) <= 1) {       \
      return {Reduce##name(A, axes, keep_dim, output_name)};                   \
    } else {                                                                   \
      auto rs = ReduceInternal(                                                \
          A, axes, keep_dim, output_name, Reduce##name, initial, reduce_type); \
      if (rs.size() == 0) {                                                    \
        return {Reduce##name(A, axes, keep_dim, output_name)};                 \
      } else {                                                                 \
        return rs;                                                             \
      }                                                                        \
    }                                                                          \
  }

BLOCK_SHUFFLE_REDUCE(Sum,
                     "block_shuffle_sum" + Type2StrForReduce(A->type()),
                     ir::Zero(A->type()));
BLOCK_SHUFFLE_REDUCE(Prod,
                     "block_shuffle_prod" + Type2StrForReduce(A->type()),
                     lang::One(A->type()));
BLOCK_SHUFFLE_REDUCE(Max,
                     "block_shuffle_max" + Type2StrForReduce(A->type()),
                     lang::min_value(A->type()));
BLOCK_SHUFFLE_REDUCE(Min,
                     "block_shuffle_min" + Type2StrForReduce(A->type()),
                     lang::max_value(A->type()));
BLOCK_SHUFFLE_REDUCE(All,
                     "block_shuffle_all" + Type2StrForReduce(A->type()),
                     Expr(true));
BLOCK_SHUFFLE_REDUCE(Any,
                     "block_shuffle_any" + Type2StrForReduce(A->type()),
                     Expr(false));

bool WithoutLastDimInReduce(const std::vector<ir::Expr>& inshape,
                            const std::vector<int>& axes) {
  // if last axis is in reduce.
  if (std::find(axes.begin(), axes.end(), inshape.size() - 1) != axes.end() ||
      std::find(axes.begin(), axes.end(), -1) != axes.end()) {
    return false;
  }

  int sum_last_axes = 1;
  for (int idx = axes.back() + 1; idx < inshape.size(); ++idx) {
    sum_last_axes *= inshape[idx].as_int32();
  }

  if (sum_last_axes > 1) {
    return true;
  } else {
    return false;
  }
}

using BlockReduceFunc =
    std::function<std::vector<ir::Tensor>(const ir::Tensor&,
                                          const std::vector<int>&,
                                          const bool,
                                          const std::string&)>;

std::vector<ir::Tensor> TwoStepBlockReduceInternal(
    const ir::Tensor& A,
    const std::vector<int>& axes,
    const bool keep_dim,
    const std::string& output_name,
    ReduceFunc reduce_func,
    BlockReduceFunc block_reduce_func,
    ir::Expr initial) {
  PADDLE_ENFORCE_EQ(
      !WithoutLastDimInReduce(A->shape, axes),
      true,
      ::common::errors::InvalidArgument("Can't find last axis in reduce!"));
  // If the number of current device SM is smaller than the number of SM
  // required by Warp Reduce, the performance of Warp Reduce is better.
  // Otherwise, use Block Reduce.
  auto max_num_threads = cinn::common::DefaultDeviceTarget().max_num_threads();
  int need_reduce_last_count = 1;
  for (int i = 0; i < A->shape.size(); i++) {
    if (find(axes.begin(), axes.end(), i) == axes.end()) {
      need_reduce_last_count *= A->shape[i].as_int32();
    }
  }
  int warp_reduce_need_sm_count =
      ceil((need_reduce_last_count * 32) /
           static_cast<float>(
               cinn::common::DefaultDeviceTarget().get_max_threads_per_sm()));
  // Set Num_max_threads to 32 is Warp Reduce
  if (cinn::common::DefaultDeviceTarget().get_multi_processor_count() <
      warp_reduce_need_sm_count) {
    max_num_threads = 32;
  }

  int lane = A->shape[axes.back()].as_int32();
  int index = static_cast<int>(axes.size()) - 2;
  for (; index >= 0; --index) {
    if (lane >= max_num_threads / 2) {
      break;
    }
    if (axes[index] != axes[index + 1] - 1) {
      break;
    }
    lane *= A->shape[axes[index]].as_int32();
  }
  std::vector<int> first_axes(axes.begin(), axes.begin() + index + 1);
  std::vector<int> second_axes(axes.begin() + index + 1, axes.end());

  bool keep_dim_first = keep_dim;
  bool keep_dim_second = keep_dim;
  auto reduce_reshape_func = [&first_axes,
                              &keep_dim_first,
                              &second_axes,
                              &keep_dim_second,
                              A,
                              axes,
                              keep_dim,
                              output_name,
                              lane,
                              index,
                              max_num_threads,
                              &initial]() {
    bool check_bound = true;
    std::vector<Expr> out_shape(A->shape.begin(),
                                A->shape.begin() + second_axes.front());
    if (second_axes.size() == 1) {
      int times = 1;
      int tail = max_num_threads;
      for (; tail >= max_num_threads / 2; --tail) {
        if (lane % tail == 0) {
          check_bound = false;
          break;
        }
      }
      if (!check_bound) {
        times = lane / tail;
        out_shape.emplace_back(times);
        out_shape.emplace_back(tail);
      } else {
        times = (lane + max_num_threads - 1) / max_num_threads;
        out_shape.emplace_back(times);
        out_shape.emplace_back(max_num_threads);
      }
    } else {
      int times = 1;
      int head = A->shape[second_axes.front()].as_int32();
      int tail = lane / head;
      // from (1024, 512) check one size as tail.
      for (int idx = (max_num_threads / tail);
           idx > (max_num_threads / 2 / tail);
           --idx) {
        if (head % idx == 0) {
          check_bound = false;
          times = idx;
          tail *= idx;
          break;
        }
      }
      if (!check_bound) {
        out_shape.emplace_back(head / times);
        out_shape.emplace_back(tail);
      } else {
        times = max_num_threads / tail;
        out_shape.emplace_back((head + times - 1) / times);
        out_shape.emplace_back(tail * times);
      }
    }
    first_axes.push_back(out_shape.size() - 2);

    int tail_count = 0;
    if (keep_dim) {
      second_axes = {static_cast<int>(out_shape.size()) - 1};
      if (out_shape.size() > A->shape.size()) {
        keep_dim_second = false;
      } else {
        keep_dim_second = true;
        tail_count = A->shape.size() - out_shape.size();
        for (int idx = 0; idx < tail_count; ++idx) {
          out_shape.push_back(Expr(1));
        }
      }
    } else {
      second_axes = {static_cast<int>(out_shape.size()) -
                     static_cast<int>(first_axes.size()) - 1};
    }

    int size_without_tail = out_shape.size() - tail_count;
    std::vector<int> tail_strides(A->shape.size() - (size_without_tail - 2), 1);
    for (int idx = static_cast<int>(tail_strides.size()) - 2,
             index = static_cast<int>(A->shape.size()) - 1;
         idx >= 0;
         --idx, --index) {
      tail_strides[idx] = tail_strides[idx + 1] * A->shape[index].as_int32();
    }
    auto out = Compute(
        out_shape,
        [=](const std::vector<Expr>& indices) -> Expr {
          Expr index =
              indices[size_without_tail - 1] +
              indices[size_without_tail - 2] * out_shape[size_without_tail - 1];
          std::vector<Expr> tmp_indices(
              indices.begin(), indices.begin() + size_without_tail - 2);
          // last and the second of last.
          auto selected = ir::LT::Make(index, Expr(lane));
          for (auto tail_stride : tail_strides) {
            tmp_indices.push_back(index / Expr(tail_stride));
            index = index % Expr(tail_stride);
          }

          PADDLE_ENFORCE_EQ(tmp_indices.size(),
                            A->shape.size(),
                            ::common::errors::InvalidArgument(
                                "indices size is not equal to Input shape!"));
          if (check_bound) {
            return ir::Select::Make(selected, A(tmp_indices), initial);
          } else {
            return A(tmp_indices);
          }
        },
        UniqName(output_name + "_reshape"));
    return out;
  };
  std::vector<ir::Tensor> results;
  if (lane > max_num_threads) {
    VLOG(3) << "Do Reduce Reshape!";
    results.push_back(reduce_reshape_func());
  } else {
    if (!keep_dim) {
      for (auto& axis : second_axes) {
        axis -= first_axes.size();
      }
    }
  }
  if (first_axes.size()) {
    VLOG(3) << "Do Reduce Internal!";
    results.push_back(reduce_func(results.size() ? results.back() : A,
                                  first_axes,
                                  keep_dim_first,
                                  output_name + "_internal"));
    results.back()->WithBuffer("local");
  }
  if (second_axes.size()) {
    VLOG(3) << "Do Block Reduce!";
    auto res = block_reduce_func(results.size() ? results.back() : A,
                                 second_axes,
                                 keep_dim_second,
                                 output_name);
    results.push_back(res[1]);
    results.push_back(res[0]);
  }
  std::reverse(results.begin(), results.end());
  return results;
}

std::vector<ir::Tensor> TwoStepBlockReduceSum(const ir::Tensor& A,
                                              const std::vector<int>& axes,
                                              const bool keep_dim,
                                              const std::string& output_name) {
  return TwoStepBlockReduceInternal(A,
                                    axes,
                                    keep_dim,
                                    output_name,
                                    ReduceSum,
                                    BlockReduceSumInternal,
                                    ir::Zero(A->type()));
}

std::vector<ir::Tensor> TwoStepBlockReduceProd(const ir::Tensor& A,
                                               const std::vector<int>& axes,
                                               const bool keep_dim,
                                               const std::string& output_name) {
  return TwoStepBlockReduceInternal(A,
                                    axes,
                                    keep_dim,
                                    output_name,
                                    ReduceProd,
                                    BlockReduceProdInternal,
                                    lang::One(A->type()));
}

std::vector<ir::Tensor> TwoStepBlockReduceMax(const ir::Tensor& A,
                                              const std::vector<int>& axes,
                                              const bool keep_dim,
                                              const std::string& output_name) {
  return TwoStepBlockReduceInternal(A,
                                    axes,
                                    keep_dim,
                                    output_name,
                                    ReduceMax,
                                    BlockReduceMaxInternal,
                                    lang::min_value(A->type()));
}

std::vector<ir::Tensor> TwoStepBlockReduceMin(const ir::Tensor& A,
                                              const std::vector<int>& axes,
                                              const bool keep_dim,
                                              const std::string& output_name) {
  return TwoStepBlockReduceInternal(A,
                                    axes,
                                    keep_dim,
                                    output_name,
                                    ReduceMin,
                                    BlockReduceMinInternal,
                                    lang::max_value(A->type()));
}

std::vector<ir::Tensor> TwoStepBlockReduceAll(const ir::Tensor& A,
                                              const std::vector<int>& axes,
                                              const bool keep_dim,
                                              const std::string& output_name) {
  return TwoStepBlockReduceInternal(A,
                                    axes,
                                    keep_dim,
                                    output_name,
                                    ReduceAll,
                                    BlockReduceAllInternal,
                                    Expr(true));
}

std::vector<ir::Tensor> TwoStepBlockReduceAny(const ir::Tensor& A,
                                              const std::vector<int>& axes,
                                              const bool keep_dim,
                                              const std::string& output_name) {
  return TwoStepBlockReduceInternal(A,
                                    axes,
                                    keep_dim,
                                    output_name,
                                    ReduceAny,
                                    BlockReduceAnyInternal,
                                    Expr(false));
}

std::string ReduceOpAndTypeStr(const ir::Expr& op,
                               const cinn::common::Type type) {
  if (op.As<ir::Add>()) {
    if (type.is_bool()) {
      return "any";
    }
    return "sum" + Type2StrForReduce(type);
  } else if (op.As<ir::Mul>()) {
    if (type.is_bool()) {
      return "all";
    }
    return "prod" + Type2StrForReduce(type);
  } else if (op.As<ir::Max>()) {
    return "max" + Type2StrForReduce(type);
  } else if (op.As<ir::Min>()) {
    return "min" + Type2StrForReduce(type);
  } else if (op.As<ir::And>()) {
    return "all";
  } else if (op.As<ir::Or>()) {
    return "any";
  }
  PADDLE_THROW(::common::errors::InvalidArgument(
      "No matching reduce template for op: %s, type: %s", op, type));
}

std::string CrossThreadReduceExternalFuncName(const ir::Expr& op,
                                              const ir::Expr& tensor) {
  std::string op_and_type = ReduceOpAndTypeStr(op, tensor.as_tensor()->type());
  return "cinn_block_reduce_" + op_and_type;
}

std::string DiscreteReduceExternalFuncName(const ir::Expr& op,
                                           const ir::Expr& tensor) {
  std::string op_and_type = ReduceOpAndTypeStr(op, tensor.as_tensor()->type());
  return "cinn_discrete_reduce_" + op_and_type;
}

std::string GridReduceExternalFuncName(const ir::Expr& op,
                                       const cinn::common::Type type) {
  std::string op_and_type = ReduceOpAndTypeStr(op, type);
  return "cinn_grid_reduce_" + op_and_type;
}

}  // namespace pe
}  // namespace hlir
}  // namespace cinn
