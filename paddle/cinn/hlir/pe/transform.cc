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

#include "paddle/cinn/hlir/pe/transform.h"

#include <algorithm>
#include <utility>

#include "paddle/cinn/common/cas.h"
#include "paddle/cinn/common/context.h"
#include "paddle/cinn/common/ir_util.h"
#include "paddle/cinn/hlir/op/op_util.h"
#include "paddle/cinn/hlir/pe/elementwise.h"
#include "paddle/cinn/hlir/pe/schedule.h"
#include "paddle/cinn/ir/tensor.h"
#include "paddle/cinn/lang/compute.h"
#include "paddle/cinn/utils/string.h"
#include "paddle/common/enforce.h"
#include "paddle/common/errors.h"
namespace cinn {
namespace hlir {
namespace pe {

using cinn::lang::Compute;
using ir::Tensor;

namespace utils {
std::vector<std::vector<int>> GetMatmulNewShapes(
    const std::vector<std::vector<int>>& inputs_shape,
    bool trans_x,
    bool trans_y) {
  PADDLE_ENFORCE_EQ(inputs_shape.size(),
                    2UL,
                    ::common::errors::InvalidArgument(
                        "The matmul should only have two inputs."));
  const auto &x_shape = inputs_shape[0], &y_shape = inputs_shape[1];
  PADDLE_ENFORCE_EQ(!x_shape.empty(),
                    true,
                    ::common::errors::InvalidArgument(
                        "The shape of matmul input 'x' should not empty."));
  PADDLE_ENFORCE_EQ(!y_shape.empty(),
                    true,
                    ::common::errors::InvalidArgument(
                        "The shape of matmul input 'y' should not empty."));

  auto matmul_info = [&]() {
    std::stringstream ss;
    ss << std::boolalpha << "matmul(X:"
       << "[" << cinn::utils::Join(x_shape, ", ") << "], Y:"
       << "[" << cinn::utils::Join(y_shape, ", ") << "]"
       << ", trans_x=" << trans_x << ", trans_y=" << trans_y << ")";
    return ss.str();
  };
  VLOG(4) << "Try infer " << matmul_info() << "'s correct shape";

  std::vector<std::vector<int>> new_shape(3);
  auto& new_x_shape = new_shape[0];
  auto& new_y_shape = new_shape[1];
  auto& out_shape = new_shape[2];

  int x_dim = x_shape.size(), y_dim = y_shape.size();
  int max_dim = std::max(x_shape.size(), y_shape.size());
  int out_dim = max_dim >= 3 ? 3 : (max_dim <= 2 ? 2 : max_dim);

  auto get_input_shape = [out_dim](const std::vector<int>& old_shape) {
    PADDLE_ENFORCE_GE(old_shape.size(),
                      2UL,
                      ::common::errors::InvalidArgument(
                          "The shape of matmul input should greater equal 2"));
    std::vector<int> res;
    res.resize(out_dim, 1);
    // [a, b, m, d] -> [a*b, m, d]
    for (int i = 0; i < old_shape.size() - 2; ++i) {
      res[0] *= old_shape[i];
    }
    res[out_dim - 2] = old_shape[old_shape.size() - 2];
    res[out_dim - 1] = old_shape[old_shape.size() - 1];
    return res;
  };

  if (max_dim == 1) {
    // vector * vector
    PADDLE_ENFORCE_EQ(x_shape[0] == y_shape[0],
                      true,
                      ::common::errors::InvalidArgument(
                          "The matmul input X's numbers must be equal to Y's "
                          "numbers,when X/Y's dims =1. But here %s.",
                          matmul_info()));

    new_x_shape = trans_x ? std::vector<int>{x_shape[0], 1}
                          : std::vector<int>{1, x_shape[0]};
    new_y_shape = trans_y ? std::vector<int>{1, y_shape[0]}
                          : std::vector<int>{y_shape[0], 1};
    // [m] * [m] -> [], which aligns with Paddle's matmul
    out_shape = {};
  } else if (x_dim == 1) {
    // vector * matrix
    int y_K = trans_y ? y_shape[max_dim - 1] : y_shape[max_dim - 2];
    PADDLE_ENFORCE_EQ(y_K,
                      x_shape[0],
                      ::common::errors::InvalidArgument(
                          "The K dimension of Y should equal to X.shape[0]"));

    // set x shape for broadcast
    new_x_shape.resize(out_dim, 1);
    if (trans_x) {
      // [m] * [a, b, m, d] -> [1, m, 1] * [a*b, m, d]
      new_x_shape[out_dim - 2] = x_shape[0];
    } else {
      // [m] * [a, b, m, d] -> [1, 1, m] * [a*b, m, d]
      new_x_shape[out_dim - 1] = x_shape[0];
    }

    new_y_shape = get_input_shape(y_shape);

    // set output shape after broadcast
    out_shape = y_shape;
    if (trans_y) {
      // [m] * [a, b, d, m] -> [a, b, d]
      out_shape.erase(out_shape.end() - 1);
    } else {
      // [m] * [a, b, m, d] -> [a, b, d]
      out_shape.erase(out_shape.end() - 2);
    }

  } else if (y_dim == 1) {
    // matrix * vector
    int x_K = trans_x ? x_shape[max_dim - 2] : x_shape[max_dim - 1];
    PADDLE_ENFORCE_EQ(x_K,
                      y_shape[0],
                      ::common::errors::InvalidArgument(
                          "The K dimension of X should equal to Y.shape[0]"));

    // set y shape for broadcast
    // [a, b, c, m] * [m] -> [a*b, c, m] * [1, m, 1]
    new_x_shape = get_input_shape(x_shape);

    new_y_shape.resize(out_dim, 1);
    if (trans_y) {
      // [a, b, c, m] * [m] -> [a*b, c, m] * [1, 1, m]
      new_y_shape[out_dim - 1] = y_shape[0];
    } else {
      // [a, b, c, m] * [m] -> [a*b, c, m] * [1, m, 1]
      new_y_shape[out_dim - 2] = y_shape[0];
    }

    out_shape = x_shape;
    if (trans_x) {
      // [a, b, m, c] * [m] -> [a, b, c]
      out_shape.erase(out_shape.end() - 2);
    } else {
      // [a, b, c, m] * [m] -> [a, b, c]
      out_shape.erase(out_shape.end() - 1);
    }
  } else {
    // matrix * matrix
    int x_K = trans_x ? x_shape[x_dim - 2] : x_shape[x_dim - 1];
    int y_K = trans_y ? y_shape[y_dim - 1] : y_shape[y_dim - 2];
    PADDLE_ENFORCE_EQ(x_K,
                      y_K,
                      ::common::errors::InvalidArgument(
                          "The K dimension of matmul not equal."));

    // [c, m] * [a, b, m, d] -> [1, c, m] * [a*b, m, d]
    new_x_shape = get_input_shape(x_shape);
    // [a, b, c, m] * [m, d] -> [a*b, c, m] * [1, m, d]
    new_y_shape = get_input_shape(y_shape);

    // get output shape
    // [a, b, c, m] * [a, b, m, d] -> [a, b, c, d]
    int M = trans_x ? x_shape[x_dim - 1] : x_shape[x_dim - 2];
    int N = trans_y ? y_shape[y_dim - 2] : y_shape[y_dim - 1];

    out_shape.resize(max_dim, 1);
    out_shape[max_dim - 2] = M;
    out_shape[max_dim - 1] = N;

    // get the batch dimension after broadcast
    int x_pos = x_dim - 3, y_pos = y_dim - 3, out_pos = max_dim - 3;
    while (x_pos >= 0 && y_pos >= 0) {
      PADDLE_ENFORCE_EQ(
          x_shape[x_pos] == y_shape[y_pos] || x_shape[x_pos] == 1 ||
              y_shape[y_pos] == 1,
          true,
          ::common::errors::InvalidArgument("Input X and Y's batch dimension "
                                            "should be same or 1. But here %s.",
                                            matmul_info()));

      out_shape[out_pos] =
          (x_shape[x_pos] == 1) ? y_shape[y_pos] : x_shape[x_pos];

      out_pos--;
      x_pos--;
      y_pos--;
    }

    while (x_pos >= 0) {
      out_shape[out_pos--] = x_shape[x_pos--];
    }
    while (y_pos >= 0) {
      out_shape[out_pos--] = x_shape[y_pos--];
    }
  }

  return new_shape;
}

std::vector<std::vector<int>> GetMulNewShapes(
    const std::vector<std::vector<int>>& inputs_shape,
    int x_num_col_dims,
    int y_num_col_dims,
    bool is_infer) {
  PADDLE_ENFORCE_EQ(inputs_shape.size(),
                    2UL,
                    ::common::errors::InvalidArgument(
                        "The mul should only have two inputs."));
  const auto &x_shape = inputs_shape[0], &y_shape = inputs_shape[1];
  PADDLE_ENFORCE_EQ(!x_shape.empty(),
                    true,
                    ::common::errors::InvalidArgument(
                        "The shape of matmul input 'x' should not empty."));
  PADDLE_ENFORCE_EQ(!y_shape.empty(),
                    true,
                    ::common::errors::InvalidArgument(
                        "The shape of matmul input 'y' should not empty."));

  auto mul_info = [&]() {
    std::stringstream ss;
    ss << std::boolalpha << "mul(X:"
       << "[" << cinn::utils::Join(x_shape, ", ") << "], Y:"
       << "[" << cinn::utils::Join(y_shape, ", ") << "]"
       << ", x_num_col_dims=" << x_num_col_dims
       << ", y_num_col_dims=" << y_num_col_dims << ")";
    return ss.str();
  };
  VLOG(4) << "Try infer " << mul_info() << "'s correct shape";

  std::vector<std::vector<int>> new_shape(3);
  auto& new_x_shape = new_shape[0];
  auto& new_y_shape = new_shape[1];
  auto& out_shape = new_shape[2];

  auto flatten_shape = [&](const std::vector<int>& shape, int num_col_dims) {
    if (shape.size() <= 2) {
      return shape;
    }

    if (num_col_dims < 0) {
      num_col_dims += shape.size();
    }

    PADDLE_ENFORCE_GT(num_col_dims,
                      0,
                      ::common::errors::InvalidArgument(
                          "The [num_col_dims] should not be 0 in "
                          "mul op. Please check."));
    PADDLE_ENFORCE_LT(
        num_col_dims,
        shape.size(),
        ::common::errors::InvalidArgument("The [num_col_dims] > rank(input) in "
                                          "mul op. Please check."));

    std::vector<int> res(2, 1);
    for (int i = 0; i < num_col_dims; ++i) {
      res[0] *= shape[i];
    }
    for (int i = num_col_dims; i < shape.size(); ++i) {
      res[1] *= shape[i];
    }
    return res;
  };

  new_x_shape = flatten_shape(x_shape, x_num_col_dims);
  new_y_shape = flatten_shape(y_shape, y_num_col_dims);

  for (int i = 0; i < x_num_col_dims; ++i) {
    out_shape.emplace_back(x_shape[i]);
  }
  if (is_infer) {
    for (int i = 0; i < y_num_col_dims; ++i) {
      out_shape.emplace_back(y_shape[i]);
    }
  } else {
    for (int i = y_num_col_dims; i < y_shape.size(); ++i) {
      out_shape.emplace_back(y_shape[i]);
    }
  }

  return new_shape;
}
}  // namespace utils

std::vector<Tensor> Matmul(const Tensor& A,
                           const Tensor& B,
                           bool trans_a,
                           bool trans_b,
                           float alpha,
                           const std::string& name) {
  std::vector<Expr> shape_A = A->shape;
  std::vector<Expr> shape_B = B->shape;
  int a_dim = shape_A.size();
  int b_dim = shape_B.size();
  PADDLE_ENFORCE_EQ(
      a_dim == 3U || a_dim == 2U,
      true,
      ::common::errors::InvalidArgument(
          "Tensor_A's dim should be 2 or 3 while current dim is %d.", a_dim));
  PADDLE_ENFORCE_EQ(
      b_dim == 3U || b_dim == 2U,
      true,
      ::common::errors::InvalidArgument(
          "Tensor_B's dim should be 2 or 3 while current dim is %d.", b_dim));
  PADDLE_ENFORCE_EQ(a_dim,
                    b_dim,
                    ::common::errors::InvalidArgument(
                        "tensor_A's dim should be same with tensor_B"));

  Expr x_width = trans_a ? shape_A[a_dim - 2] : shape_A.back();
  Expr y_height = trans_b ? shape_B.back() : shape_B[b_dim - 2];
  Expr M = trans_a ? shape_A.back() : shape_A[a_dim - 2];
  Expr N = trans_b ? shape_B[b_dim - 2] : shape_B.back();
  PADDLE_ENFORCE_EQ(
      is_zero(x_width - y_height),
      true,
      ::common::errors::InvalidArgument(
          "Matrix multiplication requires x_width to be same with y_height."));
  std::vector<Expr> output_shape;
  std::vector<ir::Tensor> out;
  if (a_dim == 3) {
    int max_batch = std::max(shape_A[0].as_int32(), shape_B[0].as_int32());
    output_shape = {Expr(max_batch), M, N};
  } else {
    output_shape = {M, N};
  }
  Var reduce_k(x_width, UniqName("reduce_k"));
  auto temp = Compute(
      output_shape,
      [=](const std::vector<Expr>& indice) {
        int out_dim = indice.size();
        std::vector<Expr> A_indice;
        std::vector<Expr> B_indice;
        PADDLE_ENFORCE_EQ(
            out_dim == 3U || out_dim == 2U,
            true,
            ::common::errors::InvalidArgument(
                "Indice size should be 2 or 3 while current dim is %d.",
                out_dim));
        if (out_dim == 3U) {
          // batch
          A_indice.push_back(indice[0]);
          B_indice.push_back(indice[0]);
        }
        A_indice.push_back(indice[out_dim - 2]);
        A_indice.push_back(reduce_k);
        B_indice.push_back(reduce_k);
        B_indice.push_back(indice[out_dim - 1]);
        if (trans_a) {
          std::swap(A_indice[out_dim - 2], A_indice[out_dim - 1]);
        }
        if (trans_b) {
          std::swap(B_indice[out_dim - 2], B_indice[out_dim - 1]);
        }
        return lang::ReduceSum(A(A_indice) * B(B_indice), {reduce_k});
      },
      UniqName("temp_matmul_out"));
  if (alpha != 1) {
    auto res = Compute(
        output_shape,
        [=](const std::vector<Expr>& indice) {
          return temp(indice) * ir::Cast::Make(temp->type(), Expr(alpha));
        },
        name);
    return {res, temp};
  } else {
    return {temp};
  }
}

std::vector<ir::Tensor> Split(
    const ir::Tensor& A,
    int axis,
    const std::vector<std::vector<int>>& output_shapes,
    const std::vector<std::string>& names) {
  if (axis < 0) axis += A->shape.size();
  auto output_size = output_shapes.size();

  // compute select index list
  // if   index = [2, 3, 4, 5]
  // then start = [0, 2, 5, 9]
  std::vector<int> start(output_size, 0);
  for (int i = 1; i < output_size; ++i) {
    start[i] = start[i - 1] + output_shapes[i - 1][axis];
  }

  std::vector<std::vector<Expr>> out_shape(output_size, std::vector<Expr>{});
  for (int i = 0; i < output_size; ++i) {
    for (int val : output_shapes[i]) {
      out_shape[i].emplace_back(Expr(val));
    }
  }

  std::vector<ir::Tensor> res(output_size);
  PADDLE_ENFORCE_EQ(output_size,
                    names.size(),
                    ::common::errors::InvalidArgument(
                        "The output size should be equal to the names size."));
  for (int i = 0; i < output_size; ++i) {
    res[i] = Compute(
        out_shape[i],
        [=](const std::vector<Expr>& indice) {
          auto temp = indice;
          temp[axis] = optim::ArithSimplify(temp[axis] + Expr(start[i]));
          return A(temp);
        },
        names[i]);
  }
  return res;
}

ir::Tensor Concat(const ir::Tensor& A,
                  const ir::Tensor& B,
                  int axis,
                  const std::string& name) {
  if (axis < 0) axis += A->shape.size();
  PADDLE_ENFORCE_EQ(
      A->shape.size(),
      B->shape.size(),
      ::common::errors::InvalidArgument(
          "Dimensions of inputs A and B in Concat should be equal! Please "
          "check."));
  std::vector<Expr> output_shape = A->shape;
  Expr pivot = A->shape[axis];
  output_shape[axis] =
      optim::ArithSimplify(output_shape[axis] + B->shape[axis]);
  auto res = Compute(
      output_shape,
      [=](const std::vector<Expr>& indice) {
        auto indice_B = indice;
        indice_B[axis] = indice_B[axis] - pivot;
        return ir::Select::Make(indice[axis] < pivot, A(indice), B(indice_B));
      },
      name);
  return res;
}

ir::Tensor Concat(const std::vector<ir::Tensor>& input_tensors,
                  int axis,
                  const std::string& name) {
  // input size 1 is valid for Concat
  int input_size = input_tensors.size();
  PADDLE_ENFORCE_GE(input_size,
                    1U,
                    ::common::errors::InvalidArgument(
                        "Concat should have at least 1 input tensors"));
  std::vector<Expr> output_shape = input_tensors[0]->shape;
  int input_dim = output_shape.size();
  PADDLE_ENFORCE_EQ(
      axis >= -input_dim && axis < input_dim,
      true,
      ::common::errors::InvalidArgument(
          "Concat's axis should be in [-R, R), but get axis: %d, R: %d.",
          axis,
          input_dim));
  if (axis < 0) axis += output_shape.size();

  for (int i = 1; i < input_size; i++) {
    PADDLE_ENFORCE_EQ(
        input_tensors[i]->shape.size(),
        input_dim,
        ::common::errors::InvalidArgument(
            "Dimensions of inputs tensors in Concat should be equal! Please "
            "check."));
    output_shape[axis] = optim::ArithSimplify(output_shape[axis] +
                                              input_tensors[i]->shape[axis]);
  }

  auto res = Compute(
      output_shape,
      [=](const std::vector<Expr>& indice) {
        auto ret = input_tensors[0](indice);
        Expr accumulate_shape = Expr(0);
        for (int i = 0; i < input_size - 1; i++) {
          accumulate_shape = optim::ArithSimplify(
              accumulate_shape + input_tensors[i]->shape[axis]);
          std::vector<Expr> new_indice = indice;
          new_indice[axis] =
              ir::Cast::Make(accumulate_shape.type(), indice[axis]) -
              accumulate_shape;
          ret =
              ir::Select::Make(ir::Cast::Make(accumulate_shape.type(),
                                              indice[axis]) < accumulate_shape,
                               ret,
                               input_tensors[i + 1](new_indice));
        }
        return ret;
      },
      name);
  return res;
}

std::vector<Tensor> MatmulV2(const Tensor& A,
                             const Tensor& B,
                             bool trans_a,
                             bool trans_b,
                             float alpha,
                             const std::string& name,
                             const cinn::common::Target& target) {
  std::vector<Expr> shape_A = A->shape;
  std::vector<Expr> shape_B = B->shape;
  int a_dim = shape_A.size();
  int b_dim = shape_B.size();
  PADDLE_ENFORCE_EQ(
      a_dim == 3U || a_dim == 2U,
      true,
      ::common::errors::InvalidArgument(
          "Tensor_A's dim should be 2 or 3 while current dim is %d.", a_dim));
  PADDLE_ENFORCE_EQ(
      b_dim == 3U || b_dim == 2U,
      true,
      ::common::errors::InvalidArgument(
          "Tensor_B's dim should be 2 or 3 while current dim is %d.", b_dim));
  PADDLE_ENFORCE_EQ(a_dim,
                    b_dim,
                    ::common::errors::InvalidArgument(
                        "tensor_A's dim should be same with tensor_B"));

  Expr x_width = trans_a ? shape_A[a_dim - 2] : shape_A.back();
  Expr y_height = trans_b ? shape_B.back() : shape_B[b_dim - 2];
  Expr M = trans_a ? shape_A.back() : shape_A[a_dim - 2];
  Expr N = trans_b ? shape_B[b_dim - 2] : shape_B.back();
  PADDLE_ENFORCE_EQ(
      is_zero(x_width - y_height),
      true,
      ::common::errors::InvalidArgument(
          "Matrix multiplication requires x_width to be same with y_height."));
  Var reduce_k(x_width, UniqName("reduce_k"));
  std::vector<Expr> output_shape;
  std::vector<ir::Tensor> out;

  if (a_dim == 3) {
    int max_batch = std::max(shape_A[0].as_int32(), shape_B[0].as_int32());
    output_shape = {Expr(max_batch), M, N};
  } else {
    output_shape = {M, N};
  }
  // array packing
  int shape_B_N = N.as_int32();
  int bn = GetArrayPackingFactor(shape_B_N, B->type(), target);
  // {N / bn, K, bn}
  std::vector<Expr> packedB_shape = {Expr(shape_B_N / bn), y_height, Expr(bn)};
  if (b_dim == 3) {
    packedB_shape.insert(packedB_shape.begin(), output_shape[0]);
  }
  auto packedB = Compute(
      packedB_shape,
      [=](const std::vector<Expr>& indice) {
        std::vector<Expr> indice_b;
        int indice_dim = indice.size();
        PADDLE_ENFORCE_GE(indice_dim,
                          3,
                          ::common::errors::InvalidArgument(
                              "packedB's dim should be at least 3."));
        if (indice_dim == 4) {
          // batch
          indice_b.push_back(indice[0]);
        }
        // k
        indice_b.push_back(indice[indice_dim - 2]);
        indice_b.push_back(Expr(bn) * indice[indice_dim - 3] + indice.back());
        if (trans_b) {
          std::swap(indice_b.back(), indice_b[indice_b.size() - 2]);
        }
        return B(indice_b);
      },
      UniqName("packedB"));

  auto res = Compute(
      output_shape,
      [=](const std::vector<Expr>& indice) {
        std::vector<Expr> indice_a;
        std::vector<Expr> indice_b;
        int out_dim = indice.size();
        PADDLE_ENFORCE_EQ(
            out_dim == 3U || out_dim == 2U,
            true,
            ::common::errors::InvalidArgument(
                "Indice size should be 2 or 3 while current dim is %d.",
                out_dim));
        if (out_dim == 3) {
          // batch
          indice_a.push_back(indice[0]);
          indice_b.push_back(indice[0]);
        }
        indice_a.push_back(indice[out_dim - 2]);
        indice_a.push_back(reduce_k);
        indice_b.push_back(indice[out_dim - 1] / Expr(bn));
        indice_b.push_back(reduce_k);
        indice_b.push_back(indice[out_dim - 1] % Expr(bn));
        if (trans_a) {
          std::swap(indice_a.back(), indice_a[indice_a.size() - 2]);
        }
        if (alpha == 1) {
          return lang::ReduceSum(A(indice_a) * packedB(indice_b), {reduce_k});
        } else {
          return lang::ReduceSum(A(indice_a) * packedB(indice_b) *
                                     ir::Cast::Make(A->type(), Expr(alpha)),
                                 {reduce_k});
        }
      },
      UniqName("matmulV2_out"));
  return {res, packedB};
}

std::vector<Tensor> MatmulMKL(const Tensor& A,
                              const Tensor& B,
                              bool trans_a,
                              bool trans_b,
                              float alpha,
                              const std::string& name,
                              const cinn::common::Target& target) {
  PADDLE_ENFORCE_EQ(std::holds_alternative<common::X86Arch>(target.arch),
                    true,
                    ::common::errors::InvalidArgument(
                        "Mkl should be used in the cpu environment."));
  std::vector<Expr> shape_A = A->shape;
  std::vector<Expr> shape_B = B->shape;
  int a_dim = shape_A.size();
  int b_dim = shape_B.size();
  PADDLE_ENFORCE_EQ(
      a_dim == 3U || a_dim == 2U,
      true,
      ::common::errors::InvalidArgument(
          "Tensor_A's dim should be 2 or 3 while current dim is %d.", a_dim));
  PADDLE_ENFORCE_EQ(
      b_dim == 3U || b_dim == 2U,
      true,
      ::common::errors::InvalidArgument(
          "Tensor_B's dim should be 2 or 3 while current dim is %d.", b_dim));
  PADDLE_ENFORCE_EQ(a_dim,
                    b_dim,
                    ::common::errors::InvalidArgument(
                        "tensor_A's dim should be same with tensor_B"));
  if (a_dim == 3U) {
    PADDLE_ENFORCE_EQ(shape_A.front(),
                      shape_B.front(),
                      ::common::errors::InvalidArgument(
                          "tensor A and B's batch size should be same."));
  }

  Expr x_width = trans_a ? shape_A[a_dim - 2] : shape_A.back();
  Expr y_height = trans_b ? shape_B.back() : shape_B[b_dim - 2];
  Expr M = trans_a ? shape_A.back() : shape_A[a_dim - 2];
  Expr N = trans_b ? shape_B[b_dim - 2] : shape_B.back();
  PADDLE_ENFORCE_EQ(
      is_zero(x_width - y_height),
      true,
      ::common::errors::InvalidArgument(
          "Matrix multiplication requires x_width to be same with y_height."));

  ir::Tensor call;
  if (a_dim == 2U) {
    call = Compute(
        {Expr(1)},
        [=]() -> Expr {
          return lang::CallExtern("cinn_cpu_mkl_gemm_fp32",
                                  {
                                      Expr(alpha),  // alpha
                                      M,            // M
                                      N,            // N
                                      x_width,      // K
                                      cinn::common::make_bool(trans_a),  // ta
                                      cinn::common::make_bool(trans_b),  // tb
                                      shape_A.back(),                    // lda
                                      shape_B.back(),                    // ldb
                                      N,                                 // ldc
                                      cinn::common::make_zero<float>(),  // beta
                                      A,                                 // A
                                      B,                                 // B
                                  });
        },
        UniqName("matmul_mkl_out"));
  } else {
    // batch matmul
    call = Compute(
        {Expr(1)},
        [=]() -> Expr {
          return lang::CallExtern("cinn_cpu_mkl_gemm_batch_fp32",
                                  {
                                      Expr(alpha),      // alpha
                                      shape_A.front(),  // batch
                                      M,                // M
                                      N,                // N
                                      x_width,          // K
                                      cinn::common::make_bool(trans_a),  // ta
                                      cinn::common::make_bool(trans_b),  // tb
                                      shape_A.back(),                    // lda
                                      shape_B.back(),                    // ldb
                                      N,                                 // ldc
                                      M * x_width,  // a_stride
                                      N * x_width,  // b_stride
                                      M * N,        // c_stride
                                      cinn::common::make_zero<float>(),  // beta
                                      A,                                 // A
                                      B,                                 // B
                                  });
        },
        UniqName("batch_matmul_mkl_out"));
  }
  auto out = call->TupleGet(0);
  out->WithBuffer(A->type());
  return {out, call};
}

int GetMulFactor(int shape,
                 const Type& type,
                 const cinn::common::Target& target) {
  int split_base = GetBasicFactor(type, target);
  int split_factor = 1;
  for (size_t i = split_base; i >= 1; --i) {
    if (shape % i == 0) {
      split_factor = i;
      break;
    }
  }
  return split_factor;
}

std::vector<Tensor> MulBaseCallImpl(common::UnknownArch,
                                    const Tensor& A,
                                    const Tensor& B,
                                    const std::string& name,
                                    const cinn::common::Target& target) {
  LOG(FATAL) << "NotImplemented.";
}

std::vector<Tensor> MulBaseCallImpl(common::X86Arch,
                                    const Tensor& A,
                                    const Tensor& B,
                                    const std::string& name,
                                    const cinn::common::Target& target) {
  std::vector<Expr> output_shape;
  PADDLE_ENFORCE_EQ(
      A->shape.size(),
      2U,
      ::common::errors::InvalidArgument(
          "tensor_A's shape size should be two while current shape size is %d",
          A->shape.size()));
  PADDLE_ENFORCE_EQ(
      B->shape.size(),
      2U,
      ::common::errors::InvalidArgument(
          "tensor_B's shape size should be two while current shape size is %d",
          B->shape.size()));
  PADDLE_ENFORCE_EQ(A->shape[1],
                    B->shape[1],
                    ::common::errors::InvalidArgument(
                        "tensor_A's last shape should be same with tensor_B"));
  output_shape.push_back(A->shape[0]);
  output_shape.push_back(B->shape[0]);

  int reduce_dim = A->shape[1].as_int32();
  int split_factor = GetMulFactor(reduce_dim, A->type(), target);
  Var reduce_k_first(
      ir::Cast::Make(A->shape[1]->type(), Expr(reduce_dim / split_factor)),
      UniqName("reduce_k_first"));
  auto mul_reduce_first = Compute(
      {A->shape[0], B->shape[0], Expr(split_factor)},
      [=](const std::vector<Expr>& indice) {
        PADDLE_ENFORCE_EQ(
            indice.size(),
            3U,
            ::common::errors::InvalidArgument(
                "indice size should be three while current size is %d",
                indice.size()));
        return lang::ReduceSum(
            A({indice[0], reduce_k_first * Expr(split_factor) + indice[2]}) *
                B({indice[1], reduce_k_first * Expr(split_factor) + indice[2]}),
            {reduce_k_first});
      },
      UniqName("mul_reduce_k_first"));
  Var reduce_k_second(ir::Cast::Make(A->shape[1]->type(), Expr(split_factor)),
                      UniqName("reduce_k_second"));
  return {Compute(
              output_shape,
              [=](const std::vector<Expr>& indice) {
                std::vector<Expr> new_indice = indice;
                new_indice.push_back(reduce_k_second);
                return lang::ReduceSum(mul_reduce_first(new_indice),
                                       {reduce_k_second});
              },
              name),
          mul_reduce_first};
}

std::vector<Tensor> MulBaseCallImpl(common::ARMArch,
                                    const Tensor& A,
                                    const Tensor& B,
                                    const std::string& name,
                                    const cinn::common::Target& target) {
  LOG(FATAL) << "NotImplemented.";
}

std::vector<Tensor> MulBaseCallImplNvHygon(const Tensor& A,
                                           const Tensor& B,
                                           const std::string& name,
                                           const cinn::common::Target& target) {
  std::vector<Expr> output_shape;
  PADDLE_ENFORCE_EQ(
      A->shape.size(),
      2U,
      ::common::errors::InvalidArgument(
          "tensor_A's shape size should be two while current shape size is %d",
          A->shape.size()));
  PADDLE_ENFORCE_EQ(
      B->shape.size(),
      2U,
      ::common::errors::InvalidArgument(
          "tensor_B's shape size should be two while current shape size is %d",
          B->shape.size()));
  PADDLE_ENFORCE_EQ(A->shape[1],
                    B->shape[1],
                    ::common::errors::InvalidArgument(
                        "tensor_A's last shape should be same with tensor_B"));
  output_shape.push_back(A->shape[0]);
  output_shape.push_back(B->shape[0]);

  Var reduce_k(A->shape[1], UniqName("reduce_k"));
  return {Compute(
      output_shape,
      [=](const std::vector<Expr>& indice) {
        std::vector<Expr> A_indice;
        std::vector<Expr> B_indice;
        PADDLE_ENFORCE_EQ(
            indice.size(),
            2U,
            ::common::errors::InvalidArgument(
                "indice size should be two while current size is %d",
                indice.size()));
        A_indice.push_back(indice[0]);
        B_indice.push_back(indice[1]);
        A_indice.push_back(reduce_k);
        B_indice.push_back(reduce_k);
        return lang::ReduceSum(A(A_indice) * B(B_indice), {reduce_k});
      },
      name)};
}

std::vector<Tensor> MulBaseCallImpl(common::NVGPUArch,
                                    const Tensor& A,
                                    const Tensor& B,
                                    const std::string& name,
                                    const cinn::common::Target& target) {
  MulBaseCallImplNvHygon(A, B, name, target);
}

std::vector<Tensor> MulBaseCallImpl(common::HygonDCUArchHIP,
                                    const Tensor& A,
                                    const Tensor& B,
                                    const std::string& name,
                                    const cinn::common::Target& target) {
  MulBaseCallImplNvHygon(A, B, name, target);
}

std::vector<Tensor> MulBaseCallImpl(common::HygonDCUArchSYCL,
                                    const Tensor& A,
                                    const Tensor& B,
                                    const std::string& name,
                                    const cinn::common::Target& target) {
  MulBaseCallImplNvHygon(A, B, name, target);
}

std::vector<Tensor> MulBaseCall(const Tensor& A,
                                const Tensor& B,
                                const std::string& name,
                                const cinn::common::Target& target) {
  return std::visit(
      [&](const auto& impl) {
        return MulBaseCallImpl(impl, A, B, name, target);
      },
      target.arch.variant());
}

std::vector<Tensor> MulBase(const Tensor& A,
                            const Tensor& B,
                            const std::string& name,
                            const cinn::common::Target& target) {
  return MulBaseCall(A, B, name, target);
}

std::vector<Tensor> Mul(const Tensor& A,
                        const Tensor& B,
                        int x_num_col_dims,
                        const std::vector<Expr>& output_shape,
                        const Var& axis_k,
                        const std::string& name) {
  return {Compute(
      output_shape,
      [=](const std::vector<Expr>& indice) {
        std::vector<Expr> A_indice;
        std::vector<Expr> B_indice;
        A_indice.insert(
            A_indice.begin(), indice.begin(), indice.begin() + x_num_col_dims);
        B_indice.insert(
            B_indice.begin(), indice.begin() + x_num_col_dims, indice.end());
        A_indice.push_back(axis_k);
        B_indice.push_back(axis_k);
        return lang::ReduceSum(A(A_indice) * B(B_indice), {axis_k});
      },
      name)};
}

std::vector<Tensor> MulMKL(const Tensor& A,
                           const Tensor& B,
                           const std::string& name,
                           const cinn::common::Target& target) {
  PADDLE_ENFORCE_EQ(std::holds_alternative<cinn::common::X86Arch>(target.arch),
                    true,
                    ::common::errors::InvalidArgument(
                        "Mkl should be used in the cpu environment."));
  std::vector<Expr> shape_A = A->shape;
  std::vector<Expr> shape_B = B->shape;
  int a_dim = shape_A.size();
  int b_dim = shape_B.size();
  PADDLE_ENFORCE_EQ(
      a_dim,
      2U,
      ::common::errors::InvalidArgument(
          "tensor_A's shape size should be two while current shape size is %d",
          a_dim));
  PADDLE_ENFORCE_EQ(
      b_dim,
      2U,
      ::common::errors::InvalidArgument(
          "tensor_B's shape size should be two while current shape size is %d",
          b_dim));
  // A: [M, K], B: [N, K]
  Expr x_width = shape_A[1];
  Expr y_height = shape_B[1];
  Expr M = shape_A[0];
  Expr N = shape_B[0];
  PADDLE_ENFORCE_EQ(
      is_zero(x_width - y_height),
      true,
      ::common::errors::InvalidArgument(
          "Matrix multiplication requires x_width to be same with y_height."));
  PADDLE_ENFORCE_EQ(A->shape[1],
                    B->shape[1],
                    ::common::errors::InvalidArgument(
                        "tensor_A's last shape should be same with tensor_B"));

  auto call = Compute(
      {Expr(1)},
      [=]() -> Expr {
        return lang::CallExtern("cinn_cpu_mkl_gemm_fp32",
                                {
                                    Expr(1.0f),                        // alpha
                                    M,                                 // M
                                    N,                                 // N
                                    x_width,                           // K
                                    cinn::common::make_bool(false),    // ta
                                    cinn::common::make_bool(true),     // tb
                                    shape_A.back(),                    // lda
                                    shape_B.back(),                    // ldb
                                    N,                                 // ldc
                                    cinn::common::make_zero<float>(),  // beta
                                    A,                                 // A
                                    B,                                 // B
                                });
      },
      UniqName("mul_mkl_out"));
  auto out = call->TupleGet(0);
  out->WithBuffer(A->type());
  return {out, call};
}

void GetLayoutTransformInfo(
    const ir::Layout& src_layout,
    const ir::Layout& dst_layout,
    absl::flat_hash_map<int, std::vector<int>>* split_index_map) {
  PADDLE_ENFORCE_GT(
      dst_layout.ndims(),
      src_layout.ndims(),
      ::common::errors::InvalidArgument(
          "dst_layout's ndims should be larger than src_layout's ndims"));
  int offset = 'A' - 'a';
  PADDLE_ENFORCE_EQ(
      dst_layout.axis_names().size(),
      dst_layout.ndims(),
      ::common::errors::InvalidArgument(
          "dst_layout's axis_names size should be equal to ndims"));
  for (int i = dst_layout.ndims() - 1; i >= 0; i--) {
    char axis_name = dst_layout.axis_names(i);
    char prim_axis_name = axis_name;
    if (axis_name >= 'a' && axis_name <= 'z') {
      prim_axis_name += offset;
      int factor = dst_layout[i]->upper_bound.as_int32();

      PADDLE_ENFORCE_GT(factor,
                        0,
                        ::common::errors::InvalidArgument(
                            "sub-axis factor should be larger than 0"));
      int src_primal_index = src_layout.axis_names().find(prim_axis_name);
      int dst_primal_index = dst_layout.axis_names().find(prim_axis_name);
      PADDLE_ENFORCE_EQ(
          src_primal_index != src_layout.axis_names().npos,
          true,
          ::common::errors::InvalidArgument(
              "Src primal index should not be equal to src layout npos."));
      PADDLE_ENFORCE_EQ(
          dst_primal_index != dst_layout.axis_names().npos,
          true,
          ::common::errors::InvalidArgument(
              "Dst primal index should not be equal to dst layout npos."));
      (*split_index_map)[src_primal_index] = {dst_primal_index, i, factor};
    } else {
      int src_primal_index = src_layout.axis_names().find(prim_axis_name);
      if (split_index_map->find(src_primal_index) != split_index_map->end())
        continue;
      PADDLE_ENFORCE_EQ(
          src_primal_index != src_layout.axis_names().npos,
          true,
          ::common::errors::InvalidArgument(
              "Src primal index should not be equal to src layout npos."));
      (*split_index_map)[src_primal_index] = {i};
    }
  }
}

std::vector<Expr> InferShapeLayoutTransform(
    const std::vector<Expr>& input_shapes,
    const ir::Layout& old_layout,
    const ir::Layout& new_layout,
    absl::flat_hash_map<int, std::vector<int>>* split_index_map) {
  int src_dim = old_layout.ndims();
  int dst_dim = new_layout.ndims();
  std::vector<Expr> output_shape(dst_dim);
  PADDLE_ENFORCE_EQ(input_shapes.size(),
                    src_dim,
                    ::common::errors::InvalidArgument(
                        "input_shapes size should be equal to src_dim"));

  if (src_dim == dst_dim) {
    PADDLE_ENFORCE_EQ(old_layout.name(),
                      new_layout.name(),
                      ::common::errors::InvalidArgument(
                          "src_layout should be equal to dst_layout"));
    return input_shapes;
  } else if (src_dim < dst_dim) {
    GetLayoutTransformInfo(old_layout, new_layout, split_index_map);
    for (int i = 0; i < src_dim; i++) {
      PADDLE_ENFORCE_EQ(
          split_index_map->find(i) != split_index_map->end(),
          true,
          ::common::errors::InvalidArgument(
              "Spilt index map found should not be equal to end."));
      if ((*split_index_map)[i].size() == 3) {
        int dst_prim_index = (*split_index_map)[i][0];
        int dst_sub_index = (*split_index_map)[i][1];
        int factor = (*split_index_map)[i][2];
        Expr chunk_shape = optim::ArithSimplify(input_shapes[i] / factor);
        Expr block_shape = Expr(factor);
        output_shape[dst_prim_index] = chunk_shape;
        output_shape[dst_sub_index] = block_shape;
      } else if ((*split_index_map)[i].size() == 1) {
        int dst_prim_index = (*split_index_map)[i][0];
        output_shape[dst_prim_index] = input_shapes[i];
      }
    }
  } else {
    GetLayoutTransformInfo(new_layout, old_layout, split_index_map);
    for (int i = 0; i < dst_dim; i++) {
      PADDLE_ENFORCE_EQ(
          split_index_map->find(i) != split_index_map->end(),
          true,
          ::common::errors::InvalidArgument(
              "Spilt index map found should not be equal to end."));
      if ((*split_index_map)[i].size() == 3) {
        int src_prim_index = (*split_index_map)[i][0];
        int src_sub_index = (*split_index_map)[i][1];
        int factor = (*split_index_map)[i][2];
        PADDLE_ENFORCE_GE(
            input_shapes.size(),
            src_sub_index,
            ::common::errors::InvalidArgument(
                "input_shapes size should be larger than src_sub_index"));
        PADDLE_ENFORCE_EQ(
            input_shapes[src_sub_index].as_int32(),
            factor,
            ::common::errors::InvalidArgument(
                "input_shapes[src_sub_index] should be equal to factor"));
        output_shape[i] =
            optim::ArithSimplify(input_shapes[src_prim_index] * factor);
      } else if ((*split_index_map)[i].size() == 1) {
        int src_prim_index = (*split_index_map)[i][0];
        output_shape[i] = input_shapes[src_prim_index];
      }
    }
  }
  VLOG(4) << "output_shape: " << output_shape;
  return output_shape;
}

ir::Tensor LayoutTransform(const Tensor& input,
                           const std::string& src_layout,
                           const std::string& dst_layout,
                           const std::string& name) {
  PADDLE_ENFORCE_EQ(
      src_layout != dst_layout,
      true,
      ::common::errors::InvalidArgument("Dst layout is same with src_layout, "
                                        "should not do layout transform."));
  // NCHW -> NCHWxc
  // NCHWxc -> NCHW
  // OIHW -> OIHWxixo
  // OIHWxixo -> OIHW
  PADDLE_ENFORCE_GE(src_layout.size(),
                    4U,
                    ::common::errors::InvalidArgument(
                        "src_layout size should be larger than 4"));
  PADDLE_ENFORCE_GE(dst_layout.size(),
                    4U,
                    ::common::errors::InvalidArgument(
                        "dst_layout size should be larger than 4"));
  absl::flat_hash_map<int, std::vector<int>> split_index_map;
  // transform shape
  int offset = 'A' - 'a';
  ir::Layout old_layout(src_layout);
  ir::Layout new_layout(dst_layout);
  int src_dim = old_layout.ndims();
  int dst_dim = new_layout.ndims();
  std::vector<Expr> output_shape = InferShapeLayoutTransform(
      input->shape, old_layout, new_layout, &split_index_map);
  PADDLE_ENFORCE_EQ(output_shape.size(),
                    dst_dim,
                    ::common::errors::InvalidArgument(
                        "output_shape size should be equal to dst_dim"));

  auto res = Compute(
      output_shape,
      [=](const std::vector<Expr>& indice) {
        // transform indice
        std::vector<Expr> new_indice(src_dim);
        int min_dim = std::min(src_dim, dst_dim);
        for (int i = 0; i < min_dim; i++) {
          PADDLE_ENFORCE_EQ(
              split_index_map.find(i) != split_index_map.end(),
              true,
              ::common::errors::InvalidArgument(
                  "Spilt index map found should not be equal to end."));
          std::vector<int> split_infos = split_index_map.at(i);
          if (split_infos.size() == 3) {
            int prim_index = split_infos[0];
            int sub_index = split_infos[1];
            int factor = split_infos[2];
            if (dst_dim > src_dim) {
              new_indice[i] = optim::ArithSimplify(indice[prim_index] * factor +
                                                   indice[sub_index]);
            } else {
              new_indice[prim_index] = optim::ArithSimplify(indice[i] / factor);
              new_indice[sub_index] = optim::ArithSimplify(indice[i] % factor);
            }

          } else if (split_infos.size() == 1) {
            int prim_index = split_infos[0];
            if (dst_dim > src_dim) {
              new_indice[i] = indice[prim_index];
            } else {
              new_indice[prim_index] = indice[i];
            }
          }
        }
        VLOG(4) << "new_indice: " << new_indice;

        return input(new_indice);
      },
      name);
  return {res};
}

ir::Tensor Reverse(const ir::Tensor& input,
                   const std::vector<int>& axis,
                   const std::string& output_name) {
  for (auto& val : axis) {
    PADDLE_ENFORCE_EQ(
        val >= 0 && val < static_cast<int>(input->shape.size()),
        true,
        ::common::errors::InvalidArgument("Axis should be [0,n_dim)."));
  }
  std::vector<Expr> shape = input->shape;
  return lang::Compute(
      input->shape,
      [=](const std::vector<Expr>& indices) {
        std::vector<Expr> out_indices(indices.begin(), indices.end());
        for (auto idx : axis) {
          out_indices[idx] = shape[idx] - Expr(1) - out_indices[idx];
        }
        return input(out_indices);
      },
      output_name);
}

ir::Tensor Transpose(const ir::Tensor& input,
                     const std::vector<int>& axis,
                     const std::string& output_name) {
  PADDLE_ENFORCE_GE(
      input->shape.size(),
      axis.size(),
      ::common::errors::InvalidArgument("input shape size should be equal to "
                                        "or greater than the axis's size."));
  for (int idx = 0; idx < axis.size(); ++idx) {
    PADDLE_ENFORCE_EQ(axis[idx] >= 0 && axis[idx] < axis.size(),
                      true,
                      ::common::errors::InvalidArgument(
                          "Axis value should be among [0,axis.size())."));
    for (int idy = idx + 1; idy < axis.size(); ++idy) {
      PADDLE_ENFORCE_NE(
          axis[idx],
          axis[idy],
          ::common::errors::InvalidArgument("axis value can't repeat!"));
    }
  }
  // compute output shape
  const std::vector<Expr>& shape = input->shape;
  std::vector<Expr> output_shape = input->shape;
  for (auto idx = 0; idx < axis.size(); ++idx) {
    output_shape[idx] = shape[axis[idx]];
  }

  // transpose axis to map output to input
  // new_axis = axis(T)
  std::vector<int> new_axis;
  for (int idx = 0; idx < axis.size(); ++idx) {
    for (int idy = 0; idy < axis.size(); ++idy) {
      if (idx == axis[idy]) {
        new_axis.push_back(idy);
      }
    }
  }
  // If new_axis size is less than output shape, add axis to the end.
  // For example: input shape is [2,3,4], axis is [1,0],
  //              output_shape is [3,2,4], the new_axis is [1,0,2]
  while (new_axis.size() < output_shape.size()) {
    new_axis.push_back(new_axis.size());
  }

  return lang::Compute(
      output_shape,
      [=](const std::vector<Expr>& indices) {
        std::vector<Expr> out_indices;
        for (auto idx : new_axis) {
          out_indices.push_back(indices[idx]);
        }
        return input(out_indices);
      },
      output_name);
}

int UpdateNegAxis(int axis, int rank) {
  if (axis < 0) {
    PADDLE_ENFORCE_GE(
        axis + rank,
        0,
        ::common::errors::InvalidArgument("The axis of slice is out of range"));
    return axis + rank;
  }
  return axis;
}

ir::Tensor Slice(const ir::Tensor& A,
                 const std::vector<int>& starts,
                 const std::vector<int>& const_axes,
                 const std::vector<int>& strides,
                 const std::vector<int>& decrease_axis,
                 const std::vector<Expr>& output_shape,
                 const std::string& output_name) {
  std::vector<int> input_shape;
  for (const auto& shape : A->shape) {
    input_shape.emplace_back(shape.as_int32());
  }
  std::vector<int> axes;
  std::transform(const_axes.begin(),
                 const_axes.end(),
                 std::back_inserter(axes),
                 [rank = A->shape.size()](const int axis) -> int {
                   return UpdateNegAxis(axis, rank);
                 });
  std::vector<int> new_starts(starts);
  for (int i = 0; i < axes.size(); i++) {
    if (new_starts[i] < -input_shape[axes[i]]) {
      new_starts[i] = 0;
    } else if (new_starts[i] < 0) {
      new_starts[i] = input_shape[axes[i]] + new_starts[i];
    } else if (new_starts[i] > input_shape[axes[i]]) {
      new_starts[i] = input_shape[axes[i]] - 1;
    }
  }

  // output = input[starts:ends:strides]
  // Note that when strides < 0, the output reverse:
  // data=[[1,2,3,4],[5,6,7,8],]
  // axes=[0,1]
  // starts=[1,3]
  // ends=[2,0]
  // strides=[1,-1]
  // ==> result=[[8,7,6],]
  return Compute(
      output_shape,
      [=](const std::vector<Expr>& indice) {
        std::vector<Expr> temp;
        int indice_i = 0;
        for (int i = 0; i < input_shape.size(); ++i) {
          if (std::find(decrease_axis.cbegin(), decrease_axis.cend(), i) !=
              decrease_axis.cend()) {
            temp.emplace_back(0);
          } else {
            temp.emplace_back(indice[indice_i]);
            indice_i++;
          }
        }
        for (int i = 0; i < axes.size(); i++) {
          temp[axes[i]] =
              temp[axes[i]] * Expr(strides[i]) + Expr(new_starts[i]);
        }
        return A(temp);
      },
      output_name);
}

ir::Tensor SliceSymbolic(const ir::Tensor& A,
                         const std::vector<Expr>& starts,
                         const std::vector<int>& const_axes,
                         const std::vector<Expr>& strides,
                         const std::vector<int>& decrease_axis,
                         const std::vector<Expr>& output_shape,
                         const std::string& output_name) {
  std::vector<Expr> input_shape;
  for (const auto& shape : A->shape) {
    input_shape.emplace_back(shape);
  }

  std::vector<Expr> new_starts = starts;
  std::vector<int64_t> axes;
  std::transform(const_axes.begin(),
                 const_axes.end(),
                 std::back_inserter(axes),
                 [rank = A->shape.size()](const int axis) -> int {
                   return UpdateNegAxis(axis, rank);
                 });

  for (int i = 0; i < axes.size(); i++) {
    if (input_shape[axes[i]].is_constant() && new_starts[i].is_constant()) {
      if (new_starts[i].as_int64() < -input_shape[axes[i]].as_int64()) {
        new_starts[i] = ir::Expr(0);
      } else if (new_starts[i].as_int64() < 0) {
        new_starts[i] = input_shape[axes[i]].as_int64() + new_starts[i];
      } else if (new_starts[i].as_int64() > input_shape[axes[i]].as_int64()) {
        new_starts[i] = input_shape[axes[i]].as_int64() - ir::Expr(1);
      }
    } else if (new_starts[i]
                   .is_constant()) {  // input_shape[axes[i]] is not constant
      if (new_starts[i].as_int64() < 0) {
        new_starts[i] = ir::Max::Make(
            Expr(0), ir::Add::Make(input_shape[axes[i]], new_starts[i]));
      } else {
        new_starts[i] = ir::Min::Make(input_shape[axes[i]], new_starts[i]);
      }
    } else if (input_shape[axes[i]]
                   .is_constant()) {  // new_starts[i] is not constant, only
                                      // support new_starts[i] >= 0
      new_starts[i] = ir::Min::Make(input_shape[axes[i]], new_starts[i]);
    } else {  // both are not constant, only support new_starts[i] >= 0
      new_starts[i] = ir::Min::Make(input_shape[axes[i]], new_starts[i]);
    }
  }

  // output = input[starts:ends:strides]
  // Note that when strides < 0, the output reverse:
  // data=[[1,2,3,4],[5,6,7,8],]
  // axes=[0,1]
  // starts=[1,3]
  // ends=[2,0]
  // strides=[1,-1]
  // ==> result=[[8,7,6],]
  return Compute(
      output_shape,
      [=](const std::vector<Expr>& indice) {
        std::vector<Expr> temp;
        int indice_i = 0;
        for (int i = 0; i < input_shape.size(); ++i) {
          if (std::find(decrease_axis.cbegin(), decrease_axis.cend(), i) !=
              decrease_axis.cend()) {
            temp.emplace_back(0);
          } else {
            temp.emplace_back(indice[indice_i]);
            indice_i++;
          }
        }
        for (int i = 0; i < axes.size(); i++) {
          temp[axes[i]] =
              temp[axes[i]] * Expr(strides[i]) + Expr(new_starts[i]);
        }
        return A(temp);
      },
      output_name);
}

ir::Tensor SliceAssign(const ir::Tensor& input,
                       const ir::Tensor& assign,
                       const std::vector<int>& axes,
                       const std::vector<int>& starts,
                       const std::vector<int>& ends,
                       const std::vector<int>& strides,
                       const std::string& output_name) {
  PADDLE_ENFORCE_EQ(axes.size(),
                    starts.size(),
                    ::common::errors::InvalidArgument(
                        "axes's size should be equal to starts's size"));
  PADDLE_ENFORCE_EQ(axes.size(),
                    ends.size(),
                    ::common::errors::InvalidArgument(
                        "axes's size should be equal to ends's size"));
  PADDLE_ENFORCE_EQ(axes.size(),
                    strides.size(),
                    ::common::errors::InvalidArgument(
                        "axes's size should be equal to strides's size"));

  std::vector<int> input_shape;
  for (const auto& shape : input->shape) {
    input_shape.emplace_back(shape.as_int32());
  }
  std::vector<int> new_starts(starts);
  std::vector<int> new_ends(ends);
  std::vector<int> new_strides(strides);
  for (int i = 0; i < axes.size(); i++) {
    PADDLE_ENFORCE_LT(axes[i],
                      input->shape.size(),
                      ::common::errors::InvalidArgument(
                          "axes should less than input's shape size"));

    if (new_starts[i] < 0) {
      new_starts[i] = input_shape[axes[i]] + new_starts[i];
      PADDLE_ENFORCE_GE(new_starts[i],
                        0,
                        ::common::errors::InvalidArgument(
                            "The value of [starts] should not less than 0"));
    }
    if (new_starts[i] > input_shape[axes[i]]) {
      new_starts[i] = input_shape[axes[i]];
    }
    if (new_ends[i] < 0) {
      new_ends[i] = input_shape[axes[i]] + new_ends[i];
      PADDLE_ENFORCE_GE(new_ends[i],
                        0,
                        ::common::errors::InvalidArgument(
                            "The value of [ends] should not less than 0"));
    }
    if (new_ends[i] > input_shape[axes[i]]) {
      new_ends[i] = input_shape[axes[i]];
    }

    // if strides < 0, starts > ends, we need swap them
    PADDLE_ENFORCE_NE(strides[i],
                      0,
                      ::common::errors::InvalidArgument(
                          "[strides] should not be 0 ! Please Check."));
    if (strides[i] < 0) {
      PADDLE_ENFORCE_GT(
          new_starts[i],
          new_ends[i],
          ::common::errors::InvalidArgument(
              "[starts] should greater than [ends] when [strides] < 0"));
      // if strides > 0, the range is [starts, ends)
      // but if strides < 0, the range is (ends, starts]
      auto tmp = new_starts[i];
      new_starts[i] =
          new_ends[i] + 1;    // the new starts should not contain ends[i]
      new_ends[i] = tmp + 1;  // the new ends should contain starts[i]

      new_strides[i] = -new_strides[i];
    } else {
      PADDLE_ENFORCE_LT(
          new_starts[i],
          new_ends[i],
          ::common::errors::InvalidArgument(
              "[starts] should less than [ends] when [strides] > 0"));
    }
  }

  // input[starts:ends:strides] = assign
  auto output_tensor = Compute(
      input->shape,
      [=](const std::vector<Expr>& indice) {
        ir::Expr is_assigned = ir::Expr(true);
        std::vector<ir::Expr> tmp_indice = indice;
        for (int idx = 0; idx < axes.size(); ++idx) {
          // get input axis to be assigned
          auto tmp_axis = indice[axes[idx]];
          // get assign axis
          Expr out_axis;
          if (strides[idx] > 0) {
            out_axis = tmp_axis - ir::Expr(new_starts[idx]);
          } else {
            // when strides < 0, reverse input to output.
            // the value of ends is not contained in slice, so `ends - 1`
            out_axis = ir::Expr(new_ends[idx] - 1) - tmp_axis;
          }
          // axis >= start
          auto ge = ir::GE::Make(tmp_axis, ir::Expr(new_starts[idx]));
          // axis < ends
          auto lt = ir::LT::Make(tmp_axis, ir::Expr(new_ends[idx]));
          // check start <= axis < ends
          auto inside = ir::And::Make(ge, lt);
          // check (axis - starts) % strides == 0
          auto mod = ir::EQ::Make(
              ir::Mod::Make(out_axis, Expr(new_strides[idx])), Expr(0));
          // check start <= axis < ends and (axis - starts) % strides == 0
          is_assigned = ir::And::Make(is_assigned, ir::And::Make(inside, mod));
          // update axis for assign tensor
          tmp_indice[axes[idx]] = out_axis / Expr(new_strides[idx]);
        }
        return ir::Select::Make(is_assigned, assign(tmp_indice), input(indice));
      },
      output_name);
  return output_tensor;
}

ir::Tensor Gather(const ir::Tensor& x,
                  const ir::Tensor& index,
                  const std::vector<Expr>& output_shape,
                  int axis,
                  const std::string& name) {
  PADDLE_ENFORCE_EQ(x->shape.size(),
                    index->shape.size(),
                    ::common::errors::InvalidArgument(
                        "The rank of x and index must be same."));
  // The implementation details are explained below.
  // If output_shape = [2, 4, 3] and axis = 0, `Compute` can be translated as
  // the following code:
  // {
  //   for (i, 0, 2)
  //   {
  //     for (j, 0, 4)
  //     {
  //       for (k, 0, 3)
  //       {
  //         index_select_output[i, j, k] = X[int32(Index[i, j, k]), j, k]
  //       }
  //     }
  //   }
  // }
  auto output_tensor = Compute(
      output_shape,
      [x, index, axis](const std::vector<Expr>& indice) {
        // 1) indice is got from `output_shape`
        // 2) transformed_indice is used in the input `x`
        std::vector<Expr> transformed_indice = indice;
        // The element type of index maybe int64, but the index type is limited
        // to int32 in CINN. See the below link for more details:
        // https://github.com/PaddlePaddle/CINN/blob/85ab4981a38926dc5c1dbf672762cec335d2b857/cinn/ir/ir.cc#L477
        transformed_indice[axis] =
            ir::Cast::Make(cinn::common::Int(32), index(indice));
        return x(transformed_indice);
      },
      name);
  return output_tensor;
}

ir::Tensor Gather(const ir::Tensor& x,
                  const ir::Tensor& index,
                  int axis,
                  const std::vector<Expr>& output_shape,
                  const std::string& name) {
  // The implementation details are explained below.
  // If output_shape = [2, 4, 3] and axis = 0, `Compute` can be translated as
  // the following code:
  // {
  //   for (i, 0, 2)
  //   {
  //     for (j, 0, 4)
  //     {
  //       for (k, 0, 3)
  //       {
  //         index_select_output[i, j, k] = X[index(i), j, k]
  //       }
  //     }
  //   }
  // }
  auto output_tensor = Compute(
      output_shape,
      [x, index, axis](const std::vector<Expr>& indice) {
        // 1) indice is got from `output_shape`
        // 2) transformed_indice is used in the input `x`
        std::vector<Expr> transformed_indice = indice;

        auto index_indice = std::vector<Expr>({indice[axis]});

        if (index.ndims() > 1) {
          PADDLE_ENFORCE_EQ(
              index.ndims(),
              2,
              ::common::errors::InvalidArgument(
                  "index.ndims() should be 2 when index.ndims() is not 0 or 1"
                  "in gather_op, but received value is [%d].",
                  index.ndims()));
          PADDLE_ENFORCE_EQ(
              index->shape[1],
              Expr(1),
              ::common::errors::InvalidArgument(
                  "index->shape[1] should be 1 when index.ndims() = 2"
                  "in gather_op."));

          index_indice.push_back(Expr(0));
        }

        transformed_indice[axis] = index(index_indice);

        return x(transformed_indice);
      },
      name);
  return output_tensor;
}

ir::Tensor ScatterAssign(const ir::Tensor& input,
                         const ir::Tensor& updates,
                         const ir::Tensor& index,
                         const cinn::common::Target& target,
                         const int axis,
                         const std::string& output_name) {
  PADDLE_ENFORCE_EQ(
      index->type(),
      cinn::common::Int(32),
      ::common::errors::InvalidArgument("index's type should be int32"));
  std::string extern_fun_name;
  target.arch.Match(
      [&](common::UnknownArch) {
        PADDLE_THROW(::common::errors::Fatal(
            "ScatterAssign only support X86 and NVGPU ! Please Check.\n"));
      },
      [&](common::X86Arch) { extern_fun_name.assign("cinn_host_find_int"); },
      [&](common::ARMArch) {
        PADDLE_THROW(::common::errors::Fatal(
            "ScatterAssign only support X86 and NVGPU ! Please Check.\n"));
      },
      [&](common::NVGPUArch) { extern_fun_name.assign("cinn_cuda_find_int"); },
      [&](common::HygonDCUArchHIP) {
        extern_fun_name.assign("cinn_hip_find_int");
      },
      [&](common::HygonDCUArchSYCL) {
        extern_fun_name.assign("cinn_sycl_find_int");
      });

  auto pos_axis = axis;
  if (pos_axis < 0) pos_axis += input->shape.size();

  auto res = Compute(
      input->shape,
      [=](const std::vector<Expr>& indice) {
        // find whether indice[axis] in Index,
        // then return id if found Index[id] == indice[axis]
        // else return -1
        auto id = lang::CallExtern(extern_fun_name,
                                   {index, index->shape[0], indice[pos_axis]});

        std::vector<Expr> indice_updates = indice;
        indice_updates[pos_axis] = id;

        // check whether Index[id] == cur_index and return by check result
        return ir::Select::Make(
            ir::EQ::Make(id, Expr(-1)), input(indice), updates(indice_updates));
      },
      UniqName(output_name));
  return res;
}

ir::Tensor ScatterAdd(const ir::Tensor& input,
                      const ir::Tensor& updates,
                      const ir::Tensor& index,
                      const cinn::common::Target& target,
                      const int axis,
                      const std::string& output_name) {
  auto ScatterAddNvHygon = [&] {
    PADDLE_ENFORCE_EQ(
        index->type(),
        cinn::common::Int(32),
        ::common::errors::InvalidArgument("index's type should be int32"));
    PADDLE_ENFORCE_EQ(index->shape.size(),
                      1,
                      ::common::errors::InvalidArgument(
                          "The dimension of param [index] of IndexAdd "
                          "should be 1 ! Please Check."));
    PADDLE_ENFORCE_EQ(input->type(),
                      updates->type(),
                      ::common::errors::InvalidArgument(
                          "The data types for input and updates "
                          "should be identical ! Please Check."));

    auto pos_axis = axis;
    if (pos_axis < 0) pos_axis += input->shape.size();
    PADDLE_ENFORCE_EQ(pos_axis >= 0 && pos_axis < input->shape.size(),
                      true,
                      ::common::errors::InvalidArgument(
                          "Param [axis] of IndexAdd should satisfy 0 <= axis < "
                          "input.shape ! Please Check.\n"));

    // compute each dimension's stride, it is used for indice2offset.
    // for shape=[1,2,3,4], strides=[2*3*4,3*4,4*1,1]=[24, 12, 4, 1]
    std::vector<int> strides(updates->shape.size(), 1);
    for (int i = updates->shape.size() - 2; i >= 0; --i) {
      strides[i] = strides[i + 1] * updates->shape[i + 1].as_int32();
    }

    // compute multi-dimension index(without axis's) to scalar offset,
    // offset = offset + indice[i] * strides[i];
    auto indice2offset = [&](const std::vector<Expr>& indice) -> Expr {
      Expr offset(0);
      for (int i = 0; i < pos_axis; ++i) {
        offset = offset + indice[i] * Expr(strides[i]);
      }
      for (int i = pos_axis + 1; i < updates->shape.size(); ++i) {
        offset = offset + indice[i] * Expr(strides[i]);
      }
      return offset;
    };

    const std::string& extern_func_name =
        GetExternFuncName(target, input->type(), "index_add");

    // assume shape=[1,2,3], axis=1, `cinn_cuda_index_add` extern function do
    // following compute: out[i][j][k] = input[i][j][k] for l in
    // range(index.size()):
    //   if index[l] == j:
    //      out[i][j][k] += update[i][l][k]
    auto output = Compute(
        input->shape,
        [=](const std::vector<Expr>& indice) {
          return lang::CallExtern(extern_func_name,
                                  {input(indice),
                                   indice[pos_axis],
                                   updates,
                                   indice2offset(indice),
                                   Expr(strides[pos_axis]),
                                   index,
                                   index->shape[0]});
        },
        UniqName(output_name));

    return output;
  };

  return target.arch.Match(
      [&](std::variant<common::UnknownArch, common::X86Arch, common::ARMArch>)
          -> ir::Tensor {
        PADDLE_THROW(::common::errors::InvalidArgument(
            "Op IndexAdd only support NVGPU and "
            "HygonDCU now ! Please Check.\n"));
      },
      [&](common::NVGPUArch) { return ScatterAddNvHygon(); },
      [&](std::variant<common::HygonDCUArchHIP, common::HygonDCUArchSYCL>) {
        return ScatterAddNvHygon();
      });
}

}  // namespace pe
}  // namespace hlir
}  // namespace cinn
