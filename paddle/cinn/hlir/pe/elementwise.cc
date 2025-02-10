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

#include "paddle/cinn/hlir/pe/elementwise.h"

#include <algorithm>
#include <string>

#include "paddle/cinn/common/cas.h"
#include "paddle/cinn/common/dim_expr_converter.h"
#include "paddle/cinn/common/integer_set.h"
#include "paddle/cinn/hlir/framework/pir/trivial_op_util.h"
#include "paddle/cinn/hlir/op/op_util.h"
#include "paddle/cinn/ir/op/ir_operators.h"
#include "paddle/cinn/utils/functional.h"
#include "paddle/common/enforce.h"
namespace cinn {
namespace hlir {
namespace pe {

using ir::Expr;
using ir::Tensor;
using lang::Compute;

#define HLIR_IMP_UNARY_PE(name__)                                  \
  std::vector<ir::Tensor> name__(const Tensor& A,                  \
                                 const std::string& output_name) { \
    return {Compute(                                               \
        A->shape,                                                  \
        [=](const std::vector<Expr>& indice) {                     \
          return lang::name__(A(indice));                          \
        },                                                         \
        output_name)};                                             \
  }

#define HLIR_MKL_IMP_UNARY_PE(name__, ex_name__)                           \
  std::vector<ir::Tensor> name__##MKL(const Tensor& A,                     \
                                      const std::string& output_name) {    \
    PADDLE_ENFORCE_EQ(A->type().is_float(),                                \
                      true,                                                \
                      ::common::errors::InvalidArgument(                   \
                          "The type should be float or double. "           \
                          "Please provide a valid type."));                \
    std::string fn_name =                                                  \
        "cinn_mkl_" #ex_name__ "_v_fp" + std::to_string(A->type().bits()); \
    auto call = Compute(                                                   \
        {Expr(1)},                                                         \
        [=]() -> Expr { return lang::CallExtern(fn_name, {A}); },          \
        output_name);                                                      \
    auto out = call->TupleGet(0);                                          \
    out->WithBuffer(A->type());                                            \
    return {out, call};                                                    \
  }

HLIR_MKL_IMP_UNARY_PE(Exp, exp);
HLIR_MKL_IMP_UNARY_PE(Erf, erf);
HLIR_MKL_IMP_UNARY_PE(Sqrt, sqrt);
HLIR_MKL_IMP_UNARY_PE(Log, log);
HLIR_MKL_IMP_UNARY_PE(Log2, log2);
HLIR_MKL_IMP_UNARY_PE(Log10, log10);
HLIR_MKL_IMP_UNARY_PE(Floor, floor);
HLIR_MKL_IMP_UNARY_PE(Ceil, ceil);
HLIR_MKL_IMP_UNARY_PE(Round, round);
HLIR_MKL_IMP_UNARY_PE(Tanh, tanh);
HLIR_MKL_IMP_UNARY_PE(Trunc, trunc);
HLIR_MKL_IMP_UNARY_PE(Cos, cos);
HLIR_MKL_IMP_UNARY_PE(Sin, sin);
HLIR_MKL_IMP_UNARY_PE(Cosh, cosh);
HLIR_MKL_IMP_UNARY_PE(Tan, tan);
HLIR_MKL_IMP_UNARY_PE(Sinh, sinh);
HLIR_MKL_IMP_UNARY_PE(Acos, acos);
HLIR_MKL_IMP_UNARY_PE(Acosh, acosh);
HLIR_MKL_IMP_UNARY_PE(Asin, asin);
HLIR_MKL_IMP_UNARY_PE(Asinh, asinh);
HLIR_MKL_IMP_UNARY_PE(Atan, atan);
HLIR_MKL_IMP_UNARY_PE(Atanh, atanh);

HLIR_IMP_UNARY_PE(Exp);
HLIR_IMP_UNARY_PE(Erf);
HLIR_IMP_UNARY_PE(Sqrt);
HLIR_IMP_UNARY_PE(Log);
HLIR_IMP_UNARY_PE(Log2);
HLIR_IMP_UNARY_PE(Log10);
HLIR_IMP_UNARY_PE(Floor);
HLIR_IMP_UNARY_PE(Ceil);
HLIR_IMP_UNARY_PE(Round);
HLIR_IMP_UNARY_PE(Trunc);
HLIR_IMP_UNARY_PE(Cos);
HLIR_IMP_UNARY_PE(Cosh);
HLIR_IMP_UNARY_PE(Tan);
HLIR_IMP_UNARY_PE(Sin);
HLIR_IMP_UNARY_PE(Sinh);
HLIR_IMP_UNARY_PE(Acos);
HLIR_IMP_UNARY_PE(Acosh);
HLIR_IMP_UNARY_PE(Asin);
HLIR_IMP_UNARY_PE(Asinh);
HLIR_IMP_UNARY_PE(Atan);
HLIR_IMP_UNARY_PE(Atanh);
HLIR_IMP_UNARY_PE(IsNan);
HLIR_IMP_UNARY_PE(Tanh);
HLIR_IMP_UNARY_PE(IsFinite);
HLIR_IMP_UNARY_PE(IsInf);

HLIR_IMP_UNARY_PE(Negative);
HLIR_IMP_UNARY_PE(Identity);
HLIR_IMP_UNARY_PE(LogicalNot);
HLIR_IMP_UNARY_PE(BitwiseNot);
HLIR_IMP_UNARY_PE(Sigmoid);
HLIR_IMP_UNARY_PE(Sign);
HLIR_IMP_UNARY_PE(Abs);
HLIR_IMP_UNARY_PE(Rsqrt);
HLIR_IMP_UNARY_PE(Cbrt);
HLIR_IMP_UNARY_PE(Clz);
HLIR_IMP_UNARY_PE(Popc);

ir::Tensor Squeeze(const ir::Tensor& A,
                   const std::vector<int>& axes,
                   const std::string& output_name) {
  std::vector<int> position;
  std::vector<Expr> output_shape;
  if (axes.size()) {
    // if axis < 0, plus tensor rank.
    std::vector<int> naxes;
    for (auto axis : axes) {
      if (axis < 0) {
        axis += A->shape.size();
      }

      naxes.push_back(axis);
    }
    for (int idx = 0; idx < A->shape.size(); ++idx) {
      // if can't find idx in axis
      if (std::find(naxes.begin(), naxes.end(), idx) == naxes.end()) {
        output_shape.push_back(A->shape[idx]);
        position.push_back(idx);
      } else {
        PADDLE_ENFORCE_EQ(A->shape[idx],
                          Expr(1),
                          ::common::errors::InvalidArgument(
                              "The dimension to squeeze must be 1."));
      }
    }
  } else {
    for (int idx = 0; idx < A->shape.size(); ++idx) {
      if (A->shape[idx] != Expr(1)) {
        output_shape.push_back(A->shape[idx]);
        position.push_back(idx);
      }
    }
  }

  auto res = Compute(
      output_shape,
      [=](const std::vector<Expr>& indices) {
        std::vector<Expr> out_indices(A->shape.size(), Expr(0));
        for (int idx = 0; idx < indices.size(); ++idx) {
          out_indices[position[idx]] = indices[idx];
        }
        return A(out_indices);
      },
      output_name);
  return res;
}

ir::Tensor ExpandDims(const ir::Tensor& A,
                      const std::vector<int>& axes,
                      const std::vector<int>& out_shape,
                      const std::string& output_name) {
  const auto& posi_axes = utils::GetPositiveAxes(axes, out_shape.size());

  return Compute(
      ToCinnExprs(out_shape),
      [=](const std::vector<Expr>& indice) {
        std::vector<Expr> idx;
        int axes_pos = 0;
        for (int i = 0; i < indice.size(); ++i) {
          if (axes_pos < posi_axes.size() && posi_axes[axes_pos] == i) {
            ++axes_pos;
          } else {
            idx.push_back(indice[i]);
          }
        }
        PADDLE_ENFORCE_EQ(idx.size(),
                          A->shape.size(),
                          ::common::errors::InvalidArgument(
                              "The index size not equal with the input rank."));
        return A(idx);
      },
      UniqName(output_name));
}

Expr ReshapeHandler(const ir::Tensor& A,
                    const std::vector<Expr>& out_shape,
                    const std::vector<Expr>& indice) {
  using framework::pir::trivial_fusion_detail::ComposeUtils::ConcatVector;
  common::cas_intervals_t var_intervals =
      common::CollectVarIntervalsOfExprs(ConcatVector(A->shape, out_shape));
  common::SymbolicExprAnalyzer symbolic_expr_analyzer(var_intervals);

  std::vector<Expr> A_strides(A->shape.size());
  std::vector<Expr> B_strides(out_shape.size());
  std::vector<Expr> A_indice(A->shape.size(), Expr(0));

  const auto UpdateIndice = [&](int A_s, int A_e, int B_s, int B_e) {
    Expr offset = indice[B_s] * B_strides[B_s];
    for (int i = B_s + 1; i < B_e; i++) {
      offset = offset + indice[i] * B_strides[i];
    }
    for (int i = A_s; i < A_e; i++) {
      Expr temp = offset / A_strides[i];
      if (i > A_s) {
        temp = temp % A->shape[i];
      }
      A_indice[i] = optim::ArithSimplify(temp);
    }
  };

  Expr A_size{1}, B_size;
  int A_s, A_e = A->shape.size();
  int B_s, B_e = out_shape.size();

  for (A_s = A_e - 1; A_s >= 0; A_s--) {
    A_strides[A_s] = A_size;
    A_size = A_size * A->shape[A_s];
    B_size = Expr(1);
    for (B_s = B_e - 1; B_s >= 0; B_s--) {
      B_strides[B_s] = B_size;
      B_size = B_size * out_shape[B_s];
      std::optional<bool> eq = symbolic_expr_analyzer.ProveEQ(A_size, B_size);
      if (eq.value_or(false)) {
        UpdateIndice(A_s, A_e, B_s, B_e);
        A_e = A_s;
        B_e = B_s;
        A_size = Expr(1);
        break;
      }
    }
  }

  if (A_e > 0 && B_e > 0) {
    UpdateIndice(0, A_e, 0, B_e);
  }

  return A(A_indice);
}

ir::Tensor Reshape(const ir::Tensor& A,
                   const std::vector<int>& new_shape,
                   const std::string& name) {
  std::vector<Expr> new_expr_shape;
  for (int i : new_shape) {
    new_expr_shape.push_back(Expr(i));
  }
  auto res = Compute(
      new_expr_shape,
      [=](const std::vector<Expr>& indice) {
        return ReshapeHandler(A, new_expr_shape, indice);
      },
      name);
  return res;
}

ir::Tensor Reshape(const ir::Tensor& A,
                   const std::vector<ir::Dim>& new_shape,
                   const std::string& name) {
  std::vector<Expr> new_expr_shape;
  for (auto& i : new_shape) {
    new_expr_shape.push_back(i->dim_expr);
  }
  auto res = Compute(
      new_expr_shape,
      [=](const std::vector<Expr>& indice) {
        return ReshapeHandler(A, new_expr_shape, indice);
      },
      name);
  return res;
}

ir::Tensor Cast(const ir::Tensor& A,
                const Type& dtype,
                const std::string& name) {
  auto res = Compute(
      A->shape,
      [=](const std::vector<Expr>& indices) {
        return ir::Cast::Make(dtype, A(indices));
      },
      name);
  return res;
}

ir::Tensor Store(const ir::Tensor& A, const std::string& name) {
  auto res = Compute(
      A->shape,
      [=](const std::vector<Expr>& indices) { return A(indices); },
      name);
  return res;
}

ir::Tensor Arange(const float start,
                  const float stop,
                  const float step,
                  const Type& dtype,
                  const std::string& output_name) {
  int num = static_cast<int>(std::ceil((stop - start) / step));
  ir::Tensor res = lang::Compute(
      {Expr(num)},
      [=](const std::vector<ir::Expr>& indices) {
        return ir::Cast::Make(
            dtype,
            Expr(start) +
                Expr(step) * ir::Cast::Make(cinn::common::F32(), indices[0]));
      },
      output_name);
  return res;
}

ir::Tensor Tril(const ir::Tensor& A,
                const int diagonal,
                const std::vector<ir::Dim>& out_shape,
                const std::string& name) {
  ir::Tensor res = Compute(
      ToCinnExprs(out_shape),
      [=](const std::vector<Expr>& indice) {
        PADDLE_ENFORCE_GE(indice.size(),
                          size_t(2),
                          ::common::errors::InvalidArgument(
                              "The Tril op input tensor must have a rank "
                              "greater than or equal to 2."));
        std::vector<Expr> new_indice(indice.end() - 2, indice.end());
        Expr col_indice = indice.back();
        return ir::Select::Make(new_indice[0] >= new_indice[1] - diagonal,
                                A(indice),
                                ir::Zero(A->type()));
      },
      name);
  return res;
}

ir::Tensor GenerateShape(const std::vector<ir::Tensor>& inputs,
                         const cinn::dialect::SymbolBindings& symbol_bindings,
                         const std::vector<symbol::DimExpr>& output_dim_exprs,
                         const std::vector<ir::Dim>& out_shape,
                         const std::vector<Type>& out_type,
                         const std::string& name) {
  if (output_dim_exprs.size() != 1) {
    VLOG(4) << "pe::GenerateShape will return a meaningless tensor when "
               "output_dim_exprs.size() != 1";
    return Compute(
        {Expr(1l)},
        [=](const std::vector<Expr>& indice) { return Expr(1l); },
        name);
  }
  cinn::common::DimExprConverterWithSymbolBindings converter(inputs,
                                                             symbol_bindings);
  auto res = Compute(
      ToCinnExprs(out_shape),
      [=, &converter](const std::vector<Expr>& indice) {
        auto dim_expr = converter.ConvertToIrExpr(output_dim_exprs[0]);

        if (out_type[0] == type_of<int32_t>()) {
          dim_expr = ir::Cast::Make(type_of<int32_t>(), dim_expr);
        }

        return dim_expr;
      },
      name);
  return res;
}

ir::Tensor IsClose(const ir::Tensor& x,
                   const ir::Tensor& y,
                   int axis,
                   float rtol,
                   float atol,
                   bool equal_nan,
                   const std::string& out_name) {
  // [To do] axis is not used in the op.
  // For each a=x[i], b=y[i]:
  // ```
  // if (isnan(a) || isnan(b)) {
  //   out = equal_nan && isnan(a) == isnan(b);
  // } else {
  //   T left = (a > b ? a - b : b - a);
  //   T right = atol + (b > 0 ? rtol * b : (-rtol) * b);
  //   T diff = (left > right ? left - right : right - left);
  //   out = a == b || left <= right || diff <= 1e-15;
  // }
  // ```
  auto fnop = [&](const Expr& a, const Expr& b) {
    // check whether x or y is nan
    auto check_x_nan = lang::IsNan(a);
    auto check_y_nan = lang::IsNan(b);

    // out = equal_nan && isnan(a) == isnan(b);
    auto check_nan_same =
        Expr(equal_nan) && ir::EQ::Make(check_x_nan, check_y_nan);

    // check whether x and y are close
    // T left = (a > b ? a - b : b - a);
    auto left = ir::Select::Make(a > b, a - b, b - a);
    // T right = atol + (b > 0 ? rtol * b : (-rtol) * b);
    auto right = ir::Cast::Make(x->type(), atol) +
                 ir::Select::Make(b > ir::Zero(b->type()),
                                  ir::Cast::Make(x->type(), rtol) * b,
                                  ir::Cast::Make(x->type(), -rtol) * b);
    // T diff = (left > right ? left - right : right - left);
    auto diff = ir::Select::Make(left > right, left - right, right - left);
    // out = a == b || left <= right || diff <= 1e-15;
    auto check_diff = (ir::EQ::Make(a, b) || (left <= right)) ||
                      (diff <= lang::Epsilon(diff->type()));

    return ir::Select::Make(
        check_x_nan || check_y_nan, check_nan_same, check_diff);
  };
  auto fn = [=](const std::vector<Expr>& indice) {
    PADDLE_ENFORCE_EQ(
        indice.size(),
        y->shape.size(),
        ::common::errors::InvalidArgument(
            "The indice size should be equal to y's shape size."));
    return fnop(x(indice), y(indice));
  };
  auto res = Compute(x->shape, fn, out_name);
  return res;
}

}  // namespace pe
}  // namespace hlir
}  // namespace cinn
