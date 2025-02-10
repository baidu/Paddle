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

#include "paddle/cinn/common/cas.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>

#include "paddle/cinn/common/arithmetic.h"
#include "paddle/cinn/common/ir_util.h"
#include "paddle/cinn/ir/ir_mutator.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/ir_visitor.h"
#include "paddle/cinn/ir/op/ir_operators.h"
#include "paddle/cinn/ir/utils/ir_copy.h"
#include "paddle/cinn/ir/utils/ir_nodes_collector.h"
#include "paddle/cinn/utils/string.h"
#include "paddle/common/enforce.h"
namespace cinn {
namespace common {
using namespace ir;  // NOLINT

Expr AutoSimplify(
    const Expr& u,
    const absl::flat_hash_map<std::string, CasInterval>& var_intervals) {
  VLOG(7) << "Begin AutoSimplify: " << u;
  Expr copied = ir::ir_utils::IRCopy(u);
  if (copied.type().is_float()) {
    return copied;
  }
  copied = detail::ConvertCinnToCAS(copied);
  absl::flat_hash_map<std::string, CasInterval> s_var_intervals;
  for (auto& item : var_intervals) {
    if (item.second.e_l.defined() && item.second.e_r.defined()) {
      Expr e_l = detail::ConvertCinnToCAS(item.second.e_l);
      Expr e_r = detail::ConvertCinnToCAS(item.second.e_r);
      s_var_intervals.emplace(item.first, CasInterval(e_l, e_r));
    } else {
      s_var_intervals.emplace(item.first,
                              CasInterval(item.second.l, item.second.r));
    }
  }
  copied = CasSimplify(copied, s_var_intervals);
  copied = detail::ConvertCasToCinn(copied);
  VLOG(7) << "End AutoSimplify " << copied;
  return copied;
}

int gcd(int a, int b) {
  // Everything divides 0
  if (a == 0) return b;
  if (b == 0) return a;
  if (a == 1 || b == 1) return 1;
  if (a < 0 || b < 0) {
    return gcd(std::abs(a), std::abs(b));
  }

  // base case
  if (a == b) return a;

  // a is greater
  if (a > b) return gcd(a - b, b);
  return gcd(a, b - a);
}

//////// All the following symbolic computation methods are implemented
/// referencing to the book <Computer Algebra and
/// Symbolic Computation - Joel S. Cohen>

template <typename T>
std::vector<T> EraseFront(const std::vector<T>& vs) {
  return std::vector<T>(vs.begin() + 1, vs.end());
}

template <typename T>
std::vector<T> Concat(const std::vector<T>& as, const std::vector<T>& bs) {
  auto res = as;
  res.insert(std::end(res), bs.begin(), bs.end());
  return res;
}

// 3*x*2*y => 3*2
// x => 1
Expr ProductGetConstantPart(Expr u) {
  auto* product = u.As<Product>();
  if (product) {
    std::vector<Expr> constant_operands;
    for (auto& i : product->operands()) {
      if (i.is_constant()) {
        constant_operands.push_back(i);
      }
    }
    if (constant_operands.empty())
      return make_const(u->type(), 1);
    else if (constant_operands.size() == 1)
      return constant_operands.front();
    else
      return Product::Make(constant_operands);
  }
  return make_const(u->type(), 1);
}

// 3*x*2*y => x*y
// x => x
Expr ProductGetNonConstantPart(Expr u) {
  auto* product = u.As<Product>();
  if (product) {
    std::vector<Expr> nonconstant_operands;
    for (auto& i : product->operands()) {
      if (!i.is_constant()) {
        nonconstant_operands.push_back(i);
      }
    }
    if (nonconstant_operands.empty()) {
      return make_const(u->type(), 1);
    } else if (nonconstant_operands.size() == 1) {
      return nonconstant_operands.front();
    } else {
      return Product::Make(nonconstant_operands);
    }
  }
  return u;
}

namespace detail {

// Is a Divisible to b.
// @{
bool IsDivisible(int64_t a, int64_t b) {
  PADDLE_ENFORCE_NE(
      b,
      0,
      ::common::errors::InvalidArgument("The divisor %d should not be 0.", b));
  return a % b == 0;
}
bool IsDivisible(const Sum* a, int b);

// If int a Divisible to any operands of product b
bool IsDivisible(int a, const Product* b) {
  if (a < 0) return false;
  for (auto& item : b->operands()) {
    if (item.As<IntImm>() && item.As<IntImm>()->value > 0 &&
        IsDivisible(a, item.As<IntImm>()->value))
      return true;
  }
  return false;
}
bool IsDivisible(const Product* a, int b) {
  for (auto& item : a->operands()) {
    if (item.As<IntImm>() && IsDivisible(item.As<IntImm>()->value, b)) {
      return true;
    }
    if (item.As<Sum>() && IsDivisible(item.As<Sum>(), b)) return true;
  }
  return false;
}
bool IsDivisible(const Sum* a, int b) {
  for (auto& item : a->operands()) {
    auto* vi = item.As<IntImm>();
    auto* vp = item.As<Product>();
    if (vi && IsDivisible(vi->value, b)) continue;
    if (vp && IsDivisible(vp, b)) continue;
    return false;
  }
  return true;
}
bool IsDivisible(Expr a, int b) {
  auto* ai = a.As<IntImm>();
  auto* as = a.As<Sum>();
  auto* ap = a.As<Product>();

  if (ai) return IsDivisible(ai->value, b);
  if (as) return IsDivisible(as, b);
  if (ap) return IsDivisible(ap, b);
  return false;
}
// @}

//! Divide a by b, NOTE that a should be divisible by b.
// @{
Expr Divide(const Product* a, int b);
Expr Divide(const Sum* a, int b) {
  std::vector<Expr> args;
  for (auto& item : a->operands()) {
    if (item.As<IntImm>())
      args.push_back(make_const(item.type(), item.As<IntImm>()->value / b));
    else if (item.As<Product>())
      args.push_back(Divide(item.As<Product>(), b));
    else
      CINN_NOT_IMPLEMENTED
  }
  return Sum::Make(args);
}
Expr Divide(const Product* a, int b) {
  std::vector<Expr> args;
  int i = 0;
  int times = -1;
  bool is_divisible = false;
  for (i = 0; i < a->operands().size(); i++) {
    auto* a_i = a->operand(i).As<IntImm>();
    if (a_i && a_i->value % b == 0) {
      times = a_i->value / b;
      is_divisible = true;
      break;
    }
  }
  // Case is_divisible : a = 8x and b = 4 and a/b = 2x
  // Case !is_divisible : a = 2x and b = 8 and a/b = x/4
  if (is_divisible) {
    // NOTE that a should be divisible by b.
    if (times != 1) {
      args.push_back(make_const(a->type(), times));
    }
    for (int j = 0; j < a->operands().size(); j++) {
      if (j == i) continue;
      args.push_back(a->operand(j));
    }
    return Product::Make(args);
  } else {
    for (i = 0; i < a->operands().size(); i++) {
      auto* a_i = a->operand(i).As<IntImm>();
      if (a_i && b % a_i->value == 0) {
        b = b / a_i->value;
      } else {
        args.push_back(a->operand(i));
      }
    }
    return FracOp::Make(Product::Make(args), Expr(b));
  }
  return Product::Make(args);
}

// @}

inline int Iquot(int n, int d) { return n / d; }

inline int Irem(int n, int d) {
  int k = Iquot(n, d);
  return n - d * k;
}

Expr CasSimplifyMutator::SimplifyRationalNumber(Expr u) {
  auto* frac_n = u.As<FracOp>();
  if (frac_n) {
    Expr n = frac_n->a();
    Expr d = frac_n->b();

    auto* ni = n.As<IntImm>();
    auto* di = d.As<IntImm>();

    PADDLE_ENFORCE_EQ(
        ni && di,
        true,
        ::common::errors::InvalidArgument("Ni and Di should not be null."));
    int nv = ni->value;
    int dv = di->value;

    if (Irem(nv, dv) == 0) {
      return Expr(make_const(u.type(), Iquot(nv, dv)));
    } else {
      int g = gcd(nv, dv);
      if (dv > 0) {
        return FracOp::Make(make_const(Iquot(nv, g)), make_const(Iquot(dv, g)));
      } else {
        return FracOp::Make(make_const(Iquot(-nv, g)),
                            make_const(Iquot(-dv, g)));
      }
    }
  }
  return u;
}

Expr SumOrProductGetSingleElementsRec(Expr u) {
  auto* product = u.As<Product>();
  auto* sum = u.As<Sum>();
  if (product && product->operands().size() == 1) {
    return SumOrProductGetSingleElementsRec(u->operands.front());
  }
  if (sum && sum->operands().size() == 1) {
    return SumOrProductGetSingleElementsRec(u->operands.front());
  }
  return u;
}

// Order, reference to Page 85.
bool ExprPosCmp::operator()(const Expr& a, const Expr& b) {
  // O-1, 1 <| 2
  VLOG(7) << "Begin ExprPosCmp, a: " << a << ", b: " << b;
  if (a.is_constant() && b.is_constant()) {
    return a.get_constant() < b.get_constant();
  }

  // O-2, both are symbols, compare by the lexicographical order.
  if (a.As<_Var_>() && b.As<_Var_>()) {
    return a.As<_Var_>()->name < b.As<_Var_>()->name;
  }

  // O-3, if a and b are either both products or both sums, compare by each
  // element similar to lexicographical order.
  if ((a.As<Product>() && b.As<Product>()) || (a.As<Add>() && b.As<Add>())) {
    auto& aoprs = a->operands;
    auto& boprs = b->operands;
    int m = std::min(aoprs.size(), boprs.size());

    for (int i = 0; i < m; i++) {
      // ugly compare representation in string.
      auto& aopr = aoprs[aoprs.size() - 1 - i];
      auto& bopr = boprs[boprs.size() - 1 - i];
      if (aopr != bopr) return operator()(aopr, bopr);
    }

    return aoprs.size() < boprs.size();
  }

  // customized case, if both are mod
  {
    auto* am = a.As<Mod>();
    auto* bm = b.As<Mod>();
    if (am && bm) {
      if (am->b() != bm->b()) {
        return operator()(am->b(), bm->b());
      }
      return operator()(am->a(), bm->a());
    }
  }

  // O-7, if a is an integer or fraction and v is any other type, 1 < x
  if (a.As<IntImm>() || a.As<FloatImm>() || a.As<FracOp>()) {
    if (!(b.As<IntImm>() || b.As<FloatImm>() || b.As<FracOp>())) return true;
  }
  if (b.As<IntImm>() || b.As<FloatImm>() || b.As<FracOp>()) {
    if (!(a.As<IntImm>() || a.As<FloatImm>() || a.As<FracOp>())) return false;
  }

  // O-8, if a is a product, v is a sum, fractional, or symbol
  {
    auto* ap = a.As<Product>();

    if (ap && (b.As<Sum>() || b.As<Call>() || b.As<_Var_>() || b.As<Mod>())) {
      return operator()(a, Product::Make({b}));
    }
  }

  {
    if (a.As<Mod>()) {
      if (!b.As<Mod>()) {
        // Todo: may be wrong especially for negative value
        return operator()(a, Mod::Make(b, Sum::Make({b, Expr(1)})));
      }
    }
  }

  // O-10, if a is a sum, b is a function, or symbol
  {
    if (a.As<Sum>()) {
      if (b.As<_Var_>()) {
        return operator()(a.As<Sum>()->operand(0), {b});
      }
    }
  }

  return false;
}

std::vector<Expr> CasSimplifyMutator::MergeProduct(const std::vector<Expr>& p,
                                                   const std::vector<Expr>& q) {
  return MergeExprs(p,
                    q,
                    std::bind(&CasSimplifyMutator::SimplifyBinaryProduct,
                              this,
                              std::placeholders::_1,
                              std::placeholders::_2));
}

std::vector<Expr> CasSimplifyMutator::SimplifyBinaryProduct(Expr left,
                                                            Expr right) {
  // SPRDREC-1
  if (!left.As<Product>() && !right.As<Product>()) {
    auto a = left;
    auto b = right;

    auto* ai = a.As<IntImm>();
    auto* af = a.As<FloatImm>();
    auto* bi = b.As<IntImm>();
    auto* bf = b.As<FloatImm>();

    // case 1, both are constants
    if (a.is_constant() && b.is_constant()) {
      if (ai) return {make_const(a.type(), ai->value * bi->value)};
      if (af) return {make_const(a.type(), af->value * bf->value)};
    }

    if (a.As<Max>() || a.As<Min>() || b.As<Max>() || b.As<Min>()) {
      // cinn_min/cinn_max(a, b) * 2 = cinn_min/cinn_max(2*a, 2*b)
      // 2 * cinn_min/cinn_max(a, b) = cinn_min/cinn_max(2*a, 2*b)
      // cinn_min/cinn_max(a, b) * -2 = cinn_max/cinn_min(-2*b, -2*a)
      // -2 * cinn_min/cinn_max(a, b) = cinn_max/cinn_min(-2*b, -2*a)
      Expr const_oper;
      Expr cmp_oper;
      int const_value;
      if (ai) {
        const_oper = a;
        cmp_oper = b;
        const_value = ai->value;
      }
      if (af) {
        const_oper = a;
        cmp_oper = b;
        const_value = af->value;
      }
      if (bi) {
        const_oper = b;
        cmp_oper = a;
        const_value = bi->value;
      }
      if (bf) {
        const_oper = b;
        cmp_oper = a;
        const_value = bf->value;
      }
      if (const_value == 0) {
        return {make_const(a->type(), 0)};
      }
      if (cmp_oper.defined() && const_oper.defined()) {
        auto cmp_min = cmp_oper.As<Min>();
        auto cmp_max = cmp_oper.As<Max>();
        if (const_value > 0) {
          if (cmp_min) {
            return {CasSimplify(
                Min::Make(CasSimplify(Product::Make({cmp_min->a(), const_oper}),
                                      var_intervals),
                          CasSimplify(Product::Make({cmp_min->b(), const_oper}),
                                      var_intervals)),
                var_intervals)};
          }
          if (cmp_max) {
            return {CasSimplify(
                Max::Make(CasSimplify(Product::Make({cmp_max->a(), const_oper}),
                                      var_intervals),
                          CasSimplify(Product::Make({cmp_max->b(), const_oper}),
                                      var_intervals)),
                var_intervals)};
          }
        } else {
          if (cmp_min) {
            return {CasSimplify(
                Max::Make(CasSimplify(Product::Make({cmp_min->b(), const_oper}),
                                      var_intervals),
                          CasSimplify(Product::Make({cmp_min->a(), const_oper}),
                                      var_intervals)),
                var_intervals)};
          }
          if (cmp_max) {
            return {CasSimplify(
                Min::Make(CasSimplify(Product::Make({cmp_max->b(), const_oper}),
                                      var_intervals),
                          CasSimplify(Product::Make({cmp_max->a(), const_oper}),
                                      var_intervals)),
                var_intervals)};
          }
        }
      }
    }

    {  // FracOp related constants.
      // NOTE the integer division is weried in C language, 1/2 = 0, that is
      // huge different from a real CAS.
      auto* af = a.As<FracOp>();
      auto* bf = b.As<FracOp>();
      // 1/2 * 2/3
      if (af && bf && a->type().is_float()) {
        return {CasSimplify(FracOp::Make(Product::Make({af->a(), bf->a()}),
                                         Product::Make({af->b(), bf->b()})),
                            var_intervals)};
      }
      if (af && !bf && a->type().is_float()) {
        return {CasSimplify(FracOp::Make(Product::Make({af->a(), b}), af->b()),
                            var_intervals)};
      }
      if (!af && bf && a->type().is_float()) {
        return {CasSimplify(FracOp::Make(Product::Make({bf->a(), a}), bf->b()),
                            var_intervals)};
      }
    }

    // case 2
    // x*1 -> a
    if (ai && ai->value == 1) return {b};
    if (af && af->value == 1.f) return {b};
    // 1*x -> x
    if (bi && bi->value == 1) return {a};
    if (bf && bf->value == 1.f) return {a};

    {
      auto* a_sum = a.As<Sum>();
      auto* b_sum = b.As<Sum>();

      if (b_sum) {
        std::vector<Expr> args;
        for (auto& v : b_sum->operands()) {
          args.push_back(CasSimplify(Product::Make({a, v}), var_intervals));
        }
        return {SimplifySum(Sum::Make(args))};
      }

      if (a_sum) {
        std::vector<Expr> args;
        for (auto& v : a_sum->operands()) {
          args.push_back(CasSimplify(Product::Make({b, v}), var_intervals));
        }
        return {SimplifySum(Sum::Make(args))};
      }
    }

    // case 4, b <| a
    {
      if (ExprPosCmp()(b, a)) {
        return {b, a};
      }
    }

    return {left, right};
  }

  // SPRDREC-2, Page 101
  if (left.As<Product>() || right.As<Product>()) {
    auto a = left;
    auto b = right;

    auto* a_product = a.As<Product>();
    auto* b_product = b.As<Product>();
    // case 1
    if (a_product && b_product) {
      return MergeProduct(a_product->operands(), b_product->operands());
    }

    // case 2
    if (a_product) {
      return MergeProduct(a_product->operands(), {b});
    }

    // case 3
    if (b_product) {
      return MergeProduct({a}, b_product->operands());
    }
  }

  return {left, right};
}

std::vector<Expr> CasSimplifyMutator::SimplifyProductRec(
    const std::vector<Expr>& operands) {
  if (operands.size() < 2)
    return {CasSimplify(operands.front(), var_intervals)};
  auto mid_it = operands.begin() + operands.size() / 2;
  auto&& left = SimplifyProductRec(std::vector<Expr>(operands.begin(), mid_it));
  auto&& right = SimplifyProductRec(std::vector<Expr>(mid_it, operands.end()));
  return MergeProduct(left, right);
}

Expr CasSimplifyMutator::SimplifyProduct(Expr a) {
  a = SumOrProductGetSingleElementsRec(a);
  // We reuse the Mul node for production.
  auto* prod = a.As<Product>();
  if (!prod) return a;

  const auto& _operands = prod->operands();
  std::vector<Expr> operands;
  for (auto& e : _operands) operands.push_back(CasSimplify(e, var_intervals));
#ifdef CINN_DEBUG
  {
    std::stringstream ss;
    for (auto& v : operands) {
      ss << v << " ";
    }
    VLOG(7) << "operands: " << ss.str();
  };
#endif

  // SPRD-2
  // 0*x... = 0
  for (auto& opr : operands) {
    auto* opri = opr.As<IntImm>();
    auto* oprf = opr.As<FloatImm>();
    if (opri && opri->value == 0) return make_const(a.type(), 0);
    if (oprf && oprf->value == 0) return make_const(a.type(), 0);
  }

  // SPRD-3
  // prod(x) = x, single number.
  if (operands.size() == 1) {
    auto* first_s = operands.front().As<Sum>();
    auto* first_p = operands.front().As<Product>();
    return operands[0];
  }

  // SPRD-4
  return Product::Make(SimplifyProductRec(operands));
}

Expr CasSimplifyMutator::SimplifySum(Expr u) {
  u = SumOrProductGetSingleElementsRec(u);

  auto* sum = u.As<Sum>();
  PADDLE_ENFORCE_NOT_NULL(
      sum, ::common::errors::InvalidArgument("Sum should not be null."));

  auto& operands = sum->operands();

  auto temp = SimplifySpecificSum(u);
  // If temp has been simplified, return it.
  if (!temp.As<Sum>()) return temp;

  operands = temp.As<Sum>()->operands();

  auto args = SimplifySumRec(operands);
  if (args.empty()) return make_const(u.type(), 0);
  if (args.size() == 1) return args[0];
  return Sum::Make(args);
}

std::vector<Expr> CasSimplifyMutator::MergeExprs(
    const std::vector<Expr>& p,
    const std::vector<Expr>& q,
    const std::function<std::vector<Expr>(Expr, Expr)>& binary_merge) {
  std::vector<Expr> res;
  int li = 0, lj = 0;
  while (li < p.size() && lj < q.size()) {
    auto&& p1 = p[li];
    auto&& q1 = q[lj];
    auto&& h = binary_merge(p1, q1);
    if (h.size() == 2 && h[0] == p1 && h[1] == q1) {
      ++li;
      res.emplace_back(std::move(h.front()));
    } else if (h.size() == 2 && h[0] == q1 && h[1] == p1) {
      ++lj;
      res.emplace_back(std::move(h.front()));
    } else {
      ++li;
      ++lj;
      std::move(h.begin(), h.end(), std::back_inserter(res));
    }
  }

  if (li < p.size()) res.insert(res.end(), p.begin() + li, p.end());
  if (lj < q.size()) res.insert(res.end(), q.begin() + lj, q.end());
  return std::move(res);
}

// This implementation is similar to MergeProduct
std::vector<Expr> CasSimplifyMutator::MergeSum(const std::vector<Expr>& p,
                                               const std::vector<Expr>& q) {
#ifdef CINN_DEBUG
  {
    std::stringstream ss;
    for (auto& x : p) ss << x << " ";

    VLOG(7) << "MergeSum p(" << ss.str() << ")";
    ss.str("");

    for (auto& x : q) ss << x << " ";
    VLOG(7) << "MergeSum q(" << ss.str() << ")";
    ss.str("");
  }
#endif

  return MergeExprs(p, q, [this](Expr left, Expr right) -> std::vector<Expr> {
    auto&& h = SimplifyBinarySum(std::move(left), std::move(right));
    if (h.size() == 1 && h[0].is_constant() && h[0].get_constant() == 0) {
      return {};
    } else {
      return std::move(h);
    }
  });
}

std::vector<Expr> CasSimplifyMutator::SimplifyBinarySum(Expr left, Expr right) {
  // SPRDREC-1
  if (!left.As<Sum>() && !right.As<Sum>()) {
    auto a = left;
    auto b = right;
    // clang-format off
    auto* ai = a.As<IntImm>();
    auto* au = a.As<UIntImm>();
    auto* af = a.As<FloatImm>();
    auto* bi = b.As<IntImm>();
    auto* bu = b.As<UIntImm>();
    auto* bf = b.As<FloatImm>();

    // case 1, both are constants
    if (a.is_constant() && b.is_constant()) {
      if (ai) return {make_const(a.type(), ai->value + bi->value)};
      if (af) return {make_const(a.type(), af->value + bf->value)};
      if (au) return {make_const(a.type(), au->value + bu->value)};
    }
    // clang-format on

    // cinn_min/cinn_max(a, b)+c = cinn_min/cinn_max(a+c, b+c)
    // c + cinn_min/cinn_max(a, b) = cinn_min/cinn_max(a+c, b+c)
    auto* a_min = a.As<Min>();
    auto* a_max = a.As<Max>();
    auto* b_min = b.As<Min>();
    auto* b_max = b.As<Max>();
    if (a_min) {
      return {CasSimplify(
          Min::Make(CasSimplify(Sum::Make({a_min->a(), b}), var_intervals),
                    CasSimplify(Sum::Make({a_min->b(), b}), var_intervals)),
          var_intervals)};
    }
    if (a_max) {
      return {CasSimplify(
          Max::Make(CasSimplify(Sum::Make({a_max->a(), b}), var_intervals),
                    CasSimplify(Sum::Make({a_max->b(), b}), var_intervals)),
          var_intervals)};
    }
    if (b_min) {
      return {CasSimplify(
          Min::Make(CasSimplify(Sum::Make({b_min->a(), a}), var_intervals),
                    CasSimplify(Sum::Make({b_min->b(), a}), var_intervals)),
          var_intervals)};
    }
    if (b_max) {
      return {CasSimplify(
          Max::Make(CasSimplify(Sum::Make({b_max->a(), a}), var_intervals),
                    CasSimplify(Sum::Make({b_max->b(), a}), var_intervals)),
          var_intervals)};
    }

    // case 2
    // x*1 -> a
    if (ai && ai->value == 0) return {b};
    if (af && af->value == 0.f) return {b};
    // 1*x -> x
    if (bi && bi->value == 0) return {a};
    if (bf && bf->value == 0.f) return {a};

    // customized case for Mod
    {
      auto* am = a.As<Mod>();
      auto* bm = b.As<Mod>();
      if (am && bm) {
        if (am->b() == bm->b() && ProductGetNonConstantPart(am->a()) ==
                                      ProductGetNonConstantPart(bm->a())) {
          return {CasSimplify(Mod::Make(Sum::Make({am->a(), bm->a()}), am->b()),
                              var_intervals)};
        }
      }
    }

    // case 3
    // Here is different from SimplifySumRec, to deal with cases like 3x + (-2x)
    // = 2x
    auto a_non_constant = ProductGetNonConstantPart(a);
    auto b_non_constant = ProductGetNonConstantPart(b);
    if (a_non_constant.defined() && b_non_constant.defined() &&
        a_non_constant == b_non_constant) {
      VLOG(7) << "a " << a;
      VLOG(7) << "b " << b;
      Expr s = SimplifySum(
          Sum::Make({ProductGetConstantPart(a), ProductGetConstantPart(b)}));
      Expr p = Product::Make({s, ProductGetNonConstantPart(a)});
      return {CasSimplify(p, var_intervals)};
    }

    // case 4, b <| a
    {
      if (ExprPosCmp()(b, a)) {
        return {b, a};
      }
    }

    return {left, right};
  }

  // SPRDREC-2, Page 101
  if (left.As<Sum>() || right.As<Sum>()) {
    auto a = left;
    auto b = right;

    auto* a_sum = a.As<Sum>();
    auto* b_sum = b.As<Sum>();

    // case 1
    if (a_sum && b_sum) {
      return MergeSum(a_sum->operands(), b_sum->operands());
    }

    // case 2
    if (a_sum) {
      return MergeSum(a_sum->operands(), {b});
    }

    // case 3
    if (b_sum) {
      return MergeSum({a}, b_sum->operands());
    }
  }

  return {left, right};
}

// The implementation is similar to SimplifyProductRec
std::vector<Expr> CasSimplifyMutator::SimplifySumRec(
    const std::vector<Expr>& operands) {
#ifdef CINN_DEBUG
  {
    std::stringstream ss;
    for (auto& o : operands) {
      ss << o.node_type() << " " << o << " ";
    }
    VLOG(7) << "SimplifySumRec operands: " << ss.str();
  }
#endif
  PADDLE_ENFORCE_EQ(
      !operands.empty(),
      true,
      ::common::errors::InvalidArgument("Operands should not be empty."));
  if (operands.size() < 2)
    return {CasSimplify(operands.front(), var_intervals)};
  auto mid_it = operands.begin() + operands.size() / 2;
  auto&& left = SimplifySumRec(std::vector<Expr>(operands.begin(), mid_it));
  auto&& right = SimplifySumRec(std::vector<Expr>(mid_it, operands.end()));
  return MergeSum(left, right);
}

void CasSimplifyMutator::AddBaseAndSimplify(Expr* base, Expr bound) {
  if ((*base).defined()) {
    *base = Sum::Make({*base, bound});
  } else {
    *base = bound;
  }
  *base = CasSimplify(*base, var_intervals);
}

void CasSimplifyMutator::UnfoldBound(Expr* lower_bound,
                                     Expr* upper_bound,
                                     Expr var,
                                     bool unfold_const_bound) {
  PADDLE_ENFORCE_NOT_NULL(
      lower_bound,
      ::common::errors::InvalidArgument("Lower bound should not be null."));
  PADDLE_ENFORCE_NOT_NULL(
      upper_bound,
      ::common::errors::InvalidArgument("Upper bound should not be null."));
  auto v_var = var.As<_Var_>();
  PADDLE_ENFORCE_NOT_NULL(
      v_var, ::common::errors::InvalidArgument("Var should not be null."));
  if (var_intervals.count(v_var->name)) {
    auto& interval = var_intervals.at(v_var->name);
    if (interval.e_l.defined() && interval.e_r.defined()) {
      AddBaseAndSimplify(lower_bound, interval.e_l);
      AddBaseAndSimplify(upper_bound, interval.e_r);
    } else if (unfold_const_bound) {
      // unfold var's const bound
      AddBaseAndSimplify(lower_bound, Expr(interval.l));
      AddBaseAndSimplify(upper_bound, Expr(interval.r));
    } else {
      // no unfold var's const bound for var simplification
      AddBaseAndSimplify(lower_bound, var);
      AddBaseAndSimplify(upper_bound, var);
    }
  } else if (!unfold_const_bound) {
    // not get var's bound for var simplification
    AddBaseAndSimplify(lower_bound, var);
    AddBaseAndSimplify(upper_bound, var);
  } else {
    PADDLE_THROW(::common::errors::InvalidArgument("can't get the bound"));
  }
}

bool CasSimplifyMutator::GetVarBound(Expr* lower_bound,
                                     Expr* upper_bound,
                                     Expr var,
                                     bool unfold_const_bound) {
  PADDLE_ENFORCE_NOT_NULL(
      lower_bound,
      ::common::errors::InvalidArgument("Lower bound should not be null."));
  PADDLE_ENFORCE_NOT_NULL(
      upper_bound,
      ::common::errors::InvalidArgument("Upper bound should not be null."));
  auto v_var = var.As<_Var_>();
  auto v_product = var.As<Product>();
  auto v_frac = var.As<FracOp>();
  if (v_var && (var_intervals.count(v_var->name) || !unfold_const_bound)) {
    UnfoldBound(lower_bound, upper_bound, var, unfold_const_bound);
    return true;
  } else if (v_product) {
    // only deal with 2*x
    Expr p_lower_bound;
    Expr p_upper_bound;
    Expr const_oper = ProductGetConstantPart(var);
    Expr non_const_oper = ProductGetNonConstantPart(var);
    auto v_var = non_const_oper.As<_Var_>();
    if (v_var && var_intervals.count(v_var->name)) {
      Expr v_lower, v_upper;
      UnfoldBound(&v_lower, &v_upper, non_const_oper, unfold_const_bound);
      auto const_v = const_oper.get_constant();
      PADDLE_ENFORCE_EQ(v_lower.defined() && v_upper.defined(),
                        true,
                        ::common::errors::InvalidArgument(
                            "V lower and upper should be defined."));
      if (const_v > 0) {
        p_lower_bound = Product::Make({const_oper, v_lower});
        p_upper_bound = Product::Make({const_oper, v_upper});
      } else {
        p_lower_bound = Product::Make({const_oper, v_upper});
        p_upper_bound = Product::Make({const_oper, v_lower});
      }
      AddBaseAndSimplify(lower_bound, p_lower_bound);
      AddBaseAndSimplify(upper_bound, p_upper_bound);
      return true;
    }
  } else if (v_frac) {
    // only deal with x/2
    Expr p_lower_bound;
    Expr p_upper_bound;
    Expr non_const_oper = v_frac->a();
    Expr const_oper = v_frac->b();
    auto v_var = non_const_oper.As<_Var_>();
    if (v_var && var_intervals.count(v_var->name)) {
      Expr v_lower, v_upper;
      UnfoldBound(&v_lower, &v_upper, non_const_oper, unfold_const_bound);
      auto const_v = const_oper.get_constant();
      PADDLE_ENFORCE_EQ(v_lower.defined() && v_upper.defined(),
                        true,
                        ::common::errors::InvalidArgument(
                            "V lower and upper should be defined."));
      if (const_v > 0) {
        p_lower_bound = FracOp::Make(v_lower, const_oper);
        p_upper_bound = FracOp::Make(v_upper, const_oper);
      } else {
        p_lower_bound = FracOp::Make(v_upper, const_oper);
        p_upper_bound = FracOp::Make(v_lower, const_oper);
      }
      AddBaseAndSimplify(lower_bound, p_lower_bound);
      AddBaseAndSimplify(upper_bound, p_upper_bound);
      return true;
    }
  }
  return false;
}

bool CasSimplifyMutator::GetOperandBound(Expr* lower_bound,
                                         Expr* upper_bound,
                                         Expr v,
                                         bool unfold_const_bound) {
  // only support simple operand of int, var and var's product with int
  PADDLE_ENFORCE_NOT_NULL(
      lower_bound,
      ::common::errors::InvalidArgument("Lower bound should not be null."));
  PADDLE_ENFORCE_NOT_NULL(
      upper_bound,
      ::common::errors::InvalidArgument("Upper bound should not be null."));
  auto* v_int = v.As<IntImm>();
  if (v_int) {
    AddBaseAndSimplify(lower_bound, v);
    AddBaseAndSimplify(upper_bound, v);
    return true;
  } else if (GetVarBound(lower_bound, upper_bound, v, unfold_const_bound)) {
    return true;
  }
  return false;
}

bool CasSimplifyMutator::GetSumBound(Expr* lower_bound,
                                     Expr* upper_bound,
                                     Expr sum,
                                     bool unfold_const_bound) {
  // only support sum of int, var and var's product with int
  PADDLE_ENFORCE_NOT_NULL(
      lower_bound,
      ::common::errors::InvalidArgument("Lower bound should not be null."));
  PADDLE_ENFORCE_NOT_NULL(
      upper_bound,
      ::common::errors::InvalidArgument("Upper bound should not be null."));
  auto bound_sum = sum.As<Sum>();
  // CHECK(bound_sum);
  bool get_bound = true;
  Expr sum_lower_bound, sum_upper_bound;
  if (bound_sum) {
    for (Expr& v : bound_sum->operands()) {
      if (!GetOperandBound(
              &sum_lower_bound, &sum_upper_bound, v, unfold_const_bound)) {
        get_bound = false;
        break;
      }
    }
    if (get_bound) {
      *lower_bound = sum_lower_bound;
      *upper_bound = sum_upper_bound;
    }
    return get_bound;
  }
  return false;
}

bool CasSimplifyMutator::GetExprBound(Expr* lower_bound,
                                      Expr* upper_bound,
                                      Expr expr,
                                      bool unfold_const_bound) {
  // only support min's operands as sum, int or var or var's product with int or
  // min/max
  auto bound_sum = expr.As<Sum>();
  auto bound_min = expr.As<Min>();
  auto bound_max = expr.As<Max>();
  bool get_bound = true;
  if (bound_sum) {
    get_bound = GetSumBound(lower_bound, upper_bound, expr, unfold_const_bound);
  } else if (bound_min) {
    get_bound = GetMinBound(lower_bound, upper_bound, expr, unfold_const_bound);
  } else if (bound_max) {
    get_bound = GetMaxBound(lower_bound, upper_bound, expr, unfold_const_bound);
  } else if (!GetOperandBound(
                 lower_bound, upper_bound, expr, unfold_const_bound)) {
    return false;
  }
  return get_bound;
}

bool CasSimplifyMutator::GetMinBound(Expr* lower_bound,
                                     Expr* upper_bound,
                                     Expr min,
                                     bool unfold_const_bound) {
  // only support min's operands as sum, int or var or var's product with int or
  // min/max
  auto bound_min = min.As<Min>();
  PADDLE_ENFORCE_NOT_NULL(
      bound_min,
      ::common::errors::InvalidArgument("Bound min should not be null."));
  bool get_bound = true;
  Expr a_lower_bound, a_upper_bound, b_lower_bound, b_upper_bound;
  get_bound =
      get_bound &&
      GetExprBound(
          &a_lower_bound, &a_upper_bound, bound_min->a(), unfold_const_bound) &&
      GetExprBound(
          &b_lower_bound, &b_upper_bound, bound_min->b(), unfold_const_bound);
  if (get_bound) {
    *lower_bound =
        CasSimplify(Min::Make(a_lower_bound, b_lower_bound), var_intervals);
    *upper_bound =
        CasSimplify(Min::Make(a_upper_bound, b_upper_bound), var_intervals);
  }
  return get_bound;
}

bool CasSimplifyMutator::GetMaxBound(Expr* lower_bound,
                                     Expr* upper_bound,
                                     Expr max,
                                     bool unfold_const_bound) {
  auto bound_max = max.As<Max>();
  PADDLE_ENFORCE_NOT_NULL(
      bound_max,
      ::common::errors::InvalidArgument("Bound max should not be null."));
  bool get_bound = true;
  Expr a_lower_bound, a_upper_bound, b_lower_bound, b_upper_bound;
  get_bound =
      get_bound &&
      GetExprBound(
          &a_lower_bound, &a_upper_bound, bound_max->a(), unfold_const_bound) &&
      GetExprBound(
          &b_lower_bound, &b_upper_bound, bound_max->b(), unfold_const_bound);
  if (get_bound) {
    *lower_bound =
        CasSimplify(Max::Make(a_lower_bound, b_lower_bound), var_intervals);
    *upper_bound =
        CasSimplify(Max::Make(a_upper_bound, b_upper_bound), var_intervals);
  }
  return get_bound;
}

bool CasSimplifyMutator::SimplifySpecificSumMod(Expr* result, Expr a, Expr b) {
  // case1: (32+(-x))%33 = 32-x%33 (0<=x<=32)
  // case2: (x-32)%33 = x%33 - 32%33 (0<=x<=32)
  auto a_sum = a.As<Sum>();
  auto b_i = b.As<IntImm>();
  if (!a_sum || !b_i) {
    return false;
  }
  // if 0 < b < 3, (3a+b) % 6 = (3a % 6) + (b % 6)
  if (a_sum->operands().size() == 2) {
    a_sum->operands()[0] = CasSimplify(a_sum->operands()[0], var_intervals);
    auto sum_a_prod = a_sum->operands()[0].As<Product>();
    auto sum_b_var = a_sum->operands()[1].As<_Var_>();
    if (sum_a_prod && sum_b_var && var_intervals.count(sum_b_var->name)) {
      auto sum_a_prod_b_int = sum_a_prod->operand(1).As<IntImm>();
      if (sum_a_prod_b_int)
        std::swap(sum_a_prod->operand(0), sum_a_prod->operand(1));
      auto sum_a_prod_a_int = sum_a_prod->operand(0).As<IntImm>();
      auto& interval = var_intervals.at(sum_b_var->name);
      int b_abs = std::abs(b_i->value);
      int sum_prod_a_abs = std::abs(sum_a_prod_a_int->value);
      if (sum_a_prod_a_int && (b_abs % sum_prod_a_abs == 0)) {
        if (std::abs(interval.l) < sum_prod_a_abs &&
            std::abs(interval.r) < sum_prod_a_abs) {
          *result = CasSimplify(
              Sum::Make({CasSimplify(Mod::Make(a_sum->operands()[0], b),
                                     var_intervals),
                         CasSimplify(Mod::Make(a_sum->operands()[1], b),
                                     var_intervals)}),
              var_intervals);
          return true;
        }
      }
    }
  }
  return cinn::common::DefaultDeviceTarget().arch.Match(
      [&](common::NVGPUArch) { return false; },
      [&](std::variant<common::HygonDCUArchHIP, common::HygonDCUArchSYCL>) {
        return false;
      },
      [&](std::variant<common::UnknownArch, common::X86Arch, common::ARMArch>) {
        int const_value = 0;
        Expr lower_bound;
        Expr upper_bound;
        Expr rest_oper;
        bool can_simplify = true;
        bool has_int = false;
        // fold only the expr bound(may contains the var) and try to simplify
        // the var
        Expr unfolded_lower_bound, unfolded_upper_bound;
        for (Expr& v : a_sum->operands()) {
          auto* v_int = v.As<IntImm>();
          if (v_int) {
            const_value += v_int->value;
            has_int = true;
          } else if (GetVarBound(&lower_bound, &upper_bound, v, false)) {
            AddBaseAndSimplify(&rest_oper, v);
          } else {
            can_simplify = false;
            break;
          }
        }
        can_simplify = can_simplify && has_int &&
                       std::abs(const_value) % b_i->value == b_i->value - 1 &&
                       lower_bound.defined() && upper_bound.defined() &&
                       rest_oper.defined();
        // further infer the vars' bound by the intervals infos, try to get the
        // constant
        if (can_simplify) {
          std::vector<Expr> bounds = {lower_bound, upper_bound};
          for (int i = 0; i < bounds.size(); ++i) {
            Expr bound = bounds[i];
            Expr bound_l, bound_r;
            GetExprBound(&bound_l, &bound_r, bound);
            if (i == 0 && bound_l.defined()) {
              lower_bound = bound_l;
            }
            if (i == 1 && bound_r.defined()) {
              upper_bound = bound_r;
            }
          }
        } else {
          return false;
        }
        // case1: (32+(-x))%33 = 32-x%33 (0<=x<=32)
        // case2: (x-32)%33 = x%33 - 32%33 (0<=x<=32)
        can_simplify = can_simplify && lower_bound.is_constant();
        bool case1 = can_simplify && const_value >= 0 &&
                     lower_bound.get_constant() >= -const_value &&
                     upper_bound.is_constant() &&
                     upper_bound.get_constant() <= 0;
        bool case2 = can_simplify && const_value <= 0 &&
                     lower_bound.get_constant() >= 0 &&
                     upper_bound.is_constant() &&
                     upper_bound.get_constant() <= -const_value;
        can_simplify = can_simplify && (case1 || case2);
        if (can_simplify) {
          Expr const_expr;
          if (const_value < 0) {
            const_expr = make_const(b->type(), const_value % b_i->value);
          } else {
            const_expr = make_const(b->type(), const_value % b_i->value);
          }
          *result = CasSimplify(
              Sum::Make({const_expr,
                         CasSimplify(Mod::Make(rest_oper, b), var_intervals)}),
              var_intervals);
          return true;
        }
        return false;
      });
}

// Return if the var's interval is nonnegative.
inline bool IsVarNonnegative(
    const absl::flat_hash_map<std::string, CasInterval>& var_intervals,
    const std::string& var_name) {
  return var_intervals.count(var_name) && var_intervals.at(var_name).l >= 0;
}

// Return if the var is binded with thread or block in cuda(which implies it is
// non-negative).
inline bool IsVarBinded(const std::string& var_name) {
  return utils::StartsWith(var_name, "threadIdx") ||
         utils::StartsWith(var_name, "blockIdx");
}

/**
 * Return if exprs are still all nonnegative vars.
 * @param all_nonnegative_var is previous exprs all nonnegative vars.
 * @param arg_var the pointer of this var.
 * @param var_intervals intervals of each var.
 * @return if exprs are still all nonnegative vars.
 */
inline bool IsVarAllNonnegative(
    bool all_nonnegative_var,
    _Var_* arg_var,
    const absl::flat_hash_map<std::string, CasInterval>& var_intervals) {
  // All exprs all nonnegative vars if previous exprs are nonnegative
  // vars(all_nonnegative_var == true) and this expr is a var (arg_var !=
  // nullptr) and (this var's interval is nonnegative or this var is binded to
  // thread or block in cuda).
  return all_nonnegative_var && arg_var &&
         (IsVarNonnegative(var_intervals, arg_var->name) ||
          IsVarBinded(arg_var->name));
}

Expr CasSimplifyMutator::SimplifyMod(Expr u) {
  VLOG(6) << "SimplifyMod:" << u;
  auto* node = u.As<Mod>();
  PADDLE_ENFORCE_NOT_NULL(
      node, ::common::errors::InvalidArgument("Node should not be null."));

  auto a = CasSimplify(node->a(), var_intervals);
  auto b = CasSimplify(node->b(), var_intervals);

  auto* a_i = a.As<IntImm>();
  auto* a_product = a.As<Product>();
  auto* a_sum = a.As<Sum>();
  auto* a_var = a.As<_Var_>();
  auto* a_mod = a.As<Mod>();
  auto* a_add = a.As<Add>();

  auto* b_i = b.As<IntImm>();

  // 7 % 3
  if (a_i && b_i) {
    return make_const(a_i->type(), a_i->value % b_i->value);
  }

  // x % 1 = 0
  if (b_i && b_i->value == 1) return make_const(b_i->type(), 0);

  // handle cases:
  // (x * 6) % 2 = 0
  // (x * 2) % 6 = (x % 3) * 2
  if (b_i && a_product && b_i->value > 0) {
    for (int i = 0; i < a_product->operands().size(); i++) {
      auto a_op_i = a_product->operand(i);
      if (a_op_i.As<IntImm>() && a_op_i.As<IntImm>()->value > 0) {
        int a_op_int = a_op_i.As<IntImm>()->value;
        // case: (x * 6) % 2 = 0
        if (a_op_int % b_i->value == 0) return make_const(a_product->type(), 0);
        // case: (x * y * 2) % 6 = ((x * y) % 3) * 2
        if (b_i->value % a_op_int == 0) {
          int new_b = b_i->value / a_op_int;
          std::vector<Expr> a_operands = a_product->operands();
          a_operands.erase(a_operands.begin() + i);
          return Product::Make(
              {SimplifyMod(Mod::Make(Product::Make(a_operands), Expr(new_b))),
               Expr(a_op_int)});
        }
      }
    }
  }

  // (x % 16) % 4 = x % 4
  if (a_mod && b_i) {
    VLOG(6) << "Simplify sequential mod";
    auto* a_b_i = a_mod->b().As<IntImm>();
    if (a_b_i && a_b_i->value != 0 && a_b_i->value % b_i->value == 0) {
      auto e = SimplifyMod(Mod::Make(a_mod->a(), b_i));
      VLOG(6) << "Reduce Mod from " << u << " to " << e;
      return e;
    }
  }

  // 0 % x = 0, 1 % x = 1
  if (a_i && (a_i->value == 0 || a_i->value == 1)) return a;

  if (b_i && a_var && var_intervals.count(a_var->name)) {
    auto& interval = var_intervals.at(a_var->name);
    int b_abs = std::abs(b_i->value);
    // x\in[1, 3] % 4 = x
    if (std::abs(interval.l) < b_abs && std::abs(interval.r) < b_abs) return a;
    // [3,3] % 3 = 0
    if (interval.l == interval.r && interval.l % b_abs == 0)
      return make_const(b_i->type(), 0);
  }

  if (a_product && b_i) {
    if (IsDivisible(a_product, b_i->value)) {
      return make_const(Int(32), 0);
    }
  }

  // (4*x + k*y)%2 = (k*y) %2
  // (2x+y+z) % 2 = (y+z) % 2
  if (a_sum && b_i) {
    VLOG(6) << "A SUM ";
    std::vector<Expr> sum_args;
    for (auto& v : a_sum->operands()) {
      if (!IsDivisible(v, b_i->value)) {
        VLOG(6) << v;
        sum_args.push_back(v);
      }
    }

    if (sum_args.empty()) return make_const(b_i->type(), 0);
    // handle the case: (2x+y+z) % 2 = (y+z) % 2 when y>=0 and z>=0
    if (sum_args.size() == 1) {
      return SimplifyMod(Mod::Make(sum_args[0], b));
    } else if (sum_args.size() < a_sum->operands().size()) {
      bool all_nonnegative_var = true;
      bool all_nonnegative_int = true;
      for (int i = 0; i < sum_args.size(); i++) {
        auto* arg_var = sum_args[i].As<_Var_>();
        all_nonnegative_var =
            IsVarAllNonnegative(all_nonnegative_var, arg_var, var_intervals);
        auto* arg_int = sum_args[i].As<IntImm>();
        all_nonnegative_int =
            all_nonnegative_int && arg_int && arg_int->value >= 0;
      }
      VLOG(6) << all_nonnegative_var << " " << all_nonnegative_int;
      if (all_nonnegative_var)
        return SimplifyMod(Mod::Make(Sum::Make(sum_args), b));
      if (all_nonnegative_int) {
        int sum_value = 0;
        for (auto& i : sum_args) sum_value += i.As<IntImm>()->value;
        return make_const(a_sum->type(), sum_value % b_i->value);
      }
      return SimplifyMod(Mod::Make(Sum::Make(sum_args), b));
    } else if (sum_args.size() == a_sum->operands().size()) {
      if (b_i->value > 0 && !var_intervals.empty()) {
        // case1: (32+(-x))%33 = 32-x%33 (0<=x<=32)
        // case2: (x-32))%33 = x%33 - 32%33 (0<=x<=32)
        Expr result;
        // TODO(phlrain): disable this simplify
        /*
              if (SimplifySpecificSumMod(&result, a, b)) {
                return result;
              }
        */
      }
      return Mod::Make(a, b);
    }
  }

  return Mod::Make(a, b);
}

Expr CasSimplifyMutator::SimplifyMinAndMax(Expr u) {
  // simplify min/max
  auto* u_max = u.As<Max>();
  auto* u_min = u.As<Min>();
  if (u_max) {
    Expr a = CasSimplify(u_max->a(), var_intervals);
    Expr b = CasSimplify(u_max->b(), var_intervals);
    bool is_a_const = a.is_constant();
    bool is_b_const = b.is_constant();
    if (is_a_const && is_b_const) {
      return a.get_constant() >= b.get_constant() ? a : b;
    }
    Expr lower_bound, upper_bound;
    Expr const_operand, non_const_operand;
    if (is_a_const) {
      const_operand = a;
      non_const_operand = b;
    }
    if (is_b_const) {
      const_operand = b;
      non_const_operand = a;
    }
    if (const_operand.defined() && non_const_operand.defined()) {
      auto const_size = const_operand.get_constant();
      // unfold var with bounds
      if (GetExprBound(&lower_bound, &upper_bound, non_const_operand, true)) {
        // if non_const_operand's lower_bound is larger than const_operand, then
        // non_const_operand must be larger than const_operand
        if (lower_bound.is_constant() &&
            const_size <= lower_bound.get_constant()) {
          return non_const_operand;
        }
        // if non_const_operand's upper_bound is smaller than a, then
        // const_operand must be larger than non_const_operand
        if (upper_bound.is_constant() &&
            const_size >= upper_bound.get_constant()) {
          return const_operand;
        }
      }
      // not unfold var for var may be eliminated in the calculation
      if (GetExprBound(&lower_bound, &upper_bound, non_const_operand, false)) {
        // if non_const_operand's lower_bound is larger than const_operand, then
        // non_const_operand must be larger than const_operand
        lower_bound = CasSimplify(lower_bound, var_intervals);
        upper_bound = CasSimplify(upper_bound, var_intervals);
        if (lower_bound.is_constant() &&
            const_size <= lower_bound.get_constant()) {
          return non_const_operand;
        }
        // if non_const_operand's upper_bound is smaller than a, then
        // const_operand must be larger than non_const_operand
        if (upper_bound.is_constant() &&
            const_size >= upper_bound.get_constant()) {
          return const_operand;
        }
      }
    }
    return ir::Max::Make(a, b);
  }

  if (u_min) {
    Expr a = CasSimplify(u_min->a(), var_intervals);
    Expr b = CasSimplify(u_min->b(), var_intervals);
    bool is_a_const = a.is_constant();
    bool is_b_const = b.is_constant();
    if (is_a_const && is_b_const) {
      return a.get_constant() <= b.get_constant() ? a : b;
    }
    Expr lower_bound, upper_bound;
    Expr const_operand, non_const_operand;
    if (is_a_const) {
      const_operand = a;
      non_const_operand = b;
    }
    if (is_b_const) {
      const_operand = b;
      non_const_operand = a;
    }
    if (const_operand.defined() && non_const_operand.defined()) {
      auto const_size = const_operand.get_constant();
      if (GetExprBound(&lower_bound, &upper_bound, non_const_operand, true)) {
        // if non_const_operand's lower_bound is larger than const_operand, then
        // non_const_operand must be larger than const_operand
        if (lower_bound.is_constant() &&
            const_size <= lower_bound.get_constant()) {
          return const_operand;
        }
        // if non_const_operand's upper_bound is smaller than a, then
        // const_operand must be larger than non_const_operand
        if (upper_bound.is_constant() &&
            const_size >= upper_bound.get_constant()) {
          return non_const_operand;
        }
      }
      if (GetExprBound(&lower_bound, &upper_bound, non_const_operand, false)) {
        // if non_const_operand's lower_bound is larger than const_operand, then
        // non_const_operand must be larger than const_operand
        if (lower_bound.is_constant() &&
            const_size <= lower_bound.get_constant()) {
          return const_operand;
        }
        // if non_const_operand's upper_bound is smaller than a, then
        // const_operand must be larger than non_const_operand
        if (upper_bound.is_constant() &&
            const_size >= upper_bound.get_constant()) {
          return non_const_operand;
        }
      }
    }
    return ir::Min::Make(a, b);
  }
  return u;
}

Expr CasSimplifyMutator::SimplifyCmp(Expr u) {
  Expr a = operator()(u->operand(0));
  Expr b = operator()(u->operand(1));

  if (a.is_constant() && b.is_constant()) {
    switch (u->node_type()) {
      case ir::IrNodeTy::LT:
        return Expr(a.get_constant() < b.get_constant());
      case ir::IrNodeTy::LE:
        return Expr(a.get_constant() <= b.get_constant());
      case ir::IrNodeTy::GT:
        return Expr(a.get_constant() > b.get_constant());
      case ir::IrNodeTy::GE:
        return Expr(a.get_constant() >= b.get_constant());
      case ir::IrNodeTy::EQ:
        return Expr(a.get_constant() == b.get_constant());
      case ir::IrNodeTy::NE:
        return Expr(a.get_constant() != b.get_constant());
    }
  }

  return u;
}

/**
 * deal with index's div-mod add simplification, temporary solution, not cover
 * all situations. case 1: (m / n) * n + m % n = m (m, n's type is int) case 2:
 * (m / n1) * n3 + (n2 * m) % n3 = n2 * m if n3 = n1 * n2 (m, n1, n2, n3's type
 * is int)
 */
Expr CasSimplifyMutator::SimplifySpecificSum(Expr tmp) {
  auto sum = tmp.As<Sum>();
  if (!sum) {
    return tmp;
  }
  if (sum->operands().size() == 1U) return sum->operand(0);
  Expr left = sum->operand(0);
  Expr right = sum->operand(1);
  auto left_mod = left.As<Mod>();
  auto right_mod = right.As<Mod>();
  auto left_mul = left.As<Product>();
  auto right_mul = right.As<Product>();
  auto left_div = left.As<FracOp>();
  auto right_div = right.As<FracOp>();
  // normalize to left mul and right mod
  if (right_mul && left_mod) {
    left_mul = right_mul;
    right_mod = left_mod;
  }
  // normalize to left div and right mod
  if (right_div && left_mod) {
    left_div = right_div;
    right_mod = left_mod;
  }
  if (!right_mod || (!left_mul && !left_div)) {
    return tmp;
  }
  PADDLE_ENFORCE_GE(right_mod->operands().size(),
                    2U,
                    ::common::errors::InvalidArgument(
                        "right_mod's operands size should be greater than 2"));
  Expr mod_left = right_mod->operand(0);
  Expr mod_right = right_mod->operand(1);
  if (!mod_left->type().is_integer() || !mod_right->type().is_integer()) {
    return tmp;
  }
  if (left_mul) {
    // case 1: (m / n) * n + m % n = m (m, n's type is int)
    // case 2: (m / n1) * n3 + (n2 * m) % n3 = n2 * m if n3 = n1 * n2 (m, n1,
    // n2, n3's type is int)
    PADDLE_ENFORCE_GE(left_mul->operands().size(),
                      2U,
                      ::common::errors::InvalidArgument(
                          "left_mul's operands size should be greater than 2"));

    if (left_mul->operands().size() > 2) return tmp;

    Expr mul_left = left_mul->operand(0);
    Expr mul_right = left_mul->operand(1);

    // handle the case1 : n * (m / n)  + m % n = (m / n) * n + m % n = m
    // handle the case2 : n3 * (m / n1) + (n2 * m) % n3 = (m / n1) * n3 + (n2 *
    // m) % n3 = n2 * m if n3 = n1 * n2
    if (MathEqual(mod_right, mul_left)) {
      mul_left = left_mul->operand(1);
      mul_right = left_mul->operand(0);
    } else if (!MathEqual(mod_right, mul_right)) {
      return tmp;
    }
    auto div = mul_left.As<FracOp>();
    if (!div) {
      return tmp;
    }
    PADDLE_ENFORCE_GE(div->operands().size(),
                      2U,
                      ::common::errors::InvalidArgument(
                          "div's operands size should be greater than 2"));
    Expr div_left = div->operand(0);
    Expr div_right = div->operand(1);
    if (!div_left->type().is_integer() || !div_right->type().is_integer()) {
      return tmp;
    }
    if (MathEqual(div_left * mod_right, mod_left * div_right)) {
      tmp = mod_left;
      for (int i = 2; i < sum->operands().size(); i++) {
        tmp = tmp + sum->operand(i);
      }
      return tmp;
    }
  }
  return tmp;
}

Expr CasSimplifyMutator::operator()(Expr u) {
  if (u.As<Min>() || u.As<Max>()) {
    return SimplifyMinAndMax(u);
  }

  u = detail::SumOrProductGetSingleElementsRec(u);

  if (u.is_constant() || u.As<_Var_>()) return u;

  if (u.As<FracOp>()) {
    u = SimplifyFracOp(u);
    auto tmp = FurtherSimplifyFracWithInterval(u, var_intervals);
    if (!tmp.same_as(u)) return operator()(tmp);
    return u;
  }

  if (u.As<Product>()) {
    return detail::SumOrProductGetSingleElementsRec(SimplifyProduct(u));
  }

  if (u.As<Sum>()) {
    auto tmp = detail::SumOrProductGetSingleElementsRec(SimplifySum(u));
    // deal with index's div-mod add simplification, temporary solution, not
    // cover all situations. case 1: (m / n) * n + m % n = m (m, n's type is
    // int) case 2: (m / n1) * n3 + (n2 * m) % n3 = n2 * m if n3 = n1 * n2 (m,
    // n1, n2, n3's type is int) case 3: m / n2 + (n1 * m) % n3 = n1 * m if n3 =
    // n1 * n2 (m, n1, n2, n3's type is int)
    return SimplifySpecificSum(tmp);
  }

  if (u.As<Mod>()) {
    return detail::SumOrProductGetSingleElementsRec(SimplifyMod(u));
  }

  if (u.is_cmp()) {
    return SimplifyCmp(u);
  }

  switch (u.node_type()) {
    case ir::IrNodeTy::And:
    case ir::IrNodeTy::Or:
    case ir::IrNodeTy::Not:
      return SimplifyCond(u);
    default:
      break;
  }

  return u;
}

bool CASasSymbol(Expr expr) {
  auto* load_n = expr.As<Load>();
  auto* var_n = expr.As<_Var_>();
  auto* broadcast_n = expr.As<Broadcast>();

  return load_n || var_n || broadcast_n;
}

Expr ConvertCinnToCAS(Expr expr) {
  VLOG(7) << "Begin ConvertCinnToCAS " << expr;
  struct Mutator : public ir::IRMutator<ir::Expr*> {
    void operator()(Expr* expr) { Visit(expr); }
    void Visit(Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

   private:
    void Visit(const Add* op, Expr* expr) override {
      auto a = op->a();
      auto b = op->b();

      Visit(&a);
      Visit(&b);

      bool is_zero_a = a.is_constant() && a.get_constant() == 0;
      bool is_zero_b = b.is_constant() && b.get_constant() == 0;
      if (is_zero_a) {
        *expr = b;
        return;
      } else if (is_zero_b) {
        *expr = a;
        return;
      }
      // TODO(phlrain): disable convert add to Sum
      // Add support different input datatype
      // Sum don't support different input datatype
      // this convert may failed
      // *expr = Sum::Make({a, b});
    }
    void Visit(const Mul* op, Expr* expr) override {
      auto a = op->a();
      auto b = op->b();

      Visit(&a);
      Visit(&b);

      if (a.is_constant() && a.get_constant() == 0) {
        *expr = make_const(a->type(), 0);
        return;
      }

      if (a.is_constant() && a.get_constant() == 1) {
        *expr = b;
        return;
      }

      if (b.is_constant() && b.get_constant() == 0) {
        *expr = make_const(b->type(), 0);
        return;
      }

      if (b.is_constant() && b.get_constant() == 1) {
        *expr = a;
        return;
      }

      *expr = Product::Make({a, b});
    }

    void Visit(const Sub* op, Expr* expr) override {
      auto a = op->a();
      auto b = op->b();

      Visit(&a);
      Visit(&b);

      bool is_zero_a = a.is_constant() && a.get_constant() == 0;
      bool is_zero_b = b.is_constant() && b.get_constant() == 0;
      if (is_zero_a) {
        *expr = Product::Make({make_const(b->type(), -1), b});
        return;
      } else if (is_zero_b) {
        *expr = a;
        return;
      }

      b = Product::Make({make_const(b->type(), -1), b});
      *expr = Sum::Make({a, b});
    }

    void Visit(const Div* op, Expr* expr) override {
      auto a = op->a();
      auto b = op->b();

      Visit(&a);
      Visit(&b);

      PADDLE_ENFORCE_EQ(
          !is_zero(b),
          true,
          ::common::errors::InvalidArgument("Dividend should not be zero."));

      if (a.is_constant() && a.get_constant() == 0) {
        *expr = make_const(a->type(), 0);
        return;
      }

      if (b.is_constant() && b.get_constant() == 1) {
        *expr = a;
        return;
      }

      // int division, NOTE that 3/2 = 1, 3./2 = 1.5
      *expr = FracOp::Make(a, b);
    }

    void Visit(const Minus* op, Expr* expr) override {
      auto a = op->v();

      Visit(&a);

      if (a.is_constant()) {
        auto value = a.get_constant();
        if (value == 0) {
          *expr = make_const(a->type(), 0);
          return;
        }
      }

      *expr = Product::Make({make_const(a->type(), -1), a});
    }
  };

  Mutator()(&expr);
  return expr;
}

/**
 * @brief Given an expr, visit it. If there is an ir::Min and its operands are 1
 * constant value and 1 inconstant value, return the constant min value. For
 * example, if a < min(5, b), then we get a < 5 and a < b. Using a < 5 to
 * simplify the condition ensures correctness, though not sufficient.
 */
Expr ReplaceMinToConstant(Expr expr) {
  Expr copied = ir::ir_utils::IRCopy(expr);
  struct Mutator : public ir::IRMutator<ir::Expr*> {
    void operator()(Expr* expr) { Visit(expr); }
    void Visit(Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

   private:
    void Visit(const Min* op, Expr* expr) override {
      auto a = op->a();
      auto b = op->b();

      Visit(&a);
      Visit(&b);

      auto min_a = op->a();
      auto min_b = op->b();
      if (min_a.is_constant() && !min_b.is_constant()) {
        PADDLE_ENFORCE_EQ(
            min_a->type().is_integer(),
            true,
            ::common::errors::InvalidArgument("Min a should be an integer."));
        *expr = ir::ir_utils::IRCopy(min_a);
      } else if (min_b.is_constant() && !min_a.is_constant()) {
        PADDLE_ENFORCE_EQ(
            min_b->type().is_integer(),
            true,
            ::common::errors::InvalidArgument("Min b should be an integer."));
        *expr = ir::ir_utils::IRCopy(min_b);
      }
    }
  };
  Mutator()(&copied);
  return copied;
}

/**
 * @brief Given an expr, visit it. If there is an ir::Max and its operands are 1
 * constant value and 1 inconstant value, return the constant max value.
 */
Expr ReplaceMaxToConstant(Expr expr) {
  Expr copied = ir::ir_utils::IRCopy(expr);
  struct Mutator : public ir::IRMutator<ir::Expr*> {
    void operator()(Expr* expr) { Visit(expr); }
    void Visit(Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

   private:
    void Visit(const Max* op, Expr* expr) override {
      auto a = op->a();
      auto b = op->b();

      Visit(&a);
      Visit(&b);

      auto max_a = op->a();
      auto max_b = op->b();
      if (max_a.is_constant() && !max_b.is_constant()) {
        PADDLE_ENFORCE_EQ(
            max_a->type().is_integer(),
            true,
            ::common::errors::InvalidArgument("Max a should be an integer."));
        *expr = ir::ir_utils::IRCopy(max_a);
      } else if (max_b.is_constant() && !max_a.is_constant()) {
        PADDLE_ENFORCE_EQ(
            max_b->type().is_integer(),
            true,
            ::common::errors::InvalidArgument("Max b should be an integer."));
        *expr = ir::ir_utils::IRCopy(max_b);
      }
    }
  };
  Mutator()(&copied);
  return copied;
}

Expr ConvertCasToCinn(Expr expr) {
  VLOG(7) << "Begin ConvertCasToCinn : " << expr;

  struct Mutator : ir::IRMutator<Expr*> {
    void operator()(Expr* expr) { Visit(expr); }
    void Visit(Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

   private:
    void Visit(const Product* op, Expr* expr) override {
      std::vector<Expr> operands;
      auto* node = expr->As<Product>();
      for (auto& v : node->operands()) {
        auto c = v;
        Mutator()(&c);
        operands.push_back(c);
      }

      PADDLE_ENFORCE_EQ(
          !operands.empty(),
          true,
          ::common::errors::InvalidArgument("Operands should not be empty."));
      if (operands.size() == 1) {
        *expr = operands[0];
      } else if (operands.size() == 2) {
        *expr = Mul::Make(operands[0], operands[1]);
      } else {
        auto a = operands[0];
        auto b = Product::Make(EraseFront(operands));
        Mutator()(&b);
        *expr = Mul::Make(a, b);
      }

      // process the Mul
      Visit(expr);
    }

    void Visit(const Sum* op, Expr* expr) override {
      std::vector<Expr> operands;
      auto* node = expr->As<Sum>();
      for (auto& v : node->operands()) {
        auto c = v;
        Mutator()(&c);
        operands.push_back(c);
      }

      PADDLE_ENFORCE_EQ(
          !operands.empty(),
          true,
          ::common::errors::InvalidArgument("Operands should not be empty."));
      if (operands.size() == 1) {
        *expr = operands[0];
      } else if (operands.size() == 2) {
        *expr = Add::Make(operands[0], operands[1]);
      } else {
        auto a = operands[0];
        auto b = Sum::Make(EraseFront(operands));
        Mutator()(&b);
        *expr = Add::Make(a, b);
      }

      // process the sum
      Visit(expr);
    }

    void Visit(const FracOp* op, Expr* expr) override {
      auto a = op->a();
      auto b = op->b();

      Visit(&a);
      Visit(&b);

      PADDLE_ENFORCE_EQ(
          !is_zero(b),
          true,
          ::common::errors::InvalidArgument("Dividend should not be zero."));
      *expr = Div::Make(a, b);
      Visit(expr);
    }

    // a + -1*b -> a-b
    void Visit(const Add* op, Expr* expr) override {
      auto a = op->a();
      auto b = op->b();

      Visit(&a);
      Visit(&b);

      auto* bp = b.As<ir::Mul>();
      if (bp && bp->a().is_constant() && bp->a().get_constant() == -1.f) {
        *expr = Sub::Make(a, bp->b());
      } else {
        *expr = Add::Make(a, b);
      }
    }
  };

  Mutator()(&expr);
  return expr;
}

bool IsExprCasCompatible(Expr expr) {
  auto teller = [](const Expr* expr) {
    return expr->As<Add>() || expr->As<Sub>() || expr->As<Mul>() ||
           expr->As<Div>();
  };
  return ir::ir_utils::CollectIRNodes(expr, teller).empty();
}

// Partially divide a by b. e.g. (2x+y)/2 => x + y/2
Expr DividePartially(Sum* a, int b) {
  std::vector<Expr> external_sum_args, sum_args;

  for (auto& item : a->operands()) {
    if (item.As<Product>() && (IsDivisible(item.As<Product>(), b) ||
                               IsDivisible(b, item.As<Product>()))) {
      external_sum_args.push_back(Divide(item.As<Product>(), b));
    } else if (item.As<IntImm>() && IsDivisible(item.As<IntImm>()->value, b)) {
      external_sum_args.push_back(
          make_const(item.type(), item.As<IntImm>()->value / b));
    } else {
      sum_args.push_back(item);
    }
  }

  if (!external_sum_args.empty()) {
    if (sum_args.empty()) return Sum::Make(external_sum_args);
    Expr internal_sum =
        sum_args.size() == 1 ? sum_args[0] : Sum::Make(sum_args);
    Expr new_frac = FracOp::Make(internal_sum, make_const(a->type(), b));
    return Sum::Make(Concat(external_sum_args, {new_frac}));
  }
  return Expr(a);
}

bool IsMonotonical(Expr u, Var v) {
  auto* up = u.As<Product>();
  auto* uv = u.As<_Var_>();

  if (uv && uv->name == v->name) return true;
  if (up) {
    for (auto& item : up->operands()) {
      if (IsMonotonical(item, v)) return true;
    }
  }
  return false;
}

// Should be called after SimplifyFracOp. If y is integer and $y\in \[0, 3\]$,
// then y/4=0
Expr CasSimplifyMutator::FurtherSimplifyFracWithInterval(
    Expr expr,
    const absl::flat_hash_map<std::string, CasInterval>& var_intervals) {
  auto* node = expr.As<FracOp>();
  if (!node) return expr;
  auto a = CasSimplify(node->a(), var_intervals);
  auto b = CasSimplify(node->b(), var_intervals);

  auto* ai = a.As<IntImm>();
  auto* bi = b.As<IntImm>();
  auto* av = a.As<_Var_>();
  auto* bv = b.As<_Var_>();
  auto* ap = a.As<Product>();
  // case: y / 4, y\in[0,3]
  if (bi) {
    if (av) {
      auto it = var_intervals.find(av->name);
      if (it != var_intervals.end() &&
          std::abs(it->second.r) < std::abs(bi->value) &&
          std::abs(it->second.l) < std::abs(bi->value))
        return make_const(a.type(), 0);
    }
  }
  // case: 1/y, y\in(2, 100)
  if (ai) {
    if (bv) {
      auto it = var_intervals.find(bv->name);
      auto ai_abs = std::abs(ai->value);
      if (it != var_intervals.end()) {
        VLOG(7) << "found " << bv->name << " " << it->second << " "
                << " ai " << ai_abs;
      }
      if (it != var_intervals.end() && std::abs(it->second.r) > ai_abs &&
          std::abs(it->second.l) > ai_abs) {
        return make_const(a.type(), 0);
      }
    }
  }
  return expr;
}

Expr SimplifyConstantFrac(FracOp* node) {
  auto* ai = node->a().As<ir::IntImm>();
  auto* au = node->a().As<ir::UIntImm>();
  auto* af = node->a().As<ir::FloatImm>();

  if (ai) {
    auto* bi = node->b().As<ir::IntImm>();
    PADDLE_ENFORCE_NOT_NULL(
        bi, ::common::errors::InvalidArgument("Bi should not be null."));
    return make_const(ai->type(), ai->value / bi->value);
  }

  if (au) {
    auto* bu = node->b().As<ir::UIntImm>();
    PADDLE_ENFORCE_NOT_NULL(
        bu, ::common::errors::InvalidArgument("Bu should not be null."));
    return make_const(au->type(), au->value / bu->value);
  }

  if (af) {
    auto* bf = node->b().As<ir::FloatImm>();
    PADDLE_ENFORCE_NOT_NULL(
        af, ::common::errors::InvalidArgument("Af should not be null."));
    return make_const(af->type(), af->value / bf->value);
  }
  CINN_NOT_IMPLEMENTED
  return Expr();
}

Expr CasSimplifyMutator::SimplifyFracOp(Expr expr) {
  VLOG(7) << "CAS simplify Frac " << expr;
  auto* node = expr.As<FracOp>();
  auto a = CasSimplify(node->a(), var_intervals);
  auto b = CasSimplify(node->b(), var_intervals);

  // update frac op node
  expr = ir::FracOp::Make(a, b);
  node = expr.As<FracOp>();

  auto* ap = a.As<Product>();
  auto* bp = b.As<Product>();
  auto* as = a.As<Sum>();
  auto* bi = b.As<IntImm>();
  auto* ai = a.As<IntImm>();
  auto* af = a.As<FloatImm>();
  auto* bf = b.As<FloatImm>();
  auto* av = a.As<_Var_>();
  auto* bv = b.As<_Var_>();

  // case 1
  // integer constant division: 64/3
  if (node->is_constant()) {
    if (int_compute_) {
      return SimplifyConstantFrac(node);
    } else {
      return SimplifyRationalNumber(expr);
    }
  }

  // case 2
  // sum/x or product/x is divisible
  if (bi) {
    auto* a_sum = a.As<Sum>();
    auto* a_product = a.As<Product>();
    // divisible
    if (a_sum && IsDivisible(a_sum, bi->value)) return Divide(a_sum, bi->value);
    if (a_product) {
      if (IsDivisible(a_product, bi->value) ||
          IsDivisible(bi->value, a_product)) {
        return Divide(a_product, bi->value);
      } else {
        return FracOp::Make(a, b);
      }
    }

    // if 0 < b < 3, (3a+b) / 6 = (3a / 6) + (b / 6)
    if (a_sum && a_sum->operands().size() == 2) {
      a_sum->operands()[0] = CasSimplify(a_sum->operands()[0], var_intervals);
      auto sum_a_prod = a_sum->operands()[0].As<Product>();
      auto sum_b_var = a_sum->operands()[1].As<_Var_>();
      if (sum_a_prod && sum_b_var && var_intervals.count(sum_b_var->name)) {
        auto sum_a_prod_b_int = sum_a_prod->operand(1).As<IntImm>();
        if (sum_a_prod_b_int)
          std::swap(sum_a_prod->operand(0), sum_a_prod->operand(1));
        auto sum_a_prod_a_int = sum_a_prod->operand(0).As<IntImm>();
        auto& interval = var_intervals.at(sum_b_var->name);
        int b_abs = std::abs(bi->value);
        if (sum_a_prod_a_int) {
          int sum_prod_a_abs = std::abs(sum_a_prod_a_int->value);
          if (b_abs % sum_prod_a_abs == 0 &&
              std::abs(interval.l) < sum_prod_a_abs &&
              std::abs(interval.r) < sum_prod_a_abs) {
            return CasSimplify(
                Sum::Make({CasSimplify(FracOp::Make(a_sum->operands()[0], b),
                                       var_intervals),
                           CasSimplify(FracOp::Make(a_sum->operands()[1], b),
                                       var_intervals)}),
                var_intervals);
          }
        }
      }
    }

    // not divisible
    /*
    if (a_sum) {
      auto expr = DividePartially(a_sum, bi->value);
      return expr;
    }
     */
  }

  // cinn_min/cinn_max(a, b)/2 = cinn_min/cinn_max(a/2, b/2)
  if ((bi && bi->value > 0) || (bf && bf->value > 0)) {
    auto cmp_min = a.As<Min>();
    auto cmp_max = a.As<Max>();
    if (cmp_min) {
      return {CasSimplify(
          Min::Make(CasSimplify(FracOp::Make(cmp_min->a(), b), var_intervals),
                    CasSimplify(FracOp::Make(cmp_min->b(), b), var_intervals)),
          var_intervals)};
    }
    if (cmp_max) {
      return {CasSimplify(
          Max::Make(CasSimplify(FracOp::Make(cmp_max->a(), b), var_intervals),
                    CasSimplify(FracOp::Make(cmp_max->b(), b), var_intervals)),
          var_intervals)};
    }
  }

  if (av && bi) {
    if (var_intervals.count(av->name)) {
      auto& interval = var_intervals.at(av->name);
      int b_abs = std::abs(bi->value);
      if (std::abs(interval.l) < b_abs && std::abs(interval.r) < b_abs)
        return make_const(bi->type(), 0);
      return FracOp::Make(a, b);
    }
  }

  // (32x+y)/32 = x + y/32
  if (as && bi) {
    std::vector<Expr> external_sum_args;
    std::vector<Expr> internal_sum_args;
    for (auto& e : as->operands()) {
      if (IsDivisible(e, bi->value)) {
        if (e.As<Sum>())
          external_sum_args.push_back(Divide(e.As<Sum>(), bi->value));
        if (e.As<IntImm>())
          external_sum_args.push_back(
              make_const(bi->type(), e.As<IntImm>()->value / bi->value));
        if (e.As<Product>())
          external_sum_args.push_back(Divide(e.As<Product>(), bi->value));
      } else {
        internal_sum_args.push_back(e);
      }
    }

    Expr external_sum, internal_sum;
    if (!external_sum_args.empty()) {
      if (external_sum_args.size() == 1)
        external_sum = external_sum_args.front();
      else
        external_sum = Sum::Make(external_sum_args);
    }

    if (!internal_sum_args.empty()) {
      internal_sum = FracOp::Make(Sum::Make(internal_sum_args), b);
    }

    if (external_sum.defined() && internal_sum.defined()) {
      return CasSimplify(Sum::Make({external_sum, internal_sum}),
                         var_intervals);
    }
    if (external_sum.defined()) return CasSimplify(external_sum, var_intervals);
    return internal_sum;
  }

  // solve the case: 2abc / b
  // Both avs and bvs should be sorted first.
  auto reduce_product_div_product = [](const std::vector<Expr>& avs,
                                       const std::vector<Expr>& bvs) {
    std::vector<Expr> avs1, bvs1;
    int i = 0;
    int j = 0;

    ExprPosCmp cmp;

    while (i < avs.size() && j < bvs.size()) {
      auto& a = avs[i];
      auto& b = bvs[j];
      if (a.is_constant() && b.is_constant()) {
        auto* ai = a.As<IntImm>();
        auto* bi = b.As<IntImm>();
        auto* af = a.As<FloatImm>();
        auto* bf = b.As<FloatImm>();
        if (ai) {
          PADDLE_ENFORCE_NOT_NULL(
              bi, ::common::errors::InvalidArgument("Bi should not be null."));
          int g = gcd(ai->value, bi->value);
          int a_d = ai->value / g;
          int b_d = bi->value / g;

          avs1.push_back(make_const(a.type(), a_d));
          if (b_d != 1) bvs1.push_back(make_const(b.type(), b_d));
        } else if (af || bf) {
          double value = af->value / bf->value;
          const auto& ftype = af ? af->type() : bf->type();
          avs1.push_back(make_const(ftype, value));
        } else {
          avs1.push_back(a);
          bvs1.push_back(b);
        }

        // CHECK(!af) << a << " " << b;
        i++;
        j++;
      } else if (avs[i] == bvs[j]) {
        i++;
        j++;
      } else {
        // <
        if (cmp(avs[i], bvs[j])) {
          avs1.push_back(avs[i++]);
        } else {
          bvs1.push_back(bvs[j++]);
        }
      }
    }
    while (i < avs.size()) {
      avs1.push_back(avs[i++]);
    }
    while (j < bvs.size()) {
      bvs1.push_back(bvs[j++]);
    }
    if (avs1.empty()) return make_const(avs[0].type(), 1);
    if (bvs1.empty()) return Product::Make(avs1);

    return FracOp::Make(Product::Make(avs1), Product::Make(bvs1));
  };

  {
    // TODO(SunNy820828449): fix in future.
    // std::vector<Expr> a_args, b_args;
    // if (ap)
    //   a_args = ap->operands();
    // else
    //   a_args.push_back(a);
    // if (bp)
    //   b_args = bp->operands();
    // else
    //   b_args.push_back(b);
    // return reduce_product_div_product(a_args, b_args);
  }

  // x / x
  if (a.type().is_int() && b.type().is_int() && av && bv) {
    if (a == b) return make_const(a.type(), 1);
  }

  if (node->a().same_as(a) && node->b().same_as(b)) return expr;
  return FracOp::Make(a, b);
}

Expr CasSimplifyMutator::SimplifyCond(Expr u) {
  switch (u->node_type()) {
      // -------------------------- NOT -----------------------------
    case ir::IrNodeTy::Not: {
      auto* node = u.As<ir::Not>();
      Expr v = operator()(node->v());
      switch (v.node_type()) {
          // Not 1 = (1 == 0)
        case ir::IrNodeTy::IntImm:
          return Expr(v.As<IntImm>()->value == 0);
          // Not Not v = v
        case ir::IrNodeTy::Not:
          return v;
          // Not <= is >
        case ir::IrNodeTy::LE:
          return ir::GT::Make(v->operand(0), v->operand(1));
          // Not < is >=
        case ir::IrNodeTy::LT:
          return ir::GE::Make(v->operand(0), v->operand(1));
          // Not >= is <
        case ir::IrNodeTy::GE:
          return ir::LT::Make(v->operand(0), v->operand(1));
          // Not > is <=
        case ir::IrNodeTy::GT:
          return ir::LE::Make(v->operand(0), v->operand(1));
        default:
          return ir::Not::Make(v);
      }
    } break;
      // -------------------------- AND OR -----------------------------
    case ir::IrNodeTy::And:
    case ir::IrNodeTy::Or: {
      Expr a = operator()(u->operand(0));
      Expr b = operator()(u->operand(1));
      if (a.is_constant() || b.is_constant()) {
        if (u.As<ir::And>()) {
          // 1 && b is b
          if (a.As<ir::UIntImm>()) {
            return a.As<ir::UIntImm>()->value ? b : Expr(false);
          }
          // a && 1 is a
          if (b.As<ir::UIntImm>()) {
            return b.As<ir::UIntImm>()->value ? a : Expr(false);
          }
          return ir::And::Make(a, b);
        }
        if (u.As<ir::Or>()) {
          // 1 || b is 1
          if (a.As<ir::UIntImm>()) {
            return a.As<ir::UIntImm>()->value ? a : b;
          }
          // a || 1 is 1
          if (b.As<ir::UIntImm>()) {
            return b.As<ir::UIntImm>()->value ? b : a;
          }
        }
        return ir::Or::Make(a, b);
      }

      return u;
    }

    default:
      return u;
  }
}

}  // namespace detail

Expr CasSimplify(
    Expr u,
    const absl::flat_hash_map<std::string, CasInterval>& var_intervals) {
  return detail::CasSimplifyMutator(var_intervals)(u);
}

Expr SolveInequality(Expr inequality, Var val) {
  auto copied = AutoSimplify(inequality);

  auto* le_n = copied.As<ir::LE>();
  auto* lt_n = copied.As<ir::LT>();
  auto* gt_n = copied.As<ir::GT>();
  auto* ge_n = copied.As<ir::GE>();

  Expr a, b;

#define __(x__)   \
  if (x__) {      \
    a = x__->a(); \
    b = x__->b(); \
  }
  __(le_n)
  __(lt_n)
  __(gt_n)
  __(ge_n)
#undef __
  Expr all = AutoSimplify(a - b);

  // if (cinn::common::IsPureMath(a) && cinn::common::IsPureMath(b)) {
  if (true) {
    auto _res_positive_ = cinn::common::Solve(a, b, val);  // NOLINT
    auto& res = std::get<0>(_res_positive_);
    auto& positive = std::get<1>(_res_positive_);
    // Simplify it with CAS to avoid random result from GiNac.
    res = AutoSimplify(res);
    res = cinn::common::cast(res, val->type());

    if (le_n) {
      if (positive) return ir::LE::Make(val, res);
      return ir::GE::Make(val, res);
    }
    if (lt_n) {
      if (positive) return ir::LT::Make(val, res);
      return ir::GT::Make(val, res);
    }
    if (ge_n) {
      if (positive) return ir::GE::Make(val, res);
      return ir::LE::Make(val, res);
    }
    if (gt_n) {
      if (positive) return ir::GT::Make(val, res);
      return ir::LT::Make(val, res);
    }
  } else {
    return AutoSimplify(inequality);
  }
  return Expr();
}

}  // namespace common
}  // namespace cinn
