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

#include "paddle/cinn/ir/op/ir_operators.h"

#include <limits>
#include <string>

#include "paddle/cinn/common/const_fold.h"
#include "paddle/cinn/common/simplify_special_pattern.h"
#include "paddle/cinn/common/target.h"
#include "paddle/cinn/common/type.h"
#include "paddle/cinn/hlir/op/op_util.h"
#include "paddle/cinn/lang/compute.h"
#include "paddle/cinn/runtime/flags.h"

namespace cinn {
namespace ir {
using attr_t = absl::variant<int, float, bool, std::string>;

Expr operator<<(Expr a, Expr b) {
  PADDLE_ENFORCE_EQ(a.type().is_int() || a.type().is_uint(),
                    true,
                    ::common::errors::InvalidArgument(
                        "The input's type should be int or uint."));
  PADDLE_ENFORCE_EQ(b.type().is_int() || b.type().is_uint(),
                    true,
                    ::common::errors::InvalidArgument(
                        "The input's type should be int or uint."));
  auto int_a = a.As<IntImm>();
  auto int_b = b.As<IntImm>();
  Type t_a = a.type();
  Type t_b = b.type();
  if (t_a.is_index_type() && t_b.is_index_type()) {
    if (int_b) {
      PADDLE_ENFORCE_EQ(
          int_b->value >= 0 && int_b->value < t_a.bits(),
          true,
          ::common::errors::InvalidArgument(
              "Shift amount must be non-negative and less than %d for type %s.",
              t_a.bits(),
              t_a));
      if (int_b->value == 0) return a;
    }
    if (int_a && int_b) {
      return Expr(int_a->value << int_b->value);
    }
  }
  return lang::CallExtern("left_shift", {a, b}, {{"vectorizable", false}});
}

Expr operator>>(Expr a, Expr b) {
  PADDLE_ENFORCE_EQ(a.type().is_int() || a.type().is_uint(),
                    true,
                    ::common::errors::InvalidArgument(
                        "The input's type should be int or uint."));
  PADDLE_ENFORCE_EQ(b.type().is_int() || b.type().is_uint(),
                    true,
                    ::common::errors::InvalidArgument(
                        "The input's type should be int or uint."));
  auto int_a = a.As<IntImm>();
  auto int_b = b.As<IntImm>();
  Type t_a = a.type();
  Type t_b = b.type();
  if (t_a.is_index_type() && t_b.is_index_type()) {
    if (int_b) {
      PADDLE_ENFORCE_EQ(
          int_b->value >= 0 && int_b->value < t_a.bits(),
          true,
          ::common::errors::InvalidArgument(
              "Shift amount must be non-negative and less than %d for type %s.",
              t_a.bits(),
              t_a));
      if (int_b->value == 0) return a;
    }
    if (int_a && int_b) {
      return Expr(int_a->value >> int_b->value);
    }
  }
  return lang::CallExtern("right_shift", {a, b}, {{"vectorizable", false}});
}

Expr BitwiseOrCallImpl(common::UnknownArch,
                       const Target &target,
                       Expr a,
                       Expr b) {
  std::stringstream ss;
  ss << "Unsupported arch: " << target.arch_str() << " for bitwise_or.";
  PADDLE_THROW(::common::errors::InvalidArgument(ss.str()));
}

Expr BitwiseOrCallImpl(common::X86Arch, const Target &target, Expr a, Expr b) {
  return lang::CallExtern("bitwise_or", {a, b}, {{"vectorizable", false}});
}

Expr BitwiseOrCallImpl(common::ARMArch, const Target &target, Expr a, Expr b) {
  std::stringstream ss;
  ss << "Unsupported arch: " << target.arch_str() << " for bitwise_or.";
  PADDLE_THROW(::common::errors::InvalidArgument(ss.str()));
}

Expr BitwiseOrCallImpl(common::NVGPUArch,
                       const Target &target,
                       Expr a,
                       Expr b) {
  Type t_a = a.type();
  auto func_name = hlir::GetExternFuncName(target, t_a, "bitwise_or");
  return lang::CallExtern(func_name, {a, b}, {{"vectorizable", false}});
}

Expr BitwiseOrCallImpl(common::HygonDCUArchHIP,
                       const Target &target,
                       Expr a,
                       Expr b) {
  Type t_a = a.type();
  auto func_name = hlir::GetExternFuncName(target, t_a, "bitwise_or");
  return lang::CallExtern(func_name, {a, b}, {{"vectorizable", false}});
}

Expr BitwiseOrCallImpl(common::HygonDCUArchSYCL,
                       const Target &target,
                       Expr a,
                       Expr b) {
  Type t_a = a.type();
  auto func_name = hlir::GetExternFuncName(target, t_a, "bitwise_or");
  return lang::CallExtern(func_name, {a, b}, {{"vectorizable", false}});
}

Expr BitwiseOrCall(const Target &target, Expr a, Expr b) {
  return std::visit(
      [&](const auto &arch) { return BitwiseOrCallImpl(arch, target, a, b); },
      target.arch.variant());
}

Expr operator|(Expr a, Expr b) {
  PADDLE_ENFORCE_EQ(a.type().is_int() || a.type().is_uint(),
                    true,
                    ::common::errors::InvalidArgument(
                        "The input's type should be int or uint."));
  PADDLE_ENFORCE_EQ(b.type().is_int() || b.type().is_uint(),
                    true,
                    ::common::errors::InvalidArgument(
                        "The input's type should be int or uint."));
  auto int_a = a.As<IntImm>();
  auto int_b = b.As<IntImm>();
  Type t_a = a.type();
  Type t_b = b.type();
  if (t_a.is_index_type() && t_b.is_index_type()) {
    if (int_a && int_b) {
      return Expr(int_a->value | int_b->value);
    }
  }
  auto target = cinn::runtime::CurrentTarget::GetCurrentTarget();
  return BitwiseOrCall(target, a, b);
}

Expr BitwiseAndCallImpl(common::UnknownArch,
                        const Target &target,
                        Expr a,
                        Expr b) {
  std::stringstream ss;
  ss << "Unsupported arch: " << target.arch_str() << " for bitwise_and.";
  PADDLE_THROW(::common::errors::InvalidArgument(ss.str()));
}

Expr BitwiseAndCallImpl(common::X86Arch, const Target &target, Expr a, Expr b) {
  return lang::CallExtern("bitwise_and", {a, b}, {{"vectorizable", false}});
}

Expr BitwiseAndCallImpl(common::ARMArch, const Target &target, Expr a, Expr b) {
  std::stringstream ss;
  ss << "Unsupported arch: " << target.arch_str() << " for bitwise_and.";
  PADDLE_THROW(::common::errors::InvalidArgument(ss.str()));
}

Expr BitwiseAndCallImpl(common::NVGPUArch,
                        const Target &target,
                        Expr a,
                        Expr b) {
  Type t_a = a.type();
  auto func_name = hlir::GetExternFuncName(target, t_a, "bitwise_and");
  return lang::CallExtern(func_name, {a, b}, {{"vectorizable", false}});
}

Expr BitwiseAndCallImpl(common::HygonDCUArchHIP,
                        const Target &target,
                        Expr a,
                        Expr b) {
  Type t_a = a.type();
  auto func_name = hlir::GetExternFuncName(target, t_a, "bitwise_and");
  return lang::CallExtern(func_name, {a, b}, {{"vectorizable", false}});
}

Expr BitwiseAndCallImpl(common::HygonDCUArchSYCL,
                        const Target &target,
                        Expr a,
                        Expr b) {
  Type t_a = a.type();
  auto func_name = hlir::GetExternFuncName(target, t_a, "bitwise_and");
  return lang::CallExtern(func_name, {a, b}, {{"vectorizable", false}});
}

Expr BitwiseAndCall(const Target &target, Expr a, Expr b) {
  return std::visit(
      [&](const auto &arch) { return BitwiseAndCallImpl(arch, target, a, b); },
      target.arch.variant());
}

Expr operator&(Expr a, Expr b) {
  PADDLE_ENFORCE_EQ(a.type().is_int() || a.type().is_uint(),
                    true,
                    ::common::errors::InvalidArgument(
                        "The input's type should be int or uint."));
  PADDLE_ENFORCE_EQ(b.type().is_int() || b.type().is_uint(),
                    true,
                    ::common::errors::InvalidArgument(
                        "The input's type should be int or uint."));
  auto int_a = a.As<IntImm>();
  auto int_b = b.As<IntImm>();
  Type t_a = a.type();
  Type t_b = b.type();
  if (t_a.is_index_type() && t_b.is_index_type()) {
    if (int_a && int_b) {
      return Expr(int_a->value & int_b->value);
    }
  }
  auto target = cinn::runtime::CurrentTarget::GetCurrentTarget();
  return BitwiseAndCall(target, a, b);
}

Expr BitwiseXorCallImpl(common::UnknownArch,
                        const Target &target,
                        Expr a,
                        Expr b) {
  std::stringstream ss;
  ss << "Unsupported arch: " << target.arch_str() << " for bitwise_xor.";
  PADDLE_THROW(::common::errors::InvalidArgument(ss.str()));
}

Expr BitwiseXorCallImpl(common::X86Arch, const Target &target, Expr a, Expr b) {
  return lang::CallExtern("bitwise_xor", {a, b}, {{"vectorizable", false}});
}

Expr BitwiseXorCallImpl(common::ARMArch, const Target &target, Expr a, Expr b) {
  std::stringstream ss;
  ss << "Unsupported arch: " << target.arch_str() << " for bitwise_xor.";
  PADDLE_THROW(::common::errors::InvalidArgument(ss.str()));
}

Expr BitwiseXorCallImpl(common::NVGPUArch,
                        const Target &target,
                        Expr a,
                        Expr b) {
  Type t_a = a.type();
  auto func_name = hlir::GetExternFuncName(target, t_a, "bitwise_xor");
  return lang::CallExtern(func_name, {a, b}, {{"vectorizable", false}});
}

Expr BitwiseXorCallImpl(common::HygonDCUArchHIP,
                        const Target &target,
                        Expr a,
                        Expr b) {
  Type t_a = a.type();
  auto func_name = hlir::GetExternFuncName(target, t_a, "bitwise_xor");
  return lang::CallExtern(func_name, {a, b}, {{"vectorizable", false}});
}

Expr BitwiseXorCallImpl(common::HygonDCUArchSYCL,
                        const Target &target,
                        Expr a,
                        Expr b) {
  Type t_a = a.type();
  auto func_name = hlir::GetExternFuncName(target, t_a, "bitwise_xor");
  return lang::CallExtern(func_name, {a, b}, {{"vectorizable", false}});
}

Expr BitwiseXorCall(const Target &target, Expr a, Expr b) {
  return std::visit(
      [&](const auto &arch) { return BitwiseXorCallImpl(arch, target, a, b); },
      target.arch.variant());
}

Expr operator^(Expr a, Expr b) {
  PADDLE_ENFORCE_EQ(a.type().is_int() || a.type().is_uint(),
                    true,
                    ::common::errors::InvalidArgument(
                        "The input's type should be int or uint."));
  PADDLE_ENFORCE_EQ(b.type().is_int() || b.type().is_uint(),
                    true,
                    ::common::errors::InvalidArgument(
                        "The input's type should be int or uint."));
  auto int_a = a.As<IntImm>();
  auto int_b = b.As<IntImm>();
  Type t_a = a.type();
  Type t_b = b.type();
  if (t_a.is_index_type() && t_b.is_index_type()) {
    if (int_a && int_b) {
      return Expr(int_a->value ^ int_b->value);
    }
  }
  auto target = cinn::runtime::CurrentTarget::GetCurrentTarget();
  return BitwiseXorCall(target, a, b);
}

Expr BitwiseNotCallImpl(common::UnknownArch, const Target &target, Expr a) {
  std::stringstream ss;
  ss << "Unsupported arch: " << target.arch_str() << " for bitwise_not.";
  PADDLE_THROW(::common::errors::InvalidArgument(ss.str()));
}

Expr BitwiseNotCallImpl(common::X86Arch, const Target &target, Expr a) {
  return lang::CallExtern("bitwise_not", {a}, {{"vectorizable", false}});
}

Expr BitwiseNotCallImpl(common::ARMArch, const Target &target, Expr a) {
  std::stringstream ss;
  ss << "Unsupported arch: " << target.arch_str() << " for bitwise_not.";
  PADDLE_THROW(::common::errors::InvalidArgument(ss.str()));
}

Expr BitwiseNotCallImpl(common::NVGPUArch, const Target &target, Expr a) {
  auto func_name = hlir::GetExternFuncName(target, a->type(), "bitwise_not");
  return lang::CallExtern(func_name, {a}, {{"vectorizable", false}});
}

Expr BitwiseNotCallImpl(common::HygonDCUArchHIP, const Target &target, Expr a) {
  auto func_name = hlir::GetExternFuncName(target, a->type(), "bitwise_not");
  return lang::CallExtern(func_name, {a}, {{"vectorizable", false}});
}

Expr BitwiseNotCallImpl(common::HygonDCUArchSYCL,
                        const Target &target,
                        Expr a) {
  auto func_name = hlir::GetExternFuncName(target, a->type(), "bitwise_not");
  return lang::CallExtern(func_name, {a}, {{"vectorizable", false}});
}

Expr BitwiseNotCall(const Target &target, Expr a) {
  return std::visit(
      [&](const auto &arch) { return BitwiseNotCallImpl(arch, target, a); },
      target.arch.variant());
}

Expr operator~(Expr a) {
  PADDLE_ENFORCE_EQ(a.type().is_int() || a.type().is_uint(),
                    true,
                    ::common::errors::InvalidArgument(
                        "The input's type should be int or uint."));
  auto target = cinn::runtime::CurrentTarget::GetCurrentTarget();
  return BitwiseNotCall(target, a);
}

static IndexExpr SimplifyAdd(const IndexExpr &lhs, const IndexExpr &rhs) {
  // 3 + 4 ===> 7.
  if (auto constRes = cinn::common::TryConstFold<ir::Add>(lhs, rhs))
    return constRes.value();

  // 3 + d0 ===> d0 + 3.
  // d0 + (d1 + d2) ===> (d1 + d2) + d0.
  if (!ComparePriority(lhs, rhs)) {
    return rhs + lhs;
  }

  // (d0 + 2) + 3 ===> d0 + 5.
  auto rhsConst = rhs.As<IntImm>();
  auto lhsAdd = lhs.As<Add>();
  if (lhsAdd && rhsConst) {
    if (auto lrhs = lhsAdd->b().as_index().As<IntImm>()) {
      return lhsAdd->a().as_index() +
             IndexExpr(lrhs->type(), lrhs->value + rhsConst->value);
    }
  }

  // (d0 + 2) + d1 ===> d0 + d1 + 2.
  if (lhsAdd) {
    if (auto lrhs = lhsAdd->b().as_index().As<IntImm>()) {
      return lhsAdd->a().as_index() + rhs +
             IndexExpr(lrhs->type(), lrhs->value);
    }
  }
  // expr * c1 + expr * c2 ===> expr * (c1 + c2)
  auto lhsMul = lhs.As<Mul>();
  auto rhsMul = rhs.As<Mul>();

  IndexExpr first = lhs, second = rhs;
  int64_t lconst = 1, rconst = 1;

  if (lhsMul) {
    if (auto lrhs = lhsMul->b().as_index().As<IntImm>()) {
      lconst = lrhs->value;
      first = lhsMul->a().as_index();
    }
  }

  if (rhsMul) {
    if (auto rrhs = rhsMul->b().as_index().As<IntImm>()) {
      rconst = rrhs->value;
      second = rhsMul->a().as_index();
    }
  }

  if (first == second) {
    return first * IndexExpr(lhs->type(), lconst + rconst);
  }

  if (lconst != 1 && rconst != 1) {
    if (lconst == rconst)
      return (first + second) * IndexExpr(lhs->type(), lconst);
    if (lconst == -rconst)
      return (first - second) * IndexExpr(lhs->type(), lconst);
  }

  // deal corner case!
  if (auto cornerRes = SimplifyAddCornerCase(lhs, rhs)) {
    return cornerRes.value();
  }

  // (d0 + d1) + (d2 + d3) ===> ((d0 + d1) + d2) + d3.
  if (auto rhsAdd = rhs.As<Add>()) {
    return lhs + rhsAdd->a().as_index() + rhsAdd->b().as_index();
  }

  if (!rhs.As<IntImm>()) {
    // dynamic branch!
    if (common::IsSumPartialBySymbol(lhs, rhs))
      return common::SimplifySymbolicAdd(lhs, rhs);

    if (auto rhs_mul = rhs.As<ir::Mul>()) {
      if (rhs_mul->b().as_index().is_constant()) {
        if (common::IsSumPartialBySymbol(lhs, rhs_mul->a().as_index())) {
          return common::SimplifySymbolicAdd(
              lhs, rhs_mul->a().as_index(), rhs_mul->b().as_index());
        }
      }
    }

    // (S0 * S1 * S2) + (S2 * S1 * S3) ==> (S0 + S3) * (S1 * S2)
    auto MergeByCommonFactor =
        [](const ir::IndexExpr &lhs,
           const ir::IndexExpr &rhs) -> std::optional<ir::IndexExpr> {
      auto flatten_mul_lhs = common::GetFlattenExprs<ir::Mul>(lhs);
      auto flatten_mul_rhs = common::GetFlattenExprs<ir::Mul>(rhs);

      if (flatten_mul_lhs.size() > 1 && flatten_mul_rhs.size() > 1) {
        ir::IndexExpr common_factor(lhs.type(), 1);
        std::unordered_map<ir::IndexExpr, int> lhs_count;
        std::unordered_map<ir::IndexExpr, int> rhs_count;

        for (auto &l : flatten_mul_lhs) lhs_count[l]++;
        for (auto &r : flatten_mul_rhs) rhs_count[r]++;
        // Find common factor
        for (auto &l : flatten_mul_lhs) {
          if (rhs_count[l] > 0) {
            common_factor = l * common_factor;
            rhs_count[l]--;
            lhs_count[l]--;
          }
        }
        // Find Lhs remainder
        ir::IndexExpr lhs_remainder(lhs.type(), 1);
        for (auto &l : flatten_mul_lhs) {
          while (lhs_count[l] > 0) {
            lhs_remainder = l * lhs_remainder;
            lhs_count[l]--;
          }
        }
        // Find Rhs remainder
        ir::IndexExpr rhs_remainder(rhs.type(), 1);
        for (auto &r : flatten_mul_rhs) {
          while (rhs_count[r] > 0) {
            rhs_remainder = r * rhs_remainder;
            rhs_count[r]--;
          }
        }

        if (common_factor != ir::IndexExpr(1))
          return (lhs_remainder + rhs_remainder) * common_factor;
      }
      return std::nullopt;
    };

    auto flatten_add_lhs = common::GetFlattenExprs<ir::Add>(lhs);

    bool found = false;
    ir::IndexExpr res(lhs.type(), 0);
    for (size_t i = 0; i < flatten_add_lhs.size(); ++i) {
      auto merge_res = MergeByCommonFactor(flatten_add_lhs[i], rhs);
      if (!found && merge_res.has_value()) {
        res = res + merge_res.value();
        found = true;
      } else {
        res = res + flatten_add_lhs[i];
      }
    }
    if (found) return res;
  }

  return Add::Make(lhs, rhs);
}

static IndexExpr SimplifyMul(const IndexExpr &lhs, const IndexExpr &rhs) {
  // 3 * 4 ===> 12.
  if (auto constRes = cinn::common::TryConstFold<ir::Mul>(lhs, rhs))
    return constRes.value();

  // 3 * d0 ===> d0 * 3.
  // d0 * (d1 + d2) ===> (d1 + d2) * d0.
  if (!ComparePriority(lhs, rhs)) {
    return rhs * lhs;
  }

  // (d0 * 2) * 3 ===> d0 * 6.
  auto rhsConst = rhs.As<IntImm>();
  auto lhsMul = lhs.As<Mul>();
  if (lhsMul && rhsConst) {
    if (auto lrhs = lhsMul->b().as_index().As<IntImm>()) {
      return lhsMul->a().as_index() *
             IndexExpr(lrhs->type(), lrhs->value * rhsConst->value);
    }
  }

  // (d0 + 3) * 5 ===> d0 * 5 + 15.
  auto lhsAdd = lhs.As<Add>();
  if (lhsAdd && rhsConst) {
    if (auto lrhs = lhsAdd->b().as_index().As<IntImm>()) {
      return lhsAdd->a().as_index() * rhs +
             IndexExpr(lrhs->type(), lrhs->value * rhsConst->value);
    }
  }

  // (d0 * 2) * d1 ===> d0 * d1 * 2.
  if (lhsMul) {
    if (auto lrhs = lhsMul->b().as_index().As<IntImm>()) {
      return lhsMul->a().as_index() * rhs *
             IndexExpr(lrhs->type(), lrhs->value);
    }
  }

  // deal corner case!
  if (auto cornerRes = SimplifyMulCornerCase(lhs, rhs)) {
    return cornerRes.value();
  }

  // (d0 * d1) * (d2 * d3) ===> ((d0 * d1) * d2) * d3.
  if (auto rhsMul = rhs.As<Mul>()) {
    return lhs * rhsMul->a().as_index() * rhsMul->b().as_index();
  }
  return Mul::Make(lhs, rhs);
}

static IndexExpr SimplifyDiv(const IndexExpr &lhs, const IndexExpr &rhs) {
  // 15 / 3 ===> 5.
  if (auto constRes = cinn::common::TryConstFold<ir::Div>(lhs, rhs))
    return constRes.value();

  // deal corner case!
  if (auto cornerRes = SimplifyDivCornerCase(lhs, rhs)) {
    return cornerRes.value();
  }

  // static branch!
  if (auto rhsConst = rhs.As<IntImm>()) {
    auto lhsAdd = lhs.As<Add>();
    auto lhsMul = lhs.As<Mul>();
    auto lhsDiv = lhs.As<Div>();

    // (expr1 * c1 * c2 + expr2 * c1 * c3) / c1 ===> expr1 * c2 + expr2 * c3.
    if (lhsAdd) {
      int64_t llhsFactor = lhsAdd->a().as_index().GetLargestMultiplyPart();
      int64_t lrhsFactor = lhsAdd->b().as_index().GetLargestMultiplyPart();
      if (llhsFactor % rhsConst->value == 0 &&
          lrhsFactor % rhsConst->value == 0) {
        return lhsAdd->a().as_index() / rhs + lhsAdd->b().as_index() / rhs;
      }
    }

    // expr1 * (c1 * c2) / c1 ===> expr1 * c2.
    if (lhsMul) {
      if (auto lrhs = lhsMul->b().as_index().As<IntImm>()) {
        if (lrhs->value % rhsConst->value == 0) {
          return lhsMul->a().as_index() *
                 IndexExpr(lrhs->type(), lrhs->value / rhsConst->value);
        }
      }
    }
  } else {
    // dynamic branch!
    if (auto res = DivByPartMul(lhs, rhs, ir::IrNodeTy::Div)) {
      return res.value();
    }
  }

  // static and dynamic common branch!
  // S0 / S1 / 2 ===> S0 / (S1 * 2).
  if (auto lhsDiv = lhs.As<Div>()) {
    return lhsDiv->a().as_index() / (lhsDiv->b().as_index() * rhs);
  }

  return Div::Make(lhs, rhs);
}

static IndexExpr SimplifyMod(const IndexExpr &lhs, const IndexExpr &rhs) {
  // 15 % 4 ===> 3.
  if (auto constRes = cinn::common::TryConstFold<ir::Mod>(lhs, rhs))
    return constRes.value();

  // deal corner case!
  if (auto cornerRes = SimplifyModCornerCase(lhs, rhs)) {
    return cornerRes.value();
  }

  // static branch!
  if (auto rhsConst = rhs.As<IntImm>()) {
    auto lhsAdd = lhs.As<Add>();
    auto lhsMod = lhs.As<Mod>();

    // (expr1 * c1 * c2+ expr2 * c3) % c1 ===> expr2 * c3 % c1.
    if (lhsAdd) {
      int64_t llhsFactor = lhsAdd->a().as_index().GetLargestMultiplyPart();
      int64_t lrhsFactor = lhsAdd->b().as_index().GetLargestMultiplyPart();
      if (llhsFactor % rhsConst->value == 0)
        return lhsAdd->b().as_index() % rhs;
      if (lrhsFactor % rhsConst->value == 0)
        return lhsAdd->a().as_index() % rhs;
    }

    // expr1 * (c1 * c2) % c1 ===> 0.
    if (lhs.GetLargestMultiplyPart() % rhsConst->value == 0)
      return IndexExpr(0);

    // expr1 % (c1 * c2) % c1 ===> expr1 % c1.
    if (lhsMod) {
      int64_t llhsFactor = lhsMod->b().as_index().GetLargestMultiplyPart();
      if (llhsFactor % rhsConst->value == 0)
        return lhsMod->a().as_index() % rhs;
    }
  } else {
    // dynamic branch!
    if (auto res = DivByPartMul(lhs, rhs, ir::IrNodeTy::Mod)) {
      return IndexExpr(0);
    }
  }

  // static and dynamic common branch!
  if (auto res = SimplifyComplexMod(lhs, rhs)) {
    return res.value();
  }

  return Mod::Make(lhs, rhs);
}

IndexExpr operator-(const IndexExpr &a) { return a * (-1); }
IndexExpr operator-(const IndexExpr &a, const IndexExpr &b) { return a + (-b); }
IndexExpr operator-(int64_t a, const IndexExpr &b) { return a + (-b); }
IndexExpr operator-(const IndexExpr &a, int64_t b) { return a + (-b); }
IndexExpr operator-(int32_t a, const IndexExpr &b) { return a + (-b); }
IndexExpr operator-(const IndexExpr &a, int32_t b) { return a + (-b); }

// Macro to define binary operators
#define DEFINE_BINARY_OPERATORS(OP, FUNC)                         \
  IndexExpr operator OP(const IndexExpr &a, const IndexExpr &b) { \
    return FUNC(a, b);                                            \
  }                                                               \
  IndexExpr operator OP(int64_t a, const IndexExpr &b) {          \
    return IndexExpr(a) OP b;                                     \
  }                                                               \
  IndexExpr operator OP(const IndexExpr &a, int64_t b) {          \
    return a OP IndexExpr(b);                                     \
  }                                                               \
  IndexExpr operator OP(int32_t a, const IndexExpr &b) {          \
    return IndexExpr(a) OP b;                                     \
  }                                                               \
  IndexExpr operator OP(const IndexExpr &a, int32_t b) {          \
    return a OP IndexExpr(b);                                     \
  }

DEFINE_BINARY_OPERATORS(+, SimplifyAdd)
DEFINE_BINARY_OPERATORS(*, SimplifyMul)
DEFINE_BINARY_OPERATORS(/, SimplifyDiv)
DEFINE_BINARY_OPERATORS(%, SimplifyMod)

// Undefine the macro to prevent it from affecting other parts of the code
#undef DEFINE_BINARY_OPERATORS

}  // namespace ir
}  // namespace cinn
