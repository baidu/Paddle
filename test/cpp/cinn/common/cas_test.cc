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

#include <gtest/gtest.h>

#include "paddle/cinn/cinn.h"
#include "paddle/cinn/common/common.h"
#include "paddle/cinn/common/ir_util.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/op/ir_operators.h"
#include "paddle/cinn/utils/string.h"

namespace cinn {
namespace common {

using cinn::common::make_const;
using utils::GetStreamCnt;
using utils::Join;
using utils::Trim;
using namespace ir;  // NOLINT

TEST(CAS, number_cal) {
  // 1 * 100 * -1 + 0 + 1001
  auto u1 = Sum::Make(
      {Product::Make({Expr(1), Expr(100), Expr(-1)}), Expr(0), Expr(1001)});
  LOG(INFO) << u1;
}

TEST(CAS, cmp) {
  detail::ExprPosCmp cmp;

  Var x = ir::_Var_::Make("x", Int(32));
  Var y = ir::_Var_::Make("y", Int(32));
  Var z = ir::_Var_::Make("z", Int(32));

  EXPECT_EQ(cmp(x, Expr(1)), false);
  EXPECT_EQ(cmp(Expr(1), x), true);

  // x * y * z > x * y
  EXPECT_EQ(cmp(ir::Product::Make({x, y, z}), ir::Product::Make({x, y})),
            false);
  // x * y * z > 10 * y * z
  EXPECT_EQ(
      cmp(ir::Product::Make({x, y, z}), ir::Product::Make({Expr(10), y, z})),
      false);
  // 1 * y * z < 10 * y * z
  EXPECT_EQ(cmp(ir::Product::Make({Expr(1), y, z}),
                ir::Product::Make({Expr(10), y, z})),
            true);
}

TEST(CAS, SimplifySum) {
  Var x = ir::_Var_::Make("x", Int(32));
  Var y = ir::_Var_::Make("y", Int(32));
  Var z = ir::_Var_::Make("z", Int(32));
  // x + y + z + 0
  auto u1 = Sum::Make({x, y, z, make_const(0)});
  // x*1 + y + z + 0
  auto u2 = Sum::Make({Product::Make({x, Expr(1)}), y, z, make_const(0)});
  // z + 1 + y + x + zx
  auto u3 = CasSimplify(Sum::Make({z, Expr(1), y, x, Product::Make({z, x})}));
  // z + 1 + y + 3 + x + 0 + zx
  auto u4 = CasSimplify(
      Sum::Make({z, Expr(1), y, Expr(3), x, Expr(0), Product::Make({z, x})}));
  // (-1 * x) + x
  auto u5 = CasSimplify(Sum::Make({Product::Make({Expr(-1), x}), x}));
  // x2 + 3zy + -3*yz + -2x + 1
  auto u6 = CasSimplify(Sum::Make({Product::Make({x, Expr(2)}),
                                   Product::Make({z, y, Expr(3)}),
                                   Product::Make({Expr(-3), y, z}),
                                   Product::Make({Expr(-2), x}),
                                   Expr(1)}));

  EXPECT_EQ(GetStreamCnt(CasSimplify(u1)), "(x + y + z)");
  EXPECT_EQ(GetStreamCnt(CasSimplify(u2)), "(x + y + z)");
  EXPECT_EQ(GetStreamCnt(u3), "(1 + x + y + z + (x * z))");
  EXPECT_EQ(GetStreamCnt(u4), "(4 + x + y + z + (x * z))");
  EXPECT_EQ(GetStreamCnt(u5), "0");
  EXPECT_EQ(GetStreamCnt(u6), "1");
}

TEST(CAS, SimplifyProduct) {
  Var x = ir::_Var_::Make("x", Int(32));
  Var y = ir::_Var_::Make("y", Int(32));
  Var z = ir::_Var_::Make("z", Int(32));

  // zyx*(-1)
  auto u2 = CasSimplify(Product::Make({z, y, x, Expr(-1)}));

  EXPECT_EQ(GetStreamCnt(u2), "(-1 * x * y * z)");
}

TEST(CAS, SimplifyMod) {
  Var x = ir::_Var_::Make("x", Int(32));
  Var y = ir::_Var_::Make("y", Int(32));
  Var z = ir::_Var_::Make("z", Int(32));

  // 2*x % 2 = 0
  auto u1 = CasSimplify(Mod::Make(Product::Make({x, Expr(2)}), Expr(2)));
  // (x+y+z) % 2 = x%2 + y%2 + z%2
  auto u2 = CasSimplify(Mod::Make(Sum::Make({x, y, z}), Expr(2)));
  // x%2 + 1%2 + x%2
  auto u3 = CasSimplify(Sum::Make({Mod::Make(x, Expr(2)),
                                   Mod::Make(Expr(1), Expr(2)),
                                   Mod::Make(x, Expr(2))}));

  EXPECT_EQ(GetStreamCnt(u1), "0");
  EXPECT_EQ(GetStreamCnt(u2), "((x + y + z) % 2)");
  EXPECT_EQ(GetStreamCnt(u3), "1");
}

TEST(CAS, SimplifyModForVectorize) {
  Var x = ir::_Var_::Make("x", Int(32));
  Var y = ir::_Var_::Make("y", Int(32));

  // (((8*x + 1024*y) % 802816) % 7168) %64
  // = (8*x + 1024*y) %64           // since 7168 and 802816 is k*64
  // = (8*x) % 64                   // since 1024 is k*64
  // = (8*x - ((8*x) // 64) * 64    // since mod definition a%b = a - (a//b)*b
  // = (8*x) - (x//8)*64
  // = (8*x) - (x//8)*(8*8)
  // = 8*(x-(x//8)*8)               // since mod definition
  // = 8*(x%8)
  auto u1 = CasSimplify(
      Mod::Make(Mod::Make(Mod::Make(Sum::Make({Product::Make({x, Expr(8)}),
                                               Product::Make({y, Expr(1024)})}),
                                    Expr(802816)),
                          Expr(7168)),
                Expr(64)));
  std::cout << GetStreamCnt(u1);
  EXPECT_EQ(GetStreamCnt(u1), "((x % 8) * 8)");
}

TEST(CAS, ConvertCinnToCAS) {
  Placeholder<float> A("A", {10, 10});
  Placeholder<float> B("B", {10, 10});

  auto C = Compute(
      {Expr(10), Expr(10)},
      [&](Expr i, Expr j) {
        return A(i, j) + 0.f + 1.f + 2.f * B(i, j) + 0.f * B(i, j) * A(i, j);
      },
      "C");

  Expr body = C->body();
  LOG(INFO) << "body " << body;

  body = detail::ConvertCinnToCAS(body);
  body = CasSimplify(body);
  EXPECT_EQ(GetStreamCnt(body),
            "(1.00000000f + A[i, j] + (2.00000000f * B[i, j]))");
  body = detail::ConvertCasToCinn(body);
  EXPECT_EQ(GetStreamCnt(body),
            "(1.00000000f + (A[i, j] + (2.00000000f * B[i, j])))");
}

TEST(CAS, FracOp) {
  Var x = ir::_Var_::Make("x", Int(32));
  Var y = ir::_Var_::Make("y", Int(32));
  Var z = ir::_Var_::Make("z", Int(32));

  auto u1 = AutoSimplify(Div::Make(Expr(1), x) * x);
  EXPECT_EQ(GetStreamCnt(u1), "((1 / x) * x)");
  // 64x/32 + y + 64/32
  auto u2 = AutoSimplify(Expr(64) * x / Expr(32) + y + Expr(64) / Expr(32));
  ASSERT_EQ(GetStreamCnt(u2), "(2 + ((2 * x) + y))");
  // 1/32 * y * z * 32768 * 2
  auto u3 = AutoSimplify(Expr(1) / Expr(32) * y * z * 32768 * 2);
  EXPECT_EQ(GetStreamCnt(u3), "0");
  // 32768 * (32x + y) + y
  auto u4 = AutoSimplify(Expr(32768) * (((Expr(32) * x) + y) / 32));
  EXPECT_EQ(GetStreamCnt(u4), "((32768 * (y / 32)) + (32768 * x))");

  cinn::common::cas_intervals_t var_intervals;
  var_intervals.emplace("y", cinn::common::CasInterval(0, 31));
  auto u = AutoSimplify((Expr(x) * 32 + y) / 32, var_intervals);
  EXPECT_EQ(GetStreamCnt(u), "x");

  u = AutoSimplify((Expr(x) * 33 + y) / 32, var_intervals);
  EXPECT_EQ(GetStreamCnt(u), "(((33 * x) + y) / 32)");

  u = AutoSimplify(Expr(125) / 8 - 1);
  EXPECT_EQ(GetStreamCnt(u), "14");
}

#define OUTPUT_EQUAL(s__) EXPECT_EQ(GetStreamCnt(u), s__);

TEST(CAS, Mod) {
  Var x = ir::_Var_::Make("x", Int(32));
  Var y = ir::_Var_::Make("y", Int(32));
  Var z = ir::_Var_::Make("z", Int(32));
  Var k = ir::_Var_::Make("k", Int(32));

  absl::flat_hash_map<std::string, CasInterval> var_intervals0, var_intervals1;
  var_intervals0.emplace("x", CasInterval{0, 3});
  var_intervals0.emplace("y", CasInterval{0, 3});
  var_intervals0.emplace("z", CasInterval{0, 3});
  var_intervals0.emplace("k", CasInterval{0, 3});

  Expr u;
  u = AutoSimplify(x % 5);
  EXPECT_EQ(GetStreamCnt(u), "(x % 5)");
  OUTPUT_EQUAL("(x % 5)")

  u = AutoSimplify((5 + x) % 5);
  OUTPUT_EQUAL("(x % 5)")

  u = AutoSimplify((x + 5 * y + 1 + 1 + 3 - z * 3) % 5);
  OUTPUT_EQUAL("((x + (-3 * z)) % 5)")

  // u = AutoSimplify((x + 5) % 5, var_intervals0);
  // OUTPUT_EQUAL("x")

  // u = AutoSimplify((x + y + 5) % 5, var_intervals0);
  // OUTPUT_EQUAL("((x + y) % 5)")

  // u = AutoSimplify((x + 20 * y + 5) % 5, var_intervals0);
  // OUTPUT_EQUAL("x")

  u = AutoSimplify(
      (x % 32) + ((32768 * (x / 32)) + ((32768 * y) + ((32 * z) + (128 * k)))));
  OUTPUT_EQUAL(
      "((32768 * (x / 32)) + ((x % 32) + ((128 * k) + ((32768 * y) + (32 * "
      "z)))))");

  u = AutoSimplify(
      (x % 32) + ((32768 * (x / 32)) + ((32768 * y) + ((32 * z) + (128 * k)))),
      var_intervals0);
  OUTPUT_EQUAL("((128 * k) + (x + ((32768 * y) + (32 * z))))")

  // (2x+y+z) % 2 = (y+z) % 2
  u = AutoSimplify((2 * x + y + z) % 2, var_intervals0);
  OUTPUT_EQUAL("((y + z) % 2)")

  // 0 % x = 0
  u = AutoSimplify(0 % x);
  OUTPUT_EQUAL("0")

  // 1 % x = 1
  u = AutoSimplify(1 % x);
  OUTPUT_EQUAL("1")

  // (x * 6) % 2 = 0
  u = AutoSimplify((x * 6) % 2);
  OUTPUT_EQUAL("0")

  // (x * 2) % 6 = (x % 3) * 2
  u = AutoSimplify((x * 2) % 6);
  OUTPUT_EQUAL("((x % 3) * 2)")

  // 7 % 3 = 1
  u = AutoSimplify(Expr(7) % Expr(3));
  OUTPUT_EQUAL("1")

  // x % 1 = 0
  u = AutoSimplify(x % 1);
  OUTPUT_EQUAL("0")

  // (m / n) * n + m % n = m (m, n's type is int)
  u = AutoSimplify((x / 10) * 10 + x % 10);
  OUTPUT_EQUAL("x")

  u = AutoSimplify(((x + y * 2) / 10) * 10 + (x + y * 2) % 10 + 3 * z);
  OUTPUT_EQUAL("(x + ((2 * y) + (3 * z)))")
}

TEST(CAS, IntConnerCase) {
  Var x = ir::_Var_::Make("x", Int(32));
  Var y = ir::_Var_::Make("y", Int(32));
  Var z = ir::_Var_::Make("z", Int(32));

  auto u1 = AutoSimplify(Expr(1) / 32);
  EXPECT_EQ(GetStreamCnt(u1), "0");
  auto u2 = AutoSimplify(x / 32 + (x * 32 + 64) / 32);
  EXPECT_EQ(GetStreamCnt(u2), "((x / 32) + (2 + x))");
  // (32x+y)/32 * 1024 * 32
  auto u3 = AutoSimplify((((((32 * x) + y) / 32) * 1024) * 32));
  EXPECT_EQ(GetStreamCnt(u3), "((32768 * (y / 32)) + (32768 * x))");

  auto u4 = AutoSimplify(Expr(1) / 3);
  EXPECT_EQ(GetStreamCnt(u4), "0");

  absl::flat_hash_map<std::string, CasInterval> var_intervals0, var_intervals1;
  var_intervals0.emplace("y", CasInterval{2, 3});
  var_intervals1.emplace("y", CasInterval{0, 3});

  auto u5 = AutoSimplify(Expr(1) / y, var_intervals0);
  EXPECT_EQ(GetStreamCnt(u5), "0");
  auto u6 = AutoSimplify(y / 4, var_intervals0);
  EXPECT_EQ(GetStreamCnt(u6), "0");

  auto u7 = AutoSimplify(1 / y, var_intervals1);
  EXPECT_EQ(GetStreamCnt(u7), "(1 / y)");
  auto u8 = AutoSimplify(-1 / y, var_intervals1);
  EXPECT_EQ(GetStreamCnt(u8), "(-1 / y)");
}

TEST(SolveInequality, basic) {
  Var x("x", Int(32));
  Var y("y", Int(32));

#define TEST_SOLVE(expr__, str__) \
  EXPECT_EQ(GetStreamCnt(SolveInequality(expr__, x)), str__);
  TEST_SOLVE(x * -1 + 20 < 0, "(x > 20)");
  TEST_SOLVE(x * 2 + 3 < x * 10 - 20, "(x > 2)");
  TEST_SOLVE(x * -1 < -1, "(x > 1)");
  TEST_SOLVE(Expr(2) * x * -1 - x < x + 200, "(x > -50)");
  TEST_SOLVE(Expr(2) * x + 30 - x * 3 + y * 23 < 2, "(x > (28 + (23 * y)))");
  TEST_SOLVE(x + ir::Min::Make(Expr(2), Expr(3) * y) < 100,
             "(x < cinn_max((100 + (-3 * y)), 98))");
}

TEST(CAS, SimplifyCompoundMod) {
  {  // (-a % 4) * (-1)
    Var x = ir::_Var_::Make("x", Int(32));
    auto p0 = ir::Product::Make({ir::Mod::Make(-x, Expr(4)), Expr(-1)});
    LOG(INFO) << "p0 " << p0;
    auto p2 = AutoSimplify(p0);
    LOG(INFO) << "simplified " << p2;
    EXPECT_EQ(GetStreamCnt(p2), "(-1 * ((-1 * x) % 4))");
  }
  {  // (33 + x % 34) + -33
    Var x = ir::_Var_::Make("x", Int(32));
    auto p0 = ir::Sum::Make(
        {Expr(33), ir::Sum::Make({ir::Mod::Make(x, Expr(4)), Expr(-33)})});
    LOG(INFO) << "p0 " << p0;
    auto p2 = AutoSimplify(p0);
    LOG(INFO) << "simplified " << p2;
    EXPECT_EQ(GetStreamCnt(p2), "(x % 4)");
  }
  {  // 33 + (x % 2 + (-16))
    Var x = ir::_Var_::Make("x", Int(32));
    auto p0 = ir::Sum::Make(
        {Expr(33),
         ir::Sum::Make({ir::Mod::Make(x, Expr(2)),
                        ir::Product::Make({Expr(-1), Expr(16)})})});
    LOG(INFO) << "p0 " << p0;
    auto p2 = AutoSimplify(p0);
    LOG(INFO) << "simplified " << p2;
    EXPECT_EQ(GetStreamCnt(p2), "(17 + (x % 2))");
  }
  {  // (32- x1 - 16 * x2) % 33
    Var x1 = ir::_Var_::Make("x1", Int(32));
    Var x2 = ir::_Var_::Make("x2", Int(32));
    auto p0 =
        ir::Mod::Make(ir::Sum::Make({Expr(32), -x1, Expr(16) * -x2}), Expr(33));
    LOG(INFO) << "p0 " << p0;
    absl::flat_hash_map<std::string, CasInterval> var_intervals;
    var_intervals.emplace("x1", CasInterval{0, 15});
    var_intervals.emplace("x2", CasInterval{0, 1});
    auto p2 = AutoSimplify(p0, var_intervals);
    LOG(INFO) << "simplified " << p2;
#ifdef CINN_WITH_CUDA
    EXPECT_EQ(GetStreamCnt(p2), "((32 + ((-1 * x1) + (-16 * x2))) % 33)");
#else
    EXPECT_EQ(GetStreamCnt(p2), "(32 + (((-1 * x1) + (-16 * x2)) % 33))");
#endif
  }
}
TEST(CAS, SimplifyNegative) {
  {  // (-1*x) /2
    Var x = ir::_Var_::Make("x", Int(32));
    auto p0 = ir::FracOp::Make(-x, Expr(2));
    LOG(INFO) << "p0 " << p0;
    auto p2 = AutoSimplify(p0);
    LOG(INFO) << "simplified " << p2;
    EXPECT_EQ(GetStreamCnt(p2), "((-1 * x) / 2)");
  }
  {  // minus(1)
    auto p0 = ir::Minus::Make(Expr(1));
    LOG(INFO) << "p0 " << p0;
    auto p2 = AutoSimplify(p0);
    LOG(INFO) << "simplified " << p2;
    EXPECT_EQ(GetStreamCnt(p2), "-1");
  }
}

TEST(CAS, SimplifyMinMax) {
  {  // 1+cinn_min(15, x)
    Var x = ir::_Var_::Make("x", Int(32));
    auto p0 = ir::Sum::Make({Expr(1), ir::Min::Make(Expr(15), x)});
    LOG(INFO) << "p0 " << p0;
    auto p2 = CasSimplify(p0);
    LOG(INFO) << "simplified " << p2;
    EXPECT_EQ(GetStreamCnt(p2), "cinn_min(16, (1 + x))");
  }
  {  // 2*cinn_min(15, x)
    Var x = ir::_Var_::Make("x", Int(32));
    auto p0 = ir::Product::Make({Expr(2), ir::Min::Make(Expr(15), x)});
    LOG(INFO) << "p0 " << p0;
    auto p2 = CasSimplify(p0);
    LOG(INFO) << "simplified " << p2;
    EXPECT_EQ(GetStreamCnt(p2), "cinn_min(30, (2 * x))");
  }
  {  // cinn_min(15, x)/2
    Var x = ir::_Var_::Make("x", Int(32));
    auto p0 = ir::FracOp::Make(ir::Min::Make(Expr(15), x), Expr(2));
    LOG(INFO) << "p0 " << p0;
    auto p2 = CasSimplify(p0);
    LOG(INFO) << "simplified " << p2;
    EXPECT_EQ(GetStreamCnt(p2), "cinn_min(7, (x / 2))");
  }
  {  // -(cinn_min(16, 3400-x-1)-1)/2 + x
    Var x = ir::_Var_::Make("x", Int(32));
    auto p0 =
        ir::FracOp::Make(ir::Min::Make(Expr(16), 3400 - x - 1) - 1, Expr(2));
    p0 = -p0 + x;
    LOG(INFO) << "p0 " << p0;
    auto p2 = AutoSimplify(p0);
    LOG(INFO) << "simplified " << p2;
    EXPECT_EQ(GetStreamCnt(p2),
              "cinn_max((-1699 + ((-1 * ((-1 * x) / 2)) + x)), (-7 + x))");
  }
  {  // cinn_max((-1 * (3399 + (-16 * i_j_fused_outer))), -15)
    Var x = ir::_Var_::Make("x", Int(32));
    auto p0 = ir::Max::Make(
        ir::Product::Make(
            {Expr(-1), ir::Sum::Make({Expr(3399), Expr(-16) * x})}),
        Expr(-15));
    LOG(INFO) << "p0 " << p0;
    auto p2 = AutoSimplify(p0);
    LOG(INFO) << "simplified " << p2;
    EXPECT_EQ(GetStreamCnt(p2), "cinn_max((-3399 + (16 * x)), -15)");
  }
}

TEST(CAS, cond) {
  {
    Expr cond = Expr(2) > Expr(1);
    EXPECT_EQ(GetStreamCnt(CasSimplify(cond)), "true");
  }
  {
    Var a("a");
    Expr cond = (Expr(2) > Expr(1)) && (a < 20);
    EXPECT_EQ(GetStreamCnt(CasSimplify(cond)), "(a < 20)");
  }
  {
    Var a("a");
    Expr cond = (Expr(2) < Expr(1)) && (a < 20);
    EXPECT_EQ(GetStreamCnt(CasSimplify(cond)), "false");
  }
}

TEST(CAS, SimplifyFracOp) {
  Expr frac = Expr(1) / Expr(7) / Expr(6) / Expr(5) / Expr(4);
  EXPECT_EQ(GetStreamCnt(AutoSimplify(frac)), "0");
}

}  // namespace common
}  // namespace cinn
