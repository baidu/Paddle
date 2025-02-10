// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

// TODO(yifan): Add unittest here
#include "paddle/cinn/common/equation_graph_topo_walker.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

namespace adt {
namespace common {

using VT = int;
using FT = std::string;
/*
Graph ex:

  1-> "1->10" -> 10
  2-> "2->20" -> 20
*/

TEST(EquationGraphTopoWalker, simple1) {
  auto F4V = [](VT variable, const std::function<void(FT)>& visitor) {
    if (variable == 1) {
      visitor("1->10");
    } else if (variable == 2) {
      visitor("2->20");
    }
  };
  auto InV4F = [](FT function, const std::function<void(VT)>& visitor) {
    if (function == "1->10") {
      visitor(1);
    } else if (function == "2->20") {
      visitor(2);
    }
  };
  auto OutV4F = [](FT function, const std::function<void(VT)>& visitor) {
    if (function == "1->10") {
      visitor(10);
    } else if (function == "2->20") {
      visitor(20);
    }
  };
  cinn::EquationGraphTopoWalker<VT, FT> walker(F4V, InV4F, OutV4F);
  std::vector<FT> outputs;
  std::function<void(FT)> FunctionVisitor = [&](FT function) {
    outputs.push_back(function);
  };
  walker.WalkFunction(1, FunctionVisitor);

  std::vector<FT> expected{"1->10"};
  EXPECT_TRUE((outputs == expected));
}

/*
Graph ex:

  1 -> "1->10, 1->11" -> 10
                      -> 11
  2 -> "2->20" -> 20
  3 -> "3->30, 3->31" -> 30
                      -> 31
*/
TEST(EquationGraphTopoWalker, simple2) {
  auto F4V = [](VT variable, const std::function<void(FT)>& visitor) {
    if (variable == 1) {
      visitor("1->10, 1->11");
    } else if (variable == 2) {
      visitor("2->20");
    } else if (variable == 3) {
      visitor("3->30, 3->31");
    }
  };
  auto InV4F = [](FT function, const std::function<void(VT)>& visitor) {
    if (function == "1->10, 1->11") {
      visitor(1);
    } else if (function == "2->20") {
      visitor(2);
    } else if (function == "3->30, 3->31") {
      visitor(3);
    }
  };
  auto OutV4F = [](FT function, const std::function<void(VT)>& visitor) {
    if (function == "1->10, 1->11") {
      visitor(10);
      visitor(11);
    } else if (function == "2->20") {
      visitor(20);
    } else if (function == "3->30, 3->31") {
      visitor(30);
      visitor(31);
    }
  };
  cinn::EquationGraphTopoWalker<VT, FT> walker(F4V, InV4F, OutV4F);
  std::vector<VT> outputs;
  std::function<void(VT)> VariableVisitor = [&](VT variable) {
    outputs.push_back(variable);
  };
  walker.WalkVariable(1, VariableVisitor);
  std::vector<VT> expected{1, 10, 11};
  EXPECT_TRUE((outputs == expected));
}

}  // namespace common
}  // namespace adt
