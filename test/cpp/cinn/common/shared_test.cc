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

#include "paddle/cinn/common/shared.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include "paddle/cinn/common/object.h"

namespace cinn {
namespace common {

struct A : public Object {
  const char *type_info() const override { return "A"; }

  Shared<A> other;
};

class B : public Object {};

TEST(Shared, test) {
  Shared<A> a_ref(make_shared<A>());
  ASSERT_EQ(ref_count(a_ref.get()).val(), 1);

  {  // local copy
    Shared<A> b = a_ref;
    EXPECT_EQ(ref_count(a_ref.get()).val(), 2);
    ASSERT_EQ(ref_count(b.get()).val(), 2);
  }

  ASSERT_EQ(ref_count(a_ref.get()).val(), 1);
}

TEST(Shared, cycle_share) {
  {
    Shared<A> a_ref(make_shared<A>());
    a_ref->other = a_ref;
    ASSERT_EQ(a_ref->__ref_count__.val(), 2);
  }
}

}  // namespace common
}  // namespace cinn
