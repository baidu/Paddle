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

#include "paddle/cinn/poly/compute_at_transform.h"

#include <gtest/gtest.h>

namespace cinn {
namespace poly {

TEST(ComputeAtTransform2, basic) {
  isl::ctx ctx(isl_ctx_alloc());
  isl::set pdomain(ctx, "{ p[i,j]: 0<=i,j<100 }");
  isl::map ptransform(ctx,
                      "{ p[i,j]->p[t0,t1,t2]: t0=i%4 and t1=i/4 and t2=j }");
  isl::set cdomain(ctx, "{ c[i,j,k]: 0<=i,j,k<50 }");
  isl::map ctransform(
      ctx, "{ c[i,j,k]->c[t0,t1,t2,t3]: t0=i/4 and t1=i%4 and t2=j and t3=k }");

  isl::map access(
      ctx, "{ c[i,j,k]->p[i,j]; c[i,j,k]->p[i+1,j]; c[i,j,k]->p[i-1,j] }");

  poly::ComputeAtTransform t(
      pdomain, cdomain, access, ptransform, ctransform, 1);
  t();

  t.DisplayC();

  isl::map pschedule(ctx,
                     "{ p[i0,i1,i2,i3,i4] -> [t0,t1,t1t, t2,t3,t4,t5]: t0=i0 "
                     "and t1=i1 and t2=i2 and t3=i3 and t4=i4 "
                     "and t5=0 and t1t=0 }");
  isl::map cschedule(
      ctx,
      "[_c_0,_c_1] -> { c[i0,i1,i2,i3] -> [t0,t1,t1t,t2,t3,t4,t5]: t0=i0 and "
      "t1=i1 and t2=i2 and t3=i3 "
      "and t4=0 and t5=0 and t1t=1 }");

  t.DisplayC(pschedule.release(), cschedule.release());

  LOG(INFO) << "shape:";
  auto shape = t.GetProducerAdjustedShape();
  for (int i = 0; i < shape.size(); i++) {
    LOG(INFO) << shape[i];
  }
}

}  // namespace poly
}  // namespace cinn
