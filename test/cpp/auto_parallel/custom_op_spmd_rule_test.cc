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

#include "paddle/phi/api/ext/op_meta_info.h"
#include "paddle/phi/api/ext/spmd_infer.h"
#include "test/cpp/auto_parallel/spmd_rule_test_util.h"

namespace paddle {
namespace distributed {
namespace auto_parallel {
TEST(CustomOp, Ctor) {
  // test with concat rule
  std::vector<int64_t> mesh_shape = {2, 2};
  std::vector<int64_t> process_ids = {0, 1, 2, 3};
  std::vector<std::string> dim_names = {"x", "y"};
  ProcessMesh process_mesh(mesh_shape, process_ids, dim_names);

  std::vector<std::vector<int64_t>> shapes = {
      {16, 16, 16}, {4, 16, 16}, {2, 16, 16}};
  std::vector<std::vector<int64_t>> dim_mappings = {
      {-1, 0, 1}, {-1, 1, 0}, {-1, -1, 0}};
  std::vector<std::vector<int64_t>> partial_status = {{}, {}, {1}};

  auto build_inputs = [&] {
    std::vector<phi::distributed::DistMetaTensor> inputs;
    for (int i = 0; i < 3; i++) {
      auto t_dist_attr = TensorDistAttr();
      t_dist_attr.set_process_mesh(process_mesh);
      t_dist_attr.set_dims_mapping(dim_mappings[i]);
      t_dist_attr.set_dynamic_dims({false, false, false});
      auto input = phi::distributed::DistMetaTensor(
          common::make_ddim(shapes[i]), t_dist_attr);
      inputs.push_back(input);
    }
    return inputs;
  };

  // test 1, inputs are aligned according to cost, and partial status is cleared
  auto inputs = build_inputs();

  auto forward_spmd_func =
      PD_INFER_SPMD_RULE(phi::distributed::ConcatInferSpmd);
  int axis = 0;
  std::vector<CustomSpmdInferTensorArg> infer_inputs = {inputs};
  std::vector<CustomSpmdInferAttrArg> attrs = {axis};

  auto inferred_dist_attrs = forward_spmd_func(infer_inputs, attrs);
  // list of tensor => single tensor
  EXPECT_EQ(inferred_dist_attrs.first.size(), static_cast<size_t>(1));
  EXPECT_EQ(inferred_dist_attrs.second.size(), static_cast<size_t>(1));
  EXPECT_TRUE(
      paddle::holds_alternative<std::vector<phi::distributed::TensorDistAttr>>(
          inferred_dist_attrs.first[0]));
  EXPECT_TRUE(paddle::holds_alternative<phi::distributed::TensorDistAttr>(
      inferred_dist_attrs.second[0]));
  auto& inputs_infer1 =
      PADDLE_GET_CONST(std::vector<phi::distributed::TensorDistAttr>,
                       inferred_dist_attrs.first[0]);

  for (auto e : inputs_infer1) {
    check_dim_mapping(e, {-1, 1, 0});
    check_partial_dims(e, {});
  }
  check_dim_mapping(inferred_dist_attrs.second[0], {-1, 1, 0});
  check_partial_dims(inferred_dist_attrs.second[0], {});
}

TEST(CustomOp, Register) {
  OpMetaInfoBuilder builder("test_custom_op_spmd", 0);
  auto iter = OpMetaInfoMap::Instance().GetMap().find("test_custom_op_spmd");
  EXPECT_TRUE(iter != OpMetaInfoMap::Instance().GetMap().end());
  EXPECT_TRUE(OpMetaInfoHelper::GetInferSpmdFn(iter->second[0]) == nullptr);
  builder.SetInferSpmdFn(PD_INFER_SPMD_RULE(phi::distributed::ConcatInferSpmd));
  EXPECT_TRUE(OpMetaInfoHelper::GetInferSpmdFn(iter->second[0]) != nullptr);
}
}  // namespace auto_parallel
}  // namespace distributed
}  // namespace paddle
