/* Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "test/cpp/auto_parallel/spmd_rule_test_util.h"

namespace paddle {
namespace distributed {
namespace auto_parallel {
TEST(FusedRmsNormSPMDRule, test_fused_rms_norm) {
  // build input data class
  std::vector<int64_t> x_shape = {64, 32, 1024};
  std::vector<int64_t> scale_shape = {1024};
  std::vector<int64_t> variance_shape = {64, 32};

  std::vector<int64_t> mesh_shape = {2, 3};
  std::vector<int64_t> process_ids = {0, 1, 2, 3, 4, 5};
  std::vector<std::string> dim_names = {"x", "y"};
  ProcessMesh process_mesh(mesh_shape, process_ids, dim_names);

  TensorDistAttr x_dist_attr = TensorDistAttr();
  x_dist_attr.set_process_mesh(process_mesh);
  x_dist_attr.set_dims_mapping(std::vector<int64_t>({1, -1, -1}));
  x_dist_attr.set_dynamic_dims(std::vector<bool>({false, false, false}));

  TensorDistAttr scale_dist_attr = TensorDistAttr();
  scale_dist_attr.set_process_mesh(process_mesh);
  scale_dist_attr.set_dims_mapping(std::vector<int64_t>({-1}));
  scale_dist_attr.set_dynamic_dims(std::vector<bool>({false}));

  x_dist_attr.set_dims_mapping({1, -1, -1});
  scale_dist_attr.set_dims_mapping({-1});

  phi::distributed::DistMetaTensor x(common::make_ddim(x_shape), x_dist_attr);
  phi::distributed::DistMetaTensor scale(common::make_ddim(scale_shape),
                                         scale_dist_attr);
  auto inferred_dist_attrs = phi::distributed::RmsNormInferSpmd(x, scale, 0.5);

  size_t input_size = 2;
  size_t output_size = 2;
  EXPECT_EQ(inferred_dist_attrs.first.size(), input_size);
  EXPECT_EQ(inferred_dist_attrs.second.size(), output_size);
  check_dim_mapping(inferred_dist_attrs.first[0], {1, -1, -1});
  check_dim_mapping(inferred_dist_attrs.first[1], {-1});
  check_dim_mapping(inferred_dist_attrs.second[0], {1, -1, -1});
  check_dim_mapping(inferred_dist_attrs.second[1], {1, -1});

  VLOG(4) << "test1 done.";

  x_dist_attr.set_dims_mapping({1, 0, -1});
  scale_dist_attr.set_dims_mapping({0});
  x = phi::distributed::DistMetaTensor(common::make_ddim(x_shape), x_dist_attr);
  scale = phi::distributed::DistMetaTensor(common::make_ddim(scale_shape),
                                           scale_dist_attr);

  inferred_dist_attrs = phi::distributed::RmsNormInferSpmd(x, scale, 0.5);
  check_dim_mapping(inferred_dist_attrs.first[0], {1, 0, -1});
  check_dim_mapping(inferred_dist_attrs.first[1], {-1});
  check_dim_mapping(inferred_dist_attrs.second[0], {1, 0, -1});
  check_dim_mapping(inferred_dist_attrs.second[1], {1, 0});
  VLOG(4) << "test2 done.";

  TensorDistAttr out_dist_attr = TensorDistAttr();
  out_dist_attr.set_process_mesh(process_mesh);
  out_dist_attr.set_dims_mapping(std::vector<int64_t>({0, 1, -1}));
  out_dist_attr.set_dynamic_dims(std::vector<bool>({false, false, false}));
  phi::distributed::DistMetaTensor out(common::make_ddim(x_shape),
                                       out_dist_attr);

  TensorDistAttr invvar_dist_attr = TensorDistAttr();
  invvar_dist_attr.set_process_mesh(process_mesh);
  invvar_dist_attr.set_dims_mapping(std::vector<int64_t>({0, -1}));
  invvar_dist_attr.set_dynamic_dims(std::vector<bool>({false, false}));
  phi::distributed::DistMetaTensor invvar(common::make_ddim(variance_shape),
                                          invvar_dist_attr);

  inferred_dist_attrs =
      phi::distributed::RmsNormInferSpmdReverse(x, scale, out, invvar, 0.5);
  check_dim_mapping(inferred_dist_attrs.first[0], {0, 1, -1});
  check_dim_mapping(inferred_dist_attrs.first[1], {-1});
  check_dim_mapping(inferred_dist_attrs.second[0], {0, 1, -1});
  check_dim_mapping(inferred_dist_attrs.second[1], {0, 1});
  VLOG(4) << "test3 done.";

  x_dist_attr.set_dims_mapping({0, 1, -1});
  x = phi::distributed::DistMetaTensor(common::make_ddim(x_shape), x_dist_attr);
  inferred_dist_attrs =
      phi::distributed::RmsNormGradInferSpmd(x, scale, invvar, out, 0.5);

  check_dim_mapping(inferred_dist_attrs.first[0], {0, 1, -1});
  check_dim_mapping(inferred_dist_attrs.first[1], {-1});
  check_dim_mapping(inferred_dist_attrs.first[2], {0, 1});
  check_dim_mapping(inferred_dist_attrs.first[3], {0, 1, -1});
  check_dim_mapping(inferred_dist_attrs.second[0], {0, 1, -1});
  check_dim_mapping(inferred_dist_attrs.second[1], {-1});
  check_partial_dims(inferred_dist_attrs.second[1], {0, 1});
}
}  // namespace auto_parallel
}  // namespace distributed
}  // namespace paddle
