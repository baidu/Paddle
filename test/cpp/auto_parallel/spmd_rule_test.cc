/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/common/scalar.h"
#include "test/cpp/auto_parallel/spmd_rule_test_util.h"

namespace paddle {
namespace distributed {
namespace auto_parallel {

TEST(MatmulSPMDRule, Ctor) {
  // build input data class
  std::vector<int64_t> x_shape = {64, 32};
  std::vector<int64_t> y_shape = {32, 48};

  std::vector<int64_t> mesh_shape = {2, 3};
  std::vector<int64_t> process_ids = {0, 1, 2, 3, 4, 5};
  std::vector<std::string> dim_names = {"x", "y"};
  ProcessMesh process_mesh(mesh_shape, process_ids, dim_names);

  TensorDistAttr x_dist_attr = TensorDistAttr();
  x_dist_attr.set_process_mesh(process_mesh);
  x_dist_attr.set_dims_mapping(std::vector<int64_t>({1, -1}));
  x_dist_attr.set_dynamic_dims(std::vector<bool>({false, false}));

  TensorDistAttr y_dist_attr = TensorDistAttr();
  y_dist_attr.set_process_mesh(process_mesh);
  y_dist_attr.set_dims_mapping(std::vector<int64_t>({-1, -1}));
  y_dist_attr.set_dynamic_dims(std::vector<bool>({false, false}));

  size_t input_size = 2;
  size_t output_size = 1;

  phi::distributed::DistMetaTensor x(common::make_ddim(x_shape), x_dist_attr);
  phi::distributed::DistMetaTensor y(common::make_ddim(y_shape), y_dist_attr);

  auto matmul_spmd_rule =
      phi::distributed::SpmdRuleFactory::Instance().GetSpmdRule("matmul");

  // mk[1, -1],kn[-1, -1] --> mk[1, -1],kn[-1, -1] = nm[1, -1] partial[]
  phi::distributed::InferSpmdContext ctx(
      {x, y}, {/*trans_x=*/false, /*trans_x=*/false});
  auto inferred_dist_attrs = matmul_spmd_rule.InferForward(ctx);

  EXPECT_EQ(inferred_dist_attrs.first.size(), input_size);
  EXPECT_EQ(inferred_dist_attrs.second.size(), output_size);
  check_dim_mapping(inferred_dist_attrs.first[0], {1, -1});
  check_dim_mapping(inferred_dist_attrs.first[1], {-1, -1});
  check_dim_mapping(inferred_dist_attrs.second[0], {1, -1});
  EXPECT_EQ(is_partial(inferred_dist_attrs.second[0]), false);
  VLOG(4) << "test1 done." << std::endl << std::endl << std::endl;

  // mk[-1,-1],kn[-1,0] --> mk[-1,-1],kn[-1,0] = nm[-1,0] partial[]
  x_dist_attr.set_dims_mapping({-1, -1});
  y_dist_attr.set_dims_mapping({-1, 0});
  x = phi::distributed::DistMetaTensor(common::make_ddim(x_shape), x_dist_attr);
  y = phi::distributed::DistMetaTensor(common::make_ddim(y_shape), y_dist_attr);
  ctx = phi::distributed::InferSpmdContext(
      {x, y}, {/*trans_x=*/false, /*trans_x=*/false});
  inferred_dist_attrs = matmul_spmd_rule.InferForward(ctx);
  check_dim_mapping(inferred_dist_attrs.first[0], {-1, -1});
  check_dim_mapping(inferred_dist_attrs.first[1], {-1, 0});
  check_dim_mapping(inferred_dist_attrs.second[0], {-1, 0});
  EXPECT_EQ(is_partial(inferred_dist_attrs.second[0]), false);
  VLOG(4) << "test2 done." << std::endl << std::endl << std::endl;
  // mk[1, 0],kn[-1,-1] --> mk[1, 0],kn[0, -1] = nm[1, -1] partial[0]: done
  x_dist_attr.set_dims_mapping({1, 0});
  y_dist_attr.set_dims_mapping({-1, -1});
  x = phi::distributed::DistMetaTensor(common::make_ddim(x_shape), x_dist_attr);
  y = phi::distributed::DistMetaTensor(common::make_ddim(y_shape), y_dist_attr);
  ctx = phi::distributed::InferSpmdContext(
      {x, y}, {/*trans_x=*/false, /*trans_x=*/false});
  inferred_dist_attrs = matmul_spmd_rule.InferForward(ctx);
  check_dim_mapping(inferred_dist_attrs.first[0], {1, 0});
  check_dim_mapping(inferred_dist_attrs.first[1], {0, -1});
  check_dim_mapping(inferred_dist_attrs.second[0], {1, -1});
  EXPECT_EQ(is_partial(inferred_dist_attrs.second[0]), true);
  check_partial_dims(inferred_dist_attrs.second[0], {0});
  VLOG(4) << "test3 done." << std::endl << std::endl << std::endl;

  // mk[-1,-1],kn[1,0] --> mk[-1, 1],kn[1, 0] = nm[-1, 0] partial[1]: done
  x_dist_attr.set_dims_mapping({-1, -1});
  y_dist_attr.set_dims_mapping({1, 0});
  x = phi::distributed::DistMetaTensor(common::make_ddim(x_shape), x_dist_attr);
  y = phi::distributed::DistMetaTensor(common::make_ddim(y_shape), y_dist_attr);
  ctx = phi::distributed::InferSpmdContext(
      {x, y}, {/*trans_x=*/false, /*trans_x=*/false});
  inferred_dist_attrs = matmul_spmd_rule.InferForward(ctx);
  check_dim_mapping(inferred_dist_attrs.first[0], {-1, 1});
  check_dim_mapping(inferred_dist_attrs.first[1], {1, 0});
  check_dim_mapping(inferred_dist_attrs.second[0], {-1, 0});
  EXPECT_EQ(is_partial(inferred_dist_attrs.second[0]), true);
  check_partial_dims(inferred_dist_attrs.second[0], {1});
  VLOG(4) << "test4 done." << std::endl << std::endl << std::endl;

  // abcmk[1, 0, -1, -1],kn[-1, -1] --> abcmk[1, 0, -1, -1],kn[-1, -1] =
  // abcmn[1, 0, -1, -1] partial[]: done
  x_shape = {512, 48, 64, 32};
  x_dist_attr.set_dims_mapping({0, 1, -1, -1});
  y_dist_attr.set_dims_mapping({-1, -1});
  x = phi::distributed::DistMetaTensor(common::make_ddim(x_shape), x_dist_attr);
  y = phi::distributed::DistMetaTensor(common::make_ddim(y_shape), y_dist_attr);
  ctx = phi::distributed::InferSpmdContext(
      {x, y}, {/*trans_x=*/false, /*trans_x=*/false});
  inferred_dist_attrs = matmul_spmd_rule.InferForward(ctx);
  check_dim_mapping(inferred_dist_attrs.first[0], {0, 1, -1, -1});
  check_dim_mapping(inferred_dist_attrs.first[1], {-1, -1});
  check_dim_mapping(inferred_dist_attrs.second[0], {0, 1, -1, -1});
  EXPECT_EQ(is_partial(inferred_dist_attrs.second[0]), false);
  VLOG(4) << "test5 done." << std::endl << std::endl << std::endl;

  // abcmk[1, -1, -1, 0],kn[-1, -1] --> abcmk[1, -1, -1, 0],kn[0, -1] = abcmn[1,
  // -1, -1, -1] partial[0]: done
  x_dist_attr.set_dims_mapping({1, -1, -1, 0});
  y_dist_attr.set_dims_mapping({-1, -1});
  x = phi::distributed::DistMetaTensor(common::make_ddim(x_shape), x_dist_attr);
  y = phi::distributed::DistMetaTensor(common::make_ddim(y_shape), y_dist_attr);
  ctx = phi::distributed::InferSpmdContext(
      {x, y}, {/*trans_x=*/false, /*trans_x=*/false});
  inferred_dist_attrs = matmul_spmd_rule.InferForward(ctx);
  check_dim_mapping(inferred_dist_attrs.first[0], {1, -1, -1, 0});
  check_dim_mapping(inferred_dist_attrs.first[1], {0, -1});
  check_dim_mapping(inferred_dist_attrs.second[0], {1, -1, -1, -1});
  EXPECT_EQ(is_partial(inferred_dist_attrs.second[0]), true);
  check_partial_dims(inferred_dist_attrs.second[0], {0});
  VLOG(4) << "test6 done." << std::endl << std::endl << std::endl;

  // abcmk[1, -1, -1, 0], kn[-1, -1] --> abcmk[1, -1, -1, 0],kn[-1, -1] =
  // abcmn[1, -1, 0, -1] partial[]: done
  x_dist_attr.set_dims_mapping({1, -1, -1, 0});
  y_dist_attr.set_dims_mapping({-1, -1});
  x = phi::distributed::DistMetaTensor(common::make_ddim(x_shape), x_dist_attr);
  y = phi::distributed::DistMetaTensor(common::make_ddim(y_shape), y_dist_attr);
  ctx = phi::distributed::InferSpmdContext(
      {x, y}, {/*trans_x=*/true, /*trans_x=*/false});
  inferred_dist_attrs = matmul_spmd_rule.InferForward(ctx);

  check_dim_mapping(inferred_dist_attrs.first[0], {1, -1, -1, 0});
  check_dim_mapping(inferred_dist_attrs.first[1], {-1, -1});
  check_dim_mapping(inferred_dist_attrs.second[0], {1, -1, 0, -1});
  EXPECT_EQ(is_partial(inferred_dist_attrs.second[0]), false);

  VLOG(4) << "test7 done." << std::endl << std::endl << std::endl;

  // abcmk[-1, -1, -1, -1], kn[1, 0] --> abcmk[-1, -1, -1, 0],kn[1, 0] =
  // abcmn[-1, -1, -1, 1] partial[0]: done
  x_dist_attr.set_dims_mapping({-1, -1, -1, -1});
  y_dist_attr.set_dims_mapping({1, 0});
  x = phi::distributed::DistMetaTensor(common::make_ddim(x_shape), x_dist_attr);
  y = phi::distributed::DistMetaTensor(common::make_ddim(y_shape), y_dist_attr);
  ctx = phi::distributed::InferSpmdContext(
      {x, y}, {/*trans_x=*/false, /*trans_x=*/true});
  inferred_dist_attrs = matmul_spmd_rule.InferForward(ctx);
  check_dim_mapping(inferred_dist_attrs.first[0], {-1, -1, -1, 0});
  check_dim_mapping(inferred_dist_attrs.first[1], {1, 0});
  check_dim_mapping(inferred_dist_attrs.second[0], {-1, -1, -1, 1});
  EXPECT_EQ(is_partial(inferred_dist_attrs.second[0]), true);
  check_partial_dims(inferred_dist_attrs.second[0], {0});
  clean_partial_dims(&inferred_dist_attrs.second[0], {0});
  EXPECT_EQ(is_partial(inferred_dist_attrs.second[0]), false);
  VLOG(4) << "test8 done." << std::endl << std::endl << std::endl;

  // abcmk[-1, -1, 0, 1]+trans_x=true, kn[1, 0]+trans_y=true --> abcmk[-1, -1,
  // 0, -1],kn[-1, 0] = abcmn[-1, -1, 1, -1] partial[0]: done
  x_dist_attr.set_dims_mapping({-1, -1, 0, 1});
  y_dist_attr.set_dims_mapping({1, 0});
  x = phi::distributed::DistMetaTensor(common::make_ddim(x_shape), x_dist_attr);
  y = phi::distributed::DistMetaTensor(common::make_ddim(y_shape), y_dist_attr);
  ctx = phi::distributed::InferSpmdContext(
      {x, y}, {/*trans_x=*/true, /*trans_x=*/true});
  inferred_dist_attrs = matmul_spmd_rule.InferForward(ctx);

  check_dim_mapping(inferred_dist_attrs.first[0], {-1, -1, 0, 1});
  check_dim_mapping(inferred_dist_attrs.first[1],
                    {-1, 0});  // conflict and should be changed to [-1, 0]
  check_dim_mapping(inferred_dist_attrs.second[0], {-1, -1, 1, -1});
  check_partial_dims(inferred_dist_attrs.second[0], {0});

  clean_partial_status(&inferred_dist_attrs.second[0]);
  EXPECT_EQ(is_partial(inferred_dist_attrs.second[0]), false);
  EXPECT_ANY_THROW(set_partial_status(&inferred_dist_attrs.second[0], {1}));
  VLOG(4) << "test9 done." << std::endl << std::endl << std::endl;

  // abcmk[-1, -1, 1, 0], kn[1, 0] --> abcmk[-1, -1, -1, 0],kn[1, 0] =
  // abcmn[-1, -1, -1, 1] partial[0]: done
  x_dist_attr.set_dims_mapping({-1, -1, 1, 0});
  y_dist_attr.set_dims_mapping({1, 0});
  x = phi::distributed::DistMetaTensor(common::make_ddim(x_shape), x_dist_attr);
  y = phi::distributed::DistMetaTensor(common::make_ddim(y_shape), y_dist_attr);
  ctx = phi::distributed::InferSpmdContext(
      {x, y}, {/*trans_x=*/true, /*trans_x=*/true});
  EXPECT_ANY_THROW(inferred_dist_attrs = matmul_spmd_rule.InferForward(ctx));
  // Error
  VLOG(4) << "test10 done." << std::endl << std::endl << std::endl;

  // abcmk[-1, -1, 1, 0], kn[0, 1] --> abcmk[-1, -1, 1, 0],kn[0, 1] =
  // abcmn[-1, -1, 1, -1] partial[0]:
  x_dist_attr.set_dims_mapping({-1, -1, 0, 1});
  y_dist_attr.set_dims_mapping({1, 0});
  x = phi::distributed::DistMetaTensor(common::make_ddim(x_shape), x_dist_attr);
  y = phi::distributed::DistMetaTensor(common::make_ddim(y_shape), y_dist_attr);
  ctx = phi::distributed::InferSpmdContext(
      {x, y}, {/*trans_x=*/true, /*trans_x=*/true});
  inferred_dist_attrs = matmul_spmd_rule.InferForward(ctx);
  check_dim_mapping(inferred_dist_attrs.second[0], {-1, -1, 1, -1});
  EXPECT_EQ(is_partial(inferred_dist_attrs.second[0]), true);
  check_partial_dims(inferred_dist_attrs.second[0], {0});

  // try to clean partial on a dim which is not partial
  EXPECT_ANY_THROW(clean_partial_dims(&inferred_dist_attrs.second[0], {1}));
  // try to clean partial on a dims which is sharded
  EXPECT_ANY_THROW(set_partial_status(&inferred_dist_attrs.second[0], {1}));

  // clean partial and then re-set again
  clean_partial_dims(&inferred_dist_attrs.second[0], {0});
  EXPECT_EQ(is_partial(inferred_dist_attrs.second[0]), false);
  set_partial_status(&inferred_dist_attrs.second[0], {0});
  EXPECT_EQ(is_partial(inferred_dist_attrs.second[0]), true);
  check_partial_dims(inferred_dist_attrs.second[0], {0});
  VLOG(4) << "test11 done." << std::endl << std::endl << std::endl;
}

TEST(LayerNormSPMDRule, Ctor) {
  // build input data class
  std::vector<int64_t> x_shape = {64, 32, 1024};
  std::vector<int64_t> scale_shape = {1024};
  std::vector<int64_t> bias_shape = {1024};

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

  TensorDistAttr bias_dist_attr = TensorDistAttr();
  bias_dist_attr.set_process_mesh(process_mesh);
  bias_dist_attr.set_dims_mapping(std::vector<int64_t>({-1}));
  bias_dist_attr.set_dynamic_dims(std::vector<bool>({false}));

  float epsilon = 1e-5;
  int begin_norm_axis = 2;

  auto layer_norm_rule =
      phi::distributed::SpmdRuleFactory::Instance().GetSpmdRule("layer_norm");

  // ijk[1, -1, -1], k[-1], k[-1] --> ijk[1, -1, -1], z[1], z[1], z=ij,
  // begin_norm_axis=2
  begin_norm_axis = 2;
  x_dist_attr.set_dims_mapping({1, -1, -1});
  scale_dist_attr.set_dims_mapping({-1});
  bias_dist_attr.set_dims_mapping({-1});
  phi::distributed::DistMetaTensor x(common::make_ddim(x_shape), x_dist_attr);
  phi::distributed::DistMetaTensor scale(common::make_ddim(scale_shape),
                                         scale_dist_attr);
  phi::distributed::DistMetaTensor bias(common::make_ddim(bias_shape),
                                        bias_dist_attr);
  phi::distributed::InferSpmdContext ctx({x, scale, bias},
                                         {epsilon, begin_norm_axis});
  auto inferred_dist_attrs = layer_norm_rule.InferForward(ctx);

  size_t input_size = 3;
  size_t output_size = 3;
  EXPECT_EQ(inferred_dist_attrs.first.size(), input_size);
  EXPECT_EQ(inferred_dist_attrs.second.size(), output_size);
  check_dim_mapping(inferred_dist_attrs.first[0], {1, -1, -1});
  check_dim_mapping(inferred_dist_attrs.first[1], {-1});
  check_dim_mapping(inferred_dist_attrs.first[2], {-1});
  check_dim_mapping(inferred_dist_attrs.second[0], {1, -1, -1});
  check_dim_mapping(inferred_dist_attrs.second[1], {1, -1});
  check_dim_mapping(inferred_dist_attrs.second[2], {1, -1});
  VLOG(4) << "test1 done.";

  // ijk[1, 0, -1],k[0],k[0] --> ijk[1, -1, -1],z[1, 0],z[1, 0],
  // begin_norm_axis=2
  begin_norm_axis = 2;
  x_dist_attr.set_dims_mapping({1, 0, -1});
  scale_dist_attr.set_dims_mapping({0});
  bias_dist_attr.set_dims_mapping({0});
  x = phi::distributed::DistMetaTensor(common::make_ddim(x_shape), x_dist_attr);
  scale = phi::distributed::DistMetaTensor(common::make_ddim(scale_shape),
                                           scale_dist_attr);
  bias = phi::distributed::DistMetaTensor(common::make_ddim(bias_shape),
                                          bias_dist_attr);
  ctx = phi::distributed::InferSpmdContext({x, scale, bias},
                                           {epsilon, begin_norm_axis});
  inferred_dist_attrs = layer_norm_rule.InferForward(ctx);

  check_dim_mapping(inferred_dist_attrs.first[0], {1, 0, -1});
  check_dim_mapping(inferred_dist_attrs.first[1], {-1});
  check_dim_mapping(inferred_dist_attrs.first[2], {-1});
  check_dim_mapping(inferred_dist_attrs.second[0], {1, 0, -1});
  check_dim_mapping(inferred_dist_attrs.second[1], {1, 0});
  check_dim_mapping(inferred_dist_attrs.second[2], {1, 0});
  VLOG(4) << "test2 done.";

  // ijk[0, -1, -1],y[-1],y[1] --> ijk[0, -1, -1], i[0], i[0], y=jk,
  // begin_norm_axis=1
  begin_norm_axis = 1;
  x_dist_attr.set_dims_mapping({0, -1, -1});
  scale_dist_attr.set_dims_mapping({-1});
  bias_dist_attr.set_dims_mapping({1});
  x = phi::distributed::DistMetaTensor(common::make_ddim(x_shape), x_dist_attr);
  scale = phi::distributed::DistMetaTensor(common::make_ddim(scale_shape),
                                           scale_dist_attr);
  bias = phi::distributed::DistMetaTensor(common::make_ddim(bias_shape),
                                          bias_dist_attr);
  ctx = phi::distributed::InferSpmdContext({x, scale, bias},
                                           {epsilon, begin_norm_axis});
  inferred_dist_attrs = layer_norm_rule.InferForward(ctx);

  check_dim_mapping(inferred_dist_attrs.first[0], {0, -1, -1});
  check_dim_mapping(inferred_dist_attrs.first[1], {-1});
  check_dim_mapping(inferred_dist_attrs.first[2], {-1});
  check_dim_mapping(inferred_dist_attrs.second[0], {0, -1, -1});
  check_dim_mapping(inferred_dist_attrs.second[1], {0});
  check_dim_mapping(inferred_dist_attrs.second[2], {0});
  VLOG(4) << "test3 done.";
}

TEST(MatmulSPMDRuleInferBackward, Ctor) {
  // build input data class
  std::vector<int64_t> x_shape = {512, 1024, 64, 32};
  std::vector<int64_t> y_shape = {512, 1, 32, 48};
  std::vector<int64_t> out_shape = {512, 1024, 64, 48};

  std::vector<int64_t> mesh_shape = {2, 3};
  std::vector<int64_t> process_ids = {0, 1, 2, 3, 4, 5};
  std::vector<std::string> dim_names = {"x", "y"};
  ProcessMesh process_mesh(mesh_shape, process_ids, dim_names);

  TensorDistAttr x_dist_attr = TensorDistAttr();
  x_dist_attr.set_process_mesh(process_mesh);
  x_dist_attr.set_dims_mapping(
      std::vector<int64_t>({-1, 1, 0, -1}));  // no affect
  x_dist_attr.set_dynamic_dims(std::vector<bool>({false, false}));

  TensorDistAttr y_dist_attr = TensorDistAttr();
  y_dist_attr.set_process_mesh(process_mesh);
  y_dist_attr.set_dims_mapping(
      std::vector<int64_t>({0, 1, -1, -1}));  // no affect
  y_dist_attr.set_dynamic_dims(std::vector<bool>({false, false}));

  TensorDistAttr out_dist_attr = TensorDistAttr();
  out_dist_attr.set_process_mesh(process_mesh);
  out_dist_attr.set_dims_mapping(std::vector<int64_t>({-1, -1, 1, -1}));
  out_dist_attr.set_dynamic_dims(std::vector<bool>({false, false}));
  out_dist_attr.set_partial_status(std::vector<int64_t>({0}));

  phi::distributed::DistMetaTensor x(common::make_ddim(x_shape), x_dist_attr);
  phi::distributed::DistMetaTensor y(common::make_ddim(y_shape), y_dist_attr);
  phi::distributed::DistMetaTensor out(common::make_ddim(out_shape),
                                       out_dist_attr);

  auto matmul_spmd_rule =
      phi::distributed::SpmdRuleFactory::Instance().GetSpmdRule("matmul");

  // TODO(zyc) update in future: propagate the partial in inferbackward
  // abmn[-1, -1, 1, -1] + partial[0] --> abmk[-1, -1, 1, -1], a1kn[-1, -1, -1,
  // -1]
  phi::distributed::InferSpmdContext ctx(
      {x, y, out}, {/*trans_x=*/false, /*trans_x=*/false});
  auto inferred_dist_attrs = matmul_spmd_rule.InferBackward(ctx);

  size_t input_size = 2;
  size_t output_size = 1;
  EXPECT_EQ(inferred_dist_attrs.first.size(), input_size);
  EXPECT_EQ(inferred_dist_attrs.second.size(), output_size);

  check_dim_mapping(inferred_dist_attrs.first[0], {-1, -1, 1, -1});
  check_dim_mapping(inferred_dist_attrs.first[1], {-1, -1, -1, -1});
  check_dim_mapping(inferred_dist_attrs.second[0], {-1, -1, 1, -1});
  EXPECT_EQ(is_partial(inferred_dist_attrs.first[0]), false);
  EXPECT_EQ(is_partial(inferred_dist_attrs.first[1]), false);
  EXPECT_EQ(is_partial(inferred_dist_attrs.second[0]), true);
  VLOG(4) << "test1 done." << std::endl << std::endl << std::endl;
}

TEST(ReplicatedSPMDRule, Ctor) {
  // build input data class
  std::vector<int64_t> x_shape = {10, 10, 32, 48};
  std::vector<int64_t> y_shape = {32, 48};
  std::vector<int64_t> out1_shape = {10, 10, 32, 48};
  std::vector<int64_t> out2_shape = {10, 32, 48};

  std::vector<int64_t> mesh_shape = {2, 3};
  std::vector<int64_t> process_ids = {0, 1, 2, 3, 4, 5};
  std::vector<std::string> dim_names = {"x", "y"};
  ProcessMesh process_mesh(mesh_shape, process_ids, dim_names);

  TensorDistAttr x_dist_attr = TensorDistAttr();
  x_dist_attr.set_process_mesh(process_mesh);
  x_dist_attr.set_dims_mapping(std::vector<int64_t>({-1, 1, -1, -1}));
  x_dist_attr.set_dynamic_dims(std::vector<bool>({false, false}));

  TensorDistAttr y_dist_attr = TensorDistAttr();
  y_dist_attr.set_process_mesh(process_mesh);
  y_dist_attr.set_dims_mapping(std::vector<int64_t>({0, -1}));  // no affect
  y_dist_attr.set_dynamic_dims(std::vector<bool>({false, false}));

  TensorDistAttr out1_dist_attr = TensorDistAttr();
  out1_dist_attr.set_process_mesh(process_mesh);
  out1_dist_attr.set_dims_mapping(std::vector<int64_t>({-1, -1, 1, -1}));
  out1_dist_attr.set_dynamic_dims(std::vector<bool>({false, false}));

  TensorDistAttr out2_dist_attr = TensorDistAttr();
  out2_dist_attr.set_process_mesh(process_mesh);
  out2_dist_attr.set_dims_mapping(std::vector<int64_t>({-1, 1, -1}));
  out2_dist_attr.set_dynamic_dims(std::vector<bool>({false, false}));

  phi::distributed::DistMetaTensor x(common::make_ddim(x_shape), x_dist_attr);
  phi::distributed::DistMetaTensor y(common::make_ddim(y_shape), y_dist_attr);
  phi::distributed::DistMetaTensor out1(common::make_ddim(out1_shape),
                                        out1_dist_attr);
  phi::distributed::DistMetaTensor out2(common::make_ddim(out2_shape),
                                        out2_dist_attr);

  // 2 inputs 2 outputs
  // call in vector arguments format
  auto inferred_dist_attrs_st =
      phi::distributed::ReplicatedInferSpmd({&x, &y}, {&out1, &out2});
  // call in variadic arguments format
  auto inferred_dist_attrs_dy =
      phi::distributed::VariadicReplicatedInferSpmd(x, y, &out1, &out2);

  size_t input_size = 2;
  size_t output_size = 2;
  EXPECT_EQ(inferred_dist_attrs_st.first.size(), input_size);
  EXPECT_EQ(inferred_dist_attrs_st.second.size(), output_size);
  EXPECT_EQ(inferred_dist_attrs_dy.first.size(), input_size);
  EXPECT_EQ(inferred_dist_attrs_dy.second.size(), output_size);

  check_dim_mapping(inferred_dist_attrs_st.first[0], {-1, -1, -1, -1});
  check_dim_mapping(inferred_dist_attrs_st.first[1], {-1, -1});
  check_dim_mapping(inferred_dist_attrs_st.second[0], {-1, -1, -1, -1});
  check_dim_mapping(inferred_dist_attrs_st.second[1], {-1, -1, -1});
  EXPECT_EQ(is_partial(inferred_dist_attrs_st.first[0]), false);
  EXPECT_EQ(is_partial(inferred_dist_attrs_st.first[1]), false);
  EXPECT_EQ(is_partial(inferred_dist_attrs_st.second[0]), false);
  EXPECT_EQ(is_partial(inferred_dist_attrs_st.second[1]), false);
  EXPECT_EQ(inferred_dist_attrs_st.first, inferred_dist_attrs_dy.first);
  EXPECT_EQ(inferred_dist_attrs_st.second, inferred_dist_attrs_dy.second);
  VLOG(4) << "test1 done." << std::endl << std::endl << std::endl;

  // 3 inputs 1 outputs
  // call in vector arguments format
  inferred_dist_attrs_st =
      phi::distributed::ReplicatedInferSpmd({&x, &y, &out1}, {&out2});
  // call in variadic arguments format
  inferred_dist_attrs_dy =
      phi::distributed::VariadicReplicatedInferSpmd(x, y, out1, &out2);

  input_size = 3;
  output_size = 1;
  EXPECT_EQ(inferred_dist_attrs_st.first.size(), input_size);
  EXPECT_EQ(inferred_dist_attrs_st.second.size(), output_size);
  EXPECT_EQ(inferred_dist_attrs_dy.first.size(), input_size);
  EXPECT_EQ(inferred_dist_attrs_dy.second.size(), output_size);
  check_dim_mapping(inferred_dist_attrs_dy.first[0], {-1, -1, -1, -1});
  check_dim_mapping(inferred_dist_attrs_dy.first[1], {-1, -1});
  check_dim_mapping(inferred_dist_attrs_dy.first[2], {-1, -1, -1, -1});
  check_dim_mapping(inferred_dist_attrs_dy.second[0], {-1, -1, -1});
  EXPECT_EQ(inferred_dist_attrs_st.first, inferred_dist_attrs_dy.first);
  EXPECT_EQ(inferred_dist_attrs_st.second, inferred_dist_attrs_dy.second);
  VLOG(4) << "test2 done." << std::endl << std::endl << std::endl;

  // 1 inputs 3 outputs backward
  // call in vector arguments format
  inferred_dist_attrs_st =
      phi::distributed::ReplicatedInferSpmdReverse({&x}, {&y, &out1, &out2});
  // call in variadic arguments format
  inferred_dist_attrs_dy =
      phi::distributed::VariadicReplicatedInferSpmdReverse(x, &y, &out1, &out2);

  input_size = 1;
  output_size = 3;
  EXPECT_EQ(inferred_dist_attrs_st.first.size(), input_size);
  EXPECT_EQ(inferred_dist_attrs_st.second.size(), output_size);
  EXPECT_EQ(inferred_dist_attrs_dy.first.size(), input_size);
  EXPECT_EQ(inferred_dist_attrs_dy.second.size(), output_size);

  check_dim_mapping(inferred_dist_attrs_dy.first[0], {-1, -1, -1, -1});
  check_dim_mapping(inferred_dist_attrs_dy.second[0], {-1, -1});
  check_dim_mapping(inferred_dist_attrs_dy.second[1], {-1, -1, -1, -1});
  check_dim_mapping(inferred_dist_attrs_dy.second[2], {-1, -1, -1});
  EXPECT_EQ(inferred_dist_attrs_st.first, inferred_dist_attrs_dy.first);
  EXPECT_EQ(inferred_dist_attrs_st.second, inferred_dist_attrs_dy.second);
  VLOG(4) << "test3 done." << std::endl << std::endl << std::endl;
}

TEST(DefaultDataParallelSPMDRule, Ctor) {
  // build input data class
  std::vector<int64_t> x_shape = {10, 10, 32, 48};
  std::vector<int64_t> y_shape = {32, 48};
  std::vector<int64_t> out1_shape = {10, 10, 32, 48};
  std::vector<int64_t> out2_shape = {10, 32, 48};

  std::vector<int64_t> mesh_shape = {2, 3};
  std::vector<int64_t> process_ids = {0, 1, 2, 3, 4, 5};
  std::vector<std::string> dim_names = {"x", "y"};
  ProcessMesh process_mesh(mesh_shape, process_ids, dim_names);

  TensorDistAttr x_dist_attr = TensorDistAttr();
  x_dist_attr.set_process_mesh(process_mesh);
  x_dist_attr.set_dims_mapping(std::vector<int64_t>({-1, -1, -1, 1}));
  x_dist_attr.set_dynamic_dims(std::vector<bool>({false, false}));

  TensorDistAttr y_dist_attr = TensorDistAttr();
  y_dist_attr.set_process_mesh(process_mesh);
  y_dist_attr.set_dims_mapping(std::vector<int64_t>({0, 1}));
  y_dist_attr.set_dynamic_dims(std::vector<bool>({false, false}));

  TensorDistAttr out1_dist_attr = TensorDistAttr();
  out1_dist_attr.set_process_mesh(process_mesh);
  out1_dist_attr.set_dims_mapping(std::vector<int64_t>({-1, -1, 1, -1}));
  out1_dist_attr.set_dynamic_dims(std::vector<bool>({false, false}));

  TensorDistAttr out2_dist_attr = TensorDistAttr();
  out2_dist_attr.set_process_mesh(process_mesh);
  out2_dist_attr.set_dims_mapping(std::vector<int64_t>({-1, 1, -1}));
  out2_dist_attr.set_dynamic_dims(std::vector<bool>({false, false}));

  phi::distributed::DistMetaTensor x(common::make_ddim(x_shape), x_dist_attr);
  phi::distributed::DistMetaTensor y(common::make_ddim(y_shape), y_dist_attr);
  phi::distributed::DistMetaTensor out1(common::make_ddim(out1_shape),
                                        out1_dist_attr);
  phi::distributed::DistMetaTensor out2(common::make_ddim(out2_shape),
                                        out2_dist_attr);

  // 2 inputs 2 outputs, batch axis sharding is propagated while other axes are
  // replicated call in vector arguments format
  auto inferred_dist_attrs_st =
      phi::distributed::DefaultDataParallelInferSpmd({&x, &y}, {&out1, &out2});
  // call in variadic arguments format
  auto inferred_dist_attrs_dy =
      phi::distributed::VariadicDefaultDataParallelInferSpmd(
          x, y, &out1, &out2);

  size_t input_size = 2;
  size_t output_size = 2;
  EXPECT_EQ(inferred_dist_attrs_st.first.size(), input_size);
  EXPECT_EQ(inferred_dist_attrs_st.second.size(), output_size);
  EXPECT_EQ(inferred_dist_attrs_dy.first.size(), input_size);
  EXPECT_EQ(inferred_dist_attrs_dy.second.size(), output_size);
  check_dim_mapping(inferred_dist_attrs_st.first[0], {0, -1, -1, -1});
  check_dim_mapping(inferred_dist_attrs_st.first[1], {0, -1});
  check_dim_mapping(inferred_dist_attrs_st.second[0], {0, -1, -1, -1});
  check_dim_mapping(inferred_dist_attrs_st.second[1], {0, -1, -1});
  EXPECT_EQ(is_partial(inferred_dist_attrs_st.first[0]), false);
  EXPECT_EQ(is_partial(inferred_dist_attrs_st.first[1]), false);
  EXPECT_EQ(is_partial(inferred_dist_attrs_st.second[0]), false);
  EXPECT_EQ(is_partial(inferred_dist_attrs_st.second[1]), false);

  EXPECT_EQ(inferred_dist_attrs_st.first, inferred_dist_attrs_dy.first);
  EXPECT_EQ(inferred_dist_attrs_st.second, inferred_dist_attrs_dy.second);
  VLOG(4) << "test1 done." << std::endl << std::endl << std::endl;

  // 1 inputs 3 outputs, batch axis is un-sharded
  // call in vector arguments format
  inferred_dist_attrs_st =
      phi::distributed::DefaultDataParallelInferSpmd({&x}, {&y, &out1, &out2});
  // call in variadic arguments format
  inferred_dist_attrs_dy =
      phi::distributed::VariadicDefaultDataParallelInferSpmd(
          x, &y, &out1, &out2);

  input_size = 1;
  output_size = 3;
  EXPECT_EQ(inferred_dist_attrs_st.first.size(), input_size);
  EXPECT_EQ(inferred_dist_attrs_st.second.size(), output_size);
  EXPECT_EQ(inferred_dist_attrs_dy.first.size(), input_size);
  EXPECT_EQ(inferred_dist_attrs_dy.second.size(), output_size);

  check_dim_mapping(inferred_dist_attrs_dy.first[0], {-1, -1, -1, -1});
  check_dim_mapping(inferred_dist_attrs_dy.second[0], {-1, -1});
  check_dim_mapping(inferred_dist_attrs_dy.second[1], {-1, -1, -1, -1});
  check_dim_mapping(inferred_dist_attrs_dy.second[2], {-1, -1, -1});

  EXPECT_EQ(inferred_dist_attrs_st.first, inferred_dist_attrs_dy.first);
  EXPECT_EQ(inferred_dist_attrs_st.second, inferred_dist_attrs_dy.second);
  VLOG(4) << "test2 done." << std::endl << std::endl << std::endl;

  // conflict on batch axis
  x_dist_attr.set_dims_mapping(std::vector<int64_t>({0, -1, -1, -1}));
  y_dist_attr.set_dims_mapping(std::vector<int64_t>({-1, -1}));
  out1_dist_attr.set_dims_mapping(std::vector<int64_t>({1, -1, -1, -1}));
  x = phi::distributed::DistMetaTensor(common::make_ddim(x_shape), x_dist_attr);
  y = phi::distributed::DistMetaTensor(common::make_ddim(y_shape), y_dist_attr);
  out1 = phi::distributed::DistMetaTensor(common::make_ddim(out1_shape),
                                          out1_dist_attr);

  EXPECT_ANY_THROW(inferred_dist_attrs_st =
                       phi::distributed::DefaultDataParallelInferSpmd(
                           {&x, &y, &out1}, {&out2}));
  // call in variadic arguments format
  EXPECT_ANY_THROW(inferred_dist_attrs_dy =
                       phi::distributed::VariadicDefaultDataParallelInferSpmd(
                           x, y, out1, &out2));

  VLOG(4) << "test3 done." << std::endl << std::endl << std::endl;

  // 2 inputs 2 outputs, backward
  // call in vector arguments format
  out1_dist_attr.set_dims_mapping(std::vector<int64_t>({-1, 0, 1, -1}));
  out2_dist_attr.set_dims_mapping(std::vector<int64_t>({0, 1, -1}));
  out1 = phi::distributed::DistMetaTensor(common::make_ddim(out1_shape),
                                          out1_dist_attr);
  out2 = phi::distributed::DistMetaTensor(common::make_ddim(out2_shape),
                                          out2_dist_attr);

  inferred_dist_attrs_st =
      phi::distributed::DefaultDataParallelInferSpmdReverse({&x, &y},
                                                            {&out1, &out2});
  // call in variadic arguments format
  inferred_dist_attrs_dy =
      phi::distributed::VariadicDefaultDataParallelInferSpmdReverse(
          x, y, &out1, &out2);

  input_size = 2;
  output_size = 2;
  EXPECT_EQ(inferred_dist_attrs_st.first.size(), input_size);
  EXPECT_EQ(inferred_dist_attrs_st.second.size(), output_size);
  EXPECT_EQ(inferred_dist_attrs_dy.first.size(), input_size);
  EXPECT_EQ(inferred_dist_attrs_dy.second.size(), output_size);
  check_dim_mapping(inferred_dist_attrs_dy.first[0], {0, -1, -1, -1});
  check_dim_mapping(inferred_dist_attrs_dy.first[1], {0, -1});
  check_dim_mapping(inferred_dist_attrs_dy.second[0], {0, -1, -1, -1});
  check_dim_mapping(inferred_dist_attrs_dy.second[1], {0, -1, -1});
  EXPECT_EQ(inferred_dist_attrs_st.first, inferred_dist_attrs_dy.first);
  EXPECT_EQ(inferred_dist_attrs_st.second, inferred_dist_attrs_dy.second);
  VLOG(4) << "test4 done." << std::endl << std::endl << std::endl;
}
TEST(ConcatRule, Ctor) {
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
  auto inferred_dist_attrs = phi::distributed::ConcatInferSpmd(inputs, 0);
  // list of tensor => single tensor
  EXPECT_EQ(inferred_dist_attrs.first.size(), static_cast<size_t>(1));
  EXPECT_EQ(inferred_dist_attrs.second.size(), static_cast<size_t>(1));
  EXPECT_TRUE(
      paddle::holds_alternative<std::vector<phi::distributed::TensorDistAttr>>(
          inferred_dist_attrs.first[0]));
  EXPECT_TRUE(paddle::holds_alternative<phi::distributed::TensorDistAttr>(
      inferred_dist_attrs.second[0]));
  auto& inputs_infer1 = PADDLE_GET_CONST(std::vector<TensorDistAttr>,
                                         inferred_dist_attrs.first[0]);
  for (auto e : inputs_infer1) {
    check_dim_mapping(e, {-1, 1, 0});
    check_partial_dims(e, {});
  }
  check_dim_mapping(inferred_dist_attrs.second[0], {-1, 1, 0});
  check_partial_dims(inferred_dist_attrs.second[0], {});

  auto build_output = [&](const TensorDistAttr& t_dist_attr,
                          const std::vector<int64_t>& shape) {
    return phi::distributed::DistMetaTensor(common::make_ddim(shape),
                                            t_dist_attr);
  };

  auto& output_dist_attr =
      PADDLE_GET_CONST(TensorDistAttr, inferred_dist_attrs.second[0]);
  auto output = build_output(output_dist_attr, {22, 16, 16});
  // test reverse
  auto inferred_reverse_attrs =
      phi::distributed::ConcatInferSpmdReverse(inputs, output, 0);
  auto& inputs_infer1_reverse = PADDLE_GET_CONST(
      std::vector<TensorDistAttr>, inferred_reverse_attrs.first[0]);
  for (auto e : inputs_infer1_reverse) {
    check_dim_mapping(e, {-1, 1, 0});
    check_partial_dims(e, {});
  }
  check_dim_mapping(inferred_reverse_attrs.second[0],
                    output_dist_attr.dims_mapping());
  // test grad
  auto inferred_grad_attrs =
      phi::distributed::ConcatGradInferSpmdDynamic(inputs, output, 0);
  auto& inputs_infer1_grad = PADDLE_GET_CONST(std::vector<TensorDistAttr>,
                                              inferred_grad_attrs.first[0]);
  for (auto e : inputs_infer1_grad) {
    check_dim_mapping(e, {-1, 1, 0});
    check_partial_dims(e, {});
  }
  check_dim_mapping(inferred_grad_attrs.first[1],
                    output_dist_attr.dims_mapping());
  auto& inferred_grad = PADDLE_GET_CONST(std::vector<TensorDistAttr>,
                                         inferred_grad_attrs.second[0]);
  for (auto e : inferred_grad) {
    check_dim_mapping(e, {-1, 1, 0});
    check_partial_dims(e, {});
  }

  // test 2，force replicate along concat axis
  inputs = build_inputs();
  inferred_dist_attrs = phi::distributed::ConcatInferSpmd(inputs, 1);
  // list of tensor => single tensor
  EXPECT_EQ(inferred_dist_attrs.first.size(), static_cast<size_t>(1));
  EXPECT_EQ(inferred_dist_attrs.second.size(), static_cast<size_t>(1));
  EXPECT_TRUE(
      paddle::holds_alternative<std::vector<phi::distributed::TensorDistAttr>>(
          inferred_dist_attrs.first[0]));
  EXPECT_TRUE(paddle::holds_alternative<phi::distributed::TensorDistAttr>(
      inferred_dist_attrs.second[0]));
  auto& inputs_infer2 = PADDLE_GET_CONST(std::vector<TensorDistAttr>,
                                         inferred_dist_attrs.first[0]);
  for (auto e : inputs_infer2) {
    check_dim_mapping(e, {1, -1, 0});
    check_partial_dims(e, {});
  }
  check_dim_mapping(inferred_dist_attrs.second[0], {1, -1, 0});
  check_partial_dims(inferred_dist_attrs.second[0], {});
}

TEST(StackRule, Ctor) {
  std::vector<int64_t> mesh_shape = {2, 2};
  std::vector<int64_t> process_ids = {0, 1, 2, 3};
  std::vector<std::string> dim_names = {"x", "y"};
  ProcessMesh process_mesh(mesh_shape, process_ids, dim_names);

  int input_size = 3;
  std::vector<int64_t> input_shape = {16, 8, 4};
  std::vector<std::vector<int64_t>> dim_mappings = {
      {-1, 0, 1}, {-1, 1, 0}, {-1, -1, 0}};
  std::vector<std::vector<int64_t>> partial_status = {{}, {}, {1}};

  auto build_inputs = [&] {
    std::vector<phi::distributed::DistMetaTensor> inputs;
    for (int i = 0; i < input_size; i++) {
      auto t_dist_attr = TensorDistAttr();
      t_dist_attr.set_process_mesh(process_mesh);
      t_dist_attr.set_dims_mapping(dim_mappings[i]);
      t_dist_attr.set_dynamic_dims({false, false, false});
      auto input = phi::distributed::DistMetaTensor(
          common::make_ddim(input_shape), t_dist_attr);
      inputs.push_back(input);
    }
    return inputs;
  };

  auto build_output = [&](const TensorDistAttr& t_dist_attr, int stack_dim) {
    std::vector<int64_t> output_shape;
    std::transform(input_shape.begin(),
                   input_shape.begin() + stack_dim,
                   std::back_inserter(output_shape),
                   [](int64_t x) { return x; });
    output_shape.push_back(input_size);
    std::transform(input_shape.begin() + stack_dim,
                   input_shape.end(),
                   std::back_inserter(output_shape),
                   [](int64_t x) { return x; });
    return phi::distributed::DistMetaTensor(common::make_ddim(output_shape),
                                            t_dist_attr);
  };

  // test 1, inputs are aligned according to cost.
  auto inputs = build_inputs();
  auto inferred_dist_attrs = phi::distributed::StackInferSpmd(inputs, 0);
  // list of tensor => single tensor
  EXPECT_EQ(inferred_dist_attrs.first.size(), static_cast<size_t>(1));
  EXPECT_EQ(inferred_dist_attrs.second.size(), static_cast<size_t>(1));
  EXPECT_TRUE(
      paddle::holds_alternative<std::vector<phi::distributed::TensorDistAttr>>(
          inferred_dist_attrs.first[0]));
  EXPECT_TRUE(paddle::holds_alternative<phi::distributed::TensorDistAttr>(
      inferred_dist_attrs.second[0]));
  auto& inputs_infer1 = PADDLE_GET_CONST(std::vector<TensorDistAttr>,
                                         inferred_dist_attrs.first[0]);
  for (auto e : inputs_infer1) {
    check_dim_mapping(e, {-1, 1, 0});
    check_partial_dims(e, {});
  }
  check_dim_mapping(inferred_dist_attrs.second[0], {-1, -1, 1, 0});
  check_partial_dims(inferred_dist_attrs.second[0], {});

  auto output_dist_attr =
      PADDLE_GET_CONST(TensorDistAttr, inferred_dist_attrs.second[0]);
  auto output = build_output(output_dist_attr, 0);
  // test reverse
  auto inferred_reverse_attrs =
      phi::distributed::StackInferSpmdReverse(inputs, output, 0);
  auto& inputs_infer1_reverse = PADDLE_GET_CONST(
      std::vector<TensorDistAttr>, inferred_reverse_attrs.first[0]);
  for (auto e : inputs_infer1_reverse) {
    check_dim_mapping(e, {-1, 1, 0});
    check_partial_dims(e, {});
  }
  check_dim_mapping(inferred_reverse_attrs.second[0],
                    output_dist_attr.dims_mapping());
  // test grad
  auto inferred_grad_attrs = phi::distributed::StackGradInferSpmd(output, 0);
  check_dim_mapping(inferred_grad_attrs.first[0],
                    output_dist_attr.dims_mapping());
  auto& inferred_grad = PADDLE_GET_CONST(std::vector<TensorDistAttr>,
                                         inferred_grad_attrs.second[0]);
  for (auto e : inferred_grad) {
    check_dim_mapping(e, {-1, 1, 0});
    check_partial_dims(e, {});
  }

  // test 2，force replicate along concat axis
  inputs = build_inputs();
  inferred_dist_attrs = phi::distributed::StackInferSpmd(inputs, 1);
  // list of tensor => single tensor
  EXPECT_EQ(inferred_dist_attrs.first.size(), static_cast<size_t>(1));
  EXPECT_EQ(inferred_dist_attrs.second.size(), static_cast<size_t>(1));
  EXPECT_TRUE(
      paddle::holds_alternative<std::vector<phi::distributed::TensorDistAttr>>(
          inferred_dist_attrs.first[0]));
  EXPECT_TRUE(paddle::holds_alternative<phi::distributed::TensorDistAttr>(
      inferred_dist_attrs.second[0]));
  auto& inputs_infer2 = PADDLE_GET_CONST(std::vector<TensorDistAttr>,
                                         inferred_dist_attrs.first[0]);
  for (auto e : inputs_infer2) {
    check_dim_mapping(e, {-1, 1, 0});
    check_partial_dims(e, {});
  }
  check_dim_mapping(inferred_dist_attrs.second[0], {-1, -1, 1, 0});
  check_partial_dims(inferred_dist_attrs.second[0], {});
}

TEST(WhereRule, Ctor) {
  std::vector<int64_t> mesh_shape = {2, 2};
  std::vector<int64_t> process_ids = {0, 1, 2, 3};
  std::vector<std::string> dim_names = {"x", "y"};
  ProcessMesh process_mesh(mesh_shape, process_ids, dim_names);

  std::vector<std::vector<int64_t>> shapes = {{16, 16, 16}, {16, 16}, {16}};
  std::vector<std::vector<int64_t>> dim_mappings = {{-1, 0, -1}, {-1, 0}, {-1}};

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

  auto inputs = build_inputs();
  auto inferred_dist_attrs = phi::distributed::WhereGradInferSpmd(
      inputs[0], inputs[1], inputs[2], inputs[0]);

  EXPECT_EQ(inferred_dist_attrs.first.size(), static_cast<size_t>(4));
  EXPECT_EQ(inferred_dist_attrs.second.size(), static_cast<size_t>(2));

  check_dim_mapping(inferred_dist_attrs.first[0], {-1, 0, -1});
  check_dim_mapping(inferred_dist_attrs.first[1], {0, -1});
  check_dim_mapping(inferred_dist_attrs.first[2], {-1});
  check_dim_mapping(inferred_dist_attrs.first[3], {-1, 0, -1});

  check_dim_mapping(inferred_dist_attrs.second[0], {0, -1});
  check_partial_dims(inferred_dist_attrs.second[0], {});
  check_dim_mapping(inferred_dist_attrs.second[1], {-1});
  check_partial_dims(inferred_dist_attrs.second[1], {0});
}

TEST(ReduceMaxRule, Ctor) {
  std::vector<int64_t> mesh_shape = {2};
  std::vector<int64_t> process_ids = {0, 1};
  std::vector<std::string> dim_names = {"x"};
  ProcessMesh process_mesh(mesh_shape, process_ids, dim_names);

  // test forward
  auto t_dist_attr = TensorDistAttr();
  t_dist_attr.set_process_mesh(process_mesh);
  t_dist_attr.set_dims_mapping({-1, 0, -1});
  t_dist_attr.set_dynamic_dims({false, false, false});
  phi::distributed::DistMetaTensor x = phi::distributed::DistMetaTensor(
      common::make_ddim({4, 6, 8}), t_dist_attr);
  phi::IntArray axis = {1};
  bool keep_dim = false;
  phi::distributed::SpmdInfo forward_info =
      phi::distributed::ReductionMaxInferSpmdDynamic(x, axis, keep_dim);
  check_dim_mapping(forward_info.second[0], {-1, -1});
  check_partial_dims(forward_info.second[0], {0});
  // test backward
  phi::distributed::DistMetaTensor out = phi::distributed::DistMetaTensor(
      common::make_ddim({4, 8}),
      PADDLE_GET_CONST(TensorDistAttr, forward_info.second[0]));
  phi::distributed::DistMetaTensor out_grad = out;
  phi::distributed::SpmdInfo backward_info =
      phi::distributed::ReductionGradInferSpmd(
          x, out, out_grad, {1}, false, false);
  check_partial_dims(backward_info.first[1], {});
  check_dim_mapping(backward_info.second[0], {-1, -1, -1});
  check_partial_dims(backward_info.second[0], {});
}

TEST(ReduceAllRule, Ctor) {
  std::vector<int64_t> mesh_shape = {2};
  std::vector<int64_t> process_ids = {0, 1};
  std::vector<std::string> dim_names = {"x"};
  ProcessMesh process_mesh(mesh_shape, process_ids, dim_names);

  // test forward
  auto t_dist_attr = TensorDistAttr();
  t_dist_attr.set_process_mesh(process_mesh);
  t_dist_attr.set_dims_mapping({-1, 0, -1});
  t_dist_attr.set_dynamic_dims({false, false, false});
  phi::distributed::DistMetaTensor x =
      phi::distributed::DistMetaTensor(phi::make_ddim({4, 6, 8}), t_dist_attr);
  phi::IntArray axis = {1};
  bool keep_dim = false;
  phi::distributed::SpmdInfo forward_info =
      phi::distributed::ReductionAllInferSpmdDynamic(x, axis, keep_dim);
  check_dim_mapping(forward_info.second[0], {-1, -1});
  check_partial_dims(forward_info.second[0], {0});
}

TEST(Numel, Ctor) {
  std::vector<int64_t> mesh_shape = {2, 2};
  std::vector<int64_t> process_ids = {0, 1, 2, 3};
  std::vector<std::string> dim_names = {"x", "y"};
  ProcessMesh process_mesh(mesh_shape, process_ids, dim_names);

  std::vector<int64_t> shape = {16, 16, 16};
  std::vector<int64_t> dims_mapping = {-1, 0, -1};

  auto t_dist_attr = TensorDistAttr();
  t_dist_attr.set_process_mesh(process_mesh);
  t_dist_attr.set_dims_mapping(dims_mapping);
  t_dist_attr.set_dynamic_dims({false, false, false});
  auto input =
      phi::distributed::DistMetaTensor(common::make_ddim(shape), t_dist_attr);
  auto inferred_dist_attrs = phi::distributed::NumelInferSpmd(input);
  EXPECT_EQ(inferred_dist_attrs.first.size(), static_cast<size_t>(1));
  EXPECT_EQ(inferred_dist_attrs.second.size(), static_cast<size_t>(1));
  check_dim_mapping(inferred_dist_attrs.first[0], dims_mapping);
  check_dim_mapping(inferred_dist_attrs.second[0], {});
  check_partial_dims(inferred_dist_attrs.second[0], {0});
}

TEST(Triu, Ctor) {
  std::vector<int64_t> mesh_shape = {2, 2};
  std::vector<int64_t> process_ids = {0, 1, 2, 3};
  std::vector<std::string> dim_names = {"x", "y"};
  ProcessMesh process_mesh(mesh_shape, process_ids, dim_names);

  std::vector<int64_t> shape = {16, 16, 16};
  std::vector<int64_t> dims_mapping = {0, -1, 1};

  auto t_dist_attr = TensorDistAttr();
  t_dist_attr.set_process_mesh(process_mesh);
  t_dist_attr.set_dims_mapping(dims_mapping);
  t_dist_attr.set_dynamic_dims({false, false, false});
  auto input =
      phi::distributed::DistMetaTensor(common::make_ddim(shape), t_dist_attr);
  auto inferred_dist_attrs = phi::distributed::TriuGradInferSpmd(input, 0);
  EXPECT_EQ(inferred_dist_attrs.first.size(), static_cast<size_t>(1));
  EXPECT_EQ(inferred_dist_attrs.second.size(), static_cast<size_t>(1));
  check_dim_mapping(inferred_dist_attrs.first[0], {0, -1, -1});
  check_dim_mapping(inferred_dist_attrs.second[0], {0, -1, -1});
  check_partial_dims(inferred_dist_attrs.second[0], {});
}

TEST(LayerNorm, Ctor) {
  using phi::distributed::PartialStatus;
  std::vector<int64_t> mesh_shape = {2, 2};
  std::vector<int64_t> process_ids = {0, 1, 2, 3};
  std::vector<std::string> dim_names = {"x", "y"};
  ProcessMesh process_mesh(mesh_shape, process_ids, dim_names);

  std::vector<int64_t> x_shapes = {16, 32, 32};

  auto build_input = [&](const std::vector<int64_t>& shape,
                         const std::vector<int64_t>& dim_mapping) {
    auto t_dist_attr = TensorDistAttr();
    t_dist_attr.set_process_mesh(process_mesh);
    t_dist_attr.set_dims_mapping(dim_mapping);
    t_dist_attr.set_dynamic_dims({false, false, false});
    auto input =
        phi::distributed::DistMetaTensor(common::make_ddim(shape), t_dist_attr);
    return input;
  };
  // test 1
  auto x = build_input(x_shapes, {0, 1, -1});
  auto out_grad = build_input(x_shapes, {0, 1, -1});
  auto mean = build_input({16, 32}, {0, 1});
  auto variance = build_input({16, 32}, {0, 1});
  auto scale = build_input({32}, {0});
  auto bias = build_input({32}, {0});

  auto spmd1 =
      LayerNormGradInferSpmd(x, scale, bias, mean, variance, out_grad, 1.0, 2);

  EXPECT_EQ(spmd1.first.size(), static_cast<size_t>(6));
  EXPECT_EQ(spmd1.second.size(), static_cast<size_t>(3));

  check_dim_mapping(spmd1.first[0], {0, 1, -1});
  check_dim_mapping(spmd1.first[1], {-1});
  check_dim_mapping(spmd1.first[2], {-1});
  check_dim_mapping(spmd1.first[3], {0, 1});
  check_dim_mapping(spmd1.first[4], {0, 1});
  check_dim_mapping(spmd1.first[5], {0, 1, -1});
  check_dim_mapping(spmd1.second[0], {0, 1, -1});
  check_dim_mapping(spmd1.second[1], {-1});
  check_dim_mapping(spmd1.second[2], {-1});
  check_partial_dims(spmd1.second[1], {0, 1});
  check_partial_dims(spmd1.second[2], {0, 1});
  // test 2
  mean = build_input({16}, {0});
  variance = build_input({16}, {0});
  scale = build_input({32, 32}, {0, 1});
  bias = build_input({32, 32}, {0, 1});
  auto spmd2 =
      LayerNormGradInferSpmd(x, scale, bias, mean, variance, out_grad, 1.0, 1);
  EXPECT_EQ(spmd2.first.size(), static_cast<size_t>(6));
  EXPECT_EQ(spmd2.second.size(), static_cast<size_t>(3));
  check_dim_mapping(spmd2.first[0], {0, -1, -1});
  check_dim_mapping(spmd2.first[1], {-1, -1});
  check_dim_mapping(spmd2.first[2], {-1, -1});
  check_dim_mapping(spmd2.first[3], {0});
  check_dim_mapping(spmd2.first[4], {0});
  check_dim_mapping(spmd2.first[5], {0, -1, -1});
  check_dim_mapping(spmd2.second[0], {0, -1, -1});
  check_dim_mapping(spmd2.second[1], {-1, -1});
  check_dim_mapping(spmd2.second[2], {-1, -1});
  check_partial_dims(spmd2.second[1], {0});
  check_partial_dims(spmd2.second[2], {0});
}

TEST(FlashAtt, Ctor) {
  std::vector<int64_t> mesh_shape = {2, 2};
  std::vector<int64_t> process_ids = {0, 1, 2, 3};
  std::vector<std::string> dim_names = {"x", "y"};
  ProcessMesh process_mesh(mesh_shape, process_ids, dim_names);

  auto build_input = [&](const std::vector<int64_t>& shape,
                         const std::vector<int64_t>& dim_mapping) {
    auto t_dist_attr = TensorDistAttr();
    t_dist_attr.set_process_mesh(process_mesh);
    t_dist_attr.set_dims_mapping(dim_mapping);
    t_dist_attr.set_dynamic_dims(std::vector<bool>(shape.size(), false));
    auto input =
        phi::distributed::DistMetaTensor(common::make_ddim(shape), t_dist_attr);
    return input;
  };

  // b, s, m, h
  std::vector<int64_t> qkv_shape = {2, 256, 2, 128};
  std::vector<int64_t> dim_mapping = {0, 1, -1, -1};

  auto qkv = build_input(qkv_shape, dim_mapping);
  auto mask = build_input({}, {});
  auto seed_offset = build_input({}, {});

  auto spmd1 = FlashAttInferSpmd(
      qkv, qkv, qkv, seed_offset, mask, 0.5, false, false, false, "");

  EXPECT_EQ(spmd1.first.size(), static_cast<size_t>(5));
  EXPECT_EQ(spmd1.second.size(), static_cast<size_t>(4));
  check_dim_mapping(spmd1.first[0], {0, -1, -1, -1});
  check_dim_mapping(spmd1.first[1], {0, -1, -1, -1});
  check_dim_mapping(spmd1.first[2], {0, -1, -1, -1});
  check_dim_mapping(spmd1.first[3], {});
  check_dim_mapping(spmd1.first[4], {});
  check_dim_mapping(spmd1.second[0], {0, -1, -1, -1});
  check_dim_mapping(spmd1.second[1], {0, -1, -1, -1});
  check_dim_mapping(spmd1.second[2], {0, -1, -1});
  check_dim_mapping(spmd1.second[3], {-1});

  auto out = build_input(qkv_shape, {0, -1, 1, -1});
  auto softmax_lse = build_input({2, 2, 256}, {0, 1, -1});
  auto out_grad = build_input(qkv_shape, {-1, -1, -1, -1});

  auto spmd2 = FlashAttGradInferSpmd(
      qkv, qkv, qkv, out, softmax_lse, seed_offset, mask, out_grad, 0.5, false);

  EXPECT_EQ(spmd2.first.size(), static_cast<size_t>(8));
  EXPECT_EQ(spmd2.second.size(), static_cast<size_t>(3));

  check_dim_mapping(spmd2.first[0], {0, -1, 1, -1});
  check_dim_mapping(spmd2.first[1], {0, -1, 1, -1});
  check_dim_mapping(spmd2.first[2], {0, -1, 1, -1});
  check_dim_mapping(spmd2.first[3], {0, -1, 1, -1});
  check_dim_mapping(spmd2.first[4], {0, 1, -1});
  check_dim_mapping(spmd2.first[5], {});
  check_dim_mapping(spmd2.first[6], {});
  check_dim_mapping(spmd2.first[7], {0, -1, 1, -1});
  check_dim_mapping(spmd2.second[0], {0, -1, 1, -1});
  check_dim_mapping(spmd2.second[1], {0, -1, 1, -1});
  check_dim_mapping(spmd2.second[2], {0, -1, 1, -1});
}

TEST(Util, Ctor) {
  // test equal test not equal
  using phi::distributed::PartialStatus;
  using phi::distributed::PlacementEqual;
  using phi::distributed::ReplicatedStatus;
  using phi::distributed::ShardStatus;
  auto a = std::make_shared<PartialStatus>(phi::ReduceType::kRedSum);
  auto b = std::make_shared<PartialStatus>(phi::ReduceType::kRedMin);
  EXPECT_TRUE(PlacementEqual(a, a));
  EXPECT_TRUE(!PlacementEqual(a, b));
  auto c = std::make_shared<ShardStatus>(0);
  auto d = std::make_shared<ShardStatus>(1);
  EXPECT_TRUE(!PlacementEqual(a, c));
  EXPECT_TRUE(!PlacementEqual(b, c));
  EXPECT_TRUE(PlacementEqual(c, c));
  EXPECT_TRUE(!PlacementEqual(c, d));
  auto e = std::make_shared<ReplicatedStatus>();
  EXPECT_TRUE(PlacementEqual(e, e));
  EXPECT_TRUE(!PlacementEqual(a, e));
  EXPECT_TRUE(!PlacementEqual(b, e));
  EXPECT_TRUE(!PlacementEqual(c, e));
  EXPECT_TRUE(!PlacementEqual(d, e));
}

TEST(Transpose, Ctor) {
  std::vector<int64_t> mesh_shape = {2, 2};
  std::vector<int64_t> process_ids = {0, 1, 2, 3};
  std::vector<std::string> dim_names = {"x", "y"};
  ProcessMesh process_mesh(mesh_shape, process_ids, dim_names);

  std::vector<int64_t> shape = {6, 8, 10};
  std::vector<int64_t> dims_mapping = {0, -1, 1};

  TensorDistAttr t_dist_attr = TensorDistAttr();
  t_dist_attr.set_process_mesh(process_mesh);
  t_dist_attr.set_dims_mapping(dims_mapping);
  t_dist_attr.set_dynamic_dims({false, false, false});
  phi::distributed::DistMetaTensor x =
      phi::distributed::DistMetaTensor(common::make_ddim(shape), t_dist_attr);
  std::vector<int> perm = {1, 2, -3};
  // test forward
  phi::distributed::SpmdInfo forward_spmd_info =
      phi::distributed::TransposeInferSpmd(x, perm);
  EXPECT_EQ(forward_spmd_info.first.size(), static_cast<size_t>(1));
  EXPECT_EQ(forward_spmd_info.second.size(), static_cast<size_t>(1));
  check_dim_mapping(forward_spmd_info.first[0], {0, -1, 1});
  check_dim_mapping(forward_spmd_info.second[0], {-1, 1, 0});
  check_partial_dims(forward_spmd_info.second[0], {});
  // test backward
  phi::distributed::DistMetaTensor out_grad = phi::distributed::DistMetaTensor(
      common::make_ddim({8, 10, 6}),
      PADDLE_GET_CONST(TensorDistAttr, forward_spmd_info.second[0]));
  phi::distributed::SpmdInfo backward_spmd_info =
      TransposeGradInferSpmd(out_grad, perm);
  EXPECT_EQ(backward_spmd_info.first.size(), static_cast<size_t>(1));
  EXPECT_EQ(backward_spmd_info.second.size(), static_cast<size_t>(1));
  check_dim_mapping(backward_spmd_info.first[0], {-1, 1, 0});
  check_dim_mapping(backward_spmd_info.second[0], {0, -1, 1});
  check_partial_dims(backward_spmd_info.second[0], {});
}

TEST(FusedRope, Ctor) {
  std::vector<int64_t> mesh_shape = {2, 2};
  std::vector<int64_t> process_ids = {0, 1, 2, 3};
  std::vector<std::string> dim_names = {"x", "y"};
  ProcessMesh process_mesh(mesh_shape, process_ids, dim_names);

  auto build_input = [&](const std::vector<int64_t>& shape,
                         const std::vector<int64_t>& dim_mapping) {
    TensorDistAttr t_dist_attr;
    t_dist_attr.set_process_mesh(process_mesh);
    t_dist_attr.set_dims_mapping(dim_mapping);
    t_dist_attr.set_dynamic_dims(std::vector<bool>(shape.size(), false));
    auto input =
        phi::distributed::DistMetaTensor(common::make_ddim(shape), t_dist_attr);
    return input;
  };

  phi::distributed::DistMetaTensor q =
      build_input({16, 2048, 64, 128}, {0, 1, -1, -1});
  phi::distributed::DistMetaTensor none;

  // 1. test forward
  // 1.1 only q input
  phi::distributed::SpmdInfo forward_spmd_info =
      phi::distributed::FusedRopeInferSpmd(
          q, none, none, none, none, none, false, false);
  EXPECT_EQ(forward_spmd_info.first.size(), static_cast<size_t>(6));
  EXPECT_EQ(forward_spmd_info.second.size(), static_cast<size_t>(3));
  check_dim_mapping(forward_spmd_info.first[0], {0, -1, -1, -1});
  check_dim_mapping(forward_spmd_info.first[1], {});
  check_dim_mapping(forward_spmd_info.first[2], {});
  check_dim_mapping(forward_spmd_info.first[3], {});
  check_dim_mapping(forward_spmd_info.first[4], {});
  check_dim_mapping(forward_spmd_info.first[5], {});
  check_dim_mapping(forward_spmd_info.second[0], {0, -1, -1, -1});
  check_dim_mapping(forward_spmd_info.second[1], {});
  check_dim_mapping(forward_spmd_info.second[2], {});
  check_partial_dims(forward_spmd_info.second[0], {});

  // 1.2 q, k, sin, cos, position_ids
  phi::distributed::DistMetaTensor k =
      build_input({16, 2048, 64, 128}, {-1, 1, -1, 0});
  phi::distributed::DistMetaTensor sin =
      build_input({1, 2048, 1, 128}, {-1, 0, -1, 1});
  phi::distributed::DistMetaTensor cos =
      build_input({1, 2048, 1, 128}, {-1, 1, -1, -1});
  phi::distributed::DistMetaTensor position_ids =
      build_input({16, 2048}, {0, 1});
  forward_spmd_info = phi::distributed::FusedRopeInferSpmd(
      q, k, none, sin, cos, position_ids, false, false);
  EXPECT_EQ(forward_spmd_info.first.size(), static_cast<size_t>(6));
  EXPECT_EQ(forward_spmd_info.second.size(), static_cast<size_t>(3));
  check_dim_mapping(forward_spmd_info.first[0], {0, -1, -1, -1});
  check_dim_mapping(forward_spmd_info.first[1], {0, -1, -1, -1});
  check_dim_mapping(forward_spmd_info.first[2], {});
  check_dim_mapping(forward_spmd_info.first[3], {-1, -1, -1, -1});
  check_dim_mapping(forward_spmd_info.first[4], {-1, -1, -1, -1});
  check_dim_mapping(forward_spmd_info.first[5], {0, -1});
  check_dim_mapping(forward_spmd_info.second[0], {0, -1, -1, -1});
  check_dim_mapping(forward_spmd_info.second[1], {0, -1, -1, -1});
  check_dim_mapping(forward_spmd_info.second[2], {});
  check_partial_dims(forward_spmd_info.second[0], {});
  check_partial_dims(forward_spmd_info.second[1], {});
  // 2. test backward
  phi::distributed::SpmdInfo backward_spmd_info =
      FusedRopeGradInferSpmd(sin, cos, position_ids, q, k, none, false, false);
  EXPECT_EQ(backward_spmd_info.first.size(), static_cast<size_t>(6));
  EXPECT_EQ(backward_spmd_info.second.size(), static_cast<size_t>(3));
  check_dim_mapping(backward_spmd_info.first[0], {-1, -1, -1, -1});
  check_dim_mapping(backward_spmd_info.first[1], {-1, -1, -1, -1});
  check_dim_mapping(backward_spmd_info.first[2], {0, -1});
  check_dim_mapping(backward_spmd_info.first[3], {0, -1, -1, -1});
  check_dim_mapping(backward_spmd_info.first[4], {0, -1, -1, -1});
  check_dim_mapping(backward_spmd_info.first[5], {});
  check_dim_mapping(backward_spmd_info.second[0], {0, -1, -1, -1});
  check_dim_mapping(backward_spmd_info.second[1], {0, -1, -1, -1});
  check_dim_mapping(backward_spmd_info.second[2], {});
  check_partial_dims(backward_spmd_info.second[0], {});
  check_partial_dims(backward_spmd_info.second[1], {});

  // 3. test reverse
  phi::distributed::DistMetaTensor out_q =
      build_input({16, 2048, 64, 128}, {0, 1, -1, -1});
  phi::distributed::DistMetaTensor out_k =
      build_input({16, 2048, 64, 128}, {-1, 1, -1, 0});
  phi::distributed::SpmdInfo reverse_spmd_info = FusedRopeInferSpmdReverse(
      q, k, none, sin, cos, position_ids, out_q, out_k, none, false, false);
  EXPECT_EQ(reverse_spmd_info.first.size(), static_cast<size_t>(6));
  EXPECT_EQ(reverse_spmd_info.second.size(), static_cast<size_t>(3));
  check_dim_mapping(reverse_spmd_info.first[0], {0, -1, -1, -1});
  check_dim_mapping(reverse_spmd_info.first[1], {0, -1, -1, -1});
  check_dim_mapping(reverse_spmd_info.first[2], {});
  check_dim_mapping(reverse_spmd_info.first[3], {-1, -1, -1, -1});
  check_dim_mapping(reverse_spmd_info.first[4], {-1, -1, -1, -1});
  check_dim_mapping(reverse_spmd_info.first[5], {0, -1});
  check_dim_mapping(reverse_spmd_info.second[0], {0, -1, -1, -1});
  check_dim_mapping(reverse_spmd_info.second[1], {0, -1, -1, -1});
  check_dim_mapping(reverse_spmd_info.second[2], {});
}

TEST(Reshape, Ctor) {
  std::vector<int64_t> mesh_shape = {2, 2};
  std::vector<int64_t> process_ids = {0, 1, 2, 3};
  std::vector<std::string> dim_names = {"x", "y"};
  ProcessMesh process_mesh(mesh_shape, process_ids, dim_names);

  auto build_input = [&](const std::vector<int64_t>& shape,
                         const std::vector<int64_t>& dim_mapping) {
    auto t_dist_attr = TensorDistAttr();
    t_dist_attr.set_process_mesh(process_mesh);
    t_dist_attr.set_dims_mapping(dim_mapping);
    t_dist_attr.set_dynamic_dims(std::vector<bool>(shape.size(), false));
    auto input =
        phi::distributed::DistMetaTensor(common::make_ddim(shape), t_dist_attr);
    return input;
  };

  // b s h; dp , mp
  auto input = build_input({2, 1024, 1024}, {0, 1, -1});
  // [b, s, h] => [b, s, nh, h/nh]
  auto spmd = ReshapeInferSpmd(input, {2, 1024, 4, -1});
  EXPECT_EQ(spmd.first.size(), static_cast<size_t>(1));
  EXPECT_EQ(spmd.second.size(), static_cast<size_t>(1));
  check_dim_mapping(spmd.first[0], {0, 1, -1});
  check_dim_mapping(spmd.second[0], {0, 1, -1, -1});

  auto out_grad = build_input({2, 1024, 4, 1024 / 4}, {-1, -1, -1, -1});
  auto x = build_input({2, 1024, 1024}, {0, 1, -1});
  auto spmd_grad = ReshapeGradInferSpmd(x, out_grad);
  EXPECT_EQ(spmd_grad.first.size(), static_cast<size_t>(2));
  EXPECT_EQ(spmd_grad.second.size(), static_cast<size_t>(1));
  check_dim_mapping(spmd_grad.first[0], {0, 1, -1});
  check_dim_mapping(spmd_grad.first[1], {0, 1, -1, -1});
  check_dim_mapping(spmd_grad.second[0], {0, 1, -1});
}

TEST(ElementwiseUnaryLike, Ctor) {
  std::vector<int64_t> mesh_shape = {2, 2};
  std::vector<int64_t> process_ids = {0, 1, 2, 3};
  std::vector<std::string> dim_names = {"x", "y"};
  ProcessMesh process_mesh(mesh_shape, process_ids, dim_names);

  std::vector<int64_t> shape = {16, 16, 16};
  std::vector<int64_t> dims_mapping = {0, -1, 1};

  auto t_dist_attr = TensorDistAttr();
  t_dist_attr.set_process_mesh(process_mesh);
  t_dist_attr.set_dims_mapping(dims_mapping);
  t_dist_attr.set_dynamic_dims({false, false, false});

  auto check_element_unary_like = [&dims_mapping](auto& spmd_info) {
    EXPECT_EQ(spmd_info.first.size(), static_cast<size_t>(1));
    EXPECT_EQ(spmd_info.second.size(), static_cast<size_t>(1));
    check_dim_mapping(spmd_info.first[0], dims_mapping);
    check_dim_mapping(spmd_info.second[0], dims_mapping);
    check_partial_dims(spmd_info.second[0], {});
  };

  // cast
  auto input =
      phi::distributed::DistMetaTensor(common::make_ddim(shape), t_dist_attr);
  auto inferred_dist_attrs =
      phi::distributed::CastInferSpmd(input, phi::DataType::FLOAT32);

  check_element_unary_like(inferred_dist_attrs);
  // full like
  input =
      phi::distributed::DistMetaTensor(common::make_ddim(shape), t_dist_attr);
  inferred_dist_attrs =
      phi::distributed::FullLikeInferSpmd(input, 1.0, phi::DataType::FLOAT32);
  check_element_unary_like(inferred_dist_attrs);

  // pow
  input =
      phi::distributed::DistMetaTensor(common::make_ddim(shape), t_dist_attr);
  inferred_dist_attrs = phi::distributed::PowInferSpmd(input, 2);
  check_element_unary_like(inferred_dist_attrs);

  // pow backward
  input =
      phi::distributed::DistMetaTensor(common::make_ddim(shape), t_dist_attr);
  inferred_dist_attrs = phi::distributed::PowGradInferSpmd(input, input, 2);

  // scale
  input =
      phi::distributed::DistMetaTensor(common::make_ddim(shape), t_dist_attr);
  inferred_dist_attrs =
      phi::distributed::ScaleInferSpmd(input, 1.0, 1.0, false);
  check_element_unary_like(inferred_dist_attrs);
}

TEST(EmbeddingGradInferSpmd, Ctor) {
  // build input data class
  std::vector<int64_t> x_shape = {4, 5};
  std::vector<int64_t> w_shape = {10, 3};
  std::vector<int64_t> out_grad_shape = {4, 5, 3};

  std::vector<int64_t> mesh_shape = {2, 3};
  std::vector<int64_t> process_ids = {0, 1, 2, 3, 4, 5};
  std::vector<std::string> dim_names = {"x", "y"};
  ProcessMesh process_mesh(mesh_shape, process_ids, dim_names);

  // indices is shard, embedding table is replicated,
  TensorDistAttr x_dist_attr = TensorDistAttr();
  x_dist_attr.set_process_mesh(process_mesh);
  x_dist_attr.set_dims_mapping(std::vector<int64_t>({1, -1}));
  x_dist_attr.set_dynamic_dims(std::vector<bool>({false, false}));

  TensorDistAttr w_dist_attr = TensorDistAttr();
  w_dist_attr.set_process_mesh(process_mesh);
  w_dist_attr.set_dims_mapping(std::vector<int64_t>({-1, -1}));
  w_dist_attr.set_dynamic_dims(std::vector<bool>({false, false}));

  TensorDistAttr out_grad_dist_attr = TensorDistAttr();
  out_grad_dist_attr.set_process_mesh(process_mesh);
  out_grad_dist_attr.set_dims_mapping(std::vector<int64_t>({-1, -1, -1}));
  out_grad_dist_attr.set_dynamic_dims(std::vector<bool>({false, false}));

  phi::distributed::DistMetaTensor x(phi::make_ddim(x_shape), x_dist_attr);
  phi::distributed::DistMetaTensor w(phi::make_ddim(w_shape), w_dist_attr);
  phi::distributed::DistMetaTensor out_grad(phi::make_ddim(out_grad_shape),
                                            out_grad_dist_attr);

  auto spmdinfo = EmbeddingGradInferSpmd(x, w, out_grad, -1, false);

  EXPECT_EQ(spmdinfo.first.size(), 3UL);
  EXPECT_EQ(spmdinfo.second.size(), 1UL);

  EXPECT_EQ(get_dims_mapping(spmdinfo.first[0]), std::vector<int64_t>({1, -1}));
  EXPECT_EQ(get_dims_mapping(spmdinfo.first[1]),
            std::vector<int64_t>({-1, -1}));
  EXPECT_EQ(get_dims_mapping(spmdinfo.first[2]),
            std::vector<int64_t>({1, -1, -1}));

  EXPECT_EQ(get_dims_mapping(spmdinfo.second[0]),
            std::vector<int64_t>({-1, -1}));
  EXPECT_EQ(
      PADDLE_GET_CONST(phi::distributed::TensorDistAttr, spmdinfo.second[0])
          .is_partial(),
      true);
  VLOG(4) << "Test EmbeddingGradInferSpmd with sharding indices and "
             "replicating weight"
          << std::endl
          << std::endl
          << std::endl;

  // Indices' rank is greater than 1, x and weight is replicated, out_grad is
  // sharded along axis 1
  x_dist_attr.set_dims_mapping({-1, -1});
  w_dist_attr.set_dims_mapping({-1, 1});
  out_grad_dist_attr.set_dims_mapping({-1, 1, -1});
  x = phi::distributed::DistMetaTensor(phi::make_ddim(x_shape), x_dist_attr);
  w = phi::distributed::DistMetaTensor(phi::make_ddim(w_shape), w_dist_attr);
  out_grad = phi::distributed::DistMetaTensor(phi::make_ddim(out_grad_shape),
                                              out_grad_dist_attr);

  spmdinfo = EmbeddingGradInferSpmd(x, w, out_grad, -1, false);

  EXPECT_EQ(spmdinfo.first.size(), 3UL);
  EXPECT_EQ(spmdinfo.second.size(), 1UL);

  EXPECT_EQ(get_dims_mapping(spmdinfo.first[0]), std::vector<int64_t>({-1, 1}));
  EXPECT_EQ(get_dims_mapping(spmdinfo.first[1]), std::vector<int64_t>({-1, 1}));
  EXPECT_EQ(get_dims_mapping(spmdinfo.first[2]),
            std::vector<int64_t>({-1, 1, -1}));

  EXPECT_EQ(get_dims_mapping(spmdinfo.second[0]),
            std::vector<int64_t>({-1, -1}));
  EXPECT_EQ(
      PADDLE_GET_CONST(phi::distributed::TensorDistAttr, spmdinfo.second[0])
          .is_partial(),
      true);
  VLOG(4) << "Test EmbeddingGradInferSpmd with replicating indices and "
             "sharding weight along col axis."
          << std::endl
          << std::endl
          << std::endl;

  // Indices' rank equals 1, indices and out_grad is sharded.
  x_shape = {5};
  w_shape = {10, 3};
  out_grad_shape = {5, 3};

  x_dist_attr.set_dims_mapping(std::vector<int64_t>({0}));
  w_dist_attr.set_dims_mapping(std::vector<int64_t>({-1, -1}));
  out_grad_dist_attr.set_dims_mapping(std::vector<int64_t>({-1, 1}));

  x = phi::distributed::DistMetaTensor(phi::make_ddim(x_shape), x_dist_attr);
  w = phi::distributed::DistMetaTensor(phi::make_ddim(w_shape), w_dist_attr);
  out_grad = phi::distributed::DistMetaTensor(phi::make_ddim(out_grad_shape),
                                              out_grad_dist_attr);

  spmdinfo = EmbeddingGradInferSpmd(x, w, out_grad, -1, false);

  EXPECT_EQ(spmdinfo.first.size(), 3UL);
  EXPECT_EQ(spmdinfo.second.size(), 1UL);

  EXPECT_EQ(get_dims_mapping(spmdinfo.first[0]), std::vector<int64_t>({0}));
  EXPECT_EQ(get_dims_mapping(spmdinfo.first[1]),
            std::vector<int64_t>({-1, -1}));
  EXPECT_EQ(get_dims_mapping(spmdinfo.first[2]), std::vector<int64_t>({0, 1}));

  EXPECT_EQ(get_dims_mapping(spmdinfo.second[0]),
            std::vector<int64_t>({-1, 1}));
  EXPECT_EQ(
      PADDLE_GET_CONST(phi::distributed::TensorDistAttr, spmdinfo.second[0])
          .is_partial(),
      true);
  VLOG(4) << "Test EmbeddingGradInferSpmd with sharding weight and out_grad."
          << std::endl
          << std::endl
          << std::endl;

  x_shape = {12, 16};
  w_shape = {10, 4};
  out_grad_shape = {12, 16, 4};

  x_dist_attr.set_dims_mapping(std::vector<int64_t>({-1, -1}));
  w_dist_attr.set_dims_mapping(std::vector<int64_t>({-1, -0}));
  out_grad_dist_attr.set_dims_mapping(std::vector<int64_t>({-1, -1, 0}));

  x = phi::distributed::DistMetaTensor(phi::make_ddim(x_shape), x_dist_attr);
  w = phi::distributed::DistMetaTensor(phi::make_ddim(w_shape), w_dist_attr);
  out_grad = phi::distributed::DistMetaTensor(phi::make_ddim(out_grad_shape),
                                              out_grad_dist_attr);

  spmdinfo = EmbeddingGradInferSpmd(x, w, out_grad, -1, false);

  EXPECT_EQ(spmdinfo.first.size(), 3UL);
  EXPECT_EQ(spmdinfo.second.size(), 1UL);

  EXPECT_EQ(get_dims_mapping(spmdinfo.first[0]),
            std::vector<int64_t>({-1, -1}));
  EXPECT_EQ(get_dims_mapping(spmdinfo.first[1]), std::vector<int64_t>({-1, 0}));
  EXPECT_EQ(get_dims_mapping(spmdinfo.first[2]),
            std::vector<int64_t>({-1, -1, 0}));

  EXPECT_EQ(get_dims_mapping(spmdinfo.second[0]),
            std::vector<int64_t>({-1, 0}));
  EXPECT_EQ(
      PADDLE_GET_CONST(phi::distributed::TensorDistAttr, spmdinfo.second[0])
          .is_partial(),
      false);
  VLOG(4) << "Test EmbeddingGradInferSpmd with sharding weight and out_grad."
          << std::endl
          << std::endl
          << std::endl;
}

TEST(SqueezeGradInferSpmd, Ctor) {
  std::vector<int64_t> x_shape = {1, 32, 1, 48};
  std::vector<int64_t> out_grad_shape = {32, 48};

  std::vector<int64_t> mesh_shape = {2, 3};
  std::vector<int64_t> process_ids = {0, 1, 2, 3, 4, 5};
  std::vector<std::string> dim_names = {"x", "y"};
  ProcessMesh process_mesh(mesh_shape, process_ids, dim_names);

  TensorDistAttr x_dist_attr = TensorDistAttr();
  x_dist_attr.set_process_mesh(process_mesh);
  x_dist_attr.set_dims_mapping(std::vector<int64_t>({-1, 1, -1, -1}));
  x_dist_attr.set_dynamic_dims(std::vector<bool>({false, false, false, false}));

  TensorDistAttr out_grad_dist_attr = TensorDistAttr();
  out_grad_dist_attr.set_process_mesh(process_mesh);
  out_grad_dist_attr.set_dims_mapping(std::vector<int64_t>({-1, 1}));
  out_grad_dist_attr.set_dynamic_dims(std::vector<bool>({false, false}));

  phi::distributed::DistMetaTensor x(phi::make_ddim(x_shape), x_dist_attr);
  phi::distributed::DistMetaTensor out_grad(phi::make_ddim(out_grad_shape),
                                            out_grad_dist_attr);

  auto spmdinfo = SqueezeGradInferSpmd(x, out_grad);

  EXPECT_EQ(spmdinfo.first.size(), 2UL);
  EXPECT_EQ(spmdinfo.second.size(), 1UL);

  EXPECT_EQ(get_dims_mapping(spmdinfo.first[0]),
            std::vector<int64_t>({-1, 1, -1, -1}));
  EXPECT_EQ(get_dims_mapping(spmdinfo.first[1]), std::vector<int64_t>({-1, 1}));
  EXPECT_EQ(get_dims_mapping(spmdinfo.second[0]),
            std::vector<int64_t>({-1, -1, -1, 1}));
  EXPECT_DOUBLE_EQ(
      PADDLE_GET_CONST(TensorDistAttr, spmdinfo.second[0]).is_partial(), false);

  x_dist_attr.set_dims_mapping({-1, 0, -1, 1});
  out_grad_dist_attr.set_dims_mapping({0, 1});
  x = phi::distributed::DistMetaTensor(phi::make_ddim(x_shape), x_dist_attr);
  out_grad = phi::distributed::DistMetaTensor(phi::make_ddim(out_grad_shape),
                                              out_grad_dist_attr);

  spmdinfo = SqueezeGradInferSpmd(x, out_grad);

  EXPECT_EQ(spmdinfo.first.size(), 2UL);
  EXPECT_EQ(spmdinfo.second.size(), 1UL);

  EXPECT_EQ(get_dims_mapping(spmdinfo.first[0]),
            std::vector<int64_t>({-1, 0, -1, 1}));
  EXPECT_EQ(get_dims_mapping(spmdinfo.first[1]), std::vector<int64_t>({0, 1}));
  EXPECT_EQ(get_dims_mapping(spmdinfo.second[0]),
            std::vector<int64_t>({-1, 0, -1, 1}));
  EXPECT_DOUBLE_EQ(
      PADDLE_GET_CONST(TensorDistAttr, spmdinfo.second[0]).is_partial(), false);
}

TEST(UnsqueezeGradInferSpmd, Ctor) {
  std::vector<int64_t> x_shape = {32, 48};
  std::vector<int64_t> out_grad_shape = {1, 32, 1, 48};

  std::vector<int64_t> mesh_shape = {2, 3};
  std::vector<int64_t> process_ids = {0, 1, 2, 3, 4, 5};
  std::vector<std::string> dim_names = {"x", "y"};
  ProcessMesh process_mesh(mesh_shape, process_ids, dim_names);

  TensorDistAttr x_dist_attr = TensorDistAttr();
  x_dist_attr.set_process_mesh(process_mesh);
  x_dist_attr.set_dims_mapping(std::vector<int64_t>({1, -1}));
  x_dist_attr.set_dynamic_dims(std::vector<bool>({false, false}));

  TensorDistAttr out_grad_dist_attr = TensorDistAttr();
  out_grad_dist_attr.set_process_mesh(process_mesh);
  out_grad_dist_attr.set_dims_mapping(std::vector<int64_t>({-1, 1, -1, -1}));
  out_grad_dist_attr.set_dynamic_dims(
      std::vector<bool>({false, false, false, false}));

  phi::distributed::DistMetaTensor x(phi::make_ddim(x_shape), x_dist_attr);
  phi::distributed::DistMetaTensor out_grad(phi::make_ddim(out_grad_shape),
                                            out_grad_dist_attr);

  auto spmdinfo = UnsqueezeGradInferSpmd(x, out_grad);

  EXPECT_EQ(spmdinfo.first.size(), 2UL);
  EXPECT_EQ(spmdinfo.second.size(), 1UL);

  EXPECT_EQ(get_dims_mapping(spmdinfo.first[0]), std::vector<int64_t>({1, -1}));
  EXPECT_EQ(get_dims_mapping(spmdinfo.first[1]),
            std::vector<int64_t>({-1, 1, -1, -1}));
  EXPECT_EQ(get_dims_mapping(spmdinfo.second[0]),
            std::vector<int64_t>({1, -1}));
  EXPECT_DOUBLE_EQ(
      PADDLE_GET_CONST(TensorDistAttr, spmdinfo.second[0]).is_partial(), false);

  x_dist_attr.set_dims_mapping({0, 1});
  out_grad_dist_attr.set_dims_mapping({-1, 0, -1, 1});
  x = phi::distributed::DistMetaTensor(phi::make_ddim(x_shape), x_dist_attr);
  out_grad = phi::distributed::DistMetaTensor(phi::make_ddim(out_grad_shape),
                                              out_grad_dist_attr);

  spmdinfo = UnsqueezeGradInferSpmd(x, out_grad);

  EXPECT_EQ(spmdinfo.first.size(), 2UL);
  EXPECT_EQ(spmdinfo.second.size(), 1UL);

  EXPECT_EQ(get_dims_mapping(spmdinfo.first[0]), std::vector<int64_t>({0, 1}));
  EXPECT_EQ(get_dims_mapping(spmdinfo.first[1]),
            std::vector<int64_t>({-1, 0, -1, 1}));
  EXPECT_EQ(get_dims_mapping(spmdinfo.second[0]), std::vector<int64_t>({0, 1}));
  EXPECT_DOUBLE_EQ(
      PADDLE_GET_CONST(TensorDistAttr, spmdinfo.second[0]).is_partial(), false);
}

TEST(ScatterGradInferSpmd, Ctor) {
  std::vector<int64_t> index_shape = {16};
  std::vector<int64_t> updates_shape = {32, 32, 48};
  std::vector<int64_t> out_grad_shape = {64, 32, 48};

  std::vector<int64_t> mesh_shape = {2, 3};
  std::vector<int64_t> process_ids = {0, 1, 2, 3, 4, 5};
  std::vector<std::string> dim_names = {"x", "y"};
  ProcessMesh process_mesh(mesh_shape, process_ids, dim_names);

  TensorDistAttr index_dist_attr = TensorDistAttr();
  index_dist_attr.set_process_mesh(process_mesh);
  TensorDistAttr updates_dist_attr = TensorDistAttr();
  updates_dist_attr.set_process_mesh(process_mesh);
  TensorDistAttr out_grad_dist_attr = TensorDistAttr();
  out_grad_dist_attr.set_process_mesh(process_mesh);

  // [0], [-1, -1, 1], [0, -1, 1] -->
  // inputs: [-1], [-1, -1, 1], [-1, -1, 1]
  // x_grad: [-1, -1, 1], updates_grad: [-1, -1, 1]
  index_dist_attr.set_dims_mapping({0});
  updates_dist_attr.set_dims_mapping({-1, -1, 1});
  out_grad_dist_attr.set_dims_mapping({0, -1, 1});
  phi::distributed::DistMetaTensor index(phi::make_ddim(index_shape),
                                         index_dist_attr);
  phi::distributed::DistMetaTensor updates(phi::make_ddim(updates_shape),
                                           updates_dist_attr);
  phi::distributed::DistMetaTensor out_grad(phi::make_ddim(out_grad_shape),
                                            out_grad_dist_attr);
  auto spmdinfo = ScatterGradInferSpmd(index, updates, out_grad, false);
  EXPECT_EQ(spmdinfo.first.size(), 3UL);
  EXPECT_EQ(spmdinfo.second.size(), 2UL);

  EXPECT_EQ(get_dims_mapping(spmdinfo.first[0]), std::vector<int64_t>({-1}));
  EXPECT_EQ(get_dims_mapping(spmdinfo.first[1]),
            std::vector<int64_t>({-1, -1, 1}));
  EXPECT_EQ(get_dims_mapping(spmdinfo.first[2]),
            std::vector<int64_t>({-1, -1, 1}));
  EXPECT_EQ(get_dims_mapping(spmdinfo.second[0]),
            std::vector<int64_t>({-1, -1, 1}));
  EXPECT_EQ(get_dims_mapping(spmdinfo.second[1]),
            std::vector<int64_t>({-1, -1, 1}));

  // [0], [0, -1, 1], [-1, 0, 1] -->
  // inputs: [-1], [-1, -1, 1], [-1, 0, 1]
  // x_grad: [-1, 0, 1], updates_grad: [-1, 0, 1]
  index_dist_attr.set_dims_mapping({0});
  updates_dist_attr.set_dims_mapping({0, -1, 1});
  out_grad_dist_attr.set_dims_mapping({-1, 0, 1});
  index = phi::distributed::DistMetaTensor(phi::make_ddim(index_shape),
                                           index_dist_attr);
  updates = phi::distributed::DistMetaTensor(phi::make_ddim(updates_shape),
                                             updates_dist_attr);
  out_grad = phi::distributed::DistMetaTensor(phi::make_ddim(out_grad_shape),
                                              out_grad_dist_attr);
  spmdinfo = ScatterGradInferSpmd(index, updates, out_grad, false);
  EXPECT_EQ(spmdinfo.first.size(), 3UL);
  EXPECT_EQ(spmdinfo.second.size(), 2UL);

  EXPECT_EQ(get_dims_mapping(spmdinfo.first[0]), std::vector<int64_t>({-1}));
  EXPECT_EQ(get_dims_mapping(spmdinfo.first[1]),
            std::vector<int64_t>({-1, -1, 1}));
  EXPECT_EQ(get_dims_mapping(spmdinfo.first[2]),
            std::vector<int64_t>({-1, 0, 1}));
  EXPECT_EQ(get_dims_mapping(spmdinfo.second[0]),
            std::vector<int64_t>({-1, 0, 1}));
  EXPECT_EQ(get_dims_mapping(spmdinfo.second[1]),
            std::vector<int64_t>({-1, 0, 1}));
}

TEST(GatherGradInferSpmd, Ctor) {
  std::vector<int64_t> x_shape = {64, 32, 48};
  std::vector<int64_t> index_shape = {16};
  std::vector<int64_t> out_grad_shape = {16, 32, 48};
  phi::Scalar axis(0);

  std::vector<int64_t> mesh_shape = {2, 3};
  std::vector<int64_t> process_ids = {0, 1, 2, 3, 4, 5};

  std::vector<std::string> dim_names = {"x", "y"};
  ProcessMesh process_mesh(mesh_shape, process_ids, dim_names);

  TensorDistAttr x_dist_attr = TensorDistAttr();
  x_dist_attr.set_process_mesh(process_mesh);
  TensorDistAttr index_dist_attr = TensorDistAttr();
  index_dist_attr.set_process_mesh(process_mesh);
  TensorDistAttr out_grad_dist_attr = TensorDistAttr();
  out_grad_dist_attr.set_process_mesh(process_mesh);

  // axis = 0
  // [0, -1, 1], [0], [0, -1, 1] -->
  // inputs: [0, -1, 1], [-1], [-1, -1, 1]
  // x_grad: [-1, -1, 1]
  axis = 0;
  x_dist_attr.set_dims_mapping({0, -1, 1});
  index_dist_attr.set_dims_mapping({0});
  out_grad_dist_attr.set_dims_mapping({0, -1, 1});
  phi::distributed::DistMetaTensor x(phi::make_ddim(x_shape), x_dist_attr);
  phi::distributed::DistMetaTensor index(phi::make_ddim(index_shape),
                                         index_dist_attr);
  phi::distributed::DistMetaTensor out_grad(phi::make_ddim(out_grad_shape),
                                            out_grad_dist_attr);
  auto spmdinfo = GatherGradInferSpmd(x, index, out_grad, axis);
  EXPECT_EQ(spmdinfo.first.size(), 3UL);
  EXPECT_EQ(spmdinfo.second.size(), 1UL);

  EXPECT_EQ(get_dims_mapping(spmdinfo.first[0]),
            std::vector<int64_t>({0, -1, 1}));
  EXPECT_EQ(get_dims_mapping(spmdinfo.first[1]), std::vector<int64_t>({-1}));
  EXPECT_EQ(get_dims_mapping(spmdinfo.first[2]),
            std::vector<int64_t>({-1, -1, 1}));
  EXPECT_EQ(get_dims_mapping(spmdinfo.second[0]),
            std::vector<int64_t>({-1, -1, 1}));

  // 0-d tensor
  // axis = 1
  // [0, -1, 1], [-1], [0, 1] -->
  // inputs: [0, -1, 1], [-1], [0, 1]
  // x_grad: [0, -1, 1]
  axis = 1;
  index_shape = {};
  out_grad_shape = {64, 48};
  x_dist_attr.set_dims_mapping({0, -1, 1});
  index_dist_attr.set_dims_mapping({-1});
  out_grad_dist_attr.set_dims_mapping({0, 1});
  x = phi::distributed::DistMetaTensor(phi::make_ddim(x_shape), x_dist_attr);
  index = phi::distributed::DistMetaTensor(phi::make_ddim(index_shape),
                                           index_dist_attr);
  out_grad = phi::distributed::DistMetaTensor(phi::make_ddim(out_grad_shape),
                                              out_grad_dist_attr);
  spmdinfo = GatherGradInferSpmd(x, index, out_grad, axis);
  EXPECT_EQ(spmdinfo.first.size(), 3UL);
  EXPECT_EQ(spmdinfo.second.size(), 1UL);

  EXPECT_EQ(get_dims_mapping(spmdinfo.first[0]),
            std::vector<int64_t>({0, -1, 1}));
  EXPECT_EQ(get_dims_mapping(spmdinfo.first[1]), std::vector<int64_t>({-1}));
  EXPECT_EQ(get_dims_mapping(spmdinfo.first[2]), std::vector<int64_t>({0, 1}));
  EXPECT_EQ(get_dims_mapping(spmdinfo.second[0]),
            std::vector<int64_t>({0, -1, 1}));
}

TEST(GatherNdGradInferSpmd, Ctor) {
  std::vector<int64_t> x_shape = {32};
  std::vector<int64_t> index_shape = {16};
  std::vector<int64_t> out_grad_shape = {16};

  std::vector<int64_t> mesh_shape = {2, 3};
  std::vector<int64_t> process_ids = {0, 1, 2, 3, 4, 5};

  std::vector<std::string> dim_names = {"x", "y"};
  ProcessMesh process_mesh(mesh_shape, process_ids, dim_names);

  TensorDistAttr x_dist_attr = TensorDistAttr();
  x_dist_attr.set_process_mesh(process_mesh);
  TensorDistAttr index_dist_attr = TensorDistAttr();
  index_dist_attr.set_process_mesh(process_mesh);
  TensorDistAttr out_grad_dist_attr = TensorDistAttr();
  out_grad_dist_attr.set_process_mesh(process_mesh);

  // inputs: [-1], [0] --> [0]
  // x_grad: [-1]
  x_dist_attr.set_dims_mapping({-1});
  index_dist_attr.set_dims_mapping({0});
  out_grad_dist_attr.set_dims_mapping({0});
  phi::distributed::DistMetaTensor x(phi::make_ddim(x_shape), x_dist_attr);
  phi::distributed::DistMetaTensor index(phi::make_ddim(index_shape),
                                         index_dist_attr);
  phi::distributed::DistMetaTensor out_grad(phi::make_ddim(out_grad_shape),
                                            out_grad_dist_attr);
  auto spmdinfo = GatherNdGradInferSpmd(x, index, out_grad);
  EXPECT_EQ(spmdinfo.first.size(), 3UL);
  EXPECT_EQ(spmdinfo.second.size(), 1UL);

  EXPECT_EQ(get_dims_mapping(spmdinfo.first[0]), std::vector<int64_t>({-1}));
  EXPECT_EQ(get_dims_mapping(spmdinfo.first[1]), std::vector<int64_t>({0}));
  EXPECT_EQ(get_dims_mapping(spmdinfo.first[2]), std::vector<int64_t>({0}));
  EXPECT_EQ(get_dims_mapping(spmdinfo.second[0]), std::vector<int64_t>({-1}));

  // inputs: [-1, -1], [0, -1, -1] --> [0, -1, -1]
  // x_grad: [-1, -1]
  x_shape = {64, 32};
  index_shape = {16, 16, 1};
  out_grad_shape = {16, 16, 32};
  x_dist_attr.set_dims_mapping({-1, -1});
  index_dist_attr.set_dims_mapping({0, -1, -1});
  out_grad_dist_attr.set_dims_mapping({0, -1, -1});
  x = phi::distributed::DistMetaTensor(phi::make_ddim(x_shape), x_dist_attr);
  index = phi::distributed::DistMetaTensor(phi::make_ddim(index_shape),
                                           index_dist_attr);
  out_grad = phi::distributed::DistMetaTensor(phi::make_ddim(out_grad_shape),
                                              out_grad_dist_attr);
  spmdinfo = GatherNdGradInferSpmd(x, index, out_grad);
  EXPECT_EQ(spmdinfo.first.size(), 3UL);
  EXPECT_EQ(spmdinfo.second.size(), 1UL);

  EXPECT_EQ(get_dims_mapping(spmdinfo.first[0]),
            std::vector<int64_t>({-1, -1}));
  EXPECT_EQ(get_dims_mapping(spmdinfo.first[1]),
            std::vector<int64_t>({0, -1, -1}));
  EXPECT_EQ(get_dims_mapping(spmdinfo.first[2]),
            std::vector<int64_t>({0, -1, -1}));
  EXPECT_EQ(get_dims_mapping(spmdinfo.second[0]),
            std::vector<int64_t>({-1, -1}));
}

TEST(CumSumGradInferSpmd, Ctor) {
  std::vector<int64_t> x_shape = {64, 32, 48};
  std::vector<int64_t> out_grad_shape = {64, 32, 48};

  std::vector<int64_t> mesh_shape = {2, 4};
  std::vector<int64_t> process_ids = {0, 1, 2, 3, 4, 5, 6, 7};

  std::vector<std::string> dim_names = {"x", "y"};
  ProcessMesh process_mesh(mesh_shape, process_ids, dim_names);

  TensorDistAttr x_dist_attr = TensorDistAttr();
  x_dist_attr.set_process_mesh(process_mesh);
  TensorDistAttr out_grad_dist_attr = TensorDistAttr();
  out_grad_dist_attr.set_process_mesh(process_mesh);

  // axis = 1
  // [0, 1, -1], [0, 1, -1] -->
  // inputs: [0, 1, -1], [0, -1, -1]
  // x_grad: [0, -1, -1]
  x_dist_attr.set_dims_mapping({0, 1, -1});
  out_grad_dist_attr.set_dims_mapping({0, 1, -1});
  phi::distributed::DistMetaTensor x(phi::make_ddim(x_shape), x_dist_attr);
  phi::distributed::DistMetaTensor out_grad(phi::make_ddim(out_grad_shape),
                                            out_grad_dist_attr);
  auto spmdinfo = CumSumGradInferSpmd(x, out_grad, 1, false, false, false);
  EXPECT_EQ(spmdinfo.first.size(), 2UL);
  EXPECT_EQ(spmdinfo.second.size(), 1UL);

  EXPECT_EQ(get_dims_mapping(spmdinfo.first[0]),
            std::vector<int64_t>({0, 1, -1}));
  EXPECT_EQ(get_dims_mapping(spmdinfo.first[1]),
            std::vector<int64_t>({0, -1, -1}));
  EXPECT_EQ(get_dims_mapping(spmdinfo.second[0]),
            std::vector<int64_t>({0, -1, -1}));

  // axis = -1
  // flatten = true
  // [0, 1, -1], [-1] -->
  // inputs: [0, 1, -1], [-1]
  // x_grad: [-1, -1, -1]
  x_dist_attr.set_dims_mapping({0, 1, -1});
  out_grad_dist_attr.set_dims_mapping({-1});
  x = phi::distributed::DistMetaTensor(phi::make_ddim(x_shape), x_dist_attr);
  out_grad = phi::distributed::DistMetaTensor(phi::make_ddim({64 * 32 * 48}),
                                              out_grad_dist_attr);
  spmdinfo = CumSumGradInferSpmd(x, out_grad, -1, true, false, false);
  EXPECT_EQ(spmdinfo.first.size(), 2UL);
  EXPECT_EQ(spmdinfo.second.size(), 1UL);

  EXPECT_EQ(get_dims_mapping(spmdinfo.first[0]),
            std::vector<int64_t>({0, 1, -1}));
  EXPECT_EQ(get_dims_mapping(spmdinfo.first[1]), std::vector<int64_t>({-1}));
  EXPECT_EQ(get_dims_mapping(spmdinfo.second[0]),
            std::vector<int64_t>({-1, -1, -1}));
}

TEST(Flatten, Ctor) {
  std::vector<int64_t> mesh_shape = {2, 2};
  std::vector<int64_t> process_ids = {0, 1, 2, 3};
  std::vector<std::string> dim_names = {"x", "y"};
  ProcessMesh process_mesh(mesh_shape, process_ids, dim_names);

  auto build_input = [&](const std::vector<int64_t>& shape,
                         const std::vector<int64_t>& dim_mapping) {
    auto t_dist_attr = TensorDistAttr();
    t_dist_attr.set_process_mesh(process_mesh);
    t_dist_attr.set_dims_mapping(dim_mapping);
    t_dist_attr.set_dynamic_dims(std::vector<bool>(shape.size(), false));
    auto input =
        phi::distributed::DistMetaTensor(common::make_ddim(shape), t_dist_attr);
    return input;
  };

  // [b, h/ph, w/pw, c, ph, pw]; dp
  auto input1 = build_input({4, 16, 16, 4, 2, 2}, {0, -1, -1, -1, -1, -1});
  // [b, h/ph, w/pw, c, ph, pw] => [b, h/ph, w/pw, hidden_size]
  auto spmd1 = FlattenInferSpmd(input1, -3, -1);
  EXPECT_EQ(spmd1.first.size(), static_cast<size_t>(1));
  EXPECT_EQ(spmd1.second.size(), static_cast<size_t>(1));
  check_dim_mapping(spmd1.first[0], {0, -1, -1, -1, -1, -1});
  check_dim_mapping(spmd1.second[0], {0, -1, -1, -1});

  // [b, h/ph, w/pw, c, ph, pw]; dp, mp
  auto input2 = build_input({4, 16, 16, 4, 2, 2}, {-1, 0, -1, 1, -1, -1});
  auto spmd2 = FlattenInferSpmd(input2, 1, 4);
  EXPECT_EQ(spmd2.first.size(), static_cast<size_t>(1));
  EXPECT_EQ(spmd2.second.size(), static_cast<size_t>(1));
  check_dim_mapping(spmd2.first[0], {-1, 0, -1, -1, -1, -1});
  check_dim_mapping(spmd2.second[0], {-1, 0, -1});

  // [b, s, nh, h/nh]; dp , mp
  auto input3 = build_input({2, 1024, 32, 32}, {0, -1, 1, -1});
  // [b, s, nh, h/nh] => [b, s, h]
  auto spmd3 = FlattenInferSpmd(input3, 2, 3);
  EXPECT_EQ(spmd3.first.size(), static_cast<size_t>(1));
  EXPECT_EQ(spmd3.second.size(), static_cast<size_t>(1));
  check_dim_mapping(spmd3.first[0], {0, -1, 1, -1});
  check_dim_mapping(spmd3.second[0], {0, -1, 1});

  // [b, c, d, h, w]; dp, mp
  auto input4 = build_input({4, 16, 16, 4, 16}, {-1, -1, 0, 1, -1});
  auto spmd4 = FlattenInferSpmd(input4, 1, 4);
  EXPECT_EQ(spmd4.first.size(), static_cast<size_t>(1));
  EXPECT_EQ(spmd4.second.size(), static_cast<size_t>(1));
  check_dim_mapping(spmd4.first[0], {-1, -1, -1, -1, -1});
  check_dim_mapping(spmd4.second[0], {-1, -1});

  auto out_grad = build_input({2, 1024, 1024}, {0, -1, 1});
  auto x = build_input({2, 1024, 4, 1024 / 4}, {0, -1, 1, -1});
  auto spmd_grad = FlattenGradInferSpmd(x, out_grad);
  EXPECT_EQ(spmd_grad.first.size(), static_cast<size_t>(2));
  EXPECT_EQ(spmd_grad.second.size(), static_cast<size_t>(1));
  check_dim_mapping(spmd_grad.first[0], {0, -1, 1, -1});
  check_dim_mapping(spmd_grad.first[1], {0, -1, 1});
  check_dim_mapping(spmd_grad.second[0], {0, -1, 1, -1});
}

TEST(Conv2dSPMDRuleInferForward, Ctor) {
  // build input data class
  std::vector<int64_t> input_shape = {2, 4, 8, 8};
  std::vector<int64_t> filter_shape = {10, 4, 3, 3};

  std::vector<int64_t> mesh_shape = {2, 2, 2};
  std::vector<int64_t> process_ids = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<std::string> dim_names = {"dp", "mp", "pp"};
  ProcessMesh process_mesh(mesh_shape, process_ids, dim_names);

  TensorDistAttr input_dist_attr = TensorDistAttr();
  input_dist_attr.set_process_mesh(process_mesh);
  input_dist_attr.set_dims_mapping(std::vector<int64_t>({-1, -1, -1, -1}));
  input_dist_attr.set_dynamic_dims(
      std::vector<bool>({false, false, false, false}));

  TensorDistAttr filter_dist_attr = TensorDistAttr();
  filter_dist_attr.set_process_mesh(process_mesh);
  filter_dist_attr.set_dims_mapping(std::vector<int64_t>({-1, -1, -1, -1}));
  filter_dist_attr.set_dynamic_dims(
      std::vector<bool>({false, false, false, false}));

  phi::distributed::DistMetaTensor input(common::make_ddim(input_shape),
                                         input_dist_attr);
  phi::distributed::DistMetaTensor filter(common::make_ddim(filter_shape),
                                          filter_dist_attr);

  // test 1
  input_dist_attr.set_dims_mapping(std::vector<int64_t>({0, -1, -1, -1}));
  filter_dist_attr.set_dims_mapping(std::vector<int64_t>({-1, -1, -1, -1}));

  input = phi::distributed::DistMetaTensor(common::make_ddim(input_shape),
                                           input_dist_attr);
  filter = phi::distributed::DistMetaTensor(common::make_ddim(filter_shape),
                                            filter_dist_attr);
  auto inferred_dist_attrs = phi::distributed::Conv2dInferSpmd(input, filter);

  check_dim_mapping(inferred_dist_attrs.first[0], {0, -1, -1, -1});
  check_dim_mapping(inferred_dist_attrs.first[1], {-1, -1, -1, -1});
  check_dim_mapping(inferred_dist_attrs.second[0], {0, -1, -1, -1});

  // test 2
  input_dist_attr.set_dims_mapping(std::vector<int64_t>({-1, -1, -1, -1}));
  filter_dist_attr.set_dims_mapping(std::vector<int64_t>({0, -1, -1, -1}));

  input = phi::distributed::DistMetaTensor(common::make_ddim(input_shape),
                                           input_dist_attr);
  filter = phi::distributed::DistMetaTensor(common::make_ddim(filter_shape),
                                            filter_dist_attr);
  inferred_dist_attrs = phi::distributed::Conv2dInferSpmd(input, filter);

  check_dim_mapping(inferred_dist_attrs.first[0], {-1, -1, -1, -1});
  check_dim_mapping(inferred_dist_attrs.first[1], {0, -1, -1, -1});
  check_dim_mapping(inferred_dist_attrs.second[0], {-1, 0, -1, -1});

  // test 3
  input_dist_attr.set_dims_mapping(std::vector<int64_t>({0, -1, -1, -1}));
  filter_dist_attr.set_dims_mapping(std::vector<int64_t>({1, -1, -1, -1}));

  input = phi::distributed::DistMetaTensor(common::make_ddim(input_shape),
                                           input_dist_attr);
  filter = phi::distributed::DistMetaTensor(common::make_ddim(filter_shape),
                                            filter_dist_attr);
  inferred_dist_attrs = phi::distributed::Conv2dInferSpmd(input, filter);

  check_dim_mapping(inferred_dist_attrs.first[0], {0, -1, -1, -1});
  check_dim_mapping(inferred_dist_attrs.first[1], {1, -1, -1, -1});
  check_dim_mapping(inferred_dist_attrs.second[0], {0, 1, -1, -1});

  // test 4
  input_dist_attr.set_dims_mapping(std::vector<int64_t>({-1, 0, -1, -1}));
  filter_dist_attr.set_dims_mapping(std::vector<int64_t>({-1, 0, -1, -1}));

  input = phi::distributed::DistMetaTensor(common::make_ddim(input_shape),
                                           input_dist_attr);
  filter = phi::distributed::DistMetaTensor(common::make_ddim(filter_shape),
                                            filter_dist_attr);

  inferred_dist_attrs = phi::distributed::Conv2dInferSpmd(input, filter);

  check_dim_mapping(inferred_dist_attrs.first[0], {-1, 0, -1, -1});
  check_dim_mapping(inferred_dist_attrs.first[1], {-1, 0, -1, -1});
  check_dim_mapping(inferred_dist_attrs.second[0], {-1, -1, -1, -1});
  EXPECT_EQ(is_partial(inferred_dist_attrs.second[0]), true);

  // test 5
  input_dist_attr.set_dims_mapping(std::vector<int64_t>({0, 2, -1, -1}));
  filter_dist_attr.set_dims_mapping(std::vector<int64_t>({1, 2, -1, -1}));

  input = phi::distributed::DistMetaTensor(common::make_ddim(input_shape),
                                           input_dist_attr);
  filter = phi::distributed::DistMetaTensor(common::make_ddim(filter_shape),
                                            filter_dist_attr);

  inferred_dist_attrs = phi::distributed::Conv2dInferSpmd(input, filter);

  check_dim_mapping(inferred_dist_attrs.first[0], {0, 2, -1, -1});
  check_dim_mapping(inferred_dist_attrs.first[1], {1, 2, -1, -1});
  check_dim_mapping(inferred_dist_attrs.second[0], {0, 1, -1, -1});
  EXPECT_EQ(is_partial(inferred_dist_attrs.second[0]), true);
}

TEST(Conv2dSPMDRuleInferBackward, Ctor) {
  // build input data class
  std::vector<int64_t> input_shape = {2, 4, 8, 8};
  std::vector<int64_t> filter_shape = {10, 4, 3, 3};
  std::vector<int64_t> output_shape = {2, 10, 6, 6};

  std::vector<int64_t> mesh_shape = {2, 2, 2};
  std::vector<int64_t> process_ids = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<std::string> dim_names = {"dp", "mp", "pp"};
  ProcessMesh process_mesh(mesh_shape, process_ids, dim_names);

  TensorDistAttr input_dist_attr = TensorDistAttr();
  input_dist_attr.set_process_mesh(process_mesh);
  input_dist_attr.set_dims_mapping(std::vector<int64_t>({-1, -1, -1, -1}));
  input_dist_attr.set_dynamic_dims(
      std::vector<bool>({false, false, false, false}));

  TensorDistAttr filter_dist_attr = TensorDistAttr();
  filter_dist_attr.set_process_mesh(process_mesh);
  filter_dist_attr.set_dims_mapping(std::vector<int64_t>({-1, -1, -1, -1}));
  filter_dist_attr.set_dynamic_dims(
      std::vector<bool>({false, false, false, false}));

  TensorDistAttr output_dist_attr = TensorDistAttr();
  output_dist_attr.set_process_mesh(process_mesh);
  output_dist_attr.set_dims_mapping(std::vector<int64_t>({0, 1, -1, -1}));
  output_dist_attr.set_dynamic_dims(
      std::vector<bool>({false, false, false, false}));

  phi::distributed::DistMetaTensor input(common::make_ddim(input_shape),
                                         input_dist_attr);
  phi::distributed::DistMetaTensor filter(common::make_ddim(filter_shape),
                                          filter_dist_attr);
  phi::distributed::DistMetaTensor output(common::make_ddim(output_shape),
                                          output_dist_attr);

  auto inferred_dist_attrs =
      phi::distributed::Conv2dInferSpmdReverse(input, filter, output);

  check_dim_mapping(inferred_dist_attrs.first[0], {0, -1, -1, -1});
  check_dim_mapping(inferred_dist_attrs.first[1], {1, -1, -1, -1});
  check_dim_mapping(inferred_dist_attrs.second[0], {0, 1, -1, -1});
}

TEST(Conv2dGradSPMDRule, Ctor) {
  // build input data class
  std::vector<int64_t> input_shape = {2, 4, 8, 8};
  std::vector<int64_t> filter_shape = {10, 4, 3, 3};
  std::vector<int64_t> output_grad_shape = {2, 10, 6, 6};

  std::vector<int64_t> mesh_shape = {2, 2, 2};
  std::vector<int64_t> process_ids = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<std::string> dim_names = {"dp", "mp", "pp"};
  ProcessMesh process_mesh(mesh_shape, process_ids, dim_names);

  TensorDistAttr input_dist_attr = TensorDistAttr();
  input_dist_attr.set_process_mesh(process_mesh);
  input_dist_attr.set_dims_mapping(std::vector<int64_t>({-1, -1, -1, -1}));
  input_dist_attr.set_dynamic_dims(
      std::vector<bool>({false, false, false, false}));

  TensorDistAttr filter_dist_attr = TensorDistAttr();
  filter_dist_attr.set_process_mesh(process_mesh);
  filter_dist_attr.set_dims_mapping(std::vector<int64_t>({-1, -1, -1, -1}));
  filter_dist_attr.set_dynamic_dims(
      std::vector<bool>({false, false, false, false}));

  TensorDistAttr output_grad_dist_attr = TensorDistAttr();
  output_grad_dist_attr.set_process_mesh(process_mesh);
  output_grad_dist_attr.set_dims_mapping(
      std::vector<int64_t>({-1, -1, -1, -1}));
  output_grad_dist_attr.set_dynamic_dims(
      std::vector<bool>({false, false, false, false}));

  phi::distributed::DistMetaTensor input(common::make_ddim(input_shape),
                                         input_dist_attr);
  phi::distributed::DistMetaTensor filter(common::make_ddim(filter_shape),
                                          filter_dist_attr);
  phi::distributed::DistMetaTensor output_grad(
      common::make_ddim(output_grad_shape), output_grad_dist_attr);

  // test 1
  auto inferred_dist_attrs =
      phi::distributed::Conv2dGradInferSpmd(input, filter, output_grad);
  EXPECT_EQ(inferred_dist_attrs.first.size(), (size_t)3);
  EXPECT_EQ(inferred_dist_attrs.second.size(), (size_t)2);
  check_dim_mapping(inferred_dist_attrs.first[0], {-1, -1, -1, -1});
  check_dim_mapping(inferred_dist_attrs.first[1], {-1, -1, -1, -1});
  check_dim_mapping(inferred_dist_attrs.first[2], {-1, -1, -1, -1});
  check_dim_mapping(inferred_dist_attrs.second[0], {-1, -1, -1, -1});
  check_dim_mapping(inferred_dist_attrs.second[1], {-1, -1, -1, -1});
  EXPECT_EQ(is_partial(inferred_dist_attrs.first[2]), false);
  VLOG(4) << "test 1 done";

  // test 2
  input_dist_attr.set_dims_mapping(std::vector<int64_t>({0, -1, -1, -1}));
  filter_dist_attr.set_dims_mapping(std::vector<int64_t>({-1, -1, -1, -1}));
  output_grad_dist_attr.set_dims_mapping(std::vector<int64_t>({0, -1, -1, -1}));

  input = phi::distributed::DistMetaTensor(common::make_ddim(input_shape),
                                           input_dist_attr);
  filter = phi::distributed::DistMetaTensor(common::make_ddim(filter_shape),
                                            filter_dist_attr);
  output_grad = phi::distributed::DistMetaTensor(
      common::make_ddim(output_grad_shape), output_grad_dist_attr);
  inferred_dist_attrs =
      phi::distributed::Conv2dGradInferSpmd(input, filter, output_grad);

  check_dim_mapping(inferred_dist_attrs.first[0], {0, -1, -1, -1});
  check_dim_mapping(inferred_dist_attrs.first[1], {-1, -1, -1, -1});
  check_dim_mapping(inferred_dist_attrs.first[2], {0, -1, -1, -1});
  check_dim_mapping(inferred_dist_attrs.second[0], {0, -1, -1, -1});
  check_dim_mapping(inferred_dist_attrs.second[1], {-1, -1, -1, -1});
  EXPECT_EQ(is_partial(inferred_dist_attrs.first[2]), false);
  EXPECT_EQ(is_partial(inferred_dist_attrs.second[1]), true);

  // test 3
  input_dist_attr.set_dims_mapping(std::vector<int64_t>({-1, -1, -1, -1}));
  filter_dist_attr.set_dims_mapping(std::vector<int64_t>({0, -1, -1, -1}));
  output_grad_dist_attr.set_dims_mapping(std::vector<int64_t>({-1, 0, -1, -1}));

  input = phi::distributed::DistMetaTensor(common::make_ddim(input_shape),
                                           input_dist_attr);
  filter = phi::distributed::DistMetaTensor(common::make_ddim(filter_shape),
                                            filter_dist_attr);
  output_grad = phi::distributed::DistMetaTensor(
      common::make_ddim(output_grad_shape), output_grad_dist_attr);
  inferred_dist_attrs =
      phi::distributed::Conv2dGradInferSpmd(input, filter, output_grad);

  check_dim_mapping(inferred_dist_attrs.first[0], {-1, -1, -1, -1});
  check_dim_mapping(inferred_dist_attrs.first[1], {0, -1, -1, -1});
  check_dim_mapping(inferred_dist_attrs.first[2], {-1, 0, -1, -1});
  check_dim_mapping(inferred_dist_attrs.second[0], {-1, -1, -1, -1});
  check_dim_mapping(inferred_dist_attrs.second[1], {0, -1, -1, -1});
  EXPECT_EQ(is_partial(inferred_dist_attrs.first[2]), false);
  EXPECT_EQ(is_partial(inferred_dist_attrs.second[0]), true);

  // test 4
  input_dist_attr.set_dims_mapping(std::vector<int64_t>({0, -1, -1, -1}));
  filter_dist_attr.set_dims_mapping(std::vector<int64_t>({1, -1, -1, -1}));
  output_grad_dist_attr.set_dims_mapping(std::vector<int64_t>({0, 1, -1, -1}));

  input = phi::distributed::DistMetaTensor(common::make_ddim(input_shape),
                                           input_dist_attr);
  filter = phi::distributed::DistMetaTensor(common::make_ddim(filter_shape),
                                            filter_dist_attr);
  output_grad = phi::distributed::DistMetaTensor(
      common::make_ddim(output_grad_shape), output_grad_dist_attr);
  inferred_dist_attrs =
      phi::distributed::Conv2dGradInferSpmd(input, filter, output_grad);

  check_dim_mapping(inferred_dist_attrs.first[0], {0, -1, -1, -1});
  check_dim_mapping(inferred_dist_attrs.first[1], {1, -1, -1, -1});
  check_dim_mapping(inferred_dist_attrs.first[2], {0, 1, -1, -1});
  check_dim_mapping(inferred_dist_attrs.second[0], {0, -1, -1, -1});
  check_dim_mapping(inferred_dist_attrs.second[1], {1, -1, -1, -1});
  EXPECT_EQ(is_partial(inferred_dist_attrs.first[2]), false);
  EXPECT_EQ(is_partial(inferred_dist_attrs.second[0]), true);
  EXPECT_EQ(is_partial(inferred_dist_attrs.second[1]), true);

  // test 5
  input_dist_attr.set_dims_mapping(std::vector<int64_t>({0, 2, -1, -1}));
  filter_dist_attr.set_dims_mapping(std::vector<int64_t>({1, 2, -1, -1}));
  output_grad_dist_attr.set_dims_mapping(std::vector<int64_t>({0, 1, -1, -1}));
  output_grad_dist_attr.set_partial_status(std::vector<int64_t>({2}));

  input = phi::distributed::DistMetaTensor(common::make_ddim(input_shape),
                                           input_dist_attr);
  filter = phi::distributed::DistMetaTensor(common::make_ddim(filter_shape),
                                            filter_dist_attr);
  output_grad = phi::distributed::DistMetaTensor(
      common::make_ddim(output_grad_shape), output_grad_dist_attr);

  inferred_dist_attrs =
      phi::distributed::Conv2dGradInferSpmd(input, filter, output_grad);

  check_dim_mapping(inferred_dist_attrs.first[0], {0, 2, -1, -1});
  check_dim_mapping(inferred_dist_attrs.first[1], {1, 2, -1, -1});
  check_dim_mapping(inferred_dist_attrs.first[2], {0, 1, -1, -1});
  check_dim_mapping(inferred_dist_attrs.second[0], {0, 2, -1, -1});
  check_dim_mapping(inferred_dist_attrs.second[1], {1, 2, -1, -1});
  EXPECT_EQ(is_partial(inferred_dist_attrs.first[2]), true);
  EXPECT_EQ(is_partial(inferred_dist_attrs.second[0]), true);
  EXPECT_EQ(is_partial(inferred_dist_attrs.second[1]), true);
}

TEST(Dropout, Ctor) {
  std::vector<int64_t> mesh_shape = {2, 2};
  std::vector<int64_t> process_ids = {0, 1, 2, 3};
  std::vector<std::string> dim_names = {"dp", "mp"};
  ProcessMesh process_mesh(mesh_shape, process_ids, dim_names);

  // test forward
  auto t_dist_attr = TensorDistAttr();
  t_dist_attr.set_process_mesh(process_mesh);
  t_dist_attr.set_dims_mapping({0, -1, -1});
  t_dist_attr.set_dynamic_dims({false, false, false});
  phi::distributed::DistMetaTensor x = phi::distributed::DistMetaTensor(
      common::make_ddim({4, 6, 8}), t_dist_attr);
  phi::distributed::DistMetaTensor seed_tensor;
  phi::distributed::SpmdInfo forward_info =
      phi::distributed::DropoutFwdInferSpmd(
          x, seed_tensor, 0.5, false, "", 0, true);
  check_dim_mapping(forward_info.first[0], {0, -1, -1});
  check_dim_mapping(forward_info.first[1], {});
  check_dim_mapping(forward_info.second[0], {0, -1, -1});
  check_dim_mapping(forward_info.second[1], {0, -1, -1});
  // test backward
  phi::distributed::DistMetaTensor out_grad = phi::distributed::DistMetaTensor(
      common::make_ddim({4, 6, 8}), t_dist_attr);
  phi::distributed::DistMetaTensor mask = out_grad;
  phi::distributed::SpmdInfo backward_info =
      phi::distributed::DropoutBwdInferSpmd(mask, out_grad, 0.5, false, "");
  check_dim_mapping(backward_info.first[0], {0, -1, -1});
  check_dim_mapping(backward_info.first[1], {0, -1, -1});
  check_dim_mapping(backward_info.second[0], {0, -1, -1});
}

}  // namespace auto_parallel
}  // namespace distributed
}  // namespace paddle
