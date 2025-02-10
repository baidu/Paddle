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

using phi::distributed::ArgDistAttr;
using phi::distributed::DistMetaTensor;

void test_moe_gate_dispatch_spmd(
    const std::vector<std::vector<int64_t>>& input_shapes,
    const std::vector<std::vector<int64_t>>& input_dims_mappings,
    const std::pair<std::vector<std::vector<int64_t>>,
                    std::vector<std::vector<int64_t>>>& expected_dims_mappings,
    int64_t k,
    int64_t capacity,
    bool use_pad,
    bool test_bwd_spmd = false) {
  size_t num_inputs = 0;
  if (test_bwd_spmd) {
    num_inputs = 5;
  } else {
    num_inputs = 2;
  }

  EXPECT_EQ(input_shapes.size(), num_inputs)
      << "The number of input_shapes must be" << num_inputs << ", but got "
      << input_shapes.size();
  EXPECT_EQ(input_dims_mappings.size(), num_inputs)
      << "The number of input_dims_mapping must be" << num_inputs
      << ", but got " << input_dims_mappings.size();

  std::vector<int64_t> mesh_shape = {4};
  std::vector<int64_t> process_ids = {0, 1, 2, 3};
  std::vector<std::string> dim_names = {"dpmp"};
  ProcessMesh process_mesh(mesh_shape, process_ids, dim_names);

  std::vector<DistMetaTensor> dist_meta_tensors;
  for (size_t i = 0; i < num_inputs; ++i) {
    TensorDistAttr dist_attr = TensorDistAttr();
    dist_attr.set_process_mesh(process_mesh);

    const std::vector<int64_t>& shape = input_shapes[i];
    const std::vector<int64_t>& dim_mapping = input_dims_mappings[i];
    EXPECT_EQ(shape.size(), dim_mapping.size())
        << "The size of shape and dim_mapping for input " << i
        << " must be the same, but got " << shape.size()
        << " != " << dim_mapping.size();

    dist_attr.set_dims_mapping(dim_mapping);
    dist_attr.set_dynamic_dims(std::vector<bool>(shape.size(), false));

    dist_meta_tensors.push_back(
        DistMetaTensor(common::make_ddim(shape), dist_attr));
  }

  phi::distributed::SpmdInfo spmd_info;
  if (test_bwd_spmd) {
    spmd_info =
        phi::distributed::MoEGateDispatchBwdInferSpmd(dist_meta_tensors[0],
                                                      dist_meta_tensors[1],
                                                      dist_meta_tensors[2],
                                                      dist_meta_tensors[3],
                                                      dist_meta_tensors[4],
                                                      k,
                                                      capacity,
                                                      use_pad);
  } else {
    spmd_info = phi::distributed::MoEGateDispatchFwdInferSpmd(
        dist_meta_tensors[0], dist_meta_tensors[1], k, capacity, use_pad);
  }

  for (size_t i = 0; i < 2; ++i) {
    std::vector<ArgDistAttr> dist_attrs;
    std::vector<std::vector<int64_t>> dims_mappings;
    if (i == 0) {
      dist_attrs = spmd_info.first;
      dims_mappings = expected_dims_mappings.first;
    } else {
      dist_attrs = spmd_info.second;
      dims_mappings = expected_dims_mappings.second;
    }
    EXPECT_EQ(dist_attrs.size(), dims_mappings.size())
        << "The size of dist_attr and expected_dims must be the same, but got "
        << dist_attrs.size() << " != " << dims_mappings.size();

    for (size_t j = 0; j < dist_attrs.size(); ++j) {
      const ArgDistAttr& inferred_attr = dist_attrs[j];
      const std::vector<int64_t>& expected_dims_mapping = dims_mappings[j];
      check_dim_mapping(inferred_attr, expected_dims_mapping);
    }
  }
}

TEST(MoECombineSPMDRule, test_moe_gate_dispatch_spmd) {
  int64_t s = 1024, h = 512, k = 2, e = 8, capacity = 1024;
  bool use_pad = true;
  const std::vector<std::vector<int64_t>>& forward_input_shapes = {{s, h},
                                                                   {s, e}};
  const std::vector<std::vector<int64_t>>& backward_input_shapes = {
      {s, k}, {k, s}, {s, k}, {e, capacity, h}, {s, k}};

  // replicated case, forward
  std::vector<std::vector<int64_t>> input_dims_mappings = {{-1, -1}, {-1, -1}};
  std::pair<std::vector<std::vector<int64_t>>,
            std::vector<std::vector<int64_t>>>
      expected_dims_mappings = {
          {{-1, -1}, {-1, -1}},
          {{-1, -1, -1}, {-1, -1}, {-1, -1}, {-1}, {-1, -1}}};
  test_moe_gate_dispatch_spmd(forward_input_shapes,
                              input_dims_mappings,
                              expected_dims_mappings,
                              k,
                              capacity,
                              use_pad);

  // replicated case, backward
  input_dims_mappings = {{-1, -1}, {-1, -1}, {-1, -1}, {-1, -1, -1}, {-1, -1}};
  expected_dims_mappings = {
      {{-1, -1}, {-1, -1}, {-1, -1}, {-1, -1, -1}, {-1, -1}},
      {{-1, -1}, {-1, -1}}};
  test_moe_gate_dispatch_spmd(backward_input_shapes,
                              input_dims_mappings,
                              expected_dims_mappings,
                              k,
                              capacity,
                              use_pad,
                              true);

  // ep case, forward
  input_dims_mappings = {{0, -1}, {-1, -1}};
  expected_dims_mappings = {{{0, -1}, {0, -1}},
                            {{-1, 0, -1}, {0, -1}, {-1, 0}, {-1}, {0, -1}}};
  test_moe_gate_dispatch_spmd(forward_input_shapes,
                              input_dims_mappings,
                              expected_dims_mappings,
                              k,
                              capacity,
                              use_pad);

  // ep case, backward
  input_dims_mappings = {{0, -1}, {-1, 0}, {0, -1}, {-1, 0, -1}, {0, -1}};
  expected_dims_mappings = {{{0, -1}, {-1, 0}, {0, -1}, {-1, 0, -1}, {0, -1}},
                            {{0, -1}, {0, -1}}};
  test_moe_gate_dispatch_spmd(backward_input_shapes,
                              input_dims_mappings,
                              expected_dims_mappings,
                              k,
                              capacity,
                              use_pad,
                              true);
}

}  // namespace auto_parallel
}  // namespace distributed
}  // namespace paddle
