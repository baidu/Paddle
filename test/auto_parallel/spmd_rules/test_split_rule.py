# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
from collections import OrderedDict

from paddle.distributed.auto_parallel.static.dist_attribute import (
    DistTensorSpec,
    TensorDistAttr,
)
from paddle.distributed.fleet import auto
from paddle.framework import core


class TestReductionSPMDRule(unittest.TestCase):
    """
    Unit tests for split spmd rule.
    """

    def setUp(self):
        x_shape = [64, 32, 48]
        process_mesh = auto.ProcessMesh(mesh=[0, 1, 2, 3])

        x_tensor_dist_attr = TensorDistAttr()
        x_tensor_dist_attr.dims_mapping = [1, 0]
        x_tensor_dist_attr.process_mesh = process_mesh
        self.x_dist_tensor_spec = DistTensorSpec(x_shape, x_tensor_dist_attr)

    def test_single_mesh_dim(self):
        # num_or_sections = 2, axis = 1
        # [0, -1, -1] --> [0, -1, -1], [0, -1, -1], [0, -1, -1]
        self.rule = core.get_phi_spmd_rule("split_with_num")
        self.attrs = OrderedDict()
        self.attrs['num'] = 2
        self.attrs['axis'] = 1
        self.x_dist_tensor_spec.set_dims_mapping([0, -1, -1])
        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec, self.attrs['num'], self.attrs['axis']
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(inferred_input_dist_attrs), 1)
        self.assertEqual(len(inferred_output_dist_attrs), 2)

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [0, -1, -1])
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [0, -1, -1]
        )
        self.assertEqual(
            inferred_output_dist_attrs[1].dims_mapping, [0, -1, -1]
        )

        # num_or_sections = [15, 16, 17], axis = 2
        # [0, -1, -1] --> [0, -1, -1], [0, -1, -1], [0, -1, -1], [0, -1, -1]
        self.rule = core.get_phi_spmd_rule("split")
        self.attrs = OrderedDict()
        self.attrs['sections'] = [15, 16, 17]
        self.attrs['axis'] = 2
        self.x_dist_tensor_spec.set_dims_mapping([0, -1, -1])
        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec, self.attrs['sections'], self.attrs['axis']
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(inferred_input_dist_attrs), 1)
        self.assertEqual(len(inferred_output_dist_attrs), 3)

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [0, -1, -1])
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [0, -1, -1]
        )
        self.assertEqual(
            inferred_output_dist_attrs[1].dims_mapping, [0, -1, -1]
        )
        self.assertEqual(
            inferred_output_dist_attrs[2].dims_mapping, [0, -1, -1]
        )

        # num_or_sections = [15, 16, 17], axis = 2
        # [-1, -1, 0] --> [-1, -1, -1], [-1, -1, -1], [-1, -1, -1], [-1, -1, -1]
        self.attrs = OrderedDict()
        self.attrs['sections'] = [15, 16, 17]
        self.attrs['axis'] = 2
        self.x_dist_tensor_spec.set_dims_mapping([-1, -1, 0])
        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec, self.attrs['sections'], self.attrs['axis']
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(inferred_input_dist_attrs), 1)
        self.assertEqual(len(inferred_output_dist_attrs), 3)

        self.assertEqual(
            inferred_input_dist_attrs[0].dims_mapping, [-1, -1, -1]
        )
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [-1, -1, -1]
        )
        self.assertEqual(
            inferred_output_dist_attrs[1].dims_mapping, [-1, -1, -1]
        )
        self.assertEqual(
            inferred_output_dist_attrs[2].dims_mapping, [-1, -1, -1]
        )

        # num_or_sections = 2, axis = -2
        # [0, -1, -1] --> [0, -1, -1], [0, -1, -1], [0, -1, -1]
        self.rule = core.get_phi_spmd_rule("split_with_num")
        self.attrs = OrderedDict()
        self.attrs['num'] = 2
        self.attrs['axis'] = -2
        self.x_dist_tensor_spec.set_dims_mapping([0, -1, -1])
        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec, self.attrs['num'], self.attrs['axis']
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(inferred_input_dist_attrs), 1)
        self.assertEqual(len(inferred_output_dist_attrs), 2)

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [0, -1, -1])
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [0, -1, -1]
        )
        self.assertEqual(
            inferred_output_dist_attrs[1].dims_mapping, [0, -1, -1]
        )

    def test_multi_mesh_dim(self):
        process_mesh = auto.ProcessMesh(mesh=[[0, 1, 2], [3, 4, 5]])
        self.x_dist_tensor_spec.set_process_mesh(process_mesh)
        self.x_dist_tensor_spec.shape = [96, 32, 48, 24]

        # num_or_sections = 3, axis = -1
        # [0, 1, -1, -1] --> [0, 1, -1, -1], [0, 1, -1, -1], [0, 1, -1, -1], [0, 1, -1, -1]
        self.rule = core.get_phi_spmd_rule("split_with_num")
        self.attrs = OrderedDict()
        self.attrs['num'] = 3
        self.attrs['axis'] = -1
        self.x_dist_tensor_spec.set_dims_mapping([0, 1, -1, -1])
        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec, self.attrs['num'], self.attrs['axis']
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(inferred_input_dist_attrs), 1)
        self.assertEqual(len(inferred_output_dist_attrs), 3)

        self.assertEqual(
            inferred_input_dist_attrs[0].dims_mapping, [0, 1, -1, -1]
        )
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [0, 1, -1, -1]
        )
        self.assertEqual(
            inferred_output_dist_attrs[1].dims_mapping, [0, 1, -1, -1]
        )
        self.assertEqual(
            inferred_output_dist_attrs[2].dims_mapping, [0, 1, -1, -1]
        )

        # num_or_sections = [32, 32, 32], axis = 0
        # [0, 1, -1, -1] --> [-1, 1, -1, -1], [-1, 1, -1, -1], [-1, 1, -1, -1], [-1, 1, -1, -1]
        self.rule = core.get_phi_spmd_rule("split")
        self.attrs = OrderedDict()
        self.attrs['sections'] = [32, 32, 32]
        self.attrs['axis'] = 0
        self.x_dist_tensor_spec.set_dims_mapping([0, 1, -1, -1])
        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec, self.attrs['sections'], self.attrs['axis']
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(inferred_input_dist_attrs), 1)
        self.assertEqual(len(inferred_output_dist_attrs), 3)

        self.assertEqual(
            inferred_input_dist_attrs[0].dims_mapping, [-1, 1, -1, -1]
        )
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [-1, 1, -1, -1]
        )
        self.assertEqual(
            inferred_output_dist_attrs[1].dims_mapping, [-1, 1, -1, -1]
        )
        self.assertEqual(
            inferred_output_dist_attrs[2].dims_mapping, [-1, 1, -1, -1]
        )

    def test_backward_single_mesh_dim(self):
        x_shape = self.x_dist_tensor_spec.shape

        # num_or_sections = 2, axis = 1
        # [0, -1, -1], [0, -1, -1] --> [0, -1, -1], [0, -1, -1], [0, -1, -1]
        # (outputs --> input, outputs)
        self.rule = core.get_phi_spmd_rule("split_with_num")
        self.attrs = OrderedDict()
        self.attrs['num'] = 2
        self.attrs['axis'] = 1
        self.out_spec_list = []
        self.out_spec_list.append(DistTensorSpec(self.x_dist_tensor_spec))
        self.out_spec_list.append(DistTensorSpec(self.x_dist_tensor_spec))
        self.out_spec_list[0].shape = [x_shape[0], x_shape[1] // 2, x_shape[2]]
        self.out_spec_list[1].shape = [x_shape[0], x_shape[1] // 2, x_shape[2]]
        self.out_spec_list[0].set_dims_mapping([0, -1, -1])
        self.out_spec_list[1].set_dims_mapping([0, -1, -1])
        result_dist_attrs = self.rule.infer_backward(
            self.x_dist_tensor_spec,
            self.out_spec_list,
            self.attrs['num'],
            self.attrs['axis'],
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(inferred_input_dist_attrs), 1)
        self.assertEqual(len(inferred_output_dist_attrs), 2)

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [0, -1, -1])
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [0, -1, -1]
        )
        self.assertEqual(
            inferred_output_dist_attrs[1].dims_mapping, [0, -1, -1]
        )

        # num_or_sections = [15, 16, 17], axis = 2
        # [0, -1, -1], [0, -1, -1], [0, -1, -1] -->
        # [0, -1, -1], [0, -1, -1], [0, -1, -1], [0, -1, -1]
        # (outputs --> input, outputs)
        self.rule = core.get_phi_spmd_rule("split")
        self.attrs = OrderedDict()
        self.attrs['sections'] = [15, 16, 17]
        self.attrs['axis'] = 2
        self.out_spec_list = []
        self.out_spec_list.append(DistTensorSpec(self.x_dist_tensor_spec))
        self.out_spec_list.append(DistTensorSpec(self.x_dist_tensor_spec))
        self.out_spec_list.append(DistTensorSpec(self.x_dist_tensor_spec))
        self.out_spec_list[0].shape = [x_shape[0], x_shape[1], 15]
        self.out_spec_list[1].shape = [x_shape[0], x_shape[1], 16]
        self.out_spec_list[2].shape = [x_shape[0], x_shape[1], 17]
        self.out_spec_list[0].set_dims_mapping([0, -1, -1])
        self.out_spec_list[1].set_dims_mapping([0, -1, -1])
        self.out_spec_list[2].set_dims_mapping([0, -1, -1])
        result_dist_attrs = self.rule.infer_backward(
            self.x_dist_tensor_spec,
            self.out_spec_list,
            self.attrs['sections'],
            self.attrs['axis'],
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(inferred_input_dist_attrs), 1)
        self.assertEqual(len(inferred_output_dist_attrs), 3)

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [0, -1, -1])
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [0, -1, -1]
        )
        self.assertEqual(
            inferred_output_dist_attrs[1].dims_mapping, [0, -1, -1]
        )
        self.assertEqual(
            inferred_output_dist_attrs[2].dims_mapping, [0, -1, -1]
        )

        # num_or_sections = [15, 16, 17], axis = 2
        # [-1, -1, -1], [-1, -1, -1], [-1, -1, -1] -->
        # [-1, -1, -1], [-1, -1, -1], [-1, -1, -1], [-1, -1, -1]
        # (outputs --> input, outputs)
        self.attrs = OrderedDict()
        self.attrs['sections'] = [15, 16, 17]
        self.attrs['axis'] = 2
        self.out_spec_list = []
        self.out_spec_list.append(DistTensorSpec(self.x_dist_tensor_spec))
        self.out_spec_list.append(DistTensorSpec(self.x_dist_tensor_spec))
        self.out_spec_list.append(DistTensorSpec(self.x_dist_tensor_spec))
        self.out_spec_list[0].shape = [x_shape[0], x_shape[1], 15]
        self.out_spec_list[1].shape = [x_shape[0], x_shape[1], 16]
        self.out_spec_list[2].shape = [x_shape[0], x_shape[1], 17]
        self.out_spec_list[0].set_dims_mapping([-1, -1, -1])
        self.out_spec_list[1].set_dims_mapping([-1, -1, -1])
        self.out_spec_list[2].set_dims_mapping([-1, -1, -1])
        result_dist_attrs = self.rule.infer_backward(
            self.x_dist_tensor_spec,
            self.out_spec_list,
            self.attrs['sections'],
            self.attrs['axis'],
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(inferred_input_dist_attrs), 1)
        self.assertEqual(len(inferred_output_dist_attrs), 3)

        self.assertEqual(
            inferred_input_dist_attrs[0].dims_mapping, [-1, -1, -1]
        )
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [-1, -1, -1]
        )
        self.assertEqual(
            inferred_output_dist_attrs[1].dims_mapping, [-1, -1, -1]
        )
        self.assertEqual(
            inferred_output_dist_attrs[2].dims_mapping, [-1, -1, -1]
        )

        # num_or_sections = 2, axis = -2
        # [0, -1, -1], [0, -1, -1] --> [0, -1, -1], [0, -1, -1], [0, -1, -1]
        # (outputs --> input, outputs)
        self.rule = core.get_phi_spmd_rule("split_with_num")
        self.attrs = OrderedDict()
        self.attrs['num'] = 2
        self.attrs['axis'] = -2
        self.out_spec_list = []
        self.out_spec_list.append(DistTensorSpec(self.x_dist_tensor_spec))
        self.out_spec_list.append(DistTensorSpec(self.x_dist_tensor_spec))
        self.out_spec_list[0].shape = [x_shape[0], x_shape[1] // 2, x_shape[2]]
        self.out_spec_list[1].shape = [x_shape[0], x_shape[1] // 2, x_shape[2]]
        self.out_spec_list[0].set_dims_mapping([0, -1, -1])
        self.out_spec_list[1].set_dims_mapping([0, -1, -1])
        result_dist_attrs = self.rule.infer_backward(
            self.x_dist_tensor_spec,
            self.out_spec_list,
            self.attrs['num'],
            self.attrs['axis'],
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(inferred_input_dist_attrs), 1)
        self.assertEqual(len(inferred_output_dist_attrs), 2)

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [0, -1, -1])
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [0, -1, -1]
        )
        self.assertEqual(
            inferred_output_dist_attrs[1].dims_mapping, [0, -1, -1]
        )

        # num_or_sections = 2, axis = -2
        # [-1, 0, -1], [-1, -1, -1] --> [-1, -1, -1], [-1, -1, -1], [-1, -1, -1]
        # (outputs --> input, outputs)
        self.rule = core.get_phi_spmd_rule("split_with_num")
        self.attrs = OrderedDict()
        self.attrs['num'] = 2
        self.attrs['axis'] = -2
        self.out_spec_list = []
        self.out_spec_list.append(DistTensorSpec(self.x_dist_tensor_spec))
        self.out_spec_list.append(DistTensorSpec(self.x_dist_tensor_spec))
        self.out_spec_list[0].shape = [x_shape[0], x_shape[1] // 2, x_shape[2]]
        self.out_spec_list[1].shape = [x_shape[0], x_shape[1] // 2, x_shape[2]]
        self.out_spec_list[0].set_dims_mapping([-1, 0, -1])
        self.out_spec_list[1].set_dims_mapping([-1, -1, -1])
        result_dist_attrs = self.rule.infer_backward(
            self.x_dist_tensor_spec,
            self.out_spec_list,
            self.attrs['num'],
            self.attrs['axis'],
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(inferred_input_dist_attrs), 1)
        self.assertEqual(len(inferred_output_dist_attrs), 2)

        self.assertEqual(
            inferred_input_dist_attrs[0].dims_mapping, [-1, -1, -1]
        )
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [-1, -1, -1]
        )
        self.assertEqual(
            inferred_output_dist_attrs[1].dims_mapping, [-1, -1, -1]
        )

    def test_backward_multi_mesh_dim(self):
        x_shape = [96, 32, 48, 24]
        process_mesh = auto.ProcessMesh(mesh=[[0, 1, 2], [3, 4, 5]])
        self.x_dist_tensor_spec.set_process_mesh(process_mesh)
        self.x_dist_tensor_spec.shape = x_shape

        # num_or_sections = 3, axis = -1
        # [0, 1, -1, -1], [0, 1, -1, -1], [0, 1, -1, -1] -->
        # [0, 1, -1, -1], [0, 1, -1, -1], [0, 1, -1, -1], [0, 1, -1, -1]
        # (outputs --> input, outputs)
        self.rule = core.get_phi_spmd_rule("split_with_num")
        self.attrs = OrderedDict()
        self.attrs['num'] = 3
        self.attrs['axis'] = -1
        self.out_spec_list = []
        self.out_spec_list.append(DistTensorSpec(self.x_dist_tensor_spec))
        self.out_spec_list.append(DistTensorSpec(self.x_dist_tensor_spec))
        self.out_spec_list.append(DistTensorSpec(self.x_dist_tensor_spec))
        self.out_spec_list[0].shape = [
            x_shape[0],
            x_shape[1],
            x_shape[2],
            x_shape[3] // 3,
        ]
        self.out_spec_list[1].shape = [
            x_shape[0],
            x_shape[1],
            x_shape[2],
            x_shape[3] // 3,
        ]
        self.out_spec_list[2].shape = [
            x_shape[0],
            x_shape[1],
            x_shape[2],
            x_shape[3] // 3,
        ]
        self.out_spec_list[0].set_dims_mapping([0, 1, -1, -1])
        self.out_spec_list[1].set_dims_mapping([0, 1, -1, -1])
        self.out_spec_list[2].set_dims_mapping([0, 1, -1, -1])
        result_dist_attrs = self.rule.infer_backward(
            self.x_dist_tensor_spec,
            self.out_spec_list,
            self.attrs['num'],
            self.attrs['axis'],
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(inferred_input_dist_attrs), 1)
        self.assertEqual(len(inferred_output_dist_attrs), 3)

        self.assertEqual(
            inferred_input_dist_attrs[0].dims_mapping, [0, 1, -1, -1]
        )
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [0, 1, -1, -1]
        )
        self.assertEqual(
            inferred_output_dist_attrs[1].dims_mapping, [0, 1, -1, -1]
        )
        self.assertEqual(
            inferred_output_dist_attrs[2].dims_mapping, [0, 1, -1, -1]
        )

        # num_or_sections = [32, 32, 32], axis = 0
        # [-1, 1, -1, -1], [-1, 1, -1, -1], [-1, 1, -1, -1] -->
        # [-1, 1, -1, -1], [-1, 1, -1, -1], [-1, 1, -1, -1], [-1, 1, -1, -1]
        # (outputs --> input, outputs)
        self.rule = core.get_phi_spmd_rule("split")
        self.attrs = OrderedDict()
        self.attrs['sections'] = [32, 32, 32]
        self.attrs['axis'] = 0
        self.out_spec_list = []
        self.out_spec_list.append(DistTensorSpec(self.x_dist_tensor_spec))
        self.out_spec_list.append(DistTensorSpec(self.x_dist_tensor_spec))
        self.out_spec_list.append(DistTensorSpec(self.x_dist_tensor_spec))
        self.out_spec_list[0].shape = [32, x_shape[1], x_shape[2], x_shape[3]]
        self.out_spec_list[1].shape = [32, x_shape[1], x_shape[2], x_shape[3]]
        self.out_spec_list[2].shape = [32, x_shape[1], x_shape[2], x_shape[3]]
        self.out_spec_list[0].set_dims_mapping([-1, 1, -1, -1])
        self.out_spec_list[1].set_dims_mapping([-1, 1, -1, -1])
        self.out_spec_list[2].set_dims_mapping([-1, 1, -1, -1])
        result_dist_attrs = self.rule.infer_backward(
            self.x_dist_tensor_spec,
            self.out_spec_list,
            self.attrs['sections'],
            self.attrs['axis'],
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(inferred_input_dist_attrs), 1)
        self.assertEqual(len(inferred_output_dist_attrs), 3)

        self.assertEqual(
            inferred_input_dist_attrs[0].dims_mapping, [-1, 1, -1, -1]
        )
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [-1, 1, -1, -1]
        )
        self.assertEqual(
            inferred_output_dist_attrs[1].dims_mapping, [-1, 1, -1, -1]
        )
        self.assertEqual(
            inferred_output_dist_attrs[2].dims_mapping, [-1, 1, -1, -1]
        )

        # num_or_sections = [32, 32, 32], axis = 2
        # [0, -1, 1, -1], [-1, 1, -1, -1], [-1, -1, -1, -1] -->
        # [0, -1, -1, -1], [0, -1, -1, -1], [0, -1, -1, -1], [0, -1, -1, -1]
        # (outputs --> input, outputs)
        self.rule = core.get_phi_spmd_rule("split")
        self.attrs = OrderedDict()
        self.attrs['sections'] = [32, 32, 32]
        self.attrs['axis'] = 2
        self.out_spec_list = []
        self.out_spec_list.append(DistTensorSpec(self.x_dist_tensor_spec))
        self.out_spec_list.append(DistTensorSpec(self.x_dist_tensor_spec))
        self.out_spec_list.append(DistTensorSpec(self.x_dist_tensor_spec))
        self.out_spec_list[0].shape = [32, x_shape[1], x_shape[2], x_shape[3]]
        self.out_spec_list[1].shape = [32, x_shape[1], x_shape[2], x_shape[3]]
        self.out_spec_list[2].shape = [32, x_shape[1], x_shape[2], x_shape[3]]
        self.out_spec_list[0].set_dims_mapping([0, -1, 1, -1])
        self.out_spec_list[1].set_dims_mapping([-1, 1, -1, -1])
        self.out_spec_list[2].set_dims_mapping([-1, -1, -1, -1])
        result_dist_attrs = self.rule.infer_backward(
            self.x_dist_tensor_spec,
            self.out_spec_list,
            self.attrs['sections'],
            self.attrs['axis'],
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(inferred_input_dist_attrs), 1)
        self.assertEqual(len(inferred_output_dist_attrs), 3)

        self.assertEqual(
            inferred_input_dist_attrs[0].dims_mapping, [0, -1, -1, -1]
        )
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [0, -1, -1, -1]
        )
        self.assertEqual(
            inferred_output_dist_attrs[1].dims_mapping, [0, -1, -1, -1]
        )
        self.assertEqual(
            inferred_output_dist_attrs[2].dims_mapping, [0, -1, -1, -1]
        )


if __name__ == "__main__":
    unittest.main()
