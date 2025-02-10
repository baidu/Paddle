# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.distributed.auto_parallel.static.dist_attribute import (
    DistTensorSpec,
    TensorDistAttr,
)
from paddle.distributed.fleet import auto
from paddle.framework import core


class TestFusedDropoutAddSPMDRule(unittest.TestCase):
    """
    Unit tests for fused_dropout_add spmd rule.
    """

    def setUp(self):
        self.process_mesh = auto.ProcessMesh(mesh=[[0, 1, 2, 3], [4, 5, 6, 7]])
        self.shapes = [[2, 4], [2, 4]]
        self.dim_mappings = [[0, -1], [0, -1]]

    def build_inputs(self):
        inputs = {}
        x_dist_attr = TensorDistAttr()
        x_dist_attr.dims_mapping = [0, -1]
        x_dist_attr.process_mesh = self.process_mesh
        inputs['x'] = DistTensorSpec([2, 4], x_dist_attr)
        y_dist_attr = TensorDistAttr()
        y_dist_attr.dims_mapping = [0, -1]
        y_dist_attr.process_mesh = self.process_mesh
        inputs['y'] = DistTensorSpec([2, 4], y_dist_attr)
        return inputs

    def build_outputs(self):
        outputs = {}
        out_dist_attr = TensorDistAttr()
        out_dist_attr.dims_mapping = [0, -1]
        out_dist_attr.process_mesh = self.process_mesh
        outputs['out'] = DistTensorSpec([2, 4], out_dist_attr)
        seed_offset_dist_attr = TensorDistAttr()
        seed_offset_dist_attr.dims_mapping = [-1]
        seed_offset_dist_attr.process_mesh = self.process_mesh
        outputs['seed_offset'] = DistTensorSpec([2], seed_offset_dist_attr)
        return outputs

    def test_infer_forward(self):
        inputs = self.build_inputs()
        rule = core.get_phi_spmd_rule("fused_dropout_add")
        inferred_dist_attrs = rule.infer_forward(inputs['x'], inputs['y'])

        self.assertEqual(len(inferred_dist_attrs), 2)

        inferred_input_dist_attrs = inferred_dist_attrs[0]
        self.assertEqual(len(inferred_input_dist_attrs), 2)
        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [0, -1])
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [0, -1])

        inferred_output_dist_attrs = inferred_dist_attrs[1]
        self.assertEqual(len(inferred_output_dist_attrs), 2)
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [0, -1])
        self.assertEqual(inferred_output_dist_attrs[1].dims_mapping, [-1])

    def test_infer_backward(self):
        inputs = self.build_inputs()
        outputs = self.build_outputs()
        rule = core.get_phi_spmd_rule("fused_dropout_add")
        inferred_dist_attrs = rule.infer_backward(
            inputs['x'], inputs['y'], outputs['out'], outputs['seed_offset']
        )

        self.assertEqual(len(inferred_dist_attrs), 2)

        inferred_input_dist_attrs = inferred_dist_attrs[0]
        self.assertEqual(len(inferred_input_dist_attrs), 2)
        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [0, -1])
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [0, -1])

        inferred_output_dist_attrs = inferred_dist_attrs[1]
        self.assertEqual(len(inferred_output_dist_attrs), 2)
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [0, -1])
        self.assertEqual(inferred_output_dist_attrs[1].dims_mapping, [-1])


if __name__ == "__main__":
    unittest.main()
