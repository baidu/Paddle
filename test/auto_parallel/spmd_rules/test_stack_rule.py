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


class TestStackSPMDRule(unittest.TestCase):
    """
    Unit tests for split spmd rule.
    """

    def setUp(self):
        self.process_mesh = auto.ProcessMesh(mesh=[[0, 1, 2, 3], [4, 5, 6, 7]])
        self.shapes = [[2, 4], [2, 4]]
        self.dim_mappings = [[0, -1], [0, -1]]

    def build_inputs(self):
        inputs = []
        for shape, dim_mapping in zip(self.shapes, self.dim_mappings):
            tensor_dist_attr = TensorDistAttr()
            tensor_dist_attr.dims_mapping = dim_mapping
            tensor_dist_attr.process_mesh = self.process_mesh
            inputs.append(DistTensorSpec(shape, tensor_dist_attr))
        return inputs

    def build_outputs(self):
        tensor_dist_attr = TensorDistAttr()
        tensor_dist_attr.dims_mapping = [0, -1, -1]
        tensor_dist_attr.process_mesh = self.process_mesh
        return DistTensorSpec([2, 4, 2], tensor_dist_attr)

    def test_infer_forward(self):
        inputs = self.build_inputs()
        rule = core.get_phi_spmd_rule("stack")
        inferred_dist_attrs = rule.infer_forward(inputs, -1)

        inferred_output_dist_attrs = inferred_dist_attrs[1]
        self.assertEqual(len(inferred_output_dist_attrs), 1)
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [0, -1, -1]
        )

    def test_infer_backward(self):
        inputs = self.build_inputs()
        output = self.build_outputs()
        rule = core.get_phi_spmd_rule("stack")
        inferred_dist_attrs = rule.infer_backward(inputs, output, -1)

        inferred_input_dist_attrs = inferred_dist_attrs[0]
        self.assertEqual(len(inferred_input_dist_attrs), 1)
        for input_dist_attr in inferred_input_dist_attrs[0]:
            self.assertEqual(input_dist_attr.dims_mapping, [0, -1])


if __name__ == "__main__":
    unittest.main()
