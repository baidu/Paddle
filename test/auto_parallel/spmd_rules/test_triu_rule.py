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

from paddle.distributed.auto_parallel.static.dist_attribute import (
    DistTensorSpec,
    TensorDistAttr,
)
from paddle.distributed.fleet import auto
from paddle.framework import core


class TestTriuSPMDRule(unittest.TestCase):
    """
    Unit tests for split spmd rule.
    """

    def setUp(self):
        self.process_mesh = auto.ProcessMesh(mesh=[[0, 1], [2, 3]])
        self.shape = [16, 16, 16]
        self.dims_mapping = [0, -1, 1]

    def build_input(self):
        tensor_dist_attr = TensorDistAttr()
        tensor_dist_attr.dims_mapping = self.dims_mapping
        tensor_dist_attr.process_mesh = self.process_mesh
        return DistTensorSpec(self.shape, tensor_dist_attr)

    def test_triu_forward(self):
        input = self.build_input()
        rule = core.get_phi_spmd_rule("triu")
        inferred_dist_attrs = rule.infer_forward(input, 0)
        inferred_input_dist_attrs = inferred_dist_attrs[0]
        self.assertEqual(len(inferred_input_dist_attrs), 1)
        inferred_output_dist_attrs = inferred_dist_attrs[1]
        self.assertEqual(len(inferred_output_dist_attrs), 1)
        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [0, -1, -1])
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [0, -1, -1]
        )

    def test_triu_reverse(self):
        input = self.build_input()
        rule = core.get_phi_spmd_rule("triu")
        inferred_dist_attrs = rule.infer_backward(input, input, 0)
        inferred_input_dist_attrs = inferred_dist_attrs[0]
        self.assertEqual(len(inferred_input_dist_attrs), 1)
        inferred_output_dist_attrs = inferred_dist_attrs[1]
        self.assertEqual(len(inferred_output_dist_attrs), 1)
        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [0, -1, -1])
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [0, -1, -1]
        )

    def test_tril_triu_forward(self):
        input = self.build_input()
        rule = core.get_phi_spmd_rule("tril_triu")
        inferred_dist_attrs = rule.infer_forward(input, 0, False)
        inferred_input_dist_attrs = inferred_dist_attrs[0]
        self.assertEqual(len(inferred_input_dist_attrs), 1)
        inferred_output_dist_attrs = inferred_dist_attrs[1]
        self.assertEqual(len(inferred_output_dist_attrs), 1)
        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [0, -1, -1])
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [0, -1, -1]
        )

    def test_tril_triu_reverse(self):
        input = self.build_input()
        rule = core.get_phi_spmd_rule("tril_triu")
        inferred_dist_attrs = rule.infer_backward(input, input, 0, False)
        inferred_input_dist_attrs = inferred_dist_attrs[0]
        self.assertEqual(len(inferred_input_dist_attrs), 1)
        inferred_output_dist_attrs = inferred_dist_attrs[1]
        self.assertEqual(len(inferred_output_dist_attrs), 1)
        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [0, -1, -1])
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [0, -1, -1]
        )


if __name__ == "__main__":
    unittest.main()
