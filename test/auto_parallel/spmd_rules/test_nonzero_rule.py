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


class TestNonZeroSPMDRule(unittest.TestCase):
    """
    Unit tests for nonzero spmd rule.
    """

    def setUp(self):
        self.rule1 = core.get_phi_spmd_rule("nonzero")
        process_mesh = auto.ProcessMesh(mesh=[[0, 1], [2, 3]])
        x_shape = [16, 16, 16]
        x_tensor_dist_attr = TensorDistAttr()
        x_tensor_dist_attr.process_mesh = process_mesh
        self.x_dist_tensor_spec = DistTensorSpec(x_shape, x_tensor_dist_attr)

        output_shape = [8, 8]
        output_tensor_dist_attr = TensorDistAttr()
        output_tensor_dist_attr.process_mesh = process_mesh
        self.output_dist_tensor_spec = DistTensorSpec(
            output_shape, output_tensor_dist_attr
        )

    def test_infer_forward(self):
        # [1, 1, 1] (x) -->
        # [-1, -1, -1], [-1, -1] (x, output)
        self.x_dist_tensor_spec.set_dims_mapping([1, 1, 1])

        inferred_dist_attr = self.rule1.infer_forward(self.x_dist_tensor_spec)

        self.assertEqual(len(inferred_dist_attr), 2)
        inferred_input_dist_attr = inferred_dist_attr[0]
        inferred_output_dist_attr = inferred_dist_attr[1]

        self.assertEqual(len(inferred_input_dist_attr), 1)
        self.assertEqual(len(inferred_output_dist_attr), 1)

        self.assertEqual(inferred_input_dist_attr[0].dims_mapping, [-1, -1, -1])
        self.assertEqual(inferred_output_dist_attr[0].dims_mapping, [-1, -1])

        # [-1, -1, -1] (x) -->
        # [-1, -1, -1], [-1, -1] (x, output)
        self.x_dist_tensor_spec.set_dims_mapping([-1, -1, -1])

        inferred_dist_attr = self.rule1.infer_forward(self.x_dist_tensor_spec)

        self.assertEqual(len(inferred_dist_attr), 2)
        inferred_input_dist_attr = inferred_dist_attr[0]
        inferred_output_dist_attr = inferred_dist_attr[1]

        self.assertEqual(len(inferred_input_dist_attr), 1)
        self.assertEqual(len(inferred_output_dist_attr), 1)

        self.assertEqual(inferred_input_dist_attr[0].dims_mapping, [-1, -1, -1])
        self.assertEqual(inferred_output_dist_attr[0].dims_mapping, [-1, -1])

    def test_infer_reverse(self):
        # [1, 1, 1], [-1, -1] (x, output) -->
        # [-1, -1, -1], [-1, -1] (x, output)
        self.x_dist_tensor_spec.set_dims_mapping([1, 1, 1])
        self.output_dist_tensor_spec.set_dims_mapping([1, 1])

        inferred_dist_attr = self.rule1.infer_backward(
            self.x_dist_tensor_spec, self.output_dist_tensor_spec
        )

        self.assertEqual(len(inferred_dist_attr), 2)
        inferred_input_dist_attr = inferred_dist_attr[0]
        inferred_output_dist_attr = inferred_dist_attr[1]

        self.assertEqual(len(inferred_input_dist_attr), 1)
        self.assertEqual(len(inferred_output_dist_attr), 1)

        self.assertEqual(inferred_input_dist_attr[0].dims_mapping, [-1, -1, -1])
        self.assertEqual(inferred_output_dist_attr[0].dims_mapping, [-1, -1])

        # [-1, -1, -1], [-1, -1] (x, output) -->
        # [-1, -1, -1], [-1, -1] (x, output)
        self.x_dist_tensor_spec.set_dims_mapping([1, 1, 1])
        self.output_dist_tensor_spec.set_dims_mapping([-1, -1])

        inferred_dist_attr = self.rule1.infer_backward(
            self.x_dist_tensor_spec, self.output_dist_tensor_spec
        )

        self.assertEqual(len(inferred_dist_attr), 2)
        inferred_input_dist_attr = inferred_dist_attr[0]
        inferred_output_dist_attr = inferred_dist_attr[1]

        self.assertEqual(len(inferred_input_dist_attr), 1)
        self.assertEqual(len(inferred_output_dist_attr), 1)

        self.assertEqual(inferred_input_dist_attr[0].dims_mapping, [-1, -1, -1])
        self.assertEqual(inferred_output_dist_attr[0].dims_mapping, [-1, -1])


if __name__ == "__main__":
    unittest.main()
