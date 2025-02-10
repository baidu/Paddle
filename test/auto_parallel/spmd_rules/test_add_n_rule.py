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


class TestAddNSPMDRule(unittest.TestCase):
    """
    Unit tests for add_n spmd rule.
    """

    def setUp(self):
        self.rule1 = core.get_phi_spmd_rule("add_n")
        process_mesh = auto.ProcessMesh(mesh=[[0, 1], [2, 3]])
        self.x_shape = [16, 16, 16]
        self.x_tensor_dist_attr = TensorDistAttr()
        self.x_tensor_dist_attr.process_mesh = process_mesh

        self.y_shape = [16, 16, 16]
        self.y_tensor_dist_attr = TensorDistAttr()
        self.y_tensor_dist_attr.process_mesh = process_mesh

        self.output_shape = [16, 16, 16]
        self.output_tensor_dist_attr = TensorDistAttr()
        self.output_tensor_dist_attr.process_mesh = process_mesh

    def test_infer_forward(self):
        # [0, -1, -1], [-1, -1, -1] (x,y) -->
        # [0, -1, -1], [0, -1, -1] (x, y)
        # [0, -1, -1] (output)
        self.x_dist_tensor_spec = DistTensorSpec(
            self.x_shape, self.x_tensor_dist_attr
        )
        self.x_dist_tensor_spec.set_dims_mapping([0, -1, -1])
        self.y_dist_tensor_spec = DistTensorSpec(
            self.y_shape, self.y_tensor_dist_attr
        )
        self.y_dist_tensor_spec.set_dims_mapping([0, -1, -1])

        inferred_dist_attr = self.rule1.infer_forward(
            [self.x_dist_tensor_spec, self.y_dist_tensor_spec]
        )

        self.assertEqual(len(inferred_dist_attr), 2)
        inferred_input_dist_attr = inferred_dist_attr[0]
        inferred_output_dist_attr = inferred_dist_attr[1]

        self.assertEqual(len(inferred_input_dist_attr), 1)
        self.assertEqual(len(inferred_input_dist_attr[0]), 2)
        self.assertEqual(len(inferred_output_dist_attr), 1)

        self.assertEqual(
            inferred_input_dist_attr[0][0].dims_mapping, [0, -1, -1]
        )
        self.assertEqual(
            inferred_input_dist_attr[0][1].dims_mapping, [0, -1, -1]
        )
        self.assertEqual(inferred_output_dist_attr[0].dims_mapping, [0, -1, -1])

        # [0, -1, -1], [-1, -1, -1] (x, y) partial_dim=[1] -->
        # [0, -1, -1], [0, -1, -1]  (x, y) partial_dim=[1]
        # [0, -1, -1] (output) partial_dim=[1]
        self.x_tensor_dist_attr._set_partial_dims([1])
        self.x_dist_tensor_spec = DistTensorSpec(
            self.x_shape, self.x_tensor_dist_attr
        )
        self.x_dist_tensor_spec.set_dims_mapping([0, -1, -1])
        self.y_tensor_dist_attr._set_partial_dims([1])
        self.y_dist_tensor_spec = DistTensorSpec(
            self.y_shape, self.y_tensor_dist_attr
        )
        self.y_dist_tensor_spec.set_dims_mapping([-1, -1, -1])

        inferred_dist_attr = self.rule1.infer_forward(
            [self.x_dist_tensor_spec, self.y_dist_tensor_spec]
        )

        self.assertEqual(len(inferred_dist_attr), 2)
        inferred_input_dist_attr = inferred_dist_attr[0]
        inferred_output_dist_attr = inferred_dist_attr[1]

        self.assertEqual(len(inferred_input_dist_attr), 1)
        self.assertEqual(len(inferred_input_dist_attr[0]), 2)
        self.assertEqual(len(inferred_output_dist_attr), 1)

        self.assertEqual(
            inferred_input_dist_attr[0][0].dims_mapping, [0, -1, -1]
        )
        self.assertEqual(inferred_input_dist_attr[0][0]._is_partial(), True)
        self.assertEqual(inferred_input_dist_attr[0][0]._partial_dims(), {1})

        self.assertEqual(
            inferred_input_dist_attr[0][1].dims_mapping, [0, -1, -1]
        )
        self.assertEqual(inferred_input_dist_attr[0][1]._is_partial(), True)
        self.assertEqual(inferred_input_dist_attr[0][1]._partial_dims(), {1})

        self.assertEqual(inferred_output_dist_attr[0].dims_mapping, [0, -1, -1])
        self.assertEqual(inferred_output_dist_attr[0]._is_partial(), True)
        self.assertEqual(inferred_output_dist_attr[0]._partial_dims(), {1})

        # [0, -1, -1] partial_dim=[0], [-1, -1, -1]partial_dim=[1] (x,y)  -->
        # [0, -1, -1], [0, -1, -1] (x, y)
        # [0, -1, -1] (output)
        self.x_tensor_dist_attr._clean_partial_dims([1])
        self.x_tensor_dist_attr._set_partial_dims([0])
        self.x_dist_tensor_spec = DistTensorSpec(
            self.x_shape, self.x_tensor_dist_attr
        )
        self.x_dist_tensor_spec.set_dims_mapping([0, -1, -1])

        self.y_tensor_dist_attr._clean_partial_dims([1])
        self.y_tensor_dist_attr._set_partial_dims([1])
        self.y_dist_tensor_spec = DistTensorSpec(
            self.y_shape, self.y_tensor_dist_attr
        )
        self.y_dist_tensor_spec.set_dims_mapping([-1, -1, -1])

        inferred_dist_attr = self.rule1.infer_forward(
            [self.x_dist_tensor_spec, self.y_dist_tensor_spec]
        )

        self.assertEqual(len(inferred_dist_attr), 2)
        inferred_input_dist_attr = inferred_dist_attr[0]
        inferred_output_dist_attr = inferred_dist_attr[1]

        self.assertEqual(len(inferred_input_dist_attr), 1)
        self.assertEqual(len(inferred_input_dist_attr[0]), 2)
        self.assertEqual(len(inferred_output_dist_attr), 1)

        self.assertEqual(
            inferred_input_dist_attr[0][0].dims_mapping, [0, -1, -1]
        )
        self.assertEqual(inferred_input_dist_attr[0][0]._is_partial(), False)

        self.assertEqual(
            inferred_input_dist_attr[0][1].dims_mapping, [0, -1, -1]
        )
        self.assertEqual(inferred_input_dist_attr[0][1]._is_partial(), False)

        self.assertEqual(inferred_output_dist_attr[0].dims_mapping, [0, -1, -1])
        self.assertEqual(inferred_output_dist_attr[0]._is_partial(), False)


if __name__ == "__main__":
    unittest.main()
