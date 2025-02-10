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


class TestElementwiseSPMDRule(unittest.TestCase):
    def setUp(self):
        self.unary_rule = core.get_phi_spmd_rule("relu")
        self.binary_rule = core.get_phi_spmd_rule("add")

        x_shape = [64, 36]
        y_shape = [64, 36]
        process_mesh = auto.ProcessMesh(mesh=[0, 1, 2, 3])

        x_tensor_dist_attr = TensorDistAttr()
        x_tensor_dist_attr.dims_mapping = [1, 0]
        x_tensor_dist_attr.process_mesh = process_mesh
        self.x_dist_tensor_spec = DistTensorSpec(x_shape, x_tensor_dist_attr)

        y_tensor_dist_attr = TensorDistAttr()
        y_tensor_dist_attr.dims_mapping = [0, -1]
        y_tensor_dist_attr.process_mesh = process_mesh
        self.y_dist_tensor_spec = DistTensorSpec(y_shape, y_tensor_dist_attr)

        self.out_dist_tensor_spec = DistTensorSpec(self.x_dist_tensor_spec)

        self.attrs = []

    def test_single_mesh_dim(self):
        # [0, -1], [-1, -1] --> [0, -1], [0, -1], [0, -1]
        self.x_dist_tensor_spec.set_dims_mapping([0, -1])
        self.y_dist_tensor_spec.set_dims_mapping([-1, -1])
        result_dist_attrs = self.binary_rule.infer_forward(
            self.x_dist_tensor_spec, self.y_dist_tensor_spec
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [0, -1])
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [0, -1])
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [0, -1])

        # [0, -1], [-1, 0] --> [0, -1], [0, -1], [0, -1]
        self.x_dist_tensor_spec.set_dims_mapping([0, -1])
        self.y_dist_tensor_spec.set_dims_mapping([-1, 0])
        result_dist_attrs = self.binary_rule.infer_forward(
            self.x_dist_tensor_spec, self.y_dist_tensor_spec
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [0, -1])
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [0, -1])
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [0, -1])

        # [-1, -1], [-1, -1] --> [-1, -1], [-1, -1], [-1, -1]
        self.x_dist_tensor_spec.set_dims_mapping([-1, -1])
        self.y_dist_tensor_spec.set_dims_mapping([-1, -1])

        result_dist_attrs = self.binary_rule.infer_forward(
            self.x_dist_tensor_spec, self.y_dist_tensor_spec
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [-1, -1])
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [-1, -1])
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [-1, -1])

        # [-1, 0]--> [-1, 0], [-1, 0]
        self.x_dist_tensor_spec.set_dims_mapping([-1, 0])

        result_dist_attrs = self.unary_rule.infer_forward(
            self.x_dist_tensor_spec
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [-1, 0])
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [-1, 0])

    def test_single_mesh_dim_broadcast(self):
        self.x_dist_tensor_spec.shape = [64, 36, 12]
        self.y_dist_tensor_spec.shape = [12]

        # [0, -1, -1], [-1] --> [0, -1, -1], [-1], [0, -1, -1]
        self.x_dist_tensor_spec.set_dims_mapping([0, -1, -1])
        self.y_dist_tensor_spec.set_dims_mapping([-1])

        resulted_dist_attrs = self.binary_rule.infer_forward(
            self.x_dist_tensor_spec, self.y_dist_tensor_spec
        )
        inferred_input_dist_attrs = resulted_dist_attrs[0]
        inferred_output_dist_attrs = resulted_dist_attrs[1]

        self.assertEqual(len(resulted_dist_attrs), 2)
        self.assertEqual(len(inferred_input_dist_attrs), 2)
        self.assertEqual(len(inferred_output_dist_attrs), 1)

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [0, -1, -1])
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [-1])
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [0, -1, -1]
        )

        # [-1, 0, -1], [-1] --> [-1, 0, -1], [-1], [-1, 0, -1]
        self.x_dist_tensor_spec.set_dims_mapping([-1, 0, -1])
        self.y_dist_tensor_spec.set_dims_mapping([-1])

        resulted_dist_attrs = self.binary_rule.infer_forward(
            self.x_dist_tensor_spec, self.y_dist_tensor_spec
        )
        inferred_input_dist_attrs = resulted_dist_attrs[0]
        inferred_output_dist_attrs = resulted_dist_attrs[1]

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [-1, 0, -1])
        self.assertEqual((inferred_input_dist_attrs[1].dims_mapping), [-1])
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [-1, 0, -1]
        )

        # [-1, -1, 0], [-1] --> [-1, -1, 0], [0], [-1, -1, 0]
        self.x_dist_tensor_spec.set_dims_mapping([-1, -1, 0])
        self.y_dist_tensor_spec.set_dims_mapping([-1])

        resulted_dist_attrs = self.binary_rule.infer_forward(
            self.x_dist_tensor_spec, self.y_dist_tensor_spec
        )
        inferred_input_dist_attrs = resulted_dist_attrs[0]
        inferred_output_dist_attrs = resulted_dist_attrs[1]

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [-1, -1, 0])
        self.assertEqual((inferred_input_dist_attrs[1].dims_mapping), [0])
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [-1, -1, 0]
        )

        # [-1, -1, -1], [0] --> [-1, -1, 0], [0], [-1, -1, 0]
        self.x_dist_tensor_spec.set_dims_mapping([-1, -1, -1])
        self.y_dist_tensor_spec.set_dims_mapping([0])
        resulted_dist_attrs = self.binary_rule.infer_forward(
            self.x_dist_tensor_spec, self.y_dist_tensor_spec
        )
        inferred_input_dist_attrs = resulted_dist_attrs[0]
        inferred_output_dist_attrs = resulted_dist_attrs[1]

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [-1, -1, 0])
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [0])
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [-1, -1, 0]
        )

        self.x_dist_tensor_spec.shape = [64, 36, 12]
        self.y_dist_tensor_spec.shape = [1, 12]
        # [-1, 0, -1], [-1, -1] --> [-1, 0, -1], [-1, -1], [-1, 0, -1]
        self.x_dist_tensor_spec.set_dims_mapping([-1, 0, -1])
        self.y_dist_tensor_spec.set_dims_mapping([-1, -1])

        resulted_dist_attrs = self.binary_rule.infer_forward(
            self.x_dist_tensor_spec, self.y_dist_tensor_spec
        )
        inferred_input_dist_attrs = resulted_dist_attrs[0]
        inferred_output_dist_attrs = resulted_dist_attrs[1]

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [-1, 0, -1])
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [-1, -1])
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [-1, 0, -1]
        )

        self.x_dist_tensor_spec.shape = [64, 1, 1, 12]
        self.y_dist_tensor_spec.shape = [64, 32, 12]
        # [0, -1, -1, -1], [-1, -1, -1] --> [0, -1, -1, -1], [-1, -1, -1], [0, -1, -1, -1]
        self.x_dist_tensor_spec.set_dims_mapping([0, -1, -1, -1])
        self.y_dist_tensor_spec.set_dims_mapping([-1, -1, -1])

        resulted_dist_attrs = self.binary_rule.infer_forward(
            self.x_dist_tensor_spec, self.y_dist_tensor_spec
        )
        inferred_input_dist_attrs = resulted_dist_attrs[0]
        inferred_output_dist_attrs = resulted_dist_attrs[1]

        self.assertEqual(
            inferred_input_dist_attrs[0].dims_mapping, [0, -1, -1, -1]
        )
        self.assertEqual(
            inferred_input_dist_attrs[1].dims_mapping, [-1, -1, -1]
        )
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [0, -1, -1, -1]
        )

        # [-1, -1, -1, -1], [0, -1, -1] --> [-1, -1, -1, -1], [0, -1, -1], [-1, 0, -1, -1]
        self.x_dist_tensor_spec.set_dims_mapping([-1, -1, -1, -1])
        self.y_dist_tensor_spec.set_dims_mapping([0, -1, -1])

        resulted_dist_attrs = self.binary_rule.infer_forward(
            self.x_dist_tensor_spec, self.y_dist_tensor_spec
        )
        inferred_input_dist_attrs = resulted_dist_attrs[0]
        inferred_output_dist_attrs = resulted_dist_attrs[1]

        self.assertEqual(
            inferred_input_dist_attrs[0].dims_mapping, [-1, -1, -1, -1]
        )
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [0, -1, -1])
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [-1, -0, -1, -1]
        )

    def test_multi_mesh_dim(self):
        process_mesh = auto.ProcessMesh([[0, 1, 2], [3, 4, 5]])
        self.x_dist_tensor_spec.set_process_mesh(process_mesh)
        self.y_dist_tensor_spec.set_process_mesh(process_mesh)
        self.x_dist_tensor_spec.shape = [96, 24, 48]
        self.y_dist_tensor_spec.shape = [96, 24, 48]

        # [0, 1, -1], [-1, -1, -1] --> [0, 1, -1], [0, 1, -1], [0, 1, -1]
        self.x_dist_tensor_spec.set_dims_mapping([0, 1, -1])
        self.y_dist_tensor_spec.set_dims_mapping([-1, -1, -1])

        resulted_dist_attrs = self.binary_rule.infer_forward(
            self.x_dist_tensor_spec, self.y_dist_tensor_spec
        )
        inferred_input_dist_attrs = resulted_dist_attrs[0]
        inferred_output_dist_attrs = resulted_dist_attrs[1]

        self.assertEqual(len(resulted_dist_attrs), 2)
        self.assertEqual(len(inferred_input_dist_attrs), 2)
        self.assertEqual(len(inferred_output_dist_attrs), 1)

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [0, 1, -1])
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [0, 1, -1])
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [0, 1, -1])

        # [0, -1, -1], [-1, 1, 0] --> [0, 1, -1], [0, 1, -1], [0, 1, -1]
        self.x_dist_tensor_spec.set_dims_mapping([0, -1, -1])
        self.y_dist_tensor_spec.set_dims_mapping([-1, 1, 0])
        resulted_dist_attrs = self.binary_rule.infer_forward(
            self.x_dist_tensor_spec, self.y_dist_tensor_spec
        )
        inferred_input_dist_attrs = resulted_dist_attrs[0]
        inferred_output_dist_attrs = resulted_dist_attrs[1]

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [0, 1, -1])
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [0, 1, -1])
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [0, 1, -1])

    def test_multi_mesh_dim_broadcast(self):
        process_mesh = auto.ProcessMesh([[0, 1, 2], [3, 4, 5]])
        self.x_dist_tensor_spec.set_process_mesh(process_mesh)
        self.y_dist_tensor_spec.set_process_mesh(process_mesh)
        self.x_dist_tensor_spec.shape = [96, 24, 48]
        self.y_dist_tensor_spec.shape = [48]

        # [0, -1, -1], [1] --> [0, -1, 1], [1], [0, -1, 1]
        self.x_dist_tensor_spec.set_dims_mapping([0, -1, -1])
        self.y_dist_tensor_spec.set_dims_mapping([1])

        resulted_dist_attrs = self.binary_rule.infer_forward(
            self.x_dist_tensor_spec, self.y_dist_tensor_spec
        )
        inferred_input_dist_attrs = resulted_dist_attrs[0]
        inferred_output_dist_attrs = resulted_dist_attrs[1]

        self.assertEqual(len(resulted_dist_attrs), 2)
        self.assertEqual(len(inferred_input_dist_attrs), 2)
        self.assertEqual(len(inferred_output_dist_attrs), 1)

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [0, -1, 1])
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [1])
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [0, -1, 1])

        # [0, 1, -1], [0] --> [0, 1, -1], [-1], [0, 1, -1]
        self.x_dist_tensor_spec.set_dims_mapping([0, 1, -1])
        self.y_dist_tensor_spec.set_dims_mapping([0])

        resulted_dist_attrs = self.binary_rule.infer_forward(
            self.x_dist_tensor_spec, self.y_dist_tensor_spec
        )
        inferred_input_dist_attrs = resulted_dist_attrs[0]
        inferred_output_dist_attrs = resulted_dist_attrs[1]

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [0, 1, -1])
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [-1])
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [0, 1, -1])

        self.x_dist_tensor_spec.shape = [96, 1, 1, 48]
        self.y_dist_tensor_spec.shape = [96, 24, 48]
        # [-1, -1, -1, 1], [0, -1, 1] --> [-1, -1, -1, 1], [0, -1, 1], [-1, 0, -1, 1]
        self.x_dist_tensor_spec.set_dims_mapping([-1, -1, -1, 1])
        self.y_dist_tensor_spec.set_dims_mapping([0, -1, 1])

        resulted_dist_attrs = self.binary_rule.infer_forward(
            self.x_dist_tensor_spec, self.y_dist_tensor_spec
        )
        inferred_input_dist_attrs = resulted_dist_attrs[0]
        inferred_output_dist_attrs = resulted_dist_attrs[1]

        self.assertEqual(
            inferred_input_dist_attrs[0].dims_mapping, [-1, -1, -1, 1]
        )
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [0, -1, 1])
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [-1, 0, -1, 1]
        )

    def test_backward_single_mesh_dim(self):
        # [0, -1] --> [0, -1], [0, -1], [0, -1] (output --> inputs, output)
        self.out_dist_tensor_spec.set_dims_mapping([0, -1])
        result_dist_attrs = self.binary_rule.infer_backward(
            self.x_dist_tensor_spec,
            self.y_dist_tensor_spec,
            self.out_dist_tensor_spec,
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [0, -1])
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [0, -1])
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [0, -1])

        # [-1, -1] --> [-1, -1], [-1, -1], [-1, -1] (output --> inputs, output)
        self.out_dist_tensor_spec.set_dims_mapping([-1, -1])

        result_dist_attrs = self.binary_rule.infer_backward(
            self.x_dist_tensor_spec,
            self.y_dist_tensor_spec,
            self.out_dist_tensor_spec,
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [-1, -1])
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [-1, -1])
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [-1, -1])

        # [-1, 0]--> [-1, 0], [-1, 0] (output --> inputs, output)
        self.out_dist_tensor_spec.set_dims_mapping([-1, 0])

        result_dist_attrs = self.unary_rule.infer_backward(
            self.x_dist_tensor_spec, self.out_dist_tensor_spec
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [-1, 0])
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [-1, 0])

    def test_backward_single_mesh_dim_broadcast(self):
        self.x_dist_tensor_spec.shape = [64, 36, 12]
        self.y_dist_tensor_spec.shape = [12]
        self.out_dist_tensor_spec.shape = [64, 36, 12]

        # [0, -1, -1] --> [0, -1, -1], [-1], [0, -1, -1] (output --> inputs, output)
        self.out_dist_tensor_spec.set_dims_mapping([0, -1, -1])

        resulted_dist_attrs = self.binary_rule.infer_backward(
            self.x_dist_tensor_spec,
            self.y_dist_tensor_spec,
            self.out_dist_tensor_spec,
        )
        inferred_input_dist_attrs = resulted_dist_attrs[0]
        inferred_output_dist_attrs = resulted_dist_attrs[1]

        self.assertEqual(len(resulted_dist_attrs), 2)
        self.assertEqual(len(inferred_input_dist_attrs), 2)
        self.assertEqual(len(inferred_output_dist_attrs), 1)

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [0, -1, -1])
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [-1])
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [0, -1, -1]
        )

        # [-1, 0, -1] --> [-1, 0, -1], [-1], [-1, 0, -1] (output --> inputs, output)
        self.out_dist_tensor_spec.set_dims_mapping([-1, 0, -1])

        resulted_dist_attrs = self.binary_rule.infer_backward(
            self.x_dist_tensor_spec,
            self.y_dist_tensor_spec,
            self.out_dist_tensor_spec,
        )
        inferred_input_dist_attrs = resulted_dist_attrs[0]
        inferred_output_dist_attrs = resulted_dist_attrs[1]

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [-1, 0, -1])
        self.assertEqual((inferred_input_dist_attrs[1].dims_mapping), [-1])
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [-1, 0, -1]
        )

        # [-1, -1, 0] --> [-1, -1, 0], [0], [-1, -1, 0] (output --> inputs, output)
        self.out_dist_tensor_spec.set_dims_mapping([-1, -1, 0])

        resulted_dist_attrs = self.binary_rule.infer_backward(
            self.x_dist_tensor_spec,
            self.y_dist_tensor_spec,
            self.out_dist_tensor_spec,
        )
        inferred_input_dist_attrs = resulted_dist_attrs[0]
        inferred_output_dist_attrs = resulted_dist_attrs[1]

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [-1, -1, 0])
        self.assertEqual((inferred_input_dist_attrs[1].dims_mapping), [0])
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [-1, -1, 0]
        )

        self.x_dist_tensor_spec.shape = [64, 36, 12]
        self.y_dist_tensor_spec.shape = [1, 12]
        self.out_dist_tensor_spec.shape = [64, 36, 12]
        # [-1, 0, -1] --> [-1, 0, -1], [-1, -1], [-1, 0, -1] (output --> inputs, output)
        self.out_dist_tensor_spec.set_dims_mapping([-1, 0, -1])

        resulted_dist_attrs = self.binary_rule.infer_backward(
            self.x_dist_tensor_spec,
            self.y_dist_tensor_spec,
            self.out_dist_tensor_spec,
        )
        inferred_input_dist_attrs = resulted_dist_attrs[0]
        inferred_output_dist_attrs = resulted_dist_attrs[1]

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [-1, 0, -1])
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [-1, -1])
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [-1, 0, -1]
        )

        self.x_dist_tensor_spec.shape = [64, 1, 1, 12]
        self.y_dist_tensor_spec.shape = [64, 32, 12]
        self.out_dist_tensor_spec.shape = [64, 64, 32, 12]
        # [0, -1, -1, -1] --> [0, -1, -1, -1], [-1, -1, -1], [0, -1, -1, -1] (output --> inputs, output)
        self.out_dist_tensor_spec.set_dims_mapping([0, -1, -1, -1])

        resulted_dist_attrs = self.binary_rule.infer_backward(
            self.x_dist_tensor_spec,
            self.y_dist_tensor_spec,
            self.out_dist_tensor_spec,
        )
        inferred_input_dist_attrs = resulted_dist_attrs[0]
        inferred_output_dist_attrs = resulted_dist_attrs[1]

        self.assertEqual(
            inferred_input_dist_attrs[0].dims_mapping, [0, -1, -1, -1]
        )
        self.assertEqual(
            inferred_input_dist_attrs[1].dims_mapping, [-1, -1, -1]
        )
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [0, -1, -1, -1]
        )

        # [-1, 0, -1, -1] --> [-1, -1, -1, -1], [0, -1, -1], [-1, 0, -1, -1] (output --> inputs, output)
        self.out_dist_tensor_spec.set_dims_mapping([-1, 0, -1, -1])

        resulted_dist_attrs = self.binary_rule.infer_backward(
            self.x_dist_tensor_spec,
            self.y_dist_tensor_spec,
            self.out_dist_tensor_spec,
        )
        inferred_input_dist_attrs = resulted_dist_attrs[0]
        inferred_output_dist_attrs = resulted_dist_attrs[1]

        self.assertEqual(
            inferred_input_dist_attrs[0].dims_mapping, [-1, -1, -1, -1]
        )
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [0, -1, -1])
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [-1, -0, -1, -1]
        )

    def test_backward_multi_mesh_dim(self):
        process_mesh = auto.ProcessMesh([[0, 1, 2], [3, 4, 5]])
        self.x_dist_tensor_spec.set_process_mesh(process_mesh)
        self.y_dist_tensor_spec.set_process_mesh(process_mesh)
        self.x_dist_tensor_spec.shape = [96, 24, 48]
        self.y_dist_tensor_spec.shape = [96, 24, 48]
        self.out_dist_tensor_spec.shape = [96, 24, 48]

        # [0, 1, -1] --> [0, 1, -1], [0, 1, -1], [0, 1, -1] (output --> inputs, output)
        self.out_dist_tensor_spec.set_dims_mapping([0, 1, -1])

        resulted_dist_attrs = self.binary_rule.infer_backward(
            self.x_dist_tensor_spec,
            self.y_dist_tensor_spec,
            self.out_dist_tensor_spec,
        )
        inferred_input_dist_attrs = resulted_dist_attrs[0]
        inferred_output_dist_attrs = resulted_dist_attrs[1]

        self.assertEqual(len(resulted_dist_attrs), 2)
        self.assertEqual(len(inferred_input_dist_attrs), 2)
        self.assertEqual(len(inferred_output_dist_attrs), 1)

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [0, 1, -1])
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [0, 1, -1])
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [0, 1, -1])

    def test_backward_multi_mesh_dim_broadcast(self):
        process_mesh = auto.ProcessMesh([[0, 1, 2], [3, 4, 5]])
        self.x_dist_tensor_spec.set_process_mesh(process_mesh)
        self.y_dist_tensor_spec.set_process_mesh(process_mesh)
        self.x_dist_tensor_spec.shape = [96, 24, 48]
        self.y_dist_tensor_spec.shape = [48]
        self.out_dist_tensor_spec.shape = [96, 24, 48]

        # [0, -1, 1] --> [0, -1, 1], [1], [0, -1, 1] (output --> inputs, output)
        self.out_dist_tensor_spec.set_dims_mapping([0, -1, 1])

        resulted_dist_attrs = self.binary_rule.infer_backward(
            self.x_dist_tensor_spec,
            self.y_dist_tensor_spec,
            self.out_dist_tensor_spec,
        )
        inferred_input_dist_attrs = resulted_dist_attrs[0]
        inferred_output_dist_attrs = resulted_dist_attrs[1]

        self.assertEqual(len(resulted_dist_attrs), 2)
        self.assertEqual(len(inferred_input_dist_attrs), 2)
        self.assertEqual(len(inferred_output_dist_attrs), 1)

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [0, -1, 1])
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [1])
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [0, -1, 1])

        # [0, 1, -1] --> [0, 1, -1], [-1], [0, 1, -1] (output --> inputs, output)
        self.out_dist_tensor_spec.set_dims_mapping([0, 1, -1])

        resulted_dist_attrs = self.binary_rule.infer_backward(
            self.x_dist_tensor_spec,
            self.y_dist_tensor_spec,
            self.out_dist_tensor_spec,
        )
        inferred_input_dist_attrs = resulted_dist_attrs[0]
        inferred_output_dist_attrs = resulted_dist_attrs[1]

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [0, 1, -1])
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [-1])
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [0, 1, -1])

        self.x_dist_tensor_spec.shape = [96, 1, 1, 48]
        self.y_dist_tensor_spec.shape = [96, 24, 48]
        self.out_dist_tensor_spec.shape = [96, 96, 24, 48]

        # [-1, 0, -1, 1] --> [-1, -1, -1, 1], [0, -1, 1], [-1, 0, -1, 1] (output --> inputs, output)
        self.out_dist_tensor_spec.set_dims_mapping([-1, 0, -1, 1])

        resulted_dist_attrs = self.binary_rule.infer_backward(
            self.x_dist_tensor_spec,
            self.y_dist_tensor_spec,
            self.out_dist_tensor_spec,
        )
        inferred_input_dist_attrs = resulted_dist_attrs[0]
        inferred_output_dist_attrs = resulted_dist_attrs[1]

        self.assertEqual(
            inferred_input_dist_attrs[0].dims_mapping, [-1, -1, -1, 1]
        )
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [0, -1, 1])
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [-1, 0, -1, 1]
        )

    def test_single_mesh_dim_greater_than(self):
        binary_rule = core.get_phi_spmd_rule("greater_than")
        # [0, -1], [-1, -1] --> [0, -1], [0, -1], [0, -1]
        self.x_dist_tensor_spec.set_dims_mapping([0, -1])
        self.y_dist_tensor_spec.set_dims_mapping([-1, -1])
        result_dist_attrs = binary_rule.infer_forward(
            self.x_dist_tensor_spec, self.y_dist_tensor_spec
        )
        self.assertEqual(len(result_dist_attrs), 2)

        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(inferred_input_dist_attrs), 2)
        self.assertEqual(len(inferred_output_dist_attrs), 1)

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [0, -1])
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [0, -1])
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [0, -1])

        # [0, -1], [-1, 0] --> [0, -1], [0, -1], [0, -1]
        self.x_dist_tensor_spec.set_dims_mapping([0, -1])
        self.y_dist_tensor_spec.set_dims_mapping([-1, 0])
        result_dist_attrs = binary_rule.infer_forward(
            self.x_dist_tensor_spec, self.y_dist_tensor_spec
        )
        self.assertEqual(len(result_dist_attrs), 2)

        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(inferred_input_dist_attrs), 2)
        self.assertEqual(len(inferred_output_dist_attrs), 1)

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [0, -1])
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [0, -1])
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [0, -1])

    def test_single_mesh_dim_broadcast_greater_than(self):
        binary_rule = core.get_phi_spmd_rule("greater_than")
        self.x_dist_tensor_spec.shape = [1, 64]
        self.y_dist_tensor_spec.shape = [64]

        # [-1, 0], [-1] --> [-1, 0], [0], [-1, 0]
        self.x_dist_tensor_spec.set_dims_mapping([-1, 0])
        self.y_dist_tensor_spec.set_dims_mapping([-1])

        resulted_dist_attrs = binary_rule.infer_forward(
            self.x_dist_tensor_spec, self.y_dist_tensor_spec
        )

        self.assertEqual(len(resulted_dist_attrs), 2)

        inferred_input_dist_attrs = resulted_dist_attrs[0]
        inferred_output_dist_attrs = resulted_dist_attrs[1]

        self.assertEqual(len(inferred_input_dist_attrs), 2)
        self.assertEqual(len(inferred_output_dist_attrs), 1)

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [-1, 0])
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [0])
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [-1, 0])

    def test_single_mesh_dim_less_than(self):
        binary_rule = core.get_phi_spmd_rule("less_than")
        # [0, -1], [-1, -1] --> [0, -1], [0, -1], [0, -1]
        self.x_dist_tensor_spec.set_dims_mapping([0, -1])
        self.y_dist_tensor_spec.set_dims_mapping([-1, -1])
        result_dist_attrs = binary_rule.infer_forward(
            self.x_dist_tensor_spec, self.y_dist_tensor_spec
        )
        self.assertEqual(len(result_dist_attrs), 2)

        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(inferred_input_dist_attrs), 2)
        self.assertEqual(len(inferred_output_dist_attrs), 1)

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [0, -1])
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [0, -1])
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [0, -1])

        # [0, -1], [-1, 0] --> [0, -1], [0, -1], [0, -1]
        self.x_dist_tensor_spec.set_dims_mapping([0, -1])
        self.y_dist_tensor_spec.set_dims_mapping([-1, 0])
        result_dist_attrs = binary_rule.infer_forward(
            self.x_dist_tensor_spec, self.y_dist_tensor_spec
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [0, -1])
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [0, -1])
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [0, -1])

    def test_single_mesh_dim_broadcast_less_than(self):
        binary_rule = core.get_phi_spmd_rule("less_than")
        self.x_dist_tensor_spec.shape = [1, 64]
        self.y_dist_tensor_spec.shape = [64]

        # [-1, 0], [-1] --> [-1, 0], [0], [-1, 0]
        self.x_dist_tensor_spec.set_dims_mapping([-1, 0])
        self.y_dist_tensor_spec.set_dims_mapping([-1])

        resulted_dist_attrs = binary_rule.infer_forward(
            self.x_dist_tensor_spec, self.y_dist_tensor_spec
        )

        self.assertEqual(len(resulted_dist_attrs), 2)

        inferred_input_dist_attrs = resulted_dist_attrs[0]
        inferred_output_dist_attrs = resulted_dist_attrs[1]

        self.assertEqual(len(inferred_input_dist_attrs), 2)
        self.assertEqual(len(inferred_output_dist_attrs), 1)

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [-1, 0])
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [0])
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [-1, 0])


if __name__ == "__main__":
    unittest.main()
