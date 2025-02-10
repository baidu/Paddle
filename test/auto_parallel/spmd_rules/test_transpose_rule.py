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


class TestTransposeSPMDRule(unittest.TestCase):
    """
    Unit tests for reduction spmd rule.
    """

    def setUp(self):
        self.rule = core.get_phi_spmd_rule("transpose")

        x_shape = [64, 36]
        process_mesh = auto.ProcessMesh(mesh=[0, 1, 2, 3])

        x_tensor_dist_attr = TensorDistAttr()
        x_tensor_dist_attr.dims_mapping = [1, 0]
        x_tensor_dist_attr.process_mesh = process_mesh
        self.x_dist_tensor_spec = DistTensorSpec(x_shape, x_tensor_dist_attr)

        self.out_dist_tensor_spec = DistTensorSpec(self.x_dist_tensor_spec)

        self.attrs = OrderedDict([('perm', [0, 1, 2, 3])])

    def test_single_mesh_dim(self):
        # perm = [1, 0]
        # [0, -1] --> [0, -1], [-1, 0]
        self.attrs['perm'] = [1, 0]
        self.x_dist_tensor_spec.set_dims_mapping([0, -1])
        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec, self.attrs['perm']
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(inferred_input_dist_attrs), 1)
        self.assertEqual(len(inferred_output_dist_attrs), 1)
        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [0, -1])
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [-1, 0])

        # perm = [0, 1]
        # [0, -1] --> [0, -1], [0, -1]
        self.attrs['perm'] = [0, 1]
        self.x_dist_tensor_spec.set_dims_mapping([0, -1])
        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec, self.attrs['perm']
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [0, -1])
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [0, -1])

        # perm = [0, 2, 3, 1]
        # [-1, -1, 0, -1] --> [-1, -1, 0, -1], [-1, 0, -1, -1]
        self.x_dist_tensor_spec.shape = [64, 48, 36, 24]
        self.attrs['perm'] = [0, 2, 3, 1]
        self.x_dist_tensor_spec.set_dims_mapping([-1, -1, 0, -1])
        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec, self.attrs['perm']
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(
            inferred_input_dist_attrs[0].dims_mapping, [-1, -1, 0, -1]
        )
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [-1, 0, -1, -1]
        )

    def test_multi_mesh_dim(self):
        process_mesh = auto.ProcessMesh(mesh=[[0, 1, 2], [3, 4, 5]])
        self.x_dist_tensor_spec.set_process_mesh(process_mesh)
        self.x_dist_tensor_spec.shape = [64, 48, 36, 24]

        # perm = [0, 2, 3, 1]
        # [-1, 0, 1, -1] --> [-1, 0, 1, -1], [-1, 1, -1, 0]
        self.attrs['perm'] = [0, 2, 3, 1]
        self.x_dist_tensor_spec.set_dims_mapping([-1, 0, 1, -1])
        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec, self.attrs['perm']
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(inferred_input_dist_attrs), 1)
        self.assertEqual(len(inferred_output_dist_attrs), 1)
        self.assertEqual(
            inferred_input_dist_attrs[0].dims_mapping, [-1, 0, 1, -1]
        )
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [-1, 1, -1, 0]
        )

        # perm = [0, 2, 3, 1]
        # [-1, -1, -1, -1] --> [-1, -1, -1, -1], [-1, -1, -1, -1]
        self.attrs['perm'] = [0, 2, 3, 1]
        self.x_dist_tensor_spec.set_dims_mapping([-1, -1, -1, -1])
        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec, self.attrs['perm']
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(
            inferred_input_dist_attrs[0].dims_mapping, [-1, -1, -1, -1]
        )
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [-1, -1, -1, -1]
        )

        # perm = [-1, 0, -2, 1]
        # [-1, -1, 0, 1] --> [-1, -1, 0, 1], [1, -1, 0, -1]
        self.attrs['perm'] = [-1, 0, -2, 1]
        self.x_dist_tensor_spec.set_dims_mapping([-1, -1, 0, 1])
        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec, self.attrs['perm']
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(
            inferred_input_dist_attrs[0].dims_mapping, [-1, -1, 0, 1]
        )
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [1, -1, 0, -1]
        )

    def test_backward_single_mesh_dim(self):
        # perm = [1, 0]
        # [-1, 0] --> [0, -1], [-1, 0] (output --> input, output)
        self.attrs['perm'] = [1, 0]
        self.out_dist_tensor_spec.shape = [36, 64]
        self.out_dist_tensor_spec.set_dims_mapping([-1, 0])
        result_dist_attrs = self.rule.infer_backward(
            self.x_dist_tensor_spec,
            self.out_dist_tensor_spec,
            self.attrs['perm'],
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(inferred_input_dist_attrs), 1)
        self.assertEqual(len(inferred_output_dist_attrs), 1)
        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [0, -1])
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [-1, 0])

        # perm = [0, 1]
        # [0, -1] --> [0, -1], [0, -1] (output --> input, output)
        self.attrs['perm'] = [0, 1]
        self.out_dist_tensor_spec.shape = [64, 36]
        self.out_dist_tensor_spec.set_dims_mapping([0, -1])
        result_dist_attrs = self.rule.infer_backward(
            self.x_dist_tensor_spec,
            self.out_dist_tensor_spec,
            self.attrs['perm'],
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [0, -1])
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [0, -1])

        # perm = [0, 2, 3, 1]
        # [-1, 0, -1, -1] --> [-1, -1, 0, -1], [-1, 0, -1, -1] (output --> input, output)
        self.x_dist_tensor_spec.shape = [64, 48, 36, 24]
        self.attrs['perm'] = [0, 2, 3, 1]
        self.out_dist_tensor_spec.shape = [64, 36, 24, 48]

        self.out_dist_tensor_spec.set_dims_mapping([-1, 0, -1, -1])
        result_dist_attrs = self.rule.infer_backward(
            self.x_dist_tensor_spec,
            self.out_dist_tensor_spec,
            self.attrs['perm'],
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(
            inferred_input_dist_attrs[0].dims_mapping, [-1, -1, 0, -1]
        )
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [-1, 0, -1, -1]
        )

    def test_backward_multi_mesh_dim(self):
        process_mesh = auto.ProcessMesh(mesh=[[0, 1, 2], [3, 4, 5]])
        self.x_dist_tensor_spec.set_process_mesh(process_mesh)
        self.x_dist_tensor_spec.shape = [64, 48, 36, 24]
        self.out_dist_tensor_spec.set_process_mesh(process_mesh)

        # perm = [0, 2, 3, 1]
        # [-1, 1, -1, 0] --> [-1, 0, 1, -1], [-1, 1, -1, 0] (output --> input, output)
        self.attrs['perm'] = [0, 2, 3, 1]
        self.out_dist_tensor_spec.shape = [64, 36, 24, 48]
        self.out_dist_tensor_spec.set_dims_mapping([-1, 1, -1, 0])
        result_dist_attrs = self.rule.infer_backward(
            self.x_dist_tensor_spec,
            self.out_dist_tensor_spec,
            self.attrs['perm'],
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(inferred_input_dist_attrs), 1)
        self.assertEqual(len(inferred_output_dist_attrs), 1)
        self.assertEqual(
            inferred_input_dist_attrs[0].dims_mapping, [-1, 0, 1, -1]
        )
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [-1, 1, -1, 0]
        )

        # perm = [0, 2, 3, 1]
        # [-1, -1, -1, -1] --> [-1, -1, -1, -1], [-1, -1, -1, -1] (output --> input, output)
        self.attrs['perm'] = [0, 2, 3, 1]
        self.out_dist_tensor_spec.set_dims_mapping([-1, -1, -1, -1])
        result_dist_attrs = self.rule.infer_backward(
            self.x_dist_tensor_spec,
            self.out_dist_tensor_spec,
            self.attrs['perm'],
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(
            inferred_input_dist_attrs[0].dims_mapping, [-1, -1, -1, -1]
        )
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [-1, -1, -1, -1]
        )

        # perm = [-1, 0, -2, 1]
        # [1, -1, 0, -1] --> [-1, -1, 0, 1], [1, -1, 0, -1] (output --> input, output)
        self.x_dist_tensor_spec.shape = [64, 48, 36, 24]
        self.attrs['perm'] = [-1, 0, -2, 1]
        self.out_dist_tensor_spec.shape = [24, 64, 36, 48]
        self.out_dist_tensor_spec.set_dims_mapping([1, -1, 0, -1])
        result_dist_attrs = self.rule.infer_backward(
            self.x_dist_tensor_spec,
            self.out_dist_tensor_spec,
            self.attrs['perm'],
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(
            inferred_input_dist_attrs[0].dims_mapping, [-1, -1, 0, 1]
        )
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [1, -1, 0, -1]
        )


if __name__ == "__main__":
    unittest.main()
