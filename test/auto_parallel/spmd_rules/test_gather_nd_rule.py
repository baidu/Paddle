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


class TestGatherNdSPMDRule(unittest.TestCase):
    """
    Unit tests for gather_nd spmd rule.
    """

    def setUp(self):
        x_shape = [10, 20]
        index_shape = [2, 4, 1]
        self.process_mesh = auto.ProcessMesh(mesh=[[0, 1, 2, 3], [4, 5, 6, 7]])
        self.rule = core.get_phi_spmd_rule("gather_nd")

        x_dist_attr = TensorDistAttr()
        x_dist_attr.dims_mapping = [-1, -1]
        x_dist_attr.process_mesh = self.process_mesh
        self.x_spec = DistTensorSpec(x_shape, x_dist_attr)

        index_dist_attr = TensorDistAttr()
        index_dist_attr.dims_mapping = [0, -1, -1]
        index_dist_attr.process_mesh = self.process_mesh
        self.index_spec = DistTensorSpec(index_shape, index_dist_attr)

    def test_forward_mesh_dim(self):
        # dims_mapping: [-1, -1], [0, -1, -1] --> [0, -1, -1]
        self.x_spec.set_dims_mapping([-1, -1])
        self.index_spec.set_dims_mapping([0, -1, -1])

        result_dist_attrs = self.rule.infer_forward(
            self.x_spec, self.index_spec
        )

        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [-1, -1])
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [0, -1, -1])
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [0, -1, -1]
        )

    def test_forward_mesh_dim_diff_shape(self):
        # dims_mapping: [-1], [0, -1] --> [0]
        x_shape = [10]
        index_shape = [8, 1]
        x_dist_attr = TensorDistAttr()
        x_dist_attr.dims_mapping = [-1]
        x_dist_attr.process_mesh = self.process_mesh
        x_spec = DistTensorSpec(x_shape, x_dist_attr)

        index_dist_attr = TensorDistAttr()
        index_dist_attr.dims_mapping = [0, -1]
        index_dist_attr.process_mesh = self.process_mesh
        index_spec = DistTensorSpec(index_shape, index_dist_attr)

        result_dist_attrs = self.rule.infer_forward(x_spec, index_spec)

        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [-1])
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [0, -1])
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [0])

    def test_forward_mesh_dim_same_shape(self):
        # dims_mapping: [-1], [0] --> [0]
        x_shape = [10]
        index_shape = [4]
        x_dist_attr = TensorDistAttr()
        x_dist_attr.dims_mapping = [-1]
        x_dist_attr.process_mesh = self.process_mesh
        x_spec = DistTensorSpec(x_shape, x_dist_attr)

        index_dist_attr = TensorDistAttr()
        index_dist_attr.dims_mapping = [0]
        index_dist_attr.process_mesh = self.process_mesh
        index_spec = DistTensorSpec(index_shape, index_dist_attr)

        result_dist_attrs = self.rule.infer_forward(x_spec, index_spec)

        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [-1])
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [0])
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [0])

    def test_reverse_mesh_dim(self):
        self.x_spec.set_process_mesh(self.process_mesh)
        self.index_spec.set_process_mesh(self.process_mesh)

        out_shape = [2, 4, 20]
        out_dist_attr = TensorDistAttr()
        out_dist_attr.dims_mapping = [0, -1, -1]
        out_dist_attr.process_mesh = self.process_mesh
        self.out_spec = DistTensorSpec(out_shape, out_dist_attr)

        # axis = 1
        # [0, -1, -1] --> [-1, -1], [0, -1, -1]
        result_dist_attrs = self.rule.infer_backward(
            self.x_spec,
            self.index_spec,
            self.out_spec,
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [-1, -1])
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [0, -1, -1])
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [0, -1, -1]
        )

    def test_reverse_mesh_dim_same_shape(self):
        x_shape = [10]
        index_shape = [4]
        x_dist_attr = TensorDistAttr()
        x_dist_attr.dims_mapping = [-1]
        x_dist_attr.process_mesh = self.process_mesh
        x_spec = DistTensorSpec(x_shape, x_dist_attr)

        index_dist_attr = TensorDistAttr()
        index_dist_attr.dims_mapping = [0]
        index_dist_attr.process_mesh = self.process_mesh
        index_spec = DistTensorSpec(index_shape, index_dist_attr)
        x_spec.set_process_mesh(self.process_mesh)
        index_spec.set_process_mesh(self.process_mesh)

        out_shape = [4]
        out_dist_attr = TensorDistAttr()
        out_dist_attr.dims_mapping = [0]
        out_dist_attr.process_mesh = self.process_mesh

        out_spec = DistTensorSpec(out_shape, out_dist_attr)

        # [0] --> [-1], [0]
        result_dist_attrs = self.rule.infer_backward(
            x_spec,
            index_spec,
            out_spec,
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [-1])
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [0])
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [0])

    def test_reverse_mesh_dim_diff_shape(self):
        # dims_mapping: [-1], [0, -1] --> [0]
        x_shape = [10]
        index_shape = [8, 1]
        x_dist_attr = TensorDistAttr()
        x_dist_attr.dims_mapping = [-1]
        x_dist_attr.process_mesh = self.process_mesh
        x_spec = DistTensorSpec(x_shape, x_dist_attr)

        index_dist_attr = TensorDistAttr()
        index_dist_attr.dims_mapping = [0, -1]
        index_dist_attr.process_mesh = self.process_mesh
        index_spec = DistTensorSpec(index_shape, index_dist_attr)
        x_spec.set_process_mesh(self.process_mesh)
        index_spec.set_process_mesh(self.process_mesh)

        out_shape = [8]
        out_dist_attr = TensorDistAttr()
        out_dist_attr.dims_mapping = [0]
        out_dist_attr.process_mesh = self.process_mesh

        out_spec = DistTensorSpec(out_shape, out_dist_attr)

        # [0] --> [-1], [0, -1]
        result_dist_attrs = self.rule.infer_backward(
            x_spec,
            index_spec,
            out_spec,
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [-1])
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [0, -1])
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [0])


if __name__ == "__main__":
    unittest.main()
