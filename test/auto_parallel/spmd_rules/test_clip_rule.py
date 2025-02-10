# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
        self.clip_rule = core.get_phi_spmd_rule("clip")

        x_shape = [64, 36]
        process_mesh = auto.ProcessMesh(mesh=[0, 1, 2, 3])

        x_tensor_dist_attr = TensorDistAttr()
        x_tensor_dist_attr.dims_mapping = [1, 0]
        x_tensor_dist_attr.process_mesh = process_mesh
        self.x_dist_tensor_spec = DistTensorSpec(x_shape, x_tensor_dist_attr)

        self.out_dist_tensor_spec = DistTensorSpec(self.x_dist_tensor_spec)

        self.attrs = []

    def test_single_mesh_dim(self):
        # [-1, 0]--> [-1, 0], [-1, 0]
        self.x_dist_tensor_spec.set_dims_mapping([-1, 0])

        result_dist_attrs = self.clip_rule.infer_forward(
            self.x_dist_tensor_spec
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [-1, 0])
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [-1, 0])

    def test_backward_single_mesh_dim(self):
        # [-1, 0]--> [-1, 0], [-1, 0] (output --> inputs, output)
        self.out_dist_tensor_spec.set_dims_mapping([-1, 0])

        result_dist_attrs = self.clip_rule.infer_backward(
            self.x_dist_tensor_spec, self.out_dist_tensor_spec
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [-1, 0])
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [-1, 0])


if __name__ == "__main__":
    unittest.main()
