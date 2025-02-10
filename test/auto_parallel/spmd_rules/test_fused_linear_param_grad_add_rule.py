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


class TestFusedLinearParamGradAddSPMDRule(unittest.TestCase):
    """
    Unit tests for split spmd rule.
    """

    def setUp(self):
        self.process_mesh = auto.ProcessMesh(mesh=[[0, 1], [2, 3]])

    def build_inputs(self, dims_mapping, shape):
        tensor_dist_attr = TensorDistAttr()
        tensor_dist_attr.dims_mapping = dims_mapping
        tensor_dist_attr.process_mesh = self.process_mesh
        return DistTensorSpec(shape, tensor_dist_attr)

    def test_infer_forward(self):
        rule = core.get_phi_spmd_rule("fused_linear_param_grad_add")

        # test mp split by col
        input = self.build_inputs([0, -1, -1], [2, 512, 1024])
        out_grad = self.build_inputs([0, -1, 1], [2, 512, 2048])
        dweight = self.build_inputs([], [])
        dbias = self.build_inputs([], [])
        inferred_dist_attrs = rule.infer_forward(
            input, out_grad, dweight, dbias, 0, True
        )
        self.assertEqual(inferred_dist_attrs[1][0].dims_mapping, [-1, 1])
        self.assertEqual(inferred_dist_attrs[1][1].dims_mapping, [1])

        # test mp split by row
        input = self.build_inputs([0, -1, 1], [2, 512, 1024])
        out_grad = self.build_inputs([0, -1, -1], [2, 512, 2048])
        dweight = self.build_inputs([], [])
        dbias = self.build_inputs([], [])
        inferred_dist_attrs = rule.infer_forward(
            input, out_grad, dweight, dbias, 0, True
        )
        self.assertEqual(inferred_dist_attrs[1][0].dims_mapping, [1, -1])
        self.assertEqual(inferred_dist_attrs[1][1].dims_mapping, [-1])


if __name__ == "__main__":
    unittest.main()
