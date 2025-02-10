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


class TestOneHotSPMDRule(unittest.TestCase):
    def setUp(self):
        self.rule = core.get_phi_spmd_rule("one_hot")
        self.process_mesh = auto.ProcessMesh(mesh=[[0, 1, 2, 3], [4, 5, 6, 7]])
        self.x_shape = [4, 1024]  # [B,S]
        self.attrs = OrderedDict([('num_classes', 30000)])
        self.attrs['num_classes'] = 30000
        self.out_shape = [4, 1024, self.attrs['num_classes']]

        self.x_dist_attr = TensorDistAttr()
        self.x_dist_attr.process_mesh = self.process_mesh
        self.x_spec = DistTensorSpec(self.x_shape, self.x_dist_attr)

    def test_one_hot_infer_spmd(self):
        # [0, 1] --> [0, 1], [0, 1, -1]
        self.x_spec.set_dims_mapping([0, 1])
        result_dist_attrs = self.rule.infer_forward(
            self.x_spec,
            self.attrs['num_classes'],
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(inferred_input_dist_attrs), 1)
        self.assertEqual(len(inferred_output_dist_attrs), 1)

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [0, 1])
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [0, 1, -1])

    def test_one_hot_infer_spmd_reverse(self):
        out_dist_attr = TensorDistAttr()
        out_dist_attr.process_mesh = self.process_mesh
        self.out_spec = DistTensorSpec(self.out_shape, out_dist_attr)

        # [0, 1], [0, 1, -1] --> [0, 1], [0, 1, -1]
        self.x_spec.set_dims_mapping([0, 1])
        self.out_spec.set_dims_mapping([0, 1, -1])
        result_dist_attrs = self.rule.infer_backward(
            self.x_spec,
            self.out_spec,
            self.attrs['num_classes'],
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(inferred_input_dist_attrs), 1)
        self.assertEqual(len(inferred_output_dist_attrs), 1)

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [0, 1])
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [0, 1, -1])

        # [-1, -1], [0, -1, 1] --> [0, -1], [0, -1, -1]
        self.x_spec.set_dims_mapping([-1, -1])
        self.out_spec.set_dims_mapping([0, -1, 1])
        result_dist_attrs = self.rule.infer_backward(
            self.x_spec,
            self.out_spec,
            self.attrs['num_classes'],
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(inferred_input_dist_attrs), 1)
        self.assertEqual(len(inferred_output_dist_attrs), 1)

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [0, -1])
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [0, -1, -1]
        )


if __name__ == "__main__":
    unittest.main()
