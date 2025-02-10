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


class TestUnbindSPMDRule(unittest.TestCase):
    """
    Unit tests for split spmd rule.
    """

    def setUp(self):
        x_shape = [64, 2, 48]
        process_mesh = auto.ProcessMesh(mesh=[[0, 1, 2], [3, 4, 5]])
        self.rule = core.get_phi_spmd_rule("unbind")

        x_dist_attr = TensorDistAttr()
        x_dist_attr.dims_mapping = [-1, -1, -1]
        x_dist_attr.process_mesh = process_mesh
        self.x_spec = DistTensorSpec(x_shape, x_dist_attr)
        self.attrs = OrderedDict()
        self.attrs["axis"] = 1

    def test_infer_spmd(self):
        # axis = 1
        # [-1, -1, -1] --> [-1, -1, -1], [-1, -1], [-1, -1]
        self.attrs['axis'] = 1
        self.x_spec.set_dims_mapping([-1, -1, -1])
        result_dist_attrs = self.rule.infer_forward(
            self.x_spec, self.attrs['axis']
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(inferred_input_dist_attrs), 1)
        self.assertEqual(len(inferred_output_dist_attrs), 2)

        self.assertEqual(
            inferred_input_dist_attrs[0].dims_mapping, [-1, -1, -1]
        )
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [-1, -1])
        self.assertEqual(inferred_output_dist_attrs[1].dims_mapping, [-1, -1])

        # axis = 1
        # [0, -1, 1] --> [0, -1, 1], [0, 1], [0, 1]
        self.attrs['axis'] = 1
        self.x_spec.set_dims_mapping([0, -1, 1])
        result_dist_attrs = self.rule.infer_forward(
            self.x_spec, self.attrs['axis']
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(inferred_input_dist_attrs), 1)
        self.assertEqual(len(inferred_output_dist_attrs), 2)

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [0, -1, 1])
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [0, 1])
        self.assertEqual(inferred_output_dist_attrs[1].dims_mapping, [0, 1])

        # axis = -2
        # [0, 1, -1] --> [0, -1, -1], [0, -1], [0, -1]
        self.attrs['axis'] = -2
        self.x_spec.set_dims_mapping([0, 1, -1])
        result_dist_attrs = self.rule.infer_forward(
            self.x_spec, self.attrs['axis']
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(inferred_input_dist_attrs), 1)
        self.assertEqual(len(inferred_output_dist_attrs), 2)

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [0, -1, -1])
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [0, -1])
        self.assertEqual(inferred_output_dist_attrs[1].dims_mapping, [0, -1])

        # axis = 3
        # raise error
        self.attrs['axis'] = 3
        with self.assertRaises(ValueError):
            result_dist_attrs = self.rule.infer_forward(
                self.x_spec, self.attrs['axis']
            )

    def test_infer_spmd_reverse(self):
        self.out_spec0 = DistTensorSpec(self.x_spec)
        self.out_spec1 = DistTensorSpec(self.x_spec)
        self.out_spec0.shape = [64, 48]
        self.out_spec1.shape = [64, 48]

        # axis = 1
        # [0, 1], [0, -1] --> [0, -1, 1], [0, 1], [0, 1]
        # (outputs --> input, outputs)
        self.attrs['axis'] = 1
        self.out_spec_list = []
        self.out_spec0.set_dims_mapping([0, 1])
        self.out_spec1.set_dims_mapping([0, -1])
        result_dist_attrs = self.rule.infer_backward(
            self.x_spec,
            [self.out_spec0, self.out_spec1],
            self.attrs['axis'],
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(inferred_input_dist_attrs), 1)
        self.assertEqual(len(inferred_output_dist_attrs), 2)

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [0, -1, 1])
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [0, 1])
        self.assertEqual(inferred_output_dist_attrs[1].dims_mapping, [0, 1])

        # axis = -2
        # [0, -1], [-1, 1] --> [0, -1, 1], [0, 1], [0, 1]
        self.attrs['axis'] = -2
        self.out_spec0.set_dims_mapping([0, -1])
        self.out_spec1.set_dims_mapping([-1, 1])
        result_dist_attrs = self.rule.infer_backward(
            self.x_spec,
            [self.out_spec0, self.out_spec1],
            self.attrs['axis'],
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(inferred_input_dist_attrs), 1)
        self.assertEqual(len(inferred_output_dist_attrs), 2)

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [0, -1, 1])
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [0, 1])
        self.assertEqual(inferred_output_dist_attrs[1].dims_mapping, [0, 1])

        # axis = 1
        # [0, 1], [1, 0] --> raise error
        self.attrs['axis'] = 1
        self.out_spec0.set_dims_mapping([0, 1])
        self.out_spec1.set_dims_mapping([1, 0])
        with self.assertRaises(NotImplementedError):
            result_dist_attrs = self.rule.infer_backward(
                self.x_spec,
                [self.out_spec0, self.out_spec1],
                self.attrs['axis'],
            )


if __name__ == "__main__":
    unittest.main()
