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


class TestSoftmaxSPMDRule(unittest.TestCase):
    def setUp(self):
        self.rule1 = core.get_phi_spmd_rule("softmax")
        self.rule2 = core.get_phi_spmd_rule("log_softmax")

        x_shape = [8, 16, 48]
        process_mesh = auto.ProcessMesh(mesh=[[0, 1, 2, 3], [4, 5, 6, 7]])
        x_tensor_dist_attr = TensorDistAttr()
        x_tensor_dist_attr.process_mesh = process_mesh
        self.x_dist_tensor_spec = DistTensorSpec(x_shape, x_tensor_dist_attr)

        self.out_dist_tensor_spec = DistTensorSpec(self.x_dist_tensor_spec)

        self.attrs = OrderedDict([('axis', -1)])

    def test_softmax_infer_forward(self):
        # sharding on batch axis I
        self.x_dist_tensor_spec.set_dims_mapping([1, -1, -1])
        result_dist_attrs = self.rule1.infer_forward(
            self.x_dist_tensor_spec, self.attrs['axis']
        )
        self.assertEqual(len(result_dist_attrs), 2)
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(inferred_input_dist_attrs), 1)
        self.assertEqual(len(inferred_output_dist_attrs), 1)

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [1, -1, -1])
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [1, -1, -1]
        )

        # sharding on batch axis II
        self.x_dist_tensor_spec.set_dims_mapping([-1, 1, -1])
        result_dist_attrs = self.rule1.infer_forward(
            self.x_dist_tensor_spec, self.attrs['axis']
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [-1, 1, -1])
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [-1, 1, -1]
        )

        # sharding on softmax_axis
        self.x_dist_tensor_spec.set_dims_mapping([1, -1, 0])
        result_dist_attrs = self.rule1.infer_forward(
            self.x_dist_tensor_spec, self.attrs['axis']
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [1, -1, -1])
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [1, -1, -1]
        )

        # sharding on softmax_axis + axis = 1
        self.attrs = {
            'axis': 1,
        }
        self.x_dist_tensor_spec.set_dims_mapping([-1, 1, 0])
        result_dist_attrs = self.rule1.infer_forward(
            self.x_dist_tensor_spec, self.attrs['axis']
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [-1, -1, 0])
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [-1, -1, 0]
        )

        # sharding on softmax_axis + axis = -2
        self.attrs = {
            'axis': -2,
        }
        self.x_dist_tensor_spec.set_dims_mapping([-1, 1, 0])
        result_dist_attrs = self.rule1.infer_forward(
            self.x_dist_tensor_spec, self.attrs['axis']
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [-1, -1, 0])
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [-1, -1, 0]
        )

    def test_softmax_infer_backward(self):
        # sharding on batch axis I
        self.out_dist_tensor_spec.set_dims_mapping([1, -1, -1])
        result_dist_attrs = self.rule1.infer_backward(
            self.x_dist_tensor_spec,
            self.out_dist_tensor_spec,
            self.attrs['axis'],
        )
        self.assertEqual(len(result_dist_attrs), 2)
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(inferred_input_dist_attrs), 1)
        self.assertEqual(len(inferred_output_dist_attrs), 1)

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [1, -1, -1])
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [1, -1, -1]
        )

        # sharding on batch axis II
        self.out_dist_tensor_spec.set_dims_mapping([-1, 1, -1])
        result_dist_attrs = self.rule1.infer_backward(
            self.x_dist_tensor_spec,
            self.out_dist_tensor_spec,
            self.attrs['axis'],
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [-1, 1, -1])
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [-1, 1, -1]
        )

        # sharding on softmax_axis
        self.out_dist_tensor_spec.set_dims_mapping([1, -1, 0])
        result_dist_attrs = self.rule1.infer_backward(
            self.x_dist_tensor_spec,
            self.out_dist_tensor_spec,
            self.attrs['axis'],
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [1, -1, -1])
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [1, -1, -1]
        )

        # sharding on softmax_axis + axis = 1
        self.attrs = {
            'axis': 1,
        }
        self.out_dist_tensor_spec.set_dims_mapping([-1, 1, 0])
        result_dist_attrs = self.rule1.infer_backward(
            self.x_dist_tensor_spec,
            self.out_dist_tensor_spec,
            self.attrs['axis'],
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [-1, -1, 0])
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [-1, -1, 0]
        )

        # sharding on softmax_axis + axis = -2
        self.attrs = {
            'axis': -2,
        }
        self.out_dist_tensor_spec.set_dims_mapping([-1, 1, 0])
        result_dist_attrs = self.rule1.infer_backward(
            self.x_dist_tensor_spec,
            self.out_dist_tensor_spec,
            self.attrs['axis'],
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [-1, -1, 0])
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [-1, -1, 0]
        )


if __name__ == "__main__":
    unittest.main()
