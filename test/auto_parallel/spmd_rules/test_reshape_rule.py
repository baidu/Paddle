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


class TestReshapeSPMDRule(unittest.TestCase):
    def setUp(self):
        self.rule = core.get_phi_spmd_rule("reshape")

        x_shape = [6, 12, 48, 24]
        process_mesh = auto.ProcessMesh(mesh=[[0, 1, 2], [3, 4, 5]])

        x_tensor_dist_attr = TensorDistAttr()
        x_tensor_dist_attr.dims_mapping = [-1, -1, -1, -1]
        x_tensor_dist_attr.process_mesh = process_mesh
        self.x_dist_tensor_spec = DistTensorSpec(x_shape, x_tensor_dist_attr)

        self.attrs = OrderedDict([('shape', [1, 72, 48, 4, 6])])

    def test_reshape_infer_forward(self):
        # shape: [6, 12, 48, 24] --> [1, 72, 48, 4, 6]
        # dims_mapping: [0, -1, 1, -1] --> [0, -1, 1, -1] [-1, 0, 1, -1, -1]
        self.x_dist_tensor_spec.set_dims_mapping([0, -1, 1, -1])
        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec, self.attrs['shape']
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(inferred_input_dist_attrs), 1)
        self.assertEqual(len(inferred_output_dist_attrs), 1)
        self.assertEqual(
            inferred_input_dist_attrs[0].dims_mapping, [0, -1, 1, -1]
        )
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [-1, 0, 1, -1, -1]
        )

        # shape: [6, 12, 48, 24] --> [1, 72, 48, 4, 6]
        # dims_mapping: [-1, 0, -1, 1] --> [-1, -1, -1, -1] [-1, -1, -1, -1, -1]
        self.x_dist_tensor_spec.set_dims_mapping([-1, 0, -1, 1])
        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec, self.attrs['shape']
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(
            inferred_input_dist_attrs[0].dims_mapping, [-1, -1, -1, -1]
        )
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [-1, -1, -1, -1, -1]
        )

        # shape: [6, 12, 48, 24] --> [1, 72, 48, 4, 6]
        # dims_mapping: [1, -1, -1, 0] --> [1, -1, -1, 0] [-1, 1, -1, 0, -1]
        self.x_dist_tensor_spec.set_dims_mapping([1, -1, -1, 0])
        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec, self.attrs['shape']
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(
            inferred_input_dist_attrs[0].dims_mapping, [1, -1, -1, 0]
        )
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [-1, 1, -1, 0, -1]
        )

        # shape: [6, 12, 48, 24] --> [3, 24, 6, 8, 24]
        # dims_mapping: [0, 1, -1, -1] --> [-1, -1, -1, -1] [-1, -1, -1, -1, -1]
        self.attrs["shape"] = [3, 24, 6, 8, 24]
        self.x_dist_tensor_spec.set_dims_mapping([0, 1, -1, -1])

        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec, self.attrs['shape']
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(
            inferred_input_dist_attrs[0].dims_mapping, [-1, -1, -1, -1]
        )
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [-1, -1, -1, -1, -1]
        )

        # shape: [6, 12, 48, 24] --> [3, 24, 6, 8, 24]
        # dims_mapping: [1, -1, -1, 0] --> [1, -1, -1, 0] [1, -1, -1, -1, 0]
        self.x_dist_tensor_spec.set_dims_mapping([1, -1, -1, 0])

        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec, self.attrs['shape']
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(
            inferred_input_dist_attrs[0].dims_mapping, [1, -1, -1, 0]
        )
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [1, -1, -1, -1, 0]
        )

        # shape: [6, 12, 48, 24] --> [3, 24, 6, -1, 24]
        # dims_mapping: [-1, -1, 0, 1] --> [-1, -1, 0, 1], [-1, -1, 0, -1, 1]
        self.attrs["shape"] = [3, 24, 6, -1, 24]
        self.x_dist_tensor_spec.set_dims_mapping([-1, -1, 0, 1])

        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec, self.attrs['shape']
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(
            inferred_input_dist_attrs[0].dims_mapping, [-1, -1, 0, 1]
        )
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [-1, -1, 0, -1, 1]
        )

        # shape: [6, 12, 48, 24] --> [1, 72, 0, 4, 6]
        # dims_mapping: [1, -1, -1, 0] --> [1, -1, -1, 0] [-1, 1, -1, 0, -1]
        self.attrs["shape"] = [1, 72, 0, 4, 6]
        self.x_dist_tensor_spec.set_dims_mapping([1, -1, -1, 0])
        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec, self.attrs['shape']
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(
            inferred_input_dist_attrs[0].dims_mapping, [1, -1, -1, 0]
        )
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [-1, 1, -1, 0, -1]
        )

        # shape: [6, 12, 48, 24] --> [6, 12, 48, 24]
        # dims_mapping: [-1, -1, 0, 1] --> [-1, -1, 0, 1], [-1, -1, 0, 1]
        self.attrs["shape"] = [6, 12, 48, 24]
        self.x_dist_tensor_spec.set_dims_mapping([-1, -1, 0, 1])
        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec, self.attrs['shape']
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(
            inferred_input_dist_attrs[0].dims_mapping, [-1, -1, 0, 1]
        )
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [-1, -1, 0, 1]
        )

        # shape: [6, 12, 48, 24] --> [72, 3, 16, 24]
        # dims_mapping: [0, -1, 1, -1] --> [0, -1, 1, -1], [0, 1, -1, -1]
        self.attrs["shape"] = [72, 3, 16, 24]
        self.x_dist_tensor_spec.set_dims_mapping([0, -1, 1, -1])
        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec, self.attrs['shape']
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(
            inferred_input_dist_attrs[0].dims_mapping, [0, -1, 1, -1]
        )
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [0, 1, -1, -1]
        )

        # shape: [6, 12, 48, 24] --> [72, 3, 16, 24]
        # dims_mapping: [1, -1, 0, -1] --> [1, -1, -1, -1], [1, -1, -1, -1]
        self.attrs["shape"] = [72, 3, 16, 24]
        self.x_dist_tensor_spec.set_dims_mapping([1, -1, 0, -1])
        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec, self.attrs['shape']
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(
            inferred_input_dist_attrs[0].dims_mapping, [1, -1, -1, -1]
        )
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [1, -1, -1, -1]
        )

        # shape: [1, 72, 48, 4, 6] --> [6, 12, 48, 24]
        # dims_mapping: [-1, 1, -1, 0, -1] --> [-1, 1, -1, 0, -1] [1, -1, -1, 0]
        self.x_dist_tensor_spec.shape = [1, 72, 48, 4, 6]
        self.attrs["shape"] = [6, 12, 48, 24]
        self.x_dist_tensor_spec.set_dims_mapping([-1, 1, -1, 0, -1])
        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec, self.attrs['shape']
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(
            inferred_input_dist_attrs[0].dims_mapping, [-1, 1, -1, 0, -1]
        )
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [1, -1, -1, 0]
        )

        # shape: [8, 1024, 3072] --> [0, 0, -1, 192]
        # dims_mapping: [0, 1, -1] --> [0, 1, -1], [0, 1, -1, -1]
        self.x_dist_tensor_spec.shape = [8, 1024, 3072]
        self.attrs["shape"] = [0, 0, -1, 192]
        self.x_dist_tensor_spec.set_dims_mapping([0, 1, -1])
        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec, self.attrs['shape']
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [0, 1, -1])
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [0, 1, -1, -1]
        )

        # shape: [-1, -1, 3072] --> [0, 0, -1, 192]
        # dims_mapping: [0, 1, -1] --> [0, 1, -1], [0, 1, -1, -1]
        self.x_dist_tensor_spec.shape = [-1, -1, 3072]
        self.attrs["shape"] = [0, 0, -1, 192]
        self.x_dist_tensor_spec.set_dims_mapping([0, 1, -1])
        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec, self.attrs['shape']
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [0, 1, -1])
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [0, 1, -1, -1]
        )

        # shape: [-1, -1, 3072] --> [0, 0, -1, 192]
        # dims_mapping: [0, -1, 1] --> [0, -1, -1], [0, -1, -1, -1]
        self.x_dist_tensor_spec.shape = [-1, -1, 3072]
        self.attrs["shape"] = [0, 0, -1, 192]
        self.x_dist_tensor_spec.set_dims_mapping([0, -1, 1])
        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec, self.attrs['shape']
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [0, -1, -1])
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [0, -1, -1, -1]
        )

        # shape: [-1, -1, 3072] --> [0, 0, -1, 192]
        # dims_mapping: [1, -1, 0] --> [1, -1, 0], [1, -1, 0, -1]
        self.x_dist_tensor_spec.shape = [-1, -1, 3072]
        self.attrs["shape"] = [0, 0, -1, 192]
        self.x_dist_tensor_spec.set_dims_mapping([1, -1, 0])
        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec, self.attrs['shape']
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [1, -1, 0])
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [1, -1, 0, -1]
        )

        # shape: [1, 2048, 12288] --> [0, 0, 6, 2048]
        # dims_mapping: [0, -1, 1] --> [0, -1, 1], [0, -1, 1, -1]
        self.x_dist_tensor_spec.shape = [1, 2048, 12288]
        self.attrs["shape"] = [0, 0, 6, 2048]
        self.x_dist_tensor_spec.set_dims_mapping([0, -1, 1])
        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec, self.attrs['shape']
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [0, -1, 1])
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [0, -1, 1, -1]
        )

        # shape: [6, 12, 48, 24] --> [3, 24, 6, -1, -1]
        # raise error
        self.attrs["shape"] = [3, 24, 6, -1, -1]
        with self.assertRaises(ValueError):
            self.rule.infer_forward(
                self.x_dist_tensor_spec, self.attrs['shape']
            )

    def test_reshape_infer_backward(self):
        process_mesh = auto.ProcessMesh(mesh=[[0, 1, 2], [3, 4, 5]])

        output_tensor_dist_attr = TensorDistAttr()
        output_tensor_dist_attr.dims_mapping = [-1, -1, -1, -1]
        output_tensor_dist_attr.process_mesh = process_mesh

        # shape: [6, 12, 48, 24] --> [1, 72, 48, 4, 6] (input --> output)
        # dims_mapping: [-1, 0, 1, -1, -1] --> [0, -1, 1, -1], [-1, 0, 1, -1, -1] (output --> input, output)
        self.output_dist_tensor_spec = DistTensorSpec(
            [1, 72, 48, 4, 6], output_tensor_dist_attr
        )
        self.output_dist_tensor_spec.set_dims_mapping([-1, 0, 1, -1, -1])
        result_dist_attrs = self.rule.infer_backward(
            self.x_dist_tensor_spec,
            self.output_dist_tensor_spec,
            self.attrs['shape'],
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(inferred_input_dist_attrs), 1)
        self.assertEqual(len(inferred_output_dist_attrs), 1)
        self.assertEqual(
            inferred_input_dist_attrs[0].dims_mapping, [0, -1, 1, -1]
        )
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [-1, 0, 1, -1, -1]
        )

        # shape: [6, 12, 48, 24] --> [1, 72, 48, 4, 6] (input --> output)
        # dims_mapping: [-1, -1, -1, -1, -1] --> [-1, -1, -1, -1], [-1, -1, -1, -1, -1] (output --> input, output)
        self.output_dist_tensor_spec.shape = [1, 72, 48, 4, 6]
        self.output_dist_tensor_spec.set_dims_mapping([-1, -1, -1, -1, -1])
        result_dist_attrs = self.rule.infer_backward(
            self.x_dist_tensor_spec,
            self.output_dist_tensor_spec,
            self.attrs['shape'],
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(
            inferred_input_dist_attrs[0].dims_mapping, [-1, -1, -1, -1]
        )
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [-1, -1, -1, -1, -1]
        )

        # shape: [6, 12, 48, 24] --> [1, 72, 48, 4, 6] (input --> output)
        # dims_mapping: [-1, 1, -1, 0, -1] --> [1, -1, -1, 0] [-1, 1, -1, 0, -1] (output --> input, output)
        self.output_dist_tensor_spec.shape = [1, 72, 48, 4, 6]
        self.output_dist_tensor_spec.set_dims_mapping([-1, 1, -1, 0, -1])
        result_dist_attrs = self.rule.infer_backward(
            self.x_dist_tensor_spec,
            self.output_dist_tensor_spec,
            self.attrs['shape'],
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(
            inferred_input_dist_attrs[0].dims_mapping, [1, -1, -1, 0]
        )
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [-1, 1, -1, 0, -1]
        )

        # shape: [6, 12, 48, 24] --> [3, 24, 6, 8, 24] (input --> output)
        # dims_mapping: [1, -1, -1, -1, 0] --> [1, -1, -1, 0], [1, -1, -1, -1, 0] (output --> input, output)
        self.output_dist_tensor_spec.shape = [3, 24, 6, 8, 24]
        self.output_dist_tensor_spec.set_dims_mapping([1, -1, -1, -1, 0])

        result_dist_attrs = self.rule.infer_backward(
            self.x_dist_tensor_spec,
            self.output_dist_tensor_spec,
            self.attrs['shape'],
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(
            inferred_input_dist_attrs[0].dims_mapping, [1, -1, -1, 0]
        )
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [1, -1, -1, -1, 0]
        )

        # shape: [6, 12, 48, 24] --> [3, 24, 6, 8, 24] (input --> output)
        # dims_mapping: [-1, -1, 0, -1, 1] --> [-1, -1, 0, 1], [-1, -1, 0, -1, 1] (output --> input, output)
        self.output_dist_tensor_spec.shape = [3, 24, 6, 8, 24]
        self.output_dist_tensor_spec.set_dims_mapping([-1, -1, 0, -1, 1])

        result_dist_attrs = self.rule.infer_backward(
            self.x_dist_tensor_spec,
            self.output_dist_tensor_spec,
            self.attrs['shape'],
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(
            inferred_input_dist_attrs[0].dims_mapping, [-1, -1, 0, 1]
        )
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [-1, -1, 0, -1, 1]
        )

        # shape: [6, 12, 48, 24] --> [6, 12, 48, 24] (input --> output)
        # dims_mapping: [-1, -1, 0, 1] --> [-1, -1, 0, 1], [-1, -1, 0, 1] (output --> input, output)
        self.output_dist_tensor_spec.shape = [6, 12, 48, 24]
        self.output_dist_tensor_spec.set_dims_mapping([-1, -1, 0, 1])
        result_dist_attrs = self.rule.infer_backward(
            self.x_dist_tensor_spec,
            self.output_dist_tensor_spec,
            self.attrs['shape'],
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(
            inferred_input_dist_attrs[0].dims_mapping, [-1, -1, 0, 1]
        )
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [-1, -1, 0, 1]
        )

        # shape: [6, 12, 48, 24] --> [72, 3, 16, 24] (input --> output)
        # dims_mapping: [0, 1, -1, -1] --> [0, -1, 1, -1], [0, 1, -1, -1] (output --> input, output)
        self.output_dist_tensor_spec.shape = [72, 3, 16, 24]
        self.output_dist_tensor_spec.set_dims_mapping([0, 1, -1, -1])
        result_dist_attrs = self.rule.infer_backward(
            self.x_dist_tensor_spec,
            self.output_dist_tensor_spec,
            self.attrs['shape'],
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(
            inferred_input_dist_attrs[0].dims_mapping, [0, -1, 1, -1]
        )
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [0, 1, -1, -1]
        )

        # shape: [6, 12, 48, 24] --> [72, 3, 16, 24] (input --> output)
        # dims_mapping: [1, -1, -1, -1] --> [1, -1, -1, -1], [1, -1, -1, -1] (output --> input, output)
        self.output_dist_tensor_spec.shape = [72, 3, 16, 24]
        self.output_dist_tensor_spec.set_dims_mapping([1, -1, -1, -1])
        result_dist_attrs = self.rule.infer_backward(
            self.x_dist_tensor_spec,
            self.output_dist_tensor_spec,
            self.attrs['shape'],
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(
            inferred_input_dist_attrs[0].dims_mapping, [1, -1, -1, -1]
        )
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [1, -1, -1, -1]
        )

        # shape: [6, 12, 48, 24] --> [1, 72, 48, 4, 6] (input --> output)
        # dims_mapping: [-1, 0, -1, -1, 1] --> [0, -1, -1, -1], [-1, 0, -1, -1, -1] (output --> input, output)
        self.output_dist_tensor_spec.shape = [1, 72, 48, 4, 6]
        self.output_dist_tensor_spec.set_dims_mapping([-1, 0, -1, -1, 1])
        result_dist_attrs = self.rule.infer_backward(
            self.x_dist_tensor_spec,
            self.output_dist_tensor_spec,
            self.attrs['shape'],
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(
            inferred_input_dist_attrs[0].dims_mapping, [0, -1, -1, -1]
        )
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [-1, 0, -1, -1, -1]
        )

        # shape: [6, 12, 48, 24] --> [3, 24, 6, 8, 24] (input --> output)
        # dims_mapping: [-1, 1, -1, -1, 0] --> [-1, -1, -1, 0], [-1, -1, -1, -1, 0] (output --> input, output)
        self.output_dist_tensor_spec.shape = [3, 24, 6, 8, 24]
        self.output_dist_tensor_spec.set_dims_mapping([-1, 1, -1, -1, 0])
        result_dist_attrs = self.rule.infer_backward(
            self.x_dist_tensor_spec,
            self.output_dist_tensor_spec,
            self.attrs['shape'],
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(
            inferred_input_dist_attrs[0].dims_mapping, [-1, -1, -1, 0]
        )
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [-1, -1, -1, -1, 0]
        )

        # shape: [8, 1024, 3072] --> [0, 0, -1, 192] (input --> output)
        # dims_mapping: [0, 1, -1, -1] --> [0, 1, -1], [0, 1, -1, -1] (output --> input, output)
        self.x_dist_tensor_spec.shape = [8, 1024, 3072]
        self.output_dist_tensor_spec.shape = [0, 0, -1, 192]
        self.attrs["shape"] = [0, 0, -1, 192]
        self.output_dist_tensor_spec.set_dims_mapping([0, 1, -1, -1])
        result_dist_attrs = self.rule.infer_backward(
            self.x_dist_tensor_spec,
            self.output_dist_tensor_spec,
            self.attrs['shape'],
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [0, 1, -1])
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [0, 1, -1, -1]
        )

        # shape: [-1, -1, 3072] --> [0, 0, -1, 192] (input --> output)
        # dims_mapping: [0, 1, -1, -1] --> [0, 1, -1], [0, 1, -1, -1] (output --> input, output)
        self.x_dist_tensor_spec.shape = [-1, -1, 3072]
        self.output_dist_tensor_spec.shape = [0, 0, -1, 192]
        self.attrs["shape"] = [0, 0, -1, 192]
        self.output_dist_tensor_spec.set_dims_mapping([0, 1, -1, -1])
        result_dist_attrs = self.rule.infer_backward(
            self.x_dist_tensor_spec,
            self.output_dist_tensor_spec,
            self.attrs['shape'],
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [0, 1, -1])
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [0, 1, -1, -1]
        )

        # shape: [-1, -1, 3072] --> [0, 0, -1, 192] (input --> output)
        # dims_mapping: [0, -1, 1, -1] --> [0, -1, 1], [0, -1, 1, -1] (output --> input, output)
        self.x_dist_tensor_spec.shape = [-1, -1, 3072]
        self.output_dist_tensor_spec.shape = [0, 0, -1, 192]
        self.attrs["shape"] = [0, 0, -1, 192]
        self.output_dist_tensor_spec.set_dims_mapping([0, -1, 1, -1])
        result_dist_attrs = self.rule.infer_backward(
            self.x_dist_tensor_spec,
            self.output_dist_tensor_spec,
            self.attrs['shape'],
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [0, -1, 1])
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [0, -1, 1, -1]
        )


if __name__ == "__main__":
    unittest.main()
