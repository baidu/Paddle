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
from collections import OrderedDict

from paddle.distributed.auto_parallel.static.dist_attribute import (
    DistTensorSpec,
    TensorDistAttr,
)
from paddle.distributed.fleet import auto
from paddle.framework import core


class TestPNormSPMDRule(unittest.TestCase):
    """
    Unit tests for p_norm spmd rule.
    """

    def config(self):
        self.kernel = "p_norm"

    def setUp(self):
        self.config()
        self.rule = core.get_phi_spmd_rule(self.kernel)

        x_shape = [64, 32]
        process_mesh = auto.ProcessMesh(mesh=[0, 1, 2, 3])

        x_tensor_dist_attr = TensorDistAttr()
        x_tensor_dist_attr.dims_mapping = [1, 0]
        x_tensor_dist_attr.process_mesh = process_mesh
        self.x_dist_tensor_spec = DistTensorSpec(x_shape, x_tensor_dist_attr)

        self.out_dist_tensor_spec = DistTensorSpec(self.x_dist_tensor_spec)

        self.attrs = OrderedDict(
            [
                ('porder', 2.0),
                ('axis', 0),
                ('epsilon', 1.0e-12),
                ('keepdims', False),
                ('asvector', False),
            ]
        )

    def test_infer_forward(self):
        # reduce on dim 0, keepdims = false, asvector = false
        # [0, -1] --> [0, -1], [-1], partial_on_dim:[0]
        self.attrs['axis'] = 0
        self.attrs['keepdims'] = False
        self.attrs['asvector'] = False
        self.x_dist_tensor_spec.set_dims_mapping([0, -1])
        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec,
            self.attrs['porder'],
            self.attrs['axis'],
            self.attrs['epsilon'],
            self.attrs['keepdims'],
            self.attrs['asvector'],
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(inferred_input_dist_attrs), 1)
        self.assertEqual(len(inferred_output_dist_attrs), 1)

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [0, -1])
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [-1])
        self.assertEqual(inferred_output_dist_attrs[0]._is_partial(), True)
        self.assertEqual(inferred_output_dist_attrs[0]._partial_dims(), {0})

        # reduce on dim 0, keepdims = true, asvector = false
        # [0, -1] --> [0, -1], [-1, -1], partial_on_dim:[0]

        self.attrs['keepdims'] = True
        self.attrs['axis'] = 0
        self.attrs['asvector'] = False
        self.x_dist_tensor_spec.set_dims_mapping([0, -1])
        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec,
            self.attrs['porder'],
            self.attrs['axis'],
            self.attrs['epsilon'],
            self.attrs['keepdims'],
            self.attrs['asvector'],
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [0, -1])
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [-1, -1])
        self.assertEqual(inferred_output_dist_attrs[0]._is_partial(), True)
        self.assertEqual(inferred_output_dist_attrs[0]._partial_dims(), {0})

        # reduce on dim 1, keepdims = false, asvector = false
        # [0, -1] --> [0, -1], [0], partial_on_dim:[]
        self.attrs['keepdims'] = False
        self.attrs['axis'] = 1
        self.attrs['asvector'] = False
        self.x_dist_tensor_spec.set_dims_mapping([0, -1])
        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec,
            self.attrs['porder'],
            self.attrs['axis'],
            self.attrs['epsilon'],
            self.attrs['keepdims'],
            self.attrs['asvector'],
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [0, -1])
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [0])
        self.assertEqual(inferred_output_dist_attrs[0]._is_partial(), False)

        # reduce on dim 1, keepdims = true, asvector = false
        # [0, -1] --> [0, -1], [0, -1], partial_on_dim:[]
        self.attrs['keepdims'] = True
        self.attrs['axis'] = 1
        self.attrs['asvector'] = False
        self.x_dist_tensor_spec.set_dims_mapping([0, -1])
        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec,
            self.attrs['porder'],
            self.attrs['axis'],
            self.attrs['epsilon'],
            self.attrs['keepdims'],
            self.attrs['asvector'],
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [0, -1])
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [0, -1])
        self.assertEqual(inferred_output_dist_attrs[0]._is_partial(), False)

        # reduce on dim 0 and 1, keepdims = false, asvector = true
        # [0, -1] --> [0, -1], [], partial_on_dim:[0]
        self.attrs['keepdims'] = False
        self.attrs['axis'] = 0
        self.attrs['asvector'] = True
        self.x_dist_tensor_spec.set_dims_mapping([0, -1])
        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec,
            self.attrs['porder'],
            self.attrs['axis'],
            self.attrs['epsilon'],
            self.attrs['keepdims'],
            self.attrs['asvector'],
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [0, -1])
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [])
        self.assertEqual(inferred_output_dist_attrs[0]._is_partial(), True)
        self.assertEqual(inferred_output_dist_attrs[0]._partial_dims(), {0})

        # reduce on dim 0 and 1, keepdims = true, asvector = true
        # [0, -1] --> [0, -1], [-1, -1], partial_on_dim:[0]
        self.attrs['keepdims'] = True
        self.attrs['axis'] = 0
        self.attrs['asvector'] = True
        self.x_dist_tensor_spec.set_dims_mapping([0, -1])
        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec,
            self.attrs['porder'],
            self.attrs['axis'],
            self.attrs['epsilon'],
            self.attrs['keepdims'],
            self.attrs['asvector'],
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [0, -1])
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [-1, -1])
        self.assertEqual(inferred_output_dist_attrs[0]._is_partial(), True)
        self.assertEqual(inferred_output_dist_attrs[0]._partial_dims(), {0})

    def test_infer_backward(self):
        # reduce on dim 0, keepdims = false, asvector = false
        # [-1] --> [-1, -1], [-1] (output --> input, output)
        self.attrs['keepdims'] = False
        self.attrs['axis'] = 0
        self.attrs['asvector'] = False
        self.out_dist_tensor_spec.shape = [32]
        self.out_dist_tensor_spec.set_dims_mapping([-1])
        result_dist_attrs = self.rule.infer_backward(
            self.x_dist_tensor_spec,
            self.out_dist_tensor_spec,
            self.attrs['porder'],
            self.attrs['axis'],
            self.attrs['epsilon'],
            self.attrs['keepdims'],
            self.attrs['asvector'],
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(inferred_input_dist_attrs), 1)
        self.assertEqual(len(inferred_output_dist_attrs), 1)

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [-1, -1])
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [-1])

        # reduce on dim 0, keepdims = true, asvector = false
        # [-1, -1] --> [-1, -1], [-1, -1] (output --> input, output)
        self.attrs['keepdims'] = True
        self.attrs['axis'] = 0
        self.attrs['asvector'] = False
        self.out_dist_tensor_spec.shape = [1, 32]
        self.out_dist_tensor_spec.set_dims_mapping([-1, -1])
        result_dist_attrs = self.rule.infer_backward(
            self.x_dist_tensor_spec,
            self.out_dist_tensor_spec,
            self.attrs['porder'],
            self.attrs['axis'],
            self.attrs['epsilon'],
            self.attrs['keepdims'],
            self.attrs['asvector'],
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [-1, -1])
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [-1, -1])

        # reduce on dim 1, keepdims = false, asvector = false
        # [0] --> [0, -1], [0] (output --> input, output)
        self.attrs['keepdims'] = False
        self.attrs['axis'] = 1
        self.attrs['asvector'] = False
        self.out_dist_tensor_spec.shape = [64]
        self.out_dist_tensor_spec.set_dims_mapping([0])
        result_dist_attrs = self.rule.infer_backward(
            self.x_dist_tensor_spec,
            self.out_dist_tensor_spec,
            self.attrs['porder'],
            self.attrs['axis'],
            self.attrs['epsilon'],
            self.attrs['keepdims'],
            self.attrs['asvector'],
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [0, -1])
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [0])

        # reduce on dim 1, keepdims = true, asvector = false
        # [0, -1] --> [0, -1], [0, -1] (output --> input, output)
        self.attrs['keepdims'] = True
        self.attrs['axis'] = 1
        self.attrs['asvector'] = False
        self.out_dist_tensor_spec.shape = [64, 1]
        self.out_dist_tensor_spec.set_dims_mapping([0, -1])
        result_dist_attrs = self.rule.infer_backward(
            self.x_dist_tensor_spec,
            self.out_dist_tensor_spec,
            self.attrs['porder'],
            self.attrs['axis'],
            self.attrs['epsilon'],
            self.attrs['keepdims'],
            self.attrs['asvector'],
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [0, -1])
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [0, -1])

        # reduce on dim 0 and 1, keepdims = false, asvector = true
        # [] --> [-1, -1], [] (output --> input, output)
        self.attrs['keepdims'] = False
        self.attrs['axis'] = 0
        self.attrs['asvector'] = True
        self.out_dist_tensor_spec.shape = []
        self.out_dist_tensor_spec.set_dims_mapping([])
        result_dist_attrs = self.rule.infer_backward(
            self.x_dist_tensor_spec,
            self.out_dist_tensor_spec,
            self.attrs['porder'],
            self.attrs['axis'],
            self.attrs['epsilon'],
            self.attrs['keepdims'],
            self.attrs['asvector'],
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [-1, -1])
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [])

        # reduce on dim 0 and 1, keepdims = true, asvector = true
        # [-1, -1] --> [-1, -1], [-1, -1] (output --> input, output)
        self.attrs['keepdims'] = True
        self.attrs['axis'] = 0
        self.attrs['asvector'] = True
        self.out_dist_tensor_spec.shape = [1, 1]
        self.out_dist_tensor_spec.set_dims_mapping([-1, -1])
        result_dist_attrs = self.rule.infer_backward(
            self.x_dist_tensor_spec,
            self.out_dist_tensor_spec,
            self.attrs['porder'],
            self.attrs['axis'],
            self.attrs['epsilon'],
            self.attrs['keepdims'],
            self.attrs['asvector'],
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [-1, -1])
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [-1, -1])


if __name__ == "__main__":
    unittest.main()
