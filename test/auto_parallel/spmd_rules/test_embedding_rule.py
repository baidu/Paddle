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


class TestEmbeddingSPMDRule(unittest.TestCase):
    def setUp(self):
        self.rule1 = core.get_phi_spmd_rule("lookup_table_v2")

    def test_embedding_infer_forward(self):
        # forward setup
        x_shape = [4, 1024]  # [B,S]
        table_shape = [512, 768]  # [V,H]
        process_mesh = auto.ProcessMesh(mesh=[[0, 1, 2, 3], [4, 5, 6, 7]])

        x_tensor_dist_attr = TensorDistAttr()
        x_tensor_dist_attr.process_mesh = process_mesh
        self.x_dist_tensor_spec = DistTensorSpec(x_shape, x_tensor_dist_attr)

        table_tensor_dist_attr = TensorDistAttr()
        table_tensor_dist_attr.process_mesh = process_mesh
        self.table_dist_tensor_spec = DistTensorSpec(
            table_shape, table_tensor_dist_attr
        )

        self.attrs = OrderedDict([('padding_idx', -1), ('sparse', False)])

        # data parallel
        self.x_dist_tensor_spec.set_dims_mapping([1, -1])
        self.table_dist_tensor_spec.set_dims_mapping([-1, -1])
        result_dist_attrs = self.rule1.infer_forward(
            self.x_dist_tensor_spec,
            self.table_dist_tensor_spec,
            self.attrs['padding_idx'],
            self.attrs['sparse'],
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(inferred_input_dist_attrs), 2)
        self.assertEqual(len(inferred_output_dist_attrs), 1)

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [1, -1])
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [-1, -1])
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [1, -1, -1]
        )

        # table col-wise parallel & dp
        self.x_dist_tensor_spec.set_dims_mapping([1, -1])
        self.table_dist_tensor_spec.set_dims_mapping([-1, 0])
        result_dist_attrs = self.rule1.infer_forward(
            self.x_dist_tensor_spec,
            self.table_dist_tensor_spec,
            self.attrs['padding_idx'],
            self.attrs['sparse'],
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [1, -1])
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [-1, 0])
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [1, -1, 0])

        # table row-wise parallel & dp
        self.x_dist_tensor_spec.set_dims_mapping([1, -1])
        self.table_dist_tensor_spec.set_dims_mapping([0, -1])
        result_dist_attrs = self.rule1.infer_forward(
            self.x_dist_tensor_spec,
            self.table_dist_tensor_spec,
            self.attrs['padding_idx'],
            self.attrs['sparse'],
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [1, -1])
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [0, -1])
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [1, -1, -1]
        )
        self.assertEqual(inferred_output_dist_attrs[0]._is_partial(), True)
        self.assertEqual(inferred_output_dist_attrs[0]._partial_dims(), {0})

        # table row-wise parallel & padding_idx
        self.x_dist_tensor_spec.set_dims_mapping([1, -1])
        self.table_dist_tensor_spec.set_dims_mapping([0, -1])
        self.attrs['padding_idx'] = 128
        with self.assertRaises(ValueError):
            result_dist_attrs = self.rule1.infer_forward(
                [self.x_dist_tensor_spec, self.table_dist_tensor_spec],
                self.attrs,
            )

        # table row-wise parallel & sparse
        self.x_dist_tensor_spec.set_dims_mapping([1, -1])
        self.table_dist_tensor_spec.set_dims_mapping([0, -1])
        self.attrs['padding_idx'] = -1
        self.attrs['sparse'] = True
        with self.assertRaises(ValueError):
            result_dist_attrs = self.rule1.infer_forward(
                self.x_dist_tensor_spec,
                self.table_dist_tensor_spec,
                self.attrs['padding_idx'],
                self.attrs['sparse'],
            )

    def test_embedding_infer_backward(self):
        # backward setup
        process_mesh = auto.ProcessMesh(mesh=[[0, 1, 2, 3], [4, 5, 6, 7]])

        x_shape = [4, 1024]  # [B,S]
        table_shape = [512, 768]  # [V,H]

        x_tensor_dist_attr = TensorDistAttr()
        x_tensor_dist_attr.process_mesh = (
            process_mesh  # not set the dims mapping is ok.
        )
        self.x_dist_tensor_spec = DistTensorSpec(x_shape, x_tensor_dist_attr)

        table_tensor_dist_attr = TensorDistAttr()
        table_tensor_dist_attr.process_mesh = (
            process_mesh  # not set the dims mapping is ok.
        )
        self.table_dist_tensor_spec = DistTensorSpec(
            table_shape, table_tensor_dist_attr
        )

        out_shape = [4, 1024, 768]  # [B,S, H]
        out_tensor_dist_attr = TensorDistAttr()
        out_tensor_dist_attr.process_mesh = process_mesh
        self.out_dist_tensor_spec = DistTensorSpec(
            out_shape, out_tensor_dist_attr
        )

        self.attrs = OrderedDict([('padding_idx', -1), ('sparse', False)])

        # data parallel
        self.out_dist_tensor_spec.set_dims_mapping([1, -1, -1])
        result_dist_attrs = self.rule1.infer_backward(
            self.x_dist_tensor_spec,
            self.table_dist_tensor_spec,
            self.out_dist_tensor_spec,
            self.attrs['padding_idx'],
            self.attrs['sparse'],
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(inferred_input_dist_attrs), 2)
        self.assertEqual(len(inferred_output_dist_attrs), 1)

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [1, -1])
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [-1, -1])
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [1, -1, -1]
        )

        # table col-wise parallel & dp
        self.out_dist_tensor_spec.set_dims_mapping([-1, 0, 1])
        result_dist_attrs = self.rule1.infer_backward(
            self.x_dist_tensor_spec,
            self.table_dist_tensor_spec,
            self.out_dist_tensor_spec,
            self.attrs['padding_idx'],
            self.attrs['sparse'],
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [-1, 0])
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [-1, 1])
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [-1, 0, 1])

        # sharded on multiple broadcast axes
        self.out_dist_tensor_spec.set_dims_mapping([1, 0, -1])

        result_dist_attrs = self.rule1.infer_backward(
            self.x_dist_tensor_spec,
            self.table_dist_tensor_spec,
            self.out_dist_tensor_spec,
            self.attrs['padding_idx'],
            self.attrs['sparse'],
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [1, 0])
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [-1, -1])
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [1, 0, -1])

        # table row-wise parallel
        # skipped


if __name__ == "__main__":
    unittest.main()
