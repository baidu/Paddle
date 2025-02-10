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


class TestEmbeddingSPMDRule(unittest.TestCase):
    def setUp(self):
        self.rule = core.get_phi_spmd_rule("c_embedding")

    def test_c_embedding_infer_forward(self):
        # forward setup
        table_shape = [512, 768]  # [V,H]
        x_shape = [4, 1024]  # [B,S]
        process_mesh = auto.ProcessMesh(mesh=[[0, 1, 2, 3], [4, 5, 6, 7]])
        table_tensor_dist_attr = TensorDistAttr()
        table_tensor_dist_attr.process_mesh = process_mesh
        self.table_dist_tensor_spec = DistTensorSpec(
            table_shape, table_tensor_dist_attr
        )
        x_tensor_dist_attr = TensorDistAttr()
        x_tensor_dist_attr.process_mesh = process_mesh
        self.x_dist_tensor_spec = DistTensorSpec(x_shape, x_tensor_dist_attr)
        self.attrs = OrderedDict([('start_index', 0), ('vocab_size', -1)])

        # data parallel
        self.table_dist_tensor_spec.set_dims_mapping([-1, -1])
        self.x_dist_tensor_spec.set_dims_mapping([1, -1])
        result_dist_attrs = self.rule.infer_forward(
            self.table_dist_tensor_spec,
            self.x_dist_tensor_spec,
            self.attrs['start_index'],
            self.attrs['vocab_size'],
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(inferred_input_dist_attrs), 2)
        self.assertEqual(len(inferred_output_dist_attrs), 1)
        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [-1, -1])
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [1, -1])
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [1, -1, -1]
        )

        # table row-wise parallel
        self.table_dist_tensor_spec.set_dims_mapping([1, -1])
        self.x_dist_tensor_spec.set_dims_mapping([-1, -1])
        result_dist_attrs = self.rule.infer_forward(
            self.table_dist_tensor_spec,
            self.x_dist_tensor_spec,
            self.attrs['start_index'],
            self.attrs['vocab_size'],
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [1, -1])
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [-1, -1])
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [-1, -1, -1]
        )
        self.assertEqual(inferred_output_dist_attrs[0]._is_partial(), True)
        self.assertEqual(inferred_output_dist_attrs[0]._partial_dims(), {1})

    def test_c_embedding_infer_backward(self):
        # backward setup
        process_mesh = auto.ProcessMesh(mesh=[[0, 1, 2, 3], [4, 5, 6, 7]])
        table_shape = [512, 768]  # [V,H]
        x_shape = [4, 1024]  # [B,S]
        table_tensor_dist_attr = TensorDistAttr()
        table_tensor_dist_attr.process_mesh = process_mesh
        self.table_dist_tensor_spec = DistTensorSpec(
            table_shape, table_tensor_dist_attr
        )
        x_tensor_dist_attr = TensorDistAttr()
        x_tensor_dist_attr.process_mesh = process_mesh
        self.x_dist_tensor_spec = DistTensorSpec(x_shape, x_tensor_dist_attr)
        out_shape = [4, 1024, 768]  # [B,S,H]
        out_tensor_dist_attr = TensorDistAttr()
        out_tensor_dist_attr.process_mesh = process_mesh
        self.out_dist_tensor_spec = DistTensorSpec(
            out_shape, out_tensor_dist_attr
        )
        self.attrs = OrderedDict([('start_index', 0), ('vocab_size', -1)])

        # table row-wise parallel
        self.table_dist_tensor_spec.set_dims_mapping([1, -1])
        self.x_dist_tensor_spec.set_dims_mapping([-1, -1])
        self.out_dist_tensor_spec.set_dims_mapping([-1, -1, -1])
        result_dist_attrs = self.rule.infer_backward(
            self.table_dist_tensor_spec,
            self.x_dist_tensor_spec,
            self.out_dist_tensor_spec,
            self.attrs['start_index'],
            self.attrs['vocab_size'],
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(inferred_input_dist_attrs), 3)
        self.assertEqual(len(inferred_output_dist_attrs), 1)
        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [1, -1])
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [-1, -1])
        self.assertEqual(
            inferred_input_dist_attrs[2].dims_mapping, [-1, -1, -1]
        )
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [1, -1])

        # data parallel
        self.x_dist_tensor_spec.set_dims_mapping([0, -1])
        self.out_dist_tensor_spec.set_dims_mapping([0, -1, -1])
        result_dist_attrs = self.rule.infer_backward(
            self.table_dist_tensor_spec,
            self.x_dist_tensor_spec,
            self.out_dist_tensor_spec,
            self.attrs['start_index'],
            self.attrs['vocab_size'],
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(inferred_input_dist_attrs), 3)
        self.assertEqual(len(inferred_output_dist_attrs), 1)
        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [1, -1])
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [0, -1])
        self.assertEqual(inferred_input_dist_attrs[2].dims_mapping, [0, -1, -1])
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [1, -1])


if __name__ == "__main__":
    unittest.main()
