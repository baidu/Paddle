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


class TestFlashAttentionSPMDRule(unittest.TestCase):
    """
    Unit tests for layer_norm spmd rule.
    """

    def setUp(self):
        self.rule = core.get_phi_spmd_rule("flash_attention")
        process_mesh = auto.ProcessMesh(mesh=[[0, 1, 2], [3, 4, 5]])
        self.process_mesh = process_mesh

        q_tensor_dist_attr = TensorDistAttr()
        q_tensor_dist_attr.process_mesh = process_mesh
        q_tensor_dist_attr.dims_mapping = [0, -1, 1, -1]
        q_shape = [2, 512, 64, 1024]
        q_spec = DistTensorSpec(q_shape, q_tensor_dist_attr)
        self.q_spec = q_spec

        k_tensor_dist_attr = TensorDistAttr()
        k_tensor_dist_attr.process_mesh = process_mesh
        k_tensor_dist_attr.dims_mapping = [0, -1, -1, -1]
        k_shape = [2, 1024, 64, 1024]
        k_spec = DistTensorSpec(k_shape, k_tensor_dist_attr)
        self.k_spec = k_spec

        v_tensor_dist_attr = TensorDistAttr()
        v_tensor_dist_attr.process_mesh = process_mesh
        v_tensor_dist_attr.dims_mapping = [0, -1, -1, -1]
        v_shape = [2, 1024, 64, 512]
        v_spec = DistTensorSpec(v_shape, k_tensor_dist_attr)
        self.v_spec = v_spec

        out_tensor_dist_attr = TensorDistAttr()
        out_tensor_dist_attr.process_mesh = process_mesh
        out_tensor_dist_attr.dims_mapping = [0, -1, 1, -1]
        out_shape = [2, 512, 64, 512]
        out_spec = DistTensorSpec(out_shape, out_tensor_dist_attr)
        self.out_spec = out_spec

        softmax_lse_dist_attr = TensorDistAttr()
        softmax_lse_dist_attr.process_mesh = process_mesh
        softmax_lse_dist_attr.dims_mapping = [0, 1, -1]
        softmax_lse_shape = [2, 64, 512]
        softmax_lse_spec = DistTensorSpec(
            softmax_lse_shape, softmax_lse_dist_attr
        )
        self.softmax_lse_spec = softmax_lse_spec

    def create_empty_tensor(self):
        dist_attr = TensorDistAttr()
        dist_attr.process_mesh = self.process_mesh
        dist_attr.dims_mapping = []
        shape = []
        return DistTensorSpec(shape, dist_attr)

    def test_infer_forward(self):
        result_dist_attrs = self.rule.infer_forward(
            self.q_spec,
            self.k_spec,
            self.v_spec,
            self.create_empty_tensor(),
            self.create_empty_tensor(),
            0.0,
            False,
            False,
            False,
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(inferred_input_dist_attrs), 5)
        self.assertEqual(len(inferred_output_dist_attrs), 4)

        self.assertEqual(
            inferred_input_dist_attrs[0].dims_mapping, [0, -1, 1, -1]
        )
        self.assertEqual(
            inferred_input_dist_attrs[1].dims_mapping, [0, -1, 1, -1]
        )
        self.assertEqual(
            inferred_input_dist_attrs[2].dims_mapping, [0, -1, 1, -1]
        )
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [0, -1, 1, -1]
        )
        self.assertEqual(inferred_output_dist_attrs[2].dims_mapping, [0, 1, -1])

    def test_infer_backward(self):
        result_dist_attrs = self.rule.infer_backward(
            self.q_spec,
            self.k_spec,
            self.v_spec,
            self.create_empty_tensor(),
            self.create_empty_tensor(),
            self.out_spec,
            self.create_empty_tensor(),
            self.softmax_lse_spec,
            self.create_empty_tensor(),
            0.0,
            False,
            False,
            False,
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(inferred_input_dist_attrs), 5)
        self.assertEqual(len(inferred_output_dist_attrs), 4)

        self.assertEqual(
            inferred_input_dist_attrs[0].dims_mapping, [0, -1, 1, -1]
        )
        self.assertEqual(
            inferred_input_dist_attrs[1].dims_mapping, [0, -1, 1, -1]
        )
        self.assertEqual(
            inferred_input_dist_attrs[2].dims_mapping, [0, -1, 1, -1]
        )
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [0, -1, 1, -1]
        )
        self.assertEqual(inferred_output_dist_attrs[2].dims_mapping, [0, 1, -1])


if __name__ == "__main__":
    unittest.main()
