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


class TestCSoftmaxWithMultiLabelCrossEntropySPMDRule(unittest.TestCase):
    """
    Unit tests for c_softmax_with_cross_entropy spmd rule.
    """

    def setUp(self):
        self.rule1 = core.get_phi_spmd_rule(
            "c_softmax_with_multi_label_cross_entropy"
        )
        process_mesh = auto.ProcessMesh(mesh=[[0, 1], [2, 3]])
        logit_shape = [16, 128, 128]
        logit_tensor_dist_attr = TensorDistAttr()
        logit_tensor_dist_attr.process_mesh = process_mesh
        self.logit_dist_tensor_spec = DistTensorSpec(
            logit_shape, logit_tensor_dist_attr
        )

        label_shape = [16, 128, 4]
        label_tensor_dist_attr = TensorDistAttr()
        label_tensor_dist_attr.process_mesh = process_mesh
        self.label_dist_tensor_spec = DistTensorSpec(
            label_shape, label_tensor_dist_attr
        )

        smooth_weight_shape = [16, 128, 4]
        smooth_weight_tensor_dist_attr = TensorDistAttr()
        smooth_weight_tensor_dist_attr.process_mesh = process_mesh
        self.smooth_weight_dist_tensor_spec = DistTensorSpec(
            smooth_weight_shape, smooth_weight_tensor_dist_attr
        )

        softmax_shape = [16, 128, 128]
        softmax_tensor_dist_attr = TensorDistAttr()
        softmax_tensor_dist_attr.process_mesh = process_mesh
        self.softmax_dist_tensor_spec = DistTensorSpec(
            softmax_shape, softmax_tensor_dist_attr
        )

        loss_shape = [16, 128, 1]
        loss_tensor_dist_attr = TensorDistAttr()
        loss_tensor_dist_attr.process_mesh = process_mesh
        self.loss_dist_tensor_spec = DistTensorSpec(
            loss_shape, loss_tensor_dist_attr
        )

        self.attrs = OrderedDict(
            [
                ('ignore_index', -1),
                ('ring_id', 0),
                ('rank', 0),
                ('nranks', 2),
            ]
        )

    def test_infer_forward(self):
        # llama MP case
        # [-1, -1, 1] [-1, -1, -1] (logit, label) -->
        # [-1, -1, 1] [-1, -1, -1] (logit, label)
        # [-1, -1, 1] [-1, -1, -1] (softmax, loss)
        self.logit_dist_tensor_spec.set_dims_mapping([-1, -1, 1])
        self.label_dist_tensor_spec.set_dims_mapping([-1, -1, -1])
        self.smooth_weight_dist_tensor_spec.set_dims_mapping([-1, -1, -1])

        infered_dist_attr = self.rule1.infer_forward(
            self.logit_dist_tensor_spec,
            self.label_dist_tensor_spec,
            self.smooth_weight_dist_tensor_spec,
            self.attrs['ignore_index'],
            self.attrs['ring_id'],
            self.attrs['rank'],
            self.attrs['nranks'],
        )

        self.assertEqual(len(infered_dist_attr), 2)
        infered_input_dist_attr = infered_dist_attr[0]
        infered_output_dist_attr = infered_dist_attr[1]

        self.assertEqual(len(infered_input_dist_attr), 3)
        self.assertEqual(len(infered_output_dist_attr), 2)

        self.assertEqual(
            infered_input_dist_attr[0].dims_mapping, [-1, -1, 1]
        )  # logit
        self.assertEqual(
            infered_input_dist_attr[1].dims_mapping, [-1, -1, -1]
        )  # label
        self.assertEqual(
            infered_input_dist_attr[2].dims_mapping, [-1, -1, -1]
        )  # smooth_weight
        self.assertEqual(
            infered_output_dist_attr[0].dims_mapping, [-1, -1, 1]
        )  # softmax
        self.assertEqual(
            infered_output_dist_attr[1].dims_mapping, [-1, -1, -1]
        )  # loss

        # llama MP-DP case
        # [0, -1, 1] [0, -1, -1] (logit, label) -->
        # [0, -1, 1] [0, -1, -1] (logit, label)
        # [0, -1, 1] [0, -1, -1] (softmax, loss)
        self.logit_dist_tensor_spec.set_dims_mapping([0, -1, 1])
        self.label_dist_tensor_spec.set_dims_mapping([0, -1, -1])
        self.smooth_weight_dist_tensor_spec.set_dims_mapping([0, -1, -1])

        infered_dist_attr = self.rule1.infer_forward(
            self.logit_dist_tensor_spec,
            self.label_dist_tensor_spec,
            self.smooth_weight_dist_tensor_spec,
            self.attrs['ignore_index'],
            self.attrs['ring_id'],
            self.attrs['rank'],
            self.attrs['nranks'],
        )

        self.assertEqual(len(infered_dist_attr), 2)
        infered_input_dist_attr = infered_dist_attr[0]
        infered_output_dist_attr = infered_dist_attr[1]

        self.assertEqual(len(infered_input_dist_attr), 3)
        self.assertEqual(len(infered_output_dist_attr), 2)

        self.assertEqual(
            infered_input_dist_attr[0].dims_mapping, [0, -1, 1]
        )  # logit
        self.assertEqual(
            infered_input_dist_attr[1].dims_mapping, [0, -1, -1]
        )  # label
        self.assertEqual(
            infered_input_dist_attr[2].dims_mapping, [0, -1, -1]
        )  # smooth_weight
        self.assertEqual(
            infered_output_dist_attr[0].dims_mapping, [0, -1, 1]
        )  # softmax
        self.assertEqual(
            infered_output_dist_attr[1].dims_mapping, [0, -1, -1]
        )  # loss


if __name__ == "__main__":
    unittest.main()
