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

import numpy as np

from paddle.distributed.auto_parallel.static.dist_attribute import (
    DistTensorSpec,
    TensorDistAttr,
)
from paddle.distributed.fleet import auto
from paddle.framework import core


class TestLayerNormSPMDRule(unittest.TestCase):
    """
    Unit tests for layer_norm spmd rule.
    """

    def setUp(self):
        self.rule = core.get_phi_spmd_rule("layer_norm")

        x_shape = [64, 32, 1024]
        scale_shape = [1024]
        bias_shape = [1024]
        process_mesh = auto.ProcessMesh(mesh=[[0, 1, 2], [3, 4, 5]])

        x_tensor_dist_attr = TensorDistAttr()
        x_tensor_dist_attr.dims_mapping = [-1, -1, -1]
        x_tensor_dist_attr.process_mesh = process_mesh
        self.x_spec = DistTensorSpec(x_shape, x_tensor_dist_attr)

        self.scale_spec = DistTensorSpec(self.x_spec)
        self.bias_spec = DistTensorSpec(self.x_spec)
        self.scale_spec.shape = scale_shape
        self.scale_spec.set_dims_mapping([-1])
        self.bias_spec.shape = bias_shape
        self.bias_spec.set_dims_mapping([-1])

        self.out_spec = DistTensorSpec(self.x_spec)
        self.mean_spec = DistTensorSpec(self.x_spec)
        self.var_spec = DistTensorSpec(self.x_spec)

        self.attrs = OrderedDict([('epsilon', 1e-3), ('begin_norm_axis', 2)])

    def test_infer_forward(self):
        # ijk[1, -1, -1], k[-1], k[-1] -->
        # ijk[1, -1, -1], k[-1], k[-1], (inputs)
        # ijk[1, -1, -1], ij[1, -1], ij[1, -1],(outputs)
        # begin_norm_axis=2
        self.x_spec.set_dims_mapping([1, -1, -1])
        self.bias_spec.set_dims_mapping([-1])
        self.scale_spec.set_dims_mapping([-1])

        result_dist_attrs = self.rule.infer_forward(
            self.x_spec,
            self.scale_spec,
            self.bias_spec,
            self.attrs['epsilon'],
            self.attrs['begin_norm_axis'],
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(inferred_input_dist_attrs), 3)
        self.assertEqual(len(inferred_output_dist_attrs), 3)

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [1, -1, -1])
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [-1])
        self.assertEqual(inferred_input_dist_attrs[2].dims_mapping, [-1])
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [1, -1, -1]
        )
        self.assertEqual(inferred_output_dist_attrs[1].dims_mapping, [1, -1])
        self.assertEqual(inferred_output_dist_attrs[2].dims_mapping, [1, -1])

        # ijk[1, 0, -1],k[0],k[0] -->
        # [1, 0, -1], [-1], [-1] (inputs)
        # [1, 0, -1], [1, 0], [1, 0] (outputs)
        # begin_norm_axis=2
        self.x_spec.set_dims_mapping([1, 0, -1])
        self.scale_spec.set_dims_mapping([0])
        self.bias_spec.set_dims_mapping([0])

        result_dist_attrs = self.rule.infer_forward(
            self.x_spec,
            self.scale_spec,
            self.bias_spec,
            self.attrs['epsilon'],
            self.attrs['begin_norm_axis'],
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(inferred_input_dist_attrs), 3)
        self.assertEqual(len(inferred_output_dist_attrs), 3)

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [1, 0, -1])
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [-1])
        self.assertEqual(inferred_input_dist_attrs[2].dims_mapping, [-1])
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [1, 0, -1])
        self.assertEqual(inferred_output_dist_attrs[1].dims_mapping, [1, 0])
        self.assertEqual(inferred_output_dist_attrs[2].dims_mapping, [1, 0])

        # ijk[0, -1, -1],y[-1],y[1] -->
        # ijk[0, -1, -1],y[-1],y[-1], (inputs)
        # ijk[0, -1, -1], ij[0], ij[0], y=jk (outputs)
        # begin_norm_axis=1
        self.attrs['begin_norm_axis'] = 1
        self.x_spec.set_dims_mapping([0, -1, -1])
        x_shape = self.x_spec.shape
        self.scale_spec.shape = [x_shape[1] * x_shape[2]]
        self.bias_spec.shape = [x_shape[1] * x_shape[2]]
        self.scale_spec.set_dims_mapping([-1])
        self.bias_spec.set_dims_mapping([1])
        self.mean_spec.shape = [x_shape[1]]
        self.var_spec.shape = [x_shape[1]]

        result_dist_attrs = self.rule.infer_forward(
            self.x_spec,
            self.scale_spec,
            self.bias_spec,
            self.attrs['epsilon'],
            self.attrs['begin_norm_axis'],
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(inferred_input_dist_attrs), 3)
        self.assertEqual(len(inferred_output_dist_attrs), 3)

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [0, -1, -1])
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [-1])
        self.assertEqual(inferred_input_dist_attrs[2].dims_mapping, [-1])
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [0, -1, -1]
        )
        self.assertEqual(inferred_output_dist_attrs[1].dims_mapping, [0])
        self.assertEqual(inferred_output_dist_attrs[2].dims_mapping, [0])

    def test_infer_forward_without_bias(self):
        # ijk[1, -1, -1], k[-1], k[-1] -->
        # ijk[1, -1, -1], k[-1], k[-1], (inputs)
        # ijk[1, -1, -1], ij[1, -1], ij[1, -1],(outputs)
        # begin_norm_axis=2
        self.x_spec.set_dims_mapping([1, -1, -1])
        self.bias_spec = DistTensorSpec([0], TensorDistAttr())
        self.scale_spec.set_dims_mapping([-1])

        result_dist_attrs = self.rule.infer_forward(
            self.x_spec,
            self.scale_spec,
            self.bias_spec,
            self.attrs['epsilon'],
            self.attrs['begin_norm_axis'],
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(inferred_input_dist_attrs), 3)
        self.assertEqual(len(inferred_output_dist_attrs), 3)

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [1, -1, -1])
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [-1])
        self.assertEqual(inferred_input_dist_attrs[2].dims_mapping, [-1])
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [1, -1, -1]
        )
        self.assertEqual(inferred_output_dist_attrs[1].dims_mapping, [1, -1])
        self.assertEqual(inferred_output_dist_attrs[2].dims_mapping, [1, -1])

        # ijk[1, 0, -1],k[0],k[0] -->
        # [1, 0, -1], [-1], [-1] (inputs)
        # [1, 0, -1], [1, 0], [1, 0] (outputs)
        self.x_spec.set_dims_mapping([1, 0, -1])
        self.scale_spec = DistTensorSpec([0], TensorDistAttr())
        self.bias_spec.set_dims_mapping([0])

        result_dist_attrs = self.rule.infer_forward(
            self.x_spec,
            self.scale_spec,
            self.bias_spec,
            self.attrs['epsilon'],
            self.attrs['begin_norm_axis'],
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(inferred_input_dist_attrs), 3)
        self.assertEqual(len(inferred_output_dist_attrs), 3)

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [1, 0, -1])
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [-1])
        self.assertEqual(inferred_input_dist_attrs[2].dims_mapping, [-1])
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [1, 0, -1])
        self.assertEqual(inferred_output_dist_attrs[1].dims_mapping, [1, 0])
        self.assertEqual(inferred_output_dist_attrs[2].dims_mapping, [1, 0])

    def test_infer_backward(self):
        # [1, -1, -1], [1, -1], [1, -1] (outputs) -->
        # [1, -1, -1], [-1], [-1], (inputs)
        # [1, -1, -1], [1, -1], [1, -1] (outputs)
        # begin_norm_axis=2
        self.attrs['begin_norm_axis'] = 2
        self.scale_spec.shape = [1024]
        self.bias_spec.shape = [1024]
        self.mean_spec.shape = self.x_spec.shape[
            : self.attrs['begin_norm_axis']
        ]
        self.var_spec.shape = self.x_spec.shape[: self.attrs['begin_norm_axis']]

        self.out_spec.set_dims_mapping([1, -1, -1])
        self.mean_spec.set_dims_mapping([1, -1])
        self.var_spec.set_dims_mapping([1, -1])

        result_dist_attrs = self.rule.infer_backward(
            self.x_spec,
            self.scale_spec,
            self.bias_spec,
            self.out_spec,
            self.mean_spec,
            self.var_spec,
            self.attrs['epsilon'],
            self.attrs['begin_norm_axis'],
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(inferred_input_dist_attrs), 3)
        self.assertEqual(len(inferred_output_dist_attrs), 3)

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [1, -1, -1])
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [-1])
        self.assertEqual(inferred_input_dist_attrs[2].dims_mapping, [-1])
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [1, -1, -1]
        )
        self.assertEqual(inferred_output_dist_attrs[1].dims_mapping, [1, -1])
        self.assertEqual(inferred_output_dist_attrs[2].dims_mapping, [1, -1])

        # [0, -1, -1], [0, -1], [0, -1] (outputs) -->
        # [0, -1, -1], [-1], [-1], (inputs)
        # [0, -1, -1], [0, -1], [0, -1] (outputs)
        # begin_norm_axis=2
        self.attrs['begin_norm_axis'] = 2
        self.scale_spec.shape = [
            np.prod(self.x_spec.shape[self.attrs['begin_norm_axis'] :])
        ]
        self.bias_spec.shape = [
            np.prod(self.x_spec.shape[self.attrs['begin_norm_axis'] :])
        ]
        self.mean_spec.shape = self.x_spec.shape[
            : self.attrs['begin_norm_axis']
        ]
        self.var_spec.shape = self.x_spec.shape[: self.attrs['begin_norm_axis']]

        self.out_spec.set_dims_mapping([0, -1, -1])
        self.mean_spec.set_dims_mapping([0, -1])
        self.var_spec.set_dims_mapping([0, -1])

        result_dist_attrs = self.rule.infer_backward(
            self.x_spec,
            self.scale_spec,
            self.bias_spec,
            self.out_spec,
            self.mean_spec,
            self.var_spec,
            self.attrs['epsilon'],
            self.attrs['begin_norm_axis'],
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(inferred_input_dist_attrs), 3)
        self.assertEqual(len(inferred_output_dist_attrs), 3)

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [0, -1, -1])
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [-1])
        self.assertEqual(inferred_input_dist_attrs[2].dims_mapping, [-1])
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [0, -1, -1]
        )
        self.assertEqual(inferred_output_dist_attrs[1].dims_mapping, [0, -1])
        self.assertEqual(inferred_output_dist_attrs[2].dims_mapping, [0, -1])

        # [-1, -1, -1], [0, -1], [-1, 1] (outputs) -->
        # [0, 1, -1], [-1], [-1], (inputs)
        # [0, 1, -1], [0, 1], [0, 1] (outputs)
        # begin_norm_axis=2
        self.attrs['begin_norm_axis'] = 2
        self.scale_spec.shape = [
            np.prod(self.x_spec.shape[self.attrs['begin_norm_axis'] :])
        ]
        self.bias_spec.shape = [
            np.prod(self.x_spec.shape[self.attrs['begin_norm_axis'] :])
        ]
        self.mean_spec.shape = self.x_spec.shape[
            : self.attrs['begin_norm_axis']
        ]
        self.var_spec.shape = self.x_spec.shape[: self.attrs['begin_norm_axis']]

        self.out_spec.set_dims_mapping([-1, -1, -1])
        self.mean_spec.set_dims_mapping([0, -1])
        self.var_spec.set_dims_mapping([-1, 1])

        result_dist_attrs = self.rule.infer_backward(
            self.x_spec,
            self.scale_spec,
            self.bias_spec,
            self.out_spec,
            self.mean_spec,
            self.var_spec,
            self.attrs['epsilon'],
            self.attrs['begin_norm_axis'],
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(inferred_input_dist_attrs), 3)
        self.assertEqual(len(inferred_output_dist_attrs), 3)

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [0, 1, -1])
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [-1])
        self.assertEqual(inferred_input_dist_attrs[2].dims_mapping, [-1])
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [0, 1, -1])
        self.assertEqual(inferred_output_dist_attrs[1].dims_mapping, [0, 1])
        self.assertEqual(inferred_output_dist_attrs[2].dims_mapping, [0, 1])

        # [-1, 1, -1], [-1, -1], [-1, -1] (outputs) -->
        # [-1, 1, -1], [-1], [-1], (inputs)
        # [-1, 1, -1], [-1, 1], [-1, 1] (outputs)
        # begin_norm_axis=2
        self.attrs['begin_norm_axis'] = 2
        self.scale_spec.shape = [
            np.prod(self.x_spec.shape[self.attrs['begin_norm_axis'] :])
        ]
        self.bias_spec.shape = [
            np.prod(self.x_spec.shape[self.attrs['begin_norm_axis'] :])
        ]
        self.mean_spec.shape = self.x_spec.shape[
            : self.attrs['begin_norm_axis']
        ]
        self.var_spec.shape = self.x_spec.shape[: self.attrs['begin_norm_axis']]

        self.out_spec.set_dims_mapping([-1, 1, -1])
        self.mean_spec.set_dims_mapping([-1, -1])
        self.var_spec.set_dims_mapping([-1, -1])

        result_dist_attrs = self.rule.infer_backward(
            self.x_spec,
            self.scale_spec,
            self.bias_spec,
            self.out_spec,
            self.mean_spec,
            self.var_spec,
            self.attrs['epsilon'],
            self.attrs['begin_norm_axis'],
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(inferred_input_dist_attrs), 3)
        self.assertEqual(len(inferred_output_dist_attrs), 3)

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [-1, 1, -1])
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [-1])
        self.assertEqual(inferred_input_dist_attrs[2].dims_mapping, [-1])
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [-1, 1, -1]
        )
        self.assertEqual(inferred_output_dist_attrs[1].dims_mapping, [-1, 1])
        self.assertEqual(inferred_output_dist_attrs[2].dims_mapping, [-1, 1])

        # [1, -1, -1], [0, -1], [-1, -1] (outputs) --> error
        # begin_norm_axis=2
        self.attrs['begin_norm_axis'] = 2
        self.scale_spec.shape = [
            np.prod(self.x_spec.shape[self.attrs['begin_norm_axis'] :])
        ]
        self.bias_spec.shape = [
            np.prod(self.x_spec.shape[self.attrs['begin_norm_axis'] :])
        ]
        self.mean_spec.shape = self.x_spec.shape[
            : self.attrs['begin_norm_axis']
        ]
        self.var_spec.shape = self.x_spec.shape[: self.attrs['begin_norm_axis']]

        self.out_spec.set_dims_mapping([1, -1, -1])
        self.mean_spec.set_dims_mapping([0, -1])
        self.var_spec.set_dims_mapping([-1, -1])

        with self.assertRaises(NotImplementedError):
            result_dist_attrs = self.rule.infer_backward(
                self.x_spec,
                self.scale_spec,
                self.bias_spec,
                self.out_spec,
                self.mean_spec,
                self.var_spec,
                self.attrs['epsilon'],
                self.attrs['begin_norm_axis'],
            )

        # [-1, 1, -1], [0, -1], [-1, -1] (outputs) -->
        # [0, 1, -1], [-1], [-1] (inputs)
        # [0, 1, -1], [0, 1], [0, 1] (outputs)
        # begin_norm_axis=2
        self.attrs['begin_norm_axis'] = 2
        self.scale_spec.shape = [
            np.prod(self.x_spec.shape[self.attrs['begin_norm_axis'] :])
        ]
        self.bias_spec.shape = [
            np.prod(self.x_spec.shape[self.attrs['begin_norm_axis'] :])
        ]
        self.mean_spec.shape = self.x_spec.shape[
            : self.attrs['begin_norm_axis']
        ]
        self.var_spec.shape = self.x_spec.shape[: self.attrs['begin_norm_axis']]

        self.out_spec.set_dims_mapping([-1, 1, -1])
        self.mean_spec.set_dims_mapping([0, -1])
        self.var_spec.set_dims_mapping([-1, -1])

        result_dist_attrs = self.rule.infer_backward(
            self.x_spec,
            self.scale_spec,
            self.bias_spec,
            self.out_spec,
            self.mean_spec,
            self.var_spec,
            self.attrs['epsilon'],
            self.attrs['begin_norm_axis'],
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(inferred_input_dist_attrs), 3)
        self.assertEqual(len(inferred_output_dist_attrs), 3)

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [0, 1, -1])
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [-1])
        self.assertEqual(inferred_input_dist_attrs[2].dims_mapping, [-1])
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [0, 1, -1])
        self.assertEqual(inferred_output_dist_attrs[1].dims_mapping, [0, 1])
        self.assertEqual(inferred_output_dist_attrs[2].dims_mapping, [0, 1])

        # [0, 1, -1], [-1, -1], [-1, -1] (outputs) -->
        # [0, 1, -1], [-1], [-1] (inputs)
        # [0, 1, -1], [0, 1], [0, 1] (outputs)
        # begin_norm_axis=2
        self.attrs['begin_norm_axis'] = 2
        self.scale_spec.shape = [
            np.prod(self.x_spec.shape[self.attrs['begin_norm_axis'] :])
        ]
        self.bias_spec.shape = [
            np.prod(self.x_spec.shape[self.attrs['begin_norm_axis'] :])
        ]
        self.mean_spec.shape = self.x_spec.shape[
            : self.attrs['begin_norm_axis']
        ]
        self.var_spec.shape = self.x_spec.shape[: self.attrs['begin_norm_axis']]

        self.out_spec.set_dims_mapping([0, 1, -1])
        self.mean_spec.set_dims_mapping([-1, -1])
        self.var_spec.set_dims_mapping([-1, -1])

        result_dist_attrs = self.rule.infer_backward(
            self.x_spec,
            self.scale_spec,
            self.bias_spec,
            self.out_spec,
            self.mean_spec,
            self.var_spec,
            self.attrs['epsilon'],
            self.attrs['begin_norm_axis'],
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(inferred_input_dist_attrs), 3)
        self.assertEqual(len(inferred_output_dist_attrs), 3)

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [0, 1, -1])
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [-1])
        self.assertEqual(inferred_input_dist_attrs[2].dims_mapping, [-1])
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [0, 1, -1])
        self.assertEqual(inferred_output_dist_attrs[1].dims_mapping, [0, 1])
        self.assertEqual(inferred_output_dist_attrs[2].dims_mapping, [0, 1])

        # [0, -1, -1], [-1, 1], [-1, -1] (outputs) -->
        # [0, 1, -1], [-1], [-1], (inputs)
        # [0, 1, -1], [0, 1], [0, 1] (outputs)
        # begin_norm_axis=1
        self.attrs['begin_norm_axis'] = 2
        self.scale_spec.shape = [
            np.prod(self.x_spec.shape[self.attrs['begin_norm_axis'] :])
        ]
        self.bias_spec.shape = [
            np.prod(self.x_spec.shape[self.attrs['begin_norm_axis'] :])
        ]
        self.mean_spec.shape = self.x_spec.shape[
            : self.attrs['begin_norm_axis']
        ]
        self.var_spec.shape = self.x_spec.shape[: self.attrs['begin_norm_axis']]

        self.out_spec.set_dims_mapping([0, -1, -1])
        self.mean_spec.set_dims_mapping([-1, 1])
        self.var_spec.set_dims_mapping([-1, -1])

        result_dist_attrs = self.rule.infer_backward(
            self.x_spec,
            self.scale_spec,
            self.bias_spec,
            self.out_spec,
            self.mean_spec,
            self.var_spec,
            self.attrs['epsilon'],
            self.attrs['begin_norm_axis'],
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(inferred_input_dist_attrs), 3)
        self.assertEqual(len(inferred_output_dist_attrs), 3)

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [0, 1, -1])
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [-1])
        self.assertEqual(inferred_input_dist_attrs[2].dims_mapping, [-1])
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [0, 1, -1])
        self.assertEqual(inferred_output_dist_attrs[1].dims_mapping, [0, 1])
        self.assertEqual(inferred_output_dist_attrs[2].dims_mapping, [0, 1])


if __name__ == "__main__":
    unittest.main()
