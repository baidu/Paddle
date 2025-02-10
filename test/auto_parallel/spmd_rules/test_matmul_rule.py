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


class TestMatmulSPMDRule(unittest.TestCase):
    def setUp(self):
        # After replaced all spmd rules by phi impl, we can recover the
        # api name to `get_spmd_rule`
        self.rule = core.get_phi_spmd_rule("matmul")

        self.attrs = OrderedDict([('trans_x', False), ('trans_y', False)])

    def test_matmul_infer_forward(self):
        # forward setup
        x_shape = [64, 32]
        y_shape = [32, 48]
        process_mesh = auto.ProcessMesh(mesh=[[0, 1, 2], [3, 4, 5]])

        x_tensor_dist_attr = TensorDistAttr()
        x_tensor_dist_attr.dims_mapping = [1, 0]
        x_tensor_dist_attr.process_mesh = process_mesh
        self.x_dist_tensor_spec = DistTensorSpec(x_shape, x_tensor_dist_attr)

        y_tensor_dist_attr = TensorDistAttr()
        y_tensor_dist_attr.dims_mapping = [0, -1]
        y_tensor_dist_attr.process_mesh = process_mesh
        self.y_dist_tensor_spec = DistTensorSpec(y_shape, y_tensor_dist_attr)

        # TODO test partial: mk[1, 0],kn[0, -1] --> mk[1, 0],kn[0, -1] = nm[1, -1] partial[0]
        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec, self.y_dist_tensor_spec, False, False
        )

        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(inferred_input_dist_attrs), 2)
        self.assertEqual(len(inferred_output_dist_attrs), 1)

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [1, 0])
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [0, -1])
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [1, -1])
        self.assertEqual(inferred_output_dist_attrs[0]._is_partial(), True)
        self.assertEqual(inferred_output_dist_attrs[0]._partial_dims(), {0})

        # test row parallel: mk[1, -1],kn[-1, -1] --> mk[1, -1],kn[-1, -1] = nm[1, -1] partial[]
        self.x_dist_tensor_spec.set_dims_mapping([1, -1])
        self.y_dist_tensor_spec.set_dims_mapping([-1, -1])

        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec, self.y_dist_tensor_spec, False, False
        )

        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [1, -1])
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [-1, -1])
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [1, -1])
        self.assertEqual(inferred_output_dist_attrs[0]._is_partial(), False)

        # test row parallel: mk[1, -1],kn[-1, -1] --> mk[1, -1],kn[-1, -1] = nm[1, -1] partial[]
        self.x_dist_tensor_spec.set_dims_mapping([1, -1])
        self.y_dist_tensor_spec.set_dims_mapping([-1, -1])

        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec, self.y_dist_tensor_spec, False, False
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [1, -1])
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [-1, -1])
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [1, -1])
        self.assertEqual(inferred_output_dist_attrs[0]._is_partial(), False)

        # test n parallel: mk[-1, -1],kn[-1, 0] --> mk[-1, -1],kn[-1, 0] = nm[-1, 0] partial[]
        self.x_dist_tensor_spec.set_dims_mapping([-1, -1])
        self.y_dist_tensor_spec.set_dims_mapping([-1, 0])

        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec, self.y_dist_tensor_spec, False, False
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [-1, -1])
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [-1, 0])
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [-1, 0])
        self.assertEqual(inferred_output_dist_attrs[0]._is_partial(), False)

        # test partial with propagation: mk[1, 0],kn[-1,-1] --> mk[1, 0],kn[0, -1] = nm[1, -1] partial[0]
        self.x_dist_tensor_spec.set_dims_mapping([1, 0])
        self.y_dist_tensor_spec.set_dims_mapping([-1, -1])

        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec, self.y_dist_tensor_spec, False, False
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [1, 0])
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [0, -1])
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [1, -1])
        self.assertEqual(inferred_output_dist_attrs[0]._is_partial(), True)
        self.assertEqual(inferred_output_dist_attrs[0]._partial_dims(), {0})

        # mk[-1,-1],kn[1,0] --> mk[-1, 1],kn[1, 0] = nm[-1, 0] partial[1]:
        self.x_dist_tensor_spec.set_dims_mapping([-1, -1])
        self.y_dist_tensor_spec.set_dims_mapping([1, 0])

        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec, self.y_dist_tensor_spec, False, False
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [-1, 1])
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [1, 0])
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [-1, 0])
        self.assertEqual(inferred_output_dist_attrs[0]._is_partial(), True)
        self.assertEqual(inferred_output_dist_attrs[0]._partial_dims(), {1})

        # abcmk[1, 0, -1, -1],kn[-1, -1] --> abcmk[1, 0, -1, -1],kn[-1, -1] = abcmn[1, 0, -1, -1] partial[]: done
        self.x_dist_tensor_spec.shape = [512, 48, 64, 32]
        self.x_dist_tensor_spec.set_dims_mapping([0, 1, -1, -1])
        self.y_dist_tensor_spec.set_dims_mapping([-1, -1])

        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec, self.y_dist_tensor_spec, False, False
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(
            inferred_input_dist_attrs[0].dims_mapping, [0, 1, -1, -1]
        )
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [-1, -1])
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [0, 1, -1, -1]
        )
        self.assertEqual(inferred_output_dist_attrs[0]._is_partial(), False)

        # abcmk[1, -1, -1, 0],kn[-1, -1] --> abcmk[1, -1, -1, 0],kn[0, -1] = abcmn[1,-1, -1, -1] partial[0]
        self.x_dist_tensor_spec.set_dims_mapping([1, -1, -1, 0])
        self.y_dist_tensor_spec.set_dims_mapping([-1, -1])

        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec, self.y_dist_tensor_spec, False, False
        )

        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(
            inferred_input_dist_attrs[0].dims_mapping, [1, -1, -1, 0]
        )
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [0, -1])
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [1, -1, -1, -1]
        )
        self.assertEqual(inferred_output_dist_attrs[0]._is_partial(), True)
        self.assertEqual(inferred_output_dist_attrs[0]._partial_dims(), {0})

        # trans_x = True, abcmk[1, -1, -1, 0], kn[-1, -1] --> abcmk[1, -1, -1, 0],kn[-1, -1] = abcmn[1, -1, 0, -1] partial[]
        self.x_dist_tensor_spec.set_dims_mapping([1, -1, -1, 0])
        self.y_dist_tensor_spec.set_dims_mapping([-1, -1])

        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec, self.y_dist_tensor_spec, True, False
        )

        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(
            inferred_input_dist_attrs[0].dims_mapping, [1, -1, -1, 0]
        )
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [-1, -1])
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [1, -1, 0, -1]
        )
        self.assertEqual(inferred_output_dist_attrs[0]._is_partial(), False)

        # trans_y = True, abcmk[-1, -1, -1, -1], kn[1, 0] --> abcmk[-1, -1, -1, 0],kn[1, 0] = abcmn[-1, -1, -1, 1] partial[0]: done
        self.x_dist_tensor_spec.set_dims_mapping([-1, -1, -1, -1])
        self.y_dist_tensor_spec.set_dims_mapping([1, 0])

        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec, self.y_dist_tensor_spec, False, True
        )

        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(
            inferred_input_dist_attrs[0].dims_mapping, [-1, -1, -1, 0]
        )
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [1, 0])
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [-1, -1, -1, 1]
        )
        self.assertEqual(inferred_output_dist_attrs[0]._is_partial(), True)
        self.assertEqual(inferred_output_dist_attrs[0]._partial_dims(), {0})
        inferred_output_dist_attrs[0]._clean_partial_dims([0])
        self.assertEqual(inferred_output_dist_attrs[0]._is_partial(), False)

        # trans_y = True, trans_x = True, abcmk[-1, -1, 0, 1], kn[1, 0] --> abcmk[-1, -1, 0, 1]],kn[-1, 0] = abcmn[-1, -1, 1, -1] partial[0]
        # multiple mesh dim shard same tensor axis
        self.x_dist_tensor_spec.set_dims_mapping([-1, -1, 0, 1])
        self.y_dist_tensor_spec.set_dims_mapping([1, 0])

        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec, self.y_dist_tensor_spec, True, True
        )

        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(
            inferred_input_dist_attrs[0].dims_mapping, [-1, -1, 0, 1]
        )
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [-1, 0])
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [-1, -1, 1, -1]
        )
        self.assertEqual(inferred_output_dist_attrs[0]._is_partial(), True)
        self.assertEqual(inferred_output_dist_attrs[0]._partial_dims(), {0})
        inferred_output_dist_attrs[0]._clean_partial_status()
        self.assertEqual(inferred_output_dist_attrs[0]._is_partial(), False)

        # trans_y = True, trans_x = True, abcmk[-1, -1, 1, 0], kn[1, 0] --> error:
        # one tensor axis shard multiple mesh dim
        self.x_dist_tensor_spec.set_dims_mapping([-1, -1, 1, -1])
        self.y_dist_tensor_spec.set_dims_mapping([-1, 0])
        self.attrs['trans_x'] = True
        self.attrs['trans_y'] = True
        with self.assertRaises(NotImplementedError):
            result_dist_attrs = self.rule.infer_forward(
                self.x_dist_tensor_spec,
                self.y_dist_tensor_spec,
                self.attrs['trans_x'],
                self.attrs['trans_y'],
            )

    def test_matmul_infer_backward(self):
        # backward setup
        x_shape = [64, 32]
        y_shape = [32, 48]
        out_shape = [64, 48]
        process_mesh = auto.ProcessMesh(mesh=[[0, 1, 2], [3, 4, 5]])

        x_tensor_dist_attr = TensorDistAttr()
        x_tensor_dist_attr.dims_mapping = [-1, -1]
        x_tensor_dist_attr.process_mesh = process_mesh
        self.x_dist_tensor_spec = DistTensorSpec(x_shape, x_tensor_dist_attr)

        y_tensor_dist_attr = TensorDistAttr()
        y_tensor_dist_attr.dims_mapping = [-1, -1]
        y_tensor_dist_attr.process_mesh = process_mesh
        self.y_dist_tensor_spec = DistTensorSpec(y_shape, y_tensor_dist_attr)

        out_tensor_dist_attr = TensorDistAttr()
        out_tensor_dist_attr.dims_mapping = [1, 0]
        out_tensor_dist_attr.process_mesh = process_mesh
        self.out_dist_tensor_spec = DistTensorSpec(
            out_shape, out_tensor_dist_attr
        )

        # mn[1, 0] --> mk[1, -1],kn[-1, 0]
        result_dist_attrs = self.rule.infer_backward(
            self.x_dist_tensor_spec,
            self.y_dist_tensor_spec,
            self.out_dist_tensor_spec,
            self.attrs['trans_x'],
            self.attrs['trans_y'],
        )

        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(inferred_input_dist_attrs), 2)
        self.assertEqual(len(inferred_output_dist_attrs), 1)

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [1, -1])
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [-1, 0])
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [1, 0])
        self.assertEqual(inferred_input_dist_attrs[0]._is_partial(), False)
        self.assertEqual(inferred_input_dist_attrs[1]._is_partial(), False)
        self.assertEqual(inferred_output_dist_attrs[0]._is_partial(), False)

        # test on broadcast axes propagation
        # abmn[1, 0, -1, -1] --> 1mk[-1, -1, -1], abkn[1, 0, -1, -1]
        self.out_dist_tensor_spec.shape = [512, 48, 64, 48]
        self.x_dist_tensor_spec.shape = [1, 64, 32]
        self.y_dist_tensor_spec.shape = [512, 48, 32, 48]
        self.x_dist_tensor_spec.set_dims_mapping(
            [0, -1, 1]
        )  # dims mapping of input should not influence inferbackward
        self.y_dist_tensor_spec.set_dims_mapping(
            [
                -1,
                -1,
                1,
                0,
            ]
        )  # dims mapping of input should not influence inferbackward
        self.out_dist_tensor_spec.set_dims_mapping([1, 0, -1, -1])
        result_dist_attrs = self.rule.infer_backward(
            self.x_dist_tensor_spec,
            self.y_dist_tensor_spec,
            self.out_dist_tensor_spec,
            self.attrs['trans_x'],
            self.attrs['trans_y'],
        )

        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(
            inferred_input_dist_attrs[0].dims_mapping, [-1, -1, -1]
        )
        self.assertEqual(
            inferred_input_dist_attrs[1].dims_mapping, [1, 0, -1, -1]
        )
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [1, 0, -1, -1]
        )

        # abmn[-1, 0, -1, 1] --> abmk[-1, 0, -1, -1], a1kn[-1, -1, -1, 1]
        self.out_dist_tensor_spec.shape = [512, 48, 64, 48]
        self.x_dist_tensor_spec.shape = [512, 48, 64, 32]
        self.y_dist_tensor_spec.shape = [512, 1, 32, 48]
        self.out_dist_tensor_spec.set_dims_mapping([-1, 0, -1, 1])

        result_dist_attrs = self.rule.infer_backward(
            self.x_dist_tensor_spec,
            self.y_dist_tensor_spec,
            self.out_dist_tensor_spec,
            self.attrs['trans_x'],
            self.attrs['trans_y'],
        )

        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(
            inferred_input_dist_attrs[0].dims_mapping, [-1, 0, -1, -1]
        )
        self.assertEqual(
            inferred_input_dist_attrs[1].dims_mapping, [-1, -1, -1, 1]
        )
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [-1, 0, -1, 1]
        )

        # trans_x = true, trans_y = true, abmn[-1, -1, 0, 1] --> abmk[-1, -1, -1, 0], a1kn[-1, -1, 1, -1]
        self.out_dist_tensor_spec.shape = [512, 48, 64, 48]
        self.x_dist_tensor_spec.shape = [512, 48, 32, 64]
        self.y_dist_tensor_spec.shape = [512, 1, 48, 32]
        self.out_dist_tensor_spec.set_dims_mapping([-1, -1, 0, 1])
        self.attrs['trans_x'] = True
        self.attrs['trans_y'] = True

        result_dist_attrs = self.rule.infer_backward(
            self.x_dist_tensor_spec,
            self.y_dist_tensor_spec,
            self.out_dist_tensor_spec,
            self.attrs['trans_x'],
            self.attrs['trans_y'],
        )

        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(
            inferred_input_dist_attrs[0].dims_mapping, [-1, -1, -1, 0]
        )
        self.assertEqual(
            inferred_input_dist_attrs[1].dims_mapping, [-1, -1, 1, -1]
        )
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [-1, -1, 0, 1]
        )

        # # trans_x = true, trans_y = true, abmn[-1, 1, 0, 1] --> error:
        # one mesh dim shard multiple tensor axes
        self.out_dist_tensor_spec.set_dims_mapping([-1, 1, 0, 1])
        with self.assertRaises(RuntimeError):
            result_dist_attrs = self.rule.infer_backward(
                self.x_dist_tensor_spec,
                self.y_dist_tensor_spec,
                self.out_dist_tensor_spec,
                self.attrs['trans_x'],
                self.attrs['trans_y'],
            )


if __name__ == "__main__":
    unittest.main()
