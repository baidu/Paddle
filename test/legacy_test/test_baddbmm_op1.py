#   Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
from op_test import OpTest, convert_float_to_uint16

import paddle
from paddle import base
from paddle.base import Program, core, program_guard


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_float16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA or not support float16",
)
class TestBaddBmmFP16Op(OpTest):
    def setUp(self):
        self.op_type = "baddbmm"
        self.python_api = paddle.baddbmm
        self.init_dtype_type()
        self.inputs = {
            'Input': np.random.random((2, 10, 5)).astype(self.dtype),
            'X': np.random.random((2, 10, 10)).astype(self.dtype),
            'Y': np.random.random((2, 10, 5)).astype(self.dtype),
        }
        self.outputs = {
            'Out': self.inputs['Input']
            + np.matmul(self.inputs['X'], self.inputs['Y'])
        }

        self.place = core.CUDAPlace(0)

    def init_dtype_type(self):
        self.dtype = np.float16

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad_normal(self):
        self.check_grad_with_place(self.place, ['Input', 'X', 'Y'], 'Out')

    def test_check_grad_x(self):
        self.check_grad_with_place(self.place, ['X'], 'Out', no_grad_set=None)

    def test_check_grad_y(self):
        self.check_grad_with_place(self.place, ['Y'], 'Out', no_grad_set=None)

    def test_check_grad_input(self):
        self.check_grad_with_place(
            self.place, ['Input'], 'Out', no_grad_set=None
        )


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA or not support bfloat16",
)
class TestBaddBmmBF16Op(OpTest):
    def setUp(self):
        self.op_type = "baddbmm"
        self.python_api = paddle.baddbmm
        self.init_dtype_type()
        self.inputs = {
            'Input': np.random.random((2, 50, 1)).astype(self.dtype),
            'X': np.random.random((2, 50, 5)).astype(self.dtype),
            'Y': np.random.random((2, 5, 10)).astype(self.dtype),
        }
        self.outputs = {
            'Out': self.inputs['Input']
            + np.matmul(self.inputs['X'], self.inputs['Y'])
        }

        self.inputs['Input'] = convert_float_to_uint16(self.inputs['Input'])
        self.inputs['X'] = convert_float_to_uint16(self.inputs['X'])
        self.inputs['Y'] = convert_float_to_uint16(self.inputs['Y'])
        self.outputs['Out'] = convert_float_to_uint16(self.outputs['Out'])
        self.place = core.CUDAPlace(0)

    def init_dtype_type(self):
        self.dtype = np.uint16
        self.np_dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad_normal(self):
        self.check_grad_with_place(self.place, ['Input', 'X', 'Y'], 'Out')

    def test_check_grad_x(self):
        self.check_grad_with_place(self.place, ['X'], 'Out', no_grad_set=None)

    def test_check_grad_y(self):
        self.check_grad_with_place(self.place, ['Y'], 'Out', no_grad_set=None)

    def test_check_grad_input(self):
        self.check_grad_with_place(
            self.place, ['Input'], 'Out', no_grad_set=None
        )


class TestBaddBmmOpError(unittest.TestCase):
    # test error
    def test_errors(self):
        with program_guard(Program(), Program()):
            # The input type of baddbmm_op must be Variable.

            input = base.create_lod_tensor(
                np.array([[[-1, -1], [-1, -1]], [[-1, -1], [-1, -1]]]),
                [[2]],
                base.CPUPlace(),
            )
            x1 = base.create_lod_tensor(
                np.array([[[-1, -1], [-1, -1]], [[-1, -1], [-1, -1]]]),
                [[2]],
                base.CPUPlace(),
            )
            x2 = base.create_lod_tensor(
                np.array([[[-1, -1], [-1, -1]], [[-1, -1], [-1, -1]]]),
                [[2]],
                base.CPUPlace(),
            )
            self.assertRaises(TypeError, paddle.baddbmm, input, x1, x2)

            paddle.enable_static()
            # The input dtype of baddbmm_op must be float32 or float64.
            input = paddle.static.data(
                name='input',
                shape=[2, 4, 4],
                dtype="int32",
            )
            x3 = paddle.static.data(name='x3', shape=[2, 4, 4], dtype="int32")
            x4 = paddle.static.data(name='x4', shape=[2, 4, 4], dtype="int32")
            self.assertRaises(TypeError, paddle.baddbmm, input, x3, x4)
            # x and y dimension mismatch
            x5 = paddle.static.data(
                name='x5',
                shape=[2, 4, 5],
                dtype="float32",
            )
            x6 = paddle.static.data(
                name='x6',
                shape=[2, 4, 4],
                dtype="float32",
            )
            self.assertRaises(ValueError, paddle.baddbmm, input, x5, x6)
            # input and x are not broadcastable
            x7 = paddle.static.data(
                name='x7',
                shape=[2, 4, 4],
                dtype="float32",
            )
            x8 = paddle.static.data(
                name='x8',
                shape=[2, 4, 4],
                dtype="float32",
            )
            input1 = paddle.static.data(
                name='input1',
                shape=[2, 2, 4],
                dtype="float32",
            )
            self.assertRaises(ValueError, paddle.baddbmm, input1, x7, x8)
            # input and x are not broadcastable
            x9 = paddle.static.data(
                name='x9',
                shape=[2, 4, 4],
                dtype="float32",
            )
            x10 = paddle.static.data(
                name='x10',
                shape=[2, 4, 4],
                dtype="float32",
            )
            input2 = paddle.static.data(
                name='input2',
                shape=[2, 1, 2],
                dtype="float32",
            )
            self.assertRaises(ValueError, paddle.baddbmm, input2, x9, x10)
            x11 = paddle.static.data(
                name='x11',
                shape=[2, 4, 4],
                dtype="float32",
            )
            x12 = paddle.static.data(
                name='x12', shape=[2, 4, 4], dtype="float32"
            )
            input3 = paddle.static.data(
                name='input3',
                shape=[2, 4, 2],
                dtype="float32",
            )
            self.assertRaises(ValueError, paddle.baddbmm, input3, x11, x12)
            x13 = paddle.static.data(
                name='x13',
                shape=[2, 4, 4],
                dtype="float32",
            )
            x14 = paddle.static.data(
                name='x14',
                shape=[2, 4, 4],
                dtype="float32",
            )
            input4 = paddle.static.data(
                name='input4',
                shape=[2, 3, 1],
                dtype="float32",
            )
            self.assertRaises(ValueError, paddle.baddbmm, input4, x13, x14)


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
