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
from op_test import OpTest

import paddle


class TestBaddBmm1Op(OpTest):
    # test basic
    def setUp(self):
        self.op_type = "baddbmm"
        self.prim_op_type = "comp"
        self.python_api = paddle.baddbmm
        self.public_python_api = paddle.baddbmm
        self.init_dtype_type()
        self.inputs = {
            'Input': np.random.random((1, 10, 10)).astype(self.dtype),
            'X': np.random.random((1, 10, 10)).astype(self.dtype),
            'Y': np.random.random((1, 10, 10)).astype(self.dtype),
        }
        self.outputs = {
            'Out': self.inputs['Input']
            + np.matmul(self.inputs['X'], self.inputs['Y'])
        }

    def init_dtype_type(self):
        self.dtype = np.float64

    def test_check_output(self):
        self.check_output(check_pir=True, check_prim_pir=True)

    def test_check_grad_normal(self):
        self.check_grad(
            ['Input', 'X', 'Y'],
            'Out',
            check_pir=True,
            check_prim_pir=True,
        )

    def test_check_grad_x(self):
        self.check_grad(
            ['X'],
            'Out',
            no_grad_set=None,
            check_pir=True,
            check_prim_pir=True,
        )

    def test_check_grad_y(self):
        self.check_grad(
            ['Y'],
            'Out',
            no_grad_set=None,
            check_pir=True,
            check_prim_pir=True,
        )

    def test_check_grad_input(self):
        self.check_grad(
            ['Input'],
            'Out',
            no_grad_set=None,
            check_pir=True,
            check_prim_pir=True,
        )


class TestBaddBmm1FP16Op(TestBaddBmm1Op):
    def init_dtype_type(self):
        self.dtype = np.float16

    def test_check_output(self):
        self.check_output(atol=1e-2)


class TestBaddBmm1Op2(TestBaddBmm1Op):
    # test alpha and beta
    def setUp(self):
        self.op_type = "baddbmm"
        self.prim_op_type = "comp"
        self.python_api = paddle.baddbmm
        self.public_python_api = paddle.baddbmm
        self.dtype = np.float64
        self.init_dtype_type()
        self.inputs = {
            'Input': np.random.random((1, 10, 10)).astype(self.dtype),
            'X': np.random.random((1, 10, 10)).astype(self.dtype),
            'Y': np.random.random((1, 10, 10)).astype(self.dtype),
        }
        self.attrs = {
            'Alpha': 0.1,
            'Beta': 1.0,
        }
        self.outputs = {
            'Out': self.attrs['Beta'] * self.inputs['Input']
            + self.attrs['Alpha']
            * np.matmul(self.inputs['X'], self.inputs['Y'])
        }


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
