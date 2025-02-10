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


class TestBaddBmmOp3(OpTest):
    def setUp(self):
        self.op_type = "baddbmm"
        self.prim_op_type = "comp"
        self.python_api = paddle.baddbmm
        self.public_python_api = paddle.baddbmm
        self.dtype = np.float64
        self.init_dtype_type()
        self.inputs = {
            'Input': np.random.random((2, 10, 5)).astype(self.dtype),
            'X': np.random.random((2, 10, 10)).astype(self.dtype),
            'Y': np.random.random((2, 10, 5)).astype(self.dtype),
        }
        self.attrs = {
            'Alpha': 0.5,
            'Beta': 2.0,
        }
        self.outputs = {
            'Out': self.attrs['Beta'] * self.inputs['Input']
            + self.attrs['Alpha']
            * np.matmul(self.inputs['X'], self.inputs['Y'])
        }

    def init_dtype_type(self):
        pass

    def test_check_output(self):
        self.check_output(check_pir=True, check_prim_pir=True)

    def test_check_grad_normal(self):
        self.check_grad(
            ['Input', 'X', 'Y'], 'Out', check_pir=True, check_prim_pir=True
        )

    def test_check_grad_x(self):
        self.check_grad(
            ['X'], 'Out', no_grad_set=None, check_pir=True, check_prim_pir=True
        )

    def test_check_grad_y(self):
        self.check_grad(
            ['Y'], 'Out', no_grad_set=None, check_pir=True, check_prim_pir=True
        )

    def test_check_grad_input(self):
        self.check_grad(
            ['Input'],
            'Out',
            no_grad_set=None,
            check_pir=True,
            check_prim_pir=True,
        )


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
