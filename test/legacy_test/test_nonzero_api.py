# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.base import Program, program_guard


def call_nonzero(x):
    input = paddle.to_tensor(x)
    return paddle.nonzero(x=input)


class TestNonZeroAPI(unittest.TestCase):
    def test_nonzero_api_as_tuple(self):
        paddle.enable_static()
        data = np.array([[1, 0], [0, 1]], dtype='float32')
        with program_guard(Program(), Program()):
            x = paddle.static.data(name='x', shape=[-1, 2], dtype='float32')
            if not paddle.framework.use_pir_api():
                x.desc.set_need_check_feed(False)
            y = paddle.nonzero(x, as_tuple=True)
            self.assertEqual(type(y), tuple)
            self.assertEqual(len(y), 2)
            index_list = [index.reshape([-1, 1]) for index in y]
            z = paddle.concat(index_list, axis=1)
            exe = base.Executor(base.CPUPlace())

            (res,) = exe.run(
                feed={'x': data}, fetch_list=[z], return_numpy=False
            )
        expect_out = np.array([[0, 0], [1, 1]])
        np.testing.assert_allclose(expect_out, np.array(res), rtol=1e-05)

        data = np.array([1, 1, 0], dtype="float32")
        with program_guard(Program(), Program()):
            x = paddle.static.data(name='x', shape=[-1], dtype='float32')
            if not paddle.framework.use_pir_api():
                x.desc.set_need_check_feed(False)
            y = paddle.nonzero(x, as_tuple=True)
            self.assertEqual(type(y), tuple)
            self.assertEqual(len(y), 1)
            index_list = [index.reshape([-1, 1]) for index in y]
            z = paddle.concat(index_list, axis=1)
            exe = base.Executor(base.CPUPlace())
            (res,) = exe.run(
                feed={'x': data}, fetch_list=[z], return_numpy=False
            )
        expect_out = np.array([[0], [1]])
        np.testing.assert_allclose(expect_out, np.array(res), rtol=1e-05)

    def test_nonzero_api(self):
        paddle.enable_static()
        data = np.array([[1, 0], [0, 1]], dtype="float32")
        with program_guard(Program(), Program()):
            x = paddle.static.data(name='x', shape=[-1, 2], dtype='float32')
            if not paddle.framework.use_pir_api():
                x.desc.set_need_check_feed(False)
            y = paddle.nonzero(x)
            exe = base.Executor(base.CPUPlace())
            (res,) = exe.run(
                feed={'x': data}, fetch_list=[y], return_numpy=False
            )
        expect_out = np.array([[0, 0], [1, 1]])
        np.testing.assert_allclose(expect_out, np.array(res), rtol=1e-05)

        data = np.array([1, 1, 0], dtype="float32")
        with program_guard(Program(), Program()):
            x = paddle.static.data(name='x', shape=[-1], dtype='float32')
            if not paddle.framework.use_pir_api():
                x.desc.set_need_check_feed(False)
            y = paddle.nonzero(x)
            exe = base.Executor(base.CPUPlace())
            (res,) = exe.run(
                feed={'x': data}, fetch_list=[y], return_numpy=False
            )
        expect_out = np.array([[0], [1]])
        np.testing.assert_allclose(expect_out, np.array(res), rtol=1e-05)

    def test_dygraph_api(self):
        data_x = np.array([[True, False], [False, True]])
        with base.dygraph.guard():
            x = paddle.to_tensor(data_x)
            z = paddle.nonzero(x)
            np_z = z.numpy()
        expect_out = np.array([[0, 0], [1, 1]])


# Base case
class TestNonzeroOp(OpTest):
    def setUp(self):
        '''Test where_index op with random value'''
        np.random.seed(2023)
        self.op_type = "where_index"
        self.python_api = call_nonzero
        self.init_shape()
        self.init_dtype()

        self.inputs = self.create_inputs()
        self.outputs = self.return_outputs()

    def test_check_output(self):
        self.check_output(check_pir=True, check_symbol_infer=False)

    def init_shape(self):
        self.shape = [8, 8]

    def init_dtype(self):
        self.dtype = np.float64

    def create_inputs(self):
        return {
            'Condition': np.random.randint(5, size=self.shape).astype(
                self.dtype
            )
        }

    def return_outputs(self):
        return {'Out': np.transpose(np.nonzero(self.inputs['Condition']))}


class TestNonzeroFP32Op(TestNonzeroOp):
    def init_shape(self):
        self.shape = [2, 10, 2]

    def init_dtype(self):
        self.dtype = np.float32


class TestNonzeroFP16Op(TestNonzeroOp):
    def init_shape(self):
        self.shape = [3, 4, 7]

    def init_dtype(self):
        self.dtype = np.float16


class TestNonzeroBF16(OpTest):
    def setUp(self):
        '''Test where_index op with bfloat16 dtype'''
        np.random.seed(2023)
        self.op_type = "where_index"
        self.python_api = call_nonzero
        self.init_shape()
        self.init_dtype()

        self.inputs = self.create_inputs()
        self.outputs = self.return_outputs()

    def test_check_output(self):
        self.check_output(check_pir=True, check_symbol_infer=False)

    def init_shape(self):
        self.shape = [12, 9]

    def init_dtype(self):
        self.dtype = np.uint16

    def create_inputs(self):
        return {
            'Condition': convert_float_to_uint16(
                np.random.randint(5, size=self.shape).astype(np.float32)
            )
        }

    def return_outputs(self):
        return {'Out': np.transpose(np.nonzero(self.inputs['Condition']))}


if __name__ == "__main__":
    unittest.main()
