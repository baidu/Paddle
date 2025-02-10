#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import OpTest, convert_float_to_uint16, paddle_static_guard

import paddle
from paddle import base
from paddle.base import core


class TestBmmOp(OpTest):
    def setUp(self):
        self.op_type = "bmm"
        self.prim_op_type = "comp"
        self.python_api = paddle.tensor.bmm
        self.public_python_api = paddle.tensor.bmm
        X = np.random.random((10, 3, 4)).astype("float64")
        Y = np.random.random((10, 4, 5)).astype("float64")
        self.inputs = {'X': X, 'Y': Y}
        Out = np.matmul(X, Y)
        self.outputs = {'Out': Out}

    def test_check_output(self):
        self.check_output(check_pir=True, check_prim_pir=True)

    def test_checkout_grad(self):
        self.check_grad(['X', 'Y'], 'Out', check_pir=True)


class TestBmmFP16Op(OpTest):
    def setUp(self):
        self.op_type = "bmm"
        self.prim_op_type = "comp"
        self.dtype = np.float16
        self.python_api = paddle.tensor.bmm
        self.public_python_api = paddle.tensor.bmm
        X = np.random.random((10, 3, 4)).astype("float16")
        Y = np.random.random((10, 4, 5)).astype("float16")
        self.inputs = {'X': X, 'Y': Y}
        Out = np.matmul(X, Y)
        self.outputs = {'Out': Out}

    def test_check_output(self):
        self.check_output(check_pir=True, check_prim_pir=True)

    def test_checkout_grad(self):
        self.check_grad(['X', 'Y'], 'Out', check_pir=True)


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA or not support bfloat16",
)
class TestBmmBF16Op(OpTest):
    def setUp(self):
        self.op_type = "bmm"
        self.prim_op_type = "comp"
        self.dtype = np.uint16
        self.python_api = paddle.tensor.bmm
        self.public_python_api = paddle.tensor.bmm
        X = np.random.random((10, 3, 4)).astype("float32")
        Y = np.random.random((10, 4, 5)).astype("float32")
        self.inputs = {'X': X, 'Y': Y}
        Out = np.matmul(X, Y)
        self.outputs = {'Out': Out}

        self.inputs['X'] = convert_float_to_uint16(self.inputs['X'])
        self.inputs['Y'] = convert_float_to_uint16(self.inputs['Y'])
        self.outputs['Out'] = convert_float_to_uint16(self.outputs['Out'])
        self.place = core.CUDAPlace(0)

    def test_check_output(self):
        self.check_output_with_place(
            self.place, check_pir=True, check_prim_pir=True
        )

    def test_checkout_grad(self):
        self.check_grad_with_place(
            self.place, ['X', 'Y'], 'Out', check_pir=True
        )


class API_TestBmm(unittest.TestCase):

    def test_out(self):
        with paddle_static_guard():
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                data1 = paddle.static.data(
                    'data1', shape=[-1, 3, 4], dtype='float64'
                )
                data2 = paddle.static.data(
                    'data2', shape=[-1, 4, 5], dtype='float64'
                )
                result_bmm = paddle.bmm(data1, data2)
                place = base.CPUPlace()
                exe = base.Executor(place)
                input1 = np.random.random([10, 3, 4]).astype('float64')
                input2 = np.random.random([10, 4, 5]).astype('float64')
                (result,) = exe.run(
                    feed={"data1": input1, "data2": input2},
                    fetch_list=[result_bmm],
                )
                expected_result = np.matmul(input1, input2)
            np.testing.assert_allclose(expected_result, result, rtol=1e-05)


class API_TestDygraphBmm(unittest.TestCase):
    def test_out(self):
        input1 = np.array(
            [
                [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
                [[3.0, 3.0, 3.0], [4.0, 4.0, 4.0]],
            ]
        )
        input2 = np.array(
            [
                [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
                [[4.0, 4.0], [5.0, 5.0], [6.0, 6.0]],
            ]
        )
        with base.dygraph.guard():
            x = paddle.to_tensor(input1)
            y = paddle.to_tensor(input2)
            out = paddle.bmm(x, y)
            out_np = out.numpy()
        expected_result = np.matmul(input1, input2)
        np.testing.assert_allclose(expected_result, out_np, rtol=1e-05)


class TestBmmAPIError(unittest.TestCase):
    def test_api_error(self):
        x_data = np.arange(24, dtype='float32').reshape((2, 3, 4))
        y_data = np.arange(16, dtype='float32').reshape((2, 4, 2))
        y_data_wrong1 = np.arange(16, dtype='float32').reshape((2, 2, 4))
        y_data_wrong2 = np.arange(16, dtype='float32').reshape((2, 2, 2, 2))
        y_data_wrong3 = np.arange(24, dtype='float32').reshape((3, 4, 2))
        self.assertRaises(ValueError, paddle.bmm, x_data, y_data_wrong1)
        self.assertRaises(ValueError, paddle.bmm, x_data, y_data_wrong2)
        self.assertRaises(ValueError, paddle.bmm, x_data, y_data_wrong3)


class TestBmmOpCaseComplex64(TestBmmOp):
    def setUp(self):
        self.op_type = "bmm"
        self.python_api = paddle.tensor.bmm
        self.public_python_api = paddle.tensor.bmm
        self.dtype = np.complex64
        X = (
            np.random.uniform(1, 5, (10, 3, 4))
            + 1j * np.random.uniform(1, 5, (10, 3, 4))
        ).astype("complex64")
        Y = (
            np.random.uniform(1, 5, (10, 4, 2))
            + 1j * np.random.uniform(1, 5, (10, 4, 2))
        ).astype("complex64")
        self.inputs = {'X': X, 'Y': Y}
        Out = np.matmul(X, Y)
        self.outputs = {'Out': Out}

    def test_check_output(self):
        self.check_output(check_pir=True, check_prim=False)
        # pass

    def test_checkout_grad(self):
        self.check_grad(['X', 'Y'], 'Out', check_pir=True, check_prim=False)


class TestBmmOpCaseComplex128(TestBmmOp):
    def setUp(self):
        self.op_type = "bmm"
        self.python_api = paddle.tensor.bmm
        self.public_python_api = paddle.tensor.bmm
        self.dtype = np.complex128
        X = (
            np.random.uniform(1, 5, (10, 3, 4))
            + 1j * np.random.uniform(1, 5, (10, 3, 4))
        ).astype("complex128")
        Y = (
            np.random.uniform(1, 5, (10, 4, 2))
            + 1j * np.random.uniform(1, 5, (10, 4, 2))
        ).astype("complex128")
        self.inputs = {'X': X, 'Y': Y}
        Out = np.matmul(X, Y)
        self.outputs = {'Out': Out}

    def test_check_output(self):
        self.check_output(check_pir=True, check_prim=False)
        # pass

    def test_checkout_grad(self):
        self.check_grad(['X', 'Y'], 'Out', check_pir=True, check_prim=False)


if __name__ == "__main__":
    unittest.main()
