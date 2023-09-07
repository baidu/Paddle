#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
from eager_op_test import (
    OpTest,
    convert_float_to_uint16,
    convert_uint16_to_float,
)

import paddle
from paddle.base import core


def numpy_topk(x, k=1, axis=-1, largest=True):
    if axis < 0:
        axis = len(x.shape) + axis
    if largest:
        indices = np.argsort(-x, axis=axis)
    else:
        indices = np.argsort(x, axis=axis)
    if largest:
        value = -np.sort(-x, axis=axis)
    else:
        value = np.sort(x, axis=axis)
    indices = indices.take(indices=range(0, k), axis=axis)
    value = value.take(indices=range(0, k), axis=axis)
    return value, indices


class TestTopkOp(OpTest):
    def init_args(self):
        self.k = 3
        self.axis = 1
        self.largest = True

    def setUp(self):
        self.op_type = "top_k_v2"
        self.prim_op_type = "prim"
        self.python_api = paddle.topk
        self.public_python_api = paddle.topk
        self.dtype = np.float64
        self.input_data = np.random.rand(10, 20)
        self.init_args()
        self.if_enable_cinn()
        self.inputs = {'X': self.input_data}
        self.attrs = {'k': self.k, 'axis': self.axis, 'largest': self.largest}
        output, indices = numpy_topk(
            self.input_data, axis=self.axis, k=self.k, largest=self.largest
        )
        self.outputs = {'Out': output, 'Indices': indices}

    def if_enable_cinn(self):
        pass

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', check_prim=True)


class TestTopkOp_ZeroDim(TestTopkOp):
    def init_args(self):
        self.k = 1
        self.axis = 0
        self.largest = True

    def setUp(self):
        self.op_type = "top_k_v2"
        self.prim_op_type = "prim"
        self.python_api = paddle.topk
        self.public_python_api = paddle.topk
        self.dtype = np.float64
        self.input_data = np.random.random(())
        self.init_args()
        self.if_enable_cinn()
        self.inputs = {'X': self.input_data}
        self.attrs = {'k': self.k, 'largest': self.largest}
        output, indices = self.input_data, np.array(0).astype('int64')
        self.outputs = {'Out': output, 'Indices': indices}

    def if_enable_cinn(self):
        pass


class TestTopkOp1(TestTopkOp):
    def init_args(self):
        self.k = 3
        self.axis = 0
        self.largest = False


class TestTopkOp2(TestTopkOp):
    def init_args(self):
        self.k = 4
        self.axis = 0
        self.largest = False


class TestTopkOp3(TestTopkOp):
    def init_args(self):
        self.k = 6
        self.axis = 1
        self.largest = True

    def setUp(self):
        self.op_type = "top_k_v2"
        self.prim_op_type = "prim"
        self.python_api = paddle.topk
        self.public_python_api = paddle.topk
        self.dtype = np.float64
        self.input_data = np.random.rand(16, 100)
        self.init_args()
        self.inputs = {'X': self.input_data}
        self.attrs = {'k': self.k, 'axis': self.axis, 'largest': self.largest}
        output, indices = numpy_topk(
            self.input_data, axis=self.axis, k=self.k, largest=self.largest
        )
        self.outputs = {'Out': output, 'Indices': indices}


class TestTopkOp4(TestTopkOp):
    def init_args(self):
        self.k = 3
        self.axis = 1
        self.largest = True

    def setUp(self):
        self.op_type = "top_k_v2"
        self.prim_op_type = "prim"
        self.python_api = paddle.topk
        self.public_python_api = paddle.topk
        self.dtype = np.float64
        self.input_data = np.random.rand(10, 10, 5)
        self.init_args()
        self.if_enable_cinn()
        self.inputs = {'X': self.input_data}
        self.attrs = {'k': self.k, 'axis': self.axis, 'largest': self.largest}
        output, indices = numpy_topk(
            self.input_data, axis=self.axis, k=self.k, largest=self.largest
        )
        self.outputs = {'Out': output, 'Indices': indices}


class TestTopkOp5(TestTopkOp):
    def init_args(self):
        self.k = 3
        self.axis = 1
        self.largest = True

    def setUp(self):
        self.op_type = "top_k_v2"
        self.prim_op_type = "prim"
        self.python_api = paddle.topk
        self.public_python_api = paddle.topk
        self.dtype = np.float64
        self.input_data = np.random.rand(10, 10, 5)
        self.init_args()
        self.if_enable_cinn()
        self.inputs = {'X': self.input_data}
        self.attrs = {'k': self.k, 'axis': self.axis, 'largest': self.largest}
        output, indices = numpy_topk(
            self.input_data, axis=self.axis, k=self.k, largest=self.largest
        )
        self.outputs = {'Out': output, 'Indices': indices}


class TestTopkOp6(TestTopkOp):
    def init_args(self):
        self.k = 3
        self.axis = 1
        self.largest = True

    def setUp(self):
        self.op_type = "top_k_v2"
        self.prim_op_type = "prim"
        self.python_api = paddle.topk
        self.public_python_api = paddle.topk
        self.dtype = np.float32
        self.input_data = np.random.rand(10, 10, 5)
        self.init_args()
        self.if_enable_cinn()
        self.inputs = {'X': self.input_data}
        self.attrs = {'k': self.k, 'axis': self.axis, 'largest': self.largest}
        output, indices = numpy_topk(
            self.input_data, axis=self.axis, k=self.k, largest=self.largest
        )
        self.outputs = {'Out': output, 'Indices': indices}


class TestTopkOp7(TestTopkOp):
    def init_args(self):
        self.k = 10
        self.axis = 1
        self.largest = True

    def setUp(self):
        self.op_type = "top_k_v2"
        self.prim_op_type = "prim"
        self.python_api = paddle.topk
        self.public_python_api = paddle.topk
        self.dtype = np.float16
        self.input_data = np.random.rand(10, 20, 10)
        self.init_args()
        self.if_enable_cinn()
        self.inputs = {'X': self.input_data}
        self.attrs = {'k': self.k, 'axis': self.axis, 'largest': self.largest}
        output, indices = numpy_topk(
            self.input_data, axis=self.axis, k=self.k, largest=self.largest
        )
        self.outputs = {'Out': output, 'Indices': indices}


class TestTopkFP16Op(TestTopkOp):
    def setUp(self):
        self.op_type = "top_k_v2"
        self.python_api = paddle.topk
        self.public_python_api = paddle.topk
        self.dtype = np.float16
        self.prim_op_type = "prim"
        self.input_data = np.random.rand(10, 20).astype(self.dtype)
        self.init_args()
        self.if_enable_cinn()
        self.inputs = {'X': self.input_data}
        self.attrs = {'k': self.k, 'axis': self.axis, 'largest': self.largest}
        output, indices = numpy_topk(
            self.input_data, axis=self.axis, k=self.k, largest=self.largest
        )
        self.outputs = {'Out': output, 'Indices': indices}


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA or not support the bfloat16",
)
class TestTopkBF16Op(TestTopkOp):
    def setUp(self):
        self.op_type = "top_k_v2"
        self.python_api = paddle.topk
        self.public_python_api = paddle.topk
        self.dtype = np.uint16
        self.prim_op_type = "prim"
        self.input_data = np.random.random([10, 20]).astype(np.float32)
        self.init_args()
        self.if_enable_cinn()
        self.inputs = {'X': convert_float_to_uint16(self.input_data)}
        self.input_data = convert_uint16_to_float(self.inputs['X'])
        self.attrs = {'k': self.k, 'axis': self.axis, 'largest': self.largest}
        output, indices = numpy_topk(
            self.input_data, axis=self.axis, k=self.k, largest=self.largest
        )
        self.outputs = {
            'Out': convert_float_to_uint16(output),
            'Indices': indices,
        }

    def if_enable_cinn(self):
        self.enable_cinn = False

    def test_check_output(self):
        place = core.CUDAPlace(0)
        self.check_output_with_place(place)

    def test_check_grad(self):
        place = core.CUDAPlace(0)
        self.check_grad_with_place(place, ['X'], 'Out', check_prim=True)


class TestTopKAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        self.input_data = np.random.rand(6, 7, 8)
        self.large_input_data = np.random.rand(2, 1030)

    def run_dygraph(self, place):
        with paddle.base.dygraph.guard(place):
            input_tensor = paddle.to_tensor(self.input_data)
            large_input_tensor = paddle.to_tensor(self.large_input_data)
            # test case for basic test case 1
            paddle_result = paddle.topk(input_tensor, k=2)
            numpy_result = numpy_topk(self.input_data, k=2)
            np.testing.assert_allclose(
                paddle_result[0].numpy(), numpy_result[0], rtol=1e-05
            )
            np.testing.assert_allclose(
                paddle_result[1].numpy(), numpy_result[1], rtol=1e-05
            )
            # test case for basic test case 2 with axis
            paddle_result = paddle.topk(input_tensor, k=2, axis=1)
            numpy_result = numpy_topk(self.input_data, k=2, axis=1)
            np.testing.assert_allclose(
                paddle_result[0].numpy(), numpy_result[0], rtol=1e-05
            )
            np.testing.assert_allclose(
                paddle_result[1].numpy(), numpy_result[1], rtol=1e-05
            )
            # test case for basic test case 3 with tensor K
            k_tensor = paddle.to_tensor(np.array([2]))
            paddle_result = paddle.topk(input_tensor, k=k_tensor, axis=1)
            numpy_result = numpy_topk(self.input_data, k=2, axis=1)
            np.testing.assert_allclose(
                paddle_result[0].numpy(), numpy_result[0], rtol=1e-05
            )
            np.testing.assert_allclose(
                paddle_result[1].numpy(), numpy_result[1], rtol=1e-05
            )
            # test case for basic test case 4 with tensor largest
            k_tensor = paddle.to_tensor(np.array([2]))
            paddle_result = paddle.topk(
                input_tensor, k=2, axis=1, largest=False
            )
            numpy_result = numpy_topk(
                self.input_data, k=2, axis=1, largest=False
            )
            np.testing.assert_allclose(
                paddle_result[0].numpy(), numpy_result[0], rtol=1e-05
            )
            np.testing.assert_allclose(
                paddle_result[1].numpy(), numpy_result[1], rtol=1e-05
            )
            # test case for basic test case 5 with axis -1
            k_tensor = paddle.to_tensor(np.array([2]))
            paddle_result = paddle.topk(
                input_tensor, k=2, axis=-1, largest=False
            )
            numpy_result = numpy_topk(
                self.input_data, k=2, axis=-1, largest=False
            )
            np.testing.assert_allclose(
                paddle_result[0].numpy(), numpy_result[0], rtol=1e-05
            )
            np.testing.assert_allclose(
                paddle_result[1].numpy(), numpy_result[1], rtol=1e-05
            )
            # test case for basic test case 6 for the partial sort
            paddle_result = paddle.topk(large_input_tensor, k=1, axis=-1)
            numpy_result = numpy_topk(self.large_input_data, k=1, axis=-1)
            np.testing.assert_allclose(
                paddle_result[0].numpy(), numpy_result[0], rtol=1e-05
            )
            np.testing.assert_allclose(
                paddle_result[1].numpy(), numpy_result[1], rtol=1e-05
            )
            # test case for basic test case 7 for the unsorted
            paddle_result = paddle.topk(input_tensor, k=2, axis=1, sorted=False)
            sort_paddle = numpy_topk(
                np.array(paddle_result[0].numpy()), axis=1, k=2
            )
            numpy_result = numpy_topk(self.input_data, k=2, axis=1)
            np.testing.assert_allclose(
                sort_paddle[0], numpy_result[0], rtol=1e-05
            )

    def run_static(self, place):
        paddle.enable_static()
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            input_tensor = paddle.static.data(
                name="x", shape=[6, 7, 8], dtype="float64"
            )
            large_input_tensor = paddle.static.data(
                name="large_x", shape=[2, 1030], dtype="float64"
            )
            k_tensor = paddle.static.data(name="k", shape=[1], dtype="int32")
            result1 = paddle.topk(input_tensor, k=2)
            result2 = paddle.topk(input_tensor, k=2, axis=-1)
            result3 = paddle.topk(input_tensor, k=k_tensor, axis=1)
            self.assertEqual(result3[0].shape, (6, -1, 8))
            self.assertEqual(result3[1].shape, (6, -1, 8))
            result4 = paddle.topk(input_tensor, k=2, axis=1, largest=False)
            result5 = paddle.topk(input_tensor, k=2, axis=-1, largest=False)
            result6 = paddle.topk(large_input_tensor, k=1, axis=-1)
            result7 = paddle.topk(input_tensor, k=2, axis=1, sorted=False)
            exe = paddle.static.Executor(place)
            input_data = np.random.rand(10, 20).astype("float64")
            large_input_data = np.random.rand(2, 100).astype("float64")
            paddle_result = exe.run(
                feed={
                    "x": self.input_data,
                    "large_x": self.large_input_data,
                    "k": np.array([2]).astype("int32"),
                },
                fetch_list=[
                    result1[0],
                    result1[1],
                    result2[0],
                    result2[1],
                    result3[0],
                    result3[1],
                    result4[0],
                    result4[1],
                    result5[0],
                    result5[1],
                    result6[0],
                    result6[1],
                    result7[0],
                    result7[1],
                ],
            )
            numpy_result = numpy_topk(self.input_data, k=2)
            np.testing.assert_allclose(
                paddle_result[0], numpy_result[0], rtol=1e-05
            )
            np.testing.assert_allclose(
                paddle_result[1], numpy_result[1], rtol=1e-05
            )
            numpy_result = numpy_topk(self.input_data, k=2, axis=-1)
            np.testing.assert_allclose(
                paddle_result[2], numpy_result[0], rtol=1e-05
            )
            np.testing.assert_allclose(
                paddle_result[3], numpy_result[1], rtol=1e-05
            )
            numpy_result = numpy_topk(self.input_data, k=2, axis=1)
            np.testing.assert_allclose(
                paddle_result[4], numpy_result[0], rtol=1e-05
            )
            np.testing.assert_allclose(
                paddle_result[5], numpy_result[1], rtol=1e-05
            )
            numpy_result = numpy_topk(
                self.input_data, k=2, axis=1, largest=False
            )
            np.testing.assert_allclose(
                paddle_result[6], numpy_result[0], rtol=1e-05
            )
            np.testing.assert_allclose(
                paddle_result[7], numpy_result[1], rtol=1e-05
            )
            numpy_result = numpy_topk(
                self.input_data, k=2, axis=-1, largest=False
            )
            np.testing.assert_allclose(
                paddle_result[8], numpy_result[0], rtol=1e-05
            )
            np.testing.assert_allclose(
                paddle_result[9], numpy_result[1], rtol=1e-05
            )
            numpy_result = numpy_topk(self.large_input_data, k=1, axis=-1)
            np.testing.assert_allclose(
                paddle_result[10], numpy_result[0], rtol=1e-05
            )
            np.testing.assert_allclose(
                paddle_result[11], numpy_result[1], rtol=1e-05
            )
            sort_paddle = numpy_topk(paddle_result[12], axis=1, k=2)
            numpy_result = numpy_topk(self.input_data, k=2, axis=1)
            np.testing.assert_allclose(
                sort_paddle[0], numpy_result[0], rtol=1e-05
            )

    def test_cases(self):
        places = [core.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(core.CUDAPlace(0))
        for place in places:
            self.run_dygraph(place)
            self.run_static(place)

    def test_errors(self):
        with paddle.base.dygraph.guard():
            x = paddle.to_tensor([1, 2, 3])
            with self.assertRaises(BaseException):
                paddle.topk(x, k=-1)

            with self.assertRaises(BaseException):
                paddle.topk(x, k=0)


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
