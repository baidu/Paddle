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
from op_test import OpTest, convert_float_to_uint16

import paddle
import paddle.nn.functional as F
from paddle.base import core

np.random.seed(10)


def ref_log_softmax(x):
    shiftx = x - np.max(x)
    out = shiftx - np.log(np.exp(shiftx).sum())
    return out


def ref_log_softmax_grad(x, axis):
    if axis < 0:
        axis += len(x.shape)
    out = np.apply_along_axis(ref_log_softmax, axis, x)
    axis_dim = x.shape[axis]
    dout = np.full_like(x, fill_value=1.0 / x.size)
    dx = dout - np.exp(out) * dout.copy().sum(axis=axis, keepdims=True).repeat(
        axis_dim, axis=axis
    )
    return dx


class TestLogSoftmaxOp(OpTest):
    def setUp(self):
        self.op_type = 'log_softmax'
        self.prim_op_type = "comp"
        self.python_api = F.log_softmax
        self.public_python_api = F.log_softmax
        self.dtype = 'float64'
        self.shape = [2, 3, 4, 5]
        self.axis = -1
        self.set_attrs()

        x = np.random.uniform(0.1, 1.0, self.shape).astype(self.dtype)
        out = np.apply_along_axis(ref_log_softmax, self.axis, x)
        self.x_grad = ref_log_softmax_grad(x, self.axis)

        self.inputs = {'X': x}
        self.outputs = {'Out': out}
        self.attrs = {'axis': self.axis}

    def set_attrs(self):
        pass

    def test_check_output(self):
        self.check_output(check_pir=True, check_prim_pir=True)

    def test_check_grad(self):
        self.check_grad(
            ['X'], ['Out'], user_defined_grads=[self.x_grad], check_pir=True
        )


class TestLogSoftmaxOp_ZeroDim(TestLogSoftmaxOp):
    def setUp(self):
        self.op_type = 'log_softmax'
        self.prim_op_type = "comp"
        self.python_api = F.log_softmax
        self.public_python_api = F.log_softmax
        self.dtype = 'float64'

        x = np.random.uniform(0.1, 1.0, []).astype(self.dtype)
        out = np.array(0.0).astype(self.dtype)

        self.inputs = {'X': x}
        self.outputs = {'Out': out}
        self.attrs = {'axis': -1}

    def test_check_output(self):
        self.check_output(check_pir=True, check_prim_pir=True)

    def test_check_grad(self):
        self.check_grad(['X'], ['Out'], check_pir=True)


class TestLogSoftmaxShape(TestLogSoftmaxOp):
    def set_attrs(self):
        self.shape = [12, 10]


class TestLogSoftmaxFp32_Avx(TestLogSoftmaxOp):
    def set_attrs(self):
        self.dtype = 'float32'


class TestLogSoftmaxAxis(TestLogSoftmaxOp):
    def set_attrs(self):
        self.axis = 1


class TestLogSoftmaxFP16OP(TestLogSoftmaxOp):
    def set_attrs(self):
        self.dtype = np.float16

    def test_check_output(self):
        self.check_output(atol=1e-3, check_pir=True, check_prim_pir=True)

    def test_check_grad(self):
        self.check_grad(['X'], ['Out'], max_relative_error=1e-2, check_pir=True)


class TestLogSoftmaxShapeFP16OP(TestLogSoftmaxFP16OP):
    def set_attrs(self):
        self.dtype = np.float16
        self.shape = [12, 10]


class TestLogSoftmaxAxisFP16OP(TestLogSoftmaxFP16OP):
    def set_attrs(self):
        self.dtype = np.float16
        self.axis = 1


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestLogSoftmaxBF16Op(OpTest):
    def setUp(self):
        self.op_type = 'log_softmax'
        self.prim_op_type = "comp"
        self.python_api = F.log_softmax
        self.public_python_api = F.log_softmax
        self.dtype = np.uint16
        self.shape = [2, 3, 4, 5]
        self.axis = -1

        x = np.random.uniform(0.1, 1.0, self.shape).astype(np.float32)
        out = np.apply_along_axis(ref_log_softmax, self.axis, x)
        self.x_grad = ref_log_softmax_grad(x, self.axis)

        self.inputs = {'X': convert_float_to_uint16(x)}
        self.outputs = {'Out': convert_float_to_uint16(out)}
        self.attrs = {'axis': self.axis}

    def test_check_output(self):
        place = core.CUDAPlace(0)
        self.check_output_with_place(place, check_pir=True, check_prim_pir=True)

    def test_check_grad(self):
        place = core.CUDAPlace(0)
        self.check_grad_with_place(
            place,
            ['X'],
            ['Out'],
            user_defined_grads=[self.x_grad],
            check_pir=True,
        )


class TestLogSoftmaxLargeDimFP16OP(TestLogSoftmaxOp):
    def set_attrs(self):
        self.dtype = np.float16
        self.shape = [16, 100000]


class TestNNLogSoftmaxAPI(unittest.TestCase):
    def setUp(self):
        self.x_shape = [2, 3, 4, 5]
        self.x = np.random.uniform(-1.0, 1.0, self.x_shape).astype(np.float32)
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.base.core.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    def check_api(self, axis=-1):
        ref_out = np.apply_along_axis(ref_log_softmax, axis, self.x)

        logsoftmax = paddle.nn.LogSoftmax(axis)
        # test static api
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data(name='x', shape=self.x_shape)
            y = logsoftmax(x)
            exe = paddle.static.Executor(self.place)
            out = exe.run(feed={'x': self.x}, fetch_list=[y])
        np.testing.assert_allclose(out[0], ref_out, rtol=1e-05)

        # test dygraph api
        paddle.disable_static()
        x = paddle.to_tensor(self.x)
        y = logsoftmax(x)
        np.testing.assert_allclose(y.numpy(), ref_out, rtol=1e-05)
        paddle.enable_static()

    def test_check_api(self):
        for axis in [-1, 1]:
            self.check_api(axis)


class TestNNFunctionalLogSoftmaxAPI(unittest.TestCase):
    def setUp(self):
        self.x_shape = [2, 3, 4, 5]
        self.x = np.random.uniform(-1, 1, self.x_shape).astype(np.float32)
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.base.core.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    def check_api(self, axis=-1, dtype=None):
        x = self.x.copy()
        if dtype is not None:
            x = x.astype(dtype)
        ref_out = np.apply_along_axis(ref_log_softmax, axis, x)
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data(name='x', shape=self.x_shape)
            y = F.log_softmax(x, axis, dtype)
            exe = paddle.static.Executor(self.place)
            out = exe.run(feed={'x': self.x}, fetch_list=[y])
        np.testing.assert_allclose(out[0], ref_out, rtol=1e-05)

        paddle.disable_static()
        x = paddle.to_tensor(self.x)
        y = F.log_softmax(x, axis, dtype)
        np.testing.assert_allclose(y.numpy(), ref_out, rtol=1e-05)
        paddle.enable_static()

    def test_check_api(self):
        for axis in [-1, 1]:
            self.check_api(axis)
        self.check_api(-1, 'float64')

    def test_errors(self):
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data(name='X1', shape=[100], dtype='int32')
            self.assertRaises(TypeError, F.log_softmax, x)

            x = paddle.static.data(name='X2', shape=[100], dtype='float32')
            self.assertRaises(TypeError, F.log_softmax, x, dtype='int32')


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
