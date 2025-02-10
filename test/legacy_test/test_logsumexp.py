#  Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
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
from paddle.base import core


def ref_logsumexp(x, axis=None, keepdim=False, reduce_all=False):
    if isinstance(axis, int):
        axis = (axis,)
    elif isinstance(axis, list):
        axis = tuple(axis)
    if reduce_all:
        axis = None
    out = np.log(np.exp(x).sum(axis=axis, keepdims=keepdim))
    return out


def logsumexp_wrapper(x, axis=None, keepdim=False, allreduce=False):
    if allreduce:
        return paddle.logsumexp(x, None, keepdim)
    return paddle.logsumexp(x, axis, keepdim)


def logsumexp_op_grad(x, axis=None, keepdim=False, reduce_all=False):
    paddle.disable_static()
    tensor_x = paddle.to_tensor(x)
    tensor_x.stop_gradient = False
    out = logsumexp_wrapper(tensor_x, axis, keepdim, reduce_all)
    grad = paddle.grad(out, [tensor_x])
    x_grad = grad[0].numpy()
    paddle.enable_static()
    return x_grad


def logsumexp_ref_grad(x):
    sum = np.exp(x).sum()
    return np.exp(x) / sum


class TestLogsumexp(OpTest):
    def setUp(self):
        self.op_type = 'logsumexp'
        self.prim_op_type = "prim"
        self.python_api = logsumexp_wrapper
        self.public_python_api = logsumexp_wrapper
        self.shape = [2, 3, 4, 5]
        self.dtype = 'float64'
        self.axis = [-1]
        self.keepdim = False
        self.reduce_all = False
        self.set_attrs()

        np.random.seed(10)
        x = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        out = ref_logsumexp(x, self.axis, self.keepdim, self.reduce_all)

        self.inputs = {'X': x}
        self.outputs = {'Out': out}
        self.attrs = {
            'axis': self.axis,
            'keepdim': self.keepdim,
            'reduce_all': self.reduce_all,
        }
        self.user_defined_grads = None
        self.user_defined_grad_outputs = None

    def set_attrs(self):
        pass

    def test_check_output(self):
        self.check_output(
            check_pir=True,
            check_prim_pir=True,
        )

    def test_check_grad(self):
        self.check_grad(
            ['X'],
            ['Out'],
            user_defined_grads=self.user_defined_grads,
            user_defined_grad_outputs=self.user_defined_grad_outputs,
            check_pir=True,
            check_prim_pir=True,
        )


class TestLogsumexp_ZeroDim(TestLogsumexp):
    def set_attrs(self):
        self.shape = []
        self.axis = []


class TestLogsumexp_shape(TestLogsumexp):
    def set_attrs(self):
        self.shape = [4, 5, 6]


class TestLogsumexp_axis(TestLogsumexp):
    def set_attrs(self):
        self.axis = [0, -1]


class TestLogsumexp_axis_all(TestLogsumexp):
    def set_attrs(self):
        self.axis = [0, 1, 2, 3]


class TestLogsumexp_keepdim(TestLogsumexp):
    def set_attrs(self):
        self.keepdim = True


class TestLogsumexp_reduce_all(TestLogsumexp):
    def set_attrs(self):
        self.reduce_all = True


class TestLogsumexp_FP32(TestLogsumexp):
    def set_attrs(self):
        self.dtype = 'float32'

    def test_check_grad(self):
        self.__class__.dtype = self.dtype
        x_grad = logsumexp_op_grad(self.inputs['X'])
        ref_x_grad = logsumexp_ref_grad(self.inputs['X'])
        np.testing.assert_allclose(x_grad, ref_x_grad, rtol=1e-08, atol=1e-08)


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestLogsumexp_FP16(TestLogsumexp):
    def set_attrs(self):
        self.dtype = 'float16'

    def test_check_output(self):
        place = core.CUDAPlace(0)
        self.check_output_with_place(
            place,
            check_pir=True,
            check_prim_pir=True,
        )

    def test_check_grad(self):
        place = core.CUDAPlace(0)
        self.check_grad_with_place(
            place,
            ['X'],
            'Out',
            check_pir=True,
            check_prim_pir=True,
        )

    def set_attrs_addition(self):
        pass


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and not support the bfloat16",
)
class TestLogsumexpBF16Op(TestLogsumexp):
    def setUp(self):
        self.op_type = 'logsumexp'
        self.prim_op_type = "prim"
        self.python_api = logsumexp_wrapper
        self.public_python_api = logsumexp_wrapper
        self.dtype = np.uint16
        self.shape = [2, 3, 4, 5]
        self.axis = [-1]
        self.keepdim = False
        self.reduce_all = False
        self.set_attrs()
        x = np.random.uniform(-1, 1, self.shape).astype(np.float64)
        out = ref_logsumexp(x, self.axis, self.keepdim, self.reduce_all)
        self.inputs = {'X': convert_float_to_uint16(x)}
        self.outputs = {'Out': convert_float_to_uint16(out)}
        self.attrs = {
            'axis': self.axis,
            'keepdim': self.keepdim,
            'reduce_all': self.reduce_all,
        }

    def test_check_output(self):
        place = core.CUDAPlace(0)
        self.check_output_with_place(
            place,
            check_pir=True,
            check_prim_pir=True,
        )

    def test_check_grad(self):
        place = core.CUDAPlace(0)
        self.check_grad_with_place(
            place,
            ['X'],
            'Out',
            check_pir=True,
            check_prim_pir=True,
        )

    def set_attrs(self):
        pass


class TestLogsumexpError(unittest.TestCase):
    def test_errors(self):
        with paddle.static.program_guard(paddle.static.Program()):
            self.assertRaises(TypeError, paddle.logsumexp, 1)
            x1 = paddle.static.data(name='x1', shape=[120], dtype="bool")
            self.assertRaises(TypeError, paddle.logsumexp, x1)


class TestLogsumexpAPI(unittest.TestCase):
    def setUp(self):
        self.shape = [2, 3, 4, 5]
        self.x = np.random.uniform(-1, 1, self.shape).astype(np.float32)
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.base.core.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    def api_case(self, axis=None, keepdim=False):
        out_ref = ref_logsumexp(self.x, axis, keepdim)
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data('X', self.shape)
            out = paddle.logsumexp(x, axis, keepdim)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'X': self.x}, fetch_list=[out])
        np.testing.assert_allclose(res[0], out_ref, rtol=1e-05)

        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x)
        out = paddle.logsumexp(x, axis, keepdim)
        np.testing.assert_allclose(out.numpy(), out_ref, rtol=1e-05)
        paddle.enable_static()

    def test_api(self):
        self.api_case()
        self.api_case(2)
        self.api_case([-1])
        self.api_case([2, -3])
        self.api_case((0, 1, -1))
        self.api_case(keepdim=True)

    def test_alias(self):
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x)
        out1 = paddle.logsumexp(x)
        out2 = paddle.tensor.logsumexp(x)
        out3 = paddle.tensor.math.logsumexp(x)
        out_ref = ref_logsumexp(self.x)
        for out in [out1, out2, out3]:
            np.testing.assert_allclose(out.numpy(), out_ref, rtol=1e-05)
        paddle.enable_static()


# Test logsumexp bug
class TestLogZeroError(unittest.TestCase):
    def test_errors(self):
        with paddle.base.dygraph.guard():

            def test_0_size():
                array = np.array([], dtype=np.float32)
                x = paddle.to_tensor(
                    np.reshape(array, [0, 0, 0]), dtype='float32'
                )
                paddle.logsumexp(x, axis=1)

            self.assertRaises(ValueError, test_0_size)


if __name__ == '__main__':
    unittest.main()
