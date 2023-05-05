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
from paddle import fluid
from paddle.fluid import core


def _mode1D(a):
    sorted_inds = np.argsort(a, kind='stable')
    sorted_array = a[sorted_inds]
    max_freq = 0
    cur_freq = 0
    mode = -1
    for i in range(len(sorted_array)):
        cur_freq += 1
        if i == len(sorted_array) - 1 or sorted_array[i] != sorted_array[i + 1]:
            if cur_freq > max_freq:
                mode = sorted_array[i]
                index = sorted_inds[i]
                max_freq = cur_freq
            cur_freq = 0
    return mode, index


def cal_mode(a, axis, keepdim=False):
    if axis < 0:
        axis = len(a.shape) + axis
    in_dims = list(range(a.ndim))
    a_view = np.transpose(a, in_dims[:axis] + in_dims[axis + 1 :] + [axis])
    inds = np.ndindex(a_view.shape[:-1])
    modes = np.empty(a_view.shape[:-1], dtype=a.dtype)
    indexes = np.empty(a_view.shape[:-1], dtype=np.int64)
    for ind in inds:
        modes[ind], indexes[ind] = _mode1D(a_view[ind])
    if keepdim:
        newshape = list(a.shape)
        newshape[axis] = 1
        modes = modes.reshape(newshape)
        indexes = indexes.reshape(newshape)
    return modes, indexes


def init_numeric_grads(input_shape, out_indices, axis, dtype):
    if axis < 0:
        axis = len(input_shape) + axis
    grad = np.zeros(input_shape).astype(dtype)
    in_dims = list(range(grad.ndim))
    if axis == len(input_shape) - 1:
        a_view = grad
    else:
        a_view = np.transpose(
            grad,
            in_dims[:axis] + in_dims[axis + 1 :] + [axis],
        )
    idx = np.array(out_indices).flatten()
    inds = np.ndindex(a_view.shape[:-1])
    for i, ind in enumerate(inds):
        a_view[ind][idx[i]] = 1 / np.prod(out_indices.shape)
    if axis == len(input_shape) - 1:
        grad = a_view
    else:
        grad = np.transpose(
            a_view,
            in_dims[:axis] + in_dims[-1:] + in_dims[axis:-1],
        )
    return grad


class TestModeOp(OpTest):
    def init_args(self):
        self.axis = 1

    def setUp(self):
        self.op_type = "mode"
        self.python_api = paddle.mode
        self.dtype = np.float64
        self.input_data = np.random.rand(2, 64, 1)
        self.init_args()
        self.inputs = {'X': self.input_data}
        self.attrs = {'axis': self.axis}
        output, indices = cal_mode(self.input_data, axis=self.axis)
        self.outputs = {'Out': output, 'Indices': indices}

    def test_check_output(self):
        paddle.enable_static()
        self.check_output()

    def test_check_grad(self):
        paddle.enable_static()
        grad = init_numeric_grads(
            self.input_data.shape,
            self.outputs['Indices'],
            self.axis,
            np.float64,
        )
        self.check_grad({'X'}, 'Out', user_defined_grads=[grad])


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestModeFP16Op(OpTest):
    def init_args(self):
        self.axis = 1

    def setUp(self):
        self.op_type = "mode"
        self.python_api = paddle.mode
        self.dtype = np.float16
        self.input_data = np.random.rand(2, 64, 1).astype(np.float16)
        self.init_args()
        self.inputs = {'X': self.input_data.astype(self.dtype)}
        self.attrs = {'axis': self.axis}
        output, indices = cal_mode(self.input_data, axis=self.axis)
        self.outputs = {'Out': output, 'Indices': indices}

    def test_check_output(self):
        paddle.enable_static()
        place = core.CUDAPlace(0)
        if core.is_float16_supported(place):
            self.check_output_with_place(place)

    def test_check_grad(self):
        paddle.enable_static()
        place = core.CUDAPlace(0)
        grad = init_numeric_grads(
            self.input_data.shape,
            self.outputs['Indices'],
            self.axis,
            np.float32,
        )

        if core.is_float16_supported(place):
            self.check_grad_with_place(
                place, {'X'}, 'Out', user_defined_grads=[grad]
            )


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and not support the bfloat16",
)
class TestModeBF16Op(OpTest):
    def init_args(self):
        self.axis = 1

    def setUp(self):
        self.op_type = "mode"
        self.__class__.op_type = "mode"
        self.python_api = paddle.mode
        self.dtype = np.uint16
        self.input_data = np.random.rand(2, 64, 1).astype(np.float32)
        self.init_args()
        self.input_data = convert_uint16_to_float(
            convert_float_to_uint16(self.input_data)
        )

        self.inputs = {'X': convert_float_to_uint16(self.input_data)}
        self.attrs = {'axis': self.axis}
        output, indices = cal_mode(self.input_data, axis=self.axis)
        self.outputs = {
            'Out': convert_float_to_uint16(output),
            'Indices': indices,
        }

    def test_check_output(self):
        place = core.CUDAPlace(0)
        paddle.enable_static()
        if core.is_bfloat16_supported(place):
            self.check_output_with_place(place)

    def test_check_grad(self):
        place = core.CUDAPlace(0)
        paddle.enable_static()
        grad = init_numeric_grads(
            self.input_data.shape, self.outputs['Indices'], self.axis
        )
        if core.is_bfloat16_supported(place):
            self.check_grad_with_place(
                place, {'X'}, 'Out', user_defined_grads=[grad]
            )


class TestModeOpLastdim(OpTest):
    def init_args(self):
        self.axis = -1

    def setUp(self):
        self.op_type = "mode"
        self.python_api = paddle.mode
        self.dtype = np.float64
        self.input_data = np.random.rand(2, 1, 1, 2, 30)
        self.init_args()
        self.inputs = {'X': self.input_data}
        self.attrs = {'axis': self.axis}
        output, indices = cal_mode(self.input_data, axis=self.axis)
        self.outputs = {'Out': output, 'Indices': indices}

    def test_check_output(self):
        paddle.enable_static()
        self.check_output()

    def test_check_grad(self):
        paddle.enable_static()
        grad = init_numeric_grads(
            self.input_data.shape,
            self.outputs['Indices'],
            self.axis,
            np.float64,
        )
        self.check_grad({'X'}, 'Out', user_defined_grads=[grad])


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestModeFP16OpLastdim(TestModeFP16Op):
    def init_args(self):
        self.axis = -1

    def setUp(self):
        self.op_type = "mode"
        self.python_api = paddle.mode
        self.dtype = np.float16
        self.input_data = np.random.rand(2, 1, 1, 2, 30).astype(np.float16)
        self.init_args()
        self.inputs = {'X': self.input_data.astype(self.dtype)}
        self.attrs = {'axis': self.axis}
        output, indices = cal_mode(self.input_data, axis=self.axis)
        self.outputs = {'Out': output, 'Indices': indices}


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestModeBF16OpLastdim(TestModeBF16Op):
    def init_args(self):
        self.axis = -1

    def setUp(self):
        self.op_type = "mode"
        self.python_api = paddle.mode
        self.dtype = np.uint16
        self.input_data = np.random.rand(2, 1, 1, 2, 30).astype(np.float32)
        self.init_args()
        self.input_data = convert_uint16_to_float(
            convert_float_to_uint16(self.input_data)
        )
        self.inputs = {'X': convert_float_to_uint16(self.input_data)}
        self.attrs = {'axis': self.axis}
        output, indices = cal_mode(self.input_data, axis=self.axis)
        self.outputs = {
            'Out': convert_float_to_uint16(output),
            'Indices': indices,
        }


class TestModeOpKernels(unittest.TestCase):
    def setUp(self):
        self.axises = [-1, 1]
        np.random.seed(666)
        self.inputs = np.ceil(np.random.rand(2, 10, 10) * 1000)

    def test_mode_op(self):
        def test_cpu_kernel():
            paddle.set_device('cpu')
            tensor = paddle.to_tensor(self.inputs)
            for axis in self.axises:
                value_expect, indice_expect = cal_mode(self.inputs, axis)
                v, inds = paddle.mode(tensor, axis)
                np.testing.assert_allclose(v.numpy(), value_expect, rtol=1e-05)

                value_expect, indice_expect = cal_mode(
                    self.inputs, axis, keepdim=True
                )
                v, inds = paddle.mode(tensor, axis, keepdim=True)
                np.testing.assert_allclose(v.numpy(), value_expect, rtol=1e-05)

        def test_gpu_kernel():
            paddle.set_device('gpu')
            tensor = paddle.to_tensor(self.inputs)
            for axis in self.axises:
                value_expect, indice_expect = cal_mode(self.inputs, axis)
                v, inds = paddle.mode(tensor, axis)
                np.testing.assert_allclose(v.numpy(), value_expect, rtol=1e-05)

                value_expect, indice_expect = cal_mode(
                    self.inputs, axis, keepdim=True
                )
                v, inds = paddle.mode(tensor, axis, keepdim=True)
                np.testing.assert_allclose(v.numpy(), value_expect, rtol=1e-05)

        paddle.disable_static()
        test_cpu_kernel()
        if fluid.core.is_compiled_with_cuda():
            test_gpu_kernel()


class TestModeOpErrors(unittest.TestCase):
    def setUp(self):
        self.x = paddle.uniform([2, 10, 20, 25], dtype='float32')

        def test_dim_range_error():
            self.x.mode(axis=5)

        self.assertRaises(ValueError, test_dim_range_error)


class TestModeOpInStatic(unittest.TestCase):
    def setUp(self):
        np.random.seed(666)
        self.input_data = np.ceil(
            np.random.random((2, 10, 10)) * 1000, dtype=np.float64
        )

    def test_run_static(self):
        paddle.enable_static()
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            input_tensor = paddle.static.data(
                name="x", shape=[2, 10, 10], dtype="float64"
            )

            result = paddle.mode(input_tensor, axis=1)
            expect_value = cal_mode(self.input_data, axis=1)[0]
            exe = paddle.static.Executor(paddle.CPUPlace())
            paddle_result = exe.run(
                feed={"x": self.input_data}, fetch_list=[result]
            )[0]
            np.testing.assert_allclose(paddle_result, expect_value, rtol=1e-05)


class TestModeZeroError(unittest.TestCase):
    def test_errors(self):
        with paddle.fluid.dygraph.guard():

            def test_0_size():
                array = np.array([], dtype=np.float32)
                x = paddle.to_tensor(np.reshape(array, [0, 0]), dtype='float32')
                paddle.mode(x, axis=0)

            self.assertRaises(ValueError, test_0_size)


if __name__ == '__main__':
    unittest.main()
