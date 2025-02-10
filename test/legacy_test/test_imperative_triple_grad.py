# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from unittest import TestCase

import numpy as np

import paddle
from paddle import base
from paddle.base.wrapped_decorator import wrap_decorator


def _dygraph_guard_(func):
    def __impl__(*args, **kwargs):
        if base.in_dygraph_mode():
            return func(*args, **kwargs)
        else:
            with base.dygraph.guard():
                return func(*args, **kwargs)

    return __impl__


dygraph_guard = wrap_decorator(_dygraph_guard_)


def random_var(size, low=-1, high=1, dtype='float32'):
    np.random.seed(2021)
    x_np = np.random.uniform(low=low, high=high, size=size).astype(dtype)
    return paddle.to_tensor(x_np)


class TestDygraphTripleGradMatmul(TestCase):
    def test_matmul_triple_grad(self):
        input_numpy = np.ones([3, 3]) * 2
        x = paddle.to_tensor(input_numpy, stop_gradient=False, dtype='float32')
        y = paddle.to_tensor(input_numpy, stop_gradient=False, dtype='float32')
        out = paddle.matmul(x, y, False, False)

        new_out_g = paddle.to_tensor(
            np.ones([3, 3]), stop_gradient=False, dtype='float32'
        )
        new_x_g, new_y_g = paddle.grad(
            [out], [x, y], [new_out_g], retain_graph=True, create_graph=True
        )

        new_x_g_g = paddle.to_tensor(
            np.ones([3, 3]), stop_gradient=False, dtype='float32'
        )
        new_y_g_g = paddle.to_tensor(
            np.ones([3, 3]), stop_gradient=False, dtype='float32'
        )
        new_a, new_b, new_c = paddle.grad(
            [new_x_g, new_y_g],
            [x, y, new_out_g],
            [new_x_g_g, new_y_g_g],
            retain_graph=True,
            create_graph=True,
        )

        new_a.backward()

        out_ref = np.ones([3, 3]) * 12.0
        np.testing.assert_array_equal(out.numpy(), out_ref)

        new_x_g_ref = np.ones([3, 3]) * 6.0
        new_y_g_ref = np.ones([3, 3]) * 6.0
        np.testing.assert_array_equal(new_x_g.numpy(), new_x_g_ref)
        np.testing.assert_array_equal(new_y_g.numpy(), new_y_g_ref)

        new_a_ref = np.ones([3, 3]) * 3.0
        new_b_ref = np.ones([3, 3]) * 3.0
        new_c_ref = np.ones([3, 3]) * 12.0

        np.testing.assert_array_equal(new_a.numpy(), new_a_ref)
        np.testing.assert_array_equal(new_b.numpy(), new_b_ref)
        np.testing.assert_array_equal(new_c.numpy(), new_c_ref)

        x_grad_ref = np.ones([3, 3]) * 0.0
        assert x.grad is None

        y_grad_ref = np.ones([3, 3]) * 0.0
        assert y.grad is None

        new_out_g_ref = np.ones([3, 3]) * 3.0
        np.testing.assert_array_equal(new_out_g.grad.numpy(), new_out_g_ref)

        new_x_g_g_ref = np.ones([3, 3]) * 0.0
        new_y_g_g_ref = np.ones([3, 3]) * 3.0
        assert new_x_g_g.grad is None
        np.testing.assert_array_equal(new_y_g_g.grad.numpy(), new_y_g_g_ref)


class TestDygraphTripleGrad(TestCase):
    def setUp(self):
        self.sort_sum_gradient = False
        self.shape = [5, 5]

    def grad(
        self,
        outputs,
        inputs,
        grad_outputs=None,
        no_grad_vars=None,
        retain_graph=None,
        create_graph=False,
        allow_unused=False,
    ):
        base.set_flags({'FLAGS_sort_sum_gradient': self.sort_sum_gradient})
        return base.dygraph.grad(
            outputs=outputs,
            inputs=inputs,
            grad_outputs=grad_outputs,
            no_grad_vars=no_grad_vars,
            retain_graph=retain_graph,
            create_graph=create_graph,
            allow_unused=allow_unused,
        )

    @dygraph_guard
    def func_exception(self):
        with self.assertRaises(AssertionError):
            self.grad(None, None)

        shape = self.shape

        with self.assertRaises(AssertionError):
            self.grad(1, random_var(shape))

        with self.assertRaises(AssertionError):
            self.grad(random_var(shape), 1)

        with self.assertRaises(AssertionError):
            self.grad([1], [random_var(shape)])

        with self.assertRaises(AssertionError):
            self.grad([random_var(shape)], [1])

        with self.assertRaises(AssertionError):
            self.grad(
                [random_var(shape), random_var(shape)],
                [random_var(shape)],
                [random_var(shape)],
            )

        with self.assertRaises(AssertionError):
            self.grad(
                [random_var(shape)], [random_var(shape)], no_grad_vars=[1]
            )

        with self.assertRaises(AssertionError):
            self.grad([random_var(shape)], [random_var(shape)], no_grad_vars=1)

    @dygraph_guard
    def func_example_with_gradient_and_create_graph(self):
        x = random_var(self.shape)
        x.retain_grads()
        x_np = x.numpy()
        x.stop_gradient = False

        y = random_var(self.shape)
        y_np = y.numpy()
        y.stop_gradient = False

        z = random_var(self.shape)
        z_np = z.numpy()
        numel = z_np.size
        z.stop_gradient = False

        out = paddle.nn.functional.sigmoid(paddle.matmul(x, y) + z)
        out_np = out.numpy()

        (dx_actual,) = self.grad([out], [x], create_graph=True)
        # Theoretical result based on math calculation
        dout = np.ones(self.shape).astype('float32')
        dx_expected = np.matmul(
            dout * out_np * (1 - out_np), np.transpose(y_np)
        )
        np.testing.assert_allclose(dx_actual.numpy(), dx_expected, rtol=1e-05)

        (ddx_actual,) = self.grad([dx_actual], [x], create_graph=True)
        # Theoretical result based on math calculation
        DDY = np.zeros(self.shape).astype('float32')
        DDX = np.ones(self.shape).astype('float32')
        double_grad_tmp1 = np.matmul(
            dout * out_np * (1 - out_np), np.transpose(DDY)
        )
        double_grad_tmp2 = np.matmul(DDX, y_np) + np.matmul(x_np, DDY)
        double_grad_tmp3 = (
            (1 - 2 * out_np) * dout * double_grad_tmp2 * out_np * (1 - out_np)
        )
        ddx_expected = double_grad_tmp1 + np.matmul(
            double_grad_tmp3, np.transpose(y_np)
        )
        np.testing.assert_allclose(ddx_actual.numpy(), ddx_expected, rtol=1e-05)

        # Theoretical result based on math calculation
        d_ddout = np.zeros(self.shape).astype('float32')
        tmp0 = np.matmul(DDX, y_np) + np.matmul(x_np, DDY)
        tmp1 = (1 - 2 * out_np) * ((1 - 2 * out_np) * dout * tmp0 * tmp0)
        tmp2 = (
            tmp0 * (1 - 2 * out_np) * d_ddout
            - 2 * dout * (1 - out_np) * out_np * tmp0 * tmp0
        )
        dddx_expected = np.matmul(
            ((tmp1 + tmp2) * out_np * (1 - out_np)), np.transpose(y_np)
        )

        ddx_actual.backward()
        dddx_grad_actual = x.gradient()
        np.testing.assert_allclose(dddx_grad_actual, dddx_expected, rtol=1e-05)

    def test_all_cases(self):
        self.func_exception()
        self.func_example_with_gradient_and_create_graph()


class TestDygraphTripleGradBroadcastCase(TestCase):
    def setUp(self):
        self.sort_sum_gradient = False
        self.x_shape = [3, 2, 2]
        self.y_shape = [1, 2, 2]
        self.z_shape = [2, 2]

    def grad(
        self,
        outputs,
        inputs,
        grad_outputs=None,
        no_grad_vars=None,
        retain_graph=None,
        create_graph=False,
        allow_unused=False,
    ):
        base.set_flags({'FLAGS_sort_sum_gradient': self.sort_sum_gradient})
        return base.dygraph.grad(
            outputs=outputs,
            inputs=inputs,
            grad_outputs=grad_outputs,
            no_grad_vars=no_grad_vars,
            retain_graph=retain_graph,
            create_graph=create_graph,
            allow_unused=allow_unused,
        )

    @dygraph_guard
    def func_example_with_gradient_and_create_graph(self):
        x = random_var(self.x_shape)
        x.retain_grads()
        x_np = x.numpy()
        x.stop_gradient = False

        y = random_var(self.y_shape)
        y_np = y.numpy()
        y.stop_gradient = False

        z = random_var(self.z_shape)
        z_np = z.numpy()
        numel = z_np.size
        z.stop_gradient = False

        out = paddle.nn.functional.sigmoid(paddle.matmul(x, y) + z)
        out_np = out.numpy()

        (dx_actual,) = self.grad([out], [x], create_graph=True)
        # Theoretical result based on math calculation
        dout = np.ones(self.x_shape).astype('float32')
        dx_expected = np.matmul(
            dout * out_np * (1 - out_np), np.transpose(y_np, axes=(0, 2, 1))
        )
        np.testing.assert_allclose(dx_actual.numpy(), dx_expected, rtol=1e-05)

        (ddx_actual,) = self.grad([dx_actual], [x], create_graph=True)
        # Theoretical result based on math calculation
        DDY = np.zeros(self.y_shape).astype('float32')
        DDX = np.ones(self.x_shape).astype('float32')
        double_grad_tmp1 = np.matmul(
            dout * out_np * (1 - out_np), np.transpose(DDY, axes=(0, 2, 1))
        )
        double_grad_tmp2 = np.matmul(DDX, y_np) + np.matmul(x_np, DDY)
        double_grad_tmp3 = (
            (1 - 2 * out_np) * dout * double_grad_tmp2 * out_np * (1 - out_np)
        )
        ddx_expected = double_grad_tmp1 + np.matmul(
            double_grad_tmp3, np.transpose(y_np, axes=(0, 2, 1))
        )
        np.testing.assert_allclose(ddx_actual.numpy(), ddx_expected, rtol=1e-05)

        # Theoretical result based on math calculation
        d_ddout = np.zeros(self.x_shape).astype('float32')
        tmp0 = np.matmul(DDX, y_np) + np.matmul(x_np, DDY)
        tmp1 = (1 - 2 * out_np) * ((1 - 2 * out_np) * dout * tmp0 * tmp0)
        tmp2 = (
            tmp0 * (1 - 2 * out_np) * d_ddout
            - 2 * dout * (1 - out_np) * out_np * tmp0 * tmp0
        )
        dddx_expected = np.matmul(
            ((tmp1 + tmp2) * out_np * (1 - out_np)),
            np.transpose(y_np, axes=(0, 2, 1)),
        )

        ddx_actual.backward()
        dddx_grad_actual = x.gradient()
        np.testing.assert_allclose(dddx_grad_actual, dddx_expected, rtol=1e-05)

    def test_all_cases(self):
        self.func_example_with_gradient_and_create_graph()


# d_ddout is none, dtype is float32
class TestDygraphTripleGradMatmulcase1(TestCase):
    def setUp(self):
        self.input_numpy_x = None
        self.input_numpy_y = None
        self.input_numpy_dout = None
        self.input_numpy_ddx = None
        self.input_numpy_ddy = None
        self.places = ["cpu"]
        if paddle.is_compiled_with_cuda():
            self.places.append("gpu")

    def actual(self):
        x = paddle.to_tensor(
            self.input_numpy_x, stop_gradient=False, dtype='float32'
        )
        y = paddle.to_tensor(
            self.input_numpy_y, stop_gradient=False, dtype='float32'
        )
        out = paddle.matmul(x, y, False, False)

        dout = paddle.to_tensor(
            self.input_numpy_dout, stop_gradient=False, dtype='float32'
        )
        (dx, dy) = paddle.grad(
            [out], [x, y], [dout], retain_graph=True, create_graph=True
        )
        ddx = paddle.to_tensor(
            self.input_numpy_ddx, stop_gradient=False, dtype='float32'
        )
        ddy = paddle.to_tensor(
            self.input_numpy_ddy, stop_gradient=False, dtype='float32'
        )
        dx_double_grad, dy_double_grad = paddle.grad(
            [dx, dy],
            [x, y],
            [ddx, ddy],
            retain_graph=True,
            create_graph=True,
        )
        # d_x, d_y should be none because ddd_out = None
        d_dout, d_ddx, d_ddy = paddle.grad(
            [dx_double_grad, dy_double_grad],
            [dout, ddx, ddy],
            retain_graph=False,
            create_graph=False,
        )
        return d_dout, d_ddx, d_ddy

    # case1: d_ddout is none, dims != 1
    def test_matmul_triple_grad_case1(self):
        def init_data():
            self.input_numpy_x = np.random.random([3, 3]).astype('float32')
            self.input_numpy_y = np.random.random([3, 3]).astype('float32')
            self.input_numpy_dout = np.ones([3, 3], dtype="float32")
            self.input_numpy_ddx = np.ones([3, 3], dtype="float32")
            self.input_numpy_ddy = np.ones([3, 3], dtype="float32")

        init_data()
        d_dout_expected = np.ones([3, 3], dtype="float32") * 6
        d_ddx_expected = np.ones([3, 3], dtype="float32") * 3
        d_ddy_expected = np.ones([3, 3], dtype="float32") * 3
        expected_results = (
            d_dout_expected,
            d_ddx_expected,
            d_ddy_expected,
        )

        for place in self.places:
            paddle.device.set_device(place)
            actual_results = self.actual()
            for expected_result, actual_result in zip(
                expected_results, actual_results
            ):
                np.testing.assert_allclose(
                    expected_result, actual_result, rtol=1e-6
                )

    # case2: d_ddout is none, dims = 1
    def test_matmul_triple_grad_case2(self):
        def init_data():
            self.input_numpy_x = np.random.random(
                [
                    3,
                ]
            ).astype('float32')
            self.input_numpy_y = np.random.random(
                [
                    3,
                ]
            ).astype('float32')
            self.input_numpy_dout = np.ones([1], dtype="float32")
            self.input_numpy_ddx = np.ones([3], dtype="float32")
            self.input_numpy_ddy = np.ones([3], dtype="float32")

        init_data()
        d_dout_expected = np.ones([1], dtype="float32") * 6
        d_ddx_expected = np.ones(
            [
                3,
            ],
            dtype="float32",
        )
        d_ddy_expected = np.ones(
            [
                3,
            ],
            dtype="float32",
        )
        expected_results = (
            d_dout_expected,
            d_ddx_expected,
            d_ddy_expected,
        )

        for place in self.places:
            paddle.device.set_device(place)
            actual_results = self.actual()
            for expected_result, actual_result in zip(
                expected_results, actual_results
            ):
                np.testing.assert_allclose(
                    expected_result, actual_result, rtol=1e-6
                )

    # case3: d_ddout is none , with broadcast
    def test_matmul_triple_grad_case3(self):
        def init_data():
            self.input_numpy_x = np.random.random([3, 1]).astype('float32')
            self.input_numpy_y = np.random.random(
                [
                    1,
                ]
            ).astype('float32')
            self.input_numpy_dout = np.ones([3], dtype="float32")
            self.input_numpy_ddx = np.ones([3, 1], dtype="float32")
            self.input_numpy_ddy = np.ones([1], dtype="float32")

        init_data()
        d_dout_expected = (
            np.ones(
                [
                    3,
                ],
                dtype="float32",
            )
            * 2
        )
        d_ddx_expected = np.ones([3, 1], dtype="float32")
        d_ddy_expected = np.ones([1], dtype="float32") * 3
        expected_results = (
            d_dout_expected,
            d_ddx_expected,
            d_ddy_expected,
        )

        for place in self.places:
            paddle.device.set_device(place)
            actual_results = self.actual()
            for expected_result, actual_result in zip(
                expected_results, actual_results
            ):
                np.testing.assert_allclose(
                    expected_result, actual_result, rtol=1e-6
                )


'''
# d_ddout is none, dtype is complex64
class TestDygraphTripleGradMatmulcase2(TestCase):
    def setUp(self):
        self.input_numpy_x = None
        self.input_numpy_y = None
        self.input_numpy_dout = None
        self.input_numpy_ddx = None
        self.input_numpy_ddy = None
        self.input_numpy_ddx_conj = None
        self.input_numpy_ddy_conj = None
        self.input_numpy_dout_conj = None
        self.places = ["cpu"]
        if paddle.is_compiled_with_cuda():
            self.places.append("gpu")

    def actual(self):
        x = paddle.to_tensor(
            self.input_numpy_x, stop_gradient=False, dtype='complex64'
        )
        y = paddle.to_tensor(
            self.input_numpy_y, stop_gradient=False, dtype='complex64'
        )
        out = paddle.matmul(x, y, False, False)

        dout = paddle.to_tensor(
            self.input_numpy_dout, stop_gradient=False, dtype='complex64'
        )
        (dx, dy) = paddle.grad(
            [out], [x, y], [dout], retain_graph=True, create_graph=True
        )
        ddx = paddle.to_tensor(
            self.input_numpy_ddx, stop_gradient=False, dtype='complex64'
        )
        ddy = paddle.to_tensor(
            self.input_numpy_ddy, stop_gradient=False, dtype='complex64'
        )
        dx_double_grad, dy_double_grad = paddle.grad(
            [dx, dy],
            [x, y],
            [ddx, ddy],
            retain_graph=True,
            create_graph=True,
        )
        d_x, d_y, d_dout, d_ddx, d_ddy = paddle.grad(
            [dx_double_grad, dy_double_grad],
            [x, y, dout, ddx, ddy],
            retain_graph=False,
            create_graph=False,
        )
        return d_x, d_y, d_dout, d_ddx, d_ddy

    # case1: no d_ddout, dims = 1, dtype is complex64
    def test_matmul_triple_grad_case1(self):
        def init_data():
            self.input_numpy_x = np.random.random([3]).astype(
                'float32'
            ) + 1j * np.random.random(
                [
                    3,
                ]
            ).astype(
                'float32'
            )
            self.input_numpy_y = np.random.random([3]).astype(
                'float32'
            ) + 1j * np.random.random(
                [
                    3,
                ]
            ).astype(
                'float32'
            )
            self.input_numpy_dout = np.ones(
                [
                    1,
                ],
                dtype="float32",
            )
            self.input_numpy_ddx = np.ones(
                [
                    3,
                ],
                dtype="float32",
            )
            self.input_numpy_ddy = np.ones(
                [
                    3,
                ],
                dtype="float32",
            )
            self.input_numpy_ddx_conj = np.conjugate(self.input_numpy_ddx)
            self.input_numpy_ddy_conj = np.conjugate(self.input_numpy_ddy)
            self.input_numpy_dout_conj = np.conjugate(self.input_numpy_dout)

        init_data()
        d_x_expected = np.zeros(
            [
                3,
            ],
            dtype="float32",
        )
        d_y_expected = np.zeros(
            [
                3,
            ],
            dtype="float32",
        )
        d_dout_expected = np.matmul(
            self.input_numpy_ddy_conj,
            np.ones(
                [
                    3,
                ],
                dtype="float32",
            ),
        ) + np.matmul(
            self.input_numpy_ddx_conj,
            np.ones(
                [
                    3,
                ],
                dtype="float32",
            ),
        )
        d_ddx_expected = (
            np.ones(
                [
                    3,
                ],
                dtype="float32",
            )
            * self.input_numpy_dout_conj[0]
        )
        d_ddy_expected = (
            np.ones(
                [
                    3,
                ],
                dtype="float32",
            )
            * self.input_numpy_dout_conj[0]
        )
        expected_results = (
            d_x_expected,
            d_y_expected,
            d_dout_expected,
            d_ddx_expected,
            d_ddy_expected,
        )

        for place in self.places:
            paddle.device.set_device(place)
            actual_results = self.actual()
            for expected_result, actual_result in zip(
                expected_results, actual_results
            ):
                np.testing.assert_allclose(
                    expected_result, actual_result, rtol=1e-6
                )
'''


# d_ddout is none, d_dx is none, dtype is float32
class TestDygraphTripleGradMatmulcase3(TestCase):
    def setUp(self):
        self.input_numpy_x = None
        self.input_numpy_y = None
        self.input_numpy_dout = None
        self.input_numpy_ddx = None
        self.input_numpy_ddy = None
        self.places = ["cpu"]
        if paddle.is_compiled_with_cuda():
            self.places.append("gpu")

    def actual(self):
        x = paddle.to_tensor(
            self.input_numpy_x, stop_gradient=False, dtype='float32'
        )
        y = paddle.to_tensor(
            self.input_numpy_y, stop_gradient=False, dtype='float32'
        )
        out = paddle.matmul(x, y, False, False)

        dout = paddle.to_tensor(
            self.input_numpy_dout, stop_gradient=False, dtype='float32'
        )
        (dx, dy) = paddle.grad(
            [out], [x, y], [dout], retain_graph=True, create_graph=True
        )
        ddx = paddle.to_tensor(
            self.input_numpy_ddx, stop_gradient=False, dtype='float32'
        )
        ddy = paddle.to_tensor(
            self.input_numpy_ddy, stop_gradient=False, dtype='float32'
        )
        (dy_double_grad,) = paddle.grad(
            [dx, dy],
            [y],
            [ddx, ddy],
            retain_graph=True,
            create_graph=True,
        )
        # d_x d_y is None because (double grad out_put ddout grad tensor)d_ddout is None
        # d_ddy is None because (double grad out_put dx grad tensor) d_dx and d_ddout is None
        d_dout, d_ddx = paddle.grad(
            [dy_double_grad],
            [dout, ddx],
            retain_graph=False,
            create_graph=False,
        )
        return d_dout, d_ddx

    # case1: d_ddout is none, d_dx is none, dims != 1
    def test_matmul_triple_grad_case1(self):
        def init_data():
            self.input_numpy_x = np.random.random([3, 3]).astype('float32')
            self.input_numpy_y = np.random.random([3, 3]).astype('float32')
            self.input_numpy_dout = np.ones([3, 3], dtype="float32")
            self.input_numpy_ddx = np.ones([3, 3], dtype="float32")
            self.input_numpy_ddy = np.ones([3, 3], dtype="float32")

        init_data()
        d_dout_expected = np.ones([3, 3], dtype="float32") * 3
        d_ddx_expected = np.ones([3, 3], dtype="float32") * 3
        expected_results = (
            d_dout_expected,
            d_ddx_expected,
        )

        for place in self.places:
            paddle.device.set_device(place)
            actual_results = self.actual()
            for expected_result, actual_result in zip(
                expected_results, actual_results
            ):
                np.testing.assert_allclose(
                    expected_result, actual_result, rtol=1e-6
                )

    # #case2: d_ddout is none, d_dx is none, dims = 1
    def test_matmul_triple_grad_case2(self):
        def init_data():
            self.input_numpy_x = np.random.random(
                [
                    3,
                ]
            ).astype('float32')
            self.input_numpy_y = np.random.random(
                [
                    3,
                ]
            ).astype('float32')
            self.input_numpy_dout = np.ones([1], dtype="float32")
            self.input_numpy_ddx = np.ones([3], dtype="float32")
            self.input_numpy_ddy = np.ones([3], dtype="float32")

        init_data()
        d_dout_expected = np.ones([1], dtype="float32") * 3
        d_ddx_expected = np.ones(
            [
                3,
            ],
            dtype="float32",
        )
        expected_results = (
            d_dout_expected,
            d_ddx_expected,
        )

        for place in self.places:
            paddle.device.set_device(place)
            actual_results = self.actual()
            for expected_result, actual_result in zip(
                expected_results, actual_results
            ):
                np.testing.assert_allclose(
                    expected_result, actual_result, rtol=1e-6
                )

    # #case3: d_ddout is none, d_dx is none , with broadcast
    def test_matmul_triple_grad_case3(self):
        def init_data():
            self.input_numpy_x = np.random.random([3, 1]).astype('float32')
            self.input_numpy_y = np.random.random(
                [
                    1,
                ]
            ).astype('float32')
            self.input_numpy_dout = np.ones([3], dtype="float32")
            self.input_numpy_ddx = np.ones([3, 1], dtype="float32")
            self.input_numpy_ddy = np.ones([1], dtype="float32")

        init_data()
        d_dout_expected = np.ones(
            [
                3,
            ],
            dtype="float32",
        )
        d_ddx_expected = np.ones([3, 1], dtype="float32")
        expected_results = (
            d_dout_expected,
            d_ddx_expected,
        )

        for place in self.places:
            paddle.device.set_device(place)
            actual_results = self.actual()
            for expected_result, actual_result in zip(
                expected_results, actual_results
            ):
                np.testing.assert_allclose(
                    expected_result, actual_result, rtol=1e-6
                )


'''
# d_ddout is none, d_dx is none, dtype is complex64
class TestDygraphTripleGradMatmulcase4(TestCase):
    def setUp(self):
        self.input_numpy_x = None
        self.input_numpy_y = None
        self.input_numpy_dout = None
        self.input_numpy_ddx = None
        self.input_numpy_ddy = None
        self.input_numpy_ddx_conj = None
        self.input_numpy_dout_conj = None
        self.places = ["cpu"]
        if paddle.is_compiled_with_cuda():
            self.places.append("gpu")

    def actual(self):
        x = paddle.to_tensor(
            self.input_numpy_x, stop_gradient=False, dtype='complex64'
        )
        y = paddle.to_tensor(
            self.input_numpy_y, stop_gradient=False, dtype='complex64'
        )
        out = paddle.matmul(x, y, False, False)

        dout = paddle.to_tensor(
            self.input_numpy_dout, stop_gradient=False, dtype='complex64'
        )
        (dx, dy) = paddle.grad(
            [out], [x, y], [dout], retain_graph=True, create_graph=True
        )
        ddx = paddle.to_tensor(
            self.input_numpy_ddx, stop_gradient=False, dtype='complex64'
        )
        ddy = paddle.to_tensor(
            self.input_numpy_ddy, stop_gradient=False, dtype='complex64'
        )
        (dy_double_grad,) = paddle.grad(
            [dx, dy],
            [y],
            [ddx, ddy],
            retain_graph=True,
            create_graph=True,
        )
        d_x, d_y, d_dout, d_ddx, d_ddy = paddle.grad(
            [dy_double_grad],
            [x, y, dout, ddx, ddy],
            retain_graph=False,
            create_graph=False,
        )
        return d_x, d_y, d_dout, d_ddx, d_ddy

    # case1: no d_ddout,no d_dx, dims = 1
    def test_matmul_triple_grad_case1(self):
        def init_data():
            self.input_numpy_x = np.random.random([3]).astype(
                'float32'
            ) + 1j * np.random.random(
                [
                    3,
                ]
            ).astype(
                'float32'
            )
            self.input_numpy_y = np.random.random([3]).astype(
                'float32'
            ) + 1j * np.random.random(
                [
                    3,
                ]
            ).astype(
                'float32'
            )
            self.input_numpy_dout = np.ones(
                [
                    1,
                ],
                dtype="float32",
            )
            self.input_numpy_ddx = np.ones(
                [
                    3,
                ],
                dtype="float32",
            )
            self.input_numpy_ddy = np.ones(
                [
                    3,
                ],
                dtype="float32",
            )
            self.input_numpy_ddx_conj = np.conjugate(self.input_numpy_ddx)
            self.input_numpy_dout_conj = np.conjugate(self.input_numpy_dout)

        init_data()
        d_x_expected = np.zeros(
            [
                3,
            ],
            dtype="float32",
        )
        d_y_expected = np.zeros(
            [
                3,
            ],
            dtype="float32",
        )
        d_dout_expected = np.matmul(
            self.input_numpy_ddx_conj,
            np.ones(
                [
                    3,
                ],
                dtype="float32",
            ),
        )
        d_ddx_expected = (
            np.ones(
                [
                    3,
                ],
                dtype="float32",
            )
            * self.input_numpy_dout_conj[0]
        )
        d_ddy_expected = np.zeros(
            [
                3,
            ],
            dtype="float32",
        )
        expected_results = (
            d_x_expected,
            d_y_expected,
            d_dout_expected,
            d_ddx_expected,
            d_ddy_expected,
        )

        for place in self.places:
            paddle.device.set_device(place)
            actual_results = self.actual()
            for expected_result, actual_result in zip(
                expected_results, actual_results
            ):
                np.testing.assert_allclose(
                    expected_result, actual_result, rtol=1e-6
                )
'''


# d_ddout is none, d_dy is none, dtype is float32
class TestDygraphTripleGradMatmulcase5(TestCase):
    def setUp(self):
        self.input_numpy_x = None
        self.input_numpy_y = None
        self.input_numpy_dout = None
        self.input_numpy_ddx = None
        self.input_numpy_ddy = None
        self.places = ["cpu"]
        if paddle.is_compiled_with_cuda():
            self.places.append("gpu")

    def actual(self):
        x = paddle.to_tensor(
            self.input_numpy_x, stop_gradient=False, dtype='float32'
        )
        y = paddle.to_tensor(
            self.input_numpy_y, stop_gradient=False, dtype='float32'
        )
        out = paddle.matmul(x, y, False, False)

        dout = paddle.to_tensor(
            self.input_numpy_dout, stop_gradient=False, dtype='float32'
        )
        (dx, dy) = paddle.grad(
            [out], [x, y], [dout], retain_graph=True, create_graph=True
        )
        ddx = paddle.to_tensor(
            self.input_numpy_ddx, stop_gradient=False, dtype='float32'
        )
        ddy = paddle.to_tensor(
            self.input_numpy_ddy, stop_gradient=False, dtype='float32'
        )
        (dx_double_grad,) = paddle.grad(
            [dx, dy],
            [x],
            [ddx, ddy],
            retain_graph=True,
            create_graph=True,
        )
        d_dout, d_ddy = paddle.grad(
            [dx_double_grad],
            [dout, ddy],
            retain_graph=False,
            create_graph=False,
        )
        return d_dout, d_ddy

    # case1: d_ddout is none, d_dy is none, dims != 1
    def test_matmul_triple_grad_case1(self):
        def init_data():
            self.input_numpy_x = np.random.random([3, 3]).astype('float32')
            self.input_numpy_y = np.random.random([3, 3]).astype('float32')
            self.input_numpy_dout = np.ones([3, 3], dtype="float32")
            self.input_numpy_ddx = np.ones([3, 3], dtype="float32")
            self.input_numpy_ddy = np.ones([3, 3], dtype="float32")

        init_data()
        d_dout_expected = np.ones([3, 3], dtype="float32") * 3
        d_ddy_expected = np.ones([3, 3], dtype="float32") * 3
        expected_results = (
            d_dout_expected,
            d_ddy_expected,
        )

        for place in self.places:
            paddle.device.set_device(place)
            actual_results = self.actual()
            for expected_result, actual_result in zip(
                expected_results, actual_results
            ):
                np.testing.assert_allclose(
                    expected_result, actual_result, rtol=1e-6
                )

    # #case2: d_ddout is none, d_dy is none, dims = 1
    def test_matmul_triple_grad_case2(self):
        def init_data():
            self.input_numpy_x = np.random.random(
                [
                    3,
                ]
            ).astype('float32')
            self.input_numpy_y = np.random.random(
                [
                    3,
                ]
            ).astype('float32')
            self.input_numpy_dout = np.ones([1], dtype="float32")
            self.input_numpy_ddx = np.ones([3], dtype="float32")
            self.input_numpy_ddy = np.ones([3], dtype="float32")

        init_data()
        d_dout_expected = np.ones([1], dtype="float32") * 3
        d_ddy_expected = np.ones(
            [
                3,
            ],
            dtype="float32",
        )
        expected_results = (
            d_dout_expected,
            d_ddy_expected,
        )

        for place in self.places:
            paddle.device.set_device(place)
            actual_results = self.actual()
            for expected_result, actual_result in zip(
                expected_results, actual_results
            ):
                np.testing.assert_allclose(
                    expected_result, actual_result, rtol=1e-6
                )

    # #case3: d_ddout is none, d_dy is none , with broadcast
    def test_matmul_triple_grad_case3(self):
        def init_data():
            self.input_numpy_x = np.random.random([3, 1]).astype('float32')
            self.input_numpy_y = np.random.random(
                [
                    1,
                ]
            ).astype('float32')
            self.input_numpy_dout = np.ones([3], dtype="float32")
            self.input_numpy_ddx = np.ones([3, 1], dtype="float32")
            self.input_numpy_ddy = np.ones([1], dtype="float32")

        init_data()
        d_dout_expected = np.ones(
            [
                3,
            ],
            dtype="float32",
        )
        d_ddy_expected = np.ones([1], dtype="float32") * 3
        expected_results = (
            d_dout_expected,
            d_ddy_expected,
        )

        for place in self.places:
            paddle.device.set_device(place)
            actual_results = self.actual()
            for expected_result, actual_result in zip(
                expected_results, actual_results
            ):
                np.testing.assert_allclose(
                    expected_result, actual_result, rtol=1e-6
                )


'''
TODO(Ruting) test complex dtype when composite api support
# d_ddout is none, d_dy is none, dtype is complex64
class TestDygraphTripleGradMatmulcase6(TestCase):
    def setUp(self):
        self.input_numpy_x = None
        self.input_numpy_y = None
        self.input_numpy_dout = None
        self.input_numpy_ddx = None
        self.input_numpy_ddy = None
        self.input_numpy_ddy_conj = None
        self.input_numpy_dout_conj = None
        self.places = ["cpu"]
        if paddle.is_compiled_with_cuda():
            self.places.append("gpu")

    def actual(self):
        x = paddle.to_tensor(
            self.input_numpy_x, stop_gradient=False, dtype='complex64'
        )
        y = paddle.to_tensor(
            self.input_numpy_y, stop_gradient=False, dtype='complex64'
        )
        out = paddle.matmul(x, y, False, False)

        dout = paddle.to_tensor(
            self.input_numpy_dout, stop_gradient=False, dtype='complex64'
        )
        (dx, dy) = paddle.grad(
            [out], [x, y], [dout], retain_graph=True, create_graph=True
        )
        ddx = paddle.to_tensor(
            self.input_numpy_ddx, stop_gradient=False, dtype='complex64'
        )
        ddy = paddle.to_tensor(
            self.input_numpy_ddy, stop_gradient=False, dtype='complex64'
        )
        (dx_double_grad,) = paddle.grad(
            [dx, dy],
            [x],
            [ddx, ddy],
            retain_graph=True,
            create_graph=True,
        )
        d_x, d_y, d_dout, d_ddx, d_ddy = paddle.grad(
            [dx_double_grad],
            [x, y, dout, ddx, ddy],
            retain_graph=False,
            create_graph=False,
        )
        return d_x, d_y, d_dout, d_ddx, d_ddy

    # case1: no d_ddout,no d_dy, dims = 1
    def test_matmul_triple_grad_case1(self):
        def init_data():
            self.input_numpy_x = np.random.random([3]).astype(
                'float32'
            ) + 1j * np.random.random(
                [
                    3,
                ]
            ).astype(
                'float32'
            )
            self.input_numpy_y = np.random.random([3]).astype(
                'float32'
            ) + 1j * np.random.random(
                [
                    3,
                ]
            ).astype(
                'float32'
            )
            self.input_numpy_dout = np.ones(
                [
                    1,
                ],
                dtype="float32",
            )
            self.input_numpy_ddx = np.ones(
                [
                    3,
                ],
                dtype="float32",
            )
            self.input_numpy_ddy = np.ones(
                [
                    3,
                ],
                dtype="float32",
            )
            self.input_numpy_ddy_conj = np.conjugate(self.input_numpy_ddy)
            self.input_numpy_dout_conj = np.conjugate(self.input_numpy_dout)

        init_data()
        d_x_expected = np.zeros(
            [
                3,
            ],
            dtype="float32",
        )
        d_y_expected = np.zeros(
            [
                3,
            ],
            dtype="float32",
        )
        d_dout_expected = np.matmul(
            self.input_numpy_ddy_conj,
            np.ones(
                [
                    3,
                ],
                dtype="float32",
            ),
        )
        d_ddx_expected = np.zeros(
            [
                3,
            ],
            dtype="float32",
        )
        d_ddy_expected = (
            np.ones(
                [
                    3,
                ],
                dtype="float32",
            )
            * self.input_numpy_dout_conj[0]
        )
        expected_results = (
            d_x_expected,
            d_y_expected,
            d_dout_expected,
            d_ddx_expected,
            d_ddy_expected,
        )

        for place in self.places:
            paddle.device.set_device(place)
            actual_results = self.actual()
            for expected_result, actual_result in zip(
                expected_results, actual_results
            ):
                np.testing.assert_allclose(
                    expected_result, actual_result, rtol=1e-6
                )
'''

if __name__ == '__main__':
    unittest.main()
