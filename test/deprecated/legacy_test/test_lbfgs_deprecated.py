# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.nn.functional as F
from paddle.incubate.optimizer.functional.lbfgs import minimize_lbfgs

np.random.seed(123)


def test_static_graph(func, x0, line_search_fn='strong_wolfe', dtype='float32'):
    dimension = x0.shape[0]
    paddle.enable_static()
    main = paddle.static.Program()
    startup = paddle.static.Program()
    with paddle.static.program_guard(main, startup):
        X = paddle.static.data(name='x', shape=[dimension], dtype=dtype)
        Y = minimize_lbfgs(func, X, line_search_fn=line_search_fn, dtype=dtype)

    exe = paddle.static.Executor()
    exe.run(startup)
    return exe.run(main, feed={'x': x0}, fetch_list=[Y])


def test_static_graph_H0(func, x0, H0, dtype='float32'):
    paddle.enable_static()
    main = paddle.static.Program()
    startup = paddle.static.Program()
    with paddle.static.program_guard(main, startup):
        X = paddle.static.data(name='x', shape=[x0.shape[0]], dtype=dtype)
        H = paddle.static.data(
            name='h', shape=[H0.shape[0], H0.shape[1]], dtype=dtype
        )
        Y = minimize_lbfgs(
            func, X, initial_inverse_hessian_estimate=H, dtype=dtype
        )

    exe = paddle.static.Executor()
    exe.run(startup)
    return exe.run(main, feed={'x': x0, 'h': H0}, fetch_list=[Y])


def test_dynamic_graph(
    func, x0, H0=None, line_search_fn='strong_wolfe', dtype='float32'
):
    paddle.disable_static()
    x0 = paddle.to_tensor(x0)
    if H0 is not None:
        H0 = paddle.to_tensor(H0)
    return minimize_lbfgs(
        func,
        x0,
        initial_inverse_hessian_estimate=H0,
        line_search_fn=line_search_fn,
        dtype=dtype,
    )


class TestLbfgs(unittest.TestCase):
    def test_quadratic_nd(self):
        for dimension in [1, 10]:
            minimum = np.random.random(size=[dimension]).astype('float32')
            scale = np.exp(np.random.random(size=[dimension]).astype('float32'))

            def func(x):
                minimum_ = paddle.assign(minimum)
                scale_ = paddle.assign(scale)
                return paddle.sum(
                    paddle.multiply(scale_, (F.square_error_cost(x, minimum_)))
                )

            x0 = np.random.random(size=[dimension]).astype('float32')
            results = test_static_graph(func, x0)
            np.testing.assert_allclose(minimum, results[2], rtol=1e-05)

            results = test_dynamic_graph(func, x0)
            np.testing.assert_allclose(minimum, results[2].numpy(), rtol=1e-05)

    def test_inf_minima(self):
        extreme_point = np.array([-1, 2]).astype('float32')

        def func(x):
            # df = 3(x - 1.01)(x - 0.99)
            # f = x^3 - 3x^2 + 3*1.01*0.99x
            return (
                x * x * x / 3.0
                - (extreme_point[0] + extreme_point[1]) * x * x / 2
                + extreme_point[0] * extreme_point[1] * x
            )

        x0 = np.array([-1.7]).astype('float32')
        results = test_static_graph(func, x0)
        self.assertFalse(results[0][0])

    def test_multi_minima(self):
        def func(x):
            # df = 12(x + 1.1)(x - 0.2)(x - 0.8)
            # f = 3*x^4+0.4*x^3-5.46*x^2+2.112*x
            # minimum = -1.1 or 0.8.
            # All these minima may be reached from appropriate starting points.
            return 3 * x**4 + 0.4 * x**3 - 5.64 * x**2 + 2.112 * x

        x0 = np.array([0.82], dtype='float64')

        results = test_static_graph(func, x0, dtype='float64')
        np.testing.assert_allclose(0.8, results[2], rtol=1e-05)

    def test_rosenbrock(self):
        # The Rosenbrock function is a standard optimization test case.
        a = np.random.random(size=[1]).astype('float32')
        minimum = [a.item(), (a**2).item()]
        b = np.random.random(size=[1]).astype('float32')

        def func(position):
            # f(x, y) = (a - x)^2 + b (y - x^2)^2
            # minimum = (a, a^2)
            x, y = position[0], position[1]
            c = (a - x) ** 2 + b * (y - x**2) ** 2
            # the return can't be np array[1], or in jacobin will cause flat error
            return c[0]

        x0 = np.random.random(size=[2]).astype('float32')

        results = test_dynamic_graph(func, x0)
        np.testing.assert_allclose(minimum, results[2], rtol=1e-05)

    def test_exception(self):
        def func(x):
            return paddle.dot(x, x)

        x0 = np.random.random(size=[2]).astype('float32')
        H0 = np.array([[2.0, 0.0], [0.0, 0.9]]).astype('float32')

        # test dtype is not float32 or float64
        x1 = np.random.random(size=[2]).astype('int32')
        self.assertRaises(
            ValueError, test_static_graph, func, x1, dtype='int32'
        )

        # test initial_inverse_hessian_estimate is good
        results = test_static_graph_H0(func, x0, H0, dtype='float32')
        np.testing.assert_allclose([0.0, 0.0], results[2], rtol=1e-05)
        self.assertTrue(results[0][0])

        # test initial_inverse_hessian_estimate is bad and float64
        x2 = np.random.random(size=[2]).astype('float64')
        H1 = np.array([[1.0, 2.0], [3.0, 1.0]]).astype('float64')
        self.assertRaises(
            ValueError, test_static_graph_H0, func, x2, H0=H1, dtype='float64'
        )


if __name__ == '__main__':
    unittest.main()
