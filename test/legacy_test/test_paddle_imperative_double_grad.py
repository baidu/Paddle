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
from unittest import TestCase

import numpy as np

import paddle
import paddle.nn.functional as F
from paddle import base
from paddle.base.wrapped_decorator import wrap_decorator


def _dygraph_guard_(func):
    def __impl__(*args, **kwargs):
        if paddle.in_dynamic_mode():
            return func(*args, **kwargs)
        else:
            with base.dygraph.guard():
                return func(*args, **kwargs)

    return __impl__


dygraph_guard = wrap_decorator(_dygraph_guard_)


def random_var(size, low=-1, high=1, dtype='float32'):
    x_np = np.random.uniform(low=low, high=high, size=size).astype(dtype)
    return paddle.to_tensor(x_np)


class TestDygraphDoubleGrad(TestCase):
    def setUp(self):
        self.sort_sum_gradient = False
        self.shape = [5, 10]

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
        return paddle.grad(
            outputs=outputs,
            inputs=inputs,
            grad_outputs=grad_outputs,
            no_grad_vars=no_grad_vars,
            retain_graph=retain_graph,
            create_graph=create_graph,
            allow_unused=allow_unused,
        )

    @dygraph_guard
    def test_exception(self):
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
    def test_simple_example(self):
        x = random_var(self.shape)
        x.stop_gradient = False
        y = x + 1

        for create_graph in [False, True]:
            (dx,) = self.grad(
                [x], [x], create_graph=create_graph, retain_graph=True
            )
            self.assertEqual(dx.shape, x.shape)
            self.assertTrue(np.all(dx.numpy() == 1))
            self.assertNotEqual(dx.stop_gradient, create_graph)

            (dx_mul_2,) = self.grad(
                [y, x], [x], create_graph=create_graph, retain_graph=True
            )
            self.assertEqual(dx_mul_2.shape, x.shape)
            self.assertTrue(np.all(dx_mul_2.numpy() == 2))
            self.assertNotEqual(dx_mul_2.stop_gradient, create_graph)

            (none_grad,) = self.grad(
                [x], [y], create_graph=create_graph, allow_unused=True
            )
            self.assertIsNone(none_grad)

            (grad_with_none_and_not_none,) = self.grad(
                [x, y], [y], create_graph=create_graph
            )
            self.assertTrue(grad_with_none_and_not_none.shape, x.shape)
            self.assertTrue(np.all(grad_with_none_and_not_none.numpy() == 1))
            self.assertNotEqual(
                grad_with_none_and_not_none.stop_gradient, create_graph
            )

    @dygraph_guard
    def test_none_one_initial_gradient(self):
        numel = 1
        for s in self.shape:
            numel *= s

        half_numel = int(numel / 2)
        half_x_positive = np.random.uniform(low=1, high=2, size=[half_numel])
        half_x_negative = np.random.uniform(
            low=-2, high=-1, size=[numel - half_numel]
        )
        x_np = np.array(list(half_x_positive) + list(half_x_negative)).astype(
            'float32'
        )
        np.random.shuffle(x_np)

        x = paddle.to_tensor(x_np)
        x.stop_gradient = False

        alpha = 0.2
        y = paddle.nn.functional.leaky_relu(x, alpha)
        y = y * y
        z = y * y

        x_np = x.numpy()
        relu_x_np = np.maximum(x_np, alpha * x_np).astype('float32')
        relu_x_grad_np = ((x_np > 0) + (x_np < 0) * alpha).astype('float32')
        dy_expected = (relu_x_np * relu_x_grad_np * 2).astype('float32')
        dz_expected = (np.power(relu_x_np, 3) * relu_x_grad_np * 4).astype(
            'float32'
        )

        random_grad_y = random_var(y.shape, low=1, high=2)
        random_grad_z = random_var(z.shape, low=1, high=2)
        ones_grad_y = np.ones(y.shape).astype('float32')
        ones_grad_z = np.ones(z.shape).astype('float32')

        original_random_grad_y = random_grad_y.numpy()
        original_random_grad_z = random_grad_z.numpy()

        for grad_y in [random_grad_y]:
            for grad_z in [random_grad_z]:
                for create_graph in [False, True]:
                    (dx_actual,) = self.grad(
                        outputs=[y, z],
                        inputs=[x],
                        grad_outputs=[grad_y, grad_z],
                        create_graph=create_graph,
                        retain_graph=True,
                    )

                    grad_y_np = (
                        ones_grad_y if grad_y is None else grad_y.numpy()
                    )
                    grad_z_np = (
                        ones_grad_z if grad_z is None else grad_z.numpy()
                    )

                    dx_expected = (
                        dy_expected * grad_y_np + dz_expected * grad_z_np
                    )
                    np.testing.assert_allclose(
                        dx_actual.numpy(), dx_expected, rtol=1e-05
                    )

                    if grad_y is not None:
                        self.assertTrue(grad_y.stop_gradient)
                        np.testing.assert_array_equal(
                            grad_y.numpy(), original_random_grad_y
                        )

                    if grad_z is not None:
                        self.assertTrue(grad_z.stop_gradient)
                        np.testing.assert_array_equal(
                            grad_z.numpy(), original_random_grad_z
                        )

    @dygraph_guard
    def func_example_with_gradient_accumulation_and_create_graph(self):
        x = random_var(self.shape)
        x_np = x.numpy()
        numel = x_np.size
        x.stop_gradient = False

        y = F.relu(x)
        z = y + 1
        w = z * z

        w_mean = paddle.mean(w)
        del y, z, w

        (dx_actual,) = self.grad([w_mean], [x], create_graph=True)
        del w_mean

        self.assertFalse(dx_actual.stop_gradient)

        # Theoretical result based on math calculation
        dx_expected = (
            1.0 / float(numel) * (np.maximum(x_np, 0) + 1) * (x_np > 0) * 2
        ).astype('float32')
        np.testing.assert_allclose(dx_actual.numpy(), dx_expected, rtol=1e-05)

    @dygraph_guard
    def test_example_with_gradient_accumulation_and_no_grad_vars(self):
        x = random_var(self.shape)
        x_np = x.numpy()
        numel = x_np.size
        x.stop_gradient = False

        y1 = F.relu(x)
        y2 = F.relu(x)
        z = y1 + y2
        w = z * z

        w_mean = paddle.mean(w)
        del y1, z, w

        (dx_actual,) = self.grad(
            [w_mean], [x], create_graph=True, no_grad_vars=[y2]
        )

        self.assertFalse(y2.stop_gradient)
        self.assertFalse(dx_actual.stop_gradient)

        dx_expected = (
            1.0
            / float(numel)
            * (np.maximum(x_np, 0) + y2.numpy())
            * (x_np > 0)
            * 2
        ).astype('float32')
        np.testing.assert_allclose(dx_actual.numpy(), dx_expected, rtol=1e-05)

    @dygraph_guard
    def test_example_with_gradient_accumulation_and_not_create_graph(self):
        x = random_var(self.shape)
        x_np = x.numpy()
        numel = x_np.size
        x.stop_gradient = False

        y = F.relu(x)
        z = y + 1
        w = z * z

        w_mean = paddle.mean(w)
        del y, z, w

        (dx_actual,) = self.grad([w_mean], [x], create_graph=False)
        del w_mean

        self.assertTrue(dx_actual.stop_gradient)

        dx_expected = (
            1.0 / float(numel) * (np.maximum(x_np, 0) + 1) * (x_np > 0) * 2
        ).astype('float32')

        np.testing.assert_allclose(dx_actual.numpy(), dx_expected, rtol=1e-05)


class TestDygraphDoubleGradSortGradient(TestDygraphDoubleGrad):
    def setUp(self):
        self.sort_sum_gradient = True
        self.shape = [5, 10]


if __name__ == '__main__':
    unittest.main()
