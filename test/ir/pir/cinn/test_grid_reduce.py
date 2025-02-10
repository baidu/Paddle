# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import numpy
import utils

import paddle
from paddle.static import InputSpec


class TestGridReduce(unittest.TestCase):
    def eval(self, dy_compute, init_inputs, input_spec=None):
        paddle.seed(2024)
        inputs = init_inputs()
        dy_out = dy_compute(*inputs)

        static_compute = utils.apply_to_static(
            dy_compute, use_cinn=True, input_spec=input_spec
        )
        st_out = static_compute(*inputs)

        for a, b in zip(
            paddle.utils.flatten(dy_out), paddle.utils.flatten(st_out)
        ):
            numpy.testing.assert_allclose(a, b, atol=1e-3, rtol=1e-4)

    def test_all_reduce(self):
        def func(x):
            return paddle.sum(x)

        def init():
            x = paddle.randn([32, 128, 256])
            return (x,)

        self.eval(func, init)

    def test_continuous_reduce(self):
        def func(x):
            return paddle.sum(x, axis=(0, 2, 3))

        def init():
            x = paddle.randn([64, 8, 56, 56])
            return (x,)

        self.eval(func, init)

    def test_discrete_reduce(self):
        def func(x):
            return paddle.sum(x, axis=(0, 1, 2))

        def init():
            x = paddle.randn([256, 28, 28, 80])
            return (x,)

        self.eval(func, init)

    def test_multiple_reduce(self):
        def func(x):
            n = 512 * 14 * 14
            sum_x = paddle.sum(x, axis=(0, 2, 3))
            sum_x2 = paddle.sum(x * x, axis=(0, 2, 3))
            mean_x = sum_x / n
            mean_x2 = sum_x2 / n
            mean_x_2 = mean_x * mean_x
            return mean_x2 - mean_x_2

        def init():
            x = paddle.randn([512, 8, 14, 14])
            return (x,)

        self.eval(func, init)

    def test_multiple_downstream(self):
        def func(a, b):
            a = paddle.sum(a, axis=(0, 1))
            return a, a + b

        def init():
            a = paddle.randn([400, 300, 20, 10])
            b = paddle.randn([20, 10])
            return a, b

        self.eval(func, init)

    def test_continuous_reduce_dynamic_rd(self):
        def func(x):
            return paddle.sum(x, axis=(0, 2, 3))

        def init():
            x = paddle.randn([500, 3, 20, 20])
            return (x,)

        input_spec = [InputSpec([-1, 3, -1, -1])]

        self.eval(func, init, input_spec)

    def test_discrete_reduce_dynamic_rd(self):
        def func(x):
            return paddle.sum(x, axis=(0, 1, 2))

        def init():
            x = paddle.randn([96, 15, 20, 100])
            return (x,)

        input_spec = [InputSpec([96, -1, -1, 100])]

        self.eval(func, init, input_spec)


if __name__ == "__main__":
    unittest.main()
