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

from __future__ import annotations

import math
import unittest

from test_case_base import (
    TestCaseBase,
    test_instruction_translator_cache_context,
)

import paddle
from paddle.jit.sot.psdb import check_no_breakgraph
from paddle.jit.sot.utils import (
    allow_dynamic_shape_guard,
)


def dynamic_shape_input_func1(x):
    s = x.shape[0]
    return x + s


def dynamic_int_input_func1(x, n):
    x = paddle.reshape(x, [n, -1])
    return (x + n) * 2 - 1, (-n + 1) * 2 - 1, type(n) is int


def dynamic_int_input_func2(x, n):
    return x + n[1]


def dynamic_int_input_func3(x, n):
    if n < 4:
        return 1
    x = paddle.reshape(x, [n, -1])
    return (x + n) * 2 - 1, (-n + 1) * 2 - 1


def dynamic_shape_access_inner_var_shape(x):
    y = x + 1
    return y.shape[0]


def dynamic_shape_in_list(x, shape):
    return x.reshape(shape)


def dynamic_shape_int_mul_float(x):
    y = x * 0.5
    z = math.sin(y)  # Trigger get_py_value
    return z


class CustomConv(paddle.nn.Conv2D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @paddle.jit.to_static(full_graph=False)
    def forward(self, x):
        return paddle.nn.functional.conv2d(
            x,
            self.weight,
            self.bias,
            [self._stride[0] + 1, self._stride[1]],
            self._padding,
            self._dilation,
            self._groups,
            self._data_format,
        )


def pool2d_fallback(x, kernel_size):
    return paddle.nn.functional.max_pool2d(x, kernel_size=kernel_size)


class TestOpcodeExecutorDynamicShapeCache(TestCaseBase):
    def test_dynamic_int_input_cache_hit_case1(self):
        with allow_dynamic_shape_guard(
            True
        ), test_instruction_translator_cache_context() as ctx:
            self.assert_results(
                dynamic_int_input_func1, paddle.randn([3, 4, 5]), 1
            )
            self.assertEqual(ctx.translate_count, 1)
            for i in range(2, 6):
                self.assert_results(
                    dynamic_int_input_func1, paddle.randn([3, 4, 5]), i
                )
                self.assertEqual(ctx.translate_count, 2)

    def test_dynamic_int_input_cache_hit_case2(self):
        with allow_dynamic_shape_guard(
            True
        ), test_instruction_translator_cache_context() as ctx:
            self.assert_results(
                dynamic_int_input_func2, paddle.randn([3, 4, 5]), {1: 1}
            )
            self.assertEqual(ctx.translate_count, 1)
            for i in range(2, 6):
                self.assert_results(
                    dynamic_int_input_func2, paddle.randn([3, 4, 5]), {1: i}
                )
                self.assertEqual(ctx.translate_count, 2)

    def test_dynamic_int_input_cache_hit_case3(self):
        with allow_dynamic_shape_guard(
            True
        ), test_instruction_translator_cache_context() as ctx:
            for i in range(0, 6):
                self.assert_results(
                    dynamic_int_input_func3, paddle.randn([3, 4, 5]), i
                )
                self.assertEqual(ctx.translate_count, i + 1)

    def test_dynamic_shape_input_cache_hit_case1(self):
        with allow_dynamic_shape_guard(
            True
        ), test_instruction_translator_cache_context() as ctx:
            self.assert_results(
                dynamic_shape_input_func1, paddle.randn([1, 4, 5])
            )
            self.assertEqual(ctx.translate_count, 1)
            for i in range(2, 6):
                self.assert_results(
                    dynamic_shape_input_func1, paddle.randn([i, 4, 5])
                )
                self.assertEqual(ctx.translate_count, 2)

    def test_dynamic_shape_input_cache_hit_case2(self):
        with allow_dynamic_shape_guard(
            True
        ), test_instruction_translator_cache_context() as ctx:
            self.assert_results(
                dynamic_shape_access_inner_var_shape, paddle.randn([1, 4, 5])
            )
            self.assertEqual(ctx.translate_count, 1)
            for i in range(2, 6):
                self.assert_results(
                    dynamic_shape_access_inner_var_shape,
                    paddle.randn([i, 4, 5]),
                )
                self.assertEqual(ctx.translate_count, 2)

    def test_dynamic_shape_cast(self):
        with allow_dynamic_shape_guard(
            True
        ), test_instruction_translator_cache_context() as ctx:
            func1 = check_no_breakgraph(lambda n: bool(n))
            func2 = check_no_breakgraph(lambda n: int(n))
            func3 = check_no_breakgraph(lambda n: float(n))
            for func in [func1, func2, func3]:
                self.assert_results(func, 1)
                self.assert_results(func, 2)

    def test_dynamic_shape_in_list(self):
        with allow_dynamic_shape_guard(
            True
        ), test_instruction_translator_cache_context() as ctx:
            self.assert_results(
                dynamic_shape_in_list,
                paddle.randn([1, 4, 5]),
                [4, 5],
            )
            self.assertEqual(ctx.translate_count, 1)
            for i in range(2, 6):
                self.assert_results(
                    dynamic_shape_in_list,
                    paddle.randn([i, 4, 5]),
                    [i * 4, 5],
                )
                self.assertEqual(ctx.translate_count, 2)

    def test_conv_dynamic_shape_stride_fallback(self):
        with allow_dynamic_shape_guard(
            True
        ), test_instruction_translator_cache_context() as ctx:
            for i in range(1, 5):
                conv = CustomConv(3, 3, 3, stride=i)
                conv(paddle.randn([1, 3, 224, 224]))
                self.assertEqual(ctx.translate_count, i)

    def test_conv_dynamic_shape_kernel_size_fallback(self):
        with allow_dynamic_shape_guard(
            True
        ), test_instruction_translator_cache_context() as ctx:
            for i in range(1, 5):
                x = paddle.randn([1, 3, 224, 224])
                self.assert_results(pool2d_fallback, x, i)
                self.assertEqual(ctx.translate_count, i)

    def test_pad_dynamic_shape_fallback(self):
        with allow_dynamic_shape_guard(
            True
        ), test_instruction_translator_cache_context() as ctx:
            pad_func = check_no_breakgraph(
                lambda x, n: paddle.nn.functional.pad(x, [0, n, 0, 0])
            )
            for i in range(1, 5):
                self.assert_results(pad_func, paddle.randn([1, 3, 224, 224]), i)
                self.assertEqual(ctx.translate_count, i)

    def test_dynamic_shape_int_mul_float(self):
        with allow_dynamic_shape_guard(
            True
        ), test_instruction_translator_cache_context() as ctx:
            for i in range(1, 6):
                self.assert_results(dynamic_shape_int_mul_float, i)


if __name__ == '__main__':
    unittest.main()
