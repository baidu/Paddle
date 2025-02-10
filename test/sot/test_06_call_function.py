# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

from test_case_base import (
    TestCaseBase,
    test_instruction_translator_cache_context,
)

import paddle
from paddle.jit.sot.psdb import check_no_breakgraph


def add(x, y):
    return x + y


def sub(x, y):
    return x - y


def foo_1(x: paddle.Tensor):
    m = x + 1
    y = add(m * 3, m * 2)
    return y


def foo_2(x: paddle.Tensor):
    m = x + 1
    y = sub(m * 3, m * 2)
    return y


def foo_3(x: paddle.Tensor):
    m = x + 1
    y = sub(m * 3, m * 2)
    y = sub(y, y)
    y = sub(y, y)
    return y


def nest_2(x):
    return x + 1


def nest_1(x):
    return (x - 1) * 2


def foo_4(x: paddle.Tensor):
    m = x + 1
    m = nest_1(m)
    return m


def fn_with_varargs_and_kwargs(x, *args, **kwargs):
    return (
        x
        + args[0]
        + args[1]
        - args[2]
        + kwargs['a'] * kwargs['b']
        + kwargs['c']
    )


def foo_5(x: paddle.Tensor):
    m = x + 1
    m = fn_with_varargs_and_kwargs(
        m, x + 1, x + 2, x + 3, a=x + 4, b=x + 5, c=x + 6
    )
    return m


def fn_with_default_value(x, y=1, z=2):
    return x + y + z


def foo_6(x: paddle.Tensor):
    m = x + 1
    m = fn_with_default_value(m, m + 10)
    m = fn_with_default_value(m + 42)
    return m


def fn_with_default_value_and_varargs_kwargs(x, y=1, *args, **kwargs):
    return x + y + args[0] + kwargs['a']


def foo_7(x: paddle.Tensor):
    m = x + 1
    m = fn_with_default_value_and_varargs_kwargs(m, m + 1, m + 2, a=m + 3)
    return m


def fn_with_default_value_and_varargs_kwargs_kwonly_1(
    x, y=1, *args, z, **kwargs
):
    return x + y + args[0] + kwargs['a'] + z


def fn_with_default_value_and_varargs_kwargs_kwonly_2(
    x, y=1, *args, z=10, **kwargs
):
    return x + y + args[0] + kwargs['a'] + z


def foo_8(x: paddle.Tensor):
    m = x + 1
    m = fn_with_default_value_and_varargs_kwargs_kwonly_1(
        m, m + 1, m + 2, a=m + 3, z=m + 4
    )
    m = fn_with_default_value_and_varargs_kwargs_kwonly_2(
        m, m + 1, m + 2, a=m + 3
    )
    return m


class TestCall(TestCaseBase):
    def test_call1(self):
        self.assert_results(foo_1, paddle.to_tensor(2))

    def test_call2(self):
        self.assert_results(foo_2, paddle.to_tensor(3))

    def test_call3(self):
        self.assert_results(foo_3, paddle.to_tensor(4))

    def test_call4(self):
        self.assert_results(foo_4, paddle.to_tensor(5))

    def test_call5(self):
        self.assert_results(foo_5, paddle.to_tensor(6))

    def test_call6(self):
        self.assert_results(foo_6, paddle.to_tensor(7))

    def test_call7(self):
        self.assert_results(foo_7, paddle.to_tensor(8))

    def test_call8(self):
        self.assert_results(foo_8, paddle.to_tensor(9))


def apply_fn(fn, x):
    return fn(x)


def fn1(x):
    return x + 1


def fn2(x):
    return x - 1


class TestApplyDifferentFunctions(TestCaseBase):
    def test_apply_fn(self):
        x = 1
        with test_instruction_translator_cache_context() as ctx:
            self.assertEqual(ctx.translate_count, 0)
            self.assert_results(apply_fn, fn1, x)
            self.assertEqual(ctx.translate_count, 1)
            self.assert_results(apply_fn, fn2, x)
            self.assertEqual(ctx.translate_count, 2)
            self.assert_results(apply_fn, fn1, x)
            self.assertEqual(ctx.translate_count, 2)


@check_no_breakgraph
def positional_only_basic(x, /, y):
    z = x + y
    return x + z


def positional_only_breakgraph(x, /, y):
    z = x + y
    paddle.jit.sot.psdb.breakgraph()
    return x + z


@check_no_breakgraph
def keyword_only_basic(x, *, y):
    z = x + y
    return x + z


def keyword_only_breakgraph(x, *, y):
    z = x + y
    paddle.jit.sot.psdb.breakgraph()
    return x + z


class TestPositionalKeywordOnly(TestCaseBase):
    def test_positional_only_basic(self):
        self.assert_results(positional_only_basic, 1, 2)

    def test_positional_only_breakgraph(self):
        self.assert_results(positional_only_breakgraph, 1, 2)

    def test_keyword_only_basic(self):
        self.assert_results(keyword_only_basic, x=1, y=2)

    def test_keyword_only_breakgraph(self):
        self.assert_results(keyword_only_breakgraph, x=1, y=2)


if __name__ == "__main__":
    unittest.main()
