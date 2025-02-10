# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from dygraph_to_static_utils import (
    Dy2StTestBase,
    enable_to_static_guard,
    exe_sequential_run_guard,
    test_ast_only,
)

import paddle
from paddle.framework import use_pir_api
from paddle.jit.dy2static.utils import Dygraph2StaticException

SEED = 2020
np.random.seed(SEED)


class TestDy2staticException(Dy2StTestBase):
    def setUp(self):
        self.x = np.random.random([10, 16]).astype('float32')
        self.dyfunc = None
        self.error = "Your if/else have different number of return value."

    @test_ast_only
    def test_error(self):
        if self.dyfunc:
            with self.assertRaisesRegex(Dygraph2StaticException, self.error):
                with enable_to_static_guard(True):
                    self.assertTrue(paddle.jit.to_static(self.dyfunc)(self.x))


def test_continue_in_for(x):
    x = paddle.to_tensor(x)
    for i in range(10):
        x += 1
        if i > 5:
            continue
            x += 10086
        x += i
    return x


def test_continue_in_for_at_end(x):
    x = paddle.to_tensor(x)
    for i in range(10):
        x += 1
        if i > 5:
            continue
    return x


def test_continue_in_while(x):
    x = paddle.to_tensor(x)
    i = paddle.tensor.fill_constant(shape=[1], dtype='int64', value=0)
    while i < 10:
        i += 1
        if i > 5:
            continue
            x += 10086
        x += i
    return x


def test_break_in_for(x):
    x = paddle.to_tensor(x)
    for i in range(10):
        x += 1
        if i > 5:
            break
            x += 10086
        x += i
    return x


def test_break_in_for_at_end(x):
    x = paddle.to_tensor(x)
    for i in range(10):
        x += 1
        if i > 5:
            break
    return x


def test_break_in_while(x):
    x = paddle.to_tensor(x)
    i = paddle.tensor.fill_constant(shape=[1], dtype='int64', value=0)
    while i < 10:
        i += 1
        if i > 5:
            break
            x += 10086
        x += i
    return x


def test_break_continue_in_for(x):
    x = paddle.to_tensor(x)

    for i in range(1, 10, 1):
        if i <= 4:
            x += 1
            continue
        else:
            x += 10010
            break
        x += 10086

    a = paddle.tensor.fill_constant(shape=[1], dtype='int64', value=0)
    b = paddle.tensor.fill_constant(shape=[1], dtype='int64', value=3)
    # b = 10
    # TODO: add Raise Error and suggestion for usage:
    #   Py for contains break/continue depends on control-flow.
    for i in range(b):
        if a <= 4:
            x += 1
            a += 1
            continue
        else:
            x += 10010
            break
        x += 10086

    return x


def test_for_in_else(x):
    x = paddle.to_tensor(x)

    # Case 1:
    if False:
        pass
    else:
        for i in range(0, 10):
            if i > 5:
                x += 1
                break
            x += i

    # Case 2:
    if False:
        pass
    else:
        for i in range(0, 10):
            x += 1
            break
            x += i
    return x


def while_loop_class_var(x):
    class Foo:
        def __init__(self):
            self.a = 3
            self.b = 4
            self.c = 5

    foo = Foo()
    i = paddle.to_tensor(x)
    while i < 10:
        foo.b = paddle.zeros(shape=[1], dtype='float32')
        foo.c = foo.b + foo.a
        i += 1
        if foo.c < 0:
            continue
        if foo.c > 6:
            break
    return foo.c


def test_optim_break_in_for(x):
    x = paddle.to_tensor(x)
    for i in range(10):
        if x.sum() > 5:
            break
            x += 10086
        x += i
        if i < 3:
            x = x * 2
    return x


def test_optim_break_in_while(x):
    x = paddle.to_tensor(x)
    i = paddle.tensor.fill_constant(shape=[1], dtype='int64', value=0)
    while i < 10:
        if i > 5:
            break
            x += 10086
        x += i
        i += 1
    return x


class TestContinueBase(Dy2StTestBase):
    def setUp(self):
        self.input = np.zeros(1).astype('int64')

    def init_dygraph_func(self):
        raise NotImplementedError(
            "For Continue test should implement init_dygraph_func"
        )

    def run_dygraph_mode(self):
        res = self.dygraph_func(self.input)
        return res.numpy()

    def run_static_mode(self):
        res = paddle.jit.to_static(self.dygraph_func)(self.input)
        return res.numpy()


class TestContinueInFor(TestContinueBase):
    def init_dygraph_func(self):
        self.dygraph_func = test_continue_in_for

    def test_transformed_static_result(self):
        self.init_dygraph_func()
        dygraph_res = self.run_dygraph_mode()
        static_res = self.run_static_mode()
        np.testing.assert_allclose(
            dygraph_res,
            static_res,
            rtol=1e-05,
            err_msg=f'dygraph res is {dygraph_res}\nstatic_res is {static_res}',
        )


class TestContinueNotPirBase(TestContinueInFor):
    def test_transformed_static_result(self):
        self.init_dygraph_func()
        dygraph_res = self.run_dygraph_mode()
        static_res = self.run_static_mode()
        np.testing.assert_allclose(
            dygraph_res,
            static_res,
            rtol=1e-05,
            err_msg=f'dygraph res is {dygraph_res}\nstatic_res is {static_res}',
        )


class TestContinueInForAtEnd(TestContinueInFor):
    def init_dygraph_func(self):
        self.dygraph_func = test_continue_in_for_at_end


class TestBreakInFor(TestContinueInFor):
    def init_dygraph_func(self):
        self.dygraph_func = test_break_in_for


class TestBreakInForAtEnd(TestContinueInFor):
    def init_dygraph_func(self):
        self.dygraph_func = test_break_in_for_at_end


class TestBreakContinueInFor(TestContinueBase):
    def init_dygraph_func(self):
        self.dygraph_func = test_break_continue_in_for

    def test_transformed_static_result(self):
        self.init_dygraph_func()
        dygraph_res = self.run_dygraph_mode()
        static_res = self.run_static_mode()
        np.testing.assert_allclose(
            dygraph_res,
            static_res,
            rtol=1e-05,
            err_msg=f'dygraph res is {dygraph_res}\nstatic_res is {static_res}',
        )


class TestForInElse(TestContinueInFor):
    def init_dygraph_func(self):
        self.dygraph_func = test_for_in_else


class TestContinueInWhile(TestContinueNotPirBase):
    def init_dygraph_func(self):
        self.dygraph_func = test_continue_in_while

    def test_transformed_static_result(self):
        self.init_dygraph_func()
        dygraph_res = self.run_dygraph_mode()
        static_res = self.run_static_mode()
        np.testing.assert_allclose(
            dygraph_res,
            static_res,
            rtol=1e-05,
            err_msg=f'dygraph res is {dygraph_res}\nstatic_res is {static_res}',
        )


class TestBreakInWhile(TestContinueInWhile):
    def init_dygraph_func(self):
        self.dygraph_func = test_break_in_while

    def test_transformed_static_result(self):
        self.init_dygraph_func()
        dygraph_res = self.run_dygraph_mode()
        static_res = self.run_static_mode()
        np.testing.assert_allclose(
            dygraph_res,
            static_res,
            rtol=1e-05,
            err_msg=f'dygraph res is {dygraph_res}\nstatic_res is {static_res}',
        )


class TestWhileLoopClassVar(TestContinueInWhile):
    def init_dygraph_func(self):
        self.dygraph_func = while_loop_class_var

    def test_transformed_static_result(self):
        self.init_dygraph_func()
        dygraph_res = self.run_dygraph_mode()
        static_res = self.run_static_mode()
        np.testing.assert_allclose(
            dygraph_res,
            static_res,
            rtol=1e-05,
            err_msg=f'dygraph res is {dygraph_res}\nstatic_res is {static_res}',
        )


class TestOptimBreakInFor(TestDy2staticException):
    def setUp(self):
        self.x = np.random.random([10, 16]).astype('float32')
        self.dyfunc = test_optim_break_in_for
        self.error = (
            "In a loop using the `break`. In the case of motion to static"
        )


class TestOptimBreakInWhile(TestContinueInWhile):
    def init_dygraph_func(self):
        self.dygraph_func = test_optim_break_in_while

    def test_transformed_static_result(self):
        self.init_dygraph_func()
        dygraph_res = self.run_dygraph_mode()
        # NOTE(SigureMo): Temporarily run the test in sequential run mode to avoid dependency
        # on the execution order of the test cases.
        if use_pir_api():
            with exe_sequential_run_guard(True):
                static_res = self.run_static_mode()
        else:
            static_res = self.run_static_mode()
        np.testing.assert_allclose(
            dygraph_res,
            static_res,
            rtol=1e-05,
            err_msg=f'dygraph res is {dygraph_res}\nstatic_res is {static_res}',
        )


if __name__ == '__main__':
    unittest.main()
