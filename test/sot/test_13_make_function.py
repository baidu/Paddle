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

# MAKE_FUNCTION
# CALL_FUNCTION_KW
from __future__ import annotations

import unittest

from test_case_base import TestCaseBase

import paddle


def make_fn_simple(x: paddle.Tensor):
    def fn(a, b, c, d):
        return a + b + c + d

    return fn(1, 2, 3, 4) + x


def make_fn_default(x: paddle.Tensor):
    def fn(a, b, c=5, d=3):
        return a + b + c + d

    return fn(1, 2) + fn(1, 2, c=3) + x


def make_fn_annotation(x: paddle.Tensor):
    def fn(a, b: int, c: int, d):
        return a + b + c + d

    return fn(1, 2, 3, 4) + x


def make_fn_kwdefault(x: paddle.Tensor):
    def fn(a, b, *, c=3, d=4):
        return a + b + c + d

    return fn(1, 2) + fn(3, 4, c=1, d=2) + x


def make_fn_closure(x: paddle.Tensor):
    def fn(a, b, c, d):
        y = x
        return a + b + c + d

    return fn(1, 2, 3, 4) + x


def make_fn_mix(x: paddle.Tensor):
    def fn(a: int = 1, b: float = 2.0, /, *, c: int = 4, d: float = 5):
        # y = x
        return a + b + c + d

    return fn(2, 3, c=1, d=2.0) + x


def make_fn_tensor_default(x: paddle.Tensor):
    tensor = paddle.to_tensor(1.0)

    def fn(a, b, c=tensor):
        return a + b + c

    return fn(1, 2) + x


def make_fn_tensor_kwdefault(x: paddle.Tensor):
    tensor = paddle.to_tensor(1.0)

    def fn(*args, c=tensor):
        return args[0] + args[1] + c

    return fn(1, 2, c=3) + x


class TestMakeFunction(TestCaseBase):
    def test_simple(self):
        self.assert_results(make_fn_simple, paddle.to_tensor(1))
        self.assert_results(make_fn_default, paddle.to_tensor(1))
        self.assert_results(make_fn_annotation, paddle.to_tensor(1))
        self.assert_results(make_fn_kwdefault, paddle.to_tensor(1))
        self.assert_results(make_fn_closure, paddle.to_tensor(1))
        self.assert_results(make_fn_mix, paddle.to_tensor(1))

    def test_tensor_default(self):
        self.assert_results(make_fn_tensor_default, paddle.to_tensor(1))
        self.assert_results(make_fn_tensor_kwdefault, paddle.to_tensor(1))


if __name__ == "__main__":
    unittest.main()
