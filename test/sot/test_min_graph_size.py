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

from __future__ import annotations

import unittest

from test_case_base import TestCaseBase

import paddle
from paddle.jit import sot
from paddle.jit.sot.utils import min_graph_size_guard


def case_for(x, vars):
    x = x + 1
    sot.psdb.breakgraph()
    for y in vars:
        x += y
    return x


def case_if(x):
    x = x + 1
    if x > 5:
        x += 3
    else:
        x += 4
    return x


def case_call(x):
    y = paddle.to_tensor(x.numpy())
    x += y
    return x


def call_with_kwargs_inner(x):
    return paddle.to_tensor(x.numpy())


def call_with_kwargs(x):
    y = call_with_kwargs_inner(x=x)
    x += y
    return x


def case_all(x, vars):
    x = x + 1
    for y in vars:
        z = paddle.to_tensor(x.numpy())
        x += z
        x += y
        if x > 5:
            x += y
        else:
            x += 3
    return x


class CustomLayer(paddle.nn.Layer):
    def forward(self, x):
        return self.forward_features(x)

    def forward_features(self, x):
        return x.numpy()


class TestMinGraphSize(TestCaseBase):
    @min_graph_size_guard(10)
    def test_cases(self):
        x = paddle.to_tensor(1)
        self.assert_results(case_for, x, [1, 2, 3])
        self.assert_results(case_if, x)
        self.assert_results(case_call, x)
        self.assert_results(case_all, x, [4, 5, 6])

    @min_graph_size_guard(10)
    def test_layer(self):
        x = paddle.to_tensor(1)
        layer = CustomLayer()
        self.assert_results(layer.forward, x)

    @min_graph_size_guard(10)
    def test_layer_multiple_times(self):
        x = paddle.to_tensor(1)
        layer = CustomLayer()
        for _ in range(8):
            self.assert_results(layer.forward, x)

    @min_graph_size_guard(10)
    def test_call_with_kwargs(self):
        x = paddle.to_tensor(1)
        self.assert_results(call_with_kwargs, x)


if __name__ == "__main__":
    unittest.main()
