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

from test_case_base import TestCaseBase, test_with_faster_guard

import paddle
from paddle.jit.sot.utils import strict_mode_guard


def test_enumerate_1(x: int, y: int):
    for id, val in enumerate(range(x)):
        if id % 2 == 0:
            y += val
    return y


def test_enumerate_2(x: list):
    return list(enumerate(x))


def test_enumerate_3(x: list):
    return tuple(enumerate(x))


def test_enumerate_4(x: paddle.Tensor):
    sum = 0
    for idx, val in enumerate(x):
        sum += val
    return sum


# TODO(zmh): support range for tensor
def test_enumerate_5(x: paddle.Tensor):
    sum = 0

    for idx, val in enumerate(x):
        for i in range(val):
            sum += val
    return sum


def test_enumerate_6(x: paddle.Tensor):
    sum = 0

    for idx, val in enumerate(x):
        for i in range(idx):
            sum += val
    return sum


def test_enumerate_7(x: paddle.Tensor):
    sum = 0
    x = x.flatten()
    for idx, val in enumerate(x):
        sum += val
    return sum


# TODO(zmh): support -1
def test_enumerate_8(x: paddle.Tensor):
    sum = 0
    x = paddle.nonzero(x, as_tuple=False)
    for idx, val in enumerate(x):
        sum += val
    return sum


def test_enumerate_10(layer_list, x):
    sum = 0
    for idx, layer in enumerate(layer_list):
        sum += layer(x)
    return sum


class TestEnumerate(TestCaseBase):
    @test_with_faster_guard
    def test_cases(self):
        x = 8
        y = 5
        ty = paddle.randn((10, 10))
        layer_list = paddle.nn.LayerList(
            [paddle.nn.Linear(10, 10) for _ in range(3)]
        )

        self.assert_results(test_enumerate_1, x, y)
        self.assert_results(test_enumerate_2, [2, 4, 6, 8, 10])
        self.assert_results(test_enumerate_3, [2, 4, 6, 8, 10])

        self.assert_results(test_enumerate_4, ty)
        # TODO(zmh): support range for tensor

        with strict_mode_guard(False):
            self.assert_results(test_enumerate_5, paddle.to_tensor([1, 2, 3]))
        self.assert_results(test_enumerate_6, paddle.to_tensor([1, 2, 3]))
        self.assert_results(test_enumerate_7, ty)
        # TODO(zmh): support -1

        with strict_mode_guard(False):
            self.assert_results(test_enumerate_8, ty)

        self.assert_results(test_enumerate_10, layer_list, paddle.randn((10,)))


if __name__ == "__main__":
    unittest.main()
