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

import numpy as np
from test_case_base import TestCaseBase

import paddle
from paddle.jit.sot.utils import strict_mode_guard


def numpy_add(x, y):
    out = paddle.to_tensor(x.numpy() + y.numpy())
    return out


def tensor_add_numpy(x, y):
    ret = x + y
    return ret


def large_numpy_array_to_tensor(x):
    return paddle.to_tensor(x)


class TestNumpy(TestCaseBase):
    @strict_mode_guard(False)
    def test_numpy_add(self):
        x = paddle.to_tensor([2])
        y = paddle.to_tensor([3])
        self.assert_results(numpy_add, x, y)

    def test_tensor_add_numpy_number(self):
        x = paddle.to_tensor([1.0])
        y = np.int64(2)
        self.assert_results(tensor_add_numpy, x, y)
        self.assert_results(tensor_add_numpy, y, x)

    @strict_mode_guard(False)
    def test_tensor_add_numpy_array(self):
        x = paddle.to_tensor([1.0])
        y = np.array(2.0)
        self.assert_results(tensor_add_numpy, x, y)
        self.assert_results(tensor_add_numpy, y, x)

    def test_large_numpy_array_to_tensor(self):
        # size should be larger than 1024*1024, because we throw an exception
        # when the size is larger than 1024*1024 in assign API (to_tensor static branch)
        x = np.random.rand(1024, 1024, 2).astype(np.float32)
        self.assert_results(large_numpy_array_to_tensor, x)


if __name__ == "__main__":
    unittest.main()
