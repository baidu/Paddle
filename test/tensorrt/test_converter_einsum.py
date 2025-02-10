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

import numpy as np
from tensorrt_test_base import TensorRTBaseTest

import paddle


def einsum_wrapper(equation, x):
    if not isinstance(x, list):
        x = [x]
    out = paddle.einsum(equation, *x)
    return out[0]


class TestEinsumCase1TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = einsum_wrapper
        self.api_args = {
            "equation": "ijk->ij",
            "x": np.random.randn(2, 3, 4).astype("float32"),
        }
        self.program_config = {"feed_list": ["x"]}

        self.min_shape = {"x": [1, 3, 4]}
        self.opt_shape = {"x": [2, 3, 4]}
        self.max_shape = {"x": [4, 3, 4]}

    def test_trt_result(self):
        self.check_trt_result()


class TestEinsumCase2TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = einsum_wrapper
        self.api_args = {
            "equation": "abcd,bcd->a",
            "operands": [
                np.random.randn(2, 3, 4, 5).astype("float32"),
                np.random.randn(3, 4, 5).astype("float32"),
            ],
        }
        self.program_config = {"feed_list": ["operands"]}

        self.min_shape = {"operands_0": [1, 2, 3, 4], "operands_1": [2, 3, 4]}
        self.opt_shape = {"operands_0": [2, 3, 4, 5], "operands_1": [3, 4, 5]}
        self.max_shape = {"operands_0": [4, 6, 8, 9], "operands_1": [6, 8, 9]}

    def test_trt_result(self):
        self.check_trt_result()


class TestEinsumCaseTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = einsum_wrapper
        self.api_args = {
            "equation": "mij,jk->ki",
            "operands": [
                np.random.randn(2, 3, 4).astype("float16"),
                np.random.randn(4, 3).astype("float16"),
            ],
        }
        self.program_config = {"feed_list": ["operands"]}

        self.min_shape = {"operands_0": [1, 3, 4], "operands_1": [1, 3]}
        self.opt_shape = {"operands_0": [2, 3, 4], "operands_1": [4, 3]}
        self.max_shape = {"operands_0": [4, 3, 4], "operands_1": [6, 3]}

    def test_trt_result(self):
        self.check_trt_result(precision_mode="fp16")


if __name__ == '__main__':
    unittest.main()
