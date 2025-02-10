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


class TestMatmulTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.matmul
        self.api_args = {
            "x": np.random.randn(2, 3).astype("float32"),
            "y": np.random.randn(3, 2).astype("float32"),
            "transpose_x": False,
            "transpose_y": False,
        }
        self.program_config = {"feed_list": ["x", "y"]}
        self.min_shape = {"x": [1, 3], "y": [3, 2]}
        self.opt_shape = {"x": [1, 3], "y": [3, 2]}
        self.max_shape = {"x": [5, 3], "y": [3, 2]}

    def test_trt_result(self):
        self.check_trt_result(rtol=1e-3, atol=1e-3)


class TestTransposeTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.transpose
        self.api_args = {
            "x": np.random.randn(2, 3, 4).astype("float32"),
            "perm": [1, 0, 2],
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3, 4]}
        self.opt_shape = {"x": [1, 3, 4]}
        self.max_shape = {"x": [5, 3, 4]}

    def test_trt_result(self):
        self.check_trt_result()


class TestBmmTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.bmm
        self.api_args = {
            "x": np.random.randn(2, 2, 3).astype("float32"),
            "y": np.random.randn(2, 3, 2).astype("float32"),
        }
        self.program_config = {"feed_list": ["x", "y"]}
        self.min_shape = {"x": [1, 2, 3], "y": [1, 3, 2]}
        self.opt_shape = {"x": [1, 2, 3], "y": [1, 3, 2]}
        self.max_shape = {"x": [5, 2, 3], "y": [5, 3, 2]}

    def test_trt_result(self):
        self.check_trt_result()


class TestFlipTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.flip
        self.api_args = {
            "x": np.random.randn(2, 3, 4).astype("float32"),
            "axis": [0, 2],
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3, 4]}
        self.opt_shape = {"x": [1, 3, 4]}
        self.max_shape = {"x": [5, 3, 4]}

    def test_trt_result(self):
        self.check_trt_result()


class TestFlipNegAxisTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.flip
        self.api_args = {
            "x": np.random.randn(2, 3, 4).astype("float32"),
            "axis": [-1, -3],
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3, 4]}
        self.opt_shape = {"x": [1, 3, 4]}
        self.max_shape = {"x": [5, 3, 4]}

    def test_trt_result(self):
        self.check_trt_result()


class TestFlipIntTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.flip
        self.api_args = {
            "x": np.random.randn(2, 3, 4).astype("int64"),
            "axis": [0, 2],
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3, 4]}
        self.opt_shape = {"x": [1, 3, 4]}
        self.max_shape = {"x": [5, 3, 4]}

    def test_trt_result(self):
        self.check_trt_result()


class TestFlipIntNegAxisTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.flip
        self.api_args = {
            "x": np.random.randn(2, 3, 4).astype("int64"),
            "axis": [-1, -3],
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3, 4]}
        self.opt_shape = {"x": [1, 3, 4]}
        self.max_shape = {"x": [5, 3, 4]}

    def test_trt_result(self):
        self.check_trt_result()


class TestPNormTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.linalg.norm
        self.api_args = {
            "x": np.random.randn(2, 3, 4).astype("float32"),
            "p": 2,
            "axis": -1,
            "keepdim": False,
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3, 4]}
        self.opt_shape = {"x": [2, 3, 4]}
        self.max_shape = {"x": [4, 3, 4]}

    def test_trt_result(self):
        self.check_trt_result()


class TestPNormCase1TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.linalg.norm
        self.api_args = {
            "x": np.random.randn(2, 3).astype("float16"),
            "p": 2,
            "axis": -1,
            "keepdim": False,
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3]}
        self.opt_shape = {"x": [2, 3]}
        self.max_shape = {"x": [4, 3]}

    def test_trt_result(self):
        self.check_trt_result(precision_mode="fp16")


if __name__ == '__main__':
    unittest.main()
