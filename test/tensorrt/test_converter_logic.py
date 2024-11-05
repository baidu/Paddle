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


class TestGreaterThanFloat32TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.greater_than
        self.api_args = {
            "x": np.random.randn(2, 3).astype(np.float32),
            "y": np.random.randn(3).astype(np.float32),
        }
        self.program_config = {"feed_list": ["x", "y"]}
        self.min_shape = {"x": [1, 3], "y": [3]}
        self.max_shape = {"x": [5, 3], "y": [3]}

    def test_trt_result(self):
        self.check_trt_result()


class TestGreaterThanInt64TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.greater_than
        self.api_args = {
            "x": np.random.randn(3).astype(np.int64),
            "y": np.random.randn(3).astype(np.int64),
        }
        self.program_config = {"feed_list": ["x", "y"]}
        self.min_shape = {"x": [1], "y": [1]}
        self.max_shape = {"x": [5], "y": [5]}

    def test_trt_result(self):
        self.check_trt_result()


class TestLessThanFloat32TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.less_than
        self.api_args = {
            "x": np.random.randn(2, 3).astype(np.float32),
            "y": np.random.randn(3).astype(np.float32),
        }
        self.program_config = {"feed_list": ["x", "y"]}
        self.min_shape = {"x": [1, 3], "y": [3]}
        self.max_shape = {"x": [5, 3], "y": [3]}

    def test_trt_result(self):
        self.check_trt_result()


class TestLessThanInt64TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.less_than
        self.api_args = {
            "x": np.random.randn(3).astype(np.int64),
            "y": np.random.randn(3).astype(np.int64),
        }
        self.program_config = {"feed_list": ["x", "y"]}
        self.min_shape = {"x": [1], "y": [1]}
        self.max_shape = {"x": [5], "y": [5]}

    def test_trt_result(self):
        self.check_trt_result()


if __name__ == '__main__':
    unittest.main()
