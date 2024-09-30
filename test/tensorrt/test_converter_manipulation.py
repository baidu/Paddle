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


class TestConcatTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.concat
        self.api_args = {
            "x": [
                np.array([[1, 2, 3], [4, 5, 6]]).astype("float32"),
                np.array([[11, 12, 13], [14, 15, 16]]).astype("float32"),
                np.array([[21, 22], [23, 24]]).astype("float32"),
            ],
            "axis": -1,
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [[1, 3], [1, 3], [1, 2]]}
        self.max_shape = {"x": [[5, 3], [5, 3], [5, 2]]}

    def test_trt_result(self):
        self.check_trt_result()


class TestFlattenTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.flatten
        self.api_args = {
            "x": np.random.random([2, 1, 1, 19]).astype("float32"),
            "start_axis": 1,
            "stop_axis": 2,
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 1, 1, 19]}
        self.max_shape = {"x": [10, 1, 1, 19]}

    def test_trt_result(self):
        self.check_trt_result()


class TestExpandTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.expand
        self.api_args = {
            "x": np.random.randn(1, 3).astype("float32"),
            "shape": [6, 3],
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3]}
        self.max_shape = {"x": [6, 3]}

    def test_trt_result(self):
        self.check_trt_result()


class TestExpandWithShapeTensorTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.expand
        self.api_args = {
            "x": np.random.randn(1, 3).astype("float32"),
            "shape": np.array([6, 3]).astype("int32"),
        }
        self.program_config = {"feed_list": ["x", "shape"]}
        self.min_shape = {"x": [1, 3]}
        self.max_shape = {"x": [6, 3]}

    def test_trt_result(self):
        self.check_trt_result()


class TestExpandWithDiffRankTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.expand
        self.api_args = {
            "x": np.array([1, 2, 3]).astype("float32"),
            "shape": [2, 3],
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {}
        self.max_shape = {}

    def test_trt_result(self):
        self.check_trt_result()


class TestExpandAsTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.expand_as
        self.api_args = {
            "x": np.array([[1, 2, 3]]).astype("float32"),
            "y": np.array([[1, 2, 3], [4, 5, 6], [1, 2, 3], [4, 5, 6]]).astype(
                "int32"
            ),
        }
        self.program_config = {"feed_list": ["x", "y"]}
        self.min_shape = {"x": [1, 3]}
        self.max_shape = {"x": [4, 3]}

    def test_trt_result(self):
        self.check_trt_result()


if __name__ == '__main__':
    unittest.main()
