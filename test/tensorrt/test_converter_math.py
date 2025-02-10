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


class TestMaxTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.max
        self.api_args = {
            "x": np.random.randn(2, 4).astype("float32"),
            "axis": [0, 1],
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 4]}
        self.opt_shape = {"x": [2, 4]}
        self.max_shape = {"x": [5, 4]}

    def test_trt_result(self):
        self.check_trt_result()


class TestDivideTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.divide
        self.api_args = {
            "x": np.random.randn(2, 3).astype("float32"),
            "y": np.random.randn(2, 3).astype("float32"),
        }
        self.program_config = {"feed_list": ["x", "y"]}
        self.min_shape = {"x": [1, 3], "y": [1, 3]}
        self.opt_shape = {"x": [2, 3], "y": [2, 3]}
        self.max_shape = {"x": [5, 3], "y": [5, 3]}

    def test_trt_result(self):
        self.check_trt_result()


class TestMultiplyTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.multiply
        self.api_args = {
            "x": np.random.randn(2, 3).astype("float32"),
            "y": np.random.randn(2, 3).astype("float32"),
        }
        self.program_config = {"feed_list": ["x", "y"]}
        self.min_shape = {"x": [1, 3], "y": [1, 3]}
        self.opt_shape = {"x": [2, 3], "y": [2, 3]}
        self.max_shape = {"x": [5, 3], "y": [5, 3]}

    def test_trt_result(self):
        self.check_trt_result()


class TestSubtractTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.subtract
        self.api_args = {
            "x": np.random.randn(2, 3).astype("float32"),
            "y": np.random.randn(2, 3).astype("float32"),
        }
        self.program_config = {"feed_list": ["x", "y"]}
        self.min_shape = {"x": [1, 3], "y": [1, 3]}
        self.opt_shape = {"x": [2, 3], "y": [2, 3]}
        self.max_shape = {"x": [5, 3], "y": [5, 3]}

    def test_trt_result(self):
        self.check_trt_result()


class TestAddTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.add
        self.api_args = {
            "x": np.random.randn(2, 3).astype("float32"),
            "y": np.random.randn(2, 3).astype("float32"),
        }
        self.program_config = {"feed_list": ["x", "y"]}
        self.min_shape = {"x": [1, 3], "y": [1, 3]}
        self.opt_shape = {"x": [2, 3], "y": [2, 3]}
        self.max_shape = {"x": [5, 3], "y": [5, 3]}

    def test_trt_result(self):
        self.check_trt_result()


class TestElementwisePowTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle._C_ops.elementwise_pow
        self.api_args = {
            "x": np.random.randn(2, 3).astype(np.float32),
            "y": np.random.randn(2, 3).astype(np.float32),
        }
        self.program_config = {"feed_list": ["x", "y"]}
        self.min_shape = {"x": [1, 3], "y": [1, 3]}
        self.opt_shape = {"x": [2, 3], "y": [2, 3]}
        self.max_shape = {"x": [5, 3], "y": [5, 3]}

    def test_trt_result_fp16(self):
        self.check_trt_result(precision_mode="fp16")

    def test_trt_result_fp32(self):
        self.check_trt_result()


class TestRemainderFloatTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.remainder
        self.api_args = {
            "x": np.random.randn(2, 3).astype("float32"),
            "y": np.random.uniform(low=0.1, high=1, size=(2, 3)).astype(
                "float32"
            ),  # Ensure y is non-zero
        }
        self.dynamic_shape_data = {
            "x": lambda shape: np.random.randn(*shape).astype("float32"),
            "y": lambda shape: np.random.uniform(
                low=0.1, high=1, size=shape
            ).astype("float32"),
        }
        self.program_config = {"feed_list": ["x", "y"]}
        self.min_shape = {"x": [1, 3], "y": [1, 3]}
        self.opt_shape = {"x": [2, 3], "y": [2, 3]}
        self.max_shape = {"x": [5, 3], "y": [5, 3]}

    def test_trt_result(self):
        self.check_trt_result()


class TestRemainderIntTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.remainder
        self.api_args = {
            "x": np.random.randint(1, 10, size=(2, 3)).astype("int64"),
            "y": np.random.randint(1, 10, size=(2, 3)).astype(
                "int64"
            ),  # Ensure y is non-zero
        }
        self.dynamic_shape_data = {
            "x": lambda shape: np.random.randint(1, 10, size=shape).astype(
                "int64"
            ),
            "y": lambda shape: np.random.randint(1, 10, size=shape).astype(
                "int64"
            ),
        }
        self.program_config = {"feed_list": ["x", "y"]}
        self.min_shape = {"x": [1, 3], "y": [1, 3]}
        self.opt_shape = {"x": [2, 3], "y": [2, 3]}
        self.max_shape = {"x": [5, 3], "y": [5, 3]}

    def test_trt_result(self):
        self.check_trt_result()


class TestMinTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.min
        self.api_args = {
            "x": np.random.randn(2, 4).astype("float32"),
            "axis": [0, 1],
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 4]}
        self.opt_shape = {"x": [2, 4]}
        self.max_shape = {"x": [5, 4]}

    def test_trt_result(self):
        self.check_trt_result()


class TestSumTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.sum
        self.api_args = {
            "x": np.random.randn(2, 4, 6).astype("int64"),
            "axis": [1, 1],
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 4, 6]}
        self.opt_shape = {"x": [2, 4, 6]}
        self.max_shape = {"x": [5, 4, 6]}

    def test_trt_result(self):
        self.check_trt_result()


class TestSum1TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.sum
        self.api_args = {
            "x": np.random.randn(2, 4, 6).astype("float32"),
            "axis": [1, 1],
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 4, 6]}
        self.opt_shape = {"x": [2, 4, 6]}
        self.max_shape = {"x": [5, 4, 6]}

    def test_trt_result(self):
        self.check_trt_result()


class TestAnyTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.any
        self.api_args = {
            "x": np.random.randn(2, 3, 2).astype("bool"),
            "axis": [1, 1],
            "keepdim": True,
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3, 2]}
        self.opt_shape = {"x": [2, 3, 2]}
        self.max_shape = {"x": [5, 3, 2]}

    def test_trt_result(self):
        self.check_trt_result()


class TestAny1TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.any
        self.api_args = {
            "x": np.random.randn(2, 3, 2).astype("bool"),
            "axis": [1, 1],
            "keepdim": False,
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3, 2]}
        self.opt_shape = {"x": [2, 3, 2]}
        self.max_shape = {"x": [5, 3, 2]}

    def test_trt_result(self):
        self.check_trt_result()


class TestAllTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.all
        self.api_args = {
            "x": np.random.randn(2, 3, 2).astype("bool"),
            "axis": [1, 1],
            "keepdim": True,
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3, 2]}
        self.opt_shape = {"x": [2, 3, 2]}
        self.max_shape = {"x": [5, 3, 2]}

    def test_trt_result(self):
        self.check_trt_result()


class TestAll1TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.all
        self.api_args = {
            "x": np.random.randn(2, 3, 2).astype("bool"),
            "axis": [1, 1],
            "keepdim": False,
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3, 2]}
        self.opt_shape = {"x": [2, 3, 2]}
        self.max_shape = {"x": [5, 3, 2]}

    def test_trt_result(self):
        self.check_trt_result()


class TestCumsumCase1TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.cumsum
        self.api_args = {
            "x": np.random.randn(2, 2, 3).astype("float32"),
            "axis": -1,
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 2, 3]}
        self.opt_shape = {"x": [2, 2, 3]}
        self.max_shape = {"x": [5, 2, 3]}

    def test_trt_result_fp16(self):
        self.check_trt_result(precision_mode="fp16")

    def test_trt_result_fp32(self):
        self.check_trt_result()


class TestCumsumCase2TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.cumsum
        self.api_args = {
            "x": np.random.randn(2, 2, 3).astype("float32"),
            "axis": 1,
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 2, 3]}
        self.opt_shape = {"x": [2, 2, 3]}
        self.max_shape = {"x": [5, 2, 3]}

    def test_trt_result_fp16(self):
        self.check_trt_result(precision_mode="fp16")

    def test_trt_result_fp32(self):
        self.check_trt_result()


class TestCumsumCase3TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.cumsum
        self.api_args = {
            "x": np.random.randn(2, 2, 3).astype("float32"),
            "axis": 0,
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 2, 3]}
        self.opt_shape = {"x": [2, 2, 3]}
        self.max_shape = {"x": [5, 2, 3]}

    def test_trt_result_fp16(self):
        self.check_trt_result(precision_mode="fp16")

    def test_trt_result_fp32(self):
        self.check_trt_result()


class TestCumsumCase4TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.cumsum
        self.api_args = {
            "x": np.random.randn(2, 2, 3).astype("int64"),
            "axis": 0,
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 2, 3]}
        self.opt_shape = {"x": [2, 2, 3]}
        self.max_shape = {"x": [5, 2, 3]}

    def test_trt_result(self):
        self.check_trt_result()


class TestFloorDivideFloatTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.floor_divide
        self.api_args = {
            "x": np.random.randn(2, 3).astype("float32"),
            "y": np.random.randn(2, 3).astype("float32"),
        }
        self.program_config = {"feed_list": ["x", "y"]}
        self.min_shape = {"x": [1, 3], "y": [1, 3]}
        self.opt_shape = {"x": [2, 3], "y": [2, 3]}
        self.max_shape = {"x": [5, 3], "y": [5, 3]}

    def test_trt_result(self):
        self.check_trt_result()


class TestFloorDivideIntTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.floor_divide
        self.api_args = {
            "x": np.random.randint(low=1, high=100, size=(2, 3), dtype="int64"),
            "y": np.random.randint(low=1, high=100, size=(2, 3), dtype="int64"),
        }
        self.dynamic_shape_data = {
            "x": lambda shape: np.random.randint(
                1, 100, size=shape, dtype="int64"
            ),
            "y": lambda shape: np.random.randint(
                1, 100, size=shape, dtype="int64"
            ),
        }
        self.program_config = {"feed_list": ["x", "y"]}
        self.min_shape = {"x": [1, 3], "y": [1, 3]}
        self.opt_shape = {"x": [2, 3], "y": [2, 3]}
        self.max_shape = {"x": [5, 3], "y": [5, 3]}

    def test_trt_result(self):
        self.check_trt_result()


class TestLogFloatTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.log
        self.api_args = {
            "x": np.random.randn(2, 3).astype("float32"),
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3]}
        self.opt_shape = {"x": [2, 3]}
        self.max_shape = {"x": [5, 3]}

    def test_trt_result(self):
        self.check_trt_result()


class TestLogIntTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.log
        self.api_args = {
            "x": np.random.randn(2, 3).astype("int32"),
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3]}
        self.opt_shape = {"x": [2, 3]}
        self.max_shape = {"x": [5, 3]}

    def test_trt_result(self):
        self.check_trt_result()


class TestClipTRTPatternCase1(TensorRTBaseTest):
    '''min/max is attr, and x/min/max is float'''

    def setUp(self):
        self.python_api = paddle.clip
        self.api_args = {
            "x": np.array([[2, 0.3, 0.5, 0.9], [0.1, 0.2, 6, 7]]).astype(
                "float32"
            ),
            "min": 2.2,
            "max": 5.5,
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 4]}
        self.opt_shape = {"x": [2, 4]}
        self.max_shape = {"x": [5, 4]}

    def test_trt_result(self):
        self.check_trt_result()


class TestClipTRTPatternCase2(TensorRTBaseTest):
    def setUp(self):
        '''min/max is attr, and x is int, min/max is float'''
        self.python_api = paddle.clip
        self.api_args = {
            "x": np.array([[2, 3, 5, 9], [1, 2, 6, 7]]).astype("int64"),
            "min": 2.2,
            "max": 5.5,
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 4]}
        self.opt_shape = {"x": [2, 4]}
        self.max_shape = {"x": [5, 4]}

    def test_trt_result(self):
        self.check_trt_result()


class TestClipTRTPatternCase3(TensorRTBaseTest):
    '''min/max is input, and x/min/max is float'''

    def setUp(self):
        self.python_api = paddle.clip
        self.api_args = {
            "x": np.array([[2, 0.3, 0.5, 0.9], [0.1, 0.2, 6, 7]]).astype(
                "float32"
            ),
            "min": np.array([2.2]).astype("float32"),
            "max": np.array([5.2]).astype("float32"),
        }
        self.program_config = {"feed_list": ["x", "min", "max"]}
        self.min_shape = {"x": [1, 4]}
        self.opt_shape = {"x": [2, 4]}
        self.max_shape = {"x": [5, 4]}

    def test_trt_result(self):
        self.check_trt_result()


class TestClipTRTPatternCase4(TensorRTBaseTest):
    '''min/max is input, and x is int, min/max is float'''

    def setUp(self):
        self.python_api = paddle.clip
        self.api_args = {
            "x": np.array([[2, 3, 5, 9], [1, 2, 6, 7]]).astype("int64"),
            "min": np.array([2]).astype("float32"),
            "max": np.array([5]).astype("float32"),
        }
        self.program_config = {"feed_list": ["x", "min", "max"]}
        self.min_shape = {"x": [1, 4]}
        self.opt_shape = {"x": [2, 4]}
        self.max_shape = {"x": [5, 4]}

    def test_trt_result(self):
        self.check_trt_result()


class TestIsnanFP32TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.isnan
        self.api_args = {
            "x": np.random.randn(2, 3).astype("float32"),
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3]}
        self.opt_shape = {"x": [2, 3]}
        self.max_shape = {"x": [5, 3]}

    def test_trt_result(self):
        self.check_trt_result()


class TestIsnanFP16TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.isnan
        self.api_args = {
            "x": np.random.randn(2, 3).astype("float32"),
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3]}
        self.opt_shape = {"x": [2, 3]}
        self.max_shape = {"x": [5, 3]}

    def test_trt_result(self):
        self.check_trt_result(precision_mode="fp16")


class TestIsnanIntTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.isnan
        self.api_args = {
            "x": np.random.randn(2, 3).astype("int64"),
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3]}
        self.opt_shape = {"x": [2, 3]}
        self.max_shape = {"x": [5, 3]}

    def test_trt_result(self):
        self.check_trt_result()


class TestMaximumTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.maximum
        self.api_args = {
            "x": np.random.randn(2, 3, 4).astype("float32"),
            "y": np.random.randn(2, 3, 4).astype("float32"),
        }
        self.program_config = {"feed_list": ["x", "y"]}
        self.min_shape = {"x": [1, 3, 4], "y": [1, 3, 4]}
        self.opt_shape = {"x": [2, 3, 4], "y": [2, 3, 4]}
        self.max_shape = {"x": [5, 3, 4], "y": [5, 3, 4]}

    def test_trt_result_fp16(self):
        self.check_trt_result(precision_mode="fp16")

    def test_trt_result_fp32(self):
        self.check_trt_result()


class TestMaximumBroadcastTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.maximum
        self.api_args = {
            "x": np.random.randn(2, 3, 4).astype("float32"),
            "y": np.random.randn(4).astype("float32"),
        }
        self.program_config = {"feed_list": ["x", "y"]}
        self.min_shape = {"x": [1, 3, 4], "y": [4]}
        self.opt_shape = {"x": [2, 3, 4], "y": [4]}
        self.max_shape = {"x": [5, 3, 4], "y": [4]}

    def test_trt_result_fp16(self):
        self.check_trt_result(precision_mode="fp16")

    def test_trt_result_fp32(self):
        self.check_trt_result()


class TestMaximumIntTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.maximum
        self.api_args = {
            "x": np.random.randint(
                low=1, high=100, size=(2, 3, 4), dtype="int64"
            ),
            "y": np.random.randint(
                low=1, high=100, size=(2, 3, 4), dtype="int64"
            ),
        }
        self.dynamic_shape_data = {
            "x": lambda shape: np.random.randint(
                1, 100, size=shape, dtype="int64"
            ),
            "y": lambda shape: np.random.randint(
                1, 100, size=shape, dtype="int64"
            ),
        }
        self.program_config = {"feed_list": ["x", "y"]}
        self.min_shape = {"x": [1, 3, 4], "y": [1, 3, 4]}
        self.opt_shape = {"x": [2, 3, 4], "y": [2, 3, 4]}
        self.max_shape = {"x": [5, 3, 4], "y": [5, 3, 4]}

    def test_trt_result_fp16(self):
        self.check_trt_result(precision_mode="fp16")

    def test_trt_result_fp32(self):
        self.check_trt_result()


class TestMinimumTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.minimum
        self.api_args = {
            "x": np.random.randn(2, 3, 4).astype("float32"),
            "y": np.random.randn(2, 3, 4).astype("float32"),
        }
        self.program_config = {"feed_list": ["x", "y"]}
        self.min_shape = {"x": [1, 3, 4], "y": [1, 3, 4]}
        self.opt_shape = {"x": [2, 3, 4], "y": [2, 3, 4]}
        self.max_shape = {"x": [5, 3, 4], "y": [5, 3, 4]}

    def test_trt_result_fp16(self):
        self.check_trt_result(precision_mode="fp16")

    def test_trt_result_fp32(self):
        self.check_trt_result()


class TestMinimumBroadcastTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.minimum
        self.api_args = {
            "x": np.random.randn(2, 3, 4).astype("float32"),
            "y": np.random.randn(4).astype("float32"),
        }
        self.program_config = {"feed_list": ["x", "y"]}
        self.min_shape = {"x": [1, 3, 4], "y": [4]}
        self.opt_shape = {"x": [2, 3, 4], "y": [4]}
        self.max_shape = {"x": [5, 3, 4], "y": [4]}

    def test_trt_result_fp16(self):
        self.check_trt_result(precision_mode="fp16")

    def test_trt_result_fp32(self):
        self.check_trt_result()


class TestMinimumIntTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.minimum
        self.api_args = {
            "x": np.random.randint(
                low=1, high=100, size=(2, 3, 4), dtype="int64"
            ),
            "y": np.random.randint(
                low=1, high=100, size=(2, 3, 4), dtype="int64"
            ),
        }
        self.dynamic_shape_data = {
            "x": lambda shape: np.random.randint(
                1, 100, size=shape, dtype="int64"
            ),
            "y": lambda shape: np.random.randint(
                1, 100, size=shape, dtype="int64"
            ),
        }
        self.program_config = {"feed_list": ["x", "y"]}
        self.min_shape = {"x": [1, 3, 4], "y": [1, 3, 4]}
        self.opt_shape = {"x": [2, 3, 4], "y": [2, 3, 4]}
        self.max_shape = {"x": [5, 3, 4], "y": [5, 3, 4]}

    def test_trt_result_fp16(self):
        self.check_trt_result(precision_mode="fp16")

    def test_trt_result_fp32(self):
        self.check_trt_result()


class TestGreaterEqualTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle._C_ops.greater_equal
        self.api_args = {
            "x": np.random.randn(2, 3).astype("float32"),
            "y": np.random.randn(2, 3).astype("float32"),
        }
        self.program_config = {"feed_list": ["x", "y"]}
        self.min_shape = {"x": [1, 3], "y": [1, 3]}
        self.opt_shape = {"x": [1, 3], "y": [1, 3]}
        self.max_shape = {"x": [5, 3], "y": [5, 3]}

    def test_trt_result_fp16(self):
        self.check_trt_result(precision_mode="fp16")

    def test_trt_result_fp32(self):
        self.check_trt_result()


class TestGreaterEqual_TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle._C_ops.greater_equal_
        self.api_args = {
            "x": np.random.randn(2, 3).astype("float32"),
            "y": np.random.randn(2, 3).astype("float32"),
        }
        self.program_config = {"feed_list": ["x", "y"]}
        self.min_shape = {"x": [1, 3], "y": [1, 3]}
        self.opt_shape = {"x": [1, 3], "y": [1, 3]}
        self.max_shape = {"x": [5, 3], "y": [5, 3]}

    def test_trt_result_fp16(self):
        self.check_trt_result(precision_mode="fp16")

    def test_trt_result_fp32(self):
        self.check_trt_result()


class TestGreaterEqualINTTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle._C_ops.greater_equal
        self.api_args = {
            "x": np.random.randn(2, 3).astype("int64"),
            "y": np.random.randn(2, 3).astype("int64"),
        }
        self.program_config = {"feed_list": ["x", "y"]}
        self.min_shape = {"x": [1, 3], "y": [1, 3]}
        self.opt_shape = {"x": [1, 3], "y": [1, 3]}
        self.max_shape = {"x": [5, 3], "y": [5, 3]}

    def test_trt_result_fp16(self):
        self.check_trt_result(precision_mode="fp16")

    def test_trt_result_fp32(self):
        self.check_trt_result()


class TestGreaterEqual_INTTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle._C_ops.greater_equal_
        self.api_args = {
            "x": np.random.randn(2, 3).astype("int64"),
            "y": np.random.randn(2, 3).astype("int64"),
        }
        self.program_config = {"feed_list": ["x", "y"]}
        self.min_shape = {"x": [1, 3], "y": [1, 3]}
        self.opt_shape = {"x": [1, 3], "y": [1, 3]}
        self.max_shape = {"x": [5, 3], "y": [5, 3]}

    def test_trt_result_fp16(self):
        self.check_trt_result(precision_mode="fp16")

    def test_trt_result_fp32(self):
        self.check_trt_result()


class TestGreaterEqualErrorTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle._C_ops.greater_equal
        self.api_args = {
            "x": np.random.randn(2, 3).astype("bool"),
            "y": np.random.randn(2, 3).astype("bool"),
        }
        self.program_config = {"feed_list": ["x", "y"]}
        self.min_shape = {"x": [1, 3], "y": [1, 3]}
        self.opt_shape = {"x": [1, 3], "y": [1, 3]}
        self.max_shape = {"x": [5, 3], "y": [5, 3]}

    def test_trt_result(self):
        self.check_marker(expected_result=False)


class TestLessEqualTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle._C_ops.less_equal
        self.api_args = {
            "x": np.random.randn(2, 3).astype("float32"),
            "y": np.random.randn(2, 3).astype("float32"),
        }
        self.program_config = {"feed_list": ["x", "y"]}
        self.min_shape = {"x": [1, 3], "y": [1, 3]}
        self.opt_shape = {"x": [1, 3], "y": [1, 3]}
        self.max_shape = {"x": [5, 3], "y": [5, 3]}

    def test_trt_result_fp16(self):
        self.check_trt_result(precision_mode="fp16")

    def test_trt_result_fp32(self):
        self.check_trt_result()


class TestLessEqual_TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle._C_ops.less_equal_
        self.api_args = {
            "x": np.random.randn(2, 3).astype("float32"),
            "y": np.random.randn(2, 3).astype("float32"),
        }
        self.program_config = {"feed_list": ["x", "y"]}
        self.min_shape = {"x": [1, 3], "y": [1, 3]}
        self.opt_shape = {"x": [1, 3], "y": [1, 3]}
        self.max_shape = {"x": [5, 3], "y": [5, 3]}

    def test_trt_result_fp16(self):
        self.check_trt_result(precision_mode="fp16")

    def test_trt_result_fp32(self):
        self.check_trt_result()


class TestLessEqualINTTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle._C_ops.less_equal
        self.api_args = {
            "x": np.random.randn(2, 3).astype("int64"),
            "y": np.random.randn(2, 3).astype("int64"),
        }
        self.program_config = {"feed_list": ["x", "y"]}
        self.min_shape = {"x": [1, 3], "y": [1, 3]}
        self.opt_shape = {"x": [1, 3], "y": [1, 3]}
        self.max_shape = {"x": [5, 3], "y": [5, 3]}

    def test_trt_result_fp16(self):
        self.check_trt_result(precision_mode="fp16")

    def test_trt_result_fp32(self):
        self.check_trt_result()


class TestLessEqual_INTTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle._C_ops.less_equal_
        self.api_args = {
            "x": np.random.randn(2, 3).astype("int64"),
            "y": np.random.randn(2, 3).astype("int64"),
        }
        self.program_config = {"feed_list": ["x", "y"]}
        self.min_shape = {"x": [1, 3], "y": [1, 3]}
        self.opt_shape = {"x": [1, 3], "y": [1, 3]}
        self.max_shape = {"x": [5, 3], "y": [5, 3]}

    def test_trt_result_fp16(self):
        self.check_trt_result(precision_mode="fp16")

    def test_trt_result_fp32(self):
        self.check_trt_result()


class TestLessEqualErrorTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle._C_ops.less_equal
        self.api_args = {
            "x": np.random.randn(2, 3).astype("bool"),
            "y": np.random.randn(2, 3).astype("bool"),
        }
        self.program_config = {"feed_list": ["x", "y"]}
        self.min_shape = {"x": [1, 3], "y": [1, 3]}
        self.opt_shape = {"x": [1, 3], "y": [1, 3]}
        self.max_shape = {"x": [5, 3], "y": [5, 3]}

    def test_trt_result(self):
        self.check_marker(expected_result=False)


if __name__ == '__main__':
    unittest.main()
