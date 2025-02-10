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
from paddle import _C_ops


class TestCast0TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.cast
        self.api_args = {
            "x": np.random.randn(7, 3).astype("float32"),
            "out_dtype": "bool",
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [3, 3]}
        self.opt_shape = {"x": [5, 3]}
        self.max_shape = {"x": [10, 3]}

    def test_trt_result(self):
        self.check_trt_result()


class TestCast1TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.cast
        self.api_args = {
            "x": np.random.randn(7, 3).astype("float16"),
            "out_dtype": "int32",
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [3, 3]}
        self.opt_shape = {"x": [5, 3]}
        self.max_shape = {"x": [10, 3]}

    def test_trt_result(self):
        self.check_trt_result()


class TestCast2TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.cast
        self.api_args = {
            "x": np.random.randn(7, 3).astype("float32"),
            "out_dtype": "int64",
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [3, 3]}
        self.opt_shape = {"x": [5, 3]}
        self.max_shape = {"x": [10, 3]}

    def test_trt_result(self):
        self.check_trt_result()


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
        self.opt_shape = {"x": [[5, 3], [5, 3], [5, 2]]}
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
        self.opt_shape = {"x": [10, 1, 1, 19]}
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
        self.opt_shape = {"x": [6, 3]}
        self.max_shape = {"x": [6, 3]}

    def test_trt_result(self):
        self.check_trt_result()


class TestExpandWithShapeTensorTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.expand
        self.api_args = {
            "x": np.random.randn(1, 3).astype("float32"),
            "shape": np.array([6, 3]).astype("int64"),
        }
        self.program_config = {"feed_list": ["x", "shape"]}
        self.min_shape = {"x": [1, 3]}
        self.opt_shape = {"x": [6, 3]}
        self.max_shape = {"x": [6, 3]}

    def test_trt_result(self):
        self.check_trt_result()


def slice_api(x, axes, starts, ends, infer_flags, decrease_axis):
    return _C_ops.slice(x, axes, starts, ends, infer_flags, decrease_axis)


class TestSliceWithDecreaseAxisTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = slice_api
        self.api_args = {
            "x": np.random.random([6, 6, 64, 64]).astype("float32"),
            "axes": [0, 1],
            "starts": [0, 1],
            "ends": [2, 2],
            "infer_flags": [1, 1],
            "decrease_axis": [1],
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [2, 6, 64, 64]}
        self.opt_shape = {"x": [4, 6, 64, 64]}
        self.max_shape = {"x": [8, 6, 64, 64]}

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
        self.opt_shape = {}
        self.max_shape = {}

    def test_trt_result(self):
        self.check_trt_result()


class TestSliceTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.slice
        self.api_args = {
            "x": np.random.random([6, 6, 64, 64]).astype("float32"),
            "axes": [0, 1],
            "starts": [-2, -3],
            "ends": [-1, -1],
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [2, 6, 64, 64]}
        self.opt_shape = {"x": [4, 6, 64, 64]}
        self.max_shape = {"x": [8, 6, 64, 64]}

    def test_trt_result(self):
        self.check_trt_result()


class TestExpandAsTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.expand_as
        self.api_args = {
            "x": np.array([[1, 2, 3]]).astype("float32"),
            "y": np.array([[1, 2, 3], [4, 5, 6], [1, 2, 3], [4, 5, 6]]).astype(
                "int64"
            ),
        }
        self.program_config = {"feed_list": ["x", "y"]}
        self.min_shape = {"x": [1, 3]}
        self.opt_shape = {"x": [4, 3]}
        self.max_shape = {"x": [4, 3]}

    def test_trt_result(self):
        self.check_trt_result()


class TestSliceWithInputStartTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.slice
        self.api_args = {
            "x": np.random.random([5, 4, 5, 6]).astype("float32"),
            "axes": [0, 1, 2],
            "starts": np.array([1, 0, 2]).astype("int64"),
            "ends": np.array([3, 3, 4]).astype("int64"),
        }
        self.program_config = {"feed_list": ["x", "starts", "ends"]}
        self.min_shape = {"x": [3, 4, 5, 6]}
        self.opt_shape = {"x": [6, 4, 5, 6]}
        self.max_shape = {"x": [6, 4, 5, 6]}

    def test_trt_result(self):
        self.check_trt_result()


class TestGatherCase1TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.gather
        self.api_args = {
            "x": np.random.random([3, 4, 10]).astype("float32"),
            "index": np.array([0, 2]).astype("int64"),
            "axis": 1,
        }
        self.program_config = {"feed_list": ["x", "index"]}
        self.min_shape = {"x": [1, 4, 10], "index": [1]}
        self.opt_shape = {"x": [1, 4, 10], "index": [1]}
        self.max_shape = {"x": [5, 4, 10], "index": [5]}

    def test_trt_result(self):
        self.check_trt_result()


class TestGatherCase2TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.gather
        self.api_args = {
            "x": np.random.random([3, 4, 10]).astype("int64"),
            "index": np.array([0, 2]).astype("int64"),
            "axis": 1,
        }
        self.program_config = {"feed_list": ["x", "index"]}
        self.min_shape = {"x": [1, 4, 10], "index": [1]}
        self.opt_shape = {"x": [1, 4, 10], "index": [1]}
        self.max_shape = {"x": [5, 4, 10], "index": [5]}

    def test_trt_result(self):
        self.check_trt_result()


class TestGatherCase3TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.gather
        self.api_args = {
            "x": np.random.random([3, 4, 10]).astype("int64"),
            "index": np.array([0, 2]).astype("int64"),
            "axis": np.array([2]).astype("int64"),
        }
        self.program_config = {"feed_list": ["x", "index", "axis"]}
        self.min_shape = {"x": [1, 4, 10], "index": [1]}
        self.opt_shape = {"x": [1, 4, 10], "index": [1]}
        self.max_shape = {"x": [5, 4, 10], "index": [5]}

    def test_trt_result(self):
        self.check_marker(expected_result=False)


class TestSplitWithNumTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.split
        self.api_args = {
            "x": np.random.randn(3, 9, 5).astype("float32"),
            "num_or_sections": 3,
            "axis": 1,
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 9, 5]}
        self.opt_shape = {"x": [3, 9, 5]}
        self.max_shape = {"x": [3, 9, 5]}

    def test_trt_result(self):
        self.check_trt_result()


class TestSplitWithNumAxisTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.split
        self.api_args = {
            "x": np.random.randn(3, 9, 5).astype("float32"),
            "num_or_sections": 3,
            "axis": np.array([1]).astype("int64"),
        }
        self.program_config = {"feed_list": ["x", "axis"]}
        self.min_shape = {"x": [1, 9, 5]}
        self.opt_shape = {"x": [3, 9, 5]}
        self.max_shape = {"x": [3, 9, 5]}

    def test_trt_result(self):
        self.check_trt_result()


class TestSplitWithNumAllTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.split
        self.api_args = {
            "x": np.random.randn(1, 2).astype("float32"),
            "num_or_sections": 2,
            "axis": np.array([1]).astype("int64"),
        }
        self.program_config = {"feed_list": ["x", "axis"]}
        self.min_shape = {"x": [1, 2]}
        self.opt_shape = {"x": [1, 2]}
        self.max_shape = {"x": [3, 2]}

    def test_trt_result(self):
        self.check_trt_result()


class TestSplitWithNumNegativeAxisTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.split
        self.api_args = {
            "x": np.random.randn(3, 9, 5).astype("float32"),
            "num_or_sections": 3,
            "axis": -2,
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 9, 5]}
        self.opt_shape = {"x": [2, 9, 5]}
        self.max_shape = {"x": [3, 9, 5]}

    def test_trt_result(self):
        self.check_trt_result()


class TestSplitTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.split
        self.api_args = {
            "x": np.random.randn(3, 9, 5).astype("float32"),
            "num_or_sections": [2, 4, 3],
            "axis": -2,
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 9, 5]}
        self.opt_shape = {"x": [2, 9, 5]}
        self.max_shape = {"x": [3, 9, 5]}

    def test_trt_result(self):
        self.check_trt_result()


class TestSplitAxisTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.split
        self.api_args = {
            "x": np.random.randn(3, 9, 5).astype("float32"),
            "num_or_sections": [2, 4, 3],
            "axis": np.array([1]).astype("int64"),
        }
        self.program_config = {"feed_list": ["x", "axis"]}
        self.min_shape = {"x": [1, 9, 5]}
        self.opt_shape = {"x": [2, 9, 5]}
        self.max_shape = {"x": [3, 9, 5]}

    def test_trt_result(self):
        self.check_trt_result()


def split_api(input, num_or_sections, dim):
    return _C_ops.split(input, num_or_sections, dim)


class TestSplitDynamicSectionsTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = split_api
        self.api_args = {
            "x": np.random.randn(3, 9, 5).astype("float32"),
            "num_or_sections": np.array([2, 4, 3]).astype("int64"),
            "axis": 1,
        }
        self.program_config = {"feed_list": ["x", "num_or_sections"]}
        self.min_shape = {"x": [1, 9, 5]}
        self.opt_shape = {"x": [2, 9, 5]}
        self.max_shape = {"x": [3, 9, 5]}

    def test_trt_result(self):
        self.check_trt_result()


class TestSplitDynamicSectionAndAxisTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = split_api
        self.api_args = {
            "x": np.random.randn(3, 9, 5).astype("float32"),
            "num_or_sections": np.array([2, 4, 3]).astype("int64"),
            "axis": np.array([1]).astype("int64"),
        }
        self.program_config = {"feed_list": ["x", "num_or_sections", "axis"]}
        self.min_shape = {"x": [1, 9, 5]}
        self.opt_shape = {"x": [2, 9, 5]}
        self.max_shape = {"x": [3, 9, 5]}

    def test_trt_result(self):
        self.check_trt_result()


class TestStackTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.stack
        self.api_args = {
            "x": [
                np.array([[1.0, 2.0]]).astype("float32"),
                np.array([[3.0, 4.0]]).astype("float32"),
                np.array([[5.0, 6.0]]).astype("float32"),
            ],
            "axis": 0,
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [[1, 2], [1, 2], [1, 2]]}
        self.opt_shape = {"x": [[2, 2], [2, 2], [2, 2]]}
        self.max_shape = {"x": [[3, 2], [3, 2], [3, 2]]}

    def test_trt_result(self):
        self.check_trt_result()


class TestStackCase2TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.stack
        self.api_args = {
            "x": [
                np.array([[1, 2]]).astype("int64"),
                np.array([[3, 4]]).astype("int64"),
                np.array([[5, 6]]).astype("int64"),
            ],
            "axis": -1,
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [[1, 2], [1, 2], [1, 2]]}
        self.opt_shape = {"x": [[2, 2], [2, 2], [2, 2]]}
        self.max_shape = {"x": [[3, 2], [3, 2], [3, 2]]}

    def test_trt_result(self):
        self.check_trt_result()


class TestTileTRTPatternCase0(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.tile
        self.api_args = {
            "x": np.random.randn(1, 2, 3).astype("float32"),
            "repeat_times": (2, 4),
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 2, 3]}
        self.opt_shape = {"x": [2, 2, 3]}
        self.max_shape = {"x": [2, 2, 3]}

    def test_trt_result(self):
        self.check_trt_result()


class TestTileTRTPatternCase1(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.tile
        self.api_args = {
            "x": np.random.randn(1, 2, 3).astype("int64"),
            "repeat_times": np.array([1, 2, 3, 4]).astype("int64"),
        }
        self.program_config = {"feed_list": ["x", "repeat_times"]}
        self.min_shape = {"x": [1, 2, 3]}
        self.opt_shape = {"x": [2, 2, 3]}
        self.max_shape = {"x": [2, 2, 3]}

    def test_trt_result(self):
        self.check_trt_result()


class TestTileTRTPatternCase2(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.tile
        self.api_args = {
            "x": np.random.randn(1, 2, 3).astype("float32"),
            "repeat_times": [1, 2, 3],
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 2, 3]}
        self.opt_shape = {"x": [2, 2, 3]}
        self.max_shape = {"x": [2, 2, 3]}

    def test_trt_result(self):
        self.check_trt_result()


class TestTakeAlongAxisCase0TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.take_along_axis
        self.api_args = {
            "X": np.random.random([3, 4, 10]).astype("float32"),
            "Index": np.random.randint(0, 2, size=(3, 4, 10)).astype("int64"),
            "axis": 1,
        }
        self.program_config = {"feed_list": ["X", "Index"]}
        self.min_shape = {"X": [1, 4, 10], "Index": [1, 4, 10]}
        self.opt_shape = {"X": [3, 4, 10], "Index": [3, 4, 10]}
        self.max_shape = {"X": [5, 4, 10], "Index": [5, 4, 10]}

    def test_trt_result(self):
        self.check_trt_result()


class TestTakeAlongAxisCase1TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.take_along_axis
        self.api_args = {
            "X": np.random.random([3, 4, 10]).astype("float32"),
            "Index": np.random.randint(0, 2, size=(3, 4, 10)).astype("int64"),
            "axis": -1,
        }
        self.program_config = {"feed_list": ["X", "Index"]}
        self.min_shape = {"X": [1, 4, 10], "Index": [1, 4, 10]}
        self.opt_shape = {"X": [3, 4, 10], "Index": [3, 4, 10]}
        self.max_shape = {"X": [5, 4, 10], "Index": [5, 4, 10]}

    def test_trt_result(self):
        self.check_trt_result()


class TestTakeAlongAxisFP16TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.take_along_axis
        self.api_args = {
            "X": np.random.random([3, 4, 10]).astype("float32"),
            "Index": np.random.randint(0, 2, size=(3, 4, 10)).astype("int64"),
            "axis": 1,
        }
        self.program_config = {"feed_list": ["X", "Index"]}
        self.min_shape = {"X": [1, 4, 10], "Index": [1, 4, 10]}
        self.opt_shape = {"X": [3, 4, 10], "Index": [3, 4, 10]}
        self.max_shape = {"X": [5, 4, 10], "Index": [5, 4, 10]}

    def test_trt_result(self):
        self.check_trt_result(precision_mode="fp16")


class TestStrideSliceCase1TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.strided_slice
        self.api_args = {
            "x": np.random.random([3, 4, 10]).astype("float32"),
            "axes": [0, 1, 2],
            "starts": [1, 0, 2],
            "ends": [2, 3, 4],
            "strides": [1, 1, 1],
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 4, 10]}
        self.opt_shape = {"x": [2, 4, 10]}
        self.max_shape = {"x": [5, 4, 10]}

    def test_trt_result(self):
        self.check_trt_result()


class TestStrideSliceCase2TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.strided_slice
        self.api_args = {
            "x": np.random.random([3, 4, 10]).astype("int64"),
            "axes": [0, 1, 2],
            "starts": [1, 0, 2],
            "ends": [2, 3, 4],
            "strides": [1, 1, 1],
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 4, 10]}
        self.opt_shape = {"x": [2, 4, 10]}
        self.max_shape = {"x": [5, 4, 10]}

    def test_trt_result(self):
        self.check_trt_result()


class TestStrideSliceCase3TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.strided_slice
        self.api_args = {
            "x": np.random.random([3, 4, 10]).astype("bool"),
            "axes": [0, 1, 2],
            "starts": [0, -1, 0],
            "ends": [2, -3, 5],
            "strides": [1, -1, 1],
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 4, 10]}
        self.opt_shape = {"x": [2, 4, 10]}
        self.max_shape = {"x": [5, 4, 10]}

    def test_trt_result(self):
        self.check_trt_result()


class TestStrideSliceCase4TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.strided_slice
        self.api_args = {
            "x": np.random.random([1, 56, 56, 128]).astype("float32"),
            "axes": [1, 2],
            "starts": [0, 0],
            "ends": [6, 6],
            "strides": [2, 2],
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 56, 56, 128]}
        self.opt_shape = {"x": [3, 56, 56, 128]}
        self.max_shape = {"x": [2, 56, 56, 128]}

    def test_trt_result(self):
        self.check_trt_result()


class TestStrideSliceCase5TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.strided_slice
        self.api_args = {
            "x": np.random.random([1, 56, 56, 128]).astype("float32"),
            "axes": [1, 2],
            "starts": [
                1,
                1,
            ],
            "ends": [10000, 10000],
            "strides": [2, 2],
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 56, 56, 128]}
        self.opt_shape = {"x": [3, 56, 56, 128]}
        self.max_shape = {"x": [3, 56, 56, 128]}

    def test_trt_result(self):
        self.check_trt_result()


class TestRollCase1TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.roll
        self.api_args = {
            "x": np.random.random([3, 4, 10]).astype("float32"),
            "shift": 1,
            "axis": 0,
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 4, 10]}
        self.opt_shape = {"x": [2, 4, 10]}
        self.max_shape = {"x": [5, 4, 10]}

    def test_trt_result(self):
        self.check_trt_result()


class TestRollCase2TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.roll
        self.api_args = {
            "x": np.random.random([3, 4, 10]).astype("int64"),
            "shift": 1,
            "axis": 1,
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 4, 10]}
        self.opt_shape = {"x": [2, 4, 10]}
        self.max_shape = {"x": [5, 4, 10]}

    def test_trt_result(self):
        self.check_trt_result()


class TestRollCase3TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.roll
        self.api_args = {
            "x": np.random.random([3, 4, 10]).astype("float32"),
            "shift": np.array([1]).astype("int64"),
            "axis": 1,
        }
        self.program_config = {"feed_list": ["x", "shift"]}
        self.min_shape = {"x": [1, 4, 10]}
        self.opt_shape = {"x": [2, 4, 10]}
        self.max_shape = {"x": [5, 4, 10]}

    def test_trt_result(self):
        self.check_trt_result()


class TestSqueezeTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.squeeze
        self.api_args = {
            "x": np.random.random([1, 1, 28]).astype("float32"),
            "axis": 1,
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 1, 28]}
        self.opt_shape = {"x": [2, 1, 28]}
        self.max_shape = {"x": [5, 1, 28]}

    def test_trt_result(self):
        self.check_trt_result()


class TestSqueezeCase1TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.squeeze
        self.api_args = {
            "x": np.random.random([1, 1, 28]).astype("int64"),
            "axis": 1,
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 1, 28]}
        self.opt_shape = {"x": [2, 1, 28]}
        self.max_shape = {"x": [5, 1, 28]}

    def test_trt_result(self):
        self.check_trt_result()


class TestNumelTRTCase1Pattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.numel
        self.api_args = {
            "x": np.random.randn(2, 3).astype("float32"),
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3]}
        self.opt_shape = {"x": [2, 3]}
        self.max_shape = {"x": [5, 3]}

    def test_trt_result(self):
        self.check_trt_result()

    def test_fp16_result(self):
        self.check_trt_result(precision_mode="fp16")


class TestNumelTRTCase2Pattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.numel
        self.api_args = {
            "x": np.random.randn(1, 2, 33, 33).astype("int64"),
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 2, 33, 33]}
        self.opt_shape = {"x": [2, 2, 33, 33]}
        self.max_shape = {"x": [5, 2, 33, 33]}

    def test_trt_result(self):
        self.check_trt_result()


if __name__ == '__main__':
    unittest.main()
