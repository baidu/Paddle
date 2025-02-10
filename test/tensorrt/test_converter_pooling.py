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


def pool2d_api(
    x,
    ksize=[],
    strides=[],
    paddings=[],
    ceil_mode=False,
    exclusive=True,
    data_format="NCHW",
    pooling_type="max",
    global_pooling=False,
    adaptive=False,
    padding_algorithm="EXPLICIT",
):
    return paddle._C_ops.pool2d(
        x,
        ksize,
        strides,
        paddings,
        ceil_mode,
        exclusive,
        data_format,
        pooling_type,
        global_pooling,
        adaptive,
        padding_algorithm,
    )


class TestPoolingTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.nn.AvgPool2D(kernel_size=2, stride=1)
        self.api_args = {
            "x": np.random.randn(1, 1, 2, 3).astype("float32"),
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 1, 2, 3]}
        self.opt_shape = {"x": [1, 1, 2, 3]}
        self.max_shape = {"x": [5, 1, 2, 3]}

    def test_trt_result(self):
        self.check_trt_result()


class TestPoolingTRTCase1Pattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = pool2d_api
        self.api_args = {
            "x": np.random.randn(1, 1, 2, 3).astype("float32"),
            "ksize": [2, 3],
            "strides": [1, 2],
            "paddings": [0, 0],
            "ceil_mode": False,
            "exclusive": False,
            "data_format": "NCHW",
            "pooling_type": "avg",
            "global_pooling": False,
            "adaptive": False,
            "padding_algorithm": "VALID",
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 1, 2, 3]}
        self.opt_shape = {"x": [1, 1, 2, 3]}
        self.max_shape = {"x": [5, 1, 2, 3]}

    def test_trt_result(self):
        self.check_trt_result()


class TestPoolingTRTCase2Pattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = pool2d_api
        self.api_args = {
            "x": np.random.randn(1, 1, 2, 3).astype("float32"),
            "ksize": [2, 3],
            "strides": [1, 2],
            "paddings": [0, 0],
            "ceil_mode": True,
            "exclusive": True,
            "data_format": "NCHW",
            "pooling_type": "max",
            "global_pooling": False,
            "adaptive": False,
            "padding_algorithm": "SAME",
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 1, 2, 3]}
        self.opt_shape = {"x": [1, 1, 2, 3]}
        self.max_shape = {"x": [5, 1, 2, 3]}

    def test_trt_result(self):
        self.check_trt_result()


class TestPoolingTRTCase3Pattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = pool2d_api
        self.api_args = {
            "x": np.random.randn(1, 1, 2, 3).astype("float32"),
            "ksize": [2, 3],
            "strides": [1, 2],
            "paddings": [0, 0],
            "ceil_mode": True,
            "exclusive": True,
            "data_format": "NCHW",
            "pooling_type": "max",
            "global_pooling": True,
            "adaptive": False,
            "padding_algorithm": "SAME",
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 1, 2, 3]}
        self.opt_shape = {"x": [1, 1, 2, 3]}
        self.max_shape = {"x": [5, 1, 2, 3]}

    def test_trt_result(self):
        self.check_trt_result()


class TestPoolingTRTCase4Pattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = pool2d_api
        self.api_args = {
            "x": np.random.randn(1, 1, 5, 5).astype("float32"),
            "ksize": [3, 3],
            "strides": [1, 1],
            "paddings": [0, 0],
            "ceil_mode": False,
            "exclusive": False,
            "data_format": "NCHW",
            "pooling_type": "avg",
            "global_pooling": True,
            "adaptive": False,
            "padding_algorithm": "SAME",
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 1, 5, 5]}
        self.opt_shape = {"x": [1, 1, 5, 5]}
        self.max_shape = {"x": [5, 1, 5, 5]}

    def test_trt_result(self):
        self.check_trt_result()


class TestPoolingTRTCase5Pattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = pool2d_api
        self.api_args = {
            "x": np.random.randn(1, 16, 56, 56).astype("float32"),
            "ksize": [2, 2],
            "strides": [1, 1],
            "paddings": [0, 0],
            "ceil_mode": False,
            "exclusive": True,
            "data_format": "NCHW",
            "pooling_type": "avg",
            "global_pooling": False,
            "adaptive": True,
            "padding_algorithm": "EXPLICIT",
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 16, 56, 56]}
        self.opt_shape = {"x": [1, 16, 56, 56]}
        self.max_shape = {"x": [5, 16, 56, 56]}

    def test_trt_result(self):
        self.check_trt_result()


class TestPoolingTRTCase6Pattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = pool2d_api
        self.api_args = {
            "x": np.random.randn(1, 3, 5, 5).astype("float32"),
            "ksize": [1, 1],
            "strides": [1, 1],
            "paddings": [0, 0],
            "ceil_mode": False,
            "exclusive": True,
            "data_format": "NCHW",
            "pooling_type": "avg",
            "global_pooling": False,
            "adaptive": True,
            "padding_algorithm": "EXPLICIT",
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3, 5, 5]}
        self.opt_shape = {"x": [1, 3, 5, 5]}
        self.max_shape = {"x": [2, 3, 5, 5]}

    def test_trt_result(self):
        self.check_trt_result()


class TestPoolingTRTMarker(TensorRTBaseTest):
    def setUp(self):
        self.python_api = pool2d_api
        self.api_args = {
            "x": np.random.randn(1, 3, 5, 5).astype("float32"),
            "ksize": [6, 6],
            "strides": [2, 2],
            "paddings": [0, 0],
            "ceil_mode": False,
            "exclusive": False,
            "data_format": "NCHW",
            "pooling_type": "avg",
            "global_pooling": False,
            "adaptive": False,
            "padding_algorithm": "EXPLICIT",
        }
        self.program_config = {"feed_list": ["x"]}
        self.target_marker_op = "pd_op.pool2d"

    def test_trt_result(self):
        self.check_marker(expected_result=False)


def pool3d_api(
    x,
    ksize=[],
    strides=[],
    paddings=[],
    ceil_mode=False,
    exclusive=True,
    data_format="NCHW",
    pooling_type="max",
    global_pooling=False,
    adaptive=False,
    padding_algorithm="EXPLICIT",
):
    return paddle._C_ops.pool3d(
        x,
        ksize,
        strides,
        paddings,
        ceil_mode,
        exclusive,
        data_format,
        pooling_type,
        global_pooling,
        adaptive,
        padding_algorithm,
    )


class TestPooling3dTRTCase1Pattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = pool3d_api
        self.api_args = {
            "x": np.random.randn(1, 3, 5, 5, 5).astype("float32"),
            "ksize": [1, 1, 1],
            "strides": [1, 1, 1],
            "paddings": [0, 0, 0],
            "ceil_mode": False,
            "exclusive": True,
            "data_format": "NCHW",
            "pooling_type": "avg",
            "global_pooling": False,
            "adaptive": False,
            "padding_algorithm": "EXPLICIT",
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3, 5, 5, 5]}
        self.opt_shape = {"x": [1, 3, 5, 5, 5]}
        self.max_shape = {"x": [2, 3, 5, 5, 5]}

    def test_fp32_trt_result(self):
        self.check_trt_result()

    def test_fp16_trt_result(self):
        self.check_trt_result(precision_mode="fp16")


class TestPooling3dTRTCase2Pattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = pool3d_api
        self.api_args = {
            "x": np.ones([1, 3, 5, 5, 5]).astype("float32"),
            "ksize": [1, 1, 1],
            "strides": [1, 1, 1],
            "paddings": [0, 0, 0],
            "ceil_mode": False,
            "exclusive": True,
            "data_format": "NCHW",
            "pooling_type": "avg",
            "global_pooling": False,
            "adaptive": True,
            "padding_algorithm": "EXPLICIT",
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3, 5, 5, 5]}
        self.opt_shape = {"x": [1, 3, 5, 5, 5]}
        self.max_shape = {"x": [2, 3, 5, 5, 5]}

    def test_fp32_trt_result(self):
        self.check_trt_result()

    def test_fp16_trt_result(self):
        self.check_trt_result(precision_mode="fp16")


class TestPooling3dTRTCase3Pattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = pool3d_api
        self.api_args = {
            "x": np.ones([1, 3, 5, 5, 5]).astype("float32"),
            "ksize": [1, 1, 1],
            "strides": [1, 1, 1],
            "paddings": [0, 0, 0],
            "ceil_mode": False,
            "exclusive": True,
            "data_format": "NCHW",
            "pooling_type": "avg",
            "global_pooling": True,
            "adaptive": False,
            "padding_algorithm": "EXPLICIT",
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3, 5, 5, 5]}
        self.opt_shape = {"x": [1, 3, 5, 5, 5]}
        self.max_shape = {"x": [2, 3, 5, 5, 5]}

    def test_fp32_trt_result(self):
        self.check_trt_result()

    def test_fp16_trt_result(self):
        self.check_trt_result(precision_mode="fp16")


class TestPooling3dTRTCase4Pattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = pool3d_api
        self.api_args = {
            "x": np.random.randn(1, 3, 5, 5, 5).astype("float32"),
            "ksize": [1, 1, 1],
            "strides": [1, 1, 1],
            "paddings": [0, 0, 0],
            "ceil_mode": False,
            "exclusive": True,
            "data_format": "NCHW",
            "pooling_type": "avg",
            "global_pooling": False,
            "adaptive": False,
            "padding_algorithm": "SAME",
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3, 5, 5, 5]}
        self.opt_shape = {"x": [1, 3, 5, 5, 5]}
        self.max_shape = {"x": [2, 3, 5, 5, 5]}

    def test_fp32_trt_result(self):
        self.check_trt_result()

    def test_fp16_trt_result(self):
        self.check_trt_result(precision_mode="fp16")


class TestPooling3dTRTCase5Pattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = pool3d_api
        self.api_args = {
            "x": np.random.randn(1, 3, 5, 5, 5).astype("float32"),
            "ksize": [1, 1, 1],
            "strides": [1, 1, 1],
            "paddings": [0, 0, 0],
            "ceil_mode": False,
            "exclusive": True,
            "data_format": "NCHW",
            "pooling_type": "avg",
            "global_pooling": False,
            "adaptive": False,
            "padding_algorithm": "VALID",
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3, 5, 5, 5]}
        self.opt_shape = {"x": [1, 3, 5, 5, 5]}
        self.max_shape = {"x": [2, 3, 5, 5, 5]}

    def test_fp32_trt_result(self):
        self.check_trt_result()

    def test_fp16_trt_result(self):
        self.check_trt_result(precision_mode="fp16")


class TestPooling3dTRTCase6Pattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = pool3d_api
        self.api_args = {
            "x": np.ones([1, 3, 5, 5, 5]).astype("float32"),
            "ksize": [1, 1, 1],
            "strides": [1, 1, 1],
            "paddings": [0, 0, 0],
            "ceil_mode": False,
            "exclusive": True,
            "data_format": "NCHW",
            "pooling_type": "max",
            "global_pooling": True,
            "adaptive": False,
            "padding_algorithm": "EXPLICIT",
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3, 5, 5, 5]}
        self.opt_shape = {"x": [1, 3, 5, 5, 5]}
        self.max_shape = {"x": [2, 3, 5, 5, 5]}

    def test_fp32_trt_result(self):
        self.check_trt_result()

    def test_fp16_trt_result(self):
        self.check_trt_result(precision_mode="fp16")


if __name__ == '__main__':
    unittest.main()
