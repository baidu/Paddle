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


def conv2d_wrapper(x):
    conv = paddle.nn.Conv2D(3, 3, (3, 3))
    return conv(x)


def conv2d_python_api(x, padding="SAME", stride=(1, 1)):
    conv = paddle.nn.Conv2D(3, 3, (3, 3), padding=padding, stride=stride)
    return conv(x)


class TestConv2dTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = conv2d_wrapper
        self.api_args = {
            "x": np.random.random([2, 3, 8, 8]).astype("float32"),
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3, 8, 8]}
        self.opt_shape = {"x": [2, 3, 8, 8]}
        self.max_shape = {"x": [10, 3, 8, 8]}
        self.disable_passes = ['constant_folding_pass', 'conv2d_add_fuse_pass']

    def test_trt_result_fp16(self):
        self.check_trt_result(precision_mode="fp16")

    def test_trt_result_fp32(self):
        self.check_trt_result()


class TestConv2dPaddingAlgorithmTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = conv2d_python_api
        self.api_args = {
            "x": np.random.random([2, 3, 8, 8]).astype("float32"),
            "padding": "SAME",
            "stride": (1, 2),
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3, 8, 8]}
        self.opt_shape = {"x": [2, 3, 8, 8]}
        self.max_shape = {"x": [10, 3, 8, 8]}
        self.disable_passes = ['constant_folding_pass', 'conv2d_add_fuse_pass']

    def test_trt_result(self):
        self.check_trt_result()


class TestConv2dPaddingTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = conv2d_python_api

        self.api_args = {
            "x": np.random.random([2, 3, 8, 8]).astype("float32"),
            "padding": "VALID",
        }

        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3, 8, 8]}
        self.opt_shape = {"x": [2, 3, 8, 8]}
        self.max_shape = {"x": [10, 3, 8, 8]}
        self.disable_passes = ['constant_folding_pass', 'conv2d_add_fuse_pass']

    def test_trt_result(self):
        self.check_trt_result()


def conv2dtranspose_wrapper(
    x,
    stride=1,
    padding=0,
    output_padding=[],
    output_size=None,
    padding_algorithm="EXPLICIT",
    groups=1,
    dilation=1,
    data_format="NCDHW",
):
    if data_format == "AnyLayout":
        data_format = "NCDHW"
    if padding_algorithm is None:
        padding_algorithm = "EXPLICIT"
    weight = paddle.static.create_parameter(
        name="weight",
        shape=[3, 6, 3, 3],
        dtype="float32",
        default_initializer=paddle.nn.initializer.Normal(mean=0.0, std=1.0),
    )
    return _C_ops.conv2d_transpose(
        x,
        weight,
        stride,
        padding,
        output_padding,
        output_size,
        padding_algorithm,
        groups,
        dilation,
        data_format,
    )


class TestConv2dTransposeTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = conv2dtranspose_wrapper
        self.api_args = {
            "x": np.random.random([2, 3, 5, 5]).astype("float32"),
            "stride": [1, 1],
            "padding": [1, 1],
            "output_padding": [],
            "output_size": [7, 7],
            "padding_algorithm": "VALID",
            "groups": 1,
            "dilation": [1, 1],
            "data_format": "NCHW",
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3, 5, 5]}
        self.opt_shape = {"x": [2, 3, 5, 5]}
        self.max_shape = {"x": [4, 3, 5, 5]}

    def test_trt_result(self):
        self.check_trt_result()


class TestConv2dTransposePaddingAlgorithmTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = conv2dtranspose_wrapper
        self.api_args = {
            "x": np.random.random([2, 3, 5, 5]).astype("float32"),
            "stride": [1, 1],
            "padding": [1, 0, 1, 2],
            "output_padding": [],
            "output_size": None,
            "padding_algorithm": "SAME",
            "groups": 1,
            "dilation": [1, 1],
            "data_format": "NCHW",
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3, 5, 5]}
        self.opt_shape = {"x": [2, 3, 5, 5]}
        self.max_shape = {"x": [4, 3, 5, 5]}

    def test_trt_result(self):
        self.check_trt_result()


class TestConv2dTransposeOutputPaddingTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = conv2dtranspose_wrapper
        self.api_args = {
            "x": np.random.random([2, 3, 5, 5]).astype("float32"),
            "stride": [2, 2],
            "padding": [2, 2],
            "output_padding": [1, 1],
            "output_size": None,
            "padding_algorithm": "EXPLICIT",
            "groups": 1,
            "dilation": [1, 1],
            "data_format": "NCHW",
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3, 5, 5]}
        self.opt_shape = {"x": [2, 3, 5, 5]}
        self.max_shape = {"x": [4, 3, 5, 5]}


def depthwise_conv2d_wrapper(x):
    conv = paddle.nn.Conv2D(2, 2, (3, 3), groups=2)
    return conv(x)


def depthwise_conv2d_python_api(
    x, padding="SAME", stride=(1, 1), dilation=(1, 1)
):
    conv = paddle.nn.Conv2D(
        2,
        2,
        (3, 3),
        groups=2,
        padding=padding,
        stride=stride,
        dilation=dilation,
    )
    return conv(x)


class TestDepthwiseConv2dTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = depthwise_conv2d_wrapper
        self.api_args = {"x": np.random.random([3, 2, 8, 8]).astype("float32")}
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 2, 8, 8]}
        self.opt_shape = {"x": [3, 2, 8, 8]}
        self.max_shape = {"x": [10, 2, 8, 8]}

    def test_trt_result(self):
        self.check_trt_result()


class TestDepthwiseConv2dPaddingTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = depthwise_conv2d_python_api
        self.api_args = {
            "x": np.random.random([3, 2, 8, 8]).astype("float32"),
            "padding": "VALID",
            "stride": (1, 2),
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 2, 8, 8]}
        self.opt_shape = {"x": [3, 2, 8, 8]}
        self.max_shape = {"x": [10, 2, 8, 8]}

    def test_trt_result(self):
        self.check_trt_result()


class TestDepthwiseConv2dSameTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = depthwise_conv2d_python_api
        self.api_args = {
            "x": np.random.random([3, 2, 8, 8]).astype("float32"),
            "padding": "SAME",
            "stride": (1, 2),
            "dialation": (2, 2),
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 2, 8, 8]}
        self.opt_shape = {"x": [3, 2, 8, 8]}
        self.max_shape = {"x": [10, 2, 8, 8]}

    def test_trt_result(self):
        self.check_trt_result()


def depthwise_conv2d_transpose_wrapper(x):
    conv = paddle.nn.Conv2DTranspose(2, 2, (3, 3), groups=2)
    return conv(x)


def depthwise_conv2d_transpose_python_api(
    x, padding="SAME", stride=(1, 1), dilation=(1, 1)
):
    conv = paddle.nn.Conv2DTranspose(2, 2, (3, 3), groups=2)
    return conv(x)


class TestDepthwiseConv2dTransposeTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = depthwise_conv2d_transpose_wrapper
        self.api_args = {"x": np.random.random([3, 2, 8, 8]).astype("float32")}
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 2, 8, 8]}
        self.opt_shape = {"x": [3, 2, 8, 8]}
        self.max_shape = {"x": [10, 2, 8, 8]}

    def test_trt_result(self):
        self.check_trt_result()


class TestDepthwiseConv2dTransposeSameTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = depthwise_conv2d_transpose_python_api
        self.api_args = {
            "x": np.random.random([3, 2, 8, 8]).astype("float32"),
            "padding": "SAME",
            "stride": (1, 2),
            "dialation": (2, 2),
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 2, 8, 8]}
        self.opt_shape = {"x": [3, 2, 8, 8]}
        self.max_shape = {"x": [10, 2, 8, 8]}

    def test_trt_result(self):
        self.check_trt_result()


class TestDepthwiseConv2dTransposeValidTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = depthwise_conv2d_transpose_python_api
        self.api_args = {
            "x": np.random.random([3, 2, 8, 8]).astype("float32"),
            "padding": "VALID",
            "stride": (1, 2),
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 2, 8, 8]}
        self.opt_shape = {"x": [3, 2, 8, 8]}
        self.max_shape = {"x": [10, 2, 8, 8]}

    def test_trt_result(self):
        self.check_trt_result()


def conv3d_wrapper(x):
    conv = paddle.nn.Conv3D(3, 3, (3, 3, 3))
    return conv(x)


def conv3d_python_api(x, padding="SAME", stride=(1, 1, 1)):
    conv = paddle.nn.Conv3D(3, 3, (3, 3, 3), padding=padding, stride=stride)
    return conv(x)


class TestConv3dTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = conv3d_wrapper
        self.api_args = {
            "x": np.random.random([2, 3, 8, 8, 8]).astype("float32"),
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3, 8, 8, 8]}
        self.opt_shape = {"x": [1, 3, 8, 8, 8]}
        self.max_shape = {"x": [10, 3, 8, 8, 8]}

    def test_trt_result_fp16(self):
        self.check_trt_result(rtol=1e-3, atol=1e-3, precision_mode="fp16")

    def test_trt_result_fp32(self):
        self.check_trt_result()


class TestConv3dPaddingAlgorithmTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = conv3d_python_api
        self.api_args = {
            "x": np.random.random([2, 3, 8, 8, 8]).astype("float32"),
            "paddings": "SAME",
            "stride": (1, 1, 1),
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3, 8, 8, 8]}
        self.opt_shape = {"x": [1, 3, 8, 8, 8]}
        self.max_shape = {"x": [10, 3, 8, 8, 8]}

    def test_trt_result_fp16(self):
        self.check_trt_result(rtol=1e-3, atol=1e-3, precision_mode="fp16")

    def test_trt_result_fp32(self):
        self.check_trt_result()


def depthwise_conv3d_transpose_wrapper(x):
    conv = paddle.nn.Conv3DTranspose(
        in_channels=2, out_channels=2, kernel_size=(3, 3, 3)
    )
    return conv(x)


def depthwise_conv3d_transpose_python_api(
    x, padding="SAME", stride=(1, 1, 1), dilation=(1, 1, 1)
):
    conv = paddle.nn.Conv3DTranspose(
        in_channels=2,
        out_channels=2,
        kernel_size=(3, 3, 3),
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    return conv(x)


def depthwise_conv3d_transpose_wrapper_outpadding(x, output_padding):
    conv = paddle.nn.Conv3DTranspose(
        in_channels=3,
        out_channels=3,
        kernel_size=(3, 3, 3),
        stride=2,
        output_padding=output_padding,
    )
    return conv(x)


def conv3d_transpose_with_algorithm(x, algorithm):
    conv = paddle.nn.Conv3DTranspose(
        in_channels=3,
        out_channels=3,
        kernel_size=(3, 3, 3),
        padding=algorithm,
    )
    return conv(x)


class TestDepthwiseConv3dTransposeTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = depthwise_conv3d_transpose_wrapper
        self.api_args = {
            "x": np.random.random([3, 2, 8, 8, 8]).astype("float32")
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 2, 8, 8, 8]}
        self.opt_shape = {"x": [1, 2, 8, 8, 8]}
        self.max_shape = {"x": [10, 2, 8, 8, 8]}

    def test_trt_result_fp16(self):
        self.check_trt_result(rtol=1e-3, atol=1e-3, precision_mode="fp16")

    def test_trt_result_fp32(self):
        self.check_trt_result()


class TestDepthwiseConv3dTransposeSameTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = conv3d_transpose_with_algorithm
        self.api_args = {
            "x": np.random.random([2, 3, 8, 8, 8]).astype("float32"),
            "padding_algorithm": "SAME",
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3, 8, 8, 8]}
        self.opt_shape = {"x": [1, 3, 8, 8, 8]}
        self.max_shape = {"x": [10, 3, 8, 8, 8]}

    def test_trt_result_fp16(self):
        self.check_trt_result(rtol=1e-3, atol=1e-3, precision_mode="fp16")

    def test_trt_result_fp32(self):
        self.check_trt_result()


class TestDepthwiseConv3dTransposeOutputPaddingTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = depthwise_conv3d_transpose_wrapper_outpadding
        self.api_args = {
            "x": np.random.random([2, 3, 8, 8, 8]).astype("float32"),
            "output_padding": [1, 1, 1],
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3, 8, 8, 8]}
        self.opt_shape = {"x": [1, 3, 8, 8, 8]}
        self.max_shape = {"x": [10, 3, 8, 8, 8]}

    def test_trt_result(self):
        with self.assertRaises(ValueError) as context:
            self.check_trt_result()


class TestDepthwiseConv3dTransposeOutputPadding2TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = depthwise_conv3d_transpose_wrapper_outpadding
        self.api_args = {
            "x": np.random.random([2, 3, 8, 8, 8]).astype("float32"),
            "output_padding": [0, 0, 0],
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3, 8, 8, 8]}
        self.opt_shape = {"x": [1, 3, 8, 8, 8]}
        self.max_shape = {"x": [10, 3, 8, 8, 8]}

    def test_trt_result_fp16(self):
        self.check_trt_result(rtol=1e-3, atol=1e-3, precision_mode="fp16")

    def test_trt_result_fp32(self):
        self.check_trt_result()


class TestFusedConv2dAddActTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = conv2d_wrapper
        self.api_args = {
            "x": np.random.random([2, 3, 8, 8]).astype("float32"),
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3, 8, 8]}
        self.opt_shape = {"x": [2, 3, 8, 8]}
        self.max_shape = {"x": [10, 3, 8, 8]}
        self.disable_passes = []

    def test_trt_result_fp16(self):
        self.check_trt_result(precision_mode="fp16")

    def test_trt_result_fp32(self):
        self.check_trt_result()


if __name__ == '__main__':
    unittest.main()
