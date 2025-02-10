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


def dropout_wrapper(x, p, mode):
    out = _C_ops.dropout(
        x,
        None,
        p,
        True,
        mode,
        0,
        True,
    )
    return out


class TestDropoutWithUpscaleModeTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = dropout_wrapper
        self.api_args = {
            "x": np.random.random([1, 2, 3]).astype("float32"),
            "p": 0,
            "mode": "upscale_in_train",
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 2, 3]}
        self.opt_shape = {"x": [1, 2, 3]}
        self.max_shape = {"x": [10, 2, 3]}

    def test_trt_result(self):
        self.check_trt_result()


class TestDropoutWithDowngradeModeTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = dropout_wrapper
        self.api_args = {
            "x": np.random.random([1, 2, 3]).astype("float32"),
            "p": 0,
            "mode": "downgrade_in_infer",
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 2, 3]}
        self.opt_shape = {"x": [1, 2, 3]}
        self.max_shape = {"x": [10, 2, 3]}

    def test_trt_result(self):
        self.check_trt_result()


def upsample_bilinear(x):
    upsample = paddle.nn.Upsample(size=[12, 12], mode="bilinear")
    return upsample(x)


def bilinear_python_api(x, OutSize, SizeTensor, Scale, attrs):
    return _C_ops.bilinear_interp(
        x,
        OutSize,
        SizeTensor,
        Scale,
        attrs['data_layout'],
        attrs['out_d'],
        attrs['out_h'],
        attrs['out_w'],
        attrs['scale'] if 'scale' in attrs else [],
        attrs['interp_method'],
        attrs['align_corners'],
        attrs['align_mode'],
    )


def nearest_python_api(x, OutSize, SizeTensor, Scale, attrs):
    return _C_ops.nearest_interp(
        x,
        OutSize,
        SizeTensor,
        Scale,
        attrs['data_layout'],
        attrs['out_d'],
        attrs['out_h'],
        attrs['out_w'],
        attrs['scale'] if 'scale' in attrs else [],
        attrs['interp_method'],
        attrs['align_corners'],
        attrs['align_mode'],
    )


def embedding_python_api(x, weight, attrs):
    return _C_ops.embedding(
        x,
        weight,
        attrs['padding_idx'],
        attrs['sparse'],
    )


def unbind_python_api(x, attrs):
    return _C_ops.unbind(
        x,
        attrs['axis'],
    )


class TestBilinearScaleTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = bilinear_python_api
        self.api_args = {
            "x": np.random.random([2, 3, 6, 10]).astype("float32"),
            "OutSize": None,
            "SizeTensor": None,
            "Scale": None,
            "attrs": {
                "data_layout": "NCHW",
                "scale": [2.0, 2.0],
                "out_h": 12,
                "out_w": 12,
                "out_d": -1,
                "interp_method": "bilinear",
                "align_corners": True,
                "align_mode": 1,
            },
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [2, 3, 6, 10]}
        self.opt_shape = {"x": [2, 3, 6, 10]}
        self.max_shape = {"x": [12, 3, 6, 10]}

    def test_trt_result(self):
        self.check_trt_result()


class TestBilinearNHWCTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = bilinear_python_api
        x_nchw = np.random.random([2, 3, 6, 10]).astype("float32")
        x_nhwc = np.transpose(x_nchw, (0, 2, 3, 1))
        self.api_args = {
            "x": x_nhwc,
            "OutSize": None,
            "SizeTensor": None,
            "Scale": None,
            "attrs": {
                "data_layout": "NHWC",
                "scale": [],
                "out_h": 12,
                "out_w": 12,
                "out_d": -1,
                "interp_method": "bilinear",
                "align_corners": False,
                "align_mode": 0,
            },
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [2, 6, 10, 3]}
        self.opt_shape = {"x": [2, 6, 10, 3]}
        self.max_shape = {"x": [12, 6, 10, 3]}

    def test_trt_result(self):
        self.check_trt_result()


class TestBilinearOutSizeTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = bilinear_python_api
        self.api_args = {
            "x": np.random.random([2, 3, 6, 10]).astype("float32"),
            "OutSize": np.array([12, 12], dtype="int32"),
            "SizeTensor": None,
            "Scale": None,
            "attrs": {
                "data_layout": "NCHW",
                "scale": [],
                "out_h": 12,
                "out_w": 12,
                "out_d": -1,
                "interp_method": "bilinear",
                "align_corners": False,
                "align_mode": 0,
            },
        }
        self.program_config = {"feed_list": ["x", "OutSize"]}
        self.min_shape = {"x": [2, 3, 6, 10]}
        self.opt_shape = {"x": [2, 3, 6, 10]}
        self.max_shape = {"x": [12, 3, 6, 10]}

    def test_trt_result(self):
        self.check_trt_result()


class TestBilinearSizeTensorTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = bilinear_python_api
        self.api_args = {
            "x": np.random.random([2, 3, 6, 10]).astype("float32"),
            "OutSize": None,
            "SizeTensor": [
                np.array([2], dtype="int64"),
                np.array([2], dtype="int64"),
            ],
            "Scale": None,
            "attrs": {
                "data_layout": "NCHW",
                "scale": [],
                "out_h": -1,
                "out_w": -1,
                "out_d": -1,
                "interp_method": "bilinear",
                "align_corners": False,
                "align_mode": 0,
            },
        }
        self.program_config = {
            "feed_list": ["x", "OutSize", "SizeTensor", "Scale"]
        }
        self.min_shape = {"x": [2, 3, 6, 10]}
        self.opt_shape = {"x": [2, 3, 6, 10]}
        self.max_shape = {"x": [12, 3, 6, 10]}

    def test_trt_result(self):
        self.check_trt_result()


class TestNearestNHWCTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = nearest_python_api
        x_nchw = np.random.random([2, 3, 6, 10]).astype("float32")
        x_nhwc = np.transpose(x_nchw, (0, 2, 3, 1))
        self.api_args = {
            "x": x_nhwc,
            "OutSize": None,
            "SizeTensor": None,
            "Scale": None,
            "attrs": {
                "data_layout": "NHWC",
                "scale": [],
                "out_h": 12,
                "out_w": 12,
                "out_d": -1,
                "interp_method": "nearest",
                "align_corners": False,
                "align_mode": 1,
            },
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [2, 6, 10, 3]}
        self.opt_shape = {"x": [2, 6, 10, 3]}
        self.max_shape = {"x": [12, 6, 10, 3]}

    def test_trt_result(self):
        self.check_trt_result()


class TestNearestSizeTensorTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = nearest_python_api
        x_nchw = np.random.random([2, 3, 6, 10]).astype("float32")
        self.api_args = {
            "x": x_nchw,
            "OutSize": None,
            "SizeTensor": [
                np.array([12], dtype="int64"),
                np.array([12], dtype="int64"),
            ],
            "Scale": None,
            "attrs": {
                "data_layout": "NCHW",
                "scale": [],
                "out_h": 12,
                "out_w": 12,
                "out_d": -1,
                "interp_method": "nearest",
                "align_corners": False,
                "align_mode": 0,
            },
        }
        self.program_config = {"feed_list": ["x", "SizeTensor"]}
        self.min_shape = {"x": [2, 3, 6, 10]}
        self.opt_shape = {"x": [2, 3, 6, 10]}
        self.max_shape = {"x": [12, 3, 6, 10]}

    def test_trt_result(self):
        self.check_trt_result()


class TestEmbeddingTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = embedding_python_api
        x = np.array([[3, 16, 24], [6, 4, 47]]).astype(np.int64)
        weight = np.random.uniform(-1, 1, [64, 4]).astype('float32')
        self.api_args = {
            "x": x,
            "weight": weight,
            "attrs": {
                "padding_idx": -1,
                "sparse": False,
            },
        }
        self.dynamic_shape_data = {
            "x": lambda shape: np.random.randint(1, 64, size=shape).astype(
                "int64"
            ),
            "weight": lambda shape: np.random.randint(-1, 1, size=shape).astype(
                "float32"
            ),
        }
        self.program_config = {"feed_list": ["x", "weight"]}
        self.min_shape = {"x": [1, 3], "weight": [64, 4]}
        self.opt_shape = {"x": [2, 3], "weight": [64, 4]}
        self.max_shape = {"x": [16, 3], "weight": [64, 4]}

    def test_trt_result(self):
        self.check_trt_result()

    def test_trt_result_fp16(self):
        self.check_trt_result(precision_mode="fp16")


class TestUnbindTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = unbind_python_api
        x = np.random.random([3, 400, 196, 80]).astype(np.float32)
        self.api_args = {
            "x": x,
            "attrs": {
                "axis": 1,
            },
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {
            "x": [1, 400, 196, 80],
        }
        self.opt_shape = {"x": [2, 400, 196, 80]}
        self.max_shape = {"x": [3, 400, 196, 80]}

    def test_trt_result(self):
        self.check_trt_result()

    def test_trt_result_fp16(self):
        self.check_trt_result(precision_mode="fp16")


class TestNearestOutAndScaleTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = nearest_python_api
        x_nchw = np.random.random([2, 3, 6, 10]).astype("float32")
        self.api_args = {
            "x": x_nchw,
            "OutSize": None,
            "SizeTensor": None,
            "Scale": None,
            "attrs": {
                "data_layout": "NCHW",
                "scale": [2, 2],
                "out_h": 12,
                "out_w": 12,
                "out_d": -1,
                "interp_method": "nearest",
                "align_corners": True,
                "align_mode": 1,
            },
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [2, 3, 6, 10]}
        self.opt_shape = {"x": [2, 3, 6, 10]}
        self.max_shape = {"x": [12, 3, 6, 10]}

    def test_trt_result(self):
        self.check_trt_result()


class TestBilinearTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = upsample_bilinear
        self.api_args = {"x": np.random.random([2, 3, 6, 10]).astype("float32")}
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [2, 3, 6, 10]}
        self.opt_shape = {"x": [2, 3, 6, 10]}
        self.max_shape = {"x": [12, 3, 6, 10]}

    def test_trt_result(self):
        self.check_trt_result()


def upsample_nearest(x):
    upsample = paddle.nn.Upsample(size=[12, 12], mode="nearest")
    return upsample(x)


class TestNearestInterpTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = upsample_nearest
        self.api_args = {"x": np.random.random([2, 3, 6, 10]).astype("float32")}
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [2, 3, 6, 10]}
        self.opt_shape = {"x": [2, 3, 6, 10]}
        self.max_shape = {"x": [12, 3, 6, 10]}

    def test_trt_result(self):
        self.check_trt_result()


def linear_interp_test(
    x,
    OutSize=None,
    SizeTensor=None,
    Scale=None,
    data_layout='NCHW',
    out_d=-1,
    out_h=-1,
    out_w=-1,
    scale=[],
    interp_method='linear',
    align_corners=True,
    align_mode=0,
):
    return paddle._C_ops.linear_interp(
        x,
        OutSize,
        SizeTensor,
        Scale,
        data_layout,
        out_d,
        out_h,
        out_w,
        scale,
        interp_method,
        align_corners,
        align_mode,
    )


class TestLinearInterpTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = linear_interp_test
        self.api_args = {
            "x": np.random.random([1, 18, 144]).astype("float32"),
            "OutSize": None,
            "SizeTensor": None,
            "Scale": None,
            "data_layout": "NCHW",
            "out_d": -1,
            "out_h": -1,
            "out_w": 288,
            "scale": [],
            "interp_method": "linear",
            "align_corners": False,
            "align_mode": 0,
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 18, 144]}
        self.opt_shape = {"x": [2, 18, 144]}
        self.max_shape = {"x": [3, 18, 144]}

    def test_trt_result(self):
        self.check_trt_result()

    def test_fp16_trt_result(self):
        self.check_trt_result(precision_mode="fp16")


class TestLinearInterpCase1TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = linear_interp_test
        self.api_args = {
            "x": np.random.random([1, 18, 144]).astype("float32"),
            "OutSize": None,
            "SizeTensor": None,
            "Scale": None,
            "data_layout": "NHWC",
            "out_d": -1,
            "out_h": -1,
            "out_w": 288,
            "scale": [],
            "interp_method": "linear",
            "align_corners": False,
            "align_mode": 0,
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 18, 144]}
        self.opt_shape = {"x": [2, 18, 144]}
        self.max_shape = {"x": [3, 18, 144]}

    def test_trt_result(self):
        self.check_trt_result()

    def test_fp16_trt_result(self):
        self.check_trt_result(precision_mode="fp16")


class TestLinearInterpCase2TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = linear_interp_test
        self.api_args = {
            "x": np.random.random([1, 18, 144]).astype("float32"),
            "OutSize": None,
            "SizeTensor": None,
            "Scale": None,
            "data_layout": "NHWC",
            "out_d": -1,
            "out_h": -1,
            "out_w": 288,
            "scale": [],
            "interp_method": "linear",
            "align_corners": False,
            "align_mode": 0,
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 18, 144]}
        self.opt_shape = {"x": [2, 18, 144]}
        self.max_shape = {"x": [3, 18, 144]}

    def test_trt_result(self):
        self.check_trt_result()

    def test_fp16_trt_result(self):
        self.check_trt_result(precision_mode="fp16")


class TestLinearInterpCase3TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = linear_interp_test
        self.api_args = {
            "x": np.random.random([1, 18, 144]).astype("float32"),
            "OutSize": None,
            "SizeTensor": None,
            "Scale": None,
            "data_layout": "NHWC",
            "out_d": -1,
            "out_h": -1,
            "out_w": 288,
            "scale": [],
            "interp_method": "linear",
            "align_corners": True,
            "align_mode": 0,
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 18, 144]}
        self.opt_shape = {"x": [2, 18, 144]}
        self.max_shape = {"x": [3, 18, 144]}

    def test_trt_result(self):
        self.check_trt_result()

    def test_fp16_trt_result(self):
        self.check_trt_result(precision_mode="fp16")


class TestLinearInterpCase4TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = linear_interp_test
        self.api_args = {
            "x": np.random.random([1, 3, 64]).astype("float32"),
            "OutSize": None,
            "SizeTensor": None,
            "Scale": None,
            "data_layout": "NCHW",
            "out_d": -1,
            "out_h": -1,
            "out_w": -1,
            "scale": [1.0],
            "interp_method": "linear",
            "align_corners": False,
            "align_mode": 0,
        }
        self.program_config = {"feed_list": ["x", "Scale"]}
        self.min_shape = {"x": [1, 3, 64]}
        self.opt_shape = {"x": [2, 3, 64]}
        self.max_shape = {"x": [4, 3, 64]}

    def test_trt_result(self):
        self.check_trt_result()

    def test_fp16_trt_result(self):
        self.check_trt_result(precision_mode="fp16")


class TestLinearInterpCase5TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = linear_interp_test
        self.api_args = {
            "x": np.random.random([1, 3, 64]).astype("float32"),
            "OutSize": None,
            "SizeTensor": None,
            "Scale": None,
            "data_layout": "NHWC",
            "out_d": -1,
            "out_h": -1,
            "out_w": -1,
            "scale": [1.0],
            "interp_method": "linear",
            "align_corners": True,
            "align_mode": 0,
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3, 64]}
        self.opt_shape = {"x": [2, 3, 64]}
        self.max_shape = {"x": [4, 3, 64]}

    def test_trt_result(self):
        self.check_trt_result()

    def test_fp16_trt_result(self):
        self.check_trt_result(precision_mode="fp16")


class TestLinearInterpCase6TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = linear_interp_test
        self.api_args = {
            "x": np.random.random([1, 18, 144]).astype("float32"),
            "OutSize": np.array([288], dtype="int32"),
            "SizeTensor": [
                np.array([288], dtype="int64"),
            ],
            "Scale": None,
            "data_layout": "NHWC",
            "out_d": -1,
            "out_h": -1,
            "out_w": 288,
            "scale": [],
            "interp_method": "linear",
            "align_corners": True,
            "align_mode": 0,
        }
        self.program_config = {"feed_list": ["x", "OutSize", "SizeTensor"]}
        self.min_shape = {"x": [1, 18, 144]}
        self.opt_shape = {"x": [2, 18, 144]}
        self.max_shape = {"x": [4, 18, 144]}

    def test_trt_result(self):
        self.check_trt_result()

    def test_fp16_trt_result(self):
        self.check_trt_result(precision_mode="fp16")


class TestLinearInterpCase7TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = linear_interp_test
        self.api_args = {
            "x": np.random.random([1, 18, 144]).astype("float32"),
            "OutSize": np.array([288], dtype="int32"),
            "SizeTensor": [
                np.array([288], dtype="int64"),
            ],
            "Scale": None,
            "data_layout": "NCHW",
            "out_d": -1,
            "out_h": -1,
            "out_w": 288,
            "scale": [],
            "interp_method": "linear",
            "align_corners": True,
            "align_mode": 0,
        }
        self.program_config = {"feed_list": ["x", "OutSize", "SizeTensor"]}
        self.min_shape = {"x": [1, 18, 144]}
        self.opt_shape = {"x": [2, 18, 144]}
        self.max_shape = {"x": [4, 18, 144]}

    def test_trt_result(self):
        self.check_trt_result()

    def test_fp16_trt_result(self):
        self.check_trt_result(precision_mode="fp16")


class TestLinearInterpCase8TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = linear_interp_test
        self.api_args = {
            "x": np.random.random([1, 18, 144]).astype("float32"),
            "OutSize": None,
            "SizeTensor": [
                np.array([288], dtype="int64"),
            ],
            "Scale": None,
            "data_layout": "NCHW",
            "out_d": -1,
            "out_h": -1,
            "out_w": 288,
            "scale": [],
            "interp_method": "linear",
            "align_corners": True,
            "align_mode": 0,
        }
        self.program_config = {"feed_list": ["x", "SizeTensor"]}
        self.min_shape = {"x": [1, 18, 144]}
        self.opt_shape = {"x": [2, 18, 144]}
        self.max_shape = {"x": [4, 18, 144]}

    def test_trt_result(self):
        self.check_trt_result()

    def test_fp16_trt_result(self):
        self.check_trt_result(precision_mode="fp16")


if __name__ == "__main__":
    unittest.main()
