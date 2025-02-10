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


def api_wrapper(x):
    return paddle._C_ops.share_data(x)


def multiclass_nms3(
    bboxes,
    scores,
    rois_num=None,
    score_threshold=0.3,
    nms_top_k=4,
    keep_top_k=1,
    nms_threshold=0.3,
    normalized=True,
    nms_eta=1.5,
    background_label=-1,
    return_index=False,
    return_rois_num=True,
    name=None,
):
    attrs = (
        score_threshold,
        nms_top_k,
        keep_top_k,
        nms_threshold,
        normalized,
        nms_eta,
        background_label,
    )
    output, index, nms_rois_num = _C_ops.multiclass_nms3(
        bboxes, scores, rois_num, *attrs
    )
    if not return_index:
        index = None
    return output, nms_rois_num, index


class TestMulticlassNMS3TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = multiclass_nms3
        self.api_args = {
            "bboxes": np.random.randn(2, 5, 4).astype("float32"),
            "scores": np.random.randn(2, 4, 5).astype("float32"),
        }
        self.program_config = {"feed_list": ["bboxes", "scores"]}
        self.min_shape = {"bboxes": [1, 5, 4], "scores": [1, 4, 5]}
        self.opt_shape = {"bboxes": [2, 5, 4], "scores": [2, 4, 5]}
        self.max_shape = {"bboxes": [3, 5, 4], "scores": [3, 4, 5]}

    def test_trt_result(self):
        self.check_trt_result()


class TestMulticlassNMS3Marker(TensorRTBaseTest):
    def setUp(self):
        self.python_api = multiclass_nms3
        self.api_args = {
            "bboxes": np.random.randn(2, 5, 4, 1).astype("float32"),
            "scores": np.random.randn(2, 4, 5, 1).astype("float32"),
        }
        self.program_config = {"feed_list": ["bboxes", "scores"]}
        self.target_marker_op = "pd_op.multiclass_nms3"

    def test_trt_result(self):
        self.check_marker(expected_result=False)


def set_value(
    x, starts, ends, steps, axes, decrease_axes, none_axes, shape, values
):
    output = _C_ops.set_value(
        x,
        starts,
        ends,
        steps,
        axes,
        decrease_axes,
        none_axes,
        shape,
        values,
    )
    return output


def set_value_(
    x, starts, ends, steps, axes, decrease_axes, none_axes, shape, values
):
    output = _C_ops.set_value_(
        x,
        starts,
        ends,
        steps,
        axes,
        decrease_axes,
        none_axes,
        shape,
        values,
    )
    return output


def set_value_with_tensor(
    x, values, starts, ends, steps, axes, decrease_axes, none_axes, shape
):
    output = _C_ops.set_value_with_tensor(
        x,
        values,
        starts,
        ends,
        steps,
        axes,
        decrease_axes,
        none_axes,
        shape,
    )
    return output


def set_value_with_tensor_(
    x, values, starts, ends, steps, axes, decrease_axes, none_axes, shape
):
    output = _C_ops.set_value_with_tensor_(
        x,
        values,
        starts,
        ends,
        steps,
        axes,
        decrease_axes,
        none_axes,
        shape,
    )
    return output


class TestSetValueTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = set_value
        self.api_args = {
            "x": np.ones([10, 2]).astype("float32"),
            "starts": [0],
            "ends": [1],
            "steps": [1],
            "axes": [1],
            "decrease_axes": [],
            "none_axes": [],
            "shape": [],
            "values": [10.0],
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 2]}
        self.opt_shape = {"x": [2, 2]}
        self.max_shape = {"x": [20, 2]}

    def test_trt_result(self):
        self.check_trt_result()


# starts/ends/steps is not one element
class TestSetValueMarkerCase1(TensorRTBaseTest):
    def setUp(self):
        self.python_api = set_value
        self.api_args = {
            "x": np.ones([10, 2]).astype("float32"),
            "starts": [0, 0],
            "ends": [1, 1],
            "steps": [1, 1],
            "axes": [1],
            "decrease_axes": [],
            "none_axes": [],
            "shape": [],
            "values": [10.0],
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 2]}
        self.opt_shape = {"x": [2, 2]}
        self.max_shape = {"x": [5, 2]}

    def test_trt_result(self):
        self.check_marker(expected_result=False)


# decrease_axes has element
class TestSetValueMarkerCase2(TensorRTBaseTest):
    def setUp(self):
        self.python_api = set_value
        self.api_args = {
            "x": np.ones([10, 2]).astype("float32"),
            "starts": [0],
            "ends": [1],
            "steps": [1],
            "axes": [1],
            "decrease_axes": [1],
            "none_axes": [],
            "shape": [],
            "values": [10.0],
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 2]}
        self.opt_shape = {"x": [2, 2]}
        self.max_shape = {"x": [20, 2]}

    def test_trt_result(self):
        self.check_marker(expected_result=False)


# values has more than one element
class TestSetValueMarkerCase3(TensorRTBaseTest):
    def setUp(self):
        self.python_api = set_value
        self.api_args = {
            "x": np.ones([10, 2]).astype("float32"),
            "starts": [0],
            "ends": [1],
            "steps": [1],
            "axes": [1],
            "decrease_axes": [],
            "none_axes": [],
            "shape": [],
            "values": [10.0, 0],
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 2]}
        self.opt_shape = {"x": [2, 2]}
        self.max_shape = {"x": [20, 2]}

    def test_trt_result(self):
        self.check_marker(expected_result=False)


# values has int element
class TestSetValueMarkerCase4(TensorRTBaseTest):
    def setUp(self):
        self.python_api = set_value
        self.api_args = {
            "x": np.ones([10, 2]).astype("float32"),
            "starts": [0],
            "ends": [1],
            "steps": [1],
            "axes": [1],
            "decrease_axes": [],
            "none_axes": [],
            "shape": [],
            "values": [10],
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 2]}
        self.opt_shape = {"x": [2, 2]}
        self.max_shape = {"x": [20, 2]}

    def test_trt_result(self):
        self.check_marker(expected_result=False)


# starts is not constant value
class TestSetValueMarkerCase5(TensorRTBaseTest):
    def setUp(self):
        self.python_api = set_value
        self.api_args = {
            "x": np.ones([10, 2]).astype("float32"),
            "starts": np.zeros([1]).astype("int64"),
            "ends": [1],
            "steps": [1],
            "axes": [1],
            "decrease_axes": [],
            "none_axes": [],
            "shape": [],
            "values": [10.0],
        }
        self.program_config = {"feed_list": ["x", "starts"]}
        self.min_shape = {"x": [1, 2]}
        self.opt_shape = {"x": [2, 2]}
        self.max_shape = {"x": [20, 2]}

    def test_trt_result(self):
        self.check_marker(expected_result=False)


class TestSetValue_TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = set_value_
        self.api_args = {
            "x": np.ones([10, 2]).astype("float32"),
            "starts": [0],
            "ends": [1],
            "steps": [1],
            "axes": [1],
            "decrease_axes": [],
            "none_axes": [],
            "shape": [],
            "values": [10.0],
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 2]}
        self.opt_shape = {"x": [2, 2]}
        self.max_shape = {"x": [20, 2]}

    def test_trt_result(self):
        self.check_trt_result()


class TestSetValueWithTensorTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = set_value_with_tensor
        self.api_args = {
            "x": np.ones([2, 3, 3]).astype("float32"),
            "values": np.random.randn(2, 2, 3).astype("float32"),
            "starts": [0],
            "ends": [2],
            "steps": [1],
            "axes": [1],
            "decrease_axes": [],
            "none_axes": [],
            "shape": [],
        }
        self.program_config = {"feed_list": ["x", "values"]}
        self.min_shape = {"x": [1, 3, 3], "values": [1, 2, 3]}
        self.opt_shape = {"x": [2, 3, 3], "values": [2, 2, 3]}
        self.max_shape = {"x": [4, 3, 3], "values": [4, 2, 3]}

    def test_trt_result(self):
        self.check_trt_result()


# values is int type
class TestSetValueWithTensorMarkerCase1(TensorRTBaseTest):
    def setUp(self):
        self.python_api = set_value_with_tensor
        self.api_args = {
            "x": np.ones([2, 3, 3]).astype("float32"),
            "values": np.random.randn(2, 2, 3).astype("int32"),
            "starts": [0],
            "ends": [2],
            "steps": [1],
            "axes": [1],
            "decrease_axes": [],
            "none_axes": [],
            "shape": [],
        }
        self.program_config = {"feed_list": ["x", "values"]}
        self.min_shape = {"x": [1, 3, 3], "values": [1, 2, 3]}
        self.opt_shape = {"x": [2, 3, 3], "values": [2, 2, 3]}
        self.max_shape = {"x": [4, 3, 3], "values": [4, 2, 3]}

    def test_trt_result(self):
        self.check_marker(expected_result=False)


class TestSetValueWithTensor_TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = set_value_with_tensor_
        self.api_args = {
            "x": np.ones([2, 3, 3]).astype("float32"),
            "values": np.random.randn(2, 2, 3).astype("float32"),
            "starts": [0],
            "ends": [2],
            "steps": [1],
            "axes": [1],
            "decrease_axes": [],
            "none_axes": [],
            "shape": [],
        }
        self.program_config = {"feed_list": ["x", "values"]}
        self.min_shape = {"x": [1, 3, 3], "values": [1, 2, 3]}
        self.opt_shape = {"x": [2, 3, 3], "values": [2, 2, 3]}
        self.max_shape = {"x": [4, 3, 3], "values": [4, 2, 3]}

    def test_trt_result(self):
        self.check_trt_result()


class TestShareDataTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = api_wrapper
        self.api_args = {
            "x": np.random.rand(4, 3, 5).astype("float32"),
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [4, 3, 5]}
        self.opt_shape = {"x": [5, 3, 5]}
        self.max_shape = {"x": [6, 3, 5]}

    def test_trt_result(self):
        self.check_trt_result()


class TestTemporalShiftTRTPatternBasic(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.nn.functional.temporal_shift
        self.api_args = {
            "x": np.random.random([4, 9, 7, 7]).astype(np.float32),
            "seg_num": 2,
            "shift_ratio": 0.2,
            "data_format": "NCHW",
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [2, 9, 7, 7]}
        self.opt_shape = {"x": [2, 9, 7, 7]}
        self.max_shape = {"x": [8, 9, 7, 7]}

    def test_trt_result_fp16(self):
        self.check_trt_result(precision_mode="fp16")

    def test_trt_result_fp32(self):
        self.check_trt_result()


class TestTemporalShiftTRTPatternZeroSlice(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.nn.functional.temporal_shift
        self.api_args = {
            "x": np.random.random([4, 2, 7, 7]).astype(np.float32),
            "seg_num": 2,
            "shift_ratio": 0.2,
            "data_format": "NCHW",
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [2, 2, 7, 7]}
        self.opt_shape = {"x": [2, 2, 7, 7]}
        self.max_shape = {"x": [8, 2, 7, 7]}

    def test_trt_result_fp16(self):
        self.check_trt_result(precision_mode="fp16")

    def test_trt_result_fp32(self):
        self.check_trt_result()


class TestTemporalShiftTRTPatternDifferentSegNum(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.nn.functional.temporal_shift
        self.api_args = {
            "x": np.random.random([4, 9, 7, 7]).astype(np.float32),
            "seg_num": 4,
            "shift_ratio": 0.2,
            "data_format": "NCHW",
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [4, 9, 7, 7]}
        self.opt_shape = {"x": [4, 9, 7, 7]}
        self.max_shape = {"x": [8, 9, 7, 7]}

    def test_trt_result_fp16(self):
        self.check_trt_result(precision_mode="fp16")

    def test_trt_result_fp32(self):
        self.check_trt_result()


class TestTemporalShiftTRTPatternDifferentShiftRatio(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.nn.functional.temporal_shift
        self.api_args = {
            "x": np.random.random([4, 9, 7, 7]).astype(np.float32),
            "seg_num": 2,
            "shift_ratio": 0.4,
            "data_format": "NCHW",
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [2, 9, 7, 7]}
        self.opt_shape = {"x": [2, 9, 7, 7]}
        self.max_shape = {"x": [8, 9, 7, 7]}

    def test_trt_result_fp16(self):
        self.check_trt_result(precision_mode="fp16")

    def test_trt_result_fp32(self):
        self.check_trt_result()


class TestTemporalShiftTRTPatternDifferentDataFormat(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.nn.functional.temporal_shift
        self.api_args = {
            "x": np.random.random([4, 9, 7, 7]).astype(np.float32),
            "seg_num": 2,
            "shift_ratio": 0.2,
            "name": None,
            "data_format": "NHWC",
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [2, 9, 7, 7]}
        self.opt_shape = {"x": [2, 9, 7, 7]}
        self.max_shape = {"x": [8, 9, 7, 7]}

    def test_trt_result_fp16(self):
        self.check_trt_result(precision_mode="fp16")

    def test_trt_result_fp32(self):
        self.check_trt_result()


class TestTemporalShiftTRTPatternMinMaxShape(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.nn.functional.temporal_shift
        self.api_args = {
            "x": np.random.random([4, 9, 7, 7]).astype(np.float32),
            "seg_num": 2,
            "shift_ratio": 0.2,
            "data_format": "NCHW",
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [2, 9, 7, 7]}
        self.opt_shape = {"x": [2, 9, 7, 7]}
        self.max_shape = {"x": [10, 9, 7, 7]}

    def test_trt_result_fp16(self):
        self.check_trt_result(precision_mode="fp16")

    def test_trt_result_fp32(self):
        self.check_trt_result()


def wrapper_temporal_shift(x):
    return paddle.nn.functional.temporal_shift(x=x, seg_num=2, shift_ratio=0.2)


class TestTemporalShiftTRTPatternError1(TensorRTBaseTest):
    def setUp(self):
        self.python_api = wrapper_temporal_shift
        self.api_args = {
            "x": np.random.random([4, 9, 7, 7]).astype(np.float32),
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [2, 9, 7, 7]}
        self.opt_shape = {"x": [2, 9, 7, 7]}
        self.max_shape = {"x": [10, 9, 7, 7]}

    def test_trt_result(self):
        self.check_marker(expected_result=False)


def affine_channel(x, scale_shape, bias_shape, layout):
    scale = paddle.static.create_parameter(
        shape=scale_shape, dtype='float32', name="scale"
    )
    bias = paddle.static.create_parameter(
        shape=bias_shape, dtype='float32', name="bias"
    )
    return _C_ops.affine_channel(x, scale, bias, layout)


class TestAffineChannelTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = affine_channel
        self.api_args = {
            "x": np.random.random((2, 100, 3, 3)).astype("float32"),
            "scale_shape": [100],
            "bias_shape": [100],
            "layout": "NCHW",
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 100, 3, 3]}
        self.opt_shape = {"x": [2, 100, 3, 3]}
        self.max_shape = {"x": [3, 100, 3, 3]}

    def test_fp32_trt_result(self):
        self.check_trt_result()

    def test_fp16_trt_result(self):
        self.check_trt_result(precision_mode="fp16")


class TestAffineChannelCase1TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = affine_channel
        self.api_args = {
            "x": np.random.random((2, 3, 3, 100)).astype("float32"),
            "scale_shape": [100],
            "bias_shape": [100],
            "layout": "NHWC",
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3, 3, 100]}
        self.opt_shape = {"x": [2, 3, 3, 100]}
        self.max_shape = {"x": [3, 3, 3, 100]}

    def test_fp32_trt_result(self):
        self.check_trt_result()

    def test_fp16_trt_result(self):
        self.check_trt_result(precision_mode="fp16")


def anchor_generator(x, anchor_sizes, aspect_ratios, variances, stride, offset):
    return _C_ops.anchor_generator(
        x, anchor_sizes, aspect_ratios, variances, stride, offset
    )


class TestAnchorGeneratorTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = anchor_generator
        self.api_args = {
            "x": np.random.random((2, 3, 3, 100)).astype("float32"),
            "anchor_sizes": [64.0, 128.0, 256.0],
            "aspect_ratios": [0.5, 1, 2],
            "variances": [1.0, 1.0, 1.0, 1.0],
            "stride": [16.0, 16.0],
            "offset": 0.5,
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3, 3, 100]}
        self.opt_shape = {"x": [2, 3, 3, 100]}
        self.max_shape = {"x": [3, 3, 3, 100]}

    def test_fp32_trt_result(self):
        self.check_trt_result()

    def test_fp16_trt_result(self):
        self.check_trt_result(precision_mode="fp16")


class TestAnchorGeneratorCase1TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = anchor_generator
        self.api_args = {
            "x": np.random.random((2, 3, 64, 64)).astype("float32"),
            "anchor_sizes": [64.0, 128.0, 256.0],
            "aspect_ratios": [0.4, 1.2, 3],
            "variances": [0.5, 1.0, 0.5, 1.0],
            "stride": [16.0, 32.0],
            "offset": 0.8,
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [2, 3, 64, 64]}
        self.opt_shape = {"x": [2, 3, 64, 64]}
        self.max_shape = {"x": [3, 3, 64, 64]}

    def test_fp32_trt_result(self):
        self.check_trt_result()

    def test_fp16_trt_result(self):
        self.check_trt_result(precision_mode="fp16")


def shuffle_channel_wrapper(x, group=1):
    return _C_ops.shuffle_channel(x, group)


class TestShuffleChannelTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = shuffle_channel_wrapper
        self.api_args = {
            "x": np.random.random((10, 16, 4, 4)).astype("float32"),
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [10, 16, 4, 4]}
        self.opt_shape = {"x": [10, 16, 4, 4]}
        self.max_shape = {"x": [10, 16, 4, 4]}

    def test_fp32_trt_result(self):
        self.check_trt_result()

    def test_fp16_trt_result(self):
        self.check_trt_result(precision_mode="fp16")


def full_batch_size_like_wrapper(x, dtype, value, batch_dim):
    place = paddle.CPUPlace()
    out_shape = [-1, 5, 1]
    return _C_ops.full_batch_size_like(
        x, out_shape, dtype, value, batch_dim, batch_dim, place
    )


class TestFullBatchSizeLikeTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = full_batch_size_like_wrapper
        self.api_args = {
            "x": np.random.random((2, 3, 4)).astype("float32"),
            "dtype": paddle.float32,
            "value": 2.0,
            "batch_dim": 0,
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [2, 3, 4]}
        self.opt_shape = {"x": [3, 3, 4]}
        self.max_shape = {"x": [4, 3, 4]}

    def test_fp32_trt_result(self):
        self.check_trt_result()

    def test_fp16_trt_result(self):
        self.check_trt_result(precision_mode="fp16")


if __name__ == '__main__':
    unittest.main()
