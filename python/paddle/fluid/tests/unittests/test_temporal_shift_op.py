#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import OpTest, convert_float_to_uint16

import paddle
from paddle.fluid import core


def temporal_shift(x, seg_num, shift_ratio, data_format):
    if data_format == "NHWC":
        x = np.transpose(x, (0, 3, 1, 2))
    shape = x.shape
    reshape_x = x.reshape((-1, seg_num, shape[1], shape[2], shape[3]))
    pad_x = np.pad(
        reshape_x, ((0, 0), (1, 1), (0, 0), (0, 0), (0, 0)), 'constant'
    )
    c1 = int(shape[1] * shift_ratio)
    c2 = int(shape[1] * 2 * shift_ratio)
    slice1 = pad_x[:, :seg_num, :c1, :, :]
    slice2 = pad_x[:, 2 : seg_num + 2, c1:c2, :, :]
    slice3 = pad_x[:, 1 : seg_num + 1, c2:, :, :]
    concat_x = np.concatenate([slice1, slice2, slice3], axis=2)
    out = concat_x.reshape(shape)
    if data_format == "NHWC":
        out = np.transpose(out, (0, 2, 3, 1))
    return out


class TestTemporalShift(OpTest):
    def setUp(self):
        self.initTestCase()
        self.op_type = 'temporal_shift'
        self.python_api = paddle.nn.functional.temporal_shift
        x = np.random.random(self.x_shape).astype(self.dtype)

        self.attrs = {
            "seg_num": self.seg_num,
            "shift_ratio": self.shift_ratio,
            "data_format": self.data_format,
        }

        self.inputs = {
            "X": x,
        }

        output = temporal_shift(
            x, self.seg_num, self.shift_ratio, self.data_format
        )
        self.outputs = {"Out": output}
        self.python_out_sig = ["Out"]

    def test_check_output(self):
        self.check_output(check_eager=True)

    def test_check_grad_ignore_uv(self):
        self.check_grad(['X'], 'Out', check_eager=True)

    def initTestCase(self):
        self.x_shape = (6, 4, 4, 4)
        self.seg_num = 3
        self.shift_ratio = 0.25
        self.dtype = 'float64'
        self.data_format = 'NCHW'


class TestTemporalShift2(TestTemporalShift):
    def initTestCase(self):
        self.x_shape = (4, 9, 7, 7)
        self.seg_num = 2
        self.shift_ratio = 0.2
        self.data_format = 'NCHW'


class TestTemporalShift3(TestTemporalShift):
    def initTestCase(self):
        self.x_shape = (3, 10, 5, 5)
        self.seg_num = 1
        self.shift_ratio = 0.3
        self.data_format = 'NCHW'


class TestTemporalShift4(TestTemporalShift):
    def initTestCase(self):
        self.x_shape = (6, 5, 5, 4)
        self.seg_num = 3
        self.shift_ratio = 0.25
        self.data_format = 'NHWC'


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_float16_supported(core.CUDAPlace(0)),
    "core is not complied with CUDA and not support the float16",
)
class TestTemporalShiftFP16OP(OpTest):
    def initTestCase(self):
        self.x_shape = (3, 10, 5, 5)
        self.seg_num = 1
        self.shift_ratio = 0.3
        self.dtype = np.float16
        self.data_format = 'NCHW'

    def setUp(self):
        self.initTestCase()
        self.op_type = 'temporal_shift'
        self.python_api = paddle.nn.functional.temporal_shift
        self.__class__.op_type = self.op_type
        x = np.random.random(self.x_shape).astype(np.float32)

        self.attrs = {
            "seg_num": self.seg_num,
            "shift_ratio": self.shift_ratio,
            "data_format": self.data_format,
        }

        self.inputs = {
            "X": x.astype(self.dtype),
        }

        output = temporal_shift(
            x, self.seg_num, self.shift_ratio, self.data_format
        )
        self.outputs = {"Out": output}
        self.python_out_sig = ["Out"]

    def test_check_output(self):
        place = core.CUDAPlace(0)
        self.check_output_with_place(place, check_eager=True, atol=1e-3)

    def test_check_grad_ignore_uv(self):
        place = core.CUDAPlace(0)
        self.check_grad_with_place(
            place,
            ['X'],
            'Out',
            check_eager=True,
            max_relative_error=1e-2,
        )


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not complied with CUDA and not support the bfloat16",
)
class TestTemporalShiftBF16(OpTest):
    def initTestCase(self):
        self.x_shape = (3, 10, 5, 5)
        self.seg_num = 1
        self.shift_ratio = 0.3
        self.dtype = np.uint16
        self.data_format = 'NCHW'

    def setUp(self):
        self.initTestCase()
        self.op_type = 'temporal_shift'
        self.python_api = paddle.nn.functional.temporal_shift
        self.__class__.op_type = self.op_type
        x = np.random.random(self.x_shape).astype(np.float32)

        self.attrs = {
            "seg_num": self.seg_num,
            "shift_ratio": self.shift_ratio,
            "data_format": self.data_format,
        }

        self.inputs = {
            "X": convert_float_to_uint16(x),
        }

        output = temporal_shift(
            x, self.seg_num, self.shift_ratio, self.data_format
        )
        self.outputs = {"Out": convert_float_to_uint16(output)}
        self.python_out_sig = ["Out"]

    def test_check_output(self):
        place = core.CUDAPlace(0)
        self.check_output_with_place(place, check_eager=True, atol=1e-3)

    def test_check_grad_ignore_uv(self):
        place = core.CUDAPlace(0)
        self.check_grad_with_place(
            place,
            ['X'],
            'Out',
            check_eager=True,
            max_relative_error=1e-2,
        )


class TestTemporalShiftAPI(unittest.TestCase):
    def test_api(self):
        input = paddle.randn([6, 4, 2, 2])

        out_from_function = paddle.nn.functional.temporal_shift(
            x=input, seg_num=2, shift_ratio=0.2
        )

        # dygraph
        with paddle.fluid.dygraph.guard():
            input = paddle.randn([6, 4, 2, 2])
            out = paddle.nn.functional.temporal_shift(
                x=input, seg_num=2, shift_ratio=0.2
            )

    def test_static_fp16_gpu(self):
        if paddle.fluid.core.is_compiled_with_cuda():
            place = paddle.CUDAPlace(0)
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                input = np.random.random([4, 4, 112, 112]).astype("float16")

                x = paddle.static.data(
                    name="x", shape=[4, 4, 112, 112], dtype="float16"
                )

                y = paddle.nn.functional.temporal_shift(
                    x=x, seg_num=2, shift_ratio=0.2
                )

                exe = paddle.static.Executor(place)
                res = exe.run(
                    paddle.static.default_main_program(),
                    feed={
                        "x": input,
                    },
                    fetch_list=[y],
                )

    def test_error(self):
        def attr_data_format():
            input = paddle.randn([6, 4, 2, 2])
            out = paddle.nn.functional.temporal_shift(
                x=input, seg_num=2, shift_ratio=0.2, data_format="HWC"
            )

        self.assertRaises(ValueError, attr_data_format)


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
