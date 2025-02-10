#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from get_test_cover_info import (
    XPUOpTestWrapper,
    create_test_class,
    get_xpu_op_support_types,
)
from op_test import convert_float_to_uint16, convert_uint16_to_float
from op_test_xpu import XPUOpTest

import paddle
import paddle.nn.functional as F
from paddle import nn
from paddle.base import Executor, Program, default_main_program, program_guard

paddle.enable_static()


class XPUTestPad3dOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'pad3d'

    class TestPad3dOp(XPUOpTest):
        def setUp(self):
            paddle.enable_static()
            self.op_type = "pad3d"
            self.dtype = self.in_type
            self.place = paddle.XPUPlace(0)

            self.value = 0.0
            self.initTestCase()
            self.python_api = paddle.nn.functional.pad
            self.inputs = {'X': np.random.random(self.shape).astype(self.dtype)}
            self.attrs = {}
            if self.variable_paddings:
                self.attrs['paddings'] = []
                self.inputs['Paddings'] = (
                    np.array(self.paddings).flatten().astype("int32")
                )
            else:
                self.attrs['paddings'] = (
                    np.array(self.paddings).flatten().astype("int32")
                )
            self.attrs['value'] = self.value
            self.attrs['mode'] = self.mode
            self.attrs['data_format'] = self.data_format
            if self.data_format == "NCDHW":
                paddings = [
                    (0, 0),
                    (0, 0),
                    (self.paddings[4], self.paddings[5]),
                    (self.paddings[2], self.paddings[3]),
                    (self.paddings[0], self.paddings[1]),
                ]
            else:
                paddings = [
                    (0, 0),
                    (self.paddings[4], self.paddings[5]),
                    (self.paddings[2], self.paddings[3]),
                    (self.paddings[0], self.paddings[1]),
                    (0, 0),
                ]
            if self.mode == "constant":
                out = np.pad(
                    self.inputs['X'],
                    paddings,
                    mode=self.mode,
                    constant_values=self.value,
                )
            elif self.mode == "reflect":
                out = np.pad(self.inputs['X'], paddings, mode=self.mode)
            elif self.mode == "replicate":
                out = np.pad(self.inputs['X'], paddings, mode="edge")
            elif self.mode == "circular":
                out = np.pad(self.inputs['X'], paddings, mode="wrap")
            self.outputs = {'Out': out}

        def test_check_output(self):
            self.check_output(check_dygraph=True)

        def test_check_grad_normal(self):
            self.check_grad(['X'], 'Out', check_dygraph=True)

        def initTestCase(self):
            self.shape = (2, 3, 4, 5, 6)
            self.paddings = [0, 0, 0, 0, 0, 0]
            self.mode = "constant"
            self.data_format = "NCDHW"
            self.pad_value = 0.0
            self.variable_paddings = False

    class TestCase1(TestPad3dOp):
        def initTestCase(self):
            self.shape = (2, 3, 4, 5, 6)
            self.paddings = [0, 1, 2, 3, 4, 5]
            self.mode = "constant"
            self.data_format = "NCDHW"
            self.value = 1.0
            self.variable_paddings = False

    class TestCase2(TestPad3dOp):
        def initTestCase(self):
            self.shape = (2, 3, 4, 5, 6)
            self.paddings = [1, 1, 1, 1, 1, 1]
            self.mode = "constant"
            self.data_format = "NDHWC"
            self.value = 1.0
            self.variable_paddings = False

    class TestCase3(TestPad3dOp):
        def initTestCase(self):
            self.shape = (2, 3, 4, 5, 6)
            self.paddings = [0, 1, 1, 0, 2, 3]
            self.mode = "reflect"
            self.data_format = "NCDHW"
            self.variable_paddings = False

    class TestCase4(TestPad3dOp):
        def initTestCase(self):
            self.shape = (4, 4, 4, 4, 4)
            self.paddings = [0, 1, 2, 1, 2, 3]
            self.mode = "reflect"
            self.data_format = "NDHWC"
            self.variable_paddings = False

    class TestCase5(TestPad3dOp):
        def initTestCase(self):
            self.shape = (2, 3, 4, 5, 6)
            self.paddings = [0, 1, 2, 3, 2, 1]
            self.mode = "replicate"
            self.data_format = "NCDHW"
            self.variable_paddings = False

    class TestCase6(TestPad3dOp):
        def initTestCase(self):
            self.shape = (4, 4, 4, 4, 4)
            self.paddings = [5, 4, 2, 1, 2, 3]
            self.mode = "replicate"
            self.data_format = "NDHWC"
            self.variable_paddings = False

    class TestCase7(TestPad3dOp):
        def initTestCase(self):
            self.shape = (2, 3, 4, 5, 6)
            self.paddings = [0, 1, 2, 3, 4, 5]
            self.mode = "constant"
            self.data_format = "NCDHW"
            self.value = 1.0
            self.variable_paddings = True

    class TestCase8(TestPad3dOp):
        def initTestCase(self):
            self.shape = (2, 3, 4, 5, 6)
            self.paddings = [0, 1, 2, 3, 4, 5]
            self.mode = "constant"
            self.data_format = "NDHWC"
            self.value = 1.0
            self.variable_paddings = True

    class TestPadAPI(unittest.TestCase):
        def setUp(self):
            self.places = [paddle.XPUPlace(0)]
            self.dtype = self.in_type

        def check_static_result_1(self, place):
            paddle.enable_static()
            with program_guard(Program(), Program()):
                input_shape = (1, 2, 3, 4, 5)
                pad = [1, 2, 1, 1, 3, 4]
                mode = "constant"
                value = 100
                input_data = np.random.rand(*input_shape).astype(self.dtype)
                x = paddle.static.data(
                    name="x", shape=input_shape, dtype=self.dtype
                )
                result = F.pad(
                    x=x, pad=pad, value=value, mode=mode, data_format="NCDHW"
                )
                exe = Executor(place)
                fetches = exe.run(
                    default_main_program(),
                    feed={"x": input_data},
                    fetch_list=[result],
                )

                np_out = self._get_numpy_out(input_data, pad, mode, value)
                np.testing.assert_allclose(fetches[0], np_out, rtol=1e-05)

        def check_static_result_2(self, place):
            paddle.enable_static()
            with program_guard(Program(), Program()):
                input_shape = (2, 3, 4, 5, 6)
                pad = [1, 2, 1, 1, 1, 2]
                mode = "reflect"
                input_data = np.random.rand(*input_shape).astype(self.dtype)
                x = paddle.static.data(
                    name="x", shape=input_shape, dtype=self.dtype
                )
                result1 = F.pad(x=x, pad=pad, mode=mode, data_format="NCDHW")
                result2 = F.pad(x=x, pad=pad, mode=mode, data_format="NDHWC")
                exe = Executor(place)
                fetches = exe.run(
                    default_main_program(),
                    feed={"x": input_data},
                    fetch_list=[result1, result2],
                )

                np_out1 = self._get_numpy_out(
                    input_data, pad, mode, data_format="NCDHW"
                )
                np_out2 = self._get_numpy_out(
                    input_data, pad, mode, data_format="NDHWC"
                )
                np.testing.assert_allclose(fetches[0], np_out1, rtol=1e-05)
                np.testing.assert_allclose(fetches[1], np_out2, rtol=1e-05)

        def check_static_result_3(self, place):
            paddle.enable_static()
            with program_guard(Program(), Program()):
                input_shape = (2, 3, 4, 5, 6)
                pad = [1, 2, 1, 1, 3, 4]
                mode = "replicate"
                input_data = np.random.rand(*input_shape).astype(self.dtype)
                x = paddle.static.data(
                    name="x", shape=input_shape, dtype=self.dtype
                )
                result1 = F.pad(x=x, pad=pad, mode=mode, data_format="NCDHW")
                result2 = F.pad(x=x, pad=pad, mode=mode, data_format="NDHWC")
                exe = Executor(place)
                fetches = exe.run(
                    default_main_program(),
                    feed={"x": input_data},
                    fetch_list=[result1, result2],
                )

                np_out1 = self._get_numpy_out(
                    input_data, pad, mode, data_format="NCDHW"
                )
                np_out2 = self._get_numpy_out(
                    input_data, pad, mode, data_format="NDHWC"
                )
                np.testing.assert_allclose(fetches[0], np_out1, rtol=1e-05)
                np.testing.assert_allclose(fetches[1], np_out2, rtol=1e-05)

        def _get_numpy_out(
            self, input_data, pad, mode, value=0, data_format="NCDHW"
        ):
            if mode == "constant" and len(pad) == len(input_data.shape) * 2:
                pad = np.reshape(pad, (-1, 2)).tolist()
            elif data_format == "NCDHW":
                pad = [
                    (0, 0),
                    (0, 0),
                    (pad[4], pad[5]),
                    (pad[2], pad[3]),
                    (pad[0], pad[1]),
                ]
            elif data_format == "NDHWC":
                pad = [
                    (0, 0),
                    (pad[4], pad[5]),
                    (pad[2], pad[3]),
                    (pad[0], pad[1]),
                    (0, 0),
                ]
            elif data_format == "NCHW":
                pad = [
                    (0, 0),
                    (0, 0),
                    (pad[2], pad[3]),
                    (pad[0], pad[1]),
                ]
            elif data_format == "NHWC":
                pad = [
                    (0, 0),
                    (pad[2], pad[3]),
                    (pad[0], pad[1]),
                    (0, 0),
                ]
            elif data_format == "NCL":
                pad = [
                    (0, 0),
                    (0, 0),
                    (pad[0], pad[1]),
                ]
            elif data_format == "NLC":
                pad = [
                    (0, 0),
                    (pad[0], pad[1]),
                    (0, 0),
                ]

            if self.dtype == np.uint16:
                input_data = convert_uint16_to_float(input_data)

            if mode == "constant":
                out = np.pad(input_data, pad, mode=mode, constant_values=value)
            elif mode == "reflect":
                out = np.pad(input_data, pad, mode=mode)
            elif mode == "replicate":
                out = np.pad(input_data, pad, mode="edge")
            elif mode == "circular":
                out = np.pad(input_data, pad, mode="wrap")

            if self.dtype == np.uint16:
                out = convert_float_to_uint16(out)
            return out

        def test_static(self):
            for place in self.places:
                self.check_static_result_1(place=place)
                self.check_static_result_2(place=place)
                self.check_static_result_3(place=place)

        def test_dygraph_1(self):
            # TODO: remove fp16 limit after support of pad op
            if self.dtype == np.float16:
                return
            paddle.disable_static()
            input_shape = (1, 2, 3, 4, 5)
            pad = [1, 2, 1, 1, 3, 4]
            pad_3 = [1, 2, 1, 1, 3, 4, 5, 6, 7, 8]
            mode = "constant"
            value = 100
            input_data = np.random.rand(*input_shape).astype(self.dtype)
            np_out1 = self._get_numpy_out(
                input_data, pad, mode, value, data_format="NCDHW"
            )
            np_out2 = self._get_numpy_out(
                input_data, pad, mode, value, data_format="NDHWC"
            )
            np_out3 = self._get_numpy_out(
                input_data, pad_3, mode, value, data_format="NCDHW"
            )
            tensor_data = paddle.to_tensor(input_data)

            y1 = F.pad(
                tensor_data,
                pad=pad,
                mode=mode,
                value=value,
                data_format="NCDHW",
            )
            y2 = F.pad(
                tensor_data,
                pad=pad,
                mode=mode,
                value=value,
                data_format="NDHWC",
            )
            y3 = F.pad(
                tensor_data,
                pad=pad_3,
                mode=mode,
                value=value,
                data_format="NCDHW",
            )

            np.testing.assert_allclose(y1.numpy(), np_out1, rtol=1e-05)
            np.testing.assert_allclose(y2.numpy(), np_out2, rtol=1e-05)
            np.testing.assert_allclose(y3.numpy(), np_out3, rtol=1e-05)

        def test_dygraph_2(self):
            # TODO: remove fp16 limit after support of pad op
            if self.dtype == np.float16:
                return
            paddle.disable_static()
            input_shape = (2, 3, 4, 5)
            pad = [1, 1, 3, 4]
            pad_3 = [1, 2, 1, 1, 3, 4, 5, 6]
            mode = "constant"
            value = 100
            input_data = np.random.rand(*input_shape).astype(self.dtype)
            np_out1 = self._get_numpy_out(
                input_data, pad, mode, value, data_format="NCHW"
            )
            np_out2 = self._get_numpy_out(
                input_data, pad, mode, value, data_format="NHWC"
            )
            np_out3 = self._get_numpy_out(
                input_data, pad_3, mode, value, data_format="NCHW"
            )

            tensor_data = paddle.to_tensor(input_data)
            tensor_pad = paddle.to_tensor(pad, dtype="int32")

            y1 = F.pad(
                tensor_data,
                pad=tensor_pad,
                mode=mode,
                value=value,
                data_format="NCHW",
            )
            y2 = F.pad(
                tensor_data,
                pad=tensor_pad,
                mode=mode,
                value=value,
                data_format="NHWC",
            )
            y3 = F.pad(
                tensor_data,
                pad=pad_3,
                mode=mode,
                value=value,
                data_format="NCHW",
            )

            np.testing.assert_allclose(y1.numpy(), np_out1, rtol=1e-05)
            np.testing.assert_allclose(y2.numpy(), np_out2, rtol=1e-05)
            np.testing.assert_allclose(y3.numpy(), np_out3, rtol=1e-05)

        def test_dygraph_3(self):
            # TODO: remove fp16 limit after support of pad op
            if self.dtype == np.float16:
                return
            paddle.disable_static()
            input_shape = (3, 4, 5)
            pad = [3, 4]
            pad_3 = [3, 4, 5, 6, 7, 8]
            mode = "constant"
            value = 100
            input_data = np.random.rand(*input_shape).astype(self.dtype)
            np_out1 = self._get_numpy_out(
                input_data, pad, mode, value, data_format="NCL"
            )
            np_out2 = self._get_numpy_out(
                input_data, pad, mode, value, data_format="NLC"
            )
            np_out3 = self._get_numpy_out(
                input_data, pad_3, mode, value, data_format="NCL"
            )
            tensor_data = paddle.to_tensor(input_data)
            tensor_pad = paddle.to_tensor(pad, dtype="int32")

            y1 = F.pad(
                tensor_data,
                pad=tensor_pad,
                mode=mode,
                value=value,
                data_format="NCL",
            )
            y2 = F.pad(
                tensor_data,
                pad=tensor_pad,
                mode=mode,
                value=value,
                data_format="NLC",
            )
            y3 = F.pad(
                tensor_data,
                pad=pad_3,
                mode=mode,
                value=value,
                data_format="NCL",
            )

            np.testing.assert_allclose(y1.numpy(), np_out1, rtol=1e-05)
            np.testing.assert_allclose(y2.numpy(), np_out2, rtol=1e-05)
            np.testing.assert_allclose(y3.numpy(), np_out3, rtol=1e-05)

    class TestPad1dAPI(unittest.TestCase):
        def _get_numpy_out(
            self, input_data, pad, mode, value=0.0, data_format="NCL"
        ):
            if data_format == "NCL":
                pad = [
                    (0, 0),
                    (0, 0),
                    (pad[0], pad[1]),
                ]
            else:
                pad = [
                    (0, 0),
                    (pad[0], pad[1]),
                    (0, 0),
                ]

            if self.dtype == np.uint16:
                input_data = convert_uint16_to_float(input_data)

            if mode == "constant":
                out = np.pad(input_data, pad, mode=mode, constant_values=value)
            elif mode == "reflect":
                out = np.pad(input_data, pad, mode=mode)
            elif mode == "replicate":
                out = np.pad(input_data, pad, mode="edge")
            elif mode == "circular":
                out = np.pad(input_data, pad, mode="wrap")

            if self.dtype == np.uint16:
                out = convert_float_to_uint16(out)

            return out

        def setUp(self):
            self.places = [paddle.XPUPlace(0)]
            self.dtype = self.in_type

        def test_class(self):
            paddle.disable_static()
            for place in self.places:
                input_shape = (3, 4, 5)
                pad = [1, 2]
                pad_int = 1
                value = 100
                input_data = np.random.rand(*input_shape).astype(self.dtype)

                pad_reflection = nn.Pad1D(padding=pad, mode="reflect")
                pad_replication = nn.Pad1D(padding=pad, mode="replicate")
                pad_constant = nn.Pad1D(
                    padding=pad, mode="constant", value=value
                )
                pad_constant_int = nn.Pad1D(
                    padding=pad_int, mode="constant", value=value
                )
                pad_circular = nn.Pad1D(padding=pad, mode="circular")

                data = paddle.to_tensor(input_data)

                output = pad_reflection(data)
                np_out = self._get_numpy_out(
                    input_data, pad, "reflect", data_format="NCL"
                )
                np.testing.assert_allclose(output.numpy(), np_out, rtol=1e-05)

                output = pad_replication(data)
                np_out = self._get_numpy_out(
                    input_data, pad, "replicate", data_format="NCL"
                )
                np.testing.assert_allclose(output.numpy(), np_out, rtol=1e-05)

                output = pad_constant(data)
                np_out = self._get_numpy_out(
                    input_data, pad, "constant", value=value, data_format="NCL"
                )
                np.testing.assert_allclose(output.numpy(), np_out, rtol=1e-05)

                output = pad_constant_int(data)
                np_out = self._get_numpy_out(
                    input_data,
                    [pad_int] * 2,
                    "constant",
                    value=value,
                    data_format="NCL",
                )
                np.testing.assert_allclose(output.numpy(), np_out, rtol=1e-05)

    class TestPad2dAPI(unittest.TestCase):
        def _get_numpy_out(
            self, input_data, pad, mode, value=0.0, data_format="NCHW"
        ):
            if data_format == "NCHW":
                pad = [
                    (0, 0),
                    (0, 0),
                    (pad[2], pad[3]),
                    (pad[0], pad[1]),
                ]
            else:
                pad = [
                    (0, 0),
                    (pad[2], pad[3]),
                    (pad[0], pad[1]),
                    (0, 0),
                ]

            if self.dtype == np.uint16:
                input_data = convert_uint16_to_float(input_data)

            if mode == "constant":
                out = np.pad(input_data, pad, mode=mode, constant_values=value)
            elif mode == "reflect":
                out = np.pad(input_data, pad, mode=mode)
            elif mode == "replicate":
                out = np.pad(input_data, pad, mode="edge")
            elif mode == "circular":
                out = np.pad(input_data, pad, mode="wrap")

            if self.dtype == np.uint16:
                out = convert_float_to_uint16(out)

            return out

        def setUp(self):
            self.places = [paddle.XPUPlace(0)]
            self.dtype = self.in_type

        def test_class(self):
            paddle.disable_static()
            for place in self.places:
                input_shape = (3, 4, 5, 6)
                pad = [1, 2, 2, 1]
                pad_int = 1
                value = 100
                input_data = np.random.rand(*input_shape).astype(self.dtype)

                pad_reflection = nn.Pad2D(padding=pad, mode="reflect")
                pad_replication = nn.Pad2D(padding=pad, mode="replicate")
                pad_constant = nn.Pad2D(
                    padding=pad, mode="constant", value=value
                )
                pad_constant_int = nn.Pad2D(
                    padding=pad_int, mode="constant", value=value
                )
                pad_circular = nn.Pad2D(padding=pad, mode="circular")

                data = paddle.to_tensor(input_data)

                output = pad_reflection(data)
                np_out = self._get_numpy_out(
                    input_data, pad, "reflect", data_format="NCHW"
                )
                np.testing.assert_allclose(output.numpy(), np_out, rtol=1e-05)

                output = pad_replication(data)
                np_out = self._get_numpy_out(
                    input_data, pad, "replicate", data_format="NCHW"
                )
                np.testing.assert_allclose(output.numpy(), np_out, rtol=1e-05)

                output = pad_constant(data)
                np_out = self._get_numpy_out(
                    input_data, pad, "constant", value=value, data_format="NCHW"
                )
                np.testing.assert_allclose(output.numpy(), np_out, rtol=1e-05)

                output = pad_constant_int(data)
                np_out = self._get_numpy_out(
                    input_data,
                    [pad_int] * 4,
                    "constant",
                    value=value,
                    data_format="NCHW",
                )
                np.testing.assert_allclose(output.numpy(), np_out, rtol=1e-05)

    class TestPad3dAPI(unittest.TestCase):
        def _get_numpy_out(
            self, input_data, pad, mode, value=0.0, data_format="NCDHW"
        ):
            if data_format == "NCDHW":
                pad = [
                    (0, 0),
                    (0, 0),
                    (pad[4], pad[5]),
                    (pad[2], pad[3]),
                    (pad[0], pad[1]),
                ]
            else:
                pad = [
                    (0, 0),
                    (pad[4], pad[5]),
                    (pad[2], pad[3]),
                    (pad[0], pad[1]),
                    (0, 0),
                ]

            if self.dtype == np.uint16:
                input_data = convert_uint16_to_float(input_data)
            if mode == "constant":
                out = np.pad(input_data, pad, mode=mode, constant_values=value)
            elif mode == "reflect":
                out = np.pad(input_data, pad, mode=mode)
            elif mode == "replicate":
                out = np.pad(input_data, pad, mode="edge")
            elif mode == "circular":
                out = np.pad(input_data, pad, mode="wrap")

            if self.dtype == np.uint16:
                out = convert_float_to_uint16(out)

            return out

        def setUp(self):
            self.places = [paddle.XPUPlace(0)]
            self.dtype = self.in_type

        def test_class(self):
            paddle.disable_static()
            for place in self.places:
                input_shape = (3, 4, 5, 6, 7)
                pad = [1, 2, 2, 1, 1, 0]
                pad_int = 1
                value = 100
                input_data = np.random.rand(*input_shape).astype(self.dtype)

                pad_reflection = nn.Pad3D(padding=pad, mode="reflect")
                pad_replication = nn.Pad3D(padding=pad, mode="replicate")
                pad_constant = nn.Pad3D(
                    padding=pad, mode="constant", value=value
                )
                pad_constant_int = nn.Pad3D(
                    padding=pad_int, mode="constant", value=value
                )
                pad_circular = nn.Pad3D(padding=pad, mode="circular")

                data = paddle.to_tensor(input_data)

                output = pad_reflection(data)
                np_out = self._get_numpy_out(
                    input_data, pad, "reflect", data_format="NCDHW"
                )
                np.testing.assert_allclose(output.numpy(), np_out, rtol=1e-05)

                output = pad_replication(data)
                np_out = self._get_numpy_out(
                    input_data, pad, "replicate", data_format="NCDHW"
                )
                np.testing.assert_allclose(output.numpy(), np_out, rtol=1e-05)

                output = pad_constant(data)
                np_out = self._get_numpy_out(
                    input_data,
                    pad,
                    "constant",
                    value=value,
                    data_format="NCDHW",
                )
                np.testing.assert_allclose(output.numpy(), np_out, rtol=1e-05)

                output = pad_constant_int(data)
                np_out = self._get_numpy_out(
                    input_data,
                    [pad_int] * 6,
                    "constant",
                    value=value,
                    data_format="NCDHW",
                )
                np.testing.assert_allclose(output.numpy(), np_out, rtol=1e-05)

        def test_pad_tensor(self):
            paddle.disable_static()
            for place in self.places:
                input_shape = (3, 4, 5, 6, 7)
                pad = [1, 2, 2, 1, 1, 0]
                pad_tensor = paddle.to_tensor(pad)
                input_data = np.random.rand(*input_shape).astype(self.dtype)

                pad_reflection_ncdhw = nn.Pad3D(
                    padding=pad_tensor, mode="reflect", data_format="NCDHW"
                )
                pad_reflection_ndhwc = nn.Pad3D(
                    padding=pad_tensor, mode="reflect", data_format="NDHWC"
                )
                data = paddle.to_tensor(input_data)

                output = pad_reflection_ncdhw(data)
                np_out = self._get_numpy_out(
                    input_data, pad, "reflect", data_format="NCDHW"
                )
                np.testing.assert_allclose(output.numpy(), np_out, rtol=1e-05)

                output = pad_reflection_ndhwc(data)
                np_out = self._get_numpy_out(
                    input_data, pad, "reflect", data_format="NDHWC"
                )
                np.testing.assert_allclose(output.numpy(), np_out, rtol=1e-05)

    class TestPad3dOpError(unittest.TestCase):
        def setUp(self):
            self.places = [paddle.XPUPlace(0)]
            self.dtype = self.in_type

        def test_errors(self):
            def test_variable():
                input_shape = (1, 2, 3, 4, 5)
                data = np.random.rand(*input_shape).astype(self.dtype)
                y = F.pad(x=data, pad=[1, 1, 1, 1, 1, 1], data_format="NCDHW")

            def test_reflect_1():
                input_shape = (1, 2, 3, 4, 5)
                data = np.random.rand(*input_shape).astype(self.dtype)
                x = paddle.to_tensor(data)
                y = F.pad(
                    x,
                    pad=[5, 6, 1, 1, 1, 1],
                    value=1,
                    mode='reflect',
                    data_format="NCDHW",
                )

            def test_reflect_2():
                input_shape = (1, 2, 3, 4, 5)
                data = np.random.rand(*input_shape).astype(self.dtype)
                x = paddle.to_tensor(data)
                y = F.pad(
                    x,
                    pad=[1, 1, 4, 3, 1, 1],
                    value=1,
                    mode='reflect',
                    data_format="NCDHW",
                )

            def test_reflect_3():
                input_shape = (1, 2, 3, 4, 5)
                data = np.random.rand(*input_shape).astype(self.dtype)
                x = paddle.to_tensor(data)
                y = F.pad(
                    x,
                    pad=[1, 1, 1, 1, 2, 3],
                    value=1,
                    mode='reflect',
                    data_format="NCDHW",
                )

            def test_replicate_1():
                input_shape = (1, 2, 0, 4, 5)
                data = np.random.rand(*input_shape).astype(self.dtype)
                x = paddle.to_tensor(data)
                y = F.pad(
                    x,
                    pad=[1, 1, 1, 1, 2, 3],
                    mode='replicate',
                    data_format="NCDHW",
                )

            paddle.disable_static()
            for place in self.places:
                self.assertRaises(ValueError, test_variable)
                self.assertRaises(Exception, test_reflect_1)
                self.assertRaises(Exception, test_reflect_2)
                self.assertRaises(Exception, test_reflect_3)
                self.assertRaises(Exception, test_replicate_1)
            paddle.enable_static()

    class TestPadDataformatError(unittest.TestCase):
        def test_errors(self):
            def test_ncl():
                input_shape = (1, 2, 3, 4)
                pad = paddle.to_tensor(np.array([2, 1, 2, 1]).astype('int32'))
                data = (
                    np.arange(np.prod(input_shape), dtype=np.float64).reshape(
                        input_shape
                    )
                    + 1
                )
                my_pad = nn.Pad1D(
                    padding=pad, mode="replicate", data_format="NCL"
                )
                data = paddle.to_tensor(data)
                result = my_pad(data)

            def test_nchw():
                input_shape = (1, 2, 4)
                pad = paddle.to_tensor(np.array([2, 1, 2, 1]).astype('int32'))
                data = (
                    np.arange(np.prod(input_shape), dtype=np.float64).reshape(
                        input_shape
                    )
                    + 1
                )
                my_pad = nn.Pad1D(
                    padding=pad, mode="replicate", data_format="NCHW"
                )
                data = paddle.to_tensor(data)
                result = my_pad(data)

            def test_ncdhw():
                input_shape = (1, 2, 3, 4)
                pad = paddle.to_tensor(np.array([2, 1, 2, 1]).astype('int32'))
                data = (
                    np.arange(np.prod(input_shape), dtype=np.float64).reshape(
                        input_shape
                    )
                    + 1
                )
                my_pad = nn.Pad1D(
                    padding=pad, mode="replicate", data_format="NCDHW"
                )
                data = paddle.to_tensor(data)
                result = my_pad(data)

            self.assertRaises(AssertionError, test_ncl)
            self.assertRaises(AssertionError, test_nchw)
            self.assertRaises(AssertionError, test_ncdhw)


support_types = get_xpu_op_support_types('pad3d')
for stype in support_types:
    create_test_class(globals(), XPUTestPad3dOp, stype)


if __name__ == '__main__':
    unittest.main()
