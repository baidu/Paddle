# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
from paddle import base

paddle.enable_static()


def strided_slice_native_forward(input, axes, starts, ends, strides):
    dim = input.ndim
    start = []
    end = []
    stride = []
    for i in range(dim):
        start.append(0)
        end.append(input.shape[i])
        stride.append(1)

    for i in range(len(axes)):
        start[axes[i]] = starts[i]
        end[axes[i]] = ends[i]
        stride[axes[i]] = strides[i]

    result = {
        1: lambda input, start, end, stride: input[
            start[0] : end[0] : stride[0]
        ],
        2: lambda input, start, end, stride: input[
            start[0] : end[0] : stride[0], start[1] : end[1] : stride[1]
        ],
        3: lambda input, start, end, stride: input[
            start[0] : end[0] : stride[0],
            start[1] : end[1] : stride[1],
            start[2] : end[2] : stride[2],
        ],
        4: lambda input, start, end, stride: input[
            start[0] : end[0] : stride[0],
            start[1] : end[1] : stride[1],
            start[2] : end[2] : stride[2],
            start[3] : end[3] : stride[3],
        ],
        5: lambda input, start, end, stride: input[
            start[0] : end[0] : stride[0],
            start[1] : end[1] : stride[1],
            start[2] : end[2] : stride[2],
            start[3] : end[3] : stride[3],
            start[4] : end[4] : stride[4],
        ],
        6: lambda input, start, end, stride: input[
            start[0] : end[0] : stride[0],
            start[1] : end[1] : stride[1],
            start[2] : end[2] : stride[2],
            start[3] : end[3] : stride[3],
            start[4] : end[4] : stride[4],
            start[5] : end[5] : stride[5],
        ],
    }[dim](input, start, end, stride)

    return result


class TestStrideSliceOp(OpTest):
    def setUp(self):
        self.initTestCase()
        self.op_type = 'strided_slice'
        self.python_api = paddle.strided_slice
        self.output = strided_slice_native_forward(
            self.input, self.axes, self.starts, self.ends, self.strides
        )

        self.inputs = {'Input': self.input}
        self.outputs = {'Out': self.output}
        self.attrs = {
            'axes': self.axes,
            'starts': self.starts,
            'ends': self.ends,
            'strides': self.strides,
            'infer_flags': self.infer_flags,
        }

    def test_check_output(self):
        self.check_output(check_cinn=True, check_pir=True)

    def test_check_grad(self):
        self.check_grad({'Input'}, 'Out', check_cinn=True, check_pir=True)

    def initTestCase(self):
        self.input = np.random.rand(100)
        self.axes = [0]
        self.starts = [-4]
        self.ends = [-3]
        self.strides = [1]
        self.infer_flags = [1]


class TestStrideSliceOp1(TestStrideSliceOp):
    def initTestCase(self):
        self.input = np.random.rand(100)
        self.axes = [0]
        self.starts = [3]
        self.ends = [8]
        self.strides = [1]
        self.infer_flags = [1]


class TestStrideSliceOp2(TestStrideSliceOp):
    def initTestCase(self):
        self.input = np.random.rand(100)
        self.axes = [0]
        self.starts = [5]
        self.ends = [0]
        self.strides = [-1]
        self.infer_flags = [1]


class TestStrideSliceOp3(TestStrideSliceOp):
    def initTestCase(self):
        self.input = np.random.rand(100)
        self.axes = [0]
        self.starts = [-1]
        self.ends = [-3]
        self.strides = [-1]
        self.infer_flags = [1]


class TestStrideSliceOp4(TestStrideSliceOp):
    def initTestCase(self):
        self.input = np.random.rand(3, 4, 10)
        self.axes = [0, 1, 2]
        self.starts = [0, -1, 0]
        self.ends = [2, -3, 5]
        self.strides = [1, -1, 1]
        self.infer_flags = [1, 1, 1]


class TestStrideSliceOp5(TestStrideSliceOp):
    def initTestCase(self):
        self.input = np.random.rand(5, 5, 5)
        self.axes = [0, 1, 2]
        self.starts = [1, 0, 0]
        self.ends = [2, 1, 3]
        self.strides = [1, 1, 1]
        self.infer_flags = [1, 1, 1]


class TestStrideSliceOp6(TestStrideSliceOp):
    def initTestCase(self):
        self.input = np.random.rand(5, 5, 5)
        self.axes = [0, 1, 2]
        self.starts = [1, -1, 0]
        self.ends = [2, -3, 3]
        self.strides = [1, -1, 1]
        self.infer_flags = [1, 1, 1]


class TestStrideSliceOp7(TestStrideSliceOp):
    def initTestCase(self):
        self.input = np.random.rand(5, 5, 5)
        self.axes = [0, 1, 2]
        self.starts = [1, 0, 0]
        self.ends = [2, 2, 3]
        self.strides = [1, 1, 1]
        self.infer_flags = [1, 1, 1]


class TestStrideSliceOp8(TestStrideSliceOp):
    def initTestCase(self):
        self.input = np.random.rand(1, 100, 1)
        self.axes = [1]
        self.starts = [1]
        self.ends = [2]
        self.strides = [1]
        self.infer_flags = [1]


class TestStrideSliceOp9(TestStrideSliceOp):
    def initTestCase(self):
        self.input = np.random.rand(1, 100, 1)
        self.axes = [1]
        self.starts = [-1]
        self.ends = [-2]
        self.strides = [-1]
        self.infer_flags = [1]


class TestStrideSliceOp10(TestStrideSliceOp):
    def initTestCase(self):
        self.input = np.random.rand(10, 10)
        self.axes = [0, 1]
        self.starts = [1, 0]
        self.ends = [2, 2]
        self.strides = [1, 1]
        self.infer_flags = [1, 1]


class TestStrideSliceOp11(TestStrideSliceOp):
    def initTestCase(self):
        self.input = np.random.rand(3, 3, 3, 4)
        self.axes = [0, 1, 2, 3]
        self.starts = [1, 0, 0, 0]
        self.ends = [2, 2, 3, 4]
        self.strides = [1, 1, 1, 2]
        self.infer_flags = [1, 1, 1, 1]


class TestStrideSliceOp12(TestStrideSliceOp):
    def initTestCase(self):
        self.input = np.random.rand(3, 3, 3, 4, 5)
        self.axes = [0, 1, 2, 3, 4]
        self.starts = [1, 0, 0, 0, 0]
        self.ends = [2, 2, 3, 4, 4]
        self.strides = [1, 1, 1, 1, 1]
        self.infer_flags = [1, 1, 1, 1]


class TestStrideSliceOp13(TestStrideSliceOp):
    def initTestCase(self):
        self.input = np.random.rand(3, 3, 3, 6, 7, 8)
        self.axes = [0, 1, 2, 3, 4, 5]
        self.starts = [1, 0, 0, 0, 1, 2]
        self.ends = [2, 2, 3, 1, 2, 8]
        self.strides = [1, 1, 1, 1, 1, 2]
        self.infer_flags = [1, 1, 1, 1, 1]


class TestStrideSliceOp14(TestStrideSliceOp):
    def initTestCase(self):
        self.input = np.random.rand(4, 4, 4, 4)
        self.axes = [1, 2, 3]
        self.starts = [-5, 0, -7]
        self.ends = [-1, 2, 4]
        self.strides = [1, 1, 1]
        self.infer_flags = [1, 1, 1]


# Non divisible test case
class TestStrideSliceOp15(TestStrideSliceOp):
    def initTestCase(self):
        self.input = np.random.rand(4, 320)
        self.axes = [1]
        self.starts = [0]
        self.ends = [320]
        self.strides = [4]
        self.infer_flags = [1]


class TestStrideSliceOp16(TestStrideSliceOp):
    def initTestCase(self):
        self.input = np.random.rand(4, 320)
        self.axes = [1]
        self.starts = [1]
        self.ends = [320]
        self.strides = [4]
        self.infer_flags = [1]


class TestStrideSliceOp17(TestStrideSliceOp):
    def initTestCase(self):
        self.input = np.random.rand(4, 320)
        self.axes = [1]
        self.starts = [2]
        self.ends = [320]
        self.strides = [4]
        self.infer_flags = [1]


class TestStrideSliceOp18(TestStrideSliceOp):
    def initTestCase(self):
        self.input = np.random.rand(4, 320)
        self.axes = [1]
        self.starts = [3]
        self.ends = [320]
        self.strides = [4]
        self.infer_flags = [1]


class TestStrideSliceOp19(TestStrideSliceOp):
    def initTestCase(self):
        self.input = np.random.rand(4, 320)
        self.axes = [1]
        self.starts = [4]
        self.ends = [320]
        self.strides = [4]
        self.infer_flags = [1]


# 0-size test case
class TestStrideSliceOp20(TestStrideSliceOp):
    def initTestCase(self):
        self.input = np.random.rand(4, 320)
        self.axes = [1]
        self.starts = [4]
        self.ends = [8]
        self.strides = [4]
        self.infer_flags = [1]


class TestStrideSliceOp21(TestStrideSliceOp):
    def initTestCase(self):
        self.input = np.random.rand(4, 100)
        self.axes = [1]
        self.starts = [-101]
        self.ends = [-101]
        self.strides = [1]
        self.infer_flags = [1]

    def test_check_grad(self):
        pass


class TestStrideSliceOpBool(TestStrideSliceOp):
    def test_check_grad(self):
        pass


class TestStrideSliceOpBool1D(TestStrideSliceOpBool):
    def initTestCase(self):
        self.input = np.random.rand(100).astype("bool")
        self.axes = [0]
        self.starts = [3]
        self.ends = [8]
        self.strides = [1]
        self.infer_flags = [1]


class TestStrideSliceOpBool2D(TestStrideSliceOpBool):
    def initTestCase(self):
        self.input = np.random.rand(10, 10).astype("bool")
        self.axes = [0, 1]
        self.starts = [1, 0]
        self.ends = [2, 2]
        self.strides = [1, 1]
        self.infer_flags = [1, 1]


class TestStrideSliceOpBool3D(TestStrideSliceOpBool):
    def initTestCase(self):
        self.input = np.random.rand(3, 4, 10).astype("bool")
        self.axes = [0, 1, 2]
        self.starts = [0, -1, 0]
        self.ends = [2, -3, 5]
        self.strides = [1, -1, 1]
        self.infer_flags = [1, 1, 1]


class TestStrideSliceOpBool4D(TestStrideSliceOpBool):
    def initTestCase(self):
        self.input = np.random.rand(3, 3, 3, 4).astype("bool")
        self.axes = [0, 1, 2, 3]
        self.starts = [1, 0, 0, 0]
        self.ends = [2, 2, 3, 4]
        self.strides = [1, 1, 1, 2]
        self.infer_flags = [1, 1, 1, 1]


class TestStrideSliceOpBool5D(TestStrideSliceOpBool):
    def initTestCase(self):
        self.input = np.random.rand(3, 3, 3, 4, 5).astype("bool")
        self.axes = [0, 1, 2, 3, 4]
        self.starts = [1, 0, 0, 0, 0]
        self.ends = [2, 2, 3, 4, 4]
        self.strides = [1, 1, 1, 1, 1]
        self.infer_flags = [1, 1, 1, 1]


class TestStrideSliceOpBool6D(TestStrideSliceOpBool):
    def initTestCase(self):
        self.input = np.random.rand(3, 3, 3, 6, 7, 8).astype("bool")
        self.axes = [0, 1, 2, 3, 4, 5]
        self.starts = [1, 0, 0, 0, 1, 2]
        self.ends = [2, 2, 3, 1, 2, 8]
        self.strides = [1, 1, 1, 1, 1, 2]
        self.infer_flags = [1, 1, 1, 1, 1]


class TestStridedSliceOp_starts_ListTensor(OpTest):
    def setUp(self):
        self.op_type = "strided_slice"
        self.python_api = paddle.strided_slice
        self.config()

        starts_tensor = []
        for index, ele in enumerate(self.starts):
            starts_tensor.append(
                ("x" + str(index), np.ones(1).astype('int32') * ele)
            )
        self.inputs = {'Input': self.input, 'StartsTensorList': starts_tensor}
        self.outputs = {'Out': self.output}
        self.attrs = {
            'axes': self.axes,
            'starts': self.starts_infer,
            'ends': self.ends,
            'strides': self.strides,
            'infer_flags': self.infer_flags,
        }

    def config(self):
        self.input = np.random.random([3, 4, 5, 6]).astype("float64")
        self.starts = [1, 0, 2]
        self.ends = [3, 3, 4]
        self.axes = [0, 1, 2]
        self.strides = [1, 1, 1]
        self.infer_flags = [1, -1, 1]
        self.output = strided_slice_native_forward(
            self.input, self.axes, self.starts, self.ends, self.strides
        )

        self.starts_infer = [1, 10, 2]

    def test_check_output(self):
        self.check_output(check_pir=True, check_symbol_infer=False)

    def test_check_grad_normal(self):
        self.check_grad(
            ['Input'], 'Out', max_relative_error=0.006, check_pir=True
        )


class TestStridedSliceOp_ends_ListTensor(OpTest):
    def setUp(self):
        self.op_type = "strided_slice"
        self.python_api = paddle.strided_slice
        self.config()

        ends_tensor = []
        for index, ele in enumerate(self.ends):
            ends_tensor.append(
                ("x" + str(index), np.ones(1).astype('int32') * ele)
            )

        self.inputs = {'Input': self.input, 'EndsTensorList': ends_tensor}
        self.outputs = {'Out': self.output}
        self.attrs = {
            'axes': self.axes,
            'starts': self.starts,
            'ends': self.ends_infer,
            'strides': self.strides,
            'infer_flags': self.infer_flags,
        }

    def config(self):
        self.input = np.random.random([3, 4, 5, 6]).astype("float64")
        self.starts = [1, 0, 0]
        self.ends = [3, 3, 4]
        self.axes = [0, 1, 2]
        self.strides = [1, 1, 2]
        self.infer_flags = [1, -1, 1]
        self.output = strided_slice_native_forward(
            self.input, self.axes, self.starts, self.ends, self.strides
        )

        self.ends_infer = [3, 1, 4]

    def test_check_output(self):
        self.check_output(check_pir=True, check_symbol_infer=False)

    def test_check_grad_normal(self):
        self.check_grad(
            ['Input'], 'Out', max_relative_error=0.006, check_pir=True
        )


class TestStridedSliceOp_starts_Tensor(OpTest):
    def setUp(self):
        self.op_type = "strided_slice"
        self.python_api = paddle.strided_slice
        self.config()
        self.inputs = {
            'Input': self.input,
            "StartsTensor": np.array(self.starts, dtype="int32"),
        }
        self.outputs = {'Out': self.output}
        self.attrs = {
            'axes': self.axes,
            # 'starts': self.starts,
            'ends': self.ends,
            'strides': self.strides,
            'infer_flags': self.infer_flags,
        }

    def config(self):
        self.input = np.random.random([3, 4, 5, 6]).astype("float64")
        self.starts = [1, 0, 2]
        self.ends = [2, 3, 4]
        self.axes = [0, 1, 2]
        self.strides = [1, 1, 1]
        self.infer_flags = [-1, -1, -1]
        self.output = strided_slice_native_forward(
            self.input, self.axes, self.starts, self.ends, self.strides
        )

    def test_check_output(self):
        self.check_output(check_pir=True, check_symbol_infer=False)

    def test_check_grad_normal(self):
        self.check_grad(
            ['Input'], 'Out', max_relative_error=0.006, check_pir=True
        )


class TestStridedSliceOp_ends_Tensor(OpTest):
    def setUp(self):
        self.op_type = "strided_slice"
        self.python_api = paddle.strided_slice
        self.config()
        self.inputs = {
            'Input': self.input,
            "EndsTensor": np.array(self.ends, dtype="int32"),
        }
        self.outputs = {'Out': self.output}
        self.attrs = {
            'axes': self.axes,
            'starts': self.starts,
            # 'ends': self.ends,
            'strides': self.strides,
            'infer_flags': self.infer_flags,
        }

    def config(self):
        self.input = np.random.random([3, 4, 5, 6]).astype("float64")
        self.starts = [1, 0, 2]
        self.ends = [2, 3, 4]
        self.axes = [0, 1, 2]
        self.strides = [1, 1, 1]
        self.infer_flags = [-1, -1, -1]
        self.output = strided_slice_native_forward(
            self.input, self.axes, self.starts, self.ends, self.strides
        )

    def test_check_output(self):
        self.check_output(check_pir=True, check_symbol_infer=False)

    def test_check_grad_normal(self):
        self.check_grad(
            ['Input'], 'Out', max_relative_error=0.006, check_pir=True
        )


class TestStridedSliceOp_listTensor_Tensor(OpTest):
    def setUp(self):
        self.config()
        ends_tensor = []
        for index, ele in enumerate(self.ends):
            ends_tensor.append(
                ("x" + str(index), np.ones(1).astype('int32') * ele)
            )
        self.op_type = "strided_slice"
        self.python_api = paddle.strided_slice

        self.inputs = {
            'Input': self.input,
            "StartsTensor": np.array(self.starts, dtype="int32"),
            "EndsTensorList": ends_tensor,
        }
        self.outputs = {'Out': self.output}
        self.attrs = {
            'axes': self.axes,
            # 'starts': self.starts,
            # 'ends': self.ends,
            'strides': self.strides,
            'infer_flags': self.infer_flags,
        }

    def config(self):
        self.input = np.random.random([3, 4, 5, 6]).astype("float64")
        self.starts = [1, 0, 2]
        self.ends = [2, 3, 4]
        self.axes = [0, 1, 2]
        self.strides = [1, 1, 1]
        self.infer_flags = [-1, -1, -1]
        self.output = strided_slice_native_forward(
            self.input, self.axes, self.starts, self.ends, self.strides
        )

    def test_check_output(self):
        self.check_output(check_pir=True, check_symbol_infer=False)

    def test_check_grad_normal(self):
        self.check_grad(
            ['Input'], 'Out', max_relative_error=0.006, check_pir=True
        )


class TestStridedSliceOp_strides_Tensor(OpTest):
    def setUp(self):
        self.op_type = "strided_slice"
        self.python_api = paddle.strided_slice
        self.config()
        self.inputs = {
            'Input': self.input,
            "StridesTensor": np.array(self.strides, dtype="int32"),
        }
        self.outputs = {'Out': self.output}
        self.attrs = {
            'axes': self.axes,
            'starts': self.starts,
            'ends': self.ends,
            # 'strides': self.strides,
            'infer_flags': self.infer_flags,
        }

    def config(self):
        self.input = np.random.random([3, 4, 5, 6]).astype("float64")
        self.starts = [1, -1, 2]
        self.ends = [2, 0, 4]
        self.axes = [0, 1, 2]
        self.strides = [1, -1, 1]
        self.infer_flags = [-1, -1, -1]
        self.output = strided_slice_native_forward(
            self.input, self.axes, self.starts, self.ends, self.strides
        )

    def test_check_output(self):
        self.check_output(check_pir=True, check_symbol_infer=False)

    def test_check_grad_normal(self):
        self.check_grad(
            ['Input'], 'Out', max_relative_error=0.006, check_pir=True
        )


class TestStrideSliceFP16Op(OpTest):
    def setUp(self):
        self.initTestCase()
        self.op_type = 'strided_slice'
        self.dtype = np.float16
        self.python_api = paddle.strided_slice
        self.output = strided_slice_native_forward(
            self.input, self.axes, self.starts, self.ends, self.strides
        )

        self.inputs = {'Input': self.input.astype(self.dtype)}
        self.outputs = {'Out': self.output}
        self.attrs = {
            'axes': self.axes,
            'starts': self.starts,
            'ends': self.ends,
            'strides': self.strides,
            'infer_flags': self.infer_flags,
        }

    def test_check_output(self):
        self.check_output(check_cinn=True, check_pir=True)

    def test_check_grad(self):
        self.check_grad({'Input'}, 'Out', check_cinn=True, check_pir=True)

    def initTestCase(self):
        self.input = np.random.rand(100)
        self.axes = [0]
        self.starts = [-4]
        self.ends = [-3]
        self.strides = [1]
        self.infer_flags = [1]


class TestStrideSliceBF16Op(OpTest):
    def setUp(self):
        self.initTestCase()
        self.op_type = 'strided_slice'
        self.dtype = np.uint16
        self.python_api = paddle.strided_slice
        self.output = strided_slice_native_forward(
            self.input, self.axes, self.starts, self.ends, self.strides
        )

        self.inputs = {
            'Input': convert_float_to_uint16(self.input.astype(np.float32))
        }
        self.outputs = {'Out': convert_float_to_uint16(self.output)}
        self.attrs = {
            'axes': self.axes,
            'starts': self.starts,
            'ends': self.ends,
            'strides': self.strides,
            'infer_flags': self.infer_flags,
        }

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad(self):
        self.check_grad({'Input'}, 'Out', check_pir=True)

    def initTestCase(self):
        self.input = np.random.rand(100)
        self.axes = [0]
        self.starts = [-4]
        self.ends = [-3]
        self.strides = [1]
        self.infer_flags = [1]


# Test python API
# class TestStridedSliceAPI(unittest.TestCase):

#     def test_static_api(self):
#         paddle.enable_static()
#         place = base.CPUPlace()
#         input = np.random.random([3, 4, 5, 6]).astype("float64")
#         with paddle.static.program_guard(paddle.static.Program()):
#             minus_1 = paddle.tensor.fill_constant([], "int32", -1)
#             minus_3 = paddle.tensor.fill_constant([], "int32", -3)
#             starts = paddle.static.data(name='starts', shape=[3], dtype='int32')
#             ends = paddle.static.data(name='ends', shape=[3], dtype='int32')
#             strides = paddle.static.data(
#                 name='strides', shape=[3], dtype='int32'
#             )

#             x = paddle.static.data(
#                 name="x",
#                 shape=[3, 4, 5, 6],
#                 dtype="float64",
#             )
#             out_1 = paddle.strided_slice(
#                 x,
#                 axes=[0, 1, 2],
#                 starts=[-3, 0, 2],
#                 ends=[3, 100, -1],
#                 strides=[1, 1, 1],
#             )
#             out_2 = paddle.strided_slice(
#                 x,
#                 axes=[0, 1, 3],
#                 starts=[minus_3, 0, 2],
#                 ends=[3, 100, -1],
#                 strides=[1, 1, 1],
#             )
#             out_3 = paddle.strided_slice(
#                 x,
#                 axes=[0, 1, 3],
#                 starts=[minus_3, 0, 2],
#                 ends=[3, 100, minus_1],
#                 strides=[1, 1, 1],
#             )
#             out_4 = paddle.strided_slice(
#                 x, axes=[0, 1, 2], starts=starts, ends=ends, strides=strides
#             )

#             out_5 = x[-3:3, 0:100:2, -1:2:-1]
#             out_6 = x[minus_3:3:1, 0:100:2, :, minus_1:2:minus_1]
#             out_7 = x[minus_1, 0:100:2, :, -1:2:-1]

#             exe = paddle.static.Executor(place)
#             res_1, res_2, res_3, res_4, res_5, res_6, res_7 = exe.run(
#                 paddle.static.default_main_program(),
#                 feed={
#                     "x": input,
#                     'starts': np.array([-3, 0, 2]).astype("int32"),
#                     'ends': np.array([3, 2147483647, -1]).astype("int32"),
#                     'strides': np.array([1, 1, 1]).astype("int32"),
#                 },
#                 fetch_list=[out_1, out_2, out_3, out_4, out_5, out_6, out_7],
#             )
#             np.testing.assert_array_equal(res_1, input[-3:3, 0:100, 2:-1, :])
#             np.testing.assert_array_equal(res_2, input[-3:3, 0:100, :, 2:-1])
#             np.testing.assert_array_equal(res_3, input[-3:3, 0:100, :, 2:-1])
#             np.testing.assert_array_equal(res_4, input[-3:3, 0:100, 2:-1, :])
#             np.testing.assert_array_equal(
#                 res_5, input[-3:3, 0:100:2, -1:2:-1, :]
#             )
#             np.testing.assert_array_equal(
#                 res_6, input[-3:3, 0:100:2, :, -1:2:-1]
#             )
#             np.testing.assert_array_equal(res_7, input[-1, 0:100:2, :, -1:2:-1])

#     def test_dygraph_op(self):
#         x = paddle.zeros(shape=[3, 4, 5, 6], dtype="float32")
#         axes = [1, 2, 3]
#         starts = [-3, 0, 2]
#         ends = [3, 2, 4]
#         strides_1 = [1, 1, 1]
#         sliced_1 = paddle.strided_slice(
#             x, axes=axes, starts=starts, ends=ends, strides=strides_1
#         )
#         assert sliced_1.shape == [3, 2, 2, 2]

#     @unittest.skipIf(
#         not paddle.is_compiled_with_cuda(),
#         "Cannot use CUDAPinnedPlace in CPU only version",
#     )
#     def test_cuda_pinned_place(self):
#         with paddle.base.dygraph.guard():
#             x = paddle.to_tensor(
#                 np.random.randn(2, 10), place=paddle.CUDAPinnedPlace()
#             )
#             self.assertTrue(x.place.is_cuda_pinned_place())
#             y = x[:, ::2]
#             self.assertFalse(x.place.is_cuda_pinned_place())
#             self.assertFalse(y.place.is_cuda_pinned_place())


class ArrayLayer(paddle.nn.Layer):
    def __init__(self, input_size=224, output_size=10, array_size=1):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.array_size = array_size
        for i in range(self.array_size):
            setattr(
                self,
                self.create_name(i),
                paddle.nn.Linear(input_size, output_size),
            )

    def create_name(self, index):
        return 'linear_' + str(index)

    def forward(self, inps):
        array = []
        for i in range(self.array_size):
            linear = getattr(self, self.create_name(i))
            array.append(linear(inps))

        tensor_array = self.create_tensor_array(array)

        tensor_array = self.array_slice(tensor_array)

        array1 = paddle.concat(tensor_array)
        array2 = paddle.concat(tensor_array[::-1])
        return array1 + array2 * array2

    def get_all_grads(self, param_name='weight'):
        grads = []
        for i in range(self.array_size):
            linear = getattr(self, self.create_name(i))
            param = getattr(linear, param_name)

            g = param.grad
            if g is not None:
                g = g.numpy()

            grads.append(g)

        return grads

    def clear_all_grad(self):
        param_names = ['weight', 'bias']
        for i in range(self.array_size):
            linear = getattr(self, self.create_name(i))
            for p in param_names:
                param = getattr(linear, p)
                param.clear_gradient()

    def array_slice(self, array):
        return array

    def create_tensor_array(self, tensors):
        tensor_array = None
        for i, tensor in enumerate(tensors):
            index = paddle.full(shape=[1], dtype='int64', fill_value=i)
            if tensor_array is None:
                tensor_array = paddle.tensor.array_write(tensor, i=index)
            else:
                paddle.tensor.array_write(tensor, i=index, array=tensor_array)
        return tensor_array


# class TestStridedSliceTensorArray(unittest.TestCase):
#     def setUp(self):
#         paddle.disable_static()

#     def grad_equal(self, g1, g2):
#         if g1 is None:
#             g1 = np.zeros_like(g2)
#         if g2 is None:
#             g2 = np.zeros_like(g1)
#         return np.array_equal(g1, g2)

#     def is_grads_equal(self, g1, g2):
#         for i, g in enumerate(g1):
#             self.assertTrue(
#                 self.grad_equal(g, g2[i]),
#                 msg=f"gradient_1:\n{g} \ngradient_2:\n{g2}",
#             )

#     def is_grads_equal_zeros(self, grads):
#         for g in grads:
#             self.assertTrue(
#                 self.grad_equal(np.zeros_like(g), g),
#                 msg=f"The gradient should be zeros, but received \n{g}",
#             )

#     def create_case(self, net):
#         inps1 = paddle.randn([1, net.input_size], dtype='float32')
#         inps2 = inps1.detach().clone()
#         l1 = net(inps1)
#         s1 = l1.numpy()
#         l1.sum().backward()
#         grads_dy = net.get_all_grads()
#         net.clear_all_grad()
#         grads_zeros = net.get_all_grads()

#         self.is_grads_equal_zeros(grads_zeros)

#         func = paddle.jit.to_static(net.forward, full_graph=True)
#         l2 = func(inps2)
#         s2 = l2.numpy()
#         l2.sum().backward()
#         grads_static = net.get_all_grads()
#         net.clear_all_grad()
#         # compare result of dygraph and static
#         self.is_grads_equal(grads_static, grads_dy)
#         np.testing.assert_array_equal(
#             s1,
#             s2,
#             err_msg=f'dygraph graph result:\n{l1.numpy()} \nstatic dygraph result:\n{l2.numpy()}',
#         )

#     def test_strided_slice_tensor_array_cuda_pinned_place(self):
#         if paddle.device.is_compiled_with_cuda():
#             with paddle.base.dygraph.guard():

#                 class Simple(paddle.nn.Layer):
#                     def __init__(self):
#                         super().__init__()

#                     def forward(self, inps):
#                         tensor_array = None
#                         for i, tensor in enumerate(inps):
#                             index = paddle.full(
#                                 shape=[1], dtype='int64', fill_value=i
#                             )
#                             if tensor_array is None:
#                                 tensor_array = paddle.tensor.array_write(
#                                     tensor, i=index
#                                 )
#                             else:
#                                 paddle.tensor.array_write(
#                                     tensor, i=index, array=tensor_array
#                                 )

#                         array1 = paddle.concat(tensor_array)
#                         array2 = paddle.concat(tensor_array[::-1])
#                         return array1 + array2 * array2

#                 net = Simple()
#                 func = paddle.jit.to_static(net.forward, full_graph=True)

#                 inps1 = paddle.to_tensor(
#                     np.random.randn(2, 10),
#                     place=paddle.CUDAPinnedPlace(),
#                     stop_gradient=False,
#                 )
#                 inps2 = paddle.to_tensor(
#                     np.random.randn(2, 10),
#                     place=paddle.CUDAPinnedPlace(),
#                     stop_gradient=False,
#                 )

#                 self.assertTrue(inps1.place.is_cuda_pinned_place())
#                 self.assertTrue(inps2.place.is_cuda_pinned_place())

#                 result = func([inps1, inps2])

#                 self.assertFalse(result.place.is_cuda_pinned_place())

#     def test_strided_slice_tensor_array(self):
#         class Net01(ArrayLayer):
#             def array_slice(self, tensors):
#                 return tensors[::-1]

#         self.create_case(Net01(array_size=10))

#         class Net02(ArrayLayer):
#             def array_slice(self, tensors):
#                 return tensors[::-2]

#         self.create_case(Net02(input_size=112, array_size=11))

#         class Net03(ArrayLayer):
#             def array_slice(self, tensors):
#                 return tensors[::-3]

#         self.create_case(Net03(input_size=112, array_size=9))

#         class Net04(ArrayLayer):
#             def array_slice(self, tensors):
#                 return tensors[1::-4]

#         self.create_case(Net04(input_size=112, array_size=9))

#         class Net05(ArrayLayer):
#             def array_slice(self, tensors):
#                 return tensors[:7:-4]

#         self.create_case(Net05(input_size=112, array_size=9))

#         class Net06(ArrayLayer):
#             def array_slice(self, tensors):
#                 return tensors[8:0:-4]

#         self.create_case(Net06(input_size=112, array_size=9))

#         class Net07(ArrayLayer):
#             def array_slice(self, tensors):
#                 return tensors[8:1:-4]

#         self.create_case(Net07(input_size=112, array_size=9))

#         class Net08(ArrayLayer):
#             def array_slice(self, tensors):
#                 return tensors[::2]

#         self.create_case(Net08(input_size=112, array_size=11))

#         class Net09(ArrayLayer):
#             def array_slice(self, tensors):
#                 return tensors[::3]

#         self.create_case(Net09(input_size=112, array_size=9))

#         class Net10(ArrayLayer):
#             def array_slice(self, tensors):
#                 return tensors[1::4]

#         self.create_case(Net10(input_size=112, array_size=9))

#         class Net11(ArrayLayer):
#             def array_slice(self, tensors):
#                 return tensors[:8:4]

#         self.create_case(Net11(input_size=112, array_size=9))

#         class Net12(ArrayLayer):
#             def array_slice(self, tensors):
#                 return tensors[1:8:4]

#         self.create_case(Net12(input_size=112, array_size=9))

#         class Net13(ArrayLayer):
#             def array_slice(self, tensors):
#                 return tensors[8:10:4]

#         self.create_case(Net13(input_size=112, array_size=13))

#         class Net14(ArrayLayer):
#             def array_slice(self, tensors):
#                 return tensors[3:10:4]

#         self.create_case(Net14(input_size=112, array_size=13))

#         class Net15(ArrayLayer):
#             def array_slice(self, tensors):
#                 return tensors[2:10:4]

#         self.create_case(Net15(input_size=112, array_size=13))

#         class Net16(ArrayLayer):
#             def array_slice(self, tensors):
#                 return tensors[3:10:3]

#         self.create_case(Net16(input_size=112, array_size=13))

#         class Net17(ArrayLayer):
#             def array_slice(self, tensors):
#                 return tensors[3:15:3]

#         self.create_case(Net17(input_size=112, array_size=13))

#         class Net18(ArrayLayer):
#             def array_slice(self, tensors):
#                 return tensors[0:15:3]

#         self.create_case(Net18(input_size=112, array_size=13))

#         class Net19(ArrayLayer):
#             def array_slice(self, tensors):
#                 return tensors[-1:-5:-3]

#         self.create_case(Net19(input_size=112, array_size=13))

#         class Net20(ArrayLayer):
#             def array_slice(self, tensors):
#                 return tensors[-1:-6:-3]

#         self.create_case(Net20(input_size=112, array_size=13))

#         class Net21(ArrayLayer):
#             def array_slice(self, tensors):
#                 return tensors[-3:-6:-3]

#         self.create_case(Net21(input_size=112, array_size=13))

#         class Net22(ArrayLayer):
#             def array_slice(self, tensors):
#                 return tensors[-5:-1:3]

#         self.create_case(Net22(input_size=112, array_size=13))

#         class Net23(ArrayLayer):
#             def array_slice(self, tensors):
#                 return tensors[-6:-1:3]

#         self.create_case(Net23(input_size=112, array_size=13))

#         class Net24(ArrayLayer):
#             def array_slice(self, tensors):
#                 return tensors[-6:-3:3]

#         self.create_case(Net24(input_size=112, array_size=13))

#         class Net25(ArrayLayer):
#             def array_slice(self, tensors):
#                 return tensors[0::3]

#         self.create_case(Net25(input_size=112, array_size=13))

#         class Net26(ArrayLayer):
#             def array_slice(self, tensors):
#                 return tensors[-60:20:3]

#         self.create_case(Net26(input_size=112, array_size=13))

#         class Net27(ArrayLayer):
#             def array_slice(self, tensors):
#                 return tensors[-3:-60:-3]

#         self.create_case(Net27(input_size=112, array_size=13))


@unittest.skipIf(
    not base.core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestStridedSliceFloat16(unittest.TestCase):
    def init_test_case(self):
        self.op_type = 'strided_slice'
        self.python_api = paddle.strided_slice
        self.input_shape = [3, 3, 3, 6, 7, 8]
        self.axes = [0, 1, 2, 3, 4, 5]
        self.starts = [1, 0, 0, 0, 1, 2]
        self.ends = [2, 2, 3, 1, 2, 8]
        self.strides = [1, 1, 1, 1, 1, 2]
        self.infer_flags = [1, 1, 1, 1, 1]

    def check_main(self, x_np, dtype):
        paddle.disable_static()
        x_np = x_np.astype(dtype)
        x = paddle.to_tensor(x_np)
        x.stop_gradient = False
        output = strided_slice_native_forward(
            x, self.axes, self.starts, self.ends, self.strides
        )
        x_grad = paddle.grad(output, x)
        output_np = output[0].numpy().astype('float32')
        x_grad_np = x_grad[0].numpy().astype('float32')
        paddle.enable_static()
        return output_np, x_grad_np

    def test_check(self):
        self.init_test_case()
        x_np = np.random.random(self.input_shape).astype("float16")

        output_np_fp16, x_grad_np_fp16 = self.check_main(x_np, 'float16')
        output_np_fp32, x_grad_np_fp32 = self.check_main(x_np, 'float32')

        np.testing.assert_allclose(output_np_fp16, output_np_fp32)

        np.testing.assert_allclose(x_grad_np_fp16, x_grad_np_fp32)


if __name__ == "__main__":
    unittest.main()
