#   Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.base import core


class TestClipOp(OpTest):
    def setUp(self):
        self.max_relative_error = 0.006
        self.python_api = paddle.clip
        self.public_python_api = paddle.clip

        self.inputs = {}
        self.initTestCase()

        self.op_type = "clip"
        self.prim_op_type = "comp"
        self.attrs = {}
        self.attrs['min'] = self.min
        self.attrs['max'] = self.max
        if 'Min' in self.inputs:
            min_v = self.inputs['Min']
        else:
            min_v = self.attrs['min']

        if 'Max' in self.inputs:
            max_v = self.inputs['Max']
        else:
            max_v = self.attrs['max']

        input = np.random.random(self.shape).astype(self.dtype)
        input[np.abs(input - min_v) < self.max_relative_error] = 0.5
        input[np.abs(input - max_v) < self.max_relative_error] = 0.5
        self.inputs['X'] = input
        self.outputs = {'Out': np.clip(self.inputs['X'], min_v, max_v)}
        self.check_cinn = ('Min' not in self.inputs) and (
            'Max' not in self.inputs
        )

    def test_check_output(self):
        paddle.enable_static()
        self.check_output(
            check_cinn=self.check_cinn,
            check_pir=True,
            check_prim_pir=True,
            check_symbol_infer=False,
        )
        paddle.disable_static()

    def test_check_grad_normal(self):
        paddle.enable_static()
        self.check_grad(['X'], 'Out', check_pir=True)
        paddle.disable_static()

    def initTestCase(self):
        self.dtype = np.float32
        self.shape = (4, 10, 10)
        self.max = 0.8
        self.min = 0.3
        self.inputs['Max'] = np.array([0.8]).astype(self.dtype)
        self.inputs['Min'] = np.array([0.1]).astype(self.dtype)


class TestCase1(TestClipOp):
    def initTestCase(self):
        self.dtype = np.float32
        self.shape = (8, 16, 8)
        self.max = 0.7
        self.min = 0.0


class TestCase2(TestClipOp):
    def initTestCase(self):
        self.dtype = np.float32
        self.shape = (8, 16)
        self.max = 1.0
        self.min = 0.0


class TestCase3(TestClipOp):
    def initTestCase(self):
        self.dtype = np.float32
        self.shape = (4, 8, 16)
        self.max = 0.7
        self.min = 0.2


class TestCase4(TestClipOp):
    def initTestCase(self):
        self.dtype = np.float32
        self.shape = (4, 8, 8)
        self.max = 0.7
        self.min = 0.2
        self.inputs['Max'] = np.array([0.8]).astype(self.dtype)
        self.inputs['Min'] = np.array([0.3]).astype(self.dtype)


class TestCase5(TestClipOp):
    def initTestCase(self):
        self.dtype = np.float32
        self.shape = (4, 8, 16)
        self.max = 0.5
        self.min = 0.5


class TestFP16Case1(TestClipOp):
    def initTestCase(self):
        self.dtype = np.float16
        self.shape = (8, 16, 8)
        self.max = 0.7
        self.min = 0.0


class TestFP16Case2(TestClipOp):
    def initTestCase(self):
        self.dtype = np.float16
        self.shape = (8, 16)
        self.max = 1.0
        self.min = 0.0


class TestFP16Case3(TestClipOp):
    def initTestCase(self):
        self.dtype = np.float16
        self.shape = (4, 8, 16)
        self.max = 0.7
        self.min = 0.2


class TestFP16Case4(TestClipOp):
    def initTestCase(self):
        self.dtype = np.float16
        self.shape = (4, 8, 8)
        self.max = 0.7
        self.min = 0.2
        self.inputs['Max'] = np.array([0.8]).astype(self.dtype)
        self.inputs['Min'] = np.array([0.3]).astype(self.dtype)


class TestFP16Case5(TestClipOp):
    def initTestCase(self):
        self.dtype = np.float16
        self.shape = (4, 8, 16)
        self.max = 0.5
        self.min = 0.5


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA or not support the bfloat16",
)
class TestClipBF16Op(OpTest):
    def setUp(self):
        self.max_relative_error = 0.006
        self.python_api = paddle.clip
        self.public_python_api = paddle.clip

        self.inputs = {}
        self.initTestCase()

        self.op_type = "clip"
        self.prim_op_type = "comp"
        self.attrs = {}
        self.attrs['min'] = self.min
        self.attrs['max'] = self.max
        if 'Min' in self.inputs:
            min_v = self.inputs['Min']
        else:
            min_v = self.attrs['min']

        if 'Max' in self.inputs:
            max_v = self.inputs['Max']
        else:
            max_v = self.attrs['max']

        input = np.random.random(self.shape).astype(np.float32)
        input[np.abs(input - min_v) < self.max_relative_error] = 0.5
        input[np.abs(input - max_v) < self.max_relative_error] = 0.5
        self.inputs['X'] = convert_float_to_uint16(input)
        out = np.clip(input, min_v, max_v)
        self.outputs = {'Out': convert_float_to_uint16(out)}

    def test_check_output(self):
        if paddle.is_compiled_with_cuda():
            place = paddle.CUDAPlace(0)
            paddle.enable_static()
            self.check_output_with_place(
                place,
                check_pir=True,
                check_prim_pir=True,
                check_symbol_infer=False,
            )
            paddle.disable_static()

    def test_check_grad_normal(self):
        if paddle.is_compiled_with_cuda():
            place = paddle.CUDAPlace(0)
            paddle.enable_static()
            self.check_grad_with_place(place, ['X'], 'Out', check_pir=True)
            paddle.disable_static()

    def initTestCase(self):
        self.shape = (4, 10, 10)
        self.max = 0.8
        self.min = 0.3
        self.inputs['Max'] = np.array([0.8]).astype(np.float32)
        self.inputs['Min'] = np.array([0.1]).astype(np.float32)


class TestBF16Case1(TestClipBF16Op):
    def initTestCase(self):
        self.shape = (8, 16, 8)
        self.max = 0.7
        self.min = 0.0


class TestBF16Case2(TestClipBF16Op):
    def initTestCase(self):
        self.shape = (8, 16)
        self.max = 1.0
        self.min = 0.0


class TestBF16Case3(TestClipBF16Op):
    def initTestCase(self):
        self.shape = (4, 8, 16)
        self.max = 0.7
        self.min = 0.2


class TestBF16Case4(TestClipBF16Op):
    def initTestCase(self):
        self.shape = (4, 8, 8)
        self.max = 0.7
        self.min = 0.2
        self.inputs['Max'] = np.array([0.8]).astype(np.float32)
        self.inputs['Min'] = np.array([0.3]).astype(np.float32)


class TestBF16Case5(TestClipBF16Op):
    def initTestCase(self):
        self.shape = (4, 8, 16)
        self.max = 0.5
        self.min = 0.5


class TestClipOpError(unittest.TestCase):

    def test_errors(self):
        paddle.enable_static()
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            input_data = np.random.random((2, 4)).astype("float32")

            def test_Variable():
                paddle.clip(x=input_data, min=-1.0, max=1.0)

            self.assertRaises(TypeError, test_Variable)
        paddle.disable_static()


class TestClipAPI(unittest.TestCase):
    def _executed_api(self, x, min=None, max=None):
        return paddle.clip(x, min, max)

    def test_clip(self):
        paddle.enable_static()
        data_shape = [1, 9, 9, 4]
        data = np.random.random(data_shape).astype('float32')
        place = (
            base.CUDAPlace(0)
            if base.core.is_compiled_with_cuda()
            else base.CPUPlace()
        )
        exe = base.Executor(place)

        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            images = paddle.static.data(
                name='image', shape=data_shape, dtype='float32'
            )
            min = paddle.static.data(name='min', shape=[1], dtype='float32')
            max = paddle.static.data(name='max', shape=[1], dtype='float32')
            out_1 = self._executed_api(images, min=min, max=max)
            out_2 = self._executed_api(images, min=0.2, max=0.9)
            out_3 = self._executed_api(images, min=0.3)
            out_4 = self._executed_api(images, max=0.7)
            out_5 = self._executed_api(images, min=min)
            out_6 = self._executed_api(images, max=max)
            out_7 = self._executed_api(images, max=-1.0)
            out_8 = self._executed_api(images)
            out_9 = self._executed_api(
                paddle.cast(images, 'float64'), min=0.2, max=0.9
            )
            out_10 = self._executed_api(
                paddle.cast(images * 10, 'int32'), min=2, max=8
            )
            out_11 = self._executed_api(
                paddle.cast(images * 10, 'int64'), min=2, max=8
            )

        (
            res1,
            res2,
            res3,
            res4,
            res5,
            res6,
            res7,
            res8,
            res9,
            res10,
            res11,
        ) = exe.run(
            main,
            feed={
                "image": data,
                "min": np.array([0.2]).astype('float32'),
                "max": np.array([0.8]).astype('float32'),
            },
            fetch_list=[
                out_1,
                out_2,
                out_3,
                out_4,
                out_5,
                out_6,
                out_7,
                out_8,
                out_9,
                out_10,
                out_11,
            ],
        )

        np.testing.assert_allclose(res1, data.clip(0.2, 0.8), rtol=1e-05)
        np.testing.assert_allclose(res2, data.clip(0.2, 0.9), rtol=1e-05)
        np.testing.assert_allclose(res3, data.clip(min=0.3), rtol=1e-05)
        np.testing.assert_allclose(res4, data.clip(max=0.7), rtol=1e-05)
        np.testing.assert_allclose(res5, data.clip(min=0.2), rtol=1e-05)
        np.testing.assert_allclose(res6, data.clip(max=0.8), rtol=1e-05)
        np.testing.assert_allclose(res7, data.clip(max=-1), rtol=1e-05)
        np.testing.assert_allclose(res8, data, rtol=1e-05)
        np.testing.assert_allclose(
            res9, data.astype(np.float64).clip(0.2, 0.9), rtol=1e-05
        )
        np.testing.assert_allclose(
            res10, (data * 10).astype(np.int32).clip(2, 8), rtol=1e-05
        )
        np.testing.assert_allclose(
            res11, (data * 10).astype(np.int64).clip(2, 8), rtol=1e-05
        )
        paddle.disable_static()

    def test_clip_dygraph(self):
        paddle.disable_static()
        place = (
            base.CUDAPlace(0)
            if base.core.is_compiled_with_cuda()
            else base.CPUPlace()
        )
        paddle.disable_static(place)
        data_shape = [1, 9, 9, 4]
        data = np.random.random(data_shape).astype('float32')
        images = paddle.to_tensor(data, dtype='float32')
        v_min = paddle.to_tensor(np.array([0.2], dtype=np.float32))
        v_max = paddle.to_tensor(np.array([0.8], dtype=np.float32))

        out_1 = self._executed_api(images, min=0.2, max=0.8)
        images = paddle.to_tensor(data, dtype='float32')
        out_2 = self._executed_api(images, min=0.2, max=0.9)
        images = paddle.to_tensor(data, dtype='float32')
        out_3 = self._executed_api(images, min=v_min, max=v_max)

        out_4 = self._executed_api(
            paddle.cast(images * 10, 'int32'), min=2, max=8
        )
        out_5 = self._executed_api(
            paddle.cast(images * 10, 'int64'), min=2, max=8
        )
        # test with numpy.generic
        out_6 = self._executed_api(images, min=np.abs(0.2), max=np.abs(0.8))

        np.testing.assert_allclose(
            out_1.numpy(), data.clip(0.2, 0.8), rtol=1e-05
        )
        np.testing.assert_allclose(
            out_2.numpy(), data.clip(0.2, 0.9), rtol=1e-05
        )
        np.testing.assert_allclose(
            out_3.numpy(), data.clip(0.2, 0.8), rtol=1e-05
        )
        np.testing.assert_allclose(
            out_4.numpy(), (data * 10).astype(np.int32).clip(2, 8), rtol=1e-05
        )
        np.testing.assert_allclose(
            out_5.numpy(), (data * 10).astype(np.int64).clip(2, 8), rtol=1e-05
        )
        np.testing.assert_allclose(
            out_6.numpy(), data.clip(0.2, 0.8), rtol=1e-05
        )

    def test_clip_dygraph_default_max(self):
        paddle.disable_static()
        x_int32 = paddle.to_tensor([1, 2, 3], dtype="int32")
        x_int64 = paddle.to_tensor([1, 2, 3], dtype="int64")
        x_f32 = paddle.to_tensor([1, 2, 3], dtype="float32")
        egr_out1 = paddle.clip(x_int32, min=1)
        egr_out2 = paddle.clip(x_int64, min=1)
        egr_out3 = paddle.clip(x_f32, min=1)
        x_int32 = paddle.to_tensor([1, 2, 3], dtype="int32")
        x_int64 = paddle.to_tensor([1, 2, 3], dtype="int64")
        x_f32 = paddle.to_tensor([1, 2, 3], dtype="float32")
        out1 = paddle.clip(x_int32, min=1)
        out2 = paddle.clip(x_int64, min=1)
        out3 = paddle.clip(x_f32, min=1)
        np.testing.assert_allclose(out1.numpy(), egr_out1.numpy(), rtol=1e-05)
        np.testing.assert_allclose(out2.numpy(), egr_out2.numpy(), rtol=1e-05)
        np.testing.assert_allclose(out3.numpy(), egr_out3.numpy(), rtol=1e-05)

    def test_errors(self):
        paddle.enable_static()
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            x1 = paddle.static.data(name='x1', shape=[1], dtype="int16")
            x2 = paddle.static.data(name='x2', shape=[1], dtype="int8")
            self.assertRaises(TypeError, paddle.clip, x=x1, min=0.2, max=0.8)
            self.assertRaises(TypeError, paddle.clip, x=x2, min=0.2, max=0.8)
        paddle.disable_static()


class TestClipOpFp16(unittest.TestCase):

    def test_fp16(self):
        if base.core.is_compiled_with_cuda():
            paddle.enable_static()
            data_shape = [1, 9, 9, 4]
            data = np.random.random(data_shape).astype('float16')

            with paddle.static.program_guard(paddle.static.Program()):
                images = paddle.static.data(
                    name='image1', shape=data_shape, dtype='float16'
                )
                min = paddle.static.data(
                    name='min1', shape=[1], dtype='float16'
                )
                max = paddle.static.data(
                    name='max1', shape=[1], dtype='float16'
                )
                out = paddle.clip(images, min, max)
                place = paddle.CUDAPlace(0)
                exe = paddle.static.Executor(place)
                res1 = exe.run(
                    feed={
                        "image1": data,
                        "min1": np.array([0.2]).astype('float16'),
                        "max1": np.array([0.8]).astype('float16'),
                    },
                    fetch_list=[out],
                )
            paddle.disable_static()


class TestInplaceClipAPI(TestClipAPI):
    def _executed_api(self, x, min=None, max=None):
        return x.clip_(min, max)


class TestClipTensorAPI(unittest.TestCase):
    def initCase(self):
        self.x_shape = [10, 10, 1]
        self.min_shape = [10]
        self.max_shape = [10]
        self.dtype = 'float32'

    def setUp(self):
        self.initCase()
        self.place = (
            base.CUDAPlace(0)
            if base.core.is_compiled_with_cuda()
            else base.CPUPlace()
        )
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        if self.min_shape is None:
            self.min = None
        else:
            self.min = np.random.random(self.min_shape).astype(self.dtype)
        if self.max_shape is None:
            self.max = None
        else:
            self.max = np.random.random(self.max_shape).astype(self.dtype)
        self.out_np = self.x.clip(self.min, self.max)

    def check_dygraph_api(self):
        if self.dtype == 'float16':
            return
        paddle.disable_static(self.place)
        x_pd = paddle.to_tensor(self.x)
        if self.min is None:
            min = None
        else:
            min = paddle.to_tensor(self.min)
        if self.max is None:
            max = None
        else:
            max = paddle.to_tensor(self.max)
        out_pd = paddle.clip(x_pd, min, max)
        np.testing.assert_allclose(self.out_np, out_pd.numpy())
        paddle.enable_static()

    def check_static_api(self):
        if self.dtype == 'float16':
            return
        paddle.enable_static()
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        exe = paddle.static.Executor(self.place)
        with paddle.static.program_guard(main_program, startup_program):
            x_pd = paddle.static.data(
                name='x', shape=self.x_shape, dtype=self.dtype
            )
            if self.min is not None:
                min_pd = paddle.static.data(
                    name='min', shape=self.min_shape, dtype=self.dtype
                )
            else:
                min_pd = None
            if self.max is not None:
                max_pd = paddle.static.data(
                    name='max', shape=self.max_shape, dtype=self.dtype
                )
            else:
                max_pd = None
            out_pd = paddle.clip(x_pd, min_pd, max_pd)
        res = exe.run(
            main_program,
            feed={'x': self.x, 'min': self.min, 'max': self.max},
            fetch_list=[out_pd],
        )
        np.testing.assert_allclose(self.out_np, res[0])
        paddle.disable_static()

    def check_inplace_api(self):
        if self.dtype == 'float16':
            return
        paddle.disable_static(self.place)
        x_pd = paddle.rand(self.x_shape, dtype=self.dtype)
        min_pd = paddle.rand([self.x_shape[0]], dtype=self.dtype)
        max_pd = paddle.rand([self.x_shape[0]], dtype=self.dtype)
        out_np = x_pd.numpy().clip(min_pd.numpy(), max_pd.numpy())
        x_pd.clip_(min_pd, max_pd)
        np.testing.assert_allclose(out_np, x_pd.numpy())
        paddle.enable_static()

    def test_fp16_api(self):
        if base.core.is_compiled_with_cuda():
            if self.dtype == 'float16':
                paddle.enable_static()
                main_program = paddle.static.Program()
                startup_program = paddle.static.Program()
                exe = paddle.static.Executor(self.place)
                with paddle.static.program_guard(main_program, startup_program):
                    x_pd = paddle.static.data(
                        name='x', shape=self.x_shape, dtype=self.dtype
                    )
                    if self.min is not None:
                        min_pd = paddle.static.data(
                            name='min', shape=self.min_shape, dtype=self.dtype
                        )
                    else:
                        min_pd = None
                    if self.max is not None:
                        max_pd = paddle.static.data(
                            name='max', shape=self.max_shape, dtype=self.dtype
                        )
                    else:
                        max_pd = None
                    out_pd = paddle.clip(x_pd, min_pd, max_pd)
                res = exe.run(
                    main_program,
                    feed={
                        'x': self.x,
                        'min': self.min,
                        'max': self.max,
                    },
                    fetch_list=[out_pd],
                )
                paddle.disable_static()


class TestClipTensorCase4(TestClipTensorAPI):
    def initCase(self):
        self.x_shape = [10, 1, 10]
        self.min_shape = [10]
        self.max_shape = [10]
        self.dtype = 'float64'


class TestClipTensorCase5(TestClipTensorAPI):
    def initCase(self):
        self.x_shape = [10, 1, 10]
        self.min_shape = [10]
        self.max_shape = [10]
        self.dtype = 'float32'


class TestClipTensorCase6(TestClipTensorAPI):
    def initCase(self):
        self.x_shape = [10, 1, 10]
        self.min_shape = [10]
        self.max_shape = [10]
        self.dtype = 'float16'


class TestClipTensorCase7(TestClipTensorAPI):
    def initCase(self):
        self.x_shape = [10, 1, 10]
        self.min_shape = None
        self.max_shape = [10]
        self.dtype = 'float64'


class TestClipTensorCase8(TestClipTensorAPI):
    def initCase(self):
        self.x_shape = [10, 1, 10]
        self.min_shape = None
        self.max_shape = [10]
        self.dtype = 'float32'


class TestClipTensorCase9(TestClipTensorAPI):
    def initCase(self):
        self.x_shape = [10, 1, 10]
        self.min_shape = None
        self.max_shape = [10]
        self.dtype = 'float16'


class TestClipTensorCase10(TestClipTensorAPI):
    def initCase(self):
        self.x_shape = [10, 1, 10]
        self.min_shape = [10]
        self.max_shape = None
        self.dtype = 'float64'


class TestClipTensorCase11(TestClipTensorAPI):
    def initCase(self):
        self.x_shape = [10, 1, 10]
        self.min_shape = [10]
        self.max_shape = None
        self.dtype = 'float32'


class TestClipTensorCase12(TestClipTensorAPI):
    def initCase(self):
        self.x_shape = [10, 1, 10]
        self.min_shape = [10]
        self.max_shape = None
        self.dtype = 'float16'


class TestClipTensorCase13(TestClipTensorAPI):
    def initCase(self):
        self.dtype = 'int32'
        self.x_shape = [10, 1, 10]
        self.min_shape = [10]
        self.max_shape = [10]


class TestClipTensorCase14(TestClipTensorAPI):
    def initCase(self):
        self.dtype = 'int64'
        self.x_shape = [10, 1, 10]
        self.min_shape = [10]
        self.max_shape = [10]


class TestClipTensorCase15(TestClipTensorAPI):
    def initCase(self):
        self.dtype = 'int32'
        self.x_shape = [10, 1, 10]
        self.min_shape = None
        self.max_shape = [10]


class TestClipTensorCase16(TestClipTensorAPI):
    def initCase(self):
        self.dtype = 'int64'
        self.x_shape = [10, 1, 10]
        self.min_shape = None
        self.max_shape = [10]


class TestClipTensorCase17(TestClipTensorAPI):
    def initCase(self):
        self.dtype = 'int32'
        self.x_shape = [10, 1, 10]
        self.min_shape = [10]
        self.max_shape = None


class TestClipTensorCase18(TestClipTensorAPI):
    def initCase(self):
        self.dtype = 'int64'
        self.x_shape = [10, 1, 10]
        self.min_shape = [10]
        self.max_shape = None


class TestClipTensorCase19(TestClipTensorAPI):
    def initCase(self):
        self.dtype = 'float32'
        self.x_shape = [10]
        self.min_shape = [10, 1, 10]
        self.max_shape = [10]


class TestClipTensorCase20(TestClipTensorAPI):
    def initCase(self):
        self.dtype = 'float32'
        self.x_shape = [10]
        self.min_shape = [10]
        self.max_shape = [10, 1, 10]


class TestClipTensorOp(OpTest):
    def setUp(self):
        self.max_relative_error = 0.006
        self.op_type = "clip_tensor"
        self.python_api = paddle.clip

        self.inputs = {}
        self.initTestCase()
        input = np.random.random(self.shape).astype(self.dtype)
        min_v = np.full(self.shape, self.min_value).astype(self.dtype)
        max_v = np.full(self.shape, self.max_value).astype(self.dtype)

        input[np.abs(input - min_v) < self.max_relative_error] = 0.5
        input[np.abs(input - max_v) < self.max_relative_error] = 0.5

        self.inputs['min'] = min_v
        self.inputs['max'] = max_v
        self.inputs['x'] = input
        self.outputs = {'out': np.clip(input, min_v, max_v)}

    def test_check_output(self):
        paddle.enable_static()
        self.check_output(check_pir=True)

    def test_check_grad_normal(self):
        paddle.enable_static()
        self.check_grad(['x'], 'out', check_pir=True)

    def initTestCase(self):
        self.dtype = np.float32
        self.shape = (8, 5, 6)
        self.min_value = 0.8
        self.max_value = 0.3


class TestClipTensorOpCase1(TestClipTensorOp):
    def initTestCase(self):
        self.dtype = np.float32
        self.shape = (5, 6, 8)
        self.max_value = 0.7
        self.min_value = 0.0


class TestClipTensorOpCase2(TestClipTensorOp):
    def initTestCase(self):
        self.dtype = np.float32
        self.shape = (8, 5, 6)
        self.max_value = 1.0
        self.min_value = 0.0


class TestClipTensorOpCase3(TestClipTensorOp):
    def initTestCase(self):
        self.dtype = np.float32
        self.shape = (4, 8, 6)
        self.max_value = 0.7
        self.min_value = 0.2


class TestClipTensorOpFP16Case1(TestClipTensorOp):
    def initTestCase(self):
        self.dtype = np.float16
        self.shape = (5, 6, 8)
        self.max_value = 0.7
        self.min_value = 0.0


class TestClipTensorOpFP16Case2(TestClipTensorOp):
    def initTestCase(self):
        self.dtype = np.float16
        self.shape = (8, 5, 6)
        self.max_value = 1.0
        self.min_value = 0.0


class TestClipTensorOpFP16Case3(TestClipTensorOp):
    def initTestCase(self):
        self.dtype = np.float16
        self.shape = (5, 8, 6)
        self.max_value = 0.7
        self.min_value = 0.2


class TestClipTensorOpFP64Case1(TestClipTensorOp):
    def initTestCase(self):
        self.dtype = np.float64
        self.shape = (8, 6, 5)
        self.max_value = 0.7
        self.min_value = 0.0


class TestClipTensorOpFP64Case2(TestClipTensorOp):
    def initTestCase(self):
        self.dtype = np.float64
        self.shape = (8, 5, 6)
        self.max_value = 1.0
        self.min_value = 0.0


class TestClipTensorOpFP64Case3(TestClipTensorOp):
    def initTestCase(self):
        self.dtype = np.float64
        self.shape = (4, 8, 6)
        self.max_value = 0.7
        self.min_value = 0.2


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and not support the bfloat16",
)
class TestClipTensorBF16Op(OpTest):
    def setUp(self):
        self.max_relative_error = 0.006
        self.op_type = "clip_tensor"
        self.python_api = paddle.clip
        self.inputs = {}
        self.initTestCase()

        self.inputs['x'] = np.random.random(self.shape).astype(np.float32)
        self.inputs['min'] = np.full(self.shape, self.min_value).astype(
            np.float32
        )
        self.inputs['max'] = np.full(self.shape, self.max_value).astype(
            np.float32
        )
        min_v = self.inputs['min']
        max_v = self.inputs['max']

        self.inputs['x'][
            np.abs(self.inputs['x'] - min_v) < self.max_relative_error
        ] = 0.5
        self.inputs['x'][
            np.abs(self.inputs['x'] - max_v) < self.max_relative_error
        ] = 0.5

        self.inputs['x'] = convert_float_to_uint16(self.inputs['x'])
        self.inputs['min'] = convert_float_to_uint16(self.inputs['min'])
        self.inputs['max'] = convert_float_to_uint16(self.inputs['max'])
        out = np.clip(self.inputs['x'], min_v, max_v)

        self.outputs = {'out': convert_float_to_uint16(out)}

    def test_check_output(self):
        place = paddle.CUDAPlace(0)
        paddle.enable_static()
        self.check_output_with_place(place)

    def test_check_grad_normal(self):
        place = paddle.CUDAPlace(0)
        paddle.enable_static()
        self.check_grad_with_place(place, ['x'], 'out')

    def initTestCase(self):
        self.shape = (8, 5, 6)
        self.min_value = 0.8
        self.max_value = 0.3


class TestClipTensorOBF16Case1(TestClipTensorBF16Op):
    def initTestCase(self):
        self.shape = (8, 6, 5)
        self.max_value = 0.7
        self.min_value = 0.0


class TestClipTensorOpBF16Case2(TestClipTensorBF16Op):
    def initTestCase(self):
        self.shape = (5, 8, 6)
        self.max_value = 1.0
        self.min_value = 0.0


class TestClipTensorOpBF16Case3(TestClipTensorBF16Op):
    def initTestCase(self):
        self.shape = (4, 8, 7)
        self.max_value = 0.7
        self.min_value = 0.2


if __name__ == '__main__':
    unittest.main()
