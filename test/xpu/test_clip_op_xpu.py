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
from get_test_cover_info import (
    XPUOpTestWrapper,
    create_test_class,
    get_xpu_op_support_types,
)
from op_test import convert_float_to_uint16, convert_uint16_to_float
from op_test_xpu import XPUOpTest

import paddle
from paddle import base
from paddle.base import Program, core, program_guard


class XPUTestClipOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'clip'
        self.use_dynamic_create_class = False

    class TestClipOp(XPUOpTest):
        def setUp(self):
            self.python_api = paddle.clip
            self.init_dtype()
            self.set_xpu()
            self.op_type = "clip"
            self.place = paddle.XPUPlace(0)
            self.inputs = {}
            self.init_data()
            self.set_attrs()
            self.set_inputs()
            if self.dtype == np.uint16:
                self.outputs = {
                    'Out': convert_float_to_uint16(
                        np.clip(
                            convert_uint16_to_float(self.inputs['X']),
                            np.array([self.min_v]).astype(np.float32).item(),
                            np.array([self.min_v]).astype(np.float32).item(),
                        )
                    )
                }
            else:
                self.outputs = {
                    'Out': np.clip(
                        self.inputs['X'],
                        np.array([self.min_v]).astype(self.dtype).item(),
                        np.array([self.max_v]).astype(self.dtype).item(),
                    )
                }

        def set_xpu(self):
            self.__class__.use_xpu = True
            self.__class__.no_need_check_grad = False
            self.__class__.op_type = self.dtype

        def init_data(self):
            self.shape = (4, 10, 10)
            self.max = 0.8
            self.min = 0.3

        def set_inputs(self):
            if 'Min' in self.inputs:
                min_v = self.inputs['Min']
            else:
                min_v = self.attrs['min']

            if 'Max' in self.inputs:
                max_v = self.inputs['Max']
            else:
                max_v = self.attrs['max']

            self.min_v = min_v
            self.max_v = max_v
            self.max_relative_error = 0.006
            input = np.random.random(self.shape).astype("float32")
            input[np.abs(input - min_v) < self.max_relative_error] = 0.5
            input[np.abs(input - max_v) < self.max_relative_error] = 0.5
            if self.dtype == np.uint16:
                input = convert_float_to_uint16(input)
            else:
                input = input.astype(self.dtype)
            self.inputs['X'] = input

        def set_attrs(self):
            self.attrs = {}
            self.attrs['min'] = self.min
            self.attrs['max'] = self.max

        def init_dtype(self):
            self.dtype = self.in_type

        def test_check_output(self):
            paddle.enable_static()
            self.check_output_with_place(self.place)
            paddle.disable_static()

        def test_check_grad(self):
            if hasattr(self, "no_need_check_grad") and self.no_need_check_grad:
                return
            if core.is_compiled_with_xpu():
                paddle.enable_static()
                self.check_grad_with_place(
                    self.place, ['X'], 'Out', check_dygraph=True
                )
                paddle.disable_static()

    class TestClipOp1(TestClipOp):
        def init_data(self):
            self.shape = (8, 16, 8)
            self.max = 0.7
            self.min = 0.0

    class TestClipOp2(TestClipOp):
        def init_data(self):
            self.shape = (8, 16)
            self.max = 1.0
            self.min = 0.0

    class TestClipOp3(TestClipOp):
        def init_data(self):
            self.shape = (4, 8, 16)
            self.max = 0.7
            self.min = 0.2

    class TestClipOp4(TestClipOp):
        def init_data(self):
            self.shape = (4, 8, 8)
            self.max = 0.7
            self.min = 0.2
            self.inputs['Max'] = np.array([0.8]).astype('float32')
            self.inputs['Min'] = np.array([0.3]).astype('float32')

    class TestClipOp5(TestClipOp):
        def init_data(self):
            self.shape = (4, 8, 16)
            self.max = 0.5
            self.min = 0.5


class TestClipOpError(unittest.TestCase):
    def test_errors(self):
        paddle.enable_static()
        with program_guard(Program(), Program()):
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
        train_prog = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(train_prog, startup):
            data_shape = [1, 9, 9, 4]
            data = np.random.random(data_shape).astype('float32')
            images = paddle.static.data(
                name='image', shape=data_shape, dtype='float32'
            )
            min = paddle.static.data(name='min', shape=[1], dtype='float32')
            max = paddle.static.data(name='max', shape=[1], dtype='float32')

            place = (
                base.XPUPlace(0)
                if base.core.is_compiled_with_xpu()
                else base.CPUPlace()
            )
            exe = base.Executor(place)

            out_1 = self._executed_api(images, min=min, max=max)
            out_2 = self._executed_api(images, min=0.2, max=0.9)
            out_3 = self._executed_api(images, min=0.3)
            out_4 = self._executed_api(images, max=0.7)
            out_5 = self._executed_api(images, min=min)
            out_6 = self._executed_api(images, max=max)
            out_7 = self._executed_api(images, max=-1.0)
            out_8 = self._executed_api(images)
            res1, res2, res3, res4, res5, res6, res7, res8 = exe.run(
                train_prog,
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
                ],
            )

            np.testing.assert_allclose(res1, data.clip(0.2, 0.8))
            np.testing.assert_allclose(res2, data.clip(0.2, 0.9))
            np.testing.assert_allclose(res3, data.clip(min=0.3))
            np.testing.assert_allclose(res4, data.clip(max=0.7))
            np.testing.assert_allclose(res5, data.clip(min=0.2))
            np.testing.assert_allclose(res6, data.clip(max=0.8))
            np.testing.assert_allclose(res7, data.clip(max=-1))
            np.testing.assert_allclose(res8, data)
        paddle.disable_static()

    def test_clip_dygraph(self):
        paddle.disable_static()
        place = (
            base.XPUPlace(0)
            if base.core.is_compiled_with_xpu()
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

        np.testing.assert_allclose(out_1.numpy(), data.clip(0.2, 0.8))
        np.testing.assert_allclose(out_2.numpy(), data.clip(0.2, 0.9))
        np.testing.assert_allclose(out_3.numpy(), data.clip(0.2, 0.8))

    def test_errors(self):
        paddle.enable_static()
        x1 = paddle.static.data(name='x1', shape=[1], dtype="int16")
        x2 = paddle.static.data(name='x2', shape=[1], dtype="int8")
        self.assertRaises(TypeError, paddle.clip, x=x1, min=0.2, max=0.8)
        self.assertRaises(TypeError, paddle.clip, x=x2, min=0.2, max=0.8)
        paddle.disable_static()


class TestInplaceClipAPI(TestClipAPI):
    def _executed_api(self, x, min=None, max=None):
        return x.clip_(min, max)


support_types = get_xpu_op_support_types('clip')
for stype in support_types:
    # TODO(lilujia): disable int32 and int64 test temporarily, as xdnn not support corresponding resuce_mean
    if stype in ["int32", "int64"]:
        continue
    create_test_class(globals(), XPUTestClipOp, stype)


class TestClipTensorAPI(unittest.TestCase):
    def initCase(self):
        self.x_shape = [10, 10, 1]
        self.min_shape = [10]
        self.max_shape = [10]
        self.dtype = 'float32'

    def setUp(self):
        self.initCase()
        self.place = (
            base.XPUPlace(0)
            if base.core.is_compiled_with_xpu()
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
        paddle.disable_static(self.place)
        x_pd = paddle.rand(self.x_shape, dtype=self.dtype)
        min_pd = paddle.rand([self.x_shape[0]], dtype=self.dtype)
        max_pd = paddle.rand([self.x_shape[0]], dtype=self.dtype)
        out_np = x_pd.numpy().clip(min_pd.numpy(), max_pd.numpy())
        x_pd.clip_(min_pd, max_pd)
        np.testing.assert_allclose(out_np, x_pd.numpy())
        paddle.enable_static()


class TestClipTensorCase1(TestClipTensorAPI):
    def initCase(self):
        self.x_shape = [10, 1, 10]
        self.min_shape = [10]
        self.max_shape = [10]
        self.dtype = 'float32'



class TestClipTensorCase2(TestClipTensorAPI):
    def initCase(self):
        self.x_shape = [10, 1, 10]
        self.min_shape = None
        self.max_shape = [10]
        self.dtype = 'float32'


class TestClipTensorCase3(TestClipTensorAPI):
    def initCase(self):
        self.x_shape = [10, 1, 10]
        self.min_shape = [10]
        self.max_shape = None
        self.dtype = 'float32'


class TestClipTensorCase4(TestClipTensorAPI):
    def initCase(self):
        self.dtype = 'int32'
        self.x_shape = [10, 1, 10]
        self.min_shape = [10]
        self.max_shape = [10]


class TestClipTensorCase5(TestClipTensorAPI):
    def initCase(self):
        self.dtype = 'int64'
        self.x_shape = [10, 1, 10]
        self.min_shape = [10]
        self.max_shape = [10]


class TestClipTensorCase6(TestClipTensorAPI):
    def initCase(self):
        self.dtype = 'int32'
        self.x_shape = [10, 1, 10]
        self.min_shape = None
        self.max_shape = [10]


class TestClipTensorCase7(TestClipTensorAPI):
    def initCase(self):
        self.dtype = 'int64'
        self.x_shape = [10, 1, 10]
        self.min_shape = None
        self.max_shape = [10]


class TestClipTensorCase8(TestClipTensorAPI):
    def initCase(self):
        self.dtype = 'int32'
        self.x_shape = [10, 1, 10]
        self.min_shape = [10]
        self.max_shape = None


class TestClipTensorCase9(TestClipTensorAPI):
    def initCase(self):
        self.dtype = 'int64'
        self.x_shape = [10, 1, 10]
        self.min_shape = [10]
        self.max_shape = None


class TestClipTensorCase10(TestClipTensorAPI):
    def initCase(self):
        self.dtype = 'float32'
        self.x_shape = [10]
        self.min_shape = [10, 1, 10]
        self.max_shape = [10]


class TestClipTensorCase11(TestClipTensorAPI):
    def initCase(self):
        self.dtype = 'float32'
        self.x_shape = [10]
        self.min_shape = [10]
        self.max_shape = [10, 1, 10]


class XPUTestClipTensorOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'clip_tensor'
        self.use_dynamic_create_class = False

    class ClipTensorOp(XPUOpTest):
        def setUp(self):
            self.python_api = paddle.clip
            self.inputs = {}
            self.init_dtype()
            self.set_xpu()
            self.op_type = "clip_tensor"
            self.place = paddle.XPUPlace(0)
            self.init_data()
            self.set_inputs()
            if self.dtype == np.uint16:
                self.outputs = {
                    'out': convert_float_to_uint16(
                        np.clip(
                            convert_uint16_to_float(self.inputs['x']),
                            convert_uint16_to_float(self.inputs['min']),
                            convert_uint16_to_float(self.inputs['max']),
                        )
                    )
                }
            else:
                self.outputs = {
                    'out': np.clip(
                        self.inputs['x'],
                        self.inputs['min'],
                        self.inputs['max'],
                    )
                }

        def set_xpu(self):
            self.__class__.use_xpu = True
            self.__class__.no_need_check_grad = False
            self.__class__.op_type = self.dtype

        def init_data(self):
            self.shape = (10, 1, 10)
            self.min_value = 0.8
            self.max_value = 0.3

        def set_inputs(self):
            self.inputs['x'] = np.random.random(self.shape).astype("float32")
            self.inputs['min'] = np.full(self.shape, self.min_value).astype(
                'float32'
            )
            self.inputs['max'] = np.full(self.shape, self.max_value).astype(
                'float32'
            )

            self.min_v = self.inputs['min']
            self.max_v = self.inputs['max']

            self.max_relative_error = 0.006
            self.inputs['x'][
                np.abs(self.inputs['x'] - self.min_v) < self.max_relative_error
            ] = 0.5
            self.inputs['x'][
                np.abs(self.inputs['x'] - self.max_v) < self.max_relative_error
            ] = 0.5
            if self.dtype == np.uint16:
                self.inputs['x'] = convert_float_to_uint16(self.inputs['x'])
                self.inputs['min'] = convert_float_to_uint16(self.inputs['min'])
                self.inputs['max'] = convert_float_to_uint16(self.inputs['max'])
            else:
                self.inputs['x'] = self.inputs['x'].astype(self.dtype)
                self.inputs['min'] = self.inputs['min'].astype(self.dtype)
                self.inputs['max'] = self.inputs['max'].astype(self.dtype)

        def init_dtype(self):
            self.dtype = self.in_type

        def test_check_output(self):
            paddle.enable_static()
            self.check_output_with_place(self.place)
            paddle.disable_static()

        def test_check_grad(self):
            if hasattr(self, "no_need_check_grad") and self.no_need_check_grad:
                return
            if core.is_compiled_with_xpu():
                paddle.enable_static()
                self.check_grad_with_place(self.place, ['x'], 'out')
                paddle.disable_static()

    class TestClipTensorOp1(ClipTensorOp):
        def init_data(self):
            self.shape = (8, 6, 8)
            self.max_value = 0.7
            self.min_value = 0.0

    class TestClipTensorOp2(ClipTensorOp):
        def init_data(self):
            self.shape = (8, 8, 6)
            self.max_value = 1.0
            self.min_value = 0.0

    class TestClipTensorOp3(ClipTensorOp):
        def init_data(self):
            self.shape = (4, 8, 6)
            self.max_value = 0.7
            self.min_value = 0.2

    class TestClipTensorOp4(ClipTensorOp):
        def init_data(self):
            self.shape = (4, 8, 6)
            self.max_value = 0.5
            self.min_value = 0.5


support_types = get_xpu_op_support_types('clip_tensor')
for stype in support_types:
    # TODO: disable int32 and int64 test temporarily, as xdnn not support corresponding resuce_mean
    if stype in ["int32", "int64"]:
        continue
    create_test_class(globals(), XPUTestClipTensorOp, stype)

if __name__ == '__main__':
    unittest.main()
