# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import OpTest, convert_float_to_uint16, convert_uint16_to_float

import paddle
from paddle import base
from paddle.autograd.ir_backward import grad
from paddle.base import Program, core, program_guard
from paddle.base.backward import append_backward


class TestWhereOp(OpTest):
    def setUp(self):
        self.op_type = 'where'
        self.prim_op_type = 'prim'
        self.python_api = paddle.where
        self.public_python_api = paddle.where
        self.check_cinn = True
        self.init_config()
        self.inputs = {'Condition': self.cond, 'X': self.x, 'Y': self.y}
        self.outputs = {'Out': np.where(self.cond, self.x, self.y)}

    def test_check_output(self):
        self.check_output(check_cinn=self.check_cinn, check_pir=True)

    def test_check_grad(self):
        self.check_grad(
            ['X', 'Y'],
            'Out',
            check_cinn=self.check_cinn,
            check_pir=True,
            check_prim_pir=True,
        )

    def init_config(self):
        self.x = np.random.uniform((-3), 5, 100).astype('float64')
        self.y = np.random.uniform((-3), 5, 100).astype('float64')
        self.cond = np.zeros(100).astype('bool')


class TestWhereOp2(TestWhereOp):
    def init_config(self):
        self.x = np.random.uniform((-5), 5, (60, 2)).astype('float64')
        self.y = np.random.uniform((-5), 5, (60, 2)).astype('float64')
        self.cond = np.ones((60, 2)).astype('bool')


class TestWhereFP16OP(TestWhereOp):
    def init_config(self):
        self.dtype = np.float16
        self.x = np.random.uniform((-5), 5, (60, 2)).astype(self.dtype)
        self.y = np.random.uniform((-5), 5, (60, 2)).astype(self.dtype)
        self.cond = np.ones((60, 2)).astype('bool')


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and not support the bfloat16",
)
class TestWhereBF16OP(OpTest):
    def setUp(self):
        self.op_type = 'where'
        self.prim_op_type = 'prim'
        self.dtype = np.uint16
        self.python_api = paddle.where
        self.public_python_api = paddle.where
        self.check_cinn = True
        self.init_config()
        self.inputs = {
            'Condition': self.cond,
            'X': convert_float_to_uint16(self.x),
            'Y': convert_float_to_uint16(self.y),
        }
        self.outputs = {
            'Out': convert_float_to_uint16(np.where(self.cond, self.x, self.y))
        }

    def test_check_output(self):
        place = core.CUDAPlace(0)
        self.check_output_with_place(
            place, check_cinn=self.check_cinn, check_pir=True
        )

    def test_check_grad(self):
        place = core.CUDAPlace(0)
        self.check_grad_with_place(
            place,
            ['X', 'Y'],
            'Out',
            numeric_grad_delta=0.05,
            check_cinn=self.check_cinn,
            check_pir=True,
            check_prim_pir=True,
        )

    def init_config(self):
        self.x = np.random.uniform((-5), 5, (60, 2)).astype(np.float32)
        self.y = np.random.uniform((-5), 5, (60, 2)).astype(np.float32)
        self.cond = np.random.randint(2, size=(60, 2)).astype('bool')


class TestWhereOp3(TestWhereOp):
    def init_config(self):
        self.x = np.random.uniform((-3), 5, (20, 2, 4)).astype('float64')
        self.y = np.random.uniform((-3), 5, (20, 2, 4)).astype('float64')
        self.cond = np.array(np.random.randint(2, size=(20, 2, 4)), dtype=bool)


class TestWhereAPI(unittest.TestCase):
    def setUp(self):
        self.init_data()

    def init_data(self):
        self.shape = [10, 15]
        self.cond = np.array(np.random.randint(2, size=self.shape), dtype=bool)
        self.x = np.random.uniform((-2), 3, self.shape).astype(np.float32)
        self.y = np.random.uniform((-2), 3, self.shape).astype(np.float32)
        self.out = np.where(self.cond, self.x, self.y)

    def ref_x_backward(self, dout):
        return np.where(self.cond, dout, 0)

    def ref_y_backward(self, dout):
        return np.where(~self.cond, dout, 0)

    def test_api(self, use_cuda=False):
        for x_stop_gradient in [False, True]:
            for y_stop_gradient in [False, True]:
                with paddle.static.program_guard(
                    paddle.static.Program(), paddle.static.Program()
                ):
                    cond = paddle.static.data(
                        name='cond', shape=[-1, *self.shape], dtype='bool'
                    )
                    if not paddle.framework.use_pir_api():
                        cond.desc.set_need_check_feed(False)
                    x = paddle.static.data(
                        name='x', shape=[-1, *self.shape], dtype='float32'
                    )
                    if not paddle.framework.use_pir_api():
                        x.desc.set_need_check_feed(False)
                    y = paddle.static.data(
                        name='y', shape=[-1, *self.shape], dtype='float32'
                    )
                    if not paddle.framework.use_pir_api():
                        y.desc.set_need_check_feed(False)
                    x.stop_gradient = x_stop_gradient
                    if not paddle.framework.use_pir_api():
                        x.desc.set_need_check_feed(False)
                    y.stop_gradient = y_stop_gradient
                    if not paddle.framework.use_pir_api():
                        y.desc.set_need_check_feed(False)
                    result = paddle.where(cond, x, y)
                    result.stop_gradient = False
                    append_backward(paddle.mean(result))
                    for use_cuda in [False, True]:
                        if use_cuda and (not base.core.is_compiled_with_cuda()):
                            break
                        place = (
                            base.CUDAPlace(0) if use_cuda else base.CPUPlace()
                        )
                        exe = base.Executor(place)
                        if paddle.framework.use_pir_api():
                            fetch_list = [result]
                            out = exe.run(
                                paddle.static.default_main_program(),
                                feed={
                                    'cond': self.cond,
                                    'x': self.x,
                                    'y': self.y,
                                },
                                fetch_list=fetch_list,
                            )
                            np.testing.assert_array_equal(out[0], self.out)
                        else:
                            fetch_list = [result, result.grad_name]
                            if x_stop_gradient is False:
                                fetch_list.append(x.grad_name)
                            if y_stop_gradient is False:
                                fetch_list.append(y.grad_name)
                            out = exe.run(
                                paddle.static.default_main_program(),
                                feed={
                                    'cond': self.cond,
                                    'x': self.x,
                                    'y': self.y,
                                },
                                fetch_list=fetch_list,
                            )
                            np.testing.assert_array_equal(out[0], self.out)
                            if x_stop_gradient is False:
                                np.testing.assert_array_equal(
                                    out[2], self.ref_x_backward(out[1])
                                )
                                if y.stop_gradient is False:
                                    np.testing.assert_array_equal(
                                        out[3], self.ref_y_backward(out[1])
                                    )
                            elif y.stop_gradient is False:
                                np.testing.assert_array_equal(
                                    out[2], self.ref_y_backward(out[1])
                                )

    def test_pir_api(self, use_cuda=False):
        for x_stop_gradient in [False, True]:
            for y_stop_gradient in [False, True]:
                with paddle.pir_utils.IrGuard(), paddle.static.program_guard(
                    paddle.static.Program(), paddle.static.Program()
                ):
                    cond = paddle.static.data(
                        name='cond', shape=self.shape, dtype='bool'
                    )
                    x = paddle.static.data(
                        name='x', shape=self.shape, dtype='float32'
                    )
                    y = paddle.static.data(
                        name='y', shape=self.shape, dtype='float32'
                    )
                    x.stop_gradient = x_stop_gradient
                    y.stop_gradient = y_stop_gradient
                    result = paddle.where(cond, x, y)
                    result.stop_gradient = False
                    loss = paddle.mean(result)
                    [x_grad, y_grad] = grad(loss, (x, y))
                    default_main_program = paddle.static.default_main_program()
                    fetch_list = [result]
                    if x_stop_gradient is False:
                        fetch_list.append(x_grad)
                    if y_stop_gradient is False:
                        fetch_list.append(y_grad)
                    for use_cuda in [False, True]:
                        if use_cuda and (not base.core.is_compiled_with_cuda()):
                            break
                        place = (
                            base.CUDAPlace(0) if use_cuda else base.CPUPlace()
                        )
                        exe = base.Executor(place)

                        out = exe.run(
                            default_main_program,
                            feed={'cond': self.cond, 'x': self.x, 'y': self.y},
                            fetch_list=fetch_list,
                        )
                        np.testing.assert_array_equal(out[0], self.out)
                        if x_stop_gradient is False:
                            np.testing.assert_array_equal(
                                out[1], self.ref_x_backward(out[1])
                            )
                            if y.stop_gradient is False:
                                np.testing.assert_array_equal(
                                    out[2], self.ref_y_backward(out[2])
                                )
                        elif y.stop_gradient is False:
                            np.testing.assert_array_equal(
                                out[1], self.ref_y_backward(out[1])
                            )

    def test_api_broadcast(self, use_cuda=False):
        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program):
            x = paddle.static.data(name='x', shape=[-1, 4, 1], dtype='float32')
            y = paddle.static.data(name='y', shape=[-1, 4, 2], dtype='float32')
            x_i = (
                np.array([[0.9383, 0.1983, 3.2, 1.2]])
                .astype('float32')
                .reshape([1, 4, 1])
            )
            y_i = (
                np.array([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
                .astype('float32')
                .reshape([1, 4, 2])
            )
            result = paddle.where((x > 1), x=x, y=y)
            for use_cuda in [False, True]:
                if use_cuda and (not base.core.is_compiled_with_cuda()):
                    return
                place = base.CUDAPlace(0) if use_cuda else base.CPUPlace()
                exe = base.Executor(place)
                out = exe.run(
                    paddle.static.default_main_program(),
                    feed={'x': x_i, 'y': y_i},
                    fetch_list=[result],
                )
                np.testing.assert_array_equal(
                    out[0], np.where((x_i > 1), x_i, y_i)
                )

    def test_scalar(self):
        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program):
            cond_shape = [4]
            cond = paddle.static.data(
                name='cond', shape=cond_shape, dtype='bool'
            )
            x_data = 1.0
            y_data = 2.0
            cond_data = np.array([False, False, True, True]).astype('bool')
            result = paddle.where(condition=cond, x=x_data, y=y_data)
            for use_cuda in [False, True]:
                if use_cuda and (not base.core.is_compiled_with_cuda()):
                    return
                place = base.CUDAPlace(0) if use_cuda else base.CPUPlace()
                exe = base.Executor(place)
                out = exe.run(
                    paddle.static.default_main_program(),
                    feed={'cond': cond_data},
                    fetch_list=[result],
                )
                expect = np.where(cond_data, x_data, y_data)
                np.testing.assert_array_equal(out[0], expect)

    def __test_where_with_broadcast_static(self, cond_shape, x_shape, y_shape):
        paddle.enable_static()
        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program):
            cond = paddle.static.data(
                name='cond', shape=cond_shape, dtype='bool'
            )
            x = paddle.static.data(name='x', shape=x_shape, dtype='float32')
            y = paddle.static.data(name='y', shape=y_shape, dtype='float32')
            cond_data_tmp = np.random.random(size=cond_shape).astype('float32')
            cond_data = cond_data_tmp < 0.3
            x_data = np.random.random(size=x_shape).astype('float32')
            y_data = np.random.random(size=y_shape).astype('float32')
            result = paddle.where(condition=cond, x=x, y=y)
            for use_cuda in [False, True]:
                if use_cuda and (not base.core.is_compiled_with_cuda()):
                    return
                place = base.CUDAPlace(0) if use_cuda else base.CPUPlace()
                exe = base.Executor(place)
                out = exe.run(
                    paddle.static.default_main_program(),
                    feed={'cond': cond_data, 'x': x_data, 'y': y_data},
                    fetch_list=[result],
                )
                expect = np.where(cond_data, x_data, y_data)
                np.testing.assert_array_equal(out[0], expect)

    def __test_where_with_type_promotion(
        self, x_dtype, y_dtype, expected_dtype=None
    ):
        paddle.enable_static()
        main_program = paddle.static.Program()
        shape = [3, 10]
        with paddle.static.program_guard(main_program):
            cond = paddle.static.data(name='cond', shape=[3, 10], dtype='bool')
            x = paddle.static.data(name='x', shape=shape, dtype=x_dtype)
            y = paddle.static.data(name='y', shape=shape, dtype=y_dtype)
            cond_data_tmp = np.random.random(size=shape).astype('float32')
            cond_data = cond_data_tmp < 0.3

            if x_dtype != 'bfloat16':
                x_data = np.random.random(size=shape).astype(x_dtype)
            else:
                x_data = convert_float_to_uint16(
                    np.random.random(size=shape).astype('float32')
                )
            if y_dtype != 'bfloat16':
                y_data = np.random.random(size=shape).astype(y_dtype)
            else:
                y_data = convert_float_to_uint16(
                    np.random.random(size=shape).astype('float32')
                )
            result = paddle.where(condition=cond, x=x, y=y)
            for use_cuda in [False, True]:
                if use_cuda and (not base.core.is_compiled_with_cuda()):
                    return
                place = base.CUDAPlace(0) if use_cuda else base.CPUPlace()
                exe = base.Executor(place)
                out = exe.run(
                    paddle.static.default_main_program(),
                    feed={'cond': cond_data, 'x': x_data, 'y': y_data},
                    fetch_list=[result],
                )
                if x_dtype == 'bfloat16' or y_dtype == 'bfloat16':
                    x_data_convert = (
                        convert_uint16_to_float(x_data)
                        if x_dtype == 'bfloat16'
                        else x_data
                    )
                    y_data_convert = (
                        convert_uint16_to_float(y_data)
                        if y_dtype == 'bfloat16'
                        else y_data
                    )
                    expect = np.where(cond_data, x_data_convert, y_data_convert)
                    np.testing.assert_array_equal(out[0], expect)
                    self.assertEqual(out[0].dtype.__str__(), expected_dtype)
                else:
                    expect = np.where(cond_data, x_data, y_data)
                    np.testing.assert_array_equal(out[0], expect)
                    self.assertEqual(out[0].dtype, expect.dtype)

    def test_static_api_broadcast_1(self):
        cond_shape = [2, 4]
        a_shape = [2, 2, 4]
        b_shape = [2, 2, 4]
        self.__test_where_with_broadcast_static(cond_shape, a_shape, b_shape)

    def test_static_api_broadcast_2(self):
        cond_shape = [2, 1]
        a_shape = [2, 2, 4]
        b_shape = [2, 2, 4]
        self.__test_where_with_broadcast_static(cond_shape, a_shape, b_shape)

    def test_static_api_broadcast_3(self):
        cond_shape = [2, 2, 1]
        a_shape = [2, 2, 4]
        b_shape = [2, 2, 4]
        self.__test_where_with_broadcast_static(cond_shape, a_shape, b_shape)

    def test_static_api_broadcast_4(self):
        cond_shape = [2, 1, 4]
        a_shape = [2, 2, 4]
        b_shape = [2, 2, 4]
        self.__test_where_with_broadcast_static(cond_shape, a_shape, b_shape)

    def test_static_api_broadcast_5(self):
        cond_shape = [3, 2, 2, 4]
        a_shape = [2, 2, 4]
        b_shape = [2, 2, 4]
        self.__test_where_with_broadcast_static(cond_shape, a_shape, b_shape)

    def test_static_api_broadcast_6(self):
        cond_shape = [2, 2, 4]
        a_shape = [2, 2, 1]
        b_shape = [2, 2, 1]
        self.__test_where_with_broadcast_static(cond_shape, a_shape, b_shape)

    def test_static_api_broadcast_7(self):
        cond_shape = [2, 2, 4]
        a_shape = [2, 1, 4]
        b_shape = [2, 1, 4]
        self.__test_where_with_broadcast_static(cond_shape, a_shape, b_shape)

    def test_static_api_broadcast_8(self):
        cond_shape = [3, 2, 2, 4]
        a_shape = [2, 2, 1]
        b_shape = [2, 2, 1]
        self.__test_where_with_broadcast_static(cond_shape, a_shape, b_shape)

    def test_static_api_type_promotion_fp16_fp32(self):
        x_dtype = 'float16'
        y_dtype = 'float32'
        self.__test_where_with_type_promotion(x_dtype, y_dtype)
        self.__test_where_with_type_promotion(y_dtype, x_dtype)

    def test_static_api_type_promotion_fp16_fp64(self):
        x_dtype = 'float16'
        y_dtype = 'float64'
        self.__test_where_with_type_promotion(x_dtype, y_dtype)
        self.__test_where_with_type_promotion(y_dtype, x_dtype)

    def test_static_api_type_promotion_fp32_fp64(self):
        x_dtype = 'float32'
        y_dtype = 'float64'
        self.__test_where_with_type_promotion(x_dtype, y_dtype)
        self.__test_where_with_type_promotion(y_dtype, x_dtype)

    @unittest.skipIf(
        not (
            paddle.is_compiled_with_cuda()
            and paddle.base.core.supports_bfloat16()
        ),
        "bf16 is not supported in current device",
    )
    def test_static_api_type_promotion_bf16_fp16(self):
        x_dtype = 'bfloat16'
        y_dtype = 'float16'
        self.__test_where_with_type_promotion(x_dtype, y_dtype, 'float32')
        self.__test_where_with_type_promotion(y_dtype, x_dtype, 'float32')

    @unittest.skipIf(
        not (
            paddle.is_compiled_with_cuda()
            and paddle.base.core.supports_bfloat16()
        ),
        "bf16 is not supported in current device",
    )
    def test_static_api_type_promotion_bf16_fp32(self):
        x_dtype = 'bfloat16'
        y_dtype = 'float32'
        self.__test_where_with_type_promotion(x_dtype, y_dtype, 'float32')
        self.__test_where_with_type_promotion(y_dtype, x_dtype, 'float32')

    @unittest.skipIf(
        not (
            paddle.is_compiled_with_cuda()
            and paddle.base.core.supports_bfloat16()
        ),
        "bf16 is not supported in current device",
    )
    def test_static_api_type_promotion_bf16_fp64(self):
        x_dtype = 'bfloat16'
        y_dtype = 'float64'
        self.__test_where_with_type_promotion(x_dtype, y_dtype, 'float64')
        self.__test_where_with_type_promotion(y_dtype, x_dtype, 'float64')


class TestWhereDygraphAPI(unittest.TestCase):
    def test_api(self):
        with base.dygraph.guard():
            x_i = np.array([0.9383, 0.1983, 3.2, 1.2]).astype('float64')
            y_i = np.array([1.0, 1.0, 1.0, 1.0]).astype('float64')
            cond_i = np.array([False, False, True, True]).astype('bool')
            x = paddle.to_tensor(x_i)
            y = paddle.to_tensor(y_i)
            cond = paddle.to_tensor(cond_i)
            out = paddle.where(cond, x, y)
            np.testing.assert_array_equal(
                out.numpy(), np.where(cond_i, x_i, y_i)
            )

    def test_scalar(self):
        with base.dygraph.guard():
            cond_i = np.array([False, False, True, True]).astype('bool')
            x = 1.0
            y = 2.0
            cond = paddle.to_tensor(cond_i)
            out = paddle.where(cond, x, y)
            np.testing.assert_array_equal(out.numpy(), np.where(cond_i, x, y))

    def __test_where_with_broadcast_dygraph(self, cond_shape, a_shape, b_shape):
        with base.dygraph.guard():
            cond_tmp = paddle.rand(cond_shape)
            cond = cond_tmp < 0.3
            a = paddle.rand(a_shape)
            b = paddle.rand(b_shape)
            result = paddle.where(cond, a, b)
            result = result.numpy()
            expect = np.where(cond, a, b)
            np.testing.assert_array_equal(expect, result)

    def test_dygraph_api_broadcast_1(self):
        cond_shape = [2, 4]
        a_shape = [2, 2, 4]
        b_shape = [2, 2, 4]
        self.__test_where_with_broadcast_dygraph(cond_shape, a_shape, b_shape)

    def test_dygraph_api_broadcast_2(self):
        cond_shape = [2, 1]
        a_shape = [2, 2, 4]
        b_shape = [2, 2, 4]
        self.__test_where_with_broadcast_dygraph(cond_shape, a_shape, b_shape)

    def test_dygraph_api_broadcast_3(self):
        cond_shape = [2, 2, 1]
        a_shape = [2, 2, 4]
        b_shape = [2, 2, 4]
        self.__test_where_with_broadcast_dygraph(cond_shape, a_shape, b_shape)

    def test_dygraph_api_broadcast_4(self):
        cond_shape = [2, 1, 4]
        a_shape = [2, 2, 4]
        b_shape = [2, 2, 4]
        self.__test_where_with_broadcast_dygraph(cond_shape, a_shape, b_shape)

    def test_dygraph_api_broadcast_5(self):
        cond_shape = [3, 2, 2, 4]
        a_shape = [2, 2, 4]
        b_shape = [2, 2, 4]
        self.__test_where_with_broadcast_dygraph(cond_shape, a_shape, b_shape)

    def test_dygraph_api_broadcast_6(self):
        cond_shape = [2, 2, 4]
        a_shape = [2, 2, 1]
        b_shape = [2, 2, 1]
        self.__test_where_with_broadcast_dygraph(cond_shape, a_shape, b_shape)

    def test_dygraph_api_broadcast_7(self):
        cond_shape = [2, 2, 4]
        a_shape = [2, 1, 4]
        b_shape = [2, 1, 4]
        self.__test_where_with_broadcast_dygraph(cond_shape, a_shape, b_shape)

    def test_dygraph_api_broadcast_8(self):
        cond_shape = [3, 2, 2, 4]
        a_shape = [2, 2, 1]
        b_shape = [2, 2, 1]
        self.__test_where_with_broadcast_dygraph(cond_shape, a_shape, b_shape)

    def test_where_type_promotion_f2_f4(self):
        with base.dygraph.guard():
            cond = np.array([False, False, True, True]).astype('bool')
            x = np.array([0.9383, 0.1983, 3.2, 1.2]).astype('float16')
            y = np.array([1.0, 1.0, 1.0, 1.0]).astype('float32')
            np.testing.assert_equal(
                paddle.float32,
                paddle.where(
                    paddle.to_tensor(cond),
                    paddle.to_tensor(x),
                    paddle.to_tensor(y),
                ).dtype,
            )
            np.testing.assert_array_equal(
                np.where(cond, x, y),
                paddle.where(
                    paddle.to_tensor(cond),
                    paddle.to_tensor(x),
                    paddle.to_tensor(y),
                ).numpy(),
            )
            np.testing.assert_array_equal(
                np.where(cond, y, x),
                paddle.where(
                    paddle.to_tensor(cond),
                    paddle.to_tensor(y),
                    paddle.to_tensor(x),
                ).numpy(),
            )

    def test_where_type_promotion_f2_f8(self):
        with base.dygraph.guard():
            cond = np.array([False, False, True, True]).astype('bool')
            x = np.array([0.9383, 0.1983, 3.2, 1.2]).astype('float16')
            y = np.array([1.0, 1.0, 1.0, 1.0]).astype('float64')
            np.testing.assert_equal(
                paddle.float64,
                paddle.where(
                    paddle.to_tensor(cond),
                    paddle.to_tensor(x),
                    paddle.to_tensor(y),
                ).dtype,
            )
            np.testing.assert_array_equal(
                np.where(cond, x, y),
                paddle.where(
                    paddle.to_tensor(cond),
                    paddle.to_tensor(x),
                    paddle.to_tensor(y),
                ).numpy(),
            )
            np.testing.assert_array_equal(
                np.where(cond, y, x),
                paddle.where(
                    paddle.to_tensor(cond),
                    paddle.to_tensor(y),
                    paddle.to_tensor(x),
                ).numpy(),
            )

    def test_where_type_promotion_f4_f8(self):
        with base.dygraph.guard():
            cond = np.array([False, False, True, True]).astype('bool')
            x = np.array([0.9383, 0.1983, 3.2, 1.2]).astype('float32')
            y = np.array([1.0, 1.0, 1.0, 1.0]).astype('float64')
            np.testing.assert_equal(
                paddle.float64,
                paddle.where(
                    paddle.to_tensor(cond),
                    paddle.to_tensor(x),
                    paddle.to_tensor(y),
                ).dtype,
            )
            np.testing.assert_array_equal(
                np.where(cond, x, y),
                paddle.where(
                    paddle.to_tensor(cond),
                    paddle.to_tensor(x),
                    paddle.to_tensor(y),
                ).numpy(),
            )
            np.testing.assert_array_equal(
                np.where(cond, y, x),
                paddle.where(
                    paddle.to_tensor(cond),
                    paddle.to_tensor(y),
                    paddle.to_tensor(x),
                ).numpy(),
            )

    def test_where_type_promotion_f2_bf(self):
        with base.dygraph.guard():
            cond = np.array([False, False, True, True]).astype('bool')
            x = np.array([0.9383, 0.1983, 3.2, 1.2]).astype('float16')
            y = np.array([1.0, 1.0, 1.0, 1.0]).astype('float32')
            y = convert_uint16_to_float(convert_float_to_uint16(y))
            np.testing.assert_equal(
                paddle.float32,
                paddle.where(
                    paddle.to_tensor(cond),
                    paddle.to_tensor(x),
                    paddle.to_tensor(y),
                ).dtype,
            )
            np.testing.assert_array_equal(
                np.where(cond, x, y),
                paddle.where(
                    paddle.to_tensor(cond),
                    paddle.to_tensor(x),
                    paddle.to_tensor(y),
                ).numpy(),
            )
            np.testing.assert_array_equal(
                np.where(cond, y, x),
                paddle.where(
                    paddle.to_tensor(cond),
                    paddle.to_tensor(y),
                    paddle.to_tensor(x),
                ).numpy(),
            )

    def test_where_type_promotion_f4_bf(self):
        with base.dygraph.guard():
            cond = np.array([False, False, True, True]).astype('bool')
            x = np.array([0.9383, 0.1983, 3.2, 1.2]).astype('float32')
            y = np.array([1.0, 1.0, 1.0, 1.0]).astype('float32')
            y = convert_uint16_to_float(convert_float_to_uint16(y))
            np.testing.assert_equal(
                paddle.float32,
                paddle.where(
                    paddle.to_tensor(cond),
                    paddle.to_tensor(x),
                    paddle.to_tensor(y),
                ).dtype,
            )
            np.testing.assert_array_equal(
                np.where(cond, x, y),
                paddle.where(
                    paddle.to_tensor(cond),
                    paddle.to_tensor(x),
                    paddle.to_tensor(y),
                ).numpy(),
            )
            np.testing.assert_array_equal(
                np.where(cond, y, x),
                paddle.where(
                    paddle.to_tensor(cond),
                    paddle.to_tensor(y),
                    paddle.to_tensor(x),
                ),
            )

    def test_where_type_promotion_f8_bf(self):
        with base.dygraph.guard():
            cond = np.array([False, False, True, True]).astype('bool')
            x = np.array([0.9383, 0.1983, 3.2, 1.2]).astype('float64')
            y = np.array([1.0, 1.0, 1.0, 1.0]).astype('float32')
            y = convert_uint16_to_float(convert_float_to_uint16(y))
            np.testing.assert_equal(
                paddle.float64,
                paddle.where(
                    paddle.to_tensor(cond),
                    paddle.to_tensor(x),
                    paddle.to_tensor(y),
                ).dtype,
            )
            np.testing.assert_array_equal(
                np.where(cond, x, y),
                paddle.where(
                    paddle.to_tensor(cond),
                    paddle.to_tensor(x),
                    paddle.to_tensor(y),
                ).numpy(),
            )
            np.testing.assert_array_equal(
                np.where(cond, y, x),
                paddle.where(
                    paddle.to_tensor(cond),
                    paddle.to_tensor(y),
                    paddle.to_tensor(x),
                ).numpy(),
            )

    def test_where_condition(self):
        data = np.array([[True, False], [False, True]])
        with program_guard(Program(), Program()):
            x = paddle.static.data(name='x', shape=[(-1), 2], dtype='bool')
            if not paddle.framework.use_pir_api():
                x.desc.set_need_check_feed(False)
            y = paddle.where(x)
            self.assertEqual(type(y), tuple)
            self.assertEqual(len(y), 2)
            index_list = [index.reshape([-1, 1]) for index in y]
            z = paddle.concat(index_list, axis=1)
            exe = base.Executor(base.CPUPlace())
            (res,) = exe.run(
                feed={'x': data}, fetch_list=[z], return_numpy=False
            )
        expect_out = np.array([[0, 0], [1, 1]])
        np.testing.assert_allclose(expect_out, np.array(res), rtol=1e-05)
        data = np.array([True, True, False])
        with program_guard(Program(), Program()):
            x = paddle.static.data(name='x', shape=[(-1)], dtype='bool')
            if not paddle.framework.use_pir_api():
                x.desc.set_need_check_feed(False)
            y = paddle.where(x)
            self.assertEqual(type(y), tuple)
            self.assertEqual(len(y), 1)
            index_list = [index.reshape([-1, 1]) for index in y]
            z = paddle.concat(index_list, axis=1)
            exe = base.Executor(base.CPUPlace())
            (res,) = exe.run(
                feed={'x': data}, fetch_list=[z], return_numpy=False
            )
        expect_out = np.array([[0], [1]])
        np.testing.assert_allclose(expect_out, np.array(res), rtol=1e-05)


class TestWhereDygraphAPIBroadcast(unittest.TestCase):
    def test_broadcast_scalar(self):
        with base.dygraph.guard():
            x_i = np.random.randn(4, 5, 6).astype('float64')
            y_i = -1.0
            cond_i = np.random.randn(1, 1, 6).astype('bool')
            x = paddle.to_tensor(x_i)
            y = paddle.to_tensor(y_i)
            cond = paddle.to_tensor(cond_i)
            out = paddle.where(cond, x, y)
            np.testing.assert_array_equal(
                out.numpy(), np.where(cond_i, x_i, y_i)
            )

    def test_broadcast_to_x(self):
        with base.dygraph.guard():
            x_i = np.random.randn(4, 5, 6).astype('float64')
            y_i = np.random.randn(1, 5, 6).astype('float64')
            cond_i = np.random.randn(1, 1, 6).astype('bool')
            x = paddle.to_tensor(x_i)
            y = paddle.to_tensor(y_i)
            cond = paddle.to_tensor(cond_i)
            out = paddle.where(cond, x, y)
            np.testing.assert_array_equal(
                out.numpy(), np.where(cond_i, x_i, y_i)
            )

    def test_broadcast_to_y(self):
        with base.dygraph.guard():
            x_i = np.random.randn(1, 5, 6).astype('float64')
            y_i = np.random.randn(4, 5, 6).astype('float64')
            cond_i = np.random.randn(1, 1, 6).astype('bool')
            x = paddle.to_tensor(x_i)
            y = paddle.to_tensor(y_i)
            cond = paddle.to_tensor(cond_i)
            out = paddle.where(cond, x, y)
            np.testing.assert_array_equal(
                out.numpy(), np.where(cond_i, x_i, y_i)
            )

    def test_broadcast_to_cond(self):
        with base.dygraph.guard():
            x_i = np.random.randn(1, 1, 6).astype('float64')
            y_i = np.random.randn(1, 5, 1).astype('float64')
            cond_i = np.random.randn(4, 5, 6).astype('bool')
            x = paddle.to_tensor(x_i)
            y = paddle.to_tensor(y_i)
            cond = paddle.to_tensor(cond_i)
            out = paddle.where(cond, x, y)
            np.testing.assert_array_equal(
                out.numpy(), np.where(cond_i, x_i, y_i)
            )

    def test_can_not_broadcast(self):
        with base.dygraph.guard():
            x_i = np.random.randn(1, 1, 6).astype('float64')
            y_i = np.random.randn(1, 5, 3).astype('float64')
            cond_i = np.random.randn(4, 5, 6).astype('bool')
            x = paddle.to_tensor(x_i)
            y = paddle.to_tensor(y_i)
            cond = paddle.to_tensor(cond_i)

            with self.assertRaises(ValueError):
                _ = paddle.where(cond, x, y)


class TestWhereDygraphAPIDtypePromotion(unittest.TestCase):
    def test_dtype_auto_promotion_float(self):
        with base.dygraph.guard():
            x_i = np.random.randn(4, 5, 6).astype('float32')
            y_i = np.random.randn(4, 5, 6).astype('float64')
            cond_i = np.random.randn(4, 5, 6).astype('bool')
            x = paddle.to_tensor(x_i)
            y = paddle.to_tensor(y_i)
            cond = paddle.to_tensor(cond_i)
            out = paddle.where(cond, x, y)
            self.assertEqual(out.dtype, y.dtype)
            np.testing.assert_array_equal(
                out.numpy(), np.where(cond_i, x_i, y_i)
            )


class TestWhereOpError(unittest.TestCase):
    def test_errors(self):
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            x_i = np.array([0.9383, 0.1983, 3.2, 1.2]).astype('float64')
            y_i = np.array([1.0, 1.0, 1.0, 1.0]).astype('float64')
            cond_i = np.array([False, False, True, True]).astype('bool')

            def test_Variable():
                paddle.where(cond_i, x_i, y_i)

            self.assertRaises(TypeError, test_Variable)

            def test_Value():
                with paddle.pir_utils.IrGuard():
                    paddle.where(cond_i, x_i, y_i)

            self.assertRaises(TypeError, test_Value)

            def test_type():
                with paddle.pir_utils.OldIrGuard():
                    x = paddle.static.data(
                        name='x', shape=[-1, 4], dtype='bool'
                    )
                    if not paddle.framework.use_pir_api():
                        x.desc.set_need_check_feed(False)
                    y = paddle.static.data(
                        name='y', shape=[-1, 4], dtype='float16'
                    )
                    if not paddle.framework.use_pir_api():
                        y.desc.set_need_check_feed(False)
                    cond = paddle.static.data(
                        name='cond', shape=[-1, 4], dtype='int32'
                    )
                    if not paddle.framework.use_pir_api():
                        cond.desc.set_need_check_feed(False)
                    paddle.where(cond, x, y)

            self.assertRaises(TypeError, test_type)

    def test_value_error(self):
        with base.dygraph.guard():
            cond_shape = [2, 2, 4]
            cond_tmp = paddle.rand(cond_shape)
            cond = cond_tmp < 0.3
            a = paddle.rand(cond_shape)
            self.assertRaises(ValueError, paddle.where, cond, a)


class TestWhereDygraphAPINonBoolCondition(unittest.TestCase):
    def test_condition_with_wrong_dtype(self):
        with base.dygraph.guard():
            cond = paddle.to_tensor([True, False])

            for dtype in [
                paddle.int64,
                paddle.int32,
                paddle.float32,
                paddle.float64,
            ]:
                cond_wrong_dtype = cond.to(dtype)
                with self.assertRaises(ValueError):
                    paddle.where(cond_wrong_dtype, 1, 0)

    def test_condition_inplace_with_wrong_dtype(self):
        with base.dygraph.guard():
            cond = paddle.to_tensor([True, False])

            x = paddle.zeros_like(cond).astype("float32")
            for dtype in [
                paddle.int64,
                paddle.int32,
                paddle.float32,
                paddle.float64,
            ]:
                cond_wrong_dtype = cond.to(dtype)
                with self.assertRaises(ValueError):
                    x = x.where_(cond_wrong_dtype, x, x)


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
