#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import os
import unittest

import gradient_checker
import numpy as np
from decorator_helper import prog_scope
from op_test import OpTest, OpTestTool, convert_float_to_uint16
from test_sum_op import TestReduceOPTensorAxisBase

import paddle
from paddle import base
from paddle.base import Program, core, program_guard

np.random.seed(10)


def mean_wrapper(x, axis=None, keepdim=False, reduce_all=False):
    if reduce_all:
        return paddle.mean(x, list(range(len(x.shape))), keepdim)
    return paddle.mean(x, axis, keepdim)


def reduce_mean_wrapper(x, axis=0, keepdim=False, reduce_all=False):
    if reduce_all:
        return paddle.mean(x, list(range(len(x.shape))), keepdim)
    return paddle.mean(x, axis, keepdim)


class TestMeanOp(OpTest):
    def setUp(self):
        self.op_type = "mean"
        self.python_api = paddle.mean
        self.public_python_api = paddle.mean
        self.dtype = np.float64
        self.init_dtype_type()
        self.init_prim_type()
        self.inputs = {'X': np.random.random((10, 10)).astype(self.dtype)}
        self.outputs = {'Out': np.mean(self.inputs["X"])}

    def init_prim_type(self):
        self.prim_op_type = "comp"

    def init_dtype_type(self):
        pass

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_checkout_grad(self):
        self.check_grad(['X'], 'Out', check_pir=True, check_prim_pir=True)


class TestMeanAllOp(OpTest):
    def setUp(self):
        self.op_type = "mean_all"
        self.python_api = paddle.mean_all
        self.public_python_api = paddle.mean_all
        self.dtype = np.float64
        self.init_dtype_type()
        self.init_prim_type()
        self.inputs = {'X': np.random.random((10, 10)).astype(self.dtype)}
        self.outputs = {'Out': np.mean(self.inputs["X"])}

    def init_prim_type(self):
        self.prim_op_type = "comp"

    def init_dtype_type(self):
        pass

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_checkout_grad(self):
        self.check_grad(['X'], 'Out', check_pir=True, check_prim_pir=True)


class TestMeanOpPrim(TestMeanOp):
    def init_prim_type(self):
        self.prim_op_type = "prim"


class TestMeanOp_ZeroDim(OpTest):
    def setUp(self):
        self.op_type = "mean"
        self.python_api = paddle.mean
        self.dtype = np.float64
        self.public_python_api = paddle.mean
        self.init_prim_type()
        self.inputs = {'X': np.random.random([]).astype(self.dtype)}
        self.outputs = {'Out': np.mean(self.inputs["X"])}

    def init_prim_type(self):
        self.prim_op_type = "comp"

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_checkout_grad(self):
        self.check_grad(['X'], 'Out', check_pir=True, check_prim_pir=True)


class TestMeanAllOp_ZeroDim(OpTest):
    def setUp(self):
        self.op_type = "mean_all"
        self.python_api = paddle.mean_all
        self.dtype = np.float64
        self.public_python_api = paddle.mean_all
        self.init_prim_type()
        self.inputs = {'X': np.random.random([]).astype(self.dtype)}
        self.outputs = {'Out': np.mean(self.inputs["X"])}

    def init_prim_type(self):
        self.prim_op_type = "comp"

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_checkout_grad(self):
        self.check_grad(['X'], 'Out', check_pir=True, check_prim_pir=True)


class TestMeanOp_ZeroDim_Prim(TestMeanOp_ZeroDim):
    def init_prim_type(self):
        self.prim_op_type = "prim"


class TestMeanOpError(unittest.TestCase):
    def setUp(self):
        self.x_shape = [2, 3, 4, 5]
        self.x = np.random.uniform(-1, 1, self.x_shape).astype(np.int32)
        self.place = (
            paddle.CUDAPlace(0)
            if core.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    def test_errors(self):
        paddle.enable_static()
        with program_guard(Program(), Program()):
            # The input type of mean_op must be Variable.
            input1 = 12
            self.assertRaises(TypeError, paddle.mean, input1)

            if paddle.is_compiled_with_cuda():
                input3 = paddle.static.data(
                    name='input3', shape=[-1, 4], dtype="float16"
                )
                paddle.nn.functional.softmax(input3)

        paddle.disable_static()


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestFP16MeanOp(TestMeanOp):
    def init_dtype_type(self):
        self.dtype = np.float16
        self.__class__.no_need_check_grad = True

    def test_check_output(self):
        place = core.CUDAPlace(0)
        if core.is_float16_supported(place):
            self.check_output_with_place(place, check_pir=True)

    def test_checkout_grad(self):
        place = core.CUDAPlace(0)
        if core.is_float16_supported(place):
            with base.dygraph.guard():
                x_np = np.random.random((10, 10)).astype(self.dtype)
                x = paddle.to_tensor(x_np)
                x.stop_gradient = False
                y = paddle.mean(x)
                dx = paddle.grad(y, x)[0].numpy()
                dx_expected = self.dtype(1.0 / np.prod(x_np.shape)) * np.ones(
                    x_np.shape
                ).astype(self.dtype)
                np.testing.assert_array_equal(dx, dx_expected)


@OpTestTool.skip_if_not_cpu_bf16()
class TestBF16MeanOp(TestMeanOp):
    def init_dtype_type(self):
        self.dtype = np.uint16

    def test_check_output(self):
        paddle.enable_static()
        self.check_output_with_place(core.CPUPlace(), check_pir=True)

    def test_checkout_grad(self):
        place = core.CPUPlace()
        self.check_grad_with_place(place, ['X'], 'Out', check_pir=True)


def ref_reduce_mean(x, axis=None, keepdim=False, reduce_all=False):
    if isinstance(axis, list):
        axis = tuple(axis)
    if reduce_all:
        axis = None
    return np.mean(x, axis=axis, keepdims=keepdim)


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_float16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA",
)
class TestReduceMeanOp(OpTest):
    def setUp(self):
        self.op_type = 'reduce_mean'
        self.python_api = reduce_mean_wrapper
        self.public_python_api = reduce_mean_wrapper
        self.init_prim_type()
        self.dtype = 'float64'
        self.init_shapes()
        self.axis = [0]
        if self.shape == []:
            self.axis = []
        self.keepdim = False
        self.set_attrs()
        self.if_enable_cinn()

        np.random.seed(10)
        x_np = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        if not hasattr(self, "reduce_all") and not x_np.shape == ():
            self.reduce_all = (not self.axis) or len(self.axis) == len(x_np)
        if x_np.shape == ():
            self.reduce_all = True
        out_np = ref_reduce_mean(x_np, self.axis, self.keepdim, self.reduce_all)
        self.inputs = {'X': x_np}
        self.outputs = {'Out': out_np}
        self.attrs = {
            'dim': self.axis,
            'keep_dim': self.keepdim,
            'reduce_all': self.reduce_all,
        }

    def init_prim_type(self):
        self.prim_op_type = "comp"

    def init_shapes(self):
        self.shape = [2, 3, 4, 5]

    def set_attrs(self):
        pass

    def if_enable_cinn(self):
        pass

    def test_check_output(self):
        if self.dtype != 'float16':
            self.check_output(
                check_prim=True, check_prim_pir=True, check_pir=True
            )
        else:
            place = paddle.CUDAPlace(0)
            self.check_output_with_place(
                place=place,
                check_prim=True,
                check_prim_pir=True,
                check_pir=True,
            )

    def test_check_grad(self):
        if self.dtype != 'float16':
            self.check_grad(
                ['X'],
                ['Out'],
                check_prim=True,
                check_prim_pir=True,
                check_pir=True,
            )
        else:
            place = paddle.CUDAPlace(0)
            self.check_grad_with_place(
                place,
                ['X'],
                ['Out'],
                numeric_grad_delta=0.5,
                check_prim=True,
                check_prim_pir=True,
                check_pir=True,
            )


class TestReduceMeanOpPrim(TestReduceMeanOp):
    def init_prim_type(self):
        self.prim_op_type = "prim"

    def test_check_output(self):
        if self.dtype != 'float16':
            self.check_output(check_prim_pir=True, check_pir=True)
        else:
            place = paddle.CUDAPlace(0)
            self.check_output_with_place(
                place=place,
                check_prim_pir=True,
                check_pir=True,
            )

    def test_check_grad(self):
        if self.dtype != 'float16':
            self.check_grad(
                ['X'],
                ['Out'],
                check_prim_pir=True,
                check_pir=True,
            )
        else:
            place = paddle.CUDAPlace(0)
            self.check_grad_with_place(
                place,
                ['X'],
                ['Out'],
                numeric_grad_delta=0.5,
                check_prim_pir=True,
                check_pir=True,
            )


class TestReduceMeanOp_ZeroDim(TestReduceMeanOp):
    def init_shapes(self):
        self.shape = []
        self.enable_cinn = False


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and do not support bfloat16",
)
class TestReduceMeanBF16Op(OpTest):
    def setUp(self):
        self.op_type = 'reduce_mean'
        self.python_api = reduce_mean_wrapper
        self.public_python_api = reduce_mean_wrapper
        self.prim_op_type = "comp"
        self.dtype = np.uint16
        self.shape = [2, 3, 4, 5]
        self.axis = [0]
        self.keepdim = False
        self.set_attrs()
        self.if_enable_cinn()

        np.random.seed(10)
        x_np = np.random.uniform(-1, 1, self.shape).astype(np.float32)
        if not hasattr(self, "reduce_all"):
            self.reduce_all = (not self.axis) or len(self.axis) == len(x_np)

        out_np = ref_reduce_mean(x_np, self.axis, self.keepdim, self.reduce_all)
        self.inputs = {'X': convert_float_to_uint16(x_np)}
        self.outputs = {'Out': convert_float_to_uint16(out_np)}
        self.attrs = {
            'dim': self.axis,
            'keep_dim': self.keepdim,
            'reduce_all': self.reduce_all,
        }

    def if_enable_cinn(self):
        self.enable_cinn = False

    def set_attrs(self):
        pass

    def test_check_output(self):
        place = paddle.CUDAPlace(0)
        self.check_output_with_place(place, check_prim=True)

    def test_check_grad(self):
        place = paddle.CUDAPlace(0)
        self.check_grad_with_place(
            place,
            ['X'],
            ['Out'],
            numeric_grad_delta=0.05,
            check_prim=True,
            check_prim_pir=True,
        )


class TestReduceMeanOpDefaultAttrs(TestReduceMeanOp):
    def setUp(self):
        self.op_type = 'reduce_mean'
        self.python_api = reduce_mean_wrapper
        self.public_python_api = reduce_mean_wrapper
        self.prim_op_type = "comp"
        self.dtype = 'float64'
        self.shape = [2, 3, 4, 5]

        x_np = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        out_np = np.mean(x_np, axis=0)
        self.inputs = {'X': x_np}
        self.outputs = {'Out': out_np}


class TestReduceMeanOpDefaultAttrsForPrim(TestReduceMeanOpPrim):
    def setUp(self):
        self.op_type = 'reduce_mean'
        self.python_api = reduce_mean_wrapper
        self.public_python_api = reduce_mean_wrapper
        self.init_prim_type()
        self.dtype = 'float64'
        self.shape = [2, 3, 4, 5]

        x_np = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        out_np = np.mean(x_np, axis=0)
        self.inputs = {'X': x_np}
        self.outputs = {'Out': out_np}


class TestReduceMeanOpFloat32(TestReduceMeanOp):
    def set_attrs(self):
        self.dtype = 'float32'


class TestReduceMeanOpFloat32Prim(TestReduceMeanOpPrim):
    def set_attrs(self):
        self.dtype = 'float32'


class TestReduceMeanOpFloat16(TestReduceMeanOp):
    def set_attrs(self):
        self.dtype = 'float16'


class TestReduceMeanOpFloat16Prim(TestReduceMeanOpPrim):
    def set_attrs(self):
        self.dtype = 'float16'


class TestReduceMeanOpShape1D(TestReduceMeanOp):
    def set_attrs(self):
        self.shape = [100]


class TestReduceMeanOpShape1DFP16(TestReduceMeanOp):
    def set_attrs(self):
        self.shape = [100]
        self.dtype = 'float16'


class TestReduceMeanOpShape6D(TestReduceMeanOp):
    def set_attrs(self):
        self.shape = [2, 3, 4, 5, 6, 7]


class TestReduceMeanOpShape6DBF16(TestReduceMeanBF16Op):
    def set_attrs(self):
        self.shape = [2, 3, 4, 5, 6, 7]


class TestReduceMeanOpShape6DFP16(TestReduceMeanOp):
    def set_attrs(self):
        self.shape = [2, 3, 4, 5, 6, 7]
        self.dtype = 'float16'


class TestReduceMeanOpAxisAll(TestReduceMeanOp):
    def set_attrs(self):
        self.axis = [0, 1, 2, 3]


class TestReduceMeanOpAxisAllPrim(TestReduceMeanOpPrim):
    def set_attrs(self):
        self.axis = [0, 1, 2, 3]


class TestReduceMeanOpAxisAllFP16(TestReduceMeanOp):
    def set_attrs(self):
        self.axis = [0, 1, 2, 3]
        self.dtype = 'float16'


class TestReduceMeanOpAxisAllFP16Prim(TestReduceMeanOpPrim):
    def set_attrs(self):
        self.axis = [0, 1, 2, 3]
        self.dtype = 'float16'


class TestReduceMeanOpAxisAllBF16(TestReduceMeanBF16Op):
    def set_attrs(self):
        self.axis = [0, 1, 2, 3]


class TestReduceMeanOpAxisTuple(TestReduceMeanOp):
    def set_attrs(self):
        self.axis = (0, 1, 2)


class TestReduceMeanOpAxisTupleFP16(TestReduceMeanOp):
    def set_attrs(self):
        self.axis = (0, 1, 2)
        self.dtype = 'float16'


class TestReduceMeanOpAxisTupleBF16(TestReduceMeanBF16Op):
    def set_attrs(self):
        self.axis = (0, 1, 2)


class TestReduceMeanOpAxisNegative(TestReduceMeanOp):
    def set_attrs(self):
        self.axis = [-2, -1]


class TestReduceMeanOpAxisNegativeFP16(TestReduceMeanOp):
    def set_attrs(self):
        self.axis = [-2, -1]
        self.dtype = 'float16'


class TestReduceMeanOpAxisNegativeFP16Prim(TestReduceMeanOpPrim):
    def set_attrs(self):
        self.axis = [-2, -1]
        self.dtype = 'float16'


class TestReduceMeanOpAxisNegativeBF16(TestReduceMeanBF16Op):
    def set_attrs(self):
        self.axis = [-2, -1]


class TestReduceMeanOpKeepdimTrue1(TestReduceMeanOp):
    def set_attrs(self):
        self.keepdim = True


class TestReduceMeanOpKeepdimTrue1FP16(TestReduceMeanOp):
    def set_attrs(self):
        self.keepdim = True
        self.dtype = 'float16'


class TestReduceMeanOpKeepdimTrue1BF16(TestReduceMeanBF16Op):
    def set_attrs(self):
        self.keepdim = True


class TestReduceMeanOpKeepdimTrue2(TestReduceMeanOp):
    def set_attrs(self):
        self.axis = [0, 1, 2, 3]
        self.keepdim = True


class TestReduceMeanOpKeepdimTrue2FP16(TestReduceMeanOp):
    def set_attrs(self):
        self.axis = [0, 1, 2, 3]
        self.keepdim = True
        self.dtype = 'float16'


class TestReduceMeanOpKeepdimTrue2BF16(TestReduceMeanBF16Op):
    def set_attrs(self):
        self.axis = [0, 1, 2, 3]
        self.keepdim = True


class TestReduceMeanOpReduceAllTrue(TestReduceMeanOp):
    def set_attrs(self):
        self.reduce_all = True


class TestReduceMeanOpReduceAllTrueFP16(TestReduceMeanOp):
    def set_attrs(self):
        self.reduce_all = True
        self.dtype = 'float16'


class TestReduceMeanOpReduceAllTrueBF16(TestReduceMeanBF16Op):
    def set_attrs(self):
        self.reduce_all = True


class TestMeanAPI(unittest.TestCase):
    # test paddle.tensor.stat.mean

    def setUp(self):
        self.x_shape = [2, 3, 4, 5]
        self.x = np.random.uniform(-1, 1, self.x_shape).astype(np.float32)
        self.place = (
            paddle.CUDAPlace(0)
            if core.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    def test_api_static(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data('X', self.x_shape)
            out1 = paddle.mean(x)
            out2 = paddle.tensor.mean(x)
            out3 = paddle.tensor.stat.mean(x)
            axis = np.arange(len(self.x_shape)).tolist()
            out4 = paddle.mean(x, axis)
            out5 = paddle.mean(x, tuple(axis))

            exe = paddle.static.Executor(self.place)
            res = exe.run(
                feed={'X': self.x}, fetch_list=[out1, out2, out3, out4, out5]
            )
        out_ref = np.mean(self.x)
        for out in res:
            np.testing.assert_allclose(out, out_ref, rtol=0.0001)

    def test_api_dygraph(self):
        paddle.disable_static(self.place)

        def test_case(x, axis=None, keepdim=False):
            x_tensor = paddle.to_tensor(x)
            out = paddle.mean(x_tensor, axis, keepdim)
            if isinstance(axis, list):
                axis = tuple(axis)
                if len(axis) == 0:
                    axis = None
            out_ref = np.mean(x, axis, keepdims=keepdim)
            np.testing.assert_allclose(out.numpy(), out_ref, rtol=0.0001)

        test_case(self.x)
        test_case(self.x, [])
        test_case(self.x, -1)
        test_case(self.x, keepdim=True)
        test_case(self.x, 2, keepdim=True)
        test_case(self.x, [0, 2])
        test_case(self.x, (0, 2))
        test_case(self.x, [0, 1, 2, 3])
        paddle.enable_static()

    def test_base_api(self):
        with base.program_guard(base.Program(), base.Program()):
            x = paddle.static.data("x", shape=[10, 10], dtype="float32")
            out = paddle.mean(x=x, axis=1)
            place = base.CPUPlace()
            exe = base.Executor(place)
            x_np = np.random.rand(10, 10).astype(np.float32)
            res = exe.run(feed={"x": x_np}, fetch_list=[out])
        np.testing.assert_allclose(res[0], np.mean(x_np, axis=1), rtol=1e-05)

        with base.dygraph.guard():
            x_np = np.random.rand(10, 10).astype(np.float32)
            x = paddle.to_tensor(x_np)
            out = paddle.mean(x=x, axis=1)
        np.testing.assert_allclose(
            out.numpy(), np.mean(x_np, axis=1), rtol=1e-05
        )

    def test_errors(self):
        paddle.disable_static()
        x = np.random.uniform(-1, 1, [10, 12]).astype('float32')
        x = paddle.to_tensor(x)
        self.assertRaises(Exception, paddle.mean, x, -3)
        self.assertRaises(Exception, paddle.mean, x, 2)


class TestMeanWithTensorAxis1(TestReduceOPTensorAxisBase):
    def init_data(self):
        self.pd_api = paddle.mean
        self.np_api = np.mean
        self.x = paddle.randn([10, 5, 9, 9], dtype='float64')
        self.np_axis = np.array([1, 2], dtype='int64')
        self.tensor_axis = paddle.to_tensor([1, 2], dtype='int64')


class TestMeanWithTensorAxis2(TestReduceOPTensorAxisBase):
    def init_data(self):
        self.pd_api = paddle.mean
        self.np_api = np.mean
        self.x = paddle.randn([10, 10, 9, 9], dtype='float64')
        self.np_axis = np.array([0, 1, 2], dtype='int64')
        self.tensor_axis = [
            0,
            paddle.to_tensor([1], 'int64'),
            paddle.to_tensor([2], 'int64'),
        ]


class TestMeanDoubleGradCheck(unittest.TestCase):
    def mean_wrapper(self, x):
        return paddle.mean(x[0])

    @prog_scope()
    def func(self, place):
        # the shape of input variable should be clearly specified, not include -1.
        eps = 0.005
        dtype = np.float32

        data = paddle.static.data('data', [3, 4, 5], dtype)
        data.persistable = True
        out = paddle.mean(data)
        data_arr = np.random.uniform(-1, 1, data.shape).astype(dtype)

        gradient_checker.double_grad_check(
            [data], out, x_init=[data_arr], place=place, eps=eps
        )
        gradient_checker.double_grad_check_for_dygraph(
            self.mean_wrapper, [data], out, x_init=[data_arr], place=place
        )

    def test_grad(self):
        paddle.enable_static()
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            places.append(base.CPUPlace())
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for p in places:
            self.func(p)


class TestMeanTripleGradCheck(unittest.TestCase):
    def mean_wrapper(self, x):
        return paddle.mean(x[0])

    @prog_scope()
    def func(self, place):
        # the shape of input variable should be clearly specified, not include -1.
        eps = 0.005
        dtype = np.float32

        data = paddle.static.data('data', [3, 4, 5], dtype)
        data.persistable = True
        out = paddle.mean(data)
        data_arr = np.random.uniform(-1, 1, data.shape).astype(dtype)

        gradient_checker.triple_grad_check(
            [data], out, x_init=[data_arr], place=place, eps=eps
        )
        gradient_checker.triple_grad_check_for_dygraph(
            self.mean_wrapper, [data], out, x_init=[data_arr], place=place
        )

    def test_grad(self):
        paddle.enable_static()
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            places.append(base.CPUPlace())
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for p in places:
            self.func(p)


class TestMeanWithDimTensor(unittest.TestCase):
    def setUp(self):
        self.x_shape = [2, 3, 4, 5]
        self.x = np.random.uniform(-1, 1, self.x_shape).astype(np.float32)
        self.place = paddle.CPUPlace()

    def test_mean_with_dim_tesnor(self):
        old_flag = paddle.base.framework.get_flags(
            "FLAGS_enable_pir_in_executor"
        )["FLAGS_enable_pir_in_executor"]
        try:
            main_program = paddle.static.Program()
            startup_program = paddle.static.Program()
            with paddle.static.program_guard(main_program, startup_program):
                x = paddle.static.data('X', self.x_shape)
                axis = np.arange(len(self.x_shape)).tolist()
                dim = paddle.to_tensor(axis)
                out = paddle.mean(x, dim)

            paddle.framework.set_flags({"FLAGS_enable_pir_in_executor": True})
            exe = paddle.static.Executor(self.place)
            exe.run(startup_program)
            res = exe.run(
                main_program,
                feed={'X': self.x},
                fetch_list=[out],
            )
            out_ref = np.mean(self.x)
            for out in res:
                np.testing.assert_allclose(out, out_ref, rtol=0.0001)
        finally:
            if not old_flag:
                paddle.framework.set_flags(
                    {"FLAGS_enable_pir_in_executor": False}
                )


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
