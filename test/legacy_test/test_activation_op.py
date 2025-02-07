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
import warnings
from contextlib import contextmanager

import numpy as np
from op_test import OpTest, convert_float_to_uint16
from scipy.special import erf, expit
from utils import static_guard

import paddle
import paddle.nn.functional as F
from paddle import base, static
from paddle.base import Program, core, program_guard
from paddle.base.layer_helper import LayerHelper

devices = ['cpu', 'gpu']


@contextmanager
def dynamic_guard():
    paddle.disable_static()
    try:
        yield
    finally:
        paddle.enable_static()


class TestSqrtOpError(unittest.TestCase):

    def test_errors(self):
        with static_guard():
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                # The input type of sqrt op must be Variable or numpy.ndarray.
                in1 = 1
                self.assertRaises(TypeError, paddle.sqrt, in1)
                # The input dtype of sqrt op must be float16, float32, float64.
                in2 = paddle.static.data(
                    name='input2', shape=[-1, 12, 10], dtype="int32"
                )
                self.assertRaises(TypeError, paddle.sqrt, in2)

                in3 = paddle.static.data(
                    name='input3', shape=[-1, 12, 10], dtype="float16"
                )
                paddle.sqrt(x=in3)


class TestActivation(OpTest):
    def setUp(self):
        self.op_type = "exp"
        self.init_dtype()
        self.init_shape()
        self.init_kernel_type()
        self.if_enable_cinn()
        self.python_api = paddle.exp
        self.public_python_api = paddle.exp

        np.random.seed(2049)
        x = np.random.uniform(0.1, 1, self.shape).astype(self.dtype)
        out = np.exp(x)

        self.inputs = {'X': OpTest.np_dtype_to_base_dtype(x)}
        self.outputs = {'Out': out}

        self.convert_input_output()

    def test_check_output(self):
        self.check_output(check_pir_onednn=self.check_pir_onednn)

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out', check_pir_onednn=self.check_pir_onednn)

    def init_dtype(self):
        self.dtype = np.float64

    def init_shape(self):
        self.shape = [11, 17]

    def init_kernel_type(self):
        pass

    def convert_input_output(self):
        pass

    def if_enable_cinn(self):
        pass


class TestActivation_ZeroDim(TestActivation):
    def init_shape(self):
        self.shape = []


class TestExpFp32_Prim(OpTest):
    def setUp(self):
        self.op_type = "exp"
        self.prim_op_type = "prim"
        self.init_dtype()
        self.init_shape()
        self.python_api = paddle.exp
        self.public_python_api = paddle.exp

        np.random.seed(2049)
        x = np.random.uniform(0.1, 1, self.shape).astype(self.dtype)
        out = np.exp(x)

        self.inputs = {'X': OpTest.np_dtype_to_base_dtype(x)}
        self.outputs = {'Out': out}
        self.if_enable_cinn()
        self.convert_input_output()

    def test_check_output(self):
        self.check_output(
            check_pir=True,
            check_pir_onednn=self.check_pir_onednn,
            check_symbol_infer=False,
        )

    def test_check_grad(self):
        self.check_grad(
            ['X'],
            'Out',
            check_prim=True,
            check_pir=True,
            check_prim_pir=True,
            check_pir_onednn=self.check_pir_onednn,
        )

    def init_dtype(self):
        self.dtype = np.float32

    def init_shape(self):
        self.shape = [12, 17]

    def if_enable_cinn(self):
        pass

    def convert_input_output(self):
        pass


class TestExpFp64_Prim(TestExpFp32_Prim):
    def init_dtype(self):
        self.dtype = np.float64


class TestExpPrim_ZeroDim(TestExpFp32_Prim):
    def init_shape(self):
        self.shape = []


class TestExp_Complex64(OpTest):
    def setUp(self):
        self.op_type = "exp"
        self.python_api = paddle.exp
        self.public_python_api = paddle.exp
        self.init_dtype()
        self.init_shape()
        self.if_enable_cinn()
        np.random.seed(1024)
        x = (
            np.random.uniform(-1, 1, self.shape)
            + 1j * np.random.uniform(-1, 1, self.shape)
        ).astype(self.dtype)
        out = np.exp(x)
        self.inputs = {'X': OpTest.np_dtype_to_base_dtype(x)}
        self.outputs = {'Out': out}
        self.convert_input_output()

    def test_check_output(self):
        self.check_output(
            check_pir=True,
            check_pir_onednn=self.check_pir_onednn,
            check_symbol_infer=False,
        )

    def test_check_grad(self):
        self.check_grad(
            ['X'],
            'Out',
            max_relative_error=0.006,
            check_pir=True,
            check_pir_onednn=self.check_pir_onednn,
        )

    def init_dtype(self):
        self.dtype = np.complex64

    def init_shape(self):
        self.shape = [10, 12]

    def if_enable_cinn(self):
        pass

    def convert_input_output(self):
        pass


class TestExp_Complex128(TestExp_Complex64):
    def init_dtype(self):
        self.dtype = np.complex128


class Test_Exp_Op_Fp16(unittest.TestCase):

    def test_api_fp16(self):
        with static_guard():
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                np_x = np.array([[2, 3, 4], [7, 8, 9]])
                x = paddle.to_tensor(np_x, dtype='float16')
                out = paddle.exp(x)
                if core.is_compiled_with_cuda():
                    place = paddle.CUDAPlace(0)
                    exe = paddle.static.Executor(place)
                    (res,) = exe.run(fetch_list=[out])
                    x_expect = np.exp(np_x.astype('float16'))
                    np.testing.assert_allclose(res, x_expect, rtol=1e-3)


class Test_Exp_Op_Int(unittest.TestCase):
    def test_api_int(self):
        paddle.disable_static()
        for dtype in ('int32', 'int64', 'float16'):
            np_x = np.array([[2, 3, 4], [7, 8, 9]], dtype=dtype)
            x = paddle.to_tensor(np_x, dtype=dtype)
            y = paddle.exp(x)
            x_expect = np.exp(np_x)
            np.testing.assert_allclose(y.numpy(), x_expect, rtol=1e-3)
        paddle.enable_static()


class TestExpm1(TestActivation):
    def setUp(self):
        self.op_type = "expm1"
        self.prim_op_type = "prim"
        self.python_api = paddle.expm1
        self.public_python_api = paddle.expm1
        self.init_dtype()
        self.init_shape()

        np.random.seed(2049)
        x = np.random.uniform(0.1, 1, self.shape).astype(self.dtype)
        if self.dtype == np.complex64 or self.dtype == np.complex128:
            x = (
                np.random.uniform(-1, 1, self.shape)
                + 1j * np.random.uniform(-1, 1, self.shape)
            ).astype(self.dtype)
        out = np.expm1(x)

        self.inputs = {'X': OpTest.np_dtype_to_base_dtype(x)}
        self.outputs = {'Out': out}
        self.convert_input_output()

    def test_check_grad(self):
        self.check_grad(
            ['X'],
            'Out',
            check_pir=True,
            check_pir_onednn=self.check_pir_onednn,
            check_prim_pir=True,
        )

    def test_check_output(self):
        self.check_output(
            check_pir=True,
            check_pir_onednn=self.check_pir_onednn,
            check_symbol_infer=False,
        )


class TestExpm1_Complex64(TestExpm1):
    def init_dtype(self):
        self.dtype = np.complex64

    def test_check_grad(self):
        self.check_grad(
            ['X'], 'Out', check_pir=True, check_pir_onednn=self.check_pir_onednn
        )

    def test_check_output(self):
        self.check_output(
            check_pir=True, check_pir_onednn=self.check_pir_onednn
        )


class TestExpm1_Complex128(TestExpm1_Complex64):
    def init_dtype(self):
        self.dtype = np.complex128


class TestExpm1_ZeroDim(TestExpm1):
    def init_shape(self):
        self.shape = []


class TestExpm1API(unittest.TestCase):
    def init_dtype(self):
        self.dtype = 'float64'
        self.shape = [11, 17]

    def setUp(self):
        self.init_dtype()
        self.x = np.random.uniform(0.1, 1, self.shape).astype(self.dtype)
        self.out_ref = np.expm1(self.x)

        self.place = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            self.place.append(paddle.CPUPlace())
        if core.is_compiled_with_cuda():
            self.place.append(paddle.CUDAPlace(0))

    def test_static_api(self):
        def run(place):
            with static_guard():
                with paddle.static.program_guard(paddle.static.Program()):
                    X = paddle.static.data('X', self.shape, dtype=self.dtype)
                    out = paddle.expm1(X)
                    exe = paddle.static.Executor(place)
                    res = exe.run(feed={'X': self.x})
            for r in res:
                np.testing.assert_allclose(self.out_ref, r, rtol=1e-05)

        for place in self.place:
            run(place)

    def test_dygraph_api(self):
        with dynamic_guard():

            def run(place):
                X = paddle.to_tensor(self.x)
                out = paddle.expm1(X)
                np.testing.assert_allclose(
                    self.out_ref, out.numpy(), rtol=1e-05
                )

            for place in self.place:
                run(place)


class Test_Expm1_Op_Int(unittest.TestCase):
    def test_api_int(self):
        paddle.disable_static()
        for dtype in ('int32', 'int64', 'float16'):
            np_x = np.array([[2, 3, 4], [7, 8, 9]], dtype=dtype)
            x = paddle.to_tensor(np_x, dtype=dtype)
            y = paddle.expm1(x)
            x_expect = np.expm1(np_x)
            np.testing.assert_allclose(y.numpy(), x_expect, rtol=1e-3)
        paddle.enable_static()


class TestParameter:
    def test_out_name(self):
        with static_guard():
            with base.program_guard(base.Program()):
                np_x = np.array([0.1]).astype('float32').reshape((-1, 1))
                data = paddle.static.data(
                    name="X", shape=[-1, 1], dtype="float32"
                )
                out = eval(f"paddle.{self.op_type}(data, name='Y')")
                place = base.CPUPlace()
                exe = base.Executor(place)
                (result,) = exe.run(feed={"X": np_x}, fetch_list=[out])
                expected = eval(f"np.{self.op_type}(np_x)")
                np.testing.assert_allclose(result, expected, rtol=1e-05)

    def test_dygraph(self):
        with base.dygraph.guard():
            np_x = np.array([0.1])
            x = paddle.to_tensor(np_x)
            z = eval(f"paddle.{self.op_type}(x).numpy()")
            z_expected = eval(f"np.{self.op_type}(np_x)")
            np.testing.assert_allclose(z, z_expected, rtol=1e-05)


class TestSigmoid(TestActivation):
    def setUp(self):
        self.op_type = "sigmoid"
        self.prim_op_type = "prim"
        self.python_api = paddle.nn.functional.sigmoid
        self.public_python_api = paddle.nn.functional.sigmoid
        self.init_dtype()
        self.init_shape()
        self.if_enable_cinn()
        np.random.seed(1024)
        x = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        if self.dtype == np.complex64 or self.dtype == np.complex128:
            x = (
                np.random.uniform(-1, 1, self.shape)
                + 1j * np.random.uniform(-1, 1, self.shape)
            ).astype(self.dtype)
        out = 1 / (1 + np.exp(-x))

        self.inputs = {'X': OpTest.np_dtype_to_base_dtype(x)}
        self.outputs = {'Out': out}

        self.convert_input_output()

    def init_dtype(self):
        self.dtype = np.float32

    def if_enable_cinn(self):
        pass

    def test_check_output(self):
        self.check_output(
            check_pir=True,
            check_prim_pir=True,
            check_pir_onednn=self.check_pir_onednn,
            check_symbol_infer=False,
        )

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(
            ['X'],
            'Out',
            max_relative_error=0.01,
            check_prim=True,
            check_pir=True,
            check_prim_pir=True,
            check_pir_onednn=self.check_pir_onednn,
        )


class TestSigmoid_Complex64(TestSigmoid):
    def init_dtype(self):
        self.dtype = np.complex64

    def test_check_output(self):
        with paddle.static.scope_guard(paddle.static.Scope()):
            self.check_output(
                check_prim=False,
                check_prim_pir=False,
                check_pir_onednn=self.check_pir_onednn,
            )

    def test_check_grad(self):
        self.check_grad(
            ['X'],
            'Out',
            max_relative_error=0.006,
            check_prim=False,
            check_pir=True,
            check_prim_pir=False,
            check_pir_onednn=self.check_pir_onednn,
        )


class TestSigmoid_Complex128(TestSigmoid_Complex64):
    def init_dtype(self):
        self.dtype = np.complex128

    def test_check_grad(self):
        self.check_grad(
            ['X'],
            'Out',
            check_prim=False,
            check_pir=True,
            check_prim_pir=False,
            check_pir_onednn=self.check_pir_onednn,
        )


class TestSigmoid_ZeroDim(TestSigmoid):
    def init_shape(self):
        self.shape = []


@unittest.skipIf(
    not core.is_compiled_with_cuda() or core.is_compiled_with_rocm(),
    "core is not compiled with CUDA",
)
class TestSigmoidBF16(OpTest):
    def setUp(self):
        self.op_type = "sigmoid"
        self.prim_op_type = "comp"
        self.python_api = paddle.nn.functional.sigmoid
        self.public_python_api = paddle.nn.functional.sigmoid
        self.init_dtype()
        self.init_shape()
        self.if_enable_cinn()
        np.random.seed(1024)
        x = np.random.uniform(-1, 1, self.shape).astype(np.float32)
        out = 1 / (1 + np.exp(-x))

        self.inputs = {
            'X': OpTest.np_dtype_to_base_dtype(convert_float_to_uint16(x))
        }
        self.outputs = {'Out': convert_float_to_uint16(out)}

    def init_dtype(self):
        self.dtype = np.uint16

    def init_shape(self):
        self.shape = [11, 17]

    def if_enable_cinn(self):
        self.enable_cinn = False

    def test_check_output(self):
        place = core.CUDAPlace(0)
        self.check_output_with_place(
            place,
            check_prim=True,
            check_pir=True,
            check_prim_pir=True,
            check_pir_onednn=self.check_pir_onednn,
            check_symbol_infer=False,
        )

    def test_check_grad(self):
        place = core.CUDAPlace(0)
        self.check_grad_with_place(
            place,
            ['X'],
            'Out',
            check_prim=True,
            check_pir=True,
            check_prim_pir=True,
        )


'''
class TestSigmoidBF16_ZeroDim(TestSigmoidBF16):

    def init_shape(self):
        self.shape = []
'''


class TestSilu(TestActivation):
    def setUp(self):
        self.op_type = "silu"
        self.prim_op_type = "comp"
        self.python_api = paddle.nn.functional.silu
        self.public_python_api = paddle.nn.functional.silu
        self.init_dtype()
        self.init_shape()
        self.if_enable_cinn()

        np.random.seed(1024)
        x = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        if self.dtype == np.complex64 or self.dtype == np.complex128:
            x = (
                np.random.uniform(-1, 1, self.shape)
                + 1j * np.random.uniform(-1, 1, self.shape)
            ).astype(self.dtype)
        out = x / (np.exp(-x) + 1)
        self.inputs = {'X': OpTest.np_dtype_to_base_dtype(x)}
        self.outputs = {'Out': out}

        self.convert_input_output()

    def init_dtype(self):
        self.dtype = np.float32

    def if_enable_cinn(self):
        pass

    def test_check_output(self):
        if self.dtype == np.complex64 or self.dtype == np.complex128:
            self.check_output(
                check_pir=True,
                check_pir_onednn=self.check_pir_onednn,
                check_symbol_infer=False,
            )
        else:
            self.check_output(
                check_prim=True,
                check_pir=True,
                check_prim_pir=True,
                check_pir_onednn=self.check_pir_onednn,
                check_symbol_infer=False,
            )

    def test_check_grad(self):
        # TODO(BeingGod): set `check_prim=True` when `fill_constant` supports `complex` dtype
        if self.dtype == np.complex64 or self.dtype == np.complex128:
            self.check_grad(
                ['X'],
                'Out',
                check_pir=True,
                check_pir_onednn=self.check_pir_onednn,
            )
        else:
            self.check_grad(
                ['X'],
                'Out',
                check_prim=True,
                check_pir=True,
                check_prim_pir=True,
                check_pir_onednn=self.check_pir_onednn,
            )


class TestSilu_ZeroDim(TestSilu):
    def init_shape(self):
        self.shape = []


class TestSilu_Complex64(TestSilu):
    def init_dtype(self):
        self.dtype = np.complex64


class TestSilu_Complex128(TestSilu):
    def init_dtype(self):
        self.dtype = np.complex128


class TestSiluAPI(unittest.TestCase):
    # test paddle.nn.Silu, paddle.nn.functional.silu
    def setUp(self):
        self.x_np = np.random.uniform(-1, 1, [11, 17]).astype('float32')
        self.place = (
            paddle.CUDAPlace(0)
            if core.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    def test_static_api(self):
        with static_guard():
            with paddle.static.program_guard(paddle.static.Program()):
                x = paddle.static.data('X', [11, 17])
                out1 = F.silu(x)
                m = paddle.nn.Silu()
                out2 = m(x)
                exe = paddle.static.Executor(self.place)
                res = exe.run(feed={'X': self.x_np}, fetch_list=[out1, out2])
            out_ref = self.x_np / (1 + np.exp(-self.x_np))
            for r in res:
                np.testing.assert_allclose(out_ref, r, rtol=1e-05)

    def test_dygraph_api(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.x_np)
        out1 = F.silu(x)
        m = paddle.nn.Silu()
        out2 = m(x)
        out_ref = self.x_np / (1 + np.exp(-self.x_np))
        for r in [out1, out2]:
            np.testing.assert_allclose(out_ref, r.numpy(), rtol=1e-05)
        paddle.enable_static()

    def test_errors(self):
        with static_guard():
            with paddle.static.program_guard(paddle.static.Program()):
                # The input type must be Variable.
                self.assertRaises(TypeError, F.silu, 1)
                # The input dtype must be float16, float32, float64.
                x_int32 = paddle.static.data(
                    name='x_int32', shape=[11, 17], dtype='int32'
                )
                self.assertRaises(TypeError, F.silu, x_int32)
                # support the input dtype is float16
                x_fp16 = paddle.static.data(
                    name='x_fp16', shape=[11, 17], dtype='float16'
                )
                F.silu(x_fp16)


class TestLogSigmoid(TestActivation):
    def setUp(self):
        self.op_type = "logsigmoid"
        self.python_api = paddle.nn.functional.log_sigmoid
        self.init_dtype()
        self.init_shape()

        np.random.seed(2048)
        if self.dtype is np.complex64 or self.dtype is np.complex128:
            x = (
                np.random.uniform(-1, 1, self.shape)
                + 1j * np.random.uniform(-1, 1, self.shape)
            ).astype(self.dtype)
        else:
            x = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        out = np.log(1 / (1 + np.exp(-x)))
        self.inputs = {'X': OpTest.np_dtype_to_base_dtype(x)}
        self.outputs = {'Out': out}

        self.convert_input_output()

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(
            ['X'],
            'Out',
            max_relative_error=0.008,
            check_pir=True,
            check_pir_onednn=self.check_pir_onednn,
        )


class TestLogSigmoidComplex64(TestLogSigmoid):
    def init_dtype(self):
        self.dtype = np.complex64


class TestLogSigmoidComplex128(TestLogSigmoid):
    def init_dtype(self):
        self.dtype = np.complex128


class TestLogSigmoid_ZeroDim(TestLogSigmoid):
    def init_shape(self):
        self.shape = []


class TestLogSigmoidAPI(unittest.TestCase):
    # test paddle.nn.LogSigmoid, paddle.nn.functional.log_sigmoid
    def setUp(self):
        np.random.seed(1024)
        self.x_np = np.random.uniform(-1, 1, [11, 17]).astype('float32')
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    def test_static_api(self):
        with static_guard():
            with paddle.static.program_guard(paddle.static.Program()):
                x = paddle.static.data('X', [11, 17])
                out1 = F.log_sigmoid(x)
                m = paddle.nn.LogSigmoid()
                out2 = m(x)
                exe = paddle.static.Executor(self.place)
                res = exe.run(feed={'X': self.x_np}, fetch_list=[out1, out2])
            out_ref = np.log(1 / (1 + np.exp(-self.x_np)))
            for r in res:
                np.testing.assert_allclose(out_ref, r, rtol=1e-05)

    def test_dygraph_api(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.x_np)
        out1 = F.log_sigmoid(x)
        m = paddle.nn.LogSigmoid()
        out2 = m(x)
        out_ref = np.log(1 / (1 + np.exp(-self.x_np)))
        for r in [out1, out2]:
            np.testing.assert_allclose(out_ref, r.numpy(), rtol=1e-05)
        paddle.enable_static()

    def test_errors(self):
        with static_guard():
            with paddle.static.program_guard(paddle.static.Program()):
                # The input type must be Variable.
                self.assertRaises(TypeError, F.log_sigmoid, 1)
                # The input dtype must be float16, float32, float64.
                x_int32 = paddle.static.data(
                    name='x_int32', shape=[11, 17], dtype='int32'
                )
                self.assertRaises(TypeError, F.log_sigmoid, x_int32)
                # support the input dtype is float16
                x_fp16 = paddle.static.data(
                    name='x_fp16', shape=[11, 17], dtype='float16'
                )
                F.log_sigmoid(x_fp16)


class TestTanh(TestActivation, TestParameter):
    def setUp(self):
        self.op_type = "tanh"
        self.prim_op_type = "prim"
        self.python_api = paddle.tanh
        self.public_python_api = paddle.tanh
        self.init_dtype()
        self.init_shape()
        self.if_enable_cinn()

        np.random.seed(1024)
        x = np.random.uniform(0.1, 1, self.shape).astype(self.dtype)
        if self.dtype == np.complex64 or self.dtype == np.complex128:
            x = (
                np.random.uniform(-1, 1, self.shape)
                + 1j * np.random.uniform(-1, 1, self.shape)
            ).astype(self.dtype)
        out = np.tanh(x)
        self.inputs = {'X': OpTest.np_dtype_to_base_dtype(x)}
        self.outputs = {'Out': out}
        self.convert_input_output()

    def test_check_output(self):
        self.check_output(
            check_pir=True,
            check_pir_onednn=self.check_pir_onednn,
            check_symbol_infer=False,
        )

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        # TODO(ScottWong98): set `check_prim=False` when `fill_any_like` supports `complex` dtype
        if self.dtype == np.complex64 or self.dtype == np.complex128:
            self.check_grad(
                ['X'],
                'Out',
                check_prim=False,
                check_prim_pir=False,
                check_pir=True,
                check_pir_onednn=self.check_pir_onednn,
            )
        else:
            self.check_grad(
                ['X'],
                'Out',
                check_prim=True,
                check_pir=True,
                check_prim_pir=True,
                check_pir_onednn=self.check_pir_onednn,
            )

    def init_dtype(self):
        # TODO If dtype is float64, the output (Out) has diff at CPUPlace
        # when using and not using inplace. Therefore, set dtype as float32
        # for now.
        self.dtype = np.float32

    def if_enable_cinn(self):
        pass


class TestTanh_Complex64(TestTanh):
    def init_dtype(self):
        self.dtype = np.complex64

    def test_check_output(self):
        with paddle.static.scope_guard(paddle.static.Scope()):
            self.check_output(
                check_pir=True, check_pir_onednn=self.check_pir_onednn
            )


class TestTanh_Complex128(TestTanh):
    def init_dtype(self):
        self.dtype = np.complex128

    def test_check_output(self):
        with paddle.static.scope_guard(paddle.static.Scope()):
            self.check_output(
                check_pir=True, check_pir_onednn=self.check_pir_onednn
            )


class TestTanh_ZeroDim(TestTanh):
    def init_shape(self):
        self.shape = []


class TestTanhAPI(unittest.TestCase):
    # test paddle.tanh, paddle.nn.tanh, paddle.nn.functional.tanh
    def setUp(self):
        self.dtype = 'float32'
        np.random.seed(1024)
        self.x_np = np.random.uniform(-1, 1, [10, 12]).astype(self.dtype)
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )
        self.executed_api()

    def executed_api(self):
        self.tanh = F.tanh

    def test_static_api(self):
        with static_guard():
            with paddle.static.program_guard(paddle.static.Program()):
                x = paddle.static.data('X', [10, 12], self.dtype)
                out1 = self.tanh(x)
                th = paddle.nn.Tanh()
                out2 = th(x)
                exe = paddle.static.Executor(self.place)
                res = exe.run(feed={'X': self.x_np}, fetch_list=[out1, out2])
            out_ref = np.tanh(self.x_np)
            for r in res:
                np.testing.assert_allclose(out_ref, r, rtol=1e-05)

    def test_dygraph_api(self):
        with dynamic_guard():
            x = paddle.to_tensor(self.x_np)
            out1 = F.tanh(x)
            out2 = paddle.tanh(x)
            th = paddle.nn.Tanh()
            out3 = th(x)
            out_ref = np.tanh(self.x_np)
            for r in [out1, out2, out3]:
                np.testing.assert_allclose(out_ref, r.numpy(), rtol=1e-05)

    def test_errors(self):
        with static_guard():
            with paddle.static.program_guard(paddle.static.Program()):
                # The input type must be Variable.
                self.assertRaises(TypeError, self.tanh, 1)
                # The input dtype must be float16, float32.
                x_int32 = paddle.static.data(
                    name='x_int32', shape=[12, 10], dtype='int32'
                )
                self.assertRaises(TypeError, self.tanh, x_int32)
                # support the input dtype is float16
                x_fp16 = paddle.static.data(
                    name='x_fp16', shape=[12, 10], dtype='float16'
                )
                self.tanh(x_fp16)


class TestTanhInplaceAPI(TestTanhAPI):
    # test paddle.tanh_
    def executed_api(self):
        self.tanh = paddle.tanh_


class TestAtan(TestActivation, TestParameter):
    def setUp(self):
        self.op_type = "atan"
        self.prim_op_type = "prim"
        self.python_api = paddle.atan
        self.public_python_api = paddle.atan
        self.init_dtype()
        self.init_shape()

        np.random.seed(1024)
        x = np.random.uniform(0.1, 1, self.shape).astype(self.dtype)
        if self.dtype == np.complex64 or self.dtype == np.complex128:
            x = (
                np.random.uniform(0.1, 1, self.shape)
                + 1j * np.random.uniform(0.1, 1, self.shape)
            ).astype(self.dtype)
        out = np.arctan(x)

        self.inputs = {'X': OpTest.np_dtype_to_base_dtype(x)}
        self.outputs = {'Out': out}
        self.convert_input_output()

    def test_check_output(self):
        self.check_output(
            check_pir=True,
            check_pir_onednn=self.check_pir_onednn,
            check_symbol_infer=False,
        )

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        if self.dtype == np.complex64 or self.dtype == np.complex128:
            self.check_grad(
                ['X'],
                'Out',
                check_pir=True,
                check_pir_onednn=self.check_pir_onednn,
            )
        else:
            self.check_grad(
                ['X'],
                'Out',
                check_pir=True,
                check_pir_onednn=self.check_pir_onednn,
                check_prim_pir=True,
            )

    def test_out_name(self):
        with static_guard():
            with base.program_guard(base.Program()):
                np_x = np.array([0.1]).astype('float32').reshape((-1, 1))
                data = paddle.static.data(
                    name="X", shape=[-1, 1], dtype="float32"
                )
                out = paddle.atan(data, name='Y')
                place = base.CPUPlace()
                exe = base.Executor(place)
                (result,) = exe.run(feed={"X": np_x}, fetch_list=[out])
                expected = np.arctan(np_x)
                self.assertEqual(result, expected)

    def test_dygraph(self):
        with base.dygraph.guard():
            np_x = np.array([0.1])
            x = paddle.to_tensor(np_x)
            z = paddle.atan(x).numpy()
            z_expected = np.arctan(np_x)
            self.assertEqual(z, z_expected)


class TestAtan_Complex64(TestAtan):
    def init_dtype(self):
        self.dtype = np.complex64


class TestAtan_Complex128(TestAtan):
    def init_dtype(self):
        self.dtype = np.complex128


class TestAtan_ZeroDim(TestAtan):
    def init_shape(self):
        self.shape = []


class TestSinh(TestActivation):
    def setUp(self):
        self.op_type = "sinh"
        self.python_api = paddle.sinh
        self.init_dtype()
        self.init_shape()

        np.random.seed(1024)
        x = np.random.uniform(0.1, 1, self.shape).astype(self.dtype)
        if self.dtype == np.complex64 or self.dtype == np.complex128:
            x = (
                np.random.uniform(0.1, 1, self.shape)
                + 1j * np.random.uniform(0.1, 1, self.shape)
            ).astype(self.dtype)
        out = np.sinh(x)
        self.inputs = {'X': OpTest.np_dtype_to_base_dtype(x)}
        self.outputs = {'Out': out}

        self.convert_input_output()

    def test_check_output(self):
        self.check_output(
            check_pir=True,
            check_pir_onednn=self.check_pir_onednn,
            check_symbol_infer=False,
        )

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(
            ['X'], 'Out', check_pir=True, check_pir_onednn=self.check_pir_onednn
        )


class TestSinh_Complex64(TestSinh):
    def init_dtype(self):
        self.dtype = np.complex64


class TestSinh_Complex128(TestSinh):
    def init_dtype(self):
        self.dtype = np.complex128


class TestSinh_ZeroDim(TestSinh):
    def init_shape(self):
        self.shape = []


class TestSinhAPI(unittest.TestCase):
    def test_dygraph(self):
        with base.dygraph.guard():
            np_x = np.array([0.1])
            x = paddle.to_tensor(np_x)
            z = paddle.sinh(x).numpy()
            z_expected = np.sinh(np_x)
            np.testing.assert_allclose(z, z_expected, rtol=1e-05)

    def test_api(self):
        with static_guard():
            test_data_shape = [11, 17]
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                input_x = np.random.uniform(0.1, 1, test_data_shape).astype(
                    "float32"
                )
                data_x = paddle.static.data(
                    name="data_x",
                    shape=test_data_shape,
                    dtype="float32",
                )

                pd_sinh_out = paddle.sinh(data_x)
                exe = base.Executor(place=base.CPUPlace())
                exe.run(paddle.static.default_startup_program())
                (np_sinh_res,) = exe.run(
                    paddle.static.default_main_program(),
                    feed={"data_x": input_x},
                    fetch_list=[pd_sinh_out],
                )

            expected_res = np.sinh(input_x)
            np.testing.assert_allclose(np_sinh_res, expected_res, rtol=1e-05)

    def test_backward(self):
        test_data_shape = [11, 17]
        with base.dygraph.guard():
            input_x = np.random.uniform(0.1, 1, test_data_shape).astype(
                "float32"
            )
            var = paddle.to_tensor(input_x)
            var.stop_gradient = False
            loss = paddle.sinh(var)
            loss.backward()
            grad_var = var.gradient()
            self.assertEqual(grad_var.shape, input_x.shape)


class TestSinhOpError(unittest.TestCase):

    def test_errors(self):
        with static_guard():
            with paddle.static.program_guard(paddle.static.Program()):
                # The input type must be Variable.
                self.assertRaises(TypeError, paddle.sinh, 1)
                # The input dtype must be float16, float32, float64.
                x_int32 = paddle.static.data(
                    name='x_int32', shape=[12, 10], dtype='int32'
                )
                self.assertRaises(TypeError, paddle.sinh, x_int32)
                # support the input dtype is float16
                if paddle.is_compiled_with_cuda():
                    x_fp16 = paddle.static.data(
                        name='x_fp16', shape=[12, 10], dtype='float16'
                    )
                    paddle.sinh(x_fp16)


class TestCosh(TestActivation):
    def setUp(self):
        self.op_type = "cosh"
        self.python_api = paddle.cosh
        self.init_dtype()
        self.init_shape()

        np.random.seed(1024)
        x = np.random.uniform(0.1, 1, self.shape).astype(self.dtype)
        if self.dtype == np.complex64 or self.dtype == np.complex128:
            x = (
                np.random.uniform(0.1, 1, self.shape)
                + 1j * np.random.uniform(0.1, 1, self.shape)
            ).astype(self.dtype)
        out = np.cosh(x)
        self.inputs = {'X': OpTest.np_dtype_to_base_dtype(x)}
        self.outputs = {'Out': out}

        self.convert_input_output()

    def test_check_output(self):
        self.check_output(
            check_pir=True,
            check_pir_onednn=self.check_pir_onednn,
            check_symbol_infer=False,
        )

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        if self.dtype == np.complex64 or self.dtype == np.complex128:
            # Complex64 [CPU]: AssertionError: 0.006845869 not less than or equal to 0.005
            self.check_grad(
                ['X'],
                'Out',
                max_relative_error=0.007,
                check_pir=True,
                check_pir_onednn=self.check_pir_onednn,
            )
        else:
            self.check_grad(
                ['X'],
                'Out',
                check_pir=True,
                check_pir_onednn=self.check_pir_onednn,
            )


class TestCosh_Complex64(TestCosh):
    def init_dtype(self):
        self.dtype = np.complex64


class TestCosh_Complex128(TestCosh):
    def init_dtype(self):
        self.dtype = np.complex128


class TestCosh_ZeroDim(TestCosh):
    def init_shape(self):
        self.shape = []


class TestCoshAPI(unittest.TestCase):
    def test_dygraph(self):
        with base.dygraph.guard():
            np_x = np.array([0.1])
            x = paddle.to_tensor(np_x)
            z = paddle.cosh(x).numpy()
            z_expected = np.cosh(np_x)
            np.testing.assert_allclose(z, z_expected, rtol=1e-05)

    def test_api(self):
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with static_guard():
            test_data_shape = [11, 17]
            input_x = np.random.uniform(0.1, 1, test_data_shape).astype(
                "float32"
            )
            with base.program_guard(main, startup):
                data_x = paddle.static.data(
                    name="data_x",
                    shape=test_data_shape,
                    dtype="float32",
                )

                pd_cosh_out = paddle.cosh(data_x)
                exe = base.Executor(place=base.CPUPlace())
                (np_cosh_res,) = exe.run(
                    main,
                    feed={"data_x": input_x},
                    fetch_list=[pd_cosh_out],
                )

            expected_res = np.cosh(input_x)
            np.testing.assert_allclose(np_cosh_res, expected_res, rtol=1e-05)

    def test_backward(self):
        test_data_shape = [11, 17]
        with base.dygraph.guard():
            input_x = np.random.uniform(0.1, 1, test_data_shape).astype(
                "float32"
            )
            var = paddle.to_tensor(input_x)
            var.stop_gradient = False
            loss = paddle.cosh(var)
            loss.backward()
            grad_var = var.gradient()
            self.assertEqual(grad_var.shape, input_x.shape)


class TestCoshOpError(unittest.TestCase):

    def test_errors(self):
        with static_guard():
            with program_guard(Program()):
                # The input type must be Variable.
                self.assertRaises(TypeError, paddle.cosh, 1)
                # The input dtype must be float16, float32, float64.
                x_int32 = paddle.static.data(
                    name='x_int32', shape=[12, 10], dtype='int32'
                )
                self.assertRaises(TypeError, paddle.cosh, x_int32)
                # support the input dtype is float16
                x_fp16 = paddle.static.data(
                    name='x_fp16', shape=[12, 10], dtype='float16'
                )
                paddle.cosh(x_fp16)


def ref_tanhshrink(x):
    out = x - np.tanh(x)
    return out


class TestTanhshrink(TestActivation):
    def setUp(self):
        self.op_type = "tanh_shrink"
        self.python_api = paddle.nn.functional.tanhshrink
        self.init_dtype()
        self.init_shape()

        np.random.seed(1024)
        x = np.random.uniform(10, 20, self.shape).astype(self.dtype)
        out = ref_tanhshrink(x)
        self.inputs = {'X': OpTest.np_dtype_to_base_dtype(x)}
        self.outputs = {'Out': out}

        self.convert_input_output()

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(
            ['X'], 'Out', check_pir=True, check_pir_onednn=self.check_pir_onednn
        )

    def test_check_output(self):
        self.check_output(
            check_pir=True,
            check_pir_onednn=self.check_pir_onednn,
            check_symbol_infer=False,
        )


class TestTanhshrink_ZeroDim(TestTanhshrink):
    def init_shape(self):
        self.shape = []


class TestTanhshrinkAPI(unittest.TestCase):
    # test paddle.nn.Tanhshrink, paddle.nn.functional.tanhshrink
    def setUp(self):
        np.random.seed(1024)
        self.x_np = np.random.uniform(10, 20, [10, 17]).astype(np.float64)
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    def test_static_api(self):
        with static_guard():
            with paddle.static.program_guard(paddle.static.Program()):
                x = paddle.static.data('X', self.x_np.shape, self.x_np.dtype)
                out1 = F.tanhshrink(x)
                tanhshrink = paddle.nn.Tanhshrink()
                out2 = tanhshrink(x)
                exe = paddle.static.Executor(self.place)
                res = exe.run(feed={'X': self.x_np}, fetch_list=[out1, out2])
            out_ref = ref_tanhshrink(self.x_np)
            for r in res:
                np.testing.assert_allclose(out_ref, r, rtol=1e-05)

    def test_dygraph_api(self):
        with dynamic_guard():
            x = paddle.to_tensor(self.x_np)
            out1 = F.tanhshrink(x)
            tanhshrink = paddle.nn.Tanhshrink()
            out2 = tanhshrink(x)
            out_ref = ref_tanhshrink(self.x_np)
            for r in [out1, out2]:
                np.testing.assert_allclose(out_ref, r.numpy(), rtol=1e-05)

    def test_errors(self):
        with static_guard():
            with paddle.static.program_guard(paddle.static.Program()):
                # The input type must be Variable.
                self.assertRaises(TypeError, F.tanhshrink, 1)
                # The input dtype must be float16, float32, float64.
                x_int32 = paddle.static.data(
                    name='x_int32', shape=[12, 10], dtype='int32'
                )
                self.assertRaises(TypeError, F.tanhshrink, x_int32)
                # support the input dtype is float16
                if paddle.is_compiled_with_cuda():
                    x_fp16 = paddle.static.data(
                        name='x_fp16', shape=[12, 10], dtype='float16'
                    )
                    F.tanhshrink(x_fp16)


def ref_hardshrink(x, threshold):
    out = np.copy(x)
    out[(out >= -threshold) & (out <= threshold)] = 0
    return out


class TestHardShrink(TestActivation):
    def setUp(self):
        self.op_type = "hard_shrink"
        self.python_api = paddle.nn.functional.hardshrink
        self.init_dtype()
        self.init_shape()

        self.threshold = 0.5
        self.set_attrs()
        np.random.seed(1024)
        x = np.random.uniform(-1, 1, self.shape).astype(self.dtype) * 10
        out = ref_hardshrink(x, self.threshold)
        self.inputs = {'X': OpTest.np_dtype_to_base_dtype(x)}
        self.outputs = {'Out': out}

        self.attrs = {'threshold': self.threshold}

        self.convert_input_output()

    def init_shape(self):
        self.shape = [10, 12]

    def set_attrs(self):
        pass

    def test_check_output(self):
        self.check_output(
            check_pir=True,
            check_pir_onednn=self.check_pir_onednn,
            check_symbol_infer=False,
        )

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(
            ['X'], 'Out', check_pir=True, check_pir_onednn=self.check_pir_onednn
        )


class TestHardShrink_threshold_negative(TestHardShrink):
    def set_attrs(self):
        self.threshold = -0.1


'''
class TestHardShrink_ZeroDim(TestHardShrink):

    def init_shape(self):
        self.shape = []
'''


class TestHardShrinkAPI(unittest.TestCase):
    # test paddle.nn.Hardshrink, paddle.nn.functional.hardshrink
    def setUp(self):
        np.random.seed(1024)
        self.x_np = np.random.uniform(-1, 1, [10, 12]).astype('float32')
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    def test_static_api(self):
        with static_guard():
            with paddle.static.program_guard(paddle.static.Program()):
                x = paddle.static.data('X', [10, 12], dtype="float32")
                out1 = F.hardshrink(x)
                hd = paddle.nn.Hardshrink()
                out2 = hd(x)
                exe = paddle.static.Executor(self.place)
                res = exe.run(feed={'X': self.x_np}, fetch_list=[out1, out2])
            out_ref = ref_hardshrink(self.x_np, 0.5)
            for r in res:
                np.testing.assert_allclose(out_ref, r, rtol=1e-05)

    def test_dygraph_api(self):
        with dynamic_guard():
            x = paddle.to_tensor(self.x_np)
            out1 = F.hardshrink(x)
            hd = paddle.nn.Hardshrink()
            out2 = hd(x)
            out_ref = ref_hardshrink(self.x_np, 0.5)
            for r in [out1, out2]:
                np.testing.assert_allclose(out_ref, r.numpy(), rtol=1e-05)

            out1 = F.hardshrink(x, 0.6)
            hd = paddle.nn.Hardshrink(0.6)
            out2 = hd(x)
            out_ref = ref_hardshrink(self.x_np, 0.6)
            for r in [out1, out2]:
                np.testing.assert_allclose(out_ref, r.numpy(), rtol=1e-05)

    def test_errors(self):
        with static_guard():
            with paddle.static.program_guard(paddle.static.Program()):
                # The input type must be Variable.
                self.assertRaises(TypeError, F.hardshrink, 1)
                # The input dtype must be float16, float32, float64.
                x_int32 = paddle.static.data(
                    name='x_int32', shape=[12, 10], dtype='int32'
                )
                self.assertRaises(TypeError, F.hardshrink, x_int32)
                # support the input dtype is float16
                x_fp16 = paddle.static.data(
                    name='x_fp16', shape=[12, 10], dtype='float16'
                )
                F.hardshrink(x_fp16)


def ref_hardtanh(x, min=-1.0, max=1.0):
    out = np.copy(x)
    out[np.abs(x - min) < 0.005] = min + 0.02
    out[np.abs(x - max) < 0.005] = max + 0.02
    out = np.minimum(np.maximum(x, min), max)
    return out


class TestHardtanhAPI(unittest.TestCase):
    # test paddle.nn.Hardtanh, paddle.nn.functional.hardtanh
    def setUp(self):
        np.random.seed(1024)
        self.x_np = np.random.uniform(-3, 3, [10, 12]).astype('float32')
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    def test_static_api(self):
        with static_guard():
            with paddle.static.program_guard(paddle.static.Program()):
                x = paddle.static.data('X', [10, 12], dtype="float32")
                out1 = F.hardtanh(x)
                m = paddle.nn.Hardtanh()
                out2 = m(x)
                exe = paddle.static.Executor(self.place)
                res = exe.run(feed={'X': self.x_np}, fetch_list=[out1, out2])
            out_ref = ref_hardtanh(self.x_np)
            for r in res:
                np.testing.assert_allclose(out_ref, r, rtol=1e-05)

    def test_dygraph_api(self):
        with dynamic_guard():
            x = paddle.to_tensor(self.x_np)
            out1 = F.hardtanh(x)
            m = paddle.nn.Hardtanh()
            out2 = m(x)
            out_ref = ref_hardtanh(self.x_np)
            for r in [out1, out2]:
                np.testing.assert_allclose(out_ref, r.numpy(), rtol=1e-05)

            out1 = F.hardtanh(x, -2.0, 2.0)
            m = paddle.nn.Hardtanh(-2.0, 2.0)
            out2 = m(x)
            out_ref = ref_hardtanh(self.x_np, -2.0, 2.0)
            for r in [out1, out2]:
                np.testing.assert_allclose(out_ref, r.numpy(), rtol=1e-05)

    def test_errors(self):
        with static_guard():
            with paddle.static.program_guard(paddle.static.Program()):
                # The input type must be Variable.
                self.assertRaises(TypeError, F.hardtanh, 1)
                # The input dtype must be float16, float32, float64.
                x_int32 = paddle.static.data(
                    name='x_int32', shape=[12, 10], dtype='int32'
                )
                self.assertRaises(TypeError, F.hardtanh, x_int32)
                # support the input dtype is float16
                x_fp16 = paddle.static.data(
                    name='x_fp16', shape=[12, 10], dtype='float16'
                )
                F.hardtanh(x_fp16)


def ref_softshrink(x, threshold=0.5):
    out = np.copy(x)
    out = (out < -threshold) * (out + threshold) + (out > threshold) * (
        out - threshold
    )
    return out


class TestSoftshrink(TestActivation):
    def setUp(self):
        self.op_type = "softshrink"
        self.python_api = paddle.nn.functional.softshrink
        self.init_dtype()
        self.init_shape()

        threshold = 0.8

        np.random.seed(1023)
        x = np.random.uniform(0.25, 10, self.shape).astype(self.dtype)
        out = ref_softshrink(x, threshold)

        self.inputs = {'X': OpTest.np_dtype_to_base_dtype(x)}
        self.outputs = {'Out': out}

        self.attrs = {"lambda": threshold}

    def test_check_output(self):
        self.check_output(
            check_pir=True,
            check_pir_onednn=self.check_pir_onednn,
            check_symbol_infer=False,
        )

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(
            ['X'], 'Out', check_pir=True, check_pir_onednn=self.check_pir_onednn
        )


class TestSoftshrink_ZeroDim(TestSoftshrink):
    def init_shape(self):
        self.shape = []


class TestSoftshrinkAPI(unittest.TestCase):
    # test paddle.nn.Softshrink, paddle.nn.functional.softshrink
    def setUp(self):
        self.threshold = 0.8
        np.random.seed(1024)
        self.x_np = np.random.uniform(0.25, 10, [10, 12]).astype(np.float64)
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    def test_static_api(self):
        with static_guard():
            with paddle.static.program_guard(paddle.static.Program()):
                x = paddle.static.data('X', self.x_np.shape, self.x_np.dtype)
                out1 = F.softshrink(x, self.threshold)
                softshrink = paddle.nn.Softshrink(self.threshold)
                out2 = softshrink(x)
                exe = paddle.static.Executor(self.place)
                res = exe.run(feed={'X': self.x_np}, fetch_list=[out1, out2])
            out_ref = ref_softshrink(self.x_np, self.threshold)
            for r in res:
                np.testing.assert_allclose(out_ref, r, rtol=1e-05)

    def test_dygraph_api(self):
        with dynamic_guard():
            x = paddle.to_tensor(self.x_np)
            out1 = F.softshrink(x, self.threshold)
            softshrink = paddle.nn.Softshrink(self.threshold)
            out2 = softshrink(x)
            out_ref = ref_softshrink(self.x_np, self.threshold)
            for r in [out1, out2]:
                np.testing.assert_allclose(out_ref, r.numpy(), rtol=1e-05)

    def test_errors(self):
        with static_guard():
            with paddle.static.program_guard(paddle.static.Program()):
                # The input type must be Variable.
                self.assertRaises(TypeError, F.softshrink, 1)
                # The input dtype must be float16, float32, float64.
                x_int32 = paddle.static.data(
                    name='x_int32', shape=[12, 10], dtype='int32'
                )
                self.assertRaises(TypeError, F.softshrink, x_int32)
                # The threshold must be no less than zero
                x_fp32 = paddle.static.data(
                    name='x_fp32', shape=[12, 10], dtype='float32'
                )
                self.assertRaises(ValueError, F.softshrink, x_fp32, -1.0)
                # support the input dtype is float16
                if core.is_compiled_with_cuda():
                    x_fp16 = paddle.static.data(
                        name='x_fp16', shape=[12, 10], dtype='float16'
                    )
                    F.softshrink(x_fp16)


class TestSqrt(TestActivation, TestParameter):
    def setUp(self):
        self.op_type = "sqrt"
        self.prim_op_type = "prim"
        self.python_api = paddle.sqrt
        self.public_python_api = paddle.sqrt

        self.init_dtype()
        self.init_shape()
        self.if_enable_cinn()

        np.random.seed(1023)
        x = np.random.uniform(0.1, 1, self.shape).astype(self.dtype)
        if self.dtype == np.complex64 or self.dtype == np.complex128:
            x = (
                np.random.uniform(-1, 1, self.shape)
                + 1j * np.random.uniform(-1, 1, self.shape)
            ).astype(self.dtype)
        out = np.sqrt(x)

        self.inputs = {'X': OpTest.np_dtype_to_base_dtype(x)}
        self.outputs = {'Out': out}
        self.convert_input_output()

    def if_enable_cinn(self):
        pass

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        if self.dtype not in [np.complex64, np.complex128]:
            self.check_grad(
                ['X'],
                'Out',
                check_prim=True,
                check_pir=True,
                check_prim_pir=True,
                check_pir_onednn=self.check_pir_onednn,
            )
        else:
            self.check_grad(
                ['X'],
                'Out',
            )

    def test_check_output(self):
        self.check_output(
            check_prim=True,
            check_pir=True,
            check_prim_pir=True,
            check_pir_onednn=self.check_pir_onednn,
            check_symbol_infer=False,
        )


class TestSqrtPrimFp32(TestActivation):
    def setUp(self):
        self.op_type = "sqrt"
        self.prim_op_type = "prim"
        self.python_api = paddle.sqrt
        self.public_python_api = paddle.sqrt
        self.init_dtype()
        self.init_shape()
        self.if_enable_cinn()
        np.random.seed(1023)
        x = np.random.uniform(0.1, 1, self.shape).astype(self.dtype)
        out = np.sqrt(x)

        self.inputs = {'X': OpTest.np_dtype_to_base_dtype(x)}
        self.outputs = {'Out': out}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(
            ['X'],
            'Out',
            check_prim=True,
            check_pir=True,
            check_prim_pir=True,
            check_pir_onednn=self.check_pir_onednn,
        )

    def test_check_output(self):
        self.check_output(
            check_pir=True,
            check_prim_pir=True,
            check_pir_onednn=self.check_pir_onednn,
            check_symbol_infer=False,
        )

    def init_dtype(self):
        self.dtype = np.float32

    def if_enable_cinn(self):
        pass


class TestSqrt_ZeroDim(TestSqrt):
    def init_shape(self):
        self.shape = []


class TestSqrt_Complex64(TestSqrt):
    def init_dtype(self):
        self.dtype = np.complex64


class TestSqrt_Complex128(TestSqrt):
    def init_dtype(self):
        self.dtype = np.complex128


@unittest.skipIf(
    not core.is_compiled_with_cuda() or core.is_compiled_with_rocm(),
    "core is not compiled with CUDA",
)
class TestSqrtBF16(OpTest):
    def setUp(self):
        self.op_type = "sqrt"
        self.prim_op_type = "prim"
        self.python_api = paddle.sqrt
        self.public_python_api = paddle.sqrt
        self.init_dtype()
        self.init_shape()
        self.if_enable_cinn()

        np.random.seed(1023)
        x = np.random.uniform(0.1, 1, self.shape).astype(np.float32)
        out = np.sqrt(x)

        self.inputs = {
            'X': OpTest.np_dtype_to_base_dtype(convert_float_to_uint16(x))
        }
        self.outputs = {'Out': convert_float_to_uint16(out)}

    def init_dtype(self):
        self.dtype = np.uint16

    def init_shape(self):
        self.shape = [11, 17]

    def if_enable_cinn(self):
        self.enable_cinn = False

    def test_check_output(self):
        place = core.CUDAPlace(0)
        self.check_output_with_place(
            place,
            check_pir=True,
            check_prim_pir=True,
            check_pir_onednn=self.check_pir_onednn,
            check_symbol_infer=False,
        )

    def test_check_grad(self):
        place = core.CUDAPlace(0)
        self.check_grad_with_place(
            place,
            ['X'],
            'Out',
            check_prim=True,
            check_pir=True,
            check_prim_pir=True,
        )


class TestSqrtComp(TestActivation, TestParameter):
    def setUp(self):
        self.op_type = "sqrt"
        self.prim_op_type = "prim"
        self.python_api = paddle.sqrt
        self.public_python_api = paddle.sqrt
        self.init_dtype()
        self.init_shape()
        self.if_enable_cinn()

        np.random.seed(1023)
        x = np.random.uniform(0.1, 1, self.shape).astype(self.dtype)
        out = np.sqrt(x)

        self.inputs = {'X': OpTest.np_dtype_to_base_dtype(x)}
        self.outputs = {'Out': out}
        self.convert_input_output()

    def if_enable_cinn(self):
        pass

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(
            ['X'],
            'Out',
            check_dygraph=True,
            check_prim=True,
            check_pir=True,
            check_prim_pir=True,
            check_pir_onednn=self.check_pir_onednn,
        )

    def test_check_output(self):
        self.check_output(
            check_dygraph=True,
            check_prim=True,
            check_pir=True,
            check_prim_pir=True,
            check_pir_onednn=self.check_pir_onednn,
            check_symbol_infer=False,
        )


class TestSqrtCompFp32(TestActivation):
    def setUp(self):
        self.op_type = "sqrt"
        self.prim_op_type = "prim"
        self.python_api = paddle.sqrt
        self.public_python_api = paddle.sqrt
        self.init_dtype()
        self.init_shape()
        self.if_enable_cinn()
        np.random.seed(1023)
        x = np.random.uniform(0.1, 1, self.shape).astype(self.dtype)
        out = np.sqrt(x)

        self.inputs = {'X': OpTest.np_dtype_to_base_dtype(x)}
        self.outputs = {'Out': out}

    def if_enable_cinn(self):
        pass

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(
            ['X'],
            'Out',
            check_dygraph=True,
            check_prim=True,
            check_pir=True,
            check_prim_pir=True,
            check_pir_onednn=self.check_pir_onednn,
        )

    def test_check_output(self):
        self.check_output(
            check_dygraph=True,
            check_prim=True,
            check_pir=True,
            check_prim_pir=True,
            check_pir_onednn=self.check_pir_onednn,
            check_symbol_infer=False,
        )

    def init_dtype(self):
        self.dtype = np.float32


class TestSqrtComp_ZeroDim(TestSqrtComp):
    def init_shape(self):
        self.shape = []


class TestRsqrt(TestActivation):
    def setUp(self):
        self.op_type = "rsqrt"
        self.python_api = paddle.rsqrt
        self.public_python_api = paddle.rsqrt
        self.prim_op_type = "prim"
        self.init_dtype()
        self.init_shape()
        self.if_enable_cinn()

        np.random.seed(1024)
        x = np.random.uniform(0.1, 1, self.shape).astype(self.dtype)
        out = 1.0 / np.sqrt(x)

        self.inputs = {'X': OpTest.np_dtype_to_base_dtype(x)}
        self.outputs = {'Out': out}
        self.convert_input_output()

    def init_shape(self):
        self.shape = [10, 12]

    def if_enable_cinn(self):
        pass

    def test_check_output(self):
        self.check_output(
            check_pir=True,
            check_pir_onednn=self.check_pir_onednn,
            check_symbol_infer=False,
        )

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(
            ['X'],
            'Out',
            max_relative_error=0.0005,
            check_prim=True,
            check_pir=True,
            check_prim_pir=True,
            check_pir_onednn=self.check_pir_onednn,
        )


class TestRsqrt_ZeroDim(TestRsqrt):
    def init_shape(self):
        self.shape = []

    def if_enable_cinn(self):
        self.enable_cinn = False


class TestAbs(TestActivation):
    def setUp(self):
        self.op_type = "abs"
        self.prim_op_type = "prim"
        self.python_api = paddle.abs
        self.public_python_api = paddle.abs
        self.init_dtype()
        self.init_shape()
        self.if_enable_cinn()

        np.random.seed(1024)
        x = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        # Because we set delta = 0.005 in calculating numeric gradient,
        # if x is too small, such as 0.002, x_neg will be -0.003
        # x_pos will be 0.007, so the numeric gradient is inaccurate.
        # we should avoid this
        x[np.abs(x) < 0.005] = 0.02
        out = np.abs(x)

        self.inputs = {'X': OpTest.np_dtype_to_base_dtype(x)}
        self.outputs = {'Out': out}
        self.convert_input_output()

    def init_shape(self):
        self.shape = [4, 25]

    def if_enable_cinn(self):
        pass

    def test_check_output(self):
        self.check_output(
            check_pir=True,
            check_pir_onednn=self.check_pir_onednn,
            check_symbol_infer=False,
        )

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(
            ['X'],
            'Out',
            check_prim=True,
            check_pir=True,
            check_prim_pir=True,
            check_pir_onednn=self.check_pir_onednn,
        )


class TestAbs_ZeroDim(TestAbs):
    def init_shape(self):
        self.shape = []


class TestCeil(TestActivation):
    def setUp(self):
        self.op_type = "ceil"
        self.prim_op_type = "prim"
        self.python_api = paddle.ceil
        self.public_python_api = paddle.ceil
        self.init_dtype()
        self.init_shape()

        np.random.seed(1024)
        x = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        out = np.ceil(x)

        self.inputs = {'X': OpTest.np_dtype_to_base_dtype(x)}
        self.outputs = {'Out': out}
        self.convert_input_output()

    def init_shape(self):
        self.shape = [10, 12]

    def test_check_output(self):
        self.check_output(
            check_pir=True,
            check_pir_onednn=self.check_pir_onednn,
            check_symbol_infer=False,
        )

    # The same reason with TestFloor
    def test_check_grad(self):
        pass

    def test_check_grad_for_prim(self):
        # the gradient on floor, ceil, round is undefined.
        # we return zero as gradient, but the numpy return nan.
        # for prim, we compare result with eager python api,
        # so, we use only_prim flag to express we only test prim.
        if core.is_compiled_with_cuda():
            self.check_grad_with_place(
                paddle.CUDAPlace(0),
                ['X'],
                'Out',
                check_pir=True,
                check_prim_pir=True,
            )


class TestCeil_ZeroDim(TestCeil):
    def init_shape(self):
        self.shape = []


class TestFloor(TestActivation):
    def setUp(self):
        self.op_type = "floor"
        self.prim_op_type = "prim"
        self.python_api = paddle.floor
        self.public_python_api = paddle.floor
        self.init_dtype()
        self.init_shape()
        self.if_enable_cinn()

        np.random.seed(1024)
        x = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        out = np.floor(x)

        self.inputs = {'X': OpTest.np_dtype_to_base_dtype(x)}
        self.outputs = {'Out': out}
        self.convert_input_output()

    def init_shape(self):
        self.shape = [10, 12]

    def if_enable_cinn(self):
        pass

    def test_check_output(self):
        self.check_output(
            check_pir=True,
            check_pir_onednn=self.check_pir_onednn,
            check_symbol_infer=False,
        )

    # the gradient on floor, ceil, round is undefined.
    # we return zero as gradient, but the numpy return nan
    # The same reason with TestFloor
    def test_check_grad(self):
        pass

    def test_check_grad_for_prim(self):
        # the gradient on floor, ceil, round is undefined.
        # we return zero as gradient, but the numpy return nan.
        # for prim, we compare result with eager python api,
        # so, we use only_prim flag to express we only test prim.
        if core.is_compiled_with_cuda():
            self.check_grad_with_place(
                paddle.CUDAPlace(0),
                ['X'],
                'Out',
                check_prim=True,
                only_check_prim=True,
                check_pir=True,
                check_prim_pir=True,
            )


class TestFloor_ZeroDim(TestFloor):
    def init_shape(self):
        self.shape = []


class TestCos(TestActivation):
    def setUp(self):
        self.op_type = "cos"
        self.python_api = paddle.cos
        self.public_python_api = paddle.cos
        self.prim_op_type = "prim"
        self.init_dtype()
        self.init_shape()
        self.if_enable_cinn()

        np.random.seed(1024)
        x = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        if self.dtype == np.complex64 or self.dtype == np.complex128:
            x = (
                np.random.uniform(-1, 1, self.shape)
                + 1j * np.random.uniform(-1, 1, self.shape)
            ).astype(self.dtype)
        out = np.cos(x)
        self.inputs = {'X': OpTest.np_dtype_to_base_dtype(x)}
        self.outputs = {'Out': out}
        self.convert_input_output()

    def init_shape(self):
        self.shape = [10, 12]

    def test_check_output(self):
        self.check_output(
            check_pir=True,
            check_pir_onednn=self.check_pir_onednn,
            check_symbol_infer=False,
        )

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        # TODO(ScottWong98): set `check_prim=False` when `fill_any_like` supports `complex` dtype
        if self.dtype == np.complex64 or self.dtype == np.complex128:
            # Complex64 [GPU]: AssertionError: 0.0057843705 not less than or equal to 0.005
            self.check_grad(
                ['X'],
                'Out',
                check_prim=False,
                max_relative_error=0.006,
                check_pir=True,
                check_pir_onednn=self.check_pir_onednn,
            )
        else:
            self.check_grad(
                ['X'],
                'Out',
                check_prim=True,
                check_pir=True,
                check_prim_pir=True,
                check_pir_onednn=self.check_pir_onednn,
            )

    def if_enable_cinn(self):
        pass


class TestCos_Complex64(TestCos):
    def init_dtype(self):
        self.dtype = np.complex64


class TestCos_Complex128(TestCos):
    def init_dtype(self):
        self.dtype = np.complex128


class TestCos_ZeroDim(TestCos):
    def init_shape(self):
        self.shape = []


class TestTan(TestActivation):
    def setUp(self):
        np.random.seed(1024)
        self.op_type = "tan"
        self.python_api = paddle.tan
        self.init_dtype()
        self.init_shape()

        self.x_np = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        if self.dtype == np.complex64 or self.dtype == np.complex128:
            self.x_np = (
                np.random.uniform(-1, 1, self.shape)
                + 1j * np.random.uniform(-1, 1, self.shape)
            ).astype(self.dtype)
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

        out = np.tan(self.x_np)

        self.inputs = {'X': OpTest.np_dtype_to_base_dtype(self.x_np)}
        self.outputs = {'Out': out}
        self.convert_input_output()

    def init_shape(self):
        self.shape = [10, 12]

    def test_check_output(self):
        self.check_output(
            check_pir=True,
            check_pir_onednn=self.check_pir_onednn,
            check_symbol_infer=False,
        )

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(
            ['X'], 'Out', check_pir=True, check_pir_onednn=self.check_pir_onednn
        )


class TestTan_float32(TestTan):
    def init_dtype(self):
        self.dtype = "float32"


class TestTan_Complex64(TestTan):
    def init_dtype(self):
        self.dtype = np.complex64


class TestTan_Complex128(TestTan):
    def init_dtype(self):
        self.dtype = np.complex128


class TestTan_ZeroDim(TestTan):
    def init_shape(self):
        self.shape = []


class TestTanAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(1024)
        self.dtype = 'float32'
        self.x_np = np.random.uniform(-1, 1, [11, 17]).astype(self.dtype)
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    def test_dygraph_api(self):
        with dynamic_guard():
            x = paddle.to_tensor(self.x_np)
            out_test = paddle.tan(x)
            out_ref = np.tan(self.x_np)
            np.testing.assert_allclose(out_ref, out_test.numpy(), rtol=1e-05)

    def test_static_api(self):
        with static_guard():
            with paddle.static.program_guard(paddle.static.Program()):
                x = paddle.static.data('X', [11, 17], self.dtype)
                out = paddle.tan(x)
                exe = paddle.static.Executor(self.place)
                res = exe.run(feed={'X': self.x_np}, fetch_list=[out])
            out_ref = np.tan(self.x_np)
            np.testing.assert_allclose(out_ref, res[0], rtol=1e-05)

    def test_backward(self):
        test_data_shape = [11, 17]
        with base.dygraph.guard():
            input_x = np.random.uniform(0.1, 1, test_data_shape).astype(
                "float32"
            )
            var = paddle.to_tensor(input_x)
            var.stop_gradient = False
            loss = paddle.tan(var)
            loss.backward()
            grad_var = var.gradient()
            self.assertEqual(grad_var.shape, input_x.shape)


class TestAcos(TestActivation):
    def setUp(self):
        self.op_type = "acos"
        self.python_api = paddle.acos
        self.init_dtype()
        self.init_shape()

        np.random.seed(1024)
        x = np.random.uniform(-0.95, 0.95, self.shape).astype(self.dtype)
        if self.dtype == np.complex64 or self.dtype == np.complex128:
            x = (
                np.random.uniform(-0.95, 0.95, self.shape)
                + 1j * np.random.uniform(-0.95, 0.95, self.shape)
            ).astype(self.dtype)
        out = np.arccos(x)

        self.inputs = {'X': OpTest.np_dtype_to_base_dtype(x)}
        self.outputs = {'Out': out}
        self.convert_input_output()

    def init_shape(self):
        self.shape = [10, 12]

    def test_check_output(self):
        self.check_output(
            check_pir=True,
            check_pir_onednn=self.check_pir_onednn,
            check_symbol_infer=False,
        )

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(
            ['X'], 'Out', check_pir=True, check_pir_onednn=self.check_pir_onednn
        )


class TestAcos_Complex64(TestAcos):
    def init_dtype(self):
        self.dtype = np.complex64


class TestAcos_Complex128(TestAcos):
    def init_dtype(self):
        self.dtype = np.complex128


class TestAcos_ZeroDim(TestAcos):
    def init_shape(self):
        self.shape = []


class TestSin(TestActivation, TestParameter):
    def setUp(self):
        self.op_type = "sin"
        self.python_api = paddle.sin
        self.public_python_api = paddle.sin
        self.prim_op_type = "prim"
        self.init_dtype()
        self.init_shape()
        self.if_enable_cinn()

        np.random.seed(1024)
        x = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        if self.dtype == np.complex64 or self.dtype == np.complex128:
            x = (
                np.random.uniform(-1, 1, self.shape)
                + 1j * np.random.uniform(-1, 1, self.shape)
            ).astype(self.dtype)
        out = np.sin(x)
        self.inputs = {'X': OpTest.np_dtype_to_base_dtype(x)}
        self.outputs = {'Out': out}
        self.convert_input_output()

    def init_shape(self):
        self.shape = [10, 12]

    def test_out_name(self):
        # inherit from `TestParameter`
        super().test_out_name()

    def test_check_output(self):
        self.check_output(
            check_pir=True,
            check_pir_onednn=self.check_pir_onednn,
            check_symbol_infer=False,
        )

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        # TODO(ScottWong98): set `check_prim=False` when `fill_any_like` supports `complex` dtype
        if self.dtype == np.complex64 or self.dtype == np.complex128:
            self.check_grad(
                ['X'],
                'Out',
                check_prim=False,
                check_pir=True,
                check_pir_onednn=self.check_pir_onednn,
            )
        else:
            self.check_grad(
                ['X'],
                'Out',
                check_prim=True,
                check_pir=True,
                check_prim_pir=True,
                check_pir_onednn=self.check_pir_onednn,
            )

    def if_enable_cinn(self):
        pass


class TestSin_Complex64(TestSin):
    def init_dtype(self):
        self.dtype = np.complex64


class TestSin_Complex128(TestSin):
    def init_dtype(self):
        self.dtype = np.complex128


class TestSin_ZeroDim(TestSin):
    def init_shape(self):
        self.shape = []


class TestAsin(TestActivation):
    def setUp(self):
        self.op_type = "asin"
        self.python_api = paddle.asin
        self.init_dtype()
        self.init_shape()

        np.random.seed(2048)
        x = np.random.uniform(-0.95, 0.95, self.shape).astype(self.dtype)
        if self.dtype == np.complex64 or self.dtype == np.complex128:
            x = (
                np.random.uniform(-0.95, 0.95, self.shape)
                + 1j * np.random.uniform(-0.95, 0.95, self.shape)
            ).astype(self.dtype)
        out = np.arcsin(x)

        self.inputs = {'X': OpTest.np_dtype_to_base_dtype(x)}
        self.outputs = {'Out': out}
        self.convert_input_output()

    def init_shape(self):
        self.shape = [10, 12]

    def test_check_output(self):
        self.check_output(
            check_pir=True,
            check_pir_onednn=self.check_pir_onednn,
            check_symbol_infer=False,
        )

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(
            ['X'], 'Out', check_pir=True, check_pir_onednn=self.check_pir_onednn
        )


class TestAsin_Complex64(TestAsin):
    def init_dtype(self):
        self.dtype = np.complex64


class TestAsin_Complex128(TestAsin):
    def init_dtype(self):
        self.dtype = np.complex128


class TestAsin_ZeroDim(TestAsin):
    def init_shape(self):
        self.shape = []


class TestAcosh(TestActivation):
    def setUp(self):
        self.op_type = "acosh"
        self.python_api = paddle.acosh
        self.init_dtype()
        self.init_shape()

        np.random.seed(1024)
        x = np.random.uniform(2, 3, self.shape).astype(self.dtype)
        if self.dtype == np.complex64 or self.dtype == np.complex128:
            x = (
                np.random.uniform(2, 3, self.shape)
                + 1j * np.random.uniform(2, 3, self.shape)
            ).astype(self.dtype)
        out = np.arccosh(x)

        self.inputs = {'X': OpTest.np_dtype_to_base_dtype(x)}
        self.outputs = {'Out': out}
        self.convert_input_output()

    def init_shape(self):
        self.shape = [10, 12]

    def test_check_output(self):
        self.check_output(
            check_pir=True,
            check_pir_onednn=self.check_pir_onednn,
            check_symbol_infer=False,
        )

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        if self.dtype == np.complex64:
            # Complex64[CPU]: AssertionError: 0.012431525 not less than or equal to 0.005
            self.check_grad(
                ['X'],
                'Out',
                max_relative_error=0.02,
                check_pir=True,
                check_pir_onednn=self.check_pir_onednn,
            )
        else:
            self.check_grad(
                ['X'],
                'Out',
                check_pir=True,
                check_pir_onednn=self.check_pir_onednn,
            )


class TestAcosh_Complex64(TestAcosh):
    def init_dtype(self):
        self.dtype = np.complex64


class TestAcosh_Complex128(TestAcosh):
    def init_dtype(self):
        self.dtype = np.complex128


class TestAcosh_ZeroDim(TestAcosh):
    def init_shape(self):
        self.shape = []


class TestAsinh(TestActivation):
    def setUp(self):
        self.op_type = "asinh"
        self.python_api = paddle.asinh
        self.init_dtype()
        self.init_shape()

        np.random.seed(1024)
        x = np.random.uniform(1, 2, self.shape).astype(self.dtype)
        if self.dtype == np.complex64 or self.dtype == np.complex128:
            x = (
                np.random.uniform(1, 2, self.shape)
                + 1j * np.random.uniform(1, 2, self.shape)
            ).astype(self.dtype)
        out = np.arcsinh(x)

        self.inputs = {'X': OpTest.np_dtype_to_base_dtype(x)}
        self.outputs = {'Out': out}
        self.convert_input_output()

    def init_shape(self):
        self.shape = [10, 12]

    def test_check_output(self):
        self.check_output(
            check_pir=True,
            check_pir_onednn=self.check_pir_onednn,
            check_symbol_infer=False,
        )

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        if self.dtype == np.complex64 or self.dtype == np.complex128:
            # Complex64 [CPU]: AssertionError: 0.006898686 not less than or equal to 0.005
            self.check_grad(
                ['X'],
                'Out',
                max_relative_error=0.007,
                check_pir=True,
                check_pir_onednn=self.check_pir_onednn,
            )
        else:
            self.check_grad(
                ['X'],
                'Out',
                check_pir=True,
                check_pir_onednn=self.check_pir_onednn,
            )


class TestAsinh_Complex64(TestAsinh):
    def init_dtype(self):
        self.dtype = np.complex64


class TestAsinh_Complex128(TestAsinh):
    def init_dtype(self):
        self.dtype = np.complex128


class TestAsinh_ZeroDim(TestAsinh):
    def init_shape(self):
        self.shape = []


class TestAtanh(TestActivation):
    def setUp(self):
        self.op_type = "atanh"
        self.python_api = paddle.atanh
        self.init_dtype()
        self.init_shape()

        np.random.seed(400)
        x = np.random.uniform(-0.9, 0.9, self.shape).astype(self.dtype)
        if self.dtype == np.complex64 or self.dtype == np.complex128:
            x = (
                np.random.uniform(-0.9, 0.9, self.shape)
                + 1j * np.random.uniform(-0.9, 0.9, self.shape)
            ).astype(self.dtype)
        out = np.arctanh(x)

        self.inputs = {'X': OpTest.np_dtype_to_base_dtype(x)}
        self.outputs = {'Out': out}
        self.convert_input_output()

    def init_shape(self):
        self.shape = [10, 12]

    def test_check_output(self):
        self.check_output(
            check_pir=True,
            check_pir_onednn=self.check_pir_onednn,
            check_symbol_infer=False,
        )

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(
            ['X'], 'Out', check_pir=True, check_pir_onednn=self.check_pir_onednn
        )


class TestAtanh_Complex64(TestAtanh):
    def init_dtype(self):
        self.dtype = np.complex64


class TestAtanh_Complex128(TestAtanh):
    def init_dtype(self):
        self.dtype = np.complex128


class TestAtanh_ZeroDim(TestAtanh):
    def init_shape(self):
        self.shape = []


class TestRound(TestActivation):
    def setUp(self):
        self.op_type = "round"
        self.python_api = paddle.round
        self.init_dtype()
        self.init_shape()
        self.init_decimals()

        np.random.seed(1024)
        x = np.random.uniform(-1, 1, self.shape).astype(self.dtype) * 100
        out = np.round(x, decimals=self.decimals)

        self.inputs = {'X': OpTest.np_dtype_to_base_dtype(x)}
        self.outputs = {'Out': out}
        self.attrs = {'decimals': self.decimals}
        self.convert_input_output()

    def init_shape(self):
        self.shape = [10, 12]

    def init_decimals(self):
        self.decimals = 0

    def test_check_output(self):
        self.check_output(
            check_pir=True,
            check_pir_onednn=self.check_pir_onednn,
            check_symbol_infer=False,
        )

    def test_check_grad(self):
        pass


class TestRound_ZeroDim(TestRound):
    def init_shape(self):
        self.shape = []


class TestRound_decimals1(TestRound):
    def init_decimals(self):
        self.decimals = 2

    def test_round_api(self):
        with dynamic_guard():
            for device in devices:
                if device == 'cpu' or (
                    device == 'gpu' and paddle.is_compiled_with_cuda()
                ):
                    x_np = (
                        np.random.uniform(-1, 1, self.shape).astype(self.dtype)
                        * 100
                    )
                    out_expect = np.round(x_np, decimals=self.decimals)
                    x_paddle = paddle.to_tensor(
                        x_np, dtype=self.dtype, place=device
                    )
                    y = paddle.round(x_paddle, decimals=self.decimals)
                    np.testing.assert_allclose(y.numpy(), out_expect, rtol=1e-3)


class TestRound_decimals2(TestRound_decimals1):
    def init_decimals(self):
        self.decimals = -1


class TestRelu(TestActivation):
    def setUp(self):
        self.op_type = "relu"
        self.python_api = paddle.nn.functional.relu
        self.prim_op_type = "comp"
        self.public_python_api = paddle.nn.functional.relu
        self.init_dtype()
        self.init_shape()
        self.if_enable_cinn()

        np.random.seed(1024)
        x = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        # The same reason with TestAbs
        x[np.abs(x) < 0.005] = 0.02
        out = np.maximum(x, 0)
        self.inputs = {'X': x}

        self.outputs = {'Out': out}
        self.convert_input_output()

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(
            ['X'],
            'Out',
            check_prim=True,
            check_pir=True,
            check_prim_pir=True,
            check_pir_onednn=self.check_pir_onednn,
        )

    def test_check_output(self):
        self.check_output(
            check_prim=True,
            check_pir=True,
            check_prim_pir=True,
            check_pir_onednn=self.check_pir_onednn,
            check_symbol_infer=False,
        )

    def if_enable_cinn(self):
        pass


class TestRelu_ZeroDim(TestRelu):
    def init_shape(self):
        self.shape = []


class TestRelu_NanInput(TestActivation):
    def setUp(self):
        self.init_dtype()
        self.init_shape()
        self.if_enable_cinn()

        np.random.seed(1024)
        x = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        # The same reason with TestAbs
        x[np.abs(x) < 0.005] = 0.02
        x[-1] = float('nan')
        tensor_x = paddle.to_tensor(x)
        out = paddle.nn.functional.relu(tensor_x)
        self.outputs_paddle = out

    def test_check_output(self):
        self.assertTrue(
            paddle.isnan(self.outputs_paddle).cast('int32').sum() > 0
        )

    def test_check_grad(self):
        pass


class TestReluAPI(unittest.TestCase):
    # test paddle.nn.ReLU, paddle.nn.functional.relu
    def setUp(self):
        np.random.seed(1024)
        self.x_np = np.random.uniform(-1, 1, [10, 12]).astype('float32')
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )
        self.executed_api()

    def executed_api(self):
        self.relu = F.relu

    def test_static_api(self):
        with static_guard():
            with paddle.static.program_guard(paddle.static.Program()):
                x = paddle.static.data('X', [10, 12], dtype="float32")
                out1 = self.relu(x)
                m = paddle.nn.ReLU()
                out2 = m(x)
                exe = paddle.static.Executor(self.place)
                res = exe.run(feed={'X': self.x_np}, fetch_list=[out1, out2])
            out_ref = np.maximum(self.x_np, 0)
            for r in res:
                np.testing.assert_allclose(out_ref, r, rtol=1e-05)

    def test_dygraph_api(self):
        with dynamic_guard():
            x = paddle.to_tensor(self.x_np)
            m = paddle.nn.ReLU()
            out1 = m(x)
            out2 = self.relu(x)
            out_ref = np.maximum(self.x_np, 0)
            for r in [out1, out2]:
                np.testing.assert_allclose(out_ref, r.numpy(), rtol=1e-05)

    def test_errors(self):
        with static_guard():
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                # The input type must be Variable.
                self.assertRaises(TypeError, self.relu, 1)
                # The input dtype must be float16, float32, float64.
                x_int32 = paddle.static.data(
                    name='x_int32', shape=[10, 12], dtype='int32'
                )
                self.assertRaises(TypeError, self.relu, x_int32)
                # support the input dtype is float16
                x_fp16 = paddle.static.data(
                    name='x_fp16', shape=[10, 12], dtype='float16'
                )
                self.relu(x_fp16)


class TestReluInplaceAPI(TestReluAPI):
    # test paddle.nn.functional.relu_
    def executed_api(self):
        self.relu = F.relu_


def ref_leaky_relu(x, alpha=0.01):
    out = np.copy(x)
    out[out < 0] *= alpha
    return out


class TestLeakyRelu(TestActivation):
    def get_alpha(self):
        return 0.02

    def setUp(self):
        self.op_type = "leaky_relu"
        self.python_api = paddle.nn.functional.leaky_relu
        self.public_python_api = paddle.nn.functional.leaky_relu
        self.prim_op_type = "comp"
        self.init_dtype()
        self.init_shape()
        self.if_enable_cinn()
        alpha = self.get_alpha()

        np.random.seed(1024)
        x = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        # The same reason with TestAbs
        x[np.abs(x) < 0.005] = 0.05
        out = ref_leaky_relu(x, alpha)

        self.inputs = {'X': x}
        self.outputs = {'Out': out}
        self.attrs = {'alpha': alpha}
        self.convert_input_output()

    def if_enable_cinn(self):
        pass

    def test_check_output(self):
        self.check_output(
            check_prim=True,
            check_pir=True,
            check_prim_pir=True,
            check_pir_onednn=self.check_pir_onednn,
            check_symbol_infer=False,
        )

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(
            ['X'],
            'Out',
            check_prim=True,
            check_pir=True,
            check_prim_pir=True,
            check_pir_onednn=self.check_pir_onednn,
        )


class TestLeakyReluAlpha1(TestLeakyRelu):
    def get_alpha(self):
        return 2


class TestLeakyReluAlpha2(TestLeakyRelu):
    def get_alpha(self):
        return -0.01


class TestLeakyReluAlpha3(TestLeakyRelu):
    def get_alpha(self):
        return -2.0


class TestLeakyRelu_ZeroDim(TestLeakyRelu):
    def init_shape(self):
        self.shape = []

    def if_enable_cinn(self):
        pass


class TestLeakyReluAPI(unittest.TestCase):
    # test paddle.nn.LeakyReLU, paddle.nn.functional.leaky_relu,
    def setUp(self):
        np.random.seed(1024)
        self.x_np = np.random.uniform(-1, 1, [10, 12]).astype('float32')
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    def test_static_api(self):
        with static_guard():
            with paddle.static.program_guard(paddle.static.Program()):
                x = paddle.static.data('X', [10, 12], dtype="float32")
                out1 = F.leaky_relu(x)
                m = paddle.nn.LeakyReLU()
                out2 = m(x)
                exe = paddle.static.Executor(self.place)
                res = exe.run(feed={'X': self.x_np}, fetch_list=[out1, out2])
            out_ref = ref_leaky_relu(self.x_np)
            for r in res:
                np.testing.assert_allclose(out_ref, r, rtol=1e-05)

    def test_dygraph_api(self):
        with dynamic_guard():
            x = paddle.to_tensor(self.x_np)
            out1 = F.leaky_relu(x)
            m = paddle.nn.LeakyReLU()
            out2 = m(x)
            out_ref = ref_leaky_relu(self.x_np)
            for r in [out1, out2]:
                np.testing.assert_allclose(out_ref, r.numpy(), rtol=1e-05)

            out1 = F.leaky_relu(x, 0.6)
            m = paddle.nn.LeakyReLU(0.6)
            out2 = m(x)
            out_ref = ref_leaky_relu(self.x_np, 0.6)
            for r in [out1, out2]:
                np.testing.assert_allclose(out_ref, r.numpy(), rtol=1e-05)

    def test_errors(self):
        with static_guard():
            with paddle.static.program_guard(paddle.static.Program()):
                # The input type must be Variable.
                self.assertRaises(TypeError, F.leaky_relu, 1)
                # The input dtype must be float16, float32, float64.
                x_int32 = paddle.static.data(
                    name='x_int32', shape=[12, 10], dtype='int32'
                )
                self.assertRaises(TypeError, F.leaky_relu, x_int32)
                # support the input dtype is float16
                x_fp16 = paddle.static.data(
                    name='x_fp16', shape=[12, 10], dtype='float16'
                )
                F.leaky_relu(x_fp16)


def gelu(x, approximate):
    if approximate:
        y_ref = (
            0.5
            * x
            * (
                1.0
                + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3)))
            )
        )
    else:
        y_ref = 0.5 * x * (1 + erf(x / np.sqrt(2)))
    return y_ref.astype(x.dtype)


class TestGeluApproximate(TestActivation):
    def setUp(self):
        self.op_type = "gelu"
        self.prim_op_type = "comp"
        self.python_api = paddle.nn.functional.gelu
        self.public_python_api = paddle.nn.functional.gelu
        self.init_dtype()
        self.init_shape()
        approximate = True
        np.random.seed(1024)
        x = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        out = gelu(x, approximate)

        self.inputs = {'X': x}
        self.outputs = {'Out': out}
        self.attrs = {"approximate": approximate}

        # The backward decomposite of gelu is inconsistent with raw kernel on
        # cpu device, lower threshold to support 1e-8 for pass the unittest
        self.rev_comp_rtol = 1e-8
        self.rev_comp_atol = 1e-8
        # Cumulative error occurs between comp and cinn, so that we also set cinn_rtol to 1e-8 as rev_comp_rtol = 1e-8
        self.cinn_rtol = 1e-8
        self.cinn_atol = 1e-8

    def test_check_output(self):
        self.check_output(
            check_prim=True,
            check_pir=True,
            check_prim_pir=False,
            check_pir_onednn=self.check_pir_onednn,
            check_symbol_infer=False,
        )

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(
            ['X'],
            'Out',
            check_prim=True,
            check_pir=True,
            check_prim_pir=True,
            check_pir_onednn=self.check_pir_onednn,
        )


class TestGelu(TestActivation):
    def setUp(self):
        self.op_type = "gelu"
        self.prim_op_type = "comp"
        self.python_api = paddle.nn.functional.gelu
        self.public_python_api = paddle.nn.functional.gelu
        self.init_dtype()
        self.init_shape()
        # Todo: Under float64, only this accuracy is currently supported, for further processing
        self.fw_comp_rtol = 1e-7
        approximate = False
        np.random.seed(2048)
        x = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        out = gelu(x, approximate)
        self.if_enable_cinn()

        self.inputs = {'X': OpTest.np_dtype_to_base_dtype(x)}
        self.outputs = {'Out': out}
        self.convert_input_output()
        self.attrs = {"approximate": approximate}
        # The backward decomposite of gelu is inconsistent with raw kernel on
        # cpu, lower threshold to support 1e-8 for pass the unittest
        self.rev_comp_rtol = 1e-8
        self.rev_comp_atol = 1e-8
        # Cumulative error occurs between comp and cinn, so that we also set cinn_rtol to 1e-8 as rev_comp_rtol = 1e-8
        self.cinn_rtol = 1e-8
        self.cinn_atol = 1e-8

    def if_enable_cinn(self):
        pass

    def test_check_output(self):
        self.check_output(
            check_prim=True,
            check_pir=True,
            check_prim_pir=True,
            check_pir_onednn=self.check_pir_onednn,
            check_symbol_infer=False,
        )

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(
            ['X'],
            'Out',
            check_prim=True,
            check_pir=True,
            check_prim_pir=True,
            check_pir_onednn=self.check_pir_onednn,
        )


class TestGelu_ZeroDim(TestGelu):
    def init_shape(self):
        self.shape = []


class TestGELUAPI(unittest.TestCase):
    # test paddle.nn.GELU, paddle.nn.functional.gelu
    def setUp(self):
        np.random.seed(1024)
        self.x_np = np.random.uniform(-1, 1, [11, 17]).astype('float32')
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )
        self.enable_cinn = False

        # The backward decomposite of gelu is inconsistent with raw kernel on
        # cpu, lower threshold to support 1e-8 for pass the unittest
        self.rev_comp_rtol = 1e-8
        self.rev_comp_atol = 1e-8

    def test_static_api(self):
        with static_guard():
            with paddle.static.program_guard(paddle.static.Program()):
                x = paddle.static.data('X', [11, 17], dtype="float32")
                out1 = F.gelu(x)
                m = paddle.nn.GELU()
                out2 = m(x)
                exe = paddle.static.Executor(self.place)
                res = exe.run(feed={'X': self.x_np}, fetch_list=[out1, out2])
            out_ref = gelu(self.x_np, False)
            for r in res:
                np.testing.assert_allclose(out_ref, r, rtol=1e-05)

    def test_dygraph_api(self):
        with dynamic_guard():
            x = paddle.to_tensor(self.x_np)
            out1 = F.gelu(x)
            m = paddle.nn.GELU()
            out2 = m(x)
            out_ref = gelu(self.x_np, False)
            for r in [out1, out2]:
                np.testing.assert_allclose(out_ref, r.numpy(), rtol=1e-05)

            out1 = F.gelu(x, True)
            m = paddle.nn.GELU(True)
            out2 = m(x)
            out_ref = gelu(self.x_np, True)
            for r in [out1, out2]:
                np.testing.assert_allclose(out_ref, r.numpy(), rtol=1e-05)

    def test_errors(self):
        with static_guard():
            with paddle.static.program_guard(paddle.static.Program()):
                # The input type must be Variable.
                self.assertRaises(TypeError, F.gelu, 1)
                # The input dtype must be float16, float32, float64.
                x_int32 = paddle.static.data(
                    name='x_int32', shape=[11, 17], dtype='int32'
                )
                self.assertRaises(TypeError, F.gelu, x_int32)
                # support the input dtype is float16
                x_fp16 = paddle.static.data(
                    name='x_fp16', shape=[11, 17], dtype='float16'
                )
                F.gelu(x_fp16)


class TestBRelu(TestActivation):
    def setUp(self):
        self.op_type = "brelu"
        self.python_api = paddle.nn.functional.hardtanh
        self.init_dtype()

        np.random.seed(1024)
        x = np.random.uniform(-5, 10, [10, 12]).astype(self.dtype)
        t_min = 1.0
        t_max = 4.0
        # The same with TestAbs
        x[np.abs(x - t_min) < 0.005] = t_min + 0.02
        x[np.abs(x - t_max) < 0.005] = t_max + 0.02
        t = np.copy(x)
        t[t < t_min] = t_min
        t[t > t_max] = t_max

        self.inputs = {'X': OpTest.np_dtype_to_base_dtype(x)}
        self.outputs = {'Out': t}
        self.convert_input_output()
        self.attrs = {'t_min': t_min, 't_max': t_max}

    def test_check_output(self):
        self.check_output(
            check_pir=True,
            check_pir_onednn=self.check_pir_onednn,
            check_symbol_infer=False,
        )

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(
            ['X'], 'Out', check_pir=True, check_pir_onednn=self.check_pir_onednn
        )


def ref_relu6(x, threshold=6.0):
    out = np.copy(x)
    out[np.abs(x - threshold) < 0.005] = threshold + 0.02
    out = np.minimum(np.maximum(x, 0), threshold)
    return out


class TestRelu6(TestActivation):
    def setUp(self):
        self.op_type = "relu6"
        self.init_dtype()
        self.init_shape()
        self.python_api = paddle.nn.functional.relu6
        self.prim_op_type = "comp"
        self.public_python_api = paddle.nn.functional.relu6

        np.random.seed(1024)
        x = np.random.uniform(-1, 10, self.shape).astype(self.dtype)
        x[np.abs(x) < 0.005] = 0.02
        out = ref_relu6(x)

        self.attrs = {'threshold': 6.0}

        self.inputs = {'X': OpTest.np_dtype_to_base_dtype(x)}
        self.outputs = {'Out': out}
        self.convert_input_output()

    def init_shape(self):
        self.shape = [10, 12]

    def test_check_output(self):
        self.check_output(
            check_pir=True,
            check_prim_pir=True,
            check_pir_onednn=self.check_pir_onednn,
            check_symbol_infer=False,
        )

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(
            ['X'],
            'Out',
            check_pir=True,
            check_pir_onednn=self.check_pir_onednn,
            check_prim_pir=True,
        )


class TestRelu6_ZeroDim(TestRelu6):
    def init_shape(self):
        self.shape = []


class TestRelu6API(unittest.TestCase):
    # test paddle.nn.ReLU6, paddle.nn.functional.relu6
    def setUp(self):
        np.random.seed(1024)
        self.x_np = np.random.uniform(-1, 10, [10, 12]).astype(np.float64)
        self.x_np[np.abs(self.x_np) < 0.005] = 0.02
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    def test_static_api(self):
        with static_guard():
            with paddle.static.program_guard(paddle.static.Program()):
                x = paddle.static.data('X', self.x_np.shape, self.x_np.dtype)
                out1 = F.relu6(x)
                relu6 = paddle.nn.ReLU6()
                out2 = relu6(x)
                exe = paddle.static.Executor(self.place)
                res = exe.run(feed={'X': self.x_np}, fetch_list=[out1, out2])
            out_ref = ref_relu6(self.x_np)
            for r in res:
                np.testing.assert_allclose(out_ref, r, rtol=1e-05)

    def test_dygraph_api(self):
        with dynamic_guard():
            x = paddle.to_tensor(self.x_np)
            out1 = F.relu6(x)
            relu6 = paddle.nn.ReLU6()
            out2 = relu6(x)
            out_ref = ref_relu6(self.x_np)
            for r in [out1, out2]:
                np.testing.assert_allclose(out_ref, r.numpy(), rtol=1e-05)

    def test_base_api(self):
        with static_guard():
            with base.program_guard(base.Program()):
                x = paddle.static.data('X', self.x_np.shape, self.x_np.dtype)
                out = paddle.nn.functional.relu6(x)
                exe = base.Executor(self.place)
                res = exe.run(feed={'X': self.x_np}, fetch_list=[out])
            out_ref = ref_relu6(self.x_np)
            np.testing.assert_allclose(out_ref, res[0], rtol=1e-05)

    def test_errors(self):
        with static_guard():
            with paddle.static.program_guard(paddle.static.Program()):
                # The input type must be Variable.
                self.assertRaises(TypeError, F.relu6, 1)
                # The input dtype must be float16, float32, float64.
                x_int32 = paddle.static.data(
                    name='x_int32', shape=[12, 10], dtype='int32'
                )
                self.assertRaises(TypeError, F.relu6, x_int32)
                # support the input dtype is float16
                if paddle.is_compiled_with_cuda():
                    x_fp16 = paddle.static.data(
                        name='x_fp16', shape=[12, 10], dtype='float16'
                    )
                    F.relu6(x_fp16)


class TestRelu6APIWarnings(unittest.TestCase):
    def test_warnings(self):
        with static_guard():
            with warnings.catch_warnings(record=True) as context:
                warnings.simplefilter("always")

                helper = LayerHelper("relu6")
                data = paddle.static.data(
                    name='data', shape=[None, 3, 32, 32], dtype='float32'
                )
                out = helper.create_variable_for_type_inference(
                    dtype=data.dtype
                )
                os.environ['FLAGS_print_extra_attrs'] = "1"
                helper.append_op(
                    type="relu6",
                    inputs={'X': data},
                    outputs={'Out': out},
                    attrs={'threshold': 6.0},
                )
                self.assertTrue(
                    "op relu6 use extra_attr: threshold"
                    in str(context[-1].message)
                )
                os.environ['FLAGS_print_extra_attrs'] = "0"


def ref_hardswish(x, threshold=6.0, scale=6.0, offset=3.0):
    x_dtype = x.dtype
    if x_dtype == 'float16':
        x_dtype = 'float16'
        x = x.astype('float32')
    return (
        x * np.minimum(np.maximum(x + offset, 0.0), threshold) / scale
    ).astype(x_dtype)


class TestHardSwish(TestActivation):
    def setUp(self):
        self.op_type = 'hard_swish'
        self.init_dtype()
        self.init_shape()
        self.prim_op_type = "comp"
        self.python_api = paddle.nn.functional.hardswish
        self.public_python_api = paddle.nn.functional.hardswish

        np.random.seed(1024)
        if self.dtype is np.complex64 or self.dtype is np.complex128:
            x = (
                np.random.uniform(-6, 6, self.shape)
                + 1j * np.random.uniform(-6, 6, self.shape)
            ).astype(self.dtype)
        else:
            x = np.random.uniform(-6, 6, self.shape).astype(self.dtype)
        threshold = 6.0
        scale = 6.0
        offset = 3.0
        # the same with TestAbs
        x[np.abs(x + offset) < 0.005] = 0.02
        x[np.abs(x - threshold + offset) < 0.005] = threshold - offset + 0.02
        out = ref_hardswish(x, threshold, scale, offset)

        self.inputs = {'X': OpTest.np_dtype_to_base_dtype(x)}
        self.outputs = {'Out': out}
        self.convert_input_output()
        self.attrs = {'threshold': threshold, 'scale': scale, 'offset': offset}

    def init_shape(self):
        self.shape = [10, 12]

    def if_only_check_prim(self):
        return False

    def test_check_grad(self):
        self.check_grad(
            ['X'],
            'Out',
            check_prim=(
                True
                if self.dtype not in [np.complex64, np.complex128]
                else False
            ),
            only_check_prim=self.if_only_check_prim(),
            check_pir=True,
            check_prim_pir=(
                True
                if self.dtype not in [np.complex64, np.complex128]
                else False
            ),
            check_pir_onednn=self.check_pir_onednn,
        )

    def test_check_output(self):
        self.check_output(
            check_prim=(
                True
                if self.dtype not in [np.complex64, np.complex128]
                else False
            ),
            check_pir=True,
            check_prim_pir=(
                True
                if self.dtype not in [np.complex64, np.complex128]
                else False
            ),
            check_pir_onednn=self.check_pir_onednn,
        )


class TestHardSwish_ZeroDim(TestHardSwish):
    def init_shape(self):
        self.shape = []


class TestHardSwishComplex64(TestHardSwish):
    def init_dtype(self):
        self.dtype = np.complex64


class TestHardSwishComplex128(TestHardSwish):
    def init_dtype(self):
        self.dtype = np.complex128


class TestHardswishAPI(unittest.TestCase):
    # test paddle.nn.Hardswish, paddle.nn.functional.hardswish
    def setUp(self):
        self.x_np = np.random.uniform(-1, 1, [10, 12]).astype(np.float64)
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    def test_static_api(self):
        with static_guard():
            with paddle.static.program_guard(paddle.static.Program()):
                x = paddle.static.data('X', self.x_np.shape, self.x_np.dtype)
                out1 = F.hardswish(x)
                m = paddle.nn.Hardswish()
                out2 = m(x)
                exe = paddle.static.Executor(self.place)
                res = exe.run(feed={'X': self.x_np}, fetch_list=[out1, out2])
            out_ref = ref_hardswish(self.x_np)
            for r in res:
                np.testing.assert_allclose(out_ref, r, rtol=1e-05)

    def test_dygraph_api(self):
        with dynamic_guard():
            x = paddle.to_tensor([11648.0, 11448.0])
            out1 = F.hardswish(x)
            m = paddle.nn.Hardswish()
            out2 = m(x)
            out_ref = [11648.0, 11448.0]
            for r in [out1, out2]:
                np.testing.assert_allclose(out_ref, r.numpy(), rtol=1e-05)

    def test_base_api(self):
        with static_guard():
            with base.program_guard(base.Program()):
                x = paddle.static.data('X', self.x_np.shape, self.x_np.dtype)
                out = paddle.nn.functional.hardswish(x)
                exe = base.Executor(self.place)
                res = exe.run(feed={'X': self.x_np}, fetch_list=[out])
            out_ref = ref_hardswish(self.x_np)
            np.testing.assert_allclose(out_ref, res[0], rtol=1e-05)

        with dynamic_guard():
            x = paddle.to_tensor(self.x_np)
            out = paddle.nn.functional.hardswish(x)
            np.testing.assert_allclose(out_ref, out.numpy(), rtol=1e-05)

    def test_errors(self):
        with static_guard():
            with paddle.static.program_guard(paddle.static.Program()):
                # The input type must be Variable.
                self.assertRaises(TypeError, F.hardswish, 1)
                # The input dtype must be float16, float32, float64.
                x_int32 = paddle.static.data(
                    name='x_int32', shape=[12, 10], dtype='int32'
                )
                self.assertRaises(TypeError, F.hardswish, x_int32)
                # support the input dtype is float16
                x_fp16 = paddle.static.data(
                    name='x_fp16', shape=[12, 10], dtype='float16'
                )
                F.hardswish(x_fp16)


class TestSoftRelu(TestActivation):
    def setUp(self):
        self.op_type = "soft_relu"
        self.init_dtype()

        np.random.seed(4096)
        x = np.random.uniform(-3, 3, [4, 4]).astype(self.dtype)
        threshold = 2.0
        # The same reason with TestAbs
        x[np.abs(x - threshold) < 0.005] = threshold + 0.02
        x[np.abs(x + threshold) < 0.005] = -threshold - 0.02
        t = np.copy(x)
        t[t < -threshold] = -threshold
        t[t > threshold] = threshold
        out = np.log(np.exp(t) + 1)

        self.inputs = {'X': OpTest.np_dtype_to_base_dtype(x)}
        self.outputs = {'Out': out}
        self.convert_input_output()
        self.attrs = {'threshold': threshold}

    def test_check_output(self):
        self.check_output(
            check_dygraph=False, check_pir_onednn=self.check_pir_onednn
        )

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(
            ['X'],
            'Out',
            max_relative_error=0.02,
            check_dygraph=False,
            check_pir_onednn=self.check_pir_onednn,
        )


def elu(x, alpha):
    out_ref = np.where(x > 0, x, alpha * (np.exp(x) - 1))
    return out_ref.astype(x.dtype)


class TestELU(TestActivation):
    def setUp(self):
        self.op_type = "elu"
        self.init_dtype()
        self.init_shape()
        self.python_api = paddle.nn.functional.elu
        self.prim_op_type = "comp"
        self.public_python_api = paddle.nn.functional.elu

        np.random.seed(1024)
        x = np.random.uniform(-3, 3, self.shape).astype(self.dtype)
        alpha = self.get_alpha()
        out = elu(x, alpha)
        # Note: unlike other Relu extensions, point 0 on standard ELU function (i.e. alpha = 1)
        # is differentiable, so we can skip modifications like x[np.abs(x) < 0.005] = 0.02 here

        self.inputs = {'X': OpTest.np_dtype_to_base_dtype(x)}
        self.outputs = {'Out': out}
        self.convert_input_output()
        self.attrs = {'alpha': alpha}

    def init_shape(self):
        self.shape = [10, 12]

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(
            ['X'],
            'Out',
            check_pir=True,
            check_prim_pir=True,
            check_pir_onednn=self.check_pir_onednn,
        )

    def test_check_output(self):
        self.check_output(
            check_prim_pir=True,
            check_pir_onednn=self.check_pir_onednn,
            check_symbol_infer=False,
        )

    def get_alpha(self):
        return 1.0


class TestELUAlpha(TestELU):
    def get_alpha(self):
        return -0.2


class TestELU_ZeroDim(TestELU):
    def init_shape(self):
        self.shape = []


class TestELUAPI(unittest.TestCase):
    # test paddle.nn.ELU, paddle.nn.functional.elu
    def setUp(self):
        np.random.seed(1024)
        self.x_np = np.random.uniform(-3, 3, [10, 12]).astype('float32')
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )
        self.executed_api()

    def executed_api(self):
        self.elu = F.elu

    def test_static_api(self):
        with static_guard():
            with paddle.static.program_guard(paddle.static.Program()):
                x = paddle.static.data('X', [10, 12], dtype="float32")
                out1 = self.elu(x)
                m = paddle.nn.ELU()
                out2 = m(x)
                exe = paddle.static.Executor(self.place)
                res = exe.run(feed={'X': self.x_np}, fetch_list=[out1, out2])
            out_ref = elu(self.x_np, 1.0)
            for r in res:
                np.testing.assert_allclose(out_ref, r, rtol=1e-05)

    def test_dygraph_api(self):
        with dynamic_guard():
            x = paddle.to_tensor(self.x_np)
            out1 = self.elu(x)
            x = paddle.to_tensor(self.x_np)
            m = paddle.nn.ELU()
            out2 = m(x)
            out_ref = elu(self.x_np, 1.0)
            for r in [out1, out2]:
                np.testing.assert_allclose(out_ref, r.numpy(), rtol=1e-05)

            out1 = self.elu(x, 0.2)
            x = paddle.to_tensor(self.x_np)
            m = paddle.nn.ELU(0.2)
            out2 = m(x)
            out_ref = elu(self.x_np, 0.2)
            for r in [out1, out2]:
                np.testing.assert_allclose(out_ref, r.numpy(), rtol=1e-05)

    def test_errors(self):
        with static_guard():
            with paddle.static.program_guard(paddle.static.Program()):
                # The input type must be Variable.
                self.assertRaises(TypeError, self.elu, 1)
                # The input dtype must be float16, float32, float64.
                x_int32 = paddle.static.data(
                    name='x_int32', shape=[10, 12], dtype='int32'
                )
                self.assertRaises(TypeError, self.elu, x_int32)
                # support the input dtype is float16
                x_fp16 = paddle.static.data(
                    name='x_fp16', shape=[10, 12], dtype='float16'
                )
                self.elu(x_fp16)


class TestELUInplaceAPI(TestELUAPI):
    # test paddle.nn.functional.elu_
    def executed_api(self):
        self.elu = F.elu_

    def test_alpha_error(self):
        with dynamic_guard():
            x = paddle.to_tensor(self.x_np)
            self.assertRaises(Exception, F.elu_, x, -0.2)


def celu(x, alpha):
    out_ref = np.maximum(0, x) + np.minimum(0, alpha * (np.exp(x / alpha) - 1))
    return out_ref.astype(x.dtype)


class TestCELU(TestActivation):
    def setUp(self):
        self.op_type = "celu"
        self.init_dtype()
        self.init_shape()

        self.python_api = paddle.nn.functional.celu
        np.random.seed(1024)
        x = np.random.uniform(-3, 3, self.shape).astype(self.dtype)
        alpha = 1.5
        out = celu(x, alpha)

        self.inputs = {'X': OpTest.np_dtype_to_base_dtype(x)}
        self.outputs = {'Out': out}
        self.convert_input_output()
        self.attrs = {'alpha': alpha}

    def init_shape(self):
        self.shape = [10, 12]

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(
            ['X'], 'Out', check_pir=True, check_pir_onednn=self.check_pir_onednn
        )


class TestCELU_ZeroDim(TestCELU):
    def init_shape(self):
        self.shape = []


class TestCELUAPI(unittest.TestCase):
    # test paddle.nn.CELU, paddle.nn.functional.celu
    def setUp(self):
        np.random.seed(1024)
        self.x_np = np.random.uniform(-3, 3, [10, 12]).astype('float32')
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )
        self.executed_api()

    def executed_api(self):
        self.celu = F.celu

    def test_static_api(self):
        with static_guard():
            with paddle.static.program_guard(paddle.static.Program()):
                x = paddle.static.data('X', [10, 12], dtype="float32")
                out1 = self.celu(x, 1.5)
                m = paddle.nn.CELU(1.5)
                out2 = m(x)
                exe = paddle.static.Executor(self.place)
                res = exe.run(feed={'X': self.x_np}, fetch_list=[out1, out2])
            out_ref = celu(self.x_np, 1.5)
            for r in res:
                np.testing.assert_allclose(out_ref, r, rtol=1e-05)

    def test_dygraph_api(self):
        with dynamic_guard():
            x = paddle.to_tensor(self.x_np)
            out1 = self.celu(x, 1.5)
            x = paddle.to_tensor(self.x_np)
            m = paddle.nn.CELU(1.5)
            out2 = m(x)
            out_ref = celu(self.x_np, 1.5)
            for r in [out1, out2]:
                np.testing.assert_allclose(out_ref, r.numpy(), rtol=1e-05)

            out1 = self.celu(x, 0.2)
            x = paddle.to_tensor(self.x_np)
            m = paddle.nn.CELU(0.2)
            out2 = m(x)
            out_ref = celu(self.x_np, 0.2)
            for r in [out1, out2]:
                np.testing.assert_allclose(out_ref, r.numpy(), rtol=1e-05)

    def test_errors(self):
        with static_guard():
            with paddle.static.program_guard(paddle.static.Program()):
                # The input type must be Variable.
                self.assertRaises(TypeError, self.celu, 1)
                # The input dtype must be float16, float32, float64.
                x_int32 = paddle.static.data(
                    name='x_int32', shape=[10, 12], dtype='int32'
                )
                self.assertRaises(TypeError, self.celu, x_int32)
                # The alpha must be not equal 0
                x_fp32 = paddle.static.data(
                    name='x_fp32', shape=[10, 12], dtype='float32'
                )
                self.assertRaises(ZeroDivisionError, F.celu, x_fp32, 0)
                # support the input dtype is float16
                x_fp16 = paddle.static.data(
                    name='x_fp16', shape=[10, 12], dtype='float16'
                )
                self.celu(x_fp16)


class TestReciprocal(TestActivation):
    def setUp(self):
        self.op_type = "reciprocal"
        self.python_api = paddle.reciprocal
        self.public_python_api = paddle.reciprocal
        self.prim_op_type = "comp"
        self.init_dtype()
        self.init_shape()

        np.random.seed(1024)
        x = np.random.uniform(1, 2, self.shape).astype(self.dtype)
        if self.dtype == np.complex64 or self.dtype == np.complex128:
            x = (
                np.random.uniform(-1, 1, self.shape)
                + 1j * np.random.uniform(-1, 1, self.shape)
            ).astype(self.dtype)
        out = np.reciprocal(x)

        self.inputs = {'X': OpTest.np_dtype_to_base_dtype(x)}
        self.outputs = {'Out': out}
        self.convert_input_output()

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        if self.dtype == np.complex64 or self.dtype == np.complex128:
            self.check_grad(
                ['X'],
                'Out',
                max_relative_error=0.03,
                check_pir=True,
                check_pir_onednn=self.check_pir_onednn,
            )
        else:
            self.check_grad(
                ['X'],
                'Out',
                max_relative_error=0.01,
                check_pir=True,
                check_pir_onednn=self.check_pir_onednn,
            )

    def test_check_output(self):
        self.check_output(
            check_pir=True,
            check_prim_pir=True,
            check_pir_onednn=self.check_pir_onednn,
            check_symbol_infer=False,
        )


class TestReciprocal_Complex64(TestReciprocal):
    def init_dtype(self):
        self.dtype = np.complex64

    def test_check_output(self):
        self.check_output(
            check_pir=True,
            check_pir_onednn=self.check_pir_onednn,
        )


class TestReciprocal_Complex128(TestReciprocal):
    def init_dtype(self):
        self.dtype = np.complex128

    def test_check_output(self):
        self.check_output(
            check_pir=True,
            check_pir_onednn=self.check_pir_onednn,
        )


class TestReciprocal_ZeroDim(TestReciprocal):
    def init_shape(self):
        self.shape = []


class TestLog(TestActivation):
    def setUp(self):
        self.op_type = "log"
        self.prim_op_type = "prim"
        self.python_api = paddle.log
        self.public_python_api = paddle.log
        self.init_dtype()
        self.init_shape()
        self.if_enable_cinn()

        np.random.seed(1024)
        x = np.random.uniform(0.1, 1, self.shape).astype(self.dtype)
        if self.dtype == np.complex64 or self.dtype == np.complex128:
            x = (
                np.random.uniform(0.1, 1, self.shape)
                + 1j * np.random.uniform(0.1, 1, self.shape)
            ).astype(self.dtype)
        out = np.log(x)

        self.inputs = {'X': OpTest.np_dtype_to_base_dtype(x)}
        self.outputs = {'Out': out}
        self.convert_input_output()

    def if_enable_cinn(self):
        pass

    def test_check_output(self):
        self.check_output(
            check_pir=True,
            check_pir_onednn=self.check_pir_onednn,
            check_symbol_infer=False,
        )

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(
            ['X'],
            'Out',
            check_prim=True,
            check_pir=True,
            check_prim_pir=True,
            check_pir_onednn=self.check_pir_onednn,
        )


class TestLog_Complex64(TestLog):
    def init_dtype(self):
        self.dtype = np.complex64

    def test_check_grad(self):
        self.check_grad(
            ['X'], 'Out', check_pir=True, check_pir_onednn=self.check_pir_onednn
        )

    def test_check_output(self):
        self.check_output(
            check_pir=True, check_pir_onednn=self.check_pir_onednn
        )

    def test_api_complex(self):
        paddle.disable_static()
        for device in devices:
            if device == 'cpu' or (
                device == 'gpu' and paddle.is_compiled_with_cuda()
            ):
                np_x = np.array([[2, 3, 4], [7, 8, 9]], dtype=self.dtype)
                x = paddle.to_tensor(np_x, dtype=self.dtype, place=device)
                y = paddle.log(x)
                x_expect = np.log(np_x)
                np.testing.assert_allclose(y.numpy(), x_expect, rtol=1e-3)
        paddle.enable_static()

    def test_grad_grad(self):
        paddle.disable_static()
        x_numpy = (
            np.random.uniform(0.1, 1, self.shape)
            + 1j * np.random.uniform(0.1, 1, self.shape)
        ).astype(self.dtype)

        expected_ddx = np.conj(-1 / np.power(x_numpy, 2))

        x = paddle.to_tensor(x_numpy, stop_gradient=False)
        y = paddle.log(x)
        dx = paddle.grad(
            outputs=[y], inputs=[x], create_graph=True, retain_graph=True
        )[0]
        ddx = paddle.grad(outputs=[dx], inputs=[x], retain_graph=True)[0]
        np.testing.assert_allclose(ddx.numpy(), expected_ddx, rtol=1e-3)


class TestLog_Complex128(TestLog_Complex64):
    def init_dtype(self):
        self.dtype = np.complex128


class Test_Log_Op_Fp16(unittest.TestCase):
    def test_api_fp16(self):
        with static_guard():
            with static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                x = [[2, 3, 4], [7, 8, 9]]
                x = paddle.to_tensor(x, dtype='float16')
                out = paddle.log(x)
                if core.is_compiled_with_cuda():
                    place = paddle.CUDAPlace(0)
                    exe = paddle.static.Executor(place)
                    (res,) = exe.run(fetch_list=[out])

    def test_api_bf16(self):
        with static_guard():
            with static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                x = [[2, 3, 4], [7, 8, 9]]
                x = paddle.to_tensor(x, dtype='bfloat16')
                out = paddle.log(x)
                if core.is_compiled_with_cuda():
                    place = paddle.CUDAPlace(0)
                    exe = paddle.static.Executor(place)
                    (res,) = exe.run(fetch_list=[out])


class Test_Log_Op_Int(unittest.TestCase):
    def test_api_int(self):
        paddle.disable_static()
        for dtype in ('int32', 'int64', 'float16'):
            np_x = np.array([[2, 3, 4], [7, 8, 9]], dtype=dtype)
            x = paddle.to_tensor(np_x, dtype=dtype)
            y = paddle.log(x)
            x_expect = np.log(np_x)
            np.testing.assert_allclose(y.numpy(), x_expect, rtol=1e-3)
        paddle.enable_static()


class TestLog_ZeroDim(TestLog):
    def init_shape(self):
        self.shape = []


class TestLog2(TestActivation):
    def setUp(self):
        self.op_type = "log2"
        self.python_api = paddle.log2
        self.init_dtype()
        self.init_shape()

        x = np.random.uniform(0.1, 1, self.shape).astype(self.dtype)
        if self.dtype == np.complex64 or self.dtype == np.complex128:
            x = (
                np.random.uniform(0.1, 1, self.shape)
                + 1j * np.random.uniform(0.1, 1, self.shape)
            ).astype(self.dtype)
        out = np.log2(x)

        self.inputs = {'X': OpTest.np_dtype_to_base_dtype(x)}
        self.outputs = {'Out': out}
        self.convert_input_output()

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(
            ['X'], 'Out', check_pir=True, check_pir_onednn=self.check_pir_onednn
        )

    def test_api(self):
        with static_guard():
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                input_x = np.random.uniform(0.1, 1, [11, 17]).astype("float64")
                data_x = paddle.static.data(
                    name="data_x", shape=[11, 17], dtype="float64"
                )

                out1 = paddle.log2(data_x)
                exe = paddle.static.Executor(place=base.CPUPlace())
                exe.run(paddle.static.default_startup_program())
                (res1,) = exe.run(
                    paddle.static.default_main_program(),
                    feed={"data_x": input_x},
                    fetch_list=[out1],
                )
            expected_res = np.log2(input_x)
            np.testing.assert_allclose(res1, expected_res, rtol=1e-05)

        # dygraph
        with base.dygraph.guard():
            np_x = np.random.uniform(0.1, 1, [11, 17]).astype("float64")
            data_x = paddle.to_tensor(np_x)
            z = paddle.log2(data_x)
            np_z = z.numpy()
            z_expected = np.array(np.log2(np_x))
        np.testing.assert_allclose(np_z, z_expected, rtol=1e-05)


class TestLog2_Complex64(TestLog2):
    def init_dtype(self):
        self.dtype = np.complex64

    def test_check_output(self):
        self.check_output(
            check_pir=True, check_pir_onednn=self.check_pir_onednn
        )

    def test_api_complex(self):
        paddle.disable_static()
        for device in devices:
            if device == 'cpu' or (
                device == 'gpu' and paddle.is_compiled_with_cuda()
            ):
                np_x = np.array([[2, 3, 4], [7, 8, 9]], dtype=self.dtype)
                x = paddle.to_tensor(np_x, dtype=self.dtype, place=device)
                y = paddle.log2(x)
                x_expect = np.log2(np_x)
                np.testing.assert_allclose(y.numpy(), x_expect, rtol=1e-3)
        paddle.enable_static()


class TestLog2_Complex128(TestLog2_Complex64):
    def init_dtype(self):
        self.dtype = np.complex128


class TestLog2_ZeroDim(TestLog2):
    def init_shape(self):
        self.shape = []


class TestLog2_Op_Int(unittest.TestCase):
    def test_api_int(self):
        paddle.disable_static()
        for dtype in ['int32', 'int64', 'float16']:
            np_x = np.array([[2, 3, 4], [7, 8, 9]], dtype=dtype)
            x = paddle.to_tensor(np_x, dtype=dtype)
            y = paddle.log2(x)
            x_expect = np.log2(np_x)
            np.testing.assert_allclose(y.numpy(), x_expect, rtol=1e-3)
        paddle.enable_static()

    def test_api_bf16(self):
        with static_guard():
            with static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                x = [[2, 3, 4], [7, 8, 9]]
                x = paddle.to_tensor(x, dtype='bfloat16')
                out = paddle.log2(x)
                if core.is_compiled_with_cuda():
                    place = paddle.CUDAPlace(0)
                    exe = paddle.static.Executor(place)
                    (res,) = exe.run(fetch_list=[out])


class TestLog10(TestActivation):
    def setUp(self):
        self.op_type = "log10"
        self.python_api = paddle.log10
        self.init_dtype()
        self.init_shape()

        x = np.random.uniform(0.1, 1, self.shape).astype(self.dtype)
        if self.dtype == np.complex64 or self.dtype == np.complex128:
            x = (
                np.random.uniform(0.1, 1, self.shape)
                + 1j * np.random.uniform(0.1, 1, self.shape)
            ).astype(self.dtype)
        out = np.log10(x)

        self.inputs = {'X': OpTest.np_dtype_to_base_dtype(x)}
        self.outputs = {'Out': out}
        self.convert_input_output()

    def test_check_output(self):
        self.check_output(
            check_pir=True,
            check_pir_onednn=self.check_pir_onednn,
            check_symbol_infer=False,
        )

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(
            ['X'], 'Out', check_pir=True, check_pir_onednn=self.check_pir_onednn
        )


class TestLog10_Complex64(TestLog10):
    def init_dtype(self):
        self.dtype = np.complex64

    def test_api_complex(self):
        paddle.disable_static()
        for device in devices:
            if device == 'cpu' or (
                device == 'gpu' and paddle.is_compiled_with_cuda()
            ):
                np_x = np.array([[2, 3, 4], [7, 8, 9]], dtype=self.dtype)
                x = paddle.to_tensor(np_x, dtype=self.dtype, place=device)
                y = paddle.log10(x)
                x_expect = np.log10(np_x)
                np.testing.assert_allclose(y.numpy(), x_expect, rtol=1e-3)
        paddle.enable_static()


class TestLog10_Complex128(TestLog10_Complex64):
    def init_dtype(self):
        self.dtype = np.complex128


class TestLog10_ZeroDim(TestLog10):
    def init_shape(self):
        self.shape = []


class TestLog10_Op_Int(unittest.TestCase):
    def test_api_int(self):
        paddle.disable_static()
        for dtype in ['int32', 'int64', 'float16']:
            np_x = np.array([[2, 3, 4], [7, 8, 9]], dtype=dtype)
            x = paddle.to_tensor(np_x, dtype=dtype)
            y = paddle.log10(x)
            x_expect = np.log10(np_x)
            np.testing.assert_allclose(y.numpy(), x_expect, rtol=1e-3)
        paddle.enable_static()

    def test_api_bf16(self):
        paddle.enable_static()
        with static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            x = [[2, 3, 4], [7, 8, 9]]
            x = paddle.to_tensor(x, dtype='bfloat16')
            out = paddle.log10(x)
            if core.is_compiled_with_cuda():
                place = paddle.CUDAPlace(0)
                exe = paddle.static.Executor(place)
                (res,) = exe.run(fetch_list=[out])


class TestLog10API(unittest.TestCase):

    def test_api(self):
        with static_guard():
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                input_x = np.random.uniform(0.1, 1, [11, 17]).astype("float64")
                data_x = paddle.static.data(
                    name="data_x", shape=[11, 17], dtype="float64"
                )

                out1 = paddle.log10(data_x)
                exe = paddle.static.Executor(place=paddle.CPUPlace())
                exe.run(paddle.static.default_startup_program())
                (res1,) = exe.run(
                    paddle.static.default_main_program(),
                    feed={"data_x": input_x},
                    fetch_list=[out1],
                )
            expected_res = np.log10(input_x)
            np.testing.assert_allclose(res1, expected_res, rtol=1e-05)

        # dygraph
        with base.dygraph.guard():
            np_x = np.random.uniform(0.1, 1, [11, 17]).astype("float64")
            data_x = paddle.to_tensor(np_x)
            z = paddle.log10(data_x)
            np_z = z.numpy()
            z_expected = np.array(np.log10(np_x))
        np.testing.assert_allclose(np_z, z_expected, rtol=1e-05)


class TestLog1p(TestActivation):
    def setUp(self):
        self.op_type = "log1p"
        self.python_api = paddle.log1p
        self.init_dtype()
        self.init_shape()

        np.random.seed(1024)
        x = np.random.uniform(0.1, 1, self.shape).astype(self.dtype)
        if self.dtype == np.complex64 or self.dtype == np.complex128:
            x = (
                np.random.uniform(0.1, 1, self.shape)
                + 1j * np.random.uniform(0.1, 1, self.shape)
            ).astype(self.dtype)
        out = np.log1p(x)

        self.inputs = {'X': OpTest.np_dtype_to_base_dtype(x)}
        self.outputs = {'Out': out}
        self.convert_input_output()

    def test_check_output(self):
        self.check_output(
            check_pir=True,
            check_pir_onednn=self.check_pir_onednn,
            check_symbol_infer=False,
        )

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(
            ['X'], 'Out', check_pir=True, check_pir_onednn=self.check_pir_onednn
        )


class TestLog1p_Complex64(TestLog1p):
    def init_dtype(self):
        self.dtype = np.complex64

    def test_api_complex(self):
        paddle.disable_static()
        for device in devices:
            if device == 'cpu' or (
                device == 'gpu' and paddle.is_compiled_with_cuda()
            ):
                np_x = np.array([[2, 3, 4], [7, 8, 9]], dtype=self.dtype)
                x = paddle.to_tensor(np_x, dtype=self.dtype, place=device)
                y = paddle.log1p(x)
                x_expect = np.log1p(np_x)
                np.testing.assert_allclose(y.numpy(), x_expect, rtol=1e-3)
        paddle.enable_static()


class TestLog1p_Complex128(TestLog1p_Complex64):
    def init_dtype(self):
        self.dtype = np.complex128


class Test_Log1p_Op_Fp16(unittest.TestCase):

    def test_api_fp16(self):
        with static_guard():
            with static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                x = [[2, 3, 4], [7, 8, 9]]
                x = paddle.to_tensor(x, dtype='float16')
                out = paddle.log1p(x)
                if core.is_compiled_with_cuda():
                    place = paddle.CUDAPlace(0)
                    exe = paddle.static.Executor(place)
                    (res,) = exe.run(fetch_list=[out])


class TestLog1p_Op_Int(unittest.TestCase):
    def test_api_int(self):
        paddle.disable_static()
        for dtype in ['int32', 'int64', 'float16']:
            np_x = np.array([[2, 3, 4], [7, 8, 9]], dtype=dtype)
            x = paddle.to_tensor(np_x, dtype=dtype)
            y = paddle.log1p(x)
            x_expect = np.log1p(np_x)
            np.testing.assert_allclose(y.numpy(), x_expect, rtol=1e-3)
        paddle.enable_static()

    def test_api_bf16(self):
        with static_guard():
            with static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                x = [[2, 3, 4], [7, 8, 9]]
                x = paddle.to_tensor(x, dtype='bfloat16')
                out = paddle.log1p(x)
                if core.is_compiled_with_cuda():
                    place = paddle.CUDAPlace(0)
                    exe = paddle.static.Executor(place)
                    (res,) = exe.run(fetch_list=[out])


class TestLog1p_ZeroDim(TestLog1p):
    def init_shape(self):
        self.shape = []


class TestLog1pAPI(unittest.TestCase):

    def test_api(self):
        with static_guard():
            with base.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                input_x = np.random.uniform(0.1, 1, [11, 17]).astype("float64")
                data_x = paddle.static.data(
                    name="data_x",
                    shape=[11, 17],
                    dtype="float64",
                )

                out1 = paddle.log1p(data_x)
                exe = base.Executor(place=base.CPUPlace())
                exe.run(paddle.static.default_startup_program())
                (res1,) = exe.run(
                    paddle.static.default_main_program(),
                    feed={"data_x": input_x},
                    fetch_list=[out1],
                )
            expected_res = np.log1p(input_x)
            np.testing.assert_allclose(res1, expected_res, rtol=1e-05)

        # dygraph
        with base.dygraph.guard():
            np_x = np.random.uniform(0.1, 1, [11, 17]).astype("float64")
            data_x = paddle.to_tensor(np_x)
            z = paddle.log1p(data_x)
            np_z = z.numpy()
            z_expected = np.array(np.log1p(np_x))
        np.testing.assert_allclose(np_z, z_expected, rtol=1e-05)


class TestSquare(TestActivation):
    def setUp(self):
        self.op_type = "square"
        self.python_api = paddle.square
        self.prim_op_type = "comp"
        self.public_python_api = paddle.square
        self.init_dtype()
        self.init_shape()

        np.random.seed(1024)
        x = np.random.uniform(0.1, 1, self.shape).astype(self.dtype)
        if self.dtype == np.complex64 or self.dtype == np.complex128:
            x = (
                np.random.uniform(-1, 1, self.shape)
                + 1j * np.random.uniform(-1, 1, self.shape)
            ).astype(self.dtype)
        out = np.square(x)

        self.inputs = {'X': OpTest.np_dtype_to_base_dtype(x)}
        self.outputs = {'Out': out}
        self.convert_input_output()

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(
            ['X'],
            'Out',
            max_relative_error=0.007,
            check_pir=True,
            check_prim_pir=True,
            check_pir_onednn=self.check_pir_onednn,
        )

    def test_check_output(self):
        self.check_output(
            check_pir=True,
            check_prim_pir=True,
            check_pir_onednn=self.check_pir_onednn,
            check_symbol_infer=False,
        )


class TestSquare_Complex64(TestSquare):
    def init_dtype(self):
        self.dtype = np.complex64

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(
            ['X'],
            'Out',
            max_relative_error=0.007,
            check_pir=True,
            check_pir_onednn=self.check_pir_onednn,
        )


class TestSquare_Complex128(TestSquare):
    def init_dtype(self):
        self.dtype = np.complex128

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(
            ['X'],
            'Out',
            max_relative_error=0.007,
            check_pir=True,
            check_pir_onednn=self.check_pir_onednn,
        )


class TestSquare_ZeroDim(TestSquare):
    def init_shape(self):
        self.shape = []


@unittest.skipIf(
    not core.is_compiled_with_cuda() or core.is_compiled_with_rocm(),
    "core is not compiled with CUDA",
)
class TestSquareBF16(OpTest):
    def setUp(self):
        self.op_type = "square"
        self.python_api = paddle.square
        self.prim_op_type = "comp"
        self.public_python_api = paddle.square
        self.init_dtype()

        np.random.seed(1024)
        x = np.random.uniform(0.1, 1, [11, 17]).astype(np.float32)
        out = np.square(x)

        self.inputs = {
            'X': OpTest.np_dtype_to_base_dtype(convert_float_to_uint16(x))
        }
        self.outputs = {'Out': convert_float_to_uint16(out)}

    def init_dtype(self):
        self.dtype = np.uint16

    def test_check_output(self):
        place = core.CUDAPlace(0)
        self.check_output_with_place(
            place,
            check_pir=True,
            check_prim_pir=True,
            check_pir_onednn=self.check_pir_onednn,
            check_symbol_infer=False,
        )

    def test_check_grad(self):
        place = core.CUDAPlace(0)
        self.check_grad_with_place(
            place,
            ['X'],
            'Out',
            numeric_grad_delta=0.5,
            check_pir=True,
            check_prim_pir=True,
        )


class TestPow(TestActivation):
    def setUp(self):
        self.op_type = "pow"
        self.prim_op_type = "comp"
        self.python_api = paddle.pow
        self.public_python_api = paddle.pow
        self.init_dtype()
        self.init_shape()
        self.init_factor()
        self.if_enable_cinn()

        np.random.seed(1024)
        x = np.random.uniform(1, 2, self.shape).astype(self.dtype)
        if self.dtype == np.complex64 or self.dtype == np.complex128:
            x = (
                np.random.uniform(0.1, 1, self.shape)
                + 1j * np.random.uniform(0.1, 1, self.shape)
            ).astype(self.dtype)
            x = np.array([3 + 5j]).astype(np.complex64)

        out = np.power(x, self.factor)

        self.inputs = {'X': OpTest.np_dtype_to_base_dtype(x)}
        self.outputs = {'Out': out}
        self.attrs = {'factor': self.factor}
        self.convert_input_output()

    def if_enable_cinn(self):
        pass

    def test_check_output(self):
        self.check_output(
            check_prim=True,
            check_prim_pir=True,
            check_pir=True,
            check_pir_onednn=self.check_pir_onednn,
            check_symbol_infer=False,
        )

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(
            ['X'],
            'Out',
            check_prim=True,
            check_prim_pir=True,
            check_pir=True,
            check_pir_onednn=self.check_pir_onednn,
        )

    def init_factor(self):
        self.factor = 3.0


class TestPow_Complex64(TestPow):
    def init_dtype(self):
        self.dtype = np.complex64

    def test_check_grad(self):
        self.check_grad(
            ['X'], 'Out', check_pir=True, check_pir_onednn=self.check_pir_onednn
        )

    def test_check_output(self):
        self.check_output(
            check_pir=True, check_pir_onednn=self.check_pir_onednn
        )


class TestPow_ZeroDim(TestPow):
    def init_shape(self):
        self.shape = []


class TestPow_API(TestActivation):
    def test_api(self):
        with static_guard():
            input = np.random.uniform(1, 2, [11, 17]).astype("float32")
            x = paddle.static.data(name="x", shape=[11, 17], dtype="float32")
            res = paddle.static.data(
                name="res", shape=[11, 17], dtype="float32"
            )

            factor_1 = 2.0
            factor_2 = paddle.tensor.fill_constant([1], "float32", 3.0)
            out_1 = paddle.pow(x, factor_1)
            out_2 = paddle.pow(x, factor_2)
            out_4 = paddle.pow(x, factor_1, name='pow_res')
            out_6 = paddle.pow(x, factor_2)
            self.assertEqual(('pow_res' in out_4.name), True)

            exe = base.Executor(place=base.CPUPlace())
            res_1, res_2, res, res_6 = exe.run(
                base.default_main_program(),
                feed={"x": input},
                fetch_list=[out_1, out_2, res, out_6],
            )

            np.testing.assert_allclose(
                res_1, np.power(input, 2), rtol=1e-5, atol=1e-8
            )
            np.testing.assert_allclose(
                res_2, np.power(input, 3), rtol=1e-5, atol=1e-8
            )
            np.testing.assert_allclose(
                res_6, np.power(input, 3), rtol=1e-5, atol=1e-8
            )


def ref_stanh(x, scale_a=0.67, scale_b=1.7159):
    out = scale_b * np.tanh(x * scale_a)
    return out


class TestSTanh(TestActivation):
    def get_scale_a(self):
        return 0.67

    def get_scale_b(self):
        return 1.7159

    def setUp(self):
        self.op_type = "stanh"
        self.python_api = paddle.stanh
        self.init_dtype()
        self.init_shape()

        scale_a = self.get_scale_a()
        scale_b = self.get_scale_b()

        np.random.seed(1024)
        if self.dtype is np.complex64 or self.dtype is np.complex128:
            x = (
                np.random.uniform(0.1, 1, self.shape)
                + 1j * np.random.uniform(0.1, 1, self.shape)
            ).astype(self.dtype)
        else:
            x = np.random.uniform(0.1, 1, self.shape).astype(self.dtype)
        # The same reason with TestAbs
        out = ref_stanh(x, scale_a, scale_b)

        self.inputs = {'X': OpTest.np_dtype_to_base_dtype(x)}
        self.outputs = {'Out': out}
        self.attrs = {'scale_a': scale_a, 'scale_b': scale_b}
        self.convert_input_output()

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(
            ['X'], 'Out', check_pir=True, check_pir_onednn=self.check_pir_onednn
        )

    def test_check_output(self):
        self.check_output(
            check_pir=True,
            check_pir_onednn=self.check_pir_onednn,
            check_symbol_infer=False,
        )


class TestSTanhScaleA(TestSTanh):
    def get_scale_a(self):
        return 2.0


class TestSTanhScaleB(TestSTanh):
    def get_scale_b(self):
        return 0.5


class TestSTanh_ZeroDim(TestSTanh):
    def init_shape(self):
        self.shape = []


class TestSTanhComplex64(TestSTanh):
    def init_dtype(self):
        self.dtype = np.complex64


class TestSTanhComplex128(TestSTanh):
    def init_dtype(self):
        self.dtype = np.complex128


class TestSTanhAPI(unittest.TestCase):
    # test paddle.nn.stanh
    def get_scale_a(self):
        return 0.67

    def get_scale_b(self):
        return 1.7159

    def setUp(self):
        np.random.seed(1024)
        self.x_np = np.random.uniform(-1, 1, [10, 12]).astype('float32')
        self.scale_a = self.get_scale_a()
        self.scale_b = self.get_scale_b()
        self.place = (
            paddle.CUDAPlace(0)
            if core.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    def test_static_api(self):
        with static_guard():
            with paddle.static.program_guard(paddle.static.Program()):
                x = paddle.static.data('X', [10, 12])
                out = paddle.stanh(x, self.scale_a, self.scale_b)
                exe = paddle.static.Executor(self.place)
                res = exe.run(feed={'X': self.x_np}, fetch_list=[out])
            out_ref = ref_stanh(self.x_np, self.scale_a, self.scale_b)
            for r in res:
                np.testing.assert_allclose(out_ref, r, rtol=1e-05)

    def test_dygraph_api(self):
        with dynamic_guard():
            x = paddle.to_tensor(self.x_np)
            out = paddle.stanh(x, self.scale_a, self.scale_b)
            out_ref = ref_stanh(self.x_np, self.scale_a, self.scale_b)
            for r in [out]:
                np.testing.assert_allclose(out_ref, r.numpy(), rtol=1e-05)

    def test_base_api(self):
        with static_guard():
            with base.program_guard(base.Program()):
                x = paddle.static.data('X', [10, 12], dtype="float32")
                out = paddle.stanh(x, self.scale_a, self.scale_b)
                exe = base.Executor(self.place)
                res = exe.run(feed={'X': self.x_np}, fetch_list=[out])
            out_ref = ref_stanh(self.x_np, self.scale_a, self.scale_b)
            np.testing.assert_allclose(out_ref, res[0], rtol=1e-05)

    def test_errors(self):
        with static_guard():
            with paddle.static.program_guard(paddle.static.Program()):
                # The input type must be Variable.
                self.assertRaises(TypeError, paddle.stanh, 1)
                # The input dtype must be float16, float32, float64.
                x_int32 = paddle.static.data(
                    name='x_int32', shape=[12, 10], dtype='int32'
                )
                self.assertRaises(TypeError, paddle.stanh, x_int32)
                # support the input dtype is float16
                if core.is_compiled_with_cuda():
                    x_fp16 = paddle.static.data(
                        name='x_fp16', shape=[12, 10], dtype='float16'
                    )
                    paddle.stanh(x_fp16)


class TestSTanhAPIScaleA(TestSTanhAPI):
    def get_scale_a(self):
        return 2.0


class TestSTanhAPIScaleB(TestSTanhAPI):
    def get_scale_b(self):
        return 0.5


def ref_softplus(x, beta=1, threshold=20):
    x_beta = beta * x
    out = np.select(
        [x_beta <= threshold, x_beta > threshold],
        [np.log(1 + np.exp(x_beta)) / beta, x],
    )
    return out


class TestSoftplus(TestActivation):
    def setUp(self):
        self.op_type = "softplus"
        self.python_api = paddle.nn.functional.softplus
        self.init_dtype()
        self.init_shape()

        beta = 2
        threshold = 15

        np.random.seed(1024)
        x = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        if self.dtype == np.complex64 or self.dtype == np.complex128:
            x = (
                np.random.uniform(-1, 1, self.shape)
                + 1j * np.random.uniform(-1, 1, self.shape)
            ).astype(self.dtype)
        out = ref_softplus(x, beta, threshold)
        self.inputs = {'X': x}
        self.attrs = {'beta': beta, "threshold": threshold}
        self.outputs = {'Out': out}

    def init_shape(self):
        self.shape = [10, 12]

    def test_check_output(self):
        self.check_output(
            check_pir=True,
            check_pir_onednn=self.check_pir_onednn,
            check_symbol_infer=False,
        )

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(
            ['X'], 'Out', check_pir=True, check_pir_onednn=self.check_pir_onednn
        )


class TestSoftplus_Complex64(TestSoftplus):
    def init_dtype(self):
        self.dtype = np.complex64

    def test_check_grad(self):
        self.check_grad(
            ['X'],
            'Out',
            max_relative_error=0.06,
            check_pir=True,
            check_pir_onednn=self.check_pir_onednn,
        )


class TestSoftplus_Complex128(TestSoftplus):
    def init_dtype(self):
        self.dtype = np.complex128


class TestSoftplus_ZeroDim(TestSoftplus):
    def init_shape(self):
        self.shape = []


@unittest.skipIf(
    not core.is_compiled_with_cuda() or core.is_compiled_with_rocm(),
    "core is not compiled with CUDA",
)
class TestSoftplusBF16(OpTest):
    def setUp(self):
        self.op_type = "softplus"
        self.init_dtype()
        self.python_api = paddle.nn.functional.softplus

        beta = 2
        threshold = 15

        np.random.seed(1024)
        x = np.random.uniform(-1, 1, [10, 12]).astype(np.float32)
        out = ref_softplus(x, beta, threshold)
        self.inputs = {'X': convert_float_to_uint16(x)}
        self.attrs = {'beta': beta, "threshold": threshold}
        self.outputs = {'Out': convert_float_to_uint16(out)}

    def init_dtype(self):
        self.dtype = np.uint16

    def test_check_output(self):
        place = core.CUDAPlace(0)
        self.check_output_with_place(
            place,
            check_pir=True,
            check_pir_onednn=self.check_pir_onednn,
            check_symbol_infer=False,
        )

    def test_check_grad(self):
        place = core.CUDAPlace(0)
        self.check_grad_with_place(
            place, ['X'], 'Out', numeric_grad_delta=0.05, check_pir=True
        )


class TestSoftplusAPI(unittest.TestCase):
    # test paddle.nn.Softplus, paddle.nn.functional.softplus
    def setUp(self):
        self.beta = 2
        self.threshold = 15
        np.random.seed(1024)
        self.x_np = np.random.uniform(-1, 1, [10, 12]).astype(np.float64)
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    def test_static_api(self):
        with static_guard():
            with paddle.static.program_guard(paddle.static.Program()):
                x = paddle.static.data('X', self.x_np.shape, self.x_np.dtype)
                out1 = F.softplus(x, self.beta, self.threshold)
                softplus = paddle.nn.Softplus(self.beta, self.threshold)
                out2 = softplus(x)
                exe = paddle.static.Executor(self.place)
                res = exe.run(feed={'X': self.x_np}, fetch_list=[out1, out2])
            out_ref = ref_softplus(self.x_np, self.beta, self.threshold)
            for r in res:
                np.testing.assert_allclose(out_ref, r, rtol=1e-05)

    def test_dygraph_api(self):
        with dynamic_guard():
            x = paddle.to_tensor(self.x_np)
            out1 = F.softplus(x, self.beta, self.threshold)
            softplus = paddle.nn.Softplus(self.beta, self.threshold)
            out2 = softplus(x)
            out_ref = ref_softplus(self.x_np, self.beta, self.threshold)
            for r in [out1, out2]:
                np.testing.assert_allclose(out_ref, r.numpy(), rtol=1e-05)

    def test_errors(self):
        with static_guard():
            with paddle.static.program_guard(paddle.static.Program()):
                # The input type must be Variable.
                self.assertRaises(TypeError, F.softplus, 1)
                # The input dtype must be float16, float32, float64.
                x_int32 = paddle.static.data(
                    name='x_int32', shape=[12, 10], dtype='int32'
                )
                self.assertRaises(TypeError, F.softplus, x_int32)
                # support the input dtype is float16
                if paddle.is_compiled_with_cuda():
                    x_fp16 = paddle.static.data(
                        name='x_fp16', shape=[12, 10], dtype='float16'
                    )
                    F.softplus(x_fp16)


def ref_softsign(x):
    out = np.divide(x, 1 + np.abs(x))
    return out


class TestSoftsign(TestActivation):
    def setUp(self):
        self.op_type = "softsign"
        self.prim_op_type = "comp"
        self.init_dtype()
        self.init_shape()

        self.python_api = paddle.nn.functional.softsign
        self.public_python_api = paddle.nn.functional.softsign

        np.random.seed(1024)
        x = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        if self.dtype == np.complex64 or self.dtype == np.complex128:
            x = (
                np.random.uniform(-1, 1, self.shape)
                + 1j * np.random.uniform(-1, 1, self.shape)
            ).astype(self.dtype)
        out = ref_softsign(x)

        self.inputs = {'X': OpTest.np_dtype_to_base_dtype(x)}
        self.outputs = {'Out': out}
        self.convert_input_output()

    def init_shape(self):
        self.shape = [10, 12]

    def test_check_output(self):
        if self.dtype == np.complex64 or self.dtype == np.complex128:
            self.check_output(
                check_pir=True,
                check_pir_onednn=self.check_pir_onednn,
                check_symbol_infer=False,
            )
        else:
            self.check_output(
                check_pir=True,
                check_pir_onednn=self.check_pir_onednn,
                check_prim_pir=True,
                check_symbol_infer=False,
            )

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        if self.dtype == np.complex64 or self.dtype == np.complex128:
            self.check_grad(
                ['X'],
                'Out',
                check_pir=True,
                check_pir_onednn=self.check_pir_onednn,
            )
        else:
            self.check_grad(
                ['X'],
                'Out',
                check_pir=True,
                check_pir_onednn=self.check_pir_onednn,
                check_prim_pir=True,
            )


class TestSoftsign_Complex64(TestSoftsign):
    def init_dtype(self):
        self.dtype = np.complex64


class TestSoftsign_Complex128(TestSoftsign):
    def init_dtype(self):
        self.dtype = np.complex128


class TestSoftsign_ZeroDim(TestSoftsign):
    def init_shape(self):
        self.shape = []


class TestSoftsignAPI(unittest.TestCase):
    # test paddle.nn.Softsign, paddle.nn.functional.softsign
    def setUp(self):
        np.random.seed(1024)
        self.x_np = np.random.uniform(-1, 1, [10, 12]).astype(np.float64)
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    def test_static_api(self):
        with static_guard():
            with paddle.static.program_guard(paddle.static.Program()):
                x = paddle.static.data('X', self.x_np.shape, self.x_np.dtype)
                out1 = F.softsign(x)
                softsign = paddle.nn.Softsign()
                out2 = softsign(x)
                exe = paddle.static.Executor(self.place)
                res = exe.run(feed={'X': self.x_np}, fetch_list=[out1, out2])
            out_ref = ref_softsign(self.x_np)
            for r in res:
                np.testing.assert_allclose(out_ref, r, rtol=1e-05)

    def test_dygraph_api(self):
        with dynamic_guard():
            x = paddle.to_tensor(self.x_np)
            out1 = F.softsign(x)
            softsign = paddle.nn.Softsign()
            out2 = softsign(x)
            out_ref = ref_softsign(self.x_np)
            for r in [out1, out2]:
                np.testing.assert_allclose(out_ref, r.numpy(), rtol=1e-05)

    def test_errors(self):
        with static_guard():
            with paddle.static.program_guard(paddle.static.Program()):
                # The input type must be Variable.
                self.assertRaises(TypeError, F.softsign, 1)
                # The input dtype must be float16, float32, float64.
                x_int32 = paddle.static.data(
                    name='x_int32', shape=[12, 10], dtype='int32'
                )
                self.assertRaises(TypeError, F.softsign, x_int32)
                # support the input dtype is float16
                if core.is_compiled_with_cuda():
                    x_fp16 = paddle.static.data(
                        name='x_fp16', shape=[12, 10], dtype='float16'
                    )
                    F.softsign(x_fp16)


def ref_thresholded_relu(x, threshold=1.0, value=0.0):
    out = (x > threshold) * x + (x <= threshold) * value
    return out


class TestThresholdedRelu(TestActivation):
    def setUp(self):
        self.op_type = "thresholded_relu"
        self.init_dtype()
        self.init_shape()
        self.python_api = paddle.nn.functional.thresholded_relu

        threshold = 15
        value = 5

        np.random.seed(1024)
        x = np.random.uniform(-20, 20, self.shape).astype(self.dtype)
        x[np.abs(x) < 0.005] = 0.02
        out = ref_thresholded_relu(x, threshold, value)

        self.inputs = {'X': OpTest.np_dtype_to_base_dtype(x)}
        self.outputs = {'Out': out}
        self.attrs = {"threshold": threshold, "value": value}
        self.convert_input_output()

    def init_shape(self):
        self.shape = [10, 12]

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(
            ['X'], 'Out', check_pir=True, check_pir_onednn=self.check_pir_onednn
        )

    def test_check_output(self):
        self.check_output(
            check_pir=True,
            check_pir_onednn=self.check_pir_onednn,
            check_symbol_infer=False,
        )


class TestThresholdedRelu_ZeroDim(TestThresholdedRelu):
    def init_shape(self):
        self.shape = []


class TestThresholdedReluAPI(unittest.TestCase):
    # test paddle.nn.ThresholdedReLU, paddle.nn.functional.thresholded_relu
    def setUp(self):
        self.threshold = 15
        self.value = 5
        np.random.seed(1024)
        self.x_np = np.random.uniform(-20, 20, [10, 12]).astype(np.float64)
        self.x_np[np.abs(self.x_np) < 0.005] = 0.02
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    def test_static_api(self):
        with static_guard():
            with paddle.static.program_guard(paddle.static.Program()):
                x = paddle.static.data('X', self.x_np.shape, self.x_np.dtype)
                out1 = F.thresholded_relu(x, self.threshold, self.value)
                thresholded_relu = paddle.nn.ThresholdedReLU(
                    self.threshold, self.value
                )
                out2 = thresholded_relu(x)
                exe = paddle.static.Executor(self.place)
                res = exe.run(feed={'X': self.x_np}, fetch_list=[out1, out2])
            out_ref = ref_thresholded_relu(
                self.x_np, self.threshold, self.value
            )
            for r in res:
                np.testing.assert_allclose(out_ref, r, rtol=1e-05)

    def test_dygraph_api(self):
        with dynamic_guard():
            x = paddle.to_tensor(self.x_np)
            out1 = F.thresholded_relu(x, self.threshold, self.value)
            thresholded_relu = paddle.nn.ThresholdedReLU(
                self.threshold, self.value
            )
            out2 = thresholded_relu(x)
            out_ref = ref_thresholded_relu(
                self.x_np, self.threshold, self.value
            )
            for r in [out1, out2]:
                np.testing.assert_allclose(out_ref, r.numpy(), rtol=1e-05)

    def test_errors(self):
        with static_guard():
            with paddle.static.program_guard(paddle.static.Program()):
                # The input type must be Variable.
                self.assertRaises(TypeError, F.thresholded_relu, 1)
                # The input dtype must be float16, float32, float64.
                x_int32 = paddle.static.data(
                    name='x_int32', shape=[12, 10], dtype='int32'
                )
                self.assertRaises(TypeError, F.thresholded_relu, x_int32)
                # support the input dtype is float16
                if paddle.is_compiled_with_cuda():
                    x_fp16 = paddle.static.data(
                        name='x_fp16', shape=[12, 10], dtype='float16'
                    )
                    F.thresholded_relu(x_fp16)


def ref_hardsigmoid(x, slope=0.166666666666667, offset=0.5):
    return np.maximum(np.minimum(x * slope + offset, 1.0), 0.0).astype(x.dtype)


class TestHardSigmoid(TestActivation):
    def setUp(self):
        self.op_type = "hard_sigmoid"
        self.prim_op_type = "comp"
        self.dtype = 'float64'
        self.slope = 0.166666666666667
        self.offset = 0.5
        self.set_attrs()
        self.init_shape()
        self.python_api = paddle.nn.functional.hardsigmoid
        self.public_python_api = paddle.nn.functional.hardsigmoid

        x = np.random.uniform(-5, 5, self.shape).astype(self.dtype)
        lower_threshold = -self.offset / self.slope
        upper_threshold = (1.0 - self.offset) / self.slope

        # Same reason as TestAbs
        delta = 0.005
        x[np.abs(x - lower_threshold) < delta] = lower_threshold - 0.02
        x[np.abs(x - upper_threshold) < delta] = upper_threshold - 0.02

        out = ref_hardsigmoid(x, self.slope, self.offset)

        self.attrs = {'slope': self.slope, 'offset': self.offset}

        self.inputs = {'X': OpTest.np_dtype_to_base_dtype(x)}
        self.outputs = {'Out': out}
        self.convert_input_output()

    def init_shape(self):
        self.shape = [10, 12]

    def set_attrs(self):
        pass

    def test_check_output(self):
        self.check_output(
            check_pir=True,
            check_prim_pir=True,
            check_pir_onednn=self.check_pir_onednn,
            check_symbol_infer=False,
        )

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(
            ['X'], 'Out', check_pir=True, check_pir_onednn=self.check_pir_onednn
        )


class TestHardSigmoidFP32(TestHardSigmoid):
    def set_attrs(self):
        self.dtype = 'float32'


class TestHardSigmoidSlopeOffset(TestHardSigmoid):
    def set_attrs(self):
        self.slope = 0.2
        self.offset = 0.4


class TestHardSigmoid_ZeroDim(TestHardSigmoid):
    def init_shape(self):
        self.shape = []


class TestHardsigmoidAPI(unittest.TestCase):
    # test paddle.nn.Hardsigmoid, paddle.nn.functional.hardsigmoid
    def setUp(self):
        self.x_np = np.random.uniform(-1, 1, [10, 12]).astype(np.float64)
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    def test_static_api(self):
        with static_guard():
            with paddle.static.program_guard(paddle.static.Program()):
                x = paddle.static.data('X', self.x_np.shape, self.x_np.dtype)
                out1 = F.hardsigmoid(x)
                m = paddle.nn.Hardsigmoid()
                out2 = m(x)
                exe = paddle.static.Executor(self.place)
                res = exe.run(feed={'X': self.x_np}, fetch_list=[out1, out2])
            out_ref = ref_hardsigmoid(self.x_np)
            for r in res:
                np.testing.assert_allclose(out_ref, r, rtol=1e-05)

    def test_dygraph_api(self):
        with dynamic_guard():
            x = paddle.to_tensor(self.x_np)
            out1 = F.hardsigmoid(x)
            m = paddle.nn.Hardsigmoid()
            out2 = m(x)
            out_ref = ref_hardsigmoid(self.x_np)
            for r in [out1, out2]:
                np.testing.assert_allclose(out_ref, r.numpy(), rtol=1e-05)

    def test_base_api(self):
        with static_guard():
            with base.program_guard(base.Program()):
                x = paddle.static.data('X', self.x_np.shape, self.x_np.dtype)
                out = paddle.nn.functional.hardsigmoid(x, slope=0.2)
                exe = base.Executor(self.place)
                res = exe.run(feed={'X': self.x_np}, fetch_list=[out])
            out_ref = ref_hardsigmoid(self.x_np, 0.2, 0.5)
            np.testing.assert_allclose(out_ref, res[0], rtol=1e-05)

        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x_np)
        out = paddle.nn.functional.hardsigmoid(x, slope=0.2)
        np.testing.assert_allclose(out_ref, out.numpy(), rtol=1e-05)

    def test_errors(self):
        with static_guard():
            with paddle.static.program_guard(paddle.static.Program()):
                # The input type must be Variable.
                self.assertRaises(TypeError, F.hardsigmoid, 1)
                # The input dtype must be float16, float32, float64.
                x_int32 = paddle.static.data(
                    name='x_int32', shape=[12, 10], dtype='int32'
                )
                self.assertRaises(TypeError, F.hardsigmoid, x_int32)
                # support the input dtype is float16
                x_fp16 = paddle.static.data(
                    name='x_fp16', shape=[12, 10], dtype='float16'
                )
                F.hardsigmoid(x_fp16)


def ref_swish(x):
    out = x * expit(x)
    return out


class TestSwish(TestActivation):
    def setUp(self):
        self.op_type = "swish"
        self.prim_op_type = "comp"
        self.python_api = paddle.nn.functional.swish
        self.public_python_api = paddle.nn.functional.swish
        self.init_dtype()
        self.init_shape()

        np.random.seed(1024)
        x = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        out = ref_swish(x)

        self.inputs = {'X': OpTest.np_dtype_to_base_dtype(x)}
        self.outputs = {'Out': out}
        self.attrs = {'beta': 1.0}
        self.convert_input_output()

    def init_shape(self):
        self.shape = [10, 12]

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(
            ['X'],
            'Out',
            check_pir=True,
            check_pir_onednn=self.check_pir_onednn,
            check_prim_pir=True,
        )


class TestSwish_ZeroDim(TestSwish):
    def init_shape(self):
        self.shape = []


class TestSwishAPI(unittest.TestCase):
    # test paddle.nn.Swish, paddle.nn.functional.swish
    def setUp(self):
        np.random.seed(1024)
        self.x_np = np.random.uniform(-1, 1, [10, 12]).astype(np.float64)
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    def test_static_api(self):
        with static_guard():
            with paddle.static.program_guard(paddle.static.Program()):
                x = paddle.static.data('X', self.x_np.shape, self.x_np.dtype)
                out1 = F.swish(x)
                swish = paddle.nn.Swish()
                out2 = swish(x)
                exe = paddle.static.Executor(self.place)
                res = exe.run(feed={'X': self.x_np}, fetch_list=[out1, out2])
            out_ref = ref_swish(self.x_np)
            for r in res:
                np.testing.assert_allclose(out_ref, r, rtol=1e-05)

    def test_dygraph_api(self):
        with dynamic_guard():
            x = paddle.to_tensor(self.x_np)
            out1 = F.swish(x)
            swish = paddle.nn.Swish()
            out2 = swish(x)
            out_ref = ref_swish(self.x_np)
            for r in [out1, out2]:
                np.testing.assert_allclose(out_ref, r.numpy(), rtol=1e-05)

    def test_base_api(self):
        with static_guard():
            with base.program_guard(base.Program()):
                x = paddle.static.data('X', self.x_np.shape, self.x_np.dtype)
                out = paddle.nn.functional.swish(x)
                exe = base.Executor(self.place)
                res = exe.run(feed={'X': self.x_np}, fetch_list=[out])
            out_ref = ref_swish(self.x_np)
            np.testing.assert_allclose(out_ref, res[0], rtol=1e-05)

    def test_errors(self):
        with static_guard():
            with paddle.static.program_guard(paddle.static.Program()):
                # The input type must be Variable.
                self.assertRaises(TypeError, F.swish, 1)
                # The input dtype must be float16, float32, float64.
                x_int32 = paddle.static.data(
                    name='x_int32', shape=[12, 10], dtype='int32'
                )
                self.assertRaises(TypeError, F.swish, x_int32)
                # support the input dtype is float16
                if paddle.is_compiled_with_cuda():
                    x_fp16 = paddle.static.data(
                        name='x_fp16', shape=[12, 10], dtype='float16'
                    )
                    F.swish(x_fp16)


def ref_mish(x, threshold=20.0):
    softplus = np.select(
        [x <= threshold, x > threshold], [np.log(1 + np.exp(x)), x]
    )
    return x * np.tanh(softplus)


class TestMish(TestActivation):
    def setUp(self):
        self.op_type = "mish"
        self.python_api = paddle.nn.functional.mish
        self.init_dtype()
        self.init_shape()

        np.random.seed(1024)
        x = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        out = ref_mish(x)

        self.inputs = {'X': OpTest.np_dtype_to_base_dtype(x)}
        self.outputs = {'Out': out}
        self.convert_input_output()

    def init_shape(self):
        self.shape = [10, 12]

    def test_check_output(self):
        self.check_output(
            check_pir=True,
            check_pir_onednn=self.check_pir_onednn,
            check_symbol_infer=False,
        )

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(
            ['X'], 'Out', check_pir=True, check_pir_onednn=self.check_pir_onednn
        )


class TestMish_ZeroDim(TestMish):
    def init_shape(self):
        self.shape = []


class TestMishAPI(unittest.TestCase):
    # test paddle.nn.Mish, paddle.nn.functional.mish
    def setUp(self):
        np.random.seed(1024)
        self.x_np = np.random.uniform(-1, 1, [10, 12]).astype(np.float64)
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    def test_static_api(self):
        with static_guard():
            with paddle.static.program_guard(paddle.static.Program()):
                x = paddle.static.data('X', self.x_np.shape, self.x_np.dtype)
                out1 = F.mish(x)
                mish = paddle.nn.Mish()
                out2 = mish(x)
                exe = paddle.static.Executor(self.place)
                res = exe.run(feed={'X': self.x_np}, fetch_list=[out1, out2])
            out_ref = ref_mish(self.x_np)
            for r in res:
                np.testing.assert_allclose(out_ref, r, rtol=1e-05)

    def test_dygraph_api(self):
        with dynamic_guard():
            x = paddle.to_tensor(self.x_np)
            out1 = F.mish(x)
            mish = paddle.nn.Mish()
            out2 = mish(x)
            out_ref = ref_mish(self.x_np)
            for r in [out1, out2]:
                np.testing.assert_allclose(out_ref, r.numpy(), rtol=1e-05)

    def test_base_api(self):
        with static_guard():
            with paddle.static.program_guard(paddle.static.Program()):
                x = paddle.static.data('X', self.x_np.shape, self.x_np.dtype)
                out = paddle.nn.functional.mish(x)
                exe = base.Executor(self.place)
                res = exe.run(feed={'X': self.x_np}, fetch_list=[out])
            out_ref = ref_mish(self.x_np)
            np.testing.assert_allclose(out_ref, res[0], rtol=1e-05)

    def test_errors(self):
        with static_guard():
            with paddle.static.program_guard(paddle.static.Program()):
                # The input type must be Variable.
                self.assertRaises(TypeError, F.mish, 1)
                # The input dtype must be float16, float32, float64.
                x_int32 = paddle.static.data(
                    name='x_int32', shape=[12, 10], dtype='int32'
                )
                self.assertRaises(TypeError, F.mish, x_int32)
                # support the input dtype is float16
                x_fp16 = paddle.static.data(
                    name='x_fp16', shape=[12, 10], dtype='float16'
                )
                F.mish(x_fp16)


# ------------------ Test Cudnn Activation----------------------
def create_test_act_cudnn_class(parent, atol=1e-3, grad_atol=1e-3):
    @unittest.skipIf(
        not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
    )
    class TestActCudnn(parent):
        def init_kernel_type(self):
            self.attrs = {"use_cudnn": True}

    cls_name = "{}_{}".format(parent.__name__, "cudnn")
    TestActCudnn.__name__ = cls_name
    globals()[cls_name] = TestActCudnn


create_test_act_cudnn_class(TestRelu)
create_test_act_cudnn_class(TestRelu6)
create_test_act_cudnn_class(TestSigmoid)
create_test_act_cudnn_class(TestTanh)


# ------------------ Test Fp16 ----------------------
def create_test_act_fp16_class(
    parent,
    atol=1e-3,
    grad_check=True,
    check_dygraph=True,
    check_prim=False,
    check_prim_pir=False,
    enable_cinn=False,
    check_pir=False,
    grad_atol=1e-2,
    **kwargs,
):
    @unittest.skipIf(
        not paddle.is_compiled_with_cuda(), "core is not compiled with CUDA"
    )
    class TestActFp16(parent):
        def setUp(self):
            super().setUp()
            for k, v in kwargs.items():
                setattr(self, k, v)

        def init_dtype(self):
            self.dtype = np.float16

        def if_enable_cinn(self):
            self.enable_cinn = enable_cinn

        def test_check_output(self):
            place = core.CUDAPlace(0)
            support_fp16 = core.is_float16_supported(place)
            if support_fp16:
                self.check_output_with_place(
                    place,
                    atol=atol,
                    check_dygraph=check_dygraph,
                    check_prim=check_prim,
                    check_prim_pir=check_prim_pir,
                    check_pir=check_pir,
                    check_pir_onednn=self.check_pir_onednn,
                )

        def test_check_grad(self):
            place = core.CUDAPlace(0)
            support_fp16 = core.is_float16_supported(place)
            if support_fp16 and grad_check:
                self.check_grad_with_place(
                    place,
                    ['X'],
                    'Out',
                    check_dygraph=check_dygraph,
                    check_prim=check_prim,
                    check_prim_pir=check_prim_pir,
                    max_relative_error=grad_atol,
                    check_pir=check_pir,
                )

    cls_name = "{}_{}".format(parent.__name__, "FP16OP")
    TestActFp16.__name__ = cls_name
    globals()[cls_name] = TestActFp16


create_test_act_fp16_class(TestActivation)
create_test_act_fp16_class(
    TestExpFp32_Prim, check_prim=True, enable_cinn=True, check_prim_pir=True
)
create_test_act_fp16_class(TestExpm1, check_prim_pir=True)
create_test_act_fp16_class(
    TestSigmoid,
    check_prim=True,
    enable_cinn=True,
    check_pir=True,
    check_prim_pir=True,
)
create_test_act_fp16_class(
    TestSilu, check_prim=True, enable_cinn=True, check_prim_pir=True
)
create_test_act_fp16_class(TestLogSigmoid, check_pir=True)
create_test_act_fp16_class(
    TestTanh, check_prim=True, check_prim_pir=True, enable_cinn=True
)
create_test_act_fp16_class(TestTanhshrink, check_pir=True)
create_test_act_fp16_class(TestHardShrink, check_pir=True)
create_test_act_fp16_class(TestSoftshrink, check_pir=True)
create_test_act_fp16_class(
    TestSqrt,
    check_prim=True,
    enable_cinn=True,
    check_pir=True,
    check_prim_pir=True,
)
create_test_act_fp16_class(
    TestSqrtComp,
    check_prim=True,
    enable_cinn=True,
    check_pir=True,
    check_prim_pir=True,
)
create_test_act_fp16_class(
    TestAbs,
    check_prim=True,
    enable_cinn=True,
    check_pir=True,
    check_prim_pir=True,
)
create_test_act_fp16_class(
    TestCeil,
    grad_check=False,
    check_pir=True,
    check_prim_pir=True,
)
create_test_act_fp16_class(
    TestFloor,
    check_prim=True,
    grad_check=False,
    enable_cinn=True,
    check_pir=True,
    check_prim_pir=True,
)
create_test_act_fp16_class(TestCos, check_pir=True, check_prim_pir=True)
create_test_act_fp16_class(TestTan, check_pir=True)
create_test_act_fp16_class(TestCosh, check_pir=True)
create_test_act_fp16_class(TestAcos, check_pir=True)
create_test_act_fp16_class(TestSin, check_pir=True, check_prim_pir=True)
create_test_act_fp16_class(TestSinh, check_pir=True)
create_test_act_fp16_class(TestAsin, check_pir=True)
create_test_act_fp16_class(TestAtan, check_pir=True, check_prim_pir=True)
create_test_act_fp16_class(TestAcosh, check_pir=True)
create_test_act_fp16_class(TestAsinh, check_pir=True)
create_test_act_fp16_class(TestAtanh, check_pir=True)
create_test_act_fp16_class(TestRound, grad_check=False, check_pir=True)
create_test_act_fp16_class(
    TestRelu,
    check_prim=True,
    enable_cinn=True,
    check_pir=True,
    check_prim_pir=True,
)
create_test_act_fp16_class(
    TestGelu,
    check_prim=True,
    check_prim_pir=True,
    check_pir=True,
    enable_cinn=True,
    rev_comp_rtol=1e-3,
    rev_comp_atol=1e-3,
    cinn_rtol=1e-3,
    cinn_atol=1e-3,
)
create_test_act_fp16_class(TestBRelu, check_pir=True)
create_test_act_fp16_class(TestRelu6)
create_test_act_fp16_class(TestSoftRelu, check_dygraph=False)
create_test_act_fp16_class(TestELU, check_pir=True, check_prim_pir=True)
create_test_act_fp16_class(TestCELU, check_pir=True)
create_test_act_fp16_class(TestReciprocal, check_pir=True)
create_test_act_fp16_class(TestLog, check_prim=True, check_pir=True)
create_test_act_fp16_class(TestLog2, check_pir=True)
create_test_act_fp16_class(TestLog10, check_pir=True)
create_test_act_fp16_class(TestLog1p, check_pir=True)
create_test_act_fp16_class(TestSquare, check_pir=True, check_prim_pir=True)
create_test_act_fp16_class(TestPow, check_prim=True, check_prim_pir=True)
create_test_act_fp16_class(TestPow_API)
create_test_act_fp16_class(TestSTanh)
create_test_act_fp16_class(TestSoftplus, check_pir=True)
create_test_act_fp16_class(TestSoftsign, check_pir=True)
create_test_act_fp16_class(TestThresholdedRelu, check_pir=True)
create_test_act_fp16_class(TestHardSigmoid, check_pir=True)
create_test_act_fp16_class(TestSwish)
create_test_act_fp16_class(
    TestHardSwish, check_prim=True, check_pir=True, check_prim_pir=True
)
create_test_act_fp16_class(TestMish, check_pir=True)
create_test_act_fp16_class(
    TestLeakyRelu,
    check_prim=True,
    enable_cinn=True,
    check_pir=True,
    check_prim_pir=True,
)
create_test_act_fp16_class(
    TestLeakyReluAlpha1, check_prim=True, enable_cinn=True, check_prim_pir=True
)
create_test_act_fp16_class(
    TestLeakyReluAlpha2, check_prim=True, enable_cinn=True, check_prim_pir=True
)
create_test_act_fp16_class(
    TestLeakyReluAlpha3, check_prim=True, enable_cinn=True, check_prim_pir=True
)
create_test_act_fp16_class(
    TestLeakyRelu_ZeroDim, check_prim=True, check_prim_pir=True
)
create_test_act_fp16_class(
    TestRsqrt,
    check_prim=True,
    enable_cinn=True,
    check_pir=True,
    check_prim_pir=True,
)


def create_test_act_bf16_class(
    parent,
    atol=1e-2,
    grad_check=True,
    check_dygraph=True,
    check_prim=False,
    enable_cinn=False,
    check_pir=False,
    check_prim_pir=False,
    grad_atol=1e-2,
    **kwargs,
):
    @unittest.skipIf(
        not core.is_compiled_with_cuda()
        or not core.is_bfloat16_supported(core.CUDAPlace(0)),
        "core is not compiled with CUDA and do not support bfloat16",
    )
    class TestActBF16(parent):
        def setUp(self):
            super().setUp()
            for k, v in kwargs.items():
                setattr(self, k, v)

        def init_dtype(self):
            self.dtype = np.float32

        def if_enable_cinn(self):
            self.enable_cinn = enable_cinn

        def convert_input_output(self):
            self.inputs = {'X': convert_float_to_uint16(self.inputs['X'])}
            self.outputs = {'Out': convert_float_to_uint16(self.outputs['Out'])}
            self.dtype = np.uint16

        def test_check_output(self):
            place = core.CUDAPlace(0)
            self.check_output_with_place(
                place,
                atol=atol,
                check_prim=check_prim,
                check_pir=check_pir,
                check_prim_pir=check_prim_pir,
                check_pir_onednn=self.check_pir_onednn,
            )

        def test_check_grad(self):
            place = core.CUDAPlace(0)
            if grad_check:
                self.check_grad_with_place(
                    place,
                    ['X'],
                    'Out',
                    max_relative_error=grad_atol,
                    check_prim=check_prim,
                    check_pir=check_pir,
                    check_prim_pir=check_prim_pir,
                )

    cls_name = "{}_{}".format(parent.__name__, "BF16OP")
    TestActBF16.__name__ = cls_name
    globals()[cls_name] = TestActBF16


create_test_act_bf16_class(TestActivation)
create_test_act_bf16_class(
    TestExpFp32_Prim, check_prim=True, check_prim_pir=True
)
create_test_act_bf16_class(TestExpm1, check_prim_pir=True)
create_test_act_bf16_class(
    TestSigmoid, check_prim=True, check_pir=True, check_prim_pir=True
)
create_test_act_bf16_class(TestSilu, check_prim=True, check_prim_pir=True)
create_test_act_bf16_class(TestLogSigmoid, check_pir=True)
create_test_act_bf16_class(TestTanh, check_prim=True, check_prim_pir=True)
create_test_act_bf16_class(TestTanhshrink, check_pir=True)
create_test_act_bf16_class(TestHardShrink, check_pir=True)
create_test_act_bf16_class(TestSoftshrink, check_pir=True)
create_test_act_bf16_class(
    TestSqrt, check_prim=True, check_pir=True, check_prim_pir=True
)
create_test_act_bf16_class(
    TestSqrtComp, check_prim=True, check_pir=True, check_prim_pir=True
)
create_test_act_bf16_class(
    TestAbs, check_prim=True, check_pir=True, check_prim_pir=True
)
create_test_act_bf16_class(
    TestCeil,
    grad_check=False,
    check_pir=True,
    check_prim_pir=True,
)
create_test_act_bf16_class(
    TestFloor,
    grad_check=False,
    check_prim=True,
    check_pir=True,
    check_prim_pir=True,
)
create_test_act_bf16_class(TestCos, check_pir=True, check_prim_pir=True)
create_test_act_bf16_class(TestTan, check_pir=True)
create_test_act_bf16_class(TestCosh, check_pir=True)
create_test_act_bf16_class(TestAcos, check_pir=True)
create_test_act_bf16_class(TestSin, check_pir=True, check_prim_pir=True)
create_test_act_bf16_class(TestSinh, check_pir=True)
create_test_act_bf16_class(TestAsin, check_pir=True)
create_test_act_bf16_class(TestAtan, check_pir=True, check_prim_pir=True)
create_test_act_bf16_class(TestAcosh, check_pir=True)
create_test_act_bf16_class(TestAsinh, check_pir=True)
create_test_act_bf16_class(TestAtanh, check_pir=True)
create_test_act_bf16_class(TestRound, grad_check=False, check_pir=True)
create_test_act_bf16_class(
    TestRelu, check_prim=True, check_pir=True, check_prim_pir=True
)
create_test_act_bf16_class(
    TestGelu,
    check_prim=True,
    check_pir=True,
    rev_comp_rtol=1e-2,
    rev_comp_atol=1e-2,
    cinn_rtol=1e-2,
    cinn_atol=1e-2,
)
create_test_act_bf16_class(TestBRelu, check_pir=True)
create_test_act_bf16_class(TestRelu6)
create_test_act_bf16_class(TestSoftRelu, check_dygraph=False)
create_test_act_bf16_class(TestELU, check_pir=True, check_prim_pir=True)
create_test_act_bf16_class(TestCELU, check_pir=True)
create_test_act_bf16_class(TestReciprocal, check_pir=True)
create_test_act_bf16_class(TestLog, check_prim=True, check_pir=True)
create_test_act_bf16_class(TestLog2, check_pir=True)
create_test_act_bf16_class(TestLog10, check_pir=True)
create_test_act_bf16_class(TestLog1p, check_pir=True)
create_test_act_bf16_class(TestSquare, check_pir=True, check_prim_pir=True)
create_test_act_bf16_class(TestPow, check_prim=True)
create_test_act_bf16_class(TestPow_API)
create_test_act_bf16_class(TestSTanh)
create_test_act_bf16_class(TestSoftplus, check_pir=True)
create_test_act_bf16_class(TestSoftsign, check_pir=True)
create_test_act_bf16_class(TestThresholdedRelu, check_pir=True)
create_test_act_bf16_class(TestHardSigmoid, check_pir=True)
create_test_act_bf16_class(TestSwish)
create_test_act_bf16_class(
    TestHardSwish, check_prim=True, check_pir=True, check_prim_pir=True
)
create_test_act_bf16_class(TestMish, check_pir=True)
create_test_act_bf16_class(
    TestLeakyRelu, check_prim=True, check_pir=True, check_prim_pir=True
)
create_test_act_bf16_class(
    TestLeakyReluAlpha1, check_prim=True, check_prim_pir=True
)
create_test_act_bf16_class(
    TestLeakyReluAlpha2, check_prim=True, check_prim_pir=True
)
create_test_act_bf16_class(
    TestLeakyReluAlpha3, check_prim=True, check_prim_pir=True
)
create_test_act_bf16_class(
    TestLeakyRelu_ZeroDim, check_prim=True, check_prim_pir=True
)
create_test_act_bf16_class(
    TestRsqrt, check_prim=True, check_pir=True, check_prim_pir=True
)

if __name__ == "__main__":
    unittest.main()
