#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import itertools
import unittest

import numpy as np
from op_test import OpTest
from utils import dygraph_guard, static_guard

import paddle
from paddle import base, static
from paddle.base import core


class TestQrOp(OpTest):
    def setUp(self):
        with static_guard():
            self.python_api = paddle.linalg.qr
            np.random.seed(7)
            self.op_type = "qr"
            a, q, r = self.get_input_and_output()
            self.inputs = {"X": a}
            self.attrs = {"mode": self.get_mode()}
            self.outputs = {"Q": q, "R": r}

    def get_dtype(self):
        return "float64"

    def get_mode(self):
        return "reduced"

    def get_shape(self):
        return (11, 11)

    def _get_places(self):
        places = []
        places.append(base.CPUPlace())
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        return places

    def get_input_and_output(self):
        dtype = self.get_dtype()
        shape = self.get_shape()
        mode = self.get_mode()
        assert mode != "r", "Cannot be backward in r mode."
        a = np.random.rand(*shape).astype(dtype)
        q, r = np.linalg.qr(a, mode=mode)
        return a, q, r

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad_normal(self):
        self.check_grad(
            ['X'],
            ['Q', 'R'],
            numeric_grad_delta=1e-5,
            max_relative_error=1e-6,
            check_pir=True,
        )


class TestQrOpCase1(TestQrOp):
    def get_shape(self):
        return (10, 12)


class TestQrOpCase2(TestQrOp):
    def get_shape(self):
        return (16, 15)


class TestQrOpCase3(TestQrOp):
    def get_shape(self):
        return (2, 12, 16)


class TestQrOpCase4(TestQrOp):
    def get_shape(self):
        return (3, 16, 15)


class TestQrOpCase5(TestQrOp):
    def get_mode(self):
        return "complete"

    def get_shape(self):
        return (10, 12)


class TestQrOpCase6(TestQrOp):
    def get_mode(self):
        return "complete"

    def get_shape(self):
        return (2, 10, 12)


@unittest.skipIf(
    core.is_compiled_with_xpu(),
    "Skip XPU for complex dtype is not fully supported",
)
class TestQrOpcomplex(TestQrOp):
    def get_input_and_output(self):
        dtype = self.get_dtype()
        shape = self.get_shape()
        mode = self.get_mode()
        assert mode != "r", "Cannot be backward in r mode."
        a_real = np.random.rand(*shape).astype(dtype)
        a_imag = np.random.rand(*shape).astype(dtype)
        a = a_real + 1j * a_imag
        q, r = np.linalg.qr(a, mode=mode)
        return a, q, r


@unittest.skipIf(
    core.is_compiled_with_xpu(),
    "Skip XPU for complex dtype is not fully supported",
)
class TestQrOpcomplexCase1(TestQrOpcomplex):
    def get_shape(self):
        return (16, 15)


@unittest.skipIf(
    core.is_compiled_with_xpu(),
    "Skip XPU for complex dtype is not fully supported",
)
class TestQrOpcomplexCase2(TestQrOpcomplex):
    def get_shape(self):
        return (3, 16, 15)


@unittest.skipIf(
    core.is_compiled_with_xpu(),
    "Skip XPU for complex dtype is not fully supported",
)
class TestQrOpcomplexCase3(TestQrOpcomplex):
    def get_shape(self):
        return (12, 15)


@unittest.skipIf(
    core.is_compiled_with_xpu(),
    "Skip XPU for complex dtype is not fully supported",
)
class TestQrOpcomplexCase4(TestQrOpcomplex):
    def get_shape(self):
        return (3, 12, 15)


class TestQrAPI(unittest.TestCase):
    def test_dygraph(self):
        def run_qr_dygraph(shape, mode, dtype):
            if dtype == "float32":
                np_dtype = np.float32
            elif dtype == "float64":
                np_dtype = np.float64
            elif dtype == "complex64":
                np_dtype = np.complex64
            elif dtype == "complex128":
                np_dtype = np.complex128
            if np.issubdtype(np_dtype, np.complexfloating):
                a_dtype = np.float32 if np_dtype == np.complex64 else np.float64
                a_real = np.random.rand(*shape).astype(a_dtype)
                a_imag = np.random.rand(*shape).astype(a_dtype)
                a = a_real + 1j * a_imag
            else:
                a = np.random.rand(*shape).astype(np_dtype)
            places = []
            places.append('cpu')
            if core.is_compiled_with_cuda():
                places.append('gpu')
            for place in places:
                if mode == "r":
                    np_r = np.linalg.qr(a, mode=mode)
                else:
                    np_q, np_r = np.linalg.qr(a, mode=mode)

                x = paddle.to_tensor(a, dtype=dtype, place=place)
                if mode == "r":
                    r = paddle.linalg.qr(x, mode=mode)
                    np.testing.assert_allclose(r, np_r, rtol=1e-05, atol=1e-05)
                else:
                    q, r = paddle.linalg.qr(x, mode=mode)
                    np.testing.assert_allclose(q, np_q, rtol=1e-05, atol=1e-05)
                    np.testing.assert_allclose(r, np_r, rtol=1e-05, atol=1e-05)

        with dygraph_guard():
            np.random.seed(7)
            tensor_shapes = [
                (0, 3),
                (3, 5),
                (5, 5),
                (5, 3),  # 2-dim Tensors
                (0, 3, 5),
                (4, 0, 5),
                (5, 4, 0),
                (2, 3, 5),
                (3, 5, 5),
                (4, 5, 3),  # 3-dim Tensors
                (0, 5, 3, 5),
                (2, 5, 3, 5),
                (3, 5, 5, 5),
                (4, 5, 5, 3),  # 4-dim Tensors
            ]
            modes = ["reduced", "complete", "r"]
            dtypes = ["float32", "float64", 'complex64', 'complex128']
            for tensor_shape, mode, dtype in itertools.product(
                tensor_shapes, modes, dtypes
            ):
                run_qr_dygraph(tensor_shape, mode, dtype)

    def test_static(self):
        def run_qr_static(shape, mode, dtype):
            if dtype == "float32":
                np_dtype = np.float32
            elif dtype == "float64":
                np_dtype = np.float64
            elif dtype == "complex64":
                np_dtype = np.complex64
            elif dtype == "complex128":
                np_dtype = np.complex128
            if np.issubdtype(np_dtype, np.complexfloating):
                a_dtype = np.float32 if np_dtype == np.complex64 else np.float64
                a_real = np.random.rand(*shape).astype(a_dtype)
                a_imag = np.random.rand(*shape).astype(a_dtype)
                a = a_real + 1j * a_imag
            else:
                a = np.random.rand(*shape).astype(np_dtype)
            places = []
            places.append(paddle.CPUPlace())
            if core.is_compiled_with_cuda():
                places.append(paddle.CUDAPlace(0))
            for place in places:
                with static.program_guard(static.Program(), static.Program()):
                    if mode == "r":
                        np_r = np.linalg.qr(a, mode=mode)
                    else:
                        np_q, np_r = np.linalg.qr(a, mode=mode)
                    x = paddle.static.data(
                        name="input", shape=shape, dtype=dtype
                    )
                    if mode == "r":
                        r = paddle.linalg.qr(x, mode=mode)
                        exe = base.Executor(place=place)
                        fetches = exe.run(
                            feed={"input": a},
                            fetch_list=[r],
                        )
                        np.testing.assert_allclose(
                            fetches[0], np_r, rtol=1e-05, atol=1e-05
                        )
                    else:
                        q, r = paddle.linalg.qr(x, mode=mode)
                        exe = base.Executor(place=place)
                        fetches = exe.run(
                            feed={"input": a},
                            fetch_list=[q, r],
                        )
                        np.testing.assert_allclose(
                            fetches[0], np_q, rtol=1e-05, atol=1e-05
                        )
                        np.testing.assert_allclose(
                            fetches[1], np_r, rtol=1e-05, atol=1e-05
                        )

        with static_guard():
            np.random.seed(7)
            tensor_shapes = [
                (0, 3),
                (3, 5),
                (5, 5),
                (5, 3),  # 2-dim Tensors
                (0, 3, 5),
                (4, 0, 5),
                (5, 4, 0),
                (4, 5, 3),  # 3-dim Tensors
                (0, 5, 3, 5),
                (2, 5, 3, 5),
                (3, 5, 5, 5),
                (4, 5, 5, 3),  # 4-dim Tensors
            ]
            modes = ["reduced", "complete", "r"]
            dtypes = ["float32", "float64", 'complex64', 'complex128']
            for tensor_shape, mode, dtype in itertools.product(
                tensor_shapes, modes, dtypes
            ):
                run_qr_static(tensor_shape, mode, dtype)


if __name__ == "__main__":
    unittest.main()
