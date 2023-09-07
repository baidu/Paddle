# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from eager_op_test import OpTest, convert_float_to_uint16
from scipy.special import psi

import paddle
from paddle import base, static
from paddle.base import core


class TestDigammaOp(OpTest):
    def setUp(self):
        # switch to static
        paddle.enable_static()

        self.op_type = 'digamma'
        self.python_api = paddle.digamma
        self.init_dtype_type()
        shape = (5, 32)
        data = np.random.random(shape).astype(self.dtype) + 1
        self.inputs = {'X': data}
        result = np.ones(shape).astype(self.dtype)
        result = psi(data)
        self.outputs = {'Out': result}

    def init_dtype_type(self):
        self.dtype = np.float64

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['X'], 'Out')


class TestDigammaOpFp32(TestDigammaOp):
    def init_dtype_type(self):
        self.dtype = np.float32

    def test_check_grad_normal(self):
        self.check_grad(['X'], 'Out')


class TestDigammaFP16Op(TestDigammaOp):
    def init_dtype_type(self):
        self.dtype = np.float16


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA or not support bfloat16",
)
class TestDigammaBF16Op(OpTest):
    def setUp(self):
        # switch to static
        paddle.enable_static()

        self.op_type = 'digamma'
        self.python_api = paddle.digamma
        self.init_dtype_type()
        shape = (5, 32)
        data = np.random.random(shape).astype(self.np_dtype) + 1
        self.inputs = {'X': convert_float_to_uint16(data)}
        result = np.ones(shape).astype(self.np_dtype)
        result = psi(data)
        self.outputs = {'Out': convert_float_to_uint16(result)}

    def init_dtype_type(self):
        self.dtype = np.uint16
        self.np_dtype = np.float32

    def test_check_output(self):
        # bfloat16 needs to set the parameter place
        self.check_output_with_place(core.CUDAPlace(0))

    def test_check_grad_normal(self):
        self.check_grad_with_place(core.CUDAPlace(0), ['X'], 'Out')


class TestDigammaAPI(unittest.TestCase):
    def setUp(self):
        # switch to static
        paddle.enable_static()
        # prepare test attrs
        self.dtypes = ["float32", "float64"]
        self.places = [paddle.CPUPlace()]
        if paddle.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))
        self._shape = [8, 3, 32, 32]

    def test_in_static_mode(self):
        def init_input_output(dtype):
            input = np.random.random(self._shape).astype(dtype)
            return {'x': input}, psi(input)

        for dtype in self.dtypes:
            input_dict, sc_res = init_input_output(dtype)
            for place in self.places:
                with static.program_guard(static.Program()):
                    x = static.data(name="x", shape=self._shape, dtype=dtype)
                    out = paddle.digamma(x)

                    exe = static.Executor(place)
                    out_value = exe.run(feed=input_dict, fetch_list=[out.name])
                    np.testing.assert_allclose(out_value[0], sc_res, rtol=1e-05)

    def test_in_dynamic_mode(self):
        for dtype in self.dtypes:
            input = np.random.random(self._shape).astype(dtype)
            sc_res = psi(input)
            for place in self.places:
                # it is more convenient to use `guard` than `enable/disable_**` here
                with base.dygraph.guard(place):
                    input_t = paddle.to_tensor(input)
                    res = paddle.digamma(input_t).numpy()
                    np.testing.assert_allclose(res, sc_res, rtol=1e-05)

    def test_name_argument(self):
        with static.program_guard(static.Program()):
            x = static.data(name="x", shape=self._shape, dtype=self.dtypes[0])
            out = paddle.digamma(x, name="digamma_res")
            self.assertTrue("digamma_res" in out.name)

    def test_dtype_error(self):
        # in static graph mode
        with self.assertRaises(TypeError):
            with static.program_guard(static.Program()):
                x = static.data(name="x", shape=self._shape, dtype="int32")
                out = paddle.digamma(x, name="digamma_res")

        # in dynamic mode
        with self.assertRaises(RuntimeError):
            with base.dygraph.guard():
                input = np.random.random(self._shape).astype("int32")
                input_t = paddle.to_tensor(input)
                res = paddle.digamma(input_t)


if __name__ == "__main__":
    unittest.main()
