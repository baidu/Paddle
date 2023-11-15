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

import unittest

import numpy as np
from op_test import OpTest, convert_float_to_uint16

import paddle
from paddle.base import core
from paddle.pir_utils import test_with_pir_api


def output_hist(out):
    hist, _ = np.histogram(out, bins=2)
    hist = hist.astype("float32")
    hist /= float(out.size)
    prob = 0.5 * np.ones(2)
    return hist, prob


class TestBernoulliOp(OpTest):
    def setUp(self):
        self.op_type = "bernoulli"
        self.init_dtype()
        self.init_test_case()
        self.inputs = {"X": self.x}
        self.attrs = {}
        self.outputs = {"Out": self.out}

    def init_dtype(self):
        self.dtype = np.float32

    def init_test_case(self):
        self.x = np.random.uniform(size=(1000, 784)).astype(self.dtype)
        self.out = np.zeros((1000, 784)).astype(self.dtype)

    def test_check_output(self):
        self.check_output_customized(self.verify_output, check_pir=True)

    def verify_output(self, outs):
        hist, prob = output_hist(np.array(outs[0]))
        np.testing.assert_allclose(hist, prob, rtol=0, atol=0.01)


class TestBernoulliApi(unittest.TestCase):
    def test_dygraph(self):
        paddle.disable_static()
        x = paddle.rand([1024, 1024])
        out = paddle.bernoulli(x)
        paddle.enable_static()
        hist, prob = output_hist(out.numpy())
        np.testing.assert_allclose(hist, prob, rtol=0, atol=0.01)

    @test_with_pir_api
    def test_static(self):
        x = paddle.rand([1024, 1024])
        out = paddle.bernoulli(x)
        exe = paddle.static.Executor(paddle.CPUPlace())
        out = exe.run(
            paddle.static.default_main_program(), fetch_list=[out]
        )
        hist, prob = output_hist(out[0])
        np.testing.assert_allclose(hist, prob, rtol=0, atol=0.01)


class TestRandomValue(unittest.TestCase):
    def test_fixed_random_number(self):
        # Test GPU Fixed random number, which is generated by 'curandStatePhilox4_32_10_t'
        if not paddle.is_compiled_with_cuda():
            return

        print("Test Fixed Random number on GPU------>")
        paddle.disable_static()
        paddle.set_device('gpu')
        paddle.seed(100)
        np.random.seed(100)

        x_np = np.random.rand(32, 1024, 1024)

        x = paddle.to_tensor(x_np, dtype='float64')
        y = paddle.bernoulli(x).numpy()
        index0, index1, index2 = np.nonzero(y)
        self.assertEqual(np.sum(index0), 260028995)
        self.assertEqual(np.sum(index1), 8582429431)
        self.assertEqual(np.sum(index2), 8581445798)
        expect = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        np.testing.assert_array_equal(y[16, 500, 500:510], expect)

        x = paddle.to_tensor(x_np, dtype='float32')
        y = paddle.bernoulli(x).numpy()
        index0, index1, index2 = np.nonzero(y)
        self.assertEqual(np.sum(index0), 260092343)
        self.assertEqual(np.sum(index1), 8583509076)
        self.assertEqual(np.sum(index2), 8582778540)
        expect = [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0]
        np.testing.assert_array_equal(y[16, 500, 500:510], expect)

        paddle.enable_static()


class TestBernoulliFP16Op(TestBernoulliOp):
    def init_dtype(self):
        self.dtype = np.float16


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not complied with CUDA and not support the bfloat16",
)
class TestBernoulliBF16Op(TestBernoulliOp):
    def init_dtype(self):
        self.dtype = np.uint16

    def test_check_output(self):
        place = core.CUDAPlace(0)
        self.check_output_with_place_customized(self.verify_output, place)

    def init_test_case(self):
        self.x = convert_float_to_uint16(
            np.random.uniform(size=(1000, 784)).astype("float32")
        )
        self.out = convert_float_to_uint16(
            np.zeros((1000, 784)).astype("float32")
        )

    def verify_output(self, outs):
        hist, prob = output_hist(np.array(outs[0]))
        np.testing.assert_allclose(hist, prob, atol=0.01)


if __name__ == "__main__":
    unittest.main()
