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

import unittest
import paddle
import numpy as np
from op_test import OpTest

paddle.seed(100)


class TestExponentialOp1(OpTest):

    def setUp(self):
        paddle.enable_static()
        self.op_type = "exponential"
        self.config()

        self.attrs = {"lambda": self.lam}
        self.inputs = {'X': np.empty([1024, 1024], dtype=self.dtype)}
        self.outputs = {'Out': np.ones([1024, 1024], dtype=self.dtype)}

    def config(self):
        self.lam = 0.5
        self.dtype = "float64"

    def test_check_output(self):
        self.check_output_customized(self.verify_output)

    def verify_output(self, outs):
        hist1, _ = np.histogram(outs[0], range=(0, 5))
        hist1 = hist1.astype("float32")
        hist1 = hist1 / float(outs[0].size)

        data_np = np.random.exponential(1. / self.lam, [1024, 1024])
        hist2, _ = np.histogram(data_np, range=(0, 5))
        hist2 = hist2.astype("float32")
        hist2 = hist2 / float(data_np.size)

        np.testing.assert_allclose(hist1, hist2, rtol=0.02)

    def test_check_grad_normal(self):
        self.check_grad(
            ['X'],
            'Out',
            user_defined_grads=[np.zeros([1024, 1024], dtype=self.dtype)],
            user_defined_grad_outputs=[
                np.random.rand(1024, 1024).astype(self.dtype)
            ])


class TestExponentialOp2(TestExponentialOp1):

    def config(self):
        self.lam = 0.25
        self.dtype = "float32"


class TestExponentialAPI(unittest.TestCase):

    def test_static(self):
        with paddle.static.program_guard(paddle.static.Program(),
                                         paddle.static.Program()):
            x_np = np.full([10, 10], -1.)
            x = paddle.static.data(name="X", shape=[10, 10], dtype='float64')
            x.exponential_(1.0)

            exe = paddle.static.Executor()
            out = exe.run(paddle.static.default_main_program(),
                          feed={"X": x_np},
                          fetch_list=[x])
            self.assertTrue(np.min(out) >= 0)

    def test_dygraph(self):
        paddle.disable_static()
        x = paddle.full([10, 10], -1., dtype='float32')
        x.stop_gradient = False
        y = 2 * x
        y.exponential_(0.5)
        print(y)
        self.assertTrue(np.min(y.numpy()) >= 0)

        y.backward()
        np.testing.assert_array_equal(x.grad.numpy(), np.zeros([10, 10]))
        paddle.enable_static()

    def test_fixed_random_number(self):
        # Test GPU Fixed random number, which is generated by 'curandStatePhilox4_32_10_t'
        if not paddle.is_compiled_with_cuda():
            return

        # Different GPU generatte different random value. Only test V100 here.
        if not "V100" in paddle.device.cuda.get_device_name():
            return

        print("Test Fixed Random number on V100 GPU------>")
        paddle.disable_static()
        paddle.set_device('gpu')
        paddle.seed(2021)

        x = paddle.empty([64, 3, 1024, 1024], dtype="float32")
        x.exponential_(1.0)
        x_np = x.numpy()
        expect = [
            0.80073667, 0.2249291, 0.07734892, 1.25392, 0.14013891, 0.45736602,
            1.9735607, 0.30490234, 0.57100505, 0.8115938
        ]

        np.testing.assert_allclose(x_np[0, 0, 0, 0:10], expect, rtol=1e-05)
        expect = [
            1.4296371e+00, 9.5411777e-01, 5.2575850e-01, 2.4805880e-01,
            1.2322118e-04, 8.4604341e-01, 2.1111444e-01, 1.4143821e+00,
            2.8194717e-01, 1.1360573e+00
        ]
        np.testing.assert_allclose(x_np[16, 1, 300, 200:210],
                                   expect,
                                   rtol=1e-05)
        expect = [
            1.3448033, 0.35146526, 1.7380928, 0.32012638, 0.10396296,
            0.51344526, 0.15308502, 0.18712929, 0.03888268, 0.20771872
        ]
        np.testing.assert_allclose(x_np[32, 1, 600, 500:510],
                                   expect,
                                   rtol=1e-05)
        expect = [
            0.5107464, 0.20970327, 2.1986802, 1.580056, 0.31036147, 0.43966478,
            0.9056133, 0.30119267, 1.4797124, 1.4319834
        ]
        np.testing.assert_allclose(x_np[48, 2, 900, 800:810],
                                   expect,
                                   rtol=1e-05)
        expect = [
            3.4640615, 1.1019983, 0.41195083, 0.22681557, 0.291846, 0.53617656,
            1.5791925, 2.4645927, 0.04094889, 0.9057725
        ]
        np.testing.assert_allclose(x_np[63, 2, 1023, 1000:1010],
                                   expect,
                                   rtol=1e-05)

        x = paddle.empty([10, 10], dtype="float32")
        x.exponential_(3.0)
        x_np = x.numpy()
        expect = [
            0.02831675, 0.1691551, 0.6798956, 0.69347525, 0.0243443, 0.22180498,
            0.30574575, 0.9839696, 0.2834912, 0.59420055
        ]
        np.testing.assert_allclose(x_np[5, 0:10], expect, rtol=1e-05)

        x = paddle.empty([16, 2, 1024, 768], dtype="float64")
        x.exponential_(0.25)
        x_np = x.numpy()
        expect = [
            10.0541229, 12.67860643, 1.09850734, 7.35289643, 2.65471225,
            3.86217432, 2.97902086, 2.92744479, 2.67927152, 0.19667352
        ]
        np.testing.assert_allclose(x_np[0, 0, 0, 100:110], expect, rtol=1e-05)
        expect = [
            0.68328125, 3.1454553, 0.92158376, 1.95842188, 1.05296941,
            12.93242051, 5.20255978, 3.3588624, 1.57377174, 5.73194183
        ]
        np.testing.assert_allclose(x_np[4, 0, 300, 190:200], expect, rtol=1e-05)
        expect = [
            1.37973974, 3.45036798, 7.94625406, 1.62610973, 0.31032122,
            4.13596493, 1.98494535, 1.13207041, 8.30592769, 2.81460147
        ]
        np.testing.assert_allclose(x_np[8, 1, 600, 300:310], expect, rtol=1e-05)
        expect = [
            2.27710811, 12.25003028, 2.96409124, 4.72405788, 0.67917249,
            4.35856718, 0.46870976, 2.31120149, 9.61595826, 4.64446271
        ]
        np.testing.assert_allclose(x_np[12, 1, 900, 500:510],
                                   expect,
                                   rtol=1e-05)
        expect = [
            0.95883744, 1.57316361, 15.22524512, 20.49559882, 13.70008548,
            3.29430143, 3.90390424, 0.9146657, 0.80972249, 0.33376219
        ]
        np.testing.assert_allclose(x_np[15, 1, 1023, 750:760],
                                   expect,
                                   rtol=1e-05)

        x = paddle.empty([512, 768], dtype="float64")
        x.exponential_(0.3)
        x_np = x.numpy()
        expect = [
            8.79266704, 4.79596009, 2.75480243, 6.04670011, 0.35379556,
            0.76864868, 3.17428251, 0.26556859, 12.22485885, 10.51690383
        ]
        np.testing.assert_allclose(x_np[0, 200:210], expect, rtol=1e-05)
        expect = [
            5.6341126, 0.52243418, 5.36410796, 6.83672002, 11.9243311,
            5.85985566, 5.75169548, 0.13877972, 6.1348385, 3.82436519
        ]
        np.testing.assert_allclose(x_np[300, 400:410], expect, rtol=1e-05)
        expect = [
            4.94883581, 0.56345306, 0.85841585, 1.92287801, 6.10036656,
            1.19524847, 3.64735434, 5.19618716, 2.57467974, 3.49152791
        ]
        np.testing.assert_allclose(x_np[500, 700:710], expect, rtol=1e-05)

        x = paddle.empty([10, 10], dtype="float64")
        x.exponential_(4.0)
        x_np = x.numpy()
        expect = [
            0.15713826, 0.56395964, 0.0680941, 0.00316643, 0.27046853,
            0.19852724, 0.12776634, 0.09642974, 0.51977551, 1.33739699
        ]
        np.testing.assert_allclose(x_np[5, 0:10], expect, rtol=1e-05)

        paddle.enable_static()


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
