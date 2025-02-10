# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import numpy

os.environ['FLAGS_prim_all'] = 'true'
os.environ['FLAGS_prim_enable_dynamic'] = 'true'
os.environ['FLAGS_print_ir'] = '1'
os.environ['FLAGS_enable_pir_api'] = '1'
os.environ['FLAGS_use_cinn'] = '1'

from utils import check_jit_kernel_number

import paddle

build_strategy = paddle.static.BuildStrategy()
build_strategy.build_cinn_pass = True


class TestFusion(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def compare_result(
        self, dy_compute, data_init, input_spec, expect_fusion_num
    ):
        static_compute = paddle.jit.to_static(
            full_graph=True,
            build_strategy=build_strategy,
            input_spec=input_spec(),
        )(dy_compute)
        inputs = data_init()
        dy_out = dy_compute(*inputs)
        st_out = static_compute(*inputs)
        if isinstance(dy_out, paddle.Tensor):
            numpy.testing.assert_allclose(dy_out, st_out, atol=1e-5, rtol=1e-6)
            return
        for d, s in zip(dy_out, st_out):
            numpy.testing.assert_allclose(d, s, atol=1e-5, rtol=1e-6)
        check_jit_kernel_number(static_compute, expect_fusion_num)

    def test_softmax(self):
        def func(var_40, var_106):
            var_109 = paddle.cast(var_40, dtype='float32')
            var_110 = var_106 + var_109
            var_111 = paddle.cast(var_110, dtype='float32')
            var_112 = paddle.max(var_111, keepdim=True, axis=[-1])
            var_113 = var_111 - var_112
            var_114 = paddle.exp(var_113)
            var_115 = paddle.sum(var_114, keepdim=True, axis=[-1])
            var_116 = var_114 / var_115
            var_117 = paddle.cast(var_116, dtype='float32')
            return var_116, var_117

        def init():
            var_40 = paddle.rand([1, 1, 17, 17])
            var_40 = paddle.cast(var_40, 'float64')
            var_106 = paddle.rand([1, 32, 17, 17])
            return var_40, var_106

        def input_spec():
            return [
                paddle.static.InputSpec(
                    shape=[1, 1, 17, 17], dtype='float64', name='var_40'
                ),
                paddle.static.InputSpec(
                    shape=[1, 32, 17, 17], dtype='float32', name='var_106'
                ),
            ]

        self.compare_result(func, init, input_spec, 1)

    def test_horizontal_1(self):
        def func(x):
            ret1 = x * 2 * 2
            ret2 = paddle.reshape(ret1, [1, 1, 17, 17])
            return ret1, ret2

        def init():
            x = paddle.rand([17, 17])
            return (x,)

        def input_spec():
            return None

        self.compare_result(func, init, input_spec, 1)

    def test_horizontal_2(self):
        def func(x):
            ret1 = x * 2 * 2
            ret2 = paddle.reshape(ret1, [1, 17, 1, 1, 17, 1, 1])
            return ret1, ret2

        def init():
            x = paddle.rand([17, 17])
            return (x,)

        def input_spec():
            return None

        self.compare_result(func, init, input_spec, 1)


if __name__ == "__main__":
    pass
