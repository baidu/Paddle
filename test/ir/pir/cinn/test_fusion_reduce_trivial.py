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
os.environ['FLAGS_deny_cinn_ops'] = 'slice;'

from utils import check_jit_kernel_number

import paddle

build_strategy = paddle.static.BuildStrategy()
build_strategy.build_cinn_pass = True


class TestFusion(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def compare_result(self, dy_compute, data_init, expect_fusion_num):
        static_compute = paddle.jit.to_static(
            full_graph=True, build_strategy=build_strategy
        )(dy_compute)
        inputs = data_init()
        dy_out = dy_compute(*inputs)
        st_out = static_compute(*inputs)
        numpy.testing.assert_allclose(dy_out, st_out, atol=1e-5, rtol=1e-6)
        check_jit_kernel_number(static_compute, expect_fusion_num)

    def test_R_T_can_fuse(self):
        def func(x):
            o = x.sum(-1, keepdim=True)
            r = x + o
            return r

        def init():
            return [paddle.rand((32, 33, 34))]

        self.compare_result(func, init, 1)

    def test_R_T_can_fuse_2(self):
        # dim smaller
        def func(x):
            o = x.sum(-1, keepdim=True)
            o = o.reshape([1, -1])
            return o * 2

        def init():
            return [paddle.rand((32, 33, 34))]

        self.compare_result(func, init, 1)

    def test_R_T_can_not_fuse(self):
        # dim smaller
        def func(x):
            o = x.sum(-1, keepdim=True)
            m = o + x
            m = m.reshape([1, -1])
            return m * 2

        def init():
            return [paddle.rand((32, 33, 34))]

        self.compare_result(func, init, 2)


if __name__ == "__main__":
    unittest.main()
