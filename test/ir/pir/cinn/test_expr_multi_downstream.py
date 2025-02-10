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
os.environ['FLAGS_cinn_new_cluster_op_method'] = '1'
os.environ['FLAGS_deny_cinn_ops'] = 'slice;'

import paddle

build_strategy = paddle.static.BuildStrategy()
build_strategy.build_cinn_pass = True


def init():
    var_0 = paddle.rand([216])
    return [var_0]


def func(x):
    t = x.reshape([4, 2, 27])
    sum1 = t.mean(-1, keepdim=True)
    t2 = t - sum1
    t3 = t2 * t2
    t5 = t3.sum(-1, keepdim=True)
    return t, t5


def input_spec():
    return [paddle.static.InputSpec(shape=[216], dtype='float32', name='var_0')]


class TestCase(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def compare_result(self, dy_compute, data_init):
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

    def test_case(self):
        self.compare_result(func, init)


if __name__ == "__main__":
    unittest.main()
