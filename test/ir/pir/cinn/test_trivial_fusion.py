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

import paddle

build_strategy = paddle.static.BuildStrategy()
build_strategy.build_cinn_pass = True


def generate_input_spec(rank_dtype_list):
    input_spec = []
    for rank, dtype in rank_dtype_list:
        input_spec.append(
            paddle.static.InputSpec(shape=[None] * rank, dtype=dtype)
        )
    return input_spec


class TestTrivialFusion(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def compare_result(self, dy_compute, input_spec, data_init):
        inputs = data_init()
        dy_out = dy_compute(*inputs)
        static_compute = paddle.jit.to_static(
            full_graph=True,
            build_strategy=build_strategy,
            input_spec=input_spec,
        )(dy_compute)
        st_out = static_compute(*inputs)
        for a, b in zip(
            paddle.utils.flatten(dy_out), paddle.utils.flatten(st_out)
        ):
            numpy.testing.assert_allclose(a, b, atol=1e-5, rtol=1e-6)

    def test_generate_shape_concat(self):
        def func(x, y, z):
            x = paddle.cast(x, 'int64')
            y = paddle.cast(y, 'int64')
            a = paddle.shape(z)[0:1]
            b = paddle.concat([a, x, y], axis=0)
            return b

        def init():
            x = paddle.rand([1])
            y = paddle.rand([1])
            z = paddle.rand([32, 32, 32])
            return x, y, z

        def input_spec():
            return [
                paddle.static.InputSpec(shape=[], dtype='float32', name='x'),
                paddle.static.InputSpec(shape=[], dtype='float32', name='y'),
                paddle.static.InputSpec(
                    shape=[-1, -1, -1], dtype='float32', name='z'
                ),
            ]

        self.compare_result(func, input_spec(), init)


if __name__ == "__main__":
    unittest.main()
