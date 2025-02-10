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
        numpy.testing.assert_allclose(dy_out, st_out, atol=1e-5, rtol=1e-6)

    def test_simple_trivial_fusions(self):
        def func(x):
            x = x * 2
            x = x + 1
            x = paddle.nn.functional.relu(x)
            x = paddle.transpose(x, perm=[0, 2, 1])
            x = x.reshape((-1, 128))
            return x

        def init():
            x = paddle.rand((32, 32, 128))
            return (x,)

        input_spec = generate_input_spec([(3, 'float32')])
        self.compare_result(func, input_spec, init)

    def test_trivial_fusion_slice_and_concat(self):
        def func(x, y):
            x = x * 2
            y = y * 2
            x = x[:, :, :64]
            y = y[:, :, :64]
            z = paddle.concat([x, y], axis=-1)
            return z

        def init():
            x = paddle.rand((32, 32, 128))
            y = paddle.rand((32, 32, 128))
            return (x, y)

        input_spec = generate_input_spec([(3, 'float32'), (3, 'float32')])
        self.compare_result(func, input_spec, init)

    def test_trivial_fusion_gather_nd(self):
        def func(x, y):
            x = x * 2
            output = paddle.gather_nd(x, y)
            return output

        def init():
            x = paddle.to_tensor(
                [[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]]
            )
            index = paddle.to_tensor([[0, 1]])
            return (x, index)

        input_spec = [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 2], dtype='int32'),
        ]
        self.compare_result(func, input_spec, init)

    def test_broadcast(self):
        def func(x, y):
            output = x + y
            return output

        def init():
            x = paddle.rand((32, 1))
            y = paddle.rand((1, 32))
            return (x, y)

        input_spec = generate_input_spec([(2, 'float32'), (2, 'float32')])
        self.compare_result(func, input_spec, init)


if __name__ == "__main__":
    unittest.main()
