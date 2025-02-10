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
import sys
import unittest
from os.path import dirname

import numpy as np

os.environ['FLAGS_prim_all'] = 'true'
os.environ['FLAGS_prim_enable_dynamic'] = 'true'
os.environ['FLAGS_print_ir'] = '1'
os.environ['FLAGS_enable_pir_api'] = '1'
os.environ['FLAGS_use_cinn'] = '1'
os.environ['FLAGS_deny_cinn_ops'] = 'slice;'


import paddle
from paddle import nn

sys.path.append(dirname(dirname(__file__)))

import utils


class BroadcastSubgraph(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self, x, in_1, in_2, in_3, in_5
    ):  # [512, 512] [512, 1], [512, 512], [512, 1] , [512, 512]
        t0 = paddle.transpose(x, [1, 0])
        t2 = in_1 * in_1
        t3 = paddle.expand(t2, [512, 512])
        t4 = in_2 / t3
        t5 = t4 * -1
        t6 = t5 * t0
        t7 = t6.sum([1], keepdim=False)
        t8 = t7.reshape([512, 1])
        t10 = 1 / in_1
        t12 = t10 * t0
        t14 = t12 * in_2
        t16 = 1 / in_3
        t17 = t16 * t8
        t18 = t17 * in_5
        t19 = t17 * in_5
        t21 = t14 + t18
        t22 = t21 + t19
        return t22


class TestIfSubgraph(unittest.TestCase):
    def setUp(self):
        paddle.seed(2024)
        self.prepare_data()

    def prepare_data(self):
        # self.shape = [22, 64, 56]
        self.x = paddle.randn([512, 512], dtype="float32")
        self.x.stop_gradient = False

        self.in_1 = paddle.randn([512, 1], dtype="float32")
        self.in_1.stop_gradient = False

        self.in_2 = paddle.randn([512, 512], dtype="float32")
        self.in_2.stop_gradient = False

        self.in_3 = paddle.randn([512, 1], dtype="float32")
        self.in_3.stop_gradient = False

        self.in_5 = paddle.randn([512, 512], dtype="float32")
        self.in_5.stop_gradient = False

    def check_jit_kernel_info(self, static_fn):
        utils.check_jit_kernel_number(static_fn, 2)
        utils.check_jit_kernel_structure(
            static_fn,
            {
                'if_0': {utils.JIT_KERNEL_NAME: 1},
                'else_0': {},
                utils.JIT_KERNEL_NAME: 1,
            },
        )

    def eval(self, use_cinn):
        net = BroadcastSubgraph()

        net = utils.apply_to_static(net, use_cinn)
        net.eval()
        out = net(self.x, self.in_1, self.in_2, self.in_3, self.in_5)
        return out

    def test_eval(self):
        dy_out = self.eval(use_cinn=False)
        cinn_out = self.eval(use_cinn=True)
        np.testing.assert_allclose(
            cinn_out.numpy(), dy_out.numpy(), atol=1e-2, rtol=1e-2
        )


if __name__ == '__main__':
    unittest.main()
