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

os.environ['FLAGS_prim_enable_dynamic'] = 'true'

import numpy as np

import paddle
from paddle import nn
from paddle.static import InputSpec

sys.path.append(dirname(dirname(__file__)))

import utils


class IfSubgraph(nn.Layer):
    def __init__(self):
        super().__init__()

    def exp_sub(self, x):
        y = paddle.exp(x)
        return y - x

    def forward(self, x):
        if x.shape[-1] > 1:
            x = self.exp_sub(x)
        x = paddle.sin(x)
        return x


class TestIfSubgraph(unittest.TestCase):
    def setUp(self):
        paddle.seed(2024)
        self.prepare_data()

    def prepare_data(self):
        self.shape = [1, 2048]
        self.x = paddle.randn(self.shape, dtype="float32")
        self.x.stop_gradient = False

    def check_jit_kernel_info(self, static_fn):
        utils.check_jit_kernel_number(static_fn, 3)
        utils.check_jit_kernel_structure(
            static_fn,
            {
                'if_0': {utils.JIT_KERNEL_NAME: 1},
                'else_0': {},
                utils.JIT_KERNEL_NAME: 2,
            },
        )

    def eval(self, use_cinn):
        net = IfSubgraph()
        input_spec = [
            InputSpec(shape=[None, None], dtype="bool"),
        ]
        net = utils.apply_to_static(net, use_cinn, input_spec)
        net.eval()
        out = net(self.x)
        if use_cinn:
            self.check_jit_kernel_info(net.forward)
        return out

    def test_eval(self):
        dy_out = self.eval(use_cinn=False)
        cinn_out = self.eval(use_cinn=True)
        np.testing.assert_allclose(
            cinn_out.numpy(), dy_out.numpy(), atol=1e-6, rtol=1e-6
        )


if __name__ == '__main__':
    unittest.main()
