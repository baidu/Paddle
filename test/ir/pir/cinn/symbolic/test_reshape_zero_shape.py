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

os.environ['FLAGS_prim_all'] = 'true'
os.environ['FLAGS_prim_enable_dynamic'] = 'true'
os.environ['FLAGS_print_ir'] = '1'
os.environ['FLAGS_enable_pir_api'] = '1'
os.environ['FLAGS_use_cinn'] = '1'
os.environ['FLAGS_deny_cinn_ops'] = 'slice;'


import numpy as np

import paddle
from paddle import nn
from paddle.static import InputSpec

sys.path.append(dirname(dirname(__file__)))

import utils


class ReshapeZeroShapeNet(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # "O" represents COPY semantics.
        out = paddle.reshape(x, shape=[0, 0, 32, 128])
        out = paddle.sin(out)
        return out


class TestReshapeZeroShape(unittest.TestCase):
    def setUp(self):
        paddle.seed(2024)
        self.prepare_data()

    def prepare_data(self):
        self.shape = [4, 4, 4096]
        self.x = paddle.randn(self.shape, dtype="float32")
        self.x.stop_gradient = False

    def check_jit_kernel_info(self, static_fn):
        utils.check_jit_kernel_number(static_fn, 1)

    def eval(self, use_cinn):
        net = ReshapeZeroShapeNet()
        input_spec = [
            InputSpec(shape=[None, None, 4096], dtype="float32"),
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
