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

import sys
import unittest
from os.path import dirname

import paddle
from paddle import nn
from paddle.static import InputSpec

sys.path.append(dirname(dirname(__file__)))

import utils


class ConcatSliceScaleNet(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        x_shape = paddle.shape(x)
        # Use 'y' to generate 'cond' and 'right' to avoid
        # useless operations in paddle.where api.
        cond = y.cast(dtype="bool")
        right = y

        z = paddle.where(cond, y, right)
        out0 = paddle.concat([x, z], axis=1)
        out1 = out0[x_shape[1] :]
        out2 = out1 * 1
        return out2


class TestConcatSliceScale(unittest.TestCase):
    def setUp(self):
        paddle.seed(2024)
        self.prepare_data()

    def prepare_data(self):
        self.x = paddle.randint(0, 100, [32, 128], dtype="int64")
        self.y = paddle.randint(0, 100, [32, 1], dtype="int64")

    def check_jit_kernel_info(self, static_fn):
        utils.check_jit_kernel_number(static_fn, 1)
        utils.check_jit_kernel_structure(static_fn, {utils.JIT_KERNEL_NAME: 1})

    def eval(self, use_cinn):
        net = ConcatSliceScaleNet()
        input_spec = [
            InputSpec(shape=[None, None], dtype="int64"),
            InputSpec(shape=[None, 1], dtype="int64"),
        ]
        net = utils.apply_to_static(net, use_cinn, input_spec)
        net.eval()
        out = net(self.x, self.y)
        if use_cinn:
            self.check_jit_kernel_info(net.forward)
        return out

    def test_eval(self):
        pass
        # dy_out = self.eval(use_cinn=False)
        # if utils.unittest_use_cinn():
        #     cinn_out = self.eval(use_cinn=True)
        #     np.testing.assert_allclose(
        #         cinn_out.numpy(), dy_out.numpy(), atol=1e-6, rtol=1e-6
        #     )


if __name__ == '__main__':
    unittest.main()
