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

# repo: PaddleClas
# model: ppcls^configs^ImageNet^ResNeXt101_wsl^ResNeXt101_32x8d_wsl
# api||paddle.nn.functional.pooling.adaptive_avg_pool2d,api||paddle.tensor.manipulation.squeeze,api||paddle.nn.functional.common.linear
import unittest

import numpy as np

import paddle


class SIR88(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.var_473 = self.create_parameter(
            shape=[2048, 1000],
            dtype=paddle.float32,
        )
        self.var_474 = self.create_parameter(
            shape=[1000],
            dtype=paddle.float32,
        )

    def forward(
        self,
        var_470,  # (shape: [22, 2048, 7, 7], dtype: paddle.float32, stop_gradient: False)
    ):
        var_471 = paddle.nn.functional.pooling.adaptive_avg_pool2d(
            var_470, output_size=1, data_format='NCHW', name=None
        )
        var_472 = paddle.tensor.manipulation.squeeze(var_471, axis=[2, 3])
        var_475 = paddle.nn.functional.common.linear(
            x=var_472, weight=self.var_473, bias=self.var_474, name=None
        )
        return var_475


class TestSIR88(unittest.TestCase):
    def setUp(self):
        self.inputs = (
            paddle.rand(shape=[22, 2048, 7, 7], dtype=paddle.float32),
        )
        self.net = SIR88()

    def train(self, net, to_static, with_prim=False, with_cinn=False):
        paddle.set_flags({'FLAGS_prim_all': with_prim})
        if to_static:
            if with_cinn:
                build_strategy = paddle.static.BuildStrategy()
                build_strategy.build_cinn_pass = True
                net = paddle.jit.to_static(
                    net, build_strategy=build_strategy, full_graph=True
                )
            else:
                net = paddle.jit.to_static(net, full_graph=True)
        outs = net(*self.inputs)
        return outs

    def test_ast_prim_cinn(self):
        st_out = self.train(self.net, to_static=True)
        cinn_out = self.train(
            self.net, to_static=True, with_prim=True, with_cinn=True
        )
        for st, cinn in zip(
            paddle.utils.flatten(st_out), paddle.utils.flatten(cinn_out)
        ):
            np.testing.assert_allclose(st.numpy(), cinn.numpy(), atol=1e-8)


if __name__ == '__main__':
    unittest.main()
