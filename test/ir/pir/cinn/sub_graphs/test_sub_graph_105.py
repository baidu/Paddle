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

# repo: PaddleDetection
# model: configs^ppyoloe^ppyoloe_crn_l_300e_coco_single_dy2st_train
# api||paddle.nn.functional.conv._conv_nd,api||paddle.tensor.ops.sigmoid,method||flatten,method||transpose,method||flatten,method||transpose
import unittest

import numpy as np

import paddle


class SIR167(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.var_838 = self.create_parameter(
            shape=[68, 192, 3, 3],
            dtype=paddle.float32,
        )
        self.var_839 = self.create_parameter(
            shape=[68],
            dtype=paddle.float32,
        )

    def forward(
        self,
        var_830,  # (shape: [1, 192, 56, 56], dtype: paddle.float32, stop_gradient: False)
        var_831,  # (shape: [1, 80, 56, 56], dtype: paddle.float32, stop_gradient: False)
    ):
        var_840 = paddle.nn.functional.conv._conv_nd(
            var_830,
            self.var_838,
            bias=self.var_839,
            stride=[1, 1],
            padding=[1, 1],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_841 = paddle.tensor.ops.sigmoid(var_831)
        var_842 = var_841.flatten(2)
        var_843 = var_842.transpose([0, 2, 1])
        var_844 = var_840.flatten(2)
        var_845 = var_844.transpose([0, 2, 1])
        return var_840, var_841, var_843, var_845


class TestSIR167(unittest.TestCase):
    def setUp(self):
        self.inputs = (
            paddle.rand(shape=[1, 192, 56, 56], dtype=paddle.float32),
            paddle.rand(shape=[1, 80, 56, 56], dtype=paddle.float32),
        )
        self.net = SIR167()

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
