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

# repo: llm_sub_grpah
# model: qwen
# method:__getitem__||method:__getitem__||method:__mul__||method:__getitem__||method:__getitem__||method:__neg__||api:paddle.tensor.manipulation.concat||method:__mul__||method:__add__||method:__mul__||method:__getitem__||method:__getitem__||method:__neg__||api:paddle.tensor.manipulation.concat||method:__mul__||method:__add__
import unittest

import numpy as np

import paddle


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        var_0,  # (shape: [1, 64, 8, 128], dtype: paddle.float32, stop_gradient: False)
        var_1,  # (shape: [1, 64, 1, 128], dtype: paddle.float32, stop_gradient: True)
        var_2,  # (shape: [1, 64, 1, 128], dtype: paddle.float32, stop_gradient: True)
        var_3,  # (shape: [1, 64, 8, 128], dtype: paddle.float32, stop_gradient: False)
    ):
        var_4 = var_1[
            (
                slice(None, None, None),
                slice(None, 64, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        var_5 = var_2[
            (
                slice(None, None, None),
                slice(None, 64, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        var_6 = var_3 * var_4
        var_7 = var_3[
            (
                Ellipsis,
                slice(None, 64, None),
            )
        ]
        var_8 = var_3[
            (
                Ellipsis,
                slice(64, None, None),
            )
        ]
        var_9 = -var_8
        var_10 = paddle.tensor.manipulation.concat([var_9, var_7], axis=-1)
        var_11 = var_10 * var_5
        var_12 = var_6 + var_11
        var_13 = var_0 * var_4
        var_14 = var_0[
            (
                Ellipsis,
                slice(None, 64, None),
            )
        ]
        var_15 = var_0[
            (
                Ellipsis,
                slice(64, None, None),
            )
        ]
        var_16 = -var_15
        var_17 = paddle.tensor.manipulation.concat([var_16, var_14], axis=-1)
        var_18 = var_17 * var_5
        var_19 = var_13 + var_18
        return var_12, var_19


def create_paddle_inputs():
    inputs = (
        paddle.rand(shape=[1, 64, 8, 128], dtype=paddle.float32),
        paddle.rand(shape=[1, 64, 1, 128], dtype=paddle.float32),
        paddle.rand(shape=[1, 64, 1, 128], dtype=paddle.float32),
        paddle.rand(shape=[1, 64, 8, 128], dtype=paddle.float32),
    )
    return inputs


class TestLayer(unittest.TestCase):
    def setUp(self):
        self.inputs = create_paddle_inputs()
        self.net = LayerCase()

    def train(self, net, to_static, with_prim=False, with_cinn=False):
        if to_static:
            paddle.set_flags({'FLAGS_prim_all': with_prim})
            if with_cinn:
                build_strategy = paddle.static.BuildStrategy()
                build_strategy.build_cinn_pass = True
                net = paddle.jit.to_static(
                    net, build_strategy=build_strategy, full_graph=True
                )
            else:
                net = paddle.jit.to_static(net, full_graph=True)
        paddle.seed(123)
        outs = net(*self.inputs)
        return outs

    def test_ast_prim_cinn(self):
        st_out = self.train(self.net, to_static=True)
        cinn_out = self.train(
            self.net, to_static=True, with_prim=True, with_cinn=False
        )
        for st, cinn in zip(
            paddle.utils.flatten(st_out), paddle.utils.flatten(cinn_out)
        ):
            np.testing.assert_allclose(st.numpy(), cinn.numpy(), atol=1e-8)


if __name__ == '__main__':
    unittest.main()
