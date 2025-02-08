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
# model: ppcls^configs^ImageNet^MixNet^MixNet_S
# api:paddle.tensor.manipulation.split||api:paddle.tensor.manipulation.split||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.conv._conv_nd||api:paddle.tensor.manipulation.concat
from base import *  # noqa: F403

from paddle.static import InputSpec


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.parameter_0 = self.create_parameter(
            shape=[90, 1, 9, 9],
            dtype=paddle.float32,
        )
        self.parameter_1 = self.create_parameter(
            shape=[90, 1, 3, 3],
            dtype=paddle.float32,
        )
        self.parameter_2 = self.create_parameter(
            shape=[90, 1, 5, 5],
            dtype=paddle.float32,
        )
        self.parameter_3 = self.create_parameter(
            shape=[90, 1, 7, 7],
            dtype=paddle.float32,
        )

    def forward(
        self,
        var_0,  # (shape: [22, 360, 14, 14], dtype: paddle.float32, stop_gradient: False)
    ):
        var_1, var_2, var_3, var_4 = paddle.tensor.manipulation.split(
            var_0, [90, 90, 90, 90], axis=1
        )
        var_5 = paddle.nn.functional.conv._conv_nd(
            var_1,
            self.parameter_1,
            bias=None,
            stride=[1, 1],
            padding=[1, 1],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=90,
            data_format='NCHW',
            channel_dim=1,
            op_type='depthwise_conv2d',
            use_cudnn=False,
        )
        var_6 = paddle.nn.functional.conv._conv_nd(
            var_2,
            self.parameter_2,
            bias=None,
            stride=[1, 1],
            padding=[2, 2],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=90,
            data_format='NCHW',
            channel_dim=1,
            op_type='depthwise_conv2d',
            use_cudnn=False,
        )
        var_7 = paddle.nn.functional.conv._conv_nd(
            var_3,
            self.parameter_3,
            bias=None,
            stride=[1, 1],
            padding=[3, 3],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=90,
            data_format='NCHW',
            channel_dim=1,
            op_type='depthwise_conv2d',
            use_cudnn=False,
        )
        var_8 = paddle.nn.functional.conv._conv_nd(
            var_4,
            self.parameter_0,
            bias=None,
            stride=[1, 1],
            padding=[4, 4],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=90,
            data_format='NCHW',
            channel_dim=1,
            op_type='depthwise_conv2d',
            use_cudnn=False,
        )
        var_9 = paddle.tensor.manipulation.concat(
            (var_5, var_6, var_7, var_8),
            axis=1,
        )
        return var_9


class TestLayer(TestBase):
    def init(self):
        self.input_specs = [
            InputSpec(
                shape=(-1, 360, -1, -1),
                dtype=paddle.float32,
                name=None,
                stop_gradient=False,
            )
        ]
        self.inputs = (
            paddle.rand(shape=[22, 360, 14, 14], dtype=paddle.float32),
        )
        self.net = LayerCase


if __name__ == '__main__':
    unittest.main()
