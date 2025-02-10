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
# model: ppcls^configs^ImageNet^LeViT^LeViT_128
# api:paddle.tensor.manipulation.reshape||api:paddle.tensor.linalg.transpose||api:paddle.tensor.linalg.transpose||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||api:paddle.tensor.manipulation.concat||api:paddle.tensor.linalg.transpose||method:reshape||api:paddle.tensor.linalg.transpose||api:paddle.tensor.linalg.matmul||method:__mul__||method:__add__||api:paddle.nn.functional.activation.softmax||api:paddle.tensor.linalg.matmul||api:paddle.tensor.linalg.transpose||api:paddle.tensor.manipulation.reshape
from base import *  # noqa: F403


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.parameter_0 = self.create_parameter(
            shape=[16, 49],
            dtype=paddle.float32,
        )

    def forward(
        self,
        var_0,  # (shape: [22, 16, 256], dtype: paddle.float32, stop_gradient: False)
        var_1,  # (shape: [22, 16, 49, 16], dtype: paddle.float32, stop_gradient: False)
        var_2,  # (shape: [22, 16, 49, 64], dtype: paddle.float32, stop_gradient: False)
        var_3,  # (shape: [16, 49], dtype: paddle.int64, stop_gradient: True)
    ):
        var_4 = paddle.tensor.manipulation.reshape(var_0, [22, 16, 16, 16])
        var_5 = paddle.tensor.linalg.transpose(var_4, perm=[0, 2, 1, 3])
        var_6 = paddle.tensor.linalg.transpose(self.parameter_0, (1, 0))
        concat_list = []
        for var in var_3:
            concat_list.append(paddle.tensor.manipulation.gather(var_6, var))
        var_7 = paddle.tensor.manipulation.concat(concat_list)
        var_8 = paddle.tensor.linalg.transpose(var_7, (1, 0))
        var_9 = var_8.reshape((0, 16, 49))
        var_10 = paddle.tensor.linalg.transpose(var_1, perm=[0, 1, 3, 2])
        var_11 = paddle.tensor.linalg.matmul(var_5, var_10)
        var_12 = var_11 * 0.25
        var_13 = var_12 + var_9
        var_14 = paddle.nn.functional.activation.softmax(var_13)
        var_15 = paddle.tensor.linalg.matmul(var_14, var_2)
        var_16 = paddle.tensor.linalg.transpose(var_15, perm=[0, 2, 1, 3])
        var_17 = paddle.tensor.manipulation.reshape(var_16, [22, -1, 1024])
        return var_17


class TestLayer(TestBase):
    def init(self):
        # FIXME(Aurelius84): CI timeout > 600 s
        # self.input_specs = [
        #     InputSpec(
        #         shape=(-1, -1, -1),
        #         dtype=paddle.float32,
        #         name=None,
        #         stop_gradient=False,
        #     ),
        #     InputSpec(
        #         shape=(-1, -1, -1, -1),
        #         dtype=paddle.float32,
        #         name=None,
        #         stop_gradient=False,
        #     ),
        #     InputSpec(
        #         shape=(-1, -1, -1, -1),
        #         dtype=paddle.float32,
        #         name=None,
        #         stop_gradient=False,
        #     ),
        #     InputSpec(
        #         shape=(-1, -1),
        #         dtype=paddle.int64,
        #         name=None,
        #         stop_gradient=True,
        #     ),
        # ]
        self.inputs = (
            paddle.rand(shape=[22, 16, 256], dtype=paddle.float32),
            paddle.rand(shape=[22, 16, 49, 16], dtype=paddle.float32),
            paddle.rand(shape=[22, 16, 49, 64], dtype=paddle.float32),
            paddle.randint(low=0, high=10, shape=[16, 49], dtype=paddle.int64),
        )
        self.net = LayerCase
        self.with_train = True


if __name__ == '__main__':
    unittest.main()
