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

import unittest
from collections import OrderedDict

import numpy as np

import paddle
from paddle import base


class MyLayer(paddle.nn.Layer):
    def __init__(self, num_stacked_param):
        super().__init__()
        # create ParameterDict with iterable Parameters
        self.params = self.paddle_imperative_ParameterDict(num_stacked_param)

    def paddle_imperative_ParameterDict(self, num_stacked_param):
        return paddle.nn.ParameterDict(
            [
                (
                    't' + str(i),
                    paddle.create_parameter(shape=[2, 2], dtype='float32'),
                )
                for i in range(num_stacked_param)
            ]
        )

    def forward(self, x):
        for i, key in enumerate(self.params):
            x = paddle.matmul(x, self.params[key])
        return x


class TestImperativeContainerParameterDict(unittest.TestCase):
    def parameter_dict(self):
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )
        data_np = np.random.uniform(-1, 1, [5, 2]).astype('float32')
        with base.dygraph.guard():
            x = paddle.to_tensor(data_np)
            num_stacked_param = 4
            model = MyLayer(num_stacked_param)
            self.assertEqual(len(model.params), num_stacked_param)
            res = model(x)
            self.assertListEqual(res.shape, [5, 2])
            loss = paddle.mean(res)
            loss.backward()

            model.params['t' + str(num_stacked_param - 1)] = (
                paddle.create_parameter(shape=[2, 3], dtype='float32')
            )
            res = model(x)
            self.assertListEqual(res.shape, [5, 3])
            parameter = OrderedDict(
                [
                    (
                        't' + str(num_stacked_param),
                        paddle.create_parameter(shape=[3, 4], dtype='float32'),
                    )
                ]
            )
            model.params.update(parameter)
            self.assertEqual(len(model.params), num_stacked_param + 1)
            res = model(x)
            self.assertListEqual(res.shape, [5, 4])
            loss = paddle.mean(res)
            loss.backward()

    def test_parameter_dict(self):
        self.parameter_dict()


if __name__ == '__main__':
    unittest.main()
