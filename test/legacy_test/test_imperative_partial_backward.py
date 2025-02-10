# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np

import paddle
from paddle import base


class TestImperativePartialBackward(unittest.TestCase):
    def test_partial_backward(self):
        with base.dygraph.guard():
            x = np.random.randn(2, 4, 5).astype("float32")
            x = paddle.to_tensor(x)
            linear1 = paddle.nn.Linear(5, 10)
            linear2 = paddle.nn.Linear(5, 10)

            y = linear1(x[:, :2])
            z = linear2(x[:, 2:])
            loss = paddle.mean(y)
            loss.backward()

            for param in linear1.parameters():
                self.assertIsNotNone(param._grad_ivar())

            for param in linear2.parameters():
                self.assertIsNone(param._grad_ivar())

            optimizer = paddle.optimizer.Adam(
                parameters=(linear1.parameters() + linear2.parameters())
            )
            _, params_grads = optimizer.minimize(loss)

            self.assertListEqual(
                sorted([p.name for p in linear1.parameters()]),
                sorted([p_g[0].name for p_g in params_grads]),
            )

            linear1.clear_gradients()
            linear2.clear_gradients()


if __name__ == '__main__':
    unittest.main()
