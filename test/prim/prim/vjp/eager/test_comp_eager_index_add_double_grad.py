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

import numpy as np
import parameterized as param

import paddle


@param.parameterized_class(
    ('x', 'index', 'axis', 'value', 'cotangent', 'dtype'),
    [
        (
            np.random.randn(4, 3, 2),  # x
            np.random.randint(-3, 3, size=(16,)),  # index
            1,  # axis
            np.random.randint(0, 3, size=(4, 16, 2)),  # value
            np.random.rand(4, 3, 2),  # cotangent
            np.float32,  # dtype
        ),
    ],
)
class TestTakeAlongAxisTanhDoubleGrad(unittest.TestCase):
    def test_index_add_tanh_double_grad(self):
        x_tensor = paddle.to_tensor(
            self.x, dtype=self.dtype, stop_gradient=False
        )
        value_tensor = paddle.to_tensor(
            self.value, dtype=self.dtype, stop_gradient=False
        )
        index_tensor = paddle.to_tensor(self.index, dtype="int64")
        dout_tensor = paddle.to_tensor(
            self.cotangent, dtype=self.dtype, stop_gradient=False
        )
        out = paddle.index_add(x_tensor, index_tensor, self.axis, value_tensor)

        out = paddle.tanh(out)

        dx = paddle.grad(out, x_tensor, dout_tensor, create_graph=True)[0]

        ddx = paddle.grad(dx, dout_tensor, create_graph=True)[0]


if __name__ == '__main__':
    unittest.main()
