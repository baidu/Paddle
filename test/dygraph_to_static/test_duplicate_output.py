#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
from dygraph_to_static_util import (
    dy2static_unittest,
    test_and_compare_with_new_ir,
)

import paddle

np.random.seed(1)

if paddle.base.is_compiled_with_cuda():
    place = paddle.base.CUDAPlace(0)
else:
    place = paddle.base.CPUPlace()


class SimpleNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self._linear = paddle.nn.Linear(1, 1)

    def forward(self, x):
        """forward with duplicate outputs."""
        x = self._linear(x)
        return x, x


@dy2static_unittest
class TestDuplicateOutput(unittest.TestCase):
    """
    TestCase for the transformation from control flow `if/else`
    dependent on tensor in Dygraph into Static `base.layers.cond`.
    """

    def setUp(self):
        self.net = paddle.jit.to_static(SimpleNet())
        self.x = paddle.to_tensor([1.0])

    @test_and_compare_with_new_ir(False)
    def _run_static(self):
        param = self.net.parameters()
        param[0].clear_grad()

        loss0, loss1 = self.net(self.x)
        loss0.backward()

        self.assertEqual(param[0].grad.numpy(), 1.0)

    def test_ast_to_func(self):
        self._run_static()


if __name__ == '__main__':
    unittest.main()
