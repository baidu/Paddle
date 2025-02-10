#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.tensor import random


class TestManualSeed(unittest.TestCase):
    def test_seed(self):
        base.enable_dygraph()

        gen = paddle.seed(12312321111)
        x = random.gaussian([10], dtype="float32")
        st1 = gen.get_state()
        x1 = random.gaussian([10], dtype="float32")
        gen.set_state(st1)
        x2 = random.gaussian([10], dtype="float32")
        gen.manual_seed(12312321111)
        x3 = random.gaussian([10], dtype="float32")
        x_np = x.numpy()
        x1_np = x1.numpy()
        x2_np = x2.numpy()
        x3_np = x3.numpy()

        if (
            not base.core.is_compiled_with_cuda()
            and not base.core.is_compiled_with_xpu()
        ):
            np.testing.assert_allclose(x1_np, x2_np, rtol=1e-05)
            np.testing.assert_allclose(x_np, x3_np, rtol=1e-05)


if __name__ == '__main__':
    unittest.main()
