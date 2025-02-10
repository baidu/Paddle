#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import copy
import itertools
import os
import unittest

import numpy as np
from op_test import OpTest

import paddle
from paddle import base
from paddle.base import core


class TestSolveOpAPI_1(unittest.TestCase):
    def setUp(self):
        self.A = paddle.to_tensor([[2, 5],[4, 8]], dtype='float32')
        self.b = paddle.to_tensor([[1, 1],[1, 1]], dtype='float32')
        self.LU, self.pivots = paddle.linalg.lu(self.A)
        self.x = paddle.linalg.solve(self.A, self.b)
        self.place = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            self.place.append(base.CPUPlace())
        if core.is_compiled_with_cuda():
            self.place.append(base.CUDAPlace(0))


    def test_dygraph(self):
        def run(place):
            paddle.disable_static(place)
            lu_solve_x = paddle.linalg.lu_solve(self.b, self.LU, self.pivots)
            solve_x = paddle.linalg.solve(self.A, self.b)

            
            np.testing.assert_allclose(
                lu_solve_x.numpy(), solve_x.numpy(), rtol=1e-05
            )
            paddle.enable_static()

        for place in self.place:
            run(place)


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
