#   Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
import scipy.linalg
from op_test import OpTest

import paddle
from paddle import base
from paddle.base import core

class TestLuSolveAPI(unittest.TestCase):
    def test_dygraph(self):
        paddle.disable_static(base.CPUPlace())
        A = np.array([[2, 5, 8, 7], [5, 2, 2, 8], [7, 5, 6, 6], [5, 4, 4, 8]])
        b = np.array([1, 1, 1, 1])
        lu, piv = scipy.linalg.lu_factor(A)
        x = scipy.linalg.lu_solve((lu, piv), b)

        lu_pd = paddle.to_tensor(lu, dtype='float32')
        piv_pd = paddle.to_tensor(piv, dtype='int32')
        b_pd = paddle.to_tensor(b, dtype='float32')
        x_pd = paddle.linalg.lu_solve(b_pd, lu_pd, piv_pd)
        x_np = x_pd.numpy()
        assert np.allclose(x, x_np)

if __name__ == "__main__":
    unittest.main()
