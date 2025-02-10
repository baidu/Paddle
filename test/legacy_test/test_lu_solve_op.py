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

import copy
import itertools
import os
import unittest

import numpy as np
import scipy
import scipy.linalg
from op_test import OpTest

import paddle
from paddle import base
from paddle.base import core

class TestLUAPIError(unittest.TestCase):
    def test_errors(self):
        with paddle.base.dygraph.guard():
            # The size of b shoule gather than 2.
            def test_b_size():
                b = paddle.randn([3])
                lu = paddle.randn([3, 3])
                pivots = paddle.randn([3])
                paddle.linalg.lu_solve(b, lu, pivots)

            self.assertRaises(ValueError, test_b_size)

            # The size of lu shoule gather than 2.
            def test_lu_size():
                b = paddle.randn([3, 1])
                lu = paddle.randn([3])
                pivots = paddle.randn([3])
                paddle.linalg.lu_solve(b, lu, pivots)

            self.assertRaises(ValueError, test_lu_size)

            # The size of pivots shoule gather than 1.
            def test_pivots_size():
                b = paddle.randn([3, 1])
                lu = paddle.randn([3, 3])
                pivots = paddle.randn([])
                paddle.linalg.lu_solve(b, lu, pivots)

            self.assertRaises(ValueError, test_pivots_size)

            # b.shape[-2] shoule equal to lu.shape[-2].
            def test_b_lu_shape():
                b = paddle.randn([1, 3])
                lu = paddle.randn([3, 3])
                pivots = paddle.randn([3])
                paddle.linalg.lu_solve(b, lu, pivots)

            self.assertRaises(ValueError, test_b_lu_shape)

            # lu.shape[-1] shoule equal to pivots.shape[-1].
            def test_b_pivots_shape():
                b = paddle.randn([3, 1])
                lu = paddle.randn([3, 3])
                pivots = paddle.randn([2])
                paddle.linalg.lu_solve(b, lu, pivots)

            self.assertRaises(ValueError, test_b_pivots_shape)

            # lu.shape[-2] shoule equal to lu.shape[-1].
            def test_lu_shape():
                b = paddle.randn([3, 1])
                lu = paddle.randn([3, 2])
                pivots = paddle.randn([3])
                paddle.linalg.lu_solve(b, lu, pivots)

            self.assertRaises(ValueError, test_lu_shape)



if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
