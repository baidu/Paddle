# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import os
import unittest
import paddle
import numpy as np
from paddle.utils.cpp_extension import load
from utils import paddle_includes, extra_compile_args

dispatch_op = load(
    name='dispatch_op',
    sources=['dispatch_test_op.cc'],
    extra_include_paths=paddle_includes,  # add for Coverage CI
    extra_cflags=extra_compile_args)  # add for Coverage CI


class TestJitDispatch(unittest.TestCase):
    def setUp(self):
        paddle.set_device('cpu')

    def test_dispatch_float_and(self):
        dtypes = ["float32", "float64", "float16"]
        for dtype in dtypes:
            x = paddle.ones([2, 2], dtype=dtype)
            out = dispatch_op.dispatch_test_float_and(x)

    def test_dispatch_float_and2(self):
        dtypes = ["float32", "float64", "float16", "bool"]
        for dtype in dtypes:
            x = paddle.ones([2, 2], dtype=dtype)
            out = dispatch_op.dispatch_test_float_and2(x)


if __name__ == '__main__':
    unittest.main()
