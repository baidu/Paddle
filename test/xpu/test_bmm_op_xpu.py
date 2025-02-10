#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# # Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at #
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import numpy as np
from get_test_cover_info import (
    XPUOpTestWrapper,
    create_test_class,
    get_xpu_op_support_types,
)
from op_test_xpu import XPUOpTest

import paddle

paddle.enable_static()


class XPUTestBmmOp(XPUOpTestWrapper):
    """
    func desc:: https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/bmm_cn.html#bmm
    """

    def __init__(self):
        self.op_name = 'bmm'
        self.use_dynamic_create_class = False

    class TestBmmOp(XPUOpTest):
        def setUp(self):
            self.init_dtype()
            self.set_xpu()
            self.op_type = "bmm"
            self.place = paddle.XPUPlace(0)
            self.set_shape()
            X = np.random.random(self.Xshape).astype(self.dtype)
            Y = np.random.random(self.Yshape).astype(self.dtype)
            self.inputs = {'X': X, 'Y': Y}

            Out = np.matmul(X, Y)
            self.outputs = {'Out': Out}

        def init_dtype(self):
            self.dtype = self.in_type

        def set_shape(self):
            self.Xshape = (10, 3, 4)
            self.Yshape = (10, 4, 5)

        def set_xpu(self):
            self.__class__.use_xpu = True
            self.__class__.no_need_check_grad = False
            self.__class__.op_type = self.in_type

        def test_check_output(self):
            self.check_output_with_place(self.place, atol=5e-3)

        def test_check_grad_normal(self):
            self.check_grad_with_place(self.place, ['X', 'Y'], 'Out')

    class TestBmmOp1(TestBmmOp):
        def set_shape(self):
            self.Xshape = (3, 3, 3)
            self.Yshape = (3, 3, 3)

    class TestBmmOp2(TestBmmOp):
        def set_shape(self):
            self.Xshape = (128, 3, 16)
            self.Yshape = (128, 16, 3)

    class TestBmmOp3(TestBmmOp):
        def set_shape(self):
            self.Xshape = (2048, 16, 27)
            self.Yshape = (2048, 27, 16)

    class TestBmmOp4(TestBmmOp):
        def set_shape(self):
            self.Xshape = (2, 27, 27)
            self.Yshape = (2, 27, 27)

    class TestBmmOp5(TestBmmOp):
        def set_shape(self):
            self.Xshape = (2, 1, 1)
            self.Yshape = (2, 1, 1)


support_types = get_xpu_op_support_types('bmm')
for stype in support_types:
    create_test_class(globals(), XPUTestBmmOp, stype)

if __name__ == '__main__':
    unittest.main()
