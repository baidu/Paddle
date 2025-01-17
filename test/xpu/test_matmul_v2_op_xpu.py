#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from get_test_cover_info import (
    XPUOpTestWrapper,
    check_run_big_shape_test,
    create_test_class,
    get_xpu_op_support_types,
)
from op_test import (
    convert_float_to_uint16,
    skip_check_grad_ci,
)
from op_test_xpu import XPUOpTest

import paddle


def reference_matmul(X, Y, transpose_X=False, transpose_Y=False):
    """Reference forward implementation using np.matmul."""
    # np.matmul does not support the transpose flags, so we manually
    # transpose X and Y appropriately.
    if transpose_X:
        if X.ndim == 1:
            X = X.reshape((X.size,))
        elif X.ndim == 2:
            X = X.T
        else:
            dim = list(range(len(X.shape)))
            dim[-1], dim[len(X.shape) - 2] = dim[len(X.shape) - 2], dim[-1]
            X = np.transpose(X, tuple(dim))
    if transpose_Y:
        if Y.ndim == 1:
            Y = Y.reshape((Y.size,))
        else:
            dim = list(range(len(Y.shape)))
            dim[-1], dim[len(Y.shape) - 2] = dim[len(Y.shape) - 2], dim[-1]
            Y = np.transpose(Y, tuple(dim))
    Out = np.matmul(X, Y)
    return Out


class XPUTestMatmulV2Op(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = "matmul_v2"
        self.use_dynamic_create_class = False

    class TestMatMulV2Op(XPUOpTest):
        """
        case 1
        """

        def config(self):
            self.x_shape = (100,)
            self.y_shape = (100,)
            self.trans_x = False
            self.trans_y = False

        def setUp(self):
            self.dtype = self.in_type
            self.config()
            self.op_type = "matmul_v2"
            import os

            os.environ["XPU_PADDLE_L3_SIZE"] = str(13 * 1024 * 1024)
            x = np.random.random(self.x_shape)
            y = np.random.random(self.y_shape)

            # -0.1 ~ 0.1
            x = -0.1 + 0.2 * x
            y = -0.1 + 0.2 * y
            result = reference_matmul(x, y, self.trans_x, self.trans_y)
            if self.dtype == np.uint16:
                x = convert_float_to_uint16(x)
                y = convert_float_to_uint16(y)
            else:
                x = x.astype(self.dtype)
                y = y.astype(self.dtype)

            result = result.astype(self.dtype)

            self.inputs = {
                'X': x,
                'Y': y,
            }
            self.attrs = {'trans_x': self.trans_x, 'trans_y': self.trans_y}
            self.outputs = {'Out': result}

        def test_check_output(self):
            place = paddle.XPUPlace(0)
            self.check_output_with_place(place)

        def test_check_grad(self):
            if (
                hasattr(self.__class__, "no_need_check_grad")
                and self.__class__.no_need_check_grad
            ):
                return
            place = paddle.XPUPlace(0)
            self.check_grad_with_place(place, ['X', 'Y'], 'Out')

    class TestMatMulOp2(TestMatMulV2Op):
        """
        case 2
        """

        def config(self):
            self.x_shape = 100
            self.y_shape = (100, 3)
            self.trans_x = False
            self.trans_y = False

    class TestMatMulOp3(TestMatMulV2Op):
        """
        case 3
        """

        def config(self):
            self.x_shape = (100,)
            self.y_shape = (1, 1, 100, 2)
            self.trans_x = False
            self.trans_y = False

    class TestMatMulOp4(TestMatMulV2Op):
        """
        case 4
        """

        def config(self):
            self.x_shape = (1, 1, 100, 1)
            self.y_shape = (1, 100)
            self.trans_x = False
            self.trans_y = False

    class TestMatMulOp5(TestMatMulV2Op):
        """
        case 5
        """

        def config(self):
            self.x_shape = (1, 1, 100, 1)
            self.y_shape = (100,)
            self.trans_x = True
            self.trans_y = False

    class TestMatMulOp6(TestMatMulV2Op):
        """
        case 6
        """

        def config(self):
            self.x_shape = (1, 2, 102, 10)
            self.y_shape = (2, 10, 111)
            self.trans_x = False
            self.trans_y = False

    class TestMatMulOp7(TestMatMulV2Op):
        """
        case 7
        """

        def config(self):
            self.x_shape = (1, 2, 100, 1)
            self.y_shape = (2, 100, 12)
            self.trans_x = True
            self.trans_y = False

    class TestMatMulOp8(TestMatMulV2Op):
        """
        case 8
        """

        def config(self):
            self.x_shape = (1, 1, 2, 100)
            self.y_shape = (1, 1, 100, 2)
            self.trans_x = False
            self.trans_y = False

    class TestMatMulOp9(TestMatMulV2Op):
        """
        case 9
        """

        def config(self):
            self.x_shape = (5, 20, 7)
            self.y_shape = (5, 7, 7)
            self.trans_x = False
            self.trans_y = True

    class TestMatMulOp10(TestMatMulV2Op):
        """
        case 10
        """

        def config(self):
            self.x_shape = (3, 20, 8)
            self.y_shape = (3, 20, 8)
            self.trans_x = True
            self.trans_y = False

    class TestMatMulOp11(TestMatMulV2Op):
        """
        case 11
        """

        def config(self):
            self.x_shape = (2, 20, 11)
            self.y_shape = (11, 30)
            self.trans_x = False
            self.trans_y = False

    class TestMatMulOp12(TestMatMulV2Op):
        """
        case 12
        """

        def config(self):
            self.x_shape = (1, 20, 100)
            self.y_shape = (100,)
            self.trans_x = False
            self.trans_y = False

    class TestMatMulOp13(TestMatMulV2Op):
        """
        case 13
        """

        def config(self):
            self.x_shape = (2, 2, 10, 10)
            self.y_shape = (2, 2, 10, 10)
            self.trans_x = True
            self.trans_y = False

    class TestMatMulOp14(TestMatMulV2Op):
        """
        case 14_1
        """

        def config(self):
            self.x_shape = (7, 2, 100, 10)
            self.y_shape = (7, 2, 10, 90)
            self.trans_x = False
            self.trans_y = False

    class TestMatMulOp15(TestMatMulV2Op):
        """
        case 14_2
        """

        def config(self):
            self.x_shape = (3, 2, 4, 10)
            self.y_shape = (3, 2, 4, 10)
            self.trans_x = False
            self.trans_y = True

    class TestMatMulOp17(TestMatMulV2Op):
        """
        case 17 : to check the gradient for special case
        """

        def config(self):
            self.x_shape = (2, 1, 100)
            self.y_shape = 100
            self.trans_x = False
            self.trans_y = False

    @skip_check_grad_ci(
        reason="[skip shape check] Use y_shape(17) to test case in ppyoloe."
    )
    class TestMatMulOp18(TestMatMulV2Op):
        """
        case 18 : for ppyoloe model
        """

        def config(self):
            self.x_shape = (8, 11, 4, 17)
            self.y_shape = 17
            self.trans_x = False
            self.trans_y = False

    class TestMatMulOp19(TestMatMulV2Op):
        """
        case 19 : (x.ndim <= 2) && (y.ndim >= 3),
                  x need to broadcast and trans_y is false
        """

        def config(self):
            self.x_shape = (10, 20)
            self.y_shape = (2, 20, 4)
            self.trans_x = False
            self.trans_y = False

    class TestMatMulOp20(TestMatMulV2Op):
        """
        case 20 : (x.ndim <= 2) && (y.ndim >= 3),
                  x need to broadcast and trans_y is false
        """

        def config(self):
            self.x_shape = (20, 10)
            self.y_shape = (2, 20, 4)
            self.trans_x = True
            self.trans_y = False

    class TestMatMulOp21(TestMatMulV2Op):
        """
        case 21 : (x.ndim >= 3) && (y.ndim <= 2),
                  trans_x is true
        """

        def config(self):
            self.x_shape = (10, 100, 4)
            self.y_shape = (100, 10)
            self.trans_x = True
            self.trans_y = False

    class TestMatMulOp22(TestMatMulV2Op):
        """
        case 22 : (x.ndim <= 2) && (y.ndim >= 3)
        """

        def config(self):
            self.x_shape = (10, 100)
            self.y_shape = (5, 100, 4)
            self.trans_x = False
            self.trans_y = False

    class TestMatMulOp23(TestMatMulV2Op):
        """
        case 23 : (x.ndim <= 2) && (y.ndim >= 3),
                  trans_y is True
        """

        def config(self):
            self.x_shape = (10, 100)
            self.y_shape = (5, 4, 100)
            self.trans_x = False
            self.trans_y = True

    @check_run_big_shape_test()
    class TestMatMulOpLargeShape1(TestMatMulV2Op):
        """
        Large Shape for EB
        """

        def config(self):
            self.x_shape = (8192, 5120)
            self.y_shape = (5120, 1920)
            self.trans_x = False
            self.trans_y = False

    @check_run_big_shape_test()
    class TestMatMulOpLargeShape2(TestMatMulV2Op):
        def config(self):
            self.x_shape = (1024, 5120)
            self.y_shape = (5120, 32)
            self.trans_x = False
            self.trans_y = False

    @check_run_big_shape_test()
    class TestMatMulOpLargeShape3(TestMatMulV2Op):
        def config(self):
            self.x_shape = (8192, 32)
            self.y_shape = (32, 1920)
            self.trans_x = False
            self.trans_y = False

    @check_run_big_shape_test()
    class TestMatMulOpLargeShape4(TestMatMulV2Op):
        def config(self):
            self.x_shape = (8192, 640)
            self.y_shape = (640, 5120)
            self.trans_x = False
            self.trans_y = False

    @check_run_big_shape_test()
    class TestMatMulOpLargeShape5(TestMatMulV2Op):
        def config(self):
            self.x_shape = (640, 32)
            self.y_shape = (1024, 32)
            self.trans_x = False
            self.trans_y = True

    @check_run_big_shape_test()
    class TestMatMulOpLlama13B1(TestMatMulV2Op):
        def config(self):
            self.x_shape = (512, 5120)
            self.y_shape = (5120, 5120)
            self.trans_x = False
            self.trans_y = False

    @check_run_big_shape_test()
    class TestMatMulOpLlama13B2(TestMatMulV2Op):
        def config(self):
            self.x_shape = (512, 5120)
            self.y_shape = (5120, 13824)
            self.trans_x = False
            self.trans_y = False


support_types = get_xpu_op_support_types('matmul_v2')
for stype in support_types:
    create_test_class(globals(), XPUTestMatmulV2Op, stype)

if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
