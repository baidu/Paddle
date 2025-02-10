# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright (c) 2022 NVIDIA Corporation. All rights reserved.
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
    create_test_class,
    get_xpu_op_support_types,
)
from op_test_xpu import XPUOpTest

import paddle
from paddle import _C_ops
from paddle.base import core


def gelu(x):
    y_ref = (
        0.5
        * x
        * (1.0 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))
    )
    return y_ref.astype(x.dtype)


def relu(x):
    mask = x > 0
    return x * mask


def get_output(X, Y, bias, act):
    out = np.dot(X, Y) + bias
    if act == 'relu':
        return relu(out)
    elif act == 'gelu':
        return gelu(out)
    else:
        return out


def matmul(x, y, bias, trans_x, trans_y):
    x = np.array(x)
    if trans_x:
        x = np.ascontiguousarray(np.transpose(x))
    if trans_y:
        y = np.ascontiguousarray(np.transpose(y))
    z = np.matmul(x, y)
    if bias is None:
        return z
    else:
        return z + bias


def matmul_grad(x, y, bias, dz, trans_x, trans_y):
    if trans_x:
        if trans_y:
            dx = matmul(y, dz, None, True, True)
            dy = matmul(dz, x, None, True, True)
        else:
            dx = matmul(y, dz, None, False, True)
            dy = matmul(x, dz, None, False, False)
    else:
        if trans_y:
            dx = matmul(dz, y, None, False, False)
            dy = matmul(dz, x, None, True, False)
        else:
            dx = matmul(dz, y, None, False, True)
            dy = matmul(x, dz, None, True, False)
    if bias is None:
        dbias = None
    else:
        dbias = np.sum(dz, axis=0, keepdims=False)
    return dx, dy, dbias


class XPUTestFuseGemmOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'fused_gemm_epilogue'
        self.use_dynamic_create_class = False

    class TestFuseGemmBase(XPUOpTest):
        def setUp(self):
            self.__class__.no_need_check_grad = True
            self.op_type = "fused_gemm_epilogue"
            self.init_dtype_type()
            self.init_datas_shape_and_attrs()
            self.inputs = {
                'X': np.random.random(self.x_shape).astype(self.dtype) - 0.5,
                'Y': np.random.random(self.y_shape).astype(self.dtype) - 0.5,
                'Bias': np.random.random(self.bias_shape).astype(self.dtype)
                - 0.5,
            }

            if self.trans_x:
                numpy_input_x = (
                    self.inputs['X'].reshape((self.x_shape[0], -1)).T
                )
            else:
                numpy_input_x = self.inputs['X'].reshape((-1, self.x_shape[-1]))

            if self.trans_y:
                numpy_input_y = self.inputs['Y'].T
            else:
                numpy_input_y = self.inputs['Y']

            self.outputs = {
                'Out': self.cal_output(
                    numpy_input_x,
                    numpy_input_y,
                    self.inputs['Bias'],
                    self.activation,
                ).reshape(self.out_shape)
            }
            self.attrs = {
                "activation": self.activation,
                "trans_y": self.trans_y,
                "trans_x": self.trans_x,
            }

        def cal_output(self, X, Y, bias, act):
            return get_output(X, Y, bias, act)

        def init_dtype_type(self):
            self.dtype = self.in_type
            self.atol = 1e-2
            if self.dtype == np.float16:
                self.atol = 1e-3

        def init_datas_shape_and_attrs(self):
            self.x_shape = [8, 4]
            self.y_shape = [4, 128]
            self.bias_shape = [
                128,
            ]
            self.out_shape = [8, 128]
            self.activation = "relu"
            self.trans_y = False
            self.trans_x = False

        def test_check_output(self):
            self.check_output_with_place(core.XPUPlace(0), atol=self.atol)

    class TestFuseGemmEpilogueOp1(TestFuseGemmBase):
        def init_datas_shape_and_attrs(self):
            self.x_shape = [4, 8]
            self.y_shape = [4, 128]
            self.bias_shape = [
                128,
            ]
            self.out_shape = [8, 128]
            self.activation = "relu"
            self.trans_y = False
            self.trans_x = True

    class TestFuseGemmEpilogueOp2(TestFuseGemmBase):
        def init_datas_shape_and_attrs(self):
            self.x_shape = [8, 4]
            self.y_shape = [128, 4]
            self.bias_shape = [
                128,
            ]
            self.out_shape = [8, 128]
            self.activation = "relu"
            self.trans_y = True
            self.trans_x = False

    class TestFuseGemmEpilogueOp3(TestFuseGemmBase):
        def init_datas_shape_and_attrs(self):
            self.x_shape = [4, 8]
            self.y_shape = [128, 4]
            self.bias_shape = [
                128,
            ]
            self.out_shape = [8, 128]
            self.activation = "relu"
            self.trans_y = True
            self.trans_x = True

    class TestFuseGemmEpilogueOp4(TestFuseGemmBase):
        def init_datas_shape_and_attrs(self):
            self.x_shape = [2, 2, 8, 4]
            self.y_shape = [4, 128]
            self.bias_shape = [
                128,
            ]
            self.out_shape = [2, 2, 8, 128]
            self.activation = "relu"
            self.trans_y = False
            self.trans_x = False

    # class TestFuseGemmEpilogueOp5(TestFuseGemmBase):
    #     def init_datas_shape_and_attrs(self):
    #         self.x_shape = [2, 2, 4, 8]
    #         self.y_shape = [4, 128]
    #         self.bias_shape = [
    #             128,
    #         ]
    #         self.out_shape = [2, 2, 8, 128]
    #         self.activation = "relu"
    #         self.trans_y = False
    #         self.trans_x = True

    #     def cal_output(self, X, Y, bias, act):
    #         out = (
    #             np.dot(np.transpose(self.inputs['X'], axes=(0, 1, 3, 2)), Y)
    #             + bias
    #         )

    #         return relu(out)

    class TestFuseGemmEpilogueOp6(TestFuseGemmBase):
        def init_datas_shape_and_attrs(self):
            self.x_shape = [8, 4]
            self.y_shape = [4, 128]
            self.bias_shape = [
                128,
            ]
            self.out_shape = [8, 128]
            self.activation = "gelu"
            self.trans_y = False
            self.trans_x = False

    class TestFuseGemmEpilogueOp7(TestFuseGemmBase):
        def init_datas_shape_and_attrs(self):
            self.x_shape = [8, 4]
            self.y_shape = [4, 128]
            self.bias_shape = [
                128,
            ]
            self.out_shape = [8, 128]
            self.activation = "none"
            self.trans_y = False
            self.trans_x = False


class TestEagerFusedGemmEpilogue(unittest.TestCase):
    def setUp(self):
        paddle.set_device('xpu')

    def test_case_act(self):
        paddle.disable_static()
        x_np = np.random.random((8, 4)).astype(np.float32) - 0.5
        y_np = np.random.random((4, 128)).astype(np.float32) - 0.5
        bias_np = np.random.random((128,)).astype(np.float32) - 0.5
        x = paddle.to_tensor(x_np)
        y = paddle.to_tensor(y_np)
        bias = paddle.to_tensor(bias_np)
        x.stop_gradient = False
        y.stop_gradient = False

        out1, _ = _C_ops.fused_gemm_epilogue(x, y, bias, False, False, 'none')
        out2, _ = _C_ops.fused_gemm_epilogue(x, y, bias, False, False, 'relu')
        out3, _ = _C_ops.fused_gemm_epilogue(x, y, bias, False, False, 'gelu')

        out_np1 = get_output(x_np, y_np, bias_np, 'none')
        out_np2 = get_output(x_np, y_np, bias_np, 'relu')
        out_np3 = get_output(x_np, y_np, bias_np, 'gelu')

        np.testing.assert_allclose(out1, out_np1, atol=1e-04, rtol=1e-2)
        np.testing.assert_allclose(out2, out_np2, atol=1e-04, rtol=1e-2)
        np.testing.assert_allclose(out3, out_np3, atol=1e-03, rtol=1e-2)

        out_grad_np1 = np.random.randint(
            low=-20, high=20, size=out_np1.shape
        ).astype(np.float32)
        paddle.autograd.backward(
            out1, grad_tensors=[paddle.to_tensor(out_grad_np1)]
        )

        x_grad_np, y_grad_np, bias_grad_np = matmul_grad(
            x_np, y_np, bias_np, out_grad_np1, False, False
        )
        np.testing.assert_allclose(x.grad.numpy(), x_grad_np, atol=1e-02, rtol=5e-2)
        self.assertEqual(y_grad_np.shape, y_np.shape)
        np.testing.assert_allclose(y.grad.numpy(), y_grad_np, atol=1e-03, rtol=5e-2)

        paddle.enable_static()


support_types = get_xpu_op_support_types('fused_gemm_epilogue')
for stype in support_types:
    create_test_class(globals(), XPUTestFuseGemmOp, stype)

if __name__ == "__main__":
    paddle.enable_static()
    np.random.seed(0)
    unittest.main()
