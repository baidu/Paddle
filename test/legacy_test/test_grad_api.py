# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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


class dy_to_st(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self._param_attr = base.ParamAttr(
            initializer=paddle.nn.initializer.Constant(value=0.1)
        )
        self.w1 = self.create_parameter(
            attr=self._param_attr, shape=[2, 2], dtype='float32', is_bias=False
        )
        self.b1 = self.create_parameter(
            attr=self._param_attr, shape=[2, 2], dtype='float32', is_bias=False
        )

    @paddle.jit.to_static(full_graph=True)
    def forward(self, x):
        self.x = x
        self.y = paddle.matmul(self.x, self.w1)
        self.z = paddle.add(self.y, self.b1)
        self.k = paddle.tanh(self.z)
        return self.k

    @paddle.jit.to_static(full_graph=True)
    def backward(self, x, k_grad):
        x = x
        y = paddle.matmul(x, self.w1)
        z = paddle.add(y, self.b1)
        k = paddle.tanh(z)
        z_grad = paddle._C_ops.tanh_grad(k, k_grad)
        y_grad, b1_grad = paddle._C_ops.add_grad(y, self.b1, z_grad, -1)
        x_grad, w1_grad = paddle._C_ops.matmul_grad(
            x, self.w1, y_grad, False, False
        )
        return x_grad, z_grad, y_grad, w1_grad, b1_grad


class dygraph(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self._param_attr = base.ParamAttr(
            initializer=paddle.nn.initializer.Constant(value=0.1)
        )
        self.w1 = self.create_parameter(
            attr=self._param_attr, shape=[2, 2], dtype='float32', is_bias=False
        )
        self.b1 = self.create_parameter(
            attr=self._param_attr, shape=[2, 2], dtype='float32', is_bias=False
        )

    def forward(self, x):
        self.x = x
        self.y = paddle.matmul(self.x, self.w1)
        self.z = paddle.add(self.y, self.b1)
        self.k = paddle.tanh(self.z)
        return self.k

    def backward(self, k_grad):
        z_grad = paddle._C_ops.tanh_grad(self.k, k_grad)
        y_grad, b1_grad = paddle._C_ops.add_grad(self.y, self.b1, z_grad, -1)
        x_grad, w1_grad = paddle._C_ops.matmul_grad(
            self.x, self.w1, y_grad, False, False
        )
        return x_grad, z_grad, y_grad, w1_grad, b1_grad


class TestBaseLayer(unittest.TestCase):
    def test_dy_to_st(self):
        layer = dy_to_st()
        x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]], dtype='float32')
        out_grad = paddle.to_tensor([[1.0, 1.0], [1.0, 1.0]], dtype='float32')
        x.stop_gradient = False

        out = layer(x)
        with paddle.no_grad():
            x_grad, z_grad, y_grad, w1_grad, b1_grad = layer.backward(
                x, out_grad
            )
        out.backward(out_grad)
        x_grad_check = x.grad
        w1_grad_check = layer.w1.grad
        b1_grad_check = layer.b1.grad
        assert np.allclose(x_grad.numpy(), x_grad_check.numpy())
        assert np.allclose(w1_grad.numpy(), w1_grad_check.numpy())
        assert np.allclose(b1_grad.numpy(), b1_grad_check.numpy())

    def test_dygraph(self):
        layer = dygraph()
        x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]], dtype='float32')
        out_grad = paddle.to_tensor([[1.0, 1.0], [1.0, 1.0]], dtype='float32')
        x.stop_gradient = False

        out = layer(x)
        x_grad, z_grad, y_grad, w1_grad, b1_grad = layer.backward(out_grad)
        out.backward(out_grad)

        x_grad_check = x.grad
        w1_grad_check = layer.w1.grad
        b1_grad_check = layer.b1.grad
        assert np.allclose(x_grad.numpy(), x_grad_check.numpy())
        assert np.allclose(w1_grad.numpy(), w1_grad_check.numpy())
        assert np.allclose(b1_grad.numpy(), b1_grad_check.numpy())


if __name__ == '__main__':
    unittest.main()
