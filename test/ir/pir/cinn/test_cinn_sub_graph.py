# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import utils

import paddle


def exp_sub(x):
    y = paddle.exp(x)
    z = y - x
    return z


def layer_norm(x, weight, bias):
    num = paddle.full([1], x.shape[-1])
    eps = paddle.full([1], 1e-5)
    sum1 = paddle.sum(x, axis=-1, keepdim=True)
    mean = sum1 / num
    t1 = x - mean
    t2 = t1 * t1
    t3 = paddle.sum(t2, axis=-1, keepdim=True)
    t3 = t3 / num
    t4 = t3 + eps
    t5 = paddle.sqrt(t4)
    t7 = t1 / t5

    return t7 * weight + bias


def dropout(x):
    rand = paddle.uniform(x.shape, min=0.0, max=1.0, dtype="float32")
    zero = paddle.full([1], 0.0)

    mask = paddle.greater_equal(rand, zero)

    out = x * paddle.cast(mask, x.dtype)
    return out


class CINNSubGraphNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.fn = exp_sub

    def forward(self, x):
        out = self.fn(x)
        return out


class CINNSoftmaxSubGraphNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.fn = paddle.nn.functional.softmax

    def forward(self, x, axis=-1):
        out = self.fn(x, axis=axis)
        return out


class CINNSliceSubGraphNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.fn = paddle.nn.functional.softmax

    def forward(self, x, d1, d2, d3, d4):
        t1 = x[:, d1 * d2 : d1 * d2 + d3 * d4]
        out = t1.reshape([t1.shape[0], d3, d4])

        return out


class CINNAddSubGraphNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        t = x.sum(axis=[-1], keepdim=True)
        return x + t


class CINNLayerNormSubGraphNet(paddle.nn.Layer):
    def __init__(self, hidden_size):
        super().__init__()
        self.fn = layer_norm
        self.weight = self.create_parameter(
            shape=[hidden_size], dtype="float64"
        )
        self.bias = self.create_parameter(shape=[hidden_size], dtype="float64")

    def forward(self, x, weight, bias):
        out = paddle.nn.functional.layer_norm(x, x.shape[-1], weight, bias)
        return out


class CINNAddDropoutLayerNormSubGraphNet(paddle.nn.Layer):
    def __init__(self, hidden_size):
        super().__init__()
        self.add = paddle.add
        self.dropout = dropout
        self.layer_norm = paddle.nn.functional.layer_norm

        self.weight = self.create_parameter(
            shape=[hidden_size], dtype="float64"
        )
        self.bias = self.create_parameter(shape=[hidden_size], dtype="float64")

    def forward(self, x, y, weight, bias):
        t1 = self.add(x, y)
        t2 = self.dropout(t1)
        t2 = x
        out = self.layer_norm(t2, t2.shape[-1], self.weight, self.bias)
        return out

        out = paddle.nn.functional.layer_norm(
            x, x.shape[-1], self.weight, self.bias
        )
        return out


class CINNDropoutSubGraphNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.fn = paddle.nn.functional.dropout

    def forward(self, x):
        out = self.fn(x)
        return out


class TestCinnSubGraphBase(unittest.TestCase):
    """
    Test Pir API + @to_static + CINN.
    """

    def setUp(self):
        paddle.seed(2022)
        self.prepare_data()

    def prepare_data(self):
        self.shape = [128, 128, 768]
        self.axis = -1
        self.x = paddle.uniform(self.shape, dtype="float64", min=-0.5, max=0.5)
        self.x.stop_gradient = False

    def check_jit_kernel_info(self, static_fn):
        utils.check_jit_kernel_number(static_fn, 1)
        utils.check_jit_kernel_structure(static_fn, {utils.JIT_KERNEL_NAME: 1})


class TestCinnExpSubNet(TestCinnSubGraphBase):
    def eval(self, use_cinn):
        paddle.seed(2022)
        net = CINNSubGraphNet()
        net = utils.apply_to_static(net, use_cinn)
        net.eval()
        out = net(self.x)
        if use_cinn:
            self.check_jit_kernel_info(net.forward)
        return out

    def test_eval(self):
        cinn_out = self.eval(use_cinn=True)
        dy_out = self.eval(use_cinn=False)
        np.testing.assert_allclose(cinn_out.numpy(), dy_out.numpy(), atol=1e-8)


class TestCinnSoftmax(TestCinnSubGraphBase):
    def train(self, use_cinn):
        paddle.seed(2022)
        net = CINNSoftmaxSubGraphNet()
        net = utils.apply_to_static(net, use_cinn)
        out = net(self.x, self.axis)

        loss = out.sum()
        loss.backward()
        return out, self.x.gradient()

    def test_forward(self):
        cinn_out, cinn_grad = self.train(use_cinn=True)
        dy_out, dy_grad = self.train(use_cinn=False)
        np.testing.assert_allclose(cinn_out.numpy(), dy_out.numpy(), atol=1e-8)
        np.testing.assert_allclose(cinn_grad, dy_grad, atol=1e-8)


class TestCinnSlice(TestCinnSubGraphBase):
    def train(self, use_cinn):
        paddle.seed(2022)
        net = CINNSliceSubGraphNet()

        input_spec = [
            paddle.static.InputSpec(
                shape=[-1, -1], dtype='float32', name='in_x'
            ),
            paddle.static.InputSpec(shape=[1], dtype='int64', name='d1'),
            paddle.static.InputSpec(shape=[1], dtype='int64', name='d2'),
            paddle.static.InputSpec(shape=[1], dtype='int64', name='d3'),
            paddle.static.InputSpec(shape=[1], dtype='int64', name='42'),
        ]

        self.x = paddle.uniform([16, 256], dtype="float64", min=-0.5, max=0.5)
        self.d1 = paddle.full([1], fill_value=4, dtype="int64")
        self.d2 = paddle.full([1], fill_value=16, dtype="int64")
        self.d3 = paddle.full([1], fill_value=4, dtype="int64")
        self.d4 = paddle.full([1], fill_value=4, dtype="int64")

        net = utils.apply_to_static(net, use_cinn, input_spec=input_spec)
        out = net(self.x, self.d1, self.d2, self.d3, self.d4)

        return out

    def test_forward(self):
        cinn_out = self.train(use_cinn=True)
        dy_out = self.train(use_cinn=False)
        np.testing.assert_allclose(cinn_out.numpy(), dy_out.numpy(), atol=1e-8)


class TestCinnSmallSoftmax(TestCinnSoftmax):
    def prepare_data(self):
        self.shape = [1, 1, 17, 17]
        self.axis = -1
        self.x = paddle.uniform(self.shape, dtype="float64", min=-0.5, max=0.5)
        self.x.stop_gradient = False


class TestReduceAs(TestCinnSubGraphBase):
    def train(self, use_cinn):
        paddle.seed(2022)
        net = CINNAddSubGraphNet()

        input_spec = [
            paddle.static.InputSpec(shape=[-1, -1], dtype='float32', name='x')
        ]

        self.x = paddle.uniform([16, 256], dtype="float64", min=-0.5, max=0.5)
        self.x.stop_gradient = False

        net = utils.apply_to_static(net, use_cinn, input_spec=input_spec)
        out = net(self.x)

        loss = out.sum()
        loss.backward()

        return out

    def test_forward(self):
        cinn_out = self.train(use_cinn=True)
        dy_out = self.train(use_cinn=False)
        np.testing.assert_allclose(cinn_out.numpy(), dy_out.numpy(), atol=1e-8)


class TestLayerNormMultiNorm(TestCinnSubGraphBase):
    def train(self, use_cinn):
        paddle.seed(2022)
        net = paddle.nn.LayerNorm([32, 32])

        input_spec = [
            paddle.static.InputSpec(
                shape=[16, 16, 32, 32], dtype='float64', name='x'
            )
        ]

        self.x = paddle.uniform(
            [16, 16, 32, 32], dtype="float64", min=-0.5, max=0.5
        )
        self.x.stop_gradient = False

        net = utils.apply_to_static(net, use_cinn, input_spec=input_spec)
        out = net(self.x)

        loss = out.sum()
        loss.backward()

        return out

    def test_forward(self):
        cinn_out = self.train(use_cinn=True)
        dy_out = self.train(use_cinn=False)
        np.testing.assert_allclose(cinn_out.numpy(), dy_out.numpy(), atol=1e-8)


# class TestCinnLayerNorm(TestCinnSubGraphBase):
#     def train(self, use_cinn):
#         paddle.seed(2022)
#         self.prepare_data()
#         net = CINNLayerNormSubGraphNet(self.shape[-1])
#         net = utils.apply_to_static(net, use_cinn)
#         # net.eval()
#         weight = paddle.ones(shape=[self.shape[-1]], dtype="float64")
#         weight.stop_gradient = False
#         bias = paddle.ones(shape=[self.shape[-1]], dtype="float64")
#         bias.stop_gradient = False
#         self.x.stop_gradient = False
#         out = net(self.x, weight, bias)
#         loss = out.sum()
#         loss.backward()

#         return out, self.x.gradient(), weight.gradient(), bias.gradient()

#     def test_train(self):
#         cinn_out, cinn_x_grad, cinn_w_grad, cinn_b_grad = self.train(
#             use_cinn=True
#         )

#         dy_out, dy_x_grad, dy_w_grad, dy_b_grad = self.train(use_cinn=False)
#         np.testing.assert_allclose(cinn_out.numpy(), dy_out.numpy(), atol=1e-8)
#         np.testing.assert_allclose(cinn_x_grad, dy_x_grad, atol=1e-8)
#         np.testing.assert_allclose(cinn_w_grad, dy_w_grad, atol=1e-8)
#         np.testing.assert_allclose(cinn_b_grad, dy_b_grad, atol=1e-8)


# class TestAddDropoutLayerNorm(TestCinnSubGraphBase):
#     def train(self, use_cinn):
#         paddle.seed(2022)
#         net = CINNAddDropoutLayerNormSubGraphNet(self.shape[-1])
#         net = utils.apply_to_static(net, use_cinn)
#         # net.eval()
#         weight = paddle.ones(shape=[self.shape[-1]], dtype="float32")
#         bias = paddle.ones(shape=[self.shape[-1]], dtype="float32")
#         out = net(self.x, self.x, weight, bias)
#         return out

#     def test_forward(self):
#         cinn_out = self.train(use_cinn=True)
#         dy_out = self.train(use_cinn=False)

#         np.testing.assert_allclose(
#             cinn_out.numpy(), dy_out.numpy(), atol=1e-8, rtol=1e-4
#         )


# class TestCinnDropout(TestCinnSubGraphBase):
#     def train(self, use_cinn):
#         paddle.seed(2022)
#         net = CINNDropoutSubGraphNet()
#         net = utils.apply_to_static(net, use_cinn)
#         out = net(self.x)
# class TestCinnLayerNorm(TestCinnSubGraphBase):
#     def eval(self, use_cinn):
#         paddle.seed(2022)
#         net = CINNLayerNormSubGraphNet(self.shape[-1])
#         net = utils.apply_to_static(net, use_cinn)
#         net.eval()
#         weight = paddle.ones(shape=[self.shape[-1]], dtype="float32")
#         bias = paddle.ones(shape=[self.shape[-1]], dtype="float32")
#         out = net(self.x, weight, bias)
#         return out

#     def test_eval(self):
#         cinn_out = self.eval(use_cinn=True)
#         dy_out = self.eval(use_cinn=False)
#         # TODO(Aurelius84): Apply assert_allclose logic,
#         # but need figure out why atol only satisfy 1e-7
#         np.testing.assert_allclose(cinn_out.numpy(), dy_out.numpy(), atol=1e-7)


# class TestAddDropoutLayerNorm(TestCinnSubGraphBase):
#     def eval(self, use_cinn):
#         paddle.seed(2022)
#         net = CINNAddDropoutLayerNormSubGraphNet(self.shape[-1])
#         net = utils.apply_to_static(net, use_cinn)
#         net.eval()
#         weight = paddle.ones(shape=[self.shape[-1]], dtype="float32")
#         bias = paddle.ones(shape=[self.shape[-1]], dtype="float32")
#         out = net(self.x, self.x, weight, bias)
#         if use_cinn:
#             self.check_jit_kernel_info(net.forward)
#         return out

#     def test_eval(self):
#         cinn_out = self.eval(use_cinn=True)
#         dy_out = self.eval(use_cinn=False)

#         np.testing.assert_allclose(
#             cinn_out.numpy(), dy_out.numpy(), atol=1e-8, rtol=1e-4
#         )


# class TestCinnDropout(TestCinnSubGraphBase):
#     def train(self, use_cinn):
#         paddle.seed(2022)
#         net = CINNDropoutSubGraphNet()
#         net = utils.apply_to_static(net, use_cinn)
#         out = net(self.x)

#         loss = out.mean()
#         loss.backward()
#         if use_cinn:
#             self.check_jit_kernel_info(net.forward)
#         return out

#     def test_forward(self):
#         cinn_out = self.train(use_cinn=True)
#         dy_out = self.train(use_cinn=False)
#         np.testing.assert_allclose(cinn_out.numpy(), dy_out.numpy(), atol=1e-8)


# class TestCinnEvalPrim(TestCinnSubGraphBase):
#     def prepare_data(self):
#         self.shape = [1, 2048, 768]
#         self.hidden_states = paddle.randn(self.shape, dtype="float32")
#         self.hidden_states.stop_gradient = False

# def eval(self, use_cinn):
#     paddle.seed(2022)
#     net = CINNSoftmaxSubGraphNet()
#     net = utils.apply_to_static(net, use_cinn)
#     net.eval()
#     out = net(self.hidden_states)

#     if use_cinn:
#         ops = [
#             op.name()
#             for op in net.forward.program_cache.last()[-1][-1]
#             .train_program.program.global_block()
#             .ops
#         ]
#         assert (
#             "pd_op.softmax" not in ops
#         ), f"after prim, pd_op.softmax should not exist, but got {ops}"
#         assert (
#             "pd_op.exp" in ops
#         ), f"after prim, pd_op.softmax should not exist, but got {ops}"
#         self.check_jit_kernel_info(net.forward)

#         return out

#     def test_eval(self):
#         cinn_out = self.eval(use_cinn=True)
#         dy_out = self.eval(use_cinn=False)
#         np.testing.assert_allclose(
#             cinn_out.numpy(), dy_out.numpy(), atol=1e-6, rtol=1e-6
#         )


if __name__ == '__main__':
    unittest.main()
