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
import tempfile
import unittest

import numpy as np
from dygraph_to_static_utils import (
    Dy2StTestBase,
    test_pir_only,
)

import paddle


class GradLayer(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x.stop_gradient = False
        y = x * x
        dx = paddle.grad(outputs=[y], inputs=[x])[0]
        return dx


class GradLinearLayer(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.linear = paddle.nn.Linear(5, 5, bias_attr=False)

    def forward(self, x):
        x.stop_gradient = False
        tmp = x + x
        for i in range(10):
            tmp = self.linear(tmp)
        out = tmp
        dx = paddle.grad(
            [out], [x], None, create_graph=True, allow_unused=False
        )[0]
        return dx


class NoGradLinearLayer(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.linear = paddle.nn.Linear(5, 5, bias_attr=False)

    def forward(self, x):
        x.stop_gradient = False

        with paddle.no_grad():
            y = self.linear(x)

        out = y + x
        return out


class TestGrad(Dy2StTestBase):
    def setUp(self):
        self.func = GradLayer()
        self.x = paddle.ones(shape=[10, 2, 5], dtype='float32')
        self.x.stop_gradient = False

    def test_forward(self):
        dygraph_res = self.func(self.x).numpy()
        static_res = paddle.jit.to_static(self.func)(self.x).numpy()
        np.testing.assert_allclose(static_res, dygraph_res, rtol=1e-05)


class TestGradLinear(TestGrad):
    def setUp(self):
        self.func = GradLinearLayer()
        self.x = paddle.ones(shape=[10, 2, 5], dtype='float32')
        self.x.stop_gradient = False

        self.temp_dir = tempfile.TemporaryDirectory()
        self.infer_model_path = os.path.join(
            self.temp_dir.name, 'double_grad_infer_model'
        )
        self.train_model_path = os.path.join(
            self.temp_dir.name, 'double_grad_train_model'
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_save_infer_program(self):
        static_fn = paddle.jit.to_static(self.func)
        input_spec = [
            paddle.static.InputSpec(shape=[10, 2, 5], dtype='float32')
        ]
        paddle.jit.save(static_fn, self.infer_model_path, input_spec=input_spec)
        load_func = paddle.jit.load(self.infer_model_path)

        origin_res = static_fn(self.x).numpy()
        load_res = load_func(self.x).numpy()
        np.testing.assert_allclose(origin_res, load_res, rtol=1e-05)

    def test_save_train_program(self):
        static_fn = paddle.jit.to_static(self.func)
        grad_clip = paddle.nn.ClipGradByGlobalNorm(2.0)
        optimizer = paddle.optimizer.SGD(
            learning_rate=0.01,
            grad_clip=grad_clip,
            parameters=static_fn.parameters(),
        )
        for i in range(10):
            out = static_fn(self.x)
            avg_loss = paddle.mean(paddle.abs(out - 1))
            avg_loss.backward()
            optimizer.minimize(avg_loss)

            static_fn.clear_gradients()

        paddle.jit.save(static_fn, self.train_model_path)
        load_func = paddle.jit.load(self.train_model_path)

        origin_res = static_fn(self.x).numpy()
        load_res = load_func(self.x).numpy()
        np.testing.assert_allclose(origin_res, load_res, rtol=1e-05)


class TestNoGradLinear(TestGradLinear):
    def setUp(self):
        self.func = NoGradLinearLayer()
        self.x = paddle.ones(shape=[10, 2, 5], dtype='float32')
        self.x.stop_gradient = False

        self.temp_dir = tempfile.TemporaryDirectory()
        self.infer_model_path = os.path.join(
            self.temp_dir.name, 'no_grad_infer_model'
        )
        self.train_model_path = os.path.join(
            self.temp_dir.name, 'no_grad_train_model'
        )

    def tearDown(self):
        self.temp_dir.cleanup()


class UnuseGradVarLayer(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, var_0, var_1):
        var_1 = var_1 + 1
        return var_0, var_1


class TestUnuseGradVar(Dy2StTestBase):
    @test_pir_only
    def test_run(self):
        layer = UnuseGradVarLayer()
        layer = paddle.jit.to_static(layer)

        x = paddle.to_tensor([1.0])
        y = paddle.to_tensor([2.0])
        x.stop_gradient = False
        y.stop_gradient = False

        out1, out2 = layer(x, y)
        out = out1 + out2
        out.backward()
        np.testing.assert_array_equal(out.numpy(), [4])
        np.testing.assert_array_equal(x.grad.numpy(), [1])


class NoGradNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.linear = paddle.nn.Linear(3, 4)

    def forward(self, x):
        with paddle.no_grad():
            out = self.linear(x)
        return out


class TestNoGrad(Dy2StTestBase):
    @test_pir_only
    def test_run(self):
        net = NoGradNet()
        net = paddle.jit.to_static(net)
        x = paddle.rand([2, 3], 'float32')
        x.stop_gradient = False
        out = net(x)
        np.testing.assert_array_equal(out.stop_gradient, True)


def grad_with_if_case(x):
    y = paddle.tanh(x)
    if x.numel() > 0:
        t = paddle.grad([y], [x])[0]
        return t
    return paddle.ones_like(x, dtype='float32')


class TestGradWithIf(Dy2StTestBase):
    @test_pir_only
    def test_grad_with_if(self):
        fn = grad_with_if_case
        static_fn = paddle.jit.to_static(fn)
        x = paddle.randn([2, 2])
        x.stop_gradient = False
        dx = fn(x)
        dx_st = static_fn(x)
        np.testing.assert_allclose(dx, dx_st)


if __name__ == '__main__':
    unittest.main()
