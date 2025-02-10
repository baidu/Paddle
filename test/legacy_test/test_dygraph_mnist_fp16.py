# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.nn import Linear


class SimpleImgConvPool(paddle.nn.Layer):
    def __init__(
        self,
        num_channels,
        num_filters,
        filter_size,
        pool_size,
        pool_stride,
        pool_padding=0,
        pool_type='max',
        global_pooling=False,
        conv_stride=1,
        conv_padding=0,
        conv_dilation=1,
        conv_groups=1,
        act=None,
        use_cudnn=False,
        dtype='float32',
        param_attr=None,
        bias_attr=None,
    ):
        super().__init__()

        self._conv2d = paddle.nn.Conv2D(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=conv_stride,
            padding=conv_padding,
            dilation=conv_dilation,
            groups=conv_groups,
            weight_attr=param_attr,
            bias_attr=bias_attr,
        )

        self._pool2d = paddle.nn.MaxPool2D(
            kernel_size=pool_size,
            stride=pool_stride,
            padding=pool_padding,
        )

    def forward(self, inputs):
        x = self._conv2d(inputs)
        x = self._pool2d(x)
        return x


class MNIST(paddle.nn.Layer):
    def __init__(self, dtype="float32"):
        super().__init__()

        self._simple_img_conv_pool_1 = SimpleImgConvPool(
            num_channels=3,
            num_filters=20,
            filter_size=5,
            pool_size=2,
            pool_stride=2,
            act="relu",
            dtype=dtype,
            use_cudnn=True,
        )

        self._simple_img_conv_pool_2 = SimpleImgConvPool(
            num_channels=20,
            num_filters=50,
            filter_size=5,
            pool_size=2,
            pool_stride=2,
            act="relu",
            dtype=dtype,
            use_cudnn=True,
        )

        self.pool_2_shape = 50 * 53 * 53
        SIZE = 10
        scale = (2.0 / (self.pool_2_shape**2 * SIZE)) ** 0.5
        self._linear = Linear(
            self.pool_2_shape,
            10,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Normal(mean=0.0, std=scale)
            ),
        )

    def forward(self, inputs, label):
        x = paddle.nn.functional.relu(self._simple_img_conv_pool_1(inputs))
        x = paddle.nn.functional.relu(self._simple_img_conv_pool_2(x))
        x = paddle.reshape(x, shape=[-1, self.pool_2_shape])
        cost = self._linear(x)
        cost = paddle.nn.functional.softmax(cost)
        loss = paddle.nn.functional.cross_entropy(
            cost, label, reduction='none', use_softmax=False
        )
        avg_loss = paddle.mean(loss)
        return avg_loss


class TestMnist(unittest.TestCase):
    def func_mnist_fp16(self):
        if not base.is_compiled_with_cuda():
            return
        x = np.random.randn(1, 3, 224, 224).astype("float32")
        y = np.random.randint(10, size=[1, 1], dtype="int64")
        with base.dygraph.guard(base.CUDAPlace(0)):
            model = MNIST(dtype="float32")
            x = paddle.to_tensor(x)
            y = paddle.to_tensor(y)

            # using amp.auto_cast because paddle.nn.Conv2D doesn't support setting dtype
            with paddle.amp.auto_cast(dtype='float16'):
                loss = model(x, y)
            print(loss.numpy())

    def test_mnist_fp16(self):
        self.func_mnist_fp16()


if __name__ == "__main__":
    unittest.main()
