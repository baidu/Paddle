# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
from test_prim_sub_graph_backward_dynamic_shape import (
    TestPrimBaseWithGrad,
    TestPrimThreeWithGrad,
    TestPrimTwoWithGrad,
    apply_to_static,
)

import paddle
from paddle.framework import core
from paddle.static import InputSpec


def floor_net(x):
    return paddle.floor(x)


def fmax_net(x, y):
    return paddle.fmax(x, y)


def fmin_net(x, y):
    return paddle.fmin(x, y)


def gather_net(x, y):
    return paddle.gather(x, y, 1)


def gather_nd_net(x, y):
    return paddle.gather_nd(x, y)


def gelu_net1(x):
    return paddle.nn.functional.gelu(x, approximate=True)


def gelu_net2(x):
    return paddle.nn.functional.gelu(x, approximate=False)


def group_norm_net1(x, y, z, epsilon=1e-5, num_groups=10):
    return paddle._C_ops.group_norm(x, y, z, epsilon, num_groups, "NCHW")


def group_norm_net2(x, epsilon=1e-5, num_groups=10):
    return paddle._C_ops.group_norm(x, None, None, epsilon, num_groups, "NCHW")


def group_norm_net3(x, y, z, epsilon=1e-5, num_groups=10):
    return paddle._C_ops.group_norm(x, y, z, epsilon, num_groups, "NHWC")


def group_norm_net4(x, epsilon=1e-5, num_groups=10):
    return paddle._C_ops.group_norm(x, None, None, epsilon, num_groups, "NHWC")


def hardsigmoid_net(x):
    return paddle.nn.functional.hardsigmoid(x)


def hardswish_net(x):
    return paddle.nn.functional.hardswish(x)


class TestPrimFloorWithGrad(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.floor_grad"
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = floor_net
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimFmaxWithGrad1(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.fmax_grad"
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, 40]
        self.y_shape = [30, 200, 40]
        self.init_y_shape = [None, None, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.y[2, 10, 5:] = np.nan
        self.net = fmax_net
        self.enable_cinn = False
        self.tol = 1e-6


# class TestPrimFmaxWithGrad2(TestPrimTwoWithGrad):
#     def setUp(self):
#         np.random.seed(2023)
#         self.op_name = "pd_op.fmax_grad"
#         self.dtype = "float32"
#         self.x_shape = [30, 200, 40]
#         self.init_x_shape = [None, None, 40]
#         self.y_shape = [30, 200, 40]
#         self.init_y_shape = [30, 200, 40]
#         self.x = np.random.random(self.x_shape).astype(self.dtype)
#         self.y = np.random.random(self.y_shape).astype(self.dtype)
#         self.x[3, 100, 20:] = np.nan
#         self.net = fmax_net
#         self.enable_cinn = False
#         self.tol = 1e-6


class TestPrimFmaxWithGrad3(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.fmax_grad"
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [30, 200, 40]
        self.y_shape = [30, 200, 40]
        self.init_y_shape = [None, None, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.y[2, 10, 5:] = np.nan
        self.net = fmax_net
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimFmaxWithGrad4(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.fmax_grad"
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [30, 200, 40]
        self.y_shape = [30, 200, 40]
        self.init_y_shape = [None, None, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.x[2, 10, 5:] = np.nan
        self.y[2, 10, 5:] = np.nan
        self.x[10, 9, :] = self.y[10, 9, :]
        self.net = fmax_net
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimFminWithGrad1(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.fmin_grad"
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, 40]
        self.y_shape = [30, 200, 40]
        self.init_y_shape = [None, None, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.y[2, 10, 5:] = np.nan
        self.net = fmin_net
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimFminWithGrad2(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.fmin_grad"
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, 40]
        self.y_shape = [30, 200, 40]
        self.init_y_shape = [30, 200, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.x[3, 100, 20:] = np.nan
        self.net = fmin_net
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimFminWithGrad3(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.fmin_grad"
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [30, 200, 40]
        self.y_shape = [30, 200, 40]
        self.init_y_shape = [None, None, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.y[2, 10, 5:] = np.nan
        self.net = fmin_net
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimFminWithGrad4(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.fmin_grad"
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [30, 200, 40]
        self.y_shape = [30, 200, 40]
        self.init_y_shape = [None, None, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.x[2, 10, 5:] = np.nan
        self.y[2, 10, 5:] = np.nan
        self.x[10, 9, :] = self.y[10, 9, :]
        self.net = fmin_net
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimGatherWithGrad1(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.dtype = "float32"
        self.x_shape = [10, 88, 10]
        self.init_x_shape = [None, None, 10]
        self.y_shape = [3]
        self.init_y_shape = [None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.array([1, 3, 5], dtype="int32")
        self.net = gather_net
        self.enable_cinn = False
        self.tol = 1e-6

    def base_net(self, flag=None):
        if flag == "prim":
            core._set_prim_backward_enabled(True)
        x = paddle.to_tensor(self.x, stop_gradient=False)
        y = paddle.to_tensor(self.y)
        if flag == "prim":
            fn = apply_to_static(
                self.net,
                use_cinn=self.enable_cinn,
                input_spec=[
                    InputSpec(shape=self.init_x_shape, dtype='float32'),
                    InputSpec(shape=self.init_y_shape, dtype='int32'),
                ],
            )
            fn.train()
        else:
            fn = self.net
        res = fn(x, y)
        res.backward()
        x_grad = x.gradient()
        if flag == "prim":
            core._set_prim_backward_enabled(False)
        return res, [x_grad]


class TestPrimGatherWithGrad2(TestPrimGatherWithGrad1):
    def setUp(self):
        np.random.seed(2024)
        self.dtype = "float32"
        self.x_shape = [10, 88, 10]
        self.init_x_shape = [None, 88, None]
        self.y_shape = [3]
        self.init_y_shape = [None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.array([1, 3, 5], dtype="int32")
        self.net = gather_net
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimGatherNdWithGrad(TestPrimGatherWithGrad1):
    def setUp(self):
        np.random.seed(2024)
        self.dtype = "float32"
        self.x_shape = [100, 100]
        self.init_x_shape = [None, None, 10]
        self.y_shape = [2, 2]
        self.init_y_shape = [None, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.array([[1, 1], [2, 1]], dtype="int32")
        self.net = gather_nd_net
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimGeluWithGrad1(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.gelu_grad"
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = gelu_net1
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimGeluWithGrad2(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.gelu_grad"
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = gelu_net2
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimGeluWithGrad3(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.gelu_grad"
        self.dtype = "float16"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, None]
        self.x = np.random.uniform(-1, 1, size=self.x_shape).astype(self.dtype)
        self.net = gelu_net1
        self.enable_cinn = False
        self.rtol = 1e-5
        self.atol = 0.0005

    def test_prim_all_dynamic(self):
        if not paddle.is_compiled_with_cuda():
            return
        place = core.CUDAPlace(0)
        if not core.is_float16_supported(place):
            return

        res_ref, grad_ref = self.base_net()
        res, grad = self.base_net("prim")

        for ref, actual in zip(res_ref, res):
            np.testing.assert_allclose(
                ref, actual, rtol=self.rtol, atol=self.atol
            )

        for dr, d in zip(grad_ref, grad):
            np.testing.assert_allclose(dr, d, rtol=self.rtol, atol=self.atol)


class TestPrimGeluWithGrad4(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.gelu_grad"
        self.dtype = "float16"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, None]
        self.x = np.random.uniform(-1, 1, size=self.x_shape).astype(self.dtype)
        self.net = gelu_net2
        self.enable_cinn = False
        self.rtol = 1e-5
        self.atol = 0.0005

    def test_prim_all_dynamic(self):
        if not paddle.is_compiled_with_cuda():
            return
        place = core.CUDAPlace(0)
        if not core.is_float16_supported(place):
            return

        res_ref, grad_ref = self.base_net()
        res, grad = self.base_net("prim")

        for ref, actual in zip(res_ref, res):
            np.testing.assert_allclose(
                ref, actual, rtol=self.rtol, atol=self.atol
            )

        for dr, d in zip(grad_ref, grad):
            np.testing.assert_allclose(dr, d, rtol=self.rtol, atol=self.atol)


class TestPrimGroupNormWithGrad1(TestPrimThreeWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.group_norm_grad"
        self.dtype = "float32"
        self.x_shape = [30, 60, 50, 60]
        self.init_x_shape = [None, None, None, 60]
        self.y_shape = [60]
        self.init_y_shape = [None]
        self.z_shape = [60]
        self.init_z_shape = [None]
        self.x = np.random.uniform(-0.5, 0.5, self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.z = np.random.random(self.z_shape).astype(self.dtype)
        self.net = group_norm_net1
        self.enable_cinn = False
        self.tol = 1e-4


class TestPrimGroupNormWithGrad2(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.group_norm_grad"
        self.dtype = "float32"
        self.x_shape = [30, 60, 50, 60]
        self.init_x_shape = [None, 60, None, None]
        self.x = np.random.uniform(-0.5, 0.5, self.x_shape).astype(self.dtype)
        self.net = group_norm_net2
        self.enable_cinn = False
        self.tol = 1e-5


class TestPrimGroupNormWithGrad3(TestPrimThreeWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.group_norm_grad"
        self.dtype = "float32"
        self.x_shape = [30, 60, 50, 60]
        self.init_x_shape = [None, 60, None, None]
        self.y_shape = [60]
        self.init_y_shape = [None]
        self.z_shape = [60]
        self.init_z_shape = [None]
        self.x = np.random.uniform(-0.5, 0.5, self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.z = np.random.random(self.z_shape).astype(self.dtype)
        self.net = group_norm_net3
        self.enable_cinn = False
        self.tol = 1e-2


class TestPrimGroupNormWithGrad4(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.group_norm_grad"
        self.dtype = "float32"
        self.x_shape = [30, 60, 50, 60]
        self.init_x_shape = [None, 60, None, None]
        self.x = np.random.uniform(-0.2, 0.2, self.x_shape).astype(self.dtype)
        self.net = group_norm_net4
        self.enable_cinn = False
        self.tol = 1e-2


class TestPrimHardsigmoidWithGrad(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.hardsigmoid_grad"
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = hardsigmoid_net
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimHardswishWithGrad(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.hardswish_grad"
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = hardswish_net
        self.enable_cinn = False
        self.tol = 1e-6


if __name__ == "__main__":
    unittest.main()
