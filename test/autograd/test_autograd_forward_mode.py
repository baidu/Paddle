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

import sys
import unittest

import numpy as np

import paddle

sys.path.insert(0, '.')


class TestFWDContext(unittest.TestCase):
    def test_enter_exit_dual_level(self):
        paddle.autograd.enter_dual_level()
        assert (
            paddle.autograd.forward_mode._current_level == 0
        ), "The first enter dual level should be 0."
        paddle.autograd.exit_dual_level()
        assert (
            paddle.autograd.forward_mode._current_level == -1
        ), "The current dual level should be -1."

    def test_reverse_on_forward(self):
        x = paddle.randn([100, 100])
        x.stop_gradient = False

        def func(x):
            return paddle.tanh(x)

        with paddle.autograd.dual_level():
            x_primal = x
            x_tangent = paddle.randn(x.shape)
            x_dual = paddle.autograd.forward_mode.make_dual(x_primal, x_tangent)

            y_dual = func(x_dual)

            y_primal, y_tangent = paddle.autograd.forward_mode.unpack_dual(
                y_dual
            )

        print(f"x_primal.stop_gradient = {x_primal.stop_gradient}")
        print(f"x_tangent.stop_gradient = {x_tangent.stop_gradient}")
        print(f"y_primal.stop_gradient = {y_primal.stop_gradient}")
        print(f"y_tangent.stop_gradient = {y_tangent.stop_gradient}")

        y = func(x)
        dy_dx_bwd = paddle.grad(y, x, x_tangent, create_graph=True)[0]

        vv = paddle.randn(y_tangent.shape)
        ddy_ddx_bwdfwd = paddle.grad(
            y_tangent, x_primal, vv, create_graph=False
        )[0]
        ddy_ddx_bwdbwd = paddle.grad(dy_dx_bwd, x, vv, create_graph=False)[0]

        np.testing.assert_allclose(
            dy_dx_bwd.numpy(), y_tangent.numpy(), 1e-6, 1e-6
        )
        np.testing.assert_allclose(
            ddy_ddx_bwdfwd.numpy(), ddy_ddx_bwdbwd.numpy(), 1e-6, 1e-6
        )


if __name__ == "__main__":
    unittest.main()
