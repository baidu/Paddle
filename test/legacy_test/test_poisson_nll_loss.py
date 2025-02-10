#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.nn.functional as F
from paddle.base import core

np.random.seed(100)


def ref_poisson_nll_loss(
    input,
    label,
    log_input=True,
    full=False,
    epsilon=1e-8,
    reduction="mean",
):
    if epsilon <= 0:
        raise ValueError(
            f"The value of `epsilon` in PoissonNLLLoss should be positive, but received {epsilon:f}, which is not allowed"
        )

    if reduction not in ['sum', 'mean', 'none']:
        raise ValueError(
            "The value of 'reduction' in SoftMarginLoss should be 'sum', 'mean' or 'none', but "
            f"received {reduction}, which is not allowed."
        )
    loss_out = 0
    if log_input:
        loss_out = np.exp(input) - label * input
    else:
        loss_out = input - label * np.log(input + epsilon)
    if full:
        stirling_approx = (
            label * np.log(label) - label + 0.5 * np.log(2 * np.pi * label)
        )
        loss_out += np.where(
            label > 1, stirling_approx, np.zeros_like(stirling_approx)
        )

    if reduction == 'none':
        return loss_out
    elif reduction == 'sum':
        return [np.sum(loss_out)]
    elif reduction == 'mean':
        return [np.mean(loss_out)]


class TestPoissonNLLLossBasicCase(unittest.TestCase):
    def setUp(self, dtype="float32"):
        self.shape = [10, 2]
        self.dtype = dtype
        self.input_np = np.random.random(self.shape).astype(self.dtype)
        self.label_np = np.random.random(self.shape).astype(self.dtype)
        self.place = (
            paddle.CUDAPlace(0)
            if core.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    def test_static_case(
        self,
        dtype="float32",
        log_input=True,
        full=False,
        epsilon=1e-8,
        reduction="mean",
    ):
        self.setUp(dtype)
        paddle.enable_static()
        prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        with paddle.static.program_guard(prog, startup_prog):
            input = paddle.static.data('input', self.shape, dtype)
            label = paddle.static.data('label', self.shape, dtype)
            out1 = F.poisson_nll_loss(
                input,
                label,
                log_input=log_input,
                full=full,
                epsilon=epsilon,
                reduction=reduction,
            )
            poisson_nll_loss = paddle.nn.PoissonNLLLoss(
                log_input=log_input,
                full=full,
                epsilon=epsilon,
                reduction=reduction,
            )
            out2 = poisson_nll_loss(input, label)
        exe = paddle.static.Executor(self.place)
        exe.run(startup_prog)
        res = exe.run(
            prog,
            feed={'input': self.input_np, 'label': self.label_np},
            fetch_list=[out1, out2],
        )
        out_ref = ref_poisson_nll_loss(
            self.input_np,
            self.label_np,
            log_input=log_input,
            full=full,
            epsilon=epsilon,
            reduction=reduction,
        )
        for r in res:
            np.allclose(out_ref, r, rtol=1e-5)

    def test_dynamic_case(
        self,
        dtype="float32",
        log_input=True,
        full=False,
        epsilon=1e-8,
        reduction="mean",
        type=None,
    ):
        self.setUp(dtype)
        paddle.disable_static(self.place)

        input_x = paddle.to_tensor(self.input_np)
        label = paddle.to_tensor(self.label_np)
        out_ref = ref_poisson_nll_loss(
            self.input_np,
            self.label_np,
            log_input=log_input,
            full=full,
            epsilon=epsilon,
            reduction=reduction,
        )
        out1 = F.poisson_nll_loss(
            input_x,
            label,
            log_input=log_input,
            full=full,
            epsilon=epsilon,
            reduction=reduction,
        )
        if type == 'test_err_reduction':
            self.assertRaises(
                ValueError,
                paddle.nn.functional.poisson_nll_loss,
                input=input_x,
                label=label,
                log_input=log_input,
                full=full,
                epsilon=epsilon,
                reduction="unsupported reduction",
            )
        elif type == 'test_err_epsilon':
            self.assertRaises(
                ValueError,
                paddle.nn.functional.poisson_nll_loss,
                input=input_x,
                label=label,
                log_input=log_input,
                full=full,
                epsilon=-1,
                reduction="mean",
            )
        poisson_nll_loss = paddle.nn.PoissonNLLLoss(
            log_input=log_input, full=full, epsilon=epsilon, reduction=reduction
        )
        out2 = poisson_nll_loss(input_x, label)

        for r in [out1, out2]:
            np.allclose(out_ref, r.numpy(), rtol=1e-5)
        paddle.enable_static()

    def test_api(self):
        pass


class TestPoissonNLLLossErrCase(TestPoissonNLLLossBasicCase):
    def test_err_reduction(self):
        self.test_dynamic_case(type="test_err_reduction")

    def test_err_epsilon(self):
        self.test_dynamic_case(type="test_err_epsilon")

    def test_api(self):
        self.test_err_reduction()
        self.test_err_epsilon()


class TestPoissonNLLLossFloat16Case(TestPoissonNLLLossBasicCase):
    def test_api(self):
        if core.is_compiled_with_cuda():
            self.test_static_case(dtype="float16")
            self.test_dynamic_case(dtype="float16")


class TestPoissonNLLLossBfloat16Case(TestPoissonNLLLossBasicCase):
    def test_api(self):
        if core.is_compiled_with_cuda():
            self.test_static_case(dtype="uint16")
            self.test_dynamic_case(dtype="uint16")


class TestPoissonNLLLossFloat32Case(TestPoissonNLLLossBasicCase):
    def test_api(self):
        self.test_static_case(dtype="float32")
        self.test_dynamic_case(dtype="float32")


class TestPoissonNLLLossFloat64Case(TestPoissonNLLLossBasicCase):
    def test_api(self):
        self.test_static_case(dtype="float64")
        self.test_dynamic_case(dtype="float64")


class TestPoissonNLLLossNoLogInputCase(TestPoissonNLLLossBasicCase):
    def test_api(self):
        self.test_static_case(log_input=False)
        self.test_dynamic_case(log_input=False)


class TestPoissonNLLLossFulllossCase(TestPoissonNLLLossBasicCase):
    def test_api(self):
        self.test_static_case(full=True)
        self.test_dynamic_case(full=True)


class TestPoissonNLLLossSumReductionCase(TestPoissonNLLLossBasicCase):
    def test_api(self):
        self.test_static_case(reduction="sum")
        self.test_dynamic_case(reduction="sum")


if __name__ == "__main__":
    unittest.main()
