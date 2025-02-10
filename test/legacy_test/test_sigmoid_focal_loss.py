# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import unittest

import numpy as np

import paddle
from paddle import base


def call_sfl_functional(
    logit, label, normalizer, alpha=0.25, gamma=2.0, reduction='sum'
):
    res = paddle.nn.functional.sigmoid_focal_loss(
        logit, label, normalizer, alpha=alpha, gamma=gamma, reduction=reduction
    )
    return res


def test_static(
    place,
    logit_np,
    label_np,
    normalizer_np,
    alpha=0.25,
    gamma=2.0,
    reduction='sum',
):
    paddle.enable_static()
    prog = paddle.static.Program()
    startup_prog = paddle.static.Program()
    with paddle.static.program_guard(prog, startup_prog):
        logit = paddle.static.data(
            name='logit', shape=logit_np.shape, dtype='float64'
        )
        label = paddle.static.data(
            name='label', shape=label_np.shape, dtype='float64'
        )
        feed_dict = {"logit": logit_np, "label": label_np}

        normalizer = None
        if normalizer_np is not None:
            normalizer = paddle.static.data(
                name='normalizer', shape=normalizer_np.shape, dtype='float64'
            )
            feed_dict["normalizer"] = normalizer_np

        res = call_sfl_functional(
            logit, label, normalizer, alpha, gamma, reduction
        )
        exe = paddle.static.Executor(place)
        static_result = exe.run(prog, feed=feed_dict, fetch_list=[res])
    return static_result


def test_dygraph(
    place,
    logit_np,
    label_np,
    normalizer_np,
    alpha=0.25,
    gamma=2.0,
    reduction='sum',
):
    paddle.disable_static()
    logit = paddle.to_tensor(logit_np)
    label = paddle.to_tensor(label_np)
    normalizer = None
    if normalizer_np is not None:
        normalizer = paddle.to_tensor(normalizer_np)
    dy_res = call_sfl_functional(
        logit, label, normalizer, alpha, gamma, reduction
    )
    dy_result = dy_res.numpy()
    paddle.enable_static()
    return dy_result


def calc_sigmoid_focal_loss(
    logit_np, label_np, normalizer_np, alpha=0.25, gamma=2.0, reduction='sum'
):
    loss = (
        np.maximum(logit_np, 0)
        - logit_np * label_np
        + np.log(1 + np.exp(-np.abs(logit_np)))
    )

    pred = 1 / (1 + np.exp(-logit_np))
    p_t = pred * label_np + (1 - pred) * (1 - label_np)

    if alpha is not None:
        alpha_t = alpha * label_np + (1 - alpha) * (1 - label_np)
        loss = alpha_t * loss

    if gamma is not None:
        loss = loss * ((1 - p_t) ** gamma)

    if normalizer_np is not None:
        loss = loss / normalizer_np

    if reduction == 'mean':
        loss = np.mean(loss)
    elif reduction == 'sum':
        loss = np.sum(loss)

    return loss


class TestSigmoidFocalLoss(unittest.TestCase):

    def test_SigmoidFocalLoss(self):
        logit_np = np.random.uniform(0.1, 0.8, size=(2, 3, 4, 10)).astype(
            np.float64
        )
        label_np = np.random.randint(0, 2, size=(2, 3, 4, 10)).astype(
            np.float64
        )
        normalizer_nps = [
            np.asarray([np.sum(label_np > 0)], dtype=label_np.dtype),
            None,
        ]
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not base.core.is_compiled_with_cuda()
        ):
            places.append(base.CPUPlace())
        if base.core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        reductions = ['sum', 'mean', 'none']
        alphas = [0.25, 0.5]
        gammas = [3, 0.0]
        for place in places:
            for reduction in reductions:
                for alpha in alphas:
                    for gamma in gammas:
                        for normalizer_np in normalizer_nps:
                            (static_result,) = test_static(
                                place,
                                logit_np,
                                label_np,
                                normalizer_np,
                                alpha,
                                gamma,
                                reduction,
                            )
                            dy_result = test_dygraph(
                                place,
                                logit_np,
                                label_np,
                                normalizer_np,
                                alpha,
                                gamma,
                                reduction,
                            )
                            expected = calc_sigmoid_focal_loss(
                                logit_np,
                                label_np,
                                normalizer_np,
                                alpha,
                                gamma,
                                reduction,
                            )
                            np.testing.assert_allclose(
                                static_result, expected, rtol=1e-05
                            )
                            np.testing.assert_allclose(
                                static_result, dy_result, rtol=1e-05
                            )
                            np.testing.assert_allclose(
                                dy_result, expected, rtol=1e-05
                            )

    def test_SigmoidFocalLoss_error(self):
        paddle.disable_static()
        logit = paddle.to_tensor([[0.97], [0.91], [0.03]], dtype='float32')
        label = paddle.to_tensor([[1.0], [1.0], [0.0]], dtype='float32')
        self.assertRaises(
            ValueError,
            paddle.nn.functional.sigmoid_focal_loss,
            logit=logit,
            label=label,
            normalizer=None,
            reduction="unsupported reduction",
        )
        paddle.enable_static()


if __name__ == "__main__":
    unittest.main()
