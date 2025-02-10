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

import os
import unittest

import numpy as np

import paddle


def call_MultiLabelSoftMarginLoss_layer(
    input,
    label,
    weight=None,
    reduction='mean',
):
    multi_label_margin_loss = paddle.nn.MultiLabelSoftMarginLoss(
        weight=weight, reduction=reduction
    )
    res = multi_label_margin_loss(
        input=input,
        label=label,
    )
    return res


def call_MultiLabelSoftMarginLoss_functional(
    input,
    label,
    weight=None,
    reduction='mean',
):
    res = paddle.nn.functional.multi_label_soft_margin_loss(
        input,
        label,
        reduction=reduction,
        weight=weight,
    )
    return res


def test_static(
    place,
    input_np,
    label_np,
    weight_np=None,
    reduction='mean',
    functional=False,
):
    paddle.enable_static()
    prog = paddle.static.Program()
    startup_prog = paddle.static.Program()
    with paddle.static.program_guard(prog, startup_prog):
        input = paddle.static.data(
            name='input', shape=input_np.shape, dtype='float64'
        )
        label = paddle.static.data(
            name='label', shape=label_np.shape, dtype='float64'
        )
        feed_dict = {
            "input": input_np,
            "label": label_np,
        }
        weight = None
        if weight_np is not None:
            weight = paddle.static.data(
                name='weight', shape=weight_np.shape, dtype='float64'
            )
            feed_dict['weight'] = weight_np

        if functional:
            res = call_MultiLabelSoftMarginLoss_functional(
                input=input, label=label, weight=weight, reduction=reduction
            )
        else:
            res = call_MultiLabelSoftMarginLoss_layer(
                input=input, label=label, weight=weight, reduction=reduction
            )

        exe = paddle.static.Executor(place)
        (static_result,) = exe.run(prog, feed=feed_dict, fetch_list=[res])
    return static_result


def test_dygraph(
    place, input_np, label_np, weight=None, reduction='mean', functional=False
):
    with paddle.base.dygraph.base.guard():
        input = paddle.to_tensor(input_np)
        label = paddle.to_tensor(label_np)
        if weight is not None:
            weight = paddle.to_tensor(weight)

        if functional:
            dy_res = call_MultiLabelSoftMarginLoss_functional(
                input=input, label=label, weight=weight, reduction=reduction
            )
        else:
            dy_res = call_MultiLabelSoftMarginLoss_layer(
                input=input, label=label, weight=weight, reduction=reduction
            )
        dy_result = dy_res.numpy()
        return dy_result


def calc_multi_label_margin_loss(
    input,
    label,
    weight=None,
    reduction="mean",
):
    def LogSigmoid(x):
        return np.log(1 / (1 + np.exp(-x)))

    loss = -(label * LogSigmoid(input) + (1 - label) * LogSigmoid(-input))

    if weight is not None:
        loss = loss * weight

    loss = loss.mean(axis=-1)  # only return N loss values

    if reduction == "none":
        return loss
    elif reduction == "mean":
        return np.mean(loss)
    elif reduction == "sum":
        return np.sum(loss)


class TestMultiLabelMarginLoss(unittest.TestCase):

    def test_MultiLabelSoftMarginLoss(self):
        input = np.random.uniform(0.1, 0.8, size=(5, 5)).astype(np.float64)
        label = np.random.randint(0, 2, size=(5, 5)).astype(np.float64)

        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not paddle.device.is_compiled_with_cuda()
        ):
            places.append('cpu')
        if paddle.device.is_compiled_with_cuda():
            places.append('gpu')
        reductions = ['sum', 'mean', 'none']
        for place in places:
            for reduction in reductions:
                expected = calc_multi_label_margin_loss(
                    input=input, label=label, reduction=reduction
                )

                dy_result = test_dygraph(
                    place=place,
                    input_np=input,
                    label_np=label,
                    reduction=reduction,
                )

                static_result = test_static(
                    place=place,
                    input_np=input,
                    label_np=label,
                    reduction=reduction,
                )
                np.testing.assert_allclose(static_result, expected, rtol=1e-05)
                np.testing.assert_allclose(static_result, dy_result, rtol=1e-05)
                np.testing.assert_allclose(dy_result, expected, rtol=1e-05)
                static_functional = test_static(
                    place=place,
                    input_np=input,
                    label_np=label,
                    reduction=reduction,
                    functional=True,
                )
                dy_functional = test_dygraph(
                    place=place,
                    input_np=input,
                    label_np=label,
                    reduction=reduction,
                    functional=True,
                )
                np.testing.assert_allclose(
                    static_functional, expected, rtol=1e-05
                )
                np.testing.assert_allclose(
                    static_functional, dy_functional, rtol=1e-05
                )
                np.testing.assert_allclose(dy_functional, expected, rtol=1e-05)

    def test_MultiLabelSoftMarginLoss_error(self):
        paddle.disable_static()
        self.assertRaises(
            ValueError,
            paddle.nn.MultiLabelSoftMarginLoss,
            reduction="unsupported reduction",
        )
        input = paddle.to_tensor([[0.1, 0.3]], dtype='float32')
        label = paddle.to_tensor([[0.0, 1.0]], dtype='float32')
        self.assertRaises(
            ValueError,
            paddle.nn.functional.multi_label_soft_margin_loss,
            input=input,
            label=label,
            reduction="unsupported reduction",
        )
        paddle.enable_static()

    def test_MultiLabelSoftMarginLoss_weights(self):
        input = np.random.uniform(0.1, 0.8, size=(5, 5)).astype(np.float64)
        label = np.random.randint(0, 2, size=(5, 5)).astype(np.float64)
        weight = np.random.randint(0, 2, size=(5, 5)).astype(np.float64)
        place = 'cpu'
        reduction = 'mean'
        expected = calc_multi_label_margin_loss(
            input=input, label=label, weight=weight, reduction=reduction
        )

        dy_result = test_dygraph(
            place=place,
            input_np=input,
            label_np=label,
            weight=weight,
            reduction=reduction,
        )

        static_result = test_static(
            place=place,
            input_np=input,
            label_np=label,
            weight_np=weight,
            reduction=reduction,
        )
        np.testing.assert_allclose(static_result, expected, rtol=1e-05)
        np.testing.assert_allclose(static_result, dy_result, rtol=1e-05)
        np.testing.assert_allclose(dy_result, expected, rtol=1e-05)
        static_functional = test_static(
            place=place,
            input_np=input,
            label_np=label,
            weight_np=weight,
            reduction=reduction,
            functional=True,
        )
        dy_functional = test_dygraph(
            place=place,
            input_np=input,
            label_np=label,
            weight=weight,
            reduction=reduction,
            functional=True,
        )
        np.testing.assert_allclose(static_functional, expected, rtol=1e-05)
        np.testing.assert_allclose(static_functional, dy_functional, rtol=1e-05)
        np.testing.assert_allclose(dy_functional, expected, rtol=1e-05)

    def test_MultiLabelSoftMarginLoss_dimension(self):
        paddle.disable_static()

        input = paddle.to_tensor([[0.1, 0.3], [1, 2]], dtype='float32')
        label = paddle.to_tensor([[0.2, 0.1]], dtype='float32')
        self.assertRaises(
            ValueError,
            paddle.nn.functional.multi_label_soft_margin_loss,
            input=input,
            label=label,
        )
        paddle.enable_static()


if __name__ == "__main__":
    unittest.main()
