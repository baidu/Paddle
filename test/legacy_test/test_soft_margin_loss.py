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


def test_static_layer(
    place,
    input_np,
    label_np,
    reduction='mean',
):
    paddle.enable_static()
    prog = paddle.static.Program()
    startup_prog = paddle.static.Program()
    with paddle.static.program_guard(prog, startup_prog):
        input = paddle.static.data(
            name='input', shape=input_np.shape, dtype=input_np.dtype
        )
        label = paddle.static.data(
            name='label', shape=label_np.shape, dtype=label_np.dtype
        )
        sm_loss = paddle.nn.loss.SoftMarginLoss(reduction=reduction)
        res = sm_loss(input, label)
        exe = paddle.static.Executor(place)
        (static_result,) = exe.run(
            prog, feed={"input": input_np, "label": label_np}, fetch_list=[res]
        )
    return static_result


def test_static_functional(
    place,
    input_np,
    label_np,
    reduction='mean',
):
    paddle.enable_static()
    prog = paddle.static.Program()
    startup_prog = paddle.static.Program()
    with paddle.static.program_guard(prog, startup_prog):
        input = paddle.static.data(
            name='input', shape=input_np.shape, dtype=input_np.dtype
        )
        label = paddle.static.data(
            name='label', shape=label_np.shape, dtype=label_np.dtype
        )

        res = paddle.nn.functional.soft_margin_loss(
            input, label, reduction=reduction
        )
        exe = paddle.static.Executor(place)
        (static_result,) = exe.run(
            prog, feed={"input": input_np, "label": label_np}, fetch_list=[res]
        )
    return static_result


def test_dygraph_layer(
    place,
    input_np,
    label_np,
    reduction='mean',
):
    paddle.disable_static()
    sm_loss = paddle.nn.loss.SoftMarginLoss(reduction=reduction)
    dy_res = sm_loss(paddle.to_tensor(input_np), paddle.to_tensor(label_np))
    dy_result = dy_res.numpy()
    paddle.enable_static()
    return dy_result


def test_dygraph_functional(
    place,
    input_np,
    label_np,
    reduction='mean',
):
    paddle.disable_static()
    input = paddle.to_tensor(input_np)
    label = paddle.to_tensor(label_np)

    dy_res = paddle.nn.functional.soft_margin_loss(
        input, label, reduction=reduction
    )
    dy_result = dy_res.numpy()
    paddle.enable_static()
    return dy_result


def calc_softmarginloss(
    input_np,
    label_np,
    reduction='mean',
):
    expected = np.log(1 + np.exp(-label_np * input_np))
    # expected = np.mean(expected, axis=-1)

    if reduction == 'mean':
        expected = np.mean(expected)
    elif reduction == 'sum':
        expected = np.sum(expected)
    else:
        expected = expected

    return expected


class TestSoftMarginLoss(unittest.TestCase):

    def test_SoftMarginLoss(self):
        input_np = np.random.uniform(0.1, 0.8, size=(5, 5)).astype(np.float64)
        types = [np.int32, np.int64, np.float32, np.float64]
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
                for _type in types:
                    label_np = np.random.randint(0, 2, size=(5, 5)).astype(
                        _type
                    )
                    label_np[label_np == 0] = -1
                    static_result = test_static_layer(
                        place, input_np, label_np, reduction
                    )
                    dy_result = test_dygraph_layer(
                        place, input_np, label_np, reduction
                    )
                    expected = calc_softmarginloss(
                        input_np, label_np, reduction
                    )
                    np.testing.assert_allclose(
                        static_result, expected, rtol=1e-05
                    )
                    np.testing.assert_allclose(
                        static_result, dy_result, rtol=1e-05
                    )
                    np.testing.assert_allclose(dy_result, expected, rtol=1e-05)
                    static_functional = test_static_functional(
                        place, input_np, label_np, reduction
                    )
                    dy_functional = test_dygraph_functional(
                        place, input_np, label_np, reduction
                    )
                    np.testing.assert_allclose(
                        static_functional, expected, rtol=1e-05
                    )
                    np.testing.assert_allclose(
                        static_functional, dy_functional, rtol=1e-05
                    )
                    np.testing.assert_allclose(
                        dy_functional, expected, rtol=1e-05
                    )

    def test_SoftMarginLoss_error(self):
        paddle.disable_static()
        self.assertRaises(
            ValueError,
            paddle.nn.loss.SoftMarginLoss,
            reduction="unsupported reduction",
        )
        input = paddle.to_tensor([[0.1, 0.3]], dtype='float32')
        label = paddle.to_tensor([[-1.0, 1.0]], dtype='float32')
        self.assertRaises(
            ValueError,
            paddle.nn.functional.soft_margin_loss,
            input=input,
            label=label,
            reduction="unsupported reduction",
        )
        paddle.enable_static()


if __name__ == "__main__":
    unittest.main()
