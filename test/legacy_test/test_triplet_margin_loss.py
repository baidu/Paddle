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


def call_TripletMarginLoss_layer(
    input,
    positive,
    negative,
    p=2,
    margin=0.3,
    swap=False,
    eps=1e-6,
    reduction='mean',
):
    triplet_margin_loss = paddle.nn.TripletMarginLoss(
        p=p, epsilon=eps, margin=margin, swap=swap, reduction=reduction
    )
    res = triplet_margin_loss(
        input=input,
        positive=positive,
        negative=negative,
    )
    return res


def call_TripletMarginLoss_functional(
    input,
    positive,
    negative,
    p=2,
    margin=0.3,
    swap=False,
    eps=1e-6,
    reduction='mean',
):
    res = paddle.nn.functional.triplet_margin_loss(
        input=input,
        positive=positive,
        negative=negative,
        p=p,
        epsilon=eps,
        margin=margin,
        swap=swap,
        reduction=reduction,
    )
    return res


def test_static(
    place,
    input_np,
    positive_np,
    negative_np,
    p=2,
    margin=0.3,
    swap=False,
    eps=1e-6,
    reduction='mean',
    functional=False,
):
    prog = paddle.static.Program()
    startup_prog = paddle.static.Program()
    with paddle.static.program_guard(prog, startup_prog):
        input = paddle.static.data(
            name='input', shape=input_np.shape, dtype='float64'
        )
        positive = paddle.static.data(
            name='positive', shape=positive_np.shape, dtype='float64'
        )
        negative = paddle.static.data(
            name='negative', shape=negative_np.shape, dtype='float64'
        )
        feed_dict = {
            "input": input_np,
            "positive": positive_np,
            "negative": negative_np,
        }

        if functional:
            res = call_TripletMarginLoss_functional(
                input=input,
                positive=positive,
                negative=negative,
                p=p,
                eps=eps,
                margin=margin,
                swap=swap,
                reduction=reduction,
            )
        else:
            res = call_TripletMarginLoss_layer(
                input=input,
                positive=positive,
                negative=negative,
                p=p,
                eps=eps,
                margin=margin,
                swap=swap,
                reduction=reduction,
            )

        exe = paddle.static.Executor(place)
        static_result = exe.run(prog, feed=feed_dict, fetch_list=[res])[0]
    return static_result


def test_dygraph(
    place,
    input,
    positive,
    negative,
    p=2,
    margin=0.3,
    swap=False,
    eps=1e-6,
    reduction='mean',
    functional=False,
):
    paddle.disable_static()
    input = paddle.to_tensor(input)
    positive = paddle.to_tensor(positive)
    negative = paddle.to_tensor(negative)

    if functional:
        dy_res = call_TripletMarginLoss_functional(
            input=input,
            positive=positive,
            negative=negative,
            p=p,
            eps=eps,
            margin=margin,
            swap=swap,
            reduction=reduction,
        )
    else:
        dy_res = call_TripletMarginLoss_layer(
            input=input,
            positive=positive,
            negative=negative,
            p=p,
            eps=eps,
            margin=margin,
            swap=swap,
            reduction=reduction,
        )
    dy_result = dy_res.numpy()
    paddle.enable_static()
    return dy_result


def calc_triplet_margin_loss(
    input,
    positive,
    negative,
    p=2,
    margin=0.3,
    swap=False,
    reduction='mean',
):
    positive_dist = np.linalg.norm((input - positive), p, axis=1)
    negative_dist = np.linalg.norm((input - negative), p, axis=1)

    if swap:
        swap_dist = np.linalg.norm((positive - negative), p, axis=1)
        negative_dist = np.minimum(negative_dist, swap_dist)
    expected = np.maximum(positive_dist - negative_dist + margin, 0)

    if reduction == 'mean':
        expected = np.mean(expected)
    elif reduction == 'sum':
        expected = np.sum(expected)
    else:
        expected = expected

    return expected


class TestTripletMarginLoss(unittest.TestCase):

    def test_TripletMarginLoss(self):
        shape = (2, 2)
        input = np.random.uniform(0.1, 0.8, size=shape).astype(np.float64)
        positive = np.random.uniform(0, 2, size=shape).astype(np.float64)
        negative = np.random.uniform(0, 2, size=shape).astype(np.float64)

        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not paddle.device.is_compiled_with_cuda()
        ):
            places.append(paddle.CPUPlace())
        if paddle.device.is_compiled_with_cuda():
            places.append(paddle.CUDAPlace(0))
        reductions = ['sum', 'mean', 'none']
        for place in places:
            for reduction in reductions:
                expected = calc_triplet_margin_loss(
                    input=input,
                    positive=positive,
                    negative=negative,
                    reduction=reduction,
                )

                dy_result = test_dygraph(
                    place=place,
                    input=input,
                    positive=positive,
                    negative=negative,
                    reduction=reduction,
                )

                static_result = test_static(
                    place=place,
                    input_np=input,
                    positive_np=positive,
                    negative_np=negative,
                    reduction=reduction,
                )
                np.testing.assert_allclose(
                    static_result, expected, rtol=1e-5, atol=1e-8
                )
                np.testing.assert_allclose(
                    static_result, dy_result, rtol=1e-5, atol=1e-8
                )
                np.testing.assert_allclose(
                    dy_result, expected, rtol=1e-5, atol=1e-8
                )
                static_functional = test_static(
                    place=place,
                    input_np=input,
                    positive_np=positive,
                    negative_np=negative,
                    reduction=reduction,
                    functional=True,
                )
                dy_functional = test_dygraph(
                    place=place,
                    input=input,
                    positive=positive,
                    negative=negative,
                    reduction=reduction,
                    functional=True,
                )
                np.testing.assert_allclose(
                    static_functional, expected, rtol=1e-5, atol=1e-8
                )
                np.testing.assert_allclose(
                    static_functional, dy_functional, rtol=1e-5, atol=1e-8
                )
                np.testing.assert_allclose(
                    dy_functional, expected, rtol=1e-5, atol=1e-8
                )

    def test_TripletMarginLoss_error(self):
        paddle.disable_static()
        self.assertRaises(
            ValueError,
            paddle.nn.loss.TripletMarginLoss,
            reduction="unsupported reduction",
        )
        input = paddle.to_tensor([[0.1, 0.3]], dtype='float32')
        positive = paddle.to_tensor([[0.0, 1.0]], dtype='float32')
        negative = paddle.to_tensor([[0.2, 0.1]], dtype='float32')
        self.assertRaises(
            ValueError,
            paddle.nn.functional.triplet_margin_loss,
            input=input,
            positive=positive,
            negative=negative,
            reduction="unsupported reduction",
        )
        paddle.enable_static()

    def test_TripletMarginLoss_dimension(self):
        paddle.disable_static()

        input = paddle.to_tensor([[0.1, 0.3], [1, 2]], dtype='float32')
        positive = paddle.to_tensor([[0.0, 1.0]], dtype='float32')
        negative = paddle.to_tensor([[0.2, 0.1]], dtype='float32')
        self.assertRaises(
            ValueError,
            paddle.nn.functional.triplet_margin_loss,
            input=input,
            positive=positive,
            negative=negative,
        )
        TMLoss = paddle.nn.loss.TripletMarginLoss()
        self.assertRaises(
            ValueError,
            TMLoss,
            input=input,
            positive=positive,
            negative=negative,
        )
        paddle.enable_static()

    def test_TripletMarginLoss_swap(self):
        reduction = 'mean'
        place = paddle.CPUPlace()
        shape = (2, 2)
        input = np.random.uniform(0.1, 0.8, size=shape).astype(np.float64)
        positive = np.random.uniform(0, 2, size=shape).astype(np.float64)
        negative = np.random.uniform(0, 2, size=shape).astype(np.float64)
        expected = calc_triplet_margin_loss(
            input=input,
            swap=True,
            positive=positive,
            negative=negative,
            reduction=reduction,
        )

        dy_result = test_dygraph(
            place=place,
            swap=True,
            input=input,
            positive=positive,
            negative=negative,
            reduction=reduction,
        )

        static_result = test_static(
            place=place,
            swap=True,
            input_np=input,
            positive_np=positive,
            negative_np=negative,
            reduction=reduction,
        )
        np.testing.assert_allclose(
            static_result, expected, rtol=1e-5, atol=1e-8
        )
        np.testing.assert_allclose(
            static_result, dy_result, rtol=1e-5, atol=1e-8
        )
        np.testing.assert_allclose(dy_result, expected, rtol=1e-5, atol=1e-8)
        static_functional = test_static(
            place=place,
            swap=True,
            input_np=input,
            positive_np=positive,
            negative_np=negative,
            reduction=reduction,
            functional=True,
        )
        dy_functional = test_dygraph(
            place=place,
            swap=True,
            input=input,
            positive=positive,
            negative=negative,
            reduction=reduction,
            functional=True,
        )
        np.testing.assert_allclose(
            static_functional, expected, rtol=1e-5, atol=1e-8
        )
        np.testing.assert_allclose(
            static_functional, dy_functional, rtol=1e-5, atol=1e-8
        )
        np.testing.assert_allclose(
            dy_functional, expected, rtol=1e-5, atol=1e-8
        )

    def test_TripletMarginLoss_margin(self):
        paddle.disable_static()

        input = paddle.to_tensor([[0.1, 0.3]], dtype='float32')
        positive = paddle.to_tensor([[0.0, 1.0]], dtype='float32')
        negative = paddle.to_tensor([[0.2, 0.1]], dtype='float32')
        margin = -0.5
        self.assertRaises(
            ValueError,
            paddle.nn.functional.triplet_margin_loss,
            margin=margin,
            input=input,
            positive=positive,
            negative=negative,
        )
        paddle.enable_static()

    def test_TripletMarginLoss_p(self):
        p = 3
        shape = (2, 2)
        reduction = 'mean'
        place = paddle.CPUPlace()
        input = np.random.uniform(0.1, 0.8, size=shape).astype(np.float64)
        positive = np.random.uniform(0, 2, size=shape).astype(np.float64)
        negative = np.random.uniform(0, 2, size=shape).astype(np.float64)
        expected = calc_triplet_margin_loss(
            input=input,
            p=p,
            positive=positive,
            negative=negative,
            reduction=reduction,
        )

        dy_result = test_dygraph(
            place=place,
            p=p,
            input=input,
            positive=positive,
            negative=negative,
            reduction=reduction,
        )

        static_result = test_static(
            place=place,
            p=p,
            input_np=input,
            positive_np=positive,
            negative_np=negative,
            reduction=reduction,
        )
        np.testing.assert_allclose(
            static_result, expected, rtol=1e-5, atol=1e-8
        )
        np.testing.assert_allclose(
            static_result, dy_result, rtol=1e-5, atol=1e-8
        )
        np.testing.assert_allclose(dy_result, expected, rtol=1e-5, atol=1e-8)
        static_functional = test_static(
            place=place,
            p=p,
            input_np=input,
            positive_np=positive,
            negative_np=negative,
            reduction=reduction,
            functional=True,
        )
        dy_functional = test_dygraph(
            place=place,
            p=p,
            input=input,
            positive=positive,
            negative=negative,
            reduction=reduction,
            functional=True,
        )
        np.testing.assert_allclose(
            static_functional, expected, rtol=1e-5, atol=1e-8
        )
        np.testing.assert_allclose(
            static_functional, dy_functional, rtol=1e-5, atol=1e-8
        )
        np.testing.assert_allclose(
            dy_functional, expected, rtol=1e-5, atol=1e-8
        )


if __name__ == "__main__":
    unittest.main()
