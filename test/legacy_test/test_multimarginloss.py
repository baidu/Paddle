#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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


def call_MultiMarginLoss_layer(
    input,
    label,
    p=1,
    margin=1.0,
    weight=None,
    reduction='mean',
):
    triplet_margin_loss = paddle.nn.MultiMarginLoss(
        p=p, margin=margin, weight=weight, reduction=reduction
    )
    res = triplet_margin_loss(
        input=input,
        label=label,
    )
    return res


def call_MultiMarginLoss_functional(
    input,
    label,
    p=1,
    margin=1.0,
    weight=None,
    reduction='mean',
):
    res = paddle.nn.functional.multi_margin_loss(
        input=input,
        label=label,
        p=p,
        margin=margin,
        weight=weight,
        reduction=reduction,
    )
    return res


def test_static(
    place,
    input_np,
    label_np,
    p=1,
    margin=1.0,
    weight_np=None,
    reduction='mean',
    functional=False,
):
    prog = paddle.static.Program()
    startup_prog = paddle.static.Program()
    with paddle.static.program_guard(prog, startup_prog):
        input = paddle.static.data(
            name='input', shape=input_np.shape, dtype=input_np.dtype
        )
        label = paddle.static.data(
            name='label', shape=label_np.shape, dtype=label_np.dtype
        )
        feed_dict = {
            "input": input_np,
            "label": label_np,
        }
        weight = None
        if weight_np is not None:
            weight = paddle.static.data(
                name='weight', shape=weight_np.shape, dtype=weight_np.dtype
            )
            feed_dict['weight'] = weight_np
        if functional:
            res = call_MultiMarginLoss_functional(
                input=input,
                label=label,
                p=p,
                margin=margin,
                weight=weight,
                reduction=reduction,
            )
        else:
            res = call_MultiMarginLoss_layer(
                input=input,
                label=label,
                p=p,
                margin=margin,
                weight=weight,
                reduction=reduction,
            )

        exe = paddle.static.Executor(place)
        static_result = exe.run(prog, feed=feed_dict, fetch_list=[res])
    return static_result[0]


def test_static_data_shape(
    place,
    input_np,
    label_np,
    wrong_label_shape=None,
    weight_np=None,
    wrong_weight_shape=None,
    functional=False,
):
    prog = paddle.static.Program()
    startup_prog = paddle.static.Program()
    with paddle.static.program_guard(prog, startup_prog):
        input = paddle.static.data(
            name='input', shape=input_np.shape, dtype=input_np.dtype
        )
        if wrong_label_shape is None:
            label_shape = label_np.shape
        else:
            label_shape = wrong_label_shape
        label = paddle.static.data(
            name='label', shape=label_shape, dtype=label_np.dtype
        )
        feed_dict = {
            "input": input_np,
            "label": label_np,
        }
        weight = None
        if weight_np is not None:
            if wrong_weight_shape is None:
                weight_shape = weight_np.shape
            else:
                weight_shape = wrong_weight_shape
            weight = paddle.static.data(
                name='weight', shape=weight_shape, dtype=weight_np.dtype
            )
            feed_dict['weight'] = weight_np
        if functional:
            res = call_MultiMarginLoss_functional(
                input=input,
                label=label,
                weight=weight,
            )
        else:
            res = call_MultiMarginLoss_layer(
                input=input,
                label=label,
                weight=weight,
            )

        exe = paddle.static.Executor(place)
        static_result = exe.run(prog, feed=feed_dict, fetch_list=[res])
    return static_result


def test_dygraph(
    place,
    input,
    label,
    p=1,
    margin=1.0,
    weight=None,
    reduction='mean',
    functional=False,
):
    paddle.disable_static()
    input = paddle.to_tensor(input)
    label = paddle.to_tensor(label)

    if weight is not None:
        weight = paddle.to_tensor(weight)
    if functional:
        dy_res = call_MultiMarginLoss_functional(
            input=input,
            label=label,
            p=p,
            margin=margin,
            weight=weight,
            reduction=reduction,
        )
    else:
        dy_res = call_MultiMarginLoss_layer(
            input=input,
            label=label,
            p=p,
            margin=margin,
            weight=weight,
            reduction=reduction,
        )
    dy_result = dy_res.numpy()
    paddle.enable_static()
    return dy_result


def calc_multi_margin_loss(
    input,
    label,
    p=1,
    margin=1.0,
    weight=None,
    reduction='mean',
):
    index_sample = np.array(
        [input[i, label[i]] for i in range(label.size)]
    ).reshape(-1, 1)
    if weight is None:
        expected = (
            np.mean(np.maximum(margin + input - index_sample, 0.0) ** p, axis=1)
            - margin**p / input.shape[1]
        )
    else:
        weight = np.array(
            [weight[label[i]] for i in range(label.size)]
        ).reshape(-1, 1)
        expected = np.mean(
            np.maximum(weight * (margin + input - index_sample), 0.0) ** p,
            axis=1,
        ) - weight * (margin**p / input.shape[1])

    if reduction == 'mean':
        expected = np.mean(expected)
    elif reduction == 'sum':
        expected = np.sum(expected)
    else:
        expected = expected

    return expected


class TestMultiMarginLoss(unittest.TestCase):

    def test_MultiMarginLoss(self):
        batch_size = 5
        num_classes = 2
        shape = (batch_size, num_classes)
        input = np.random.uniform(0.1, 0.8, size=shape).astype(np.float64)
        label = np.random.uniform(0, input.shape[1], size=(batch_size,)).astype(
            np.int64
        )

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
                expected = calc_multi_margin_loss(
                    input=input, label=label, reduction=reduction
                )

                dy_result = test_dygraph(
                    place=place,
                    input=input,
                    label=label,
                    reduction=reduction,
                )

                static_result = test_static(
                    place=place,
                    input_np=input,
                    label_np=label,
                    reduction=reduction,
                )
                np.testing.assert_allclose(static_result, expected)
                np.testing.assert_allclose(static_result, dy_result)
                np.testing.assert_allclose(dy_result, expected)
                static_functional = test_static(
                    place=place,
                    input_np=input,
                    label_np=label,
                    reduction=reduction,
                    functional=True,
                )
                dy_functional = test_dygraph(
                    place=place,
                    input=input,
                    label=label,
                    reduction=reduction,
                    functional=True,
                )
                np.testing.assert_allclose(static_functional, expected)
                np.testing.assert_allclose(static_functional, dy_functional)
                np.testing.assert_allclose(dy_functional, expected)

    def test_MultiMarginLoss_error(self):
        paddle.disable_static()
        self.assertRaises(
            ValueError,
            paddle.nn.MultiMarginLoss,
            reduction="unsupported reduction",
        )
        input = paddle.to_tensor([[0.1, 0.3]], dtype='float32')
        label = paddle.to_tensor([0], dtype='int32')
        self.assertRaises(
            ValueError,
            paddle.nn.functional.multi_margin_loss,
            input=input,
            label=label,
            reduction="unsupported reduction",
        )
        paddle.enable_static()

    def test_MultiMarginLoss_dimension(self):
        paddle.disable_static()

        input = paddle.to_tensor([[0.1, 0.3], [1, 2]], dtype='float32')
        label = paddle.to_tensor([0, 1, 1], dtype='int32')

        self.assertRaises(
            ValueError,
            paddle.nn.functional.multi_margin_loss,
            input=input,
            label=label,
        )
        MMLoss = paddle.nn.MultiMarginLoss()
        self.assertRaises(
            ValueError,
            MMLoss,
            input=input,
            label=label,
        )
        paddle.enable_static()

    def test_MultiMarginLoss_p(self):
        p = 2
        batch_size = 5
        num_classes = 2
        shape = (batch_size, num_classes)
        reduction = 'mean'
        place = paddle.CPUPlace()
        input = np.random.uniform(0.1, 0.8, size=shape).astype(np.float64)
        label = np.random.uniform(0, input.shape[1], size=(batch_size,)).astype(
            np.int64
        )
        expected = calc_multi_margin_loss(
            input=input, p=p, label=label, reduction=reduction
        )

        dy_result = test_dygraph(
            place=place,
            p=p,
            input=input,
            label=label,
            reduction=reduction,
        )

        static_result = test_static(
            place=place,
            p=p,
            input_np=input,
            label_np=label,
            reduction=reduction,
        )
        np.testing.assert_allclose(static_result, expected)
        np.testing.assert_allclose(static_result, dy_result)
        np.testing.assert_allclose(dy_result, expected)
        static_functional = test_static(
            place=place,
            p=p,
            input_np=input,
            label_np=label,
            reduction=reduction,
            functional=True,
        )
        dy_functional = test_dygraph(
            place=place,
            p=p,
            input=input,
            label=label,
            reduction=reduction,
            functional=True,
        )
        np.testing.assert_allclose(static_functional, expected)
        np.testing.assert_allclose(static_functional, dy_functional)
        np.testing.assert_allclose(dy_functional, expected)

    def test_MultiMarginLoss_weight(self):
        batch_size = 5
        num_classes = 2
        shape = (batch_size, num_classes)
        reduction = 'mean'
        place = paddle.CPUPlace()
        input = np.random.uniform(0.1, 0.8, size=shape).astype(np.float64)
        label = np.random.uniform(0, input.shape[1], size=(batch_size,)).astype(
            np.int64
        )
        weight = np.random.uniform(0, 2, size=(num_classes,)).astype(np.float64)
        expected = calc_multi_margin_loss(
            input=input, label=label, weight=weight, reduction=reduction
        )

        dy_result = test_dygraph(
            place=place,
            input=input,
            label=label,
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
        np.testing.assert_allclose(static_result, expected)
        np.testing.assert_allclose(static_result, dy_result)
        np.testing.assert_allclose(dy_result, expected)
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
            input=input,
            label=label,
            weight=weight,
            reduction=reduction,
            functional=True,
        )
        np.testing.assert_allclose(static_functional, expected)
        np.testing.assert_allclose(static_functional, dy_functional)
        np.testing.assert_allclose(dy_functional, expected)

    def test_MultiMarginLoss_static_data_shape(self):
        batch_size = 5
        num_classes = 2
        shape = (batch_size, num_classes)
        place = paddle.CPUPlace()
        input = np.random.uniform(0.1, 0.8, size=shape).astype(np.float64)
        label = np.random.uniform(0, input.shape[1], size=(batch_size,)).astype(
            np.int64
        )
        weight = np.random.uniform(0, 2, size=(num_classes,)).astype(np.float64)

        self.assertRaises(
            ValueError,
            test_static_data_shape,
            place=place,
            input_np=input,
            label_np=label,
            wrong_label_shape=(10,),
            functional=True,
        )
        self.assertRaises(
            ValueError,
            test_static_data_shape,
            place=place,
            input_np=input,
            label_np=label,
            wrong_label_shape=(10,),
            functional=False,
        )
        self.assertRaises(
            ValueError,
            test_static_data_shape,
            place=place,
            input_np=input,
            label_np=label,
            weight_np=weight,
            wrong_weight_shape=(3,),
            functional=True,
        )
        self.assertRaises(
            ValueError,
            test_static_data_shape,
            place=place,
            input_np=input,
            label_np=label,
            weight_np=weight,
            wrong_weight_shape=(3,),
            functional=False,
        )


if __name__ == "__main__":
    unittest.main()
