#   Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.distributed import fleet
from paddle.distributed.collective import (
    _c_softmax_with_multi_label_cross_entropy,
)


def stable_softmax(x):
    """Compute the softmax of vector x in a numerically stable way."""
    # clip to shiftx, otherwise, when calc loss with
    # log(exp(shiftx)), may get log(0)=INF
    shiftx = x - np.max(x)
    exps = np.exp(shiftx)
    return exps / np.sum(exps)


def cross_entropy(softmax, label, soft_label, axis, ignore_index=-1):
    if soft_label:
        return (-label * np.log(softmax)).sum(axis=axis, keepdims=True)
    shape = softmax.shape
    axis %= len(shape)
    n = int(np.prod(shape[:axis]))
    axis_dim = shape[axis]
    remain = int(np.prod(shape[axis + 1 :]))
    softmax_reshape = softmax.reshape((n, axis_dim, remain))
    label_reshape = label.reshape((n, 1, remain))
    result = np.zeros_like(label_reshape, dtype=softmax.dtype)
    for i in range(n):
        for j in range(remain):
            lbl = label_reshape[i, 0, j]
            if lbl != ignore_index:
                result[i, 0, j] -= np.log(softmax_reshape[i, lbl, j])
    return result.reshape(label.shape)


def softmax_with_cross_entropy_grad(softmax, label, loss_grad, axis):
    logit_grad = softmax.copy()
    shape = softmax.shape
    axis %= len(shape)
    n = int(np.prod(shape[:axis]))
    d = int(np.prod(shape[axis:]))
    for i in range(n * d):
        row = int(i / d)
        col = i % d
        logit_grad[row][col] = (
            logit_grad[row][col] - label[row][col]
        ) * loss_grad[row][0]
    return logit_grad


class TestCSoftmaxWithCrossEntropy(unittest.TestCase):
    def setUp(self):
        self.num_class = 100
        self.batch_size = 1024
        self.C = 4
        strategy = fleet.DistributedStrategy()
        model_parallel_size = 2
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": model_parallel_size,
            "pp_degree": 1,
        }
        fleet.init(is_collective=True, strategy=strategy)

    def get_input_data(self, data_type="float32"):
        # get data that is shared by both ranks
        np.random.seed(os.getuid())

        label = []
        for _ in range(self.batch_size):
            tmp = np.random.choice(
                range(0, self.num_class), size=self.C, replace=False
            )
            label.append(tmp)
        label = np.array(label)

        prob = np.random.rand(self.batch_size, self.C)
        row_sum = np.sum(prob, axis=1, keepdims=True)
        smooth_weight = prob / row_sum

        local_elements = int(self.num_class / 2)
        # get input data for rank 0
        np.random.seed(0)
        input0 = np.random.uniform(
            low=-40.0, high=40.0, size=(self.batch_size, local_elements)
        ).astype(data_type)

        # get input data for rank 1
        np.random.seed(1)
        input1 = np.random.uniform(
            low=-40.0, high=40.0, size=(self.batch_size, local_elements)
        ).astype(data_type)

        np.random.seed(os.getuid())
        loss_grad = np.random.rand(self.batch_size, 1).astype(data_type)

        return input0, input1, label, smooth_weight, loss_grad

    def cross_entropy_np(
        self, input0, input1, label, smooth_weight, ignore_index, loss_grad
    ):
        # get combined input data
        inputs = np.concatenate((input0, input1), axis=1)

        # calculate analytic result
        need_softmax = np.apply_along_axis(stable_softmax, 1, inputs)

        need_label = np.zeros_like(inputs)
        for i in range(len(label)):
            for j, c in enumerate(label[i]):
                need_label[i][c] = smooth_weight[i][j]

        need_loss = cross_entropy(
            need_softmax, need_label, True, 1, ignore_index=ignore_index
        )

        need_logits_grad = softmax_with_cross_entropy_grad(
            need_softmax, need_label, loss_grad, 1
        )
        return need_softmax, need_loss, need_logits_grad

    def test_model(self, data_type="float32"):
        rank = fleet.worker_index()

        input0, input1, label, smooth_weight, loss_grad = self.get_input_data(
            data_type=data_type
        )

        # convert to paddle tensor
        input0_pd = paddle.to_tensor(input0, stop_gradient=False)
        input1_pd = paddle.to_tensor(input1, stop_gradient=False)
        label_pd = paddle.to_tensor(label)
        smooth_weight_pd = paddle.to_tensor(smooth_weight).astype(data_type)
        loss_grad_pd = paddle.to_tensor(loss_grad)

        ignore_index = -100

        if rank == 0:
            loss, softmax = _c_softmax_with_multi_label_cross_entropy(
                input0_pd,
                label_pd,
                smooth_weight_pd,
                ignore_index=ignore_index,
                return_softmax=True,
            )
        else:
            loss, softmax = _c_softmax_with_multi_label_cross_entropy(
                input1_pd,
                label_pd,
                smooth_weight_pd,
                ignore_index=ignore_index,
                return_softmax=True,
            )
        paddle.device.cuda.synchronize()
        softmax_list = []
        paddle.distributed.all_gather(softmax_list, softmax)
        softmax = np.concatenate(
            (softmax_list[0].numpy(), softmax_list[1].numpy()), axis=1
        )

        paddle.autograd.backward([loss], [loss_grad_pd])

        grad_list = []
        paddle.distributed.all_gather(grad_list, eval(f'input{rank}_pd.grad'))
        inputs_grad = paddle.concat(grad_list, axis=-1)

        # calculate numpy cross entropy result
        need_softmax, need_loss, need_logits_grad = self.cross_entropy_np(
            input0, input1, label, smooth_weight, ignore_index, loss_grad
        )

        # compare results
        rtol_f = 1e-6
        np.testing.assert_allclose(loss.numpy(), need_loss, rtol=rtol_f)
        np.testing.assert_allclose(softmax, need_softmax, rtol=rtol_f)
        rtol_b = 1e-5
        np.testing.assert_allclose(
            inputs_grad.numpy(), need_logits_grad, rtol=rtol_b
        )

    def test_model_with_sum_multi_label_loss(self, data_type="float32"):
        rank = fleet.worker_index()

        input0, input1, label, smooth_weight, loss_grad = self.get_input_data(
            data_type=data_type
        )

        # convert to paddle tensor
        input0_pd = paddle.to_tensor(input0, stop_gradient=False)
        input1_pd = paddle.to_tensor(input1, stop_gradient=False)
        label_pd = paddle.to_tensor(label)
        smooth_weight_pd = paddle.to_tensor(smooth_weight).astype(data_type)
        loss_grad_pd = paddle.to_tensor(loss_grad)

        ignore_index = -100

        if rank == 0:
            loss_tmp, softmax = _c_softmax_with_multi_label_cross_entropy(
                input0_pd,
                label_pd,
                smooth_weight_pd,
                ignore_index=ignore_index,
                return_softmax=True,
                sum_multi_label_loss=False,
            )
        else:
            loss_tmp, softmax = _c_softmax_with_multi_label_cross_entropy(
                input1_pd,
                label_pd,
                smooth_weight_pd,
                ignore_index=ignore_index,
                return_softmax=True,
                sum_multi_label_loss=False,
            )
        paddle.device.cuda.synchronize()
        loss = paddle.sum(loss_tmp, axis=-1, keepdim=True)
        softmax_list = []
        paddle.distributed.all_gather(softmax_list, softmax)
        softmax = np.concatenate(
            (softmax_list[0].numpy(), softmax_list[1].numpy()), axis=1
        )

        paddle.autograd.backward([loss], [loss_grad_pd])

        grad_list = []
        paddle.distributed.all_gather(grad_list, eval(f'input{rank}_pd.grad'))
        inputs_grad = paddle.concat(grad_list, axis=-1)

        # calculate numpy cross entropy result
        need_softmax, need_loss, need_logits_grad = self.cross_entropy_np(
            input0, input1, label, smooth_weight, ignore_index, loss_grad
        )

        # compare results
        rtol_f = 1e-6
        np.testing.assert_allclose(loss.numpy(), need_loss, rtol=rtol_f)
        np.testing.assert_allclose(softmax, need_softmax, rtol=rtol_f)
        rtol_b = 1e-5
        np.testing.assert_allclose(
            inputs_grad.numpy(), need_logits_grad, rtol=rtol_b
        )


if __name__ == '__main__':
    unittest.main()
