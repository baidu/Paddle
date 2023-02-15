# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import platform
import time
import unittest

import numpy as np
from bert import Bert, BertPretrainingCriterion, create_pretraining_dataset

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core

SEED = 2020
EPOCH = 4


def train(to_static, enable_prim, enable_cinn):
    if core.is_compiled_with_cuda():
        paddle.set_device('gpu')
    else:
        paddle.set_device('cpu')
    np.random.seed(SEED)
    paddle.seed(SEED)
    paddle.framework.random._manual_program_seed(SEED)
    fluid.core._set_prim_all_enabled(
        enable_prim and platform.system() == 'Linux'
    )

    bert = Bert()
    criterion = BertPretrainingCriterion(30522)
    if to_static:
        # input_sepc = [
        #     InputSpec(shape=(-1, -1), dtype=paddle.int64, name='input_ids'),
        #     InputSpec(shape=(-1, -1), dtype=paddle.int64, name='segment_ids'),
        #     None,
        #     InputSpec(shape=(-1, 1, 1, -1), dtype=paddle.float32, name='input_mask'),
        #     InputSpec(shape=(-1,), dtype=paddle.int32, name='masked_lm_positions'),
        # ]
        build_strategy = paddle.static.BuildStrategy()
        if enable_cinn:
            build_strategy.build_cinn_pass = True
        bert = paddle.jit.to_static(bert, None, build_strategy=build_strategy)

    optimizer = fluid.optimizer.Adam(parameter_list=bert.parameters())

    train_data_loader, _ = create_pretraining_dataset(
        './bert_training_data.npz', 20, {}, batch_size=32, worker_init=None
    )

    global_step = 0
    losses = []
    for epoch in range(EPOCH):
        for step, batch in enumerate(train_data_loader):
            start_time = time.time()
            (
                input_ids,
                segment_ids,
                input_mask,
                masked_lm_positions,
                masked_lm_labels,
                next_sentence_labels,
                masked_lm_scale,
            ) = batch

            # [[32, 128], [32, 128], [32, 1, 1, 128], [600], [600, 1], [32, 1], [1]]
            # [VarType.INT64, VarType.INT64, VarType.FP32, VarType.INT32, VarType.INT64, VarType.INT64, VarType.FP32]

            # input_ids = paddle.randint(0, 10000, [32, 128], 'int64')
            # input_ids[:, 0] = 101
            # input_ids[:, -1] = 102
            # segment_ids = paddle.randint(0, 2, [32, 128], 'int64')
            # input_mask = paddle.randint(0, 2000, [32, 1, 1, 128]).astype('float32')
            # masked_lm_positions = paddle.randint(0, 2000, [600]).astype('int32')
            # # masked_lm_labels = paddle.randn([600, 1]).astype('int64')
            # masked_lm_labels = paddle.full([600, 1], 1, dtype='int64')
            # # masked_lm_labels = paddle.randint(0, 10000, [600, 1], 'int64')
            # # next_sentence_labels = paddle.randn([32, 1]).astype('int64')
            # next_sentence_labels = paddle.full([32, 1], 0, dtype='int64')
            # # masked_lm_scale = paddle.randn([1]).astype('float32')
            # masked_lm_scale = paddle.full([1], 1, dtype='float32')
            # if i == 0:
            #     input_ids[0, -1] = 0

            # breakpoint()
            prediction_scores, seq_relationship_score = bert(
                input_ids=input_ids,
                token_type_ids=segment_ids,
                attention_mask=input_mask,
                masked_positions=masked_lm_positions,
            )

            loss = criterion(
                prediction_scores,
                seq_relationship_score,
                masked_lm_labels,
                next_sentence_labels,
                masked_lm_scale,
            )

            loss.backward()
            optimizer.minimize(loss)
            bert.clear_gradients()

            losses.append(loss)

            print(
                "global_step: {}, epoch: {}, step: {}, loss: {}, batch_cost: {:.5}".format(
                    global_step,
                    epoch,
                    step,
                    loss.numpy(),
                    time.time() - start_time,
                )
            )

            global_step += 1
            if global_step >= 10:
                return losses


class TestResnet(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dy2st = train(to_static=True, enable_prim=False, enable_cinn=False)

    def test_prim(self):
        dy2st_prim = train(to_static=True, enable_prim=True, enable_cinn=False)
        # NOTE: Now dy2st is equal to dy2st_prim. With the splitting of kernels, the threshold here may need to be adjusted
        # np.testing.assert_allclose(self.dy2st, dy2st_prim, rtol=1e-6)

    @unittest.skipIf(
        not paddle.is_compiled_with_cinn(), "padle is not compiled with CINN"
    )
    def test_cinn(self):
        dy2st_cinn = train(to_static=True, enable_prim=False, enable_cinn=True)

    #     # TODO(0x45f): The following is only temporary thresholds, and the final thresholds needs to be discussed
    #     # np.testing.assert_allclose(self.dy2st[0:2], dy2st_cinn[0:2], rtol=1e-3)
    #     # np.testing.assert_allclose(self.dy2st, dy2st_cinn, rtol=1e-1)

    @unittest.skipIf(
        not paddle.is_compiled_with_cinn(), "padle is not compiled with CINN"
    )
    def test_prim_cinn(self):
        dy2st_prim_cinn = train(
            to_static=True, enable_prim=True, enable_cinn=True
        )

    #     # TODO(0x45f): The following is only temporary thresholds, and the final thresholds need to be discussed
    #     # np.testing.assert_allclose(
    #     #     self.dy2st[0:2], dy2st_prim_cinn[0:2], rtol=1e-2
    #     # )
    #     # np.testing.assert_allclose(self.dy2st, dy2st_prim_cinn, rtol=1e-1)


if __name__ == '__main__':
    unittest.main()
