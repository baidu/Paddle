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
import os
import sys
import unittest
from os.path import dirname

import numpy as np

os.environ['FLAGS_prim_all'] = 'true'
os.environ['FLAGS_prim_enable_dynamic'] = 'true'
os.environ['FLAGS_print_ir'] = '1'
os.environ['FLAGS_enable_pir_api'] = '1'
os.environ['FLAGS_use_cinn'] = '1'

import paddle
from paddle import nn
from paddle.static import InputSpec

sys.path.append(dirname(dirname(__file__)))
import utils


class PrepareDecoderAttentionMask(nn.Layer):
    def __init__(self):
        super().__init__()

    # [batch_size, src_length] -> [batch_size, 1, tgt_length, src_length]
    def _expand_2d_mask(self, mask, target_length):
        batch_size, src_length = mask.shape[0], mask.shape[-1]

        mask = mask[:, None, None, :].astype("bool")
        mask.stop_gradient = True
        expanded_mask = mask.expand([batch_size, 1, target_length, src_length])

        return expanded_mask

    def _make_causal_mask(self, input_ids_shape):
        batch_size, seq_len = input_ids_shape

        mask = paddle.tril(paddle.ones((seq_len, seq_len), dtype="bool"))

        # [bs, 1, seq_len, seq_len]
        return mask[None, None, :, :].expand([batch_size, 1, seq_len, seq_len])

    def forward(self, input_ids, attention_mask):
        input_shape = paddle.shape(input_ids)

        expanded_attn_mask = self._expand_2d_mask(
            attention_mask, target_length=input_shape[-1]
        )
        combined_attention_mask = self._make_causal_mask(input_shape)
        if input_shape[-1] > 1:
            expanded_attn_mask = expanded_attn_mask & combined_attention_mask
        expanded_attn_mask = paddle.where(
            expanded_attn_mask, 0.0, paddle.finfo("float32").min
        ).astype("float32")
        return expanded_attn_mask


class TestPrepareDecoderAttentionMask(unittest.TestCase):
    def setUp(self):
        paddle.seed(2024)
        self.prepare_data()

    def prepare_data(self):
        self.input_ids = paddle.randint(
            low=0, high=2048, shape=[1, 2048], dtype="int64"
        )
        self.input_ids.stop_gradient = False

        self.attention_mask = paddle.ones([1, 2048], dtype="bool")
        self.attention_mask.stop_gradient = False

    def eval(self, use_cinn=False, mode="static"):
        net = PrepareDecoderAttentionMask()
        input_spec = [
            InputSpec(shape=[None, None], dtype="int64"),
            InputSpec(shape=[None, None], dtype="bool"),
        ]
        if mode == "static":
            net = utils.apply_to_static(net, use_cinn, input_spec)
            net.eval()
        out = net(self.input_ids, self.attention_mask)
        return out

    def test_eval(self):
        eager_outs = self.eval(mode="eager")
        dy_outs = self.eval(use_cinn=True)

        for cinn_out, dy_out in zip(eager_outs, dy_outs):
            np.testing.assert_allclose(
                cinn_out.numpy(), dy_out.numpy(), atol=1e-8
            )


if __name__ == '__main__':
    unittest.main()
