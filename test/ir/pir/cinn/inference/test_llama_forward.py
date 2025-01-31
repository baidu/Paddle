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

os.environ["FLAGS_prim_forward_blacklist"] = "pd_op.embedding;pd_op.softmax"

import paddle
from paddle.static import InputSpec

sys.path.append(dirname(dirname(__file__)))

import llama_test_model
import utils


class TestLlamaModel(unittest.TestCase):
    def setUp(self):
        paddle.seed(2024)
        self.prepare_data()

    def prepare_data(self):
        self.config = llama_test_model.LlamaConfig()
        self.input_ids = paddle.to_tensor(
            [
                [
                    1,
                    29871,
                    31201,
                    236,
                    138,
                    141,
                    30287,
                    30557,
                    30015,
                    233,
                    187,
                    172,
                    31969,
                    31325,
                    31043,
                    30374,
                    30024,
                ]
            ],
            dtype="int64",
        )
        self.position_ids = paddle.to_tensor(
            [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]],
            dtype="int64",
        )
        self.attention_mask = paddle.to_tensor(
            [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype="int64"
        )

    def check_jit_kernel_info(self, static_fn):
        utils.check_jit_kernel_number(static_fn, 1)
        utils.check_jit_kernel_structure(static_fn, {utils.JIT_KERNEL_NAME: 1})

    def eval(self, use_cinn):
        paddle.seed(2024)
        net = llama_test_model.LlamaModel(self.config)
        input_spec = [
            InputSpec(shape=[None, None], dtype='int64'),  # input_ids
            InputSpec(shape=[None, None], dtype='int64'),  # position_ids
            InputSpec(shape=[None, None], dtype='int64'),  # attention_mask
        ]
        net = utils.apply_to_static(net, use_cinn, input_spec)
        net.eval()
        out = net(self.input_ids, self.position_ids, self.attention_mask)
        return out

    def test_eval(self):
        dy_out = self.eval(use_cinn=False)
        cinn_out = self.eval(use_cinn=True)
        np.testing.assert_allclose(
            cinn_out.numpy(), dy_out.numpy(), atol=1e-4, rtol=1e-5
        )


if __name__ == '__main__':
    unittest.main()
