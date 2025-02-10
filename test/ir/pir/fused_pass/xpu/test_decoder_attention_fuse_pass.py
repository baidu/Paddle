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

import math
import unittest

import numpy as np
from pass_test import PassTest

import paddle
import paddle.nn.functional as F
from paddle.base import core

paddle.enable_static()


@unittest.skipIf(
    core.get_xpu_device_version(0) == core.XPUVersion.XPU3,
    "do not supported on XPU3",
)
class TestDecoderAttentionFusePattern(PassTest):
    r"""
        q             k             v
        |             |             |
     reshape       reshape       reshape
        |             |             |
    transpose     transpose     transpose
        |             |           |
        |                \     /
        |              qk_matmul
        |                   |
        |                   |
        |                 scale
        |                   |
         \              qk_softmax
           \                 |
             \             /
               \          /
                 qkv_matmul
                    |
                transpose
                    |
                 reshape
                    |
                   out
    """

    def is_program_valid(self, program):
        return True

    def build_ir_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                # set input shape
                batch_size = 1  # paddle.randint(1, 50).item()
                seqlen = 1  # paddle.randint(100, 2000).item()
                input_shape = [batch_size, seqlen, 256]

                input_q = paddle.static.data(
                    name='input_q', shape=input_shape, dtype='float32'
                )
                input_k = paddle.static.data(
                    name='input_k', shape=input_shape, dtype='float32'
                )
                input_v = paddle.static.data(
                    name='input_v', shape=input_shape, dtype='float32'
                )

                reshape_q = paddle.reshape(input_q, [0, 0, 8, 32])
                reshape_k = paddle.reshape(input_k, [0, 0, 8, 32])
                reshape_v = paddle.reshape(input_v, [0, 0, 8, 32])

                trans_q = paddle.transpose(reshape_q, [0, 2, 1, 3])
                trans_k = paddle.transpose(reshape_k, [0, 2, 1, 3])
                trans_v = paddle.transpose(reshape_v, [0, 2, 1, 3])

                matmul_qk = paddle.matmul(
                    trans_q, trans_k, transpose_x=False, transpose_y=True
                )
                scale_qk = paddle.scale(
                    matmul_qk, scale=1 / math.sqrt(32), bias=0.0
                )
                softmax_qk = F.softmax(scale_qk, axis=-1, dtype="float32")

                matmul_qkv = paddle.matmul(softmax_qk, trans_v)
                trans_qkv = paddle.transpose(matmul_qkv, [0, 2, 1, 3])
                out = paddle.reshape(trans_qkv, [batch_size, seqlen, 256])
                out = paddle.assign(out)

                self.feeds = {
                    "input_q": np.random.random(
                        (batch_size, seqlen, 256)
                    ).astype("float32"),
                    "input_k": np.random.random(
                        (batch_size, seqlen, 256)
                    ).astype("float32"),
                    "input_v": np.random.random(
                        (batch_size, seqlen, 256)
                    ).astype("float32"),
                }
                self.fetch_list = [out]
                return [main_prog, start_prog]

    def setUp(self):
        if core.is_compiled_with_xpu():
            self.places.append(paddle.XPUPlace(0))
        self.skip_accuracy_verification = True

    def sample_program(self):
        yield self.build_ir_program(), False

    def test_check_output(self):
        self.check_pass_correct()


if __name__ == "__main__":
    unittest.main()
