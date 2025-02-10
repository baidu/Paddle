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

import math
import unittest

import hypothesis.strategies as st
import numpy as np
from auto_scan_test import PassAutoScanTest
from program_config import OpConfig, ProgramConfig, TensorConfig

from paddle.base import core


@unittest.skipIf(
    core.get_xpu_device_version(0) == core.XPUVersion.XPU3,
    "Unsupported on XPU3",
)
class TestCrossAttentionXPUFusePass(PassAutoScanTest):
    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(use_xpu=True)
        yield config, ["cross_attention_xpu"], (1e-1, 1e-1)

    def sample_program_config(self, draw):
        # Here we will compose a program
        # Still has some risks that the program is invalid or cause bug while running
        # Use function `is_program_valid` to filter the invalid programs before running
        # Use function `add_skip_pass_case` to ignore the programs even if they cause bug while running

        # q: matmul + add + reshape + transpose + scale
        q_mul_op = OpConfig(
            "matmul_v2",
            inputs={"X": ["input_q"], "Y": ["q_mul_w"]},
            outputs={"Out": ["q_mul_out"]},
            trans_x=False,
            trans_y=False,
        )
        q_add_op = OpConfig(
            "elementwise_add",
            inputs={"X": ["q_mul_out"], "Y": ["q_add_bias"]},
            outputs={"Out": ["q_add_out"]},
            axis=-1,
        )
        q_reshape_op = OpConfig(
            "reshape2",
            inputs={"X": ["q_add_out"]},
            outputs={"Out": ["q_reshape_out"], "XShape": ["q_reshape_xshape"]},
            shape=[0, 0, 4, 32],
        )
        q_transpose_op = OpConfig(
            "transpose2",
            inputs={"X": ["q_reshape_out"]},
            outputs={
                "Out": ["q_transpose_out"],
                "XShape": ["q_transpose_xshape"],
            },
            axis=[0, 2, 1, 3],
        )
        q_scale_op = OpConfig(
            "scale",
            inputs={"X": ["q_transpose_out"]},
            outputs={"Out": ["q_scale_out"]},
            scale=1.0 / math.sqrt(32.0),
            bias=0,
            bias_after_scale=True,
        )
        # k: matmul + add + reshape + transpose
        k_mul_op = OpConfig(
            "matmul_v2",
            inputs={"X": ["input_kv"], "Y": ["k_mul_w"]},
            outputs={"Out": ["k_mul_out"]},
            trans_x=False,
            trans_y=False,
        )
        k_add_op = OpConfig(
            "elementwise_add",
            inputs={"X": ["k_mul_out"], "Y": ["k_add_bias"]},
            outputs={"Out": ["k_add_out"]},
            axis=-1,
        )
        k_reshape_op = OpConfig(
            "reshape2",
            inputs={"X": ["k_add_out"]},
            outputs={"Out": ["k_reshape_out"], "XShape": ["k_reshape_xshape"]},
            shape=[0, 0, 4, 32],
        )
        k_transpose_op = OpConfig(
            "transpose2",
            inputs={"X": ["k_reshape_out"]},
            outputs={
                "Out": ["k_transpose_out"],
                "XShape": ["k_transpose_xshape"],
            },
            axis=[0, 2, 1, 3],
        )
        # v: matmul + add + reshape + transpose
        v_mul_op = OpConfig(
            "matmul_v2",
            inputs={"X": ["input_kv"], "Y": ["v_mul_w"]},
            outputs={"Out": ["v_mul_out"]},
            trans_x=False,
            trans_y=False,
        )
        v_add_op = OpConfig(
            "elementwise_add",
            inputs={"X": ["v_mul_out"], "Y": ["v_add_bias"]},
            outputs={"Out": ["v_add_out"]},
            axis=-1,
        )
        v_reshape_op = OpConfig(
            "reshape2",
            inputs={"X": ["v_add_out"]},
            outputs={"Out": ["v_reshape_out"], "XShape": ["v_reshape_xshape"]},
            shape=[0, 0, 4, 32],
        )
        v_transpose_op = OpConfig(
            "transpose2",
            inputs={"X": ["v_reshape_out"]},
            outputs={
                "Out": ["v_transpose_out"],
                "XShape": ["v_transpose_xshape"],
            },
            axis=[0, 2, 1, 3],
        )
        # qk_matmul + add + softmax
        qk_matmul_op = OpConfig(
            "matmul_v2",
            inputs={"X": ["q_scale_out"], "Y": ["k_transpose_out"]},
            outputs={"Out": ["qk_matmul_out"]},
            trans_x=False,
            trans_y=True,
        )
        qk_add_op = OpConfig(
            "elementwise_add",
            inputs={"X": ["qk_matmul_out"], "Y": ["qk_add_mask"]},
            outputs={"Out": ["qk_add_out"]},
            axis=-1,
        )
        qk_softmax_op = OpConfig(
            "softmax",
            inputs={"X": ["qk_add_out"]},
            outputs={"Out": ["qk_softmax_out"]},
            axis=-1,
        )
        # qkv_malmul + transpose + reshape
        qkv_matmul_op = OpConfig(
            "matmul_v2",
            inputs={"X": ["qk_softmax_out"], "Y": ["v_transpose_out"]},
            outputs={"Out": ["qkv_matmul_out"]},
            trans_x=False,
            trans_y=False,
        )
        qkv_transpose_op = OpConfig(
            "transpose2",
            inputs={"X": ["qkv_matmul_out"]},
            outputs={
                "Out": ["qkv_transpose_out"],
                "XShape": ["qkv_transpose_xshape"],
            },
            axis=[0, 2, 1, 3],
        )
        qkv_reshape_op = OpConfig(
            "reshape2",
            inputs={"X": ["qkv_transpose_out"]},
            outputs={
                "Out": ["qkv_reshape_out"],
                "XShape": ["qkv_reshape_xshape"],
            },
            shape=[0, 0, 128],
        )

        ops = [
            q_mul_op,
            q_add_op,
            q_reshape_op,
            q_transpose_op,
            q_scale_op,
            k_mul_op,
            k_add_op,
            k_reshape_op,
            k_transpose_op,
            v_mul_op,
            v_add_op,
            v_reshape_op,
            v_transpose_op,
            qk_matmul_op,
            qk_add_op,
            qk_softmax_op,
            qkv_matmul_op,
            qkv_transpose_op,
            qkv_reshape_op,
        ]

        # set input shape
        batch_size = draw(st.integers(min_value=1, max_value=10))
        q_seqlen = draw(st.integers(min_value=1, max_value=128))
        kv_seqlen = draw(st.integers(min_value=1, max_value=256))
        # batch_size = 1
        # q_seqlen = 2
        # kv_seqlen = 62
        hidden_dim = 128

        input_q_shape = [batch_size, q_seqlen, hidden_dim]
        input_kv_shape = [batch_size, kv_seqlen, hidden_dim]
        q_mul_w_shape = [input_q_shape[2], input_q_shape[2]]
        k_mul_w_shape = [input_kv_shape[2], input_kv_shape[2]]
        v_mul_w_shape = [input_kv_shape[2], input_kv_shape[2]]
        q_add_bias_shape = [input_q_shape[2]]
        qk_add_mask_shape = [q_seqlen, kv_seqlen]

        program_config = ProgramConfig(
            ops=ops,
            inputs={
                "input_q": TensorConfig(shape=input_q_shape),
                "input_kv": TensorConfig(shape=input_kv_shape),
                "qk_add_mask": TensorConfig(shape=qk_add_mask_shape),
            },
            weights={
                "q_mul_w": TensorConfig(shape=q_mul_w_shape),
                "k_mul_w": TensorConfig(shape=k_mul_w_shape),
                "v_mul_w": TensorConfig(shape=v_mul_w_shape),
                "q_add_bias": TensorConfig(shape=q_add_bias_shape),
                "k_add_bias": TensorConfig(shape=q_add_bias_shape),
                "v_add_bias": TensorConfig(shape=q_add_bias_shape),
            },
            outputs=["qkv_reshape_out"],
        )
        return program_config

    def test(self):
        self.run_and_statis(
            quant=False,
            max_examples=2,
            min_success_num=2,
            passes=["cross_attention_xpu_fuse_pass"],
        )


if __name__ == "__main__":
    np.random.seed(200)
    unittest.main()
