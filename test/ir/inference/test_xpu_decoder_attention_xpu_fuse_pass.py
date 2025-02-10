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

import hypothesis.strategies as st
from auto_scan_test import PassAutoScanTest
from program_config import OpConfig, ProgramConfig, TensorConfig

from paddle.base import core


@unittest.skipIf(
    core.get_xpu_device_version(0) == core.XPUVersion.XPU3,
    "only supported on XPU3",
)
class TestDecoderAttentionXPUFusePass(PassAutoScanTest):
    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(use_xpu=True)
        yield config, ["qkv_attention_xpu"], (1e-1, 1e-1)

    def sample_program_config(self, draw):
        # set input shape
        batch_size = draw(st.integers(min_value=1, max_value=50))
        seqlen = draw(st.integers(min_value=100, max_value=2000))
        input_shape = [batch_size, seqlen, 256]

        # Here we will compose a program
        # Still has some risks that the program is invalid or cause bug while running
        # Use function `is_program_valid` to filter the invalid programs before running
        # Use function `add_skip_pass_case` to ignore the programs even if they cause bug while running
        reshape2_1_op = OpConfig(
            "reshape2",
            inputs={"X": ["input_q"]},
            outputs={
                "Out": ["reshape2_1_out"],
                "XShape": ["reshape2_1_xshape"],
            },
            shape=[0, 0, 8, 32],
        )
        reshape2_2_op = OpConfig(
            "reshape2",
            inputs={"X": ["input_k"]},
            outputs={
                "Out": ["reshape2_2_out"],
                "XShape": ["reshape2_2_xshape"],
            },
            shape=[0, 0, 8, 32],
        )
        reshape2_3_op = OpConfig(
            "reshape2",
            inputs={"X": ["input_v"]},
            outputs={
                "Out": ["reshape2_3_out"],
                "XShape": ["reshape2_3_xshape"],
            },
            shape=[0, 0, 8, 32],
        )
        transpose2_1_op = OpConfig(
            "transpose2",
            inputs={"X": ["reshape2_1_out"]},
            outputs={
                "Out": ["transpose2_1_out"],
                "XShape": ["transpose2_1_xshape"],
            },
            axis=[0, 2, 1, 3],
        )
        transpose2_2_op = OpConfig(
            "transpose2",
            inputs={"X": ["reshape2_2_out"]},
            outputs={
                "Out": ["transpose2_2_out"],
                "XShape": ["transpose2_2_xshape"],
            },
            axis=[0, 2, 1, 3],
        )
        transpose2_3_op = OpConfig(
            "transpose2",
            inputs={"X": ["reshape2_3_out"]},
            outputs={
                "Out": ["transpose2_3_out"],
                "XShape": ["transpose2_3_xshape"],
            },
            axis=[0, 2, 1, 3],
        )
        qk_matmul_op = OpConfig(
            "matmul_v2",
            inputs={"X": ["transpose2_1_out"], "Y": ["transpose2_2_out"]},
            outputs={"Out": ["qk_matmul_out"]},
            trans_x=False,
            trans_y=True,
        )
        scale_op = OpConfig(
            "scale",
            inputs={"X": ["qk_matmul_out"]},
            outputs={"Out": ["scale_out"]},
            scale=1 / math.sqrt(32),
            bias=0,
            bias_after_scale=True,
        )
        qk_softmax_op = OpConfig(
            "softmax",
            inputs={"X": ["scale_out"]},
            outputs={"Out": ["qk_softmax_out"]},
            axis=-1,
        )
        qkv_matmul_op = OpConfig(
            "matmul_v2",
            inputs={"X": ["qk_softmax_out"], "Y": ["transpose2_3_out"]},
            outputs={"Out": ["qkv_matmul_out"]},
            trans_x=False,
            trans_y=False,
        )
        transpose2_4_op = OpConfig(
            "transpose2",
            inputs={"X": ["qkv_matmul_out"]},
            outputs={
                "Out": ["transpose2_4_out"],
                "XShape": ["transpose2_4_xshape"],
            },
            axis=[0, 2, 1, 3],
        )
        reshape2_4_op = OpConfig(
            "reshape2",
            inputs={"X": ["transpose2_4_out"]},
            outputs={"Out": ["output"], "XShape": ["reshape2_4_xshape"]},
            shape=[0, 0, 256],
        )

        ops = [
            reshape2_1_op,
            reshape2_2_op,
            reshape2_3_op,
            transpose2_1_op,
            transpose2_2_op,
            transpose2_3_op,
            qk_matmul_op,
            scale_op,
            qk_softmax_op,
            qkv_matmul_op,
            transpose2_4_op,
            reshape2_4_op,
        ]

        program_config = ProgramConfig(
            ops=ops,
            inputs={
                "input_q": TensorConfig(shape=input_shape),
                "input_k": TensorConfig(shape=input_shape),
                "input_v": TensorConfig(shape=input_shape),
            },
            weights={},
            outputs=["output"],
        )
        return program_config

    def test(self):
        self.run_and_statis(
            quant=False,
            max_examples=25,
            passes=["decoder_attention_xpu_fuse_pass"],
        )


if __name__ == "__main__":
    unittest.main()
