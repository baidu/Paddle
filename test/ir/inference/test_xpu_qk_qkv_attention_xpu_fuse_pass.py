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

import unittest

import hypothesis.strategies as st
from auto_scan_test import PassAutoScanTest
from program_config import OpConfig, ProgramConfig, TensorConfig


class TestGatherAddTransposePass1(PassAutoScanTest):
    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(use_xpu=True)
        yield config, ["qkv_attention_xpu"], (1e-1, 1e-1)

    def sample_program_config(self, draw):
        # set input shape
        batch_size = draw(st.integers(min_value=1, max_value=50))
        input_shape = [batch_size, 100, 576]

        # Here we will compose a program
        # Still has some risks that the program is invalid or cause bug while running
        # Use function `is_program_valid` to filter the invalid programs before running
        # Use function `add_skip_pass_case` to ignore the programs even if they cause bug while running
        reshape_1_op = OpConfig(
            "reshape2",
            inputs={"X": ["input"]},
            outputs={"Out": ["reshape_1_out"], "XShape": ["reshape_1_xshape"]},
            shape=[-1, 100, 3, 12, 16],
        )
        transpose2_1_op = OpConfig(
            "transpose2",
            inputs={"X": ["reshape_1_out"]},
            outputs={
                "Out": ["transpose2_1_out"],
                "XShape": ["transpose2_1_xshape"],
            },
            axis=[2, 0, 3, 1, 4],
        )
        slice_1_op = OpConfig(
            "slice",
            inputs={
                "Input": ["transpose2_1_out"],
            },
            starts=[0],
            ends=[1],
            axes=[0],
            decrease_axis=[0],
            outputs={"Out": ["slice_1_out"]},
        )
        slice_2_op = OpConfig(
            "slice",
            inputs={
                "Input": ["transpose2_1_out"],
            },
            starts=[1],
            ends=[2],
            axes=[0],
            decrease_axis=[0],
            outputs={"Out": ["slice_2_out"]},
        )
        slice_3_op = OpConfig(
            "slice",
            inputs={
                "Input": ["transpose2_1_out"],
            },
            starts=[2],
            ends=[3],
            axes=[0],
            decrease_axis=[0],
            outputs={"Out": ["slice_3_out"]},
        )
        transpose2_2_op = OpConfig(
            "transpose2",
            inputs={"X": ["slice_2_out"]},
            outputs={
                "Out": ["transpose2_2_out"],
                "XShape": ["transpose2_2_xshape"],
            },
            axis=[0, 1, 3, 2],
        )
        scale_op = OpConfig(
            "scale",
            inputs={"X": ["slice_1_out"]},
            outputs={"Out": ["scale_out"]},
            scale=0.25,
            bias=0,
            bias_after_scale=True,
        )
        qk_matmul_op = OpConfig(
            "matmul_v2",
            inputs={"X": ["scale_out"], "Y": ["transpose2_2_out"]},
            outputs={"Out": ["qk_matmul_out"]},
            trans_x=False,
            trans_y=False,
        )
        qk_softmax_op = OpConfig(
            "softmax",
            inputs={"X": ["qk_matmul_out"]},
            outputs={"Out": ["qk_softmax_out"]},
            axis=-1,
        )
        qkv_matmul_op = OpConfig(
            "matmul_v2",
            inputs={"X": ["qk_softmax_out"], "Y": ["slice_3_out"]},
            outputs={"Out": ["qkv_matmul_out"]},
            trans_x=False,
            trans_y=False,
        )
        transpose2_3_op = OpConfig(
            "transpose2",
            inputs={"X": ["qkv_matmul_out"]},
            outputs={
                "Out": ["transpose2_3_out"],
                "XShape": ["transpose2_3_xshape"],
            },
            axis=[0, 2, 1, 3],
        )
        reshape_2_op = OpConfig(
            "reshape2",
            inputs={"X": ["transpose2_3_out"]},
            outputs={"Out": ["output"], "XShape": ["reshape_2_xshape"]},
            shape=[-1, 100, 192],
        )

        ops = [
            reshape_1_op,
            transpose2_1_op,
            slice_1_op,
            slice_2_op,
            slice_3_op,
            transpose2_2_op,
            scale_op,
            qk_matmul_op,
            qk_softmax_op,
            qkv_matmul_op,
            transpose2_3_op,
            reshape_2_op,
        ]

        program_config = ProgramConfig(
            ops=ops,
            inputs={
                "input": TensorConfig(shape=input_shape),
            },
            weights={},
            outputs=["output"],
        )
        return program_config

    def test(self):
        self.run_and_statis(
            quant=False,
            max_examples=25,
            passes=["qk_qkv_attention_xpu_fuse_pass"],
        )


class TestGatherAddTransposePass2(PassAutoScanTest):
    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(use_xpu=True)
        yield config, ["qkv_attention_xpu"], (1e-1, 1e-1)

    def sample_program_config(self, draw):
        # set input shape
        batch_size = draw(st.integers(min_value=1, max_value=50))
        input_shape = [batch_size, 100, 576]

        # Here we will compose a program
        # Still has some risks that the program is invalid or cause bug while running
        # Use function `is_program_valid` to filter the invalid programs before running
        # Use function `add_skip_pass_case` to ignore the programs even if they cause bug while running
        reshape_1_op = OpConfig(
            "reshape2",
            inputs={"X": ["input"]},
            outputs={"Out": ["reshape_1_out"], "XShape": ["reshape_1_xshape"]},
            shape=[-1, 100, 3, 12, 16],
        )
        transpose2_1_op = OpConfig(
            "transpose2",
            inputs={"X": ["reshape_1_out"]},
            outputs={
                "Out": ["transpose2_1_out"],
                "XShape": ["transpose2_1_xshape"],
            },
            axis=[2, 0, 3, 1, 4],
        )
        slice_1_op = OpConfig(
            "slice",
            inputs={
                "Input": ["transpose2_1_out"],
            },
            starts=[0],
            ends=[1],
            axes=[0],
            decrease_axis=[0],
            outputs={"Out": ["slice_1_out"]},
        )
        slice_2_op = OpConfig(
            "slice",
            inputs={
                "Input": ["transpose2_1_out"],
            },
            starts=[1],
            ends=[2],
            axes=[0],
            decrease_axis=[0],
            outputs={"Out": ["slice_2_out"]},
        )
        slice_3_op = OpConfig(
            "slice",
            inputs={
                "Input": ["transpose2_1_out"],
            },
            starts=[2],
            ends=[3],
            axes=[0],
            decrease_axis=[0],
            outputs={"Out": ["slice_3_out"]},
        )
        transpose2_2_op = OpConfig(
            "transpose2",
            inputs={"X": ["slice_2_out"]},
            outputs={
                "Out": ["transpose2_2_out"],
                "XShape": ["transpose2_2_xshape"],
            },
            axis=[0, 1, 3, 2],
        )
        qk_matmul_op = OpConfig(
            "matmul_v2",
            inputs={"X": ["slice_1_out"], "Y": ["transpose2_2_out"]},
            outputs={"Out": ["qk_matmul_out"]},
            trans_x=False,
            trans_y=False,
        )
        scale_op = OpConfig(
            "scale",
            inputs={"X": ["qk_matmul_out"]},
            outputs={"Out": ["scale_out"]},
            scale=0.25,
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
            inputs={"X": ["qk_softmax_out"], "Y": ["slice_3_out"]},
            outputs={"Out": ["qkv_matmul_out"]},
            trans_x=False,
            trans_y=False,
        )
        transpose2_3_op = OpConfig(
            "transpose2",
            inputs={"X": ["qkv_matmul_out"]},
            outputs={
                "Out": ["transpose2_3_out"],
                "XShape": ["transpose2_3_xshape"],
            },
            axis=[0, 2, 1, 3],
        )
        reshape_2_op = OpConfig(
            "reshape2",
            inputs={"X": ["transpose2_3_out"]},
            outputs={"Out": ["output"], "XShape": ["reshape_2_xshape"]},
            shape=[-1, 100, 192],
        )

        ops = [
            reshape_1_op,
            transpose2_1_op,
            slice_1_op,
            slice_2_op,
            slice_3_op,
            transpose2_2_op,
            qk_matmul_op,
            scale_op,
            qk_softmax_op,
            qkv_matmul_op,
            transpose2_3_op,
            reshape_2_op,
        ]

        program_config = ProgramConfig(
            ops=ops,
            inputs={
                "input": TensorConfig(shape=input_shape),
            },
            weights={},
            outputs=["output"],
        )
        return program_config

    def test(self):
        self.run_and_statis(
            quant=False,
            max_examples=25,
            passes=["qk_qkv_attention_xpu_fuse_pass"],
        )


if __name__ == "__main__":
    unittest.main()
