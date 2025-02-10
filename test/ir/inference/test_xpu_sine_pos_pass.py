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
from functools import partial

import hypothesis.strategies as st
import numpy as np
from auto_scan_test import PassAutoScanTest
from program_config import OpConfig, ProgramConfig, TensorConfig

from paddle.base import core


@unittest.skipIf(
    core.get_xpu_device_version(0) == core.XPUVersion.XPU3,
    "Unsupported on XPU3",
)
class TestSinePosXPUFusePass(PassAutoScanTest):
    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(use_xpu=True)
        yield config, ["sine_pos_xpu"], (1e-3, 1e-3)

    def sample_program_config(self, draw):
        x_shape = draw(
            st.lists(
                st.integers(min_value=1, max_value=10), min_size=3, max_size=3
            )
        )
        x_shape[1] = draw(st.integers(min_value=100, max_value=512))
        x_shape[2] = draw(st.integers(min_value=1, max_value=1))
        y_shape = draw(
            st.lists(
                st.integers(min_value=128, max_value=128),
                min_size=1,
                max_size=1,
            )
        )

        def generate_data(shape):
            return np.random.random(shape).astype(np.float32)

        # Here we will compose a program
        # Still has some risks that the program is invalid or cause bug while running
        # Use function `is_program_valid` to filter the invalid programs before running
        # Use function `add_skip_pass_case` to ignore the programs even if they cause bug while running
        mul_op = OpConfig(
            "elementwise_mul",
            inputs={"X": ["x"], "Y": ["y"]},
            outputs={"Out": ["mul_out"]},
            axis=-1,
        )
        slice1_op = OpConfig(
            "strided_slice",
            inputs={"Input": ["mul_out"]},
            outputs={"Out": ["slice1_out"]},
            axes=[2],
            starts=[0],
            strides=[2],
            ends=[128],
            infer_flags=[1],
        )
        sin_op = OpConfig(
            "sin",
            inputs={"X": ["slice1_out"]},
            outputs={"Out": ["sin_out"]},
        )
        slice2_op = OpConfig(
            "strided_slice",
            inputs={"Input": ["mul_out"]},
            outputs={"Out": ["slice2_out"]},
            axes=[2],
            starts=[1],
            strides=[2],
            ends=[128],
            infer_flags=[1],
        )
        cos_op = OpConfig(
            "cos",
            inputs={"X": ["slice2_out"]},
            outputs={"Out": ["cos_out"]},
        )
        stack_op = OpConfig(
            "stack",
            inputs={"X": ["sin_out", "cos_out"]},
            outputs={"Y": ["stack_out"]},
            axis=3,
        )
        flatten_op = OpConfig(
            "flatten_contiguous_range",
            inputs={"X": ["stack_out"]},
            outputs={"Out": ["flatten_out"]},
            start_axis=2,
            stop_axis=3,
        )

        ops = [
            mul_op,
            slice1_op,
            slice2_op,
            sin_op,
            cos_op,
            stack_op,
            flatten_op,
        ]

        program_config = ProgramConfig(
            ops=ops,
            inputs={
                "x": TensorConfig(data_gen=partial(generate_data, x_shape)),
                "y": TensorConfig(data_gen=partial(generate_data, y_shape)),
            },
            weights={},
            outputs=ops[-1].outputs["Out"],
        )
        return program_config

    def test(self):
        self.run_and_statis(
            quant=False,
            max_examples=25,
            passes=["sine_pos_fuse_pass"],
        )


if __name__ == "__main__":
    unittest.main()
