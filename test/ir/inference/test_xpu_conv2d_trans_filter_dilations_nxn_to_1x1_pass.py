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


class TestConv2dTransFilterDilationsNxNTo1x1PassPass(PassAutoScanTest):
    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(use_xpu=True)
        yield config, ["conv2d"], (1e-3, 1e-3)

    def is_program_valid(self, prog_config):
        paddings = prog_config.ops[0].attrs["paddings"]
        strides = prog_config.ops[0].attrs["strides"]
        groups = prog_config.ops[0].attrs["groups"]
        padding_algorithm = prog_config.ops[0].attrs["padding_algorithm"]
        dilations = prog_config.ops[0].attrs["dilations"]
        data_format = prog_config.ops[0].attrs["data_format"]
        filter_shape = prog_config.weights["conv2d_weight"].shape
        input_shape = prog_config.inputs["conv2d_input"].shape
        if data_format != "NCHW":
            return False
        if padding_algorithm == "VALID":
            if (
                (input_shape[2] - (dilations[0] * (filter_shape[2] - 1) + 1))
                / strides[0]
                + 1
            ) < 1 or (
                (input_shape[3] - (dilations[1] * (filter_shape[3] - 1) + 1))
                / strides[1]
                + 1
            ) < 1:
                return False
        if padding_algorithm == "EXPLICIT":
            if (
                (
                    input_shape[2]
                    + 2 * paddings[0]
                    - (dilations[0] * (filter_shape[2] - 1) + 1)
                )
                / strides[0]
                + 1
            ) < 1 or (
                (
                    input_shape[3]
                    + 2 * paddings[1]
                    - (dilations[1] * (filter_shape[3] - 1) + 1)
                )
                / strides[1]
                + 1
            ) < 1:
                return False
        if data_format == "NCHW":
            if input_shape[1] != filter_shape[1] * groups:
                return False
            if filter_shape[0] % groups != 0:
                return False
        return True

    def sample_program_config(self, draw):
        data_format = draw(st.sampled_from(["NCHW"]))

        x_shape = draw(
            st.lists(
                st.integers(min_value=12, max_value=12), min_size=4, max_size=4
            )
        )
        x_shape[1] = draw(st.integers(min_value=1, max_value=10))

        # 3. Generate legal shape of input:Y of conv2d
        w_shape = draw(
            st.lists(
                st.integers(min_value=3, max_value=3), min_size=4, max_size=4
            )
        )
        if data_format == "NCHW":
            w_shape[1] = x_shape[1]

        padding_algorithm = draw(st.sampled_from(["EXPLICIT", "VALID"]))

        groups = draw(st.integers(min_value=1, max_value=1))

        paddings = draw(
            st.lists(
                st.integers(min_value=1, max_value=1), min_size=2, max_size=2
            )
        )
        strides = draw(
            st.lists(
                st.integers(min_value=1, max_value=1), min_size=2, max_size=2
            )
        )
        dilations = draw(
            st.lists(
                st.integers(min_value=1, max_value=5), min_size=2, max_size=2
            )
        )

        def generate_data(shape):
            return np.random.random(shape).astype(np.float32)

        # Here we will compose a program
        # Still has some risks that the program is invalid or cause bug while running
        # Use function `is_program_valid` to filter the invalid programs before running
        # Use function `add_skip_pass_case` to ignore the programs even if they cause bug while running
        conv2d_op = OpConfig(
            "conv2d",
            inputs={
                "Input": ["conv2d_input"],
                "Filter": ["conv2d_weight"],
            },
            outputs={"Output": ["conv2d_out"]},
            data_format=data_format,
            dilations=dilations,
            padding_algorithm=padding_algorithm,
            groups=groups,
            paddings=paddings,
            strides=strides,
            has_bias=False,
        )

        program_config = ProgramConfig(
            ops=[conv2d_op],
            inputs={
                "conv2d_input": TensorConfig(
                    data_gen=partial(generate_data, x_shape)
                ),
            },
            weights={
                "conv2d_weight": TensorConfig(
                    data_gen=partial(generate_data, w_shape)
                ),
            },
            outputs=["conv2d_out"],
        )
        return program_config

    def test(self):
        self.run_and_statis(
            quant=False,
            max_examples=25,
            passes=["conv2d_trans_filter_dilations_nxn_to_1x1_pass"],
        )


if __name__ == "__main__":
    unittest.main()
