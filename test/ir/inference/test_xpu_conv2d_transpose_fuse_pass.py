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


class TestFcXPUFusePass(PassAutoScanTest):
    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(use_xpu=True)
        yield config, ["conv2d_transpose_xpu"], (1e-3, 1e-3)

    def sample_program_config(self, draw):
        x_shape = draw(
            st.lists(
                st.integers(min_value=2, max_value=8), min_size=4, max_size=4
            )
        )
        y_shape = x_shape[1]
        weight_shape = [10, x_shape[1], 4, 4]
        has_bn = draw(st.booleans())
        has_add = draw(st.booleans())
        has_relu = draw(st.booleans())
        deconv_op = OpConfig(
            "conv2d_transpose",
            inputs={"Input": ["input_x"]},
            outputs={"Output": ["output_x"]},
            data_format="NCHW",
            dilations=[1, 1],
            groups=1,
            paddings=[0, 0],
            padding_algorithm="EXPLICIT",
            strides=[4, 4],
        )
        add_op = OpConfig(
            "elementwise_add",
            inputs={"X": ["output_x"], "Y": ["bias"]},
            outputs={"Out": ["add_out"]},
            axis=1,
        )
        bn_op = OpConfig(
            "batch_norm",
            inputs={
                "X": ["add_out"],
                "Bias": ["bn_bias"],
                "Mean": ["bn_mean"],
                "Scale": ["bn_scale"],
                "Variance": ["bn_var"],
            },
            outputs={
                "Y": ["bn_y"],
                "MeanOut": ["bn_mean_out"],
                "SavedMean": ["bn_mean_save"],
                "SavedVariance": ["bn_save_var"],
                "VarianceOut": ["var_out"],
            },
            data_layout="NCHW",
            epsilon=0.000009999999747378752,
        )
        relu_op = OpConfig(
            "relu", inputs={"X": ["bn_y"]}, outputs={"Out": ["relu_out"]}
        )
        ops = [deconv_op, bn_op, add_op, relu_op]
        program_config = ProgramConfig(
            ops=ops,
            weights={
                "Filter": TensorConfig(shape=weight_shape),
                "bias": TensorConfig(shape=y_shape),
            },
            inputs={
                "input_x": TensorConfig(shape=x_shape),
                "bn_bias": TensorConfig(shape=y_shape),
                "bn_mean": TensorConfig(shape=y_shape),
                "bn_scale": TensorConfig(shape=y_shape),
                "bn_var": TensorConfig(shape=y_shape),
            },
            outputs=ops[-1].outputs["Out"],
        )
        return program_config

    def test(self):
        self.run_and_statis(
            quant=False,
            max_examples=25,
            passes=["conv2d_transpose_xpu_fuse_pass"],
        )


if __name__ == "__main__":
    unittest.main()
