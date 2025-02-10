# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import annotations

import itertools
import unittest
from functools import partial
from typing import Any

import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import SkipReasons, TrtLayerAutoScanTest

import paddle.inference as paddle_infer


class TrtConvertDepthwiseConv2dTest(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        inputs = program_config.inputs
        weights = program_config.weights
        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]

        if (
            inputs['input_data'].shape[1]
            != weights['conv2d_weight'].shape[1] * attrs[0]['groups']
        ):
            return False

        return True

    def sample_program_configs(self):
        self.trt_param.workspace_size = 1073741824

        def generate_input1(batch, attrs: list[dict[str, Any]]):
            groups = attrs[0]['groups']
            return np.ones([batch, groups, 64, 64]).astype(np.float32)

        def generate_weight1(attrs: list[dict[str, Any]]):
            return np.random.random([24, 1, 3, 3]).astype(np.float32)

        batch_options = [1]
        strides_options = [[1, 2]]
        paddings_options = [[0, 3]]
        groups_options = [1]
        padding_algorithm_options = ['EXPLICIT', 'SAME', 'VALID']
        dilations_options = [[1, 1]]
        data_format_options = ['NCHW']

        configurations = [
            batch_options,
            strides_options,
            paddings_options,
            groups_options,
            padding_algorithm_options,
            dilations_options,
            data_format_options,
        ]

        for (
            batch,
            strides,
            paddings,
            groups,
            padding_algorithm,
            dilations,
            data_format,
        ) in itertools.product(*configurations):
            attrs = [
                {
                    "strides": strides,
                    "paddings": paddings,
                    "groups": groups,
                    "padding_algorithm": padding_algorithm,
                    "dilations": dilations,
                    "data_format": data_format,
                }
            ]

            ops_config = [
                {
                    "op_type": "depthwise_conv2d",
                    "op_inputs": {
                        "Input": ["input_data"],
                        "Filter": ["conv2d_weight"],
                    },
                    "op_outputs": {"Output": ["output_data"]},
                    "op_attrs": attrs[0],
                }
            ]
            ops = self.generate_op_config(ops_config)

            program_config = ProgramConfig(
                ops=ops,
                weights={
                    "conv2d_weight": TensorConfig(
                        data_gen=partial(generate_weight1, attrs)
                    )
                },
                inputs={
                    "input_data": TensorConfig(
                        data_gen=partial(generate_input1, batch, attrs)
                    )
                },
                outputs=["output_data"],
            )

            yield program_config

    def sample_predictor_configs(
        self, program_config
    ) -> tuple[paddle_infer.Config, list[int], float]:
        def generate_dynamic_shape(attrs):
            groups = attrs[0]['groups']
            self.dynamic_shape.min_input_shape = {
                "input_data": [1, groups, 32, 32],
                "output_data": [1, 24, 32, 32],
            }
            self.dynamic_shape.max_input_shape = {
                "input_data": [4, groups, 64, 64],
                "output_data": [4, 24, 64, 64],
            }
            self.dynamic_shape.opt_input_shape = {
                "input_data": [1, groups, 64, 64],
                "output_data": [1, 24, 64, 64],
            }

        def clear_dynamic_shape():
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num():
            return 1, 2

        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]

        # for static_shape
        clear_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield self.create_inference_config(), generate_trt_nodes_num(), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield self.create_inference_config(), generate_trt_nodes_num(), (
            5e-3,
            1e-3,
        )
        self.trt_param.precision = paddle_infer.PrecisionType.Int8
        program_config.set_input_type(np.float32)
        yield self.create_inference_config(), generate_trt_nodes_num(), (
            1e-3,
            1e-3,
        )

        # for dynamic_shape
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield self.create_inference_config(), generate_trt_nodes_num(), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield self.create_inference_config(), generate_trt_nodes_num(), (
            5e-3,
            1e-3,
        )
        self.trt_param.precision = paddle_infer.PrecisionType.Int8
        program_config.set_input_type(np.float32)
        yield self.create_inference_config(), generate_trt_nodes_num(), (
            5e-3,
            5e-3,
        )

    def add_skip_trt_case(self):
        def teller1(program_config, predictor_config):
            if (
                program_config.ops[0].attrs['padding_algorithm'] == "SAME"
                or program_config.ops[0].attrs['padding_algorithm'] == "VALID"
            ):
                return True
            return False

        self.add_skip_case(
            teller1,
            SkipReasons.TRT_NOT_IMPLEMENTED,
            "When padding_algorithm is 'SAME' or 'VALID', Trt dose not support. In this case, trt build error is caused by scale op.",
        )

        def teller2(program_config, predictor_config):
            if self.trt_param.precision == paddle_infer.PrecisionType.Int8:
                return True
            return False

        self.add_skip_case(
            teller2,
            SkipReasons.TRT_NOT_IMPLEMENTED,
            "When precisionType is int8 without relu op, output is different between Trt and Paddle.",
        )

    def test(self):
        self.add_skip_trt_case()
        self.run_test()

    def test_quant(self):
        self.add_skip_trt_case()
        self.run_test(quant=True)


if __name__ == "__main__":
    unittest.main()
