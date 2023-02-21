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

import unittest
from functools import partial
from typing import Any, Dict, List

import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import TrtLayerAutoScanTest

import paddle.inference as paddle_infer


class TrtConvertDeformableConvTest(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        inputs = program_config.inputs
        weights = program_config.weights
        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]

        if (
            inputs['input_data'].shape[1]
            != weights['filter_data'].shape[1] * attrs[0]['groups']
        ):
            return False

        return True

    def sample_program_configs(self):
        def compute_output_size(
            input_size: List[int],
            kernel_sizes: List[int],
            attrs: List[Dict[str, Any]],
        ):
            strides = attrs[0]['strides']
            paddings = attrs[0]['paddings']
            dilations = attrs[0]['dilations']
            output_size = []
            for i, k, s, p, d in zip(
                input_size, kernel_sizes, strides, paddings, dilations
            ):
                k = d * (k - 1) + 1
                output_size.append((i + 2 * p - k) // s + 1)
            return output_size

        def generate_input1(
            batch: int,
            input_size: List[int],
            kernel_sizes: List[int],
            attrs: List[Dict[str, Any]],
        ):
            return np.random.random([batch, 3] + input_size).astype(np.float32)

        def generate_offset1(
            batch: int,
            input_size: List[int],
            kernel_sizes: List[int],
            attrs: List[Dict[str, Any]],
        ):
            output_size = compute_output_size(input_size, kernel_sizes, attrs)
            return np.random.random(
                [batch, 2 * np.prod(kernel_sizes)] + output_size
            ).astype(np.float32)

        def generate_mask1(
            batch: int,
            input_size: List[int],
            kernel_sizes: List[int],
            attrs: List[Dict[str, Any]],
        ):
            output_size = compute_output_size(input_size, kernel_sizes, attrs)
            return np.random.random(
                [batch, np.prod(kernel_sizes)] + output_size
            ).astype(np.float32)

        def generate_filter1(
            batch: int,
            input_size: List[int],
            kernel_sizes: List[int],
            attrs: List[Dict[str, Any]],
        ):
            return np.random.random([6, 3] + kernel_sizes).astype(np.float32)

        for batch in [
            1,
        ]:
            for input_size in [[32, 32]]:
                for kernel_sizes in [[3, 3]]:
                    for strides in [[2, 2]]:
                        for paddings in [[0, 2]]:
                            for groups in [
                                1,
                            ]:
                                for dilations in [[2, 2]]:
                                    dics = [
                                        {
                                            "strides": strides,
                                            "paddings": paddings,
                                            "groups": groups,
                                            "dilations": dilations,
                                            "deformable_groups": 1,
                                            "im2col_step": 1,
                                        }
                                    ]

                                ops_config = [
                                    {
                                        "op_type": "deformable_conv",
                                        "op_inputs": {
                                            "Input": ["input_data"],
                                            "Offset": ["offset_data"],
                                            "Mask": ["mask_data"],
                                            "Filter": ["filter_data"],
                                        },
                                        "op_outputs": {
                                            "Output": ["output_data"]
                                        },
                                        "op_attrs": dics[0],
                                    }
                                ]
                                ops = self.generate_op_config(ops_config)

                                program_config = ProgramConfig(
                                    ops=ops,
                                    weights={
                                        "filter_data": TensorConfig(
                                            data_gen=partial(
                                                generate_filter1,
                                                batch,
                                                input_size,
                                                kernel_sizes,
                                                dics,
                                            )
                                        )
                                    },
                                    inputs={
                                        "input_data": TensorConfig(
                                            data_gen=partial(
                                                generate_input1,
                                                batch,
                                                input_size,
                                                kernel_sizes,
                                                dics,
                                            )
                                        ),
                                        "offset_data": TensorConfig(
                                            data_gen=partial(
                                                generate_offset1,
                                                batch,
                                                input_size,
                                                kernel_sizes,
                                                dics,
                                            )
                                        ),
                                        "mask_data": TensorConfig(
                                            data_gen=partial(
                                                generate_mask1,
                                                batch,
                                                input_size,
                                                kernel_sizes,
                                                dics,
                                            )
                                        ),
                                    },
                                    outputs=["output_data"],
                                )

                                yield program_config

    def sample_predictor_configs(
        self, program_config
    ) -> (paddle_infer.Config, List[int], float):
        def generate_dynamic_shape(attrs):
            # The input.dims[1] must be equal to the weight's length.
            self.dynamic_shape.min_input_shape = {
                "input_data": [1, 3, 32, 32],
                "offset_data": [1, 18, 14, 16],
                "mask_data": [1, 9, 14, 16],
            }
            self.dynamic_shape.max_input_shape = {
                "input_data": [1, 3, 32, 32],
                "offset_data": [1, 18, 14, 16],
                "mask_data": [1, 9, 14, 16],
            }
            self.dynamic_shape.opt_input_shape = {
                "input_data": [1, 3, 32, 32],
                "offset_data": [1, 18, 14, 16],
                "mask_data": [1, 9, 14, 16],
            }

        def clear_dynamic_shape():
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            # TODO: This is just the example, need to be fixed.
            if len(attrs[0]['paddings']) == 4:
                return 1, 2
            else:
                return 1, 4

        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]

        # for static_shape
        clear_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, False
        ), 1e-5

        # for dynamic_shapw
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True
        ), (1e-5, 1e-5)

    def test(self):
        self.trt_param.workspace_size = 1 << 28
        self.run_test()


if __name__ == "__main__":
    unittest.main()
