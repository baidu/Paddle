# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import unittest
from functools import partial
from typing import Any

import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import TrtLayerAutoScanTest

import paddle.inference as paddle_infer


class TrtConvertExpandV2Test(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        if self.dtype in [0, 1, 4]:
            return False
        if self.dims != 4 and self.dtype != 2:
            return False
        return True

    def sample_program_configs(self):
        def generate_input1(attrs: list[dict[str, Any]]):
            if self.dims == 4:
                self.input_shape = [1, 1, 4, 6]
                if self.dtype == 0:
                    return np.random.random([1, 1, 4, 6]).astype(np.bool_)
                elif self.dtype == 1:
                    return np.random.random([1, 1, 4, 6]).astype(np.int16)
                elif self.dtype == 2:
                    return np.random.random([1, 1, 4, 6]).astype(np.int32)
                elif self.dtype == 3:
                    return np.random.random([1, 1, 4, 6]).astype(np.int64)
                elif self.dtype == 4:
                    return np.random.random([1, 1, 4, 6]).astype(np.float16)
                elif self.dtype == 5:
                    return np.random.random([1, 1, 4, 6]).astype(np.float32)
                elif self.dtype == 6:
                    return np.random.random([1, 1, 4, 6]).astype(np.float32)
                else:
                    return np.random.random([1, 1, 4, 6]).astype(np.int32)
            elif self.dims == 3:
                self.input_shape = [1, 8, 6]
                return np.random.random([1, 8, 6]).astype(np.int32)
            elif self.dims == 2:
                self.input_shape = [1, 48]
                return np.random.random([1, 48]).astype(np.int32)
            elif self.dims == 1:
                self.input_shape = [48]
                return np.random.random([48]).astype(np.int32)

        def generate_weight1(attrs: list[dict[str, Any]]):
            return np.array([1, 48]).astype(np.int32)

        def generate_shapeT1_data(attrs: list[dict[str, Any]]):
            return np.array([2]).astype(np.int32)

        def generate_shapeT2_data(attrs: list[dict[str, Any]]):
            return np.array([24]).astype(np.int32)

        for dims in [1, 2, 3, 4]:
            for value in [2]:
                for dtype in [-1, 0, 1, 2, 3, 4, 5, 6]:
                    dics = [
                        {
                            "value": value,
                            "dtype": dtype,
                        },
                    ]
                    self.dims = dims
                    self.dtype = dtype
                    dics_input = [{"X": ["fill_any_like_input"]}]

                    ops_config = [
                        {
                            "op_type": "fill_any_like",
                            "op_inputs": dics_input[0],
                            "op_outputs": {"Out": ["fill_any_like_out"]},
                            "op_attrs": dics[0],
                        }
                    ]
                    ops = self.generate_op_config(ops_config)
                    program_config = ProgramConfig(
                        ops=ops,
                        weights={},
                        inputs={
                            "fill_any_like_input": TensorConfig(
                                data_gen=partial(generate_input1, dics)
                            )
                        },
                        outputs=["fill_any_like_out"],
                    )

                    yield program_config

    def generate_dynamic_shape(self, attrs):
        if self.dims == 4:
            self.dynamic_shape.min_input_shape = {
                "fill_any_like_input": [1, 1, 4, 6]
            }
            self.dynamic_shape.max_input_shape = {
                "fill_any_like_input": [10, 1, 4, 6]
            }
            self.dynamic_shape.opt_input_shape = {
                "fill_any_like_input": [1, 1, 4, 6]
            }
        elif self.dims == 3:
            self.dynamic_shape.min_input_shape = {
                "fill_any_like_input": [1, 8, 6]
            }
            self.dynamic_shape.max_input_shape = {
                "fill_any_like_input": [4, 8, 6]
            }
            self.dynamic_shape.opt_input_shape = {
                "fill_any_like_input": [1, 8, 6]
            }
        elif self.dims == 2:
            self.dynamic_shape.min_input_shape = {
                "fill_any_like_input": [1, 48]
            }
            self.dynamic_shape.max_input_shape = {
                "fill_any_like_input": [4, 48]
            }
            self.dynamic_shape.opt_input_shape = {
                "fill_any_like_input": [1, 48]
            }
        elif self.dims == 1:
            self.dynamic_shape.min_input_shape = {"fill_any_like_input": [48]}
            self.dynamic_shape.max_input_shape = {"fill_any_like_input": [48]}
            self.dynamic_shape.opt_input_shape = {"fill_any_like_input": [48]}
        return self.dynamic_shape

    def sample_predictor_configs(
        self, program_config, run_pir=False
    ) -> tuple[paddle_infer.Config, list[int], int]:

        def clear_dynamic_shape():
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            if not dynamic_shape:
                return 0, 3
            else:
                return 1, 2

        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]

        clear_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, False
        ), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, False
        ), 1e-5

        # for dynamic_shape
        self.generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True
        ), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True
        ), 1e-5

    def add_skip_trt_case(self):
        pass

    def test(self):
        self.add_skip_trt_case()
        self.run_test(run_pir=True)


if __name__ == "__main__":
    unittest.main()
