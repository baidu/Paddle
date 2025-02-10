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

import unittest
from functools import partial
from typing import Any

import hypothesis.strategies as st
import numpy as np
from hypothesis import given
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import TrtLayerAutoScanTest

import paddle.inference as paddle_infer


class TrtConvertTileTest(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        inputs = program_config.inputs
        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]
        for x in attrs[0]['repeat_times']:
            if x <= 0:
                return False

        return True

    def sample_program_configs(self, *args, **kwargs):
        def generate_input1(attrs: list[dict[str, Any]]):
            return np.ones([1, 2]).astype(np.float32)

        dics = [{"repeat_times": kwargs['repeat_times']}]

        ops_config = [
            {
                "op_type": "tile",
                "op_inputs": {"X": ["input_data"]},
                "op_outputs": {"Out": ["tile_output_data"]},
                "op_attrs": dics[0],
            }
        ]
        ops = self.generate_op_config(ops_config)

        program_config = ProgramConfig(
            ops=ops,
            weights={},
            inputs={
                "input_data": TensorConfig(
                    data_gen=partial(generate_input1, dics)
                )
            },
            outputs=["tile_output_data"],
        )

        yield program_config

    def generate_dynamic_shape(self, attrs):
        self.dynamic_shape.min_input_shape = {"input_data": [1, 2]}
        self.dynamic_shape.max_input_shape = {"input_data": [4, 3]}
        self.dynamic_shape.opt_input_shape = {"input_data": [1, 3]}
        return self.dynamic_shape

    def sample_predictor_configs(
        self, program_config, run_pir=False
    ) -> tuple[paddle_infer.Config, list[int], float]:

        def clear_dynamic_shape():
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            ver = paddle_infer.get_trt_compile_version()
            if ver[0] * 1000 + ver[1] * 100 + ver[0] * 10 >= 7000:
                return 1, 2
            else:
                return 0, 3

        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]

        # for static_shape
        clear_dynamic_shape()
        if not run_pir:
            self.trt_param.precision = paddle_infer.PrecisionType.Float32
            program_config.set_input_type(np.float32)
            yield self.create_inference_config(), generate_trt_nodes_num(
                attrs, False
            ), 1e-5
            self.trt_param.precision = paddle_infer.PrecisionType.Half
            program_config.set_input_type(np.float16)
            yield self.create_inference_config(), generate_trt_nodes_num(
                attrs, False
            ), 1e-3

        # for dynamic_shape
        self.generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True
        ), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True
        ), 1e-3

    @given(repeat_times=st.sampled_from([[1], [1, 2], [0, 3]]))
    def test(self, *args, **kwargs):
        self.run_test(run_pir=True, *args, **kwargs)


class TrtConvertTileTest2(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_configs(self):
        def generate_input1(attrs: list[dict[str, Any]]):
            return np.ones([1, 2]).astype(np.float32)

        dics = [{}]
        dics_input = [
            {"X": ["tile_input"], "RepeatTimes": ["repeat_times"]},
        ]
        ops_config = [
            {
                "op_type": "fill_constant",
                "op_inputs": {},
                "op_outputs": {"Out": ["repeat_times"]},
                "op_attrs": {
                    "dtype": 2,
                    "str_value": "1",
                    "value": 1.0,
                    "shape": [1],
                },
            },
            {
                "op_type": "tile",
                "op_inputs": dics_input[0],
                "op_outputs": {"Out": ["tile_out"]},
                "op_attrs": dics[0],
            },
        ]
        ops = self.generate_op_config(ops_config)
        program_config = ProgramConfig(
            ops=ops,
            weights={},
            inputs={
                "tile_input": TensorConfig(
                    data_gen=partial(generate_input1, dics)
                )
            },
            outputs=["tile_out"],
        )

        yield program_config

    def generate_dynamic_shape(self, attrs):
        self.dynamic_shape.min_input_shape = {"tile_input": [1, 2]}
        self.dynamic_shape.max_input_shape = {"tile_input": [4, 3]}
        self.dynamic_shape.opt_input_shape = {"tile_input": [1, 2]}
        return self.dynamic_shape

    def sample_predictor_configs(
        self, program_config, run_pir=False
    ) -> tuple[paddle_infer.Config, list[int], float]:
        def clear_dynamic_shape():
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            return 1, 2

        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]

        # for static_shape
        clear_dynamic_shape()
        if not run_pir:
            self.trt_param.precision = paddle_infer.PrecisionType.Float32
            program_config.set_input_type(np.float32)
            yield self.create_inference_config(), generate_trt_nodes_num(
                attrs, False
            ), 1e-5
            self.trt_param.precision = paddle_infer.PrecisionType.Half
            program_config.set_input_type(np.float16)
            yield self.create_inference_config(), generate_trt_nodes_num(
                attrs, False
            ), 1e-3

        # for dynamic_shape
        self.generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True
        ), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True
        ), 1e-3

    def add_skip_trt_case(self):
        pass

    def test(self):
        self.add_skip_trt_case()
        self.run_test(run_pir=True)


class TrtConvertTileTest3(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]
        return True

    def sample_program_configs(self):
        def generate_input1(attrs: list[dict[str, Any]]):
            return np.ones([1, 2]).astype(np.float32)

        dics = [{}]
        dics_input = [
            {
                "X": ["tile_input"],
                "repeat_times_tensor": ["repeat_times1", "repeat_times2"],
            },
        ]
        ops_config = [
            {
                "op_type": "fill_constant",
                "op_inputs": {},
                "op_outputs": {"Out": ["repeat_times1"]},
                "op_attrs": {
                    "dtype": 2,
                    "str_value": "10",
                    "value": 1.0,
                    "shape": [1],
                },
            },
            {
                "op_type": "fill_constant",
                "op_inputs": {},
                "op_outputs": {"Out": ["repeat_times2"]},
                "op_attrs": {
                    "dtype": 2,
                    "str_value": "12",
                    "value": 1.0,
                    "shape": [1],
                },
            },
            {
                "op_type": "tile",
                "op_inputs": dics_input[0],
                "op_outputs": {"Out": ["tile_out"]},
                "op_attrs": dics[0],
            },
        ]
        ops = self.generate_op_config(ops_config)
        program_config = ProgramConfig(
            ops=ops,
            weights={},
            inputs={
                "tile_input": TensorConfig(
                    data_gen=partial(generate_input1, dics)
                )
            },
            outputs=["tile_out"],
        )

        yield program_config

    def generate_dynamic_shape(self, attrs):
        self.dynamic_shape.min_input_shape = {"tile_input": [1, 2]}
        self.dynamic_shape.max_input_shape = {"tile_input": [4, 3]}
        self.dynamic_shape.opt_input_shape = {"tile_input": [1, 2]}
        return self.dynamic_shape

    def sample_predictor_configs(
        self, program_config, run_pir=False
    ) -> tuple[paddle_infer.Config, list[int], float]:
        def clear_dynamic_shape():
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            return 1, 2

        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]

        # for static_shape
        clear_dynamic_shape()
        if not run_pir:
            self.trt_param.precision = paddle_infer.PrecisionType.Float32
            program_config.set_input_type(np.float32)
            yield self.create_inference_config(), generate_trt_nodes_num(
                attrs, False
            ), 1e-5
            self.trt_param.precision = paddle_infer.PrecisionType.Half
            program_config.set_input_type(np.float16)
            yield self.create_inference_config(), generate_trt_nodes_num(
                attrs, False
            ), 1e-3

        # for dynamic_shape
        self.generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True
        ), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True
        ), 1e-3

    def add_skip_trt_case(self):
        pass

    def test(self):
        self.add_skip_trt_case()
        self.run_test(run_pir=True)


if __name__ == "__main__":
    unittest.main()
