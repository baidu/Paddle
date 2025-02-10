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

import copy
import unittest

import numpy as np

import paddle
from paddle.base import core
from paddle.tensorrt.converter import PaddleToTensorRTConverter
from paddle.tensorrt.export import (
    Input,
    PrecisionMode,
    TensorRTConfig,
)
from paddle.tensorrt.util import (
    mark_builtin_op,
    run_pir_pass,
    run_trt_partition,
    warmup_shape_infer,
)


class TensorRTBaseTest(unittest.TestCase):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.python_api = None
        self.api_args = None
        self.program_config = None
        self.min_shape = None
        self.opt_shape = None
        self.max_shape = None
        self.target_marker_op = ""
        self.dynamic_shape_data = {}
        self.disable_passes = [
            "constant_folding_pass",
        ]

    def create_fake_program(self):
        if self.python_api is None:
            raise ValueError(
                "The unittest must specify a python api that will be used for building pir program."
            )
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            api_args = copy.deepcopy(self.api_args)
            for feed_name in self.program_config["feed_list"]:
                if self.api_args[feed_name] is None:
                    continue
                if isinstance(self.api_args[feed_name], dict):
                    new_list_args = []
                    for sub_arg_name, sub_arg_value in self.api_args[
                        feed_name
                    ].items():
                        if (
                            feed_name in self.min_shape.keys()
                            and feed_name in self.opt_shape.keys()
                            and feed_name in self.max_shape.keys()
                        ):
                            input_shape_without_dynamic_dim = (
                                sub_arg_value.shape[1:]
                            )
                            input_dynamic_shape = [-1]
                            input_dynamic_shape.extend(
                                input_shape_without_dynamic_dim
                            )
                            input_shape = input_dynamic_shape
                        else:
                            input_shape = []
                            input_shape_without_dynamic_dim = (
                                sub_arg_value.shape[0:]
                            )
                            input_shape.extend(input_shape_without_dynamic_dim)

                        input_dtype = sub_arg_value.dtype
                        input_data = paddle.static.data(
                            name=sub_arg_name,
                            shape=input_shape,
                            dtype=input_dtype,
                        )
                        new_list_args.append(input_data)
                    api_args[feed_name] = new_list_args
                else:
                    empty_min_max_shape = (
                        self.min_shape is None
                        or self.max_shape is None
                        or self.opt_shape is None
                    )

                    if (
                        not empty_min_max_shape
                        and feed_name in self.min_shape.keys()
                        and feed_name in self.opt_shape.keys()
                        and feed_name in self.max_shape.keys()
                    ):
                        # dynamic shape condition
                        input_shape_without_dynamic_dim = self.api_args[
                            feed_name
                        ].shape[1:]
                        input_shape = [-1]
                        input_shape.extend(input_shape_without_dynamic_dim)
                    else:
                        input_shape = self.api_args[feed_name].shape

                    input_dtype = self.api_args[feed_name].dtype

                    input_data = paddle.static.data(
                        name=feed_name,
                        shape=input_shape,
                        dtype=input_dtype,
                    )
                    api_args[feed_name] = input_data
            actual_args = []
            for name, value in api_args.items():
                actual_args.append(value)
            output = self.python_api(*actual_args)
            fetch_list = []
            if isinstance(output, tuple):
                fetch_list = [out for out in list(output) if out is not None]
            else:
                fetch_list.append(output)
        return main_program, startup_program, fetch_list

    def run_program(self, main_program, fetch_list):
        place = (
            paddle.CUDAPlace(0)
            if core.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )
        exe = paddle.static.Executor(place)
        feed_data = dict()  # noqa: C408
        for feed_name in self.program_config["feed_list"]:
            if self.api_args[feed_name] is None:
                continue
            if isinstance(self.api_args[feed_name], dict):
                for sub_arg_name, sub_arg_value in self.api_args[
                    feed_name
                ].items():
                    feed_data[sub_arg_name] = sub_arg_value
            else:
                feed_data[feed_name] = self.api_args[feed_name]
        ret = exe.run(main_program, feed=feed_data, fetch_list=fetch_list)
        return ret

    def prepare_feed(self):
        for arg_name, arg_value in self.api_args.items():
            # deal with condition that input is a list tensor
            if (
                isinstance(self.api_args[arg_name], list)
                and arg_name in self.program_config["feed_list"]
            ):
                new_list_args = dict()  # noqa: C408
                for i in range(len(self.api_args[arg_name])):
                    sub_arg_name = arg_name + str(i)
                    new_list_args[sub_arg_name] = self.api_args[arg_name][i]
                self.api_args[arg_name] = new_list_args

    def check_trt_result(self, rtol=1e-5, atol=1e-5, precision_mode="fp32"):
        paddle.framework.set_flags({"FLAGS_trt_min_group_size": 1})
        with paddle.pir_utils.IrGuard():
            self.prepare_feed()
            main_program, startup_program, fetch_list = (
                self.create_fake_program()
            )
            place = (
                paddle.CUDAPlace(0)
                if core.is_compiled_with_cuda()
                else paddle.CPUPlace()
            )
            exe = paddle.static.Executor(place)
            # init all parameter
            exe.run(startup_program)
            fetch_num = len(fetch_list)
            if isinstance(fetch_list[0], list):
                fetch_index = [i for i, v in enumerate(fetch_list)]
            else:
                fetch_index = [v.index() for v in fetch_list]
            output_expected = self.run_program(main_program, fetch_list)

            min_shape_data = dict()  # noqa: C408
            opt_shape_data = dict()  # noqa: C408
            max_shape_data = dict()  # noqa: C408
            for feed_name in self.program_config["feed_list"]:
                if self.api_args[feed_name] is None:
                    continue
                if isinstance(self.api_args[feed_name], dict):
                    # shape_tensor
                    if (
                        feed_name not in self.min_shape.keys()
                        and feed_name not in self.max_shape.keys()
                        and feed_name not in self.opt_shape.keys()
                    ):
                        for sub_feed_name, sub_feed_value in self.api_args[
                            feed_name
                        ].items():
                            min_shape_data[sub_feed_name] = sub_feed_value
                            opt_shape_data[sub_feed_name] = sub_feed_value
                            max_shape_data[sub_feed_name] = sub_feed_value
                            continue
                    else:
                        # not shape_tensor
                        for i in range(len(self.min_shape[feed_name])):
                            sub_feed_name = feed_name + str(i)
                            min_shape_data[sub_feed_name] = np.random.randn(
                                *self.min_shape[feed_name][i]
                            ).astype(
                                self.api_args[feed_name][sub_feed_name].dtype
                            )
                            opt_shape_data[sub_feed_name] = np.random.randn(
                                *self.opt_shape[feed_name][i]
                            ).astype(
                                self.api_args[feed_name][sub_feed_name].dtype
                            )
                            max_shape_data[sub_feed_name] = np.random.randn(
                                *self.max_shape[feed_name][i]
                            ).astype(
                                self.api_args[feed_name][sub_feed_name].dtype
                            )
                else:
                    # shape_tensor is list
                    if (
                        feed_name not in self.min_shape.keys()
                        and feed_name not in self.max_shape.keys()
                        and feed_name not in self.opt_shape.keys()
                    ):
                        min_shape_data[feed_name] = self.api_args[feed_name]
                        opt_shape_data[feed_name] = self.api_args[feed_name]
                        max_shape_data[feed_name] = self.api_args[feed_name]
                        continue
                    else:
                        if self.dynamic_shape_data:
                            min_shape_data[feed_name] = self.dynamic_shape_data[
                                feed_name
                            ](self.min_shape[feed_name])
                            opt_shape_data[feed_name] = self.dynamic_shape_data[
                                feed_name
                            ](self.opt_shape[feed_name])
                            max_shape_data[feed_name] = self.dynamic_shape_data[
                                feed_name
                            ](self.max_shape[feed_name])
                        else:
                            min_shape_data[feed_name] = np.random.randn(
                                *self.min_shape[feed_name]
                            ).astype(self.api_args[feed_name].dtype)
                            opt_shape_data[feed_name] = np.random.randn(
                                *self.opt_shape[feed_name]
                            ).astype(self.api_args[feed_name].dtype)
                            max_shape_data[feed_name] = np.random.randn(
                                *self.max_shape[feed_name]
                            ).astype(self.api_args[feed_name].dtype)

            # run pir pass(including some constant fold pass, dead code elimination pass, fusion pass and trt_op_marker_pass)
            main_program = run_pir_pass(
                main_program,
                disable_passes=self.disable_passes,
            )

            scope = paddle.static.global_scope()
            main_program = warmup_shape_infer(
                main_program,
                min_shape_feed=min_shape_data,
                opt_shape_feed=opt_shape_data,
                max_shape_feed=max_shape_data,
                scope=scope,
            )
            for op in main_program.global_block().ops[::-1]:
                # Remove all invalid fetch op
                if op.name() == "pd_op.fetch":
                    main_program.global_block().remove_op(op)

            # Adding marker labels to builtin ops facilitates convert processing, but they ultimately do not enter the TensorRT subgraph.
            mark_builtin_op(main_program)

            # run trt_sub_graph_extract_pass()
            program_with_trt = run_trt_partition(main_program)

            # run TRTConverter(would lower group_op into tensorrt_engine_op)
            trt_config = None

            input = Input(
                min_input_shape=self.min_shape,
                optim_input_shape=self.opt_shape,
                max_input_shape=self.max_shape,
            )
            trt_config = TensorRTConfig(inputs=[input])
            if precision_mode == "fp16":
                trt_config.precision_mode = PrecisionMode.FP16

            converter = PaddleToTensorRTConverter(
                program_with_trt, scope, trt_config
            )
            converter.convert_program_to_trt()

            # check whether has trt op
            has_trt_op = False
            for op in program_with_trt.global_block().ops:
                if op.name() == "pd_op.tensorrt_engine":
                    has_trt_op = True
            self.assertEqual(has_trt_op, True)

            trt_fetch_list = []
            split_op = program_with_trt.global_block().ops[-1]
            if split_op.name() == "builtin.split":
                trt_fetch_list = [
                    split_op.result(index) for index in fetch_index
                ]
            else:
                raise ValueError(
                    "The last op of convert pir Program in test must be split op that is the next op of pd_op.engine."
                )
            output_trt = self.run_program(program_with_trt, trt_fetch_list)

        # Check that the results are close to each other within a tolerance of 1e-3
        for i in range(fetch_num):
            np.testing.assert_allclose(
                output_expected[i],
                output_trt[i],
                rtol=rtol,
                atol=atol,
            )

    def check_marker(self, expected_result):
        paddle.framework.set_flags({"FLAGS_trt_min_group_size": 1})
        with paddle.pir_utils.IrGuard():
            main_program, startup_program, fetch_list = (
                self.create_fake_program()
            )
            main_program = run_pir_pass(
                main_program,
                disable_passes=self.disable_passes,
            )
            marker_result = False
            for op in main_program.global_block().ops:
                if op.name() == self.target_marker_op:
                    marker_result = op.attrs().get("__l_trt__", False)
            self.assertEqual(marker_result, expected_result)
