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

import abc
import enum
import os
import shutil
import time
import unittest
from typing import Any, Callable

import hypothesis
import hypothesis.strategies as st
import numpy as np
from hypothesis import given, settings
from program_config import (
    OpConfig,
    ProgramConfig,
    create_fake_model,
    create_quant_model,
)

import paddle
import paddle.inference as paddle_infer
from paddle import pir
from paddle.base.core import PassVersionChecker
from paddle.static.log_helper import get_logger

# windows and xpu not support tensort
if os.name != 'nt' and (not os.getenv('WITH_XPU')):
    try:
        from paddle.tensorrt.export import (
            Input,
            PrecisionMode,
            TensorRTConfig,
            convert_to_trt,
        )
    except ImportError:
        raise RuntimeError("TensorRT package is not available.")

LOGLEVEL = os.environ.get("PADDLE_TEST_LOGLEVEL", "INFO").upper()
logging = get_logger(
    __name__, LOGLEVEL, fmt='%(asctime)s-%(levelname)s: %(message)s'
)

settings.register_profile(
    "ci",
    max_examples=100,
    suppress_health_check=hypothesis.HealthCheck.all(),
    deadline=None,
    print_blob=True,
    derandomize=True,
    report_multiple_bugs=False,
)
settings.register_profile(
    "dev",
    max_examples=1000,
    suppress_health_check=hypothesis.HealthCheck.all(),
    deadline=None,
    print_blob=True,
    derandomize=True,
    report_multiple_bugs=False,
)
if (
    float(os.getenv("TEST_NUM_PERCENT_CASES", default="1.0")) < 1
    or os.getenv("HYPOTHESIS_TEST_PROFILE", "dev") == "ci"
):
    settings.load_profile("ci")
else:
    settings.load_profile("dev")


class IgnoreReasons(enum.Enum):
    # Paddle not support, but trt support, we need to add the feature.
    TRT_NOT_IMPLEMENTED = 0
    # TRT not support.
    TRT_NOT_SUPPORT = 1
    # Accuracy is abnormal after enabling pass.
    PASS_ACCURACY_ERROR = 2
    # Accuracy is abnormal after enabling onednn.
    ONEDNN_ACCURACY_ERROR = 3
    # Accuracy is abnormal after enabling cutlass.
    CUTLASS_ACCURACY_ERROR = 3


# TODO(wilber): just for backward compatible
SkipReasons = IgnoreReasons


class AutoScanTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        np.random.seed(1024)
        paddle.enable_static()
        super().__init__(*args, **kwargs)
        self.ignore_cases = []
        abs_dir = os.path.abspath(os.path.dirname(__file__))
        self.cache_dir = os.path.join(
            abs_dir, str(self.__module__) + '_cache_dir'
        )
        self.available_passes_in_framework = set()
        self.num_ran_programs = 0
        self.num_invalid_programs = 0
        self.num_ignore_tests = 0
        self.num_predictor_kinds = 0

    @abc.abstractmethod
    def sample_program_configs(self):
        """
        Generate all config with the combination of different Input tensor shape and
        different Attr values.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def sample_predictor_configs(self):
        raise NotImplementedError

    @abc.abstractmethod
    def add_ignore_check_case(
        self,
        teller: list[Callable[[ProgramConfig, paddle_infer.Config], bool]],
        reason: IgnoreReasons,
        note: str,
    ):
        self.ignore_cases.append((teller, reason, note))

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def run_test_config(
        self, model, params, prog_config, pred_config, feed_data
    ) -> dict[str, np.ndarray]:
        """
        Test a single case.
        """
        with paddle.pir_utils.OldIrGuard():
            pred_config.set_model_buffer(model, len(model), params, len(params))
            predictor = paddle_infer.create_predictor(pred_config)
            self.available_passes_in_framework = (
                self.available_passes_in_framework
                | set(pred_config.pass_builder().all_passes())
            )
            for name, _ in prog_config.inputs.items():
                input_tensor = predictor.get_input_handle(name)
                input_tensor.copy_from_cpu(feed_data[name]["data"])
                if feed_data[name]["lod"] is not None:
                    input_tensor.set_lod(feed_data[name]["lod"])

            predictor.run()
        result = {}
        for out_name, o_name in zip(
            prog_config.outputs, predictor.get_output_names()
        ):
            result[out_name] = predictor.get_output_handle(o_name).copy_to_cpu()
        return result

    def transform_to_trt_program(self, pir_program, trt_config):
        if trt_config.input_data_type == 'float16':
            trt_config.precision_mode = PrecisionMode.FP16

        paddle.framework.set_flags({"FLAGS_trt_min_group_size": 1})
        # translalte pir program to trt program
        scope = paddle.static.global_scope()
        program_with_trt = convert_to_trt(pir_program, trt_config, scope)

        return program_with_trt

    @abc.abstractmethod
    def assert_tensors_near(
        self,
        atol: float,
        rtol: float,
        tensor: dict[str, np.array],
        baseline: dict[str, np.array],
    ):
        for key, arr in tensor.items():
            self.assertTrue(
                baseline[key].shape == arr.shape,
                f"The output shapes are not equal, the baseline shape is {baseline[key].shape}, but got {arr.shape}",
            )
            diff = abs(baseline[key] - arr)
            np.testing.assert_allclose(
                baseline[key],
                arr,
                rtol=rtol,
                atol=atol,
                err_msg=f"Output has diff, Maximum absolute error: {np.amax(diff)}",
            )

    @abc.abstractmethod
    def run_test(self, quant=False):
        raise NotImplementedError

    def generate_op_config(
        self, ops_config: list[dict[str, Any]]
    ) -> list[OpConfig]:
        ops = []
        for i in range(len(ops_config)):
            op_config = ops_config[i]
            if 'outputs_dtype' in op_config:
                ops.append(
                    OpConfig(
                        type=op_config['op_type'],
                        inputs=op_config['op_inputs'],
                        outputs=op_config['op_outputs'],
                        attrs=op_config['op_attrs'],
                        outputs_dtype=op_config['outputs_dtype'],
                    )
                )
            else:
                ops.append(
                    OpConfig(
                        type=op_config['op_type'],
                        inputs=op_config['op_inputs'],
                        outputs=op_config['op_outputs'],
                        attrs=op_config['op_attrs'],
                    )
                )
        return ops

    @abc.abstractmethod
    def ignore_log(self, msg: str):
        logging.debug(f"SKIP: {msg}")

    @abc.abstractmethod
    def fail_log(self, msg: str):
        logging.error(f"FAIL: {msg}")

    @abc.abstractmethod
    def info_log(self, msg: str):
        logging.debug(f"INFO: {msg}")

    @abc.abstractmethod
    def success_log(self, msg: str):
        logging.debug(f"SUCCESS: {msg}")

    @abc.abstractmethod
    def create_inference_config(
        self,
        passes: list[str] | None = None,
        use_gpu: bool = False,
        use_mkldnn: bool = False,
        use_xpu: bool = False,
        ir_optim: bool | None = None,
    ):
        config = paddle_infer.Config()
        config.switch_ir_debug(True)
        config.set_optim_cache_dir(self.cache_dir)
        config.disable_glog_info()
        if ir_optim is not None:
            config.switch_ir_optim(ir_optim)
        if use_gpu:
            config.enable_use_gpu(100, 0)
        if not use_mkldnn:
            config.disable_mkldnn()
        if use_xpu:
            config.enable_xpu()
        if passes is not None:
            config.pass_builder().set_passes(passes)
            self.passes = passes
        return config


class MkldnnAutoScanTest(AutoScanTest):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run_test(self, quant=False, *args, **kwargs):
        status = True

        for prog_config in self.sample_program_configs(*args, **kwargs):
            # if program is invalid, we should skip that cases.
            if not self.is_program_valid(prog_config):
                continue

            with paddle.pir_utils.OldIrGuard():
                main_program_desc, util_program = create_fake_model(prog_config)
                model = main_program_desc.serialize_to_string()
                place = paddle.base.CPUPlace()
                executor = paddle.base.Executor(place)
                scope = paddle.base.Scope()
                with paddle.base.scope_guard(scope):
                    executor.run(util_program)
                    params = scope.find_var("out_var_0").get_bytes()

            if quant:
                model, params = create_quant_model(model, params)

            feed_data = {}
            for name, tensor_config in prog_config.inputs.items():
                feed_data[name] = {
                    "data": tensor_config.data,
                    "lod": tensor_config.lod,
                }
            results: list[dict[str, np.ndarray]] = []

            # baseline: cpu no ir_optim run
            base_config = self.create_inference_config(ir_optim=False)
            results.append(
                self.run_test_config(
                    model, params, prog_config, base_config, feed_data
                )
            )
            self.success_log(f"baseline program_config: {prog_config}")
            self.success_log(
                f"baseline predictor_config: {self.inference_config_str(base_config)}"
            )

            for pred_config, (atol, rtol) in self.sample_predictor_configs(
                prog_config
            ):
                # skip info
                ignore_flag = False
                for ignore_info in self.ignore_cases:
                    if ignore_info[0](prog_config, pred_config):
                        ignore_flag = True
                        if (
                            ignore_info[1]
                            == IgnoreReasons.ONEDNN_ACCURACY_ERROR
                        ):
                            self.ignore_log(
                                f"[ONEDNN_ACCURACY_ERROR] {ignore_info[2]} vs {self.inference_config_str(pred_config)}"
                            )
                        else:
                            raise NotImplementedError
                        break

                if os.path.exists(self.cache_dir):
                    shutil.rmtree(self.cache_dir)
                if not os.path.exists(self.cache_dir):
                    os.mkdir(self.cache_dir)

                try:
                    results.append(
                        self.run_test_config(
                            model, params, prog_config, pred_config, feed_data
                        )
                    )
                    self.assert_tensors_near(
                        atol, rtol, results[-1], results[0]
                    )

                    self.success_log(f"program_config: {prog_config}")
                    self.success_log(
                        f"predictor_config: {self.inference_config_str(pred_config)}"
                    )
                except Exception as e:
                    self.fail_log(f"program_config: {prog_config}")
                    self.fail_log(
                        f"predictor_config: {self.inference_config_str(pred_config)}"
                    )
                    self.fail_log(f"\033[1;31m ERROR INFO: {e}\033[0m")
                    if not ignore_flag:
                        status = False
                    continue

        self.assertTrue(status)

    def inference_config_str(self, config) -> str:
        dic = {}
        enable_mkldnn = config.mkldnn_enabled()
        dic["use_mkldnn"] = enable_mkldnn
        enable_gpu = config.use_gpu()
        dic["use_gpu"] = enable_gpu
        return str(dic)


class PirMkldnnAutoScanTest(MkldnnAutoScanTest):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run_test_config(
        self, model, params, prog_config, pred_config, feed_data
    ) -> dict[str, np.ndarray]:
        """
        Test a single case.
        """
        pred_config.enable_new_ir(True)
        pred_config.switch_ir_optim(False)
        pred_config.enable_new_executor()
        result = super().run_test_config(
            model, params, prog_config, pred_config, feed_data
        )
        pred_config.enable_new_ir(False)
        return result


class PassAutoScanTest(AutoScanTest):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.passes = []

    def check_op_version(self):
        status = True
        for pass_name in self.passes:
            if pass_name not in self.available_passes_in_framework:
                continue
            if not PassVersionChecker.IsCompatible(pass_name):
                self.fail_log(f"{pass_name} version check failed.")
                status = False
        return status

    def add_ignore_pass_case(self):
        return

    def assert_op_list(self, op_list_after_fusion):
        if not self.passes:
            raise ValueError(
                "In PassAutoScan you should give a valid pass name."
            )
        last_passed_program = os.path.join(
            self.cache_dir, self.passes[-1] + ".pdmodel"
        )
        if not os.path.exists(last_passed_program):
            raise ValueError(
                f"Cannot find file {last_passed_program}, please make sure that your pass name is correct"
            )
        model_bytes = paddle.static.load_from_file(last_passed_program)
        pg = paddle.static.deserialize_program(model_bytes)
        main_block = pg.desc.block(0)
        after_op_list = []
        for i in range(main_block.op_size()):
            if main_block.op(i).type() in ["feed", "fetch"]:
                continue
            after_op_list.append(main_block.op(i).type())
        self.assertTrue(
            op_list_after_fusion == after_op_list,
            f"Expected operator list after fusion is {op_list_after_fusion}, but now it's {after_op_list}",
        )

    def run_and_statis(
        self,
        quant=False,
        max_examples=100,
        reproduce=None,
        min_success_num=25,
        max_duration=1000,
        passes=None,
    ):
        if os.getenv("HYPOTHESIS_TEST_PROFILE", "ci") == "dev":
            max_examples *= 10
            min_success_num *= 10
            # while at ce phase, there"s no limit on time
            max_duration = -1
        start_time = time.time()
        settings.register_profile(
            "ci",
            max_examples=max_examples,
            suppress_health_check=hypothesis.HealthCheck.all(),
            deadline=None,
            print_blob=True,
            derandomize=True,
            report_multiple_bugs=False,
        )
        settings.load_profile("ci")
        assert (
            passes is not None
        ), "Parameter of passes must be defined in function run_and_statis."
        self.passes = passes

        self.add_ignore_pass_case()

        def program_generator(draw):
            return self.sample_program_config(draw)

        def run_test(prog_config):
            return self.run_test(quant=quant, prog_configs=[prog_config])

        generator = st.composite(program_generator)
        loop_func = given(generator())(run_test)
        if reproduce is not None:
            loop_func = reproduce(loop_func)
        logging.info(f"Start to running test of {type(self)}")
        loop_func()
        self.info_log(
            "===================Statistical Information==================="
        )
        self.info_log(
            f"Number of Generated Programs: {self.num_ran_programs + self.num_invalid_programs}"
        )
        logging.info(f"Number of Invalid Programs: {self.num_invalid_programs}")
        logging.info(f"Number of Ran Programs: {self.num_ran_programs}")
        logging.info(f"Number of Ignore Tests: {self.num_ignore_tests}")
        successful_ran_programs = int(
            self.num_ran_programs
            - self.num_ignore_tests / max(self.num_predictor_kinds, 1)
        )
        self.info_log(
            f"Number of successfully ran programs approximately equal to {successful_ran_programs}"
        )
        if successful_ran_programs < min_success_num:
            self.fail_log(
                "satisfied_programs = ran_programs - num_ignore_tests / num_predictor_kinds"
            )
            self.fail_log(
                f"At least {min_success_num} programs need to ran successfully, but now only about {successful_ran_programs} programs satisfied."
            )
            raise AssertionError
        used_time = time.time() - start_time
        if max_duration > 0 and used_time > max_duration:
            self.fail_log(
                f"The duration exceeds {max_duration} seconds, if this is necessary, try to set a larger number for parameter `max_duration`."
            )
            raise AssertionError

    def run_test(self, quant=False, prog_configs=None):
        status = True

        for prog_config in prog_configs:
            # if program is invalid, we should skip that cases.
            if not self.is_program_valid(prog_config):
                self.num_invalid_programs += 1
                continue
            self.num_ran_programs += 1

            with paddle.pir_utils.OldIrGuard():
                main_program_desc, util_program = create_fake_model(prog_config)
                model = main_program_desc.serialize_to_string()
                place = paddle.base.CPUPlace()
                executor = paddle.base.Executor(place)
                scope = paddle.base.Scope()
                with paddle.base.scope_guard(scope):
                    executor.run(util_program)
                    params = scope.find_var("out_var_0").get_bytes()

            if quant:
                model, params = create_quant_model(model, params)

            feed_data = {}
            for name, tensor_config in prog_config.inputs.items():
                feed_data[name] = {
                    "data": tensor_config.data,
                    "lod": tensor_config.lod,
                }

            self.num_predictor_kinds = 0
            for (
                pred_config,
                op_list,
                (atol, rtol),
            ) in self.sample_predictor_configs(prog_config):
                self.num_predictor_kinds += 1

                # skip info
                ignore_flag = False
                for ignore_info in self.ignore_cases:
                    if ignore_info[0](prog_config, pred_config):
                        ignore_flag = True
                        self.num_ignore_tests += 1
                        if ignore_info[1] == IgnoreReasons.PASS_ACCURACY_ERROR:
                            self.ignore_log(
                                f"[PASS_ACCURACY_ERROR] {ignore_info[2]} vs {self.inference_config_str(pred_config)}"
                            )
                        else:
                            raise NotImplementedError
                        break

                if os.path.exists(self.cache_dir):
                    shutil.rmtree(self.cache_dir)
                if not os.path.exists(self.cache_dir):
                    os.mkdir(self.cache_dir)

                # baseline: no ir_optim run
                base_config = self.create_inference_config(
                    ir_optim=False, use_gpu=pred_config.use_gpu()
                )
                try:
                    # baseline
                    base_result = self.run_test_config(
                        model, params, prog_config, base_config, feed_data
                    )
                    self.success_log(
                        f"baseline program_config: {self.inference_config_str(base_config)}"
                    )

                    if os.path.exists(self.cache_dir):
                        shutil.rmtree(self.cache_dir)

                    pred_result = self.run_test_config(
                        model, params, prog_config, pred_config, feed_data
                    )
                    self.assert_tensors_near(
                        atol, rtol, pred_result, base_result
                    )
                    if not ignore_flag:
                        self.assert_op_list(op_list)

                    self.success_log(f"program_config: {prog_config}")
                    self.success_log(
                        f"predictor_config: {self.inference_config_str(pred_config)}"
                    )
                except Exception as e:
                    self.fail_log(f"program_config: {prog_config}")
                    self.fail_log(
                        f"predictor_config: {self.inference_config_str(pred_config)}"
                    )
                    self.fail_log(f"\033[1;31m ERROR INFO: {e}\033[0m")
                    if not ignore_flag:
                        status = False
                    continue

        status = self.check_op_version() and status
        self.assertTrue(status)

    def inference_config_str(self, config) -> str:
        dic = {}
        enable_mkldnn = config.mkldnn_enabled()
        dic["use_mkldnn"] = enable_mkldnn
        enable_gpu = config.use_gpu()
        dic['use_gpu'] = enable_gpu
        enable_xpu = config.use_xpu()
        dic['use_xpu'] = enable_xpu
        if not self.passes:
            dic["passes"] = self.passes

        enable_trt = config.tensorrt_engine_enabled()
        trt_precision = config.tensorrt_precision_mode()
        trt_dynamic_shape = config.tensorrt_dynamic_shape_enabled()
        if enable_trt:
            dic["use_trt"] = True
            dic["trt_precision"] = trt_precision
            dic["use_dynamic_shape"] = trt_dynamic_shape
        else:
            dic["use_trt"] = False
        return str(dic)

    def create_trt_inference_config(self) -> paddle_infer.Config:
        config = paddle_infer.Config()
        config.disable_glog_info()
        config.enable_use_gpu(100, 0)
        config.set_optim_cache_dir(self.cache_dir)
        config.switch_ir_debug()
        return config


class TrtLayerAutoScanTest(AutoScanTest):
    class TensorRTParam:
        """
        TensorRT subgraph engine parameters.
        """

        def __init__(
            self,
            workspace_size,
            max_batch_size,
            min_subgraph_size,
            precision,
            use_static,
            use_calib_mode,
        ):
            self.workspace_size = workspace_size
            self.max_batch_size = max_batch_size
            self.min_subgraph_size = min_subgraph_size
            self.precision = precision
            self.use_static = use_static
            self.use_calib_mode = use_calib_mode

    class DynamicShapeParam:
        """
        Prepare TensorRT subgraph engine dynamic shape parameters.
        """

        def __init__(
            self,
            min_input_shape,
            max_input_shape,
            opt_input_shape,
            disable_trt_plugin_fp16,
        ):
            self.min_input_shape = min_input_shape
            self.max_input_shape = max_input_shape
            self.opt_input_shape = opt_input_shape
            self.disable_trt_plugin_fp16 = disable_trt_plugin_fp16

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trt_param = self.TensorRTParam(
            workspace_size=8192,
            max_batch_size=4,
            min_subgraph_size=0,
            precision=paddle_infer.PrecisionType.Float32,
            use_static=True,
            use_calib_mode=False,
        )
        self.dynamic_shape = self.DynamicShapeParam({}, {}, {}, False)
        self.num_percent_cases = float(
            os.getenv("TEST_NUM_PERCENT_CASES", default="1.0")
        )

        # Use a separate random generator for skipping tests
        self.skip_rng = np.random.default_rng(int(time.strftime("%W")))
        self.optimization_level = None

    def create_inference_config(self, use_trt=True) -> paddle_infer.Config:
        config = paddle_infer.Config()
        config.disable_glog_info()
        config.enable_use_gpu(100, 0)
        config.set_optim_cache_dir(self.cache_dir)
        if use_trt:
            config.switch_ir_debug()
            config.enable_tensorrt_engine(
                max_batch_size=self.trt_param.max_batch_size,
                workspace_size=self.trt_param.workspace_size,
                min_subgraph_size=self.trt_param.min_subgraph_size,
                precision_mode=self.trt_param.precision,
                use_static=self.trt_param.use_static,
                use_calib_mode=self.trt_param.use_calib_mode,
            )
            if self.dynamic_shape.min_input_shape and (
                self.dynamic_shape.min_input_shape.keys()
                == self.dynamic_shape.max_input_shape.keys()
                == self.dynamic_shape.opt_input_shape.keys()
            ):
                config.set_trt_dynamic_shape_info(
                    self.dynamic_shape.min_input_shape,
                    self.dynamic_shape.max_input_shape,
                    self.dynamic_shape.opt_input_shape,
                    self.dynamic_shape.disable_trt_plugin_fp16,
                )
            if self.optimization_level is not None:
                config.set_tensorrt_optimization_level(self.optimization_level)
        return config

    def assert_tensors_near(
        self,
        atol: float,
        rtol: float,
        tensor,
        baseline,
    ):
        if isinstance(tensor, dict) and isinstance(baseline, dict):
            for key, arr in tensor.items():
                self.assertEqual(
                    baseline[key].shape,
                    arr.shape,
                    f"The output shapes are not equal, the baseline shape is {baseline[key].shape}, but got {arr.shape}",
                )
                np.testing.assert_allclose(
                    arr, baseline[key], rtol=rtol, atol=atol
                )
        elif isinstance(tensor, list) and isinstance(baseline, list):
            for value_t, value_b in zip(tensor, baseline):
                self.assertEqual(
                    value_t.shape,
                    value_b.shape,
                    f"The output shapes are not equal, the baseline shape is {value_b.shape}, but got {value_t.shape}",
                )
                np.testing.assert_allclose(
                    value_t, value_b, rtol=rtol, atol=atol
                )
        else:
            raise ValueError("Input types are not supported")

    def assert_op_size(self, trt_engine_num, paddle_op_num):
        fp32_last_pass = "transpose_flatten_concat_fuse_pass"
        fp16_last_pass = "tensorrt_subgraph_pass"
        last_passed_program = os.path.join(
            self.cache_dir, f"{fp32_last_pass}.pdmodel"
        )
        if not os.path.exists(last_passed_program):
            last_passed_program = os.path.join(
                self.cache_dir, f"{fp16_last_pass}.pdmodel"
            )
        model_bytes = paddle.static.load_from_file(last_passed_program)
        pg = paddle.static.deserialize_program(model_bytes)
        main_block = pg.desc.block(0)
        op_size = main_block.op_size()
        op_types = [
            main_block.op(i).type() == "tensorrt_engine" for i in range(op_size)
        ]
        trt_engine_size = sum(op_types)
        paddle_op_size = op_size - trt_engine_size
        self.assertEqual(
            trt_engine_num,
            trt_engine_size,
            f"Expected trt_engine_num is {trt_engine_num}, but got {trt_engine_size}!",
        )
        self.assertEqual(
            paddle_op_num,
            paddle_op_size,
            f"Expected paddle_op_num is {paddle_op_num}, but got {paddle_op_size}!",
        )

    def inference_config_str(self, config: paddle_infer.Config) -> str:
        dic = {}
        enable_trt = config.tensorrt_engine_enabled()
        trt_precision = config.tensorrt_precision_mode()
        trt_dynamic_shape = config.tensorrt_dynamic_shape_enabled()
        if enable_trt:
            dic["use_trt"] = True
            dic["trt_precision"] = trt_precision
            dic["use_dynamic_shape"] = trt_dynamic_shape
        else:
            dic["use_trt"] = False
        return str(dic)

    def run_test(
        self,
        quant=False,
        explicit=False,
        skip_baseline=False,
        run_pir=False,
        *args,
        **kwargs,
    ):
        all_passes = True

        def random_to_skip():
            if self.skip_rng.random() < self.num_percent_cases:
                return False
            return True

        for prog_config in self.sample_program_configs(*args, **kwargs):
            paddle.enable_static()
            if random_to_skip():
                continue
            # if program is invalid, we should skip that cases.
            if not self.is_program_valid(prog_config):
                continue
            if run_pir and os.name != 'nt' and (not os.getenv('WITH_XPU')):
                # get pir program from old program
                main_program_desc, util_program = create_fake_model(
                    prog_config, run_pir=True
                )
                # transform program from old ir to new ir
                startup_program = pir.translate_to_pir(util_program.desc)
                pir_main_program = pir.translate_to_pir(main_program_desc)
                with paddle.pir_utils.IrGuard():
                    with paddle.static.program_guard(
                        pir_main_program, startup_program
                    ):
                        feed_dict = {}
                        feed_data = prog_config.get_feed_data()
                        for key, value in feed_data.items():

                            feed_dict[key] = value['data']

                        place = (
                            paddle.CUDAPlace(0)
                            if paddle.is_compiled_with_cuda()
                            else paddle.CPUPlace()
                        )
                        out_put = pir_main_program.get_output_value_by_name(
                            prog_config.outputs[0]
                        )
                        in_put = out_put.get_defining_op().operand_source(0)
                        exe = paddle.static.Executor(place)
                        exe.run(startup_program)
                        static_out = exe.run(
                            pir_main_program,
                            feed=feed_dict,
                            fetch_list=[in_put],
                        )

                        for (
                            pred_config,
                            nodes_num,
                            threshold,
                        ) in self.sample_predictor_configs(
                            prog_config, run_pir=True
                        ):
                            if os.path.exists(self.cache_dir):
                                shutil.rmtree(self.cache_dir)
                            if isinstance(threshold, float):
                                atol = threshold
                                rtol = 1e-4
                            elif isinstance(threshold, (list, tuple)):
                                atol = threshold[0]
                                rtol = threshold[1]
                            else:
                                raise NotImplementedError

                            is_fp8 = (
                                pred_config.tensorrt_precision_mode()
                                == paddle_infer.PrecisionType.Int8
                            )
                            if (not is_fp8 and quant) or (
                                is_fp8 and not (quant or explicit)
                            ):
                                continue

                            if explicit:
                                pred_config.enable_tensorrt_explicit_quantization()
                                self.assertTrue(
                                    pred_config.tensorrt_explicit_quantization_enabled()
                                )

                            ignore_flag = False
                            for teller, reason, note in self.ignore_cases:
                                if teller(prog_config, pred_config):
                                    ignore_flag = True
                                    if (
                                        reason
                                        == IgnoreReasons.TRT_NOT_IMPLEMENTED
                                    ):
                                        self.ignore_log(
                                            f"[TRT_NOT_IMPLEMENTED] {note} vs {self.inference_config_str(pred_config)}"
                                        )
                                    elif (
                                        reason == IgnoreReasons.TRT_NOT_SUPPORT
                                    ):
                                        self.ignore_log(
                                            f"[TRT_NOT_SUPPORT] {note} vs {self.inference_config_str(pred_config)}"
                                        )
                                    else:
                                        raise NotImplementedError
                                    break
                            if ignore_flag:
                                continue
                            attrs = [
                                prog_config.ops[i].attrs
                                for i in range(len(prog_config.ops))
                            ]
                            dynamic_shape = self.generate_dynamic_shape()

                            main_program_desc, util_program = create_fake_model(
                                prog_config,
                                run_pir=True,
                                dynamic_shape=dynamic_shape,
                            )
                            # transform program from old ir to new ir
                            startup_program = pir.translate_to_pir(
                                util_program.desc
                            )
                            pir_main_program = pir.translate_to_pir(
                                main_program_desc
                            )

                            inputs = []
                            first_key = next(iter(prog_config.get_feed_data()))
                            input_data_type = prog_config.get_feed_data()[
                                first_key
                            ]['data'].dtype
                            if not self.dynamic_shape.min_input_shape:
                                continue
                            for key in dynamic_shape.min_input_shape.keys():
                                input_config = Input(
                                    min_input_shape=tuple(
                                        self.dynamic_shape.min_input_shape[key]
                                    ),
                                    optim_input_shape=tuple(
                                        self.dynamic_shape.opt_input_shape[key]
                                    ),
                                    max_input_shape=tuple(
                                        self.dynamic_shape.max_input_shape[key]
                                    ),
                                    input_data_type=str(input_data_type),
                                )
                                inputs.append(input_config)
                            trt_config = TensorRTConfig(inputs=inputs)
                            trt_config.input_data_type = input_data_type
                            trt_program = self.transform_to_trt_program(
                                pir_main_program, trt_config
                            )

                            assert any(
                                op.name() == "pd_op.tensorrt_engine"
                                for op in trt_program.global_block().ops
                            ), "trt_program does not contain any tensorrt_engine ops."

                            feed_data = prog_config.get_feed_data()
                            for key, value in feed_data.items():
                                feed_dict[key] = value['data']
                            trt_output = exe.run(
                                trt_program, feed=feed_dict, fetch_list=[in_put]
                            )
                            self.assert_tensors_near(
                                atol, rtol, trt_output, static_out
                            )
                            paddle.framework.set_flags(
                                {"FLAGS_trt_min_group_size": 3}
                            )
            else:
                with paddle.pir_utils.OldIrGuard():
                    main_program_desc, util_program = create_fake_model(
                        prog_config
                    )
                    model = main_program_desc.serialize_to_string()

                    place = paddle.base.CPUPlace()
                    executor = paddle.base.Executor(place)
                    scope = paddle.base.Scope()
                    with paddle.base.scope_guard(scope):
                        executor.run(util_program)
                        params = scope.find_var("out_var_0").get_bytes()
                if quant:
                    with paddle.pir_utils.OldIrGuard():
                        model, params = create_quant_model(model, params)

                if not skip_baseline:
                    # baseline: gpu run, we only test float32
                    gpu_config = self.create_inference_config(use_trt=False)
                    baseline_result = self.run_test_config(
                        model,
                        params,
                        prog_config,
                        gpu_config,
                        prog_config.get_feed_data(),
                    )
                    self.success_log(f"baseline program_config: {prog_config}")

                for (
                    pred_config,
                    nodes_num,
                    threshold,
                ) in self.sample_predictor_configs(prog_config):
                    if os.path.exists(self.cache_dir):
                        shutil.rmtree(self.cache_dir)

                    if isinstance(threshold, float):
                        atol = threshold
                        rtol = 1e-4
                    elif isinstance(threshold, (list, tuple)):
                        atol = threshold[0]
                        rtol = threshold[1]
                    else:
                        raise NotImplementedError

                    is_fp8 = (
                        pred_config.tensorrt_precision_mode()
                        == paddle_infer.PrecisionType.Int8
                    )
                    if (not is_fp8 and quant) or (
                        is_fp8 and not (quant or explicit)
                    ):
                        continue

                    if explicit:
                        pred_config.enable_tensorrt_explicit_quantization()
                        self.assertTrue(
                            pred_config.tensorrt_explicit_quantization_enabled()
                        )

                    ignore_flag = False
                    for teller, reason, note in self.ignore_cases:
                        if teller(prog_config, pred_config):
                            ignore_flag = True
                            if reason == IgnoreReasons.TRT_NOT_IMPLEMENTED:
                                self.ignore_log(
                                    f"[TRT_NOT_IMPLEMENTED] {note} vs {self.inference_config_str(pred_config)}"
                                )
                            elif reason == IgnoreReasons.TRT_NOT_SUPPORT:
                                self.ignore_log(
                                    f"[TRT_NOT_SUPPORT] {note} vs {self.inference_config_str(pred_config)}"
                                )
                            else:
                                raise NotImplementedError
                            break

                    if ignore_flag:
                        continue

                    try:
                        with paddle.pir_utils.OldIrGuard():
                            main_program_desc, util_program = create_fake_model(
                                prog_config
                            )
                            model = main_program_desc.serialize_to_string()
                            place = paddle.base.CPUPlace()
                            executor = paddle.base.Executor(place)
                            scope = paddle.base.Scope()
                            with paddle.base.scope_guard(scope):
                                executor.run(util_program)
                                params = scope.find_var("out_var_0").get_bytes()
                        if quant:
                            model, params = create_quant_model(model, params)
                        feed_data = prog_config.get_feed_data()
                        pred_config_deserialize = paddle_infer.Config(
                            pred_config
                        )
                        trt_result = self.run_test_config(
                            model, params, prog_config, pred_config, feed_data
                        )
                        self.assert_tensors_near(
                            atol, rtol, trt_result, baseline_result
                        )
                        trt_engine_num, paddle_op_num = nodes_num
                        self.assert_op_size(trt_engine_num, paddle_op_num)
                        # deserialize test
                        if trt_engine_num > 0:
                            self.run_test_config(
                                model,
                                params,
                                prog_config,
                                pred_config_deserialize,
                                feed_data,
                            )

                        self.success_log(f"program_config: {prog_config}")
                        self.success_log(
                            f"predictor_config: {self.inference_config_str(pred_config)}"
                        )
                    except Exception as e:
                        self.fail_log(f"program_config: {prog_config}")
                        self.fail_log(
                            f"predictor_config: {self.inference_config_str(pred_config)}"
                        )
                        self.fail_log(f"\033[1;31m ERROR INFO: {e}\033[0m")
                        all_passes = False

        self.assertTrue(all_passes)

    # TODO(wilber): just for backward compatible
    def add_skip_case(
        self,
        teller: list[Callable[[ProgramConfig, paddle_infer.Config], bool]],
        reason: IgnoreReasons,
        note: str,
    ):
        self.ignore_cases.append((teller, reason, note))


class CutlassAutoScanTest(AutoScanTest):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run_test(self, quant=False, *args, **kwargs):
        status = True

        for prog_config in self.sample_program_configs(*args, **kwargs):
            # if program is invalid, we should skip that cases.
            if not self.is_program_valid(prog_config):
                continue

            with paddle.pir_utils.OldIrGuard():
                main_program_desc, util_program = create_fake_model(prog_config)
                model = main_program_desc.serialize_to_string()
                place = paddle.base.CPUPlace()
                executor = paddle.base.Executor(place)
                scope = paddle.base.Scope()
                with paddle.base.scope_guard(scope):
                    executor.run(util_program)
                    params = scope.find_var("out_var_0").get_bytes()

            feed_data = {}
            for name, tensor_config in prog_config.inputs.items():
                feed_data[name] = {
                    'data': tensor_config.data,
                    'lod': tensor_config.lod,
                }
            results: list[dict[str, np.ndarray]] = []

            # baseline: gpu no ir_optim run
            base_config = self.create_inference_config(
                ir_optim=False, use_gpu=True
            )
            logging.info('RUN program_config: ' + str(prog_config))
            results.append(
                self.run_test_config(
                    model, params, prog_config, base_config, feed_data
                )
            )
            self.success_log('RUN_GPU_BASELINE done')

            for pred_config, (atol, rtol) in self.sample_predictor_configs(
                prog_config
            ):
                # skip info
                ignore_flag = False
                for ignore_info in self.ignore_cases:
                    if ignore_info[0](prog_config, pred_config):
                        ignore_flag = True
                        if (
                            ignore_info[1]
                            == IgnoreReasons.CUTLASS_ACCURACY_ERROR
                        ):
                            self.ignore_log(
                                "[CUTLASS_ACCURACY_ERROR] "
                                + ignore_info[2]
                                + ' '
                                + ' vs '
                                + self.inference_config_str(pred_config)
                            )
                        else:
                            raise NotImplementedError
                        break

                if os.path.exists(self.cache_dir):
                    shutil.rmtree(self.cache_dir)
                if not os.path.exists(self.cache_dir):
                    os.mkdir(self.cache_dir)

                try:
                    results.append(
                        self.run_test_config(
                            model, params, prog_config, pred_config, feed_data
                        )
                    )
                    self.assert_tensors_near(
                        atol, rtol, results[-1], results[0]
                    )
                except Exception as e:
                    self.fail_log(
                        self.inference_config_str(pred_config)
                        + f'\033[1;31m \nERROR INFO: {e}\033[0m'
                    )
                    if not ignore_flag:
                        status = False
                    continue
                self.success_log(
                    'RUN predictor_config '
                    + self.inference_config_str(pred_config)
                    + ' done'
                )

        self.assertTrue(status)

    def inference_config_str(self, config) -> str:
        dic = {}
        enable_gpu = config.use_gpu()
        dic['use_gpu'] = enable_gpu
        return str(dic)
