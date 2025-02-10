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

from __future__ import annotations

import importlib
import inspect
import logging
import os
import sys
import unittest
from contextlib import contextmanager
from enum import Flag, auto
from functools import wraps
from pathlib import Path

import paddle
from paddle import get_flags, set_flags, static
from paddle.jit.api import sot_mode_guard
from paddle.jit.sot.opcode_translator.executor.executor_cache import (
    OpcodeExecutorCache,
)
from paddle.jit.sot.utils.envs import min_graph_size_guard
from paddle.utils.environments import (
    BooleanEnvironmentVariable,
    EnvironmentVariableGuard,
)

"""
# Usage:
class MyTest(Dy2StTestBase):
    @set_to_static_mode(
        ToStaticMode.AST | ToStaticMode.SOT
    )
    @set_ir_mode(IrMode.LEGACY_IR | IrMode.PT | IrMode.PIR)
    def test_case1(self):
        raise ValueError("MyTest 1")

    def test_case2(self):
        raise ValueError("MyTest 2")


class MyTest2(MyTest):
    def test_case1(self):
        raise ValueError("MyTest2 1")
"""

logger = logging.getLogger("Dygraph to static utils")
logger.setLevel(logging.WARNING)

ENV_ENABLE_PIR_WITH_PT_IN_DY2ST = BooleanEnvironmentVariable(
    "FLAGS_enable_pir_with_pt_in_dy2st", True
)
ENV_EXE_SEQUENTIAL_RUN = BooleanEnvironmentVariable(
    "FLAGS_new_executor_sequential_run", False
)


class ToStaticMode(Flag):
    AST = auto()
    SOT = auto()
    # SOT with MIN_GRAPH_SIZE=10, we only test SOT_MGS10 + LEGACY_IR to avoid regression
    SOT_MGS10 = auto()

    def lower_case_name(self):
        return self.name.lower()


class IrMode(Flag):
    LEGACY_IR = auto()
    # pir translator mode, Reference link: https://github.com/PaddlePaddle/community/blob/master/pfcc/paddle-code-reading/IR_Dialect/program_translator.md
    PT = auto()
    # using native pir api mode
    PIR = auto()

    def lower_case_name(self):
        return self.name.lower()


DEFAULT_TO_STATIC_MODE = (
    ToStaticMode.AST | ToStaticMode.SOT | ToStaticMode.SOT_MGS10
)
DEFAULT_IR_MODE = IrMode.PT | IrMode.PIR

DISABLED_TO_STATIC_TEST_FILES = {
    ToStaticMode.AST: [],
    ToStaticMode.SOT: [],
    ToStaticMode.SOT_MGS10: [],
}

DISABLED_IR_TEST_FILES = {
    IrMode.LEGACY_IR: [],
    IrMode.PT: [
        "test_tensor_hook",
    ],
    IrMode.PIR: [],
}


@contextmanager
def pir_dygraph_guard():
    in_dygraph_mode = paddle.in_dynamic_mode()
    with paddle.pir_utils.IrGuard():
        if in_dygraph_mode:
            paddle.disable_static()
        yield


@contextmanager
def legacy_ir_dygraph_guard():
    in_dygraph_mode = paddle.in_dynamic_mode()
    with paddle.pir_utils.OldIrGuard():
        if in_dygraph_mode:
            paddle.disable_static()
        yield


def to_ast_test(fn):
    """
    convert run AST
    """

    @wraps(fn)
    def impl(*args, **kwargs):
        logger.info("[AST] running AST")
        with sot_mode_guard(False):
            fn(*args, **kwargs)

    return impl


def to_sot_test(fn):
    """
    convert run SOT
    """

    @wraps(fn)
    def impl(*args, **kwargs):
        logger.info("[SOT] running SOT (MIN_GRAPH_SIZE=0)")

        OpcodeExecutorCache().clear()
        with sot_mode_guard(True):
            with min_graph_size_guard(0):
                fn(*args, **kwargs)

    return impl


def to_sot_mgs10_test(fn):
    """
    convert run SOT and MIN_GRAPH_SIZE=10
    """

    @wraps(fn)
    def impl(*args, **kwargs):
        logger.info("[SOT_MGS10] running SOT (MIN_GRAPH_SIZE=10)")

        OpcodeExecutorCache().clear()
        with sot_mode_guard(True):
            with min_graph_size_guard(10):
                fn(*args, **kwargs)

    return impl


def to_legacy_ir_test(fn):
    @wraps(fn)
    def impl(*args, **kwargs):
        logger.info("[LEGACY_IR] running legacy ir")
        with legacy_ir_dygraph_guard():
            pt_in_dy2st_flag = ENV_ENABLE_PIR_WITH_PT_IN_DY2ST.name
            original_flag_value = get_flags(pt_in_dy2st_flag)[pt_in_dy2st_flag]
            with EnvironmentVariableGuard(
                ENV_ENABLE_PIR_WITH_PT_IN_DY2ST, False
            ):
                try:
                    set_flags({pt_in_dy2st_flag: False})
                    ir_outs = fn(*args, **kwargs)
                finally:
                    set_flags({pt_in_dy2st_flag: original_flag_value})
                return ir_outs

    return impl


def to_pt_test(fn):
    @wraps(fn)
    def impl(*args, **kwargs):
        logger.info("[PT] running PT")
        with legacy_ir_dygraph_guard():
            pt_in_dy2st_flag = ENV_ENABLE_PIR_WITH_PT_IN_DY2ST.name
            original_flag_value = get_flags(pt_in_dy2st_flag)[pt_in_dy2st_flag]
            if os.environ.get('FLAGS_use_stride_kernel', False):
                return
            with static.scope_guard(static.Scope()):
                with static.program_guard(static.Program()):
                    with EnvironmentVariableGuard(
                        ENV_ENABLE_PIR_WITH_PT_IN_DY2ST, True
                    ):
                        try:
                            set_flags({pt_in_dy2st_flag: True})
                            ir_outs = fn(*args, **kwargs)
                        finally:
                            set_flags({pt_in_dy2st_flag: original_flag_value})
        return ir_outs

    return impl


def to_pir_test(fn):
    @wraps(fn)
    def impl(*args, **kwargs):
        logger.info("[PIR] running pir")
        with pir_dygraph_guard():
            ir_outs = fn(*args, **kwargs)
        return ir_outs

    return impl


# Metaclass and BaseClass
class Dy2StTestMeta(type):
    TO_STATIC_HANDLER_MAP = {
        ToStaticMode.AST: to_ast_test,
        ToStaticMode.SOT: to_sot_test,
        ToStaticMode.SOT_MGS10: to_sot_mgs10_test,
    }

    IR_HANDLER_MAP = {
        IrMode.LEGACY_IR: to_legacy_ir_test,
        IrMode.PT: to_pt_test,
        IrMode.PIR: to_pir_test,
    }

    def __new__(cls, name, bases, attrs):
        module_name = attrs["__module__"]
        filepath = sys.modules[module_name].__file__
        assert filepath
        filename = Path(filepath).stem
        new_attrs = {}
        original_test_cases = {
            key: value
            for key, value in attrs.items()
            if key.startswith("test") and inspect.isfunction(value)
        }
        logger.info(f"[creating {name}]")
        new_attrs.update(
            {
                key: value
                for key, value in attrs.items()
                if key not in original_test_cases
            }
        )
        for fn_name, fn in original_test_cases.items():
            logger.info(f"Generating {fn_name}")
            # Disable inherited test cases
            for base in bases:
                for attr in dir(base):
                    if attr.startswith(f"{fn_name}__"):
                        new_attrs[attr] = None
            fn_to_static_modes = getattr(
                fn, "to_static_mode", DEFAULT_TO_STATIC_MODE
            )
            fn_ir_modes = getattr(fn, "ir_mode", DEFAULT_IR_MODE)
            fn_disabled_test_cases = getattr(fn, "disabled_test_cases", [])
            logger.info(f"fn_to_static_modes: {fn_to_static_modes}")
            logger.info(f"fn_ir_modes: {fn_ir_modes}")
            logger.info(f"fn_disabled_test_cases: {fn_disabled_test_cases}")
            # Get all valid test cases with to_static_mode and ir_mode
            to_static_with_ir_modes = [
                (to_static_mode, ir_mode)
                for to_static_mode in ToStaticMode
                for ir_mode in IrMode
                if to_static_mode & fn_to_static_modes and ir_mode & fn_ir_modes
            ]
            # Filter out disabled test cases by decorator
            to_static_with_ir_modes = list(
                filter(
                    lambda flags: (flags not in fn_disabled_test_cases),
                    to_static_with_ir_modes,
                )
            )
            # Filter out disabled test cases by file
            to_static_with_ir_modes = list(
                filter(
                    lambda flags: (
                        filename not in DISABLED_TO_STATIC_TEST_FILES[flags[0]]
                        and filename not in DISABLED_IR_TEST_FILES[flags[1]]
                    ),
                    to_static_with_ir_modes,
                )
            )
            # Generate all test cases
            for to_static_mode, ir_mode in to_static_with_ir_modes:
                if (
                    to_static_mode == ToStaticMode.SOT_MGS10
                    and ir_mode != IrMode.PIR
                ):
                    # SOT_MGS10 only test with PIR
                    continue
                new_attrs[
                    Dy2StTestMeta.test_case_name(
                        fn_name, to_static_mode, ir_mode
                    )
                ] = Dy2StTestMeta.convert_test_case(fn, to_static_mode, ir_mode)
        return type.__new__(cls, name, bases, new_attrs)

    @staticmethod
    def test_case_name(original_name: str, to_static_mode, ir_mode):
        return f"{original_name}__{to_static_mode.lower_case_name()}_{ir_mode.lower_case_name()}"

    @staticmethod
    def convert_test_case(fn, to_static_mode, ir_mode):
        fn = Dy2StTestMeta.IR_HANDLER_MAP[ir_mode](fn)
        fn = Dy2StTestMeta.TO_STATIC_HANDLER_MAP[to_static_mode](fn)
        return fn


class Dy2StTestBase(unittest.TestCase, metaclass=Dy2StTestMeta):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


# Base decorators
def set_to_static_mode(mode: ToStaticMode):
    def decorator(fn):
        fn.to_static_mode = mode
        return fn

    return decorator


def set_ir_mode(mode: IrMode):
    def decorator(fn):
        fn.ir_mode = mode
        return fn

    return decorator


def disable_test_case(flags: tuple[ToStaticMode, IrMode]):
    def decorator(fn):
        disabled_test_cases = getattr(fn, "disabled_test_cases", [])
        disabled_test_cases.append(flags)
        fn.disabled_test_cases = disabled_test_cases
        return fn

    return decorator


# Sugar decorators
# These decorators can be simply composed by base decorators
def test_ast_only(fn):
    fn = set_to_static_mode(ToStaticMode.AST)(fn)
    return fn


def test_sot_only(fn):
    fn = set_to_static_mode(ToStaticMode.SOT | ToStaticMode.SOT_MGS10)(fn)
    return fn


def test_legacy_only(fn):
    fn = set_ir_mode(IrMode.LEGACY_IR)(fn)
    return fn


def test_pt_only(fn):
    fn = set_ir_mode(IrMode.PT)(fn)
    return fn


def test_pir_only(fn):
    fn = set_ir_mode(IrMode.PIR)(fn)
    return fn


def test_legacy_and_pt(fn):
    fn = set_ir_mode(IrMode.LEGACY_IR | IrMode.PT)(fn)
    return fn


def test_pt_and_pir(fn):
    fn = set_ir_mode(IrMode.PT | IrMode.PIR)(fn)
    return fn


def test_legacy_and_pir(fn):
    fn = set_ir_mode(IrMode.LEGACY_IR | IrMode.PIR)(fn)
    return fn


def test_legacy_and_pt_and_pir(fn):
    fn = set_ir_mode(IrMode.LEGACY_IR | IrMode.PT | IrMode.PIR)(fn)
    return fn


# Some decorators for save CI time
def test_default_mode_only(fn):
    # Some unittests has high time complexity, we only test them with default mode
    fn = set_to_static_mode(ToStaticMode.SOT)(fn)
    fn = set_ir_mode(IrMode.PT)(fn)
    return fn


def test_default_and_pir(fn):
    # Some unittests has high time complexity, we only test them with default mode
    fn = set_to_static_mode(ToStaticMode.SOT)(fn)
    fn = set_ir_mode(IrMode.PT | IrMode.PIR)(fn)
    return fn


def test_sot_mgs0_only(fn):
    fn = set_to_static_mode(ToStaticMode.SOT)(fn)
    return fn


# For debug
def show_all_test_cases(test_class):
    logger.info(f"[showing {test_class.__name__}]")
    for attr in dir(test_class):
        if attr.startswith("test"):
            fn = getattr(test_class, attr)
            logger.info(f"{attr}: {fn}")


# Other utilities
def import_module_from_path(module_name, module_path):
    """A better way to import module from other directory than using sys.path.append"""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def import_legacy_test_utils():
    test_root = Path(__file__).parent.parent
    legacy_test_utils_path = test_root / "legacy_test/utils.py"
    legacy_test_utils = import_module_from_path(
        "legacy_test_utils", legacy_test_utils_path
    )
    return legacy_test_utils


legacy_test_utils = import_legacy_test_utils()
dygraph_guard = legacy_test_utils.dygraph_guard
static_guard = legacy_test_utils.static_guard


@contextmanager
def enable_to_static_guard(flag: bool):
    program_translator = paddle.jit.api.ProgramTranslator()
    original_flag_value = program_translator.enable_to_static
    program_translator.enable(flag)
    try:
        yield
    finally:
        program_translator.enable(original_flag_value)


@contextmanager
def exe_sequential_run_guard(value: bool):
    exe_sequential_run_flag = ENV_EXE_SEQUENTIAL_RUN.name
    original_flag_value = paddle.get_flags(exe_sequential_run_flag)[
        exe_sequential_run_flag
    ]
    with EnvironmentVariableGuard(ENV_EXE_SEQUENTIAL_RUN, value):
        try:
            set_flags({exe_sequential_run_flag: value})
            yield
        finally:
            set_flags({exe_sequential_run_flag: original_flag_value})
