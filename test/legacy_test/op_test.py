#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import functools
import inspect
import os
import pathlib
import random
import struct
import sys
import unittest
import warnings
from collections import defaultdict
from contextlib import contextmanager
from copy import copy

import numpy as np
from auto_parallel_op_test import (
    dump_test_info,
    gen_auto_parallel_test_file,
    get_subprocess_command,
    get_subprocess_runtime_envs,
    get_test_info_and_generated_test_path,
    is_ban_auto_parallel_test,
    run_subprocess,
)
from op import Operator
from prim_op_test import OpTestUtils, PrimForwardChecker, PrimGradChecker
from testsuite import append_input_output, append_loss_ops, create_op, set_input

# Add test/legacy and test to sys.path
legacy_test_dir = pathlib.Path(__file__).parent  # test/legacy_test
test_dir = legacy_test_dir.parent  # test
sys.path.append(str(legacy_test_dir.absolute()))
sys.path.append(str(test_dir.absolute()))

from white_list import (
    check_shape_white_list,
    compile_vs_runtime_white_list,
    no_check_set_white_list,
    no_grad_set_white_list,
    op_accuracy_white_list,
    op_threshold_white_list,
)

import paddle
from legacy_test.utils import pir_executor_guard, static_guard
from paddle import base
from paddle.autograd.ir_backward import grad as ir_grad
from paddle.base import Scope, core, unique_name
from paddle.base.backward import append_backward
from paddle.base.executor import Executor, scope_guard
from paddle.base.framework import (
    OpProtoHolder,
    Program,
    _current_expected_place,
    canonicalize_attrs,
    get_flags,
    set_flags,
)
from paddle.base.wrapped_decorator import signature_safe_contextmanager


@signature_safe_contextmanager
def paddle_static_guard():
    try:
        paddle.enable_static()
        yield
    finally:
        paddle.disable_static()


def check_out_dtype(api_fn, in_specs, expect_dtypes, target_index=0, **configs):
    """
    Determines whether dtype of output tensor is as expected.

    Args:
        api_fn(callable):  paddle api function
        in_specs(list[tuple]): list of shape and dtype information for constructing input tensor of api_fn, such as [(shape, dtype), (shape, dtype)].
        expect_dtypes(list[str]): expected dtype of output tensor.
        target_index(int): indicate which one from in_specs to infer the dtype of output.
        config(dict): other arguments of paddle api function

    Example:
        check_out_dtype(base.layers.pad_constant_like, [([2,3,2,3], 'float64'), ([1, 3, 1,3], )], ['float32', 'float64', 'int64'], target_index=1, pad_value=0.)

    """
    with paddle.pir_utils.OldIrGuard():
        for i, expect_dtype in enumerate(expect_dtypes):
            with paddle.static.program_guard(paddle.static.Program()):
                input_t = []
                for index, spec in enumerate(in_specs):
                    if len(spec) == 1:
                        shape = spec[0]
                        dtype = (
                            expect_dtype if target_index == index else 'float32'
                        )
                    elif len(spec) == 2:
                        shape, dtype = spec
                    else:
                        raise ValueError(
                            f"Value of in_specs[{index}] should contains two elements: [shape, dtype]"
                        )
                    input_t.append(
                        paddle.static.data(
                            name=f'data_{index}', shape=shape, dtype=dtype
                        )
                    )

                out = api_fn(*input_t, **configs)
                out_dtype = base.data_feeder.convert_dtype(out.dtype)

                if out_dtype != expect_dtype:
                    raise ValueError(
                        f"Expected out.dtype is {expect_dtype}, but got {out_dtype} from {api_fn.__name__}."
                    )


def _set_use_system_allocator(value=None):
    USE_SYSTEM_ALLOCATOR_FLAG = "FLAGS_use_system_allocator"
    old_value = core.globals()[USE_SYSTEM_ALLOCATOR_FLAG]
    value = old_value if value is None else value
    core.globals()[USE_SYSTEM_ALLOCATOR_FLAG] = value
    return old_value


def randomize_probability(batch_size, class_num, dtype='float32'):
    prob = np.random.uniform(0.1, 1.0, size=(batch_size, class_num)).astype(
        dtype
    )
    prob_sum = prob.sum(axis=1)
    for i in range(len(prob)):
        prob[i] /= prob_sum[i]
    return prob


def get_numeric_gradient(
    place,
    scope,
    op,
    inputs,
    input_to_check,
    output_names,
    delta=0.005,
    in_place=False,
):
    # FIXME: change this method by compile time concepts
    set_input(scope, op, inputs, place)

    def product(dim):
        return functools.reduce(lambda a, b: a * b, dim, 1)

    tensor_to_check = scope.find_var(input_to_check).get_tensor()
    tensor_size = product(tensor_to_check.shape())
    tensor_to_check_dtype = tensor_to_check._dtype()
    if tensor_to_check_dtype == paddle.float32:
        tensor_to_check_dtype = np.float32
    elif tensor_to_check_dtype == paddle.float64:
        tensor_to_check_dtype = np.float64
    elif tensor_to_check_dtype == paddle.float16:
        tensor_to_check_dtype = np.float16
        # set delta as np.float16, will automatic convert to float32, float64
        delta = np.array(delta).astype(np.float16)
    elif tensor_to_check_dtype == paddle.bfloat16:
        tensor_to_check_dtype = np.float32
    elif tensor_to_check_dtype == paddle.complex64:
        tensor_to_check_dtype = np.complex64
    elif tensor_to_check_dtype == paddle.complex128:
        tensor_to_check_dtype = np.complex128
    else:
        raise ValueError(
            "Not supported data type "
            + str(tensor_to_check_dtype)
            + ", tensor name : "
            + str(input_to_check)
        )

    def get_output():
        sum = []
        op.run(scope, place)
        for output_name in output_names:
            output_numpy = np.array(scope.find_var(output_name).get_tensor())
            # numpy.dtype does not have bfloat16, thus we use numpy.uint16 to
            # store bfloat16 data, and need to be converted to float to check
            # the floating precision.
            if tensor_to_check._dtype() == paddle.bfloat16:
                output_numpy = convert_uint16_to_float(output_numpy)
            sum.append(output_numpy.astype(tensor_to_check_dtype).mean())
        return tensor_to_check_dtype(np.array(sum).sum() / len(output_names))

    gradient_flat = np.zeros(shape=(tensor_size,), dtype=tensor_to_check_dtype)

    def __get_elem__(tensor, i):
        if tensor_to_check_dtype == np.float16:
            numpy_tensor = np.array(tensor).astype(np.float16)
            numpy_tensor = numpy_tensor.flatten()
            return numpy_tensor[i]
        elif tensor_to_check._dtype() == paddle.bfloat16:
            numpy_tensor = np.array(tensor).astype(np.uint16)
            numpy_tensor = numpy_tensor.flatten()
            return struct.unpack(
                '<f',
                struct.pack('<I', np.uint32(numpy_tensor[i]) << np.uint32(16)),
            )[0]
        elif tensor_to_check_dtype == np.float32:
            return tensor._get_float_element(i)
        elif tensor_to_check_dtype == np.float64:
            return tensor._get_double_element(i)
        elif tensor_to_check_dtype == np.complex64:
            return tensor._get_complex64_element(i)
        elif tensor_to_check_dtype == np.complex128:
            return tensor._get_complex128_element(i)
        else:
            raise TypeError(
                f"Unsupported test data type {tensor_to_check_dtype}."
            )

    def __set_elem__(tensor, i, e):
        if tensor_to_check_dtype == np.float16:
            numpy_tensor = np.array(tensor).astype(np.float16)
            shape = numpy_tensor.shape
            numpy_tensor = numpy_tensor.flatten()
            numpy_tensor[i] = e
            numpy_tensor = numpy_tensor.reshape(shape)
            tensor.set(numpy_tensor, place)
        elif tensor_to_check._dtype() == paddle.bfloat16:
            numpy_tensor = np.array(tensor).astype(np.uint16)
            shape = numpy_tensor.shape
            numpy_tensor = numpy_tensor.flatten()
            numpy_tensor[i] = np.uint16(copy_bits_from_float_to_uint16(e))
            numpy_tensor = numpy_tensor.reshape(shape)
            tensor.set(numpy_tensor, place)
        elif tensor_to_check_dtype == np.float32:
            tensor._set_float_element(i, e)
        elif tensor_to_check_dtype == np.float64:
            tensor._set_double_element(i, e)
        elif tensor_to_check_dtype == np.complex64:
            return tensor._set_complex64_element(i, e)
        elif tensor_to_check_dtype == np.complex128:
            return tensor._set_complex128_element(i, e)
        else:
            raise TypeError(
                f"Unsupported test data type {tensor_to_check_dtype}."
            )

    # we only compute gradient of one element each time.
    # we use a for loop to compute the gradient of every element.
    for i in range(tensor_size):
        if in_place:
            set_input(scope, op, inputs, place)

        # get one input element throw it's index i.
        origin = __get_elem__(tensor_to_check, i)
        # add delta to it, run op and then get the sum of the result tensor.
        x_pos = origin + delta
        __set_elem__(tensor_to_check, i, x_pos)
        y_pos = get_output()

        if tensor_to_check_dtype in [np.complex64, np.complex128]:
            if in_place:
                set_input(scope, op, inputs, place)
            x_pos_j = origin + 1j * delta
            __set_elem__(tensor_to_check, i, x_pos_j)
            y_pos_j = get_output()

        if in_place:
            set_input(scope, op, inputs, place)

        x_neg = origin - delta
        __set_elem__(tensor_to_check, i, x_neg)
        y_neg = get_output()

        if tensor_to_check_dtype in [np.complex64, np.complex128]:
            if in_place:
                set_input(scope, op, inputs, place)

            x_neg_j = origin - 1j * delta
            __set_elem__(tensor_to_check, i, x_neg_j)
            y_neg_j = get_output()

        __set_elem__(tensor_to_check, i, origin)

        if tensor_to_check_dtype in [np.complex64, np.complex128]:
            # always assume real output, because this function has
            # no input for dl/di, though it should do. so there di will be zero

            # TODO: Here is a trick to be consistent with the existing OpTest, it
            # need to support variable gradients input
            f_ajoint = np.array(1 + 0j)
            df_over_dr = (y_pos - y_neg) / delta / 2
            df_over_di = (y_pos_j - y_neg_j) / delta / 2

            dl_over_du, dl_over_dv = f_ajoint.real, f_ajoint.imag

            du_over_dr, dv_over_dr = df_over_dr.real, df_over_dr.imag

            du_over_di, dv_over_di = df_over_di.real, df_over_di.imag

            dl_over_dr = np.sum(
                dl_over_du * du_over_dr + dl_over_dv * dv_over_dr
            )
            dl_over_di = np.sum(
                dl_over_du * du_over_di + dl_over_dv * dv_over_di
            )
            gradient_flat[i] = dl_over_dr + 1j * dl_over_di
        else:
            df_over_dr = y_pos - y_neg
            gradient_flat[i] = df_over_dr / delta / 2

        __set_elem__(tensor_to_check, i, origin)

    return gradient_flat.reshape(tensor_to_check.shape())


def skip_check_grad_ci(reason=None):
    """Decorator to skip check_grad CI.

    Check_grad is required for Op test cases. However, there are some special
    cases that do not need to do check_grad. This decorator is used to skip the
    check_grad of the above cases.

    Note: the execution of unit test will not be skipped. It just avoids check_grad
    checking in tearDownClass method by setting a `no_need_check_grad` flag.

    Example:
        @skip_check_grad_ci(reason="For inference, check_grad is not required.")
        class TestInference(OpTest):
    """
    if not isinstance(reason, str):
        raise AssertionError("The reason for skipping check_grad is required.")

    def wrapper(cls):
        cls.no_need_check_grad = True
        return cls

    return wrapper


def skip_check_inplace_ci(reason=None):
    if not isinstance(reason, str):
        raise AssertionError(
            "The reason for skipping check_inplace is required."
        )

    def wrapper(cls):
        cls.no_need_check_inplace = True
        return cls

    return wrapper


def copy_bits_from_float_to_uint16(f):
    return struct.unpack('<I', struct.pack('<f', f))[0] >> 16


def convert_float_to_uint16(float_list, data_format="NCHW"):
    if data_format == "NHWC":
        float_list = np.transpose(float_list, [0, 3, 1, 2])

    new_output = []
    for x in np.nditer(float_list):
        new_output.append(np.uint16(copy_bits_from_float_to_uint16(x)))
    new_output = np.reshape(new_output, float_list.shape).view(np.uint16)

    if data_format == "NHWC":
        new_output = np.transpose(new_output, [0, 2, 3, 1])
    return new_output


def convert_uint16_to_float(in_list):
    in_list = np.asarray(in_list)
    out = np.vectorize(
        lambda x: struct.unpack(
            '<f', struct.pack('<I', np.uint32(x) << np.uint32(16))
        )[0],
        otypes=[np.float32],
    )(in_list.flat)
    return np.reshape(out, in_list.shape)


@contextmanager
def auto_parallel_test_guard(test_info_path, generated_test_file_path):
    test_info_file, generated_test_file = None, None
    if os.path.exists(test_info_path):
        raise OSError(
            f"{test_info_path} which stores test info should not exist. Please delete it firstly."
        )
    if os.path.exists(generated_test_file_path):
        raise OSError(
            f"{generated_test_file_path} which stores test code should not exist. Please delete it firstly."
        )
    test_info_file = open(test_info_path, "wb")
    generated_test_file = open(generated_test_file_path, "wb")
    try:
        yield
    finally:
        if test_info_file is not None:
            test_info_file.close()
        if generated_test_file is not None:
            generated_test_file.close()
        if os.path.exists(test_info_path):
            os.remove(test_info_path)
        if os.path.exists(generated_test_file_path):
            os.remove(generated_test_file_path)


class OpTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        '''Fix random seeds to remove randomness from tests'''
        cls._np_rand_state = np.random.get_state()
        cls._py_rand_state = random.getstate()
        cls.call_once = False
        cls.dtype = None
        cls.outputs = {}
        cls.input_shape_is_large = True
        cls.is_calc_ref = False
        cls.check_prim = False
        cls.check_prim_pir = False
        cls._check_cinn = False
        cls.check_pir_onednn = False

        # Todo(CZ): to be removed in future
        core._clear_prim_vjp_skip_default_ops()

        np.random.seed(123)
        random.seed(124)

        cls._use_system_allocator = _set_use_system_allocator(True)

    @classmethod
    def tearDownClass(cls):
        """Restore random seeds"""
        np.random.set_state(cls._np_rand_state)
        random.setstate(cls._py_rand_state)

        _set_use_system_allocator(cls._use_system_allocator)

        if hasattr(cls, 'check_prim') and os.getenv('FLAGS_prim_test_log'):
            print("check prim end!")

        def is_empty_grad_op(op_type):
            all_op_kernels = core._get_all_register_op_kernels()
            grad_op = op_type + '_grad'
            if grad_op in all_op_kernels.keys():
                if is_mkldnn_op_test():
                    grad_op_kernels = all_op_kernels[grad_op]
                    for grad_op_kernel in grad_op_kernels:
                        if 'MKLDNN' in grad_op_kernel:
                            return False
                else:
                    return False
            return True

        def is_xpu_op_test():
            return hasattr(cls, "use_xpu") and cls.use_xpu

        def is_mkldnn_op_test():
            return hasattr(cls, "use_mkldnn") and cls.use_mkldnn

        def is_rocm_op_test():
            return core.is_compiled_with_rocm()

        def is_custom_device_op_test():
            return hasattr(cls, "use_custom_device") and cls.use_custom_device

        def is_complex_test():
            return (
                hasattr(cls, "test_complex")
                and cls.test_complex
                or (cls.dtype in [np.complex64, np.complex128])
            )

        if not hasattr(cls, "op_type"):
            raise AssertionError(
                "This test do not have op_type in class attrs, "
                "please set self.__class__.op_type=the_real_op_type manually."
            )

        # case in NO_FP64_CHECK_GRAD_CASES and op in NO_FP64_CHECK_GRAD_OP_LIST should be fixed
        if (
            not hasattr(cls, "no_need_check_grad")
            and not is_empty_grad_op(cls.op_type)
            and not is_complex_test()
        ):
            if cls.dtype is None or (
                cls.dtype == np.float16
                and cls.op_type
                not in op_accuracy_white_list.NO_FP16_CHECK_GRAD_OP_LIST
                and not hasattr(cls, "exist_check_grad")
            ):
                raise AssertionError(
                    f"This test of {cls.op_type} op needs check_grad."
                )

            # check for op test with fp64 precision, but not check onednn op test for now
            if (
                cls.dtype in [np.float32, np.float64]
                and cls.op_type
                not in op_accuracy_white_list.NO_FP64_CHECK_GRAD_OP_LIST
                and not hasattr(cls, 'exist_fp64_check_grad')
                and not is_xpu_op_test()
                and not is_mkldnn_op_test()
                and not is_rocm_op_test()
                and not is_custom_device_op_test()
                and not cls.check_prim
                and not cls.check_prim_pir
            ):
                raise AssertionError(
                    f"This test of {cls.op_type} op needs check_grad with fp64 precision."
                )

            if (
                not cls.input_shape_is_large
                and cls.op_type
                not in check_shape_white_list.NEED_TO_FIX_OP_LIST
                and not is_xpu_op_test()
            ):
                raise AssertionError(
                    "Number of element(s) of input should be large than or equal to 100 for "
                    + cls.op_type
                    + " Op."
                )

    def try_call_once(self, data_type):
        if not self.call_once:
            self.call_once = True
            self.dtype = data_type

    def is_bfloat16_op(self):
        # self.dtype is the dtype of inputs, and is set in infer_dtype_from_inputs_outputs.
        # Make sure this function is called after calling infer_dtype_from_inputs_outputs.
        return (
            self.dtype == np.uint16
            or (
                hasattr(self, 'output_dtype') and self.output_dtype == np.uint16
            )
            or (
                hasattr(self, 'mkldnn_data_type')
                and self.mkldnn_data_type == "bfloat16"
            )
            or (
                hasattr(self, 'attrs')
                and 'mkldnn_data_type' in self.attrs
                and self.attrs['mkldnn_data_type'] == 'bfloat16'
            )
        )

    def is_float16_op(self):
        # self.dtype is the dtype of inputs, and is set in infer_dtype_from_inputs_outputs.
        # Make sure this function is called after calling infer_dtype_from_inputs_outputs.
        return (
            self.dtype == np.float16
            or self.dtype == "float16"
            or (
                hasattr(self, 'output_dtype')
                and self.output_dtype == np.float16
            )
            or (
                hasattr(self, 'mkldnn_data_type')
                and self.mkldnn_data_type == "float16"
            )
            or (
                hasattr(self, 'attrs')
                and 'mkldnn_data_type' in self.attrs
                and self.attrs['mkldnn_data_type'] == 'float16'
            )
        )

    def is_mkldnn_op(self):
        return (hasattr(self, "use_mkldnn") and self.use_mkldnn) or (
            hasattr(self, "attrs")
            and "use_mkldnn" in self.attrs
            and self.attrs["use_mkldnn"]
        )

    def is_xpu_op(self):
        return (hasattr(self, "use_xpu") and self.use_xpu) or (
            hasattr(self, "attrs")
            and "use_xpu" in self.attrs
            and self.attrs["use_xpu"]
        )

    def is_fp16_compared_with_fp32(self):
        return self.is_float16_op() and (
            self.op_type
            not in op_accuracy_white_list.NO_FP16_COMPARED_WITH_FP32_OP_LIST
        )

    def is_bf16_compared_with_fp32(self):
        return self.is_bfloat16_op() and (
            self.op_type
            not in op_accuracy_white_list.NO_BF16_COMPARED_WITH_FP32_OP_LIST
        )

    def is_compared_with_fp32(self):
        return (
            self.is_fp16_compared_with_fp32()
            or self.is_bf16_compared_with_fp32()
        )

    def enable_cal_ref_output(self):
        self.is_calc_ref = True

    def disable_cal_ref_output(self):
        self.is_calc_ref = False

    def _enable_check_cinn_test(self, place, inputs, outputs):
        # if the test not run in cuda or the paddle not compile with CINN, skip cinn test
        if (
            not core.is_compiled_with_cinn()
            or not core.is_compiled_with_cuda()
            or not isinstance(place, base.CUDAPlace)
        ):
            return False
        # CINN not support bfloat16 now, skip cinn test
        if self.is_bfloat16_op():
            return False
        # CINN not support 0D-tensor now, skip cinn test
        for var in inputs.values():
            if len(var.shape()) == 0:
                return False
        for var in outputs.values():
            if len(var.shape) == 0:
                return False
        # CINN not support dynamic shape now, skip cinn test
        # TODO(thisjiang): cannot check dynamic shape op automatic, should do manually now
        return True

    # set the self.output_dtype .
    def infer_dtype_from_inputs_outputs(self, inputs, outputs):
        def is_np_data(input):
            return isinstance(input, (np.ndarray, np.generic))

        def infer_dtype(numpy_dict, dtype_set):
            assert isinstance(
                numpy_dict, dict
            ), "self.inputs, self.outputs must be numpy_dict"
            # the inputs are as follows:
            # case 1: inputs = {'X': x}
            # case 2: inputs = {'X': (x, x_lod)}
            # case 3: inputs = {"X": [("x0", x0), ("x1", x1), ("x2", x2)]}
            # case 4: inputs = {'X': [("x1", (x1, [x1_lod1])), ("x2", (x2, [x2_.lod2]))]}
            # TODO(juncaipeng) infer dtype from inputs maybe obtain wrong type.
            for _, var_value in numpy_dict.items():
                if is_np_data(var_value):  # case 1
                    dtype_set.add(var_value.dtype)
                elif isinstance(var_value, (list, tuple)):  # case 2, 3, 4
                    for sub_val_value in var_value:
                        if is_np_data(sub_val_value):  # case 2
                            dtype_set.add(sub_val_value.dtype)
                        elif len(sub_val_value) > 1 and is_np_data(
                            sub_val_value[1]
                        ):  # case 3
                            dtype_set.add(sub_val_value[1].dtype)
                        elif (
                            len(sub_val_value) > 1
                            and isinstance(sub_val_value[1], (list, tuple))
                            and is_np_data(sub_val_value[1][0])
                        ):  # case 4
                            dtype_set.add(sub_val_value[1][0].dtype)

        # infer dtype from inputs, and dtype means the precision of the test
        # collect dtype of all inputs
        input_dtype_set = set()
        infer_dtype(inputs, input_dtype_set)
        dtype_list = [
            np.dtype(np.complex128),
            np.dtype(np.complex64),
            np.dtype(np.float64),
            np.dtype(np.float32),
            np.dtype(np.float16),
            np.dtype(np.int64),
            np.dtype(np.int32),
            np.dtype(np.uint16),
            np.dtype(np.int16),
            np.dtype(np.int8),
            np.dtype(np.uint8),
            np.dtype(np.bool_),
        ]
        # check the dtype in dtype_list in order, select the first dtype that in dtype_set
        for dtype in dtype_list:
            if dtype in input_dtype_set:
                self.dtype = dtype
                break
        # save input dtype in class attr
        self.__class__.dtype = self.dtype

        # infer dtype of outputs
        output_dtype_set = set()
        infer_dtype(outputs, output_dtype_set)
        for dtype in dtype_list:
            if dtype in output_dtype_set:
                self.output_dtype = dtype
                break

    def feed_var(self, input_vars, place):
        feed_map = {}
        for var_name in input_vars:
            if isinstance(input_vars[var_name], list):
                for name, np_value in self.inputs[var_name]:
                    tensor = core.DenseTensor()
                    if isinstance(np_value, tuple):
                        tensor.set(np_value[0], place)
                        dtype = np.array(np_value[1]).dtype

                        if self.is_calc_ref:
                            # convert the float16 to float by numpy.astype
                            if dtype == np.float16:
                                if isinstance(np_value[1], list):
                                    tensor.set_recursive_sequence_lengths(
                                        np.array(np_value[1]).astype(np.float32)
                                    )
                                else:
                                    tensor.set_recursive_sequence_lengths(
                                        np_value[1].astype(np.float32)
                                    )
                            # convert the bfloat16 to float by convert_uint16_to_float
                            # provided in this file
                            elif dtype == np.uint16:
                                if isinstance(np_value[1], list):
                                    tensor.set_recursive_sequence_lengths(
                                        convert_uint16_to_float(
                                            np.array(np_value[1])
                                        )
                                    )
                                else:
                                    tensor.set_recursive_sequence_lengths(
                                        convert_uint16_to_float(np_value[1])
                                    )
                            else:
                                tensor.set_recursive_sequence_lengths(
                                    np_value[1]
                                )
                        else:
                            tensor.set_recursive_sequence_lengths(np_value[1])
                    else:
                        if self.is_calc_ref:
                            if np_value.dtype == np.float16:
                                tensor.set(np_value.astype(np.float32), place)
                            elif np_value.dtype == np.uint16:
                                tensor.set(
                                    convert_uint16_to_float(np_value), place
                                )
                            else:
                                tensor.set(np_value, place)
                        else:
                            tensor.set(np_value, place)
                    feed_map[name] = tensor
            else:
                tensor = core.DenseTensor()
                if isinstance(self.inputs[var_name], tuple):
                    tensor.set(self.inputs[var_name][0], place)
                    if self.is_calc_ref:
                        if isinstance(self.inputs[var_name][1], list):
                            dtype = np.array(self.inputs[var_name][1]).dtype
                            if dtype == np.float16:
                                tensor.set_recursive_sequence_lengths(
                                    np.array(self.inputs[var_name][1]).astype(
                                        np.float32
                                    )
                                )
                            elif dtype == np.uint16:
                                tensor.set_recursive_sequence_lengths(
                                    convert_uint16_to_float(
                                        np.array(self.inputs[var_name][1])
                                    )
                                )
                            else:
                                tensor.set_recursive_sequence_lengths(
                                    self.inputs[var_name][1]
                                )

                        elif self.inputs[var_name][1].dtype == np.float16:
                            tensor.set_recursive_sequence_lengths(
                                self.inputs[var_name][1].astype(np.float32)
                            )
                        elif self.inputs[var_name][1].dtype == np.uint16:
                            tensor.set_recursive_sequence_lengths(
                                convert_uint16_to_float(
                                    self.inputs[var_name][1]
                                )
                            )
                        else:
                            tensor.set_recursive_sequence_lengths(
                                self.inputs[var_name][1]
                            )
                    else:
                        tensor.set_recursive_sequence_lengths(
                            self.inputs[var_name][1]
                        )
                else:
                    if self.is_calc_ref:
                        if self.inputs[var_name].dtype == np.float16:
                            tensor.set(
                                self.inputs[var_name].astype(np.float32), place
                            )
                        elif self.inputs[var_name].dtype == np.uint16:
                            tensor.set(
                                convert_uint16_to_float(self.inputs[var_name]),
                                place,
                            )
                        else:
                            tensor.set(self.inputs[var_name], place)
                    else:
                        tensor.set(self.inputs[var_name], place)
                feed_map[var_name] = tensor

        return feed_map

    def _append_ops(self, block):
        self.__class__.op_type = (
            self.op_type
        )  # for ci check, please not delete it for now
        if self.is_mkldnn_op():
            self.__class__.use_mkldnn = True

        if self.is_xpu_op():
            self.__class__.use_xpu = True

        op_proto = OpProtoHolder.instance().get_op_proto(self.op_type)
        # "infer datatype from inputs and outputs for this test case"

        if self.is_float16_op():
            self.dtype = np.float16
            self.__class__.dtype = self.dtype
            self.output_dtype = np.float16
        elif self.is_bfloat16_op():
            self.dtype = np.uint16
            self.__class__.dtype = self.dtype
            self.output_dtype = np.uint16
        else:
            self.infer_dtype_from_inputs_outputs(self.inputs, self.outputs)

        inputs = append_input_output(
            block, op_proto, self.inputs, True, self.dtype, self.is_calc_ref
        )
        outputs = append_input_output(
            block, op_proto, self.outputs, False, self.dtype, self.is_calc_ref
        )

        if hasattr(self, "cache_name_list"):
            for name in self.cache_name_list:
                inputs[name] = block.create_var(
                    name=name,
                    persistable=True,
                    type=core.VarDesc.VarType.RAW,
                    stop_gradient=True,
                )
        op = block.append_op(
            type=self.op_type,
            inputs=inputs,
            outputs=outputs,
            attrs=copy(self.attrs) if hasattr(self, "attrs") else {},
        )
        # infer variable type and infer shape in compile-time
        op.desc.infer_var_type(block.desc)
        op.desc.infer_shape(block.desc)

        return op

    def _get_io_vars(self, block, numpy_inputs):
        inputs = {}
        for name, value in numpy_inputs.items():
            if isinstance(value, list):
                var_list = [
                    block.var(sub_name) for sub_name, sub_value in value
                ]
                inputs[name] = var_list
            else:
                inputs[name] = block.var(name)
        return inputs

    def _get_inputs(self, block):
        return self._get_io_vars(block, self.inputs)

    def _get_outputs(self, block):
        return self._get_io_vars(block, self.outputs)

    def calc_output(self, place):
        outs, _ = self._calc_output(place)
        return outs

    def _create_var_from_numpy(self, value):
        if isinstance(value, tuple):
            data = value[0]
            lod = value[1]
            v = paddle.to_tensor(data)
            v.value().get_tensor().set_recursive_sequence_lengths(lod)
            return v
        else:
            return paddle.to_tensor(value)

    def get_sequence_batch_size_1_input(self, lod=None, shape=None):
        """Get LegacyLoD input data whose batch size is 1.
        All sequence related OP unittests should call this function to contain the case of batch size = 1.
        Args:
            lod (list[list of int], optional): Length-based LoD, length of lod[0] should be 1. Default: [[13]].
            shape (list, optional): Shape of input, shape[0] should be equals to lod[0][0]. Default: [13, 23].
        Returns:
            tuple (ndarray, lod) : LegacyLoD input data whose batch size is 1.
        """
        if lod is None:
            lod = [[13]]
        if shape is None:
            shape = [13, 23]
        assert len(lod[0]) == 1
        assert lod[0][0] == shape[0]
        x = np.random.uniform(0.1, 1, shape).astype('float32')
        return (x, lod)

    def lod_has_single_zero(self, lod):
        for i in range(len(lod) - 2):
            if lod[i] != 0 and lod[i + 1] == 0 and lod[i + 2] != 0:
                return True
        return False

    def lod_has_continuous_zero(self, lod):
        for i in range(len(lod) - 3):
            if (
                lod[i] != 0
                and lod[i + 1] == 0
                and lod[i + 2] == 0
                and lod[i + 3] != 0
            ):
                return True
        return False

    def get_sequence_instance_size_0_input(self, lod=None, shape=None):
        """Get LegacyLoD input data whose instance size is 0.
        All sequence related OP unittests should call this function to contain the case of instance size is 0.
        Args:
            lod (list[list of int], optional): Length-based LoD, lod[0]'s size must at least eight, lod[0] must at least two zeros at the beginning and at least two zeros at the end, the middle position of lod[0] contains a single zero and multiple zero. Default: [[0, 0, 4, 0, 3, 0, 0, 5, 0, 0]].
            shape (list, optional): Shape of input, shape[0] should be equals to lod[0][0]. Default: [13, 23].
        Returns:
            tuple (ndarray, lod): LegacyLoD input data whose instance size is 0.
        """
        if lod is None:
            lod = [[0, 0, 4, 0, 3, 0, 0, 5, 0, 0]]
        if shape is None:
            shape = [12, 10]
        assert len(lod[0]) >= 8
        assert (
            lod[0][0] == 0
            and lod[0][1] == 0
            and lod[0][-1] == 0
            and lod[0][-2] == 0
        )
        assert self.lod_has_single_zero(lod[0]) is True
        assert self.lod_has_continuous_zero(lod[0]) is True
        assert sum(lod[0]) == shape[0]

        x = np.random.uniform(0.1, 1, shape).astype('float32')
        return (x, lod)

    def append_input_output_for_dygraph(
        self, op_proto, np_list, is_input, if_return_inputs_grad_dict, block
    ):
        def create_var(
            np_value,
            name,
            is_input,
            if_return_inputs_grad_dict,
            is_calc_ref=False,
        ):
            np_value_temp = np_value
            has_lod = False
            lod_temp = None
            if isinstance(np_value, tuple):
                np_value_temp = np_value[0]
                has_lod = True
                lod_temp = np_value[1]

            if is_input:
                if self.is_calc_ref and np_value_temp.dtype == np.float16:
                    v = self._create_var_from_numpy(
                        np_value_temp.astype(np.float32)
                    )
                else:
                    v = self._create_var_from_numpy(np_value_temp)

                if if_return_inputs_grad_dict:
                    v.stop_gradient = False
                    v.retain_grads()

                if has_lod:
                    v.value().get_tensor().set_recursive_sequence_lengths(
                        lod_temp
                    )
            else:
                if self.is_calc_ref and np_value_temp.dtype == np.float16:
                    v = block.create_var(
                        name=name,
                        dtype=np.float32,
                        type=core.VarDesc.VarType.DENSE_TENSOR,
                        persistable=False,
                        stop_gradient=False,
                    )
                else:
                    v = block.create_var(
                        name=name,
                        dtype=np_value_temp.dtype,
                        type=core.VarDesc.VarType.DENSE_TENSOR,
                        persistable=False,
                        stop_gradient=False,
                    )
            return v

        # prepare variable for input or output
        var_dict = defaultdict(list)
        if if_return_inputs_grad_dict:
            inputs_grad_dict = defaultdict()
        proto_list = op_proto.inputs if is_input else op_proto.outputs
        for var_proto in proto_list:
            name = var_proto.name
            if (name not in np_list) and var_proto.dispensable:
                continue
            if name not in np_list:
                assert var_proto.intermediate, f"{name} not found"
                v = block.create_var(
                    dtype='float32', type=core.VarDesc.VarType.DENSE_TENSOR
                )
                var_dict[name].append(v)
                if if_return_inputs_grad_dict:
                    inputs_grad_dict[name] = v
                continue
            if var_proto.duplicable:
                assert isinstance(
                    np_list[name], list
                ), f"Duplicable {name} should be set as list"
                var_list = []
                slot_name = name
                for name, np_value in np_list[slot_name]:
                    v = create_var(
                        np_value,
                        name,
                        is_input,
                        if_return_inputs_grad_dict,
                        self.is_calc_ref,
                    )
                    var_list.append(v)
                    if if_return_inputs_grad_dict:
                        inputs_grad_dict[name] = v
                var_dict[slot_name] = var_list
            else:
                nplist_value_temp = None
                name_temp = None
                if isinstance(np_list[name], list):
                    nplist_value_temp = np_list[name][0]
                    name_temp = name
                else:
                    nplist_value_temp = np_list[name]
                    name_temp = unique_name.generate(f"{name}_out")
                v = create_var(
                    nplist_value_temp,
                    name_temp,
                    is_input,
                    if_return_inputs_grad_dict,
                    self.is_calc_ref,
                )
                var_dict[name].append(v)
                if if_return_inputs_grad_dict:
                    inputs_grad_dict[name] = v

        if if_return_inputs_grad_dict:
            return var_dict, inputs_grad_dict
        else:
            return var_dict

    def _check_api_outs_by_dygraph_outs(self, api_outs, dygraph_outs, place):
        """for quick verify, here we take a simplest strategy:
        1. we only check variable in api_outs.
        2. we simply check the numpy (tensor) .
        3. we set atol and rtol as 1e-5, because they are unrelated to dtype.
        """
        for name in api_outs:
            np_api = np.array(api_outs[name])
            np_dyg = np.array(dygraph_outs[name])
            assert (
                np_api.shape == np_dyg.shape
            ), f"Operator ({self.op_type}) : Output ({name}) shape mismatch, expect shape is {np_dyg.shape}, but actual shape is {np_api.shape}"
            np.testing.assert_allclose(
                np_api,
                np_dyg,
                rtol=1e-05,
                equal_nan=False,
                err_msg='Operator ('
                + self.op_type
                + ') Output ('
                + name
                + ') has diff at '
                + str(place)
                + '\nExpect '
                + str(np_dyg)
                + '\n'
                + 'But Got'
                + str(np_api)
                + ' in class '
                + self.__class__.__name__,
            )

    def _calc_python_api_output(self, place, egr_inps=None, egr_oups=None):
        """set egr_inps and egr_oups = None if you want to create it by yourself."""

        def construct_output_dict_by_kernel_sig(ret_tuple, output_sig):
            if hasattr(self, "python_out_sig"):
                output_sig = self.python_out_sig
            if not isinstance(ret_tuple, (tuple, list)):
                ret_tuple = [ret_tuple]
            if len(output_sig) == len(ret_tuple):
                # [assumption]: we assume {"Out": [Tensor]}
                return {a: [b] for a, b in zip(output_sig, ret_tuple)}
            else:
                # [assumption]: return multi-Tensor in a single output. such as paddle.split()
                assert (
                    len(output_sig) == 1
                ), "Don't support multi-output with multi-tensor output. (May be you can use set `python_out_sig`, see `test_squeeze2_op` as a example.)"
                return {output_sig[0]: ret_tuple}

        def cal_python_api(python_api, args, kernel_sig):
            inputs_sig, attrs_sig, outputs_sig = kernel_sig
            args = OpTestUtils.assumption_assert_and_transform(
                args, len(inputs_sig)
            )
            ret_tuple = python_api(*args)
            result = construct_output_dict_by_kernel_sig(ret_tuple, outputs_sig)
            if hasattr(self, "python_out_sig_sub_name"):
                for key in self.python_out_sig_sub_name.keys():
                    for i in range(len(self.python_out_sig_sub_name[key])):
                        result[key][0][i].name = self.python_out_sig_sub_name[
                            key
                        ][i]
            return result

        with base.dygraph.base.guard(place=place):
            block = base.framework.default_main_program().global_block()
            op_proto = OpProtoHolder.instance().get_op_proto(self.op_type)
            # prepare input variable
            dygraph_tensor_inputs = (
                egr_inps
                if egr_inps
                else self.append_input_output_for_dygraph(
                    op_proto, self.inputs, True, False, block
                )
            )
            # prepare output variable
            dygraph_tensor_outputs = (
                egr_oups
                if egr_oups
                else self.append_input_output_for_dygraph(
                    op_proto, self.outputs, False, False, block
                )
            )

            # prepare attributes
            attrs_outputs = {}
            if hasattr(self, "attrs"):
                for attrs_name in self.attrs:
                    if self.attrs[attrs_name] is not None:
                        attrs_outputs[attrs_name] = self.attrs[attrs_name]

            kernel_sig = OpTestUtils._get_kernel_signature(
                self.op_type,
                dygraph_tensor_inputs,
                dygraph_tensor_outputs,
                canonicalize_attrs(attrs_outputs, op_proto),
            )
            if not kernel_sig or (
                len(kernel_sig[0]) == 0
                and len(kernel_sig[1]) == 0
                and len(kernel_sig[2]) == 0
            ):
                return None
            if not hasattr(self, "python_api"):
                print(kernel_sig)
            assert hasattr(
                self, "python_api"
            ), f"Detect there is KernelSignature for `{self.op_type}` op, please set the `self.python_api` if you set check_dygraph = True"
            args = OpTestUtils.prepare_python_api_arguments(
                self.python_api,
                dygraph_tensor_inputs,
                attrs_outputs,
                kernel_sig,
                target_dtype=paddle.core.VarDesc.VarType,
            )
            """ we directly return the cal_python_api value because the value is already tensor.
            """
            return cal_python_api(self.python_api, args, kernel_sig)

    def _calc_dygraph_output(
        self,
        place,
        parallel=False,
        no_check_set=None,
        egr_inps=None,
        egr_oups=None,
    ):
        self.__class__.op_type = (
            self.op_type
        )  # for ci check, please not delete it for now
        with base.dygraph.base.guard(place=place):
            block = base.framework.default_main_program().global_block()

            op_proto = OpProtoHolder.instance().get_op_proto(self.op_type)

            # prepare input variable
            inputs = (
                egr_inps
                if egr_inps
                else self.append_input_output_for_dygraph(
                    op_proto, self.inputs, True, False, block
                )
            )
            # prepare output variable
            outputs = (
                egr_oups
                if egr_oups
                else self.append_input_output_for_dygraph(
                    op_proto, self.outputs, False, False, block
                )
            )

            # prepare attributes
            attrs_outputs = {}
            if hasattr(self, "attrs"):
                for attrs_name in self.attrs:
                    if self.attrs[attrs_name] is not None:
                        attrs_outputs[attrs_name] = self.attrs[attrs_name]

            block.append_op(
                type=self.op_type,
                inputs=inputs,
                outputs=outputs,
                attrs=attrs_outputs if hasattr(self, "attrs") else None,
            )
            return outputs

    def get_kernel_signature(self, place, egr_inps=None, egr_oups=None):
        with base.dygraph.base.guard(place=place):
            block = base.framework.default_main_program().global_block()
            op_proto = OpProtoHolder.instance().get_op_proto(self.op_type)
            # prepare input variable
            dygraph_tensor_inputs = (
                egr_inps
                if egr_inps
                else self.append_input_output_for_dygraph(
                    op_proto, self.inputs, True, False, block
                )
            )
            # prepare output variable
            dygraph_tensor_outputs = (
                egr_oups
                if egr_oups
                else self.append_input_output_for_dygraph(
                    op_proto, self.outputs, False, False, block
                )
            )

            # prepare attributes
            attrs_outputs = {}
            if hasattr(self, "attrs"):
                for attrs_name in self.attrs:
                    if self.attrs[attrs_name] is not None:
                        attrs_outputs[attrs_name] = self.attrs[attrs_name]
            kernel_sig = OpTestUtils._get_kernel_signature(
                self.op_type,
                dygraph_tensor_inputs,
                dygraph_tensor_outputs,
                canonicalize_attrs(attrs_outputs, op_proto),
            )
            if not kernel_sig or (
                len(kernel_sig[0]) == 0
                and len(kernel_sig[1]) == 0
                and len(kernel_sig[2]) == 0
            ):
                return None
            if not hasattr(self, "python_api"):
                print(kernel_sig)
            assert hasattr(
                self, "python_api"
            ), f"Detect there is KernelSignature for `{self.op_type}` op, please set the `self.python_api` if you set check_dygraph = True"
            return kernel_sig

    def get_ir_input_attr_dict_and_feed(self, stop_gradient):
        attrs_outputs = {}
        if hasattr(self, "attrs"):
            for attrs_name in self.attrs:
                if self.attrs[attrs_name] is not None:
                    attrs_outputs[attrs_name] = self.attrs[attrs_name]
        input_dict = {}
        static_inputs = defaultdict(list)
        feed = {}
        for name, item in self.inputs.items():
            if isinstance(item, (list, tuple)):
                for tup in item:
                    dtype = (
                        "bfloat16"
                        if OpTestUtils.is_bfloat16_type(tup[1].dtype)
                        else tup[1].dtype
                    )
                    x = paddle.static.data(
                        name=str(tup[0]), shape=tup[1].shape, dtype=dtype
                    )
                    x.stop_gradient = stop_gradient
                    static_inputs[name].append(x)
                    feed.update({str(tup[0]): tup[1]})
                    input_dict.update({str(tup[0]): x})
            else:
                dtype = (
                    "bfloat16"
                    if OpTestUtils.is_bfloat16_type(item.dtype)
                    else item.dtype
                )
                x = paddle.static.data(name=name, shape=item.shape, dtype=dtype)
                x.stop_gradient = stop_gradient
                static_inputs[name].append(x)
                feed.update({name: item})
                input_dict.update({name: x})
        return static_inputs, attrs_outputs, input_dict, feed

    def _need_fetch(self, sig_name):
        if sig_name in self.outputs:
            return True
        for _, value in self.outputs.items():
            if not isinstance(value, (tuple, list)):
                continue
            for var_name, _ in value:
                if sig_name == var_name:
                    return True
        return False

    def _calc_pir_output(self, place, no_check_set=None, inps=None, oups=None):
        """set egr_inps and egr_oups = None if you want to create it by yourself."""

        def construct_output_dict_by_kernel_sig(ret_tuple, output_sig):
            if hasattr(self, "python_out_sig"):
                output_sig = self.python_out_sig
            if not isinstance(ret_tuple, (tuple, list)):
                ret_tuple = [ret_tuple]
            if len(output_sig) == len(ret_tuple):
                # [assumption]: we assume {"Out": [Tensor]}
                return {a: [b] for a, b in zip(output_sig, ret_tuple)}
            else:
                # [assumption]: return multi-Tensor in a single output. such as paddle.split()
                assert (
                    len(output_sig) == 1
                ), "Don't support multi-output with multi-tensor output. (May be you can use set `python_out_sig`, see `test_squeeze2_op` as a example.)"
                return {output_sig[0]: ret_tuple}

        # get kernel signature
        kernel_sig = self.get_kernel_signature(place)
        ir_program = paddle.static.Program()
        with paddle.static.program_guard(ir_program):
            with scope_guard(Scope()):
                # prepare inps attributes feed
                (
                    static_inputs,
                    attrs,
                    input_dict,
                    feed,
                ) = self.get_ir_input_attr_dict_and_feed(stop_gradient=True)
                # prepare args
                args = OpTestUtils.prepare_python_api_arguments(
                    self.python_api,
                    static_inputs,
                    attrs,
                    kernel_sig,
                    target_dtype=paddle.pir.core.DataType,
                )
                inputs_sig, attrs_sig, outputs_sig = kernel_sig
                if hasattr(self, "python_out_sig"):
                    outputs_sig = self.python_out_sig
                args = OpTestUtils.assumption_assert_and_transform(
                    args, len(inputs_sig)
                )
                ret_tuple = self.python_api(*args)
                fetch_list = getattr(self, "fetch_list", [])
                # if the fetch_list is customized by user, we use it directly.
                # if not, fill the fetch_list by the user configured outputs in test.
                # filter ret_tuple
                ret_to_check = []
                if len(fetch_list) == 0:
                    if isinstance(ret_tuple, (tuple, list)):
                        assert len(ret_tuple) == len(outputs_sig)
                        for var, sig_name in zip(ret_tuple, outputs_sig):
                            if no_check_set is not None and var in no_check_set:
                                continue
                            if not self._need_fetch(sig_name):
                                continue
                            if isinstance(var, list):
                                ret_to_check.append(var)
                                for v in var:
                                    fetch_list.append(v)
                            else:
                                ret_to_check.append(var)
                                fetch_list.append(var)
                    elif isinstance(ret_tuple, paddle.base.libpaddle.pir.Value):
                        fetch_list.append(ret_tuple)
                        ret_to_check = ret_tuple
                    elif ret_tuple is None:
                        pass
                    else:
                        raise ValueError(
                            "output of python api should be Value or list of Value or tuple of Value"
                        )

                # executor run
                executor = Executor(place)
                outs = executor.run(
                    ir_program, feed=feed, fetch_list=[fetch_list]
                )
                outputs_sig = [
                    sig_name
                    for sig_name in outputs_sig
                    if self._need_fetch(sig_name)
                ]

                if paddle.utils.is_sequence(
                    ret_to_check
                ) and paddle.utils.is_sequence(outs):
                    outs = paddle.utils.pack_sequence_as(ret_to_check, outs)

                result = construct_output_dict_by_kernel_sig(outs, outputs_sig)
                if hasattr(self, "python_out_sig_sub_name"):
                    for key in self.python_out_sig_sub_name.keys():
                        result[key][0] = {
                            a: [b]
                            for a, b in zip(
                                self.python_out_sig_sub_name[key],
                                result[key][0],
                            )
                        }
                return result

    def _check_ir_output(self, place, program, feed_map, fetch_list, outs):
        if os.getenv("FLAGS_PIR_OPTEST") is None:
            return
        if os.getenv("FLAGS_PIR_OPTEST_WHITE_LIST") is None:
            return
        if self.check_prim or self.check_prim_pir:
            return
        if self._check_cinn:
            return
        stored_flag = get_flags(
            [
                'FLAGS_enable_pir_in_executor',
                "FLAGS_pir_apply_inplace_pass",
            ]
        )
        try:
            set_flags(
                {
                    "FLAGS_enable_pir_in_executor": True,
                    "FLAGS_pir_apply_inplace_pass": 0,
                }
            )
            new_scope = paddle.static.Scope()
            executor = Executor(place)
            new_program = None
            if isinstance(program, paddle.static.CompiledProgram):
                new_program = base.CompiledProgram(
                    program._program, build_strategy=program._build_strategy
                )
            else:
                new_program = program.clone()
            ir_outs = executor.run(
                new_program,
                feed=feed_map,
                fetch_list=fetch_list,
                return_numpy=False,
                scope=new_scope,
            )
            assert len(outs) == len(
                ir_outs
            ), "Fetch result should have same length when executed in pir"

            check_method = np.testing.assert_array_equal
            if os.getenv("FLAGS_PIR_OPTEST_RELAX_CHECK", None) == "True":

                def relaxed_check(x, y, err_msg=""):
                    np.testing.assert_allclose(
                        x, y, err_msg=err_msg, atol=1e-6, rtol=1e-6
                    )

                check_method = relaxed_check
            if os.getenv("FLAGS_PIR_NO_CHECK", None) == "True":
                check_method = lambda x, y, err_msg: None

            for i in range(len(outs)):
                check_method(
                    outs[i],
                    ir_outs[i],
                    err_msg='Operator Check ('
                    + self.op_type
                    + ') has diff at '
                    + str(place)
                    + '\nExpect '
                    + str(outs[i])
                    + '\n'
                    + 'But Got'
                    + str(ir_outs[i])
                    + ' in class '
                    + self.__class__.__name__,
                )
        finally:
            set_flags(stored_flag)

    def _calc_output(
        self,
        place,
        parallel=False,
        no_check_set=None,
        loss=None,
        enable_inplace=None,
        for_inplace_test=None,
        check_cinn=False,
    ):
        with paddle.pir_utils.OldIrGuard():
            if hasattr(self, "attrs"):
                for k, v in self.attrs.items():
                    if isinstance(v, paddle.base.core.DataType):
                        self.attrs[k] = paddle.pir.core.datatype_to_vartype[v]
            program = Program()
            block = program.global_block()
            op = self._append_ops(block)

            inputs = self._get_inputs(block)
            outputs = self._get_outputs(block)
            feed_map = self.feed_var(inputs, place)

            if for_inplace_test:
                # Some variables' tensors hold no buffer (tensor's _holder is NULL), like XShape in reshape2 op,
                # and the shapes of those variables contain 0 (eg. Xshape.shape = [0, 2, 5]).
                # Set persistable for those variables in order to get them from global_scope for inplace grad test directly other than feed them,
                # since feed op calls check_memory_size() which fails when tensor's holder_ is NULL.
                for out_name in op.output_arg_names:
                    var = block.var(out_name)
                    if 0 in var.shape:
                        var.persistable = True
            original_program = program
            if parallel:
                use_cuda = False
                if isinstance(place, base.CUDAPlace):
                    use_cuda = True
                compiled_prog = base.CompiledProgram(program)
                program = compiled_prog
            fetch_list = getattr(self, "fetch_list", [])
            # if the fetch_list is customized by user, we use it directly.
            # if not, fill the fetch_list by the user configured outputs in test.
            if len(fetch_list) == 0:
                for var_name, var in outputs.items():
                    if no_check_set is not None and var_name in no_check_set:
                        continue
                    if isinstance(var, list):
                        for v in var:
                            fetch_list.append(v.name)
                    else:
                        fetch_list.append(var.name)
            # if the fetch_list still empty, fill the fetch_list by the operator output.
            if len(fetch_list) == 0:
                for out_name, out_dup in Operator.get_op_outputs(self.op_type):
                    fetch_list.append(str(out_name))

            enable_cinn_test = check_cinn and self._enable_check_cinn_test(
                place, feed_map, outputs
            )
            if enable_cinn_test:
                if hasattr(self, 'cinn_atol'):
                    self.atol = self.cinn_atol
                if hasattr(self, 'cinn_rtol'):
                    self.rtol = self.cinn_rtol

            if (enable_inplace is not None) or enable_cinn_test:
                build_strategy = base.BuildStrategy()
                if enable_inplace is not None:
                    build_strategy.enable_inplace = enable_inplace
                if enable_cinn_test:
                    build_strategy.build_cinn_pass = check_cinn
                    self._check_cinn = enable_cinn_test

                compiled_prog = base.CompiledProgram(
                    program, build_strategy=build_strategy
                )
                program = compiled_prog

            executor = Executor(place)

            outs = executor.run(
                program,
                feed=feed_map,
                fetch_list=fetch_list,
                return_numpy=False,
            )

            self._check_ir_output(place, program, feed_map, fetch_list, outs)

            self.op = op
            self.program = original_program
        if for_inplace_test:
            return outs, fetch_list, feed_map, original_program, op.desc
        else:
            return outs, fetch_list

    def _compare_symbol(self, program, outs):
        i = 0
        # check that all ops have defined the InferSymbolicShapeInterface
        if paddle.base.libpaddle.pir.all_ops_defined_symbol_infer(program):
            # compare expect & actual
            shape_analysis = (
                paddle.base.libpaddle.pir.get_shape_constraint_ir_analysis(
                    program
                )
            )
            for block in program.blocks:
                for op in block.ops:
                    if op.name() == "pd_op.fetch":
                        for j, var in enumerate(op.results()):
                            if (
                                var.is_dense_tensor_type()
                                or var.is_selected_row_type()
                            ):
                                shape_or_data = (
                                    shape_analysis.get_shape_or_data_for_var(
                                        var
                                    )
                                )
                                expect_shape = outs[i].shape
                                if np.issubdtype(outs[i].dtype, np.integer):
                                    expect_data = outs[i].flatten().tolist()
                                else:
                                    expect_data = []
                                i += 1
                                if not shape_or_data.is_equal(
                                    expect_shape, expect_data
                                ):
                                    raise AssertionError(
                                        f"The shape or data of Operator {self.op_type}'s result_value[{j}] is different from expected."
                                    )
        else:
            # TODO(gongshaotian): raise error
            pass

    def _infer_and_compare_symbol(self, place):
        """Don't calculate the program, only infer the shape of var"""

        kernel_sig = self.get_kernel_signature(place)
        program = paddle.static.Program()
        with paddle.static.program_guard(program):
            scope = Scope()
            with scope_guard(scope):
                # prepare inps attributes feed
                (
                    static_inputs,
                    attrs,
                    input_dict,
                    feed,
                ) = self.get_ir_input_attr_dict_and_feed(stop_gradient=True)
                # prepare args
                args = OpTestUtils.prepare_python_api_arguments(
                    self.python_api,
                    static_inputs,
                    attrs,
                    kernel_sig,
                    target_dtype=paddle.pir.core.DataType,
                )
                inputs_sig, attrs_sig, outputs_sig = kernel_sig
                if hasattr(self, "python_out_sig"):
                    outputs_sig = self.python_out_sig
                args = OpTestUtils.assumption_assert_and_transform(
                    args, len(inputs_sig)
                )
                # add op to program
                ret_tuple = self.python_api(*args)
                fetch_list = getattr(self, "fetch_list", [])
                # if the fetch_list is customized by user, we use it directly.
                # if not, fill the fetch_list by the user configured outputs in test.
                # filter ret_tuple
                ret_to_check = []
                if len(fetch_list) == 0:
                    if isinstance(ret_tuple, (tuple, list)):
                        assert len(ret_tuple) == len(outputs_sig)
                        for var, sig_name in zip(ret_tuple, outputs_sig):
                            if not self._need_fetch(sig_name):
                                continue
                            if isinstance(var, list):
                                ret_to_check.append(var)
                                for v in var:
                                    fetch_list.append(v)
                            else:
                                ret_to_check.append(var)
                                fetch_list.append(var)
                    elif isinstance(ret_tuple, paddle.base.libpaddle.pir.Value):
                        fetch_list.append(ret_tuple)
                        ret_to_check = ret_tuple
                    elif ret_tuple is None:
                        pass
                    else:
                        raise ValueError(
                            "output of python api should be Value or list of Value or tuple of Value"
                        )

                # executor run
                executor = Executor(place)
                outs = executor.run(program, feed=feed, fetch_list=[fetch_list])
                # get fetch program
                fetch_list = executor._check_fetch_list([fetch_list])
                fetch_program, _, _ = (
                    executor._executor_cache.get_pir_program_and_executor(
                        program=program,
                        feed=feed,
                        fetch_list=fetch_list,
                        feed_var_name='feed',
                        fetch_var_name='fetch',
                        place=place,
                        scope=scope,
                        plan=None,
                    )
                )

                self._compare_symbol(fetch_program, outs)

    def _compare_expect_and_actual_outputs(
        self, place, fetch_list, expect_outs, actual_outs, inplace_atol=None
    ):
        """Compare expect outs and actual outs of an tested op.

        Args:
            place (CPUPlace | CUDAPlace): The place where the op runs.
            fetch_list (list): The outputs of tested op.
            expect_outs (list): The expect outs of tested op.
            actual_outs (list): The actual outs of tested op.
            inplace_atol (float): The tolerable error, only set when tested op doesn't ensure computational consistency, like group_norm op.

        Returns:
            None.
        """
        # compare expect_outs and actual_outs
        for i, name in enumerate(fetch_list):
            # Note(zhiqiu): inplace_atol should be only set when op doesn't ensure
            # computational consistency.
            # When inplace_atol is not None, the inplace check uses numpy.allclose
            # to check inplace result instead of numpy.array_equal.
            expect_out = np.array(expect_outs[i])
            actual_out = np.array(actual_outs[i])
            assert (
                actual_out.shape == expect_out.shape
            ), f"Operator ({self.op_type}) : Output ({name}) shape mismatch, expect shape is {expect_out.shape}, but actual shape is {actual_out.shape}"
            if inplace_atol is not None:
                np.testing.assert_allclose(
                    expect_out,
                    actual_out,
                    rtol=1e-03 if self.dtype == np.uint16 else 1e-5,
                    atol=inplace_atol,
                    err_msg='Operator ('
                    + self.op_type
                    + ') Output ('
                    + name
                    + ') has diff at '
                    + str(place)
                    + ' when using and not using inplace'
                    + '\nExpect '
                    + str(expect_out)
                    + '\n'
                    + 'But Got'
                    + str(actual_out)
                    + ' in class '
                    + self.__class__.__name__,
                )
            else:
                np.testing.assert_array_equal(
                    expect_out,
                    actual_out,
                    err_msg='Output ('
                    + name
                    + ') has diff at '
                    + str(place)
                    + ' when using and not using inplace'
                    + '\nExpect '
                    + str(expect_out)
                    + '\n'
                    + 'But Got'
                    + str(actual_out)
                    + ' in class '
                    + self.__class__.__name__
                    + '\n',
                )

    def _construct_grad_program_from_forward(
        self, fwd_program, grad_op_desc, op_grad_to_var
    ):
        """Generate grad_program which contains the grad_op.

        Args:
            fwd_program (tuple): The program that contains grad_op_desc's corresponding forward op.
            grad_op_desc (OpDesc): The OpDesc of grad op.
            op_grad_to_var (dict): The relation of variables in grad op and its forward op.

        Returns:
            grad_program (program): The program which contains the grad_op.
        """
        with paddle.pir_utils.OldIrGuard():
            grad_program = Program()
            grad_block = grad_program.global_block()
            new_op_desc = grad_block.desc.append_op()
            new_op_desc.copy_from(grad_op_desc)
            grad_program._sync_with_cpp()

            # Create grad vars based on fwd vars (shape and dtype)
            for arg in (
                grad_op_desc.input_arg_names() + grad_op_desc.output_arg_names()
            ):
                fwd_var_name = op_grad_to_var.get(arg, None)
                if fwd_var_name is None:
                    fwd_var_name = arg
                fwd_var = fwd_program.global_block().vars.get(fwd_var_name)
                assert fwd_var is not None, f"{fwd_var_name} cannot be found"
                grad_var = grad_block.create_var(
                    name=arg,
                    dtype=fwd_var.dtype,
                    shape=fwd_var.shape,
                    type=fwd_var.type,
                    persistable=False,
                )

                # Some variables' tensors hold no buffer (tensor's _holder is NULL), like XShape in reshape2 op,
                # and the shapes of those variables contain 0 (eg. Xshape.shape = [0, 2, 5]).
                # Set persistable for those variables in order to get them from global_scope for inplace grad test directly other than feed them,
                # since feed op calls check_memory_size() which fails when tensor's holder_ is NULL.
                if 0 in grad_var.shape:
                    grad_var.persistable = True
            grad_program._sync_with_cpp()
            return grad_program

    def _construct_grad_feed_map_from_forward(
        self, place, fwd_res, grad_op_desc, op_grad_to_var
    ):
        """Generate grad_feed_map for grad_program.

        since we don`t really check gradient accuracy, but check the consistency when using and not using inplace,
        we use fwd outs (also inputs sometimes) to construct grad inputs.

        Args:
            place (CPUPlace | CUDAPlace): The place where the op runs.
            fwd_res (tuple): The outputs of its forward op, in the same form as returns of _calc_outputs() when for_inplace_test is True.
                i.e., tuple(fwd_outs, fwd_fetch_list, fwd_feed_map, fwd_program, fwd_op_desc)
            grad_op_desc (OpDesc): The OpDesc of grad op.
            op_grad_to_var (dict): The relation of variables in grad op and its fwd_op.

        Returns:
            grad_feed_map (dict): The feed_map of grad_op.
        """
        (
            fwd_outs,
            fwd_fetch_list,
            fwd_feed_map,
            fwd_program,
            fwd_op_desc,
        ) = fwd_res
        p = core.Place()
        p.set_place(place)
        grad_feed_map = {}
        for arg in grad_op_desc.input_arg_names():
            if arg in fwd_feed_map.keys():
                grad_feed_map[arg] = fwd_feed_map[arg]._copy(p)
            else:
                fwd_var_name = op_grad_to_var.get(arg, None)
                if fwd_var_name is None:
                    fwd_var_name = arg

                for i, out_name in enumerate(fwd_fetch_list):
                    if out_name == fwd_var_name:
                        # don't feed variables whose tensors hold no buffer (shape contains 0 like shape = [0,2,5] and holder_ is NULL), like XShape in reshape2 op.
                        # get them from global_scope directly since we have set them persistable in fwd execution
                        if 0 in fwd_program.global_block().var(out_name).shape:
                            continue
                        else:
                            grad_feed_map[arg] = fwd_outs[i]._copy(p)

        return grad_feed_map

    def _get_need_run_ops(self, op_desc, fwd_op_desc=None):
        """Postorder traversal of the 'grad' tree to get all ops that need to run during inplace test.
        An op needs to run during inplace check if,
        (1) it has infer_inplace,
        (2) it has infer_inplace in its grad descendants. (since we need its outputs as to construct its grad's inputs)

        Args:
            op_desc (OpDesc): The op_desc of current op.
            fwd_op_desc (OpDesc): The op_desc of current op's forward op, None if current op has no forward op.
                E.g. relu's fwd_op is None, relu_grad's fwd_op is relu, relu_grad_grad's fwd_op is relu_grad, etc.

        Returns:
            need_run_ops (list[(op_desc, fwd_op_desc)]): The ops that need to run during inplace test.
        """
        need_run_ops = []
        visited_ops = []

        def _dfs_grad_op(op_desc, fwd_op_desc=None):
            visited_ops.append(op_desc.type())
            has_infer_inplace = base.core.has_infer_inplace(op_desc.type())
            has_grad_op_maker = base.core.has_grad_op_maker(op_desc.type())
            has_infer_inplace_in_grad_descendants = False
            if not has_grad_op_maker:
                has_infer_inplace_in_descendants = False
            else:
                # get grad_op_desc
                grad_op_desc_list, op_grad_to_var = core.get_grad_op_desc(
                    op_desc, set(), []
                )
                if not grad_op_desc_list:
                    has_infer_inplace_in_grad_descendants = False
                else:
                    for i, grad_op_desc in enumerate(grad_op_desc_list):
                        if (
                            grad_op_desc.type() not in visited_ops
                            and _dfs_grad_op(grad_op_desc, fwd_op_desc=op_desc)
                        ):
                            has_infer_inplace_in_grad_descendants = True
            if has_infer_inplace or has_infer_inplace_in_grad_descendants:
                need_run_ops.append((op_desc, fwd_op_desc))
                return True
            else:
                return False

        _dfs_grad_op(op_desc, fwd_op_desc=fwd_op_desc)
        return need_run_ops

    def _check_forward_inplace(
        self, place, no_check_set=None, inplace_atol=None
    ):
        """Check the inplace correctness of given op (self.op_type).
        Run the op twice with same inputs, one enable inplace and another disable, compare their outputs.

        Args:
            place (CPUPlace | CUDAPlace): The place where the op runs.
            no_check_set (list): The names of outputs that needn't check, like XShape of reshape op.
            inplace_atol (float): The tolerable error, only set when op doesn't ensure computational consistency, like group_norm op.

        Returns:
            expect_res (tuple(outs, fetch_list, feed_map, program, op_desc)): The results of given op.
                We return this to construct grad_program and grad_feed_map for grad inplace check.
        """
        # _calc_output() returns in the form tuple(outs, fetch_list, feed_map, program, op_desc) when for_inplace_test=True.
        expect_res = self._calc_output(
            place,
            no_check_set=no_check_set,
            enable_inplace=False,
            for_inplace_test=True,
        )
        actual_res = self._calc_output(
            place,
            no_check_set=no_check_set,
            enable_inplace=True,
            for_inplace_test=True,
        )
        # compare expect_outs and actual_outs
        self._compare_expect_and_actual_outputs(
            place,
            expect_res[1],
            expect_res[0],
            actual_res[0],
            inplace_atol=inplace_atol,
        )
        return expect_res

    def _calc_grad_output(
        self, place, fwd_res, grad_op_desc, enable_inplace=None
    ):
        """Calculate grad_output for given grad_op_desc.

        since we don`t really check gradient accuracy, but check the consistency when using and not using inplace,
        we use fwd outs (also inputs sometimes) to construct grad inputs.

        Args:
            place (CPUPlace | CUDAPlace): The place where the op runs.
            fwd_res (tuple): The outputs of its forward op, in the same form as returns of _calc_outputs() when for_inplace_test is True.
                i.e., tuple(fwd_outs, fwd_fetch_list, fwd_feed_map, fwd_program, fwd_op_desc).
            grad_op_desc (OpDesc): The OpDesc of grad op.
            enable_inplace (bool): Enable inplace or not.

        Returns:
            res (tuple(outs, fetch_list, feed_map, program, op_desc)): The results of given grad_op_desc.
        """
        with static_guard():
            (
                fwd_outs,
                fwd_fetch_list,
                fwd_feed_map,
                fwd_program,
                fwd_op_desc,
            ) = fwd_res
            grad_op_desc_list, op_grad_to_var = core.get_grad_op_desc(
                fwd_op_desc, set(), []
            )
            grad_program = self._construct_grad_program_from_forward(
                fwd_program, grad_op_desc, op_grad_to_var
            )
            grad_feed_map = self._construct_grad_feed_map_from_forward(
                place, fwd_res, grad_op_desc, op_grad_to_var
            )
            grad_fetch_list = grad_op_desc.output_arg_names()
            exe = Executor(place)
            program = grad_program
            if enable_inplace is not None:
                build_strategy = base.BuildStrategy()
                build_strategy.enable_inplace = enable_inplace
                compiled_program = base.CompiledProgram(
                    grad_program, build_strategy=build_strategy
                )
                program = compiled_program

            outs = exe.run(
                program,
                feed=grad_feed_map,
                fetch_list=grad_fetch_list,
                return_numpy=False,
            )
        return outs, grad_fetch_list, grad_feed_map, grad_program, grad_op_desc

    def _check_grad_inplace(
        self, place, fwd_res, grad_op_desc, inplace_atol=None
    ):
        """Check the inplace correctness of given grad_op_desc.

        Run the grad op twice with same inputs, one enable inplace and another disable, compare their outputs.
        It works like _check_forward_inplace, but the way to construct program and feed_map differs.
        So we define a new function for grad, grad_grad, etc.

        Args:
            place (CPUPlace | CUDAPlace): The place where the op runs.
            fwd_res (tuple): The outputs of its forward op, in the same form as returns of _calc_outputs() when for_inplace_test is True.
                i.e., tuple(fwd_outs, fwd_fetch_list, fwd_feed_map, fwd_program, fwd_op_desc).
            grad_op_desc (OpDesc): The OpDesc of grad op.
            inplace_atol (float): The tolerable error, only set when op doesn't ensure computational consistency, like group_norm op.

        Returns:
            expect_res (tuple(outs, fetch_list, feed_map, program, op_desc)): The results of given op.
                We return this to construct grad_program and grad_feed_map for grad inplace check.
        """
        expect_res = self._calc_grad_output(
            place, fwd_res, grad_op_desc, enable_inplace=False
        )
        actual_res = self._calc_grad_output(
            place, fwd_res, grad_op_desc, enable_inplace=True
        )

        self._compare_expect_and_actual_outputs(
            place,
            expect_res[1],
            expect_res[0],
            actual_res[0],
            inplace_atol=inplace_atol,
        )
        return expect_res

    def check_inplace_output_with_place(
        self, place, no_check_set=None, inplace_atol=None
    ):
        """Check the inplace correctness of given op, its grad op, its grad_grad op, etc.

        (1) Get all ops need to run. (see conditions in _get_need_run_ops())
        (2) Run op in need_run_ops, and do inplace check if it has infer_inplace.

        Args:
            place (CPUPlace | CUDAPlace): The place where the op runs.
            no_check_set (list): The names of outputs that needn't check, like XShape of reshape op.
            inplace_atol (float): The tolerable error, only set when op doesn't ensure computational consistency, like group_norm op.

        Returns:
            None
        """
        if getattr(self, "no_need_check_inplace", False):
            return

        if (
            os.getenv("FLAGS_enable_pir_in_executor")
            or os.getenv("FLAGS_enable_pir_api")
            or get_flags("FLAGS_enable_pir_in_executor")[
                "FLAGS_enable_pir_in_executor"
            ]
            or get_flags("FLAGS_enable_pir_api")["FLAGS_enable_pir_api"]
        ):
            return

        has_infer_inplace = base.core.has_infer_inplace(self.op_type)
        has_grad_op_maker = base.core.has_grad_op_maker(self.op_type)
        fwd_res = self._calc_output(
            place, no_check_set=no_check_set, for_inplace_test=True
        )
        op_desc = fwd_res[4]
        need_run_ops = self._get_need_run_ops(op_desc)

        res = {}
        if hasattr(self, 'attrs') and bool(self.attrs.get('use_xpu', False)):
            return
        for op_desc, father_op_desc in reversed(need_run_ops):
            # The first one is the forward op
            has_infer_inplace = base.core.has_infer_inplace(op_desc.type())
            if op_desc.type() == self.op_type:
                if has_infer_inplace:
                    res[op_desc] = self._check_forward_inplace(
                        place,
                        no_check_set=no_check_set,
                        inplace_atol=inplace_atol,
                    )
                else:
                    res[op_desc] = self._calc_output(
                        place, no_check_set=no_check_set, for_inplace_test=True
                    )
            else:
                # TODO(zhiqiu): enhance inplace_grad test for ops (sum and activation) using mkldnn
                # skip op that use_mkldnn currently
                flags_use_mkldnn = base.core.globals()["FLAGS_use_mkldnn"]
                attrs_use_mkldnn = hasattr(self, 'attrs') and bool(
                    self.attrs.get('use_mkldnn', False)
                )
                if flags_use_mkldnn or attrs_use_mkldnn:
                    warnings.warn(
                        "check inplace_grad for ops using mkldnn is not supported"
                    )
                    continue
                if has_infer_inplace:
                    fwd_res = res[father_op_desc]
                    res[op_desc] = self._check_grad_inplace(
                        place, fwd_res, op_desc, inplace_atol=inplace_atol
                    )
                else:
                    res[op_desc] = self._calc_grad_output(
                        place, fwd_res, op_desc
                    )

    def check_output_with_place(
        self,
        place,
        atol=0,
        rtol=0,
        no_check_set=None,
        equal_nan=False,
        check_dygraph=True,
        check_prim=False,
        check_prim_pir=False,
        only_check_prim=False,
        inplace_atol=None,
        check_cinn=False,
        check_pir=False,
        check_auto_parallel=False,
        check_pir_onednn=False,
        check_symbol_infer=True,
    ):
        core._set_prim_all_enabled(False)
        core.set_prim_eager_enabled(False)
        if not self.is_mkldnn_op():
            set_flags({"FLAGS_use_mkldnn": False})

        if hasattr(self, "use_custom_device") and self.use_custom_device:
            check_dygraph = False

        def find_imperative_actual(target_name, dygraph_outs, place):
            for name in dygraph_outs:
                if name == target_name:
                    return dygraph_outs[name][0]
                var_list = dygraph_outs[name]
                for i, var in enumerate(var_list):
                    if isinstance(var, list):
                        for tensor in var:
                            if tensor.name == target_name:
                                return tensor
                    elif (
                        isinstance(var, paddle.Tensor)
                        and var.name == target_name
                    ):
                        return dygraph_outs[name][i]
            self.assertTrue(
                False,
                f"Found failed {dygraph_outs.keys()} {target_name}",
            )

        def find_imperative_expect(target_name, dygraph_outs, place):
            for name in dygraph_outs:
                if name == target_name:
                    return dygraph_outs[name][0]
                var_list = dygraph_outs[name]
                for i, var in enumerate(var_list):
                    if var.name == target_name:
                        return dygraph_outs[name][i]
            self.assertTrue(
                False,
                f"Found failed {dygraph_outs.keys()} {target_name}",
            )

        def find_actual(target_name, fetch_list):
            found = [
                i
                for i, var_name in enumerate(fetch_list)
                if var_name == target_name
            ]
            self.assertTrue(
                len(found) == 1, f"Found {len(found)} {target_name}"
            )
            return found[0]

        def find_expect(target_name, fetch_list):
            found = [
                i
                for i, var_name in enumerate(fetch_list)
                if var_name == target_name
            ]
            self.assertTrue(
                len(found) == 1, f"Found {len(found)} {target_name}"
            )
            return found[0]

        class Checker:
            """base class for check with self.outputs.
            currently don't support check between checkers.
            """

            def __init__(self, op_test, expect_dict):
                """expect_dict is the self.outputs
                support : {str: [numpy]} and {str: [(str, numpy), (str, numpy)]}
                """
                self.expects = expect_dict
                self.checker_name = "checker"
                self.op_test = op_test  # stop the op_test object.
                self.op_type = op_test.op_type

            def init(self):
                pass

            def convert_uint16_to_float(self, actual_np, expect_np):
                raise NotImplementedError("base class, not implement!")

            def calculate_output(self):
                """
                judge whether convert current output and expect to uint16.
                return True | False
                """

            def _is_skip_name(self, name):
                if name not in self.expects:
                    return True
                if no_check_set is not None and name in no_check_set:
                    return True
                return False

            def find_actual_value(self, name):
                """return: (actual_tensor(var_base), actual_numpy)"""
                raise NotImplementedError("base class, not implement!")

            def find_expect_value(self, name):
                """return: (expect_tensor(var_base), actual_numpy)"""
                raise NotImplementedError("base class, not implement!")

            def _compare_numpy(self, name, actual_np, expect_np):
                expect_np = np.array(expect_np)
                assert (
                    actual_np.shape == expect_np.shape
                ), f"Operator ({self.op_type}) : Output ({name}) shape mismatch, expect shape is {expect_np.shape}, but actual shape is {actual_np.shape}"
                np.testing.assert_allclose(
                    actual_np,
                    expect_np,
                    atol=self.atol if hasattr(self, 'atol') else atol,
                    rtol=self.rtol if hasattr(self, 'rtol') else rtol,
                    equal_nan=equal_nan,
                    err_msg=(
                        "Operator ("
                        + self.op_type
                        + ") Output ("
                        + name
                        + ") has diff at "
                        + str(place)
                        + " in "
                        + self.checker_name
                    ),
                )

            def compare_single_output_with_expect(self, name, expect):
                actual, actual_np = self.find_actual_value(name)
                # expect_np = expect[0] if isinstance(expect, tuple) else expect
                if self.op_test.is_compared_with_fp32():
                    expect, expect_np = self.find_expect_value(name)
                else:
                    expect_np = (
                        expect[0]
                        if isinstance(expect, (tuple, list))
                        else expect
                    )
                actual_np, expect_np = self.convert_uint16_to_float_ifneed(
                    actual_np, expect_np
                )
                # modify there for fp32 check
                self._compare_numpy(name, actual_np, expect_np)

            def compare_outputs_with_expects(self):
                for out_name, out_dup in Operator.get_op_outputs(self.op_type):
                    if self._is_skip_name(out_name):
                        continue
                    if out_dup:
                        # if self.output = {'name': [(subname, Tensor), (subname, Tensor)]}
                        sub_out = self.expects[out_name]
                        if not isinstance(sub_out, list):
                            raise AssertionError(
                                "sub_out type %s is not list", type(sub_out)
                            )
                        for item in sub_out:
                            sub_out_name, expect = item[0], item[1]
                            self.compare_single_output_with_expect(
                                sub_out_name, expect
                            )
                    else:
                        expect = self.expects[out_name]
                        self.compare_single_output_with_expect(out_name, expect)

            def check(self):
                """
                return None means ok, raise Error means failed.

                the main enter point of Checker class
                """
                self.init()
                self.calculate_output()
                self.compare_outputs_with_expects()

        class StaticChecker(Checker):
            def init(self):
                self.checker_name = "static checker"

            def calculate_output(self):
                outs, fetch_list = self.op_test._calc_output(
                    place, no_check_set=no_check_set, check_cinn=check_cinn
                )
                self.outputs = outs
                self.fetch_list = fetch_list
                if self.op_test.is_compared_with_fp32():
                    self.op_test.enable_cal_ref_output()
                    ref_outs, ref_fetch_list = self.op_test._calc_output(
                        place, no_check_set=no_check_set
                    )
                    self.op_test.disable_cal_ref_output()
                    self.ref_outputs = ref_outs
                    self.ref_fetch_list = ref_fetch_list

            def find_actual_value(self, name):
                idx = find_actual(name, self.fetch_list)
                actual = self.outputs[idx]
                actual_t = np.array(actual)
                return actual, actual_t

            def find_expect_value(self, name):
                idx = find_expect(name, self.ref_fetch_list)
                expect = self.ref_outputs[idx]
                expect_t = np.array(expect)
                return expect, expect_t

            def convert_uint16_to_float_ifneed(self, actual_np, expect_np):
                """
                judge whether convert current output and expect to uint16.
                return True | False
                """
                if actual_np.dtype == np.uint16:
                    if expect_np.dtype in [np.float32, np.float64]:
                        actual_np = convert_uint16_to_float(actual_np)
                    self.rtol = 1.0e-2
                elif actual_np.dtype == np.float16:
                    self.rtol = 1.0e-3
                else:
                    self.rtol = 1.0e-5
                if (
                    expect_np.dtype == np.uint16
                    and actual_np.dtype == np.uint16
                ):
                    nonlocal atol
                    expect_np = convert_uint16_to_float(expect_np)
                    actual_np = convert_uint16_to_float(actual_np)
                    atol = max(atol, 0.03)
                return actual_np, expect_np

        class DygraphChecker(Checker):
            def init(self):
                self.checker_name = "dygraph checker"

            def calculate_output(self):
                # we only check end2end api when check_dygraph=True
                self.is_python_api_test = True
                dygraph_outs = self.op_test._calc_python_api_output(place)
                if dygraph_outs is None:
                    self.is_python_api_test = False
                    # missing KernelSignature, fall back to eager middle output.
                    dygraph_outs = self.op_test._calc_dygraph_output(
                        place, no_check_set=no_check_set
                    )
                self.outputs = dygraph_outs
                if self.op_test.is_compared_with_fp32():
                    self.op_test.enable_cal_ref_output()
                    self.is_python_api_test = True
                    self.ref_outputs = self.op_test._calc_python_api_output(
                        place
                    )
                    if self.ref_outputs is None:
                        self.is_python_api_test = False
                        # missing KernelSignature, fall back to eager middle output.
                        self.ref_outputs = self.op_test._calc_dygraph_output(
                            place, no_check_set=no_check_set
                        )
                    self.op_test.disable_cal_ref_output()

            def _compare_numpy(self, name, actual_np, expect_np):
                expect_np = np.array(expect_np)
                assert (
                    actual_np.shape == expect_np.shape
                ), f"Operator ({self.op_type}) : Output ({name}) shape mismatch, expect shape is {expect_np.shape}, but actual shape is {actual_np.shape}"
                np.testing.assert_allclose(
                    actual_np,
                    expect_np,
                    atol=atol,
                    rtol=self.rtol if hasattr(self, 'rtol') else rtol,
                    equal_nan=equal_nan,
                    err_msg=(
                        "Operator ("
                        + self.op_type
                        + ") Output ("
                        + name
                        + ") has diff at "
                        + str(place)
                        + " in "
                        + self.checker_name
                    ),
                )

            def convert_uint16_to_float_ifneed(self, actual_np, expect_np):
                if actual_np.dtype == np.uint16:
                    self.rtol = 1.0e-2
                elif actual_np.dtype == np.float16:
                    self.rtol = 1.0e-3
                else:
                    self.rtol = 1.0e-5
                if self.op_test.is_bfloat16_op():
                    if actual_np.dtype == np.uint16:
                        actual_np = convert_uint16_to_float(actual_np)
                    if expect_np.dtype == np.uint16:
                        expect_np = convert_uint16_to_float(expect_np)
                return actual_np, expect_np

            def find_actual_value(self, name):
                with base.dygraph.base.guard(place=place):
                    imperative_actual = find_imperative_actual(
                        name, self.outputs, place
                    )
                    imperative_actual_t = np.array(
                        imperative_actual.value().get_tensor()
                    )
                    return imperative_actual, imperative_actual_t

            def find_expect_value(self, name):
                with base.dygraph.base.guard(place=place):
                    imperative_expect = find_imperative_expect(
                        name, self.ref_outputs, place
                    )
                    imperative_expect_t = np.array(
                        imperative_expect.value().get_tensor()
                    )
                    return imperative_expect, imperative_expect_t

            def _is_skip_name(self, name):
                # if in final state and kernel signature don't have name, then skip it.
                if (
                    self.is_python_api_test
                    and hasattr(self.op_test, "python_out_sig")
                    and name not in self.op_test.python_out_sig
                ):
                    return True
                return super()._is_skip_name(name)

        class PirChecker(Checker):
            def init(self):
                self.checker_name = "pir checker"

            def calculate_output(self):
                self.is_python_api_test = True
                pir_outs = self.op_test._calc_pir_output(place)
                if pir_outs is None:
                    self.is_python_api_test = False
                    # missing KernelSignature, fall back to eager middle output.
                    pir_outs = self.op_test._calc_dygraph_output(
                        place, no_check_set=no_check_set
                    )
                self.outputs = pir_outs

                if self.op_test.is_compared_with_fp32():
                    self.op_test.enable_cal_ref_output()
                    self.is_python_api_test = True
                    self.ref_outputs = self.op_test._calc_pir_output(place)
                    if self.ref_outputs is None:
                        self.is_python_api_test = False
                        # missing KernelSignature, fall back to eager middle output.
                        self.ref_outputs = self.op_test._calc_dygraph_output(
                            place, no_check_set=no_check_set
                        )
                    self.op_test.disable_cal_ref_output()

            def _compare_numpy(self, name, actual_np, expect_np):
                expect_np = np.array(expect_np)
                assert (
                    actual_np.shape == expect_np.shape
                ), f"Operator ({self.op_type}) : Output ({name}) shape mismatch, expect shape is {expect_np.shape}, but actual shape is {actual_np.shape}"
                np.testing.assert_allclose(
                    actual_np,
                    expect_np,
                    atol=atol,
                    rtol=self.rtol if hasattr(self, 'rtol') else rtol,
                    equal_nan=equal_nan,
                    err_msg=(
                        "Operator ("
                        + self.op_type
                        + ") Output ("
                        + name
                        + ") has diff at "
                        + str(place)
                        + " in "
                        + self.checker_name
                    ),
                )

            def convert_uint16_to_float_ifneed(self, actual_np, expect_np):
                if actual_np.dtype == np.uint16:
                    self.rtol = 1.0e-2
                elif actual_np.dtype == np.float16:
                    self.rtol = 1.0e-3
                else:
                    self.rtol = 1.0e-5
                if self.op_test.is_bfloat16_op():
                    if actual_np.dtype == np.uint16:
                        actual_np = convert_uint16_to_float(actual_np)
                    if expect_np.dtype == np.uint16:
                        expect_np = convert_uint16_to_float(expect_np)
                return actual_np, expect_np

            def find_pir_actual(self, target_name, pir_outs, place):
                for name in pir_outs:
                    if name == target_name:
                        return pir_outs[name][0]

                    sub_dict = pir_outs[name][0]
                    if isinstance(sub_dict, dict):
                        for key, value in sub_dict.items():
                            if key == target_name:
                                return value[0]

                raise AssertionError("No pir output named " + target_name)

            def find_pir_expect(self, target_name, dygraph_outs, place):
                for name in dygraph_outs:
                    if name == target_name:
                        return dygraph_outs[name][0]
                    var_list = dygraph_outs[name]
                    for i, var in enumerate(var_list):
                        if isinstance(var, list):
                            for tensor in var:
                                if tensor.name == target_name:
                                    return tensor
                        elif (
                            isinstance(var, paddle.Tensor)
                            and var.name == target_name
                        ):
                            return dygraph_outs[name][i]
                raise AssertionError("No pir ref_output named " + target_name)

            def find_actual_value(self, target_name):
                with paddle.pir.core.program_guard(
                    paddle.pir.core.default_main_program()
                ):
                    actual = self.find_pir_actual(
                        target_name, self.outputs, place
                    )
                    actual_t = np.array(actual)
                    return actual, actual_t

            def find_expect_value(self, target_name):
                with paddle.pir.core.program_guard(
                    paddle.pir.core.default_main_program()
                ):
                    expect = self.find_pir_expect(
                        target_name, self.ref_outputs, place
                    )
                    expect_t = np.array(expect)
                    return expect, expect_t

            def _is_skip_name(self, name):
                # if in final state and kernel signature don't have name, then skip it.
                if (
                    self.is_python_api_test
                    and hasattr(self.op_test, "python_out_sig")
                    and name not in self.op_test.python_out_sig
                ):
                    return True
                return super()._is_skip_name(name)

        class SymbolInferChecker(Checker):
            def check(self):
                """return None means ok, raise Error means failed."""
                self.init()
                self.infer_and_compare_symbol()

            def init(self):
                self.checker_name = "symbol infer checker"

            def infer_and_compare_symbol(self):
                """infer symbol and compare it with actual shape and data"""
                self.is_python_api_test = True
                self.op_test._infer_and_compare_symbol(place)

        # set some flags by the combination of arguments.
        if self.is_float16_op():
            self.dtype = np.float16
            self.__class__.dtype = self.dtype
            self.output_dtype = np.float16
        elif self.is_bfloat16_op():
            self.dtype = np.uint16
            self.__class__.dtype = self.dtype
            self.output_dtype = np.uint16
        else:
            self.infer_dtype_from_inputs_outputs(self.inputs, self.outputs)
        if (
            self.dtype == np.float64
            and self.op_type
            not in op_threshold_white_list.NEED_FIX_FP64_CHECK_OUTPUT_THRESHOLD_OP_LIST
        ):
            atol = 0

        if self.is_bfloat16_op():
            if self.is_mkldnn_op():
                check_dygraph = False

                if (
                    hasattr(self, 'force_fp32_output')
                    and self.force_fp32_output
                ):
                    atol = max(atol, 0.01)
                else:
                    atol = max(atol, 2)
            else:
                atol = max(atol, 0.01)

        if self.is_float16_op():
            atol = max(atol, 0.001)

        if no_check_set is not None:
            if (
                self.op_type
                not in no_check_set_white_list.no_check_set_white_list
            ):
                raise AssertionError(
                    f"no_check_set of op {self.op_type} must be set to None."
                )

        if check_prim:
            with paddle.pir_utils.OldIrGuard():
                prim_checker = PrimForwardChecker(self, place)
                prim_checker.check()
                # Support operators which are not in the NO_FP64_CHECK_GRAD_OP_LIST list can be test prim with fp32
                self.__class__.check_prim = True
                self.__class__.op_type = self.op_type

        if check_prim_pir:
            with paddle.pir_utils.IrGuard():
                prim_checker = PrimForwardChecker(self, place)
                prim_checker.check()
                # Support operators which are not in the NO_FP64_CHECK_GRAD_OP_LIST list can be test prim with fp32
                self.__class__.check_prim_pir = True
                self.__class__.op_type = self.op_type
        if only_check_prim:
            return

        if check_auto_parallel:
            if is_ban_auto_parallel_test(place):
                pass
            else:
                (
                    forward_test_info_path,
                    generated_forward_test_path,
                ) = get_test_info_and_generated_test_path(
                    self.__class__.__name__, self.op_type, backward=False
                )
                with auto_parallel_test_guard(
                    forward_test_info_path, generated_forward_test_path
                ):
                    dump_test_info(
                        self, place, forward_test_info_path, backward=False
                    )
                    python_api_info = {
                        "api_name": self.python_api.__name__,
                        "api_module": (
                            inspect.getmodule(self.python_api).__name__
                            if inspect.getmodule(
                                self.python_api
                            ).__name__.startswith("paddle")
                            else pathlib.Path(
                                inspect.getmodule(self.python_api).__file__
                            ).stem
                        ),
                    }
                    # code gen for auto parallel forward test
                    gen_auto_parallel_test_file(
                        check_grad=False,
                        test_info_path=forward_test_info_path,
                        test_file_path=generated_forward_test_path,
                        python_api_info=python_api_info,
                    )
                    runtime_envs = get_subprocess_runtime_envs(place)
                    start_command = get_subprocess_command(
                        runtime_envs["CUDA_VISIBLE_DEVICES"],
                        generated_forward_test_path,
                        log_dir=(
                            self.log_dir if hasattr(self, "log_dir") else None
                        ),
                    )
                    run_subprocess(start_command, runtime_envs, timeout=120)

        static_checker = StaticChecker(self, self.outputs)
        static_checker.check()
        outs, fetch_list = static_checker.outputs, static_checker.fetch_list

        if check_pir_onednn and isinstance(
            place, paddle.base.libpaddle.CPUPlace
        ):
            with pir_executor_guard():
                pir_onednn_static_checker = StaticChecker(self, self.outputs)
                pir_onednn_static_checker.check()

        if check_dygraph:
            dygraph_checker = DygraphChecker(self, self.outputs)
            dygraph_checker.check()
            dygraph_dygraph_outs = dygraph_checker.outputs

        if check_pir:
            if (
                type(place) is paddle.base.libpaddle.CPUPlace
                or type(place) is paddle.base.libpaddle.CUDAPlace
            ):
                with paddle.pir_utils.IrGuard():
                    pir_checker = PirChecker(self, self.outputs)
                    pir_checker.check()

        if check_pir and check_symbol_infer:
            if (
                type(place) is paddle.base.libpaddle.CPUPlace
                or type(place) is paddle.base.libpaddle.CUDAPlace
            ):
                with paddle.pir_utils.IrGuard():
                    symbol_checker = SymbolInferChecker(self, self.outputs)
                    symbol_checker.check()

        # Note(zhiqiu): inplace_atol should be only set when op doesn't ensure
        # computational consistency.
        # For example, group_norm uses AtomicAdd on CUDAPlace, which do not ensure
        # computation order when multiple threads write the same address. So the
        # result of group_norm is non-deterministic when datatype is float.
        # When inplace_atol is not None, the inplace check uses numpy.allclose
        # to check inplace result instead of numpy.array_equal.
        if inplace_atol is not None:
            warnings.warn(
                "inplace_atol should only be set when op doesn't ensure computational consistency, please check it!"
            )
        # Check inplace for given op, its grad op, its grad_grad op, etc.
        # No effect on original OpTest
        # Currently not support ParallelExecutor on XPUPlace.
        if not paddle.is_compiled_with_xpu() and not isinstance(
            place, core.CustomPlace
        ):
            self.check_inplace_output_with_place(
                place, no_check_set=no_check_set, inplace_atol=inplace_atol
            )

        if check_dygraph:
            return outs, dygraph_dygraph_outs, fetch_list
        else:
            return outs, fetch_list

    def check_compile_vs_runtime(self, fetch_list, fetch_outs):
        def find_fetch_index(target_name, fetch_list):
            found = [
                i
                for i, var_name in enumerate(fetch_list)
                if var_name == target_name
            ]
            if len(found) == 0:
                return -1
            else:
                self.assertTrue(
                    len(found) == 1,
                    f"Found {len(found)} {target_name}",
                )
                return found[0]

        for name in self.op.desc.output_names():
            var_names = self.op.desc.output(name)
            for var_name in var_names:
                i = find_fetch_index(var_name, fetch_list)
                if i == -1:
                    # The output is dispensable or intermediate.
                    break
                out = fetch_outs[i]
                if isinstance(out, core.DenseTensor):
                    lod_level_runtime = len(out.lod())
                else:
                    if isinstance(out, core.DenseTensorArray):
                        warnings.warn(
                            "The check of DenseTensorArray's lod_level is not implemented now!"
                        )
                    lod_level_runtime = 0

                var = self.program.global_block().var(var_name)
                if var.type == core.VarDesc.VarType.DENSE_TENSOR:
                    lod_level_compile = var.lod_level
                else:
                    lod_level_compile = 0
                self.assertEqual(
                    lod_level_compile,
                    lod_level_runtime,
                    "The lod_level of Output ("
                    + name
                    + ") is different between compile-time and runtime ("
                    + str(lod_level_compile)
                    + " vs "
                    + str(lod_level_runtime)
                    + ")",
                )

    def _get_places(self):
        if self.dtype == np.float16 or self.dtype == "float16":
            if core.is_compiled_with_cuda() and core.op_support_gpu(
                self.op_type
            ):
                place = core.CUDAPlace(0)
                if core.is_float16_supported(place):
                    return [place]
                else:
                    return []
            else:
                return []
        places = []
        cpu_only = self._cpu_only if hasattr(self, '_cpu_only') else False
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in [
                '1',
                'true',
                'on',
            ]
            or not (
                core.is_compiled_with_cuda()
                and core.op_support_gpu(self.op_type)
                and not cpu_only
            )
            or self.op_type
            in [
                'gaussian_random',
                'lrn',
            ]
        ):
            places.append(base.CPUPlace())
        if (
            core.is_compiled_with_cuda()
            and core.op_support_gpu(self.op_type)
            and not cpu_only
        ):
            places.append(core.CUDAPlace(0))
        return places

    def check_output(
        self,
        atol=1e-5,
        rtol=1e-5,
        no_check_set=None,
        equal_nan=False,
        check_dygraph=True,
        check_prim=False,
        check_prim_pir=False,
        inplace_atol=None,
        check_cinn=False,
        only_check_prim=False,
        check_pir=False,
        check_auto_parallel=False,
        check_pir_onednn=False,
        check_symbol_infer=True,
    ):
        self.__class__.op_type = self.op_type
        if self.is_mkldnn_op():
            self.__class__.use_mkldnn = True

        if self.is_xpu_op():
            self.__class__.use_xpu = True

        if hasattr(self, "use_custom_device") and self.use_custom_device:
            check_dygraph = False

        places = self._get_places()
        for place in places:
            res = self.check_output_with_place(
                place,
                atol,
                rtol,
                no_check_set,
                equal_nan,
                check_dygraph=check_dygraph,
                check_prim=check_prim,
                check_prim_pir=check_prim_pir,
                only_check_prim=only_check_prim,
                inplace_atol=inplace_atol,
                check_cinn=check_cinn,
                check_pir=check_pir,
                check_auto_parallel=check_auto_parallel,
                check_pir_onednn=check_pir_onednn,
                check_symbol_infer=check_symbol_infer,
            )
            if not res and only_check_prim:
                continue
            if check_dygraph:
                outs, dygraph_dygraph_outs, fetch_list = res
            else:
                outs, fetch_list = res
            if (
                self.op_type
                not in compile_vs_runtime_white_list.COMPILE_RUN_OP_WHITE_LIST
            ):
                if os.getenv("FLAGS_enable_pir_in_executor"):
                    return
                self.check_compile_vs_runtime(fetch_list, outs)

    def check_output_customized(
        self, checker, custom_place=None, check_pir=False
    ):
        self.__class__.op_type = self.op_type
        places = self._get_places()
        if custom_place:
            places.append(custom_place)
        for place in places:
            outs = self.calc_output(place)
            outs = [np.array(out) for out in outs]
            outs.sort(key=len)
            checker(outs)
            if check_pir:
                with paddle.pir_utils.IrGuard():
                    outs_p = self._calc_pir_output(place)
                    outs_p = [outs_p[out] for out in outs_p]
                    outs_p.sort(key=len)
                    checker(outs_p[0])

    def check_output_with_place_customized(
        self, checker, place, check_pir=False
    ):
        outs = self.calc_output(place)
        outs = [np.array(out) for out in outs]
        outs.sort(key=len)
        checker(outs)
        if check_pir:
            with paddle.pir_utils.IrGuard():
                outs_p = self._calc_pir_output(place)
                outs_p = [outs_p[out][0] for out in outs_p]
                outs_p.sort(key=len)
                checker(outs_p)

    def _assert_is_close(
        self,
        numeric_grads,
        analytic_grads,
        names,
        max_relative_error,
        msg_prefix,
        atol=1e-5,
    ):
        for a, b, name in zip(numeric_grads, analytic_grads, names):
            assert tuple(a.shape) == tuple(
                b.shape
            ), f"Operator ({self.op_type}) : Output ({name}) gradient shape mismatch, expect shape is {a.shape}, but actual shape is {b.shape}"
            # Used by bfloat16 for now to solve precision problem
            if self.is_bfloat16_op():
                if a.size == 0:
                    self.assertTrue(b.size == 0)
                np.testing.assert_allclose(
                    b,
                    a,
                    rtol=max_relative_error,
                    atol=atol,
                    equal_nan=False,
                    err_msg=(
                        f"Operator {self.op_type} error, {msg_prefix} variable {name} (shape: {a.shape}, dtype: {self.dtype}) max gradient diff over limit"
                    ),
                )
            else:
                # It asserts np.abs(a - b) / np.abs(a) < max_relative_error, in which
                # max_relative_error is 1e-7. According to the value of np.abs(a), we
                # change np.abs(a) to achieve dynamic threshold. For example, if
                # the value of np.abs(a) is between 1e-10 and 1e-8, we set np.abs(a)*=1e4.
                # Therefore, it asserts np.abs(a - b) / (np.abs(a)*1e4) < max_relative_error,
                # which is the same as np.abs(a - b) / np.abs(a) < max_relative_error*1e4.
                abs_a = np.abs(a)
                if abs_a.ndim > 0:
                    if (
                        self.dtype == np.float64
                        and self.op_type
                        not in op_threshold_white_list.NEED_FIX_FP64_CHECK_GRAD_THRESHOLD_OP_LIST
                    ):
                        abs_a[abs_a < 1e-10] = 1e-3
                        abs_a[
                            np.logical_and(abs_a > 1e-10, abs_a <= 1e-8)
                        ] *= 1e4
                        abs_a[
                            np.logical_and(abs_a > 1e-8, abs_a <= 1e-6)
                        ] *= 1e2
                    elif self.is_bfloat16_op():
                        abs_a[abs_a < 1e-2] = 1
                    else:
                        abs_a[abs_a < 1e-3] = 1
                elif abs_a.ndim == 0:
                    if (
                        self.dtype == np.float64
                        and self.op_type
                        not in op_threshold_white_list.NEED_FIX_FP64_CHECK_GRAD_THRESHOLD_OP_LIST
                    ):
                        if abs_a < 1e-10:
                            abs_a = 1e-3
                        elif abs_a > 1e-10 and abs_a <= 1e-8:
                            abs_a = abs_a * 1e4
                        elif abs_a > 1e-8 and abs_a <= 1e-6:
                            abs_a = abs_a * 1e2
                    elif self.is_bfloat16_op():
                        abs_a = 1 if abs_a < 1e-2 else abs_a
                    else:
                        abs_a = 1 if abs_a < 1e-3 else abs_a

                if self.dtype == np.bool_:
                    diff_mat = np.abs(a ^ b) / abs_a
                else:
                    diff_mat = np.abs(a - b) / abs_a
                max_diff = np.max(diff_mat)

                def err_msg():
                    offset = np.argmax(diff_mat > max_relative_error)
                    return (
                        f"Operator {self.op_type} error, {msg_prefix} variable {name} (shape: {a.shape!s}, dtype: {self.dtype}) "
                        f"max gradient diff {max_diff:e} over limit {max_relative_error:e}, "
                        f"the first error element is {offset}, expected {a.flatten()[offset].item():e}, but got {b.flatten()[offset].item():e}."
                    )

                self.assertLessEqual(max_diff, max_relative_error, err_msg())

    def _check_grad_helper(self):
        if self.is_float16_op():
            self.dtype = np.float16
            self.__class__.dtype = self.dtype
            self.output_dtype = np.float16
        elif self.is_bfloat16_op():
            self.dtype = np.uint16
            self.__class__.dtype = self.dtype
            self.output_dtype = np.uint16
        else:
            self.infer_dtype_from_inputs_outputs(self.inputs, self.outputs)
        self.__class__.op_type = self.op_type
        self.__class__.exist_check_grad = True
        if self.dtype == np.float64:
            self.__class__.exist_fp64_check_grad = True

    def check_grad(
        self,
        inputs_to_check,
        output_names,
        no_grad_set=None,
        numeric_grad_delta=0.005,
        in_place=False,
        max_relative_error=0.005,
        user_defined_grads=None,
        user_defined_grad_outputs=None,
        check_dygraph=True,
        check_prim=False,
        check_prim_pir=False,
        only_check_prim=False,
        atol=1e-5,
        check_cinn=False,
        check_pir=False,
        check_auto_parallel=False,
        check_pir_onednn=False,
    ):
        if hasattr(self, "use_custom_device") and self.use_custom_device:
            check_dygraph = False

        self._check_grad_helper()
        places = self._get_places()
        for place in places:
            self.check_grad_with_place(
                place,
                inputs_to_check,
                output_names,
                no_grad_set,
                numeric_grad_delta,
                in_place,
                max_relative_error,
                user_defined_grads,
                user_defined_grad_outputs,
                check_dygraph=check_dygraph,
                check_prim=check_prim,
                check_prim_pir=check_prim_pir,
                only_check_prim=only_check_prim,
                atol=atol,
                check_cinn=check_cinn,
                check_pir=check_pir,
                check_auto_parallel=check_auto_parallel,
                check_pir_onednn=check_pir_onednn,
            )

    def check_grad_with_place_for_static(
        self,
        user_defined_grads,
        inputs_to_check,
        place,
        output_names,
        no_grad_set,
        user_defined_grad_outputs,
        numeric_place,
        numeric_grad_delta,
        in_place,
        check_cinn,
        max_relative_error,
        atol,
    ):
        if user_defined_grads is None and self.is_compared_with_fp32():
            self.enable_cal_ref_output()
            numeric_grads = self._get_gradient(
                inputs_to_check,
                place,
                output_names,
                no_grad_set,
                user_defined_grad_outputs,
            )
            self.disable_cal_ref_output()
        else:
            numeric_grads = user_defined_grads or [
                get_numeric_gradient(
                    numeric_place,
                    self.scope,
                    self.op,
                    self.inputs,
                    input_to_check,
                    output_names,
                    delta=numeric_grad_delta,
                    in_place=in_place,
                )
                for input_to_check in inputs_to_check
            ]

        analytic_grads = self._get_gradient(
            inputs_to_check,
            place,
            output_names,
            no_grad_set,
            user_defined_grad_outputs,
            check_cinn=check_cinn,
        )
        # comparison of bf16 results will happen as fp32
        # loop over list of grads and convert bf16 to fp32

        fp32_analytic_grads = []
        for grad in analytic_grads:
            if grad.dtype == np.uint16:
                grad = convert_uint16_to_float(grad)
                max_relative_error = max(max_relative_error, 0.01)
            fp32_analytic_grads.append(grad)
        analytic_grads = fp32_analytic_grads

        fp32_numeric_grads = []
        for grad in numeric_grads:
            if grad.dtype == np.uint16:
                grad = convert_uint16_to_float(grad)
                max_relative_error = max(max_relative_error, 0.01)
            fp32_numeric_grads.append(grad)
        numeric_grads = fp32_numeric_grads

        if self.is_float16_op():
            max_relative_error = max(max_relative_error, 0.001)
        self._assert_is_close(
            numeric_grads,
            analytic_grads,
            inputs_to_check,
            max_relative_error,
            f"Gradient Check On {place}",
            atol=atol,
        )

        return numeric_grads

    def check_grad_with_place(
        self,
        place,
        inputs_to_check,
        output_names,
        no_grad_set=None,
        numeric_grad_delta=0.005,
        in_place=False,
        max_relative_error=0.005,
        user_defined_grads=None,
        user_defined_grad_outputs=None,
        check_dygraph=True,
        check_prim=False,
        check_prim_pir=False,
        only_check_prim=False,
        numeric_place=None,
        atol=1e-5,
        check_cinn=False,
        check_pir=False,
        check_auto_parallel=False,
        check_pir_onednn=False,
    ):
        if hasattr(self, "use_custom_device") and self.use_custom_device:
            check_dygraph = False

        if not self.is_mkldnn_op():
            set_flags({"FLAGS_use_mkldnn": False})

        core._set_prim_all_enabled(False)
        core.set_prim_eager_enabled(False)
        if check_prim:
            with paddle.pir_utils.OldIrGuard():
                self._check_grad_helper()
                prim_grad_checker = PrimGradChecker(
                    self,
                    place,
                    inputs_to_check,
                    output_names,
                    no_grad_set,
                    user_defined_grad_outputs,
                )
                prim_grad_checker.check()
                # Support operators which are not in the NO_FP64_CHECK_GRAD_OP_LIST list can be test prim with fp32
                self.__class__.check_prim = True

        if check_prim_pir:
            with paddle.pir_utils.IrGuard():
                self._check_grad_helper()
                prim_grad_checker = PrimGradChecker(
                    self,
                    place,
                    inputs_to_check,
                    output_names,
                    no_grad_set,
                    user_defined_grad_outputs,
                )
                prim_grad_checker.check()
                # Support operators which are not in the NO_FP64_CHECK_GRAD_OP_LIST list can be test prim with fp32
                self.__class__.check_prim_pir = True

        if only_check_prim:
            return

        if check_auto_parallel:
            if is_ban_auto_parallel_test(place):
                pass
            else:
                (
                    grad_test_info_path,
                    generated_grad_test_path,
                ) = get_test_info_and_generated_test_path(
                    self.__class__.__name__, self.op_type, backward=True
                )
                with auto_parallel_test_guard(
                    grad_test_info_path, generated_grad_test_path
                ):
                    backward_extra_test_info = {}
                    backward_extra_test_info["inputs_to_check"] = (
                        inputs_to_check
                    )
                    backward_extra_test_info["output_names"] = output_names
                    backward_extra_test_info["no_grad_set"] = no_grad_set
                    backward_extra_test_info["user_defined_grad_outputs"] = (
                        user_defined_grad_outputs
                    )
                    dump_test_info(
                        self,
                        place,
                        grad_test_info_path,
                        backward=True,
                        backward_extra_test_info=backward_extra_test_info,
                    )
                    python_api_info = {
                        "api_name": self.python_api.__name__,
                        "api_module": (
                            inspect.getmodule(self.python_api).__name__
                            if inspect.getmodule(
                                self.python_api
                            ).__name__.startswith("paddle")
                            else pathlib.Path(
                                inspect.getmodule(self.python_api).__file__
                            ).stem
                        ),
                    }
                    # code gen for auto parallel grad test
                    gen_auto_parallel_test_file(
                        check_grad=False,
                        test_info_path=grad_test_info_path,
                        test_file_path=generated_grad_test_path,
                        python_api_info=python_api_info,
                    )
                    runtime_envs = get_subprocess_runtime_envs(place)

                    num_devices = len(
                        runtime_envs["CUDA_VISIBLE_DEVICES"].split(",")
                    )
                    if num_devices > paddle.device.cuda.device_count():
                        self.skipTest("number of GPUs is not enough")

                    start_command = get_subprocess_command(
                        runtime_envs["CUDA_VISIBLE_DEVICES"],
                        generated_grad_test_path,
                        log_dir=(
                            self.log_dir if hasattr(self, "log_dir") else None
                        ),
                    )
                    run_subprocess(start_command, runtime_envs, timeout=120)

        self.scope = core.Scope()
        op_inputs = self.inputs if hasattr(self, "inputs") else {}
        op_outputs = self.outputs if hasattr(self, "outputs") else {}
        op_attrs = self.attrs if hasattr(self, "attrs") else {}
        self._check_grad_helper()
        if self.is_bfloat16_op():
            if self.is_mkldnn_op():
                check_dygraph = False
            atol = max(atol, 0.01)

        if self.is_float16_op():
            atol = max(atol, 0.001)

        if (
            self.dtype == np.float64
            and self.op_type
            not in op_threshold_white_list.NEED_FIX_FP64_CHECK_GRAD_THRESHOLD_OP_LIST
        ):
            numeric_grad_delta = 1e-5
            max_relative_error = 1e-7

        cache_list = None
        if hasattr(self, "cache_name_list"):
            cache_list = self.cache_name_list

        # oneDNN numeric gradient should use CPU kernel
        use_onednn = False
        if op_attrs.get("use_mkldnn"):
            op_attrs["use_mkldnn"] = False
            use_onednn = True
        if hasattr(self, "attrs"):
            for k, v in self.attrs.items():
                if isinstance(v, paddle.base.core.DataType):
                    self.attrs[k] = paddle.pir.core.datatype_to_vartype[v]

        self.op = create_op(
            self.scope,
            self.op_type,
            op_inputs,
            op_outputs,
            op_attrs,
            cache_list=cache_list,
        )

        if use_onednn:
            op_attrs["use_mkldnn"] = True

        if no_grad_set is None:
            no_grad_set = set()
        else:
            if (
                (self.op_type not in no_grad_set_white_list.NEED_TO_FIX_OP_LIST)
                and (
                    self.op_type not in no_grad_set_white_list.NOT_CHECK_OP_LIST
                )
                and (not self.is_bfloat16_op())
            ):
                raise AssertionError(
                    "no_grad_set must be None, op_type is "
                    + self.op_type
                    + " Op."
                )

        for input_to_check in inputs_to_check:
            set_input(self.scope, self.op, self.inputs, place)
            tensor_to_check = self.scope.find_var(input_to_check).get_tensor()
            tensor_size = functools.reduce(
                lambda a, b: a * b, tensor_to_check.shape(), 1
            )
            tensor_ndim = len(tensor_to_check.shape())
            # for 0D Tensor, it's additional case for OP, so not raise error
            if tensor_ndim > 0 and tensor_size < 100:
                self.__class__.input_shape_is_large = False

        if type(output_names) is not list:
            output_names = [output_names]

        if numeric_place is None:
            numeric_place = place

        with paddle.pir_utils.OldIrGuard():
            numeric_grads = self.check_grad_with_place_for_static(
                user_defined_grads,
                inputs_to_check,
                place,
                output_names,
                no_grad_set,
                user_defined_grad_outputs,
                numeric_place,
                numeric_grad_delta,
                in_place,
                check_cinn,
                max_relative_error,
                atol,
            )

        if check_pir_onednn and isinstance(
            place, paddle.base.libpaddle.CPUPlace
        ):
            with pir_executor_guard():
                self.check_grad_with_place_for_static(
                    user_defined_grads,
                    inputs_to_check,
                    place,
                    output_names,
                    no_grad_set,
                    user_defined_grad_outputs,
                    numeric_place,
                    numeric_grad_delta,
                    in_place,
                    check_cinn,
                    max_relative_error,
                    atol,
                )

        if check_dygraph:
            with base.dygraph.base.guard(place):
                dygraph_dygraph_grad = self._get_dygraph_grad(
                    inputs_to_check,
                    place,
                    output_names,
                    user_defined_grad_outputs,
                    no_grad_set,
                    check_dygraph,
                )
                fp32_grads = []
                for grad in dygraph_dygraph_grad:
                    if grad.dtype == np.uint16:
                        grad = convert_uint16_to_float(grad)
                        max_relative_error = max(max_relative_error, 0.03)
                    fp32_grads.append(grad)
                dygraph_dygraph_grad = fp32_grads
                self._assert_is_close(
                    numeric_grads,
                    dygraph_dygraph_grad,
                    inputs_to_check,
                    max_relative_error,
                    f"Gradient Check On {place}",
                    atol=atol,
                )

        # get pir gradient
        if check_pir:
            if (
                type(place) is paddle.base.libpaddle.CPUPlace
                or type(place) is paddle.base.libpaddle.CUDAPlace
            ):
                with paddle.pir_utils.IrGuard():
                    pir_grad = self._get_ir_gradient(
                        inputs_to_check,
                        place,
                        output_names,
                        user_defined_grad_outputs,
                        no_grad_set,
                    )
                fp32_analytic_grads = []
                for grad in pir_grad:
                    if grad.dtype == np.uint16:
                        grad = convert_uint16_to_float(grad)
                        max_relative_error = max(max_relative_error, 0.01)
                    fp32_analytic_grads.append(grad)
                pir_grad = fp32_analytic_grads
                if self.is_float16_op():
                    max_relative_error = max(max_relative_error, 0.01)
                self._assert_is_close(
                    numeric_grads,
                    pir_grad,
                    inputs_to_check,
                    max_relative_error,
                    f"Gradient Check On {place}",
                    atol=atol,
                )

    def _find_var_in_dygraph(self, output_vars, name):
        if name in output_vars:
            return output_vars[name]
        else:
            for output_vars_index in output_vars:
                for output_vars_selected in output_vars[output_vars_index]:
                    if isinstance(output_vars_selected, list):
                        for tensor in output_vars_selected:
                            if tensor.name == name:
                                return [tensor]
                    elif isinstance(output_vars_selected, paddle.Tensor):
                        if output_vars_selected.name == name:
                            return [output_vars_selected]
        raise AssertionError(name, " not in outputs:", output_vars.keys())

    def _get_dygraph_grad(
        self,
        inputs_to_check,
        place,
        output_names,
        user_defined_grad_outputs=None,
        no_grad_set=None,
        check_dygraph=True,
    ):
        if hasattr(self, "use_custom_device") and self.use_custom_device:
            check_dygraph = False

        with base.dygraph.base.guard(place=place):
            block = base.framework.default_main_program().global_block()

            op_proto = OpProtoHolder.instance().get_op_proto(self.op_type)

            # prepare input variable
            inputs, inputs_grad_dict = self.append_input_output_for_dygraph(
                op_proto, self.inputs, True, True, block
            )

            # prepare output variable
            outputs = self.append_input_output_for_dygraph(
                op_proto, self.outputs, False, False, block
            )

            # prepare attributes
            attrs_outputs = {}
            if hasattr(self, "attrs"):
                for attrs_name in self.attrs:
                    if self.attrs[attrs_name] is not None:
                        attrs_outputs[attrs_name] = self.attrs[attrs_name]

            if check_dygraph:
                dygraph_outputs = self._calc_python_api_output(
                    place, inputs, outputs
                )
                if dygraph_outputs is None:
                    # missing KernelSignature, fall back to eager middle output.
                    dygraph_outputs = self._calc_dygraph_output(
                        place, egr_inps=inputs, egr_oups=outputs
                    )

            outputs = dygraph_outputs

            if self.dtype == np.uint16:
                cast_inputs = []
                for output_name in output_names:
                    cast_input = self._find_var_in_dygraph(outputs, output_name)
                    cast_inputs = cast_inputs + cast_input
                cast_outputs = []
                for cast_input in cast_inputs:
                    if isinstance(cast_input, paddle.Tensor):
                        cast_outputs.append(
                            paddle.cast(cast_input, paddle.float32)
                        )
                    else:
                        raise TypeError(
                            f"Unsupported test data type {type(cast_input)}."
                        )

                outputs = {}
                for i in range(len(output_names)):
                    outputs.update({output_names[i]: [cast_outputs[i]]})
            outputs_valid = {}
            for output_name in output_names:
                outputs_valid[output_name] = self._find_var_in_dygraph(
                    outputs, output_name
                )

            if user_defined_grad_outputs is None:
                if len(outputs_valid) == 1:
                    for outputs_valid_key in outputs_valid:
                        loss = paddle.mean(outputs_valid[outputs_valid_key][0])
                else:
                    avg_sum = []
                    for cur_loss in outputs_valid:
                        cur_avg_loss = paddle.mean(outputs_valid[cur_loss][0])
                        avg_sum.append(cur_avg_loss)
                    loss_sum = paddle.add_n(avg_sum)
                    loss = paddle.scale(
                        loss_sum, scale=1.0 / float(len(avg_sum))
                    )
                loss.backward()

                fetch_list_grad = []
                for inputs_to_check_name in inputs_to_check:
                    a = inputs_grad_dict[inputs_to_check_name].gradient()
                    fetch_list_grad.append(a)
                return fetch_list_grad
            else:
                # user_defined_grad_outputs here are numpy arrays
                if not isinstance(user_defined_grad_outputs, list):
                    user_defined_grad_outputs = [user_defined_grad_outputs]
                grad_outputs = []
                for grad_out_value in user_defined_grad_outputs:
                    grad_outputs.append(paddle.to_tensor(grad_out_value))
                # delete the inputs which no need to calculate grad
                for no_grad_val in no_grad_set:
                    del inputs[no_grad_val]
                grad_inputs = paddle.grad(
                    outputs=paddle.utils.flatten(outputs),
                    inputs=paddle.utils.flatten(inputs),
                    grad_outputs=grad_outputs,
                )
                return [grad.numpy(False) for grad in grad_inputs]

    @staticmethod
    def _numpy_to_lod_tensor(np_value, lod, place):
        tensor = core.DenseTensor()
        tensor.set(np_value, place)
        if lod is not None:
            tensor.set_recursive_sequence_lengths(lod)
        return tensor

    @staticmethod
    def np_dtype_to_base_dtype(input):
        return input

    @staticmethod
    def base_dtype_to_np_dtype(self, dtype):
        return dtype

    @staticmethod
    def np_value_to_base_value(input):
        return input

    def cast_bf16_output(self, block, cast_inputs):
        output_names = []
        for i in range(0, len(cast_inputs)):
            cast_output = block.create_var(
                dtype="float32", shape=cast_inputs[i].shape
            )
            cast_op = block.append_op(
                inputs={"X": cast_inputs[i]},
                outputs={"Out": cast_output},
                type="cast",
                attrs={
                    "in_dtype": core.VarDesc.VarType.BF16,
                    "out_dtype": core.VarDesc.VarType.FP32,
                },
            )
            cast_op.desc.infer_var_type(block.desc)
            cast_op.desc.infer_shape(block.desc)
            output_names.append(cast_output.name)
        return output_names

    def _check_ir_grad_output(
        self, place, program, scope, feed_dict, fetch_list, gradients
    ):
        if os.getenv("FLAGS_PIR_OPTEST") is None:
            return
        if os.getenv("FLAGS_PIR_OPTEST_WHITE_LIST") is None:
            return
        if self.check_prim or self.check_prim_pir:
            return
        if self._check_cinn:
            return

        stored_flag = get_flags(
            [
                'FLAGS_enable_pir_in_executor',
                "FLAGS_pir_apply_inplace_pass",
            ]
        )
        try:
            set_flags(
                {
                    "FLAGS_enable_pir_in_executor": True,
                    "FLAGS_pir_apply_inplace_pass": 0,
                }
            )
            executor = Executor(place)
            new_gradients = list(
                map(
                    np.array,
                    executor.run(
                        program,
                        feed_dict,
                        fetch_list,
                        scope=scope,
                        return_numpy=False,
                    ),
                )
            )

            check_method = np.testing.assert_array_equal
            if os.getenv("FLAGS_PIR_OPTEST_RELAX_CHECK", None) == "True":

                def relaxed_check_method(x, y, err_msg):
                    atol = 1e-6
                    rtol = 1e-6
                    if x.dtype == np.float16:
                        atol = 1e-5
                        rtol = 1e-3
                    np.testing.assert_allclose(
                        x, y, err_msg=err_msg, atol=atol, rtol=rtol
                    )

                check_method = relaxed_check_method

            if os.getenv("FLAGS_PIR_NO_CHECK", None) == "True":

                def no_check_method(x, y, err_msg):
                    pass

                check_method = no_check_method

            for i in range(len(new_gradients)):
                check_method(
                    gradients[i],
                    new_gradients[i],
                    err_msg='Operator GradCheck ('
                    + self.op_type
                    + ') has diff at '
                    + str(place)
                    + '\nExpect '
                    + str(gradients[i])
                    + '\n'
                    + 'But Got'
                    + str(new_gradients[i])
                    + ' in class '
                    + self.__class__.__name__,
                )
        finally:
            set_flags(stored_flag)

    def _get_gradient(
        self,
        input_to_check,
        place,
        output_names,
        no_grad_set,
        user_defined_grad_outputs=None,
        parallel=False,
        check_cinn=False,
    ):
        with paddle.pir_utils.OldIrGuard():
            prog = Program()
            scope = core.Scope()
            ir_scope = core.Scope()
            block = prog.global_block()
            self._append_ops(block)

            inputs = self._get_inputs(block)
            outputs = self._get_outputs(block)
            feed_dict = self.feed_var(inputs, place)

            if user_defined_grad_outputs is None:
                if self.dtype == np.uint16 and not self.is_calc_ref:
                    cast_inputs = list(map(block.var, output_names))
                    if self.op_type in ["broadcast_tensors", "meshgrid"]:
                        output_names = self.cast_bf16_output(block, cast_inputs)
                    else:
                        cast_outputs = block.create_var(
                            dtype="float32", shape=cast_inputs[0].shape
                        )
                        cast_op = block.append_op(
                            inputs={"X": cast_inputs},
                            outputs={"Out": cast_outputs},
                            type="cast",
                            attrs={
                                "in_dtype": core.VarDesc.VarType.BF16,
                                "out_dtype": core.VarDesc.VarType.FP32,
                            },
                        )
                        cast_op.desc.infer_var_type(block.desc)
                        cast_op.desc.infer_shape(block.desc)
                        output_names = [cast_outputs.name]
                loss = append_loss_ops(block, output_names)
                param_grad_list = append_backward(
                    loss=loss,
                    parameter_list=input_to_check,
                    no_grad_set=no_grad_set,
                )
                fetch_list = [g for p, g in param_grad_list]
            else:
                assert (
                    parallel is False
                ), "unsupported parallel mode when giving custom grad outputs."
                # user_defined_grad_outputs here are numpy arrays
                if not isinstance(user_defined_grad_outputs, list):
                    user_defined_grad_outputs = [user_defined_grad_outputs]
                grad_outputs = []
                for grad_out_value in user_defined_grad_outputs:
                    # `persistable` is used to avoid executor create new var in local scope
                    var = block.create_var(
                        shape=grad_out_value.shape,
                        dtype=grad_out_value.dtype,
                        persistable=True,
                    )
                    true_var = scope.var(var.name)
                    tensor = true_var.get_tensor()
                    tensor.set(grad_out_value, place)
                    grad_outputs.append(var)
                    if os.getenv("FLAGS_PIR_OPTEST") is not None:
                        ir_true_var = ir_scope.var(var.name)
                        ir_tensor = ir_true_var.get_tensor()
                        ir_tensor.set(grad_out_value, place)

                targets = [
                    outputs[name] for name in outputs if name in output_names
                ]
                inputs = [
                    inputs[name] for name in input_to_check if name in inputs
                ]
                grad_inputs = paddle.static.gradients(
                    targets, inputs, grad_outputs, no_grad_set
                )
                fetch_list = [grad.name for grad in grad_inputs]

            enable_cinn_test = check_cinn and self._enable_check_cinn_test(
                place, feed_dict, outputs
            )
            if enable_cinn_test:
                if hasattr(self, 'cinn_atol'):
                    self.atol = self.cinn_atol
                if hasattr(self, 'cinn_rtol'):
                    self.rtol = self.cinn_rtol

            if parallel or enable_cinn_test:
                use_cuda = False
                if isinstance(place, base.CUDAPlace):
                    use_cuda = True

                build_strategy = None
                if enable_cinn_test:
                    build_strategy = base.BuildStrategy()
                    build_strategy.build_cinn_pass = check_cinn
                    self._check_cinn = True

                compiled_prog = base.CompiledProgram(prog, build_strategy)
                prog = compiled_prog
            executor = base.Executor(place)
            res = list(
                map(
                    np.array,
                    executor.run(
                        prog,
                        feed_dict,
                        fetch_list,
                        scope=scope,
                        return_numpy=False,
                    ),
                )
            )

            self._check_ir_grad_output(
                place, prog, ir_scope, feed_dict, fetch_list, res
            )

        return res

    def _find_var_in_pir(self, output_vars, target_name):
        for name in output_vars:
            if name == target_name:
                return output_vars[name]

            sub_dict = output_vars[name][0]
            if isinstance(sub_dict, dict):
                for key, value in sub_dict.items():
                    if key == target_name:
                        return value
        raise AssertionError(
            target_name, " not in outputs:", output_vars.keys()
        )

    def _get_ir_gradient(
        self,
        inputs_to_check,
        place,
        output_names,
        user_defined_grad_outputs=None,
        no_grad_set=None,
    ):
        def construct_output_dict_by_kernel_sig(ret_tuple, output_sig):
            if hasattr(self, "python_out_sig"):
                output_sig = self.python_out_sig
            if not isinstance(ret_tuple, (tuple, list)):
                ret_tuple = [ret_tuple]
            if len(output_sig) == len(ret_tuple):
                # [assumption]: we assume {"Out": [Tensor]}
                return {a: [b] for a, b in zip(output_sig, ret_tuple)}
            else:
                # [assumption]: return multi-Tensor in a single output. such as paddle.split()
                assert (
                    len(output_sig) == 1
                ), "Don't support multi-output with multi-tensor output. (May be you can use set `python_out_sig`, see `test_squeeze2_op` as a example.)"
                return {output_sig[0]: ret_tuple}

        # get kernel signature
        kernel_sig = self.get_kernel_signature(place)
        ir_program = paddle.static.Program()
        with paddle.static.program_guard(ir_program):
            with scope_guard(Scope()):
                # prepare inps attributes feed
                (
                    static_inputs,
                    attrs,
                    inputs_dict,
                    feed,
                ) = self.get_ir_input_attr_dict_and_feed(stop_gradient=False)
                # prepare args
                args = OpTestUtils.prepare_python_api_arguments(
                    self.python_api,
                    static_inputs,
                    attrs,
                    kernel_sig,
                    target_dtype=paddle.pir.core.DataType,
                )
                inputs_sig, attrs_sig, outputs_sig = kernel_sig
                args = OpTestUtils.assumption_assert_and_transform(
                    args, len(inputs_sig)
                )
                grad_outputs = []
                if user_defined_grad_outputs is not None:
                    # user_defined_grad_outputs here are numpy arrays
                    if not isinstance(user_defined_grad_outputs, list):
                        user_defined_grad_outputs = [user_defined_grad_outputs]
                    for grad_out_value, idx in zip(
                        user_defined_grad_outputs,
                        range(len(user_defined_grad_outputs)),
                    ):
                        grad_val = paddle.static.data(
                            name=f'val_grad_{idx}',
                            shape=grad_out_value.shape,
                            dtype=grad_out_value.dtype,
                        )
                        grad_outputs.append(grad_val)
                        feed.update({f'val_grad_{idx}': grad_out_value})
                    # delete the inputs which no need to calculate grad
                    for no_grad_val in no_grad_set:
                        del static_inputs[no_grad_val]

                ret_tuple = self.python_api(*args)
                outputs = construct_output_dict_by_kernel_sig(
                    ret_tuple, outputs_sig
                )
                if hasattr(self, "python_out_sig_sub_name"):
                    for key in self.python_out_sig_sub_name.keys():
                        outputs[key][0] = {
                            a: [b]
                            for a, b in zip(
                                self.python_out_sig_sub_name[key],
                                outputs[key][0],
                            )
                        }
                fetch_list = getattr(self, "fetch_list", [])

                # cast outputs
                if self.dtype == np.uint16:
                    cast_inputs = []
                    for output_name in output_names:
                        cast_input = self._find_var_in_pir(outputs, output_name)
                        cast_inputs = cast_inputs + cast_input
                    cast_outputs = []
                    for cast_input in cast_inputs:
                        if isinstance(
                            cast_input, paddle.base.libpaddle.pir.Value
                        ):
                            cast_outputs.append(
                                paddle.cast(
                                    cast_input,
                                    paddle.base.core.DataType.FLOAT32,
                                )
                            )
                        else:
                            raise TypeError(
                                f"Unsupported test data type {type(cast_input)}."
                            )

                    outputs = {}
                    for i in range(len(output_names)):
                        outputs.update({output_names[i]: [cast_outputs[i]]})

                outputs_valid = {}
                for output_name in output_names:
                    outputs_valid[output_name] = self._find_var_in_pir(
                        outputs, output_name
                    )
                loss_inputs = []
                for input_name in inputs_to_check:
                    loss_inputs.append(inputs_dict[input_name])

                if user_defined_grad_outputs is None:
                    if len(outputs_valid) == 1:
                        for outputs_valid_key in outputs_valid:
                            loss = paddle.mean(
                                outputs_valid[outputs_valid_key][0]
                            )
                    else:
                        avg_sum = []
                        for cur_loss in outputs_valid:
                            cur_avg_loss = paddle.mean(
                                outputs_valid[cur_loss][0]
                            )
                            avg_sum.append(cur_avg_loss)
                        loss_sum = paddle.add_n(avg_sum)
                        loss = paddle.scale(
                            loss_sum, scale=1.0 / float(len(avg_sum))
                        )

                    grad_inputs = ir_grad(
                        outputs=paddle.utils.flatten(loss),
                        inputs=paddle.utils.flatten(loss_inputs),
                        grad_outputs=None,
                    )
                else:
                    grad_inputs = ir_grad(
                        outputs=paddle.utils.flatten(outputs),
                        inputs=paddle.utils.flatten(static_inputs),
                        grad_outputs=grad_outputs,
                    )
                fetch_list = list(grad_inputs)
                # executor run
                executor = paddle.static.Executor(place)
                outs = executor.run(
                    ir_program,
                    feed=feed,
                    fetch_list=fetch_list,
                )
                return outs


class OpTestTool:
    @classmethod
    def skip_if(cls, condition: object, reason: str):
        return unittest.skipIf(condition, reason)

    @classmethod
    def skip_if_not_cpu_bf16(cls):
        return OpTestTool.skip_if(
            not (
                isinstance(_current_expected_place(), core.CPUPlace)
                and core.supports_bfloat16()
            ),
            "Place does not support BF16 evaluation",
        )

    @classmethod
    def skip_if_not_cpu(cls):
        return OpTestTool.skip_if(
            not isinstance(_current_expected_place(), core.CPUPlace),
            "OneDNN supports only CPU for now",
        )
