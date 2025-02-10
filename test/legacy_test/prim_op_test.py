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

import os
import struct
from collections import defaultdict
from functools import partial

import numpy as np

import paddle
from legacy_test import config
from legacy_test.utils import dygraph_guard, static_guard
from paddle.autograd.backward_utils import ValueSet
from paddle.autograd.ir_backward import grad as ir_grad
from paddle.base import Scope, core
from paddle.base.executor import scope_guard
from paddle.base.framework import (
    OpProtoHolder,
    _dygraph_tracer,
    canonicalize_attrs,
    in_dygraph_mode,
    in_pir_mode,
    paddle_type_to_proto_type,
    use_pir_api,
)
from paddle.decomposition import decompose
from paddle.incubate.autograd import primapi
from paddle.jit.dy2static.utils import parse_arg_and_kwargs
from paddle.pir.core import vartype_to_datatype


def flatten(nest_list):
    out = []
    for i in nest_list:
        if isinstance(i, (list, tuple)):
            tmp_list = flatten(i)
            for j in tmp_list:
                out.append(j)
        else:
            out.append(i)
    return out


def _as_list(x):
    if x is None:
        return []
    return list(x) if isinstance(x, (list, tuple)) else [x]


def convert_uint16_to_float(in_list):
    in_list = np.asarray(in_list)
    out = np.vectorize(
        lambda x: struct.unpack(
            '<f', struct.pack('<I', np.uint32(x) << np.uint32(16))
        )[0],
        otypes=[np.float32],
    )(in_list.flat)
    return np.reshape(out, in_list.shape)


def patch_for_one_hot(inputs, attrs, args):
    if 'depth_tensor' in inputs.keys():
        args[1] = inputs['depth_tensor'].item()
    else:
        args[1] = attrs['depth']
    return args


# TODO(wanghao107): OpTestUtils will be moved to op_test.py
class OpTestUtils:
    @classmethod
    def _get_kernel_signature(
        cls, op_type, eager_tensor_inputs, eager_tensor_outputs, attrs_outputs
    ):
        try:
            op_proto = OpProtoHolder.instance().get_op_proto(op_type)
            canonicalized_attrs = canonicalize_attrs(attrs_outputs, op_proto)
        except ValueError:
            canonicalized_attrs = attrs_outputs
        try:
            kernel_sig = _dygraph_tracer()._get_kernel_signature(
                op_type,
                eager_tensor_inputs,
                eager_tensor_outputs,
                canonicalized_attrs,
            )
        except RuntimeError as re:
            """we think the kernel_sig is missing."""
            kernel_sig = None
            print(
                f"[Warning: op_test.py] Kernel Signature is not found for {op_type}, fall back to intermediate state."
            )
        return kernel_sig

    @classmethod
    def prepare_python_api_arguments(
        cls, api, op_proto_ins, op_proto_attrs, kernel_sig, target_dtype=None
    ):
        """map from `op proto inputs and attrs` to `api input list and api attrs dict`

        NOTE: the op_proto_attrs and op_proto_ins is a default dict. default value is []
        """

        class Empty:
            pass

        def is_empty(a):
            return isinstance(a, Empty)

        def get_default(idx, defaults):
            assert not isinstance(
                defaults[idx], Empty
            ), f"{idx}-th params of python api don't have default value."
            return defaults[idx]

        def to_defaults_list(params, defaults):
            return [defaults[p] for p in params if p in defaults]

        def parse_attri_value(name, op_inputs, op_proto_attrs):
            """parse true value from inputs and attrs, if there is no name passed by OpTest, return Empty
            1. if the name in op_attrs, use the op_attrs[name]
            2. if the name in op_inputs, convert the op_inputs to [type of default value]
            3. if the name not in op_attrs ans op_inputs, return Empty. (this will use the default value from python api)
            """
            if name in op_proto_attrs:
                return op_proto_attrs[name]
            elif name in op_inputs:
                if len(op_inputs[name]) == 1:
                    # why don't use numpy().item() : if the Tensor is float64, we will change it to python.float32, where we loss accuracy: [allclose_op]
                    # why we reconstruct a tensor: because we want the tensor in cpu.
                    if in_dygraph_mode():
                        return paddle.to_tensor(
                            op_inputs[name][0].numpy(), place='cpu'
                        )
                    else:
                        return op_inputs[name][0]
                else:
                    # if this is a list (test_unsqueeze2_op): we just pass it into the python api.
                    return op_inputs[name]
            else:
                return Empty()

        def convert_dtype(dtype, target_dtype):
            if target_dtype is None:
                return dtype
            if (
                isinstance(dtype, core.VarDesc.VarType)
                and target_dtype is paddle.pir.core.DataType
            ):
                return vartype_to_datatype[dtype]
            if (
                isinstance(dtype, paddle.pir.core.DataType)
                and target_dtype is core.VarDesc.VarType
            ):
                return paddle_type_to_proto_type[dtype]
            return dtype

        # NOTE(xiongkun): the logic of constructing parameters:
        # for example:
        #    python api: cumprod(x, dim, dtype=None, name=None)
        #    kernel sig: [["x"], ["dim"], ["out"]]"
        #
        # we will construct a lot of list with the same length : len == len(api_params), here is 4
        #    api_params = ["x", "dim", "dtype", "name"]
        #    api_defaults = [Empty, Empty, None, None]; empty means no defaults.
        #    inputs_and_attrs = ["x", "dim"] , the length may shorter or longer than api_params
        #    input_arguments = [RealValue in self.inputs and self.attrs]
        # then ,we will loop for the api_params, construct a result list:
        #    if the name in ['name', 'dtype', 'out', 'output'], we will use the default value
        #    else, we will consume a input_arguments. (because the name is not corresponding, so we only use the order)

        api_params, api_defaults = parse_arg_and_kwargs(api)
        api_defaults = to_defaults_list(api_params, api_defaults)
        api_defaults = [
            Empty() for i in range(len(api_params) - len(api_defaults))
        ] + api_defaults

        # patch for one hot -> fill the api params
        if "one_hot" in str(api):
            api_defaults = [None for x in range(len(api_params))]

        assert len(api_defaults) == len(
            api_params
        ), "Error happens. contack xiongkun03 to solve."
        inputs_sig, attrs_sig, outputs_sig = kernel_sig
        inputs_and_attrs = inputs_sig + attrs_sig
        input_arguments = [
            op_proto_ins.get(name, Empty()) for name in inputs_sig
        ] + [
            parse_attri_value(name, op_proto_ins, op_proto_attrs)
            for name in attrs_sig
        ]
        results = []
        # hack support variable length parameter(such as paddle.meshgrid(*args,**kwargs)
        if api_params == []:
            results.append(input_arguments)
            return results
        api_ignore_param_list = {'name', 'dtype', 'out', 'output'}
        idx_of_op_proto_arguments = 0
        for idx, arg_name in enumerate(api_params):
            if arg_name in api_ignore_param_list:
                to_append = (
                    get_default(idx, api_defaults)
                    if arg_name not in op_proto_attrs
                    else op_proto_attrs[arg_name]
                )
                results.append(to_append)
                if idx_of_op_proto_arguments < len(input_arguments):
                    idx_of_op_proto_arguments += 1
            else:
                if idx_of_op_proto_arguments < len(input_arguments):
                    tmp = input_arguments[idx_of_op_proto_arguments]
                    idx_of_op_proto_arguments += 1
                else:
                    # tmp = Empty()  # use the default value
                    tmp = parse_attri_value(
                        arg_name, op_proto_ins, op_proto_attrs
                    )

                if isinstance(tmp, Empty):
                    results.append(get_default(idx, api_defaults))
                else:
                    results.append(tmp)
        assert len(results) == len(api_params)

        results = paddle.utils.map_structure(
            partial(convert_dtype, target_dtype=target_dtype), results
        )
        return results

    @classmethod
    def assumption_assert_and_transform(cls, args, inp_num):
        """
        transform inputs by the following rules:
            Note: it may not be possible to distinguish list with one Tensor,you should use wrapper to distinguish.
            1. [Tensor] -> Tensor
            2. [Tensor, Tensor, ...] -> list of Tensors
            3. None -> None
            4. Others: raise Error

        only support "X" is list of Tensor, currently don't support other structure like dict.
        """
        inp_args = [
            [inp] if inp is None else inp for inp in args[:inp_num]
        ]  # convert None -> [None]
        for inp in inp_args:
            assert isinstance(
                inp, list
            ), "currently only support `X` is [Tensor], don't support other structure."
        args = [inp[0] if len(inp) == 1 else inp for inp in inp_args] + args[
            inp_num:
        ]
        return args

    @classmethod
    def is_bfloat16_type(cls, np_type):
        if np_type == np.dtype('uint16'):
            return True
        return False


def apply_to_static(net, use_cinn):
    build_strategy = paddle.static.BuildStrategy()
    build_strategy.build_cinn_pass = use_cinn
    return paddle.jit.to_static(
        net, build_strategy=build_strategy, full_graph=True
    )


class PrimNet(paddle.nn.Layer):
    def __init__(self, public_python_api):
        super().__init__()
        self.public_python_api = public_python_api

    def forward(self, args):
        out = self.public_python_api(*args)
        return out


class PrimForwardChecker:
    def __init__(self, op_test, place):
        self.checker_name = "PrimForwardChecker"
        self.place = place
        self.op_test = op_test
        self.init()
        self.init_checker()

    def init(self):
        pass

    def init_checker(self):
        assert hasattr(
            self.op_test, 'prim_op_type'
        ), "if you want to test comp op, please set prim_op_type with 'prim' or 'comp' in setUp function."
        assert self.op_test.prim_op_type in [
            "comp",
            "prim",
        ], "prim_op_type must be comp or prim in setUp function."
        assert hasattr(
            self.op_test, 'dtype'
        ), "Please set dtype in setUp function."
        self.op_type = self.op_test.op_type
        self.prim_op_type = self.op_test.prim_op_type
        assert hasattr(
            self.op_test, 'public_python_api'
        ), "If you want to check prim, please set public_python_api in setUp function."
        self.public_python_api = self.op_test.public_python_api
        self.dtype = np.dtype(self.op_test.dtype)
        self.inputs = self.op_test.inputs
        self.attrs = (
            self.op_test.attrs if hasattr(self.op_test, 'attrs') else {}
        )
        self.outputs = self.op_test.outputs
        self.init_checker_threshold()
        self.enable_fw_comp = (
            self.op_test.enable_fw_comp
            if hasattr(self.op_test, 'enable_fw_comp')
            else True
        )
        self.enable_rev_comp = (
            self.op_test.enable_rev_comp
            if hasattr(self.op_test, 'enable_rev_comp')
            else True
        )
        self.enable_cinn = (
            self.op_test.enable_cinn
            if hasattr(self.op_test, 'enable_cinn')
            else True
        )
        if os.getenv('FLAGS_enable_cinn'):
            self.enable_cinn = True
        self.enable_check_eager_comp = (
            self.op_test.enable_check_eager_comp
            if hasattr(self.op_test, 'enable_check_eager_comp')
            else True
        )
        self.enable_check_static_comp = (
            self.op_test.enable_check_static_comp
            if hasattr(self.op_test, 'enable_check_static_comp')
            else True
        )
        self.enable_check_jit_comp = (
            self.op_test.enable_check_jit_comp
            if hasattr(self.op_test, 'enable_check_jit_comp')
            else True
        )
        self.enable_check_jit_comp_with_cinn = (
            self.op_test.enable_check_jit_comp_with_cinn
            if hasattr(self.op_test, 'enable_check_jit_comp_with_cinn')
            else True
        )
        self.kernel_sig = self.get_kernel_sig()

    def init_checker_threshold(self):
        if hasattr(self.op_test, 'jit_comp_rtol'):
            self.jit_comp_rtol = self.op_test.jit_comp_rtol
        else:
            self.jit_comp_rtol = (
                config.TOLERANCE[self.dtype]['jit_comp']['rtol']
                if self.dtype in config.TOLERANCE
                else 0
            )

        if hasattr(self.op_test, 'jit_comp_atol'):
            self.jit_comp_atol = self.op_test.jit_comp_atol
        else:
            self.jit_comp_atol = (
                config.TOLERANCE[self.dtype]['jit_comp']['atol']
                if self.dtype in config.TOLERANCE
                else 0
            )

        if hasattr(self.op_test, 'fw_comp_rtol'):
            self.fw_comp_rtol = self.op_test.fw_comp_rtol
        else:
            self.fw_comp_rtol = (
                config.TOLERANCE[self.dtype]['fw_comp']['rtol']
                if self.dtype in config.TOLERANCE
                else 0
            )

        if hasattr(self.op_test, 'fw_comp_atol'):
            self.fw_comp_atol = self.op_test.fw_comp_atol
        else:
            self.fw_comp_atol = (
                config.TOLERANCE[self.dtype]['fw_comp']['atol']
                if self.dtype in config.TOLERANCE
                else 0
            )

        if hasattr(self.op_test, 'rev_comp_rtol'):
            self.rev_comp_rtol = self.op_test.rev_comp_rtol
        else:
            self.rev_comp_rtol = (
                config.TOLERANCE[self.dtype]['rev_comp']['rtol']
                if self.dtype in config.TOLERANCE
                else 0
            )

        if hasattr(self.op_test, 'rev_comp_atol'):
            self.rev_comp_atol = self.op_test.rev_comp_atol
        else:
            self.rev_comp_atol = (
                config.TOLERANCE[self.dtype]['rev_comp']['atol']
                if self.dtype in config.TOLERANCE
                else 0
            )

        if hasattr(self.op_test, 'cinn_rtol'):
            self.cinn_rtol = self.op_test.cinn_rtol
        else:
            self.cinn_rtol = (
                config.TOLERANCE[self.dtype]['cinn']['rtol']
                if self.dtype in config.TOLERANCE
                else 0
            )

        if hasattr(self.op_test, 'cinn_atol'):
            self.cinn_atol = self.op_test.cinn_atol
        else:
            self.cinn_atol = (
                config.TOLERANCE[self.dtype]['cinn']['atol']
                if self.dtype in config.TOLERANCE
                else 0
            )

    def check(self):
        if (
            type(self.place) is paddle.base.libpaddle.CUDAPlace
            and not paddle.is_compiled_with_cuda()
        ):
            return
        self.eager_desire = self.get_eager_desire()
        if not in_pir_mode():
            if self.enable_check_static_comp:
                self.check_static_comp()
            if self.enable_check_jit_comp:
                self.check_jit_comp()
            if self.enable_check_jit_comp_with_cinn:
                self.check_jit_comp_with_cinn()
        else:
            if self.enable_check_static_comp:
                with scope_guard(Scope()):
                    self.check_static_comp()
            if self.enable_check_jit_comp:
                with scope_guard(Scope()):
                    self.check_jit_comp()

    def get_kernel_sig(self):
        with dygraph_guard():
            if type(self.place) is paddle.base.libpaddle.CPUPlace:
                paddle.device.set_device("cpu")
            if type(self.place) is paddle.base.libpaddle.CUDAPlace:
                paddle.device.set_device("gpu:0")
            (
                eager_tensor_inputs,
                attrs_outputs,
                _,
            ) = self.get_eager_input_attr_and_inputdict(stop_gradient=True)
            eager_tensor_outputs = self.get_eager_empty_output(
                stop_gradient=True
            )
            kernel_sig = OpTestUtils._get_kernel_signature(
                self.op_type,
                eager_tensor_inputs,
                eager_tensor_outputs,
                attrs_outputs,
            )
        return kernel_sig

    def get_eager_desire(self):
        with dygraph_guard():
            if type(self.place) is paddle.base.libpaddle.CPUPlace:
                paddle.device.set_device("cpu")
            if type(self.place) is paddle.base.libpaddle.CUDAPlace:
                paddle.device.set_device("gpu:0")
            (
                eager_tensor_inputs,
                attrs_outputs,
                _,
            ) = self.get_eager_input_attr_and_inputdict(stop_gradient=True)
            args = OpTestUtils.prepare_python_api_arguments(
                self.public_python_api,
                eager_tensor_inputs,
                attrs_outputs,
                self.kernel_sig,
                target_dtype=paddle.core.VarDesc.VarType,
            )
            if "one_hot" in self.op_type:
                args = patch_for_one_hot(self.inputs, self.attrs, args)
            inputs_sig, _, _ = self.kernel_sig
            args = OpTestUtils.assumption_assert_and_transform(
                args, len(inputs_sig)
            )
            ret = flatten(_as_list(self.public_python_api(*args)))
            ret = paddle.utils.map_structure(lambda x: x.numpy(), ret)
            if OpTestUtils.is_bfloat16_type(self.dtype):
                ret = paddle.utils.map_structure(
                    lambda x: convert_uint16_to_float(x), ret
                )
        return ret

    def get_eager_input_attr_and_inputdict(self, stop_gradient):
        attrs_outputs = {}
        for attrs_name in self.attrs:
            if self.attrs[attrs_name] is not None:
                attrs_outputs[attrs_name] = self.attrs[attrs_name]
        input_dict = {}
        eager_inputs = defaultdict(list)
        for name, item in self.inputs.items():
            if isinstance(item, list):
                for tup in item:
                    dtype = (
                        "bfloat16"
                        if OpTestUtils.is_bfloat16_type(tup[1].dtype)
                        else tup[1].dtype
                    )
                    x = paddle.to_tensor(
                        data=tup[1],
                        place=self.place,
                        stop_gradient=stop_gradient,
                        dtype=dtype,
                    )
                    eager_inputs[name].append(x)
                    input_dict.update({str(tup[0]): x})
            else:
                dtype = (
                    "bfloat16"
                    if OpTestUtils.is_bfloat16_type(item.dtype)
                    else item.dtype
                )
                x = paddle.to_tensor(
                    data=item,
                    place=self.place,
                    stop_gradient=stop_gradient,
                    dtype=dtype,
                )
                eager_inputs[name].append(x)
                input_dict.update({name: x})
        return eager_inputs, attrs_outputs, input_dict

    def get_eager_empty_output(self, stop_gradient):
        eager_outputs = defaultdict(list)
        for name, item in self.outputs.items():
            if isinstance(item, list):
                for tup in item:
                    dtype = (
                        "bfloat16"
                        if OpTestUtils.is_bfloat16_type(tup[1].dtype)
                        else tup[1].dtype
                    )
                    x = paddle.to_tensor(
                        data=[],
                        place=self.place,
                        stop_gradient=stop_gradient,
                        dtype=dtype,
                    )
                    eager_outputs[name].append(x)
            else:
                dtype = (
                    "bfloat16"
                    if OpTestUtils.is_bfloat16_type(item.dtype)
                    else item.dtype
                )
                x = paddle.to_tensor(
                    data=[],
                    place=self.place,
                    stop_gradient=stop_gradient,
                    dtype=dtype,
                )
                eager_outputs[name].append(x)
        return eager_outputs

    def get_static_input_attr_inputdict_and_feed(self, stop_gradient):
        attrs_outputs = {}
        for attrs_name in self.attrs:
            if self.attrs[attrs_name] is not None:
                attrs_outputs[attrs_name] = self.attrs[attrs_name]
        input_dict = {}
        static_inputs = defaultdict(list)
        feed = {}
        for name, item in self.inputs.items():
            if isinstance(item, list):
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

    def check_eager_comp(self):
        pass

    def check_static_comp(self):
        # forward comp only for comp op
        if self.prim_op_type == "prim":
            return
        with static_guard():
            core._set_prim_forward_enabled(self.enable_fw_comp)
            startup_program, main_program = (
                paddle.static.Program(),
                paddle.static.Program(),
            )
            with paddle.static.program_guard(main_program, startup_program):
                (
                    static_inputs,
                    attrs,
                    input_dict,
                    feed,
                ) = self.get_static_input_attr_inputdict_and_feed(
                    stop_gradient=True
                )
                args = OpTestUtils.prepare_python_api_arguments(
                    self.public_python_api,
                    static_inputs,
                    attrs,
                    self.kernel_sig,
                    target_dtype=(
                        paddle.pir.core.DataType
                        if in_pir_mode()
                        else paddle.core.VarDesc.VarType
                    ),
                )
                if "one_hot" in self.op_type:
                    args = patch_for_one_hot(self.inputs, self.attrs, args)
                inputs_sig, _, _ = self.kernel_sig
                args = OpTestUtils.assumption_assert_and_transform(
                    args, len(inputs_sig)
                )
                ret = flatten(_as_list(self.public_python_api(*args)))

                if not in_pir_mode():
                    primapi.to_prim(main_program.blocks)
                else:
                    before_ops = [
                        op.name() for op in main_program.global_block().ops
                    ]
                    ret = decompose(main_program, ret)
                    after_ops = [
                        op.name() for op in main_program.global_block().ops
                    ]

                    assert (
                        before_ops != after_ops
                    ), f"For {after_ops} , since op which has been decomposed should not exist, the op list should differ from origin ones."

                # ensure the operator not in program if check_prim is True
                if not in_pir_mode():
                    forward_ops = [op.type for op in main_program.blocks[0].ops]
                    assert (
                        self.op_type not in forward_ops
                    ), f"{self.op_type} shouldn't appear in program when check_prim is True"
                exe = paddle.static.Executor(self.place)
                exe.run(startup_program)
                ret = exe.run(main_program, feed=feed, fetch_list=ret)
                if OpTestUtils.is_bfloat16_type(self.dtype):
                    ret = paddle.utils.map_structure(
                        lambda x: convert_uint16_to_float(x), ret
                    )
        # check static forward
        if len(ret) != len(self.eager_desire):
            msg = (
                f"The static comp forward api out tensor nums is different with eager forward api out tensor nums on {self.place}."
                f'when enable_fw_comp is {self.enable_fw_comp}, static comp forward api out tensor nums = {len(ret)}, eager forward api out tensor nums = {len(self.eager_desire)}. \n'
            )
            raise RuntimeError(msg)
        for i in range(len(ret)):
            np.testing.assert_allclose(
                ret[i],
                self.eager_desire[i],
                rtol=self.fw_comp_rtol,
                atol=self.fw_comp_atol,
                err_msg=(
                    'Check static comp forward api out failed. Mismatch between static comp '
                    f'and eager on {self.place!s}, when enable_fw_comp is {self.enable_fw_comp},'
                    f'the forward api out tensor\'s index is : {i} \n'
                    f'static comp forward api out tensor:\n{ret[i]}\n '
                    f'eager forward api out tensor:\n{self.eager_desire[i]}\n'
                ),
            )
        with dygraph_guard():
            core._set_prim_forward_enabled(False)

    def check_jit_comp(self):
        if self.prim_op_type == "prim":
            return
        with dygraph_guard():
            if type(self.place) == paddle.base.libpaddle.CPUPlace:
                paddle.device.set_device("cpu")
            if type(self.place) == paddle.base.libpaddle.CUDAPlace:
                paddle.device.set_device("gpu:0")
            atol = (
                self.fw_comp_atol if self.enable_fw_comp else self.jit_comp_atol
            )
            rtol = (
                self.fw_comp_rtol if self.enable_fw_comp else self.jit_comp_rtol
            )
            core._set_prim_forward_enabled(self.enable_fw_comp)
            (
                eager_tensor_inputs,
                attrs_outputs,
                _,
            ) = self.get_eager_input_attr_and_inputdict(stop_gradient=True)
            args = OpTestUtils.prepare_python_api_arguments(
                self.public_python_api,
                eager_tensor_inputs,
                attrs_outputs,
                self.kernel_sig,
                target_dtype=(
                    paddle.pir.core.DataType
                    if use_pir_api()
                    else paddle.core.VarDesc.VarType
                ),
            )
            if "one_hot" in self.op_type:
                args = patch_for_one_hot(self.inputs, self.attrs, args)
            inputs_sig, _, _ = self.kernel_sig
            args = OpTestUtils.assumption_assert_and_transform(
                args, len(inputs_sig)
            )
            net = PrimNet(self.public_python_api)
            net = apply_to_static(net, False)
            # ensure the operator not in program if check_prim is True
            if not use_pir_api():
                forward_ops = [
                    op.type
                    for op in net.forward.get_concrete_program(args)[1]
                    .forward_program.block(0)
                    .ops
                ]
                assert (
                    self.op_type not in forward_ops
                ), f"{self.op_type} shouldn't appear in program when check_prim is True"
            ret = flatten(_as_list(net(args)))
            ret = paddle.utils.map_structure(lambda x: x.numpy(), ret)
            if OpTestUtils.is_bfloat16_type(self.dtype):
                ret = paddle.utils.map_structure(
                    lambda x: convert_uint16_to_float(x), ret
                )
            # check jit comp forward
            if len(ret) != len(self.eager_desire):
                msg = (
                    f"The jit comp forward api out tensor nums is different with eager forward api out tensor nums on {self.place}."
                    f'when enable_fw_comp is {self.enable_fw_comp}, jit comp forward api out tensor nums = {len(ret)}, eager forward api out tensor nums = {len(self.eager_desire)}. \n'
                )
                raise RuntimeError(msg)
            for i in range(len(ret)):
                np.testing.assert_allclose(
                    ret[i],
                    self.eager_desire[i],
                    rtol=rtol,
                    atol=atol,
                    err_msg=(
                        'Check jit comp forward api out failed. Mismatch between jit comp '
                        f'and eager on {self.place!s}, when enable_fw_comp is {self.enable_fw_comp},'
                        f'the forward api out tensor\'s index is : {i} \n'
                        f'jit comp forward api out tensor:\n{ret[i]}\n '
                        f'eager forward api out tensor:\n{self.eager_desire[i]}\n'
                    ),
                )
            core._set_prim_forward_enabled(False)
            net.forward.program_cache.clear()

    def check_jit_comp_with_cinn(self):
        if self.prim_op_type == "prim":
            return
        # cinn doesn't support cpu place
        if (
            type(self.place) == paddle.base.libpaddle.CPUPlace
            and self.enable_cinn
            and core.is_compiled_with_cinn()
        ):
            return
        with dygraph_guard():
            atol = (
                self.cinn_atol
                if self.enable_cinn and core.is_compiled_with_cinn()
                else self.fw_comp_atol
            )
            rtol = (
                self.cinn_rtol
                if self.enable_cinn and core.is_compiled_with_cinn()
                else self.fw_comp_rtol
            )
            core._set_prim_forward_enabled(self.enable_fw_comp)
            if type(self.place) is paddle.base.libpaddle.CPUPlace:
                paddle.device.set_device("cpu")
            if type(self.place) is paddle.base.libpaddle.CUDAPlace:
                paddle.device.set_device("gpu:0")
            (
                eager_tensor_inputs,
                attrs_outputs,
                _,
            ) = self.get_eager_input_attr_and_inputdict(stop_gradient=True)
            args = OpTestUtils.prepare_python_api_arguments(
                self.public_python_api,
                eager_tensor_inputs,
                attrs_outputs,
                self.kernel_sig,
                target_dtype=(
                    paddle.pir.core.DataType
                    if use_pir_api()
                    else paddle.core.VarDesc.VarType
                ),
            )
            inputs_sig, _, _ = self.kernel_sig
            args = OpTestUtils.assumption_assert_and_transform(
                args, len(inputs_sig)
            )
            net = PrimNet(self.public_python_api)
            net = apply_to_static(
                net, core.is_compiled_with_cinn() and self.enable_cinn
            )
            # check the operator not in program if check prim is True
            forward_ops = [
                op.type
                for op in net.forward.get_concrete_program(args)[1]
                .forward_program.block(0)
                .ops
            ]
            assert (
                self.op_type not in forward_ops
            ), f"{self.op_type} shouldn't appear in program when check_prim is True"
            ret = flatten(_as_list(net(args)))
            ret = paddle.utils.map_structure(lambda x: x.numpy(), ret)
            if OpTestUtils.is_bfloat16_type(self.dtype):
                ret = paddle.utils.map_structure(
                    lambda x: convert_uint16_to_float(x), ret
                )
            # check jit comp forward
            if len(ret) != len(self.eager_desire):
                msg = (
                    f"The jit comp with cinn forward api out tensor nums is different with eager forward api out tensor nums on {self.place}."
                    f'when enable_fw_comp is {self.enable_fw_comp}, enable_cinn is {core.is_compiled_with_cinn() and self.enable_cinn}, jit comp forward api out tensor nums = {len(ret)}, eager forward api out tensor nums = {len(self.eager_desire)}. \n'
                )
                raise RuntimeError(msg)
            for i in range(len(ret)):
                np.testing.assert_allclose(
                    ret[i],
                    self.eager_desire[i],
                    rtol=rtol,
                    atol=atol,
                    err_msg=(
                        f'Check jit comp with cinn forward api out failed. Mismatch between jit comp and eager on {self.place!s}, '
                        f'when enable_fw_comp is {self.enable_fw_comp}, '
                        f'enable_cinn is {core.is_compiled_with_cinn() and self.enable_cinn}, '
                        f'the forward api out tensor\'s index is : {i} \n'
                        f'jit comp forward api out tensor:\n{ret[i]}\n '
                        f'eager forward api out tensor:\n{self.eager_desire[i]}\n'
                    ),
                )
            core._set_prim_forward_enabled(False)
            net.forward.program_cache.clear()


class PrimGradChecker(PrimForwardChecker):
    def __init__(
        self,
        op_test,
        place,
        inputs_to_check,
        output_names,
        no_grad_set,
        grad_outputs,
    ):
        PrimForwardChecker.__init__(self, op_test, place)
        self.inputs_to_check = inputs_to_check
        self.output_names = output_names
        self.no_grad_set = no_grad_set
        self.grad_outputs = grad_outputs

    def init(self):
        self.checker_name = "PrimGradChecker"

    def check(self):
        if (
            type(self.place) is paddle.base.libpaddle.CUDAPlace
            and not paddle.is_compiled_with_cuda()
        ):
            return
        self.eager_desire = self.get_eager_desire()
        if not in_pir_mode():
            if self.enable_check_eager_comp:
                self.check_eager_comp()
            if self.enable_check_static_comp:
                self.check_static_comp()
            if self.enable_check_jit_comp:
                self.check_jit_comp()
            if self.enable_check_jit_comp_with_cinn:
                self.check_jit_comp_with_cinn()
        else:
            if self.enable_check_static_comp:
                with scope_guard(Scope()):
                    self.check_static_comp()
            if self.enable_check_jit_comp:
                with scope_guard(Scope()):
                    self.check_jit_comp()

    def get_output_dict(self, np_outputs, api_outputs, outputs_sig):
        assert len(api_outputs) <= len(
            outputs_sig
        ), f"forward api outputs length must be the less than or equal to KernelSignature outputs,but receive {len(api_outputs)} and {len(outputs_sig)}"
        output_dict = {}
        for i in range(len(api_outputs)):
            output_name = outputs_sig[i]
            if output_name in np_outputs and isinstance(
                np_outputs[output_name], list
            ):
                for j, tup in enumerate(np_outputs[output_name]):
                    output_dict.update({tup[0]: api_outputs[i][j]})
            else:
                output_dict.update({output_name: api_outputs[i]})
        return output_dict

    def gen_eager_grad_outputs(self):
        if self.grad_outputs is None:
            return None
        eager_vs = []
        for np_v in self.grad_outputs:
            eager_vs.append(
                paddle.to_tensor(
                    data=np_v,
                    place=self.place,
                    dtype=(
                        "bfloat16"
                        if OpTestUtils.is_bfloat16_type(np_v.dtype)
                        else np_v.dtype
                    ),
                )
            )
        return eager_vs

    def gen_static_grad_outputs_and_feed(self):
        if self.grad_outputs is None:
            return None, {}
        static_vs = []
        feed = {}
        for i, np_v in enumerate(self.grad_outputs):
            static_vs.append(
                paddle.static.data(
                    name='v_' + str(i),
                    shape=np_v.shape,
                    dtype=(
                        "bfloat16"
                        if OpTestUtils.is_bfloat16_type(np_v.dtype)
                        else np_v.dtype
                    ),
                )
            )
            feed.update({'v_' + str(i): np_v})
        return static_vs, feed

    def gen_no_grad_set(self, var_dict):
        if self.no_grad_set is None:
            return None
        no_grad_set = ValueSet() if in_pir_mode() else set()
        for name in self.no_grad_set:
            if name in var_dict:
                no_grad_set.add(var_dict[name])
        return no_grad_set

    def get_eager_desire(self):
        with dygraph_guard():
            if type(self.place) is paddle.base.libpaddle.CPUPlace:
                paddle.device.set_device("cpu")
            if type(self.place) is paddle.base.libpaddle.CUDAPlace:
                paddle.device.set_device("gpu:0")
            (
                eager_tensor_inputs,
                attrs_outputs,
                inputs_dict,
            ) = self.get_eager_input_attr_and_inputdict(stop_gradient=False)
            args = OpTestUtils.prepare_python_api_arguments(
                self.public_python_api,
                eager_tensor_inputs,
                attrs_outputs,
                self.kernel_sig,
                target_dtype=paddle.core.VarDesc.VarType,
            )
            inputs_sig, _, outputs_sig = self.kernel_sig
            if hasattr(self.op_test, "python_out_sig"):
                outputs_sig = self.op_test.python_out_sig
            args = OpTestUtils.assumption_assert_and_transform(
                args, len(inputs_sig)
            )
            ret = _as_list(self.public_python_api(*args))
            outputs_dict = self.get_output_dict(self.outputs, ret, outputs_sig)
            ys = []
            if isinstance(self.output_names, list):
                for output_name in self.output_names:
                    ys.append(outputs_dict[output_name])
            else:
                ys.append(outputs_dict[self.output_names])
            xs = []
            if isinstance(self.inputs_to_check, list):
                for input_name in self.inputs_to_check:
                    xs.append(inputs_dict[input_name])
            else:
                xs.append(inputs_dict[self.inputs_to_check])
            vs = self.gen_eager_grad_outputs()
            no_grad_vars = self.gen_no_grad_set(
                var_dict={**inputs_dict, **outputs_dict}
            )
            ret = paddle.grad(
                ys, xs, vs, allow_unused=True, no_grad_vars=no_grad_vars
            )
            ret = paddle.utils.map_structure(lambda x: x.numpy(), ret)
            if OpTestUtils.is_bfloat16_type(self.dtype):
                ret = paddle.utils.map_structure(
                    lambda x: convert_uint16_to_float(x), ret
                )
        return ret

    def check_eager_comp(self):
        if self.prim_op_type == "comp":
            return
        with dygraph_guard():
            if type(self.place) is paddle.base.libpaddle.CPUPlace:
                paddle.device.set_device("cpu")
            if type(self.place) is paddle.base.libpaddle.CUDAPlace:
                paddle.device.set_device("gpu:0")
            atol = self.rev_comp_atol
            rtol = self.rev_comp_rtol
            core.set_prim_eager_enabled(self.enable_rev_comp)
            actual_ret = self.get_eager_desire()
            # check static forward
            if len(actual_ret) != len(self.eager_desire):
                msg = (
                    f"The eager comp grad out tensor nums is different with eager grad out tensor nums on {self.place}."
                    f'when enable_rev_comp is {self.enable_rev_comp}, eager comp grad api out tensor nums = {len(actual_ret)}, eager grad out tensor nums = {len(self.eager_desire)}. \n'
                )
                raise RuntimeError(msg)
            for i in range(len(actual_ret)):
                np.testing.assert_allclose(
                    actual_ret[i],
                    self.eager_desire[i],
                    rtol=atol,
                    atol=rtol,
                    err_msg=(
                        'Check eager comp grad out failed. Mismatch between eager comp '
                        f'and eager on {self.place!s}, when enable_rev_comp is {self.enable_rev_comp},'
                        f'the eager comp grad out tensor\'s index is : {i} \n'
                        f'eager comp grad out tensor:\n{actual_ret[i]}\n eager grad out tensor:\n{self.eager_desire[i]}\n'
                    ),
                )
            core.set_prim_eager_enabled(False)

    def check_static_comp(self):
        if self.prim_op_type == "prim":
            core._set_prim_backward_enabled(self.enable_rev_comp)
        else:
            core._set_prim_forward_enabled(self.enable_fw_comp)
            core._set_prim_backward_enabled(self.enable_rev_comp)
        atol = self.rev_comp_atol if self.enable_rev_comp else self.fw_comp_atol
        rtol = self.rev_comp_rtol if self.enable_rev_comp else self.fw_comp_rtol
        with static_guard():
            startup_program, main_program = (
                paddle.static.Program(),
                paddle.static.Program(),
            )
            with paddle.static.program_guard(main_program, startup_program):
                (
                    static_inputs,
                    attrs,
                    inputs_dict,
                    feed,
                ) = self.get_static_input_attr_inputdict_and_feed(
                    stop_gradient=False
                )
                args = OpTestUtils.prepare_python_api_arguments(
                    self.public_python_api,
                    static_inputs,
                    attrs,
                    self.kernel_sig,
                    target_dtype=(
                        paddle.pir.core.DataType
                        if in_pir_mode()
                        else paddle.core.VarDesc.VarType
                    ),
                )
                inputs_sig, _, outputs_sig = self.kernel_sig
                if hasattr(self.op_test, "python_out_sig"):
                    outputs_sig = self.op_test.python_out_sig
                args = OpTestUtils.assumption_assert_and_transform(
                    args, len(inputs_sig)
                )
                fw_outs = _as_list(self.public_python_api(*args))
                if not in_pir_mode():
                    primapi.to_prim(main_program.blocks)
                else:
                    fw_outs = decompose(main_program, fw_outs)
                outputs_dict = self.get_output_dict(
                    self.outputs, fw_outs, outputs_sig
                )
                ys = []
                if isinstance(self.output_names, list):
                    for output_name in self.output_names:
                        ys.append(outputs_dict[output_name])
                else:
                    ys.append(outputs_dict[self.output_names])
                xs = []
                if isinstance(self.inputs_to_check, list):
                    for input_name in self.inputs_to_check:
                        xs.append(inputs_dict[input_name])
                else:
                    xs.append(inputs_dict[self.inputs_to_check])
                vs, vs_feed = self.gen_static_grad_outputs_and_feed()
                feed.update(vs_feed)
                no_grad_vars = self.gen_no_grad_set(
                    var_dict={**inputs_dict, **outputs_dict}
                )
                if not in_pir_mode():
                    ret = paddle.static.gradients(
                        ys, xs, vs, no_grad_set=no_grad_vars
                    )
                else:
                    ret = ir_grad(ys, xs, vs, no_grad_vars=no_grad_vars)
                # check the backward operator not in program when check_prim is True
                if not in_pir_mode():
                    ops = [op.type for op in main_program.blocks[0].ops]
                    backward_op_type = self.op_type + "_grad"
                    assert (
                        backward_op_type not in ops
                    ), f"{backward_op_type} shouldn't appear in program when check_prim is True"
                elif self.prim_op_type == "prim":
                    grad_ops = []
                    for op in main_program.global_block().ops:
                        if op.name().endswith("_grad"):
                            grad_ops.append(op.name())
                    assert (
                        not grad_ops
                    ), f"For {grad_ops} , grad op shouldn't appear in program when check_prim is True"
                exe = paddle.static.Executor(self.place)
                exe.run(startup_program)
                actual_ret = exe.run(main_program, feed=feed, fetch_list=ret)
                if OpTestUtils.is_bfloat16_type(self.dtype):
                    actual_ret = paddle.utils.map_structure(
                        lambda x: convert_uint16_to_float(x), actual_ret
                    )
        # check static grad out
        if len(actual_ret) != len(self.eager_desire):
            msg = (
                f"The static comp grad out tensor nums is different with eager grad out tensor nums on {self.place}."
                f'when enable_fw_comp is {self.enable_fw_comp},enable_rev_comp is {self.enable_rev_comp}, static comp grad out tensor nums = {len(actual_ret)}, eager grad out tensor nums = {len(self.eager_desire)}. \n'
            )
            raise RuntimeError(msg)
        for i in range(len(actual_ret)):
            np.testing.assert_allclose(
                actual_ret[i],
                self.eager_desire[i],
                rtol=rtol,
                atol=atol,
                err_msg=(
                    'Check static comp grad out failed. Mismatch between static comp '
                    f'and eager on {self.place}, when enable_fw_comp is {self.enable_fw_comp},enable_rev_comp is { self.enable_rev_comp},'
                    f'the forward api out tensor\'s index is : {i} \n'
                    f'static comp grad out tensor:\n{actual_ret[i]}\n eager grad out tensor:\n{self.eager_desire[i]}\n'
                ),
            )
        core._set_prim_forward_enabled(False)
        core._set_prim_backward_enabled(False)

    def check_jit_comp(self):
        with dygraph_guard():
            if type(self.place) is paddle.base.libpaddle.CPUPlace:
                paddle.device.set_device("cpu")
            if type(self.place) is paddle.base.libpaddle.CUDAPlace:
                paddle.device.set_device("gpu:0")
            if self.prim_op_type == "prim":
                core._set_prim_backward_enabled(self.enable_rev_comp)
            else:
                core._set_prim_forward_enabled(self.enable_fw_comp)
                core._set_prim_backward_enabled(self.enable_rev_comp)
            atol = (
                self.fw_comp_atol
                if self.enable_fw_comp and not self.enable_rev_comp
                else self.jit_comp_atol
            )
            rtol = (
                self.fw_comp_rtol
                if self.enable_fw_comp and not self.enable_rev_comp
                else self.jit_comp_rtol
            )
            atol = self.rev_comp_atol if self.enable_rev_comp else atol
            rtol = self.rev_comp_rtol if self.enable_rev_comp else rtol
            (
                eager_tensor_inputs,
                attrs_outputs,
                inputs_dict,
            ) = self.get_eager_input_attr_and_inputdict(stop_gradient=False)
            args = OpTestUtils.prepare_python_api_arguments(
                self.public_python_api,
                eager_tensor_inputs,
                attrs_outputs,
                self.kernel_sig,
                target_dtype=(
                    paddle.pir.core.DataType
                    if use_pir_api()
                    else paddle.core.VarDesc.VarType
                ),
            )
            inputs_sig, _, outputs_sig = self.kernel_sig
            args = OpTestUtils.assumption_assert_and_transform(
                args, len(inputs_sig)
            )
            net = PrimNet(self.public_python_api)
            net = apply_to_static(net, False)
            # check the backward operator not in program when check_prim is True

            if not use_pir_api():
                ops = [
                    op.type
                    for op in net.forward.get_concrete_program(args)[1]
                    .backward_program.block(0)
                    .ops
                ]
                backward_op_type = self.op_type + "_grad"
                assert (
                    backward_op_type not in ops
                ), f"{backward_op_type} shouldn't appear in program when check_prim is True"
            out = _as_list(net(args))
            if hasattr(self.op_test, "python_out_sig"):
                outputs_sig = self.op_test.python_out_sig
            outputs_dict = self.get_output_dict(self.outputs, out, outputs_sig)
            ys = []
            if isinstance(self.output_names, list):
                for output_name in self.output_names:
                    ys.append(outputs_dict[output_name])
            else:
                ys.append(outputs_dict[self.output_names])
            xs = []
            if isinstance(self.inputs_to_check, list):
                for input_name in self.inputs_to_check:
                    xs.append(inputs_dict[input_name])
            else:
                xs.append(inputs_dict[self.inputs_to_check])
            vs = self.gen_eager_grad_outputs()
            no_grad_vars = self.gen_no_grad_set(
                var_dict={**inputs_dict, **outputs_dict}
            )
            ret = paddle.grad(
                ys, xs, vs, allow_unused=True, no_grad_vars=no_grad_vars
            )
            ret = paddle.utils.map_structure(lambda x: x.numpy(), ret)
            if OpTestUtils.is_bfloat16_type(self.dtype):
                ret = paddle.utils.map_structure(
                    lambda x: convert_uint16_to_float(x), ret
                )
            # check jit comp grad out
            if len(ret) != len(self.eager_desire):
                msg = (
                    f"The jit comp grad out tensor nums is different with eager grad out tensor nums on {self.place}."
                    f'when enable_fw_comp is {self.enable_fw_comp}, enable_rev_comp is {self.enable_rev_comp}, jit comp grad out tensor nums = {len(ret)}, eager grad out tensor nums = {len(self.eager_desire)}. \n'
                )
                raise RuntimeError(msg)
            for i in range(len(ret)):
                np.testing.assert_allclose(
                    ret[i],
                    self.eager_desire[i],
                    rtol=rtol,
                    atol=atol,
                    err_msg=(
                        'Check jit comp grad out failed. Mismatch between jit comp '
                        f'and eager on {self.place!s}, when enable_fw_comp is {self.enable_fw_comp}, '
                        f'enable_rev_comp is {self.enable_rev_comp},the grad out tensor\'s index is : {i} \n'
                        f'jit comp grad out tensor:\n{ret[i]}\n eager grad out out tensor:\n{self.eager_desire[i]}\n'
                    ),
                )
            core._set_prim_forward_enabled(False)
            core._set_prim_backward_enabled(False)
            net.forward.program_cache.clear()

    def check_jit_comp_with_cinn(self):
        # cinn doesn't support cpu place
        if (
            type(self.place) is paddle.base.libpaddle.CPUPlace
            and self.enable_cinn
            and core.is_compiled_with_cinn()
        ):
            return
        with dygraph_guard():
            if type(self.place) is paddle.base.libpaddle.CPUPlace:
                paddle.device.set_device("cpu")
            if type(self.place) is paddle.base.libpaddle.CUDAPlace:
                paddle.device.set_device("gpu:0")
            if self.prim_op_type == "prim":
                core._set_prim_backward_enabled(self.enable_rev_comp)
            else:
                core._set_prim_forward_enabled(self.enable_fw_comp)
                core._set_prim_backward_enabled(self.enable_rev_comp)
            if self.enable_cinn and core.is_compiled_with_cinn():
                atol = self.cinn_atol
                rtol = self.cinn_rtol
            else:
                atol = (
                    self.fw_comp_atol
                    if self.enable_fw_comp and not self.enable_rev_comp
                    else self.jit_comp_atol
                )
                rtol = (
                    self.fw_comp_rtol
                    if self.enable_fw_comp and not self.enable_rev_comp
                    else self.jit_comp_rtol
                )
                atol = self.rev_comp_atol if self.enable_rev_comp else atol
                rtol = self.rev_comp_rtol if self.enable_rev_comp else rtol
            (
                eager_tensor_inputs,
                attrs_outputs,
                inputs_dict,
            ) = self.get_eager_input_attr_and_inputdict(stop_gradient=False)
            args = OpTestUtils.prepare_python_api_arguments(
                self.public_python_api,
                eager_tensor_inputs,
                attrs_outputs,
                self.kernel_sig,
                target_dtype=(
                    paddle.pir.core.DataType
                    if use_pir_api()
                    else paddle.core.VarDesc.VarType
                ),
            )
            inputs_sig, _, outputs_sig = self.kernel_sig
            args = OpTestUtils.assumption_assert_and_transform(
                args, len(inputs_sig)
            )
            net = PrimNet(self.public_python_api)
            net = apply_to_static(
                net, core.is_compiled_with_cinn() and self.enable_cinn
            )
            # check the backward operator not in program when check_prim is True
            ops = [
                op.type
                for op in net.forward.get_concrete_program(args)[1]
                .backward_program.block(0)
                .ops
            ]
            backward_op_type = self.op_type + "_grad"
            assert (
                backward_op_type not in ops
            ), f"{backward_op_type} shouldn't appear in program when check_prim is True"

            out = _as_list(net(args))
            if hasattr(self.op_test, "python_out_sig"):
                outputs_sig = self.op_test.python_out_sig
            outputs_dict = self.get_output_dict(self.outputs, out, outputs_sig)
            ys = []
            if isinstance(self.output_names, list):
                for output_name in self.output_names:
                    ys.append(outputs_dict[output_name])
            else:
                ys.append(outputs_dict[self.output_names])
            xs = []
            if isinstance(self.inputs_to_check, list):
                for input_name in self.inputs_to_check:
                    xs.append(inputs_dict[input_name])
            else:
                xs.append(inputs_dict[self.inputs_to_check])
            vs = self.gen_eager_grad_outputs()
            no_grad_vars = self.gen_no_grad_set(
                var_dict={**inputs_dict, **outputs_dict}
            )
            ret = paddle.grad(
                ys, xs, vs, allow_unused=True, no_grad_vars=no_grad_vars
            )
            ret = paddle.utils.map_structure(lambda x: x.numpy(), ret)
            if OpTestUtils.is_bfloat16_type(self.dtype):
                ret = paddle.utils.map_structure(
                    lambda x: convert_uint16_to_float(x), ret
                )
            # check jit comp grad out
            if len(ret) != len(self.eager_desire):
                msg = (
                    f"The jit comp with cinn grad out tensor nums is different with eager grad out tensor nums on {self.place}."
                    f'when enable_fw_comp is {self.enable_fw_comp}, enable_rev_comp is {self.enable_rev_comp}, enable_cinn is {self.enable_cinn and core.is_compiled_with_cinn()}, jit comp grad out tensor nums = {len(ret)}, eager grad out tensor nums = {len(self.eager_desire)}. \n'
                )
                raise RuntimeError(msg)
            for i in range(len(ret)):
                np.testing.assert_allclose(
                    ret[i],
                    self.eager_desire[i],
                    rtol=rtol,
                    atol=atol,
                    err_msg=(
                        'Check jit comp with cinn grad out failed. Mismatch between jit comp with cinn '
                        f'and eager on {self.place!s}, when enable_fw_comp is {self.enable_fw_comp}, '
                        f'enable_rev_comp is {self.enable_rev_comp}, enable_cinn is {self.enable_cinn and core.is_compiled_with_cinn()},'
                        f'the grad out tensor\'s index is : {i} ,jit comp with cinn grad out tensor:\n{ret[i]}\n eager grad out out tensor:\n{self.eager_desire[i]}\n'
                    ),
                )

            core._set_prim_forward_enabled(False)
            core._set_prim_backward_enabled(False)
            net.forward.program_cache.clear()
