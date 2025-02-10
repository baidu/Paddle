# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np

import paddle
import paddle.pir.core as ir_static
from paddle import _legacy_C_ops
from paddle.autograd.backward_utils import ValueDict
from paddle.autograd.ir_backward import grad
from paddle.base import core, framework
from paddle.base.compiler import BuildStrategy
from paddle.base.data_feeder import check_type
from paddle.base.dygraph.base import switch_to_static_graph
from paddle.pir import Value, fake_value, is_fake_value

from .logging_utils import TranslatorLogger
from .utils import (
    RETURN_NO_VALUE_MAGIC_NUM,
    auto_layout_is_enabled,
    backend_guard,
    cinn_is_enabled,
    cse_is_enabled,
)

if TYPE_CHECKING:
    from .program_translator import ConcreteProgram

__all__ = []

prog_logger = TranslatorLogger()


FAKE_VALUE_NAME = "FakeValue"


def get_value_name(value):
    if is_fake_value(value):
        return FAKE_VALUE_NAME
    return value.name


def apply_general_passes(
    program, *, enable_cse=True, enable_delete_assert_op=True
):
    pm = paddle.pir.PassManager(2)
    if enable_cse:
        pm.add_pass("common_subexpression_elimination_pass", {})
    if enable_delete_assert_op:
        pm.add_pass("delete_assert_op_pass", {})
    pm.run(program)


class NestSequence:
    """
    A wrapper class that easily to flatten and restore the nest structure of
    given sequence. It also remove the duplicate variables in the sequence.
    For example:
    >>> t = [v1, v2, v1]
    >>> m = tolist(t)
    [v1, v2]
    >>> m.restore([t1, t2])
    [t1, t2, t1]
    """

    def __init__(self, raw_input):
        self._raw_input = raw_input
        self._var_map, self._var_list = self._tolist()

    @property
    def var_list(self):
        return self._var_list

    def _tolist(self):
        """
        Flattens the nested sequences into single list and remove duplicate variables + non-variable elements.
        """
        variable_map = ValueDict()  # value -> list idx
        variable_list = []
        for value in paddle.utils.flatten(self._raw_input):
            if not isinstance(value, Value):
                continue
            if value in variable_map:
                # remove duplicate values.
                continue
            variable_map[value] = len(variable_list)
            variable_list.append(value)
        return variable_map, variable_list

    def restore(self, tensor_result_list):
        """
        Restores the nested sequence from tensor list.
        """
        assert len(self._var_list) == len(tensor_result_list)

        def to_tensor_result(x):
            if isinstance(x, Value):
                return tensor_result_list[self._var_map[x]]
            return x

        return paddle.utils.pack_sequence_as(
            self._raw_input,
            list(map(to_tensor_result, paddle.utils.flatten(self._raw_input))),
        )

    @cached_property
    def quick_index_map(self):
        raw_inputs = self._raw_input
        if len(raw_inputs) == 1:
            raw_inputs = raw_inputs[0]
        assert all(isinstance(v, Value) for v in raw_inputs)
        return [self._var_map[v] for v in raw_inputs]

    def quick_restore(self, tensor_list):
        return [tensor_list[idx] for idx in self.quick_index_map]

    def __getitem__(self, item):
        return self._var_list[item]


class RunnableProgram:
    """a pir program ready for run_program_op to run. constructed by 3 parts:
    - pir program (pir::Program)
    - in_out_values
        - input_x values ([string | pir::Value])
        - input_param values ([string | pir::Value])
        - output values ([string | pir::Value])
    - forward_backward_ranges
        - forward_range (tuple(Int, Int)) | None
        - backward_range (tuple(Int, Int)) | None
    """

    @staticmethod
    def _get_program_all_values(program):
        all_values = []

        def extend_values(block):
            all_values.extend(block.kwargs().values())
            for op in block.ops:
                all_values.extend(op.results())
                for block in op.blocks():
                    extend_values(block)

        extend_values(program.global_block())
        return all_values

    @staticmethod
    def _get_name_value_map_from_program(program) -> dict[str, Value]:
        name_to_value_dict: dict[str, Value] = {FAKE_VALUE_NAME: fake_value()}
        for value in RunnableProgram._get_program_all_values(program):
            for name in value._names:
                name_to_value_dict[name] = value
        return name_to_value_dict

    @cached_property
    def name_value_map(self):
        return RunnableProgram._get_name_value_map_from_program(self.program)

    def convert_name(self, values):
        if len(values) == 0:
            return []
        if isinstance(values[0], str):
            return values
        return [get_value_name(v) for v in values]

    @cached_property
    def x_values(self):
        return [self.name_value_map[v] for v in self.x_names]

    @cached_property
    def param_values(self):
        return [self.name_value_map[v] for v in self.param_names]

    @cached_property
    def out_values(self):
        return [self.name_value_map[v] for v in self.out_names]

    @cached_property
    def x_grad_values(self):
        return [self.name_value_map[v] for v in self.x_grad_names]

    @cached_property
    def param_grad_values(self):
        return [self.name_value_map[v] for v in self.p_grad_names]

    @cached_property
    def out_grad_values(self):
        return [self.name_value_map[v] for v in self.o_grad_names]

    def __init__(
        self,
        program,
        in_out_values,
        grad_in_out_values=None,
        forward_range=None,
        backward_range=None,
    ):
        assert isinstance(
            in_out_values, tuple
        ), "in_out_values must be tuple with len == 3"
        assert (
            len(in_out_values) == 3
        ), "in_out_values must be tuple with len == 3"
        assert isinstance(
            in_out_values[0], list
        ), "in_out_values must be tuple with len == 3"
        self.program = program
        self.x_names = self.convert_name(in_out_values[0])
        self.param_names = self.convert_name(in_out_values[1])
        self.out_names = self.convert_name(in_out_values[2])
        self.forward_range = forward_range
        self.backward_range = backward_range
        self.has_splited = False
        self.finish_pass = False
        if self.forward_range is None:
            self.forward_range = (0, len(self.program.global_block().ops))
        if self.backward_range is None:
            self.backward_range = (
                len(self.program.global_block().ops),
                len(self.program.global_block().ops),
            )
        if grad_in_out_values is None:
            grad_in_out_values = [], [], []
        self.x_grad_names = self.convert_name(grad_in_out_values[0])
        self.p_grad_names = self.convert_name(grad_in_out_values[1])
        self.o_grad_names = self.convert_name(grad_in_out_values[2])

        # Flag operator, indicating the operator between the forward subgraph and the backward subgraph. After self.program is updated by the pass, it is recommended to use the self.update_op_range interface to update the forward_range and backward_range.
        self.fwd_end_next_op = (
            self.program.global_block().ops[self.forward_range[1]]
            if self.forward_range[1] < len(self.program.global_block().ops)
            else None
        )
        self.bwd_start_pre_op = (
            self.program.global_block().ops[self.backward_range[0] - 1]
            if (
                self.backward_range[0] > 0
                and self.backward_range[0] - 1
                < len(self.program.global_block().ops)
            )
            else None
        )
        self.bwd_end_nex_op = (
            self.program.global_block().ops[self.backward_range[1]]
            if self.backward_range[1] < len(self.program.global_block().ops)
            else None
        )

    def update_op_range(self):
        if self.fwd_end_next_op is None or self.bwd_start_pre_op is None:
            self.forward_range = (0, len(self.program.global_block().ops))
            self.backward_range = (
                len(self.program.global_block().ops),
                len(self.program.global_block().ops),
            )
        else:
            fwd_start = self.forward_range[0]
            fwd_end = self.forward_range[1]
            bwd_start = self.backward_range[0]
            bwd_end = self.backward_range[1]
            for idx, op in enumerate(self.program.global_block().ops):
                if op == self.fwd_end_next_op:
                    fwd_end = idx
                if op == self.bwd_start_pre_op:
                    bwd_start = idx + 1
                if op == self.bwd_end_nex_op:
                    bwd_end = idx

            if self.bwd_end_nex_op is None:
                bwd_end = len(self.program.global_block().ops)

            self.forward_range = (fwd_start, fwd_end)
            self.backward_range = (bwd_start, bwd_end)

    def clone(self):
        cloned_program, _ = paddle.base.libpaddle.pir.clone_program(
            self.program
        )
        return RunnableProgram(
            cloned_program,
            (self.x_names, self.param_names, self.out_names),
            None,
            self.forward_range,
            self.backward_range,
        )

    def split_forward_backward(self):
        assert (
            self.has_splited is False
        ), "Please ensure only split once! don't call split_forward_backward manually."
        self.has_splited = True
        self.update_op_range()
        [
            fwd_prog,
            bwd_prog,
        ], prog_attr = paddle.base.libpaddle.pir.split_program(
            self.program,
            self.x_values,
            self.param_values,
            self.out_values,
            self.x_grad_values,
            self.param_grad_values,
            self.out_grad_values,
            self.forward_range,
            self.backward_range,
        )
        return [fwd_prog, bwd_prog], prog_attr

    def apply_pir_program_pass(self, pass_fn, sym_program=None):
        """
        Main entries for pass function, without considering any input/output and forward segmentation.
        pass_fn' signature is:

        1. This function will change forward and backward program.
        2. call self.program_attr means start to run.
        so we can't call this function after program_attr is called.

        def pass_fn(forward_program, backward_program):
            return forward_program, backward_program
        """
        origin_fwd = self.forward_program
        origin_bwd = self.backward_program

        prog_logger.log(
            1,
            f"******** [JIT] PIR forward program before PIR PASS ********\n{origin_fwd} ",
        )
        prog_logger.log(
            1,
            f"******** [JIT] PIR backward program before PIR PASS ********\n{origin_bwd} ",
        )
        # NOTE(dev): Add this line to trigger program_name_attr logic
        program_name_attr = self.program_name_attr
        self.forward_program, self.backward_program = pass_fn(
            origin_fwd, origin_bwd, program_name_attr, sym_program
        )
        prog_logger.log(
            1,
            f"******** [JIT] PIR forward program after PIR PASS ********\n{origin_fwd} ",
        )
        prog_logger.log(
            1,
            f"******** [JIT] PIR backward program after PIR PASS ********\n{origin_bwd} ",
        )

    def is_distributed_program(self):
        for op in self.program.global_block().ops:
            if op.dist_attr is not None:
                return True
        return False

    def apply_dist_pass_for_origin_program(self):
        if self.is_distributed_program():
            paddle.distributed.auto_parallel.static.mix_to_dist_pass.apply_mix2dist_pass(
                self.program
            )

    def apply_dist_pass_for_whole_program(self):
        if self.is_distributed_program():
            paddle.distributed.auto_parallel.static.mix_to_dist_pass.apply_mix2dist_pass(
                self.program
            )
            paddle.distributed.auto_parallel.static.pir_pass.apply_partition_pass(
                self.program
            )
            paddle.distributed.auto_parallel.static.pir_pass.ReshardPasses.apply_reshard_pass(
                self.program
            )
            paddle.base.libpaddle.pir.apply_dist2dense_pass(self.program)
            paddle.distributed.auto_parallel.static.pir_pass.remove_unuseful_comm_op_pass(
                self.program
            )

    # cached property can ensure program is splited only once.
    @cached_property
    def _forward_backward_program(self):
        return self.split_forward_backward()

    @cached_property  # shouldn't changed when call this once.
    def program_attr(self):
        assert (
            self.finish_pass is False
        ), "program_attr() is called by PartialProgramLayer, don't call it manually, use program_name_attr instead."
        # can't apply pass after call this function.
        self.finish_pass = True
        fwd_map = RunnableProgram._get_name_value_map_from_program(
            self.forward_program
        )
        bwd_map = RunnableProgram._get_name_value_map_from_program(
            self.backward_program
        )
        value_program_attr = {}
        for k, ns in self.program_name_attr.items():
            if k.startswith("f"):
                values = [fwd_map.get(n, fake_value()) for n in ns]
            elif k.startswith("b"):
                values = [bwd_map.get(n, fake_value()) for n in ns]
            elif k == "no_need_buffers":
                values = [fwd_map.get(n, fake_value()) for n in ns]
            else:
                raise ValueError(f"Unknown program attr: {k}")
            value_program_attr[k] = values

        rename_mapping = {}
        rename_mapping = RunnableProgram.unify_value_names(
            self.forward_program, rename_mapping
        )
        rename_mapping = RunnableProgram.unify_value_names(
            self.backward_program, rename_mapping
        )
        return value_program_attr

    @staticmethod
    def unify_value_names(
        program, rename_mapping: dict[str, str]
    ) -> dict[str, str]:
        """Ensure every value at most has one name in the program."""
        rename_mapping = dict(rename_mapping)
        for value in RunnableProgram._get_program_all_values(program):
            if not value.has_name:
                continue
            new_name = value.name  # get first name
            new_name = rename_mapping.get(new_name, new_name)
            rename_mapping.update(
                value._rename(new_name, program.global_block())
            )
        # Get all values again because some values has been erased.
        for value in RunnableProgram._get_program_all_values(program):
            if value.has_name:
                assert (
                    value._has_only_one_name()
                ), f"Expected all values in Program have only one name, but {value} has multiple names: {value._names}"
        return rename_mapping

    @cached_property
    def program_name_attr(self):
        origin_attr = self._forward_backward_program[1]
        _program_attr = {}
        for k, vs in origin_attr.items():
            _program_attr[k] = [get_value_name(v) for v in vs]
        return _program_attr

    @cached_property
    def forward_program(self):
        return self._forward_backward_program[0][0]

    @cached_property
    def backward_program(self):
        return self._forward_backward_program[0][1]


class PartialProgramLayerHook:
    def before_append_backward(self, forward_program, src_vars):
        return forward_program, src_vars

    def after_append_backward(
        self,
        whole_program,
        inputs,
        src_vars,
        grad_outputs,
        forward_end_idx,
        backward_start_idx,
    ):
        return whole_program, forward_end_idx, src_vars

    def after_infer(self, infer_program):
        return infer_program


class OperatorIndexPreservePass:
    OP_NAME_PREFIX = "preserved_index_"
    counter = 0

    def __init__(self, index, pass_fn):
        self.name = f"{OperatorIndexPreservePass.OP_NAME_PREFIX}{OperatorIndexPreservePass.counter}"
        OperatorIndexPreservePass.counter += 1
        self.pass_fn = pass_fn
        self.index = index

    def __call__(self, program):
        if len(program.global_block().ops) == 0:
            assert self.index == 0
            return self.pass_fn(program)
        paddle.base.libpaddle.pir.append_shadow_output(
            program,
            program.global_block().ops[0].result(0),
            self.name,
            self.index,
        )
        program = self.pass_fn(program)
        new_index = 0
        for op in program.global_block().ops:
            if (
                op.name() == "builtin.shadow_output"
                and self.name in op.attrs()["output_name"]
            ):
                break
            new_index += 1
        # remove forward_backward_separator
        if new_index >= len(program.global_block().ops):
            raise RuntimeError(
                f"Can't find index preserve label {self.name}, don't remove it in pass."
            )
        program.global_block().remove_op(program.global_block().ops[new_index])
        self.index = new_index
        return program


class IndicesPreservePass:
    def __init__(self, indices, pass_fn):
        self.pass_fn = pass_fn
        self.indices = indices
        self.new_indices = None

    def __call__(self, program):
        passes = [self.pass_fn]
        for idx, index in enumerate(self.indices):
            passes.append(OperatorIndexPreservePass(index, passes[idx]))
        new_program = passes[-1](program)

        self.new_indices = [p.index for p in passes[1:]]
        return new_program


class ValuePreservePass:
    OP_NAME_PREFIX = "preserved_value_"

    def __init__(self, values, use_cinn_pass):
        self.values = values
        self.use_cinn_pass = use_cinn_pass
        self.program = None

    def apply(self, program):
        raise RuntimeError("Not implemented.")

    @staticmethod
    def create_name_generator(prefix):
        count = 0

        def name_gen():
            nonlocal count
            name = f"{prefix}{count}"
            count += 1
            return name

        return name_gen

    @staticmethod
    def attach_preserved_name(value, program, value2name, name_generator):
        if is_fake_value(value):
            return None
        if value in value2name:
            return value2name[value]
        name = name_generator()
        value2name[value] = name
        paddle.base.libpaddle.pir.append_shadow_output(
            program,
            value,
            name,
            len(program.global_block().ops),
        )
        return name

    def __call__(self, program):
        # create preserved op for args
        value2name = ValueDict()
        name_generator = ValuePreservePass.create_name_generator(
            ValuePreservePass.OP_NAME_PREFIX
        )
        names = paddle.utils.map_structure(
            lambda value: ValuePreservePass.attach_preserved_name(
                value, program, value2name, name_generator  # noqa: F821
            ),
            self.values,
        )
        # NOTE(SigureMo): Value maybe removed in pass, don't use value2name after pass
        del value2name

        # apply program pass
        program = self.apply(program)

        # collect new value
        name2new_value = {}
        to_remove_op = []
        for op in program.global_block().ops:
            if op.name() == "builtin.shadow_output":
                if op.attrs()["output_name"].startswith(
                    ValuePreservePass.OP_NAME_PREFIX
                ):
                    name2new_value[op.attrs()["output_name"]] = op.operand(
                        0
                    ).source()
                    to_remove_op.append(op)

        # remove old op
        for op in to_remove_op:
            program.global_block().remove_op(op)

        self.values = paddle.utils.map_structure(
            lambda name: name2new_value.get(name, fake_value()), names
        )
        return program


class FullGraphPreProcessPass(ValuePreservePass):
    def apply(self, program):
        program = paddle.base.libpaddle.pir.apply_bn_add_act_pass(program)
        if self.use_cinn_pass:
            # NOTE(gongshaotian): execute infer_symbolic_shape_pass before reduce_as_sum_pass
            pm = paddle.base.libpaddle.pir.PassManager()
            pm.add_pass("delete_assert_op_pass", {})
            paddle.base.libpaddle.pir.infer_symbolic_shape_pass(pm, program)
            pm.run(program)
            self.program = program
            paddle.base.libpaddle.pir.reduce_as_sum_pass(pm, program)
            pm.run(program)
        return program


class PartialProgramLayer:
    """
    PartialProgramLayer wraps all the ops from layers decorated by `@to_static`
    and execute them as a static subgraph.

    .. note::
        **1. This is a very low level API. Users should not use this API
             directly. Please use `partial_program_from(concrete_program)`
             to create it.
        **2. TensorArray is not currently supported in the output.

    Args:
        main_program(Program): The main program that contains ops need to be executed.
        inputs(list[Variable]): The input list of the decorated function by `@to_static`.
        outputs(list[Variable]): The output list of the decorated function by `@to_static`.
        parameters(list[Tensor]|None): All trainable parameters included in the program. Default None.

    Returns:
        Layer: A Layer object that run all ops internally in static graph mode.
    """

    def __init__(
        self, main_program, inputs, outputs, parameters=None, **kwargs
    ):
        super().__init__()
        self._inputs = NestSequence(inputs)
        self._outputs = NestSequence(outputs)
        self._params, self._param_values = (
            parameters if parameters is not None else ([], [])
        )

        self._build_strategy = kwargs.get('build_strategy', BuildStrategy())
        assert isinstance(self._build_strategy, BuildStrategy)
        self._origin_main_program = self._verify_program(
            main_program, self._outputs
        )
        if parameters is not None:
            parameters[0][:] = self._params
            parameters[1][:] = self._param_values
        with paddle.base.framework._dygraph_guard(paddle.base.dygraph.Tracer()):
            self._cuda_graph_vec = self._create_cuda_graph_vec()
        self._cuda_graph_capture_mode = ""
        self._cuda_graph_pool_id = 0
        # Set default mode to train
        self.training = True
        self._program_extra_info = {}

        amp_dtype, custom_white_list, custom_black_list = None, None, None
        tracer = framework._dygraph_tracer()
        if tracer:
            custom_white_list, custom_black_list = tracer._get_amp_op_list()
            amp_dtype = tracer._amp_dtype
        if amp_dtype is not None and amp_dtype in ['float16', 'bfloat16']:
            # For AMP training
            self._amp_list = (
                paddle.static.amp.fp16_lists.AutoMixedPrecisionLists(
                    custom_white_list=custom_white_list,
                    custom_black_list=custom_black_list,
                    dtype=amp_dtype,
                )
            )

        # program_id -> list(scope)
        self._scope_cache = {}
        self._hookers = []
        self._backend = kwargs.get('backend', None)
        self._grad_var_names = {}
        self.full_graph_program_with_symbol_shape = None

    def __call__(self, inputs):
        """
        Execute static graph by Interpreter and Return dynamic Tensors.
        """
        in_vars = self._prepare_inputs(inputs)
        out_vars = self._prepare_outputs()
        attrs = self._prepare_attributes(in_sot_mode=False)
        _legacy_C_ops.pir_run_program(
            self._valid_vars(in_vars),
            self._valid_vars(self._params),
            self._valid_vars(out_vars),
            self._create_scope_vec(
                program_id=self.program_id, use_scope_cache=True
            ),
            self._cuda_graph_vec,
            *attrs,
        )
        restored_nest_out = self._restore_out(out_vars)
        return self._remove_no_value(restored_nest_out)

    def sot_call(self, inputs):
        """
        In sot, inputs and outputs of partial program only contain tensors, so we can skip some step to speed up
        """
        out_vars = self._prepare_outputs()
        attrs = self._prepare_attributes(in_sot_mode=True)
        _legacy_C_ops.pir_run_program(
            self._valid_vars(inputs),
            self._valid_vars(self._params),
            self._valid_vars(out_vars),
            self._create_scope_vec(
                program_id=self.program_id, use_scope_cache=True
            ),
            self._cuda_graph_vec,
            *attrs,
        )
        return self._outputs.quick_restore(out_vars)

    @cached_property
    def origin_runnable_program(self) -> RunnableProgram:
        inputs = list(self._inputs.var_list)
        outputs = list(self._outputs.var_list)
        params = self._param_values
        paddle.base.libpaddle.pir.append_shadow_outputs(
            self._origin_main_program,
            outputs,
            len(self._origin_main_program.global_block().ops),
            "output_",
        )
        return RunnableProgram(
            self._origin_main_program, (inputs, params, outputs)
        )

    def add_hooker(self, hooker):
        self._hookers.append(hooker)

    def _get_scope(self, program_id=None, use_scope_cache=False):
        if not use_scope_cache:
            return core.Scope()
        if program_id not in self._scope_cache:
            self._scope_cache[program_id] = []
        cached_scopes = self._scope_cache[program_id]
        for scope in cached_scopes:
            if scope._can_reused:
                return scope
        scope = core.Scope()
        cached_scopes.append(scope)
        return scope

    # whole
    @switch_to_static_graph
    def _create_program(self, is_infer_mode=False) -> RunnableProgram:
        if is_infer_mode:

            def pass_fn(
                forward_program,
                backward_program,
                program_name_attr,
                whole_program,
            ):
                # common pass
                pm = paddle.base.libpaddle.pir.PassManager()
                paddle.base.libpaddle.pir.infer_symbolic_shape_pass(
                    pm, forward_program
                )
                pm.run(forward_program)

                apply_general_passes(
                    forward_program,
                    enable_cse=cse_is_enabled(),
                    enable_delete_assert_op=cinn_is_enabled(
                        self._build_strategy, self._backend
                    ),
                )

                # if-else pass
                if cinn_is_enabled(self._build_strategy, self._backend):
                    paddle.base.libpaddle.pir.apply_cinn_pass(forward_program)
                else:
                    paddle.base.libpaddle.pir.check_infer_symbolic_if_need(
                        forward_program
                    )

                return forward_program, backward_program

            # TODO(xiongkun) who to transfer the pruning program?
            infer_program = self.origin_runnable_program.clone()
            if auto_layout_is_enabled():
                pm = paddle.pir.PassManager(2)
                pm.add_pass("auto_layout_pass", {})
                pm.run(infer_program.program)
            for hooker in self._hookers:
                hooker.after_infer(infer_program)
            infer_program.apply_pir_program_pass(pass_fn)
            return infer_program
        else:
            train_program: RunnableProgram = (
                self.origin_runnable_program.clone()
            )
            train_program.apply_dist_pass_for_origin_program()

            # Author(liujinnan): auto_layout_pass should be applied to the original_program, before append backward. So we put it here.
            if auto_layout_is_enabled():
                pm = paddle.pir.PassManager(2)
                pm.add_pass("auto_layout_pass", {})
                pm.run(train_program.program)
            train_program = self._append_backward_desc(train_program)
            # Note: Only set grad type once after initializing train program. So we put it here.
            self._set_grad_type(self._params, train_program)

            def pass_fn(
                forward_program,
                backward_program,
                program_name_attr,
                whole_program,
            ):
                def get_shape_analysis(program):
                    return paddle.base.libpaddle.pir.get_shape_constraint_ir_analysis(
                        program
                    )

                def set_backward_program_shape_analysis(
                    forward_program, backward_program
                ):
                    forward_shape_analysis = get_shape_analysis(forward_program)
                    backward_shape_analysis = get_shape_analysis(
                        backward_program
                    )

                    forward_name_value_map = {
                        name: item
                        for item in forward_program.list_vars()
                        for name in item._names
                    }

                    def share_symbol_shape_from_forward_to_backward(
                        forward_value, backward_value
                    ):
                        backward_shape_analysis.set_shape_or_data_for_var(
                            backward_value,
                            forward_shape_analysis.get_shape_or_data_for_var(
                                forward_value
                            ),
                        )

                    def get_kwargs_forward_matched_value(kw_name, kw_value):
                        if kw_name in program_name_attr['bo_g']:
                            idx = program_name_attr['bo_g'].index(kw_name)
                            return forward_name_value_map[
                                program_name_attr['fo'][idx]
                            ]
                        elif kw_name in forward_name_value_map:
                            return forward_name_value_map[kw_name]
                        else:
                            raise Exception(f"kw_args: {kw_name} not found")

                    for [kw_name, kw_value] in (
                        backward_program.global_block().kwargs().items()
                    ):
                        forward_matched_value = (
                            get_kwargs_forward_matched_value(kw_name, kw_value)
                        )
                        share_symbol_shape_from_forward_to_backward(
                            forward_matched_value, kw_value
                        )

                def init_forward_and_backward_program_shape_analysis(
                    forward_program, backward_program, whole_analysis
                ):
                    forward_shape_analysis = get_shape_analysis(forward_program)
                    backward_shape_analysis = get_shape_analysis(
                        backward_program
                    )
                    forward_shape_analysis.register_symbol_cstr_from_shape_analysis(
                        whole_analysis
                    )
                    backward_shape_analysis.register_symbol_cstr_from_shape_analysis(
                        whole_analysis
                    )

                def process_program_vars(
                    program, whole_analysis, is_target_value=None
                ):
                    program_analysis = get_shape_analysis(program)
                    missing_vars = []
                    for var in program.list_vars():
                        if whole_analysis.has_shape_or_data_for_var(var):
                            program_analysis.set_shape_or_data_for_var(
                                var,
                                whole_analysis.get_shape_or_data_for_var(var),
                            )
                        elif is_fake_value(var) or (
                            is_target_value and is_target_value(var)
                        ):
                            continue
                        else:
                            missing_vars.append(var)

                    for var in missing_vars:
                        program_analysis.get_shape_or_data_for_var(var)

                apply_general_passes(
                    forward_program,
                    enable_cse=cse_is_enabled(),
                    enable_delete_assert_op=cinn_is_enabled(
                        self._build_strategy, self._backend
                    ),
                )
                apply_general_passes(
                    backward_program,
                    enable_cse=cse_is_enabled(),
                    enable_delete_assert_op=cinn_is_enabled(
                        self._build_strategy, self._backend
                    ),
                )
                if cinn_is_enabled(self._build_strategy, self._backend):
                    kw_values = (
                        backward_program.global_block().kwargs().values()
                    )
                    is_kw_value = lambda v: any(
                        v.is_same(kw) for kw in kw_values
                    )

                    whole_analysis = get_shape_analysis(whole_program)

                    init_forward_and_backward_program_shape_analysis(
                        forward_program, backward_program, whole_analysis
                    )
                    process_program_vars(
                        forward_program,
                        whole_analysis,
                        is_target_value=lambda _: False,
                    )
                    set_backward_program_shape_analysis(
                        forward_program, backward_program
                    )
                    process_program_vars(
                        backward_program,
                        whole_analysis,
                        is_target_value=is_kw_value,
                    )
                    paddle.base.libpaddle.pir.apply_cinn_pass(
                        forward_program, False
                    )
                    paddle.base.libpaddle.pir.apply_cinn_pass(
                        backward_program, False
                    )
                else:
                    paddle.base.libpaddle.pir.check_infer_symbolic_if_need(
                        forward_program
                    )
                return forward_program, backward_program

            train_program.apply_pir_program_pass(
                pass_fn, self.full_graph_program_with_symbol_shape
            )
            return train_program

    @cached_property
    def _train_program_id(self):
        program_id = paddle.utils._hash_with_id(self.train_program, self)
        return program_id

    @cached_property
    def _infer_program_id(self):
        return paddle.utils._hash_with_id(self.infer_program, self)

    @property
    def program(self) -> RunnableProgram:
        """
        Return current train or eval program.
        """
        if self.training:
            return self.train_program
        else:
            return self.infer_program

    @property
    def program_id(self):
        """
        Return current train or eval program hash id.
        """
        if self.training:
            return self._train_program_id
        else:
            return self._infer_program_id

    @cached_property
    def train_program(self) -> RunnableProgram:
        return self._create_program()

    @cached_property
    def infer_program(self) -> RunnableProgram:
        return self._create_program(is_infer_mode=True)

    def _verify_program(self, main_program, outputs):
        """
        Verify that the program parameter is initialized, prune some unused params,
        and remove redundant op callstack.
        """
        # Check all params from main program can be found in self._params
        self._check_params_all_inited(main_program)

        return main_program

    @switch_to_static_graph
    def _append_backward_desc(
        self, train_runnable_program: RunnableProgram
    ) -> RunnableProgram:
        program = train_runnable_program.program
        targets = train_runnable_program.out_values
        # TODO(@zhuoge): refine the interface, use runnable_program to apply passes.
        for hooker in self._hookers:
            program, targets = hooker.before_append_backward(program, targets)
        inputs = train_runnable_program.x_values
        params = train_runnable_program.param_values
        combined_inputs = list(itertools.chain(inputs, params))
        forward_end_idx = len(program.global_block().ops)
        forward_end_op = None
        if forward_end_idx > 0:
            forward_end_op = program.global_block().ops[-1]
        grad_info_map = [None] * len(combined_inputs)
        with backend_guard(self._backend):
            check_type(
                targets,
                'targets',
                (Value, list, tuple),
                'paddle.static.gradients',
            )
            with ir_static.program_guard(program, None):
                # create outputs_grad for backward to avoid full and full_like op.
                forward_outputs_grads = []
                for out_value in targets:
                    if out_value.stop_gradient is True:
                        forward_outputs_grads.append(fake_value())
                    else:
                        value = paddle.full_like(
                            out_value,
                            fill_value=1.0,
                            dtype=out_value.dtype,
                        )
                        forward_outputs_grads.append(value)
                paddle.base.libpaddle.pir.append_shadow_outputs(
                    program,
                    forward_outputs_grads,
                    len(program.global_block().ops),
                    "grad_input_",
                )
                op_between_forward_and_backward = (
                    len(program.global_block().ops) - forward_end_idx
                )

                # call grad to get backward ops.
                if (
                    len(
                        list(
                            filter(lambda x: x.stop_gradient is False, targets)
                        )
                    )
                    > 0
                ):
                    grad_info_map = grad(
                        inputs=combined_inputs,
                        outputs=list(
                            filter(lambda x: x.stop_gradient is False, targets)
                        ),
                        grad_outputs=list(
                            filter(
                                lambda x: not is_fake_value(x),
                                forward_outputs_grads,
                            )
                        ),
                    )
                    if forward_end_op is not None:
                        for idx, op in enumerate(program.global_block().ops):
                            if op == forward_end_op:
                                forward_end_idx = idx + 1
                                break

            for hooker in self._hookers:
                (
                    program,
                    forward_end_idx,
                    targets,
                ) = hooker.after_append_backward(
                    program,
                    combined_inputs,
                    targets,
                    forward_outputs_grads,
                    forward_end_idx,
                    forward_end_idx + op_between_forward_and_backward,
                )

        mapping_value = lambda x: x if isinstance(x, Value) else fake_value()
        inputs_size = len(inputs)
        x_grad_value = list(map(mapping_value, grad_info_map[0:inputs_size]))
        p_grad_value = list(map(mapping_value, grad_info_map[inputs_size:]))
        o_grad_value = list(map(mapping_value, forward_outputs_grads))

        # insert grads name for RunnableProgram (we need name for grad_inputs and grad_outputs)
        input_grads_to_append = list(
            filter(lambda x: not is_fake_value(x), o_grad_value)
        )
        output_grads_to_append = list(
            filter(lambda x: not is_fake_value(x), x_grad_value + p_grad_value)
        )
        backward_end_op_index = len(program.global_block().ops)
        paddle.base.libpaddle.pir.append_shadow_outputs(
            program,
            output_grads_to_append,
            backward_end_op_index,
            "grad_output_",
        )

        backward_start_op_index = (
            forward_end_idx + op_between_forward_and_backward
        )

        # construct a runnable program.
        fused_bn_add_act_pass = FullGraphPreProcessPass(
            [inputs, params, targets, x_grad_value, p_grad_value, o_grad_value],
            cinn_is_enabled(self._build_strategy, self._backend),
        )
        forward_index_pass = IndicesPreservePass(
            [forward_end_idx, backward_start_op_index, backward_end_op_index],
            fused_bn_add_act_pass,
        )

        program = forward_index_pass(program)
        self.full_graph_program_with_symbol_shape = (
            fused_bn_add_act_pass.program
        )
        (
            inputs,
            params,
            targets,
            x_grad_value,
            p_grad_value,
            o_grad_value,
        ) = fused_bn_add_act_pass.values
        (
            forward_end_idx,
            backward_start_op_index,
            backward_end_op_index,
        ) = forward_index_pass.new_indices

        whole_program = RunnableProgram(
            program,
            (inputs, params, targets),
            (x_grad_value, p_grad_value, o_grad_value),
            (0, forward_end_idx),
            (backward_start_op_index, backward_end_op_index),
        )
        whole_program.apply_dist_pass_for_whole_program()
        return whole_program

    def _prepare_attributes(self, in_sot_mode=False):
        attrs = [
            'forward_program',
            self.program.forward_program,
            'backward_program',
            self.program.backward_program,
            'is_test',
            not self.training,
            'program_id',
            self.program_id,
            'in_sot_mode',
            in_sot_mode,
        ]
        for key, val in self.program.program_attr.items():
            attrs.append(key)
            attrs.append(val)

        if self._cuda_graph_capture_mode:
            attrs.extend(
                (
                    'cuda_graph_capture_mode',
                    self._cuda_graph_capture_mode,
                    'cuda_graph_pool_id',
                    self._cuda_graph_pool_id,
                )
            )
        return attrs

    def _prepare_inputs(self, inputs):
        """
        Prepare inputs, outputs, attrs.
        """
        assert isinstance(inputs, (tuple, list))
        # Flatten inputs with nested structure into single list.
        flatten_inputs = paddle.utils.flatten(inputs)
        # Convert variable into Tensor and feed in training data.
        input_vars = []
        expected_place = framework._current_expected_place()
        for i, value in enumerate(flatten_inputs):
            if isinstance(value, np.ndarray):
                var = None
                var = core.eager.Tensor(
                    value=value,
                    persistable=False,
                    place=expected_place,
                    zero_copy=True,
                )
            elif isinstance(value, core.eager.Tensor):
                # NOTE(Aurelius84): If var is on CPUPlace, it will be transformed multi times
                # into CUDAPlace when it's as input of multi Ops. so we move it in advance
                # to avoid this problem.
                if value.stop_gradient and not value.place._equals(
                    expected_place
                ):
                    var = value._copy_to(expected_place, False)
                    var.stop_gradient = True
                else:
                    var = value
            else:
                continue
            input_vars.append(var)
        return input_vars

    def _prepare_outputs(self):
        return paddle.framework.core.create_empty_tensors_with_values(
            self._outputs.var_list
        )

    def _create_scope_vec(self, program_id=None, use_scope_cache=False):
        inner_scope = self._get_scope(
            program_id=program_id, use_scope_cache=use_scope_cache
        )
        return [inner_scope]

    def _create_cuda_graph_vec(self):
        var = core.eager.Tensor(
            core.VarDesc.VarType.FP32,
            [],
            "cuda_graph",
            core.VarDesc.VarType.RAW,
            True,
        )
        var.stop_gradient = True
        return var

    def _restore_out(self, out_vars):
        """
        Restores same nested outputs by only replacing the Variable with Tensor.
        """
        outs = self._outputs.restore(out_vars)
        if outs is not None and len(outs) == 1:
            outs = outs[0]
        return outs

    @switch_to_static_graph
    def _clone_for_test(self, main_program):
        return main_program.clone(for_test=True)

    def _is_no_value(self, var):
        if isinstance(var, core.eager.Tensor) and var.shape == [1]:
            # NOTE: .numpy() will insert MemcpySync operation, it hits performance.
            if var.numpy()[0] == RETURN_NO_VALUE_MAGIC_NUM:
                return True
        return False

    def _remove_no_value(self, out_vars):
        """
        Removes invalid value for various-length return statement
        """
        if isinstance(out_vars, core.eager.Tensor):
            if self._is_no_value(out_vars):
                return None
            return out_vars
        elif isinstance(out_vars, (tuple, list)):
            if isinstance(out_vars, tuple):
                res = tuple(
                    var for var in out_vars if not self._is_no_value(var)
                )
            else:
                res = [var for var in out_vars if not self._is_no_value(var)]

            has_removed = len(out_vars) > len(res)
            # len(out_vars) > len(res) means we have removed var. This is
            # preventing out_vars is empty or just one element at the beginning
            if len(res) == 0 and has_removed:
                return None
            elif len(res) == 1 and has_removed:
                return res[0]
            return res

        return out_vars

    def _set_grad_type(self, params, train_program: RunnableProgram):
        # NOTE: if user set sparse gradient mode, the param's gradient
        # will be SelectedRows, not DenseTensor. But tracer will just
        # set param grad Tensor by forward Tensor(DenseTensor)
        # If we don't change grad_var type here, RunProgramOp need
        # transform SelectedRows to DenseTensor forcibly, it may not
        # be user wanted result.
        forward_params_grads = train_program.param_grad_values
        train_program = train_program.program
        for param, value in zip(params, forward_params_grads):
            if is_fake_value(value):
                continue
            if value.is_selected_row_type():
                param._set_grad_type(
                    paddle.base.core.VarDesc.VarType.SELECTED_ROWS
                )
            elif value.is_dense_tensor_type():
                param._set_grad_type(
                    paddle.base.core.VarDesc.VarType.DENSE_TENSOR
                )
            else:
                raise NotImplementedError(
                    "only support selected_row and dense_tensor grad type."
                )

    def _check_params_all_inited(self, main_program):
        """
        Check all params from main program are already initialized, see details as follows:
            1. all parameters in self._params should be type `framework.EagerParamBase` which are created in dygraph.
            2. all parameters from transformed program can be found in self._params.
               Because they share same data with EagerParamBase of original dygraph.
        """
        if not isinstance(self._params, (list, tuple)):
            raise TypeError(
                f"Type of self._params in PartialProgramLayer should be list or tuple, but received {type(self._params)}."
            )

        param_and_buffer_names_set = set()
        for i, var in enumerate(self._params):
            # self._params contains parameters and buffers with persistable=True.
            if not isinstance(var, core.eager.Tensor):
                raise TypeError(
                    f'Type of self._params[{i}] in PartialProgramLayer should be Parameter or Variable, but received {type(var)}.'
                )
            param_and_buffer_names_set.add(var.name)

    def _valid_vars(self, vars):
        return vars if vars else None


def partial_program_from(
    concrete_program: ConcreteProgram, from_method: bool = False
) -> PartialProgramLayer:
    inputs = concrete_program.inputs

    # NOTE(SigureMo): Remove the first arg `self` from method args.
    if inputs and from_method:
        inputs = inputs[1:]

    return PartialProgramLayer(
        concrete_program.main_program,
        inputs,
        concrete_program.outputs,
        concrete_program.parameters,
        **concrete_program.kwargs,
    )
